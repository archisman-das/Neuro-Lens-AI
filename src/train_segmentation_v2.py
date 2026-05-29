"""v2 segmentation trainer: bigger, more general, more accurate.

Upgrades over src/train_segmentation_torch.py (v1):
  - Backbone:        random 24-filter Attention U-Net -> segmentation_models_pytorch
                     U-Net with ResNet34 ImageNet-pretrained encoder (~24M params).
  - Loss:            Dice + BCE (same) plus Focal Tversky for the LGG dataset's
                     class imbalance.
  - Augmentation:    flip + rot90  ->  albumentations pipeline (affine +
                     elastic deform + brightness/contrast + gamma + gaussian
                     noise + gaussian blur + grid distortion + horizontal flip).
                     Validation/test pipelines do only resize + normalize.
  - Normalization:   /255  ->  ImageNet per-channel mean/std (matches the
                     ResNet34 pretrained encoder).
  - Resolution:      192  ->  256
  - Precision:       FP32  ->  mixed precision (FP16) via torch.cuda.amp.
                     ~2x throughput on RTX 4060 + slightly lower power draw.
  - Optimizer:       Adam + ReduceLROnPlateau  ->  AdamW + cosine annealing
                     with linear warmup over the first 3 epochs.
  - Epochs:          25 / patience 8  ->  60 / patience 15
  - Multi-dataset:   train on LGG (real radiologist masks, FLAIR) AND
                     Kaggle (Otsu pseudo-masks, T1c) together. Forces the
                     model to learn a single representation across two
                     different MRI modalities, which is what 'generalize'
                     actually means in practice.
  - TTA:             inference-time augmentation (horizontal + vertical flip
                     averaging) via the existing torch path - hook is in
                     dashboard.py once we want it.
  - Crash-resilient: per-epoch last.pt + history.json + --resume support
                     (carried over from v1).

Output: segmentation_artifacts/attention_unet_v2/{best_model.pt, last.pt,
history.json, evaluation_metrics.json, training.log}

The saved checkpoint includes encoder_name + base architecture so the dashboard
can rebuild the same model when loading.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.utils.data import ConcatDataset, DataLoader, Dataset

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform(image_size: int) -> A.Compose:
    """Light augmentation pipeline. Tuned for medical MRI.

    Earlier (v3 first run) we used heavy aug: affine+shear+translate+elastic+
    grid distortion+noise+blur. Combined with batch=8 + no gradient clipping
    that caused training divergence at epoch 8 (train_loss jumped 0.227 -> 0.369
    over two epochs, val Dice dropped 0.74 -> 0.54).

    Removed: ElasticTransform, GridDistortion, MedianBlur. Softened: smaller
    affine ranges, no shear, no translate, lower per-aug probabilities.
    """
    return A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_AREA),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Affine(
            scale=(0.95, 1.05),
            rotate=(-10, 10),
            p=0.5,
        ),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
            A.RandomGamma(gamma_limit=(90, 110)),
        ], p=0.4),
        A.OneOf([
            A.GaussNoise(),
            A.GaussianBlur(blur_limit=3),
        ], p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_eval_transform(image_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class SegDatasetV2(Dataset):
    """Reads <split_dir>/images/*.png paired with <split_dir>/masks/*.png and
    applies an albumentations transform. Pairing is by filename stem."""

    def __init__(self, split_dir: Path, transform: A.Compose, name: str = ''):
        self.split_dir = Path(split_dir)
        self.transform = transform
        self.name = name or self.split_dir.parent.name
        images_dir = self.split_dir / 'images'
        masks_dir = self.split_dir / 'masks'
        if not images_dir.exists() or not masks_dir.exists():
            raise FileNotFoundError(f'Missing images/ or masks/ under {self.split_dir}')
        image_paths = sorted([*images_dir.glob('*.png'), *images_dir.glob('*.jpg'), *images_dir.glob('*.jpeg')])
        mask_lookup = {p.stem: p for p in masks_dir.glob('*.png')}
        self.pairs = [(ip, mask_lookup[ip.stem]) for ip in image_paths if ip.stem in mask_lookup]
        if not self.pairs:
            raise ValueError(f'No image/mask pairs found under {self.split_dir}')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        ip, mp = self.pairs[idx]
        img = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)
        out = self.transform(image=img, mask=mask)
        return out['image'], out['mask'].unsqueeze(0)


def dice_score(probs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    p = probs.contiguous().view(probs.size(0), -1)
    t = targets.contiguous().view(targets.size(0), -1)
    inter = (p * t).sum(dim=1)
    return ((2.0 * inter + smooth) / (p.sum(dim=1) + t.sum(dim=1) + smooth)).mean()


def iou_score(probs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    p = probs.contiguous().view(probs.size(0), -1)
    t = targets.contiguous().view(targets.size(0), -1)
    inter = (p * t).sum(dim=1)
    union = p.sum(dim=1) + t.sum(dim=1) - inter
    return ((inter + smooth) / (union + smooth)).mean()


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    return 1.0 - dice_score(torch.sigmoid(logits), targets, smooth)


def focal_tversky_loss(logits: torch.Tensor, targets: torch.Tensor,
                       alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75,
                       smooth: float = 1e-6) -> torch.Tensor:
    """Focal Tversky helps with severe foreground/background imbalance, which
    LGG has (most pixels are background, tumor area is tiny)."""
    probs = torch.sigmoid(logits)
    p = probs.contiguous().view(probs.size(0), -1)
    t = targets.contiguous().view(targets.size(0), -1)
    tp = (p * t).sum(dim=1)
    fn = ((1 - p) * t).sum(dim=1)
    fp = (p * (1 - t)).sum(dim=1)
    tversky = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return ((1 - tversky) ** gamma).mean()


def combined_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Dice + BCE only. Focal Tversky was producing NaNs under FP16 mixed
    precision (the (1 - tversky)**gamma term underflows when tversky -> 1
    on confidently-correct batches). Dropped at v3 run #3 after NaN at ep 8."""
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dl = dice_loss(logits, targets)
    return 0.5 * bce + 0.5 * dl


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device,
             threshold: float = 0.5, name: str = '') -> dict:
    model.eval()
    dice_sum = iou_sum = pix_sum = bce_sum = 0.0
    pos_true = pos_pred = inter = union = 0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                logits = model(x)
                probs = torch.sigmoid(logits)
            binp = (probs >= threshold).float()
            dice_sum += float(dice_score(binp, y))
            iou_sum += float(iou_score(binp, y))
            pix_sum += float((binp == y).float().mean())
            bce_sum += float(F.binary_cross_entropy_with_logits(logits.float(), y))
            pos_true += int(y.sum().item())
            pos_pred += int(binp.sum().item())
            inter += int((binp * y).sum().item())
            union += int(((binp + y) >= 1).float().sum().item())
            n += 1
    if n == 0:
        return {}
    out = {
        'split': name or 'eval',
        'dice': dice_sum / n,
        'iou': iou_sum / n,
        'pixel_accuracy': pix_sum / n,
        'bce_loss': bce_sum / n,
        'positive_voxels_true': pos_true,
        'positive_voxels_pred': pos_pred,
        'micro_dice': (2 * inter) / max(pos_true + pos_pred, 1),
        'micro_iou': inter / max(union, 1),
    }
    return out


def main():
    parser = argparse.ArgumentParser(description='V2 segmentation trainer (SMP UNet + ResNet34 + heavy aug + FP16).')
    parser.add_argument('--data_dirs', nargs='+', default=['dataset_lgg', 'dataset_real'],
                        help='One or more dataset roots, each with train/val/test/{images,masks}/. '
                             'Default trains on LGG (real masks) + Kaggle (pseudo-masks) for cross-modality.')
    parser.add_argument('--output_dir', default='segmentation_artifacts/attention_unet_v2')
    parser.add_argument('--encoder', default='resnet34',
                        help='SMP encoder backbone. Options include resnet34, resnet50, efficientnet-b0, mobilenet_v2.')
    parser.add_argument('--encoder_weights', default='imagenet')
    parser.add_argument('--architecture', default='Unet',
                        choices=['Unet', 'UnetPlusPlus', 'MAnet', 'Linknet', 'DeepLabV3Plus'])
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed-precision (default: enabled on CUDA).')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0,
                        help='Max grad L2 norm. Prevents the divergence seen at epoch 8 of v3 run #1.')
    parser.add_argument('--max_gpu_clock_mhz', type=int, default=1500,
                        help='Brownout guard: refuse to start if GPU max clock > this. '
                             'Set the cap in admin PS: nvidia-smi --lock-gpu-clocks=210,<limit>')
    parser.add_argument('--skip_gpu_cap_check', action='store_true',
                        help='Bypass the brownout-guard clock-cap check.')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('[warn] CUDA not available; falling back to CPU.', flush=True)
        args.device = 'cpu'
    device = torch.device(args.device)
    amp_enabled = (device.type == 'cuda') and (not args.no_amp)
    print(f'[info] device={device} amp={amp_enabled}'
          + (f' ({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else ''), flush=True)

    # Live brownout-monitor thread. Polls nvidia-smi every 30s and writes
    # clocks/temp/power to <output_dir>/gpu_telemetry.csv. If the lock cap
    # isn't actually in effect, this gives us the data after a crash (the
    # nvidia-smi --query for clocks.max.graphics returns the silicon ceiling
    # which is useless for verifying a runtime lock, so we observe behaviour
    # under load instead).
    if device.type == 'cuda':
        import csv
        import subprocess
        import threading
        telemetry_path = Path(args.output_dir) / 'gpu_telemetry.csv'
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        _stop = {'flag': False}

        def _telemetry_loop():
            with telemetry_path.open('a', newline='', encoding='utf-8') as fh:
                w = csv.writer(fh)
                if telemetry_path.stat().st_size == 0:
                    w.writerow(['timestamp', 'gpu_clock_mhz', 'mem_clock_mhz',
                                'temp_c', 'power_w', 'util_pct', 'mem_used_mb'])
                while not _stop['flag']:
                    try:
                        out = subprocess.check_output(
                            ['nvidia-smi',
                             '--query-gpu=clocks.gr,clocks.mem,temperature.gpu,power.draw,utilization.gpu,memory.used',
                             '--format=csv,noheader,nounits'],
                            stderr=subprocess.DEVNULL, timeout=5,
                        ).decode().strip()
                        parts = [p.strip() for p in out.split(',')]
                        w.writerow([time.strftime('%H:%M:%S')] + parts)
                        fh.flush()
                        if int(float(parts[0])) > args.max_gpu_clock_mhz:
                            print(f'[telemetry] WARN gpu_clock={parts[0]} MHz exceeds cap '
                                  f'{args.max_gpu_clock_mhz}; check nvidia-smi --lock-gpu-clocks.',
                                  flush=True)
                    except Exception:
                        pass
                    time.sleep(30)

        _t = threading.Thread(target=_telemetry_loop, daemon=True)
        _t.start()
        print(f'[info] GPU telemetry logging to {telemetry_path} every 30s', flush=True)

    train_tf = build_train_transform(args.image_size)
    eval_tf = build_eval_transform(args.image_size)

    train_datasets, val_datasets, test_datasets = [], [], []
    for d in args.data_dirs:
        d = Path(d)
        if not d.exists():
            print(f'[warn] data_dir not found, skipping: {d}', flush=True)
            continue
        for sub, target, tf in [('train', train_datasets, train_tf),
                                ('val', val_datasets, eval_tf),
                                ('test', test_datasets, eval_tf)]:
            split = d / sub
            try:
                ds = SegDatasetV2(split, tf, name=f'{d.name}/{sub}')
                target.append(ds)
                print(f'[info] {d.name}/{sub}: {len(ds)} samples', flush=True)
            except (FileNotFoundError, ValueError) as exc:
                print(f'[warn] skip {d.name}/{sub}: {exc}', flush=True)

    if not train_datasets or not val_datasets:
        raise RuntimeError('No usable train/val datasets after scanning --data_dirs.')

    train_ds = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    val_ds = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
    test_ds = ConcatDataset(test_datasets) if len(test_datasets) > 1 else (test_datasets[0] if test_datasets else None)
    print(f'[info] total train={len(train_ds)} val={len(val_ds)}'
          + (f' test={len(test_ds)}' if test_ds else ''), flush=True)

    common = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common) if test_ds else None

    SmpClass = getattr(smp, args.architecture)
    model = SmpClass(
        encoder_name=args.encoder,
        encoder_weights=args.encoder_weights if args.encoder_weights and args.encoder_weights != 'none' else None,
        in_channels=3,
        classes=1,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[info] model: {args.architecture} + {args.encoder} (pretrained={args.encoder_weights}) - {n_params:,} params', flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup_epochs), eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / 'best_model.pt'
    last_path = output_dir / 'last.pt'
    history_path = output_dir / 'history.json'

    history = {'train_loss': [], 'val_dice': [], 'val_iou': [], 'val_loss': [], 'lr': []}
    best_val_dice = -1.0
    epochs_without_improve = 0
    start_epoch = 0

    if args.resume and last_path.exists():
        prev = torch.load(str(last_path), map_location=device, weights_only=False)
        model.load_state_dict(prev['state_dict'])
        if 'optimizer_state' in prev:
            optimizer.load_state_dict(prev['optimizer_state'])
        if 'scheduler_state' in prev:
            try:
                scheduler.load_state_dict(prev['scheduler_state'])
            except Exception:
                pass
        if 'scaler_state' in prev and amp_enabled:
            try:
                scaler.load_state_dict(prev['scaler_state'])
            except Exception:
                pass
        history = prev.get('history', history)
        best_val_dice = float(prev.get('best_val_dice', best_val_dice))
        epochs_without_improve = int(prev.get('epochs_without_improve', 0))
        start_epoch = int(prev.get('epoch', 0))
        print(f'[info] Resumed from {last_path} at epoch {start_epoch} (best_val_dice={best_val_dice:.4f})', flush=True)

    base_lr = args.learning_rate
    for epoch in range(start_epoch, args.epochs):
        # Linear warmup over the first warmup_epochs, then cosine
        if epoch < args.warmup_epochs:
            warm_lr = base_lr * (epoch + 1) / max(1, args.warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = warm_lr

        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_steps = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp_enabled):
                logits = model(x)
                loss = combined_loss(logits, y)
            if amp_enabled:
                scaler.scale(loss).backward()
                # Unscale BEFORE clipping so the clip threshold is in real (FP32) units.
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
                # NaN guard: scaler.step will internally check infs but doubling
                # up here makes the skip explicit and avoids any chance of a
                # poisoned weight update reaching the model.
                if not torch.isfinite(grad_norm):
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    print(f'[nan_guard] step skipped (grad_norm not finite)', flush=True)
                else:
                    scaler.step(optimizer)
                    scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
                if not torch.isfinite(grad_norm):
                    optimizer.zero_grad(set_to_none=True)
                    print(f'[nan_guard] step skipped (grad_norm not finite)', flush=True)
                else:
                    optimizer.step()
            running_loss += float(loss)
            n_steps += 1

        if epoch >= args.warmup_epochs:
            scheduler.step()

        train_loss = running_loss / max(n_steps, 1)
        vm = evaluate(model, val_loader, device, name='val')
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_dice'].append(vm['dice'])
        history['val_iou'].append(vm['iou'])
        history['val_loss'].append(vm['bce_loss'])
        history['lr'].append(lr_now)
        print(
            f'[epoch {epoch+1:02d}/{args.epochs}] '
            f'train_loss={train_loss:.4f}  val_dice={vm["dice"]:.4f}  val_iou={vm["iou"]:.4f}  '
            f'val_bce={vm["bce_loss"]:.4f}  lr={lr_now:.2e}  ({elapsed:.1f}s)',
            flush=True,
        )

        if vm['dice'] > best_val_dice:
            best_val_dice = vm['dice']
            epochs_without_improve = 0
            torch.save({
                'state_dict': model.state_dict(),
                'config': vars(args),
                'val_metrics': vm,
                'epoch': epoch + 1,
                'architecture': args.architecture,
                'encoder': args.encoder,
                'image_size': args.image_size,
            }, best_path)
            print(f'        -> new best val_dice={best_val_dice:.4f}; weights saved to {best_path}', flush=True)
        else:
            epochs_without_improve += 1

        torch.save({
            'state_dict': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state': scaler.state_dict() if amp_enabled else None,
            'config': vars(args),
            'val_metrics': vm,
            'epoch': epoch + 1,
            'history': history,
            'best_val_dice': best_val_dice,
            'epochs_without_improve': epochs_without_improve,
            'architecture': args.architecture,
            'encoder': args.encoder,
            'image_size': args.image_size,
        }, last_path)
        with history_path.open('w', encoding='utf-8') as fh:
            json.dump(history, fh, indent=2)

        if epochs_without_improve >= args.patience:
            print(f'[info] Early stopping: no improvement in {args.patience} epochs.', flush=True)
            break

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
    eval_payload = {'val': evaluate(model, val_loader, device, name='val')}
    if test_loader is not None:
        eval_payload['test'] = evaluate(model, test_loader, device, name='test')
    # Also report per-source-dataset test performance so we can see whether
    # the model generalizes across modalities.
    if len(test_datasets) > 1:
        for ds in test_datasets:
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                pin_memory=(device.type == 'cuda'))
            eval_payload[f'test_{ds.name.replace("/", "_")}'] = evaluate(model, loader, device, name=ds.name)
    with (output_dir / 'evaluation_metrics.json').open('w', encoding='utf-8') as fh:
        json.dump(eval_payload, fh, indent=2)
    print('[info] Final evaluation:')
    print(json.dumps(eval_payload, indent=2))

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        epochs_x = list(range(1, len(history['train_loss']) + 1))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(epochs_x, history['train_loss'], label='train loss')
        axes[0].plot(epochs_x, history['val_loss'], label='val BCE')
        axes[0].legend(); axes[0].set_xlabel('epoch'); axes[0].set_title('Loss')
        axes[1].plot(epochs_x, history['val_dice'], label='val dice')
        axes[1].plot(epochs_x, history['val_iou'], label='val IoU')
        axes[1].legend(); axes[1].set_xlabel('epoch'); axes[1].set_title('Validation metrics')
        plt.tight_layout()
        plt.savefig(output_dir / 'training_curves.png', dpi=120)
        plt.close()
    except Exception as exc:
        print(f'[warn] plot failed: {exc}', flush=True)

    print(f'[done] Best val Dice = {best_val_dice:.4f}')


if __name__ == '__main__':
    main()
