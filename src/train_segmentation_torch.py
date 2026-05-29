"""Train Attention U-Net (PyTorch) on GPU for brain-tumor segmentation.

Why PyTorch: see the docstring of src/segmentation_torch.py. Short version:
TF 2.21 has no Windows-native GPU support; the user has an RTX 4060 with
PyTorch + CUDA 12.6 already working, so we train here on GPU.

Expected input layout (produced by generate_pseudo_masks.py):

    <data_dir>/train/images/*.png
    <data_dir>/train/masks/*.png       (0/255, paired by basename)
    <data_dir>/val/images/*.png
    <data_dir>/val/masks/*.png
    <data_dir>/test/images/*.png
    <data_dir>/test/masks/*.png

Outputs: segmentation_artifacts/attention_unet/{best_model.pt, history.json,
training_curves.png, evaluation_metrics.json}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
# tensorboard import removed - not needed and was forcing a dependency on the
# `tensorboard` package which isn't part of the PyTorch install.

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from src.segmentation_torch import (
    AttentionUNet,
    combined_dice_bce_loss,
    dice_coefficient,
    iou_metric,
)


class SegDataset(Dataset):
    def __init__(self, split_dir: Path, image_size: int, augment: bool = False):
        self.image_size = image_size
        self.augment = augment
        images_dir = Path(split_dir) / 'images'
        masks_dir = Path(split_dir) / 'masks'
        if not images_dir.exists() or not masks_dir.exists():
            raise FileNotFoundError(
                f'Missing images/ or masks/ under {split_dir}. '
                'Run `python generate_pseudo_masks.py` first.'
            )
        image_paths = sorted([*images_dir.glob('*.png'), *images_dir.glob('*.jpg'), *images_dir.glob('*.jpeg')])
        mask_lookup = {p.stem: p for p in masks_dir.glob('*.png')}
        self.pairs = []
        for ip in image_paths:
            if ip.stem in mask_lookup:
                self.pairs.append((ip, mask_lookup[ip.stem]))
        if not self.pairs:
            raise ValueError(f'No (image, mask) pairs found under {split_dir}.')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        ip, mp = self.pairs[idx]
        img = cv2.imread(str(ip))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size))
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if mask.shape[0] != self.image_size or mask.shape[1] != self.image_size:
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            # Light spatial augmentation: hflip + 90-deg rotation.
            if np.random.rand() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
                mask = np.ascontiguousarray(mask[:, ::-1])
            if np.random.rand() < 0.5:
                img = np.ascontiguousarray(img[::-1, :])
                mask = np.ascontiguousarray(mask[::-1, :])
            k = int(np.random.randint(0, 4))
            if k:
                img = np.ascontiguousarray(np.rot90(img, k=k))
                mask = np.ascontiguousarray(np.rot90(mask, k=k))

        img = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)
        mask = (mask.astype(np.float32) / 255.0 > 0.5).astype(np.float32)
        return (
            torch.from_numpy(img),
            torch.from_numpy(mask).unsqueeze(0),
        )


def _evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.5) -> dict:
    model.eval()
    dice_sum = 0.0
    iou_sum = 0.0
    pix_acc_sum = 0.0
    bce_sum = 0.0
    n_batches = 0
    pos_voxels = 0
    pred_pos_voxels = 0
    inter_voxels = 0
    union_voxels = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.sigmoid(logits)
            binp = (probs >= threshold).float()
            dice_sum += float(dice_coefficient(binp, y))
            iou_sum += float(iou_metric(binp, y))
            pix_acc_sum += float((binp == y).float().mean())
            bce_sum += float(F.binary_cross_entropy_with_logits(logits, y))
            pos_voxels += int(y.sum().item())
            pred_pos_voxels += int(binp.sum().item())
            inter_voxels += int((binp * y).sum().item())
            union_voxels += int(((binp + y) >= 1).float().sum().item())
            n_batches += 1
    if n_batches == 0:
        return {}
    return {
        'dice': dice_sum / n_batches,
        'iou': iou_sum / n_batches,
        'pixel_accuracy': pix_acc_sum / n_batches,
        'bce_loss': bce_sum / n_batches,
        'positive_voxels_true': pos_voxels,
        'positive_voxels_pred': pred_pos_voxels,
        'micro_dice': (2 * inter_voxels) / max(pos_voxels + pred_pos_voxels, 1),
        'micro_iou': inter_voxels / max(union_voxels, 1),
    }


def main():
    parser = argparse.ArgumentParser(description='Train Attention U-Net on GPU.')
    parser.add_argument('--data_dir', default='dataset_real')
    parser.add_argument('--output_dir', default='segmentation_artifacts/attention_unet')
    # Safer defaults after the May 29 Kernel-Power 41 crash: smaller image,
    # smaller batch, smaller base_filters -> ~3x less peak GPU power draw.
    # Override at the CLI if you want to push closer to the original settings.
    parser.add_argument('--image_size', type=int, default=192)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--base_filters', type=int, default=24)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--dice_weight', type=float, default=0.6)
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers. 0 is safest on Windows.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda', help='cuda | cpu')
    parser.add_argument('--patience', type=int, default=6,
                        help='Early-stopping patience on val Dice.')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from <output_dir>/last.pt if it exists, '
                             'including model + optimizer + history + epoch index.')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('[warn] CUDA requested but not available; falling back to CPU.', flush=True)
        args.device = 'cpu'
    device = torch.device(args.device)
    print(f'[info] Using device: {device}'
          + (f' ({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else ''), flush=True)

    data_dir = Path(args.data_dir)
    train_ds = SegDataset(data_dir / 'train', args.image_size, augment=True)
    val_ds = SegDataset(data_dir / 'val', args.image_size, augment=False)
    test_ds = SegDataset(data_dir / 'test', args.image_size, augment=False) if (data_dir / 'test').exists() else None

    print(f'[info] Train: {len(train_ds)}  Val: {len(val_ds)}'
          + (f'  Test: {len(test_ds)}' if test_ds else ''), flush=True)

    common = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common) if test_ds else None

    model = AttentionUNet(in_channels=3, base_filters=args.base_filters, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[info] Model parameters: {n_params:,}', flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / 'best_model.pt'
    last_path = output_dir / 'last.pt'
    history_path = output_dir / 'history.json'

    history = {'train_loss': [], 'val_dice': [], 'val_iou': [], 'val_loss': [], 'lr': []}
    best_val_dice = -1.0
    epochs_without_improve = 0
    start_epoch = 0

    # Resume from the per-epoch checkpoint if requested. This is the safety
    # mechanism added after a Kernel-Power 41 system crash wiped training at
    # epoch 4/25 - the next crash should cost <=1 epoch instead of everything.
    if args.resume and last_path.exists():
        prev = torch.load(str(last_path), map_location=device, weights_only=False)
        model.load_state_dict(prev['state_dict'])
        if 'optimizer_state' in prev:
            optimizer.load_state_dict(prev['optimizer_state'])
        if 'scheduler_state' in prev:
            try:
                scheduler.load_state_dict(prev['scheduler_state'])
            except Exception as exc:  # pragma: no cover
                print(f'[warn] could not restore scheduler state: {exc}', flush=True)
        history = prev.get('history', history)
        best_val_dice = float(prev.get('best_val_dice', best_val_dice))
        epochs_without_improve = int(prev.get('epochs_without_improve', 0))
        start_epoch = int(prev.get('epoch', 0))
        print(f'[info] Resumed from {last_path} at epoch {start_epoch} (best_val_dice={best_val_dice:.4f})', flush=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_dice = 0.0
        n_steps = 0
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = combined_dice_bce_loss(logits, y, dice_weight=args.dice_weight)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                running_dice += float(dice_coefficient((probs >= 0.5).float(), y))
            running_loss += float(loss)
            n_steps += 1

        train_loss = running_loss / max(n_steps, 1)
        train_dice = running_dice / max(n_steps, 1)

        val_metrics = _evaluate(model, val_loader, device)
        scheduler.step(val_metrics['dice'])

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_dice'].append(val_metrics['dice'])
        history['val_iou'].append(val_metrics['iou'])
        history['val_loss'].append(val_metrics['bce_loss'])
        history['lr'].append(lr_now)
        print(
            f'[epoch {epoch+1:02d}/{args.epochs}] '
            f'train_loss={train_loss:.4f}  train_dice~={train_dice:.4f}  '
            f'val_dice={val_metrics["dice"]:.4f}  val_iou={val_metrics["iou"]:.4f}  '
            f'lr={lr_now:.2e}  ({elapsed:.1f}s)',
            flush=True,
        )

        if val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            epochs_without_improve = 0
            torch.save({
                'state_dict': model.state_dict(),
                'config': vars(args),
                'val_metrics': val_metrics,
                'epoch': epoch + 1,
            }, best_path)
            print(f'        -> new best val_dice={best_val_dice:.4f}; weights saved to {best_path}', flush=True)
        else:
            epochs_without_improve += 1

        # Per-epoch resilience: write the full state + history to disk every
        # epoch so a power cut loses at most one epoch.
        torch.save({
            'state_dict': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'config': vars(args),
            'val_metrics': val_metrics,
            'epoch': epoch + 1,
            'history': history,
            'best_val_dice': best_val_dice,
            'epochs_without_improve': epochs_without_improve,
        }, last_path)
        with history_path.open('w', encoding='utf-8') as fh:
            json.dump(history, fh, indent=2)

        if epochs_without_improve >= args.patience:
            print(f'[info] Early stopping: no improvement in {args.patience} epochs.', flush=True)
            break

    with history_path.open('w', encoding='utf-8') as fh:
        json.dump(history, fh, indent=2)

    # Final test evaluation using the best checkpoint.
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
    eval_payload = {'val': _evaluate(model, val_loader, device)}
    if test_loader is not None:
        eval_payload['test'] = _evaluate(model, test_loader, device)
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
    except Exception as exc:  # pragma: no cover
        print(f'[warn] matplotlib plot failed: {exc}', flush=True)

    print('[done] Best val Dice =', f'{best_val_dice:.4f}')


if __name__ == '__main__':
    main()
