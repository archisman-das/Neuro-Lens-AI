"""v5 segmentation trainer: tumor-positive + healthy-brain joint training.

Headline differences from v3:
  1. Reads dataset_v5 (positive + negative samples; negatives have empty
     PNG masks). Built by `python prepare_v5_dataset.py`.
  2. Balanced batch sampler: each batch contains ~50% positive
     (non-empty mask) and ~50% negative (empty mask). Without this the
     gradient is dominated by whichever class has more samples and the
     positive bias / no-detection bias persist.
  3. Loss is Dice (positives only) + BCE (all samples). On empty-mask
     samples Dice is undefined; we contribute zero Dice loss for them
     and rely on BCE alone to penalise spurious positive predictions.
  4. Optional modality-dropout augmentation: with probability `p_mod_drop`,
     randomly mask one or two of the three input channels (set to channel
     mean). Teaches the model to handle single-modality grayscale inputs
     without needing the v3 cascade T1c trick.
  5. Reports false-positive-rate on validation (pixels predicted >=
     threshold inside scans whose ground-truth mask is empty) in addition
     to per-scan Dice / IoU. This is the metric that matters for the
     no-tumor-bias bug; v3's val_dice never measured it.

Output: segmentation_artifacts/attention_unet_v5/
  best_model.pt        (val Dice + tiny FP-rate penalty -> single picked checkpoint)
  last.pt              (most recent epoch)
  history.json         per-epoch metrics
  training.log         text log including FP rate
  training_curves.png  Dice + FP-rate vs epoch
  evaluation_metrics.json  final val + test report

Usage:
  python src/train_segmentation_v5.py --data_dir dataset_v5 \\
      --epochs 25 --batch_size 8 --image_size 256

Run prepare_v5_dataset.py first to build dataset_v5/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# -----------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------

def _set_seed(s: int = 42) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------

class V5SegDataset(Dataset):
    """Dataset that exposes per-sample 'has_tumor' flag for balanced sampling."""

    def __init__(self, root: Path, image_size: int, p_mod_drop: float = 0.0,
                  augment: bool = False, imagenet_normalize: bool = True):
        self.root = Path(root)
        self.image_size = image_size
        self.p_mod_drop = p_mod_drop
        self.augment = augment
        self.imagenet_normalize = imagenet_normalize
        images = sorted((self.root / 'images').iterdir())
        masks = sorted((self.root / 'masks').iterdir())
        assert len(images) == len(masks), (
            f'image/mask count mismatch in {root}: {len(images)} vs {len(masks)}'
        )
        # Compute has_tumor by mask non-zero on a cheap thumbnail.
        self.samples = []
        for img_p, msk_p in zip(images, masks):
            assert img_p.stem == msk_p.stem, f'pair mismatch: {img_p.name} vs {msk_p.name}'
            try:
                m = np.array(Image.open(msk_p).convert('L').resize((64, 64), Image.NEAREST))
                has_tumor = bool((m > 127).any())
            except Exception:
                has_tumor = False
            self.samples.append((img_p, msk_p, has_tumor))

    def __len__(self) -> int:
        return len(self.samples)

    def has_tumor_flags(self) -> list:
        return [s[2] for s in self.samples]

    def __getitem__(self, i: int):
        img_p, msk_p, has_tumor = self.samples[i]
        img = Image.open(img_p).convert('RGB').resize((self.image_size, self.image_size), Image.BILINEAR)
        msk = Image.open(msk_p).convert('L').resize((self.image_size, self.image_size), Image.NEAREST)
        x = np.asarray(img, dtype=np.float32) / 255.0
        y = (np.asarray(msk, dtype=np.uint8) > 127).astype(np.float32)

        if self.augment:
            # Horizontal flip
            if random.random() < 0.5:
                x = x[:, ::-1, :].copy()
                y = y[:, ::-1].copy()
            # Vertical flip
            if random.random() < 0.2:
                x = x[::-1, :, :].copy()
                y = y[::-1, :].copy()
            # Brightness / contrast jitter
            if random.random() < 0.5:
                x = np.clip(x * (1.0 + (random.random() - 0.5) * 0.2), 0, 1)
                x = np.clip(x + (random.random() - 0.5) * 0.1, 0, 1)

        # Modality dropout. Zero (or grey) out 1 or 2 channels at random.
        # Forces the model to predict from any single-modality view.
        if self.augment and self.p_mod_drop > 0 and random.random() < self.p_mod_drop:
            n_drop = random.choice([1, 2])
            chans = random.sample([0, 1, 2], n_drop)
            for c in chans:
                x[:, :, c] = x[:, :, c].mean()  # grey replacement, not zero

        if self.imagenet_normalize:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            x = (x - mean) / std

        x_t = torch.from_numpy(x.transpose(2, 0, 1).copy()).float()
        y_t = torch.from_numpy(y[None].copy()).float()
        return x_t, y_t, float(has_tumor)


# -----------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------

def _build_model() -> nn.Module:
    """SMP UNet + ResNet34 (matches v3) so we can compare cleanly."""
    import segmentation_models_pytorch as smp
    return smp.Unet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
    )


# -----------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------

class DiceBceLoss(nn.Module):
    """Dice (positives only) + BCE (all samples).

    For samples with an entirely empty target mask, Dice is degenerate
    (target_sum=0). We zero out the Dice contribution for those and let
    BCE drive the gradient. BCE penalises any false-positive pixel in
    those samples - exactly the regularisation v3 lacked.
    """
    def __init__(self, dice_w: float = 0.7, bce_w: float = 0.3,
                  pos_weight: float = 1.0):
        super().__init__()
        self.dice_w = dice_w
        self.bce_w = bce_w
        self.register_buffer('pos_weight', torch.tensor(pos_weight))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        pred = torch.sigmoid(logits)
        # Per-sample target sum used to mask Dice for empty targets.
        target_sum = target.flatten(1).sum(dim=1)  # (B,)
        pos_mask = (target_sum > 0).float()  # 1 if positive sample, 0 if negative
        # Dice per sample
        inter = (pred * target).flatten(1).sum(dim=1)
        denom = pred.flatten(1).sum(dim=1) + target.flatten(1).sum(dim=1)
        dice = (2 * inter + eps) / (denom + eps)
        dice_loss_per_sample = (1.0 - dice) * pos_mask
        # Reduce safely.
        n_pos = pos_mask.sum().clamp(min=1.0)
        dice_loss = dice_loss_per_sample.sum() / n_pos
        # BCE per pixel, all samples.
        bce = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight,
        )
        return self.dice_w * dice_loss + self.bce_w * bce


# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------

@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
                threshold: float = 0.5) -> dict:
    """Compute Dice + IoU on positive samples and FP-rate on negatives."""
    model.eval()
    dices = []
    ious = []
    fp_rates = []  # for each negative sample: fraction of pixels predicted >= threshold
    n_pos = 0
    n_neg = 0
    for x, y, has in loader:
        x = x.to(device)
        y = y.to(device)
        p = torch.sigmoid(model(x))
        m = (p >= threshold).float()
        # Per-sample metrics
        for i in range(x.size(0)):
            yi = y[i]
            mi = m[i]
            if yi.sum() > 0:
                inter = (mi * yi).sum().item()
                pred_sum = mi.sum().item()
                tgt_sum = yi.sum().item()
                d = (2 * inter + 1e-6) / (pred_sum + tgt_sum + 1e-6)
                u = (inter + 1e-6) / (pred_sum + tgt_sum - inter + 1e-6)
                dices.append(d)
                ious.append(u)
                n_pos += 1
            else:
                fp_rates.append(mi.mean().item())
                n_neg += 1
    return {
        'n_positive': n_pos,
        'n_negative': n_neg,
        'dice_mean': float(np.mean(dices)) if dices else 0.0,
        'iou_mean': float(np.mean(ious)) if ious else 0.0,
        'fp_rate_mean': float(np.mean(fp_rates)) if fp_rates else 0.0,
        'fp_rate_p95': float(np.percentile(fp_rates, 95)) if fp_rates else 0.0,
    }


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def _make_balanced_loader(ds: V5SegDataset, batch_size: int, num_workers: int,
                            shuffle_class_balance: bool = True) -> DataLoader:
    """50/50 batches of positives vs negatives via WeightedRandomSampler."""
    flags = ds.has_tumor_flags()
    n_pos = sum(1 for f in flags if f)
    n_neg = sum(1 for f in flags if not f)
    if n_pos == 0 or n_neg == 0 or not shuffle_class_balance:
        return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            pin_memory=True, drop_last=True)
    w_pos = 1.0 / n_pos
    w_neg = 1.0 / n_neg
    weights = [w_pos if f else w_neg for f in flags]
    sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                        pin_memory=True, drop_last=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='dataset_v5')
    ap.add_argument('--output_dir', default='segmentation_artifacts/attention_unet_v5')
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--image_size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-5)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--p_mod_drop', type=float, default=0.3,
                     help='Probability of dropping 1-2 channels per train sample.')
    ap.add_argument('--bce_pos_weight', type=float, default=2.0,
                     help='Multiplier on positive-pixel BCE; >1 reduces false-negative rate.')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--resume', default=None, help='Path to a .pt to resume from.')
    args = ap.parse_args()

    _set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[info] device={device}, dataset={args.data_dir}, output={args.output_dir}')

    train_ds = V5SegDataset(Path(args.data_dir) / 'train', args.image_size,
                              p_mod_drop=args.p_mod_drop, augment=True)
    val_ds = V5SegDataset(Path(args.data_dir) / 'val', args.image_size, augment=False)
    test_ds = V5SegDataset(Path(args.data_dir) / 'test', args.image_size, augment=False)
    print(f'[info] train n={len(train_ds)} '
          f'(pos={sum(train_ds.has_tumor_flags())}, '
          f'neg={len(train_ds) - sum(train_ds.has_tumor_flags())})')
    print(f'[info] val   n={len(val_ds)}')
    print(f'[info] test  n={len(test_ds)}')

    train_loader = _make_balanced_loader(train_ds, args.batch_size, args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)

    model = _build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = DiceBceLoss(dice_w=0.7, bce_w=0.3, pos_weight=args.bce_pos_weight).to(device)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list = []
    best_score = -1.0
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = int(ckpt.get('epoch', 0)) + 1
        best_score = float(ckpt.get('best_score', -1.0))
        print(f'[info] resumed from {args.resume} at epoch {start_epoch}')

    log_path = out_dir / 'training.log'
    with log_path.open('a', encoding='utf-8') as logf:
        for ep in range(start_epoch, args.epochs + 1):
            model.train()
            t0 = time.time()
            train_loss = 0.0
            n_batches = 0
            for x, y, _has in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        logits = model(x)
                        loss = loss_fn(logits, y)
                    if not torch.isfinite(loss):
                        # NaN guard
                        continue
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if not torch.isfinite(grad_norm):
                        scaler.update()
                        continue
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(x)
                    loss = loss_fn(logits, y)
                    if not torch.isfinite(loss):
                        continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                train_loss += float(loss)
                n_batches += 1
            scheduler.step()
            avg_loss = train_loss / max(n_batches, 1)

            val_metrics = _evaluate(model, val_loader, device)
            # Composite score: Dice on positives MINUS a penalty proportional to
            # FP-rate on negatives. A model that gets 0.7 Dice but 30% pixel-FPR
            # on healthy brains scores 0.7 - 0.3*5 = -0.8 - worse than a model
            # that gets 0.6 Dice with 1% FPR (0.6 - 0.01*5 = 0.55). This is the
            # objective that actually matches the bug we're trying to fix.
            composite = val_metrics['dice_mean'] - 5.0 * val_metrics['fp_rate_mean']
            elapsed = time.time() - t0
            line = (
                f'[epoch {ep:02d}/{args.epochs}] '
                f'train_loss={avg_loss:.4f}  '
                f'val_dice={val_metrics["dice_mean"]:.4f}  '
                f'val_iou={val_metrics["iou_mean"]:.4f}  '
                f'val_fp_rate={val_metrics["fp_rate_mean"]:.4f}  '
                f'val_fp_p95={val_metrics["fp_rate_p95"]:.4f}  '
                f'composite={composite:.4f}  '
                f'lr={scheduler.get_last_lr()[0]:.2e}  '
                f'({elapsed:.1f}s)'
            )
            print(line)
            logf.write(line + '\n')
            logf.flush()
            history.append({
                'epoch': ep, 'train_loss': avg_loss, 'val': val_metrics,
                'composite': composite,
            })

            torch.save({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': ep, 'best_score': best_score,
                'architecture': 'Unet', 'encoder': 'resnet34',
                'image_size': args.image_size,
                'config': {'base_filters': 32, 'dropout': 0.2,
                           'image_size': args.image_size, '_v5': True},
            }, out_dir / 'last.pt')

            if composite > best_score:
                best_score = composite
                torch.save({
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': ep, 'best_score': best_score,
                    'architecture': 'Unet', 'encoder': 'resnet34',
                    'image_size': args.image_size,
                    'config': {'base_filters': 32, 'dropout': 0.2,
                               'image_size': args.image_size, '_v5': True},
                }, out_dir / 'best_model.pt')
                print(f'        -> new best composite={best_score:.4f}; weights saved.')

        # Final test evaluation on the best checkpoint.
        best = torch.load(out_dir / 'best_model.pt', map_location=device, weights_only=False)
        model.load_state_dict(best['state_dict'])
        test_metrics = _evaluate(model, test_loader, device)
        final = {'val_best': max(history, key=lambda h: h['composite'])['val'],
                  'test': test_metrics, 'best_composite': best_score}
        print(f'[done] best composite={best_score:.4f}  '
               f'test_dice={test_metrics["dice_mean"]:.4f}  '
               f'test_fp_rate={test_metrics["fp_rate_mean"]:.4f}')
        with (out_dir / 'evaluation_metrics.json').open('w', encoding='utf-8') as fh:
            json.dump(final, fh, indent=2)
        with (out_dir / 'history.json').open('w', encoding='utf-8') as fh:
            json.dump(history, fh, indent=2)


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)
    main()
