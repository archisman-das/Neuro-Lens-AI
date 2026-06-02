"""Retrain v9b SDF geometric tower with per-image brain SDFs as targets.

Why this script exists (measured 2026-06-02):
  The original SDF tower (trained in v9b Stage 2 alongside the DDPM)
  hit AUC = 0.10 for tumor-vs-healthy on OOD — meaning its anomaly
  score is *anti*-correlated with tumor presence. The bug is the
  training target: the tower was forced to reproduce a fixed
  synthetic ellipse template SDF regardless of input. So at inference
  on OOD healthy brains (different scanners / preprocessing / views),
  the tower predicts ellipse, but the input's actual brain shape
  deviates from that ellipse, and the (pred - template)^2 anomaly
  fires on the healthy brain rather than on tumor mass effect.

Proper retrain:
  - Target per training image = SDF computed from THAT image's own
    foreground-thresholded brain mask (via scipy distance transform).
  - Loss = MSE(predicted_sdf, image_sdf).
  - Training set = healthy slices only (HealthyOnlyDataset), so the
    tower learns "what a healthy MRI's brain SDF looks like" given the
    image — a normative model in the proper sense.

At inference:
  - Compute image_sdf for the input via the same auto-threshold pipeline.
  - tower predicts predicted_sdf from the image.
  - Anomaly = (predicted_sdf - image_sdf)^2. If the tower has learned a
    strong "image -> healthy SDF" mapping, it will be CORRECT on healthy
    inputs (low anomaly) and WRONG on tumor inputs whose appearance
    triggers the tower to predict a 'wrong' (healthy-shaped) SDF that
    no longer matches the actual distorted SDF — high anomaly.

This is the proper normative anomaly signal the geometry tower should
have been using all along.

Output: v9b_artifacts/v9b_stage2_sdf_v2/last.pt (separate from the
original Stage 2 checkpoint so we can compare side-by-side without
clobbering anything that's already shipped).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.research.sdf_geometric_tower import GeometricSDFTower
from src.train_v9b_stage1_jepa import HealthyOnlyDataset
from src.checkpoint_utils import atomic_save


def compute_image_sdf(img_tensor: torch.Tensor,
                       fg_threshold_frac: float = 0.05) -> torch.Tensor:
    """Compute a normalized 2D SDF from an MRI image batch.

    img_tensor: (B, 3, H, W) in [0, 1] (or any positive range).
    Returns (B, 1, H, W) SDF normalized to roughly [-1, 1]:
      negative inside the brain (further inside = more negative),
      positive outside, zero on boundary.

    Brain mask = pixels brighter than `fg_threshold_frac * max(image)`.
    SDF computed via scipy distance_transform_edt on CPU per-image then
    re-uploaded to GPU. ~5 ms per 256x256 image — acceptable for the
    retraining hot loop.
    """
    from scipy.ndimage import distance_transform_edt
    device = img_tensor.device
    B, C, H, W = img_tensor.shape
    gray = img_tensor.mean(dim=1)             # (B, H, W) grayscale
    out = torch.zeros(B, 1, H, W, device=device)
    for b in range(B):
        g = gray[b].cpu().numpy()
        if g.max() <= 0:
            out[b, 0] = 0
            continue
        thr = fg_threshold_frac * float(g.max())
        mask = (g > thr).astype(np.float32)
        if mask.sum() < 10 or (1 - mask).sum() < 10:
            out[b, 0] = 0
            continue
        d_in  = distance_transform_edt(mask)       # distance to nearest 0 (outside)
        d_out = distance_transform_edt(1 - mask)   # distance to nearest 1 (inside)
        sdf = d_out - d_in                          # >0 outside brain, <0 inside
        # Normalise: scale by half the image dim so values fall in ~[-1,1]
        sdf = sdf / (max(H, W) * 0.5)
        sdf = np.clip(sdf, -1.5, 1.5).astype(np.float32)
        out[b, 0] = torch.from_numpy(sdf).to(device)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='dataset_v8')
    ap.add_argument('--output_dir', default='v9b_artifacts/v9b_stage2_sdf_v2')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--image_size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--init_from', default='v9b_artifacts/v9b_stage2/last.pt',
                     help='Warm-start SDF weights from the original Stage 2 ckpt (faster convergence).')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[init] device={device}'
          + (f' ({torch.cuda.get_device_name(0)})' if device == 'cuda' else ''))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'training.log'
    log_path.write_text('')  # fresh log

    def log(msg: str):
        print(msg, flush=True)
        with log_path.open('a', encoding='utf-8') as f:
            f.write(msg + '\n')

    # Dataset: healthy slices only, same loader as Stage 1/2
    ds = HealthyOnlyDataset(Path(args.data_dir), image_size=args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.num_workers,
                         pin_memory=(device == 'cuda'), drop_last=True)
    log(f'[init] dataset={args.data_dir}  n={len(ds)}  '
        f'batches/epoch={len(loader)}')

    # Model
    model = GeometricSDFTower(image_size=args.image_size, base_ch=32).to(device)
    if args.init_from and Path(args.init_from).exists():
        try:
            ck = torch.load(args.init_from, map_location=device, weights_only=False)
            sd = ck.get('sdf_state_dict', ck)
            miss, unexp = model.load_state_dict(sd, strict=False)
            log(f'[init] warm-start from {args.init_from}  '
                f'(missing={len(miss)} unexpected={len(unexp)})')
        except Exception as exc:
            log(f'[init] warm-start failed ({exc}); using random init')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    t_total = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        t_ep = time.perf_counter()
        loss_sum = 0.0
        n_batches = 0
        for batch in loader:
            x = batch.to(device, non_blocking=True)
            target_sdf = compute_image_sdf(x)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                pred = model(x)
                loss = F.mse_loss(pred, target_sdf)
            if device == 'cuda':
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
            loss_sum += float(loss)
            n_batches += 1
        ep_loss = loss_sum / max(n_batches, 1)
        ep_time = time.perf_counter() - t_ep
        log(f'[epoch {epoch+1:02d}/{args.epochs}]  sdf_v2_mse={ep_loss:.6f}  ({ep_time:.1f}s)')

        # Save after every epoch (atomic, crash-safe)
        atomic_save({
            'sdf_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': vars(args),
            'description': 'SDF tower v2: trained against per-image brain SDFs',
        }, out_dir / 'last.pt')

    log(f'\n[done] total {(time.perf_counter()-t_total)/60:.1f} min')
    log(f'[saved] {out_dir / "last.pt"}')


if __name__ == '__main__':
    main()
