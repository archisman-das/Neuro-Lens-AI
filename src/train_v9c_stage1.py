"""Train v9c JEPA predictor on frozen DINOv2 backbone.

Phase 3 of the v9c plan. Uses the expanded dataset_v8 healthy pool
(now ~31k slices across 5 source studies: kaggle_neg, openneuro,
ixi2d, radiata-DLBS, radiata-NKI-RS, radiata-OASIS-1/2 — plus
BraTS no-tumor slices).

Only the JEPA predictor head is trainable (~5-10M params); DINOv2
ViT-B/14 stays frozen. This is dramatically cheaper than the v9b
from-scratch pretrain — should converge in 20-40 epochs on Colab A100.

CLI:
  python src/train_v9c_stage1.py \
    --data_dir dataset_v8 --output_dir v9b_artifacts/v9c_stage1 \
    --epochs 30 --batch_size 16 --num_workers 4 --amp --resume auto
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.checkpoint_utils import atomic_save
from src.research.v9c_dinov2_jepa import V9CModel
from src.train_v9b_stage1_jepa import HealthyOnlyDataset


def collate_uint8(batch):
    """Returns a list of HxWx3 uint8 numpy arrays — DINOv2's image
    processor expects PIL/numpy, not tensors."""
    out = []
    for t in batch:
        # HealthyOnlyDataset returns CHW float in [0,1] — convert back
        arr = (t.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        out.append(arr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='dataset_v8')
    ap.add_argument('--output_dir', default='v9b_artifacts/v9c_stage1')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--predictor_depth', type=int, default=6)
    ap.add_argument('--predictor_dim', type=int, default=384)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--resume', default='auto')
    ap.add_argument('--checkpoint_every_steps', type=int, default=200)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[init] device={device}'
          + (f' ({torch.cuda.get_device_name(0)})' if device == 'cuda' else ''))

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    log_path = out / 'training.log'

    def log(msg):
        print(msg, flush=True)
        with log_path.open('a', encoding='utf-8') as f:
            f.write(msg + '\n')

    # HealthyOnlyDataset uses image_size=256 to match v9b. DINOv2 needs
    # 224 input but the image processor handles the resize.
    ds = HealthyOnlyDataset(Path(args.data_dir), image_size=256)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.num_workers,
                         pin_memory=(device == 'cuda'), drop_last=True,
                         collate_fn=collate_uint8)
    log(f'[init] dataset n={len(ds)}  batches/epoch={len(loader)}')

    model = V9CModel(predictor_depth=args.predictor_depth,
                      predictor_dim=args.predictor_dim, device=device)
    # Only predictor params are trainable
    n_trainable = sum(p.numel() for p in model.predictor.parameters()
                       if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.dino.parameters())
    log(f'[init] predictor params: {n_trainable:,}  frozen DINOv2: {n_frozen:,}')

    optimizer = torch.optim.Adam(model.predictor.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    start_epoch = 0
    global_step = 0
    resume_path = out / 'last.pt'
    if args.resume == 'auto' and resume_path.exists():
        try:
            ck = torch.load(str(resume_path), map_location=device, weights_only=False)
            model.predictor.load_state_dict(ck['predictor_state_dict'])
            optimizer.load_state_dict(ck['optimizer_state_dict'])
            start_epoch = ck.get('epoch', 0)
            global_step = ck.get('global_step', 0)
            log(f'[resume] from epoch={start_epoch} step={global_step}')
        except Exception as exc:
            log(f'[resume] failed: {exc}; starting fresh')

    t_total = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        t_ep = time.perf_counter()
        loss_sum = 0.0; n_batches = 0
        for images in loader:
            # `images` is a list of uint8 numpy arrays at this point
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=args.amp):
                loss = model.training_step(images)
            if args.amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), max_norm=1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
            loss_sum += float(loss); n_batches += 1; global_step += 1
            if global_step % args.checkpoint_every_steps == 0:
                atomic_save({
                    'predictor_state_dict': model.predictor.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'args': vars(args),
                    'description': 'v9c: JEPA predictor on frozen DINOv2-base',
                }, resume_path)
        ep_loss = loss_sum / max(n_batches, 1)
        log(f'[epoch {epoch+1:03d}/{args.epochs}]  loss={ep_loss:.4f}  '
            f'({time.perf_counter()-t_ep:.1f}s)')
        # End-of-epoch save
        atomic_save({
            'predictor_state_dict': model.predictor.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1,
            'global_step': global_step,
            'args': vars(args),
            'description': 'v9c: JEPA predictor on frozen DINOv2-base',
        }, resume_path)
    log(f'\n[done] total {(time.perf_counter()-t_total)/60:.1f} min')
    log(f'[saved] {resume_path}')


if __name__ == '__main__':
    main()
