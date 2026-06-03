"""Train CrossJEPA Method 2: modality -> modality.

Prerequisite: 4 frozen per-modality I-JEPA teachers (one per
{T1, T1c, T2, FLAIR}). Train each via vanilla I-JEPA on healthy slices
of that modality only (use the existing src/train_v9b_stage1_jepa.py
infra; just filter the dataset to one modality at a time).

Until those teachers exist, this script raises a clear error pointing
at the teacher-training step.

Usage (Colab A100, after teachers are built):
    python src/train_v9c_method2_mod2mod.py \\
        --brats_root /content/brats \\
        --teachers_dir v9c_artifacts/modality_teachers \\
        --output_dir v9c_artifacts/crossjepa_method2 \\
        --image_size 256 --batch_size 8 --epochs 30 --lr 2e-4 --amp
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.checkpoint_utils import atomic_save
from src.research.v9c_crossjepa.conditioning import MODALITIES
from src.research.v9c_crossjepa.dataset_3d import (
    Mod2ModDataset, collate_mod2mod, discover_brats_scans,
)
from src.research.v9c_crossjepa.modality_to_modality import Mod2ModModel


class _IJEPATeacherWrapper(nn.Module):
    """Adapts a vanilla I-JEPA target_encoder to the teacher API
    expected by Mod2ModModel: `.embed_slice2d(x_2d) -> (B, embed_dim)`.
    """

    def __init__(self, vit_encoder: nn.Module, image_size: int = 256):
        super().__init__()
        self.encoder = vit_encoder
        self.image_size = image_size
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

    @torch.no_grad()
    def embed_slice2d(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W). I-JEPA's ViTEncoder expects 3 channels — tile.
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        z = self.encoder(x)                # (B, N, D)
        return z.mean(dim=1)               # (B, D)


def _load_modality_teachers(teachers_dir: Path,
                              image_size: int, device: str) -> dict:
    """Load 4 per-modality teachers. Each is a vanilla I-JEPA checkpoint
    saved by src/train_v9b_stage1_jepa.py, named like {modality}.pt."""
    from src.research.jepa import IJEPAModel
    out = {}
    missing = []
    for m in MODALITIES:
        p = teachers_dir / f'{m}.pt'
        if not p.exists():
            missing.append(m)
            continue
        ck = torch.load(str(p), map_location=device, weights_only=False)
        a = ck.get('args', {})
        net = IJEPAModel(image_size=a.get('image_size', image_size),
                          patch_size=a.get('patch_size', 16),
                          in_chans=3,
                          embed_dim=a.get('embed_dim', 384),
                          depth=a.get('depth', 12),
                          heads=a.get('heads', 6),
                          predictor_dim=a.get('predictor_dim', 192),
                          predictor_depth=a.get('predictor_depth', 6))
        net.load_state_dict(ck.get('model_state_dict', ck), strict=False)
        net = net.to(device).eval()
        out[m] = _IJEPATeacherWrapper(net.target_encoder, image_size=image_size).to(device)
    if missing:
        raise RuntimeError(
            f'Method 2 requires a frozen teacher for each modality.\n'
            f'  Missing teachers in {teachers_dir}: {missing}\n'
            f'  Train each via vanilla I-JEPA on healthy slices of that\n'
            f'  modality, save as <modality>.pt (e.g. T1.pt). See the\n'
            f'  v9b_stage1 trainer for reference.')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--brats_root', required=True,
                     help='Directory containing one subdir per BraTS patient.')
    ap.add_argument('--teachers_dir', required=True,
                     help='Directory with the 4 per-modality I-JEPA teacher .pt files.')
    ap.add_argument('--output_dir', default='v9c_artifacts/crossjepa_method2')
    ap.add_argument('--image_size', type=int, default=256)
    ap.add_argument('--patch_size', type=int, default=16)
    ap.add_argument('--embed_dim', type=int, default=384)
    ap.add_argument('--depth', type=int, default=12)
    ap.add_argument('--heads', type=int, default=6)
    ap.add_argument('--predictor_dim', type=int, default=192)
    ap.add_argument('--predictor_depth', type=int, default=6)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--resume', default='auto')
    ap.add_argument('--checkpoint_every_steps', type=int, default=200)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[init] device={device}')

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    log_path = out / 'training.log'

    def log(msg):
        print(msg, flush=True)
        with log_path.open('a', encoding='utf-8') as f:
            f.write(msg + '\n')

    # 1. Per-modality frozen teachers
    log(f'[init] loading 4 frozen modality teachers from {args.teachers_dir}')
    teachers = _load_modality_teachers(Path(args.teachers_dir), args.image_size, device)

    # 2. Dataset
    scan_index = discover_brats_scans(args.brats_root)
    log(f'[init] {len(scan_index)} BraTS patients indexed')
    ds = Mod2ModDataset(scan_index=scan_index, image_size=args.image_size, shuffle=True)
    loader = DataLoader(ds, batch_size=args.batch_size,
                         num_workers=args.num_workers, pin_memory=(device == 'cuda'),
                         collate_fn=collate_mod2mod)

    # 3. Model
    model = Mod2ModModel(
        frozen_teachers=teachers,
        image_size=args.image_size, patch_size=args.patch_size,
        embed_dim=args.embed_dim, depth=args.depth, heads=args.heads,
        predictor_dim=args.predictor_dim, predictor_depth=args.predictor_depth,
        teacher_embed_dim=args.embed_dim,
    ).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'[init] trainable params = {n_train/1e6:.1f}M')
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.05)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    # 4. Resume
    start_epoch = 0
    global_step = 0
    resume_path = out / 'last.pt'
    if args.resume == 'auto' and resume_path.exists():
        try:
            ck = torch.load(str(resume_path), map_location=device, weights_only=False)
            model.patch_embed.load_state_dict(ck['patch_embed_state_dict'])
            model.backbone.load_state_dict(ck['backbone_state_dict'])
            model.predictor.load_state_dict(ck['predictor_state_dict'])
            model.conditioning.load_state_dict(ck['conditioning_state_dict'])
            optimizer.load_state_dict(ck['optimizer_state_dict'])
            start_epoch = ck.get('epoch', 0)
            global_step = ck.get('global_step', 0)
            log(f'[resume] from epoch={start_epoch} step={global_step}')
        except Exception as exc:
            log(f'[resume] failed ({exc}); starting fresh')

    # 5. Train
    t_total = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_sum = 0.0
        cos_sum = 0.0
        n_steps = 0
        t_ep = time.perf_counter()
        for batch in loader:
            if batch is None:
                continue
            # Move tensors to device
            for k, v in list(batch.items()):
                if torch.is_tensor(v):
                    batch[k] = v.to(device, non_blocking=True)
                elif isinstance(v, dict):
                    batch[k] = {kk: vv.to(device, non_blocking=True)
                                 if torch.is_tensor(vv) else vv
                                 for kk, vv in v.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=args.amp):
                out_dict = model.training_step(batch)
                loss = out_dict['loss']
            if args.amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
            loss_sum += float(loss); cos_sum += float(out_dict['cos_sim'])
            n_steps += 1; global_step += 1
            if global_step % args.checkpoint_every_steps == 0:
                atomic_save({
                    'patch_embed_state_dict': model.patch_embed.state_dict(),
                    'backbone_state_dict': model.backbone.state_dict(),
                    'predictor_state_dict': model.predictor.state_dict(),
                    'conditioning_state_dict': model.conditioning.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch + 1, 'global_step': global_step,
                    'args': vars(args),
                    'description': 'CrossJEPA Method 2: modality->modality with 4 frozen teachers',
                }, resume_path)
        ep_loss = loss_sum / max(n_steps, 1)
        ep_cos = cos_sum / max(n_steps, 1)
        log(f'[epoch {epoch+1:03d}/{args.epochs}]  loss={ep_loss:.4f}  '
            f'cos_sim={ep_cos:.4f}  steps={n_steps}  '
            f'({time.perf_counter()-t_ep:.1f}s)')
        atomic_save({
            'patch_embed_state_dict': model.patch_embed.state_dict(),
            'backbone_state_dict': model.backbone.state_dict(),
            'predictor_state_dict': model.predictor.state_dict(),
            'conditioning_state_dict': model.conditioning.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1, 'global_step': global_step,
            'args': vars(args),
            'description': 'CrossJEPA Method 2: modality->modality with 4 frozen teachers',
        }, resume_path)
    log(f'\n[done] total {(time.perf_counter()-t_total)/60:.1f} min')
    log(f'[saved] {resume_path}')


if __name__ == '__main__':
    main()
