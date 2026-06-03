"""Train CrossJEPA Method 1: 3D volume -> 2D slice (frozen v8 teacher).

Usage (Colab A100/H100):
    python src/train_v9c_method1_vol2slice.py \\
        --scans_glob '/content/brats/*.nii.gz' \\
        --v8_ckpt model/best_micro.pt \\
        --output_dir v9c_artifacts/crossjepa_method1 \\
        --volume_size 144 192 192 \\
        --batch_size 2 \\
        --epochs 30 \\
        --lr 2e-4 \\
        --num_workers 2 --amp --resume auto

Disk-pressure note: the dataloader STREAMS volumes from the local glob;
it does not pre-materialize. For HF-streamed setups, pre-download to
the Colab VM via `huggingface_hub.snapshot_download` and point at that
local cache.
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.checkpoint_utils import atomic_save
from src.research.v9c_crossjepa.dataset_3d import Vol2SliceDataset
from src.research.v9c_crossjepa.v8_teacher import V8FrozenTeacher
from src.research.v9c_crossjepa.volume_to_slice import Vol2SliceModel


def _collate(batch):
    """Stack volumes + scalars; carry through scan_id list."""
    keys_stack = ('volume', 'target_slice_rgb', 'plane_idx', 'slice_idx_norm',
                  'voxel_spacing', 'intensity_hist')
    out = {k: torch.stack([s[k] for s in batch], dim=0) for k in keys_stack}
    out['scan_ids'] = [s['scan_id'] for s in batch]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scans_glob', required=True,
                     help='Glob pattern for .nii.gz volumes (BraTS, LGG, etc.)')
    ap.add_argument('--v8_ckpt', default='model/best_micro.pt',
                     help='Frozen v8 UNet checkpoint (encoder is peeled off)')
    ap.add_argument('--output_dir', default='v9c_artifacts/crossjepa_method1')
    ap.add_argument('--volume_size', nargs=3, type=int, default=[144, 192, 192],
                     help='(D, H, W) after center-crop / zero-pad')
    ap.add_argument('--in_channels', type=int, default=4,
                     help='Input channel width for the 3D ViT')
    ap.add_argument('--patch_size', type=int, default=16)
    ap.add_argument('--encoder_dim', type=int, default=384)
    ap.add_argument('--encoder_depth', type=int, default=12)
    ap.add_argument('--encoder_heads', type=int, default=6)
    ap.add_argument('--predictor_dim', type=int, default=192)
    ap.add_argument('--predictor_depth', type=int, default=6)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--slices_per_volume', type=int, default=6)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--resume', default='auto')
    ap.add_argument('--checkpoint_every_steps', type=int, default=200)
    # --- Preprocessing-invariance augmentation (Fix A, June 2026) ---
    # Critical: the initial train run learned dataset-specific
    # preprocessing rather than the healthy-brain manifold, so on
    # held-out IXI it scored healthy volumes HIGHER than tumors. These
    # augmentations randomly perturb intensity/contrast/gamma/bias-field
    # to force preprocessing invariance.
    ap.add_argument('--augment', action='store_true',
                     help='Enable intensity/contrast/gamma/bias-field augmentation '
                          '(strongly recommended for the v2 retrain).')
    ap.add_argument('--aug_intensity_jitter', type=float, default=0.20)
    ap.add_argument('--aug_contrast_jitter', type=float, default=0.30)
    ap.add_argument('--aug_gamma_jitter', type=float, default=0.30)
    ap.add_argument('--aug_bias_field_strength', type=float, default=0.15)
    ap.add_argument('--aug_noise_std', type=float, default=0.02)
    ap.add_argument('--aug_renormalize_pct', type=float, default=0.5)
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

    # 1. Frozen v8 teacher (encoder side only)
    log(f'[init] loading frozen v8 teacher from {args.v8_ckpt}')
    teacher = V8FrozenTeacher.from_unet_checkpoint(
        args.v8_ckpt, in_channels=3, image_size=384, device=device)
    log(f'  v8 teacher embed dim = {teacher.embed_dim}')

    # 2. Dataset
    scan_paths = sorted(glob.glob(args.scans_glob))
    if not scan_paths:
        sys.exit(f'ERROR: no scans matched {args.scans_glob!r}')
    log(f'[init] {len(scan_paths)} volumes matched')
    ds = Vol2SliceDataset(scan_paths=scan_paths,
                            volume_size=tuple(args.volume_size),
                            in_channels=args.in_channels,
                            slices_per_volume=args.slices_per_volume,
                            shuffle=True,
                            augment=args.augment,
                            aug_intensity_jitter=args.aug_intensity_jitter,
                            aug_contrast_jitter=args.aug_contrast_jitter,
                            aug_gamma_jitter=args.aug_gamma_jitter,
                            aug_bias_field_strength=args.aug_bias_field_strength,
                            aug_noise_std=args.aug_noise_std,
                            aug_renormalize_pct=args.aug_renormalize_pct)
    log(f'[init] augmentation = {"ON" if args.augment else "OFF"}'
        + (f' (jitter={args.aug_intensity_jitter}, contrast={args.aug_contrast_jitter}, '
           f'gamma={args.aug_gamma_jitter}, bias={args.aug_bias_field_strength}, '
           f'noise={args.aug_noise_std}, renorm_p={args.aug_renormalize_pct})'
           if args.augment else ''))
    loader = DataLoader(ds, batch_size=args.batch_size,
                         num_workers=args.num_workers, pin_memory=(device == 'cuda'),
                         collate_fn=_collate)

    # 3. Model + optimizer
    model = Vol2SliceModel(
        v8_teacher=teacher,
        volume_size=tuple(args.volume_size),
        in_chans=args.in_channels,
        patch_size=args.patch_size,
        encoder_dim=args.encoder_dim,
        encoder_depth=args.encoder_depth,
        encoder_heads=args.encoder_heads,
        predictor_dim=args.predictor_dim,
        predictor_depth=args.predictor_depth,
    ).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.teacher.parameters())
    log(f'[init] trainable params = {n_train/1e6:.1f}M  '
        f'frozen teacher = {n_frozen/1e6:.1f}M')
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
            model.encoder.load_state_dict(ck['encoder_state_dict'])
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
        # Keep teacher in eval no matter what
        model.teacher.encoder.eval()
        loss_sum = 0.0
        cos_sum = 0.0
        n_steps = 0
        t_ep = time.perf_counter()
        for batch in loader:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device, non_blocking=True)
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
                    'encoder_state_dict': model.encoder.state_dict(),
                    'predictor_state_dict': model.predictor.state_dict(),
                    'conditioning_state_dict': model.conditioning.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'args': vars(args),
                    'description': 'CrossJEPA Method 1: 3D ViT + v8 frozen teacher',
                }, resume_path)
        ep_loss = loss_sum / max(n_steps, 1)
        ep_cos = cos_sum / max(n_steps, 1)
        log(f'[epoch {epoch+1:03d}/{args.epochs}]  loss={ep_loss:.4f}  '
            f'cos_sim={ep_cos:.4f}  steps={n_steps}  '
            f'({time.perf_counter()-t_ep:.1f}s)')
        atomic_save({
            'encoder_state_dict': model.encoder.state_dict(),
            'predictor_state_dict': model.predictor.state_dict(),
            'conditioning_state_dict': model.conditioning.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1,
            'global_step': global_step,
            'args': vars(args),
            'description': 'CrossJEPA Method 1: 3D ViT + v8 frozen teacher',
        }, resume_path)
    log(f'\n[done] total {(time.perf_counter()-t_total)/60:.1f} min')
    log(f'[saved] {resume_path}')


if __name__ == '__main__':
    main()
