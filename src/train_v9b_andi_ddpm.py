"""Train an UNCONDITIONAL DDPM with pyramidal noise — proper ANDi setup.

Why this script exists (measured 2026-06-02):
  The current Stage 2 DDPM was trained with standard Gaussian noise AND
  conditioned on JEPA latents of the input. ANDi (arXiv 2312.01904)
  requires:
    1. Unconditional model (otherwise the conditioning is a cheat at
       test time — the model perfectly reconstructs ANY input)
    2. Pyramidal Gaussian noise during training (forces the model to
       learn multi-scale anatomy, which is what makes per-timestep
       denoising error a useful anomaly signal at inference)

  Our previous ANDi inference attempt (scripts/eval_ood_*) gave
  near-zero anomaly scores across all inputs because neither (1) nor
  (2) held. This script fixes both.

Output:
  v9b_artifacts/v9b_andi_ddpm/last.pt
  v9b_artifacts/v9b_andi_ddpm/training.log

Then run scripts/eval_ood_andi.py to evaluate.

Run (Colab A100):
  python src/train_v9b_andi_ddpm.py \
    --data_dir /content/neurolens/dataset_v8 \
    --output_dir /content/drive/MyDrive/neurolens/v9b_andi_ddpm \
    --epochs 100 --batch_size 32 --image_size 256 \
    --num_workers 4 --amp --checkpoint_every_steps 200 --resume auto

Expected wall-clock: ~3-4 h on A100, ~6-8 h on T4.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from .checkpoint_utils import atomic_save  # type: ignore
    from .research.pyramidal_noise import pyramidal_noise_like  # type: ignore
    from .research.latent_diffusion_decoder import LatentConditionedDDPM  # type: ignore
    from .train_v9b_stage1_jepa import HealthyOnlyDataset  # type: ignore
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.checkpoint_utils import atomic_save  # type: ignore
    from src.research.pyramidal_noise import pyramidal_noise_like  # type: ignore
    from src.research.latent_diffusion_decoder import LatentConditionedDDPM  # type: ignore
    from src.train_v9b_stage1_jepa import HealthyOnlyDataset  # type: ignore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='dataset_v8')
    ap.add_argument('--output_dir', default='v9b_artifacts/v9b_andi_ddpm')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--image_size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--cond_dim', type=int, default=384,
                     help='Kept for arch compat; unused since cond is zeros (unconditional)')
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--checkpoint_every_steps', type=int, default=200)
    ap.add_argument('--resume', default='auto')
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

    ds = HealthyOnlyDataset(Path(args.data_dir), image_size=args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.num_workers, pin_memory=(device == 'cuda'),
                         drop_last=True)
    log(f'[init] dataset n={len(ds)}  batches/epoch={len(loader)}')

    # Unconditional DDPM — keep the cond_dim slot, but we'll always pass
    # zeros so the model effectively becomes unconditional.
    ddpm = LatentConditionedDDPM(in_chans=3, base_ch=32,
                                   cond_dim=args.cond_dim).to(device)
    optimizer = torch.optim.Adam(ddpm.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    start_epoch = 0
    global_step = 0
    resume_path = out / 'last.pt'
    if args.resume == 'auto' and resume_path.exists():
        try:
            ck = torch.load(str(resume_path), map_location=device, weights_only=False)
            ddpm.load_state_dict(ck['model_state_dict'])
            optimizer.load_state_dict(ck['optimizer_state_dict'])
            start_epoch = ck.get('epoch', 0)
            global_step = ck.get('global_step', 0)
            log(f'[resume] from epoch={start_epoch} step={global_step}')
        except Exception as exc:
            log(f'[resume] failed ({exc}); starting fresh')

    t_total = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        ddpm.train()
        t_ep = time.perf_counter()
        loss_sum = 0.0
        n_batches = 0
        for x0 in loader:
            x0 = x0.to(device, non_blocking=True)
            B = x0.size(0)
            # Zero conditioning -> unconditional behaviour
            cond = torch.zeros(B, args.cond_dim, device=device)
            t = torch.randint(0, ddpm.num_train_timesteps, (B,), device=device)
            # *** Pyramidal noise — the ANDi training trick ***
            noise = pyramidal_noise_like(x0)
            x_t = ddpm.q_sample(x0, t, noise)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=args.amp):
                pred = ddpm.net(x_t, t, cond)
                loss = F.mse_loss(pred, noise)
            if args.amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddpm.parameters(), max_norm=1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
            loss_sum += float(loss); n_batches += 1; global_step += 1
            # Crash-safe atomic checkpoint
            if global_step % args.checkpoint_every_steps == 0:
                atomic_save({
                    'model_state_dict': ddpm.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch + 1,  # next epoch to resume from
                    'global_step': global_step,
                    'args': vars(args),
                    'description': 'unconditional DDPM trained with pyramidal noise (ANDi)',
                }, resume_path)
        ep_loss = loss_sum / max(n_batches, 1)
        log(f'[epoch {epoch+1:03d}/{args.epochs}]  loss={ep_loss:.4f}  '
            f'({time.perf_counter()-t_ep:.1f}s)')
        # End-of-epoch save
        atomic_save({
            'model_state_dict': ddpm.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1,
            'global_step': global_step,
            'args': vars(args),
            'description': 'unconditional DDPM trained with pyramidal noise (ANDi)',
        }, resume_path)

    log(f'\n[done] total {(time.perf_counter()-t_total)/60:.1f} min')
    log(f'[saved] {resume_path}')


if __name__ == '__main__':
    main()
