"""v9b Stage-2: train diffusion decoder + SDF geometric tower on healthy MRI.

Uses the pretrained JEPA (from Stage-1) as a FROZEN encoder to extract
global "healthy latent" conditioning vectors. Trains:
  1. Latent-conditioned DDPM to reconstruct healthy MRI from healthy latent
  2. SDF geometric tower to predict the brain SDF from healthy MRI

Both heads are trained jointly on the same healthy-only dataset used in
Stage 1. After this stage, at inference:
  - JEPA produces appearance-anomaly map (prediction error)
  - SDF tower produces geometry-anomaly map (SDF deviation)
  - DDPM generates healthy counterfactual given current JEPA latent
"""
from __future__ import annotations
import argparse, math, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from .checkpoint_utils import atomic_save as _atomic_save  # type: ignore
    from .train_v9b_stage1_jepa import HealthyOnlyDataset  # type: ignore
    from .research.jepa import IJEPAModel  # type: ignore
    from .research.latent_diffusion_decoder import LatentConditionedDDPM  # type: ignore
    from .research.sdf_geometric_tower import GeometricSDFTower  # type: ignore
    from .research.geometric_prior import synthetic_brain_sdf_template  # type: ignore
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.checkpoint_utils import atomic_save as _atomic_save  # type: ignore
    from src.train_v9b_stage1_jepa import HealthyOnlyDataset  # type: ignore
    from src.research.jepa import IJEPAModel  # type: ignore
    from src.research.latent_diffusion_decoder import LatentConditionedDDPM  # type: ignore
    from src.research.sdf_geometric_tower import GeometricSDFTower  # type: ignore
    from src.research.geometric_prior import synthetic_brain_sdf_template  # type: ignore


def jepa_global_latent(model: IJEPAModel, x: torch.Tensor) -> torch.Tensor:
    """Mean-pooled target-encoder embedding -> (B, embed_dim) global latent."""
    with torch.no_grad():
        z = model.encode_full(x)  # (B, N, D)
    return z.mean(dim=1)  # (B, D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset_v8")
    ap.add_argument("--extra_dirs", nargs="*", default=[])
    ap.add_argument("--output_dir", default="segmentation_artifacts/v9b_stage2")
    ap.add_argument("--jepa_ckpt", required=True,
                    help="Path to pretrained JEPA from Stage 1 (.pt)")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--lambda_ddpm", type=float, default=1.0)
    ap.add_argument("--lambda_sdf", type=float, default=0.5)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--cache_in_ram", action="store_true")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--checkpoint_every_steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    # Load JEPA (frozen)
    ckpt = torch.load(args.jepa_ckpt, map_location=device, weights_only=False)
    ja = ckpt.get("args", {})
    jepa = IJEPAModel(
        image_size=ja.get("image_size", args.image_size),
        patch_size=ja.get("patch_size", 16),
        embed_dim=ja.get("embed_dim", 384),
        depth=ja.get("depth", 12),
        heads=ja.get("heads", 6),
        predictor_dim=ja.get("predictor_dim", 192),
        predictor_depth=ja.get("predictor_depth", 6),
    ).to(device)
    jepa.load_state_dict(ckpt["model_state_dict"])
    jepa.eval()
    for p in jepa.parameters(): p.requires_grad = False
    print(f"[v9b-stage2] loaded JEPA from {args.jepa_ckpt}  embed_dim={jepa.embed_dim}",
          flush=True)

    # Build DDPM + SDF tower
    ddpm = LatentConditionedDDPM(in_chans=3, base_ch=32, cond_dim=jepa.embed_dim).to(device)
    sdf_tower = GeometricSDFTower(image_size=args.image_size, base_ch=32).to(device)
    print(f"[v9b-stage2] ddpm_params={sum(p.numel() for p in ddpm.parameters())/1e6:.1f}M  "
          f"sdf_params={sum(p.numel() for p in sdf_tower.parameters())/1e6:.1f}M",
          flush=True)

    optimizer = torch.optim.AdamW(
        list(ddpm.parameters()) + list(sdf_tower.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )

    ds = HealthyOnlyDataset(args.data_dir, args.image_size, args.extra_dirs,
                             cache_in_ram=args.cache_in_ram)
    print(f"[v9b-stage2] healthy_scans={len(ds)}", flush=True)
    extra = {"persistent_workers": True, "prefetch_factor": 4} if args.num_workers > 0 else {}
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True, **extra)

    # Atlas template SDF (synthetic for now; load_external_sdf for FreeSurfer later)
    sdf_template = synthetic_brain_sdf_template(args.image_size).to(device)
    sdf_template_b = sdf_template.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # Resume
    start_epoch, global_step = 1, 0
    if args.resume:
        rp = out / "last.pt" if args.resume.lower() == "auto" else Path(args.resume)
        if rp.exists():
            ck = torch.load(rp, map_location=device, weights_only=False)
            ddpm.load_state_dict(ck["ddpm_state_dict"])
            sdf_tower.load_state_dict(ck["sdf_state_dict"])
            optimizer.load_state_dict(ck["optimizer_state_dict"])
            start_epoch = ck["epoch"] + 1
            global_step = ck["global_step"]
            print(f"[v9b-stage2] resumed epoch={start_epoch}  step={global_step}", flush=True)

    amp_enabled = bool(args.amp) and device.type == "cuda" and torch.cuda.is_bf16_supported()
    log_path = out / "training.log"

    for epoch in range(start_epoch, args.epochs + 1):
        ddpm.train(); sdf_tower.train()
        t0 = time.time()
        ddpm_loss_sum = sdf_loss_sum = 0.0
        for x in loader:
            x = x.to(device, non_blocking=True)
            cond = jepa_global_latent(jepa, x)  # (B, D)
            optimizer.zero_grad(set_to_none=True)
            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    l_ddpm = ddpm.training_loss(x, cond)
                    l_sdf = sdf_tower.training_loss(x, sdf_template_b.expand(x.size(0), -1, -1, -1))
                    loss = args.lambda_ddpm * l_ddpm + args.lambda_sdf * l_sdf
            else:
                l_ddpm = ddpm.training_loss(x, cond)
                l_sdf = sdf_tower.training_loss(x, sdf_template_b.expand(x.size(0), -1, -1, -1))
                loss = args.lambda_ddpm * l_ddpm + args.lambda_sdf * l_sdf
            loss.backward()
            optimizer.step()
            ddpm_loss_sum += float(l_ddpm.item()); sdf_loss_sum += float(l_sdf.item())
            global_step += 1
            if args.checkpoint_every_steps > 0 and global_step % args.checkpoint_every_steps == 0:
                _atomic_save({
                    "ddpm_state_dict": ddpm.state_dict(),
                    "sdf_state_dict": sdf_tower.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch - 1, "global_step": global_step,
                    "args": vars(args),
                }, out / "last.pt")
        n = max(1, len(loader))
        line = (f"[epoch {epoch:03d}/{args.epochs}]  ddpm={ddpm_loss_sum/n:.4f}  "
                f"sdf={sdf_loss_sum/n:.6f}  ({time.time() - t0:.1f}s)")
        print(line, flush=True)
        with log_path.open("a") as f: f.write(line + "\n")
        _atomic_save({
            "ddpm_state_dict": ddpm.state_dict(),
            "sdf_state_dict": sdf_tower.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch, "global_step": global_step, "args": vars(args),
        }, out / "last.pt")
    print(f"[v9b-stage2] done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
