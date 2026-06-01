"""v9b Stage-1: I-JEPA self-supervised pretrain on healthy brain MRI.

Uses healthy-only scans (Kaggle no_tumor split from dataset_v8 + any
additional healthy datasets you mount). No tumor labels needed.

Crash-safe checkpointing + AMP + RAM cache + --resume auto.
"""
from __future__ import annotations
import argparse, math, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

try:
    from .checkpoint_utils import atomic_save as _atomic_save  # type: ignore
    from .research.jepa import IJEPAModel, make_jepa_masks  # type: ignore
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.checkpoint_utils import atomic_save as _atomic_save  # type: ignore
    from src.research.jepa import IJEPAModel, make_jepa_masks  # type: ignore


# --------------- Healthy-only dataset ---------------

class HealthyOnlyDataset(Dataset):
    """Loads ONLY healthy scans (empty masks) from a dataset_v8-like layout.

    Walks {root}/{train,val}/{images,masks}/ and yields image paths whose
    corresponding mask is all-zero. For dataset_v8 train this gives ~1.5K
    Kaggle no_tumor scans -- small but enough for an initial I-JEPA pretrain.
    To pretrain at scale, add IXI/OASIS/HCP volumes by passing extra dirs.
    """
    def __init__(self, root: Path, image_size: int = 256, extra_image_dirs: list = None,
                 cache_in_ram: bool = False):
        self.image_size = image_size
        self.samples = []
        root = Path(root)
        for split in ("train", "val"):
            img_dir = root / split / "images"
            msk_dir = root / split / "masks"
            if not img_dir.exists(): continue
            for ip in sorted(img_dir.iterdir()):
                mp = msk_dir / ip.name
                if not mp.exists(): continue
                # Cheap healthy-check via thumbnail
                try:
                    m = np.array(Image.open(mp).convert('L').resize((32, 32), Image.NEAREST))
                    if not (m > 127).any():
                        self.samples.append(ip)
                except Exception:
                    pass
        for d in (extra_image_dirs or []):
            d = Path(d)
            if d.exists():
                for ip in sorted(d.iterdir()):
                    if ip.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'):
                        self.samples.append(ip)
        # Optional RAM cache (raw bytes; PIL decodes on demand).
        self._cache = None
        if cache_in_ram:
            t0 = time.time()
            self._cache = {str(p): p.read_bytes() for p in self.samples}
            mb = sum(len(v) for v in self._cache.values()) / 1e6
            print(f"[v9b-stage1] cached {mb:.0f} MB in {time.time()-t0:.0f}s", flush=True)

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        p = self.samples[i]
        if self._cache is not None:
            import io
            img = Image.open(io.BytesIO(self._cache[str(p)]))
        else:
            img = Image.open(p)
        img = img.convert('RGB').resize((self.image_size, self.image_size), Image.BILINEAR)
        x = np.asarray(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
        return torch.from_numpy(x.transpose(2, 0, 1).copy()).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset_v8")
    ap.add_argument("--extra_dirs", nargs="*", default=[],
                    help="Optional extra healthy image directories (IXI, OASIS, ...)")
    ap.add_argument("--output_dir", default="segmentation_artifacts/v9b_jepa_pretrain")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--embed_dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--predictor_dim", type=int, default=192)
    ap.add_argument("--predictor_depth", type=int, default=6)
    ap.add_argument("--ema_momentum", type=float, default=0.996)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.04)
    ap.add_argument("--warmup_epochs", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--cache_in_ram", action="store_true")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--checkpoint_every_steps", type=int, default=500)
    ap.add_argument("--n_target_blocks", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    ds = HealthyOnlyDataset(args.data_dir, args.image_size, args.extra_dirs,
                            cache_in_ram=args.cache_in_ram)
    print(f"[v9b-stage1] healthy_scans={len(ds)}  image_size={args.image_size}  "
          f"patch_size={args.patch_size}", flush=True)
    assert len(ds) >= 100, "need at least 100 healthy scans for JEPA pretrain"
    extra = {"persistent_workers": True, "prefetch_factor": 4} if args.num_workers > 0 else {}
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True, **extra)

    model = IJEPAModel(
        image_size=args.image_size, patch_size=args.patch_size,
        embed_dim=args.embed_dim, depth=args.depth, heads=args.heads,
        predictor_dim=args.predictor_dim, predictor_depth=args.predictor_depth,
        ema_momentum=args.ema_momentum,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[v9b-stage1] params={n_params/1e6:.1f}M", flush=True)

    # Only context_encoder + predictor get gradients; target_encoder is EMA.
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * max(1, len(loader))
    warmup_steps = args.warmup_epochs * max(1, len(loader))
    def lr_at(step):
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    # Resume
    start_epoch, global_step = 1, 0
    resume_path = None
    if args.resume:
        resume_path = out / "last.pt" if args.resume.lower() == "auto" else Path(args.resume)
        if not resume_path.exists():
            resume_path = None
            print(f"[v9b-stage1] --resume but no checkpoint, fresh start", flush=True)
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        print(f"[v9b-stage1] resumed from {resume_path}  epoch={start_epoch}  step={global_step}",
              flush=True)

    amp_enabled = bool(args.amp) and device.type == "cuda" and torch.cuda.is_bf16_supported()
    log_path = out / "training.log"

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        for x in loader:
            x = x.to(device, non_blocking=True)
            for g in optimizer.param_groups:
                g["lr"] = args.lr * lr_at(global_step)
            masks = make_jepa_masks(model.grid_size, x.size(0),
                                     n_target=args.n_target_blocks, device=device)
            optimizer.zero_grad(set_to_none=True)
            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out_dict = model(x, masks)
                    loss = out_dict["loss"]
            else:
                out_dict = model(x, masks)
                loss = out_dict["loss"]
            loss.backward()
            optimizer.step()
            model.ema_update()
            loss_sum += float(loss.item())
            global_step += 1
            if args.checkpoint_every_steps > 0 and global_step % args.checkpoint_every_steps == 0:
                _atomic_save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch - 1, "global_step": global_step,
                    "args": vars(args),
                }, out / "last.pt")
        avg = loss_sum / max(1, len(loader))
        line = (f"[epoch {epoch:03d}/{args.epochs}]  loss={avg:.4f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                f"({time.time() - t0:.1f}s)")
        print(line, flush=True)
        with log_path.open("a") as f: f.write(line + "\n")
        _atomic_save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch, "global_step": global_step, "args": vars(args),
        }, out / "last.pt")
        if epoch == 1 or epoch % 10 == 0:
            _atomic_save({"model_state_dict": model.state_dict(), "epoch": epoch,
                          "args": vars(args)},
                         out / f"jepa_epoch_{epoch:03d}.pt")
    print(f"[v9b-stage1] done. final_loss={avg:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
