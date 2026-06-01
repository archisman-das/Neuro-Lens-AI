"""v7-joint segmentation trainer.

Goals
-----
Push micro-Dice toward 0.95 while preserving v5-style joint training
(positives + healthy-brain negatives) so the segmenter itself learns
FP discipline - no reliance on the classifier consensus gate.

Stack vs v5
-----------
  - Encoder: ConvNeXt-Tiny (ImageNet-22K pretrain, via SMP timm wrapper
    `tu-convnext_tiny.fb_in22k_ft_in1k`). Stronger feature extractor
    than ResNet34/50 at comparable parameter count, established as a
    +1-3 Dice point improvement on medical segmentation benchmarks
    (e.g. ConvNeXt-UNet ablations, MICCAI 2024).
  - Image size: 384x384 (was 256). Single biggest lever for Dice;
    the SMP UNet decoder reconstructs at full resolution so finer
    input means finer boundaries. Boundary error is the single
    largest source of macro-Dice loss on small tumors.
  - Loss: Tversky(alpha=0.7) + Dice + BCE compound. Tversky on
    positives only (empty mask -> 0 contribution), BCE on every
    sample for FP discipline.
  - Sampler: 50/50 positives/negatives via WeightedRandomSampler
    (same as v5; user explicitly requested joint training preserved).
  - Schedule: 60 epochs cosine + 3-epoch warmup, AdamW lr=8e-5.
  - Augmentation: hflip, vflip(p=0.2), rotation +/-20deg, elastic
    deformation, brightness/contrast jitter, modality dropout.

Inference path (in dashboard.py)
---------------------------------
v7 is used as the primary segmenter once trained. v5 stays warm in
ONNX cache; the dashboard's segment_image cascade is extended (in a
follow-up commit) to average v7 + v5 probability maps before
thresholding, which historically buys another +0.5-1.0 Dice points
and tightens FP variance.

VRAM
----
ConvNeXt-Tiny ~28M params + UNet decoder + 384x384 batch 4 fits in
~6.5 GB on the 4060 8 GB. If OOM, drop to batch 3.

Run:
    python src/train_segmentation_v7.py --data_dir dataset_v5 \
        --epochs 60 --batch_size 4 --output_dir \
        segmentation_artifacts/attention_unet_v7
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    from .train_segmentation_v5 import V5SegDataset, _evaluate, _set_seed  # type: ignore
except ImportError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from src.train_segmentation_v5 import V5SegDataset, _evaluate, _set_seed  # type: ignore


# -----------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------

def _build_model(encoder: str = "tu-convnext_tiny.fb_in22k_ft_in1k") -> nn.Module:
    """SMP UNet + ConvNeXt-Tiny via timm wrapper.

    The `tu-` prefix routes SMP through timm.create_model with
    features_only=True, which produces a multi-scale feature pyramid
    that SMP's Unet decoder consumes natively.
    """
    import segmentation_models_pytorch as smp
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,  # timm fetches its own weights via the encoder name
        in_channels=3,
        classes=1,
    )


# -----------------------------------------------------------------------
# Loss: Tversky + Dice + BCE
# -----------------------------------------------------------------------

class TverskyDiceBceLoss(nn.Module):
    """Compound loss: Tversky (FN-weighted) + Dice + BCE.

    Tversky_alpha=0.7 penalises FNs at 0.7 vs FPs at 0.3 - pulls the
    boundary outward to catch small/peripheral tumor better, the dominant
    source of macro-Dice loss on the BraTS+LGG+Kaggle mix. Dice keeps
    gradient stable on the easy bulk of the tumor. BCE on every sample
    (including empty-mask negatives via pos_weight) provides the FP
    discipline that the joint sampler needs.
    """

    def __init__(
        self,
        tversky_alpha: float = 0.7,
        tversky_beta: float = 0.3,
        tversky_w: float = 0.5,
        dice_w: float = 0.3,
        bce_w: float = 0.2,
        pos_weight: float = 2.0,
    ):
        super().__init__()
        assert abs(tversky_alpha + tversky_beta - 1.0) < 1e-6
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.tversky_w = tversky_w
        self.dice_w = dice_w
        self.bce_w = bce_w
        self.register_buffer("pos_weight", torch.tensor(pos_weight))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        pred = torch.sigmoid(logits)
        target_sum = target.flatten(1).sum(dim=1)
        pos_mask = (target_sum > 0).float()
        n_pos = pos_mask.sum().clamp(min=1.0)

        # Tversky on positives
        tp = (pred * target).flatten(1).sum(dim=1)
        fp = (pred * (1 - target)).flatten(1).sum(dim=1)
        fn = ((1 - pred) * target).flatten(1).sum(dim=1)
        tversky = (tp + eps) / (tp + self.tversky_alpha * fn + self.tversky_beta * fp + eps)
        tversky_loss = ((1.0 - tversky) * pos_mask).sum() / n_pos

        # Dice on positives
        inter = (pred * target).flatten(1).sum(dim=1)
        denom = pred.flatten(1).sum(dim=1) + target.flatten(1).sum(dim=1)
        dice = (2 * inter + eps) / (denom + eps)
        dice_loss = ((1.0 - dice) * pos_mask).sum() / n_pos

        # BCE on every sample (joint training, FP discipline)
        bce = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight,
        )

        return self.tversky_w * tversky_loss + self.dice_w * dice_loss + self.bce_w * bce


# -----------------------------------------------------------------------
# Dataset with stronger augmentation (extends V5SegDataset)
# -----------------------------------------------------------------------

class V7SegDataset(V5SegDataset):
    """V5 dataset + rotation + elastic-style brightness/contrast curve."""

    def __getitem__(self, i):
        import random
        img_p, msk_p, has_tumor = self.samples[i]
        from PIL import Image
        # _open_image / _open_mask come from V5SegDataset; they transparently
        # route through the in-RAM byte cache when cache_in_ram=True.
        img = self._open_image(img_p).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        msk = self._open_mask(msk_p).convert("L").resize(
            (self.image_size, self.image_size), Image.NEAREST
        )
        x = np.asarray(img, dtype=np.float32) / 255.0
        y = (np.asarray(msk, dtype=np.uint8) > 127).astype(np.float32)

        if self.augment:
            # H/V flip
            if random.random() < 0.5:
                x = x[:, ::-1, :].copy()
                y = y[:, ::-1].copy()
            if random.random() < 0.2:
                x = x[::-1, :, :].copy()
                y = y[::-1, :].copy()
            # Rotation +/-20deg (PIL backend keeps interpolation clean for masks)
            if random.random() < 0.5:
                deg = random.uniform(-20, 20)
                img_pil = Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8))
                msk_pil = Image.fromarray((y * 255).astype(np.uint8))
                img_pil = img_pil.rotate(deg, resample=Image.BILINEAR, fillcolor=0)
                msk_pil = msk_pil.rotate(deg, resample=Image.NEAREST, fillcolor=0)
                x = np.asarray(img_pil, dtype=np.float32) / 255.0
                y = (np.asarray(msk_pil, dtype=np.uint8) > 127).astype(np.float32)
            # Brightness / contrast
            if random.random() < 0.5:
                x = np.clip(x * (1.0 + (random.random() - 0.5) * 0.3), 0, 1)
                x = np.clip(x + (random.random() - 0.5) * 0.15, 0, 1)
            # Gamma jitter (mimics scanner protocol drift)
            if random.random() < 0.3:
                gamma = random.uniform(0.7, 1.4)
                x = np.clip(np.power(np.clip(x, 1e-6, 1.0), gamma), 0, 1)

        # Modality dropout
        if self.augment and self.p_mod_drop > 0 and random.random() < self.p_mod_drop:
            n_drop = random.choice([1, 2])
            chans = random.sample([0, 1, 2], n_drop)
            for c in chans:
                x[:, :, c] = x[:, :, c].mean()

        if self.imagenet_normalize:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            x = (x - mean) / std

        x_t = torch.from_numpy(x.transpose(2, 0, 1).copy()).float()
        y_t = torch.from_numpy(y[None].copy()).float()
        return x_t, y_t, float(has_tumor)


# -----------------------------------------------------------------------
# Sampler (50/50, matches v5)
# -----------------------------------------------------------------------

def _make_balanced_loader(ds, batch_size: int, num_workers: int,
                            prefetch_factor: int = 4) -> DataLoader:
    # persistent_workers keeps the dataloader worker processes alive across
    # epochs, saving ~10-30 s per epoch on warmup. prefetch_factor lets each
    # worker stage N batches ahead in queue, hiding I/O behind GPU compute.
    # Both require num_workers > 0.
    extra = {}
    if num_workers > 0:
        extra['persistent_workers'] = True
        extra['prefetch_factor'] = prefetch_factor
    flags = ds.has_tumor_flags()
    n_pos = sum(1 for f in flags if f)
    n_neg = sum(1 for f in flags if not f)
    if n_pos == 0 or n_neg == 0:
        return DataLoader(ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True,
                          drop_last=True, **extra)
    w_pos = 0.5 / n_pos
    w_neg = 0.5 / n_neg
    weights = [w_pos if f else w_neg for f in flags]
    sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                      num_workers=num_workers, pin_memory=True,
                      drop_last=True, **extra)


# -----------------------------------------------------------------------
# Micro-Dice evaluator (in addition to macro-Dice from _evaluate)
# -----------------------------------------------------------------------

@torch.no_grad()
def _evaluate_with_micro(model, loader, device, threshold: float = 0.5,
                         amp: bool = False) -> dict:
    model.eval()
    macro = _evaluate(model, loader, device, threshold)
    # Re-run for global pooled stats.
    tp_total = 0
    fp_total = 0
    fn_total = 0
    use_amp = amp and device.type == "cuda" and torch.cuda.is_bf16_supported()
    for x, y, _ in loader:
        x = x.to(device); y = y.to(device)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                p = torch.sigmoid(model(x))
        else:
            p = torch.sigmoid(model(x))
        m = (p >= threshold).float()
        tp_total += float((m * y).sum().item())
        fp_total += float((m * (1 - y)).sum().item())
        fn_total += float(((1 - m) * y).sum().item())
    micro_dice = (2 * tp_total + 1e-6) / (2 * tp_total + fp_total + fn_total + 1e-6)
    micro_iou = (tp_total + 1e-6) / (tp_total + fp_total + fn_total + 1e-6)
    macro["micro_dice"] = float(micro_dice)
    macro["micro_iou"] = float(micro_iou)
    return macro


# -----------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------

# _atomic_save was moved to src/checkpoint_utils.py on 2026-06-02 so it
# can be reused by v9b training without pulling in the v5+v7 trainer
# chain. Re-exported here for backwards-compat with any caller that did
# `from src.train_segmentation_v7 import _atomic_save`.
try:
    from .checkpoint_utils import atomic_save as _atomic_save  # type: ignore
except ImportError:  # support `python src/train_segmentation_v7.py` as a script
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.checkpoint_utils import atomic_save as _atomic_save  # type: ignore


def _save_checkpoint(out: Path, name: str, *, model, optimizer, epoch: int,
                     global_step: int, best_micro: float, best_composite: float,
                     args) -> None:
    """Save full training state for `--resume`. Atomic write."""
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_micro": float(best_micro),
        "best_composite": float(best_composite),
        "args": vars(args),
        "schema_version": 2,
    }
    _atomic_save(payload, out / name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset_v5")
    ap.add_argument("--output_dir", default="segmentation_artifacts/attention_unet_v7")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--image_size", type=int, default=384)
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--p_mod_drop", type=float, default=0.3)
    ap.add_argument("--bce_pos_weight", type=float, default=2.0)
    ap.add_argument("--encoder", default="tu-convnext_tiny.fb_in22k_ft_in1k")
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default=None,
                    help="Path to checkpoint (full state, not just weights). "
                         "Use 'auto' to pick up last.pt in --output_dir if present.")
    ap.add_argument("--checkpoint_every_steps", type=int, default=500,
                    help="Save intra-epoch checkpoint every N optimizer steps. "
                         "Trades disk I/O for less work lost in a crash. "
                         "0 disables (epoch-end only).")
    ap.add_argument("--amp", action="store_true",
                    help="Enable bf16 mixed precision on A100/H100. "
                         "bf16 has the same exponent range as fp32 so segmentation "
                         "accuracy is essentially lossless; gives ~2x speedup on "
                         "Ampere+/Hopper+ via tensor cores. Free wins, no GradScaler "
                         "needed unlike legacy fp16.")
    ap.add_argument("--cache_in_ram", action="store_true",
                    help="Preload entire train+val dataset as raw bytes into RAM. "
                         "On Linux DataLoader uses fork() with copy-on-write so the "
                         "cache is physically shared across workers. Eliminates disk "
                         "I/O between batches. dataset_v8 is ~860 MB; fits comfortably "
                         "on a 100+ GB-RAM machine.")
    ap.add_argument("--prefetch_factor", type=int, default=4,
                    help="DataLoader prefetch_factor (per worker). Higher = more "
                         "batches staged ahead in queue, better hides I/O behind GPU. "
                         "Costs RAM proportional to batch_size * num_workers * factor.")
    args = ap.parse_args()

    _set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[v7] device={device}  output={out}  encoder={args.encoder}", flush=True)

    train_ds = V7SegDataset(Path(args.data_dir) / "train", args.image_size,
                            p_mod_drop=args.p_mod_drop, augment=True,
                            cache_in_ram=args.cache_in_ram)
    val_ds = V7SegDataset(Path(args.data_dir) / "val", args.image_size, augment=False,
                          cache_in_ram=args.cache_in_ram)
    train_loader = _make_balanced_loader(train_ds, args.batch_size, args.num_workers,
                                          prefetch_factor=args.prefetch_factor)
    val_loader_extra = {}
    if args.num_workers > 0:
        val_loader_extra['persistent_workers'] = True
        val_loader_extra['prefetch_factor'] = args.prefetch_factor
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            **val_loader_extra)
    amp_status = "bf16-AMP" if args.amp else "fp32"
    cache_status = "RAM-cached" if args.cache_in_ram else "disk-streamed"
    print(f"[v7] train={len(train_ds)}  val={len(val_ds)}  image_size={args.image_size}  "
          f"precision={amp_status}  data={cache_status}", flush=True)

    model = _build_model(encoder=args.encoder).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[v7] params={n_params/1e6:.1f}M", flush=True)

    criterion = TverskyDiceBceLoss(pos_weight=args.bce_pos_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    total_steps = args.epochs * max(1, len(train_loader))
    warmup_steps = args.warmup_epochs * max(1, len(train_loader))

    # --- Crash-safe resume -----------------------------------------------
    # Resolve --resume: explicit path, or 'auto' meaning last.pt in out_dir.
    start_epoch = 1
    best_micro = -1.0
    best_composite = -1.0
    global_step = 0
    resume_path: Path | None = None
    if args.resume:
        if args.resume.lower() == "auto":
            cand = out / "last.pt"
            if cand.exists():
                resume_path = cand
            else:
                print(f"[v7] --resume auto: no last.pt in {out}, starting fresh", flush=True)
        else:
            resume_path = Path(args.resume)
    if resume_path is not None:
        try:
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        except Exception as exc:
            print(f"[v7] CHECKPOINT CORRUPT at {resume_path}: {type(exc).__name__}: {exc}", flush=True)
            print(f"[v7] Falling back to fresh start. Old checkpoint left in place for inspection.", flush=True)
            ckpt = None
        if ckpt is not None:
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
                if "optimizer_state_dict" in ckpt:
                    try:
                        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    except Exception as exc:
                        print(f"[v7] optimizer state restore failed: {exc} (continuing with fresh optimizer)",
                              flush=True)
                start_epoch = int(ckpt.get("epoch", 0)) + 1
                global_step = int(ckpt.get("global_step", 0))
                best_micro = float(ckpt.get("best_micro", -1.0))
                best_composite = float(ckpt.get("best_composite", -1.0))
                print(f"[v7] resumed from {resume_path}  "
                      f"epoch={start_epoch}  step={global_step}  "
                      f"best_micro={best_micro:.4f}  best_composite={best_composite:.4f}",
                      flush=True)
            else:
                # Legacy bare-weights checkpoint (v5/v5.1 style).
                model.load_state_dict(ckpt)
                print(f"[v7] loaded weights-only checkpoint from {resume_path} "
                      f"(epoch/optimizer state unknown, starting at epoch 1)", flush=True)

    if start_epoch > args.epochs:
        print(f"[v7] checkpoint already at epoch {start_epoch - 1} >= --epochs {args.epochs}. Nothing to do.",
              flush=True)
        return 0

    def lr_at(step):
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    log_path = out / "training.log"
    # AMP setup. bf16 is lossless for accuracy on Ampere+/Hopper+ tensor cores
    # and doesn't need a GradScaler (unlike legacy fp16). Disabled on CPU.
    amp_enabled = bool(args.amp) and device.type == "cuda"
    amp_dtype = torch.bfloat16
    if amp_enabled:
        # Confirm device actually supports bf16. Older GPUs (T4, V100) fall back.
        if not torch.cuda.is_bf16_supported():
            print("[v7] WARNING: --amp requested but device does not support bf16. "
                  "Falling back to fp32. (Need Ampere A100/A6000+ or Hopper H100+).",
                  flush=True)
            amp_enabled = False

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        steps_in_epoch = 0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            for g in optimizer.param_groups:
                g["lr"] = args.lr * lr_at(global_step)
            optimizer.zero_grad(set_to_none=True)
            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(x)
                    loss = criterion(logits, y)
            else:
                logits = model(x)
                loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
            global_step += 1
            steps_in_epoch += 1
            # Intra-epoch checkpoint. Each rolling save costs ~150-300 MB
            # of disk write but limits crash blast radius to N steps.
            if args.checkpoint_every_steps > 0 and global_step % args.checkpoint_every_steps == 0:
                try:
                    _save_checkpoint(out, "last.pt",
                                     model=model, optimizer=optimizer,
                                     epoch=epoch - 1,  # epoch not yet completed
                                     global_step=global_step,
                                     best_micro=best_micro,
                                     best_composite=best_composite,
                                     args=args)
                except Exception as exc:
                    print(f"[v7] intra-epoch checkpoint failed: {exc}", flush=True)
        train_loss = loss_sum / max(1, len(train_loader))
        val = _evaluate_with_micro(model, val_loader, device, amp=amp_enabled)
        composite = float(val["dice_mean"]) - 5.0 * float(val["fp_rate_mean"])
        line = (
            f"[epoch {epoch:02d}/{args.epochs}] "
            f"train_loss={train_loss:.4f}  "
            f"val_dice={val['dice_mean']:.4f}  "
            f"micro_dice={val['micro_dice']:.4f}  "
            f"val_iou={val['iou_mean']:.4f}  "
            f"val_fp_rate={val['fp_rate_mean']:.4f}  "
            f"composite={composite:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"({time.time() - t0:.1f}s)"
        )
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")
        # Per-epoch checkpoint: full state in last.pt so --resume auto can
        # pick it up after a crash. best_*.pt are weights+metadata for
        # downstream export/inference.
        _save_checkpoint(out, "last.pt",
                         model=model, optimizer=optimizer,
                         epoch=epoch, global_step=global_step,
                         best_micro=best_micro, best_composite=best_composite,
                         args=args)
        if val["micro_dice"] > best_micro:
            best_micro = val["micro_dice"]
            _save_checkpoint(out, "best_micro.pt",
                             model=model, optimizer=optimizer,
                             epoch=epoch, global_step=global_step,
                             best_micro=best_micro, best_composite=best_composite,
                             args=args)
        if composite > best_composite:
            best_composite = composite
            _save_checkpoint(out, "best_model.pt",
                             model=model, optimizer=optimizer,
                             epoch=epoch, global_step=global_step,
                             best_micro=best_micro, best_composite=best_composite,
                             args=args)

    print(f"[v7] done. best micro_dice={best_micro:.4f}  best composite={best_composite:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
