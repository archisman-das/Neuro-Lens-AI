"""v9 brain-2D trainer: integrates all v9 research heads with crash-safe training.

Architecture summary (see src/research/_v10_universal_hyperbolic/v10_model.py for details)
---------------------------------------------------------------
  Geometric prior (SDF) -> SMP UNet encoder (ConvNeXt-Tiny, 4-channel)
    -> bottleneck -> latent_dim=256
    -> hyperbolic projection (learnable curvature)
    -> causal SCM head: (anatomy, tumor, scanner) + NOTEARS DAG
    -> recompose (anatomy + tumor) -> SMP UNet decoder
    -> mask logits

  AND on a parallel branch:
    -> SCM with z_tumor=0 -> counterfactual healthy decoder
    -> counterfactual image + residual

Multi-task loss:
  L = L_seg (Tversky + Dice + BCE)         <- segmentation
    + lambda_o * (L_o_at + L_o_as + L_o_ts) <- SCM orthogonality
    + lambda_dag * L_dag                    <- NOTEARS acyclicity
    + lambda_forbid * L_dag_forbidden       <- biological priors
    + lambda_cf * L_cf_recon                <- counterfactual reconstruction
    + lambda_hyp * L_hyp_reg                <- hyperbolic embedding reg

All loss weights default to safe values (small SCM regularizers) so the
segmentation task dominates initially. Trained with the same crash-safe
checkpointing + AMP + RAM cache infrastructure as v7.

Designed to run on the same dataset_v8 as v7/v8. No multi-organ scope.
Multi-organ extension is v10.

Usage (Colab or local):
  python src/train_segmentation_v10.py --data_dir dataset_v8 \
    --output_dir segmentation_artifacts/attention_unet_v9 \
    --epochs 100 --batch_size 32 --image_size 384 \
    --amp --cache_in_ram --resume auto
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    from .train_segmentation_v5 import V5SegDataset, _evaluate, _set_seed  # type: ignore
    from .train_segmentation_v7 import V7SegDataset, TverskyDiceBceLoss, _evaluate_with_micro, \
        _save_checkpoint, _atomic_save  # type: ignore
    from .research._v10_universal_hyperbolic.v10_model import V10Model  # type: ignore
except ImportError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from src.train_segmentation_v5 import V5SegDataset, _evaluate, _set_seed  # type: ignore
    from src.train_segmentation_v7 import V7SegDataset, TverskyDiceBceLoss, _evaluate_with_micro, \
        _save_checkpoint, _atomic_save  # type: ignore
    from src.research._v10_universal_hyperbolic.v10_model import V10Model  # type: ignore


# -----------------------------------------------------------------------
# Balanced sampler (matches v7)
# -----------------------------------------------------------------------

def _make_balanced_loader(ds, batch_size: int, num_workers: int,
                          prefetch_factor: int = 4) -> DataLoader:
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
# v9 evaluation (extends v7 evaluator with SCM monitoring)
# -----------------------------------------------------------------------

@torch.no_grad()
def _evaluate_v9(model, loader, device, threshold: float = 0.5, amp: bool = False) -> dict:
    """Evaluate v9 model: standard seg metrics + SCM disentanglement health."""
    model.eval()
    use_amp = amp and device.type == "cuda" and torch.cuda.is_bf16_supported()
    dices, ious, fp_rates = [], [], []
    tp_total = fp_total = fn_total = 0
    ortho_at_sum = ortho_as_sum = ortho_ts_sum = 0.0
    dag_sum = 0.0
    cf_recon_loss_sum = 0.0
    n_batches = 0
    n_pos = n_neg = 0

    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x, return_counterfactual=True)
        else:
            out = model(x, return_counterfactual=True)
        logits = out["mask_logits"]
        p = torch.sigmoid(logits)
        m = (p >= threshold).float()
        # Per-sample seg metrics
        for i in range(x.size(0)):
            yi, mi = y[i], m[i]
            if yi.sum() > 0:
                inter = (mi * yi).sum().item()
                pred_sum = mi.sum().item()
                tgt_sum = yi.sum().item()
                dices.append((2 * inter + 1e-6) / (pred_sum + tgt_sum + 1e-6))
                ious.append((inter + 1e-6) / (pred_sum + tgt_sum - inter + 1e-6))
                n_pos += 1
            else:
                fp_rates.append(mi.mean().item())
                n_neg += 1
            tp_total += float((mi * yi).sum().item())
            fp_total += float((mi * (1 - yi)).sum().item())
            fn_total += float(((1 - mi) * yi).sum().item())

        # SCM metrics (averaged across batch)
        aux = out["aux_losses"]
        ortho_at_sum += float(aux["ortho_at"].item())
        ortho_as_sum += float(aux["ortho_as"].item())
        ortho_ts_sum += float(aux["ortho_ts"].item())
        dag_sum += float(aux["dag"].item())

        # Counterfactual reconstruction loss (only meaningful for empty-mask negatives)
        if out["x_counterfactual"] is not None:
            # For healthy scans (y empty), x_cf should equal x.
            healthy_mask = (y.sum(dim=(1, 2, 3)) == 0).float()
            if healthy_mask.any():
                per_sample_loss = (out["x_counterfactual"] - x).abs().mean(dim=(1, 2, 3))
                cf_recon_loss_sum += float((per_sample_loss * healthy_mask).sum() / healthy_mask.sum())

        n_batches += 1

    import numpy as _np
    micro_dice = (2 * tp_total + 1e-6) / (2 * tp_total + fp_total + fn_total + 1e-6)
    return {
        "n_positive": n_pos,
        "n_negative": n_neg,
        "dice_mean": float(_np.mean(dices)) if dices else 0.0,
        "iou_mean": float(_np.mean(ious)) if ious else 0.0,
        "micro_dice": float(micro_dice),
        "fp_rate_mean": float(_np.mean(fp_rates)) if fp_rates else 0.0,
        "fp_rate_p95": float(_np.percentile(fp_rates, 95)) if fp_rates else 0.0,
        "ortho_at": ortho_at_sum / max(1, n_batches),
        "ortho_as": ortho_as_sum / max(1, n_batches),
        "ortho_ts": ortho_ts_sum / max(1, n_batches),
        "dag_h": dag_sum / max(1, n_batches),
        "cf_recon_loss": cf_recon_loss_sum / max(1, n_batches),
    }


# -----------------------------------------------------------------------
# v9 save (full state, atomic)
# -----------------------------------------------------------------------

def _save_v9_checkpoint(out: Path, name: str, *, model, optimizer, epoch: int,
                        global_step: int, best_micro: float, best_composite: float,
                        args) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_micro": float(best_micro),
        "best_composite": float(best_composite),
        "args": vars(args),
        "schema_version": "v9_brain2d_1",
    }
    _atomic_save(payload, out / name)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset_v8")
    ap.add_argument("--output_dir", default="segmentation_artifacts/attention_unet_v9")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--image_size", type=int, default=384)
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--p_mod_drop", type=float, default=0.3)
    ap.add_argument("--bce_pos_weight", type=float, default=2.0)
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default=None,
                    help="Path to checkpoint or 'auto' for last.pt in output_dir.")
    ap.add_argument("--checkpoint_every_steps", type=int, default=500)
    ap.add_argument("--amp", action="store_true",
                    help="bf16 mixed precision on A100/H100.")
    ap.add_argument("--cache_in_ram", action="store_true",
                    help="Preload dataset bytes in RAM (~900 MB for dataset_v8).")

    # v9-specific model hyperparams
    ap.add_argument("--latent_dim", type=int, default=256)
    ap.add_argument("--anatomy_dim", type=int, default=128)
    ap.add_argument("--tumor_dim", type=int, default=64)
    ap.add_argument("--scanner_dim", type=int, default=32)
    ap.add_argument("--no_counterfactual", action="store_true",
                    help="Disable counterfactual healthy decoder (lighter).")
    ap.add_argument("--no_geometric_prior", action="store_true",
                    help="Disable SDF geometric prior (3-channel input instead of 4).")
    ap.add_argument("--hyperbolic_curvature_init", type=float, default=1.0)

    # v9-specific loss weights (multi-task balance)
    ap.add_argument("--lambda_ortho", type=float, default=0.05,
                    help="Weight on SCM orthogonality losses (anatomy-tumor, anatomy-scanner, tumor-scanner).")
    ap.add_argument("--lambda_dag", type=float, default=0.01,
                    help="Weight on NOTEARS DAG-ness loss.")
    ap.add_argument("--lambda_forbidden", type=float, default=0.05,
                    help="Weight on forbidden-edge penalty (scanner->anatomy, tumor->anatomy).")
    ap.add_argument("--lambda_cf", type=float, default=0.10,
                    help="Weight on counterfactual reconstruction loss.")
    args = ap.parse_args()

    _set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[v9] device={device}  output={out}", flush=True)

    train_ds = V7SegDataset(Path(args.data_dir) / "train", args.image_size,
                            p_mod_drop=args.p_mod_drop, augment=True,
                            cache_in_ram=args.cache_in_ram)
    val_ds = V7SegDataset(Path(args.data_dir) / "val", args.image_size, augment=False,
                          cache_in_ram=args.cache_in_ram)
    train_loader = _make_balanced_loader(train_ds, args.batch_size, args.num_workers,
                                          prefetch_factor=args.prefetch_factor)
    val_extra = {}
    if args.num_workers > 0:
        val_extra["persistent_workers"] = True
        val_extra["prefetch_factor"] = args.prefetch_factor
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, **val_extra)
    amp_status = "bf16-AMP" if args.amp else "fp32"
    cache_status = "RAM-cached" if args.cache_in_ram else "disk-streamed"
    print(f"[v9] train={len(train_ds)}  val={len(val_ds)}  image_size={args.image_size}  "
          f"precision={amp_status}  data={cache_status}", flush=True)

    # Build model with all v9 heads
    model = V10Model(
        image_size=args.image_size,
        latent_dim=args.latent_dim,
        anatomy_dim=args.anatomy_dim,
        tumor_dim=args.tumor_dim,
        scanner_dim=args.scanner_dim,
        use_counterfactual=not args.no_counterfactual,
        use_geometric_prior=not args.no_geometric_prior,
        hyperbolic_curvature_init=args.hyperbolic_curvature_init,
    ).to(device)
    n_params = model.num_parameters()
    print(f"[v9] params={n_params/1e6:.1f}M  "
          f"(counterfactual={'on' if not args.no_counterfactual else 'off'}, "
          f"geometric_prior={'on' if not args.no_geometric_prior else 'off'})", flush=True)

    criterion = TverskyDiceBceLoss(pos_weight=args.bce_pos_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    total_steps = args.epochs * max(1, len(train_loader))
    warmup_steps = args.warmup_epochs * max(1, len(train_loader))

    # ---- Resume support ----
    start_epoch = 1
    best_micro = -1.0
    best_composite = -1.0
    global_step = 0
    resume_path = None
    if args.resume:
        if args.resume.lower() == "auto":
            cand = out / "last.pt"
            if cand.exists():
                resume_path = cand
            else:
                print(f"[v9] --resume auto: no last.pt in {out}, starting fresh", flush=True)
        else:
            resume_path = Path(args.resume)
    if resume_path is not None:
        try:
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        except Exception as exc:
            print(f"[v9] CHECKPOINT CORRUPT at {resume_path}: {exc}", flush=True)
            ckpt = None
        if ckpt is not None and isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            try:
                model.load_state_dict(ckpt["model_state_dict"])
                if "optimizer_state_dict" in ckpt:
                    try:
                        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    except Exception as exc:
                        print(f"[v9] optimizer state restore failed: {exc} (continuing fresh)",
                              flush=True)
                start_epoch = int(ckpt.get("epoch", 0)) + 1
                global_step = int(ckpt.get("global_step", 0))
                best_micro = float(ckpt.get("best_micro", -1.0))
                best_composite = float(ckpt.get("best_composite", -1.0))
                print(f"[v9] resumed from {resume_path}  epoch={start_epoch}  "
                      f"step={global_step}  best_micro={best_micro:.4f}", flush=True)
            except RuntimeError as exc:
                print(f"[v9] model state mismatch (probably arch change): {exc}", flush=True)
                print(f"[v9] starting fresh from epoch 1", flush=True)

    if start_epoch > args.epochs:
        print(f"[v9] checkpoint already past target epoch {args.epochs}. Nothing to do.",
              flush=True)
        return 0

    def lr_at(step):
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    log_path = out / "training.log"
    amp_enabled = bool(args.amp) and device.type == "cuda"
    if amp_enabled and not torch.cuda.is_bf16_supported():
        print("[v9] WARNING: --amp requested but no bf16 support, falling back to fp32",
              flush=True)
        amp_enabled = False

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        seg_loss_sum = 0.0
        scm_loss_sum = 0.0
        cf_loss_sum = 0.0
        steps_in_epoch = 0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            for g in optimizer.param_groups:
                g["lr"] = args.lr * lr_at(global_step)
            optimizer.zero_grad(set_to_none=True)

            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out_dict = model(x, return_counterfactual=not args.no_counterfactual)
                    logits = out_dict["mask_logits"]
                    seg_loss = criterion(logits, y)
                    aux = out_dict["aux_losses"]
                    scm_loss = (args.lambda_ortho * (aux["ortho_at"] + aux["ortho_as"] + aux["ortho_ts"])
                                + args.lambda_dag * aux["dag"]
                                + args.lambda_forbidden * aux["dag_forbidden"])
                    cf_loss = torch.tensor(0.0, device=device)
                    if out_dict["x_counterfactual"] is not None and args.lambda_cf > 0:
                        cf_loss = model.cf_decoder.reconstruction_loss(
                            x, out_dict["x_counterfactual"], y,
                            lambda_outside=1.0, lambda_inside=0.5,
                        )
                    loss = seg_loss + scm_loss + args.lambda_cf * cf_loss
            else:
                out_dict = model(x, return_counterfactual=not args.no_counterfactual)
                logits = out_dict["mask_logits"]
                seg_loss = criterion(logits, y)
                aux = out_dict["aux_losses"]
                scm_loss = (args.lambda_ortho * (aux["ortho_at"] + aux["ortho_as"] + aux["ortho_ts"])
                            + args.lambda_dag * aux["dag"]
                            + args.lambda_forbidden * aux["dag_forbidden"])
                cf_loss = torch.tensor(0.0, device=device)
                if out_dict["x_counterfactual"] is not None and args.lambda_cf > 0:
                    cf_loss = model.cf_decoder.reconstruction_loss(
                        x, out_dict["x_counterfactual"], y,
                        lambda_outside=1.0, lambda_inside=0.5,
                    )
                loss = seg_loss + scm_loss + args.lambda_cf * cf_loss

            loss.backward()
            optimizer.step()

            seg_loss_sum += float(seg_loss.item())
            scm_loss_sum += float(scm_loss.item())
            cf_loss_sum += float(cf_loss.item()) if torch.is_tensor(cf_loss) else 0.0
            global_step += 1
            steps_in_epoch += 1

            if args.checkpoint_every_steps > 0 and global_step % args.checkpoint_every_steps == 0:
                try:
                    _save_v9_checkpoint(out, "last.pt",
                                        model=model, optimizer=optimizer,
                                        epoch=epoch - 1,
                                        global_step=global_step,
                                        best_micro=best_micro,
                                        best_composite=best_composite,
                                        args=args)
                except Exception as exc:
                    print(f"[v9] intra-epoch save failed: {exc}", flush=True)

        n = max(1, steps_in_epoch)
        train_seg = seg_loss_sum / n
        train_scm = scm_loss_sum / n
        train_cf = cf_loss_sum / n

        val = _evaluate_v9(model, val_loader, device, amp=amp_enabled)
        composite = float(val["dice_mean"]) - 5.0 * float(val["fp_rate_mean"])
        c_curvature = float(model.hyperbolic.c.detach().item())
        line = (
            f"[epoch {epoch:02d}/{args.epochs}] "
            f"seg={train_seg:.4f}  scm={train_scm:.4f}  cf={train_cf:.4f}  | "
            f"dice={val['dice_mean']:.4f}  micro={val['micro_dice']:.4f}  "
            f"fp={val['fp_rate_mean']:.4f}  comp={composite:.4f}  | "
            f"o_at={val['ortho_at']:.4f}  o_as={val['ortho_as']:.4f}  "
            f"o_ts={val['ortho_ts']:.4f}  dag_h={val['dag_h']:.4f}  "
            f"c={c_curvature:.3f}  cf_recon={val['cf_recon_loss']:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  ({time.time() - t0:.1f}s)"
        )
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

        # Save checkpoints (last + best by micro + best by composite)
        _save_v9_checkpoint(out, "last.pt",
                            model=model, optimizer=optimizer,
                            epoch=epoch, global_step=global_step,
                            best_micro=best_micro, best_composite=best_composite,
                            args=args)
        if val["micro_dice"] > best_micro:
            best_micro = val["micro_dice"]
            _save_v9_checkpoint(out, "best_micro.pt",
                                model=model, optimizer=optimizer,
                                epoch=epoch, global_step=global_step,
                                best_micro=best_micro, best_composite=best_composite,
                                args=args)
        if composite > best_composite:
            best_composite = composite
            _save_v9_checkpoint(out, "best_model.pt",
                                model=model, optimizer=optimizer,
                                epoch=epoch, global_step=global_step,
                                best_micro=best_micro, best_composite=best_composite,
                                args=args)

    print(f"[v9] done. best micro={best_micro:.4f}  best composite={best_composite:.4f}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
