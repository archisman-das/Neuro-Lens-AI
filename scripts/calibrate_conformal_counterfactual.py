"""Calibrate ConformalCounterfactualSegmenter on the v5 validation set.

Run after v5 training finishes:

    python scripts/calibrate_conformal_counterfactual.py \
        --data_dir dataset_v5/val \
        --model_path segmentation_artifacts/attention_unet_v5/best_model.pt \
        --out_dir conformal_artifacts \
        --alpha 0.10

Produces one JSON state file per intervention in the standard battery:
    conformal_artifacts/identity.json
    conformal_artifacts/modality_keep_T1.json
    conformal_artifacts/modality_keep_T1c.json
    conformal_artifacts/modality_keep_FLAIR.json
    conformal_artifacts/intensity_shift_+0.10.json
    conformal_artifacts/intensity_shift_-0.10.json
    conformal_artifacts/contrast_scale_0.70.json
    conformal_artifacts/contrast_scale_1.50.json

These are loaded at inference time by dashboard.py to produce voxelwise
prediction sets with the (1 - alpha) coverage guarantee under each
intervention.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

from src.research.conformal_counterfactual_seg import (  # noqa: E402
    CalibrationSample,
    ConformalCounterfactualSegmenter,
    standard_intervention_battery,
)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _make_seg_fn(model_path: Path, image_size: int):
    """Load the v5 segmentation model into a callable: HxWx3 [0,1] -> HxW [0,1]."""
    import torch

    import segmentation_models_pytorch as smp

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(model_path, map_location=device, weights_only=False)
    # Three observed checkpoint formats:
    #   1. Bare state_dict (raw tensor map)
    #   2. v5 trainer wrapper: {"state_dict", "architecture", "encoder", ...}
    #   3. Legacy wrapper: {"model_state_dict": ...}
    encoder = "resnet34"
    arch = "Unet"
    if isinstance(state, dict) and "state_dict" in state:
        encoder = state.get("encoder", encoder) or encoder
        arch = state.get("architecture", arch) or arch
        weights = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        weights = state["model_state_dict"]
    else:
        weights = state
    SmpClass = getattr(smp, arch)
    model = SmpClass(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(weights)
    model.eval().to(device)

    @torch.no_grad()
    def seg_fn(x: np.ndarray) -> np.ndarray:
        # x is HxWx3 in [0,1]
        h, w, _ = x.shape
        if (h, w) != (image_size, image_size):
            x = np.asarray(
                Image.fromarray((x * 255).astype(np.uint8)).resize(
                    (image_size, image_size), Image.BILINEAR
                ),
                dtype=np.float32,
            ) / 255.0
        xn = (x - IMAGENET_MEAN) / IMAGENET_STD
        t = torch.from_numpy(xn.transpose(2, 0, 1)[None]).float().to(device)
        logits = model(t)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        if prob.shape != (h, w):
            prob = np.asarray(
                Image.fromarray((prob * 255).astype(np.uint8)).resize(
                    (w, h), Image.BILINEAR
                ),
                dtype=np.float32,
            ) / 255.0
        return prob.astype(np.float32)

    return seg_fn


def _load_calibration(data_dir: Path, image_size: int, limit: int | None) -> list:
    img_dir = data_dir / "images"
    msk_dir = data_dir / "masks"
    img_paths = sorted(img_dir.iterdir())
    if limit is not None:
        img_paths = img_paths[:limit]
    samples: list = []
    for ip in img_paths:
        mp = msk_dir / ip.name
        if not mp.exists():
            continue
        img = np.asarray(
            Image.open(ip).convert("RGB").resize((image_size, image_size), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        msk = np.asarray(
            Image.open(mp).convert("L").resize((image_size, image_size), Image.NEAREST),
            dtype=np.float32,
        )
        msk = (msk > 127).astype(np.float32)
        # Only keep positives for calibration: empty masks make the
        # voxel pool degenerate (no positives, predicted-uncertain pool
        # is usually empty for well-trained models). Coverage guarantee
        # still applies marginally over positive scans which is what
        # clinicians care about.
        if msk.sum() < 16:  # at least 4x4 tumor footprint
            continue
        samples.append(CalibrationSample(image=img, mask=msk))
    return samples


def _intervention_slug(iv) -> str:
    d = iv.to_dict()
    name = d["name"]
    p = d["params"]
    if name == "identity":
        return "identity"
    if name == "modality":
        label = {0: "T1", 1: "T1c", 2: "FLAIR"}.get(p["keep_channel"], str(p["keep_channel"]))
        return f"modality_keep_{label}"
    if name == "intensity_shift":
        sign = "+" if p["delta"] >= 0 else "-"
        return f"intensity_shift_{sign}{abs(p['delta']):.2f}"
    if name == "contrast_scale":
        return f"contrast_scale_{p['gamma']:.2f}"
    return name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, type=Path)
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="cap calibration set size (debug only)",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[1/3] loading model from {args.model_path}", flush=True)
    seg_fn = _make_seg_fn(args.model_path, args.image_size)
    print(f"[2/3] loading calibration scans from {args.data_dir}", flush=True)
    samples = _load_calibration(args.data_dir, args.image_size, args.limit)
    print(f"      n={len(samples)} positive scans", flush=True)
    if len(samples) < 30:
        raise SystemExit(
            f"need >=30 calibration scans, got {len(samples)}. "
            f"point --data_dir at dataset_v5/val (or test) and ensure positive scans exist."
        )

    print(f"[3/3] calibrating {len(standard_intervention_battery())} interventions", flush=True)
    summary: dict = {"alpha": args.alpha, "n_calib": len(samples), "interventions": []}
    for iv in standard_intervention_battery():
        slug = _intervention_slug(iv)
        seg = ConformalCounterfactualSegmenter(
            seg_fn=seg_fn,
            intervention=iv,
            alpha=args.alpha,
        )
        report = seg.calibrate(samples, verbose=True)
        out_path = args.out_dir / f"{slug}.json"
        seg.save(out_path)
        summary["interventions"].append(
            {
                "slug": slug,
                "path": str(out_path.relative_to(args.out_dir.parent)),
                "q": report.q,
                "empirical_coverage": report.empirical_coverage_on_calib,
                "n_voxels": report.n_voxels_used,
            }
        )
        print(f"  -> {out_path.name}  q={report.q:.4f}  cov={report.empirical_coverage_on_calib:.3f}")

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"done. {args.out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
