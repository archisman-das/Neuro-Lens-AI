"""Glue layer between the conformal-counterfactual module and dashboard.py.

Keeps the dashboard import surface tiny: a single
`analyze(image_bytes, factual_prob_map) -> dict | None` call that handles
artifact lookup, batched intervention inference, and result packing.

Behaviour
---------
- If `conformal_artifacts/` does not exist, returns None. The dashboard
  will see no `conformal_counterfactual` field and the UI degrades cleanly.
- Otherwise loads every `*.json` calibration file, runs the corresponding
  intervention through the same ONNX session used for the factual
  segmentation, and returns a compact summary plus optional PNG heatmaps
  of (cf_prob, abstain, certified_disagree) per intervention.

The point of integration in the dashboard is **after** the cascade pick.
We re-use the factual probability map that the cascade already computed,
so the only added cost is one forward pass per intervention.
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from .conformal_counterfactual_seg import (
    ConformalCounterfactualSegmenter,
    IdentityIntervention,
    intervention_from_dict,
)


ARTIFACTS_DIR_ENV = "CONFORMAL_ARTIFACTS_DIR"
DEFAULT_ARTIFACTS_DIR = "conformal_artifacts"
RETURN_HEATMAPS_ENV = "CONFORMAL_RETURN_HEATMAPS"


def _artifacts_dir() -> Optional[Path]:
    p = Path(os.environ.get(ARTIFACTS_DIR_ENV, DEFAULT_ARTIFACTS_DIR))
    if not p.exists() or not p.is_dir():
        return None
    if not any(p.glob("*.json")):
        return None
    return p


def _viridis_rgb(g: np.ndarray) -> np.ndarray:
    """Cheap 5-anchor viridis approximation, numpy-only.

    Matches the colormap helper used elsewhere in dashboard.py so the
    conformal heatmaps look consistent with the segmentation overlays.
    """
    g = np.clip(g.astype(np.float32), 0.0, 1.0)
    anchors = np.array(
        [
            [0.267, 0.005, 0.329],   # 0.00
            [0.282, 0.140, 0.458],   # 0.25
            [0.254, 0.265, 0.530],   # 0.50
            [0.207, 0.372, 0.553],   # 0.75
            [0.993, 0.906, 0.144],   # 1.00
        ],
        dtype=np.float32,
    )
    t = g * 4.0
    lo = np.clip(np.floor(t).astype(np.int32), 0, 3)
    hi = np.clip(lo + 1, 0, 4)
    frac = (t - lo)[..., None]
    out = anchors[lo] * (1.0 - frac) + anchors[hi] * frac
    return (out * 255).astype(np.uint8)


def _encode_png_data_url(arr_uint8: np.ndarray) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr_uint8).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


@dataclass
class _LoadedArtifact:
    slug: str
    state: dict


def _load_artifacts(dirpath: Path) -> List[_LoadedArtifact]:
    out: List[_LoadedArtifact] = []
    for p in sorted(dirpath.glob("*.json")):
        if p.name == "summary.json":
            continue
        try:
            import json
            d = json.loads(p.read_text())
            out.append(_LoadedArtifact(slug=p.stem, state=d))
        except Exception:
            continue
    return out


def analyze(
    *,
    image_array: np.ndarray,
    factual_prob: np.ndarray,
    seg_fn: Callable[[np.ndarray], np.ndarray],
    threshold: float,
    return_heatmaps: Optional[bool] = None,
    seg_fn_batched: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Optional[Dict]:
    """Run the calibrated intervention battery on one image.

    Parameters
    ----------
    image_array: HxWx3 float in [0,1] - the same input the factual
        segmenter consumed (post-resize, pre-normalisation, so the
        intervention acts on raw pixel space, not on normalised tensors).
    factual_prob: HxW float in [0,1] - the factual probability map the
        cascade winner produced. Used as the baseline against which each
        counterfactual is compared (effect = cf_prob - factual).
    seg_fn: callable(image HxWx3) -> prob HxW. The dashboard wraps the
        existing ONNX/PyTorch path so each intervention reuses the cached
        session and incurs only the cost of one forward pass.
    threshold: same threshold the factual segmenter used (e.g. 0.5).
    return_heatmaps: when True, embed PNG data-URLs for cf_prob and
        certified_disagree maps so the UI can render them. When False,
        return only scalar summary metrics. Default reads
        CONFORMAL_RETURN_HEATMAPS env var (off in production for payload
        size, on for local debugging).

    Returns
    -------
    None when no calibration artifacts are found (graceful no-op).
    Otherwise a dict with:
      {
        "available": True,
        "artifacts_dir": str,
        "n_interventions": int,
        "interventions": [
          {
            "slug": str,                          # filename minus .json
            "label": str,                         # human-readable
            "q": float,                           # calibrated conformal margin
            "alpha": float,
            "abstain_fraction": float,            # voxelwise share in [0,1]
            "certified_disagree_fraction": float, # |cf - factual| > q
            "intervention_cf_area_px": int,       # |cf >= threshold|
            "intervention_effect_mean": float,    # mean(cf - factual)
            "cf_heatmap"?: data-url,
            "disagree_heatmap"?: data-url,
          },
          ...
        ],
        "factual_baseline_area_px": int,
        "summary": {
          "max_disagree_fraction": float,
          "max_disagree_intervention": str,
          "most_robust_intervention": str,  # lowest disagree fraction
        }
      }
    """
    artifacts = _artifacts_dir()
    if artifacts is None:
        return None
    loaded = _load_artifacts(artifacts)
    if not loaded:
        return None

    if return_heatmaps is None:
        flag = os.environ.get(RETURN_HEATMAPS_ENV, "").strip().lower()
        return_heatmaps = flag in ("1", "true", "yes")

    factual_area = int((factual_prob >= float(threshold)).sum())

    # ---- FAST PATH: batch all N interventions into ONE ORT call ----
    # Apply each intervention to image_array, stack to (N, H, W, 3), pass
    # through seg_fn_batched in a single forward pass. Saves N-1 ORT calls
    # = ~5-6 sec on cpu-basic (vs 8 sequential ~6-8 sec). Falls back to
    # the slow per-intervention loop if seg_fn_batched is not provided.
    batched_cf_probs = None
    if seg_fn_batched is not None:
        try:
            interventions = [intervention_from_dict(la.state["intervention"])
                              for la in loaded]
            stacked = np.stack(
                [iv.apply(image_array) for iv in interventions], axis=0
            )  # (N, H, W, 3)
            batched_cf_probs = seg_fn_batched(stacked)  # (N, H, W)
            assert batched_cf_probs.shape[0] == len(loaded), \
                f"batched seg_fn returned {batched_cf_probs.shape[0]} != {len(loaded)} probs"
        except Exception:
            # If anything in the fast path fails, fall back to slow path.
            batched_cf_probs = None

    rows: List[Dict] = []
    for idx, la in enumerate(loaded):
        iv = intervention_from_dict(la.state["intervention"])
        seg = ConformalCounterfactualSegmenter(
            seg_fn=seg_fn,
            intervention=iv,
            alpha=float(la.state["alpha"]),
            threshold=float(la.state["threshold"]),
        )
        seg.q = float(la.state["q"])
        seg.band_low = float(la.state["band_low"])
        seg.band_high = float(la.state["band_high"])
        seg.include_positive_in_pool = bool(la.state["include_positive_in_pool"])

        # Use the pre-computed batched cf_prob if available; otherwise the
        # slow per-intervention seg_fn call inside seg.predict().
        if batched_cf_probs is not None:
            cf_prob_pre = batched_cf_probs[idx].astype(np.float32)
            lower = np.clip(cf_prob_pre - seg.q, 0.0, 1.0)
            upper = np.clip(cf_prob_pre + seg.q, 0.0, 1.0)
            certain_fg = lower > seg.threshold
            certain_bg = upper < seg.threshold
            abstain = ~(certain_fg | certain_bg)
            out = {
                "cf_prob": cf_prob_pre,
                "lower": lower, "upper": upper,
                "certain_fg": certain_fg, "certain_bg": certain_bg,
                "abstain": abstain,
                "intervention": iv.to_dict(),
                "q": float(seg.q),
                "alpha": float(seg.alpha),
            }
        else:
            out = seg.predict(image_array)
        cf_prob = out["cf_prob"]
        effect = cf_prob - factual_prob
        certified_disagree = np.abs(effect) > out["q"]
        cf_area = int((cf_prob >= float(threshold)).sum())

        row: Dict = {
            "slug": la.slug,
            "label": _human_label(la.slug),
            "q": float(out["q"]),
            "alpha": float(out["alpha"]),
            "abstain_fraction": float(out["abstain"].mean()),
            "certified_disagree_fraction": float(certified_disagree.mean()),
            "intervention_cf_area_px": cf_area,
            "intervention_effect_mean": float(effect.mean()),
            "intervention_effect_abs_max": float(np.abs(effect).max()),
        }
        if return_heatmaps:
            row["cf_heatmap"] = _encode_png_data_url(_viridis_rgb(cf_prob))
            row["disagree_heatmap"] = _encode_png_data_url(
                _viridis_rgb(certified_disagree.astype(np.float32))
            )
        rows.append(row)

    # Pick most-disagreeing and most-robust interventions for at-a-glance.
    most_disagree = max(rows, key=lambda r: r["certified_disagree_fraction"], default=None)
    most_robust = min(rows, key=lambda r: r["certified_disagree_fraction"], default=None)
    return {
        "available": True,
        "artifacts_dir": str(artifacts),
        "n_interventions": len(rows),
        "interventions": rows,
        "factual_baseline_area_px": factual_area,
        "summary": {
            "max_disagree_fraction": (
                float(most_disagree["certified_disagree_fraction"]) if most_disagree else 0.0
            ),
            "max_disagree_intervention": most_disagree["slug"] if most_disagree else "",
            "most_robust_intervention": most_robust["slug"] if most_robust else "",
        },
        "_method": "weighted-conformal-counterfactual (CONSeg + CausalX-Net unified)",
    }


def _human_label(slug: str) -> str:
    if slug == "identity":
        return "Factual (no intervention)"
    if slug.startswith("modality_keep_"):
        ch = slug.replace("modality_keep_", "")
        return f"do(M = {ch}-only)"
    if slug.startswith("intensity_shift_"):
        sign = "+" if "+" in slug else "-"
        val = slug.split("_")[-1].replace("+", "").replace("-", "")
        return f"do(I = I {sign} {val})"
    if slug.startswith("contrast_scale_"):
        val = slug.split("_")[-1]
        return f"do(gamma = {val})"
    return slug
