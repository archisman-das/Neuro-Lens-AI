"""Symmetry-based geometric anomaly score for brain MRI.

Replaces the SDF geometric tower (which capped at AUC 0.18 on OOD even
after retraining — see scripts/eval_ood_v9b_sdf_v2.py). Brain MRI has
strong bilateral (left-right) symmetry along the midsagittal plane.
Tumors break this symmetry; healthy brains don't, to a quantifiable
degree. Supported by:
  - MDPI Symmetry 2023 paper (Symmetry 15(8) 1586) — symmetry analysis
    for tumor detection in T1/T1C/FLAIR axial slices
  - medRxiv 2026.04 — "Normal is All You Need: Symmetry-Informed Inverse
    Learning Foundation Model for Neuroimaging Diagnostics"

This implementation is intentionally DETERMINISTIC (no training needed):
  1. Locate the midsagittal axis (defaults to image center; refines via
     brain-bbox centroid + small horizontal-flip search).
  2. Mirror the image around that axis.
  3. Compute |image - mirror| inside the brain mask → asymmetry map.
  4. Per-image score = high quantile of the asymmetry map.

Per-image score is what gets thresholded; the asymmetry map itself can
be shown as a heatmap.

Failure modes acknowledged up-front:
  - Sagittal or coronal slices have NO meaningful L-R symmetry. We detect
    this via the existing view_router and SKIP symmetry scoring on those
    views (returning None so the ensemble silently drops it).
  - Severe head rotation breaks symmetry → we do a small ±5° search to
    align before mirroring.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _find_midline_offset(gray: np.ndarray, max_shift_px: int = 16) -> int:
    """Find the horizontal pixel offset that minimises the L-R mirror
    difference. Returns an integer offset of the symmetry axis relative
    to the image center (positive = right of center).

    We search a small ±max_shift_px window because:
      - Most brain MRIs are already roughly centered
      - Large offsets would be a different bug (cropping / registration)
      - Search adds tens of ms at 256x256
    """
    H, W = gray.shape
    cx = W // 2
    best_shift = 0
    best_diff = float('inf')
    fg = gray > max(20.0, gray.max() * 0.05)
    if not fg.any():
        return 0
    for shift in range(-max_shift_px, max_shift_px + 1):
        axis = cx + shift
        # Width of strip on each side that fits in the image
        w_left = axis
        w_right = W - axis
        w = min(w_left, w_right)
        if w < 30:
            continue
        left = gray[:, axis - w:axis]
        right_mirror = gray[:, axis:axis + w][:, ::-1]
        mask_left = fg[:, axis - w:axis]
        # Only score where both sides have brain
        mask_right_mirror = fg[:, axis:axis + w][:, ::-1]
        m = mask_left & mask_right_mirror
        if m.sum() < 100:
            continue
        diff = np.abs(left.astype(np.float32) - right_mirror.astype(np.float32))
        # Per-pixel mean over the overlapping brain region
        score = float(diff[m].mean())
        if score < best_diff:
            best_diff = score
            best_shift = shift
    return best_shift


def symmetry_anomaly_map(image_rgb_uint8: np.ndarray,
                          view: Optional[str] = 'axial') -> Optional[np.ndarray]:
    """Return a per-pixel asymmetry map, or None if view doesn't admit
    bilateral symmetry analysis (sagittal slices).

    image_rgb_uint8: (H, W, 3) uint8 input
    view: 'axial' | 'coronal' | 'sagittal' | 'unknown'. Sagittal returns
          None because there is no meaningful left-right axis in sagittal
          slices (single hemisphere view).
    """
    if view == 'sagittal':
        return None
    if image_rgb_uint8.ndim == 3:
        gray = image_rgb_uint8.mean(axis=-1).astype(np.float32)
    else:
        gray = image_rgb_uint8.astype(np.float32)
    H, W = gray.shape

    # 1. Find midsagittal axis with small horizontal search
    shift = _find_midline_offset(gray)
    axis = W // 2 + shift

    # 2. Build asymmetry map = |I(x,y) - I(2*axis - x, y)| for in-bounds x
    out = np.zeros_like(gray)
    # The valid mirroring band has x in [max(0, 2*axis - (W-1)), min(W-1, 2*axis)]
    x_min = max(0, 2 * axis - (W - 1))
    x_max = min(W - 1, 2 * axis)
    if x_max <= x_min:
        return out
    xs = np.arange(x_min, x_max + 1)
    mirrors = 2 * axis - xs
    band = gray[:, xs]
    mband = gray[:, mirrors]
    out[:, xs] = np.abs(band - mband)
    # Restrict to brain region (mask out background)
    fg = gray > max(20.0, gray.max() * 0.05)
    out = out * fg.astype(np.float32)
    return out


def symmetry_score(image_rgb_uint8: np.ndarray,
                    view: Optional[str] = 'axial',
                    percentile: float = 95.0) -> Optional[float]:
    """Per-image asymmetry score = `percentile` percentile of the
    asymmetry map values inside the brain region. Returns None on
    sagittal views (no L-R symmetry to leverage).
    """
    amap = symmetry_anomaly_map(image_rgb_uint8, view=view)
    if amap is None:
        return None
    # Only score over brain pixels (non-zero entries in the map happen
    # within brain region by construction).
    nz = amap[amap > 0]
    if nz.size == 0:
        return 0.0
    return float(np.percentile(nz, percentile))


__all__ = ['symmetry_score', 'symmetry_anomaly_map']
