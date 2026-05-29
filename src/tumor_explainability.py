"""Deterministic feature extractor for the LLM-explanation pipeline.

Given (image, segmentation mask, classifier outputs, optional Grad-CAM heatmap)
this module computes EVERY non-LLM-needing piece of information that can be
deduced about the tumor and the model's behavior on it. The output dict is fed
to the LLM by src/llm_explain.py as structured context, so the LLM grounds its
explanation in real numbers rather than hallucinating.

Categories produced (top-level keys of the returned dict):
  - input_summary           image / mask shape, pixel spacing, total brain area
  - geometry                mask area, centroid, bbox, axes, eccentricity, ...
  - components              per-connected-component features (multifocality)
  - localization            heuristic hemisphere/lobe/depth
  - intensity_per_channel   mean/std/min/max/contrast for each RGB channel
  - texture                 GLCM contrast/homogeneity/energy/correlation, entropy
  - multimodal              T1c-enhance / T2-edema / T1-necrosis heuristics
                            (only when image is a multimodal RGB stack)
  - model_behavior          per-model probabilities, inter-model agreement,
                            Grad-CAM peak + mask alignment
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_all_features(
    image_rgb: np.ndarray,
    mask_bin: np.ndarray,
    *,
    pixel_spacing_mm: float = 1.0,
    classifier_results: Optional[dict] = None,
    gradcam_heatmap: Optional[np.ndarray] = None,
    multimodal_channels: Optional[tuple[str, str, str]] = None,
) -> dict:
    """Compute every feature we can derive deterministically from inputs.

    Args:
      image_rgb: (H, W, 3) uint8 or float in [0,1]. The input MRI shown to the user.
      mask_bin:  (H, W) binary 0/1 or 0/255 segmentation prediction from the U-Net.
      pixel_spacing_mm: Physical size of one pixel side in mm. Used to convert
        pixel-space measurements to mm/mm**2. We don't know this for arbitrary
        uploads, so callers can pass the dataset-specific value when known.
      classifier_results: Optional {model_name: {"probability": float, "label": str, ...}}
        from /predict for the same image. Used for model_behavior fields.
      gradcam_heatmap: Optional (H, W) float in [0,1]. If supplied we compute
        Grad-CAM alignment with the segmentation mask.
      multimodal_channels: When the image is a 3-channel modality stack (e.g.
        BraTS T1c/T2/FLAIR), pass the channel names so we can give the LLM
        modality-specific intensity readouts. None means "channels are just RGB".
    """
    image_rgb = _to_uint8(image_rgb)
    mask = _normalize_mask(mask_bin)
    brain_mask = _estimate_brain_mask(image_rgb)

    out: dict = {}
    out['input_summary'] = _input_summary(image_rgb, mask, brain_mask, pixel_spacing_mm)
    out['geometry'] = _geometry(mask, pixel_spacing_mm)
    out['components'] = _components(mask, pixel_spacing_mm)
    out['localization'] = _localization(mask, brain_mask)
    out['intensity_per_channel'] = _intensity_per_channel(image_rgb, mask, brain_mask, multimodal_channels)
    out['texture'] = _texture(image_rgb, mask)
    if multimodal_channels is not None:
        out['multimodal'] = _multimodal_heuristics(image_rgb, mask, brain_mask, multimodal_channels)
    out['model_behavior'] = _model_behavior(classifier_results, mask, gradcam_heatmap)
    # Single-channel radiology features (work on Kaggle-style RGB without modality split):
    out['morphology'] = _morphology(image_rgb, mask, brain_mask)
    out['mass_effect'] = _mass_effect(mask, brain_mask)
    out['internal_architecture'] = _internal_architecture(image_rgb, mask)
    out['grade_evidence'] = _grade_evidence(out)
    out['quality'] = _quality_assessment(image_rgb, brain_mask, mask)
    out['overall_confidence'] = _overall_confidence(out)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_uint8(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb.dtype == np.uint8:
        return image_rgb
    a = np.asarray(image_rgb)
    if a.max() <= 1.0:
        a = (a * 255.0)
    return np.clip(a, 0, 255).astype(np.uint8)


def _normalize_mask(mask_bin: np.ndarray) -> np.ndarray:
    m = np.asarray(mask_bin)
    if m.ndim == 3:
        m = m[..., 0]
    if m.dtype != np.uint8:
        m = (m > 0).astype(np.uint8) * 255
    elif m.max() > 1:
        m = (m > 127).astype(np.uint8) * 255
    else:
        m = m.astype(np.uint8) * 255
    return m  # 0 / 255 uint8


def _estimate_brain_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Rough brain mask via intensity threshold + largest-component cleanup.

    Used to compute features like 'tumor area / brain area' and 'distance from
    skull'. Not perfect but stable across modalities.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, m = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + int(np.argmax(areas))
    return np.where(labels == keep, 255, 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Feature blocks
# ---------------------------------------------------------------------------


def _input_summary(image_rgb, mask, brain_mask, pixel_spacing_mm) -> dict:
    h, w = mask.shape
    brain_area_px = int((brain_mask > 0).sum())
    return {
        'image_height_px': int(h),
        'image_width_px': int(w),
        'pixel_spacing_mm_per_side': float(pixel_spacing_mm),
        'brain_area_px': brain_area_px,
        'brain_area_mm2': float(brain_area_px * pixel_spacing_mm ** 2),
        'image_dtype': str(image_rgb.dtype),
        'tumor_present': bool((mask > 0).any()),
    }


def _geometry(mask, pixel_spacing_mm) -> dict:
    area_px = int((mask > 0).sum())
    if area_px == 0:
        return {'area_px': 0, 'area_mm2': 0.0, 'note': 'no tumor predicted'}

    ys, xs = np.where(mask > 0)
    centroid = (float(xs.mean()), float(ys.mean()))
    bbox_x0, bbox_y0 = int(xs.min()), int(ys.min())
    bbox_x1, bbox_y1 = int(xs.max()), int(ys.max())
    bbox_w = bbox_x1 - bbox_x0 + 1
    bbox_h = bbox_y1 - bbox_y0 + 1

    # Find all external contours. For multifocal masks we previously mixed
    # total mask area with the largest-only perimeter/hull, producing
    # nonsensical solidity > 1 and circularity > 1. Now:
    #   - solidity uses the convex hull of ALL contour points combined.
    #   - circularity is computed from the SUM of contour perimeters and the
    #     SUM of contour areas, so it represents the full mask, not one blob.
    #   - major/minor axis / eccentricity / orientation still come from the
    #     largest contour (these are inherently per-component features).
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest = max(contours, key=cv2.contourArea) if contours else None
    total_perimeter = float(sum(cv2.arcLength(c, True) for c in contours))
    total_contour_area = float(sum(cv2.contourArea(c) for c in contours)) if contours else 0.0
    if contours:
        all_pts = np.vstack(contours)
        convex_hull = cv2.convexHull(all_pts)
        hull_area = float(cv2.contourArea(convex_hull))
    else:
        hull_area = float(area_px)
    # Use total_contour_area (cv2-measured, may differ slightly from pixel
    # count when contours include sub-pixel curves) but clip to area_px so
    # solidity stays <= 1.0 in all cases.
    solidity_numer = min(total_contour_area, float(area_px))
    solidity = float(solidity_numer / hull_area) if hull_area > 0 else 0.0
    solidity = max(0.0, min(1.0, solidity))

    major_axis = minor_axis = orientation_deg = eccentricity = 0.0
    if largest is not None and len(largest) >= 5:
        (cx_e, cy_e), (a, b), theta = cv2.fitEllipse(largest)
        major_axis = float(max(a, b))
        minor_axis = float(min(a, b))
        orientation_deg = float(theta)
        if major_axis > 0:
            eccentricity = float(math.sqrt(max(0.0, 1.0 - (minor_axis / major_axis) ** 2)))

    equivalent_diameter = float(2.0 * math.sqrt(area_px / math.pi))
    # Use total perimeter + total contour area; clamp to [0, 1].
    circularity = (float(4 * math.pi * total_contour_area / (total_perimeter ** 2))
                   if total_perimeter > 0 else 0.0)
    circularity = max(0.0, min(1.0, circularity))
    perimeter_px = total_perimeter

    return {
        'area_px': area_px,
        'area_mm2': float(area_px * pixel_spacing_mm ** 2),
        'centroid_xy_px': centroid,
        'bounding_box_xywh_px': (bbox_x0, bbox_y0, bbox_w, bbox_h),
        'equivalent_diameter_px': equivalent_diameter,
        'major_axis_px': major_axis,
        'minor_axis_px': minor_axis,
        'orientation_deg': orientation_deg,
        'eccentricity': eccentricity,                  # 0 = circle, ~1 = elongated
        'perimeter_px': perimeter_px,
        'circularity': circularity,                    # 1 = perfect circle, < 1 = irregular
        'solidity': solidity,                          # 1 = convex, <1 = has concavities
        'extent': float(area_px / max(bbox_w * bbox_h, 1)),
        'shape_complexity': float(perimeter_px ** 2 / max(4 * math.pi * area_px, 1)),
    }


def _components(mask, pixel_spacing_mm) -> dict:
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return {'n_components': 0, 'multifocal': False, 'components': []}
    comps = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        comps.append({
            'index': i,
            'area_px': a,
            'area_mm2': float(a * pixel_spacing_mm ** 2),
            'centroid_xy_px': (cx, cy),
            'bbox_xywh_px': (
                int(stats[i, cv2.CC_STAT_LEFT]),
                int(stats[i, cv2.CC_STAT_TOP]),
                int(stats[i, cv2.CC_STAT_WIDTH]),
                int(stats[i, cv2.CC_STAT_HEIGHT]),
            ),
        })
    comps.sort(key=lambda c: c['area_px'], reverse=True)
    largest_frac = comps[0]['area_px'] / sum(c['area_px'] for c in comps)
    return {
        'n_components': n - 1,
        'multifocal': (n - 1) > 1 and largest_frac < 0.9,
        'largest_component_area_fraction': float(largest_frac),
        'components': comps,
    }


def _localization(mask, brain_mask) -> dict:
    if (mask > 0).sum() == 0:
        return {'note': 'no tumor predicted'}
    ys_b, xs_b = np.where(brain_mask > 0)
    if len(xs_b) == 0:
        return {'note': 'no brain mask detected'}
    brain_left, brain_right = int(xs_b.min()), int(xs_b.max())
    brain_top, brain_bottom = int(ys_b.min()), int(ys_b.max())
    brain_w = max(brain_right - brain_left, 1)
    brain_h = max(brain_bottom - brain_top, 1)
    midline_x = (brain_left + brain_right) / 2

    ys, xs = np.where(mask > 0)
    cx, cy = float(xs.mean()), float(ys.mean())
    rel_x = (cx - brain_left) / brain_w   # 0 = left of brain, 1 = right
    rel_y = (cy - brain_top) / brain_h    # 0 = top, 1 = bottom

    hemisphere = 'left' if cx < midline_x else 'right'

    if rel_y < 0.33:
        ap = 'anterior (frontal)'
    elif rel_y < 0.66:
        ap = 'middle (central)'
    else:
        ap = 'posterior (occipital/parietal)'

    if rel_x < 0.33:
        ml_pos = 'lateral-left'
    elif rel_x < 0.66:
        ml_pos = 'midline / paramedian'
    else:
        ml_pos = 'lateral-right'

    # Approximate lobe by quadrant - rough heuristic, not radiology-grade.
    if rel_y < 0.4:
        lobe_hint = 'frontal lobe (approx.)'
    elif rel_y > 0.7 and (rel_x < 0.3 or rel_x > 0.7):
        lobe_hint = 'temporal lobe (approx.)'
    elif rel_y > 0.7:
        lobe_hint = 'occipital lobe (approx.)'
    else:
        lobe_hint = 'parietal lobe (approx.)'

    # Distance from mask centroid to nearest brain-perimeter pixel (skull edge).
    edge = cv2.Canny(brain_mask, 50, 150)
    edge_ys, edge_xs = np.where(edge > 0)
    if len(edge_xs):
        d = np.sqrt((edge_xs - cx) ** 2 + (edge_ys - cy) ** 2)
        dist_to_skull_px = float(d.min())
    else:
        dist_to_skull_px = 0.0

    depth_label = 'peripheral / cortical' if dist_to_skull_px < 0.15 * max(brain_w, brain_h) else 'deep / subcortical'

    # Midline-shift indicator from brain symmetry.
    left_area = int((brain_mask[:, :int(midline_x)] > 0).sum())
    right_area = int((brain_mask[:, int(midline_x):] > 0).sum())
    asymmetry_ratio = float(abs(left_area - right_area) / max(left_area + right_area, 1))

    return {
        'hemisphere': hemisphere,
        'anterior_posterior': ap,
        'medial_lateral': ml_pos,
        'approximate_lobe_hint': lobe_hint,
        'relative_xy_in_brain_bbox': (float(rel_x), float(rel_y)),
        'distance_to_skull_px': dist_to_skull_px,
        'depth_label': depth_label,
        'brain_left_right_asymmetry_ratio': asymmetry_ratio,
        'midline_shift_suspected': asymmetry_ratio > 0.07,
    }


def _intensity_per_channel(image_rgb, mask, brain_mask, channel_names) -> dict:
    out = {}
    channels = channel_names or ('R', 'G', 'B')
    for i, name in enumerate(channels):
        ch = image_rgb[..., i].astype(np.float32)
        in_mask = ch[mask > 0]
        if in_mask.size == 0:
            out[name] = {'note': 'no tumor pixels'}
            continue
        in_brain_outside = ch[(brain_mask > 0) & (mask == 0)]
        bg_mean = float(in_brain_outside.mean()) if in_brain_outside.size else 0.0
        out[name] = {
            'mean': float(in_mask.mean()),
            'std': float(in_mask.std()),
            'median': float(np.median(in_mask)),
            'min': float(in_mask.min()),
            'max': float(in_mask.max()),
            'p10_p90': (float(np.percentile(in_mask, 10)), float(np.percentile(in_mask, 90))),
            'mean_in_brain_outside_tumor': bg_mean,
            'tumor_vs_brain_contrast': float(in_mask.mean() - bg_mean),
            'relative_intensity_ratio': float(in_mask.mean() / bg_mean) if bg_mean > 1e-3 else None,
            'hyperintense_vs_brain': bool(in_mask.mean() > bg_mean * 1.10),
            'hypointense_vs_brain': bool(in_mask.mean() < bg_mean * 0.85),
        }
    return out


def _texture(image_rgb, mask) -> dict:
    """GLCM and entropy summary computed on the green channel (proxy for brain)."""
    if (mask > 0).sum() < 20:
        return {'note': 'tumor too small for texture analysis'}
    try:
        from skimage.feature import graycomatrix, graycoprops
        from skimage.measure import shannon_entropy
    except ImportError:
        return {'note': 'scikit-image not installed; skipping texture'}

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    ys, xs = np.where(mask > 0)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    patch = gray[y0:y1, x0:x1]
    if patch.size < 16:
        return {'note': 'tumor patch too small'}
    patch_q = (patch // 16).astype(np.uint8)   # quantize to 16 grey levels
    distances = [1, 2]
    angles = [0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]
    glcm = graycomatrix(patch_q, distances=distances, angles=angles, levels=16, symmetric=True, normed=True)
    out = {}
    for prop in ('contrast', 'homogeneity', 'energy', 'correlation', 'dissimilarity'):
        vals = graycoprops(glcm, prop=prop)
        out[prop] = float(vals.mean())
    out['shannon_entropy'] = float(shannon_entropy(patch))
    out['heterogeneity_score'] = float(patch.std() / max(patch.mean(), 1e-3))
    return out


def _multimodal_heuristics(image_rgb, mask, brain_mask, channel_names) -> dict:
    """For BraTS-style stacks (T1c, T2, FLAIR) we can read enhancement / edema
    / necrosis straight from the per-channel intensities inside vs around the
    mask. The user passes channel_names so we know which channel is which."""
    name_to_idx = {n.lower(): i for i, n in enumerate(channel_names)}
    out = {}
    H, W = mask.shape[:2]

    def _ch(channel_key):
        idx = name_to_idx.get(channel_key)
        if idx is None:
            return None
        ch = image_rgb[..., idx].astype(np.float32)
        in_mask = ch[mask > 0]
        in_brain_outside = ch[(brain_mask > 0) & (mask == 0)]
        return ch, in_mask, in_brain_outside

    t1c = _ch('t1ce') or _ch('t1c')
    t2 = _ch('t2')
    flair = _ch('flair')
    t1 = _ch('t1')

    if t1c is not None:
        _, in_mask, in_bg = t1c
        thr = float(np.percentile(in_bg, 80)) if in_bg.size else float(in_mask.mean())
        enhancing_frac = float((in_mask > thr).mean())
        out['t1c_enhancing_fraction'] = enhancing_frac
        out['t1c_mean_intensity_in_tumor'] = float(in_mask.mean())
        out['t1c_predominantly_enhancing'] = enhancing_frac > 0.4
    if t2 is not None:
        _, in_mask, in_bg = t2
        if in_bg.size:
            ratio = float(in_mask.mean() / max(in_bg.mean(), 1e-3))
            out['t2_hyperintensity_ratio'] = ratio
            out['t2_strongly_hyperintense'] = ratio > 1.25
    if flair is not None:
        # Edema halo: dilate the mask, look at the difference ring.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dilated = cv2.dilate(mask, kernel, iterations=3)
        halo = (dilated > 0) & (mask == 0) & (brain_mask > 0)
        flair_ch, in_mask, in_bg = flair
        if halo.any() and in_bg.size:
            halo_mean = float(flair_ch[halo].mean())
            out['flair_peritumoral_mean'] = halo_mean
            out['flair_brain_background_mean'] = float(in_bg.mean())
            out['edema_halo_ratio'] = float(halo_mean / max(in_bg.mean(), 1e-3))
            out['edema_likely'] = halo_mean > in_bg.mean() * 1.20
    if t1 is not None and t1c is not None:
        # Necrosis: low T1c relative to T1, or dark central core inside the mask.
        t1c_ch, t1c_in_mask, _ = t1c
        t1_ch, t1_in_mask, _ = t1
        if t1c_in_mask.size and t1_in_mask.size:
            necrotic_frac = float((t1c_in_mask < np.percentile(t1c_in_mask, 25)).mean())
            out['t1c_low_intensity_fraction'] = necrotic_frac
            out['necrosis_likely'] = necrotic_frac > 0.2 and bool(out.get('t1c_predominantly_enhancing'))

    return out


def _morphology(image_rgb, mask, brain_mask) -> dict:
    """Border definition + internal heterogeneity grounded on the actual pixels.

    Radiologists care about: how sharp is the tumor border (well-circumscribed
    vs infiltrative), and how uniform is the inside.

    Border sharpness: mean intensity gradient magnitude on a 2 px ring around
    the mask boundary. Higher = sharper border. Reported as both raw value and
    a categorical label ('sharp', 'moderate', 'ill-defined').

    Heterogeneity within tumor: std/mean of intensity inside the mask, plus an
    explicit number of distinct intensity zones from a 3-cluster k-means on the
    masked region.
    """
    if (mask > 0).sum() < 30:
        return {'note': 'tumor too small for morphology analysis'}
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Border band = boundary ring.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(mask, kernel, iterations=1)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    border_band = ((dilated > 0) & (eroded == 0)).astype(np.uint8)
    # Sobel magnitude.
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    border_grad = grad_mag[border_band > 0]
    if border_grad.size == 0:
        border_score = 0.0
    else:
        border_score = float(border_grad.mean())
    # Calibrate against the brain's overall mean gradient so the score is
    # comparable across scans of different contrast.
    brain_grad = grad_mag[brain_mask > 0]
    brain_grad_mean = float(brain_grad.mean()) if brain_grad.size else 1.0
    border_relative = border_score / max(brain_grad_mean, 1.0)
    if border_relative > 1.6:
        border_label = 'sharp / well-circumscribed'
    elif border_relative > 1.1:
        border_label = 'moderately defined'
    else:
        border_label = 'ill-defined / infiltrative'

    # Internal heterogeneity zones via k-means (k=3).
    in_mask = gray[mask > 0].astype(np.float32).reshape(-1, 1)
    n_intensity_zones = 0
    cluster_means: list[float] = []
    if in_mask.shape[0] >= 30:
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(in_mask, K=3, bestLabels=None,
                                          criteria=crit, attempts=2, flags=cv2.KMEANS_PP_CENTERS)
        cluster_means = sorted([float(c[0]) for c in centers])
        # Count distinct zones (centers must differ by > 12 grey levels to be considered separate).
        n_intensity_zones = 1
        prev = cluster_means[0]
        for m in cluster_means[1:]:
            if m - prev > 12:
                n_intensity_zones += 1
            prev = m

    return {
        'border_gradient_mean': border_score,
        'brain_gradient_mean': brain_grad_mean,
        'border_relative_to_brain': float(border_relative),
        'border_label': border_label,
        'internal_intensity_zones': int(n_intensity_zones),
        'internal_intensity_cluster_means': cluster_means,
    }


def _mass_effect(mask, brain_mask) -> dict:
    """Heuristic mass effect indicators:
      - Brain-shape symmetry: difference in horizontally-mirrored intersection.
        A healthy brain is roughly bilaterally symmetric; a large mass distorts
        the contralateral side.
      - Tumor displacement from brain centroid: how far is the tumor centroid
        from the brain's centroid, normalised by brain radius? Closer = central
        / midline, further = peripheral.
      - Tumor-to-brain-area ratio: large tumors carry more mass effect risk.
    """
    if (mask > 0).sum() == 0 or (brain_mask > 0).sum() == 0:
        return {'note': 'mass effect not computable'}
    h, w = mask.shape
    # Symmetry.
    mirrored = brain_mask[:, ::-1]
    inter = int(((brain_mask > 0) & (mirrored > 0)).sum())
    union = int(((brain_mask > 0) | (mirrored > 0)).sum())
    symmetry_iou = float(inter / union) if union else 0.0
    # Brain centroid + radius.
    ys_b, xs_b = np.where(brain_mask > 0)
    brain_cx = float(xs_b.mean())
    brain_cy = float(ys_b.mean())
    brain_radius = float(np.sqrt(((xs_b - brain_cx) ** 2 + (ys_b - brain_cy) ** 2).mean()))
    # Tumor centroid + size.
    ys_t, xs_t = np.where(mask > 0)
    cx = float(xs_t.mean())
    cy = float(ys_t.mean())
    tumor_area = int((mask > 0).sum())
    brain_area = int((brain_mask > 0).sum())
    tumor_to_brain_ratio = float(tumor_area / max(brain_area, 1))
    dist_to_brain_centroid = float(math.hypot(cx - brain_cx, cy - brain_cy))
    rel_to_brain_radius = float(dist_to_brain_centroid / max(brain_radius, 1.0))
    # Compose an evidence label.
    evidence = 0
    if symmetry_iou < 0.85:
        evidence += 1
    if tumor_to_brain_ratio > 0.05:
        evidence += 1
    if tumor_to_brain_ratio > 0.10:
        evidence += 1
    if evidence == 0:
        label = 'no significant mass effect indicators'
    elif evidence == 1:
        label = 'mild mass effect possible'
    else:
        label = 'substantial mass effect likely'
    return {
        'brain_symmetry_iou': symmetry_iou,
        'tumor_to_brain_area_ratio': tumor_to_brain_ratio,
        'tumor_to_brain_centroid_distance_px': dist_to_brain_centroid,
        'tumor_centroid_relative_to_brain_radius': rel_to_brain_radius,
        'mass_effect_label': label,
    }


def _internal_architecture(image_rgb, mask) -> dict:
    """Single-channel necrosis / rim-enhancement / hemorrhage / calcification
    hints. Works without modality split by reading intensity percentiles inside
    the tumor mask. These are HEURISTICS - they cite numbers, not diagnoses.
    """
    if (mask > 0).sum() < 30:
        return {'note': 'tumor too small for architecture analysis'}
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    in_mask = gray[mask > 0]
    mean_i = float(in_mask.mean())

    # Necrosis-like fraction: pixels well below the tumor mean (potential cavity).
    p25 = float(np.percentile(in_mask, 25))
    p75 = float(np.percentile(in_mask, 75))
    necrosis_thresh = max(p25 - 10, 0)
    necrosis_fraction = float((in_mask < necrosis_thresh).mean()) if necrosis_thresh > 0 else 0.0

    # Rim vs core ratio: distance-transform partition.
    dist_inside = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    max_d = float(dist_inside.max())
    if max_d <= 0:
        rim_vs_core = 1.0
        rim_label = 'undeterminable'
    else:
        rim_band = (mask > 0) & (dist_inside < max_d * 0.3)
        core_band = (mask > 0) & (dist_inside >= max_d * 0.6)
        rim_vals = gray[rim_band]
        core_vals = gray[core_band]
        if rim_vals.size and core_vals.size:
            rim_vs_core = float(rim_vals.mean() / max(core_vals.mean(), 1.0))
        else:
            rim_vs_core = 1.0
        if rim_vs_core > 1.2:
            rim_label = 'rim-enhancing pattern (rim brighter than core)'
        elif rim_vs_core < 0.85:
            rim_label = 'inverse pattern (core brighter than rim)'
        else:
            rim_label = 'homogeneous (rim ~ core)'

    # Hemorrhage-like blobs: clusters of pixels at >P95 inside the mask.
    p95 = float(np.percentile(in_mask, 95))
    high_mask = ((gray >= p95) & (mask > 0)).astype(np.uint8) * 255
    n_high, _, stats_h, _ = cv2.connectedComponentsWithStats(high_mask, connectivity=8)
    hemorrhage_blob_count = int(sum(1 for i in range(1, n_high) if stats_h[i, cv2.CC_STAT_AREA] >= 4))
    # Calcification-like blobs: clusters at <P5 inside the mask.
    p5 = float(np.percentile(in_mask, 5))
    low_mask = ((gray <= p5) & (mask > 0)).astype(np.uint8) * 255
    n_low, _, stats_l, _ = cv2.connectedComponentsWithStats(low_mask, connectivity=8)
    calcification_blob_count = int(sum(1 for i in range(1, n_low) if stats_l[i, cv2.CC_STAT_AREA] >= 4))

    return {
        'mean_intensity_in_tumor': mean_i,
        'p25_intensity_in_tumor': p25,
        'p75_intensity_in_tumor': p75,
        'necrosis_like_fraction_single_channel': necrosis_fraction,
        'rim_vs_core_intensity_ratio': rim_vs_core,
        'rim_pattern_label': rim_label,
        'hyperdense_blob_count_inside_tumor': hemorrhage_blob_count,
        'hypodense_blob_count_inside_tumor': calcification_blob_count,
    }


def _grade_evidence(features_so_far: dict) -> dict:
    """Composite WHO-grade evidence score (NOT a diagnosis).

    Builds a 0..1 score from radiographic features classically associated with
    higher-grade tumors: necrotic appearance, marked heterogeneity, irregular
    shape, mass effect, large volume. Each component is documented so the
    narrative can cite exactly why the score is what it is.

    This is a research heuristic - it correlates with grade in published series
    but is NOT a substitute for histology.
    """
    g = features_so_far.get('geometry') or {}
    t = features_so_far.get('texture') or {}
    arch = features_so_far.get('internal_architecture') or {}
    morph = features_so_far.get('morphology') or {}
    me = features_so_far.get('mass_effect') or {}
    comps = features_so_far.get('components') or {}
    mm = features_so_far.get('multimodal') or {}

    pts = []  # (component_name, value 0..1, weight, explanation)
    # Necrosis (single-channel or multimodal).
    nec = arch.get('necrosis_like_fraction_single_channel', 0) or 0
    nec_mm = 1.0 if mm.get('necrosis_likely') else 0.0
    nec_score = max(min(nec / 0.25, 1.0), nec_mm)
    pts.append(('necrosis', nec_score, 0.25,
                f'necrosis_like_fraction={nec:.2f}'
                + (', multimodal necrosis flag set' if nec_mm > 0 else '')))
    # Heterogeneity.
    het = t.get('heterogeneity_score', 0) or 0
    zones = morph.get('internal_intensity_zones', 1) or 1
    het_score = min(het / 0.5, 1.0) * 0.5 + min((zones - 1) / 2.0, 1.0) * 0.5
    pts.append(('heterogeneity', het_score, 0.20,
                f'heterogeneity_score={het:.2f}, intensity_zones={zones}'))
    # Irregular margin.
    sol = g.get('solidity', 1) or 1
    circ = g.get('circularity', 1) or 1
    irreg_score = max(0.0, min(1.0, (1.0 - sol) / 0.30)) * 0.5 + max(0.0, min(1.0, (1.0 - circ) / 0.50)) * 0.5
    pts.append(('irregular_margin', irreg_score, 0.15,
                f'solidity={sol:.2f}, circularity={circ:.2f}'))
    # Mass effect.
    me_label = me.get('mass_effect_label', '') or ''
    me_score = 0.0
    if 'substantial' in me_label:
        me_score = 1.0
    elif 'mild' in me_label:
        me_score = 0.5
    pts.append(('mass_effect', me_score, 0.15, f'label="{me_label}"'))
    # Large volume.
    area = g.get('area_px', 0) or 0
    vol_score = min(area / 5000.0, 1.0)
    pts.append(('volume', vol_score, 0.10, f'area_px={area}'))
    # Edema halo.
    edema = 1.0 if mm.get('edema_likely') else 0.0
    pts.append(('peritumoral_edema', edema, 0.10, f'edema_likely={bool(mm.get("edema_likely"))}'))
    # Multifocality.
    multi = 1.0 if comps.get('multifocal') else 0.0
    pts.append(('multifocality', multi, 0.05,
                f'n_components={comps.get("n_components", 0)}, multifocal={bool(comps.get("multifocal"))}'))

    total_weight = sum(w for _, _, w, _ in pts)
    score = sum(v * w for _, v, w, _ in pts) / max(total_weight, 1e-6)
    if score >= 0.6:
        band = 'high (features classically associated with HGG)'
    elif score >= 0.35:
        band = 'intermediate (mixed features)'
    else:
        band = 'low (features more consistent with LGG / benign appearance)'
    return {
        'score_0_to_1': float(score),
        'evidence_band': band,
        'components': [
            {'name': n, 'value_0_to_1': float(v), 'weight': float(w), 'detail': d}
            for (n, v, w, d) in pts
        ],
        'disclaimer': 'Heuristic radiographic score. NOT a histological grade.',
    }


def _quality_assessment(image_rgb, brain_mask, mask) -> dict:
    """Image-quality signals that affect how much trust to put in the output.

    - Brain mask area (very small = poor skull stripping or wrong modality).
    - Average brain intensity (very dark = under-exposed scan).
    - Tumor-relative-to-brain (very small mask < ~50 px could be noise).
    - Mask boundary smoothness (very ragged = uncertain seg).
    """
    h, w = brain_mask.shape
    brain_area = int((brain_mask > 0).sum())
    tumor_area = int((mask > 0).sum())
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    brain_mean = float(gray[brain_mask > 0].mean()) if brain_area else 0.0
    notes = []
    if brain_area < 0.05 * h * w:
        notes.append('brain mask is unusually small; skull-strip / modality mismatch possible')
    if brain_mean < 30:
        notes.append('average brain intensity is very low; under-exposed scan')
    if 0 < tumor_area < 50:
        notes.append('predicted mask is very small; could be noise')
    # Mask boundary smoothness: ratio of perimeter to expected perimeter of an
    # equivalent-area circle. Very ragged = high ratio.
    rag = 1.0
    if tumor_area > 30:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            peri = float(cv2.arcLength(max(contours, key=cv2.contourArea), True))
            expected = 2 * math.pi * math.sqrt(tumor_area / math.pi)
            rag = float(peri / max(expected, 1.0))
    if rag > 1.8:
        notes.append(f'mask boundary is unusually ragged (perimeter-to-expected ratio {rag:.2f})')
    return {
        'brain_area_px': brain_area,
        'tumor_area_px': tumor_area,
        'mean_brain_intensity': brain_mean,
        'mask_perimeter_to_expected_ratio': rag,
        'quality_warnings': notes,
        'overall_quality_label': 'good' if not notes else 'review-required',
    }


def _overall_confidence(features_so_far: dict) -> dict:
    """Net confidence we project to the user (0..1), with the inputs that
    drove it so the narrative can cite them.

    Drivers:
      + classifier inter-model agreement (mean prob × agreement boost)
      + grad-cam ↔ segmentation alignment
      + image quality (no warnings)
      - very small / very ragged mask -> deflate
    """
    mb = features_so_far.get('model_behavior') or {}
    q = features_so_far.get('quality') or {}
    g = features_so_far.get('geometry') or {}
    score = 0.5  # neutral prior

    mean_p = mb.get('mean_probability_tumor')
    agree = mb.get('models_agreement')
    if isinstance(mean_p, (int, float)):
        # Map mean_p ∈ [0,1] to ±0.25 around 0.5.
        score = 0.5 + (float(mean_p) - 0.5) * 0.5
        if agree == 'unanimous':
            score = max(0.0, min(1.0, score + 0.10))
        elif agree == 'mixed':
            score = max(0.0, min(1.0, score - 0.10))

    if mb.get('gradcam_segmentation_aligned'):
        score = min(1.0, score + 0.10)
    elif mb.get('gradcam_to_segmentation_distance_px') is not None:
        score = max(0.0, score - 0.05)

    warnings = q.get('quality_warnings') or []
    if warnings:
        score = max(0.0, score - 0.05 * len(warnings))

    if 0 < (g.get('area_px') or 0) < 50:
        score = max(0.0, score - 0.10)

    if score >= 0.80:
        band = 'high'
        action = 'Findings are well-supported by classifier and segmentation evidence.'
    elif score >= 0.60:
        band = 'moderate'
        action = 'Findings are plausible; radiologist review recommended before any decision.'
    elif score >= 0.40:
        band = 'low'
        action = 'Output is uncertain; treat as an exploratory cue, not a finding.'
    else:
        band = 'very-low'
        action = 'Strong evidence is lacking; do not rely on this output.'
    return {
        'score_0_to_1': float(score),
        'band': band,
        'action_recommendation': action,
    }


def _model_behavior(classifier_results, mask, gradcam_heatmap) -> dict:
    """Per-model probabilities, inter-model agreement, Grad-CAM alignment."""
    out: dict = {'classifier_results': classifier_results or {}}
    if classifier_results:
        probs = []
        for name, r in classifier_results.items():
            if isinstance(r, dict) and isinstance(r.get('probability'), (int, float)):
                probs.append((name, float(r['probability'])))
        if probs:
            ps = [p for _, p in probs]
            out['mean_probability_tumor'] = float(np.mean(ps))
            out['probability_std_across_models'] = float(np.std(ps))
            out['models_agreement'] = (
                'unanimous' if max(ps) - min(ps) < 0.05
                else 'consistent' if max(ps) - min(ps) < 0.15
                else 'mixed'
            )
            out['per_model_probabilities'] = dict(probs)

    if gradcam_heatmap is not None and (mask > 0).any():
        h, w = mask.shape[:2]
        cam = np.asarray(gradcam_heatmap, dtype=np.float32)
        if cam.shape != (h, w):
            cam = cv2.resize(cam, (w, h))
        if cam.max() > 1.0:
            cam = cam / 255.0
        py, px = np.unravel_index(int(np.argmax(cam)), cam.shape)
        ys, xs = np.where(mask > 0)
        mcx, mcy = float(xs.mean()), float(ys.mean())
        dist = float(math.hypot(px - mcx, py - mcy))
        diag = float(math.hypot(h, w))
        out['gradcam_peak_xy_px'] = (int(px), int(py))
        out['segmentation_centroid_xy_px'] = (mcx, mcy)
        out['gradcam_to_segmentation_distance_px'] = dist
        out['gradcam_segmentation_aligned'] = bool(dist < 0.10 * diag)

        # Overlap between thresholded grad-cam and mask.
        cam_bin = (cam > 0.5).astype(np.uint8)
        mask_bin = (mask > 0).astype(np.uint8)
        inter = int((cam_bin & mask_bin).sum())
        union = int((cam_bin | mask_bin).sum())
        out['gradcam_mask_iou'] = float(inter / union) if union else 0.0

    return out
