"""View detection + per-view inference policy for brain MRI 2D slices.

Cheap rule-based view classifier (axial / coronal / sagittal) using brain
bbox aspect ratio and left-right symmetry. Then maps the detected view to:

  - a segmentation threshold (per-view, learned from the OOD threshold
    sweep in scripts/eval_ood_cascade.py)
  - whether to trust the binary classifier consensus
    (cnn+transfer+vit was trained on axial Kaggle 4-class; on non-axial
    inputs it's unreliable and over-suppresses true positives)

This is the "view-conditioning" and "per-view threshold routing"
mitigations from the OOD analysis. No training required — purely
geometric.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ViewPolicy:
    view: str                  # 'axial' | 'coronal' | 'sagittal' | 'unknown'
    confidence: float          # 0..1
    threshold: float           # v8 binarisation threshold for this view
    trust_classifier: bool     # whether to apply classifier consensus suppression
    reason: str                # human-readable explanation


# Calibrated on the OOD eval (scripts/eval_ood_cascade.py):
#   axial coronal-healthy at t=0.40 -> FP=8% (was 58% at 0.20)
#   axial UniData T1-axial-FFE at t<=0.20 -> 25%, hopeless either way
#   coronal UniData T1_coronal/T2_coronal: 100% at t<=0.20, 0% at t>=0.40
#   sagittal UniData T1_sagittal: 100% up to t=0.50, drops past 0.55
POLICY = {
    'axial':    {'threshold': 0.30, 'trust_classifier': True},
    'coronal':  {'threshold': 0.20, 'trust_classifier': False},
    'sagittal': {'threshold': 0.40, 'trust_classifier': False},
    'unknown':  {'threshold': 0.25, 'trust_classifier': True},
}


def detect_view(image_rgb: np.ndarray, modality_hint: Optional[str] = None) -> ViewPolicy:
    """Detect view from a single 2D RGB MRI slice.

    Decision tree (cheap, no model):
      1. Compute foreground bbox via intensity threshold.
      2. Compute LR symmetry score = mean|x - mirror(x)| / mean(x).
      3. Sagittal: aspect > 1.10 AND lr_diff > 0.30 (asymmetric profile)
      4. Coronal: aspect < 1.05 AND height ~= width ish but with vertical
         elongation cue from top-row foreground extent
      5. Axial: default (most common, square-ish, symmetric)

    modality_hint (e.g. "T1_sagittal", "FLAIR") overrides if it embeds a
    view keyword — we trust the DICOM header over geometry when present.
    """
    if modality_hint:
        h = modality_hint.lower()
        if 'sag' in h:
            p = POLICY['sagittal']
            return ViewPolicy('sagittal', 1.0, p['threshold'],
                              p['trust_classifier'], f'modality_hint={modality_hint}')
        if 'cor' in h:
            p = POLICY['coronal']
            return ViewPolicy('coronal', 1.0, p['threshold'],
                              p['trust_classifier'], f'modality_hint={modality_hint}')
        if 'tra' in h or 'ax' in h or 'flair' in h or 'dwi' in h:
            # FLAIR + DWI default to axial acquisition in clinical practice
            p = POLICY['axial']
            return ViewPolicy('axial', 0.9, p['threshold'],
                              p['trust_classifier'], f'modality_hint={modality_hint}')

    if image_rgb.ndim == 3:
        gray = image_rgb.mean(axis=-1).astype(np.float32)
    else:
        gray = image_rgb.astype(np.float32)

    # Foreground detection: 5% of max intensity is the brain/non-air boundary
    fg = gray > max(20.0, gray.max() * 0.05)
    if not fg.any():
        p = POLICY['unknown']
        return ViewPolicy('unknown', 0.0, p['threshold'],
                          p['trust_classifier'], 'no-foreground')

    rows = np.where(fg.any(axis=1))[0]
    cols = np.where(fg.any(axis=0))[0]
    rmin, rmax = int(rows.min()), int(rows.max())
    cmin, cmax = int(cols.min()), int(cols.max())
    bbox_h = rmax - rmin + 1
    bbox_w = cmax - cmin + 1
    aspect = bbox_w / max(1, bbox_h)

    # Symmetry inside the bbox
    bbox = gray[rmin:rmax + 1, cmin:cmax + 1]
    if bbox.shape[1] % 2 == 1:
        bbox = bbox[:, :-1]
    mirror = bbox[:, ::-1]
    lr_diff = float(np.mean(np.abs(bbox - mirror)) / (bbox.mean() + 1e-6))

    # Sagittal: asymmetric, elongated AP
    if lr_diff > 0.30 and aspect > 1.10:
        p = POLICY['sagittal']
        return ViewPolicy('sagittal', 0.8, p['threshold'], p['trust_classifier'],
                           f'asym(lr_diff={lr_diff:.2f})+elongated(aspect={aspect:.2f})')

    # Coronal: tall-ish, symmetric (brain visible from front: round + cerebellum at bottom)
    if aspect < 1.00:
        p = POLICY['coronal']
        return ViewPolicy('coronal', 0.6, p['threshold'], p['trust_classifier'],
                           f'tall(aspect={aspect:.2f}), sym(lr_diff={lr_diff:.2f})')

    # Default axial
    p = POLICY['axial']
    return ViewPolicy('axial', 0.7, p['threshold'], p['trust_classifier'],
                       f'square(aspect={aspect:.2f}), sym(lr_diff={lr_diff:.2f})')


def cascade_decision(*, seg_max_prob: float, seg_area_at_view_thresh: int,
                      classifier_mean_p: Optional[float],
                      classifier_band: Optional[str],
                      view_policy: ViewPolicy,
                      v8_override_min_area: int = 200,
                      v8_override_min_prob: float = 0.70,
                      min_tumor_area: int = 50) -> tuple:
    """Decide the final TUMOR/no_tumor verdict with view-aware policy.

    Returns (verdict, reason).

    Logic:
      1. Did v8 see something at the view-aware threshold? (area >= 50)
      2. If view says don't trust classifier (non-axial) -> use v8 alone.
      3. v8 confidence override: if v8 fired AND has high area+prob, do
         not let classifier consensus suppress it (the classifier ensemble
         is OOD-ignorant of DWI/FLAIR/non-axial, and the OOD eval showed
         it suppresses real tumors).
      4. Otherwise apply the original classifier suppression rule.
    """
    seg_says_tumor = seg_area_at_view_thresh >= min_tumor_area

    if not view_policy.trust_classifier:
        # Non-axial -> classifier is unreliable, defer entirely to v8.
        return (('TUMOR', f'v8@{view_policy.view}_view(threshold={view_policy.threshold})')
                if seg_says_tumor else
                ('no_tumor', f'v8_negative@{view_policy.view}'))

    # v8 confidence override: classifier cannot suppress a strong v8 hit.
    v8_strong = (seg_area_at_view_thresh >= v8_override_min_area
                  and seg_max_prob >= v8_override_min_prob)
    if v8_strong:
        return ('TUMOR',
                f'v8_strong(area={seg_area_at_view_thresh},pmax={seg_max_prob:.2f}) '
                f'overrides_classifier_consensus')

    # Original suppression rule for moderate v8 + axial view.
    if classifier_band in ('high', 'moderate') and classifier_mean_p is not None and classifier_mean_p <= 0.3:
        return ('no_tumor',
                f'classifier_consensus_no_tumor(mean_p={classifier_mean_p:.2f},{classifier_band})')
    if classifier_band in ('high', 'moderate') and classifier_mean_p is not None and classifier_mean_p >= 0.7:
        return ('TUMOR',
                f'classifier_consensus_tumor(mean_p={classifier_mean_p:.2f},{classifier_band})')

    # Mixed / no classifier signal -> trust v8.
    return (('TUMOR', f'v8@{view_policy.view}_mixed_classifier')
            if seg_says_tumor else
            ('no_tumor', f'v8_negative_mixed_classifier'))


def confidence_tier(*, seg_max_prob: float, seg_area_at_view_thresh: int,
                     classifier_mean_p: Optional[float]) -> str:
    """3-tier confidence for a TUMOR-labelled prediction.

    Returns 'high' (definitive tumor) or 'requires_review' (possible tumor,
    flag for human radiologist).

    Rule derived from confidence_band_analysis.py over 272 TUMOR-predicted
    samples (ID + OOD combined):
      - seg_max         AUC = 0.906 for TP-vs-FP separation
      - clf_mean        AUC = 0.800
      - seg_max+clf_mean AUC = 0.941 (combined)

    The chosen rule flags 76% of false positives while only flagging 8% of
    true positives — a ~10x more accurate way of saying "I don't know":

        requires_review iff:
            seg_max < 0.75
            OR (clf_mean < 0.30 AND seg_area < 200)

    UI impact: in the dashboard a 'requires_review' verdict produces an
    amber banner ("Possible finding — human review recommended") instead
    of the red "TUMOR detected" banner, while keeping the segmentation
    overlay visible for the reviewer.
    """
    cm = classifier_mean_p if classifier_mean_p is not None else -1.0
    low_seg = seg_max_prob < 0.75
    weak_classifier_and_small = cm < 0.30 and seg_area_at_view_thresh < 200
    if low_seg or weak_classifier_and_small:
        return 'requires_review'
    return 'high'


__all__ = ['detect_view', 'cascade_decision', 'confidence_tier',
            'ViewPolicy', 'POLICY']
