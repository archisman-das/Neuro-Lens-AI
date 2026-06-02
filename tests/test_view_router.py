"""Tests for src/research/view_router.py.

Covers:
  - View detection on synthetic axial/coronal/sagittal masks
  - Modality-hint override (DICOM SeriesDescription strings)
  - cascade_decision: trust_classifier=False bypasses suppression
  - cascade_decision: v8_strong overrides high-confidence no_tumor
  - cascade_decision: classic suppression still fires on axial when neither
    override applies
"""
from __future__ import annotations

import numpy as np

from src.research.view_router import (
    detect_view, cascade_decision, confidence_tier, ViewPolicy, POLICY,
)


def _make_synthetic(shape='square', size=256):
    """Make a synthetic 'brain' image with a known aspect-ratio profile.
    shape in {'square', 'tall', 'wide_asym'}"""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    if shape == 'square':
        # Symmetric, square-ish foreground -> should look axial
        img[60:200, 60:200] = 200
    elif shape == 'tall':
        # Tall (height > width) -> should look coronal
        img[20:240, 80:175] = 200
    elif shape == 'wide_asym':
        # Wide foreground bbox preserved (aspect > 1.10) + LEFT-vs-RIGHT
        # asymmetric internal pattern inside the bbox (avoids shrinking
        # the bbox by leaving the rim intensity intact).
        img[60:200, 20:220] = 200
        # Bright spot on left, dark hole on right -> strong lr_diff while
        # keeping bbox unchanged.
        img[80:180, 30:100] = 50     # darker patch on the left
        img[80:180, 150:210] = 250   # brighter patch on the right
    return img


def test_detect_axial_from_geometry():
    p = detect_view(_make_synthetic('square'))
    assert p.view == 'axial', f'expected axial, got {p.view}'
    assert p.threshold == POLICY['axial']['threshold']
    assert p.trust_classifier is True


def test_detect_coronal_from_geometry():
    p = detect_view(_make_synthetic('tall'))
    assert p.view == 'coronal', f'expected coronal, got {p.view}'
    assert p.trust_classifier is False


def test_detect_sagittal_from_geometry():
    p = detect_view(_make_synthetic('wide_asym'))
    assert p.view == 'sagittal', f'expected sagittal, got {p.view}'
    assert p.trust_classifier is False


def test_modality_hint_overrides_geometry():
    # Square image but with SAG hint -> sagittal policy
    p = detect_view(_make_synthetic('square'), modality_hint='T1_sagittal')
    assert p.view == 'sagittal'
    assert 'modality_hint' in p.reason


def test_modality_hint_coronal():
    p = detect_view(_make_synthetic('square'), modality_hint='T2_coronal')
    assert p.view == 'coronal'


def test_modality_hint_flair_is_axial():
    p = detect_view(_make_synthetic('tall'), modality_hint='FLAIR')
    assert p.view == 'axial'  # FLAIR defaults to axial acquisition


def test_modality_hint_dwi_is_axial():
    p = detect_view(_make_synthetic('tall'), modality_hint='DWI')
    assert p.view == 'axial'


def test_cascade_non_axial_bypasses_suppression():
    # Coronal view with high-confidence no_tumor from classifiers
    p_cor = ViewPolicy('coronal', 0.8, 0.20, False, 'test')
    verdict, reason = cascade_decision(
        seg_max_prob=0.5,
        seg_area_at_view_thresh=100,
        classifier_mean_p=0.05,
        classifier_band='high',
        view_policy=p_cor,
    )
    assert verdict == 'TUMOR', 'non-axial view must trust v8 over classifier'
    assert 'coronal' in reason


def test_cascade_v8_strong_overrides_classifier_no_tumor():
    p_ax = ViewPolicy('axial', 0.9, 0.30, True, 'test')
    verdict, reason = cascade_decision(
        seg_max_prob=0.95,
        seg_area_at_view_thresh=5000,
        classifier_mean_p=0.05,
        classifier_band='high',
        view_policy=p_ax,
    )
    assert verdict == 'TUMOR', 'v8 strong should override classifier suppression'
    assert 'v8_strong' in reason
    assert 'overrides_classifier_consensus' in reason


def test_cascade_classic_suppression_on_axial():
    p_ax = ViewPolicy('axial', 0.9, 0.30, True, 'test')
    verdict, reason = cascade_decision(
        seg_max_prob=0.4,        # weak seg, not strong enough to override
        seg_area_at_view_thresh=100,
        classifier_mean_p=0.05,
        classifier_band='high',
        view_policy=p_ax,
    )
    assert verdict == 'no_tumor', 'axial + weak seg + classifier no_tumor -> suppressed'
    assert 'classifier_consensus_no_tumor' in reason


def test_cascade_axial_tumor_consensus_passes():
    p_ax = ViewPolicy('axial', 0.9, 0.30, True, 'test')
    verdict, reason = cascade_decision(
        seg_max_prob=0.8,
        seg_area_at_view_thresh=400,
        classifier_mean_p=0.85,
        classifier_band='moderate',
        view_policy=p_ax,
    )
    assert verdict == 'TUMOR'
    assert 'classifier_consensus_tumor' in reason or 'v8_strong' in reason


def test_cascade_empty_seg_returns_no_tumor():
    p_cor = ViewPolicy('coronal', 0.6, 0.20, False, 'test')
    verdict, _reason = cascade_decision(
        seg_max_prob=0.1,
        seg_area_at_view_thresh=0,
        classifier_mean_p=None,
        classifier_band=None,
        view_policy=p_cor,
    )
    assert verdict == 'no_tumor'


def test_no_foreground_image_returns_unknown():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    p = detect_view(img)
    assert p.view == 'unknown'
    assert p.confidence == 0.0


# ----------------- confidence_tier tests ---------------------------------

def test_tier_high_when_strong_seg_and_strong_classifier():
    t = confidence_tier(seg_max_prob=0.95, seg_area_at_view_thresh=2000,
                         classifier_mean_p=0.80)
    assert t == 'high'


def test_tier_review_when_seg_max_below_threshold():
    # The dominant rule: seg_max < 0.75 -> review regardless of other signals
    t = confidence_tier(seg_max_prob=0.50, seg_area_at_view_thresh=2000,
                         classifier_mean_p=0.90)
    assert t == 'requires_review'


def test_tier_review_when_weak_classifier_and_small_mask():
    # Even with seg_max above 0.75, weak classifier + small mask -> review
    t = confidence_tier(seg_max_prob=0.80, seg_area_at_view_thresh=100,
                         classifier_mean_p=0.10)
    assert t == 'requires_review'


def test_tier_high_when_classifier_weak_but_mask_large():
    # Large mask rescues the prediction even if classifier is weak
    t = confidence_tier(seg_max_prob=0.85, seg_area_at_view_thresh=500,
                         classifier_mean_p=0.10)
    assert t == 'high'


def test_tier_high_when_classifier_strong_even_with_small_mask():
    # Strong classifier + strong seg -> high regardless of mask size
    t = confidence_tier(seg_max_prob=0.80, seg_area_at_view_thresh=80,
                         classifier_mean_p=0.60)
    assert t == 'high'


def test_tier_review_with_no_classifier_signal():
    # Missing classifier_mean_p shouldn't crash the tiering
    t = confidence_tier(seg_max_prob=0.60, seg_area_at_view_thresh=300,
                         classifier_mean_p=None)
    assert t == 'requires_review'   # seg_max < 0.75 -> review


def test_tier_boundary_at_seg_max_0_75():
    # exactly 0.75 -> high (rule is strict <)
    t_at = confidence_tier(seg_max_prob=0.75, seg_area_at_view_thresh=500,
                            classifier_mean_p=0.50)
    t_below = confidence_tier(seg_max_prob=0.749, seg_area_at_view_thresh=500,
                                classifier_mean_p=0.50)
    assert t_at == 'high'
    assert t_below == 'requires_review'
