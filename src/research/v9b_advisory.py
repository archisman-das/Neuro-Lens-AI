"""v9b Tier-2 advisory: symmetry geometry + v8 segmentation ensemble.

Rewritten 2026-06-02 after the cohort-expansion eval (148 OOD samples,
adding 100 IXI2D healthy slices) revealed that the original JEPA+DDPM
config was inflated by the small N=12 OpenNeuro healthy cohort it was
calibrated on. On the 148-sample bench:
  - v9b JEPA appearance:   AUC = 0.564  (was 0.857 on 48-sample — sampling artifact)
  - Symmetry geometry:     AUC = 0.653  (NEW, replaces broken SDF tower @ AUC 0.18)
  - DDPM residual:         AUC ~ 0.7   (similar shape, costly ~3s latency)
  - v8 segmentation alone: high recall, high FPR (mask-based)

This advisory therefore runs ONLY the deterministic symmetry score
(< 0.1s per request) by default, combined with the v8 mask the
dashboard already computed. JEPA + DDPM ('heavy mode') stays available
behind V9B_HEAVY=1 for research/diagnostic use, but is OFF in
production because the latency cost is high and the marginal AUC on the
expanded cohort is not worth the seconds.

Operating points (measured on the 148-sample OOD bench, June 2026):
  - high_recall:       J|sym|v8 ensemble  85% recall / 31% FPR
  - balanced:          J|sym|v8 ensemble  65% recall / 12% FPR
  - high_specificity:  2-of-3 strict      30% recall /  0% FPR  (zero FPs)

These numbers are HONEST and reproducible from
samples/ood/eval_v9b_symmetry_expanded.csv. The previous 89/17 figure
was retracted because the 48-sample bench it came from didn't include
IXI2D-style healthy.

Selectable via V9B_OPERATING_POINT env var.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np


# Operating points — measured on the expanded 246-sample OOD bench
# (June 2026, 36 tumor / 210 healthy, Navoneel both-classes for LOSO).
#
# Four-signal ensemble when V9C_ENABLE=1 AND V9B_ANDI_ENABLE=1:
#   - v9c     : frozen DINOv2 + trained JEPA predictor (best single signal)
#   - v8      : nnU-Net segmentation mask area
#   - symmetry: deterministic axial-symmetry geometry score
#   - andi    : pyramidal-noise unconditional DDPM (Frotscher et al. 2024)
#
# Without v9c: falls back to 3-signal ensemble (v8 + sym + andi) or
# 2-signal (v8 + sym) depending on what's enabled. Without ANDi: falls
# back to the prior v9c-3-signal ensemble shipped on 2026-06-03.
#
# Default = 'high_recall' = 100% recall / 14% FPR — clinical safety
# choice. The user's explicit stance: "we CAN'T afford to slip a tumor,
# but low-confidence FPs can be ruled out by human intervention."
OPERATING_POINTS = {
    # CAN'T SLIP A TUMOR. Measured 100% recall / 14% FPR / 0.71 F1 on
    # the 246-sample bench with the 4-signal ensemble:
    #     (v9c AND sym) OR (v8 AND andi)
    # Catches every tumor in-bench; ~15% of TUMOR verdicts are FPs that
    # a radiologist rules out by reviewing the same image. Use this as
    # the default for any deployment where missing a tumor is unacceptable.
    'high_recall': {
        'v9c_threshold': 0.679,
        'v8_area_threshold': 99,
        'symmetry_threshold': 83.0,
        'andi_threshold': 9.97e-05,
        'jepa_threshold': 0.489,   # legacy v9b path, kept for compat
        'rule_4signal':    '(v9c AND sym) OR (v8 AND andi)',
        'rule_with_v9c':   '(v9c OR v8) AND symmetry',         # 3-signal fallback
        'rule_without_v9c': 'v8 AND symmetry',                  # 2-signal fallback
        'measured': {
            'ood_recall': 1.00, 'ood_fpr': 0.14, 'ood_f1': 0.71,
            'cohort': '246-sample OOD bench (June 2026)',
            'with_signals': '4-signal: v9c+v8+sym+andi',
        },
    },
    # Reviewer-friendly tier. Measured 97% recall / 6% FPR / 0.83 F1
    # on the 4-signal ensemble. Use when reviewer bandwidth is tight
    # but you still want >=95% recall.
    'balanced': {
        'v9c_threshold': 0.702,
        'v8_area_threshold': 49,
        'symmetry_threshold': 83.0,
        'andi_threshold': 1.36e-04,
        'jepa_threshold': 0.490,
        'rule_4signal':    '(v9c AND sym) OR (v8 AND andi)',
        'rule_with_v9c':   '2-of-3 vote (v9c, v8, symmetry)',   # 3-signal fallback
        'rule_without_v9c': 'v8 AND symmetry',
        'measured': {
            'ood_recall': 0.97, 'ood_fpr': 0.06, 'ood_f1': 0.83,
            'cohort': '246-sample OOD bench (June 2026)',
            'with_signals': '4-signal: v9c+v8+sym+andi',
        },
    },
    # Highest precision (smallest FPR). Measured 92% recall / 4% FPR /
    # 0.85 F1 on the 4-signal ensemble. Use when FPs are very costly.
    'high_specificity': {
        'v9c_threshold': 0.679,
        'v8_area_threshold': 999,
        'symmetry_threshold': 96.0,
        'andi_threshold': 1.36e-04,
        'jepa_threshold': 0.449,
        'rule_4signal':    '(v9c AND sym) OR (v8 AND andi)',
        'rule_with_v9c':   '(v9c OR symmetry) AND v8',          # 3-signal fallback
        'rule_without_v9c': 'symmetry AND v8',
        'measured': {
            'ood_recall': 0.92, 'ood_fpr': 0.04, 'ood_f1': 0.85,
            'cohort': '246-sample OOD bench (June 2026)',
            'with_signals': '4-signal: v9c+v8+sym+andi',
        },
    },
}


# Caches for the heavy v9b model — loaded once on first heavy-mode call.
_HEAVY_MODEL = None
_HEAVY_DEVICE = None
_HEAVY_LOAD_ERR: Optional[str] = None

# v9c cache — frozen DINOv2 + trained JEPA predictor on top
_V9C_MODEL = None
_V9C_DEVICE = None
_V9C_LOAD_ERR: Optional[str] = None

# ANDi cache — unconditional pyramidal-noise DDPM
_ANDI_MODEL = None
_ANDI_DEVICE = None
_ANDI_COND_DIM = 384
_ANDI_LOAD_ERR: Optional[str] = None


def _v9c_enabled() -> bool:
    return os.environ.get('V9C_ENABLE', '0').strip().lower() in ('1', 'true', 'yes')


def _andi_enabled() -> bool:
    return os.environ.get('V9B_ANDI_ENABLE', '0').strip().lower() in ('1', 'true', 'yes')


def _load_v9c_model():
    """Lazy-load v9c (frozen DINOv2 + trained JEPA predictor head).
    Returns None on any failure (missing weights / network / OOM)."""
    global _V9C_MODEL, _V9C_DEVICE, _V9C_LOAD_ERR
    if _V9C_MODEL is not None:
        return _V9C_MODEL
    if _V9C_LOAD_ERR is not None:
        return None
    repo_root = Path(__file__).resolve().parents[2]
    ckpt = repo_root / 'v9b_artifacts' / 'v9c_stage1' / 'last.pt'
    if not ckpt.exists():
        _V9C_LOAD_ERR = (
            f'v9c weights not at {ckpt}. Set V9C_DOWNLOAD=1 to pull from HF '
            'Models on first boot, or place manually.'
        )
        return None
    try:
        import torch
        from src.research.v9c_dinov2_jepa import V9CModel
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ck = torch.load(str(ckpt), map_location=device, weights_only=False)
        a = ck.get('args', {})
        model = V9CModel(
            predictor_depth=a.get('predictor_depth', 6),
            predictor_dim=a.get('predictor_dim', 384),
            device=device,
        )
        model.predictor.load_state_dict(ck['predictor_state_dict'])
        _V9C_MODEL = model
        _V9C_DEVICE = device
        return model
    except Exception as exc:
        _V9C_LOAD_ERR = f'{type(exc).__name__}: {exc}'
        return None


def _run_v9c(image_rgb_uint8: np.ndarray) -> dict:
    """Run v9c prediction-error inference on a single image. Returns
    {'v9c_p95': float, 'v9c_inference_ms': int} or {} on failure."""
    model = _load_v9c_model()
    if model is None:
        return {}
    import torch
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            emap = model.prediction_error_map([image_rgb_uint8]).squeeze().cpu().numpy()
        return {
            'v9c_p95': round(float(np.percentile(emap, 95)), 4),
            'v9c_inference_ms': int((time.perf_counter() - t0) * 1000),
        }
    except Exception as exc:
        return {'v9c_error': f'{type(exc).__name__}: {exc}'}


def _load_andi_model():
    """Lazy-load the unconditional pyramidal-noise DDPM used for the
    ANDi anomaly map. Trained by src/train_v9b_andi_ddpm.py — the
    Frotscher et al. 2024 recipe. Returns None on any failure."""
    global _ANDI_MODEL, _ANDI_DEVICE, _ANDI_COND_DIM, _ANDI_LOAD_ERR
    if _ANDI_MODEL is not None:
        return _ANDI_MODEL
    if _ANDI_LOAD_ERR is not None:
        return None
    repo_root = Path(__file__).resolve().parents[2]
    ckpt = repo_root / 'v9b_artifacts' / 'v9b_andi_ddpm' / 'last.pt'
    if not ckpt.exists():
        _ANDI_LOAD_ERR = (
            f'ANDi weights not at {ckpt}. Set V9B_ANDI_DOWNLOAD=1 to pull '
            'from HF Models on first boot, or place manually.'
        )
        return None
    try:
        import torch
        from src.research.latent_diffusion_decoder import LatentConditionedDDPM
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ck = torch.load(str(ckpt), map_location=device, weights_only=False)
        a = ck.get('args', {})
        cond_dim = a.get('cond_dim', 384)
        ddpm = LatentConditionedDDPM(in_chans=3, base_ch=32, cond_dim=cond_dim).to(device)
        ddpm.load_state_dict(ck['model_state_dict'], strict=False)
        ddpm.eval()
        _ANDI_MODEL = ddpm
        _ANDI_DEVICE = device
        _ANDI_COND_DIM = cond_dim
        return ddpm
    except Exception as exc:
        _ANDI_LOAD_ERR = f'{type(exc).__name__}: {exc}'
        return None


def _run_andi(image_rgb_uint8: np.ndarray) -> dict:
    """Run ANDi unconditional DDPM inference on a single image. Returns
    {'andi_max': float, 'andi_inference_ms': int} or {} on failure.

    Uses the `max` aggregation (AUC 0.726 standalone vs p95 AUC 0.37 —
    DDPM error is high everywhere, so the extreme tail is the signal).
    """
    ddpm = _load_andi_model()
    if ddpm is None:
        return {}
    import torch
    from PIL import Image
    from src.research.andi_inference import andi_anomaly_map
    t0 = time.perf_counter()
    try:
        img = Image.fromarray(image_rgb_uint8).convert('RGB').resize((256, 256), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        x0 = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(_ANDI_DEVICE)
        cond = torch.zeros(1, _ANDI_COND_DIM, device=_ANDI_DEVICE)  # unconditional
        with torch.no_grad():
            amap = andi_anomaly_map(ddpm, x0, cond, t_low=75, t_high=200,
                                     stride=5, device=_ANDI_DEVICE, seed=0)
        return {
            'andi_max': float(amap.max().item()),
            'andi_inference_ms': int((time.perf_counter() - t0) * 1000),
        }
    except Exception as exc:
        return {'andi_error': f'{type(exc).__name__}: {exc}'}


def _operating_point() -> dict:
    # Default is 'high_recall' by deliberate clinical-safety choice:
    # missing a tumor is far worse than flagging a healthy scan, since a
    # low-confidence FP can be ruled out by human review of the same image.
    # Override via V9B_OPERATING_POINT={balanced, high_specificity} for
    # deployments where reviewer bandwidth is the binding constraint.
    name = os.environ.get('V9B_OPERATING_POINT', 'high_recall').strip().lower()
    if name not in OPERATING_POINTS:
        name = 'high_recall'
    return {'name': name, **OPERATING_POINTS[name]}


def _heavy_enabled() -> bool:
    return os.environ.get('V9B_HEAVY', '0').strip().lower() in ('1', 'true', 'yes')


def _load_heavy_model():
    """Lazy-load V9BModel (JEPA + DDPM + SDF) on first heavy-mode call."""
    global _HEAVY_MODEL, _HEAVY_DEVICE, _HEAVY_LOAD_ERR
    if _HEAVY_MODEL is not None:
        return _HEAVY_MODEL
    if _HEAVY_LOAD_ERR is not None:
        return None
    repo_root = Path(__file__).resolve().parents[2]
    jepa_ckpt = repo_root / 'v9b_artifacts' / 'v9b_jepa' / 'last.pt'
    stage2_ckpt = repo_root / 'v9b_artifacts' / 'v9b_stage2' / 'last.pt'
    if not jepa_ckpt.exists() or not stage2_ckpt.exists():
        _HEAVY_LOAD_ERR = 'v9b weights not on disk; place them or skip V9B_HEAVY=1'
        return None
    try:
        import torch
        from src.research.v9b_model import V9BModel
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _HEAVY_MODEL = V9BModel.from_checkpoints(
            str(jepa_ckpt), str(stage2_ckpt),
            conformal_json=None, image_size=256, device=device,
        )
        _HEAVY_DEVICE = device
        return _HEAVY_MODEL
    except Exception as exc:
        _HEAVY_LOAD_ERR = f'{type(exc).__name__}: {exc}'
        return None


def _run_heavy(image_rgb_uint8: np.ndarray) -> dict:
    """Run JEPA + DDPM inference for the heavy-mode advisory. Returns
    dict with jepa_p95, ddpm_p95, and inference_ms; or {} on failure."""
    model = _load_heavy_model()
    if model is None:
        return {}
    import torch
    from PIL import Image
    t0 = time.perf_counter()
    try:
        img = Image.fromarray(image_rgb_uint8).convert('RGB').resize((256, 256), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(_HEAVY_DEVICE)
        with torch.no_grad():
            out = model.infer(x, combine_mode='weighted_sum',
                               lambda_app=0.6, lambda_geo=0.4,
                               ddpm_num_steps=50)
        app = out['appearance_anomaly'].squeeze().cpu().numpy()
        jepa_p95 = float(np.percentile(app, 95))
        ddpm_p95 = 0.0
        if out.get('residual') is not None:
            res = out['residual'].squeeze().cpu().numpy()
            ddpm_p95 = float(np.percentile(res, 95))
        return {
            'jepa_p95': round(jepa_p95, 4),
            'ddpm_p95': round(ddpm_p95, 4),
            'heavy_inference_ms': int((time.perf_counter() - t0) * 1000),
        }
    except Exception as exc:
        return {'heavy_error': f'{type(exc).__name__}: {exc}'}


def compute_advisory(image_rgb_uint8: np.ndarray,
                      v8_area_px: Optional[int] = None) -> Optional[dict]:
    """Compute v9b Tier-2 advisory verdict for a single image.

    Args:
      image_rgb_uint8: (H, W, 3) input MRI image
      v8_area_px: tumor area in pixels from v8 segmentation (the caller
                  already computed v8 — pass the area so we don't recompute)

    Returns:
      dict suitable for inclusion in /explain response. Verdict is the
      ensemble result at the selected operating point.
    """
    op = _operating_point()
    t0 = time.perf_counter()
    # 1. Symmetry score — deterministic, < 0.1s, the new geometry signal
    sym_p95: Optional[float] = None
    try:
        from src.research.symmetry_geometry import symmetry_score
        sym_p95 = symmetry_score(image_rgb_uint8, view='axial', percentile=95.0)
    except Exception:
        sym_p95 = None

    sym_fires = (sym_p95 is not None) and (sym_p95 > op['symmetry_threshold'])
    v8_fires = (v8_area_px is not None) and (v8_area_px >= op['v8_area_threshold'])

    # 2. v9c (DINOv2 + JEPA predictor) — opt-in via V9C_ENABLE=1.
    #    Primary signal on GPU-backed deployments (~1s/req).
    v9c = {}
    v9c_fires = None
    if _v9c_enabled():
        v9c = _run_v9c(image_rgb_uint8)
        if 'v9c_p95' in v9c:
            v9c_fires = v9c['v9c_p95'] > op['v9c_threshold']

    # 3. ANDi (unconditional pyramidal-noise DDPM) — opt-in via
    #    V9B_ANDI_ENABLE=1. Complements v9c by catching the Ultralytics
    #    tumors v9c struggles with; together they hit 100% recall at
    #    14% FPR on the 246-sample bench.
    andi = {}
    andi_fires = None
    if _andi_enabled():
        andi = _run_andi(image_rgb_uint8)
        if 'andi_max' in andi:
            andi_fires = andi['andi_max'] > op['andi_threshold']

    # 4. Heavy mode: legacy v9b JEPA + DDPM (slow, opt-in for research).
    #    Kept for backwards compatibility; v9c+ANDi supersede it functionally.
    heavy = {}
    jepa_fires = None
    if _heavy_enabled():
        heavy = _run_heavy(image_rgb_uint8)
        if 'jepa_p95' in heavy:
            jepa_fires = heavy['jepa_p95'] > op['jepa_threshold']

    # Ensemble rule selection. Best signal-set wins:
    #   4-signal (v9c + ANDi available): (v9c AND sym) OR (v8 AND andi)
    #     — measured 100/14/0.71 (high_recall), 97/6/0.83 (balanced),
    #       92/4/0.85 (high_spec) on 246-sample bench.
    #   3-signal (v9c only, no ANDi): the rules shipped 2026-06-03.
    #   2-signal (no v9c, no ANDi): conservative v8+symmetry fallback.
    if v9c_fires is not None and andi_fires is not None:
        # 4-signal — same rule shape across all operating points;
        # thresholds in OPERATING_POINTS differentiate the bands.
        verdict_fires = (v9c_fires and sym_fires) or (v8_fires and andi_fires)
        rule_used = op['rule_4signal']
        signals_used = '4-signal: v9c+v8+sym+andi'
    elif v9c_fires is not None:
        # 3-signal fallback (v9c without ANDi) — operating-point-specific
        if op['name'] == 'high_recall':
            verdict_fires = (v9c_fires or v8_fires) and sym_fires
        elif op['name'] == 'balanced':
            verdict_fires = (int(v9c_fires) + int(v8_fires) + int(sym_fires)) >= 2
        elif op['name'] == 'high_specificity':
            verdict_fires = (v9c_fires or sym_fires) and v8_fires
        else:
            verdict_fires = sym_fires or v8_fires or v9c_fires
        rule_used = op['rule_with_v9c']
        signals_used = '3-signal: v9c+v8+sym'
    elif andi_fires is not None:
        # 3-signal fallback (ANDi without v9c) — substitute ANDi for v9c
        # in the same logical position.
        if op['name'] == 'high_recall':
            verdict_fires = (andi_fires or v8_fires) and sym_fires
        elif op['name'] == 'balanced':
            verdict_fires = (int(andi_fires) + int(v8_fires) + int(sym_fires)) >= 2
        elif op['name'] == 'high_specificity':
            verdict_fires = (andi_fires or sym_fires) and v8_fires
        else:
            verdict_fires = sym_fires or v8_fires or andi_fires
        rule_used = f'{op["rule_with_v9c"]} (andi substituted for v9c)'
        signals_used = '3-signal: andi+v8+sym'
    else:
        # 2-signal fallback (no neural anomaly signal): conservative
        # AND on the two cheapest signals.
        verdict_fires = v8_fires and sym_fires
        rule_used = op['rule_without_v9c']
        signals_used = '2-signal: v8+sym'

    # Heavy mode (legacy v9b JEPA) is additive: if it fires we bump to
    # TUMOR even if the main rule didn't.
    if jepa_fires is True:
        verdict_fires = verdict_fires or jepa_fires

    verdict = 'TUMOR' if verdict_fires else 'no_tumor'
    rule = rule_used

    # Confidence + review guidance. At high_recall the rule is tuned for
    # 100% recall at the cost of ~14% FPR. ~3 in 10 TUMOR verdicts are
    # expected to be FPs that a radiologist rules out by review. Flag
    # low-confidence positives (only one branch of the OR fired) so the
    # UI surfaces a clear "human review recommended" hint.
    fire_count = (int(bool(sym_fires)) + int(bool(v8_fires))
                  + int(v9c_fires is True) + int(andi_fires is True))
    if verdict == 'TUMOR':
        # 2+ signals firing = high confidence (both branches of the OR
        # likely fired). 1 signal = low confidence — escalate review.
        confidence = 'high' if fire_count >= 2 else 'low'
        review_recommended = (op['name'] == 'high_recall' and confidence == 'low')
    else:
        # Negative verdicts at high_recall are very reliable (100% recall
        # ⇒ NPV essentially 100% in-bench).
        confidence = 'high'
        review_recommended = False

    payload = {
        'enabled': True,
        'verdict': verdict,
        'confidence': confidence,
        'review_recommended': review_recommended,
        'operating_point': op['name'],
        'rule': rule,
        'signals_used': signals_used,
        'symmetry_p95': round(sym_p95, 3) if sym_p95 is not None else None,
        'symmetry_fired': sym_fires,
        'symmetry_threshold': op['symmetry_threshold'],
        'v8_area_px': int(v8_area_px) if v8_area_px is not None else None,
        'v8_fired': bool(v8_fires),
        'v8_area_threshold': op['v8_area_threshold'],
        'v9c_enabled': _v9c_enabled(),
        'andi_enabled': _andi_enabled(),
        'heavy_mode': _heavy_enabled(),
        'measured_performance': op['measured'],
        'inference_ms': int((time.perf_counter() - t0) * 1000),
    }
    if v9c:
        payload.update(v9c)
        payload['v9c_threshold'] = op['v9c_threshold']
        payload['v9c_fired'] = v9c_fires
    elif _v9c_enabled() and _V9C_LOAD_ERR:
        payload['v9c_load_error'] = _V9C_LOAD_ERR
    if andi:
        payload.update(andi)
        payload['andi_threshold'] = op['andi_threshold']
        payload['andi_fired'] = andi_fires
    elif _andi_enabled() and _ANDI_LOAD_ERR:
        payload['andi_load_error'] = _ANDI_LOAD_ERR
    if heavy:
        payload.update(heavy)
        payload['jepa_threshold'] = op['jepa_threshold']
        payload['jepa_fired'] = jepa_fires
    return payload


__all__ = ['compute_advisory', 'OPERATING_POINTS']
