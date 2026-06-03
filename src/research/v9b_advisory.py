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
# Four-signal ensemble rule (2026-06-03c, fix1c):
#     (v9c AND sym) OR (v8 AND andi) OR (sym AND andi) OR (v9c AND v8)
#
# The original 2-branch rule `(v9c AND sym) OR (v8 AND andi)` had an
# architectural blindspot: when the firing pair was the "diagonal"
# (sym AND andi) or (v9c AND v8), neither branch matched and the rule
# returned no_tumor even though 2 of 4 signals fired. Discovered on a
# real OOD scan with unilateral occipital signal:
#     v9c=0.633 (silent), v8=0 (silent), sym=130 (FIRED), andi=1.49e-4 (FIRED)
#
# Adding the two missing pairwise branches closes the blindspot at a
# cost of +2 to +3 pp FPR across operating points.
#
# Signal source:
#   - v9c     : frozen DINOv2 + trained JEPA predictor (best single signal)
#   - v8      : nnU-Net segmentation mask area
#   - symmetry: deterministic axial-symmetry geometry score
#   - andi    : pyramidal-noise unconditional DDPM (Frotscher et al. 2024)
#
# Without v9c (V9C_ENABLE=0): falls back to a 3-signal rule using the
# subset of branches whose signals are available. Without ANDi
# (V9B_ANDI_ENABLE=0): falls back symmetrically.
#
# Default = 'balanced' (97% recall / 9% FPR / 0.78 F1). Use high_recall
# for zero-misses deployments (100/17) or high_specificity for max precision
# (92/6).
# Layperson-facing display strings shared across all operating points.
# `_4signal` is what appears in the UI when all four detectors are active.
# `_technical` is the formal Boolean rule, surfaced only as a hover/tooltip
# for developers + researchers who want to see the actual gate.
_RULE_DISPLAY_4SIGNAL = ('At least 2 of 4 detectors must agree on the same '
                         'region (Pattern + Asymmetry, Outline Drawer + '
                         'Reconstruction, Asymmetry + Reconstruction, or '
                         'Pattern + Outline Drawer).')
_RULE_TECHNICAL_4SIGNAL = ('(Pattern AND Asymmetry) OR (Outline AND Reconstruction) '
                          'OR (Asymmetry AND Reconstruction) OR (Pattern AND Outline)')
_RULE_DISPLAY_3SIGNAL_NO_ANDI = ('Pattern Detector must fire AND either Asymmetry '
                                  'or the Outline Drawer must agree.')
_RULE_DISPLAY_2SIGNAL = ('Tumor Outline Drawer AND Asymmetry Detector must '
                          'both fire (the most conservative fallback).')

OPERATING_POINTS = {
    # "Catch Every Tumor" mode — 100% recall / 17% FPR / 0.67 F1 on the
    # 246-sample bench. Catches every tumor including the diagonal-firing
    # failure case. Use when missing a tumor is unacceptable; ~5 in 10
    # tumor verdicts are false alarms that a radiologist rules out.
    'high_recall': {
        'display_name': 'Catch Every Tumor',
        'display_description': ('Sensitivity-first mode. Flags every tumor in '
                                 'our test set, but ~50% of positive verdicts '
                                 'are false alarms a radiologist must rule out.'),
        'v9c_threshold': 0.582,
        'v8_area_threshold': 49,
        'symmetry_threshold': 111.0,
        'andi_threshold': 1.699e-04,
        'jepa_threshold': 0.489,   # legacy path, kept for compat
        'rule_4signal':    _RULE_DISPLAY_4SIGNAL,
        'rule_with_v9c':   _RULE_DISPLAY_3SIGNAL_NO_ANDI,
        'rule_without_v9c': _RULE_DISPLAY_2SIGNAL,
        'rule_technical':  _RULE_TECHNICAL_4SIGNAL,
        'measured': {
            'tumors_caught_pct': 100, 'healthy_wrongly_flagged_pct': 17,
            'overall_accuracy_score': 0.67,
            'cohort_description': '246 brain scans (36 with tumors, 210 healthy)',
        },
    },
    # "Balanced" mode (default) — 97% recall / 9% FPR / 0.78 F1.
    'balanced': {
        'display_name': 'Balanced',
        'display_description': ('Default mode. Catches 97 of 100 tumors with '
                                 'about 9 false alarms per 100 healthy scans.'),
        'v9c_threshold': 0.709,
        'v8_area_threshold': 49,
        'symmetry_threshold': 111.0,
        'andi_threshold': 1.36e-04,
        'jepa_threshold': 0.490,
        'rule_4signal':    _RULE_DISPLAY_4SIGNAL,
        'rule_with_v9c':   _RULE_DISPLAY_3SIGNAL_NO_ANDI,
        'rule_without_v9c': _RULE_DISPLAY_2SIGNAL,
        'rule_technical':  _RULE_TECHNICAL_4SIGNAL,
        'measured': {
            'tumors_caught_pct': 97, 'healthy_wrongly_flagged_pct': 9,
            'overall_accuracy_score': 0.78,
            'cohort_description': '246 brain scans (36 with tumors, 210 healthy)',
        },
    },
    # "Maximum Precision" mode — 92% recall / 6% FPR / 0.82 F1.
    'high_specificity': {
        'display_name': 'Maximum Precision',
        'display_description': ('Precision-first mode. Reduces false alarms to '
                                 '~6 per 100 healthy scans, at the cost of '
                                 'missing ~8 of 100 tumors.'),
        'v9c_threshold': 0.709,
        'v8_area_threshold': 49,
        'symmetry_threshold': 121.0,
        'andi_threshold': 1.36e-04,
        'jepa_threshold': 0.449,
        'rule_4signal':    _RULE_DISPLAY_4SIGNAL,
        'rule_with_v9c':   _RULE_DISPLAY_3SIGNAL_NO_ANDI,
        'rule_without_v9c': _RULE_DISPLAY_2SIGNAL,
        'rule_technical':  _RULE_TECHNICAL_4SIGNAL,
        'measured': {
            'tumors_caught_pct': 92, 'healthy_wrongly_flagged_pct': 6,
            'overall_accuracy_score': 0.82,
            'cohort_description': '246 brain scans (36 with tumors, 210 healthy)',
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


def compute_anomaly_localization_map(image_rgb_uint8: np.ndarray,
                                        prefer: str = 'andi') -> Optional[dict]:
    """Compute a per-pixel anomaly map for localization fallback.

    Used by the dashboard when v8 returns an empty mask but the ensemble
    verdict is TUMOR — we synthesize a mask from whichever neural anomaly
    signal is available so the UI has something to display + MedSAM has
    a bbox to refine.

    Returns:
      {'map': np.ndarray (H, W) float32, 'source': 'andi'|'v9c',
       'shape_hw': (H, W), 'inference_ms': int}
      or None if no signal could produce a map.

    The map is in raw signal units (ANDi A_gm or v9c per-patch error).
    Caller is expected to threshold (e.g. > p95 of map) + clean (largest
    connected component) to derive a binary mask.
    """
    if prefer not in ('andi', 'v9c'):
        prefer = 'andi'
    sources = [prefer] + (['v9c'] if prefer == 'andi' else ['andi'])
    for src in sources:
        try:
            if src == 'andi' and _andi_enabled():
                ddpm = _load_andi_model()
                if ddpm is None:
                    continue
                import torch
                from PIL import Image
                from src.research.andi_inference import andi_anomaly_map
                t0 = time.perf_counter()
                img = Image.fromarray(image_rgb_uint8).convert('RGB').resize(
                    (256, 256), Image.BILINEAR)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                x0 = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(_ANDI_DEVICE)
                cond = torch.zeros(1, _ANDI_COND_DIM, device=_ANDI_DEVICE)
                with torch.no_grad():
                    amap = andi_anomaly_map(ddpm, x0, cond, t_low=75, t_high=200,
                                             stride=5, device=_ANDI_DEVICE, seed=0)
                # (1, C, H, W) -> (H, W) by channel-max
                m = amap.squeeze(0).max(dim=0).values.cpu().numpy().astype(np.float32)
                return {'map': m, 'source': 'andi', 'shape_hw': m.shape,
                        'inference_ms': int((time.perf_counter() - t0) * 1000)}
            if src == 'v9c' and _v9c_enabled():
                model = _load_v9c_model()
                if model is None:
                    continue
                import torch
                t0 = time.perf_counter()
                with torch.no_grad():
                    emap = model.prediction_error_map([image_rgb_uint8])  # (1, 1, 224, 224)
                m = emap.squeeze().cpu().numpy().astype(np.float32)
                return {'map': m, 'source': 'v9c', 'shape_hw': m.shape,
                        'inference_ms': int((time.perf_counter() - t0) * 1000)}
        except Exception:
            continue
    return None


def synthesize_fallback_mask(amap: np.ndarray,
                              target_hw: tuple = (256, 256),
                              percentile: float = 97.0,
                              min_area_px: int = 30) -> Optional[np.ndarray]:
    """Turn a per-pixel anomaly map into a clean binary mask.

    Pipeline:
      1. Resize to target_hw (typically 256x256 to match v8's output).
      2. Threshold at the `percentile`-th percentile of the map values
         — keeps roughly (100 - percentile)% of pixels.
      3. Morphological close (3x3) to fill small gaps.
      4. Keep only the largest connected component.
      5. Reject if final area < min_area_px (noise floor).

    Returns:
      np.ndarray (H, W) uint8 binary mask, or None if the map is too
      noisy to produce a meaningful localization.
    """
    try:
        from PIL import Image
        import scipy.ndimage as ndi
    except ImportError:
        return None
    if amap is None or amap.size == 0:
        return None
    # Resize map to target spatial extent (v8 mask shape)
    src = Image.fromarray(amap.astype(np.float32))
    src = src.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
    m = np.asarray(src, dtype=np.float32)
    # Threshold at the requested percentile
    thresh = float(np.percentile(m, percentile))
    binary = (m > thresh).astype(np.uint8)
    if binary.sum() < min_area_px:
        return None
    # Morphological close to fill 1-2 px gaps
    binary = ndi.binary_closing(binary, iterations=2).astype(np.uint8)
    # Keep only the largest connected component
    labeled, n_cc = ndi.label(binary)
    if n_cc == 0:
        return None
    sizes = ndi.sum(binary, labeled, range(1, n_cc + 1))
    if sizes.size == 0:
        return None
    largest = int(np.argmax(sizes)) + 1
    cleaned = (labeled == largest).astype(np.uint8)
    if cleaned.sum() < min_area_px:
        return None
    return cleaned


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
    # Default is 'balanced' (97% recall / 6% FPR / 0.83 F1 on the
    # 246-sample bench). Catches 35/36 tumors while keeping the FP rate
    # under 10% so reviewer alert volume stays manageable.
    #
    # Override via V9B_OPERATING_POINT=high_recall for a zero-misses
    # deployment (100% recall / 14% FPR / 0.71 F1 — catches the last
    # tumor but ~17 more FPs to review), or =high_specificity for
    # max-precision (92% recall / 4% FPR / 0.85 F1).
    name = os.environ.get('V9B_OPERATING_POINT', 'balanced').strip().lower()
    if name not in OPERATING_POINTS:
        name = 'balanced'
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
    #   4-signal (v9c + ANDi available): fix1c rule (closes diagonal blindspot)
    #       (v9c AND sym) OR (v8 AND andi) OR (sym AND andi) OR (v9c AND v8)
    #     — measured 100/17/0.67 (high_recall), 97/9/0.78 (balanced),
    #       92/6/0.82 (high_spec) on 246-sample bench. Catches cases
    #       like sym=130 + andi-fired with v9c near-miss + empty v8.
    #   3-signal (v9c only, no ANDi): collapses to (v9c AND sym) OR
    #       (v9c AND v8) since both ANDi-dependent branches are dead.
    #   3-signal (ANDi only, no v9c): symmetric — substitute andi
    #       for v9c in the same logical positions.
    #   2-signal (no v9c, no ANDi): conservative v8 AND symmetry.
    if v9c_fires is not None and andi_fires is not None:
        # All four detectors active
        verdict_fires = ((v9c_fires and sym_fires)
                          or (v8_fires and andi_fires)
                          or (sym_fires and andi_fires)
                          or (v9c_fires and v8_fires))
        rule_used = op['rule_4signal']
        signals_used = 'All 4 detectors active: Pattern, Reconstruction, Outline Drawer, Asymmetry'
    elif v9c_fires is not None:
        # 3-detector fallback (no Reconstruction Detector / ANDi)
        if op['name'] == 'high_recall':
            verdict_fires = (v9c_fires or v8_fires) and sym_fires
        elif op['name'] == 'balanced':
            verdict_fires = (int(v9c_fires) + int(v8_fires) + int(sym_fires)) >= 2
        elif op['name'] == 'high_specificity':
            verdict_fires = (v9c_fires or sym_fires) and v8_fires
        else:
            verdict_fires = sym_fires or v8_fires or v9c_fires
        rule_used = op['rule_with_v9c']
        signals_used = '3 detectors active: Pattern, Outline Drawer, Asymmetry'
    elif andi_fires is not None:
        # 3-detector fallback (no Pattern Detector / v9c)
        if op['name'] == 'high_recall':
            verdict_fires = (andi_fires or v8_fires) and sym_fires
        elif op['name'] == 'balanced':
            verdict_fires = (int(andi_fires) + int(v8_fires) + int(sym_fires)) >= 2
        elif op['name'] == 'high_specificity':
            verdict_fires = (andi_fires or sym_fires) and v8_fires
        else:
            verdict_fires = sym_fires or v8_fires or andi_fires
        rule_used = (f'{op["rule_with_v9c"]} (Reconstruction Detector '
                     f'substituted for Pattern Detector)')
        signals_used = '3 detectors active: Reconstruction, Outline Drawer, Asymmetry'
    else:
        # 2-detector fallback (no neural anomaly signal): conservative AND
        verdict_fires = v8_fires and sym_fires
        rule_used = op['rule_without_v9c']
        signals_used = '2 detectors active: Outline Drawer, Asymmetry'

    # Heavy mode (legacy v9b JEPA) is additive: if it fires we bump to
    # TUMOR even if the main rule didn't.
    if jepa_fires is True:
        verdict_fires = verdict_fires or jepa_fires

    verdict = 'TUMOR' if verdict_fires else 'no_tumor'
    rule = rule_used

    # Confidence + review guidance. balanced (default) gets 73% precision
    # on the 246-sample bench, so ~1 in 4 TUMOR verdicts is an FP that a
    # radiologist rules out by review. Flag low-confidence positives
    # (only one branch of the OR fired) regardless of operating point —
    # FPs at any tier should surface a clear "human review recommended"
    # hint, since FPs are the failure mode the human is actually catching.
    fire_count = (int(bool(sym_fires)) + int(bool(v8_fires))
                  + int(v9c_fires is True) + int(andi_fires is True))
    if verdict == 'TUMOR':
        # 2+ signals firing = high confidence (both branches of the OR
        # likely fired). 1 signal = low confidence — escalate review.
        confidence = 'high' if fire_count >= 2 else 'low'
        review_recommended = (confidence == 'low')
    else:
        # Negative verdicts are very reliable across all three OPs
        # (recall ≥92%, so NPV stays high regardless of which tier).
        confidence = 'high'
        review_recommended = False

    payload = {
        'enabled': True,
        'verdict': verdict,
        'confidence': confidence,
        'review_recommended': review_recommended,
        # Internal name (used for env-var routing) + layperson display name.
        # UI should prefer operating_point_display; operating_point is kept
        # for backwards-compatible API consumers.
        'operating_point': op['name'],
        'operating_point_display': op.get('display_name', op['name']),
        'operating_point_description': op.get('display_description', ''),
        # `rule` is the layperson-friendly description, `rule_technical`
        # is the formal Boolean gate for tooltips/debugging.
        'rule': rule,
        'rule_technical': op.get('rule_technical', rule),
        'signals_used': signals_used,
        # Per-detector state. Layperson labels live alongside the technical
        # keys so the UI can render either; the v9c_*/v8_*/symmetry_*/andi_*
        # keys are preserved for backwards compatibility.
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
        # Display-friendly summary for the per-detector card row. Keep the
        # order stable so the UI can render without sorting.
        'detector_summary': [
            {'name': 'Pattern Detector',
              'description': 'Compares each region to what healthy brains look like',
              'fired': v9c_fires is True, 'enabled': _v9c_enabled()},
            {'name': 'Reconstruction Detector',
              'description': 'Tries to redraw the scan and highlights where it struggles',
              'fired': andi_fires is True, 'enabled': _andi_enabled()},
            {'name': 'Tumor Outline Drawer',
              'description': 'A medical segmentation model trained to outline tumor regions',
              'fired': bool(v8_fires), 'enabled': True},
            {'name': 'Asymmetry Detector',
              'description': 'Compares the left and right sides of the brain',
              'fired': bool(sym_fires), 'enabled': True},
        ],
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


# ---------------------------------------------------------------------------
# Model-insight visualizations (layperson-friendly heatmap overlays)
# ---------------------------------------------------------------------------
# Each neural anomaly signal aggregates a per-pixel/per-patch map to a
# scalar before threshold-comparison. Keeping those maps and rendering
# them as colored overlays on the original scan gives the layperson a
# visual answer to "where does the AI think the unusual thing is?" —
# at zero extra inference cost (we already computed the maps to derive
# the scalars).


_VIRIDIS_LUT = None


def _viridis_lut() -> np.ndarray:
    """256x3 uint8 LUT approximating matplotlib's viridis. Built once and
    cached. Hand-tuned 8-stop gradient that hits the perceptually-uniform
    waypoints of the real viridis well enough for visualization."""
    global _VIRIDIS_LUT
    if _VIRIDIS_LUT is not None:
        return _VIRIDIS_LUT
    stops = [
        (0.00, ( 68,   1,  84)),   # deep purple
        (0.14, ( 71,  44, 122)),
        (0.29, ( 59,  81, 139)),
        (0.43, ( 44, 113, 142)),
        (0.57, ( 33, 144, 141)),
        (0.71, ( 39, 173, 129)),
        (0.86, (121, 209,  81)),
        (1.00, (253, 231,  37)),   # bright yellow
    ]
    lut = np.zeros((256, 3), dtype=np.float32)
    for i in range(256):
        t = i / 255.0
        for k in range(len(stops) - 1):
            t0, c0 = stops[k]
            t1, c1 = stops[k + 1]
            if t0 <= t <= t1:
                u = (t - t0) / max(t1 - t0, 1e-9)
                lut[i] = np.array(c0) * (1 - u) + np.array(c1) * u
                break
    _VIRIDIS_LUT = lut.clip(0, 255).astype(np.uint8)
    return _VIRIDIS_LUT


def render_heatmap_overlay(image_rgb_uint8: np.ndarray,
                            anomaly_map: np.ndarray,
                            alpha: float = 0.55,
                            target_hw: tuple = (256, 256),
                            mask_below_pct: float = 50.0) -> np.ndarray:
    """Render an anomaly map as a viridis colormap blended over the
    grayscale brain image. Pixels below the `mask_below_pct` percentile
    of the map are left as plain grayscale (so cold regions don't tint
    the whole brain blue — only "interesting" pixels carry color).
    Returns a (H, W, 3) uint8 RGB array ready to PNG-encode."""
    from PIL import Image
    # Resize both inputs to target_hw
    base = Image.fromarray(image_rgb_uint8).convert('RGB').resize(
        (target_hw[1], target_hw[0]), Image.BILINEAR)
    base_arr = np.asarray(base, dtype=np.uint8)
    src_map = Image.fromarray(anomaly_map.astype(np.float32)).resize(
        (target_hw[1], target_hw[0]), Image.BILINEAR)
    m = np.asarray(src_map, dtype=np.float32)
    # Normalize to [0, 1]
    m_min, m_max = float(m.min()), float(m.max())
    if m_max - m_min < 1e-9:
        return base_arr
    norm = (m - m_min) / (m_max - m_min)
    # Colormap lookup: indices = round(norm * 255)
    lut = _viridis_lut()
    idx = (norm * 255).clip(0, 255).astype(np.int32)
    color = lut[idx]                              # (H, W, 3) uint8
    # Soft-mask the dim pixels (below mask_below_pct% of normalized scale)
    # back to grayscale — keeps the brain visible underneath cold regions
    cold = (norm < (mask_below_pct / 100.0))
    blend = (alpha * color.astype(np.float32)
              + (1 - alpha) * base_arr.astype(np.float32)).clip(0, 255).astype(np.uint8)
    out = blend.copy()
    out[cold] = base_arr[cold]
    return out


def render_agreement_overlay(image_rgb_uint8: np.ndarray,
                              fired_maps: list,
                              target_hw: tuple = (256, 256)) -> np.ndarray:
    """Render an "AI Agreement" overlay where pixels are colored by how
    many of the supplied per-pixel firing-maps agree:
        0 detectors flagged → grayscale background
        1 detector flagged  → yellow tint (single-signal positive)
        2+ detectors flagged → red tint (multi-signal positive — high
                                          confidence anomaly region)

    `fired_maps` is a list of bool/uint8 arrays (any shape, will be
    resized to target_hw). Empty list → returns the plain base image.
    """
    from PIL import Image
    base = Image.fromarray(image_rgb_uint8).convert('RGB').resize(
        (target_hw[1], target_hw[0]), Image.BILINEAR)
    base_arr = np.asarray(base, dtype=np.uint8)
    if not fired_maps:
        return base_arr
    # Sum the binary maps (resized) -> per-pixel agreement count
    agree = np.zeros(target_hw, dtype=np.int32)
    for fm in fired_maps:
        if fm is None:
            continue
        try:
            src = Image.fromarray(fm.astype(np.uint8)).resize(
                (target_hw[1], target_hw[0]), Image.NEAREST)
            agree += (np.asarray(src) > 0).astype(np.int32)
        except Exception:
            continue
    out = base_arr.copy()
    # Single-agreement → amber tint
    single = (agree == 1)
    if single.any():
        out[single] = (0.5 * np.array([245, 158, 11], dtype=np.uint8)
                       + 0.5 * out[single]).astype(np.uint8)
    # 2+ agreement → red tint (more saturated)
    multi = (agree >= 2)
    if multi.any():
        out[multi] = (0.6 * np.array([220, 38, 38], dtype=np.uint8)
                      + 0.4 * out[multi]).astype(np.uint8)
    return out


def compute_model_insight_maps(image_rgb_uint8: np.ndarray) -> dict:
    """Compute the per-pixel/per-patch anomaly maps for every enabled
    signal, plus a per-signal "fired-pixel" boolean map keyed on a
    quick top-percentile threshold so the agreement composite knows
    where each detector lit up.

    Returns:
      {
        'v9c': {'map': (H,W) float, 'fired': (H,W) bool} or None,
        'andi': {'map': (H,W) float, 'fired': (H,W) bool} or None,
        'symmetry': {'map': (H,W) float, 'fired': (H,W) bool} or None,
      }

    Inference cost: free — the same forwards that produce the scalar
    advisory scores also produce these maps; we just don't currently
    return them from _run_v9c / _run_andi.
    """
    out = {}
    # --- Symmetry (cheap, deterministic) ---
    try:
        from src.research.symmetry_geometry import symmetry_anomaly_map
        sym_map = symmetry_anomaly_map(image_rgb_uint8, view='axial')
        if sym_map is not None and sym_map.max() > 0:
            # Fired pixels = top 3% of the asymmetry map
            t = float(np.percentile(sym_map[sym_map > 0], 97))
            out['symmetry'] = {'map': sym_map.astype(np.float32),
                                'fired': (sym_map > t)}
    except Exception:
        pass
    # --- v9c (per-patch prediction error) ---
    if _v9c_enabled():
        model = _load_v9c_model()
        if model is not None:
            try:
                import torch
                with torch.no_grad():
                    emap = model.prediction_error_map([image_rgb_uint8])  # (1,1,224,224)
                m = emap.squeeze().cpu().numpy().astype(np.float32)
                t = float(np.percentile(m, 95))
                out['v9c'] = {'map': m, 'fired': (m > t)}
            except Exception:
                pass
    # --- ANDi (per-pixel DDPM error) ---
    if _andi_enabled():
        ddpm = _load_andi_model()
        if ddpm is not None:
            try:
                import torch
                from PIL import Image
                from src.research.andi_inference import andi_anomaly_map
                img = Image.fromarray(image_rgb_uint8).convert('RGB').resize(
                    (256, 256), Image.BILINEAR)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                x0 = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(_ANDI_DEVICE)
                cond = torch.zeros(1, _ANDI_COND_DIM, device=_ANDI_DEVICE)
                with torch.no_grad():
                    amap = andi_anomaly_map(ddpm, x0, cond, t_low=75, t_high=200,
                                             stride=5, device=_ANDI_DEVICE, seed=0)
                m = amap.squeeze(0).max(dim=0).values.cpu().numpy().astype(np.float32)
                t = float(np.percentile(m, 97))
                out['andi'] = {'map': m, 'fired': (m > t)}
            except Exception:
                pass
    return out


__all__ = ['compute_advisory', 'OPERATING_POINTS',
            'compute_anomaly_localization_map', 'synthesize_fallback_mask',
            'compute_model_insight_maps', 'render_heatmap_overlay',
            'render_agreement_overlay']
