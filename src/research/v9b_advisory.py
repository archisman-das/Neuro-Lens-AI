"""v9b Tier-2 advisory: normative-JEPA + DDPM ensemble verdict.

Wraps the v9b inference pipeline as a single function call that
produces a tumor/no_tumor advisory verdict for the dashboard. Designed
to be optional and slow-path: gated by V9B_ENABLE env var, default OFF.

Shipping config (chosen from the OOD Pareto frontier, 2026-06-02):
  Rule:    JEPA OR DDPM
  Thresholds: JEPA p95 > 0.427  OR  DDPM residual p95 > 2.018
  Measured: recall=89%  FPR=17%  on the 48-sample OOD bench

Why this exists alongside v8 + view-aware cascade:
  - v8 segmentation handles the 86% recall mainstream and is fast (~100ms)
  - v9b catches several OOD cases v8 misses (esp. UniData multimodal,
    Navoneel binary), at the cost of ~5s extra latency per request
  - Used as a *second opinion*: when v8 says no_tumor and v9b fires, the
    UI shows an amber "v9b advisory: anomaly suspected, radiologist
    review recommended" banner. When v8 fires AND v9b confirms, the
    verdict is high-confidence tumor.

Latency budget:
  - JEPA prediction_error_map: ~3.5s on RTX 4060, ~30s+ on CPU
  - DDPM ddim_sample (50 steps): ~1.5s GPU, ~30s+ CPU
  - Total: ~5s GPU, ~60s CPU -> DEFAULT OFF on Spaces, opt-in for local GPU
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# Module-level cached model so we don't reload weights per request
_V9B_MODEL = None
_V9B_MODEL_DEVICE = None
_V9B_LOAD_ERR: Optional[str] = None


# Shipping thresholds (from scripts/eval_ood_ensemble.py Pareto sweep)
JEPA_THRESHOLD = 0.427
DDPM_THRESHOLD = 2.018


def _is_enabled() -> bool:
    return os.environ.get('V9B_ENABLE', '0').strip().lower() in ('1', 'true', 'yes')


def _load_model_once():
    """Lazy-load V9BModel on first call. Returns None on any failure
    (missing weights, OOM, etc.) — the caller treats no-model as opt-out."""
    global _V9B_MODEL, _V9B_MODEL_DEVICE, _V9B_LOAD_ERR
    if _V9B_MODEL is not None:
        return _V9B_MODEL
    if _V9B_LOAD_ERR is not None:
        # We already tried and failed; don't keep retrying on every request.
        return None
    repo_root = Path(__file__).resolve().parents[2]
    jepa_ckpt = repo_root / 'v9b_artifacts' / 'v9b_jepa' / 'last.pt'
    stage2_ckpt = repo_root / 'v9b_artifacts' / 'v9b_stage2' / 'last.pt'
    if not jepa_ckpt.exists() or not stage2_ckpt.exists():
        _V9B_LOAD_ERR = (
            f'v9b weights not found at {jepa_ckpt} / {stage2_ckpt}. '
            'Set V9B_DOWNLOAD=1 to fetch them from HF Models on first boot, '
            'or place them manually.'
        )
        return None
    try:
        from src.research.v9b_model import V9BModel
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = V9BModel.from_checkpoints(
            str(jepa_ckpt), str(stage2_ckpt),
            conformal_json=None, image_size=256, device=device,
        )
        _V9B_MODEL = model
        _V9B_MODEL_DEVICE = device
        return model
    except Exception as exc:
        _V9B_LOAD_ERR = f'{type(exc).__name__}: {exc}'
        return None


def _preprocess(image_rgb_uint8: np.ndarray, device: str) -> torch.Tensor:
    """Image as uint8 H x W x 3 -> normalized (1, 3, 256, 256) tensor on device."""
    from PIL import Image
    img = Image.fromarray(image_rgb_uint8).convert('RGB').resize((256, 256), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)


def compute_advisory(image_rgb_uint8: np.ndarray,
                      *, run_ddpm: bool = True) -> Optional[dict]:
    """Compute v9b Tier-2 advisory verdict for a single image.

    Returns None if v9b is disabled / weights missing / inference failed.
    Otherwise returns a dict suitable for inclusion in /explain response:
        {
            'enabled': True,
            'verdict': 'TUMOR'|'no_tumor',
            'rule': 'JEPA OR DDPM',
            'jepa_p95': float, 'jepa_fired': bool, 'jepa_threshold': float,
            'ddpm_p95': float, 'ddpm_fired': bool, 'ddpm_threshold': float,
            'inference_ms': int,
            'measured_performance': {
                'oo d_recall': 0.89, 'ood_fpr': 0.17,
                'cohort': '48-sample OOD test bench (OpenNeuro + Ultralytics + Navoneel + UniData)'
            }
        }
    """
    if not _is_enabled():
        return None
    model = _load_model_once()
    if model is None:
        return {
            'enabled': False, 'reason': _V9B_LOAD_ERR or 'model not loaded',
        }

    import time
    t0 = time.perf_counter()
    try:
        x = _preprocess(image_rgb_uint8, _V9B_MODEL_DEVICE)
        with torch.no_grad():
            out = model.infer(
                x,
                combine_mode='weighted_sum',
                lambda_app=0.6, lambda_geo=0.4,
                ddpm_num_steps=50 if run_ddpm else 0,
            )
        app_map = out['appearance_anomaly'].squeeze().cpu().numpy()
        jepa_p95 = float(np.percentile(app_map, 95))
        ddpm_p95 = 0.0
        if run_ddpm and out['residual'] is not None:
            res = out['residual'].squeeze().cpu().numpy()
            ddpm_p95 = float(np.percentile(res, 95))
        jepa_fired = jepa_p95 > JEPA_THRESHOLD
        ddpm_fired = ddpm_p95 > DDPM_THRESHOLD
        verdict = 'TUMOR' if (jepa_fired or ddpm_fired) else 'no_tumor'
        return {
            'enabled': True,
            'verdict': verdict,
            'rule': 'JEPA OR DDPM',
            'jepa_p95': round(jepa_p95, 4),
            'jepa_fired': jepa_fired,
            'jepa_threshold': JEPA_THRESHOLD,
            'ddpm_p95': round(ddpm_p95, 4),
            'ddpm_fired': ddpm_fired,
            'ddpm_threshold': DDPM_THRESHOLD,
            'inference_ms': int((time.perf_counter() - t0) * 1000),
            'measured_performance': {
                'ood_recall': 0.89,
                'ood_fpr': 0.17,
                'cohort': '48-sample OOD (OpenNeuro + Ultralytics + Navoneel + UniData)',
            },
        }
    except Exception as exc:
        return {
            'enabled': True,
            'verdict': None,
            'error': f'{type(exc).__name__}: {exc}',
            'inference_ms': int((time.perf_counter() - t0) * 1000),
        }


__all__ = ['compute_advisory', 'JEPA_THRESHOLD', 'DDPM_THRESHOLD']
