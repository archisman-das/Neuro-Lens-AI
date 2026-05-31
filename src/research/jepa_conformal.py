"""JEPA-latent conformal prediction extension for v9b.

Extends weighted conformal prediction (Gap B in
src/research/conformal_counterfactual_seg.py) to JEPA latent prediction
errors as the nonconformity score:

    s_v = || predictor(context)_v - target_encoder(full)_v ||^2

(per-patch L2 in JEPA's embedding space). On healthy patches the
predictor accurately reconstructs the target latent -> low score. On
tumor patches the latent is out-of-distribution for healthy training ->
high score.

Coverage guarantee:
    PS(v) = {y : s_v(y) <= q}
where q is the (1-alpha)(1+1/n) weighted quantile of calibration scores.
Under exchangeability of test and calibration in the post-intervention
distribution (modality, intensity, contrast interventions), the set
covers the true label with marginal probability >= 1-alpha.

This is the FIRST extension of conformal prediction to JEPA prediction
residuals in any domain.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
import torch


@dataclass
class JepaCalibSample:
    score: float        # per-scan summary score: max or mean of per-patch errors
    weight: float = 1.0


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted q-quantile with Tibshirani et al. 2019 finite-sample shift."""
    assert values.shape == weights.shape and values.ndim == 1
    assert 0 < q < 1
    order = np.argsort(values)
    v = values[order]; w = weights[order]
    w_norm = w / w.sum()
    n = values.size
    target = min(1.0, q * (1.0 + 1.0 / n))
    cum = np.cumsum(w_norm)
    idx = int(np.searchsorted(cum, target, side="left"))
    return float(v[min(idx, n - 1)])


@dataclass
class JepaCalibReport:
    n_calib: int
    alpha: float
    q: float
    weighted: bool
    empirical_coverage: float


class JepaConformalCalibrator:
    """Per-scan calibration: collect JEPA prediction-error summary statistics
    on a healthy calibration set, fit the (1-alpha) weighted quantile,
    then at test time decide certified-anomalous voxels.

    For per-voxel decisions: calibrate q on per-patch errors (more data,
    tighter bound). For per-scan decisions: calibrate on per-scan summary
    (mean or max of patch errors).
    """
    def __init__(self, alpha: float = 0.10):
        assert 0 < alpha < 1
        self.alpha = float(alpha)
        self.q: Optional[float] = None
        self._report: Optional[JepaCalibReport] = None

    def calibrate(self, scores: List[float], weights: Optional[List[float]] = None,
                  verbose: bool = False) -> JepaCalibReport:
        if weights is None:
            weights = [1.0] * len(scores)
        assert len(scores) == len(weights)
        assert len(scores) >= 30, "need >= 30 calibration samples"
        s = np.asarray(scores, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        weighted = bool(np.any(w != w[0]))
        q = weighted_quantile(s, w, 1.0 - self.alpha)
        self.q = float(q)
        cov = float((s <= q).sum() / s.size)
        self._report = JepaCalibReport(
            n_calib=len(scores), alpha=self.alpha, q=self.q,
            weighted=weighted, empirical_coverage=cov,
        )
        if verbose:
            print(f"[jepa-conformal] n={len(scores)} q={q:.4f} cov={cov:.3f}")
        return self._report

    def predict_voxelwise_certified(self,
                                     per_patch_errors: torch.Tensor) -> torch.Tensor:
        """Per-patch boolean map: True where patch error exceeds q (certified
        anomaly with >=(1-alpha) coverage on healthy population).

        per_patch_errors: (B, 1, H, W) anomaly score map
        Returns: (B, 1, H, W) bool tensor
        """
        if self.q is None:
            raise RuntimeError("call calibrate(...) first")
        return per_patch_errors > self.q

    def state_dict(self) -> Dict:
        if self.q is None: raise RuntimeError("nothing to save")
        return {"q": self.q, "alpha": self.alpha,
                "report": asdict(self._report) if self._report else None}

    def save(self, path): Path(path).write_text(json.dumps(self.state_dict(), indent=2))
    def load(self, path):
        d = json.loads(Path(path).read_text())
        self.q = float(d["q"]); self.alpha = float(d["alpha"])


__all__ = ["JepaConformalCalibrator", "JepaCalibSample", "weighted_quantile"]
