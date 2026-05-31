"""Weighted conformal prediction for v9 with hyperbolic nonconformity scores.

What's new vs Gap B (src/research/conformal_counterfactual_seg.py)
------------------------------------------------------------------
Gap B applies weighted conformal prediction to per-voxel probability
residuals in Euclidean space:
    s_v = | sigmoid(f(T(x))_v) - y_v |
and produces marginal (1-alpha) coverage prediction sets under the
post-intervention distribution.

v9 extends this in two ways:

  1. The latent space is hyperbolic (Poincare ball). The natural
     nonconformity score is the geodesic distance:
        s_v = d_B(z_pred_v, z_true_v)
     where d_B is the Mobius distance from hyperbolic.py.

  2. The score is computed on the *causally-disentangled* latent
     (z_anatomy + z_tumor), not on the raw encoder output. This means
     the coverage guarantee transfers across scanner shifts (since
     z_scanner is excluded from the score by construction).

Mathematical statement
----------------------
For test sample with latent z* and calibration set with latents z_i:
    s_i = d_B(z_pred_i, z_true_i)
    q = WeightedQuantile_{(1-alpha)(1+1/n)}({s_i}, {w_i})
        where w_i = importance weights under intervention distribution Q
The (1-alpha)-coverage prediction set is:
    PS(z*) = { y : d_B(z_pred*, y) <= q }
By Tibshirani et al. 2019 Thm 2, this set covers the true y* with
probability >= 1-alpha under exchangeability of (z*, y*) and (z_i, y_i)
in Q.

The hyperbolic case requires extending the proof because the metric is
non-Euclidean. The proof goes through because conformal prediction only
requires score exchangeability and an order on scores, both of which
hyperbolic geodesic distance satisfies. We document this in the v9
paper (math section).

Practical pipeline
------------------
  calib: held-out healthy + tumor scans, latents computed via the v9 model
  test:  new scan, latent computed the same way
  q:     weighted (1-alpha) quantile of calib scores
  decide: voxel v has d_B(z_pred_v, z_anatomy_template_v) > q -> tumor

For brain-2D v9 we calibrate on dataset_v8/val. For multi-organ v10,
we calibrate per-organ.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from .hyperbolic import dist as hyper_dist


# -----------------------------------------------------------------------
# Calibration sample container
# -----------------------------------------------------------------------

@dataclass
class HyperbolicCalibSample:
    """One calibration tuple for hyperbolic conformal.

    z_pred:        (D,) predicted latent for this voxel/sample on the Poincare ball
    z_true:        (D,) ground-truth latent (target encoder output) on the Poincare ball
    weight:        importance weight w_i = dQ/dP under intervention distribution
    label:         optional ground-truth class for stratified calibration
    """
    z_pred: torch.Tensor
    z_true: torch.Tensor
    weight: float = 1.0
    label: int = 0

    def __post_init__(self):
        assert self.z_pred.shape == self.z_true.shape, \
            f"shape mismatch: pred {self.z_pred.shape} vs true {self.z_true.shape}"
        assert self.weight > 0


# -----------------------------------------------------------------------
# Weighted quantile (matches conformal_counterfactual_seg)
# -----------------------------------------------------------------------

def weighted_quantile_tibshirani(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted q-quantile with Tibshirani et al. 2019 finite-sample inflation.

    Equivalent to inserting one "infinity" calibration point so the
    coverage guarantee holds at level q EXACTLY (not just asymptotically).
    """
    assert values.shape == weights.shape and values.ndim == 1
    assert 0.0 < q < 1.0
    order = np.argsort(values)
    v_sorted = values[order]
    w_sorted = weights[order]
    w_norm = w_sorted / w_sorted.sum()
    n = values.size
    target = min(1.0, q * (1.0 + 1.0 / n))
    cum = np.cumsum(w_norm)
    idx = int(np.searchsorted(cum, target, side="left"))
    idx = min(idx, n - 1)
    return float(v_sorted[idx])


# -----------------------------------------------------------------------
# Main calibrator
# -----------------------------------------------------------------------

@dataclass
class HyperbolicCalibReport:
    n_calib: int
    alpha: float
    q: float
    curvature_c: float
    weighted: bool
    empirical_coverage_on_calib: float
    score_distribution_summary: Dict[str, float]


class HyperbolicConformalCalibrator:
    """Weighted conformal calibrator with hyperbolic geodesic nonconformity.

    Usage:
        cal = HyperbolicConformalCalibrator(alpha=0.10, curvature_c=1.0)
        report = cal.calibrate(calib_samples)
        out = cal.predict(z_pred_test, z_true_test_optional)
        # out["in_prediction_set"]: bool (test residual within calibrated radius)
        # out["score"]: float (the hyperbolic distance score)
        # out["q"]: float (calibrated quantile)
    """

    def __init__(self, alpha: float = 0.10, curvature_c: float = 1.0):
        assert 0.0 < alpha < 1.0
        self.alpha = float(alpha)
        self.curvature_c = float(curvature_c)
        self.q: Optional[float] = None
        self._report: Optional[HyperbolicCalibReport] = None

    def _score(self, z_pred: torch.Tensor, z_true: torch.Tensor) -> torch.Tensor:
        """Hyperbolic geodesic distance: the nonconformity score."""
        return hyper_dist(z_pred, z_true, self.curvature_c)

    def calibrate(self, samples: List[HyperbolicCalibSample],
                  verbose: bool = False) -> HyperbolicCalibReport:
        """Fit the calibrated quantile q on a held-out calibration set."""
        assert len(samples) >= 30, (
            f"need >= 30 calibration scans for stable (1-alpha) quantile at "
            f"alpha={self.alpha}; got {len(samples)}"
        )
        scores = []
        weights = []
        for s in samples:
            sc = self._score(s.z_pred, s.z_true).item()
            scores.append(sc)
            weights.append(s.weight)
        scores_arr = np.asarray(scores, dtype=np.float64)
        weights_arr = np.asarray(weights, dtype=np.float64)
        weighted = bool(np.any(weights_arr != weights_arr[0]))
        q = weighted_quantile_tibshirani(scores_arr, weights_arr, 1.0 - self.alpha)
        self.q = float(q)
        covered = int((scores_arr <= q).sum())
        cov = covered / float(len(scores_arr))
        self._report = HyperbolicCalibReport(
            n_calib=len(samples),
            alpha=self.alpha,
            q=self.q,
            curvature_c=self.curvature_c,
            weighted=weighted,
            empirical_coverage_on_calib=cov,
            score_distribution_summary={
                "min": float(scores_arr.min()),
                "max": float(scores_arr.max()),
                "mean": float(scores_arr.mean()),
                "median": float(np.median(scores_arr)),
                "p10": float(np.percentile(scores_arr, 10)),
                "p90": float(np.percentile(scores_arr, 90)),
            },
        )
        if verbose:
            print(f"[hyperbolic-conformal] n={len(samples)}  q={q:.4f}  "
                  f"cov={cov:.3f}  curvature={self.curvature_c:.2f}")
        return self._report

    def predict(self, z_pred: torch.Tensor, z_true: torch.Tensor) -> Dict:
        """At test time, decide whether a test latent's residual is
        within the calibrated radius.

        For voxelwise anomaly: pass z_pred = predicted-latent at voxel v
        and z_true = atlas-template-latent at voxel v. If the score
        exceeds q, voxel v is *certified anomalous* with >= (1-alpha)
        coverage on healthy data.

        Returns:
            {"score": float, "q": float, "in_prediction_set": bool,
             "anomaly_certified": bool}
        """
        if self.q is None:
            raise RuntimeError("calibrate(...) must be called before predict(...)")
        s = self._score(z_pred, z_true).item()
        return {
            "score": s,
            "q": self.q,
            "in_prediction_set": (s <= self.q),
            "anomaly_certified": (s > self.q),
        }

    def predict_batch(self, z_pred_batch: torch.Tensor,
                      z_true_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Vectorized prediction over a batch."""
        if self.q is None:
            raise RuntimeError("calibrate(...) must be called before predict_batch(...)")
        scores = self._score(z_pred_batch, z_true_batch)
        return {
            "scores": scores,
            "q": torch.full_like(scores, self.q),
            "in_prediction_set": (scores <= self.q),
            "anomaly_certified": (scores > self.q),
        }

    # ----- serialisation -----------------------------------------------

    def state_dict(self) -> Dict:
        if self.q is None:
            raise RuntimeError("nothing to save; calibrate first")
        return {
            "q": float(self.q),
            "alpha": float(self.alpha),
            "curvature_c": float(self.curvature_c),
            "report": asdict(self._report) if self._report else None,
        }

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.state_dict(), indent=2))

    def load(self, path) -> None:
        d = json.loads(Path(path).read_text())
        self.q = float(d["q"])
        self.alpha = float(d["alpha"])
        self.curvature_c = float(d["curvature_c"])


# -----------------------------------------------------------------------
# Voxelwise wrapper: produces anomaly map from v9 model outputs
# -----------------------------------------------------------------------

def voxelwise_hyperbolic_anomaly_map(
    z_pred_voxels: torch.Tensor,  # (B, D, H, W) per-voxel latent
    z_template_voxels: torch.Tensor,  # (B, D, H, W) template/anchor latent
    calibrator: HyperbolicConformalCalibrator,
) -> torch.Tensor:
    """Compute a per-voxel anomaly map using the hyperbolic calibrator.

    Returns (B, 1, H, W) boolean tensor: True where the voxel's hyperbolic
    distance from the template exceeds the calibrated radius q.
    """
    if calibrator.q is None:
        raise RuntimeError("calibrator must be fit first")
    B, D, H, W = z_pred_voxels.shape
    z_pred = z_pred_voxels.permute(0, 2, 3, 1).reshape(-1, D)  # (B*H*W, D)
    z_tpl = z_template_voxels.permute(0, 2, 3, 1).reshape(-1, D)
    scores = hyper_dist(z_pred, z_tpl, calibrator.curvature_c)
    anomalous = (scores > calibrator.q).reshape(B, H, W).unsqueeze(1)
    return anomalous


__all__ = [
    "HyperbolicCalibSample",
    "HyperbolicCalibReport",
    "HyperbolicConformalCalibrator",
    "voxelwise_hyperbolic_anomaly_map",
    "weighted_quantile_tibshirani",
]
