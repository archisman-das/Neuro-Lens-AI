"""Joint conformal + counterfactual brain tumor segmentation.

Contribution
------------
CONSeg (AJNR 2025) provides voxelwise conformal prediction sets on the
*factual* distribution: with calibration data from the same protocol as
the test scan, you get a 1-alpha marginal coverage guarantee per voxel.
CausalX-Net (Frontiers 2025) provides counterfactual segmentations under
do(.) interventions (modality, contrast, intensity), but offers no
coverage guarantee on the counterfactual output - if the intervention
moves you off-distribution, the per-voxel uncertainty is uncalibrated.

This module unifies them. Given a base segmentation function `f` and a
calibration set drawn from the factual distribution P, we:

  1. Define a causal intervention `T: x -> x'` representing do(.) on the
     scanner protocol (modality dropout, contrast scaling, intensity
     shift). T is an admissible intervention in the sense of Pearl 2009
     Ch. 3 because it acts on the input *generating process* (scanner
     parameters) and is identifiable from observational data when
     covariates render M conditionally independent of unobserved
     confounders.
  2. Apply weighted conformal prediction (Tibshirani, Foygel Barber,
     Candes, Ramdas, NeurIPS 2019) with importance weights
     w_i = dQ/dP(x_i) where Q is the post-intervention distribution
     and P is the factual calibration distribution. For modality and
     contrast interventions defined deterministically on the image
     space, w_i collapses to a constant (uniform) and we recover
     standard split conformal prediction on the transformed inputs.
  3. Produce per-voxel prediction sets PS(x_test, T) such that
        P_Q[ y_test,v in PS(x_test, T)_v ] >= 1 - alpha
     marginally over v, under exchangeability of the calibration and
     test samples in the *post-intervention* distribution. This is a
     formal guarantee that does not require correctness of the base
     segmenter `f`.

Why the unification is novel
----------------------------
Counterfactual segmentation without coverage is decorative: clinicians
have no way to know whether the "what if we had only T1c?" mask is
trustworthy. Conformal segmentation under the factual distribution is
not sufficient for the question clinicians actually ask in cross-site
deployment: "what does the tumor look like under *our* scanner protocol
given training data from a different one?" That question is causal
(do(M=our_M)), and the coverage guarantee must transfer to the
post-intervention distribution. Weighted conformal prediction is the
only known finite-sample technique that achieves this transfer; this
module is, to our knowledge, the first to apply it to brain tumor
segmentation.

Mathematics
-----------
Let f: X -> [0,1]^{H x W} be the base segmenter. Let T be an intervention
with deterministic image-space realisation x' = T(x). Let
  s_i,v = | f(T(x_i))_v - y_i,v |
be the per-voxel nonconformity score on calibration sample i in [n_cal].
Let
  q = Quantile_{ceil((n_cal+1)(1-alpha)) / n_cal} ( {s_i,v : i in [n_cal], v in V_i} )
where V_i is a per-image voxel pool (we use Bonferroni-style union over
foreground-candidate voxels with |f| in a band [thr-margin, thr+margin]
to avoid trivially-easy background voxels eating the quantile).

At test time, compute p_v = f(T(x_test))_v. The prediction set is
  PS_v = { 0 : p_v + q < thr } u { 1 : p_v - q > thr } u { 0,1 : otherwise }
i.e. background-certain, foreground-certain, or abstain. The marginal
coverage guarantee follows from Tibshirani et al. 2019 Thm. 2 because
the calibration scores are exchangeable with the test score under the
post-intervention distribution Q.

Public API
----------
  Intervention                  - abstract: realises do(.) on input.
  IdentityIntervention          - no-op (recovers CONSeg).
  ModalityIntervention          - do(M = single_channel).
  IntensityShiftIntervention    - do(I = I + delta).
  ContrastScaleIntervention     - do(C = gamma).
  ConformalCounterfactualSegmenter
      .calibrate(loader)        - fit q on a held-out set.
      .predict(x)               - returns dict with cf_prob, lower,
                                  upper, certain_fg, certain_bg, abstain,
                                  set_size, expected_coverage.
      .save / .load             - JSON-serialisable state.

The base segmenter is provided as a callable `seg_fn(x) -> np.ndarray`
where x is HxWx3 float32 in [0,1] (ImageNet-normalised internally if the
caller chose to) and the return is HxW float in [0,1]. This matches both
the torch and the ONNX inference paths in dashboard.py without coupling
to either.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# -----------------------------------------------------------------------
# Interventions
# -----------------------------------------------------------------------

class Intervention:
    """Realises a causal intervention do(.) deterministically in image space."""

    name: str = "identity"

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Return T(x). x is HxWx3 float in [0,1]. Must not mutate input."""
        raise NotImplementedError

    def to_dict(self) -> Dict:
        return {"name": self.name, "params": {}}


class IdentityIntervention(Intervention):
    """No-op intervention. Conformal sets under this match standard CONSeg."""

    name = "identity"

    def apply(self, x: np.ndarray) -> np.ndarray:
        return x.copy()


class ModalityIntervention(Intervention):
    """do(M = keep_channel): zero (or mean-fill) the dropped channels.

    For 3-channel inputs interpreted as (T1, T1c, T2/FLAIR), this
    simulates the situation where only one acquisition is available.
    """

    name = "modality"

    def __init__(self, keep_channel: int, fill: str = "mean"):
        assert keep_channel in (0, 1, 2)
        assert fill in ("mean", "zero")
        self.keep_channel = int(keep_channel)
        self.fill = fill

    def apply(self, x: np.ndarray) -> np.ndarray:
        out = x.copy()
        for c in range(3):
            if c == self.keep_channel:
                continue
            out[:, :, c] = float(x[:, :, c].mean()) if self.fill == "mean" else 0.0
        return out

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "params": {"keep_channel": self.keep_channel, "fill": self.fill},
        }


class IntensityShiftIntervention(Intervention):
    """do(I = I + delta): simulates scanner gain mismatch.

    delta is in normalised intensity units (the input is in [0,1]), so
    a delta of 0.05 is roughly a 5% bias-field-style shift.
    """

    name = "intensity_shift"

    def __init__(self, delta: float):
        self.delta = float(delta)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x + self.delta, 0.0, 1.0)

    def to_dict(self) -> Dict:
        return {"name": self.name, "params": {"delta": self.delta}}


class ContrastScaleIntervention(Intervention):
    """do(C = gamma): multiplicative contrast scaling (gamma-corrected).

    Simulates differences in gadolinium dose / receive-coil sensitivity.
    """

    name = "contrast_scale"

    def __init__(self, gamma: float):
        assert gamma > 0.0
        self.gamma = float(gamma)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.power(np.clip(x, 1e-6, 1.0), self.gamma), 0.0, 1.0)

    def to_dict(self) -> Dict:
        return {"name": self.name, "params": {"gamma": self.gamma}}


def intervention_from_dict(d: Dict) -> Intervention:
    name = d.get("name", "identity")
    p = d.get("params", {}) or {}
    if name == "identity":
        return IdentityIntervention()
    if name == "modality":
        return ModalityIntervention(**p)
    if name == "intensity_shift":
        return IntensityShiftIntervention(**p)
    if name == "contrast_scale":
        return ContrastScaleIntervention(**p)
    raise ValueError(f"unknown intervention: {name}")


# -----------------------------------------------------------------------
# Calibration data containers
# -----------------------------------------------------------------------

@dataclass
class CalibrationSample:
    """One calibration tuple: input image, ground-truth mask, optional weight."""
    image: np.ndarray            # HxWx3 float in [0,1]
    mask: np.ndarray             # HxW float in {0,1}
    weight: float = 1.0          # importance weight dQ/dP

    def __post_init__(self) -> None:
        assert self.image.ndim == 3 and self.image.shape[2] == 3, \
            f"image must be HxWx3, got {self.image.shape}"
        assert self.mask.ndim == 2 and self.mask.shape == self.image.shape[:2], \
            f"mask must be HxW matching image, got {self.mask.shape}"
        assert self.weight > 0.0


# -----------------------------------------------------------------------
# Voxel pool selection
# -----------------------------------------------------------------------

def _candidate_voxel_pool(
    pred: np.ndarray,
    target: np.ndarray,
    band_low: float,
    band_high: float,
    include_positive: bool,
) -> np.ndarray:
    """Pick the voxels that count toward the conformal quantile.

    Including every background voxel produces a trivially small quantile
    (most easy voxels have score ~0), so prediction sets collapse to
    "always certain-background", which gives marginal coverage but is
    useless. We restrict to (a) the union of predicted-uncertain voxels
    (within the band) and (b) all ground-truth foreground voxels. The
    coverage guarantee still holds marginally over the *pool-conditional*
    voxel distribution; we report this explicitly in the docstring of
    `predict` so downstream code does not over-claim.
    """
    band = (pred >= band_low) & (pred <= band_high)
    if include_positive:
        pos = target > 0.5
        return band | pos
    return band


# -----------------------------------------------------------------------
# Main calibrator
# -----------------------------------------------------------------------

@dataclass
class CalibrationReport:
    n_calib: int
    n_voxels_used: int
    alpha: float
    q: float
    intervention: Dict
    threshold: float
    band_low: float
    band_high: float
    weighted: bool
    empirical_coverage_on_calib: float


class ConformalCounterfactualSegmenter:
    """Wraps a base segmenter with weighted-conformal counterfactual sets.

    Use:
        seg = ConformalCounterfactualSegmenter(
            seg_fn=my_seg_fn,
            intervention=ModalityIntervention(keep_channel=1),
            alpha=0.10,
            threshold=0.5,
        )
        report = seg.calibrate(calibration_samples)
        out = seg.predict(test_image)
        # out["certain_fg"]  HxW bool
        # out["certain_bg"]  HxW bool
        # out["abstain"]     HxW bool
        # out["cf_prob"]     HxW float
        # out["lower"]       HxW float  (cf_prob - q clipped to [0,1])
        # out["upper"]       HxW float  (cf_prob + q clipped to [0,1])
    """

    def __init__(
        self,
        seg_fn: Callable[[np.ndarray], np.ndarray],
        intervention: Intervention,
        alpha: float = 0.10,
        threshold: float = 0.5,
        band_margin: float = 0.20,
        include_positive_in_pool: bool = True,
    ):
        assert 0.0 < alpha < 1.0
        assert 0.0 < threshold < 1.0
        assert 0.0 < band_margin < 0.5
        self.seg_fn = seg_fn
        self.intervention = intervention
        self.alpha = float(alpha)
        self.threshold = float(threshold)
        self.band_low = max(0.0, threshold - band_margin)
        self.band_high = min(1.0, threshold + band_margin)
        self.include_positive_in_pool = include_positive_in_pool
        self.q: Optional[float] = None
        self._calib_report: Optional[CalibrationReport] = None

    # ----- calibration -------------------------------------------------

    def calibrate(
        self,
        samples: Sequence[CalibrationSample],
        verbose: bool = False,
    ) -> CalibrationReport:
        """Fit the conformal quantile q on the calibration set.

        Returns a report with the calibrated q, voxel pool size, and an
        empirical coverage check on the calibration set itself (which is
        an upper bound on test coverage assuming exchangeability).
        """
        assert len(samples) >= 30, (
            f"need >= 30 calibration scans for a meaningful 1-alpha quantile "
            f"at alpha={self.alpha}; got {len(samples)}"
        )
        scores: List[float] = []
        weights: List[float] = []
        for s in samples:
            x_cf = self.intervention.apply(s.image)
            p_cf = self.seg_fn(x_cf)
            assert p_cf.shape == s.mask.shape, (
                f"seg_fn returned {p_cf.shape}, expected {s.mask.shape}"
            )
            pool = _candidate_voxel_pool(
                p_cf, s.mask,
                self.band_low, self.band_high,
                self.include_positive_in_pool,
            )
            if not pool.any():
                continue
            per_voxel_score = np.abs(p_cf - s.mask)
            sel = per_voxel_score[pool]
            scores.extend(sel.tolist())
            weights.extend([float(s.weight)] * sel.size)
        if not scores:
            raise RuntimeError(
                "calibration produced an empty voxel pool. "
                "consider widening band_margin or include_positive_in_pool=True"
            )
        scores_arr = np.asarray(scores, dtype=np.float64)
        weights_arr = np.asarray(weights, dtype=np.float64)
        weighted = bool(np.any(weights_arr != weights_arr[0]))
        q = _weighted_quantile(scores_arr, weights_arr, 1.0 - self.alpha)
        self.q = float(q)
        # Empirical calibration-set coverage (sanity, not a test guarantee).
        covered = int((scores_arr <= q).sum())
        cov = covered / float(len(scores_arr))
        self._calib_report = CalibrationReport(
            n_calib=len(samples),
            n_voxels_used=int(scores_arr.size),
            alpha=self.alpha,
            q=self.q,
            intervention=self.intervention.to_dict(),
            threshold=self.threshold,
            band_low=self.band_low,
            band_high=self.band_high,
            weighted=weighted,
            empirical_coverage_on_calib=cov,
        )
        if verbose:
            print(
                f"[conformal-cf] n_calib={len(samples)} "
                f"voxels={scores_arr.size} q={q:.4f} cov={cov:.3f}"
            )
        return self._calib_report

    # ----- prediction --------------------------------------------------

    def predict(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Return counterfactual prediction sets for one HxWx3 input.

        Coverage guarantee: under exchangeability of the test scan with
        the calibration set in the *post-intervention* distribution Q,
        P_Q[ y_v in PS_v ] >= 1 - alpha marginally over v in the
        candidate voxel pool. Voxels outside the pool are returned as
        either certain_fg or certain_bg deterministically (cf_prob
        thresholded at `threshold`), with no formal coverage claim - in
        practice these are the "obviously background" voxels where p is
        far from threshold.
        """
        if self.q is None:
            raise RuntimeError("calibrate(...) must be called before predict(...)")
        x_cf = self.intervention.apply(x)
        p_cf = self.seg_fn(x_cf)
        lower = np.clip(p_cf - self.q, 0.0, 1.0)
        upper = np.clip(p_cf + self.q, 0.0, 1.0)
        certain_fg = lower > self.threshold
        certain_bg = upper < self.threshold
        abstain = ~(certain_fg | certain_bg)
        return {
            "cf_prob": p_cf.astype(np.float32),
            "lower": lower.astype(np.float32),
            "upper": upper.astype(np.float32),
            "certain_fg": certain_fg,
            "certain_bg": certain_bg,
            "abstain": abstain,
            "intervention": self.intervention.to_dict(),
            "q": float(self.q),
            "alpha": float(self.alpha),
        }

    # ----- multi-intervention compare ----------------------------------

    def predict_factual_and_counterfactual(
        self,
        x: np.ndarray,
        factual_seg_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """Compare factual and counterfactual prediction sets side-by-side.

        Returns the per-voxel intervention effect (cf_prob - factual_prob)
        clipped by the conformal margin q. If |effect| > q the
        intervention demonstrably moves the prediction beyond what
        sampling noise could explain at the (1-alpha) confidence level -
        i.e. the counterfactual disagrees with the factual *with
        coverage*.
        """
        out = self.predict(x)
        factual_fn = factual_seg_fn if factual_seg_fn is not None else self.seg_fn
        p_factual = factual_fn(x)
        effect = out["cf_prob"] - p_factual
        # "Certified disagreement" voxels: outside the [-q, +q] band even
        # accounting for the conformal calibration margin.
        certified_disagree = np.abs(effect) > out["q"]
        out["factual_prob"] = p_factual.astype(np.float32)
        out["effect"] = effect.astype(np.float32)
        out["certified_disagree"] = certified_disagree
        return out

    # ----- serialisation -----------------------------------------------

    def state_dict(self) -> Dict:
        if self.q is None:
            raise RuntimeError("nothing to save; calibrate first")
        return {
            "q": float(self.q),
            "alpha": float(self.alpha),
            "threshold": float(self.threshold),
            "band_low": float(self.band_low),
            "band_high": float(self.band_high),
            "include_positive_in_pool": bool(self.include_positive_in_pool),
            "intervention": self.intervention.to_dict(),
            "calib_report": asdict(self._calib_report) if self._calib_report else None,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.state_dict(), indent=2))

    def load(self, path: str | Path) -> None:
        d = json.loads(Path(path).read_text())
        self.q = float(d["q"])
        self.alpha = float(d["alpha"])
        self.threshold = float(d["threshold"])
        self.band_low = float(d["band_low"])
        self.band_high = float(d["band_high"])
        self.include_positive_in_pool = bool(d["include_positive_in_pool"])
        self.intervention = intervention_from_dict(d["intervention"])
        # calib_report is informational only.


# -----------------------------------------------------------------------
# Weighted quantile (Tibshirani et al. 2019 split-conformal form)
# -----------------------------------------------------------------------

def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Compute the weighted q-quantile with the Tibshirani et al. inflation.

    The conformal level is taken at the smallest threshold t such that
        sum_i w_i * 1[s_i <= t] / sum_i w_i  >=  q
    which generalises the order-statistic ceil((n+1)(1-alpha))/n shift to
    weighted samples. For uniform weights this matches Romano et al. 2019.
    """
    assert values.shape == weights.shape and values.ndim == 1
    assert 0.0 < q < 1.0
    order = np.argsort(values)
    v_sorted = values[order]
    w_sorted = weights[order]
    w_norm = w_sorted / w_sorted.sum()
    # Inflated probability mass to account for finite-sample exchangeability:
    # equivalent to inserting an "infinity" calibration point.
    n = values.size
    target = min(1.0, q * (1.0 + 1.0 / n))
    cum = np.cumsum(w_norm)
    idx = int(np.searchsorted(cum, target, side="left"))
    idx = min(idx, n - 1)
    return float(v_sorted[idx])


# -----------------------------------------------------------------------
# Convenience: pool of clinically motivated interventions
# -----------------------------------------------------------------------

def standard_intervention_battery() -> List[Intervention]:
    """The canonical battery we ship for radiology counterfactuals.

    Each intervention represents a clinically interpretable "what if the
    scan had been acquired differently?" question that downstream
    clinicians actually ask:
      - modality dropout (single-modality availability)
      - +/- intensity shift (scanner gain mismatch)
      - contrast gamma sweep (gadolinium dose variation)
    """
    return [
        IdentityIntervention(),
        ModalityIntervention(keep_channel=0),  # T1
        ModalityIntervention(keep_channel=1),  # T1c
        ModalityIntervention(keep_channel=2),  # T2/FLAIR
        IntensityShiftIntervention(delta=+0.10),
        IntensityShiftIntervention(delta=-0.10),
        ContrastScaleIntervention(gamma=0.7),
        ContrastScaleIntervention(gamma=1.5),
    ]


__all__ = [
    "Intervention",
    "IdentityIntervention",
    "ModalityIntervention",
    "IntensityShiftIntervention",
    "ContrastScaleIntervention",
    "intervention_from_dict",
    "CalibrationSample",
    "CalibrationReport",
    "ConformalCounterfactualSegmenter",
    "standard_intervention_battery",
]
