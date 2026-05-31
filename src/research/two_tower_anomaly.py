"""Two-tower anomaly combiner for v9b.

Combines the JEPA appearance-anomaly map with the SDF geometric-anomaly
map into a unified anomaly score. Supports weighted-sum, AND, and OR
combination modes. Each map is normalized to [0,1] via per-tower
calibrated quantiles before combining.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def normalize_per_tower(map_: torch.Tensor, q_low: float = 0.01,
                         q_high: float = 0.99) -> torch.Tensor:
    """Robust min-max normalize a single-tower anomaly map to [0,1]
    using the q_low / q_high quantiles (drops outliers)."""
    flat = map_.flatten()
    lo = torch.quantile(flat, q_low)
    hi = torch.quantile(flat, q_high)
    return ((map_ - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)


def combine_two_towers(appearance: torch.Tensor, geometry: torch.Tensor,
                       mode: str = "weighted_sum", lambda_app: float = 0.6,
                       lambda_geo: float = 0.4,
                       q_app: float = None, q_geo: float = None) -> torch.Tensor:
    """Combine appearance + geometry anomaly maps.

    appearance, geometry: (B, 1, H, W) per-voxel anomaly scores
    mode:
      weighted_sum -> lambda_app * app + lambda_geo * geo  (continuous score)
      and          -> 1 if app > q_app AND geo > q_geo, else 0 (binary)
      or           -> 1 if app > q_app OR  geo > q_geo, else 0 (binary)
    """
    assert appearance.shape == geometry.shape
    app_n = normalize_per_tower(appearance)
    geo_n = normalize_per_tower(geometry)
    if mode == "weighted_sum":
        return lambda_app * app_n + lambda_geo * geo_n
    if mode == "and":
        if q_app is None or q_geo is None:
            raise ValueError("AND mode requires per-tower quantiles q_app, q_geo")
        return ((app_n > q_app) & (geo_n > q_geo)).float()
    if mode == "or":
        if q_app is None or q_geo is None:
            raise ValueError("OR mode requires per-tower quantiles q_app, q_geo")
        return ((app_n > q_app) | (geo_n > q_geo)).float()
    raise ValueError(f"unknown mode: {mode}")


__all__ = ["combine_two_towers", "normalize_per_tower"]
