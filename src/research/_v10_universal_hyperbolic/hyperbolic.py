"""Poincare ball model + Mobius operations for v9 hyperbolic tumor embedding.

Why hyperbolic
--------------
Brain anatomy is hierarchical (cortex -> regions -> networks -> voxels).
Tumor taxonomy is hierarchical (tumor -> type -> subtype -> grade).
Both have exponentially branching tree structure.

Euclidean embeddings distort hierarchies because the volume of a Euclidean
ball grows polynomially with radius. Hyperbolic embeddings preserve tree
structure with provably lower distortion because the volume of a Poincare
ball grows EXPONENTIALLY with radius -- matching the exponential branching
of biological hierarchies (Nickel & Kiela 2017, Ganea et al. 2018).

This module implements the Poincare ball ops we need from scratch in
pure PyTorch. No external geometry libraries (geoopt etc) so Colab
installs stay fast and the code is auditable.

References
----------
Ganea, Becigneul, Hofmann. "Hyperbolic Neural Networks." NeurIPS 2018.
Nickel, Kiela. "Poincare Embeddings for Learning Hierarchical Representations." NeurIPS 2017.
Khrulkov et al. "Hyperbolic Image Embeddings." CVPR 2020.

API surface
-----------
  expmap0(x, c)         -- Euclidean tangent at origin -> Poincare point
  logmap0(y, c)         -- Poincare point -> Euclidean tangent at origin
  mobius_add(x, y, c)   -- x (+)_c y
  mobius_matvec(M, x, c)-- M (X)_c x
  dist(x, y, c)         -- hyperbolic geodesic distance
  HyperbolicLinear      -- Mobius linear layer
  HyperbolicProjection  -- Euclidean features -> Poincare ball
  PoincareDistance      -- pairwise hyperbolic distance module

Curvature convention: Poincare ball of curvature -c (c > 0). c=1 is the
standard unit ball. Curvature is learnable via softplus reparameterization
in the modules.

Numerical stability: points are clipped to (1 - EPS) / sqrt(c) of the
boundary; gradients near the boundary diverge so the clip is essential
(Ganea et al. 2018 recommendation, EPS = 4e-3).
"""

from __future__ import annotations

import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 4e-3
MIN_NORM = 1e-15


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _as_tensor_like(c: Union[float, torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(c):
        return c.to(dtype=ref.dtype, device=ref.device)
    return torch.tensor(c, dtype=ref.dtype, device=ref.device)


def _project_to_ball(x: torch.Tensor, c: Union[float, torch.Tensor]) -> torch.Tensor:
    sqrt_c = _as_tensor_like(c, x).clamp_min(MIN_NORM).sqrt()
    max_norm = (1.0 - EPS) / sqrt_c
    norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(MIN_NORM)
    cond = (norm > max_norm).expand_as(x)
    projected = x * (max_norm / norm)
    return torch.where(cond, projected, x)


# -----------------------------------------------------------------------
# Core ops on the Poincare ball
# -----------------------------------------------------------------------

def expmap0(v: torch.Tensor, c: Union[float, torch.Tensor] = 1.0) -> torch.Tensor:
    """Exponential map at origin: T_0 B^n -> B^n.

    expmap0(v) = tanh(sqrt(c) * ||v||) * v / (sqrt(c) * ||v||)
    """
    c_t = _as_tensor_like(c, v)
    sqrt_c = c_t.clamp_min(MIN_NORM).sqrt()
    v_norm = v.norm(dim=-1, keepdim=True, p=2).clamp_min(MIN_NORM)
    gamma = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm)
    return _project_to_ball(gamma * v, c)


def logmap0(y: torch.Tensor, c: Union[float, torch.Tensor] = 1.0) -> torch.Tensor:
    """Logarithmic map at origin: B^n -> T_0 B^n. Inverse of expmap0."""
    c_t = _as_tensor_like(c, y)
    sqrt_c = c_t.clamp_min(MIN_NORM).sqrt()
    y_norm = y.norm(dim=-1, keepdim=True, p=2).clamp_min(MIN_NORM)
    return torch.atanh(torch.clamp(sqrt_c * y_norm, max=1.0 - EPS)) / sqrt_c * (y / y_norm)


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: Union[float, torch.Tensor] = 1.0) -> torch.Tensor:
    """Mobius addition: x (+)_c y. Not commutative in general."""
    c_t = _as_tensor_like(c, x)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1.0 + 2.0 * c_t * xy + c_t * y2) * x + (1.0 - c_t * x2) * y
    denom = 1.0 + 2.0 * c_t * xy + (c_t ** 2) * x2 * y2
    return _project_to_ball(num / denom.clamp_min(MIN_NORM), c)


def mobius_matvec(M: torch.Tensor, x: torch.Tensor, c: Union[float, torch.Tensor] = 1.0) -> torch.Tensor:
    """Mobius matrix-vector product: M (X)_c x via expmap0(M @ logmap0(x))."""
    Mx = logmap0(x, c) @ M.transpose(-1, -2)
    return expmap0(Mx, c)


def dist(x: torch.Tensor, y: torch.Tensor, c: Union[float, torch.Tensor] = 1.0) -> torch.Tensor:
    """Hyperbolic geodesic distance on the Poincare ball.

    d_c(x, y) = (2 / sqrt(c)) * arctanh(sqrt(c) * || -x (+) y ||)
    """
    c_t = _as_tensor_like(c, x)
    sqrt_c = c_t.clamp_min(MIN_NORM).sqrt()
    diff = mobius_add(-x, y, c)
    diff_norm = diff.norm(dim=-1, p=2).clamp_min(MIN_NORM)
    return (2.0 / sqrt_c) * torch.atanh(torch.clamp(sqrt_c * diff_norm, max=1.0 - EPS))


# -----------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------

class HyperbolicProjection(nn.Module):
    """Euclidean features -> Poincare ball via expmap0.

    Curvature c is learnable by default via softplus reparameterization
    (raw_c is the unconstrained parameter; c = softplus(raw_c) > 0).
    """

    def __init__(self, c_init: float = 1.0, learnable: bool = True):
        super().__init__()
        if learnable:
            inverse_softplus = math.log(math.expm1(c_init))
            self._raw_c = nn.Parameter(torch.tensor(inverse_softplus))
        else:
            self.register_buffer("_raw_c", torch.tensor(c_init))
        self._learnable = learnable

    @property
    def c(self) -> torch.Tensor:
        if self._learnable:
            return F.softplus(self._raw_c) + MIN_NORM
        return self._raw_c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return expmap0(x, self.c)


class HyperbolicLinear(nn.Module):
    """Mobius linear layer (Ganea et al. 2018 Eq 22)."""

    def __init__(self, in_features: int, out_features: int, c_init: float = 1.0,
                  bias: bool = True, learnable_c: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        if learnable_c:
            inverse_softplus = math.log(math.expm1(c_init))
            self._raw_c = nn.Parameter(torch.tensor(inverse_softplus))
        else:
            self.register_buffer("_raw_c", torch.tensor(c_init))
        self._learnable_c = learnable_c
        self.reset_parameters()

    @property
    def c(self) -> torch.Tensor:
        if self._learnable_c:
            return F.softplus(self._raw_c) + MIN_NORM
        return self._raw_c

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Mx = mobius_matvec(self.weight, x, self.c)
        if self.bias is not None:
            bias_h = expmap0(self.bias.unsqueeze(0), self.c)
            return mobius_add(Mx, bias_h, self.c)
        return Mx


class PoincareDistance(nn.Module):
    """Pairwise Poincare distance as an nn.Module."""

    def __init__(self, c: float = 1.0, learnable_c: bool = False):
        super().__init__()
        if learnable_c:
            inverse_softplus = math.log(math.expm1(c))
            self._raw_c = nn.Parameter(torch.tensor(inverse_softplus))
        else:
            self.register_buffer("_raw_c", torch.tensor(c))
        self._learnable_c = learnable_c

    @property
    def c(self) -> torch.Tensor:
        if self._learnable_c:
            return F.softplus(self._raw_c) + MIN_NORM
        return self._raw_c

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return dist(x, y, self.c)


__all__ = [
    "expmap0",
    "logmap0",
    "mobius_add",
    "mobius_matvec",
    "dist",
    "HyperbolicProjection",
    "HyperbolicLinear",
    "PoincareDistance",
]
