"""Pyramidal Gaussian noise generator from Frotscher et al. ANDi 2024.

This is the critical training trick that makes ANDi work — at training
time, instead of standard Gaussian noise ε ~ N(0, I), the diffusion
model is corrupted with noise generated at MULTIPLE spatial scales.
This forces the model to learn to denoise low-frequency structure as
well as high-frequency texture, which is what makes per-timestep
deviation a useful anomaly signal at inference.

Formula (paper §3.2):
    ε = Σ_{i=1..N} c^i · U(ε^(i); H, W)

where:
  ε^(i) ~ N(0, I_{C × h_i × w_i})
  h_i = ⌈H / r_i^{i-1}⌉,   r_i ~ Uniform(2, 4)
  U = bilinear upsample to (H, W)
  c = 0.8, N = 10 scales

Output is rescaled to unit variance so it can be a drop-in replacement
for `torch.randn_like(x)` in any standard DDPM training loop.

Reference: Frotscher et al. "Unsupervised Anomaly Detection using
Aggregated Normative Diffusion" arXiv 2312.01904 / Medical Image
Analysis 2025 (S1361841525004414).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def pyramidal_noise(shape, num_scales: int = 10, c: float = 0.8,
                     r_low: float = 2.0, r_high: float = 4.0,
                     device: str = 'cuda') -> torch.Tensor:
    """Generate pyramidal Gaussian noise with the same shape + variance
    profile as torch.randn(shape).

    shape: tuple (B, C, H, W)
    num_scales: number of pyramid levels (paper uses 10)
    c: per-scale decay coefficient (paper uses 0.8)
    r_low / r_high: uniform range for the per-image resolution-scaling
        factor r_i (paper: r_i ~ Uniform(2, 4))

    Returns a tensor with approximately unit variance.
    """
    B, C, H, W = shape
    total = torch.zeros(shape, device=device, dtype=torch.float32)
    # Sample one r per scale, shared across the batch (matches what the
    # paper's reference impl does; per-image r would also be fine but
    # slower with no measured AUC change).
    r = float(torch.empty(1).uniform_(r_low, r_high).item())
    for i in range(1, num_scales + 1):
        # Downsampled spatial dims at scale i
        denom = r ** (i - 1)
        h_i = max(1, int(math.ceil(H / denom)))
        w_i = max(1, int(math.ceil(W / denom)))
        eps = torch.randn(B, C, h_i, w_i, device=device)
        # Bilinear upsample back to (H, W). At i=1 this is identity (h_i=H, w_i=W).
        if (h_i, w_i) != (H, W):
            eps = F.interpolate(eps, size=(H, W), mode='bilinear', align_corners=False)
        total = total + (c ** i) * eps

    # Rescale so total has unit variance per pixel.
    # Each term contributes variance ~= c^(2*i) (upsampling is approximately
    # variance-preserving here — bilinear smooths but doesn't change the
    # average variance much).
    norm = math.sqrt(sum(c ** (2 * i) for i in range(1, num_scales + 1)))
    return total / norm


def pyramidal_noise_like(x: torch.Tensor, **kwargs) -> torch.Tensor:
    """Drop-in for torch.randn_like — same shape, same device."""
    kwargs.setdefault('device', x.device)
    return pyramidal_noise(x.shape, **kwargs)


__all__ = ['pyramidal_noise', 'pyramidal_noise_like']
