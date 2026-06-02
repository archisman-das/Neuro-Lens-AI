"""ANDi (Aggregated Normative Diffusion) inference on our v9b DDPM.

From Frotscher et al. "Unsupervised Anomaly Detection using Aggregated
Normative Diffusion" (arXiv 2312.01904 / ScienceDirect S1361841525004414).

Caveat: the paper's full method requires training the DDPM with pyramidal
Gaussian noise. Our Stage 2 DDPM was trained with STANDARD Gaussian noise
(it was originally meant as a counterfactual decoder, not an ANDi
detector). So this implements the INFERENCE-TIME aggregation only — we
test whether that alone improves over single-step reconstruction.

If results are encouraging (AUC > 0.75 on the 148-sample OOD bench),
ship it. If not, the followup is to retrain the DDPM with pyramidal
noise (~3h Colab), which the paper shows is the critical training trick.

Inference algorithm (from paper §3.3):
  For each t in [T_l, T_u]:
    1. ε ~ N(0, I)
    2. x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε        (q_sample)
    3. Ground-truth posterior mean μ_q(x_t, x_0, t)
    4. Model-predicted posterior mean μ_θ(x_t, t)
    5. d_t = (μ_q - μ_θ)²                     (per-pixel)
  Aggregate via geometric mean:
    A_gm = exp(mean_t(log(d_t + ε_floor)))
  Per-image anomaly score = high quantile of A_gm.
"""
from __future__ import annotations

from typing import Optional

import torch


@torch.no_grad()
def _posterior_mean(ddpm, x_t: torch.Tensor, x_0: torch.Tensor,
                     t: torch.Tensor) -> torch.Tensor:
    """Ground-truth DDPM posterior mean μ_q(x_t, x_0, t).

    Following Ho et al. 2020 eqn (7):
        μ_q = (√(ᾱ_{t-1}) β_t) / (1 - ᾱ_t) · x_0
              + (√(α_t)(1 - ᾱ_{t-1})) / (1 - ᾱ_t) · x_t
    where α_t = 1 - β_t (per-step), ᾱ_t cumulative.
    """
    betas = ddpm.betas[t][:, None, None, None]               # β_t
    a_cum = ddpm.alphas_cum[t][:, None, None, None]          # ᾱ_t
    a_cum_prev = ddpm.alphas_cum_prev[t][:, None, None, None]  # ᾱ_{t-1}
    a_t = 1.0 - betas                                          # α_t = 1 - β_t
    coef0 = (a_cum_prev.sqrt() * betas) / (1.0 - a_cum)
    coef_xt = (a_t.sqrt() * (1.0 - a_cum_prev)) / (1.0 - a_cum)
    return coef0 * x_0 + coef_xt * x_t


@torch.no_grad()
def _predicted_posterior_mean(ddpm, x_t: torch.Tensor, t: torch.Tensor,
                               cond: torch.Tensor) -> torch.Tensor:
    """Model-predicted posterior mean μ_θ(x_t, t, cond).

    The DDPM predicts noise ε_θ; we convert that to a predicted x_0,
    then plug into the same posterior-mean formula as ground truth.
    """
    a_cum = ddpm.alphas_cum[t][:, None, None, None]
    eps_pred = ddpm.net(x_t, t, cond)
    x0_pred = (x_t - (1.0 - a_cum).sqrt() * eps_pred) / a_cum.sqrt()
    # Clamp x0 to a reasonable range (the model can produce wild
    # extrapolations on OOD inputs which would dominate the squared error)
    x0_pred = x0_pred.clamp(-3, 3)
    return _posterior_mean(ddpm, x_t, x0_pred, t)


@torch.no_grad()
def andi_anomaly_map(ddpm, x_0: torch.Tensor, cond: torch.Tensor,
                      t_low: int = 75, t_high: int = 200,
                      stride: int = 5,
                      device: str = 'cuda',
                      seed: Optional[int] = 0) -> torch.Tensor:
    """Compute ANDi aggregated-normative-diffusion anomaly map.

    Args:
      ddpm: trained DDPM with q_sample + net (noise predictor)
      x_0: (B, C, H, W) input image
      cond: (B, cond_dim) conditioning vector (the JEPA latent for us)
      t_low, t_high: timestep range to aggregate over (default 75-200
        from paper §4.1)
      stride: step within [t_low, t_high]. Paper uses every t (stride=1)
        with 125 forwards per image. We default stride=5 for 25 forwards
        (~250ms per image on RTX 4060 vs 1.25s) — empirical pilot showed
        AUC barely moves between stride=1 and stride=5.
      seed: torch RNG seed for reproducible noise samples

    Returns:
      A_gm: (B, C, H, W) per-pixel anomaly map (geometric mean over t).
    """
    if seed is not None:
        g = torch.Generator(device=device).manual_seed(seed)
    else:
        g = None

    B, C, H, W = x_0.shape
    eps_floor = 1e-8
    log_sum = torch.zeros_like(x_0)
    n = 0
    for t_int in range(t_low, t_high, stride):
        t = torch.full((B,), t_int, device=device, dtype=torch.long)
        # Standard Gaussian noise at inference (paper §4.3: works better
        # than pyramidal at inference time)
        if g is not None:
            noise = torch.randn(x_0.shape, generator=g, device=device)
        else:
            noise = torch.randn_like(x_0)
        x_t = ddpm.q_sample(x_0, t, noise)
        mu_q = _posterior_mean(ddpm, x_t, x_0, t)
        mu_theta = _predicted_posterior_mean(ddpm, x_t, t, cond)
        d_t = (mu_q - mu_theta).pow(2)
        log_sum = log_sum + torch.log(d_t + eps_floor)
        n += 1
    A_gm = torch.exp(log_sum / max(n, 1))
    return A_gm


@torch.no_grad()
def andi_score(ddpm, jepa, image_rgb_uint8, percentile: float = 95.0,
                stride: int = 5, device: str = 'cuda') -> float:
    """End-to-end ANDi per-image anomaly score.

    Uses JEPA's global encoded latent as the DDPM conditioning (matching
    how Stage 2 was trained).
    """
    import numpy as np
    from PIL import Image
    img = Image.fromarray(image_rgb_uint8).convert('RGB').resize((256, 256), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
    cond = jepa.encode_full(x).mean(dim=1)
    amap = andi_anomaly_map(ddpm, x, cond, stride=stride, device=device)
    return float(torch.quantile(amap.flatten(), percentile / 100.0).item())


__all__ = ['andi_anomaly_map', 'andi_score']
