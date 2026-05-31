"""Latent diffusion decoder for v9b: generative healthy counterfactual.

Conditioned on the JEPA-encoded "healthy" latent (z_anatomy with
tumor-region patches inpainted via JEPA self-prediction), generates a
photo-quality reconstruction of what the scan would look like with no
tumor. The residual |input - counterfactual| is the clean tumor mass
visualization, INDEPENDENT of the explicit segmentation mask.

Architecture: small DDPM in image space conditioned on a global latent.
We keep it image-space (not full latent diffusion) for simplicity on
brain-2D scope; the latent conditioning gives enough capacity for the
counterfactual task. Upgrade to true LDM (VAE encoder + diffusion in
latent space) for v10 / scale-up.

References:
  - Ho, Jain, Abbeel. "DDPMs." NeurIPS 2020.
  - Rombach et al. "Stable Diffusion (Latent Diffusion)." CVPR 2022.
  - Sanchez et al. "What is the Right Way to Combine Causal Inference
    and ML for Healthcare?" MICCAI 2023 (diffusion for tumor counterfactual).
"""
from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    """Standard DDPM timestep sinusoidal embedding."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResidualBlock(nn.Module):
    """GroupNorm-SiLU-Conv-GroupNorm-SiLU-Conv with timestep + cond injection."""
    def __init__(self, in_ch, out_ch, time_dim=256, cond_dim=384):
        super().__init__()
        self.in_proj = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.gn1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.cond_proj = nn.Linear(cond_dim, out_ch)

    def forward(self, x, t_emb, cond):
        h = self.conv1(F.silu(self.gn1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None] + self.cond_proj(cond)[:, :, None, None]
        h = self.conv2(F.silu(self.gn2(h)))
        return h + self.in_proj(x)


class CondUNet(nn.Module):
    """Compact conditional UNet for DDPM noise prediction.

    Conditioning: global vector (B, cond_dim) -- the JEPA "healthy" latent.
    """
    def __init__(self, in_chans=3, base_ch=32, time_dim=256, cond_dim=384):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2), nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.time_dim = time_dim
        # Encoder
        self.in_conv = nn.Conv2d(in_chans, base_ch, 3, padding=1)
        self.down1 = ResidualBlock(base_ch, base_ch * 2, time_dim, cond_dim)
        self.down2 = ResidualBlock(base_ch * 2, base_ch * 4, time_dim, cond_dim)
        # Middle
        self.mid = ResidualBlock(base_ch * 4, base_ch * 4, time_dim, cond_dim)
        # Decoder (skip connections via channel concat)
        self.up2 = ResidualBlock(base_ch * 4 + base_ch * 4, base_ch * 2, time_dim, cond_dim)
        self.up1 = ResidualBlock(base_ch * 2 + base_ch * 2, base_ch, time_dim, cond_dim)
        self.out_conv = nn.Conv2d(base_ch + base_ch, in_chans, 1)

    def forward(self, x, t, cond):
        t_emb = self.time_mlp(sinusoidal_timestep_embedding(t, self.time_dim))
        h0 = self.in_conv(x)                              # base_ch @ H
        h1 = self.down1(h0, t_emb, cond)                  # 2x  @ H
        d1 = F.avg_pool2d(h1, 2)                          # 2x  @ H/2
        h2 = self.down2(d1, t_emb, cond)                  # 4x  @ H/2
        d2 = F.avg_pool2d(h2, 2)                          # 4x  @ H/4
        m = self.mid(d2, t_emb, cond)                     # 4x  @ H/4
        u2 = F.interpolate(m, scale_factor=2, mode="nearest")
        u2 = self.up2(torch.cat([u2, h2], dim=1), t_emb, cond)  # 2x @ H/2
        u1 = F.interpolate(u2, scale_factor=2, mode="nearest")
        u1 = self.up1(torch.cat([u1, h1], dim=1), t_emb, cond)  # base_ch @ H
        return self.out_conv(torch.cat([u1, h0], dim=1))


class LatentConditionedDDPM(nn.Module):
    """DDPM diffusion model conditioned on the JEPA healthy latent.

    Linear beta schedule (standard); 1000 training timesteps; 50-step DDIM
    sampling at inference for ~5x speedup.
    """
    def __init__(self, in_chans=3, base_ch=32, cond_dim=384, num_train_timesteps=1000,
                 beta_start=1e-4, beta_end=2e-2):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        alphas = 1.0 - betas
        alphas_cum = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cum", alphas_cum)
        self.register_buffer("alphas_cum_prev",
                              torch.cat([torch.ones(1), alphas_cum[:-1]]))
        self.net = CondUNet(in_chans=in_chans, base_ch=base_ch,
                             time_dim=256, cond_dim=cond_dim)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None: noise = torch.randn_like(x0)
        a_cum = self.alphas_cum[t][:, None, None, None]
        return a_cum.sqrt() * x0 + (1 - a_cum).sqrt() * noise

    def training_loss(self, x0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B = x0.size(0)
        t = torch.randint(0, self.num_train_timesteps, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred_noise = self.net(x_t, t, cond)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def ddim_sample(self, shape, cond: torch.Tensor, num_steps: int = 50,
                    eta: float = 0.0, device: str = "cuda") -> torch.Tensor:
        """DDIM deterministic sampling (eta=0) of healthy counterfactual."""
        x = torch.randn(shape, device=device)
        t_schedule = torch.linspace(self.num_train_timesteps - 1, 0, num_steps,
                                     device=device).long()
        for i, t in enumerate(t_schedule):
            t_b = torch.full((shape[0],), int(t.item()), device=device, dtype=torch.long)
            pred_noise = self.net(x, t_b, cond)
            a_t = self.alphas_cum[t]
            a_prev = (self.alphas_cum[t_schedule[i + 1]]
                      if i + 1 < num_steps else torch.tensor(1.0, device=device))
            x0_pred = (x - (1 - a_t).sqrt() * pred_noise) / a_t.sqrt()
            sigma = eta * ((1 - a_prev) / (1 - a_t)).sqrt() * (1 - a_t / a_prev).sqrt()
            dir_xt = (1 - a_prev - sigma ** 2).clamp_min(0).sqrt() * pred_noise
            noise = torch.randn_like(x) if eta > 0 else 0.0
            x = a_prev.sqrt() * x0_pred + dir_xt + sigma * noise
        return x


__all__ = ["LatentConditionedDDPM", "CondUNet"]
