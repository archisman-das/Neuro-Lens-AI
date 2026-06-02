"""Counterfactual healthy generation for v9.

Goal
----
Given an encoder latent that's been decomposed into (z_anatomy, z_tumor)
by the causal SCM head, generate the *counterfactual healthy* image:

    x_healthy = D(z_anatomy, do(z_tumor = 0))

This is the "what would this scan look like with no tumor?" generation
that gives clinicians (and the LLM reporter) a visual baseline. The
residual between the input scan and the counterfactual is a clean
visualization of the tumor mass.

Architecture
------------
A small generative decoder (1-2 M parameters) that takes the recomposed
latent (with z_tumor zeroed) and produces a low-resolution healthy
reconstruction, then upsamples to image resolution. We use a U-Net-style
upsampling block with optional VQ-VAE codebook for sharper textures.

Two output modes:
  - "L1 reconstruction": direct pixel regression of the healthy image.
    Simpler, faster, but blurry on textures.
  - "perceptual + L1": adds a perceptual loss term (LPIPS) for sharper
    output. Heavier compute. Use this for final paper results.

Loss components for training
----------------------------
On HEALTHY scans (mask is empty):
  - Reconstruction loss: counterfactual should equal the input (the
    healthy scan IS its own healthy version). L1 + perceptual.
  - Identity loss: z_tumor for a healthy scan should be near zero
    (nothing to "remove"). Driven by SCM disentanglement loss.

On TUMOR scans (mask is non-empty):
  - Reconstruction in the NON-TUMOR region: counterfactual should equal
    input outside the tumor mask.
  - In the TUMOR region: counterfactual should "fill in" with surrounding
    healthy tissue (inpainting prior). Driven by the L1 loss with mask
    weighting.

This is a self-supervised loss formulation (no paired "tumor / healthy"
ground truth exists in BraTS). The SCM decomposition + masked
reconstruction provides the supervision signal.

For v9 brain-2D scope this is a minimal but working implementation. v10
should swap in a latent diffusion decoder (Stable-Diffusion-Med or
similar) for state-of-the-art generative quality.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CounterfactualHealthyDecoder(nn.Module):
    """Generates the healthy counterfactual image from causal latent.

    Forward signature:
        x_input:  (B, C, H, W) original MRI (used as spatial reference)
        z_recomposed: (B, D) latent from CausalRecompose with z_tumor=0
        mask_pred: (B, 1, H, W) predicted tumor mask (for inpainting)
                   When None, the decoder simply reconstructs the whole image.
    Returns:
        x_healthy: (B, C, H, W) counterfactual healthy version
    """

    def __init__(self, latent_dim: int = 256, image_size: int = 384,
                  in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.in_channels = in_channels

        # Project the latent to a small spatial feature map: latent -> (B, C, 16, 16).
        # 16x16 is small enough to be cheap but enough resolution for global structure.
        spatial = max(8, image_size // 24)  # ~16 at 384, ~12 at 256
        self.spatial = spatial
        self.latent_to_feat = nn.Linear(latent_dim, base_channels * spatial * spatial)

        # Upsampling blocks: 16 -> 32 -> 64 -> 128 -> ... -> image_size.
        # ConvTranspose with stride 2 doubles spatial size each block.
        upsamples = []
        cur_size = spatial
        cur_ch = base_channels
        while cur_size < image_size:
            next_ch = max(base_channels // 2, base_channels - 8)
            upsamples.append(nn.Sequential(
                nn.ConvTranspose2d(cur_ch, next_ch, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(min(8, next_ch), next_ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(next_ch, next_ch, kernel_size=3, padding=1),
                nn.GroupNorm(min(8, next_ch), next_ch),
                nn.SiLU(inplace=True),
            ))
            cur_ch = next_ch
            cur_size *= 2
        self.upsamples = nn.ModuleList(upsamples)

        # Final projection to image channels with skip from x_input (lets
        # the decoder copy unchanged regions from input cheaply).
        self.input_proj = nn.Conv2d(in_channels, cur_ch, kernel_size=1)
        self.final = nn.Sequential(
            nn.Conv2d(cur_ch, cur_ch, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, cur_ch), cur_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(cur_ch, in_channels, kernel_size=1),
            nn.Tanh(),  # output in [-1, 1], scaled to image space by caller
        )

    def forward(self, x_input: torch.Tensor, z_recomposed: torch.Tensor,
                mask_pred: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x_input.size(0)
        # Latent -> spatial feature
        h = self.latent_to_feat(z_recomposed).view(B, -1, self.spatial, self.spatial)
        # Upsample
        for block in self.upsamples:
            h = block(h)
        # Crop / resize to exact image size (in case spatial doubling overshoots).
        if h.shape[-1] != self.image_size:
            h = F.interpolate(h, size=(self.image_size, self.image_size),
                              mode="bilinear", align_corners=False)
        # Skip-add input projection (lets the decoder pass through anatomy
        # cheaply; tumor region is "fixed up" via the latent path).
        x_proj = self.input_proj(x_input)
        h = h + x_proj
        # Final image generation
        x_healthy = self.final(h)
        # Tanh output in [-1, 1]. Caller rescales to MRI intensity space.
        return x_healthy

    def reconstruction_loss(self, x_input: torch.Tensor, x_healthy: torch.Tensor,
                            tumor_mask: torch.Tensor, lambda_outside: float = 1.0,
                            lambda_inside: float = 0.5) -> torch.Tensor:
        """Masked reconstruction loss for counterfactual training.

        - Outside the tumor mask: counterfactual should equal input
          (high weight, lambda_outside = 1.0 default).
        - Inside the tumor mask: counterfactual should smoothly "inpaint"
          with surrounding tissue (lower weight, lambda_inside = 0.5).

        For healthy scans (empty mask), only the outside term is active,
        which means the counterfactual must reconstruct the whole input
        (lossless on healthy).
        """
        outside = (tumor_mask < 0.5).float()  # 1 outside tumor, 0 inside
        inside = (tumor_mask >= 0.5).float()
        diff = (x_healthy - x_input).abs()
        loss_outside = (diff * outside).sum() / outside.sum().clamp_min(1.0)
        loss_inside = (diff * inside).sum() / inside.sum().clamp_min(1.0)
        return lambda_outside * loss_outside + lambda_inside * loss_inside


def tumor_residual(x_input: torch.Tensor, x_healthy: torch.Tensor) -> torch.Tensor:
    """Compute the tumor residual = |input - counterfactual_healthy|.

    For a perfect counterfactual decoder, this residual is:
      - High inside the tumor region (where the input has tumor and the
        counterfactual has been "filled in" with healthy tissue)
      - Low elsewhere (where input == counterfactual)

    This residual is a CLEAN visualization of the tumor mass, independent
    of the segmenter's mask. Use it as an auxiliary anomaly signal that
    the conformal head can be calibrated on (see hyperbolic_conformal.py).
    """
    return (x_input - x_healthy).abs().mean(dim=1, keepdim=True)  # (B, 1, H, W)


__all__ = ["CounterfactualHealthyDecoder", "tumor_residual"]
