"""SDF/INR geometric tower for v9b.

A trainable implicit neural representation that learns the signed-distance
function of healthy brain anatomy. Tumor candidates are voxels where the
predicted SDF deviates significantly from the atlas template (geometric
deviation = the geometry-tower's anomaly score).

Pipeline:
  Input MRI -> small CNN -> per-voxel feature -> SIREN INR -> predicted SDF
  Anatomical SDF template (atlas) -> reference SDF
  Geometric anomaly = (predicted SDF - reference SDF)^2

Trained on healthy scans only with the loss:
  L = MSE(predicted SDF, atlas template SDF)
i.e. forces the INR to converge to "healthy brain" geometric shape.
On tumor scans, the INR cannot reproduce the atlas (because tumor warps
geometry) -> large deviation = strong geometric anomaly signal.
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometricSDFTower(nn.Module):
    """CNN encoder + per-voxel INR-style SDF predictor.

    Input: (B, 3, H, W) MRI
    Output: (B, 1, H, W) predicted SDF
    Deviation from template SDF = geometric anomaly map.
    """
    def __init__(self, image_size: int = 256, base_ch: int = 32):
        super().__init__()
        self.image_size = image_size
        self.enc = nn.Sequential(
            nn.Conv2d(3, base_ch, 3, padding=1),
            nn.GroupNorm(min(8, base_ch), base_ch), nn.SiLU(),
            nn.Conv2d(base_ch, base_ch * 2, 3, padding=1, stride=2),
            nn.GroupNorm(min(8, base_ch * 2), base_ch * 2), nn.SiLU(),
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, padding=1, stride=2),
            nn.GroupNorm(min(8, base_ch * 4), base_ch * 4), nn.SiLU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1),
            nn.GroupNorm(min(8, base_ch * 2), base_ch * 2), nn.SiLU(),
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1),
            nn.GroupNorm(min(8, base_ch), base_ch), nn.SiLU(),
            nn.Conv2d(base_ch, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc(x)
        sdf = self.dec(h)
        # Match output size if rounding mismatches
        if sdf.shape[-2:] != x.shape[-2:]:
            sdf = F.interpolate(sdf, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return sdf

    def training_loss(self, x: torch.Tensor, sdf_template: torch.Tensor) -> torch.Tensor:
        """Train on healthy scans only: predicted SDF should match the
        atlas template (the model learns "this is what healthy SDF looks like")."""
        pred = self.forward(x)
        return F.mse_loss(pred, sdf_template)

    @torch.no_grad()
    def anomaly_map(self, x: torch.Tensor, sdf_template: torch.Tensor) -> torch.Tensor:
        """Geometric anomaly = (predicted_SDF - template_SDF)^2.

        On healthy scans this is near zero (model trained to match template).
        On tumor scans, the geometry deviates -> high anomaly.
        """
        pred = self.forward(x)
        return (pred - sdf_template).pow(2)


__all__ = ["GeometricSDFTower"]
