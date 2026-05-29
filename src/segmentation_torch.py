"""PyTorch Attention U-Net for brain-tumor segmentation.

Why PyTorch and not the existing TF segmentation_models.py? TensorFlow 2.21 on
native Windows has no GPU support — only CPU — and there is no path back to
TF + CUDA on Windows without dropping to Python 3.10 / TF 2.10 / CUDA 11.2 /
cuDNN 8.1, or moving to WSL2. The user has an RTX 4060 with PyTorch 2.11 +
CUDA 12.6 already working, so we train the U-Net in PyTorch on GPU and keep
the existing TF classifier code unchanged.

The architecture mirrors the Attention U-Net described in Oktay et al. (MIDL
2018) and matches the TF reference in src/segmentation_models.py: four-level
encoder/decoder, attention gates on the skip connections, BatchNorm + ReLU +
Dropout, binary sigmoid output. Default base_filters=64 gives ~31M params; for
a small dataset on a single GPU base_filters=32 (~7M) is usually enough.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two 3x3 conv -> BN -> ReLU with Dropout in the middle."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al.) applied to a skip connection.

    g (gating from decoder, upsampled to skip resolution) and x (skip from
    encoder) are 1x1-projected to a shared dimension, summed, ReLU'd, then
    1x1-projected to a single channel + sigmoid -> attention map alpha.
    Output = alpha * x (element-wise gating on the skip connection).
    """

    def __init__(self, x_ch: int, g_ch: int, inter_ch: int):
        super().__init__()
        self.theta_x = nn.Conv2d(x_ch, inter_ch, kernel_size=1, bias=False)
        self.phi_g = nn.Conv2d(g_ch, inter_ch, kernel_size=1, bias=False)
        self.psi = nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True)

    def forward(self, x, g):
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode='bilinear', align_corners=False)
        attn = F.relu(self.theta_x(x) + self.phi_g(g), inplace=True)
        attn = torch.sigmoid(self.psi(attn))
        return x * attn


class AttentionUNet(nn.Module):
    """4-level Attention U-Net for binary segmentation. Input: (B, 3, H, W)."""

    def __init__(self, in_channels: int = 3, base_filters: int = 32, dropout: float = 0.2):
        super().__init__()
        f = base_filters
        # Encoder
        self.enc1 = ConvBlock(in_channels, f, dropout)
        self.enc2 = ConvBlock(f, f * 2, dropout)
        self.enc3 = ConvBlock(f * 2, f * 4, dropout)
        self.enc4 = ConvBlock(f * 4, f * 8, dropout)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(f * 8, f * 16, dropout)

        # Decoder
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.att4 = AttentionGate(x_ch=f * 8, g_ch=f * 8, inter_ch=f * 4)
        self.dec4 = ConvBlock(f * 16, f * 8, dropout)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.att3 = AttentionGate(x_ch=f * 4, g_ch=f * 4, inter_ch=f * 2)
        self.dec3 = ConvBlock(f * 8, f * 4, dropout)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(x_ch=f * 2, g_ch=f * 2, inter_ch=f)
        self.dec2 = ConvBlock(f * 4, f * 2, dropout)

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.att1 = AttentionGate(x_ch=f, g_ch=f, inter_ch=max(f // 2, 1))
        self.dec1 = ConvBlock(f * 2, f, dropout)

        # Output: 1-channel logits; apply sigmoid outside (or use BCEWithLogits).
        self.out_conv = nn.Conv2d(f, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        u4 = self.up4(b)
        a4 = self.att4(e4, u4)
        d4 = self.dec4(torch.cat([u4, a4], dim=1))

        u3 = self.up3(d4)
        a3 = self.att3(e3, u3)
        d3 = self.dec3(torch.cat([u3, a3], dim=1))

        u2 = self.up2(d3)
        a2 = self.att2(e2, u2)
        d2 = self.dec2(torch.cat([u2, a2], dim=1))

        u1 = self.up1(d2)
        a1 = self.att1(e1, u1)
        d1 = self.dec1(torch.cat([u1, a1], dim=1))
        return self.out_conv(d1)  # logits


def dice_coefficient(probs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Per-batch Dice on binary probability maps. Returns scalar in [0,1]."""
    probs = probs.contiguous().view(probs.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)
    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    return ((2.0 * inter + smooth) / (denom + smooth)).mean()


def iou_metric(probs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    probs = probs.contiguous().view(probs.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)
    inter = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1) - inter
    return ((inter + smooth) / (union + smooth)).mean()


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return 1.0 - dice_coefficient(probs, targets, smooth)


def combined_dice_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    dice_weight: float = 0.6,
) -> torch.Tensor:
    """Weighted Dice + BCE-with-logits, matching the TF combined_loss(0.6, 0.4)."""
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dl = dice_loss(logits, targets)
    return dice_weight * dl + (1.0 - dice_weight) * bce


__all__ = [
    'AttentionUNet',
    'ConvBlock',
    'AttentionGate',
    'dice_coefficient',
    'iou_metric',
    'dice_loss',
    'combined_dice_bce_loss',
]
