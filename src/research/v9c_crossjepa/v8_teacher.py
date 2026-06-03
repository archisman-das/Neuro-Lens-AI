"""Frozen v8 ConvNeXt-Tiny encoder wrapper — Method 1's target side.

The shipped v8 segmentation model is a `tu-convnext_tiny.fb_in22k_ft_in1k`
encoder + UNet decoder (see src/train_segmentation_v7.py for build code).
For CrossJEPA Method 1 we only need the ENCODER side — specifically the
final-stage feature map, average-pooled to a fixed-length embedding the
predictor regresses against.

CrossJEPA principle: the target MUST be frozen (every learnable-teacher
variant they tried collapsed). This module enforces that with `eval()` +
`requires_grad_(False)` + a runtime assertion.

We accept either:
  - The full UNet checkpoint (.pt) — we extract the encoder via
    `model.encoder` (segmentation_models_pytorch convention)
  - A pre-saved encoder-only checkpoint — direct load

Embedding dimension:
  ConvNeXt-Tiny's final stage = 768 channels. After global-avg-pool over
  the (H/32, W/32) spatial grid we get a (B, 768) vector — that's what
  the predictor targets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ConvNeXt-Tiny final-stage channel count (the embedding dim we expose)
V8_EMBED_DIM = 768


class V8FrozenTeacher(nn.Module):
    """Wraps the v8 ConvNeXt-Tiny encoder for embedding extraction.

    Two ways to populate it:
      a) `V8FrozenTeacher.from_unet_checkpoint(ckpt_path)` — load the full
         shipped UNet, peel off `model.encoder` (segmentation_models_pytorch
         convention).
      b) `V8FrozenTeacher(encoder_module)` — pass an already-instantiated
         encoder (e.g. from a fresh smp build).

    All forward calls run under `torch.no_grad()`; the underlying
    parameters are `requires_grad_(False)` so even if someone bypasses
    the no_grad context, they can't accidentally update teacher weights.
    """

    def __init__(self, encoder: nn.Module, image_size: int = 384):
        super().__init__()
        self.encoder = encoder
        self.image_size = image_size
        self.embed_dim = V8_EMBED_DIM
        # Hard-freeze + assert
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self._assert_frozen()

    def _assert_frozen(self) -> None:
        for n, p in self.encoder.named_parameters():
            assert not p.requires_grad, (
                f'V8FrozenTeacher: param {n} has requires_grad=True. '
                'CrossJEPA mandates a frozen teacher — refusing to proceed.'
            )

    @classmethod
    def from_unet_checkpoint(cls, ckpt_path: str | Path,
                              encoder_name: str = 'tu-convnext_tiny.fb_in22k_ft_in1k',
                              in_channels: int = 3,
                              image_size: int = 384,
                              device: str = 'cuda') -> 'V8FrozenTeacher':
        """Rebuild the v8 UNet, load weights, then peel off the encoder.
        Requires segmentation_models_pytorch (smp) at import time.
        """
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise RuntimeError(
                'segmentation_models_pytorch is required to rebuild the v8 '
                'UNet. Install via `pip install segmentation-models-pytorch`.'
            ) from exc
        net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=1,
        )
        sd = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        # Tolerate prefixes from DDP / Lightning checkpoints
        sd = {k.replace('module.', '').replace('model.', ''): v for k, v in sd.items()}
        miss, unexp = net.load_state_dict(sd, strict=False)
        if miss:
            print(f'  [V8FrozenTeacher] missing {len(miss)} keys (decoder OK to miss)')
        # Move to device + extract encoder
        net = net.to(device).eval()
        teacher = cls(net.encoder, image_size=image_size)
        teacher.to(device)
        return teacher

    @torch.no_grad()
    def embed_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a batch of 2D slices.
        Args:
          x: (B, C, H, W) float tensor in [0, 1], H == W == self.image_size
             expected (will be resized if not).
        Returns:
          (B, V8_EMBED_DIM) embedding from the final encoder stage,
          global-avg-pooled.
        """
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                               mode='bilinear', align_corners=False)
        # smp encoders return a list of feature maps at each stage; we take
        # the deepest one.
        feats = self.encoder(x)
        deepest = feats[-1] if isinstance(feats, (list, tuple)) else feats
        # (B, C, H', W') -> (B, C)
        pooled = F.adaptive_avg_pool2d(deepest, output_size=1).flatten(1)
        # Sanity: must match V8_EMBED_DIM
        assert pooled.shape[-1] == self.embed_dim, (
            f'V8FrozenTeacher: expected {self.embed_dim}-d embedding, '
            f'got {pooled.shape[-1]}. Was the encoder built correctly?'
        )
        return pooled

    @torch.no_grad()
    def embed_slice(self, slice_rgb_uint8: np.ndarray) -> np.ndarray:
        """Convenience: embed a single (H, W, 3) uint8 numpy slice. Used
        by the cache-builder which sees one slice at a time."""
        arr = slice_rgb_uint8.astype(np.float32) / 255.0
        device = next(self.encoder.parameters()).device
        x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
        emb = self.embed_batch(x)
        return emb.squeeze(0).cpu().numpy()


__all__ = ['V8FrozenTeacher', 'V8_EMBED_DIM']
