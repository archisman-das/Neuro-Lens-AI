"""Lightweight 3D ViT for CrossJEPA Method 1.

Conservative architecture (ViT-S-style):
  patch_size  = 16            (cube; matches the proposal's "16^3 patch")
  embed_dim   = 384
  depth       = 12
  heads       = 6             (head_dim = 64)
  mlp_ratio   = 4.0

On a typical BraTS volume after cropping to (192, 192, 144) and patching
at 16^3: tokens = 12 * 12 * 9 = 1296. Manageable on A100 80GB with
batch_size=2. On T4 16GB we'll need to crop more aggressively (see the
volume_to_slice trainer for the crop policy).

We reuse TransformerBlock from src/research/jepa.py rather than
re-implementing — it's identical to a standard MHA + MLP block and the
shapes work for 3D once we move to (B, N_tokens, D) sequence form.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn

# Reuse the existing ViT block from v9b — same shape semantics
from src.research.jepa import TransformerBlock


def _3d_sincos_posemb(grid: Sequence[int], embed_dim: int) -> torch.Tensor:
    """3D sin-cos positional embedding. grid = (d, h, w) tokens.
    Returns (1, d*h*w, embed_dim)."""
    assert embed_dim % 6 == 0, (
        f'embed_dim must be divisible by 6 for 3D sin-cos; got {embed_dim}')
    d, h, w = grid
    per_axis = embed_dim // 3        # one third of dim per spatial axis
    half = per_axis // 2

    def _1d(n: int) -> torch.Tensor:
        pos = torch.arange(n, dtype=torch.float32)
        omega = torch.arange(half, dtype=torch.float32) / half
        omega = 1.0 / (10000.0 ** omega)
        out = pos[:, None] * omega[None]
        return torch.cat([torch.sin(out), torch.cos(out)], dim=-1)   # (n, per_axis)

    ed = _1d(d).unsqueeze(1).unsqueeze(2).expand(d, h, w, per_axis)
    eh = _1d(h).unsqueeze(0).unsqueeze(2).expand(d, h, w, per_axis)
    ew = _1d(w).unsqueeze(0).unsqueeze(1).expand(d, h, w, per_axis)
    out = torch.cat([ed, eh, ew], dim=-1).reshape(d * h * w, embed_dim)
    return out.unsqueeze(0)


class PatchEmbed3D(nn.Module):
    """3D non-overlapping patch embedding via Conv3d."""

    def __init__(self, volume_size: Sequence[int] = (192, 192, 144),
                 patch_size: int = 16, in_chans: int = 4, embed_dim: int = 384):
        super().__init__()
        d, h, w = volume_size
        assert d % patch_size == 0 and h % patch_size == 0 and w % patch_size == 0, (
            f'volume_size {volume_size} must be divisible by patch_size {patch_size}'
        )
        self.volume_size = tuple(volume_size)
        self.patch_size = patch_size
        self.grid = (d // patch_size, h // patch_size, w // patch_size)
        self.num_patches = self.grid[0] * self.grid[1] * self.grid[2]
        self.proj = nn.Conv3d(in_chans, embed_dim,
                                kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W) -> (B, N, embed_dim)
        z = self.proj(x)                                  # (B, D', H', W', E)
        return z.flatten(2).transpose(1, 2)               # (B, N, E)


class ViT3DEncoder(nn.Module):
    """3D ViT used as the learnable context encoder in Method 1.

    Same trunk style as v9b's ViTEncoder but on 3D patches. Outputs
    per-patch tokens; the predictor consumes those + the (plane, slice,
    pose, hist) conditioning to predict the 2D-slice embedding.
    """

    def __init__(self, volume_size: Sequence[int] = (192, 192, 144),
                 patch_size: int = 16, in_chans: int = 4, embed_dim: int = 384,
                 depth: int = 12, heads: int = 6, mlp_ratio: float = 4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed3D(volume_size, patch_size, in_chans, embed_dim)
        self.register_buffer(
            'pos_embed',
            _3d_sincos_posemb(self.patch_embed.grid, embed_dim),
            persistent=False,
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.grid = self.patch_embed.grid

    def forward(self, x: torch.Tensor,
                 keep_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns (B, N, embed_dim) — all tokens if keep_indices is None,
        otherwise the gathered subset."""
        z = self.patch_embed(x) + self.pos_embed.to(x.device)
        if keep_indices is not None:
            z = z.gather(1, keep_indices.unsqueeze(-1).expand(-1, -1, z.size(-1)))
        for blk in self.blocks:
            z = blk(z)
        return self.norm(z)

    def pooled_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Mean-pool over all tokens — useful for the predictor's
        "summarize the volume" cross-attention input."""
        z = self.forward(x)
        return z.mean(dim=1)


__all__ = ['PatchEmbed3D', 'ViT3DEncoder', '_3d_sincos_posemb']
