"""Conditioning encoders for the CrossJEPA predictor.

The CrossJEPA paper (Nazar et al. 2511.18424) shows that conditioning
the predictor on "nuisance" cross-domain information — pose, color,
modality identity — acts as a **gradient sink**: the predictor absorbs
the nuisance, freeing the encoder to learn semantic invariants.

For brain MRI we need:

Method 1 (3D volume -> 2D slice):
  - plane          : 3-way categorical (axial, sagittal, coronal)
  - slice_idx      : continuous, normalized to [0, 1]
  - voxel_spacing  : (sx, sy, sz) continuous in mm
  - intensity_hist : 48-bin histogram of the chosen 2D slice

Method 2 (modality -> modality):
  - target_modality_id : 4-way categorical sinusoidal
  - patch_position     : (y, x) for 2D positional context
  - intensity_hist     : 48-bin per-channel histogram of the context
                         subset, concatenated

All encoders produce a fixed-dim embedding that gets concatenated/added
to the predictor's input stream.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.nn as nn


# Plane / modality enumerations — keep in sync with dataset_3d.py
PLANES = ('axial', 'sagittal', 'coronal')
MODALITIES = ('T1', 'T1c', 'T2', 'FLAIR')
PLANE_TO_IDX = {p: i for i, p in enumerate(PLANES)}
MODALITY_TO_IDX = {m: i for i, m in enumerate(MODALITIES)}


def _sinusoidal_embedding(values: torch.Tensor, dim: int,
                            max_period: float = 10000.0) -> torch.Tensor:
    """Standard sinusoidal positional embedding (Vaswani et al.) applied to
    arbitrary continuous values. `values` is (B,) or (B, K); output is
    (B, dim) or (B, K, dim).
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=values.device) / half
    )
    args = values.unsqueeze(-1) * freqs
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
    return emb


class PlaneEmbedding(nn.Module):
    """3-way categorical embedding for {axial, sagittal, coronal}."""

    def __init__(self, dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(len(PLANES), dim)

    def forward(self, plane_idx: torch.Tensor) -> torch.Tensor:
        return self.embed(plane_idx)


class ModalityEmbedding(nn.Module):
    """4-way categorical embedding for {T1, T1c, T2, FLAIR}, sinusoidal
    initialization so the embeddings start with smooth structure rather
    than a random table — easier for the predictor to absorb."""

    def __init__(self, dim: int = 32):
        super().__init__()
        self.dim = dim
        # Initialize from sinusoidal embedding of indices 0..3
        init = _sinusoidal_embedding(
            torch.arange(len(MODALITIES), dtype=torch.float32), dim)
        self.embed = nn.Embedding(len(MODALITIES), dim)
        with torch.no_grad():
            self.embed.weight.copy_(init)

    def forward(self, modality_idx: torch.Tensor) -> torch.Tensor:
        return self.embed(modality_idx)


class ContinuousEmbedding(nn.Module):
    """Sinusoidal embedding for a continuous scalar (e.g. normalized
    slice index, voxel spacing axis). Wraps the scalar to fill the
    available frequency band by scaling to [0, max_value]."""

    def __init__(self, dim: int = 32, max_value: float = 1.0,
                 max_period: float = 1000.0):
        super().__init__()
        self.dim = dim
        self.max_value = max_value
        self.max_period = max_period

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled = x * (self.max_period / max(self.max_value, 1e-9))
        return _sinusoidal_embedding(scaled, self.dim, max_period=self.max_period)


class VoxelSpacingEmbedding(nn.Module):
    """3-D voxel spacing (sx, sy, sz) in mm. Each axis gets its own
    sinusoidal embedding; concatenated to a single per-sample vector."""

    def __init__(self, dim_per_axis: int = 16):
        super().__init__()
        self.dim_per_axis = dim_per_axis
        self.per_axis = ContinuousEmbedding(dim_per_axis, max_value=5.0)

    def forward(self, spacing: torch.Tensor) -> torch.Tensor:
        # spacing: (B, 3) in mm
        embs = [self.per_axis(spacing[:, i]) for i in range(3)]
        return torch.cat(embs, dim=-1)

    @property
    def out_dim(self) -> int:
        return 3 * self.dim_per_axis


class IntensityHistogramEmbedding(nn.Module):
    """Encodes a 48-bin intensity histogram (computed at data-prep time)
    into a fixed-dim vector via a small MLP. For Method 2, the per-
    channel histograms of the context subset get concatenated before
    being passed in (so the MLP input dim is 48 * n_channels_in_context).
    """

    def __init__(self, hist_dim: int = 48, out_dim: int = 64,
                 n_channels_max: int = 4):
        super().__init__()
        self.hist_dim = hist_dim
        self.out_dim = out_dim
        self.n_channels_max = n_channels_max
        in_dim = hist_dim * n_channels_max
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        # hist: (B, hist_dim * n_channels_max) — padded with zeros for
        # absent channels so the MLP sees a constant input width.
        return self.mlp(hist)


class Method1Conditioning(nn.Module):
    """Combined conditioning vector for Method 1 (3D->2D).

    Inputs (all batched):
      plane_idx        : (B,)        long, 0..2
      slice_idx_norm   : (B,)        float in [0, 1] (index / volume_depth)
      voxel_spacing    : (B, 3)      float, mm
      intensity_hist   : (B, hist_dim) float, 48-bin slice histogram

    Output: (B, out_dim) embedding to be added to the predictor's mask
    token at the target slice position (the gradient sink).
    """

    def __init__(self, embed_dim: int = 192, hist_dim: int = 48):
        super().__init__()
        self.plane = PlaneEmbedding(dim=32)
        self.slice = ContinuousEmbedding(dim=32, max_value=1.0)
        self.spacing = VoxelSpacingEmbedding(dim_per_axis=16)
        # Method 1 history is single-channel (the rendered 2D slice)
        self.hist = IntensityHistogramEmbedding(
            hist_dim=hist_dim, out_dim=64, n_channels_max=1)
        in_dim = 32 + 32 + self.spacing.out_dim + self.hist.out_dim
        self.proj = nn.Linear(in_dim, embed_dim)
        self.out_dim = embed_dim

    def forward(self, plane_idx: torch.Tensor, slice_idx_norm: torch.Tensor,
                voxel_spacing: torch.Tensor, intensity_hist: torch.Tensor) -> torch.Tensor:
        parts = [
            self.plane(plane_idx),
            self.slice(slice_idx_norm),
            self.spacing(voxel_spacing),
            self.hist(intensity_hist),
        ]
        return self.proj(torch.cat(parts, dim=-1))


class Method1CrossModalConditioning(nn.Module):
    """LEAK-FREE conditioning for cross-modal Method 1 (3D-of-source-modality
    -> 2D-slice-of-target-modality).

    Fixes two CrossJEPA-correctness issues vs Method1Conditioning:

      Issue 1 (intensity-hist leak): the old conditioning included the
        TARGET slice's intensity histogram, which gave the predictor
        almost the answer (~"the answer looks like X intensities").
        This conditioning omits intensity_hist entirely. The predictor
        gets only position/identity information, NOT content.

      Issue 2 (real cross-modal gap): adds source_modality_idx and
        target_modality_idx as true nuisances. Encoder is given a T1
        volume; predictor knows "I'm trying to predict the T1c
        appearance of slice 87". The cross-modal gap (T1 vs T1c
        contrast) is what the encoder must bridge by learning
        intensity-invariant anatomical features.

    Inputs (all batched):
      plane_idx           (B,)   long, 0..2  (axial/sagittal/coronal)
      slice_idx_norm      (B,)   float in [0, 1]
      voxel_spacing       (B, 3) float in mm
      source_modality_idx (B,)   long, 0..3  (the volume's modality)
      target_modality_idx (B,)   long, 0..3  (the slice's modality)

    Output: (B, embed_dim) — added to predictor's query token (gradient
    sink, exactly as CrossJEPA prescribes).
    """

    def __init__(self, embed_dim: int = 192):
        super().__init__()
        self.plane = PlaneEmbedding(dim=32)
        self.slice = ContinuousEmbedding(dim=32, max_value=1.0)
        self.spacing = VoxelSpacingEmbedding(dim_per_axis=16)
        self.source_modality = ModalityEmbedding(dim=32)
        self.target_modality = ModalityEmbedding(dim=32)
        in_dim = 32 + 32 + self.spacing.out_dim + 32 + 32
        self.proj = nn.Linear(in_dim, embed_dim)
        self.out_dim = embed_dim

    def forward(self, plane_idx: torch.Tensor, slice_idx_norm: torch.Tensor,
                voxel_spacing: torch.Tensor,
                source_modality_idx: torch.Tensor,
                target_modality_idx: torch.Tensor) -> torch.Tensor:
        parts = [
            self.plane(plane_idx),
            self.slice(slice_idx_norm),
            self.spacing(voxel_spacing),
            self.source_modality(source_modality_idx),
            self.target_modality(target_modality_idx),
        ]
        return self.proj(torch.cat(parts, dim=-1))


class Method2Conditioning(nn.Module):
    """Combined conditioning vector for Method 2 (modality->modality).

    Inputs (all batched):
      target_modality_idx : (B,)        long, 0..3 — the gradient sink
      slice_pose          : (B, 2)      float (y, x) patch position
                                         in [0, 1]
      context_hist        : (B, 48*4)   float, per-channel histogram of
                                         the context subset, channels NOT
                                         in subset zeroed

    Output: (B, embed_dim) to be added to the predictor's target token.
    """

    def __init__(self, embed_dim: int = 192, hist_dim: int = 48):
        super().__init__()
        self.modality = ModalityEmbedding(dim=32)
        self.pose_y = ContinuousEmbedding(dim=16, max_value=1.0)
        self.pose_x = ContinuousEmbedding(dim=16, max_value=1.0)
        self.hist = IntensityHistogramEmbedding(
            hist_dim=hist_dim, out_dim=64, n_channels_max=len(MODALITIES))
        in_dim = 32 + 16 + 16 + self.hist.out_dim
        self.proj = nn.Linear(in_dim, embed_dim)
        self.out_dim = embed_dim

    def forward(self, target_modality_idx: torch.Tensor, slice_pose: torch.Tensor,
                context_hist: torch.Tensor) -> torch.Tensor:
        parts = [
            self.modality(target_modality_idx),
            self.pose_y(slice_pose[:, 0]),
            self.pose_x(slice_pose[:, 1]),
            self.hist(context_hist),
        ]
        return self.proj(torch.cat(parts, dim=-1))


__all__ = [
    'PLANES', 'MODALITIES', 'PLANE_TO_IDX', 'MODALITY_TO_IDX',
    'PlaneEmbedding', 'ModalityEmbedding', 'ContinuousEmbedding',
    'VoxelSpacingEmbedding', 'IntensityHistogramEmbedding',
    'Method1Conditioning', 'Method1CrossModalConditioning',
    'Method2Conditioning',
]
