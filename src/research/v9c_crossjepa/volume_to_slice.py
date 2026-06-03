"""CrossJEPA Method 1: 3D volume -> 2D slice (frozen v8 ConvNeXt-Tiny teacher).

Pipeline per training step:

  3D volume (B, C, D, H, W)
        |
        v
  [ ViT3DEncoder ]                <- learnable
        |
        v   pooled token  (B, embed_dim)
        +-- + slice/plane/voxel-spacing/hist conditioning  (gradient sink)
        v
  [ SliceEmbeddingPredictor ]     <- learnable
        |
        v   predicted slice embedding  (B, V8_EMBED_DIM = 768)

  vs

  v8 ConvNeXt embedding of that same slice  (FROZEN target)

  Loss = smooth-L1 in v8 embedding space.

At inference: given any new volume, for each slice along each plane
produce a prediction; compare to the v8 teacher's actual embedding; the
residual norm becomes the per-slice anomaly score (and after the
weighted-conformal calibration it becomes a (1-alpha)-coverage anomaly
flag).
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.research.jepa import TransformerBlock
from .conditioning import Method1Conditioning
from .v8_teacher import V8FrozenTeacher, V8_EMBED_DIM
from .vit_3d import ViT3DEncoder


class SliceEmbeddingPredictor(nn.Module):
    """Predictor head that maps (volume_summary, conditioning) -> v8
    embedding for a specific slice.

    Architecture:
      - Input: (B, encoder_dim) volume summary token from the 3D ViT,
        plus the (B, predictor_dim) conditioning vector
      - Project encoder summary to predictor_dim, add conditioning, run
        through `depth` transformer blocks
      - Project back out to V8_EMBED_DIM

    We use cross-attention over the full volume's patch tokens (not just
    the pooled summary) for richer context — the conditioning vector is
    the query, the volume tokens are the keys/values.
    """

    def __init__(self, encoder_dim: int = 384, predictor_dim: int = 192,
                 depth: int = 6, heads: int = 6,
                 conditioning_dim: int = 192,
                 out_dim: int = V8_EMBED_DIM):
        super().__init__()
        self.proj_volume = nn.Linear(encoder_dim, predictor_dim)
        self.proj_cond = nn.Linear(conditioning_dim, predictor_dim)
        # Per-position query token (the "predict the slice embedding"
        # learnable summary)
        self.query_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.trunc_normal_(self.query_token, std=0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(predictor_dim, heads, mlp_ratio=2.0)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(predictor_dim)
        self.proj_out = nn.Linear(predictor_dim, out_dim)

    def forward(self, volume_tokens: torch.Tensor,
                conditioning: torch.Tensor) -> torch.Tensor:
        """
        volume_tokens: (B, N, encoder_dim) — all 3D ViT patch tokens
        conditioning : (B, conditioning_dim)
        Returns: (B, V8_EMBED_DIM) predicted slice embedding.
        """
        B = volume_tokens.size(0)
        ctx = self.proj_volume(volume_tokens)                  # (B, N, P)
        cond = self.proj_cond(conditioning).unsqueeze(1)       # (B, 1, P)
        q = self.query_token.expand(B, 1, -1) + cond           # (B, 1, P)
        # Sequence: [query, volume tokens]. Self-attention lets the query
        # token attend to the volume; the gradient sink (conditioning)
        # rides on the query.
        seq = torch.cat([q, ctx], dim=1)                       # (B, 1+N, P)
        for blk in self.blocks:
            seq = blk(seq)
        out = self.norm(seq[:, :1, :])                          # (B, 1, P)
        return self.proj_out(out.squeeze(1))                    # (B, V8_EMBED_DIM)


class Vol2SliceModel(nn.Module):
    """End-to-end Method 1 model. Holds:
      - learnable 3D ViT encoder
      - learnable predictor head (with conditioning)
      - frozen v8 teacher (passed in at construction)

    Trainable parameters = encoder + predictor + conditioning module.
    Teacher parameters are frozen and never returned by .parameters().
    """

    def __init__(self, v8_teacher: V8FrozenTeacher,
                 volume_size: Sequence[int] = (192, 192, 144),
                 in_chans: int = 4, patch_size: int = 16,
                 encoder_dim: int = 384, encoder_depth: int = 12,
                 encoder_heads: int = 6,
                 predictor_dim: int = 192, predictor_depth: int = 6,
                 hist_dim: int = 48):
        super().__init__()
        self.encoder = ViT3DEncoder(volume_size=volume_size,
                                      patch_size=patch_size, in_chans=in_chans,
                                      embed_dim=encoder_dim, depth=encoder_depth,
                                      heads=encoder_heads)
        self.conditioning = Method1Conditioning(embed_dim=predictor_dim,
                                                  hist_dim=hist_dim)
        self.predictor = SliceEmbeddingPredictor(
            encoder_dim=encoder_dim, predictor_dim=predictor_dim,
            depth=predictor_depth, heads=encoder_heads,
            conditioning_dim=predictor_dim, out_dim=v8_teacher.embed_dim,
        )
        # NOTE: we DO NOT register the teacher as a submodule so its
        # params stay out of .parameters() iter; held as a plain attr.
        object.__setattr__(self, '_teacher', v8_teacher)

    @property
    def teacher(self) -> V8FrozenTeacher:
        return self._teacher

    def forward_predict(self, volume: torch.Tensor, plane_idx: torch.Tensor,
                          slice_idx_norm: torch.Tensor,
                          voxel_spacing: torch.Tensor,
                          intensity_hist: torch.Tensor) -> torch.Tensor:
        """Predict the v8 embedding of the target slice from the 3D
        volume + conditioning. Returns (B, V8_EMBED_DIM)."""
        tokens = self.encoder(volume)                                  # (B, N, E)
        cond = self.conditioning(plane_idx, slice_idx_norm,
                                  voxel_spacing, intensity_hist)        # (B, P)
        return self.predictor(tokens, cond)

    def training_step(self, batch: dict) -> dict:
        """One CrossJEPA training step.

        batch keys (all batched tensors on the correct device):
          - volume          (B, C, D, H, W) float in [0, 1]
          - target_slice_rgb (B, 3, H_2d, W_2d) float in [0, 1]   <- for teacher
          - plane_idx       (B,)  long
          - slice_idx_norm  (B,)  float
          - voxel_spacing   (B, 3) float
          - intensity_hist  (B, hist_dim) float
        """
        # Predict
        pred = self.forward_predict(
            batch['volume'], batch['plane_idx'], batch['slice_idx_norm'],
            batch['voxel_spacing'], batch['intensity_hist'])
        # Frozen teacher — wrapped in no_grad inside .embed_batch but we
        # also forbid grads with the surrounding context for extra safety
        with torch.no_grad():
            target = self.teacher.embed_batch(batch['target_slice_rgb'])
        # Smooth-L1 (Huber) in embedding space — robust to outlier dims
        loss = F.smooth_l1_loss(pred, target)
        # Diagnostics: cosine similarity for trend tracking
        with torch.no_grad():
            cos = F.cosine_similarity(pred, target, dim=-1).mean()
        return {'loss': loss, 'cos_sim': cos.detach()}

    @torch.no_grad()
    def anomaly_score(self, volume: torch.Tensor, target_slice_rgb: torch.Tensor,
                       plane_idx: torch.Tensor, slice_idx_norm: torch.Tensor,
                       voxel_spacing: torch.Tensor,
                       intensity_hist: torch.Tensor) -> torch.Tensor:
        """Returns (B,) per-sample anomaly score = L2 residual between
        predicted and teacher embedding. Lower = more in-distribution."""
        pred = self.forward_predict(volume, plane_idx, slice_idx_norm,
                                      voxel_spacing, intensity_hist)
        target = self.teacher.embed_batch(target_slice_rgb)
        return (pred - target).pow(2).mean(dim=-1)


__all__ = ['SliceEmbeddingPredictor', 'Vol2SliceModel']
