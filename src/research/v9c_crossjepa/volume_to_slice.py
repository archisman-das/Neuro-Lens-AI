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


# ---------------------------------------------------------------------------
# Multi-teacher Method 1 — Option A and Option C
# ---------------------------------------------------------------------------
# Two strategies for combining multiple frozen teachers (e.g. v8 + DINOv2)
# inside one CrossJEPA training run. Both share the 3D ViT encoder and the
# conditioning module; they differ in HOW the predictor head is structured.
#
#   Option A ("dual heads"): predictor emits one tensor PER teacher,
#     loss is the weighted sum across teachers. Encoder must satisfy
#     all teachers simultaneously => richer learned representation.
#
#   Option C ("teacher id as conditioning"): single predictor head,
#     output dim = max teacher embed_dim. Per-step, randomly sample which
#     teacher to target; pass that teacher_id as part of the conditioning
#     vector (the gradient sink absorbs the choice). Forces the encoder
#     to learn teacher-invariant features because the gradient signal
#     averages over teachers.


from typing import List, Sequence as _Seq

from .teachers import BaseFrozenTeacher


class MultiHeadSliceEmbeddingPredictor(nn.Module):
    """Predictor with K parallel output heads, one per teacher.

    Shared trunk (predictor blocks + cross-attention over volume tokens)
    then K independent linear projections, one to each teacher's embed_dim.
    """

    def __init__(self, encoder_dim: int, predictor_dim: int,
                 depth: int, heads: int, conditioning_dim: int,
                 teacher_embed_dims: _Seq[int]):
        super().__init__()
        self.proj_volume = nn.Linear(encoder_dim, predictor_dim)
        self.proj_cond = nn.Linear(conditioning_dim, predictor_dim)
        self.query_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.trunc_normal_(self.query_token, std=0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(predictor_dim, heads, mlp_ratio=2.0)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(predictor_dim)
        # One projection head per teacher
        self.heads = nn.ModuleList([
            nn.Linear(predictor_dim, d) for d in teacher_embed_dims
        ])
        self.n_teachers = len(teacher_embed_dims)

    def forward(self, volume_tokens: torch.Tensor,
                conditioning: torch.Tensor) -> List[torch.Tensor]:
        B = volume_tokens.size(0)
        ctx = self.proj_volume(volume_tokens)
        cond = self.proj_cond(conditioning).unsqueeze(1)
        q = self.query_token.expand(B, 1, -1) + cond
        seq = torch.cat([q, ctx], dim=1)
        for blk in self.blocks:
            seq = blk(seq)
        h = self.norm(seq[:, :1, :]).squeeze(1)        # (B, predictor_dim)
        return [head(h) for head in self.heads]        # list of (B, dim_t)


class TeacherIdConditionedPredictor(nn.Module):
    """Single output head (max embed_dim across all teachers) +
    teacher-id embedding added to the conditioning vector (the gradient
    sink). At training time, one teacher is sampled per step; the loss
    compares the prediction's first `teacher.embed_dim` dims to that
    teacher's target.

    At inference we can either:
      - call once per teacher_id and ensemble the residuals, or
      - call with a fixed teacher_id and use its residual as the anomaly score.
    """

    def __init__(self, encoder_dim: int, predictor_dim: int,
                 depth: int, heads: int, conditioning_dim: int,
                 out_dim: int, n_teachers: int):
        super().__init__()
        self.proj_volume = nn.Linear(encoder_dim, predictor_dim)
        self.proj_cond = nn.Linear(conditioning_dim, predictor_dim)
        # teacher_id embedding — sinusoidal-ish, learnable
        self.teacher_embed = nn.Embedding(n_teachers, predictor_dim)
        nn.init.trunc_normal_(self.teacher_embed.weight, std=0.02)
        self.query_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.trunc_normal_(self.query_token, std=0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(predictor_dim, heads, mlp_ratio=2.0)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(predictor_dim)
        self.proj_out = nn.Linear(predictor_dim, out_dim)
        self.out_dim = out_dim
        self.n_teachers = n_teachers

    def forward(self, volume_tokens: torch.Tensor,
                conditioning: torch.Tensor,
                teacher_id: torch.Tensor) -> torch.Tensor:
        """
        volume_tokens  : (B, N, encoder_dim)
        conditioning   : (B, conditioning_dim)
        teacher_id     : (B,) long — which teacher to predict for
        Returns (B, out_dim) — caller slices the first teacher.embed_dim
        dims when comparing to that teacher's target.
        """
        B = volume_tokens.size(0)
        ctx = self.proj_volume(volume_tokens)
        cond = self.proj_cond(conditioning).unsqueeze(1)
        tid = self.teacher_embed(teacher_id).unsqueeze(1)   # (B, 1, P)
        # Query token absorbs BOTH the conditioning AND the teacher-id —
        # CrossJEPA gradient sink for both nuisances simultaneously.
        q = self.query_token.expand(B, 1, -1) + cond + tid
        seq = torch.cat([q, ctx], dim=1)
        for blk in self.blocks:
            seq = blk(seq)
        h = self.norm(seq[:, :1, :]).squeeze(1)
        return self.proj_out(h)


class Vol2SliceModelMulti(nn.Module):
    """Multi-teacher Method 1. Holds a list of frozen teachers and a
    predictor configured per `mode`:

        mode='dual_heads'  -> Option A. Per-teacher prediction heads,
                              loss = sum_i weights[i] * L1(pred_i, target_i)
        mode='teacher_id'  -> Option C. Single output head, teacher_id
                              fed as conditioning; one teacher sampled
                              per step. The query token absorbs the
                              teacher-id choice (gradient sink).

    For Option C the single output dim is max(teacher.embed_dim for
    teacher in teachers) — predictions are sliced to each teacher's dim
    before computing loss. This keeps the param count modest while
    supporting heterogeneous teacher widths.
    """

    def __init__(self, teachers: List[BaseFrozenTeacher],
                 mode: str = 'dual_heads',
                 volume_size: _Seq[int] = (144, 192, 192),
                 in_chans: int = 1, patch_size: int = 16,
                 encoder_dim: int = 384, encoder_depth: int = 12,
                 encoder_heads: int = 6,
                 predictor_dim: int = 192, predictor_depth: int = 6,
                 hist_dim: int = 48,
                 teacher_weights: Optional[_Seq[float]] = None):
        super().__init__()
        if mode not in ('dual_heads', 'teacher_id'):
            raise ValueError(f"mode must be 'dual_heads' or 'teacher_id', got {mode!r}")
        if not teachers:
            raise ValueError('Need at least 1 teacher')
        self.mode = mode
        self.encoder = ViT3DEncoder(volume_size=volume_size,
                                      patch_size=patch_size, in_chans=in_chans,
                                      embed_dim=encoder_dim, depth=encoder_depth,
                                      heads=encoder_heads)
        self.conditioning = Method1Conditioning(embed_dim=predictor_dim,
                                                  hist_dim=hist_dim)
        teacher_dims = [t.embed_dim for t in teachers]
        if mode == 'dual_heads':
            self.predictor = MultiHeadSliceEmbeddingPredictor(
                encoder_dim=encoder_dim, predictor_dim=predictor_dim,
                depth=predictor_depth, heads=encoder_heads,
                conditioning_dim=predictor_dim,
                teacher_embed_dims=teacher_dims,
            )
        else:  # teacher_id
            self.predictor = TeacherIdConditionedPredictor(
                encoder_dim=encoder_dim, predictor_dim=predictor_dim,
                depth=predictor_depth, heads=encoder_heads,
                conditioning_dim=predictor_dim,
                out_dim=max(teacher_dims), n_teachers=len(teachers),
            )
        # Per-teacher loss weights (Option A only; ignored in Option C)
        if teacher_weights is None:
            teacher_weights = [1.0] * len(teachers)
        assert len(teacher_weights) == len(teachers)
        self.teacher_weights = list(teacher_weights)
        # Teachers held off the parameter list (same trick as Vol2SliceModel)
        object.__setattr__(self, '_teachers', list(teachers))

    @property
    def teachers(self) -> List[BaseFrozenTeacher]:
        return self._teachers

    @property
    def n_teachers(self) -> int:
        return len(self._teachers)

    def training_step(self, batch: dict) -> dict:
        if self.mode == 'dual_heads':
            return self._training_step_dual(batch)
        return self._training_step_teacher_id(batch)

    def _shared_tokens_cond(self, batch: dict) -> tuple:
        tokens = self.encoder(batch['volume'])
        cond = self.conditioning(
            batch['plane_idx'], batch['slice_idx_norm'],
            batch['voxel_spacing'], batch['intensity_hist'])
        return tokens, cond

    # ----- Option A: dual-heads -----------------------------------------
    def _training_step_dual(self, batch: dict) -> dict:
        tokens, cond = self._shared_tokens_cond(batch)
        preds = self.predictor(tokens, cond)              # list of (B, dim_t)
        total_loss = 0.0
        out = {}
        for i, (t, w) in enumerate(zip(self._teachers, self.teacher_weights)):
            with torch.no_grad():
                tgt = t.embed_batch(batch['target_slice_rgb'])
            l_i = F.smooth_l1_loss(preds[i], tgt)
            total_loss = total_loss + w * l_i
            with torch.no_grad():
                cos_i = F.cosine_similarity(preds[i], tgt, dim=-1).mean()
            out[f'loss_t{i}'] = l_i.detach()
            out[f'cos_t{i}'] = cos_i
        out['loss'] = total_loss
        return out

    # ----- Option C: teacher-id conditioning ---------------------------
    def _training_step_teacher_id(self, batch: dict) -> dict:
        B = batch['volume'].size(0)
        device = batch['volume'].device
        # Sample one teacher uniformly per sample. Could be fixed per batch
        # (same teacher across the whole batch) for slightly cheaper
        # teacher forward; per-sample sampling reduces gradient variance.
        teacher_id = torch.randint(0, self.n_teachers, (B,), device=device)
        tokens, cond = self._shared_tokens_cond(batch)
        pred = self.predictor(tokens, cond, teacher_id)   # (B, out_dim)
        # Build targets per-sample by routing to the right teacher.
        # For batching efficiency: compute each teacher's targets ONCE on
        # the full batch, then index by teacher_id mask. (Cheaper than
        # per-sample teacher calls.)
        total_loss = 0.0
        out = {}
        loss_terms = 0
        for ti, teacher in enumerate(self._teachers):
            mask = (teacher_id == ti)
            if not mask.any():
                continue
            with torch.no_grad():
                tgt_full = teacher.embed_batch(batch['target_slice_rgb'])
            tgt = tgt_full[mask]                          # (Bi, dim_t)
            pred_sliced = pred[mask, :teacher.embed_dim]  # (Bi, dim_t)
            l_i = F.smooth_l1_loss(pred_sliced, tgt)
            total_loss = total_loss + l_i
            loss_terms += 1
            with torch.no_grad():
                cos_i = F.cosine_similarity(pred_sliced, tgt, dim=-1).mean()
            out[f'loss_t{ti}'] = l_i.detach()
            out[f'cos_t{ti}'] = cos_i
        out['loss'] = total_loss / max(loss_terms, 1)
        return out

    @torch.no_grad()
    def anomaly_score(self, volume: torch.Tensor, target_slice_rgb: torch.Tensor,
                       plane_idx: torch.Tensor, slice_idx_norm: torch.Tensor,
                       voxel_spacing: torch.Tensor,
                       intensity_hist: torch.Tensor,
                       teacher_weights: Optional[_Seq[float]] = None,
                       teacher_id: Optional[int] = None) -> torch.Tensor:
        """Per-sample anomaly score.

        - dual_heads: weighted sum of per-teacher L2 residuals (weights
          default to self.teacher_weights).
        - teacher_id: if `teacher_id` is an int, compute residual for that
          specific teacher; if None, ensemble across all teachers.
        """
        batch = {
            'volume': volume, 'target_slice_rgb': target_slice_rgb,
            'plane_idx': plane_idx, 'slice_idx_norm': slice_idx_norm,
            'voxel_spacing': voxel_spacing, 'intensity_hist': intensity_hist,
        }
        tokens, cond = self._shared_tokens_cond(batch)
        if self.mode == 'dual_heads':
            preds = self.predictor(tokens, cond)
            w = teacher_weights if teacher_weights is not None else self.teacher_weights
            agg = None
            for i, t in enumerate(self._teachers):
                tgt = t.embed_batch(target_slice_rgb)
                r = (preds[i] - tgt).pow(2).mean(dim=-1)
                agg = r * w[i] if agg is None else agg + w[i] * r
            return agg
        # teacher_id mode
        if teacher_id is not None:
            B = volume.size(0)
            tid = torch.full((B,), teacher_id, dtype=torch.long, device=volume.device)
            pred = self.predictor(tokens, cond, tid)
            t = self._teachers[teacher_id]
            tgt = t.embed_batch(target_slice_rgb)
            return (pred[:, :t.embed_dim] - tgt).pow(2).mean(dim=-1)
        # Ensemble across all teachers
        agg = None
        for ti, teacher in enumerate(self._teachers):
            B = volume.size(0)
            tid = torch.full((B,), ti, dtype=torch.long, device=volume.device)
            pred = self.predictor(tokens, cond, tid)
            tgt = teacher.embed_batch(target_slice_rgb)
            r = (pred[:, :teacher.embed_dim] - tgt).pow(2).mean(dim=-1)
            agg = r if agg is None else agg + r
        return agg / self.n_teachers


from .conditioning import Method1CrossModalConditioning


class Vol2SliceCrossModalModel(nn.Module):
    """End-to-end CROSS-MODAL Method 1c model. Mirrors Vol2SliceModel but
    uses Method1CrossModalConditioning (no intensity-hist leak) and
    trains on (source_modality_volume -> target_modality_slice) pairs.

    This is the CrossJEPA-correct setup for brain MRI:
      - Source = 3D volume of modality A (e.g. T1)
      - Target = 2D slice of modality B (e.g. T1c) at the same
        anatomical position
      - The semantic gap (T1 vs T1c contrast) is what the encoder must
        bridge by learning intensity-invariant anatomy
      - Conditioning carries ONLY position + modality identity, NEVER
        the answer (target's histogram)
    """

    def __init__(self, teacher: BaseFrozenTeacher,
                 volume_size: Sequence[int] = (144, 192, 192),
                 in_chans: int = 1, patch_size: int = 16,
                 encoder_dim: int = 384, encoder_depth: int = 12,
                 encoder_heads: int = 6,
                 predictor_dim: int = 192, predictor_depth: int = 6):
        super().__init__()
        self.encoder = ViT3DEncoder(volume_size=volume_size,
                                      patch_size=patch_size, in_chans=in_chans,
                                      embed_dim=encoder_dim, depth=encoder_depth,
                                      heads=encoder_heads)
        self.conditioning = Method1CrossModalConditioning(
            embed_dim=predictor_dim)
        self.predictor = SliceEmbeddingPredictor(
            encoder_dim=encoder_dim, predictor_dim=predictor_dim,
            depth=predictor_depth, heads=encoder_heads,
            conditioning_dim=predictor_dim, out_dim=teacher.embed_dim,
        )
        object.__setattr__(self, '_teacher', teacher)

    @property
    def teacher(self) -> BaseFrozenTeacher:
        return self._teacher

    def forward_predict(self, volume: torch.Tensor, plane_idx: torch.Tensor,
                          slice_idx_norm: torch.Tensor,
                          voxel_spacing: torch.Tensor,
                          source_modality_idx: torch.Tensor,
                          target_modality_idx: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(volume)
        cond = self.conditioning(plane_idx, slice_idx_norm, voxel_spacing,
                                  source_modality_idx, target_modality_idx)
        return self.predictor(tokens, cond)

    def training_step(self, batch: dict) -> dict:
        pred = self.forward_predict(
            batch['volume'], batch['plane_idx'], batch['slice_idx_norm'],
            batch['voxel_spacing'],
            batch['source_modality_idx'], batch['target_modality_idx'])
        with torch.no_grad():
            target = self.teacher.embed_batch(batch['target_slice_rgb'])
        loss = F.smooth_l1_loss(pred, target)
        with torch.no_grad():
            cos = F.cosine_similarity(pred, target, dim=-1).mean()
        return {'loss': loss, 'cos_sim': cos.detach()}

    @torch.no_grad()
    def anomaly_score(self, volume: torch.Tensor,
                       target_slice_rgb: torch.Tensor,
                       plane_idx: torch.Tensor, slice_idx_norm: torch.Tensor,
                       voxel_spacing: torch.Tensor,
                       source_modality_idx: torch.Tensor,
                       target_modality_idx: torch.Tensor) -> torch.Tensor:
        pred = self.forward_predict(volume, plane_idx, slice_idx_norm,
                                      voxel_spacing,
                                      source_modality_idx, target_modality_idx)
        target = self.teacher.embed_batch(target_slice_rgb)
        return (pred - target).pow(2).mean(dim=-1)


__all__ = [
    'SliceEmbeddingPredictor', 'Vol2SliceModel',
    'MultiHeadSliceEmbeddingPredictor', 'TeacherIdConditionedPredictor',
    'Vol2SliceModelMulti', 'Vol2SliceCrossModalModel',
]
