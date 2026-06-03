"""CrossJEPA Method 2: modality -> modality.

For each training pair we:
  1. pick a non-empty context subset  S subset {T1, T1c, T2, FLAIR}
  2. pick a target modality            m  not in S
  3. encode S with the shared learnable backbone
  4. ask the predictor to reconstruct the teacher-m embedding of the
     same scan's modality-m slice
  5. compute smooth-L1 loss vs the (frozen) teacher-m embedding

Architecture details:

  PerModalityPatchEmbedder
    4 small Conv2d patch embedders (one per modality). Each gets the
    SAME embed_dim so concatenated tokens flow into the shared backbone
    without resizing.

  SharedModalityViT
    A plain 2D ViT-S that consumes the concatenated context tokens. The
    modality identity of each context token is preserved via a per-
    modality additive bias added to each token's positional embedding.

  ModalityPredictor
    Cross-attention predictor: a learnable target query token (carrying
    the target_modality_id + pose + intensity-hist conditioning, the
    gradient sink) attends over the encoded context tokens to produce
    the predicted target embedding.

  Mod2ModModel
    Holds the encoder/predictor + a list of 4 FROZEN modality-specific
    teachers (passed at construction time). Trainable params = encoder
    + predictor + conditioning + patch embedders.

Training the per-modality teachers is a separate one-time pretraining
job (vanilla I-JEPA on healthy slices of each modality). Until those
exist this module is importable but `training_step` will raise.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.research.jepa import TransformerBlock, sinusoidal_2d_posemb
from .conditioning import (MODALITIES, MODALITY_TO_IDX, Method2Conditioning,
                             ModalityEmbedding)


class PerModalityPatchEmbedder(nn.Module):
    """Four parallel Conv2d patch embedders (one per modality), each
    producing tokens at a shared embed_dim so the backbone consumes a
    homogeneous sequence regardless of which modalities are in S."""

    def __init__(self, image_size: int = 256, patch_size: int = 16,
                 embed_dim: int = 384):
        super().__init__()
        assert image_size % patch_size == 0
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2
        # One Conv2d per modality (single-channel input each, since each
        # modality is its own scalar image).
        self.adapters = nn.ModuleDict({
            m: nn.Conv2d(1, embed_dim,
                          kernel_size=patch_size, stride=patch_size)
            for m in MODALITIES
        })
        # Per-modality additive bias added to positional embeddings so
        # the backbone can tell which token came from which modality
        # without depending on positional information alone.
        self.modality_bias = nn.Parameter(
            torch.zeros(len(MODALITIES), embed_dim))
        nn.init.trunc_normal_(self.modality_bias, std=0.02)
        # Shared 2D sin-cos positional embedding
        self.register_buffer(
            'pos_embed',
            sinusoidal_2d_posemb(self.grid_size, embed_dim),
            persistent=False,
        )

    def forward(self, modality_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        modality_inputs: dict of {modality_name: (B, 1, H, W) tensor}.
        Only present modalities are in the dict (no key for held-out
        target).

        Returns: (B, N_total, embed_dim) concatenated tokens, where
        N_total = num_patches * |S|. The caller must remember the
        ordering used here when building context_indices for the
        predictor — we sort by MODALITIES order for determinism.
        """
        if not modality_inputs:
            raise ValueError('modality_inputs cannot be empty (S must be non-empty)')
        ordered = [m for m in MODALITIES if m in modality_inputs]
        device = next(iter(modality_inputs.values())).device
        pos = self.pos_embed.to(device)
        chunks = []
        for m in ordered:
            x = modality_inputs[m]
            z = self.adapters[m](x).flatten(2).transpose(1, 2)   # (B, P, D)
            mod_idx = MODALITY_TO_IDX[m]
            z = z + pos + self.modality_bias[mod_idx][None, None, :]
            chunks.append(z)
        return torch.cat(chunks, dim=1)


class SharedModalityViT(nn.Module):
    """Plain ViT trunk for Method 2 — consumes the concatenated context
    tokens from PerModalityPatchEmbedder and produces a sequence of
    refined context embeddings."""

    def __init__(self, embed_dim: int = 384, depth: int = 12,
                 heads: int = 6, mlp_ratio: float = 4.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens)


class ModalityPredictor(nn.Module):
    """Cross-attention head: target query (built from conditioning) ->
    attend over context tokens -> predict the held-out modality's
    teacher embedding.

    Output dim equals the teacher's embedding dim (per-modality teachers
    are all ViT-S with the same embed_dim by construction = 384).
    """

    def __init__(self, context_dim: int = 384, predictor_dim: int = 192,
                 depth: int = 6, heads: int = 6, out_dim: int = 384,
                 conditioning_dim: int = 192, n_query_tokens: int = 16):
        super().__init__()
        self.proj_ctx = nn.Linear(context_dim, predictor_dim)
        self.proj_cond = nn.Linear(conditioning_dim, predictor_dim)
        # Multiple query tokens so the predictor can spatially distribute
        # its prediction (16 queries -> mean-pool at the end for the
        # teacher's global embedding). Heavier but more stable.
        self.queries = nn.Parameter(torch.zeros(1, n_query_tokens, predictor_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(predictor_dim, heads, mlp_ratio=2.0)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(predictor_dim)
        self.proj_out = nn.Linear(predictor_dim, out_dim)
        self.n_query_tokens = n_query_tokens

    def forward(self, context: torch.Tensor,
                conditioning: torch.Tensor) -> torch.Tensor:
        B = context.size(0)
        ctx = self.proj_ctx(context)
        cond = self.proj_cond(conditioning).unsqueeze(1)
        q = self.queries.expand(B, -1, -1) + cond
        seq = torch.cat([q, ctx], dim=1)
        for blk in self.blocks:
            seq = blk(seq)
        out = self.norm(seq[:, :self.n_query_tokens, :]).mean(dim=1)
        return self.proj_out(out)


class Mod2ModModel(nn.Module):
    """Method 2 end-to-end model. Frozen teachers are held off-the-
    parameter-list (same trick as Vol2SliceModel) and looked up by name
    when computing the target.

    `frozen_teachers` is a dict of {modality_name: nn.Module} where each
    teacher exposes `.embed_slice2d(x_2d) -> (B, embed_dim)`.
    """

    def __init__(self, frozen_teachers: Dict[str, nn.Module],
                 image_size: int = 256, patch_size: int = 16,
                 embed_dim: int = 384, depth: int = 12, heads: int = 6,
                 predictor_dim: int = 192, predictor_depth: int = 6,
                 hist_dim: int = 48, teacher_embed_dim: int = 384):
        super().__init__()
        missing = [m for m in MODALITIES if m not in frozen_teachers]
        if missing:
            raise ValueError(
                f'Method 2 requires a frozen teacher for each modality. '
                f'Missing: {missing}. Train them first via vanilla I-JEPA '
                f'on healthy brains in that modality.'
            )
        # Assert all teachers truly frozen + same embed dim
        for m, t in frozen_teachers.items():
            for n, p in t.named_parameters():
                assert not p.requires_grad, (
                    f'Teacher for modality {m!r} has trainable param {n}. '
                    'CrossJEPA mandates frozen teachers.')
        self._teachers = frozen_teachers   # plain attr, NOT a submodule

        self.patch_embed = PerModalityPatchEmbedder(
            image_size=image_size, patch_size=patch_size, embed_dim=embed_dim)
        self.backbone = SharedModalityViT(
            embed_dim=embed_dim, depth=depth, heads=heads)
        self.conditioning = Method2Conditioning(
            embed_dim=predictor_dim, hist_dim=hist_dim)
        self.predictor = ModalityPredictor(
            context_dim=embed_dim, predictor_dim=predictor_dim,
            depth=predictor_depth, heads=heads,
            out_dim=teacher_embed_dim, conditioning_dim=predictor_dim,
        )

    def forward_predict(self, context_inputs: Dict[str, torch.Tensor],
                          target_modality_idx: torch.Tensor,
                          slice_pose: torch.Tensor,
                          context_hist: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(context_inputs)
        refined = self.backbone(tokens)
        cond = self.conditioning(target_modality_idx, slice_pose, context_hist)
        return self.predictor(refined, cond)

    def training_step(self, batch: dict) -> dict:
        """One CrossJEPA Method 2 step.

        batch:
          context_inputs       : dict of {modality_name: (B, 1, H, W)}
          target_modality_name : list of str (per-sample target — for now
                                  all samples must share the same target
                                  modality within a batch for the teacher
                                  forward to be a single call)
          target_input         : (B, 1, H, W) the held-out modality image
          target_modality_idx  : (B,) long
          slice_pose           : (B, 2) float
          context_hist         : (B, 48*4) float (per-channel hist, zeros
                                  for absent channels)
        """
        target_modality_names = batch['target_modality_name']
        if isinstance(target_modality_names, (list, tuple)):
            tn0 = target_modality_names[0]
            if not all(n == tn0 for n in target_modality_names):
                raise ValueError(
                    'All samples in a batch must share the same target '
                    'modality (teacher is called once per batch).')
            tn = tn0
        else:
            tn = target_modality_names
        teacher = self._teachers[tn]
        # The dataset/collate use `intensity_hist` for the concatenated
        # per-channel histogram. Accept either key for compat.
        hist = batch.get('intensity_hist', batch.get('context_hist'))
        if hist is None:
            raise KeyError("batch must contain 'intensity_hist' (per-channel histogram)")
        pred = self.forward_predict(batch['context_inputs'],
                                      batch['target_modality_idx'],
                                      batch['slice_pose'],
                                      hist)
        with torch.no_grad():
            target = teacher.embed_slice2d(batch['target_input'])
        loss = F.smooth_l1_loss(pred, target)
        with torch.no_grad():
            cos = F.cosine_similarity(pred, target, dim=-1).mean()
        return {'loss': loss, 'cos_sim': cos.detach()}


__all__ = [
    'PerModalityPatchEmbedder', 'SharedModalityViT', 'ModalityPredictor',
    'Mod2ModModel',
]
