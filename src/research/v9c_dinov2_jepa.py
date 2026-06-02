"""v9c: Normative JEPA on a frozen DINOv2 backbone.

Architecture (after Phase 2 foundation-model eval, June 2026):
  - Frozen DINOv2-base ViT-B/14 (86M params, no training, never updated)
    Provides 256 patch tokens of dim 768 per 224x224 input.
  - Learnable JEPA predictor head (small, ~5-10M params)
    Takes a context subset of patch tokens, predicts the masked target
    tokens in the FROZEN DINOv2 feature space (smooth-L1 loss).
  - At inference, prediction error per patch becomes the per-patch
    anomaly map. Conformal calibration on healthy gives the threshold.

Why this beats training I-JEPA from scratch (our previous v9b):
  - DINOv2 features were learned on 142M+ images via SSL. Strong, robust
    representations vs. our 17k-slice from-scratch ViT.
  - Phase 2 LOSO AUC = 0.84 on Navoneel (within-source) via simple linear
    probe on DINOv2 features alone. The JEPA head's job is to take that
    base feature quality and turn it into a normative anomaly score.

What we KEEP from v9b:
  - Normative training (healthy data only)
  - JEPA masked-patch prediction objective (not pixel reconstruction)
  - Conformal calibration for threshold
  - Symmetry geometry as a complementary signal in the advisory ensemble

What we DROP:
  - Training the encoder from scratch (was the weak point — see v9b
    JEPA AUC 0.564 on expanded OOD bench)
  - SDF tower (broken; symmetry replaces it)
  - DDPM healthy counterfactual (slow + AUC-marginal — can be added back
    as a v9c.b extension if needed)
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# DINOv2 ViT-B/14 native specs
DINOV2_PATCH_SIZE = 14
DINOV2_EMBED_DIM = 768
DINOV2_DEFAULT_INPUT = 224   # -> 16x16=256 patch tokens per image


def _load_dinov2(device: str = 'cuda'):
    """Lazy-load DINOv2-base ViT-B/14 via HuggingFace transformers."""
    from transformers import AutoModel, AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base').to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, proc


class JEPAPredictorOnDINO(nn.Module):
    """Lightweight transformer predictor over DINOv2 patch tokens.

    Inputs:
      ctx_tokens  (B, Kc, D)  DINOv2 features at the unmasked context positions
      ctx_idx     (B, Kc)     positional indices of those tokens in the
                              full token grid (0..N-1 where N=256 by default)
      tgt_idx     (B, Kt)     positional indices of the masked target tokens

    Output:
      pred_tokens (B, Kt, D)  predicted DINOv2 feature values at the
                              target positions.

    The predictor is intentionally small (3-6 layers, half DINOv2's
    dim) — DINOv2 features are already strong; we just need a small
    head to combine them.
    """

    def __init__(self, dino_dim: int = DINOV2_EMBED_DIM,
                 pred_dim: int = 384,
                 depth: int = 6, heads: int = 6,
                 num_patches: int = 256):
        super().__init__()
        self.proj_in = nn.Linear(dino_dim, pred_dim)
        self.proj_out = nn.Linear(pred_dim, dino_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        # Learnable position embedding for both context and target slots
        self.pos_embed = nn.Parameter(
            self._sinusoidal_2d_posemb(num_patches, pred_dim))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=pred_dim, nhead=heads, dim_feedforward=pred_dim * 4,
                dropout=0.0, activation='gelu', batch_first=True,
                norm_first=True,
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(pred_dim)

    @staticmethod
    def _sinusoidal_2d_posemb(num_patches: int, dim: int) -> torch.Tensor:
        """2D sin-cos pos embed for a square patch grid (e.g. 16x16=256)."""
        grid = int(round(math.sqrt(num_patches)))
        assert grid * grid == num_patches, f'expect square grid; got {num_patches}'

        def _1d(d, pos):
            omega = torch.arange(d // 2, dtype=torch.float32) / (d / 2.0)
            omega = 1.0 / (10000.0 ** omega)
            out = pos[..., None] * omega[None]
            return torch.cat([torch.sin(out), torch.cos(out)], dim=-1)

        pos = torch.arange(grid, dtype=torch.float32)
        gh, gw = torch.meshgrid(pos, pos, indexing='ij')
        eh = _1d(dim // 2, gh.flatten())
        ew = _1d(dim // 2, gw.flatten())
        return torch.cat([eh, ew], dim=-1).unsqueeze(0)   # (1, N, D)

    def forward(self, ctx_tokens, ctx_idx, tgt_idx):
        B, Kc, _ = ctx_tokens.shape
        Kt = tgt_idx.size(1)
        # Project context into predictor-dim + add their pos embeddings
        pos = self.pos_embed.to(ctx_tokens.device).expand(B, -1, -1)
        ctx = self.proj_in(ctx_tokens)
        ctx = ctx + pos.gather(
            1, ctx_idx.unsqueeze(-1).expand(-1, -1, ctx.size(-1)))
        # Mask tokens at the target positions + their pos embeddings
        tgt = self.mask_token.expand(B, Kt, -1).clone()
        tgt = tgt + pos.gather(
            1, tgt_idx.unsqueeze(-1).expand(-1, -1, tgt.size(-1)))
        # Concatenate and run through the small transformer
        seq = torch.cat([ctx, tgt], dim=1)
        for blk in self.blocks:
            seq = blk(seq)
        seq = self.norm(seq)
        # Slice out the target portion and project back to DINOv2 dim
        return self.proj_out(seq[:, -Kt:, :])


def make_jepa_masks(num_patches: int, batch_size: int,
                     n_context: int = 1, n_target: int = 4,
                     target_scale=(0.15, 0.20), context_scale=(0.85, 1.0),
                     device: str = 'cpu') -> dict:
    """Same masking strategy as our original I-JEPA implementation —
    sample several rectangular 'target' blocks + one 'context' region
    excluding them. Returns flat patch indices.

    For a 16x16 grid (256 patches) this gives ~50-100 target tokens
    + ~150-220 context tokens per image, shared across the batch.
    """
    grid = int(round(math.sqrt(num_patches)))
    N = grid * grid
    used = torch.zeros(N, dtype=torch.bool)
    tgt_sets = []
    for _ in range(n_target):
        scale = torch.empty(1).uniform_(*target_scale).item()
        area = max(1, int(scale * N))
        side = max(1, int(math.sqrt(area)))
        side = min(side, grid)
        top = torch.randint(0, grid - side + 1, (1,)).item()
        left = torch.randint(0, grid - side + 1, (1,)).item()
        rs = torch.arange(top, top + side)
        cs = torch.arange(left, left + side)
        gr, gc = torch.meshgrid(rs, cs, indexing='ij')
        idx = (gr * grid + gc).flatten()
        tgt_sets.append(idx)
        used[idx] = True
    tgt_flat = torch.cat(tgt_sets).unique()
    cscale = torch.empty(1).uniform_(*context_scale).item()
    n_ctx = max(1, int(cscale * N))
    cands = torch.arange(N)[~used]
    if cands.numel() == 0:
        cands = torch.arange(N)
    perm = torch.randperm(cands.numel())
    ctx_flat = cands[perm[:min(n_ctx, cands.numel())]]
    Kc = ctx_flat.numel()
    Kt = tgt_flat.numel()
    return {
        'context_indices': ctx_flat.unsqueeze(0).expand(batch_size, Kc).to(device),
        'target_indices': tgt_flat.unsqueeze(0).expand(batch_size, Kt).to(device),
    }


class V9CModel(nn.Module):
    """End-to-end v9c: frozen DINOv2 + JEPA predictor.

    Provides:
      - training_step(images, masks) -> loss for normative pretraining
      - prediction_error_map(images) -> per-patch anomaly map at inference

    DINOv2 is held frozen on `freeze=True`. Only the predictor is
    trainable.
    """

    def __init__(self, predictor_depth: int = 6, predictor_dim: int = 384,
                 device: str = 'cuda'):
        super().__init__()
        self._dev = device
        self.dino, self.proc = _load_dinov2(device)
        self.dino.eval()
        # DINOv2-base at 224x224 = 16x16=256 patches
        self.num_patches = 256
        self.predictor = JEPAPredictorOnDINO(
            dino_dim=DINOV2_EMBED_DIM, pred_dim=predictor_dim,
            depth=predictor_depth, num_patches=self.num_patches,
        ).to(device)

    @torch.no_grad()
    def _encode_full(self, images_uint8: list) -> torch.Tensor:
        """images_uint8: list of HxWx3 numpy uint8. Returns (B, N, D)
        patch tokens (excluding CLS token)."""
        pix = self.proc(images=images_uint8, return_tensors='pt').pixel_values.to(self._dev)
        out = self.dino(pixel_values=pix, output_hidden_states=False)
        # DINOv2's output has shape (B, 1+N, D) with CLS at index 0.
        return out.last_hidden_state[:, 1:, :]   # (B, N, D)

    def training_step(self, images_uint8: list) -> torch.Tensor:
        """Compute the JEPA loss for normative pretraining."""
        with torch.no_grad():
            full_tokens = self._encode_full(images_uint8)        # (B, N, D)
        B = full_tokens.size(0)
        device = full_tokens.device
        masks = make_jepa_masks(self.num_patches, B, device=str(device))
        ci, ti = masks['context_indices'], masks['target_indices']
        ctx_tokens = full_tokens.gather(
            1, ci.unsqueeze(-1).expand(-1, -1, full_tokens.size(-1)))
        pred = self.predictor(ctx_tokens, ci, ti)
        tgt = full_tokens.gather(
            1, ti.unsqueeze(-1).expand(-1, -1, full_tokens.size(-1)))
        # LayerNorm both before loss (matches I-JEPA recipe)
        pred = F.layer_norm(pred, [pred.size(-1)])
        tgt = F.layer_norm(tgt, [tgt.size(-1)])
        return F.smooth_l1_loss(pred, tgt)

    @torch.no_grad()
    def prediction_error_map(self, images_uint8: list) -> torch.Tensor:
        """At inference: for each patch position p, mask it as the lone
        target and predict it from all other patches. Return per-patch
        residual upscaled to a (B, 1, H, W) anomaly map.

        N=256 patches -> 256 forward passes per image. Same compute
        profile as our v9b prediction_error_map; the heavy lifting is
        the DINOv2 forward (also done 256 times since context changes
        slightly each iteration — but we cache the FULL encode once).
        """
        full = self._encode_full(images_uint8)               # (B, N, D)
        full = F.layer_norm(full, [full.size(-1)])
        B, N, D = full.shape
        device = full.device
        errors = torch.zeros(B, N, device=device)
        all_idx = torch.arange(N, device=device)
        for p in range(N):
            ci = all_idx[all_idx != p].unsqueeze(0).expand(B, -1)
            ti = torch.full((B, 1), p, device=device, dtype=torch.long)
            ctx_tokens = full.gather(
                1, ci.unsqueeze(-1).expand(-1, -1, D))
            pred = self.predictor(ctx_tokens, ci, ti)        # (B, 1, D)
            pred = F.layer_norm(pred, [pred.size(-1)])
            tgt = full.gather(1, ti.unsqueeze(-1).expand(-1, -1, D))
            errors[:, p] = (pred - tgt).pow(2).mean(dim=(1, 2))
        # Reshape (B, N) -> (B, 1, 16, 16) -> upsample to (B, 1, 224, 224)
        gs = int(round(math.sqrt(N)))
        emap = errors.view(B, 1, gs, gs)
        return F.interpolate(emap, size=(224, 224),
                              mode='bilinear', align_corners=False)


__all__ = ['V9CModel', 'JEPAPredictorOnDINO', 'make_jepa_masks']
