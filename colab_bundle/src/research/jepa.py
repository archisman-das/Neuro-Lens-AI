"""I-JEPA implementation for v9b normative brain MRI pretraining.

Encoder + predictor + EMA target encoder + masking strategy + I-JEPA model.

Adapted from Assran et al. 2023 (I-JEPA) for 2D brain MRI (dataset_v8 scale).
3D extension is straightforward (swap PatchEmbed 2D->3D); kept 2D for Colab
T4/A100 feasibility and to reuse dataset_v8 directly.

Key novel pieces vs vanilla I-JEPA:
  - First normative pretrain on healthy brain MRI (US-JEPA is ultrasound)
  - Latent prediction error map preserved + exported per-voxel for v9b conformal
"""
from __future__ import annotations
import math
from typing import Tuple, List, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------- Patch embedding + positional encoding -----------------

class PatchEmbed2D(nn.Module):
    def __init__(self, image_size: int = 256, patch_size: int = 16,
                 in_chans: int = 3, embed_dim: int = 384):
        super().__init__()
        assert image_size % patch_size == 0
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, N, D)
        return self.proj(x).flatten(2).transpose(1, 2)


def sinusoidal_2d_posemb(grid_size: int, embed_dim: int) -> torch.Tensor:
    """2D sin-cos positional embedding for ViT (MAE-style)."""
    def _1d(emb_dim, pos):
        omega = torch.arange(emb_dim // 2, dtype=torch.float32) / (emb_dim / 2.0)
        omega = 1.0 / (10000.0 ** omega)
        out = pos[..., None] * omega[None]
        return torch.cat([torch.sin(out), torch.cos(out)], dim=-1)

    pos = torch.arange(grid_size, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(pos, pos, indexing="ij")
    eh = _1d(embed_dim // 2, grid_h.flatten())
    ew = _1d(embed_dim // 2, grid_w.flatten())
    return torch.cat([eh, ew], dim=-1).unsqueeze(0)  # (1, N, D)


# ----------------- ViT blocks -----------------

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 6, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """Plain ViT trunk used as I-JEPA context AND target encoder backbone."""

    def __init__(self, image_size: int = 256, patch_size: int = 16,
                 in_chans: int = 3, embed_dim: int = 384,
                 depth: int = 12, heads: int = 6, mlp_ratio: float = 4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed2D(image_size, patch_size, in_chans, embed_dim)
        self.register_buffer(
            "pos_embed",
            sinusoidal_2d_posemb(self.patch_embed.grid_size, embed_dim),
            persistent=False,
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.grid_size = self.patch_embed.grid_size

    def forward(self, x: torch.Tensor, keep_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """If keep_indices is given (B, K), only those patch tokens are processed
        (saves memory + makes the encoder a "context encoder" that sees only
        unmasked patches, as in I-JEPA)."""
        z = self.patch_embed(x) + self.pos_embed.to(x.device)
        if keep_indices is not None:
            B = z.size(0)
            z = z.gather(1, keep_indices.unsqueeze(-1).expand(-1, -1, z.size(-1)))
        for blk in self.blocks:
            z = blk(z)
        return self.norm(z)


class JEPAPredictor(nn.Module):
    """Small transformer that predicts target-block latents from context
    latents + learned mask tokens at the target positions."""

    def __init__(self, embed_dim: int = 384, predictor_dim: int = 192,
                 depth: int = 6, heads: int = 6, grid_size: int = 16):
        super().__init__()
        self.proj_in = nn.Linear(embed_dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.register_buffer(
            "pos_embed",
            sinusoidal_2d_posemb(grid_size, predictor_dim),
            persistent=False,
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(predictor_dim, heads, mlp_ratio=2.0) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(predictor_dim)
        self.proj_out = nn.Linear(predictor_dim, embed_dim)
        self.grid_size = grid_size

    def forward(self, context_tokens: torch.Tensor,
                context_indices: torch.Tensor,
                target_indices: torch.Tensor) -> torch.Tensor:
        """
        context_tokens: (B, Kc, D) encoder output for context patches
        context_indices: (B, Kc) patch indices (in 0..N-1) for context
        target_indices: (B, Kt) patch indices for target patches to predict
        Returns: (B, Kt, D) predicted target patch latents
        """
        B = context_tokens.size(0)
        # Expand pos_embed (1, N, Dp) to (B, N, Dp) for per-batch gather.
        pos = self.pos_embed.to(context_tokens.device).expand(B, -1, -1)
        # Project context + add their positional embeddings
        ctx = self.proj_in(context_tokens)
        ctx = ctx + pos.gather(1, context_indices.unsqueeze(-1).expand(-1, -1, ctx.size(-1)))
        # Build target mask tokens at the target positions
        Kt = target_indices.size(1)
        tgt = self.mask_token.expand(B, Kt, -1).clone()
        tgt = tgt + pos.gather(1, target_indices.unsqueeze(-1).expand(-1, -1, tgt.size(-1)))
        # Concatenate ctx + tgt and run through predictor blocks
        seq = torch.cat([ctx, tgt], dim=1)
        for blk in self.blocks:
            seq = blk(seq)
        seq = self.norm(seq)
        # Slice out the predicted target portion + project back to encoder dim
        return self.proj_out(seq[:, -Kt:, :])


# ----------------- Masking strategy -----------------

def make_jepa_masks(grid_size: int, batch_size: int, n_context: int = 1,
                    n_target: int = 4, target_scale: Tuple[float, float] = (0.15, 0.20),
                    context_scale: Tuple[float, float] = (0.85, 1.0),
                    device: str = "cpu") -> Dict[str, torch.Tensor]:
    """I-JEPA masking: sample several "target" rectangles + one "context"
    region that excludes them. Returns flat patch indices.

    For batch_size scans we sample SHARED masks to keep tensors regular.
    """
    N = grid_size * grid_size
    # Sample target blocks (multiple rectangles)
    target_idx_sets = []
    used = torch.zeros(N, dtype=torch.bool)
    for _ in range(n_target):
        scale = torch.empty(1).uniform_(*target_scale).item()
        area = max(1, int(scale * N))
        side = max(1, int(math.sqrt(area)))
        side = min(side, grid_size)
        top = torch.randint(0, grid_size - side + 1, (1,)).item()
        left = torch.randint(0, grid_size - side + 1, (1,)).item()
        rows = torch.arange(top, top + side)
        cols = torch.arange(left, left + side)
        rs, cs = torch.meshgrid(rows, cols, indexing="ij")
        idx = (rs * grid_size + cs).flatten()
        target_idx_sets.append(idx)
        used[idx] = True
    target_flat = torch.cat(target_idx_sets).unique()
    # Sample context region (large area excluding target patches)
    cscale = torch.empty(1).uniform_(*context_scale).item()
    n_ctx = max(1, int(cscale * N))
    candidates = torch.arange(N)[~used]
    if candidates.numel() == 0:
        candidates = torch.arange(N)
    perm = torch.randperm(candidates.numel())
    n_take = min(n_ctx, candidates.numel())
    context_flat = candidates[perm[:n_take]]
    # Expand to batch
    Kc = context_flat.numel()
    Kt = target_flat.numel()
    return {
        "context_indices": context_flat.unsqueeze(0).expand(batch_size, Kc).to(device),
        "target_indices": target_flat.unsqueeze(0).expand(batch_size, Kt).to(device),
    }


# ----------------- I-JEPA model wrapper -----------------

class IJEPAModel(nn.Module):
    """Full I-JEPA: context encoder + predictor + EMA target encoder.

    Pretrain: target encoder is updated by EMA of context encoder, NOT by
    gradients. Loss is L2 between predicted target tokens and EMA target
    tokens, computed only at target patch positions.

    At inference (after pretraining), use target encoder's full forward as
    the "appearance latent" and the predictor's residual as the anomaly
    score per patch (v9b's appearance tower).
    """

    def __init__(self, image_size: int = 256, patch_size: int = 16,
                 in_chans: int = 3, embed_dim: int = 384, depth: int = 12,
                 heads: int = 6, predictor_dim: int = 192,
                 predictor_depth: int = 6, ema_momentum: float = 0.996):
        super().__init__()
        self.context_encoder = ViTEncoder(image_size, patch_size, in_chans,
                                          embed_dim, depth, heads)
        self.predictor = JEPAPredictor(embed_dim=embed_dim,
                                       predictor_dim=predictor_dim,
                                       depth=predictor_depth,
                                       heads=heads,
                                       grid_size=self.context_encoder.grid_size)
        # Target encoder: same arch, updated by EMA from context_encoder.
        # Initialize identical to context encoder, then make non-trainable.
        self.target_encoder = ViTEncoder(image_size, patch_size, in_chans,
                                         embed_dim, depth, heads)
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.ema_momentum = ema_momentum
        self.grid_size = self.context_encoder.grid_size
        self.embed_dim = embed_dim

    @torch.no_grad()
    def ema_update(self) -> None:
        m = self.ema_momentum
        for p_tgt, p_ctx in zip(self.target_encoder.parameters(),
                                 self.context_encoder.parameters()):
            p_tgt.data.mul_(m).add_(p_ctx.data, alpha=1.0 - m)

    def forward(self, x: torch.Tensor, masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ci = masks["context_indices"]
        ti = masks["target_indices"]
        ctx_tokens = self.context_encoder(x, keep_indices=ci)  # (B, Kc, D)
        pred = self.predictor(ctx_tokens, ci, ti)  # (B, Kt, D)
        with torch.no_grad():
            full = self.target_encoder(x)  # (B, N, D)
            tgt = full.gather(1, ti.unsqueeze(-1).expand(-1, -1, full.size(-1)))
            tgt = F.layer_norm(tgt, [tgt.size(-1)])  # standard I-JEPA normalisation
        loss = F.smooth_l1_loss(pred, tgt)
        return {"loss": loss, "pred": pred, "target": tgt}

    @torch.no_grad()
    def encode_full(self, x: torch.Tensor) -> torch.Tensor:
        """Encode full image with the TARGET encoder (EMA, more stable than context)."""
        return self.target_encoder(x)  # (B, N, D)

    @torch.no_grad()
    def prediction_error_map(self, x: torch.Tensor) -> torch.Tensor:
        """At inference: encode full image, then for each patch predict it
        from the rest, measure per-patch prediction error. Returns (B, 1, H, W)
        upsampled to image_size.

        For each patch position p we treat p as the lone "target" and the
        rest as context; the predictor's residual is the patch-wise
        appearance anomaly score.
        """
        B = x.size(0)
        N = self.grid_size * self.grid_size
        device = x.device
        full = self.target_encoder(x)  # (B, N, D)
        full = F.layer_norm(full, [full.size(-1)])
        errors = torch.zeros(B, N, device=device)
        all_idx = torch.arange(N, device=device)
        for p in range(N):
            ci = all_idx[all_idx != p].unsqueeze(0).expand(B, -1)
            ti = torch.full((B, 1), p, device=device, dtype=torch.long)
            ctx_tokens = self.context_encoder(x, keep_indices=ci)
            pred = self.predictor(ctx_tokens, ci, ti)  # (B, 1, D)
            tgt = full.gather(1, ti.unsqueeze(-1).expand(-1, -1, full.size(-1)))
            errors[:, p] = (pred - tgt).pow(2).mean(dim=(1, 2))
        gs = self.grid_size
        emap = errors.view(B, 1, gs, gs)
        return F.interpolate(emap, size=(x.size(-2), x.size(-1)),
                              mode="bilinear", align_corners=False)


__all__ = ["ViTEncoder", "JEPAPredictor", "IJEPAModel", "make_jepa_masks"]
