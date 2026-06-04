"""Frozen-teacher API + concrete wrappers for v8, DINOv2, and MedSAM.

All teachers expose the same minimal contract so Vol2SliceModel can
treat them interchangeably:

    teacher.embed_batch(x_2d_rgb)  ->  (B, embed_dim) torch.Tensor
    teacher.embed_dim              ->  int

This lets Method 1 swap teacher choices via CLI flag without code
changes. It also makes the multi-teacher variants (Option A = dual
prediction, Option C = teacher-id conditioning) trivial: the model
just iterates a list[BaseFrozenTeacher].

CrossJEPA invariant enforced uniformly:
  - All teachers run `eval()` + `requires_grad_(False)` at construction
  - A `_assert_frozen()` runtime check refuses to proceed if any
    parameter has requires_grad=True
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Re-exported here for backwards-compat (the original v8_teacher module
# defined this constant). Modules that import `V8_EMBED_DIM` from
# v8_teacher.py still work — that file just re-imports from here now.
V8_EMBED_DIM = 768


class BaseFrozenTeacher(nn.Module):
    """Common API for any frozen teacher used by Method 1.

    Concrete subclasses must:
      - set `self.embed_dim` (int) in __init__
      - implement `embed_batch(x: Tensor) -> Tensor` returning (B, embed_dim)
        with shape semantics: x is (B, C, H, W) float in [0, 1].

    The subclass should also call `self._freeze()` once the underlying
    nn modules are set up.
    """

    embed_dim: int

    def _freeze(self) -> None:
        """Set eval() + requires_grad_(False) on every contained module."""
        for m in self.children():
            if isinstance(m, nn.Module):
                m.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        self._assert_frozen()

    def _assert_frozen(self) -> None:
        for n, p in self.named_parameters():
            assert not p.requires_grad, (
                f'{type(self).__name__}: param {n!r} has requires_grad=True. '
                'CrossJEPA mandates a frozen teacher — refusing to proceed.'
            )

    @torch.no_grad()
    def embed_batch(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def embed_slice(self, slice_rgb_uint8: np.ndarray) -> np.ndarray:
        """Convenience wrapper: embed a single (H, W, 3) uint8 slice."""
        arr = slice_rgb_uint8.astype(np.float32) / 255.0
        device = next(self.parameters()).device
        x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
        return self.embed_batch(x).squeeze(0).cpu().numpy()


class V8FrozenTeacher(BaseFrozenTeacher):
    """Frozen v8 ConvNeXt-Tiny encoder (peeled off the segmentation UNet).

    Embedding dim = 768 (ConvNeXt-Tiny's last-stage channels).
    """

    def __init__(self, encoder: nn.Module, image_size: int = 384):
        super().__init__()
        self.encoder = encoder
        self.image_size = image_size
        self.embed_dim = V8_EMBED_DIM
        self._freeze()

    @classmethod
    def from_unet_checkpoint(cls, ckpt_path: str | Path,
                              encoder_name: str = 'tu-convnext_tiny.fb_in22k_ft_in1k',
                              in_channels: int = 3, image_size: int = 384,
                              device: str = 'cuda') -> 'V8FrozenTeacher':
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise RuntimeError(
                'segmentation_models_pytorch required to rebuild v8 UNet. '
                'pip install segmentation-models-pytorch') from exc
        net = smp.Unet(encoder_name=encoder_name, encoder_weights=None,
                        in_channels=in_channels, classes=1)
        sd = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        sd = {k.replace('module.', '').replace('model.', ''): v for k, v in sd.items()}
        miss, _ = net.load_state_dict(sd, strict=False)
        if miss:
            print(f'  [V8FrozenTeacher] decoder layers missing ({len(miss)} keys, expected)')
        net = net.to(device).eval()
        return cls(net.encoder, image_size=image_size).to(device)

    @torch.no_grad()
    def embed_batch(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                               mode='bilinear', align_corners=False)
        feats = self.encoder(x)
        deepest = feats[-1] if isinstance(feats, (list, tuple)) else feats
        pooled = F.adaptive_avg_pool2d(deepest, output_size=1).flatten(1)
        assert pooled.shape[-1] == self.embed_dim
        return pooled


class DINOv2FrozenTeacher(BaseFrozenTeacher):
    """Frozen DINOv2 ViT-B/14 image encoder via HuggingFace transformers.

    Why this teacher (vs v8):
      - Trained on 142M natural images with massive intensity / contrast
        augmentation — intrinsically robust to preprocessing shifts that
        broke v8 on held-out IXI (raw AUC = 0.00 inverted).
      - 86M params (vs v8's 27M) → richer feature space.
      - Self-supervised → no task-collapse bias toward tumor regions.
      - In our prior LOSO bake-off, DINOv2 beat every medical foundation
        model we tried (AUC 0.842 vs RAD-DINO 0.269 vs BiomedCLIP 0.536).

    Embedding dim = 768 (matches v8 — predictor head can stay the same
    width for backwards compatibility with the v1 v8-teacher trainer).
    We pool the patch tokens (mean over the 256 patches at 224x224) to
    get a global slice representation.
    """

    DEFAULT_MODEL = 'facebook/dinov2-base'

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = 'cuda',
                 use_cls_token: bool = False):
        super().__init__()
        try:
            from transformers import AutoModel, AutoImageProcessor
        except ImportError as exc:
            raise RuntimeError(
                'transformers required for DINOv2 teacher. '
                'pip install transformers') from exc
        self.model_id = model_id
        self.use_cls_token = use_cls_token
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.backbone = AutoModel.from_pretrained(model_id).to(device).eval()
        self.image_size = 224
        # DINOv2-base hidden size = 768
        self.embed_dim = int(self.backbone.config.hidden_size)
        self._freeze()

    @torch.no_grad()
    def embed_batch(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) float in [0, 1]. DINOv2 expects 224x224 input
        normalized via its image processor's mean/std — we apply that
        inline (no PIL roundtrip)."""
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                               mode='bilinear', align_corners=False)
        # Apply the processor's mean/std (broadcast across B, C, H, W)
        mean = torch.tensor(self.processor.image_mean,
                             dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
        std = torch.tensor(self.processor.image_std,
                            dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
        x = (x - mean) / std
        out = self.backbone(pixel_values=x)
        # last_hidden_state: (B, 1+256, 768)  [CLS, patches]
        if self.use_cls_token:
            pooled = out.last_hidden_state[:, 0, :]   # CLS
        else:
            pooled = out.last_hidden_state[:, 1:, :].mean(dim=1)  # mean patch
        return pooled


class MedSAMFrozenTeacher(BaseFrozenTeacher):
    """Frozen MedSAM image encoder via HuggingFace transformers.

    MedSAM (Ma et al., Nat Commun 2024) is a SAM ViT-B image encoder +
    prompt encoder + mask decoder, finetuned on 1.5M medical image-mask
    pairs across many modalities. We use ONLY its vision encoder for
    embedding extraction — same trick as the v8 wrapper (peel off the
    encoder, ignore the prompt/decoder side).

    Embedding dim = 256 (the dim AFTER SAM's neck convolution; this is
    the feature representation MedSAM's prompt encoder consumes).
    Pre-neck ViT features are 768-d but they're not the "what MedSAM
    thinks this image is" representation — the neck output is.

    Why this teacher (vs v8):
      - Trained on 1.5M medical image-mask pairs (vs v8's ~6k BraTS volumes)
        → much broader preprocessing distribution seen during training
      - 89M params (vs v8's 27M)
      - Inherits SAM's natural-image pretraining (11M images) under the
        medical finetune → some intensity-robustness baked in

    Honest caveats (compared to DINOv2):
      - MedSAM's image encoder is OPTIMIZED for "given a bbox, produce a
        mask" — task-collapse risk. Features may not be ideal as
        standalone embeddings the way DINOv2's self-supervised features
        are. Empirical question we resolve via held-out IXI AUC.
      - Untested in our prior LOSO bake-off (only DINOv2 was tested
        there at AUC=0.842 vs medical-foundations at 0.27-0.54).

    HF repo: 'flaviagiammarino/medsam-vit-base' (~358 MB).
    """

    DEFAULT_REPO = 'flaviagiammarino/medsam-vit-base'

    def __init__(self, repo_id: str = DEFAULT_REPO, device: str = 'cuda'):
        super().__init__()
        try:
            from transformers import SamModel, SamProcessor   # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                'transformers required for MedSAM teacher. '
                'pip install transformers') from exc
        self.repo_id = repo_id
        self.processor = SamProcessor.from_pretrained(repo_id)
        full = SamModel.from_pretrained(repo_id).to(device).eval()
        # We only need the vision encoder. Drop the prompt encoder + mask
        # decoder to save GPU memory. (Both are still in `full`'s state
        # dict but unreferenced, so they get GC'd after we extract.)
        self.vision_encoder = full.vision_encoder
        # SAM's vision encoder output is (B, 256, 64, 64) after the neck
        self.embed_dim = 256
        self.image_size = 1024
        # SAM uses ImageNet-style normalization (mean/std from the
        # SamProcessor's image_processor)
        self._mean = list(self.processor.image_processor.image_mean)
        self._std = list(self.processor.image_processor.image_std)
        self._freeze()

    @torch.no_grad()
    def embed_batch(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) float in [0, 1]. SAM expects 1024x1024 inputs
        with ImageNet normalization; we apply both inline (no PIL
        roundtrip via SamProcessor — that path is for prompted inference
        and is much slower)."""
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                               mode='bilinear', align_corners=False)
        mean = torch.tensor(self._mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
        std = torch.tensor(self._std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
        x = (x - mean) / std
        # SamModel.vision_encoder returns a SamVisionEncoderOutput with
        # last_hidden_state of shape (B, 256, 64, 64) post-neck.
        out = self.vision_encoder(pixel_values=x)
        feat = out.last_hidden_state                    # (B, 256, 64, 64)
        # Global average pool to a fixed-length vector
        pooled = feat.mean(dim=(2, 3))                  # (B, 256)
        assert pooled.shape[-1] == self.embed_dim, (
            f'MedSAMFrozenTeacher: expected {self.embed_dim}-d embedding, '
            f'got {pooled.shape[-1]}.')
        return pooled


def build_teacher(name: str, v8_ckpt: Optional[str] = None,
                    device: str = 'cuda') -> BaseFrozenTeacher:
    """Factory that builds a teacher by short name. Used by the trainer
    so the `--teachers v8,dinov2,medsam` CLI arg can map cleanly to
    instances."""
    n = name.strip().lower()
    if n == 'v8':
        if v8_ckpt is None:
            raise ValueError("v8 teacher requires v8_ckpt path")
        return V8FrozenTeacher.from_unet_checkpoint(v8_ckpt, device=device)
    if n in ('dinov2', 'dinov2-base'):
        return DINOv2FrozenTeacher(device=device)
    if n == 'dinov2-large':
        return DINOv2FrozenTeacher(model_id='facebook/dinov2-large', device=device)
    if n in ('medsam', 'medsam-vit-base'):
        return MedSAMFrozenTeacher(device=device)
    raise ValueError(
        f'unknown teacher name: {name!r}. '
        f'Known: v8, dinov2, dinov2-large, medsam')


__all__ = [
    'BaseFrozenTeacher',
    'V8FrozenTeacher', 'DINOv2FrozenTeacher', 'MedSAMFrozenTeacher',
    'V8_EMBED_DIM', 'build_teacher',
]
