"""v9 integrated model: brings every research head together.

Architecture (brain-2D scope; multi-organ 3D is the v10 extension)
------------------------------------------------------------------

    Input (B, 3, H, W)
        |
        v
    GeometricPriorConditioning  -- prepends SDF channel -> (B, 4, H, W)
        |
        v
    SMP UNet encoder (ConvNeXt-Tiny, in_channels=4)
        |
        |--- bottleneck latent --> Adaptive pool -> (B, latent_dim)
        |                              |
        |                              v
        |                          HyperbolicProjection -- to Poincare ball
        |                              |
        |                              v
        |                          CausalSCMHead -- (z_anatomy, z_tumor, z_scanner) + DAG losses
        |                              |
        |                              +--> CounterfactualHealthyDecoder (do(z_tumor=0))
        |                              |        |
        |                              |        v
        |                              |    x_counterfactual_healthy
        |                              |
        |                              v
        |                          Recomposed latent (anatomy + tumor)
        |                              |
        v                              v
    SMP UNet decoder ---------- (broadcast recomposed latent into bottleneck)
        |
        v
    Tumor mask logits (B, 1, H, W)

Forward returns a dict with all intermediate signals (for training the
multi-task loss) and the final mask.

The hyperbolic conformal calibration step is offline -- after training,
run scripts/calibrate_v9_conformal.py to compute the calibrated quantile
q on dataset_v8/val. At inference, the calibrator decorates the mask
with coverage-bounded per-voxel decisions.

For Colab/v8 hardware, this model adds ~3-4M params on top of the
~32M v7 baseline. Total ~36M, comfortably fits in any A100/T4 VRAM.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .hyperbolic import HyperbolicProjection, logmap0
from .causal_scm import CausalSCMHead
from ..geometric_prior import GeometricPriorConditioning
from ..counterfactual_decoder import CounterfactualHealthyDecoder, tumor_residual


class V10Model(nn.Module):
    """Integrated v10 segmentation model with all research heads.

    Args:
        encoder_name: SMP encoder name (default ConvNeXt-Tiny via timm)
        image_size: input image side length (default 384)
        latent_dim: dimensionality of the hyperbolic latent (default 256)
        anatomy_dim / tumor_dim / scanner_dim: SCM split dims
        use_counterfactual: if True, builds + uses the healthy-counterfactual
            decoder. Set False during inference if you only need masks.
        use_geometric_prior: if False, skips the SDF conditioning (in_channels=3)
        hyperbolic_curvature_init: initial curvature c of the Poincare ball

    Forward returns:
        {
            "mask_logits": (B, 1, H, W) tumor mask logits,
            "z_euclidean": (B, latent_dim) encoder bottleneck features,
            "z_hyperbolic": (B, latent_dim) projected on Poincare ball,
            "z_anatomy": (B, anatomy_dim),
            "z_tumor": (B, tumor_dim),
            "z_scanner": (B, scanner_dim),
            "x_counterfactual": (B, 3, H, W) or None,
            "tumor_residual": (B, 1, H, W) or None,
            "aux_losses": {
                "ortho_at": ..., "ortho_as": ..., "ortho_ts": ...,
                "dag": ..., "dag_forbidden": ...,
            },
        }
    """

    def __init__(
        self,
        encoder_name: str = "tu-convnext_tiny.fb_in22k_ft_in1k",
        image_size: int = 384,
        latent_dim: int = 256,
        anatomy_dim: int = 128,
        tumor_dim: int = 64,
        scanner_dim: int = 32,
        use_counterfactual: bool = True,
        use_geometric_prior: bool = True,
        hyperbolic_curvature_init: float = 1.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.use_counterfactual = use_counterfactual
        self.use_geometric_prior = use_geometric_prior

        # ---- Geometric prior conditioning ---------------------------
        in_channels = 3
        if use_geometric_prior:
            self.geometric_prior = GeometricPriorConditioning(image_size=image_size, mode="concat")
            in_channels = 4  # 3 (RGB) + 1 (SDF channel)
        else:
            self.geometric_prior = None

        # ---- Encoder + decoder (SMP UNet wrapping) ------------------
        import segmentation_models_pytorch as smp
        self._unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=1,
        )
        # We need access to encoder features (bottleneck) for the
        # hyperbolic/SCM branch. SMP exposes the encoder as .encoder; the
        # bottleneck is encoder(features)[-1].
        # The encoder's bottleneck channel count depends on backbone.
        # For ConvNeXt-Tiny, the last feature map is 768 channels.
        encoder_out_ch = self._unet.encoder.out_channels[-1]
        self.encoder_out_ch = encoder_out_ch
        # Project bottleneck (B, 768, h, w) -> (B, latent_dim)
        self.bottleneck_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder_out_ch, latent_dim),
        )
        # Re-project recomposed latent back into the bottleneck feature
        # map so the SMP decoder can consume it.
        self.recompose_to_bottleneck = nn.Linear(
            anatomy_dim + tumor_dim, encoder_out_ch
        )

        # ---- Hyperbolic projection + causal SCM ---------------------
        self.hyperbolic = HyperbolicProjection(c_init=hyperbolic_curvature_init, learnable=True)
        self.scm = CausalSCMHead(
            in_dim=latent_dim,
            anatomy_dim=anatomy_dim,
            tumor_dim=tumor_dim,
            scanner_dim=scanner_dim,
            decoder_in_dim=anatomy_dim + tumor_dim,
        )

        # ---- Counterfactual healthy decoder -------------------------
        if use_counterfactual:
            self.cf_decoder = CounterfactualHealthyDecoder(
                latent_dim=anatomy_dim + tumor_dim,
                image_size=image_size,
                in_channels=3,  # output RGB MRI
            )
        else:
            self.cf_decoder = None

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Pass x through the encoder, return the bottleneck features.

        For ConvNeXt-Tiny via SMP/timm, encoder() returns a list of
        feature maps at each scale; the last is the bottleneck.
        """
        features = self._unet.encoder(x)
        return features  # list of feature maps, last is bottleneck

    def _decode(self, features: list, bottleneck_replacement: torch.Tensor) -> torch.Tensor:
        """Run the SMP decoder, optionally replacing the bottleneck features.

        Used at training: we replace the encoder's bottleneck with a
        recomposed version derived from the causal latent. This wires the
        SCM head into the decoder's gradient path.
        """
        # SMP UNet decoder expects features[0..N-1] then bottleneck.
        # We replace features[-1] with `bottleneck_replacement` (broadcast
        # spatially from the projected recompose).
        old_bottleneck = features[-1]
        b, c, h, w = old_bottleneck.shape
        replaced = bottleneck_replacement.view(b, c, 1, 1).expand(b, c, h, w)
        # Mix with original (residual) so we don't destroy spatial signal
        # the encoder captured; the SCM modulates rather than overwrites.
        mixed = 0.5 * old_bottleneck + 0.5 * replaced
        new_features = features[:-1] + [mixed]
        decoder_out = self._unet.decoder(new_features)
        mask_logits = self._unet.segmentation_head(decoder_out)
        return mask_logits

    def forward(self, x: torch.Tensor, return_counterfactual: bool = True) -> Dict:
        # Stage 1: geometric prior conditioning
        if self.use_geometric_prior:
            x_cond = self.geometric_prior(x)  # (B, 4, H, W)
        else:
            x_cond = x

        # Stage 2: encoder
        features = self._encode(x_cond)
        bottleneck = features[-1]  # (B, encoder_out_ch, h, w)
        z_eucl = self.bottleneck_proj(bottleneck)  # (B, latent_dim)

        # Stage 3: hyperbolic projection
        z_hyp = self.hyperbolic(z_eucl)  # (B, latent_dim) on Poincare ball

        # Stage 4: causal SCM head (operates on the *tangent space* projection
        # of the hyperbolic latent for compatibility with Euclidean SplitHead)
        # Map back to tangent space for the SCM's Euclidean ops.
        z_tangent = logmap0(z_hyp, self.hyperbolic.c)
        recomposed, scm_aux = self.scm(z_tangent, counterfactual_healthy=False)
        # `recomposed` is the SCM's recomposed latent (anatomy + tumor, no scanner)

        # Stage 5: decode segmentation mask
        decoder_bottleneck = self.recompose_to_bottleneck(recomposed)
        mask_logits = self._decode(features, decoder_bottleneck)

        # Stage 6: counterfactual healthy generation (optional)
        x_cf = None
        residual = None
        if self.cf_decoder is not None and return_counterfactual:
            # Re-run SCM with z_tumor zeroed
            cf_recomposed, _ = self.scm(z_tangent, counterfactual_healthy=True)
            x_cf = self.cf_decoder(x, cf_recomposed)
            residual = tumor_residual(x, x_cf)

        return {
            "mask_logits": mask_logits,
            "z_euclidean": z_eucl,
            "z_hyperbolic": z_hyp,
            "z_tangent": z_tangent,
            "z_anatomy": scm_aux["z_anatomy"],
            "z_tumor": scm_aux["z_tumor"],
            "z_scanner": scm_aux["z_scanner"],
            "x_counterfactual": x_cf,
            "tumor_residual": residual,
            "hyperbolic_curvature": self.hyperbolic.c.detach(),
            "aux_losses": {
                "ortho_at": scm_aux["ortho_at"],
                "ortho_as": scm_aux["ortho_as"],
                "ortho_ts": scm_aux["ortho_ts"],
                "dag": scm_aux["dag"],
                "dag_forbidden": scm_aux["dag_forbidden"],
                "adjacency": scm_aux["adjacency"],
            },
        }

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = ["V10Model"]
