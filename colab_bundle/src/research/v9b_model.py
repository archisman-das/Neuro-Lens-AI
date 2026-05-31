"""v9b integrated model: JEPA + SDF tower + DDPM + two-tower combine + conformal.

Composes all v9b research heads into a single inference pipeline. Loads:
  - Pretrained JEPA (Stage 1)
  - Trained DDPM + SDF tower (Stage 2)
  - Calibrated conformal threshold (Stage 3)
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn.functional as F

from .jepa import IJEPAModel
from .latent_diffusion_decoder import LatentConditionedDDPM
from .sdf_geometric_tower import GeometricSDFTower
from .two_tower_anomaly import combine_two_towers
from .jepa_conformal import JepaConformalCalibrator
from .geometric_prior import synthetic_brain_sdf_template


class V9BModel:
    """End-to-end v9b inference pipeline (not an nn.Module -- a composition).

    Stages set at construction (each can be None for partial pipelines):
      jepa:       IJEPAModel pretrained on healthy MRI
      ddpm:       LatentConditionedDDPM trained to generate healthy counterfactual
      sdf_tower:  GeometricSDFTower trained on healthy atlas SDF
      conformal:  JepaConformalCalibrator (calibrated)
    """
    def __init__(self,
                 jepa: Optional[IJEPAModel] = None,
                 ddpm: Optional[LatentConditionedDDPM] = None,
                 sdf_tower: Optional[GeometricSDFTower] = None,
                 conformal: Optional[JepaConformalCalibrator] = None,
                 image_size: int = 256,
                 device: str = "cuda"):
        self.jepa = jepa
        self.ddpm = ddpm
        self.sdf_tower = sdf_tower
        self.conformal = conformal
        self.image_size = image_size
        self.device = device
        if jepa is not None:
            jepa.eval()
            for p in jepa.parameters(): p.requires_grad = False
        if ddpm is not None:
            ddpm.eval()
            for p in ddpm.parameters(): p.requires_grad = False
        if sdf_tower is not None:
            sdf_tower.eval()
            for p in sdf_tower.parameters(): p.requires_grad = False

    @classmethod
    def from_checkpoints(cls, jepa_ckpt: str, stage2_ckpt: str,
                          conformal_json: Optional[str] = None,
                          image_size: int = 256, device: str = "cuda") -> "V9BModel":
        dev = torch.device(device)
        # JEPA
        ck = torch.load(jepa_ckpt, map_location=dev, weights_only=False)
        ja = ck.get("args", {})
        jepa = IJEPAModel(
            image_size=ja.get("image_size", image_size),
            patch_size=ja.get("patch_size", 16),
            embed_dim=ja.get("embed_dim", 384),
            depth=ja.get("depth", 12),
            heads=ja.get("heads", 6),
            predictor_dim=ja.get("predictor_dim", 192),
            predictor_depth=ja.get("predictor_depth", 6),
        ).to(dev)
        jepa.load_state_dict(ck["model_state_dict"])
        # Stage 2
        ck2 = torch.load(stage2_ckpt, map_location=dev, weights_only=False)
        ddpm = LatentConditionedDDPM(in_chans=3, base_ch=32, cond_dim=jepa.embed_dim).to(dev)
        ddpm.load_state_dict(ck2["ddpm_state_dict"])
        sdf_tower = GeometricSDFTower(image_size=image_size, base_ch=32).to(dev)
        sdf_tower.load_state_dict(ck2["sdf_state_dict"])
        # Conformal (optional)
        conformal = None
        if conformal_json and Path(conformal_json).exists():
            conformal = JepaConformalCalibrator(alpha=0.10)
            conformal.load(conformal_json)
        return cls(jepa=jepa, ddpm=ddpm, sdf_tower=sdf_tower, conformal=conformal,
                   image_size=image_size, device=str(dev))

    @torch.no_grad()
    def infer(self, x: torch.Tensor,
              combine_mode: str = "weighted_sum",
              lambda_app: float = 0.6, lambda_geo: float = 0.4,
              ddpm_num_steps: int = 50) -> Dict:
        """Run the full v9b pipeline on a batch.

        x: (B, 3, H, W) input MRI batch
        Returns dict with all intermediate maps + final certified mask.
        """
        assert self.jepa is not None, "jepa required"
        x = x.to(self.device)
        B = x.size(0)
        # 1. Appearance tower: per-patch JEPA prediction error -> upsampled map
        app_map = self.jepa.prediction_error_map(x)  # (B, 1, H, W)
        # 2. Geometry tower: SDF deviation from atlas
        geo_map = None
        if self.sdf_tower is not None:
            sdf_tpl = synthetic_brain_sdf_template(self.image_size).to(self.device)
            geo_map = self.sdf_tower.anomaly_map(
                x, sdf_tpl.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
            )
        # 3. Combined anomaly
        combined = None
        if geo_map is not None:
            combined = combine_two_towers(app_map, geo_map, mode=combine_mode,
                                           lambda_app=lambda_app, lambda_geo=lambda_geo)
        else:
            combined = app_map
        # 4. Conformal-certified binary mask
        certified_mask = None
        if self.conformal is not None and self.conformal.q is not None:
            certified_mask = self.conformal.predict_voxelwise_certified(combined)
        # 5. Healthy counterfactual via DDPM
        counterfactual = None
        residual = None
        if self.ddpm is not None:
            cond = self.jepa.encode_full(x).mean(dim=1)  # (B, D)
            counterfactual = self.ddpm.ddim_sample(
                shape=x.shape, cond=cond, num_steps=ddpm_num_steps, device=self.device,
            )
            residual = (x - counterfactual).abs().mean(dim=1, keepdim=True)
        return {
            "appearance_anomaly": app_map,
            "geometry_anomaly": geo_map,
            "combined_anomaly": combined,
            "certified_mask": certified_mask,
            "counterfactual": counterfactual,
            "residual": residual,
        }


__all__ = ["V9BModel"]
