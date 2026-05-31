"""Unit tests for v9b modules: JEPA, DDPM decoder, SDF tower, two-tower
combiner, JEPA conformal, mesh extraction, MNI152 registration.

Run with: python tests/test_v9b_components.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.jepa import (  # noqa: E402
    IJEPAModel, ViTEncoder, JEPAPredictor, make_jepa_masks,
)
from src.research.latent_diffusion_decoder import LatentConditionedDDPM  # noqa: E402
from src.research.sdf_geometric_tower import GeometricSDFTower  # noqa: E402
from src.research.two_tower_anomaly import combine_two_towers, normalize_per_tower  # noqa: E402
from src.research.jepa_conformal import JepaConformalCalibrator, weighted_quantile  # noqa: E402
from src.research.mesh_extraction import extract_tumor_mesh, stack_2d_to_pseudo_3d  # noqa: E402
from src.research.mni152_registration import (  # noqa: E402
    voxel_to_mni_approx, distances_to_landmarks, tumor_atlas_report,
)
from src.research.geometric_prior import synthetic_brain_sdf_template  # noqa: E402


# ----------------- JEPA -----------------

def test_vit_encoder_shape():
    enc = ViTEncoder(image_size=64, patch_size=8, embed_dim=64, depth=2, heads=4)
    x = torch.randn(2, 3, 64, 64)
    z = enc(x)
    assert z.shape == (2, 64, 64), f"got {z.shape}"  # (B, N=8*8, D=64)


def test_vit_encoder_keep_indices():
    enc = ViTEncoder(image_size=64, patch_size=8, embed_dim=64, depth=2, heads=4)
    x = torch.randn(2, 3, 64, 64)
    keep = torch.arange(20).unsqueeze(0).expand(2, -1)
    z = enc(x, keep_indices=keep)
    assert z.shape == (2, 20, 64)


def test_jepa_predictor_shape():
    pred = JEPAPredictor(embed_dim=64, predictor_dim=32, depth=2, heads=4, grid_size=8)
    ctx = torch.randn(2, 20, 64)
    ci = torch.arange(20).unsqueeze(0).expand(2, -1)
    ti = torch.arange(20, 30).unsqueeze(0).expand(2, -1)
    out = pred(ctx, ci, ti)
    assert out.shape == (2, 10, 64)


def test_make_jepa_masks():
    masks = make_jepa_masks(grid_size=16, batch_size=4, n_target=2)
    assert masks["context_indices"].shape[0] == 4
    assert masks["target_indices"].shape[0] == 4
    assert masks["context_indices"].max() < 16 * 16
    assert masks["target_indices"].max() < 16 * 16


def test_ijepa_training_forward():
    model = IJEPAModel(image_size=64, patch_size=8, embed_dim=64, depth=2,
                       heads=4, predictor_dim=32, predictor_depth=2)
    x = torch.randn(2, 3, 64, 64)
    masks = make_jepa_masks(model.grid_size, 2)
    out = model(x, masks)
    assert "loss" in out and out["loss"].requires_grad
    out["loss"].backward()


def test_ijepa_ema_update():
    model = IJEPAModel(image_size=64, patch_size=8, embed_dim=32, depth=2,
                       heads=4, predictor_dim=16, predictor_depth=2)
    # Snapshot target weights
    p0 = next(model.target_encoder.parameters()).clone()
    # Modify context_encoder
    for p in model.context_encoder.parameters():
        p.data.add_(torch.ones_like(p) * 0.1)
    model.ema_update()
    p1 = next(model.target_encoder.parameters())
    # Target should have moved toward context (since momentum < 1)
    assert not torch.allclose(p0, p1), "EMA didn't update target weights"


def test_ijepa_prediction_error_map():
    model = IJEPAModel(image_size=32, patch_size=8, embed_dim=32, depth=2,
                       heads=4, predictor_dim=16, predictor_depth=2)
    x = torch.randn(1, 3, 32, 32)
    emap = model.prediction_error_map(x)
    assert emap.shape == (1, 1, 32, 32)
    assert (emap >= 0).all()


# ----------------- DDPM -----------------

def test_ddpm_training_loss():
    ddpm = LatentConditionedDDPM(in_chans=3, base_ch=8, cond_dim=16,
                                   num_train_timesteps=100)
    x = torch.randn(2, 3, 32, 32)
    cond = torch.randn(2, 16)
    loss = ddpm.training_loss(x, cond)
    assert loss.requires_grad
    loss.backward()


def test_ddpm_ddim_sampling():
    ddpm = LatentConditionedDDPM(in_chans=3, base_ch=8, cond_dim=16,
                                   num_train_timesteps=100)
    cond = torch.randn(1, 16)
    out = ddpm.ddim_sample((1, 3, 32, 32), cond, num_steps=5, device="cpu")
    assert out.shape == (1, 3, 32, 32)


# ----------------- SDF tower -----------------

def test_sdf_tower_forward():
    tower = GeometricSDFTower(image_size=64, base_ch=8)
    x = torch.randn(2, 3, 64, 64)
    sdf = tower(x)
    assert sdf.shape == (2, 1, 64, 64)


def test_sdf_tower_training_loss():
    tower = GeometricSDFTower(image_size=64, base_ch=8)
    x = torch.randn(2, 3, 64, 64)
    tpl = synthetic_brain_sdf_template(64).unsqueeze(0).unsqueeze(0).expand(2, -1, -1, -1)
    loss = tower.training_loss(x, tpl)
    assert loss.requires_grad


# ----------------- Two-tower combiner -----------------

def test_combine_two_towers_weighted_sum():
    a = torch.rand(2, 1, 32, 32)
    g = torch.rand(2, 1, 32, 32)
    c = combine_two_towers(a, g, mode="weighted_sum", lambda_app=0.5, lambda_geo=0.5)
    assert c.shape == (2, 1, 32, 32)
    assert (c >= 0).all() and (c <= 1).all()


def test_combine_two_towers_and():
    a = torch.rand(2, 1, 32, 32)
    g = torch.rand(2, 1, 32, 32)
    c = combine_two_towers(a, g, mode="and", q_app=0.5, q_geo=0.5)
    assert c.shape == (2, 1, 32, 32)
    assert set(c.unique().tolist()).issubset({0.0, 1.0})


# ----------------- JEPA conformal -----------------

def test_weighted_quantile_uniform():
    v = np.linspace(0, 1, 100)
    w = np.ones(100)
    q = weighted_quantile(v, w, 0.9)
    assert 0.85 < q < 0.95


def test_jepa_conformal_calibrator():
    np.random.seed(0)
    scores = np.random.exponential(1.0, 200).tolist()
    cal = JepaConformalCalibrator(alpha=0.10)
    report = cal.calibrate(scores, verbose=False)
    assert report.empirical_coverage >= 0.85
    assert cal.q > 0


def test_jepa_conformal_predict_voxelwise():
    cal = JepaConformalCalibrator(alpha=0.10)
    cal.calibrate(np.random.exponential(1.0, 100).tolist())
    errors = torch.rand(2, 1, 16, 16)
    mask = cal.predict_voxelwise_certified(errors)
    assert mask.shape == (2, 1, 16, 16)
    assert mask.dtype == torch.bool


# ----------------- Mesh extraction -----------------

def test_extract_tumor_mesh_simple():
    try:
        import skimage  # noqa
    except ImportError:
        print("scikit-image not installed, skipping mesh test")
        return
    vol = np.zeros((10, 32, 32), dtype=np.float32)
    vol[3:7, 12:20, 12:20] = 1.0
    mesh = extract_tumor_mesh(vol)
    assert mesh["n_verts"] > 0
    assert mesh["n_faces"] > 0
    assert mesh["volume_mm3"] == 4 * 8 * 8  # cube voxels


def test_stack_2d_to_pseudo_3d():
    m = np.ones((32, 32))
    vol = stack_2d_to_pseudo_3d(m)
    assert vol.shape == (1, 32, 32)


# ----------------- MNI152 registration -----------------

def test_voxel_to_mni_center():
    mni = voxel_to_mni_approx((0, 128, 128), (1, 256, 256), input_orientation="axial_2d")
    assert -1 < mni[0] < 1  # center x
    assert -20 < mni[1] < 20  # center y near y=0
    assert mni[2] == 0.0  # 2D -> z=0


def test_distances_to_landmarks():
    d = distances_to_landmarks((0, 0, 0))
    for v in d.values():
        assert v >= 0


def test_tumor_atlas_report_full():
    vol = np.zeros((1, 256, 256), dtype=np.float32)
    vol[0, 100:150, 100:150] = 1.0
    rep = tumor_atlas_report(vol)
    assert rep["volume_mm3"] == 50 * 50
    assert rep["centroid_mni"] is not None
    assert len(rep["nearest_landmarks"]) == 3


if __name__ == "__main__":
    print("=== JEPA ===")
    test_vit_encoder_shape(); test_vit_encoder_keep_indices()
    test_jepa_predictor_shape(); test_make_jepa_masks()
    test_ijepa_training_forward(); test_ijepa_ema_update()
    test_ijepa_prediction_error_map()
    print("=== DDPM decoder ===")
    test_ddpm_training_loss(); test_ddpm_ddim_sampling()
    print("=== SDF tower ===")
    test_sdf_tower_forward(); test_sdf_tower_training_loss()
    print("=== Two-tower combiner ===")
    test_combine_two_towers_weighted_sum(); test_combine_two_towers_and()
    print("=== JEPA conformal ===")
    test_weighted_quantile_uniform(); test_jepa_conformal_calibrator()
    test_jepa_conformal_predict_voxelwise()
    print("=== Mesh extraction ===")
    test_extract_tumor_mesh_simple(); test_stack_2d_to_pseudo_3d()
    print("=== MNI152 registration ===")
    test_voxel_to_mni_center(); test_distances_to_landmarks()
    test_tumor_atlas_report_full()
    print("\nAll v9b component tests passed.")
