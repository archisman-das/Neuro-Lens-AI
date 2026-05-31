"""Unit tests for v10 (parked Universal Causal-Hyperbolic) modules.

Covers: causal SCM, geometric prior (shared with v9b), counterfactual
decoder (shared with v9b), hyperbolic conformal, integrated v10 model.

Run with: python tests/test_v10_components.py
or via pytest: pytest tests/test_v10_components.py

v10 is parked at src/research/_v10_universal_hyperbolic/. Active research
direction is v9b (Normative JEPA + Conformal); see proposals/v9b_*.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research._v10_universal_hyperbolic.causal_scm import (  # noqa: E402
    CausalSCMHead, CausalSplitHead, CausalRecompose,
    LearnableDAGAdjacency, orthogonality_loss,
)
from src.research.geometric_prior import (  # noqa: E402
    GeometricPriorConditioning, synthetic_brain_sdf_template, make_coord_grid,
    SIRENImplicitSDF,
)
from src.research.counterfactual_decoder import (  # noqa: E402
    CounterfactualHealthyDecoder, tumor_residual,
)
from src.research._v10_universal_hyperbolic.hyperbolic_conformal import (  # noqa: E402
    HyperbolicCalibSample, HyperbolicConformalCalibrator,
    voxelwise_hyperbolic_anomaly_map, weighted_quantile_tibshirani,
)
from src.research._v10_universal_hyperbolic.hyperbolic import expmap0  # noqa: E402


# ----------------------------------------------------------------------
# Causal SCM
# ----------------------------------------------------------------------

def test_causal_split_dims():
    """SplitHead produces 3 streams with correct dims."""
    head = CausalSplitHead(in_dim=256, anatomy_dim=128, tumor_dim=64, scanner_dim=32)
    z = torch.randn(8, 256)
    z_a, z_t, z_s = head(z)
    assert z_a.shape == (8, 128)
    assert z_t.shape == (8, 64)
    assert z_s.shape == (8, 32)


def test_orthogonality_loss():
    """Orthogonality loss (cross-correlation Frobenius norm) is small for
    independent vectors, larger for correlated, and works on different dims.

    For identical vectors of dim D: diagonal of normalized cross-cov is 1,
    off-diagonal small, mean of squared ~ D/D^2 = 1/D. For independent
    random vectors of dim D, samples N: all entries small (concentration
    of measure), mean of squared ~ 1/N. So parallel > orthogonal but the
    margin shrinks with D.
    """
    n = 100
    z1 = torch.randn(n, 32)
    # Orthogonal: independent random
    z2_ortho = torch.randn(n, 32)
    loss_ortho = orthogonality_loss(z1, z2_ortho)
    # Parallel: identical
    z2_parallel = z1.clone()
    loss_parallel = orthogonality_loss(z1, z2_parallel)
    assert loss_parallel > loss_ortho, \
        f"parallel loss {loss_parallel.item():.4f} should exceed ortho {loss_ortho.item():.4f}"
    # Cross-correlation magnitude for identical 32-dim vectors: ~1/32 = 0.031.
    assert loss_parallel.item() > 0.02, \
        f"parallel cross-correlation should be at least 1/D ~ 0.03, got {loss_parallel.item():.4f}"
    # Test mixed-dim case (the bug we fixed)
    z3 = torch.randn(n, 16)
    loss_mixed = orthogonality_loss(z1, z3)
    assert torch.isfinite(loss_mixed).all() and loss_mixed.item() >= 0, \
        f"mixed-dim loss should be finite + non-negative, got {loss_mixed.item()}"
    assert loss_mixed.item() < 1.0, \
        f"mixed-dim independent loss should be near 0, got {loss_mixed.item()}"


def test_dag_acyclicity():
    """NOTEARS h(A) is 0 for a DAG, > 0 for a cyclic adjacency."""
    dag = LearnableDAGAdjacency()
    # Set A to be a clean DAG: anatomy -> tumor -> scanner only.
    A = torch.zeros(3, 3)
    A[0, 1] = 0.5  # anatomy -> tumor
    A[1, 2] = 0.5  # tumor -> scanner
    dag.raw.data = A
    h = dag.h().item()
    assert abs(h) < 0.5, f"clean DAG should give small h, got {h}"
    # Now inject a cycle: scanner -> anatomy.
    A_cyclic = A.clone()
    A_cyclic[2, 0] = 0.5
    dag.raw.data = A_cyclic
    h_cyc = dag.h().item()
    assert h_cyc > h, f"cyclic A should give larger h ({h_cyc}) than acyclic ({h})"


def test_scm_head_full_forward():
    """Full SCM head produces recomposed latent + all aux losses."""
    scm = CausalSCMHead(in_dim=256, anatomy_dim=128, tumor_dim=64, scanner_dim=32,
                         decoder_in_dim=256)
    z = torch.randn(8, 256)
    recomposed, aux = scm(z)
    assert recomposed.shape == (8, 256)
    for k in ("ortho_at", "ortho_as", "ortho_ts", "dag", "dag_forbidden",
              "z_anatomy", "z_tumor", "z_scanner"):
        assert k in aux, f"missing aux key: {k}"
    # Counterfactual: z_tumor should be zeroed
    recomposed_cf, _ = scm(z, counterfactual_healthy=True)
    # Different recomposed because z_tumor is zeroed
    assert not torch.allclose(recomposed, recomposed_cf)


# ----------------------------------------------------------------------
# Geometric prior
# ----------------------------------------------------------------------

def test_synthetic_brain_sdf():
    """SDF template has correct sign convention."""
    sdf = synthetic_brain_sdf_template(size=64)
    # Center should be inside (negative SDF)
    center = sdf[32, 32].item()
    # Corner should be outside (positive SDF)
    corner = sdf[0, 0].item()
    assert center < 0, f"center SDF should be negative (inside), got {center}"
    assert corner > 0, f"corner SDF should be positive (outside), got {corner}"


def test_geometric_prior_concat_mode():
    """Concat mode adds 1 channel."""
    prior = GeometricPriorConditioning(image_size=64, mode="concat")
    x = torch.randn(2, 3, 64, 64)
    out = prior(x)
    assert out.shape == (2, 4, 64, 64), f"expected (2, 4, 64, 64), got {out.shape}"


def test_geometric_prior_blend_mode():
    """Blend mode preserves channel count."""
    prior = GeometricPriorConditioning(image_size=64, mode="blend", refine=False)
    x = torch.randn(2, 3, 64, 64)
    out = prior(x)
    assert out.shape == (2, 3, 64, 64)


def test_siren_implicit_sdf():
    """SIREN INR output matches expected dimensions."""
    inr = SIRENImplicitSDF(hidden_dim=64, n_layers=3)
    coords = make_coord_grid(16)
    out = inr(coords)
    assert out.shape == (16, 16, 1)


# ----------------------------------------------------------------------
# Counterfactual decoder
# ----------------------------------------------------------------------

def test_counterfactual_decoder_shape():
    """Decoder produces image-shaped output."""
    dec = CounterfactualHealthyDecoder(latent_dim=192, image_size=64)
    x = torch.randn(2, 3, 64, 64)
    z = torch.randn(2, 192)
    out = dec(x, z)
    assert out.shape == (2, 3, 64, 64)
    assert (out >= -1).all() and (out <= 1).all(), "tanh output should be in [-1, 1]"


def test_counterfactual_reconstruction_loss():
    """Reconstruction loss is 0 when x_healthy == x_input."""
    dec = CounterfactualHealthyDecoder(latent_dim=192, image_size=64)
    x = torch.randn(2, 3, 64, 64)
    mask = torch.zeros(2, 1, 64, 64)  # healthy scan
    loss = dec.reconstruction_loss(x, x, mask)
    assert loss.item() < 1e-5, f"loss should be ~0 when x_healthy == x, got {loss}"


def test_tumor_residual():
    """Residual is high where input differs from counterfactual."""
    x = torch.zeros(1, 3, 64, 64)
    x_cf = torch.zeros(1, 3, 64, 64)
    x[0, :, 20:40, 20:40] = 1.0  # "tumor" in center
    residual = tumor_residual(x, x_cf)
    assert residual[0, 0, 30, 30].item() > 0.5, "residual should be high in tumor region"
    assert residual[0, 0, 5, 5].item() < 0.1, "residual should be low outside tumor"


# ----------------------------------------------------------------------
# Hyperbolic conformal
# ----------------------------------------------------------------------

def test_weighted_quantile_uniform():
    """Uniform weights should match standard quantile."""
    values = torch.linspace(0, 1, 100).numpy()
    weights = torch.ones(100).numpy()
    q = weighted_quantile_tibshirani(values, weights, 0.9)
    # With inflation, should be near 0.91 (slightly above 0.9 by Tibshirani correction)
    assert 0.85 < q < 0.95, f"expected quantile ~0.9, got {q}"


def test_hyperbolic_calibrator():
    """Calibrator empirical coverage matches target (1-alpha)."""
    import numpy as np
    torch.manual_seed(0)
    np.random.seed(0)
    # Synthetic calibration: predict z_pred = z_true + small noise on Poincare ball
    samples = []
    for _ in range(200):
        z_true_eu = torch.randn(16) * 0.3
        z_true = expmap0(z_true_eu, c=1.0)
        # Predicted = true + noise (in tangent space)
        z_pred_eu = z_true_eu + torch.randn(16) * 0.1
        z_pred = expmap0(z_pred_eu, c=1.0)
        samples.append(HyperbolicCalibSample(z_pred=z_pred, z_true=z_true))
    cal = HyperbolicConformalCalibrator(alpha=0.10, curvature_c=1.0)
    report = cal.calibrate(samples)
    assert 0.85 <= report.empirical_coverage_on_calib <= 1.0, \
        f"coverage {report.empirical_coverage_on_calib} should be >= 0.85 (target 0.90)"
    assert cal.q > 0
    assert cal.q < 5  # reasonable for unit Poincare ball


def test_hyperbolic_calibrator_predict():
    """Calibrator predict produces well-formed output dict."""
    torch.manual_seed(1)
    samples = []
    for _ in range(50):
        z_true = expmap0(torch.randn(8) * 0.3, c=1.0)
        z_pred = expmap0(torch.randn(8) * 0.3, c=1.0)
        samples.append(HyperbolicCalibSample(z_pred=z_pred, z_true=z_true))
    cal = HyperbolicConformalCalibrator(alpha=0.10, curvature_c=1.0)
    cal.calibrate(samples)
    z_pred_test = expmap0(torch.randn(8) * 0.3, c=1.0)
    z_true_test = expmap0(torch.randn(8) * 0.3, c=1.0)
    out = cal.predict(z_pred_test, z_true_test)
    for k in ("score", "q", "in_prediction_set", "anomaly_certified"):
        assert k in out
    assert isinstance(out["in_prediction_set"], bool)


def test_voxelwise_anomaly_map():
    """Voxelwise wrapper produces correctly-shaped boolean map."""
    torch.manual_seed(2)
    samples = []
    for _ in range(50):
        z_true = expmap0(torch.randn(4) * 0.3, c=1.0)
        z_pred = expmap0(torch.randn(4) * 0.3, c=1.0)
        samples.append(HyperbolicCalibSample(z_pred=z_pred, z_true=z_true))
    cal = HyperbolicConformalCalibrator(alpha=0.10, curvature_c=1.0)
    cal.calibrate(samples)
    z_pred_vox = expmap0(torch.randn(2, 4, 16, 16) * 0.3, c=1.0)
    z_tpl_vox = expmap0(torch.randn(2, 4, 16, 16) * 0.3, c=1.0)
    amap = voxelwise_hyperbolic_anomaly_map(z_pred_vox, z_tpl_vox, cal)
    assert amap.shape == (2, 1, 16, 16)
    assert amap.dtype == torch.bool


# ----------------------------------------------------------------------
# V10 integrated model
# ----------------------------------------------------------------------

def test_v10_model_forward_smoke():
    """V10Model end-to-end forward produces all expected outputs."""
    try:
        import segmentation_models_pytorch as smp  # noqa
    except ImportError:
        print("SMP not installed, skipping integrated model test")
        return

    from src.research._v10_universal_hyperbolic.v10_model import V10Model
    # Small image size to keep test fast; ConvNeXt-Tiny still loads.
    model = V10Model(
        image_size=128,
        latent_dim=128,
        anatomy_dim=64,
        tumor_dim=32,
        scanner_dim=16,
        use_counterfactual=True,
        use_geometric_prior=True,
    ).eval()
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out = model(x, return_counterfactual=True)
    expected_keys = ["mask_logits", "z_euclidean", "z_hyperbolic", "z_tangent",
                     "z_anatomy", "z_tumor", "z_scanner",
                     "x_counterfactual", "tumor_residual",
                     "hyperbolic_curvature", "aux_losses"]
    for k in expected_keys:
        assert k in out, f"missing key {k}"
    assert out["mask_logits"].shape == (2, 1, 128, 128)
    assert out["x_counterfactual"].shape == (2, 3, 128, 128)
    assert out["tumor_residual"].shape == (2, 1, 128, 128)
    assert out["z_anatomy"].shape == (2, 64)
    assert out["z_tumor"].shape == (2, 32)
    assert out["z_scanner"].shape == (2, 16)
    # Hyperbolic latent must be inside the Poincare ball
    z_h_norm = out["z_hyperbolic"].norm(dim=-1)
    assert (z_h_norm < 1.0).all(), \
        f"hyperbolic latent should be in unit ball, max norm {z_h_norm.max()}"


if __name__ == "__main__":
    print("=== causal SCM ===")
    test_causal_split_dims()
    test_orthogonality_loss()
    test_dag_acyclicity()
    test_scm_head_full_forward()
    print("=== geometric prior ===")
    test_synthetic_brain_sdf()
    test_geometric_prior_concat_mode()
    test_geometric_prior_blend_mode()
    test_siren_implicit_sdf()
    print("=== counterfactual decoder ===")
    test_counterfactual_decoder_shape()
    test_counterfactual_reconstruction_loss()
    test_tumor_residual()
    print("=== hyperbolic conformal ===")
    test_weighted_quantile_uniform()
    test_hyperbolic_calibrator()
    test_hyperbolic_calibrator_predict()
    test_voxelwise_anomaly_map()
    print("=== v10 integrated model ===")
    test_v10_model_forward_smoke()
    print("\nAll v10 component tests passed.")
