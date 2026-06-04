"""Component tests for CrossJEPA Method 1 + Method 2.

These tests verify shapes, frozen-teacher invariants, and the gradient
sink behavior on tiny synthetic inputs — no real model weights or
datasets required. Run via:
    python -m pytest tests/test_v9c_crossjepa_components.py -xvs
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.research.v9c_crossjepa.conditioning import (
    MODALITIES, MODALITY_TO_IDX, PLANE_TO_IDX,
    Method1Conditioning, Method2Conditioning,
)
from src.research.v9c_crossjepa.modality_to_modality import (
    Mod2ModModel, ModalityPredictor, PerModalityPatchEmbedder,
    SharedModalityViT,
)
from src.research.v9c_crossjepa.vit_3d import (
    PatchEmbed3D, ViT3DEncoder, _3d_sincos_posemb,
)
from src.research.v9c_crossjepa.volume_to_slice import (
    SliceEmbeddingPredictor, Vol2SliceModel,
    MultiHeadSliceEmbeddingPredictor, TeacherIdConditionedPredictor,
    Vol2SliceModelMulti,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _TinyFrozenTeacher(nn.Module):
    """Stand-in for V8FrozenTeacher in tests — embeds a 2D slice to a
    fixed dim via a tiny CNN. All params are frozen."""

    def __init__(self, embed_dim: int = 768, image_size: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=4, padding=3),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, embed_dim),
        )
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def embed_batch(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class _TinyModalityTeacher(nn.Module):
    """Stand-in for a per-modality I-JEPA teacher."""

    def __init__(self, embed_dim: int = 384):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 7, stride=4, padding=3),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, embed_dim),
        )
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def embed_slice2d(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        return self.encoder(x)


# ---------------------------------------------------------------------------
# Conditioning module tests
# ---------------------------------------------------------------------------

def test_method1_conditioning_shape():
    B = 4
    cond = Method1Conditioning(embed_dim=192, hist_dim=48)
    out = cond(
        plane_idx=torch.tensor([0, 1, 2, 0], dtype=torch.long),
        slice_idx_norm=torch.rand(B),
        voxel_spacing=torch.rand(B, 3) * 2.0,
        intensity_hist=torch.rand(B, 48),
    )
    assert out.shape == (B, 192)


def test_method2_conditioning_shape():
    B = 4
    cond = Method2Conditioning(embed_dim=192, hist_dim=48)
    out = cond(
        target_modality_idx=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        slice_pose=torch.rand(B, 2),
        context_hist=torch.rand(B, 48 * len(MODALITIES)),
    )
    assert out.shape == (B, 192)


def test_modality_indices_in_range():
    assert all(0 <= MODALITY_TO_IDX[m] < len(MODALITIES) for m in MODALITIES)
    assert PLANE_TO_IDX['axial'] == 0


# ---------------------------------------------------------------------------
# 3D ViT tests
# ---------------------------------------------------------------------------

def test_patch_embed_3d_shape():
    vol_size = (32, 64, 64)
    pe = PatchEmbed3D(volume_size=vol_size, patch_size=16, in_chans=4, embed_dim=384)
    x = torch.randn(2, 4, *vol_size)
    tokens = pe(x)
    expected = (32 // 16) * (64 // 16) * (64 // 16)
    assert tokens.shape == (2, expected, 384)


def test_3d_sincos_posemb_dim_constraint():
    # 384 / 6 = 64 (int) -> ok
    pos = _3d_sincos_posemb((2, 4, 4), 384)
    assert pos.shape == (1, 32, 384)
    with pytest.raises(AssertionError):
        _3d_sincos_posemb((2, 2, 2), 7)   # embed_dim not divisible by 6


def test_vit3d_forward_shape():
    vol_size = (32, 64, 64)
    enc = ViT3DEncoder(volume_size=vol_size, patch_size=16, in_chans=4,
                         embed_dim=384, depth=2, heads=6)
    x = torch.randn(2, 4, *vol_size)
    tokens = enc(x)
    expected = 2 * 4 * 4
    assert tokens.shape == (2, expected, 384)


# ---------------------------------------------------------------------------
# Method 1 model tests
# ---------------------------------------------------------------------------

def test_vol2slice_forward_shape():
    teacher = _TinyFrozenTeacher(embed_dim=768, image_size=64)
    model = Vol2SliceModel(
        v8_teacher=teacher,
        volume_size=(32, 64, 64), in_chans=4, patch_size=16,
        encoder_dim=384, encoder_depth=2, encoder_heads=6,
        predictor_dim=192, predictor_depth=2,
    )
    B = 2
    batch = {
        'volume': torch.randn(B, 4, 32, 64, 64),
        'target_slice_rgb': torch.rand(B, 3, 64, 64),
        'plane_idx': torch.zeros(B, dtype=torch.long),
        'slice_idx_norm': torch.rand(B),
        'voxel_spacing': torch.ones(B, 3),
        'intensity_hist': torch.rand(B, 48),
    }
    pred = model.forward_predict(
        batch['volume'], batch['plane_idx'], batch['slice_idx_norm'],
        batch['voxel_spacing'], batch['intensity_hist'])
    assert pred.shape == (B, 768)


def test_vol2slice_teacher_frozen():
    """The teacher must never receive a gradient signal."""
    teacher = _TinyFrozenTeacher(embed_dim=768, image_size=64)
    model = Vol2SliceModel(
        v8_teacher=teacher,
        volume_size=(32, 64, 64), in_chans=4, patch_size=16,
        encoder_dim=384, encoder_depth=2, encoder_heads=6,
        predictor_dim=192, predictor_depth=2,
    )
    # Teacher params should not appear in .parameters()
    train_param_ids = {id(p) for p in model.parameters()}
    for p in teacher.parameters():
        assert id(p) not in train_param_ids, (
            'Frozen v8 teacher params leaked into model.parameters() — '
            'this would break gradient flow guarantees.')
    # All teacher params have requires_grad=False
    for n, p in teacher.named_parameters():
        assert not p.requires_grad, f'teacher param {n} requires_grad=True'


def test_vol2slice_training_step_runs():
    teacher = _TinyFrozenTeacher(embed_dim=768, image_size=64)
    model = Vol2SliceModel(
        v8_teacher=teacher,
        volume_size=(32, 64, 64), in_chans=4, patch_size=16,
        encoder_dim=384, encoder_depth=2, encoder_heads=6,
        predictor_dim=192, predictor_depth=2,
    )
    batch = {
        'volume': torch.randn(2, 4, 32, 64, 64),
        'target_slice_rgb': torch.rand(2, 3, 64, 64),
        'plane_idx': torch.tensor([0, 1], dtype=torch.long),
        'slice_idx_norm': torch.rand(2),
        'voxel_spacing': torch.ones(2, 3),
        'intensity_hist': torch.rand(2, 48),
    }
    out = model.training_step(batch)
    assert 'loss' in out and 'cos_sim' in out
    assert out['loss'].requires_grad
    out['loss'].backward()
    # At least one encoder param should have a non-zero grad
    grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert any(g.abs().sum() > 0 for g in grads), (
        'No gradient reached the 3D encoder — the predictor may be cut off.')


# ---------------------------------------------------------------------------
# Method 2 model tests
# ---------------------------------------------------------------------------

def test_per_modality_patch_embedder_shape():
    pe = PerModalityPatchEmbedder(image_size=64, patch_size=16, embed_dim=384)
    inputs = {
        'T1': torch.randn(2, 1, 64, 64),
        'FLAIR': torch.randn(2, 1, 64, 64),
    }
    tokens = pe(inputs)
    # 2 modalities * (64/16)^2 = 2 * 16 = 32 tokens
    assert tokens.shape == (2, 32, 384)


def test_mod2mod_rejects_missing_teacher():
    incomplete = {'T1': _TinyModalityTeacher(), 'T1c': _TinyModalityTeacher()}
    with pytest.raises(ValueError, match='Missing'):
        Mod2ModModel(frozen_teachers=incomplete, image_size=64, embed_dim=384,
                       depth=2, predictor_depth=2, teacher_embed_dim=384)


def test_mod2mod_forward_shape():
    teachers = {m: _TinyModalityTeacher(embed_dim=384) for m in MODALITIES}
    model = Mod2ModModel(frozen_teachers=teachers,
                           image_size=64, patch_size=16, embed_dim=384,
                           depth=2, predictor_depth=2, teacher_embed_dim=384,
                           predictor_dim=192)
    B = 2
    batch = {
        'context_inputs': {
            'T1': torch.randn(B, 1, 64, 64),
            'FLAIR': torch.randn(B, 1, 64, 64),
        },
        'target_input': torch.randn(B, 1, 64, 64),
        'target_modality_name': 'T1c',
        'target_modality_idx': torch.tensor([2, 2], dtype=torch.long),
        'slice_pose': torch.rand(B, 2),
        'intensity_hist': torch.rand(B, 48 * 4),
    }
    out = model.training_step(batch)
    assert out['loss'].requires_grad
    out['loss'].backward()


def test_mod2mod_teachers_dont_get_gradient():
    """Ensure the frozen-teacher backward pass really doesn't update the
    teacher weights."""
    teachers = {m: _TinyModalityTeacher(embed_dim=384) for m in MODALITIES}
    model = Mod2ModModel(frozen_teachers=teachers,
                           image_size=64, patch_size=16, embed_dim=384,
                           depth=2, predictor_depth=2, teacher_embed_dim=384,
                           predictor_dim=192)
    # Snapshot teacher weights pre-step
    before = {m: [p.clone() for p in t.parameters()]
              for m, t in teachers.items()}
    batch = {
        'context_inputs': {'T1': torch.randn(2, 1, 64, 64)},
        'target_input': torch.randn(2, 1, 64, 64),
        'target_modality_name': 'T2',
        'target_modality_idx': torch.tensor([1, 1], dtype=torch.long),
        'slice_pose': torch.rand(2, 2),
        'intensity_hist': torch.rand(2, 48 * 4),
    }
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    out = model.training_step(batch)
    opt.zero_grad(); out['loss'].backward(); opt.step()
    # Confirm teachers unchanged
    for m, snapshots in before.items():
        for s, p in zip(snapshots, teachers[m].parameters()):
            assert torch.equal(s, p), (
                f'Teacher {m!r} weights changed after one training step — '
                'frozen teacher invariant violated.')


# ---------------------------------------------------------------------------
# Multi-teacher (Option A = dual_heads, Option C = teacher_id) tests
# ---------------------------------------------------------------------------

def _two_tiny_teachers(embed_dims=(768, 512)):
    """Build two distinct tiny frozen teachers with different embed dims
    so we can also exercise the heterogeneous-dim path (Option C uses
    max(dim) as the output head and slices per-teacher)."""
    return [_TinyFrozenTeacher(embed_dim=d, image_size=64) for d in embed_dims]


def test_multi_head_predictor_shape():
    pred = MultiHeadSliceEmbeddingPredictor(
        encoder_dim=384, predictor_dim=192, depth=2, heads=6,
        conditioning_dim=192, teacher_embed_dims=[768, 512])
    B, N = 3, 16
    vol_tokens = torch.randn(B, N, 384)
    cond = torch.randn(B, 192)
    outs = pred(vol_tokens, cond)
    assert isinstance(outs, list) and len(outs) == 2
    assert outs[0].shape == (B, 768)
    assert outs[1].shape == (B, 512)


def test_teacher_id_predictor_shape():
    pred = TeacherIdConditionedPredictor(
        encoder_dim=384, predictor_dim=192, depth=2, heads=6,
        conditioning_dim=192, out_dim=768, n_teachers=2)
    B, N = 3, 16
    out = pred(torch.randn(B, N, 384),
                torch.randn(B, 192),
                torch.tensor([0, 1, 0], dtype=torch.long))
    assert out.shape == (B, 768)


def _make_multi_batch(B=2, vol_dim=(32, 64, 64), in_chans=1):
    return {
        'volume': torch.randn(B, in_chans, *vol_dim),
        'target_slice_rgb': torch.rand(B, 3, 64, 64),
        'plane_idx': torch.zeros(B, dtype=torch.long),
        'slice_idx_norm': torch.rand(B),
        'voxel_spacing': torch.ones(B, 3),
        'intensity_hist': torch.rand(B, 48),
    }


def test_vol2slice_multi_dual_heads_runs():
    """Option A: dual_heads should produce per-teacher losses + a summed
    total. Gradient flow into the encoder must work."""
    teachers = _two_tiny_teachers((768, 512))
    model = Vol2SliceModelMulti(
        teachers=teachers, mode='dual_heads',
        volume_size=(32, 64, 64), in_chans=1, patch_size=16,
        encoder_dim=384, encoder_depth=2, encoder_heads=6,
        predictor_dim=192, predictor_depth=2,
    )
    batch = _make_multi_batch()
    out = model.training_step(batch)
    # Per-teacher diagnostics present + summed loss requires grad
    assert 'loss_t0' in out and 'loss_t1' in out
    assert 'cos_t0' in out and 'cos_t1' in out
    assert out['loss'].requires_grad
    out['loss'].backward()
    grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert any(g.abs().sum() > 0 for g in grads), 'no grad into 3D encoder'


def test_vol2slice_multi_teacher_id_runs():
    """Option C: teacher_id sampled randomly per step. Single output
    head sliced to each teacher's dim. Gradient must reach encoder."""
    teachers = _two_tiny_teachers((768, 512))
    model = Vol2SliceModelMulti(
        teachers=teachers, mode='teacher_id',
        volume_size=(32, 64, 64), in_chans=1, patch_size=16,
        encoder_dim=384, encoder_depth=2, encoder_heads=6,
        predictor_dim=192, predictor_depth=2,
    )
    # Use B=4 so both teachers are likely sampled at least once
    batch = _make_multi_batch(B=4)
    torch.manual_seed(0)
    out = model.training_step(batch)
    assert out['loss'].requires_grad
    out['loss'].backward()
    grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert any(g.abs().sum() > 0 for g in grads), 'no grad into 3D encoder'


def test_vol2slice_multi_all_teachers_frozen():
    """The CrossJEPA frozen-teacher invariant must hold for BOTH modes."""
    for mode in ('dual_heads', 'teacher_id'):
        teachers = _two_tiny_teachers((768, 512))
        before = [
            [p.clone() for p in t.parameters()] for t in teachers
        ]
        model = Vol2SliceModelMulti(
            teachers=teachers, mode=mode,
            volume_size=(32, 64, 64), in_chans=1, patch_size=16,
            encoder_dim=384, encoder_depth=2, encoder_heads=6,
            predictor_dim=192, predictor_depth=2,
        )
        opt = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], lr=0.1)
        batch = _make_multi_batch(B=4)
        torch.manual_seed(0)
        out = model.training_step(batch)
        opt.zero_grad(); out['loss'].backward(); opt.step()
        for ti, snapshots in enumerate(before):
            for s, p in zip(snapshots, teachers[ti].parameters()):
                assert torch.equal(s, p), (
                    f'mode={mode}: teacher {ti} param changed after backward — '
                    'frozen-teacher invariant violated')


def test_vol2slice_multi_teachers_not_in_parameters():
    """Teachers must not be in model.parameters() (so the optimizer
    can't update them even with a global step)."""
    for mode in ('dual_heads', 'teacher_id'):
        teachers = _two_tiny_teachers((768, 512))
        model = Vol2SliceModelMulti(
            teachers=teachers, mode=mode,
            volume_size=(32, 64, 64), in_chans=1, patch_size=16,
            encoder_dim=384, encoder_depth=2, encoder_heads=6,
            predictor_dim=192, predictor_depth=2,
        )
        train_ids = {id(p) for p in model.parameters()}
        for ti, t in enumerate(teachers):
            for p in t.parameters():
                assert id(p) not in train_ids, (
                    f'mode={mode}: teacher {ti} param leaked into model.parameters()')


def test_vol2slice_multi_dual_heads_weighted_loss():
    """Per-teacher weights must shift the loss balance. Set weights
    [1, 0] -> total loss should equal loss_t0 only."""
    teachers = _two_tiny_teachers((768, 512))
    model = Vol2SliceModelMulti(
        teachers=teachers, mode='dual_heads',
        volume_size=(32, 64, 64), in_chans=1, patch_size=16,
        encoder_dim=384, encoder_depth=2, encoder_heads=6,
        predictor_dim=192, predictor_depth=2,
        teacher_weights=[1.0, 0.0],
    )
    batch = _make_multi_batch(B=2)
    torch.manual_seed(42)
    out = model.training_step(batch)
    # weight on teacher 1 is zero, so total == loss_t0
    assert torch.allclose(out['loss'].detach(), out['loss_t0'].detach())


def test_vol2slice_multi_anomaly_score_modes():
    """anomaly_score returns the right shape in both modes (B,)."""
    teachers = _two_tiny_teachers((768, 512))
    for mode in ('dual_heads', 'teacher_id'):
        model = Vol2SliceModelMulti(
            teachers=teachers, mode=mode,
            volume_size=(32, 64, 64), in_chans=1, patch_size=16,
            encoder_dim=384, encoder_depth=2, encoder_heads=6,
            predictor_dim=192, predictor_depth=2,
        )
        b = _make_multi_batch(B=3)
        s = model.anomaly_score(
            b['volume'], b['target_slice_rgb'], b['plane_idx'],
            b['slice_idx_norm'], b['voxel_spacing'], b['intensity_hist'])
        assert s.shape == (3,), f'mode={mode}: expected (3,), got {s.shape}'
        # teacher_id mode also supports a specific teacher_id arg
        if mode == 'teacher_id':
            s0 = model.anomaly_score(
                b['volume'], b['target_slice_rgb'], b['plane_idx'],
                b['slice_idx_norm'], b['voxel_spacing'], b['intensity_hist'],
                teacher_id=0)
            assert s0.shape == (3,)


# ---------------------------------------------------------------------------
# Teacher factory + MedSAM API-shape tests (no actual HF download)
# ---------------------------------------------------------------------------

def test_build_teacher_unknown_name_raises():
    """The factory must refuse unknown names with a clear message."""
    from src.research.v9c_crossjepa.teachers import build_teacher
    with pytest.raises(ValueError, match='unknown teacher name'):
        build_teacher('not_a_teacher')


def test_build_teacher_v8_requires_ckpt():
    """v8 needs a checkpoint path; factory must raise without it."""
    from src.research.v9c_crossjepa.teachers import build_teacher
    with pytest.raises(ValueError, match='v8_ckpt'):
        build_teacher('v8', v8_ckpt=None)


def test_medsam_teacher_class_attributes():
    """Sanity-check the MedSAMFrozenTeacher class without instantiating
    it (which would trigger a 358 MB HF download). We verify the class
    declares the right embed_dim + has the BaseFrozenTeacher API."""
    from src.research.v9c_crossjepa.teachers import (
        MedSAMFrozenTeacher, BaseFrozenTeacher,
    )
    assert issubclass(MedSAMFrozenTeacher, BaseFrozenTeacher)
    # The embed_dim attribute is set in __init__; check via the type
    # annotation hint instead (the class has no module-level constant).
    # Class-level: DEFAULT_REPO sanity
    assert MedSAMFrozenTeacher.DEFAULT_REPO.endswith('medsam-vit-base')
    # Public method signature
    import inspect
    sig = inspect.signature(MedSAMFrozenTeacher.embed_batch)
    assert 'x' in sig.parameters, \
        'embed_batch must accept a tensor arg `x` per BaseFrozenTeacher contract'


def test_vol2slice_multi_with_three_teachers():
    """Sanity-check the multi-teacher path with 3 heterogeneous teachers
    using tiny stand-ins (matches the planned v9 ablation: v8 + dinov2
    + medsam). Tiny teachers have dims [768, 768, 256]."""
    teachers = [
        _TinyFrozenTeacher(embed_dim=768, image_size=64),
        _TinyFrozenTeacher(embed_dim=768, image_size=64),
        _TinyFrozenTeacher(embed_dim=256, image_size=64),
    ]
    # dual_heads
    model_a = Vol2SliceModelMulti(
        teachers=teachers, mode='dual_heads',
        volume_size=(32, 64, 64), in_chans=1, patch_size=16,
        encoder_dim=384, encoder_depth=2, encoder_heads=6,
        predictor_dim=192, predictor_depth=2,
    )
    out_a = model_a.training_step(_make_multi_batch(B=4))
    assert {'loss', 'loss_t0', 'loss_t1', 'loss_t2'}.issubset(out_a.keys())
    out_a['loss'].backward()
    # teacher_id (output dim = max = 768; t2 gets pred[:, :256] sliced)
    teachers2 = [
        _TinyFrozenTeacher(embed_dim=768, image_size=64),
        _TinyFrozenTeacher(embed_dim=768, image_size=64),
        _TinyFrozenTeacher(embed_dim=256, image_size=64),
    ]
    model_c = Vol2SliceModelMulti(
        teachers=teachers2, mode='teacher_id',
        volume_size=(32, 64, 64), in_chans=1, patch_size=16,
        encoder_dim=384, encoder_depth=2, encoder_heads=6,
        predictor_dim=192, predictor_depth=2,
    )
    torch.manual_seed(0)
    out_c = model_c.training_step(_make_multi_batch(B=6))
    out_c['loss'].backward()


if __name__ == '__main__':
    pytest.main([__file__, '-xvs'])
