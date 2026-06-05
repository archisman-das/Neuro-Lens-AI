"""3D MRI dataset loaders for CrossJEPA Method 1 + Method 2.

Two dataset families:

  Vol2SliceDataset    : yields (volume, target_slice_rgb, plane, slice_idx,
                          voxel_spacing, intensity_hist). For Method 1.
                          Any 3D MRI volume works (BraTS, LGG, healthy).

  Mod2ModDataset      : yields (context_inputs_dict, target_input, target_
                          modality_name, slice_pose, intensity_hist). For
                          Method 2. Requires 4-modality co-registered
                          volumes — BraTS only in practice.

Both support two backend modes:
  - 'local'  : iterate a directory of NIfTI files.
  - 'hf'     : stream from a HuggingFace dataset repo (the recommended
                Colab path — no large local download).

NIfTI loading uses nibabel + intensity normalization to [0, 1] via robust
percentile clipping (1st and 99th percentile), which is the standard
medical-imaging pipeline used by BraTS preprocessing.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .conditioning import MODALITIES, MODALITY_TO_IDX, PLANES, PLANE_TO_IDX


def _load_nifti(path: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Returns (volume_data_HWD, voxel_spacing_mm). Lazy-imports nibabel."""
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError(
            'nibabel required for NIfTI loading. Install via `pip install nibabel`.'
        ) from exc
    img = nib.load(str(path))
    arr = np.asarray(img.dataobj, dtype=np.float32)
    # NIfTI is typically (X, Y, Z); we standardize to (D=Z, H=Y, W=X) for
    # the dataset interface — D is axial slice axis.
    if arr.ndim == 3:
        arr = arr.transpose(2, 1, 0)
    elif arr.ndim == 4:
        # (X, Y, Z, T) -> (T, D, H, W)
        arr = arr.transpose(3, 2, 1, 0)
    sx, sy, sz = (float(s) for s in img.header.get_zooms()[:3])
    return arr, (sz, sy, sx)  # spacing in (D, H, W) order


def _robust_normalize(volume: np.ndarray,
                       lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """Clip to [p1, p99] then min-max to [0, 1]. Robust to scanner noise
    and bright skull/fat artifacts."""
    nz = volume[volume > 0]
    if nz.size < 100:
        return np.zeros_like(volume, dtype=np.float32)
    lo, hi = np.percentile(nz, [lo_pct, hi_pct])
    if hi - lo < 1e-6:
        return np.zeros_like(volume, dtype=np.float32)
    out = ((volume - lo) / (hi - lo)).clip(0.0, 1.0)
    return out.astype(np.float32)


def _crop_or_pad_volume(volume: np.ndarray,
                         target: Tuple[int, int, int]) -> np.ndarray:
    """Center-crop or zero-pad each axis to `target` (D, H, W)."""
    out_d, out_h, out_w = target
    D, H, W = volume.shape[:3]
    # Crop / pad each axis independently
    def _fit(arr, axis, want, current):
        if current == want:
            return arr
        if current > want:
            off = (current - want) // 2
            return np.take(arr, range(off, off + want), axis=axis)
        pad_total = want - current
        pad_lo = pad_total // 2
        pad_hi = pad_total - pad_lo
        pads = [(0, 0)] * arr.ndim
        pads[axis] = (pad_lo, pad_hi)
        return np.pad(arr, pads, mode='constant', constant_values=0)
    out = _fit(volume, 0, out_d, D)
    out = _fit(out, 1, out_h, H)
    out = _fit(out, 2, out_w, W)
    return out


def _intensity_histogram(image_2d: np.ndarray, n_bins: int = 48) -> np.ndarray:
    """Normalized 48-bin intensity histogram over a 2D slice (assumed in
    [0, 1]). Returns (n_bins,) float32 summing to 1."""
    hist, _ = np.histogram(image_2d, bins=n_bins, range=(0.0, 1.0))
    total = float(hist.sum())
    if total <= 0:
        return np.zeros(n_bins, dtype=np.float32)
    return (hist / total).astype(np.float32)


def _slice_volume(volume: np.ndarray, plane: str, idx: int) -> np.ndarray:
    """Extract a 2D slice from a (D, H, W) or (D, H, W, C) volume."""
    if plane == 'axial':
        return volume[idx]
    if plane == 'sagittal':
        return volume[:, :, idx]
    if plane == 'coronal':
        return volume[:, idx, :]
    raise ValueError(f'unknown plane: {plane!r}')


def _slice_to_rgb_uint8(slice_arr: np.ndarray,
                          target_hw: Optional[tuple] = None) -> np.ndarray:
    """(H, W) float in [0, 1] -> (H', W', 3) uint8 for the v8 teacher.

    If target_hw is provided, the slice is resized (bilinear) to that
    shape. This is important when collating slices from different planes
    of a non-cube volume: axial / sagittal / coronal each give different
    (H, W) and torch.stack would fail without a common shape.
    """
    if slice_arr.ndim == 3 and slice_arr.shape[-1] in (1, 3, 4):
        slice_arr = slice_arr[..., 0]
    arr = (slice_arr.clip(0, 1) * 255.0).astype(np.uint8)
    if target_hw is not None and arr.shape != tuple(target_hw):
        try:
            from PIL import Image as _Img
            arr = np.asarray(
                _Img.fromarray(arr).resize((target_hw[1], target_hw[0]),
                                              _Img.BILINEAR),
                dtype=np.uint8)
        except ImportError:
            # PIL not available — pad/crop fallback
            arr = _center_crop_pad_2d(arr, target_hw)
    return np.stack([arr, arr, arr], axis=-1)


def _apply_intensity_aug(volume: np.ndarray,
                           rng: random.Random,
                           intensity_jitter: float,
                           contrast_jitter: float,
                           gamma_jitter: float,
                           bias_field_strength: float,
                           noise_std: float,
                           renormalize_pct: float = 0.5) -> np.ndarray:
    """In-volume preprocessing-invariance augmentation. All transforms
    are applied to the whole volume so per-slice statistics shift
    consistently (otherwise the encoder would just learn to ignore
    the augmentation).

    The five transforms:
      1. INTENSITY jitter — additive shift in [-intensity_jitter, +intensity_jitter]
      2. CONTRAST jitter — multiplicative scaling in [1 - contrast_jitter, 1 + contrast_jitter]
      3. GAMMA jitter — non-linear x ** gamma, gamma in [1 - g, 1 + g]
      4. BIAS FIELD — smooth low-frequency multiplicative field
         (simulates scanner inhomogeneity, the #1 source of per-volume
         intensity drift across MRI sites)
      5. RENORMALIZE — with probability renormalize_pct, re-clip to the
         volume's own p1/p99 (simulates a downstream pipeline that
         re-normalizes images before feeding the model — exactly the
         class of preprocessing shift that broke us on IXI)

    Each individual transform is applied with prob 0.7 so the encoder
    sees a mix of "clean" and "shifted" inputs per epoch.
    """
    out = volume.astype(np.float32, copy=True)
    if rng.random() < 0.7 and intensity_jitter > 0:
        shift = rng.uniform(-intensity_jitter, intensity_jitter)
        out = out + shift
    if rng.random() < 0.7 and contrast_jitter > 0:
        scale = rng.uniform(1.0 - contrast_jitter, 1.0 + contrast_jitter)
        out = (out - 0.5) * scale + 0.5
    if rng.random() < 0.7 and gamma_jitter > 0:
        gamma = rng.uniform(1.0 - gamma_jitter, 1.0 + gamma_jitter)
        out = np.sign(out) * (np.abs(out).clip(1e-6, None) ** gamma)
    if rng.random() < 0.7 and bias_field_strength > 0:
        # Smooth low-frequency multiplicative field — simulates scanner
        # bias inhomogeneity. Generate at 4x4x4, upsample by repeat to
        # the volume shape (piecewise-constant is fine for augmentation).
        coarse = (np.random.rand(4, 4, 4).astype(np.float32) - 0.5) * 2 * bias_field_strength
        bf = coarse
        for axis, target_size in enumerate(out.shape):
            reps = int(np.ceil(target_size / bf.shape[axis]))
            bf = bf.repeat(reps, axis=axis)
            bf = np.take(bf, range(target_size), axis=axis)
        out = out * (1.0 + bf)
    if rng.random() < 0.7 and noise_std > 0:
        out = out + np.random.randn(*out.shape).astype(np.float32) * noise_std
    out = out.clip(0.0, 1.0)
    if rng.random() < renormalize_pct:
        nz = out[out > 0]
        if nz.size > 100:
            lo, hi = np.percentile(nz, [1.0, 99.0])
            if hi - lo > 1e-6:
                out = ((out - lo) / (hi - lo)).clip(0.0, 1.0).astype(np.float32)
    return out


def _center_crop_pad_2d(arr: np.ndarray, target: tuple) -> np.ndarray:
    """Center-crop or zero-pad a 2D array to `target` (h, w)."""
    th, tw = target
    h, w = arr.shape
    out = np.zeros(target, dtype=arr.dtype)
    src_h0 = max(0, (h - th) // 2)
    src_w0 = max(0, (w - tw) // 2)
    dst_h0 = max(0, (th - h) // 2)
    dst_w0 = max(0, (tw - w) // 2)
    copy_h = min(h - src_h0, th - dst_h0)
    copy_w = min(w - src_w0, tw - dst_w0)
    out[dst_h0:dst_h0 + copy_h, dst_w0:dst_w0 + copy_w] = (
        arr[src_h0:src_h0 + copy_h, src_w0:src_w0 + copy_w])
    return out


class Vol2SliceDataset(IterableDataset):
    """Yields training samples for Method 1.

    Each sample:
      volume          : (C, D, H, W) torch.float32 in [0, 1]
      target_slice_rgb: (3, H_2d, W_2d) torch.float32 in [0, 1]
      plane_idx       : torch.long scalar
      slice_idx_norm  : torch.float scalar
      voxel_spacing   : (3,) torch.float
      intensity_hist  : (hist_dim,) torch.float

    The dataset is iterable (not map-style) so it can wrap a HF
    streaming source without materializing all volumes.
    """

    def __init__(self, scan_paths: List[str] | Sequence[str],
                 volume_size: Tuple[int, int, int] = (144, 192, 192),
                 in_channels: int = 1,
                 slices_per_volume: int = 6, hist_dim: int = 48,
                 planes: Sequence[str] = PLANES, seed: int = 0,
                 shuffle: bool = True,
                 slice_resize_hw: Tuple[int, int] = (256, 256),
                 # ---- preprocessing-invariance augmentations (fix A) ----
                 augment: bool = False,
                 aug_intensity_jitter: float = 0.20,
                 aug_contrast_jitter: float = 0.30,
                 aug_gamma_jitter: float = 0.30,
                 aug_bias_field_strength: float = 0.15,
                 aug_noise_std: float = 0.02,
                 aug_renormalize_pct: float = 0.5):
        super().__init__()
        self.scan_paths = list(scan_paths)
        self.volume_size = volume_size
        self.in_channels = in_channels
        self.slices_per_volume = slices_per_volume
        self.hist_dim = hist_dim
        self.planes = tuple(planes)
        self.rng = random.Random(seed)
        self.shuffle = shuffle
        self.slice_resize_hw = tuple(slice_resize_hw)
        # Augmentation parameters. When `augment=True` we apply a chain
        # of intensity/contrast/gamma/bias-field perturbations to BOTH
        # the volume AND the target slice (the v8 teacher then sees the
        # augmented slice, so the prediction target moves consistently).
        # This forces the encoder to learn preprocessing-pipeline-
        # invariant healthy-brain features rather than memorizing the
        # specific intensity distribution of the training set (which
        # was the failure mode caught by the IXI held-out eval).
        self.augment = augment
        self.aug_intensity_jitter = aug_intensity_jitter
        self.aug_contrast_jitter = aug_contrast_jitter
        self.aug_gamma_jitter = aug_gamma_jitter
        self.aug_bias_field_strength = aug_bias_field_strength
        self.aug_noise_std = aug_noise_std
        self.aug_renormalize_pct = aug_renormalize_pct

    def __iter__(self) -> Iterator[dict]:
        paths = list(self.scan_paths)
        if self.shuffle:
            self.rng.shuffle(paths)
        for p in paths:
            try:
                vol, spacing = _load_nifti(p)
            except Exception:
                continue
            if vol.ndim == 4:
                # Multi-modality volume (T, D, H, W) — use first modality
                vol = vol[0]
            vol = _robust_normalize(vol)
            vol = _crop_or_pad_volume(vol, self.volume_size)
            # Apply preprocessing-invariance augmentation to the volume
            # ONCE before slicing — so all extracted slices and their
            # teacher targets see the same shift. This is critical: if
            # we re-augmented per slice, the encoder would learn to
            # average out the noise, not become invariant to it.
            if self.augment:
                vol = _apply_intensity_aug(
                    vol, self.rng,
                    intensity_jitter=self.aug_intensity_jitter,
                    contrast_jitter=self.aug_contrast_jitter,
                    gamma_jitter=self.aug_gamma_jitter,
                    bias_field_strength=self.aug_bias_field_strength,
                    noise_std=self.aug_noise_std,
                    renormalize_pct=self.aug_renormalize_pct,
                )
            D, H, W = vol.shape
            scan_id = Path(p).stem
            for _ in range(self.slices_per_volume):
                plane = self.rng.choice(self.planes)
                n_slices = (D if plane == 'axial'
                             else W if plane == 'sagittal' else H)
                lo, hi = int(0.2 * n_slices), int(0.8 * n_slices)
                idx = self.rng.randint(lo, max(lo, hi - 1))
                slice_2d = _slice_volume(vol, plane, idx)
                slice_rgb = _slice_to_rgb_uint8(slice_2d,
                                                  target_hw=self.slice_resize_hw)
                hist = _intensity_histogram(slice_2d, n_bins=self.hist_dim)
                # Volume tensor: (C, D, H, W). For single-channel volumes
                # we replicate to in_channels to match the model's
                # expected input width (e.g. 4 channels for the BraTS
                # baseline architecture).
                vol_t = torch.from_numpy(vol).unsqueeze(0)
                if self.in_channels > 1:
                    vol_t = vol_t.expand(self.in_channels, -1, -1, -1).contiguous()
                yield {
                    'volume': vol_t,
                    'target_slice_rgb': torch.from_numpy(slice_rgb)
                                                .permute(2, 0, 1)
                                                .float() / 255.0,
                    'plane_idx': torch.tensor(PLANE_TO_IDX[plane], dtype=torch.long),
                    'slice_idx_norm': torch.tensor(idx / max(n_slices - 1, 1),
                                                     dtype=torch.float32),
                    'voxel_spacing': torch.tensor(spacing, dtype=torch.float32),
                    'intensity_hist': torch.from_numpy(hist),
                    'scan_id': scan_id,
                    'plane': plane,
                    'slice_idx': idx,
                }


class Mod2ModDataset(IterableDataset):
    """Yields training samples for Method 2 (BraTS 4-modality).

    `scan_index` is a list of dicts; each dict has paths for all 4
    modalities of one patient:
        {'T1': str, 'T1c': str, 'T2': str, 'FLAIR': str, 'scan_id': str}

    Each iteration step:
      - load all 4 modalities for one patient (after normalization +
        crop to a common shape)
      - pick a random target modality m
      - pick a random non-empty subset S of remaining modalities
      - pick a random axial slice from the central 60%
      - emit {context_inputs: {m: (1, H, W) for m in S},
              target_input: (1, H, W), target_modality_name: m, ...}

    Because the predictor's teacher is called once per batch, the
    collate_fn should group samples by `target_modality_name`. The
    `collate_mod2mod` helper below does that.
    """

    def __init__(self, scan_index: List[dict], image_size: int = 256,
                 hist_dim: int = 48, seed: int = 0,
                 shuffle: bool = True):
        super().__init__()
        self.scan_index = list(scan_index)
        self.image_size = image_size
        self.hist_dim = hist_dim
        self.rng = random.Random(seed)
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[dict]:
        idx = list(range(len(self.scan_index)))
        if self.shuffle:
            self.rng.shuffle(idx)
        for i in idx:
            scan = self.scan_index[i]
            modality_data = {}
            spacing = None
            for m in MODALITIES:
                p = scan.get(m)
                if p is None:
                    continue
                try:
                    vol, sp = _load_nifti(p)
                except Exception:
                    continue
                vol = _robust_normalize(vol)
                vol = _crop_or_pad_volume(vol, (155, self.image_size, self.image_size))
                modality_data[m] = vol
                if spacing is None:
                    spacing = sp
            if len(modality_data) < 2:
                continue
            present = list(modality_data.keys())
            target = self.rng.choice(present)
            remaining = [m for m in present if m != target]
            # Non-empty random subset of remaining
            subset_size = self.rng.randint(1, len(remaining))
            context_mods = self.rng.sample(remaining, subset_size)
            # Pick a random axial slice (central 60%)
            D = modality_data[target].shape[0]
            lo, hi = int(0.2 * D), int(0.8 * D)
            slice_idx = self.rng.randint(lo, hi - 1)
            # Context inputs
            ctx = {m: torch.from_numpy(modality_data[m][slice_idx])
                              .unsqueeze(0).float()
                   for m in context_mods}
            target_slice = modality_data[target][slice_idx]
            target_input = torch.from_numpy(target_slice).unsqueeze(0).float()
            # Per-channel intensity histogram (48 * 4 = 192-d), zeros for
            # absent channels
            hist_concat = np.zeros(self.hist_dim * len(MODALITIES), dtype=np.float32)
            for m in context_mods:
                h = _intensity_histogram(modality_data[m][slice_idx],
                                          n_bins=self.hist_dim)
                offset = MODALITY_TO_IDX[m] * self.hist_dim
                hist_concat[offset:offset + self.hist_dim] = h
            yield {
                'context_inputs': ctx,
                'target_input': target_input,
                'target_modality_name': target,
                'target_modality_idx': torch.tensor(MODALITY_TO_IDX[target],
                                                      dtype=torch.long),
                'slice_pose': torch.tensor(
                    [slice_idx / max(D - 1, 1), 0.5], dtype=torch.float32),
                'intensity_hist': torch.from_numpy(hist_concat),
                'scan_id': scan.get('scan_id', f'scan_{i}'),
            }


class Vol2SliceCrossModalDataset(IterableDataset):
    """Cross-modal Method 1c training set (BraTS 4-modality).

    This is the *proper* CrossJEPA setup for brain MRI:
      - Source: 3D volume of one MR sequence A (e.g., T1)
      - Target: 2D slice of a DIFFERENT MR sequence B at the same
        anatomical location (B != A)
      - Real semantic gap (different contrast in T1 vs T1c vs T2 vs
        FLAIR despite identical anatomy)
      - Encoder must learn intensity-invariant anatomical features to
        bridge the modality gap

    Per-sample emit:
      volume               : (C, D, H, W) float in [0, 1]  — source modality A
      target_slice_rgb     : (3, H_2d, W_2d) float — target modality B slice
      plane_idx            : long scalar
      slice_idx_norm       : float scalar in [0, 1]
      voxel_spacing        : (3,) float, mm
      source_modality_idx  : long scalar  (0=T1, 1=T1c, 2=T2, 3=FLAIR)
      target_modality_idx  : long scalar  (different from source)

    For normative pretraining we filter to TUMOR-FREE slices using the
    seg.nii.gz mask. The model learns "what does healthy modality-B
    anatomy look like, given modality-A volume context". At inference,
    slices with tumor will fail this prediction (out of training
    distribution) -> anomaly signal.

    Use with scan_index produced by `discover_brats_scans()`.
    """

    def __init__(self, scan_index: List[dict], image_size: int = 256,
                 volume_size: Tuple[int, int, int] = (144, 192, 192),
                 in_channels: int = 1,
                 pairs_per_volume: int = 6,
                 slice_resize_hw: Tuple[int, int] = (256, 256),
                 require_tumor_free_slice: bool = True,
                 min_healthy_slices: int = 3,
                 seed: int = 0, shuffle: bool = True,
                 augment: bool = False,
                 aug_intensity_jitter: float = 0.20,
                 aug_contrast_jitter: float = 0.30,
                 aug_gamma_jitter: float = 0.30,
                 aug_bias_field_strength: float = 0.15,
                 aug_noise_std: float = 0.02,
                 aug_renormalize_pct: float = 0.5):
        super().__init__()
        self.scan_index = list(scan_index)
        self.image_size = image_size
        self.volume_size = volume_size
        self.in_channels = in_channels
        self.pairs_per_volume = pairs_per_volume
        self.slice_resize_hw = tuple(slice_resize_hw)
        self.require_tumor_free_slice = require_tumor_free_slice
        self.min_healthy_slices = min_healthy_slices
        self.rng = random.Random(seed)
        self.shuffle = shuffle
        # Augmentation (same as Vol2SliceDataset for consistency)
        self.augment = augment
        self.aug_intensity_jitter = aug_intensity_jitter
        self.aug_contrast_jitter = aug_contrast_jitter
        self.aug_gamma_jitter = aug_gamma_jitter
        self.aug_bias_field_strength = aug_bias_field_strength
        self.aug_noise_std = aug_noise_std
        self.aug_renormalize_pct = aug_renormalize_pct

    @staticmethod
    def _find_seg_path(scan_entry: dict) -> Optional[str]:
        """Find the segmentation file by looking for it next to any of
        the modality files. scan_entry has paths like 'T1', 'T1c', etc."""
        for k, p in scan_entry.items():
            if k not in MODALITIES:
                continue
            # BraTS-2020/2021: <pid>_seg.nii.gz
            # BraTS-2023:      <pid>-seg.nii.gz
            for suffix in ('_seg.nii.gz', '-seg.nii.gz'):
                cand = Path(p).parent / (Path(p).parent.name + suffix)
                if cand.exists():
                    return str(cand)
            # Fall back to glob
            for cand in Path(p).parent.iterdir():
                if cand.name.lower().endswith(('_seg.nii.gz', '-seg.nii.gz')):
                    return str(cand)
        return None

    def _load_4mod(self, scan_entry: dict):
        """Load all 4 modalities of a BraTS patient + the seg mask if
        require_tumor_free_slice is True. All co-registered, so the same
        slice_idx maps to the same anatomy across modalities."""
        modality_vols = {}
        spacing = None
        for m in MODALITIES:
            p = scan_entry.get(m)
            if p is None:
                continue
            try:
                vol, sp = _load_nifti(p)
            except Exception:
                continue
            vol = _robust_normalize(vol)
            vol = _crop_or_pad_volume(vol, self.volume_size)
            modality_vols[m] = vol
            if spacing is None:
                spacing = sp
        seg_mask = None
        if self.require_tumor_free_slice:
            seg_path = self._find_seg_path(scan_entry)
            if seg_path is not None:
                try:
                    import nibabel as nib   # type: ignore
                    seg = np.asarray(nib.load(seg_path).dataobj, dtype=np.uint8)
                    if seg.ndim == 3:
                        seg = seg.transpose(2, 1, 0)
                    seg = _crop_or_pad_volume(seg.astype(np.float32),
                                               self.volume_size)
                    seg_mask = (seg > 0).astype(np.uint8)
                except Exception:
                    seg_mask = None
        return modality_vols, spacing, seg_mask

    def __iter__(self) -> Iterator[dict]:
        scans = list(self.scan_index)
        if self.shuffle:
            self.rng.shuffle(scans)
        for entry in scans:
            modality_vols, spacing, seg_mask = self._load_4mod(entry)
            if len(modality_vols) < 2:
                continue
            present = list(modality_vols.keys())
            D = next(iter(modality_vols.values())).shape[0]
            # Pick healthy-slice indices (axial only for v1; coronal/
            # sagittal could be added later)
            lo, hi = int(0.2 * D), int(0.8 * D)
            if seg_mask is not None:
                tumor_area = (seg_mask > 0).reshape(D, -1).sum(axis=1)
                healthy_idx = [int(i) for i in range(lo, hi)
                                if tumor_area[i] == 0]
                if len(healthy_idx) < self.min_healthy_slices:
                    # Patient has tumor on too many slices; skip rather
                    # than corrupt training with tumor-bearing targets
                    continue
            else:
                healthy_idx = list(range(lo, hi))
            scan_id = entry.get('scan_id', 'unknown')

            for _ in range(self.pairs_per_volume):
                # Pick distinct source + target modalities
                src = self.rng.choice(present)
                tgt_candidates = [m for m in present if m != src]
                if not tgt_candidates:
                    continue
                tgt = self.rng.choice(tgt_candidates)
                idx = self.rng.choice(healthy_idx)
                # Apply augmentation to BOTH the source volume and the
                # target slice consistently (each independently — they
                # come from different modalities so the augmentation is
                # not literally shared, but the joint distribution shifts).
                vol_arr = modality_vols[src]
                if self.augment:
                    vol_arr = _apply_intensity_aug(
                        vol_arr, self.rng,
                        intensity_jitter=self.aug_intensity_jitter,
                        contrast_jitter=self.aug_contrast_jitter,
                        gamma_jitter=self.aug_gamma_jitter,
                        bias_field_strength=self.aug_bias_field_strength,
                        noise_std=self.aug_noise_std,
                        renormalize_pct=self.aug_renormalize_pct,
                    )
                target_2d = modality_vols[tgt][idx]
                if self.augment:
                    # 3D->2D aug for the target slice: simpler — just
                    # intensity/contrast jitter, no bias field
                    if self.rng.random() < 0.7:
                        target_2d = target_2d + self.rng.uniform(
                            -self.aug_intensity_jitter,
                            self.aug_intensity_jitter)
                    if self.rng.random() < 0.7:
                        target_2d = ((target_2d - 0.5)
                                      * self.rng.uniform(1.0 - self.aug_contrast_jitter,
                                                          1.0 + self.aug_contrast_jitter)
                                      + 0.5)
                    target_2d = target_2d.clip(0.0, 1.0).astype(np.float32)
                slice_rgb = _slice_to_rgb_uint8(target_2d,
                                                  target_hw=self.slice_resize_hw)
                vol_t = torch.from_numpy(vol_arr).unsqueeze(0)
                if self.in_channels > 1:
                    vol_t = vol_t.expand(self.in_channels, -1, -1, -1).contiguous()
                yield {
                    'volume': vol_t,
                    'target_slice_rgb': torch.from_numpy(slice_rgb)
                                              .permute(2, 0, 1).float() / 255.0,
                    'plane_idx': torch.tensor(PLANE_TO_IDX['axial'],
                                                dtype=torch.long),
                    'slice_idx_norm': torch.tensor(idx / max(D - 1, 1),
                                                     dtype=torch.float32),
                    'voxel_spacing': torch.tensor(spacing,
                                                    dtype=torch.float32),
                    'source_modality_idx': torch.tensor(MODALITY_TO_IDX[src],
                                                          dtype=torch.long),
                    'target_modality_idx': torch.tensor(MODALITY_TO_IDX[tgt],
                                                          dtype=torch.long),
                    'scan_id': scan_id,
                    'plane': 'axial', 'slice_idx': int(idx),
                    'source_modality': src, 'target_modality': tgt,
                }


def collate_vol2slice_crossmodal(samples: List[dict]) -> dict:
    """Standard collate for the cross-modal dataset."""
    keys = ('volume', 'target_slice_rgb', 'plane_idx', 'slice_idx_norm',
            'voxel_spacing', 'source_modality_idx', 'target_modality_idx')
    out = {k: torch.stack([s[k] for s in samples], dim=0) for k in keys}
    out['scan_ids'] = [s['scan_id'] for s in samples]
    out['source_modalities'] = [s['source_modality'] for s in samples]
    out['target_modalities'] = [s['target_modality'] for s in samples]
    return out


def collate_mod2mod(samples: List[dict]) -> Optional[dict]:
    """Collate function that groups samples by target modality so the
    per-modality teacher is called exactly once per batch. If samples
    in a single "batch" disagree on target modality, we return the
    largest homogeneous subgroup (the DataLoader caller can either skip
    None or accept the smaller effective batch)."""
    if not samples:
        return None
    # Group by target modality
    by_target: Dict[str, List[dict]] = {}
    for s in samples:
        by_target.setdefault(s['target_modality_name'], []).append(s)
    # Pick the largest group
    target = max(by_target, key=lambda t: len(by_target[t]))
    group = by_target[target]
    # Pad context_inputs across the group (different samples may have
    # different context modalities — the model handles any non-empty
    # subset). We use the UNION of context modalities across the group;
    # for samples missing a particular modality, we zero-fill.
    ctx_keys = set()
    for s in group:
        ctx_keys.update(s['context_inputs'].keys())
    ctx_keys = [m for m in MODALITIES if m in ctx_keys]
    H = W = group[0]['target_input'].shape[-1]
    ctx_batch = {}
    for m in ctx_keys:
        stacks = []
        for s in group:
            if m in s['context_inputs']:
                stacks.append(s['context_inputs'][m])
            else:
                stacks.append(torch.zeros(1, H, W, dtype=torch.float32))
        ctx_batch[m] = torch.stack(stacks, dim=0)
    return {
        'context_inputs': ctx_batch,
        'target_input': torch.stack([s['target_input'] for s in group], dim=0),
        'target_modality_name': target,
        'target_modality_idx': torch.stack([s['target_modality_idx'] for s in group]),
        'slice_pose': torch.stack([s['slice_pose'] for s in group]),
        'intensity_hist': torch.stack([s['intensity_hist'] for s in group]),
    }


def discover_brats_scans(root: str | Path) -> List[dict]:
    """Walk a BraTS-style directory and group files by patient ID +
    modality. Handles BraTS 2020 / 2021 (underscore-separated, e.g.
    `_t1.nii.gz`) AND BraTS 2023+ (dash-separated, e.g. `-t1n.nii.gz`).

    Expected layout examples:

      BraTS 2020/2021 (rocky93/BraTS_segmentation):
        <root>/BraTS2021_00000/BraTS2021_00000_t1.nii.gz
        <root>/BraTS2021_00000/BraTS2021_00000_t1ce.nii.gz
        <root>/BraTS2021_00000/BraTS2021_00000_t2.nii.gz
        <root>/BraTS2021_00000/BraTS2021_00000_flair.nii.gz

      BraTS 2023 (obi77/brats23-first-10-examples and similar):
        <root>/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t1n.nii.gz
        <root>/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t1c.nii.gz
        <root>/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t2w.nii.gz
        <root>/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t2f.nii.gz
    """
    root = Path(root)
    out = []
    # Suffixes ordered most-specific-first to avoid false matches
    # (e.g. `_t1.nii.gz` should not match `_t1ce.nii.gz`).
    suffix_to_modality = [
        # BraTS 2023 (dash-separated short names)
        ('-t1c.nii.gz', 'T1c'),
        ('-t1n.nii.gz', 'T1'),
        ('-t2f.nii.gz', 'FLAIR'),
        ('-t2w.nii.gz', 'T2'),
        # BraTS 2020/2021 (underscore-separated)
        ('_t1ce.nii.gz', 'T1c'),
        ('_t1c.nii.gz', 'T1c'),
        ('_flair.nii.gz', 'FLAIR'),
        ('_t1.nii.gz', 'T1'),
        ('_t2.nii.gz', 'T2'),
    ]
    for patient_dir in sorted(root.iterdir()):
        if not patient_dir.is_dir():
            continue
        entry = {'scan_id': patient_dir.name}
        for f in sorted(patient_dir.iterdir()):
            low = f.name.lower()
            for suf, mod in suffix_to_modality:
                if low.endswith(suf):
                    if mod not in entry:
                        entry[mod] = str(f)
                    break
        present = [m for m in MODALITIES if m in entry]
        if len(present) >= 2:
            out.append(entry)
    return out


__all__ = [
    'Vol2SliceDataset', 'Mod2ModDataset', 'collate_mod2mod',
    'discover_brats_scans', '_load_nifti', '_robust_normalize',
    '_crop_or_pad_volume', '_intensity_histogram', '_slice_volume',
    '_slice_to_rgb_uint8',
]
