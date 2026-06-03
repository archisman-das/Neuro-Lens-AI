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
                 slice_resize_hw: Tuple[int, int] = (256, 256)):
        super().__init__()
        self.scan_paths = list(scan_paths)
        self.volume_size = volume_size
        self.in_channels = in_channels
        self.slices_per_volume = slices_per_volume
        self.hist_dim = hist_dim
        self.planes = tuple(planes)
        self.rng = random.Random(seed)
        self.shuffle = shuffle
        # All extracted 2D slices are resized to this shape so axial,
        # sagittal, coronal extractions from a non-cube volume stack
        # cleanly in a DataLoader batch. v8 teacher resizes to 384
        # internally; 256 is a reasonable intermediate that keeps file
        # transfer small without losing the structure.
        self.slice_resize_hw = tuple(slice_resize_hw)

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
    modality. Returns a list of dicts compatible with Mod2ModDataset.

    Expected layout (BraTS 2020 / 2021):
        <root>/BraTS20_Training_001/BraTS20_Training_001_t1.nii.gz
        <root>/BraTS20_Training_001/BraTS20_Training_001_t1ce.nii.gz
        <root>/BraTS20_Training_001/BraTS20_Training_001_t2.nii.gz
        <root>/BraTS20_Training_001/BraTS20_Training_001_flair.nii.gz
    """
    root = Path(root)
    out = []
    suffix_to_modality = {
        '_t1.nii.gz': 'T1', '_t1ce.nii.gz': 'T1c', '_t1c.nii.gz': 'T1c',
        '_t2.nii.gz': 'T2', '_flair.nii.gz': 'FLAIR',
    }
    for patient_dir in sorted(root.iterdir()):
        if not patient_dir.is_dir():
            continue
        entry = {'scan_id': patient_dir.name}
        for f in patient_dir.iterdir():
            for suf, mod in suffix_to_modality.items():
                if f.name.lower().endswith(suf):
                    entry[mod] = str(f)
        # Need at least 2 modalities to form valid (S, m) pairs
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
