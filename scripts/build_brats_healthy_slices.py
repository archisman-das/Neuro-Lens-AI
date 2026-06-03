"""Extract per-modality healthy-looking slices from BraTS-2021 for the
4 per-modality I-JEPA teacher pretrains (Method 2 prerequisite).

For each BraTS patient (1248 fully-modal):
  - Load the segmentation mask (`*_seg.nii.gz`)
  - Identify axial slices where the tumor area is 0
    (truly healthy-looking tissue — typically near the top/bottom of the
    brain volume above/below the lesion)
  - For each of {T1, T1c, T2, FLAIR}, extract those healthy slices as
    256x256 PNG files into `dataset_v9c_modality_teachers/<modality>/`

Output layout matches what `src/train_v9b_stage1_jepa.py`'s
HealthyOnlyDataset expects (flat directory of PNG files at 256x256).

Per-modality teacher pretraining then uses one of these subdirectories:
    python src/train_v9b_stage1_jepa.py \\
        --data_dir dataset_v9c_modality_teachers/T1 \\
        --output_dir v9c_artifacts/modality_teachers/T1 \\
        --epochs 50 --batch_size 16 --amp --resume auto

Run:
    python scripts/build_brats_healthy_slices.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

BRATS_ROOT = Path('d:/datasets/brats2021')
OUT_ROOT = Path('d:/datasets/dataset_v9c_modality_teachers')
MAX_SLICES_PER_PATIENT = 12   # cap to avoid bias toward big-brain patients

# BraTS-2021 file naming
MODALITY_SUFFIX = {
    'T1': '_t1.nii.gz',
    'T1c': '_t1ce.nii.gz',
    'T2': '_t2.nii.gz',
    'FLAIR': '_flair.nii.gz',
}
SEG_SUFFIX = '_seg.nii.gz'

# Robust intensity normalization (same recipe as dataset_3d.py)
def _normalize(vol: np.ndarray) -> np.ndarray:
    nz = vol[vol > 0]
    if nz.size < 100:
        return np.zeros_like(vol, dtype=np.uint8)
    lo, hi = np.percentile(nz, [1.0, 99.0])
    if hi - lo < 1e-6:
        return np.zeros_like(vol, dtype=np.uint8)
    out = ((vol - lo) / (hi - lo)).clip(0, 1)
    return (out * 255).astype(np.uint8)


def _slice_to_png(slice_2d_uint8: np.ndarray, target: int = 256) -> Image.Image:
    img = Image.fromarray(slice_2d_uint8, mode='L').resize(
        (target, target), Image.BILINEAR)
    return img.convert('RGB')


def main():
    print(f'[init] BraTS source: {BRATS_ROOT}')
    print(f'[init] output root: {OUT_ROOT}')
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for m in MODALITY_SUFFIX:
        (OUT_ROOT / m).mkdir(exist_ok=True)

    patient_dirs = sorted(d for d in BRATS_ROOT.iterdir() if d.is_dir())
    print(f'[init] {len(patient_dirs)} BraTS patient directories')

    counts = {m: 0 for m in MODALITY_SUFFIX}
    t0 = time.perf_counter()
    last_report = t0
    n_done = 0

    for pdir in patient_dirs:
        # Locate segmentation + all 4 modalities
        seg_path = next((f for f in pdir.iterdir()
                         if f.name.lower().endswith(SEG_SUFFIX.lower())), None)
        if seg_path is None:
            continue
        try:
            seg_data = np.asarray(nib.load(str(seg_path)).dataobj, dtype=np.uint8)
        except Exception:
            continue
        # NIfTI (X, Y, Z) -> (Z, Y, X) so axial is the first axis
        if seg_data.ndim == 3:
            seg_data = seg_data.transpose(2, 1, 0)
        D = seg_data.shape[0]

        # Identify tumor-free axial slices (seg == 0 everywhere on that slice)
        tumor_area_per_slice = (seg_data > 0).reshape(D, -1).sum(axis=1)
        healthy_slice_idx = np.where(tumor_area_per_slice == 0)[0]
        # Restrict to central 60% to drop near-air slices at top/bottom
        lo, hi = int(0.2 * D), int(0.8 * D)
        healthy_slice_idx = healthy_slice_idx[
            (healthy_slice_idx >= lo) & (healthy_slice_idx < hi)]
        if len(healthy_slice_idx) == 0:
            continue
        # Subsample uniformly to MAX_SLICES_PER_PATIENT
        if len(healthy_slice_idx) > MAX_SLICES_PER_PATIENT:
            picks = np.linspace(0, len(healthy_slice_idx) - 1,
                                 MAX_SLICES_PER_PATIENT, dtype=int)
            healthy_slice_idx = healthy_slice_idx[picks]

        # For each modality, dump the healthy slices
        for m, suf in MODALITY_SUFFIX.items():
            mod_path = next((f for f in pdir.iterdir()
                              if f.name.lower().endswith(suf.lower())), None)
            if mod_path is None:
                continue
            try:
                vol = np.asarray(nib.load(str(mod_path)).dataobj, dtype=np.float32)
            except Exception:
                continue
            if vol.ndim == 3:
                vol = vol.transpose(2, 1, 0)
            vol_u8 = _normalize(vol)
            for idx in healthy_slice_idx:
                sl = vol_u8[idx]
                out_path = OUT_ROOT / m / f'{pdir.name}_slice{int(idx):03d}.png'
                if out_path.exists():
                    continue
                _slice_to_png(sl).save(out_path, optimize=False)
                counts[m] += 1
        n_done += 1
        if time.perf_counter() - last_report > 20:
            last_report = time.perf_counter()
            rate = n_done / (time.perf_counter() - t0)
            eta = (len(patient_dirs) - n_done) / max(rate, 1e-6)
            print(f'  [{n_done}/{len(patient_dirs)}]  '
                  f'rate={rate:.2f}/s  eta={eta:.0f}s  counts={counts}')

    print(f'\n[done] {n_done} patients processed in {(time.perf_counter()-t0)/60:.1f} min')
    for m, n in counts.items():
        size_mb = sum((f.stat().st_size for f in (OUT_ROOT / m).iterdir())) / 1e6
        print(f'  {m}: {n} slices, {size_mb:.0f} MB')


if __name__ == '__main__':
    main()
