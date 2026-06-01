"""Add multi-view (axial/sagittal/coronal) x multi-modal (T1/T1ce/T2/FLAIR)
BraTS slices to dataset_v8 for the classifier retraining.

Why:
  - Current dataset_v8 BraTS contribution is T1c-axial only (per
    dataset_brats/brats_split_summary.json). That's why the classifiers
    blew up on UniData's sagittal/coronal slices and on T2/FLAIR/DWI.
  - The raw BraTS 2020 NIfTIs DO contain all 4 modalities at full
    volume — we just sliced narrowly. This script extracts the missing
    diversity.

Output layout (appended to existing dataset_v8):
  dataset_v8/<split>/images/brats_<modality>_<patient>_<view>_s<slice>.png
  dataset_v8/<split>/masks/  same filename, derived from BraTS seg
  Examples:
    brats_t1_BraTS20_Training_001_ax_s060.png
    brats_t2_BraTS20_Training_001_sag_s120.png
    brats_flair_BraTS20_Training_001_cor_s100.png

Splits: reuse dataset_brats/brats_split_summary.json so patient
assignments match what the segmenter already used (no leak).

Modalities included: t1, t2, flair  (we SKIP t1ce because that's already
covered by the existing brats_t1c_* files; including it would just be
near-duplicates).

Per-patient slice budget: 6 axial + 6 sagittal + 6 coronal per
modality = 54 slices per patient per modality = 162 slices per patient
across 3 modalities. With 369 patients = ~60k slices total, distributed
80/10/10 across train/val/test by the existing patient split.

Label per slice: tumor iff seg mask >= MIN_TUMOR_AREA (50 px), matching
the rest of the repo's convention.

Run:
  python scripts/augment_v8_with_brats_mvmm.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / 'dataset_v8'
BRATS_ROOT = ROOT / 'data_sources' / 'BraTS2020_TrainingData' / 'MICCAI_BraTS2020_TrainingData'
SPLIT_JSON = ROOT / 'dataset_brats' / 'brats_split_summary.json'
MIN_TUMOR_AREA = 50

# Modalities to add (skip t1ce — already covered by existing brats_t1c_* files)
MODALITIES = ('t1', 't2', 'flair')

# Slices per plane per patient (deterministic indices across plane extent)
N_SLICES_PER_PLANE = 6
# Image output size (matches the rest of dataset_v8)
OUT_SIZE = 192


def _norm_uint8(slice2d: np.ndarray) -> np.ndarray:
    """Volume slice float -> 8-bit. Use 1-99 percentile windowing so
    bright outliers don't compress the useful dynamic range."""
    arr = slice2d.astype(np.float32)
    lo, hi = np.percentile(arr[arr > 0], (1, 99)) if (arr > 0).any() else (0.0, 1.0)
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    return arr.astype(np.uint8)


def _resize_image(arr: np.ndarray) -> np.ndarray:
    return cv2.resize(arr, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_LINEAR)


def _resize_mask(arr: np.ndarray) -> np.ndarray:
    return cv2.resize(arr, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_NEAREST)


def _slice_indices(axis_len: int, n: int) -> list[int]:
    """Evenly-spaced indices avoiding the empty top/bottom rims."""
    margin = max(1, axis_len // 10)
    return [int(round(margin + (axis_len - 2 * margin) * i / max(1, n - 1)))
            for i in range(n)]


def _patient_split_map() -> dict[str, str]:
    """Returns {patient_id -> 'train'/'val'/'test'}."""
    d = json.loads(SPLIT_JSON.read_text())
    splits = d['splits']
    out = {}
    for split_name in ('train', 'val', 'test'):
        for pid in splits[split_name]['patient_ids']:
            out[pid] = split_name
    return out


def main():
    if not BRATS_ROOT.exists():
        sys.exit(f'ERROR: BraTS NIfTIs not found at {BRATS_ROOT}')
    if not SPLIT_JSON.exists():
        sys.exit(f'ERROR: split json not found at {SPLIT_JSON}')

    pmap = _patient_split_map()
    patients = sorted(p for p in BRATS_ROOT.iterdir()
                       if p.is_dir() and p.name in pmap)
    print(f'[init] {len(patients)} BraTS patients with split assignments')

    for s in ('train', 'val', 'test'):
        (DATASET / s / 'images').mkdir(parents=True, exist_ok=True)
        (DATASET / s / 'masks').mkdir(parents=True, exist_ok=True)

    # Counters
    counts = {s: {'tumor': 0, 'no_tumor': 0} for s in ('train', 'val', 'test')}
    skipped_existing = 0
    failed_patients = 0
    t0 = time.perf_counter()

    for pi, pdir in enumerate(patients):
        pid = pdir.name
        split = pmap[pid]
        try:
            # Load seg once; load each modality lazily
            seg_nii = nib.load(str(pdir / f'{pid}_seg.nii'))
            seg = seg_nii.get_fdata()
            for modality in MODALITIES:
                mod_path = pdir / f'{pid}_{modality}.nii'
                if not mod_path.exists():
                    continue
                vol = nib.load(str(mod_path)).get_fdata()
                # vol shape: (H, W, D) = (240, 240, 155) typically
                # Axial = vol[:, :, z], Sagittal = vol[x, :, :], Coronal = vol[:, y, :]
                planes = {
                    'ax':  (vol, seg, 2),   # axial: index axis=2
                    'sag': (vol, seg, 0),   # sagittal: index axis=0
                    'cor': (vol, seg, 1),   # coronal: index axis=1
                }
                for plane_name, (v, s, axis) in planes.items():
                    indices = _slice_indices(v.shape[axis], N_SLICES_PER_PLANE)
                    for idx in indices:
                        if axis == 2:
                            img = v[:, :, idx]; m = s[:, :, idx]
                        elif axis == 0:
                            img = v[idx, :, :]; m = s[idx, :, :]
                        else:
                            img = v[:, idx, :]; m = s[:, idx, :]
                        # Skip empty slices (no brain)
                        if (img > 0).sum() < 200:
                            continue
                        img_u8 = _norm_uint8(img)
                        img_resized = _resize_image(img_u8)
                        mask_u8 = (m > 0).astype(np.uint8) * 255
                        mask_resized = _resize_mask(mask_u8)
                        # Save 3-channel grayscale image (so the trainer's RGB
                        # preprocess matches; classifiers expect 3-channel)
                        img_rgb = np.stack([img_resized] * 3, axis=-1)
                        fname = f'brats_{modality}_{pid}_{plane_name}_s{idx:03d}.png'
                        img_out = DATASET / split / 'images' / fname
                        mask_out = DATASET / split / 'masks' / fname
                        if img_out.exists() and mask_out.exists():
                            skipped_existing += 1
                            continue
                        cv2.imwrite(str(img_out), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(str(mask_out), mask_resized)
                        if int((mask_resized > 127).sum()) >= MIN_TUMOR_AREA:
                            counts[split]['tumor'] += 1
                        else:
                            counts[split]['no_tumor'] += 1
        except Exception as exc:
            print(f'  [fail] {pid}: {type(exc).__name__}: {exc}')
            failed_patients += 1
        if (pi + 1) % 25 == 0:
            elapsed = time.perf_counter() - t0
            print(f'  [{pi+1}/{len(patients)}] elapsed={elapsed:.0f}s '
                  f'(~{elapsed/(pi+1):.1f}s/patient)')

    elapsed = time.perf_counter() - t0
    print(f'\n[done] {len(patients)} patients in {elapsed/60:.1f} min '
          f'(skipped_existing={skipped_existing}, failed={failed_patients})')
    print('\nAdded slices per split:')
    for s in ('train', 'val', 'test'):
        print(f'  {s:5s}  tumor={counts[s]["tumor"]:5d}  no_tumor={counts[s]["no_tumor"]:5d}')

    # New totals
    print('\nNew dataset_v8 totals:')
    for s in ('train', 'val', 'test'):
        n = sum(1 for _ in (DATASET / s / 'images').glob('*.png'))
        print(f'  {s:5s}  total_PNGs={n}')


if __name__ == '__main__':
    main()
