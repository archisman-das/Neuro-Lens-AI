"""Prepare BraTS 2020 as 3D volumes for the v4 SOTA-aimed trainer.

Unlike prepare_brats_dataset.py (which slices to 2D), this script preserves
the volumetric structure (240x240x155) so a 3D U-Net can learn inter-slice
context - the single biggest quality gap of the v3 stack.

Output structure:

    dataset_brats_3d/<split>/<patient>.npz
        image: float32 (4, D, H, W)  channels = (t1, t1ce, t2, flair)
        mask:  uint8 (D, H, W)        binary whole-tumor
        meta:  dict with patient id, original shape, spacing

Volumes are per-patient z-score normalised per modality (within the brain
mask) and cropped to remove all-zero borders to save memory/disk. We do NOT
resample to a fixed grid here; the trainer crops random 128**3 patches at
load time so the original (240, 240, 155) shape is fine.

Patient-level split (default 70/15/15), seeded for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

MOD_SUFFIXES = {'t1': '_t1.nii', 't1ce': '_t1ce.nii', 't2': '_t2.nii', 'flair': '_flair.nii'}


def _load_volume(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj)


def _zscore_in_brain(vol: np.ndarray, brain: np.ndarray) -> np.ndarray:
    a = vol.astype(np.float32)
    sel = a[brain > 0]
    if sel.size == 0:
        return a
    mu, sd = float(sel.mean()), float(sel.std())
    if sd < 1e-3:
        return a - mu
    return (a - mu) / sd


def _patient_volume_paths(pdir: Path):
    stem = pdir.name
    out = {}
    for k, sfx in MOD_SUFFIXES.items():
        nii = pdir / f'{stem}{sfx}'
        gz = pdir / f'{stem}{sfx}.gz'
        if nii.exists():
            out[k] = nii
        elif gz.exists():
            out[k] = gz
        else:
            return None
    seg = pdir / f'{stem}_seg.nii'
    seg_gz = pdir / f'{stem}_seg.nii.gz'
    if seg.exists():
        out['seg'] = seg
    elif seg_gz.exists():
        out['seg'] = seg_gz
    else:
        return None
    return out


def _find_patient_dirs(root: Path) -> list[Path]:
    out = []
    for sub in root.rglob('*'):
        if sub.is_dir() and (list(sub.glob('*_seg.nii')) or list(sub.glob('*_seg.nii.gz'))):
            out.append(sub)
    return sorted(set(out))


def _tight_bbox(brain: np.ndarray, pad: int = 4) -> tuple[slice, slice, slice]:
    if brain.sum() == 0:
        return slice(None), slice(None), slice(None)
    ax0 = np.where(brain.any(axis=(1, 2)))[0]
    ax1 = np.where(brain.any(axis=(0, 2)))[0]
    ax2 = np.where(brain.any(axis=(0, 1)))[0]
    sh = brain.shape
    s = (
        slice(max(0, int(ax0[0]) - pad), min(sh[0], int(ax0[-1]) + 1 + pad)),
        slice(max(0, int(ax1[0]) - pad), min(sh[1], int(ax1[-1]) + 1 + pad)),
        slice(max(0, int(ax2[0]) - pad), min(sh[2], int(ax2[-1]) + 1 + pad)),
    )
    return s


def process_patient(pdir: Path, out_path: Path) -> dict:
    paths = _patient_volume_paths(pdir)
    if paths is None:
        return {'patient': pdir.name, 'skipped': True, 'reason': 'missing files'}
    seg = _load_volume(paths['seg']).astype(np.uint8)
    seg_bin = (seg > 0).astype(np.uint8)

    # Brain mask: any voxel where any modality is > 0 (BraTS volumes have
    # background zeros outside the brain).
    mods = {k: _load_volume(paths[k]) for k in MOD_SUFFIXES}
    brain = np.zeros_like(mods['t1'], dtype=np.uint8)
    for v in mods.values():
        brain = np.maximum(brain, (v > 0).astype(np.uint8))

    # Crop to brain bounding box to save memory at train time
    bbox = _tight_bbox(brain, pad=4)
    seg_bin = seg_bin[bbox]
    brain = brain[bbox]
    # Z-score per modality within the brain mask, then stack into (C, D, H, W).
    stacked = np.stack([_zscore_in_brain(mods[k][bbox], brain) for k in MOD_SUFFIXES], axis=0).astype(np.float32)
    # MONAI expects (C, H, W, D) or (C, D, H, W) - we use (C, D, H, W) and
    # let the loader transpose as needed. nibabel default is (H, W, D); after
    # our np.stack with axis=0 we have (C, H, W, D). Swap to (C, D, H, W):
    stacked = stacked.transpose(0, 3, 1, 2)  # (C, D, H, W)
    seg_bin = seg_bin.transpose(2, 0, 1)      # (D, H, W)

    np.savez_compressed(
        out_path,
        image=stacked,
        mask=seg_bin,
        patient=pdir.name,
        shape=np.array(stacked.shape),
    )
    return {
        'patient': pdir.name,
        'image_shape': list(stacked.shape),
        'mask_shape': list(seg_bin.shape),
        'tumor_voxels': int(seg_bin.sum()),
    }


def split_patients(patients: list[Path], ratios, seed: int) -> dict[str, list[Path]]:
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError('ratios must sum to 1.0')
    rng = random.Random(seed)
    shuffled = patients[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    n_test = n - n_train - n_val
    return {'train': shuffled[:n_train], 'val': shuffled[n_train:n_train + n_val], 'test': shuffled[n_train + n_val:]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='data_sources/BraTS2020_TrainingData')
    p.add_argument('--output', default='dataset_brats_3d')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train', type=float, default=0.70)
    p.add_argument('--val', type=float, default=0.15)
    p.add_argument('--test', type=float, default=0.15)
    p.add_argument('--clean', action='store_true')
    p.add_argument('--max_patients', type=int, default=0)
    args = p.parse_args()

    src = Path(args.source)
    patients = _find_patient_dirs(src)
    if not patients:
        raise FileNotFoundError(f'No BraTS patient folders found under {src}')
    if args.max_patients > 0:
        patients = patients[: args.max_patients]
    print(f'[info] Found {len(patients)} patient folders')

    split = split_patients(patients, (args.train, args.val, args.test), args.seed)
    print(f'[info] Patients: train={len(split["train"])} val={len(split["val"])} test={len(split["test"])}')

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = {'splits': {}, 'channels': list(MOD_SUFFIXES.keys()), 'storage_format': '.npz (image=(C,D,H,W) float32, mask=(D,H,W) uint8)'}
    for split_name, pdirs in split.items():
        out_split = out_root / split_name
        if args.clean and out_split.exists():
            shutil.rmtree(out_split)
        out_split.mkdir(parents=True, exist_ok=True)
        records = []
        for pdir in tqdm(pdirs, desc=f'{split_name} ({len(pdirs)} pts)'):
            out_path = out_split / f'{pdir.name}.npz'
            rec = process_patient(pdir, out_path)
            records.append(rec)
        summary['splits'][split_name] = {
            'n_patients': len(pdirs),
            'patient_ids': [p.name for p in pdirs],
            'records': records,
        }
        print(f'  {split_name}: {len(records)} patients written to {out_split}')

    summary_path = out_root / 'brats_3d_split_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'\n[done] Wrote {summary_path}')


if __name__ == '__main__':
    main()
