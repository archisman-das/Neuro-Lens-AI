"""Prepare the BraTS 2020 dataset for 2D U-Net training.

Source layout (after Kaggle dataset awsaf49/brats20-dataset-training-validation
is unzipped):

    data_sources/MICCAI_BraTS2020_TrainingData/
        BraTS20_Training_001/
            BraTS20_Training_001_t1.nii          (T1-weighted)
            BraTS20_Training_001_t1ce.nii        (T1-weighted post-contrast)
            BraTS20_Training_001_t2.nii          (T2-weighted)
            BraTS20_Training_001_flair.nii       (FLAIR)
            BraTS20_Training_001_seg.nii         (segmentation: labels {0, 1, 2, 4})
        ...

We do four things:

1. Patient-level split (70/15/15). No two slices from the same patient appear
   in different splits.
2. Axial slicing: each 240x240x155 volume becomes up to 155 2D PNGs per
   modality. We skip slices whose mask has < min_tumor_voxels of foreground
   (these are head/tail slices with no tumor) but keep --empty_ratio of them
   per patient as background-only negatives.
3. Modality stacking: each output 2D image stores three different modalities
   in the R/G/B channels so the 3-channel U-Net encoder sees more than a
   single grey channel. Default: (T1c, T2, FLAIR) -- skips T1 because T1 is
   close to T1c. With --per_modality each modality is also written as a
   single-channel duplicate so we can evaluate per-modality robustness later.
4. Mask binarisation: BraTS seg labels {1, 2, 4} are all tumor sub-classes;
   we collapse them to a single "whole tumor" binary mask. Background = 0,
   tumor = 255.

Output (drop-in compatible with src/train_segmentation_v2.py):

    dataset_brats/<split>/images/<patient>_<slice>.png
    dataset_brats/<split>/masks/<patient>_<slice>.png
    dataset_brats/brats_split_summary.json
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
from tqdm import tqdm

MODALITY_SUFFIXES = {
    't1': '_t1.nii',
    't1ce': '_t1ce.nii',
    't2': '_t2.nii',
    'flair': '_flair.nii',
}


def _normalize_slice(arr: np.ndarray) -> np.ndarray:
    """Per-slice min/max normalisation to 0..255 uint8. Handles the wildly
    different intensity ranges across MRI modalities."""
    a = arr.astype(np.float32)
    lo = float(np.percentile(a, 1.0))
    hi = float(np.percentile(a, 99.0))
    if hi - lo < 1e-3:
        return np.zeros_like(arr, dtype=np.uint8)
    a = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    return (a * 255.0).astype(np.uint8)


def _find_patient_dirs(source_root: Path) -> list[Path]:
    """Patient folders are named BraTS20_Training_xxx (or similar). Find them
    anywhere under the dataset root that contains all 4 modalities + seg."""
    found = []
    for sub in source_root.rglob('*'):
        if not sub.is_dir():
            continue
        # Quick test: does the folder name look like a patient id and contain a _seg file?
        seg_candidates = list(sub.glob('*_seg.nii')) + list(sub.glob('*_seg.nii.gz'))
        if seg_candidates:
            found.append(sub)
    return sorted(set(found))


def _load_volume(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    return np.asarray(img.dataobj)


def _patient_volume_paths(patient_dir: Path) -> dict[str, Path] | None:
    """Return {modality_key: path, 'seg': path} for a patient, or None if any modality missing."""
    stem = patient_dir.name
    out: dict[str, Path] = {}
    for key, suffix in MODALITY_SUFFIXES.items():
        nii = patient_dir / f'{stem}{suffix}'
        gz = patient_dir / f'{stem}{suffix}.gz'
        if nii.exists():
            out[key] = nii
        elif gz.exists():
            out[key] = gz
        else:
            return None
    seg = patient_dir / f'{stem}_seg.nii'
    seg_gz = patient_dir / f'{stem}_seg.nii.gz'
    if seg.exists():
        out['seg'] = seg
    elif seg_gz.exists():
        out['seg'] = seg_gz
    else:
        return None
    return out


def process_patient(
    patient_dir: Path,
    images_out: Path,
    masks_out: Path,
    image_size: int,
    min_tumor_voxels: int,
    empty_ratio: float,
    channel_keys: tuple[str, str, str],
    rng: random.Random,
) -> tuple[int, int, int]:
    """Returns (kept_positive, kept_empty, dropped_empty)."""
    paths = _patient_volume_paths(patient_dir)
    if paths is None:
        return 0, 0, 0

    seg = _load_volume(paths['seg']).astype(np.uint8)
    # BraTS labels are {0, 1, 2, 4}; collapse all non-zero to whole-tumor.
    seg_bin = (seg > 0).astype(np.uint8)
    # Slice axis is typically axis 2 (D, H, W -> axis 2 = depth/axial slices).
    n_slices = seg_bin.shape[2]

    # Load the chosen 3 modalities once for the whole volume.
    mod_vols = {k: _load_volume(paths[k]) for k in channel_keys}

    positive_slice_idxs = []
    empty_slice_idxs = []
    for s in range(n_slices):
        if int(seg_bin[:, :, s].sum()) >= min_tumor_voxels:
            positive_slice_idxs.append(s)
        else:
            empty_slice_idxs.append(s)

    rng.shuffle(empty_slice_idxs)
    n_keep_empty = int(round(len(empty_slice_idxs) * empty_ratio))
    chosen_empty = empty_slice_idxs[:n_keep_empty]
    dropped_empty = len(empty_slice_idxs) - n_keep_empty

    kept_positive = 0
    kept_empty = 0
    for s in positive_slice_idxs + chosen_empty:
        chans = []
        for k in channel_keys:
            slc = _normalize_slice(mod_vols[k][:, :, s])
            slc = cv2.resize(slc, (image_size, image_size))
            chans.append(slc)
        # Stack into RGB; cv2 wants BGR for imwrite so reverse before writing.
        rgb = np.stack(chans, axis=-1)  # H, W, 3 (R=T1c, G=T2, B=FLAIR by default)
        mask = (seg_bin[:, :, s] > 0).astype(np.uint8) * 255
        mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

        stem = f'{patient_dir.name}_s{s:03d}'
        cv2.imwrite(str(images_out / f'{stem}.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(masks_out / f'{stem}.png'), mask)
        if s in positive_slice_idxs:
            kept_positive += 1
        else:
            kept_empty += 1

    return kept_positive, kept_empty, dropped_empty


def split_patients(patients: list[Path], ratios: tuple[float, float, float], seed: int) -> dict[str, list[Path]]:
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f'Split ratios must sum to 1.0 (got {ratios})')
    rng = random.Random(seed)
    shuffled = patients[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    n_test = n - n_train - n_val
    if n_test <= 0:
        raise ValueError(f'Split produced no test patients (n={n})')
    return {'train': shuffled[:n_train], 'val': shuffled[n_train:n_train + n_val], 'test': shuffled[n_train + n_val:]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='data_sources',
                        help='Source root; the script searches recursively for BraTS patient folders.')
    parser.add_argument('--output', default='dataset_brats')
    parser.add_argument('--image_size', type=int, default=192)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train', type=float, default=0.70)
    parser.add_argument('--val', type=float, default=0.15)
    parser.add_argument('--test', type=float, default=0.15)
    parser.add_argument('--empty_ratio', type=float, default=0.15,
                        help='Fraction of tumor-empty slices to keep per patient (default 15%%).')
    parser.add_argument('--min_tumor_voxels', type=int, default=50,
                        help='Minimum tumor voxels in a slice to count as positive.')
    parser.add_argument('--channels', nargs=3, default=['t1ce', 't2', 'flair'],
                        choices=list(MODALITY_SUFFIXES.keys()),
                        help='Three modalities to stack as RGB channels. Default: T1c/T2/FLAIR.')
    parser.add_argument('--max_patients', type=int, default=0,
                        help='Cap number of patients (debugging). 0 = use all.')
    parser.add_argument('--clean', action='store_true')
    args = parser.parse_args()

    source = Path(args.source)
    patients = _find_patient_dirs(source)
    if not patients:
        raise FileNotFoundError(
            f'No BraTS patient folders found under {source}. Did the Kaggle '
            'download finish? Look for a folder named MICCAI_BraTS2020_TrainingData/.'
        )
    if args.max_patients > 0:
        patients = patients[: args.max_patients]
    print(f'[info] Found {len(patients)} patient folders')

    split = split_patients(patients, (args.train, args.val, args.test), args.seed)
    print(f'[info] Split (patients): train={len(split["train"])} val={len(split["val"])} test={len(split["test"])}')

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    summary = {
        'image_size': args.image_size,
        'channels': args.channels,
        'empty_ratio': args.empty_ratio,
        'min_tumor_voxels': args.min_tumor_voxels,
        'splits': {},
    }
    rng = random.Random(args.seed)
    for split_name, pdirs in split.items():
        images_out = output / split_name / 'images'
        masks_out = output / split_name / 'masks'
        if args.clean:
            for sub in (images_out, masks_out):
                if sub.exists():
                    shutil.rmtree(sub)
        images_out.mkdir(parents=True, exist_ok=True)
        masks_out.mkdir(parents=True, exist_ok=True)

        total_pos = total_empty = total_dropped = 0
        for pdir in tqdm(pdirs, desc=f'{split_name} ({len(pdirs)} pts)'):
            kp, ke, kd = process_patient(
                pdir, images_out, masks_out,
                image_size=args.image_size,
                min_tumor_voxels=args.min_tumor_voxels,
                empty_ratio=args.empty_ratio,
                channel_keys=tuple(args.channels),
                rng=rng,
            )
            total_pos += kp
            total_empty += ke
            total_dropped += kd

        summary['splits'][split_name] = {
            'n_patients': len(pdirs),
            'kept_positive_slices': total_pos,
            'kept_empty_slices': total_empty,
            'dropped_empty_slices': total_dropped,
            'patient_ids': [p.name for p in pdirs],
        }
        print(f'  {split_name}: positive={total_pos}  empty_kept={total_empty}  empty_dropped={total_dropped}')

    summary_path = output / 'brats_split_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'\n[done] Wrote {summary_path}')


if __name__ == '__main__':
    main()
