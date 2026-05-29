"""Prepare the LGG MRI Segmentation dataset for U-Net training.

Source layout (after `kaggle datasets download -d mateuszbuda/lgg-mri-segmentation --unzip`):

    data_sources/lgg-mri-segmentation/kaggle_3m/
        TCGA_<id>/
            TCGA_<id>_<slice>.tif        (3-channel pre/FLAIR/post image)
            TCGA_<id>_<slice>_mask.tif   (binary tumor mask, 0/255)
        ...

We split the data **by patient** (not by image) so that no two slices from the
same patient appear in different splits. This is the standard LGG protocol and
prevents the obvious leakage that destroys medical-image generalisation
estimates. Default ratios: 70% train / 15% val / 15% test by patient count.

Output (drop-in compatible with src/train_segmentation_torch.py):

    dataset_lgg/<split>/images/<TCGA_id>_<slice>.png
    dataset_lgg/<split>/masks/<TCGA_id>_<slice>.png
    dataset_lgg/lgg_split_summary.json   (per-split patient + slice counts)

We also **skip the all-zero-mask slices that come from the head/tail of each
patient series**, because the LGG dataset is mostly empty masks (the FLAIR
hyperintensity only spans a few central slices per patient). Keeping all empty
slices makes the loss collapse to "predict everything zero". The `--keep_empty`
flag overrides that if you want the unfiltered split.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def collect_patient_files(source_root: Path) -> dict[str, list[tuple[Path, Path]]]:
    """Return {patient_id: [(image_path, mask_path), ...]} for every patient."""
    candidates = []
    for sub in source_root.rglob('kaggle_3m'):
        if sub.is_dir():
            candidates.append(sub)
    if not candidates:
        # Some Kaggle bundles drop everything into the dataset root directly.
        candidates = [source_root]

    by_patient: dict[str, list[tuple[Path, Path]]] = {}
    for root in candidates:
        for patient_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            pid = patient_dir.name
            if not pid.startswith('TCGA'):
                continue
            pairs = []
            for img_path in sorted(patient_dir.glob('*.tif')):
                if img_path.stem.endswith('_mask'):
                    continue
                mask_path = patient_dir / f'{img_path.stem}_mask.tif'
                if mask_path.exists():
                    pairs.append((img_path, mask_path))
            if pairs:
                by_patient[pid] = pairs
    return by_patient


def split_patients(patients: list[str], ratios: tuple[float, float, float], seed: int) -> dict[str, list[str]]:
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
        raise ValueError(f'Split produced zero test patients (n={n}, ratios={ratios})')
    return {
        'train': shuffled[:n_train],
        'val': shuffled[n_train:n_train + n_val],
        'test': shuffled[n_train + n_val:],
    }


def write_pair(img_src: Path, mask_src: Path, images_out: Path, masks_out: Path, image_size: int) -> bool:
    """Read .tif image+mask, resize to image_size, write as PNGs. Returns True if
    the mask has any positive pixels (used by the empty-skip filter)."""
    img = cv2.imread(str(img_src), cv2.IMREAD_COLOR)
    if img is None:
        return False
    if img.shape[0] != image_size or img.shape[1] != image_size:
        img = cv2.resize(img, (image_size, image_size))

    mask = cv2.imread(str(mask_src), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False
    if mask.shape[0] != image_size or mask.shape[1] != image_size:
        mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    mask = ((mask > 127).astype(np.uint8) * 255)

    stem = img_src.stem  # TCGA_<id>_<slice>
    cv2.imwrite(str(images_out / f'{stem}.png'), img)
    cv2.imwrite(str(masks_out / f'{stem}.png'), mask)
    return bool(mask.any())


def main():
    parser = argparse.ArgumentParser(description='Prepare LGG MRI Segmentation dataset.')
    parser.add_argument('--source', default='data_sources/lgg-mri-segmentation',
                        help='Directory containing the unzipped Kaggle dataset (looks for kaggle_3m/ recursively).')
    parser.add_argument('--output', default='dataset_lgg')
    parser.add_argument('--image_size', type=int, default=192,
                        help='Resize target. Matches the U-Net training default.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train', type=float, default=0.70)
    parser.add_argument('--val', type=float, default=0.15)
    parser.add_argument('--test', type=float, default=0.15)
    parser.add_argument('--keep_empty', action='store_true',
                        help='Keep slices whose mask is fully zero. Default behaviour drops them, '
                             'because >90%% of LGG slices have empty masks and they swamp training.')
    parser.add_argument('--empty_ratio', type=float, default=0.2,
                        help='If --keep_empty is NOT set, still keep this fraction of empty slices '
                             '(per patient) as negatives. Helps the model say "no tumor here" too.')
    parser.add_argument('--clean', action='store_true',
                        help='Wipe existing dataset_lgg/{train,val,test} before writing.')
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(
            f'Source directory not found: {source}. Run:\n'
            f'  python -m kaggle datasets download -d mateuszbuda/lgg-mri-segmentation '
            f'-p {source.parent} --unzip'
        )

    print(f'[info] Scanning {source} for patient folders...')
    by_patient = collect_patient_files(source)
    if not by_patient:
        raise RuntimeError('No patient folders matching TCGA_* with paired image+mask .tifs were found.')
    patients = sorted(by_patient.keys())
    total_slices = sum(len(v) for v in by_patient.values())
    print(f'[info] Found {len(patients)} patients, {total_slices} total slices.')

    split = split_patients(patients, (args.train, args.val, args.test), args.seed)
    print(f'[info] Split counts (patients): train={len(split["train"])} '
          f'val={len(split["val"])} test={len(split["test"])}')

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    summary = {'image_size': args.image_size, 'keep_empty': args.keep_empty, 'empty_ratio': args.empty_ratio, 'patients': {}}

    rng = random.Random(args.seed)
    for split_name, pids in split.items():
        images_out = output / split_name / 'images'
        masks_out = output / split_name / 'masks'
        if args.clean:
            for sub in (images_out, masks_out):
                if sub.exists():
                    shutil.rmtree(sub)
        images_out.mkdir(parents=True, exist_ok=True)
        masks_out.mkdir(parents=True, exist_ok=True)

        kept_positive = kept_empty = dropped_empty = 0
        for pid in tqdm(pids, desc=f'{split_name} ({len(pids)} pts)'):
            pairs = by_patient[pid]
            # Bucket by whether the mask has any tumor pixels so we can keep
            # only --empty_ratio of negatives per patient.
            positive_pairs = []
            empty_pairs = []
            for img_src, mask_src in pairs:
                m = cv2.imread(str(mask_src), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    continue
                if m.any():
                    positive_pairs.append((img_src, mask_src))
                else:
                    empty_pairs.append((img_src, mask_src))
            if not args.keep_empty:
                n_keep_empty = int(round(len(empty_pairs) * args.empty_ratio))
                rng.shuffle(empty_pairs)
                empty_kept = empty_pairs[:n_keep_empty]
                dropped_empty += len(empty_pairs) - n_keep_empty
            else:
                empty_kept = empty_pairs

            for img_src, mask_src in positive_pairs:
                ok = write_pair(img_src, mask_src, images_out, masks_out, args.image_size)
                if ok:
                    kept_positive += 1
            for img_src, mask_src in empty_kept:
                write_pair(img_src, mask_src, images_out, masks_out, args.image_size)
                kept_empty += 1

        summary['patients'][split_name] = {
            'n_patients': len(pids),
            'kept_positive_slices': kept_positive,
            'kept_empty_slices': kept_empty,
            'dropped_empty_slices': dropped_empty,
            'patient_ids': pids,
        }
        print(f'  {split_name}: kept positive={kept_positive}  kept empty={kept_empty}  '
              f'dropped empty={dropped_empty}')

    summary_path = output / 'lgg_split_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'\n[done] Wrote {summary_path}')


if __name__ == '__main__':
    main()
