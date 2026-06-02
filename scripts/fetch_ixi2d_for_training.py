"""Pull full IXI2D training pool (~25k healthy slices) into dataset_v8.

Phase 1 of the v9c data-expansion plan. Adds IXI2D's full training
split as no_tumor slices in dataset_v8/train/{images,masks}/.

EXCLUDES the 100 slices already used as held-out OOD test set
(samples/ood/healthy_ixi2d/) by filename, so there is zero leakage
between training pool and OOD evaluation.

Source: iamkzntsv/IXI2D (HuggingFace, 28,275 slices from 600 IXI healthy
subjects, skull-stripped + fsaverage-registered, MIT).
"""
from __future__ import annotations

import os
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / 'dataset_v8'
OOD_HELDOUT_DIR = ROOT / 'samples' / 'ood' / 'healthy_ixi2d'


def _held_out_basenames() -> set[str]:
    """Return the IXI2D base filenames already in the OOD test cohort —
    we strip the 'ixi2d_XXXX_' prefix and use the original IXI2D basename."""
    out = set()
    if not OOD_HELDOUT_DIR.exists():
        return out
    for p in OOD_HELDOUT_DIR.glob('*.jpg'):
        # Names look like 'ixi2d_0000_18927.jpg'; the IXI2D source name
        # is the trailing portion after the second underscore.
        parts = p.stem.split('_')
        if len(parts) >= 3:
            out.add(parts[-1])   # e.g. '18927'
    return out


def main():
    cached_zip = ROOT / 'samples' / 'ood' / '_zip_tmp_ixi' / 'data' / 'train.zip'
    cached_zip.parent.mkdir(parents=True, exist_ok=True)

    if not cached_zip.exists() or cached_zip.stat().st_size < 10_000_000:
        print('[1/3] downloading iamkzntsv/IXI2D train.zip ...')
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id='iamkzntsv/IXI2D', filename='data/train.zip',
            repo_type='dataset', local_dir=str(cached_zip.parent.parent),
        )
        cached_zip = Path(downloaded)
    print(f'   zip ready: {cached_zip} ({cached_zip.stat().st_size/1e6:.1f} MB)')

    held_out = _held_out_basenames()
    print(f'[2/3] excluding {len(held_out)} slices that are already in '
          f'samples/ood/healthy_ixi2d/ (OOD test set)')

    # Ensure target directories
    for split in ('train', 'val'):
        (DATASET / split / 'images').mkdir(parents=True, exist_ok=True)
        (DATASET / split / 'masks').mkdir(parents=True, exist_ok=True)

    # Extract all .jpeg files (excluding __MACOSX). Assign to train by
    # default; route every 10th to val to keep IXI proportionally in val.
    added_train = added_val = skipped = 0
    t0 = time.perf_counter()
    with zipfile.ZipFile(cached_zip) as zf:
        valid = sorted(n for n in zf.namelist()
                        if n.lower().endswith(('.png', '.jpg', '.jpeg'))
                        and not n.startswith('__MACOSX/'))
        print(f'   {len(valid)} IXI2D images in zip')
        for i, nm in enumerate(valid):
            base = os.path.basename(nm)
            stem = base.rsplit('.', 1)[0]
            if stem in held_out:
                skipped += 1
                continue
            split = 'val' if i % 10 == 0 else 'train'
            out_name = f'ixi2d_train_{stem}.png'
            img_path = DATASET / split / 'images' / out_name
            mask_path = DATASET / split / 'masks' / out_name
            if img_path.exists() and mask_path.exists():
                # idempotent
                continue
            data = zf.read(nm)
            # Decode -> re-encode as PNG; create all-zero mask of same size
            arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                continue
            cv2.imwrite(str(img_path), arr)
            cv2.imwrite(str(mask_path),
                        np.zeros(arr.shape[:2], dtype=np.uint8))
            if split == 'train':
                added_train += 1
            else:
                added_val += 1
        elapsed = time.perf_counter() - t0
    print(f'[3/3] added {added_train} to train / {added_val} to val  '
          f'(skipped {skipped} OOD held-outs)  in {elapsed:.0f}s')

    # Final tally
    print('\nNew dataset_v8 healthy-source coverage:')
    for split in ('train', 'val'):
        img_dir = DATASET / split / 'images'
        ixi = sum(1 for p in img_dir.glob('ixi2d_*.png'))
        oneuro = sum(1 for p in img_dir.glob('oneuro_*.png'))
        kaggle = sum(1 for p in img_dir.glob('neg_kaggle*.png'))
        total = sum(1 for _ in img_dir.glob('*.png'))
        print(f'  {split:5s}  total={total:5d}  '
              f'kaggle_neg={kaggle:5d}  '
              f'openneuro={oneuro:5d}  ixi2d={ixi:5d}')


if __name__ == '__main__':
    main()
