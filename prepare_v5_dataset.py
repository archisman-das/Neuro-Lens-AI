"""Build dataset_v5: tumor-positive + healthy-brain unified dataset for v5
segmentation training.

Why a v5 dataset:
  - v3 / v4 were trained on BraTS + LGG, which are tumor-positive patient
    cohorts. Every training sample contained a tumor, so the model learned
    'always emit some mask'. On a healthy brain it produces a false-positive
    region tracing normal anatomy.
  - v5 includes healthy brains with empty masks so the model learns 'no
    tumor here, output nothing'. Combined with the existing positive samples
    and a balanced 50/50 sampler at train time, this eliminates the
    positive bias.

Positive sources (have ground-truth or pseudo masks):
  - dataset_brats_t1c/   BraTS 2020 T1c channel triplicated (real masks).
  - dataset_lgg/         LGG MRI FLAIR (real masks).

Negative source (empty masks):
  - dataset_real/{train,val,test}/no_tumor/*.jpg  - Kaggle Brain Tumor MRI
    no-tumor split. Single-modality, single-slice, but real healthy brains
    in the right intensity / contrast range for our use-case.

Output layout (compatible with src/train_segmentation_v5.py):
  dataset_v5/
    train/  val/  test/
      images/         RGB JPEG/PNG at 256x256
      masks/          single-channel PNG 0/255

Patient-level splits where applicable. For Kaggle the source is already
split into train/val/test - we preserve those.

Usage:
  python prepare_v5_dataset.py --output dataset_v5 \\
    --positive-brats dataset_brats_t1c --positive-lgg dataset_lgg \\
    --negative-kaggle dataset_real --image-size 256
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


SPLITS = ('train', 'val', 'test')


def _resize_pad(img: Image.Image, size: int) -> Image.Image:
    """Resize to (size, size). If grayscale, replicated to 3 channels."""
    img = img.convert('RGB').resize((size, size), Image.BILINEAR)
    return img


def _resize_mask(img: Image.Image, size: int) -> Image.Image:
    img = img.convert('L').resize((size, size), Image.NEAREST)
    return img


def _empty_mask(size: int) -> Image.Image:
    arr = np.zeros((size, size), dtype=np.uint8)
    return Image.fromarray(arr, mode='L')


def _copy_split(src_root: Path, dst_root: Path, split: str, size: int,
                  *, kind: str, counters: dict) -> None:
    """Copy one split's worth of images + masks to dst_root."""
    images_dir = src_root / split / 'images'
    masks_dir = src_root / split / 'masks'
    if not images_dir.exists():
        print(f'  [skip] {kind}/{split} - no images dir at {images_dir}')
        return
    dst_images = dst_root / split / 'images'
    dst_masks = dst_root / split / 'masks'
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_masks.mkdir(parents=True, exist_ok=True)

    n_copied = 0
    n_skipped = 0
    for img_path in sorted(images_dir.iterdir()):
        if not img_path.is_file():
            continue
        stem = img_path.stem
        mask_path = masks_dir / f'{stem}.png'
        if not mask_path.exists():
            mask_path = masks_dir / f'{stem}.jpg'
        if not mask_path.exists():
            n_skipped += 1
            continue
        try:
            img = Image.open(img_path)
            msk = Image.open(mask_path)
        except Exception as exc:
            print(f'  [warn] load failed for {img_path}: {exc}')
            n_skipped += 1
            continue
        out_name = f'{kind}_{stem}.png'
        _resize_pad(img, size).save(dst_images / out_name)
        _resize_mask(msk, size).save(dst_masks / out_name)
        n_copied += 1

    counters[f'positive_{kind}_{split}'] = n_copied
    print(f'  [positive:{kind}] {split}: {n_copied} copied, {n_skipped} skipped')


def _copy_negatives(neg_root: Path, dst_root: Path, splits: tuple, size: int,
                     counters: dict) -> None:
    """Pull no_tumor images from Kaggle Brain Tumor MRI's train/val/test
    splits and emit them with empty masks."""
    for split in splits:
        src = neg_root / split / 'no_tumor'
        if not src.exists():
            print(f'  [skip] negatives/{split} - no source at {src}')
            continue
        dst_images = dst_root / split / 'images'
        dst_masks = dst_root / split / 'masks'
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_masks.mkdir(parents=True, exist_ok=True)
        n_copied = 0
        empty = _empty_mask(size)
        for img_path in sorted(src.iterdir()):
            if not img_path.is_file():
                continue
            try:
                img = Image.open(img_path)
            except Exception as exc:
                print(f'  [warn] load failed for {img_path}: {exc}')
                continue
            stem = img_path.stem
            out_name = f'neg_kaggle_{stem}.png'
            _resize_pad(img, size).save(dst_images / out_name)
            empty.save(dst_masks / out_name)
            n_copied += 1
        counters[f'negative_kaggle_{split}'] = n_copied
        print(f'  [negative:kaggle] {split}: {n_copied} copied (empty masks)')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', default='dataset_v5')
    ap.add_argument('--positive-brats', default='dataset_brats_t1c',
                     help='Path to existing BraTS T1c dataset (with train/val/test/images,masks/).')
    ap.add_argument('--positive-lgg', default='dataset_lgg',
                     help='Path to existing LGG dataset.')
    ap.add_argument('--negative-kaggle', default='dataset_real',
                     help='Path to Kaggle Brain Tumor MRI; uses {train,val,test}/no_tumor/.')
    ap.add_argument('--image-size', type=int, default=256)
    ap.add_argument('--skip-brats', action='store_true')
    ap.add_argument('--skip-lgg', action='store_true')
    ap.add_argument('--skip-negatives', action='store_true')
    args = ap.parse_args()

    src_brats = Path(args.positive_brats)
    src_lgg = Path(args.positive_lgg)
    src_neg = Path(args.negative_kaggle)
    dst = Path(args.output)

    if dst.exists() and any(dst.iterdir()):
        print(f'[warn] {dst} already exists and is non-empty. New files will be added; existing are kept.')
    dst.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    counters: dict = {}

    if not args.skip_brats:
        if not src_brats.exists():
            print(f'[skip] positive_brats {src_brats} not found.')
        else:
            print(f'[positive:brats] reading from {src_brats}')
            for split in SPLITS:
                _copy_split(src_brats, dst, split, args.image_size,
                              kind='brats_t1c', counters=counters)

    if not args.skip_lgg:
        if not src_lgg.exists():
            print(f'[skip] positive_lgg {src_lgg} not found.')
        else:
            print(f'[positive:lgg] reading from {src_lgg}')
            for split in SPLITS:
                _copy_split(src_lgg, dst, split, args.image_size,
                              kind='lgg', counters=counters)

    if not args.skip_negatives:
        if not src_neg.exists():
            print(f'[skip] negative_kaggle {src_neg} not found.')
        else:
            print(f'[negative:kaggle] reading from {src_neg}')
            _copy_negatives(src_neg, dst, SPLITS, args.image_size, counters)

    print(f'\n=== prep done in {time.time() - t0:.1f}s ===')
    print('Sample counts:')
    pos_total = 0
    neg_total = 0
    for k, v in sorted(counters.items()):
        print(f'  {k:<35} {v}')
        if k.startswith('positive_'):
            pos_total += v
        elif k.startswith('negative_'):
            neg_total += v
    print(f'\n  total positive samples: {pos_total}')
    print(f'  total negative samples: {neg_total}')
    if pos_total and neg_total:
        ratio = neg_total / pos_total
        print(f'  negative / positive ratio: {ratio:.2f}')
        print(f'  -> the v5 trainer will use a balanced sampler (50/50 per batch)')


if __name__ == '__main__':
    main()
