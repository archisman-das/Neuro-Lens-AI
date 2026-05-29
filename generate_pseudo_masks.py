"""Generate weakly-supervised pseudo-masks for the Kaggle Brain Tumor MRI dataset.

The dataset that ships with this repo (dataset_real/{train,val,test}/{tumor,no_tumor})
is a classification dataset with no pixel-level ground truth. To train a U-Net
end-to-end, we synthesise pseudo-masks per image:

  - tumor images: brain region via a wide intensity gate, then high-intensity
    contrast-enhanced tumor region via Otsu on pixels inside the brain mask.
    A morphological cleanup (open + close + largest connected component) keeps
    only the most prominent bright cluster, which is the contrast-enhanced
    tumor region in T1-weighted MRI. This is a heuristic, NOT a radiologist
    label — see the disclaimer printed at the end of training.
  - no_tumor images: empty mask (all zeros).

Output layout (paired with the images):

  dataset_real/<split>/images/  <- copy of the original RGB MRI as PNG
  dataset_real/<split>/masks/   <- 0/255 grayscale mask PNG, same basename
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def _brain_mask(gray: np.ndarray) -> np.ndarray:
    """Coarse brain-region mask: drop pure-black skull/background pixels."""
    _, m = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    return m


def _tumor_mask_from_gray(gray: np.ndarray) -> np.ndarray:
    """Heuristic tumor mask: Otsu on brain-region pixels, keep top blob."""
    brain = _brain_mask(gray)
    inside = cv2.bitwise_and(gray, gray, mask=brain)
    nz = inside[brain > 0]
    if nz.size == 0:
        return np.zeros_like(gray)
    # Otsu over the brain region's intensities (rather than the whole image so
    # the threshold isn't dragged down by background pixels).
    _, otsu = cv2.threshold(nz.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = float(np.median(nz[otsu.ravel() > 0])) if (otsu > 0).any() else float(np.median(nz))
    high = (inside >= thresh).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    high = cv2.morphologyEx(high, cv2.MORPH_OPEN, kernel, iterations=1)
    high = cv2.morphologyEx(high, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Keep only the largest connected component as the tumor region.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(high, connectivity=8)
    if num_labels <= 1:
        return high
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    out = np.where(labels == best, 255, 0).astype(np.uint8)
    return out


def process_split(split_dir: Path, output_dir: Path, image_size: int) -> dict:
    """Return per-split stats."""
    tumor_src = split_dir / 'tumor'
    no_tumor_src = split_dir / 'no_tumor'
    images_out = output_dir / 'images'
    masks_out = output_dir / 'masks'
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    counts = {'tumor': 0, 'no_tumor': 0, 'tumor_with_mask': 0, 'empty_mask_skipped': 0}

    def _process(path: Path, has_tumor: bool):
        img = cv2.imread(str(path))
        if img is None:
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (image_size, image_size))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if has_tumor:
            mask = _tumor_mask_from_gray(gray)
            if int(mask.sum()) == 0:
                counts['empty_mask_skipped'] += 1
                return  # heuristic failed - skip rather than train on a wrong label
            counts['tumor_with_mask'] += 1
        else:
            mask = np.zeros_like(gray)

        stem = path.stem
        out_img_path = images_out / f'{stem}.png'
        out_mask_path = masks_out / f'{stem}.png'
        cv2.imwrite(str(out_img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_mask_path), mask)
        counts['tumor' if has_tumor else 'no_tumor'] += 1

    if tumor_src.exists():
        for p in tqdm(sorted([*tumor_src.glob('*.png'), *tumor_src.glob('*.jpg'), *tumor_src.glob('*.jpeg')]),
                      desc=f'{split_dir.name}/tumor'):
            _process(p, has_tumor=True)
    if no_tumor_src.exists():
        for p in tqdm(sorted([*no_tumor_src.glob('*.png'), *no_tumor_src.glob('*.jpg'), *no_tumor_src.glob('*.jpeg')]),
                      desc=f'{split_dir.name}/no_tumor'):
            _process(p, has_tumor=False)

    return counts


def main():
    parser = argparse.ArgumentParser(description='Generate pseudo-masks for U-Net training.')
    parser.add_argument('--source', default='dataset_real',
                        help='Source dataset root containing {train,val,test}/{tumor,no_tumor}.')
    parser.add_argument('--output', default='dataset_real',
                        help='Output dataset root. images/ and masks/ subdirs will be created per split.')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--clean', action='store_true',
                        help='Wipe existing images/ and masks/ subdirs before generating.')
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    if not source.exists():
        raise FileNotFoundError(f'Source dataset folder not found: {source}')

    all_stats = {}
    for split in ['train', 'val', 'test']:
        split_dir = source / split
        if not split_dir.exists():
            continue
        out_split_dir = output / split
        if args.clean:
            for sub in ('images', 'masks'):
                p = out_split_dir / sub
                if p.exists():
                    shutil.rmtree(p)
        all_stats[split] = process_split(split_dir, out_split_dir, args.image_size)

    summary_path = output / 'pseudo_masks_summary.json'
    with summary_path.open('w', encoding='utf-8') as fh:
        json.dump(all_stats, fh, indent=2)
    print('\nPseudo-mask generation complete.')
    print(json.dumps(all_stats, indent=2))
    print(f'Summary written to {summary_path}')
    print(
        '\nNOTE: These are weakly-supervised pseudo-masks derived from intensity '
        'thresholding, not radiologist annotations. They are usable for training '
        'a U-Net to learn an intensity-based tumor proxy and for demoing the '
        'segmentation pipeline, but they should NOT be treated as ground truth '
        'for clinical evaluation.'
    )


if __name__ == '__main__':
    main()
