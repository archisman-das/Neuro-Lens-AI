"""Build dataset_v8: v5 dataset + Figshare proper-labeled Kaggle-distribution tumors.

Why v8 over v5
--------------
v5 = BraTS positives + LGG positives + Kaggle (no_tumor class only) negatives.
On the Kaggle Brain Tumor MRI test samples (glioma / meningioma / pituitary -
all positive), v5 hits a 75% false-negative rate. Root cause: v5 saw zero
Kaggle-distribution *positive* tumor scans in training. Kaggle is a
classification dataset without segmentation masks, so we couldn't directly
add Kaggle positives to v5's training.

The Cheng et al. 2015 dataset (Figshare 1512427) closes this gap with 3,064
T1-contrast MRI scans WITH expert hand-drawn tumor segmentation masks of
exactly the three classes that the Kaggle dataset uses: meningioma (708),
glioma (1,426), pituitary (930). Same distribution, proper labels, no
pseudo-labeling.

Composition
-----------
dataset_v8/
  train/  val/  test/
    images/  (RGB at 256x256)
    masks/   (single-channel PNG, 0=bg / 255=tumor)

Sources concatenated:
  - dataset_v5/{split}/{images,masks}/*       (everything v5 had)
  - figshare_processed/{split}/{images,masks}/* (Figshare proper-labeled tumors)

Net counts (expected):
  train: ~22,525 (v5) + ~2,451 (Figshare 80%) = ~24,976
  val:   ~4,604  (v5) + ~306   (Figshare 10%) = ~4,910
  test:  ~4,651  (v5) + ~307   (Figshare 10%) = ~4,958

Usage
-----
After scripts/download_figshare_brain_tumor.py finishes:

  python prepare_v8_dataset.py \
    --v5_dir dataset_v5 --figshare_dir figshare_processed \
    --out dataset_v8 --image-size 256
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

SPLITS = ("train", "val", "test")


def _copy_resize(src_img: Path, src_msk: Path, dst_img: Path, dst_msk: Path,
                 image_size: int, overwrite: bool = False) -> bool:
    """Copy + resize one (image, mask) pair to dst paths. Skip if exists."""
    if dst_img.exists() and dst_msk.exists() and not overwrite:
        return False
    if not src_img.exists() or not src_msk.exists():
        return False
    img = Image.open(src_img).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    msk = Image.open(src_msk).convert("L").resize((image_size, image_size), Image.NEAREST)
    # Threshold mask to {0, 255} to drop any anti-aliasing artifacts from resize.
    msk_arr = np.array(msk)
    msk = Image.fromarray((msk_arr > 127).astype(np.uint8) * 255)
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_msk.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst_img)
    msk.save(dst_msk)
    return True


def _copy_split(src_root: Path, dst_root: Path, split: str, image_size: int,
                 tag: str, max_copy: int | None = None) -> int:
    """Copy all (image, mask) pairs from src_root/split into dst_root/split."""
    src_img_dir = src_root / split / "images"
    src_msk_dir = src_root / split / "masks"
    dst_img_dir = dst_root / split / "images"
    dst_msk_dir = dst_root / split / "masks"
    if not src_img_dir.exists():
        print(f"  [skip] {src_root}/{split}/images does not exist")
        return 0
    n = 0
    files = sorted(src_img_dir.iterdir())
    if max_copy is not None:
        files = files[:max_copy]
    t0 = time.time()
    for f in files:
        msk_path = src_msk_dir / f.name
        if not msk_path.exists():
            # Try .png if image was .jpg, etc.
            for ext in (".png", ".jpg", ".jpeg"):
                cand = src_msk_dir / (f.stem + ext)
                if cand.exists():
                    msk_path = cand
                    break
        if _copy_resize(f, msk_path, dst_img_dir / f.name, dst_msk_dir / f.name, image_size):
            n += 1
        if n > 0 and n % 1000 == 0:
            print(f"    [{tag}/{split}] {n} copied  ({time.time() - t0:.0f}s)")
    print(f"  [{tag}/{split}] {n} files copied in {time.time() - t0:.0f}s")
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v5_dir", default="dataset_v5", type=Path,
                    help="existing v5 dataset (BraTS + LGG + Kaggle negatives)")
    ap.add_argument("--figshare_dir", default="figshare_processed", type=Path,
                    help="output of scripts/download_figshare_brain_tumor.py")
    ap.add_argument("--out", default="dataset_v8", type=Path)
    ap.add_argument("--image-size", dest="image_size", type=int, default=256)
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-copy files even if destination exists")
    args = ap.parse_args()

    assert args.v5_dir.exists(), f"v5 dataset not found: {args.v5_dir}"
    assert args.figshare_dir.exists(), (
        f"Figshare processed dir not found: {args.figshare_dir}. "
        f"Run scripts/download_figshare_brain_tumor.py first."
    )

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Building {args.out} from:")
    print(f"  v5:       {args.v5_dir}")
    print(f"  figshare: {args.figshare_dir}")
    print(f"  size:     {args.image_size}x{args.image_size}")

    totals = {}
    for split in SPLITS:
        print(f"\n--- split: {split} ---")
        v5_n = _copy_split(args.v5_dir, args.out, split, args.image_size, tag="v5")
        fig_n = _copy_split(args.figshare_dir, args.out, split, args.image_size, tag="figshare")
        totals[split] = v5_n + fig_n
        print(f"  -> {split}: v5={v5_n}, figshare={fig_n}, total={totals[split]}")

    print("\n=== final ===")
    for split in SPLITS:
        ni = len(list((args.out / split / "images").iterdir()))
        nm = len(list((args.out / split / "masks").iterdir()))
        print(f"  {split}: images={ni}, masks={nm}")
    print(f"\ndataset_v8 ready at {args.out}")
    print("Next: python src/train_segmentation_v7.py --data_dir dataset_v8 "
          "--output_dir segmentation_artifacts/attention_unet_v8 --batch_size 2 "
          "--epochs 60 --image_size 384")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
