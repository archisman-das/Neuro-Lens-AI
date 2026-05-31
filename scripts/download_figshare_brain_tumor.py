"""Download + extract the Cheng et al. 2015 brain tumor dataset (Figshare).

Why this matters for NeuroLens
------------------------------
v5 has a 75% false-negative rate on the Kaggle Brain Tumor MRI test
samples because v5's training set used Kaggle for the negative class
only - it never saw a Kaggle *positive* tumor scan. Kaggle is a
classification dataset (no ground-truth segmentation masks), so we
can't directly use Kaggle positives to retrain.

The Cheng et al. 2015 dataset (Figshare 1512427) fills this gap with
3,064 T1-contrast-enhanced MRI scans WITH expert hand-drawn tumor
segmentation masks. Three classes:
  - meningioma (label=1):   708 scans
  - glioma (label=2):      1,426 scans
  - pituitary (label=3):     930 scans
This is the same distribution as the Kaggle Brain Tumor MRI dataset
(meningioma / glioma / pituitary), so adding these to v5's training
should fix the Kaggle-distribution FN bug.

Output layout
-------------
data_sources/figshare_brain_tumor/  raw .mat files (extracted from .zips)
figshare_processed/{train,val,test}/{images,masks}/
    figshare_{label}_{idx:05d}.png   image (512x512 uint8, RGB-triplicated)
    figshare_{label}_{idx:05d}.png   mask  (512x512 uint8, 0=bg / 255=tumor)
Train/val/test split: 80/10/10 by index, deterministic.

The .mat files are MATLAB v7.3 format (HDF5-backed). We read them via
h5py because scipy.io.loadmat can't handle v7.3.

Usage
-----
  python scripts/download_figshare_brain_tumor.py
    --out_raw data_sources/figshare_brain_tumor
    --out_processed figshare_processed
    [--skip_download]   only do the .mat -> PNG conversion
    [--skip_convert]    only do the download
"""

from __future__ import annotations

import argparse
import io
import random
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Direct download URLs from Figshare item 1512427 (Cheng et al. 2015).
# Each zip is ~200 MB; total ~880 MB. Anonymous download supported.
FIGSHARE_URLS = [
    # IDs confirmed via Figshare API v2 (article 1512427). Hardcoded names
    # are illustrative; the .mat files inside are numerically indexed
    # 1.mat..3064.mat so zip naming doesn't affect downstream conversion.
    ("brainTumorDataPublic_1-766.zip",     "https://ndownloader.figshare.com/files/3381290"),
    ("brainTumorDataPublic_767-1532.zip",  "https://ndownloader.figshare.com/files/3381296"),
    ("brainTumorDataPublic_1533-2298.zip", "https://ndownloader.figshare.com/files/3381293"),
    ("brainTumorDataPublic_2299-3064.zip", "https://ndownloader.figshare.com/files/3381302"),
]

LABEL_NAMES = {1: "meningioma", 2: "glioma", 3: "pituitary"}


# -----------------------------------------------------------------------
# Download + extract
# -----------------------------------------------------------------------

def _download(url: str, dest: Path, chunk_size: int = 1 << 20) -> Path:
    """Streaming download with progress prints every 50 MB."""
    import requests
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"  [skip] {dest.name} already present ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [get]  {dest.name} from {url}")
    t0 = time.time()
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        downloaded = 0
        last_print = 0
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded - last_print > 50_000_000 or downloaded == total:
                    pct = (100 * downloaded / total) if total else 0
                    print(f"         {downloaded / 1e6:6.0f} MB ({pct:5.1f}%)  {time.time() - t0:.0f}s")
                    last_print = downloaded
    tmp.rename(dest)
    print(f"  [ok]   {dest.name}  {dest.stat().st_size / 1e6:.0f} MB in {time.time() - t0:.0f}s")
    return dest


def _extract_zip(zip_path: Path, out_dir: Path) -> int:
    """Extract .mat files from zip to out_dir. Returns count extracted."""
    n = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.endswith(".mat"):
                target = out_dir / Path(name).name
                if target.exists():
                    continue
                with z.open(name) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                n += 1
    return n


# -----------------------------------------------------------------------
# .mat -> PNG conversion
# -----------------------------------------------------------------------

def _read_mat(mat_path: Path) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return (image[H,W] uint8, mask[H,W] uint8 {0,255}, label int)."""
    import h5py
    with h5py.File(mat_path, "r") as f:
        cj = f["cjdata"]
        # image: stored as uint16 typically
        img = np.array(cj["image"]).astype(np.float32)
        if img.ndim == 2:
            pass  # H x W
        elif img.ndim == 3 and img.shape[0] == 1:
            img = img[0]
        # Normalise to [0, 255] uint8 with robust min-max scaling.
        lo, hi = float(img.min()), float(img.max())
        if hi > lo:
            img = (img - lo) / (hi - lo) * 255.0
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)

        mask = np.array(cj["tumorMask"]).astype(np.uint8)
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        mask_u8 = (mask > 0).astype(np.uint8) * 255

        # cjdata.label is stored as a 1x1 dataset of uint16
        label_arr = np.array(cj["label"])
        label = int(label_arr.flatten()[0])
    return img_u8, mask_u8, label


def _convert_all(raw_dir: Path, processed_dir: Path,
                 splits: Tuple[float, float, float] = (0.80, 0.10, 0.10),
                 seed: int = 42) -> dict:
    from PIL import Image
    mats = sorted(raw_dir.glob("*.mat"))
    print(f"[convert] {len(mats)} .mat files in {raw_dir}")
    rng = random.Random(seed)
    indices = list(range(len(mats)))
    rng.shuffle(indices)

    n = len(mats)
    n_train = int(splits[0] * n)
    n_val = int(splits[1] * n)
    split_map = {}
    for rank, idx in enumerate(indices):
        if rank < n_train:
            split_map[idx] = "train"
        elif rank < n_train + n_val:
            split_map[idx] = "val"
        else:
            split_map[idx] = "test"

    for split in ("train", "val", "test"):
        (processed_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (processed_dir / split / "masks").mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0,
              "by_class": {"meningioma": 0, "glioma": 0, "pituitary": 0},
              "errors": []}

    t0 = time.time()
    for idx, mp in enumerate(mats):
        try:
            img_u8, mask_u8, label = _read_mat(mp)
            cls_name = LABEL_NAMES.get(label, f"label{label}")
            counts["by_class"][cls_name] = counts["by_class"].get(cls_name, 0) + 1
            split = split_map[idx]
            counts[split] += 1
            # Triplicate to RGB to match the rest of the dataset_v5 layout.
            img_rgb = np.stack([img_u8] * 3, axis=-1)
            stem = f"figshare_{cls_name}_{int(mp.stem):05d}" if mp.stem.isdigit() else f"figshare_{cls_name}_{idx:05d}"
            Image.fromarray(img_rgb).save(processed_dir / split / "images" / f"{stem}.png")
            Image.fromarray(mask_u8).save(processed_dir / split / "masks" / f"{stem}.png")
        except Exception as exc:
            counts["errors"].append(f"{mp.name}: {type(exc).__name__}: {exc}")
        if (idx + 1) % 250 == 0:
            print(f"  [{idx + 1}/{len(mats)}] {time.time() - t0:.0f}s elapsed")
    print(f"[done] {time.time() - t0:.0f}s. counts={counts}")
    return counts


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_raw", default="data_sources/figshare_brain_tumor",
                    type=Path, help="where to put raw .mat files")
    ap.add_argument("--out_processed", default="figshare_processed",
                    type=Path, help="where to put PNG-converted dataset")
    ap.add_argument("--skip_download", action="store_true")
    ap.add_argument("--skip_convert", action="store_true")
    args = ap.parse_args()

    args.out_raw.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        print("=== [1/2] downloading 4 zips from Figshare (~880 MB total) ===")
        zip_dir = args.out_raw / "zips"
        zip_dir.mkdir(parents=True, exist_ok=True)
        for fname, url in FIGSHARE_URLS:
            _download(url, zip_dir / fname)
        print("=== extracting .mat files ===")
        for fname, _ in FIGSHARE_URLS:
            n = _extract_zip(zip_dir / fname, args.out_raw)
            print(f"  [extract] {fname}: {n} .mat files")
        mats = list(args.out_raw.glob("*.mat"))
        print(f"  total .mat files: {len(mats)}")
        if len(mats) < 3000:
            print(f"  [warn] expected ~3064 .mat files, got {len(mats)}")

    if not args.skip_convert:
        print("=== [2/2] converting .mat -> PNG (image + mask) with train/val/test split ===")
        _convert_all(args.out_raw, args.out_processed)
        print(f"  output: {args.out_processed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
