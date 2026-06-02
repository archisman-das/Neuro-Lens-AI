"""Expand OOD-healthy test cohort with IXI2D slices from HuggingFace.

The iamkzntsv/IXI2D dataset (28,275 axial 2D slices from 600 healthy IXI
subjects, 133 MB, skull-stripped + fsaverage-registered) is genuinely
OOD relative to everything we've trained on:
  - v8 segmenter: trained on BraTS 2020 T1c + LGG kaggle_3m + Figshare +
    Kaggle 4-class. Never touched IXI.
  - v9b JEPA: trained on dataset_v8 augmented with OpenNeuro coronal-T1.
    Never touched IXI.

Pulling a stratified sample of N healthy IXI slices, saving to
samples/ood/healthy_ixi2d/. Caveat 3 addressed: gives us a larger OOD
healthy cohort to estimate FPR more reliably than N=12.
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'samples' / 'ood' / 'healthy_ixi2d'
OUT.mkdir(parents=True, exist_ok=True)

# How many slices to pull. 100 gives us 1pp resolution on FPR estimates
# (vs the previous 12 samples where each false positive = 8.3pp).
N_SAMPLES = 100


def main():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit('pip install huggingface_hub')

    print(f'[init] target = {N_SAMPLES} IXI2D healthy slices -> {OUT}')

    # The dataset ships as parquet (imagefolder split). Easiest path:
    # use the `datasets` library to load + iterate, OR pull individual
    # files from the LFS-hosted train split. Try the datasets library
    # first; fall back to manual parquet download.
    try:
        from datasets import load_dataset
        print('[1/3] streaming IXI2D from HuggingFace ...')
        ds = load_dataset('iamkzntsv/IXI2D', split='train', streaming=True)
        # Take every N-th slice to spread across subjects (the train split
        # is ordered by subject). 25,400 train rows, take ~100 means stride
        # = 254.
        stride = max(1, 25400 // N_SAMPLES)
        print(f'   stride={stride} -> ~{N_SAMPLES} samples spread across subjects')
        saved = 0
        t0 = time.perf_counter()
        for i, ex in enumerate(ds):
            if i % stride != 0:
                continue
            img = ex.get('image')
            if img is None:
                continue
            # ex['image'] is a PIL Image (200x200 grayscale per the dataset card)
            fname = f'ixi2d_{i:05d}.png'
            img.save(OUT / fname)
            saved += 1
            if saved >= N_SAMPLES:
                break
            if saved % 20 == 0:
                print(f'   [{saved}/{N_SAMPLES}] elapsed={time.perf_counter()-t0:.0f}s')
        print(f'[done] saved {saved} slices in {time.perf_counter()-t0:.0f}s')
    except Exception as exc:
        print(f'[fail] {type(exc).__name__}: {exc}')
        print('Fallback path: manually browse https://huggingface.co/datasets/iamkzntsv/IXI2D/tree/main')
        sys.exit(1)


if __name__ == '__main__':
    main()
