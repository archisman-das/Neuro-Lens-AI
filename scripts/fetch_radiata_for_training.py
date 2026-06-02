"""Pull healthy 2D slices from radiata-ai/brain-structure into dataset_v8.

Phase 1 (continued) of the v9c data-expansion plan. The radiata-ai dataset
aggregates 5 public studies (DLBS / IXI / NKI-RS / OASIS-1 / OASIS-2) as
T1-weighted MPRAGE NIfTI scans, all skull-stripped and MNI152-registered
at 113×137×113 / 1.5mm³. This script:

  1. Reads metadata.csv to pick a stratified sample of cognitively-normal
     subjects across sources (skipping IXI since iamkzntsv/IXI2D already
     covers it at 2D-slice resolution).
  2. Downloads each subject's NIfTI individually via hf_hub_download
     (~2 MB each, cached by HuggingFace under ~/.cache/huggingface).
  3. Extracts 5 mid-axial slices per scan, normalises with per-volume
     1-99 percentile windowing, and writes to dataset_v8/train/{images,
     masks} as 3-channel PNG + zero mask.

Default: 200 subjects each from NKI-RS / OASIS-1 / OASIS-2 / DLBS = 800
subjects × 5 slices = 4000 new healthy slices.

CLI:
  python scripts/fetch_radiata_for_training.py [--per_study N]
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / 'dataset_v8'

# Source studies to include. IXI excluded (already covered by IXI2D fetcher).
# Order also drives the stratification: deeper/larger sources first so a
# small per_study N still includes the well-curated DLBS/OASIS data.
TARGET_STUDIES = ['NKI-RS', 'OASIS-1', 'OASIS-2', 'DLBS']

# Axial slice indices in MNI 113-z space. Mid-brain region ~ z=40-80.
AXIAL_SLICE_INDICES = [45, 55, 65, 75, 85]


def _norm_uint8(slice2d: np.ndarray) -> np.ndarray:
    arr = slice2d.astype(np.float32)
    nz = arr[arr > 0]
    if nz.size == 0:
        return arr.astype(np.uint8)
    lo, hi = np.percentile(nz, (1, 99))
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    return arr.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per_study', type=int, default=200,
                     help='Subjects to sample per study (default 200)')
    ap.add_argument('--limit_total', type=int, default=10_000,
                     help='Hard cap on total slices to add (default 10000)')
    ap.add_argument('--val_fraction', type=float, default=0.10)
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit('pip install huggingface_hub')

    print('[1/4] fetching metadata.csv ...')
    mp = hf_hub_download(repo_id='radiata-ai/brain-structure',
                          filename='metadata.csv', repo_type='dataset')
    with open(mp, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    print(f'   {len(rows)} subjects in metadata')

    # Filter to healthy + in our target studies
    healthy = [r for r in rows
                if r['clinical_diagnosis'] == 'cognitively_normal'
                and r['study'] in TARGET_STUDIES]
    print(f'   {len(healthy)} healthy subjects in target studies')

    # Stratified sample
    selected: list[dict] = []
    for study in TARGET_STUDIES:
        pool = [r for r in healthy if r['study'] == study]
        pool = sorted(pool, key=lambda r: r['radiata_id'])  # deterministic
        selected.extend(pool[:args.per_study])
    print(f'[2/4] selected {len(selected)} subjects ({args.per_study} per study)')

    for split in ('train', 'val'):
        (DATASET / split / 'images').mkdir(parents=True, exist_ok=True)
        (DATASET / split / 'masks').mkdir(parents=True, exist_ok=True)

    n_added = n_failed = n_skipped = 0
    t0 = time.perf_counter()
    last_print = t0
    for i, sub in enumerate(selected):
        if n_added >= args.limit_total:
            print(f'   limit_total={args.limit_total} hit; stopping')
            break
        try:
            nii_path = hf_hub_download(repo_id='radiata-ai/brain-structure',
                                          filename=sub['t1_local_path'],
                                          repo_type='dataset')
            vol = nib.load(nii_path).get_fdata()
            # vol shape ~ (113, 137, 113); axial = axis 2
            for sl_idx in AXIAL_SLICE_INDICES:
                if sl_idx >= vol.shape[2]:
                    continue
                sl = vol[:, :, sl_idx]
                if (sl > 0).sum() < 200:   # skip near-empty edge slices
                    continue
                img_u8 = _norm_uint8(sl)
                img_resized = cv2.resize(img_u8, (192, 192),
                                          interpolation=cv2.INTER_LINEAR)
                img_rgb = np.stack([img_resized] * 3, axis=-1)
                split = ('val' if (n_added % int(1/args.val_fraction) == 0)
                          else 'train')
                fname = (f'radiata_{sub["study"]}_'
                         f'sub{sub["participant_id"]}_z{sl_idx:03d}.png')
                img_out = DATASET / split / 'images' / fname
                mask_out = DATASET / split / 'masks' / fname
                if img_out.exists() and mask_out.exists():
                    n_skipped += 1
                    continue
                cv2.imwrite(str(img_out),
                            cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(mask_out),
                            np.zeros((192, 192), dtype=np.uint8))
                n_added += 1
        except Exception as exc:
            n_failed += 1
            if n_failed <= 3:
                print(f'   [fail] {sub["t1_local_path"]}: '
                      f'{type(exc).__name__}: {str(exc)[:100]}')
        if time.perf_counter() - last_print > 30:
            last_print = time.perf_counter()
            print(f'   [{i+1}/{len(selected)}] added={n_added} '
                  f'failed={n_failed} elapsed={time.perf_counter()-t0:.0f}s')

    elapsed = time.perf_counter() - t0
    print(f'\n[3/4] done in {elapsed/60:.1f} min: '
          f'added={n_added}  skipped_existing={n_skipped}  failed={n_failed}')

    print('\n[4/4] new dataset_v8 source coverage:')
    for split in ('train', 'val'):
        img_dir = DATASET / split / 'images'
        radiata = sum(1 for _ in img_dir.glob('radiata_*.png'))
        ixi = sum(1 for _ in img_dir.glob('ixi2d_*.png'))
        oneuro = sum(1 for _ in img_dir.glob('oneuro_*.png'))
        kaggle = sum(1 for _ in img_dir.glob('neg_kaggle*.png'))
        total = sum(1 for _ in img_dir.glob('*.png'))
        print(f'  {split:5s}  total={total:5d}  kaggle_neg={kaggle:5d}  '
              f'openneuro={oneuro:5d}  ixi2d={ixi:5d}  radiata={radiata:5d}')


if __name__ == '__main__':
    main()
