"""Augment dataset_v8 negative class with OpenNeuro healthy brain slices.

Problem (measured in scripts/eval_ood_threshold_sweep_new_clfs.py):
  - dataset_v8 is 77.8% positive (19,421 tumor / 5,555 no_tumor)
  - The no_tumor 5,555 are all Kaggle 4-class (single style: skulls
    included, specific preprocessing)
  - Classifiers retrained on this gave 100% FPR on OpenNeuro healthy
    coronal T1 — they learned "if it doesn't look like a Kaggle
    no_tumor it must be a tumor"

Fix: add OpenNeuro healthy brain slices to the no_tumor class of
dataset_v8. Specifically from `g4m3r/T1w_MRI_Brain_Slices` (OpenNeuro
ds003592, the same source as the held-out OOD eval set) but using
DIFFERENT subjects so the eval cohort stays untouched.

Held-out OOD subjects (do NOT use): 01, 09, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89.
Available subjects in g4m3r: ~99 (sub-01 .. sub-99). After excluding the
12 held-out, 87 subjects remain → ~2,000 slices to add to dataset_v8.

Split assignment: by patient (no patient appears in multiple splits).
- 87 subjects → ~78 train / 4 val / 5 test (90/5/5 by patient count)
- ~20 slices per subject (stride to keep size manageable)
- Total adds ~1,560 to train, ~80 val, ~100 test no_tumor PNGs.

Each PNG is paired with an all-zero mask of the same name (so the
trainer's mask-derived labelling correctly labels it as no_tumor).

Idempotent: skips files that already exist.
"""
from __future__ import annotations

import io
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / 'dataset_v8'

# These 12 are in samples/ood/healthy_coronal_T1_openneuro/ — must not leak.
HELD_OUT_SUBJECTS = {f'{n:02d}' for n in (1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89)}

# Pull every Nth slice per subject to keep size manageable. The dataset
# has slices 141-199 per subject (~30 slices). Stride 2 -> ~15 per subj.
SLICE_STRIDE = 2
# Split assignment by patient: deterministic from subject number.
TRAIN_PCT, VAL_PCT = 0.90, 0.05  # rest = test


def _which_split(sub_num: int) -> str:
    """Deterministic per-subject split. Use sub_num mod 20:
       0..17 -> train  (90%)
       18    -> val    ( 5%)
       19    -> test   ( 5%)
    """
    bucket = sub_num % 20
    if bucket < 18:
        return 'train'
    if bucket == 18:
        return 'val'
    return 'test'


def main():
    print(f'[init] DATASET={DATASET}')
    if not DATASET.exists():
        sys.exit(f'ERROR: dataset_v8 not found at {DATASET}')

    print('[1/3] downloading OpenNeuro images.zip (one-time, ~352 MB)...')
    t0 = time.perf_counter()
    cached = ROOT / 'samples' / 'ood' / '_zip_tmp' / 'images.zip'
    cached.parent.mkdir(parents=True, exist_ok=True)
    if cached.exists() and cached.stat().st_size > 100_000_000:
        zpath = str(cached)
        print(f'   (reusing cached zip: {cached.stat().st_size/1e6:.1f} MB)')
    else:
        zpath = hf_hub_download(
            repo_id='g4m3r/T1w_MRI_Brain_Slices',
            filename='images.zip',
            repo_type='dataset',
            local_dir=str(cached.parent),
        )
    print(f'   downloaded/located in {time.perf_counter()-t0:.1f}s')

    # Verify dataset_v8 split dirs exist with images + masks subdirs.
    for s in ('train', 'val', 'test'):
        (DATASET / s / 'images').mkdir(parents=True, exist_ok=True)
        (DATASET / s / 'masks').mkdir(parents=True, exist_ok=True)

    print('[2/3] extracting per-subject, filtering, writing to dataset_v8/...')
    import re
    pat = re.compile(r'^images/sub-(\d+)_slice_(\d+)\.png$')

    per_split_counts = {'train': 0, 'val': 0, 'test': 0}
    subjects_seen: set[str] = set()
    skipped_held_out: set[str] = set()
    skipped_existing = 0
    n_added = 0

    with zipfile.ZipFile(zpath) as zf:
        all_names = zf.namelist()
        for nm in sorted(all_names):
            m = pat.match(nm)
            if not m:
                continue
            sub = m.group(1)
            sl = int(m.group(2))
            if sub in HELD_OUT_SUBJECTS:
                skipped_held_out.add(sub)
                continue
            # Apply slice stride
            if (sl % (2 * SLICE_STRIDE)) != 1:  # arbitrary phase
                continue
            subjects_seen.add(sub)
            split = _which_split(int(sub))
            base = f'oneuro_sub{sub}_slice{sl}.png'
            img_out = DATASET / split / 'images' / base
            mask_out = DATASET / split / 'masks' / base
            if img_out.exists() and mask_out.exists():
                skipped_existing += 1
                continue
            # Extract PNG bytes, write image as-is
            with zf.open(nm) as src:
                data = src.read()
            img_out.write_bytes(data)
            # Read image to get shape, create all-zero mask of same H x W
            img_arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if img_arr is None:
                img_out.unlink(missing_ok=True)
                continue
            mask = np.zeros_like(img_arr, dtype=np.uint8)
            ok = cv2.imwrite(str(mask_out), mask)
            if not ok:
                img_out.unlink(missing_ok=True)
                continue
            per_split_counts[split] += 1
            n_added += 1

    print(f'[3/3] done')
    print(f'   subjects used (added healthy):  {len(subjects_seen)}')
    print(f'   subjects held out (untouched):  {len(skipped_held_out)} '
          f'(must equal {len(HELD_OUT_SUBJECTS)})')
    print(f'   pre-existing files skipped:     {skipped_existing}')
    print(f'   total no_tumor PNGs added:      {n_added}')
    print(f'     train: {per_split_counts["train"]}')
    print(f'     val:   {per_split_counts["val"]}')
    print(f'     test:  {per_split_counts["test"]}')

    # Show new class balance per split
    print()
    print('[verify] new dataset_v8 sizes:')
    for s in ('train', 'val', 'test'):
        imgs = list((DATASET / s / 'images').glob('*.png'))
        # Quick label split: anything starting with oneuro_ is no_tumor
        # by construction; everything else uses mask sum
        oneuro_count = sum(1 for p in imgs if p.name.startswith('oneuro_'))
        print(f'   {s:5s}: total={len(imgs):5d}  '
              f'+oneuro_healthy_added={oneuro_count}')


if __name__ == '__main__':
    main()
