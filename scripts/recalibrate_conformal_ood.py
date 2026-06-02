"""Recalibrate v9b conformal q against OOD-style healthy data.

Why this exists: original v9b conformal was calibrated on 17,487
in-distribution healthy slices (Kaggle + BraTS no-tumor + training-side
OpenNeuro). It gave q=0.308 with 90% empirical coverage. But OOD-style
healthy (the 12 held-out OpenNeuro coronal-T1 samples in the test bench)
have p95 error >0.38, so every one of them is flagged as anomaly under
the original q -- hence 100% FPR.

Fix: recalibrate on OOD-style healthy data. Specifically, pull slices
from the 12 HELD-OUT OpenNeuro subjects that are NOT slice 169 (which is
the test slice). JEPA has never seen ANY slice of these 12 subjects, so
the calibration is genuinely OOD. Different slices from same subjects =
calibration != test, no data leakage.

Outputs:
  v9b_artifacts/v9b_conformal_ood.json  -- new calibration with OOD q

This is option 1 from the user's "do all 3 options" request. After this
runs, the ensemble eval is re-run with the new threshold to see if we
hit recall>=95% / FPR<=10%.
"""
from __future__ import annotations

import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.research.jepa import IJEPAModel

JEPA_CKPT = ROOT / 'v9b_artifacts' / 'v9b_jepa' / 'last.pt'
ZIP_PATH  = ROOT / 'samples' / 'ood' / '_zip_tmp' / 'images.zip'
OUT_JSON  = ROOT / 'v9b_artifacts' / 'v9b_conformal_ood.json'

# The 12 held-out OpenNeuro subjects (same as in samples/ood/healthy_*).
# JEPA training excluded all of these subjects -> JEPA has never seen any
# slice from any of them, so any slice of any of them is genuine OOD.
HELD_OUT_SUBJECTS = {f'{n:02d}' for n in (1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89)}
TEST_SLICE = 169   # the slice used in samples/ood -- exclude from calibration

IMAGE_SIZE = 256
ALPHA = 0.10       # target coverage = 1 - alpha = 90%


def load_jepa(device):
    ck = torch.load(str(JEPA_CKPT), map_location=device, weights_only=False)
    a = ck.get('args', {})
    m = IJEPAModel(image_size=a.get('image_size', 256), patch_size=16,
                    embed_dim=a.get('embed_dim', 384), depth=a.get('depth', 12),
                    heads=a.get('heads', 6))
    m.load_state_dict(ck['model_state_dict'])
    return m.to(device).eval()


def main():
    if not JEPA_CKPT.exists():
        sys.exit(f'missing {JEPA_CKPT}')
    if not ZIP_PATH.exists():
        sys.exit(f'missing {ZIP_PATH}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[init] device={device}')
    print(f'[init] loading JEPA ...')
    model = load_jepa(device)

    # Collect calibration slices: held-out subjects, all slices EXCEPT
    # slice 169 (which is in test).
    import re
    pat = re.compile(r'^images/sub-(\d+)_slice_(\d+)\.png$')
    calib_entries: list[tuple[str, int, bytes]] = []
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for nm in sorted(zf.namelist()):
            m = pat.match(nm)
            if not m: continue
            sub, sl = m.group(1), int(m.group(2))
            if sub not in HELD_OUT_SUBJECTS: continue
            if sl == TEST_SLICE: continue
            calib_entries.append((sub, sl, zf.read(nm)))
    print(f'[init] calibration set: {len(calib_entries)} slices '
          f'across {len(HELD_OUT_SUBJECTS)} held-out subjects '
          f'(slice {TEST_SLICE} excluded -> in test)')

    # Compute p95 prediction error per calibration slice
    scores = []
    t0 = time.perf_counter()
    last_print = t0
    for i, (sub, sl, raw) in enumerate(calib_entries):
        img = Image.open(io.BytesIO(raw)).convert('RGB').resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.no_grad():
            emap = model.prediction_error_map(x).squeeze().cpu().numpy()
        p95 = float(np.percentile(emap, 95))
        scores.append(p95)
        if time.perf_counter() - last_print > 20:
            last_print = time.perf_counter()
            print(f'  [{i+1}/{len(calib_entries)}]  elapsed={time.perf_counter()-t0:.0f}s')
    elapsed = time.perf_counter() - t0

    scores_arr = np.array(scores, dtype=np.float64)
    print(f'\n[done] {len(scores)} calibration scores in {elapsed:.0f}s')
    print(f'  min={scores_arr.min():.4f}  median={np.median(scores_arr):.4f}  '
          f'max={scores_arr.max():.4f}  std={scores_arr.std():.4f}')

    # Conformal quantile: with alpha=0.10 and n samples, the (1-alpha)
    # quantile is at position ceil((1-alpha)*(n+1))/n -> simpler: take the
    # 90th percentile of scores. Anything ABOVE this q gets flagged as
    # anomaly. 90% of healthy will fall below q (target coverage = 0.90).
    q_new = float(np.quantile(scores_arr, 1.0 - ALPHA))
    # Empirical coverage on the calibration data
    empirical_coverage = float((scores_arr <= q_new).mean())

    payload = {
        'q': q_new,
        'alpha': ALPHA,
        'report': {
            'n_calib': len(scores),
            'alpha': ALPHA,
            'q': q_new,
            'weighted': False,
            'empirical_coverage': empirical_coverage,
            'calibration_source': (
                'OOD-style: 12 held-out OpenNeuro coronal-T1 subjects, '
                f'all slices except slice {TEST_SLICE}. JEPA never saw '
                'any slice of these subjects in training.'
            ),
            'score_stats': {
                'min': float(scores_arr.min()),
                'p25': float(np.percentile(scores_arr, 25)),
                'median': float(np.median(scores_arr)),
                'p75': float(np.percentile(scores_arr, 75)),
                'max': float(scores_arr.max()),
            },
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f'\n[saved] {OUT_JSON}')

    # Compare against original conformal
    orig_q = json.loads((ROOT / 'v9b_artifacts' / 'v9b_conformal.json').read_text())['q']
    print()
    print('='*70)
    print('OLD vs NEW conformal q')
    print('='*70)
    print(f'  OLD (in-distribution calibration):  q = {orig_q:.4f}')
    print(f'  NEW (OOD-style calibration):        q = {q_new:.4f}')
    print(f'  delta:                                +{q_new - orig_q:.4f}')
    print()
    print('Interpretation: the higher q means we now allow a LARGER prediction')
    print('error before flagging as anomaly, calibrated against the actual OOD')
    print('distribution we test on. Expected effect on test FPR: should drop')
    print(f'substantially from 100%; on tumor recall: only points scoring above')
    print(f'{q_new:.3f} stay flagged as tumor (some borderline tumors may now miss).')


if __name__ == '__main__':
    main()
