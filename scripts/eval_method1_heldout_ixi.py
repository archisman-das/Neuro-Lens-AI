"""HELD-OUT validation of CrossJEPA Method 1.

Healthy pool: 50 IXI T1 volumes (NEVER seen during training — radiata-ai
was trained on DLBS/NKI-RS/OASIS-1/OASIS-2 only). IXI volumes pulled
from `pzarzycki/mri-oasis-1-ixi-pre:ixi_preprocessed/IXI*-T1.nii.gz`.

Tumor pool: random 50 BraTS-2021 FLAIR volumes.

Reports AUC, Pareto, and best-F1. The hypothesis: if AUC stays >= 0.95
on this truly-held-out healthy set, the model learned a real healthy-
brain manifold (not just dataset-specific features). Anything lower
than ~0.95 would mean the earlier perfect AUC was at least partly
optimization to the radiata-ai training distribution.
"""
from __future__ import annotations

import csv
import glob
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.research.v9c_crossjepa.dataset_3d import Vol2SliceDataset
from src.research.v9c_crossjepa.v8_teacher import V8FrozenTeacher
from src.research.v9c_crossjepa.volume_to_slice import Vol2SliceModel


CKPT = ROOT / 'v9b_artifacts' / 'v9c_crossjepa_method1' / 'last.pt'
V8_PT = ROOT / 'model' / 'best_micro.pt'
IXI_GLOB = 'd:/datasets/ixi_heldout/ixi_preprocessed/IXI*-T1.nii.gz'
BRATS_ROOT = Path('d:/datasets/brats2021')
OUT_CSV = ROOT / 'samples' / 'ood' / 'eval_method1_heldout_ixi_results.csv'

N_TUMOR = 50
N_SLICES = 8


def _collate(b):
    keys = ('volume', 'target_slice_rgb', 'plane_idx', 'slice_idx_norm',
            'voxel_spacing', 'intensity_hist')
    return {k: torch.stack([s[k] for s in b]) for k in keys}


def _score_volume(model, path, in_channels, volume_size, n_slices, device):
    ds = Vol2SliceDataset(
        scan_paths=[path], volume_size=volume_size, in_channels=in_channels,
        slices_per_volume=n_slices, slice_resize_hw=(256, 256),
        planes=('axial',), shuffle=False, seed=0,
    )
    per_slice = []
    for batch in DataLoader(ds, batch_size=1, collate_fn=_collate):
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device, non_blocking=True)
        with torch.no_grad():
            s = model.anomaly_score(
                batch['volume'], batch['target_slice_rgb'],
                batch['plane_idx'], batch['slice_idx_norm'],
                batch['voxel_spacing'], batch['intensity_hist'])
        per_slice.append(float(s.item()))
    return per_slice


def _auc(pos, neg):
    if not pos or not neg: return float('nan')
    wins = ties = 0
    for sp in pos:
        for sn in neg:
            if sp > sn: wins += 1
            elif sp == sn: ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[init] device={device}')
    ck = torch.load(str(CKPT), map_location=device, weights_only=False)
    args = ck.get('args', {})
    vol_size = tuple(args.get('volume_size', [144, 192, 192]))
    in_channels = args.get('in_channels', 1)

    teacher = V8FrozenTeacher.from_unet_checkpoint(
        str(V8_PT), encoder_name='tu-convnext_tiny.fb_in22k_ft_in1k',
        in_channels=3, image_size=384, device=device)
    model = Vol2SliceModel(
        v8_teacher=teacher, volume_size=vol_size, in_chans=in_channels,
        patch_size=args.get('patch_size', 16),
        encoder_dim=args.get('encoder_dim', 384),
        encoder_depth=args.get('encoder_depth', 12),
        encoder_heads=args.get('encoder_heads', 6),
        predictor_dim=args.get('predictor_dim', 192),
        predictor_depth=args.get('predictor_depth', 6),
    ).to(device)
    model.encoder.load_state_dict(ck['encoder_state_dict'])
    model.predictor.load_state_dict(ck['predictor_state_dict'])
    model.conditioning.load_state_dict(ck['conditioning_state_dict'])
    model.eval()

    # Pools
    healthy_paths = sorted(glob.glob(IXI_GLOB))
    tumor_pool = []
    for d in sorted(BRATS_ROOT.iterdir()):
        if d.is_dir():
            flair = list(d.glob('*flair.nii.gz'))
            if flair:
                tumor_pool.append(str(flair[0]))
    rng = random.Random(0)
    rng.shuffle(tumor_pool)
    tumor_paths = tumor_pool[:N_TUMOR]

    print(f'[init] HELD-OUT healthy IXI: {len(healthy_paths)} volumes')
    print(f'[init] tumor BraTS-2021 FLAIR: {len(tumor_paths)} volumes')
    print(f'[init] (radiata-ai training pool was DLBS/NKI-RS/OASIS-1/OASIS-2)')
    print(f'[init] (IXI is not in that snapshot — truly held-out healthy)')

    rows = []
    t0 = time.perf_counter()
    for label, paths in [('no_tumor', healthy_paths), ('tumor', tumor_paths)]:
        for p in paths:
            try:
                slices = _score_volume(model, p, in_channels, vol_size,
                                         N_SLICES, device)
            except Exception as exc:
                print(f'  [skip] {Path(p).stem[:50]}: {exc}')
                continue
            rows.append({
                'gt': label, 'path': p, 'scan_id': Path(p).stem,
                'mean_score': float(np.mean(slices)),
                'max_score': float(np.max(slices)),
                'p95_score': float(np.percentile(slices, 95)),
                'n_slices': len(slices),
            })
    print(f'[done] {len(rows)} volumes scored in {(time.perf_counter()-t0)/60:.1f} min')

    # Report
    for key in ('mean_score', 'max_score', 'p95_score'):
        pos = [r[key] for r in rows if r['gt'] == 'tumor']
        neg = [r[key] for r in rows if r['gt'] == 'no_tumor']
        if not pos or not neg: continue
        auc = _auc(pos, neg)
        print(f'\n=== {key} ===')
        print(f'  AUC = {auc:.4f}  (tumor n={len(pos)}, healthy n={len(neg)})')
        print(f'  tumor    mean={np.mean(pos):.4f}  '
              f'[min,p25,med,p75,max]=[{min(pos):.3f},{np.percentile(pos,25):.3f},'
              f'{np.median(pos):.3f},{np.percentile(pos,75):.3f},{max(pos):.3f}]')
        print(f'  healthy  mean={np.mean(neg):.4f}  '
              f'[min,p25,med,p75,max]=[{min(neg):.3f},{np.percentile(neg,25):.3f},'
              f'{np.median(neg):.3f},{np.percentile(neg,75):.3f},{max(neg):.3f}]')
        # Best-F1 sweep
        thresholds = sorted(set(round(v, 5) for v in pos + neg))
        best_t, best_f1 = None, -1
        for t in thresholds:
            TP = sum(1 for v in pos if v > t)
            FN = len(pos) - TP
            FP = sum(1 for v in neg if v > t)
            TN = len(neg) - FP
            re = TP/(TP+FN) if TP+FN else 0
            fp = FP/(FP+TN) if FP+TN else 0
            pr = TP/(TP+FP) if TP+FP else 0
            f1 = 2*pr*re/(pr+re) if pr+re else 0
            if f1 > best_f1:
                best_t, best_f1, best_re, best_fp, best_pr = t, f1, re, fp, pr
        print(f'  BEST F1: t={best_t:.4f}  recall={best_re:.0%}  '
              f'FPR={best_fp:.0%}  prec={best_pr:.0%}  F1={best_f1:.3f}')

    # CSV dump
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f'\n[csv] {OUT_CSV}')


if __name__ == '__main__':
    main()
