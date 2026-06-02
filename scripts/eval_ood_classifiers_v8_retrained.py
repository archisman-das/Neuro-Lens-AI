"""Re-run the brutal OOD audit on the v8-distribution-RETRAINED classifiers.

Direct apples-to-apples vs scripts/eval_ood_classifiers_brutal.py, which
audited the OLD classifiers (trained on Kaggle 4-class only). Difference:
classifiers are loaded from real_eval_v8_retrained/ instead of
real_eval_current/. Same OOD samples, same metric definitions.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Use the same eval helpers as the original brutal audit; we just swap
# the classifier ONNX paths.
from scripts.eval_ood_cascade import (
    SEG_ONNX, MIN_TUMOR_AREA, modality_of, GT,
    _sess, _preprocess_seg, _preprocess_clf, seg_tta,
)

NEW_CLF_ONNX = {
    'cnn':      ROOT / 'real_eval_v8_retrained' / 'cnn'      / 'best_weights.onnx',
    'transfer': ROOT / 'real_eval_v8_retrained' / 'transfer' / 'best_weights.onnx',
    'vit':      ROOT / 'real_eval_v8_retrained' / 'vit'      / 'best_weights.onnx',
}
# CNN is the only one NOT ImageNet-normalised (matches the original
# build and what dashboard.py knows about via ckpt['normalize_imagenet']).
NORMALIZE_IMAGENET = {'cnn': False, 'transfer': True, 'vit': True}

SAMPLES_DIR = ROOT / 'samples' / 'ood'


def classify_all(clfs: dict, img: Image.Image) -> dict:
    out = {}
    for name, sess in clfs.items():
        chw = _preprocess_clf(img, NORMALIZE_IMAGENET[name])
        logit = float(sess.run(None, {sess.get_inputs()[0].name: chw[None]})[0].reshape(-1)[0])
        out[name] = 1.0 / (1.0 + np.exp(-logit))
    return out


def main():
    for name, p in NEW_CLF_ONNX.items():
        if not p.exists():
            sys.exit(f'missing {name}: {p}')
    seg = _sess(SEG_ONNX)
    clfs = {n: _sess(p) for n, p in NEW_CLF_ONNX.items()}
    samples = sorted(p for p in SAMPLES_DIR.rglob('*')
                      if p.suffix.lower() in ('.png', '.jpg', '.jpeg')
                      and p.parent.name in GT)
    print(f'[init] {len(samples)} OOD samples; classifiers from real_eval_v8_retrained/')

    rows = []
    t0 = time.perf_counter()
    for p in samples:
        img = Image.open(p)
        gt = GT[p.parent.name]
        probs = classify_all(clfs, img)
        prob_map = seg_tta(seg, _preprocess_seg(img))
        rows.append({
            'source': p.parent.name, 'file': p.name, 'gt': gt,
            'p_cnn': probs['cnn'], 'p_transfer': probs['transfer'], 'p_vit': probs['vit'],
            'v8_pmax': float(prob_map.max()),
            'v8_area_020': int((prob_map >= 0.20).sum()),
        })
    print(f'[done] {len(rows)} samples in {time.perf_counter()-t0:.0f}s\n')

    # =================== per-classifier scorecard =========================
    print('='*78)
    print('PER-CLASSIFIER ACCURACY ON OOD (v8-retrained — fresh weights, no cascade)')
    print('='*78)
    OLD_NUMBERS = {  # from scripts/eval_ood_classifiers_brutal.py output
        'cnn':      {'recall': 0.28, 'fpr': 0.58, 'acc': 0.31},
        'transfer': {'recall': 0.36, 'fpr': 0.08, 'acc': 0.50},
        'vit':      {'recall': 0.42, 'fpr': 0.08, 'acc': 0.54},
    }
    for clf in ('cnn', 'transfer', 'vit'):
        pkey = f'p_{clf}'
        TP = sum(1 for r in rows if r['gt']=='tumor' and r[pkey]>=0.5)
        FN = sum(1 for r in rows if r['gt']=='tumor' and r[pkey]<0.5)
        FP = sum(1 for r in rows if r['gt']=='no_tumor' and r[pkey]>=0.5)
        TN = sum(1 for r in rows if r['gt']=='no_tumor' and r[pkey]<0.5)
        recall = TP/(TP+FN) if TP+FN else 0
        fpr = FP/(FP+TN) if FP+TN else 0
        acc = (TP+TN)/len(rows)
        old = OLD_NUMBERS[clf]
        d_rec = recall - old['recall']
        d_fpr = fpr - old['fpr']
        d_acc = acc - old['acc']
        print(f'\n  {clf.upper():12s}  TP={TP:2d}  FN={FN:2d}  FP={FP:2d}  TN={TN:2d}')
        print(f'    NEW (v8-retrained):  recall={recall:.0%}  FPR={fpr:.0%}  accuracy={acc:.0%}')
        print(f'    OLD (Kaggle-only):   recall={old["recall"]:.0%}  FPR={old["fpr"]:.0%}  accuracy={old["acc"]:.0%}')
        print(f'    DELTA:               recall {d_rec:+.0%}    FPR {d_fpr:+.0%}    accuracy {d_acc:+.0%}')

    # =================== per-source recall ===============================
    print('\n' + '='*78)
    print('PER-SOURCE RECALL (GT=tumor) and FPR (GT=no_tumor)')
    print('='*78)
    by_src = {}
    for r in rows:
        by_src.setdefault(r['source'], []).append(r)
    print(f'\n{"source":48s} GT     n   {"cnn":>5s} {"trans":>5s} {"vit":>5s}  {"v8seg":>6s}')
    for src in sorted(by_src):
        rs = by_src[src]
        gt = rs[0]['gt']
        n = len(rs)
        cells = []
        for clf in ('p_cnn', 'p_transfer', 'p_vit'):
            hits = sum(1 for r in rs if r[clf] >= 0.5)
            cells.append(f'{hits/n:.0%}'.rjust(5))
        v8_hits = sum(1 for r in rs if r['v8_area_020'] >= MIN_TUMOR_AREA)
        v8_metric = v8_hits/n
        kind = 'recall' if gt == 'tumor' else 'FP rate'
        print(f'  {src:46s} {gt[:6]:6s} {n:3d}   {cells[0]} {cells[1]} {cells[2]}  '
              f'{v8_metric:.0%}'.rjust(6) + f'   <- {kind}')

    # =================== consensus breakdown =============================
    print('\n' + '='*78)
    print('CLASSIFIER CONSENSUS ON 36 OOD TUMOR SAMPLES')
    print('='*78)
    tum = [r for r in rows if r['gt'] == 'tumor']
    n_all_no = sum(1 for r in tum if r['p_cnn']<0.5 and r['p_transfer']<0.5 and r['p_vit']<0.5)
    n_all_yes = sum(1 for r in tum if r['p_cnn']>=0.5 and r['p_transfer']>=0.5 and r['p_vit']>=0.5)
    n_split = len(tum) - n_all_no - n_all_yes
    print(f'\n  v8-retrained classifiers:')
    print(f'    ALL 3 say "tumor"          (clean detect):    {n_all_yes:3d} / {len(tum)}  ({n_all_yes/len(tum):.0%})')
    print(f'    SPLIT (some yes, some no):                    {n_split:3d} / {len(tum)}  ({n_split/len(tum):.0%})')
    print(f'    ALL 3 say "no_tumor"       (catastrophic):    {n_all_no:3d} / {len(tum)}  ({n_all_no/len(tum):.0%})')
    print(f'\n  OLD Kaggle-only classifiers (for reference):')
    print(f'    ALL 3 say "tumor":           0 / 36  (0%)')
    print(f'    SPLIT:                      26 / 36  (72%)')
    print(f'    ALL 3 say "no_tumor":       10 / 36  (28%)')


if __name__ == '__main__':
    main()
