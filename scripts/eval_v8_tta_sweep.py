"""TTA (4-way flip ensemble) + threshold sweep evaluation of v8 ONNX.

For each scan:
  1. Run 4 forward passes: identity, hflip, vflip, both
  2. Average the per-voxel probabilities (after inverse-transforming each)
  3. Evaluate at multiple thresholds in one shot — no re-inference per threshold

Per-threshold metrics aggregated globally + per-source.
Picks the composite-optimal and F1-optimal thresholds.

Usage:
    python scripts/eval_v8_tta_sweep.py --onnx model/best_micro.onnx
        --data_dir dataset_v8 --splits test val --output_dir model/eval_results
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def load_image_normalized(path: Path, image_size: int) -> np.ndarray:
    img = Image.open(path).convert('RGB').resize((image_size, image_size), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


def load_mask(path: Path, image_size: int) -> np.ndarray:
    msk = Image.open(path).convert('L').resize((image_size, image_size), Image.NEAREST)
    return (np.asarray(msk, dtype=np.uint8) > 127).astype(np.uint8)


def infer_source(filename: str) -> str:
    n = filename.lower()
    if n.startswith('brats_t1c_') or 'brats' in n: return 'brats_t1c'
    if 'lgg' in n: return 'lgg'
    if 'figshare' in n: return 'figshare'
    if 'no_tumor' in n or 'notumor' in n or 'kaggle' in n: return 'kaggle_negative'
    return 'unknown'


def tta_forward(sess, x_batch: np.ndarray) -> np.ndarray:
    """4-way flip TTA. Returns averaged sigmoid probabilities (B, H, W)."""
    # id
    logits = sess.run(None, {'input': x_batch})[0]
    p_id = 1.0 / (1.0 + np.exp(-logits[:, 0]))
    # hflip
    x_h = x_batch[:, :, :, ::-1].copy()
    logits = sess.run(None, {'input': x_h})[0]
    p_h = 1.0 / (1.0 + np.exp(-logits[:, 0]))
    p_h = p_h[:, :, ::-1]  # invert flip on output
    # vflip
    x_v = x_batch[:, :, ::-1, :].copy()
    logits = sess.run(None, {'input': x_v})[0]
    p_v = 1.0 / (1.0 + np.exp(-logits[:, 0]))
    p_v = p_v[:, ::-1, :]
    # both
    x_b = x_batch[:, :, ::-1, ::-1].copy()
    logits = sess.run(None, {'input': x_b})[0]
    p_b = 1.0 / (1.0 + np.exp(-logits[:, 0]))
    p_b = p_b[:, ::-1, ::-1]
    return (p_id + p_h + p_v + p_b) / 4.0


def eval_split_tta_sweep(sess, img_dir: Path, msk_dir: Path, image_size: int,
                         batch_size: int) -> Dict:
    img_paths = sorted(img_dir.iterdir())
    n = len(img_paths)
    print(f'  TTA-evaluating {n} scans from {img_dir.parent.name}...', flush=True)
    # Per-threshold totals (pooled) + per-source totals
    pooled = {t: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
                   'pos_dices': [], 'pos_recalls': [], 'pos_precs': [],
                   'pos_fn_rates': [], 'neg_fp_rates': []}
              for t in THRESHOLDS}
    by_source = defaultdict(lambda: defaultdict(lambda: {
        'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'n_pos': 0, 'n_neg': 0,
        'pos_dices': [], 'neg_fp_rates': [],
    }))
    auroc_probs = []
    auroc_labels = []
    t0 = time.time()
    for bi in range(0, n, batch_size):
        batch_paths = img_paths[bi:bi + batch_size]
        x_batch, y_batch, names, sources = [], [], [], []
        for ip in batch_paths:
            mp = msk_dir / ip.name
            if not mp.exists(): continue
            x_batch.append(load_image_normalized(ip, image_size))
            y_batch.append(load_mask(mp, image_size))
            names.append(ip.name)
            sources.append(infer_source(ip.name))
        if not x_batch: continue
        x_batch = np.stack(x_batch, axis=0).astype(np.float32)
        # TTA-averaged probabilities
        avg_probs = tta_forward(sess, x_batch)  # (B, H, W)
        for i in range(len(x_batch)):
            yi = y_batch[i]
            pri = avg_probs[i]
            src = sources[i]
            has_tumor = int(yi.sum() > 0)
            auroc_probs.append(float(pri.max()))
            auroc_labels.append(has_tumor)
            for t in THRESHOLDS:
                pi = (pri >= t).astype(np.uint8)
                tp = int((pi * yi).sum()); fp = int((pi * (1 - yi)).sum())
                fn = int(((1 - pi) * yi).sum()); tn = int(((1 - pi) * (1 - yi)).sum())
                pooled[t]['tp'] += tp; pooled[t]['fp'] += fp
                pooled[t]['fn'] += fn; pooled[t]['tn'] += tn
                if has_tumor:
                    pooled[t]['pos_dices'].append((2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6))
                    pooled[t]['pos_recalls'].append((tp + 1e-6) / (tp + fn + 1e-6))
                    pooled[t]['pos_precs'].append((tp + 1e-6) / (tp + fp + 1e-6))
                    pooled[t]['pos_fn_rates'].append((fn + 1e-6) / (tp + fn + 1e-6))
                    by_source[src][t]['pos_dices'].append(
                        (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6))
                else:
                    pooled[t]['neg_fp_rates'].append(fp / (fp + tn + 1e-6))
                    by_source[src][t]['neg_fp_rates'].append(fp / (fp + tn + 1e-6))
                by_source[src][t]['tp'] += tp; by_source[src][t]['fp'] += fp
                by_source[src][t]['fn'] += fn; by_source[src][t]['tn'] += tn
                if has_tumor:
                    by_source[src][t]['n_pos'] += 1
                else:
                    by_source[src][t]['n_neg'] += 1
        if (bi // batch_size) % 25 == 0:
            print(f'    [{bi + len(batch_paths)}/{n}] {time.time() - t0:.0f}s elapsed', flush=True)
    # Aggregate
    summary = {}
    for t in THRESHOLDS:
        agg = pooled[t]
        tp, fp, fn, tn = agg['tp'], agg['fp'], agg['fn'], agg['tn']
        micro_dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
        macro_dice = float(np.mean(agg['pos_dices'])) if agg['pos_dices'] else 0.0
        macro_recall = float(np.mean(agg['pos_recalls'])) if agg['pos_recalls'] else 0.0
        macro_prec = float(np.mean(agg['pos_precs'])) if agg['pos_precs'] else 0.0
        macro_fn_rate = float(np.mean(agg['pos_fn_rates'])) if agg['pos_fn_rates'] else 0.0
        fp_rate_mean = float(np.mean(agg['neg_fp_rates'])) if agg['neg_fp_rates'] else 0.0
        fp_rate_p95 = float(np.percentile(agg['neg_fp_rates'], 95)) if agg['neg_fp_rates'] else 0.0
        f1 = 2 * macro_prec * macro_recall / max(macro_prec + macro_recall, 1e-6)
        composite = macro_dice - 5.0 * fp_rate_mean
        summary[f'{t:.2f}'] = {
            'micro_dice': float(micro_dice), 'macro_dice': macro_dice,
            'macro_recall': macro_recall, 'macro_precision': macro_prec,
            'macro_fn_rate': macro_fn_rate, 'macro_f1': float(f1),
            'fp_rate_mean': fp_rate_mean, 'fp_rate_p95': fp_rate_p95,
            'composite': float(composite),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        }
    # AUROC (threshold-independent)
    auroc = None
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(auroc_labels)) > 1:
            auroc = float(roc_auc_score(auroc_labels, auroc_probs))
    except ImportError:
        pass
    # Pick optimal thresholds
    best_composite_t = max(summary.keys(), key=lambda t: summary[t]['composite'])
    best_f1_t = max(summary.keys(), key=lambda t: summary[t]['macro_f1'])
    # Per-source summaries (only at best composite threshold + 0.5 + 0.3 for compactness)
    src_summary = {}
    for src, td in by_source.items():
        src_summary[src] = {}
        for show_t in (best_composite_t, '0.50', '0.30'):
            if show_t in td:
                d = td[show_t]
                src_summary[src][show_t] = {
                    'n_pos': d['n_pos'], 'n_neg': d['n_neg'],
                    'macro_dice': float(np.mean(d['pos_dices'])) if d['pos_dices'] else 0.0,
                    'mean_fp_rate': float(np.mean(d['neg_fp_rates'])) if d['neg_fp_rates'] else 0.0,
                    'recall_pooled': d['tp'] / max(d['tp'] + d['fn'], 1),
                    'precision_pooled': d['tp'] / max(d['tp'] + d['fp'], 1),
                }
    return {
        'n_scans': n, 'thresholds': summary, 'auroc': auroc,
        'best_composite_threshold': best_composite_t,
        'best_f1_threshold': best_f1_t,
        'by_source': src_summary,
        'elapsed_sec': time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', default='model/best_micro.onnx')
    ap.add_argument('--data_dir', default='dataset_v8')
    ap.add_argument('--output_dir', default='model/eval_results')
    ap.add_argument('--splits', nargs='+', default=['test', 'val'])
    ap.add_argument('--image_size', type=int, default=384)
    ap.add_argument('--batch_size', type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
    sess = ort.InferenceSession(args.onnx, sess_options=so, providers=providers)
    print(f'ONNX loaded, providers: {sess.get_providers()}', flush=True)
    print(f'TTA: 4-way (id + hflip + vflip + both)', flush=True)
    print(f'Threshold sweep: {THRESHOLDS}', flush=True)

    results = {}
    out_json = out_dir / 'eval_tta_sweep.json'
    for split in args.splits:
        img_dir = Path(args.data_dir) / split / 'images'
        msk_dir = Path(args.data_dir) / split / 'masks'
        if not img_dir.exists(): continue
        print(f'\n=== TTA + sweep on {split} ===', flush=True)
        r = eval_split_tta_sweep(sess, img_dir, msk_dir, args.image_size, args.batch_size)
        results[split] = r
        # CRITICAL: save JSON immediately, before any print/format that could crash
        # on encoding issues (lesson learned from the Unicode star crash).
        try:
            out_json.write_text(json.dumps(results, indent=2, default=float),
                                encoding='utf-8')
            print(f'  [checkpoint] saved partial results to {out_json}', flush=True)
        except Exception as exc:
            print(f'  [checkpoint] save failed: {exc}', flush=True)
        # Print per-threshold table (ASCII-only; cp1252-safe)
        try:
            print(f'\n  {split} per-threshold (overall):', flush=True)
            print(f'  {"thr":>5s}  {"macro_d":>8s}  {"micro_d":>8s}  {"recall":>7s}  '
                  f'{"prec":>6s}  {"FN_rate":>8s}  {"FP_rate":>8s}  {"F1":>6s}  '
                  f'{"comp":>6s}  mark', flush=True)
            for t_str, d in r['thresholds'].items():
                mark = ''
                if t_str == r['best_composite_threshold']: mark += '*'  # composite-best
                if t_str == r['best_f1_threshold']: mark += 'F'         # F1-best
                print(f'  {t_str:>5s}  {d["macro_dice"]:>8.4f}  {d["micro_dice"]:>8.4f}'
                      f'  {d["macro_recall"]:>7.4f}  {d["macro_precision"]:>6.4f}'
                      f'  {d["macro_fn_rate"]:>8.4f}  {d["fp_rate_mean"]:>8.5f}'
                      f'  {d["macro_f1"]:>6.4f}  {d["composite"]:>6.4f}  {mark}',
                      flush=True)
            if r['auroc'] is not None:
                print(f'  AUROC: {r["auroc"]:.4f}', flush=True)
            print(f'  best composite threshold: {r["best_composite_threshold"]}', flush=True)
            print(f'  best F1 threshold:        {r["best_f1_threshold"]}', flush=True)
        except Exception as exc:
            print(f'  [print] table render failed (data is safe in JSON): {exc}',
                  flush=True)

    print(f'\nFinal save: {out_json}', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
