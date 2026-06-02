"""Full evaluation of v8 ONNX checkpoint on dataset_v8/test + samples/.

Computes every clinically relevant metric (macro Dice, micro Dice, IoU,
recall/sensitivity, precision, F1, FN rate, FP rate, specificity, AUROC),
stratified by source dataset (BraTS / LGG / Figshare / Kaggle), and
reports per-scan + pooled summaries.

Usage:
    python scripts/eval_v8_full.py --onnx model/best_micro.onnx \\
        --data_dir dataset_v8 --output_dir model/eval_results

Required files in model/:
    best_micro.onnx        (838 KB)
    best_micro.onnx.data   (121.9 MB, must be alongside .onnx)
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image_normalized(path: Path, image_size: int) -> np.ndarray:
    img = Image.open(path).convert('RGB').resize((image_size, image_size), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


def load_mask(path: Path, image_size: int) -> np.ndarray:
    msk = Image.open(path).convert('L').resize((image_size, image_size), Image.NEAREST)
    return (np.asarray(msk, dtype=np.uint8) > 127).astype(np.uint8)


def infer_source(filename: str) -> str:
    """Classify a sample by filename prefix into source dataset."""
    n = filename.lower()
    if n.startswith('brats_t1c_') or 'brats' in n: return 'brats_t1c'
    if 'lgg' in n: return 'lgg'
    if 'figshare' in n: return 'figshare'
    if 'no_tumor' in n or 'notumor' in n or 'kaggle' in n: return 'kaggle_negative'
    if 'meningioma' in n: return 'figshare_meningioma'
    if 'glioma' in n: return 'figshare_glioma'
    if 'pituitary' in n: return 'figshare_pituitary'
    return 'unknown'


def run_onnx_inference(sess, x_batch: np.ndarray) -> np.ndarray:
    """Returns sigmoid probabilities (B, H, W)."""
    logits = sess.run(None, {'input': x_batch})[0]
    return 1.0 / (1.0 + np.exp(-logits[:, 0]))


def evaluate_split(sess, img_dir: Path, msk_dir: Path, image_size: int,
                    threshold: float, batch_size: int = 16,
                    max_samples: int = None) -> Dict:
    img_paths = sorted(img_dir.iterdir())
    if max_samples is not None:
        img_paths = img_paths[:max_samples]
    print(f'  evaluating {len(img_paths)} scans from {img_dir.parent.name}...', flush=True)
    # Accumulators (per scan)
    per_scan = []  # list of dicts
    # Pooled tallies for micro metrics
    tp_total = fp_total = fn_total = tn_total = 0
    # AUROC accumulators
    mean_probs = []
    has_tumor_labels = []
    by_source = defaultdict(lambda: {'dices': [], 'fp_rates': [], 'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
                                       'n_pos': 0, 'n_neg': 0})
    t0 = time.time()
    # Batched loop
    for batch_start in range(0, len(img_paths), batch_size):
        batch_imgs = img_paths[batch_start:batch_start + batch_size]
        x_batch, y_batch, names, sources = [], [], [], []
        for ip in batch_imgs:
            mp = msk_dir / ip.name
            if not mp.exists(): continue
            x_batch.append(load_image_normalized(ip, image_size))
            y_batch.append(load_mask(mp, image_size))
            names.append(ip.name)
            sources.append(infer_source(ip.name))
        if not x_batch: continue
        x_batch = np.stack(x_batch, axis=0).astype(np.float32)
        probs = run_onnx_inference(sess, x_batch)
        preds = (probs >= threshold).astype(np.uint8)
        for i in range(len(x_batch)):
            yi = y_batch[i]; pi = preds[i]; pri = probs[i]
            tp = int((pi * yi).sum()); fp = int((pi * (1 - yi)).sum())
            fn = int(((1 - pi) * yi).sum()); tn = int(((1 - pi) * (1 - yi)).sum())
            has_tumor = int(yi.sum() > 0)
            mean_p_in_pred = float(pri[pi > 0].mean()) if pi.sum() > 0 else float(pri.max())
            src = sources[i]
            d = {
                'name': names[i], 'source': src, 'has_tumor': has_tumor,
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                'pred_area': int(pi.sum()), 'true_area': int(yi.sum()),
                'mean_prob': mean_p_in_pred, 'max_prob': float(pri.max()),
            }
            if has_tumor:
                d['dice'] = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
                d['iou'] = (tp + 1e-6) / (tp + fp + fn + 1e-6)
                d['recall'] = (tp + 1e-6) / (tp + fn + 1e-6)
                d['precision'] = (tp + 1e-6) / (tp + fp + 1e-6)
                d['fn_rate'] = (fn + 1e-6) / (tp + fn + 1e-6)
                by_source[src]['dices'].append(d['dice'])
                by_source[src]['n_pos'] += 1
            else:
                d['fp_rate'] = fp / (fp + tn + 1e-6)
                by_source[src]['fp_rates'].append(d['fp_rate'])
                by_source[src]['n_neg'] += 1
            by_source[src]['tp'] += tp; by_source[src]['fp'] += fp
            by_source[src]['fn'] += fn; by_source[src]['tn'] += tn
            per_scan.append(d)
            tp_total += tp; fp_total += fp; fn_total += fn; tn_total += tn
            mean_probs.append(float(pri.max()))
            has_tumor_labels.append(has_tumor)
        if (batch_start // batch_size) % 25 == 0:
            print(f'    [{batch_start + len(batch_imgs)}/{len(img_paths)}] '
                  f'{time.time() - t0:.0f}s elapsed', flush=True)
    # Aggregate
    pos_scans = [s for s in per_scan if s['has_tumor']]
    neg_scans = [s for s in per_scan if not s['has_tumor']]
    macro_dice = float(np.mean([s['dice'] for s in pos_scans])) if pos_scans else 0.0
    macro_iou = float(np.mean([s['iou'] for s in pos_scans])) if pos_scans else 0.0
    macro_recall = float(np.mean([s['recall'] for s in pos_scans])) if pos_scans else 0.0
    macro_precision = float(np.mean([s['precision'] for s in pos_scans])) if pos_scans else 0.0
    macro_fn_rate = float(np.mean([s['fn_rate'] for s in pos_scans])) if pos_scans else 0.0
    fp_rate_mean = float(np.mean([s['fp_rate'] for s in neg_scans])) if neg_scans else 0.0
    fp_rate_p95 = float(np.percentile([s['fp_rate'] for s in neg_scans], 95)) if neg_scans else 0.0
    micro_dice = (2 * tp_total + 1e-6) / (2 * tp_total + fp_total + fn_total + 1e-6)
    micro_iou = (tp_total + 1e-6) / (tp_total + fp_total + fn_total + 1e-6)
    specificity = tn_total / max(tn_total + fp_total, 1)
    # AUROC for tumor-vs-no-tumor classification using max-prob as score
    try:
        from sklearn.metrics import roc_auc_score
        auroc = float(roc_auc_score(has_tumor_labels, mean_probs)) if len(set(has_tumor_labels)) > 1 else None
    except ImportError:
        auroc = None
    composite = macro_dice - 5.0 * fp_rate_mean
    # Per-source summaries
    src_summary = {}
    for src, agg in by_source.items():
        d = {'n_pos': agg['n_pos'], 'n_neg': agg['n_neg'],
             'tp': agg['tp'], 'fp': agg['fp'], 'fn': agg['fn'], 'tn': agg['tn']}
        if agg['dices']:
            d['macro_dice'] = float(np.mean(agg['dices']))
            d['median_dice'] = float(np.median(agg['dices']))
        if agg['fp_rates']:
            d['mean_fp_rate'] = float(np.mean(agg['fp_rates']))
        if agg['tp'] + agg['fn'] > 0:
            d['recall_pooled'] = agg['tp'] / (agg['tp'] + agg['fn'])
        if agg['tp'] + agg['fp'] > 0:
            d['precision_pooled'] = agg['tp'] / (agg['tp'] + agg['fp'])
        src_summary[src] = d
    return {
        'n_scans': len(per_scan), 'n_positive': len(pos_scans), 'n_negative': len(neg_scans),
        'macro_dice': macro_dice, 'macro_iou': macro_iou,
        'macro_recall': macro_recall, 'macro_precision': macro_precision,
        'macro_fn_rate': macro_fn_rate,
        'fp_rate_mean': fp_rate_mean, 'fp_rate_p95': fp_rate_p95,
        'micro_dice': float(micro_dice), 'micro_iou': float(micro_iou),
        'specificity': float(specificity), 'auroc': auroc, 'composite': float(composite),
        'tp_total': tp_total, 'fp_total': fp_total, 'fn_total': fn_total, 'tn_total': tn_total,
        'by_source': src_summary,
        'elapsed_sec': time.time() - t0,
    }


def evaluate_samples_folder(sess, samples_root: Path, image_size: int, threshold: float) -> Dict:
    """Eval on samples/{tumor,no_tumor}/*.jpg — the user's manual test set."""
    out = {}
    for cls in ('tumor', 'no_tumor'):
        d = samples_root / cls
        if not d.exists(): continue
        results = []
        for p in sorted(d.iterdir()):
            x = load_image_normalized(p, image_size)[None].astype(np.float32)
            probs = run_onnx_inference(sess, x)[0]
            preds = (probs >= threshold).astype(np.uint8)
            results.append({
                'name': p.name, 'true_class': cls,
                'pred_area_px': int(preds.sum()),
                'mean_prob_in_pred': float(probs[preds > 0].mean()) if preds.sum() > 0 else 0.0,
                'max_prob': float(probs.max()),
                'predicted_tumor': bool(preds.sum() > 16),  # match v8 cascade threshold
            })
        out[cls] = results
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', default='model/best_micro.onnx')
    ap.add_argument('--data_dir', default='dataset_v8')
    ap.add_argument('--samples_dir', default='samples')
    ap.add_argument('--output_dir', default='model/eval_results')
    ap.add_argument('--image_size', type=int, default=384)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--max_samples_per_split', type=int, default=None,
                    help='Cap eval size for quick smoke test')
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Loading ONNX: {args.onnx}', flush=True)
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    providers = []
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')
    sess = ort.InferenceSession(args.onnx, sess_options=so, providers=providers)
    print(f'Providers active: {sess.get_providers()}', flush=True)
    print(f'Input: {sess.get_inputs()[0].name} {sess.get_inputs()[0].shape}', flush=True)
    print(f'Output: {sess.get_outputs()[0].name} {sess.get_outputs()[0].shape}', flush=True)
    print(f'Threshold: {args.threshold}  image_size: {args.image_size}', flush=True)

    results = {}
    data_root = Path(args.data_dir)
    for split in ('val', 'test'):
        img_dir = data_root / split / 'images'
        msk_dir = data_root / split / 'masks'
        if not img_dir.exists(): continue
        print(f'\n=== Evaluating {split} ===', flush=True)
        results[split] = evaluate_split(
            sess, img_dir, msk_dir, args.image_size, args.threshold,
            batch_size=args.batch_size, max_samples=args.max_samples_per_split,
        )
        print(f'  {split} summary:', flush=True)
        print(f'    macro_dice = {results[split]["macro_dice"]:.4f}', flush=True)
        print(f'    micro_dice = {results[split]["micro_dice"]:.4f}', flush=True)
        print(f'    macro_iou  = {results[split]["macro_iou"]:.4f}', flush=True)
        print(f'    recall     = {results[split]["macro_recall"]:.4f}', flush=True)
        print(f'    precision  = {results[split]["macro_precision"]:.4f}', flush=True)
        print(f'    FN rate    = {results[split]["macro_fn_rate"]:.4f}', flush=True)
        print(f'    FP rate    = {results[split]["fp_rate_mean"]:.4f} (p95: {results[split]["fp_rate_p95"]:.4f})', flush=True)
        print(f'    specificity= {results[split]["specificity"]:.4f}', flush=True)
        if results[split]["auroc"] is not None:
            print(f'    AUROC      = {results[split]["auroc"]:.4f}', flush=True)
        print(f'    composite  = {results[split]["composite"]:.4f}', flush=True)

    samples_root = Path(args.samples_dir)
    if samples_root.exists():
        print(f'\n=== Evaluating {samples_root} ===', flush=True)
        results['samples'] = evaluate_samples_folder(sess, samples_root, args.image_size, args.threshold)
        for cls, rows in results['samples'].items():
            n_pred_tumor = sum(1 for r in rows if r['predicted_tumor'])
            print(f'  {cls}: {n_pred_tumor}/{len(rows)} predicted as tumor', flush=True)

    out_json = out_dir / 'eval_full.json'
    out_json.write_text(json.dumps(results, indent=2, default=float))
    print(f'\nFull results saved to {out_json}', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
