"""Foundation-model linear-probe AUC on the 148-sample OOD bench.

Phase 2 of the v9c plan. Compares candidate medical-imaging foundation
models by extracting their off-the-shelf embeddings on our 148 OOD
samples, then training a linear logistic-regression probe to discriminate
tumor vs healthy. The winner becomes the frozen backbone for the
normative-JEPA head in Phase 3.

Candidates evaluated:
  - BiomedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
    Trained on 15M biomedical image-text pairs from PubMed. Specifically
    biomedical, includes some MRI in its training set.

  - RAD-DINO (microsoft/rad-dino)
    Trained on ~838k chest X-rays with DINOv2 SSL. Domain is X-ray not
    MRI, included as a sanity-check baseline — should be WORSE than
    BiomedCLIP if our hypothesis holds.

  - DINOv2 (facebook/dinov2-base)
    Generic natural-image SSL backbone. Lowest-prior baseline; if it
    beats the medical models, that means our task isn't actually
    domain-specific enough to need a medical foundation model.

Metric: AUC and 5-fold stratified CV accuracy of logistic regression
on the pooled embeddings (label = tumor vs healthy).

Outputs samples/ood/foundation_probe_results.csv.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval_ood_cascade import GT as _GT
# Ensure the IXI2D healthy cohort added during Phase 1 is included, and the
# Navoneel healthy cohort added during Phase 2 (so Navoneel is no longer
# source-monolithic and LOSO AUC becomes computable).
GT = dict(_GT)
GT.setdefault('healthy_ixi2d', 'no_tumor')
GT.setdefault('healthy_navoneel', 'no_tumor')

# Folder name -> logical source group. Used for leave-one-source-out CV
# so the probe can't cheat by recognising scanner / preprocessing
# signatures. Folders not listed here use their own name as the group.
SOURCE_GROUPS = {
    'tumor_binary_navoneel_via_miladfa7': 'navoneel',
    'healthy_navoneel': 'navoneel',
    # all other folders map to themselves (one folder = one source)
}


def _source_group(folder_name: str) -> str:
    return SOURCE_GROUPS.get(folder_name, folder_name)


SAMPLES_DIR = ROOT / 'samples' / 'ood'

# Define candidate models. Each entry tells us how to load + run inference.
# We use HuggingFace transformers for the BiomedCLIP / RAD-DINO / DINOv2.
CANDIDATES = [
    {
        'name': 'biomedclip',
        'hf_id': 'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        'loader': 'open_clip',   # BiomedCLIP ships via open_clip, not standard transformers
        'image_size': 224,
    },
    {
        'name': 'rad-dino',
        'hf_id': 'microsoft/rad-dino',
        'loader': 'transformers_dino',
        'image_size': 518,   # ViT-B/14 default
    },
    {
        'name': 'dinov2-base',
        'hf_id': 'facebook/dinov2-base',
        'loader': 'transformers_dino',
        'image_size': 224,
    },
]


def _load_open_clip(hf_id: str, device: str):
    try:
        import open_clip
    except ImportError:
        return None, 'pip install open_clip_torch'
    model, _, preprocess = open_clip.create_model_and_transforms(
        f'hf-hub:{hf_id}')
    model = model.to(device).eval()
    return (model, preprocess), None


def _load_transformers_dino(hf_id: str, device: str):
    try:
        from transformers import AutoModel, AutoImageProcessor
    except ImportError:
        return None, 'pip install transformers'
    try:
        proc = AutoImageProcessor.from_pretrained(hf_id)
        model = AutoModel.from_pretrained(hf_id).to(device).eval()
    except Exception as exc:
        return None, f'load failed: {type(exc).__name__}: {exc}'
    return (model, proc), None


@torch.no_grad()
def _embed_open_clip(loader_state, img_pil, device):
    model, preprocess = loader_state
    x = preprocess(img_pil).unsqueeze(0).to(device)
    feat = model.encode_image(x)
    return feat.squeeze(0).cpu().numpy()


@torch.no_grad()
def _embed_dino(loader_state, img_pil, device):
    model, proc = loader_state
    x = proc(images=img_pil, return_tensors='pt').pixel_values.to(device)
    out = model(pixel_values=x, output_hidden_states=False)
    # Use the CLS token (first patch of last hidden state) as embedding
    feat = out.last_hidden_state[:, 0, :]
    return feat.squeeze(0).cpu().numpy()


def _stratified_auc_probe(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    """Logistic regression with stratified k-fold CV; report AUC + accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, accuracy_score

    if len(np.unique(y)) < 2:
        return {'auc_mean': float('nan'), 'auc_std': float('nan'),
                 'acc_mean': float('nan')}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs, accs = [], []
    for train_idx, test_idx in skf.split(X, y):
        Xtr, ytr = X[train_idx], y[train_idx]
        Xte, yte = X[test_idx], y[test_idx]
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
        prob = clf.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(yte, prob))
        accs.append(accuracy_score(yte, clf.predict(Xte)))
    return {
        'auc_mean': float(np.mean(aucs)), 'auc_std': float(np.std(aucs)),
        'acc_mean': float(np.mean(accs)),
    }


def _leave_source_out_probe(X: np.ndarray, y: np.ndarray,
                              sources: list[str]) -> dict:
    """Leave-one-source-out CV. Holds out ENTIRE source per fold so the
    probe cannot learn 'which dataset this image is from' instead of
    'does this image contain a tumor'.

    If stratified-K-fold AUC is ~1.0 but LOSO AUC is ~0.5, the high
    stratified-K-fold AUC was a source-confound artifact.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    src_arr = np.array(sources)
    unique_sources = sorted(set(sources))
    if len(unique_sources) < 2:
        return {'loso_auc_mean': float('nan'), 'per_source': {}}
    aucs = {}
    for held_out in unique_sources:
        test_idx = np.where(src_arr == held_out)[0]
        train_idx = np.where(src_arr != held_out)[0]
        Xtr, ytr = X[train_idx], y[train_idx]
        Xte, yte = X[test_idx], y[test_idx]
        # If the held-out source is monolithic (all same label) we can't
        # compute AUC on it — record N/A and skip
        if len(np.unique(yte)) < 2 or len(np.unique(ytr)) < 2:
            aucs[held_out] = float('nan')
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
        prob = clf.predict_proba(Xte)[:, 1]
        aucs[held_out] = float(roc_auc_score(yte, prob))
    valid = [v for v in aucs.values() if not np.isnan(v)]
    return {
        'loso_auc_mean': float(np.mean(valid)) if valid else float('nan'),
        'per_source': aucs,
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[init] device={device}')

    samples = sorted(p for p in SAMPLES_DIR.rglob('*')
                      if p.suffix.lower() in ('.png', '.jpg', '.jpeg')
                      and p.parent.name in GT)
    print(f'[init] {len(samples)} OOD samples')
    y = np.array([1 if GT[p.parent.name] == 'tumor' else 0 for p in samples])
    sources = [_source_group(p.parent.name) for p in samples]
    print(f'[init] tumor={int(y.sum())}  healthy={int((1-y).sum())}')
    print(f'[init] unique source groups (for LOSO): {sorted(set(sources))}')
    # Tally per source group for sanity
    from collections import Counter
    for src in sorted(set(sources)):
        idx = [i for i, s in enumerate(sources) if s == src]
        ys = y[idx]
        print(f'   {src:24s}  n={len(idx):3d}  tumor={int(ys.sum())}  healthy={int((1-ys).sum())}')

    results = []
    for cand in CANDIDATES:
        print(f'\n=== {cand["name"]} ({cand["hf_id"]}) ===')
        t0 = time.perf_counter()
        if cand['loader'] == 'open_clip':
            ls, err = _load_open_clip(cand['hf_id'], device)
            embed_fn = _embed_open_clip
        elif cand['loader'] == 'transformers_dino':
            ls, err = _load_transformers_dino(cand['hf_id'], device)
            embed_fn = _embed_dino
        else:
            ls, err = None, f'unknown loader {cand["loader"]!r}'
        if err:
            print(f'  [skip] {err}')
            results.append({'name': cand['name'], 'error': err,
                            'auc_mean': None, 'acc_mean': None})
            continue
        print(f'  loaded in {time.perf_counter()-t0:.1f}s; embedding {len(samples)} samples ...')

        feats: list[np.ndarray] = []
        bad = 0
        te = time.perf_counter()
        for i, p in enumerate(samples):
            try:
                img = Image.open(p).convert('RGB')
                f = embed_fn(ls, img, device)
                feats.append(f.astype(np.float32))
            except Exception as exc:
                bad += 1
                if bad <= 3:
                    print(f'    embed fail on {p.name}: {type(exc).__name__}')
                feats.append(np.zeros(768, dtype=np.float32))   # placeholder
        embed_time = time.perf_counter() - te
        print(f'  embedded {len(feats)} ({bad} fails) in {embed_time:.0f}s '
              f'({embed_time/len(feats)*1000:.0f} ms/sample)')

        # Stack and probe
        X = np.stack(feats, axis=0)
        # Some models have variable feature size; pad/truncate to a fixed
        # consistent dim (use the actual returned dim of this model)
        D = X.shape[1]
        print(f'  feature dim = {D}')
        stats = _stratified_auc_probe(X, y, n_splits=5)
        print(f'  stratified-5-fold AUC = {stats["auc_mean"]:.4f} ± {stats["auc_std"]:.4f}   '
              f'acc = {stats["acc_mean"]:.4f}')
        # CRITICAL: leave-one-source-out probe. If stratified AUC is ~1.0
        # but LOSO AUC collapses, the probe was learning source ID not tumor.
        loso = _leave_source_out_probe(X, y, sources)
        print(f'  leave-source-out  AUC = {loso["loso_auc_mean"]:.4f}  '
              f'(by source: {loso["per_source"]})')
        results.append({
            'name': cand['name'], 'hf_id': cand['hf_id'],
            'feature_dim': D,
            'auc_mean': stats['auc_mean'], 'auc_std': stats['auc_std'],
            'acc_mean': stats['acc_mean'],
            'loso_auc': loso['loso_auc_mean'],
            'embed_time_total_s': round(embed_time, 1),
            'ms_per_sample': round(embed_time / max(len(feats), 1) * 1000, 1),
        })
        # Free GPU mem before next model
        del ls
        if device == 'cuda':
            torch.cuda.empty_cache()

    # ---- baselines for context ----
    print('\n=== for reference (from earlier audits) ===')
    print('  v9b JEPA (from scratch)      AUC = 0.564   [our baseline]')
    print('  symmetry geometry             AUC = 0.653')
    print('  DDPM residual                 AUC ~ 0.706')
    print('  v8 segmentation               (not directly comparable, mask-based)')

    # Persist
    out_csv = SAMPLES_DIR / 'foundation_probe_results.csv'
    if results:
        fields = ['name', 'hf_id', 'feature_dim', 'auc_mean', 'auc_std',
                   'acc_mean', 'embed_time_total_s', 'ms_per_sample', 'error']
        with out_csv.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in results:
                w.writerow({k: r.get(k) for k in fields})
        print(f'\n[csv] {out_csv}')


if __name__ == '__main__':
    main()
