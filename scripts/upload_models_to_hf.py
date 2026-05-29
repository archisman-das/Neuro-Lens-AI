"""Upload local ONNX weights to a HuggingFace Model repository.

Why this exists separately from the Space:
  - HuggingFace Spaces free tier caps total repo size at 1 GB. Our 5 ONNX
    models total ~440 MB which is close to the cap and crowds out everything
    else.
  - HuggingFace Model repos have a much larger free quota and are the
    canonical place to publish trained weights.
  - The dashboard's _ensure_onnx_models_downloaded() (see dashboard.py)
    pulls these files at first boot and caches them locally, so the Space
    image stays small while still serving the same models.

Usage:
  # Once: create the Model repo on HF (web UI is fine):
  #   https://huggingface.co/new -> Type: Model, Name: neurolens-models
  # Then run:
  python scripts/upload_models_to_hf.py --repo Tubai01/neurolens-models
  # The HF_TOKEN env var must be set to a token with WRITE scope.

Re-running is safe: it only uploads files whose local sha256 differs from
what's already on the Hub.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# (local path relative to repo root, target path inside the Model repo)
ARTIFACT_MAP = [
    ('segmentation_artifacts/attention_unet_v3/best_model.onnx',
     'attention_unet_v3/best_model.onnx'),
    ('segmentation_artifacts/attention_unet_t1c/best_model.onnx',
     'attention_unet_t1c/best_model.onnx'),
    ('real_eval_current/cnn/best_weights.onnx',
     'cnn/best_weights.onnx'),
    ('real_eval_current/transfer/best_weights.onnx',
     'transfer/best_weights.onnx'),
    ('real_eval_current/vit/best_weights.onnx',
     'vit/best_weights.onnx'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True,
                     help='HF Model repo id, e.g. Tubai01/neurolens-models')
    ap.add_argument('--token', default=None,
                     help='HF write token. Defaults to HF_TOKEN env var.')
    ap.add_argument('--create-if-missing', action='store_true',
                     help='Auto-create the Model repo if it doesn\'t exist.')
    ap.add_argument('--commit-message', default='Upload NeuroLens ONNX models')
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print('ERROR: huggingface_hub not installed. Run: pip install huggingface_hub')
        sys.exit(2)

    token = args.token or os.environ.get('HF_TOKEN')
    if not token:
        print('ERROR: no token. Pass --token or set HF_TOKEN env var (write scope).')
        sys.exit(2)

    api = HfApi(token=token)

    if args.create_if_missing:
        try:
            api.create_repo(repo_id=args.repo, repo_type='model', exist_ok=True)
            print(f'[ok] repo exists or created: {args.repo}')
        except Exception as exc:
            print(f'[warn] create_repo: {exc}')

    # Upload each file individually so a failure on one doesn't abort the rest.
    # The Hub deduplicates by sha; re-running with no changes is fast.
    uploaded = 0
    skipped_missing = 0
    failed: list[tuple[str, str]] = []
    total_bytes = 0
    t_start = time.perf_counter()

    for local_rel, repo_rel in ARTIFACT_MAP:
        local = ROOT / local_rel
        if not local.exists():
            print(f'[skip] {local_rel} not found locally')
            skipped_missing += 1
            continue
        size_mb = local.stat().st_size / 1e6
        total_bytes += local.stat().st_size
        print(f'[upload] {local_rel} -> {args.repo}:{repo_rel} ({size_mb:.1f} MB) ...', flush=True)
        try:
            t0 = time.perf_counter()
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=repo_rel,
                repo_id=args.repo,
                repo_type='model',
                commit_message=f'{args.commit_message}: {repo_rel}',
            )
            print(f'           done in {time.perf_counter()-t0:.1f}s')
            uploaded += 1
        except Exception as exc:
            print(f'           FAILED: {type(exc).__name__}: {exc}')
            failed.append((local_rel, str(exc)))

    elapsed = time.perf_counter() - t_start
    print('\n=== upload summary ===')
    print(f'  repo:     {args.repo}')
    print(f'  uploaded: {uploaded}')
    print(f'  skipped:  {skipped_missing} (file not found locally)')
    print(f'  failed:   {len(failed)}')
    print(f'  bytes:    {total_bytes/1e6:.1f} MB')
    print(f'  elapsed:  {elapsed:.1f}s')
    if failed:
        print('\nfailures:')
        for f, e in failed:
            print(f'  - {f}: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
