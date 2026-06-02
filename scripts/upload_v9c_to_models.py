"""Upload v9c last.pt to the HF Models repo (Tubai01/neurolens-models).

Run once after v9c training finishes. The Space pulls these weights via
dashboard.py's _ensure_onnx_models_downloaded() when V9C_DOWNLOAD=1.

Usage:
    python scripts/upload_v9c_to_models.py
HF_TOKEN must have WRITE scope on Tubai01/neurolens-models.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / 'v9b_artifacts' / 'v9c_stage1' / 'last.pt'
REPO_ID = 'Tubai01/neurolens-models'
REPO_PATH = 'v9c_stage1/last.pt'


def main():
    token = os.environ.get('HF_TOKEN')
    if not token:
        sys.exit('ERROR: HF_TOKEN env var missing. Need a WRITE-scoped token.')
    if not LOCAL.exists():
        sys.exit(f'ERROR: weights not at {LOCAL}')
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    size_mb = LOCAL.stat().st_size / 1e6
    print(f'[upload] {LOCAL.name} ({size_mb:.1f} MB) -> {REPO_ID}:{REPO_PATH}', flush=True)
    t0 = time.perf_counter()
    api.upload_file(
        path_or_fileobj=str(LOCAL),
        path_in_repo=REPO_PATH,
        repo_id=REPO_ID,
        repo_type='model',
        commit_message='Add v9c JEPA-on-DINOv2 predictor weights (loss=0.07 @ 50 ep)',
    )
    print(f'  done in {time.perf_counter()-t0:.1f}s')


if __name__ == '__main__':
    main()
