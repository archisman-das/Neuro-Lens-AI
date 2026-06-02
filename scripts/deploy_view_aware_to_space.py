"""Deploy the view-aware OOD cascade to the production HF Space.

What this ships:
  - dashboard.py (modified: step 2c view-aware suppression override)
  - src/research/view_router.py (NEW dependency imported by dashboard.py)

Why it's safe:
  - The dashboard.py import is guarded by try/except, so if the Space
    image were missing view_router.py the pipeline would silently fall
    back to the original behaviour. Uploading view_router.py FIRST then
    dashboard.py removes that risk entirely.
  - VIEW_AWARE_CASCADE_DISABLE=1 reverts to the prior behaviour
    instantly without a redeploy.
  - HfApi.upload_file is atomic per file and triggers an automatic
    Space rebuild.

Usage:
    python scripts/deploy_view_aware_to_space.py
HF_TOKEN env var must have WRITE scope on Tubai01/neurolens.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REPO_ID = 'Tubai01/neurolens-ai'
REPO_TYPE = 'space'

# Order matters: ship the dependency BEFORE the importer.
UPLOAD_ORDER = [
    ('src/research/view_router.py', 'src/research/view_router.py'),
    ('dashboard.py',                'dashboard.py'),
]


def main():
    token = os.environ.get('HF_TOKEN')
    if not token:
        sys.exit('ERROR: HF_TOKEN env var missing. Need a WRITE-scoped token.')

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit('ERROR: pip install huggingface_hub')

    api = HfApi(token=token)
    print(f'[deploy] repo={REPO_ID} type={REPO_TYPE}')

    for local_rel, repo_rel in UPLOAD_ORDER:
        local = ROOT / local_rel
        if not local.exists():
            sys.exit(f'ERROR: local file missing: {local}')
        size_kb = local.stat().st_size / 1024
        print(f'[upload] {local_rel} -> {REPO_ID}:{repo_rel} ({size_kb:.1f} KB)', flush=True)
        t0 = time.perf_counter()
        try:
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=repo_rel,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                commit_message=(
                    f'feat: view-aware OOD cascade '
                    f'({"view_router module" if "view_router" in repo_rel else "dashboard step 2c override"})'
                ),
            )
        except Exception as exc:
            sys.exit(f'  FAILED: {type(exc).__name__}: {exc}')
        print(f'          done in {time.perf_counter()-t0:.1f}s')

    print()
    print('=== deploy summary ===')
    print(f'  repo:    {REPO_ID}')
    print(f'  files:   {len(UPLOAD_ORDER)}')
    print(f'  next:    Space will rebuild automatically (Docker SDK).')
    print(f'  url:     https://huggingface.co/spaces/{REPO_ID}')
    print()
    print('Post-deploy checks:')
    print(f'  1. Watch build logs: https://huggingface.co/spaces/{REPO_ID}?logs=container')
    print(f'  2. Liveness:        GET /health')
    print(f'  3. Sanity probe:    POST /explain with a sample, check the JSON')
    print(f'                       payload contains seg["view_detection"].')
    print(f'  4. Off-switch test (if needed): set VIEW_AWARE_CASCADE_DISABLE=1 in')
    print(f'     Space Settings > Variables, redeploy.')


if __name__ == '__main__':
    main()
