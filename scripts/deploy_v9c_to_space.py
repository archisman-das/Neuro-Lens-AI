"""Deploy the v9c ensemble (v9c + v8 + symmetry) to the production HF Space.

What this ships:
  - src/research/v9c_dinov2_jepa.py   (NEW dependency, must go first)
  - src/research/v9b_advisory.py      (rewritten with v9c-aware ensemble
                                       + high_recall as the safety-first
                                       default + review_recommended flag)
  - dashboard.py                       (V9C_DOWNLOAD=1 hook for weight
                                       fetch + escalation when advisory
                                       disagrees with v8 in the
                                       more-sensitive direction)

Why it's safe:
  - All v9c logic is opt-in via V9C_ENABLE=1; without that env var, the
    advisory falls back to the prior (v8 AND symmetry) ensemble and the
    dashboard behaves exactly like the 2026-06-02 deploy.
  - Order: ship v9c_dinov2_jepa.py FIRST (so v9b_advisory.py's import
    can resolve), then the advisory, then dashboard.
  - To activate the v9c high-recall ensemble on the Space:
      1. Run scripts/upload_v9c_to_models.py once to push last.pt to
         the Models repo.
      2. Set Space env vars:  V9C_DOWNLOAD=1, V9C_ENABLE=1.
         (Optional: V9B_OPERATING_POINT=high_recall  — this is the
          default anyway, but explicit is good.)
      3. Space rebuilds and downloads weights on first boot.

Usage:
    python scripts/deploy_v9c_to_space.py
HF_TOKEN env var must have WRITE scope on Tubai01/neurolens-ai.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REPO_ID = 'Tubai01/neurolens-ai'
REPO_TYPE = 'space'

# Order matters: dependencies first so importers don't crash at startup.
# As of 2026-06-03c the dashboard refactor also ships llm_explain + frontend
# changes that surface the 4-signal verdict as the primary UI element.
UPLOAD_ORDER = [
    ('src/research/v9c_dinov2_jepa.py',          'src/research/v9c_dinov2_jepa.py'),
    ('src/research/latent_diffusion_decoder.py', 'src/research/latent_diffusion_decoder.py'),
    ('src/research/andi_inference.py',           'src/research/andi_inference.py'),
    ('src/research/pyramidal_noise.py',          'src/research/pyramidal_noise.py'),
    ('src/research/v9b_advisory.py',             'src/research/v9b_advisory.py'),
    ('src/llm_explain.py',                        'src/llm_explain.py'),
    ('dashboard.py',                              'dashboard.py'),
    ('web_dashboard/index.html',                  'web_dashboard/index.html'),
    ('web_dashboard/app.js',                      'web_dashboard/app.js'),
    ('web_dashboard/openapi.yml',                 'web_dashboard/openapi.yml'),
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
                    f'feat(v9c): ship JEPA-on-DINOv2 ensemble '
                    f'({"v9c module" if "v9c_dinov2" in repo_rel else "advisory" if "advisory" in repo_rel else "dashboard hook"})'
                ),
            )
        except Exception as exc:
            sys.exit(f'  FAILED: {type(exc).__name__}: {exc}')
        print(f'          done in {time.perf_counter()-t0:.1f}s')

    print()
    print('=== deploy summary ===')
    print(f'  repo:    {REPO_ID}')
    print(f'  files:   {len(UPLOAD_ORDER)}')
    print(f'  next:    Set V9C_DOWNLOAD=1 + V9C_ENABLE=1 in Space env vars')
    print(f'           (Settings > Variables and secrets) to activate v9c.')
    print(f'  url:     https://huggingface.co/spaces/{REPO_ID}')


if __name__ == '__main__':
    main()
