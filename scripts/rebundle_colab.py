"""Rebuild colab_bundle.zip with ALL transitive deps the v9b trainers need.

Previous miss (2026-06-02): train_v9b_stage1_jepa.py imports `_atomic_save`
from train_segmentation_v7.py, and v9b_model.py imports
`synthetic_brain_sdf_template` from research/geometric_prior.py. Neither
was in the first bundle, so Stage 1 crashed on Colab with
`ModuleNotFoundError: No module named 'src.train_segmentation_v7'`.

This script is the source of truth for what goes in the bundle.
"""
from __future__ import annotations

import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'colab_bundle.zip'

# Top-level colab_bundle/ files
BUNDLE_FILES = [
    'colab_bundle/README_COLAB.md',
    'colab_bundle/__init__.py',
    'colab_bundle/requirements_colab.txt',
    'colab_bundle/v9b_colab_train.ipynb',
]

# src/ files unzipped into /content/neurolens/src/ on the Colab VM. After
# the 2026-06-02 refactor, v9b stages import `atomic_save` from the new
# `src/checkpoint_utils.py` (no further local imports), so the v5+v7
# trainer chain is no longer needed in this bundle — only the actual v9b
# code + its research modules.
SRC_FILES = [
    'src/__init__.py',
    'src/utils.py',
    'src/checkpoint_utils.py',
    'src/train_v9b_stage1_jepa.py',
    'src/train_v9b_stage2.py',
    'src/train_v9b_andi_ddpm.py',     # NEW (June 2026): proper ANDi
                                       # DDPM training with pyramidal noise.
    'src/train_v9c_stage1.py',        # NEW (June 2026): v9c JEPA predictor
                                       # on frozen DINOv2 backbone.
    'src/v9b_inference.py',
    'src/research/__init__.py',
    'src/research/jepa.py',
    'src/research/jepa_conformal.py',
    'src/research/latent_diffusion_decoder.py',
    'src/research/sdf_geometric_tower.py',
    'src/research/symmetry_geometry.py',  # NEW: deterministic symmetry
                                          # geometry score, replaces SDF.
    'src/research/pyramidal_noise.py',    # NEW: ANDi pyramidal noise gen.
    'src/research/andi_inference.py',     # NEW: ANDi inference aggregation.
    'src/research/two_tower_anomaly.py',
    'src/research/geometric_prior.py',
    'src/research/mesh_extraction.py',
    'src/research/mni152_registration.py',
    'src/research/v9b_model.py',
    'src/research/v9b_advisory.py',       # end-to-end advisory wrapper.
    'src/research/v9c_dinov2_jepa.py',    # NEW (June 2026): v9c model
                                          # (frozen DINOv2 + JEPA predictor).
]


def main():
    t0 = time.perf_counter()
    if OUT.exists():
        OUT.unlink()
    missing: list[str] = []
    with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for rel in BUNDLE_FILES + SRC_FILES:
            p = ROOT / rel
            if not p.exists():
                missing.append(rel)
                continue
            # Inside the zip: BUNDLE_FILES go to the top level (strip
            # "colab_bundle/" prefix), SRC_FILES keep their src/ path so
            # the notebook can unzip and `python src/train_v9b_*` works.
            if rel.startswith('colab_bundle/'):
                arc = rel[len('colab_bundle/'):]
            else:
                arc = rel
            zf.write(p, arcname=arc)
    if missing:
        print('WARNING: missing source files (not bundled):')
        for m in missing:
            print(f'  - {m}')
    size_kb = OUT.stat().st_size / 1024
    print(f'\n[done] colab_bundle.zip = {size_kb:.1f} KB '
          f'({len(BUNDLE_FILES) + len(SRC_FILES) - len(missing)} files) '
          f'in {time.perf_counter()-t0:.1f}s')

    # Verify checkpoint_utils + geometric_prior are in, and that the v7
    # trainer (now removed) is NOT — confirms the bundle slimming worked.
    with zipfile.ZipFile(OUT) as zf:
        names = set(zf.namelist())
        for need in ('src/checkpoint_utils.py', 'src/research/geometric_prior.py'):
            print(f'  {"[OK] " if need in names else "[FAIL]"} {need}')
        for gone in ('src/train_segmentation_v5.py', 'src/train_segmentation_v7.py'):
            print(f'  {"[OK (removed)]" if gone not in names else "[STILL IN BUNDLE]"} {gone}')


if __name__ == '__main__':
    main()
