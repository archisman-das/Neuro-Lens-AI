"""Build the v2 CrossJEPA dataset bundle — adds IXI to the healthy mix
so the model sees TWO preprocessing pipelines during training (radiata-ai's
MNI-affine and IXI's Imperial College pipeline).

Layout inside the zip (changed from v1):

  healthy/radiata/<study>/<sub-id>/<scan>.nii.gz   - 800 radiata-ai T1 (MNI-affine)
  healthy/ixi/<scan>.nii.gz                         - 50 IXI T1 (different pipeline)
  brats/<patient-id>/<modality>.nii.gz              - 1251 BraTS-2021 4-modality

Total: ~17 GB (vs 15.15 GB for v1). The IXI addition adds ~2.3 GB but
gives the trained encoder a real second preprocessing distribution to
learn invariance against.

The Method 1 trainer's `--scans_glob` should now use:
    --scans_glob '/content/data/healthy/**/*.nii.gz'
which captures both the radiata and ixi subdirs.

Run:
    python scripts/build_crossjepa_data_v2.py
"""
from __future__ import annotations

import glob
import time
import zipfile
from pathlib import Path

HEALTHY_RADIATA_ROOT = ('c:/Users/anish/.cache/huggingface/hub/'
                         'datasets--radiata-ai--brain-structure/snapshots/'
                         'aea73e2a9734955407eb8bcf2325b479a5f39144')
HEALTHY_IXI_GLOB = 'd:/datasets/ixi_heldout/ixi_preprocessed/IXI*-T1.nii.gz'
BRATS_ROOT = Path('d:/datasets/brats2021')
OUT_PATH = Path('d:/dataset_bundle/crossjepa_data_v2.zip')

README = """# v9c CrossJEPA dataset bundle — v2 (multi-pipeline)

Adds IXI (Imperial College preprocessing) to the healthy pool so the
Method 1 encoder sees TWO distinct preprocessing distributions during
training. Combined with the new `--augment` flag in
`src/train_v9c_method1_vol2slice.py`, this addresses the failure mode
caught by the IXI held-out eval (raw AUC = 0.00 inverted — the v1
model had memorized radiata-ai's preprocessing rather than the
healthy-brain manifold).

## Contents

| Cohort | Path inside zip | Patients | Pipeline |
|---|---|---:|---|
| Healthy (radiata-ai) | `healthy/radiata/{DLBS,NKI-RS,OASIS-1,OASIS-2}/sub-*/ses-*/anat/*.nii.gz` | 800 | MNI-affine reg, brain-extracted |
| Healthy (IXI) | `healthy/ixi/IXI*-T1.nii.gz` | 50 | Imperial College pipeline (different) |
| Tumor (BraTS 2021) | `brats/BraTS2021_NNNNN/*.nii.gz` | 1251 (1248 fully-modal) | BraTS preprocessing (skull-strip + reg) |

## On Colab — Method 1 retrain with augmentation

```python
!unzip -q /content/drive/MyDrive/crossjepa_data_v2.zip -d /content/data/

!python src/train_v9c_method1_vol2slice.py \\
    --scans_glob '/content/data/healthy/**/*.nii.gz' \\
    --v8_ckpt /content/v8_best_micro.pt \\
    --output_dir /content/drive/MyDrive/v9c_crossjepa_method1_v2 \\
    --volume_size 144 192 192 --in_channels 1 \\
    --batch_size 2 --epochs 30 --lr 2e-4 --amp --resume auto \\
    --augment \\
    --aug_intensity_jitter 0.20 --aug_contrast_jitter 0.30 \\
    --aug_gamma_jitter 0.30 --aug_bias_field_strength 0.15 \\
    --aug_noise_std 0.02 --aug_renormalize_pct 0.5
```

The `--augment` flag enables intensity / contrast / gamma / bias-field
augmentation on each volume before slicing. Each transform fires with
prob 0.7 per volume; the chain forces the encoder to learn
preprocessing-pipeline-invariant features.
"""


def main():
    t0 = time.perf_counter()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        print(f'[warn] {OUT_PATH} exists ({OUT_PATH.stat().st_size/1e9:.1f} GB); deleting')
        OUT_PATH.unlink()

    radiata_files = sorted(
        glob.glob(f'{HEALTHY_RADIATA_ROOT}/*/sub-*/ses-*/anat/*.nii.gz'))
    ixi_files = sorted(glob.glob(HEALTHY_IXI_GLOB))
    brats_files = sorted(BRATS_ROOT.rglob('*.nii.gz'))
    print(f'[init] {len(radiata_files)} radiata healthy + {len(ixi_files)} IXI healthy '
          f'+ {len(brats_files)} BraTS NIfTI')

    written = 0; bytes_written = 0
    total = len(radiata_files) + len(ixi_files) + len(brats_files)
    last_report = time.perf_counter()

    with zipfile.ZipFile(OUT_PATH, 'w', zipfile.ZIP_STORED,
                          allowZip64=True) as zf:
        zf.writestr('README.md', README)
        # 1) Radiata healthy under healthy/radiata/
        for src in radiata_files:
            rel = src.split('aea73e2a9734955407eb8bcf2325b479a5f39144')[-1].lstrip('/\\')
            arc = f'healthy/radiata/{rel.replace(chr(92), "/")}'
            try:
                zf.write(src, arcname=arc)
                bytes_written += Path(src).stat().st_size
                written += 1
            except Exception as exc:
                print(f'  [warn] skip {src}: {exc}')
            if time.perf_counter() - last_report > 10:
                last_report = time.perf_counter()
                print(f'  [progress] {written}/{total} files  '
                      f'{bytes_written/1e9:.1f} GB in  '
                      f'{time.perf_counter()-t0:.0f}s')
        # 2) IXI healthy under healthy/ixi/
        for src in ixi_files:
            arc = f'healthy/ixi/{Path(src).name}'
            try:
                zf.write(src, arcname=arc)
                bytes_written += Path(src).stat().st_size
                written += 1
            except Exception as exc:
                print(f'  [warn] skip {src}: {exc}')
        # 3) BraTS under brats/
        for src in brats_files:
            arc = f'brats/{src.relative_to(BRATS_ROOT).as_posix()}'
            try:
                zf.write(src, arcname=arc)
                bytes_written += src.stat().st_size
                written += 1
            except Exception as exc:
                print(f'  [warn] skip {src}: {exc}')
            if time.perf_counter() - last_report > 10:
                last_report = time.perf_counter()
                print(f'  [progress] {written}/{total} files  '
                      f'{bytes_written/1e9:.1f} GB in  '
                      f'{time.perf_counter()-t0:.0f}s')

    print(f'\n[done] {written}/{total} files in {(time.perf_counter()-t0)/60:.1f} min')
    print(f'[done] zip = {OUT_PATH.stat().st_size/1e9:.2f} GB at {OUT_PATH}')


if __name__ == '__main__':
    main()
