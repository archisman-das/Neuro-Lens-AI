"""Build the v9c CrossJEPA dataset bundle — one zip with EVERYTHING needed
for both Method 1 and Method 2 training.

Layout inside the zip:
  healthy/<study>/<sub-id>/<scan>.nii.gz   - 800 radiata-ai T1 volumes
                                              (Method 1 normative pretrain)
  brats/<patient-id>/<modality>.nii.gz     - 1251 BraTS-2021 4-modality
                                              (Method 1 anomaly eval +
                                               Method 2 training)
  README.md                                 - manifest + how to unzip on Colab

NIfTI files are already gzipped (.nii.gz). ZIP-stored without further
compression (STORE level) — saves CPU time at zip/unzip with negligible
size penalty (~0.1%).

Output: D:/dataset_bundle/crossjepa_data.zip   (~13-14 GB)

Usage:
    python scripts/build_crossjepa_dataset_bundle.py
"""
from __future__ import annotations

import glob
import time
import zipfile
from pathlib import Path

HEALTHY_GLOB = ('c:/Users/anish/.cache/huggingface/hub/'
                'datasets--radiata-ai--brain-structure/snapshots/'
                'aea73e2a9734955407eb8bcf2325b479a5f39144')
BRATS_ROOT = Path('d:/datasets/brats2021')
OUT_PATH = Path('d:/dataset_bundle/crossjepa_data.zip')

README = """# v9c CrossJEPA dataset bundle

Single zip with all 3D MRI data needed for CrossJEPA Method 1 + Method 2
training. Upload once to Drive, unzip in Colab, train.

## Contents

| Cohort | Path inside zip | Patients | License |
|---|---|---:|---|
| Healthy T1 (radiata-ai/brain-structure) | `healthy/{DLBS,NKI-RS,OASIS-1,OASIS-2}/sub-*/ses-*/anat/*.nii.gz` | 800 | per source dataset (CC-BY / restricted research use) |
| Tumor 4-modality (BraTS 2021) | `brats/BraTS2021_NNNNN/*.nii.gz` | 1251 (1248 fully-modal) | BraTS 2021 Challenge — research use |

## On Colab

```python
from google.colab import drive
drive.mount('/content/drive')
!unzip -q /content/drive/MyDrive/crossjepa_data.zip -d /content/data/
# Method 1 (healthy normative pretrain):
!python src/train_v9c_method1_vol2slice.py \\
    --scans_glob '/content/data/healthy/**/*.nii.gz' \\
    --v8_ckpt /content/v8_best_micro.pt \\
    --output_dir /content/drive/MyDrive/v9c_crossjepa_method1 \\
    --volume_size 144 192 192 --in_channels 1 \\
    --batch_size 2 --epochs 30 --lr 2e-4 --amp --resume auto

# Method 2 (4-modality CrossJEPA, after per-modality teachers exist):
!python src/train_v9c_method2_mod2mod.py \\
    --brats_root /content/data/brats \\
    --teachers_dir /content/teachers \\
    --output_dir /content/drive/MyDrive/v9c_crossjepa_method2 \\
    --image_size 256 --batch_size 8 --epochs 30 --lr 2e-4 --amp
```

## Provenance

- Healthy: HuggingFace dataset `radiata-ai/brain-structure`, snapshot
  `aea73e2a9734955407eb8bcf2325b479a5f39144`, MNI-affine registered T1
- Tumor: HuggingFace dataset `rocky93/BraTS_segmentation`, raw BraTS 2021
  challenge volumes (4 modalities + tumor segmentation labels)
"""


def main():
    t0 = time.perf_counter()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        print(f'[warn] {OUT_PATH} exists ({OUT_PATH.stat().st_size/1e9:.1f} GB); deleting')
        OUT_PATH.unlink()

    healthy_files = sorted(
        glob.glob(f'{HEALTHY_GLOB}/*/sub-*/ses-*/anat/*.nii.gz'))
    print(f'[init] {len(healthy_files)} healthy T1 NIfTI to bundle')

    brats_files = sorted(BRATS_ROOT.rglob('*.nii.gz'))
    print(f'[init] {len(brats_files)} BraTS-2021 NIfTI to bundle')

    total = len(healthy_files) + len(brats_files)
    print(f'[init] {total} total files -> {OUT_PATH}')

    # ZIP_STORED (no compression) — NIfTI files are already gzipped; using
    # DEFLATE here would burn CPU for ~0.1% saving.
    written = 0
    bytes_written = 0
    last_report = time.perf_counter()
    with zipfile.ZipFile(OUT_PATH, 'w', zipfile.ZIP_STORED,
                          allowZip64=True) as zf:
        zf.writestr('README.md', README)

        # 1) Healthy — flatten the radiata-ai cache snapshot path so the
        #    in-zip path is `healthy/<study>/sub-*/ses-*/anat/*.nii.gz`
        for src in healthy_files:
            # Strip everything up to and including the snapshot hash
            rel = src.split('aea73e2a9734955407eb8bcf2325b479a5f39144')[-1]
            rel = rel.lstrip('/\\')
            arc = f'healthy/{rel.replace(chr(92), "/")}'
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

        # 2) BraTS — preserve patient-dir layout under brats/
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

    final_size = OUT_PATH.stat().st_size
    print(f'\n[done] {written}/{total} files written  '
          f'in {(time.perf_counter()-t0)/60:.1f} min')
    print(f'[done] zip = {final_size/1e9:.2f} GB at {OUT_PATH}')


if __name__ == '__main__':
    main()
