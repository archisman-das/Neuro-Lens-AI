"""Fetch out-of-distribution brain MRI samples for testing the deployed system.

What "OOD" means here: NOT from BraTS 2020, kaggle_3m LGG, Figshare-Cheng-2017,
or Kaggle 4-class — i.e. nothing the v8 / v5 / v3 cascade has ever seen.

Sources chosen:
  1. g4m3r/T1w_MRI_Brain_Slices (HF)  - OpenNeuro ds003592 / Spreng et al.
     neurocognitive aging study. 301 healthy adults, 30 coronal slices each.
     MIT license. Public, ungated. PNG. -> samples/ood/healthy_coronal_T1
  2. FOMO25/FOMO-MRI (HF)             - OASIS-1/OASIS-2 + others. T1, T2,
     FLAIR, T1c, PD, etc. CC BY-NC-SA 4.0. Gated (auto-approve). NIfTI.
     We extract middle slices per modality -> samples/ood/multimodal_oasis
  3. UniDataPro/brain-cancer-dataset (HF) - Proprietary DICOM tumor study.
     CC BY-NC-ND-4.0. Public. -> samples/ood/tumor_proprietary_dicom

Why coronal slices from #1: most of our training data is axial. Coronal is
a real distribution shift on top of the source-OOD shift, which is exactly
the kind of stress test the user asked for.

If a download fails (e.g. FOMO is still gated or the network is slow), the
script keeps going and reports what it managed to grab.
"""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'samples' / 'ood'
OUT.mkdir(parents=True, exist_ok=True)


def _hf_api(token: Optional[str] = None):
    from huggingface_hub import HfApi
    tok = token or os.environ.get('HF_TOKEN')
    return HfApi(token=tok)


def fetch_openneuro_t1(n: int = 15) -> List[Path]:
    """Grab N healthy T1w coronal PNGs from g4m3r/T1w_MRI_Brain_Slices.

    The dataset ships as a single images.zip (~352 MB). Download once,
    extract only the slices we want, then delete the zip to keep the
    samples folder small. 30 slices per subject -> stride 30 = one per
    subject so we get anatomical diversity, not 12 slices of one person.
    """
    from huggingface_hub import hf_hub_download
    import zipfile
    target = OUT / 'healthy_coronal_T1_openneuro'
    target.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []
    # Reuse the zip if we already downloaded it in a previous (failed) run.
    cached_zip = ROOT / 'samples' / 'ood' / '_zip_tmp' / 'images.zip'
    if cached_zip.exists() and cached_zip.stat().st_size > 100_000_000:
        zpath = str(cached_zip)
        print(f'  reusing cached zip: {cached_zip}')
    else:
        print('  downloading images.zip (one-time, ~352 MB)...')
        try:
            zpath = hf_hub_download(
                repo_id='g4m3r/T1w_MRI_Brain_Slices',
                filename='images.zip',
                repo_type='dataset',
                local_dir=str(cached_zip.parent),
            )
        except Exception as exc:
            print(f'  [fail-zip] {type(exc).__name__}: {exc}')
            return []
    # Actual zip layout: images/sub-XX_slice_YYY.png  (X 01..99, Y mid ~170)
    import re
    pat = re.compile(r'sub-(\d+)_slice_(\d+)\.png$')
    target_slice = 169  # middle-ish of the [141..199] range present
    chosen_subjects: set[str] = set()
    with zipfile.ZipFile(zpath) as zf:
        # Sort so we pick subjects in a deterministic spread (every 7th sub).
        candidates = sorted(zf.namelist())
        stride = max(1, 99 // n)
        wanted_subs = {f'{(1 + stride*k):02d}' for k in range(n)}
        for nm in candidates:
            m = pat.search(nm.rsplit('/', 1)[-1])
            if not m:
                continue
            sub, sl = m.group(1), int(m.group(2))
            if sub not in wanted_subs:
                continue
            if sub in chosen_subjects:
                continue
            if sl != target_slice:
                continue
            base = nm.rsplit('/', 1)[-1]
            with zf.open(nm) as src:
                (target / base).write_bytes(src.read())
            out.append(target / base)
            chosen_subjects.add(sub)
            print(f'  [extract] {base}')
            if len(out) >= n:
                break
    # Clean up the big zip so the samples folder stays slim.
    try:
        Path(zpath).unlink()
        # Remove the empty .cache subfolder hf_hub created if present.
        for c in target.glob('.cache'):
            import shutil; shutil.rmtree(c, ignore_errors=True)
    except Exception:
        pass
    return out


def fetch_fomo_multimodal(n_subjects: int = 2,
                           modalities: List[str] = ('t1', 't2', 'flair')) -> List[Path]:
    """Try to grab 1-2 subjects' multi-modal NIfTIs from FOMO50K and
    extract middle slices per modality.

    Gated dataset; if denied, returns []."""
    try:
        from huggingface_hub import hf_hub_download, HfApi
        import nibabel as nib
        import numpy as np
        from PIL import Image
    except Exception as exc:
        print(f'  [skip-fomo] dep missing: {exc}')
        return []

    target = OUT / 'multimodal_oasis_fomo'
    target.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []
    api = _hf_api()

    # Probe a couple of plausible paths -- the README sample shows
    # PT001_OASIS1/sub_52/ses_1/t1.nii.gz layout.
    probes = []
    for ptn in ('PT001_OASIS1', 'PT002_OASIS2'):
        for sub_n in (1, 2, 5, 10, 27, 52):
            for mod in modalities:
                probes.append(f'{ptn}/sub_{sub_n}/ses_1/{mod}.nii.gz')
    # Build sets per subject so we only keep ones where ALL modalities resolved.
    by_subject: dict[str, dict[str, Path]] = {}
    for path in probes:
        subj = '/'.join(path.split('/')[:2])
        mod = path.split('/')[-1].replace('.nii.gz', '')
        try:
            p = hf_hub_download(
                repo_id='FOMO25/FOMO-MRI',
                filename=path,
                repo_type='dataset',
                local_dir=str(target),
                local_dir_use_symlinks=False,
            )
            by_subject.setdefault(subj, {})[mod] = Path(p)
            print(f'  [ok-fomo] {path}')
            if len([k for k, v in by_subject.items() if len(v) >= len(modalities)]) >= n_subjects:
                break
        except Exception as exc:
            print(f'  [miss-fomo] {path}: {type(exc).__name__}')

    # For subjects where we got every modality, slice mid-axial + save PNG
    for subj, mods in by_subject.items():
        if len(mods) < len(modalities):
            continue
        subj_tag = subj.replace('/', '_')
        for mod_name, nii_path in mods.items():
            try:
                arr = nib.load(str(nii_path)).get_fdata()
                # take mid-axial slice (assume axes: x, y, z)
                z = arr.shape[2] // 2
                sl = arr[:, :, z]
                # 99th-pct normalize -> uint8
                p99 = np.percentile(sl, 99) or 1.0
                sl = np.clip(sl / p99, 0, 1) * 255
                img = Image.fromarray(sl.astype('uint8'))
                out_png = target / f'{subj_tag}__{mod_name}.png'
                img.save(out_png)
                out.append(out_png)
                print(f'  [save] {out_png.name}')
            except Exception as exc:
                print(f'  [slice-fail] {nii_path.name}: {exc}')
    return out


def fetch_unidata_dicom(n_per_series: int = 2) -> List[Path]:
    """Pull DICOMs from EACH of the UniDataPro Series folders (SE000001 ..
    SE000009+). Each Series is typically a different MRI sequence (T1, T2,
    FLAIR, DWI etc.) — which gives us the multi-CHANNEL/multi-MODALITY
    diversity the user asked for, on a proprietary OOD source.

    Reads SeriesDescription / Modality from the DICOM header and prefixes
    the saved PNG so the eval can see which sequence it was.
    """
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        import pydicom
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        if 'pydicom' in str(exc):
            print('  [skip-unidata] pydicom not installed; pip install pydicom')
        else:
            print(f'  [skip-unidata] dep missing: {exc}')
        return []

    target = OUT / 'tumor_proprietary_multimodal_unidata'
    target.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []
    try:
        files = list_repo_files('UniDataPro/brain-cancer-dataset', repo_type='dataset')
        # .dcm only; group by series (SE000001 .. SE000009)
        dcm = sorted(f for f in files if f.lower().endswith('.dcm'))
        by_series: dict[str, List[str]] = {}
        for f in dcm:
            parts = f.split('/')
            if len(parts) >= 3 and parts[1].startswith('SE'):
                by_series.setdefault(parts[1], []).append(f)
        print(f'  found {len(by_series)} series in repo: {sorted(by_series.keys())}')
        for series, members in sorted(by_series.items()):
            # Pick mid-series + a second offset for slice variety
            mid = len(members) // 2
            picks = [members[mid]]
            if len(members) > 4 and n_per_series > 1:
                picks.append(members[mid // 2])
            for f in picks[:n_per_series]:
                try:
                    p = hf_hub_download(
                        repo_id='UniDataPro/brain-cancer-dataset',
                        filename=f,
                        repo_type='dataset',
                        local_dir=str(target),
                    )
                except Exception as exc:
                    print(f'  [dl-fail] {f}: {type(exc).__name__}')
                    continue
                try:
                    d = pydicom.dcmread(p, force=True)
                    # Pull the actual sequence tag for downstream labelling.
                    sd = str(getattr(d, 'SeriesDescription', '') or
                              getattr(d, 'ProtocolName', '') or
                              getattr(d, 'Modality', '') or 'unk').strip()
                    sd_clean = ''.join(c for c in sd if c.isalnum() or c in '._-')[:40] or 'unk'
                    arr = d.pixel_array.astype('float32')
                    p99 = float(np.percentile(arr, 99) or 1.0)
                    arr = np.clip(arr / p99, 0, 1) * 255
                    out_name = f'{series}__{sd_clean}__{Path(p).stem}.png'
                    out_png = target / out_name
                    Image.fromarray(arr.astype('uint8')).save(out_png)
                    out.append(out_png)
                    print(f'  [ok-unidata] {series}/{Path(p).name} '
                          f'(SeriesDescription={sd!r}) -> {out_name}')
                except Exception as exc:
                    print(f'  [dicom-fail] {f}: {type(exc).__name__}: {exc}')
    except Exception as exc:
        print(f'  [skip-unidata] list_repo_files: {exc}')
    return out


def fetch_ultralytics_tumor_patients(n_patients: int = 10) -> List[Path]:
    """Pull one image per distinct patient prefix from Ultralytics/Brain-tumor.

    Filenames are <patient_id>_<frame>.jpg (e.g. 00054_145.jpg). One image
    per patient -> n_patients distinct OOD tumor patients. AGPL-3.0.
    """
    from huggingface_hub import hf_hub_download, list_repo_files
    target = OUT / 'tumor_multi_patient_ultralytics'
    target.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []
    try:
        files = list_repo_files('Ultralytics/Brain-tumor', repo_type='dataset')
    except Exception as exc:
        print(f'  [skip-ult] {exc}')
        return []
    imgs = sorted(f for f in files if f.startswith('train/images/') and f.endswith('.jpg'))
    # Group by patient prefix
    seen: dict[str, str] = {}
    for f in imgs:
        base = f.rsplit('/', 1)[-1]
        pid = base.split('_', 1)[0]
        if pid not in seen:
            seen[pid] = f
        if len(seen) >= n_patients:
            break
    for pid, f in seen.items():
        try:
            p = hf_hub_download(
                repo_id='Ultralytics/Brain-tumor',
                filename=f,
                repo_type='dataset',
                local_dir=str(target),
            )
            # Flatten the train/images/ prefix so the eval picks them up.
            flat = target / f'pt{pid}__{Path(p).name}'
            Path(p).rename(flat)
            out.append(flat)
            print(f'  [ok-ult] patient={pid} -> {flat.name}')
        except Exception as exc:
            print(f'  [fail-ult] {f}: {type(exc).__name__}: {exc}')
    return out


def fetch_navoneel_binary(n: int = 6) -> List[Path]:
    """Pull N Y*.jpg (tumor-positive) images from miladfa7's mirror of
    Navoneel Chakrabarty's binary brain-tumor-detection set."""
    from huggingface_hub import hf_hub_download
    import zipfile
    target = OUT / 'tumor_binary_navoneel_via_miladfa7'
    target.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []
    try:
        zpath = hf_hub_download(
            repo_id='miladfa7/Brain-MRI-Images-for-Brain-Tumor-Detection',
            filename='Brain MRI Images for Brain Tumor Detection.zip',
            repo_type='dataset',
            local_dir=str(target),
        )
    except Exception as exc:
        print(f'  [skip-nav] {exc}')
        return []
    with zipfile.ZipFile(zpath) as zf:
        # Navoneel layout: "yes/Y1.jpg .. Y155.jpg" + "no/N1.jpg .."
        yes = sorted(n for n in zf.namelist()
                      if '/yes/' in n.lower() and n.lower().endswith(('.jpg', '.png')))
        if not yes:
            yes = sorted(n for n in zf.namelist()
                          if 'y' in n.rsplit('/', 1)[-1].lower()[:2]
                          and n.lower().endswith(('.jpg', '.png')))
        step = max(1, len(yes) // n)
        picks = yes[::step][:n]
        for nm in picks:
            base = nm.rsplit('/', 1)[-1]
            with zf.open(nm) as src:
                (target / base).write_bytes(src.read())
            out.append(target / base)
            print(f'  [ok-nav] {base}')
    try:
        Path(zpath).unlink()
    except Exception:
        pass
    return out


def main():
    t0 = time.perf_counter()
    print('=== Fetching OOD brain MRI samples ===')
    print()
    print('-> 1/5 OpenNeuro healthy T1 coronal (g4m3r/T1w_MRI_Brain_Slices)')
    a = fetch_openneuro_t1(n=12)
    print()
    print('-> 2/5 OASIS multi-modal (FOMO25/FOMO-MRI, gated)')
    b = fetch_fomo_multimodal(n_subjects=2)
    print()
    print('-> 3/5 Proprietary tumor DICOM (UniDataPro/brain-cancer-dataset)')
    c = fetch_unidata_dicom(n_per_series=2)
    print()
    print('-> 4/5 Multi-patient tumor (Ultralytics/Brain-tumor)')
    d = fetch_ultralytics_tumor_patients(n_patients=10)
    print()
    print('-> 5/5 Binary tumor (miladfa7 mirror of Navoneel Chakrabarty)')
    e = fetch_navoneel_binary(n=6)
    print()
    print('=== Summary ===')
    print(f'  OpenNeuro healthy:        {len(a)}')
    print(f'  FOMO multi-modal:         {len(b)}')
    print(f'  UniData tumor (1 pt):     {len(c)}')
    print(f'  Ultralytics ({len(d)} pts):       {len(d)}')
    print(f'  Navoneel binary:          {len(e)}')
    print(f'  elapsed: {time.perf_counter() - t0:.1f}s')


if __name__ == '__main__':
    main()
