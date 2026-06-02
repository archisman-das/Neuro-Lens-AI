"""Fetch the Navoneel 'no' (healthy) subset alongside the existing
Navoneel 'yes' (tumor) samples — same source, same scanner, same
preprocessing.

Critical to remove the source-confound in linear-probe evaluations
(scripts/eval_foundation_models.py). After this runs, samples/ood/
contains the matched healthy subset; the LOSO probe can then compute
a valid AUC because Navoneel has BOTH classes available within the
same source.

Source: miladfa7/Brain-MRI-Images-for-Brain-Tumor-Detection on HF (mirror
of Navoneel Chakrabarty's Kaggle binary brain-tumor-detection set).
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'samples' / 'ood' / 'healthy_navoneel'
OUT.mkdir(parents=True, exist_ok=True)


def main():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit('pip install huggingface_hub')

    print('[1/3] downloading miladfa7 navoneel mirror ...')
    zpath = hf_hub_download(
        repo_id='miladfa7/Brain-MRI-Images-for-Brain-Tumor-Detection',
        filename='Brain MRI Images for Brain Tumor Detection.zip',
        repo_type='dataset',
        local_dir=str(OUT / '_zip_tmp'),
    )
    print(f'   zip ready: {zpath}')

    print('[2/3] extracting "no" (healthy) subset ...')
    with zipfile.ZipFile(zpath) as zf:
        # Navoneel layout: "no/N1.jpg .. N98.jpg" (sometimes "no/N1.JPG")
        no_files = sorted(n for n in zf.namelist()
                            if '/no/' in n.lower()
                            and n.lower().endswith(('.jpg', '.jpeg', '.png')))
        if not no_files:
            # Some mirrors use a different folder name; try a flat fallback
            no_files = sorted(n for n in zf.namelist()
                                if n.rsplit('/', 1)[-1].lower().startswith('n')
                                and n.lower().endswith(('.jpg', '.jpeg', '.png')))
        print(f'   {len(no_files)} healthy "no" files found in zip')
        for nm in no_files:
            base = nm.rsplit('/', 1)[-1]
            with zf.open(nm) as src:
                (OUT / base).write_bytes(src.read())

    # Clean up zip dir
    try:
        Path(zpath).unlink()
        (OUT / '_zip_tmp').rmdir()
    except Exception:
        pass

    n_saved = sum(1 for _ in OUT.iterdir() if _.is_file())
    print(f'[3/3] saved {n_saved} healthy Navoneel samples to {OUT}')
    print('Now you can re-run scripts/eval_foundation_models.py — Navoneel is no longer source-monolithic, so LOSO AUC will be defined for it.')


if __name__ == '__main__':
    main()
