"""One-shot helper: re-zip dataset_v8/ into dataset_v8.zip for upload to
Google Drive (the v9b Colab pipeline expects this exact filename).

Why this exists: shell-escaping a backslash inside `python -c` on Windows
is messy, so the workhorse goes in a file.
"""
from __future__ import annotations

import os
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'dataset_v8'
OUT = ROOT / 'dataset_v8.zip'


def main():
    if not SRC.exists():
        raise SystemExit(f'dataset_v8 not found at {SRC}')
    t0 = time.perf_counter()
    # ZIP_STORED — PNGs are already deflate-compressed, so DEFLATE burns
    # CPU for ~1% gain. STORED is ~10x faster.
    with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_STORED, allowZip64=True) as zf:
        nf = 0
        for r, _, fs in os.walk(SRC):
            for f in fs:
                p = Path(r) / f
                # arcname uses forward slashes for cross-OS portability
                rel = p.relative_to(ROOT).as_posix()
                zf.write(p, arcname=rel)
                nf += 1
                if nf % 10000 == 0:
                    print(f'  {nf} files, elapsed {time.perf_counter()-t0:.0f}s')
    size_gb = OUT.stat().st_size / 1e9
    print(f'\n[done] {nf} files -> {OUT.name} ({size_gb:.2f} GB) in '
          f'{time.perf_counter()-t0:.0f}s')


if __name__ == '__main__':
    main()
