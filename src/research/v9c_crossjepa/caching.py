"""Frozen-teacher embedding cache for CrossJEPA training.

CrossJEPA reports a 16h/epoch -> 2min/epoch speedup from caching target
embeddings once (instead of recomputing the frozen teacher's forward
each step). This module provides:

  - TeacherEmbeddingCache  : append-only on-disk (.npz) cache keyed by
                              (scan_id, plane, slice_idx)  for Method 1, or
                              (scan_id, modality, patch_position) for Method 2
  - prepare_method1_cache  : one-time pass over all volumes to populate the
                              cache for Method 1 (frozen v8 ConvNeXt embeddings)
  - prepare_method2_cache  : one-time pass for Method 2 (4 separate frozen
                              per-modality I-JEPA teachers)

The cache is content-addressable by SHA-1 of the (scan_id, key_tuple)
to keep filesystem layout flat — important for slow filesystems like
the Colab Drive mount.

Atomic writes via tempfile + os.replace so a Colab disconnect mid-cache
leaves the existing cache valid (worst case: redoing one shard).
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import time
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np


def _key_to_path(root: Path, scan_id: str, key_tuple: tuple) -> Path:
    """Hash-based path for a single embedding entry. Subdivides into 256
    subfolders by the first byte of the hash so no directory ever holds
    more than ~1/256 of the entries (filesystem-friendly on Drive)."""
    raw = f'{scan_id}|{key_tuple}'.encode('utf-8')
    h = hashlib.sha1(raw).hexdigest()
    return root / h[:2] / f'{h}.npz'


def _atomic_save_npz(path: Path, **arrays) -> None:
    """Write .npz atomically: write to tempfile in the same dir, then
    os.replace (atomic on POSIX + NTFS)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.cache_', suffix='.npz', dir=str(path.parent))
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class TeacherEmbeddingCache:
    """Disk-backed cache of frozen-teacher embeddings.

    Usage:
        cache = TeacherEmbeddingCache(root='cache/method1')
        # Lookup (returns None on miss):
        emb = cache.get(scan_id='brats_001', plane='axial', slice_idx=87)
        # Insert:
        cache.put(scan_id='brats_001', plane='axial', slice_idx=87,
                   embedding=emb_array)
        # Size on disk:
        n_entries, n_bytes = cache.stats()
    """

    def __init__(self, root: os.PathLike):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, scan_id: str, **key) -> Path:
        # Stable serialization of the key tuple (sorted keys -> reproducible)
        ks = tuple(sorted(key.items()))
        return _key_to_path(self.root, scan_id, ks)

    def has(self, scan_id: str, **key) -> bool:
        return self._path(scan_id, **key).exists()

    def get(self, scan_id: str, **key) -> Optional[np.ndarray]:
        p = self._path(scan_id, **key)
        if not p.exists():
            return None
        try:
            with np.load(p, allow_pickle=False) as data:
                return data['embedding']
        except Exception:
            # Corrupted cache entry (e.g. from interrupted write) — drop it
            try:
                p.unlink()
            except OSError:
                pass
            return None

    def put(self, scan_id: str, embedding: np.ndarray, **key) -> None:
        p = self._path(scan_id, **key)
        _atomic_save_npz(p, embedding=embedding.astype(np.float32))

    def get_or_compute(self, scan_id: str, compute_fn, **key) -> np.ndarray:
        """If cache miss, call `compute_fn()` -> np.ndarray, store it,
        return it. compute_fn is a 0-arg callable."""
        hit = self.get(scan_id, **key)
        if hit is not None:
            return hit
        emb = compute_fn()
        self.put(scan_id, emb, **key)
        return emb

    def stats(self) -> tuple[int, int]:
        n = 0
        total = 0
        for p in self.root.rglob('*.npz'):
            n += 1
            try:
                total += p.stat().st_size
            except OSError:
                pass
        return n, total

    def __repr__(self) -> str:
        n, b = self.stats()
        return f'TeacherEmbeddingCache(root={self.root}, n={n}, {b/1e6:.1f}MB)'


def prepare_method1_cache(scan_iter: Iterator[dict],
                            v8_teacher,
                            cache: TeacherEmbeddingCache,
                            slices_per_volume: int = 8,
                            log_every: int = 25) -> dict:
    """One-time pass that fills the Method 1 cache.

    Args:
      scan_iter: yields {'scan_id': str, 'volume': (D, H, W, C) ndarray,
                          'voxel_spacing': (3,), 'planes': ['axial', ...]}
      v8_teacher: object with .embed_slice(slice_2d_rgb_uint8) -> (D,)
      cache: TeacherEmbeddingCache instance
      slices_per_volume: stratified per (volume, plane) — picks
                         `slices_per_volume` evenly-spaced indices per plane

    Returns: dict of summary counts.
    """
    from .conditioning import PLANES as ALL_PLANES
    counts = {'volumes': 0, 'hits': 0, 'misses': 0, 'errors': 0}
    t0 = time.perf_counter()
    for scan in scan_iter:
        counts['volumes'] += 1
        sid = scan['scan_id']
        vol = scan['volume']
        D = vol.shape[0]
        H = vol.shape[1]
        W = vol.shape[2]
        planes = scan.get('planes', list(ALL_PLANES))
        for plane in planes:
            n_slices = (D if plane == 'axial' else
                         W if plane == 'sagittal' else H)
            # Stratified sample: keep the central 60% of slices
            lo, hi = int(0.2 * n_slices), int(0.8 * n_slices)
            picks = np.linspace(lo, hi - 1, slices_per_volume, dtype=int)
            for idx in picks:
                if cache.has(sid, plane=plane, slice_idx=int(idx)):
                    counts['hits'] += 1
                    continue
                try:
                    sl = _extract_slice(vol, plane, int(idx))
                    emb = v8_teacher.embed_slice(sl)
                    cache.put(sid, embedding=emb, plane=plane, slice_idx=int(idx))
                    counts['misses'] += 1
                except Exception:
                    counts['errors'] += 1
        if counts['volumes'] % log_every == 0:
            elapsed = time.perf_counter() - t0
            n_e, n_b = cache.stats()
            print(f'  [cache] {counts["volumes"]} volumes processed in {elapsed:.0f}s  '
                  f'({n_e} entries, {n_b/1e6:.0f} MB)', flush=True)
    return counts


def _extract_slice(volume: np.ndarray, plane: str, idx: int) -> np.ndarray:
    """Pull a single 2D slice from a (D, H, W, C) volume. Returns
    (H', W', 3) uint8 ready for v8's image processor."""
    if plane == 'axial':
        sl = volume[idx]               # (H, W, C)
    elif plane == 'sagittal':
        sl = volume[:, :, idx]          # (D, H, C)
    elif plane == 'coronal':
        sl = volume[:, idx, :]          # (D, W, C)
    else:
        raise ValueError(f'unknown plane {plane!r}')
    # Normalize to 0-255 uint8 + replicate to 3 channels for v8
    sl = sl.astype(np.float32)
    if sl.ndim == 3 and sl.shape[-1] >= 1:
        # Use first channel if multi-channel (v8 was trained 3-ch)
        sl = sl[..., 0]
    lo, hi = float(sl.min()), float(sl.max())
    if hi - lo < 1e-9:
        sl = np.zeros_like(sl, dtype=np.uint8)
    else:
        sl = ((sl - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)
    return np.stack([sl, sl, sl], axis=-1)


__all__ = ['TeacherEmbeddingCache', 'prepare_method1_cache']
