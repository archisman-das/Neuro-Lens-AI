"""Crash-safe checkpoint helpers.

Pulled out of src/train_segmentation_v7.py 2026-06-02 so that v9b training
scripts (and any other consumer) can use `atomic_save` without dragging in
the full v5+v7 trainer chain. This module has NO local imports — only
stdlib + torch — and is safe to ship in a minimal Colab bundle on its own.

Public API: `atomic_save(payload, path)`. The legacy underscored name
`_atomic_save` is kept as an alias so callers that imported the old name
keep working.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def atomic_save(payload: Any, path: Path) -> None:
    """Write `payload` to `path` atomically via tmp+rename.

    Critical for crash-safe checkpointing: a hard kill (or kernel-power 41,
    or a Colab disconnect) in the middle of torch.save can corrupt the
    file. Writing to a tmp file and renaming makes the operation atomic on
    Windows and POSIX — the target either has the old contents or the
    complete new contents, never a half-written file.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, tmp)
    if path.exists():
        path.unlink()
    tmp.rename(path)


# Legacy alias: the original definition in train_segmentation_v7 used a
# leading underscore (module-private style). Existing callers that did
# `from .train_segmentation_v7 import _atomic_save` should switch to
# `from .checkpoint_utils import atomic_save`, but this alias keeps the
# old import working for a transition period.
_atomic_save = atomic_save


__all__ = ['atomic_save', '_atomic_save']
