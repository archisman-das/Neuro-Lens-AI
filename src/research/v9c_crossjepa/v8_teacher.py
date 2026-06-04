"""Backwards-compat shim — V8FrozenTeacher + V8_EMBED_DIM now live in
src/research/v9c_crossjepa/teachers.py alongside DINOv2 and any future
teachers, so the multi-teacher modes (Option A = dual prediction,
Option C = teacher-id conditioning) can iterate them via a common
BaseFrozenTeacher API.

Existing call sites that do `from src.research.v9c_crossjepa.v8_teacher
import V8FrozenTeacher, V8_EMBED_DIM` continue to work unchanged.
"""
from .teachers import V8FrozenTeacher, V8_EMBED_DIM   # noqa: F401

__all__ = ['V8FrozenTeacher', 'V8_EMBED_DIM']
