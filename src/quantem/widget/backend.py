"""Compatibility wrapper for accelerated backend selection.

The backend policy now lives in :mod:`quantem.gpu.io.backends`. This module
keeps the historical ``quantem.widget.backend`` import path working during the
migration.
"""
from __future__ import annotations

from quantem.gpu.io.backends import detect_backend, resolve_backend

__all__ = ["detect_backend", "resolve_backend"]
