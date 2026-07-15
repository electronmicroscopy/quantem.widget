"""Compatibility wrapper for accelerated IO backend selection."""
from __future__ import annotations

from quantem.gpu.io.backends import *  # noqa: F401,F403
from quantem.gpu.io.backends import (  # noqa: F401
    _has_cuda,
    _has_mps,
    _nvidia_gpu_present,
    detect_backend,
    resolve_backend,
)

__all__ = ["detect_backend", "resolve_backend"]
