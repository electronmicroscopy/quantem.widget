"""Compatibility wrapper for :mod:`quantem.gpu.compute.backends`."""
from __future__ import annotations

from quantem.gpu.compute.backends import *  # noqa: F401,F403
from quantem.gpu.compute.backends import (  # noqa: F401
    _CHUNK_BYTE_BUDGET,
    _SPARSE_MASK_CHUNK_BYTE_BUDGET,
)
