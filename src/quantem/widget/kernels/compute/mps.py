"""Compatibility wrapper for :mod:`quantem.gpu.compute.mps`."""
from __future__ import annotations

from quantem.gpu.compute.mps import *  # noqa: F401,F403
from quantem.gpu.compute.mps import (  # noqa: F401
    _bin2_mask,
    _bin_mask,
    _chunk_groups,
    _chunk_nbytes,
    _row_prefix_sum_chunk_numba,
)
