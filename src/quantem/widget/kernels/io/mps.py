"""Compatibility wrapper for :mod:`quantem.gpu.io.backends.mps`."""
from __future__ import annotations

from quantem.gpu.io.backends.mps import *  # noqa: F401,F403
from quantem.gpu.io.backends.mps import (  # noqa: F401
    _MtlArray,
    _metal_buffer_alloc,
    _numpy_view,
    _parse_headers,
    _read_pixel_mask,
)
