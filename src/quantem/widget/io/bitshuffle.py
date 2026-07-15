"""Compatibility wrapper for CUDA bitshuffle kernels in :mod:`quantem.gpu.io`."""
from __future__ import annotations

from quantem.gpu.io.bitshuffle import *  # noqa: F401,F403
from quantem.gpu.io.bitshuffle import __getattr__  # noqa: F401
