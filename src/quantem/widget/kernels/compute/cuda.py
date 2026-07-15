"""Compatibility wrapper for CUDA-capable product compute in :mod:`quantem.gpu`."""
from __future__ import annotations

from quantem.gpu.compute.backends import CudaKernelCompute  # noqa: F401

__all__ = ["CudaKernelCompute"]
