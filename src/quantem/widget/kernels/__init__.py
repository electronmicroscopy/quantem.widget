"""Compatibility wrappers for backend detection and kernel dispatch.

Kernel implementations now live in :mod:`quantem.gpu`. The old
``quantem.widget.kernels`` namespace remains as a migration shim for existing
imports.
"""
from __future__ import annotations

import importlib

from quantem.gpu.io.backends import _has_cuda, _has_mps


def detect() -> str:
    """Return the available accelerated backend, preserving the old GPU-only API."""
    if _has_cuda():
        return "cuda"
    if _has_mps():
        return "mps"
    raise RuntimeError(
        "quantem.widget.kernels needs an NVIDIA GPU (CUDA) or Apple Silicon "
        "(MPS); no supported accelerated backend found."
    )


def io_backend(name: str | None = None):
    """Return the backend IO module from :mod:`quantem.gpu`."""
    backend = name or detect()
    if backend == "cuda":
        return importlib.import_module("quantem.gpu.io.hdf5")
    return importlib.import_module(f"quantem.gpu.io.backends.{backend}")


def compute_backend(name: str | None = None):
    """Return the backend compute module from :mod:`quantem.gpu`."""
    return importlib.import_module(f"quantem.gpu.compute.{name or detect()}")


__all__ = [
    "compute_backend",
    "detect",
    "io_backend",
]
