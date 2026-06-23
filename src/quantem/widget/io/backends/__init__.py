"""Decompress-backend selection for :func:`quantem.widget.io.load`.

``load()`` decompresses bitshuffle+LZ4 4D-STEM masters on whatever hardware is
present, so the caller never has to know whether the box has an NVIDIA GPU, an
Apple GPU, or neither. This module owns that choice.

Three backends:

- ``cuda`` — cupy RawKernels (the original, fastest path). Returns a cupy array
  on the GPU and feeds the full pipeline including reconstruction.
- ``mps`` — Apple Metal compute shaders on Apple Silicon. Returns a numpy array
  (unified memory). View / screen only.
- ``cpu`` — pure h5py + hdf5plugin transparent decompress. Works anywhere.
  Returns a numpy array. View / screen only.

Detection is cheap and import-light: probing a backend must never import the
heavy kernel module, only check whether its toolkit is importable. The actual
kernel modules (``backends.cuda`` / ``backends.mps`` / ``backends.cpu``) are
imported lazily by ``load()`` once a backend is resolved.
"""
from __future__ import annotations

import os
import sys


_VALID = ("cuda", "mps", "cpu")


def _nvidia_gpu_present() -> bool:
    """True on a Linux box with an NVIDIA GPU, regardless of whether cupy imports.

    Import-light (a device-node stat, no torch/cupy), so detection can tell a
    real CUDA box apart from a GPU-less one even when cupy is missing — that
    distinction is what lets us REFUSE a silent CPU fallback on a CUDA box.
    """
    return sys.platform.startswith("linux") and os.path.exists("/dev/nvidia0")


# Hardware detection lives in quantem.widget.kernels (single source of truth).
# This module keeps a cpu fallback that kernels.detect() deliberately omits
# (kernels = GPU-only by design); the cpu path here still backs backend="cpu"
# parity tests + the _load_view cpu branch.
from quantem.widget.kernels import _has_cuda, _has_mps  # noqa: E402


def detect_backend() -> str:
    """Pick the best available backend. Order: cuda > mps > cpu.

    A CUDA box NEVER silently decodes on CPU: if an NVIDIA GPU is present but
    cupy is missing/broken, raise with the install fix instead of falling back to
    the ~20x-slower CPU path (the user would never want CPU on a GPU box). cpu is
    only ever chosen on a genuinely GPU-less machine.
    """
    if _has_cuda():
        return "cuda"
    if _nvidia_gpu_present():
        raise RuntimeError(
            "NVIDIA GPU detected but the cupy CUDA backend is unavailable, so "
            "load() would fall back to slow CPU decode — refusing. Install cupy:\n"
            "  conda: mamba install -c conda-forge cupy\n"
            "  pip:   pip install cupy-cuda13x   (or cupy-cuda12x for CUDA 12)\n"
            "Then retry. To force CPU anyway, pass backend='cpu' explicitly."
        )
    if _has_mps():
        return "mps"
    return "cpu"


def resolve_backend(backend: str | None) -> str:
    """Normalize the user's ``backend=`` argument to a concrete backend.

    ``"auto"`` / ``None`` → :func:`detect_backend`. An explicit name is
    returned as-is (validated) so the caller can force a path for testing or
    to opt out of the GPU (e.g. ``backend="cpu"`` on a CUDA box for a parity
    check). Raises on an unknown name with the valid set in the message.
    """
    if backend in (None, "auto"):
        return detect_backend()
    if backend not in _VALID:
        raise ValueError(
            f"Unknown backend {backend!r}. Use 'auto', or one of {_VALID}."
        )
    return backend
