"""GPU kernels for quantem.live — auto-detected CUDA or MPS backend.

quantem.live runs on exactly two hardware targets:
- Linux + NVIDIA GPU (CUDA) — cupy kernels, the compute/reconstruction box.
- Apple Silicon MacBook (MPS) — raw Metal kernels, the viewing/screening laptop.

There is no CPU runtime backend (numpy/h5py stays a test oracle only). If neither
is present, ``detect()`` raises a clear error.

Two concerns, split by change-rate (see
``docs/dev-notes/2026-06-01-kernels-backend-architecture.md``):
- ``kernels.io`` — decode + bin + mask + scan-shape (frozen-ish).
- ``kernels.compute`` — masked-sum / prefix / bin / reduce (churns: BF/DF/DPC).

Flexible module dispatch, NO Protocol yet: ``detect()`` / ``io_backend()`` /
``compute_backend()`` return the matching submodule and the caller calls
``backend.decode(...)`` / ``backend.virtual_image(...)`` directly. Same function
names across ``cuda.py`` and ``mps.py`` by convention; a missing function raises
``AttributeError`` at the call site. A typing.Protocol contract is the eventual
target once the MPS op surface stops churning — not now.
"""
from __future__ import annotations

import importlib
import importlib.util
from functools import lru_cache


@lru_cache(maxsize=1)
def detect() -> str:
    """Return the active backend name: ``"cuda"`` or ``"mps"``.

    Order: cupy importable AND a CUDA device present -> ``cuda``; else Metal
    importable (Apple Silicon) -> ``mps``; else raise. Cached — the hardware does
    not change within a process.
    """
    if _has_cuda():
        return "cuda"
    if _has_mps():
        return "mps"
    raise RuntimeError(
        "quantem.live needs an NVIDIA GPU (CUDA) or Apple Silicon (MPS); "
        "no supported backend found."
    )


def io_backend(name: str | None = None):
    """The io submodule for the active (or named) backend — has ``decode(...)``."""
    return importlib.import_module(f"{__name__}.io.{name or detect()}")


def compute_backend(name: str | None = None):
    """The compute submodule for the active (or named) backend — has the
    virtual-image / reduction ops."""
    return importlib.import_module(f"{__name__}.compute.{name or detect()}")


def _has_cuda() -> bool:
    """True when cupy imports AND a CUDA device is actually present (the cupy
    wheel can be installed on a box with no GPU)."""
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _has_mps() -> bool:
    """True on Apple Silicon where the Metal toolkit is importable."""
    return importlib.util.find_spec("Metal") is not None
