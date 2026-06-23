"""Public I/O API for quantem.live.

Microscopy workflows import these helpers directly while exploring data at the
instrument, so keep this package initializer as a small public surface and put
implementation details in named modules.

The heavy HDF5 loader imports CuPy. Keep these exports lazy so control-plane
CLI commands can import pure helpers like ``quantem.widget.io.schema`` on a
laptop without a CUDA Python environment.
"""
from __future__ import annotations

from importlib import import_module

__version__ = "0.0.3"

_HDF5_EXPORTS = {
    "H5Writer",
    "LoadResult",
    "bin",
    "discover_masters",
    "find_emd_sibling",
    "get_metadata",
    "is_master_ready",
    "load",
    "load_parallel",
    "disk_of",
    "group_by_disk",
    "read_emd_metadata",
    "read_pixel_mask",
    "save",
    "wait_for_saves",
}

# Hugging Face dataset sharing. Kept lazy so importing quantem.widget.io stays
# cheap and huggingface_hub is only imported when a share/fetch is actually run.
_HUB_EXPORTS = {
    "upload",
    "download",
    "list_datasets",
    "delete",
    "status",
}

# Look-before-you-load folder survey (header-only memory/completeness report).
# Import-light: reads HDF5 headers only, never pixel data.
_SURVEY_EXPORTS = {"survey"}

# 2D image reader (Velox EMD HAADF / .npy) - separate from the 4D-STEM loader.
_IMAGE_EXPORTS = {"read_image", "read_image_stack", "read_images"}

# Memory profiler (disk staging + RAM + per-GPU VRAM).
_MEMORY_EXPORTS = {"memory"}

# Decompress-backend selection. Import-light (no cupy/Metal), so it stays cheap
# to ask "which backend would load() pick here?" without loading any kernel.
_BACKEND_EXPORTS = {"detect_backend", "resolve_backend"}

# Apple-Silicon MPS path. ``load(path)`` auto-detects mps on Mac and returns a
# LoadResult wrapping MPSChunked4DSTEM — no need to call load_mps_4dstem directly.
# load_mps_4dstem stays accessible for power users bypassing the unified loader
# but is intentionally NOT in __all__ (removed from public surface).
_MPS_EXPORTS = {"MPSChunked4DSTEM", "load_mps_4dstem", "clear_mps_cache"}

__all__ = [
    "H5Writer",
    "LoadResult",
    "MPSChunked4DSTEM",
    "bin",
    "clear_mps_cache",
    "delete",
    "detect_backend",
    "discover_masters",
    "download",
    "find_emd_sibling",
    "get_metadata",
    "is_master_ready",
    "list_datasets",
    "load",
    "load_parallel",
    "disk_of",
    "group_by_disk",
    "memory",
    "resolve_backend",
    "read_emd_metadata",
    "read_image",
    "read_image_stack",
    "read_images",
    "read_pixel_mask",
    "save",
    "status",
    "survey",
    "upload",
    "wait_for_saves",
    "__version__",
]


def __getattr__(name: str):
    if name in _HDF5_EXPORTS:
        module = import_module("quantem.widget.io.hdf5")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _HUB_EXPORTS:
        module = import_module("quantem.widget.io.hub")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _SURVEY_EXPORTS:
        module = import_module("quantem.widget.io.survey")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _IMAGE_EXPORTS:
        module = import_module("quantem.widget.io.image")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _MEMORY_EXPORTS:
        module = import_module("quantem.widget.io.memory")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _BACKEND_EXPORTS:
        module = import_module("quantem.widget.io.backends")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _MPS_EXPORTS:
        module = import_module("quantem.widget.io.backends.mps")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
