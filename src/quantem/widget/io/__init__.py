"""Widget-owned image, sharing, memory, and transfer I/O helpers.

Microscopy workflows import these helpers directly while exploring data at the
instrument, so keep this package initializer as a small public surface and put
implementation details in named modules.

Scientific 4D-STEM loading, saving, discovery, and inspection belong to
``quantem.gpu.io``. This namespace intentionally does not forward those APIs.
"""
from __future__ import annotations

from importlib import import_module

__version__ = "0.0.3"

# Hugging Face dataset sharing. Kept lazy so importing quantem.widget.io stays
# cheap and huggingface_hub is only imported when a share/fetch is actually run.
_HUB_EXPORTS = {
    "upload",
    "download",
    "list_datasets",
    "read_meta",
    "delete",
    "status",
}

# 2D image reader (Velox EMD HAADF / .npy) - separate from the 4D-STEM loader.
_IMAGE_EXPORTS = {"read_gif", "read_image", "read_image_stack", "read_images", "RgbImage"}

# Memory profiler (disk staging + RAM + per-GPU VRAM).
_MEMORY_EXPORTS = {"memory"}

# Data-transfer planner for large microscopy/HPC workflows. Kept import-light so
# future CLI commands can dry-run transfer plans without importing viewer code.
_DATA_TRANSFER_EXPORTS = {
    "DataTransferEntry",
    "DataTransferFile",
    "DataTransferGroup",
    "DataTransferPlan",
    "DataTransferResult",
    "DataTransferState",
    "DataTransferSummary",
    "collect_data_transfer_groups",
    "copy_data_transfer",
    "data_transfer_plan_from_dict",
    "filter_data_transfer_plan",
    "inspect_data_transfer",
    "plan_data_transfer",
    "read_data_transfer_manifest",
    "summarize_data_transfer",
    "target_masters",
    "data_transfer_load_warnings",
    "update_data_transfer_plan",
    "write_data_transfer_manifest",
}

__all__ = [
    "delete",
    "download",
    "list_datasets",
    "memory",
    "DataTransferEntry",
    "DataTransferFile",
    "DataTransferGroup",
    "DataTransferPlan",
    "DataTransferResult",
    "DataTransferState",
    "DataTransferSummary",
    "collect_data_transfer_groups",
    "copy_data_transfer",
    "data_transfer_plan_from_dict",
    "filter_data_transfer_plan",
    "inspect_data_transfer",
    "plan_data_transfer",
    "read_data_transfer_manifest",
    "summarize_data_transfer",
    "target_masters",
    "data_transfer_load_warnings",
    "update_data_transfer_plan",
    "read_meta",
    "read_gif",
    "read_image",
    "read_image_stack",
    "read_images",
    "RgbImage",
    "status",
    "upload",
    "write_data_transfer_manifest",
    "__version__",
]


def __getattr__(name: str):
    if name in _HUB_EXPORTS:
        module = import_module("quantem.widget.io.hub")
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
    if name in _DATA_TRANSFER_EXPORTS:
        module = import_module("quantem.widget.io.data_transfer")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
