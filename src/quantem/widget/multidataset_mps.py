"""Compatibility exports for the MPS multi-dataset loader in :mod:`quantem.gpu`."""
from __future__ import annotations

from quantem.gpu.io.mps_multi import (  # noqa: F401
    LazyMPSDatasets,
    LazyMacbookDatasets,
    load_4dstem_macbook,
    load_mps_datasets,
    load_macbook_datasets,
)

__all__ = [
    "LazyMPSDatasets",
    "LazyMacbookDatasets",
    "load_4dstem_macbook",
    "load_mps_datasets",
    "load_macbook_datasets",
]
