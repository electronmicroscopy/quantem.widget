"""Compatibility wrapper for virtual detector compute in :mod:`quantem.gpu`."""
from __future__ import annotations

from quantem.gpu.detector import *  # noqa: F401,F403
from quantem.gpu.detector import (  # noqa: F401
    _detector_mask,
    _resolve_backend,
)
