"""Compatibility shim for the migrated HDF5 GPU IO path.

The accelerated HDF5 load/decompress implementation now lives in
``quantem.gpu.io.hdf5``. This module preserves the historical
``quantem.widget.io.hdf5`` import path for one release while widget callers
move to the new package.
"""
from __future__ import annotations

from quantem.gpu.io.hdf5 import *  # noqa: F401,F403
from quantem.gpu.io.hdf5 import __all__, __version__  # noqa: F401
