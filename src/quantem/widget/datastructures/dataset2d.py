"""A 2D image plus its calibration + source metadata.

A HAADF survey image carries a real pixel size (and a pile of acquisition
metadata) that a bare numpy array throws away. ``Dataset2d`` keeps the array
together with its ``sampling`` (physical size per pixel) and the raw source
metadata, and is duck-typed so ``quantem.widget.Show2D`` auto-draws a real scale
bar: ``Show2D(io.read_image(path))`` just works, in nm, no extra arguments.

Deliberately minimal - it exposes exactly the attributes Show2D looks for
(``array``, ``name``, ``sampling``, ``units``) plus ``metadata`` for the raw dict.
"""
from __future__ import annotations

import numpy as np


class Dataset2d:
    """2D array + sampling/units + raw source metadata (Show2D-compatible)."""

    def __init__(self, array, *, sampling=None, units=None, name="", metadata=None):
        self.array = np.asarray(array)
        # Show2D reads sampling[-2:], so always keep a (row, col) tuple.
        self.sampling = tuple(float(s) for s in sampling) if sampling is not None else (1.0, 1.0)
        self.units = list(units) if units is not None else ["pixels", "pixels"]
        self.name = name
        self.metadata = metadata or {}

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def dtype(self):
        return self.array.dtype

    def numpy(self) -> np.ndarray:
        return self.array

    def __repr__(self) -> str:
        return f"Dataset2d(shape={self.shape}, sampling={self.sampling}, units={self.units})"
