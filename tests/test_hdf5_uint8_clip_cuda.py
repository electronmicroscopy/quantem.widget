"""CUDA regression tests for the direct uint8 HDF5 browse path."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

try:
    from quantem.widget.io.hdf5 import _clip_to_uint8, _clip_to_uint8_count  # noqa: E402
except ImportError as exc:
    if "partially initialized module 'quantem.core.datastructures'" not in str(exc):
        raise
    # MJGOAT has a neighboring editable quantem checkout that can collide during
    # collection. Keep this CUDA micro-test focused on quantem.widget's HDF5
    # module without importing the package root.
    root = Path(__file__).resolve().parents[1] / "src"
    for name in [
        "quantem.widget.io.hdf5",
        "quantem.widget.io",
        "quantem.widget",
        "quantem",
    ]:
        sys.modules.pop(name, None)
    q = types.ModuleType("quantem")
    q.__path__ = [str(root / "quantem")]
    sys.modules["quantem"] = q
    w = types.ModuleType("quantem.widget")
    w.__path__ = [str(root / "quantem" / "widget")]
    sys.modules["quantem.widget"] = w
    io = types.ModuleType("quantem.widget.io")
    io.__path__ = [str(root / "quantem" / "widget" / "io")]
    sys.modules["quantem.widget.io"] = io
    spec = importlib.util.spec_from_file_location(
        "quantem.widget.io.hdf5",
        root / "quantem" / "widget" / "io" / "hdf5.py",
    )
    h5 = importlib.util.module_from_spec(spec)
    sys.modules["quantem.widget.io.hdf5"] = h5
    assert spec.loader is not None
    spec.loader.exec_module(h5)
    _clip_to_uint8 = h5._clip_to_uint8
    _clip_to_uint8_count = h5._clip_to_uint8_count


@pytest.mark.parametrize("dtype", [np.uint16, np.uint32])
def test_clip_to_uint8_count_saturates_and_counts(dtype):
    """The direct dtype='u8' path clips counts above 255 instead of wrapping."""
    src = cp.asarray([0, 1, 254, 255, 256, 1000, 65535], dtype=dtype)
    dst = cp.empty(src.shape, dtype=cp.uint8)

    clipped = _clip_to_uint8_count(src, dst)
    cp.cuda.Device().synchronize()

    np.testing.assert_array_equal(
        cp.asnumpy(dst),
        np.asarray([0, 1, 254, 255, 255, 255, 255], dtype=np.uint8),
    )
    assert int(clipped) == 3


@pytest.mark.parametrize("dtype", [np.uint16, np.uint32])
def test_clip_to_uint8_saturates_without_counting(dtype):
    """The no-bin browse hot path clips to uint8 without the slow count pass."""
    src = cp.asarray([0, 255, 256, 4096], dtype=dtype)
    dst = cp.empty(src.shape, dtype=cp.uint8)

    assert _clip_to_uint8(src, dst) is True
    cp.cuda.Device().synchronize()

    np.testing.assert_array_equal(
        cp.asnumpy(dst),
        np.asarray([0, 255, 255, 255], dtype=np.uint8),
    )
