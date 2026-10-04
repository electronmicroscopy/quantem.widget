"""Validate complete native source admission through the public widget API."""

import numpy as np
import pytest

from quantem.widget import Show4DSTEM


SOURCE_ARGUMENTS = {
    "rans_url": "./source/",
    "rans_format": "source112-tans1024-pair-v1",
    "rans_count": 66,
    "rans_dtype": "uint16",
    "scan_shape": (512, 512),
    "detector_shape": (192, 192),
    "backend": "webgpu",
    "verbose": False,
}


def construct(**changes):
    """Construct metadata-only admission without allocating the native series."""
    return Show4DSTEM(
        np.zeros((1, 1, 1, 1), dtype=np.uint8),
        **{**SOURCE_ARGUMENTS, **changes},
    )


def test_all66_source112_preserves_browser_admission():
    widget = construct()
    try:
        assert widget._rans_format == "source112-tans1024-pair-v1"
        assert widget._rans_dtype == "uint16"
        assert widget._rans_url == "./source/"
        assert widget.gpu_memory_label == "Browser WebGPU lossless encoded source"
    finally:
        widget.close()


@pytest.mark.parametrize(
    "changes",
    [
        {"rans_count": 0},
        {"rans_count": 67},
        {"rans_count": 66.5},
        {"rans_dtype": "uint8"},
        {"scan_shape": (256, 512)},
        {"detector_shape": (96, 192)},
        {"rans_url": ""},
        {"h5_urls": ["other_master.h5"]},
        {"lazy_urls": ["other.bin"]},
    ],
)
def test_reject_incomplete_or_mixed_native_source(changes):
    with pytest.raises(ValueError):
        construct(**changes)


@pytest.mark.parametrize("count", [1, 3, 66])
def test_progressive_subset_keeps_complete_native_acquisitions(count):
    widget = construct(rans_count=count)
    try:
        assert widget.n_frames == count
        assert (widget.shape_rows, widget.shape_cols) == (512, 512)
        assert (widget.det_rows, widget.det_cols) == (192, 192)
        assert widget._rans_dtype == "uint16"
    finally:
        widget.close()
