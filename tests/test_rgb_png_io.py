"""True-color PNG read + Show2D/Show3D display (not gray-only)."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from quantem.widget import Show2D, Show3D, io
from quantem.widget.io.image import RgbImage, _normalize_image_array


def _write_rgb_png(path: Path, h=24, w=32) -> np.ndarray:
    """Magenta-left / green-right RGB figure (easy to spot if grayed)."""
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, : w // 2, 0] = 220
    rgb[:, : w // 2, 2] = 220
    rgb[:, w // 2 :, 1] = 200
    Image.fromarray(rgb, mode="RGB").save(path)
    return rgb


def test_normalize_preserves_hw3_rgb():
    rgb = np.zeros((10, 12, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    out = _normalize_image_array(rgb)
    assert out.shape == (10, 12, 3)
    assert out[..., 0].max() == 255


def test_read_image_rgb_png_returns_color(tmp_path: Path):
    path = tmp_path / "color.png"
    expected = _write_rgb_png(path)
    loaded = io.read_image(path)
    assert isinstance(loaded, RgbImage)
    assert loaded.array.shape == expected.shape
    # Must not be the old (W, 3) first-row bug.
    assert loaded.array.shape[0] == expected.shape[0]
    np.testing.assert_array_equal(loaded.array, expected)


def test_show2d_reads_rgb_png_as_color(tmp_path: Path):
    path = tmp_path / "fig.png"
    _write_rgb_png(path)
    w = Show2D(io.read_image(path), verbose=False)
    assert list(w.is_rgb) == [True]
    frame = w._rgb_frames[0]
    assert frame is not None
    assert frame.shape == (24, 32, 3)
    # Magenta half has high R and B; green half has high G.
    left = frame[:, :16]
    right = frame[:, 16:]
    assert float(left[..., 0].mean()) > 0.5
    assert float(left[..., 2].mean()) > 0.5
    assert float(right[..., 1].mean()) > 0.5


def test_show2d_from_folder_rgb(tmp_path: Path):
    folder = tmp_path / "frames"
    folder.mkdir()
    _write_rgb_png(folder / "a.png")
    _write_rgb_png(folder / "b.png", h=24, w=32)
    w = Show2D.from_folder(folder, watch=False, verbose=False)
    assert w.n_images == 2
    assert list(w.is_rgb) == [True, True]


def test_show3d_rgb_stack_from_pngs(tmp_path: Path):
    folder = tmp_path / "stack"
    folder.mkdir()
    _write_rgb_png(folder / "f0.png")
    _write_rgb_png(folder / "f1.png")
    w = Show3D.from_folder(folder, watch=False, verbose=False, apply_config_transforms=False)
    assert w.is_rgb is True
    assert w.n_slices == 2
    assert w.height == 24 and w.width == 32
    assert w._rgb_data is not None
    assert w._rgb_data.shape == (2, 24, 32, 3)
    # frame_bytes should be RGB float32, 3 channels
    assert len(w.frame_bytes) == 24 * 32 * 3 * 4


def test_show3d_direct_rgb_array():
    rgb = np.zeros((5, 16, 20, 3), dtype=np.float32)
    rgb[..., 0] = 1.0
    w = Show3D(rgb, verbose=False)
    assert w.is_rgb is True
    assert w.n_slices == 5
    assert w._rgb_data.shape == (5, 16, 20, 3)
