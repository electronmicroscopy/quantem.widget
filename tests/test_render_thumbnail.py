from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from quantem.widget.render import (
    save_thumbnail,
    thumbnail_bytes,
    thumbnail_image,
    thumbnail_webp,
)


def _gradient(shape: tuple[int, int] = (20, 40)) -> np.ndarray:
    y, x = np.mgrid[: shape[0], : shape[1]]
    return (x + 3 * y).astype(np.float32)


def _decode(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img.load()
    return img


def test_thumbnail_webp_preserves_aspect_and_renders_nonblank() -> None:
    data = thumbnail_webp(_gradient(), size=16, cmap="inferno")
    img = _decode(data)

    assert img.format == "WEBP"
    assert img.size == (16, 8)
    assert np.asarray(img).std() > 0


def test_thumbnail_bytes_supports_png_and_gray_colormap() -> None:
    data = thumbnail_bytes(_gradient((12, 12)), size=10, cmap="gray", format="png")
    img = _decode(data)

    assert img.format == "PNG"
    assert img.size == (10, 10)
    arr = np.asarray(img.convert("RGB"))
    assert np.all(arr[..., 0] == arr[..., 1])
    assert np.all(arr[..., 1] == arr[..., 2])


def test_save_thumbnail_uses_path_suffix(tmp_path: Path) -> None:
    out = save_thumbnail(_gradient(), tmp_path / "preview.webp", size=18)

    assert out == tmp_path / "preview.webp"
    with Image.open(out) as img:
        assert img.format == "WEBP"
        assert img.size == (18, 9)


def test_thumbnail_image_accepts_rgb_float_image() -> None:
    rgb = np.zeros((10, 20, 3), dtype=np.float32)
    rgb[..., 0] = np.linspace(0, 1, 20)
    rgb[..., 1] = 0.25
    img = thumbnail_image(rgb, size=10)

    assert img.mode == "RGB"
    assert img.size == (10, 5)
    assert np.asarray(img).std() > 0


def test_thumbnail_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="2D scalar"):
        thumbnail_webp(np.zeros((2, 3, 4, 5)))
    with pytest.raises(ValueError, match="percentiles"):
        thumbnail_webp(_gradient(), percentiles=(99, 1))
    with pytest.raises(ValueError, match="format"):
        thumbnail_bytes(_gradient(), format="tiff")  # type: ignore[arg-type]
