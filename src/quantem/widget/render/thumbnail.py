"""Lightweight visual thumbnails for scientific arrays.

These helpers are for previews only: folder browsers, HTML reports, smoke-test
dashboards, and static notebook review surfaces. They intentionally emit image
bytes, not quantitative data. Keep arrays in scientific file formats when
values need to be measured or reprocessed.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Literal

import numpy as np


ImageFormat = Literal["webp", "png", "jpeg", "jpg"]


def thumbnail_image(
    array,
    *,
    size: int = 256,
    cmap: str = "inferno",
    percentiles: tuple[float, float] = (1.0, 99.0),
):
    """Return a PIL image preview with the longest side equal to ``size``.

    Parameters
    ----------
    array
        2D scalar image, RGB image, or RGBA image. Scalar images are contrast
        stretched by percentile and colorized with a matplotlib colormap. RGB
        and RGBA arrays are scaled to ``uint8`` without applying a colormap.
    size
        Maximum output width or height in pixels.
    cmap
        Matplotlib colormap name for scalar input.
    percentiles
        Low/high percentile contrast window for scalar input.

    Returns
    -------
    PIL.Image.Image
        RGB preview image.
    """

    if int(size) <= 0:
        raise ValueError(f"size must be positive, got {size}")
    from PIL import Image

    frame = np.asarray(array)
    if frame.ndim == 2:
        img = _scalar_to_image(frame, cmap=cmap, percentiles=percentiles)
    elif frame.ndim == 3 and frame.shape[-1] in (3, 4):
        img = Image.fromarray(
            _rgb_to_uint8(frame),
            mode="RGBA" if frame.shape[-1] == 4 else "RGB",
        )
        img = img.convert("RGB")
    else:
        raise ValueError(
            "thumbnail_image expects a 2D scalar array or an RGB/RGBA image; "
            f"got shape {frame.shape}"
        )
    return _resize_longest_side(img, int(size))


def thumbnail_bytes(
    array,
    *,
    size: int = 256,
    cmap: str = "inferno",
    percentiles: tuple[float, float] = (1.0, 99.0),
    format: ImageFormat = "webp",
    quality: int = 85,
) -> bytes:
    """Return encoded thumbnail bytes.

    ``format="webp"`` is the default because it is compact for folder previews
    and reports. Use PNG only when lossless visual pixels are required.
    """

    fmt = _normalize_format(format)
    img = thumbnail_image(array, size=size, cmap=cmap, percentiles=percentiles)
    buf = io.BytesIO()
    save_kwargs = _save_kwargs(fmt, quality=quality)
    img.save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()


def thumbnail_webp(
    array,
    *,
    size: int = 256,
    cmap: str = "inferno",
    percentiles: tuple[float, float] = (1.0, 99.0),
    quality: int = 85,
) -> bytes:
    """Return a compact WebP thumbnail for visual preview surfaces."""

    return thumbnail_bytes(
        array,
        size=size,
        cmap=cmap,
        percentiles=percentiles,
        format="webp",
        quality=quality,
    )


def save_thumbnail(
    array,
    path: str | Path,
    *,
    size: int = 256,
    cmap: str = "inferno",
    percentiles: tuple[float, float] = (1.0, 99.0),
    format: ImageFormat | None = None,
    quality: int = 85,
) -> Path:
    """Write a thumbnail image and return the output path."""

    out = Path(path).expanduser()
    fmt = _normalize_format(format or _format_from_suffix(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(
        thumbnail_bytes(
            array,
            size=size,
            cmap=cmap,
            percentiles=percentiles,
            format=fmt,
            quality=quality,
        )
    )
    return out


def _scalar_to_image(
    array: np.ndarray,
    *,
    cmap: str,
    percentiles: tuple[float, float],
):
    from matplotlib import colormaps
    from PIL import Image

    frame = np.asarray(array, dtype=np.float32)
    if frame.ndim != 2:
        raise ValueError(f"scalar thumbnail input must be 2D, got shape {frame.shape}")
    finite = frame[np.isfinite(frame)]
    if finite.size == 0:
        scaled = np.zeros(frame.shape, dtype=np.float32)
    else:
        p_low, p_high = _validate_percentiles(percentiles)
        lo = float(np.percentile(finite, p_low))
        hi = float(np.percentile(finite, p_high))
        if hi <= lo:
            scaled = np.zeros(frame.shape, dtype=np.float32)
        else:
            scaled = np.clip((frame - lo) / (hi - lo), 0.0, 1.0)
            scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    if cmap.strip().lower() in {"gray", "grey"}:
        return Image.fromarray((scaled * 255).astype(np.uint8), mode="L").convert("RGB")
    cm = colormaps.get_cmap(cmap)
    rgb = (cm(scaled)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _rgb_to_uint8(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32, copy=False)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    if float(finite.max()) <= 1.0 and float(finite.min()) >= 0.0:
        scaled = arr * 255.0
    else:
        scaled = np.clip(arr, 0.0, 255.0)
    return np.nan_to_num(scaled, nan=0.0, posinf=255.0, neginf=0.0).astype(np.uint8)


def _resize_longest_side(img, size: int):
    from PIL import Image

    width, height = img.size
    longest = max(width, height)
    if longest <= 0 or longest == size:
        return img
    scale = size / longest
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def _validate_percentiles(percentiles: tuple[float, float]) -> tuple[float, float]:
    if len(percentiles) != 2:
        raise ValueError("percentiles must contain exactly two values")
    p_low = float(percentiles[0])
    p_high = float(percentiles[1])
    if not 0.0 <= p_low < p_high <= 100.0:
        raise ValueError(
            "percentiles must satisfy 0 <= low < high <= 100; "
            f"got {percentiles}"
        )
    return p_low, p_high


def _normalize_format(format: str) -> str:
    fmt = str(format).strip().lower()
    if fmt == "jpg":
        fmt = "jpeg"
    if fmt not in {"webp", "png", "jpeg"}:
        raise ValueError("format must be 'webp', 'png', 'jpeg', or 'jpg'")
    return "JPEG" if fmt == "jpeg" else fmt.upper()


def _format_from_suffix(path: Path) -> ImageFormat:
    suffix = path.suffix.lower().lstrip(".")
    if suffix in {"webp", "png", "jpg", "jpeg"}:
        return suffix  # type: ignore[return-value]
    return "webp"


def _save_kwargs(format: str, *, quality: int) -> dict:
    if format == "WEBP":
        return {"quality": int(quality), "method": 4}
    if format == "JPEG":
        return {"quality": int(quality), "optimize": True}
    return {}
