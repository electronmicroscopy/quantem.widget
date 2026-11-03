"""Colormap names and the one Python LUT application shared by every widget.

The browser bundle's GPUColormapEngine knows the names in ``Colormap`` (keep
in step with the js/display/luts.ts COLORMAPS table). Static renders (PNG
fallback, save_image, GIF frames) map pixels to levels through
``contrast_uint8`` (after ``signed_log1p`` under log scale) and colorize through
``colorize`` so a saved pixel sits at the LUT position the live canvas paints
for the same value. The colors come from matplotlib's tables, which differ from
the browser's ``colormaps.json`` LUTs (see ``colorize``).

``colormaps.json`` holds the control points of the display LUTs that Plot2D
draws (the table quantem.gpu's display kernels use); ``colormap_lut`` expands
them to 256 entries.
"""

import json
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

import numpy as np
from matplotlib import colormaps


class Colormap(StrEnum):
    """Colormaps the widgets accept, by frontend name."""

    INFERNO = "inferno"
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    MAGMA = "magma"
    HOT = "hot"
    GRAY = "gray"
    HSV = "hsv"
    TURBO = "turbo"
    CIVIDIS = "cividis"
    RDBU = "RdBu"
    RDBU_R = "RdBu_r"
    SEISMIC = "seismic"
    TWILIGHT = "twilight"
    TWILIGHT_SHIFTED = "twilight_shifted"


VALID_CMAPS = frozenset(member.value for member in Colormap)
COLORMAP_POINTS = json.loads((Path(__file__).parent / "colormaps.json").read_text(encoding="utf-8"))


def cmap_to_name(value: str | Colormap) -> str:
    """Frontend colormap name for a string or ``Colormap`` member."""
    return value.value if isinstance(value, Colormap) else str(value)


def is_cmap_sequence(value: object) -> bool:
    """True for a per-panel colormap list; a single string is one colormap."""
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def colorize(normalized: np.ndarray, cmap: str) -> np.ndarray:
    """Map display values in [0, 1] through a matplotlib colormap to uint8 RGB.

    The lookup is ``cmap(normalized)[..., :3] * 255`` truncated to uint8. For a
    256-entry colormap ``colorize(level / 255)`` reads entry ``level``, as the
    browser does; twilight and twilight_shifted have 510 entries, so it reads
    entry ``2 * level`` (509 at the top), the same position along the colormap.
    The tables are matplotlib's, not the control-point LUTs of ``colormaps.json``
    that the browser paints with, so a saved pixel has the canvas's LUT position
    but its color can differ. The input keeps its dtype: the LUT index is
    computed in that precision and a float32 caller must not be promoted
    behind its back. matplotlib does the ``* 255`` truncation on the LUT
    (``bytes=True``) and gathers uint8, the same value per pixel without a
    float64 RGBA image eight times the size of the result.
    """
    rgba = colormaps.get_cmap(cmap)(normalized, bytes=True)
    return np.ascontiguousarray(rgba[..., :3])


def signed_log1p(values):
    """Signed log1p, the browser's log scale (``applyLogScale``): ``log1p(v)``, and ``-log1p(-v)`` below zero.

    Diff and phase data keep their sign under log scale. ``log1p(clip(v, 0))``
    would flatten every negative value onto the bottom of the colormap, so a
    saved image would no longer match the canvas.
    """
    return np.sign(values) * np.log1p(np.abs(values))


def display_window(low: float, high: float, log_scale: bool) -> tuple[float, float]:
    """A data-space window ``[low, high]`` in display space: ``signed_log1p`` of each end under log scale.

    This is the browser's ``displayRange``: the window comes from the data's own
    limits in float64, not from the min/max of the float32 log frame, which can
    sit one float32 step away and move pixels to the neighbouring level.
    save_image, the saved-notebook preview and the Show3D export all window
    through here, so they agree with each other and with the canvas.
    """
    if log_scale:
        return float(signed_log1p(float(low))), float(signed_log1p(float(high)))
    return float(low), float(high)


def auto_window(values: np.ndarray, log_scale: bool) -> tuple[float, float]:
    """Show2D's Auto contrast: the 2-98% percentile window of the display values, or the data window when that collapses.

    Sparse or clustered data (a few bright spots on an empty field) can put
    both percentiles on one value, and that window paints a flat image. It
    falls back to ``display_window`` of the data's own min/max. save_image and
    the saved-notebook preview both take Auto through here.
    """
    display_values = signed_log1p(values) if log_scale else values
    low, high = (float(value) for value in np.percentile(display_values, (2, 98)))
    full_low, full_high = display_window(values.min(), values.max(), log_scale)
    if high - low <= max(1e-12, abs(full_high - full_low) * 1e-6):
        return full_low, full_high
    return low, high


def contrast_uint8(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Display values as the uint8 colormap levels the browser paints in the window ``[vmin, vmax]``.

    ``level = trunc(clip((v - vmin) / (vmax - vmin) * 255, 0, 255))``, the
    rule of the browser's ``applyColormap``; ``colorize(level / 255)`` reads
    the LUT position of entry ``level`` (see ``colorize``). Under log scale the values and the window are both
    already in ``signed_log1p`` space. Every static render (save_image, GIF
    frames, the saved-notebook preview) goes through here so they pick the
    same LUT entry for the same pixel.

    The edge cases follow the browser's ``displayNormalize``: NaN and -inf
    paint level 0 and +inf level 255; an empty window (``vmax <= vmin``, as for
    a constant frame) paints every finite value at 0.5, level 127. Casting NaN
    to uint8 would warn and give a platform-dependent level.
    """
    if vmax > vmin:
        levels = np.clip((values - vmin) / (vmax - vmin) * 255, 0, 255)
    else:
        levels = np.where(np.isfinite(values), 127.5, np.where(values > 0, 255.0, 0.0))
    return np.nan_to_num(levels, nan=0.0).astype(np.uint8)


def colormap_names() -> tuple[str, ...]:
    """Names of the display LUTs, in their table order."""
    return tuple(COLORMAP_POINTS)


def colormap_lut(name: str) -> np.ndarray:
    """One 256-entry float32 RGBA lookup table from the colormap's control points.

    Entry ``i`` sits at position ``i * (n - 1) / 255`` along the ``n`` control
    points; its RGB is the linear interpolation of the two neighbouring points
    in float32, rounded half up to an integer and divided by 255. Alpha is 1.
    These are the same bytes as ``quantem.gpu.display.colormap_lut``, so a
    static figure and the GPU display agree.
    """
    if name not in COLORMAP_POINTS:
        raise ValueError(f"Unknown colormap {name!r}. Choose one of: {', '.join(COLORMAP_POINTS)}.")
    points = np.asarray(COLORMAP_POINTS[name], dtype=np.float32)
    positions = np.linspace(0, len(points) - 1, 256, dtype=np.float32)
    lower = np.floor(positions).astype(np.intp)
    upper = np.minimum(lower + 1, len(points) - 1)
    fraction = (positions - lower)[:, None]
    rgb = points[lower] + fraction * (points[upper] - points[lower])
    rgba = np.empty((256, 4), dtype=np.float32)
    rgba[:, :3] = np.floor(rgb + np.float32(0.5)) / np.float32(255)
    rgba[:, 3] = 1
    return rgba
