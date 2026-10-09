"""Colormap names and the one Python LUT application shared by every widget.

The browser bundle's GPUColormapEngine knows the names in ``Colormap`` (keep
in step with the js/display/colormaps.ts COLORMAPS table). Static renders (PNG
fallback, save_image, GIF frames) colorize through ``colorize`` so a saved
pixel matches what the live canvas paints for the same normalized value.

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

    The lookup is ``cmap(normalized)[..., :3] * 255`` truncated to uint8, the
    same table the browser samples, so a saved frame and the canvas agree
    pixel for pixel. The input keeps its dtype: the LUT index is computed in
    that precision and a float32 caller must not be promoted behind its back.
    """
    rgba = colormaps.get_cmap(cmap)(normalized)
    return (rgba[..., :3] * 255).astype(np.uint8)


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
