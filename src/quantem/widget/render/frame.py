"""Colormapping and PNG encoding for widget image panels."""

from __future__ import annotations

import io

import matplotlib
import numpy as np

NO_DATA_COLOR = "#404040"


def frame_to_rgb(
    frame: np.ndarray,
    *,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    log_scale: bool = False,
) -> np.ndarray:
    """Colormap a 2D float frame into a uint8 ``(H, W, 3)`` RGB array.

    Parameters
    ----------
    frame : np.ndarray
        ``(H, W)`` scalar image.
    cmap : str
        Matplotlib colormap name.
    vmin, vmax : float, optional
        Explicit display range. Defaults to a robust 1st/99th percentile
        auto-contrast when not given.
    log_scale : bool, default=False
        Apply a ``log1p`` stretch before contrast scaling.

    Returns
    -------
    np.ndarray
        ``(H, W, 3)`` uint8 RGB image.
    """
    values = frame.astype(np.float64, copy=False)
    if log_scale:
        values = np.log1p(np.clip(values - np.nanmin(values), 0, None))
    lo = float(np.nanpercentile(values, 1)) if vmin is None else float(vmin)
    hi = float(np.nanpercentile(values, 99)) if vmax is None else float(vmax)
    if hi <= lo:
        hi = lo + 1.0
    normalized = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    colormap = matplotlib.colormaps[cmap].with_extremes(bad=NO_DATA_COLOR)
    rgba = colormap(normalized)
    return (rgba[..., :3] * 255).astype(np.uint8)


def rgb_to_png_bytes(rgb: np.ndarray) -> bytes:
    """Encode a uint8 ``(H, W, 3)`` RGB array as PNG bytes."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()
