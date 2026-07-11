"""
show3d: Interactive 3D stack viewer widget with advanced features.

For viewing a stack of 2D images (e.g., defocus sweep, time series, z-stack, movies).
Includes playback controls, statistics, ROI selection, FFT, and more.
"""

import gc
import html
import http.server
import json
import math
import os
import pathlib
import secrets
import sys
import threading
import time
import tempfile
import urllib.parse
import warnings
import weakref
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from typing import Any, Self

import anywidget
import ipywidgets
import numpy as np
import traitlets

from quantem.widget._image_folder import (
    ImageFolderRecord,
    WatchedImageFolder,
    WatchedImageFolderMixin,
)
from quantem.widget._folder_watch_status import FOLDER_WATCH_STATE_VALUES
from quantem.widget.utils.array import to_numpy
from quantem.widget.utils.recon_config import (
    _config_float,
    _load_quantem_config,
    _normalize_rotation_deg,
    _pixel_size_from_quantem_config,
    _post_crop_from_quantem_config,
    _rotate_stack_inplane,
)
from quantem.widget.show2d import _reject_unknown_kwargs
from quantem.widget.utils.state_io import (
    resolve_widget_version,
    save_state_file,
    unwrap_state_payload,
)
from quantem.widget.utils.static_fallback import StaticFallbackMixin
from quantem.widget.utils.ui import UiMode, resolve_ui_mode

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


class AnimationExportPreview:
    """Notebook-displayable result from Show3D animation exports."""

    def __init__(self, paths: Mapping[str, pathlib.Path]):
        self.paths = {str(label): pathlib.Path(path) for label, path in paths.items()}

    def __repr__(self) -> str:
        entries = ", ".join(f"{label}={path}" for label, path in self.paths.items())
        return f"AnimationExportPreview({entries})"

    def _repr_html_(self) -> str:
        cards: list[str] = []
        for label, path in self.paths.items():
            suffix = path.suffix.lower()
            src = self._html_src(path)
            escaped_label = html.escape(label)
            escaped_src = html.escape(src, quote=True)
            size = self._format_size(path)
            if suffix == ".gif":
                media = (
                    f'<img src="{escaped_src}" alt="{escaped_label}" '
                    'style="display:block;max-width:100%;border:1px solid #444;background:#111">'
                )
            elif suffix == ".mp4":
                media = (
                    f'<video src="{escaped_src}" controls muted loop '
                    'style="display:block;max-width:100%;border:1px solid #444;background:#111"></video>'
                )
            else:
                media = f'<a href="{escaped_src}">{html.escape(path.name)}</a>'
            cards.append(
                "<section>"
                f'<h4 style="margin:0 0 4px;font:600 13px system-ui,sans-serif">{escaped_label}</h4>'
                f"{media}"
                f'<div style="margin-top:4px;color:#555;font:12px system-ui,sans-serif">{html.escape(size)}</div>'
                "</section>"
            )
        return (
            '<div class="show3d-animation-export-preview" '
            'style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,max-content));'
            'gap:16px;align-items:start">'
            + "".join(cards)
            + "</div>"
        )

    @staticmethod
    def _format_size(path: pathlib.Path) -> str:
        size = path.stat().st_size
        mb = size / (1024 * 1024)
        if mb >= 10:
            return f"{mb:.1f} MB"
        if mb >= 1:
            return f"{mb:.2f} MB"
        return f"{size / 1024:.1f} KB"

    @staticmethod
    def _html_src(path: pathlib.Path) -> str:
        try:
            rel = os.path.relpath(path, pathlib.Path.cwd())
            if not rel.startswith(".."):
                return pathlib.Path(rel).as_posix()
        except ValueError:
            pass
        return path.resolve().as_uri()


def _is_show3d_page_dict(value: object) -> bool:
    """Return True for explicit Show3D page descriptors."""
    if not isinstance(value, Mapping):
        return False
    return any(key in value for key in ("stacks", "panels", "data", "array", "images"))


def _normalise_show3d_pages(
    data_args: tuple,
    *,
    panel_titles: list[str] | None,
    page_labels: Sequence[str | None] | None,
) -> tuple[tuple, list[str] | None, int, int, list[str], list[int]]:
    """Flatten paged Show3D input into the existing multi-panel stack path."""
    if len(data_args) != 1:
        return data_args, panel_titles, 1, 0, [], []

    data = data_args[0]
    explicit_page_dicts = (
        isinstance(data, (list, tuple))
        and len(data) > 0
        and all(_is_show3d_page_dict(item) for item in data)
    )
    if explicit_page_dicts:
        page_panel_arrays: list[list[np.ndarray]] = []
        flattened_titles: list[str] = []
        inferred_page_labels: list[str] = []
        panels_per_page: int | None = None
        frames_per_page: int | None = None
        image_shape: tuple[int, int] | None = None
        for page_idx, page in enumerate(data):
            raw_panels = page.get("stacks", page.get("panels", page.get("data", page.get("array", page.get("images")))))
            if raw_panels is None:
                raise ValueError(
                    f"Show3D page {page_idx} is missing stacks/panels/data/array. "
                    "Use {'title': '...', 'stacks': [...]}"
                )
            if isinstance(raw_panels, np.ndarray):
                arr = to_numpy(raw_panels)
                if arr.ndim == 3:
                    panels = [arr]
                elif arr.ndim == 4:
                    panels = [arr[i] for i in range(arr.shape[0])]
                else:
                    raise ValueError(
                        f"Show3D page {page_idx} must contain 3D stacks or a 4D panel stack, "
                        f"got shape {arr.shape}"
                    )
            else:
                panels = []
                for panel_idx, raw_panel in enumerate(raw_panels):
                    arr = to_numpy(raw_panel.array if hasattr(raw_panel, "array") else raw_panel)
                    if arr.ndim == 2:
                        arr = arr[np.newaxis, ...]
                    if arr.ndim != 3:
                        raise ValueError(
                            f"Show3D page {page_idx} panel {panel_idx} must be a 2D image or 3D stack, "
                            f"got shape {arr.shape}"
                        )
                    panels.append(arr)
            if not panels:
                raise ValueError(f"Show3D page {page_idx} must contain at least one panel")
            if panels_per_page is None:
                panels_per_page = len(panels)
                frames_per_page = int(panels[0].shape[0])
                image_shape = tuple(int(v) for v in panels[0].shape[1:])
            elif len(panels) != panels_per_page:
                raise ValueError(
                    "Every Show3D page must contain the same number of panels; "
                    f"page 0 has {panels_per_page}, page {page_idx} has {len(panels)}"
                )
            page_titles = page.get("panel_titles", page.get("labels"))
            for panel_idx, arr in enumerate(panels):
                if int(arr.shape[0]) != frames_per_page or tuple(int(v) for v in arr.shape[1:]) != image_shape:
                    raise ValueError(
                        "Every Show3D page panel must have the same frames and image shape; "
                        f"expected ({frames_per_page}, {image_shape[0]}, {image_shape[1]}), got {arr.shape}"
                    )
                if page_titles is not None:
                    if len(page_titles) != len(panels):
                        raise ValueError(
                            f"Show3D page {page_idx} panel_titles length ({len(page_titles)}) "
                            f"must match its panel count ({len(panels)})"
                        )
                    flattened_titles.append(str(page_titles[panel_idx]))
                elif panel_titles is not None:
                    if len(panel_titles) == len(panels):
                        flattened_titles.append(str(panel_titles[panel_idx]))
                    elif panels_per_page is not None and len(panel_titles) == len(data) * panels_per_page:
                        flattened_titles.append(str(panel_titles[page_idx * panels_per_page + panel_idx]))
                    else:
                        raise ValueError(
                            "panel_titles for paged Show3D must have length panels_per_page "
                            f"({len(panels)}) or n_pages * panels_per_page ({len(data) * len(panels)}), "
                            f"got {len(panel_titles)}"
                        )
                else:
                    flattened_titles.append(f"Panel {panel_idx + 1}")
            page_panel_arrays.append(panels)
            inferred_page_labels.append(str(page.get("title") or page.get("label") or f"Page {page_idx + 1}"))
        n_pages = len(page_panel_arrays)
        panels_per = int(panels_per_page or 1)
        resolved_page_labels = [
            "" if label is None else str(label)
            for label in (page_labels if page_labels is not None else inferred_page_labels)
        ]
        if len(resolved_page_labels) != n_pages:
            raise ValueError(f"page_labels length ({len(resolved_page_labels)}) must match n_pages ({n_pages})")
        flattened = tuple(panel for page in page_panel_arrays for panel in page)
        return flattened, flattened_titles, n_pages, panels_per, resolved_page_labels, [0] * n_pages

    arr = None
    if not isinstance(data, list):
        try:
            arr = to_numpy(data)
        except Exception:
            arr = None
    if arr is not None and arr.ndim == 5:
        n_pages, panels = int(arr.shape[0]), int(arr.shape[1])
        resolved_page_labels = [
            "" if label is None else str(label)
            for label in (page_labels if page_labels is not None else [f"Page {i + 1}" for i in range(n_pages)])
        ]
        if len(resolved_page_labels) != n_pages:
            raise ValueError(f"page_labels length ({len(resolved_page_labels)}) must match n_pages ({n_pages})")
        if panel_titles is not None:
            if len(panel_titles) == panels:
                resolved_titles = [str(label) for _ in range(n_pages) for label in panel_titles]
            elif len(panel_titles) == n_pages * panels:
                resolved_titles = [str(label) for label in panel_titles]
            else:
                raise ValueError(
                    "panel_titles for paged Show3D must have length panels_per_page "
                    f"({panels}) or n_pages * panels_per_page ({n_pages * panels}), got {len(panel_titles)}"
                )
        else:
            resolved_titles = [f"Panel {i + 1}" for _ in range(n_pages) for i in range(panels)]
        flattened = tuple(arr.reshape(n_pages * panels, *arr.shape[2:]))
        return flattened, resolved_titles, n_pages, panels, resolved_page_labels, [0] * n_pages

    return data_args, panel_titles, 1, 0, [], []


def _all_finite(arr: np.ndarray, *, chunk_size: int = 1_000_000) -> bool:
    """Chunked full finite scan without allocating one giant boolean array."""
    flat = np.ravel(arr)
    for start in range(0, flat.size, chunk_size):
        if not np.isfinite(flat[start : start + chunk_size]).all():
            return False
    return True


def _normalize_padding(padding: object) -> tuple[int, int]:
    """Normalize constructor spatial padding to (rows, cols)."""
    if isinstance(padding, bool):
        raise ValueError("padding must be a non-negative int or (rows, cols), not bool")
    if isinstance(padding, (int, np.integer)):
        pad = (int(padding), int(padding))
    else:
        try:
            vals = tuple(padding)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("padding must be a non-negative int or (rows, cols)") from exc
        if len(vals) != 2:
            raise ValueError("padding tuple must contain exactly two values: (rows, cols)")
        if any(isinstance(v, bool) for v in vals):
            raise ValueError("padding values must be non-negative integers, not bool")
        pad = (int(vals[0]), int(vals[1]))
    if pad[0] < 0 or pad[1] < 0:
        raise ValueError(f"padding must be non-negative, got {pad}")
    return pad


def _pad_stack(data: np.ndarray, padding: tuple[int, int],
               mode: str = "median") -> np.ndarray:
    """Pad a (N, H, W) or RGB (N, H, W, 3) stack with a spatial border.

    ``mode="median"`` (default) fills with the stack median so the border is
    contrast-neutral - it sits at the histogram centre and does not shift the
    per-frame percentile clip. ``mode="constant"`` fills with 0 (black), which
    drags the low percentile down and darkens the real image.

    Speed: ``np.pad`` has its own ``mode="median"`` but it recomputes a median
    per pad-vector and is slow on a large stack. We compute ONE scalar median
    (sampled to ~1e6 elements - the border value does not need an exact median)
    and do a constant fill, so the pad itself is a fast memset.
    """
    pad_y, pad_x = padding
    if pad_y == 0 and pad_x == 0:
        return data
    if mode == "constant":
        fill = 0.0
    elif mode == "median":
        flat = data.ravel()
        sample = flat[:: max(1, flat.size // 1_000_000)]
        fill = float(np.median(sample))
    else:
        raise ValueError(f"pad_mode must be 'median' or 'constant', got {mode!r}")
    if data.ndim == 4:
        # RGB stack: pad spatial axes only, leave channel axis alone.
        pad_width = ((0, 0), (pad_y, pad_y), (pad_x, pad_x), (0, 0))
    else:
        pad_width = ((0, 0), (pad_y, pad_y), (pad_x, pad_x))
    return np.pad(data, pad_width, mode="constant", constant_values=fill)


def _normalize_crop(crop: object) -> tuple[int, int, int, int]:
    """Normalize constructor spatial crop to (top, bottom, left, right)."""
    if isinstance(crop, bool):
        raise ValueError("crop must be a non-negative int, (rows, cols), or (top, bottom, left, right), not bool")
    if isinstance(crop, (int, np.integer)):
        c = int(crop)
        vals = (c, c, c, c)
    else:
        try:
            raw = tuple(crop)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("crop must be a non-negative int, (rows, cols), or (top, bottom, left, right)") from exc
        if len(raw) == 2:
            if any(isinstance(v, bool) for v in raw):
                raise ValueError("crop values must be non-negative integers, not bool")
            rows, cols = int(raw[0]), int(raw[1])
            vals = (rows, rows, cols, cols)
        elif len(raw) == 4:
            if any(isinstance(v, bool) for v in raw):
                raise ValueError("crop values must be non-negative integers, not bool")
            vals = tuple(int(v) for v in raw)
        else:
            raise ValueError("crop tuple must contain 2 values (rows, cols) or 4 values (top, bottom, left, right)")
    if any(v < 0 for v in vals):
        raise ValueError(f"crop must be non-negative, got {vals}")
    return vals


def _crop_stack(data: np.ndarray, crop: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a (N, H, W) or RGB (N, H, W, 3) stack by (top, bottom, left, right)."""
    top, bottom, left, right = crop
    if top == bottom == left == right == 0:
        return data
    h, w = int(data.shape[1]), int(data.shape[2])
    if top + bottom >= h or left + right >= w:
        raise ValueError(
            f"crop {crop} removes the entire image: input spatial shape is {(h, w)}"
        )
    row_stop = h - bottom if bottom else h
    col_stop = w - right if right else w
    return data[:, top:row_stop, left:col_stop, ...]


def _json_safe_metadata_value(value: Any) -> Any:
    """Convert common scientific scalar/container values to JSON-safe objects."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe_metadata_value(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_safe_metadata_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_metadata_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _normalize_frame_metadata(metadata: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    """Normalize one frame-metadata sequence to JSON-safe dictionaries."""
    if metadata is None:
        return []
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(metadata):
        if not isinstance(item, Mapping):
            raise TypeError(f"frame_metadata[{idx}] must be a mapping, got {type(item).__name__}")
        out.append(_json_safe_metadata_value(item))
    return out


def _normalize_panel_frame_metadata(
    metadata: Sequence[Sequence[Mapping[str, Any]]] | None,
) -> list[list[dict[str, Any]]]:
    """Normalize per-panel frame metadata to JSON-safe dictionaries."""
    if metadata is None:
        return []
    return [_normalize_frame_metadata(panel_metadata) for panel_metadata in metadata]


class _FormatDict(dict):
    """format_map helper that reports missing metadata keys clearly."""

    def __missing__(self, key: str) -> str:
        raise KeyError(key)


def _format_metadata_default(metadata: Mapping[str, Any]) -> str:
    """Compact, repository-neutral default metadata label."""
    if "label" in metadata and metadata["label"] not in (None, ""):
        return str(metadata["label"])
    parts: list[str] = []
    for key, value in metadata.items():
        if value in (None, ""):
            continue
        parts.append(f"{key}={value}")
    return " · ".join(parts)


def _format_metadata_label(
    metadata: Mapping[str, Any],
    formatter: str | Callable[..., object] | None,
    *,
    frame_idx: int,
    panel_idx: int | None,
) -> str:
    """Format one metadata dictionary as a display label."""
    if not metadata:
        return ""
    if formatter is None:
        return _format_metadata_default(metadata)
    if isinstance(formatter, str):
        try:
            return formatter.format_map(_FormatDict(metadata))
        except KeyError as exc:
            key = exc.args[0]
            raise KeyError(f"frame_label_format references missing metadata key {key!r}") from exc
    try:
        value = formatter(metadata, frame_idx, panel_idx)
    except TypeError:
        try:
            value = formatter(metadata, frame_idx)
        except TypeError:
            value = formatter(metadata)
    return "" if value is None else str(value)


class Colormap(str, Enum):
    """Available colormaps for image display."""

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

    def __str__(self) -> str:
        return self.value


# Names that the JS bundle's GPUColormapEngine knows about. Keep in sync with
# js/colormaps.ts COLORMAPS table.
_VALID_CMAPS = frozenset({
    "inferno", "viridis", "plasma", "magma", "hot", "gray", "hsv", "turbo",
    "cividis", "RdBu", "RdBu_r", "seismic", "twilight", "twilight_shifted",
})

# Keep a single synced playback chunk small enough for the Jupyter/AnyWidget
# Comm path to survive. The data is still exact float32; large stacks move as
# sliding windows instead of browser-hostile 256 MB+ messages.
_MAX_PLAYBACK_CHUNK_BYTES = 128 * 1024 * 1024
_MAX_PLAYBACK_FPS = 30.0


class _Show3DFrameHTTPServer(http.server.ThreadingHTTPServer):
    """Tiny localhost server for exact float32 frame reads."""

    allow_reuse_address = True
    daemon_threads = True


# Rec. 709 luminance weights. A true-color stack keeps its RGB in _rgb_data and
# a single luminance plane in _data so stats, FFT, ROI, and scale bars have one
# scalar channel to work on.
_RGB_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


class Show3D(WatchedImageFolderMixin, StaticFallbackMixin, anywidget.AnyWidget):
    """
    Interactive 3D stack viewer for sequential 2D images.

    Renders an (N, H, W) stack through a WebGPU canvas (CPU fallback when WebGPU
    is unavailable) with a sliding prefetch buffer for smooth playback. The full
    stack is held once on the Python side; scrubbing and playback ship individual
    frames or chunks over the Jupyter Comm channel. Common use cases: defocus
    sweep, time series, depth stack, in-situ movie, side-by-side trial comparison.

    Features
    --------
    - Interactive scrubber + ``play`` / ``pause`` / ``stop`` / ``goto`` with bookmark and loop range
    - Per-frame statistics, log scale, percentile auto-contrast, manual vmin/vmax
    - Diff mode (vs first frame or vs previous frame) for delta visualization
    - ROI tools (circle / square / rectangle / annular) with per-frame timeseries
    - Line profiles sampled across the full stack
    - Page-aware depth/time kymographs for matched single-panel comparisons
    - FFT panel with compact quality labels and optional Hann window
    - Side-by-side multi-panel mode with linked or independent zoom / pan / contrast
    - Panel visibility controls by index or title without removing data
    - Frame hiding (``hide``, ``show``, ``set_hidden``, ``show_all``) without rebuilding
    - Python-side PNG / PDF / TIFF single-frame save via ``save_image``
    - JSON state save/load via ``state_dict`` / ``load_state_dict`` / ``save``
    - Explicit ``free`` to release VRAM/RAM held by traitlets observers

    Parameters
    ----------
    data : array_like
        3D array of shape (N, height, width) where N is the stack dimension.
        Also accepts a 2D array (treated as a single-frame stack), a torch
        tensor (CPU or GPU, any dtype), or a quantem ``Dataset3d``. Complex
        input is rejected; cast to magnitude or phase first.
    labels : list[str] | None, optional
        Labels for each slice (e.g., ``["C10=-500nm", "C10=-400nm", ...]``).
        If ``None``, uses string slice indices.
    frame_metadata : sequence of mapping, optional
        Generic metadata for each frame. If ``labels`` is not provided, entries
        are formatted into labels using ``frame_label_format`` or a compact
        ``key=value`` default. This is repository-neutral: upstream packages can
        pass fields such as ``iteration``, ``defocus_nm``, ``loss``, ``dose``,
        or any other JSON-like values.
    panel_frame_metadata : sequence of sequence of mapping, optional
        Per-panel frame metadata, shaped ``[panel][frame]``. If
        ``panel_frame_labels`` is not provided, entries are formatted into the
        per-panel overlay labels using ``frame_label_format``.
    frame_label_format : str or callable, optional
        Formatter for metadata-derived labels. A string uses Python format
        fields, for example ``"iter {iteration} · df={defocus_nm:.1f} nm"``.
        A callable may accept ``(metadata)``, ``(metadata, frame_idx)``, or
        ``(metadata, frame_idx, panel_idx)`` and is resolved to strings before
        HTML export.
    title : str, optional
        Title to display above the image.
    cmap : str or Colormap, default Colormap.MAGMA
        Colormap name. Use Colormap enum (Colormap.MAGMA, Colormap.VIRIDIS, etc.)
        or string ("magma", "viridis", "gray", "inferno", "plasma").
    vmin : float, optional
        Minimum value for colormap. If None, uses data min.
    vmax : float, optional
        Maximum value for colormap. If None, uses data max.
    sampling : float or tuple of float, optional
        Real-space sampling for the scale bar (lateral). Pairs with ``units``.
    units : str or list of str, optional
        Unit for ``sampling`` (e.g. ``"A"`` for Angstrom, ``"nm"``). Matches the
        quantem ``Dataset`` / ``Show2D`` / ``Show4DSTEM`` convention.
    log_scale : bool, default False
        Use log scale for intensity mapping.
    auto_contrast : bool, default True
        Use percentile-based contrast (ignores vmin/vmax).
    link_contrast : bool, optional
        Share one contrast range across panels. Paged data defaults to
        ``False`` so each reconstruction uses its own automatic percentile
        range; ordinary multi-panel data defaults to ``True``. Pass ``True``
        for a matched physical scale across every page and panel.
    percentile_low : float, default 0.5
        Lower percentile for auto-contrast.
    percentile_high : float, default 99.5
        Upper percentile for auto-contrast.
    fps : float, default 30.0
        Frames per second for playback, capped at 30.
    timestamps : list of float, optional
        Timestamps for each frame (e.g., seconds or dose values).
    timestamp_unit : str, default "s"
        Unit for timestamps (e.g., "s", "ms", "e/A2").
    show_fft : bool, default False
        Show the FFT for the current frame and visible panels.
    fft_layout : {"bottom", "right", "overlay"}, default "bottom"
        Place the FFT below the real-space grid, beside it, or as one inset in
        every visible panel.
    fft_overlay_position : {"top-left", "top-right", "bottom-left", "bottom-right"}, default "top-left"
        Initial corner for inset FFTs. Inset overlays can be dragged and snap
        to the nearest corner.
    fft_overlay_size : float, default 0.35
        Inset FFT size as a fraction of its real-space panel, from 0.2 to 0.7.
    fft_overlay_zoom : float, default 1.0
        Initial shared FFT magnification for bottom, right, and overlay
        layouts. Every FFT tile or inset displays this live value as an
        ``N.N×`` badge while ``show_zoom_indicator`` is enabled.
    fft_window : bool, default True
        Apply a Hann window before computing the FFT.
    fft_metrics : bool, default True
        Show cached FFT sharpness, peak-count, and peak-SNR labels.
    size : int, default 0
        Canvas rendering size in CSS pixels (the on-screen width of the main
        viewport).  ``0`` uses the frontend default (500 px).  Pass e.g.
        ``size=800`` to enlarge for a presentation, or ``size=300`` to compress
        alongside a control panel.  This controls **display only** - the
        underlying stack resolution is never resampled; scrubbing and zoom
        still see every pixel of the full-resolution frame.
    panel_width_px : int, default 0
        Display width for each panel in CSS pixels. When ``>0``, this wins over
        ``size`` and the frontend derives the total grid canvas from the active
        column count. This is display-only; the synced ``panel_width_px`` trait
        still records source panel geometry for multi-panel frame slicing.
    padding : int or tuple[int, int], default 0
        Zero-valued spatial border added to every frame before display. Use
        this when aligned movies need extra field of view so shifted frames do
        not crop at the canvas edge. An int pads rows and columns equally;
        ``(rows, cols)`` pads each axis independently.
    crop : int or tuple[int, int] or tuple[int, int, int, int], default 0
        Spatial crop applied before padding. Use an int to crop all sides,
        ``(rows, cols)`` for symmetric row/column cropping, or
        ``(top, bottom, left, right)`` for side-specific cropping.
    max_cols : int, default 4
        Multi-panel grid wrap.  ``0`` = single row (no wrap), ``N>0`` = wrap into
        rows of at most ``N`` panels.  Default ``4`` is a good fit for a
        13"–16" laptop screen; bump to ``6`` on wide monitors or drop to ``3``
        for a portrait split layout.  Empty trailing cells in a partial last
        row are not rendered (transparent, non-interactive).
    panel_gap : int, default 0
        Gap in CSS pixels between adjacent panels.  ``0`` = flush (panels share
        an edge - useful for tiled montages), ``20`` = roomy (clear separation
        for slides).  Single-panel widgets ignore this.
    panel_title_font_size : int, default 11
        Font size in CSS pixels for the per-panel title drawn at the top of
        each multi-panel slot.  Bump to ``14–16`` for slide-projection clarity;
        drop to ``9`` to fit titles inside narrow panels on a small screen.
    hidden_panels : sequence of int or str, optional
        Multi-panel entries to collapse from the initial view. Integers are
        zero-based panel indices; strings match ``panel_titles`` exactly. Hidden
        panels stay in the widget state and exported HTML, and can be restored
        later with ``show_panel`` or ``show_all_panels``.
    panel_order : sequence of int or str, optional
        Multi-panel display order. Integers are zero-based panel indices;
        strings match ``panel_titles`` exactly. Reordering is display-only:
        source data, titles, stars, and hidden-panel state stay keyed by their
        original panel indices.
    show_panel_titles : bool, default True
        Draw the top-center per-panel title and frame counter on multi-panel
        canvases. Set ``False`` for clean GIF/video exports.
    show_resize_handles : bool, default True
        Render the bottom-right corner triangle on every real panel.  Dragging
        any handle resizes the entire multi-panel canvas (linked).  Set
        ``False`` to declutter a screenshot or printed figure where the
        operator already has the layout they want.
    show_zoom_indicator : bool, default True
        Draw the live ``1.0×``-style zoom readout at the bottom-left of every
        real-space panel and every FFT tile or inset. Set ``False`` for clean
        static layouts or when the scale bar alone is enough to communicate
        scale.
    show_scale_bar : bool, default True
        Draw the bottom-right scale bar. When ``pixel_size`` is provided the
        label uses physical units; otherwise it shows pixel units. Set
        ``False`` for GIF/video exports or uncluttered figure captures.
    show_controls : bool, default True
        Show the live control UI. Set ``False`` for a permanently clean display.
    controls_collapsed : bool, default False
        Start with the live control UI collapsed behind a small GUI toggle.
        Unlike ``show_controls=False``, users can expand the controls in the
        frontend and Python can call ``expand_controls()`` later.
    debug : bool, default False
        Show a compact frontend FPS/debug badge in the widget title row.
    save_state : bool, default False
        When False, saved notebooks omit heavy stack buffers and keep a compact
        static preview for cold reopen. Set True only for small widgets that
        must reopen interactively without rerunning the kernel.
    notebook_preview_format : {"jpeg", "webp", "png"} or None, default "jpeg"
        Static preview format used when ``save_state=False``. ``"jpeg"`` is the
        most portable notebook default, ``"webp"`` is smaller for local/report
        workflows, ``"png"`` is lossless but larger, and ``None`` disables the
        preview.
    notebook_preview_quality : int, default 88
        Lossy preview quality for JPEG/WebP, from 1 to 100. Ignored for PNG.
    notebook_preview_max_px : int, default 512
        Longest panel side for the saved-notebook preview. Lower values make
        notebooks smaller; higher values make the static fallback sharper.
    denoise : str, default "none"
        Display-only denoise method for sparse map stacks (EDS, low dose),
        applied to every playback frame and to the FFT view. Three orthogonal
        choices: ``"none"``, ``"gaussian"``, or ``"anscombe"`` (Poisson
        count-respecting smoothing); binning is the separate ``denoise_bin``
        knob. Recommendation ladder: sparse EDS -> ``"anscombe"`` with
        ``denoise_bin=2`` and sigma 6-10; very sparse -> ``"anscombe"`` with
        ``denoise_bin=4`` and sigma 8-12; decent-dose HAADF -> ``"gaussian"``
        sigma 1-2 or ``"none"``; anything quantitative -> ``"none"``. The
        compound spellings ``"bin2"``, ``"bin2_anscombe"`` and
        ``"bin4_anscombe"`` stay accepted as aliases that fold into (mode, bin);
        ``"tv"``/``"denova*"`` remain available from Python (not in the UI
        menu). Pure view transform: the stored stack, the stats row, and every
        export of raw data keep the original counts, and the lossless default is
        ``"none"``. When active, a one-line banner announces the reduction and
        how to get raw counts back. An active filter also reshapes the FFT view,
        so set ``denoise="none"`` for quantitative FFT work.
    denoise_sigma : float, default 4.0
        Smoothing scale in pixels for the Gaussian/Anscombe display filters.
    denoise_bin : {1, 2, 4}, default 1
        Display-side 2x bin passes for SNR, combined with ``denoise``. ``1``
        (the default) is lossless. Independent of ``display_bin`` (the GPU
        display budget knob).
    show_denoise : bool, default False
        Show the denoise controls row. Hidden by default to keep the widget
        clean; auto-enabled when the stack starts with an active denoise. An
        active reduction always shows its banner, even with the row hidden. The
        deprecated aliases ``display_filter``, ``display_sigma`` and
        ``spatial_bin`` are still accepted for one release and map onto
        ``denoise``, ``denoise_sigma`` and ``denoise_bin``.
    Attributes
    ----------
    render_total_ms : int or None
        End-to-end wall clock from constructor start to first browser paint,
        populated by a JS→Python round-trip after the first canvas render.
        ``None`` until the browser has actually painted; also printed to stdout
        when it fires.  Use to triage "is it Python, wire, or the browser?"
        during live acquisitions.
    render_python_build_ms : int or None
        Subset of ``render_total_ms`` covering Python ``__init__`` only.
    render_wire_js_ms : int or None
        Subset covering everything after Python returns: Comm transfer, JS
        decode, colormap, and canvas paint.

    Example
    -------
    Instantiate Show3D with a stack of 2D images, drive playback, and persist
    the display state to disk:

    >>> import numpy as np
    >>> from quantem.widget import Show3D
    >>> stack = np.random.rand(12, 256, 256).astype(np.float32)
    >>> labels = [f"C10={c:.0f}nm" for c in np.linspace(-500, -200, 12)]
    >>> w = Show3D(stack, labels=labels, title="Defocus Sweep", sampling=0.5, units="A")
    >>> w.play()  # doctest: +SKIP
    >>> w.goto(3)  # doctest: +SKIP
    >>> w.save("show3d_state.json")  # doctest: +SKIP

    Notes
    -----
    - The stack is loaded once into ``self._data`` (``float32``); ``set_image``
      replaces the data without rebuilding the widget.
    - For live acquisitions or reconstruction loops where the stack grows over
      time, construct the widget with ``offline=False`` before calling
      ``set_image``. Small initial stacks can otherwise use the saved/offline
      notebook representation, which is meant for static notebook state and
      standalone exports rather than streaming new frames.
    - ``display_bin="auto"`` keeps Show3D source pixels unchanged. For heavy
      multi-panel movies or standalone exports, pass an explicit
      ``display_bin=N`` to build a smaller display stack. This is a display
      tradeoff: it improves first load and playback, while native-pixel viewing
      requires ``display_bin=1`` or a separate high-resolution workflow.
    - Multi-panel mode is auto-detected from input shape and configured via
      ``n_panels``, ``panel_titles``, and ``link_panels``.
    - Call ``free()`` before discarding the widget; ``del`` alone will not
      release VRAM because traitlets observers pin the refcount.
    """

    _esm = pathlib.Path(__file__).parent / "static" / "show3d.js"

    # =========================================================================
    # Core State
    # =========================================================================
    # GPU memory budget for display buffers (same as Show2D)
    _GPU_DISPLAY_BUDGET_MB = 2500

    slice_idx = traitlets.CInt(0).tag(sync=True)
    n_slices = traitlets.Int(1).tag(sync=True)
    folder_waiting = traitlets.Bool(False).tag(sync=True)
    folder_status = traitlets.Unicode("").tag(sync=True)
    folder_watch_state = traitlets.Enum(
        values=FOLDER_WATCH_STATE_VALUES,
        default_value="hidden",
    ).tag(sync=True)
    folder_watch_detail = traitlets.Unicode("").tag(sync=True)
    height = traitlets.Int(1).tag(sync=True)
    width = traitlets.Int(1).tag(sync=True)
    # Display-only denoise/bin for sparse map stacks (EDS, low dose). Applied
    # per frame at every display wire-out (buffer window, frame_bytes, offline
    # pack, frame server); the stored stack is never modified and the lossless
    # default is "none". The FFT panel computes from the displayed frame, so
    # an active filter also affects the FFT view; set denoise="none"
    # for a raw FFT. Independent of display_bin (the GPU budget knob).
    denoise = traitlets.Unicode("none").tag(sync=True)
    denoise_sigma = traitlets.Float(4.0).tag(sync=True)
    denoise_bin = traitlets.Int(1).tag(sync=True)
    denoise_banner = traitlets.Unicode("").tag(sync=True)
    # The denoise controls row is hidden by default; auto-enabled when the
    # widget starts with an active denoise (house rule: knobs that explain a
    # reduced view must be discoverable).
    show_denoise = traitlets.Bool(False).tag(sync=True)
    frame_bytes = traitlets.Bytes(b"").tag(sync=True)
    # True when the stack is true-color RGB (PNG figure frames, color composites).
    # frame_bytes then carries (H*W*3) float32 RGB in [0, 1] instead of gray H*W.
    is_rgb = traitlets.Bool(False).tag(sync=True)
    # Monotonic counter incremented each time frame_bytes is written. Defensive
    # against the case where traitlets.Bytes identity-compares to suppress the
    # trait change event when JS sees the same DataView wrapper - JS subscribes
    # to this counter as a guaranteed-changing dep so render effects always
    # re-fire on slice scrubs / playback ticks.
    frame_seq = traitlets.Int(0).tag(sync=True)
    # Offline/export mode packs the full display stack so JS can slice
    # client-side without a Python kernel. _offline_stack is the compact
    # uint8 path; _offline_float_stack is exact float32 for precision-preserving
    # standalone HTML exports.
    offline = traitlets.Bool(False).tag(sync=True)
    # True only on a clone written by export_html: forces the standalone HTML to
    # render on a light/white background regardless of the viewer's OS theme.
    # Decoupled from `offline` (which selects uint8 vs float data packing).
    _export_light = traitlets.Bool(False).tag(sync=True)
    _static_fallback_jpeg = traitlets.Unicode("").tag(sync=True)
    _static_fallback_mime = traitlets.Unicode("image/jpeg").tag(sync=True)
    _offline_stack = traitlets.Bytes(b"").tag(sync=True)
    _offline_float_stack = traitlets.Bytes(b"").tag(sync=True)
    _offline_min = traitlets.Float(0.0).tag(sync=True)
    _offline_max = traitlets.Float(1.0).tag(sync=True)
    _offline_mins = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    _offline_maxs = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    # Frontend-triggered standalone HTML export. The request is JSON so repeated
    # exports of the same mode can include a unique id and still sync.
    export_request = traitlets.Unicode("").tag(sync=True)
    export_status = traitlets.Unicode("").tag(sync=True)
    export_enabled = traitlets.Bool(True).tag(sync=True)
    export_payload = traitlets.Bytes(b"").tag(sync=True)
    export_payload_id = traitlets.Unicode("").tag(sync=True)
    export_filename = traitlets.Unicode("").tag(sync=True)
    handoff_request = traitlets.Unicode("").tag(sync=True)
    handoff_status = traitlets.Unicode("").tag(sync=True)
    handoff_enabled = traitlets.Bool(True).tag(sync=True)
    prepared_view_widget = traitlets.Instance(ipywidgets.Widget, allow_none=True).tag(
        sync=True,
        **ipywidgets.widget_serialization,
    )
    # Flipped True by JS after the first colormap pass has painted to canvas.
    # Drives the truthful timing print (end-to-end, not __init__-only).
    _js_rendered = traitlets.Bool(False).tag(sync=True)
    labels = traitlets.List(traitlets.Unicode()).tag(sync=True)
    panel_frame_labels = traitlets.List(default_value=[]).tag(sync=True)
    frame_metadata = traitlets.List(default_value=[]).tag(sync=True)
    panel_frame_metadata = traitlets.List(default_value=[]).tag(sync=True)
    frame_label_format = traitlets.Unicode("").tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    show_title = traitlets.Bool(True).tag(sync=True)
    cmap = traitlets.Unicode("plasma").tag(sync=True)
    dim_label = traitlets.Unicode("Frame").tag(sync=True)
    dim_sampling = traitlets.Float(1.0).tag(sync=True)
    dim_unit = traitlets.Unicode("").tag(sync=True)

    # Multi-Panel (side-by-side stacks, independent zoom by default with optional link)
    n_panels = traitlets.Int(1).tag(sync=True)
    panel_titles = traitlets.List(traitlets.Unicode()).tag(sync=True)
    panel_width_px = traitlets.Int(0).tag(sync=True)
    shared_panel_source = traitlets.Bool(False).tag(sync=True)
    separate_panel_frames = traitlets.Bool(False).tag(sync=True)
    # Per-panel "best frame" marker. One int per panel; -1 = unset. Used to flag
    # the user's preferred iteration / trial / focal slice without losing the
    # full stack. JS draws a gold star top-right of each panel when set.
    starred = traitlets.List(traitlets.Int()).tag(sync=True)
    # Per-panel visibility. Hidden panels stay in state/export but are collapsed
    # from the canvas grid and skipped by panel-scoped frontend computations.
    hidden_panels = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    panel_order = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    # Real frame count per panel for stack comparison: stacks of different
    # lengths get auto-padded to the longest; this trait lets JS mark
    # "end-of-stack" frames (frame idx >= real[panel]). Empty = all real.
    panel_real_frames = traitlets.List(traitlets.Int()).tag(sync=True)
    n_pages = traitlets.Int(1).tag(sync=True)
    page_idx = traitlets.Int(0).tag(sync=True)
    panels_per_page = traitlets.Int(0).tag(sync=True)
    page_labels = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    page_starred = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    # Slot-level visibility for paged widgets. Hiding slot 1 means "hide the
    # second panel on every page", so page scrubbing does not bring it back.
    hidden_page_slots = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    # Single Link toggle controls both zoom and pan (independent axes proved confusing).
    link_panels = traitlets.Bool(True).tag(sync=True)
    link_contrast = traitlets.Bool(True).tag(sync=True)  # share vmin/vmax across panels
    view_state = traitlets.Dict(default_value={}).tag(sync=True)
    # 0 = single row (no wrap). N > 0 = wrap into rows of at most N panels.
    max_cols = traitlets.Int(4).tag(sync=True)
    # Per-widget customization for multi-panel display.
    show_resize_handles = traitlets.Bool(True).tag(sync=True)
    show_zoom_indicator = traitlets.Bool(True).tag(sync=True)
    show_panel_titles = traitlets.Bool(True).tag(sync=True)
    panel_title_font_size = traitlets.Int(11).tag(sync=True)
    panel_gap = traitlets.Int(0).tag(sync=True)
    # Hover-x hide feature: enables UI to drop frames from scrubber without
    # rebuilding the widget. hidden_indices is the live state; visible_indices
    # is derived (read-only).
    hideable = traitlets.Bool(False).tag(sync=True)
    hidden_indices = traitlets.List(traitlets.Int()).tag(sync=True)

    # =========================================================================
    # Playback Controls
    # =========================================================================
    playing = traitlets.Bool(False).tag(sync=True)
    reverse = traitlets.Bool(False).tag(sync=True)  # Play in reverse direction
    boomerang = traitlets.Bool(True).tag(sync=True)  # Ping-pong playback
    fps = traitlets.Float(30.0).tag(sync=True)
    # Moving-window time average (temporal binning for noisy data). 1 = off.
    # The displayed frame is the mean of `avg_window` consecutive frames.
    # Edge behavior: the window slides inward to stay FULL-WIDTH rather than
    # shrinking - so frame 0 with window 5 averages frames [0..4], and the last
    # frame averages the final 5. This keeps denoising strength constant across
    # the stack, at the cost of the average not being centered on the displayed
    # index near the ends (frame 0 shows the mean centered ~2 frames in). Even
    # windows are front-biased (window 4 = [idx-2 .. idx+1]).
    avg_window = traitlets.Int(1).tag(sync=True)
    loop = traitlets.Bool(True).tag(sync=True)
    loop_start = traitlets.Int(0).tag(sync=True)  # Start frame for loop range
    loop_end = traitlets.Int(-1).tag(sync=True)  # End frame for loop (-1 = last)
    bookmarked_frames = traitlets.List(traitlets.Int()).tag(sync=True)
    playback_path = traitlets.List(traitlets.Int()).tag(sync=True)

    # =========================================================================
    # Statistics Panel
    # =========================================================================
    show_controls = traitlets.Bool(True).tag(sync=True)
    controls_collapsed = traitlets.Bool(False).tag(sync=True)
    show_stats = traitlets.Bool(False).tag(sync=True)
    debug = traitlets.Bool(False).tag(sync=True)
    stats_mean = traitlets.Float(0.0).tag(sync=True)
    stats_min = traitlets.Float(0.0).tag(sync=True)
    stats_max = traitlets.Float(0.0).tag(sync=True)
    stats_std = traitlets.Float(0.0).tag(sync=True)
    # =========================================================================
    # Display Options
    # =========================================================================
    log_scale = traitlets.Bool(False).tag(sync=True)
    auto_contrast = traitlets.Bool(True).tag(sync=True)
    percentile_low = traitlets.Float(0.5).tag(sync=True)
    percentile_high = traitlets.Float(99.5).tag(sync=True)
    image_vmin_pct = traitlets.Float(0.0).tag(sync=True)
    image_vmax_pct = traitlets.Float(100.0).tag(sync=True)
    vmin = traitlets.Float(None, allow_none=True).tag(sync=True)
    vmax = traitlets.Float(None, allow_none=True).tag(sync=True)
    vmin_per_panel = traitlets.List(traitlets.Float(None, allow_none=True), default_value=[]).tag(sync=True)
    vmax_per_panel = traitlets.List(traitlets.Float(None, allow_none=True), default_value=[]).tag(sync=True)
    data_min = traitlets.Float(0.0).tag(sync=True)
    data_max = traitlets.Float(0.0).tag(sync=True)
    auto_vmins = traitlets.List(traitlets.Float()).tag(sync=True)
    auto_vmaxs = traitlets.List(traitlets.Float()).tag(sync=True)

    # =========================================================================
    # Scale Bar
    # =========================================================================
    pixel_size = traitlets.Float(0.0).tag(sync=True)  # 0 = no scale bar
    pixel_unit = traitlets.Unicode("A").tag(sync=True)
    scale_bar_visible = traitlets.Bool(True).tag(sync=True)
    # Canvas smoothing: False = nearest-neighbor (sharp atoms); True = bilinear.
    smooth = traitlets.Bool(True).tag(sync=True)
    # Whole-stack rotation as k * 90 deg (k = 0..3). Applied in Python by rotating
    # _data and re-broadcasting frame_bytes; cheap for typical EM stacks.
    image_rotation = traitlets.Int(0).tag(sync=True)

    # =========================================================================
    # Timestamps / Dose
    # =========================================================================
    timestamps = traitlets.List(traitlets.Float()).tag(sync=True)
    timestamp_unit = traitlets.Unicode("s").tag(sync=True)

    # =========================================================================
    # ROI Selection
    # =========================================================================
    roi_active = traitlets.Bool(False).tag(sync=True)
    roi_list = traitlets.List([]).tag(sync=True)
    roi_selected_idx = traitlets.Int(-1).tag(sync=True)
    # roi_stats: Python-only readout. JS computes its own ROI stats locally
    # (allRoiStats useMemo in index.tsx), so don't ship over Comm every ROI move.
    roi_stats = traitlets.Dict({})
    roi_plot_data = traitlets.Bytes(b"").tag(sync=True)
    # =========================================================================
    # Sizing
    # =========================================================================
    size = traitlets.Int(0).tag(sync=True)  # Canvas rendering size in CSS pixels; 0 = frontend default

    # =========================================================================
    # Diff Mode
    # =========================================================================
    diff_mode = traitlets.Unicode("off").tag(sync=True)

    # =========================================================================
    # Analysis Panels (FFT + Histogram shown together)
    # =========================================================================
    show_fft = traitlets.Bool(False).tag(sync=True)
    fft_layout = traitlets.Unicode("bottom").tag(sync=True)
    fft_overlay_position = traitlets.Unicode("top-left").tag(sync=True)
    fft_overlay_size = traitlets.Float(0.35).tag(sync=True)
    fft_overlay_zoom = traitlets.Float(1.0).tag(sync=True)
    fft_window = traitlets.Bool(True).tag(sync=True)
    fft_metrics = traitlets.Bool(True).tag(sync=True)
    widget_version = traitlets.Unicode("unknown")  # Python-only: telemetry readout
    # =========================================================================
    # Line Profile
    # =========================================================================
    profile_line = traitlets.List(traitlets.Dict()).tag(sync=True)
    profile_width = traitlets.Int(1).tag(sync=True)
    # Kymograph (space-time) side panel: when on with a drawn profile line, JS
    # samples that line on every frame from _offline_stack into a static
    # (n_slices, line_length) image - distance along the line on X, frame/time
    # on Y. Single-panel only; JS hides the toggle when n_panels > 1 or the
    # offline stack is absent (it needs every frame client-side to build it).
    show_kymograph = traitlets.Bool(False).tag(sync=True)

    # =========================================================================
    # Playback Buffer (sliding prefetch)
    # =========================================================================
    _buffer_bytes = traitlets.Bytes(b"").tag(sync=True)
    _buffer_start = traitlets.Int(0).tag(sync=True)
    _buffer_count = traitlets.Int(0).tag(sync=True)
    _prefetch_request = traitlets.Int(-1).tag(sync=True)
    frame_server_url = traitlets.Unicode("").tag(sync=True)
    frame_server_version = traitlets.Int(0).tag(sync=True)

    # Browser-side benchmark hook. Setting benchmark_request from Python starts
    # an in-widget measurement in the visible frontend; benchmark_result is
    # written back by JS when sampling finishes.
    benchmark_request = traitlets.Dict({}).tag(sync=True)
    benchmark_result = traitlets.Dict({}).tag(sync=True)

    # Render-time telemetry (set after first browser paint; docstring promises these).
    render_total_ms = traitlets.Int(allow_none=True, default_value=None)
    render_python_build_ms = traitlets.Int(allow_none=True, default_value=None)
    render_wire_js_ms = traitlets.Int(allow_none=True, default_value=None)

    _VALID_DIFF_MODES = {"off", "previous", "first"}

    @traitlets.validate("diff_mode")
    def _validate_diff_mode(self, proposal: dict) -> str:
        """Reject unknown diff modes; only off/previous/first are renderable."""
        val = proposal["value"]
        if val not in self._VALID_DIFF_MODES:
            raise traitlets.TraitError(
                f"Invalid diff_mode '{val}'. Must be one of: {sorted(self._VALID_DIFF_MODES)}"
            )
        return val

    @traitlets.validate("avg_window")
    def _validate_avg_window(self, proposal: dict) -> int:
        val = int(proposal["value"])
        if val < 1:
            raise traitlets.TraitError(f"avg_window must be >= 1, got {val}")
        if val > 15:
            raise traitlets.TraitError(f"avg_window must be <= 15, got {val}")
        return val

    @traitlets.validate("image_vmin_pct", "image_vmax_pct")
    def _validate_image_clip_pct(self, proposal: dict) -> float:
        val = float(proposal["value"])
        if not math.isfinite(val):
            raise traitlets.TraitError(f"{proposal['trait'].name} must be finite, got {val}")
        if val < 0 or val > 100:
            raise traitlets.TraitError(f"{proposal['trait'].name} must be in [0, 100], got {val}")
        return val

    @traitlets.validate("playback_path")
    def _validate_playback_path(self, proposal: dict) -> list:
        """Wrap indices into [0, n_slices) to keep JS in bounds."""
        # Wrap indices to [0, n_slices) so JS never indexes out of bounds.
        val = list(proposal["value"])
        n = max(1, int(self.n_slices))
        return [int(i) % n for i in val]

    @traitlets.validate("vmax")
    def _validate_vmax_ge_vmin(self, proposal: dict) -> float | None:
        """Reject non-finite vmax and enforce vmax >= vmin."""
        new_vmax = proposal["value"]
        if new_vmax is not None:
            if not math.isfinite(new_vmax):
                raise traitlets.TraitError(f"vmax must be finite, got {new_vmax}")
            if self.vmin is not None and new_vmax < self.vmin:
                raise traitlets.TraitError(
                    f"vmax ({new_vmax}) must be >= vmin ({self.vmin})"
                )
        return new_vmax

    @traitlets.validate("vmin")
    def _validate_vmin_le_vmax(self, proposal: dict) -> float | None:
        """Reject non-finite vmin and enforce vmin <= vmax."""
        new_vmin = proposal["value"]
        if new_vmin is not None:
            if not math.isfinite(new_vmin):
                raise traitlets.TraitError(f"vmin must be finite, got {new_vmin}")
            if self.vmax is not None and new_vmin > self.vmax:
                raise traitlets.TraitError(
                    f"vmin ({new_vmin}) must be <= vmax ({self.vmax})"
                )
        return new_vmin

    @traitlets.validate("cmap")
    def _validate_cmap(self, proposal: dict) -> str:
        """Reject unknown colormap names."""
        val = str(proposal["value"])
        if val not in _VALID_CMAPS:
            raise traitlets.TraitError(
                f"Unknown cmap {val!r}. Valid: {sorted(_VALID_CMAPS)}"
            )
        return val

    @traitlets.validate("bookmarked_frames")
    def _validate_bookmarks(self, proposal: dict) -> list:
        """Drop bookmark indices outside [0, n_slices) so markers stay onscreen."""
        # Drop indices outside [0, n_slices). JS draws bookmark markers and
        # raw out-of-range values caused offscreen / negative positions.
        n = max(1, int(self.n_slices))
        return [int(i) for i in proposal["value"] if 0 <= int(i) < n]

    @traitlets.validate("loop_end")
    def _validate_loop_end(self, proposal: dict) -> int:
        """Clamp to [0, n_slices) and enforce loop_end >= loop_start (-1 = last)."""
        # -1 sentinel = "last frame". Otherwise must be >= loop_start.
        val = int(proposal["value"])
        if val < 0:
            return val
        n = max(1, int(self.n_slices))
        val = min(val, n - 1)
        if val < int(self.loop_start):
            raise traitlets.TraitError(
                f"loop_end ({val}) must be >= loop_start ({self.loop_start})"
            )
        return val

    @traitlets.validate("loop_start")
    def _validate_loop_start(self, proposal: dict) -> int:
        """Clamp to [0, n_slices) and enforce loop_start <= loop_end."""
        val = int(proposal["value"])
        n = max(1, int(self.n_slices))
        val = max(0, min(val, n - 1))
        end = int(self.loop_end)
        if end >= 0 and val > end:
            raise traitlets.TraitError(
                f"loop_start ({val}) must be <= loop_end ({end})"
            )
        return val

    @traitlets.validate("pixel_size")
    def _validate_pixel_size(self, proposal: dict) -> float:
        """Reject non-finite or negative pixel sizes."""
        val = float(proposal["value"])
        if math.isnan(val) or math.isinf(val):
            raise traitlets.TraitError(f"pixel_size must be finite, got {val}")
        if val < 0:
            raise traitlets.TraitError(f"pixel_size must be >= 0, got {val}")
        return val

    @traitlets.validate("labels")
    def _validate_labels(self, proposal: dict) -> list:
        """Require labels length to match n_slices or be empty."""
        # Length must match n_slices for label lookups.
        val = list(proposal["value"])
        if val and len(val) != int(self.n_slices):
            raise traitlets.TraitError(
                f"labels length ({len(val)}) must equal n_slices ({self.n_slices}) or be empty"
            )
        return val

    @traitlets.validate("panel_frame_labels")
    def _validate_panel_frame_labels(self, proposal: dict) -> list:
        """Require optional per-panel frame labels to match panel/frame counts."""
        val = [list(labels) for labels in proposal["value"]]
        if not val:
            return []
        n_panels = int(self.n_panels)
        if len(val) != n_panels:
            raise traitlets.TraitError(
                f"panel_frame_labels length ({len(val)}) must equal n_panels ({n_panels}) or be empty"
            )
        n_slices = int(self.n_slices)
        real_frames = list(self.panel_real_frames)
        for panel, labels in enumerate(val):
            allowed = {n_slices}
            if panel < len(real_frames):
                allowed.add(int(real_frames[panel]))
            if len(labels) not in allowed:
                allowed_text = " or ".join(str(n) for n in sorted(allowed))
                raise traitlets.TraitError(
                    f"panel_frame_labels[{panel}] length ({len(labels)}) must equal {allowed_text}"
                )
        return val

    @traitlets.validate("frame_metadata")
    def _validate_frame_metadata(self, proposal: dict) -> list:
        """Require optional common frame metadata to match n_slices."""
        val = _normalize_frame_metadata(proposal["value"])
        if val and len(val) != int(self.n_slices):
            raise traitlets.TraitError(
                f"frame_metadata length ({len(val)}) must equal n_slices ({self.n_slices}) or be empty"
            )
        return val

    @traitlets.validate("panel_frame_metadata")
    def _validate_panel_frame_metadata(self, proposal: dict) -> list:
        """Require optional per-panel frame metadata to match panel/frame counts."""
        val = _normalize_panel_frame_metadata(proposal["value"])
        if not val:
            return []
        n_panels = int(self.n_panels)
        if len(val) != n_panels:
            raise traitlets.TraitError(
                f"panel_frame_metadata length ({len(val)}) must equal n_panels ({n_panels}) or be empty"
            )
        n_slices = int(self.n_slices)
        real_frames = list(self.panel_real_frames)
        for panel, metadata in enumerate(val):
            allowed = {n_slices}
            if panel < len(real_frames):
                allowed.add(int(real_frames[panel]))
            if len(metadata) not in allowed:
                allowed_text = " or ".join(str(n) for n in sorted(allowed))
                raise traitlets.TraitError(
                    f"panel_frame_metadata[{panel}] length ({len(metadata)}) must equal {allowed_text}"
                )
        return val

    @traitlets.validate("timestamps")
    def _validate_timestamps(self, proposal: dict) -> list:
        """Require timestamps length to match n_slices or be empty."""
        # Empty list = no timestamps. Otherwise length must match n_slices.
        val = list(proposal["value"])
        if val and len(val) != int(self.n_slices):
            raise traitlets.TraitError(
                f"timestamps length ({len(val)}) must equal n_slices ({self.n_slices}) or be empty"
            )
        return val

    @traitlets.validate("panel_titles")
    def _validate_panel_titles(self, proposal: dict) -> list:
        """Require panel_titles length to match n_panels or be empty."""
        # Empty list = default per-panel labels. Otherwise must match n_panels.
        val = list(proposal["value"])
        if val and len(val) != int(self.n_panels):
            raise traitlets.TraitError(
                f"panel_titles length ({len(val)}) must equal n_panels ({self.n_panels}) or be empty"
            )
        return val

    @traitlets.validate("starred")
    def _validate_starred(self, proposal: dict) -> list:
        """Require `starred` length to match n_panels. Each entry is a frame
        index in [-1, n_slices); -1 means no star on that panel."""
        val = list(proposal["value"])
        n_pan = int(self.n_panels)
        if not val:
            return [-1] * n_pan
        if len(val) != n_pan:
            raise traitlets.TraitError(
                f"starred length ({len(val)}) must equal n_panels ({n_pan})"
            )
        n_sl = int(self.n_slices)
        for i, v in enumerate(val):
            if v != -1 and not (0 <= v < n_sl):
                raise traitlets.TraitError(
                    f"starred[{i}] = {v} out of range [-1, {n_sl})"
                )
        return val

    @traitlets.validate("hidden_panels")
    def _validate_hidden_panels(self, proposal: dict) -> list:
        """Normalize hidden panel indices and keep at least one panel visible."""
        n_pan = int(self.n_panels)
        clean: list[int] = []
        seen: set[int] = set()
        for value in proposal["value"]:
            idx = int(value)
            if 0 <= idx < n_pan and idx not in seen:
                clean.append(idx)
                seen.add(idx)
        clean.sort()
        if len(clean) >= n_pan:
            raise traitlets.TraitError(
                "hidden_panels cannot hide every panel; at least one panel must remain visible"
            )
        return clean

    def _normalize_hidden_page_slots(
        self,
        values: Sequence[object],
        *,
        drop_if_full: bool = False,
    ) -> list[int]:
        """Normalize paged-widget hidden slots while keeping one slot visible."""
        if int(self.n_pages) <= 1 or int(self.panels_per_page) <= 0:
            return []
        n_slots = int(self.panels_per_page)
        clean_set: set[int] = set()
        for value in values or []:
            if isinstance(value, bool):
                continue
            try:
                slot = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= slot < n_slots:
                clean_set.add(slot)
        clean = sorted(clean_set)
        if len(clean) >= n_slots:
            if drop_if_full:
                clean = clean[:-1]
            else:
                raise traitlets.TraitError(
                    "hidden_page_slots cannot hide every page slot; at least one panel must remain visible"
                )
        return clean

    def _hidden_page_slots_from_panels(
        self,
        panels: Sequence[int],
        *,
        drop_if_full: bool = False,
    ) -> list[int]:
        """Map absolute hidden panel indices to reusable page slots."""
        if int(self.n_pages) <= 1 or int(self.panels_per_page) <= 0:
            return []
        n_pan = int(self.n_panels)
        per_page = int(self.panels_per_page)
        slots = [
            int(panel) % per_page
            for panel in panels
            if 0 <= int(panel) < n_pan
        ]
        return self._normalize_hidden_page_slots(slots, drop_if_full=drop_if_full)

    @traitlets.validate("hidden_page_slots")
    def _validate_hidden_page_slots(self, proposal: dict) -> list[int]:
        """Normalize hidden page slots for paged widgets."""
        return self._normalize_hidden_page_slots(proposal["value"])

    @traitlets.validate("panel_order")
    def _validate_panel_order(self, proposal: dict) -> list[int]:
        """Normalize optional display order over source panel indices."""
        n_pan = int(self.n_panels)
        values = list(proposal["value"] or [])
        if not values:
            return []
        clean: list[int] = []
        try:
            for value in values:
                if isinstance(value, bool):
                    raise TypeError
                clean.append(int(value))
        except (TypeError, ValueError) as exc:
            raise traitlets.TraitError("panel_order must contain integer panel indices") from exc
        expected = list(range(n_pan))
        if len(clean) != n_pan or sorted(clean) != expected:
            raise traitlets.TraitError(
                "panel_order must include every panel index exactly once "
                f"(expected a permutation of {expected!r})"
            )
        return clean

    @traitlets.validate("page_idx")
    def _validate_page_idx(self, proposal: dict) -> int:
        """Clamp the active page index to the available page range."""
        n_pages = max(1, int(getattr(self, "n_pages", 1)))
        return int(max(0, min(int(proposal["value"]), n_pages - 1)))

    @traitlets.validate("page_starred")
    def _validate_page_starred(self, proposal: dict) -> list[int]:
        """Normalize per-page star flags."""
        val = list(proposal["value"])
        n_pages = max(1, int(getattr(self, "n_pages", 1)))
        if not val:
            return [0] * n_pages
        if len(val) != n_pages:
            raise traitlets.TraitError(
                f"page_starred length ({len(val)}) must equal n_pages ({n_pages})"
            )
        return [1 if int(v) else 0 for v in val]

    def _panel_title_for_index(self, panel: int) -> str:
        """Return the user-facing title for a panel index."""
        if 0 <= panel < len(self.panel_titles) and self.panel_titles[panel]:
            return str(self.panel_titles[panel])
        return f"Panel {panel + 1}"

    def _resolve_panel_ref(self, panel: int | str) -> int:
        """Resolve a panel index or exact title into a zero-based panel index."""
        if isinstance(panel, bool):
            raise TypeError("panel must be an integer index or exact panel title, not bool")
        if isinstance(panel, int):
            idx = int(panel)
            if 0 <= idx < int(self.n_panels):
                return idx
            raise ValueError(f"panel index {idx} out of range [0, {self.n_panels})")
        if isinstance(panel, str):
            titles = [self._panel_title_for_index(i) for i in range(int(self.n_panels))]
            matches = [i for i, title in enumerate(titles) if title == panel]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError(
                    f"panel title {panel!r} is not unique; use a zero-based panel index instead"
                )
            available = ", ".join(repr(title) for title in titles)
            raise ValueError(f"unknown panel title {panel!r}; available titles: {available}")
        raise TypeError(
            f"panel must be an integer index or exact panel title, got {type(panel).__name__}"
        )

    def _normalize_panel_refs(
        self,
        panels: Sequence[int | str] | int | str,
        *,
        allow_empty: bool = False,
    ) -> list[int]:
        """Resolve and de-duplicate panel references while preserving order."""
        if isinstance(panels, (str, int)) and not isinstance(panels, bool):
            values: Sequence[int | str] = [panels]
        else:
            values = panels  # type: ignore[assignment]
        out: list[int] = []
        seen: set[int] = set()
        for panel in values:
            idx = self._resolve_panel_ref(panel)
            if idx not in seen:
                out.append(idx)
                seen.add(idx)
        if not out and not allow_empty:
            raise ValueError("at least one panel index or title is required")
        return out

    def _validate_panel_list_length(self, name: str, value: list) -> list:
        """Require per-panel list traits to have exactly one entry per panel."""
        n_pan = int(self.n_panels)
        if len(value) != n_pan:
            raise traitlets.TraitError(
                f"{name} length ({len(value)}) must equal n_panels ({n_pan})"
            )
        return value

    @traitlets.validate("vmin_per_panel", "vmax_per_panel")
    def _validate_panel_bounds(self, proposal: dict) -> list:
        """Require per-panel vmin/vmax lists to match n_panels and be finite."""
        name = proposal["trait"].name
        val = self._validate_panel_list_length(name, list(proposal["value"]))
        for i, bound in enumerate(val):
            if bound is not None and not math.isfinite(float(bound)):
                raise traitlets.TraitError(f"{name}[{i}] must be finite or None, got {bound}")
        other_name = "vmax_per_panel" if name == "vmin_per_panel" else "vmin_per_panel"
        other = list(getattr(self, other_name))
        if len(other) == len(val):
            for i, bound in enumerate(val):
                other_bound = other[i]
                if bound is None or other_bound is None:
                    continue
                if name == "vmin_per_panel" and float(bound) > float(other_bound):
                    raise traitlets.TraitError(
                        f"vmin_per_panel[{i}] ({bound}) must be <= vmax_per_panel[{i}] ({other_bound})"
                    )
                if name == "vmax_per_panel" and float(bound) < float(other_bound):
                    raise traitlets.TraitError(
                        f"vmax_per_panel[{i}] ({bound}) must be >= vmin_per_panel[{i}] ({other_bound})"
                    )
        return val

    def _reset_panel_contrast_traits(self) -> None:
        """Initialize per-panel contrast traits from the shared contrast defaults."""
        n_pan = int(self.n_panels)
        self.vmin_per_panel = [None] * n_pan
        self.vmax_per_panel = [None] * n_pan

    @traitlets.validate("fps")
    def _validate_fps(self, proposal: dict) -> float:
        """Reject invalid fps and cap playback at the browser budget."""
        val = float(proposal["value"])
        if not math.isfinite(val):
            raise traitlets.TraitError(f"fps must be finite, got {val}")
        if val <= 0:
            raise traitlets.TraitError(f"fps must be > 0, got {val}")
        return min(val, _MAX_PLAYBACK_FPS)

    @traitlets.validate("slice_idx")
    def _validate_slice_idx(self, proposal: dict) -> int:
        """Clamp to [0, n_slices) so stale indices don't IndexError on reload."""
        # Clamp to [0, n_slices). State load with stale index used to crash
        # _update_all → _data[idx] with IndexError on a smaller new stack.
        val = int(proposal["value"])
        n = max(1, int(self.n_slices))
        return max(0, min(val, n - 1))

    @traitlets.validate("profile_width")
    def _validate_profile_width(self, proposal: dict) -> int:
        """Reject profile widths < 1."""
        val = int(proposal["value"])
        if val < 1:
            raise traitlets.TraitError(f"profile_width must be >= 1, got {val}")
        return val

    @traitlets.validate("percentile_low")
    def _validate_percentile_low(self, proposal: dict) -> float:
        """Require percentile_low in [0, 100] and strictly < percentile_high."""
        val = float(proposal["value"])
        if not 0 <= val <= 100:
            raise traitlets.TraitError(f"percentile_low must be in [0, 100], got {val}")
        if val >= float(self.percentile_high):
            raise traitlets.TraitError(
                f"percentile_low ({val}) must be < percentile_high ({self.percentile_high})"
            )
        return val

    @traitlets.validate("percentile_high")
    def _validate_percentile_high(self, proposal: dict) -> float:
        """Require percentile_high in [0, 100] and strictly > percentile_low."""
        val = float(proposal["value"])
        if not 0 <= val <= 100:
            raise traitlets.TraitError(f"percentile_high must be in [0, 100], got {val}")
        if val <= float(self.percentile_low):
            raise traitlets.TraitError(
                f"percentile_high ({val}) must be > percentile_low ({self.percentile_low})"
            )
        return val

    @traitlets.validate("roi_selected_idx")
    def _validate_roi_selected_idx(self, proposal: dict) -> int:
        """Clamp to [0, len(roi_list)) or -1 for nothing selected."""
        # -1 = nothing selected. Otherwise clamp to [0, len(roi_list))
        # so JS doesn't index OOB and Python stats don't silently return {}.
        val = int(proposal["value"])
        if val < 0:
            return -1
        return min(val, max(0, len(self.roi_list) - 1))

    _VALID_ROI_SHAPES = {"circle", "square", "rectangle", "annular"}

    @traitlets.validate("roi_list")
    def _validate_roi_list(self, proposal: dict) -> list:
        """Reject unknown ROI shapes and non-numeric or negative geometry fields."""
        # Reject unknown shapes (used to silently fall back to circle).
        # Clamp negative radii / dims to 1 so stats reflect what user sees.
        val = list(proposal["value"])
        for i, r in enumerate(val):
            shape = r.get("shape", "circle")
            if shape not in self._VALID_ROI_SHAPES:
                raise traitlets.TraitError(
                    f"ROI {i}: unknown shape {shape!r}. "
                    f"Valid: {sorted(self._VALID_ROI_SHAPES)}"
                )
            for k in ("radius", "radius_inner", "width", "height"):
                if k in r and r[k] is not None:
                    try:
                        v = float(r[k])
                    except (TypeError, ValueError):
                        raise traitlets.TraitError(
                            f"ROI {i}: {k}={r[k]!r} is not a number"
                        )
                    if v < 0:
                        raise traitlets.TraitError(
                            f"ROI {i}: {k} must be >= 0, got {r[k]}"
                        )
        return val

    # === Construction ===

    @classmethod
    def from_gif(
        cls,
        path: str | pathlib.Path,
        *,
        fps: float | None = None,
        title: str | None = None,
        frame_labels: list[str] | bool | None = None,
        **kwargs,
    ) -> Self:
        """Open a static or animated GIF as a playable stack.

        The GIF is decoded through ``quantem.widget.io.read_gif`` into a
        grayscale ``Dataset3d``. Pass normal ``Show3D`` keyword arguments such as
        ``cmap``, ``show_controls``, ``max_cols``, or ``panel_width_px``.
        """
        from quantem.widget.io import read_gif  # noqa: PLC0415

        ds = read_gif(path)
        frame_count = int(ds.array.shape[0])
        kwargs.setdefault("title", ds.name if title is None else title)
        if fps is not None:
            kwargs.setdefault("fps", fps)
        if frame_labels is True:
            kwargs.setdefault("labels", [f"Frame {i + 1}" for i in range(frame_count)])
        elif frame_labels:
            kwargs.setdefault("labels", list(frame_labels))
        return cls(ds, **kwargs)

    @classmethod
    def from_folder(
        cls,
        path: str | pathlib.Path,
        *,
        pattern: str = "*",
        recursive: bool = False,
        watch: bool = True,
        watch_interval: float = 1.0,
        **kwargs,
    ) -> Self:
        """Play naturally ordered, full-resolution folder images as one stack.

        Stable additions update this widget in place. Every file is decoded
        through :func:`quantem.widget.io.read_image`; failed or partially
        written files remain pending for a later poll.
        """
        if "labels" in kwargs:
            raise TypeError(
                "Show3D.from_folder() derives frame labels from file paths so "
                "watched additions remain identifiable; remove labels= or "
                "construct Show3D directly."
            )
        source = WatchedImageFolder(
            path,
            pattern=pattern,
            recursive=recursive,
            interval=watch_interval,
            mode="frames",
        )
        arrays, records = source.read_initial(
            allow_empty=watch,
            require_unchanged_followup=watch,
        )
        explicit_calibration = any(
            key in kwargs
            for key in ("sampling", "units", "pixel_size", "pixel_unit")
        )

        config_data = _load_quantem_config(kwargs.get("config"))
        apply_config_transforms = bool(kwargs.get("apply_config_transforms", True))
        rotation_deg = kwargs.get("rotation_deg")
        if rotation_deg is None and config_data is not None and apply_config_transforms:
            rotation_deg = _config_float(config_data, "data", "rotation_deg") or 0.0
        config_rotation = (
            _normalize_rotation_deg(rotation_deg) if rotation_deg is not None else 0.0
        )
        post_crop = kwargs.get("post_crop")
        post_crop_was_set = post_crop is not None
        crop = _normalize_crop(kwargs.get("crop", 0))
        padding = _normalize_padding(kwargs.get("padding", 0))
        pad_mode = str(kwargs.get("pad_mode", "median"))

        def folder_transform(frame: np.ndarray) -> np.ndarray:
            stack = np.asarray(frame)[None, ...]
            if config_rotation:
                stack = _rotate_stack_inplane(stack, config_rotation)
            if post_crop_was_set:
                crop_spec = post_crop
            elif config_data is not None and apply_config_transforms:
                crop_spec = _post_crop_from_quantem_config(stack.shape, config_data)
            else:
                crop_spec = 0
            stack = _crop_stack(stack, _normalize_crop(crop_spec))
            stack = _pad_stack(_crop_stack(stack, crop), padding, pad_mode)
            return np.asarray(stack[0])

        kwargs.setdefault("title", source.folder.name)
        kwargs.setdefault("offline", False)
        kwargs.setdefault("verbose", False)
        constructor_kwargs = dict(kwargs)
        if arrays:
            initial_stack = np.stack(arrays, axis=0)
            initial_labels = [source.label(record.path) for record in records]
        else:
            display_bin = constructor_kwargs.get("display_bin", 1)
            placeholder_side = (
                max(1, int(display_bin))
                if isinstance(display_bin, int) and not isinstance(display_bin, bool)
                else 1
            )
            initial_stack = np.zeros(
                (1, placeholder_side, placeholder_side),
                dtype=np.float32,
            )
            initial_labels = [""]
            # Real folder frames are transformed by ``folder_transform`` when
            # they arrive. Keep the bootstrap array out of that data path so a
            # crop intended for a large microscope frame cannot empty it.
            constructor_kwargs.update(
                crop=0,
                padding=0,
                rotation_deg=0,
                post_crop=0,
                apply_config_transforms=False,
            )
        widget = cls(
            initial_stack,
            labels=initial_labels,
            **constructor_kwargs,
        )
        widget._folder_input_transform = folder_transform
        if not arrays:
            widget._crop = crop
            widget._padding = padding
            widget._pad_mode = pad_mode
            widget._set_folder_waiting_empty_state()
        source.attach(widget, explicit_calibration=explicit_calibration)
        if watch:
            widget.watch_folder(interval=watch_interval)
        return widget

    def __init__(
        self,
        *data_args,
        labels: list[str] | None = None,
        panel_titles: list[str] | None = None,
        panel_frame_labels: list[list[str]] | None = None,
        frame_metadata: Sequence[Mapping[str, Any]] | None = None,
        panel_frame_metadata: Sequence[Sequence[Mapping[str, Any]]] | None = None,
        frame_label_format: str | Callable[..., object] | None = None,
        page_labels: Sequence[str | None] | None = None,
        panel_real_frames: list[int] | None = None,
        hidden_panels: Sequence[int | str] | int | str | None = None,
        panel_order: Sequence[int | str] | None = None,
        title: str = "",
        ui_mode: UiMode = "interactive",
        show_title: bool | None = None,
        cmap: str | Colormap = Colormap.PLASMA,
        vmin: float | None = None,
        vmax: float | None = None,
        sampling: float | tuple[float, float] | list[float] | None = None,
        units: str | list[str] | None = None,
        smooth: bool = True,
        image_rotation: int = 0,
        log_scale: bool = False,
        auto_contrast: bool = True,
        link_contrast: bool | None = None,
        image_vmin_pct: float = 0.0,
        image_vmax_pct: float = 100.0,
        percentile_low: float = 0.5,
        percentile_high: float = 99.5,
        fps: float = 30.0,
        avg_window: int = 1,
        timestamps: list[float] | None = None,
        timestamp_unit: str = "s",
        show_fft: bool = False,
        fft_layout: str = "bottom",
        fft_overlay_position: str = "top-left",
        fft_overlay_size: float = 0.35,
        fft_overlay_zoom: float = 1.0,
        fft_window: bool = True,
        fft_metrics: bool = True,
        show_stats: bool | None = None,
        show_controls: bool | None = None,
        controls_collapsed: bool | None = None,
        debug: bool = False,
        size: int = 0,
        panel_width_px: int = 0,
        crop: int | tuple[int, int] | tuple[int, int, int, int] = 0,
        padding: int | tuple[int, int] = 0,
        pad_mode: str = "median",
        config: "Mapping | str | pathlib.Path | None" = None,
        rotation_deg: float | None = None,
        post_crop: int | tuple[int, int] | tuple[int, int, int, int] | None = None,
        apply_config_transforms: bool = True,
        diff_mode: str = "off",
        buffer_size: int = 64,
        dim_label: str = "Frame",
        use_torch: bool | None = None,
        device: str | None = None,
        display_bin: int | str = "auto",
        hideable: bool = False,
        offline: bool | None = None,
        state=None,
        save_state: bool = False,
        notebook_preview_format: str | None = "jpeg",
        notebook_preview_quality: int = 88,
        notebook_preview_max_px: int = 512,
        denoise: str = "none",
        denoise_sigma: float = 4.0,
        denoise_bin: int = 1,
        display_filter: str | None = None,
        display_sigma: float | None = None,
        spatial_bin: int | None = None,
        show_denoise: bool = False,
        verbose: bool = True,
        max_cols: int | None = None,
        panel_gap: int | None = None,
        panel_title_font_size: int | None = None,
        show_panel_titles: bool | None = None,
        show_resize_handles: bool | None = None,
        show_zoom_indicator: bool | None = None,
        show_scale_bar: bool | None = None,
        dedupe_identical_panels: bool = False,
        **kwargs,
    ):
        scale_bar_visible = kwargs.pop("scale_bar_visible", None)
        if (
            scale_bar_visible is not None
            and show_scale_bar is not None
            and bool(scale_bar_visible) != bool(show_scale_bar)
        ):
            raise ValueError("Use either show_scale_bar or scale_bar_visible, not conflicting values")
        ui = resolve_ui_mode(
            ui_mode,
            defaults={
                "show_title": True,
                "show_controls": True,
                "controls_collapsed": False,
                "show_stats": False,
                "show_panel_titles": True,
                "show_resize_handles": True,
                "show_zoom_indicator": True,
                "show_scale_bar": True,
            },
            overrides={
                "show_title": show_title,
                "show_controls": show_controls,
                "controls_collapsed": controls_collapsed,
                "show_stats": show_stats,
                "show_panel_titles": show_panel_titles,
                "show_resize_handles": show_resize_handles,
                "show_zoom_indicator": show_zoom_indicator,
                "show_scale_bar": scale_bar_visible if scale_bar_visible is not None else show_scale_bar,
            },
        )
        show_title = bool(ui["show_title"])
        show_controls = bool(ui["show_controls"])
        controls_collapsed = bool(ui["controls_collapsed"])
        show_stats = bool(ui["show_stats"])
        show_panel_titles = bool(ui["show_panel_titles"])
        show_resize_handles = bool(ui["show_resize_handles"])
        show_zoom_indicator = bool(ui["show_zoom_indicator"])
        show_scale_bar = bool(ui["show_scale_bar"])
        if hideable:
            kwargs["hideable"] = True
        kwargs["show_title"] = show_title
        if max_cols is not None:
            kwargs["max_cols"] = int(max_cols)
        if panel_gap is not None:
            kwargs["panel_gap"] = int(panel_gap)
        if panel_title_font_size is not None:
            kwargs["panel_title_font_size"] = int(panel_title_font_size)
        kwargs["show_panel_titles"] = show_panel_titles
        kwargs["show_resize_handles"] = show_resize_handles
        kwargs["show_zoom_indicator"] = show_zoom_indicator
        fft_layout = str(fft_layout).lower()
        if fft_layout not in {"bottom", "right", "overlay"}:
            raise ValueError(
                "fft_layout must be one of 'bottom', 'right', or 'overlay', "
                f"got {fft_layout!r}"
            )
        fft_overlay_position = str(fft_overlay_position).lower()
        if fft_overlay_position not in {"top-left", "top-right", "bottom-left", "bottom-right"}:
            raise ValueError(
                "fft_overlay_position must be one of 'top-left', 'top-right', "
                f"'bottom-left', or 'bottom-right', got {fft_overlay_position!r}"
            )
        fft_overlay_size = float(fft_overlay_size)
        if not np.isfinite(fft_overlay_size):
            raise ValueError(f"fft_overlay_size must be finite, got {fft_overlay_size}")
        if not 0.2 <= fft_overlay_size <= 0.7:
            raise ValueError(f"fft_overlay_size must be in [0.2, 0.7], got {fft_overlay_size}")
        fft_overlay_zoom = float(fft_overlay_zoom)
        if not np.isfinite(fft_overlay_zoom):
            raise ValueError(f"fft_overlay_zoom must be finite, got {fft_overlay_zoom}")
        if not 1.0 <= fft_overlay_zoom <= 32.0:
            raise ValueError(f"fft_overlay_zoom must be in [1.0, 32.0], got {fft_overlay_zoom}")
        kwargs["scale_bar_visible"] = show_scale_bar
        self._verbose = bool(verbose)
        panel_width_px = int(panel_width_px)
        if panel_width_px < 0:
            raise ValueError(f"panel_width_px must be >= 0, got {panel_width_px}")
        if panel_width_px > 0:
            size = panel_width_px
        kwargs.pop("show_playback", None)
        legacy_link_zoom = kwargs.pop("link_zoom", None)
        legacy_link_pan = kwargs.pop("link_pan", None)
        if "link_panels" not in kwargs:
            if legacy_link_zoom is not None and legacy_link_pan is not None:
                kwargs["link_panels"] = bool(legacy_link_zoom) and bool(legacy_link_pan)
            elif legacy_link_zoom is not None:
                kwargs["link_panels"] = bool(legacy_link_zoom)
            elif legacy_link_pan is not None:
                kwargs["link_panels"] = bool(legacy_link_pan)
        # `sampling` + `units` is the canonical quantem convention (matches
        # Dataset.sampling/.units, Show2D, Show4DSTEM). They feed the internal
        # `pixel_size`/`pixel_unit` traits the scale bar reads.
        pixel_size = 0.0
        pixel_unit = None
        if sampling is not None:
            samp = sampling[0] if isinstance(sampling, (tuple, list)) else sampling
            pixel_size = float(samp)
        if units is not None:
            pixel_unit = units[0] if isinstance(units, (tuple, list)) else units
        data_args, panel_titles, n_pages, panels_per_page, resolved_page_labels, resolved_page_starred = _normalise_show3d_pages(
            data_args,
            panel_titles=panel_titles,
            page_labels=page_labels,
        )
        if link_contrast is None:
            link_contrast = n_pages <= 1
        kwargs["link_contrast"] = bool(link_contrast)
        _t0 = time.perf_counter()
        # Reject unknown kwargs so typos raise instead of being silently ignored.
        _reject_unknown_kwargs(type(self), kwargs)
        # Deprecated display_filter-era aliases (one rc of compatibility).
        if display_filter is not None and denoise == "none":
            denoise = display_filter
        if display_sigma is not None and denoise_sigma == 4.0:
            denoise_sigma = display_sigma
        if spatial_bin is not None and denoise_bin == 1:
            denoise_bin = spatial_bin
        # save_state controls whether the heavy pixel buffers are persisted into
        # the notebook's metadata.widgets on save. Default False: a plain display
        # embeds only light traits + a static image preview, so a large z-stack
        # does not bake hundreds of MB into the .ipynb. Set True to persist
        # full interactive state so a reopened notebook restores the widget
        # without a kernel.
        self._save_state = bool(save_state)
        self.prepared_view = None
        self.prepared_view_widget = None
        self._configure_static_fallback(
            notebook_preview_format=notebook_preview_format,
            notebook_preview_quality=notebook_preview_quality,
            notebook_preview_max_px=notebook_preview_max_px,
        )
        super().__init__(**kwargs)
        self._static_fallback_mime = self._static_fallback_mime_type()
        # hold_sync() batches ALL traitlet assignments into a single comm message
        # sent when the context manager exits.  Without this, each self.x = y
        # fires a separate round-trip over the ZMQ/websocket channel, which
        # can add 30+ seconds for a large stack in VS Code Jupyter.
        if panel_real_frames is not None:
            self.panel_real_frames = list(panel_real_frames)
        with self.hold_sync():
            self._init_sync(data_args, labels=labels, panel_titles=panel_titles,
                            panel_frame_labels=panel_frame_labels,
                            frame_metadata=frame_metadata,
                            panel_frame_metadata=panel_frame_metadata,
                            frame_label_format=frame_label_format,
                            title=title, cmap=cmap, vmin=vmin, vmax=vmax,
                            n_pages=n_pages, panels_per_page=panels_per_page,
                            page_labels=resolved_page_labels, page_starred=resolved_page_starred,
                            pixel_size=pixel_size, pixel_unit=pixel_unit,
                            smooth=smooth, image_rotation=image_rotation,
                            log_scale=log_scale,
                            auto_contrast=auto_contrast,
                            image_vmin_pct=image_vmin_pct,
                            image_vmax_pct=image_vmax_pct,
                            percentile_low=percentile_low,
                            percentile_high=percentile_high, fps=fps, avg_window=avg_window,
                            timestamps=timestamps,
                            timestamp_unit=timestamp_unit, show_fft=show_fft,
                            fft_layout=fft_layout,
                            fft_overlay_position=fft_overlay_position,
                            fft_overlay_size=fft_overlay_size,
                            fft_overlay_zoom=fft_overlay_zoom,
                            fft_window=fft_window,
                            fft_metrics=fft_metrics,
                            show_stats=show_stats, show_controls=show_controls,
                            controls_collapsed=controls_collapsed, debug=debug,
                            size=size, crop=crop, padding=padding, pad_mode=pad_mode,
                            config=config, rotation_deg=rotation_deg,
                            post_crop=post_crop,
                            apply_config_transforms=apply_config_transforms,
                            diff_mode=diff_mode, buffer_size=buffer_size,
                            dim_label=dim_label, use_torch=use_torch, device=device,
                            display_bin=display_bin, offline=offline,
                            state=state, dedupe_identical_panels=dedupe_identical_panels,
                            denoise=denoise, denoise_sigma=denoise_sigma,
                            denoise_bin=denoise_bin, show_denoise=show_denoise,
                            _t0=_t0)
            if panel_order is not None:
                self.set_panel_order(panel_order)
            if hidden_panels is not None:
                self.set_hidden_panels(hidden_panels)

    def _init_sync(self, data_args: tuple, *, labels: list[str] | None,
                   panel_titles: list[str] | None,
                   panel_frame_labels: list[list[str]] | None,
                   frame_metadata: Sequence[Mapping[str, Any]] | None,
                   panel_frame_metadata: Sequence[Sequence[Mapping[str, Any]]] | None,
                   frame_label_format: str | Callable[..., object] | None,
                   title: str,
                   n_pages: int, panels_per_page: int,
                   page_labels: list[str], page_starred: list[int],
                   cmap: str | Colormap, vmin: float | None, vmax: float | None,
                   pixel_size: float, pixel_unit: str | None, smooth: bool, image_rotation: int,
                   log_scale: bool, auto_contrast: bool,
                   image_vmin_pct: float, image_vmax_pct: float,
                   percentile_low: float, percentile_high: float,
                   fps: float, avg_window: int,
                   timestamps: list[float] | None,
                   timestamp_unit: str, show_fft: bool, fft_layout: str,
                   fft_overlay_position: str, fft_overlay_size: float,
                   fft_overlay_zoom: float, fft_window: bool, fft_metrics: bool,
                   show_stats: bool | None, show_controls: bool,
                   controls_collapsed: bool, debug: bool,
                   size: int, crop: int | tuple[int, int] | tuple[int, int, int, int],
                   padding: int | tuple[int, int], pad_mode: str,
                   config, rotation_deg, post_crop, apply_config_transforms: bool,
                   diff_mode: str, buffer_size: int, dim_label: str,
                   use_torch: bool | None, device: str | None,
                   display_bin: int | str, offline: bool | None,
                   state: dict | str | pathlib.Path | None,
                   dedupe_identical_panels: bool, _t0: float,
                   denoise: str = "none", denoise_sigma: float = 4.0,
                   denoise_bin: int = 1, show_denoise: bool = False) -> None:
        """Heavy setup called synchronously by `__init__` inside `hold_sync()`.
        Validates panels, allocates frame_bytes, wires observers, and applies
        optional `state`. Split out from `__init__` so the construction surface
        reads as a clean kwargs list while the heavy work has its own scope."""
        self.widget_version = resolve_widget_version()
        self._frame_server = None
        self._frame_server_thread = None
        self._frame_server_token = secrets.token_urlsafe(18)
        self._separate_panel_data = None
        self._crop = _normalize_crop(crop)
        self._padding = _normalize_padding(padding)
        self.n_pages = int(max(1, n_pages))
        self.panels_per_page = int(max(0, panels_per_page))
        self.page_idx = 0
        self.page_labels = list(page_labels or [])
        self.page_starred = list(page_starred or [])
        if pad_mode not in ("median", "constant"):
            raise ValueError(f"pad_mode must be 'median' or 'constant', got {pad_mode!r}")
        self._pad_mode = pad_mode

        # ── QuantEM config convenience (mirrors Show3DSlices) ──
        # config supplies in-plane rotation + post-crop alignment and pixel size
        # so a ptycho z-stack calibrates with no manual array math. Explicit
        # rotation_deg / post_crop / pixel_size always win over config values.
        config_data = _load_quantem_config(config)
        post_crop_was_set = post_crop is not None
        if rotation_deg is None and config_data is not None and apply_config_transforms:
            rotation_deg = _config_float(config_data, "data", "rotation_deg") or 0.0
        config_rotation = _normalize_rotation_deg(rotation_deg) if rotation_deg is not None else 0.0

        def _apply_config_transform(arr):
            """Rotate then post-crop a (N, H, W) stack per config / explicit args."""
            if config_rotation:
                arr = _rotate_stack_inplane(arr, config_rotation)
            if post_crop_was_set:
                crop_spec = post_crop
            elif config_data is not None and apply_config_transforms:
                crop_spec = _post_crop_from_quantem_config(arr.shape, config_data)
            else:
                crop_spec = 0
            return _crop_stack(arr, _normalize_crop(crop_spec))

        # Optional torch acceleration. Do not move NumPy/Dataset input to GPU
        # merely because CUDA/MPS exists: real multi-panel ptycho stacks can be
        # many GB, and an implicit copy can OOM before the widget renders.
        self._use_torch = False
        self._device = None
        self._data_torch = None
        self._display_torch = None
        first_tensor_device = None
        if use_torch is None:
            use_torch = False
            if _HAS_TORCH:
                for d in data_args:
                    if isinstance(d, torch.Tensor):
                        use_torch = True
                        first_tensor_device = d.device
                        break
        if use_torch:
            if not _HAS_TORCH:
                raise ImportError(
                    "use_torch=True requires PyTorch. Install it with: pip install torch"
                )
            self._use_torch = True
            if device is not None:
                self._device = torch.device(device)
            elif first_tensor_device is not None:
                self._device = first_tensor_device
            else:
                self._device = torch.device(
                    "mps" if torch.backends.mps.is_available()
                    else "cuda" if torch.cuda.is_available()
                    else "cpu"
                )

        # ── Parse data args: single or multi-panel ──
        # Show3D(data) → single panel
        # Show3D(data1, data2, ...) → multi-panel (side-by-side, synced)
        if len(data_args) == 0:
            raise TypeError("Show3D requires at least one data argument")

        # Flatten: Show3D([data1, data2]) also works for multi-panel
        if (len(data_args) == 1 and isinstance(data_args[0], (list, tuple))
                and len(data_args[0]) > 0 and not isinstance(data_args[0][0], (int, float))):
            # Check if it's a list of arrays vs a single array
            first = data_args[0][0]
            if hasattr(first, 'ndim') or isinstance(first, np.ndarray):
                data_args = tuple(data_args[0])

        data = data_args[0]

        # Check if data is a Dataset3d and extract metadata
        _extracted_title = None
        _extracted_pixel_size = None
        _extracted_pixel_unit = None
        _extracted_dim_sampling = None
        _extracted_dim_unit = None
        if hasattr(data, "array") and hasattr(data, "name") and hasattr(data, "sampling"):
            _extracted_title = data.name if data.name else None
            if hasattr(data, "sampling") and len(data.sampling) >= 3:
                sampling_val = float(data.sampling[1])
                _extracted_dim_sampling = float(data.sampling[0])
                if hasattr(data, "units"):
                    units = list(data.units)
                    if len(units) >= 1:
                        _extracted_dim_unit = str(units[0])
                    if len(units) >= 2:
                        _extracted_pixel_unit = str(units[1])
                _extracted_pixel_size = sampling_val
            data = data.array

        # Convert first panel to NumPy
        data = to_numpy(data)
        if data.ndim == 2:
            data = data[None, ...]
        # True-color stack: (N, H, W, 3) or a single RGB frame (H, W, 3) with H>4.
        is_rgb = False
        if data.ndim == 3 and data.shape[-1] in (3, 4) and int(data.shape[0]) > 4:
            # Single RGB image → one-frame color stack.
            data = data[None, ..., :3]
            is_rgb = True
        elif data.ndim == 4 and data.shape[-1] in (3, 4):
            data = data[..., :3]
            is_rgb = True
        elif data.ndim != 3:
            raise ValueError(
                f"Expected (N, H, W) gray stack or (N, H, W, 3) RGB stack, got {data.ndim}D "
                f"shape {data.shape}"
            )
        if 0 in data.shape:
            raise ValueError(f"Empty stack: shape {data.shape}. All dims must be >= 1.")
        if np.iscomplexobj(data):
            raise TypeError(
                "Show3D does not accept complex data. Convert first: "
                "np.abs(arr) for magnitude or np.angle(arr) for phase."
            )
        if not _all_finite(data):
            raise ValueError(
                "Data contains NaN or inf. Clean first: "
                "np.nan_to_num(arr, nan=0, posinf=0, neginf=0)."
            )
        if is_rgb:
            # Normalize display-ready color once (uint8 → [0, 1]).
            data = data.astype(np.float32, copy=False)
            if float(np.nanmax(data)) > 1.5:
                data = data / 255.0
            data = np.clip(data, 0.0, 1.0)
        data = _pad_stack(_crop_stack(data, self._crop), self._padding, self._pad_mode)
        if is_rgb and config_rotation:
            raise ValueError(
                "Show3D RGB stacks do not support config in-plane rotation yet; "
                "pass rotation_deg=0 / apply_config_transforms=False for color PNGs."
            )
        if not is_rgb:
            data = _apply_config_transform(data)
        self.is_rgb = bool(is_rgb)
        data = self._establish_rgb_luminance(data, is_rgb=is_rgb)

        # Multi-panel: convert remaining args, validate shapes, concatenate.
        # If the caller repeats the same stack object across panels, keep the
        # data once and let JS draw that exact frame into multiple panel slots.
        # This is the "36 full-res frames across 9 panels" stress case: no
        # pre-binning, no 9x Python copy, no 604 MB synthetic browser frame.
        if len(data_args) > 1:
            if is_rgb:
                raise NotImplementedError(
                    "Show3D multi-panel mode does not yet support RGB color stacks. "
                    "Pass one RGB stack, or use Show2D for a color gallery."
                )
            def _raw_array_obj(obj):
                return obj.array if hasattr(obj, "array") else obj

            first_source_obj = _raw_array_obj(data_args[0])
            shared_source = all(_raw_array_obj(extra) is first_source_obj for extra in data_args[1:])

            def _sample_is_finite(arr: np.ndarray) -> bool:
                return _all_finite(arr)

            def _as_valid_panel(arr: np.ndarray, panel_name: str) -> np.ndarray:
                if not _sample_is_finite(arr):
                    raise ValueError(
                        f"{panel_name} contains NaN or inf. Clean first: "
                        "np.nan_to_num(arr, nan=0, posinf=0, neginf=0)."
                    )
                with np.errstate(over="ignore", invalid="ignore"):
                    arr32 = arr.astype(np.float32, copy=False)
                if not _sample_is_finite(arr32):
                    raise ValueError(
                        f"{panel_name} exceeds float32 range (|value| > 3.4e38) "
                        "after cast; rescale first."
                    )
                return arr32

            self.n_panels = len(data_args)
            if panel_titles is not None:
                self.panel_titles = list(panel_titles)
            else:
                self.panel_titles = [f"Panel {i+1}" for i in range(self.n_panels)]
            self.starred = [-1] * self.n_panels
            self.panel_order = []

            if shared_source:
                _as_valid_panel(data, "Panel 0")
                self.shared_panel_source = True
                self.panel_width_px = int(data.shape[2])
                self._panel_width = self.panel_width_px
                self._multi_panel_bin = 0
                if not self.panel_real_frames:
                    self.panel_real_frames = [int(data.shape[0])] * self.n_panels
            else:
                self.shared_panel_source = False
                # copy=False avoids a redundant 120 MB+ allocation per panel when
                # the user already passed float32 (the common case for ptycho recons).
                panels = [_as_valid_panel(data, "Panel 0")]
                for i, extra in enumerate(data_args[1:], 1):
                    if hasattr(extra, "array"):
                        extra = extra.array
                    arr = to_numpy(extra)
                    if arr.ndim == 2:
                        arr = arr[None, ...]
                    if arr.ndim != 3:
                        raise ValueError(f"Panel {i}: expected 3D array, got {arr.ndim}D")
                    if 0 in arr.shape:
                        raise ValueError(f"Panel {i}: empty stack shape {arr.shape}. All dims must be >= 1.")
                    if np.iscomplexobj(arr):
                        raise TypeError(
                            f"Panel {i}: complex data not accepted. Convert first: "
                            "np.abs(arr) for magnitude or np.angle(arr) for phase."
                        )
                    arr = _pad_stack(_crop_stack(arr, self._crop), self._padding, self._pad_mode)
                    arr = _apply_config_transform(arr)
                    # Image (H,W) must match across panels - viewer cannot composite
                    # different image sizes into one canvas.
                    if arr.shape[1:] != panels[0].shape[1:]:
                        raise ValueError(
                            f"Panel {i} image shape {arr.shape[1:]} must match panel 0 image shape {panels[0].shape[1:]}."
                        )
                    # Slice counts can differ - caller compares trials with different
                    # iteration counts. We auto-pad shorter stacks below.
                    panels.append(_as_valid_panel(arr, f"Panel {i}"))
                self.n_panels = len(panels)
                # Auto-pad short stacks to longest, auto-fill panel_real_frames so
                # JS marks end-of-stack frames. Pad by repeating each panel's last
                # frame - visually obvious vs zeros and keeps colormap range stable.
                real_n = [p.shape[0] for p in panels]
                max_n = max(real_n)
                if any(n != max_n for n in real_n):
                    padded = []
                    for p, n in zip(panels, real_n):
                        if n == max_n:
                            padded.append(p)
                        else:
                            last = p[-1:]
                            pad = np.broadcast_to(last, (max_n - n, *p.shape[1:]))
                            padded.append(np.concatenate([p, pad], axis=0))
                    panels = padded
                    if not self.panel_real_frames:
                        self.panel_real_frames = real_n
                orig_h = panels[0].shape[1]
                orig_w = panels[0].shape[2]
                identical_panels = False
                if dedupe_identical_panels and len(panels) > 1:
                    identical_panels = all(
                        p.shape == panels[0].shape and np.array_equal(p, panels[0])
                        for p in panels[1:]
                    )

                if identical_panels:
                    data = panels[0]
                    self.shared_panel_source = True
                    self.panel_width_px = int(orig_w)
                    self._panel_width = self.panel_width_px
                    self._multi_panel_bin = 0
                    if not self.panel_real_frames:
                        self.panel_real_frames = real_n
                    print(
                        "  Exact duplicate panels deduped for display: "
                        f"{self.n_panels} panels share one {orig_h}x{orig_w} source frame"
                    )
                else:
                    # For large multi-panel stress tests, apply an explicit display_bin
                    # before concatenating panels. Binning after concat would first
                    # materialize a 4096 x (4096 * panels) slab; nine 4k panels is a
                    # 604 MB frame. The full source panels remain referenced here.
                    panel_bin = display_bin if isinstance(display_bin, int) and display_bin > 1 else 1
                    if panel_bin > 1:
                        from quantem.widget.utils.array import bin2d

                        self._source_panels = panels
                        panels = [
                            np.asarray(bin2d(p, factor=panel_bin, mode="mean"), dtype=np.float32)
                            for p in panels
                        ]
                        if pixel_size > 0:
                            pixel_size = pixel_size * panel_bin
                        display_bin = 1
                        print(
                            f"  Multi-panel display bin {panel_bin}x before concat: "
                            f"{orig_h}x{orig_w} -> {panels[0].shape[1]}x{panels[0].shape[2]} per panel"
                        )
                    if panel_bin == 1:
                        # Keep non-identical 4k panels separate. Concatenating
                        # nine panels makes each playback frame 4096 x 36864
                        # (~604 MB) and exceeds practical WebGPU storage-buffer
                        # binding limits. The frame server exposes exact
                        # float32 panel frames, and the frontend renders one
                        # GPU buffer per panel without binning or quantization.
                        self.separate_panel_frames = True
                        self._separate_panel_data = panels
                        data = panels[0]
                    else:
                        # Explicitly binned panels are small enough for the
                        # legacy concatenated-frame path.
                        data = np.concatenate(panels, axis=2)
                    self._panel_width = panels[0].shape[2]
                    self.panel_width_px = self._panel_width
                    self._multi_panel_bin = panel_bin
        else:
            self.n_panels = 1
            self.shared_panel_source = False
            self._multi_panel_bin = 0
            if panel_titles is not None:
                self.panel_titles = list(panel_titles)
            self.starred = [-1]
            self.panel_order = []

        # Reject complex input - silently dropping the imaginary part on
        # ptychography probes was a real data-loss footgun. User should
        # pass np.abs(probe) for magnitude or np.angle(probe) for phase.
        if np.iscomplexobj(data):
            raise TypeError(
                "Show3D does not accept complex data. Convert first: "
                "np.abs(arr) for magnitude or np.angle(arr) for phase."
            )
        # Store data as float32 numpy array. The pre-cast NaN/inf check runs on
        # the original dtype; values that fit in float64 but exceed float32 range
        # (~3.4e38) silently overflow to inf and contaminate stats / display.
        # Sample-check the cast output and reject early if so.
        with np.errstate(over="ignore", invalid="ignore"):
            self._data = data.astype(np.float32, copy=False)
        if not _all_finite(self._data):
            raise ValueError(
                "Data exceeds float32 range (|value| > 3.4e38) after cast; "
                "values silently overflowed to inf. Rescale first: "
                "arr = arr / np.max(np.abs(arr)) or use np.log1p(np.abs(arr))."
            )

        # Create GPU copy if torch acceleration enabled
        if self._use_torch:
            self._data_torch = torch.from_numpy(self._data).to(self._device)

        # Dimensions
        self.n_slices = int(self._data.shape[0])
        orig_h = int(self._data.shape[1])
        orig_w = int(self._data.shape[2])

        # NEVER BIN (CLAUDE.md rule). Display data is always source-pixel-exact.
        # Honor explicit display_bin=N>1 only if caller asks; "auto" stays 1.
        self._display_bin = 1
        if isinstance(display_bin, int) and display_bin > 1:
            self._display_bin = display_bin

        if self.separate_panel_frames:
            self._display_data = self._data
            self.height = orig_h
            self.width = int(self.panel_width_px) * int(self.n_panels)
        elif self._display_bin > 1:
            from quantem.widget.utils.array import bin2d
            self._display_data = bin2d(self._data, factor=self._display_bin, mode="mean")
            self.height = int(self._display_data.shape[1])
            self.width = int(self._display_data.shape[2])
            if pixel_size > 0:
                pixel_size = pixel_size * self._display_bin
            print(f"  Display bin {self._display_bin}× (explicit): {orig_h}×{orig_w} → {self.height}×{self.width}")
        else:
            self._display_data = self._data
            self.height = orig_h
            self.width = orig_w
        if self.shared_panel_source:
            self.panel_width_px = self.width
        if int(self.n_pages) > 1:
            if int(self.panels_per_page) <= 0:
                raise ValueError("panels_per_page must be > 0 when n_pages > 1")
            if int(self.n_panels) != int(self.n_pages) * int(self.panels_per_page):
                raise ValueError(
                    f"paged Show3D expects n_panels == n_pages * panels_per_page, "
                    f"got {self.n_panels} != {self.n_pages} * {self.panels_per_page}"
                )
            if not self.page_labels:
                self.page_labels = [f"Page {i + 1}" for i in range(int(self.n_pages))]
            if len(self.page_labels) != int(self.n_pages):
                raise ValueError(
                    f"page_labels length ({len(self.page_labels)}) must equal n_pages ({self.n_pages})"
                )
            if len(self.page_starred) != int(self.n_pages):
                self.page_starred = [0] * int(self.n_pages)
        else:
            self.n_pages = 1
            self.page_idx = 0
            self.panels_per_page = 0
            self.page_labels = []
            self.page_starred = []
            self.hidden_page_slots = []

        # Color range (global across all frames)
        self._vmin_user = vmin
        self._vmax_user = vmax
        # Compute global min/max ONCE - each scan is bandwidth-bound (~150 ms
        # per pass on 1.34 GB float32), so eliminating the duplicate scans
        # saves ~300 ms on a 20x4k stack.
        if self.separate_panel_frames and self._separate_panel_data is not None:
            stack_min = min(float(p.min()) for p in self._separate_panel_data)
            stack_max = max(float(p.max()) for p in self._separate_panel_data)
        elif self._use_torch:
            stack_min = float(self._data_torch.min().item())
            stack_max = float(self._data_torch.max().item())
        else:
            stack_min = float(self._data.min())
            stack_max = float(self._data.max())
        self._vmin = vmin if vmin is not None else stack_min
        self._vmax = vmax if vmax is not None else stack_max
        self.data_min = stack_min
        self.data_max = stack_max
        # Cache the diff_mode='off' range so toggling Off→Previous→Off restores exact value.
        self._data_min_off = self.data_min
        self._data_max_off = self.data_max

        # Labels and frame metadata. Explicit string labels win; metadata is a
        # generic input layer that any upstream repo can adapt into.
        metadata_formatter = frame_label_format
        if isinstance(metadata_formatter, str):
            self.frame_label_format = metadata_formatter
        else:
            self.frame_label_format = ""
        self.frame_metadata = _normalize_frame_metadata(frame_metadata)
        self.panel_frame_metadata = _normalize_panel_frame_metadata(panel_frame_metadata)
        if labels is not None:
            self.labels = list(labels)
        elif self.frame_metadata:
            self.labels = [
                _format_metadata_label(meta, metadata_formatter, frame_idx=i, panel_idx=None)
                for i, meta in enumerate(self.frame_metadata)
            ]
        else:
            self.labels = [str(i) for i in range(self.n_slices)]
        if panel_frame_labels is not None:
            self.panel_frame_labels = [list(panel_labels) for panel_labels in panel_frame_labels]
        elif self.panel_frame_metadata:
            self.panel_frame_labels = [
                [
                    _format_metadata_label(meta, metadata_formatter, frame_idx=i, panel_idx=panel)
                    for i, meta in enumerate(panel_metadata)
                ]
                for panel, panel_metadata in enumerate(self.panel_frame_metadata)
            ]
        else:
            self.panel_frame_labels = []

        # Title and colormap - use extracted title if not explicitly provided
        self.title = title if title else (_extracted_title or "")
        self.cmap = str(cmap)  # Convert Colormap enum to string

        # Config sampling wins over a Dataset3d's default [1,1,1] sampling when
        # the user did not pass pixel_size explicitly: a ptycho config carries
        # the true Å/px (lateral) and slice thickness (depth) calibration.
        if config_data is not None and pixel_size == 0.0:
            config_pixel_size = _pixel_size_from_quantem_config(config_data)
            if config_pixel_size is not None:
                _extracted_pixel_size = config_pixel_size[1]
                _extracted_pixel_unit = "A"
                _extracted_dim_sampling = config_pixel_size[0]
                _extracted_dim_unit = _extracted_dim_unit or "A"
        # Use extracted pixel_size if not explicitly provided
        if pixel_size == 0.0 and _extracted_pixel_size is not None:
            pixel_size = _extracted_pixel_size
            if (
                (pixel_unit is None or pixel_unit in ("A", "Å", "angstrom", "Angstrom"))
                and _extracted_pixel_unit in ("nm", "nanometer")
            ):
                pixel_size *= 10
        if pixel_unit is None and _extracted_pixel_unit in ("nm", "nanometer"):
            pixel_unit = "A"
        elif pixel_unit is None:
            pixel_unit = _extracted_pixel_unit or "A"

        # Display options
        self.pixel_size = pixel_size
        self.pixel_unit = pixel_unit
        self.smooth = smooth
        # image_rotation: pure display-side (JS canvas transform), no data copy.
        self.image_rotation = image_rotation % 4
        self.log_scale = log_scale
        self.auto_contrast = auto_contrast
        self.image_vmin_pct = image_vmin_pct
        self.image_vmax_pct = image_vmax_pct
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.vmin = vmin
        self.vmax = vmax
        self._reset_panel_contrast_traits()
        self.fps = fps
        self.avg_window = avg_window

        # Timestamps
        if timestamps is not None:
            self.timestamps = [float(t) for t in timestamps]
        elif _extracted_dim_sampling is not None:
            self.timestamps = [float(i * _extracted_dim_sampling) for i in range(self.n_slices)]
        else:
            self.timestamps = []
        self.timestamp_unit = _extracted_dim_unit or timestamp_unit
        self.dim_label = dim_label
        self.dim_sampling = _extracted_dim_sampling or 1.0
        self.dim_unit = _extracted_dim_unit or ""
        self.diff_mode = diff_mode
        self._refresh_auto_contrast_ranges()
        self.show_fft = show_fft
        self.fft_layout = fft_layout
        self.fft_overlay_position = fft_overlay_position
        self.fft_overlay_size = fft_overlay_size
        self.fft_overlay_zoom = fft_overlay_zoom
        self.fft_window = fft_window
        self.fft_metrics = bool(fft_metrics)
        # Statistics are opt-in because they occupy vertical space in notebooks
        # and exported HTML, especially on phones.
        self.show_stats = False if show_stats is None else bool(show_stats)
        self.show_controls = show_controls
        self.controls_collapsed = bool(controls_collapsed)
        self.debug = bool(debug)
        self.size = size
        # Display-only filter knobs (view transforms; the stored stack stays
        # intact). Set before the offline pack and first buffer send so a
        # constructor-selected filter shows from the first paint.
        self._display_filter_ready = False
        self._display_filter_cache = {}
        # Compound spellings (bin2, bin2_anscombe, bin4_anscombe) are aliases
        # for (mode, bin); the traits always hold the canonical trio.
        from quantem.widget.utils.display_filter import resolve_denoise_mode

        resolved_mode, resolved_bin = resolve_denoise_mode(str(denoise), int(denoise_bin))
        self.denoise = resolved_mode
        self.denoise_sigma = float(denoise_sigma)
        self.denoise_bin = int(resolved_bin)
        self._display_filter_ready = True
        self._refresh_display_filter_banner(announce=True)
        self.show_denoise = (
            False
            if self.is_rgb
            else bool(show_denoise) or self._display_filter_active()
        )
        frame_bytes = self.height * self.width * 4  # float32
        # Exact float32 sliding window. Do not ship the whole stack when it
        # would cross the browser/Jupyter ~2 GB Comm cliff (36×4k×4k is 2.4 GB).
        max_frames = max(1, _MAX_PLAYBACK_CHUNK_BYTES // max(1, frame_bytes))
        self._buffer_size = min(buffer_size, self.n_slices, max_frames)

        # Initial position at middle
        self.slice_idx = int(self.n_slices // 2)
        self._roi_plot_timer = None

        # Offline mode stores the quantized uint8 stack for kernel-free scrub.
        # Exact standalone exports use _offline_float_stack on an export clone.
        # Default (offline=None): auto-enable uint8 when the pack fits a 1 GB
        # budget; opt out with offline=False for absurdly huge stacks.
        # Pick the right stack source for offline packing. separate_panel_frames
        # keeps each panel as its own array in self._separate_panel_data; the
        # offline path needs every panel, concatenated horizontally so JS
        # indexes by (sliceIdx * width * height) where `width` is the trait
        # value (= total concat width). Otherwise `_display_data` already holds
        # the concatenated or single-panel stack.
        offline_source = self._offline_pack_source()
        stack_bytes = int(np.prod(offline_source.shape))  # uint8 = 1 B/px
        if offline is None:
            offline = stack_bytes <= 1 * 1024 * 1024 * 1024
        if offline:
            self.offline = True
            self._pack_offline_u8_stack(offline_source)

        # Observers
        self.observe(self._on_slice_change, names=["slice_idx"])
        self.observe(
            self._on_roi_change,
            names=["roi_active", "roi_list", "roi_selected_idx"],
        )
        self.observe(self._on_playing_change, names=["playing"])
        self.observe(self._on_prefetch, names=["_prefetch_request"])
        self.observe(self._on_diff_mode_change, names=["diff_mode"])
        self.observe(self._on_export_request_change, names=["export_request"])
        self.observe(self._on_handoff_request_change, names=["handoff_request"])

        self._start_frame_server()

        # Initial update
        self._update_all()

        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(
                    json.loads(pathlib.Path(state).read_text()),
                    require_envelope=True,
                    expected_widget="Show3D",
                )
            else:
                state = unwrap_state_payload(state, expected_widget="Show3D")
            self.load_state_dict(state)

        # Stash wall-clock start on the instance; observer below prints the
        # TRUE end-to-end time after JS signals first paint.  The Python-only
        # __init__ number is misleading for widget UX.
        self._init_t0 = _t0
        self._init_py_elapsed_ms = (time.perf_counter() - _t0) * 1000
        self.observe(self._on_first_render, names=["_js_rendered"])


    # === Public API ===

    def to_show2d(
        self,
        frame: int | None = None,
        panel: Sequence[int | str] | int | str | None = None,
        *,
        title: str | None = None,
        copy: bool = True,
        include_hidden: bool = False,
    ):
        """Create a ``Show2D`` view from the current Show3D frame.

        By default this converts the current frame for all visible panels. Pass
        ``panel=...`` to choose a panel by index or title. Hidden panels are
        skipped unless ``include_hidden=True``.
        """
        from quantem.widget.show2d import Show2D

        idx = int(self.slice_idx if frame is None else frame)
        if not 0 <= idx < int(self.n_slices):
            raise ValueError(f"frame index {idx} out of range [0, {self.n_slices})")

        if panel is None:
            panel_indices = self.ordered_panels if include_hidden else self.visible_panels
        else:
            panel_indices = self._normalize_panel_refs(panel)
            if not include_hidden:
                hidden = set(int(i) for i in self.hidden_panels)
                panel_indices = [p for p in panel_indices if p not in hidden]
        if not panel_indices:
            raise ValueError("Show3D.to_show2d() needs at least one visible panel")

        frames = [np.asarray(self._get_display_panel_frame(p, idx), dtype=np.float32) for p in panel_indices]
        if copy:
            frames = [np.array(frame_arr, copy=True) for frame_arr in frames]
        labels = [self._static_panel_title(p, idx) for p in panel_indices]

        vmin = None
        vmax = None
        if any(v is not None for v in self.vmin_per_panel):
            vmin = [self.vmin_per_panel[p] if p < len(self.vmin_per_panel) else None for p in panel_indices]
        elif self.vmin is not None:
            vmin = self.vmin
        if any(v is not None for v in self.vmax_per_panel):
            vmax = [self.vmax_per_panel[p] if p < len(self.vmax_per_panel) else None for p in panel_indices]
        elif self.vmax is not None:
            vmax = self.vmax

        return Show2D(
            frames,
            labels=labels,
            title=title if title is not None else self.title,
            cmap=self.cmap,
            sampling=self.pixel_size if self.pixel_size > 0 else None,
            units=self.pixel_unit,
            scale_bar_visible=self.scale_bar_visible,
            show_fft=False,
            show_controls=self.show_controls,
            controls_collapsed=self.controls_collapsed,
            show_stats=self.show_stats,
            verbose=False,
            log_scale=self.log_scale,
            auto_contrast=self.auto_contrast,
            vmin=vmin,
            vmax=vmax,
            ncols=max(1, min(int(self.max_cols) if int(self.max_cols) > 0 else len(frames), len(frames))),
            size=int(self.size or 0),
            smooth=self.smooth,
            link_zoom=self.link_panels,
            link_pan=self.link_panels,
            link_contrast=self.link_contrast,
            display_bin=1,
            show_panel_titles=self.show_panel_titles,
            panel_title_font_size=int(self.panel_title_font_size or 11),
            gallery_gap_px=max(0, int(self.panel_gap)),
            save_state=False,
        )

    def _on_handoff_request_change(self, change: dict) -> None:
        """Build a Python-side prepared view from a frontend toolbar request."""
        raw = change.get("new") or ""
        if not raw:
            return
        try:
            request = json.loads(raw)
            mode = str(request.get("mode", "show2d")).lower()
            if mode == "clear":
                self.prepared_view = None
                self.prepared_view_widget = None
                self.handoff_status = ""
                return
            if mode != "show2d":
                raise ValueError(f"unsupported handoff mode {mode!r}")
            self.prepared_view = self.to_show2d(
                frame=request.get("frame", None),
                panel=request.get("panel", None),
            )
            self.prepared_view_widget = self.prepared_view
            n_images = int(getattr(self.prepared_view, "n_images", 0))
            self.handoff_status = f"Showing 2D with {n_images} panel{'s' if n_images != 1 else ''}"
        except Exception as exc:  # pragma: no cover - defensive comm boundary
            self.prepared_view = None
            self.prepared_view_widget = None
            self.handoff_status = f"View failed: {exc}"

    def _set_folder_waiting_empty_state(self) -> None:
        """Represent an acquisition folder with no readable frame yet."""
        empty = np.empty((0, 0, 0), dtype=np.float32)
        with self.hold_sync():
            self._data = empty
            self._display_data = empty
            if self._use_torch:
                self._data_torch = torch.empty((0, 0, 0), device=self._device)
            self._display_torch = None
            self.n_slices = 0
            self.height = 0
            self.width = 0
            self.labels = []
            self.frame_bytes = b""
            self.frame_seq += 1
            self._buffer_bytes = b""
            self._buffer_start = 0
            self._offline_stack = b""
            self._offline_float_stack = b""
            self._offline_mins = []
            self._offline_maxs = []
            self._static_fallback_jpeg = ""
            self.slice_idx = 0
            self.bookmarked_frames = []
            self.loop_start = 0
            self.loop_end = -1
            self.starred = [-1]
            self.panel_real_frames = []
            self.data_min = 0.0
            self.data_max = 0.0
            self._data_min_off = 0.0
            self._data_max_off = 0.0
            self.folder_waiting = True

    def set_image(
        self,
        data,
        labels: list[str] | None = None,
        *,
        display_bin: int | None = None,
    ) -> None:
        """Replace the stack data in place without rebuilding the widget.

        Swaps in a new stack while preserving display settings (cmap, contrast,
        log scale, pixel size, FFT toggle, playback config) so the operator can
        cycle through datasets in one widget cell. Resets state that is tied to
        the previous frame dimensions: ROIs and the line profile are cleared if
        ``(height, width)`` changes; bookmarks and the loop range are clamped
        to the new ``n_slices``; the playback prefetch buffer is invalidated;
        cached display data and any in-flight ROI debounce timer are dropped.

        Parameters
        ----------
        data : array_like
            3D stack ``(N, H, W)``, a 2D image (promoted to a single-frame
            stack), a torch tensor, or a quantem ``Dataset3d``. Complex data
            is rejected (cast to magnitude or phase first); non-finite data
            is rejected with a hint to ``np.nan_to_num``.
        labels : list[str] | None, optional
            New per-frame labels. If ``None``, string indices ``"0"..."N-1"``
            are used.
        display_bin : int | None, optional
            Explicit display-only binning for the replacement stack. ``None``
            keeps the historical ``set_image`` behavior of native-pixel
            display. Folder-backed updates pass their existing factor so a
            watched append cannot silently switch resolution.

        Returns
        -------
        None
            Mutates the widget in place. The browser canvas re-renders
            automatically via traitlet sync.

        Example
        -------
        >>> from quantem.widget import Show3D
        >>> w = Show3D(stack_a, title="A")  # doctest: +SKIP
        >>> w.set_image(stack_b, labels=[f"frame {i}" for i in range(len(stack_b))])  # doctest: +SKIP

        Notes
        -----
        - Prefer ``set_image`` over constructing a new ``Show3D`` when iterating
          through datasets in the same cell: it avoids re-creating the canvas
          and preserves the operator's contrast / zoom / cmap state.
        - For live stack growth, instantiate the widget with ``offline=False``:
          ``w = Show3D(first_frame[None], offline=False)``. Then append frames
          with ``w.set_image(np.stack(frames))`` and set
          ``w.slice_idx = len(frames) - 1`` to jump to the newest frame.
        - The new stack is cast to ``float32``. If float64 input contains values
          outside ``float32`` range, an error is raised (silent overflow to
          ``inf`` would corrupt stats).
        """
        if hasattr(data, "array") and hasattr(data, "name") and hasattr(data, "sampling"):
            data = data.array
        data = to_numpy(data)
        if data.ndim == 2:
            data = data[None, ...]
        # Mirror __init__'s true-color normalization: an (N, H, W, 3/4) stack
        # (or a single (H, W, 3/4) frame with H > 4) stays RGB through live
        # updates, so a growing color stack keeps rendering after set_image.
        is_rgb = False
        if data.ndim == 3 and data.shape[-1] in (3, 4) and int(data.shape[0]) > 4:
            data = data[None, ..., :3]
            is_rgb = True
        elif data.ndim == 4 and data.shape[-1] in (3, 4):
            data = data[..., :3]
            is_rgb = True
        elif data.ndim != 3:
            raise ValueError(
                f"Expected (N, H, W) gray stack or (N, H, W, 3) RGB stack, got {data.ndim}D "
                f"shape {data.shape}"
            )
        if is_rgb:
            data = data.astype(np.float32, copy=False)
            if data.size and float(np.nanmax(data)) > 1.5:
                data = data / 255.0
            data = np.clip(data, 0.0, 1.0)
        self.is_rgb = bool(is_rgb)
        if 0 in data.shape:
            raise ValueError(f"Empty stack: shape {data.shape}. All dims must be >= 1.")
        if not _all_finite(data):
            raise ValueError(
                "Data contains NaN or inf. Clean first: "
                "np.nan_to_num(arr, nan=0, posinf=0, neginf=0)."
            )
        data = _pad_stack(
            _crop_stack(data, getattr(self, "_crop", (0, 0, 0, 0))),
            getattr(self, "_padding", (0, 0)),
            getattr(self, "_pad_mode", "median"),
        )
        # Stop playback so JS doesn't keep painting from the stale _buffer_bytes
        # while we swap data. Invalidate cached torch view of display data.
        # Cancel any pending ROI plot timer so it doesn't fire mid-swap with
        # stale _display_data dims (race observed by audit).
        if getattr(self, "_roi_plot_timer", None) is not None:
            self._roi_plot_timer.cancel()
            self._roi_plot_timer = None
        self.playing = False
        self._display_torch = None
        # Clear ROIs / profile / bookmarks: their pixel coords are tied to the
        # previous (height, width) and may land out of bounds in the new stack.
        prev_h, prev_w = int(self.height), int(self.width)
        if np.iscomplexobj(data):
            raise TypeError(
                "Show3D does not accept complex data. Convert first: "
                "np.abs(arr) for magnitude or np.angle(arr) for phase."
            )
        # Same split as __init__: color to _rgb_data, luminance to _data. Without
        # this, a live RGB update leaves _rgb_data stale (or None on gray->color)
        # so the browser, HTML export, and raster export all read the wrong frame.
        data = self._establish_rgb_luminance(data, is_rgb=is_rgb)
        with np.errstate(over="ignore", invalid="ignore"):
            self._data = data.astype(np.float32, copy=False)
        # Pre-cast check ran on the source dtype; if float64 values exceed float32
        # range they silently become inf on cast and contaminate stats.
        if not _all_finite(self._data):
            raise ValueError(
                "Data exceeds float32 range (|value| > 3.4e38) after cast; "
                "values silently overflowed to inf. Rescale first: "
                "arr = arr / np.max(np.abs(arr)) or use np.log1p(np.abs(arr))."
            )
        if self._use_torch:
            self._data_torch = torch.from_numpy(self._data).to(self._device)
        self.n_panels = 1
        self.panel_titles = []
        self.starred = [-1]
        self.hidden_panels = []
        self.panel_order = []
        self.n_pages = 1
        self.page_idx = 0
        self.panels_per_page = 0
        self.page_labels = []
        self.page_starred = []
        self.hidden_page_slots = []
        self._reset_panel_contrast_traits()
        self._multi_panel_bin = 0
        self.shared_panel_source = False
        self.separate_panel_frames = False
        self._separate_panel_data = None
        self.panel_real_frames = []
        self._panel_width = int(data.shape[2])
        self.n_slices = int(data.shape[0])

        if display_bin is None:
            replacement_display_bin = 1
        elif isinstance(display_bin, bool) or not isinstance(display_bin, int) or display_bin < 1:
            raise ValueError(f"display_bin must be an integer >= 1, got {display_bin!r}")
        else:
            replacement_display_bin = int(display_bin)

        orig_h, orig_w = data.shape[1], data.shape[2]
        self._display_bin = replacement_display_bin
        if self._display_bin > 1:
            from quantem.widget.utils.array import bin2d

            self._display_data = bin2d(
                self._data,
                factor=self._display_bin,
                mode="mean",
            )
            self.height = int(self._display_data.shape[1])
            self.width = int(self._display_data.shape[2])
        else:
            self._display_data = self._data
            self.height = orig_h
            self.width = orig_w

        if self._use_torch:
            self.data_min = float(self._data_torch.min().item())
            self.data_max = float(self._data_torch.max().item())
        else:
            self.data_min = float(self._data.min())
            self.data_max = float(self._data.max())
        self._data_min_off = self.data_min
        self._data_max_off = self.data_max
        self._vmin = self._vmin_user if self._vmin_user is not None else self.data_min
        self._vmax = self._vmax_user if self._vmax_user is not None else self.data_max
        if labels is not None:
            self.labels = list(labels)
        else:
            self.labels = [str(i) for i in range(self.n_slices)]
        self.slice_idx = min(self.slice_idx, self.n_slices - 1)
        # Clear pixel-coord overlays if dims changed (stale ROIs / profile would
        # land out of bounds → empty stats or wrong-pixel sampling).
        if (self.height, self.width) != (prev_h, prev_w):
            self.roi_list = []
            self.roi_selected_idx = -1
            self.profile_line = []
            # profile_active is the JS-side toggle (no Python trait); skipping
        # Re-run bookmark validator with new n_slices (out-of-range markers
        # silently survived the swap otherwise).
        self.bookmarked_frames = list(self.bookmarked_frames)
        # Clamp loop range to new bounds.
        if self.loop_start >= self.n_slices:
            self.loop_start = 0
        if self.loop_end >= self.n_slices:
            self.loop_end = -1
        # Recompute buffer_size against new frame size, then invalidate JS-side
        # buffer (otherwise JS would slice the new H×W out of the old buffer).
        frame_bytes_n = self.height * self.width * 4
        max_frames = max(1, _MAX_PLAYBACK_CHUNK_BYTES // max(1, frame_bytes_n))
        self._buffer_size = min(self._buffer_size, self.n_slices, max_frames)
        self._buffer_bytes = b""
        if self.offline:
            # Kernel-less consumers (docs pages, exported HTML, saved widget
            # state) slice frames straight out of _offline_stack. Without a
            # repack it still holds the PREVIOUS stack, and any slice index
            # past its end renders blank after reopen. Pack from the color
            # source so RGB stacks keep true color (_display_data is luminance).
            self._pack_offline_u8_stack(self._offline_pack_source())
        self._bump_frame_server_version()
        self._refresh_auto_contrast_ranges()
        self._update_all()

    def _apply_folder_image_records(
        self,
        old_records: list[ImageFolderRecord],
        new_records: list[ImageFolderRecord],
        changed_arrays: dict[pathlib.Path, np.ndarray],
    ) -> None:
        """Replace folder frames while preserving path-addressed viewer state."""
        old_paths = [record.path for record in old_records]
        new_paths = [record.path for record in new_records]
        old_index = {path: idx for idx, path in enumerate(old_paths)}
        new_index = {path: idx for idx, path in enumerate(new_paths)}
        transform = getattr(self, "_folder_input_transform", np.asarray)
        display_bin = max(1, int(getattr(self, "_display_bin", 1)))

        if not old_paths:
            stack = np.stack(
                [transform(changed_arrays[path]) for path in new_paths],
                axis=0,
            )
            old_crop = self._crop
            old_padding = self._padding
            self._crop = (0, 0, 0, 0)
            self._padding = (0, 0)
            try:
                self.set_image(
                    stack,
                    labels=[self._folder_source.label(path) for path in new_paths],
                    display_bin=display_bin,
                )
            finally:
                self._crop = old_crop
                self._padding = old_padding
            return

        # For a color folder, existing frames must come from the (N, H, W, 3)
        # color stack, not the luminance plane in _data, or np.stack mixes
        # (H, W, 3) new frames with (H, W) old ones and raises inhomogeneous.
        existing = self._rgb_data if (self.is_rgb and self._rgb_data is not None) else self._data
        frames: list[np.ndarray] = []
        for path in new_paths:
            if path in changed_arrays:
                frames.append(transform(changed_arrays[path]))
            else:
                frames.append(np.asarray(existing[old_index[path]]))
        stack = np.stack(frames, axis=0)

        selected_path = (
            old_paths[int(self.slice_idx)]
            if old_paths and 0 <= int(self.slice_idx) < len(old_paths)
            else old_paths[0]
        )
        bookmark_paths = {
            old_paths[idx]
            for idx in self.bookmarked_frames
            if 0 <= int(idx) < len(old_paths)
        }
        starred_path = None
        if self.starred and 0 <= int(self.starred[0]) < len(old_paths):
            starred_path = old_paths[int(self.starred[0])]
        loop_start_path = old_paths[min(max(int(self.loop_start), 0), len(old_paths) - 1)]
        loop_end_path = None if int(self.loop_end) < 0 else old_paths[int(self.loop_end)]
        was_playing = bool(self.playing)
        roi_active = bool(self.roi_active)
        roi_list = list(self.roi_list)
        roi_selected_idx = int(self.roi_selected_idx)
        profile_line = list(self.profile_line)

        # Frames above are already transformed exactly once. Temporarily bypass
        # set_image's ordinary crop/pad step for this replacement.
        old_crop = self._crop
        old_padding = self._padding
        self._crop = (0, 0, 0, 0)
        self._padding = (0, 0)
        try:
            with self.hold_sync():
                self.set_image(
                    stack,
                    labels=[self._folder_source.label(path) for path in new_paths],
                    display_bin=display_bin,
                )
                self.slice_idx = new_index.get(selected_path, 0)
                self.bookmarked_frames = sorted(
                    new_index[path] for path in bookmark_paths if path in new_index
                )
                self.starred = [new_index[starred_path] if starred_path in new_index else -1]
                mapped_start = new_index.get(loop_start_path, 0)
                mapped_end = -1 if loop_end_path is None else new_index.get(loop_end_path, -1)
                self.loop_start = 0
                self.loop_end = mapped_end
                self.loop_start = mapped_start
                self.roi_active = roi_active
                self.roi_list = roi_list
                self.roi_selected_idx = roi_selected_idx
                self.profile_line = profile_line
                self.playing = was_playing
        finally:
            self._crop = old_crop
            self._padding = old_padding

    def __repr__(self) -> str:
        parts = f"Show3D({self.n_slices}×{self.height}×{self.width}, frame={self.slice_idx}, cmap={self.cmap}"
        if self.diff_mode != "off":
            parts += f", diff={self.diff_mode}"
        parts += ")"
        return parts

    def state_dict(self) -> dict:
        """Return a JSON-serializable snapshot of every user-tunable trait.

        Captures display config (cmap, contrast, log scale, FFT, diff mode),
        playback config (fps, loop, range, bookmarks, playback path), per-frame
        labels and timestamps, scale-bar settings, and any active ROIs and line
        profile. The raw stack data is NOT included; pair the snapshot with the
        original ``data`` argument (or re-attach via ``set_image``) on restore.

        Key order in the returned dict is deliberate: cross-validating pairs
        (``percentile_high`` before ``percentile_low``, ``loop_end`` before
        ``loop_start``) are emitted in the order the validators expect so a
        round-trip through ``load_state_dict`` cannot wedge in an intermediate
        state.

        Returns
        -------
        dict
            Mapping of trait name -> serializable value. Suitable for
            ``json.dump`` or use with ``save`` / ``load_state_dict``.

        Example
        -------
        >>> from quantem.widget import Show3D
        >>> w = Show3D(stack)  # doctest: +SKIP
        >>> state = w.state_dict()  # doctest: +SKIP
        >>> w2 = Show3D(stack)  # doctest: +SKIP
        >>> w2.load_state_dict(state)  # doctest: +SKIP

        Notes
        -----
        - Schema versioning is handled inside ``save``; the dict returned here
          is the unversioned inner payload.
        - To save directly to disk in one step, use ``save(path)`` instead.
        """
        return {
            # Widget discriminator: lets load_state_dict reject cross-widget
            # loads (e.g. Show3DSlices.load_state_dict(show3d.state_dict()))
            # cleanly instead of partially applying overlapping keys.
            "_widget": "Show3D",
            "title": self.title,
            "cmap": self.cmap,
            "log_scale": self.log_scale,
            "auto_contrast": self.auto_contrast,
            # percentile_high before percentile_low so cross-validator doesn't reject mid-load.
            "percentile_high": self.percentile_high,
            "percentile_low": self.percentile_low,
            "image_vmin_pct": self.image_vmin_pct,
            "image_vmax_pct": self.image_vmax_pct,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "link_contrast": self.link_contrast,
            "link_panels": self.link_panels,
            "view_state": dict(self.view_state),
            "vmin_per_panel": list(self.vmin_per_panel),
            "vmax_per_panel": list(self.vmax_per_panel),
            "show_title": self.show_title,
            "show_stats": self.show_stats,
            "debug": self.debug,
            "show_controls": self.show_controls,
            "controls_collapsed": self.controls_collapsed,
            "show_zoom_indicator": self.show_zoom_indicator,
            "show_fft": self.show_fft,
            "fft_layout": self.fft_layout,
            "fft_overlay_position": self.fft_overlay_position,
            "fft_overlay_size": self.fft_overlay_size,
            "fft_overlay_zoom": self.fft_overlay_zoom,
            "show_kymograph": self.show_kymograph,
            "fft_window": self.fft_window,
            "fft_metrics": self.fft_metrics,
            "pixel_size": self.pixel_size,
            "pixel_unit": self.pixel_unit,
            "smooth": self.smooth,
            "image_rotation": self.image_rotation,
            "scale_bar_visible": self.scale_bar_visible,
            "size": self.size,
            "fps": self.fps,
            "avg_window": self.avg_window,
            "loop": self.loop,
            "reverse": self.reverse,
            "boomerang": self.boomerang,
            # IMPORTANT: loop_end MUST precede loop_start in dict order.
            # Validators cross-check; loop_start with -1 sentinel skips check,
            # but loop_end=5 then loop_start=10 would otherwise raise during load.
            "loop_end": self.loop_end,
            "loop_start": self.loop_start,
            "bookmarked_frames": self.bookmarked_frames,
            "starred": list(self.starred),
            "hidden_panels": list(self.hidden_panels),
            "panel_order": list(self.panel_order),
            "n_pages": int(self.n_pages),
            "page_idx": int(self.page_idx),
            "panels_per_page": int(self.panels_per_page),
            "page_labels": list(self.page_labels),
            "page_starred": list(self.page_starred),
            "hidden_page_slots": list(self.hidden_page_slots),
            "playback_path": self.playback_path,
            "slice_idx": self.slice_idx,
            "roi_active": self.roi_active,
            "roi_list": self.roi_list,
            "roi_selected_idx": self.roi_selected_idx,
            "profile_line": self.profile_line,
            "profile_width": self.profile_width,
            "diff_mode": self.diff_mode,
            "denoise": self.denoise,
            "show_denoise": self.show_denoise,
            "denoise_sigma": self.denoise_sigma,
            "denoise_bin": self.denoise_bin,
            "dim_label": self.dim_label,
            "dim_sampling": self.dim_sampling,
            "dim_unit": self.dim_unit,
            "labels": list(self.labels),
            "panel_titles": list(self.panel_titles),
            "panel_frame_labels": [list(labels) for labels in self.panel_frame_labels],
            "frame_metadata": [dict(metadata) for metadata in self.frame_metadata],
            "panel_frame_metadata": [
                [dict(metadata) for metadata in panel_metadata]
                for panel_metadata in self.panel_frame_metadata
            ],
            "frame_label_format": self.frame_label_format,
            "timestamps": list(self.timestamps),
            "timestamp_unit": self.timestamp_unit,
        }

    def save(self, path: str):
        """Write the current widget state to a versioned JSON file.

        Wraps ``state_dict`` in a small envelope that records the widget type
        (``"Show3D"``) and a schema version so ``load_state_dict`` can refuse
        states that belong to a different widget. The raw stack data is NOT
        written; only the display / playback / ROI / profile configuration.

        Parameters
        ----------
        path : str
            Destination JSON file path. Parent directories must already exist.

        Returns
        -------
        None

        Example
        -------
        >>> from quantem.widget import Show3D
        >>> w = Show3D(stack, cmap="viridis", sampling=0.5, units="A")  # doctest: +SKIP
        >>> w.save("show3d_state.json")  # doctest: +SKIP

        Notes
        -----
        - To restore, instantiate ``Show3D`` with the (possibly different) data
          and call ``w.load_state_dict(json.loads(open(path).read()))``.
        - Use ``state_dict`` directly if you want to embed widget state in a
          larger document instead of a standalone file.
        """
        save_state_file(path, "Show3D", self.state_dict())

    def export_html(
        self,
        path: str | pathlib.Path | None = None,
        *,
        title: str | None = None,
        mode: str = "single",
        encoding: str = "full",
        downsample: int | None = None,
        quantized: bool | None = None,
    ) -> pathlib.Path:
        """Write a standalone HTML viewer for sharing.

        The exact export embeds the current float32 stack bytes and preserves
        numerical precision. The quantized export writes the existing offline
        uint8 representation plus global min/max metadata, making a smaller
        single-file report for visual sharing. Preferred export options are
        ``mode="single"``, ``encoding="full"`` or ``encoding="uint8"``. Use
        ``downsample=2`` / ``4`` / ``8`` with ``encoding="uint8"`` for compact
        report HTML from heavy multi-panel movies. ``quantized`` is kept as a
        compatibility alias for ``encoding="uint8"``. If the widget is playing
        when exported, the standalone HTML starts playback on load.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            Destination HTML path. Defaults to the current working directory
            with the widget title, stack shape, and export mode in the name.
        quantized : bool, default False
            If True, write the uint8 offline pack. If False, write exact
            float32 bytes.
        downsample : int, optional
            Spatial integer binning factor for the exported HTML only. The
            source widget data and live notebook view are unchanged. Binned
            exports multiply the scale-bar pixel size by this factor.
        title : str, optional
            Browser page title. Defaults to the widget title or class name.

        Returns
        -------
        pathlib.Path
            The written HTML file.
        """
        if self._data is None:
            raise ValueError("Cannot export HTML after free(); rebuild the widget first.")
        quantized, downsample_factor = self._normalise_html_export_options(
            mode=mode,
            encoding=encoding,
            downsample=downsample,
            quantized=quantized,
        )
        # RGB figure stacks: prefer uint8 offline (display-ready [0,1] → 0–255).
        # Float32 RGB of many full-panel figures is hundreds of MB and often fails
        # to open under file:// in Chrome.
        export_path = pathlib.Path(path) if path is not None else self._default_html_export_path(
            quantized,
            downsample=downsample_factor,
        )
        self._write_html_export(
            export_path,
            quantized=quantized,
            downsample=downsample_factor,
            title=title,
        )
        size_mb = export_path.stat().st_size / (1024 * 1024)
        label = self._export_mode_label(quantized, downsample=downsample_factor)
        self.export_status = f"Exported {export_path.name} ({size_mb:.1f} MB, {label})"
        return export_path

    def _normalise_html_export_options(
        self,
        *,
        mode: str = "single",
        encoding: str = "full",
        downsample: int | None = None,
        quantized: bool | None = None,
    ) -> tuple[bool, int]:
        raw_mode = str(mode or "single").strip().lower().replace("_", "-")
        if raw_mode in {"exact", "full"}:
            raw_mode = "single"
            encoding = "full"
        elif raw_mode in {"quantized", "uint8", "u8"}:
            raw_mode = "single"
            encoding = "uint8"
        if raw_mode != "single":
            raise ValueError("Show3D HTML export supports mode='single'")
        if downsample in (None, "", 0, "0"):
            downsample_factor = 1
        else:
            if isinstance(downsample, bool):
                raise ValueError("Show3D HTML export downsample must be an integer factor, not bool")
            downsample_factor = int(downsample)
        if downsample_factor < 1:
            raise ValueError(f"Show3D HTML export downsample must be >= 1, got {downsample!r}")
        if downsample_factor not in {1, 2, 4, 8}:
            raise ValueError("Show3D HTML export downsample must be one of 1, 2, 4, or 8")
        raw_encoding = str(encoding or "full").strip().lower().replace("_", "-")
        if quantized is True:
            raw_encoding = "uint8"
        elif quantized is False and raw_encoding in {"quantized", "uint8", "u8"}:
            raw_encoding = "uint8"
        if raw_encoding in {"full", "exact", "float32", "f32"}:
            if downsample_factor != 1:
                raise ValueError("Show3D exact float32 HTML export does not support downsample; use encoding='uint8'")
            return False, 1
        if raw_encoding in {"uint8", "u8", "quantized"}:
            return True, downsample_factor
        raise ValueError(f"unknown Show3D export encoding {encoding!r}; expected 'full' or 'uint8'")

    def load_state_dict(self, state: dict) -> None:
        """Apply a saved ``state_dict`` snapshot to this widget.

        Restores display, playback, ROI, and profile configuration from a dict
        produced by ``state_dict``. Unknown keys (typically from a newer widget
        version or a different widget type) are dropped with a ``UserWarning``
        instead of raising, so partial / forward-compatible loads succeed.
        Cross-validated trait pairs are applied atomically:

        - ``percentile_low`` / ``percentile_high`` are validated together and
          applied in whichever order keeps the invariant ``low < high``.
        - ``vmin`` / ``vmax`` are cleared first so either bound can be set
          regardless of the current contrast limits.
        - ``loop_start`` / ``loop_end`` are clamped to ``[0, n_slices)`` and
          applied in safe order via the ``-1`` sentinel.

        Parameters
        ----------
        state : dict
            Mapping previously returned by ``state_dict`` (or its on-disk
            equivalent). Old ``canvas_size`` aliases are migrated to ``size``;
            the constructor-derived ``display_bin`` key is ignored.

        Returns
        -------
        None
            Mutates the widget in place.

        Example
        -------
        >>> import json
        >>> from quantem.widget import Show3D
        >>> w = Show3D(stack)  # doctest: +SKIP
        >>> w.load_state_dict(json.load(open("show3d_state.json")))  # doctest: +SKIP

        Notes
        -----
        - Unknown keys raise ``warnings.warn`` rather than an exception so
          forward-compatible files from a future widget still load.
        - The new contrast (``vmin`` / ``vmax``) is also stored on the private
          user-override slots so subsequent ``set_image`` calls keep the
          loaded contrast pinned.
        """
        state = dict(state)
        # Reject cross-widget loads up front. Without this check, loading a
        # Show3DSlices state_dict() into Show3D would partially apply the
        # overlapping keys (cmap, log_scale, etc.) and leave the widget in a
        # plausible-but-wrong state.
        marker = state.pop("_widget", None)
        if marker is not None and marker != "Show3D":
            raise ValueError(
                f"load_state_dict: state was saved from {marker!r}, not Show3D. "
                f"Use the matching widget class to load it."
            )
        # Derive allowed keys from state_dict() so the two stay in lockstep -
        # adding a trait to state_dict() automatically lets load_state_dict()
        # accept it. Matches the Show4DSTEM pattern.
        allowed = set(self.state_dict().keys())
        unknown = []
        if "canvas_size" in state:
            state["size"] = state.pop("canvas_size")
        # display_filter-era saved states: migrate onto the denoise family.
        for old_key, new_key in (
            ("display_filter", "denoise"),
            ("display_sigma", "denoise_sigma"),
            ("spatial_bin", "denoise_bin"),
            ("display_filter_banner", "denoise_banner"),
        ):
            if old_key in state and new_key not in state:
                state[new_key] = state.pop(old_key)
        # `display_bin` is constructor/data dependent. Loading only the private
        # integer leaves display_data/height/width stale, so ignore saved values.
        state.pop("display_bin", None)
        # Removed no-op trait from older saved states.
        state.pop("show_playback", None)
        # Drop length-coupled traits when stack size differs from saved state.
        # Otherwise the labels/timestamps validators raise, breaking the common
        # workflow of saving a state from one trial and loading into another.
        n_cur = int(self.n_slices)
        for key in ("labels", "timestamps"):
            if key in state and isinstance(state[key], list) and 0 < len(state[key]) != n_cur:
                state.pop(key)
        if "frame_metadata" in state and isinstance(state["frame_metadata"], list) and 0 < len(state["frame_metadata"]) != n_cur:
            state.pop("frame_metadata")
        if "panel_frame_labels" in state and isinstance(state["panel_frame_labels"], list):
            n_pan = int(self.n_panels)
            real_frames = list(self.panel_real_frames)
            panel_labels = state["panel_frame_labels"]
            labels_ok = len(panel_labels) == n_pan
            if labels_ok:
                for panel, labels in enumerate(panel_labels):
                    if not isinstance(labels, list):
                        labels_ok = False
                        break
                    allowed_lengths = {n_cur}
                    if panel < len(real_frames):
                        allowed_lengths.add(int(real_frames[panel]))
                    if len(labels) not in allowed_lengths:
                        labels_ok = False
                        break
            if not labels_ok:
                state.pop("panel_frame_labels")
        if "panel_frame_metadata" in state and isinstance(state["panel_frame_metadata"], list):
            n_pan = int(self.n_panels)
            real_frames = list(self.panel_real_frames)
            panel_metadata = state["panel_frame_metadata"]
            metadata_ok = len(panel_metadata) == n_pan
            if metadata_ok:
                for panel, metadata in enumerate(panel_metadata):
                    if not isinstance(metadata, list):
                        metadata_ok = False
                        break
                    allowed_lengths = {n_cur}
                    if panel < len(real_frames):
                        allowed_lengths.add(int(real_frames[panel]))
                    if len(metadata) not in allowed_lengths:
                        metadata_ok = False
                        break
            if not metadata_ok:
                state.pop("panel_frame_metadata")
        # Drop `starred` if its length doesn't match current n_panels (e.g.
        # saved from a 4-panel widget, loading into a single-panel one).
        if "starred" in state and isinstance(state["starred"], list) and len(state["starred"]) != int(self.n_panels):
            state.pop("starred")
        if "panel_titles" in state and isinstance(state["panel_titles"], list) and len(state["panel_titles"]) != int(self.n_panels):
            state.pop("panel_titles")
        if "hidden_panels" in state and isinstance(state["hidden_panels"], list):
            n_pan = int(self.n_panels)
            clean_set: set[int] = set()
            for value in state["hidden_panels"]:
                if isinstance(value, bool):
                    continue
                try:
                    idx = int(value)
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < n_pan:
                    clean_set.add(idx)
            clean = sorted(clean_set)
            if len(clean) >= n_pan:
                clean = clean[:-1]
            state["hidden_panels"] = clean
        if "hidden_page_slots" in state and isinstance(state["hidden_page_slots"], list):
            state["hidden_page_slots"] = self._normalize_hidden_page_slots(
                state["hidden_page_slots"],
                drop_if_full=True,
            )
        elif int(self.n_pages) > 1 and "hidden_panels" in state and isinstance(state["hidden_panels"], list):
            state["hidden_page_slots"] = self._hidden_page_slots_from_panels(
                state["hidden_panels"],
                drop_if_full=True,
            )
        if "panel_order" in state and isinstance(state["panel_order"], list):
            try:
                order = [int(value) for value in state["panel_order"] if not isinstance(value, bool)]
            except (TypeError, ValueError):
                order = []
            if len(order) != int(self.n_panels) or sorted(order) != list(range(int(self.n_panels))):
                state.pop("panel_order")
            else:
                state["panel_order"] = order
        if int(self.n_pages) > 1:
            expected_panels_per_page = int(self.panels_per_page)
            state["n_pages"] = int(self.n_pages)
            state["panels_per_page"] = expected_panels_per_page
            if "page_idx" in state:
                try:
                    state["page_idx"] = int(max(0, min(int(state["page_idx"]), int(self.n_pages) - 1)))
                except (TypeError, ValueError):
                    state.pop("page_idx")
            if "page_labels" in state and isinstance(state["page_labels"], list):
                if len(state["page_labels"]) != int(self.n_pages):
                    state.pop("page_labels")
            if "page_starred" in state and isinstance(state["page_starred"], list):
                if len(state["page_starred"]) != int(self.n_pages):
                    state.pop("page_starred")
        else:
            for key in ("n_pages", "page_idx", "panels_per_page", "page_labels", "page_starred", "hidden_page_slots"):
                state.pop(key, None)
        # These were briefly present in the development branch. Contrast auto,
        # percentile, and log scale now stay global to match Show2D.
        for key in (
            "auto_contrast_per_panel",
            "percentile_low_per_panel",
            "percentile_high_per_panel",
            "log_scale_per_panel",
        ):
            state.pop(key, None)
        panel_len_keys = ("vmin_per_panel", "vmax_per_panel")
        for key in panel_len_keys:
            if key in state and isinstance(state[key], list) and len(state[key]) != int(self.n_panels):
                state.pop(key)
        # Apply per-panel vmin/vmax as a pair, pre-cleared, so the cross-validator
        # (vmin_per_panel[i] <= vmax_per_panel[i]) never trips mid-load on the
        # generic setattr loop when the incoming vmin exceeds the CURRENT vmax.
        pp_vmin = state.pop("vmin_per_panel", None)
        pp_vmax = state.pop("vmax_per_panel", None)
        if pp_vmin is not None or pp_vmax is not None:
            n_pan = int(self.n_panels)
            self.vmin_per_panel = [None] * n_pan
            self.vmax_per_panel = [None] * n_pan
            if pp_vmax is not None:
                self.vmax_per_panel = list(pp_vmax)
            if pp_vmin is not None:
                self.vmin_per_panel = list(pp_vmin)
        if int(self.n_panels) > 1:
            for key in ("roi_active", "roi_list", "roi_selected_idx"):
                state.pop(key, None)
        for key in list(state):
            if key not in allowed:
                unknown.append(key)
                state.pop(key)

        pct_low_marker = object()
        pct_high_marker = object()
        pct_low = state.pop("percentile_low", pct_low_marker)
        pct_high = state.pop("percentile_high", pct_high_marker)
        if pct_low is not pct_low_marker or pct_high is not pct_high_marker:
            low = float(self.percentile_low if pct_low is pct_low_marker else pct_low)
            high = float(self.percentile_high if pct_high is pct_high_marker else pct_high)
            if not (0 <= low <= 100 and 0 <= high <= 100 and low < high):
                raise traitlets.TraitError(
                    f"percentile_low ({low}) must be < percentile_high ({high}) and both in [0, 100]"
                )
            if high <= float(self.percentile_low):
                self.percentile_low = low
                self.percentile_high = high
            else:
                self.percentile_high = high
                self.percentile_low = low

        vmin_marker = object()
        vmax_marker = object()
        vmin = state.pop("vmin", vmin_marker)
        vmax = state.pop("vmax", vmax_marker)
        if vmin is not vmin_marker or vmax is not vmax_marker:
            new_vmin = self.vmin if vmin is vmin_marker else vmin
            new_vmax = self.vmax if vmax is vmax_marker else vmax
            if new_vmin is not None and new_vmax is not None and float(new_vmin) > float(new_vmax):
                raise traitlets.TraitError(f"vmin ({new_vmin}) must be <= vmax ({new_vmax})")
            self.vmin = None
            self.vmax = None
            if new_vmin is not None:
                self.vmin = float(new_vmin)
            if new_vmax is not None:
                self.vmax = float(new_vmax)

        loop_start_marker = object()
        loop_end_marker = object()
        loop_start = state.pop("loop_start", loop_start_marker)
        loop_end = state.pop("loop_end", loop_end_marker)
        if loop_start is not loop_start_marker or loop_end is not loop_end_marker:
            n = max(1, int(self.n_slices))
            start = int(self.loop_start if loop_start is loop_start_marker else loop_start)
            end = int(self.loop_end if loop_end is loop_end_marker else loop_end)
            start = max(0, min(start, n - 1))
            if end >= 0:
                end = min(end, n - 1)
                if start > end:
                    raise traitlets.TraitError(f"loop_start ({start}) must be <= loop_end ({end})")
            else:
                end = -1
            self.loop_start = 0
            self.loop_end = end
            self.loop_start = start

        for key, val in state.items():
            setattr(self, key, val)
        self._vmin_user = self.vmin
        self._vmax_user = self.vmax
        self._vmin = self.vmin if self.vmin is not None else self.data_min
        self._vmax = self.vmax if self.vmax is not None else self.data_max
        if unknown:
            warnings.warn(
                f"load_state_dict ignored unknown keys: {unknown}. "
                "These may be from a newer widget version or a different widget type.",
                stacklevel=2,
            )

    def summary(self) -> None:
        """Print a one-screen status report for the current widget.

        Sections include: title and stack shape (with pixel-size readout when
        set), current frame index and label, raw data min/max/mean, display
        config (cmap, contrast, log/linear, FFT, diff mode), playback config
        (fps, loop, reverse, boomerang) and loop range, active ROI count, line
        profile endpoints if set, and render timing once the first browser
        paint has fired. Useful for notebook reproducibility and bug reports.

        Returns
        -------
        None
            Prints to stdout.

        Example
        -------
        >>> from quantem.widget import Show3D
        >>> w = Show3D(stack, title="defocus")  # doctest: +SKIP
        >>> w.summary()  # doctest: +SKIP

        Notes
        -----
        - ``Rendered:`` reads ``(pending first browser paint)`` until the JS
          side has round-tripped its first paint timestamp.
        """
        lines = [self.title or "Show3D", "═" * 32]
        lines.append(f"Stack:    {self.n_slices}×{self.height}×{self.width}")
        if self.pixel_size > 0:
            ps = self.pixel_size
            if ps >= 10:
                lines[-1] += f" ({ps / 10:.2f} nm/px)"
            else:
                lines[-1] += f" ({ps:.2f} Å/px)"
        lines.append(f"Frame:    {self.slice_idx}/{self.n_slices - 1}")
        if self.labels and self.slice_idx < len(self.labels):
            lines[-1] += f" [{self.labels[self.slice_idx]}]"
        if hasattr(self, "_data") and self._data is not None:
            arr = self._data
            lines.append(f"Data:     min={float(arr.min()):.4g}  max={float(arr.max()):.4g}  mean={float(arr.mean()):.4g}")
        cmap = self.cmap
        scale = "log" if self.log_scale else "linear"
        if self.vmin is not None and self.vmax is not None:
            contrast = f"vmin={self.vmin:.4g}, vmax={self.vmax:.4g}"
        elif self.auto_contrast:
            contrast = "auto contrast"
        else:
            contrast = "manual contrast"
        display = f"{cmap} | {contrast} | {scale}"
        if self.show_fft:
            display += " | FFT"
            if not self.fft_window:
                display += " (no window)"
        if self.diff_mode != "off":
            display += f" | diff={self.diff_mode}"
        lines.append(f"Display:  {display}")
        lines.append(f"Playback: {self.fps} fps | loop={'on' if self.loop else 'off'} | reverse={'on' if self.reverse else 'off'} | boomerang={'on' if self.boomerang else 'off'}")
        if self.loop_start > 0 or self.loop_end >= 0:
            end = self.loop_end if self.loop_end >= 0 else self.n_slices - 1
            lines.append(f"Range:    {self.loop_start}–{end}")
        if self.roi_active and self.roi_list:
            lines.append(f"ROI:      {len(self.roi_list)} region(s)")
        if len(self.profile_line) >= 2:
            p0, p1 = self.profile_line[0], self.profile_line[1]
            lines.append(f"Profile:  ({p0['row']:.0f}, {p0['col']:.0f}) → ({p1['row']:.0f}, {p1['col']:.0f}) width={self.profile_width}")
        rt = getattr(self, "render_total_ms", None)
        if rt is not None:
            pb = getattr(self, "render_python_build_ms", 0)
            wj = getattr(self, "render_wire_js_ms", 0)
            lines.append(f"Rendered: {rt} ms total (Python build {pb} ms, wire+JS {wj} ms)")
        else:
            lines.append("Rendered: (pending first browser paint)")
        print("\n".join(lines))

    def play(self) -> Self:
        """Start playback from the current frame.

        Sets the ``playing`` trait to ``True``, which triggers the JS
        playback loop and the Python-side sliding-prefetch buffer that
        ships chunks of frames ahead of the scrubber position.

        Returns
        -------
        Self
            The widget, for chaining (``w.play().goto(0)``).

        Example
        -------
        >>> from quantem.widget import Show3D
        >>> w = Show3D(stack, fps=10)  # doctest: +SKIP
        >>> w.play()  # doctest: +SKIP
        """
        self.playing = True
        return self

    def pause(self) -> Self:
        """Pause playback at the current frame.

        Sets ``playing`` to ``False`` without resetting ``slice_idx``.
        Per-frame statistics are refreshed for the current frame on pause.

        Returns
        -------
        Self
            The widget, for chaining.

        Example
        -------
        >>> from quantem.widget import Show3D
        >>> w = Show3D(stack)  # doctest: +SKIP
        >>> w.play().pause()  # doctest: +SKIP
        """
        self.playing = False
        return self

    def stop(self) -> Self:
        """Stop playback and jump back to frame 0.

        Returns
        -------
        Self
            The widget, for chaining.

        Example
        -------
        >>> from quantem.widget import Show3D
        >>> w = Show3D(stack)  # doctest: +SKIP
        >>> w.play().stop()  # doctest: +SKIP

        Notes
        -----
        - Unlike ``pause``, ``stop`` resets ``slice_idx`` to 0. Use ``pause``
          to keep the current frame visible.
        """
        self.playing = False
        self.slice_idx = 0
        return self

    def goto(self, index: int) -> Self:
        """Jump to a specific frame index.

        The index is taken modulo ``n_slices``, so negative or out-of-range
        values wrap rather than raise.

        Parameters
        ----------
        index : int
            Target frame index. Wrapped into ``[0, n_slices)``.

        Returns
        -------
        Self
            The widget, for chaining.

        Example
        -------
        >>> from quantem.widget import Show3D
        >>> w = Show3D(stack)  # doctest: +SKIP
        >>> w.goto(5)  # doctest: +SKIP
        >>> w.goto(-1)  # last frame  # doctest: +SKIP
        """
        self.slice_idx = int(index) % self.n_slices
        return self

    def star_panel(self, panel: int = 0, frame: int | None = None) -> Self:
        """Mark a "best frame" star on a panel.

        Parameters
        ----------
        panel : int, default 0
            Panel index (0-based). For single-panel widgets, panel=0 is the
            only valid value.
        frame : int | None, default None
            Frame index to star. ``None`` stars the currently displayed frame
            (``slice_idx``).

        Returns
        -------
        Self
            The widget, for chaining.

        Example
        -------
        >>> from quantem.widget import Show3D
        >>> w = Show3D(stack_a, stack_b)  # doctest: +SKIP
        >>> w.goto(50)  # doctest: +SKIP
        >>> w.star_panel(0)            # mark frame 50 as best on panel 0
        >>> w.star_panel(1, frame=80)  # mark frame 80 as best on panel 1
        >>> w.starred_frames           # → {0: 50, 1: 80}  # doctest: +SKIP
        """
        if not (0 <= panel < self.n_panels):
            raise ValueError(f"panel {panel} out of range [0, {self.n_panels})")
        if frame is None:
            frame = int(self.slice_idx)
        starred = list(self.starred)
        if len(starred) != self.n_panels:
            starred = [-1] * self.n_panels
        starred[panel] = int(frame)
        self.starred = starred
        return self

    def unstar_panel(self, panel: int) -> Self:
        """Clear the star on a panel (sets `starred[panel] = -1`)."""
        if not (0 <= panel < self.n_panels):
            raise ValueError(f"panel {panel} out of range [0, {self.n_panels})")
        starred = list(self.starred)
        if len(starred) != self.n_panels:
            starred = [-1] * self.n_panels
        starred[panel] = -1
        self.starred = starred
        return self

    @property
    def starred_frames(self) -> dict[int, int]:
        """Mapping of panel index → starred frame index, only for panels that
        have a star set. Returns ``{}`` if no panel is starred. Useful for
        downstream code like ``best_iters = {trial: w.starred_frames.get(i)}``."""
        return {i: f for i, f in enumerate(self.starred) if f >= 0}

    @property
    def starred_pages(self) -> list[int]:
        """Zero-based page indices marked with a star."""
        return [i for i, value in enumerate(self.page_starred) if value]

    @property
    def ordered_panels(self) -> list[int]:
        """Zero-based movie panel indices in the current display order."""
        if int(self.n_pages) > 1 and int(self.panels_per_page) > 0:
            start = int(self.page_idx) * int(self.panels_per_page)
            return list(range(start, min(start + int(self.panels_per_page), int(self.n_panels))))
        order = list(self.panel_order or [])
        if len(order) == int(self.n_panels) and sorted(order) == list(range(int(self.n_panels))):
            return order
        return list(range(int(self.n_panels)))

    @property
    def visible_panels(self) -> list[int]:
        """Zero-based panel indices currently visible in the canvas grid."""
        if int(self.n_pages) > 1 and int(self.panels_per_page) > 0:
            start = int(self.page_idx) * int(self.panels_per_page)
            hidden_slots = set(self.hidden_page_slots)
            visible = [
                panel
                for panel in self.ordered_panels
                if (panel - start) not in hidden_slots
            ]
            return visible or self.ordered_panels[:1]
        hidden = set(self.hidden_panels)
        return [i for i in self.ordered_panels if i not in hidden]

    def star_page(self, page: int) -> Self:
        """Mark a page with a star."""
        idx = int(page)
        if idx < 0 or idx >= int(self.n_pages):
            raise ValueError(f"page index {idx} out of range [0, {self.n_pages})")
        starred = list(self.page_starred)
        if len(starred) != int(self.n_pages):
            starred = [0] * int(self.n_pages)
        starred[idx] = 1
        self.page_starred = starred
        return self

    def unstar_page(self, page: int) -> Self:
        """Clear the star on a page."""
        idx = int(page)
        if idx < 0 or idx >= int(self.n_pages):
            raise ValueError(f"page index {idx} out of range [0, {self.n_pages})")
        starred = list(self.page_starred)
        if len(starred) != int(self.n_pages):
            starred = [0] * int(self.n_pages)
        starred[idx] = 0
        self.page_starred = starred
        return self

    def set_hidden_panels(self, panels: Sequence[int | str] | int | str) -> Self:
        """Replace the hidden panel set by index or exact panel title.

        Hidden panels are collapsed from the canvas grid and skipped by
        panel-scoped frontend computations, but remain available in state and
        standalone HTML export. At least one panel must stay visible.

        Parameters
        ----------
        panels : sequence of int or str, int, or str
            Panels to hide. Integers are zero-based panel indices; strings must
            match a panel title exactly.

        Returns
        -------
        Show3D
            The widget, for chaining.

        Examples
        --------
        >>> w = Show3D(a, b, panel_titles=["SSB", "Mean DP"])  # doctest: +SKIP
        >>> w.set_hidden_panels(["Mean DP"])  # doctest: +SKIP
        >>> w.set_hidden_panels([1])  # doctest: +SKIP
        """
        hidden = self._normalize_panel_refs(panels, allow_empty=True)
        if int(self.n_pages) > 1:
            try:
                hidden_page_slots = self._hidden_page_slots_from_panels(hidden)
            except traitlets.TraitError as exc:
                raise ValueError(
                    "set_hidden_panels would hide every page slot; leave at least one visible"
                ) from exc
        else:
            hidden_page_slots = []
        if len(hidden) >= int(self.n_panels):
            raise ValueError("set_hidden_panels would hide every panel; leave at least one visible")
        self.hidden_panels = sorted(hidden)
        self.hidden_page_slots = hidden_page_slots
        return self

    def hide_panel(self, *panels: int | str) -> Self:
        """Hide one or more panels by zero-based index or exact title.

        Hidden panels are not removed from the widget; they can be restored with
        ``show_panel`` or ``show_all_panels`` and are preserved in HTML export.
        """
        to_hide = set(self.hidden_panels)
        to_hide.update(self._normalize_panel_refs(list(panels)))
        if int(self.n_pages) > 1:
            try:
                hidden_page_slots = self._hidden_page_slots_from_panels(sorted(to_hide))
            except traitlets.TraitError as exc:
                raise ValueError("hide_panel would hide every page slot; leave at least one visible") from exc
        else:
            hidden_page_slots = []
        if len(to_hide) >= int(self.n_panels):
            raise ValueError("hide_panel would hide every panel; leave at least one visible")
        self.hidden_panels = sorted(to_hide)
        self.hidden_page_slots = hidden_page_slots
        return self

    def show_panel(self, *panels: int | str) -> Self:
        """Restore one or more hidden panels by zero-based index or exact title."""
        to_show = set(self._normalize_panel_refs(list(panels)))
        hidden = sorted(set(self.hidden_panels) - to_show)
        self.hidden_panels = hidden
        self.hidden_page_slots = self._hidden_page_slots_from_panels(hidden)
        return self

    def show_all_panels(self) -> Self:
        """Restore every panel in the canvas grid."""
        self.hidden_panels = []
        self.hidden_page_slots = []
        return self

    def set_panel_order(self, panels: Sequence[int | str]) -> Self:
        """Set the multi-panel display order by panel index or exact title.

        The order is display-only: source data, titles, hidden state, stars,
        and per-panel contrast stay keyed by their original panel indices.
        """
        order = self._normalize_panel_refs(panels, allow_empty=True)
        if not order:
            self.panel_order = []
            return self
        expected = set(range(int(self.n_panels)))
        if len(order) != int(self.n_panels) or set(order) != expected:
            raise ValueError("set_panel_order requires every panel exactly once")
        self.panel_order = order
        return self

    def reset_panel_order(self) -> Self:
        """Restore the natural source-panel order."""
        self.panel_order = []
        return self

    def move_panel(self, panel: int | str, position: int) -> Self:
        """Move one panel to a zero-based display position."""
        idx = self._resolve_panel_ref(panel)
        order = self.ordered_panels
        order.remove(idx)
        pos = int(position)
        if pos < 0:
            pos = 0
        if pos > len(order):
            pos = len(order)
        order.insert(pos, idx)
        self.panel_order = order
        return self

    def collapse_controls(self) -> Self:
        """Collapse the live control UI while leaving the GUI toggle available."""
        self.controls_collapsed = True
        return self

    def expand_controls(self) -> Self:
        """Expand the live control UI."""
        self.controls_collapsed = False
        return self

    def toggle_controls(self) -> Self:
        """Toggle the collapsed state of the live control UI."""
        self.controls_collapsed = not bool(self.controls_collapsed)
        return self

    @property
    def visible_indices(self) -> list[int]:
        """Live list of frame indices NOT in hidden_indices. Read-only;
        mutate via set_hidden() / show_all() / hide()."""
        hidden = set(self.hidden_indices)
        return [i for i in range(self.n_slices) if i not in hidden]

    def hide(self, *indices: int) -> Self:
        """Mark one or more frames as hidden from the scrubber.

        Hidden frames are excluded from the scrubber UI and from playback but
        kept in memory; restore them with ``show`` or ``show_all``. Idempotent:
        hiding an already-hidden index is a no-op. At least one frame always
        stays visible, so a call that would hide every frame is silently
        rejected.

        Parameters
        ----------
        *indices : int
            One or more frame indices to hide. Values outside ``[0, n_slices)``
            are accepted but never become visible-or-hidden in the UI.

        Returns
        -------
        Show3D
            The widget, for chaining (``w.hide(0, 1).hide(5)``).

        Example
        -------
        >>> from quantem.widget import Show3D
        >>> w = Show3D(stack)  # doctest: +SKIP
        >>> w.hide(0, 1, 2)  # doctest: +SKIP

        Notes
        -----
        - The ``hideable`` trait must be ``True`` for the JS scrubber to show
          the hide overlay; the underlying ``hidden_indices`` trait is always
          honored regardless.
        """
        keep = set(self.hidden_indices) | {int(i) for i in indices}
        # Always keep at least one frame visible.
        if len(keep) >= self.n_slices:
            return self
        self.hidden_indices = sorted(keep)
        return self

    def show(self, *indices: int) -> Self:
        """Restore one or more previously hidden frames.

        Idempotent: indices that are not currently hidden are silently ignored.

        Parameters
        ----------
        *indices : int
            Frame indices to make visible again.

        Returns
        -------
        Show3D
            The widget, for chaining.

        Example
        -------
        >>> from quantem.widget import Show3D
        >>> w = Show3D(stack)  # doctest: +SKIP
        >>> w.hide(2).show(2)  # doctest: +SKIP
        """
        drop = {int(i) for i in indices}
        self.hidden_indices = sorted(set(self.hidden_indices) - drop)
        return self

    def set_hidden(self, indices: list[int]) -> Self:
        """Replace the hidden set wholesale.

        Discards any current ``hidden_indices`` and installs ``indices``
        clamped to ``[0, n_slices)``. At least one frame is always visible;
        if the input would hide every frame, the largest index is dropped.

        Parameters
        ----------
        indices : list[int]
            Full replacement set of hidden frame indices. Order and duplicates
            are normalized internally.

        Returns
        -------
        Show3D
            The widget, for chaining.

        Example
        -------
        >>> from quantem.widget import Show3D
        >>> w = Show3D(stack)  # doctest: +SKIP
        >>> w.set_hidden([0, 1, 2, 7])  # doctest: +SKIP
        """
        clean = sorted({int(i) for i in indices if 0 <= int(i) < self.n_slices})
        # Always keep at least one frame visible.
        if len(clean) >= self.n_slices:
            clean = clean[:-1]
        self.hidden_indices = clean
        return self

    def show_all(self) -> Self:
        """Clear the hidden set so every frame is visible.

        Returns
        -------
        Show3D
            The widget, for chaining.

        Example
        -------
        >>> from quantem.widget import Show3D
        >>> w = Show3D(stack)  # doctest: +SKIP
        >>> w.hide(0, 1, 2).show_all()  # doctest: +SKIP
        """
        self.hidden_indices = []
        return self

    @property
    def roi(self) -> dict:
        """The selected ROI dict (or the first ROI if none is selected)."""
        idx = self.roi_selected_idx
        if 0 <= idx < len(self.roi_list):
            return self.roi_list[idx]
        if self.roi_list:
            return self.roi_list[0]
        return {}

    def add_roi(self, row: int | None = None, col: int | None = None, shape: str = "square") -> Self:
        with self.hold_sync():
            self._upsert_selected_roi({
                "shape": shape,
                "row": int(self.height // 2 if row is None else row),
                "col": int(self.width // 2 if col is None else col),
            })
        return self

    def clear_rois(self) -> Self:
        with self.hold_sync():
            self.roi_list = []
            self.roi_selected_idx = -1
            self.roi_active = False
        return self

    def delete_selected_roi(self) -> Self:
        """Delete the currently selected ROI."""
        idx = int(self.roi_selected_idx)
        if idx < 0 or idx >= len(self.roi_list):
            return self
        with self.hold_sync():
            rois = [roi for i, roi in enumerate(self.roi_list) if i != idx]
            self.roi_list = rois
            self.roi_selected_idx = min(idx, len(rois) - 1) if rois else -1
            if not rois:
                self.roi_active = False
        return self

    def set_roi(self, row: int, col: int, radius: int = 10) -> Self:
        """Set selected ROI position and size (creates one if needed)."""
        with self.hold_sync():
            self._upsert_selected_roi({"shape": "circle", "row": int(row), "col": int(col), "radius": int(radius)})
        return self

    def roi_circle(self, radius: int = 10) -> Self:
        """Set selected ROI shape to circle."""
        with self.hold_sync():
            self._upsert_selected_roi({"shape": "circle", "radius": int(radius)})
        return self

    def roi_square(self, half_size: int = 10) -> Self:
        """Set selected ROI shape to square."""
        with self.hold_sync():
            self._upsert_selected_roi({"shape": "square", "radius": int(half_size)})
        return self

    def roi_rectangle(self, width: int = 20, height: int = 10) -> Self:
        """Set selected ROI shape to rectangle."""
        with self.hold_sync():
            self._upsert_selected_roi({"shape": "rectangle", "width": int(width), "height": int(height)})
        return self

    def roi_annular(self, inner: int = 5, outer: int = 10) -> Self:
        """Set selected ROI shape to annular (donut)."""
        with self.hold_sync():
            self._upsert_selected_roi({"shape": "annular", "radius_inner": int(inner), "radius": int(outer)})
        return self

    @property
    def profile(self):
        """Get profile line endpoints as [(row0, col0), (row1, col1)] or []."""
        return [(p["row"], p["col"]) for p in self.profile_line]

    @property
    def profile_values(self):
        """Get profile values for the current frame and active single-panel page."""
        if len(self.profile_line) < 2:
            return None
        p0, p1 = self.profile_line
        return self._sample_profile(p0["row"], p0["col"], p1["row"], p1["col"])

    @property
    def profile_distance(self):
        """Get total distance of the profile line in calibrated units (Å or px)."""
        if len(self.profile_line) < 2:
            return None
        p0, p1 = self.profile_line
        dc = p1["col"] - p0["col"]
        dr = p1["row"] - p0["row"]
        dist_px = (dc**2 + dr**2) ** 0.5
        if self.pixel_size > 0:
            return dist_px * self.pixel_size
        return dist_px

    def set_profile(self, start: tuple[float, float], end: tuple[float, float]) -> Self:
        """Set a line profile between two points (image pixel coordinates).

        Parameters
        ----------
        start : tuple of (row, col)
            Start point in pixel coordinates.
        end : tuple of (row, col)
            End point in pixel coordinates.
        """
        row0, col0 = start
        row1, col1 = end
        self.profile_line = [
            {"row": float(row0), "col": float(col0)},
            {"row": float(row1), "col": float(col1)},
        ]
        return self

    def clear_profile(self) -> Self:
        """Clear the current line profile."""
        self.profile_line = []
        return self

    def _resolve_profile_panel(
        self,
        *,
        panel: int | None = None,
        page: int | None = None,
    ) -> int:
        """Resolve a profile source panel, defaulting to the active page."""
        if panel is not None and page is not None:
            raise ValueError("Pass either panel or page for a profile, not both")
        if page is not None:
            if int(self.n_pages) <= 1 or int(self.panels_per_page) <= 0:
                raise ValueError("page is only valid for a paged Show3D widget")
            page_idx = int(page)
            if page_idx < 0 or page_idx >= int(self.n_pages):
                raise IndexError(f"page {page_idx} out of range [0, {self.n_pages})")
            panel_idx = page_idx * int(self.panels_per_page)
        elif panel is not None:
            panel_idx = int(panel)
        elif int(self.n_pages) > 1 and int(self.panels_per_page) > 0:
            panel_idx = int(self.page_idx) * int(self.panels_per_page)
        else:
            panel_idx = 0
        if panel_idx < 0 or panel_idx >= int(self.n_panels):
            raise IndexError(f"panel {panel_idx} out of range [0, {self.n_panels})")
        return panel_idx

    def _raw_profile_stack(self, panel: int) -> np.ndarray:
        """Return one display-resolution panel stack without diff processing."""
        if self.separate_panel_frames and self._separate_panel_data is not None:
            return np.asarray(self._separate_panel_data[panel], dtype=np.float32)
        data = np.asarray(self._display_data, dtype=np.float32)
        if int(self.n_panels) <= 1 or self.shared_panel_source:
            return data
        panel_width = int(self.panel_width_px) or data.shape[2] // int(self.n_panels)
        start = panel * panel_width
        return data[:, :, start : start + panel_width]

    def profile_all_frames(
        self,
        start: tuple[float, float] | None = None,
        end: tuple[float, float] | None = None,
        *,
        panel: int | None = None,
        page: int | None = None,
    ) -> np.ndarray:
        """Extract the line profile from every frame, returning (n_slices, n_points).

        Uses the current profile_line unless start/end are provided.
        Always samples raw data (ignores diff_mode).

        Parameters
        ----------
        start : tuple of (row, col), optional
            Start point. Overrides current profile_line.
        end : tuple of (row, col), optional
            End point. Overrides current profile_line.
        panel : int, optional
            Absolute source-panel index. Defaults to the first panel on the
            active page, or panel 0 for a non-paged widget.
        page : int, optional
            Page index for a paged widget. The first panel on that page is
            sampled. Pass ``panel`` instead when a page contains several panels.

        Returns
        -------
        np.ndarray
            Shape (n_slices, n_points) float32 array.
        """
        if start is not None and end is not None:
            row0, col0 = float(start[0]), float(start[1])
            row1, col1 = float(end[0]), float(end[1])
        elif len(self.profile_line) >= 2:
            p0, p1 = self.profile_line[0], self.profile_line[1]
            row0, col0 = p0["row"], p0["col"]
            row1, col1 = p1["row"], p1["col"]
        else:
            raise ValueError(
                "No profile line set. Call set_profile() first or pass start/end."
            )
        panel_idx = self._resolve_profile_panel(panel=panel, page=page)
        stack = self._raw_profile_stack(panel_idx)
        rows = []
        for i in range(self.n_slices):
            rows.append(self._sample_profile_on(stack[i], row0, col0, row1, col1))
        return np.stack(rows)

    def profile_all_pages(
        self,
        start: tuple[float, float] | None = None,
        end: tuple[float, float] | None = None,
        *,
        panel_slot: int = 0,
    ) -> np.ndarray:
        """Extract one spatially matched profile stack from every page.

        Returns an array shaped ``(n_pages, n_slices, n_points)``. The same
        profile endpoints and panel slot are used on every page, which makes
        raw/corrected or method-to-method depth comparisons directly aligned.
        """
        if int(self.n_pages) <= 1 or int(self.panels_per_page) <= 0:
            raise ValueError("profile_all_pages requires a paged Show3D widget")
        slot = int(panel_slot)
        if slot < 0 or slot >= int(self.panels_per_page):
            raise IndexError(
                f"panel_slot {slot} out of range [0, {self.panels_per_page})"
            )
        return np.stack(
            [
                self.profile_all_frames(
                    start,
                    end,
                    panel=page * int(self.panels_per_page) + slot,
                )
                for page in range(int(self.n_pages))
            ]
        )

    def save_image(self, path: str | pathlib.Path, *, frame_idx: int | None = None,
                   format: str | None = None, dpi: int = 150) -> pathlib.Path:
        """Save a single frame as a PNG, PDF, or TIFF file.

        The saved image is colorized with the current ``cmap`` and contrast
        (``vmin`` / ``vmax`` or percentile auto-contrast), so the saved file
        matches what the browser shows for that frame. ``diff_mode`` is
        respected: ``"previous"`` saves ``frame - frame[idx-1]`` and ``"first"``
        saves ``frame - frame[0]``.

        Parameters
        ----------
        path : str or pathlib.Path
            Output file path. Parent directories are created if needed.
        frame_idx : int | None, optional
            Frame index to save. Defaults to the current ``slice_idx``.
        format : str | None, optional
            One of ``"png"``, ``"pdf"``, ``"tiff"``. If omitted, inferred
            from the file extension; defaults to ``"png"`` if no extension.
        dpi : int, default 150
            DPI metadata written into the file.

        Returns
        -------
        pathlib.Path
            The written file path.

        Example
        -------
        >>> from quantem.widget import Show3D
        >>> w = Show3D(stack, cmap="viridis")  # doctest: +SKIP
        >>> w.save_image("frame_5.png", frame_idx=5)  # doctest: +SKIP
        >>> w.save_image("frame.pdf", dpi=300)  # doctest: +SKIP

        Notes
        -----
        - PDF output is converted to RGB internally (no alpha channel).
        - Frame indices outside ``[0, n_slices)`` raise ``IndexError``;
          unsupported extensions raise ``ValueError``.
        - For multi-frame output, call ``save_image`` in a loop.
        """
        from matplotlib import colormaps
        from PIL import Image

        path = pathlib.Path(path)
        fmt = (format or path.suffix.lstrip(".").lower() or "png").lower()
        if fmt not in ("png", "pdf", "tiff", "tif"):
            raise ValueError(f"Unsupported format: {fmt!r}. Use 'png', 'pdf', or 'tiff'.")

        idx = frame_idx if frame_idx is not None else self.slice_idx
        if idx < 0 or idx >= self.n_slices:
            raise IndexError(f"Frame index {idx} out of range [0, {self.n_slices})")

        # True color: save the RGB frame directly (no colormap), matching what
        # the browser paints. diff_mode is a scalar-intensity tool, so it stays
        # on the luminance path only.
        if self.is_rgb and self._rgb_data is not None:
            rgba = np.clip(self._rgb_data[idx] * 255.0, 0, 255).astype(np.uint8)
        else:
            # Respect diff_mode so saved frame matches what user sees.
            frame = self._data[idx]
            if self.diff_mode == "previous":
                frame = frame - self._data[idx - 1] if idx > 0 else np.zeros_like(frame)
            elif self.diff_mode == "first":
                frame = frame - self._data[0]
            normalized = self._normalize_frame(frame)
            cmap_fn = colormaps.get_cmap(self.cmap)
            rgba = (cmap_fn(normalized / 255.0) * 255).astype(np.uint8)

        img = Image.fromarray(rgba)
        if fmt == "pdf":
            Image.init()
            img = img.convert("RGB")
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path), dpi=(dpi, dpi))
        return path

    def _animation_frame_order(self, playback: str) -> list[int]:
        """Return visible frame indices in the requested animation order."""
        visible = list(self.visible_indices)
        if not visible:
            visible = list(range(int(self.n_slices)))
        mode = str(playback).lower()
        if mode == "forward":
            return visible
        if mode in {"bounce", "boomerang"}:
            if len(visible) <= 1:
                return visible
            return visible + visible[-2:0:-1]
        raise ValueError("playback must be 'forward' or 'bounce'")

    def _animation_panel_indices(self) -> list[int]:
        """Return visible panel indices for panel-only animation export."""
        panels = self.visible_panels
        if not panels:
            raise ValueError("cannot export animation with every panel hidden")
        return panels

    def _animation_frame_labels(self, panel_indices: list[int], frame_idx: int) -> list[str]:
        labels = getattr(self, "panel_frame_labels", [])
        out: list[str] = []
        for panel in panel_indices:
            if panel < len(labels) and frame_idx < len(labels[panel]):
                out.append(str(labels[panel][frame_idx]))
            else:
                out.append(f"{frame_idx + 1}/{self.n_slices}")
        return out

    def _render_animation_frames(
        self,
        *,
        quality: str,
        playback: str,
        show_frame_labels: bool,
        background: str | tuple[int, int, int],
    ) -> list[Any]:
        """Render panel-only animation frames as RGB PIL images."""
        from quantem.widget.render import gif as gif_utils

        if quality not in gif_utils.QUALITY_SCALE:
            raise ValueError(f"quality must be one of {list(gif_utils.QUALITY_SCALE)}, got {quality!r}.")
        scale = gif_utils.QUALITY_SCALE[quality]
        panel_gap = max(0, int(round(float(self.panel_gap) * scale)))
        title_font_size = max(8, int(round(float(self.panel_title_font_size) * scale)))
        pixel_size = float(self.pixel_size) if bool(self.scale_bar_visible) else 0.0
        unit = self.pixel_unit or "A"
        panel_indices = self._animation_panel_indices()
        panel_titles = [self._panel_title_for_index(panel) for panel in panel_indices]
        frames = []
        for frame_idx in self._animation_frame_order(playback):
            panel_images = []
            for panel in panel_indices:
                data = self._get_display_panel_frame(panel, frame_idx)
                img = gif_utils.colorize(self._normalize_frame(data), self.cmap)
                panel_images.append(
                    gif_utils.finalize_frame(
                        img,
                        quality,
                        pixel_size,
                        unit,
                        show_zoom_indicator=bool(self.show_zoom_indicator),
                    )
                )
            frame_panel_titles = (
                [self._static_panel_title(panel, frame_idx) for panel in panel_indices]
                if show_frame_labels
                else panel_titles
            )
            frames.append(
                gif_utils.compose_panel_grid(
                    panel_images,
                    panel_titles=frame_panel_titles,
                    frame_labels=None,
                    show_panel_titles=bool(self.show_panel_titles),
                    title_font_size=title_font_size,
                    max_cols=int(self.max_cols),
                    panel_gap=panel_gap,
                    background=background,
                )
            )
        return frames

    def save_gif(self, path: str | pathlib.Path, *, quality: str = "high",
                 fps: float | None = None, playback: str = "forward",
                 show_frame_labels: bool = False,
                 background: str | tuple[int, int, int] = "dark") -> pathlib.Path:
        """Save the z-stack panels as an animated GIF matching the live image view.

        Each frame is colorized with the current ``cmap`` and contrast
        (``vmin`` / ``vmax`` or percentile auto-contrast, ``log_scale`` honored),
        carries per-panel scale bars when enabled, respects hidden panels and
        panel titles, and excludes FFT/profiles/controls from the export.

        Parameters
        ----------
        path : str or pathlib.Path
            Output ``.gif`` path. Parent directories are created.
        quality : {"high", "medium", "low"}, default "high"
            Spatial resolution tier (1.0 / 0.6 / 0.35). GIF is always a 256-color
            palette, so quality trades resolution for file size.
        fps : float, optional
            Playback rate. Defaults to the widget's ``fps``.
        playback : {"forward", "bounce"}, default "forward"
            Frame order. ``"forward"`` is best for time series; ``"bounce"``
            plays forward then backward without duplicating endpoint frames.
        show_frame_labels : bool, default False
            Draw per-panel dynamic frame labels from ``panel_frame_labels``.
            When enabled and no labels were provided, draw ``"i/n"``.
        background : {"dark", "black", "white"} or RGB tuple, default "dark"
            Grid gutter/background color for multi-panel exports.

        Returns
        -------
        pathlib.Path
            The written GIF path.

        Notes
        -----
        Browser zoom/pan is a view-only transform and is not reflected (the full
        frame is exported). FFT overlays and analysis panels are intentionally
        omitted so the GIF contains only the scientific image panels.
        """
        from quantem.widget.render import gif as gif_utils
        fps = float(self.fps) if fps is None else float(fps)
        frames = self._render_animation_frames(
            quality=quality,
            playback=playback,
            show_frame_labels=bool(show_frame_labels),
            background=background,
        )
        return gif_utils.write_gif(frames, path, fps)

    def save_mp4(self, path: str | pathlib.Path, *, quality: str = "high",
                 fps: float | None = None, playback: str = "forward",
                 crf: int = 18, show_frame_labels: bool = False,
                 background: str | tuple[int, int, int] = "dark") -> pathlib.Path:
        """Save the z-stack panels as an H.264 MP4.

        The rendered content matches :meth:`save_gif`: image panels only, with
        current colormap/contrast, panel titles, frame labels, hidden panels,
        and scale bars. FFT/profiles/controls are omitted.

        Parameters
        ----------
        path : str or pathlib.Path
            Output ``.mp4`` path. Parent directories are created.
        quality : {"high", "medium", "low"}, default "high"
            Spatial resolution tier (1.0 / 0.6 / 0.35).
        fps : float, optional
            Playback rate. Defaults to the widget's ``fps``.
        playback : {"forward", "bounce"}, default "forward"
            Frame order.
        crf : int, default 18
            x264 constant-rate-factor. Lower values are larger and higher
            quality; 18 is visually high quality.
        show_frame_labels : bool, default False
            Draw per-panel dynamic frame labels from ``panel_frame_labels``.
            When enabled and no labels were provided, draw ``"i/n"``.
        background : {"dark", "black", "white"} or RGB tuple, default "dark"
            Grid gutter/background color for multi-panel exports.

        Returns
        -------
        pathlib.Path
            The written MP4 path.
        """
        from quantem.widget.render import gif as gif_utils
        fps = float(self.fps) if fps is None else float(fps)
        frames = self._render_animation_frames(
            quality=quality,
            playback=playback,
            show_frame_labels=bool(show_frame_labels),
            background=background,
        )
        return gif_utils.write_mp4(frames, path, fps, crf=crf)

    def save_animation_preview(
        self,
        directory: str | pathlib.Path,
        *,
        stem: str | None = None,
        formats: Sequence[str] = ("gif", "mp4"),
        quality: str = "medium",
        fps: float | None = None,
        playback: str = "forward",
        show_frame_labels: bool = False,
        background: str | tuple[int, int, int] = "dark",
    ) -> AnimationExportPreview:
        """Save GIF/MP4 animation exports and return a notebook preview.

        This is a convenience wrapper around :meth:`save_gif` and
        :meth:`save_mp4` for notebooks. It writes one or more animation files
        and returns an object with a rich HTML representation showing the media
        previews and file sizes.

        Parameters
        ----------
        directory : str or pathlib.Path
            Output directory. Created if missing.
        stem : str, optional
            Filename stem. Defaults to a slug derived from ``title``.
        formats : sequence of {"gif", "mp4"}, default ("gif", "mp4")
            Animation formats to write.
        quality : {"low", "medium", "high"}, default "medium"
            Export quality. MP4 also maps this to the same CRF values used by
            the GUI export menu.
        fps : float, optional
            Playback rate. Defaults to the widget's ``fps``.
        playback : {"forward", "bounce"}, default "forward"
            Frame order.
        show_frame_labels : bool, default False
            Draw per-panel dynamic frame labels from ``panel_frame_labels``.
        background : {"dark", "black", "white"} or RGB tuple, default "dark"
            Grid gutter/background color for multi-panel exports.

        Returns
        -------
        AnimationExportPreview
            Notebook-displayable preview object keyed by format label.
        """
        quality = self._normalise_animation_quality(quality)
        directory = pathlib.Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if stem is None:
            stem = self._default_animation_export_path("gif", quality).stem
        paths: dict[str, pathlib.Path] = {}
        for fmt in formats:
            mode = str(fmt).lower().lstrip(".")
            if mode == "gif":
                path = directory / f"{stem}.gif"
                paths[f"GIF {quality}"] = self.save_gif(
                    path,
                    quality=quality,
                    fps=fps,
                    playback=playback,
                    show_frame_labels=show_frame_labels,
                    background=background,
                )
            elif mode == "mp4":
                path = directory / f"{stem}.mp4"
                paths[f"MP4 {quality}"] = self.save_mp4(
                    path,
                    quality=quality,
                    fps=fps,
                    playback=playback,
                    crf=self._mp4_crf_for_quality(quality),
                    show_frame_labels=show_frame_labels,
                    background=background,
                )
            else:
                raise ValueError("formats must contain only 'gif' and/or 'mp4'")
        return AnimationExportPreview(paths)

    def free(self) -> None:
        """Release VRAM and RAM held by this widget.

        Drops the numpy stack, the torch view (if any), the binned display
        copy, synced frame / ROI / prefetch bytes, and cancels any in-flight
        ROI debounce timer.
        When applicable, also flushes the CuPy memory pool and the FFT plan
        cache (in case a torch tensor was a view into CuPy memory), and
        releases the active torch device cache (``mps`` or ``cuda``).

        ``del widget`` alone does NOT free memory: traitlets installs strong
        observer references that pin the widget's refcount until ``free`` is
        called.

        Returns
        -------
        None
            Mutates the widget in place. After this call, frame data is gone
            and rendering will be blank; rebuild a new widget for further use.

        Example
        -------
        >>> from quantem.widget import Show3D
        >>> w = Show3D(stack)  # doctest: +SKIP
        >>> w.free()  # doctest: +SKIP

        Notes
        -----
        - Idempotent: calling ``free`` twice is a no-op.
        - After ``free``, ``set_image`` cannot restore the widget (the
          observers are still pinned but ``_data`` is ``None``); construct a
          new ``Show3D`` if you need to re-display something.
        """
        self.stop_folder_watch()
        self._stop_frame_server()
        if self._data is None:
            return
        # Stop frontend playback FIRST so JS rAF loop tears down before we
        # null out the byte buffers it's reading from. Without this, free()
        # mid-play leaves the JS rAF rendering from a stale Float32Array
        # view into freed memory.
        self.playing = False
        # Cancel pending ROI debounce so its callback can't fire post-free.
        if self._roi_plot_timer is not None:
            self._roi_plot_timer.cancel()
            self._roi_plot_timer = None
        device = str(self._device) if self._device is not None else ""
        self._data = None
        self._data_torch = None
        self._display_data = None
        self._separate_panel_data = None
        # Offline/export stacks are synced Bytes traits; without clearing them
        # free() leaves that RAM pinned by traitlets after _data is dropped.
        for trait in (
            "frame_bytes",
            "roi_plot_data",
            "_buffer_bytes",
            "_offline_stack",
            "_offline_float_stack",
            "export_payload",
        ):
            setattr(self, trait, b"")
        gc.collect()
        # Flush cupy pool: _data may have been a torch view into cupy memory.
        if "cupy" in sys.modules:
            import cupy
            cupy.get_default_memory_pool().free_all_blocks()
            cupy.fft.config.get_plan_cache().clear()
        if device == "mps":
            torch.mps.empty_cache()
        elif device.startswith("cuda"):
            torch.cuda.empty_cache()


    # === Observers ===

    def _on_first_render(self, change):
        """Observer for `_js_rendered=True`: prints true end-to-end construction
        timing telemetry (Python + comm + JS paint) and unobserves itself."""
        from quantem.widget._timing import format_widget_render_timing
        if not change.get("new"):
            return
        total_ms = (time.perf_counter() - self._init_t0) * 1000
        py_ms = self._init_py_elapsed_ms
        shape = (self.n_slices, self.height, self.width)
        if self.separate_panel_frames and getattr(self, "_separate_panel_data", None) is not None:
            mem = sum(int(p.nbytes) for p in self._separate_panel_data)
        else:
            mem = self._data.nbytes
        self.render_total_ms = int(total_ms)
        self.render_python_build_ms = int(py_ms)
        self.render_wire_js_ms = int(total_ms - py_ms)
        if not getattr(self, "_save_state", False):
            # The frontend has decoded and painted the initial frame/chunk by
            # the time it flips `_js_rendered`. Clear synced bulk buffers so a
            # later notebook save stores the compact static fallback, not the
            # multi-frame transfer buffers. Targeted initial sync is untouched.
            self.frame_bytes = b""
            self._buffer_bytes = b""
        if getattr(self, "_verbose", True):
            print(
                format_widget_render_timing(
                    "Show3D",
                    shape=shape,
                    dtype=str(self._data.dtype) if self._data is not None else None,
                    raw_bytes=int(mem),
                    total_ms=total_ms,
                    python_ms=py_ms,
                    wire_js_ms=total_ms - py_ms,
                    extra_rows=[
                        ("Frame server", "on" if self.frame_server_url else "off"),
                        (
                            "Offline stack",
                            "on" if bool(self._offline_stack or self._offline_float_stack) else "off",
                        ),
                    ],
                ),
                flush=True,
            )
        try:
            self.unobserve(self._on_first_render, names=["_js_rendered"])
        except (ValueError, KeyError):
            pass  # observer already removed

    def _on_diff_mode_change(self, change: dict | None = None) -> None:
        """Observer: `diff_mode` flipped. Recomputes the symmetric-around-zero
        data range so the colormap pins black at 0 (positive vs negative diff
        signal stays balanced)."""
        data = self._display_data
        if self.diff_mode == "off":
            # Restore the constructor's full-resolution data range so toggling
            # Off→Previous→Off is idempotent (computing from binned data drifts).
            self.data_min = float(getattr(self, "_data_min_off", data.min()))
            self.data_max = float(getattr(self, "_data_max_off", data.max()))
        elif self.diff_mode == "previous":
            # Vectorized diff: data[1:] - data[:-1]
            # Symmetric clamp around 0 so the all-zero baseline frame at idx=0
            # stays inside the displayed range whether diffs are positive or negative.
            if self.n_slices < 2:
                self.data_min = 0.0
                self.data_max = 0.0
            else:
                diffs = data[1:] - data[:-1]
                self.data_min = min(0.0, float(diffs.min()))
                self.data_max = max(0.0, float(diffs.max()))
        elif self.diff_mode == "first":
            if self.n_slices < 2:
                self.data_min = 0.0
                self.data_max = 0.0
            else:
                diffs = data[1:] - data[0:1]
                self.data_min = min(0.0, float(diffs.min()))
                self.data_max = max(0.0, float(diffs.max()))
        else:
            self.data_min = float(data.min())
            self.data_max = float(data.max())
        self._bump_frame_server_version()
        self._refresh_auto_contrast_ranges()
        self._update_all()

    def _on_slice_change(self, change: dict | None = None) -> None:
        """Observer: `slice_idx` changed via scrub. Skipped during playback
        (playback drives frames from the JS-side prefetch buffer). Stats are
        computed JS-side from frame_bytes; Python only ships the raw bytes."""
        if self.playing:
            return
        self._update_all()

    def _on_playing_change(self, change: dict | None = None) -> None:
        """Observer: `playing` toggled. Starts the JS animation via a sliding
        buffer chunk, or refreshes stats on the held frame on stop."""
        if self.playing:
            if self.frame_server_url:
                with self.hold_sync():
                    self._buffer_start = int(self.slice_idx)
                    self._buffer_count = 0
                    self._buffer_bytes = b""
            else:
                self._send_buffer(self.slice_idx)
        else:
            self._update_all()

    def _on_prefetch(self, change: dict | None = None) -> None:
        """Observer: JS requested the next playback chunk. Re-sends the sliding
        window so playback never starves."""
        if self.frame_server_url:
            return
        if self._prefetch_request >= 0 and self.playing:
            self._send_buffer(self._prefetch_request % self.n_slices)

    def _on_roi_change(self, change: dict | None = None) -> None:
        """Handle ROI change. Stats for current frame are instant.
        Full-stack ROI plot is debounced (500ms) to avoid UI freeze during drag."""
        if int(self.n_panels) > 1:
            self.roi_stats = {}
            self.roi_plot_data = b""
            return
        # Auto-select first ROI if the user added one programmatically and
        # roi_selected_idx is still -1 (otherwise stats stay empty silently).
        if self.roi_active and self.roi_list and self.roi_selected_idx < 0:
            self.roi_selected_idx = 0
        if self.roi_active:
            self._update_roi_stats(self._get_display_frame())
            # Debounce the expensive all-frame ROI plot
            if self._roi_plot_timer is not None:
                self._roi_plot_timer.cancel()
            self._roi_plot_timer = threading.Timer(0.5, self._compute_roi_plot)
            self._roi_plot_timer.start()
        else:
            self.roi_stats = {}
            self.roi_plot_data = b""

    def _on_export_request_change(self, change: dict) -> None:
        """Handle toolbar export requests from the live notebook frontend."""
        raw = str(change.get("new") or "")
        if not raw:
            return
        try:
            payload = json.loads(raw)
            mode = str(payload.get("mode", "exact"))
            if mode == "clear":
                self.export_payload = b""
                self.export_payload_id = ""
                self.export_filename = ""
                return
            if mode in {"gif", "mp4"}:
                quality = self._normalise_animation_quality(payload.get("quality", "medium"))
                filename = str(payload.get("filename") or self._default_animation_export_path(mode, quality).name)
                request_id = str(payload.get("id") or "")
                if payload.get("download"):
                    self.export_status = f"Preparing {filename}..."
                    media = self._animation_export_bytes(mode, quality=quality)
                    self.export_filename = filename
                    self.export_payload = media
                    self.export_payload_id = request_id
                    size_mb = len(media) / (1024 * 1024)
                    self.export_status = f"Ready {filename} ({size_mb:.1f} MB)"
                else:
                    self.export_status = f"Exporting {filename}..."
                    path = self._default_animation_export_path(mode, quality)
                    if mode == "gif":
                        self.save_gif(path, quality=quality)
                    else:
                        self.save_mp4(path, quality=quality, crf=self._mp4_crf_for_quality(quality))
                    size_mb = path.stat().st_size / (1024 * 1024)
                    self.export_status = f"Exported {path.name} ({size_mb:.1f} MB)"
                return
            quantized, downsample_factor = self._normalise_html_export_options(
                mode=mode,
                encoding=str(payload.get("encoding", "full")),
                downsample=payload.get("downsample"),
                quantized=None,
            )
            if payload.get("download"):
                filename = str(payload.get("filename") or self._default_html_export_path(
                    quantized,
                    downsample=downsample_factor,
                ).name)
                request_id = str(payload.get("id") or "")
                self.export_status = f"Preparing {filename}..."
                html = self._html_export_bytes(quantized=quantized, downsample=downsample_factor)
                self.export_filename = filename
                self.export_payload = html
                self.export_payload_id = request_id
                size_mb = len(html) / (1024 * 1024)
                label = self._export_mode_label(quantized, downsample=downsample_factor)
                self.export_status = f"Ready {filename} ({size_mb:.1f} MB, {label})"
            else:
                self.export_status = f"Exporting {mode} HTML..."
                self.export_html(quantized=quantized, downsample=downsample_factor)
        except Exception as exc:
            self.export_status = f"Export failed: {exc}"

    # === Internal primitives ===

    def _default_html_export_path(self, quantized: bool, *, downsample: int = 1) -> pathlib.Path:
        """Build a stable, human-readable export filename in the kernel cwd."""
        label = self.title.strip() or "show3d"
        slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        if not slug:
            slug = "show3d"
        mode = "quantized" if quantized else "exact"
        suffix = f"_{int(downsample)}xbin" if quantized and int(downsample) > 1 else ""
        return pathlib.Path.cwd() / f"{slug}_{self.n_slices}x{self.height}x{self.width}_{mode}{suffix}.html"

    def _default_animation_export_path(self, mode: str, quality: str = "medium") -> pathlib.Path:
        """Build a stable, human-readable GIF/MP4 export filename."""
        label = self.title.strip() or "show3d"
        slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        if not slug:
            slug = "show3d"
        ext = "mp4" if mode == "mp4" else "gif"
        return pathlib.Path.cwd() / f"{slug}_{self.n_slices}x{self.height}x{self.width}_{quality}.{ext}"

    def _normalise_animation_quality(self, quality: object) -> str:
        """Validate animation export quality from frontend requests."""
        from quantem.widget.render import gif as gif_utils
        value = str(quality or "medium").lower()
        if value not in gif_utils.QUALITY_SCALE:
            raise ValueError(
                f"animation quality must be one of {list(gif_utils.QUALITY_SCALE)}, got {quality!r}"
            )
        return value

    def _mp4_crf_for_quality(self, quality: str) -> int:
        """Map UI quality labels to H.264 compression quality."""
        return {"low": 24, "medium": 21, "high": 18}[self._normalise_animation_quality(quality)]

    def _export_mode_label(self, quantized: bool, *, downsample: int = 1) -> str:
        if not quantized:
            return "full float32"
        if int(downsample) > 1:
            return f"uint8, {int(downsample)}x binned"
        return "uint8"

    def _write_html_export(
        self,
        path: str | pathlib.Path,
        *,
        quantized: bool,
        downsample: int = 1,
        title: str | None = None,
    ) -> pathlib.Path:
        """Write a standalone HTML export without updating toolbar status."""
        from ipywidgets.embed import dependency_state, embed_minimal_html

        from .export import ensure_mobile_viewport

        export_path = pathlib.Path(path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        page_title = title or self.title or "Show3D"
        export_widget = self._clone_for_html_export(quantized=quantized, downsample=downsample)
        try:
            state = dependency_state([export_widget], drop_defaults=False)
            embed_minimal_html(
                str(export_path),
                views=[export_widget],
                title=page_title,
                drop_defaults=False,
                state=state,
            )
        finally:
            export_widget.free()
        ensure_mobile_viewport(export_path)
        return export_path

    def _html_export_bytes(self, *, quantized: bool, downsample: int = 1) -> bytes:
        """Build a standalone HTML export in a temp directory and return bytes."""
        with tempfile.TemporaryDirectory(prefix="show3d-export-") as tmp:
            path = pathlib.Path(tmp) / self._default_html_export_path(
                quantized,
                downsample=downsample,
            ).name
            self._write_html_export(path, quantized=quantized, downsample=downsample)
            return path.read_bytes()

    def _animation_export_bytes(self, mode: str, *, quality: str = "medium") -> bytes:
        """Build a GIF or MP4 animation export in a temp directory and return bytes."""
        quality = self._normalise_animation_quality(quality)
        with tempfile.TemporaryDirectory(prefix="show3d-animation-export-") as tmp:
            path = pathlib.Path(tmp) / self._default_animation_export_path(mode, quality).name
            if mode == "gif":
                self.save_gif(path, quality=quality)
            elif mode == "mp4":
                self.save_mp4(path, quality=quality, crf=self._mp4_crf_for_quality(quality))
            else:
                raise ValueError(f"unsupported animation export mode {mode!r}")
            return path.read_bytes()

    def _export_data_args(self, *, downsample: int = 1) -> tuple[np.ndarray, ...]:
        """Return display-shaped data args so exported HTML matches the widget."""
        if self._display_data is None:
            raise ValueError("Cannot export HTML after free(); rebuild the widget first.")
        downsample = int(downsample)

        def _bin_gray(panel: np.ndarray) -> np.ndarray:
            if downsample > 1:
                from quantem.widget.utils.array import bin2d
                return np.ascontiguousarray(bin2d(panel, factor=downsample, mode="mean"), dtype=np.float32)
            return np.ascontiguousarray(panel, dtype=np.float32)

        def _bin_rgb_stack(stack: np.ndarray) -> np.ndarray:
            """Spatial-bin an RGB stack ``(N, H, W, 3)`` without touching channels."""
            arr = np.ascontiguousarray(stack, dtype=np.float32)
            if downsample <= 1:
                return arr
            from quantem.widget.utils.array import bin2d
            frames = [
                np.stack(
                    [bin2d(arr[i, ..., c], factor=downsample, mode="mean") for c in range(3)],
                    axis=-1,
                )
                for i in range(arr.shape[0])
            ]
            return np.ascontiguousarray(np.stack(frames, axis=0), dtype=np.float32)

        # True-color: export the RGB stack so offline HTML paints without a colormap.
        if self.is_rgb and getattr(self, "_rgb_data", None) is not None:
            return (_bin_rgb_stack(self._rgb_data),)

        n_panels = int(self.n_panels)
        if self.shared_panel_source and n_panels > 1:
            src = _bin_gray(self._display_data)
            return tuple(src for _ in range(n_panels))
        if self._separate_panel_data is not None:
            return tuple(
                _bin_gray(panel)
                for panel in self._separate_panel_data
            )
        if n_panels > 1 and int(self.panel_width_px) > 0:
            panel_w = int(self.panel_width_px)
            if int(self.width) == panel_w * n_panels:
                return tuple(
                    _bin_gray(
                        self._display_data[:, :, i * panel_w : (i + 1) * panel_w],
                    )
                    for i in range(n_panels)
                )
        return (_bin_gray(self._display_data),)

    def _establish_rgb_luminance(self, data: np.ndarray, *, is_rgb: bool) -> np.ndarray:
        """Store the color stack and return the luminance plane that drives stats.

        Show3D keeps true color in ``_rgb_data`` (``(N, H, W, 3)``) for the
        browser to paint, and a Rec. 709 luminance plane in ``_data`` for stats,
        FFT, ROI, and scale bars. ``__init__`` and :meth:`set_image` both route
        their normalized stack through here, so the two paths cannot drift - the
        source of the fig4 RGB export bug where a live update left ``_rgb_data``
        stale and ``_data`` four-dimensional.

        Parameters
        ----------
        data : np.ndarray
            Normalized stack. ``(N, H, W, 3)`` in ``[0, 1]`` when ``is_rgb``,
            otherwise ``(N, H, W)``.
        is_rgb : bool
            Whether ``data`` is a true-color stack.

        Returns
        -------
        np.ndarray
            The luminance plane ``(N, H, W)`` to assign to ``_data``. Grayscale
            input passes through unchanged.

        Examples
        --------
        >>> luma = widget._establish_rgb_luminance(rgb_stack, is_rgb=True)  # doctest: +SKIP
        >>> widget._data = luma  # doctest: +SKIP
        """
        if is_rgb:
            self._rgb_data = data
            return np.tensordot(data, _RGB_LUMA, axes=([-1], [0])).astype(np.float32)
        self._rgb_data = None
        return data

    def _offline_stack_source(self) -> np.ndarray:
        """Return the stack shape that the offline frontend indexes per frame."""
        if self._display_data is None:
            raise ValueError("Cannot export HTML after free(); rebuild the widget first.")
        # RGB: pack (N, H, W, 3) float32 so JS can slice true-color frames offline.
        if self.is_rgb and getattr(self, "_rgb_data", None) is not None:
            return np.ascontiguousarray(self._rgb_data, dtype=np.float32)
        if self.separate_panel_frames and self._separate_panel_data is not None:
            return np.concatenate(self._separate_panel_data, axis=2)
        return np.ascontiguousarray(self._display_data, dtype=np.float32)

    def _pack_offline_u8_stack(self, offline_source: np.ndarray) -> None:
        """Quantize the display stack into ``_offline_stack`` (+ min/max traits).

        This is what kernel-less consumers (docs pages, exported HTML, saved
        widget state) slice frames from. Every path that replaces the stack
        (``__init__``, ``set_image``) must repack, or indices past the old
        stack's end render blank.

        True-color RGB stacks are ``(N, H, W, 3)`` display-ready floats in
        ``[0, 1]`` (or 0–255); they pack to uint8 RGB without a gray colormap
        range so offline HTML stays small enough for Chrome ``file://``.
        """
        arr = np.ascontiguousarray(offline_source, dtype=np.float32)
        # RGB true-color path: (N, H, W, 3)
        if getattr(self, "is_rgb", False) and arr.ndim == 4 and arr.shape[-1] == 3:
            self._offline_min = 0.0
            self._offline_max = 1.0
            self._offline_mins = []
            self._offline_maxs = []
            self._offline_float_stack = b""
            mx = float(np.nanmax(arr)) if arr.size else 1.0
            if mx > 1.5:
                quantized = np.clip(arr, 0, 255).astype(np.uint8)
            else:
                quantized = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            self._offline_stack = np.ascontiguousarray(quantized).tobytes()
            return

        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            lo, hi = 0.0, 1.0
        else:
            lo = float(finite.min())
            hi = float(finite.max())
        rng = hi - lo if hi > lo else 1.0
        self._offline_min = lo
        self._offline_max = hi
        self._offline_mins = []
        self._offline_maxs = []
        panel_count = int(self.n_panels)
        panel_w = int(self.panel_width_px) if int(self.panel_width_px) > 0 else 0
        if panel_count > 1 and panel_w > 0 and arr.ndim == 3 and arr.shape[2] == panel_w * panel_count:
            q_panels = []
            mins: list[float] = []
            maxs: list[float] = []
            for panel in range(panel_count):
                panel_arr = arr[:, :, panel * panel_w : (panel + 1) * panel_w]
                panel_finite = panel_arr[np.isfinite(panel_arr)]
                if panel_finite.size == 0:
                    p_lo, p_hi = 0.0, 1.0
                else:
                    p_lo = float(panel_finite.min())
                    p_hi = float(panel_finite.max())
                p_rng = p_hi - p_lo if p_hi > p_lo else 1.0
                q_panels.append(np.clip((panel_arr - p_lo) * (255.0 / p_rng), 0, 255).astype(np.uint8))
                mins.append(p_lo)
                maxs.append(p_hi)
            quantized = np.concatenate(q_panels, axis=2)
            self._offline_mins = mins
            self._offline_maxs = maxs
        else:
            quantized = np.clip((arr - lo) * (255.0 / rng), 0, 255).astype(np.uint8)
        self._offline_stack = quantized.tobytes()

    def _pack_exact_offline_stack(self) -> None:
        """Embed the full display stack as float32 for exact standalone HTML."""
        arr = np.ascontiguousarray(self._offline_stack_source(), dtype=np.float32)
        if arr.size:
            lo = float(arr.min())
            hi = float(arr.max())
        else:
            lo, hi = 0.0, 1.0
        self.offline = True
        self._offline_min = lo
        self._offline_max = hi
        self._offline_mins = []
        self._offline_maxs = []
        self._offline_stack = b""
        self._offline_float_stack = arr.tobytes()
        self.frame_bytes = b""
        self._buffer_bytes = b""

    def _clone_for_html_export(self, *, quantized: bool, downsample: int = 1) -> Self:
        """Create an export-only widget with current state and requested packing."""
        downsample = int(downsample)
        export_pixel_size = self.pixel_size * downsample if self.pixel_size > 0 else self.pixel_size
        clone = type(self)(
            *self._export_data_args(downsample=downsample),
            labels=list(self.labels) if self.labels else None,
            panel_titles=list(self.panel_titles) if self.panel_titles else None,
            panel_frame_labels=[list(labels) for labels in self.panel_frame_labels] if self.panel_frame_labels else None,
            frame_metadata=[dict(metadata) for metadata in self.frame_metadata] if self.frame_metadata else None,
            panel_frame_metadata=[
                [dict(metadata) for metadata in panel_metadata]
                for panel_metadata in self.panel_frame_metadata
            ] if self.panel_frame_metadata else None,
            frame_label_format=self.frame_label_format or None,
            panel_real_frames=list(self.panel_real_frames) if self.panel_real_frames else None,
            title=self.title,
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            pixel_size=export_pixel_size,
            pixel_unit=self.pixel_unit,
            smooth=self.smooth,
            image_rotation=self.image_rotation,
            log_scale=self.log_scale,
            auto_contrast=self.auto_contrast,
            image_vmin_pct=self.image_vmin_pct,
            image_vmax_pct=self.image_vmax_pct,
            percentile_low=self.percentile_low,
            percentile_high=self.percentile_high,
            fps=self.fps,
            avg_window=self.avg_window,
            timestamps=list(self.timestamps) if self.timestamps else None,
            timestamp_unit=self.timestamp_unit,
            show_title=self.show_title,
            show_fft=self.show_fft,
            fft_layout=self.fft_layout,
            fft_overlay_position=self.fft_overlay_position,
            fft_overlay_size=self.fft_overlay_size,
            fft_overlay_zoom=self.fft_overlay_zoom,
            fft_window=self.fft_window,
            fft_metrics=self.fft_metrics,
            show_stats=self.show_stats,
            show_controls=self.show_controls,
            controls_collapsed=self.controls_collapsed,
            debug=self.debug,
            size=self.size,
            diff_mode=self.diff_mode,
            buffer_size=getattr(self, "_buffer_size", 64),
            dim_label=self.dim_label,
            use_torch=False,
            display_bin=1,
            offline=quantized,
            max_cols=self.max_cols,
            panel_gap=self.panel_gap,
            panel_title_font_size=self.panel_title_font_size,
            show_panel_titles=self.show_panel_titles,
            show_resize_handles=self.show_resize_handles,
            show_zoom_indicator=self.show_zoom_indicator,
            show_scale_bar=self.scale_bar_visible,
        )
        clone.n_pages = int(self.n_pages)
        clone.panels_per_page = int(self.panels_per_page)
        clone.page_labels = list(self.page_labels)
        clone.page_starred = list(self.page_starred)
        clone.page_idx = int(self.page_idx)
        clone.load_state_dict(self.state_dict())
        if downsample > 1 and clone.pixel_size > 0:
            clone.pixel_size = export_pixel_size
        if quantized:
            clone._offline_float_stack = b""
        else:
            clone._pack_exact_offline_stack()
        clone.playing = bool(self.playing)
        clone.export_enabled = False
        clone.export_status = ""
        clone.export_payload = b""
        clone.export_payload_id = ""
        clone.export_filename = ""
        clone.handoff_enabled = False
        clone.handoff_status = ""
        clone.handoff_request = ""
        clone.prepared_view = None
        clone.prepared_view_widget = None
        clone._stop_frame_server()
        clone.frame_server_url = ""
        clone._buffer_bytes = b""
        clone._export_light = True
        clone._save_state = True  # export clones must embed the offline stack;
        # get_state() drops _offline_*_stack when _save_state is False (notebook-metadata guard).
        return clone

    def _start_frame_server(self) -> None:
        """Start the localhost exact-frame endpoint used by browser playback."""
        if getattr(self, "_frame_server", None) is not None:
            return

        widget_ref = weakref.ref(self)
        token = self._frame_server_token

        class FrameHandler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt: str, *args) -> None:  # noqa: D401
                return

            def _cors(self) -> None:
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type, Range")
                self.send_header("Cache-Control", "no-store")

            def _text(self, status: int, message: str) -> None:
                body = message.encode("utf-8")
                self.send_response(status)
                self._cors()
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            def do_OPTIONS(self) -> None:
                self.send_response(204)
                self._cors()
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self) -> None:
                parsed = urllib.parse.urlsplit(self.path)
                if parsed.path != "/frame":
                    self._text(404, "not found")
                    return
                params = urllib.parse.parse_qs(parsed.query)
                if params.get("token", [""])[0] != token:
                    self._text(403, "forbidden")
                    return
                try:
                    idx = int(params.get("idx", [""])[0])
                except ValueError:
                    self._text(400, "idx must be an integer")
                    return
                panel_param = params.get("panel", [None])[0]
                try:
                    panel = int(panel_param) if panel_param is not None else None
                except ValueError:
                    self._text(400, "panel must be an integer")
                    return
                version_param = params.get("version", [None])[0]
                try:
                    version = int(version_param) if version_param is not None else None
                except ValueError:
                    self._text(400, "version must be an integer")
                    return

                widget = widget_ref()
                if widget is None:
                    self._text(410, "widget is gone")
                    return
                status, frame_or_message = widget._frame_for_http(idx, version, panel)
                if status != 200:
                    self._text(status, str(frame_or_message))
                    return

                frame = frame_or_message
                view = memoryview(frame).cast("B")
                self.send_response(200)
                self._cors()
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(view)))
                self.send_header("X-Frame-Index", str(idx))
                self.send_header("X-Frame-Shape", f"{frame.shape[0]},{frame.shape[1]}")
                self.end_headers()
                try:
                    self.wfile.write(view)
                except (BrokenPipeError, ConnectionResetError):
                    return

        try:
            server = _Show3DFrameHTTPServer(("127.0.0.1", 0), FrameHandler)
        except OSError as exc:
            warnings.warn(f"Show3D frame server disabled: {exc}", RuntimeWarning, stacklevel=2)
            self.frame_server_url = ""
            return

        thread = threading.Thread(
            target=server.serve_forever,
            name=f"Show3DFrameServer-{id(self):x}",
            daemon=True,
        )
        thread.start()
        self._frame_server = server
        self._frame_server_thread = thread
        host, port = server.server_address[:2]
        quoted_token = urllib.parse.quote(self._frame_server_token, safe="")
        self.frame_server_url = f"http://{host}:{port}/frame?token={quoted_token}"
        self._bump_frame_server_version()

    def _stop_frame_server(self) -> None:
        """Stop the localhost frame endpoint."""
        server = getattr(self, "_frame_server", None)
        thread = getattr(self, "_frame_server_thread", None)
        self._frame_server = None
        self._frame_server_thread = None
        if self.frame_server_url:
            self.frame_server_url = ""
            self._bump_frame_server_version()
        if server is None:
            return
        try:
            server.shutdown()
        finally:
            server.server_close()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def _bump_frame_server_version(self) -> None:
        self.frame_server_version = int(self.frame_server_version) + 1

    def _frame_for_http(self, idx: int, version: int | None, panel: int | None = None) -> tuple[int, np.ndarray | str]:
        if version is not None and version != self.frame_server_version:
            return 409, "stale frame server version"
        data = self._display_data
        if data is None:
            return 410, "frame data has been released"
        if idx < 0 or idx >= int(self.n_slices):
            return 416, f"frame index {idx} out of range [0, {self.n_slices})"
        if panel is not None:
            if not self.separate_panel_frames:
                return 400, "panel query is only valid for separate-panel frames"
            if panel < 0 or panel >= int(self.n_panels):
                return 416, f"panel index {panel} out of range [0, {self.n_panels})"
            frame = np.asarray(self._get_display_panel_frame(panel, idx), dtype=np.float32)
        else:
            frame = np.asarray(self._get_display_frame(idx), dtype=np.float32)
        frame = self._wire_frame(frame, cache_key=("http", idx, panel, self.diff_mode))
        if not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame)
        return 200, frame

    def _get_display_panel_frame(self, panel: int, idx: int) -> np.ndarray:
        panels = getattr(self, "_separate_panel_data", None)
        if not self.separate_panel_frames or panels is None:
            frame = self._get_display_frame(idx)
            pw = int(self.panel_width_px) or frame.shape[1] // max(1, int(self.n_panels))
            return frame[:, panel * pw : (panel + 1) * pw]
        frame = panels[panel][idx]
        if self.diff_mode == "previous":
            if idx == 0:
                return np.zeros_like(frame)
            return frame - panels[panel][idx - 1]
        if self.diff_mode == "first":
            return frame - panels[panel][0]
        return frame

    def _get_display_frame(self, idx: int | None = None) -> np.ndarray:
        """Return the (binned, possibly diff-mode) frame for display. idx=None uses
        current slice_idx. Applies `diff_mode` subtraction so the FFT / stats
        observers see the same data the browser renders."""
        if idx is None:
            idx = self.slice_idx
        if self.separate_panel_frames and getattr(self, "_separate_panel_data", None) is not None:
            return np.concatenate(
                [self._get_display_panel_frame(panel, idx) for panel in range(int(self.n_panels))],
                axis=1,
            )
        data = self._display_data
        frame = data[idx]
        if self.diff_mode == "previous":
            if idx == 0:
                return np.zeros_like(frame)
            return frame - data[idx - 1]
        if self.diff_mode == "first":
            return frame - data[0]
        return frame

    def _refresh_auto_contrast_ranges(self) -> None:
        """Precompute one stack-level auto-contrast range for JS playback.

        Show3D is a scrubber, so Auto should give a stable intensity mapping
        across frames and panels. The synced lists still have one entry per
        slice for the existing JS cache contract, but every entry carries the
        same stack percentile range.
        """
        if self.n_slices <= 0:
            self.auto_vmins = []
            self.auto_vmaxs = []
            return
        if not self.auto_contrast:
            self.auto_vmins = []
            self.auto_vmaxs = []
            return

        low_target = self.percentile_low / 100.0
        high_target = self.percentile_high / 100.0
        bins = 1024
        denom = bins - 1
        separate_panels = self.separate_panel_frames and getattr(self, "_separate_panel_data", None) is not None
        mn = float("inf")
        mx = float("-inf")
        total_size = 0
        for i in range(self.n_slices):
            if separate_panels:
                panel_frames = [self._get_display_panel_frame(panel, i) for panel in range(int(self.n_panels))]
                mn = min(mn, min(float(np.min(frame)) for frame in panel_frames))
                mx = max(mx, max(float(np.max(frame)) for frame in panel_frames))
                total_size += sum(int(frame.size) for frame in panel_frames)
            else:
                frame = self._get_display_frame(i)
                mn = min(mn, float(np.min(frame)))
                mx = max(mx, float(np.max(frame)))
                total_size += int(frame.size)
        if total_size <= 0 or not math.isfinite(mn) or not math.isfinite(mx):
            self.auto_vmins = []
            self.auto_vmaxs = []
            return
        if mn == mx:
            self.auto_vmins = [mn] * int(self.n_slices)
            self.auto_vmaxs = [mx] * int(self.n_slices)
            return

        hist = np.zeros(bins, dtype=np.int64)
        for i in range(self.n_slices):
            if separate_panels:
                for panel in range(int(self.n_panels)):
                    frame = self._get_display_panel_frame(panel, i)
                    panel_hist, _ = np.histogram(frame, bins=bins, range=(mn, mx))
                    hist += panel_hist
            else:
                frame = self._get_display_frame(i)
                frame_hist, _ = np.histogram(frame, bins=bins, range=(mn, mx))
                hist += frame_hist

        csum = np.cumsum(hist)
        low_count = int(total_size * low_target)
        high_count = int(np.ceil(total_size * high_target))
        lo = int(np.searchsorted(csum, low_count, side="left"))
        hi = int(np.searchsorted(csum, high_count, side="left"))
        lo = max(0, min(denom, lo))
        hi = max(0, min(denom, hi))
        span = mx - mn
        vmin = float(mn + (lo / denom) * span)
        vmax = float(mn + (hi / denom) * span)
        vmins = [vmin] * int(self.n_slices)
        vmaxs = [vmax] * int(self.n_slices)

        with self.hold_sync():
            self.auto_vmins = vmins
            self.auto_vmaxs = vmaxs

    # Traits that carry the bulk pixel payload. Dropped from the saved-notebook
    # snapshot when save_state is False so a plain display stays a few MB, not GB.
    _UNSAVED_HEAVY_KEYS = (
        "frame_bytes",
        "_buffer_bytes",
        "_offline_stack",
        "_offline_float_stack",
        "export_payload",
    )

    def get_state(self, key=None, drop_defaults=False):
        """Trait state for comm sync and notebook embedding.

        ipywidgets calls this with ``key=None`` to snapshot the FULL state that
        gets written into the saved notebook's ``metadata.widgets``. When
        ``save_state`` is False we drop the heavy frame buffers from that
        snapshot, so a plain Show3D does not bake a hundreds-of-MB z-stack into
        the .ipynb. Targeted syncs (``key`` is a name or set, used by hold_sync /
        send_state during live streaming) are untouched, so the frontend still
        receives ``frame_bytes`` / ``_buffer_bytes`` normally. ``save_state=True``
        embeds everything so a reopened notebook restores the interactive widget
        without a kernel.
        """
        state = super().get_state(key=key, drop_defaults=drop_defaults)
        if (
            key is None
            and not getattr(self, "_save_state", False)
            and not getattr(self, "_initial_live_mount_state", False)
        ):
            if not self._static_fallback_enabled():
                state.pop("_static_fallback_jpeg", None)
                state.pop("_static_fallback_mime", None)
            elif not self._static_fallback_jpeg:
                png = self._static_fallback_png_b64()
                if png:
                    self._store_static_fallback_preview(png)
                    state["_static_fallback_jpeg"] = self._static_fallback_jpeg
                    state["_static_fallback_mime"] = self._static_fallback_mime
            for heavy_key in self._UNSAVED_HEAVY_KEYS:
                state.pop(heavy_key, None)
        if (
            key is None
            and not getattr(self, "_initial_live_mount_state", False)
            and state.get("folder_watch_state") in {
            "watching",
            "updating",
            "waiting",
            "error",
            }
        ):
            # Saved widget state retains pixels, not the Python watcher thread.
            # Keep snapshots truthful even when exported while acquisition is
            # still running in the current kernel.
            state["folder_watch_state"] = "stopped"
            state["folder_watch_detail"] = (
                "Folder watcher is not running in saved widget state. "
                "Re-run the cell to resume live folder updates."
            )
            if state.get("folder_waiting"):
                state["folder_status"] = (
                    "No completed frame was captured in this saved widget. "
                    "Re-run the cell to resume folder watching."
                )
        return state

    def _with_initial_live_mount_state(self, fn, *args, **kwargs):
        """Expose the first live frame and watcher state during display."""
        self._initial_live_mount_state = True
        try:
            return fn(*args, **kwargs)
        finally:
            self._initial_live_mount_state = False

    def _repr_mimebundle_(self, **kwargs):
        return self._with_initial_live_mount_state(
            super()._repr_mimebundle_,
            **kwargs,
        )

    def _ipython_display_(self):
        return self._with_initial_live_mount_state(super()._ipython_display_)

    def _static_panel_frame_label(self, panel: int, idx: int) -> str:
        """Return the dynamic frame label shown in the live panel title."""
        panel_labels = self.panel_frame_labels[panel] if panel < len(self.panel_frame_labels) else []
        panel_real = self.panel_real_frames[panel] if panel < len(self.panel_real_frames) else 0
        panel_idx = min(idx, max(0, int(panel_real) - 1)) if panel_real else idx
        if panel_idx < len(panel_labels):
            text = str(panel_labels[panel_idx]).strip()
            if text and text not in {str(panel_idx), str(panel_idx + 1)}:
                return text
        if idx < len(self.labels):
            text = str(self.labels[idx]).strip()
            if text and text not in {str(idx), str(idx + 1)}:
                return text
        return ""

    def _static_panel_title(self, panel: int, idx: int) -> str:
        """Live-canvas panel title: title, optional frame label, and count."""
        title_text = (
            self._panel_title_for_index(panel)
            if int(self.n_panels) > 1 or (panel < len(self.panel_titles) and self.panel_titles[panel])
            else (self.title or self._panel_title_for_index(panel))
        )
        frame_label = self._static_panel_frame_label(panel, idx)
        panel_real = self.panel_real_frames[panel] if panel < len(self.panel_real_frames) else 0
        shown = min(idx + 1, int(panel_real)) if panel_real else idx + 1
        total = int(panel_real) if panel_real else int(self.n_slices)
        return f"{title_text}{f' · {frame_label}' if frame_label else ''} {shown}/{total}"

    def _static_overlay_texts(
        self,
        panels: list[int],
        idx: int,
        *,
        css_px: float | None = None,
    ) -> list[tuple[str, str, str, float]]:
        """Per-panel static overlays, delegated to Show2D for exact geometry."""
        preview = self._static_show2d_preview(panels=panels, idx=idx)
        if preview is None:
            return []
        return preview._static_overlay_texts(css_px=css_px)

    def _static_show2d_preview(
        self,
        *,
        panels: list[int] | None = None,
        idx: int | None = None,
    ):
        """Build the Show2D static-render proxy for the current Show3D frame."""
        if (
            self.n_slices <= 0
            or self.n_panels <= 0
            or not hasattr(self, "_display_data")
            or getattr(self, "_display_data", None) is None
        ):
            return None
        from quantem.widget.show2d import Show2D

        if panels is None:
            panels = self.visible_panels
        else:
            hidden = {int(i) for i in getattr(self, "hidden_panels", [])}
            panels = [int(p) for p in panels if int(p) not in hidden]
        if not panels:
            panels = [0]
        if idx is None:
            idx = int(self.slice_idx)
        idx = max(0, min(int(idx), int(self.n_slices) - 1))
        # True color: hand Show2D the RGB frame so the cold-reopen preview shows
        # real color, not a colormapped luminance plane. RGB is single-panel.
        if self.is_rgb and self._rgb_data is not None:
            frames = [np.ascontiguousarray(self._rgb_data[idx], dtype=np.float32)]
        else:
            frames = [np.asarray(self._get_display_panel_frame(panel, idx), dtype=np.float32) for panel in panels]
        labels = [self._static_panel_title(panel, idx) for panel in panels]
        if not frames:
            return None

        vmin = None
        vmax = None
        if any(v is not None for v in self.vmin_per_panel):
            vmin = [
                self.vmin_per_panel[p] if p < len(self.vmin_per_panel) else None
                for p in panels
            ]
        elif self.vmin is not None:
            vmin = self.vmin
        if any(v is not None for v in self.vmax_per_panel):
            vmax = [
                self.vmax_per_panel[p] if p < len(self.vmax_per_panel) else None
                for p in panels
            ]
        elif self.vmax is not None:
            vmax = self.vmax

        return Show2D(
            frames,
            labels=labels,
            title=self.title,
            cmap=self.cmap,
            sampling=self.pixel_size if self.pixel_size > 0 else None,
            units=self.pixel_unit,
            scale_bar_visible=self.scale_bar_visible,
            show_fft=False,
            show_controls=False,
            show_stats=False,
            verbose=False,
            log_scale=self.log_scale,
            auto_contrast=self.auto_contrast,
            vmin=vmin,
            vmax=vmax,
            ncols=max(1, min(int(self.max_cols) if int(self.max_cols) > 0 else len(frames), len(frames))),
            size=int(self.size or 0),
            smooth=self.smooth,
            zoom=1.0,
            link_contrast=self.link_contrast,
            display_bin=1,
            show_panel_titles=self.show_panel_titles,
            panel_title_font_size=int(self.panel_title_font_size or 11),
            gallery_gap_px=max(0, int(self.panel_gap)),
            save_state=False,
        )

    def _static_png_b64(self, *, max_px: int = 512, dpi: int = 160) -> str | None:
        """Base64 PNG of the current Show3D frame using Show2D's renderer.

        Show3D's saved-notebook preview is a current-frame gallery. It delegates
        to ``Show2D._static_png_b64`` so labels, gutters, fonts, colormap,
        zoom text, and scale bars remain pixel-aligned with Show2D.
        """
        preview = self._static_show2d_preview()
        return None if preview is None else preview._static_png_b64(max_px=max_px, dpi=dpi)

    # _repr_mimebundle_ / _ipython_display_ / static-fallback sibling plumbing
    # comes from StaticFallbackMixin (utils/static_fallback.py); this class only
    # supplies _static_png_b64 above.

    def _store_static_fallback_preview(self, png_b64: str) -> None:
        """Store a compact saved-notebook preview inside lightweight state."""
        if getattr(self, "_save_state", False):
            return
        encoded = self._encode_static_fallback_b64(png_b64)
        if not encoded:
            self._static_fallback_jpeg = ""
            self._static_fallback_mime = ""
            return
        mime, image_b64 = encoded
        self._static_fallback_jpeg = image_b64
        self._static_fallback_mime = mime

    def _get_color_range(self, frame: np.ndarray) -> tuple[float, float]:
        """Get vmin/vmax based on current settings."""
        if self.vmin is not None or self.vmax is not None:
            vmin = float(self.vmin if self.vmin is not None else self._vmin)
            vmax = float(self.vmax if self.vmax is not None else self._vmax)
            if self.log_scale:
                # Signed log so negative vmin (e.g. diff_mode) doesn't collapse to 0.
                vmin = float(np.sign(vmin) * np.log1p(abs(vmin)))
                vmax = float(np.sign(vmax) * np.log1p(abs(vmax)))
        elif self.auto_contrast:
            vmin = float(np.percentile(frame, self.percentile_low))
            vmax = float(np.percentile(frame, self.percentile_high))
        else:
            vmin = self._vmin
            vmax = self._vmax
        return vmin, vmax

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Normalize frame to uint8 with current display settings."""
        # Signed log so negatives don't collapse to zero. Matches JS `slog` for
        # signed data (diff_mode, phase, residuals, anything that can go negative).
        if self.log_scale:
            frame = np.sign(frame) * np.log1p(np.abs(frame))

        vmin, vmax = self._get_color_range(frame)

        if vmax > vmin:
            normalized = np.clip((frame - vmin) / (vmax - vmin) * 255, 0, 255)
            return normalized.astype(np.uint8)
        return np.zeros(frame.shape, dtype=np.uint8)

    def _update_all(self) -> None:
        """Ship frame_bytes to JS. Stats (mean/min/max/std/histogram) are
        recomputed entirely on the JS side from the float32 bytes - Python
        doing the same reductions wastes ~50 ms per scrub at 4k full-res.
        ROI stats stay on Python because the mask logic + per-ROI bookkeeping
        lives here. In offline mode JS slices from _offline_stack directly,
        so frame_bytes is dead weight in the HTML state - skip the write."""
        display_frame = None
        if self.roi_active or not self.separate_panel_frames:
            display_frame = self._get_display_frame()
        with self.hold_sync():
            if self.roi_active and display_frame is not None:
                self._update_roi_stats(display_frame)
            else:
                self.roi_stats = {}
            if not self.offline and not self.separate_panel_frames:
                if self.is_rgb and self._rgb_data is not None:
                    # Ship display-ready RGB float32 (H, W, 3) so JS paints true color
                    # without applying a colormap.
                    idx = int(self.slice_idx) if hasattr(self, "slice_idx") else 0
                    idx = max(0, min(idx, int(self._rgb_data.shape[0]) - 1))
                    rgb_frame = np.ascontiguousarray(self._rgb_data[idx], dtype=np.float32)
                    self.frame_bytes = rgb_frame.tobytes()
                else:
                    wire = self._wire_frame(
                        display_frame, cache_key=("buf", int(self.slice_idx), self.diff_mode)
                    )
                    self.frame_bytes = wire.tobytes()
            self.frame_seq = self.frame_seq + 1

    def _update_roi_stats(self, frame: np.ndarray) -> None:
        """Compute mean/min/max/std/area of the currently-selected ROI on `frame`
        and write to the `roi_stats` trait so the JS panel can render. Skipped
        when no ROI selected (cheap fast-path for the common scrub case)."""
        idx = self.roi_selected_idx
        if idx < 0 or idx >= len(self.roi_list):
            self.roi_stats = {}
            return
        roi = self.roi_list[idx]
        mask = self._roi_mask(roi)
        # Mask is built at display (binned) dims, matching `frame`. The torch path
        # used to index raw _data_torch[slice_idx] which is full-res → shape mismatch.
        # Stats on 16 MB binned numpy frame are <5 ms; no torch round-trip needed.
        region = frame[mask]
        if region.size > 0:
            self.roi_stats = {
                "mean": float(region.mean()),
                "min": float(region.min()),
                "max": float(region.max()),
                "std": float(region.std()),
            }
        else:
            self.roi_stats = {}

    def _compute_roi_plot(self) -> None:
        """Compute selected ROI mean for all frames. Uses display data (binned) for speed."""
        idx = self.roi_selected_idx
        if idx < 0 or idx >= len(self.roi_list):
            self.roi_plot_data = b""
            return
        mask = self._roi_mask(self.roi_list[idx])
        if mask.sum() == 0:
            self.roi_plot_data = b""
            return
        # Use _display_data (binned) - 4-16× less data than _data, same ROI result.
        # Cache torch view of _display_data on the instance so every drag doesn't
        # reallocate VRAM (was leaking ~4 GB/drag on large stacks).
        # Apply diff_mode so plot matches what the stats panel shows.
        data = self._display_data
        if self.diff_mode == "previous":
            diff = np.zeros_like(data)
            diff[1:] = data[1:] - data[:-1]
            data = diff
        elif self.diff_mode == "first":
            data = data - data[0:1]
        if self._use_torch and self.diff_mode == "off":
            if getattr(self, "_display_torch", None) is None:
                self._display_torch = torch.from_numpy(self._display_data).to(self._device)
            mask_t = torch.from_numpy(mask).to(self._device)
            masked = self._display_torch[:, mask_t]
            means = masked.mean(dim=1).cpu().numpy().astype(np.float32)
        else:
            means = np.array([float(data[i][mask].mean()) for i in range(self.n_slices)], dtype=np.float32)
        self.roi_plot_data = means.tobytes()

    def _roi_mask(self, roi: dict) -> np.ndarray:
        """Return a boolean mask of the ROI over the current frame. Supports
        circle / square / rectangle / annular shapes. Strict-< edge match the
        JS strokeRect drawing so Python stats agree with what the user sees."""
        r, c = np.ogrid[0 : self.height, 0 : self.width]
        shape = roi.get("shape", "circle")
        row = float(roi.get("row", 0))
        col = float(roi.get("col", 0))
        radius = max(1.0, float(roi.get("radius", 10)))
        if shape == "circle":
            return (c - col) ** 2 + (r - row) ** 2 <= radius**2
        if shape == "square":
            # Strict < to match JS strokeRect width = 2*radius (exclusive).
            return (np.abs(c - col) < radius) & (np.abs(r - row) < radius)
        if shape == "rectangle":
            half_w = max(1.0, float(roi.get("width", 20)) / 2.0)
            half_h = max(1.0, float(roi.get("height", 20)) / 2.0)
            # Strict < to match JS strokeRect (width=width, exclusive at edge).
            return (np.abs(c - col) < half_w) & (np.abs(r - row) < half_h)
        if shape == "annular":
            inner = max(0.0, float(roi.get("radius_inner", 5)))
            dist2 = (c - col) ** 2 + (r - row) ** 2
            return (dist2 >= inner**2) & (dist2 <= radius**2)
        return (c - col) ** 2 + (r - row) ** 2 <= radius**2

    def _upsert_selected_roi(self, updates: dict) -> None:
        """Merge `updates` into the currently-selected ROI, or append a new ROI
        if none is selected. Fills defaults + cyclic color so single-field edits
        (e.g. `radius=20`) don't strip unspecified fields."""
        if int(self.n_panels) > 1:
            raise ValueError("Show3D ROI tools are only available for single-panel widgets.")
        rois = list(self.roi_list)
        color_cycle = ["#4fc3f7", "#81c784", "#ffb74d", "#ce93d8", "#ef5350", "#ffd54f", "#90a4ae", "#a1887f"]
        defaults = {
            "shape": "circle",
            "row": int(self.height // 2),
            "col": int(self.width // 2),
            "radius": 10,
            "radius_inner": 5,
            "width": 20,
            "height": 20,
            "line_width": 2,
            "highlight": False,
            "visible": True,
            "locked": False,
        }
        if self.roi_selected_idx >= 0 and self.roi_selected_idx < len(rois):
            current = {**defaults, **rois[self.roi_selected_idx]}
            if not current.get("color"):
                current["color"] = color_cycle[self.roi_selected_idx % len(color_cycle)]
            rois[self.roi_selected_idx] = {**current, **updates}
        else:
            rois.append({**defaults, "color": color_cycle[len(rois) % len(color_cycle)], **updates})
            self.roi_selected_idx = len(rois) - 1
        self.roi_list = rois
        self.roi_active = True

    @traitlets.validate("denoise")
    def _validate_display_filter(self, proposal: dict) -> str:
        """Normalize and reject unknown display filter modes early."""
        from quantem.widget.utils.display_filter import DISPLAY_FILTER_MODES, _normalize_mode

        mode = _normalize_mode(proposal["value"])
        if mode != "none" and mode not in DISPLAY_FILTER_MODES:
            raise traitlets.TraitError(
                "denoise must be one of "
                + "|".join(DISPLAY_FILTER_MODES)
                + f" (or 'off'/'raw'); got {proposal['value']!r}"
            )
        if mode != "none" and bool(getattr(self, "is_rgb", False)):
            raise traitlets.TraitError(
                "Show3D RGB stacks are display-ready true color and do not support "
                "denoise; use denoise='none'."
            )
        return mode

    @traitlets.validate("denoise_bin")
    def _validate_spatial_bin(self, proposal: dict) -> int:
        value = int(proposal["value"])
        if value not in (1, 2, 4):
            raise traitlets.TraitError(f"denoise_bin must be 1, 2, or 4; got {value}")
        if value != 1 and bool(getattr(self, "is_rgb", False)):
            raise traitlets.TraitError(
                "Show3D RGB stacks are display-ready true color and do not support "
                "denoise_bin; use denoise_bin=1."
            )
        return value

    @traitlets.observe("denoise", "denoise_sigma", "denoise_bin")
    def _on_display_filter_change(self, change: dict) -> None:
        """Invalidate the filtered-frame cache and resend the view (no disk I/O)."""
        if not getattr(self, "_display_filter_ready", False):
            return
        if change["name"] == "denoise":
            from quantem.widget.utils.display_filter import resolve_denoise_mode

            mode, extra_bin = resolve_denoise_mode(str(change["new"]))
            if mode != str(change["new"]):
                # Compound alias: rewrite to the canonical (mode, bin) pair;
                # each set re-enters this observer with a canonical value.
                self.denoise = mode
                if extra_bin > 1:
                    self.denoise_bin = max(int(self.denoise_bin), extra_bin)
                return
        self._display_filter_cache = {}
        self._refresh_display_filter_banner(announce=True)
        if self.offline:
            self._pack_offline_u8_stack(self._offline_pack_source())
        self.frame_server_version = int(self.frame_server_version) + 1
        self._send_buffer(int(self._buffer_start))
        self._update_all()

    def _display_filter_active(self) -> bool:
        from quantem.widget.utils.display_filter import _normalize_mode

        if bool(getattr(self, "is_rgb", False)):
            return False
        return _normalize_mode(self.denoise) != "none" or int(self.denoise_bin) > 1

    def _refresh_display_filter_banner(self, *, announce: bool) -> None:
        """Sync the one-line reduction notice; print it when it changes.

        Announcing an active reduction is a house rule: the user must always
        know the view is filtered and that ``denoise='none'`` restores
        raw counts.
        """
        from quantem.widget.utils.display_filter import format_display_filter_banner

        banner = (
            ""
            if bool(getattr(self, "is_rgb", False))
            else format_display_filter_banner(
                self.denoise, float(self.denoise_sigma), int(self.denoise_bin)
            )
        )
        changed = banner != self.denoise_banner
        self.denoise_banner = banner
        if announce and banner and changed:
            print(banner)

    def _offline_pack_source(self) -> np.ndarray:
        """Offline pack source with the display filter applied per panel frame.

        Separate panels are filtered before the horizontal concat so the
        filter never bleeds across panel boundaries.
        """
        # True-color stacks pack RGB, not the luminance plane used for stats.
        if self.is_rgb and getattr(self, "_rgb_data", None) is not None:
            return np.ascontiguousarray(self._rgb_data, dtype=np.float32)
        if self.separate_panel_frames and self._separate_panel_data is not None:
            panels = self._separate_panel_data
            if self._display_filter_active():
                panels = [
                    np.stack(
                        [self._wire_frame(stack[k]) for k in range(int(stack.shape[0]))],
                        axis=0,
                    )
                    for stack in panels
                ]
            return np.concatenate(panels, axis=2)
        data = self._display_data
        if self._display_filter_active():
            data = np.stack(
                [self._wire_frame(data[k], cache_key=("off", k)) for k in range(int(data.shape[0]))],
                axis=0,
            )
        return data

    def _wire_frame(self, frame: np.ndarray, cache_key=None) -> np.ndarray:
        """Filtered VIEW copy of one display-bound frame; raw data untouched.

        Frames leaving Python for display (buffer window, frame_bytes, offline
        pack, frame server) pass through here. ROI stats keep reading the raw
        frame so reported numbers never come from a filtered view. Filtered
        frames are cached per index so scrubbing stays browser-fast; the cache
        clears whenever a knob changes.
        """
        if not self._display_filter_active():
            return frame
        if cache_key is not None and cache_key in self._display_filter_cache:
            return self._display_filter_cache[cache_key]
        from quantem.widget.utils.display_filter import apply_display_filter

        filtered = apply_display_filter(
            np.asarray(frame),
            mode=self.denoise,
            sigma=float(self.denoise_sigma),
            spatial_bin=int(self.denoise_bin),
        )
        if cache_key is not None:
            self._display_filter_cache[cache_key] = filtered
        return filtered

    def _send_buffer(self, start_idx: int) -> None:
        end_idx = start_idx + self._buffer_size
        if self.diff_mode == "off" and not self._display_filter_active():
            data = self._display_data
            if end_idx <= self.n_slices:
                chunk = data[start_idx:end_idx]
            else:
                chunk = np.concatenate(
                    [data[start_idx:], data[: end_idx - self.n_slices]]
                )
        else:
            frames = []
            for j in range(self._buffer_size):
                idx = (start_idx + j) % self.n_slices
                key = ("buf", idx, self.diff_mode)
                frames.append(self._wire_frame(self._get_display_frame(idx), cache_key=key))
            chunk = np.stack(frames)
        with self.hold_sync():
            self._buffer_start = int(start_idx)
            self._buffer_count = int(chunk.shape[0])
            self._buffer_bytes = chunk.tobytes()

    def _sample_line(self, img: np.ndarray, row0: float, col0: float, row1: float, col1: float) -> np.ndarray:
        """Sample one pixel line between two endpoints via bilinear interpolation.
        N samples = ceil(line length). Used as the building block for line
        profiles; without bilinear the profile aliases on diagonal lines."""
        h, w = img.shape
        dc, dr = col1 - col0, row1 - row0
        length = (dc**2 + dr**2) ** 0.5
        n = max(2, int(np.ceil(length)))
        t = np.linspace(0, 1, n)
        cs = col0 + t * dc
        rs = row0 + t * dr
        ci = np.floor(cs).astype(int)
        ri = np.floor(rs).astype(int)
        cf = cs - ci
        rf = rs - ri
        c0c = np.clip(ci, 0, w - 1)
        c1c = np.clip(ci + 1, 0, w - 1)
        r0c = np.clip(ri, 0, h - 1)
        r1c = np.clip(ri + 1, 0, h - 1)
        return (img[r0c, c0c] * (1 - cf) * (1 - rf) +
                img[r0c, c1c] * cf * (1 - rf) +
                img[r1c, c0c] * (1 - cf) * rf +
                img[r1c, c1c] * cf * rf)

    def _sample_profile_on(self, img: np.ndarray, row0: float, col0: float, row1: float, col1: float) -> np.ndarray:
        """Sample a width-`profile_width` strip averaged perpendicular to the line.
        Averaging across the strip reduces shot noise on thin line profiles -
        a single-pixel sample is too noisy for atomic-resolution data."""
        pw = self.profile_width
        if pw <= 1:
            return self._sample_line(img, row0, col0, row1, col1).astype(np.float32)
        dc, dr = col1 - col0, row1 - row0
        length = (dc**2 + dr**2) ** 0.5
        if length < 1e-8:
            return self._sample_line(img, row0, col0, row1, col1).astype(np.float32)
        perp_r, perp_c = -dc / length, dr / length
        half = (pw - 1) / 2.0
        offsets = np.linspace(-half, half, pw)
        accumulated = None
        for off in offsets:
            vals = self._sample_line(img, row0 + off * perp_r, col0 + off * perp_c,
                                     row1 + off * perp_r, col1 + off * perp_c)
            if accumulated is None:
                accumulated = vals.copy()
            else:
                accumulated += vals
        return (accumulated / pw).astype(np.float32)

    def _sample_profile(self, row0: float, col0: float, row1: float, col1: float) -> np.ndarray:
        """Sample the line profile on the current display frame (binned, diff-aware)
        so the returned profile matches what the user sees."""
        panel = self._resolve_profile_panel()
        frame = self._get_display_panel_frame(panel, int(self.slice_idx))
        return self._sample_profile_on(frame, row0, col0, row1, col1)
