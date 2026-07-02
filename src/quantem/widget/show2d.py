"""
show2d: Static 2D image viewer with optional FFT and histogram analysis.

For displaying a single image or a static gallery of multiple images.
Unlike Show3D (interactive), Show2D focuses on static visualization.
"""

import base64
import io
import json
import math
import os
import pathlib
import tempfile
import warnings
from enum import StrEnum
from typing import Self, Sequence

import anywidget
import matplotlib
import matplotlib.patches
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import traitlets
from quantem.widget.utils.array import _b64_safe, _resize_image, to_numpy
from quantem.widget.utils.state_io import resolve_widget_version, save_state_file, unwrap_state_payload

from quantem.core.datastructures import Dataset2d, Dataset3d


def _reject_unknown_kwargs(cls, kwargs: dict) -> None:
    """Raise TypeError if kwargs contains any key that isn't a declared trait.

    anywidget/traitlets silently accept unknown keys, which let stale notebooks
    pass obsolete params like ``pixel_size_angstrom=0.5`` with no warning.  This
    helper catches typos and renamed-trait references at construction time.
    """
    traits = set(cls.class_trait_names())
    unknown = [k for k in kwargs if k not in traits]
    if unknown:
        key = sorted(unknown)[0]
        raise TypeError(
            f"{cls.__name__}() got unexpected keyword argument {key!r}. "
            f"Check for typos or a renamed parameter (e.g. canvas_size → size, "
            f"image_width_px → size, pixel_size_angstrom → pixel_size)."
        )


def _round_to_nice(value: float) -> float:
    """Round a physical length to a 'nice' value (1, 2, 5, 10, 20, 50, ...)."""
    if value <= 0:
        return 1.0
    exp = math.floor(math.log10(value))
    base = 10 ** exp
    mantissa = value / base
    if mantissa < 1.5:
        return base
    elif mantissa < 3.5:
        return 2 * base
    elif mantissa < 7.5:
        return 5 * base
    else:
        return 10 * base


def _js_round(value: float) -> int:
    """JS ``Math.round`` (half away from zero for positives); Python's
    banker's rounding would format 2.5 as "2" where the widget shows "3"."""
    return math.floor(value + 0.5)


# Ports of js/figure.ts unit tables so the static PNG's scale bar label is
# character-identical to the live widget's canvas label.
# Length-unit ladder, each as its size in nm: a sub-1 value in one unit
# (e.g. 0.5 nm) displays as a clean integer in a smaller unit (5 Å), because
# microscopists read "5 Å", not "0.50 nm".
_LENGTH_UNITS_NM: tuple[tuple[str, float], ...] = (
    ("mm", 1e6), ("µm", 1e3), ("nm", 1.0), ("Å", 0.1), ("pm", 1e-3),
)
# Base unit (the trait's unit) -> nm. Only length units rescale; anything else
# (mrad, ps, px, ...) keeps its own unit and the decimal fallback.
_BASE_UNIT_NM: dict[str, float] = {
    "mm": 1e6, "µm": 1e3, "μm": 1e3, "micron": 1e3, "microns": 1e3, "um": 1e3,
    "nm": 1.0, "nanometer": 1.0, "nanometers": 1.0,
    "å": 0.1, "angstrom": 0.1, "angstroms": 0.1, "ang": 0.1, "a": 0.1,
    "pm": 1e-3, "picometer": 1e-3, "picometers": 1e-3,
}


def _unit_symbol(unit: str) -> str:
    """Display symbol for a unit string (port of js/figure.ts unitSymbol).

    Users pass units like "micron"/"A" on a Dataset; the widget renders the
    conventional glyph (µm, Å) so labels read like a journal figure. Unknown
    strings pass through unchanged."""
    u = (unit or "").strip()
    lc = u.lower()
    if lc in ("micron", "microns", "um") or u in ("μm", "µm"):
        return "µm"
    if lc in ("angstrom", "angstroms", "ang", "a") or u == "Å":
        return "Å"
    if lc in ("nanometer", "nanometers", "nm"):
        return "nm"
    if lc in ("picometer", "picometers", "pm"):
        return "pm"
    if lc in ("millimeter", "millimeters", "mm"):
        return "mm"
    if lc in ("picosecond", "picoseconds", "ps"):
        return "ps"
    if lc in ("femtosecond", "femtoseconds", "fs"):
        return "fs"
    if lc in ("nanosecond", "nanoseconds", "ns"):
        return "ns"
    return u


def _format_scale_label(value: float, unit: str) -> str:
    """Scale bar label (port of js/figure.ts formatScaleLabel).

    Length values auto-pick the unit that reads as a clean integer:
    0.5 nm -> "5 Å", 0.005 nm -> "5 pm". Non-length units (mrad, ps, px)
    keep their unit. ``_round_to_nice`` gives n*10^k and every ladder step
    is a power of 10, so the rescaled number is always exact."""
    nice = _round_to_nice(value)
    base_nm = _BASE_UNIT_NM.get((unit or "").strip().lower())
    if base_nm is None:
        sym = _unit_symbol(unit)
        return f"{_js_round(nice)} {sym}" if nice >= 1 else f"{nice:.2f} {sym}"
    value_nm = nice * base_nm
    # largest ladder unit where the value is >= 1 -> the fewest-digit integer
    for sym, unit_nm in _LENGTH_UNITS_NM:
        if value_nm / unit_nm >= 1:
            return f"{_js_round(value_nm / unit_nm)} {sym}"
    sym, unit_nm = _LENGTH_UNITS_NM[-1]
    return f"{_js_round(value_nm / unit_nm)} {sym}"


_OVERLAY_FONT: list[str] | None = None


def _static_overlay_font() -> list[str]:
    """Closest installed match to the widget's ``-apple-system, BlinkMacSystemFont,
    'Segoe UI', sans-serif`` stack, resolved once. Filtering to installed fonts
    avoids matplotlib findfont warnings on machines without Helvetica/Arial."""
    global _OVERLAY_FONT
    if _OVERLAY_FONT is None:
        from matplotlib import font_manager
        installed = {f.name for f in font_manager.fontManager.ttflist}
        preferred = ("Helvetica Neue", "Segoe UI", "Arial", "Liberation Sans")
        _OVERLAY_FONT = [f for f in preferred if f in installed] + ["DejaVu Sans"]
    return _OVERLAY_FONT


class Colormap(StrEnum):
    INFERNO = "inferno"
    VIRIDIS = "viridis"
    MAGMA = "magma"
    PLASMA = "plasma"
    GRAY = "gray"


class Show2D(anywidget.AnyWidget):
    """
    Static 2D image viewer with optional FFT and histogram analysis.

    Display a single image or multiple images in a gallery layout.
    For interactive stack viewing with playback, use Show3D instead.

    Parameters
    ----------
    data : array_like
        2D array (height, width) for single image, or
        3D array (N, height, width) for multiple images displayed as gallery.
    labels : list of str, optional
        Labels for each image in gallery mode.
    title : str, optional
        Title to display above the image(s).
    cmap : str, default "inferno"
        Colormap name ("magma", "viridis", "gray", "inferno", "plasma").
    sampling : float or tuple of float, optional
        Pixel size per axis ``(row, col)``. Scalar broadcasts to both axes.
        Used for scale bar display. Defaults to ``(1, 1)``.
    units : str or list of str, optional
        Unit string per axis. Scalar broadcasts to both. Common: ``"A"``,
        ``"nm"``, ``"pixels"``. Defaults to ``["pixels", "pixels"]``.
    show_fft : bool, default False
        Show FFT and histogram panels.
    show_stats : bool, default True
        Show statistics (mean, min, max, std).
    log_scale : bool, default False
        Use log scale for intensity mapping.
    auto_contrast : bool, default False
        Use percentile-based contrast.
    vmin : float, optional
        Absolute minimum intensity for color mapping. When both vmin and vmax
        are set, all gallery images share the same intensity scale: essential
        for A/B visual comparison.
    vmax : float, optional
        Absolute maximum intensity for color mapping.
    ncols : int, default 3
        Number of columns in gallery mode.
    size : int, default 0
        Canvas rendering size in CSS pixels (the on-screen width of each image).
        ``0`` uses the frontend default: 500 px for a single image, 300 px per
        image in gallery mode.  Pass e.g. ``size=800`` to enlarge for a
        presentation, or ``size=200`` to compress alongside a control panel.
        This controls **display only**: the underlying image resolution is
        never resampled; zooming into a 4K image preserves every pixel.
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

    Examples
    --------
    >>> import numpy as np
    >>> from quantem.widget import Show2D

    Single 2D NumPy array:

    >>> Show2D(np.random.rand(512, 512))

    PyTorch tensor (CPU or GPU, any dtype):

    >>> import torch
    >>> Show2D(torch.rand(512, 512))

    3D NumPy stack ``(N, H, W)`` rendered as a gallery:

    >>> Show2D(np.random.rand(6, 256, 256), ncols=3)

    List of arrays with different shapes (center-padded to a common canvas):

    >>> Show2D([np.random.rand(256, 256), np.random.rand(300, 400)])

    quantem ``Dataset2d``: title, sampling, units auto-extracted:

    >>> from quantem.core.datastructures import Dataset2d
    >>> ds = Dataset2d.from_array(np.random.rand(512, 512))
    >>> Show2D(ds)

    quantem ``Dataset3d``: gallery view of N frames with calibration:

    >>> from quantem.core.datastructures import Dataset3d
    >>> ds = Dataset3d.from_array(np.random.rand(6, 256, 256))
    >>> Show2D(ds, ncols=3)

    A/B comparison with shared contrast and linked zoom/pan:

    >>> a, b = np.random.rand(512, 512), np.random.rand(512, 512)
    >>> Show2D([a, b], vmin=0, vmax=1, link_zoom=True, link_pan=True)

    Per-image absolute contrast (one ``vmin``/``vmax`` per image):

    >>> Show2D([a, b], vmin=[0.0, 0.2], vmax=[1.0, 0.8])

    Drift comparison: diff mode adds a ``A - B`` panel alongside the originals
    (gallery becomes ``[A, B, A - B]``):

    >>> Show2D([a, b], diff_mode=True, link_zoom=True, link_pan=True)

    Large image: display-only canvas size (full resolution preserved):

    >>> Show2D(np.random.rand(4096, 4096), size=800)

    Per-panel display width for galleries. Use ``ncols`` to choose an
    intentional gallery shape, for example ``ncols=2`` for a 2×2 comparison
    of four images or ``ncols=4`` for a single row:

    >>> Show2D(np.random.rand(13, 128, 128), ncols=13, panel_width_px=70)

    Static export to PDF or PNG (vector PDF for publication figures):

    >>> w = Show2D(np.random.rand(512, 512), sampling=0.5, units="nm")
    >>> w.save_image("figure.pdf", dpi=150)
    """

    _esm = pathlib.Path(__file__).parent / "static" / "show2d.js"

    # =========================================================================
    # Core State
    # GPU memory budget for display buffers (MB). Each 4K image needs ~192 MB.
    # 12×4K = 2304 MB fits. 16+ triggers auto-bin.
    _GPU_DISPLAY_BUDGET_MB = 2500

    # =========================================================================
    widget_version = traitlets.Unicode("unknown").tag(sync=True)
    n_images = traitlets.Int(1).tag(sync=True)
    height = traitlets.Int(1).tag(sync=True)
    width = traitlets.Int(1).tag(sync=True)
    _display_bin_factor = traitlets.Int(1).tag(sync=True)  # 1 = full-res, 2/4/8 = binned
    _gpu_max_buffer_mb = traitlets.Int(0).tag(sync=True)  # GPU reports maxBufferSize (JS→Python)
    # Flipped True by JS after the first colormap pass has painted to canvas.
    # Used by the Python-side truthful timing print (end-to-end wall clock, not just __init__).
    _js_rendered = traitlets.Bool(False).tag(sync=True)
    frame_bytes = traitlets.Bytes(b"").tag(sync=True)
    # Offline mode: stack quantized to uint8 against global (min, max). 4x
    # smaller than float32 — drops standalone HTML from ~200 MB to ~110 MB.
    # JS dequantizes on read. Eye can't tell uint8 from float32 after colormap
    # reduces to 256 levels anyway.
    offline = traitlets.Bool(False).tag(sync=True)
    # True only on a clone written by export_html: forces the standalone HTML to
    # render on a light/white background regardless of the viewer's OS theme.
    _export_light = traitlets.Bool(False).tag(sync=True)
    _offline_min = traitlets.Float(0.0).tag(sync=True)
    _offline_max = traitlets.Float(1.0).tag(sync=True)
    # Per-image quantization ranges. A gallery's panels can span very different
    # intensities; one GLOBAL (min, max) then wastes most of the 256 uint8 codes
    # on whichever panel is widest, so a narrow panel keeps only a handful of
    # levels and its histogram shows a coarse comb. Quantizing each panel against
    # its OWN (min, max) gives every panel the full 256 codes -> clean histogram.
    _offline_mins = traitlets.List(trait=traitlets.Float(), default_value=[]).tag(sync=True)
    _offline_maxs = traitlets.List(trait=traitlets.Float(), default_value=[]).tag(sync=True)
    export_request = traitlets.Unicode("").tag(sync=True)
    export_status = traitlets.Unicode("").tag(sync=True)
    export_enabled = traitlets.Bool(True).tag(sync=True)
    export_payload = traitlets.Bytes(b"").tag(sync=True)
    export_payload_id = traitlets.Unicode("").tag(sync=True)
    export_filename = traitlets.Unicode("").tag(sync=True)
    labels = traitlets.List(traitlets.Unicode()).tag(sync=True)
    starred = traitlets.List(traitlets.Int()).tag(sync=True)
    hidden_panels = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    show_panel_titles = traitlets.Bool(True).tag(sync=True)
    panel_title_font_size = traitlets.Int(11).tag(sync=True)
    gallery_gap_px = traitlets.Int(8).tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    cmap = traitlets.Unicode("inferno").tag(sync=True)
    ncols = traitlets.Int(3).tag(sync=True)

    # =========================================================================
    # Display Options
    # =========================================================================
    log_scale = traitlets.Bool(False).tag(sync=True)
    auto_contrast = traitlets.Bool(False).tag(sync=True)
    vmin = traitlets.Float(None, allow_none=True).tag(sync=True)
    vmax = traitlets.Float(None, allow_none=True).tag(sync=True)
    vmins = traitlets.List(trait=traitlets.Float(allow_none=True), allow_none=True, default_value=None).tag(sync=True)
    vmaxs = traitlets.List(trait=traitlets.Float(allow_none=True), allow_none=True, default_value=None).tag(sync=True)

    # =========================================================================
    # Scale Bar
    # =========================================================================
    pixel_size = traitlets.Float(0.0).tag(sync=True)
    pixel_sizes = traitlets.List(trait=traitlets.Float(), default_value=[]).tag(sync=True)
    pixel_unit = traitlets.Unicode("pixels").tag(sync=True)
    scale_bar_visible = traitlets.Bool(True).tag(sync=True)
    size = traitlets.Int(0).tag(sync=True)  # Canvas rendering size in CSS pixels; 0 = frontend default
    smooth = traitlets.Bool(False).tag(sync=True)
    initial_zoom = traitlets.Float(1.0).tag(sync=True)
    zoom_row = traitlets.Float(None, allow_none=True).tag(sync=True)
    zoom_col = traitlets.Float(None, allow_none=True).tag(sync=True)
    # Live viewport (row0, row1, col0, col1) in image pixel coordinates.
    # Python -> JS at construction (the view_box= sugar below derives zoom +
    # zoom_row/zoom_col from it); JS -> Python on every pan/zoom (debounced
    # ~100 ms), so current_view always reflects what is on screen. The box is
    # axis-aligned: two corners (row0, col0)/(row1, col1) define all four.
    view_box = traitlets.List(trait=traitlets.Float(), default_value=[]).tag(sync=True)
    link_zoom = traitlets.Bool(False).tag(sync=True)
    link_pan = traitlets.Bool(False).tag(sync=True)
    link_contrast = traitlets.Bool(True).tag(sync=True)
    diff_mode = traitlets.Bool(False).tag(sync=True)
    diff_reference = traitlets.Int(0).tag(sync=True)

    # =========================================================================
    # UI Visibility
    # =========================================================================
    show_controls = traitlets.Bool(True).tag(sync=True)
    show_stats = traitlets.Bool(True).tag(sync=True)
    stats_mean = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_min = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_max = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_std = traitlets.List(traitlets.Float()).tag(sync=True)

    # =========================================================================
    # Analysis Panels (FFT + Histogram shown together)
    # =========================================================================
    show_fft = traitlets.Bool(False).tag(sync=True)
    fft_window = traitlets.Bool(True).tag(sync=True)

    # =========================================================================
    # Selected Image (for single-image analysis display)
    # =========================================================================
    selected_idx = traitlets.Int(0).tag(sync=True)

    # =========================================================================
    # ROI Selection
    # =========================================================================
    roi_active = traitlets.Bool(False).tag(sync=True)
    roi_list = traitlets.List([]).tag(sync=True)
    roi_selected_idx = traitlets.Int(-1).tag(sync=True)

    # =========================================================================
    # Line Profile
    # =========================================================================
    profile_line = traitlets.List(traitlets.Dict()).tag(sync=True)

    # =========================================================================
    # Per-Image Rotation
    # =========================================================================
    image_rotations = traitlets.List(traitlets.Int(), []).tag(sync=True)

    @classmethod
    def from_gif(
        cls,
        path: str | pathlib.Path,
        *,
        ncols: int | None = None,
        labels: list[str | None] | bool | None = None,
        title: str | None = None,
        **kwargs,
    ) -> Self:
        """Open a GIF as a static frame gallery.

        Animated GIFs become an ``(N, H, W)`` stack displayed as a contact sheet.
        Use ``ncols`` to choose an intentional layout, for example ``ncols=2``
        for a 2x2 denoising comparison or ``ncols=1`` for a vertical strip.
        """
        from quantem.widget.io import read_gif  # noqa: PLC0415

        ds = read_gif(path)
        frame_count = int(ds.array.shape[0])
        if ncols is None:
            ncols = max(1, math.ceil(math.sqrt(frame_count)))
        if labels is True:
            resolved_labels = [f"Frame {i + 1}" for i in range(frame_count)]
        elif labels:
            resolved_labels = list(labels)
        else:
            resolved_labels = None
        return cls(
            ds,
            labels=resolved_labels,
            title=ds.name if title is None else title,
            ncols=ncols,
            **kwargs,
        )

    def __init__(
        self,
        data: np.ndarray | list[np.ndarray],
        labels: list[str | None] = None,
        title: str = "",
        cmap: str | Colormap = Colormap.INFERNO,
        sampling: float | tuple[float, float] | list[float] | None = None,
        units: str | list[str] | None = None,
        scale_bar_visible: bool = True,
        show_fft: bool = False,
        fft_window: bool = True,
        show_controls: bool = True,
        show_stats: bool = True,
        verbose: bool = True,
        log_scale: bool = False,
        auto_contrast: bool = False,
        offline: bool = False,
        vmin: float | list | None = None,
        vmax: float | list | None = None,
        ncols: int = 3,
        size: int = 0,
        panel_width_px: int = 0,
        smooth: bool = False,
        zoom: float = 1.0,
        zoom_row: float | None = None,
        zoom_col: float | None = None,
        link_zoom: bool | None = None,
        link_pan: bool | None = None,
        link_contrast: bool = True,
        diff_mode: bool = False,
        view_box: tuple | list | None = None,
        display_bin: int | str = "auto",
        hidden_panels: Sequence[int | str] | int | str | None = None,
        starred: Sequence[int | str] | int | str | None = None,
        show_panel_titles: bool = True,
        panel_title_font_size: int = 11,
        gallery_gap_px: int = 8,
        state=None,
        save_state: bool = False,
        **kwargs,
    ):
        import time as _time
        _t0 = _time.perf_counter()
        # Reject typos and stale kwargs (e.g. image_width_px, pixel_size_angstrom).
        # anywidget/traitlets silently ignores unknown keys, which hid the
        # pixel_size_angstrom bug in show2d_all_features.ipynb for months.
        _reject_unknown_kwargs(type(self), kwargs)
        panel_width_px = int(panel_width_px)
        if panel_width_px < 0:
            raise ValueError(f"panel_width_px must be >= 0, got {panel_width_px}")
        if panel_width_px > 0:
            size = panel_width_px
        ncols = int(ncols)
        if ncols < 1:
            raise ValueError(f"ncols must be >= 1, got {ncols}")
        gallery_gap_px = int(gallery_gap_px)
        if gallery_gap_px < 0:
            raise ValueError(f"gallery_gap_px must be >= 0, got {gallery_gap_px}")
        # save_state controls whether the heavy pixel buffers are persisted into
        # the notebook's metadata.widgets on save. Default False: a plain display
        # embeds only light traits + a static PNG, so a 5-panel 4k gallery does
        # not bake ~1 GB into the .ipynb. Set True to persist full interactive
        # state so a reopened notebook restores the widget without a kernel.
        self._save_state = bool(save_state)
        super().__init__(**kwargs)
        # hold_sync() batches ALL traitlet assignments into a single comm message
        # sent when the context manager exits.  Without this, each self.x = y
        # fires a separate round-trip over the ZMQ/websocket channel, which
        # can add 20+ seconds for a 30-image gallery in VS Code Jupyter.
        with self.hold_sync():
            self._init_sync(
                data=data, labels=labels, title=title, cmap=cmap,
                sampling=sampling, units=units, scale_bar_visible=scale_bar_visible,
                show_fft=show_fft, fft_window=fft_window,
                show_controls=show_controls, show_stats=show_stats,
                log_scale=log_scale, auto_contrast=auto_contrast, offline=offline,
                vmin=vmin, vmax=vmax,
                ncols=ncols, size=size, smooth=smooth, zoom=zoom,
                zoom_row=zoom_row, zoom_col=zoom_col,
                link_zoom=link_zoom, link_pan=link_pan, link_contrast=link_contrast,
                diff_mode=diff_mode, view_box=view_box,
                display_bin=display_bin, hidden_panels=hidden_panels, starred=starred,
                show_panel_titles=show_panel_titles, panel_title_font_size=panel_title_font_size,
                gallery_gap_px=gallery_gap_px,
                verbose=verbose, state=state, _t0=_t0)

    def _init_sync(self, *, data, labels, title, cmap, sampling, units,
                   scale_bar_visible, show_fft, fft_window,
                   show_controls, show_stats, log_scale, auto_contrast, offline,
                   vmin, vmax,
                   ncols, size, smooth, zoom, zoom_row, zoom_col,
                   link_zoom, link_pan, link_contrast, diff_mode, view_box,
                   display_bin, hidden_panels, starred, show_panel_titles,
                   panel_title_font_size, gallery_gap_px, verbose, state, _t0):
        import time as _time
        self._verbose = verbose
        self.widget_version = resolve_widget_version()
        self._display_data = None  # initialized after data setup
        self._display_bin = 1

        # First-class support for quantem Dataset2d / Dataset3d:
        # auto-extract array + sampling + units from the dataset object.
        if isinstance(data, (Dataset2d, Dataset3d)) or (
            hasattr(data, "array") and hasattr(data, "name") and hasattr(data, "sampling")
        ):
            if not title and data.name:
                title = data.name
            if sampling is None:
                sampling = tuple(float(s) for s in data.sampling[-2:])
            if units is None and hasattr(data, "units"):
                units = list(data.units[-2:])
            data = data.array
        # Same auto-extract for list/tuple of Dataset2d (gallery from per-file load).
        elif isinstance(data, (list, tuple)) and len(data) > 0 and (
            isinstance(data[0], (Dataset2d, Dataset3d)) or
            (hasattr(data[0], "array") and hasattr(data[0], "sampling"))
        ):
            first = data[0]
            if sampling is None:
                sampling = tuple(float(s) for s in first.sampling[-2:])
            if units is None and hasattr(first, "units"):
                units = list(first.units[-2:])
            data = [d.array for d in data]

        # Convert NumPy / PyTorch / list inputs to a NumPy array.
        if isinstance(data, list):
            images = [to_numpy(d) for d in data]

            # Check if all images have the same shape
            shapes = [img.shape for img in images]
            if len(set(shapes)) > 1:
                # Different sizes - resize all to the largest
                max_h = max(s[0] for s in shapes)
                max_w = max(s[1] for s in shapes)
                images = [_resize_image(img, max_h, max_w) for img in images]

            data = np.stack(images)
        else:
            data = to_numpy(data)

        # Ensure 3D shape (N, H, W)
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        # Avoid redundant copy: np.asarray is a no-op when already float32 + contiguous
        if data.dtype == np.float32:
            self._data = np.array(data, dtype=np.float32, copy=True)
        else:
            self._data = np.asarray(data, dtype=np.float32)
        # Store originals for rotation reset: views into _data (no copy).
        # Only materialized as independent copies when a rotation is applied.
        self._data_original = [self._data[i] for i in range(self._data.shape[0])]
        self._originals_are_views = True
        self.n_images = int(data.shape[0])
        self.height = int(data.shape[1])
        self.width = int(data.shape[2])
        self.image_rotations = [0] * self.n_images

        # Labels
        if labels is None:
            self.labels = [f"Image {i+1}" for i in range(self.n_images)]
        else:
            self.labels = list(labels)
        self.starred = [0] * self.n_images
        self.hidden_panels = []
        self.show_panel_titles = bool(show_panel_titles)
        self.panel_title_font_size = int(panel_title_font_size)
        self.gallery_gap_px = int(gallery_gap_px)
        if starred is not None:
            self.set_starred_panels(starred)
        if hidden_panels is not None:
            self.set_hidden_panels(hidden_panels)

        # Options
        self.title = title
        self.cmap = cmap
        # Resolve sampling + units to scalar pixel_size + pixel_unit (column axis).
        # Scalar shorthand: sampling=0.5 → (0.5, 0.5). units="nm" → ["nm", "nm"].
        if sampling is None:
            self.pixel_size = 0.0
        elif isinstance(sampling, (int, float)):
            self.pixel_size = float(sampling)
        else:
            self.pixel_size = float(sampling[-1])
        if units is None:
            self.pixel_unit = "pixels"
        elif isinstance(units, str):
            self.pixel_unit = units
        else:
            self.pixel_unit = str(units[-1])
        self.scale_bar_visible = scale_bar_visible
        self.pixel_sizes = []
        self.size = size
        self.smooth = smooth
        # view_box sugar: sets zoom + zoom_row/col to center on box
        if view_box is not None:
            r0, r1, c0, c1 = [float(v) for v in view_box]
            box_h = max(1.0, r1 - r0)
            box_w = max(1.0, c1 - c0)
            zoom = float(min(self.height / box_h, self.width / box_w))
            zoom_row = (r0 + r1) / 2
            zoom_col = (c0 + c1) / 2
            self.view_box = [r0, r1, c0, c1]
        self.initial_zoom = zoom
        self.zoom_row = zoom_row
        self.zoom_col = zoom_col
        # Auto-link zoom + pan in gallery (n_images >= 2) so dragging one panel
        # follows the other — typical compare/diff workflow. Single image: no-op.
        self.link_zoom = (self.n_images >= 2) if link_zoom is None else link_zoom
        self.link_pan = (self.n_images >= 2) if link_pan is None else link_pan
        self.link_contrast = link_contrast
        self.diff_mode = diff_mode if self.n_images >= 2 else False
        if show_fft and self.height * self.width > 2048 * 2048:
            warnings.warn(
                f"FFT on {self.height}×{self.width} image ({self.height * self.width / 1e6:.1f}M pixels) "
                f"may be slow. Consider using ROI FFT for a sub-region.",
                stacklevel=2,
            )
        self.show_fft = show_fft
        self.fft_window = fft_window
        self.show_controls = show_controls
        self.show_stats = show_stats
        self.log_scale = log_scale
        self.auto_contrast = auto_contrast
        self.offline = offline
        # Accept scalar OR list for vmin/vmax. List → per-image (vmins/vmaxs).
        if isinstance(vmin, (list, tuple)) or isinstance(vmax, (list, tuple)):
            n = self.n_images
            def _expand(v):
                if v is None:
                    return [None] * n
                if isinstance(v, (list, tuple)):
                    if len(v) != n:
                        raise ValueError(f"vmin/vmax list has length {len(v)} but n_images is {n}. Pass a list of length {n} or a scalar to apply uniformly.")
                    return [None if x is None else float(x) for x in v]
                return [float(v)] * n
            self.vmins = _expand(vmin)
            self.vmaxs = _expand(vmax)
            self.vmin = None
            self.vmax = None
        else:
            self.vmin = vmin
            self.vmax = vmax
        self.ncols = ncols

        # Auto-bin for display: keep full-res in _data, send binned to JS.
        # GPU memory budget: ~2 GB for display buffers (128 MB per image at 4K).
        # At 4K: max ~16 full-res. Beyond that, auto-downsample.
        if display_bin == "auto":
            # Each 4K image needs ~192 MB GPU buffers (float32 + RGBA + read)
            # Tested: 12×4K (2.3 GB) works, 24×4K (4.6 GB) OOMs
            # Budget: 2.5 GB allows 12×4K full-res, bins above that
            gpu_budget_mb = self._GPU_DISPLAY_BUDGET_MB
            per_image_mb = (self.height * self.width * 4 * 3) / (1024 * 1024)  # 3 buffers
            total_mb = self.n_images * per_image_mb
            if total_mb > gpu_budget_mb:
                # Find minimum bin factor to fit
                for bf in [2, 4, 8]:
                    binned_mb = self.n_images * per_image_mb / (bf * bf)
                    if binned_mb <= gpu_budget_mb:
                        self._display_bin = bf
                        break
                else:
                    self._display_bin = 8
        elif isinstance(display_bin, int) and display_bin > 1:
            self._display_bin = display_bin

        if self._display_bin > 1:
            from quantem.widget.utils.array import bin2d
            orig_h, orig_w = self._data.shape[1], self._data.shape[2]
            self._display_data = bin2d(self._data, factor=self._display_bin, mode="mean")
            self.height = int(self._display_data.shape[1])
            self.width = int(self._display_data.shape[2])
            if self.pixel_size > 0:
                self.pixel_size = self.pixel_size * self._display_bin
            self._display_bin_factor = self._display_bin
            if verbose:
                print(f"  Display bin {self._display_bin}×: {orig_h}×{orig_w} → {self.height}×{self.width} ({self._display_data.nbytes // 1024 // 1024} MB)")
        else:
            self._display_data = self._data
            self._display_bin_factor = 1

        # Compute initial stats (from full-res data)
        self._compute_all_stats()

        # Send display data to JS (possibly binned)
        self._update_all_frames()

        self.selected_idx = 0

        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(
                    json.loads(pathlib.Path(state).read_text()),
                    require_envelope=True,
                )
            else:
                state = unwrap_state_payload(state)
            self.load_state_dict(state)

        # Stash wall-clock start on the instance; the observer below prints the
        # TRUE end-to-end time after JS signals first paint.  The Python-only
        # __init__ number is misleading for widget UX: a widget is not "done"
        # until the browser has painted its first frame.
        self._init_t0 = _t0
        self._init_py_elapsed_ms = (_time.perf_counter() - _t0) * 1000
        self.observe(self._on_first_render, names=["_js_rendered"])
        self.observe(self._on_export_request_change, names=["export_request"])

    @traitlets.validate("starred")
    def _validate_starred(self, proposal: dict) -> list[int]:
        """Normalize per-image star flags."""
        val = list(proposal["value"])
        n_img = int(self.n_images)
        if not val:
            return [0] * n_img
        if len(val) != n_img:
            raise traitlets.TraitError(
                f"starred length ({len(val)}) must equal n_images ({n_img})"
            )
        return [1 if int(v) else 0 for v in val]

    @traitlets.validate("hidden_panels")
    def _validate_hidden_panels(self, proposal: dict) -> list[int]:
        """Normalize hidden image indices and keep at least one image visible."""
        n_img = int(self.n_images)
        clean: list[int] = []
        seen: set[int] = set()
        for value in proposal["value"]:
            idx = int(value)
            if 0 <= idx < n_img and idx not in seen:
                clean.append(idx)
                seen.add(idx)
        clean.sort()
        if len(clean) >= n_img:
            raise traitlets.TraitError(
                "hidden_panels cannot hide every panel; at least one panel must remain visible"
            )
        return clean

    def _panel_title_for_index(self, panel: int) -> str:
        """Return the user-facing label for a Show2D image panel."""
        if 0 <= panel < len(self.labels) and self.labels[panel]:
            return str(self.labels[panel])
        return f"Image {panel + 1}"

    def _resolve_panel_ref(self, panel: int | str) -> int:
        """Resolve a panel index or exact label into a zero-based image index."""
        if isinstance(panel, bool):
            raise TypeError("panel must be an integer index or exact label, not bool")
        if isinstance(panel, int):
            idx = int(panel)
            if 0 <= idx < int(self.n_images):
                return idx
            raise ValueError(f"panel index {idx} out of range [0, {self.n_images})")
        if isinstance(panel, str):
            titles = [self._panel_title_for_index(i) for i in range(int(self.n_images))]
            matches = [i for i, title in enumerate(titles) if title == panel]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError(
                    f"panel label {panel!r} is not unique; use a zero-based panel index instead"
                )
            available = ", ".join(repr(title) for title in titles)
            raise ValueError(f"unknown panel label {panel!r}; available labels: {available}")
        raise TypeError(
            f"panel must be an integer index or exact label, got {type(panel).__name__}"
        )

    def _normalize_panel_refs(
        self,
        panels: Sequence[int | str] | int | str,
        *,
        allow_empty: bool = False,
    ) -> list[int]:
        """Resolve and de-duplicate image panel references."""
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
            raise ValueError("at least one panel index or label is required")
        return out

    def _on_first_render(self, change):
        import time as _time
        if not change.get("new"):
            return
        total_ms = (_time.perf_counter() - self._init_t0) * 1000
        py_ms = self._init_py_elapsed_ms
        shape = (f"{self.n_images}×{self.height}×{self.width}"
                 if self.n_images > 1 else f"{self.height}×{self.width}")
        mem = self._data.nbytes
        mem_str = f"{mem / (1 << 20):.0f} MB" if mem >= 1 << 20 else f"{mem / (1 << 10):.0f} KB"
        # Expose as attributes so tests and notebooks can assert on them.
        # These are the ground truth for "did JS actually paint": if they're
        # None, the JS side never signaled first render.
        self.render_total_ms = int(total_ms)
        self.render_python_build_ms = int(py_ms)
        self.render_wire_js_ms = int(total_ms - py_ms)
        if not getattr(self, "_save_state", False) and self.frame_bytes:
            # The frontend has decoded and painted the pixels by the time it
            # flips ``_js_rendered``. Clear the synced model buffer so a later
            # notebook save stores the static PNG fallback, not the full 4k
            # array, while targeted initial sync remains intact.
            self.frame_bytes = b""
        if not getattr(self, "_verbose", True):
            pass
        else:
            print(
                f"Show2D: {shape} {mem_str}: "
                f"rendered in {total_ms:.0f} ms (Python build {py_ms:.0f} ms, "
                f"wire+JS {total_ms - py_ms:.0f} ms)",
                flush=True,
            )
        # Detach observer: one-shot, we only care about the first paint.
        try:
            self.unobserve(self._on_first_render, names=["_js_rendered"])
        except (ValueError, KeyError):
            pass

    def __repr__(self) -> str:
        if self.n_images > 1:
            shape = f"{self.n_images}×{self.height}×{self.width}"
            return f"Show2D({shape}, idx={self.selected_idx}, cmap={self.cmap})"
        return f"Show2D({self.height}×{self.width}, cmap={self.cmap})"

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.log_scale:
            frame = np.log1p(np.maximum(frame, 0))
        if self.vmin is not None and self.vmax is not None:
            vmin = float(self.vmin)
            vmax = float(self.vmax)
            if self.log_scale:
                vmin = float(np.log1p(max(vmin, 0)))
                vmax = float(np.log1p(max(vmax, 0)))
        elif self.auto_contrast:
            vmin = float(np.percentile(frame, 2))
            vmax = float(np.percentile(frame, 98))
        else:
            vmin = float(frame.min())
            vmax = float(frame.max())
        if vmax > vmin:
            normalized = np.clip((frame - vmin) / (vmax - vmin) * 255, 0, 255)
            return normalized.astype(np.uint8)
        return np.zeros(frame.shape, dtype=np.uint8)

    @staticmethod
    def _downsample_static_frame(frame: np.ndarray, max_px: int) -> np.ndarray:
        """Area-downsample a frame for notebook PNG fallback rendering."""
        if max_px <= 0:
            return frame
        h, w = frame.shape[-2:]
        factor = max(1, int(math.ceil(max(h, w) / max_px)))
        if factor == 1:
            return frame

        trimmed_h = max(1, h // factor) * factor
        trimmed_w = max(1, w // factor) * factor
        trimmed = frame[:trimmed_h, :trimmed_w]
        return trimmed.reshape(
            trimmed_h // factor,
            factor,
            trimmed_w // factor,
            factor,
        ).mean(axis=(1, 3))

    def save_image(
        self,
        path: str | pathlib.Path,
        *,
        idx: int | None = None,
        format: str | None = None,
        dpi: int = 150,
        title: bool | str = False,
        colorbar: bool = False,
        scalebar: bool = False,
    ) -> pathlib.Path:
        """Save current image as PNG, PDF, or TIFF.

        When ``title``, ``colorbar``, or ``scalebar`` are enabled, the output
        is a publication-quality figure rendered via matplotlib. Otherwise a
        raw colormapped image is saved directly (faster, exact pixel output).

        Parameters
        ----------
        path : str or pathlib.Path
            Output file path.
        idx : int, optional
            Image index in gallery mode. Defaults to current selected_idx.
        format : str, optional
            'png', 'pdf', or 'tiff'. If omitted, inferred from file extension.
        dpi : int, default 150
            Output DPI.
        title : bool or str, default False
            ``True`` uses the widget title, a string sets a custom title.
        colorbar : bool, default False
            Include a colorbar showing the intensity mapping.
        scalebar : bool, default False
            Include a scale bar (requires ``pixel_size > 0``).

        Returns
        -------
        pathlib.Path
            The written file path.
        """
        from matplotlib import colormaps
        from PIL import Image

        path = pathlib.Path(path)
        fmt = (format or path.suffix.lstrip(".").lower() or "png").lower()
        if fmt not in ("png", "pdf", "tiff", "tif"):
            raise ValueError(f"Unsupported format: {fmt!r}. Use 'png', 'pdf', or 'tiff'.")

        i = idx if idx is not None else self.selected_idx
        if i < 0 or i >= self.n_images:
            raise IndexError(f"Image index {i} out of range [0, {self.n_images})")

        frame = self._data[i]
        normalized = self._normalize_frame(frame)
        cmap_fn = colormaps.get_cmap(self.cmap)
        path.parent.mkdir(parents=True, exist_ok=True)

        use_figure = title or colorbar or scalebar
        if not use_figure:
            rgba = (cmap_fn(normalized / 255.0) * 255).astype(np.uint8)
            img = Image.fromarray(rgba)
            if fmt == "pdf":
                Image.init()
                img = img.convert("RGB")
            img.save(str(path), dpi=(dpi, dpi))
            return path

        # Publication-quality figure via matplotlib
        h, w = frame.shape
        aspect = h / w
        fig_w = 6
        fig, ax = plt.subplots(figsize=(fig_w, fig_w * aspect))
        im = ax.imshow(normalized, cmap=cmap_fn, vmin=0, vmax=255, origin="upper")
        ax.axis("off")

        if title:
            label = title if isinstance(title, str) else self.title
            if label:
                ax.set_title(label, fontsize=14, fontweight="bold", pad=8)

        if colorbar:
            # Map 0–255 back to data-space values for tick labels
            if self.log_scale:
                frame_proc = np.log1p(np.maximum(frame, 0))
            else:
                frame_proc = frame
            if self.vmin is not None and self.vmax is not None:
                dmin = float(self.vmin)
                dmax = float(self.vmax)
                if self.log_scale:
                    dmin = float(np.log1p(max(dmin, 0)))
                    dmax = float(np.log1p(max(dmax, 0)))
            elif self.auto_contrast:
                dmin = float(np.percentile(frame_proc, 2))
                dmax = float(np.percentile(frame_proc, 98))
            else:
                dmin = float(frame_proc.min())
                dmax = float(frame_proc.max())
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            n_ticks = 5
            tick_positions = np.linspace(0, 255, n_ticks)
            tick_labels = [f"{dmin + (dmax - dmin) * t / 255:.4g}" for t in tick_positions]
            cb.set_ticks(tick_positions)
            cb.set_ticklabels(tick_labels)

        if scalebar and self.pixel_size > 0:
            # Compute a nice scale bar length
            target_frac = 0.2  # ~20% of image width
            raw_length_px = target_frac * w
            raw_length_phys = raw_length_px * self.pixel_size  # in Å
            nice = _round_to_nice(raw_length_phys)
            bar_px = nice / self.pixel_size
            if nice >= 10:
                label_text = f"{nice / 10:.4g} nm"
            else:
                label_text = f"{nice:.4g} Å"
            margin = 0.03
            bar_y = h * (1 - margin) - 2
            bar_x = w * (1 - margin) - bar_px
            ax.plot([bar_x, bar_x + bar_px], [bar_y, bar_y],
                    color="white", linewidth=3, solid_capstyle="butt")
            ax.plot([bar_x, bar_x + bar_px], [bar_y, bar_y],
                    color="black", linewidth=1, solid_capstyle="butt")
            ax.text(bar_x + bar_px / 2, bar_y - h * 0.02, label_text,
                    color="white", fontsize=10, fontweight="bold",
                    ha="center", va="bottom",
                    path_effects=[
                        matplotlib.patheffects.withStroke(linewidth=2, foreground="black")
                    ])

        fig.savefig(str(path), dpi=dpi, bbox_inches="tight",
                    facecolor="white", pad_inches=0.1)
        plt.close(fig)
        return path

    # Traits that carry the bulk pixel payload. Dropped from the saved-notebook
    # snapshot when save_state is False so a plain display stays a few MB, not GB.
    _UNSAVED_HEAVY_KEYS = ("frame_bytes", "export_payload")

    def get_state(self, key=None, drop_defaults=False):
        """Trait state for comm sync and notebook embedding.

        ipywidgets calls this with ``key=None`` to snapshot the FULL state that
        gets written into the saved notebook's ``metadata.widgets``. When
        ``save_state`` is False we drop the heavy image buffers from that
        snapshot, so a plain Show2D does not bake ~1 GB of pixels into the
        .ipynb. Targeted syncs (``key`` is a name or set, used by hold_sync /
        send_state during live rendering) are untouched, so the frontend still
        receives ``frame_bytes`` normally. ``save_state=True`` embeds everything
        so a reopened notebook restores the interactive widget without a kernel.
        """
        state = super().get_state(key=key, drop_defaults=drop_defaults)
        if key is None and not getattr(self, "_save_state", False):
            for heavy_key in self._UNSAVED_HEAVY_KEYS:
                state.pop(heavy_key, None)
        return state

    # Colormaps the frontend treats as sequential; a signed diff panel switches
    # to a diverging map (RdBu) because zero must sit at the visual midpoint.
    _SEQUENTIAL_CMAPS = frozenset(
        {"inferno", "viridis", "plasma", "magma", "hot", "gray", "turbo"}
    )

    @staticmethod
    def _signed_log1p(values: np.ndarray | float) -> np.ndarray | float:
        """Signed log1p, matching the frontend's ``applyLogScale``.

        The widget maps negative intensities to ``-log1p(-v)`` so diff-like
        data keeps its sign under log scale; a plain ``log1p(clip(v, 0))``
        would collapse everything below zero and diverge from the live render.
        """
        return np.sign(values) * np.log1p(np.abs(values))

    @staticmethod
    def _format_stat(value: float) -> str:
        """Format a statistic like the widget's stats row (JS ``formatNumber``).

        ``0`` stays ``"0"``; magnitudes >= 1000 or < 0.01 use two-decimal
        scientific notation with an unpadded exponent (``5.85e+3``, matching
        JS ``toExponential(2)``); everything else uses two fixed decimals.
        """
        if value == 0:
            return "0"
        if abs(value) >= 1000 or abs(value) < 0.01:
            mantissa, exponent = f"{value:.2e}".split("e")
            return f"{mantissa}e{int(exponent):+d}"
        return f"{value:.2f}"

    def _resolve_panel_display_ranges(self, frames: list[np.ndarray]) -> list[tuple[float, float]]:
        """Per-panel ``(vmin, vmax)`` in display space, mirroring the frontend.

        Precedence (same as the JS colormap effect in ``js/show2d/index.tsx``):
        per-image ``vmins[i]/vmaxs[i]`` beat the scalar ``vmin/vmax``, which
        beats ``auto_contrast`` (2/98 percentiles), which beats the frame's
        min/max. With ``link_contrast`` on a gallery and no explicit ranges,
        panels share one merged range so cross-panel intensities compare
        directly. Under ``log_scale`` the limits live in signed-log1p space,
        exactly like the shader. Ranges are computed on the FULL-resolution
        display frames so the PNG's contrast matches the widget even though
        the PNG pixels are later area-binned.
        """
        num = len(frames)
        def to_display(value: float) -> float:
            return float(self._signed_log1p(float(value))) if self.log_scale else float(value)
        has_absolute = self.vmin is not None and self.vmax is not None
        per_mins = list(self.vmins) if self.vmins else [None] * num
        per_maxs = list(self.vmaxs) if self.vmaxs else [None] * num
        has_per_image = [per_mins[i] is not None and per_maxs[i] is not None for i in range(num)]
        base_ranges: list[tuple[float, float]] = []
        for i, frame in enumerate(frames):
            if has_per_image[i]:
                base_ranges.append((to_display(per_mins[i]), to_display(per_maxs[i])))
            elif has_absolute:
                base_ranges.append((to_display(self.vmin), to_display(self.vmax)))
            else:
                base_ranges.append((to_display(frame.min()), to_display(frame.max())))
        linked_shared = (
            self.link_contrast and num >= 2 and not has_absolute and not any(has_per_image)
        )
        shared_base = None
        if linked_shared:
            shared_base = (min(r[0] for r in base_ranges), max(r[1] for r in base_ranges))
        auto_ranges: list[tuple[float, float]] = []
        use_auto = self.auto_contrast and not has_absolute
        if use_auto:
            for i, frame in enumerate(frames):
                if has_per_image[i]:
                    auto_ranges.append(base_ranges[i])
                    continue
                processed = self._signed_log1p(frame) if self.log_scale else frame
                lo, hi = (float(v) for v in np.percentile(processed, (2, 98)))
                # Sparse/clustered data can collapse the 2-98% window to a
                # point; fall back to full extrema like computeAutoRange does.
                full_lo, full_hi = float(processed.min()), float(processed.max())
                if hi - lo <= max(1e-12, abs(full_hi - full_lo) * 1e-6):
                    lo, hi = full_lo, full_hi
                auto_ranges.append((lo, hi))
            if linked_shared:
                shared_auto = (min(r[0] for r in auto_ranges), max(r[1] for r in auto_ranges))
        ranges: list[tuple[float, float]] = []
        for i in range(num):
            if use_auto and not has_per_image[i]:
                ranges.append(shared_auto if linked_shared else auto_ranges[i])
            else:
                ranges.append(shared_base if linked_shared else base_ranges[i])
        return ranges

    def _static_panel_rgb(
        self,
        frame: np.ndarray,
        vmin: float,
        vmax: float,
        cmap_name: str,
        *,
        apply_log: bool | None = None,
    ) -> np.ndarray:
        """Colormap one panel exactly as the live widget maps pixels to colors.

        Pipeline: optional signed-log1p on the data, clip-normalize into the
        display-space ``[vmin, vmax]`` window, then look up the matplotlib
        colormap (the same LUT the JS mirrors). Returning explicit RGB uint8
        keeps matplotlib's own norm machinery out of the loop, so the PNG's
        pixel mapping is byte-identical to what tests can compute independently.
        """
        from matplotlib import colormaps
        apply_log = self.log_scale if apply_log is None else apply_log
        processed = self._signed_log1p(frame) if apply_log else frame
        if vmax > vmin:
            normalized = np.clip((processed - vmin) / (vmax - vmin), 0.0, 1.0)
        else:
            normalized = np.zeros(processed.shape, dtype=np.float64)
        rgba = colormaps.get_cmap(cmap_name)(normalized)
        return (rgba[..., :3] * 255).astype(np.uint8)

    def _static_panel_specs(self) -> list[dict]:
        """Panel plan for the static PNG, mirroring the live widget's layout.

        One entry per visible image panel, plus one signed diff panel per
        non-reference image when ``diff_mode`` is on (the widget renders
        ``ref - other`` with a symmetric range around zero and a diverging
        colormap). Each spec carries the full-resolution frame plus the
        resolved display-space contrast window and a stats line, so the PNG
        renderer only has to bin and colormap.
        """
        frames_source = getattr(self, "_display_data", None)
        if frames_source is None:
            frames_source = getattr(self, "_data", None)
        if frames_source is None or len(frames_source) == 0:
            return []
        frames = [frames_source[i] for i in range(len(frames_source))]
        ranges = self._resolve_panel_display_ranges(frames)
        def stats_line(mean: float, lo: float, hi: float, std: float) -> str:
            fmt = self._format_stat
            return f"Mean {fmt(mean)}   Min {fmt(lo)}   Max {fmt(hi)}   Std {fmt(std)}"
        specs: list[dict] = []
        for i in self.visible_panels:
            label = self._panel_title_for_index(i) if self.show_panel_titles else ""
            specs.append({
                "frame": frames[i],
                "vmin": ranges[i][0],
                "vmax": ranges[i][1],
                "cmap": self.cmap,
                "apply_log": self.log_scale,
                "label": label,
                "stats": stats_line(self.stats_mean[i], self.stats_min[i],
                                    self.stats_max[i], self.stats_std[i]),
            })
        if self.diff_mode and len(frames) >= 2:
            ref = int(self.diff_reference)
            diff_cmap = "RdBu" if self.cmap in self._SEQUENTIAL_CMAPS else self.cmap
            for other in range(len(frames)):
                if other == ref:
                    continue
                diff = frames[ref] - frames[other]
                # Symmetric window centers zero on the diverging map's midpoint,
                # so positive and negative residuals read with equal weight.
                sym = float(max(abs(float(diff.min())), abs(float(diff.max())))) or 1.0
                label = ("Diff (A − B)" if len(frames) == 2
                         else f"Diff (#{ref + 1} − #{other + 1})")
                specs.append({
                    "frame": diff,
                    "vmin": -sym,
                    "vmax": sym,
                    "cmap": diff_cmap,
                    "apply_log": False,  # widget diffs raw data, never log-scaled
                    "label": label if self.show_panel_titles else "",
                    "stats": stats_line(float(diff.mean()), float(diff.min()),
                                        float(diff.max()), float(diff.std())),
                })
        return specs

    @staticmethod
    def _center_crop_slices(height: int, width: int, zoom: float) -> tuple[slice, slice]:
        """Central 1/zoom crop, matching the live widget's zoomed viewport.

        The widget at zoom z (pan 0) scales the full image about the canvas
        center, so the visible region is the central ``height/z x width/z``
        window. The static PNG must show the same pixels or the fallback
        looks nothing like the screenshot the user saved."""
        if zoom <= 1:
            return slice(0, height), slice(0, width)
        crop_h = max(1, _js_round(height / zoom))
        crop_w = max(1, _js_round(width / zoom))
        top = (height - crop_h) // 2
        left = (width - crop_w) // 2
        return slice(top, top + crop_h), slice(left, left + crop_w)

    def _static_canvas_css_px(self) -> float:
        """CSS width of the live panel canvas, the length every JS overlay
        constant (16px font, 12px margin, 60px bar target) is relative to.

        Port of js/show2d/index.tsx: the ``size`` trait when set, else
        SINGLE_IMAGE_TARGET (500) for one image / GALLERY_IMAGE_TARGET (300)
        for a gallery. Without this the static overlays would be drawn for a
        fictitious canvas size and read visibly smaller than the widget's."""
        if self.size > 0:
            return float(self.size)
        return 300.0 if self.n_images > 1 else 500.0

    def _static_overlay_texts(self, specs: list[dict] | None = None,
                              *, css_px: float | None = None) -> list[tuple[str, str, str, float]]:
        """Per-panel overlay strings for the static PNG, one tuple
        ``(label, zoom_text, bar_text, bar_px)`` per panel.

        Pure port of js/figure.ts drawScaleBarHiDPI's math, evaluated on a
        ``css_px``-wide canvas (default: the live widget's own canvas CSS
        width from ``_static_canvas_css_px``): effectiveZoom =
        zoom * cssWidth / imageWidth, target bar 60 CSS px rounded to a nice
        physical length, label via formatScaleLabel. Uncalibrated data gets
        pixelSize 1 and unit "px" exactly like the widget (show2d/index.tsx
        overlay effect). ``bar_px`` is the bar length in panel CSS px;
        ``bar_text``/``bar_px`` are ""/0.0 when the scale bar is hidden.
        Exposed separately from the renderer so tests can assert the strings
        without OCR-ing the PNG."""
        if specs is None:
            specs = self._static_panel_specs()
        if css_px is None:
            css_px = self._static_canvas_css_px()
        # widget clamps initial_zoom to [MIN_ZOOM, MAX_ZOOM] (index.tsx)
        zoom = min(max(float(self.initial_zoom) or 1.0, 0.5), 20.0)
        zoom_text = f"{zoom:.1f}×"  # JS: `${zoom.toFixed(1)}×`
        calibrated = self.pixel_size > 0
        pixel_size = self.pixel_size if calibrated else 1.0
        unit = self.pixel_unit if calibrated else "px"
        texts: list[tuple[str, str, str, float]] = []
        for spec in specs:
            if not self.scale_bar_visible:
                texts.append((spec["label"], zoom_text, "", 0.0))
                continue
            full_w = spec["frame"].shape[1]
            effective_zoom = zoom * css_px / full_w
            # 60 CSS px target bar, rounded to a nice physical length
            nice = _round_to_nice(60.0 / effective_zoom * pixel_size)
            bar_px = nice / pixel_size * effective_zoom
            texts.append((spec["label"], zoom_text,
                          _format_scale_label(nice, unit), bar_px))
        return texts

    def _static_png_b64(self, *, max_px: int = 512, dpi: int = 160) -> str | None:
        """Base64 PNG of all panels, attached to the cell output.

        With ``save_state`` False the interactive widget state is not embedded,
        so a reopened notebook (GitHub, nbviewer, cold Lab) would show nothing.
        This render mirrors the live widget panel-for-panel: same colormap,
        same per-panel or linked contrast window (resolved on the
        full-resolution frame so percentile cuts match the widget, then
        applied to the binned pixels), the same central 1/zoom viewport,
        diff panel(s) when ``diff_mode`` is on, and the widget's own in-panel
        overlays - label top-center, zoom badge bottom-left, scale bar with
        its label bottom-right - at the exact CSS-pixel geometry the JS draws
        on a panel of this size. Panels are area-mean binned to ~``max_px``
        so atomic-lattice detail averages instead of aliasing, and the render
        stays cheap on every display.
        """
        import base64
        import io as _io
        specs = self._static_panel_specs()
        if not specs:
            return None
        css_w = self._static_canvas_css_px()
        overlays = self._static_overlay_texts(specs, css_px=css_w)
        zoom = min(max(float(self.initial_zoom) or 1.0, 0.5), 20.0)
        num = len(specs)
        ncols = max(1, min(self.ncols, num))
        nrows = (num + ncols - 1) // ncols
        # cells sized to the panels' cropped aspect so every image fills its
        # cell exactly: a taller cell would pad panels with white and make the
        # horizontal gutters read wider than the vertical ones
        h0, w0 = specs[0]["frame"].shape
        rows0, cols0 = self._center_crop_slices(h0, w0, zoom)
        aspect = (rows0.stop - rows0.start) / (cols0.stop - cols0.start)
        # inter-panel gutter is the widget's own gallery gap (CSS px of the
        # live canvas), identical horizontally and vertically
        gap_frac = max(0, int(self.gallery_gap_px)) / css_w
        cell_w_in = max_px / dpi  # cell width in inches so 1 panel = max_px device px
        cell_h_in = cell_w_in * aspect
        gap_in = gap_frac * cell_w_in
        fig = plt.figure(figsize=(ncols * cell_w_in + (ncols - 1) * gap_in,
                                  nrows * cell_h_in + (nrows - 1) * gap_in))
        # wspace/hspace are fractions of cell width/height; both resolve to
        # the same gap_in inches so the white gutters match to the pixel
        grid = fig.add_gridspec(nrows, ncols, wspace=gap_frac,
                                hspace=gap_frac / aspect,
                                left=0, right=1, bottom=0, top=1)
        font_family = _static_overlay_font()
        # the live canvas is css_w CSS px wide but the PNG panel is max_px
        # device px wide, so every CSS-px size renders scaled by max_px/css_w
        point = (max_px / css_w) * 72.0 / dpi  # matplotlib points per CSS px
        for idx, (spec, (label, zoom_text, bar_text, bar_css)) in enumerate(zip(specs, overlays)):
            ax = fig.add_subplot(grid[idx // ncols, idx % ncols])
            ax.axis("off")
            frame = spec["frame"]
            rows, cols = self._center_crop_slices(frame.shape[0], frame.shape[1], zoom)
            binned = self._downsample_static_frame(frame[rows, cols], max_px=max_px)
            rgb = self._static_panel_rgb(binned, spec["vmin"], spec["vmax"],
                                         spec["cmap"], apply_log=spec["apply_log"])
            ax.imshow(rgb, interpolation="nearest")
            bin_h, bin_w = rgb.shape[:2]
            ax.set(xlim=(-0.5, bin_w - 0.5), ylim=(bin_h - 0.5, -0.5))
            # CSS px -> data px: the panel canvas is css_w CSS px wide showing
            # bin_w image pixels, so overlay geometry scales by bin_w / css_w
            css_h = css_w * bin_h / bin_w
            k = bin_w / css_w

            def css_xy(x_css: float, y_css: float) -> tuple[float, float]:
                return x_css * k - 0.5, y_css * k - 0.5
            # widget textShadow "1px 1px 0 rgba(0,0,0,0.85), 0 0 3px ..." reads
            # as a soft dark outline; a thin translucent black stroke is its
            # closest matplotlib match (a full-opacity stroke looks stenciled)
            stroke = [matplotlib.patheffects.withStroke(
                linewidth=1.5 * point, foreground=(0, 0, 0, 0.8))]
            if label:
                # panel title Box (show2d/index.tsx): top 6px inside the image,
                # centered, bold max(8, panel_title_font_size) px white @ 95%
                title_px = max(8, int(self.panel_title_font_size or 11))
                ax.text(*css_xy(css_w / 2, 6), label, color=(1, 1, 1, 0.95),
                        fontsize=title_px * point, fontweight="bold",
                        fontfamily=font_family, ha="center", va="top",
                        path_effects=stroke)
            if self.scale_bar_visible:
                # drawScaleBarHiDPI (js/figure.ts): margin 12, bar 5 px thick
                # bottom-right, 16px label centered 4px above the bar, zoom
                # badge left-aligned at x=12 sharing the bar's bottom edge,
                # all under a soft (1,1)-offset half-black shadow
                bar_x = css_w - bar_css - 12
                bar_y = css_h - 12
                ax.add_patch(matplotlib.patches.Rectangle(
                    css_xy(bar_x + 1, bar_y + 1), bar_css * k, 5 * k,
                    facecolor=(0, 0, 0, 0.5), edgecolor="none"))
                ax.add_patch(matplotlib.patches.Rectangle(
                    css_xy(bar_x, bar_y), bar_css * k, 5 * k,
                    facecolor="white", edgecolor="none"))
                ax.text(*css_xy(bar_x + bar_css / 2, bar_y - 4), bar_text,
                        color="white", fontsize=16 * point,
                        fontfamily=font_family, ha="center", va="bottom",
                        path_effects=stroke)
                ax.text(*css_xy(12, css_h - 12 + 5), zoom_text,
                        color="white", fontsize=16 * point,
                        fontfamily=font_family, ha="left", va="bottom",
                        path_effects=stroke)
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor="white",
                    bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _repr_mimebundle_(self, **kwargs):
        """Display bundle: interactive widget live, static PNG for cold reopen.

        When ``save_state`` is False we add an ``image/png`` fallback to the
        widget-view bundle. Live Jupyter renders the interactive widget (richest
        mime); a kernel-less reopen falls back to the PNG. When ``save_state`` is
        True the full state is embedded, so no static fallback is needed.
        """
        bundle = super()._repr_mimebundle_(**kwargs)
        if getattr(self, "_save_state", False) or bundle is None:
            return bundle
        png = self._static_png_b64()
        if png:
            data = bundle[0] if isinstance(bundle, tuple) else bundle
            data["image/png"] = png
        return bundle

    def _ipython_display_(self):
        """Publish the widget view plus a SEPARATE static-PNG sibling output.

        JupyterLab's widget renderer outranks ``image/png`` inside a single
        bundle: on a cold reopen with ``save_state=False`` it shows "Error
        displaying widget: model not found" and never falls back to the PNG
        living in the same output. Publishing the render as its own
        display_data output survives, because Lab only swallows the
        widget-view output. ``save_state=True`` embeds full interactive
        state, so no sibling is emitted. The sibling carries this widget's
        model id so the live frontend can hide it while the interactive view
        is mounted.

        The sibling starts as an EMPTY placeholder and is filled with the
        PNG only after the cell finishes: rendering at display time would
        block the cell output on matplotlib for every widget shown, and the
        PNG is only ever consumed by a notebook save that happens later.
        """
        from IPython.display import display
        # super() bundle only: the in-bundle PNG _repr_mimebundle_ adds for
        # direct consumers (nbsphinx, repr displays) would render here
        # synchronously; the sibling fill below covers the notebook-save path
        bundle = super()._repr_mimebundle_()
        if bundle is None:
            return
        data, metadata = bundle if isinstance(bundle, tuple) else (bundle, {})
        display(data, raw=True, metadata=metadata or None)
        if not getattr(self, "_save_state", False):
            self._display_static_sibling_deferred()

    def _static_fallback_marker(self, png_b64: str | None = None) -> str:
        """The sibling's HTML: an ``img.quantem-static-fallback`` tagged with
        this widget's model id, which is exactly what the live frontend's
        hide effect queries. The empty placeholder keeps the tag (hidden) so
        the hide keeps working after ``update_display_data`` swaps in the
        render."""
        note = "Show2D static render (for saved-notebook viewing)"
        src = f' src="data:image/png;base64,{png_b64}"' if png_b64 else ' style="display:none"'
        return (f'<img class="quantem-static-fallback" '
                f'data-quantem-model-id="{self.model_id}"{src} alt="{note}">')

    def _display_static_sibling_deferred(self):
        """Reserve the sibling output now, render the PNG after the cell ends.

        The placeholder is published with a ``display_id``; a one-shot
        ``post_execute`` hook renders the PNG once the current cell finishes
        and rewrites the placeholder via ``update_display_data``. The widget
        therefore appears instantly, the notebook file saved any time after
        the cell carries the render, and the live view never shows a
        duplicate (the placeholder is empty, then hidden by the frontend).
        ``post_execute`` fires inside the same execute request, so headless
        runners (nbconvert/nbclient) capture the update deterministically.
        Terminal IPython has no notebook file to save, so outside a kernel
        (no ZMQInteractiveShell) no sibling is emitted at all.
        """
        from IPython import get_ipython
        from IPython.display import display
        shell = get_ipython()
        if shell is None or type(shell).__name__ != "ZMQInteractiveShell":
            return
        handle = display(
            {"text/html": self._static_fallback_marker(), "text/plain": ""},
            raw=True, display_id=True,
            metadata={"quantem.widget": {"static_fallback": True}},
        )

        def fill():
            png_b64 = self._static_png_b64()
            if not png_b64 or handle is None:
                return
            handle.update(
                {
                    "image/png": base64.b64decode(png_b64),
                    "text/html": self._static_fallback_marker(png_b64),
                    "text/plain": "Show2D static render (for saved-notebook viewing)",
                },
                raw=True,
                metadata={"quantem.widget": {"static_fallback": True}},
            )

        def fill_once():
            shell.events.unregister("post_execute", fill_once)
            fill()

        shell.events.register("post_execute", fill_once)
        self._static_fallback_fill = fill  # test hook: flush the deferred render

    def state_dict(self):
        return {
            "title": self.title,
            "cmap": self.cmap,
            "log_scale": self.log_scale,
            "auto_contrast": self.auto_contrast,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "labels": list(self.labels),
            "starred": list(self.starred),
            "hidden_panels": list(self.hidden_panels),
            "show_panel_titles": self.show_panel_titles,
            "panel_title_font_size": self.panel_title_font_size,
            "show_stats": self.show_stats,
            "show_fft": self.show_fft,
            "fft_window": self.fft_window,
            "show_controls": self.show_controls,
            "pixel_size": self.pixel_size,
            "pixel_sizes": list(self.pixel_sizes),
            "pixel_unit": self.pixel_unit,
            "scale_bar_visible": self.scale_bar_visible,
            "size": self.size,
            "smooth": self.smooth,
            "initial_zoom": self.initial_zoom,
            "vmins": self.vmins,
            "vmaxs": self.vmaxs,
            "link_zoom": self.link_zoom,
            "link_pan": self.link_pan,
            "link_contrast": self.link_contrast,
            "zoom_row": self.zoom_row,
            "zoom_col": self.zoom_col,
            "view_box": list(self.view_box),
            "diff_mode": self.diff_mode,
            "ncols": self.ncols,
            "selected_idx": self.selected_idx,
            "roi_active": self.roi_active,
            "roi_list": self.roi_list,
            "roi_selected_idx": self.roi_selected_idx,
            "profile_line": self.profile_line,
            "image_rotations": list(self.image_rotations),
            "display_bin": self._display_bin,
        }

    def save(self, path: str):
        save_state_file(path, "Show2D", self.state_dict())

    def export_html(self, path: str | pathlib.Path | None = None,
                    *,
                    title: str | None = None,
                    mode: str = "single",
                    encoding: str = "full",
                    downsample: int | None = None,
                    quantized: bool | None = None) -> pathlib.Path:
        """Write a standalone HTML viewer for this widget.

        The exported file mounts the live anywidget JS bundle with the current
        widget state (data, labels, cmap, vmin/vmax, log_scale, sampling, ...).
        Opens in any browser without a Jupyter kernel.
        Preferred export options are ``mode="single"``, ``encoding="full"`` or
        ``encoding="uint8"``, and ``downsample=None``. ``quantized`` is kept as
        a compatibility alias for ``encoding="uint8"``.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            Destination HTML path.
        quantized : bool, default False
            Store the displayed image stack as uint8 with min/max metadata.
            This is smaller and visually equivalent after colormapping. The
            default stores exact float32 display values.
        title : str, optional
            Browser page title. Defaults to widget ``title`` or "Show2D".
        """
        if self._data is None:
            raise ValueError("Cannot export HTML after free(); rebuild the widget first.")

        quantized = self._normalise_html_export_options(
            mode=mode,
            encoding=encoding,
            downsample=downsample,
            quantized=quantized,
        )
        export_path = pathlib.Path(path) if path is not None else self._default_html_export_path(quantized)
        self._write_html_export(export_path, quantized=quantized, title=title)
        size_mb = export_path.stat().st_size / (1024 * 1024)
        label = self._export_mode_label(quantized)
        self.export_status = f"Exported {export_path.name} ({size_mb:.1f} MB, {label})"
        return export_path

    def _normalise_html_export_options(
        self,
        *,
        mode: str = "single",
        encoding: str = "full",
        downsample: int | None = None,
        quantized: bool | None = None,
    ) -> bool:
        raw_mode = str(mode or "single").strip().lower().replace("_", "-")
        if raw_mode in {"exact", "full"}:
            raw_mode = "single"
            encoding = "full"
        elif raw_mode in {"quantized", "uint8", "u8"}:
            raw_mode = "single"
            encoding = "uint8"
        if raw_mode != "single":
            raise ValueError("Show2D HTML export supports mode='single'")
        if downsample not in (None, 1, "1", "", 0, "0"):
            raise NotImplementedError("Show2D HTML export does not support downsample yet")
        raw_encoding = str(encoding or "full").strip().lower().replace("_", "-")
        if quantized is True:
            raw_encoding = "uint8"
        elif quantized is False and raw_encoding in {"quantized", "uint8", "u8"}:
            raw_encoding = "uint8"
        if raw_encoding in {"full", "exact", "float32", "f32"}:
            return False
        if raw_encoding in {"uint8", "u8", "quantized"}:
            return True
        raise ValueError(f"unknown Show2D export encoding {encoding!r}; expected 'full' or 'uint8'")

    def _on_export_request_change(self, change: dict) -> None:
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
            quantized = self._normalise_html_export_options(
                mode=mode,
                encoding=str(payload.get("encoding", "full")),
                downsample=payload.get("downsample"),
                quantized=None,
            )
            if payload.get("download"):
                filename = str(payload.get("filename") or self._default_html_export_path(quantized).name)
                request_id = str(payload.get("id") or "")
                self.export_status = f"Preparing {filename}..."
                html = self._html_export_bytes(quantized=quantized)
                self.export_filename = filename
                self.export_payload = html
                self.export_payload_id = request_id
                size_mb = len(html) / (1024 * 1024)
                label = self._export_mode_label(quantized)
                self.export_status = f"Ready {filename} ({size_mb:.1f} MB, {label})"
            else:
                self.export_status = f"Exporting {mode} HTML..."
                self.export_html(quantized=quantized)
        except Exception as exc:
            self.export_status = f"Export failed: {exc}"

    def _default_html_export_path(self, quantized: bool) -> pathlib.Path:
        label = self.title.strip() or "show2d"
        slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        if not slug:
            slug = "show2d"
        mode = "quantized" if quantized else "exact"
        shape = f"{self.n_images}x{self.height}x{self.width}" if self.n_images > 1 else f"{self.height}x{self.width}"
        return pathlib.Path.cwd() / f"{slug}_{shape}_{mode}.html"

    def _export_mode_label(self, quantized: bool) -> str:
        return "uint8" if quantized else "full float32"

    def _write_html_export(
        self,
        path: str | pathlib.Path,
        *,
        quantized: bool,
        title: str | None = None,
    ) -> pathlib.Path:
        from ipywidgets.embed import dependency_state, embed_minimal_html

        from .export import ensure_mobile_viewport

        export_path = pathlib.Path(path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        page_title = title or self.title or "Show2D"
        export_widget = self._clone_for_html_export(quantized=quantized)
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
            export_widget.close()
        ensure_mobile_viewport(export_path)
        return export_path

    def _html_export_bytes(self, *, quantized: bool) -> bytes:
        with tempfile.TemporaryDirectory(prefix="show2d-export-") as tmp:
            path = pathlib.Path(tmp) / self._default_html_export_path(quantized).name
            self._write_html_export(path, quantized=quantized)
            return path.read_bytes()

    def _clone_for_html_export(self, *, quantized: bool) -> Self:
        data = self._display_data if self._display_data is not None else self._data
        if data is None:
            raise ValueError("Cannot export HTML after free(); rebuild the widget first.")
        clone = type(self)(
            np.ascontiguousarray(data, dtype=np.float32),
            labels=list(self.labels),
            title=self.title,
            cmap=self.cmap,
            sampling=self.pixel_size if self.pixel_size > 0 else None,
            units=self.pixel_unit,
            scale_bar_visible=self.scale_bar_visible,
            show_fft=self.show_fft,
            fft_window=self.fft_window,
            show_controls=self.show_controls,
            show_stats=self.show_stats,
            verbose=False,
            log_scale=self.log_scale,
            auto_contrast=self.auto_contrast,
            offline=quantized,
            vmin=self.vmin if self.vmin is not None else self.vmins,
            vmax=self.vmax if self.vmax is not None else self.vmaxs,
            ncols=self.ncols,
            size=self.size,
            smooth=self.smooth,
            zoom=self.initial_zoom,
            zoom_row=self.zoom_row,
            zoom_col=self.zoom_col,
            link_zoom=self.link_zoom,
            link_pan=self.link_pan,
            link_contrast=self.link_contrast,
            diff_mode=self.diff_mode,
            hidden_panels=list(self.hidden_panels),
            starred=[i for i, value in enumerate(self.starred) if value],
            show_panel_titles=self.show_panel_titles,
            panel_title_font_size=self.panel_title_font_size,
            display_bin=1,
        )
        clone.pixel_sizes = list(self.pixel_sizes)
        clone.load_state_dict(self.state_dict())
        clone.offline = quantized
        clone._export_light = True
        clone._save_state = True
        clone.export_enabled = False
        clone.export_status = ""
        clone.export_payload = b""
        clone.export_payload_id = ""
        clone.export_filename = ""
        clone._update_all_frames()
        return clone

    def load_state_dict(self, state):
        state = dict(state)
        if "starred" in state and isinstance(state["starred"], list) and len(state["starred"]) != int(self.n_images):
            state.pop("starred")
        if "hidden_panels" in state and isinstance(state["hidden_panels"], list):
            n_img = int(self.n_images)
            clean_set: set[int] = set()
            for value in state["hidden_panels"]:
                if isinstance(value, bool):
                    continue
                try:
                    idx = int(value)
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < n_img:
                    clean_set.add(idx)
            clean = sorted(clean_set)
            if len(clean) >= n_img:
                clean = clean[:-1]
            state["hidden_panels"] = clean
        for key, val in state.items():
            # Silent migrations for renamed keys in older saved state files.
            if key == "pixel_size_angstrom":
                key = "pixel_size"
            elif key == "canvas_size":
                key = "size"
            if key == "display_bin":
                self._display_bin = val
                continue
            if hasattr(self, key):
                setattr(self, key, val)

    @property
    def current_view(self) -> dict:
        """The field of view currently on screen, in image and calibrated coordinates.

        Every pan/zoom in the browser syncs the visible region back to Python
        (debounced ~100 ms), so after zooming into a feature you can capture
        exactly where you are for figure-making: ``view = w.current_view``.
        Feed ``view["box"]`` back as ``Show2D(data, view_box=view["box"])`` to
        reproduce the same crop later. The box is axis-aligned, so the two
        corners (row0, col0) and (row1, col1) define all four.

        Returns a dict with ``row0/col0/row1/col1`` (image pixel coordinates),
        ``height/width`` (visible extent in pixels), ``zoom``, ``box`` (the
        (row0, row1, col0, col1) tuple accepted by ``view_box=``), and when
        ``pixel_size > 0`` the calibrated ``*_cal`` extents plus ``unit``.
        """
        if len(self.view_box) == 4:
            row0, row1, col0, col1 = (float(v) for v in self.view_box)
        else:
            # No pan/zoom synced from the browser yet: derive the viewport
            # from the construction-time zoom center so the property is
            # usable immediately after Show2D(data, zoom=..., zoom_row=...).
            zoom = float(self.initial_zoom) if self.initial_zoom else 1.0
            center_row = float(self.zoom_row) if self.zoom_row is not None else self.height / 2
            center_col = float(self.zoom_col) if self.zoom_col is not None else self.width / 2
            half_h, half_w = self.height / (2 * zoom), self.width / (2 * zoom)
            row0, row1 = max(0.0, center_row - half_h), min(float(self.height), center_row + half_h)
            col0, col1 = max(0.0, center_col - half_w), min(float(self.width), center_col + half_w)
        view = {"row0": row0, "col0": col0, "row1": row1, "col1": col1,
                "height": row1 - row0, "width": col1 - col0,
                "zoom": float(self.initial_zoom),
                "box": (row0, row1, col0, col1)}
        if self.pixel_size > 0:
            scale = self.pixel_size
            view.update({"row0_cal": row0 * scale, "col0_cal": col0 * scale,
                         "row1_cal": row1 * scale, "col1_cal": col1 * scale,
                         "height_cal": (row1 - row0) * scale, "width_cal": (col1 - col0) * scale,
                         "unit": self.pixel_unit})
        return view

    @property
    def visible_panels(self) -> list[int]:
        """Zero-based image panel indices currently visible in the gallery."""
        hidden = set(self.hidden_panels)
        return [i for i in range(int(self.n_images)) if i not in hidden]

    @property
    def starred_panels(self) -> list[int]:
        """Zero-based image panel indices marked with a star."""
        return [i for i, value in enumerate(self.starred) if value]

    def set_hidden_panels(self, panels: Sequence[int | str] | int | str) -> Self:
        """Replace the hidden panel set by index or exact label.

        Hidden panels remain in the widget state and standalone HTML export, but
        they are collapsed from the gallery until restored. At least one panel
        must stay visible.
        """
        hidden = self._normalize_panel_refs(panels, allow_empty=True)
        if len(hidden) >= int(self.n_images):
            raise ValueError("set_hidden_panels would hide every panel; leave at least one visible")
        self.hidden_panels = sorted(hidden)
        return self

    def hide_panel(self, *panels: int | str) -> Self:
        """Hide one or more image panels by zero-based index or exact label."""
        to_hide = set(self.hidden_panels)
        to_hide.update(self._normalize_panel_refs(list(panels)))
        if len(to_hide) >= int(self.n_images):
            raise ValueError("hide_panel would hide every panel; leave at least one visible")
        self.hidden_panels = sorted(to_hide)
        return self

    def show_panel(self, *panels: int | str) -> Self:
        """Restore one or more hidden image panels by zero-based index or exact label."""
        to_show = set(self._normalize_panel_refs(list(panels)))
        self.hidden_panels = sorted(set(self.hidden_panels) - to_show)
        return self

    def show_all_panels(self) -> Self:
        """Restore every image panel in the gallery."""
        self.hidden_panels = []
        return self

    def set_starred_panels(self, panels: Sequence[int | str] | int | str) -> Self:
        """Replace the set of starred image panels by index or exact label."""
        starred = [0] * int(self.n_images)
        for idx in self._normalize_panel_refs(panels, allow_empty=True):
            starred[idx] = 1
        self.starred = starred
        return self

    def star_panel(self, panel: int | str) -> Self:
        """Mark an image panel with a star."""
        idx = self._resolve_panel_ref(panel)
        starred = list(self.starred)
        if len(starred) != int(self.n_images):
            starred = [0] * int(self.n_images)
        starred[idx] = 1
        self.starred = starred
        return self

    def unstar_panel(self, panel: int | str) -> Self:
        """Clear the star on an image panel."""
        idx = self._resolve_panel_ref(panel)
        starred = list(self.starred)
        if len(starred) != int(self.n_images):
            starred = [0] * int(self.n_images)
        starred[idx] = 0
        self.starred = starred
        return self

    def summary(self):
        """Print a human-readable snapshot of the widget's current state.

        Reports image dimensions and pixel size, data min/max/mean, display
        settings (colormap, contrast, scale, FFT), active ROIs and profile
        line, per-image rotations, and the most recent render timings.
        """
        lines = [self.title or "Show2D", "═" * 32]
        if self.n_images > 1:
            lines.append(f"Image:    {self.n_images}×{self.height}×{self.width} ({self.ncols} cols)")
        else:
            lines.append(f"Image:    {self.height}×{self.width}")
        if self.pixel_size > 0:
            ps = self.pixel_size
            if ps >= 10:
                lines[-1] += f" ({ps / 10:.2f} nm/px)"
            else:
                lines[-1] += f" ({ps:.2f} Å/px)"
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
        lines.append(f"Display:  {display}")
        if self.roi_active and self.roi_list:
            lines.append(f"ROI:      {len(self.roi_list)} region(s)")
        if self.profile_line:
            p0, p1 = self.profile_line[0], self.profile_line[1]
            lines.append(f"Profile:  ({p0['row']:.0f}, {p0['col']:.0f}) → ({p1['row']:.0f}, {p1['col']:.0f})")
        non_zero = [(i, r * 90) for i, r in enumerate(self.image_rotations) if r % 4 != 0]
        if non_zero:
            parts = [f"#{i}={deg}°" for i, deg in non_zero]
            lines.append(f"Rotated:  {', '.join(parts)}")
        rt = getattr(self, "render_total_ms", None)
        if rt is not None:
            pb = getattr(self, "render_python_build_ms", 0)
            wj = getattr(self, "render_wire_js_ms", 0)
            lines.append(f"Rendered: {rt} ms total (Python build {pb} ms, wire+JS {wj} ms)")
        else:
            lines.append("Rendered: (pending first browser paint)")
        print("\n".join(lines))

    def _compute_all_stats(self):
        """Compute statistics for all images (vectorized over all frames)."""
        # Vectorized reduction over (H, W) is faster than per-image loops
        # for large galleries (e.g. 12×4096×4096: 164ms vs 191ms).
        axes = (1, 2) if self._data.ndim == 3 else None
        self.stats_mean = np.mean(self._data, axis=axes).ravel().tolist()
        self.stats_min = np.min(self._data, axis=axes).ravel().tolist()
        self.stats_max = np.max(self._data, axis=axes).ravel().tolist()
        self.stats_std = np.std(self._data, axis=axes).ravel().tolist()

    def _update_all_frames(self):
        """Send display data to JS (possibly binned for large galleries)."""
        data = self._display_data if self._display_data is not None else self._data
        if self.offline:
            # Quantize to uint8 PER IMAGE (4x smaller than float32). Each panel uses
            # its own (min, max) so every panel gets the full 256 codes - a global
            # range would starve narrow-range panels and comb their histograms.
            # Display-only: the colormap reduces to 256 levels anyway.
            arr = np.ascontiguousarray(data, dtype=np.float32)  # (n, H, W)
            flat = arr.reshape(arr.shape[0], -1)
            out = np.empty(flat.shape, dtype=np.uint8)
            mins, maxs = [], []
            for i in range(flat.shape[0]):
                finite = flat[i][np.isfinite(flat[i])]
                lo = float(finite.min()) if finite.size else 0.0
                hi = float(finite.max()) if finite.size else 1.0
                rng = hi - lo if hi > lo else 1.0
                out[i] = np.clip((flat[i] - lo) * (255.0 / rng), 0, 255).astype(np.uint8)
                mins.append(lo)
                maxs.append(hi)
            self._offline_mins = mins
            self._offline_maxs = maxs
            self._offline_min = mins[0]  # back-compat scalars (single-image readers)
            self._offline_max = maxs[0]
            self.frame_bytes = _b64_safe(out.tobytes())
        else:
            self.frame_bytes = _b64_safe(data.tobytes())

    def _apply_rotations(self):
        """Re-rotate each displayed image from its original by ``image_rotations[i] * 90°``.

        This is purely a display-time reorientation of each 2D image via
        ``np.rot90`` — it is NOT scan rotation (which would rotate the
        scan grid in a 4D-STEM dataset). Originals are kept in
        ``_data_original`` so successive rotations compose from the
        unrotated source rather than accumulating interpolation error.
        Mixed shapes after rotation are center-padded to a common size.
        """
        # Materialize originals as independent copies only when a non-zero
        # rotation exists (they start as views into _data to avoid 800MB copy at init)
        has_rotation = any(
            (self.image_rotations[i] if i < len(self.image_rotations) else 0) % 4 != 0
            for i in range(len(self._data_original))
        )
        # No-rotation fast path: skip 30+ MB of redundant tobytes + stats recomputation
        # on every widget init.  The observer fires once when image_rotations = [0]*n
        # is assigned in __init__; without this guard that triggered a full frame
        # rebuild + stats recompute for a no-op.
        if not has_rotation and self._originals_are_views:
            return
        if self._originals_are_views and has_rotation:
            self._data_original = [img.copy() for img in self._data_original]
            self._originals_are_views = False
        rotated = []
        for i, orig in enumerate(self._data_original):
            k = self.image_rotations[i] if i < len(self.image_rotations) else 0
            k = k % 4
            if k == 0:
                rotated.append(orig)
            else:
                rotated.append(np.rot90(orig, k=k))
        # If shapes differ after rotation, center-pad all to max dims
        shapes = [img.shape for img in rotated]
        if len(set(shapes)) > 1:
            max_h = max(s[0] for s in shapes)
            max_w = max(s[1] for s in shapes)
            padded = []
            for img in rotated:
                h, w = img.shape
                pad_top = (max_h - h) // 2
                pad_bot = max_h - h - pad_top
                pad_left = (max_w - w) // 2
                pad_right = max_w - w - pad_left
                padded.append(np.pad(img, ((pad_top, pad_bot), (pad_left, pad_right)), mode="constant", constant_values=0))
            rotated = padded
        self._data = np.stack(rotated).astype(np.float32)
        # Recompute display data if binning is active
        if self._display_bin > 1:
            from quantem.widget.utils.array import bin2d
            self._display_data = bin2d(self._data, factor=self._display_bin, mode="mean")
        else:
            self._display_data = self._data
        display = self._display_data if self._display_data is not None else self._data
        self.height = int(display.shape[1])
        self.width = int(display.shape[2])
        self._compute_all_stats()
        self._update_all_frames()

    @traitlets.observe("image_rotations")
    def _on_image_rotations_changed(self, change):
        if hasattr(self, "_data_original"):
            self._apply_rotations()

    def rotate(self, idx: int, angle: int) -> Self:
        """Rotate image ``idx`` by ``angle`` degrees (CCW-positive, matches np.rot90).

        Rotation convention follows ``np.rot90``::

            angle | image_rotations | np.rot90 k | direction
            ------+-----------------+------------+----------
              90  |        1        |     1      | 90° CCW
             180  |        2        |     2      | 180°
             -90  |        3        |     3      | 90° CW
             360  |        0        |     0      | identity

        Parameters
        ----------
        idx : int
            Image index in the gallery (0-based).
        angle : int
            Rotation angle in degrees (must be a multiple of 90).
            Positive = counter-clockwise, negative = clockwise.

        Returns
        -------
        Self
        """
        if angle % 90 != 0:
            raise ValueError(f"Rotation angle must be a multiple of 90 (got {angle}). Use 0, 90, 180, 270, or -90, -180, -270.")
        if idx < 0 or idx >= self.n_images:
            raise IndexError(f"Image index {idx} out of range [0, {self.n_images})")
        k = (angle // 90) % 4
        rots = list(self.image_rotations)
        while len(rots) < self.n_images:
            rots.append(0)
        rots[idx] = (rots[idx] + k) % 4
        self.image_rotations = rots
        return self

    def _sample_profile(self, row0, col0, row1, col1):
        img = self._data[self.selected_idx]
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
                img[r1c, c1c] * cf * rf).astype(np.float32)

    def set_profile(self, start: tuple, end: tuple):
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

    def clear_profile(self):
        """Clear the current line profile."""
        self.profile_line = []

    def _upsert_selected_roi(self, updates: dict):
        rois = list(self.roi_list)
        color_cycle = ["#4fc3f7", "#81c784", "#ffb74d", "#ce93d8", "#ef5350", "#ffd54f", "#90a4ae", "#a1887f"]
        defaults = {
            "shape": "square",
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

    def add_roi(self, row: int | None = None, col: int | None = None, shape: str = "square") -> Self:
        with self.hold_sync():
            self.roi_selected_idx = -1
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
        with self.hold_sync():
            self._upsert_selected_roi({"shape": "circle", "row": int(row), "col": int(col), "radius": int(radius)})
        return self

    def roi_circle(self, radius: int = 10) -> Self:
        with self.hold_sync():
            self._upsert_selected_roi({"shape": "circle", "radius": int(radius)})
        return self

    def roi_square(self, half_size: int = 10) -> Self:
        with self.hold_sync():
            self._upsert_selected_roi({"shape": "square", "radius": int(half_size)})
        return self

    def roi_rectangle(self, width: int = 20, height: int = 10) -> Self:
        with self.hold_sync():
            self._upsert_selected_roi({"shape": "rectangle", "width": int(width), "height": int(height)})
        return self

    def roi_annular(self, inner: int = 5, outer: int = 10) -> Self:
        with self.hold_sync():
            self._upsert_selected_roi({"shape": "annular", "radius_inner": int(inner), "radius": int(outer)})
        return self

    @property
    def profile(self):
        """Get profile line endpoints as [(row0, col0), (row1, col1)] or [].

        Returns
        -------
        list of tuple
            Line endpoints in pixel coordinates, or empty list if no profile.
        """
        return [(p["row"], p["col"]) for p in self.profile_line]

    @property
    def profile_values(self):
        """Get intensity values along the profile line as a numpy array.

        Returns
        -------
        np.ndarray or None
            Float32 array of sampled intensities, or None if no profile.
        """
        if len(self.profile_line) < 2:
            return None
        p0, p1 = self.profile_line
        return self._sample_profile(p0["row"], p0["col"], p1["row"], p1["col"])

    @property
    def profile_distance(self):
        """Get total distance of the profile line in calibrated units.

        Returns
        -------
        float or None
            Distance in angstroms (if pixel_size > 0) or pixels.
            None if no profile line is set.
        """
        if len(self.profile_line) < 2:
            return None
        p0, p1 = self.profile_line
        dc = p1["col"] - p0["col"]
        dr = p1["row"] - p0["row"]
        dist_px = (dc**2 + dr**2) ** 0.5
        if self.pixel_size > 0:
            return dist_px * self.pixel_size
        return dist_px
