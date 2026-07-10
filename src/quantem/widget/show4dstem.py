"""
show4dstem: Fast interactive 4D-STEM viewer widget.

Single chunked-torch path on every device (CUDA / MPS / CPU). Reductions cast
uint16 → float32 in scan-row chunks bounded by _CHUNK_BYTE_BUDGET, so transient
memory stays the same regardless of total dataset size.

To reduce data size, bin k-space at the dataset level before viewing:

    dataset = dataset.bin(2, axes=(2, 3))  # 2x2 k-space binning
    widget = Show4DSTEM(dataset)
"""

import base64
import gc
import html
import io
import json
import math
import os
import pathlib
import tempfile
import threading
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Self, Sequence

if TYPE_CHECKING:
    from quantem.core.datastructures import Dataset4dstem

import anywidget
import numpy as np
import torch
import traitlets
from quantem.widget.utils.array import to_numpy
from quantem.widget.utils.state_io import (
    build_json_header,
    resolve_widget_version,
    save_state_file,
    unwrap_state_payload,
)
from quantem.widget.utils.static_fallback import StaticFallbackMixin
from quantem.widget.utils.ui import UiMode, resolve_ui_mode

# Cap transient chunk memory at ~600 MB regardless of detector size.
_CHUNK_BYTE_BUDGET = 600 * 1024 * 1024
# Sparse detector-mask gathers are fastest with smaller transient slabs; large
# slabs burn time in allocation/copy overhead on full no-bin CUDA compare grids.
_SPARSE_MASK_CHUNK_BYTE_BUDGET = 64 * 1024 * 1024


def _is_recoverable_allocation_error(exc: BaseException) -> bool:
    """True for memory pressure errors where a lighter fallback can continue."""
    if isinstance(exc, MemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message or "oom" in message


def _validate_device(device: str | None) -> tuple[str, Any]:
    """Resolve a compute device without importing quantem.core at module import."""
    from quantem.core.config import validate_device

    return validate_device(device)


def _format_memory(nbytes: int) -> str:
    if nbytes >= 1 << 30:
        return f"{nbytes / (1 << 30):.1f} GB"
    if nbytes >= 1 << 20:
        return f"{nbytes / (1 << 20):.0f} MB"
    if nbytes >= 1 << 10:
        return f"{nbytes / (1 << 10):.0f} KB"
    return f"{nbytes} B"


# ============================================================================
# Constants
# ============================================================================
DEFAULT_BF_RATIO = 0.125  # BF disk radius as fraction of detector size (1/8)
MIN_LOG_VALUE = 1e-10  # Minimum value for log scale to avoid log(0)
DEFAULT_VI_ROI_RATIO = 0.15  # Default VI ROI size as fraction of scan dimension


class Show4DSTEM(StaticFallbackMixin, anywidget.AnyWidget):
    """
    Fast interactive 4D-STEM viewer with advanced features.

    Optimized for speed with binary transfer and pre-normalization.
    Works with NumPy and PyTorch arrays.

    Parameters
    ----------
    data : Dataset4dstem or array_like
        Dataset4dstem object (calibration auto-extracted), 4D array
        of shape (scan_rows, scan_cols, det_rows, det_cols), or 5D array
        of shape (n_frames, scan_rows, scan_cols, det_rows, det_cols)
        for time-series or tilt-series data.
    scan_shape : tuple, optional
        If data is flattened (N, det_rows, det_cols), provide scan dimensions.
    sampling : tuple of 4 floats, optional
        Pixel size per axis ``(scan_row, scan_col, k_row, k_col)``. Scalar
        broadcasts to all four axes. Defaults to ``(1, 1, 1, 1)``.
        Auto-extracted from ``Dataset4dstem`` if not provided.
    units : list of 4 str, optional
        Unit string per axis. Common: ``["A", "A", "mrad", "mrad"]``.
        Defaults to ``["pixels"] * 4``. Auto-extracted from
        ``Dataset4dstem`` if not provided.
    center : tuple[float, float], optional
        (center_row, center_col) of the diffraction pattern in pixels.
        If not provided, defaults to detector center.
    bf_radius : float, optional
        Bright field disk radius in pixels. If not provided, estimated as 1/8 of detector size.
    precompute_virtual_images : bool, default True
        Precompute BF/ABF/LAADF/HAADF virtual images for preset switching.
    frame_dim_label : str, optional
        Label for the frame dimension when 5D data is provided.
        Defaults to "Frame". Common values: "Tilt", "Time", "Focus".
    view_mode : {"single", "multiple"}, default "single"
        Scientific layout mode. ``"single"`` shows one selected frame/dataset.
        ``"multiple"`` shows a grid of virtual images for the first ready
        frames/datasets while sharing the detector ROI and scan cursor with the
        existing diffraction panel. Legacy aliases ``"temporal"`` and
        ``"compare"`` are accepted as ``"single"`` and ``"multiple"``.
    compare_layout : {"side", "top"}, default "side"
        Frontend layout hint for ``view_mode="multiple"``. The current widget
        renders ``"side"`` as the default shared-DP plus virtual-image grid.
    compare_cols : int, default 0
        Number of columns in the compare virtual-image grid. ``0`` selects a
        responsive automatic layout.
    compare_grid_width_px : int, default 0
        Desktop width of the compare virtual-image grid in CSS pixels. ``0``
        uses the default responsive width. The frontend resize handle updates
        this value without changing the diffraction-panel size.
    compare_panel_gap_px : int, default 0
        Gap between compare virtual-image panels in CSS pixels. ``0`` gives a
        dense, edge-to-edge grid for browsing many datasets; larger values can
        be useful in reports.
    compare_max_panels : int, default 12
        Maximum ready frames/datasets included in the compare grid.
    compare_group_mode : {"paged", "all"}, default "paged"
        Compare-grid grouping behavior. ``"paged"`` shows one group of up to
        ``compare_max_panels`` panels at a time. ``"all"`` collapses all
        visible groups into one dense grid while still computing lazy datasets
        in page-sized batches.
    compare_dp_mode : {"average", "selected"}, default "average"
        Diffraction panel source in compare mode. ``"average"`` displays the
        mean diffraction pattern at the current scan position across visible
        ready compare panels. ``"selected"`` displays the active frame/dataset.
    compare_cache_pages : int, default 16
        Number of reduced compare-grid virtual-image pages to keep in host
        memory. This caches BF/ABF/ADF/HAADF thumbnails across page changes and
        is separate from raw 4D GPU residency.
    compare_cache_max_bytes : int, optional
        Host-memory cap for the reduced compare-grid page cache. Defaults to
        512 MiB. Set to ``0`` or ``compare_cache_pages=0`` to disable.
    ui_mode : {"interactive", "presentation", "report", "minimal"}, default "interactive"
        Preset for viewer chrome. Explicit ``show_*`` keyword arguments override
        the preset.
    show_title : bool, default True
        Show the top title row.
    show_controls : bool, default True
        Show the live control UI. Set ``False`` for a permanently clean display.
    controls_collapsed : bool, default False
        Start with the live control UI collapsed. Unlike
        ``show_controls=False``, Python can call ``expand_controls()`` later.
    show_stats : bool, default True
        Show mean/min/max/std readout bars under the DP, virtual image, and FFT.
    show_scale_bar : bool, default True
        Draw scale bars on the diffraction and virtual-image canvases.
    debug : bool, default False
        Show a compact frontend FPS/debug badge in the widget title row.
    save_state : bool, default False
        When False, saved notebooks omit heavy 4D buffers and keep a compact
        static preview for cold reopen. Set True only for small widgets that
        must reopen interactively without rerunning the kernel.
    notebook_preview_format : {"jpeg", "webp", "png"} or None, default "jpeg"
        Static preview format used when ``save_state=False``. Set to ``None``
        for live-only notebooks that should publish only the interactive view.
    notebook_preview_quality : int, default 88
        Lossy preview quality for JPEG/WebP, from 1 to 100. Ignored for PNG.
    notebook_preview_max_px : int, default 512
        Longest panel side for the saved-notebook preview.
    Examples
    --------
    >>> import numpy as np
    >>> from quantem.widget.show4dstem import Show4DSTEM

    4D NumPy array ``(scan_rows, scan_cols, det_rows, det_cols)``:

    >>> Show4DSTEM(np.random.rand(64, 64, 128, 128))

    PyTorch tensor (CPU or GPU):

    >>> import torch
    >>> Show4DSTEM(torch.rand(64, 64, 128, 128))

    With explicit calibration (real-space Å, k-space mrad):

    >>> Show4DSTEM(np.random.rand(64, 64, 128, 128),
    ...            sampling=(2.39, 2.39, 0.46, 0.46),
    ...            units=["A", "A", "mrad", "mrad"])

    quantem ``Dataset4dstem`` — calibration + units auto-extracted:

    >>> from quantem.core.datastructures import Dataset4dstem
    >>> ds = Dataset4dstem.from_array(np.random.rand(64, 64, 128, 128))
    >>> Show4DSTEM(ds)

    Flattened scan ``(N, det_rows, det_cols)`` with explicit scan shape:

    >>> Show4DSTEM(np.random.rand(4096, 128, 128), scan_shape=(64, 64))

    Custom BF disk center and radius (overrides auto-detection):

    >>> Show4DSTEM(np.random.rand(64, 64, 128, 128),
    ...            center=(64, 64), bf_radius=12)

    5D time-series or tilt-series ``(n_frames, scan_r, scan_c, det_r, det_c)``:

    >>> Show4DSTEM(np.random.rand(20, 64, 64, 128, 128), frame_dim_label="Tilt")

    Raster animation (scan path through 4D dataset):

    >>> w = Show4DSTEM(np.random.rand(64, 64, 128, 128))
    >>> w.raster(step=2, interval_ms=50)

    Static export to PDF or PNG (single panel or all four):

    >>> w = Show4DSTEM(np.random.rand(64, 64, 128, 128))
    >>> w.save_image("dp.pdf", view="diffraction")
    >>> w.save_image("all.pdf", view="all")
    """

    _esm = pathlib.Path(__file__).parent / "static" / "show4dstem.js"

    # Position in scan space
    widget_version = traitlets.Unicode("unknown").tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    show_title = traitlets.Bool(True).tag(sync=True)
    pos_row = traitlets.Int(0).tag(sync=True)
    pos_col = traitlets.Int(0).tag(sync=True)

    # Shape of scan space (for slider bounds)
    shape_rows = traitlets.Int(1).tag(sync=True)
    shape_cols = traitlets.Int(1).tag(sync=True)

    # Detector shape for frontend
    det_rows = traitlets.Int(1).tag(sync=True)
    det_cols = traitlets.Int(1).tag(sync=True)

    # Raw float32 frame as bytes (JS handles scale/colormap for real-time interactivity)
    frame_bytes = traitlets.Bytes(b"").tag(sync=True)

    # Global min/max for DP normalization (computed once from sampled frames)
    dp_global_min = traitlets.Float(0.0).tag(sync=True)
    dp_global_max = traitlets.Float(1.0).tag(sync=True)

    # =========================================================================
    # Detector Calibration (for presets and scale bar)
    # =========================================================================
    center_col = traitlets.Float(0.0).tag(sync=True)  # Detector center col
    center_row = traitlets.Float(0.0).tag(sync=True)  # Detector center row
    bf_radius = traitlets.Float(0.0).tag(sync=True)  # BF disk radius (pixels)

    # =========================================================================
    # ROI Drawing (for virtual imaging)
    # roi_radius is multi-purpose by mode:
    #   - circle: radius of circle
    #   - square: half-size (distance from center to edge)
    #   - annular: outer radius (roi_radius_inner = inner radius)
    #   - rect: uses roi_width/roi_height instead
    # =========================================================================
    roi_active = traitlets.Bool(False).tag(sync=True)
    roi_mode = traitlets.Unicode("point").tag(sync=True)
    roi_center_col = traitlets.Float(0.0).tag(sync=True)
    roi_center_row = traitlets.Float(0.0).tag(sync=True)
    # Compound trait for batched row+col updates (JS sends both at once, 1 observer fires)
    roi_center = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0]).tag(
        sync=True
    )
    roi_radius = traitlets.Float(10.0).tag(sync=True)
    roi_radius_inner = traitlets.Float(5.0).tag(sync=True)
    roi_width = traitlets.Float(20.0).tag(sync=True)
    roi_height = traitlets.Float(10.0).tag(sync=True)

    # =========================================================================
    # Virtual Image (ROI-based, updates as you drag ROI on DP)
    # =========================================================================
    virtual_image_bytes = traitlets.Bytes(b"").tag(
        sync=True
    )  # Raw float32 (JS computes stats + range)

    # Offline / browser-compute mode: ship a compact 4D stack so JS runs the
    # virtual-image and DP-from-ROI reductions with no Python kernel. Inline gzip
    # is only for small datasets; companion bslz4 supports full no-bin stacks and
    # lazy multi-volume 5D stacks.
    offline = traitlets.Bool(False).tag(sync=True)
    _offline_stack = traitlets.Bytes(b"").tag(sync=True)
    # Companion-file mode: instead of inlining the stack (which blocks mount on a
    # multi-hundred-MB HTML parse), write it to a sibling .gz and fetch it at
    # runtime. The widget-state JSON stays tiny -> instant mount + instant paint
    # from the inline initial virtual image; the big stack streams in the
    # background. Needs HTTP (fetch is CORS-blocked under file://).
    _offline_url = traitlets.Unicode("").tag(sync=True)
    # Chunked companion: JSON list of {coff, clen, startScan, nScan} describing
    # gzip chunks concatenated in the companion file. Lets a stack bigger than one
    # GPU buffer (or one JS ArrayBuffer) stream chunk-by-chunk into N GPU buffers.
    _offline_chunks = traitlets.Unicode("").tag(sync=True)
    # The stack is gzip-compressed (lossless). JS decompresses with
    # DecompressionStream before upload. Detector data compresses ~2-3x, so the
    # base64'd offline payload (and the HTML it sits in) shrinks ~2-3x -> far less
    # HTML/JSON parse on cold open, especially under file://.
    _offline_gzip = traitlets.Bool(False).tag(sync=True)
    # bslz4 mode: ship HDF5 bitshuffle+LZ4 companion chunks and decompress on the
    # GPU into uint8. JSON is either a single volume {base,chunks,...} or lazy
    # multi-volume {volumes:[{base,chunks,badPx}], ...}. The browser decode is
    # bit-exact to the uint8-clipped reference (verified).
    _offline_bslz4 = traitlets.Unicode("").tag(sync=True)
    # H5-source mode: the HTML points at a sibling float32 .h5 file (the merged data); the
    # JS reads it straight off disk via WebGPU (jsfive parse + GPU bitshuffle+LZ4 decode) -
    # NOTHING embedded, the data stays a file. Needs HTTP (fetch CORS-blocked under file://).
    # The "click HTML, GPU decompresses the merged H5, real Show4DSTEM renders it" path.
    _h5_url = traitlets.Unicode("").tag(sync=True)
    # Lazy mode: a sidecar bundle URL (radial profile + CoM + frame index + data files). The JS
    # derives the virtual image from the ~100 MB profile in VRAM and lazy-fetches CBED frames from
    # disk - nothing bulk-loads. Real-time scrub + detector with no 38 GB resident.
    _lazy_url = traitlets.Unicode("").tag(sync=True)
    # Hot/dead detector pixel indices (JSON list) auto-applied by the offline WebGPU
    # compute - mirrors CUDA load(apply_mask=True) so the browser data is filtered
    # automatically (no saturated pixel dominating the VI/DP).
    _offline_bad_px = traitlets.Unicode("").tag(sync=True)

    # Frontend-triggered standalone HTML export. Show4DSTEM exports package the
    # current loaded dataset representation; optional detector binning is applied
    # to that in-memory tensor instead of reloading from the original file.
    export_request = traitlets.Unicode("").tag(sync=True)
    export_status = traitlets.Unicode("").tag(sync=True)
    export_enabled = traitlets.Bool(True).tag(sync=True)
    export_payload = traitlets.Bytes(b"").tag(sync=True)
    export_payload_id = traitlets.Unicode("").tag(sync=True)
    export_filename = traitlets.Unicode("").tag(sync=True)

    # =========================================================================
    # VI ROI (real-space region selection for summed DP)
    # =========================================================================
    vi_roi_mode = traitlets.Unicode("off").tag(sync=True)  # "off", "circle", "rect"
    vi_roi_center_row = traitlets.Float(0.0).tag(sync=True)
    vi_roi_center_col = traitlets.Float(0.0).tag(sync=True)
    # Compound (row, col) trait — JS sets in one call; one observer fires; bytes
    # never compute against split-trait state (old col + new row, or vice versa).
    vi_roi_center = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0]).tag(
        sync=True
    )
    vi_roi_radius = traitlets.Float(5.0).tag(sync=True)
    vi_roi_width = traitlets.Float(10.0).tag(sync=True)
    vi_roi_height = traitlets.Float(10.0).tag(sync=True)
    # Reduction over scan positions inside vi_roi: mean is default (size-invariant DP),
    # sum scales with area (quantitative counts), max picks brightest position per detector pixel.
    vi_roi_reduce = traitlets.Unicode("mean").tag(sync=True)
    vi_roi_dp_bytes = traitlets.Bytes(b"").tag(sync=True)  # Reduced DP from VI ROI

    # =========================================================================
    # Scale Bar
    # =========================================================================
    pixel_size = traitlets.Float(1.0).tag(sync=True)  # real-space pixel size (col axis)
    pixel_unit = traitlets.Unicode("pixels").tag(sync=True)
    k_pixel_size = traitlets.Float(1.0).tag(sync=True)  # k-space pixel size (col axis)
    k_pixel_unit = traitlets.Unicode("pixels").tag(sync=True)
    k_calibrated = traitlets.Bool(False).tag(
        sync=True
    )  # True if k-space has real units

    # =========================================================================
    # Path Animation (programmatic crosshair control)
    # =========================================================================
    path_playing = traitlets.Bool(False).tag(sync=True)
    path_index = traitlets.Int(0).tag(sync=True)
    path_length = traitlets.Int(0).tag(sync=True)
    path_interval_ms = traitlets.Int(100).tag(sync=True)  # ms between frames
    path_loop = traitlets.Bool(True).tag(sync=True)  # loop when reaching end

    # =========================================================================
    # Auto-detection trigger (frontend sets to True, backend resets to False)
    # =========================================================================

    # =========================================================================
    # Statistics for display (mean, min, max, std)
    # =========================================================================
    # dp_stats and vi_stats are computed JS-side from frame_bytes / virtual_image_bytes.
    # Keeping them out of Python traits eliminates a 4-message comm race that produced
    # mismatched bytes/min/max on rapid preset/ROI changes.

    # =========================================================================
    # Display settings (synced for programmatic export parity)
    # =========================================================================
    _static_fallback_jpeg = traitlets.Unicode("").tag(sync=True)
    _static_fallback_mime = traitlets.Unicode("image/jpeg").tag(sync=True)

    dp_colormap = traitlets.Unicode("inferno").tag(sync=True)
    vi_colormap = traitlets.Unicode("inferno").tag(sync=True)
    fft_colormap = traitlets.Unicode("inferno").tag(sync=True)

    dp_scale_mode = traitlets.Unicode("linear").tag(sync=True)  # "linear" | "log"
    vi_scale_mode = traitlets.Unicode("linear").tag(sync=True)  # "linear" | "log"
    fft_scale_mode = traitlets.Unicode("linear").tag(sync=True)  # "linear" | "log"

    dp_vmin_pct = traitlets.Float(0.0).tag(sync=True)
    dp_vmax_pct = traitlets.Float(100.0).tag(sync=True)
    vi_vmin_pct = traitlets.Float(0.0).tag(sync=True)
    vi_vmax_pct = traitlets.Float(100.0).tag(sync=True)
    fft_vmin_pct = traitlets.Float(0.0).tag(sync=True)
    fft_vmax_pct = traitlets.Float(100.0).tag(sync=True)

    # Absolute intensity bounds (override percentile sliders when both set)
    dp_vmin = traitlets.Float(None, allow_none=True).tag(sync=True)
    dp_vmax = traitlets.Float(None, allow_none=True).tag(sync=True)
    vi_vmin = traitlets.Float(None, allow_none=True).tag(sync=True)
    vi_vmax = traitlets.Float(None, allow_none=True).tag(sync=True)

    fft_auto = traitlets.Bool(True).tag(sync=True)
    show_fft = traitlets.Bool(False).tag(sync=True)
    # Single-trait preset request: JS sets to "bf"/"abf"/"adf"/"haadf" → Python
    # observer calls apply_preset() which batches the 5 ROI trait writes
    # atomically. Avoids the JS-side ordering race where individual roi_mode/
    # radius/center traits would commit in separate comm messages.
    _preset_request = traitlets.Unicode("").tag(sync=True)
    fft_window = traitlets.Bool(True).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)
    controls_collapsed = traitlets.Bool(False).tag(sync=True)
    show_stats = traitlets.Bool(True).tag(sync=True)
    show_scale_bar = traitlets.Bool(True).tag(sync=True)
    debug = traitlets.Bool(False).tag(sync=True)
    panel_width_px = traitlets.Int(0).tag(sync=True)
    dp_show_colorbar = traitlets.Bool(False).tag(sync=True)
    # VI panel auto-contrast (1st/99th percentile clip) and CSS smoothing.
    # DP panel doesn't need either — Bragg spots are best read with nearest-
    # neighbor + the slider's percentile range.
    vi_auto_contrast = traitlets.Bool(False).tag(sync=True)
    vi_smooth = traitlets.Bool(False).tag(sync=True)

    # =========================================================================
    # Frame Animation (5D time/tilt series)
    # =========================================================================
    frame_idx = traitlets.Int(0).tag(sync=True)
    n_frames = traitlets.Int(1).tag(sync=True)
    frame_dim_label = traitlets.Unicode("Frame").tag(sync=True)
    frame_labels = traitlets.List(traitlets.Unicode(), []).tag(sync=True)
    frame_playing = traitlets.Bool(False).tag(sync=True)
    frame_loop = traitlets.Bool(True).tag(sync=True)
    frame_fps = traitlets.Float(5.0).tag(sync=True)
    frame_reverse = traitlets.Bool(False).tag(sync=True)
    frame_boomerang = traitlets.Bool(False).tag(sync=True)

    # Compare-grid mode: one shared diffraction panel, many synchronized virtual
    # images. The bytes are stacked float32 arrays with shape
    # (compare_panel_count, shape_rows, shape_cols).
    view_mode = traitlets.Unicode("single").tag(sync=True)
    compare_layout = traitlets.Unicode("side").tag(sync=True)
    compare_cols = traitlets.Int(0).tag(sync=True)
    compare_grid_width_px = traitlets.Int(0).tag(sync=True)
    compare_panel_gap_px = traitlets.Int(0).tag(sync=True)
    compare_max_panels = traitlets.Int(12).tag(sync=True)
    compare_group_mode = traitlets.Unicode("paged").tag(sync=True)
    compare_page_idx = traitlets.Int(0).tag(sync=True)
    compare_page_count = traitlets.Int(1).tag(sync=True)
    compare_dp_mode = traitlets.Unicode("average").tag(sync=True)
    compare_panel_order = traitlets.List(traitlets.Int(), default_value=[]).tag(
        sync=True
    )
    compare_hidden_panels = traitlets.List(traitlets.Int(), default_value=[]).tag(
        sync=True
    )
    compare_starred_panels = traitlets.List(traitlets.Int(), default_value=[]).tag(
        sync=True
    )
    compare_virtual_image_bytes = traitlets.Bytes(b"").tag(sync=True)
    compare_panel_count = traitlets.Int(0).tag(sync=True)
    compare_panel_indices = traitlets.List(traitlets.Int(), default_value=[]).tag(
        sync=True
    )
    compare_status = traitlets.Unicode("").tag(sync=True)
    gpu_memory_label = traitlets.Unicode("").tag(sync=True)
    memory_warning = traitlets.Unicode("").tag(sync=True)

    # Export (GIF)
    _gif_export_requested = traitlets.Bool(False).tag(sync=True)
    _gif_data = traitlets.Bytes(b"").tag(sync=True)
    _gif_metadata_json = traitlets.Unicode("").tag(sync=True)

    # Line Profile (for DP panel)
    profile_line = traitlets.List(traitlets.Dict()).tag(sync=True)
    profile_width = traitlets.Int(1).tag(sync=True)

    # =========================================================================
    @staticmethod
    def _normalise_view_mode(value: str) -> str:
        mode = str(value or "single").strip().lower().replace("-", "_")
        if mode in {"default", "normal", "temporal"}:
            mode = "single"
        elif mode in {"compare", "multi"}:
            mode = "multiple"
        if mode not in {"single", "multiple"}:
            raise ValueError(f"view_mode must be 'single' or 'multiple', got {value!r}")
        return mode

    @staticmethod
    def _normalise_compare_layout(value: str) -> str:
        layout = str(value or "side").strip().lower().replace("-", "_")
        if layout not in {"side", "top"}:
            raise ValueError(f"compare_layout must be 'side' or 'top', got {value!r}")
        return layout

    @staticmethod
    def _normalise_compare_dp_mode(value: str) -> str:
        mode = str(value or "average").strip().lower().replace("-", "_")
        aliases = {"avg": "average", "mean": "average", "current": "selected"}
        mode = aliases.get(mode, mode)
        if mode not in {"average", "selected"}:
            raise ValueError(
                f"compare_dp_mode must be 'average' or 'selected', got {value!r}"
            )
        return mode

    @staticmethod
    def _normalise_compare_group_mode(value: str) -> str:
        mode = str(value or "paged").strip().lower().replace("-", "_")
        aliases = {
            "page": "paged",
            "pages": "paged",
            "group": "paged",
            "groups": "paged",
            "collapse": "all",
            "collapsed": "all",
            "single": "all",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"paged", "all"}:
            raise ValueError(
                f"compare_group_mode must be 'paged' or 'all', got {value!r}"
            )
        return mode

    def __init__(
        self,
        data: "Dataset4dstem | np.ndarray",
        scan_shape: tuple[int, int] | None = None,
        sampling: tuple[float, ...] | list[float] | None = None,
        units: list[str] | tuple[str, ...] | None = None,
        center: tuple[float, float] | None = None,
        bf_radius: float | None = None,
        precompute_virtual_images: bool = True,
        frame_dim_label: str | None = None,
        frame_labels: list[str] | None = None,
        view_mode: str = "single",
        compare_layout: str = "side",
        compare_cols: int = 0,
        compare_grid_width_px: int = 0,
        compare_panel_gap_px: int = 0,
        compare_max_panels: int = 12,
        compare_group_mode: str = "paged",
        compare_dp_mode: str = "average",
        title: str = "",
        ui_mode: UiMode = "interactive",
        show_title: bool | None = None,
        offline: bool | None = False,
        data_url: str | None = None,
        offline_codec: str = "gzip",
        offline_dtype: str = "uint8",
        show_fft: bool = False,
        fft_window: bool = True,
        show_controls: bool | None = None,
        controls_collapsed: bool | None = None,
        show_stats: bool | None = None,
        show_scale_bar: bool | None = None,
        debug: bool = False,
        panel_width_px: int = 0,
        dp_vmin: float | None = None,
        dp_vmax: float | None = None,
        vi_vmin: float | None = None,
        vi_vmax: float | None = None,
        verbose: bool = False,
        state=None,
        backend: str | None = None,
        save_state: bool = False,
        notebook_preview_format: str | None = "jpeg",
        notebook_preview_quality: int = 88,
        notebook_preview_max_px: int = 512,
        page_budget: int | str | None = None,
        page_device=None,
        page_max_vram_fraction: float = 0.98,
        page_reserve_vram_bytes: int | None = None,
        page_max_vram_bytes: int | dict | None = None,
        compare_cache_pages: int = 16,
        compare_cache_max_bytes: int | None = 512 * 1024 * 1024,
        **kwargs,
    ):
        # save_state controls whether the heavy pixel buffers (the packed 4D
        # offline stack, export payload, gif) are persisted into the notebook's
        # metadata.widgets on save. Default False: a plain display embeds only
        # light traits + a static PNG of the virtual image, so a browse widget
        # does not bake a multi-hundred-MB stack into the .ipynb. Set True to
        # persist full interactive state so a reopened notebook restores the
        # widget (offline WebGPU browse) without a kernel.
        self._save_state = bool(save_state)
        self._configure_static_fallback(
            notebook_preview_format=notebook_preview_format,
            notebook_preview_quality=notebook_preview_quality,
            notebook_preview_max_px=notebook_preview_max_px,
        )
        ui = resolve_ui_mode(
            ui_mode,
            defaults={
                "show_title": True,
                "show_controls": True,
                "controls_collapsed": False,
                "show_stats": True,
                "show_scale_bar": True,
            },
            overrides={
                "show_title": show_title,
                "show_controls": show_controls,
                "controls_collapsed": controls_collapsed,
                "show_stats": show_stats,
                "show_scale_bar": show_scale_bar,
            },
        )
        show_title = bool(ui["show_title"])
        show_controls = bool(ui["show_controls"])
        controls_collapsed = bool(ui["controls_collapsed"])
        show_stats = bool(ui["show_stats"])
        show_scale_bar = bool(ui["show_scale_bar"])
        super().__init__(**kwargs)
        self._static_fallback_mime = self._static_fallback_mime_type()
        self.widget_version = resolve_widget_version()
        panel_width_px = int(panel_width_px)
        if panel_width_px < 0:
            raise ValueError(f"panel_width_px must be >= 0, got {panel_width_px}")
        self.panel_width_px = panel_width_px
        self.view_mode = self._normalise_view_mode(view_mode)
        self.compare_layout = self._normalise_compare_layout(compare_layout)
        self.compare_dp_mode = self._normalise_compare_dp_mode(compare_dp_mode)
        compare_cols = int(compare_cols)
        if compare_cols < 0:
            raise ValueError(f"compare_cols must be >= 0, got {compare_cols}")
        self.compare_cols = compare_cols
        compare_grid_width_px = int(compare_grid_width_px)
        if compare_grid_width_px < 0:
            raise ValueError(
                f"compare_grid_width_px must be >= 0, got {compare_grid_width_px}"
            )
        self.compare_grid_width_px = compare_grid_width_px
        compare_panel_gap_px = int(compare_panel_gap_px)
        if compare_panel_gap_px < 0:
            raise ValueError(
                f"compare_panel_gap_px must be >= 0, got {compare_panel_gap_px}"
            )
        self.compare_panel_gap_px = compare_panel_gap_px
        compare_max_panels = int(compare_max_panels)
        if compare_max_panels < 1:
            raise ValueError(
                f"compare_max_panels must be >= 1, got {compare_max_panels}"
            )
        self.compare_max_panels = compare_max_panels
        self.compare_group_mode = self._normalise_compare_group_mode(compare_group_mode)
        # Backend selector. ONLY two values:
        #   None  -> auto-pick Python compute (TorchBackend on torch
        #            tensors, MetalRawBackend on ChunkedFrames). Default.
        #   'web' -> kernel ships a packed stack via _offline_stack
        #            trait; JS Show4DSTEMCompute does all reductions in
        #            browser WebGPU. Kernel stays alive. Universal GPU
        #            compute (any modern GPU).
        # Legacy aliases (kept for one release):
        #   - 'browser' / 'webgpu' -> same as 'web'
        #   - offline=True          -> same as backend='web'
        if backend in ("web", "browser", "webgpu"):
            offline = True  # routes the rest of __init__ through the offline pack path
        elif backend is not None:
            raise ValueError(
                f"backend must be 'web' or None, got {backend!r}. "
                f"Python compute backend is auto-selected from the data type "
                f"(torch tensor -> TorchBackend; ChunkedFrames -> MetalRawBackend)."
            )
        self._backend_choice = backend
        offline_dtype = str(offline_dtype).strip().lower().replace("_", "")
        if offline_dtype in {"u8", "uint8"}:
            offline_dtype = "uint8"
        elif offline_dtype in {"u16", "uint16", "full", "exact"}:
            offline_dtype = "uint16"
        else:
            raise ValueError(
                f"offline_dtype must be 'uint8' or 'uint16', got {offline_dtype!r}. "
                "Use 'uint8' for compact browse data or 'uint16' to preserve detector counts."
            )
        self._offline_dtype = offline_dtype
        _t0 = time.perf_counter()
        _verbose = verbose

        _io_labels = None

        # Extract underlying array / tensor + auto-calibrate from Dataset input
        # (duck-typed via the dual-slot private attributes _tensor / _array).
        is_dataset5dstem_input = type(data).__name__ == "Dataset5dstem" and hasattr(
            data, "frame"
        )
        tensor = None if is_dataset5dstem_input else getattr(data, "_tensor", None)
        array = None if is_dataset5dstem_input else getattr(data, "_array", None)
        if tensor is not None or array is not None:
            if not title and getattr(data, "name", ""):
                title = str(data.name)
            if sampling is None:
                sampling = tuple(float(s) for s in data.sampling)
            if units is None:
                units = list(data.units)
            data = tensor if tensor is not None else array
        elif is_dataset5dstem_input:
            if not title and getattr(data, "name", ""):
                title = str(data.name)
            if sampling is None:
                sampling = tuple(float(s) for s in data.sampling)
            if units is None:
                units = list(data.units)

        # Resolve sampling + units (4 axes for 4D-STEM):
        # [scan_row, scan_col, k_row, k_col]. Scalar/None broadcast to (1, 1, 1, 1).
        if sampling is None:
            sampling = (1.0, 1.0, 1.0, 1.0)
        elif isinstance(sampling, (int, float)):
            sampling = (float(sampling),) * 4
        else:
            sampling = tuple(float(s) for s in sampling)
        if units is None:
            units = ["pixels"] * 4
        elif isinstance(units, str):
            units = [units] * 4
        else:
            units = [str(u) for u in units]

        self.title = title
        self.show_title = show_title
        self.pixel_size = sampling[1]  # scan_col axis (horizontal scale bar)
        self.pixel_unit = units[1] if len(units) > 1 else "pixels"
        self.k_pixel_size = sampling[3] if len(sampling) > 3 else 1.0
        self.k_pixel_unit = units[3] if len(units) > 3 else "pixels"
        # k-space considered calibrated when its unit is real (mrad, 1/Å, etc.).
        self.k_calibrated = self.k_pixel_unit not in ("pixels", "")
        self.show_fft = show_fft
        self.fft_window = fft_window
        self.show_controls = show_controls
        self.controls_collapsed = controls_collapsed
        self.show_stats = show_stats
        self.show_scale_bar = show_scale_bar
        self.debug = bool(debug)
        self.dp_vmin = dp_vmin
        self.dp_vmax = dp_vmax
        self.vi_vmin = vi_vmin
        self.vi_vmax = vi_vmax
        # Path animation (configured via set_path() or raster())
        self._path_points: list[tuple[int, int]] = []
        # Suppress per-trait recompute during apply_preset batch writes
        self._suppress_roi_recompute = False
        # Accept the io.load(...) output directly so `Show4DSTEM(load(path))` just
        # works on any backend. Unwrap a LoadResult NamedTuple, then wrap a raw MPS
        # chunked-load (MPSChunked4DSTEM) for the Metal compute path.
        if hasattr(data, "_fields") and "data" in getattr(data, "_fields", ()):
            data = data.data
        # Dataset5dstem is the CUDA/MPS-friendly 5D series wrapper. Keep it as a
        # frame-backed object instead of calling `.tensor`: sharded CUDA series may
        # hold each 18 GiB no-bin master on a different GPU, and `.tensor` would
        # gather everything onto one card.
        is_dataset5dstem = type(data).__name__ == "Dataset5dstem" and hasattr(
            data, "frame"
        )
        if hasattr(data, "chunks") and not getattr(data, "_is_gpu_frames", False):
            from quantem.widget.kernels.compute.mps import ChunkedFrames

            data = ChunkedFrames(
                data, row_prefix=bool(getattr(data, "row_prefix", False))
            )
        # cupy array (io.load default on CUDA) -> ZERO-COPY torch tensor on the same
        # GPU via dlpack. Without this, the fallback cp.asnumpy round-trips the whole
        # block to CPU and re-uploads (a 19.3 GB no-bin load -> ~58 GB transient and
        # an OOM kernel crash). dlpack keeps it on-device, no copy.
        if type(data).__module__.split(".")[0] == "cupy":
            data = torch.from_dlpack(data)
        # Torch tensor input keeps its device (lets user pin a specific GPU via
        # `data.cuda(1)`). NumPy / Dataset input gets default-validated device.
        if is_dataset5dstem:
            self._device = data.device
            self._data_pre = data
            data_np = None
        elif isinstance(data, torch.Tensor) or getattr(data, "_is_gpu_frames", False):
            # `_is_gpu_frames` lets a duck-typed GPU array (e.g. a chunk-backed
            # no-bin stack that can't be one tensor) take the GPU path without a
            # numpy round-trip. It must expose .shape/.dtype/.ndim/.device and
            # single-frame integer indexing.
            self._device = data.device
            self._data_pre = data
            data_np = None
        else:
            device_str, _ = _validate_device(None)
            self._device = torch.device(device_str)
            data_np = to_numpy(data)
            self._data_pre = None
            self._saturation_value = (
                65535
                if data_np.dtype == np.uint16
                else 255
                if data_np.dtype == np.uint8
                else None
            )
        # Handle dimensionality — 5D loads eagerly for instant frame switching
        # Resolve shape from whichever input path we took
        shape = (
            tuple(self._data_pre.shape) if self._data_pre is not None else data_np.shape
        )
        ndim = len(shape)
        _tc = time.perf_counter()
        if ndim == 5:
            self.n_frames = shape[0]
            self._scan_shape = (shape[1], shape[2])
            self._det_shape = (shape[3], shape[4])
        elif ndim == 3:
            self.n_frames = 1
            if scan_shape is not None:
                self._scan_shape = scan_shape
            else:
                n = shape[0]
                side = int(n**0.5)
                if side * side != n:
                    raise ValueError(
                        f"Cannot infer square scan_shape from N={n}. "
                        f"Provide scan_shape explicitly."
                    )
                self._scan_shape = (side, side)
            self._det_shape = (shape[1], shape[2])
        elif ndim == 4:
            self.n_frames = 1
            self._scan_shape = (shape[0], shape[1])
            self._det_shape = (shape[2], shape[3])
        else:
            raise ValueError(
                f"Show4DSTEM expects a 3D ((N, det_h, det_w) flat-scan), 4D ((scan_h, scan_w, det_h, det_w)), or 5D ((n_frames, scan_h, scan_w, det_h, det_w)) array. Got {ndim}D."
            )
        if self._data_pre is not None:
            self._data = (
                self._data_pre
                if is_dataset5dstem or self._data_pre.device == self._device
                else self._data_pre.to(self._device)
            )
            del self._data_pre
            self._dataset_page_config = None
            # Out-of-core dataset paging: with many 4D datasets behind the frame
            # slider (e.g. a 40-master folder), keeping every one resident fills
            # VRAM. page_budget caps how many stay on the GPU; switching frames
            # pages the target in and evicts the least-recently-used one to RAM
            # (via _frame_data -> Dataset5dstem.frame(i)). Only a Dataset5dstem
            # can page (it owns the per-frame device list); a plain 5D tensor
            # can't be partially offloaded, so page_budget is ignored there.
            if page_budget is not None and is_dataset5dstem:
                self._dataset_page_config = {
                    "vram_frames": page_budget,
                    "device": page_device,
                    "max_vram_fraction": page_max_vram_fraction,
                    "reserve_vram_bytes": page_reserve_vram_bytes,
                    "max_vram_bytes": page_max_vram_bytes,
                }
                if self.n_frames > 1:
                    self._data.page(**self._dataset_page_config)
        else:
            if not data_np.flags.writeable:
                # torch.from_numpy shares CPU memory; make read-only memmaps safe
                # before saturation cleanup mutates detector hot pixels in-place.
                data_np = np.array(data_np, copy=True)
            self._data = torch.from_numpy(data_np).to(self._device)
            # Saturation filter: zero detector pixels at full-scale (65535 / 255).
            # PyTorch lacks unsigned int comparison kernels, but uint16 viewed
            # as int16 has identical bytes (65535 → -1) and int16 comparison
            # works on every device. Apply in scan-row chunks so the transient
            # bool mask stays bounded (≤600 MB) and fits constrained-VRAM
            # devices (Mac 24 GB unified, etc.). View-write keeps native dtype.
            sat = getattr(self, "_saturation_value", None)
            view_dtype = (
                torch.int16
                if sat is not None and self._data.dtype == torch.uint16
                else torch.int8
                if sat is not None and self._data.dtype == torch.uint8
                else None
            )
            if view_dtype is not None:
                view = self._data.view(view_dtype).reshape(-1, *self._det_shape)
                rows = view.shape[0]
                # Bool mask transient = positions × det_h × det_w bytes; cap at budget.
                pos_per_chunk = max(
                    1,
                    _CHUNK_BYTE_BUDGET
                    // max(1, self._det_shape[0] * self._det_shape[1]),
                )
                for i in range(0, rows, pos_per_chunk):
                    chunk = view[i : i + pos_per_chunk]
                    chunk.masked_fill_(chunk == -1, 0)
        # Keep native dtype (uint8/uint16) to bound memory at ~ data_size.
        # Reductions cast in chunks (bounded transient).
        if _verbose:
            if str(self._device) == "mps":
                torch.mps.synchronize()
            n_bytes = (
                self._data.nbytes
                if hasattr(self._data, "nbytes")
                else self._data.element_size() * self._data.numel()
            )
            print(
                f"  to {self._device}: {time.perf_counter() - _tc:.2f}s ({n_bytes / 1e9:.1f} GB)"
            )

        self.shape_rows = self._scan_shape[0]
        self.shape_cols = self._scan_shape[1]
        self.det_rows = self._det_shape[0]
        self.det_cols = self._det_shape[1]
        # Initial position at center
        self.pos_row = self.shape_rows // 2
        self.pos_col = self.shape_cols // 2
        # Frame dimension label (for 5D time/tilt series UI)
        self.frame_dim_label = (
            frame_dim_label if frame_dim_label is not None else "Frame"
        )
        # Per-frame labels: explicit param > inferred > empty
        resolved_labels = frame_labels or _io_labels or []
        self._frame_labels = resolved_labels
        if resolved_labels:
            self.frame_labels = list(resolved_labels)
        # Histogram axis range — first frame is enough (JS does per-frame percentile clipping).
        # Cast to float for min/max reductions: PyTorch CUDA lacks integer min/max kernels,
        # and the first slice is tiny (144 KB at 192×192) so the cast is free.
        # .frame(0) (not [0]) so a paged Dataset5dstem brings frame 0 onto the GPU
        # for this reduction instead of handing back an offloaded CPU tensor.
        if type(self._data).__name__ == "Dataset5dstem":
            first_frame = self._data.frame(0)
        elif self._data.ndim == 5:
            first_frame = self._data[0]
        else:
            first_frame = self._data
        first_frame_sample = first_frame[0] if first_frame.ndim >= 3 else first_frame
        if not torch.is_floating_point(first_frame_sample):
            first_frame_sample = first_frame_sample.float()
        self.dp_global_min = max(float(first_frame_sample.min()), MIN_LOG_VALUE)
        self.dp_global_max = float(first_frame_sample.max())
        # Cache coordinate tensors for mask creation (avoid repeated torch.arange)
        self._det_row_coords = torch.arange(
            self.det_rows, device=self._device, dtype=torch.float32
        )[:, None]
        self._det_col_coords = torch.arange(
            self.det_cols, device=self._device, dtype=torch.float32
        )[None, :]
        self._scan_row_coords = torch.arange(
            self.shape_rows, device=self._device, dtype=torch.float32
        )[:, None]
        self._scan_col_coords = torch.arange(
            self.shape_cols, device=self._device, dtype=torch.float32
        )[None, :]
        # Setup center and BF radius
        det_size = min(self.det_rows, self.det_cols)
        if center is not None and bf_radius is not None:
            self.center_row = float(center[0])
            self.center_col = float(center[1])
            self.bf_radius = float(bf_radius)
        elif center is not None:
            self.center_row = float(center[0])
            self.center_col = float(center[1])
            self.bf_radius = det_size * DEFAULT_BF_RATIO
        elif bf_radius is not None:
            self.center_col = float(self.det_cols / 2)
            self.center_row = float(self.det_rows / 2)
            self.bf_radius = float(bf_radius)
        else:
            # Neither provided - auto-detect from data
            # Set defaults first (will be overwritten by auto-detect)
            self.center_col = float(self.det_cols / 2)
            self.center_row = float(self.det_rows / 2)
            self.bf_radius = det_size * DEFAULT_BF_RATIO
            # Auto-detect center and bf_radius from the data
            _tc = time.perf_counter()
            self.auto_detect_center(update_roi=False)
            if _verbose:
                print(f"  auto_detect_center: {time.perf_counter() - _tc:.2f}s")

        # Pre-compute and cache common virtual images (BF, ABF, ADF)
        # Each cache stores (bytes, stats) tuple
        self._cached_bf_virtual = None
        self._cached_abf_virtual = None
        self._cached_adf_virtual = None
        self._cached_haadf_virtual = None
        self._compare_virtual_page_cache: OrderedDict[
            tuple[Any, ...], tuple[bytes, tuple[int, ...], str, int]
        ] = OrderedDict()
        self._compare_virtual_page_cache_bytes = 0
        self._compare_diffraction_cache: OrderedDict[
            tuple[int, int, int, int, int], tuple[bytes, int]
        ] = OrderedDict()
        self._compare_diffraction_cache_bytes = 0
        self._compare_compute_lock = threading.RLock()
        self._compare_cache_lock = threading.RLock()
        self._compare_cache_generation = 0
        self._compare_cache_warm_stop: threading.Event | None = None
        self._compare_cache_warm_thread: threading.Thread | None = None
        self._compare_cache_warm_status = "idle"
        self._compare_cache_pages = max(0, int(compare_cache_pages))
        self._compare_cache_max_bytes = (
            None
            if compare_cache_max_bytes is None
            else max(0, int(compare_cache_max_bytes))
        )
        if precompute_virtual_images and self.n_frames == 1:
            self._precompute_common_virtual_images()

        # Update frame when position changes (scale/colormap handled in JS)
        self.observe(self._update_frame, names=["pos_row", "pos_col"])
        # Observe individual ROI params
        self.observe(
            self._on_roi_change,
            names=[
                "roi_center_col",
                "roi_center_row",
                "roi_radius",
                "roi_radius_inner",
                "roi_active",
                "roi_mode",
                "roi_width",
                "roi_height",
            ],
        )
        # Observe compound roi_center for batched updates from JS
        self.observe(self._on_roi_center_change, names=["roi_center"])
        # Invalidate precomputed virtual image caches when calibration changes
        self.observe(
            self._on_calibration_change, names=["center_row", "center_col", "bf_radius"]
        )

        # Default the ROI to the bright-field disk (circle at the detected center,
        # radius = BF radius) so the FIRST render shows a real virtual image. The
        # point-detector default (roi_mode="point") is a single detector pixel and
        # paints near-black until the user picks a preset — not a useful first view.
        # Batch the writes and suppress observers while the default ROI is
        # assembled. hold_trait_notifications() defers callbacks, but still
        # emits each changed ROI trait after the block; without the explicit
        # suppression a multi-panel widget recomputes the same compare grid once
        # per trait before the final explicit first render below.
        self._suppress_roi_recompute = True
        try:
            with self.hold_trait_notifications():
                self.roi_mode = "circle"
                self.roi_center_col = self.center_col
                self.roi_center_row = self.center_row
                self.roi_center = [self.center_row, self.center_col]
                self.roi_radius = float(max(1.0, self.bf_radius))
                self.roi_active = True
        finally:
            self._suppress_roi_recompute = False

        # Compute initial virtual image and frame (once, after all ROI traits are set)
        _tc = time.perf_counter()
        self._compute_virtual_image_from_roi()
        self._refresh_compare_virtual_images()
        self._update_frame()
        if _verbose:
            print(f"  virtual image + frame: {time.perf_counter() - _tc:.2f}s")

        # Path animation: observe index changes from frontend
        self.observe(self._on_path_index_change, names=["path_index"])
        self.observe(self._on_gif_export, names=["_gif_export_requested"])
        self.observe(self._on_export_request_change, names=["export_request"])

        # Frame animation (5D): observe frame_idx changes from frontend
        self.observe(self._on_frame_idx_change, names=["frame_idx"])
        self.observe(self._on_preset_request, names=["_preset_request"])
        self.observe(
            self._on_compare_config_change,
            names=[
                "view_mode",
                "compare_max_panels",
                "n_frames",
                "compare_group_mode",
                "compare_page_idx",
                "compare_panel_order",
                "compare_hidden_panels",
            ],
        )
        self.observe(self._on_compare_dp_mode_change, names=["compare_dp_mode"])

        # Auto-detect trigger: observe changes from frontend

        # VI ROI: observe changes for summed DP computation
        # Initialize VI ROI center to scan center with reasonable default sizes
        self.vi_roi_center_row = float(self.shape_rows / 2)
        self.vi_roi_center_col = float(self.shape_cols / 2)
        # Set initial ROI size based on scan dimension
        default_roi_size = max(
            3, min(self.shape_rows, self.shape_cols) * DEFAULT_VI_ROI_RATIO
        )
        self.vi_roi_radius = float(default_roi_size)
        self.vi_roi_width = float(default_roi_size * 2)
        self.vi_roi_height = float(default_roi_size)
        self.observe(
            self._on_vi_roi_change,
            names=[
                "vi_roi_mode",
                "vi_roi_center_row",
                "vi_roi_center_col",
                "vi_roi_radius",
                "vi_roi_width",
                "vi_roi_height",
                "vi_roi_reduce",
            ],
        )
        self.observe(self._on_vi_roi_center_change, names=["vi_roi_center"])

        # The frontend can mount a tick AFTER __init__ set virtual_image_bytes /
        # frame_bytes, missing those initial change events -> the virtual image stays
        # BLACK (stats 0) until the first ROI/cursor interaction pushes new bytes.
        # Re-send the view buffers on the next kernel-IOLoop tick (after the comm +
        # frontend are up) so the initial BF virtual image paints with no interaction.
        self._schedule_initial_view_sync()
        self._offline_codec = offline_codec
        self._pack_offline(offline, data_url)

        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(
                    json.loads(pathlib.Path(state).read_text()),
                    require_envelope=True,
                )
            else:
                state = unwrap_state_payload(state)
            self.load_state_dict(state)

        self._update_gpu_memory_status()

        if _verbose:
            shape = "x".join(str(s) for s in self._data.shape)
            # Spell out the backend, device, and WHERE the data physically lives,
            # so it's obvious whether a NumPy input went to the GPU. Key off the
            # compute class first (raw Metal / cupy), else the torch device.
            cls = self._compute.__class__.__name__
            backend, where = {
                "MetalCompute": ("Apple GPU (raw Metal)", "Apple unified memory"),
                "CudaKernelCompute": ("NVIDIA GPU (CUDA, cupy)", "GPU VRAM"),
            }.get(cls, (None, None))
            if backend is None:  # TorchCompute — backend depends on the torch device
                dev = str(self._device)
                if "cuda" in dev:
                    backend, where = "NVIDIA GPU (CUDA, torch)", "GPU VRAM"
                elif "mps" in dev:
                    backend, where = "Apple GPU (Metal, torch)", "Apple unified memory"
                else:
                    backend, where = "CPU (torch)", "system RAM"
            src = (
                "NumPy input uploaded to device"
                if data_np is not None
                else "kept on input device (no copy)"
            )
            print(f"Show4DSTEM ready in {time.perf_counter() - _t0:.2f}s")
            print(f"  shape   : {shape}")
            print(f"  backend : {backend}   device={self._device}")
            print(f"  data in : {where}   ({src})")

    def _schedule_initial_view_sync(self):
        """Re-emit the view byte-buffers after the frontend connects.

        anywidget syncs initial trait state on mount, but a heavy ``Bytes`` trait set
        during ``__init__`` (virtual_image_bytes, frame_bytes) is missed when the
        frontend mounts a tick later - the virtual image then stays black (stats 0)
        until the first interaction re-pushes it. Deferring a re-send to the kernel
        IOLoop guarantees the initial virtual image + diffraction pattern paint. No-op
        outside a Jupyter kernel (no running IOLoop).
        """
        try:
            from tornado.ioloop import IOLoop

            loop = IOLoop.current()
        except Exception:
            return

        def _resend():
            for name in (
                "virtual_image_bytes",
                "frame_bytes",
                "compare_virtual_image_bytes",
            ):
                try:
                    self.send_state(name)
                except Exception:
                    pass

        # Two delays: 0.3s covers a fast local mount, 1.5s covers a slow mount
        # (Colab, heavy install) where the frontend connects later.
        for delay in (0.3, 1.5):
            try:
                loop.call_later(delay, _resend)
            except Exception:
                pass

    # Soft guide on the packed stack, NOT a hard limit. The true
    # constraints are client RAM (decoded stack + GPU buffer ~= 2x this) and, for
    # the inline path, the browser's ~500 MB JS-string parse cap (gzip keeps the
    # embedded base64 under that). TESTED on Apple Silicon: full 512x512x48x48 (604 MB)
    # opens in 3 s; 512x512x96x96 (2.4 GB) inline opens in ~23 s. Bump if the
    # target machine has the RAM; use companion-fetch for the fast path on big data.
    _OFFLINE_BUDGET_BYTES = 2000 * 1024 * 1024

    def _pack_offline(self, offline: bool | None, data_url: str | None = None) -> None:
        """Ship the 4D stack to the browser for kernel-less WebGPU compute.

        JS runs the same masked-sum / DP-from-ROI reductions in the browser, so
        the dataset stays interactive with no Python kernel (live docs, shared
        offline HTML, Colab without a GPU). By default we pack the stack as
        **uint8** by clipping detector counts to [0, 255] for compact browse
        data. Use ``offline_dtype="uint16"`` to preserve detector counts.

        ``offline=None`` auto-enables under the byte budget; ``True`` forces it
        (and warns + skips if too big); ``False`` does nothing. 5D is supported
        only for bslz4 companion directories.
        """
        if self._data.ndim not in (4, 5):
            return
        # H5-source / lazy mode: the merged data stays files on disk, read by WebGPU at runtime.
        # No uint8 pack, nothing embedded - the JS fetches the sidecars/frames and decodes on GPU.
        if getattr(self, "_h5_url", "") or getattr(self, "_lazy_url", ""):
            self.offline = True
            return
        offline_dtype = getattr(self, "_offline_dtype", "uint8")
        bytes_per_pixel = 2 if offline_dtype == "uint16" else 1
        n_bytes = self._data.numel() * bytes_per_pixel
        # Companion mode bypasses the V8 string wall (data is fetched binary, never
        # a JS string), so its limit is client RAM (decoded + GPU ~= 2x), not the
        # ~500 MB inline parse cap. Give it a much higher budget.
        budget = (4000 * 1024 * 1024) if data_url else self._OFFLINE_BUDGET_BYTES
        if offline is None:
            offline = n_bytes <= budget
        if not offline:
            return
        # bslz4 codec: ship native bitshuffle+LZ4 bytes (~6x smaller than uint16),
        # decompress on the GPU to uint8. CHUNKED into <=1 GB GPU buffers, so it is
        # NOT budget-limited - a full 512x512x192x192 (9.6 GB uint8) streams fine.
        # Needs a companion directory (data_url).
        if data_url and getattr(self, "_offline_codec", "gzip") == "bslz4":
            if self._data.ndim == 5:
                self._pack_offline_bslz4_volumes(data_url)
            else:
                self._pack_offline_bslz4(data_url)
            return
        if data_url and self._data.ndim != 4:
            print(
                "  offline browser mode skipped: gzip companion mode only supports "
                "4D stacks; use offline_codec='bslz4' for 5D companion data"
            )
            return
        if n_bytes > budget:
            print(
                f"  offline browser mode skipped: stack is {n_bytes / 1e6:.0f} MB > "
                f"{budget / 1e6:.0f} MB budget; the kernel still works"
            )
            return
        # Direct clip to the target integer range - NOT global-linear scaling.
        # For uint8, pixels <=255 are exact and larger counts clip for compact
        # browse data. For uint16, real detector counts are preserved.
        import gzip

        if offline_dtype == "uint16":
            packed = np.clip(self._data.detach().to("cpu").numpy(), 0, 65535).astype(
                np.uint16
            )
        else:
            packed = np.clip(self._data.detach().to("cpu").numpy(), 0, 255).astype(
                np.uint8
            )
        target_shape = (
            (
                self.n_frames,
                self.shape_rows,
                self.shape_cols,
                self.det_rows,
                self.det_cols,
            )
            if self._data.ndim == 5
            else (self.shape_rows, self.shape_cols, self.det_rows, self.det_cols)
        )
        packed = np.ascontiguousarray(packed.reshape(target_shape))
        self._offline_gzip = True
        self.offline = True
        scan_cols, det_size = self.shape_cols, self.det_rows * self.det_cols
        # Chunk by scan-row ranges so each chunk stays under one GPU buffer
        # (~1 GB cap). A stack bigger than one buffer (e.g. 512x512x192x192)
        # then streams chunk-by-chunk into N buffers. Small stacks = a single chunk.
        chunk_bytes = 768 * 1024 * 1024
        rows_per = max(1, chunk_bytes // max(1, scan_cols * det_size * bytes_per_pixel))
        if data_url and packed.nbytes > chunk_bytes:
            out = pathlib.Path(data_url)
            out.parent.mkdir(parents=True, exist_ok=True)
            meta, blob, coff = [], bytearray(), 0
            for r0 in range(0, self.shape_rows, rows_per):
                r1 = min(self.shape_rows, r0 + rows_per)
                cz = gzip.compress(packed[r0:r1].tobytes(), compresslevel=6)
                meta.append(
                    {
                        "coff": coff,
                        "clen": len(cz),
                        "startScan": r0 * scan_cols,
                        "nScan": (r1 - r0) * scan_cols,
                    }
                )
                blob += cz
                coff += len(cz)
            out.write_bytes(blob)
            self._offline_url = data_url
            self._offline_chunks = json.dumps(meta)
            if getattr(self, "_verbose", True):
                print(
                    f"  offline companion (chunked): {out} {len(blob) / 1e6:.0f} MB gzip, {len(meta)} chunks; streams into {len(meta)} GPU buffers"
                )
        else:
            gz = gzip.compress(packed.tobytes(), compresslevel=6)
            if data_url:
                out = pathlib.Path(data_url)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(gz)
                self._offline_url = data_url
                if getattr(self, "_verbose", True):
                    print(
                        f"  offline companion: wrote {out} ({len(gz) / 1e6:.0f} MB gzip)"
                    )
            else:
                self._offline_stack = gz  # inline single self-contained file

    def export_html(
        self,
        path: str | pathlib.Path | None = None,
        *,
        title: str | None = None,
        mode: str = "single",
        encoding: str | None = None,
        downsample: int | None = None,
        dtype: str = "uint8",
        det_bin: int = 1,
        scan_bin: int = 1,
        real_space_bin: int | None = None,
        export_kind: str = "interactive",
        dataset_scope: str = "unhidden",
    ) -> pathlib.Path:
        """Write a standalone HTML viewer.

        ``export_kind="interactive"`` packages raw 4D data into the standalone
        browser-compute widget. ``export_kind="report"`` writes a compact static
        virtual-image report with no raw 4D payload, which is the safe export path
        for lazy folder-backed viewers. ``det_bin`` bins detector pixels by mean
        over ``det_bin x det_bin`` blocks. ``scan_bin`` (or alias
        ``real_space_bin``) bins scan pixels by mean over ``scan_bin x scan_bin``
        blocks. ``dtype`` may be ``"uint8"`` or ``"uint16"``.
        """
        if self._data is None:
            raise ValueError(
                "Cannot export HTML after free(); rebuild the widget first."
            )
        dtype, det_bin, scan_bin = self._normalise_html_export_options(
            mode=mode,
            encoding=encoding,
            downsample=downsample,
            dtype=dtype,
            det_bin=det_bin,
            scan_bin=scan_bin,
            real_space_bin=real_space_bin,
        )
        kind = self._normalise_html_export_kind(export_kind)
        export_path = (
            pathlib.Path(path)
            if path is not None
            else self._default_html_export_path(
                dtype,
                det_bin,
                scan_bin,
                export_kind=kind,
            )
        )
        if kind == "report":
            self._write_html_report_export(
                export_path,
                dtype=dtype,
                det_bin=det_bin,
                scan_bin=scan_bin,
                dataset_scope=dataset_scope,
                title=title,
            )
        else:
            self._write_html_export(
                export_path,
                dtype=dtype,
                det_bin=det_bin,
                scan_bin=scan_bin,
                title=title,
            )
        size_mb = export_path.stat().st_size / (1024 * 1024)
        self.export_status = (
            f"Exported {export_path.name} "
            f"({size_mb:.1f} MB, {self._export_mode_label(dtype, det_bin, scan_bin, export_kind=kind)})"
        )
        return export_path

    def _on_export_request_change(self, change: dict) -> None:
        raw = str(change.get("new") or "")
        if not raw:
            return
        try:
            payload = json.loads(raw)
            mode = str(payload.get("mode", "uint8-bin1"))
            if mode == "clear":
                self.export_payload = b""
                self.export_payload_id = ""
                self.export_filename = ""
                return
            export_kind = self._normalise_html_export_kind(
                payload.get("export_kind", payload.get("kind", "interactive"))
            )
            dtype, det_bin, scan_bin = self._normalise_html_export_options(
                mode=mode,
                encoding=payload.get("encoding"),
                downsample=payload.get("downsample"),
                dtype=str(payload.get("dtype", "uint8")),
                det_bin=int(payload.get("det_bin", 1)),
                scan_bin=int(payload.get("scan_bin", 1)),
                real_space_bin=payload.get("real_space_bin"),
            )
            if payload.get("download"):
                filename = str(
                    payload.get("filename")
                    or self._default_html_export_path(
                        dtype, det_bin, scan_bin, export_kind=export_kind
                    ).name
                )
                request_id = str(payload.get("id") or "")
                self.export_status = f"Preparing {filename}..."
                dataset_scope = str(payload.get("dataset_scope", "unhidden"))
                if export_kind == "report":
                    html = self._html_report_export_bytes(
                        dtype=dtype,
                        det_bin=det_bin,
                        scan_bin=scan_bin,
                        dataset_scope=dataset_scope,
                    )
                else:
                    html = self._html_export_bytes(
                        dtype=dtype, det_bin=det_bin, scan_bin=scan_bin
                    )
                self.export_filename = filename
                self.export_payload = html
                self.export_payload_id = request_id
                size_mb = len(html) / (1024 * 1024)
                self.export_status = (
                    f"Ready {filename} "
                    f"({size_mb:.1f} MB, {self._export_mode_label(dtype, det_bin, scan_bin, export_kind=export_kind)})"
                )
            else:
                self.export_status = f"Exporting {mode} HTML..."
                self.export_html(
                    dtype=dtype,
                    det_bin=det_bin,
                    scan_bin=scan_bin,
                    export_kind=export_kind,
                )
        except Exception as exc:
            self.export_status = f"Export failed: {exc}"

    def _normalise_html_export_options(
        self,
        *,
        mode: str = "single",
        encoding: str | None = None,
        downsample: int | str | None = None,
        dtype: str = "uint8",
        det_bin: int = 1,
        scan_bin: int = 1,
        real_space_bin: int | str | None = None,
    ) -> tuple[str, int, int]:
        raw_mode = str(mode or "single").strip().lower().replace("_", "-")
        if "-bin" in raw_mode:
            parsed_dtype, parsed_bin = self._parse_export_mode(raw_mode)
            raw_mode = "single"
            dtype = parsed_dtype
            det_bin = parsed_bin
        if raw_mode not in {"single", "folder"}:
            raise ValueError(
                "Show4DSTEM HTML export supports mode='single' or mode='folder'"
            )
        if raw_mode == "folder" and not self._offline_bslz4:
            raise ValueError(
                "folder export is only available for Show4DSTEM widgets with a companion data folder"
            )

        if encoding is not None:
            raw_encoding = str(encoding or "uint8").strip().lower().replace("_", "-")
            if raw_encoding in {"full", "exact", "uint16", "u16"}:
                dtype = "uint16"
            elif raw_encoding in {"uint8", "u8"}:
                dtype = "uint8"
            elif raw_encoding == "auto":
                dtype = str(dtype or "uint8")
            else:
                raise ValueError(
                    f"unknown Show4DSTEM export encoding {encoding!r}; expected 'full', 'uint8', or 'auto'"
                )

        if downsample not in (None, "", 0, "0"):
            requested_bin = int(downsample)
            if det_bin not in (1, requested_bin):
                raise ValueError(
                    "Specify either downsample or det_bin, not conflicting values"
                )
            det_bin = requested_bin
        if real_space_bin not in (None, "", 0, "0"):
            requested_scan_bin = int(real_space_bin)
            if scan_bin not in (1, requested_scan_bin):
                raise ValueError(
                    "Specify either real_space_bin or scan_bin, not conflicting values"
                )
            scan_bin = requested_scan_bin
        dtype, det_bin = self._parse_export_mode(f"{dtype}-bin{det_bin}")
        scan_bin = int(scan_bin)
        if scan_bin not in (1, 2, 4, 8):
            raise ValueError(f"scan_bin must be 1, 2, 4, or 8, got {scan_bin}")
        if self.shape_rows % scan_bin != 0 or self.shape_cols % scan_bin != 0:
            raise ValueError(
                f"Scan shape {self.shape_rows}x{self.shape_cols} is not divisible by scan_bin={scan_bin}"
            )
        return dtype, det_bin, scan_bin

    def _normalise_html_export_kind(self, export_kind: Any) -> str:
        kind = str(export_kind or "interactive").strip().lower().replace("_", "-")
        if kind in {"interactive", "raw", "raw-4d", "widget"}:
            return "interactive"
        if kind in {"report", "static", "summary"}:
            return "report"
        raise ValueError("export_kind must be 'interactive' or 'report'")

    def _parse_export_mode(self, mode: str) -> tuple[str, int]:
        parts = mode.split("-bin")
        dtype = parts[0].lower()
        if dtype in ("u8", "uint8"):
            dtype = "uint8"
        elif dtype in ("u16", "uint16"):
            dtype = "uint16"
        else:
            raise ValueError(f"unknown export dtype {dtype!r}")
        det_bin = int(parts[1]) if len(parts) == 2 and parts[1] else 1
        if det_bin not in (1, 2, 4, 8):
            raise ValueError(f"det_bin must be 1, 2, 4, or 8, got {det_bin}")
        return dtype, det_bin

    def _export_mode_label(
        self,
        dtype: str,
        det_bin: int,
        scan_bin: int = 1,
        *,
        export_kind: str = "interactive",
    ) -> str:
        label = "uint8" if dtype == "uint8" else "uint16"
        parts = [("report" if export_kind == "report" else "interactive raw 4D"), label]
        if scan_bin > 1:
            parts.append(f"scan bin {scan_bin}x")
        if det_bin > 1:
            parts.append(f"detector bin {det_bin}x")
        return ", ".join(parts)

    def _default_html_export_path(
        self,
        dtype: str,
        det_bin: int,
        scan_bin: int = 1,
        *,
        export_kind: str = "interactive",
    ) -> pathlib.Path:
        label = self.title.strip() or "show4dstem"
        slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        if not slug:
            slug = "show4dstem"
        shape = (
            f"{self.n_frames}x{self.shape_rows // scan_bin}x{self.shape_cols // scan_bin}"
            f"x{self.det_rows // det_bin}x{self.det_cols // det_bin}"
            if self.n_frames > 1
            else (
                f"{self.shape_rows // scan_bin}x{self.shape_cols // scan_bin}"
                f"x{self.det_rows // det_bin}x{self.det_cols // det_bin}"
            )
        )
        prefix = "report" if export_kind == "report" else dtype
        suffix = f"{prefix}_rbin{scan_bin}_kbin{det_bin}"
        return pathlib.Path.cwd() / f"{slug}_{shape}_{suffix}.html"

    def _write_html_export(
        self,
        path: str | pathlib.Path,
        *,
        dtype: str,
        det_bin: int,
        scan_bin: int = 1,
        title: str | None = None,
    ) -> pathlib.Path:
        from ipywidgets.embed import dependency_state, embed_minimal_html

        from .export import ensure_mobile_viewport

        out = pathlib.Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if self._offline_bslz4:
            if det_bin != 1 or scan_bin != 1:
                raise ValueError(
                    "Binned interactive raw export is not available for bslz4 companion-folder "
                    "Show4DSTEM exports. Use export_kind='report' for a compact page-aware "
                    "sharing export, or export an in-memory widget for binned raw 4D HTML."
                )
            # Already packed to a bslz4 companion (the multi-volume / large path):
            # the data lives in the companion dir, so re-packing via a clone would
            # try to reshape one volume into the full 5D stack and fail. Embed this
            # widget as-is - it already references the companion via data_url - and
            # only disable its export button in the standalone copy.
            prev_enabled = self.export_enabled
            prev_save_state = self._save_state
            self.export_enabled = False
            self._save_state = True
            try:
                embed_minimal_html(
                    str(out),
                    views=[self],
                    title=title or self.title or "Show4DSTEM",
                    drop_defaults=False,
                    state=dependency_state([self], drop_defaults=False),
                )
            finally:
                self.export_enabled = prev_enabled
                self._save_state = prev_save_state
            ensure_mobile_viewport(out)
            return out
        export_widget = self._clone_for_html_export(
            dtype=dtype, det_bin=det_bin, scan_bin=scan_bin
        )
        try:
            embed_minimal_html(
                str(out),
                views=[export_widget],
                title=title or self.title or "Show4DSTEM",
                drop_defaults=False,
                state=dependency_state([export_widget], drop_defaults=False),
            )
        finally:
            export_widget.close()
        ensure_mobile_viewport(out)
        return out

    def _html_export_bytes(
        self, *, dtype: str, det_bin: int, scan_bin: int = 1
    ) -> bytes:
        with tempfile.TemporaryDirectory(prefix="show4dstem-export-") as tmp:
            path = (
                pathlib.Path(tmp)
                / self._default_html_export_path(dtype, det_bin, scan_bin).name
            )
            self._write_html_export(
                path, dtype=dtype, det_bin=det_bin, scan_bin=scan_bin
            )
            return path.read_bytes()

    def _clone_for_html_export(
        self, *, dtype: str, det_bin: int, scan_bin: int = 1
    ) -> Self:
        data = self._export_data_array(dtype=dtype, det_bin=det_bin, scan_bin=scan_bin)
        k_scale = float(det_bin)
        scan_scale = float(scan_bin)
        sampling = (
            self.pixel_size * scan_scale,
            self.pixel_size * scan_scale,
            self.k_pixel_size * k_scale,
            self.k_pixel_size * k_scale,
        )
        units = [self.pixel_unit, self.pixel_unit, self.k_pixel_unit, self.k_pixel_unit]
        center = (self.center_row / k_scale, self.center_col / k_scale)
        clone = type(self)(
            data,
            sampling=sampling,
            units=units,
            center=center,
            bf_radius=max(1.0, self.bf_radius / k_scale),
            precompute_virtual_images=False,
            frame_dim_label=self.frame_dim_label,
            frame_labels=list(self.frame_labels),
            title=self.title,
            show_title=self.show_title,
            offline=False,
            show_fft=self.show_fft,
            fft_window=self.fft_window,
            show_controls=self.show_controls,
            controls_collapsed=self.controls_collapsed,
            show_stats=self.show_stats,
            show_scale_bar=self.show_scale_bar,
            debug=self.debug,
            view_mode=self.view_mode,
            compare_layout=self.compare_layout,
            compare_cols=self.compare_cols,
            compare_grid_width_px=self.compare_grid_width_px,
            compare_panel_gap_px=self.compare_panel_gap_px,
            compare_max_panels=self.compare_max_panels,
            compare_group_mode=self.compare_group_mode,
            compare_dp_mode=self.compare_dp_mode,
            verbose=False,
        )
        clone.load_state_dict(self._export_state_for_bin(det_bin, scan_bin=scan_bin))
        clone._pack_export_inline(dtype=dtype)
        clone.export_enabled = False
        clone.export_status = ""
        clone.export_payload = b""
        clone.export_payload_id = ""
        clone.export_filename = ""
        clone._save_state = True
        return clone

    def _export_data_array(
        self, *, dtype: str, det_bin: int, scan_bin: int = 1
    ) -> np.ndarray:
        if self.det_rows % det_bin != 0 or self.det_cols % det_bin != 0:
            raise ValueError(
                f"Detector shape {self.det_rows}x{self.det_cols} is not divisible by det_bin={det_bin}"
            )
        if self.shape_rows % scan_bin != 0 or self.shape_cols % scan_bin != 0:
            raise ValueError(
                f"Scan shape {self.shape_rows}x{self.shape_cols} is not divisible by scan_bin={scan_bin}"
            )
        data = self._data
        if dtype not in {"uint8", "uint16"}:
            raise ValueError(f"unknown export dtype {dtype!r}")

        def _finish_export_chunk(
            chunk: np.ndarray, *, round_values: bool = True
        ) -> np.ndarray:
            if round_values:
                chunk = np.round(chunk)
            if dtype == "uint8":
                return np.clip(chunk, 0, 255).astype(np.uint8, copy=False)
            return np.clip(chunk, 0, 65535).astype(np.uint16, copy=False)

        def _tensor_frame_to_export_array(frame: torch.Tensor) -> np.ndarray:
            frame4 = (
                frame
                if frame.ndim == 4
                else frame.reshape(
                    self.shape_rows, self.shape_cols, self.det_rows, self.det_cols
                )
            )
            if det_bin <= 1:
                arr = frame4.detach().to("cpu").numpy()
                arr = Show4DSTEM._mean_scan_bin_array(arr, scan_bin)
                return _finish_export_chunk(arr, round_values=scan_bin > 1)
            rows_per = self._chunk_rows()
            chunks: list[np.ndarray] = []
            for r0 in range(0, self.shape_rows, rows_per):
                slab = frame4[r0 : r0 + rows_per]
                if not torch.is_floating_point(slab):
                    slab = slab.float()
                binned = slab.reshape(
                    slab.shape[0],
                    self.shape_cols,
                    self.det_rows // det_bin,
                    det_bin,
                    self.det_cols // det_bin,
                    det_bin,
                ).mean(dim=(3, 5))
                chunks.append(binned.detach().to("cpu").numpy())
            arr = np.concatenate(chunks, axis=0)
            arr = Show4DSTEM._mean_scan_bin_array(arr, scan_bin)
            return _finish_export_chunk(arr)

        if type(data).__name__ == "Dataset5dstem" and hasattr(data, "frames"):
            frames = [
                _tensor_frame_to_export_array(data[i]) for i in range(self.n_frames)
            ]
            arr = np.stack(frames, axis=0) if self.n_frames > 1 else frames[0]
            return np.ascontiguousarray(arr)
        if isinstance(data, torch.Tensor):
            if self.n_frames > 1:
                arr = np.stack(
                    [
                        _tensor_frame_to_export_array(data[i])
                        for i in range(self.n_frames)
                    ],
                    axis=0,
                )
            else:
                arr = _tensor_frame_to_export_array(data)
            return np.ascontiguousarray(arr)
        elif hasattr(data, "datasets"):
            datasets = list(data.datasets[: self.n_frames])
            missing = [idx for idx, dataset in enumerate(datasets) if dataset is None]
            if len(datasets) < self.n_frames or missing:
                raise ValueError(
                    "Cannot export lazy multi-dataset Show4DSTEM before the "
                    f"first {self.n_frames} dataset(s) are ready"
                )
            volumes = []
            for dataset in datasets:
                if hasattr(dataset, "chunks"):
                    flat = np.concatenate(
                        [np.asarray(chunk) for chunk in dataset.chunks], axis=0
                    )
                else:
                    flat = np.asarray(dataset)
                volumes.append(
                    flat.reshape(
                        self.shape_rows, self.shape_cols, self.det_rows, self.det_cols
                    )
                )
            arr = np.stack(volumes, axis=0)
        elif hasattr(data, "chunks"):
            # MacBook MPS path: data is a zero-copy ChunkedFrames (numpy views over
            # Metal buffers), not a tensor. Materialize the flat stack by concatenating
            # the chunks; the reshape below restores the scan grid.
            arr = np.concatenate([np.asarray(chunk) for chunk in data.chunks], axis=0)
        else:
            arr = np.asarray(data)
        target_shape = (
            (
                self.n_frames,
                self.shape_rows,
                self.shape_cols,
                self.det_rows,
                self.det_cols,
            )
            if self.n_frames > 1
            else (self.shape_rows, self.shape_cols, self.det_rows, self.det_cols)
        )
        arr = np.ascontiguousarray(arr.reshape(target_shape))
        if det_bin > 1:
            # MEAN, not sum: a sum over det_bin^2 pixels (64 at bin 8) overflows the
            # uint8 export ceiling on any real-count detector and the BF disk clips
            # flat. Mean keeps each binned pixel within the raw per-pixel range, so
            # it always fits uint8. Round so the average count is the nearest integer.
            if arr.ndim == 5:
                nf, sr, sc, dr, dc = arr.shape
                arr = arr.reshape(
                    nf, sr, sc, dr // det_bin, det_bin, dc // det_bin, det_bin
                ).mean(axis=(4, 6))
            else:
                sr, sc, dr, dc = arr.shape
                arr = arr.reshape(
                    sr, sc, dr // det_bin, det_bin, dc // det_bin, det_bin
                ).mean(axis=(3, 5))
        arr = Show4DSTEM._mean_scan_bin_array(arr, scan_bin)
        arr = np.round(arr) if det_bin > 1 or scan_bin > 1 else arr
        if dtype == "uint8":
            arr = np.clip(arr, 0, 255).astype(np.uint8, copy=False)
        elif dtype == "uint16":
            arr = np.clip(arr, 0, 65535).astype(np.uint16, copy=False)
        return np.ascontiguousarray(arr)

    @staticmethod
    def _mean_scan_bin_array(arr: np.ndarray, scan_bin: int) -> np.ndarray:
        arr = np.asarray(arr)
        if scan_bin <= 1:
            return arr
        if arr.ndim == 5:
            nf, sr, sc, dr, dc = arr.shape
            return arr.reshape(
                nf, sr // scan_bin, scan_bin, sc // scan_bin, scan_bin, dr, dc
            ).mean(axis=(2, 4))
        if arr.ndim == 4:
            sr, sc, dr, dc = arr.shape
            return arr.reshape(
                sr // scan_bin, scan_bin, sc // scan_bin, scan_bin, dr, dc
            ).mean(axis=(1, 3))
        if arr.ndim == 2:
            sr, sc = arr.shape
            return arr.reshape(sr // scan_bin, scan_bin, sc // scan_bin, scan_bin).mean(
                axis=(1, 3)
            )
        raise ValueError(f"Cannot scan-bin array with shape {arr.shape}")

    @staticmethod
    def _mean_detector_bin_image(arr: np.ndarray, det_bin: int) -> np.ndarray:
        arr = np.asarray(arr)
        if det_bin <= 1:
            return arr
        dr, dc = arr.shape
        return arr.reshape(dr // det_bin, det_bin, dc // det_bin, det_bin).mean(
            axis=(1, 3)
        )

    def _export_state_for_bin(self, det_bin: int, *, scan_bin: int = 1) -> dict:
        state = self.state_dict()
        detector_scale_keys = [
            "center_row",
            "center_col",
            "bf_radius",
            "roi_center_row",
            "roi_center_col",
            "roi_radius",
            "roi_radius_inner",
            "roi_width",
            "roi_height",
        ]
        if det_bin > 1:
            for key in detector_scale_keys:
                if state.get(key) is not None:
                    state[key] = float(state[key]) / det_bin
            state["k_pixel_size"] = float(state["k_pixel_size"]) * det_bin
            state["dp_vmin"] = None
            state["dp_vmax"] = None
        scan_scale_keys = [
            "pos_row",
            "pos_col",
            "vi_roi_center_row",
            "vi_roi_center_col",
            "vi_roi_radius",
            "vi_roi_width",
            "vi_roi_height",
        ]
        if scan_bin > 1:
            for key in scan_scale_keys:
                if state.get(key) is not None:
                    state[key] = float(state[key]) / scan_bin
            state["pixel_size"] = float(state["pixel_size"]) * scan_bin
            state["vi_vmin"] = None
            state["vi_vmax"] = None
        return state

    def _write_html_report_export(
        self,
        path: str | pathlib.Path,
        *,
        dtype: str,
        det_bin: int,
        scan_bin: int,
        dataset_scope: str,
        title: str | None = None,
    ) -> pathlib.Path:
        from .export import ensure_mobile_viewport

        out = pathlib.Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(
            self._html_report_export_bytes(
                dtype=dtype,
                det_bin=det_bin,
                scan_bin=scan_bin,
                dataset_scope=dataset_scope,
                title=title,
            )
        )
        ensure_mobile_viewport(out)
        return out

    def _html_report_export_bytes(
        self,
        *,
        dtype: str,
        det_bin: int,
        scan_bin: int,
        dataset_scope: str = "unhidden",
        title: str | None = None,
    ) -> bytes:
        dtype, det_bin, scan_bin = self._normalise_html_export_options(
            dtype=dtype,
            det_bin=det_bin,
            scan_bin=scan_bin,
        )
        indices = self._report_export_indices(dataset_scope)
        if not indices:
            raise ValueError(
                "No datasets are available for the requested report export scope"
            )

        report_title = title or self.title or "Show4DSTEM Report"
        max_panels = max(1, int(self.compare_max_panels))
        pages = [
            indices[i : i + max_panels] for i in range(0, len(indices), max_panels)
        ]
        cols = (
            int(self.compare_cols)
            if int(self.compare_cols) > 0
            else int(math.ceil(math.sqrt(min(max_panels, len(indices)))))
        )
        cols = max(1, min(cols, max_panels, len(indices)))
        gap_px = max(0, int(self.compare_panel_gap_px))
        preset_masks = self._report_preset_masks()
        data = getattr(self, "_data", None)
        initial_loaded = (
            set(int(idx) for idx in data.loaded_indices())
            if type(data).__name__ == "Dataset5dstem"
            and hasattr(data, "loaded_indices")
            else set()
        )

        dp_uri = self._report_diffraction_png(det_bin=det_bin)
        page_sections: list[str] = []
        for page_idx, page_indices in enumerate(pages):
            try:
                for preset_name, preset_label, mask in preset_masks:
                    tiles = self._report_virtual_tiles(
                        page_indices,
                        mask,
                        scan_bin=scan_bin,
                    )
                    tile_html = "\n".join(
                        self._report_tile_html(
                            title=self._compare_panel_title_for_index(panel_idx),
                            uri=uri,
                            panel_idx=panel_idx,
                            page_idx=page_idx,
                            preset_label=preset_label,
                        )
                        for panel_idx, uri in zip(page_indices, tiles, strict=True)
                    )
                    page_sections.append(
                        (
                            f'<section class="report-page" data-preset="{preset_name}" '
                            f'data-page="{page_idx}"><div class="grid">{tile_html}</div></section>'
                        )
                    )
            finally:
                self._release_report_export_page_data(page_indices, initial_loaded)

        metadata = {
            "scope": self._normalise_report_dataset_scope(dataset_scope),
            "datasets": len(indices),
            "pages": len(pages),
            "page_size": max_panels,
            "scan_bin": scan_bin,
            "det_bin": det_bin,
            "dtype": dtype,
            "scan_shape": [self.shape_rows, self.shape_cols],
            "detector_shape": [self.det_rows, self.det_cols],
        }
        summary_line = (
            f"{metadata['datasets']} dataset(s) · {metadata['pages']} page(s) · "
            f"scope {metadata['scope']} · rbin {scan_bin} · kbin {det_bin}"
        )
        page_options = "\n".join(
            f'<option value="{idx}">Page {idx + 1} / {len(pages)}</option>'
            for idx in range(len(pages))
        )
        preset_options = "\n".join(
            f'<option value="{name}">{label}</option>'
            for name, label, _ in preset_masks
        )
        css = f"""
        :root {{ --cols: {cols}; --gap: {gap_px}px; color-scheme: light; }}
        body {{ margin: 0; font: 13px/1.35 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #202124; background: #fff; }}
        main {{ padding: 14px; max-width: 1600px; }}
        h1 {{ margin: 0 0 6px; color: #0759c9; font-size: 20px; }}
        .sub {{ margin: 0 0 12px; color: #5f6368; max-width: 980px; }}
        .toolbar {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin: 10px 0 12px; }}
        .toolbar label {{ font-size: 11px; color: #5f6368; }}
        select, button {{ font: inherit; height: 28px; border: 1px solid #d7dce2; border-radius: 4px; background: #f8f9fa; padding: 0 8px; }}
        .layout {{ display: grid; grid-template-columns: minmax(220px, 360px) minmax(0, 1fr); gap: 12px; align-items: start; }}
        .dp img {{ width: 100%; image-rendering: pixelated; background: #000; display: block; }}
        .meta {{ margin-top: 8px; font-size: 11px; color: #5f6368; }}
        details.meta summary {{ cursor: pointer; color: #0759c9; }}
        details.meta code {{ display: block; white-space: pre-wrap; overflow-wrap: anywhere; margin-top: 4px; }}
        .grid {{ display: grid; grid-template-columns: repeat(var(--cols), minmax(0, 1fr)); gap: var(--gap); align-items: start; }}
        .tile {{ position: relative; margin: 0; min-width: 0; background: #000; overflow: hidden; }}
        .tile img {{ width: 100%; display: block; image-rendering: pixelated; }}
        .tile figcaption {{ position: absolute; top: 3px; left: 4px; right: 4px; color: #fff; font-weight: 700; font-size: 11px; text-shadow: 0 1px 2px #000; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .report-page {{ display: none; }}
        .report-page.active {{ display: block; }}
        @media (max-width: 700px) {{
          main {{ padding: 8px; }}
          .layout {{ display: block; }}
          .dp {{ margin-bottom: 8px; }}
          .grid {{ grid-template-columns: repeat(min(var(--cols), 2), minmax(0, 1fr)); }}
        }}
        """
        js = """
        const preset = document.getElementById("preset");
        const page = document.getElementById("page");
        const prev = document.getElementById("prev");
        const next = document.getElementById("next");
        const status = document.getElementById("page-status");
        const maxPage = Number(page.dataset.max || "1");
        function sync() {
          const p = preset.value;
          const pageIdx = Number(page.value || "0");
          document.querySelectorAll(".report-page").forEach((el) => {
            el.classList.toggle("active", el.dataset.preset === p && Number(el.dataset.page || "0") === pageIdx);
          });
          status.textContent = `Page ${pageIdx + 1} / ${maxPage}`;
          prev.disabled = pageIdx <= 0;
          next.disabled = pageIdx >= maxPage - 1;
        }
        preset.addEventListener("change", sync);
        page.addEventListener("change", sync);
        prev.addEventListener("click", () => { page.value = String(Math.max(0, Number(page.value) - 1)); sync(); });
        next.addEventListener("click", () => { page.value = String(Math.min(maxPage - 1, Number(page.value) + 1)); sync(); });
        sync();
        """
        body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(report_title)}</title>
  <style>{css}</style>
</head>
<body>
  <main>
    <h1>{html.escape(report_title)}</h1>
    <p class="sub">Static report export: virtual-image PNGs only. Raw interactive 4D data is not embedded.</p>
    <div class="toolbar">
      <label for="preset">Virtual image</label>
      <select id="preset">{preset_options}</select>
      <button id="prev" type="button">Prev</button>
      <label for="page">Page</label>
      <select id="page" data-max="{len(pages)}">{page_options}</select>
      <button id="next" type="button">Next</button>
      <span id="page-status" class="meta"></span>
    </div>
    <div class="layout">
      <aside class="dp">
        <img alt="Representative diffraction pattern" src="{dp_uri}">
        <div class="meta">DP at ({int(self.pos_row)}, {int(self.pos_col)}) · detector bin {det_bin}x</div>
        <div class="meta">{html.escape(summary_line)}</div>
        <details class="meta">
          <summary>Details</summary>
          <code>{html.escape(json.dumps(metadata, separators=(",", ":")))}</code>
        </details>
      </aside>
      <div class="pages">
        {"".join(page_sections)}
      </div>
    </div>
  </main>
  <script>{js}</script>
</body>
</html>"""
        return body.encode("utf-8")

    def _normalise_report_dataset_scope(self, dataset_scope: str) -> str:
        scope = str(dataset_scope or "unhidden").strip().lower().replace("_", "-")
        aliases = {
            "visible": "unhidden",
            "current": "current-page",
            "page": "current-page",
            "current-page": "current-page",
            "star": "starred",
            "stars": "starred",
            "starred": "starred",
            "all": "all",
            "unhidden": "unhidden",
        }
        if scope not in aliases:
            raise ValueError(
                "dataset_scope must be 'unhidden', 'current_page', 'starred', or 'all'"
            )
        return aliases[scope]

    def _report_export_indices(self, dataset_scope: str) -> list[int]:
        scope = self._normalise_report_dataset_scope(dataset_scope)
        if int(self.n_frames) <= 1:
            return [0]
        ordered = self.compare_ordered_panels
        visible = self.compare_visible_panels
        if scope == "all":
            return ordered
        if scope == "starred":
            starred = {
                int(idx)
                for idx in self.compare_starred_panels
                if not isinstance(idx, bool) and 0 <= int(idx) < int(self.n_frames)
            }
            return [idx for idx in ordered if idx in starred]
        if scope == "current-page":
            max_panels = max(1, int(self.compare_max_panels))
            start = max(0, int(self.compare_page_idx)) * max_panels
            hidden = {
                int(idx)
                for idx in self.compare_hidden_panels
                if not isinstance(idx, bool) and 0 <= int(idx) < int(self.n_frames)
            }
            page = ordered[start : start + max_panels]
            return [idx for idx in page if idx not in hidden]
        return visible

    def _report_preset_masks(self) -> list[tuple[str, str, Any]]:
        cx, cy, bf = self.center_col, self.center_row, max(1.0, float(self.bf_radius))
        return [
            ("bf", "BF", self._create_circular_mask(cx, cy, bf)),
            ("abf", "ABF", self._create_annular_mask(cx, cy, bf * 0.5, bf)),
            ("adf", "ADF", self._create_annular_mask(cx, cy, bf, bf * 2.0)),
            ("haadf", "HAADF", self._create_annular_mask(cx, cy, bf * 2.0, bf * 4.0)),
        ]

    def _report_diffraction_png(self, *, det_bin: int) -> str:
        raw = self._get_frame(int(self.pos_row), int(self.pos_col)).astype(
            np.float32, copy=False
        )
        if det_bin > 1:
            raw = self._mean_detector_bin_image(raw, det_bin).astype(
                np.float32, copy=False
            )
        scaled = self._apply_scale_mode(raw, self.dp_scale_mode)
        data_min = float(scaled.min()) if scaled.size else 0.0
        data_max = float(scaled.max()) if scaled.size else 0.0
        vmin, vmax = self._slider_range(
            data_min, data_max, self.dp_vmin_pct, self.dp_vmax_pct
        )
        rgb = self._render_colormap_rgb(scaled, self.dp_colormap, vmin, vmax)
        return self._png_data_uri_from_rgb(rgb)

    def _report_virtual_tiles(
        self,
        indices: Sequence[int],
        mask,
        *,
        scan_bin: int,
    ) -> list[str]:
        images = self._compare_virtual_images_for_indices(indices, mask)
        return [self._report_virtual_png(image, scan_bin=scan_bin) for image in images]

    def _report_virtual_png(self, image: np.ndarray, *, scan_bin: int) -> str:
        raw = self._mean_scan_bin_array(
            np.asarray(image, dtype=np.float32), scan_bin
        ).astype(np.float32, copy=False)
        scaled = self._apply_scale_mode(raw, self.vi_scale_mode)
        data_min = float(scaled.min()) if scaled.size else 0.0
        data_max = float(scaled.max()) if scaled.size else 0.0
        vmin, vmax = self._slider_range(
            data_min, data_max, self.vi_vmin_pct, self.vi_vmax_pct
        )
        rgb = self._render_colormap_rgb(scaled, self.vi_colormap, vmin, vmax)
        return self._png_data_uri_from_rgb(rgb)

    @staticmethod
    def _png_data_uri_from_rgb(rgb: np.ndarray) -> str:
        from PIL import Image

        image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(
            "ascii"
        )

    def _report_tile_html(
        self,
        *,
        title: str,
        uri: str,
        panel_idx: int,
        page_idx: int,
        preset_label: str,
    ) -> str:
        label = html.escape(title)
        return (
            f'<figure class="tile" data-panel="{int(panel_idx)}" data-page="{int(page_idx)}">'
            f'<img alt="{html.escape(preset_label)} virtual image for {label}" src="{uri}">'
            f"<figcaption>{label}</figcaption></figure>"
        )

    def _release_report_export_page_data(
        self,
        page_indices: Sequence[int],
        initial_loaded: set[int],
    ) -> None:
        data = getattr(self, "_data", None)
        if type(data).__name__ != "Dataset5dstem" or not getattr(
            data, "is_lazy", False
        ):
            return
        to_release = [
            int(idx)
            for idx in page_indices
            if int(idx) not in initial_loaded and int(idx) != int(self.frame_idx)
        ]
        if not to_release:
            return
        self._compute_backend = None
        self._compute_for = None
        try:
            data.release(idx=to_release)
        finally:
            self._update_gpu_memory_status()

    def _pack_export_inline(self, *, dtype: str) -> None:
        import gzip

        arr = self._data.detach().to("cpu").numpy()
        target_shape = (
            (
                self.n_frames,
                self.shape_rows,
                self.shape_cols,
                self.det_rows,
                self.det_cols,
            )
            if self.n_frames > 1
            else (self.shape_rows, self.shape_cols, self.det_rows, self.det_cols)
        )
        arr = np.ascontiguousarray(arr.reshape(target_shape))
        if dtype == "uint8":
            packed = np.clip(arr, 0, 255).astype(np.uint8, copy=False)
        elif dtype == "uint16":
            packed = np.clip(arr, 0, 65535).astype(np.uint16, copy=False)
        else:
            raise ValueError(f"unknown export dtype {dtype!r}")
        self._offline_stack = gzip.compress(
            np.ascontiguousarray(packed).tobytes(), compresslevel=6
        )
        self._offline_gzip = True
        self._offline_url = ""
        self._offline_chunks = ""
        self._offline_bslz4 = ""
        self._offline_bad_px = ""
        self.offline = True

    def _pack_offline_bslz4_volume(
        self, data, data_url: str
    ) -> tuple[list[dict], list[int], int, int]:
        """One-call bslz4 offline pack: encode the 4D stack to native bitshuffle+LZ4
        (the Arina/HDF5 codec) and write a CHUNKED companion folder the browser
        decompresses on the GPU into a uint8 stack (~6x smaller than uint16, near-CUDA
        decode, bit-exact). Large stacks (full 512x512x192x192 = 9.6 GB uint8) split
        into scan-row chunks, each <= one 1 GB GPU buffer.

        Fast path: write the stack to a temp HDF5 with the bitshuffle-lz4 filter (C
        speed) and read the raw chunks back - this produces the exact native format
        the WGSL decoder reads, far faster than a per-block Python encode.

        ``data_url`` is a DIRECTORY; chunk_NN.bin + chunk_NN.meta + index.json land
        there, and the browser fetches them relative to the exported HTML.
        """
        import struct

        import hdf5plugin
        import h5py

        if hasattr(data, "detach"):
            data = data.detach().to("cpu").numpy()
        else:
            data = np.asarray(data)
        data = data.reshape(-1, self.det_rows, self.det_cols)
        n_frames = data.shape[0]
        det_size = self.det_rows * self.det_cols
        scan_cols = self.shape_cols
        block_elems = next((b for b in (1024, 512, 256) if det_size % b == 0), det_size)
        n_blocks = det_size // block_elems
        # Auto-detect dead/hot pixels = SATURATED only (max hits the dtype ceiling),
        # matching the HDF5 pixel_mask / CUDA apply_mask. NOT a mean-outlier test -
        # that wrongly flags the bright BF disk (real signal) as "outliers".
        flat = data.reshape(n_frames, -1)
        sat = 65535 if flat.dtype == np.uint16 else 255
        bad = np.where(flat.max(axis=0) >= sat)[0]
        bad_list = bad.astype(int).tolist()
        # Encode once via HDF5 bitshuffle-lz4 (C, fast), then read native chunks back.
        tmp_h5 = tempfile.mktemp(suffix=".h5")
        with h5py.File(tmp_h5, "w") as hf:
            # Encode UINT8 (clip 0-255): the offline display is uint8 anyway, and an
            # 8-bit-plane companion decodes ~2x faster on the GPU than uint16 (16
            # planes) at the SAME size (uint16's all-zero high byte was free). Dead
            # px are auto-filtered; real counts <=255 -> near-lossless for signal.
            hf.create_dataset(
                "d",
                data=np.clip(data, 0, 255).astype(np.uint8),
                chunks=(1, self.det_rows, self.det_cols),
                **hdf5plugin.Bitshuffle(nelems=block_elems, cname="lz4"),
            )
        out = pathlib.Path(data_url)
        out.mkdir(parents=True, exist_ok=True)
        # scan-row chunks so each decoded uint8 buffer stays <= ~0.95 GB (1 GB cap).
        rows_per = max(
            1, min(self.shape_rows, (950 * 1024 * 1024) // max(1, scan_cols * det_size))
        )
        index, total = [], 0
        try:
            hf = h5py.File(tmp_h5, "r")
            ds = hf["d"]
            cidx = 0
            for r0 in range(0, self.shape_rows, rows_per):
                r1 = min(self.shape_rows, r0 + rows_per)
                f_lo, f_hi = r0 * scan_cols, r1 * scan_cols
                raw, meta = bytearray(), []
                for gf in range(f_lo, f_hi):
                    _, chunk = ds.id.read_direct_chunk((gf, 0, 0))
                    base = len(raw)
                    pos = 12
                    for b in range(n_blocks):
                        clen = struct.unpack(">I", chunk[pos : pos + 4])[0]
                        meta += [base + pos + 4, clen]
                        pos += 4 + clen
                    raw += chunk + b"\x00" * ((-len(raw)) % 4)
                (out / f"chunk_{cidx:02d}.bin").write_bytes(bytes(raw))
                np.array(meta, dtype=np.uint32).tofile(out / f"chunk_{cidx:02d}.meta")
                index.append(
                    {
                        "bin": f"chunk_{cidx:02d}.bin",
                        "meta": f"chunk_{cidx:02d}.meta",
                        "startScan": f_lo,
                        "nScan": f_hi - f_lo,
                        "nBlocksPerFrame": n_blocks,
                        "blockElems": block_elems,
                    }
                )
                total += len(raw)
                cidx += 1
            hf.close()
        finally:
            os.unlink(tmp_h5)
        (out / "index.json").write_text(
            json.dumps({"chunks": index, "nFrames": n_frames})
        )
        return index, bad_list, total, n_blocks

    def _pack_offline_bslz4(self, data_url: str) -> None:
        index, bad, total, n_blocks = self._pack_offline_bslz4_volume(
            self._data, data_url
        )
        out = pathlib.Path(data_url)
        n_frames = self.shape_rows * self.shape_cols
        self.offline = True
        self._offline_url = ""
        self._offline_stack = b""
        self._offline_chunks = ""
        self._offline_bad_px = json.dumps(bad)
        self._offline_bslz4 = json.dumps(
            {
                "base": out.name + "/",
                "chunks": index,
                "nFrames": n_frames,
                "srcDtype": "uint8",
            }
        )
        if getattr(self, "_verbose", True) and len(bad):
            print(f"  offline auto-filter: {len(bad)} hot/dead px masked")
        if getattr(self, "_verbose", True):
            det_size = self.det_rows * self.det_cols
            ratio = (n_frames * det_size * 2) / max(1, total)
            print(
                f"  offline bslz4 (chunked): {data_url}/ {total / 1e6:.0f} MB "
                f"({ratio:.1f}x vs uint16), {n_blocks} blocks/frame, GPU-decoded to uint8"
            )

    def _pack_offline_bslz4_volumes(self, data_url: str) -> None:
        """Pack a 5D stack as lazy browser WebGPU volumes.

        The frontend already decodes ``{volumes:[...]}`` lazily through the
        Dataset/frame slider. This method makes the Python exporter expose that
        path directly instead of requiring callers to mutate ``self._data`` and
        call the single-volume private packer repeatedly.
        """
        out = pathlib.Path(data_url)
        out.mkdir(parents=True, exist_ok=True)
        volumes = []
        total = 0
        n_blocks = 0
        for idx in range(int(self.n_frames)):
            vdir = out / f"vol{idx}"
            index, bad, nbytes, blocks = self._pack_offline_bslz4_volume(
                self._data[idx], str(vdir)
            )
            # base must be relative to the HTML (the data_url's PARENT), so include the
            # data_url dir name: "widget-data/vol0/", not "vol0/". Without the parent
            # prefix the browser fetches vol0/chunk_*.bin (404) -> decode returns null
            # -> the offline compute backend bails -> presets and dataset-flip go dead.
            volumes.append(
                {"base": f"{out.name}/{vdir.name}/", "chunks": index, "badPx": bad}
            )
            total += nbytes
            n_blocks = blocks
            if getattr(self, "_verbose", True):
                print(
                    f"  offline bslz4 volume {idx}: {nbytes / 1e6:.0f} MB, "
                    f"{len(index)} chunks, {len(bad)} hot/dead px masked"
                )
        self.offline = True
        self._offline_url = ""
        self._offline_stack = b""
        self._offline_chunks = ""
        self._offline_bad_px = ""
        self._offline_bslz4 = json.dumps({"volumes": volumes, "srcDtype": "uint8"})
        if getattr(self, "_verbose", True):
            n_frames = int(self.n_frames) * self.shape_rows * self.shape_cols
            det_size = self.det_rows * self.det_cols
            ratio = (n_frames * det_size * 2) / max(1, total)
            print(
                f"  offline bslz4 volumes: {data_url}/ {total / 1e6:.0f} MB "
                f"({ratio:.1f}x vs uint16), {n_blocks} blocks/frame, "
                "GPU-decoded to uint8"
            )

    def __repr__(self) -> str:
        shape = (
            f"({self.n_frames}, {self.shape_rows}, {self.shape_cols}, {self.det_rows}, {self.det_cols})"
            if self.n_frames > 1
            else f"({self.shape_rows}, {self.shape_cols}, {self.det_rows}, {self.det_cols})"
        )
        frame_info = (
            f", {self.frame_dim_label.lower()}={self.frame_idx}"
            if self.n_frames > 1
            else ""
        )
        title_info = f", title='{self.title}'" if self.title else ""
        return (
            f"Show4DSTEM(shape={shape}, "
            f"sampling=({self.pixel_size} {self.pixel_unit}, {self.k_pixel_size} {self.k_pixel_unit}), "
            f"pos=({self.pos_row}, {self.pos_col}){frame_info}{title_info})"
        )

    # Traits that carry the bulk pixel payload. Dropped from the saved-notebook
    # snapshot when save_state is False so a plain display stays a few MB, not
    # the multi-hundred-MB packed 4D stack. frame_bytes / virtual_image_bytes
    # are single 2D slices (small) and are kept so a cold reopen still has a
    # first paint; _offline_stack / export_payload / _gif_data are the bulk ones.
    _UNSAVED_HEAVY_KEYS = ("_offline_stack", "export_payload", "_gif_data")

    def get_state(self, key=None, drop_defaults=False):
        """Trait state for comm sync and notebook embedding.

        ipywidgets calls this with ``key=None`` to snapshot the FULL state that
        gets written into the saved notebook's ``metadata.widgets``. When
        ``save_state`` is False we drop the heavy buffers from that snapshot so a
        plain Show4DSTEM does not bake the packed 4D stack into the .ipynb.
        Targeted syncs (``key`` is a name or set, used by hold_sync / send_state
        during live rendering - e.g. the deferred virtual_image_bytes / frame_bytes
        re-send on mount) are untouched, so the frontend still receives every
        buffer normally. ``save_state=True`` embeds everything so a reopened
        notebook restores the interactive offline widget without a kernel.
        """
        state = super().get_state(key=key, drop_defaults=drop_defaults)
        if key is None and not getattr(self, "_save_state", False):
            if not self._static_fallback_enabled():
                state.pop("_static_fallback_jpeg", None)
                state.pop("_static_fallback_mime", None)
            elif not self._static_fallback_jpeg:
                png = self._static_png_b64()
                if png:
                    self._store_static_fallback_preview(png)
                    state["_static_fallback_jpeg"] = self._static_fallback_jpeg
                    state["_static_fallback_mime"] = self._static_fallback_mime
            for heavy_key in self._UNSAVED_HEAVY_KEYS:
                state.pop(heavy_key, None)
        return state

    @staticmethod
    def _resize_static_panel(panel, panel_px: int):
        """Resize one static preview panel to the common live-widget panel size."""
        if panel.width == panel_px and panel.height == panel_px:
            return panel
        resampling = getattr(getattr(type(panel), "Resampling", None), "LANCZOS", None)
        if resampling is None:
            from PIL import Image

            resampling = getattr(Image, "LANCZOS", Image.BICUBIC)
        return panel.resize((panel_px, panel_px), resample=resampling)

    def _render_static_panel_image(self, panel_key: str, panel_px: int):
        """Render one saved-notebook preview panel with final-size overlays."""
        from PIL import Image

        if panel_key == "virtual":
            rgb, _ = self._render_virtual_rgb()
        elif panel_key == "diffraction":
            rgb, _ = self._render_dp_rgb()
        else:
            raise ValueError(f"Unsupported static panel {panel_key!r}")
        panel = Image.fromarray(rgb, mode="RGB")
        panel = self._resize_static_panel(panel, panel_px)
        return self._decorate_panel(
            panel, panel_key, include_overlays=True, include_scalebar=True
        )

    def _static_png_b64(self, *, max_px: int = 384, dpi: int = 160) -> str | None:
        """Base64 PNG of the current virtual-image + diffraction view.

        Lightweight saved notebooks should still communicate the 4D-STEM state:
        where the scan cursor is and what diffraction pattern/ROI produced the
        virtual image. The fallback therefore mirrors the live two-panel layout
        (virtual image + diffraction), while still omitting the heavy 4D stack
        from notebook metadata when ``save_state`` is false.
        """
        if not hasattr(self, "_data"):
            return None
        raw = self._get_virtual_image_array()
        if raw is None or raw.size == 0:
            return None
        from PIL import Image, ImageDraw, ImageFont

        panel_px = int(self.panel_width_px or max_px)
        panel_px = max(64, min(panel_px, int(max_px)))
        gap = max(2, int(panel_px * 0.015))
        title_h = 0
        title = str(self.title or "").strip()
        font = ImageFont.load_default()
        if title:
            title_h = 18

        virtual = self._render_static_panel_image("virtual", panel_px)
        diffraction = self._render_static_panel_image("diffraction", panel_px)
        width = panel_px * 2 + gap
        height = panel_px + title_h
        composite = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(composite)
        if title:
            draw.text((2, 2), title, fill=(0, 0, 0), font=font)
        y0 = title_h
        composite.paste(virtual, (0, y0))
        composite.paste(diffraction, (panel_px + gap, y0))

        buf = io.BytesIO()
        composite.save(buf, format="PNG", dpi=(dpi, dpi))
        return base64.b64encode(buf.getvalue()).decode("ascii")

    # _repr_mimebundle_ / _ipython_display_ / static-fallback sibling plumbing
    # comes from StaticFallbackMixin (utils/static_fallback.py); this class only
    # supplies _static_png_b64 above.

    def _store_static_fallback_preview(self, png_b64: str) -> None:
        """Store a compact saved-notebook preview inside lightweight state."""
        if getattr(self, "_save_state", False):
            return
        encoded = self._encode_static_fallback_b64(png_b64)
        if encoded is None:
            self._static_fallback_jpeg = ""
            self._static_fallback_mime = ""
            return
        mime, image_b64 = encoded
        self._static_fallback_jpeg = image_b64
        self._static_fallback_mime = mime

    def state_dict(self):
        return {
            "title": self.title,
            "show_title": self.show_title,
            "pos_row": self.pos_row,
            "pos_col": self.pos_col,
            "pixel_size": self.pixel_size,
            "pixel_unit": self.pixel_unit,
            "k_pixel_size": self.k_pixel_size,
            "k_pixel_unit": self.k_pixel_unit,
            "k_calibrated": self.k_calibrated,
            "center_row": self.center_row,
            "center_col": self.center_col,
            "bf_radius": self.bf_radius,
            "roi_active": self.roi_active,
            "roi_mode": self.roi_mode,
            "roi_center_row": self.roi_center_row,
            "roi_center_col": self.roi_center_col,
            "roi_radius": self.roi_radius,
            "roi_radius_inner": self.roi_radius_inner,
            "roi_width": self.roi_width,
            "roi_height": self.roi_height,
            "vi_roi_mode": self.vi_roi_mode,
            "vi_roi_center_row": self.vi_roi_center_row,
            "vi_roi_center_col": self.vi_roi_center_col,
            "vi_roi_radius": self.vi_roi_radius,
            "vi_roi_width": self.vi_roi_width,
            "vi_roi_height": self.vi_roi_height,
            "vi_roi_reduce": self.vi_roi_reduce,
            "dp_colormap": self.dp_colormap,
            "vi_colormap": self.vi_colormap,
            "fft_colormap": self.fft_colormap,
            "dp_scale_mode": self.dp_scale_mode,
            "vi_scale_mode": self.vi_scale_mode,
            "fft_scale_mode": self.fft_scale_mode,
            "dp_vmin_pct": self.dp_vmin_pct,
            "dp_vmax_pct": self.dp_vmax_pct,
            "vi_vmin_pct": self.vi_vmin_pct,
            "vi_vmax_pct": self.vi_vmax_pct,
            "fft_vmin_pct": self.fft_vmin_pct,
            "fft_vmax_pct": self.fft_vmax_pct,
            "dp_vmin": self.dp_vmin,
            "dp_vmax": self.dp_vmax,
            "vi_vmin": self.vi_vmin,
            "vi_vmax": self.vi_vmax,
            "fft_auto": self.fft_auto,
            "show_fft": self.show_fft,
            "fft_window": self.fft_window,
            "show_controls": self.show_controls,
            "controls_collapsed": self.controls_collapsed,
            "show_stats": self.show_stats,
            "show_scale_bar": self.show_scale_bar,
            "debug": self.debug,
            "panel_width_px": self.panel_width_px,
            "dp_show_colorbar": self.dp_show_colorbar,
            "vi_auto_contrast": self.vi_auto_contrast,
            "vi_smooth": self.vi_smooth,
            "view_mode": self.view_mode,
            "compare_layout": self.compare_layout,
            "compare_cols": self.compare_cols,
            "compare_grid_width_px": self.compare_grid_width_px,
            "compare_panel_gap_px": self.compare_panel_gap_px,
            "compare_max_panels": self.compare_max_panels,
            "compare_group_mode": self.compare_group_mode,
            "compare_page_idx": self.compare_page_idx,
            "compare_dp_mode": self.compare_dp_mode,
            "compare_panel_order": list(self.compare_panel_order),
            "compare_hidden_panels": list(self.compare_hidden_panels),
            "compare_starred_panels": list(self.compare_starred_panels),
            "path_interval_ms": self.path_interval_ms,
            "path_loop": self.path_loop,
            "profile_line": self.profile_line,
            "profile_width": self.profile_width,
            "frame_idx": self.frame_idx,
            "frame_dim_label": self.frame_dim_label,
            "frame_labels": list(self.frame_labels),
            "frame_loop": self.frame_loop,
            "frame_fps": self.frame_fps,
            "frame_reverse": self.frame_reverse,
            "frame_boomerang": self.frame_boomerang,
        }

    def save(self, path: str):
        save_state_file(path, "Show4DSTEM", self.state_dict())

    def load_state_dict(self, state):
        allowed_keys = set(self.state_dict().keys())
        pending_pos_row = state.get("pos_row", None)
        pending_pos_col = state.get("pos_col", None)
        pending_frame_idx = state.get("frame_idx", None)
        for key, val in state.items():
            if key in {"pos_row", "pos_col", "frame_idx"}:
                continue
            if key in allowed_keys:
                if key == "view_mode":
                    val = self._normalise_view_mode(val)
                elif key == "compare_layout":
                    val = self._normalise_compare_layout(val)
                elif key == "compare_dp_mode":
                    val = self._normalise_compare_dp_mode(val)
                elif key == "compare_group_mode":
                    val = self._normalise_compare_group_mode(val)
                setattr(self, key, val)
        if pending_frame_idx is not None:
            self.frame_idx = int(max(0, min(int(pending_frame_idx), self.n_frames - 1)))
        if pending_pos_row is not None or pending_pos_col is not None:
            row = int(self.pos_row if pending_pos_row is None else pending_pos_row)
            col = int(self.pos_col if pending_pos_col is None else pending_pos_col)
            self.pos_row = int(max(0, min(row, self.shape_rows - 1)))
            self.pos_col = int(max(0, min(col, self.shape_cols - 1)))
        self._refresh_compare_virtual_images()
        self._update_frame()

    def collapse_controls(self) -> Self:
        """Collapse the live control UI programmatically."""
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

    def free(self):
        """Free GPU memory held by this widget.

        Drops EVERY reference to the data tensor and flushes the allocator pools,
        so the stack actually leaves VRAM (no kernel restart needed). The data is
        held in FOUR places, not one: ``self._data`` plus the compute backend's
        cached ``_t`` / ``_4d`` / ``_flat`` views (``self._compute_backend``) plus
        the per-op ``self._compute_for`` cache - missing any one keeps the whole
        stack pinned. And the storage is cupy-owned (``io.load`` decompresses with
        cupy; the widget wraps it via ``from_dlpack``), so when the torch refs die
        the memory returns to the CUPY pool, which ``torch.empty_cache`` cannot
        release - the cupy pool is freed too. Call before loading a new dataset.

        Examples
        --------
        >>> w.free()          # release the full stack from VRAM
        >>> del result        # free the source array
        """
        self.stop_folder_watch()
        self.stop_dataset_preload(wait=True)
        self.stop_compare_cache_warm(wait=True)

        import gc

        data = self._data
        cuda_indices: set[int] = set()
        needs_mps_clear = False

        def record_device(value) -> None:
            nonlocal needs_mps_clear
            if value is None:
                return
            dev = value.device if isinstance(value, torch.Tensor) else value
            device = torch.device(dev)
            if device.type == "cuda":
                cuda_indices.add(0 if device.index is None else int(device.index))
            elif device.type == "mps":
                needs_mps_clear = True

        record_device(getattr(self, "_device", None))
        if type(data).__name__ == "Dataset5dstem" and hasattr(data, "devices"):
            for dev in data.devices:
                record_device(dev)
        elif isinstance(data, torch.Tensor):
            record_device(data)
        record_device(getattr(self, "_compute_for", None))
        nbytes = (
            data.resident_nbytes
            if data is not None and hasattr(data, "resident_nbytes")
            else data.nbytes
            if data is not None and hasattr(data, "nbytes")
            else 0
        )
        # Every holder of the data storage (verified via a gc storage scan):
        # the widget's own ref, the compute backend's view cache, and the per-op cache.
        self._compute_backend = None
        self._compute_for = None
        if type(data).__name__ == "Dataset5dstem" and hasattr(data, "free"):
            data.free()
        self._data = None
        self._clear_compare_virtual_images()
        self.compare_status = (
            "Show4DSTEM data was freed. Re-run the load/display cell to restore "
            "the multiple grid."
        )
        gc.collect()
        if needs_mps_clear:
            try:
                torch.mps.empty_cache()
            except AttributeError:
                pass
        for idx in sorted(cuda_indices):
            with torch.cuda.device(idx):
                torch.cuda.empty_cache()
        # Storage is cupy-owned via dlpack; the freed memory sits in the cupy pool
        # until its blocks are returned to the driver.
        try:
            import cupy as cp

            for idx in sorted(cuda_indices):
                with cp.cuda.Device(idx):
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
        except ImportError:
            pass
        if nbytes > 0:
            devices = [f"cuda:{idx}" for idx in sorted(cuda_indices)]
            if needs_mps_clear:
                devices.append("mps")
            label = ", ".join(devices) if devices else "host"
            print(f"freed {_format_memory(nbytes)} ({label})")

    def summary(self):
        name = self.title if self.title else "Show4DSTEM"
        lines = [name, "═" * 32]
        if self.n_frames > 1:
            parts = [
                f"{self.n_frames} ({self.frame_dim_label}), current: {self.frame_idx}"
            ]
            parts.append(f"{self.frame_fps} fps")
            if self.frame_loop:
                parts.append("loop")
            if self.frame_reverse:
                parts.append("reverse")
            if self.frame_boomerang:
                parts.append("bounce")
            lines.append(f"Frames:   {' | '.join(parts)}")
            if self._frame_labels:
                if len(self._frame_labels) <= 4:
                    lines.append(f"Labels:   {self._frame_labels}")
                else:
                    lines.append(
                        f"Labels:   {self._frame_labels[:3]} ... ({len(self._frame_labels)} total)"
                    )
        lines.append(
            f"Scan:     {self.shape_rows}×{self.shape_cols} ({self.pixel_size:.2f} {self.pixel_unit}/px)"
        )
        lines.append(
            f"Detector: {self.det_rows}×{self.det_cols} ({self.k_pixel_size:.4f} {self.k_pixel_unit}/px)"
        )
        lines.append(f"Position: ({self.pos_row}, {self.pos_col})")
        lines.append(
            f"Center:   ({self.center_row:.1f}, {self.center_col:.1f})  BF r={self.bf_radius:.1f} px"
        )
        if self.roi_active:
            lines.append(
                f"ROI:      {self.roi_mode} at ({self.roi_center_row:.1f}, {self.roi_center_col:.1f}) r={self.roi_radius:.1f}"
            )
        if self.vi_roi_mode != "off":
            lines.append(
                f"VI ROI:   {self.vi_roi_mode} at ({self.vi_roi_center_row:.1f}, {self.vi_roi_center_col:.1f}) r={self.vi_roi_radius:.1f}"
            )
        dp_contrast = f"{self.dp_vmin_pct:.1f}-{self.dp_vmax_pct:.1f}%"
        if self.dp_vmin is not None and self.dp_vmax is not None:
            dp_contrast += f", dp_vmin={self.dp_vmin:.4g}, dp_vmax={self.dp_vmax:.4g}"
        lines.append(
            f"DP view:  {self.dp_colormap}, {self.dp_scale_mode}, {dp_contrast}"
        )
        vi_contrast = f"{self.vi_vmin_pct:.1f}-{self.vi_vmax_pct:.1f}%"
        if self.vi_vmin is not None and self.vi_vmax is not None:
            vi_contrast += f", vi_vmin={self.vi_vmin:.4g}, vi_vmax={self.vi_vmax:.4g}"
        lines.append(
            f"VI view:  {self.vi_colormap}, {self.vi_scale_mode}, {vi_contrast}"
        )
        if self.show_fft:
            fft_parts = [
                f"{self.fft_colormap}, {self.fft_scale_mode}, {self.fft_vmin_pct:.1f}-{self.fft_vmax_pct:.1f}%, auto={self.fft_auto}"
            ]
            if not self.fft_window:
                fft_parts.append("no window")
            lines.append(f"FFT view: {', '.join(fft_parts)}")
        if self.profile_line and len(self.profile_line) == 2:
            p0, p1 = self.profile_line[0], self.profile_line[1]
            lines.append(
                f"Profile:  ({p0['row']:.0f}, {p0['col']:.0f}) -> ({p1['row']:.0f}, {p1['col']:.0f}) width={self.profile_width}"
            )
        print("\n".join(lines))

    # =========================================================================
    # Convenience Properties
    # =========================================================================

    @property
    def position(self) -> tuple[int, int]:
        """Current scan position as (row, col) tuple."""
        return (self.pos_row, self.pos_col)

    @position.setter
    def position(self, value: tuple[int, int]) -> None:
        """Set scan position from (row, col) tuple."""
        self.pos_row, self.pos_col = value

    @property
    def scan_shape(self) -> tuple[int, int]:
        """Scan dimensions as (rows, cols) tuple."""
        return (self.shape_rows, self.shape_cols)

    @property
    def detector_shape(self) -> tuple[int, int]:
        """Detector dimensions as (rows, cols) tuple."""
        return (self.det_rows, self.det_cols)

    @property
    def _frame_data(self) -> torch.Tensor:
        """Per-frame data (4D or 3D flattened), accounting for 5D time/tilt series."""
        if type(self._data).__name__ == "Dataset5dstem":
            # .frame() is paging-aware: when page_budget is set it brings this
            # dataset into VRAM and evicts the least-recently-used one; when
            # paging is off it is identical to self._data[frame_idx].
            return self._data.frame(self.frame_idx)
        if self.n_frames > 1:
            return self._data[self.frame_idx]
        return self._data

    @property
    def _compute(self):
        """UI-agnostic Python compute backend for the CURRENT frame's data,
        rebuilt when the frame changes. Two families:

        * ``TorchBackend`` — torch tensor on CUDA / MPS / CPU (universal default).
        * ``MetalRawBackend`` — chunk-backed Metal frames (a 19 GB no-bin
          stack where torch.MPS overflows).

        ``backend='web'`` does NOT install a third Python backend; it sets
        ``offline=True`` so the kernel ships a uint8-packed stack to the browser
        and the JS ``Show4DSTEMCompute`` does all reductions in WebGPU. The
        Python ``_compute`` stays a Torch/MetalRaw backend for any kernel-side
        fallbacks (e.g. ``_pack_offline`` initial compute, snapshot PNG).

        Construction is cheap (views, no copy) so the backend rebuilds when
        ``_frame_data`` changes (5D time-series scrub).
        """
        fd = self._frame_data
        if getattr(self, "_compute_for", None) is not fd:
            from quantem.widget.kernels.compute.backends import compute_backend

            self._compute_backend = compute_backend(fd)
            self._compute_for = fd
        return self._compute_backend

    # =========================================================================
    # Line Profile
    # =========================================================================

    def set_profile(self, start: tuple, end: tuple) -> Self:
        row0, col0 = start
        row1, col1 = end
        self.profile_line = [
            {"row": float(row0), "col": float(col0)},
            {"row": float(row1), "col": float(col1)},
        ]
        return self

    def clear_profile(self) -> Self:
        self.profile_line = []
        return self

    @property
    def profile(self) -> list[tuple[float, float]]:
        if len(self.profile_line) == 2:
            p0, p1 = self.profile_line[0], self.profile_line[1]
            return [(p0["row"], p0["col"]), (p1["row"], p1["col"])]
        return []

    @property
    def profile_values(self):
        if len(self.profile_line) != 2:
            return None
        p0, p1 = self.profile_line[0], self.profile_line[1]
        frame = self._get_frame(self.pos_row, self.pos_col)
        return self._sample_line(frame, p0["row"], p0["col"], p1["row"], p1["col"])

    @property
    def profile_distance(self) -> float:
        if len(self.profile_line) != 2:
            return 0.0
        p0, p1 = self.profile_line[0], self.profile_line[1]
        dist_px = np.sqrt((p1["row"] - p0["row"]) ** 2 + (p1["col"] - p0["col"]) ** 2)
        if self.k_calibrated:
            return float(dist_px * self.k_pixel_size)
        return float(dist_px)

    def _sample_line(self, frame, row0, col0, row1, col1):
        h, w = frame.shape[:2]
        dc = col1 - col0
        dr = row1 - row0
        length = np.sqrt(dc * dc + dr * dr)
        n = max(2, int(np.ceil(length)))
        t = np.linspace(0.0, 1.0, n)
        c = col0 + t * dc
        r = row0 + t * dr
        ci = np.floor(c).astype(np.intp)
        ri = np.floor(r).astype(np.intp)
        cf = c - ci
        rf = r - ri
        c0 = np.clip(ci, 0, w - 1)
        c1 = np.clip(ci + 1, 0, w - 1)
        r0 = np.clip(ri, 0, h - 1)
        r1 = np.clip(ri + 1, 0, h - 1)
        return (
            frame[r0, c0] * (1 - cf) * (1 - rf)
            + frame[r0, c1] * cf * (1 - rf)
            + frame[r1, c0] * (1 - cf) * rf
            + frame[r1, c1] * cf * rf
        ).astype(np.float32)

    # =========================================================================
    # Path Animation Methods
    # =========================================================================

    def set_path(
        self,
        points: list[tuple[int, int]],
        interval_ms: int = 100,
        loop: bool = True,
        autoplay: bool = True,
    ) -> Self:
        """
        Set a custom path of scan positions to animate through.

        Parameters
        ----------
        points : list[tuple[int, int]]
            List of (row, col) scan positions to visit.
        interval_ms : int, default 100
            Time between frames in milliseconds.
        loop : bool, default True
            Whether to loop when reaching end.
        autoplay : bool, default True
            Start playing immediately.

        Returns
        -------
        Show4DSTEM
            Self for method chaining.

        Examples
        --------
        >>> widget.set_path([(0, 0), (10, 10), (20, 20), (30, 30)])
        >>> widget.set_path([(i, i) for i in range(48)], interval_ms=50)
        """
        self._path_points = list(points)
        self.path_length = len(self._path_points)
        self.path_index = 0
        self.path_interval_ms = interval_ms
        self.path_loop = loop
        if autoplay and self.path_length > 0:
            self.path_playing = True
        return self

    def play(self) -> Self:
        """Start playing the path animation."""
        if self.path_length > 0:
            self.path_playing = True
        return self

    def pause(self) -> Self:
        """Pause the path animation."""
        self.path_playing = False
        return self

    def stop(self) -> Self:
        """Stop and reset path animation to beginning."""
        self.path_playing = False
        self.path_index = 0
        return self

    def goto(self, index: int) -> Self:
        """Jump to a specific index in the path."""
        if 0 <= index < self.path_length:
            self.path_index = index
        return self

    def _on_path_index_change(self, change):
        """Called when path_index changes (from frontend timer)."""
        idx = change["new"]
        if 0 <= idx < len(self._path_points):
            row, col = self._path_points[idx]
            # Clamp to valid range
            self.pos_row = max(0, min(self.shape_rows - 1, row))
            self.pos_col = max(0, min(self.shape_cols - 1, col))

    def _on_preset_request(self, change):
        """JS preset shortcut → atomic apply_preset (no per-trait race)."""
        name = (change.get("new") or "").strip().lower()
        if name in ("bf", "abf", "adf", "haadf"):
            self.apply_preset(name)
            self._preset_request = ""  # consume trigger

    def _on_compare_config_change(self, change=None) -> None:
        """Refresh compare-grid payload after relevant config/readiness changes."""
        if getattr(self, "_suppress_folder_append_refresh", False):
            return
        if change and change.get("name") == "view_mode":
            self.view_mode = self._normalise_view_mode(change.get("new", "single"))
        if change and change.get("name") == "compare_group_mode":
            self.compare_group_mode = self._normalise_compare_group_mode(
                change.get("new", "paged")
            )
        if change and change.get("name") == "compare_hidden_panels":
            self._handle_compare_hidden_change(
                change.get("old", []), change.get("new", [])
            )
        self._refresh_compare_virtual_images()
        if self.view_mode == "multiple":
            self._ensure_current_compare_frame_visible()
            self._update_frame()
        else:
            self._compute_virtual_image_from_roi()
            self._update_frame()

    def _on_compare_dp_mode_change(self, change=None) -> None:
        """Normalize and apply compare DP source mode changes."""
        if change:
            self.compare_dp_mode = self._normalise_compare_dp_mode(
                change.get("new", "average")
            )
        if self.view_mode == "multiple":
            self._update_frame()

    def _on_frame_idx_change(self, change=None):
        """Called when frame_idx changes (5D time/tilt series).

        Recomputes virtual image and diffraction pattern for the new frame.
        Invalidates precomputed caches since they are per-frame.
        """
        if self.n_frames <= 1:
            return
        # Invalidate precomputed caches (they were for a different frame)
        self._cached_bf_virtual = None
        self._cached_abf_virtual = None
        self._cached_adf_virtual = None
        self._cached_haadf_virtual = None
        if self.view_mode == "multiple":
            self._sync_compare_page_to_frame_idx()
        # Recompute virtual image only when it is visible. In multiple mode the
        # visible virtual-image surface is the compare grid; switching back to
        # single recomputes the per-frame virtual image in _on_compare_config_change.
        if self.view_mode != "multiple":
            self._compute_virtual_image_from_roi()
        self._update_frame()
        # Recompute reduced DP if VI ROI is active
        if self.vi_roi_mode != "off":
            self._compute_vi_roi_dp()
        self._update_gpu_memory_status()

    # =========================================================================
    # Path Animation Patterns
    # =========================================================================

    def raster(
        self,
        step: int = 1,
        bidirectional: bool = False,
        interval_ms: int = 100,
        loop: bool = True,
    ) -> Self:
        """
        Play a raster scan path (row by row, left to right).

        This mimics real STEM scanning: left→right, step down, left→right, etc.

        Parameters
        ----------
        step : int, default 1
            Step size between positions.
        bidirectional : bool, default False
            If True, use snake/boustrophedon pattern (alternating direction).
            If False (default), always scan left→right like real STEM.
        interval_ms : int, default 100
            Time between frames in milliseconds.
        loop : bool, default True
            Whether to loop when reaching the end.

        Returns
        -------
        Show4DSTEM
            Self for method chaining.
        """
        points = []
        for r in range(0, self.shape_rows, step):
            cols = list(range(0, self.shape_cols, step))
            if bidirectional and (r // step % 2 == 1):
                cols = cols[::-1]  # Alternate direction for snake pattern
            for c in cols:
                points.append((r, c))
        return self.set_path(points=points, interval_ms=interval_ms, loop=loop)

    # =========================================================================
    # ROI Mode Methods
    # =========================================================================

    def roi_circle(self, radius: float | None = None) -> Self:
        """
        Switch to circle ROI mode for virtual imaging.

        In circle mode, the virtual image integrates over a circular region
        centered at the current ROI position (like a virtual bright field detector).

        Parameters
        ----------
        radius : float, optional
            Radius of the circle in pixels. If not provided, uses current value
            or defaults to half the BF radius.

        Returns
        -------
        Show4DSTEM
            Self for method chaining.

        Examples
        --------
        >>> widget.roi_circle(20)  # 20px radius circle
        >>> widget.roi_circle()    # Use default radius
        """
        self.roi_mode = "circle"
        if radius is not None:
            self.roi_radius = float(radius)
        return self

    def roi_point(self) -> Self:
        """
        Switch to point ROI mode (single-pixel indexing).

        In point mode, the virtual image shows intensity at the exact ROI position.
        This is the default mode.

        Returns
        -------
        Show4DSTEM
            Self for method chaining.
        """
        self.roi_mode = "point"
        return self

    def roi_square(self, half_size: float | None = None) -> Self:
        """
        Switch to square ROI mode for virtual imaging.

        In square mode, the virtual image integrates over a square region
        centered at the current ROI position.

        Parameters
        ----------
        half_size : float, optional
            Half-size of the square in pixels (distance from center to edge).
            A half_size of 15 creates a 30x30 pixel square.
            If not provided, uses current roi_radius value.

        Returns
        -------
        Show4DSTEM
            Self for method chaining.

        Examples
        --------
        >>> widget.roi_square(15)  # 30x30 pixel square (half_size=15)
        >>> widget.roi_square()    # Use default size
        """
        self.roi_mode = "square"
        if half_size is not None:
            self.roi_radius = float(half_size)
        return self

    def roi_annular(
        self, inner_radius: float | None = None, outer_radius: float | None = None
    ) -> Self:
        """
        Set ROI mode to annular (donut-shaped) for ADF/HAADF imaging.

        Parameters
        ----------
        inner_radius : float, optional
            Inner radius in pixels. If not provided, uses current roi_radius_inner.
        outer_radius : float, optional
            Outer radius in pixels. If not provided, uses current roi_radius.

        Returns
        -------
        Show4DSTEM
            Self for method chaining.

        Examples
        --------
        >>> widget.roi_annular(20, 50)  # ADF: inner=20px, outer=50px
        >>> widget.roi_annular(30, 80)  # HAADF: larger angles
        """
        self.roi_mode = "annular"
        if inner_radius is not None:
            self.roi_radius_inner = float(inner_radius)
        if outer_radius is not None:
            self.roi_radius = float(outer_radius)
        return self

    def roi_rect(self, width: float | None = None, height: float | None = None) -> Self:
        """
        Set ROI mode to rectangular.

        Parameters
        ----------
        width : float, optional
            Width in pixels. If not provided, uses current roi_width.
        height : float, optional
            Height in pixels. If not provided, uses current roi_height.

        Returns
        -------
        Show4DSTEM
            Self for method chaining.

        Examples
        --------
        >>> widget.roi_rect(30, 20)  # 30px wide, 20px tall
        >>> widget.roi_rect(40, 40)  # 40x40 rectangle
        """
        self.roi_mode = "rect"
        if width is not None:
            self.roi_width = float(width)
        if height is not None:
            self.roi_height = float(height)
        return self

    def auto_detect_center(self, update_roi: bool = True) -> Self:
        """
        Automatically detect BF disk center and radius using centroid.

        This method analyzes the summed diffraction pattern to find the
        bright field disk center and estimate its radius. The detected
        values are applied to the widget's calibration (center_row, center_col,
        bf_radius).

        Parameters
        ----------
        update_roi : bool, default True
            If True, also update ROI center and recompute cached virtual images.
            Set to False during __init__ when ROI is not yet initialized.

        Returns
        -------
        Show4DSTEM
            Self for method chaining.

        Examples
        --------
        >>> widget = Show4DSTEM(data)
        >>> widget.auto_detect_center()  # Auto-detect and apply
        """
        # Sum diffraction patterns over scan positions to find BF disk centroid.
        # Chunked torch INTEGER path: works identically on CUDA / MPS / CPU.
        # int64 accumulator, no float32 cast of the stack — keeps the data in its
        # native dtype (a float32 cast doubles memory and is lossy above 2^24),
        # is bit-exact, and avoids the MPS "tensor dims larger than INT_MAX"
        # error a single full-stack reduce hits once positions*det > 2^31 (a bin2
        # 512x512x96x96 stack = 2.42e9 elements). The chunk cap keeps each op's
        # element count well under 2^31; the (det, det) accumulator is tiny.
        # Mean DP over all scan positions via the compute backend (TorchCompute
        # int64-accumulates in chunks; MetalCompute uses the raw detector_sum
        # kernel). Centroid + radius are scale-invariant, so mean vs sum is the
        # same center/radius. The (det, det) result is tiny - torch for the centroid.
        mean_dp = torch.as_tensor(self._compute.mean_dp(), device=self._device).float()
        threshold = mean_dp.mean() + mean_dp.std()
        mask = mean_dp > threshold

        total = mask.sum()
        if total == 0:
            return self

        cx = float((self._det_col_coords * mask).sum() / total)
        cy = float((self._det_row_coords * mask).sum() / total)
        radius = float(torch.sqrt(total / torch.pi))

        # Apply detected values
        self.center_col = cx
        self.center_row = cy
        self.bf_radius = radius

        if update_roi:
            # Also update ROI to center
            self.roi_center_col = cx
            self.roi_center_row = cy
            # Recompute cached virtual images with new calibration
            self._precompute_common_virtual_images()

        return self

    def _get_frame(self, row: int, col: int) -> np.ndarray:
        """Get single diffraction frame at position (row, col) as numpy array.

        Via the compute backend (torch index on tensor data, Metal buffer read on
        chunk-backed frames) so the cursor DP works on every backend."""
        if self._data is None:
            return np.zeros((self.det_rows, self.det_cols), dtype=np.float32)
        return np.asarray(self._compute.frame(row * self.shape_cols + col))

    def _apply_scale_mode(self, data: np.ndarray, mode: str) -> np.ndarray:
        arr = np.asarray(data, dtype=np.float32)
        if mode == "log":
            return np.log1p(np.maximum(arr, 0.0)).astype(np.float32)
        return arr.astype(np.float32)

    def _slider_range(
        self,
        data_min: float,
        data_max: float,
        vmin_pct: float,
        vmax_pct: float,
    ) -> tuple[float, float]:
        v0 = float(max(0.0, min(100.0, vmin_pct)))
        v1 = float(max(0.0, min(100.0, vmax_pct)))
        if v1 < v0:
            v0, v1 = v1, v0
        rng = float(data_max - data_min)
        return (
            float(data_min + (v0 / 100.0) * rng),
            float(data_min + (v1 / 100.0) * rng),
        )

    def _render_colormap_rgb(
        self,
        data: np.ndarray,
        cmap_name: str,
        vmin: float,
        vmax: float,
    ) -> np.ndarray:
        from matplotlib import colormaps

        arr = np.asarray(data, dtype=np.float32)
        if vmax <= vmin:
            normalized = np.zeros_like(arr, dtype=np.float32)
        else:
            normalized = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
        rgba = colormaps.get_cmap(cmap_name)(normalized)
        return (rgba[..., :3] * 255).astype(np.uint8)

    def _get_virtual_image_array(self) -> np.ndarray:
        if not self.virtual_image_bytes:
            return np.zeros((self.shape_rows, self.shape_cols), dtype=np.float32)
        arr = np.frombuffer(self.virtual_image_bytes, dtype=np.float32)
        expected = self.shape_rows * self.shape_cols
        if arr.size != expected:
            return np.zeros((self.shape_rows, self.shape_cols), dtype=np.float32)
        return arr.reshape(self.shape_rows, self.shape_cols).copy()

    def _get_vi_roi_dp_array(self) -> np.ndarray | None:
        if self.vi_roi_mode == "off":
            return None
        self._compute_vi_roi_dp()
        if not self.vi_roi_dp_bytes:
            return None
        arr = np.frombuffer(self.vi_roi_dp_bytes, dtype=np.float32)
        expected = self.det_rows * self.det_cols
        if arr.size != expected:
            return None
        return arr.reshape(self.det_rows, self.det_cols).copy()

    def _fft_enhanced_range(self, mag: np.ndarray) -> tuple[float, float]:
        arr = np.asarray(mag, dtype=np.float32).copy()
        if arr.size == 0:
            return 0.0, 0.0
        center_row = arr.shape[0] // 2
        center_col = arr.shape[1] // 2
        neighbors = []
        if center_col - 1 >= 0:
            neighbors.append(arr[center_row, center_col - 1])
        if center_col + 1 < arr.shape[1]:
            neighbors.append(arr[center_row, center_col + 1])
        if center_row - 1 >= 0:
            neighbors.append(arr[center_row - 1, center_col])
        if center_row + 1 < arr.shape[0]:
            neighbors.append(arr[center_row + 1, center_col])
        if neighbors:
            arr[center_row, center_col] = float(np.mean(neighbors))
        dmin = float(arr.min())
        dmax = float(arr.max())
        if dmax <= dmin:
            return dmin, dmax
        pmax = float(np.percentile(arr, 99.9))
        if pmax <= dmin:
            pmax = dmax
        return dmin, pmax

    def _render_dp_rgb(self) -> tuple[np.ndarray, dict]:
        vi_roi_arr = self._get_vi_roi_dp_array()
        if vi_roi_arr is not None:
            raw = vi_roi_arr
            source = "vi_roi_dp"
        else:
            raw = self._get_frame(self.pos_row, self.pos_col).astype(np.float32)
            source = "single_frame"

        scale_mode = self.dp_scale_mode
        scaled = self._apply_scale_mode(raw, scale_mode)
        data_min = float(scaled.min()) if scaled.size else 0.0
        data_max = float(scaled.max()) if scaled.size else 0.0
        if self.dp_vmin is not None and self.dp_vmax is not None:
            vmin = float(
                self._apply_scale_mode(
                    np.array([max(self.dp_vmin, 0)], dtype=np.float32), scale_mode
                )[0]
            )
            vmax = float(
                self._apply_scale_mode(
                    np.array([max(self.dp_vmax, 0)], dtype=np.float32), scale_mode
                )[0]
            )
        else:
            vmin, vmax = self._slider_range(
                data_min, data_max, self.dp_vmin_pct, self.dp_vmax_pct
            )
        rgb = self._render_colormap_rgb(scaled, self.dp_colormap, vmin, vmax)
        metadata = {
            "source": source,
            "colormap": self.dp_colormap,
            "scale_mode": scale_mode,
            "vmin_pct": float(self.dp_vmin_pct),
            "vmax_pct": float(self.dp_vmax_pct),
            "vmin": float(vmin),
            "vmax": float(vmax),
        }
        return rgb, metadata

    def _render_virtual_rgb(self) -> tuple[np.ndarray, dict]:
        raw = self._get_virtual_image_array()
        scaled = self._apply_scale_mode(raw, self.vi_scale_mode)
        data_min = float(scaled.min()) if scaled.size else 0.0
        data_max = float(scaled.max()) if scaled.size else 0.0
        if self.vi_vmin is not None and self.vi_vmax is not None:
            vmin = float(
                self._apply_scale_mode(
                    np.array([max(self.vi_vmin, 0)], dtype=np.float32),
                    self.vi_scale_mode,
                )[0]
            )
            vmax = float(
                self._apply_scale_mode(
                    np.array([max(self.vi_vmax, 0)], dtype=np.float32),
                    self.vi_scale_mode,
                )[0]
            )
        else:
            vmin, vmax = self._slider_range(
                data_min, data_max, self.vi_vmin_pct, self.vi_vmax_pct
            )
        rgb = self._render_colormap_rgb(scaled, self.vi_colormap, vmin, vmax)
        metadata = {
            "colormap": self.vi_colormap,
            "scale_mode": self.vi_scale_mode,
            "vmin_pct": float(self.vi_vmin_pct),
            "vmax_pct": float(self.vi_vmax_pct),
            "vmin": float(vmin),
            "vmax": float(vmax),
        }
        return rgb, metadata

    def _render_fft_rgb(self) -> tuple[np.ndarray, dict]:
        virtual_raw = self._get_virtual_image_array()
        fft = np.fft.fftshift(np.fft.fft2(virtual_raw))
        mag = np.abs(fft).astype(np.float32)
        scaled = self._apply_scale_mode(mag, self.fft_scale_mode)
        if self.fft_auto:
            display_min, display_max = self._fft_enhanced_range(scaled)
        else:
            display_min = float(scaled.min()) if scaled.size else 0.0
            display_max = float(scaled.max()) if scaled.size else 0.0
        vmin, vmax = self._slider_range(
            display_min, display_max, self.fft_vmin_pct, self.fft_vmax_pct
        )
        rgb = self._render_colormap_rgb(scaled, self.fft_colormap, vmin, vmax)
        metadata = {
            "colormap": self.fft_colormap,
            "scale_mode": self.fft_scale_mode,
            "auto": bool(self.fft_auto),
            "vmin_pct": float(self.fft_vmin_pct),
            "vmax_pct": float(self.fft_vmax_pct),
            "vmin": float(vmin),
            "vmax": float(vmax),
        }
        return rgb, metadata

    _EXPORT_VIEWS = ("diffraction", "virtual", "fft", "all")
    _EXPORT_FORMATS = ("png", "pdf")

    def _validate_export_view(self, view: str | None) -> str:
        view_key = (view or "all").strip().lower()
        if view_key not in self._EXPORT_VIEWS:
            raise ValueError(
                f"Unsupported view '{view}'. Supported: {', '.join(self._EXPORT_VIEWS)}"
            )
        return view_key

    def _validate_frame_idx(self, frame_idx: int | None) -> int:
        if frame_idx is None:
            return int(self.frame_idx)
        idx = int(frame_idx)
        if idx < 0 or idx >= self.n_frames:
            raise ValueError(
                f"frame_idx={idx} is out of range [0, {self.n_frames - 1}]"
            )
        return idx

    def _validate_position(self, position: tuple[int, int] | None) -> tuple[int, int]:
        if position is None:
            return int(self.pos_row), int(self.pos_col)
        if len(position) != 2:
            raise ValueError(
                "position must be a (row, col) tuple with exactly two values"
            )
        row = int(position[0])
        col = int(position[1])
        if row < 0 or row >= self.shape_rows or col < 0 or col >= self.shape_cols:
            raise ValueError(
                f"position=({row}, {col}) is out of range for "
                f"scan_shape=({self.shape_rows}, {self.shape_cols})"
            )
        return row, col

    def _resolve_export_format(self, path: pathlib.Path, fmt: str | None) -> str:
        resolved = (fmt or path.suffix.lstrip(".") or "png").strip().lower()
        if resolved not in self._EXPORT_FORMATS:
            raise ValueError(
                f"Unsupported format '{resolved}'. Supported: {', '.join(self._EXPORT_FORMATS)}"
            )
        return resolved

    @staticmethod
    def _round_to_nice_value(value: float) -> float:
        if value <= 0:
            return 1.0
        magnitude = 10 ** math.floor(math.log10(value))
        normalized = value / magnitude
        if normalized < 1.5:
            return float(magnitude)
        if normalized < 3.5:
            return float(2 * magnitude)
        if normalized < 7.5:
            return float(5 * magnitude)
        return float(10 * magnitude)

    def _format_scale_label(self, value: float, unit: str) -> str:
        nice = self._round_to_nice_value(value)
        if unit == "Å":
            if nice >= 10:
                return f"{int(round(nice / 10))} nm"
            if nice >= 1:
                return f"{int(round(nice))} Å"
            return f"{nice:.2f} Å"
        if unit == "mrad":
            if nice >= 1000:
                return f"{int(round(nice / 1000))} rad"
            if nice >= 1:
                return f"{int(round(nice))} mrad"
            return f"{nice:.2f} mrad"
        if nice >= 1:
            return f"{int(round(nice))} px"
        return f"{nice:.1f} px"

    @staticmethod
    def _draw_crosshair(
        draw, x: float, y: float, size: float, color, width: int
    ) -> None:
        draw.line([(x - size, y), (x + size, y)], fill=color, width=width)
        draw.line([(x, y - size), (x, y + size)], fill=color, width=width)

    def _draw_scalebar_overlay(self, image, pixel_size: float, unit: str) -> None:
        from PIL import ImageDraw, ImageFont

        if pixel_size <= 0:
            return

        draw = ImageDraw.Draw(image, mode="RGBA")
        font = ImageFont.load_default()
        width, height = image.size
        margin = max(8, int(min(width, height) * 0.04))
        thickness = max(2, int(height * 0.01))
        target_bar_px = max(36, int(width * 0.15))
        target_physical = float(target_bar_px) * float(pixel_size)
        nice_physical = self._round_to_nice_value(target_physical)
        bar_px = max(12, int(round(nice_physical / float(pixel_size))))
        bar_px = min(bar_px, max(12, int(width * 0.8)))

        x1 = width - margin
        x0 = x1 - bar_px
        y1 = height - margin
        y0 = y1 - thickness

        draw.rectangle([(x0 + 1, y0 + 1), (x1 + 1, y1 + 1)], fill=(0, 0, 0, 180))
        draw.rectangle([(x0, y0), (x1, y1)], fill=(255, 255, 255, 255))

        label = self._format_scale_label(nice_physical, unit)
        label_bbox = draw.textbbox((0, 0), label, font=font)
        label_w = label_bbox[2] - label_bbox[0]
        label_h = label_bbox[3] - label_bbox[1]
        tx = x0 + (bar_px - label_w) / 2
        ty = y0 - label_h - 4
        draw.text((tx + 1, ty + 1), label, fill=(0, 0, 0, 220), font=font)
        draw.text((tx, ty), label, fill=(255, 255, 255, 255), font=font)

        zoom_label = "1.0x"
        zoom_bbox = draw.textbbox((0, 0), zoom_label, font=font)
        zoom_h = zoom_bbox[3] - zoom_bbox[1]
        zx = margin
        zy = height - margin - zoom_h
        draw.text((zx + 1, zy + 1), zoom_label, fill=(0, 0, 0, 220), font=font)
        draw.text((zx, zy), zoom_label, fill=(255, 255, 255, 255), font=font)

    def _draw_dp_overlays(self, image) -> None:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(image, mode="RGBA")
        width, height = image.size
        scale_x = float(width) / float(max(1, self.det_cols))
        scale_y = float(height) / float(max(1, self.det_rows))
        cx = float(self.roi_center_col) * scale_x
        cy = float(self.roi_center_row) * scale_y

        if self.roi_active and self.roi_mode != "point":
            stroke = (0, 220, 0, 240)
            fill = (0, 220, 0, 45)
            if self.roi_mode == "circle":
                rx = float(self.roi_radius) * scale_x
                ry = float(self.roi_radius) * scale_y
                draw.ellipse(
                    [(cx - rx, cy - ry), (cx + rx, cy + ry)],
                    outline=stroke,
                    fill=fill,
                    width=2,
                )
            elif self.roi_mode == "square":
                rx = float(self.roi_radius) * scale_x
                ry = float(self.roi_radius) * scale_y
                draw.rectangle(
                    [(cx - rx, cy - ry), (cx + rx, cy + ry)],
                    outline=stroke,
                    fill=fill,
                    width=2,
                )
            elif self.roi_mode == "rect":
                rx = (float(self.roi_width) / 2.0) * scale_x
                ry = (float(self.roi_height) / 2.0) * scale_y
                draw.rectangle(
                    [(cx - rx, cy - ry), (cx + rx, cy + ry)],
                    outline=stroke,
                    fill=fill,
                    width=2,
                )
            elif self.roi_mode == "annular":
                outer_rx = float(self.roi_radius) * scale_x
                outer_ry = float(self.roi_radius) * scale_y
                inner_rx = float(self.roi_radius_inner) * scale_x
                inner_ry = float(self.roi_radius_inner) * scale_y
                draw.ellipse(
                    [(cx - outer_rx, cy - outer_ry), (cx + outer_rx, cy + outer_ry)],
                    outline=stroke,
                    fill=fill,
                    width=2,
                )
                draw.ellipse(
                    [(cx - inner_rx, cy - inner_ry), (cx + inner_rx, cy + inner_ry)],
                    outline=stroke,
                    fill=(0, 0, 0, 0),
                    width=2,
                )

        marker_color = (0, 220, 0, 255) if self.roi_active else (255, 100, 100, 255)
        self._draw_crosshair(
            draw,
            cx,
            cy,
            size=max(6, int(min(width, height) * 0.03)),
            color=marker_color,
            width=2,
        )

        if len(self.profile_line) == 2:
            p0, p1 = self.profile_line[0], self.profile_line[1]
            x0 = float(p0["col"]) * scale_x
            y0 = float(p0["row"]) * scale_y
            x1 = float(p1["col"]) * scale_x
            y1 = float(p1["row"]) * scale_y
            draw.line(
                [(x0, y0), (x1, y1)],
                fill=(0, 200, 255, 240),
                width=max(1, int(self.profile_width)),
            )
            r = 3
            draw.ellipse([(x0 - r, y0 - r), (x0 + r, y0 + r)], fill=(0, 200, 255, 255))
            draw.ellipse([(x1 - r, y1 - r), (x1 + r, y1 + r)], fill=(0, 200, 255, 255))

    def _draw_vi_overlays(self, image) -> None:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(image, mode="RGBA")
        width, height = image.size
        scale_x = float(width) / float(max(1, self.shape_cols))
        scale_y = float(height) / float(max(1, self.shape_rows))

        px = float(self.pos_col) * scale_x
        py = float(self.pos_row) * scale_y
        self._draw_crosshair(
            draw,
            px,
            py,
            size=max(6, int(min(width, height) * 0.03)),
            color=(255, 100, 100, 240),
            width=2,
        )

        if self.vi_roi_mode == "off":
            return

        cx = float(self.vi_roi_center_col) * scale_x
        cy = float(self.vi_roi_center_row) * scale_y
        stroke = (180, 80, 255, 240)
        fill = (180, 80, 255, 45)
        if self.vi_roi_mode == "circle":
            rx = float(self.vi_roi_radius) * scale_x
            ry = float(self.vi_roi_radius) * scale_y
            draw.ellipse(
                [(cx - rx, cy - ry), (cx + rx, cy + ry)],
                outline=stroke,
                fill=fill,
                width=2,
            )
        elif self.vi_roi_mode == "square":
            rx = float(self.vi_roi_radius) * scale_x
            ry = float(self.vi_roi_radius) * scale_y
            draw.rectangle(
                [(cx - rx, cy - ry), (cx + rx, cy + ry)],
                outline=stroke,
                fill=fill,
                width=2,
            )
        elif self.vi_roi_mode == "rect":
            rx = (float(self.vi_roi_width) / 2.0) * scale_x
            ry = (float(self.vi_roi_height) / 2.0) * scale_y
            draw.rectangle(
                [(cx - rx, cy - ry), (cx + rx, cy + ry)],
                outline=stroke,
                fill=fill,
                width=2,
            )

        self._draw_crosshair(
            draw,
            cx,
            cy,
            size=max(6, int(min(width, height) * 0.03)),
            color=(180, 80, 255, 240),
            width=2,
        )

    def _decorate_panel(
        self,
        image,
        panel_key: str,
        include_overlays: bool,
        include_scalebar: bool,
    ):
        out = image.copy()
        if include_overlays:
            if panel_key == "diffraction":
                self._draw_dp_overlays(out)
            elif panel_key == "virtual":
                self._draw_vi_overlays(out)
        if include_scalebar:
            if panel_key == "diffraction":
                unit = "mrad" if self.k_calibrated else "px"
                self._draw_scalebar_overlay(out, float(self.k_pixel_size), unit)
            elif panel_key == "virtual":
                self._draw_scalebar_overlay(out, float(self.pixel_size), "Å")
        return out

    def _render_panel_image(
        self,
        panel_key: str,
        include_overlays: bool,
        include_scalebar: bool,
    ) -> tuple[Any, dict[str, Any]]:
        from PIL import Image

        if panel_key == "diffraction":
            rgb, render_meta = self._render_dp_rgb()
        elif panel_key == "virtual":
            rgb, render_meta = self._render_virtual_rgb()
        elif panel_key == "fft":
            rgb, render_meta = self._render_fft_rgb()
        else:
            raise ValueError(
                f"Unsupported panel {panel_key!r}. Valid options: 'diffraction', 'virtual', 'fft', 'all'."
            )

        panel = Image.fromarray(rgb, mode="RGB")
        panel = self._decorate_panel(
            panel, panel_key, include_overlays, include_scalebar
        )
        return panel, render_meta

    def _compose_horizontal(self, panels: list[Any]):
        from PIL import Image

        height = max(panel.height for panel in panels)
        width = sum(panel.width for panel in panels)
        composite = Image.new("RGB", (width, height), color=(0, 0, 0))
        x0 = 0
        for panel in panels:
            composite.paste(panel, (x0, 0))
            x0 += panel.width
        return composite

    def _calibration_metadata(self) -> dict[str, Any]:
        return {
            "pixel_size_angstrom": float(self.pixel_size),
            "pixel_size_unit": "Å/px",
            "k_pixel_size": float(self.k_pixel_size),
            "k_pixel_size_unit": "mrad/px" if self.k_calibrated else "px/px",
            "k_calibrated": bool(self.k_calibrated),
            "center_row": float(self.center_row),
            "center_col": float(self.center_col),
            "bf_radius": float(self.bf_radius),
        }

    def _roi_metadata(self) -> dict[str, Any]:
        return {
            "active": bool(self.roi_active),
            "mode": self.roi_mode,
            "center_row": float(self.roi_center_row),
            "center_col": float(self.roi_center_col),
            "radius": float(self.roi_radius),
            "radius_inner": float(self.roi_radius_inner),
            "width": float(self.roi_width),
            "height": float(self.roi_height),
        }

    def _vi_roi_metadata(self) -> dict[str, Any]:
        return {
            "mode": self.vi_roi_mode,
            "center_row": float(self.vi_roi_center_row),
            "center_col": float(self.vi_roi_center_col),
            "radius": float(self.vi_roi_radius),
            "width": float(self.vi_roi_width),
            "height": float(self.vi_roi_height),
        }

    def _build_image_export_metadata(
        self,
        export_path: pathlib.Path,
        view_key: str,
        fmt: str,
        render_meta: dict[str, Any],
        include_overlays: bool,
        include_scalebar: bool,
        export_kind: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            **build_json_header("Show4DSTEM"),
            "view": view_key,
            "format": fmt,
            "export_kind": export_kind,
            "path": str(export_path),
            "position": {"row": int(self.pos_row), "col": int(self.pos_col)},
            "frame_idx": int(self.frame_idx),
            "n_frames": int(self.n_frames),
            "scan_shape": {"rows": int(self.shape_rows), "cols": int(self.shape_cols)},
            "detector_shape": {"rows": int(self.det_rows), "cols": int(self.det_cols)},
            "roi": self._roi_metadata(),
            "vi_roi": self._vi_roi_metadata(),
            "calibration": self._calibration_metadata(),
            "display": render_meta,
            "include_overlays": bool(include_overlays),
            "include_scalebar": bool(include_scalebar),
        }
        if extra:
            metadata.update(extra)
        return metadata

    def save_image(
        self,
        path: str | pathlib.Path,
        view: str | None = None,
        position: tuple[int, int] | None = None,
        frame_idx: int | None = None,
        format: str | None = None,
        include_metadata: bool = True,
        metadata_path: str | pathlib.Path | None = None,
        include_overlays: bool | None = None,
        include_scalebar: bool | None = None,
        restore_state: bool = True,
        dpi: int | None = None,
    ) -> pathlib.Path:
        """
        Save the current visualization as PNG or PDF.

        Parameters
        ----------
        path : str or pathlib.Path
            Output image path.
        view : str, optional
            One of: "diffraction", "virtual", "fft", "all".
        position : tuple[int, int], optional
            Temporary scan position override as (row, col) for this export.
        frame_idx : int, optional
            Temporary frame index override for 5D data.
        format : str, optional
            "png" or "pdf". If omitted, inferred from file extension.
        include_metadata : bool, default True
            If True, writes JSON metadata next to the image.
        metadata_path : str or pathlib.Path, optional
            Override metadata JSON path.
        include_overlays : bool, default True
            Draw ROI/profile/crosshair overlays on exported panels.
        include_scalebar : bool, default True
            Draw panel scale bars on exported panels.
        restore_state : bool, default True
            If True, temporary position/frame overrides are reverted after export.
        dpi : int, optional
            Export DPI metadata.

        Returns
        -------
        pathlib.Path
            The written image path.
        """
        from PIL import Image

        export_path = pathlib.Path(path)
        view_key = self._validate_export_view(view)
        fmt = self._resolve_export_format(export_path, format)
        dpi_value = 300 if dpi is None else int(dpi)
        overlays_enabled = True if include_overlays is None else bool(include_overlays)
        scalebar_enabled = True if include_scalebar is None else bool(include_scalebar)
        if dpi_value <= 0:
            raise ValueError(f"dpi must be > 0, got {dpi_value}")

        export_path.parent.mkdir(parents=True, exist_ok=True)

        prev_row, prev_col = self.pos_row, self.pos_col
        prev_frame = self.frame_idx
        meta_path: pathlib.Path | None = None

        try:
            if frame_idx is not None:
                self.frame_idx = self._validate_frame_idx(frame_idx)
            if position is not None:
                row, col = self._validate_position(position)
                self.pos_row = row
                self.pos_col = col

            if view_key == "diffraction":
                image, dp_meta = self._render_panel_image(
                    "diffraction", overlays_enabled, scalebar_enabled
                )
                render_meta = {"diffraction": dp_meta}
            elif view_key == "virtual":
                image, vi_meta = self._render_panel_image(
                    "virtual", overlays_enabled, scalebar_enabled
                )
                render_meta = {"virtual": vi_meta}
            elif view_key == "fft":
                image, fft_meta = self._render_panel_image(
                    "fft", overlays_enabled, scalebar_enabled
                )
                render_meta = {"fft": fft_meta}
            else:
                panel_images = []
                render_meta = {}
                dp_img, dp_meta = self._render_panel_image(
                    "diffraction", overlays_enabled, scalebar_enabled
                )
                vi_img, vi_meta = self._render_panel_image(
                    "virtual", overlays_enabled, scalebar_enabled
                )
                panel_images.extend([dp_img, vi_img])
                render_meta = {"diffraction": dp_meta, "virtual": vi_meta}
                if self.show_fft:
                    fft_img, fft_meta = self._render_panel_image(
                        "fft", overlays_enabled, scalebar_enabled
                    )
                    panel_images.append(fft_img)
                    render_meta["fft"] = fft_meta
                image = self._compose_horizontal(panel_images)

            if fmt == "pdf":
                Image.init()
                image = image.convert("RGB")
                image.save(export_path, format="PDF", resolution=dpi_value)
            else:
                image.save(export_path, format="PNG", dpi=(dpi_value, dpi_value))

            if include_metadata:
                meta_path = (
                    pathlib.Path(metadata_path)
                    if metadata_path is not None
                    else export_path.with_suffix(".json")
                )
                metadata = self._build_image_export_metadata(
                    export_path=export_path,
                    view_key=view_key,
                    fmt=fmt,
                    render_meta=render_meta,
                    include_overlays=overlays_enabled,
                    include_scalebar=scalebar_enabled,
                    export_kind="single_view_image",
                    extra={"dpi": int(dpi_value)},
                )
                meta_path.write_text(json.dumps(metadata, indent=2))
        finally:
            if restore_state:
                self.frame_idx = prev_frame
                self.pos_row = prev_row
                self.pos_col = prev_col

        return export_path

    def apply_preset(self, name: str) -> Self:
        preset_name = str(name).strip().lower()
        # Batch all trait writes atomically. Without this, each individual
        # trait change fires _on_roi_change, and intermediate states (e.g. mode
        # just switched to "annular" but radius_inner still stale from the
        # previous preset) compute a wrong mask -> black VI flashes before the
        # final correct frame. hold_trait_notifications defers observers until
        # all 5 traits have committed.
        bf = self.bf_radius
        center_row = float(self.center_row)
        center_col = float(self.center_col)
        self._suppress_roi_recompute = True
        try:
            if preset_name == "bf":
                with self.hold_trait_notifications():
                    self.roi_active = True
                    self.roi_mode = "circle"
                    self.roi_center_row = center_row
                    self.roi_center_col = center_col
                    self.roi_radius = float(max(1.0, bf))
            elif preset_name == "abf":
                with self.hold_trait_notifications():
                    self.roi_active = True
                    self.roi_mode = "annular"
                    self.roi_center_row = center_row
                    self.roi_center_col = center_col
                    self.roi_radius_inner = float(max(0.5, bf * 0.5))
                    self.roi_radius = float(max(1.0, bf))
            elif preset_name == "adf":
                with self.hold_trait_notifications():
                    self.roi_active = True
                    self.roi_mode = "annular"
                    self.roi_center_row = center_row
                    self.roi_center_col = center_col
                    self.roi_radius_inner = float(max(1.0, bf))
                    self.roi_radius = float(max(bf + 1.0, bf * 2.0))
            elif preset_name == "haadf":
                with self.hold_trait_notifications():
                    self.roi_active = True
                    self.roi_mode = "annular"
                    self.roi_center_row = center_row
                    self.roi_center_col = center_col
                    self.roi_radius_inner = float(max(1.0, bf * 2.0))
                    self.roi_radius = float(max(bf * 2.0 + 1.0, bf * 4.0))
            else:
                raise ValueError(
                    f"Unknown preset {name!r}. Choices: 'bf', 'abf', 'adf', 'haadf'."
                )
        finally:
            self._suppress_roi_recompute = False
        # Single recompute with final, consistent state.
        if self.view_mode != "multiple":
            self._compute_virtual_image_from_roi()
        self._refresh_compare_virtual_images()
        return self

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        mode = self.dp_scale_mode
        scaled = self._apply_scale_mode(frame, mode)
        if self.dp_vmin is not None and self.dp_vmax is not None:
            fmin = float(
                self._apply_scale_mode(
                    np.array([max(self.dp_vmin, 0)], dtype=np.float32), mode
                )[0]
            )
            fmax = float(
                self._apply_scale_mode(
                    np.array([max(self.dp_vmax, 0)], dtype=np.float32), mode
                )[0]
            )
        else:
            fmin = float(scaled.min())
            fmax = float(scaled.max())
            fmin, fmax = self._slider_range(
                fmin, fmax, self.dp_vmin_pct, self.dp_vmax_pct
            )
        if fmax > fmin:
            return np.clip((scaled - fmin) / (fmax - fmin) * 255, 0, 255).astype(
                np.uint8
            )
        return np.zeros(frame.shape, dtype=np.uint8)

    def _on_gif_export(self, change=None):
        if not self._gif_export_requested:
            return
        self._gif_export_requested = False
        self._generate_gif()

    def _generate_gif(self):
        import io

        from matplotlib import colormaps
        from PIL import Image

        if not self._path_points:
            with self.hold_sync():
                self._gif_data = b""
                self._gif_metadata_json = ""
            return

        cmap_fn = colormaps.get_cmap(self.dp_colormap)
        duration_ms = max(10, self.path_interval_ms)

        pil_frames = []
        for row, col in self._path_points:
            row = max(0, min(self.shape_rows - 1, row))
            col = max(0, min(self.shape_cols - 1, col))
            frame = self._get_frame(row, col).astype(np.float32)
            normalized = self._normalize_frame(frame)
            rgba = cmap_fn(normalized / 255.0)
            rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
            pil_frames.append(Image.fromarray(rgb))

        if not pil_frames:
            return

        buf = io.BytesIO()
        pil_frames[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
        )
        metadata = {
            **build_json_header("Show4DSTEM"),
            "view": "diffraction",
            "format": "gif",
            "export_kind": "path_animation",
            "n_frames": int(len(pil_frames)),
            "duration_ms": int(duration_ms),
            "path_loop": bool(self.path_loop),
            "path_points": [
                {"row": int(row), "col": int(col)} for row, col in self._path_points
            ],
            "frame_idx": int(self.frame_idx),
            "n_frames_total": int(self.n_frames),
            "scan_shape": {"rows": int(self.shape_rows), "cols": int(self.shape_cols)},
            "detector_shape": {"rows": int(self.det_rows), "cols": int(self.det_cols)},
            "calibration": self._calibration_metadata(),
            "display": {
                "diffraction": {
                    "colormap": self.dp_colormap,
                    "scale_mode": self.dp_scale_mode,
                    "vmin_pct": float(self.dp_vmin_pct),
                    "vmax_pct": float(self.dp_vmax_pct),
                }
            },
        }
        with self.hold_sync():
            self._gif_metadata_json = json.dumps(metadata, indent=2)
            self._gif_data = buf.getvalue()

    def _update_frame(self, change=None):
        """Send raw float32 frame to frontend (JS handles scale/colormap)."""
        if self._data is None:
            return
        if (
            self.view_mode == "multiple"
            and self.n_frames > 1
            and self._normalise_compare_dp_mode(self.compare_dp_mode) == "average"
        ):
            frame = self._average_compare_diffraction_frame()
        else:
            frame = self._diffraction_frame_for_index(self.frame_idx)

        # Cast small frame to float32 for stats and JS transfer. Bulk data
        # stays in native dtype; only this single 192×192 (~144 KB) frame
        # gets promoted.
        if not isinstance(frame, torch.Tensor):
            frame = torch.as_tensor(np.asarray(frame))
        if frame.dtype != torch.float32:
            frame = frame.float()
        # Stats compute moved to JS (frontend has frame_bytes; computeStats() in
        # js/stats.ts does mean/min/max/std on the Float32Array directly,
        # avoiding 4 sync trait round-trips per scan-position click).
        self.frame_bytes = frame.detach().cpu().numpy().tobytes()

    def _diffraction_frame_for_index(self, frame_idx: int):
        """Return one diffraction pattern at the current scan position."""
        data_source = getattr(self, "_data", None)
        datasets = getattr(data_source, "datasets", None)
        if datasets is not None:
            idx = int(max(0, min(int(frame_idx), len(datasets) - 1)))
            dataset = datasets[idx]
            if dataset is None:
                ready = [item for item in datasets if item is not None]
                if not ready:
                    raise ValueError("no ready compare datasets")
                dataset = ready[0]
            flat_idx = int(self.pos_row) * int(self.shape_cols) + int(self.pos_col)
            return dataset.frame(flat_idx)

        if type(data_source).__name__ == "Dataset5dstem":
            data = data_source.frame(int(frame_idx))
        elif self.n_frames > 1:
            data = data_source[int(frame_idx)]
        else:
            data = data_source
        if data.ndim == 3:
            idx = self.pos_row * self.shape_cols + self.pos_col
            return data[idx]
        return data[self.pos_row, self.pos_col]

    def _compare_diffraction_cache_key(
        self, frame_idx: int
    ) -> tuple[int, int, int, int, int]:
        return (
            int(frame_idx),
            int(self.pos_row),
            int(self.pos_col),
            int(self.det_rows),
            int(self.det_cols),
        )

    def _store_compare_diffraction_cache(self, frame_idx: int, frame) -> None:
        """Cache one small DP for average compare mode without retaining raw 4D data."""
        if self._compare_cache_pages <= 0:
            return
        key = self._compare_diffraction_cache_key(frame_idx)
        try:
            if isinstance(frame, torch.Tensor):
                if frame.ndim == 4:
                    dp = frame[int(self.pos_row), int(self.pos_col)]
                elif frame.ndim == 3:
                    flat_idx = int(self.pos_row) * int(self.shape_cols) + int(
                        self.pos_col
                    )
                    dp = frame[flat_idx]
                else:
                    dp = frame
                if dp.dtype != torch.float32:
                    dp = dp.float()
                arr = np.ascontiguousarray(dp.detach().cpu().numpy(), dtype=np.float32)
            else:
                arr0 = np.asarray(frame)
                if arr0.ndim == 4:
                    arr0 = arr0[int(self.pos_row), int(self.pos_col)]
                elif arr0.ndim == 3:
                    flat_idx = int(self.pos_row) * int(self.shape_cols) + int(
                        self.pos_col
                    )
                    arr0 = arr0[flat_idx]
                arr = np.ascontiguousarray(arr0, dtype=np.float32)
        except Exception:
            return
        if arr.shape != (int(self.det_rows), int(self.det_cols)):
            return
        existing = self._compare_diffraction_cache.pop(key, None)
        if existing is not None:
            self._compare_diffraction_cache_bytes -= existing[1]
        payload = arr.tobytes()
        self._compare_diffraction_cache[key] = (payload, len(payload))
        self._compare_diffraction_cache_bytes += len(payload)
        self._trim_compare_diffraction_cache()

    def _get_cached_compare_diffraction_frame(
        self, frame_idx: int
    ) -> np.ndarray | None:
        key = self._compare_diffraction_cache_key(frame_idx)
        cached = self._compare_diffraction_cache.get(key)
        if cached is None:
            return None
        self._compare_diffraction_cache.move_to_end(key)
        payload, _ = cached
        return np.frombuffer(payload, dtype=np.float32).reshape(
            self.det_rows, self.det_cols
        )

    def _trim_compare_diffraction_cache(self) -> None:
        max_entries = max(0, int(self._compare_cache_pages)) * max(
            1, int(self.compare_max_panels)
        )
        while (
            self._compare_diffraction_cache
            and len(self._compare_diffraction_cache) > max_entries
        ):
            _, old = self._compare_diffraction_cache.popitem(last=False)
            self._compare_diffraction_cache_bytes -= old[1]

    def _reclaim_compare_allocation_failure(self) -> None:
        """Release unreferenced GPU pools after a failed compare DP load."""
        data = getattr(self, "_data", None)
        reclaim = getattr(data, "_reclaim", None)
        known_devices = getattr(data, "_known_reclaim_devices", None)
        if callable(reclaim) and callable(known_devices):
            try:
                reclaim(known_devices())
            except Exception:
                pass
        gc.collect()

    def _diffraction_frame_as_numpy(self, frame) -> np.ndarray:
        """Return one DP as contiguous float32 CPU data."""
        if isinstance(frame, torch.Tensor):
            if frame.ndim == 4:
                frame = frame[int(self.pos_row), int(self.pos_col)]
            elif frame.ndim == 3:
                flat_idx = int(self.pos_row) * int(self.shape_cols) + int(self.pos_col)
                frame = frame[flat_idx]
            if frame.dtype != torch.float32:
                frame = frame.float()
            arr = frame.detach().cpu().numpy()
        else:
            arr = np.asarray(frame)
            if arr.ndim == 4:
                arr = arr[int(self.pos_row), int(self.pos_col)]
            elif arr.ndim == 3:
                flat_idx = int(self.pos_row) * int(self.shape_cols) + int(self.pos_col)
                arr = arr[flat_idx]
        return np.ascontiguousarray(arr, dtype=np.float32)

    def _average_compare_diffraction_frame(self):
        """Average ready visible compare-panel diffraction patterns."""
        indices = self._compare_current_page_indices()
        acc_np: np.ndarray | None = None
        count = 0
        for idx in indices:
            frame = self._get_cached_compare_diffraction_frame(idx)
            if frame is None:
                try:
                    frame = self._diffraction_frame_for_index(idx)
                except BaseException as exc:
                    if not _is_recoverable_allocation_error(exc):
                        raise
                    self._reclaim_compare_allocation_failure()
                    self._set_gpu_memory_warning(
                        action="average diffraction patterns for the visible panels",
                        requested=len(indices),
                        shown=count,
                    )
                    break
                self._store_compare_diffraction_cache(idx, frame)
            frame_np = self._diffraction_frame_as_numpy(frame)
            if frame_np.shape != (int(self.det_rows), int(self.det_cols)):
                continue
            acc_np = frame_np.copy() if acc_np is None else acc_np + frame_np
            count += 1
        if acc_np is None:
            try:
                frame = self._diffraction_frame_for_index(self.frame_idx)
                self._store_compare_diffraction_cache(int(self.frame_idx), frame)
                return self._diffraction_frame_as_numpy(frame)
            except BaseException as exc:
                if not _is_recoverable_allocation_error(exc):
                    raise
                self._reclaim_compare_allocation_failure()
                self._set_gpu_memory_warning(
                    action="load a diffraction pattern for the selected panel",
                    requested=len(indices) if indices else 1,
                    shown=0,
                )
                return np.zeros(
                    (int(self.det_rows), int(self.det_cols)), dtype=np.float32
                )
        if count > 1:
            acc_np /= float(count)
        return acc_np

    def _on_roi_change(self, change=None):
        """Recompute virtual image when individual ROI params change.

        High-frequency drag updates use the compound roi_center trait instead.
        """
        if not self.roi_active:
            return
        if getattr(self, "_suppress_roi_recompute", False):
            return
        if self.view_mode != "multiple":
            self._compute_virtual_image_from_roi()
        self._refresh_compare_virtual_images()

    def _on_roi_center_change(self, change=None):
        """Handle batched roi_center updates from JS (single observer for row+col).

        This is the fast path for drag operations. JS sends [row, col] as a single
        compound trait, so only one observer fires per mouse move.
        """
        if not self.roi_active:
            return
        if getattr(self, "_suppress_roi_recompute", False):
            return
        if change and "new" in change:
            row, col = change["new"]
            # Sync to individual traits (without triggering _on_roi_change observers)
            self.unobserve(
                self._on_roi_change, names=["roi_center_col", "roi_center_row"]
            )
            self.roi_center_row = row
            self.roi_center_col = col
            self.observe(
                self._on_roi_change, names=["roi_center_col", "roi_center_row"]
            )
        if self.view_mode != "multiple":
            self._compute_virtual_image_from_roi()
        self._refresh_compare_virtual_images()

    def _on_vi_roi_center_change(self, change=None):
        """Apply compound (row, col) update atomically (avoids split-trait race)."""
        if change and "new" in change:
            row, col = change["new"]
            self.unobserve(
                self._on_vi_roi_change, names=["vi_roi_center_row", "vi_roi_center_col"]
            )
            self.vi_roi_center_row = float(row)
            self.vi_roi_center_col = float(col)
            self.observe(
                self._on_vi_roi_change, names=["vi_roi_center_row", "vi_roi_center_col"]
            )
        if self.vi_roi_mode == "off":
            self.vi_roi_dp_bytes = b""
            return
        self._compute_vi_roi_dp()

    def _on_vi_roi_change(self, change=None):
        """Recompute reduced DP when VI ROI or reduction changes."""
        if self.vi_roi_mode == "off":
            self.vi_roi_dp_bytes = b""
            return
        self._compute_vi_roi_dp()

    def _compute_vi_roi_dp(self):
        """Reduce diffraction patterns over scan positions inside VI ROI.

        Reduction selected by `vi_roi_reduce`:
        - "mean": average DP (size-invariant, default for region-of-interest analysis)
        - "sum": total counts (scales with ROI area; use for quantitative integration)
        - "max": brightest pixel per detector position across the region
        """
        if self._data is None:
            return
        if self.vi_roi_mode == "circle":
            mask = (self._scan_row_coords - self.vi_roi_center_row) ** 2 + (
                self._scan_col_coords - self.vi_roi_center_col
            ) ** 2 <= self.vi_roi_radius**2
        elif self.vi_roi_mode == "square":
            half_size = self.vi_roi_radius
            mask = (
                torch.abs(self._scan_row_coords - self.vi_roi_center_row) <= half_size
            ) & (torch.abs(self._scan_col_coords - self.vi_roi_center_col) <= half_size)
        elif self.vi_roi_mode == "rect":
            half_w = self.vi_roi_width / 2
            half_h = self.vi_roi_height / 2
            mask = (
                torch.abs(self._scan_row_coords - self.vi_roi_center_row) <= half_h
            ) & (torch.abs(self._scan_col_coords - self.vi_roi_center_col) <= half_w)
        else:
            return

        n_positions = int(mask.sum())
        if n_positions == 0:
            self.vi_roi_dp_bytes = b""
            return
        # Flat scan indices inside the ROI, reduced (mean/sum/max) by the compute
        # backend (torch gather on tensor data, Metal mean_frames on chunk-backed
        # frames). One path for every backend; gives consistent results across
        # CUDA / MPS instead of the old torch-vs-subclass index-math divergence.
        indices = (
            torch.nonzero(mask.reshape(-1), as_tuple=False).flatten().cpu().numpy()
        )
        dp = self._compute.reduce_frames(indices, self.vi_roi_reduce)
        self.vi_roi_dp_bytes = np.ascontiguousarray(dp, dtype=np.float32).tobytes()

    def _create_circular_mask(self, cx: float, cy: float, radius: float):
        """Create circular mask (boolean tensor on device)."""
        mask = (self._det_col_coords - cx) ** 2 + (
            self._det_row_coords - cy
        ) ** 2 <= radius**2
        return mask

    def _create_square_mask(self, cx: float, cy: float, half_size: float):
        """Create square mask (boolean tensor on device)."""
        mask = (torch.abs(self._det_col_coords - cx) <= half_size) & (
            torch.abs(self._det_row_coords - cy) <= half_size
        )
        return mask

    def _create_annular_mask(self, cx: float, cy: float, inner: float, outer: float):
        """Create annular (donut) mask (boolean tensor on device)."""
        dist_sq = (self._det_col_coords - cx) ** 2 + (self._det_row_coords - cy) ** 2
        mask = (dist_sq >= inner**2) & (dist_sq <= outer**2)
        return mask

    def _create_rect_mask(
        self, cx: float, cy: float, half_width: float, half_height: float
    ):
        """Create rectangular mask (boolean tensor on device)."""
        mask = (torch.abs(self._det_col_coords - cx) <= half_width) & (
            torch.abs(self._det_row_coords - cy) <= half_height
        )
        return mask

    def _current_detector_mask(self):
        """Return the detector mask for the current virtual-image ROI."""
        cx, cy = self.roi_center_col, self.roi_center_row
        if self.roi_mode == "circle" and self.roi_radius > 0:
            return self._create_circular_mask(cx, cy, self.roi_radius)
        if self.roi_mode == "square" and self.roi_radius > 0:
            return self._create_square_mask(cx, cy, self.roi_radius)
        if self.roi_mode == "annular" and self.roi_radius > 0:
            return self._create_annular_mask(
                cx, cy, self.roi_radius_inner, self.roi_radius
            )
        if self.roi_mode == "rect" and self.roi_width > 0 and self.roi_height > 0:
            return self._create_rect_mask(
                cx, cy, self.roi_width / 2, self.roi_height / 2
            )

        row = int(max(0, min(round(cy), self._det_shape[0] - 1)))
        col = int(max(0, min(round(cx), self._det_shape[1] - 1)))
        point_mask = np.zeros(self._det_shape, dtype=np.float32)
        point_mask[row, col] = 1.0
        return point_mask

    def _compare_panel_title_for_index(self, panel: int) -> str:
        """Return the user-facing label for a compare panel."""
        if 0 <= panel < len(self.frame_labels) and self.frame_labels[panel]:
            return str(self.frame_labels[panel])
        return f"{self.frame_dim_label} {panel + 1}"

    def _resolve_compare_panel_ref(self, panel: int | str) -> int:
        """Resolve a compare panel index or exact label into a frame index."""
        if isinstance(panel, bool):
            raise TypeError("panel must be an integer index or exact label, not bool")
        if isinstance(panel, int):
            idx = int(panel)
            if 0 <= idx < int(self.n_frames):
                return idx
            raise ValueError(f"panel index {idx} out of range [0, {self.n_frames})")
        if isinstance(panel, str):
            titles = [
                self._compare_panel_title_for_index(i)
                for i in range(int(self.n_frames))
            ]
            matches = [i for i, title in enumerate(titles) if title == panel]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError(
                    f"panel label {panel!r} is not unique; use a zero-based panel index instead"
                )
            available = ", ".join(repr(title) for title in titles)
            raise ValueError(
                f"unknown panel label {panel!r}; available labels: {available}"
            )
        raise TypeError(
            f"panel must be an integer index or exact label, got {type(panel).__name__}"
        )

    def _normalize_compare_panel_refs(
        self,
        panels: Sequence[int | str] | int | str,
        *,
        allow_empty: bool = False,
    ) -> list[int]:
        """Resolve and de-duplicate compare panel references."""
        if isinstance(panels, (str, int)) and not isinstance(panels, bool):
            values: Sequence[int | str] = [panels]
        else:
            values = panels  # type: ignore[assignment]
        out: list[int] = []
        seen: set[int] = set()
        for panel in values:
            idx = self._resolve_compare_panel_ref(panel)
            if idx not in seen:
                out.append(idx)
                seen.add(idx)
        if not out and not allow_empty:
            raise ValueError("at least one compare panel index or label is required")
        return out

    @property
    def compare_ordered_panels(self) -> list[int]:
        """Frame/dataset indices in the current compare display order."""
        order = list(self.compare_panel_order or [])
        n = int(self.n_frames)
        if len(order) == n and sorted(order) == list(range(n)):
            return [int(idx) for idx in order]
        return list(range(n))

    @property
    def compare_visible_panels(self) -> list[int]:
        """Frame/dataset indices currently visible in compare mode."""
        hidden = {
            int(idx)
            for idx in self.compare_hidden_panels
            if not isinstance(idx, bool) and 0 <= int(idx) < int(self.n_frames)
        }
        return [idx for idx in self.compare_ordered_panels if idx not in hidden]

    def set_compare_hidden_panels(
        self, panels: Sequence[int | str] | int | str
    ) -> Self:
        """Replace the hidden compare panel set by index or exact label."""
        hidden = self._normalize_compare_panel_refs(panels, allow_empty=True)
        if len(hidden) >= int(self.n_frames):
            raise ValueError(
                "set_compare_hidden_panels would hide every panel; leave at least one visible"
            )
        self.compare_hidden_panels = sorted(hidden)
        return self

    def hide_compare_panel(self, *panels: int | str) -> Self:
        """Hide one or more compare panels by zero-based index or exact label."""
        to_hide = set(self.compare_hidden_panels)
        to_hide.update(self._normalize_compare_panel_refs(list(panels)))
        if len(to_hide) >= int(self.n_frames):
            raise ValueError(
                "hide_compare_panel would hide every panel; leave at least one visible"
            )
        self.compare_hidden_panels = sorted(to_hide)
        return self

    def show_compare_panel(self, *panels: int | str) -> Self:
        """Restore one or more hidden compare panels by index or exact label."""
        to_show = set(self._normalize_compare_panel_refs(list(panels)))
        self.compare_hidden_panels = sorted(set(self.compare_hidden_panels) - to_show)
        return self

    def show_all_compare_panels(self) -> Self:
        """Restore every compare panel."""
        self.compare_hidden_panels = []
        return self

    @staticmethod
    def _master_key(master) -> str:
        return str(pathlib.Path(master).expanduser().resolve())

    def _attach_folder_source(
        self,
        *,
        folder,
        pattern: str,
        recursive: bool,
        scan_shape: tuple[int, int] | None,
        ready_only: bool,
        known_masters: Sequence,
        make_loader,
        validate_master=None,
        register_master=None,
        preload_all_if_fits: bool = True,
        warm_cache: bool = False,
    ) -> Self:
        """Attach lazy-folder metadata used by poll_folder/watch_folder."""
        self._folder_source = {
            "folder": pathlib.Path(folder).expanduser().resolve(),
            "pattern": str(pattern),
            "recursive": bool(recursive),
            "scan_shape": scan_shape,
            "ready_only": bool(ready_only),
            "make_loader": make_loader,
            "validate_master": validate_master,
            "register_master": register_master,
            "preload_all_if_fits": bool(preload_all_if_fits),
            "warm_cache": bool(warm_cache),
        }
        self._folder_known_masters = {
            self._master_key(master) for master in known_masters
        }
        self._folder_watch_stop = None
        self._folder_watch_thread = None
        self._folder_poll_lock = threading.Lock()
        self._compare_folder_refresh_pending = False
        return self

    def preload_all_datasets(self, *, background: bool = True) -> Self:
        """Keep every unhidden lazy dataset in VRAM when the full set fits.

        The decision uses :meth:`Dataset5dstem.residency_plan`, which knows each
        frame's shape and dtype without reading it. If memory becomes unavailable
        after planning, Show4DSTEM retains full-resolution paging and reports the
        fallback without surfacing a raw backend allocation error.
        """
        data = getattr(self, "_data", None)
        if type(data).__name__ != "Dataset5dstem" or not getattr(
            data, "is_lazy", False
        ):
            return self
        hidden = {
            int(idx)
            for idx in self.compare_hidden_panels
            if not isinstance(idx, bool) and 0 <= int(idx) < int(self.n_frames)
        }
        wanted = [idx for idx in range(int(self.n_frames)) if idx not in hidden]
        plan = data.residency_plan(wanted)
        self._raw_residency_plan = plan
        if not bool(plan.get("fits", False)):
            self._raw_preload_status = "paged"
            self._update_gpu_memory_status()
            return self

        thread = getattr(self, "_full_residency_preload_thread", None)
        if thread is not None and thread.is_alive():
            self._full_residency_preload_retry = True
            return self

        self._raw_preload_status = "loading"
        stop = threading.Event()
        self._dataset_preload_stop = stop
        self._update_gpu_memory_status()

        def _worker() -> None:
            try:
                result = data.preload_all_if_fits(wanted, cancelled=stop.is_set)
                current_hidden = {
                    int(idx)
                    for idx in self.compare_hidden_panels
                    if not isinstance(idx, bool)
                    and 0 <= int(idx) < int(self.n_frames)
                }
                if current_hidden:
                    data.release(idx=sorted(current_hidden))
                current_wanted = [
                    idx for idx in range(int(self.n_frames)) if idx not in current_hidden
                ]
                resident = set(data.vram_resident())
                complete = all(idx in resident for idx in current_wanted)
                result["resident"] = complete
                result["resident_count"] = sum(
                    idx in resident for idx in current_wanted
                )
                self._raw_residency_plan = result
                self._raw_preload_status = "resident" if complete else "paged"
                if complete:
                    self._clear_gpu_memory_warning()
                elif bool(result.get("fits", False)) and not stop.is_set():
                    self.memory_warning = (
                        "GPU memory limited: available memory changed while loading "
                        "the complete dataset series. Continuing with full-resolution "
                        "paging; close other GPU work or choose more GPUs to keep every "
                        "dataset resident."
                    )
            except BaseException:
                self._raw_preload_status = "paged"
                self.memory_warning = (
                    "GPU memory limited: the complete dataset preload could not finish. "
                    "Continuing with full-resolution paging; close other GPU work or "
                    "choose more GPUs to retry."
                )
            finally:
                self._full_residency_preload_thread = None
                self._dataset_preload_stop = None
                self._update_gpu_memory_status()
                if not stop.is_set() and bool(
                    getattr(self, "_full_residency_preload_retry", False)
                ):
                    self._full_residency_preload_retry = False
                    self.preload_all_datasets(background=True)

        if background:
            thread = threading.Thread(
                target=_worker,
                name="Show4DSTEM-preload-all",
                daemon=True,
            )
            self._full_residency_preload_thread = thread
            thread.start()
        else:
            _worker()
        return self

    def stop_dataset_preload(self, *, wait: bool = False) -> Self:
        """Stop automatic all-dataset loading after the current file read."""
        stop = getattr(self, "_dataset_preload_stop", None)
        if stop is not None:
            stop.set()
        thread = getattr(self, "_full_residency_preload_thread", None)
        if wait and thread is not None and thread is not threading.current_thread():
            thread.join()
        return self

    def wait_for_dataset_preload(self, timeout: float | None = None) -> Self:
        """Wait for an automatic all-dataset preload, primarily for verification."""
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        while True:
            thread = getattr(self, "_full_residency_preload_thread", None)
            if thread is None:
                break
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            thread.join(timeout=remaining)
            if thread.is_alive() or (
                deadline is not None and time.monotonic() >= deadline
            ):
                break
        return self

    def poll_folder(self) -> list[int]:
        """Append newly ready folder masters as lazy frames.

        Only widgets created by ``Show4DSTEM.from_folder(...)`` have a folder
        source attached. New masters start as lazy slots, then join the complete
        series preload when the updated shape/dtype footprint still fits.
        """
        source = getattr(self, "_folder_source", None)
        if source is None:
            raise RuntimeError(
                "poll_folder() is available only on Show4DSTEM.from_folder(...) widgets."
            )
        if type(getattr(self, "_data", None)).__name__ != "Dataset5dstem":
            raise RuntimeError("poll_folder() requires a lazy Dataset5dstem backing.")
        from quantem.widget.io import discover_masters, is_master_ready

        poll_lock = getattr(self, "_folder_poll_lock", None)
        if poll_lock is None:
            poll_lock = threading.Lock()
            self._folder_poll_lock = poll_lock
        with poll_lock:
            masters = discover_masters(
                str(source["folder"]),
                pattern=source["pattern"],
                recursive=source["recursive"],
                scan_shape=source["scan_shape"],
                verbose=False,
            )
            if source["ready_only"]:
                masters = [master for master in masters if is_master_ready(master)]
            known = set(getattr(self, "_folder_known_masters", set()))
            added: list[int] = []
            labels = list(self.frame_labels)
            old_n_frames = int(self.n_frames)
            old_order = list(self.compare_panel_order or [])
            custom_order = (
                len(old_order) == old_n_frames
                and sorted(int(idx) for idx in old_order) == list(range(old_n_frames))
            )
            candidates = []
            for master in masters:
                key = self._master_key(master)
                if key in known:
                    continue
                try:
                    validator = source.get("validate_master")
                    if callable(validator):
                        validator(master)
                    idx = len(self._data) + len(candidates)
                    label, loader = source["make_loader"](master, idx)
                except Exception:
                    # Incomplete or incompatible files remain unknown so a
                    # subsequent poll can retry after acquisition finishes.
                    continue
                candidates.append((master, key, idx, str(label), loader))
            for master, key, idx, label, loader in candidates:
                self._data.append_lazy_frame(loader)
                labels.append(label)
            if old_n_frames <= 1 < len(self._data):
                page_config = getattr(self, "_dataset_page_config", None)
                if page_config is not None:
                    self._data.page(**page_config)
            registrar = source.get("register_master")
            if callable(registrar):
                for master, _, idx, _, _ in candidates:
                    registrar(master, idx)
            for _, key, idx, _, _ in candidates:
                known.add(key)
                added.append(idx)
            self._folder_known_masters = known
            if not added:
                return []

            # n_frames normally triggers a compare-grid render. A watched master
            # must first remain a cold lazy slot, so publish only lightweight
            # metadata here. Page selection or cache warming performs the first
            # raw load explicitly.
            self._suppress_folder_append_refresh = True
            try:
                with self.hold_trait_notifications():
                    self.n_frames = len(self._data)
                    self.frame_labels = labels
                    if custom_order:
                        self.compare_panel_order = [*old_order, *added]
                    if self.view_mode == "multiple":
                        ordered = self._compare_ordered_ready_indices()
                        self._compare_visible_total = len(ordered)
                        self._update_compare_page_state(ordered)
                        self.compare_status = self._compare_status_for_indices(
                            self.compare_panel_indices,
                            all_groups=(
                                self._normalise_compare_group_mode(
                                    self.compare_group_mode
                                )
                                == "all"
                            ),
                        )
            finally:
                self._suppress_folder_append_refresh = False

            self._raw_preload_status = "paged"
            self._update_gpu_memory_status()
            if bool(source.get("preload_all_if_fits", False)):
                self.preload_all_datasets(background=True)
            if bool(source.get("warm_cache", False)):
                self._compare_folder_refresh_pending = True
                self.warm_compare_cache(background=True)
            return added

    def watch_folder(self, *, interval: float = 2.0) -> Self:
        """Poll the attached folder in the background and append ready masters."""
        source = getattr(self, "_folder_source", None)
        if source is None:
            raise RuntimeError(
                "watch_folder() is available only on Show4DSTEM.from_folder(...) widgets."
            )
        self.stop_folder_watch()
        import threading

        stop = threading.Event()
        self._folder_watch_stop = stop

        def _worker() -> None:
            while not stop.wait(float(interval)):
                try:
                    self.poll_folder()
                except Exception:
                    # A live microscope folder can be transiently inconsistent
                    # while files are being copied. Keep the mounted viewer alive.
                    continue

        self._folder_watch_thread = threading.Thread(
            target=_worker,
            name="Show4DSTEM-folder-watch",
            daemon=True,
        )
        self._folder_watch_thread.start()
        return self

    def stop_folder_watch(self) -> None:
        """Stop the background folder watcher, if one was started."""
        stop = getattr(self, "_folder_watch_stop", None)
        thread = getattr(self, "_folder_watch_thread", None)
        if stop is not None:
            stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        if getattr(self, "_folder_watch_stop", None) is stop:
            self._folder_watch_stop = None
        if getattr(self, "_folder_watch_thread", None) is thread:
            self._folder_watch_thread = None

    def close(self) -> None:
        """Stop background work and close the widget comm."""
        self.stop_folder_watch()
        self.stop_dataset_preload(wait=True)
        self.stop_compare_cache_warm(wait=True)
        super().close()

    def _candidate_cuda_memory_devices(self) -> list[int]:
        """CUDA device indices relevant to this widget's data."""
        indices: set[int] = set()

        def add_device(value) -> None:
            if value is None:
                return
            try:
                device = torch.device(
                    value.device if isinstance(value, torch.Tensor) else value
                )
            except Exception:
                return
            if device.type == "cuda":
                indices.add(0 if device.index is None else int(device.index))

        add_device(getattr(self, "_device", None))
        data = getattr(self, "_data", None)
        if type(data).__name__ == "Dataset5dstem" and hasattr(data, "devices"):
            for device in data.devices:
                add_device(device)
        elif isinstance(data, torch.Tensor):
            add_device(data)
        elif data is not None and hasattr(data, "device"):
            add_device(getattr(data, "device", None))
        return sorted(indices)

    def _uses_mps_memory(self) -> bool:
        """Return True when this widget's data is actually on/planned for MPS."""
        found = False

        def add_device(value) -> None:
            nonlocal found
            if value is None:
                return
            try:
                device = torch.device(
                    value.device if isinstance(value, torch.Tensor) else value
                )
            except Exception:
                return
            if device.type == "mps":
                found = True

        add_device(getattr(self, "_device", None))
        data = getattr(self, "_data", None)
        if type(data).__name__ == "Dataset5dstem" and hasattr(data, "devices"):
            for device in data.devices:
                add_device(device)
        elif isinstance(data, torch.Tensor):
            add_device(data)
        elif data is not None and hasattr(data, "device"):
            add_device(getattr(data, "device", None))
        return found

    def _format_gpu_memory_label(self) -> str:
        """Compact memory text for the top widget chrome."""
        try:
            if torch.cuda.is_available():
                parts = []
                for idx in self._candidate_cuda_memory_devices():
                    with torch.cuda.device(idx):
                        free, total = torch.cuda.mem_get_info(idx)
                    used = int(total - free)
                    parts.append(
                        f"cuda:{idx} {_format_memory(used)}/{_format_memory(int(total))}"
                    )
                if parts:
                    label = "GPU " + " | ".join(parts)
                    data = getattr(self, "_data", None)
                    if type(data).__name__ == "Dataset5dstem" and getattr(
                        data, "is_lazy", False
                    ):
                        hidden = {
                            int(idx)
                            for idx in self.compare_hidden_panels
                            if not isinstance(idx, bool)
                            and 0 <= int(idx) < int(self.n_frames)
                        }
                        wanted = [
                            idx
                            for idx in range(int(self.n_frames))
                            if idx not in hidden
                        ]
                        resident = set(data.vram_resident())
                        resident_count = sum(idx in resident for idx in wanted)
                        status = str(getattr(self, "_raw_preload_status", ""))
                        suffix = " loading" if status == "loading" else ""
                        label += (
                            f" · raw {resident_count}/{len(wanted)} resident{suffix}"
                        )
                    return label
            mps_available = bool(
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            )
            if mps_available and self._uses_mps_memory():
                live = (
                    torch.mps.current_allocated_memory()
                    if hasattr(torch.mps, "current_allocated_memory")
                    else 0
                )
                driver = (
                    torch.mps.driver_allocated_memory()
                    if hasattr(torch.mps, "driver_allocated_memory")
                    else 0
                )
                return f"GPU mps {_format_memory(int(live))} live / {_format_memory(int(driver))} driver"
        except Exception:
            return ""
        return ""

    def _update_gpu_memory_status(self) -> None:
        """Refresh the synced top-level GPU memory label."""
        label = self._format_gpu_memory_label()
        if label != self.gpu_memory_label:
            self.gpu_memory_label = label

    def _clear_gpu_memory_warning(self) -> None:
        if self.memory_warning.startswith("GPU memory limited:"):
            self.memory_warning = ""

    def _set_gpu_memory_warning(
        self,
        *,
        action: str,
        requested: int | None = None,
        shown: int | None = None,
    ) -> None:
        """Set a user-facing memory warning without leaking backend OOM text."""
        self._update_gpu_memory_status()
        shown_text = ""
        if requested is not None and shown is not None:
            shown_text = f" Showing {shown}/{requested} requested panels."
        memory_text = (
            f" Current memory: {self.gpu_memory_label}."
            if self.gpu_memory_label
            else ""
        )
        self.memory_warning = (
            f"GPU memory limited: not enough free VRAM to {action}.{shown_text}{memory_text} "
            "Free memory with w.free() or cleanup(w), close other GPU jobs, restart the kernel, "
            "or reopen with a smaller page_budget/compare_max_panels."
        )

    def _ensure_current_compare_frame_visible(self) -> None:
        """Keep selected-DP mode pointing at a visible compare-page panel."""
        if self.n_frames <= 1:
            return
        visible_page = self._compare_ready_indices()
        if int(self.frame_idx) in visible_page:
            return
        if visible_page:
            self.frame_idx = int(visible_page[0])

    def _sync_compare_page_to_frame_idx(self) -> None:
        """Show the compare-grid page that contains the active frame."""
        if self.n_frames <= 1 or self.view_mode != "multiple":
            return
        visible = self._compare_ordered_ready_indices()
        frame_idx = int(self.frame_idx)
        if frame_idx not in visible:
            return
        if frame_idx in self._compare_hidden_panel_set():
            return
        max_panels = max(1, int(self.compare_max_panels))
        page_idx = visible.index(frame_idx) // max_panels
        page_idx = int(max(0, min(page_idx, max(1, int(self.compare_page_count)) - 1)))
        if int(self.compare_page_idx) != page_idx:
            self.compare_page_idx = page_idx

    def _release_hidden_compare_panel_data(self, panels: Sequence[int]) -> None:
        """Drop lazy frame tensors for hidden compare panels without renumbering."""
        data = getattr(self, "_data", None)
        if not panels or data is None:
            return
        if type(data).__name__ != "Dataset5dstem" or not getattr(
            data, "is_lazy", False
        ):
            return
        # Drop compute views first; they may be the last reference to a frame that
        # is about to become a cold lazy slot.
        self._compute_backend = None
        self._compute_for = None
        try:
            released = data.release(idx=list(panels))
        except Exception:
            released = []
        if released:
            self._update_gpu_memory_status()

    def _handle_compare_hidden_change(self, old, new) -> None:
        old_hidden = {
            int(idx)
            for idx in (old or [])
            if not isinstance(idx, bool) and 0 <= int(idx) < int(self.n_frames)
        }
        new_hidden = {
            int(idx)
            for idx in (new or [])
            if not isinstance(idx, bool) and 0 <= int(idx) < int(self.n_frames)
        }
        if len(new_hidden) >= int(self.n_frames):
            keep = next(
                (idx for idx in self.compare_ordered_panels if idx not in old_hidden), 0
            )
            new_hidden.discard(int(keep))
            self.compare_hidden_panels = sorted(new_hidden)
        self._ensure_current_compare_frame_visible()
        self._release_hidden_compare_panel_data(sorted(new_hidden - old_hidden))

    def set_compare_panel_order(self, panels: Sequence[int | str]) -> Self:
        """Set the compare-grid display order by panel index or exact label."""
        order = self._normalize_compare_panel_refs(panels, allow_empty=True)
        if not order:
            self.compare_panel_order = []
            return self
        expected = set(range(int(self.n_frames)))
        if len(order) != int(self.n_frames) or set(order) != expected:
            raise ValueError(
                "set_compare_panel_order requires every compare panel exactly once"
            )
        self.compare_panel_order = order
        return self

    def reset_compare_panel_order(self) -> Self:
        """Restore the natural compare panel order."""
        self.compare_panel_order = []
        return self

    def move_compare_panel(self, panel: int | str, position: int) -> Self:
        """Move one compare panel to a zero-based display position."""
        idx = self._resolve_compare_panel_ref(panel)
        order = self.compare_ordered_panels
        order.remove(idx)
        pos = max(0, min(int(position), len(order)))
        order.insert(pos, idx)
        self.compare_panel_order = order
        return self

    def set_compare_page(self, page: int) -> Self:
        """Show a zero-based page of compare-grid panels."""
        page_count = max(1, int(self.compare_page_count))
        self.compare_page_idx = int(max(0, min(int(page), page_count - 1)))
        return self

    def next_compare_page(self) -> Self:
        """Advance the compare grid by one page."""
        return self.set_compare_page(int(self.compare_page_idx) + 1)

    def previous_compare_page(self) -> Self:
        """Move the compare grid back by one page."""
        return self.set_compare_page(int(self.compare_page_idx) - 1)

    def show_compare_paged_groups(self) -> Self:
        """Show one compare-grid group/page at a time."""
        self.compare_group_mode = "paged"
        return self

    def show_compare_all_groups(self) -> Self:
        """Collapse all visible compare-grid groups into one dense grid."""
        self.compare_group_mode = "all"
        return self

    def set_compare_starred_panels(
        self, panels: Sequence[int | str] | int | str
    ) -> Self:
        """Replace the set of starred compare panels by index or exact label."""
        self.compare_starred_panels = sorted(
            self._normalize_compare_panel_refs(panels, allow_empty=True)
        )
        return self

    def star_compare_panel(self, panel: int | str) -> Self:
        """Mark a compare panel with a star."""
        idx = self._resolve_compare_panel_ref(panel)
        self.compare_starred_panels = sorted(set(self.compare_starred_panels) | {idx})
        return self

    def unstar_compare_panel(self, panel: int | str) -> Self:
        """Clear the star on a compare panel."""
        idx = self._resolve_compare_panel_ref(panel)
        self.compare_starred_panels = sorted(set(self.compare_starred_panels) - {idx})
        return self

    def _clear_compare_virtual_images(self, status: str = "") -> None:
        with self.hold_trait_notifications():
            self.compare_virtual_image_bytes = b""
            self.compare_panel_count = 0
            self.compare_panel_indices = []
            self.compare_status = status

    def _empty_compare_page_status(self) -> str:
        """Status for a valid compare page that has no visible panels."""
        if self.view_mode != "multiple" or self.n_frames <= 1:
            return ""
        ordered_ready = self._compare_ordered_ready_indices()
        if not ordered_ready:
            return ""
        page_idx, page_count, max_panels = self._update_compare_page_state(
            ordered_ready
        )
        start = page_idx * max_panels
        page = ordered_ready[start : start + max_panels]
        if not page:
            return ""
        hidden = self._compare_hidden_panel_set()
        if any(idx not in hidden for idx in page):
            return ""
        page_suffix = f" · page {page_idx + 1}/{page_count}" if page_count > 1 else ""
        return f"0/{len(ordered_ready)} {self.frame_dim_label.lower()} panels · hidden{page_suffix}"

    def _compare_hidden_panel_set(self) -> set[int]:
        """Hidden compare panels as a validated set of frame indices."""
        return {
            int(idx)
            for idx in self.compare_hidden_panels
            if not isinstance(idx, bool) and 0 <= int(idx) < int(self.n_frames)
        }

    def _compare_ordered_ready_indices(self) -> list[int]:
        """Ready compare panels after user ordering, before hidden filtering."""
        if self.n_frames <= 1:
            self.compare_page_count = 1
            self.compare_page_idx = 0
            return []
        data = getattr(self, "_data", None)
        datasets = getattr(data, "datasets", None)
        if datasets is not None:
            ready = [
                idx for idx, dataset in enumerate(list(datasets)) if dataset is not None
            ]
        else:
            ready = list(range(int(self.n_frames)))
        ready_set = set(ready)
        return [idx for idx in self.compare_ordered_panels if idx in ready_set]

    def _compare_visible_ready_indices(self) -> list[int]:
        """Ready compare panels after ordering and hidden-panel filtering."""
        ordered_ready = self._compare_ordered_ready_indices()
        hidden = self._compare_hidden_panel_set()
        visible = [idx for idx in ordered_ready if idx not in hidden]
        if not visible and ordered_ready:
            # Frontend/state load should not be able to leave compare mode blank.
            keep = ordered_ready[0]
            self.compare_hidden_panels = sorted(idx for idx in hidden if idx != keep)
            visible = [keep]
        self._compare_visible_total = len(visible)
        return visible

    def _update_compare_page_state(
        self, visible: Sequence[int]
    ) -> tuple[int, int, int]:
        """Clamp compare page traits and return ``(page_idx, page_count, page_size)``."""
        max_panels = max(1, int(self.compare_max_panels))
        page_count = max(1, int(np.ceil(len(visible) / max_panels))) if visible else 1
        if int(self.compare_page_count) != page_count:
            self.compare_page_count = page_count
        page_idx = int(max(0, min(int(self.compare_page_idx), page_count - 1)))
        if int(self.compare_page_idx) != page_idx:
            self.compare_page_idx = page_idx
        return page_idx, page_count, max_panels

    def _compare_current_page_indices(self) -> list[int]:
        """Ready visible compare panels on the active page."""
        ordered_ready = self._compare_ordered_ready_indices()
        self._compare_visible_total = len(ordered_ready)
        page_idx, _, max_panels = self._update_compare_page_state(ordered_ready)
        start = page_idx * max_panels
        page = ordered_ready[start : start + max_panels]
        hidden = self._compare_hidden_panel_set()
        return [idx for idx in page if idx not in hidden]

    def _compare_ready_indices(self) -> list[int]:
        if self._normalise_compare_group_mode(self.compare_group_mode) == "all":
            visible = self._compare_visible_ready_indices()
            self._update_compare_page_state(self._compare_ordered_ready_indices())
            return list(visible)
        return self._compare_current_page_indices()

    def _virtual_image_for_chunked_dataset(self, dataset, mask) -> np.ndarray:
        """Compute one compare tile for a chunked MPS dataset slot."""
        from quantem.widget.kernels.compute.backends import compute_backend

        mask_np = (
            mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
        )
        backend = compute_backend(dataset)
        return np.asarray(backend.masked_sum(mask_np), dtype=np.float32)

    def _detector_mask_area(self, mask) -> float:
        """Number of detector pixels contributing to the current ROI."""
        if hasattr(mask, "detach"):
            area = float(mask.detach().float().sum().item())
        else:
            area = float(np.asarray(mask, dtype=np.float32).sum())
        return max(1.0, area)

    def _preload_compare_page(self, indices: Sequence[int]) -> None:
        """Ask a lazy backing series to load the visible compare page together."""
        data = getattr(self, "_data", None)
        if type(data).__name__ != "Dataset5dstem" or not hasattr(data, "preload"):
            return
        # Drop cached compute views before the backing series releases old page
        # frames. Otherwise _compute_for can keep an evicted 18 GB frame alive.
        self._compute_backend = None
        self._compute_for = None
        try:
            data.preload([int(idx) for idx in indices])
        except AttributeError:
            return

    def _compare_virtual_image_for_frame(self, idx: int, mask) -> np.ndarray:
        # Compare panels are visual previews across many datasets. Normalize by
        # detector ROI area so changing from BF to ADF does not flatten/saturate
        # panels purely because the detector mask covers more pixels. The main
        # VI remains summed for quantitative count integration.
        mask_area = self._detector_mask_area(mask)
        data = getattr(self, "_data", None)
        datasets = getattr(data, "datasets", None)
        if datasets is not None:
            dataset = datasets[int(idx)]
            if dataset is None:
                raise ValueError(f"dataset {idx} is not ready")
            vi = self._virtual_image_for_chunked_dataset(dataset, mask)
        else:
            vi = self._virtual_image_for_frame(int(idx))
        return np.asarray(vi, dtype=np.float32) / mask_area

    def _compare_status_for_indices(
        self,
        indices: Sequence[int],
        *,
        all_groups: bool,
        partial_reason: str | None = None,
    ) -> str:
        """Human-readable compare-grid status for the current visible selection."""
        visible_total = int(getattr(self, "_compare_visible_total", self.n_frames))
        label = self.frame_dim_label.lower()
        if all_groups:
            suffix = (
                f" · {partial_reason}"
                if partial_reason
                else " ready"
                if len(indices) < visible_total
                else ""
            )
            group_suffix = " · all groups" if int(self.compare_page_count) > 1 else ""
            return (
                f"{len(indices)}/{visible_total} {label} panels{suffix}{group_suffix}"
            )

        max_panels = max(1, int(self.compare_max_panels))
        page_idx = max(0, int(self.compare_page_idx))
        ordered_ready = self._compare_ordered_ready_indices()
        start = page_idx * max_panels
        expected_on_page = len(ordered_ready[start : start + max_panels])
        suffix = (
            f" · {partial_reason}"
            if partial_reason
            else ""
            if len(indices) >= expected_on_page
            else " · hidden"
        )
        page_suffix = (
            f" · page {int(self.compare_page_idx) + 1}/{int(self.compare_page_count)}"
            if int(self.compare_page_count) > 1
            else ""
        )
        return f"{len(indices)}/{visible_total} {label} panels{page_suffix}{suffix}"

    def _compare_virtual_images_for_display_indices(
        self,
        indices: Sequence[int],
        mask,
    ) -> list[np.ndarray]:
        """Compute compare-grid virtual images without overloading lazy backing data."""
        data = getattr(self, "_data", None)
        batch_builder = getattr(data, "preload_batches", None)
        if callable(batch_builder):
            batches = batch_builder(indices)
        elif self._normalise_compare_group_mode(self.compare_group_mode) == "all":
            batch_size = max(1, int(self.compare_max_panels))
            batches = [
                [int(idx) for idx in indices[start : start + batch_size]]
                for start in range(0, len(indices), batch_size)
            ]
        else:
            batches = [[int(idx) for idx in indices]]

        images: list[np.ndarray] = []
        for batch in batches:
            cached = self._get_cached_compare_preset(batch)
            if cached is not None:
                payload, cached_indices, _ = cached
                batch_stack = np.frombuffer(payload, dtype=np.float32).reshape(
                    len(cached_indices),
                    self.shape_rows,
                    self.shape_cols,
                )
                images.extend(
                    np.ascontiguousarray(image, dtype=np.float32)
                    for image in batch_stack
                )
                continue
            with self._compare_compute_lock:
                # A background warmer may have filled this page while the UI
                # waited for the GPU compute lock.
                cached = self._get_cached_compare_preset(batch)
                if cached is not None:
                    payload, cached_indices, _ = cached
                    batch_stack = np.frombuffer(payload, dtype=np.float32).reshape(
                        len(cached_indices),
                        self.shape_rows,
                        self.shape_cols,
                    )
                    images.extend(
                        np.ascontiguousarray(image, dtype=np.float32)
                        for image in batch_stack
                    )
                    continue

                self._preload_compare_page(batch)
                batch_images = self._compare_virtual_images_for_indices(batch, mask)
                if not batch_images:
                    continue
                batch_indices = tuple(batch[: len(batch_images)])
                batch_stack = np.ascontiguousarray(
                    np.stack(batch_images, axis=0),
                    dtype=np.float32,
                )
                self._store_cached_compare_preset(
                    batch_stack.tobytes(),
                    batch_indices,
                    self._compare_status_for_indices(
                        batch_indices,
                        all_groups=False,
                    ),
                )
            images.extend(batch_images)
        return images

    def _refresh_compare_virtual_images(self) -> None:
        """Build the lightweight virtual-image stack used by compare mode."""
        if getattr(self, "_data", None) is None:
            self._clear_compare_virtual_images()
            return
        if self.view_mode != "multiple":
            if self.compare_panel_count or self.compare_virtual_image_bytes:
                self._clear_compare_virtual_images()
            return
        indices = self._compare_ready_indices()
        if not indices:
            self._clear_compare_virtual_images(self._empty_compare_page_status())
            return
        if getattr(self, "_suppress_compare_recompute", False):
            return
        self._suppress_compare_recompute = True
        data = getattr(self, "_data", None)
        loaded_before = (
            tuple(data.loaded_indices())
            if type(data).__name__ == "Dataset5dstem"
            and hasattr(data, "loaded_indices")
            else None
        )
        try:
            cached = self._get_cached_compare_preset(indices)
            if cached is not None:
                payload, cached_indices, _ = cached
                cached_partial_reason = None
                if len(cached_indices) < len(indices):
                    cached_partial_reason = "memory limited"
                    self._set_gpu_memory_warning(
                        action="show the requested multiple-panel page",
                        requested=len(indices),
                        shown=len(cached_indices),
                    )
                else:
                    self._clear_gpu_memory_warning()
                with self.hold_trait_notifications():
                    self.compare_virtual_image_bytes = payload
                    self.compare_panel_count = len(cached_indices)
                    self.compare_panel_indices = list(cached_indices)
                    self.compare_status = self._compare_status_for_indices(
                        cached_indices,
                        all_groups=(
                            self._normalise_compare_group_mode(self.compare_group_mode)
                            == "all"
                        ),
                        partial_reason=cached_partial_reason,
                    )
                return
            mask = self._current_detector_mask()
            images = self._compare_virtual_images_for_display_indices(indices, mask)
            if not images:
                self._set_gpu_memory_warning(
                    action="compute any multiple-panel virtual images",
                    requested=len(indices),
                    shown=0,
                )
                with self.hold_trait_notifications():
                    self.compare_virtual_image_bytes = b""
                    self.compare_panel_count = 0
                    self.compare_panel_indices = []
                    self.compare_status = (
                        "GPU memory limited: multiple grid page was not computed."
                    )
                return
            shown_indices = [int(idx) for idx in indices[: len(images)]]
            partial_reason = None
            if len(shown_indices) < len(indices):
                partial_reason = "memory limited"
                self._set_gpu_memory_warning(
                    action="show the requested multiple-panel page",
                    requested=len(indices),
                    shown=len(shown_indices),
                )
            else:
                self._clear_gpu_memory_warning()
            stack = np.ascontiguousarray(np.stack(images, axis=0), dtype=np.float32)
            with self.hold_trait_notifications():
                self.compare_virtual_image_bytes = stack.tobytes()
                self.compare_panel_count = len(shown_indices)
                self.compare_panel_indices = shown_indices
                self.compare_status = self._compare_status_for_indices(
                    shown_indices,
                    all_groups=self._normalise_compare_group_mode(
                        self.compare_group_mode
                    )
                    == "all",
                    partial_reason=partial_reason,
                )
            self._store_cached_compare_preset(
                stack.tobytes(),
                tuple(shown_indices),
                self.compare_status,
            )
        except BaseException as exc:
            if _is_recoverable_allocation_error(exc):
                self._reclaim_compare_allocation_failure()
                self._set_gpu_memory_warning(
                    action="compute the multiple-panel page",
                    requested=len(indices),
                    shown=0,
                )
                status = "GPU memory limited: multiple grid page was not computed."
            else:
                status = f"Multiple grid unavailable: {exc}"
            with self.hold_trait_notifications():
                self.compare_virtual_image_bytes = b""
                self.compare_panel_count = 0
                self.compare_panel_indices = []
                self.compare_status = status
        finally:
            if loaded_before is not None:
                loaded_after = tuple(data.loaded_indices())
                if loaded_after != loaded_before:
                    self._update_gpu_memory_status()
            self._suppress_compare_recompute = False

    def _on_calibration_change(self, change=None):
        self._cached_bf_virtual = None
        self._cached_abf_virtual = None
        self._cached_adf_virtual = None
        self._cached_haadf_virtual = None
        self._clear_compare_virtual_page_cache()

    def _precompute_common_virtual_images(self):
        """Pre-compute BF/ABF/ADF/HAADF virtual image bytes. Annular ranges match
        apply_preset() so the cache always hits on preset clicks."""
        cx, cy, bf = self.center_col, self.center_row, self.bf_radius
        self._cached_bf_virtual = self._to_float32_bytes(
            self._fast_masked_sum(self._create_circular_mask(cx, cy, bf))
        )
        self._cached_abf_virtual = self._to_float32_bytes(
            self._fast_masked_sum(self._create_annular_mask(cx, cy, bf * 0.5, bf))
        )
        self._cached_adf_virtual = self._to_float32_bytes(
            self._fast_masked_sum(self._create_annular_mask(cx, cy, bf, bf * 2.0))
        )
        self._cached_haadf_virtual = self._to_float32_bytes(
            self._fast_masked_sum(self._create_annular_mask(cx, cy, bf * 2.0, bf * 4.0))
        )

    def _get_cached_preset(self) -> bytes | None:
        """Return cached preset bytes if current ROI matches BF/ABF/ADF preset shape."""
        # Must be centered on detector center
        if (
            abs(self.roi_center_col - self.center_col) >= 1
            or abs(self.roi_center_row - self.center_row) >= 1
        ):
            return None

        bf = self.bf_radius

        # BF: circle at bf_radius
        if self.roi_mode == "circle" and abs(self.roi_radius - bf) < 1:
            return self._cached_bf_virtual

        # ABF: annular at 0.5*bf to bf
        if (
            self.roi_mode == "annular"
            and abs(self.roi_radius_inner - bf * 0.5) < 1
            and abs(self.roi_radius - bf) < 1
        ):
            return self._cached_abf_virtual

        # ADF: annular at bf to 2*bf
        if (
            self.roi_mode == "annular"
            and abs(self.roi_radius_inner - bf) < 1
            and abs(self.roi_radius - bf * 2.0) < 1
        ):
            return self._cached_adf_virtual

        # HAADF: annular at 2*bf to 4*bf
        if (
            self.roi_mode == "annular"
            and abs(self.roi_radius_inner - bf * 2.0) < 1
            and abs(self.roi_radius - bf * 4.0) < 1
        ):
            return self._cached_haadf_virtual

        return None

    def _preset_cache_name(self) -> str | None:
        """Return the common detector preset name matching the current ROI."""
        if (
            abs(self.roi_center_col - self.center_col) >= 1
            or abs(self.roi_center_row - self.center_row) >= 1
        ):
            return None

        bf = self.bf_radius
        if self.roi_mode == "circle" and abs(self.roi_radius - bf) < 1:
            return "bf"
        if (
            self.roi_mode == "annular"
            and abs(self.roi_radius_inner - bf * 0.5) < 1
            and abs(self.roi_radius - bf) < 1
        ):
            return "abf"
        if (
            self.roi_mode == "annular"
            and abs(self.roi_radius_inner - bf) < 1
            and abs(self.roi_radius - bf * 2.0) < 1
        ):
            return "adf"
        if (
            self.roi_mode == "annular"
            and abs(self.roi_radius_inner - bf * 2.0) < 1
            and abs(self.roi_radius - bf * 4.0) < 1
        ):
            return "haadf"
        return None

    def _clear_compare_virtual_page_cache(self) -> None:
        """Drop cached compare-grid virtual-image pages."""
        self.stop_compare_cache_warm()
        with self._compare_compute_lock:
            self._compare_cache_generation += 1
            with self._compare_cache_lock:
                self._compare_virtual_page_cache.clear()
                self._compare_virtual_page_cache_bytes = 0

    def _get_cached_compare_preset(
        self,
        indices: Sequence[int],
        *,
        preset_name: str | None = None,
    ) -> tuple[bytes, tuple[int, ...], str] | None:
        key = self._compare_preset_cache_key(indices, preset_name=preset_name)
        if key is None:
            return None
        with self._compare_cache_lock:
            cached = self._compare_virtual_page_cache.get(key)
            if cached is None:
                return None
            self._compare_virtual_page_cache.move_to_end(key)
            payload, cached_indices, status, _ = cached
            return payload, cached_indices, status

    def _store_cached_compare_preset(
        self,
        payload: bytes,
        indices: tuple[int, ...],
        status: str,
        *,
        preset_name: str | None = None,
    ) -> None:
        key = self._compare_preset_cache_key(indices, preset_name=preset_name)
        if key is None:
            return
        with self._compare_cache_lock:
            existing = self._compare_virtual_page_cache.pop(key, None)
            if existing is not None:
                self._compare_virtual_page_cache_bytes -= existing[3]
            nbytes = len(payload)
            self._compare_virtual_page_cache[key] = (payload, indices, status, nbytes)
            self._compare_virtual_page_cache_bytes += nbytes
            self._trim_compare_virtual_page_cache()

    def _compare_preset_cache_key(
        self,
        indices: Sequence[int],
        *,
        preset_name: str | None = None,
    ) -> tuple[Any, ...] | None:
        preset = self._preset_cache_name() if preset_name is None else preset_name
        if preset is None or self._compare_cache_pages <= 0:
            return None
        if self._compare_cache_max_bytes == 0:
            return None
        return (
            preset,
            tuple(int(idx) for idx in indices),
            int(self.shape_rows),
            int(self.shape_cols),
            int(self.compare_max_panels),
            str(self.frame_dim_label),
        )

    def _trim_compare_virtual_page_cache(self) -> None:
        max_entries = max(0, int(self._compare_cache_pages))
        max_bytes = self._compare_cache_max_bytes
        while (
            self._compare_virtual_page_cache
            and len(self._compare_virtual_page_cache) > max_entries
        ):
            _, old = self._compare_virtual_page_cache.popitem(last=False)
            self._compare_virtual_page_cache_bytes -= old[3]
        while (
            self._compare_virtual_page_cache
            and max_bytes is not None
            and self._compare_virtual_page_cache_bytes > int(max_bytes)
        ):
            _, old = self._compare_virtual_page_cache.popitem(last=False)
            self._compare_virtual_page_cache_bytes -= old[3]

    def warm_compare_cache(
        self,
        presets: Sequence[str] = ("bf", "abf", "adf", "haadf"),
        *,
        background: bool = True,
    ) -> Self:
        """Cache standard detector views without retaining every raw 4D master.

        Folder-backed data are loaded in memory-aware batches. All requested
        detector presets are reduced while each batch is resident, then only the
        small 2D virtual images remain in host memory.
        """
        if self.view_mode != "multiple" or self.n_frames <= 1:
            return self
        names = tuple(dict.fromkeys(str(name).strip().lower() for name in presets))
        allowed = {"bf", "abf", "adf", "haadf"}
        unknown = [name for name in names if name not in allowed]
        if unknown:
            raise ValueError(
                "presets must contain only 'bf', 'abf', 'adf', or 'haadf'; "
                f"got {unknown!r}"
            )
        if not names or self._compare_cache_max_bytes == 0:
            return self

        self.stop_compare_cache_warm()
        stop = threading.Event()
        self._compare_cache_warm_stop = stop
        generation = int(self._compare_cache_generation)

        ordered = self._compare_ordered_ready_indices()
        hidden = self._compare_hidden_panel_set()
        page_size = max(1, int(self.compare_max_panels))
        pages = [
            [idx for idx in ordered[start : start + page_size] if idx not in hidden]
            for start in range(0, len(ordered), page_size)
        ]
        pages = [page for page in pages if page]
        if not pages:
            return self
        current = min(max(0, int(self.compare_page_idx)), len(pages) - 1)
        pages = pages[current:] + pages[:current]
        self._compare_cache_pages = max(
            int(self._compare_cache_pages),
            len(pages) * len(names),
        )

        masks = {
            name: mask
            for name, _, mask in self._report_preset_masks()
            if name in names
        }

        def worker() -> None:
            self._compare_cache_warm_status = "warming"
            try:
                data = getattr(self, "_data", None)
                batch_builder = getattr(data, "preload_batches", None)
                for page in pages:
                    if stop.is_set() or generation != self._compare_cache_generation:
                        self._compare_cache_warm_status = "stopped"
                        return
                    missing = [
                        name
                        for name in names
                        if self._get_cached_compare_preset(
                            page,
                            preset_name=name,
                        )
                        is None
                    ]
                    if not missing:
                        continue
                    if background:
                        # Yield the compute lock between masters so page clicks
                        # and detector changes stay responsive during a long
                        # full-resolution cache warm.
                        batches = [[int(idx)] for idx in page]
                    else:
                        batches = (
                            batch_builder(page)
                            if callable(batch_builder)
                            else [list(page)]
                        )
                    accumulated: dict[str, list[np.ndarray]] = {
                        name: [] for name in missing
                    }
                    for batch in batches:
                        if stop.is_set() or generation != self._compare_cache_generation:
                            self._compare_cache_warm_status = "stopped"
                            return
                        with self._compare_compute_lock:
                            self._preload_compare_page(batch)
                            for name in missing:
                                cached = self._get_cached_compare_preset(
                                    batch,
                                    preset_name=name,
                                )
                                if cached is not None:
                                    payload, cached_indices, _ = cached
                                    stack = np.frombuffer(
                                        payload,
                                        dtype=np.float32,
                                    ).reshape(
                                        len(cached_indices),
                                        self.shape_rows,
                                        self.shape_cols,
                                    )
                                    batch_images = [
                                        np.ascontiguousarray(image, dtype=np.float32)
                                        for image in stack
                                    ]
                                else:
                                    batch_images = self._compare_virtual_images_for_indices(
                                        batch,
                                        masks[name],
                                    )
                                accumulated[name].extend(batch_images)
                    if stop.is_set() or generation != self._compare_cache_generation:
                        self._compare_cache_warm_status = "stopped"
                        return
                    for name, images in accumulated.items():
                        if len(images) != len(page):
                            continue
                        stack = np.ascontiguousarray(
                            np.stack(images, axis=0),
                            dtype=np.float32,
                        )
                        self._store_cached_compare_preset(
                            stack.tobytes(),
                            tuple(page),
                            "cached",
                            preset_name=name,
                        )
                if stop.is_set() or generation != self._compare_cache_generation:
                    self._compare_cache_warm_status = "stopped"
                    return
                self._compare_cache_warm_status = "ready"
                if bool(getattr(self, "_compare_folder_refresh_pending", False)):
                    self._compare_folder_refresh_pending = False
                    active = self._compare_ready_indices()
                    if self._get_cached_compare_preset(active) is not None:
                        self._refresh_compare_virtual_images()
            except BaseException:
                self._compare_cache_warm_status = "failed"
            finally:
                if self._compare_cache_warm_stop is stop:
                    self._compare_cache_warm_stop = None

        if background:
            thread = threading.Thread(
                target=worker,
                name="Show4DSTEM-compare-cache",
                daemon=True,
            )
            self._compare_cache_warm_thread = thread
            thread.start()
        else:
            worker()
        return self

    def stop_compare_cache_warm(self, *, wait: bool = False) -> None:
        """Stop background detector-preset caching after the current GPU batch."""
        stop = getattr(self, "_compare_cache_warm_stop", None)
        thread = getattr(self, "_compare_cache_warm_thread", None)
        if stop is not None:
            stop.set()
        if wait and thread is not None and thread is not threading.current_thread():
            thread.join()
        if self._compare_cache_warm_stop is stop:
            self._compare_cache_warm_stop = None
        if thread is None or not thread.is_alive():
            if self._compare_cache_warm_thread is thread:
                self._compare_cache_warm_thread = None

    def _frame_data_for_index(self, frame_idx: int):
        if type(self._data).__name__ == "Dataset5dstem":
            return self._data.frame(frame_idx)  # paging-aware (see _frame_data)
        if self.n_frames > 1:
            return self._data[frame_idx]
        return self._data

    def _sparse_masked_sum_tensor_for_frame_data(
        self, data, mask
    ) -> torch.Tensor | None:
        """Compute a frame VI tensor by summing only selected detector pixels.

        Dense tensordot is simple but reads every detector pixel, so it is slow
        for sparse circular/annular detector masks on no-bin 4D-STEM. This path
        keeps the same float32 math while gathering only the mask hits.
        """
        if not isinstance(data, torch.Tensor):
            return None
        data_4d = (
            data
            if data.ndim == 4
            else data.reshape(
                self._scan_shape[0],
                self._scan_shape[1],
                *self._det_shape,
            )
        )
        det_pixels = int(self._det_shape[0] * self._det_shape[1])
        target_device = data_4d.device
        mask_bool = mask.to(device=target_device, dtype=torch.bool).reshape(-1)
        selected = int(mask_bool.sum().item())
        if selected <= 0:
            return torch.zeros(
                self._scan_shape, dtype=torch.float32, device=target_device
            )
        # Above this, dense contiguous tensordot avoids index-gather overhead.
        if selected > det_pixels // 4:
            return None

        cols = torch.nonzero(mask_bool, as_tuple=False).reshape(-1)
        flat = data_4d.reshape(-1, det_pixels)
        out = torch.empty(flat.shape[0], dtype=torch.float32, device=target_device)
        rows_per_chunk = max(
            1,
            _SPARSE_MASK_CHUNK_BYTE_BUDGET // max(1, selected * 4),
        )
        for lo in range(0, flat.shape[0], rows_per_chunk):
            hi = min(flat.shape[0], lo + rows_per_chunk)
            chunk = flat[lo:hi].index_select(1, cols)
            if not torch.is_floating_point(chunk):
                chunk = chunk.float()
            out[lo:hi] = chunk.sum(dim=1)
        return out.reshape(self._scan_shape)

    def _masked_sum_tensor_for_frame_data(self, data, mask) -> torch.Tensor | None:
        sparse = self._sparse_masked_sum_tensor_for_frame_data(data, mask)
        if sparse is not None:
            return sparse
        if not isinstance(data, torch.Tensor):
            return None
        data_4d = (
            data
            if data.ndim == 4
            else data.reshape(
                self._scan_shape[0],
                self._scan_shape[1],
                *self._det_shape,
            )
        )
        target_device = data_4d.device
        mask_f = mask.to(target_device).float()
        rows_per_chunk = self._chunk_rows()
        out = torch.zeros(self._scan_shape, dtype=torch.float32, device=target_device)
        for i in range(0, data_4d.shape[0], rows_per_chunk):
            chunk = data_4d[i : i + rows_per_chunk]
            if not torch.is_floating_point(chunk):
                chunk = chunk.float()
            out[i : i + rows_per_chunk] = torch.tensordot(
                chunk, mask_f, dims=([2, 3], [0, 1])
            )
        return out

    def _sparse_masked_sum_for_frame_data(self, data, mask) -> np.ndarray | None:
        """Compute a frame VI by summing only selected detector pixels."""
        result = self._sparse_masked_sum_tensor_for_frame_data(data, mask)
        if result is None:
            return None
        return result.cpu().numpy().astype(np.float32, copy=False)

    def _compare_virtual_image_tensor_for_frame(
        self, idx: int, mask
    ) -> torch.Tensor | None:
        data = getattr(self, "_data", None)
        if getattr(data, "datasets", None) is not None:
            return None
        frame = self._frame_data_for_index(int(idx))
        return self._masked_sum_tensor_for_frame_data(frame, mask)

    def _compare_virtual_images_for_indices(
        self, indices: Sequence[int], mask
    ) -> list[np.ndarray]:
        mask_area = self._detector_mask_area(mask)
        tensor_images: list[torch.Tensor] = []
        allocation_failed = False
        needs_fallback = False
        for idx in indices:
            try:
                image = self._compare_virtual_image_tensor_for_frame(int(idx), mask)
            except BaseException as exc:
                if not _is_recoverable_allocation_error(exc):
                    raise
                self._reclaim_compare_allocation_failure()
                allocation_failed = True
                break
            if image is None:
                needs_fallback = True
                break
            tensor_images.append(image / mask_area)
        if tensor_images and (allocation_failed or len(tensor_images) == len(indices)):
            # Enqueue every panel reduction before copying results back. For
            # Dataset5dstem sharded across cuda:0/cuda:1, this lets both GPUs
            # work concurrently instead of synchronizing after each panel.
            return [
                np.ascontiguousarray(
                    image.detach().cpu().numpy(),
                    dtype=np.float32,
                ).reshape(self.shape_rows, self.shape_cols)
                for image in tensor_images
            ]
        if allocation_failed:
            return []
        if tensor_images and not needs_fallback:
            return []
        images: list[np.ndarray] = []
        for idx in indices:
            try:
                image = self._compare_virtual_image_for_frame(idx, mask)
            except BaseException as exc:
                if not _is_recoverable_allocation_error(exc):
                    raise
                self._reclaim_compare_allocation_failure()
                break
            images.append(
                np.ascontiguousarray(
                    image,
                    dtype=np.float32,
                ).reshape(self.shape_rows, self.shape_cols)
            )
        return [image / mask_area for image in images]

    def _virtual_image_for_frame(self, frame_idx: int) -> np.ndarray:
        """Compute virtual image for a specific 5D frame without mutating traits.

        Single chunked-torch path matching _fast_masked_sum.
        """
        data = self._frame_data_for_index(frame_idx)
        cx, cy = self.roi_center_col, self.roi_center_row
        if self.roi_mode == "circle" and self.roi_radius > 0:
            mask = self._create_circular_mask(cx, cy, self.roi_radius)
        elif self.roi_mode == "square" and self.roi_radius > 0:
            mask = self._create_square_mask(cx, cy, self.roi_radius)
        elif self.roi_mode == "annular" and self.roi_radius > 0:
            mask = self._create_annular_mask(
                cx, cy, self.roi_radius_inner, self.roi_radius
            )
        elif self.roi_mode == "rect" and self.roi_width > 0 and self.roi_height > 0:
            mask = self._create_rect_mask(
                cx, cy, self.roi_width / 2, self.roi_height / 2
            )
        else:
            row = int(max(0, min(round(cy), self._det_shape[0] - 1)))
            col = int(max(0, min(round(cx), self._det_shape[1] - 1)))
            if data.ndim == 4:
                vi = data[:, :, row, col]
            else:
                vi = data[:, row, col].reshape(self._scan_shape)
            return vi.cpu().numpy().astype(np.float32, copy=False)
        out = self._masked_sum_tensor_for_frame_data(data, mask)
        if out is None:
            raise TypeError(f"unsupported frame data type {type(data)!r}")
        return out.cpu().numpy().astype(np.float32, copy=False)

    def _chunk_rows(self) -> int:
        """Pick rows-per-chunk so float32 transient stays under _CHUNK_BYTE_BUDGET.

        Float32 cast of one chunk = rows × scan_cols × det_h × det_w × 4 bytes.
        Selected slabs (e.g. vi_roi reduce) inherit the same per-row budget.
        """
        per_row = self._scan_shape[1] * self._det_shape[0] * self._det_shape[1] * 4
        return max(1, _CHUNK_BYTE_BUDGET // max(1, per_row))

    def _fast_masked_sum(self, mask) -> np.ndarray:
        """Virtual image: sum data over scan positions weighted by a detector mask.

        Delegates to the compute backend (TorchCompute chunked tensordot on any
        torch device, MetalCompute raw kernel on chunk-backed Metal frames) so the
        widget runs identically on CUDA / MPS / CPU and on the MacBook fast path.
        Returns numpy (scan_r, scan_c) float32; the only consumer is
        `_to_float32_bytes`. Verified bit-identical to the old inline tensordot
        (tests/kernels/test_backend_parity.py + frozen widget baseline)."""
        mask_np = (
            mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
        )
        return self._compute.masked_sum(mask_np)

    def _to_float32_bytes(self, arr: torch.Tensor) -> bytes:
        """Convert tensor (any numeric dtype) to float32 bytes for JS transfer.

        Cast to float32 only at the small output. Integer reductions (uint16 sums,
        int64 accumulators) get promoted here so the multi-GB raw data never gets
        copied to float. Stats (min/max/mean/std) are computed JS-side from the
        same Float32Array — keeping them out of separate traits avoids a
        comm-message ordering race where bytes from click N arrive with stats
        from click N-1, producing a wrong colormap normalization (uniform white
        flash on rapid preset switching).
        """
        if isinstance(arr, np.ndarray):
            return np.ascontiguousarray(arr, dtype=np.float32).tobytes()
        if arr.dtype != torch.float32:
            arr = arr.float()
        return arr.cpu().numpy().tobytes()

    def _compute_virtual_image_from_roi(self):
        """Compute virtual image based on ROI mode."""
        if self._data is None:
            return
        cached = self._get_cached_preset()
        if cached is not None:
            self.virtual_image_bytes = cached
            return

        cx, cy = self.roi_center_col, self.roi_center_row

        if self.roi_mode == "circle" and self.roi_radius > 0:
            mask = self._create_circular_mask(cx, cy, self.roi_radius)
        elif self.roi_mode == "square" and self.roi_radius > 0:
            mask = self._create_square_mask(cx, cy, self.roi_radius)
        elif self.roi_mode == "annular" and self.roi_radius > 0:
            mask = self._create_annular_mask(
                cx, cy, self.roi_radius_inner, self.roi_radius
            )
        elif self.roi_mode == "rect" and self.roi_width > 0 and self.roi_height > 0:
            mask = self._create_rect_mask(
                cx, cy, self.roi_width / 2, self.roi_height / 2
            )
        else:
            # Point mode: single detector pixel via a one-hot mask through the same
            # backend (sum of data[:, :, row, col] * one-hot == that pixel). One path
            # for every backend - no tensor-only fancy-indexing that breaks on Metal.
            row = int(max(0, min(round(cy), self._det_shape[0] - 1)))
            col = int(max(0, min(round(cx), self._det_shape[1] - 1)))
            point_mask = np.zeros(self._det_shape, dtype=np.float32)
            point_mask[row, col] = 1.0
            self.virtual_image_bytes = self._to_float32_bytes(
                self._fast_masked_sum(point_mask)
            )
            return

        self.virtual_image_bytes = self._to_float32_bytes(self._fast_masked_sum(mask))
