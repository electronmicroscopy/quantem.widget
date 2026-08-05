"""
show4dstem: Fast interactive 4D-STEM viewer widget.

The public factory routes Apple chunk-backed loads to the MPS Metal viewer. This
base viewer keeps CUDA CuPy inputs resident so ``quantem.gpu`` can use its
RawKernel virtual-image reducers. NumPy inputs remain useful for small UI tests.

To reduce data size, bin k-space at the dataset level before viewing:

    dataset = dataset.bin(2, axes=(2, 3))  # 2x2 k-space binning
    widget = Show4DSTEM(dataset)
"""

import base64
import gc
import hashlib
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
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Self, Sequence

if TYPE_CHECKING:
    from quantem.core.datastructures import Dataset4dstem

import anywidget
import numpy as np
import torch
import traitlets
from quantem.widget._folder_watch_status import (
    FOLDER_WATCH_STATE_VALUES,
    set_folder_watch_status,
)
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


def _local_h5_url_path(url: str, base_dir: pathlib.Path) -> pathlib.Path | None:
    """Resolve a local HDF5 URL used by browser-source exports."""
    if not url or "://" in url:
        return None
    clean = url.split("?", 1)[0].split("#", 1)[0]
    path = pathlib.Path(clean)
    if not path.is_absolute():
        path = base_dir / path
    return path


def _h5_bad_pixel_json_for_export(url: str, base_dir: pathlib.Path) -> str | None:
    """Return a JSON bad-pixel list for a local HDF5 master, or None if unreadable."""
    path = _local_h5_url_path(url, base_dir)
    if path is None or not path.exists():
        return None
    try:
        from quantem.gpu.io import inspect

        mask = inspect(path).pixel_mask
    except Exception:
        return None
    if mask is None:
        return None
    bad = np.flatnonzero(np.asarray(mask).reshape(-1) > 0).astype(int).tolist()
    return json.dumps(bad)


def _show4dstem_h5_webgpu_tuning(*, dtype: str, h5_uint8_lossless: bool) -> str:
    """Return runtime settings for browser-owned HDF5 WebGPU exports."""
    decode_dtype = "uint8" if dtype == "uint8" else "u2"
    low8 = "true" if h5_uint8_lossless else "false"
    return (
        "<script>\n"
        "if (globalThis.location?.protocol === \"file:\") {\n"
        "  globalThis.__QT_REQUIRE_LOCAL_H5_FILES = true;\n"
        "}\n"
        f'globalThis.__QT_H5_DECODE_DTYPE ??= "{decode_dtype}";\n'
        f"globalThis.__QT_H5_FORCE_LOW8 ??= {low8};\n"
        f"globalThis.__BSLZ4_LOW8_ONLY ??= {low8};\n"
        "globalThis.__BSLZ4_FRAME_WG ??= 64;\n"
        "globalThis.__BSLZ4_PIPELINE_STAGING ??= false;\n"
        "globalThis.__QT_H5_FETCH_WINDOW ??= 8;\n"
        "globalThis.__QT_H5_DECODE_QUEUE ??= 8;\n"
        "globalThis.__QT_H5_PRELOAD_WINDOW ??= 1;\n"
        "globalThis.__QT_H5_LOCAL_GROUP ??= 8;\n"
        "globalThis.__QT_H5_LOCAL_WORKERS ??= 8;\n"
        "</script>\n"
    )


def _inject_show4dstem_h5_webgpu_tuning(
    path: pathlib.Path,
    *,
    dtype: str,
    h5_uint8_lossless: bool,
) -> None:
    """Inject HDF5 WebGPU runtime settings into an exported HTML page."""
    text = path.read_text(encoding="utf-8")
    has_runtime_tuning = (
        "__QT_H5_DECODE_DTYPE" in text
        and "__BSLZ4_PIPELINE_STAGING" in text
    )
    has_file_guard = "__QT_REQUIRE_LOCAL_H5_FILES = true" in text
    if has_runtime_tuning and has_file_guard:
        return
    script = _show4dstem_h5_webgpu_tuning(
        dtype=dtype,
        h5_uint8_lossless=h5_uint8_lossless,
    )
    if "<head>" in text:
        text = text.replace("<head>", "<head>\n" + script, 1)
    else:
        text = script + text
    path.write_text(text, encoding="utf-8")


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
    DPC_row, DPC_col, SSB : array_like, optional
        Precomputed real-space maps to expose in the image panel alongside the
        ROI-derived virtual image. Each map must be real-valued with shape
        ``(scan_rows, scan_cols)`` or ``(n_frames, scan_rows, scan_cols)``.
        ``SSB`` is the phase map; complex SSB inputs are rejected so amplitude
        is not selected silently.
    vi_source : {"roi", "DPC_row", "DPC_col", "SSB"}, optional
        Initial image-panel source. Defaults to the ROI-derived virtual image.
    frame_dim_label : str, optional
        Label for the frame dimension when 5D data is provided.
        Defaults to "Frame". Common values: "Tilt", "Time", "Focus".
    view_mode : {"single", "multiple"}, default "single"
        Scientific layout mode. ``"single"`` shows one selected frame/dataset.
        ``"multiple"`` shows a grid of virtual images for the first ready
        frames/datasets while sharing the detector ROI and scan cursor with the
        existing diffraction panel.
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

    PyTorch CUDA tensor:

    >>> import torch
    >>> Show4DSTEM(torch.rand(64, 64, 128, 128, device="cuda"))

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
    folder_watch_state = traitlets.Enum(
        values=FOLDER_WATCH_STATE_VALUES,
        default_value="hidden",
    ).tag(sync=True)
    folder_watch_detail = traitlets.Unicode("").tag(sync=True)
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
    vi_source = traitlets.Unicode("roi").tag(sync=True)
    vi_product_labels = traitlets.List(
        traitlets.Unicode(), default_value=[]
    ).tag(sync=True)
    vi_product_map_frames = traitlets.Int(0).tag(sync=True)
    vi_product_maps_bytes = traitlets.Bytes(b"").tag(sync=True)
    vi_preset_labels = traitlets.List(
        traitlets.Unicode(), default_value=[]
    ).tag(sync=True)
    vi_preset_map_frames = traitlets.Int(0).tag(sync=True)
    vi_preset_maps_bytes = traitlets.Bytes(b"").tag(sync=True)

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
    # Browser-source mode: normal CLI exports point at sibling HDF5 files for
    # on-demand byte-range frame reads.
    _h5_url = traitlets.Unicode("").tag(sync=True)
    _h5_urls = traitlets.Unicode("").tag(sync=True)
    _h5_uint8_lossless = traitlets.Bool(False).tag(sync=True)
    # Lazy mode: a sidecar bundle URL (radial profile + CoM + frame index + data files). The JS
    # derives the virtual image from the ~100 MB profile in VRAM and lazy-fetches CBED frames from
    # disk - nothing bulk-loads. Real-time scrub + detector with no 38 GB resident.
    _lazy_url = traitlets.Unicode("").tag(sync=True)
    _lazy_urls = traitlets.Unicode("").tag(sync=True)
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

    # Kernel-backed SSB compute. Exported/static HTML can display precomputed
    # SSB maps, but cannot launch this Python/CUDA path.
    ssb_compute_request = traitlets.Unicode("").tag(sync=True)
    ssb_compute_status = traitlets.Unicode("").tag(sync=True)
    ssb_compute_busy = traitlets.Bool(False).tag(sync=True)
    ssb_compute_enabled = traitlets.Bool(True).tag(sync=True)
    ssb_compute_n_trials = traitlets.Int(200).tag(sync=True)
    ssb_compute_refine = traitlets.Bool(True).tag(sync=True)
    ssb_compute_bf_subsample = traitlets.Float(1.0).tag(sync=True)
    ssb_compute_bf_pixels = traitlets.Int(0).tag(sync=True)
    ssb_compute_bf_selected_pixels = traitlets.Int(0).tag(sync=True)
    ssb_compute_manual_aberrations = traitlets.Bool(False).tag(sync=True)
    ssb_compute_lock_c10 = traitlets.Bool(False).tag(sync=True)
    ssb_compute_lock_c12 = traitlets.Bool(False).tag(sync=True)
    ssb_compute_c10_nm = traitlets.Float(0.0).tag(sync=True)
    ssb_compute_c12_nm = traitlets.Float(0.0).tag(sync=True)
    ssb_compute_phi12_deg = traitlets.Float(0.0).tag(sync=True)
    ssb_compute_rotation_angle_deg = traitlets.Float(0.0).tag(sync=True)
    ssb_compute_calibration_json = traitlets.Unicode("").tag(sync=True)
    ssb_compute_calibration_filename = traitlets.Unicode("").tag(sync=True)

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

    dp_scale_mode = traitlets.Unicode("log").tag(sync=True)  # "linear" | "log"
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
    # Progressive folder-page telemetry. The full stacked payload above remains
    # the durable widget state; these small traits let the live frontend reserve
    # stable slots and report useful cold-page latency while panels stream in.
    compare_page_progressive_enabled = traitlets.Bool(False).tag(sync=True)
    compare_page_expected_indices = traitlets.List(
        traitlets.Int(), default_value=[]
    ).tag(sync=True)
    compare_page_loading = traitlets.Bool(False).tag(sync=True)
    compare_page_loaded_count = traitlets.Int(0).tag(sync=True)
    compare_page_cached_indices = traitlets.List(
        traitlets.Int(), default_value=[]
    ).tag(sync=True)
    compare_page_cache_state = traitlets.Unicode("off").tag(sync=True)
    compare_page_generation = traitlets.Int(0).tag(sync=True)
    compare_page_first_panel_ms = traitlets.Float(0.0).tag(sync=True)
    compare_page_first_fresh_ms = traitlets.Float(0.0).tag(sync=True)
    compare_page_total_ms = traitlets.Float(0.0).tag(sync=True)
    # Reliable trait-synced copy of the latest progressive tile. Custom binary
    # comm messages remain the fast path, while this sequence-backed payload
    # covers notebook frontends that drop custom buffers from daemon threads.
    compare_page_panel_bytes = traitlets.Bytes(b"").tag(sync=True)
    compare_page_panel_frame_idx = traitlets.Int(-1).tag(sync=True)
    compare_page_panel_slot = traitlets.Int(-1).tag(sync=True)
    compare_page_panel_cached = traitlets.Bool(False).tag(sync=True)
    compare_page_panel_sequence = traitlets.Int(0).tag(sync=True)
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

    def _multiple_view_active(self) -> bool:
        """Return True when the multiple-panel virtual-image surface is visible."""
        return self.view_mode == "multiple" and self.n_frames > 1

    @staticmethod
    def _normalise_vi_source(value: str | None) -> str:
        source = str(value or "roi").strip()
        key = source.lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "": "roi",
            "roi": "roi",
            "virtual": "roi",
            "virtual_image": "roi",
            "bf": "roi",
            "dpc_row": "DPC_row",
            "dpc_com_row": "DPC_row",
            "dpc_r": "DPC_row",
            "dpcr": "DPC_row",
            "dpc_col": "DPC_col",
            "dpc_com_col": "DPC_col",
            "dpc_c": "DPC_col",
            "dpcc": "DPC_col",
            "ssb": "SSB",
            "ssb_phase": "SSB",
            "phase": "SSB",
        }
        return aliases.get(key, source)

    def _normalise_vi_product_array(self, label: str, value: Any) -> np.ndarray:
        arr = np.asarray(to_numpy(value))
        if np.iscomplexobj(arr):
            raise ValueError(
                f"{label} must be a real-valued map. Pass the SSB phase map "
                "rather than a complex reconstruction or amplitude stack."
            )
        if arr.ndim == 2:
            if arr.shape != (self.shape_rows, self.shape_cols):
                raise ValueError(
                    f"{label} shape {arr.shape} does not match the scan shape "
                    f"{self.shape_rows}x{self.shape_cols}."
                )
            arr = arr.reshape(1, self.shape_rows, self.shape_cols)
        elif arr.ndim == 3:
            if arr.shape[0] != self.n_frames:
                raise ValueError(
                    f"{label} has {arr.shape[0]} frames, but Show4DSTEM has "
                    f"{self.n_frames} frame(s)."
                )
            if arr.shape[1:] != (self.shape_rows, self.shape_cols):
                raise ValueError(
                    f"{label} frame shape {arr.shape[1:]} does not match the "
                    f"scan shape {self.shape_rows}x{self.shape_cols}."
                )
        else:
            raise ValueError(
                f"{label} must have shape (scan_rows, scan_cols) or "
                f"(n_frames, scan_rows, scan_cols); got {arr.shape}."
            )
        return np.ascontiguousarray(arr, dtype=np.float32)

    def _set_vi_product_maps(self, maps: dict[str, Any]) -> None:
        products: dict[str, np.ndarray] = {}
        for label in ("DPC_row", "DPC_col", "SSB"):
            value = maps.get(label)
            if value is None:
                continue
            products[label] = self._normalise_vi_product_array(label, value)

        self._vi_product_maps = products
        labels = list(products)
        self.vi_product_labels = labels
        if not products:
            self.vi_product_map_frames = 0
            self.vi_product_maps_bytes = b""
            return

        frame_count = max(arr.shape[0] for arr in products.values())
        stacked = []
        for label in labels:
            arr = products[label]
            if arr.shape[0] == 1 and frame_count > 1:
                arr = np.broadcast_to(arr, (frame_count, self.shape_rows, self.shape_cols))
            elif arr.shape[0] != frame_count:
                raise ValueError(
                    f"{label} has {arr.shape[0]} product frame(s), but another "
                    f"product map has {frame_count}."
                )
            stacked.append(np.ascontiguousarray(arr, dtype=np.float32))
        self.vi_product_map_frames = int(frame_count)
        self.vi_product_maps_bytes = np.ascontiguousarray(
            np.stack(stacked, axis=0), dtype=np.float32
        ).tobytes()

    def _set_vi_preset_maps(self, maps: dict[str, Any]) -> None:
        presets: dict[str, np.ndarray] = {}
        for label in ("BF", "ABF", "ADF", "HAADF"):
            value = maps.get(label)
            if value is None:
                continue
            presets[label] = self._normalise_vi_product_array(label, value)

        self._vi_preset_maps = presets
        labels = list(presets)
        self.vi_preset_labels = labels
        if not presets:
            self.vi_preset_map_frames = 0
            self.vi_preset_maps_bytes = b""
            return

        frame_count = max(arr.shape[0] for arr in presets.values())
        stacked = []
        for label in labels:
            arr = presets[label]
            if arr.shape[0] == 1 and frame_count > 1:
                arr = np.broadcast_to(arr, (frame_count, self.shape_rows, self.shape_cols))
            elif arr.shape[0] != frame_count:
                raise ValueError(
                    f"{label} has {arr.shape[0]} preset frame(s), but another "
                    f"preset map has {frame_count}."
                )
            stacked.append(np.ascontiguousarray(arr, dtype=np.float32))
        self.vi_preset_map_frames = int(frame_count)
        self.vi_preset_maps_bytes = np.ascontiguousarray(
            np.stack(stacked, axis=0), dtype=np.float32
        ).tobytes()

    def _current_vi_product_map(self, source: str | None = None) -> np.ndarray | None:
        label = self._normalise_vi_source(source or self.vi_source)
        arr = getattr(self, "_vi_product_maps", {}).get(label)
        if arr is None:
            return None
        frame_count = int(arr.shape[0])
        frame = 0 if frame_count == 1 else max(0, min(int(self.frame_idx), frame_count - 1))
        return np.ascontiguousarray(arr[frame], dtype=np.float32)

    def set_vi_product_map(self, label: str, value: Any) -> Self:
        """Attach or replace a static virtual-image product map."""
        label = self._normalise_vi_source(label)
        if label not in {"DPC_row", "DPC_col", "SSB"}:
            raise ValueError(
                f"Unsupported virtual-image product {label!r}. "
                "Use one of 'DPC_row', 'DPC_col', or 'SSB'."
            )
        maps = dict(getattr(self, "_vi_product_maps", {}))
        maps[label] = value
        self._set_vi_product_maps(maps)
        if self._normalise_vi_source(self.vi_source) == label:
            self._refresh_compare_virtual_images()
        return self

    def set_vi_preset_map(self, label: str, value: Any) -> Self:
        """Attach or replace a static BF/ABF/ADF/HAADF preset map."""
        raw = str(label or "").strip().upper().replace("-", "_").replace(" ", "_")
        aliases = {
            "BRIGHT_FIELD": "BF",
            "BRIGHTFIELD": "BF",
            "DARK_FIELD": "ADF",
            "DARKFIELD": "ADF",
        }
        label = aliases.get(raw, raw)
        if label not in {"BF", "ABF", "ADF", "HAADF"}:
            raise ValueError(
                f"Unsupported virtual-image preset {label!r}. "
                "Use one of 'BF', 'ABF', 'ADF', or 'HAADF'."
            )
        maps = dict(getattr(self, "_vi_preset_maps", {}))
        maps[label] = value
        self._set_vi_preset_maps(maps)
        return self

    @staticmethod
    def _ssb_positive_pair(value: Any, *, name: str) -> tuple[float, float]:
        if isinstance(value, (int, float)) or np.isscalar(value):
            pair = (float(value), float(value))
        else:
            pair = tuple(float(v) for v in value)
            if len(pair) != 2:
                raise ValueError(f"{name} must be a scalar or length-2 pair.")
        if pair[0] <= 0 or pair[1] <= 0:
            raise ValueError(f"{name} must be positive, got {pair}.")
        return pair

    @staticmethod
    def _ssb_unit_key(unit: str | None) -> str:
        key = str(unit or "").strip().lower()
        key = key.replace("µ", "u").replace("μ", "u").replace("å", "a")
        key = key.replace("angstroms", "angstrom").replace("angstrom", "a")
        key = key.replace(" per pixel", "").replace("/pixel", "").replace("/px", "")
        key = key.replace(" ", "").replace("_", "")
        return key

    @classmethod
    def _to_angstrom(cls, value: float, unit: str, *, axis: str) -> float:
        key = cls._ssb_unit_key(unit)
        factors = {
            "a": 1.0,
            "ang": 1.0,
            "nm": 10.0,
            "pm": 0.01,
        }
        if key in factors:
            out = float(value) * factors[key]
            if out <= 0:
                raise ValueError(f"{axis} sampling must be positive, got {out}.")
            return out
        raise ValueError(
            "Compute SSB needs real-space scan sampling in Angstroms. "
            "Pass ssb_scan_sampling_A=(row_A, col_A), or construct "
            "Show4DSTEM with sampling=(scan_row, scan_col, ..., ...) and "
            "units=('A', 'A', ..., ...)."
        )

    @classmethod
    def _to_mrad(cls, value: float, unit: str) -> float:
        key = cls._ssb_unit_key(unit)
        factors = {
            "mrad": 1.0,
            "rad": 1000.0,
            "urad": 0.001,
        }
        if key not in factors:
            raise ValueError("Detector sampling is not an angular unit.")
        out = float(value) * factors[key]
        if out <= 0:
            raise ValueError(f"Detector sampling must be positive, got {out}.")
        return out

    def _resolved_ssb_scan_sampling_A(self, value: Any | None) -> tuple[float, float]:
        if value is not None:
            return self._ssb_positive_pair(value, name="ssb_scan_sampling_A")
        sampling = getattr(self, "_axis_sampling", (self.pixel_size, self.pixel_size))
        units = getattr(self, "_axis_units", [self.pixel_unit, self.pixel_unit])
        row = self._to_angstrom(
            float(sampling[0]), units[0] if len(units) > 0 else self.pixel_unit, axis="scan row"
        )
        col = self._to_angstrom(
            float(sampling[1] if len(sampling) > 1 else sampling[0]),
            units[1] if len(units) > 1 else self.pixel_unit,
            axis="scan col",
        )
        return (row, col)

    def _resolved_ssb_det_sampling_mrad(
        self, value: Any | None
    ) -> tuple[float, float] | None:
        if value is not None:
            return self._ssb_positive_pair(value, name="ssb_det_sampling_mrad")
        sampling = getattr(self, "_axis_sampling", ())
        units = getattr(self, "_axis_units", [])
        if len(sampling) < 4 or len(units) < 4:
            return None
        try:
            row = self._to_mrad(float(sampling[2]), units[2])
            col = self._to_mrad(float(sampling[3]), units[3])
        except ValueError:
            return None
        return (row, col)

    def _resolved_ssb_semiangle_mrad(
        self,
        value: float | None,
        det_sampling: tuple[float, float] | None,
    ) -> float:
        if value is not None:
            semiangle = float(value)
        elif det_sampling is not None and float(self.bf_radius) > 0:
            semiangle = float(self.bf_radius) * float(det_sampling[1])
        else:
            raise ValueError(
                "Compute SSB needs ssb_semiangle_mrad, or calibrated detector "
                "sampling in mrad/pixel plus a detected BF radius."
            )
        if semiangle <= 0:
            raise ValueError(f"ssb_semiangle_mrad must be positive, got {semiangle}.")
        return semiangle

    @staticmethod
    def _ssb_selected_bf_count(full_num_bf: int, ratio: float | None) -> int:
        """Return the BF subset count used by the quantem.gpu stride sampler."""
        full = max(0, int(full_num_bf))
        if full == 0 or ratio is None or float(ratio) >= 1.0:
            return full
        ratio_f = float(ratio)
        if ratio_f <= 0.0:
            raise ValueError(f"bf_subsample must be in (0, 1], got {ratio}.")
        stride = max(1, int(round(1.0 / ratio_f)))
        return int((full + stride - 1) // stride)

    def _ssb_cupy_frame(self):
        """Return the current 4D frame as a CuPy array plus an owner reference."""
        try:
            import cupy as cp
        except ImportError as exc:
            raise ImportError(
                "Compute SSB requires CuPy and the CUDA quantem.gpu SSB engine."
            ) from exc

        frame = self._frame_data
        if hasattr(frame, "chunks") or getattr(frame, "_is_gpu_frames", False):
            raise ValueError(
                "Compute SSB from Show4DSTEM currently requires a resident CUDA "
                "or CuPy 4D frame. For MPS/chunked data, "
                "precompute SSB separately and pass SSB=phase_map."
            )
        if isinstance(frame, torch.Tensor):
            if frame.device.type == "cuda":
                owner = frame.contiguous()
                device_index = owner.device.index
                if device_index is None:
                    device_index = torch.cuda.current_device()
                with cp.cuda.Device(int(device_index)):
                    return cp.from_dlpack(owner), owner
            raise ValueError(
                f"Compute SSB requires CUDA/CuPy; got Torch device {frame.device}."
            )

        if type(frame).__module__.split(".", 1)[0] == "cupy":
            return cp.asarray(frame), frame

        raise ValueError(
            "Compute SSB requires a CUDA tensor or CuPy array; CPU transfer is "
            "not a scientific fallback."
        )

    def _compute_ssb_phase(
        self,
        *,
        semiangle_mrad: float | None = None,
        scan_sampling_A: float | tuple[float, float] | None = None,
        det_sampling_mrad: float | tuple[float, float] | None = None,
        voltage_kV: float | None = None,
        energy_eV: float | None = None,
        bf_radius: int | None = None,
        aberrations: dict[str, float] | None = None,
        rotation_angle_deg: float | None = None,
        bf_intensity_threshold: float | None = None,
        n_trials: int | None = None,
        refine: bool | None = None,
        lock_aberrations: bool = False,
        lock_c10: bool = False,
        lock_c12: bool = False,
        seed: int | None = None,
        bf_subsample: float | None = None,
        verbose: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the CUDA SSB solver for the current 4D frame.

        Returns ``(phase, dpc_row, dpc_col)``. The DPC center-of-mass maps come
        for free: the 4D frame is already resident on the GPU for SSB, so CoM
        is two cheap reductions, aligned with the solved scan rotation.
        """
        if self.n_frames != 1:
            raise ValueError(
                "Compute SSB currently runs on a single 4D Show4DSTEM frame. "
                "For a 5D stack, open the target frame as a 4D widget or pass "
                "precomputed SSB=(n_frames, scan_rows, scan_cols)."
            )
        if (self.shape_rows, self.shape_cols) not in ((128, 128), (256, 256), (512, 512)):
            raise ValueError(
                "Compute SSB currently supports square 128x128, 256x256, or "
                f"512x512 scan grids; got {self.shape_rows}x{self.shape_cols}."
            )

        cfg = dict(getattr(self, "_ssb_compute_config", {}))
        scan_sampling = self._resolved_ssb_scan_sampling_A(
            scan_sampling_A if scan_sampling_A is not None else cfg.get("scan_sampling_A")
        )
        det_sampling = self._resolved_ssb_det_sampling_mrad(
            det_sampling_mrad if det_sampling_mrad is not None else cfg.get("det_sampling_mrad")
        )
        semiangle = self._resolved_ssb_semiangle_mrad(
            semiangle_mrad if semiangle_mrad is not None else cfg.get("semiangle_mrad"),
            det_sampling,
        )
        voltage = voltage_kV if voltage_kV is not None else cfg.get("voltage_kV")
        energy = energy_eV if energy_eV is not None else cfg.get("energy_eV")
        if voltage is None and energy is None:
            raise ValueError(
                "Compute SSB needs ssb_voltage_kV or ssb_energy_eV. "
                "Example: Show4DSTEM(data, ssb_voltage_kV=300, ...)."
            )
        trials = cfg.get("n_trials", 200) if n_trials is None else n_trials
        trials = int(trials)
        if trials < 0:
            raise ValueError(f"n_trials must be >= 0, got {trials}.")
        do_refine = bool(cfg.get("refine", True) if refine is None else refine)
        seed = int(cfg.get("seed", 42) if seed is None else seed)
        bf_subsample = cfg.get("bf_subsample") if bf_subsample is None else bf_subsample
        if bf_subsample is not None:
            bf_subsample = float(bf_subsample)
            if bf_subsample <= 0.0:
                raise ValueError(f"bf_subsample must be in (0, 1], got {bf_subsample}.")
        bf_threshold = (
            cfg.get("bf_intensity_threshold", 0.5)
            if bf_intensity_threshold is None
            else bf_intensity_threshold
        )
        bf_radius = cfg.get("bf_radius") if bf_radius is None else bf_radius
        aberrations = cfg.get("aberrations") if aberrations is None else aberrations
        rotation = (
            cfg.get("rotation_angle_deg", 0.0)
            if rotation_angle_deg is None
            else rotation_angle_deg
        )
        optimize_aberrations = None
        if aberrations is not None:
            c10 = float(aberrations.get("C10", aberrations.get("C10_nm", 0.0)))
            c12 = float(aberrations.get("C12", aberrations.get("C12_nm", 0.0)))
            if "phi12_deg" in aberrations:
                phi12_deg = float(aberrations["phi12_deg"])
                phi12 = math.radians(phi12_deg)
            else:
                phi12 = float(aberrations.get("phi12", 0.0))
                phi12_deg = math.degrees(phi12)
            aberrations = {"C10": c10, "C12": c12, "phi12": phi12}
            if lock_aberrations:
                optimize_aberrations = {
                    "C10_nm": c10,
                    "C12_nm": c12,
                    "phi12_deg": phi12_deg,
                }
        if optimize_aberrations is None and (lock_c10 or lock_c12):
            base = aberrations or {}
            locked_c10 = float(base.get("C10", self.ssb_compute_c10_nm))
            locked_c12 = float(base.get("C12", self.ssb_compute_c12_nm))
            locked_phi12_deg = (
                math.degrees(float(base["phi12"]))
                if "phi12" in base
                else float(self.ssb_compute_phi12_deg)
            )
            # Scalar pins the coefficient; tuple keeps the production search
            # range from SSB.optimize defaults. Locking C12 pins phi12 too:
            # astigmatism is a magnitude+angle pair.
            optimize_aberrations = {
                "C10_nm": locked_c10 if lock_c10 else (-400.0, 400.0),
                "C12_nm": locked_c12 if lock_c12 else (0.0, 100.0),
                "phi12_deg": locked_phi12_deg if lock_c12 else (-90.0, 90.0),
            }

        try:
            from quantem.gpu.ssb import SSB as QuantemSSB
        except ImportError as exc:
            raise ImportError(
                "Compute SSB requires quantem.gpu with the CUDA SSB engine installed."
            ) from exc

        self.ssb_compute_status = "Preparing SSB data..."
        data_gpu, owner = self._ssb_cupy_frame()
        try:
            self.ssb_compute_status = "Building SSB engine..."
            ssb = QuantemSSB(
                data_gpu,
                semiangle=float(semiangle),
                scan_sampling=scan_sampling,
                det_sampling=det_sampling,
                voltage_kV=None if voltage is None else float(voltage),
                energy=None if energy is None else float(energy),
                scan_shape=(self.shape_rows, self.shape_cols) if data_gpu.ndim == 3 else None,
                bf_intensity_threshold=float(bf_threshold),
                bf_radius=None if bf_radius is None else int(bf_radius),
                aberrations=aberrations,
                rotation_angle_deg=float(rotation),
            )
            full_bf = int(len(ssb.bf_inds_row))
            selected_bf = self._ssb_selected_bf_count(full_bf, bf_subsample)
            self.ssb_compute_bf_pixels = full_bf
            self.ssb_compute_bf_selected_pixels = selected_bf
            if lock_aberrations and aberrations is not None:
                self.ssb_compute_status = (
                    f"Using manual SSB coefficients "
                    f"({selected_bf}/{full_bf} BF pixels)..."
                )
            elif trials > 0:
                self.ssb_compute_status = (
                    f"Optimizing SSB ({trials} trials, "
                    f"{selected_bf}/{full_bf} BF pixels)..."
                )
                ssb.optimize(
                    aberrations=optimize_aberrations,
                    rotation_angle_deg=float(rotation),
                    n_trials=trials,
                    seed=seed,
                    verbose=verbose,
                    bf_subsample=bf_subsample,
                )
            # The GPU Nelder-Mead refiner has no locked mode yet; with a pinned
            # coefficient the Optuna search already respects the lock, so skip
            # refine rather than let it move the pinned value.
            if do_refine and not (lock_aberrations and aberrations is not None) and not (lock_c10 or lock_c12):
                self.ssb_compute_status = (
                    f"Refining SSB ({selected_bf}/{full_bf} BF pixels)..."
                )
                ssb.refine(verbose=verbose, bf_subsample=bf_subsample)
            self.ssb_compute_status = "Reconstructing SSB phase..."
            result = ssb.result()
            phase = result.phase
            result_aberrations = dict(
                getattr(result, "aberrations", None)
                or getattr(ssb, "aberrations", {})
                or {}
            )
            c10_nm = float(result_aberrations.get("C10", self.ssb_compute_c10_nm))
            c12_nm = float(result_aberrations.get("C12", self.ssb_compute_c12_nm))
            phi12_rad = float(
                result_aberrations.get(
                    "phi12",
                    math.radians(float(self.ssb_compute_phi12_deg)),
                )
            )
            rotation_final = float(
                getattr(result, "rotation_angle_deg", float(rotation))
            )
            self.ssb_compute_c10_nm = c10_nm
            self.ssb_compute_c12_nm = c12_nm
            self.ssb_compute_phi12_deg = math.degrees(phi12_rad)
            self.ssb_compute_rotation_angle_deg = rotation_final
            self.ssb_compute_calibration_json = json.dumps(
                {
                    "schema": "quantem.ssb.calibration.v1",
                    "created_utc": datetime.now(timezone.utc)
                    .replace(microsecond=0)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "source": {
                        "widget": "Show4DSTEM",
                        "title": self.title,
                        "frame_idx": int(self.frame_idx),
                        "scan_shape": [int(self.shape_rows), int(self.shape_cols)],
                        "detector_shape": [int(self.det_rows), int(self.det_cols)],
                    },
                    "aberrations": {
                        "C10": c10_nm,
                        "C12": c12_nm,
                        "phi12": phi12_rad,
                    },
                    "aberration_units": {
                        "C10": "nm",
                        "C12": "nm",
                        "phi12": "rad",
                    },
                    "rotation_angle_deg": rotation_final,
                    "calibration": {
                        "voltage_kV": None if voltage is None else float(voltage),
                        "energy_eV": None if energy is None else float(energy),
                        "semiangle_mrad": float(semiangle),
                        "scan_sampling_A": [float(x) for x in scan_sampling],
                        "det_sampling_mrad": [float(x) for x in det_sampling],
                        "bf_radius": None if bf_radius is None else int(bf_radius),
                        "bf_intensity_threshold": float(bf_threshold),
                        "bf_subsample": None if bf_subsample is None else float(bf_subsample),
                        "bf_pixels": full_bf,
                        "bf_selected_pixels": selected_bf,
                    },
                    "run": {
                        "n_trials": trials,
                        "refine": do_refine,
                        "manual_locked": bool(lock_aberrations and aberrations is not None),
                        "seed": seed,
                        "loss": (
                            None
                            if getattr(result, "loss", None) is None
                            else float(getattr(result, "loss"))
                        ),
                        "elapsed_s": (
                            None
                            if getattr(result, "elapsed", None) is None
                            else float(getattr(result, "elapsed"))
                        ),
                        "refine_method": getattr(result, "refine_method", None),
                        "refine_nfev": getattr(result, "refine_nfev", None),
                        "refine_elapsed_s": (
                            None
                            if getattr(result, "refine_elapsed", None) is None
                            else float(getattr(result, "refine_elapsed"))
                        ),
                    },
                },
                indent=2,
            )
            self.ssb_compute_calibration_filename = self._default_ssb_calibration_filename()
            if hasattr(phase, "get"):
                phase = phase.get()
            phase_np = np.asarray(phase, dtype=np.float32)
            # DPC comes for free here: the 4D frame is already resident on the
            # GPU, so CoM row/col is two cheap reductions. Align with the SSB
            # scan rotation so the maps follow the shared aligned-DPC convention.
            from quantem.gpu.dpc import center_of_mass
            com_k_row, com_k_col = center_of_mass(
                data_gpu, scan_shape=(self.shape_rows, self.shape_cols)
            )
            theta = math.radians(rotation_final)
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            dpc_row_np = np.ascontiguousarray(
                cos_t * com_k_row - sin_t * com_k_col, dtype=np.float32
            )
            dpc_col_np = np.ascontiguousarray(
                sin_t * com_k_row + cos_t * com_k_col, dtype=np.float32
            )
        finally:
            del owner
            gc.collect()

        if phase_np.shape != (self.shape_rows, self.shape_cols):
            raise ValueError(
                f"SSB result shape {phase_np.shape} does not match the widget "
                f"scan shape {self.shape_rows}x{self.shape_cols}."
            )
        return np.ascontiguousarray(phase_np, dtype=np.float32), dpc_row_np, dpc_col_np

    def compute_ssb(self, *, set_source: bool = True, verbose: bool = False, **kwargs) -> np.ndarray:
        """Compute SSB phase in the live kernel and attach it as the SSB map."""
        if self.ssb_compute_busy:
            raise RuntimeError("Compute SSB is already running.")
        start = time.perf_counter()
        self.ssb_compute_busy = True
        self.ssb_compute_status = "Computing SSB..."
        self.ssb_compute_calibration_json = ""
        self.ssb_compute_calibration_filename = ""
        try:
            phase, dpc_row, dpc_col = self._compute_ssb_phase(verbose=verbose, **kwargs)
            self.set_vi_product_map("DPC_row", dpc_row)
            self.set_vi_product_map("DPC_col", dpc_col)
            self.set_vi_product_map("SSB", phase)
            if set_source:
                self.vi_source = "SSB"
            elapsed = time.perf_counter() - start
            self.ssb_compute_status = (
                f"SSB ready ({phase.shape[0]}x{phase.shape[1]}, {elapsed:.1f}s)"
            )
            return phase
        except Exception as exc:
            self.ssb_compute_status = f"SSB failed: {exc}"
            raise
        finally:
            self.ssb_compute_busy = False

    def __init__(
        self,
        data: "Dataset4dstem | np.ndarray",
        scan_shape: tuple[int, int] | None = None,
        sampling: tuple[float, ...] | list[float] | None = None,
        units: list[str] | tuple[str, ...] | None = None,
        center: tuple[float, float] | None = None,
        bf_radius: float | None = None,
        precompute_virtual_images: bool = True,
        DPC_row: Any | None = None,
        DPC_col: Any | None = None,
        SSB: Any | None = None,
        vi_source: str | None = None,
        ssb_semiangle_mrad: float | None = None,
        ssb_scan_sampling_A: float | tuple[float, float] | None = None,
        ssb_det_sampling_mrad: float | tuple[float, float] | None = None,
        ssb_voltage_kV: float | None = None,
        ssb_energy_eV: float | None = None,
        ssb_bf_radius: int | None = None,
        ssb_bf_intensity_threshold: float = 0.5,
        ssb_aberrations: dict[str, float] | None = None,
        ssb_rotation_angle_deg: float = 0.0,
        ssb_n_trials: int = 200,
        ssb_refine: bool = True,
        ssb_seed: int = 42,
        ssb_bf_subsample: float | None = 1.0,
        ssb_manual_aberrations: bool = False,
        ssb_c10_nm: float = 0.0,
        ssb_c12_nm: float = 0.0,
        ssb_phi12_deg: float = 0.0,
        ssb_compute_enabled: bool = True,
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
        h5_url: str | None = None,
        h5_urls: Sequence[str] | None = None,
        lazy_url: str | None = None,
        lazy_urls: Sequence[str] | None = None,
        h5_uint8_lossless: bool = False,
        detector_shape: tuple[int, int] | None = None,
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
        # ``from_folder`` attaches these private objects to its lazy
        # Dataset5dstem before construction. Capturing them here lets the
        # initial compare refresh paint persistent previews before the factory
        # attaches its polling metadata.
        self._compare_preview_cache = getattr(data, "_compare_preview_cache", None)
        self._compare_preview_sources = getattr(
            data, "_compare_preview_sources", None
        )
        self._compare_loaded_source_signatures = getattr(
            data,
            "_compare_loaded_source_signatures",
            {},
        )
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
        # ``webgpu`` moves detector reductions into the browser. Native CUDA
        # and MPS are inferred from the typed data payload.
        if backend == "webgpu":
            offline = True  # routes the rest of __init__ through the offline pack path
        elif backend is not None:
            raise ValueError(
                f"backend must be 'webgpu' or None, got {backend!r}. "
                "Native CUDA or MPS compute is selected from the data payload."
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
        webgpu_lazy_urls = [str(value) for value in (lazy_urls or [])]
        if lazy_url:
            webgpu_lazy_urls = [str(lazy_url), *webgpu_lazy_urls]
        webgpu_lazy_urls = list(dict.fromkeys(webgpu_lazy_urls))
        webgpu_h5_urls = [str(value) for value in (h5_urls or [])]
        if h5_url:
            webgpu_h5_urls = [str(h5_url), *webgpu_h5_urls]
        # Preserve order but drop accidental duplicates; repeated URLs would waste
        # browser VRAM and create indistinguishable compare panels.
        if webgpu_h5_urls and webgpu_lazy_urls:
            raise ValueError("Use h5_urls= or lazy_urls=, not both.")
        webgpu_source_count = len(webgpu_lazy_urls) or len(webgpu_h5_urls)
        if webgpu_source_count:
            webgpu_h5_urls = list(dict.fromkeys(webgpu_h5_urls))
            if scan_shape is None:
                raise ValueError(
                    "Show4DSTEM(..., h5_urls=... or lazy_url/lazy_urls=...) "
                    "requires scan_shape=(rows, cols)."
                )
            if detector_shape is None:
                raise ValueError(
                    "Show4DSTEM(..., h5_urls=... or lazy_url/lazy_urls=...) "
                    "requires detector_shape=(rows, cols)."
                )
            if backend not in (None, "webgpu"):
                raise ValueError(
                    "Browser-source Show4DSTEM uses WebGPU; backend must be "
                    f"'webgpu' or None, got {backend!r}."
                )
            offline = True
            backend = "webgpu"

        # Extract underlying array / tensor + auto-calibrate from Dataset input
        # (duck-typed via the dual-slot private attributes _tensor / _array).
        if not webgpu_h5_urls:
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
        else:
            is_dataset5dstem_input = False

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
        self._axis_sampling = tuple(float(s) for s in sampling)
        self._axis_units = list(units)
        self.pixel_size = sampling[1]  # scan_col axis (horizontal scale bar)
        self.pixel_unit = units[1] if len(units) > 1 else "pixels"
        self.k_pixel_size = sampling[3] if len(sampling) > 3 else 1.0
        self.k_pixel_unit = units[3] if len(units) > 3 else "pixels"
        self._ssb_compute_config = {
            "semiangle_mrad": ssb_semiangle_mrad,
            "scan_sampling_A": ssb_scan_sampling_A,
            "det_sampling_mrad": ssb_det_sampling_mrad,
            "voltage_kV": ssb_voltage_kV,
            "energy_eV": ssb_energy_eV,
            "bf_radius": ssb_bf_radius,
            "bf_intensity_threshold": float(ssb_bf_intensity_threshold),
            "aberrations": ssb_aberrations,
            "rotation_angle_deg": float(ssb_rotation_angle_deg),
            "n_trials": int(ssb_n_trials),
            "refine": bool(ssb_refine),
            "seed": int(ssb_seed),
            "bf_subsample": ssb_bf_subsample,
        }
        self.ssb_compute_enabled = bool(ssb_compute_enabled)
        self.ssb_compute_n_trials = int(ssb_n_trials)
        self.ssb_compute_refine = bool(ssb_refine)
        self.ssb_compute_bf_subsample = 1.0 if ssb_bf_subsample is None else float(ssb_bf_subsample)
        self.ssb_compute_manual_aberrations = bool(ssb_manual_aberrations)
        self.ssb_compute_c10_nm = float(
            ssb_aberrations.get("C10", ssb_c10_nm)
            if ssb_aberrations is not None
            else ssb_c10_nm
        )
        self.ssb_compute_c12_nm = float(
            ssb_aberrations.get("C12", ssb_c12_nm)
            if ssb_aberrations is not None
            else ssb_c12_nm
        )
        phi12_default = (
            math.degrees(float(ssb_aberrations.get("phi12", math.radians(ssb_phi12_deg))))
            if ssb_aberrations is not None
            else ssb_phi12_deg
        )
        self.ssb_compute_phi12_deg = float(phi12_default)
        self.ssb_compute_rotation_angle_deg = float(ssb_rotation_angle_deg)
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
        # The public factory unwraps LoadResult explicitly. This implementation
        # receives the typed GPU data payload.
        self._webgpu_h5_source = bool(webgpu_source_count)
        self._cuda_compute_data = None
        self._cuda_compare_compute_backends: OrderedDict[int, Any] = OrderedDict()
        if self._webgpu_h5_source:
            self._device = torch.device("cpu")
            self._data_pre = None
            data_np = None
            rows, cols = (int(scan_shape[0]), int(scan_shape[1]))
            det_r, det_c = (int(detector_shape[0]), int(detector_shape[1]))
            if rows <= 0 or cols <= 0 or det_r <= 0 or det_c <= 0:
                raise ValueError(
                    "scan_shape and detector_shape entries must all be positive."
                )
            shape = (
                (webgpu_source_count, rows, cols, det_r, det_c)
                if webgpu_source_count > 1
                else (rows, cols, det_r, det_c)
            )
        else:
            # Dataset5dstem is the CUDA/MPS-friendly 5D series wrapper. Keep it as a
            # frame-backed object instead of calling `.tensor`: sharded CUDA series may
            # hold each 18 GiB no-bin master on a different GPU, and `.tensor` would
            # gather everything onto one card.
            is_dataset5dstem = type(data).__name__ == "Dataset5dstem" and hasattr(
                data, "frame"
            )
            # cupy array (io.load default on CUDA) -> ZERO-COPY torch tensor on the same
            # GPU via dlpack. Without this, the fallback cp.asnumpy round-trips the whole
            # block to CPU and re-uploads (a 19.3 GB no-bin load -> ~58 GB transient and
            # an OOM kernel crash). dlpack keeps it on-device, no copy.
            if type(data).__module__.split(".")[0] == "cupy":
                self._cuda_compute_data = data
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
                tuple(self._data_pre.shape)
                if self._data_pre is not None
                else data_np.shape
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
        if self._webgpu_h5_source:
            self._data = None
            self._dataset_page_config = None
        elif self._data_pre is not None:
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
        if _verbose and not self._webgpu_h5_source:
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
        self._set_vi_product_maps(
            {
                "DPC_row": DPC_row,
                "DPC_col": DPC_col,
                "SSB": SSB,
            }
        )
        initial_vi_source = self._normalise_vi_source(vi_source or "roi")
        if initial_vi_source != "roi" and initial_vi_source not in self._vi_product_maps:
            available = ["roi", *self.vi_product_labels]
            raise ValueError(
                f"vi_source={vi_source!r} requires a matching map. "
                f"Available image sources: {available}."
            )
        self.vi_source = initial_vi_source
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
        if self._webgpu_h5_source:
            self._offline_codec = offline_codec
            self._h5_url = webgpu_h5_urls[0] if len(webgpu_h5_urls) == 1 else ""
            self._h5_urls = json.dumps(webgpu_h5_urls) if len(webgpu_h5_urls) > 1 else ""
            self._lazy_url = webgpu_lazy_urls[0] if len(webgpu_lazy_urls) == 1 else ""
            self._lazy_urls = (
                json.dumps(webgpu_lazy_urls) if len(webgpu_lazy_urls) > 1 else ""
            )
            self._h5_uint8_lossless = bool(h5_uint8_lossless)
            self.offline = True
            self._offline_stack = b""
            self._offline_url = ""
            self._offline_chunks = ""
            self._offline_bslz4 = ""
            self._offline_bad_px = ""
            self._offline_gzip = False
            self.ssb_compute_enabled = False
            self.dp_global_min = MIN_LOG_VALUE
            self.dp_global_max = 1.0
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
            det_size = min(self.det_rows, self.det_cols)
            if center is not None:
                self.center_row = float(center[0])
                self.center_col = float(center[1])
            else:
                self.center_row = float(self.det_rows / 2)
                self.center_col = float(self.det_cols / 2)
            self.bf_radius = float(
                bf_radius if bf_radius is not None else det_size * DEFAULT_BF_RATIO
            )
            self._cached_bf_virtual = None
            self._cached_abf_virtual = None
            self._cached_adf_virtual = None
            self._cached_haadf_virtual = None
            self._compare_virtual_page_cache = OrderedDict()
            self._compare_virtual_page_cache_bytes = 0
            self._compare_diffraction_cache = OrderedDict()
            self._compare_diffraction_cache_bytes = 0
            self._compare_cache_pages = max(0, int(compare_cache_pages))
            self._compare_cache_max_bytes = (
                None
                if compare_cache_max_bytes is None
                else max(0, int(compare_cache_max_bytes))
            )
            self._compare_compute_lock = threading.RLock()
            self._compare_cache_lock = threading.RLock()
            self._compare_cache_generation = 0
            self._compare_cache_warm_stop = None
            self._compare_cache_warm_thread = None
            self._compare_cache_warm_status = "idle"
            self._compare_page_stop = None
            self._compare_page_thread = None
            self._compare_maintenance_stop = None
            self._compare_maintenance_thread = None
            self._compare_page_worker_lock = threading.Lock()
            self._compare_page_request_lock = threading.RLock()
            self._compare_page_generation_counter = 0
            self._compare_page_last_error = ""
            self._compare_page_last_send_error = ""
            self._compare_page_fresh_indices = set()
            self._compare_page_working_images = {}
            self._compare_page_paint_clients = set()
            self._compare_page_paint_ack_enabled = False
            self._folder_update_page_idx = -1
            self._folder_update_expected_indices = ()
            self._folder_update_backend_complete_generation = 0
            self._folder_update_painted_generation = 0
            self._folder_update_painted_page_idx = -1
            self._folder_update_paint_timeout_seconds = 30.0
            self._folder_update_paint_timeout = None
            self._folder_update_paint_timeout_generation = 0
            self.roi_mode = "circle"
            self.roi_center_col = self.center_col
            self.roi_center_row = self.center_row
            self.roi_center = [self.center_row, self.center_col]
            self.roi_radius = float(max(1.0, self.bf_radius))
            self.roi_active = True
            self.virtual_image_bytes = np.zeros(
                self.shape_rows * self.shape_cols, dtype=np.float32
            ).tobytes()
            self.frame_bytes = np.zeros(
                self.det_rows * self.det_cols, dtype=np.float32
            ).tobytes()
            visible_count = (
                min(int(self.n_frames), max(1, int(compare_max_panels)))
                if self._multiple_view_active()
                else 0
            )
            if visible_count:
                self.compare_panel_count = 0
                self.compare_panel_indices = []
                self.compare_virtual_image_bytes = b""
                self.compare_status = (
                    f"Loading {visible_count}/{int(self.n_frames)} browser WebGPU panels"
                )
            self.compare_page_count = max(
                1, math.ceil(max(1, int(self.n_frames)) / max(1, int(compare_max_panels)))
            )
            self.gpu_memory_label = (
                "Browser WebGPU lazy source"
                if webgpu_lazy_urls
                else "Browser WebGPU HDF5 source"
            )
            self.memory_warning = ""
            return
        # Histogram axis range — first frame is enough (JS does per-frame percentile clipping).
        # Cast to float for min/max reductions: PyTorch CUDA lacks integer min/max kernels,
        # and the first slice is tiny (144 KB at 192×192) so the cast is free.
        # .frame(0) (not [0]) so a paged Dataset5dstem brings frame 0 onto the GPU
        # for this reduction instead of handing back an offloaded CPU tensor.
        # Require hasattr(frame): MAPED's multi-GPU Dataset5dstem and other
        # duck-types may share the class name without the paging API.
        if type(self._data).__name__ == "Dataset5dstem" and hasattr(
            self._data, "frame"
        ):
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
            tuple[Any, ...], tuple[bytes, tuple[int, ...], str, int, str | None]
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
        self._compare_page_stop: threading.Event | None = None
        self._compare_page_thread: threading.Thread | None = None
        self._compare_maintenance_stop: threading.Event | None = None
        self._compare_maintenance_thread: threading.Thread | None = None
        self._compare_page_worker_lock = threading.Lock()
        self._compare_page_request_lock = threading.RLock()
        self._compare_page_generation_counter = 0
        self._compare_page_last_error = ""
        self._compare_page_last_send_error = ""
        self._compare_page_fresh_indices: set[int] = set()
        self._compare_page_working_images: dict[int, np.ndarray] = {}
        self._compare_page_paint_clients: set[str] = set()
        self._compare_page_paint_ack_enabled = False
        self._folder_update_page_idx = -1
        self._folder_update_expected_indices: tuple[int, ...] = ()
        self._folder_update_backend_complete_generation = 0
        self._folder_update_painted_generation = 0
        self._folder_update_painted_page_idx = -1
        self._folder_update_paint_timeout_seconds = 30.0
        self._folder_update_paint_timeout: threading.Timer | None = None
        self._folder_update_paint_timeout_generation = 0
        self.on_msg(self._handle_compare_page_paint_msg)
        self._compare_cache_pages = max(0, int(compare_cache_pages))
        self._compare_cache_max_bytes = (
            None
            if compare_cache_max_bytes is None
            else max(0, int(compare_cache_max_bytes))
        )
        preview_cache = getattr(self, "_compare_preview_cache", None)
        if preview_cache is not None and bool(getattr(preview_cache, "enabled", False)):
            self.compare_page_cache_state = "miss"
            self.compare_page_progressive_enabled = bool(
                type(self._data).__name__ == "Dataset5dstem"
                and getattr(self._data, "is_lazy", False)
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
        self.observe(self._on_vi_source_change, names=["vi_source"])
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
        if self._uses_progressive_compare_pages():
            # The selected first master is already resident for calibration.
            # Paint its diffraction pattern now; the page worker replaces this
            # with the requested average after it has loaded/cached each wave.
            frame = self._diffraction_frame_for_index(int(self.frame_idx))
            self._store_compare_diffraction_cache(int(self.frame_idx), frame)
            self.frame_bytes = np.ascontiguousarray(
                self._diffraction_frame_as_numpy(frame),
                dtype=np.float32,
            ).tobytes()
        else:
            self._update_frame()
        if _verbose:
            print(f"  virtual image + frame: {time.perf_counter() - _tc:.2f}s")

        # Path animation: observe index changes from frontend
        self.observe(self._on_path_index_change, names=["path_index"])
        self.observe(self._on_gif_export, names=["_gif_export_requested"])
        self.observe(self._on_export_request_change, names=["export_request"])
        self.observe(self._on_ssb_compute_request_change, names=["ssb_compute_request"])

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
                "MetalRawBackend": ("Apple GPU (raw Metal)", "Apple unified memory"),
                "CudaKernelCompute": ("NVIDIA GPU (CUDA, cupy)", "GPU VRAM"),
            }.get(cls, (None, None))
            if backend is None:  # TorchBackend depends on the torch device
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
                "vi_product_labels",
                "vi_product_map_frames",
                "vi_product_maps_bytes",
                "vi_preset_labels",
                "vi_preset_map_frames",
                "vi_preset_maps_bytes",
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
        # Browser-source mode: the raw frames stay as files on disk and WebGPU
        # reads HDF5 byte ranges or an explicitly supplied internal lazy source
        # at runtime. Do not pack/embed data for these browser-source paths.
        if (
            getattr(self, "_h5_url", "")
            or getattr(self, "_h5_urls", "")
            or getattr(self, "_lazy_url", "")
            or getattr(self, "_lazy_urls", "")
        ):
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
        if self._data is None and not getattr(self, "_webgpu_h5_source", False):
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
            # The interactive export fetches its data file over HTTP (CORS blocks
            # file://), so write a double-click launcher that serves the folder
            # and opens this page in Chrome. The static "report" kind is
            # self-contained and needs no launcher.
            from quantem.widget.command_launcher import write_command_launcher

            write_command_launcher(
                export_path.parent, "Show4DSTEM", viewer_html=export_path.name
            )
        size_mb = export_path.stat().st_size / (1024 * 1024)
        mode = self._export_mode_label(dtype, det_bin, scan_bin, export_kind=kind)
        if kind == "report":
            # Self-contained single HTML: no data file, opens by double-click.
            how = "self-contained HTML - double-click to open, no server needed"
        else:
            # Interactive: HTML + a data file fetched over HTTP (CORS blocks
            # file://), so it needs the local server, not a bare double-click.
            how = (
                f"WebGPU folder - reads its data file over HTTP; double-click "
                f"Show4DSTEM.command in {export_path.parent.name}/ to open (a bare "
                f"double-click of the HTML will not load the data)"
            )
        self.export_status = f"Exported {export_path.name} ({size_mb:.1f} MB, {mode}) - {how}"
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

    def _default_ssb_calibration_filename(self) -> str:
        label = self.title.strip() or "show4dstem"
        slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        if not slug:
            slug = "show4dstem"
        shape = f"{self.shape_rows}x{self.shape_cols}x{self.det_rows}x{self.det_cols}"
        return f"{slug}_{shape}_ssb_calibration.json"

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
        if getattr(self, "_webgpu_h5_source", False):
            if det_bin != 1 or scan_bin != 1:
                raise ValueError(
                    "Binned interactive raw export is not available for browser-source "
                    "Show4DSTEM exports because the browser reads the real H5 data "
                    "directly. Use det_bin=1 and scan_bin=1 to preserve the full data."
                )
            prev_enabled = self.export_enabled
            prev_save_state = self._save_state
            prev_bad_px = self._offline_bad_px
            self.export_enabled = False
            self._save_state = True
            try:
                if not self._offline_bad_px and self._h5_url:
                    embedded_bad_px = _h5_bad_pixel_json_for_export(
                        self._h5_url,
                        out.parent,
                    )
                    if embedded_bad_px is not None:
                        self._offline_bad_px = embedded_bad_px
                embed_minimal_html(
                    str(out),
                    views=[self],
                    title=title or self.title or "Show4DSTEM",
                    drop_defaults=False,
                    state=dependency_state([self], drop_defaults=False),
                )
                _inject_show4dstem_h5_webgpu_tuning(
                    out,
                    dtype=dtype,
                    h5_uint8_lossless=bool(self._h5_uint8_lossless),
                )
            finally:
                self.export_enabled = prev_enabled
                self._save_state = prev_save_state
                self._offline_bad_px = prev_bad_px
            ensure_mobile_viewport(out)
            return out
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
    ) -> "Show4DSTEM":
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
        product_kwargs = self._export_vi_product_kwargs(scan_bin=scan_bin)
        export_vi_source = (
            self.vi_source
            if self.vi_source == "roi" or self.vi_source in product_kwargs
            else "roi"
        )
        # Export data is a compact tensor, so always use the universal base
        # viewer. Backend-specific subclasses require their native storage
        # handles and must not be reconstructed from this tensor payload.
        clone = Show4DSTEM(
            data,
            sampling=sampling,
            units=units,
            center=center,
            bf_radius=max(1.0, self.bf_radius / k_scale),
            precompute_virtual_images=False,
            vi_source=export_vi_source,
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
            ssb_compute_enabled=False,
            verbose=False,
            **product_kwargs,
        )
        clone.load_state_dict(self._export_state_for_bin(det_bin, scan_bin=scan_bin))
        clone._pack_export_inline(dtype=dtype)
        clone.export_enabled = False
        clone.export_status = ""
        clone.export_payload = b""
        clone.export_payload_id = ""
        clone.export_filename = ""
        clone.ssb_compute_enabled = False
        clone.ssb_compute_status = ""
        clone.ssb_compute_request = ""
        clone.ssb_compute_busy = False
        clone.ssb_compute_calibration_json = ""
        clone.ssb_compute_calibration_filename = ""
        clone._save_state = True
        return clone

    def _export_vi_product_kwargs(self, *, scan_bin: int = 1) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for label, arr in getattr(self, "_vi_product_maps", {}).items():
            binned = Show4DSTEM._mean_scan_bin_array(arr, scan_bin)
            binned = np.ascontiguousarray(binned, dtype=np.float32)
            out[label] = binned[0] if binned.shape[0] == 1 else binned
        return out

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
        if arr.ndim == 3:
            nf, sr, sc = arr.shape
            return arr.reshape(
                nf, sr // scan_bin, scan_bin, sc // scan_bin, scan_bin
            ).mean(axis=(2, 4))
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
        self._close_compute()
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
            # A saved or exported widget model cannot retain this Python daemon.
            # Never serialize a live green/blue/amber/red badge that will be
            # untrue when the notebook is reopened without the current kernel.
            state["folder_watch_state"] = "stopped"
            state["folder_watch_detail"] = (
                "Folder watcher is not running in saved widget state. "
                "Re-run the cell to resume live folder updates."
            )
        return state

    def _with_initial_live_mount_state(self, fn, *args, **kwargs):
        """Keep live watcher state truthful during the display handshake."""
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
            "vi_source": self.vi_source,
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
                elif key == "vi_source":
                    val = self._normalise_vi_source(val)
                    if val != "roi" and val not in getattr(self, "_vi_product_maps", {}):
                        val = "roi"
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
        self.stop_compare_page_load(wait=True)
        self.stop_compare_maintenance(wait=True)
        self.stop_dataset_preload(wait=True)
        self.stop_compare_cache_warm(wait=True)
        preview_cache = getattr(self, "_compare_preview_cache", None)
        if preview_cache is not None:
            preview_cache.close()

        import gc

        data = self._data
        self._cuda_compute_data = None
        self._cuda_compare_compute_backends.clear()
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
        self._close_compute()
        if type(data).__name__ == "Dataset5dstem" and hasattr(data, "free"):
            data.free()
        self._data = None
        self.compare_page_progressive_enabled = False
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
        if type(self._data).__name__ == "Dataset5dstem" and hasattr(self._data, "frame"):
            # .frame() is paging-aware: when page_budget is set it brings this
            # dataset into VRAM and evicts the least-recently-used one; when
            # paging is off it is identical to self._data[frame_idx].
            return self._data.frame(self.frame_idx)
        if self.n_frames > 1:
            return self._data[self.frame_idx]
        return self._data

    @property
    def _compute_frame_data(self):
        """Current frame data for backend reductions, preserving CuPy when present."""
        cuda_data = getattr(self, "_cuda_compute_data", None)
        if cuda_data is not None:
            if self.n_frames > 1:
                return cuda_data[self.frame_idx]
            return cuda_data
        return self._frame_data

    @property
    def _compute(self):
        """Return the current frame's GPU detector session."""
        fd = self._compute_frame_data
        if getattr(self, "_compute_for", None) is not fd:
            from quantem.gpu.detector import prepare

            previous = getattr(self, "_compute_backend", None)
            if previous is not None:
                previous.close()
            self._compute_backend = prepare(fd)
            self._compute_for = fd
        return self._compute_backend

    def _close_compute(self) -> None:
        """Close the current detector session before releasing its data."""
        session = getattr(self, "_compute_backend", None)
        if session is not None:
            session.close()
        self._compute_backend = None
        self._compute_for = None

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

    def _on_ssb_compute_request_change(self, change):
        """JS More → Compute SSB request."""
        raw = str(change.get("new") or "")
        if not raw:
            return
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"action": raw}
        action = str(payload.get("action", "compute_ssb")).strip().lower()
        self.ssb_compute_request = ""  # consume trigger before any long work
        if action in {"clear", "reset"}:
            self.ssb_compute_status = ""
            return
        if action not in {"compute_ssb", "ssb", "compute"}:
            self.ssb_compute_status = f"SSB failed: unknown request {action!r}"
            return
        if not self.ssb_compute_enabled:
            self.ssb_compute_status = (
                "Compute SSB is available only in a live Python kernel. "
                "Precompute SSB before exporting standalone HTML."
            )
            return
        if self.ssb_compute_busy:
            self.ssb_compute_status = "SSB compute is already running."
            return

        kwargs: dict[str, Any] = {}
        key_map = {
            "semiangle_mrad": "semiangle_mrad",
            "scan_sampling_A": "scan_sampling_A",
            "det_sampling_mrad": "det_sampling_mrad",
            "voltage_kV": "voltage_kV",
            "energy_eV": "energy_eV",
            "bf_radius": "bf_radius",
            "bf_intensity_threshold": "bf_intensity_threshold",
            "rotation_angle_deg": "rotation_angle_deg",
            "n_trials": "n_trials",
            "refine": "refine",
            "seed": "seed",
            "bf_subsample": "bf_subsample",
            "lock_aberrations": "lock_aberrations",
        }
        for src_key, dst_key in key_map.items():
            if src_key in payload:
                kwargs[dst_key] = payload[src_key]
        if "n_trials" in kwargs:
            self.ssb_compute_n_trials = int(kwargs["n_trials"])
        if "refine" in kwargs:
            self.ssb_compute_refine = bool(kwargs["refine"])
        if "bf_subsample" in kwargs:
            self.ssb_compute_bf_subsample = float(kwargs["bf_subsample"])

        manual_aberrations = bool(payload.get("manual_aberrations", False))
        self.ssb_compute_manual_aberrations = manual_aberrations
        if manual_aberrations:
            c10 = float(payload.get("c10_nm", self.ssb_compute_c10_nm))
            c12 = float(payload.get("c12_nm", self.ssb_compute_c12_nm))
            phi12_deg = float(payload.get("phi12_deg", self.ssb_compute_phi12_deg))
            rotation_deg = float(
                payload.get("rotation_angle_deg", self.ssb_compute_rotation_angle_deg)
            )
            self.ssb_compute_c10_nm = c10
            self.ssb_compute_c12_nm = c12
            self.ssb_compute_phi12_deg = phi12_deg
            self.ssb_compute_rotation_angle_deg = rotation_deg
            kwargs["aberrations"] = {
                "C10": c10,
                "C12": c12,
                "phi12": math.radians(phi12_deg),
            }
            kwargs["rotation_angle_deg"] = rotation_deg
            kwargs["lock_aberrations"] = True
        lock_c10 = bool(payload.get("lock_c10", self.ssb_compute_lock_c10))
        lock_c12 = bool(payload.get("lock_c12", self.ssb_compute_lock_c12))
        self.ssb_compute_lock_c10 = lock_c10
        self.ssb_compute_lock_c12 = lock_c12
        if not manual_aberrations:
            kwargs["lock_c10"] = lock_c10
            kwargs["lock_c12"] = lock_c12

        def worker() -> None:
            try:
                self.compute_ssb(verbose=bool(payload.get("verbose", False)), **kwargs)
            except Exception:
                # compute_ssb already publishes the concise failure string.
                return

        thread = threading.Thread(
            target=worker,
            name="Show4DSTEM-compute-SSB",
            daemon=True,
        )
        thread.start()

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
        progressive = self._uses_progressive_compare_pages()
        self._refresh_compare_virtual_images()
        if self._multiple_view_active():
            # A page change normally moves the selected diffraction panel to
            # the first visible slot. Suppress that observer's direct eager
            # frame load while a folder page is being streamed in the worker.
            self._suppress_progressive_frame_update = progressive
            try:
                self._ensure_current_compare_frame_visible()
            finally:
                self._suppress_progressive_frame_update = False
            if not progressive:
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
        if self._multiple_view_active():
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
        if self._multiple_view_active():
            self._sync_compare_page_to_frame_idx()
            if getattr(self, "_suppress_progressive_frame_update", False) or (
                self._uses_progressive_compare_pages()
                and bool(self.compare_page_loading)
            ):
                return
        # Recompute virtual image only when it is visible. In multiple mode the
        # visible virtual-image surface is the compare grid; switching back to
        # single recomputes the per-frame virtual image in _on_compare_config_change.
        if not self._multiple_view_active():
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
        # Chunked torch integer reference path used by small tests.
        # int64 accumulator, no float32 cast of the stack — keeps the data in its
        # native dtype (a float32 cast doubles memory and is lossy above 2^24),
        # is bit-exact, and avoids the MPS "tensor dims larger than INT_MAX"
        # error a single full-stack reduce hits once positions*det > 2^31 (a bin2
        # 512x512x96x96 stack = 2.42e9 elements). The chunk cap keeps each op's
        # element count well under 2^31; the (det, det) accumulator is tiny.
        # Mean DP over all scan positions via the compute backend (TorchBackend
        # int64-accumulates in chunks; MetalRawBackend uses the raw detector_sum
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
        product = self._current_vi_product_map()
        if product is not None:
            return np.asarray(product, dtype=np.float32).copy()
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
                self._draw_scalebar_overlay(
                    out, float(self.pixel_size), self.pixel_unit or "px"
                )
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
                    self.vi_source = "roi"
                    self.roi_active = True
                    self.roi_mode = "circle"
                    self.roi_center_row = center_row
                    self.roi_center_col = center_col
                    self.roi_radius = float(max(1.0, bf))
            elif preset_name == "abf":
                with self.hold_trait_notifications():
                    self.vi_source = "roi"
                    self.roi_active = True
                    self.roi_mode = "annular"
                    self.roi_center_row = center_row
                    self.roi_center_col = center_col
                    self.roi_radius_inner = float(max(0.5, bf * 0.5))
                    self.roi_radius = float(max(1.0, bf))
            elif preset_name == "adf":
                with self.hold_trait_notifications():
                    self.vi_source = "roi"
                    self.roi_active = True
                    self.roi_mode = "annular"
                    self.roi_center_row = center_row
                    self.roi_center_col = center_col
                    self.roi_radius_inner = float(max(1.0, bf))
                    self.roi_radius = float(max(bf + 1.0, bf * 2.0))
            elif preset_name == "haadf":
                with self.hold_trait_notifications():
                    self.vi_source = "roi"
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
        if not self._multiple_view_active():
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

    def _current_frame_bytes(self) -> bytes | None:
        """Build the current diffraction payload without mutating widget state."""
        if self._data is None:
            return None
        if (
            self.view_mode == "multiple"
            and self.n_frames > 1
            and self._normalise_compare_dp_mode(self.compare_dp_mode) == "average"
        ):
            frame = self._average_compare_diffraction_frame()
        elif self.view_mode == "multiple" and self.n_frames > 1:
            frame = self._get_cached_compare_diffraction_frame(int(self.frame_idx))
            if frame is None:
                frame = self._diffraction_frame_for_index(self.frame_idx)
                self._store_compare_diffraction_cache(int(self.frame_idx), frame)
        else:
            frame = self._diffraction_frame_for_index(self.frame_idx)

        # Cast small frame to float32 for stats and JS transfer. Bulk data
        # stays in native dtype; only this single 192×192 (~144 KB) frame
        # gets promoted.
        if not isinstance(frame, torch.Tensor):
            # Cached diffraction payloads are often read-only np.frombuffer
            # views. Copy this tiny frame before handing it to Torch so later
            # consumers never inherit undefined write semantics or warnings.
            frame = torch.from_numpy(np.array(frame, copy=True))
        if frame.dtype != torch.float32:
            frame = frame.float()
        return frame.detach().cpu().numpy().tobytes()

    def _update_frame(self, change=None):
        """Send raw float32 frame to frontend (JS handles scale/colormap)."""
        payload = self._current_frame_bytes()
        if payload is None:
            return
        # Stats compute moved to JS (frontend has frame_bytes; computeStats() in
        # js/stats.ts does mean/min/max/std on the Float32Array directly,
        # avoiding 4 sync trait round-trips per scan-position click).
        self.frame_bytes = payload

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

        if type(data_source).__name__ == "Dataset5dstem" and hasattr(
            data_source, "frame"
        ):
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
        if self.vi_source != "roi":
            return
        if not self._multiple_view_active():
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
        if self.vi_source != "roi":
            return
        if not self._multiple_view_active():
            self._compute_virtual_image_from_roi()
        self._refresh_compare_virtual_images()

    def _on_vi_source_change(self, change=None):
        """Switch the image panel between detector-ROI VI and product maps."""
        source = self._normalise_vi_source(change.get("new") if change else self.vi_source)
        if source != self.vi_source:
            self.vi_source = source
            return
        if source != "roi" and source not in getattr(self, "_vi_product_maps", {}):
            self.vi_source = "roi"
            return

        if source == "roi":
            if not self._multiple_view_active():
                self._compute_virtual_image_from_roi()
            self._refresh_compare_virtual_images()
            return
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
        # Flat scan indices inside the ROI, reduced (mean/sum/max) by the
        # compute backend (CUDA RawKernel, raw Metal for chunk-backed frames, or
        # Torch/NumPy fallback). One widget call path keeps results consistent
        # across backends.
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

    @property
    def preview_cache_info(self) -> dict[str, Any]:
        """Persistent reduced-preview cache status for this folder viewer."""
        cache = getattr(self, "_compare_preview_cache", None)
        if cache is None:
            return {
                "enabled": False,
                "mode": "off",
                "path": None,
                "max_bytes": None,
                "entries": 0,
                "current_bytes": 0,
                "pending_writes": 0,
                "closed": False,
                "errors": 0,
                "hits": 0,
                "misses": 0,
                "invalidations": 0,
                "corruptions": 0,
                "writes": 0,
                "write_errors": 0,
                "evictions": 0,
                "bytes_read": 0,
                "bytes_written": 0,
                "lookup_ms": 0.0,
                "read_ms": 0.0,
                "write_ms": 0.0,
            }
        return dict(cache.info)

    def clear_preview_cache(self) -> Self:
        """Delete this viewer's derived disk previews, never raw 4D data."""
        cache = getattr(self, "_compare_preview_cache", None)
        if cache is not None:
            cache.clear()
        return self

    @staticmethod
    def _master_key(master) -> str:
        return str(pathlib.Path(master).expanduser().resolve())

    @staticmethod
    def _validate_folder_watch_interval(interval: float) -> float:
        """Return a finite positive folder polling interval."""
        value = float(interval)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                "watch interval must be a finite value > 0 seconds, "
                f"got {interval!r}"
            )
        return value

    def _folder_watch_is_alive(self) -> bool:
        """Return whether this widget's folder worker is actually running."""
        thread = getattr(self, "_folder_watch_thread", None)
        return bool(thread is not None and thread.is_alive())

    @staticmethod
    def _folder_readiness_signature(report: Any) -> str:
        """Canonicalize one header-only readiness signature for probation."""
        return json.dumps(
            report.source_signature,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    @staticmethod
    def _folder_readiness_is_waiting(report: Any) -> bool:
        """Classify transient acquisition states separately from bad data."""
        actual = getattr(report, "actual_frames", None)
        expected = getattr(report, "expected_frames", None)
        if actual is not None and expected is not None:
            if int(actual) < int(expected):
                return True
            if int(actual) > int(expected):
                return False
        reason = str(getattr(report, "reason", "")).casefold()
        action = str(getattr(report, "action", "")).strip().casefold()
        permanent_markers = (
            "inconsistent detector",
            "inconsistent dtype",
            "inconsistent scan_shape",
            "incompatible",
            "does not match",
            "conflicting",
            "expected at least (frame, det_row, det_col)",
        )
        corrective_prefixes = ("fix ", "repair ", "use ", "pass ")
        if any(marker in reason for marker in permanent_markers) or action.startswith(
            corrective_prefixes
        ):
            return False
        if action.startswith("wait "):
            return True
        transient_markers = (
            "missing",
            "empty",
            "zero stored frames",
            "not readable hdf5",
            "cannot be inspected",
            "changed during readiness inspection",
            "headers are incomplete",
            "header to finish writing",
            "no entry/data",
            "has no entry/data/data",
        )
        return any(marker in reason for marker in transient_markers)

    def _compact_folder_watch_detail(self, detail: str, *, limit: int = 480) -> str:
        """Keep synced status text bounded and free of the watched root path."""
        text = " ".join(str(detail).split())
        source = getattr(self, "_folder_source", None) or {}
        folder = source.get("folder")
        if folder is not None:
            root = str(pathlib.Path(folder))
            variants = {root, os.path.realpath(root)}
            variants.update(
                path[len("/private") :]
                for path in tuple(variants)
                if path.startswith("/private/")
            )
            for path in sorted(variants, key=len, reverse=True):
                text = text.replace(f"{path}{os.sep}", "")
                text = text.replace(
                    path,
                    pathlib.Path(path).name or "watched folder",
                )
        # Readiness libraries can surface a linked source outside the watched
        # root. Reduce any remaining whitespace-delimited absolute path to its
        # basename so a badge never exposes a host filesystem layout.
        compact_tokens: list[str] = []
        for token in text.split(" "):
            leading = token[: len(token) - len(token.lstrip("([{<"))]
            candidate = token[len(leading) :]
            trailing = candidate[len(candidate.rstrip(".,;:)]}>")) :]
            if trailing:
                candidate = candidate[: len(candidate) - len(trailing)]
            if candidate.startswith(os.sep) and os.sep in candidate[1:]:
                candidate = pathlib.Path(candidate).name or "source file"
            compact_tokens.append(f"{leading}{candidate}{trailing}")
        text = " ".join(compact_tokens)
        if len(text) > int(limit):
            text = f"{text[: max(0, int(limit) - 1)].rstrip()}…"
        return text

    def _folder_issue_detail(
        self,
        master: Any,
        reason: str,
        action: str = "",
    ) -> str:
        """Build compact, corrective detail for the synced watch badge."""
        name = pathlib.Path(master).name or str(master)
        reason = str(reason).strip().rstrip(".")
        action = str(action).strip()
        detail = f"{name}: {reason}." if reason else f"{name}: not ready."
        return self._compact_folder_watch_detail(f"{detail} {action}".strip())

    def _finish_folder_poll_status(
        self,
        *,
        waiting: Sequence[str],
        errors: Sequence[str],
    ) -> None:
        """Publish the final state of one successful poll while still alive."""
        self._folder_poll_waiting = str(waiting[0]) if waiting else ""
        self._folder_poll_error = str(errors[0]) if errors else ""
        if not self._folder_watch_is_alive():
            return
        if errors:
            set_folder_watch_status(self, "error", str(errors[0]))
        elif waiting:
            set_folder_watch_status(self, "waiting", str(waiting[0]))
        elif bool(getattr(self, "_folder_update_pending", False)):
            set_folder_watch_status(
                self,
                "updating",
                "Loading the newly arrived data on the visible page.",
            )
        else:
            set_folder_watch_status(self, "watching", "")

    def _finish_folder_page_update(
        self,
        generation: int,
        *,
        error: str = "",
    ) -> None:
        """Resolve a watched append only after its visible page is authoritative."""
        if not bool(getattr(self, "_folder_update_pending", False)):
            return
        pending_generation = int(
            getattr(self, "_folder_update_generation", generation)
        )
        if int(generation) != pending_generation:
            return
        if not error and bool(
            getattr(self, "_compare_page_paint_ack_enabled", False)
        ):
            self._folder_update_backend_complete_generation = int(generation)
            if (
                int(getattr(self, "_folder_update_painted_generation", 0))
                != int(generation)
                or int(getattr(self, "_folder_update_painted_page_idx", -1))
                != int(getattr(self, "_folder_update_page_idx", -1))
            ):
                self._schedule_folder_page_paint_timeout(int(generation))
                return
        self._reset_folder_page_update_tracking()
        if not self._folder_watch_is_alive():
            return
        if error:
            set_folder_watch_status(
                self,
                "error",
                self._compact_folder_watch_detail(error),
            )
        elif getattr(self, "_folder_poll_error", ""):
            set_folder_watch_status(
                self,
                "error",
                str(self._folder_poll_error),
            )
        elif getattr(self, "_folder_poll_waiting", ""):
            set_folder_watch_status(
                self,
                "waiting",
                str(self._folder_poll_waiting),
            )
        else:
            set_folder_watch_status(self, "watching", "")

    def _cancel_folder_page_paint_timeout(self) -> None:
        """Cancel the generation-scoped browser paint deadline, if any."""
        timer = getattr(self, "_folder_update_paint_timeout", None)
        self._folder_update_paint_timeout = None
        self._folder_update_paint_timeout_generation = 0
        if timer is not None:
            timer.cancel()

    def _reset_folder_page_update_tracking(self) -> None:
        """Clear one watched append's backend/browser paint lifecycle."""
        self._cancel_folder_page_paint_timeout()
        self._folder_update_pending = False
        self._folder_update_generation = 0
        self._folder_update_page_idx = -1
        self._folder_update_expected_indices = ()
        self._folder_update_backend_complete_generation = 0
        self._folder_update_painted_generation = 0
        self._folder_update_painted_page_idx = -1

    def _schedule_folder_page_paint_timeout(self, generation: int) -> None:
        """Turn a lost mounted-browser paint acknowledgement into Watch error."""
        current = getattr(self, "_folder_update_paint_timeout", None)
        if (
            current is not None
            and current.is_alive()
            and int(getattr(self, "_folder_update_paint_timeout_generation", 0))
            == int(generation)
        ):
            return
        self._cancel_folder_page_paint_timeout()
        delay = max(
            0.01,
            float(getattr(self, "_folder_update_paint_timeout_seconds", 30.0)),
        )

        def expired() -> None:
            if getattr(self, "_folder_update_paint_timeout", None) is not timer:
                return
            self._folder_update_paint_timeout = None
            self._folder_update_paint_timeout_generation = 0
            if (
                bool(getattr(self, "_folder_update_pending", False))
                and bool(getattr(self, "_compare_page_paint_ack_enabled", False))
                and int(getattr(self, "_folder_update_generation", 0))
                == int(generation)
                and int(
                    getattr(
                        self,
                        "_folder_update_backend_complete_generation",
                        0,
                    )
                )
                == int(generation)
            ):
                self._finish_folder_page_update(
                    int(generation),
                    error=(
                        "The browser did not confirm fresh visible pixels before "
                        "the paint deadline. Keep the notebook tab open, verify "
                        "the kernel connection, then retry or reload the page."
                    ),
                )

        timer = threading.Timer(delay, expired)
        timer.name = f"Show4DSTEM-paint-{int(generation)}"
        timer.daemon = True
        self._folder_update_paint_timeout = timer
        self._folder_update_paint_timeout_generation = int(generation)
        timer.start()

    def _handle_compare_page_paint_msg(
        self,
        _widget: Any,
        content: dict[str, Any],
        _buffers: list[Any],
    ) -> None:
        """Resolve watched page updates only after a mounted browser paints."""
        if not isinstance(content, dict):
            return
        message_type = str(content.get("type", ""))
        if message_type == "compare_page_paint_capability":
            if content.get("version") != 1 or not isinstance(
                content.get("active", True), bool
            ):
                return
            active = bool(content.get("active", True))
            client_id = content.get("client_id")
            if (
                isinstance(client_id, str)
                and bool(client_id)
                and len(client_id) <= 128
            ):
                if active:
                    self._compare_page_paint_clients.add(client_id)
                else:
                    self._compare_page_paint_clients.discard(client_id)
            else:
                return
            self._compare_page_paint_ack_enabled = bool(
                self._compare_page_paint_clients
            )
            if not self._compare_page_paint_ack_enabled and bool(
                getattr(self, "_folder_update_pending", False)
            ):
                generation = int(getattr(self, "_folder_update_generation", 0))
                if generation and int(
                    getattr(
                        self,
                        "_folder_update_backend_complete_generation",
                        0,
                    )
                ) == generation:
                    # The frontend unmounted after backend completion. There is
                    # no visible UI left to paint, so restore headless behavior.
                    self._finish_folder_page_update(generation)
            return
        if message_type != "compare_page_paint_ack":
            return
        if (
            not bool(getattr(self, "_compare_page_paint_ack_enabled", False))
            or content.get("version") != 1
            or content.get("paint_kind") != "fresh"
            or not bool(getattr(self, "_folder_update_pending", False))
        ):
            return

        def message_int(value: Any) -> int | None:
            if isinstance(value, bool):
                return None
            try:
                parsed = int(value)
            except (TypeError, ValueError, OverflowError):
                return None
            if isinstance(value, float) and not value.is_integer():
                return None
            return parsed

        generation = message_int(content.get("generation"))
        page_idx = message_int(content.get("page_idx"))
        pending_generation = int(getattr(self, "_folder_update_generation", 0))
        pending_page_idx = int(getattr(self, "_folder_update_page_idx", -1))
        if (
            generation is None
            or page_idx is None
            or generation != pending_generation
            or generation != int(self.compare_page_generation)
            or page_idx != pending_page_idx
            or page_idx != int(self.compare_page_idx)
        ):
            return

        painted_values = content.get("painted_indices")
        if not isinstance(painted_values, (list, tuple)):
            return
        painted: list[int] = []
        for value in painted_values:
            parsed = message_int(value)
            if parsed is None:
                return
            painted.append(parsed)
        expected = tuple(
            int(value)
            for value in getattr(self, "_folder_update_expected_indices", ())
        )
        current_expected = tuple(
            int(value) for value in self.compare_page_expected_indices
        )
        if not expected or tuple(painted) != expected or current_expected != expected:
            return

        self._folder_update_painted_generation = generation
        self._folder_update_painted_page_idx = page_idx
        if int(
            getattr(self, "_folder_update_backend_complete_generation", 0)
        ) == generation:
            self._finish_folder_page_update(generation)

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
        data = getattr(self, "_data", None)
        self.compare_page_progressive_enabled = bool(
            type(data).__name__ == "Dataset5dstem"
            and getattr(data, "is_lazy", False)
        )
        self._folder_known_masters = {
            self._master_key(master) for master in known_masters
        }
        self._folder_watch_stop = None
        self._folder_watch_thread = None
        self._folder_watch_started = False
        self._folder_ready_probation: dict[str, str] = {}
        self._folder_poll_lock = threading.Lock()
        # Folder-backed lazy data starts paged until the optional complete-
        # series preload proves every unhidden master fits. Publish that state
        # synchronously so status/debug consumers never observe a missing
        # residency value while the first progressive page is still loading.
        self._raw_preload_status = "paged"
        self._raw_residency_plan = {}
        self._compare_folder_refresh_pending = False
        self._reset_folder_page_update_tracking()
        self._folder_poll_waiting = ""
        self._folder_poll_error = ""
        set_folder_watch_status(self, "hidden", "")
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
        source attached. A newly discovered master must produce the same complete
        header/source signature on two consecutive polls before it is appended.
        New masters start as cold lazy slots, then join the complete series
        preload when the updated shape/dtype footprint still fits.
        """
        source = getattr(self, "_folder_source", None)
        if source is None:
            raise RuntimeError(
                "poll_folder() is available only on Show4DSTEM.from_folder(...) widgets."
            )
        if type(getattr(self, "_data", None)).__name__ != "Dataset5dstem":
            raise RuntimeError("poll_folder() requires a lazy Dataset5dstem backing.")
        from quantem.gpu import io as gpu_io

        poll_lock = getattr(self, "_folder_poll_lock", None)
        if poll_lock is None:
            poll_lock = threading.Lock()
            self._folder_poll_lock = poll_lock
        watch_active = self._folder_watch_is_alive()
        waiting_issues: list[str] = []
        error_issues: list[str] = []
        try:
            with poll_lock:
                if watch_active:
                    set_folder_watch_status(
                        self,
                        "updating",
                        "Checking the folder for new 4D-STEM data.",
                    )
                try:
                    # Do not pre-filter on frame count: incomplete candidates must
                    # stay visible to the readiness protocol so users see why a
                    # matching master has not appeared yet.
                    masters = gpu_io.discover(
                        str(source["folder"]),
                        pattern=source["pattern"],
                        recursive=source["recursive"],
                        scan_shape=None,
                        verbose=False,
                    )
                except ValueError as exc:
                    if "No files matching" not in str(exc):
                        raise
                    masters = []

                known = set(getattr(self, "_folder_known_masters", set()))
                probation = dict(
                    getattr(self, "_folder_ready_probation", {})
                )
                discovered_keys = {self._master_key(master) for master in masters}
                probation = {
                    key: signature
                    for key, signature in probation.items()
                    if key in discovered_keys and key not in known
                }
                added: list[int] = []
                labels = list(self.frame_labels)
                old_n_frames = int(self.n_frames)
                old_order = list(self.compare_panel_order or [])
                custom_order = (
                    len(old_order) == old_n_frames
                    and sorted(int(idx) for idx in old_order)
                    == list(range(old_n_frames))
                )
                candidates = []
                for master in masters:
                    key = self._master_key(master)
                    if key in known:
                        continue

                    try:
                        report = gpu_io.inspect(
                            master,
                            scan_shape=source["scan_shape"],
                        )
                    except Exception as exc:
                        probation.pop(key, None)
                        error_issues.append(
                            self._folder_issue_detail(
                                master,
                                f"readiness inspection failed ({type(exc).__name__}: {exc})",
                                "Check file permissions and HDF5 integrity, then retry.",
                            )
                        )
                        continue

                    if not bool(report.ready):
                        probation.pop(key, None)
                        detail = self._folder_issue_detail(
                            master,
                            report.reason,
                            report.action,
                        )
                        if self._folder_readiness_is_waiting(report):
                            waiting_issues.append(detail)
                        else:
                            error_issues.append(detail)
                        continue

                    signature = self._folder_readiness_signature(report)
                    if probation.get(key) != signature:
                        probation[key] = signature
                        waiting_issues.append(
                            self._folder_issue_detail(
                                master,
                                "complete headers found; waiting for one unchanged "
                                "follow-up readiness poll",
                                "Keep the master and linked detector files in place.",
                            )
                        )
                        continue

                    try:
                        validator = source.get("validate_master")
                        if callable(validator):
                            validator(master)
                        idx = len(self._data) + len(candidates)
                        label, loader = source["make_loader"](master, idx)
                    except Exception as exc:
                        error_issues.append(
                            self._folder_issue_detail(
                                master,
                                f"incompatible master ({type(exc).__name__}: {exc})",
                                "Use a matching scan shape, detector shape, and "
                                "dtype, or move this file out of the watched folder.",
                            )
                        )
                        # Keep the confirmed signature so a corrected contract can
                        # be retried immediately without hiding later valid files.
                        continue
                    candidates.append((master, key, idx, str(label), loader))

                self._folder_ready_probation = probation
                if candidates and watch_active:
                    set_folder_watch_status(
                        self,
                        "updating",
                        "Registering newly completed 4D-STEM data.",
                    )
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
                    probation.pop(key, None)
                    added.append(idx)
                self._folder_ready_probation = probation
                self._folder_known_masters = known

                if added:
                    # n_frames normally triggers a compare-grid render. A watched
                    # master must first remain a cold lazy slot, so publish only
                    # lightweight metadata here. Page selection or cache warming
                    # performs the first raw load explicitly.
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
                                self.compare_status = (
                                    self._compare_status_for_indices(
                                        self.compare_panel_indices,
                                        all_groups=(
                                            self._normalise_compare_group_mode(
                                                self.compare_group_mode
                                            )
                                            == "all"
                                        ),
                                    )
                                )
                    finally:
                        self._suppress_folder_append_refresh = False

                    self._raw_preload_status = "paged"
                    self._update_gpu_memory_status()
                    active = (
                        self._compare_ready_indices()
                        if self.view_mode == "multiple"
                        else []
                    )
                    refresh_active_page = bool(set(added).intersection(active))
                    if refresh_active_page:
                        # The append observer above is deliberately suppressed so
                        # a new master starts cold. Explicitly schedule only the
                        # affected visible page, then keep the badge in Updating
                        # until the progressive worker publishes authoritative
                        # current pixels for that stable slot.
                        self._folder_update_pending = True
                        self._cancel_folder_page_paint_timeout()
                        self._folder_update_generation = int(
                            self._compare_page_generation_counter
                        ) + 1
                        self._folder_update_page_idx = int(self.compare_page_idx)
                        self._folder_update_expected_indices = tuple(
                            int(idx) for idx in active
                        )
                        self._folder_update_backend_complete_generation = 0
                        self._folder_update_painted_generation = 0
                        self._folder_update_painted_page_idx = -1
                        self._refresh_compare_virtual_images()
                    else:
                        # Off-page arrivals remain cold. Optional maintenance can
                        # warm them without blocking or repainting the page the
                        # scientist is currently inspecting.
                        if bool(source.get("preload_all_if_fits", False)):
                            self.preload_all_datasets(background=True)
                        if bool(source.get("warm_cache", False)):
                            self.warm_compare_cache(background=True)

                if not masters and not known:
                    waiting_issues.append(
                        "No matching 4D-STEM master has arrived yet. Keep the "
                        "watcher running or check the folder and filename pattern."
                    )
                self._finish_folder_poll_status(
                    waiting=waiting_issues,
                    errors=error_issues,
                )
                return added
        except Exception as exc:
            self._reset_folder_page_update_tracking()
            if self._folder_watch_is_alive():
                set_folder_watch_status(
                    self,
                    "error",
                    self._compact_folder_watch_detail(
                        f"{type(exc).__name__}: {exc}. The watcher is still alive "
                        "and will retry; check the folder, storage, and file "
                        "permissions."
                    ),
                )
            raise

    def watch_folder(self, *, interval: float = 2.0) -> Self:
        """Poll the attached folder in the background and append ready masters."""
        source = getattr(self, "_folder_source", None)
        if source is None:
            raise RuntimeError(
                "watch_folder() is available only on Show4DSTEM.from_folder(...) widgets."
            )
        next_interval = self._validate_folder_watch_interval(interval)
        self.stop_folder_watch()

        stop = threading.Event()
        self._folder_watch_stop = stop
        self._folder_watch_started = True

        def _worker() -> None:
            try:
                while not stop.wait(next_interval):
                    try:
                        self.poll_folder()
                    except Exception:
                        # poll_folder publishes a corrective error and the next
                        # cycle retries without killing the mounted viewer.
                        continue
            finally:
                current = threading.current_thread()
                if getattr(self, "_folder_watch_stop", None) is stop:
                    self._folder_watch_stop = None
                if getattr(self, "_folder_watch_thread", None) is current:
                    self._folder_watch_thread = None
                if stop.is_set():
                    set_folder_watch_status(self, "stopped", "Folder watcher stopped.")
                else:
                    set_folder_watch_status(
                        self,
                        "error",
                        "Folder watch worker stopped unexpectedly. Call "
                        "watch_folder() to restart it.",
                    )

        thread = threading.Thread(
            target=_worker,
            name="Show4DSTEM-folder-watch",
            daemon=True,
        )
        self._folder_watch_thread = thread
        try:
            thread.start()
        except Exception as exc:
            if getattr(self, "_folder_watch_stop", None) is stop:
                self._folder_watch_stop = None
            if getattr(self, "_folder_watch_thread", None) is thread:
                self._folder_watch_thread = None
            set_folder_watch_status(
                self,
                "error",
                f"Folder watch worker could not start ({type(exc).__name__}: {exc}). "
                "Check the notebook kernel and retry.",
            )
            raise
        if thread.is_alive():
            set_folder_watch_status(self, "watching", "")
        else:
            set_folder_watch_status(
                self,
                "error",
                "Folder watch worker stopped before it could start polling. "
                "Call watch_folder() to retry.",
            )
        return self

    def stop_folder_watch(self) -> None:
        """Stop the background folder watcher, if one was started."""
        self._reset_folder_page_update_tracking()
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
        if bool(getattr(self, "_folder_watch_started", False)):
            set_folder_watch_status(self, "stopped", "Folder watcher stopped.")

    def close(self) -> None:
        """Stop background work and close the widget comm."""
        self._cancel_folder_page_paint_timeout()
        self.stop_folder_watch()
        self.stop_compare_page_load(wait=True)
        self.stop_compare_maintenance(wait=True)
        self.stop_dataset_preload(wait=True)
        self.stop_compare_cache_warm(wait=True)
        preview_cache = getattr(self, "_compare_preview_cache", None)
        if preview_cache is not None:
            preview_cache.close()
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
        self._close_compute()
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
            self.compare_page_expected_indices = []
            self.compare_page_loading = False
            self.compare_page_loaded_count = 0
            self.compare_page_cached_indices = []
            self.compare_page_cache_state = (
                "miss"
                if getattr(self, "_compare_preview_cache", None) is not None
                else "off"
            )
            self.compare_page_first_panel_ms = 0.0
            self.compare_page_first_fresh_ms = 0.0
            self.compare_page_total_ms = 0.0
            self.compare_page_panel_bytes = b""
            self.compare_page_panel_frame_idx = -1
            self.compare_page_panel_slot = -1
            self.compare_page_panel_cached = False

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
        from quantem.gpu.detector import prepare

        mask_np = (
            mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
        )
        backend = prepare(dataset)
        return np.asarray(backend.masked_sum(mask_np), dtype=np.float32)

    def _cuda_compare_backend_for_index(self, idx: int):
        """Return a cached CUDA compute backend for one compare-grid panel."""
        cuda_data = getattr(self, "_cuda_compute_data", None)
        if cuda_data is None:
            return None
        key = int(idx) if self.n_frames > 1 else 0
        cache = self._cuda_compare_compute_backends
        backend = cache.get(key)
        if backend is not None:
            cache.move_to_end(key)
            return backend

        from quantem.gpu.detector import prepare

        frame = cuda_data[key] if self.n_frames > 1 else cuda_data
        backend = prepare(frame)
        cache[key] = backend
        max_entries = max(1, int(getattr(self, "compare_max_panels", 1)) * 2)
        while len(cache) > max_entries:
            cache.popitem(last=False)
        return backend

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
        self._close_compute()
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

    def _compare_preview_contract(
        self,
        preset_name: str,
        *,
        mask=None,
    ) -> dict[str, Any]:
        """Scientific processing contract for one persistent preview entry."""
        preset = str(preset_name).lower()
        # Build standard masks on the host. Cache lookup must not allocate CUDA
        # memory merely to validate a derived preview.
        rows = np.arange(self.det_rows, dtype=np.float32)[:, None]
        cols = np.arange(self.det_cols, dtype=np.float32)[None, :]
        dist_sq = (cols - np.float32(self.center_col)) ** 2 + (
            rows - np.float32(self.center_row)
        ) ** 2
        bf = np.float32(max(1.0, float(self.bf_radius)))
        if preset == "bf":
            mask_array = dist_sq <= bf**2
        elif preset == "abf":
            mask_array = (dist_sq >= (bf * np.float32(0.5)) ** 2) & (
                dist_sq <= bf**2
            )
        elif preset == "adf":
            mask_array = (dist_sq >= bf**2) & (
                dist_sq <= (bf * np.float32(2.0)) ** 2
            )
        elif preset == "haadf":
            mask_array = (dist_sq >= (bf * np.float32(2.0)) ** 2) & (
                dist_sq <= (bf * np.float32(4.0)) ** 2
            )
        elif mask is not None:
            mask_array = (
                mask.detach().to("cpu").numpy()
                if hasattr(mask, "detach")
                else np.asarray(mask)
            )
        else:
            raise ValueError(f"unsupported persistent preview preset {preset_name!r}")
        mask_bytes = np.ascontiguousarray(mask_array.astype(np.uint8)).tobytes()
        return {
            "preset": preset,
            "shape": [int(self.shape_rows), int(self.shape_cols)],
            "detector_shape": [int(self.det_rows), int(self.det_cols)],
            "center": [float(self.center_row), float(self.center_col)],
            "bf_radius": float(self.bf_radius),
            "mask_sha256": hashlib.sha256(mask_bytes).hexdigest(),
            "normalization": "detector-mean-v1",
            "output_dtype": "<f4",
        }

    def _compare_preview_source(self, idx: int):
        sources = getattr(self, "_compare_preview_sources", None)
        if sources is None or int(idx) < 0 or int(idx) >= len(sources):
            return None
        return sources[int(idx)]

    def _compare_source_signature_token(
        self,
        indices: Sequence[int],
        signatures: dict[int, dict[str, Any]] | None = None,
    ) -> str | None:
        """Stable source-provenance token for one host-memory page entry."""
        cache = getattr(self, "_compare_preview_cache", None)
        sources = getattr(self, "_compare_preview_sources", None)
        if cache is None or sources is None:
            return None
        values: list[dict[str, Any]] = []
        try:
            for value in indices:
                idx = int(value)
                source = self._compare_preview_source(idx)
                if source is None:
                    return None
                signature = (signatures or {}).get(idx)
                if signature is None:
                    signature = cache.source_signature(source)
                values.append({"index": idx, "source": signature})
        except OSError:
            return "unavailable"
        return hashlib.sha256(
            json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _release_changed_compare_sources(self, indices: Sequence[int]) -> list[int]:
        """Drop resident raw/DP state when a known master or chunk changed."""
        cache = getattr(self, "_compare_preview_cache", None)
        data = getattr(self, "_data", None)
        signatures = getattr(self, "_compare_loaded_source_signatures", None)
        if (
            cache is None
            or signatures is None
            or data is None
            or not hasattr(data, "loaded_indices")
            or not hasattr(data, "release")
        ):
            return []
        loaded = set(int(idx) for idx in data.loaded_indices())
        changed: list[int] = []
        for value in indices:
            idx = int(value)
            if idx not in loaded:
                continue
            source = self._compare_preview_source(idx)
            previous = signatures.get(idx)
            if source is None or previous is None:
                continue
            try:
                current = cache.source_signature(source)
            except OSError:
                current = None
            if current == previous:
                continue
            data.release(idx)
            changed.append(idx)
            for key in list(self._compare_diffraction_cache):
                if int(key[0]) != idx:
                    continue
                payload = self._compare_diffraction_cache.pop(key)
                self._compare_diffraction_cache_bytes -= payload[1]
        return changed

    def _persistent_compare_preset_name(self) -> str | None:
        """Return a preset only when the current ROI is exactly canonical."""
        if (
            float(self.roi_center_col) != float(self.center_col)
            or float(self.roi_center_row) != float(self.center_row)
        ):
            return None
        bf = float(self.bf_radius)
        if self.roi_mode == "circle" and float(self.roi_radius) == bf:
            return "bf"
        if (
            self.roi_mode == "annular"
            and float(self.roi_radius_inner) == bf * 0.5
            and float(self.roi_radius) == bf
        ):
            return "abf"
        if (
            self.roi_mode == "annular"
            and float(self.roi_radius_inner) == bf
            and float(self.roi_radius) == bf * 2.0
        ):
            return "adf"
        if (
            self.roi_mode == "annular"
            and float(self.roi_radius_inner) == bf * 2.0
            and float(self.roi_radius) == bf * 4.0
        ):
            return "haadf"
        return None

    def _load_persistent_compare_previews(
        self,
        indices: Sequence[int],
        *,
        preset_name: str | None = None,
        mask=None,
        on_hit: Callable[[int, np.ndarray, int], bool | None] | None = None,
        source_signatures: dict[int, dict[str, Any]] | None = None,
        continue_if: Callable[[], bool] | None = None,
    ) -> dict[int, np.ndarray]:
        """Load individually validated disk previews for the current preset."""
        cache = getattr(self, "_compare_preview_cache", None)
        preset = (
            self._persistent_compare_preset_name()
            if preset_name is None
            else preset_name
        )
        if cache is None or not cache.enabled or preset is None:
            return {}
        contract = self._compare_preview_contract(preset, mask=mask)
        hits: dict[int, np.ndarray] = {}
        for value in indices:
            if continue_if is not None and not continue_if():
                break
            idx = int(value)
            source = self._compare_preview_source(idx)
            if source is None:
                continue
            image = cache.load(
                source,
                preset,
                contract,
                source_signature=(source_signatures or {}).get(idx),
            )
            if image is not None:
                hits[idx] = np.ascontiguousarray(image, dtype=np.float32)
                if callable(on_hit) and on_hit(idx, hits[idx], len(hits)) is False:
                    break
        return hits

    def _persist_compare_previews(
        self,
        payload: bytes,
        indices: Sequence[int],
        *,
        preset_name: str | None = None,
        source_signatures: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        """Queue per-master standard previews without delaying panel paint."""
        cache = getattr(self, "_compare_preview_cache", None)
        preset = (
            self._persistent_compare_preset_name()
            if preset_name is None
            else preset_name
        )
        if cache is None or not cache.enabled or preset is None or not indices:
            return
        try:
            stack = np.frombuffer(payload, dtype=np.float32).reshape(
                len(indices),
                self.shape_rows,
                self.shape_cols,
            )
            contract = self._compare_preview_contract(str(preset))
        except (TypeError, ValueError):
            return
        for value, image in zip(indices, stack, strict=True):
            idx = int(value)
            source = self._compare_preview_source(idx)
            if source is None:
                continue
            cache.store(
                source,
                str(preset),
                contract,
                image,
                source_signature=(source_signatures or {}).get(idx),
            )

    def _uses_progressive_compare_pages(self) -> bool:
        """Whether live page/config changes should use the folder scheduler."""
        data = getattr(self, "_data", None)
        return bool(
            (
                getattr(self, "_folder_source", None) is not None
                or bool(
                    getattr(
                        getattr(self, "_compare_preview_cache", None),
                        "enabled",
                        False,
                    )
                )
            )
            and type(data).__name__ == "Dataset5dstem"
            and getattr(data, "is_lazy", False)
            and self._multiple_view_active()
        )

    def _compare_page_request_is_current(
        self,
        generation: int,
        stop: threading.Event,
    ) -> bool:
        """Return False once a newer page/config request supersedes this worker."""
        return bool(
            not stop.is_set()
            and getattr(self, "_data", None) is not None
            and int(generation) == int(self._compare_page_generation_counter)
            and getattr(self, "_compare_page_stop", None) is stop
        )

    def _send_compare_page_message(
        self,
        content: dict[str, Any],
        *,
        buffer: bytes | memoryview | None = None,
    ) -> None:
        """Send progressive-page messages without making comm loss fatal."""
        try:
            self.send(content, buffers=[] if buffer is None else [buffer])
            self._compare_page_last_send_error = ""
        except Exception as exc:
            # A notebook may close its comm while a daemon worker is finishing.
            # The final synced payload remains authoritative when a comm exists.
            self._compare_page_last_send_error = (
                f"{type(exc).__name__}: {str(exc)[:1800]}"
            )

    def _progressive_compare_batches(
        self,
        indices: Sequence[int],
    ) -> list[list[int]]:
        """Memory/device-aware waves, with safe one-frame fallback."""
        requested = [int(idx) for idx in indices]
        data = getattr(self, "_data", None)
        builder = getattr(data, "progressive_batches", None)
        if not callable(builder):
            return [[idx] for idx in requested]
        try:
            proposed = list(builder(requested))
        except Exception:
            return [[idx] for idx in requested]

        requested_set = set(requested)
        seen: set[int] = set()
        waves: list[list[int]] = []
        for proposed_wave in proposed:
            try:
                values = list(proposed_wave)
            except TypeError:
                values = [proposed_wave]
            wave: list[int] = []
            for value in values:
                if isinstance(value, bool):
                    continue
                idx = int(value)
                if idx in requested_set and idx not in seen:
                    wave.append(idx)
                    seen.add(idx)
            if wave:
                waves.append(wave)
        waves.extend([[idx] for idx in requested if idx not in seen])
        return waves

    def _emit_progressive_compare_panel(
        self,
        *,
        generation: int,
        stop: threading.Event,
        page_idx: int,
        frame_idx: int,
        slot: int,
        image: np.ndarray,
        started: float,
        loaded_count: int,
        cached: bool = False,
    ) -> float | None:
        """Publish one ready panel when its request is still current."""
        panel = np.ascontiguousarray(image, dtype=np.float32).reshape(
            self.shape_rows,
            self.shape_cols,
        )
        elapsed_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        with self._compare_page_request_lock:
            if not self._compare_page_request_is_current(generation, stop):
                return None
            fresh_indices = self._compare_page_fresh_indices
            if cached and int(frame_idx) in fresh_indices:
                # A slow cache read must never replace a current-generation
                # raw result for the same stable slot.
                return elapsed_ms
            if not cached:
                fresh_indices.add(int(frame_idx))
            self._compare_page_working_images[int(frame_idx)] = panel
            if float(self.compare_page_first_panel_ms) <= 0:
                self.compare_page_first_panel_ms = elapsed_ms
            if not cached and float(self.compare_page_first_fresh_ms) <= 0:
                self.compare_page_first_fresh_ms = elapsed_ms
            cached_indices = set(int(idx) for idx in self.compare_page_cached_indices)
            if cached:
                cached_indices.add(int(frame_idx))
            else:
                cached_indices.discard(int(frame_idx))
            with self.hold_trait_notifications():
                if not cached:
                    self.compare_page_loaded_count = int(loaded_count)
                self.compare_page_cached_indices = sorted(cached_indices)
                self.compare_page_panel_bytes = panel.tobytes()
                self.compare_page_panel_frame_idx = int(frame_idx)
                self.compare_page_panel_slot = int(slot)
                self.compare_page_panel_cached = bool(cached)
                self.compare_page_panel_sequence = (
                    int(self.compare_page_panel_sequence) + 1
                )
            self._send_compare_page_message(
                {
                    "type": "compare_panel",
                    "generation": int(generation),
                    "frame_idx": int(frame_idx),
                    "slot": int(slot),
                    "page_idx": int(page_idx),
                    # Compatibility aliases keep the protocol self-describing
                    # outside the widget frontend.
                    "index": int(frame_idx),
                    "position": int(slot),
                    "page": int(page_idx),
                    "elapsed_ms": float(elapsed_ms),
                    "cached": bool(cached),
                },
                # ipywidgets' binary comm path expects an explicit byte view;
                # plain ``bytes`` can arrive as an empty/missing buffer in a
                # live anywidget custom-message callback.
                buffer=memoryview(panel).cast("B"),
            )
        return elapsed_ms

    def _publish_persistent_compare_previews(
        self,
        images: dict[int, np.ndarray],
        *,
        generation: int,
        stop: threading.Event,
        page_idx: int,
        indices: tuple[int, ...],
        started: float,
        emit: bool = True,
    ) -> bool:
        """Paint validated disk previews while authoritative CUDA work runs."""
        shown = [idx for idx in indices if idx in images]
        if not shown:
            return False
        if emit:
            for idx in shown:
                if not self._compare_page_request_is_current(generation, stop):
                    return False
                self._emit_progressive_compare_panel(
                    generation=generation,
                    stop=stop,
                    page_idx=page_idx,
                    frame_idx=idx,
                    slot=indices.index(idx),
                    image=images[idx],
                    started=started,
                    loaded_count=len(shown),
                    cached=True,
                )
        with self._compare_page_request_lock:
            if not self._compare_page_request_is_current(generation, stop):
                return False
            if not bool(self.compare_page_loading):
                return False
            merged = {
                idx: self._compare_page_working_images[idx]
                for idx in indices
                if idx in self._compare_page_working_images
            }
            shown = [idx for idx in indices if idx in merged]
            if not shown:
                return False
            cached_now = sorted(
                int(idx) for idx in self.compare_page_cached_indices if idx in merged
            )
            stack = np.ascontiguousarray(
                np.stack([merged[idx] for idx in shown], axis=0),
                dtype=np.float32,
            )
            state = (
                "cached"
                if len(cached_now) == len(indices)
                else "partial"
                if cached_now
                else "miss"
            )
            with self.hold_trait_notifications():
                # This is a valid durable fallback. Fresh panels later replace
                # the same stable frontend slots and the final full stack.
                self.compare_virtual_image_bytes = stack.tobytes()
                self.compare_panel_count = len(shown)
                self.compare_panel_indices = shown
                self.compare_page_cached_indices = cached_now
                self.compare_page_cache_state = state
                self.compare_status = (
                    f"Cached preview · refreshing current data "
                    f"{len(cached_now)}/{len(indices)}"
                    if cached_now
                    else f"Refreshing current data {len(shown)}/{len(indices)}"
                )
        return bool(cached_now)

    def _load_progressive_compare_page(
        self,
        indices: Sequence[int],
        mask,
        *,
        generation: int,
        stop: threading.Event,
        page_idx: int,
        started: float,
        emit: bool,
    ) -> dict[int, np.ndarray] | None:
        """Reduce a page in waves; optionally stream each completed panel."""
        requested = [int(idx) for idx in indices]
        slot_for = {idx: slot for slot, idx in enumerate(requested)}
        images_by_index: dict[int, np.ndarray] = {}
        self._release_changed_compare_sources(requested)

        cached = self._get_cached_compare_preset(requested)
        if cached is not None:
            payload, cached_indices, _ = cached
            stack = np.frombuffer(payload, dtype=np.float32).reshape(
                len(cached_indices),
                self.shape_rows,
                self.shape_cols,
            )
            for frame_idx, image in zip(cached_indices, stack, strict=True):
                if not self._compare_page_request_is_current(generation, stop):
                    return None
                idx = int(frame_idx)
                panel = np.ascontiguousarray(image, dtype=np.float32)
                images_by_index[idx] = panel
                if emit:
                    self._emit_progressive_compare_panel(
                        generation=generation,
                        stop=stop,
                        page_idx=page_idx,
                        frame_idx=idx,
                        slot=slot_for[idx],
                        image=panel,
                        started=started,
                        loaded_count=len(images_by_index),
                    )
            return images_by_index

        for wave in self._progressive_compare_batches(requested):
            if not self._compare_page_request_is_current(generation, stop):
                return None
            source_signatures: dict[int, dict[str, Any]] = {}
            preview_cache = getattr(self, "_compare_preview_cache", None)
            if (
                preview_cache is not None
                and preview_cache.enabled
                and self._persistent_compare_preset_name() is not None
            ):
                for idx in wave:
                    source = self._compare_preview_source(idx)
                    if source is None:
                        continue
                    try:
                        source_signatures[int(idx)] = preview_cache.source_signature(
                            source
                        )
                    except OSError:
                        continue
            pending_cache_payload: tuple[bytes, tuple[int, ...]] | None = None
            with self._compare_compute_lock:
                if not self._compare_page_request_is_current(generation, stop):
                    return None
                cached_wave = self._get_cached_compare_preset(wave)
                if cached_wave is not None:
                    payload, cached_indices, _ = cached_wave
                    stack = np.frombuffer(payload, dtype=np.float32).reshape(
                        len(cached_indices),
                        self.shape_rows,
                        self.shape_cols,
                    )
                    wave_indices = [int(idx) for idx in cached_indices]
                    wave_images = [
                        np.ascontiguousarray(image, dtype=np.float32)
                        for image in stack
                    ]
                else:
                    self._preload_compare_page(wave)
                    # Preserve each wave's tiny current-position diffraction
                    # pattern before a later wave evicts its raw 4D frame. The
                    # page-average DP can then finish without a second cold read.
                    for idx in wave:
                        if self._get_cached_compare_diffraction_frame(idx) is None:
                            try:
                                self._store_compare_diffraction_cache(
                                    idx,
                                    self._frame_data_for_index(idx),
                                )
                            except BaseException:
                                pass
                    wave_images = self._compare_virtual_images_for_indices(wave, mask)
                    wave_indices = [int(idx) for idx in wave[: len(wave_images)]]
                    if wave_images and self._compare_page_request_is_current(
                        generation, stop
                    ):
                        wave_stack = np.ascontiguousarray(
                            np.stack(wave_images, axis=0),
                            dtype=np.float32,
                        )
                        pending_cache_payload = (
                            wave_stack.tobytes(),
                            tuple(wave_indices),
                        )
            for frame_idx, image in zip(wave_indices, wave_images, strict=True):
                if not self._compare_page_request_is_current(generation, stop):
                    return None
                panel = np.ascontiguousarray(image, dtype=np.float32).reshape(
                    self.shape_rows,
                    self.shape_cols,
                )
                images_by_index[frame_idx] = panel
                if emit:
                    self._emit_progressive_compare_panel(
                        generation=generation,
                        stop=stop,
                        page_idx=page_idx,
                        frame_idx=frame_idx,
                        slot=slot_for[frame_idx],
                        image=panel,
                        started=started,
                        loaded_count=len(images_by_index),
                        cached=False,
                    )
            if (
                pending_cache_payload is not None
                and self._compare_page_request_is_current(generation, stop)
            ):
                self._store_cached_compare_preset(
                    pending_cache_payload[0],
                    pending_cache_payload[1],
                    "cached",
                    source_signatures=source_signatures,
                )
        return images_by_index

    def _prefetch_neighbor_compare_pages(
        self,
        *,
        generation: int,
        stop: threading.Event,
        page_idx: int,
        mask,
    ) -> None:
        """Warm next, then previous, current-preset pages without UI messages."""
        if (
            self._normalise_compare_group_mode(self.compare_group_mode) != "paged"
            or self._preset_cache_name() is None
            or self._compare_cache_pages <= 0
        ):
            return
        ordered = self._compare_ordered_ready_indices()
        hidden = self._compare_hidden_panel_set()
        page_size = max(1, int(self.compare_max_panels))
        pages = [
            [idx for idx in ordered[start : start + page_size] if idx not in hidden]
            for start in range(0, len(ordered), page_size)
        ]
        neighbors: list[tuple[int, list[int]]] = []
        if page_idx + 1 < len(pages) and pages[page_idx + 1]:
            neighbors.append((page_idx + 1, pages[page_idx + 1]))
        if page_idx - 1 >= 0 and pages[page_idx - 1]:
            neighbors.append((page_idx - 1, pages[page_idx - 1]))

        for neighbor_idx, indices in neighbors:
            if not self._compare_page_request_is_current(generation, stop):
                return
            if self._get_cached_compare_preset(indices) is not None:
                continue
            images = self._load_progressive_compare_page(
                indices,
                mask,
                generation=generation,
                stop=stop,
                page_idx=neighbor_idx,
                started=time.perf_counter(),
                emit=False,
            )
            if images is None or not self._compare_page_request_is_current(
                generation, stop
            ):
                return
            shown = [idx for idx in indices if idx in images]
            if len(shown) != len(indices):
                continue
            stack = np.ascontiguousarray(
                np.stack([images[idx] for idx in shown], axis=0),
                dtype=np.float32,
            )
            self._store_cached_compare_preset(
                stack.tobytes(),
                tuple(shown),
                "cached",
                source_signatures={
                    idx: self._compare_loaded_source_signatures[idx]
                    for idx in shown
                    if idx in self._compare_loaded_source_signatures
                },
                persist=False,
            )

    def _publish_progressive_compare_page(
        self,
        images: dict[int, np.ndarray],
        *,
        generation: int,
        stop: threading.Event,
        page_idx: int,
        indices: tuple[int, ...],
        all_groups: bool,
        started: float,
    ) -> bool:
        """Atomically publish the durable stack and page-complete message."""
        shown = [idx for idx in indices if idx in images]
        if not shown:
            raise MemoryError("compare page produced no panels")
        partial_reason = "memory limited" if len(shown) < len(indices) else None
        stack = np.ascontiguousarray(
            np.stack([images[idx] for idx in shown], axis=0),
            dtype=np.float32,
        )
        status = self._compare_status_for_indices(
            shown,
            all_groups=all_groups,
            partial_reason=partial_reason,
        )
        total_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        first_panel_ms = float(self.compare_page_first_panel_ms)
        payload = stack.tobytes()
        # Scheduling and final publication share this lock, making the
        # generation check plus writes atomic with respect to a new page.
        with self._compare_page_request_lock:
            if not self._compare_page_request_is_current(generation, stop):
                return False
            if partial_reason is not None:
                self._set_gpu_memory_warning(
                    action="show the requested multiple-panel page",
                    requested=len(indices),
                    shown=len(shown),
                )
            else:
                self._clear_gpu_memory_warning()
            self._store_cached_compare_preset(
                payload,
                tuple(shown),
                status,
                source_signatures={
                    idx: self._compare_loaded_source_signatures[idx]
                    for idx in shown
                    if idx in self._compare_loaded_source_signatures
                },
                persist=False,
            )
            with self.hold_trait_notifications():
                self.compare_virtual_image_bytes = payload
                self.compare_panel_count = len(shown)
                self.compare_panel_indices = shown
                self.compare_status = status
                self.compare_page_loading = False
                self.compare_page_loaded_count = len(shown)
                self.compare_page_cached_indices = []
                self.compare_page_cache_state = "fresh"
                self.compare_page_total_ms = total_ms
            self._send_compare_page_message(
                {
                    "type": "compare_page_complete",
                    "generation": int(generation),
                    "page_idx": int(page_idx),
                    "loaded_count": len(shown),
                    "status": status,
                    "first_panel_ms": first_panel_ms,
                    "total_ms": total_ms,
                    "page": int(page_idx),
                    "indices": shown,
                    "cached_indices": [],
                    "cache_state": "fresh",
                    "elapsed_ms": total_ms,
                }
            )
        self._finish_folder_page_update(generation)
        return True

    def _resume_folder_compare_maintenance(
        self,
        *,
        generation: int,
        stop: threading.Event,
    ) -> None:
        """Resume configured folder maintenance sequentially after prefetch."""
        source = getattr(self, "_folder_source", None) or {}
        preload = bool(source.get("preload_all_if_fits", False))
        warm = bool(source.get("warm_cache", False))
        if not preload and not warm:
            return

        # A prior generation may still be between starting and joining its
        # managed preload/warmer. Retire that wrapper before installing a new
        # one so close() and the next foreground page have one joinable owner.
        self.stop_compare_maintenance(wait=True)
        maintenance_stop = threading.Event()

        def active() -> bool:
            return bool(
                not stop.is_set()
                and not maintenance_stop.is_set()
                and getattr(self, "_data", None) is not None
                and int(generation) == int(self._compare_page_generation_counter)
            )

        def worker() -> None:
            try:
                if preload and active():
                    self.preload_all_datasets(background=True)
                    self.wait_for_dataset_preload()
                if warm and active():
                    # The preload above is fully joined before cache warming
                    # starts; a new foreground page stops/joins this warmer.
                    self.warm_compare_cache(background=True)
            finally:
                current = threading.current_thread()
                with self._compare_page_request_lock:
                    if self._compare_maintenance_thread is current:
                        self._compare_maintenance_thread = None
                    if self._compare_maintenance_stop is maintenance_stop:
                        self._compare_maintenance_stop = None

        thread = threading.Thread(
            target=worker,
            name=f"Show4DSTEM-maintenance-{generation}",
            daemon=True,
        )
        with self._compare_page_request_lock:
            if not self._compare_page_request_is_current(generation, stop):
                return
            self._compare_maintenance_stop = maintenance_stop
            self._compare_maintenance_thread = thread
            thread.start()

    def stop_compare_maintenance(self, *, wait: bool = False) -> Self:
        """Stop and optionally join folder preload/cache maintenance."""
        with self._compare_page_request_lock:
            stop = self._compare_maintenance_stop
            thread = self._compare_maintenance_thread
            if stop is not None:
                stop.set()

        # The wrapper can be blocked waiting for either managed child. Stop the
        # children before joining their owner, otherwise close() can deadlock.
        self.stop_dataset_preload(wait=wait)
        self.stop_compare_cache_warm(wait=wait)
        if wait and thread is not None and thread is not threading.current_thread():
            thread.join()
        with self._compare_page_request_lock:
            if self._compare_maintenance_stop is stop:
                self._compare_maintenance_stop = None
            if (
                self._compare_maintenance_thread is thread
                and (thread is None or not thread.is_alive())
            ):
                self._compare_maintenance_thread = None
        return self

    def _run_progressive_compare_page(
        self,
        *,
        generation: int,
        stop: threading.Event,
        page_idx: int,
        indices: tuple[int, ...],
        all_groups: bool,
        started: float,
    ) -> None:
        """Daemon-worker body for one cancellable visible-page request."""
        visible_complete = False
        cache_thread: threading.Thread | None = None

        def join_cache_lookup() -> None:
            if (
                cache_thread is not None
                and cache_thread is not threading.current_thread()
            ):
                cache_thread.join()

        try:
            # A fully reduced host-cached page has no GPU dependency. Publish it
            # before waiting for an unrelated preload/warmer to stop so warm
            # page navigation stays effectively immediate.
            cached_visible = self._get_cached_compare_preset(indices) is not None
            preview_cache = getattr(self, "_compare_preview_cache", None)
            if (
                not cached_visible
                and preview_cache is not None
                and preview_cache.enabled
                and self._persistent_compare_preset_name() is not None
            ):
                # Disk validation/read and authoritative raw refresh run in
                # parallel. Each validated hit paints as soon as it is read;
                # fresh-generation emissions win if both finish the same slot.
                def load_disk_previews() -> None:
                    def emit_disk_hit(
                        idx: int,
                        image: np.ndarray,
                        count: int,
                    ) -> bool:
                        return (
                            self._emit_progressive_compare_panel(
                                generation=generation,
                                stop=stop,
                                page_idx=page_idx,
                                frame_idx=idx,
                                slot=indices.index(idx),
                                image=image,
                                started=started,
                                loaded_count=count,
                                cached=True,
                            )
                            is not None
                        )

                    try:
                        persistent_images = self._load_persistent_compare_previews(
                            indices,
                            on_hit=emit_disk_hit,
                            continue_if=lambda: self._compare_page_request_is_current(
                                generation,
                                stop,
                            ),
                        )
                        if persistent_images:
                            self._publish_persistent_compare_previews(
                                persistent_images,
                                generation=generation,
                                stop=stop,
                                page_idx=page_idx,
                                indices=indices,
                                started=started,
                                emit=False,
                            )
                    except BaseException:
                        return

                cache_thread = threading.Thread(
                    target=load_disk_previews,
                    name=f"Show4DSTEM-preview-read-{generation}",
                    daemon=True,
                )
                cache_thread.start()
            if cached_visible:
                images = self._load_progressive_compare_page(
                    indices,
                    None,
                    generation=generation,
                    stop=stop,
                    page_idx=page_idx,
                    started=started,
                    emit=True,
                )
                if images is None or not self._publish_progressive_compare_page(
                    images,
                    generation=generation,
                    stop=stop,
                    page_idx=page_idx,
                    indices=indices,
                    all_groups=all_groups,
                    started=started,
                ):
                    return
                visible_complete = True

            # Only one page worker may prepare/use GPU state at a time. A newer
            # request still returns immediately; it sets this worker's stop flag
            # and waits here until the current wave reaches a cancellation point.
            with self._compare_page_worker_lock:
                if not self._compare_page_request_is_current(generation, stop):
                    return
                # Full-series preload and cache warming also read/decode into the
                # same CUDA contexts. Stop and join both before foreground access
                # to avoid allocator/decompressor races near VRAM capacity.
                self.stop_compare_maintenance(wait=True)
                self.stop_dataset_preload(wait=True)
                self.stop_compare_cache_warm(wait=True)
                if not self._compare_page_request_is_current(generation, stop):
                    return

                loaded_before = None
                data = getattr(self, "_data", None)
                if hasattr(data, "loaded_indices"):
                    loaded_before = tuple(data.loaded_indices())
                mask = self._current_detector_mask()
                if not cached_visible:
                    images = self._load_progressive_compare_page(
                        indices,
                        mask,
                        generation=generation,
                        stop=stop,
                        page_idx=page_idx,
                        started=started,
                        emit=True,
                    )
                    if images is None or not self._publish_progressive_compare_page(
                        images,
                        generation=generation,
                        stop=stop,
                        page_idx=page_idx,
                        indices=indices,
                        all_groups=all_groups,
                        started=started,
                    ):
                        return
                    visible_complete = True
                    # Visible traits/canvases are already complete. Join the
                    # bounded lookup before secondary DP/neighbor prefetch so
                    # background cache validation does not contend with them.
                    join_cache_lookup()

                # The VI page is already complete and timed. Update the shared
                # DP as secondary state; wave-time DP caching normally makes this
                # host-only and it can never overwrite a newer generation.
                frame_payload = None
                try:
                    with self._compare_compute_lock:
                        if self._compare_page_request_is_current(generation, stop):
                            frame_payload = self._current_frame_bytes()
                except BaseException:
                    pass
                if frame_payload is not None:
                    with self._compare_page_request_lock:
                        if self._compare_page_request_is_current(generation, stop):
                            self.frame_bytes = frame_payload
                if loaded_before is not None and (
                    tuple(data.loaded_indices()) != loaded_before
                ):
                    with self._compare_page_request_lock:
                        if self._compare_page_request_is_current(generation, stop):
                            self._update_gpu_memory_status()

                # Visible work is complete. Use remaining idle time to reduce the
                # most likely navigation targets; cancellation remains checked at
                # every wave and these background results never emit UI messages.
                try:
                    self._prefetch_neighbor_compare_pages(
                        generation=generation,
                        stop=stop,
                        page_idx=page_idx,
                        mask=mask,
                    )
                except BaseException:
                    pass
            if self._compare_page_request_is_current(generation, stop):
                self._resume_folder_compare_maintenance(
                    generation=generation,
                    stop=stop,
                )
        except BaseException as exc:
            join_cache_lookup()
            if visible_complete:
                return
            if not self._compare_page_request_is_current(generation, stop):
                return
            total_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
            with self._compare_page_request_lock:
                if not self._compare_page_request_is_current(generation, stop):
                    return
                self._compare_page_last_error = (
                    f"{type(exc).__name__}: {str(exc)[:1800]}"
                )
                if _is_recoverable_allocation_error(exc):
                    self._reclaim_compare_allocation_failure()
                    self._set_gpu_memory_warning(
                        action="compute the multiple-panel page",
                        requested=len(indices),
                        shown=int(self.compare_page_loaded_count),
                    )
                    status = (
                        "GPU memory limited: multiple grid page was not computed."
                    )
                else:
                    status = "Multiple grid unavailable: page could not be loaded."
                shown = [
                    idx
                    for idx in indices
                    if idx in self._compare_page_working_images
                ]
                merged_payload = (
                    np.ascontiguousarray(
                        np.stack(
                            [self._compare_page_working_images[idx] for idx in shown],
                            axis=0,
                        ),
                        dtype=np.float32,
                    ).tobytes()
                    if shown
                    else b""
                )
                has_cached_fallback = bool(self.compare_page_cached_indices)
                with self.hold_trait_notifications():
                    if shown:
                        # Trait fallback/reconnect receives the same merged
                        # cached+fresh pixels that were last painted, never the
                        # original stale disk-only stack with fresh labels.
                        self.compare_virtual_image_bytes = merged_payload
                        self.compare_panel_count = len(shown)
                        self.compare_panel_indices = shown
                    self.compare_page_loading = False
                    self.compare_page_total_ms = total_ms
                    if has_cached_fallback:
                        self.compare_page_cache_state = "warning"
                        self.compare_status = (
                            "Cached preview · refresh failed; retained validated "
                            "derived images. Retry when the source/GPU is available."
                        )
                    elif shown:
                        self.compare_page_cache_state = "warning"
                        self.compare_status = (
                            "Partial current page · refresh failed; retry when "
                            "the source/GPU is available."
                        )
                    else:
                        self.compare_page_cache_state = "miss"
                        self.compare_status = status
                self._send_compare_page_message(
                    {
                        "type": "compare_page_complete",
                        "generation": int(generation),
                        "page_idx": int(page_idx),
                        "loaded_count": int(self.compare_page_loaded_count),
                        "status": self.compare_status,
                        "first_panel_ms": float(self.compare_page_first_panel_ms),
                        "total_ms": total_ms,
                        "page": int(page_idx),
                        "indices": [],
                        "cached_indices": list(self.compare_page_cached_indices),
                        "cache_state": str(self.compare_page_cache_state),
                        "elapsed_ms": total_ms,
                    }
                )
            self._finish_folder_page_update(
                generation,
                error=(
                    "The newly arrived data could not refresh the visible page. "
                    "A validated cached preview was retained; check the source "
                    "files and GPU memory, then retry."
                    if has_cached_fallback
                    else "The newly arrived data could not load on the visible "
                    "page; check the source files and GPU memory, then retry."
                ),
            )
        finally:
            join_cache_lookup()
            current = threading.current_thread()
            if getattr(self, "_compare_page_thread", None) is current:
                self._compare_page_thread = None

    def _schedule_progressive_compare_page(self, indices: Sequence[int]) -> None:
        """Cancel the prior request and start a daemon worker for this page."""
        requested = tuple(int(idx) for idx in indices)
        page_idx = int(self.compare_page_idx)
        all_groups = (
            self._normalise_compare_group_mode(self.compare_group_mode) == "all"
        )
        started = time.perf_counter()
        with self._compare_page_request_lock:
            prior = getattr(self, "_compare_page_stop", None)
            if prior is not None:
                prior.set()
            self._compare_page_generation_counter += 1
            generation = int(self._compare_page_generation_counter)
            if bool(getattr(self, "_folder_update_pending", False)):
                # A page/config change supersedes the generation that first
                # carried the watched append. Retarget the proof to the new
                # visible page and cancel the old generation's deadline.
                self._cancel_folder_page_paint_timeout()
                self._folder_update_generation = generation
                self._folder_update_page_idx = page_idx
                self._folder_update_expected_indices = requested
                self._folder_update_backend_complete_generation = 0
                self._folder_update_painted_generation = 0
                self._folder_update_painted_page_idx = -1
            stop = threading.Event()
            self._compare_page_stop = stop
            self._compare_page_last_error = ""
            self._compare_page_last_send_error = ""
            self._compare_page_fresh_indices = set()
            self._compare_page_working_images = {}
            with self.hold_trait_notifications():
                self.compare_page_expected_indices = list(requested)
                self.compare_page_loading = True
                self.compare_page_loaded_count = 0
                self.compare_page_cached_indices = []
                self.compare_page_cache_state = (
                    "miss"
                    if getattr(self, "_compare_preview_cache", None) is not None
                    else "off"
                )
                self.compare_page_generation = generation
                self.compare_page_first_panel_ms = 0.0
                self.compare_page_first_fresh_ms = 0.0
                self.compare_page_total_ms = 0.0
                self.compare_page_panel_bytes = b""
                self.compare_page_panel_frame_idx = -1
                self.compare_page_panel_slot = -1
                self.compare_page_panel_cached = False
                self.compare_status = (
                    f"Loading {len(requested)} {self.frame_dim_label.lower()} panels"
                )
            self._send_compare_page_message(
                {
                    "type": "compare_page_start",
                    "generation": generation,
                    "page_idx": page_idx,
                    "indices": list(requested),
                    "panel_count": len(requested),
                    "shape_rows": int(self.shape_rows),
                    "shape_cols": int(self.shape_cols),
                    "page": page_idx,
                    "index": None,
                    "cached_indices": [],
                    "cache_state": str(self.compare_page_cache_state),
                    "elapsed_ms": 0.0,
                }
            )
            thread = threading.Thread(
                target=self._run_progressive_compare_page,
                kwargs={
                    "generation": generation,
                    "stop": stop,
                    "page_idx": page_idx,
                    "indices": requested,
                    "all_groups": all_groups,
                    "started": started,
                },
                name=f"Show4DSTEM-page-{generation}",
                daemon=True,
            )
            self._compare_page_thread = thread
            thread.start()

    def stop_compare_page_load(self, *, wait: bool = False) -> Self:
        """Cancel progressive visible/prefetch work after its current GPU wave."""
        canceled_folder_update = False
        with self._compare_page_request_lock:
            stop = getattr(self, "_compare_page_stop", None)
            thread = getattr(self, "_compare_page_thread", None)
            if stop is not None:
                stop.set()
            if stop is not None or (thread is not None and thread.is_alive()):
                self._compare_page_generation_counter += 1
                self.compare_page_generation = int(
                    self._compare_page_generation_counter
                )
                self.compare_page_loading = False
                if bool(getattr(self, "_folder_update_pending", False)):
                    self._reset_folder_page_update_tracking()
                    canceled_folder_update = True
            if getattr(self, "_compare_page_stop", None) is stop:
                self._compare_page_stop = None
        if wait and thread is not None and thread is not threading.current_thread():
            thread.join()
        self.stop_compare_maintenance(wait=wait)
        if thread is None or not thread.is_alive():
            if getattr(self, "_compare_page_thread", None) is thread:
                self._compare_page_thread = None
        if canceled_folder_update:
            self._finish_folder_poll_status(
                waiting=(
                    [str(self._folder_poll_waiting)]
                    if getattr(self, "_folder_poll_waiting", "")
                    else []
                ),
                errors=(
                    [str(self._folder_poll_error)]
                    if getattr(self, "_folder_poll_error", "")
                    else []
                ),
            )
        return self

    def wait_for_compare_page(self, timeout: float | None = None) -> Self:
        """Wait for visible completion and neighbor prefetch, for verification."""
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        while True:
            thread = getattr(self, "_compare_page_thread", None)
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

    def _refresh_compare_virtual_images(self) -> None:
        """Refresh compare data synchronously or stream a lazy folder page."""
        if self._uses_progressive_compare_pages():
            indices = self._compare_ready_indices()
            if not indices:
                self.stop_compare_page_load()
                self._clear_compare_virtual_images(self._empty_compare_page_status())
                return
            if getattr(self, "_suppress_compare_recompute", False):
                return
            self._schedule_progressive_compare_page(indices)
            return
        # Leaving multiple mode must invalidate a worker before clearing the
        # durable payload, otherwise a late generation could repaint the grid.
        if getattr(self, "_compare_page_stop", None) is not None:
            self.stop_compare_page_load()
        self._refresh_compare_virtual_images_sync()

    def _refresh_compare_virtual_images_sync(self) -> None:
        """Build the lightweight virtual-image stack used by compare mode."""
        if getattr(self, "_data", None) is None:
            self._clear_compare_virtual_images()
            return
        if not self._multiple_view_active():
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
            if self.vi_source != "roi":
                if self._normalise_vi_source(self.vi_source) not in getattr(
                    self, "_vi_product_maps", {}
                ):
                    self._clear_compare_virtual_images()
                    return
                shown_indices = [int(idx) for idx in indices]
                with self.hold_trait_notifications():
                    self.compare_panel_count = len(shown_indices)
                    self.compare_panel_indices = shown_indices
                    self.compare_status = self._compare_status_for_indices(
                        shown_indices,
                        all_groups=(
                            self._normalise_compare_group_mode(self.compare_group_mode)
                            == "all"
                        ),
                    )
                self._clear_gpu_memory_warning()
                return
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
        while True:
            with self._compare_cache_lock:
                cached = self._compare_virtual_page_cache.get(key)
                if cached is None:
                    return None
            payload, cached_indices, status, _, source_token = cached
            current_token = (
                self._compare_source_signature_token(cached_indices)
                if source_token is not None
                else None
            )
            with self._compare_cache_lock:
                current = self._compare_virtual_page_cache.get(key)
                if current is not cached:
                    # A background refresh published a newer value while the
                    # source signature was checked. Validate that value instead
                    # of exposing a transient cache miss to the visible page.
                    continue
                if source_token is not None and current_token != source_token:
                    self._compare_virtual_page_cache.pop(key, None)
                    self._compare_virtual_page_cache_bytes -= cached[3]
                    return None
                self._compare_virtual_page_cache.move_to_end(key)
                return payload, cached_indices, status

    def _store_cached_compare_preset(
        self,
        payload: bytes,
        indices: tuple[int, ...],
        status: str,
        *,
        preset_name: str | None = None,
        source_signatures: dict[int, dict[str, Any]] | None = None,
        persist: bool = True,
    ) -> None:
        key = self._compare_preset_cache_key(indices, preset_name=preset_name)
        if key is not None:
            source_token = self._compare_source_signature_token(
                indices,
                source_signatures,
            )
            with self._compare_cache_lock:
                existing = self._compare_virtual_page_cache.pop(key, None)
                if existing is not None:
                    self._compare_virtual_page_cache_bytes -= existing[3]
                nbytes = len(payload)
                self._compare_virtual_page_cache[key] = (
                    payload,
                    indices,
                    status,
                    nbytes,
                    source_token,
                )
                self._compare_virtual_page_cache_bytes += nbytes
                self._trim_compare_virtual_page_cache()
        if persist:
            self._persist_compare_previews(
                payload,
                indices,
                preset_name=preset_name,
                source_signatures=source_signatures,
            )

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
        preview_cache = getattr(self, "_compare_preview_cache", None)
        persistent_enabled = bool(
            preview_cache is not None and preview_cache.enabled
        )
        host_enabled = bool(
            self._compare_cache_pages > 0 and self._compare_cache_max_bytes != 0
        )
        if not names or (not host_enabled and not persistent_enabled):
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
        if host_enabled:
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
                    page_source_signatures: dict[int, dict[str, Any]] = {}
                    if persistent_enabled:
                        for idx in page:
                            source = self._compare_preview_source(idx)
                            if source is None:
                                continue
                            try:
                                page_source_signatures[int(idx)] = (
                                    preview_cache.source_signature(source)
                                )
                            except OSError:
                                continue
                    disk_complete: set[str] = set()
                    for name in names:
                        if self._get_cached_compare_preset(
                            page,
                            preset_name=name,
                        ) is not None:
                            continue
                        disk_images = self._load_persistent_compare_previews(
                            page,
                            preset_name=name,
                            mask=masks[name],
                            source_signatures=page_source_signatures,
                        )
                        if len(disk_images) != len(page):
                            continue
                        disk_complete.add(name)
                    missing = [
                        name
                        for name in names
                        if name not in disk_complete
                        and self._get_cached_compare_preset(
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
                            source_signatures=page_source_signatures,
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
        if type(self._data).__name__ == "Dataset5dstem" and hasattr(self._data, "frame"):
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
        cuda_data = getattr(self, "_cuda_compute_data", None)
        if cuda_data is not None:
            mask_np = (
                mask.detach().cpu().numpy()
                if hasattr(mask, "detach")
                else np.asarray(mask)
            )
            images: list[np.ndarray] = []
            for idx in indices:
                backend = self._cuda_compare_backend_for_index(int(idx))
                if backend is None:
                    continue
                image = backend.masked_sum(mask_np)
                images.append(
                    np.ascontiguousarray(
                        image / mask_area,
                        dtype=np.float32,
                    ).reshape(self.shape_rows, self.shape_cols)
                )
            return images
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

        Delegates to the compute backend (TorchBackend chunked tensordot on any
        torch device, MetalRawBackend raw kernel on chunk-backed Metal frames) so the
        widget follows the same detector geometry across CUDA, MPS, and test references.
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
        if self.vi_source != "roi":
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
