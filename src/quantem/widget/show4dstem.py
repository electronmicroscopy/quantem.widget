"""
show4dstem: Fast interactive 4D-STEM viewer widget.

Apple MPS GPU limit: PyTorch's MPS backend (Apple Silicon) has a hard limit
of ~2.1 billion elements (INT_MAX = 2^31 - 1) per tensor. Datasets exceeding
this automatically fall back to CPU, which is still fast on Apple Silicon
thanks to unified memory (CPU and GPU share the same RAM).

CUDA GPUs do not have this limit.

Common 4D-STEM sizes (float32):

    Scan     Detector   Elements     Size    MPS?
    128×128  128×128       268M    1.0 GB    yes
    128×128  256×256     1,074M    4.0 GB    yes
    256×256  128×128     1,074M    4.0 GB    yes
    256×256  192×192     2,416M    9.0 GB    no (auto CPU, still fast)
    256×256  256×256     4,295M   16.0 GB    no (auto CPU, still fast)
    512×512  256×256    17,180M   64.0 GB    no (auto CPU)

To reduce data size, bin k-space at the dataset level before viewing:

    dataset = dataset.bin(2, axes=(2, 3))  # 2x2 k-space binning
    widget = Show4DSTEM(dataset)
"""

import hashlib
import json
import math
import pathlib
import time
from datetime import datetime, timezone
from typing import Any, Self
from uuid import uuid4

import anywidget
import numpy as np
import torch
import traitlets

from quantem.core.config import validate_device
from quantem.widget.array_utils import to_numpy
from quantem.widget.io import IOResult, _format_memory
from quantem.widget.json_state import (
    build_json_header,
    resolve_widget_version,
    save_state_file,
    unwrap_state_payload,
)
from quantem.widget.tool_parity import (
    bind_tool_runtime_api,
    build_tool_groups,
    normalize_tool_groups,
)


# ============================================================================
# Constants
# ============================================================================
DEFAULT_BF_RATIO = 0.125  # BF disk radius as fraction of detector size (1/8)
SPARSE_MASK_THRESHOLD = 0.2  # Use sparse indexing below this mask coverage
MIN_LOG_VALUE = 1e-10  # Minimum value for log scale to avoid log(0)
DEFAULT_VI_ROI_RATIO = 0.15  # Default VI ROI size as fraction of scan dimension

class Show4DSTEM(anywidget.AnyWidget):
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
    pixel_size : float, optional
        Pixel size in Å (real-space). Used for scale bar.
        Auto-extracted from Dataset4dstem if not provided.
    k_pixel_size : float, optional
        Detector pixel size in mrad (k-space). Used for scale bar.
        Auto-extracted from Dataset4dstem if not provided.
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
    disabled_tools : list of str, optional
        Tool groups to lock while still showing controls. Supported:
        ``"display"``, ``"histogram"``, ``"stats"``, ``"navigation"``,
        ``"playback"``, ``"view"``, ``"export"``, ``"roi"``,
        ``"profile"``, ``"fft"``, ``"virtual"``, ``"frame"``, ``"all"``.
    disable_* : bool, optional
        Convenience flags mirroring ``disabled_tools`` for each tool group,
        plus ``disable_all``.
    hidden_tools : list of str, optional
        Tool groups to hide from the UI. Uses the same keys as
        ``disabled_tools``.
    hide_* : bool, optional
        Convenience flags mirroring ``disable_*`` for ``hidden_tools``.

    Examples
    --------
    >>> # From Dataset4dstem (calibration auto-extracted)
    >>> from quantem.core.io.file_readers import read_emdfile_to_4dstem
    >>> dataset = read_emdfile_to_4dstem("data.h5")
    >>> Show4DSTEM(dataset)

    >>> # From raw array with manual calibration
    >>> import numpy as np
    >>> data = np.random.rand(64, 64, 128, 128)
    >>> Show4DSTEM(data, pixel_size=2.39, k_pixel_size=0.46)

    >>> # With raster animation
    >>> widget = Show4DSTEM(dataset)
    >>> widget.raster(step=2, interval_ms=50)

    >>> # 5D time-series or tilt-series data
    >>> data_5d = np.random.rand(20, 64, 64, 128, 128)  # 20 frames
    >>> Show4DSTEM(data_5d, frame_dim_label="Tilt")
    """

    _esm = pathlib.Path(__file__).parent / "static" / "show4dstem.js"
    _css = pathlib.Path(__file__).parent / "static" / "show4dstem.css"

    # Position in scan space
    widget_version = traitlets.Unicode("unknown").tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
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
    roi_center = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0]).tag(sync=True)
    roi_radius = traitlets.Float(10.0).tag(sync=True)
    roi_radius_inner = traitlets.Float(5.0).tag(sync=True)
    roi_width = traitlets.Float(20.0).tag(sync=True)
    roi_height = traitlets.Float(10.0).tag(sync=True)

    # =========================================================================
    # Virtual Image (ROI-based, updates as you drag ROI on DP)
    # =========================================================================
    virtual_image_bytes = traitlets.Bytes(b"").tag(sync=True)  # Raw float32
    vi_data_min = traitlets.Float(0.0).tag(sync=True)  # Min of current VI for normalization
    vi_data_max = traitlets.Float(1.0).tag(sync=True)  # Max of current VI for normalization

    # =========================================================================
    # VI ROI (real-space region selection for summed DP)
    # =========================================================================
    vi_roi_mode = traitlets.Unicode("off").tag(sync=True)  # "off", "circle", "rect"
    vi_roi_center_row = traitlets.Float(0.0).tag(sync=True)
    vi_roi_center_col = traitlets.Float(0.0).tag(sync=True)
    vi_roi_radius = traitlets.Float(5.0).tag(sync=True)
    vi_roi_width = traitlets.Float(10.0).tag(sync=True)
    vi_roi_height = traitlets.Float(10.0).tag(sync=True)
    summed_dp_bytes = traitlets.Bytes(b"").tag(sync=True)  # Summed DP from VI ROI
    summed_dp_count = traitlets.Int(0).tag(sync=True)  # Number of positions summed

    # =========================================================================
    # Scale Bar
    # =========================================================================
    pixel_size = traitlets.Float(1.0).tag(sync=True)  # Å per pixel (real-space)
    k_pixel_size = traitlets.Float(1.0).tag(sync=True)  # mrad per pixel (k-space)
    k_calibrated = traitlets.Bool(False).tag(sync=True)  # True if k-space has mrad calibration

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
    auto_detect_trigger = traitlets.Bool(False).tag(sync=True)

    # =========================================================================
    # Statistics for display (mean, min, max, std)
    # =========================================================================
    dp_stats = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0, 0.0, 0.0]).tag(sync=True)
    vi_stats = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0, 0.0, 0.0]).tag(sync=True)
    mask_dc = traitlets.Bool(True).tag(sync=True)  # Mask center pixel for DP stats

    # =========================================================================
    # Display settings (synced for programmatic export parity)
    # =========================================================================
    dp_colormap = traitlets.Unicode("inferno").tag(sync=True)
    vi_colormap = traitlets.Unicode("inferno").tag(sync=True)
    fft_colormap = traitlets.Unicode("inferno").tag(sync=True)

    dp_scale_mode = traitlets.Unicode("linear").tag(sync=True)  # "linear" | "log" | "power"
    vi_scale_mode = traitlets.Unicode("linear").tag(sync=True)  # "linear" | "log" | "power"
    fft_scale_mode = traitlets.Unicode("linear").tag(sync=True)  # "linear" | "log" | "power"

    dp_power_exp = traitlets.Float(0.5).tag(sync=True)
    vi_power_exp = traitlets.Float(0.5).tag(sync=True)
    fft_power_exp = traitlets.Float(0.5).tag(sync=True)

    dp_vmin_pct = traitlets.Float(0.0).tag(sync=True)
    dp_vmax_pct = traitlets.Float(100.0).tag(sync=True)
    vi_vmin_pct = traitlets.Float(0.0).tag(sync=True)
    vi_vmax_pct = traitlets.Float(100.0).tag(sync=True)
    fft_vmin_pct = traitlets.Float(0.0).tag(sync=True)
    fft_vmax_pct = traitlets.Float(100.0).tag(sync=True)

    fft_auto = traitlets.Bool(True).tag(sync=True)
    show_fft = traitlets.Bool(False).tag(sync=True)
    fft_window = traitlets.Bool(True).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)
    dp_show_colorbar = traitlets.Bool(False).tag(sync=True)
    export_default_view = traitlets.Unicode("all").tag(sync=True)
    export_default_format = traitlets.Unicode("png").tag(sync=True)
    export_include_overlays = traitlets.Bool(True).tag(sync=True)
    export_include_scalebar = traitlets.Bool(True).tag(sync=True)
    export_default_dpi = traitlets.Int(300).tag(sync=True)

    # =========================================================================
    # Frame Animation (5D time/tilt series)
    # =========================================================================
    frame_idx = traitlets.Int(0).tag(sync=True)
    n_frames = traitlets.Int(1).tag(sync=True)
    frame_dim_label = traitlets.Unicode("Frame").tag(sync=True)
    frame_playing = traitlets.Bool(False).tag(sync=True)
    frame_loop = traitlets.Bool(True).tag(sync=True)
    frame_fps = traitlets.Float(5.0).tag(sync=True)
    frame_reverse = traitlets.Bool(False).tag(sync=True)
    frame_boomerang = traitlets.Bool(False).tag(sync=True)

    # Export (GIF)
    _gif_export_requested = traitlets.Bool(False).tag(sync=True)
    _gif_data = traitlets.Bytes(b"").tag(sync=True)
    _gif_metadata_json = traitlets.Unicode("").tag(sync=True)

    # Line Profile (for DP panel)
    profile_line = traitlets.List(traitlets.Dict()).tag(sync=True)
    profile_width = traitlets.Int(1).tag(sync=True)

    # =========================================================================
    # Tool visibility / locking
    # =========================================================================
    disabled_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)
    hidden_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)

    @classmethod
    def _normalize_tool_groups(cls, tool_groups) -> list[str]:
        return normalize_tool_groups("Show4DSTEM", tool_groups)

    @classmethod
    def _build_disabled_tools(
        cls,
        disabled_tools=None,
        disable_display: bool = False,
        disable_histogram: bool = False,
        disable_stats: bool = False,
        disable_navigation: bool = False,
        disable_playback: bool = False,
        disable_view: bool = False,
        disable_export: bool = False,
        disable_roi: bool = False,
        disable_profile: bool = False,
        disable_fft: bool = False,
        disable_virtual: bool = False,
        disable_frame: bool = False,
        disable_all: bool = False,
    ) -> list[str]:
        return build_tool_groups(
            "Show4DSTEM",
            tool_groups=disabled_tools,
            all_flag=disable_all,
            flag_map={
                "display": disable_display,
                "histogram": disable_histogram,
                "stats": disable_stats,
                "navigation": disable_navigation,
                "playback": disable_playback,
                "view": disable_view,
                "export": disable_export,
                "roi": disable_roi,
                "profile": disable_profile,
                "fft": disable_fft,
                "virtual": disable_virtual,
                "frame": disable_frame,
            },
        )

    @classmethod
    def _build_hidden_tools(
        cls,
        hidden_tools=None,
        hide_display: bool = False,
        hide_histogram: bool = False,
        hide_stats: bool = False,
        hide_navigation: bool = False,
        hide_playback: bool = False,
        hide_view: bool = False,
        hide_export: bool = False,
        hide_roi: bool = False,
        hide_profile: bool = False,
        hide_fft: bool = False,
        hide_virtual: bool = False,
        hide_frame: bool = False,
        hide_all: bool = False,
    ) -> list[str]:
        return build_tool_groups(
            "Show4DSTEM",
            tool_groups=hidden_tools,
            all_flag=hide_all,
            flag_map={
                "display": hide_display,
                "histogram": hide_histogram,
                "stats": hide_stats,
                "navigation": hide_navigation,
                "playback": hide_playback,
                "view": hide_view,
                "export": hide_export,
                "roi": hide_roi,
                "profile": hide_profile,
                "fft": hide_fft,
                "virtual": hide_virtual,
                "frame": hide_frame,
            },
        )

    @traitlets.validate("disabled_tools")
    def _validate_disabled_tools(self, proposal):
        return self._normalize_tool_groups(proposal["value"])

    @traitlets.validate("hidden_tools")
    def _validate_hidden_tools(self, proposal):
        return self._normalize_tool_groups(proposal["value"])

    def __init__(
        self,
        data: "Dataset4dstem | np.ndarray",
        scan_shape: tuple[int, int] | None = None,
        pixel_size: float | None = None,
        k_pixel_size: float | None = None,
        center: tuple[float, float] | None = None,
        bf_radius: float | None = None,
        precompute_virtual_images: bool = False,
        frame_dim_label: str | None = None,
        title: str = "",
        disabled_tools: list[str] | None = None,
        disable_display: bool = False,
        disable_histogram: bool = False,
        disable_stats: bool = False,
        disable_navigation: bool = False,
        disable_playback: bool = False,
        disable_view: bool = False,
        disable_export: bool = False,
        disable_roi: bool = False,
        disable_profile: bool = False,
        disable_fft: bool = False,
        disable_virtual: bool = False,
        disable_frame: bool = False,
        disable_all: bool = False,
        hidden_tools: list[str] | None = None,
        hide_display: bool = False,
        hide_histogram: bool = False,
        hide_stats: bool = False,
        hide_navigation: bool = False,
        hide_playback: bool = False,
        hide_view: bool = False,
        hide_export: bool = False,
        hide_roi: bool = False,
        hide_profile: bool = False,
        hide_fft: bool = False,
        hide_virtual: bool = False,
        hide_frame: bool = False,
        hide_all: bool = False,
        show_fft: bool = False,
        fft_window: bool = True,
        show_controls: bool = True,
        verbose: bool = True,
        state=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.widget_version = resolve_widget_version()
        _t0 = time.perf_counter()
        _verbose = verbose

        # Check if data is an IOResult and extract metadata
        _io_labels = None
        if isinstance(data, IOResult):
            if not title and data.title:
                title = data.title
            if data.labels:
                _io_labels = data.labels
            data = data.data

        # Extract calibration from Dataset4dstem if provided
        k_calibrated = False
        if hasattr(data, "sampling") and hasattr(data, "array"):
            # Dataset4dstem: extract calibration and array
            # sampling = [scan_rows, scan_cols, det_rows, det_cols]
            if not title and hasattr(data, "name") and data.name:
                title = str(data.name)
            units = getattr(data, "units", ["pixels"] * 4)
            if pixel_size is None and units[0] in ("Å", "angstrom", "A", "nm"):
                pixel_size = float(data.sampling[0])
                if units[0] == "nm":
                    pixel_size *= 10  # Convert nm to Å
            if k_pixel_size is None and units[2] in ("mrad", "1/Å", "1/A"):
                k_pixel_size = float(data.sampling[2])
                k_calibrated = True
            data = data.array

        self.title = title
        # Store calibration values (default to 1.0 if not provided)
        self.pixel_size = pixel_size if pixel_size is not None else 1.0
        self.k_pixel_size = k_pixel_size if k_pixel_size is not None else 1.0
        self.k_calibrated = k_calibrated or (k_pixel_size is not None)
        self.disabled_tools = self._build_disabled_tools(
            disabled_tools=disabled_tools,
            disable_display=disable_display,
            disable_histogram=disable_histogram,
            disable_stats=disable_stats,
            disable_navigation=disable_navigation,
            disable_playback=disable_playback,
            disable_view=disable_view,
            disable_export=disable_export,
            disable_roi=disable_roi,
            disable_profile=disable_profile,
            disable_fft=disable_fft,
            disable_virtual=disable_virtual,
            disable_frame=disable_frame,
            disable_all=disable_all,
        )
        self.hidden_tools = self._build_hidden_tools(
            hidden_tools=hidden_tools,
            hide_display=hide_display,
            hide_histogram=hide_histogram,
            hide_stats=hide_stats,
            hide_navigation=hide_navigation,
            hide_playback=hide_playback,
            hide_view=hide_view,
            hide_export=hide_export,
            hide_roi=hide_roi,
            hide_profile=hide_profile,
            hide_fft=hide_fft,
            hide_virtual=hide_virtual,
            hide_frame=hide_frame,
            hide_all=hide_all,
        )
        self.show_fft = show_fft
        self.fft_window = fft_window
        self.show_controls = show_controls
        # Path animation (configured via set_path() or raster())
        self._path_points: list[tuple[int, int]] = []
        # Named user presets saved during this session
        self._named_presets: dict[str, dict[str, Any]] = {}
        # Session-scoped reproducibility log for all export calls
        self._export_session_id = uuid4().hex
        self._export_session_started_utc = datetime.now(timezone.utc).isoformat()
        self._export_log: list[dict[str, Any]] = []
        # Sparse sampling state (for streaming/adaptive acquisition workflows)
        self._sparse_samples: dict[tuple[int, int, int], np.ndarray] = {}
        self._sparse_order: list[tuple[int, int, int]] = []
        # Convert to NumPy then PyTorch tensor using quantem device config
        data_np = to_numpy(data)
        device_str, _ = validate_device(None)  # Get device from quantem config
        self._device = torch.device(device_str)
        # Remove saturated hot pixels in numpy (before any torch conversion)
        saturated_value = 65535.0 if data_np.dtype == np.uint16 else 255.0 if data_np.dtype == np.uint8 else None
        if data_np.dtype != np.float32:
            _tc = time.perf_counter()
            data_np = data_np.astype(np.float32)
            if _verbose:
                print(f"  astype float32: {time.perf_counter() - _tc:.2f}s")
        if saturated_value is not None:
            data_np[data_np >= saturated_value] = 0
        # Handle dimensionality — 5D loads eagerly for instant frame switching
        ndim = data_np.ndim
        _tc = time.perf_counter()
        if ndim == 5:
            self.n_frames = data_np.shape[0]
            self._scan_shape = (data_np.shape[1], data_np.shape[2])
            self._det_shape = (data_np.shape[3], data_np.shape[4])
            if data_np.size > 2**31 - 1 and device_str == "mps":
                self._device = torch.device("cpu")
            self._data = torch.from_numpy(data_np).to(self._device)
        elif ndim == 3:
            self.n_frames = 1
            if scan_shape is not None:
                self._scan_shape = scan_shape
            else:
                n = data_np.shape[0]
                side = int(n ** 0.5)
                if side * side != n:
                    raise ValueError(
                        f"Cannot infer square scan_shape from N={n}. "
                        f"Provide scan_shape explicitly."
                    )
                self._scan_shape = (side, side)
            self._det_shape = (data_np.shape[1], data_np.shape[2])
            # MPS backend can't handle tensors >INT_MAX elements; fall back to CPU
            if data_np.size > 2**31 - 1 and device_str == "mps":
                self._device = torch.device("cpu")
            self._data = torch.from_numpy(data_np).to(self._device)
        elif ndim == 4:
            self.n_frames = 1
            self._scan_shape = (data_np.shape[0], data_np.shape[1])
            self._det_shape = (data_np.shape[2], data_np.shape[3])
            if data_np.size > 2**31 - 1 and device_str == "mps":
                self._device = torch.device("cpu")
            self._data = torch.from_numpy(data_np).to(self._device)
        else:
            raise ValueError(f"Expected 3D, 4D, or 5D array, got {ndim}D")
        if _verbose:
            if str(self._device) == "mps":
                torch.mps.synchronize()
            print(f"  to {self._device}: {time.perf_counter() - _tc:.2f}s ({data_np.nbytes / 1e9:.1f} GB)")

        self.shape_rows = self._scan_shape[0]
        self.shape_cols = self._scan_shape[1]
        self.det_rows = self._det_shape[0]
        self.det_cols = self._det_shape[1]
        # Initial position at center
        self.pos_row = self.shape_rows // 2
        self.pos_col = self.shape_cols // 2
        # Frame dimension label (for 5D time/tilt series UI)
        self.frame_dim_label = frame_dim_label if frame_dim_label is not None else "Frame"
        # Store per-frame labels from IOResult (e.g. master file stems)
        self._frame_labels = _io_labels if _io_labels else []
        # Histogram axis range — first frame is enough (JS does per-frame percentile clipping)
        first_frame = self._data[0] if self._data.ndim == 5 else self._data
        self.dp_global_min = max(float(first_frame.min()), MIN_LOG_VALUE)
        self.dp_global_max = float(first_frame.max())
        # Cache coordinate tensors for mask creation (avoid repeated torch.arange)
        self._det_row_coords = torch.arange(self.det_rows, device=self._device, dtype=torch.float32)[:, None]
        self._det_col_coords = torch.arange(self.det_cols, device=self._device, dtype=torch.float32)[None, :]
        self._scan_row_coords = torch.arange(self.shape_rows, device=self._device, dtype=torch.float32)[:, None]
        self._scan_col_coords = torch.arange(self.shape_cols, device=self._device, dtype=torch.float32)[None, :]
        self._sparse_mask = np.zeros((self.n_frames, self.shape_rows, self.shape_cols), dtype=bool)
        self._dose_map = np.zeros((self.n_frames, self.shape_rows, self.shape_cols), dtype=np.float32)
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
        if precompute_virtual_images and self.n_frames == 1:
            self._precompute_common_virtual_images()

        # Update frame when position changes (scale/colormap handled in JS)
        self.observe(self._update_frame, names=["pos_row", "pos_col"])
        # Observe individual ROI params
        self.observe(self._on_roi_change, names=[
            "roi_center_col", "roi_center_row", "roi_radius", "roi_radius_inner",
            "roi_active", "roi_mode", "roi_width", "roi_height"
        ])
        # Observe compound roi_center for batched updates from JS
        self.observe(self._on_roi_center_change, names=["roi_center"])

        # Initialize default ROI at BF center
        self.roi_center_col = self.center_col
        self.roi_center_row = self.center_row
        self.roi_center = [self.center_row, self.center_col]
        self.roi_radius = self.bf_radius * 0.5  # Start with half BF radius
        self.roi_active = True
        
        # Compute initial virtual image and frame
        _tc = time.perf_counter()
        self._compute_virtual_image_from_roi()
        self._update_frame()
        if _verbose:
            print(f"  virtual image + frame: {time.perf_counter() - _tc:.2f}s")
        
        # Path animation: observe index changes from frontend
        self.observe(self._on_path_index_change, names=["path_index"])
        self.observe(self._on_gif_export, names=["_gif_export_requested"])

        # Frame animation (5D): observe frame_idx changes from frontend
        self.observe(self._on_frame_idx_change, names=["frame_idx"])

        # Auto-detect trigger: observe changes from frontend
        self.observe(self._on_auto_detect_trigger, names=["auto_detect_trigger"])

        # VI ROI: observe changes for summed DP computation
        # Initialize VI ROI center to scan center with reasonable default sizes
        self.vi_roi_center_row = float(self.shape_rows / 2)
        self.vi_roi_center_col = float(self.shape_cols / 2)
        # Set initial ROI size based on scan dimension
        default_roi_size = max(3, min(self.shape_rows, self.shape_cols) * DEFAULT_VI_ROI_RATIO)
        self.vi_roi_radius = float(default_roi_size)
        self.vi_roi_width = float(default_roi_size * 2)
        self.vi_roi_height = float(default_roi_size)
        self.observe(self._on_vi_roi_change, names=[
            "vi_roi_mode", "vi_roi_center_row", "vi_roi_center_col",
            "vi_roi_radius", "vi_roi_width", "vi_roi_height"
        ])

        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(
                    json.loads(pathlib.Path(state).read_text()),
                    require_envelope=True,
                )
            else:
                state = unwrap_state_payload(state)
            self.load_state_dict(state)

        if _verbose:
            shape = "x".join(str(s) for s in self._data.shape)
            print(f"Show4DSTEM: {shape} {self._device}, {time.perf_counter() - _t0:.2f}s total")

    def set_image(self, data, scan_shape=None):
        """Replace the 4D-STEM data. Preserves all display and ROI settings."""
        if hasattr(data, "sampling") and hasattr(data, "array"):
            data = data.array
        data_np = to_numpy(data)
        saturated_value = 65535.0 if data_np.dtype == np.uint16 else 255.0 if data_np.dtype == np.uint8 else None
        if data_np.dtype != np.float32:
            data_np = data_np.astype(np.float32)
        if saturated_value is not None:
            data_np[data_np >= saturated_value] = 0
        if data_np.ndim == 5:
            self.n_frames = data_np.shape[0]
            self._scan_shape = (data_np.shape[1], data_np.shape[2])
            self._det_shape = (data_np.shape[3], data_np.shape[4])
            if data_np.size > 2**31 - 1 and str(self._device) == "mps":
                self._device = torch.device("cpu")
            self._data = torch.from_numpy(data_np).to(self._device)
        elif data_np.ndim == 3:
            self.n_frames = 1
            if scan_shape is not None:
                self._scan_shape = scan_shape
            else:
                n = data_np.shape[0]
                side = int(n ** 0.5)
                if side * side != n:
                    raise ValueError(f"Cannot infer square scan_shape from N={n}. Provide scan_shape explicitly.")
                self._scan_shape = (side, side)
            self._det_shape = (data_np.shape[1], data_np.shape[2])
            self._data = torch.from_numpy(data_np).to(self._device)
        elif data_np.ndim == 4:
            self.n_frames = 1
            self._scan_shape = (data_np.shape[0], data_np.shape[1])
            self._det_shape = (data_np.shape[2], data_np.shape[3])
            self._data = torch.from_numpy(data_np).to(self._device)
        else:
            raise ValueError(f"Expected 3D, 4D, or 5D array, got {data_np.ndim}D")
        self.frame_idx = 0
        self.shape_rows = self._scan_shape[0]
        self.shape_cols = self._scan_shape[1]
        self.det_rows = self._det_shape[0]
        self.det_cols = self._det_shape[1]
        first_frame = self._data[0] if self._data.ndim == 5 else self._data
        self.dp_global_min = max(float(first_frame.min()), MIN_LOG_VALUE)
        self.dp_global_max = float(first_frame.max())
        self._det_row_coords = torch.arange(self.det_rows, device=self._device, dtype=torch.float32)[:, None]
        self._det_col_coords = torch.arange(self.det_cols, device=self._device, dtype=torch.float32)[None, :]
        self._scan_row_coords = torch.arange(self.shape_rows, device=self._device, dtype=torch.float32)[:, None]
        self._scan_col_coords = torch.arange(self.shape_cols, device=self._device, dtype=torch.float32)[None, :]
        self._sparse_mask = np.zeros((self.n_frames, self.shape_rows, self.shape_cols), dtype=bool)
        self._dose_map = np.zeros((self.n_frames, self.shape_rows, self.shape_cols), dtype=np.float32)
        self._sparse_samples = {}
        self._sparse_order = []
        self._cached_bf_virtual = None
        self._cached_abf_virtual = None
        self._cached_adf_virtual = None
        self.pos_row = min(self.pos_row, self.shape_rows - 1)
        self.pos_col = min(self.pos_col, self.shape_cols - 1)
        self._compute_virtual_image_from_roi()
        self._update_frame()

    def __repr__(self) -> str:
        k_unit = "mrad" if self.k_calibrated else "px"
        shape = (
            f"({self.n_frames}, {self.shape_rows}, {self.shape_cols}, {self.det_rows}, {self.det_cols})"
            if self.n_frames > 1
            else f"({self.shape_rows}, {self.shape_cols}, {self.det_rows}, {self.det_cols})"
        )
        frame_info = f", {self.frame_dim_label.lower()}={self.frame_idx}" if self.n_frames > 1 else ""
        title_info = f", title='{self.title}'" if self.title else ""
        return (
            f"Show4DSTEM(shape={shape}, "
            f"sampling=({self.pixel_size} Å, {self.k_pixel_size} {k_unit}), "
            f"pos=({self.pos_row}, {self.pos_col}){frame_info}{title_info})"
        )

    def state_dict(self):
        return {
            "title": self.title,
            "pos_row": self.pos_row,
            "pos_col": self.pos_col,
            "pixel_size": self.pixel_size,
            "k_pixel_size": self.k_pixel_size,
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
            "mask_dc": self.mask_dc,
            "dp_colormap": self.dp_colormap,
            "vi_colormap": self.vi_colormap,
            "fft_colormap": self.fft_colormap,
            "dp_scale_mode": self.dp_scale_mode,
            "vi_scale_mode": self.vi_scale_mode,
            "fft_scale_mode": self.fft_scale_mode,
            "dp_power_exp": self.dp_power_exp,
            "vi_power_exp": self.vi_power_exp,
            "fft_power_exp": self.fft_power_exp,
            "dp_vmin_pct": self.dp_vmin_pct,
            "dp_vmax_pct": self.dp_vmax_pct,
            "vi_vmin_pct": self.vi_vmin_pct,
            "vi_vmax_pct": self.vi_vmax_pct,
            "fft_vmin_pct": self.fft_vmin_pct,
            "fft_vmax_pct": self.fft_vmax_pct,
            "fft_auto": self.fft_auto,
            "show_fft": self.show_fft,
            "fft_window": self.fft_window,
            "show_controls": self.show_controls,
            "dp_show_colorbar": self.dp_show_colorbar,
            "export_default_view": self.export_default_view,
            "export_default_format": self.export_default_format,
            "export_include_overlays": self.export_include_overlays,
            "export_include_scalebar": self.export_include_scalebar,
            "export_default_dpi": self.export_default_dpi,
            "path_interval_ms": self.path_interval_ms,
            "path_loop": self.path_loop,
            "profile_line": self.profile_line,
            "profile_width": self.profile_width,
            "frame_idx": self.frame_idx,
            "frame_dim_label": self.frame_dim_label,
            "frame_loop": self.frame_loop,
            "frame_fps": self.frame_fps,
            "frame_reverse": self.frame_reverse,
            "frame_boomerang": self.frame_boomerang,
            "disabled_tools": self.disabled_tools,
            "hidden_tools": self.hidden_tools,
        }

    def save(self, path: str):
        save_state_file(path, "Show4DSTEM", self.state_dict())

    def load_state_dict(self, state):
        pending_pos_row = state.get("pos_row", None)
        pending_pos_col = state.get("pos_col", None)
        pending_frame_idx = state.get("frame_idx", None)
        for key, val in state.items():
            if key in {"pos_row", "pos_col", "frame_idx"}:
                continue
            if hasattr(self, key):
                setattr(self, key, val)
        if pending_frame_idx is not None:
            self.frame_idx = int(max(0, min(int(pending_frame_idx), self.n_frames - 1)))
        if pending_pos_row is not None or pending_pos_col is not None:
            row = int(self.pos_row if pending_pos_row is None else pending_pos_row)
            col = int(self.pos_col if pending_pos_col is None else pending_pos_col)
            self.pos_row = int(max(0, min(row, self.shape_rows - 1)))
            self.pos_col = int(max(0, min(col, self.shape_cols - 1)))

    def free(self):
        """Free GPU memory held by this widget.

        Deletes the internal data tensor, runs garbage collection, and
        flushes the MPS allocator cache. Call this before loading a new
        dataset to avoid running out of GPU memory.

        Examples
        --------
        >>> w.free()          # release ~9 GB of MPS memory
        >>> del result        # free the source numpy array
        """
        import gc

        device = str(self._device) if hasattr(self, "_device") else ""
        nbytes = self._data.nbytes if hasattr(self._data, "nbytes") else 0
        self._data = None
        gc.collect()
        if device == "mps":
            try:
                import torch
                torch.mps.empty_cache()
            except Exception:
                pass
        if nbytes > 0:
            print(f"freed {_format_memory(nbytes)} ({device})")

    def summary(self):
        name = self.title if self.title else "Show4DSTEM"
        lines = [name, "═" * 32]
        if self.n_frames > 1:
            parts = [f"{self.n_frames} ({self.frame_dim_label}), current: {self.frame_idx}"]
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
                    lines.append(f"Labels:   {self._frame_labels[:3]} ... ({len(self._frame_labels)} total)")
        lines.append(f"Scan:     {self.shape_rows}×{self.shape_cols} ({self.pixel_size:.2f} Å/px)")
        k_unit = "mrad" if self.k_calibrated else "px"
        lines.append(f"Detector: {self.det_rows}×{self.det_cols} ({self.k_pixel_size:.4f} {k_unit}/px)")
        lines.append(f"Position: ({self.pos_row}, {self.pos_col})")
        lines.append(f"Center:   ({self.center_row:.1f}, {self.center_col:.1f})  BF r={self.bf_radius:.1f} px")
        display_parts = []
        if self.mask_dc:
            display_parts.append("DC masked")
        lines.append(f"Display:  {', '.join(display_parts) if display_parts else 'default'}")
        if self.roi_active:
            lines.append(f"ROI:      {self.roi_mode} at ({self.roi_center_row:.1f}, {self.roi_center_col:.1f}) r={self.roi_radius:.1f}")
        if self.vi_roi_mode != "off":
            lines.append(f"VI ROI:   {self.vi_roi_mode} at ({self.vi_roi_center_row:.1f}, {self.vi_roi_center_col:.1f}) r={self.vi_roi_radius:.1f}")
        lines.append(
            f"DP view:  {self.dp_colormap}, {self.dp_scale_mode}, {self.dp_vmin_pct:.1f}-{self.dp_vmax_pct:.1f}%"
        )
        lines.append(
            f"VI view:  {self.vi_colormap}, {self.vi_scale_mode}, {self.vi_vmin_pct:.1f}-{self.vi_vmax_pct:.1f}%"
        )
        if self.show_fft:
            fft_parts = [f"{self.fft_colormap}, {self.fft_scale_mode}, {self.fft_vmin_pct:.1f}-{self.fft_vmax_pct:.1f}%, auto={self.fft_auto}"]
            if not self.fft_window:
                fft_parts.append("no window")
            lines.append(f"FFT view: {', '.join(fft_parts)}")
        if self.profile_line and len(self.profile_line) == 2:
            p0, p1 = self.profile_line[0], self.profile_line[1]
            lines.append(f"Profile:  ({p0['row']:.0f}, {p0['col']:.0f}) -> ({p1['row']:.0f}, {p1['col']:.0f}) width={self.profile_width}")
        if self.disabled_tools:
            lines.append(f"Locked:   {', '.join(self.disabled_tools)}")
        if self.hidden_tools:
            lines.append(f"Hidden:   {', '.join(self.hidden_tools)}")
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
        if self.n_frames > 1:
            return self._data[self.frame_idx]
        return self._data

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
        out = np.empty(n, dtype=np.float32)
        for i in range(n):
            t = i / (n - 1)
            c = col0 + t * dc
            r = row0 + t * dr
            ci, ri = int(np.floor(c)), int(np.floor(r))
            cf, rf = c - ci, r - ri
            c0c = max(0, min(w - 1, ci))
            c1c = max(0, min(w - 1, ci + 1))
            r0c = max(0, min(h - 1, ri))
            r1c = max(0, min(h - 1, ri + 1))
            out[i] = (
                frame[r0c, c0c] * (1 - cf) * (1 - rf)
                + frame[r0c, c1c] * cf * (1 - rf)
                + frame[r1c, c0c] * (1 - cf) * rf
                + frame[r1c, c1c] * cf * rf
            )
        return out

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

    def _on_auto_detect_trigger(self, change):
        """Called when auto_detect_trigger is set to True from frontend."""
        if change["new"]:
            self.auto_detect_center()
            # Reset trigger to allow re-triggering
            self.auto_detect_trigger = False

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
        # Recompute virtual image and displayed frame
        self._compute_virtual_image_from_roi()
        self._update_frame()
        # Recompute summed DP if VI ROI is active
        if self.vi_roi_mode != "off":
            self._compute_summed_dp_from_vi_roi()

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

    def roi_rect(
        self, width: float | None = None, height: float | None = None
    ) -> Self:
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
        # Sum all diffraction patterns to get average (PyTorch)
        if self._data.ndim == 5:
            summed_dp = self._data.sum(dim=(0, 1, 2))
        elif self._data.ndim == 4:
            summed_dp = self._data.sum(dim=(0, 1))
        else:
            summed_dp = self._data.sum(dim=0)

        # Threshold at mean + std to isolate BF disk
        threshold = summed_dp.mean() + summed_dp.std()
        mask = summed_dp > threshold

        # Avoid division by zero
        total = mask.sum()
        if total == 0:
            return self

        # Calculate centroid using cached coordinate grids
        cx = float((self._det_col_coords * mask).sum() / total)
        cy = float((self._det_row_coords * mask).sum() / total)

        # Estimate radius from mask area (A = pi*r^2)
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
        """Get single diffraction frame at position (row, col) as numpy array."""
        data = self._frame_data
        if data.ndim == 3:
            idx = row * self.shape_cols + col
            return data[idx].cpu().numpy()
        else:
            return data[row, col].cpu().numpy()

    def _apply_scale_mode(
        self,
        data: np.ndarray,
        mode: str,
        power_exp: float = 0.5,
    ) -> np.ndarray:
        arr = np.asarray(data, dtype=np.float32)
        if mode == "log":
            return np.log1p(np.maximum(arr, 0.0)).astype(np.float32)
        if mode == "power":
            return np.power(np.maximum(arr, 0.0), float(power_exp)).astype(np.float32)
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

    def _get_summed_dp_array(self) -> np.ndarray | None:
        if self.vi_roi_mode == "off":
            return None
        self._compute_summed_dp_from_vi_roi()
        if not self.summed_dp_bytes:
            return None
        arr = np.frombuffer(self.summed_dp_bytes, dtype=np.float32)
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
        summed_dp = self._get_summed_dp_array()
        if summed_dp is not None:
            raw = summed_dp
            source = "summed_dp"
        else:
            raw = self._get_frame(self.pos_row, self.pos_col).astype(np.float32)
            source = "single_frame"

        scale_mode = self.dp_scale_mode
        scaled = self._apply_scale_mode(raw, scale_mode, self.dp_power_exp)
        data_min = float(scaled.min()) if scaled.size else 0.0
        data_max = float(scaled.max()) if scaled.size else 0.0
        vmin, vmax = self._slider_range(data_min, data_max, self.dp_vmin_pct, self.dp_vmax_pct)
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
        scaled = self._apply_scale_mode(raw, self.vi_scale_mode, self.vi_power_exp)
        data_min = float(scaled.min()) if scaled.size else 0.0
        data_max = float(scaled.max()) if scaled.size else 0.0
        vmin, vmax = self._slider_range(data_min, data_max, self.vi_vmin_pct, self.vi_vmax_pct)
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
        scaled = self._apply_scale_mode(mag, self.fft_scale_mode, self.fft_power_exp)
        if self.fft_auto:
            display_min, display_max = self._fft_enhanced_range(scaled)
        else:
            display_min = float(scaled.min()) if scaled.size else 0.0
            display_max = float(scaled.max()) if scaled.size else 0.0
        vmin, vmax = self._slider_range(display_min, display_max, self.fft_vmin_pct, self.fft_vmax_pct)
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

    def list_export_views(self) -> tuple[str, ...]:
        return ("diffraction", "virtual", "fft", "all")

    def list_export_formats(self) -> tuple[str, ...]:
        return ("png", "pdf")

    def list_figure_templates(self) -> tuple[str, ...]:
        return ("dp_vi", "dp_vi_fft", "publication_dp_vi", "publication_dp_vi_fft")

    def list_presets(self) -> tuple[str, ...]:
        builtin = ("bf", "abf", "adf", "haadf")
        custom = tuple(sorted(self._named_presets.keys()))
        return builtin + custom

    def _validate_export_view(self, view: str | None) -> str:
        candidate = self.export_default_view if view is None else str(view)
        view_key = str(candidate).strip().lower()
        allowed = self.list_export_views()
        if view_key not in allowed:
            raise ValueError(
                f"Unsupported view '{view}'. Supported: {', '.join(allowed)}"
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

    def _resolve_export_format(
        self,
        path: pathlib.Path,
        fmt: str | None,
    ) -> str:
        if fmt is not None and str(fmt).strip():
            resolved = str(fmt).strip().lower()
        else:
            from_path = path.suffix.lstrip(".").lower()
            resolved = from_path if from_path else str(self.export_default_format).strip().lower()
        allowed = self.list_export_formats()
        if resolved not in allowed:
            raise ValueError(
                f"Unsupported format '{resolved}'. Supported: {', '.join(allowed)}"
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
    def _draw_crosshair(draw, x: float, y: float, size: float, color, width: int) -> None:
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
                draw.ellipse([(cx - rx, cy - ry), (cx + rx, cy + ry)], outline=stroke, fill=fill, width=2)
            elif self.roi_mode == "square":
                rx = float(self.roi_radius) * scale_x
                ry = float(self.roi_radius) * scale_y
                draw.rectangle([(cx - rx, cy - ry), (cx + rx, cy + ry)], outline=stroke, fill=fill, width=2)
            elif self.roi_mode == "rect":
                rx = (float(self.roi_width) / 2.0) * scale_x
                ry = (float(self.roi_height) / 2.0) * scale_y
                draw.rectangle([(cx - rx, cy - ry), (cx + rx, cy + ry)], outline=stroke, fill=fill, width=2)
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
        self._draw_crosshair(draw, cx, cy, size=max(6, int(min(width, height) * 0.03)), color=marker_color, width=2)

        if len(self.profile_line) == 2:
            p0, p1 = self.profile_line[0], self.profile_line[1]
            x0 = float(p0["col"]) * scale_x
            y0 = float(p0["row"]) * scale_y
            x1 = float(p1["col"]) * scale_x
            y1 = float(p1["row"]) * scale_y
            draw.line([(x0, y0), (x1, y1)], fill=(0, 200, 255, 240), width=max(1, int(self.profile_width)))
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
            draw.ellipse([(cx - rx, cy - ry), (cx + rx, cy + ry)], outline=stroke, fill=fill, width=2)
        elif self.vi_roi_mode == "square":
            rx = float(self.vi_roi_radius) * scale_x
            ry = float(self.vi_roi_radius) * scale_y
            draw.rectangle([(cx - rx, cy - ry), (cx + rx, cy + ry)], outline=stroke, fill=fill, width=2)
        elif self.vi_roi_mode == "rect":
            rx = (float(self.vi_roi_width) / 2.0) * scale_x
            ry = (float(self.vi_roi_height) / 2.0) * scale_y
            draw.rectangle([(cx - rx, cy - ry), (cx + rx, cy + ry)], outline=stroke, fill=fill, width=2)

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
            raise ValueError(f"Unsupported panel '{panel_key}'")

        panel = Image.fromarray(rgb, mode="RGB")
        panel = self._decorate_panel(panel, panel_key, include_overlays, include_scalebar)
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

    def _export_settings_metadata(self) -> dict[str, Any]:
        return {
            "default_view": self.export_default_view,
            "default_format": self.export_default_format,
            "include_overlays": bool(self.export_include_overlays),
            "include_scalebar": bool(self.export_include_scalebar),
            "dpi": int(self.export_default_dpi),
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
            "export_settings": self._export_settings_metadata(),
        }
        if extra:
            metadata.update(extra)
        return metadata

    @staticmethod
    def _sha256_file(path: pathlib.Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                chunk = f.read(1_048_576)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def _build_file_record(
        self,
        path: pathlib.Path,
        metadata_path: pathlib.Path | None = None,
        index: int | None = None,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "path": str(path),
            "sha256": self._sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
        if metadata_path is not None and metadata_path.exists():
            record["metadata_path"] = str(metadata_path)
            record["metadata_sha256"] = self._sha256_file(metadata_path)
            record["metadata_size_bytes"] = int(metadata_path.stat().st_size)
        if index is not None:
            record["index"] = int(index)
        return record

    def _record_export_event(self, event: dict[str, Any]) -> None:
        payload = {
            "session_id": self._export_session_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        payload.update(event)
        self._export_log.append(payload)

    def _validate_sparse_frame_idx(self, frame_idx: int | None) -> int:
        if self.n_frames <= 1:
            return 0
        if frame_idx is None:
            return int(self.frame_idx)
        idx = int(frame_idx)
        if idx < 0 or idx >= self.n_frames:
            raise ValueError(f"frame_idx={idx} is out of range [0, {self.n_frames - 1}]")
        return idx

    def _normalize_sparse_mask(self, mask: np.ndarray) -> np.ndarray:
        arr = np.asarray(mask)
        if self.n_frames <= 1:
            if arr.shape == (self.shape_rows, self.shape_cols):
                arr = arr[None, ...]
            elif arr.shape != (1, self.shape_rows, self.shape_cols):
                raise ValueError(
                    f"mask shape {arr.shape} does not match "
                    f"(scan_rows, scan_cols)=({self.shape_rows}, {self.shape_cols})"
                )
        elif arr.shape != (self.n_frames, self.shape_rows, self.shape_cols):
            raise ValueError(
                f"mask shape {arr.shape} does not match "
                f"(n_frames, scan_rows, scan_cols)=({self.n_frames}, {self.shape_rows}, {self.shape_cols})"
            )
        return arr.astype(bool, copy=False)

    def _coerce_dp_array(self, dp: np.ndarray) -> np.ndarray:
        arr = np.asarray(to_numpy(dp), dtype=np.float32)
        if arr.shape != (self.det_rows, self.det_cols):
            raise ValueError(
                f"dp shape {arr.shape} does not match detector_shape "
                f"({self.det_rows}, {self.det_cols})"
            )
        return arr

    def _write_dp_to_data(self, frame_idx: int, row: int, col: int, dp_arr: np.ndarray) -> None:
        dp_tensor = torch.from_numpy(dp_arr).to(device=self._device, dtype=torch.float32)
        if self.n_frames > 1:
            self._data[frame_idx, row, col] = dp_tensor
        elif self._data.ndim == 4:
            self._data[row, col] = dp_tensor
        else:
            flat_idx = row * self.shape_cols + col
            self._data[flat_idx] = dp_tensor

    def _ingest_scan_point_core(
        self,
        row: int,
        col: int,
        dp: np.ndarray,
        frame_idx: int,
        dose: float,
        refresh: bool,
    ) -> None:
        row_i, col_i = self._validate_position((row, col))
        frame_i = self._validate_sparse_frame_idx(frame_idx)
        dp_arr = self._coerce_dp_array(dp)
        dose_value = float(dose)
        if not np.isfinite(dose_value) or dose_value < 0:
            raise ValueError(f"dose must be finite and >= 0, got {dose}")

        key = (int(frame_i), int(row_i), int(col_i))
        if key not in self._sparse_samples:
            self._sparse_order.append(key)
        self._sparse_samples[key] = dp_arr.copy()
        self._sparse_mask[frame_i, row_i, col_i] = True
        self._dose_map[frame_i, row_i, col_i] += dose_value

        self._write_dp_to_data(frame_i, row_i, col_i, dp_arr)
        self.dp_global_min = max(min(float(self.dp_global_min), float(dp_arr.min())), MIN_LOG_VALUE)
        self.dp_global_max = max(float(self.dp_global_max), float(dp_arr.max()))

        if refresh:
            self._compute_virtual_image_from_roi()
            self._update_frame()

    def _detector_integration_kernel(self) -> tuple[np.ndarray | None, tuple[int, int] | None]:
        cx, cy = float(self.roi_center_col), float(self.roi_center_row)
        rr, cc = np.meshgrid(
            np.arange(self.det_rows, dtype=np.float32),
            np.arange(self.det_cols, dtype=np.float32),
            indexing="ij",
        )
        if self.roi_mode == "circle" and self.roi_radius > 0:
            mask = (cc - cx) ** 2 + (rr - cy) ** 2 <= float(self.roi_radius) ** 2
            return mask.astype(np.float32, copy=False), None
        if self.roi_mode == "square" and self.roi_radius > 0:
            half = float(self.roi_radius)
            mask = (np.abs(cc - cx) <= half) & (np.abs(rr - cy) <= half)
            return mask.astype(np.float32, copy=False), None
        if self.roi_mode == "annular" and self.roi_radius > 0:
            outer = float(self.roi_radius)
            inner = float(self.roi_radius_inner)
            dist_sq = (cc - cx) ** 2 + (rr - cy) ** 2
            mask = (dist_sq >= inner**2) & (dist_sq <= outer**2)
            return mask.astype(np.float32, copy=False), None
        if self.roi_mode == "rect" and self.roi_width > 0 and self.roi_height > 0:
            hw = float(self.roi_width) / 2.0
            hh = float(self.roi_height) / 2.0
            mask = (np.abs(cc - cx) <= hw) & (np.abs(rr - cy) <= hh)
            return mask.astype(np.float32, copy=False), None
        row = int(max(0, min(round(cy), self.det_rows - 1)))
        col = int(max(0, min(round(cx), self.det_cols - 1)))
        return None, (row, col)

    def _integrate_dp_value(
        self,
        dp: np.ndarray,
        mask: np.ndarray | None,
        point_idx: tuple[int, int] | None,
    ) -> float:
        arr = np.asarray(dp, dtype=np.float32)
        if point_idx is not None:
            row, col = point_idx
            return float(arr[row, col])
        if mask is None:
            return 0.0
        return float((arr * mask).sum())

    def _virtual_image_from_frame_array(self, frame_data: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame_data, dtype=np.float32)
        if arr.shape != (self.shape_rows, self.shape_cols, self.det_rows, self.det_cols):
            raise ValueError(
                f"frame_data shape {arr.shape} does not match "
                f"({self.shape_rows}, {self.shape_cols}, {self.det_rows}, {self.det_cols})"
            )
        mask, point_idx = self._detector_integration_kernel()
        if point_idx is not None:
            row, col = point_idx
            return arr[:, :, row, col].astype(np.float32, copy=False)
        return (arr * mask[None, None, :, :]).sum(axis=(2, 3)).astype(np.float32)

    @staticmethod
    def _idw_reconstruct(
        shape: tuple[int, int],
        points: np.ndarray,
        values: np.ndarray,
        power: float = 2.0,
        k_neighbors: int = 16,
    ) -> np.ndarray:
        if points.size == 0:
            return np.zeros(shape, dtype=np.float32)
        rr, cc = np.meshgrid(
            np.arange(shape[0], dtype=np.float32),
            np.arange(shape[1], dtype=np.float32),
            indexing="ij",
        )
        coords = np.stack([rr.reshape(-1), cc.reshape(-1)], axis=1)
        dist_sq = ((coords[:, None, :] - points[None, :, :]) ** 2).sum(axis=2) + 1e-6

        if k_neighbors > 0 and points.shape[0] > k_neighbors:
            idx = np.argpartition(dist_sq, kth=k_neighbors - 1, axis=1)[:, :k_neighbors]
            dist_sq = np.take_along_axis(dist_sq, idx, axis=1)
            vals_local = values[idx]
        else:
            vals_local = np.broadcast_to(values[None, :], dist_sq.shape)

        weights = 1.0 / np.power(dist_sq, power / 2.0)
        pred = (weights * vals_local).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-6)
        return pred.reshape(shape).astype(np.float32, copy=False)

    def _resolve_reference_virtual_image(
        self,
        reference: str | np.ndarray,
        frame_idx: int,
    ) -> tuple[np.ndarray, str]:
        if isinstance(reference, str):
            key = reference.strip().lower()
            if key != "full_raster":
                raise ValueError("reference must be 'full_raster' or a NumPy array")
            if self.n_frames > 1:
                frame = self._data[frame_idx].detach().cpu().numpy()
            elif self._data.ndim == 4:
                frame = self._data.detach().cpu().numpy()
            else:
                frame = self._data.detach().cpu().numpy().reshape(
                    self.shape_rows, self.shape_cols, self.det_rows, self.det_cols
                )
            return self._virtual_image_from_frame_array(frame), "full_raster"

        arr = np.asarray(to_numpy(reference), dtype=np.float32)
        if arr.shape == (self.shape_rows, self.shape_cols):
            return arr.astype(np.float32, copy=False), "virtual_image"
        if arr.shape == (self.shape_rows, self.shape_cols, self.det_rows, self.det_cols):
            return self._virtual_image_from_frame_array(arr), "frame_data"
        if arr.shape == (self.n_frames, self.shape_rows, self.shape_cols, self.det_rows, self.det_cols):
            return self._virtual_image_from_frame_array(arr[frame_idx]), "stack_frame_data"
        raise ValueError(
            "Unsupported reference shape. Expected one of: "
            f"(scan_rows, scan_cols), "
            f"(scan_rows, scan_cols, det_rows, det_cols), or "
            f"(n_frames, scan_rows, scan_cols, det_rows, det_cols)."
        )

    def _extract_sparse_samples(self, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        mask = self._sparse_mask[frame_idx]
        coords = np.argwhere(mask)
        if coords.size == 0:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        integ_mask, point_idx = self._detector_integration_kernel()
        values = np.zeros((coords.shape[0],), dtype=np.float32)
        for i, (row, col) in enumerate(coords):
            key = (int(frame_idx), int(row), int(col))
            dp = self._sparse_samples.get(key)
            if dp is None:
                dp = self._get_frame(int(row), int(col))
            values[i] = self._integrate_dp_value(dp, integ_mask, point_idx)
        points = coords.astype(np.float32, copy=False)
        return points, values

    def ingest_scan_point(
        self,
        row: int,
        col: int,
        dp: np.ndarray,
        frame_idx: int = 0,
        dose: float | None = None,
    ) -> Self:
        """
        Ingest one scanned diffraction pattern into sparse acquisition state.

        Parameters
        ----------
        row : int
            Scan-space row index.
        col : int
            Scan-space column index.
        dp : array_like
            Diffraction pattern with shape ``(det_rows, det_cols)``.
        frame_idx : int, default 0
            Frame index for 5D data.
        dose : float, optional
            Dose contribution for this acquisition event. Defaults to ``1.0``.

        Returns
        -------
        Show4DSTEM
            Self for method chaining.
        """
        self._ingest_scan_point_core(
            row=row,
            col=col,
            dp=dp,
            frame_idx=frame_idx,
            dose=1.0 if dose is None else float(dose),
            refresh=True,
        )
        self._record_export_event(
            {
                "export_kind": "ingest_scan_point",
                "frame_idx": int(self._validate_sparse_frame_idx(frame_idx)),
                "row": int(row),
                "col": int(col),
                "dose": float(1.0 if dose is None else dose),
            }
        )
        return self

    def ingest_scan_block(
        self,
        rows: list[int] | np.ndarray,
        cols: list[int] | np.ndarray,
        dp_block: np.ndarray,
        frame_idx: int = 0,
    ) -> Self:
        """
        Ingest multiple scanned diffraction patterns in one call.

        Parameters
        ----------
        rows : list[int] or np.ndarray
            Row indices for each pattern in ``dp_block``.
        cols : list[int] or np.ndarray
            Column indices for each pattern in ``dp_block``.
        dp_block : np.ndarray
            Diffraction stack with shape ``(n_points, det_rows, det_cols)``.
        frame_idx : int, default 0
            Frame index for 5D data.

        Returns
        -------
        Show4DSTEM
            Self for method chaining.
        """
        rows_arr = np.asarray(rows, dtype=np.int64).reshape(-1)
        cols_arr = np.asarray(cols, dtype=np.int64).reshape(-1)
        if rows_arr.size != cols_arr.size:
            raise ValueError("rows and cols must have the same length")

        block = np.asarray(to_numpy(dp_block), dtype=np.float32)
        if block.ndim == 2:
            block = block[None, ...]
        if block.ndim != 3 or block.shape[1:] != (self.det_rows, self.det_cols):
            raise ValueError(
                f"dp_block shape must be (n_points, {self.det_rows}, {self.det_cols}), got {block.shape}"
            )
        if block.shape[0] != rows_arr.size:
            raise ValueError(
                f"dp_block has {block.shape[0]} patterns but rows/cols specify {rows_arr.size} points"
            )

        frame_i = self._validate_sparse_frame_idx(frame_idx)
        for idx in range(rows_arr.size):
            self._ingest_scan_point_core(
                row=int(rows_arr[idx]),
                col=int(cols_arr[idx]),
                dp=block[idx],
                frame_idx=frame_i,
                dose=1.0,
                refresh=False,
            )

        self._compute_virtual_image_from_roi()
        self._update_frame()
        self._record_export_event(
            {
                "export_kind": "ingest_scan_block",
                "frame_idx": int(frame_i),
                "n_points": int(rows_arr.size),
            }
        )
        return self

    def get_sparse_state(self) -> dict[str, Any]:
        """
        Return sparse acquisition state for checkpointing or replay.

        Returns
        -------
        dict
            Sparse state with sampling mask, sampled diffraction stack,
            sampled-point coordinates, and dose map.
        """
        coords = np.argwhere(self._sparse_mask)
        sampled_points = [
            {"frame_idx": int(f), "row": int(r), "col": int(c)}
            for (f, r, c) in coords
        ]
        if coords.size:
            sampled_data = np.stack(
                [
                    self._sparse_samples.get((int(f), int(r), int(c)), self._get_frame(int(r), int(c)))
                    for (f, r, c) in coords
                ],
                axis=0,
            ).astype(np.float32, copy=False)
        else:
            sampled_data = np.zeros((0, self.det_rows, self.det_cols), dtype=np.float32)

        mask_payload = self._sparse_mask[0].copy() if self.n_frames <= 1 else self._sparse_mask.copy()
        dose_payload = self._dose_map[0].copy() if self.n_frames <= 1 else self._dose_map.copy()
        return {
            **build_json_header("Show4DSTEM"),
            "format": "json",
            "export_kind": "sparse_state_snapshot",
            "frame_idx": int(self.frame_idx),
            "scan_shape": {"rows": int(self.shape_rows), "cols": int(self.shape_cols)},
            "detector_shape": {"rows": int(self.det_rows), "cols": int(self.det_cols)},
            "mask": mask_payload,
            "sampled_data": sampled_data,
            "sampled_points": sampled_points,
            "dose_map": dose_payload,
            "n_sampled": int(len(sampled_points)),
            "total_dose": float(self._dose_map.sum()),
        }

    def set_sparse_state(
        self,
        mask: np.ndarray,
        sampled_data: np.ndarray,
    ) -> Self:
        """
        Restore sparse acquisition state from mask + sampled data.

        Parameters
        ----------
        mask : np.ndarray
            Boolean scan mask. Shape ``(scan_rows, scan_cols)`` for 4D,
            or ``(n_frames, scan_rows, scan_cols)`` for 5D.
        sampled_data : np.ndarray
            Either compact stack ``(n_sampled, det_rows, det_cols)``
            matching row-major ``mask`` order, or dense data aligned to mask:
            ``(scan_rows, scan_cols, det_rows, det_cols)`` for 4D,
            ``(n_frames, scan_rows, scan_cols, det_rows, det_cols)`` for 5D.

        Returns
        -------
        Show4DSTEM
            Self for method chaining.
        """
        mask_3d = self._normalize_sparse_mask(mask)
        coords = np.argwhere(mask_3d)

        payload = np.asarray(to_numpy(sampled_data), dtype=np.float32)
        n_points = int(coords.shape[0])

        if payload.ndim == 3:
            if payload.shape[0] != n_points or payload.shape[1:] != (self.det_rows, self.det_cols):
                raise ValueError(
                    f"Compact sampled_data must be (n_sampled, {self.det_rows}, {self.det_cols}); "
                    f"got {payload.shape} for n_sampled={n_points}"
                )
            compact = payload
        elif self.n_frames <= 1 and payload.shape == (self.shape_rows, self.shape_cols, self.det_rows, self.det_cols):
            compact = np.stack(
                [payload[int(r), int(c)] for (_, r, c) in coords],
                axis=0,
            ) if n_points else np.zeros((0, self.det_rows, self.det_cols), dtype=np.float32)
        elif payload.shape == (
            self.n_frames,
            self.shape_rows,
            self.shape_cols,
            self.det_rows,
            self.det_cols,
        ):
            compact = np.stack(
                [payload[int(f), int(r), int(c)] for (f, r, c) in coords],
                axis=0,
            ) if n_points else np.zeros((0, self.det_rows, self.det_cols), dtype=np.float32)
        else:
            raise ValueError(
                "Unsupported sampled_data shape for set_sparse_state. "
                "Use compact (n_sampled, det_rows, det_cols) or dense per-mask arrays."
            )

        self._sparse_samples = {}
        self._sparse_order = []
        self._sparse_mask = np.zeros((self.n_frames, self.shape_rows, self.shape_cols), dtype=bool)
        self._dose_map = np.zeros((self.n_frames, self.shape_rows, self.shape_cols), dtype=np.float32)

        for idx, (frame_idx, row, col) in enumerate(coords):
            self._ingest_scan_point_core(
                row=int(row),
                col=int(col),
                dp=compact[idx],
                frame_idx=int(frame_idx),
                dose=1.0,
                refresh=False,
            )

        self._compute_virtual_image_from_roi()
        self._update_frame()
        self._record_export_event(
            {
                "export_kind": "set_sparse_state",
                "n_sampled": int(n_points),
            }
        )
        return self

    def _resolve_proposal_count(
        self,
        k: int,
        frame_idx: int,
        budget: dict[str, Any] | None,
    ) -> int:
        count = int(k)
        if count < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if budget is None:
            return count

        existing_points = int(self._sparse_mask[frame_idx].sum())
        existing_dose = float(self._dose_map[frame_idx].sum())
        total_points = int(self.shape_rows * self.shape_cols)

        if "max_new_points" in budget:
            count = min(count, int(budget["max_new_points"]))
        if "max_total_points" in budget:
            count = min(count, max(0, int(budget["max_total_points"]) - existing_points))
        if "max_total_fraction" in budget:
            allowed_total = int(round(float(budget["max_total_fraction"]) * total_points))
            count = min(count, max(0, allowed_total - existing_points))
        if "max_total_dose" in budget:
            dose_per_point = float(budget.get("dose_per_point", 1.0))
            if dose_per_point <= 0:
                raise ValueError("budget['dose_per_point'] must be > 0")
            remaining = float(budget["max_total_dose"]) - existing_dose
            count = min(count, max(0, int(math.floor(remaining / dose_per_point))))
        return max(0, int(count))

    def propose_next_points(
        self,
        k: int,
        strategy: str = "adaptive",
        budget: dict[str, Any] | None = None,
    ) -> list[tuple[int, int]]:
        """
        Propose next scan points from current sparse acquisition state.

        Parameters
        ----------
        k : int
            Maximum number of new points to propose.
        strategy : str, default "adaptive"
            Proposal strategy: ``"adaptive"``, ``"random"``, or ``"raster"``.
        budget : dict, optional
            Optional constraints and strategy parameters. Supported keys:
            ``frame_idx``, ``max_new_points``, ``max_total_points``,
            ``max_total_fraction``, ``max_total_dose``, ``dose_per_point``,
            ``roi_mask``, ``seed``, ``min_spacing``, ``step``,
            ``local_window``, ``dose_lambda``, ``weights``, ``bidirectional``.

        Returns
        -------
        list[tuple[int, int]]
            Proposed ``(row, col)`` scan coordinates.
        """
        budget_dict = {} if budget is None else dict(budget)
        strategy_key = str(strategy).strip().lower()
        if strategy_key not in {"adaptive", "random", "raster"}:
            raise ValueError("strategy must be one of: adaptive, random, raster")

        frame_idx = self._validate_sparse_frame_idx(budget_dict.get("frame_idx", self.frame_idx))
        n_select = self._resolve_proposal_count(int(k), frame_idx, budget_dict)
        if n_select <= 0:
            return []

        sampled_mask = self._sparse_mask[frame_idx].copy()
        allowed_mask = ~sampled_mask
        roi_mask_raw = budget_dict.get("roi_mask", None)
        if roi_mask_raw is not None:
            roi_mask = np.asarray(roi_mask_raw, dtype=bool)
            if roi_mask.shape != (self.shape_rows, self.shape_cols):
                raise ValueError(
                    f"roi_mask shape {roi_mask.shape} must match "
                    f"scan_shape ({self.shape_rows}, {self.shape_cols})"
                )
            allowed_mask &= roi_mask

        proposals: list[tuple[int, int]] = []
        if strategy_key == "adaptive":
            local_window = int(budget_dict.get("local_window", 5))
            if local_window < 1:
                raise ValueError("budget['local_window'] must be >= 1")
            min_spacing = int(budget_dict.get("min_spacing", 2))
            if min_spacing < 0:
                raise ValueError("budget['min_spacing'] must be >= 0")
            dose_lambda = float(budget_dict.get("dose_lambda", 0.25))
            if not np.isfinite(dose_lambda):
                raise ValueError("budget['dose_lambda'] must be finite")

            default_weights = {
                "vi_gradient": 0.4,
                "vi_local_std": 0.3,
                "dp_variance": 0.3,
            }
            merged_weights = dict(default_weights)
            raw_weights = budget_dict.get("weights", None)
            if raw_weights is not None:
                for key, value in dict(raw_weights).items():
                    if key not in default_weights:
                        raise ValueError(
                            f"Unsupported adaptive weight '{key}'. "
                            f"Supported: {', '.join(default_weights.keys())}"
                        )
                    merged_weights[key] = float(value)
            weight_sum = sum(max(0.0, float(v)) for v in merged_weights.values())
            if weight_sum <= 0:
                raise ValueError("At least one adaptive weight must be > 0")
            weights = {k: max(0.0, float(v)) / weight_sum for k, v in merged_weights.items()}

            prev_frame = int(self.frame_idx)
            if frame_idx != prev_frame:
                self.frame_idx = frame_idx
            try:
                vi = self._get_virtual_image_array().astype(np.float32, copy=False)
                grad_row, grad_col = np.gradient(vi)
                vi_gradient = np.hypot(grad_row, grad_col).astype(np.float32)
                mean_local = self._box_mean_map(vi, local_window)
                mean_sq_local = self._box_mean_map(vi * vi, local_window)
                vi_local_std = np.sqrt(np.maximum(mean_sq_local - mean_local * mean_local, 0.0)).astype(np.float32)
                dp_variance = self._dp_variance_map(frame_idx=frame_idx)
            finally:
                if frame_idx != prev_frame:
                    self.frame_idx = prev_frame

            utility = (
                weights["vi_gradient"] * self._normalize_score_map(vi_gradient)
                + weights["vi_local_std"] * self._normalize_score_map(vi_local_std)
                + weights["dp_variance"] * self._normalize_score_map(dp_variance)
            ).astype(np.float32)

            frame_dose = self._dose_map[frame_idx].astype(np.float32, copy=False)
            if float(frame_dose.max()) > 0:
                utility = utility - float(dose_lambda) * (frame_dose / float(frame_dose.max()))

            picks = self._select_spaced_topk(
                scores=utility,
                k=n_select,
                min_spacing=min_spacing,
                allowed_mask=allowed_mask,
                excluded_mask=np.zeros_like(allowed_mask, dtype=bool),
            )
            proposals = [(int(r), int(c)) for (r, c) in picks]
        elif strategy_key == "random":
            coords = np.argwhere(allowed_mask)
            if coords.size:
                seed = budget_dict.get("seed", None)
                rng = np.random.default_rng(None if seed is None else int(seed))
                n_take = min(n_select, int(coords.shape[0]))
                idx = rng.choice(coords.shape[0], size=n_take, replace=False)
                chosen = coords[idx]
                proposals = [(int(r), int(c)) for r, c in chosen]
        else:
            step = int(budget_dict.get("step", 1))
            if step < 1:
                raise ValueError("budget['step'] must be >= 1")
            bidirectional = bool(budget_dict.get("bidirectional", True))
            for row in range(0, self.shape_rows, step):
                cols = list(range(0, self.shape_cols, step))
                if bidirectional and ((row // step) % 2 == 1):
                    cols.reverse()
                for col in cols:
                    if allowed_mask[row, col]:
                        proposals.append((int(row), int(col)))
                        if len(proposals) >= n_select:
                            break
                if len(proposals) >= n_select:
                    break

        self._record_export_event(
            {
                "export_kind": "propose_next_points",
                "strategy": strategy_key,
                "frame_idx": int(frame_idx),
                "k_requested": int(k),
                "k_returned": int(len(proposals)),
            }
        )
        return proposals

    def evaluate_against_reference(
        self,
        reference: str | np.ndarray = "full_raster",
        metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate sparse-sampled reconstruction against a reference image.

        Parameters
        ----------
        reference : str or np.ndarray, default "full_raster"
            Reference target. ``"full_raster"`` uses the current full dataset
            and current ROI integration settings. Arrays are also accepted
            (virtual image or full diffraction stack; see method docs).
        metrics : list[str], optional
            Metric names to compute. Supported: ``"rmse"``, ``"nrmse"``,
            ``"mae"``, ``"psnr"``.

        Returns
        -------
        dict
            Evaluation summary including sampled fraction and metric values.
        """
        metric_names = (
            ["rmse", "nrmse", "mae", "psnr"]
            if metrics is None
            else [str(name).strip().lower() for name in metrics]
        )
        supported = {"rmse", "nrmse", "mae", "psnr"}
        unknown = [name for name in metric_names if name not in supported]
        if unknown:
            raise ValueError(f"Unsupported metrics: {unknown}. Supported: {sorted(supported)}")

        frame_idx = int(self.frame_idx if self.n_frames <= 1 else self._validate_sparse_frame_idx(self.frame_idx))
        points, values = self._extract_sparse_samples(frame_idx)
        if points.shape[0] == 0:
            raise ValueError("No sparse samples available for evaluation. Ingest points first.")

        reference_vi, reference_kind = self._resolve_reference_virtual_image(reference, frame_idx)
        reconstruction = self._idw_reconstruct(
            shape=(self.shape_rows, self.shape_cols),
            points=points,
            values=values,
            power=2.0,
            k_neighbors=16,
        )

        ref = np.asarray(reference_vi, dtype=np.float32)
        pred = np.asarray(reconstruction, dtype=np.float32)
        diff = pred - ref
        mse = float(np.mean(diff * diff))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(diff)))
        ref_range = float(ref.max() - ref.min()) + 1e-6
        nrmse = float(rmse / ref_range)
        peak = float(max(float(ref.max()), 1e-6))
        psnr = 120.0 if mse <= 1e-12 else float(20.0 * np.log10(peak) - 10.0 * np.log10(mse))

        metric_values = {
            "rmse": rmse,
            "nrmse": nrmse,
            "mae": mae,
            "psnr": psnr,
        }
        selected_metrics = {name: float(metric_values[name]) for name in metric_names}

        summary = {
            "reference_kind": reference_kind,
            "frame_idx": int(frame_idx),
            "n_sampled": int(points.shape[0]),
            "sampled_fraction": float(points.shape[0] / max(1, self.shape_rows * self.shape_cols)),
            "metrics": selected_metrics,
            "scan_shape": {"rows": int(self.shape_rows), "cols": int(self.shape_cols)},
            "detector_shape": {"rows": int(self.det_rows), "cols": int(self.det_cols)},
        }
        self._record_export_event(
            {
                "export_kind": "evaluate_against_reference",
                "reference_kind": reference_kind,
                "frame_idx": int(frame_idx),
                "n_sampled": int(points.shape[0]),
                "sampled_fraction": float(summary["sampled_fraction"]),
                "metrics": selected_metrics,
            }
        )
        return summary

    def export_session_bundle(
        self,
        path: str | pathlib.Path,
    ) -> pathlib.Path:
        """
        Export a reproducible session bundle for sparse/adaptive workflows.

        The bundle includes widget state, sparse-state arrays, a current view
        image with metadata, and the reproducibility report.

        Parameters
        ----------
        path : str or pathlib.Path
            Output directory for bundle files.

        Returns
        -------
        pathlib.Path
            Path to the bundle manifest JSON.
        """
        bundle_dir = pathlib.Path(path)
        bundle_dir.mkdir(parents=True, exist_ok=True)

        state_path = bundle_dir / "widget_state.json"
        self.save(state_path)

        sparse_state = self.get_sparse_state()
        sparse_npz_path = bundle_dir / "sparse_state.npz"
        np.savez_compressed(
            sparse_npz_path,
            mask=sparse_state["mask"],
            sampled_data=sparse_state["sampled_data"],
            dose_map=sparse_state["dose_map"],
        )

        sparse_points_path = bundle_dir / "sparse_points.json"
        sparse_points_payload = {
            **build_json_header("Show4DSTEM"),
            "format": "json",
            "export_kind": "sparse_points",
            "n_sampled": int(sparse_state["n_sampled"]),
            "sampled_points": sparse_state["sampled_points"],
        }
        sparse_points_path.write_text(json.dumps(sparse_points_payload, indent=2))

        image_path = bundle_dir / "current_all.png"
        image_written = self.save_image(
            image_path,
            view="all",
            include_metadata=True,
            include_overlays=True,
            include_scalebar=True,
        )
        image_meta_path = image_written.with_suffix(".json")

        report_path = self.save_reproducibility_report(bundle_dir / "reproducibility_report.json")

        manifest_path = bundle_dir / "session_bundle_manifest.json"
        manifest_payload = {
            **build_json_header("Show4DSTEM"),
            "format": "json",
            "export_kind": "session_bundle",
            "bundle_path": str(bundle_dir),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "session_id": self._export_session_id,
            "scan_shape": {"rows": int(self.shape_rows), "cols": int(self.shape_cols)},
            "detector_shape": {"rows": int(self.det_rows), "cols": int(self.det_cols)},
            "sparse_summary": {
                "n_sampled": int(sparse_state["n_sampled"]),
                "sampled_fraction": float(
                    sparse_state["n_sampled"] / max(1, self.shape_rows * self.shape_cols * self.n_frames)
                ),
                "total_dose": float(sparse_state["total_dose"]),
            },
            "files": {
                "state": str(state_path),
                "sparse_npz": str(sparse_npz_path),
                "sparse_points_json": str(sparse_points_path),
                "image": str(image_written),
                "image_metadata": str(image_meta_path),
                "reproducibility_report": str(report_path),
            },
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2))

        self._record_export_event(
            {
                "export_kind": "session_bundle",
                "n_sampled": int(sparse_state["n_sampled"]),
                "outputs": [
                    self._build_file_record(state_path),
                    self._build_file_record(sparse_npz_path),
                    self._build_file_record(sparse_points_path),
                    self._build_file_record(image_written, metadata_path=image_meta_path),
                    self._build_file_record(report_path),
                    self._build_file_record(manifest_path),
                ],
            }
        )
        return manifest_path

    def _normalize_score_map(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return np.zeros_like(arr, dtype=np.float32)
        vmin = float(np.percentile(arr, 1.0))
        vmax = float(np.percentile(arr, 99.0))
        if vmax <= vmin:
            return np.zeros_like(arr, dtype=np.float32)
        return np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0).astype(np.float32)

    def _box_mean_map(self, values: np.ndarray, window: int) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        win = int(window)
        if win <= 1:
            return arr.copy()
        if win % 2 == 0:
            win += 1
        pad = win // 2
        padded = np.pad(arr, ((pad, pad), (pad, pad)), mode="reflect")
        integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
        sums = (
            integral[win:, win:]
            - integral[:-win, win:]
            - integral[win:, :-win]
            + integral[:-win, :-win]
        )
        return (sums / float(win * win)).astype(np.float32)

    def _dp_variance_map(self, frame_idx: int | None = None) -> np.ndarray:
        if frame_idx is None or self.n_frames <= 1:
            data = self._frame_data
        else:
            idx = self._validate_sparse_frame_idx(frame_idx)
            data = self._data[idx]
        if data.ndim == 4:
            variance = data.var(dim=(2, 3), unbiased=False)
            return variance.detach().cpu().numpy().astype(np.float32, copy=False)
        variance = data.var(dim=(1, 2), unbiased=False)
        return variance.detach().cpu().numpy().reshape(self.shape_rows, self.shape_cols).astype(np.float32, copy=False)

    def _build_coarse_points(self, step: int, bidirectional: bool) -> list[tuple[int, int]]:
        points: list[tuple[int, int]] = []
        for r in range(0, self.shape_rows, step):
            cols = list(range(0, self.shape_cols, step))
            if bidirectional and ((r // step) % 2 == 1):
                cols.reverse()
            for c in cols:
                points.append((int(r), int(c)))
        return points

    def _select_spaced_topk(
        self,
        scores: np.ndarray,
        k: int,
        min_spacing: int,
        allowed_mask: np.ndarray,
        excluded_mask: np.ndarray,
    ) -> list[tuple[int, int]]:
        work = np.asarray(scores, dtype=np.float32).copy()
        work[~allowed_mask] = -np.inf
        work[excluded_mask] = -np.inf
        selected: list[tuple[int, int]] = []
        radius = max(0, int(min_spacing))

        for _ in range(int(max(0, k))):
            flat_idx = int(np.argmax(work))
            best_score = float(work.flat[flat_idx])
            if not np.isfinite(best_score):
                break
            row, col = np.unravel_index(flat_idx, work.shape)
            selected.append((int(row), int(col)))
            if radius == 0:
                work[row, col] = -np.inf
                continue
            r0 = max(0, row - radius)
            r1 = min(work.shape[0], row + radius + 1)
            c0 = max(0, col - radius)
            c1 = min(work.shape[1], col + radius + 1)
            rr, cc = np.ogrid[r0:r1, c0:c1]
            neighborhood = (rr - row) ** 2 + (cc - col) ** 2 <= radius ** 2
            block = work[r0:r1, c0:c1]
            block[neighborhood] = -np.inf
        return selected

    def _nearest_neighbor_order(
        self,
        points: list[tuple[int, int]],
        start: tuple[int, int] | None = None,
    ) -> list[tuple[int, int]]:
        remaining = [tuple(map(int, pt)) for pt in points]
        if not remaining:
            return []

        if start is None:
            current = remaining.pop(0)
        else:
            sr, sc = int(start[0]), int(start[1])
            start_idx = min(
                range(len(remaining)),
                key=lambda i: (remaining[i][0] - sr) ** 2 + (remaining[i][1] - sc) ** 2,
            )
            current = remaining.pop(start_idx)

        ordered = [current]
        while remaining:
            cr, cc = current
            next_idx = min(
                range(len(remaining)),
                key=lambda i: (remaining[i][0] - cr) ** 2 + (remaining[i][1] - cc) ** 2,
            )
            current = remaining.pop(next_idx)
            ordered.append(current)
        return ordered

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
        include_overlays : bool, optional
            Draw ROI/profile/crosshair overlays on exported panels.
            Defaults to ``export_include_overlays``.
        include_scalebar : bool, optional
            Draw panel scale bars on exported panels.
            Defaults to ``export_include_scalebar``.
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
        dpi_value = int(self.export_default_dpi if dpi is None else dpi)
        overlays_enabled = (
            bool(self.export_include_overlays)
            if include_overlays is None
            else bool(include_overlays)
        )
        scalebar_enabled = (
            bool(self.export_include_scalebar)
            if include_scalebar is None
            else bool(include_scalebar)
        )

        if dpi_value <= 0:
            raise ValueError(f"dpi must be > 0, got {dpi_value}")

        export_path.parent.mkdir(parents=True, exist_ok=True)

        prev_row, prev_col = self.pos_row, self.pos_col
        prev_frame = self.frame_idx
        meta_path: pathlib.Path | None = None
        export_row = int(self.pos_row)
        export_col = int(self.pos_col)
        export_frame = int(self.frame_idx)

        try:
            if frame_idx is not None:
                self.frame_idx = self._validate_frame_idx(frame_idx)
            if position is not None:
                row, col = self._validate_position(position)
                self.pos_row = row
                self.pos_col = col
            export_row = int(self.pos_row)
            export_col = int(self.pos_col)
            export_frame = int(self.frame_idx)

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

            if fmt == "png":
                image.save(export_path, format="PNG", dpi=(dpi_value, dpi_value))
            else:
                image.save(export_path, format="PDF", resolution=dpi_value)

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

        self._record_export_event(
            {
                "export_kind": "single_view_image",
                "view": view_key,
                "format": fmt,
                "position": {"row": export_row, "col": export_col},
                "frame_idx": export_frame,
                "include_overlays": bool(overlays_enabled),
                "include_scalebar": bool(scalebar_enabled),
                "dpi": int(dpi_value),
                "outputs": [
                    self._build_file_record(export_path, metadata_path=meta_path),
                ],
            }
        )
        return export_path

    def _build_preset_payload(self) -> dict[str, Any]:
        return {
            "detector": {
                "center_row": float(self.center_row),
                "center_col": float(self.center_col),
                "bf_radius": float(self.bf_radius),
                "roi_active": bool(self.roi_active),
                "roi_mode": self.roi_mode,
                "roi_center_row": float(self.roi_center_row),
                "roi_center_col": float(self.roi_center_col),
                "roi_radius": float(self.roi_radius),
                "roi_radius_inner": float(self.roi_radius_inner),
                "roi_width": float(self.roi_width),
                "roi_height": float(self.roi_height),
            },
            "vi_roi": {
                "mode": self.vi_roi_mode,
                "center_row": float(self.vi_roi_center_row),
                "center_col": float(self.vi_roi_center_col),
                "radius": float(self.vi_roi_radius),
                "width": float(self.vi_roi_width),
                "height": float(self.vi_roi_height),
            },
            "display": {
                "mask_dc": bool(self.mask_dc),
                "dp_colormap": self.dp_colormap,
                "vi_colormap": self.vi_colormap,
                "fft_colormap": self.fft_colormap,
                "dp_scale_mode": self.dp_scale_mode,
                "vi_scale_mode": self.vi_scale_mode,
                "fft_scale_mode": self.fft_scale_mode,
                "dp_power_exp": float(self.dp_power_exp),
                "vi_power_exp": float(self.vi_power_exp),
                "fft_power_exp": float(self.fft_power_exp),
                "dp_vmin_pct": float(self.dp_vmin_pct),
                "dp_vmax_pct": float(self.dp_vmax_pct),
                "vi_vmin_pct": float(self.vi_vmin_pct),
                "vi_vmax_pct": float(self.vi_vmax_pct),
                "fft_vmin_pct": float(self.fft_vmin_pct),
                "fft_vmax_pct": float(self.fft_vmax_pct),
                "fft_auto": bool(self.fft_auto),
                "show_fft": bool(self.show_fft),
                "dp_show_colorbar": bool(self.dp_show_colorbar),
                "profile_line": self.profile_line,
                "profile_width": int(self.profile_width),
            },
            "export": self._export_settings_metadata(),
        }

    def _apply_preset_payload(self, preset: dict[str, Any]) -> None:
        detector = preset.get("detector", {})
        vi_roi = preset.get("vi_roi", {})
        display = preset.get("display", {})
        export = preset.get("export", {})

        detector_map = {
            "center_row": "center_row",
            "center_col": "center_col",
            "bf_radius": "bf_radius",
            "roi_active": "roi_active",
            "roi_mode": "roi_mode",
            "roi_center_row": "roi_center_row",
            "roi_center_col": "roi_center_col",
            "roi_radius": "roi_radius",
            "roi_radius_inner": "roi_radius_inner",
            "roi_width": "roi_width",
            "roi_height": "roi_height",
        }
        for key, trait_name in detector_map.items():
            if key in detector and hasattr(self, trait_name):
                setattr(self, trait_name, detector[key])

        vi_roi_map = {
            "mode": "vi_roi_mode",
            "center_row": "vi_roi_center_row",
            "center_col": "vi_roi_center_col",
            "radius": "vi_roi_radius",
            "width": "vi_roi_width",
            "height": "vi_roi_height",
        }
        for key, trait_name in vi_roi_map.items():
            if key in vi_roi and hasattr(self, trait_name):
                setattr(self, trait_name, vi_roi[key])

        for key, value in display.items():
            if hasattr(self, key):
                setattr(self, key, value)

        export_map = {
            "default_view": "export_default_view",
            "default_format": "export_default_format",
            "include_overlays": "export_include_overlays",
            "include_scalebar": "export_include_scalebar",
            "dpi": "export_default_dpi",
        }
        for key, trait_name in export_map.items():
            if key in export and hasattr(self, trait_name):
                setattr(self, trait_name, export[key])

    def save_preset(
        self,
        name: str,
        path: str | pathlib.Path | None = None,
    ) -> dict[str, Any]:
        preset_name = str(name).strip()
        if not preset_name:
            raise ValueError("Preset name must be non-empty.")
        preset_key = preset_name.lower()

        payload = self._build_preset_payload()
        self._named_presets[preset_key] = payload

        if path is not None:
            out_path = pathlib.Path(path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            serialized = {
                **build_json_header("Show4DSTEM"),
                "format": "json",
                "export_kind": "widget_preset",
                "preset_name": preset_name,
                "preset": payload,
            }
            out_path.write_text(json.dumps(serialized, indent=2))

        return payload

    def load_preset(
        self,
        name: str,
        path: str | pathlib.Path | None = None,
        apply: bool = True,
    ) -> dict[str, Any]:
        preset_name = str(name).strip()
        preset_key = preset_name.lower()
        if path is not None:
            payload = json.loads(pathlib.Path(path).read_text())
            if not isinstance(payload, dict):
                raise ValueError("Preset file must contain a JSON object.")
            if "preset" in payload:
                preset = payload["preset"]
            else:
                preset = payload
            if not isinstance(preset, dict):
                raise ValueError("Preset payload must be a JSON object.")
            if preset_name:
                self._named_presets[preset_key] = preset
        else:
            if preset_key not in self._named_presets:
                raise ValueError(
                    f"Preset '{preset_name}' not found. Available: {', '.join(self.list_presets())}"
                )
            preset = self._named_presets[preset_key]

        if apply:
            self._apply_preset_payload(preset)
        return preset

    def apply_preset(self, name: str) -> Self:
        preset_name = str(name).strip().lower()
        if preset_name == "bf":
            self.roi_active = True
            self.roi_mode = "circle"
            self.roi_center_row = float(self.center_row)
            self.roi_center_col = float(self.center_col)
            self.roi_radius = float(max(1.0, self.bf_radius))
            return self
        if preset_name == "abf":
            self.roi_active = True
            self.roi_mode = "annular"
            self.roi_center_row = float(self.center_row)
            self.roi_center_col = float(self.center_col)
            self.roi_radius_inner = float(max(0.5, self.bf_radius * 0.5))
            self.roi_radius = float(max(1.0, self.bf_radius))
            return self
        if preset_name == "adf":
            self.roi_active = True
            self.roi_mode = "annular"
            self.roi_center_row = float(self.center_row)
            self.roi_center_col = float(self.center_col)
            self.roi_radius_inner = float(max(1.0, self.bf_radius))
            self.roi_radius = float(max(self.roi_radius_inner + 1.0, self.bf_radius * 2.0))
            return self
        if preset_name == "haadf":
            self.roi_active = True
            self.roi_mode = "annular"
            self.roi_center_row = float(self.center_row)
            self.roi_center_col = float(self.center_col)
            self.roi_radius_inner = float(max(1.0, self.bf_radius * 2.0))
            self.roi_radius = float(max(self.roi_radius_inner + 1.0, self.bf_radius * 4.0))
            return self

        self.load_preset(preset_name, apply=True)
        return self

    def _resolve_figure_template(self, template: str) -> tuple[str, list[str], bool]:
        key = str(template).strip().lower()
        mapping = {
            "dp_vi": (["diffraction", "virtual"], False),
            "dp_vi_fft": (["diffraction", "virtual", "fft"], False),
            "publication_dp_vi": (["diffraction", "virtual"], True),
            "publication_dp_vi_fft": (["diffraction", "virtual", "fft"], True),
        }
        if key not in mapping:
            raise ValueError(
                f"Unsupported template '{template}'. "
                f"Supported: {', '.join(self.list_figure_templates())}"
            )
        panels, publication = mapping[key]
        return key, panels, publication

    def save_figure(
        self,
        path: str | pathlib.Path,
        template: str = "dp_vi_fft",
        position: tuple[int, int] | None = None,
        frame_idx: int | None = None,
        format: str | None = None,
        include_metadata: bool = True,
        metadata_path: str | pathlib.Path | None = None,
        include_overlays: bool | None = None,
        include_scalebar: bool | None = None,
        restore_state: bool = True,
        dpi: int | None = None,
        title: str | None = None,
        annotations: dict[str, str] | None = None,
    ) -> pathlib.Path:
        from PIL import Image, ImageDraw, ImageFont

        export_path = pathlib.Path(path)
        template_key, panel_keys, publication_style = self._resolve_figure_template(template)
        fmt = self._resolve_export_format(export_path, format)
        dpi_value = int(self.export_default_dpi if dpi is None else dpi)
        overlays_enabled = (
            bool(self.export_include_overlays)
            if include_overlays is None
            else bool(include_overlays)
        )
        scalebar_enabled = (
            bool(self.export_include_scalebar)
            if include_scalebar is None
            else bool(include_scalebar)
        )
        if dpi_value <= 0:
            raise ValueError(f"dpi must be > 0, got {dpi_value}")

        export_path.parent.mkdir(parents=True, exist_ok=True)
        font = ImageFont.load_default()

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

            panel_images: list[Any] = []
            render_meta: dict[str, Any] = {}
            for panel_key in panel_keys:
                panel, panel_meta = self._render_panel_image(
                    panel_key,
                    include_overlays=overlays_enabled,
                    include_scalebar=scalebar_enabled,
                )
                panel_images.append(panel)
                render_meta[panel_key] = panel_meta

            gap = 24 if publication_style else 8
            padding = 24 if publication_style else 10
            label_height = 22 if publication_style else 0
            title_text = title
            if title_text is None and publication_style:
                if self.n_frames > 1:
                    title_text = f"4D-STEM Figure ({self.frame_dim_label} {self.frame_idx})"
                else:
                    title_text = "4D-STEM Figure"
            title_height = 34 if title_text else 0

            max_panel_height = max(panel.height for panel in panel_images)
            total_width = padding * 2 + sum(panel.width for panel in panel_images) + gap * (len(panel_images) - 1)
            total_height = padding * 2 + title_height + label_height + max_panel_height

            figure = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))
            draw = ImageDraw.Draw(figure, mode="RGBA")

            y_title = padding
            if title_text:
                draw.text((padding, y_title), title_text, fill=(0, 0, 0, 255), font=font)

            y_panels = padding + title_height
            if publication_style:
                y_panels += label_height

            panel_names = {
                "diffraction": "Diffraction",
                "virtual": "Virtual",
                "fft": "FFT",
            }
            annotation_map = annotations or {}

            x0 = padding
            for idx, panel in enumerate(panel_images):
                panel_key = panel_keys[idx]
                if publication_style:
                    draw.text(
                        (x0, padding + title_height),
                        panel_names.get(panel_key, panel_key),
                        fill=(0, 0, 0, 255),
                        font=font,
                    )

                figure.paste(panel, (x0, y_panels))

                if publication_style:
                    draw.rectangle(
                        [(x0, y_panels), (x0 + panel.width - 1, y_panels + panel.height - 1)],
                        outline=(80, 80, 80, 255),
                        width=1,
                    )

                if panel_key in annotation_map and str(annotation_map[panel_key]).strip():
                    text = str(annotation_map[panel_key]).strip()
                    text_bbox = draw.textbbox((0, 0), text, font=font)
                    text_w = text_bbox[2] - text_bbox[0]
                    text_h = text_bbox[3] - text_bbox[1]
                    tx = x0 + 8
                    ty = y_panels + 8
                    draw.rectangle(
                        [(tx - 4, ty - 3), (tx + text_w + 4, ty + text_h + 3)],
                        fill=(0, 0, 0, 180),
                    )
                    draw.text((tx, ty), text, fill=(255, 255, 255, 255), font=font)

                x0 += panel.width + gap

            if fmt == "png":
                figure.save(export_path, format="PNG", dpi=(dpi_value, dpi_value))
            else:
                figure.save(export_path, format="PDF", resolution=dpi_value)

            if include_metadata:
                meta_path = (
                    pathlib.Path(metadata_path)
                    if metadata_path is not None
                    else export_path.with_suffix(".json")
                )
                metadata = self._build_image_export_metadata(
                    export_path=export_path,
                    view_key="figure",
                    fmt=fmt,
                    render_meta=render_meta,
                    include_overlays=overlays_enabled,
                    include_scalebar=scalebar_enabled,
                    export_kind="figure_template",
                    extra={
                        "template": template_key,
                        "panels": panel_keys,
                        "publication_style": bool(publication_style),
                        "title": title_text or "",
                        "annotations": annotation_map,
                        "dpi": int(dpi_value),
                    },
                )
                meta_path.write_text(json.dumps(metadata, indent=2))
        finally:
            if restore_state:
                self.frame_idx = prev_frame
                self.pos_row = prev_row
                self.pos_col = prev_col

        self._record_export_event(
            {
                "export_kind": "figure_template",
                "template": template_key,
                "format": fmt,
                "dpi": int(dpi_value),
                "include_overlays": bool(overlays_enabled),
                "include_scalebar": bool(scalebar_enabled),
                "outputs": [
                    self._build_file_record(export_path, metadata_path=meta_path),
                ],
            }
        )
        return export_path

    def _resolve_frame_sequence(
        self,
        frame_indices: list[int] | None,
        frame_range: tuple[int, int] | None,
    ) -> list[int]:
        if frame_indices is not None and frame_range is not None:
            raise ValueError("Use either frame_indices or frame_range, not both.")

        if frame_indices is not None:
            if len(frame_indices) == 0:
                raise ValueError("frame_indices cannot be empty.")
            return [self._validate_frame_idx(idx) for idx in frame_indices]

        if frame_range is not None:
            if len(frame_range) != 2:
                raise ValueError("frame_range must be a (start, end) tuple.")
            start, end = int(frame_range[0]), int(frame_range[1])
            if start > end:
                raise ValueError("frame_range start must be <= end.")
            return [self._validate_frame_idx(idx) for idx in range(start, end + 1)]

        return [int(i) for i in range(self.n_frames)]

    def _resolve_position_sequence(
        self,
        mode: str,
        path_points: list[tuple[int, int]] | None,
        raster_step: int,
        raster_bidirectional: bool,
    ) -> list[tuple[int, int]]:
        if mode == "path":
            points = self._path_points if path_points is None else path_points
            if not points:
                raise ValueError(
                    "Path mode requires points via set_path(...) or path_points=..."
                )
            return [self._validate_position((int(r), int(c))) for r, c in points]

        if mode == "raster":
            step = int(raster_step)
            if step < 1:
                raise ValueError("raster_step must be >= 1")
            points: list[tuple[int, int]] = []
            for r in range(0, self.shape_rows, step):
                cols = list(range(0, self.shape_cols, step))
                if raster_bidirectional and ((r // step) % 2 == 1):
                    cols.reverse()
                for c in cols:
                    points.append((int(r), int(c)))
            return points

        raise ValueError(f"Unsupported position sequence mode '{mode}'")

    def suggest_adaptive_path(
        self,
        coarse_step: int = 4,
        target_fraction: float = 0.25,
        min_spacing: int = 2,
        include_coarse: bool = True,
        coarse_bidirectional: bool = True,
        local_window: int = 5,
        dose_lambda: float = 0.25,
        weights: dict[str, float] | None = None,
        roi_mask: np.ndarray | None = None,
        update_widget_path: bool = True,
        interval_ms: int | None = None,
        loop: bool = False,
        autoplay: bool = False,
        return_maps: bool = False,
    ) -> dict[str, Any]:
        """
        Suggest a sparse adaptive scan path using coarse-to-fine utility ranking.

        The planner computes utility from current virtual-image and diffraction
        statistics, then selects spatially distributed high-utility points.

        Parameters
        ----------
        coarse_step : int, default 4
            Spacing of the initial coarse grid.
        target_fraction : float, default 0.25
            Target total sampled fraction of scan positions in (0, 1].
        min_spacing : int, default 2
            Minimum pixel spacing between selected dense points.
        include_coarse : bool, default True
            If True, include coarse-grid points in the returned path.
        coarse_bidirectional : bool, default True
            Use snake ordering for coarse-grid traversal.
        local_window : int, default 5
            Window size for local-std utility component.
        dose_lambda : float, default 0.25
            Penalty weight for re-sampling coarse points.
        weights : dict[str, float], optional
            Utility weights for keys: ``vi_gradient``, ``vi_local_std``, ``dp_variance``.
        roi_mask : np.ndarray, optional
            Optional boolean mask of shape ``scan_shape`` restricting dense picks.
        update_widget_path : bool, default True
            If True, calls ``set_path(...)`` with the suggested path.
        interval_ms : int, optional
            Path interval when ``update_widget_path=True``.
        loop : bool, default False
            Path looping behavior when ``update_widget_path=True``.
        autoplay : bool, default False
            Start playback immediately when ``update_widget_path=True``.
        return_maps : bool, default False
            If True, include utility component maps in the returned dict.

        Returns
        -------
        dict
            Planning result with coarse points, dense points, and final path.
        """
        step = int(coarse_step)
        if step < 1:
            raise ValueError(f"coarse_step must be >= 1, got {coarse_step}")

        frac = float(target_fraction)
        if frac <= 0 or frac > 1:
            raise ValueError(f"target_fraction must be in (0, 1], got {target_fraction}")

        spacing = int(min_spacing)
        if spacing < 0:
            raise ValueError(f"min_spacing must be >= 0, got {min_spacing}")

        if local_window < 1:
            raise ValueError(f"local_window must be >= 1, got {local_window}")

        if not np.isfinite(float(dose_lambda)):
            raise ValueError("dose_lambda must be finite")

        default_weights = {
            "vi_gradient": 0.4,
            "vi_local_std": 0.3,
            "dp_variance": 0.3,
        }
        merged_weights = dict(default_weights)
        if weights is not None:
            for key, value in weights.items():
                if key not in default_weights:
                    raise ValueError(
                        f"Unsupported utility weight '{key}'. "
                        f"Supported: {', '.join(default_weights.keys())}"
                    )
                merged_weights[key] = float(value)

        weight_sum = sum(max(0.0, float(v)) for v in merged_weights.values())
        if weight_sum <= 0:
            raise ValueError("At least one utility weight must be > 0.")
        normalized_weights = {
            key: max(0.0, float(value)) / weight_sum
            for key, value in merged_weights.items()
        }

        n_total = int(self.shape_rows * self.shape_cols)
        target_count = int(max(1, round(frac * n_total)))

        coarse_points = self._build_coarse_points(step=step, bidirectional=bool(coarse_bidirectional))
        coarse_count = len(coarse_points) if include_coarse else 0
        if include_coarse and target_count < coarse_count:
            raise ValueError(
                f"target_fraction={target_fraction} gives {target_count} points, "
                f"but coarse grid already has {coarse_count}. "
                "Increase target_fraction or coarse_step."
            )
        dense_count = target_count - coarse_count if include_coarse else target_count
        dense_count = max(0, int(dense_count))

        vi = self._get_virtual_image_array().astype(np.float32, copy=False)
        grad_row, grad_col = np.gradient(vi)
        vi_gradient = np.hypot(grad_row, grad_col).astype(np.float32)

        mean_local = self._box_mean_map(vi, local_window)
        mean_sq_local = self._box_mean_map(vi * vi, local_window)
        variance_local = np.maximum(mean_sq_local - mean_local * mean_local, 0.0)
        vi_local_std = np.sqrt(variance_local).astype(np.float32)

        dp_variance = self._dp_variance_map()

        grad_score = self._normalize_score_map(vi_gradient)
        local_std_score = self._normalize_score_map(vi_local_std)
        dp_var_score = self._normalize_score_map(dp_variance)

        utility = (
            normalized_weights["vi_gradient"] * grad_score
            + normalized_weights["vi_local_std"] * local_std_score
            + normalized_weights["dp_variance"] * dp_var_score
        ).astype(np.float32)

        dose_penalty = np.zeros_like(utility, dtype=np.float32)
        for row, col in coarse_points:
            dose_penalty[int(row), int(col)] = 1.0
        utility = utility - float(dose_lambda) * dose_penalty

        allowed_mask = np.ones((self.shape_rows, self.shape_cols), dtype=bool)
        if roi_mask is not None:
            mask = np.asarray(roi_mask)
            if mask.shape != (self.shape_rows, self.shape_cols):
                raise ValueError(
                    f"roi_mask shape {mask.shape} does not match scan_shape "
                    f"({self.shape_rows}, {self.shape_cols})"
                )
            allowed_mask &= mask.astype(bool)

        excluded_mask = np.zeros_like(allowed_mask, dtype=bool)
        for row, col in coarse_points:
            excluded_mask[int(row), int(col)] = True

        dense_points = self._select_spaced_topk(
            scores=utility,
            k=dense_count,
            min_spacing=spacing,
            allowed_mask=allowed_mask,
            excluded_mask=excluded_mask,
        )

        start_point = coarse_points[-1] if include_coarse and coarse_points else None
        dense_path = self._nearest_neighbor_order(dense_points, start=start_point)
        path_points = list(coarse_points) + dense_path if include_coarse else dense_path

        if update_widget_path and path_points:
            interval_value = int(self.path_interval_ms if interval_ms is None else interval_ms)
            if interval_value < 1:
                raise ValueError(f"interval_ms must be >= 1, got {interval_value}")
            self.set_path(
                points=path_points,
                interval_ms=interval_value,
                loop=bool(loop),
                autoplay=bool(autoplay),
            )

        result: dict[str, Any] = {
            "target_fraction": float(frac),
            "target_count": int(target_count),
            "coarse_step": int(step),
            "coarse_count": int(len(coarse_points)),
            "dense_count": int(len(dense_points)),
            "path_count": int(len(path_points)),
            "weights": normalized_weights,
            "dose_lambda": float(dose_lambda),
            "coarse_points": coarse_points,
            "dense_points": dense_points,
            "path_points": path_points,
            "selected_fraction": float(len(path_points) / max(1, n_total)),
        }
        if return_maps:
            result["utility_map"] = utility
            result["utility_components"] = {
                "vi_gradient": grad_score,
                "vi_local_std": local_std_score,
                "dp_variance": dp_var_score,
                "dose_penalty": dose_penalty,
            }

        self._record_export_event(
            {
                "export_kind": "adaptive_path_suggestion",
                "target_fraction": float(frac),
                "target_count": int(target_count),
                "coarse_step": int(step),
                "coarse_count": int(len(coarse_points)),
                "dense_count": int(len(dense_points)),
                "path_count": int(len(path_points)),
                "selected_fraction": float(len(path_points) / max(1, n_total)),
                "weights": normalized_weights,
                "dose_lambda": float(dose_lambda),
            }
        )
        return result

    def save_sequence(
        self,
        output_dir: str | pathlib.Path,
        mode: str = "path",
        view: str | None = None,
        format: str | None = None,
        include_metadata: bool = True,
        include_overlays: bool | None = None,
        include_scalebar: bool | None = None,
        frame_idx: int | None = None,
        position: tuple[int, int] | None = None,
        path_points: list[tuple[int, int]] | None = None,
        raster_step: int = 1,
        raster_bidirectional: bool = False,
        frame_indices: list[int] | None = None,
        frame_range: tuple[int, int] | None = None,
        filename_prefix: str | None = None,
        manifest_name: str = "save_sequence_manifest.json",
        restore_state: bool = True,
        dpi: int | None = None,
    ) -> pathlib.Path:
        output_root = pathlib.Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        mode_key = str(mode).strip().lower()
        if mode_key not in {"path", "raster", "frames"}:
            raise ValueError("mode must be one of: path, raster, frames")

        view_key = self._validate_export_view(view)
        fmt = self._resolve_export_format(pathlib.Path(f"sequence.{self.export_default_format}"), format or self.export_default_format)
        dpi_value = int(self.export_default_dpi if dpi is None else dpi)
        overlays_enabled = (
            bool(self.export_include_overlays)
            if include_overlays is None
            else bool(include_overlays)
        )
        scalebar_enabled = (
            bool(self.export_include_scalebar)
            if include_scalebar is None
            else bool(include_scalebar)
        )
        if dpi_value <= 0:
            raise ValueError(f"dpi must be > 0, got {dpi_value}")

        export_rows: list[dict[str, Any]] = []
        prefix = (
            str(filename_prefix).strip()
            if filename_prefix is not None and str(filename_prefix).strip()
            else f"{mode_key}_{view_key}"
        )

        prev_row, prev_col = self.pos_row, self.pos_col
        prev_frame = self.frame_idx
        frame_for_paths = self._validate_frame_idx(frame_idx) if frame_idx is not None else int(self.frame_idx)

        if mode_key == "frames":
            row, col = self._validate_position(position)
            frames = self._resolve_frame_sequence(frame_indices, frame_range)
            jobs = [
                {"row": int(row), "col": int(col), "frame_idx": int(fi)}
                for fi in frames
            ]
        else:
            positions = self._resolve_position_sequence(
                mode=mode_key,
                path_points=path_points,
                raster_step=raster_step,
                raster_bidirectional=raster_bidirectional,
            )
            jobs = [
                {"row": int(r), "col": int(c), "frame_idx": int(frame_for_paths)}
                for r, c in positions
            ]

        try:
            for idx, job in enumerate(jobs):
                row = int(job["row"])
                col = int(job["col"])
                fr = int(job["frame_idx"])
                basename = (
                    f"{prefix}_{idx:04d}_f{fr:04d}_r{row:04d}_c{col:04d}.{fmt}"
                )
                out_path = output_root / basename
                out_meta = out_path.with_suffix(".json") if include_metadata else None

                self.save_image(
                    out_path,
                    view=view_key,
                    position=(row, col),
                    frame_idx=fr,
                    format=fmt,
                    include_metadata=include_metadata,
                    metadata_path=out_meta,
                    include_overlays=overlays_enabled,
                    include_scalebar=scalebar_enabled,
                    restore_state=False,
                    dpi=dpi_value,
                )

                record = {
                    "index": int(idx),
                    "row": row,
                    "col": col,
                    "frame_idx": fr,
                }
                record.update(self._build_file_record(out_path, metadata_path=out_meta, index=idx))
                export_rows.append(record)
        finally:
            if restore_state:
                self.frame_idx = prev_frame
                self.pos_row = prev_row
                self.pos_col = prev_col

        manifest_path = output_root / str(manifest_name)
        manifest_payload = {
            **build_json_header("Show4DSTEM"),
            "format": "json",
            "export_kind": "sequence_batch",
            "mode": mode_key,
            "view": view_key,
            "image_format": fmt,
            "output_dir": str(output_root),
            "filename_prefix": prefix,
            "n_exports": int(len(export_rows)),
            "include_overlays": bool(overlays_enabled),
            "include_scalebar": bool(scalebar_enabled),
            "dpi": int(dpi_value),
            "scan_shape": {"rows": int(self.shape_rows), "cols": int(self.shape_cols)},
            "detector_shape": {"rows": int(self.det_rows), "cols": int(self.det_cols)},
            "exports": export_rows,
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2))

        manifest_record = self._build_file_record(manifest_path)
        self._record_export_event(
            {
                "export_kind": "sequence_batch",
                "mode": mode_key,
                "view": view_key,
                "format": fmt,
                "n_exports": int(len(export_rows)),
                "include_overlays": bool(overlays_enabled),
                "include_scalebar": bool(scalebar_enabled),
                "dpi": int(dpi_value),
                "outputs": [manifest_record],
            }
        )
        return manifest_path

    def save_reproducibility_report(
        self,
        path: str | pathlib.Path,
    ) -> pathlib.Path:
        report_path = pathlib.Path(path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **build_json_header("Show4DSTEM"),
            "format": "json",
            "export_kind": "reproducibility_report",
            "session_id": self._export_session_id,
            "session_started_utc": self._export_session_started_utc,
            "report_generated_utc": datetime.now(timezone.utc).isoformat(),
            "scan_shape": {"rows": int(self.shape_rows), "cols": int(self.shape_cols)},
            "detector_shape": {"rows": int(self.det_rows), "cols": int(self.det_cols)},
            "n_exports": int(len(self._export_log)),
            "exports": self._export_log,
        }
        report_path.write_text(json.dumps(payload, indent=2))
        return report_path

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        mode = self.dp_scale_mode
        scaled = self._apply_scale_mode(frame, mode, self.dp_power_exp)
        fmin = float(scaled.min())
        fmax = float(scaled.max())
        fmin, fmax = self._slider_range(fmin, fmax, self.dp_vmin_pct, self.dp_vmax_pct)
        if fmax > fmin:
            return np.clip((scaled - fmin) / (fmax - fmin) * 255, 0, 255).astype(np.uint8)
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
            "path_points": [{"row": int(row), "col": int(col)} for row, col in self._path_points],
            "frame_idx": int(self.frame_idx),
            "n_frames_total": int(self.n_frames),
            "scan_shape": {"rows": int(self.shape_rows), "cols": int(self.shape_cols)},
            "detector_shape": {"rows": int(self.det_rows), "cols": int(self.det_cols)},
            "calibration": {
                "pixel_size_angstrom": float(self.pixel_size),
                "pixel_size_unit": "Å/px",
                "k_pixel_size": float(self.k_pixel_size),
                "k_pixel_size_unit": "mrad/px" if self.k_calibrated else "px/px",
                "k_calibrated": bool(self.k_calibrated),
                "center_row": float(self.center_row),
                "center_col": float(self.center_col),
                "bf_radius": float(self.bf_radius),
            },
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
        # Get frame as tensor (stays on device)
        data = self._frame_data
        if data.ndim == 3:
            idx = self.pos_row * self.shape_cols + self.pos_col
            frame = data[idx]
        else:
            frame = data[self.pos_row, self.pos_col]

        # Compute stats from frame (optionally mask DC component)
        if self.mask_dc and self.det_rows > 3 and self.det_cols > 3:
            # Mask center 3x3 region for stats using detected center (not geometric center)
            cr = int(round(self.center_row))
            cc = int(round(self.center_col))
            cr = max(1, min(self.det_rows - 2, cr))
            cc = max(1, min(self.det_cols - 2, cc))
            mask = torch.ones_like(frame, dtype=torch.bool)
            mask[cr-1:cr+2, cc-1:cc+2] = False
            masked_vals = frame[mask]
            self.dp_stats = [
                float(masked_vals.mean()),
                float(masked_vals.min()),
                float(masked_vals.max()),
                float(masked_vals.std()),
            ]
        else:
            self.dp_stats = [
                float(frame.mean()),
                float(frame.min()),
                float(frame.max()),
                float(frame.std()),
            ]

        # Convert to numpy only for sending bytes to frontend
        self.frame_bytes = frame.cpu().numpy().astype(np.float32).tobytes()

    def _on_roi_change(self, change=None):
        """Recompute virtual image when individual ROI params change.

        High-frequency drag updates use the compound roi_center trait instead.
        """
        if not self.roi_active:
            return
        self._compute_virtual_image_from_roi()

    def _on_roi_center_change(self, change=None):
        """Handle batched roi_center updates from JS (single observer for row+col).

        This is the fast path for drag operations. JS sends [row, col] as a single
        compound trait, so only one observer fires per mouse move.
        """
        if not self.roi_active:
            return
        if change and "new" in change:
            row, col = change["new"]
            # Sync to individual traits (without triggering _on_roi_change observers)
            self.unobserve(self._on_roi_change, names=["roi_center_col", "roi_center_row"])
            self.roi_center_row = row
            self.roi_center_col = col
            self.observe(self._on_roi_change, names=["roi_center_col", "roi_center_row"])
        self._compute_virtual_image_from_roi()

    def _on_vi_roi_change(self, change=None):
        """Compute summed DP when VI ROI changes."""
        if self.vi_roi_mode == "off":
            self.summed_dp_bytes = b""
            self.summed_dp_count = 0
            return
        self._compute_summed_dp_from_vi_roi()

    def _compute_summed_dp_from_vi_roi(self):
        """Sum diffraction patterns from positions inside VI ROI (PyTorch)."""
        # Create mask in scan space using cached coordinates
        if self.vi_roi_mode == "circle":
            mask = (self._scan_row_coords - self.vi_roi_center_row) ** 2 + (self._scan_col_coords - self.vi_roi_center_col) ** 2 <= self.vi_roi_radius ** 2
        elif self.vi_roi_mode == "square":
            half_size = self.vi_roi_radius
            mask = (torch.abs(self._scan_row_coords - self.vi_roi_center_row) <= half_size) & (torch.abs(self._scan_col_coords - self.vi_roi_center_col) <= half_size)
        elif self.vi_roi_mode == "rect":
            half_w = self.vi_roi_width / 2
            half_h = self.vi_roi_height / 2
            mask = (torch.abs(self._scan_row_coords - self.vi_roi_center_row) <= half_h) & (torch.abs(self._scan_col_coords - self.vi_roi_center_col) <= half_w)
        else:
            return

        # Count positions in mask
        n_positions = int(mask.sum())
        if n_positions == 0:
            self.summed_dp_bytes = b""
            self.summed_dp_count = 0
            return

        self.summed_dp_count = n_positions

        # Compute average DP using masked sum (vectorized)
        data = self._frame_data
        if data.ndim == 4:
            # (scan_rows, scan_cols, det_rows, det_cols) - sum over masked scan positions
            avg_dp = data[mask].mean(dim=0)
        else:
            # Flattened: (N, det_rows, det_cols) - need to convert mask indices
            flat_indices = torch.nonzero(mask.flatten(), as_tuple=True)[0]
            avg_dp = data[flat_indices].mean(dim=0)

        # Send raw float32 (consistent with other data paths — JS handles normalization)
        self.summed_dp_bytes = avg_dp.cpu().numpy().astype(np.float32).tobytes()

    def _create_circular_mask(self, cx: float, cy: float, radius: float):
        """Create circular mask (boolean tensor on device)."""
        mask = (self._det_col_coords - cx) ** 2 + (self._det_row_coords - cy) ** 2 <= radius ** 2
        return mask

    def _create_square_mask(self, cx: float, cy: float, half_size: float):
        """Create square mask (boolean tensor on device)."""
        mask = (torch.abs(self._det_col_coords - cx) <= half_size) & (torch.abs(self._det_row_coords - cy) <= half_size)
        return mask

    def _create_annular_mask(
        self, cx: float, cy: float, inner: float, outer: float
    ):
        """Create annular (donut) mask (boolean tensor on device)."""
        dist_sq = (self._det_col_coords - cx) ** 2 + (self._det_row_coords - cy) ** 2
        mask = (dist_sq >= inner ** 2) & (dist_sq <= outer ** 2)
        return mask

    def _create_rect_mask(self, cx: float, cy: float, half_width: float, half_height: float):
        """Create rectangular mask (boolean tensor on device)."""
        mask = (torch.abs(self._det_col_coords - cx) <= half_width) & (torch.abs(self._det_row_coords - cy) <= half_height)
        return mask

    def _precompute_common_virtual_images(self):
        """Pre-compute BF/ABF/ADF virtual images for instant preset switching."""
        cx, cy, bf = self.center_col, self.center_row, self.bf_radius
        # Cache (bytes, stats, min, max) for each preset
        bf_arr = self._fast_masked_sum(self._create_circular_mask(cx, cy, bf))
        abf_arr = self._fast_masked_sum(self._create_annular_mask(cx, cy, bf * 0.5, bf))
        adf_arr = self._fast_masked_sum(self._create_annular_mask(cx, cy, bf, bf * 4.0))

        self._cached_bf_virtual = (
            self._to_float32_bytes(bf_arr, update_vi_stats=False),
            [float(bf_arr.mean()), float(bf_arr.min()), float(bf_arr.max()), float(bf_arr.std())],
            float(bf_arr.min()), float(bf_arr.max())
        )
        self._cached_abf_virtual = (
            self._to_float32_bytes(abf_arr, update_vi_stats=False),
            [float(abf_arr.mean()), float(abf_arr.min()), float(abf_arr.max()), float(abf_arr.std())],
            float(abf_arr.min()), float(abf_arr.max())
        )
        self._cached_adf_virtual = (
            self._to_float32_bytes(adf_arr, update_vi_stats=False),
            [float(adf_arr.mean()), float(adf_arr.min()), float(adf_arr.max()), float(adf_arr.std())],
            float(adf_arr.min()), float(adf_arr.max())
        )

    def _get_cached_preset(self) -> tuple[bytes, list[float], float, float] | None:
        """Check if current ROI matches a cached preset and return (bytes, stats, min, max) tuple."""
        # Must be centered on detector center
        if abs(self.roi_center_col - self.center_col) >= 1 or abs(self.roi_center_row - self.center_row) >= 1:
            return None

        bf = self.bf_radius

        # BF: circle at bf_radius
        if (self.roi_mode == "circle" and abs(self.roi_radius - bf) < 1):
            return self._cached_bf_virtual

        # ABF: annular at 0.5*bf to bf
        if (self.roi_mode == "annular" and
            abs(self.roi_radius_inner - bf * 0.5) < 1 and
            abs(self.roi_radius - bf) < 1):
            return self._cached_abf_virtual

        # ADF: annular at bf to 4*bf (combines LAADF + HAADF)
        if (self.roi_mode == "annular" and
            abs(self.roi_radius_inner - bf) < 1 and
            abs(self.roi_radius - bf * 4.0) < 1):
            return self._cached_adf_virtual

        return None

    def _fast_masked_sum(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute masked sum using PyTorch.

        Uses sparse indexing for small masks (<20% coverage) which is faster
        because it only processes non-zero pixels:
        - r=10 (1%): ~0.8ms (sparse) vs ~13ms (full)
        - r=30 (8%): ~4ms (sparse) vs ~13ms (full)

        For large masks (≥20%), uses full tensordot which has constant ~13ms.
        """
        data = self._frame_data
        mask_float = mask.float()
        n_det = self._det_shape[0] * self._det_shape[1]
        n_nonzero = int(mask.sum())
        coverage = n_nonzero / n_det

        if coverage < SPARSE_MASK_THRESHOLD:
            # Sparse: faster for small masks
            indices = torch.nonzero(mask_float.flatten(), as_tuple=True)[0]
            n_scan = self._scan_shape[0] * self._scan_shape[1]
            data_flat = data.reshape(n_scan, n_det)
            result = data_flat[:, indices].sum(dim=1).reshape(self._scan_shape)
        else:
            # Tensordot: faster for large masks
            # Reshape to 4D if needed (3D flattened data)
            if data.ndim == 3:
                data_4d = data.reshape(self._scan_shape[0], self._scan_shape[1], *self._det_shape)
            else:
                data_4d = data
            result = torch.tensordot(data_4d, mask_float, dims=([2, 3], [0, 1]))

        return result

    def _to_float32_bytes(self, arr: torch.Tensor, update_vi_stats: bool = True) -> bytes:
        """Convert tensor to float32 bytes."""
        # Compute min/max (fast on GPU)
        vmin = float(arr.min())
        vmax = float(arr.max())

        # Only update traits when requested (avoids side effects during precomputation)
        if update_vi_stats:
            self.vi_data_min = vmin
            self.vi_data_max = vmax
            self.vi_stats = [float(arr.mean()), vmin, vmax, float(arr.std())]

        return arr.cpu().numpy().astype(np.float32).tobytes()

    def _compute_virtual_image_from_roi(self):
        """Compute virtual image based on ROI mode."""
        cached = self._get_cached_preset()
        if cached is not None:
            # Cached preset returns (bytes, stats, min, max) tuple
            vi_bytes, vi_stats, vi_min, vi_max = cached
            self.virtual_image_bytes = vi_bytes
            self.vi_stats = vi_stats
            self.vi_data_min = vi_min
            self.vi_data_max = vi_max
            return

        cx, cy = self.roi_center_col, self.roi_center_row

        if self.roi_mode == "circle" and self.roi_radius > 0:
            mask = self._create_circular_mask(cx, cy, self.roi_radius)
        elif self.roi_mode == "square" and self.roi_radius > 0:
            mask = self._create_square_mask(cx, cy, self.roi_radius)
        elif self.roi_mode == "annular" and self.roi_radius > 0:
            mask = self._create_annular_mask(cx, cy, self.roi_radius_inner, self.roi_radius)
        elif self.roi_mode == "rect" and self.roi_width > 0 and self.roi_height > 0:
            mask = self._create_rect_mask(cx, cy, self.roi_width / 2, self.roi_height / 2)
        else:
            # Point mode: single-pixel indexing
            row = int(max(0, min(round(cy), self._det_shape[0] - 1)))
            col = int(max(0, min(round(cx), self._det_shape[1] - 1)))
            data = self._frame_data
            if data.ndim == 4:
                virtual_image = data[:, :, row, col]
            else:
                virtual_image = data[:, row, col].reshape(self._scan_shape)
            self.virtual_image_bytes = self._to_float32_bytes(virtual_image)
            return

        self.virtual_image_bytes = self._to_float32_bytes(self._fast_masked_sum(mask))


bind_tool_runtime_api(Show4DSTEM, "Show4DSTEM")
