"""
show3d: Interactive 3D stack viewer widget with advanced features.

For viewing a stack of 2D images (e.g., defocus sweep, time series, z-stack, movies).
Includes playback controls, statistics, ROI selection, FFT, and more.
"""

import json
import pathlib
from enum import Enum
from typing import Self

import anywidget
import numpy as np
import traitlets

from quantem.widget.array_utils import to_numpy
from quantem.widget.io import IO, IOResult
from quantem.widget.json_state import build_json_header, resolve_widget_version, save_state_file, unwrap_state_payload
from quantem.widget.tool_parity import (
    bind_tool_runtime_api,
    build_tool_groups,
    normalize_tool_groups,
)

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

class Colormap(str, Enum):
    """Available colormaps for image display."""

    INFERNO = "inferno"
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    MAGMA = "magma"
    HOT = "hot"
    GRAY = "gray"

    def __str__(self) -> str:
        return self.value


class Show3D(anywidget.AnyWidget):
    """
    Interactive 3D stack viewer with advanced features for electron microscopy.

    View a stack of 2D images along a specific dimension (e.g., defocus sweep,
    time series, depth stack, in-situ movies). Includes playback controls,
    statistics panel, ROI selection, FFT view, and more.

    Parameters
    ----------
    data : array_like
        3D array of shape (N, height, width) where N is the stack dimension.
    labels : list of str, optional
        Labels for each slice (e.g., ["C10=-500nm", "C10=-400nm", ...]).
        If None, uses slice indices.
    title : str, optional
        Title to display above the image.
    cmap : str or Colormap, default Colormap.MAGMA
        Colormap name. Use Colormap enum (Colormap.MAGMA, Colormap.VIRIDIS, etc.)
        or string ("magma", "viridis", "gray", "inferno", "plasma").
    vmin : float, optional
        Minimum value for colormap. If None, uses data min.
    vmax : float, optional
        Maximum value for colormap. If None, uses data max.
    pixel_size : float, optional
        Pixel size in Å for scale bar display.
    log_scale : bool, default False
        Use log scale for intensity mapping.
    auto_contrast : bool, default False
        Use percentile-based contrast (ignores vmin/vmax).
    percentile_low : float, default 1.0
        Lower percentile for auto-contrast.
    percentile_high : float, default 99.0
        Upper percentile for auto-contrast.
    fps : float, default 5.0
        Frames per second for playback.
    timestamps : list of float, optional
        Timestamps for each frame (e.g., seconds or dose values).
    timestamp_unit : str, default "s"
        Unit for timestamps (e.g., "s", "ms", "e/A2").
    disabled_tools : list of str, optional
        Tool groups to lock while still showing controls. Supported:
        ``"display"``, ``"histogram"``, ``"stats"``, ``"playback"``,
        ``"view"``, ``"export"``, ``"roi"``, ``"profile"``, ``"all"``.
        ``"navigation"`` is accepted as an alias of ``"playback"``.
    disable_* : bool, optional
        Convenience flags mirroring ``disabled_tools``. Includes
        ``disable_navigation`` as an alias of ``disable_playback``.
    hidden_tools : list of str, optional
        Tool groups to hide from the UI. Uses the same keys as
        ``disabled_tools``.
    hide_* : bool, optional
        Convenience flags mirroring ``disable_*`` for ``hidden_tools``.

    Examples
    --------
    >>> import numpy as np
    >>> from quantem.widget import Show3D
    >>>
    >>> # View defocus sweep
    >>> labels = [f"C10={c10:.0f}nm" for c10 in np.linspace(-500, -200, 12)]
    >>> Show3D(stack, labels=labels, title="Defocus Sweep")
    >>>
    >>> # View in-situ movie with timestamps
    >>> times = np.arange(100) * 0.1  # 100 frames at 10 fps
    >>> Show3D(movie, timestamps=times, timestamp_unit="s", fps=30)
    >>>
    >>> # With scale bar
    >>> Show3D(data, pixel_size=0.5, title="HRTEM")
    """

    _esm = pathlib.Path(__file__).parent / "static" / "show3d.js"
    _css = pathlib.Path(__file__).parent / "static" / "show3d.css"

    # =========================================================================
    # Core State
    # =========================================================================
    slice_idx = traitlets.Int(0).tag(sync=True)
    n_slices = traitlets.Int(1).tag(sync=True)
    height = traitlets.Int(1).tag(sync=True)
    width = traitlets.Int(1).tag(sync=True)
    frame_bytes = traitlets.Bytes(b"").tag(sync=True)
    labels = traitlets.List(traitlets.Unicode()).tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    cmap = traitlets.Unicode("magma").tag(sync=True)
    dim_label = traitlets.Unicode("Frame").tag(sync=True)

    # =========================================================================
    # Playback Controls
    # =========================================================================
    playing = traitlets.Bool(False).tag(sync=True)
    reverse = traitlets.Bool(False).tag(sync=True)  # Play in reverse direction
    boomerang = traitlets.Bool(False).tag(sync=True)  # Ping-pong playback
    fps = traitlets.Float(5.0).tag(sync=True)  # Default 5 FPS for easier control
    loop = traitlets.Bool(True).tag(sync=True)
    loop_start = traitlets.Int(0).tag(sync=True)  # Start frame for loop range
    loop_end = traitlets.Int(-1).tag(sync=True)  # End frame for loop (-1 = last)
    bookmarked_frames = traitlets.List(traitlets.Int()).tag(sync=True)
    playback_path = traitlets.List(traitlets.Int()).tag(sync=True)

    # =========================================================================
    # Statistics Panel
    # =========================================================================
    show_controls = traitlets.Bool(True).tag(sync=True)
    show_stats = traitlets.Bool(True).tag(sync=True)
    stats_mean = traitlets.Float(0.0).tag(sync=True)
    stats_min = traitlets.Float(0.0).tag(sync=True)
    stats_max = traitlets.Float(0.0).tag(sync=True)
    stats_std = traitlets.Float(0.0).tag(sync=True)

    # =========================================================================
    # Display Options
    # =========================================================================
    log_scale = traitlets.Bool(False).tag(sync=True)
    auto_contrast = traitlets.Bool(False).tag(sync=True)
    percentile_low = traitlets.Float(1.0).tag(sync=True)
    percentile_high = traitlets.Float(99.0).tag(sync=True)
    data_min = traitlets.Float(0.0).tag(sync=True)
    data_max = traitlets.Float(0.0).tag(sync=True)

    # =========================================================================
    # Scale Bar
    # =========================================================================
    pixel_size = traitlets.Float(0.0).tag(sync=True)  # Å/pixel, 0 = no scale bar
    scale_bar_visible = traitlets.Bool(True).tag(sync=True)

    # =========================================================================
    # Timestamps / Dose
    # =========================================================================
    timestamps = traitlets.List(traitlets.Float()).tag(sync=True)
    timestamp_unit = traitlets.Unicode("s").tag(sync=True)
    current_timestamp = traitlets.Float(0.0).tag(sync=True)

    # =========================================================================
    # ROI Selection
    # =========================================================================
    roi_active = traitlets.Bool(False).tag(sync=True)
    roi_list = traitlets.List([]).tag(sync=True)
    roi_selected_idx = traitlets.Int(-1).tag(sync=True)
    roi_stats = traitlets.Dict({}).tag(sync=True)
    roi_plot_data = traitlets.Bytes(b"").tag(sync=True)
    # =========================================================================
    # Sizing
    # =========================================================================
    canvas_size = traitlets.Int(0).tag(sync=True)  # If 0, use frontend defaults

    # =========================================================================
    # Diff Mode
    # =========================================================================
    diff_mode = traitlets.Unicode("off").tag(sync=True)

    # =========================================================================
    # Analysis Panels (FFT + Histogram shown together)
    # =========================================================================
    show_fft = traitlets.Bool(False).tag(sync=True)
    fft_window = traitlets.Bool(True).tag(sync=True)
    show_playback = traitlets.Bool(False).tag(sync=True)
    disabled_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)
    hidden_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)
    # =========================================================================
    # Line Profile
    # =========================================================================
    profile_line = traitlets.List(traitlets.Dict()).tag(sync=True)
    profile_width = traitlets.Int(1).tag(sync=True)

    # =========================================================================
    # Export (GIF / ZIP of PNGs)
    # =========================================================================
    _gif_export_requested = traitlets.Bool(False).tag(sync=True)
    _gif_data = traitlets.Bytes(b"").tag(sync=True)
    _gif_metadata_json = traitlets.Unicode("").tag(sync=True)
    _zip_export_requested = traitlets.Bool(False).tag(sync=True)
    _zip_data = traitlets.Bytes(b"").tag(sync=True)
    _bundle_export_requested = traitlets.Bool(False).tag(sync=True)
    _bundle_data = traitlets.Bytes(b"").tag(sync=True)

    # =========================================================================
    # Playback Buffer (sliding prefetch)
    # =========================================================================
    _buffer_bytes = traitlets.Bytes(b"").tag(sync=True)
    _buffer_start = traitlets.Int(0).tag(sync=True)
    _buffer_count = traitlets.Int(0).tag(sync=True)
    _prefetch_request = traitlets.Int(-1).tag(sync=True)

    @classmethod
    def _normalize_tool_groups(cls, tool_groups):
        return normalize_tool_groups("Show3D", tool_groups)

    @classmethod
    def _build_disabled_tools(
        cls,
        disabled_tools=None,
        disable_display: bool = False,
        disable_histogram: bool = False,
        disable_stats: bool = False,
        disable_playback: bool = False,
        disable_navigation: bool = False,
        disable_view: bool = False,
        disable_export: bool = False,
        disable_roi: bool = False,
        disable_profile: bool = False,
        disable_all: bool = False,
    ):
        return build_tool_groups(
            "Show3D",
            tool_groups=disabled_tools,
            all_flag=disable_all,
            flag_map={
                "display": disable_display,
                "histogram": disable_histogram,
                "stats": disable_stats,
                "playback": disable_playback or disable_navigation,
                "view": disable_view,
                "export": disable_export,
                "roi": disable_roi,
                "profile": disable_profile,
            },
        )

    @classmethod
    def _build_hidden_tools(
        cls,
        hidden_tools=None,
        hide_display: bool = False,
        hide_histogram: bool = False,
        hide_stats: bool = False,
        hide_playback: bool = False,
        hide_navigation: bool = False,
        hide_view: bool = False,
        hide_export: bool = False,
        hide_roi: bool = False,
        hide_profile: bool = False,
        hide_all: bool = False,
    ):
        return build_tool_groups(
            "Show3D",
            tool_groups=hidden_tools,
            all_flag=hide_all,
            flag_map={
                "display": hide_display,
                "histogram": hide_histogram,
                "stats": hide_stats,
                "playback": hide_playback or hide_navigation,
                "view": hide_view,
                "export": hide_export,
                "roi": hide_roi,
                "profile": hide_profile,
            },
        )

    _VALID_DIFF_MODES = {"off", "previous", "first"}

    @traitlets.validate("diff_mode")
    def _validate_diff_mode(self, proposal):
        val = proposal["value"]
        if val not in self._VALID_DIFF_MODES:
            raise traitlets.TraitError(
                f"Invalid diff_mode '{val}'. Must be one of: {sorted(self._VALID_DIFF_MODES)}"
            )
        return val

    @traitlets.validate("disabled_tools")
    def _validate_disabled_tools(self, proposal):
        return self._normalize_tool_groups(proposal["value"])

    @traitlets.validate("hidden_tools")
    def _validate_hidden_tools(self, proposal):
        return self._normalize_tool_groups(proposal["value"])

    @classmethod
    def from_path(
        cls,
        source: str | pathlib.Path,
        *,
        file_type: str | None = None,
        dataset_path: str | None = None,
        **kwargs,
    ) -> Self:
        """Create Show3D from any supported file or image folder.

        Parameters
        ----------
        source : str or pathlib.Path
            File path (any format supported by ``IO.read``) or folder path.
        file_type : str, optional
            Required for folders. Explicitly selects which files to load.
        dataset_path : str, optional
            Explicit HDF dataset path for ``.emd`` sources.
        """
        path = pathlib.Path(source)
        if path.is_dir():
            if file_type is None:
                raise ValueError("file_type is required for folder loading.")
            result = IO.folder(path, file_type=file_type, dataset_path=dataset_path)
        else:
            result = IO.file(path, dataset_path=dataset_path)

        kwargs.setdefault("title", result.title)
        if result.labels:
            kwargs.setdefault("labels", result.labels)
        if result.pixel_size is not None:
            kwargs.setdefault("pixel_size", result.pixel_size)
        data = result.data
        if data.ndim == 2:
            data = data[None, ...]
        return cls(data, **kwargs)

    @classmethod
    def from_folder(
        cls,
        folder: str | pathlib.Path,
        *,
        file_type: str,
        dataset_path: str | None = None,
        **kwargs,
    ) -> Self:
        """Create Show3D from folder containing EMD, PNG, and/or TIFF files.

        Parameters
        ----------
        folder : str or pathlib.Path
            Folder containing supported files.
        file_type : {"emd", "png", "tiff"}
            Explicitly selects which files to load from the folder.
        dataset_path : str, optional
            Explicit HDF dataset path used when ``file_type='emd'``.
        """
        return cls.from_path(folder, file_type=file_type, dataset_path=dataset_path, **kwargs)

    @classmethod
    def from_emd(
        cls,
        source: str | pathlib.Path,
        *,
        dataset_path: str | None = None,
        **kwargs,
    ) -> Self:
        """Create Show3D from an EMD file."""
        return cls.from_path(source, dataset_path=dataset_path, **kwargs)

    @classmethod
    def from_tiff(cls, source: str | pathlib.Path, **kwargs) -> Self:
        """Create Show3D from a TIFF file."""
        return cls.from_path(source, **kwargs)

    @classmethod
    def from_png(cls, source: str | pathlib.Path, **kwargs) -> Self:
        """Create Show3D from a single PNG file."""
        return cls.from_path(source, **kwargs)

    @classmethod
    def from_dm4(cls, source: str | pathlib.Path, **kwargs) -> Self:
        """Create Show3D from a DM4 file (requires rsciio)."""
        return cls.from_path(source, **kwargs)

    @classmethod
    def from_dm3(cls, source: str | pathlib.Path, **kwargs) -> Self:
        """Create Show3D from a DM3 file (requires rsciio)."""
        return cls.from_path(source, **kwargs)

    @classmethod
    def from_mrc(cls, source: str | pathlib.Path, **kwargs) -> Self:
        """Create Show3D from an MRC file (requires rsciio)."""
        return cls.from_path(source, **kwargs)

    def __init__(
        self,
        data,
        labels: list[str] | None = None,
        title: str = "",
        cmap: str | Colormap = Colormap.MAGMA,
        vmin: float | None = None,
        vmax: float | None = None,
        pixel_size: float = 0.0,
        log_scale: bool = False,
        auto_contrast: bool = False,
        percentile_low: float = 1.0,
        percentile_high: float = 99.0,
        fps: float = 5.0,
        timestamps: list[float] | None = None,
        timestamp_unit: str = "s",
        show_fft: bool = False,
        fft_window: bool = True,
        show_playback: bool = False,
        show_stats: bool = True,
        show_controls: bool = True,
        canvas_size: int = 0,
        disabled_tools: list[str] | None = None,
        disable_display: bool = False,
        disable_histogram: bool = False,
        disable_stats: bool = False,
        disable_playback: bool = False,
        disable_navigation: bool = False,
        disable_view: bool = False,
        disable_export: bool = False,
        disable_roi: bool = False,
        disable_profile: bool = False,
        disable_all: bool = False,
        hidden_tools: list[str] | None = None,
        hide_display: bool = False,
        hide_histogram: bool = False,
        hide_stats: bool = False,
        hide_playback: bool = False,
        hide_navigation: bool = False,
        hide_view: bool = False,
        hide_export: bool = False,
        hide_roi: bool = False,
        hide_profile: bool = False,
        hide_all: bool = False,
        diff_mode: str = "off",
        buffer_size: int = 64,
        dim_label: str = "Frame",
        use_torch: bool = False,
        device: str | None = None,
        state=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.widget_version = resolve_widget_version()

        # Optional torch GPU acceleration
        self._use_torch = False
        self._device = None
        self._data_torch = None
        if use_torch:
            if not _HAS_TORCH:
                raise ImportError(
                    "use_torch=True requires PyTorch. Install it with: pip install torch"
                )
            self._use_torch = True
            self._device = torch.device(
                device or (
                    "mps" if torch.backends.mps.is_available()
                    else "cuda" if torch.cuda.is_available()
                    else "cpu"
                )
            )

        # Check if data is an IOResult and extract metadata
        if isinstance(data, IOResult):
            if not title and data.title:
                title = data.title
            if pixel_size == 0.0 and data.pixel_size is not None:
                pixel_size = data.pixel_size
            if labels is None and data.labels:
                labels = data.labels
            data = data.data
            # Wrap 2D to single-frame stack for Show3D
            if hasattr(data, "ndim") and data.ndim == 2:
                data = data[None, ...]

        # Check if data is a Dataset3d and extract metadata
        _extracted_title = None
        _extracted_pixel_size = None
        if hasattr(data, "array") and hasattr(data, "name") and hasattr(data, "sampling"):
            _extracted_title = data.name if data.name else None
            # sampling is (z_sampling, y_sampling, x_sampling) - use y/x for pixel size
            if hasattr(data, "sampling") and len(data.sampling) >= 3:
                sampling_val = float(data.sampling[1])
                # pixel_size is in Å — convert if units are nm
                if hasattr(data, "units"):
                    units = list(data.units)
                    if units[1] in ("nm", "nanometer"):
                        sampling_val = sampling_val * 10  # nm → Å
                _extracted_pixel_size = sampling_val
            data = data.array

        # Convert input to NumPy (handles NumPy, CuPy, PyTorch)
        data = to_numpy(data)

        # Ensure 3D
        if data.ndim != 3:
            raise ValueError(f"Expected 3D array, got {data.ndim}D")

        # Store data as float32 numpy array
        self._data = data.astype(np.float32)

        # Create GPU copy if torch acceleration enabled
        if self._use_torch:
            self._data_torch = torch.from_numpy(self._data).to(self._device)

        # Dimensions
        self.n_slices = int(self._data.shape[0])
        self.height = int(self._data.shape[1])
        self.width = int(self._data.shape[2])

        # Color range (global across all frames)
        self._vmin_user = vmin
        self._vmax_user = vmax
        if self._use_torch:
            self._vmin = vmin if vmin is not None else float(self._data_torch.min().item())
            self._vmax = vmax if vmax is not None else float(self._data_torch.max().item())
            self.data_min = float(self._data_torch.min().item())
            self.data_max = float(self._data_torch.max().item())
        else:
            self._vmin = vmin if vmin is not None else float(self._data.min())
            self._vmax = vmax if vmax is not None else float(self._data.max())
            self.data_min = float(self._data.min())
            self.data_max = float(self._data.max())

        # Labels
        if labels is not None:
            self.labels = list(labels)
        else:
            self.labels = [str(i) for i in range(self.n_slices)]

        # Title and colormap - use extracted title if not explicitly provided
        self.title = title if title else (_extracted_title or "")
        self.cmap = str(cmap)  # Convert Colormap enum to string

        # Use extracted pixel_size if not explicitly provided
        if pixel_size == 0.0 and _extracted_pixel_size is not None:
            pixel_size = _extracted_pixel_size

        # Display options
        self.pixel_size = pixel_size
        self.log_scale = log_scale
        self.auto_contrast = auto_contrast
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.fps = fps

        # Timestamps
        if timestamps is not None:
            self.timestamps = [float(t) for t in timestamps]
        else:
            self.timestamps = []
        self.timestamp_unit = timestamp_unit
        self.dim_label = dim_label
        self.diff_mode = diff_mode
        self.show_fft = show_fft
        self.fft_window = fft_window
        self.show_playback = show_playback
        self.show_stats = show_stats
        self.show_controls = show_controls
        self.canvas_size = canvas_size
        self.disabled_tools = self._build_disabled_tools(
            disabled_tools=disabled_tools,
            disable_display=disable_display,
            disable_histogram=disable_histogram,
            disable_stats=disable_stats,
            disable_playback=disable_playback,
            disable_navigation=disable_navigation,
            disable_view=disable_view,
            disable_export=disable_export,
            disable_roi=disable_roi,
            disable_profile=disable_profile,
            disable_all=disable_all,
        )
        self.hidden_tools = self._build_hidden_tools(
            hidden_tools=hidden_tools,
            hide_display=hide_display,
            hide_histogram=hide_histogram,
            hide_stats=hide_stats,
            hide_playback=hide_playback,
            hide_navigation=hide_navigation,
            hide_view=hide_view,
            hide_export=hide_export,
            hide_roi=hide_roi,
            hide_profile=hide_profile,
            hide_all=hide_all,
        )
        frame_bytes = self.height * self.width * 4  # float32
        max_buffer_bytes = 64 * 1024 * 1024  # 64 MB cap per transfer
        min_buffer_frames = 8  # guarantee at least 8 frames for large images
        max_frames = max(min_buffer_frames, max_buffer_bytes // frame_bytes)
        self._buffer_size = min(buffer_size, self.n_slices, max_frames)

        # Initial position at middle
        self.slice_idx = int(self.n_slices // 2)

        # Observers
        self.observe(self._on_slice_change, names=["slice_idx"])
        self.observe(
            self._on_roi_change,
            names=["roi_active", "roi_list", "roi_selected_idx"],
        )
        self.observe(self._on_gif_export, names=["_gif_export_requested"])
        self.observe(self._on_zip_export, names=["_zip_export_requested"])
        self.observe(self._on_bundle_export, names=["_bundle_export_requested"])
        self.observe(self._on_playing_change, names=["playing"])
        self.observe(self._on_prefetch, names=["_prefetch_request"])
        self.observe(self._on_diff_mode_change, names=["diff_mode"])

        # Initial update
        self._update_all()

        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(
                    json.loads(pathlib.Path(state).read_text()),
                    require_envelope=True,
                )
            else:
                state = unwrap_state_payload(state)
            self.load_state_dict(state)

    def set_image(self, data, labels=None):
        """Replace the stack data. Preserves all display settings."""
        if hasattr(data, "array") and hasattr(data, "name") and hasattr(data, "sampling"):
            data = data.array
        data = to_numpy(data)
        if data.ndim != 3:
            raise ValueError(f"Expected 3D array, got {data.ndim}D")
        self._data = data.astype(np.float32)
        if self._use_torch:
            self._data_torch = torch.from_numpy(self._data).to(self._device)
        self.n_slices = int(data.shape[0])
        self.height = int(data.shape[1])
        self.width = int(data.shape[2])
        if self._use_torch:
            self.data_min = float(self._data_torch.min().item())
            self.data_max = float(self._data_torch.max().item())
        else:
            self.data_min = float(self._data.min())
            self.data_max = float(self._data.max())
        self._vmin = self._vmin_user if self._vmin_user is not None else self.data_min
        self._vmax = self._vmax_user if self._vmax_user is not None else self.data_max
        if labels is not None:
            self.labels = list(labels)
        else:
            self.labels = [str(i) for i in range(self.n_slices)]
        self.slice_idx = min(self.slice_idx, self.n_slices - 1)
        self._buffer_size = min(self._buffer_size, self.n_slices)
        self._update_all()

    def __repr__(self) -> str:
        parts = f"Show3D({self.n_slices}×{self.height}×{self.width}, frame={self.slice_idx}, cmap={self.cmap}"
        if self.diff_mode != "off":
            parts += f", diff={self.diff_mode}"
        parts += ")"
        return parts

    def state_dict(self):
        return {
            "title": self.title,
            "cmap": self.cmap,
            "log_scale": self.log_scale,
            "auto_contrast": self.auto_contrast,
            "percentile_low": self.percentile_low,
            "percentile_high": self.percentile_high,
            "show_stats": self.show_stats,
            "show_controls": self.show_controls,
            "show_fft": self.show_fft,
            "fft_window": self.fft_window,
            "show_playback": self.show_playback,
            "disabled_tools": self.disabled_tools,
            "hidden_tools": self.hidden_tools,
            "pixel_size": self.pixel_size,
            "scale_bar_visible": self.scale_bar_visible,
            "canvas_size": self.canvas_size,
            "fps": self.fps,
            "loop": self.loop,
            "reverse": self.reverse,
            "boomerang": self.boomerang,
            "loop_start": self.loop_start,
            "loop_end": self.loop_end,
            "bookmarked_frames": self.bookmarked_frames,
            "playback_path": self.playback_path,
            "roi_active": self.roi_active,
            "roi_list": self.roi_list,
            "roi_selected_idx": self.roi_selected_idx,
            "profile_line": self.profile_line,
            "profile_width": self.profile_width,
            "diff_mode": self.diff_mode,
            "dim_label": self.dim_label,
            "timestamp_unit": self.timestamp_unit,
        }

    def save(self, path: str):
        save_state_file(path, "Show3D", self.state_dict())

    def load_state_dict(self, state):
        for key, val in state.items():
            if hasattr(self, key):
                setattr(self, key, val)

    def summary(self):
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
        contrast = "auto contrast" if self.auto_contrast else "manual contrast"
        display = f"{cmap} | {contrast} | {scale}"
        if self.show_fft:
            display += " | FFT"
            if not self.fft_window:
                display += " (no window)"
        if self.diff_mode != "off":
            display += f" | diff={self.diff_mode}"
        lines.append(f"Display:  {display}")
        if self.disabled_tools:
            lines.append(f"Locked:   {', '.join(self.disabled_tools)}")
        if self.hidden_tools:
            lines.append(f"Hidden:   {', '.join(self.hidden_tools)}")
        lines.append(f"Playback: {self.fps} fps | loop={'on' if self.loop else 'off'} | reverse={'on' if self.reverse else 'off'} | boomerang={'on' if self.boomerang else 'off'}")
        if self.loop_start > 0 or self.loop_end >= 0:
            end = self.loop_end if self.loop_end >= 0 else self.n_slices - 1
            lines.append(f"Range:    {self.loop_start}–{end}")
        if self.roi_active and self.roi_list:
            lines.append(f"ROI:      {len(self.roi_list)} region(s)")
        if len(self.profile_line) >= 2:
            p0, p1 = self.profile_line[0], self.profile_line[1]
            lines.append(f"Profile:  ({p0['row']:.0f}, {p0['col']:.0f}) → ({p1['row']:.0f}, {p1['col']:.0f}) width={self.profile_width}")
        print("\n".join(lines))

    def _get_color_range(self, frame: np.ndarray) -> tuple[float, float]:
        """Get vmin/vmax based on current settings."""
        if self.auto_contrast:
            vmin = float(np.percentile(frame, self.percentile_low))
            vmax = float(np.percentile(frame, self.percentile_high))
        else:
            vmin = self._vmin
            vmax = self._vmax
        return vmin, vmax

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Normalize frame to uint8 with current display settings."""
        # Apply log scale if enabled
        if self.log_scale:
            frame = np.log1p(np.maximum(frame, 0))

        vmin, vmax = self._get_color_range(frame)

        if vmax > vmin:
            normalized = np.clip((frame - vmin) / (vmax - vmin) * 255, 0, 255)
            return normalized.astype(np.uint8)
        return np.zeros(frame.shape, dtype=np.uint8)

    def _get_display_frame(self, idx=None):
        if idx is None:
            idx = self.slice_idx
        frame = self._data[idx]
        if self.diff_mode == "previous":
            if idx == 0:
                return np.zeros_like(frame)
            return frame - self._data[idx - 1]
        if self.diff_mode == "first":
            return frame - self._data[0]
        return frame

    def _on_diff_mode_change(self, change=None):
        if self.diff_mode == "off":
            self.data_min = float(self._data.min())
            self.data_max = float(self._data.max())
        else:
            # Recompute global range for diff frames
            mins, maxs = [], []
            for i in range(self.n_slices):
                f = self._get_display_frame(i)
                mins.append(float(f.min()))
                maxs.append(float(f.max()))
            self.data_min = min(mins)
            self.data_max = max(maxs)
        self._update_all()

    def _update_all(self):
        """Update frame, stats, and all derived data. Uses hold_sync for batched transfer."""
        frame = self._get_display_frame()
        with self.hold_sync():
            if self._use_torch:
                t = self._data_torch[self.slice_idx]
                self.stats_mean = float(t.mean().item())
                self.stats_min = float(t.min().item())
                self.stats_max = float(t.max().item())
                self.stats_std = float(t.std().item())
            else:
                self.stats_mean = float(frame.mean())
                self.stats_min = float(frame.min())
                self.stats_max = float(frame.max())
                self.stats_std = float(frame.std())
            if self.timestamps and self.slice_idx < len(self.timestamps):
                self.current_timestamp = self.timestamps[self.slice_idx]
            if self.roi_active:
                self._update_roi_stats(frame)
            else:
                self.roi_stats = {}
            self.frame_bytes = frame.tobytes()

    def _roi_mask(self, roi: dict):
        r, c = np.ogrid[0 : self.height, 0 : self.width]
        shape = roi.get("shape", "circle")
        row = float(roi.get("row", 0))
        col = float(roi.get("col", 0))
        radius = max(1.0, float(roi.get("radius", 10)))
        if shape == "circle":
            return (c - col) ** 2 + (r - row) ** 2 <= radius**2
        if shape == "square":
            return (np.abs(c - col) <= radius) & (np.abs(r - row) <= radius)
        if shape == "rectangle":
            half_w = max(1.0, float(roi.get("width", 20)) / 2.0)
            half_h = max(1.0, float(roi.get("height", 20)) / 2.0)
            return (np.abs(c - col) <= half_w) & (np.abs(r - row) <= half_h)
        if shape == "annular":
            inner = max(0.0, float(roi.get("radius_inner", 5)))
            dist2 = (c - col) ** 2 + (r - row) ** 2
            return (dist2 >= inner**2) & (dist2 <= radius**2)
        return (c - col) ** 2 + (r - row) ** 2 <= radius**2

    def _update_roi_stats(self, frame: np.ndarray):
        idx = self.roi_selected_idx
        if idx < 0 or idx >= len(self.roi_list):
            self.roi_stats = {}
            return
        roi = self.roi_list[idx]
        mask = self._roi_mask(roi)
        if self._use_torch:
            mask_t = torch.from_numpy(mask).to(self._device)
            t = self._data_torch[self.slice_idx]
            region = t[mask_t]
            if region.numel() > 0:
                self.roi_stats = {
                    "mean": float(region.mean().item()),
                    "min": float(region.min().item()),
                    "max": float(region.max().item()),
                    "std": float(region.std().item()),
                }
            else:
                self.roi_stats = {}
        else:
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

    def _send_buffer(self, start_idx: int):
        end_idx = start_idx + self._buffer_size
        if self.diff_mode == "off":
            if end_idx <= self.n_slices:
                chunk = self._data[start_idx:end_idx]
            else:
                chunk = np.concatenate(
                    [self._data[start_idx:], self._data[: end_idx - self.n_slices]]
                )
        else:
            frames = []
            for j in range(self._buffer_size):
                idx = (start_idx + j) % self.n_slices
                frames.append(self._get_display_frame(idx))
            chunk = np.stack(frames)
        with self.hold_sync():
            self._buffer_start = int(start_idx)
            self._buffer_count = int(chunk.shape[0])
            self._buffer_bytes = chunk.tobytes()

    def _on_playing_change(self, change=None):
        if self.playing:
            self._send_buffer(self.slice_idx)
        else:
            # Playback stopped — refresh stats for the current frame
            self._update_all()

    def _on_prefetch(self, change=None):
        if self._prefetch_request >= 0 and self.playing:
            self._send_buffer(self._prefetch_request % self.n_slices)

    def _on_slice_change(self, change=None):
        if self.playing:
            return
        self._update_all()

    def _on_roi_change(self, change=None):
        """Handle ROI change."""
        if self.roi_active:
            self._update_roi_stats(self._get_display_frame())
            self._compute_roi_plot()
        else:
            self.roi_stats = {}
            self.roi_plot_data = b""

    def _compute_roi_plot(self):
        """Compute selected ROI mean for all frames."""
        idx = self.roi_selected_idx
        if idx < 0 or idx >= len(self.roi_list):
            self.roi_plot_data = b""
            return
        mask = self._roi_mask(self.roi_list[idx])
        if mask.sum() == 0:
            self.roi_plot_data = b""
            return
        if self._use_torch:
            mask_t = torch.from_numpy(mask).to(self._device)
            # Vectorized: (n_slices, n_masked_pixels) -> mean per frame
            masked = self._data_torch[:, mask_t]
            means = masked.mean(dim=1).cpu().numpy().astype(np.float32)
        else:
            means = np.array([float(self._data[i][mask].mean()) for i in range(self.n_slices)], dtype=np.float32)
        self.roi_plot_data = means.tobytes()

    # =========================================================================
    # Public Methods
    # =========================================================================

    def play(self) -> Self:
        """Start playback."""
        self.playing = True
        return self

    def pause(self) -> Self:
        """Pause playback."""
        self.playing = False
        return self

    def stop(self) -> Self:
        """Stop playback and reset to beginning."""
        self.playing = False
        self.slice_idx = 0
        return self

    def goto(self, index: int) -> Self:
        """Jump to a specific frame index."""
        self.slice_idx = int(index) % self.n_slices
        return self

    def set_playback_path(self, path) -> Self:
        """Set custom playback order (list of frame indices)."""
        self.playback_path = [int(i) % self.n_slices for i in path]
        return self

    def clear_playback_path(self) -> Self:
        """Clear custom playback path (revert to sequential)."""
        self.playback_path = []
        return self

    def profile_all_frames(self, start: tuple | None = None, end: tuple | None = None) -> np.ndarray:
        """Extract the line profile from every frame, returning (n_slices, n_points).

        Uses the current profile_line unless start/end are provided.
        Always samples raw data (ignores diff_mode).

        Parameters
        ----------
        start : tuple of (row, col), optional
            Start point. Overrides current profile_line.
        end : tuple of (row, col), optional
            End point. Overrides current profile_line.

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
        rows = []
        for i in range(self.n_slices):
            rows.append(self._sample_profile_on(self._data[i], row0, col0, row1, col1))
        return np.stack(rows)

    def _upsert_selected_roi(self, updates: dict):
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

    def duplicate_selected_roi(self, row_offset: int = 3, col_offset: int = 3) -> Self:
        """Duplicate selected ROI with a small offset and auto-assigned color."""
        idx = int(self.roi_selected_idx)
        if idx < 0 or idx >= len(self.roi_list):
            return self
        color_cycle = ["#4fc3f7", "#81c784", "#ffb74d", "#ce93d8", "#ef5350", "#ffd54f", "#90a4ae", "#a1887f"]
        src = dict(self.roi_list[idx])
        with self.hold_sync():
            rois = list(self.roi_list)
            src["row"] = int(np.clip(float(src.get("row", self.height // 2)) + row_offset, 0, self.height - 1))
            src["col"] = int(np.clip(float(src.get("col", self.width // 2)) + col_offset, 0, self.width - 1))
            src["color"] = color_cycle[len(rois) % len(color_cycle)]
            src["highlight"] = False
            src["visible"] = True
            src["locked"] = False
            rois.append(src)
            self.roi_list = rois
            self.roi_selected_idx = len(rois) - 1
            self.roi_active = True
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

    def _sample_line(self, img, row0, col0, row1, col1):
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

    def _sample_profile_on(self, img, row0, col0, row1, col1):
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

    def _sample_profile(self, row0, col0, row1, col1):
        return self._sample_profile_on(self._get_display_frame(), row0, col0, row1, col1)

    def set_profile(self, start: tuple, end: tuple) -> Self:
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

    @property
    def profile(self):
        """Get profile line endpoints as [(row0, col0), (row1, col1)] or []."""
        return [(p["row"], p["col"]) for p in self.profile_line]

    @property
    def profile_values(self):
        """Get intensity values along the profile line for the current frame."""
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

    def _on_gif_export(self, change=None):
        if not self._gif_export_requested:
            return
        self._gif_export_requested = False
        self._generate_gif()

    def _normalize_frames_torch(self, start: int, end: int) -> np.ndarray:
        """Batch-normalize frames [start, end] on GPU. Returns (N, H, W) uint8 numpy."""
        frames = self._data_torch[start : end + 1].clone()
        if self.log_scale:
            frames = torch.log1p(torch.clamp(frames, min=0))
        if self.auto_contrast:
            flat = frames.reshape(-1).float()
            vmin = float(torch.quantile(flat, self.percentile_low / 100.0).item())
            vmax = float(torch.quantile(flat, self.percentile_high / 100.0).item())
        else:
            vmin = self._vmin
            vmax = self._vmax
            if self.log_scale:
                vmin = float(np.log1p(max(vmin, 0)))
                vmax = float(np.log1p(max(vmax, 0)))
        if vmax > vmin:
            normalized = torch.clamp((frames - vmin) / (vmax - vmin) * 255.0, 0, 255).to(torch.uint8)
        else:
            normalized = torch.zeros_like(frames, dtype=torch.uint8)
        return normalized.cpu().numpy()

    def _generate_gif(self):
        import io

        from matplotlib import colormaps
        from PIL import Image

        start = max(0, self.loop_start)
        end = self.loop_end if self.loop_end >= 0 else self.n_slices - 1
        end = min(end, self.n_slices - 1)

        cmap_fn = colormaps.get_cmap(self.cmap)
        duration_ms = int(1000 / max(0.1, self.fps))

        pil_frames = []
        if self._use_torch:
            normalized_all = self._normalize_frames_torch(start, end)
            for i in range(normalized_all.shape[0]):
                rgba = cmap_fn(normalized_all[i] / 255.0)
                rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
                pil_frames.append(Image.fromarray(rgb))
        else:
            for i in range(start, end + 1):
                frame = self._data[i]
                normalized = self._normalize_frame(frame)
                rgba = cmap_fn(normalized / 255.0)
                rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
                pil_frames.append(Image.fromarray(rgb))

        if not pil_frames:
            with self.hold_sync():
                self._gif_data = b""
                self._gif_metadata_json = ""
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
            **build_json_header("Show3D"),
            "format": "gif",
            "export_kind": "animated_frames",
            "frame_range": {"start": int(start), "end": int(end)},
            "n_frames": int(len(pil_frames)),
            "duration_ms": int(duration_ms),
            "display": {
                "cmap": self.cmap,
                "log_scale": bool(self.log_scale),
                "auto_contrast": bool(self.auto_contrast),
                "percentile_low": float(self.percentile_low),
                "percentile_high": float(self.percentile_high),
            },
        }
        with self.hold_sync():
            self._gif_metadata_json = json.dumps(metadata, indent=2)
            self._gif_data = buf.getvalue()

    def _on_zip_export(self, change=None):
        if not self._zip_export_requested:
            return
        self._zip_export_requested = False
        self._generate_zip()

    def _generate_zip(self):
        import io
        import zipfile

        from matplotlib import colormaps
        from PIL import Image

        start = max(0, self.loop_start)
        end = self.loop_end if self.loop_end >= 0 else self.n_slices - 1
        end = min(end, self.n_slices - 1)

        cmap_fn = colormaps.get_cmap(self.cmap)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            metadata = {
                **build_json_header("Show3D"),
                "format": "zip",
                "export_kind": "png_frames",
                "frame_range": {"start": int(start), "end": int(end)},
                "n_frames": int(end - start + 1),
                "display": {"cmap": self.cmap, "log_scale": bool(self.log_scale)},
            }
            zf.writestr("metadata.json", json.dumps(metadata, indent=2))
            if self._use_torch:
                normalized_all = self._normalize_frames_torch(start, end)
                for j in range(normalized_all.shape[0]):
                    i = start + j
                    rgba = cmap_fn(normalized_all[j] / 255.0)
                    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
                    img = Image.fromarray(rgb)
                    img_buf = io.BytesIO()
                    img.save(img_buf, format="PNG")
                    label = self.labels[i] if self.labels else str(i).zfill(4)
                    zf.writestr(f"frame_{label}.png", img_buf.getvalue())
            else:
                for i in range(start, end + 1):
                    frame = self._data[i]
                    normalized = self._normalize_frame(frame)
                    rgba = cmap_fn(normalized / 255.0)
                    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
                    img = Image.fromarray(rgb)
                    img_buf = io.BytesIO()
                    img.save(img_buf, format="PNG")
                    label = self.labels[i] if self.labels else str(i).zfill(4)
                    zf.writestr(f"frame_{label}.png", img_buf.getvalue())
        self._zip_data = buf.getvalue()

    def _on_bundle_export(self, change=None):
        if not self._bundle_export_requested:
            return
        self._bundle_export_requested = False
        self._generate_bundle()

    def _roi_timeseries_csv(self) -> str:
        import csv
        import io

        rois = list(self.roi_list)
        masks = [self._roi_mask(roi) for roi in rois]
        out = io.StringIO()
        writer = csv.writer(out)
        header = ["frame_index", "label"]
        if self.timestamps and len(self.timestamps) >= self.n_slices:
            header.append(f"timestamp_{self.timestamp_unit or 'value'}")
        header.extend([f"roi_{i + 1}_mean" for i in range(len(rois))])
        writer.writerow(header)

        if self._use_torch:
            # Vectorized per-ROI means across all frames
            masks_t = [torch.from_numpy(m).to(self._device) for m in masks]
            roi_means = []
            for mask_t in masks_t:
                masked = self._data_torch[:, mask_t]  # (n_slices, n_pixels)
                if masked.shape[1] > 0:
                    roi_means.append(masked.mean(dim=1).cpu().numpy())
                else:
                    roi_means.append(np.full(self.n_slices, np.nan))
            for i in range(self.n_slices):
                row = [i, self.labels[i] if i < len(self.labels) else str(i)]
                if self.timestamps and len(self.timestamps) >= self.n_slices:
                    row.append(float(self.timestamps[i]))
                for rm in roi_means:
                    val = rm[i]
                    row.append(float(val) if not np.isnan(val) else "")
                writer.writerow(row)
        else:
            for i in range(self.n_slices):
                row = [i, self.labels[i] if i < len(self.labels) else str(i)]
                if self.timestamps and len(self.timestamps) >= self.n_slices:
                    row.append(float(self.timestamps[i]))
                frame = self._data[i]
                for mask in masks:
                    region = frame[mask]
                    row.append(float(region.mean()) if region.size > 0 else "")
                writer.writerow(row)
        return out.getvalue()

    def _generate_bundle(self):
        import io
        import zipfile

        from matplotlib import colormaps
        from PIL import Image

        idx = int(np.clip(self.slice_idx, 0, self.n_slices - 1))
        cmap_fn = colormaps.get_cmap(self.cmap)
        frame = self._data[idx]
        normalized = self._normalize_frame(frame)
        rgba = cmap_fn(normalized / 255.0)
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
        img = Image.fromarray(rgb)
        img_buf = io.BytesIO()
        img.save(img_buf, format="PNG")

        state_payload = {**build_json_header("Show3D"), "state": self.state_dict()}
        csv_text = self._roi_timeseries_csv()
        label = self.labels[idx] if idx < len(self.labels) else str(idx)
        safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(label)).strip("_") or str(idx)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"frame_{safe_label}.png", img_buf.getvalue())
            zf.writestr("roi_timeseries.csv", csv_text)
            zf.writestr("state.json", json.dumps(state_payload, indent=2))
        self._bundle_data = buf.getvalue()


    def save_image(self, path: str | pathlib.Path, *, frame_idx: int | None = None,
                   format: str | None = None, dpi: int = 150) -> pathlib.Path:
        """Save a single frame as PNG, PDF, or TIFF.

        Parameters
        ----------
        path : str or pathlib.Path
            Output file path.
        frame_idx : int, optional
            Frame index to export. Defaults to current slice_idx.
        format : str, optional
            'png', 'pdf', or 'tiff'. If omitted, inferred from file extension.
        dpi : int, default 150
            Output DPI metadata.

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

        idx = frame_idx if frame_idx is not None else self.slice_idx
        if idx < 0 or idx >= self.n_slices:
            raise IndexError(f"Frame index {idx} out of range [0, {self.n_slices})")

        frame = self._data[idx]
        normalized = self._normalize_frame(frame)
        cmap_fn = colormaps.get_cmap(self.cmap)
        rgba = (cmap_fn(normalized / 255.0) * 255).astype(np.uint8)

        img = Image.fromarray(rgba)
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path), dpi=(dpi, dpi))
        return path


bind_tool_runtime_api(Show3D, "Show3D")
