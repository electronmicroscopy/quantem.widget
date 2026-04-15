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
    # GPU memory budget for display buffers (same as Show2D)
    _GPU_DISPLAY_BUDGET_MB = 2500

    slice_idx = traitlets.Int(0).tag(sync=True)
    n_slices = traitlets.Int(1).tag(sync=True)
    height = traitlets.Int(1).tag(sync=True)
    width = traitlets.Int(1).tag(sync=True)
    frame_bytes = traitlets.Bytes(b"").tag(sync=True)
    _display_bin_factor = traitlets.Int(1).tag(sync=True)
    labels = traitlets.List(traitlets.Unicode()).tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    cmap = traitlets.Unicode("magma").tag(sync=True)
    dim_label = traitlets.Unicode("Frame").tag(sync=True)

    # Multi-Panel (synchronized side-by-side stacks)
    n_panels = traitlets.Int(1).tag(sync=True)
    panel_titles = traitlets.List(traitlets.Unicode()).tag(sync=True)

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
    # Frames marked bad but kept in the stack (downstream skips them)
    excluded_frames = traitlets.List(traitlets.Int()).tag(sync=True)

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
    vmin = traitlets.Float(None, allow_none=True).tag(sync=True)
    vmax = traitlets.Float(None, allow_none=True).tag(sync=True)
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

    def __init__(
        self,
        *data_args,
        labels: list[str] | None = None,
        panel_titles: list[str] | None = None,
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
        display_bin: int | str = "auto",
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
            if hasattr(data, "sampling") and len(data.sampling) >= 3:
                sampling_val = float(data.sampling[1])
                if hasattr(data, "units"):
                    units = list(data.units)
                    if units[1] in ("nm", "nanometer"):
                        sampling_val = sampling_val * 10  # nm → Å
                _extracted_pixel_size = sampling_val
            data = data.array

        # Convert first panel to NumPy
        data = to_numpy(data)
        if data.ndim == 2:
            data = data[None, ...]
        if data.ndim != 3:
            raise ValueError(f"Expected 3D array, got {data.ndim}D")

        # Multi-panel: convert remaining args, validate shapes, concatenate
        if len(data_args) > 1:
            panels = [data.astype(np.float32)]
            for i, extra in enumerate(data_args[1:], 1):
                if isinstance(extra, IOResult):
                    extra = extra.data
                if hasattr(extra, "array"):
                    extra = extra.array
                arr = to_numpy(extra)
                if arr.ndim == 2:
                    arr = arr[None, ...]
                if arr.ndim != 3:
                    raise ValueError(f"Panel {i}: expected 3D array, got {arr.ndim}D")
                if arr.shape[0] != panels[0].shape[0]:
                    raise ValueError(
                        f"Panel {i} has {arr.shape[0]} frames, but panel 0 has "
                        f"{panels[0].shape[0]} frames. All panels must match."
                    )
                panels.append(arr.astype(np.float32))
            self.n_panels = len(panels)
            if panel_titles is not None:
                self.panel_titles = list(panel_titles)
            else:
                self.panel_titles = [f"Panel {i+1}" for i in range(len(panels))]
            # Bin each panel FIRST (before normalize), then normalize the
            # small binned data, then concatenate. Order matters for 4K:
            #   Old: normalize 768MB×3 (4.8s) → concat 2.3GB → bin (2.3s) = 7.1s
            #   New: bin 768MB×3 (1.8s) → normalize 48MB×3 (0.1s) → concat = 1.9s
            from quantem.widget.array_utils import bin2d as _bin2d
            orig_h = panels[0].shape[1]
            frame_mb = orig_h * panels[0].shape[2] * 4 / (1024 * 1024)
            # Compute per-panel bin factor (same auto-bin logic as single panel)
            panel_bin = 1
            total_w = sum(p.shape[2] for p in panels) + 2 * (len(panels) - 1)
            combined_frame_mb = orig_h * total_w * 4 / (1024 * 1024)
            if combined_frame_mb > 32:
                for bf in [2, 4, 8]:
                    if combined_frame_mb / (bf * bf) <= 32:
                        panel_bin = bf
                        break
                else:
                    panel_bin = 8

            if panel_bin > 1:
                # Bin each panel with direct reshape+mean (no copy overhead)
                binned_panels = []
                for p in panels:
                    n_f, h_f, w_f = p.shape
                    oh = h_f // panel_bin * panel_bin
                    ow = w_f // panel_bin * panel_bin
                    binned_panels.append(
                        p[:, :oh, :ow]
                        .reshape(n_f, oh // panel_bin, panel_bin, ow // panel_bin, panel_bin)
                        .mean(axis=(2, 4)).astype(np.float32)
                    )
                # Normalize the SMALL binned data
                normalized = []
                for bp in binned_panels:
                    if bp.size > 10_000_000:
                        sample = bp.flat[::max(1, bp.size // 1_000_000)]
                        p2, p98 = np.percentile(sample, [2, 98])
                    else:
                        p2, p98 = np.percentile(bp, [2, 98])
                    rng = max(p98 - p2, 1e-10)
                    normalized.append(np.clip((bp - p2) / rng, 0, 1).astype(np.float32))
            else:
                # No binning needed — normalize at full res
                normalized = []
                for p in panels:
                    if p.size > 10_000_000:
                        sample = p.flat[::max(1, p.size // 1_000_000)]
                        p2, p98 = np.percentile(sample, [2, 98])
                    else:
                        p2, p98 = np.percentile(p, [2, 98])
                    rng = max(p98 - p2, 1e-10)
                    normalized.append(np.clip((p - p2) / rng, 0, 1).astype(np.float32))

            # Concatenate horizontally with 2px black separator
            sep_w = 2
            sep = np.zeros(
                (normalized[0].shape[0], normalized[0].shape[1], sep_w),
                dtype=np.float32,
            )
            parts = []
            for j, p in enumerate(normalized):
                if j > 0:
                    parts.append(sep)
                parts.append(p)
            data = np.concatenate(parts, axis=2)
            self._panel_width = panels[0].shape[2] // max(panel_bin, 1)
            # Multi-panel already binned — skip auto-bin below
            self._multi_panel_bin = panel_bin
        else:
            self.n_panels = 1
            self._multi_panel_bin = 0
            if panel_titles is not None:
                self.panel_titles = list(panel_titles)

        # Store data as float32 numpy array
        self._data = data.astype(np.float32)

        # Create GPU copy if torch acceleration enabled
        if self._use_torch:
            self._data_torch = torch.from_numpy(self._data).to(self._device)

        # Dimensions
        self.n_slices = int(self._data.shape[0])
        orig_h = int(self._data.shape[1])
        orig_w = int(self._data.shape[2])

        # Auto-bin for display: reduce per-frame size to speed up trait sync.
        # Multi-panel data is already binned in the multi-panel path above.
        self._display_bin = 1
        if self._multi_panel_bin > 0:
            # Multi-panel already binned — _data IS the display data
            self._display_bin = self._multi_panel_bin
        elif display_bin == "auto":
            frame_mb = orig_h * orig_w * 4 / (1024 * 1024)
            if frame_mb > 32:
                for bf in [2, 4, 8]:
                    if frame_mb / (bf * bf) <= 32:
                        self._display_bin = bf
                        break
                else:
                    self._display_bin = 8
        elif isinstance(display_bin, int) and display_bin > 1:
            self._display_bin = display_bin

        if self._multi_panel_bin > 0:
            # Multi-panel already binned — _data IS the display data
            self._display_data = self._data
            self.height = orig_h
            self.width = orig_w
            self._display_bin_factor = self._multi_panel_bin
            if pixel_size > 0:
                pixel_size = pixel_size * self._multi_panel_bin
            print(f"  Display bin {self._multi_panel_bin}× (multi-panel): {self.height}×{self.width} ({self._display_data[0].nbytes // 1024 // 1024} MB/frame)")
        elif self._display_bin > 1:
            from quantem.widget.array_utils import bin2d
            self._display_data = bin2d(self._data, factor=self._display_bin, mode="mean")
            self.height = int(self._display_data.shape[1])
            self.width = int(self._display_data.shape[2])
            self._display_bin_factor = self._display_bin
            if pixel_size > 0:
                pixel_size = pixel_size * self._display_bin
            print(f"  Display bin {self._display_bin}×: {orig_h}×{orig_w} → {self.height}×{self.width} ({self._display_data[0].nbytes // 1024 // 1024} MB/frame)")
        else:
            self._display_data = self._data
            self.height = orig_h
            self.width = orig_w
            self._display_bin_factor = 1

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
        self.vmin = vmin
        self.vmax = vmax
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

        # Auto-bin display data
        orig_h, orig_w = data.shape[1], data.shape[2]
        frame_mb = orig_h * orig_w * 4 / (1024 * 1024)
        self._display_bin = 1
        if frame_mb > 32:
            for bf in [2, 4, 8]:
                if frame_mb / (bf * bf) <= 32:
                    self._display_bin = bf
                    break
            else:
                self._display_bin = 8

        if self._display_bin > 1:
            from quantem.widget.array_utils import bin2d
            self._display_data = bin2d(self._data, factor=self._display_bin, mode="mean")
            self.height = int(self._display_data.shape[1])
            self.width = int(self._display_data.shape[2])
            self._display_bin_factor = self._display_bin
        else:
            self._display_data = self._data
            self.height = orig_h
            self.width = orig_w
            self._display_bin_factor = 1

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
        if self.excluded_frames:
            parts += f", excluded={len(self.excluded_frames)}"
        parts += ")"
        return parts

    # =========================================================================
    # Time-series stack editing
    # =========================================================================

    def _valid_frame_idx(self, idx: int) -> int:
        idx = int(idx)
        if idx < 0 or idx >= self.n_slices:
            raise IndexError(
                f"Frame index {idx} out of range [0, {self.n_slices})"
            )
        return idx

    def exclude_frame(self, idx: int) -> Self:
        """Mark a frame as bad. Kept in the stack, flagged for downstream skip."""
        idx = self._valid_frame_idx(idx)
        if idx not in self.excluded_frames:
            self.excluded_frames = sorted([*self.excluded_frames, idx])
        return self

    def include_frame(self, idx: int) -> Self:
        idx = self._valid_frame_idx(idx)
        if idx in self.excluded_frames:
            self.excluded_frames = [i for i in self.excluded_frames if i != idx]
        return self

    def toggle_excluded(self, idx: int) -> Self:
        idx = self._valid_frame_idx(idx)
        if idx in self.excluded_frames:
            return self.include_frame(idx)
        return self.exclude_frame(idx)

    def clear_excluded(self) -> Self:
        self.excluded_frames = []
        return self

    def frame_is_excluded(self, idx: int) -> bool:
        return int(idx) in self.excluded_frames

    @property
    def active_frames(self) -> list[int]:
        """Indices of frames that are not marked excluded."""
        excluded = set(self.excluded_frames)
        return [i for i in range(self.n_slices) if i not in excluded]

    def delete_frame(self, idx: int) -> Self:
        """Remove a single frame from the stack.

        This rebuilds the internal array and relabels remaining frames.
        Labels, bookmarks, playback path, and exclusion marks are remapped
        to the new indices; indices pointing at the deleted frame are dropped.
        """
        return self.delete_frames([self._valid_frame_idx(idx)])

    def delete_frames(self, indices) -> Self:
        """Remove multiple frames from the stack.

        Parameters
        ----------
        indices : iterable of int
            Frame indices to remove. Duplicates are ignored.
        """
        to_drop = sorted({self._valid_frame_idx(i) for i in indices})
        if not to_drop:
            return self
        if len(to_drop) >= self.n_slices:
            raise ValueError("Cannot delete every frame — at least one must remain.")

        keep = [i for i in range(self.n_slices) if i not in set(to_drop)]
        remap = {old: new for new, old in enumerate(keep)}

        new_data = self._data[keep]
        new_labels = (
            [self.labels[i] for i in keep]
            if self.labels and len(self.labels) == self.n_slices
            else None
        )
        new_bookmarks = [remap[i] for i in self.bookmarked_frames if i in remap]
        new_path = [remap[i] for i in self.playback_path if i in remap]
        new_excluded = [remap[i] for i in self.excluded_frames if i in remap]

        # Try to keep the current frame visible — fall back to nearest surviving.
        if self.slice_idx in remap:
            next_idx = remap[self.slice_idx]
        else:
            # pick the nearest surviving neighbor
            candidates = [remap[i] for i in keep if i >= self.slice_idx]
            next_idx = candidates[0] if candidates else (len(keep) - 1)

        self.set_image(new_data, labels=new_labels)
        self.slice_idx = int(max(0, min(next_idx, self.n_slices - 1)))
        self.bookmarked_frames = new_bookmarks
        self.playback_path = new_path
        self.excluded_frames = new_excluded
        return self

    def reorder_frames(self, order) -> Self:
        """Reorder the stack using a permutation of frame indices.

        Parameters
        ----------
        order : sequence of int
            A permutation of ``range(n_slices)`` giving the new order. Each
            index must appear exactly once.
        """
        order = [int(i) for i in order]
        if sorted(order) != list(range(self.n_slices)):
            raise ValueError(
                f"order must be a permutation of range({self.n_slices}); got {order}"
            )
        if order == list(range(self.n_slices)):
            return self

        remap = {old: new for new, old in enumerate(order)}
        new_data = self._data[order]
        new_labels = (
            [self.labels[i] for i in order]
            if self.labels and len(self.labels) == self.n_slices
            else None
        )
        new_bookmarks = [remap[i] for i in self.bookmarked_frames if i in remap]
        new_path = [remap[i] for i in self.playback_path if i in remap]
        new_excluded = sorted(remap[i] for i in self.excluded_frames if i in remap)
        next_idx = remap.get(self.slice_idx, self.slice_idx)

        self.set_image(new_data, labels=new_labels)
        self.slice_idx = int(max(0, min(next_idx, self.n_slices - 1)))
        self.bookmarked_frames = new_bookmarks
        self.playback_path = new_path
        self.excluded_frames = new_excluded
        return self

    def batch_exclude_by(
        self,
        predicate=None,
        *,
        metric: str = "mean",
        lt: float | None = None,
        gt: float | None = None,
    ) -> Self:
        """Mark frames as excluded based on a per-frame statistic.

        Either supply ``predicate`` — a callable ``(idx, frame) -> bool`` that
        returns True for frames to exclude — or use the convenience ``metric``
        + ``lt``/``gt`` form to exclude frames whose per-frame statistic is
        below ``lt`` and/or above ``gt``.

        Parameters
        ----------
        predicate : callable, optional
            ``predicate(idx, frame_array) -> bool``. Overrides the threshold API.
        metric : {"mean", "min", "max", "std"}
            Per-frame reduction used when ``predicate`` is None.
        lt : float, optional
            Exclude frames where ``metric < lt``.
        gt : float, optional
            Exclude frames where ``metric > gt``.
        """
        if predicate is None and lt is None and gt is None:
            raise ValueError("Provide predicate or at least one of lt/gt.")

        if predicate is None:
            if metric not in ("mean", "min", "max", "std"):
                raise ValueError(
                    f"metric must be mean/min/max/std, got {metric!r}"
                )
            reducer = getattr(np, metric)

            def predicate(idx, frame):
                value = float(reducer(frame))
                if lt is not None and value < lt:
                    return True
                if gt is not None and value > gt:
                    return True
                return False

        new_excluded = set(self.excluded_frames)
        for i in range(self.n_slices):
            if predicate(i, self._data[i]):
                new_excluded.add(i)
        self.excluded_frames = sorted(new_excluded)
        return self

    def state_dict(self):
        return {
            "title": self.title,
            "cmap": self.cmap,
            "log_scale": self.log_scale,
            "auto_contrast": self.auto_contrast,
            "percentile_low": self.percentile_low,
            "percentile_high": self.percentile_high,
            "vmin": self.vmin,
            "vmax": self.vmax,
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
            "excluded_frames": list(self.excluded_frames),
            "roi_active": self.roi_active,
            "roi_list": self.roi_list,
            "roi_selected_idx": self.roi_selected_idx,
            "profile_line": self.profile_line,
            "profile_width": self.profile_width,
            "diff_mode": self.diff_mode,
            "dim_label": self.dim_label,
            "timestamp_unit": self.timestamp_unit,
            "display_bin": self._display_bin,
        }

    def save(self, path: str):
        save_state_file(path, "Show3D", self.state_dict())

    def load_state_dict(self, state):
        for key, val in state.items():
            if key == "display_bin":
                self._display_bin = val
                continue
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
        if self.disabled_tools:
            lines.append(f"Locked:   {', '.join(self.disabled_tools)}")
        if self.hidden_tools:
            lines.append(f"Hidden:   {', '.join(self.hidden_tools)}")
        lines.append(f"Playback: {self.fps} fps | loop={'on' if self.loop else 'off'} | reverse={'on' if self.reverse else 'off'} | boomerang={'on' if self.boomerang else 'off'}")
        if self.loop_start > 0 or self.loop_end >= 0:
            end = self.loop_end if self.loop_end >= 0 else self.n_slices - 1
            lines.append(f"Range:    {self.loop_start}–{end}")
        if self.excluded_frames:
            sample = ", ".join(str(i) for i in self.excluded_frames[:10])
            if len(self.excluded_frames) > 10:
                sample += ", …"
            lines.append(f"Excluded: {len(self.excluded_frames)}/{self.n_slices} [{sample}]")
        if self.roi_active and self.roi_list:
            lines.append(f"ROI:      {len(self.roi_list)} region(s)")
        if len(self.profile_line) >= 2:
            p0, p1 = self.profile_line[0], self.profile_line[1]
            lines.append(f"Profile:  ({p0['row']:.0f}, {p0['col']:.0f}) → ({p1['row']:.0f}, {p1['col']:.0f}) width={self.profile_width}")
        print("\n".join(lines))

    def _get_color_range(self, frame: np.ndarray) -> tuple[float, float]:
        """Get vmin/vmax based on current settings."""
        if self.vmin is not None and self.vmax is not None:
            vmin = float(self.vmin)
            vmax = float(self.vmax)
            if self.log_scale:
                vmin = float(np.log1p(max(vmin, 0)))
                vmax = float(np.log1p(max(vmax, 0)))
        elif self.auto_contrast:
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
        data = self._display_data
        frame = data[idx]
        if self.diff_mode == "previous":
            if idx == 0:
                return np.zeros_like(frame)
            return frame - data[idx - 1]
        if self.diff_mode == "first":
            return frame - data[0]
        return frame

    def _on_diff_mode_change(self, change=None):
        data = self._display_data
        if self.diff_mode == "off":
            self.data_min = float(data.min())
            self.data_max = float(data.max())
        elif self.diff_mode == "previous":
            # Vectorized diff: data[1:] - data[:-1]
            diffs = data[1:] - data[:-1]
            self.data_min = min(0.0, float(diffs.min()))
            self.data_max = float(diffs.max())
        elif self.diff_mode == "first":
            diffs = data[1:] - data[0:1]
            self.data_min = min(0.0, float(diffs.min()))
            self.data_max = float(diffs.max())
        else:
            self.data_min = float(data.min())
            self.data_max = float(data.max())
        self._update_all()

    def _update_all(self):
        """Update frame, stats, and all derived data. Uses hold_sync for batched transfer."""
        display_frame = self._get_display_frame()
        with self.hold_sync():
            # Stats from full-res data (not binned display)
            if self._use_torch:
                t = self._data_torch[self.slice_idx]
                self.stats_mean = float(t.mean().item())
                self.stats_min = float(t.min().item())
                self.stats_max = float(t.max().item())
                self.stats_std = float(t.std().item())
            else:
                full_frame = self._data[self.slice_idx]
                self.stats_mean = float(full_frame.mean())
                self.stats_min = float(full_frame.min())
                self.stats_max = float(full_frame.max())
                self.stats_std = float(full_frame.std())
            if self.timestamps and self.slice_idx < len(self.timestamps):
                self.current_timestamp = self.timestamps[self.slice_idx]
            if self.roi_active:
                self._update_roi_stats(display_frame)
            else:
                self.roi_stats = {}
            self.frame_bytes = display_frame.tobytes()

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
        """Handle ROI change. Stats for current frame are instant.
        Full-stack ROI plot is debounced (500ms) to avoid UI freeze during drag."""
        if self.roi_active:
            self._update_roi_stats(self._get_display_frame())
            # Debounce the expensive all-frame ROI plot
            if hasattr(self, '_roi_plot_timer') and self._roi_plot_timer is not None:
                self._roi_plot_timer.cancel()
            import threading
            self._roi_plot_timer = threading.Timer(0.5, self._compute_roi_plot)
            self._roi_plot_timer.start()
        else:
            self.roi_stats = {}
            self.roi_plot_data = b""

    def _compute_roi_plot(self):
        """Compute selected ROI mean for all frames. Uses display data (binned) for speed."""
        idx = self.roi_selected_idx
        if idx < 0 or idx >= len(self.roi_list):
            self.roi_plot_data = b""
            return
        mask = self._roi_mask(self.roi_list[idx])
        if mask.sum() == 0:
            self.roi_plot_data = b""
            return
        # Use _display_data (binned) — 4-16× less data than _data, same ROI result
        data = self._display_data
        if self._use_torch:
            mask_t = torch.from_numpy(mask).to(self._device)
            t = torch.from_numpy(data).to(self._device) if not hasattr(self, '_display_torch') else self._display_torch
            masked = t[:, mask_t]
            means = masked.mean(dim=1).cpu().numpy().astype(np.float32)
        else:
            means = np.array([float(data[i][mask].mean()) for i in range(self.n_slices)], dtype=np.float32)
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
