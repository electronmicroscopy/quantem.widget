"""
show4d: General-purpose 4D data explorer widget.

Interactive dual-panel viewer for any 4D dataset (nav_rows, nav_cols, sig_rows, sig_cols).
Left panel shows a navigation image (real-space), right panel shows the signal
at the selected position. Supports ROI masking on the navigation panel to
average signals from a region, path animation, and GPU acceleration.
"""

import json
import pathlib
from typing import Self

import numpy as np
import anywidget
import traitlets

from quantem.widget.array_utils import to_numpy
from quantem.widget.io import IOResult
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
    _HAS_TORCH = False

try:
    from quantem.core.config import validate_device
    _HAS_VALIDATE_DEVICE = True
except ImportError:
    _HAS_VALIDATE_DEVICE = False

class Show4D(anywidget.AnyWidget):
    """
    General-purpose 4D data explorer.

    Displays a navigation image (left) and signal at selected position (right).
    Click/drag on the navigation image to explore. Draw ROI masks to average
    signals from a region. Supports path animation for automated scanning.

    Parameters
    ----------
    data : array_like
        4D array of shape (nav_rows, nav_cols, sig_rows, sig_cols).
        Accepts NumPy, PyTorch, CuPy, or any np.asarray()-compatible object.
    nav_image : array_like, optional
        2D array (nav_rows, nav_cols) to use as navigation image.
        If not provided, defaults to mean over signal dimensions.
    title : str, optional
        Title displayed in the widget header.
    cmap : str, default "inferno"
        Colormap for the signal panel.
    log_scale : bool, default False
        Apply log scale to signal panel.
    auto_contrast : bool, default True
        Auto-contrast with percentile clipping on signal panel.
    show_stats : bool, default True
        Show statistics bar below canvases.
    show_fft : bool, default False
        Show FFT panel for signal.
    nav_pixel_size : float, optional
        Pixel size for navigation space scale bar.
    sig_pixel_size : float, optional
        Pixel size for signal space scale bar.
    nav_pixel_unit : str, default "px"
        Unit for navigation pixel size.
    sig_pixel_unit : str, default "px"
        Unit for signal pixel size.
    percentile_low : float, default 0.5
        Low percentile for auto-contrast clipping.
    percentile_high : float, default 99.5
        High percentile for auto-contrast clipping.
    disabled_tools : list of str, optional
        Tool groups to lock while still showing controls. Supported:
        ``"display"``, ``"roi"``, ``"histogram"``, ``"profile"``,
        ``"navigation"``, ``"playback"``, ``"stats"``, ``"export"``,
        ``"view"``, ``"fft"``, ``"all"``.
    disable_* : bool, optional
        Convenience flags mirroring ``disabled_tools`` for each tool group,
        plus ``disable_all``.
    hidden_tools : list of str, optional
        Tool groups to hide from the UI. Uses the same keys as
        ``disabled_tools``.
    hide_* : bool, optional
        Convenience flags mirroring ``disable_*`` for ``hidden_tools``.
    """

    _esm = pathlib.Path(__file__).parent / "static" / "show4d.js"
    _css = pathlib.Path(__file__).parent / "static" / "show4d.css"

    # Navigation position (row, col)
    pos_row = traitlets.Int(0).tag(sync=True)
    pos_col = traitlets.Int(0).tag(sync=True)

    # Shape
    nav_rows = traitlets.Int(1).tag(sync=True)
    nav_cols = traitlets.Int(1).tag(sync=True)
    sig_rows = traitlets.Int(1).tag(sync=True)
    sig_cols = traitlets.Int(1).tag(sync=True)

    # Data transfer (raw float32 bytes)
    frame_bytes = traitlets.Bytes(b"").tag(sync=True)
    nav_image_bytes = traitlets.Bytes(b"").tag(sync=True)

    # Data ranges for JS normalization
    nav_data_min = traitlets.Float(0.0).tag(sync=True)
    nav_data_max = traitlets.Float(1.0).tag(sync=True)
    sig_data_min = traitlets.Float(0.0).tag(sync=True)
    sig_data_max = traitlets.Float(1.0).tag(sync=True)

    # ROI on navigation panel
    roi_mode = traitlets.Unicode("off").tag(sync=True)  # "off", "circle", "square", "rect", "annular"
    roi_reduce = traitlets.Unicode("mean").tag(sync=True)  # "mean", "max", "min", "sum"
    roi_center_row = traitlets.Float(0.0).tag(sync=True)
    roi_center_col = traitlets.Float(0.0).tag(sync=True)
    roi_center = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0]).tag(sync=True)
    roi_radius = traitlets.Float(5.0).tag(sync=True)
    roi_radius_inner = traitlets.Float(0.0).tag(sync=True)
    roi_width = traitlets.Float(10.0).tag(sync=True)
    roi_height = traitlets.Float(10.0).tag(sync=True)

    # Statistics ([mean, min, max, std])
    nav_stats = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0, 0.0, 0.0]).tag(sync=True)
    sig_stats = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0, 0.0, 0.0]).tag(sync=True)

    # Display traits (synced to Python)
    cmap = traitlets.Unicode("inferno").tag(sync=True)
    log_scale = traitlets.Bool(False).tag(sync=True)
    auto_contrast = traitlets.Bool(True).tag(sync=True)
    show_stats = traitlets.Bool(True).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)
    show_fft = traitlets.Bool(False).tag(sync=True)
    fft_window = traitlets.Bool(True).tag(sync=True)
    disabled_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)
    hidden_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)
    percentile_low = traitlets.Float(0.5).tag(sync=True)
    percentile_high = traitlets.Float(99.5).tag(sync=True)
    # Scale bars
    nav_pixel_size = traitlets.Float(0.0).tag(sync=True)
    sig_pixel_size = traitlets.Float(0.0).tag(sync=True)
    nav_pixel_unit = traitlets.Unicode("px").tag(sync=True)
    sig_pixel_unit = traitlets.Unicode("px").tag(sync=True)

    # Title
    title = traitlets.Unicode("").tag(sync=True)

    # Snap-to-peak
    snap_enabled = traitlets.Bool(False).tag(sync=True)
    snap_radius = traitlets.Int(5).tag(sync=True)

    # Path animation
    path_playing = traitlets.Bool(False).tag(sync=True)
    path_index = traitlets.Int(0).tag(sync=True)
    path_length = traitlets.Int(0).tag(sync=True)
    path_interval_ms = traitlets.Int(100).tag(sync=True)
    path_loop = traitlets.Bool(True).tag(sync=True)

    # Export (GIF)
    _gif_export_requested = traitlets.Bool(False).tag(sync=True)
    _gif_data = traitlets.Bytes(b"").tag(sync=True)
    _gif_metadata_json = traitlets.Unicode("").tag(sync=True)

    # Line Profile
    profile_line = traitlets.List(traitlets.Dict()).tag(sync=True)
    profile_width = traitlets.Int(1).tag(sync=True)

    @classmethod
    def _normalize_tool_groups(cls, tool_groups):
        return normalize_tool_groups("Show4D", tool_groups)

    @classmethod
    def _build_disabled_tools(
        cls,
        disabled_tools=None,
        disable_display: bool = False,
        disable_roi: bool = False,
        disable_histogram: bool = False,
        disable_profile: bool = False,
        disable_navigation: bool = False,
        disable_playback: bool = False,
        disable_stats: bool = False,
        disable_export: bool = False,
        disable_view: bool = False,
        disable_fft: bool = False,
        disable_all: bool = False,
    ):
        return build_tool_groups(
            "Show4D",
            tool_groups=disabled_tools,
            all_flag=disable_all,
            flag_map={
                "display": disable_display,
                "roi": disable_roi,
                "histogram": disable_histogram,
                "profile": disable_profile,
                "navigation": disable_navigation,
                "playback": disable_playback,
                "stats": disable_stats,
                "export": disable_export,
                "view": disable_view,
                "fft": disable_fft,
            },
        )

    @classmethod
    def _build_hidden_tools(
        cls,
        hidden_tools=None,
        hide_display: bool = False,
        hide_roi: bool = False,
        hide_histogram: bool = False,
        hide_profile: bool = False,
        hide_navigation: bool = False,
        hide_playback: bool = False,
        hide_stats: bool = False,
        hide_export: bool = False,
        hide_view: bool = False,
        hide_fft: bool = False,
        hide_all: bool = False,
    ):
        return build_tool_groups(
            "Show4D",
            tool_groups=hidden_tools,
            all_flag=hide_all,
            flag_map={
                "display": hide_display,
                "roi": hide_roi,
                "histogram": hide_histogram,
                "profile": hide_profile,
                "navigation": hide_navigation,
                "playback": hide_playback,
                "stats": hide_stats,
                "export": hide_export,
                "view": hide_view,
                "fft": hide_fft,
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
        data,
        nav_image=None,
        title="",
        cmap="inferno",
        log_scale=False,
        auto_contrast=True,
        show_stats=True,
        show_controls=True,
        show_fft=False,
        fft_window=True,
        nav_pixel_size=None,
        sig_pixel_size=None,
        nav_pixel_unit="px",
        sig_pixel_unit="px",
        percentile_low=0.5,
        percentile_high=99.5,
        snap_enabled=False,
        snap_radius=5,
        disabled_tools=None,
        disable_display=False,
        disable_roi=False,
        disable_histogram=False,
        disable_profile=False,
        disable_navigation=False,
        disable_playback=False,
        disable_stats=False,
        disable_export=False,
        disable_view=False,
        disable_fft=False,
        disable_all=False,
        hidden_tools=None,
        hide_display=False,
        hide_roi=False,
        hide_histogram=False,
        hide_profile=False,
        hide_navigation=False,
        hide_playback=False,
        hide_stats=False,
        hide_export=False,
        hide_view=False,
        hide_fft=False,
        hide_all=False,
        state=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.widget_version = resolve_widget_version()

        # Check if data is an IOResult and extract metadata
        if isinstance(data, IOResult):
            if not title and data.title:
                title = data.title
            data = data.data

        # Dataset duck typing
        if hasattr(data, "array") and hasattr(data, "sampling"):
            units = getattr(data, "units", ["pixels"] * 4)
            if nav_pixel_size is None and units[0] in ("Å", "angstrom", "A"):
                nav_pixel_size = float(data.sampling[0])
                nav_pixel_unit = "Å"
            elif nav_pixel_size is None and units[0] == "nm":
                nav_pixel_size = float(data.sampling[0]) * 10
                nav_pixel_unit = "Å"
            if sig_pixel_size is None and len(units) >= 4:
                if units[2] in ("Å", "angstrom", "A"):
                    sig_pixel_size = float(data.sampling[2])
                    sig_pixel_unit = "Å"
                elif units[2] == "nm":
                    sig_pixel_size = float(data.sampling[2]) * 10
                    sig_pixel_unit = "Å"
                elif units[2] == "mrad":
                    sig_pixel_size = float(data.sampling[2])
                    sig_pixel_unit = "mrad"
            if not title and hasattr(data, "name") and data.name:
                title = data.name
            data = data.array

        # Convert to NumPy float32
        data_np = to_numpy(data).astype(np.float32)

        if data_np.ndim != 4:
            raise ValueError(
                f"Expected 4D array (nav_rows, nav_cols, sig_rows, sig_cols), got {data_np.ndim}D"
            )

        self.nav_rows = data_np.shape[0]
        self.nav_cols = data_np.shape[1]
        self.sig_rows = data_np.shape[2]
        self.sig_cols = data_np.shape[3]
        self.title = title

        # Display traits
        self.cmap = str(cmap)
        self.log_scale = log_scale
        self.auto_contrast = auto_contrast
        self.show_stats = show_stats
        self.show_controls = show_controls
        self.show_fft = show_fft
        self.fft_window = fft_window
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.snap_enabled = snap_enabled
        self.snap_radius = snap_radius
        self.disabled_tools = self._build_disabled_tools(
            disabled_tools=disabled_tools,
            disable_display=disable_display,
            disable_roi=disable_roi,
            disable_histogram=disable_histogram,
            disable_profile=disable_profile,
            disable_navigation=disable_navigation,
            disable_playback=disable_playback,
            disable_stats=disable_stats,
            disable_export=disable_export,
            disable_view=disable_view,
            disable_fft=disable_fft,
            disable_all=disable_all,
        )
        self.hidden_tools = self._build_hidden_tools(
            hidden_tools=hidden_tools,
            hide_display=hide_display,
            hide_roi=hide_roi,
            hide_histogram=hide_histogram,
            hide_profile=hide_profile,
            hide_navigation=hide_navigation,
            hide_playback=hide_playback,
            hide_stats=hide_stats,
            hide_export=hide_export,
            hide_view=hide_view,
            hide_fft=hide_fft,
            hide_all=hide_all,
        )

        # Scale bar
        self.nav_pixel_size = nav_pixel_size if nav_pixel_size is not None else 0.0
        self.sig_pixel_size = sig_pixel_size if sig_pixel_size is not None else 0.0
        self.nav_pixel_unit = nav_pixel_unit
        self.sig_pixel_unit = sig_pixel_unit

        # Store data — GPU (PyTorch) if available, else NumPy
        if _HAS_TORCH:
            if _HAS_VALIDATE_DEVICE:
                device_str, _ = validate_device(None)
                self._device = torch.device(device_str)
            else:
                self._device = torch.device(
                    "mps" if torch.backends.mps.is_available()
                    else "cuda" if torch.cuda.is_available()
                    else "cpu"
                )
            self._data = torch.from_numpy(data_np).to(self._device)
            # Cache coordinate tensors for fast mask creation
            self._nav_row_coords = torch.arange(
                self.nav_rows, device=self._device, dtype=torch.float32
            )[:, None]
            self._nav_col_coords = torch.arange(
                self.nav_cols, device=self._device, dtype=torch.float32
            )[None, :]
        else:
            self._data = data_np
            self._device = None

        # Compute navigation image
        if nav_image is not None:
            nav_img = to_numpy(nav_image).astype(np.float32)
        elif _HAS_TORCH and isinstance(self._data, torch.Tensor):
            nav_img = self._data.mean(dim=(2, 3)).cpu().numpy()
        else:
            nav_img = data_np.mean(axis=(2, 3))
        self._nav_image = nav_img

        # Compute global signal range
        if _HAS_TORCH and isinstance(self._data, torch.Tensor):
            self.sig_data_min = float(self._data.min())
            self.sig_data_max = float(self._data.max())
        else:
            n_total = self.nav_rows * self.nav_cols
            if n_total > 100:
                rng = np.random.default_rng(42)
                indices = rng.choice(n_total, 100, replace=False)
                sampled = data_np.reshape(n_total, self.sig_rows, self.sig_cols)[indices]
                self.sig_data_min = float(sampled.min())
                self.sig_data_max = float(sampled.max())
            else:
                self.sig_data_min = float(data_np.min())
                self.sig_data_max = float(data_np.max())

        # Nav image range and bytes (sent once)
        self.nav_data_min = float(nav_img.min())
        self.nav_data_max = float(nav_img.max())
        self.nav_image_bytes = nav_img.tobytes()
        self.nav_stats = [
            float(nav_img.mean()), float(nav_img.min()),
            float(nav_img.max()), float(nav_img.std()),
        ]

        # Initial position at center
        self.pos_row = self.nav_rows // 2
        self.pos_col = self.nav_cols // 2

        # ROI defaults
        default_roi_size = max(3, min(self.nav_rows, self.nav_cols) * 0.15)
        self.roi_center_row = float(self.nav_rows / 2)
        self.roi_center_col = float(self.nav_cols / 2)
        self.roi_center = [float(self.nav_rows / 2), float(self.nav_cols / 2)]
        self.roi_radius = float(default_roi_size)
        self.roi_radius_inner = float(default_roi_size * 0.5)
        self.roi_width = float(default_roi_size * 2)
        self.roi_height = float(default_roi_size)

        # Path animation state
        self._path_points: list[tuple[int, int]] = []

        # Observers
        self.observe(self._update_frame, names=["pos_row", "pos_col"])
        self.observe(self._on_roi_change, names=[
            "roi_mode", "roi_reduce", "roi_center_row", "roi_center_col",
            "roi_radius", "roi_radius_inner", "roi_width", "roi_height",
        ])
        self.observe(self._on_roi_center_change, names=["roi_center"])
        self.observe(self._on_path_index_change, names=["path_index"])
        self.observe(self._on_gif_export, names=["_gif_export_requested"])

        # Initial frame
        self._update_frame()

        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(
                    json.loads(pathlib.Path(state).read_text()),
                    require_envelope=True,
                )
            else:
                state = unwrap_state_payload(state)
            self.load_state_dict(state)

    def set_image(self, data, nav_image=None):
        """Replace the 4D data. Preserves all display settings."""
        if hasattr(data, "array") and hasattr(data, "sampling"):
            data = data.array
        data_np = to_numpy(data).astype(np.float32)
        if data_np.ndim != 4:
            raise ValueError(
                f"Expected 4D array (nav_rows, nav_cols, sig_rows, sig_cols), got {data_np.ndim}D"
            )
        self.nav_rows = data_np.shape[0]
        self.nav_cols = data_np.shape[1]
        self.sig_rows = data_np.shape[2]
        self.sig_cols = data_np.shape[3]
        if _HAS_TORCH:
            self._data = torch.from_numpy(data_np).to(self._device)
            self._nav_row_coords = torch.arange(
                self.nav_rows, device=self._device, dtype=torch.float32
            )[:, None]
            self._nav_col_coords = torch.arange(
                self.nav_cols, device=self._device, dtype=torch.float32
            )[None, :]
        else:
            self._data = data_np
        if nav_image is not None:
            nav_img = to_numpy(nav_image).astype(np.float32)
        elif _HAS_TORCH and isinstance(self._data, torch.Tensor):
            nav_img = self._data.mean(dim=(2, 3)).cpu().numpy()
        else:
            nav_img = data_np.mean(axis=(2, 3))
        self._nav_image = nav_img
        if _HAS_TORCH and isinstance(self._data, torch.Tensor):
            self.sig_data_min = float(self._data.min())
            self.sig_data_max = float(self._data.max())
        else:
            self.sig_data_min = float(data_np.min())
            self.sig_data_max = float(data_np.max())
        self.nav_data_min = float(nav_img.min())
        self.nav_data_max = float(nav_img.max())
        self.nav_image_bytes = nav_img.tobytes()
        self.nav_stats = [
            float(nav_img.mean()), float(nav_img.min()),
            float(nav_img.max()), float(nav_img.std()),
        ]
        self.pos_row = min(self.pos_row, self.nav_rows - 1)
        self.pos_col = min(self.pos_col, self.nav_cols - 1)
        self._update_frame()

    def __repr__(self) -> str:
        return (
            f"Show4D(shape=({self.nav_rows}, {self.nav_cols}, {self.sig_rows}, {self.sig_cols}), "
            f"pos=({self.pos_row}, {self.pos_col}))"
        )

    def state_dict(self):
        return {
            "title": self.title,
            "cmap": self.cmap,
            "log_scale": self.log_scale,
            "auto_contrast": self.auto_contrast,
            "show_stats": self.show_stats,
            "show_controls": self.show_controls,
            "show_fft": self.show_fft,
            "fft_window": self.fft_window,
            "disabled_tools": self.disabled_tools,
            "hidden_tools": self.hidden_tools,
            "percentile_low": self.percentile_low,
            "percentile_high": self.percentile_high,
            "nav_pixel_size": self.nav_pixel_size,
            "sig_pixel_size": self.sig_pixel_size,
            "nav_pixel_unit": self.nav_pixel_unit,
            "sig_pixel_unit": self.sig_pixel_unit,
            "roi_mode": self.roi_mode,
            "roi_reduce": self.roi_reduce,
            "roi_center_row": self.roi_center_row,
            "roi_center_col": self.roi_center_col,
            "roi_radius": self.roi_radius,
            "roi_radius_inner": self.roi_radius_inner,
            "roi_width": self.roi_width,
            "roi_height": self.roi_height,
            "snap_enabled": self.snap_enabled,
            "snap_radius": self.snap_radius,
            "path_interval_ms": self.path_interval_ms,
            "path_loop": self.path_loop,
            "profile_line": self.profile_line,
            "profile_width": self.profile_width,
        }

    def save(self, path: str):
        save_state_file(path, "Show4D", self.state_dict())

    def load_state_dict(self, state):
        for key, val in state.items():
            if hasattr(self, key):
                setattr(self, key, val)

    def summary(self):
        lines = [self.title or "Show4D", "═" * 32]
        lines.append(f"Nav:      {self.nav_rows}×{self.nav_cols}")
        if self.nav_pixel_size > 0:
            lines[-1] += f" ({self.nav_pixel_size:.2f} {self.nav_pixel_unit}/px)"
        lines.append(f"Signal:   {self.sig_rows}×{self.sig_cols}")
        if self.sig_pixel_size > 0:
            lines[-1] += f" ({self.sig_pixel_size:.2f} {self.sig_pixel_unit}/px)"
        lines.append(f"Position: ({self.pos_row}, {self.pos_col})")
        cmap = self.cmap
        scale = "log" if self.log_scale else "linear"
        contrast = "auto contrast" if self.auto_contrast else "manual contrast"
        display = f"{cmap} | {contrast} | {scale}"
        if self.show_fft:
            display += " | FFT"
            if not self.fft_window:
                display += " (no window)"
        lines.append(f"Display:  {display}")
        if self.roi_mode != "off":
            lines.append(f"ROI:      {self.roi_mode} ({self.roi_reduce}) at ({self.roi_center_row:.1f}, {self.roi_center_col:.1f}) r={self.roi_radius:.1f}")
        if self.snap_enabled:
            lines.append(f"Snap:     ON (radius={self.snap_radius} px)")
        if self.profile_line and len(self.profile_line) == 2:
            p0, p1 = self.profile_line[0], self.profile_line[1]
            lines.append(f"Profile:  ({p0['row']:.0f}, {p0['col']:.0f}) -> ({p1['row']:.0f}, {p1['col']:.0f}) width={self.profile_width}")
        if self.disabled_tools:
            lines.append(f"Locked:   {', '.join(self.disabled_tools)}")
        if self.hidden_tools:
            lines.append(f"Hidden:   {', '.join(self.hidden_tools)}")
        print("\n".join(lines))

    @property
    def position(self) -> tuple[int, int]:
        return (self.pos_row, self.pos_col)

    @position.setter
    def position(self, value: tuple[int, int]) -> None:
        self.pos_row, self.pos_col = value

    @property
    def nav_shape(self) -> tuple[int, int]:
        return (self.nav_rows, self.nav_cols)

    @property
    def sig_shape(self) -> tuple[int, int]:
        return (self.sig_rows, self.sig_cols)

    # ── Line Profile ────────────────────────────────────────────────────────

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
        if _HAS_TORCH and isinstance(self._data, torch.Tensor):
            frame = self._data[self.pos_row, self.pos_col].cpu().numpy()
        else:
            frame = np.asarray(self._data[self.pos_row, self.pos_col], dtype=np.float32)
        return self._sample_line(frame, p0["row"], p0["col"], p1["row"], p1["col"])

    @property
    def profile_distance(self) -> float:
        if len(self.profile_line) != 2:
            return 0.0
        p0, p1 = self.profile_line[0], self.profile_line[1]
        dist_px = np.sqrt((p1["row"] - p0["row"]) ** 2 + (p1["col"] - p0["col"]) ** 2)
        if self.sig_pixel_size > 0:
            return float(dist_px * self.sig_pixel_size)
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

    # ── Path Animation ──────────────────────────────────────────────────────

    def set_path(
        self,
        points: list[tuple[int, int]],
        interval_ms: int = 100,
        loop: bool = True,
        autoplay: bool = True,
    ) -> Self:
        self._path_points = list(points)
        self.path_length = len(self._path_points)
        self.path_index = 0
        self.path_interval_ms = interval_ms
        self.path_loop = loop
        if autoplay and self.path_length > 0:
            self.path_playing = True
        return self

    def play(self) -> Self:
        if self.path_length > 0:
            self.path_playing = True
        return self

    def pause(self) -> Self:
        self.path_playing = False
        return self

    def stop(self) -> Self:
        self.path_playing = False
        self.path_index = 0
        return self

    def goto(self, index: int) -> Self:
        if 0 <= index < self.path_length:
            self.path_index = index
        return self

    def raster(
        self,
        step: int = 1,
        bidirectional: bool = False,
        interval_ms: int = 100,
        loop: bool = True,
    ) -> Self:
        points = []
        for r in range(0, self.nav_rows, step):
            cols = list(range(0, self.nav_cols, step))
            if bidirectional and (r // step % 2 == 1):
                cols = cols[::-1]
            for c in cols:
                points.append((r, c))
        return self.set_path(points=points, interval_ms=interval_ms, loop=loop)

    # ── Export (GIF) ──────────────────────────────────────────────────────────

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.log_scale:
            frame = np.log1p(np.maximum(frame, 0))
        fmin, fmax = float(frame.min()), float(frame.max())
        if self.auto_contrast:
            fmin = float(np.percentile(frame, self.percentile_low))
            fmax = float(np.percentile(frame, self.percentile_high))
        if fmax > fmin:
            return np.clip((frame - fmin) / (fmax - fmin) * 255, 0, 255).astype(np.uint8)
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

        cmap_fn = colormaps.get_cmap(self.cmap)
        duration_ms = max(10, self.path_interval_ms)

        pil_frames = []
        for row, col in self._path_points:
            row = max(0, min(self.nav_rows - 1, row))
            col = max(0, min(self.nav_cols - 1, col))
            if _HAS_TORCH and isinstance(self._data, torch.Tensor):
                frame = self._data[row, col].cpu().numpy()
            else:
                frame = np.asarray(self._data[row, col], dtype=np.float32)
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
            **build_json_header("Show4D"),
            "view": "signal",
            "format": "gif",
            "export_kind": "path_animation",
            "n_frames": int(len(pil_frames)),
            "duration_ms": int(duration_ms),
            "path_loop": bool(self.path_loop),
            "path_points": [{"row": int(row), "col": int(col)} for row, col in self._path_points],
            "scan_shape": {"rows": int(self.nav_rows), "cols": int(self.nav_cols)},
            "detector_shape": {"rows": int(self.sig_rows), "cols": int(self.sig_cols)},
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

    # ── Save Image ────────────────────────────────────────────────────────────

    def save_image(
        self,
        path: str | pathlib.Path,
        *,
        view: str | None = None,
        position: tuple[int, int] | None = None,
        format: str | None = None,
        dpi: int = 150,
    ) -> pathlib.Path:
        """Save current signal or navigation image as PNG, PDF, or TIFF.

        Parameters
        ----------
        path : str or pathlib.Path
            Output file path.
        view : str, optional
            'signal' or 'nav'. Defaults to 'signal'.
        position : tuple[int, int], optional
            Navigation position as (row, col) for the signal view.
            Defaults to current position.
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

        view_key = (view or "signal").lower()
        if view_key not in ("signal", "nav"):
            raise ValueError(f"Unknown view: {view_key!r}. Use 'signal' or 'nav'.")

        if view_key == "signal":
            row, col = position if position is not None else (self.pos_row, self.pos_col)
            if row < 0 or row >= self.nav_rows or col < 0 or col >= self.nav_cols:
                raise IndexError(
                    f"Position ({row}, {col}) out of range "
                    f"[0, {self.nav_rows}) x [0, {self.nav_cols})"
                )
            if _HAS_TORCH and isinstance(self._data, torch.Tensor):
                frame = self._data[row, col].cpu().numpy()
            else:
                frame = np.asarray(self._data[row, col], dtype=np.float32)
            normalized = self._normalize_frame(frame)
        else:
            frame = self._nav_image
            nav_min, nav_max = float(frame.min()), float(frame.max())
            if nav_max > nav_min:
                normalized = np.clip(
                    (frame - nav_min) / (nav_max - nav_min) * 255, 0, 255
                ).astype(np.uint8)
            else:
                normalized = np.zeros(frame.shape, dtype=np.uint8)

        cmap_fn = colormaps.get_cmap(self.cmap)
        rgba = (cmap_fn(normalized / 255.0) * 255).astype(np.uint8)

        img = Image.fromarray(rgba)
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path), dpi=(dpi, dpi))
        return path

    # ── Internal Observers ───────────────────────────────────────────────────

    def _update_frame(self, change=None):
        if _HAS_TORCH and isinstance(self._data, torch.Tensor):
            frame = self._data[self.pos_row, self.pos_col].cpu().numpy()
        else:
            frame = self._data[self.pos_row, self.pos_col]
        frame = np.asarray(frame, dtype=np.float32)
        with self.hold_sync():
            self.sig_stats = [
                float(frame.mean()), float(frame.min()),
                float(frame.max()), float(frame.std()),
            ]
            self.frame_bytes = frame.tobytes()

    def _on_roi_change(self, change=None):
        if self.roi_mode == "off":
            self._update_frame()
            return
        self._compute_roi_signal()

    def _on_roi_center_change(self, change=None):
        if self.roi_mode == "off":
            return
        if change and "new" in change:
            row, col = change["new"]
            self.unobserve(self._on_roi_change, names=["roi_center_row", "roi_center_col"])
            self.roi_center_row = row
            self.roi_center_col = col
            self.observe(self._on_roi_change, names=["roi_center_row", "roi_center_col"])
        self._compute_roi_signal()

    def _on_path_index_change(self, change):
        idx = change["new"]
        if 0 <= idx < len(self._path_points):
            row, col = self._path_points[idx]
            self.pos_row = max(0, min(self.nav_rows - 1, row))
            self.pos_col = max(0, min(self.nav_cols - 1, col))

    def _compute_roi_signal(self):
        cr, cc = self.roi_center_row, self.roi_center_col

        if _HAS_TORCH and isinstance(self._data, torch.Tensor):
            # GPU path
            row_coords = self._nav_row_coords
            col_coords = self._nav_col_coords

            if self.roi_mode == "circle":
                mask = (row_coords - cr) ** 2 + (col_coords - cc) ** 2 <= self.roi_radius ** 2
            elif self.roi_mode == "square":
                mask = (torch.abs(row_coords - cr) <= self.roi_radius) & (torch.abs(col_coords - cc) <= self.roi_radius)
            elif self.roi_mode == "rect":
                mask = (torch.abs(row_coords - cr) <= self.roi_height / 2) & (torch.abs(col_coords - cc) <= self.roi_width / 2)
            elif self.roi_mode == "annular":
                dist_sq = (row_coords - cr) ** 2 + (col_coords - cc) ** 2
                mask = (dist_sq >= self.roi_radius_inner ** 2) & (dist_sq <= self.roi_radius ** 2)
            else:
                self._update_frame()
                return

            n_positions = int(mask.sum())
            if n_positions == 0:
                self._update_frame()
                return

            masked_data = self._data[mask]
            reduce_ops = {
                "mean": lambda x: x.mean(dim=0),
                "max": lambda x: x.max(dim=0).values,
                "min": lambda x: x.min(dim=0).values,
                "sum": lambda x: x.sum(dim=0),
            }
            op = reduce_ops.get(self.roi_reduce, reduce_ops["mean"])
            result = op(masked_data).cpu().numpy().astype(np.float32)
        else:
            # NumPy fallback
            row_coords = np.arange(self.nav_rows)[:, None]
            col_coords = np.arange(self.nav_cols)[None, :]

            if self.roi_mode == "circle":
                mask = (row_coords - cr) ** 2 + (col_coords - cc) ** 2 <= self.roi_radius ** 2
            elif self.roi_mode == "square":
                mask = (np.abs(row_coords - cr) <= self.roi_radius) & (np.abs(col_coords - cc) <= self.roi_radius)
            elif self.roi_mode == "rect":
                mask = (np.abs(row_coords - cr) <= self.roi_height / 2) & (np.abs(col_coords - cc) <= self.roi_width / 2)
            elif self.roi_mode == "annular":
                dist_sq = (row_coords - cr) ** 2 + (col_coords - cc) ** 2
                mask = (dist_sq >= self.roi_radius_inner ** 2) & (dist_sq <= self.roi_radius ** 2)
            else:
                self._update_frame()
                return

            n_positions = int(mask.sum())
            if n_positions == 0:
                self._update_frame()
                return

            reduce_ops = {"mean": np.mean, "max": np.max, "min": np.min, "sum": np.sum}
            op = reduce_ops.get(self.roi_reduce, np.mean)
            result = op(self._data[mask], axis=0).astype(np.float32)

        with self.hold_sync():
            self.sig_stats = [
                float(result.mean()), float(result.min()),
                float(result.max()), float(result.std()),
            ]
            self.frame_bytes = result.tobytes()


bind_tool_runtime_api(Show4D, "Show4D")
