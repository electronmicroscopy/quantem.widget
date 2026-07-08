"""Interactive 1D traces with reconstruction-monitor helpers.

``Show1D`` is the line/profile companion to ``Show2D``.  It handles ordinary
1D traces, but also keeps enough linked state for live torch ptychography:
scalar histories can be appended during reconstruction, image snapshots can be
attached to iterations, and a 2D image line profile can be sampled into the
trace view.
"""

from __future__ import annotations

import csv
import json
import math
import pathlib
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any, Self

import anywidget
import ipywidgets
import numpy as np
import traitlets

from quantem.widget.export import ensure_mobile_viewport
from quantem.widget.utils.array import _b64_safe, to_numpy
from quantem.widget.utils.state_io import resolve_widget_version, save_state_file, unwrap_state_payload
from quantem.widget.utils.ui import UiMode, resolve_ui_mode

_DEFAULT_COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#999999",
]

_VALID_IMAGE_CMAPS = {
    "inferno",
    "viridis",
    "plasma",
    "magma",
    "hot",
    "gray",
    "hsv",
    "turbo",
    "RdBu",
    "cividis",
    "seismic",
    "RdBu_r",
    "twilight",
    "twilight_shifted",
}

_VALID_SNAPSHOT_CONTRAST_PRESETS = {
    "full",
    "0.5-99.5",
    "1-99",
    "2-98",
    "5-95",
}


def _as_float(value: Any) -> float:
    """Convert Python, NumPy, or torch scalar-ish values to a finite float/NaN."""

    if value is None:
        return float("nan")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _json_safe(value: Any) -> Any:
    """Convert nested values to strict JSON-compatible Python objects."""

    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, pathlib.Path):
        return str(value)
    return value


def _slug(value: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "show1d"


def _method_sort_key(method: str) -> tuple[int, float, str]:
    if method == "frame_by_frame":
        return (0, -math.inf, method)
    if method.startswith("joint_lambda_"):
        raw = method.removeprefix("joint_lambda_")
        try:
            return (1, float(raw), method)
        except ValueError:
            return (1, math.inf, method)
    return (2, math.inf, method)


def _pretty_metric_label(key: str) -> str:
    labels = {
        "rmse_per_frame_mask": "per-frame RMSE",
        "rmse_time_average_mask": "average RMSE",
        "temporal_flicker_mask": "temporal flicker",
        "mean_phase_std_mask": "mean phase std",
        "elapsed_s": "elapsed",
        "reference_final_loss": "reference final loss",
    }
    return labels.get(key, key.replace("_", " "))


def _pretty_method_label(method: str) -> str:
    if method == "frame_by_frame":
        return "frame-by-frame"
    if method.startswith("joint_lambda_"):
        return f"lambda {method.removeprefix('joint_lambda_')}"
    return method.replace("_", " ")


def _sample_single_line(
    image: np.ndarray,
    row0: float,
    col0: float,
    row1: float,
    col1: float,
) -> np.ndarray:
    """Sample a 2D image along a line using bilinear interpolation."""

    h, w = image.shape
    dc = float(col1) - float(col0)
    dr = float(row1) - float(row0)
    length = math.hypot(dc, dr)
    n = max(2, int(math.ceil(length)) + 1)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        t = i / (n - 1)
        col = float(col0) + t * dc
        row = float(row0) + t * dr
        ci = math.floor(col)
        ri = math.floor(row)
        cf = col - ci
        rf = row - ri
        c0c = max(0, min(w - 1, ci))
        c1c = max(0, min(w - 1, ci + 1))
        r0c = max(0, min(h - 1, ri))
        r1c = max(0, min(h - 1, ri + 1))
        out[i] = (
            image[r0c, c0c] * (1 - cf) * (1 - rf)
            + image[r0c, c1c] * cf * (1 - rf)
            + image[r1c, c0c] * (1 - cf) * rf
            + image[r1c, c1c] * cf * rf
        )
    return out


def sample_line_profile(
    image: np.ndarray,
    line: Sequence[Sequence[float]],
    *,
    profile_width: int = 1,
) -> np.ndarray:
    """Return values sampled along ``line`` in ``(row, col)`` image coordinates."""

    arr = np.asarray(to_numpy(image), dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {arr.shape}")
    if len(line) != 2 or len(line[0]) != 2 or len(line[1]) != 2:
        raise ValueError("line must be ((row0, col0), (row1, col1))")
    (row0, col0), (row1, col1) = line
    width = max(1, int(round(profile_width)))
    if width <= 1:
        return _sample_single_line(arr, row0, col0, row1, col1)

    dc = float(col1) - float(col0)
    dr = float(row1) - float(row0)
    length = math.hypot(dc, dr)
    if length < 1e-8:
        return _sample_single_line(arr, row0, col0, row1, col1)
    perp_r = -dc / length
    perp_c = dr / length
    half = (width - 1) / 2
    acc: np.ndarray | None = None
    for k in range(width):
        offset = -half + k
        vals = _sample_single_line(
            arr,
            float(row0) + offset * perp_r,
            float(col0) + offset * perp_c,
            float(row1) + offset * perp_r,
            float(col1) + offset * perp_c,
        )
        acc = vals if acc is None else acc + vals
    return acc / width if acc is not None else np.empty(0, dtype=np.float32)


class Show1D(anywidget.AnyWidget):
    """Interactive 1D viewer for traces, line profiles, and live reconstruction.

    Parameters
    ----------
    data : array_like, mapping, or None
        A 1D array, a 2D ``(n_traces, n_points)`` array, a list of 1D arrays, a
        mapping of ``label -> trace``, or ``None`` for a live empty monitor.
    x : array_like, optional
        Shared x positions. Defaults to point indices. Empty monitors start
        without x values and fill them as ``append`` is called.
    labels : list of str, optional
        Per-trace labels.
    colors : list of str, optional
        Per-trace CSS colors.
    title, x_label, y_label, x_unit, y_unit : str, optional
        Plot metadata shown in the widget and exported figures.
    log_scale : bool, default False
        Use logarithmic y display. Non-positive values are skipped in the plot.
    ui_mode : {"interactive", "presentation", "report", "minimal"}, default "interactive"
        Shared viewer UI preset. Explicit ``show_*`` keyword arguments override
        preset values.
    show_title, show_stats, show_review, show_legend, show_grid, show_controls : bool
        Toggle compact plot UI elements.
    controls_collapsed : bool, default False
        Start with controls hidden while keeping a recoverable ``Controls``
        button in the frontend.
    plot_height_px, side_panel_width_px : int, optional
        Initial plot height and snapshot/stats side-panel width in pixels.
    image_cmap : str, default "cividis"
        Colormap used for profile and snapshot images.
    snapshot_contrast_preset : {"full", "0.5-99.5", "1-99", "2-98", "5-95"}, default "full"
        Percentile contrast preset used for snapshot images and plot thumbnails.
    snapshot_contrast_range : sequence of 2 floats, optional
        Explicit snapshot display range. Empty uses ``snapshot_contrast_preset``;
        a two-value ``(min, max)`` range enables the draggable histogram clip.
    show_snapshot_profile : bool, default False
        Show an interactive line-profile overlay on snapshot reconstruction
        panels and a compact profile comparison below the snapshot grid.
    show_trial_notes : bool, default False
        Show the per-trial note/tag editor in the review panel. Notes and tags
        are preserved regardless of whether the editor is currently visible.
    snapshot_profile_line : sequence, optional
        Initial profile endpoints as ``((row0, col0), (row1, col1))`` in
        snapshot image coordinates.
    snapshot_thumbnail_size : int, default 48
        Size of plot-embedded snapshot thumbnails in pixels.
    snapshot_panel_width_px : int, default 0
        Initial snapshot reconstruction panel width in pixels. Use ``0`` for
        automatic fit-to-view sizing; dragging the snapshot grid corner updates
        this value in the live widget.
    snapshot_columns : int, default 0
        Number of columns used for the side-panel snapshot image grid. Use
        ``0`` for automatic overview columns, or 1 through 8 for a fixed count.
    snapshot_overlay_position : {"top-left", "top-right", "bottom-left",
            "bottom-right"}, default "top-right"
        Corner used for snapshot panel label and zoom overlays.
    snapshot_real_space_zoom : float, default 1.0
        Initial zoom for real-space snapshot image panels.
    snapshot_real_space_center : sequence of 2 floats, optional
        Initial real-space snapshot center as ``(row, col)`` in image pixels.
    snapshot_fft_zoom : float, default 1.0
        Initial zoom for snapshot FFT panels.
    snapshot_fft_center : sequence of 2 floats, optional
        Initial FFT snapshot center as ``(row, col)`` in FFT pixel coordinates.
    sampling : float or sequence of float, optional
        Snapshot/profile image sampling used for the scale bar. Scalar values
        apply to both image axes; sequences use the last value as the displayed
        column-axis pixel size, matching ``Show2D``/``Show3D``.
    units : str or sequence of str, optional
        Physical unit for ``sampling``. Sequences use the last unit. When no
        sampling is provided the frontend falls back to a pixel scale bar.
    show_scale_bar, scale_bar_visible : bool, default True
        Draw the Show3D-style scale bar and zoom readout on snapshot panels.
    show_snapshot_histogram, prefer_webgpu : bool, default True
        Show the selected snapshot histogram and prefer WebGPU for snapshot
        histogram/FFT computation when the browser supports it.
    show_snapshot_fft : bool, default False
        Show a log-magnitude FFT panel below each snapshot image.
    snapshot_fft_window : bool, default True
        Apply a Hann window before snapshot FFT computation.
    snapshot_fft_cmap : str, default "magma"
        Colormap used for snapshot FFT panels.
    state : dict or path, optional
        Restore display state saved with :meth:`save`.

    Notes
    -----
    Live use is intentionally split between high-rate scalars and lower-rate
    images: call :meth:`append` every iteration and :meth:`snapshot` every
    ``N`` iterations so notebook comms stay responsive.
    """

    _esm = pathlib.Path(__file__).parent / "static" / "show1d.js"

    widget_version = traitlets.Unicode("unknown").tag(sync=True)
    y_bytes = traitlets.Bytes(b"").tag(sync=True)
    x_bytes = traitlets.Bytes(b"").tag(sync=True)
    n_traces = traitlets.Int(0).tag(sync=True)
    n_points = traitlets.Int(0).tag(sync=True)
    labels = traitlets.List(traitlets.Unicode()).tag(sync=True)
    colors = traitlets.List(traitlets.Unicode()).tag(sync=True)
    method_labels = traitlets.List(traitlets.Unicode()).tag(sync=True)

    title = traitlets.Unicode("").tag(sync=True)
    x_label = traitlets.Unicode("").tag(sync=True)
    y_label = traitlets.Unicode("").tag(sync=True)
    x_unit = traitlets.Unicode("").tag(sync=True)
    y_unit = traitlets.Unicode("").tag(sync=True)
    log_scale = traitlets.Bool(False).tag(sync=True)
    show_title = traitlets.Bool(True).tag(sync=True)
    show_stats = traitlets.Bool(False).tag(sync=True)
    show_review = traitlets.Bool(False).tag(sync=True)
    show_legend = traitlets.Bool(True).tag(sync=True)
    show_grid = traitlets.Bool(True).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)
    controls_collapsed = traitlets.Bool(False).tag(sync=True)
    line_width = traitlets.Float(1.5).tag(sync=True)
    plot_height_px = traitlets.Int(390).tag(sync=True)
    side_panel_width_px = traitlets.Int(360).tag(sync=True)
    focused_trace = traitlets.Int(-1).tag(sync=True)
    x_range = traitlets.List(traitlets.Float()).tag(sync=True)
    y_range = traitlets.List(traitlets.Float()).tag(sync=True)
    markers = traitlets.List(traitlets.Dict()).tag(sync=True)

    stats_mean = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_min = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_max = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_std = traitlets.List(traitlets.Float()).tag(sync=True)

    snapshot_bytes = traitlets.Bytes(b"").tag(sync=True)
    n_snapshots = traitlets.Int(0).tag(sync=True)
    snapshot_height = traitlets.Int(0).tag(sync=True)
    snapshot_width = traitlets.Int(0).tag(sync=True)
    snapshot_iterations = traitlets.List(traitlets.Float()).tag(sync=True)
    snapshot_labels = traitlets.List(traitlets.Unicode()).tag(sync=True)
    snapshot_heights = traitlets.List(traitlets.Int()).tag(sync=True)
    snapshot_widths = traitlets.List(traitlets.Int()).tag(sync=True)
    snapshot_image_labels = traitlets.List(traitlets.Unicode()).tag(sync=True)
    starred_snapshot_image_labels = traitlets.List(traitlets.Unicode()).tag(sync=True)
    hidden_snapshot_image_labels = traitlets.List(traitlets.Unicode()).tag(sync=True)
    trial_notes = traitlets.Dict().tag(sync=True)
    trial_tags = traitlets.Dict().tag(sync=True)
    show_trial_notes = traitlets.Bool(False).tag(sync=True)
    show_starred_only = traitlets.Bool(False).tag(sync=True)
    trial_sort_key = traitlets.Unicode("final_loss").tag(sync=True)
    trial_sort_descending = traitlets.Bool(False).tag(sync=True)
    trial_filter_text = traitlets.Unicode("").tag(sync=True)
    top_trial_count = traitlets.Int(0).tag(sync=True)
    trial_rankings = traitlets.List(traitlets.Dict()).tag(sync=True)
    trial_alerts = traitlets.List(traitlets.Dict()).tag(sync=True)
    best_trial_label = traitlets.Unicode("").tag(sync=True)
    run_summary = traitlets.Dict().tag(sync=True)
    snapshot_group_indices = traitlets.List(traitlets.Int()).tag(sync=True)
    snapshot_group_iterations = traitlets.List(traitlets.Float()).tag(sync=True)
    snapshot_group_labels = traitlets.List(traitlets.Unicode()).tag(sync=True)
    n_snapshot_groups = traitlets.Int(0).tag(sync=True)
    selected_snapshot_idx = traitlets.Int(-1).tag(sync=True)
    selected_snapshot_group_idx = traitlets.Int(-1).tag(sync=True)
    show_snapshots = traitlets.Bool(True).tag(sync=True)
    show_snapshot_thumbnails = traitlets.Bool(True).tag(sync=True)
    show_snapshot_histogram = traitlets.Bool(True).tag(sync=True)
    show_snapshot_fft = traitlets.Bool(False).tag(sync=True)
    snapshot_fft_window = traitlets.Bool(True).tag(sync=True)
    snapshot_fft_cmap = traitlets.Unicode("magma").tag(sync=True)
    show_snapshot_profile = traitlets.Bool(False).tag(sync=True)
    snapshot_profile_line = traitlets.List(traitlets.Dict(), default_value=[]).tag(sync=True)
    snapshot_profile_height = traitlets.Int(76).tag(sync=True)
    snapshot_contrast_preset = traitlets.Unicode("full").tag(sync=True)
    snapshot_contrast_range = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    snapshot_thumbnail_size = traitlets.Int(48).tag(sync=True)
    snapshot_panel_width_px = traitlets.Int(0).tag(sync=True)
    snapshot_columns = traitlets.Int(0).tag(sync=True)
    snapshot_overlay_position = traitlets.Unicode("top-right").tag(sync=True)
    snapshot_real_space_zoom = traitlets.Float(1.0).tag(sync=True)
    snapshot_real_space_center = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    snapshot_fft_zoom = traitlets.Float(1.0).tag(sync=True)
    snapshot_fft_center = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    image_cmap = traitlets.Unicode("cividis").tag(sync=True)
    pixel_size = traitlets.Float(0.0).tag(sync=True)
    pixel_unit = traitlets.Unicode("px").tag(sync=True)
    scale_bar_visible = traitlets.Bool(True).tag(sync=True)
    prefer_webgpu = traitlets.Bool(True).tag(sync=True)
    snapshot_playing = traitlets.Bool(False).tag(sync=True)
    snapshot_fps = traitlets.Int(2).tag(sync=True)

    profile_image_bytes = traitlets.Bytes(b"").tag(sync=True)
    profile_image_height = traitlets.Int(0).tag(sync=True)
    profile_image_width = traitlets.Int(0).tag(sync=True)
    profile_line = traitlets.List(traitlets.Dict()).tag(sync=True)
    profile_width = traitlets.Int(1).tag(sync=True)

    report_metadata = traitlets.Dict().tag(sync=True)
    monitor_path = traitlets.Unicode("").tag(sync=True)
    monitor_refresh_s = traitlets.Float(0.0).tag(sync=True)
    handoff_request = traitlets.Unicode("").tag(sync=True)
    handoff_status = traitlets.Unicode("").tag(sync=True)
    handoff_enabled = traitlets.Bool(True).tag(sync=True)
    prepared_view_widget = traitlets.Instance(ipywidgets.Widget, allow_none=True).tag(
        sync=True,
        **ipywidgets.widget_serialization,
    )

    _export_light = traitlets.Bool(False).tag(sync=True)
    export_request = traitlets.Unicode("").tag(sync=True)
    export_status = traitlets.Unicode("").tag(sync=True)
    export_enabled = traitlets.Bool(True).tag(sync=True)
    export_payload = traitlets.Bytes(b"").tag(sync=True)
    export_payload_id = traitlets.Unicode("").tag(sync=True)
    export_filename = traitlets.Unicode("").tag(sync=True)

    def __init__(
        self,
        data: Any = None,
        *,
        x: Any = None,
        labels: Sequence[str] | None = None,
        colors: Sequence[str] | None = None,
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        x_unit: str = "",
        y_unit: str = "",
        log_scale: bool = False,
        ui_mode: UiMode = "interactive",
        show_title: bool | None = None,
        show_stats: bool | None = None,
        show_review: bool | None = None,
        show_legend: bool | None = None,
        show_grid: bool | None = None,
        show_controls: bool | None = None,
        controls_collapsed: bool | None = None,
        line_width: float = 1.5,
        plot_height_px: int = 390,
        side_panel_width_px: int = 360,
        profile_image: Any = None,
        profile_line: Sequence[Sequence[float]] | None = None,
        profile_width: int = 1,
        image_cmap: str = "cividis",
        snapshot_contrast_preset: str = "full",
        snapshot_contrast_range: Sequence[float] | None = None,
        snapshot_thumbnail_size: int = 48,
        snapshot_panel_width_px: int = 0,
        snapshot_columns: int = 0,
        snapshot_overlay_position: str = "top-right",
        snapshot_real_space_zoom: float = 1.0,
        snapshot_real_space_center: Sequence[float] | None = None,
        snapshot_fft_zoom: float = 1.0,
        snapshot_fft_center: Sequence[float] | None = None,
        sampling: float | Sequence[float] | None = None,
        units: str | Sequence[str] | None = None,
        pixel_size: float | None = None,
        pixel_unit: str | None = None,
        show_scale_bar: bool | None = None,
        scale_bar_visible: bool | None = None,
        show_snapshot_histogram: bool = True,
        show_snapshot_fft: bool = False,
        snapshot_fft_window: bool = True,
        snapshot_fft_cmap: str = "magma",
        show_snapshot_profile: bool = False,
        snapshot_profile_line: Sequence[Sequence[float]] | None = None,
        snapshot_profile_height: int = 76,
        starred_snapshot_image_labels: Sequence[str] | None = None,
        hidden_snapshot_image_labels: Sequence[str] | None = None,
        trial_notes: Mapping[str, str] | None = None,
        trial_tags: Mapping[str, Sequence[str]] | None = None,
        show_trial_notes: bool = False,
        show_starred_only: bool = False,
        trial_sort_key: str = "final_loss",
        trial_sort_descending: bool = False,
        trial_filter_text: str = "",
        top_trial_count: int = 0,
        prefer_webgpu: bool = True,
        monitor_path: str | pathlib.Path | None = None,
        monitor_refresh_s: float = 0.0,
        state: dict[str, Any] | str | pathlib.Path | None = None,
        **kwargs: Any,
    ) -> None:
        if (
            scale_bar_visible is not None
            and show_scale_bar is not None
            and bool(scale_bar_visible) != bool(show_scale_bar)
        ):
            raise ValueError("Use either show_scale_bar or scale_bar_visible, not conflicting values")
        super().__init__(**kwargs)
        self.widget_version = resolve_widget_version()
        self._data, inferred_labels = self._normalise_data(data)
        inferred_title = self.title
        self._x = self._normalise_x(x, self._data.shape[1])
        self._snapshots: list[np.ndarray] = []
        self._profile_image: np.ndarray | None = None

        self.n_traces = int(self._data.shape[0])
        self.n_points = int(self._data.shape[1]) if self._data.ndim == 2 else 0
        self.labels = [str(v) for v in (labels or inferred_labels)]
        self.colors = [str(c) for c in (colors or self._default_colors(self.n_traces))]
        self.method_labels = []
        self.title = title or inferred_title
        self.x_label = x_label
        self.y_label = y_label
        self.x_unit = x_unit
        self.y_unit = y_unit
        self.log_scale = bool(log_scale)
        ui = resolve_ui_mode(
            ui_mode,
            defaults={
                "show_title": True,
                "show_stats": False,
                "show_review": False,
                "show_legend": True,
                "show_grid": True,
                "show_controls": True,
                "controls_collapsed": False,
            },
            overrides={
                "show_title": show_title,
                "show_stats": show_stats,
                "show_review": show_review,
                "show_legend": show_legend,
                "show_grid": show_grid,
                "show_controls": show_controls,
                "controls_collapsed": controls_collapsed,
            },
        )
        self.show_title = bool(ui["show_title"])
        self.show_stats = bool(ui["show_stats"])
        self.show_review = bool(ui["show_review"])
        self.show_legend = bool(ui["show_legend"])
        self.show_grid = bool(ui["show_grid"])
        self.show_controls = bool(ui["show_controls"])
        self.controls_collapsed = bool(ui["controls_collapsed"])
        self.line_width = float(line_width)
        self.plot_height_px = max(220, min(720, int(plot_height_px)))
        self.side_panel_width_px = max(300, min(960, int(side_panel_width_px)))
        self.profile_width = max(1, int(profile_width))
        self.image_cmap = self._normalise_image_cmap(image_cmap)
        self.snapshot_contrast_preset = self._normalise_snapshot_contrast_preset(snapshot_contrast_preset)
        self.snapshot_contrast_range = self._normalise_snapshot_contrast_range(snapshot_contrast_range)
        self.snapshot_thumbnail_size = max(24, min(112, int(snapshot_thumbnail_size)))
        self.snapshot_panel_width_px = max(0, min(960, int(snapshot_panel_width_px)))
        self.snapshot_columns = max(0, min(8, int(snapshot_columns)))
        self.snapshot_overlay_position = self._normalise_snapshot_overlay_position(
            snapshot_overlay_position
        )
        self.snapshot_real_space_zoom = self._normalise_snapshot_view_zoom(
            snapshot_real_space_zoom
        )
        self.snapshot_real_space_center = self._normalise_snapshot_view_center(
            snapshot_real_space_center
        )
        self.snapshot_fft_zoom = self._normalise_snapshot_view_zoom(snapshot_fft_zoom)
        self.snapshot_fft_center = self._normalise_snapshot_view_center(
            snapshot_fft_center
        )
        self.pixel_size = self._resolve_pixel_size(pixel_size, sampling)
        self.pixel_unit = self._resolve_pixel_unit(pixel_unit, units)
        self.scale_bar_visible = bool(
            scale_bar_visible if scale_bar_visible is not None
            else show_scale_bar if show_scale_bar is not None
            else True
        )
        self.show_snapshot_histogram = bool(show_snapshot_histogram)
        self.show_snapshot_fft = bool(show_snapshot_fft)
        self.snapshot_fft_window = bool(snapshot_fft_window)
        self.snapshot_fft_cmap = self._normalise_image_cmap(snapshot_fft_cmap)
        self.show_snapshot_profile = bool(show_snapshot_profile)
        self.snapshot_profile_line = self._normalise_profile_line(snapshot_profile_line)
        self.snapshot_profile_height = max(44, min(220, int(snapshot_profile_height)))
        self.starred_snapshot_image_labels = self._normalise_trial_labels(starred_snapshot_image_labels or [])
        self.hidden_snapshot_image_labels = self._normalise_trial_labels(hidden_snapshot_image_labels or [])
        self.trial_notes = self._normalise_trial_notes(trial_notes or {})
        self.trial_tags = self._normalise_trial_tags(trial_tags or {})
        self.show_trial_notes = bool(show_trial_notes)
        self.show_starred_only = bool(show_starred_only)
        self.trial_sort_key = self._normalise_trial_sort_key(trial_sort_key)
        self.trial_sort_descending = bool(trial_sort_descending)
        self.trial_filter_text = str(trial_filter_text or "")
        self.top_trial_count = max(0, int(top_trial_count))
        self.prefer_webgpu = bool(prefer_webgpu)
        self.monitor_path = str(monitor_path) if monitor_path is not None else ""
        self.monitor_refresh_s = max(0.0, float(monitor_refresh_s))
        self._monitor_thread: threading.Thread | None = None
        self._monitor_stop: threading.Event | None = None
        self._monitor_mtime: float = 0.0
        self._monitor_offset: int = 0
        self._monitor_line_count: int = 0
        self.prepared_view = None
        self.prepared_view_widget = None

        self._update_stats()
        self._update_data_bytes()
        self._update_trial_analysis()
        if profile_image is not None:
            self.set_profile_image(profile_image, line=profile_line, profile_width=profile_width)

        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(
                    json.loads(pathlib.Path(state).read_text()),
                    require_envelope=True,
                    expected_widget="Show1D",
                )
            else:
                state = unwrap_state_payload(state, expected_widget="Show1D")
            self.load_state_dict(state)
        else:
            self._update_trial_analysis()

        self.observe(self._on_export_request_change, names=["export_request"])
        self.observe(self._on_handoff_request_change, names=["handoff_request"])

    @traitlets.validate("image_cmap")
    def _validate_image_cmap(self, proposal: dict[str, Any]) -> str:
        return self._normalise_image_cmap(str(proposal["value"]))

    @traitlets.validate("snapshot_contrast_preset")
    def _validate_snapshot_contrast_preset(self, proposal: dict[str, Any]) -> str:
        return self._normalise_snapshot_contrast_preset(str(proposal["value"]))

    @traitlets.validate("snapshot_contrast_range")
    def _validate_snapshot_contrast_range(self, proposal: dict[str, Any]) -> list[float]:
        return self._normalise_snapshot_contrast_range(proposal["value"])

    @traitlets.validate("plot_height_px")
    def _validate_plot_height_px(self, proposal: dict[str, Any]) -> int:
        return max(220, min(720, int(proposal["value"])))

    @traitlets.validate("side_panel_width_px")
    def _validate_side_panel_width_px(self, proposal: dict[str, Any]) -> int:
        return max(300, min(960, int(proposal["value"])))

    @traitlets.validate("snapshot_thumbnail_size")
    def _validate_snapshot_thumbnail_size(self, proposal: dict[str, Any]) -> int:
        return max(24, min(112, int(proposal["value"])))

    @traitlets.validate("snapshot_panel_width_px")
    def _validate_snapshot_panel_width_px(self, proposal: dict[str, Any]) -> int:
        return max(0, min(960, int(proposal["value"])))

    @traitlets.validate("snapshot_fps")
    def _validate_snapshot_fps(self, proposal: dict[str, Any]) -> int:
        return max(1, min(24, int(round(float(proposal["value"])))))

    @traitlets.validate("snapshot_columns")
    def _validate_snapshot_columns(self, proposal: dict[str, Any]) -> int:
        return max(0, min(8, int(proposal["value"])))

    @traitlets.validate("snapshot_overlay_position")
    def _validate_snapshot_overlay_position(self, proposal: dict[str, Any]) -> str:
        return self._normalise_snapshot_overlay_position(str(proposal["value"]))

    @traitlets.validate("snapshot_real_space_zoom", "snapshot_fft_zoom")
    def _validate_snapshot_view_zoom(self, proposal: dict[str, Any]) -> float:
        return self._normalise_snapshot_view_zoom(proposal["value"])

    @traitlets.validate("snapshot_real_space_center", "snapshot_fft_center")
    def _validate_snapshot_view_center(self, proposal: dict[str, Any]) -> list[float]:
        return self._normalise_snapshot_view_center(proposal["value"])

    @traitlets.validate("pixel_size")
    def _validate_pixel_size(self, proposal: dict[str, Any]) -> float:
        value = float(proposal["value"])
        if not math.isfinite(value):
            raise traitlets.TraitError(f"pixel_size must be finite, got {value}")
        if value < 0:
            raise traitlets.TraitError(f"pixel_size must be >= 0, got {value}")
        return value

    @traitlets.validate("snapshot_fft_cmap")
    def _validate_snapshot_fft_cmap(self, proposal: dict[str, Any]) -> str:
        return self._normalise_image_cmap(str(proposal["value"]))

    @traitlets.validate("snapshot_profile_line")
    def _validate_snapshot_profile_line(self, proposal: dict[str, Any]) -> list[dict[str, float]]:
        return self._normalise_profile_line(proposal["value"])

    @traitlets.validate("snapshot_profile_height")
    def _validate_snapshot_profile_height(self, proposal: dict[str, Any]) -> int:
        return max(44, min(220, int(proposal["value"])))

    @traitlets.validate("trial_sort_key")
    def _validate_trial_sort_key(self, proposal: dict[str, Any]) -> str:
        return self._normalise_trial_sort_key(str(proposal["value"]))

    @traitlets.validate("top_trial_count")
    def _validate_top_trial_count(self, proposal: dict[str, Any]) -> int:
        return max(0, int(proposal["value"]))

    @traitlets.validate("monitor_refresh_s")
    def _validate_monitor_refresh_s(self, proposal: dict[str, Any]) -> float:
        value = float(proposal["value"])
        if not math.isfinite(value) or value < 0:
            raise traitlets.TraitError(f"monitor_refresh_s must be >= 0, got {value}")
        return value

    @classmethod
    def live(
        cls,
        traces: Sequence[str] | None = None,
        *,
        title: str = "Live Reconstruction",
        x_label: str = "iteration",
        y_label: str = "",
        log_scale: bool = True,
        **kwargs: Any,
    ) -> Self:
        """Create an empty monitor intended for repeated :meth:`append` calls."""

        labels = [str(v) for v in (traces or [])]
        data = np.empty((len(labels), 0), dtype=np.float32)
        return cls(
            data,
            labels=labels,
            title=title,
            x_label=x_label,
            y_label=y_label,
            log_scale=log_scale,
            **kwargs,
        )

    @classmethod
    def from_loss_runs(
        cls,
        runs: Mapping[str, Any],
        *,
        x: Any = None,
        losses: Sequence[str] | None = None,
        label_template: str = "{run} · {loss}",
        title: str = "Loss Comparison",
        x_label: str = "iteration",
        y_label: str = "loss",
        **kwargs: Any,
    ) -> Self:
        """Create a multi-trace viewer from run/loss mappings.

        ``runs`` may be either ``run_label -> loss_values`` or
        ``run_label -> {loss_name: loss_values}``.  The nested form is useful
        for comparing Adam/data/regularizer histories across ptychography
        lambda sweeps while keeping stable trace labels.
        """

        if not isinstance(runs, Mapping):
            raise TypeError("runs must be a mapping of run labels to loss traces")
        requested_losses = [str(name) for name in losses] if losses is not None else None
        traces: dict[str, np.ndarray] = {}

        for run_name, run_values in runs.items():
            run_label = str(run_name)
            if isinstance(run_values, Mapping):
                loss_map = {str(name): values for name, values in run_values.items()}
                names = requested_losses if requested_losses is not None else list(loss_map)
                if not names:
                    raise ValueError(f"run {run_label!r} does not contain any loss traces")
                for loss_name in names:
                    if loss_name not in loss_map:
                        raise ValueError(f"run {run_label!r} has no loss trace {loss_name!r}")
                    trace_label = str(label_template).format(run=run_label, loss=loss_name)
                    traces[trace_label] = np.asarray(to_numpy(loss_map[loss_name]), dtype=np.float32).ravel()
                continue

            if requested_losses is not None and len(requested_losses) != 1:
                raise ValueError("bare loss traces require losses to contain exactly one name")
            loss_name = requested_losses[0] if requested_losses else "loss"
            trace_label = str(label_template).format(run=run_label, loss=loss_name)
            traces[trace_label] = np.asarray(to_numpy(run_values), dtype=np.float32).ravel()

        if not traces:
            raise ValueError("from_loss_runs requires at least one loss trace")
        return cls(
            traces,
            x=x,
            title=title,
            x_label=x_label,
            y_label=y_label,
            **kwargs,
        )

    @classmethod
    def from_image(
        cls,
        image: Any,
        *,
        line: Sequence[Sequence[float]],
        profile_width: int = 1,
        sampling: float = 1.0,
        x_unit: str = "pixels",
        title: str = "Line Profile",
        y_label: str = "value",
        **kwargs: Any,
    ) -> Self:
        """Build a line-profile viewer from a 2D image and a ``(row, col)`` line."""

        profile = sample_line_profile(image, line, profile_width=profile_width)
        distance = np.arange(profile.size, dtype=np.float32) * float(sampling)
        return cls(
            profile,
            x=distance,
            labels=["profile"],
            title=title,
            x_label="distance",
            x_unit=x_unit,
            y_label=y_label,
            profile_image=image,
            profile_line=line,
            profile_width=profile_width,
            **kwargs,
        )

    @classmethod
    def from_joint_time_report(
        cls,
        summary_path: str | pathlib.Path,
        *,
        arrays_path: str | pathlib.Path | None = None,
        metric_keys: Sequence[str] | None = None,
        frame_by_frame: bool = False,
        loss_key: str = "final_losses",
        include_reference: bool = True,
        snapshot_downsample: int = 1,
        max_snapshot_frames: int | None = None,
        title: str = "Joint-Time Ptychography Metrics",
        **kwargs: Any,
    ) -> Self:
        """Create a metric/snapshot viewer from a ducky joint-time report."""

        summary_file = pathlib.Path(summary_path)
        summary = json.loads(summary_file.read_text())
        metrics = summary.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError("summary JSON must contain a non-empty 'metrics' dict")
        methods = sorted(metrics, key=_method_sort_key)
        if frame_by_frame:
            traces_by_label: dict[str, np.ndarray] = {}
            n_frames = 0
            for method in methods:
                values = metrics[method].get(loss_key)
                if values is None:
                    continue
                arr = np.asarray(to_numpy(values), dtype=np.float32).ravel()
                if arr.size == 0:
                    continue
                n_frames = max(n_frames, arr.size)
                traces_by_label[_pretty_method_label(method)] = arr
            if not traces_by_label:
                raise ValueError(f"none of the methods contain loss trace {loss_key!r}")
            if len({arr.size for arr in traces_by_label.values()}) != 1:
                raise ValueError(f"all {loss_key!r} traces must have the same frame count")
            widget = cls(
                traces_by_label,
                x=np.arange(n_frames, dtype=np.float32),
                title=title,
                x_label="frame",
                y_label=loss_key.replace("_", " "),
                **kwargs,
            )
            widget.method_labels = [str(idx) for idx in range(n_frames)]
        else:
            keys = list(metric_keys or (
                "rmse_per_frame_mask",
                "rmse_time_average_mask",
                "temporal_flicker_mask",
                "mean_phase_std_mask",
                "elapsed_s",
            ))
            traces = []
            trace_labels = []
            for key in keys:
                vals = [_as_float(metrics[method].get(key)) for method in methods]
                if any(np.isfinite(vals)):
                    traces.append(vals)
                    trace_labels.append(_pretty_metric_label(key))
            if not traces:
                raise ValueError("none of the requested metric keys were found")

            widget = cls(
                np.asarray(traces, dtype=np.float32),
                x=np.arange(len(methods), dtype=np.float32),
                labels=trace_labels,
                title=title,
                x_label="method index",
                y_label="metric",
                **kwargs,
            )
            widget.method_labels = [str(m) for m in methods]
        widget.report_metadata = {
            "summary_path": str(summary_file),
            "data": str(summary.get("data", "")),
            "num_frames": int(summary.get("num_frames", 0) or 0),
            "num_iters": int(summary.get("num_iters", 0) or 0),
            "electrons_per_pattern": _as_float(summary.get("electrons_per_pattern")),
            "joint_init": str(summary.get("joint_init", "")),
            "loss_type": str(summary.get("loss_type", "")),
            "metrics_by_trial": {_pretty_method_label(method): dict(metrics[method]) for method in methods},
            "methods": list(methods),
            "frame_by_frame": bool(frame_by_frame),
            "loss_key": str(loss_key),
        }

        if arrays_path is None:
            candidate = summary_file.parent / "reconstructions.npz"
            arrays_path = candidate if candidate.exists() else None
        if arrays_path is not None:
            widget._load_joint_time_snapshots(
                pathlib.Path(arrays_path),
                methods,
                frame_by_frame=frame_by_frame,
                include_reference=include_reference,
                downsample=snapshot_downsample,
                max_frames=max_snapshot_frames,
            )
        widget._update_trial_analysis()
        return widget

    @classmethod
    def from_monitor_file(
        cls,
        path: str | pathlib.Path,
        *,
        title: str = "Overnight Reconstruction Monitor",
        x_label: str = "iteration",
        y_label: str = "loss",
        log_scale: bool = True,
        **kwargs: Any,
    ) -> Self:
        """Create a viewer from a file-backed JSONL reconstruction monitor.

        Each line should be a JSON object with an ``iteration`` number and any
        of ``losses``, ``snapshots``, ``metrics``, ``warnings``, ``starred``,
        ``hidden``, ``notes``, or ``tags``. Snapshot values are paths to ``.npy``
        or ``.npz`` arrays, resolved relative to the monitor file.
        """

        monitor_file = cls._resolve_monitor_file(path)
        events = cls._read_monitor_events(monitor_file)
        if not events:
            widget = cls.live(title=title, x_label=x_label, y_label=y_label, log_scale=log_scale, **kwargs)
            widget.monitor_path = str(monitor_file)
            widget.report_metadata = {
                "monitor_path": str(monitor_file),
                "monitor_events": 0,
                "monitor_warnings": [],
                "metrics_by_trial": {},
            }
            widget._monitor_offset = monitor_file.stat().st_size if monitor_file.exists() else 0
            widget._monitor_line_count = len(monitor_file.read_text(encoding="utf-8").splitlines()) if monitor_file.exists() else 0
            widget.report_metadata = {
                **dict(widget.report_metadata),
                "monitor_offset": widget._monitor_offset,
                "monitor_lines": widget._monitor_line_count,
            }
            widget._update_trial_analysis()
            return widget

        loss_names: list[str] = []
        for event in events:
            losses = event.get("losses")
            if isinstance(losses, Mapping):
                for name in losses:
                    if str(name) not in loss_names:
                        loss_names.append(str(name))
        widget = cls.live(loss_names, title=title, x_label=x_label, y_label=y_label, log_scale=log_scale, **kwargs)
        widget.monitor_path = str(monitor_file)
        widget.report_metadata = {
            "monitor_path": str(monitor_file),
            "monitor_events": 0,
            "monitor_warnings": [],
            "metrics_by_trial": {},
        }

        widget.apply_monitor_events(events, base_path=monitor_file)
        widget._monitor_offset = monitor_file.stat().st_size if monitor_file.exists() else 0
        widget._monitor_line_count = len(monitor_file.read_text(encoding="utf-8").splitlines()) if monitor_file.exists() else 0
        widget.report_metadata = {
            **dict(widget.report_metadata),
            "monitor_events": len(events),
            "monitor_offset": widget._monitor_offset,
            "monitor_lines": widget._monitor_line_count,
        }
        widget._update_trial_analysis()
        return widget

    @classmethod
    def watch_run(
        cls,
        path: str | pathlib.Path,
        *,
        refresh_s: float = 5.0,
        start: bool = True,
        **kwargs: Any,
    ) -> Self:
        """Load a monitor file and optionally poll it while the kernel is alive."""

        widget = cls.from_monitor_file(path, **kwargs)
        widget.monitor_refresh_s = max(0.0, float(refresh_s))
        if start and widget.monitor_refresh_s > 0:
            widget.start_monitor()
        return widget

    @staticmethod
    def append_monitor_event(path: str | pathlib.Path, event: Mapping[str, Any]) -> pathlib.Path:
        """Append one JSON event to a monitor JSONL file."""

        monitor_file = Show1D._resolve_monitor_file(path, create=True)
        monitor_file.parent.mkdir(parents=True, exist_ok=True)
        with monitor_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(dict(event)), allow_nan=False, sort_keys=True) + "\n")
        return monitor_file

    def append(self, x: float | None = None, **values: Any) -> Self:
        """Append one live sample to named traces.

        New trace names are added automatically and back-filled with NaN for
        earlier samples. Missing existing traces receive NaN at this x value.
        """

        if not values:
            raise ValueError("append requires at least one named value")
        if x is None:
            x = float(self.n_points if self._x is None or self._x.size == 0 else self._x[-1] + 1)
        if self._data.size == 0 and self.n_traces == 0:
            self.labels = [str(k) for k in values]
            self.colors = self._default_colors(len(self.labels))
            self._data = np.empty((len(self.labels), 0), dtype=np.float32)
            self.n_traces = len(self.labels)

        labels = list(self.labels)
        for name in values:
            if name not in labels:
                labels.append(str(name))
                filler = np.full((1, self.n_points), np.nan, dtype=np.float32)
                self._data = np.vstack([self._data, filler]) if self._data.size else filler
                self.colors = list(self.colors) + [_DEFAULT_COLORS[(len(labels) - 1) % len(_DEFAULT_COLORS)]]
        column = np.asarray([_as_float(values.get(label)) for label in labels], dtype=np.float32)
        self._data = np.column_stack([self._data, column]) if self._data.size else column[:, None]
        self._x = np.asarray([x], dtype=np.float32) if self._x is None else np.append(self._x, np.float32(x))
        self.labels = labels
        self.n_traces = int(self._data.shape[0])
        self.n_points = int(self._data.shape[1])
        self._update_stats()
        self._update_data_bytes()
        self._update_trial_analysis()
        return self

    def append_scalar(self, iteration: float | None = None, **values: Any) -> Self:
        """Alias for :meth:`append` with reconstruction-friendly naming."""

        return self.append(x=iteration, **values)

    def extend(self, x: Sequence[float] | np.ndarray | None = None, **values: Any) -> Self:
        """Append a batch of live scalar samples in one widget update.

        Parameters
        ----------
        x : sequence of float, optional
            Iteration or frame coordinates for the appended samples. If omitted,
            coordinates continue from the current final x value.
        **values : array_like
            Mapping of trace name to a 1D sequence of values. New trace names are
            added automatically and existing traces omitted from ``values`` are
            back-filled with ``NaN`` for the appended span.

        Returns
        -------
        Show1D
            The mutated widget, ready for the frontend to redraw via trait sync.
        """

        if not values:
            raise ValueError("extend requires at least one named value sequence")

        arrays: dict[str, np.ndarray] = {}
        n_new: int | None = None
        for name, raw in values.items():
            arr = np.asarray(to_numpy(raw), dtype=np.float32).ravel()
            if n_new is None:
                n_new = int(arr.size)
            elif arr.size != n_new:
                raise ValueError(
                    f"all extend value sequences must have length {n_new}; "
                    f"{name!r} has length {arr.size}"
                )
            arrays[str(name)] = arr
        if not n_new:
            return self

        if x is None:
            start = float(self.n_points if self._x is None or self._x.size == 0 else self._x[-1] + 1)
            x_values = np.arange(start, start + n_new, dtype=np.float32)
        else:
            x_values = np.asarray(to_numpy(x), dtype=np.float32).ravel()
            if x_values.size != n_new:
                raise ValueError(f"x must have length {n_new}, got {x_values.size}")

        if self._data.size == 0 and self.n_traces == 0:
            self.labels = list(arrays)
            self.colors = self._default_colors(len(self.labels))
            self._data = np.empty((len(self.labels), 0), dtype=np.float32)
            self.n_traces = len(self.labels)

        labels = list(self.labels)
        for name in arrays:
            if name not in labels:
                labels.append(name)
                filler = np.full((1, self.n_points), np.nan, dtype=np.float32)
                self._data = np.vstack([self._data, filler]) if self._data.size else filler
                self.colors = list(self.colors) + [_DEFAULT_COLORS[(len(labels) - 1) % len(_DEFAULT_COLORS)]]

        block = np.full((len(labels), n_new), np.nan, dtype=np.float32)
        for row, label in enumerate(labels):
            if label in arrays:
                block[row, :] = arrays[label]

        self._data = np.column_stack([self._data, block]) if self._data.size else block
        self._x = x_values if self._x is None else np.concatenate([self._x.astype(np.float32, copy=False), x_values])
        self.labels = labels
        self.n_traces = int(self._data.shape[0])
        self.n_points = int(self._data.shape[1])
        self._update_stats()
        self._update_data_bytes()
        self._update_trial_analysis()
        return self

    append_many = extend

    def apply_monitor_events(
        self,
        events: Sequence[Mapping[str, Any]],
        *,
        base_path: str | pathlib.Path | None = None,
    ) -> Self:
        """Apply monitor JSONL events to this live widget incrementally.

        This is the programmatic counterpart to :meth:`refresh_monitor`: callers
        can feed parsed events directly, while ``watch_run`` tails newly written
        JSONL lines and calls the same method. Existing traces, snapshots,
        review state, notes, and tags remain in place.
        """

        if not events:
            return self

        base_dir = pathlib.Path(base_path).parent if base_path is not None else pathlib.Path.cwd()
        metadata = dict(self.report_metadata)
        warnings_list = list(metadata.get("monitor_warnings", []))
        metric_map = dict(metadata.get("metrics_by_trial", {}))
        processed = int(metadata.get("monitor_events", 0) or 0)

        for event in events:
            if not isinstance(event, Mapping):
                raise ValueError("monitor events must be mappings")
            iteration = _as_float(event.get("iteration", self.n_points))

            losses = event.get("losses")
            if isinstance(losses, Mapping):
                self.append(iteration, **{str(name): _as_float(value) for name, value in losses.items()})

            metrics = event.get("metrics")
            if isinstance(metrics, Mapping):
                for label, metric_values in metrics.items():
                    if isinstance(metric_values, Mapping):
                        existing = dict(metric_map.get(str(label), {}))
                        existing.update(dict(metric_values))
                        metric_map[str(label)] = existing

            snapshots = event.get("snapshots")
            if isinstance(snapshots, Mapping):
                images: dict[str, np.ndarray] = {}
                for label, image_path in snapshots.items():
                    path = pathlib.Path(str(image_path))
                    if not path.is_absolute():
                        path = base_dir / path
                    arr = self._load_monitor_image(path)
                    if arr is not None:
                        images[str(label)] = arr
                if images:
                    label = str(event.get("label") or f"iter {iteration:g}")
                    self.snapshot(iteration, label=label, **images)

            warnings = event.get("warnings")
            if isinstance(warnings, Sequence) and not isinstance(warnings, (str, bytes)):
                warnings_list.extend(str(item) for item in warnings)
            elif warnings:
                warnings_list.append(str(warnings))

            starred = event.get("starred", [])
            if isinstance(starred, (str, bytes)) or not isinstance(starred, Sequence):
                starred = [starred] if starred else []
            for label in starred:
                self.star_trial(str(label))

            hidden = event.get("hidden", [])
            if isinstance(hidden, (str, bytes)) or not isinstance(hidden, Sequence):
                hidden = [hidden] if hidden else []
            for label in hidden:
                self.hide_trial(str(label))

            notes = event.get("notes")
            if isinstance(notes, Mapping):
                for label, note in notes.items():
                    self.set_trial_note(str(label), str(note))

            tags = event.get("tags")
            if isinstance(tags, Mapping):
                for label, values_for_label in tags.items():
                    if isinstance(values_for_label, Sequence) and not isinstance(values_for_label, (str, bytes)):
                        for tag in values_for_label:
                            self.tag_trial(str(label), str(tag))

        metadata.update(
            {
                "monitor_path": self.monitor_path or str(base_path or ""),
                "monitor_events": processed + len(events),
                "monitor_warnings": warnings_list,
                "metrics_by_trial": metric_map,
            }
        )
        metadata.pop("monitor_error", None)
        self.report_metadata = _json_safe(metadata)
        self._update_trial_analysis()
        return self

    def snapshot(
        self,
        iteration: float,
        image: Any | None = None,
        *,
        label: str | None = None,
        **images: Any,
    ) -> Self:
        """Attach one or more 2D image snapshots to an iteration/x value.

        Multiple images passed in one call are one logical snapshot group. This
        lets live ptychography monitors show related panels such as ``object``
        and ``probe`` at the same optimizer iteration.
        """

        items: list[tuple[str, Any]] = []
        group_label = label or f"iter {iteration:g}"
        if image is not None:
            items.append((label or "image", image))
        items.extend((str(name), value) for name, value in images.items())
        if not items:
            raise ValueError("snapshot requires image=... or one or more named images")
        group_idx = len(self.snapshot_group_iterations)
        first_image_idx = len(self._snapshots)
        for name, value in items:
            arr = np.asarray(to_numpy(value), dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f"snapshot {name!r} must be 2D, got shape {arr.shape}")
            self._snapshots.append(np.ascontiguousarray(arr, dtype=np.float32))
            self.snapshot_iterations = list(self.snapshot_iterations) + [float(iteration)]
            self.snapshot_labels = list(self.snapshot_labels) + [name]
            self.snapshot_image_labels = list(self.snapshot_image_labels) + [name]
            self.snapshot_group_indices = list(self.snapshot_group_indices) + [group_idx]
        self.snapshot_group_iterations = list(self.snapshot_group_iterations) + [float(iteration)]
        self.snapshot_group_labels = list(self.snapshot_group_labels) + [group_label]
        self.n_snapshot_groups = len(self.snapshot_group_iterations)
        self._update_snapshot_bytes()
        self.selected_snapshot_idx = first_image_idx
        self.selected_snapshot_group_idx = group_idx
        self._update_trial_analysis()
        return self

    def set_profile_image(
        self,
        image: Any,
        *,
        line: Sequence[Sequence[float]] | None = None,
        profile_width: int | None = None,
    ) -> Self:
        """Attach a 2D image context and optionally resample the displayed trace."""

        arr = np.asarray(to_numpy(image), dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"profile image must be 2D, got shape {arr.shape}")
        self._profile_image = np.ascontiguousarray(arr, dtype=np.float32)
        self.profile_image_height = int(arr.shape[0])
        self.profile_image_width = int(arr.shape[1])
        self.profile_image_bytes = _b64_safe(self._profile_image.tobytes())
        if profile_width is not None:
            self.profile_width = max(1, int(profile_width))
        if line is not None:
            self.profile_line = [
                {"row": float(line[0][0]), "col": float(line[0][1])},
                {"row": float(line[1][0]), "col": float(line[1][1])},
            ]
            values = sample_line_profile(arr, line, profile_width=self.profile_width)
            self.set_data(values, x=np.arange(values.size, dtype=np.float32), labels=["profile"])
        return self

    def set_data(self, data: Any, *, x: Any = None, labels: Sequence[str] | None = None) -> Self:
        """Replace the trace data while preserving display settings."""

        self._data, inferred_labels = self._normalise_data(data)
        self._x = self._normalise_x(x, self._data.shape[1])
        self.n_traces = int(self._data.shape[0])
        self.n_points = int(self._data.shape[1])
        self.labels = [str(v) for v in (labels or inferred_labels)]
        self.colors = self._default_colors(self.n_traces)
        self._update_stats()
        self._update_data_bytes()
        self._update_trial_analysis()
        return self

    def add_marker(self, x: float, *, label: str = "", kind: str = "checkpoint") -> Self:
        """Add a vertical marker at x, useful for checkpoints or events."""

        self.markers = list(self.markers) + [{"x": float(x), "label": str(label), "kind": str(kind)}]
        return self

    def clear_markers(self) -> Self:
        """Remove all event markers."""

        self.markers = []
        return self

    def play(self) -> Self:
        """Start cycling through snapshot groups in the frontend."""

        self.snapshot_playing = True
        return self

    def pause(self) -> Self:
        """Pause snapshot group playback."""

        self.snapshot_playing = False
        return self

    def stop(self) -> Self:
        """Stop playback and return to the first snapshot group."""

        self.snapshot_playing = False
        if self.n_snapshot_groups > 0:
            self.goto_snapshot(0)
        elif self.n_snapshots > 0:
            self.selected_snapshot_idx = 0
        return self

    def goto_snapshot(self, index: int) -> Self:
        """Select a snapshot group by index.

        The name intentionally follows the old single-image snapshot API.  When
        a group contains multiple images, the first image in that group becomes
        the primary selected image while the frontend shows all group members.
        """

        if self.n_snapshot_groups <= 0:
            self.selected_snapshot_group_idx = -1
            self.selected_snapshot_idx = -1 if self.n_snapshots <= 0 else max(0, min(self.n_snapshots - 1, int(index)))
            return self

        group_idx = max(0, min(self.n_snapshot_groups - 1, int(index)))
        self.selected_snapshot_group_idx = group_idx
        for image_idx, image_group_idx in enumerate(self.snapshot_group_indices):
            if int(image_group_idx) == group_idx:
                self.selected_snapshot_idx = image_idx
                break
        else:
            self.selected_snapshot_idx = -1
        return self

    def to_show2d(
        self,
        group: int | str | None = None,
        images: Sequence[int | str] | int | str | None = None,
        *,
        title: str | None = None,
        copy: bool = True,
        include_hidden: bool = False,
        respect_review_filters: bool = True,
    ):
        """Create a ``Show2D`` gallery from a snapshot group.

        The default converts the currently selected snapshot group and applies
        the same hidden/starred/top/filter review state used by the frontend.
        Pass ``images=...`` to choose group-local image indices or labels.
        """

        from quantem.widget.show2d import Show2D

        image_indices = self._snapshot_image_indices_for_group(group)
        image_indices = self._normalise_snapshot_image_refs(images, image_indices)

        selected: list[int] = []
        for image_idx in image_indices:
            label = self._snapshot_image_label(image_idx)
            if not include_hidden and self._label_in_collection(label, self.hidden_snapshot_image_labels):
                continue
            if respect_review_filters and not self._snapshot_label_passes_review_filter(label):
                continue
            selected.append(image_idx)
        if not selected:
            raise ValueError("Show1D.to_show2d() needs at least one visible snapshot image")

        frames = [np.asarray(self._snapshots[idx], dtype=np.float32) for idx in selected]
        if copy:
            frames = [np.array(frame, copy=True) for frame in frames]
        panel_labels = [self._snapshot_image_label(idx) for idx in selected]
        starred = [
            idx for idx, label in enumerate(panel_labels)
            if self._label_in_collection(label, self.starred_snapshot_image_labels)
        ]
        group_idx = self._normalise_snapshot_group_ref(group)
        group_label = (
            self.snapshot_group_labels[group_idx]
            if 0 <= group_idx < len(self.snapshot_group_labels)
            else f"snapshot {group_idx + 1}"
        )
        view_title = title if title is not None else f"{self.title or 'Show1D'} · {group_label}"
        return Show2D(
            frames,
            labels=panel_labels,
            title=view_title,
            cmap=self.image_cmap,
            sampling=self.pixel_size if self.pixel_size > 0 else None,
            units=self.pixel_unit,
            scale_bar_visible=self.scale_bar_visible,
            show_fft=self.show_snapshot_fft,
            show_controls=True,
            controls_collapsed=False,
            show_stats=True,
            auto_contrast=True,
            ncols=max(1, min(int(self.snapshot_columns) if self.snapshot_columns else len(frames), len(frames))),
            link_zoom=len(frames) > 1,
            link_pan=len(frames) > 1,
            link_contrast=len(frames) > 1,
            gallery_gap_px=0,
            show_panel_titles=True,
            panel_title_font_size=11,
            starred=starred,
            save_state=False,
            verbose=False,
        )

    def star_trial(self, label: str) -> Self:
        """Mark a reconstruction/snapshot label as a candidate to revisit."""

        self.starred_snapshot_image_labels = self._add_trial_label(
            self.starred_snapshot_image_labels,
            label,
        )
        self._update_trial_analysis()
        return self

    def unstar_trial(self, label: str) -> Self:
        """Remove a reconstruction/snapshot label from the candidate list."""

        self.starred_snapshot_image_labels = self._remove_trial_label(
            self.starred_snapshot_image_labels,
            label,
        )
        self._update_trial_analysis()
        return self

    def clear_starred_trials(self) -> Self:
        """Remove all starred reconstruction candidates."""

        self.starred_snapshot_image_labels = []
        self._update_trial_analysis()
        return self

    def hide_trial(self, label: str) -> Self:
        """Hide a reconstruction/snapshot label from plots, stats, and panels."""

        self.hidden_snapshot_image_labels = self._add_trial_label(
            self.hidden_snapshot_image_labels,
            label,
        )
        self.starred_snapshot_image_labels = self._remove_trial_label(
            self.starred_snapshot_image_labels,
            label,
        )
        self._update_trial_analysis()
        return self

    def show_trial(self, label: str) -> Self:
        """Show a reconstruction/snapshot label that was previously hidden."""

        self.hidden_snapshot_image_labels = self._remove_trial_label(
            self.hidden_snapshot_image_labels,
            label,
        )
        self._update_trial_analysis()
        return self

    def show_all_trials(self) -> Self:
        """Restore all hidden reconstruction/snapshot labels."""

        self.hidden_snapshot_image_labels = []
        self._update_trial_analysis()
        return self

    def set_trial_note(self, label: str, note: str) -> Self:
        """Attach a short note to a reconstruction/snapshot label."""

        clean = str(label).strip()
        if not clean:
            raise ValueError("trial label must be a non-empty string")
        notes = dict(self.trial_notes)
        if str(note).strip():
            notes[clean] = str(note).strip()
        else:
            notes.pop(clean, None)
        self.trial_notes = self._normalise_trial_notes(notes)
        self._update_trial_analysis()
        return self

    def clear_trial_note(self, label: str) -> Self:
        """Remove a note from a reconstruction/snapshot label."""

        return self.set_trial_note(label, "")

    def tag_trial(self, label: str, tag: str) -> Self:
        """Add a tag such as ``best lambda`` or ``probe drift`` to a trial."""

        clean_label = str(label).strip()
        clean_tag = str(tag).strip()
        if not clean_label or not clean_tag:
            raise ValueError("trial label and tag must be non-empty strings")
        tags = self._normalise_trial_tags(self.trial_tags)
        values = list(tags.get(clean_label, []))
        if clean_tag not in values:
            values.append(clean_tag)
        tags[clean_label] = values
        self.trial_tags = tags
        self._update_trial_analysis()
        return self

    def untag_trial(self, label: str, tag: str) -> Self:
        """Remove a tag from a trial."""

        clean_label = str(label).strip()
        clean_tag = str(tag).strip()
        tags = self._normalise_trial_tags(self.trial_tags)
        values = [value for value in tags.get(clean_label, []) if value != clean_tag]
        if values:
            tags[clean_label] = values
        else:
            tags.pop(clean_label, None)
        self.trial_tags = tags
        self._update_trial_analysis()
        return self

    def clear_trial_tags(self, label: str) -> Self:
        """Remove all tags from a trial."""

        tags = self._normalise_trial_tags(self.trial_tags)
        tags.pop(str(label).strip(), None)
        self.trial_tags = tags
        self._update_trial_analysis()
        return self

    def set_starred_only(self, value: bool = True) -> Self:
        """Show only starred reconstruction candidates in the frontend."""

        self.show_starred_only = bool(value)
        return self

    def set_trial_sort(
        self,
        key: str = "final_loss",
        *,
        descending: bool | None = None,
        top: int | None = None,
        filter_text: str | None = None,
    ) -> Self:
        """Set ranking/sorting controls used by the frontend review panel."""

        self.trial_sort_key = self._normalise_trial_sort_key(key)
        if descending is not None:
            self.trial_sort_descending = bool(descending)
        if top is not None:
            self.top_trial_count = max(0, int(top))
        if filter_text is not None:
            self.trial_filter_text = str(filter_text)
        self._update_trial_analysis()
        return self

    def rank_trials(self, key: str | None = None) -> list[dict[str, Any]]:
        """Recompute and return reconstruction ranking rows."""

        if key is not None:
            self.trial_sort_key = self._normalise_trial_sort_key(key)
        self._update_trial_analysis()
        return [dict(row) for row in self.trial_rankings]

    def star_best_trial(self) -> Self:
        """Star the current best ranked non-hidden trial."""

        self._update_trial_analysis()
        if self.best_trial_label:
            self.star_trial(self.best_trial_label)
        return self

    def hide_worst_trials(self, count: int = 1) -> Self:
        """Hide the worst ranked non-starred trials."""

        self._update_trial_analysis()
        visible = [
            row for row in self.trial_rankings
            if not row.get("hidden") and not row.get("starred") and row.get("label") != self.best_trial_label
        ]
        for row in visible[-max(0, int(count)):]:
            label = str(row.get("label") or "")
            if label:
                self.hide_trial(label)
        self._update_trial_analysis()
        return self

    def export_run_summary(self, path: str | pathlib.Path) -> pathlib.Path:
        """Write a JSON summary of ranking, stars, hidden trials, tags, and alerts."""

        self._update_trial_analysis()
        out = pathlib.Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(_json_safe(self.run_summary), allow_nan=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return out

    def refresh_monitor(self, *, incremental: bool = True) -> Self:
        """Refresh the current monitor file while preserving review choices.

        By default only newly appended JSONL lines are read and applied. Pass
        ``incremental=False`` to rebuild from the full monitor file, which is
        useful after a run directory is replaced or edited by hand.
        """

        if not self.monitor_path:
            raise ValueError("monitor_path is empty")
        monitor_file = self._resolve_monitor_file(self.monitor_path)

        if incremental:
            if monitor_file.exists() and monitor_file.stat().st_size < self._monitor_offset:
                incremental = False
            else:
                events, offset, consumed_lines = self._read_monitor_events_from(
                    monitor_file,
                    offset=self._monitor_offset,
                    start_line=self._monitor_line_count + 1,
                )
                self._monitor_offset = offset
                self._monitor_line_count += consumed_lines
                if events:
                    self.apply_monitor_events(events, base_path=monitor_file)
                metadata = dict(self.report_metadata)
                metadata.pop("monitor_error", None)
                metadata.update(
                    {
                        "monitor_path": str(monitor_file),
                        "monitor_offset": self._monitor_offset,
                        "monitor_lines": self._monitor_line_count,
                    }
                )
                self.report_metadata = _json_safe(metadata)
                self._update_trial_analysis()
                return self

        review_state = {
            "starred_snapshot_image_labels": list(self.starred_snapshot_image_labels),
            "hidden_snapshot_image_labels": list(self.hidden_snapshot_image_labels),
            "trial_notes": dict(self.trial_notes),
            "trial_tags": dict(self.trial_tags),
            "show_starred_only": self.show_starred_only,
            "trial_sort_key": self.trial_sort_key,
            "trial_sort_descending": self.trial_sort_descending,
            "trial_filter_text": self.trial_filter_text,
            "top_trial_count": self.top_trial_count,
        }
        fresh = type(self).from_monitor_file(
            monitor_file,
            title=self.title or "Overnight Reconstruction Monitor",
            x_label=self.x_label or "iteration",
            y_label=self.y_label or "loss",
            log_scale=self.log_scale,
            plot_height_px=self.plot_height_px,
            side_panel_width_px=self.side_panel_width_px,
            image_cmap=self.image_cmap,
            snapshot_contrast_preset=self.snapshot_contrast_preset,
            snapshot_contrast_range=list(self.snapshot_contrast_range),
            snapshot_thumbnail_size=self.snapshot_thumbnail_size,
            snapshot_panel_width_px=self.snapshot_panel_width_px,
            snapshot_columns=self.snapshot_columns,
        )
        self._data = np.ascontiguousarray(fresh._data, dtype=np.float32)
        self._x = None if fresh._x is None else np.ascontiguousarray(fresh._x, dtype=np.float32)
        self._snapshots = [snap.copy() for snap in fresh._snapshots]
        self.labels = list(fresh.labels)
        self.colors = list(fresh.colors)
        self.n_traces = fresh.n_traces
        self.n_points = fresh.n_points
        self.snapshot_iterations = list(fresh.snapshot_iterations)
        self.snapshot_labels = list(fresh.snapshot_labels)
        self.snapshot_image_labels = list(fresh.snapshot_image_labels)
        self.snapshot_group_indices = list(fresh.snapshot_group_indices)
        self.snapshot_group_iterations = list(fresh.snapshot_group_iterations)
        self.snapshot_group_labels = list(fresh.snapshot_group_labels)
        self.report_metadata = dict(fresh.report_metadata)
        self._monitor_offset = fresh._monitor_offset
        self._monitor_line_count = fresh._monitor_line_count
        self.load_state_dict(review_state)
        self._update_stats()
        self._update_data_bytes()
        self._update_snapshot_bytes()
        self._update_trial_analysis()
        return self

    def start_monitor(self) -> Self:
        """Start a lightweight polling thread for ``monitor_path``."""

        if not self.monitor_path:
            raise ValueError("monitor_path is empty")
        if self.monitor_refresh_s <= 0:
            raise ValueError("monitor_refresh_s must be > 0 to start polling")
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return self
        self._monitor_stop = threading.Event()

        def _poll() -> None:
            while self._monitor_stop is not None and not self._monitor_stop.is_set():
                try:
                    path = self._resolve_monitor_file(self.monitor_path)
                    mtime = path.stat().st_mtime
                    if mtime > self._monitor_mtime:
                        self._monitor_mtime = mtime
                        self.refresh_monitor(incremental=True)
                except Exception as exc:  # pragma: no cover - background safety path
                    self.report_metadata = {
                        **dict(self.report_metadata),
                        "monitor_error": str(exc),
                    }
                time.sleep(max(0.25, float(self.monitor_refresh_s)))

        self._monitor_thread = threading.Thread(target=_poll, name="Show1DMonitor", daemon=True)
        self._monitor_thread.start()
        return self

    def stop_monitor(self) -> Self:
        """Stop the monitor polling thread if one is running."""

        if self._monitor_stop is not None:
            self._monitor_stop.set()
        self._monitor_thread = None
        self._monitor_stop = None
        return self

    def export_csv(self, path: str | pathlib.Path, *, visible_range_only: bool = False) -> pathlib.Path:
        """Write trace values to CSV."""

        out = pathlib.Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        x = self._effective_x()
        mask = np.ones(x.shape, dtype=bool)
        if visible_range_only and len(self.x_range) == 2:
            lo, hi = self.x_range
            mask = (x >= lo) & (x <= hi)
        with out.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["x", *self.labels])
            for idx in np.flatnonzero(mask):
                writer.writerow([x[idx], *[self._data[t, idx] for t in range(self.n_traces)]])
        return out

    def save_image(
        self,
        path: str | pathlib.Path,
        *,
        format: str | None = None,
        dpi: int = 150,
    ) -> pathlib.Path:
        """Save a publication-style PNG or PDF line figure via matplotlib."""

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out = pathlib.Path(path)
        fmt = (format or out.suffix.lstrip(".").lower() or "png").lower()
        if fmt not in {"png", "pdf"}:
            raise ValueError(f"Unsupported format {fmt!r}. Use 'png' or 'pdf'.")
        out.parent.mkdir(parents=True, exist_ok=True)
        x = self._effective_x()
        fig, ax = plt.subplots(figsize=(6, 3.5), dpi=dpi)
        for idx in range(self.n_traces):
            label = self.labels[idx] if idx < len(self.labels) else f"trace {idx + 1}"
            color = self.colors[idx] if idx < len(self.colors) else None
            ax.plot(x, self._data[idx], color=color, label=label, linewidth=self.line_width)
        for marker in self.markers:
            ax.axvline(float(marker.get("x", 0.0)), color="#999", linestyle="--", linewidth=0.8)
            if marker.get("label"):
                ax.text(float(marker["x"]), ax.get_ylim()[1], str(marker["label"]), fontsize=7, va="top")
        if self.log_scale:
            ax.set_yscale("log")
        if self.show_grid:
            ax.grid(True, alpha=0.3, linestyle="--")
        if self.show_legend and self.n_traces > 1:
            ax.legend(fontsize=8, framealpha=0.8)
        xlabel = self.x_label + (f" ({self.x_unit})" if self.x_label and self.x_unit else self.x_unit)
        ylabel = self.y_label + (f" ({self.y_unit})" if self.y_label and self.y_unit else self.y_unit)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if self.show_title and self.title:
            ax.set_title(self.title)
        fig.tight_layout()
        fig.savefig(out, format=fmt, bbox_inches="tight")
        plt.close(fig)
        return out

    def state_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "labels": list(self.labels),
            "colors": list(self.colors),
            "method_labels": list(self.method_labels),
            "x_label": self.x_label,
            "y_label": self.y_label,
            "x_unit": self.x_unit,
            "y_unit": self.y_unit,
            "log_scale": self.log_scale,
            "show_title": self.show_title,
            "show_stats": self.show_stats,
            "show_review": self.show_review,
            "show_legend": self.show_legend,
            "show_grid": self.show_grid,
            "show_controls": self.show_controls,
            "controls_collapsed": self.controls_collapsed,
            "line_width": self.line_width,
            "plot_height_px": self.plot_height_px,
            "side_panel_width_px": self.side_panel_width_px,
            "focused_trace": self.focused_trace,
            "x_range": list(self.x_range),
            "y_range": list(self.y_range),
            "markers": list(self.markers),
            "selected_snapshot_idx": self.selected_snapshot_idx,
            "selected_snapshot_group_idx": self.selected_snapshot_group_idx,
            "show_snapshots": self.show_snapshots,
            "show_snapshot_thumbnails": self.show_snapshot_thumbnails,
            "show_snapshot_histogram": self.show_snapshot_histogram,
            "show_snapshot_fft": self.show_snapshot_fft,
            "snapshot_fft_window": self.snapshot_fft_window,
            "snapshot_fft_cmap": self.snapshot_fft_cmap,
            "show_snapshot_profile": self.show_snapshot_profile,
            "snapshot_profile_line": [dict(point) for point in self.snapshot_profile_line],
            "snapshot_profile_height": self.snapshot_profile_height,
            "snapshot_contrast_preset": self.snapshot_contrast_preset,
            "snapshot_contrast_range": list(self.snapshot_contrast_range),
            "snapshot_thumbnail_size": self.snapshot_thumbnail_size,
            "snapshot_panel_width_px": self.snapshot_panel_width_px,
            "snapshot_columns": self.snapshot_columns,
            "snapshot_overlay_position": self.snapshot_overlay_position,
            "snapshot_real_space_zoom": self.snapshot_real_space_zoom,
            "snapshot_real_space_center": list(self.snapshot_real_space_center),
            "snapshot_fft_zoom": self.snapshot_fft_zoom,
            "snapshot_fft_center": list(self.snapshot_fft_center),
            "image_cmap": self.image_cmap,
            "starred_snapshot_image_labels": list(self.starred_snapshot_image_labels),
            "hidden_snapshot_image_labels": list(self.hidden_snapshot_image_labels),
            "trial_notes": dict(self.trial_notes),
            "trial_tags": {str(k): list(v) for k, v in self.trial_tags.items()},
            "show_trial_notes": self.show_trial_notes,
            "show_starred_only": self.show_starred_only,
            "trial_sort_key": self.trial_sort_key,
            "trial_sort_descending": self.trial_sort_descending,
            "trial_filter_text": self.trial_filter_text,
            "top_trial_count": self.top_trial_count,
            "trial_rankings": _json_safe([dict(row) for row in self.trial_rankings]),
            "trial_alerts": _json_safe([dict(row) for row in self.trial_alerts]),
            "best_trial_label": self.best_trial_label,
            "run_summary": _json_safe(dict(self.run_summary)),
            "pixel_size": self.pixel_size,
            "pixel_unit": self.pixel_unit,
            "scale_bar_visible": self.scale_bar_visible,
            "prefer_webgpu": self.prefer_webgpu,
            "snapshot_playing": self.snapshot_playing,
            "snapshot_fps": self.snapshot_fps,
            "profile_line": list(self.profile_line),
            "profile_width": self.profile_width,
            "report_metadata": _json_safe(dict(self.report_metadata)),
            "monitor_path": self.monitor_path,
            "monitor_refresh_s": self.monitor_refresh_s,
        }

    def collapse_controls(self) -> Self:
        """Collapse controls behind the frontend ``Controls`` button."""
        self.controls_collapsed = True
        return self

    def expand_controls(self) -> Self:
        """Expand frontend controls when ``show_controls`` is enabled."""
        self.controls_collapsed = False
        return self

    def toggle_controls(self) -> Self:
        """Toggle whether frontend controls start collapsed."""
        self.controls_collapsed = not bool(self.controls_collapsed)
        return self

    def save(self, path: str | pathlib.Path) -> None:
        save_state_file(path, "Show1D", self.state_dict())

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for key, value in state.items():
            if hasattr(self, key):
                if key in {"starred_snapshot_image_labels", "hidden_snapshot_image_labels"}:
                    value = self._normalise_trial_labels(value or [])
                elif key == "trial_notes":
                    value = self._normalise_trial_notes(value or {})
                elif key == "trial_tags":
                    value = self._normalise_trial_tags(value or {})
                elif key == "trial_sort_key":
                    value = self._normalise_trial_sort_key(str(value))
                elif key == "top_trial_count":
                    value = max(0, int(value))
                elif key == "snapshot_fps":
                    value = max(1, min(24, int(round(float(value)))))
                elif key == "snapshot_profile_line":
                    value = self._normalise_profile_line(value)
                elif key == "snapshot_profile_height":
                    value = max(44, min(220, int(value)))
                elif key in {"snapshot_real_space_zoom", "snapshot_fft_zoom"}:
                    value = self._normalise_snapshot_view_zoom(value)
                elif key in {"snapshot_real_space_center", "snapshot_fft_center"}:
                    value = self._normalise_snapshot_view_center(value)
                setattr(self, key, value)
        self._update_trial_analysis()

    def export_html(
        self,
        path: str | pathlib.Path | None = None,
        *,
        title: str | None = None,
        mode: str = "single",
        encoding: str = "full",
        downsample: int | None = None,
        **_: Any,
    ) -> pathlib.Path:
        """Write a standalone interactive HTML viewer."""

        self._normalise_html_export_options(mode=mode, encoding=encoding, downsample=downsample)
        export_path = pathlib.Path(path) if path is not None else self._default_html_export_path()
        self._write_html_export(export_path, title=title)
        size_mb = export_path.stat().st_size / (1024 * 1024)
        self.export_status = f"Exported {export_path.name} ({size_mb:.1f} MB, full float32)"
        return export_path

    def summary(self) -> None:
        lines = [self.title or "Show1D", "=" * 32]
        lines.append(f"Series:   {self.n_traces} x {self.n_points} points")
        if self.labels:
            lines.append(f"Labels:   {', '.join(self.labels)}")
        if self._x is not None and self._x.size:
            lines.append(f"X range:  {float(self._x[0]):.4g} - {float(self._x[-1]):.4g}")
        display = "log" if self.log_scale else "linear"
        if self.show_grid:
            display += " | grid"
        if self.n_snapshots:
            display += f" | {self.n_snapshots} snapshot(s)"
        if self.profile_image_bytes:
            display += " | image profile"
        lines.append(f"Display:  {display}")
        print("\n".join(lines))

    def __repr__(self) -> str:
        if self.n_traces == 1:
            return f"Show1D({self.n_points} points)"
        return f"Show1D({self.n_traces} traces x {self.n_points} points)"

    def _normalise_data(self, data: Any) -> tuple[np.ndarray, list[str]]:
        if data is None:
            return np.empty((0, 0), dtype=np.float32), []
        if hasattr(data, "array") and hasattr(data, "name"):
            name = str(getattr(data, "name") or "Data")
            data = data.array
            inferred_title = name
            if not self.title:
                self.title = inferred_title
        if isinstance(data, Mapping):
            labels = [str(k) for k in data]
            arrays = [np.asarray(to_numpy(v), dtype=np.float32).ravel() for v in data.values()]
            return self._stack_equal_length(arrays), labels
        if isinstance(data, list):
            arrays = [np.asarray(to_numpy(v), dtype=np.float32).ravel() for v in data]
            return self._stack_equal_length(arrays), [f"Data {i + 1}" for i in range(len(arrays))]
        arr = np.asarray(to_numpy(data), dtype=np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim == 1:
            return arr.reshape(1, -1), ["Data"]
        if arr.ndim == 2:
            return np.ascontiguousarray(arr, dtype=np.float32), [f"Data {i + 1}" for i in range(arr.shape[0])]
        raise ValueError(f"Expected 1D or 2D data, got shape {arr.shape}")

    def _stack_equal_length(self, arrays: list[np.ndarray]) -> np.ndarray:
        if not arrays:
            return np.empty((0, 0), dtype=np.float32)
        n = arrays[0].size
        for idx, arr in enumerate(arrays):
            if arr.size != n:
                raise ValueError(
                    f"All traces must have the same length. Trace 0 has {n} "
                    f"points, trace {idx} has {arr.size}."
                )
        return np.ascontiguousarray(np.stack(arrays), dtype=np.float32)

    def _normalise_x(self, x: Any, n_points: int) -> np.ndarray | None:
        if x is None:
            return None if n_points == 0 else np.arange(n_points, dtype=np.float32)
        arr = np.asarray(to_numpy(x), dtype=np.float32).ravel()
        if arr.size != n_points:
            raise ValueError(f"x has {arr.size} points but data has {n_points} points")
        return np.ascontiguousarray(arr, dtype=np.float32)

    def _effective_x(self) -> np.ndarray:
        if self._x is None:
            return np.arange(self.n_points, dtype=np.float32)
        return self._x

    def _default_colors(self, n: int) -> list[str]:
        return [_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)] for i in range(n)]

    def _normalise_image_cmap(self, cmap: str) -> str:
        name = str(cmap)
        if name not in _VALID_IMAGE_CMAPS:
            raise ValueError(f"Unknown image_cmap {name!r}. Valid: {sorted(_VALID_IMAGE_CMAPS)}")
        return name

    def _resolve_pixel_size(self, pixel_size: float | None, sampling: float | Sequence[float] | None) -> float:
        if pixel_size is not None:
            value = float(pixel_size)
        elif sampling is None:
            return 0.0
        elif isinstance(sampling, Sequence) and not isinstance(sampling, (str, bytes)):
            if not sampling:
                return 0.0
            value = float(sampling[-1])
        else:
            value = float(sampling)
        if not math.isfinite(value):
            raise ValueError(f"sampling/pixel_size must be finite, got {value}")
        if value < 0:
            raise ValueError(f"sampling/pixel_size must be >= 0, got {value}")
        return value

    def _resolve_pixel_unit(self, pixel_unit: str | None, units: str | Sequence[str] | None) -> str:
        if pixel_unit is not None:
            return str(pixel_unit)
        if units is None:
            return "px"
        if isinstance(units, str):
            return units
        if not units:
            return "px"
        return str(units[-1])

    def _normalise_snapshot_contrast_preset(self, preset: str) -> str:
        name = str(preset).strip()
        if name not in _VALID_SNAPSHOT_CONTRAST_PRESETS:
            raise ValueError(
                "Unknown snapshot_contrast_preset "
                f"{name!r}. Valid: {sorted(_VALID_SNAPSHOT_CONTRAST_PRESETS)}"
            )
        return name

    def _normalise_snapshot_contrast_range(self, value: Sequence[float] | None) -> list[float]:
        if value is None:
            return []
        if len(value) == 0:
            return []
        if len(value) != 2:
            raise ValueError(
                "snapshot_contrast_range must be empty or contain exactly "
                f"two values, got {value!r}"
            )
        vmin = float(value[0])
        vmax = float(value[1])
        if not math.isfinite(vmin) or not math.isfinite(vmax):
            raise ValueError(
                "snapshot_contrast_range values must be finite, "
                f"got {value!r}"
            )
        if vmax <= vmin:
            raise ValueError(
                "snapshot_contrast_range must be increasing "
                f"(min < max), got {value!r}"
            )
        return [vmin, vmax]

    def _normalise_snapshot_overlay_position(self, position: str) -> str:
        name = str(position).strip().lower().replace("_", "-")
        valid = {"top-left", "top-right", "bottom-left", "bottom-right"}
        if name not in valid:
            raise ValueError(
                "snapshot_overlay_position must be one of "
                f"{sorted(valid)}, got {position!r}"
            )
        return name

    def _normalise_snapshot_view_zoom(self, value: float) -> float:
        zoom = float(value)
        if not math.isfinite(zoom):
            raise ValueError(f"snapshot view zoom must be finite, got {value!r}")
        return max(1.0, min(32.0, zoom))

    def _normalise_snapshot_view_center(self, value: Sequence[float] | None) -> list[float]:
        if value is None:
            return []
        if len(value) == 0:
            return []
        if len(value) != 2:
            raise ValueError(
                "snapshot view center must be empty or contain exactly two "
                f"(row, col) values, got {value!r}"
            )
        row = float(value[0])
        col = float(value[1])
        if not math.isfinite(row) or not math.isfinite(col):
            raise ValueError(f"snapshot view center values must be finite, got {value!r}")
        return [row, col]

    def _normalise_profile_line(self, value: Sequence[Any] | None) -> list[dict[str, float]]:
        if value is None:
            return []
        if len(value) == 0:
            return []
        if len(value) > 2:
            raise ValueError(
                "snapshot_profile_line must contain at most two "
                f"(row, col) points, got {value!r}"
            )
        out: list[dict[str, float]] = []
        for point in value:
            if isinstance(point, Mapping):
                row = float(point.get("row", math.nan))
                col = float(point.get("col", math.nan))
            else:
                if len(point) != 2:
                    raise ValueError(
                        "snapshot_profile_line points must be (row, col), "
                        f"got {point!r}"
                    )
                row = float(point[0])
                col = float(point[1])
            if not math.isfinite(row) or not math.isfinite(col):
                raise ValueError(
                    "snapshot_profile_line coordinates must be finite, "
                    f"got {point!r}"
                )
            out.append({"row": row, "col": col})
        return out

    @staticmethod
    def _trial_label_key(label: str) -> str:
        return "".join(ch.lower() for ch in str(label) if ch.isalnum())

    @classmethod
    def _normalise_trial_labels(cls, labels: Sequence[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for label in labels:
            text = str(label).strip()
            key = cls._trial_label_key(text)
            if not text or not key or key in seen:
                continue
            seen.add(key)
            out.append(text)
        return out

    @classmethod
    def _add_trial_label(cls, labels: Sequence[str], label: str) -> list[str]:
        clean = str(label).strip()
        key = cls._trial_label_key(clean)
        current = cls._normalise_trial_labels(labels)
        if not clean or not key:
            raise ValueError("trial label must be a non-empty string")
        if any(cls._trial_label_key(existing) == key for existing in current):
            return current
        return [*current, clean]

    @classmethod
    def _remove_trial_label(cls, labels: Sequence[str], label: str) -> list[str]:
        key = cls._trial_label_key(str(label).strip())
        return [existing for existing in cls._normalise_trial_labels(labels) if cls._trial_label_key(existing) != key]

    @classmethod
    def _label_in_collection(cls, label: str, values: Sequence[str]) -> bool:
        key = cls._trial_label_key(label)
        return any(cls._trial_label_key(value) == key for value in values)

    @staticmethod
    def _is_reference_snapshot_label(label: str) -> bool:
        clean = str(label).strip().lower().replace("_", " ")
        return clean in {"reference", "clean reference", "ref"} or clean.startswith("reference ")

    def _snapshot_image_label(self, index: int) -> str:
        idx = int(index)
        if 0 <= idx < len(self.snapshot_image_labels):
            return str(self.snapshot_image_labels[idx])
        if 0 <= idx < len(self.snapshot_labels):
            return str(self.snapshot_labels[idx])
        return f"image {idx + 1}"

    def _normalise_snapshot_group_ref(self, group: int | str | None = None) -> int:
        if self.n_snapshots <= 0:
            raise ValueError("Show1D has no snapshot images")
        if self.n_snapshot_groups <= 0:
            if group is None:
                return max(0, min(self.n_snapshots - 1, int(self.selected_snapshot_idx)))
            idx = int(group)
            if not 0 <= idx < self.n_snapshots:
                raise ValueError(f"snapshot index {idx} out of range [0, {self.n_snapshots})")
            return idx
        if group is None:
            idx = int(self.selected_snapshot_group_idx)
            return max(0, min(self.n_snapshot_groups - 1, idx if idx >= 0 else 0))
        if isinstance(group, str):
            key = self._trial_label_key(group)
            for idx, label in enumerate(self.snapshot_group_labels):
                if self._trial_label_key(label) == key:
                    return idx
            raise ValueError(f"unknown snapshot group {group!r}")
        idx = int(group)
        if not 0 <= idx < self.n_snapshot_groups:
            raise ValueError(f"snapshot group index {idx} out of range [0, {self.n_snapshot_groups})")
        return idx

    def _snapshot_image_indices_for_group(self, group: int | str | None = None) -> list[int]:
        group_idx = self._normalise_snapshot_group_ref(group)
        if self.n_snapshot_groups <= 0:
            return [group_idx]
        return [
            image_idx for image_idx, image_group_idx in enumerate(self.snapshot_group_indices)
            if int(image_group_idx) == group_idx and 0 <= image_idx < len(self._snapshots)
        ]

    def _normalise_snapshot_image_refs(
        self,
        refs: Sequence[int | str] | int | str | None,
        candidates: Sequence[int],
    ) -> list[int]:
        candidate_list = [int(idx) for idx in candidates]
        if refs is None:
            return candidate_list
        values: Sequence[int | str]
        if isinstance(refs, (str, bytes)) or not isinstance(refs, Sequence):
            values = [refs]  # type: ignore[list-item]
        else:
            values = refs
        out: list[int] = []
        for value in values:
            if isinstance(value, str):
                key = self._trial_label_key(value)
                matches = [idx for idx in candidate_list if self._trial_label_key(self._snapshot_image_label(idx)) == key]
                if not matches:
                    raise ValueError(f"snapshot image label {value!r} is not in the selected group")
                for idx in matches:
                    if idx not in out:
                        out.append(idx)
                continue
            raw_idx = int(value)
            if 0 <= raw_idx < len(candidate_list):
                idx = candidate_list[raw_idx]
            elif raw_idx in candidate_list:
                idx = raw_idx
            else:
                raise ValueError(f"snapshot image index {raw_idx} is not in the selected group")
            if idx not in out:
                out.append(idx)
        return out

    def _snapshot_label_passes_review_filter(self, label: str) -> bool:
        if self._is_reference_snapshot_label(label):
            return True
        key = self._trial_label_key(label)
        if self.show_starred_only and not self._label_in_collection(label, self.starred_snapshot_image_labels):
            return False
        if self.top_trial_count > 0:
            top_keys = {
                self._trial_label_key(str(row.get("label", "")))
                for row in list(self.trial_rankings)[: int(self.top_trial_count)]
            }
            if key not in top_keys:
                return False
        filter_text = str(self.trial_filter_text or "").strip().lower()
        if filter_text:
            note = str(self._lookup_by_trial_key(self.trial_notes, label) or "")
            tags = " ".join(str(tag) for tag in (self._lookup_by_trial_key(self.trial_tags, label) or []))
            haystack = f"{label} {note} {tags}".lower()
            if filter_text not in haystack:
                return False
        return True

    @classmethod
    def _normalise_trial_notes(cls, notes: Mapping[str, str]) -> dict[str, str]:
        out: dict[str, str] = {}
        seen: set[str] = set()
        for label, note in notes.items():
            text = str(note).strip()
            clean_label = str(label).strip()
            key = cls._trial_label_key(clean_label)
            if not clean_label or not key or key in seen or not text:
                continue
            seen.add(key)
            out[clean_label] = text
        return out

    @classmethod
    def _normalise_trial_tags(cls, tags: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        seen_labels: set[str] = set()
        for label, values in tags.items():
            clean_label = str(label).strip()
            key = cls._trial_label_key(clean_label)
            if not clean_label or not key or key in seen_labels:
                continue
            seen_labels.add(key)
            clean_values: list[str] = []
            seen_tags: set[str] = set()
            raw_values = [values] if isinstance(values, (str, bytes)) else values
            for tag in raw_values:
                clean_tag = str(tag).strip()
                if not clean_tag or clean_tag in seen_tags:
                    continue
                clean_values.append(clean_tag)
                seen_tags.add(clean_tag)
            if clean_values:
                out[clean_label] = clean_values
        return out

    @staticmethod
    def _normalise_trial_sort_key(key: str) -> str:
        name = str(key or "final_loss").strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "loss": "final_loss",
            "final": "final_loss",
            "rmse_per_frame_mask": "rmse",
            "rmse_time_average_mask": "rmse",
            "temporal_flicker": "flicker",
            "temporal_flicker_mask": "flicker",
            "quality": "object_quality",
        }
        name = aliases.get(name, name)
        valid = {
            "default",
            "label",
            "lambda",
            "final_loss",
            "min_loss",
            "rmse",
            "flicker",
            "object_quality",
            "probe_quality",
            "alert_count",
        }
        if name not in valid:
            raise ValueError(f"Unknown trial_sort_key {key!r}. Valid: {sorted(valid)}")
        return name

    @staticmethod
    def _parse_lambda_from_label(label: str) -> float:
        text = str(label).replace("_", " ").lower()
        if "lambda" not in text:
            return float("nan")
        tail = text.split("lambda", 1)[1].strip().split()[0] if text.split("lambda", 1)[1].strip() else ""
        try:
            return float(tail)
        except ValueError:
            return float("nan")

    @classmethod
    def _lookup_by_trial_key(cls, mapping: Mapping[str, Any], label: str) -> Any:
        key = cls._trial_label_key(label)
        for raw_label, value in mapping.items():
            if cls._trial_label_key(str(raw_label)) == key:
                return value
        return None

    def _snapshot_quality_by_label(self) -> dict[str, dict[str, float | bool]]:
        grouped: dict[str, list[tuple[float, np.ndarray]]] = {}
        for idx, snap in enumerate(self._snapshots):
            label = self.snapshot_image_labels[idx] if idx < len(self.snapshot_image_labels) else f"image {idx + 1}"
            key = self._trial_label_key(label)
            group_idx = self.snapshot_group_indices[idx] if idx < len(self.snapshot_group_indices) else idx
            iteration = (
                self.snapshot_group_iterations[group_idx]
                if 0 <= int(group_idx) < len(self.snapshot_group_iterations)
                else self.snapshot_iterations[idx] if idx < len(self.snapshot_iterations)
                else float(idx)
            )
            grouped.setdefault(key, []).append((float(iteration), snap))

        out: dict[str, dict[str, float | bool]] = {}
        for key, frames in grouped.items():
            stats: list[tuple[float, float, np.ndarray]] = []
            for iteration, image in sorted(frames, key=lambda item: item[0]):
                finite = np.asarray(image[np.isfinite(image)], dtype=np.float32)
                if finite.size == 0:
                    continue
                stats.append((float(np.nanstd(finite)), float(np.nanmean(np.abs(finite))), image))
            if not stats:
                continue
            stds = np.asarray([item[0] for item in stats], dtype=np.float32)
            means = np.asarray([item[1] for item in stats], dtype=np.float32)
            diffs: list[float] = []
            for (_, mean_abs, prev), (_, _, current) in zip(stats[:-1], stats[1:], strict=False):
                if prev.shape != current.shape:
                    continue
                denom = max(float(mean_abs), 1e-6)
                diffs.append(float(np.nanmean(np.abs(current - prev)) / denom))
            flicker = float(np.nanmedian(diffs)) if diffs else float("nan")
            quality = float(np.nanmedian(stds))
            out[key] = {
                "image_std": quality,
                "image_mean_abs": float(np.nanmedian(means)),
                "image_flicker": flicker,
                "collapsed": bool(np.nanmax(stds) < 1e-7 or np.nanmax(means) < 1e-9),
            }
        return out

    def _metric_map_for_label(self, label: str) -> dict[str, Any]:
        raw = self.report_metadata.get("metrics_by_trial", {})
        if not isinstance(raw, Mapping):
            return {}
        match = self._lookup_by_trial_key(raw, label)
        return dict(match) if isinstance(match, Mapping) else {}

    @staticmethod
    def _first_metric(metrics: Mapping[str, Any], keys: Sequence[str]) -> float:
        for key in keys:
            if key in metrics:
                value = metrics[key]
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    finite = np.asarray([_as_float(item) for item in value], dtype=np.float32)
                    finite = finite[np.isfinite(finite)]
                    return float(finite[-1]) if finite.size else float("nan")
                return _as_float(value)
        return float("nan")

    @staticmethod
    def _score_for_ranking(row: Mapping[str, Any], key: str) -> float:
        if key == "default":
            key = "final_loss"
        if key == "label":
            return float("nan")
        if key == "object_quality":
            value = _as_float(row.get("object_quality"))
            return -value if math.isfinite(value) else float("nan")
        if key == "probe_quality":
            value = _as_float(row.get("probe_quality"))
            return -value if math.isfinite(value) else float("nan")
        return _as_float(row.get(key))

    def _compute_trial_rankings(self) -> list[dict[str, Any]]:
        image_quality = self._snapshot_quality_by_label()
        rows: list[dict[str, Any]] = []
        for idx, label in enumerate(self.labels):
            y = np.asarray(self._data[idx], dtype=np.float32) if idx < self._data.shape[0] else np.empty(0, dtype=np.float32)
            finite = y[np.isfinite(y)]
            first = float(finite[0]) if finite.size else float("nan")
            final = float(finite[-1]) if finite.size else float("nan")
            min_loss = float(np.nanmin(finite)) if finite.size else float("nan")
            metrics = self._metric_map_for_label(label)
            rmse = self._first_metric(metrics, ("rmse", "rmse_per_frame_mask", "rmse_time_average_mask", "reference_rmse"))
            flicker = self._first_metric(metrics, ("flicker", "temporal_flicker", "temporal_flicker_mask", "mean_phase_std_mask"))
            key = self._trial_label_key(label)
            quality = image_quality.get(key, {})
            image_std = _as_float(quality.get("image_std"))
            object_quality = _as_float(metrics.get("object_quality", image_std))
            probe_quality = _as_float(metrics.get("probe_quality", image_std))
            row = {
                "label": str(label),
                "trace_index": idx,
                "lambda": self._parse_lambda_from_label(label),
                "first_loss": first,
                "final_loss": final,
                "min_loss": min_loss,
                "mean_loss": float(np.nanmean(finite)) if finite.size else float("nan"),
                "std_loss": float(np.nanstd(finite)) if finite.size else float("nan"),
                "rmse": rmse,
                "flicker": flicker if math.isfinite(flicker) else _as_float(quality.get("image_flicker")),
                "object_quality": object_quality,
                "probe_quality": probe_quality,
                "image_std": image_std,
                "image_collapsed": bool(quality.get("collapsed", False)),
                "nan_count": int(np.count_nonzero(~np.isfinite(y))),
                "starred": self._label_in_collection(label, self.starred_snapshot_image_labels),
                "hidden": self._label_in_collection(label, self.hidden_snapshot_image_labels),
                "note": str(self._lookup_by_trial_key(self.trial_notes, label) or ""),
                "tags": list(self._lookup_by_trial_key(self.trial_tags, label) or []),
            }
            rows.append(row)

        alerts = self._compute_trial_alerts_from_rows(rows)
        alert_counts: dict[str, int] = {}
        for alert in alerts:
            label = str(alert.get("label") or "")
            if label:
                alert_counts[label] = alert_counts.get(label, 0) + 1
        sort_key = self._normalise_trial_sort_key(self.trial_sort_key)
        for row in rows:
            row["alert_count"] = alert_counts.get(str(row["label"]), 0)
            row["score"] = self._score_for_ranking(row, sort_key)

        if sort_key == "label":
            rows.sort(key=lambda row: str(row["label"]).lower(), reverse=self.trial_sort_descending)
        else:
            rows.sort(
                key=lambda row: (
                    not math.isfinite(_as_float(row.get("score"))),
                    _as_float(row.get("score")) if math.isfinite(_as_float(row.get("score"))) else math.inf,
                    str(row["label"]).lower(),
                ),
                reverse=self.trial_sort_descending,
            )
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        return rows

    def _compute_trial_alerts_from_rows(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        alerts: list[dict[str, Any]] = []
        for row in rows:
            label = str(row.get("label") or "")
            idx = int(row.get("trace_index", -1))
            if not label or idx < 0 or idx >= self._data.shape[0]:
                continue
            y = np.asarray(self._data[idx], dtype=np.float32)
            finite = y[np.isfinite(y)]
            if int(row.get("nan_count", 0)) > 0:
                alerts.append({"label": label, "kind": "nonfinite", "severity": "error", "message": "contains NaN/inf values"})
            if finite.size >= 2:
                first = float(finite[0])
                final = float(finite[-1])
                if math.isfinite(first) and math.isfinite(final):
                    base = max(abs(first), 1e-12)
                    if final > first and (final - first) / base > 0.25:
                        alerts.append({"label": label, "kind": "worse_final", "severity": "warning", "message": "final loss is worse than initial loss"})
                    if np.nanmax(np.abs(finite)) > 10 * max(np.nanmedian(np.abs(finite)), 1e-12):
                        alerts.append({"label": label, "kind": "spike", "severity": "warning", "message": "large loss spike detected"})
            if finite.size >= 8:
                q = max(2, finite.size // 4)
                start = float(np.nanmedian(finite[:q]))
                end = float(np.nanmedian(finite[-q:]))
                improvement = (start - end) / max(abs(start), 1e-12)
                if abs(improvement) < 1e-3:
                    alerts.append({"label": label, "kind": "flat_loss", "severity": "info", "message": "loss is nearly flat"})
            if bool(row.get("image_collapsed")):
                alerts.append({"label": label, "kind": "image_collapse", "severity": "error", "message": "snapshot image appears collapsed"})
            flicker = _as_float(row.get("flicker"))
            if math.isfinite(flicker) and flicker > 0.75:
                alerts.append({"label": label, "kind": "flicker", "severity": "warning", "message": "large frame-to-frame flicker"})

        warnings = self.report_metadata.get("monitor_warnings", [])
        if isinstance(warnings, Sequence) and not isinstance(warnings, (str, bytes)):
            for message in warnings:
                alerts.append({"label": "", "kind": "monitor_warning", "severity": "warning", "message": str(message)})
        elif warnings:
            alerts.append({"label": "", "kind": "monitor_warning", "severity": "warning", "message": str(warnings)})
        return alerts

    def _build_run_summary(self, rankings: Sequence[Mapping[str, Any]], alerts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        best = next((dict(row) for row in rankings if not row.get("hidden")), {})
        return {
            "title": self.title,
            "best_trial": best.get("label", ""),
            "best_score": best.get("score", float("nan")),
            "sort_key": self.trial_sort_key,
            "starred_trials": list(self.starred_snapshot_image_labels),
            "hidden_trials": list(self.hidden_snapshot_image_labels),
            "trial_notes": dict(self.trial_notes),
            "trial_tags": {str(k): list(v) for k, v in self.trial_tags.items()},
            "rankings": [dict(row) for row in rankings],
            "alerts": [dict(alert) for alert in alerts],
            "metadata": dict(self.report_metadata),
            "shape": {"n_traces": self.n_traces, "n_points": self.n_points, "n_snapshots": self.n_snapshots},
        }

    def _update_trial_analysis(self) -> None:
        rankings = self._compute_trial_rankings() if getattr(self, "_data", np.empty(0)).size or self.labels else []
        alerts = self._compute_trial_alerts_from_rows(rankings)
        self.trial_rankings = _json_safe([dict(row) for row in rankings])
        self.trial_alerts = _json_safe([dict(alert) for alert in alerts])
        best = next((row for row in rankings if not row.get("hidden")), None)
        self.best_trial_label = str(best.get("label", "")) if best else ""
        self.run_summary = _json_safe(self._build_run_summary(rankings, alerts))

    @staticmethod
    def _resolve_monitor_file(path: str | pathlib.Path, *, create: bool = False) -> pathlib.Path:
        raw = pathlib.Path(path)
        if raw.suffix:
            return raw
        if raw.is_dir() or create:
            return raw / "show1d_monitor.jsonl"
        return raw

    @staticmethod
    def _read_monitor_events(path: pathlib.Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            text = raw.strip()
            if not text or text.startswith("#"):
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid monitor JSON on line {line_no}: {exc}") from exc
            if not isinstance(event, dict):
                raise ValueError(f"monitor line {line_no} must be a JSON object")
            events.append(event)
        return events

    @staticmethod
    def _read_monitor_events_from(
        path: pathlib.Path,
        *,
        offset: int = 0,
        start_line: int = 1,
    ) -> tuple[list[dict[str, Any]], int, int]:
        """Read complete JSONL events appended after ``offset``.

        The final line is ignored until it ends with a newline. This avoids
        parsing a half-written event while a reconstruction process is flushing
        the monitor file.
        """

        if not path.exists():
            return [], max(0, int(offset)), 0
        size = path.stat().st_size
        clean_offset = max(0, int(offset))
        if clean_offset > size:
            clean_offset = 0
            start_line = 1
        with path.open("rb") as handle:
            handle.seek(clean_offset)
            chunk = handle.read()
        if not chunk:
            return [], clean_offset, 0
        complete = chunk
        if not complete.endswith(b"\n"):
            last_newline = complete.rfind(b"\n")
            if last_newline < 0:
                return [], clean_offset, 0
            complete = complete[: last_newline + 1]
        text = complete.decode("utf-8")
        events: list[dict[str, Any]] = []
        consumed_lines = 0
        for consumed_lines, raw in enumerate(text.splitlines(), start=1):
            line_no = start_line + consumed_lines - 1
            value = raw.strip()
            if not value or value.startswith("#"):
                continue
            try:
                event = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid monitor JSON on line {line_no}: {exc}") from exc
            if not isinstance(event, dict):
                raise ValueError(f"monitor line {line_no} must be a JSON object")
            events.append(event)
        return events, clean_offset + len(complete), consumed_lines

    @staticmethod
    def _load_monitor_image(path: pathlib.Path) -> np.ndarray | None:
        if not path.exists():
            return None
        if path.suffix == ".npy":
            arr = np.load(path)
        elif path.suffix == ".npz":
            with np.load(path) as data:
                if not data.files:
                    return None
                arr = data[data.files[0]]
        else:
            return None
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[-1]
        if arr.ndim != 2:
            return None
        return np.ascontiguousarray(arr, dtype=np.float32)

    def _update_data_bytes(self) -> None:
        self.y_bytes = _b64_safe(np.ascontiguousarray(self._data, dtype=np.float32).tobytes())
        x = self._effective_x()
        self.x_bytes = _b64_safe(np.ascontiguousarray(x, dtype=np.float32).tobytes()) if x.size else b""

    def _update_stats(self) -> None:
        if self._data.size == 0 or self.n_traces == 0:
            self.stats_mean = []
            self.stats_min = []
            self.stats_max = []
            self.stats_std = []
            return
        self.stats_mean = np.nanmean(self._data, axis=1).astype(float).tolist()
        self.stats_min = np.nanmin(self._data, axis=1).astype(float).tolist()
        self.stats_max = np.nanmax(self._data, axis=1).astype(float).tolist()
        self.stats_std = np.nanstd(self._data, axis=1).astype(float).tolist()

    def _update_snapshot_bytes(self) -> None:
        if not self._snapshots:
            self.snapshot_bytes = b""
            self.n_snapshots = 0
            self.snapshot_height = 0
            self.snapshot_width = 0
            self.snapshot_heights = []
            self.snapshot_widths = []
            self.snapshot_image_labels = []
            self.snapshot_group_indices = []
            self.snapshot_group_iterations = []
            self.snapshot_group_labels = []
            self.n_snapshot_groups = 0
            self.selected_snapshot_idx = -1
            self.selected_snapshot_group_idx = -1
            return

        heights = [int(snap.shape[0]) for snap in self._snapshots]
        widths = [int(snap.shape[1]) for snap in self._snapshots]
        max_height = max(heights)
        max_width = max(widths)
        stack = np.full((len(self._snapshots), max_height, max_width), np.nan, dtype=np.float32)
        for idx, snap in enumerate(self._snapshots):
            h, w = snap.shape
            stack[idx, :h, :w] = snap
        stack = np.ascontiguousarray(stack, dtype=np.float32)
        self.n_snapshots = int(stack.shape[0])
        self.snapshot_height = int(stack.shape[1])
        self.snapshot_width = int(stack.shape[2])
        self.snapshot_heights = heights
        self.snapshot_widths = widths
        if len(self.snapshot_image_labels) != self.n_snapshots:
            self.snapshot_image_labels = list(self.snapshot_labels)
        if len(self.snapshot_group_indices) != self.n_snapshots:
            self.snapshot_group_indices = list(range(self.n_snapshots))
        if not self.snapshot_group_iterations:
            self.snapshot_group_iterations = list(self.snapshot_iterations)
        if not self.snapshot_group_labels:
            self.snapshot_group_labels = list(self.snapshot_labels)
        self.n_snapshot_groups = len(self.snapshot_group_iterations)
        self.snapshot_bytes = _b64_safe(stack.tobytes())

    def _load_joint_time_snapshots(
        self,
        path: pathlib.Path,
        methods: Sequence[str],
        *,
        frame_by_frame: bool = False,
        include_reference: bool = True,
        downsample: int = 1,
        max_frames: int | None = None,
    ) -> None:
        step = max(1, int(downsample))
        with np.load(path) as arrays:
            reference = None
            if include_reference and "reference_phase" in arrays:
                reference = np.asarray(arrays["reference_phase"][::step, ::step], dtype=np.float32)

            if frame_by_frame:
                stacks: dict[str, np.ndarray] = {}
                n_frames = 0
                for method in methods:
                    if method not in arrays:
                        continue
                    arr = np.asarray(arrays[method], dtype=np.float32)
                    if arr.ndim != 3:
                        continue
                    stacks[_pretty_method_label(method).replace(" ", "_")] = arr[:, ::step, ::step]
                    n_frames = max(n_frames, arr.shape[0])
                if max_frames is not None:
                    n_frames = min(n_frames, max(0, int(max_frames)))
                for frame in range(n_frames):
                    images: dict[str, np.ndarray] = {}
                    if reference is not None:
                        images["reference"] = reference
                    for label, stack in stacks.items():
                        if frame < stack.shape[0]:
                            images[label] = np.asarray(stack[frame], dtype=np.float32)
                    if images:
                        self.snapshot(float(frame), label=f"frame {frame}", **images)
                return

            if reference is not None:
                self.snapshot(-1.0, reference, label="clean reference")
            for idx, method in enumerate(methods):
                if method not in arrays:
                    continue
                arr = np.asarray(arrays[method], dtype=np.float32)
                if arr.ndim == 3:
                    image = arr.mean(axis=0)[::step, ::step]
                    label = f"{method} average"
                elif arr.ndim == 2:
                    image = arr[::step, ::step]
                    label = method
                else:
                    continue
                self.snapshot(float(idx), image, label=label)

    def _normalise_html_export_options(
        self,
        *,
        mode: str,
        encoding: str,
        downsample: int | None,
    ) -> None:
        raw_mode = str(mode or "single").strip().lower().replace("_", "-")
        raw_encoding = str(encoding or "full").strip().lower().replace("_", "-")
        if raw_mode not in {"single", "exact", "full"}:
            raise ValueError("Show1D HTML export supports mode='single'")
        if raw_encoding not in {"full", "exact", "float32", "f32"}:
            raise ValueError("Show1D HTML export currently supports encoding='full'")
        if downsample not in (None, 1, "1", "", 0, "0"):
            raise NotImplementedError("Show1D HTML export does not support downsample")

    def _default_html_export_path(self) -> pathlib.Path:
        label = _slug(self.title or "show1d")
        shape = f"{self.n_traces}x{self.n_points}"
        return pathlib.Path.cwd() / f"{label}_{shape}_single.html"

    def _write_html_export(self, path: str | pathlib.Path, *, title: str | None = None) -> pathlib.Path:
        from ipywidgets.embed import dependency_state, embed_minimal_html

        export_path = pathlib.Path(path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_widget = self._clone_for_html_export()
        try:
            state = dependency_state([export_widget], drop_defaults=False)
            embed_minimal_html(
                str(export_path),
                views=[export_widget],
                title=title or self.title or "Show1D",
                drop_defaults=False,
                state=state,
            )
        finally:
            export_widget.close()
        ensure_mobile_viewport(export_path)
        return export_path

    def _html_export_bytes(self) -> bytes:
        with tempfile.TemporaryDirectory(prefix="show1d-export-") as tmp:
            path = pathlib.Path(tmp) / self._default_html_export_path().name
            self._write_html_export(path)
            return path.read_bytes()

    def _clone_for_html_export(self) -> Self:
        clone = type(self)(
            np.ascontiguousarray(self._data, dtype=np.float32),
            x=None if self._x is None else np.ascontiguousarray(self._x, dtype=np.float32),
            labels=list(self.labels),
            colors=list(self.colors),
            title=self.title,
            x_label=self.x_label,
            y_label=self.y_label,
            x_unit=self.x_unit,
            y_unit=self.y_unit,
            log_scale=self.log_scale,
            show_title=self.show_title,
            show_stats=self.show_stats,
            show_review=self.show_review,
            show_legend=self.show_legend,
            show_grid=self.show_grid,
            show_controls=self.show_controls,
            controls_collapsed=self.controls_collapsed,
            line_width=self.line_width,
            plot_height_px=self.plot_height_px,
            side_panel_width_px=self.side_panel_width_px,
            image_cmap=self.image_cmap,
            snapshot_contrast_preset=self.snapshot_contrast_preset,
            snapshot_contrast_range=list(self.snapshot_contrast_range),
            snapshot_thumbnail_size=self.snapshot_thumbnail_size,
            snapshot_panel_width_px=self.snapshot_panel_width_px,
            snapshot_columns=self.snapshot_columns,
            snapshot_overlay_position=self.snapshot_overlay_position,
            snapshot_real_space_zoom=self.snapshot_real_space_zoom,
            snapshot_real_space_center=list(self.snapshot_real_space_center),
            snapshot_fft_zoom=self.snapshot_fft_zoom,
            snapshot_fft_center=list(self.snapshot_fft_center),
            starred_snapshot_image_labels=list(self.starred_snapshot_image_labels),
            hidden_snapshot_image_labels=list(self.hidden_snapshot_image_labels),
            trial_notes=dict(self.trial_notes),
            trial_tags={str(k): list(v) for k, v in self.trial_tags.items()},
            show_trial_notes=self.show_trial_notes,
            show_starred_only=self.show_starred_only,
            trial_sort_key=self.trial_sort_key,
            trial_sort_descending=self.trial_sort_descending,
            trial_filter_text=self.trial_filter_text,
            top_trial_count=self.top_trial_count,
            pixel_size=self.pixel_size,
            pixel_unit=self.pixel_unit,
            show_scale_bar=self.scale_bar_visible,
            show_snapshot_histogram=self.show_snapshot_histogram,
            show_snapshot_fft=self.show_snapshot_fft,
            snapshot_fft_window=self.snapshot_fft_window,
            snapshot_fft_cmap=self.snapshot_fft_cmap,
            show_snapshot_profile=self.show_snapshot_profile,
            snapshot_profile_line=list(self.snapshot_profile_line),
            snapshot_profile_height=self.snapshot_profile_height,
            prefer_webgpu=self.prefer_webgpu,
        )
        clone.load_state_dict(self.state_dict())
        clone.method_labels = list(self.method_labels)
        clone.report_metadata = _json_safe(dict(self.report_metadata))
        clone._snapshots = [snap.copy() for snap in self._snapshots]
        clone.snapshot_iterations = list(self.snapshot_iterations)
        clone.snapshot_labels = list(self.snapshot_labels)
        clone.snapshot_image_labels = list(self.snapshot_image_labels)
        clone.starred_snapshot_image_labels = list(self.starred_snapshot_image_labels)
        clone.hidden_snapshot_image_labels = list(self.hidden_snapshot_image_labels)
        clone.trial_notes = dict(self.trial_notes)
        clone.trial_tags = {str(k): list(v) for k, v in self.trial_tags.items()}
        clone.show_trial_notes = self.show_trial_notes
        clone.show_starred_only = self.show_starred_only
        clone.trial_sort_key = self.trial_sort_key
        clone.trial_sort_descending = self.trial_sort_descending
        clone.trial_filter_text = self.trial_filter_text
        clone.top_trial_count = self.top_trial_count
        clone.trial_rankings = _json_safe([dict(row) for row in self.trial_rankings])
        clone.trial_alerts = _json_safe([dict(row) for row in self.trial_alerts])
        clone.best_trial_label = self.best_trial_label
        clone.run_summary = _json_safe(dict(self.run_summary))
        clone.snapshot_group_indices = list(self.snapshot_group_indices)
        clone.snapshot_group_iterations = list(self.snapshot_group_iterations)
        clone.snapshot_group_labels = list(self.snapshot_group_labels)
        clone.selected_snapshot_group_idx = self.selected_snapshot_group_idx
        clone.show_snapshot_thumbnails = self.show_snapshot_thumbnails
        clone.show_snapshot_fft = self.show_snapshot_fft
        clone.snapshot_fft_window = self.snapshot_fft_window
        clone.snapshot_fft_cmap = self.snapshot_fft_cmap
        clone.snapshot_contrast_preset = self.snapshot_contrast_preset
        clone.snapshot_contrast_range = list(self.snapshot_contrast_range)
        clone.snapshot_thumbnail_size = self.snapshot_thumbnail_size
        clone.snapshot_panel_width_px = self.snapshot_panel_width_px
        clone.snapshot_columns = self.snapshot_columns
        clone.snapshot_overlay_position = self.snapshot_overlay_position
        clone.snapshot_real_space_zoom = self.snapshot_real_space_zoom
        clone.snapshot_real_space_center = list(self.snapshot_real_space_center)
        clone.snapshot_fft_zoom = self.snapshot_fft_zoom
        clone.snapshot_fft_center = list(self.snapshot_fft_center)
        clone.pixel_size = self.pixel_size
        clone.pixel_unit = self.pixel_unit
        clone.scale_bar_visible = self.scale_bar_visible
        clone.snapshot_playing = self.snapshot_playing
        clone.snapshot_fps = self.snapshot_fps
        clone.show_snapshot_profile = self.show_snapshot_profile
        clone.snapshot_profile_line = list(self.snapshot_profile_line)
        clone.snapshot_profile_height = self.snapshot_profile_height
        clone._update_snapshot_bytes()
        if self._profile_image is not None:
            clone.set_profile_image(self._profile_image, line=None)
            clone.profile_line = list(self.profile_line)
            clone.profile_width = self.profile_width
        clone._export_light = True
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
        return clone

    def _on_handoff_request_change(self, change: dict[str, Any]) -> None:
        """Build a Python-side prepared view from a frontend request."""

        raw = str(change.get("new") or "")
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
                group=request.get("group", request.get("snapshot_group", None)),
                images=request.get("images", request.get("panels", None)),
                title=request.get("title", None),
                include_hidden=bool(request.get("include_hidden", False)),
                respect_review_filters=bool(request.get("respect_review_filters", True)),
            )
            self.prepared_view_widget = self.prepared_view
            n_images = int(getattr(self.prepared_view, "n_images", 0))
            self.handoff_status = f"Showing 2D with {n_images} panel{'s' if n_images != 1 else ''}"
        except Exception as exc:  # pragma: no cover - defensive comm boundary
            self.prepared_view = None
            self.prepared_view_widget = None
            self.handoff_status = f"View failed: {exc}"

    def _on_export_request_change(self, change: dict[str, Any]) -> None:
        raw = str(change.get("new") or "")
        if not raw:
            return
        try:
            payload = json.loads(raw)
            mode = str(payload.get("mode", "single"))
            if mode == "clear":
                self.export_payload = b""
                self.export_payload_id = ""
                self.export_filename = ""
                return
            self._normalise_html_export_options(
                mode=mode,
                encoding=str(payload.get("encoding", "full")),
                downsample=payload.get("downsample"),
            )
            if payload.get("download"):
                filename = str(payload.get("filename") or self._default_html_export_path().name)
                request_id = str(payload.get("id") or "")
                self.export_status = f"Preparing {filename}..."
                html = self._html_export_bytes()
                self.export_filename = filename
                self.export_payload = html
                self.export_payload_id = request_id
                size_mb = len(html) / (1024 * 1024)
                self.export_status = f"Ready {filename} ({size_mb:.1f} MB, full float32)"
            else:
                self.export_status = f"Exporting {mode} HTML..."
                self.export_html()
        except Exception as exc:
            self.export_status = f"Export failed: {exc}"


__all__ = ["Show1D", "sample_line_profile"]
