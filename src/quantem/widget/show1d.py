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
from collections.abc import Mapping, Sequence
from typing import Any, Self

import anywidget
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
    show_title, show_stats, show_legend, show_grid, show_controls : bool
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
    snapshot_thumbnail_size : int, default 48
        Size of plot-embedded snapshot thumbnails in pixels.
    snapshot_columns : int, default 2
        Number of columns used for the side-panel snapshot image grid, clamped
        to 1 through 4.
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
    show_stats = traitlets.Bool(True).tag(sync=True)
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
    snapshot_contrast_preset = traitlets.Unicode("full").tag(sync=True)
    snapshot_thumbnail_size = traitlets.Int(48).tag(sync=True)
    snapshot_columns = traitlets.Int(2).tag(sync=True)
    image_cmap = traitlets.Unicode("cividis").tag(sync=True)
    pixel_size = traitlets.Float(0.0).tag(sync=True)
    pixel_unit = traitlets.Unicode("px").tag(sync=True)
    scale_bar_visible = traitlets.Bool(True).tag(sync=True)
    prefer_webgpu = traitlets.Bool(True).tag(sync=True)
    snapshot_playing = traitlets.Bool(False).tag(sync=True)
    snapshot_fps = traitlets.Float(2.0).tag(sync=True)

    profile_image_bytes = traitlets.Bytes(b"").tag(sync=True)
    profile_image_height = traitlets.Int(0).tag(sync=True)
    profile_image_width = traitlets.Int(0).tag(sync=True)
    profile_line = traitlets.List(traitlets.Dict()).tag(sync=True)
    profile_width = traitlets.Int(1).tag(sync=True)

    report_metadata = traitlets.Dict().tag(sync=True)

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
        snapshot_thumbnail_size: int = 48,
        snapshot_columns: int = 2,
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
        prefer_webgpu: bool = True,
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
                "show_stats": True,
                "show_legend": True,
                "show_grid": True,
                "show_controls": True,
                "controls_collapsed": False,
            },
            overrides={
                "show_title": show_title,
                "show_stats": show_stats,
                "show_legend": show_legend,
                "show_grid": show_grid,
                "show_controls": show_controls,
                "controls_collapsed": controls_collapsed,
            },
        )
        self.show_title = bool(ui["show_title"])
        self.show_stats = bool(ui["show_stats"])
        self.show_legend = bool(ui["show_legend"])
        self.show_grid = bool(ui["show_grid"])
        self.show_controls = bool(ui["show_controls"])
        self.controls_collapsed = bool(ui["controls_collapsed"])
        self.line_width = float(line_width)
        self.plot_height_px = max(220, min(720, int(plot_height_px)))
        self.side_panel_width_px = max(300, min(640, int(side_panel_width_px)))
        self.profile_width = max(1, int(profile_width))
        self.image_cmap = self._normalise_image_cmap(image_cmap)
        self.snapshot_contrast_preset = self._normalise_snapshot_contrast_preset(snapshot_contrast_preset)
        self.snapshot_thumbnail_size = max(24, min(112, int(snapshot_thumbnail_size)))
        self.snapshot_columns = max(1, min(4, int(snapshot_columns)))
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
        self.prefer_webgpu = bool(prefer_webgpu)

        self._update_stats()
        self._update_data_bytes()
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

        self.observe(self._on_export_request_change, names=["export_request"])

    @traitlets.validate("image_cmap")
    def _validate_image_cmap(self, proposal: dict[str, Any]) -> str:
        return self._normalise_image_cmap(str(proposal["value"]))

    @traitlets.validate("snapshot_contrast_preset")
    def _validate_snapshot_contrast_preset(self, proposal: dict[str, Any]) -> str:
        return self._normalise_snapshot_contrast_preset(str(proposal["value"]))

    @traitlets.validate("plot_height_px")
    def _validate_plot_height_px(self, proposal: dict[str, Any]) -> int:
        return max(220, min(720, int(proposal["value"])))

    @traitlets.validate("side_panel_width_px")
    def _validate_side_panel_width_px(self, proposal: dict[str, Any]) -> int:
        return max(300, min(640, int(proposal["value"])))

    @traitlets.validate("snapshot_thumbnail_size")
    def _validate_snapshot_thumbnail_size(self, proposal: dict[str, Any]) -> int:
        return max(24, min(112, int(proposal["value"])))

    @traitlets.validate("snapshot_columns")
    def _validate_snapshot_columns(self, proposal: dict[str, Any]) -> int:
        return max(1, min(4, int(proposal["value"])))

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
        }

        if arrays_path is None:
            candidate = summary_file.parent / "reconstructions.npz"
            arrays_path = candidate if candidate.exists() else None
        if arrays_path is not None:
            widget._load_joint_time_snapshots(pathlib.Path(arrays_path), methods)
        return widget

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
        return self

    def append_scalar(self, iteration: float | None = None, **values: Any) -> Self:
        """Alias for :meth:`append` with reconstruction-friendly naming."""

        return self.append(x=iteration, **values)

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
            "snapshot_contrast_preset": self.snapshot_contrast_preset,
            "snapshot_thumbnail_size": self.snapshot_thumbnail_size,
            "snapshot_columns": self.snapshot_columns,
            "image_cmap": self.image_cmap,
            "pixel_size": self.pixel_size,
            "pixel_unit": self.pixel_unit,
            "scale_bar_visible": self.scale_bar_visible,
            "prefer_webgpu": self.prefer_webgpu,
            "snapshot_playing": self.snapshot_playing,
            "snapshot_fps": self.snapshot_fps,
            "profile_line": list(self.profile_line),
            "profile_width": self.profile_width,
            "report_metadata": dict(self.report_metadata),
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
                setattr(self, key, value)

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

    def _load_joint_time_snapshots(self, path: pathlib.Path, methods: Sequence[str]) -> None:
        arrays = np.load(path)
        if "reference_phase" in arrays:
            self.snapshot(-1.0, arrays["reference_phase"], label="clean reference")
        for idx, method in enumerate(methods):
            if method not in arrays:
                continue
            arr = np.asarray(arrays[method], dtype=np.float32)
            if arr.ndim == 3:
                image = arr.mean(axis=0)
                label = f"{method} average"
            elif arr.ndim == 2:
                image = arr
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
            show_legend=self.show_legend,
            show_grid=self.show_grid,
            show_controls=self.show_controls,
            controls_collapsed=self.controls_collapsed,
            line_width=self.line_width,
            plot_height_px=self.plot_height_px,
            side_panel_width_px=self.side_panel_width_px,
            image_cmap=self.image_cmap,
            snapshot_contrast_preset=self.snapshot_contrast_preset,
            snapshot_thumbnail_size=self.snapshot_thumbnail_size,
            snapshot_columns=self.snapshot_columns,
            pixel_size=self.pixel_size,
            pixel_unit=self.pixel_unit,
            show_scale_bar=self.scale_bar_visible,
            show_snapshot_histogram=self.show_snapshot_histogram,
            show_snapshot_fft=self.show_snapshot_fft,
            snapshot_fft_window=self.snapshot_fft_window,
            snapshot_fft_cmap=self.snapshot_fft_cmap,
            prefer_webgpu=self.prefer_webgpu,
        )
        clone.load_state_dict(self.state_dict())
        clone.method_labels = list(self.method_labels)
        clone.report_metadata = dict(self.report_metadata)
        clone._snapshots = [snap.copy() for snap in self._snapshots]
        clone.snapshot_iterations = list(self.snapshot_iterations)
        clone.snapshot_labels = list(self.snapshot_labels)
        clone.snapshot_image_labels = list(self.snapshot_image_labels)
        clone.snapshot_group_indices = list(self.snapshot_group_indices)
        clone.snapshot_group_iterations = list(self.snapshot_group_iterations)
        clone.snapshot_group_labels = list(self.snapshot_group_labels)
        clone.selected_snapshot_group_idx = self.selected_snapshot_group_idx
        clone.show_snapshot_thumbnails = self.show_snapshot_thumbnails
        clone.show_snapshot_fft = self.show_snapshot_fft
        clone.snapshot_fft_window = self.snapshot_fft_window
        clone.snapshot_fft_cmap = self.snapshot_fft_cmap
        clone.snapshot_contrast_preset = self.snapshot_contrast_preset
        clone.snapshot_thumbnail_size = self.snapshot_thumbnail_size
        clone.snapshot_columns = self.snapshot_columns
        clone.pixel_size = self.pixel_size
        clone.pixel_unit = self.pixel_unit
        clone.scale_bar_visible = self.scale_bar_visible
        clone.snapshot_playing = self.snapshot_playing
        clone.snapshot_fps = self.snapshot_fps
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
        return clone

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
