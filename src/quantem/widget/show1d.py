"""
show1d: Interactive 1D data viewer for spectra, profiles, and time series.

The 1D counterpart to Show2D. Displays single or multiple 1D traces with
interactive zoom/pan, cursor readout, and calibrated axes. Common uses:
line profiles, ROI time series, optimization loss curves, EELS/EDX spectra,
convergence plots, and any 1D signal.
"""

import csv
import json
import pathlib
from typing import Self

import anywidget
import numpy as np
import traitlets

from quantem.widget.array_utils import to_numpy
from quantem.widget.json_state import resolve_widget_version, save_state_file, unwrap_state_payload
from quantem.widget.tool_parity import (
    bind_tool_runtime_api,
    build_tool_groups,
    normalize_tool_groups,
)

_DEFAULT_COLORS = [
    "#4fc3f7",  # light blue
    "#81c784",  # green
    "#ffb74d",  # orange
    "#ce93d8",  # purple
    "#ef5350",  # red
    "#ffd54f",  # yellow
    "#90a4ae",  # blue-gray
    "#a1887f",  # brown
]


class Show1D(anywidget.AnyWidget):
    """
    Interactive 1D data viewer for spectra, profiles, and time series.

    Display one or more 1D traces with interactive zoom, pan, cursor readout,
    and calibrated axes. Supports log scale, grid lines, legend, and
    publication-quality export.

    Parameters
    ----------
    data : array_like
        1D array for a single trace, 2D array (n_traces, n_points) for
        multiple traces, or a list of 1D arrays.
    x : array_like, optional
        Shared X-axis values. Must have the same length as data points.
        If not provided, uses integer indices [0, 1, 2, ...].
    labels : list of str, optional
        Labels for each trace. Used in legend and stats bar.
    colors : list of str, optional
        Hex color strings for each trace. If not provided, uses a default
        8-color palette.
    title : str, optional
        Title displayed above the plot.
    x_label : str, optional
        Label for the X axis (e.g. "Energy", "Distance", "Epoch").
    y_label : str, optional
        Label for the Y axis (e.g. "Counts", "Intensity", "Loss").
    x_unit : str, optional
        Unit for the X axis (e.g. "eV", "nm", "").
    y_unit : str, optional
        Unit for the Y axis.
    log_scale : bool, default False
        Use logarithmic Y axis.
    auto_contrast : bool, default False
        Clip Y-axis range to ``[percentile_low, percentile_high]`` of the
        data, revealing weak features hidden by outliers (e.g. core-loss
        edges behind a zero-loss peak in EELS).
    percentile_low : float, default 2.0
        Lower percentile for auto-contrast clipping (0–100).
    percentile_high : float, default 98.0
        Upper percentile for auto-contrast clipping (0–100).
    show_stats : bool, default True
        Show statistics bar (mean, min, max, std per trace).
    show_legend : bool, default True
        Show legend when multiple traces are displayed.
    show_grid : bool, default True
        Show grid lines on the plot.
    show_controls : bool, default True
        Show control row below the plot.
    line_width : float, default 1.5
        Line width for traces.
    grid_density : int, default 10
        Number of grid divisions per axis (5–50).
    peak_active : bool, default False
        Whether peak placement mode is on.
    disabled_tools : list of str, optional
        Tool groups to lock (visible but non-interactive):
        ``"display"``, ``"peaks"``, ``"stats"``, ``"export"``, ``"all"``.
    disable_* : bool, optional
        Convenience flags (``disable_display``, ``disable_peaks``,
        ``disable_stats``, ``disable_export``, ``disable_all``) equivalent
        to adding those keys to ``disabled_tools``.
    hidden_tools : list of str, optional
        Tool groups to hide (not rendered). Same values as
        ``disabled_tools``.
    hide_* : bool, optional
        Convenience flags mirroring ``disable_*`` for ``hidden_tools``.
    state : dict or str or Path, optional
        Restore display settings from a dict or a JSON file path.

    Examples
    --------
    >>> import numpy as np
    >>> from quantem.widget import Show1D
    >>>
    >>> # Single trace
    >>> Show1D(np.sin(np.linspace(0, 10, 200)))
    >>>
    >>> # Multiple traces with labels
    >>> x = np.linspace(0, 10, 200)
    >>> Show1D([np.sin(x), np.cos(x)], x=x, labels=["sin", "cos"])
    >>>
    >>> # Optimization loss curve
    >>> Show1D(losses, x_label="Epoch", y_label="Loss", log_scale=True)
    """

    _esm = pathlib.Path(__file__).parent / "static" / "show1d.js"
    _css = pathlib.Path(__file__).parent / "static" / "show1d.css"

    # =========================================================================
    # Core State
    # =========================================================================
    widget_version = traitlets.Unicode("unknown").tag(sync=True)
    y_bytes = traitlets.Bytes(b"").tag(sync=True)
    x_bytes = traitlets.Bytes(b"").tag(sync=True)
    n_traces = traitlets.Int(1).tag(sync=True)
    n_points = traitlets.Int(0).tag(sync=True)
    labels = traitlets.List(traitlets.Unicode()).tag(sync=True)
    colors = traitlets.List(traitlets.Unicode()).tag(sync=True)

    # =========================================================================
    # Display Options
    # =========================================================================
    title = traitlets.Unicode("").tag(sync=True)
    x_label = traitlets.Unicode("").tag(sync=True)
    y_label = traitlets.Unicode("").tag(sync=True)
    x_unit = traitlets.Unicode("").tag(sync=True)
    y_unit = traitlets.Unicode("").tag(sync=True)
    log_scale = traitlets.Bool(False).tag(sync=True)
    auto_contrast = traitlets.Bool(False).tag(sync=True)
    percentile_low = traitlets.Float(2.0).tag(sync=True)
    percentile_high = traitlets.Float(98.0).tag(sync=True)
    show_stats = traitlets.Bool(True).tag(sync=True)
    show_legend = traitlets.Bool(True).tag(sync=True)
    show_grid = traitlets.Bool(True).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)
    line_width = traitlets.Float(1.5).tag(sync=True)
    focused_trace = traitlets.Int(-1).tag(sync=True)
    grid_density = traitlets.Int(10).tag(sync=True)
    x_range = traitlets.List(traitlets.Float()).tag(sync=True)
    y_range = traitlets.List(traitlets.Float()).tag(sync=True)

    # =========================================================================
    # Peak Features
    # =========================================================================
    peak_markers = traitlets.List(traitlets.Dict()).tag(sync=True)
    peak_active = traitlets.Bool(False).tag(sync=True)
    peak_search_radius = traitlets.Int(20).tag(sync=True)
    selected_peaks = traitlets.List(traitlets.Int()).tag(sync=True)

    # =========================================================================
    # Tool Lock/Hide
    # =========================================================================
    disabled_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)
    hidden_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)

    @classmethod
    def _normalize_tool_groups(cls, tool_groups):
        return normalize_tool_groups("Show1D", tool_groups)

    @classmethod
    def _build_disabled_tools(
        cls,
        disabled_tools=None,
        disable_display: bool = False,
        disable_peaks: bool = False,
        disable_stats: bool = False,
        disable_export: bool = False,
        disable_all: bool = False,
    ):
        return build_tool_groups(
            "Show1D",
            tool_groups=disabled_tools,
            all_flag=disable_all,
            flag_map={
                "display": disable_display,
                "peaks": disable_peaks,
                "stats": disable_stats,
                "export": disable_export,
            },
        )

    @classmethod
    def _build_hidden_tools(
        cls,
        hidden_tools=None,
        hide_display: bool = False,
        hide_peaks: bool = False,
        hide_stats: bool = False,
        hide_export: bool = False,
        hide_all: bool = False,
    ):
        return build_tool_groups(
            "Show1D",
            tool_groups=hidden_tools,
            all_flag=hide_all,
            flag_map={
                "display": hide_display,
                "peaks": hide_peaks,
                "stats": hide_stats,
                "export": hide_export,
            },
        )

    @traitlets.validate("disabled_tools")
    def _validate_disabled_tools(self, proposal):
        return self._normalize_tool_groups(proposal["value"])

    @traitlets.validate("hidden_tools")
    def _validate_hidden_tools(self, proposal):
        return self._normalize_tool_groups(proposal["value"])

    # =========================================================================
    # Statistics (per-trace)
    # =========================================================================
    stats_mean = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_min = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_max = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_std = traitlets.List(traitlets.Float()).tag(sync=True)
    range_stats = traitlets.List(traitlets.Dict()).tag(sync=True)

    # =========================================================================
    # Peak FWHM
    # =========================================================================
    peak_fwhm = traitlets.List(traitlets.Dict()).tag(sync=True)

    def __init__(
        self,
        data,
        x=None,
        labels=None,
        colors=None,
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        x_unit: str = "",
        y_unit: str = "",
        log_scale: bool = False,
        auto_contrast: bool = False,
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
        show_stats: bool = True,
        show_legend: bool = True,
        show_grid: bool = True,
        show_controls: bool = True,
        line_width: float = 1.5,
        grid_density: int = 10,
        peak_active: bool = False,
        peak_search_radius: int = 20,
        disabled_tools=None,
        disable_display: bool = False,
        disable_peaks: bool = False,
        disable_stats: bool = False,
        disable_export: bool = False,
        disable_all: bool = False,
        hidden_tools=None,
        hide_display: bool = False,
        hide_peaks: bool = False,
        hide_stats: bool = False,
        hide_export: bool = False,
        hide_all: bool = False,
        state=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.widget_version = resolve_widget_version()

        # Dataset duck typing
        _extracted_title = ""
        if hasattr(data, "array") and hasattr(data, "name"):
            if data.name:
                _extracted_title = data.name
            data = data.array

        # Normalize data to 2D (n_traces, n_points)
        if isinstance(data, list):
            arrays = [to_numpy(d).astype(np.float32).ravel() for d in data]
            n_pts = len(arrays[0])
            for i, a in enumerate(arrays):
                if len(a) != n_pts:
                    raise ValueError(
                        f"All traces must have the same length. "
                        f"Trace 0 has {n_pts} points, trace {i} has {len(a)}."
                    )
            data_2d = np.stack(arrays)
        else:
            arr = to_numpy(data).astype(np.float32)
            if arr.ndim == 1:
                data_2d = arr[np.newaxis, :]
            elif arr.ndim == 2:
                data_2d = arr
            else:
                raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D.")

        self._data = data_2d
        self.n_traces = int(data_2d.shape[0])
        self.n_points = int(data_2d.shape[1])

        # X axis
        self._x = None
        if x is not None:
            x_arr = to_numpy(x).astype(np.float32).ravel()
            if len(x_arr) != self.n_points:
                raise ValueError(
                    f"x has {len(x_arr)} points but data has {self.n_points} points."
                )
            self._x = x_arr
            self.x_bytes = x_arr.tobytes()

        # Labels
        if labels is not None:
            self.labels = [str(l) for l in labels]
        else:
            if self.n_traces == 1:
                self.labels = ["Data"]
            else:
                self.labels = [f"Data {i + 1}" for i in range(self.n_traces)]

        # Colors
        if colors is not None:
            self.colors = [str(c) for c in colors]
        else:
            self.colors = [_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)] for i in range(self.n_traces)]

        # Display options
        self.title = title or _extracted_title
        self.x_label = x_label
        self.y_label = y_label
        self.x_unit = x_unit
        self.y_unit = y_unit
        self.log_scale = log_scale
        self.auto_contrast = auto_contrast
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.show_stats = show_stats
        self.show_legend = show_legend
        self.show_grid = show_grid
        self.show_controls = show_controls
        self.line_width = line_width
        self.grid_density = grid_density
        self.peak_active = peak_active
        self.peak_search_radius = peak_search_radius
        self.disabled_tools = self._build_disabled_tools(
            disabled_tools=disabled_tools,
            disable_display=disable_display,
            disable_peaks=disable_peaks,
            disable_stats=disable_stats,
            disable_export=disable_export,
            disable_all=disable_all,
        )
        self.hidden_tools = self._build_hidden_tools(
            hidden_tools=hidden_tools,
            hide_display=hide_display,
            hide_peaks=hide_peaks,
            hide_stats=hide_stats,
            hide_export=hide_export,
            hide_all=hide_all,
        )

        # Compute stats and send data
        self._compute_stats()
        self._compute_range_stats()
        self.y_bytes = self._data.tobytes()

        # Restore state
        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(
                    json.loads(pathlib.Path(state).read_text()),
                    require_envelope=True,
                )
            else:
                state = unwrap_state_payload(state)
            self.load_state_dict(state)

    def set_data(self, data, x=None, labels=None) -> Self:
        """Replace displayed data. Preserves display settings.

        Parameters
        ----------
        data : array_like
            1D, 2D (n_traces, n_points), or list of 1D arrays.
        x : array_like, optional
            New X-axis values.
        labels : list of str, optional
            New trace labels. If not provided, generates defaults.
        """
        if hasattr(data, "array") and hasattr(data, "name"):
            data = data.array

        if isinstance(data, list):
            arrays = [to_numpy(d).astype(np.float32).ravel() for d in data]
            n_pts = len(arrays[0])
            for i, a in enumerate(arrays):
                if len(a) != n_pts:
                    raise ValueError(
                        f"All traces must have the same length. "
                        f"Trace 0 has {n_pts} points, trace {i} has {len(a)}."
                    )
            data_2d = np.stack(arrays)
        else:
            arr = to_numpy(data).astype(np.float32)
            if arr.ndim == 1:
                data_2d = arr[np.newaxis, :]
            elif arr.ndim == 2:
                data_2d = arr
            else:
                raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D.")

        self._data = data_2d
        self.n_traces = int(data_2d.shape[0])
        self.n_points = int(data_2d.shape[1])

        if x is not None:
            x_arr = to_numpy(x).astype(np.float32).ravel()
            if len(x_arr) != self.n_points:
                raise ValueError(
                    f"x has {len(x_arr)} points but data has {self.n_points} points."
                )
            self._x = x_arr
            self.x_bytes = x_arr.tobytes()
        else:
            self._x = None
            self.x_bytes = b""

        if labels is not None:
            self.labels = [str(l) for l in labels]
        else:
            if self.n_traces == 1:
                self.labels = ["Data"]
            else:
                self.labels = [f"Data {i + 1}" for i in range(self.n_traces)]

        # Assign default colors if count changed
        self.colors = [_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)] for i in range(self.n_traces)]

        self._compute_stats()
        self._compute_range_stats()
        self.peak_fwhm = []
        self.y_bytes = self._data.tobytes()
        return self

    def add_trace(self, y, label=None, color=None) -> Self:
        """Append a trace.

        Parameters
        ----------
        y : array_like
            1D array with the same number of points as existing traces.
        label : str, optional
            Trace label.
        color : str, optional
            Hex color string.
        """
        arr = to_numpy(y).astype(np.float32).ravel()
        if self.n_points > 0 and len(arr) != self.n_points:
            raise ValueError(
                f"New trace has {len(arr)} points but existing traces have {self.n_points}."
            )

        if self._data.size == 0:
            self._data = arr[np.newaxis, :]
            self.n_points = int(len(arr))
        else:
            self._data = np.vstack([self._data, arr[np.newaxis, :]])

        self.n_traces = int(self._data.shape[0])

        lbl = label if label is not None else f"Data {self.n_traces}"
        self.labels = list(self.labels) + [lbl]

        clr = color if color is not None else _DEFAULT_COLORS[(self.n_traces - 1) % len(_DEFAULT_COLORS)]
        self.colors = list(self.colors) + [clr]

        self._compute_stats()
        self._compute_range_stats()
        self.y_bytes = self._data.tobytes()
        return self

    def remove_trace(self, index: int) -> Self:
        """Remove a trace by index.

        Parameters
        ----------
        index : int
            Zero-based trace index.
        """
        if index < 0 or index >= self.n_traces:
            raise IndexError(f"Trace index {index} out of range [0, {self.n_traces}).")
        self._data = np.delete(self._data, index, axis=0)
        self.n_traces = int(self._data.shape[0])
        lbls = list(self.labels)
        lbls.pop(index)
        self.labels = lbls
        clrs = list(self.colors)
        clrs.pop(index)
        self.colors = clrs
        self._compute_stats()
        self._compute_range_stats()
        self.y_bytes = self._data.tobytes()
        return self

    def clear(self) -> Self:
        """Remove all traces."""
        self._data = np.empty((0, 0), dtype=np.float32)
        self.n_traces = 0
        self.n_points = 0
        self.labels = []
        self.colors = []
        self.stats_mean = []
        self.stats_min = []
        self.stats_max = []
        self.stats_std = []
        self.range_stats = []
        self.peak_fwhm = []
        self.y_bytes = b""
        return self

    # =========================================================================
    # Peak Markers
    # =========================================================================
    @property
    def peaks(self):
        """Return list of peak marker dicts."""
        return list(self.peak_markers)

    @property
    def selected_peak_data(self):
        """Return peak marker dicts for currently selected peaks."""
        markers = list(self.peak_markers)
        return [markers[i] for i in self.selected_peaks if 0 <= i < len(markers)]

    def add_peak(self, x: float, trace_idx: int = 0, label: str = "") -> Self:
        """Add a peak marker at the given X position.

        Finds the nearest data point and searches ±5 points for a local
        maximum.

        Parameters
        ----------
        x : float
            X-axis value near the desired peak.
        trace_idx : int
            Trace index to search for the peak.
        label : str
            Optional label for the peak marker.
        """
        if trace_idx < 0 or trace_idx >= self.n_traces:
            raise IndexError(f"Trace index {trace_idx} out of range [0, {self.n_traces}).")

        trace = self._data[trace_idx]

        # Find nearest index to x
        if self._x is not None:
            nearest_idx = int(np.argmin(np.abs(self._x - x)))
        else:
            nearest_idx = int(round(x))
            nearest_idx = max(0, min(self.n_points - 1, nearest_idx))

        # Search ±5 points for local maximum
        lo = max(0, nearest_idx - 5)
        hi = min(self.n_points, nearest_idx + 6)
        region = trace[lo:hi]
        local_idx = lo + int(np.argmax(region))

        peak_x = float(self._x[local_idx]) if self._x is not None else float(local_idx)
        peak_y = float(trace[local_idx])

        marker = {
            "x": peak_x,
            "y": peak_y,
            "trace_idx": trace_idx,
            "label": label or f"{peak_x:.4g}",
            "type": "peak",
        }
        self.peak_markers = list(self.peak_markers) + [marker]
        return self

    def remove_peak(self, index: int = -1) -> Self:
        """Remove a peak marker by index (default: last)."""
        markers = list(self.peak_markers)
        if not markers:
            return self
        if index < -len(markers) or index >= len(markers):
            raise IndexError(f"Peak index {index} out of range.")
        markers.pop(index)
        self.peak_markers = markers
        return self

    def clear_peaks(self) -> Self:
        """Remove all peak markers."""
        self.peak_markers = []
        self.selected_peaks = []
        return self

    def find_peaks(
        self,
        trace_idx: int = 0,
        *,
        height=None,
        prominence: float = 0.01,
        distance: int = 1,
        width=None,
    ) -> Self:
        """Auto-detect peaks using scipy.signal.find_peaks.

        Parameters
        ----------
        trace_idx : int
            Trace index to search.
        height : float, optional
            Minimum peak height.
        prominence : float
            Minimum peak prominence (default 0.01).
        distance : int
            Minimum horizontal distance between peaks in samples.
        width : float, optional
            Minimum peak width in samples.
        """
        from scipy.signal import find_peaks as _find_peaks

        if trace_idx < 0 or trace_idx >= self.n_traces:
            raise IndexError(f"Trace index {trace_idx} out of range [0, {self.n_traces}).")

        trace = self._data[trace_idx]
        kwargs = {"distance": distance}
        if height is not None:
            kwargs["height"] = height
        if prominence is not None:
            kwargs["prominence"] = prominence
        if width is not None:
            kwargs["width"] = width

        indices, _ = _find_peaks(trace, **kwargs)

        markers = list(self.peak_markers)
        for idx in indices:
            peak_x = float(self._x[idx]) if self._x is not None else float(idx)
            peak_y = float(trace[idx])
            markers.append({
                "x": peak_x,
                "y": peak_y,
                "trace_idx": trace_idx,
                "label": f"{peak_x:.4g}",
                "type": "peak",
            })
        self.peak_markers = markers
        return self

    def export_peaks(self, path: str) -> pathlib.Path:
        """Export peak markers to CSV or JSON.

        Format is inferred from the file extension (.csv or .json).

        Parameters
        ----------
        path : str
            Output file path.
        """
        p = pathlib.Path(path)
        markers = list(self.peak_markers)
        if not markers:
            raise ValueError("No peak markers to export.")

        if p.suffix.lower() == ".csv":
            with open(p, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["x", "y", "trace_idx", "label", "type"])
                writer.writeheader()
                for m in markers:
                    writer.writerow({
                        "x": m["x"],
                        "y": m["y"],
                        "trace_idx": m["trace_idx"],
                        "label": m.get("label", ""),
                        "type": m.get("type", "peak"),
                    })
        elif p.suffix.lower() == ".json":
            p.write_text(json.dumps(markers, indent=2))
        else:
            raise ValueError(f"Unsupported format '{p.suffix}'. Use .csv or .json.")

        return p

    def save_image(
        self,
        path: str | pathlib.Path,
        *,
        format: str | None = None,
        dpi: int = 150,
        include_peaks: bool = True,
    ) -> pathlib.Path:
        """Save publication-quality figure as PNG or PDF.

        Renders traces, grid, axes, labels, and peak markers using matplotlib.

        Parameters
        ----------
        path : str or pathlib.Path
            Output file path.
        format : str, optional
            ``"png"`` or ``"pdf"``. Inferred from extension if omitted.
        dpi : int
            Output resolution (default 150).
        include_peaks : bool
            Render peak markers on the figure (default True).

        Returns
        -------
        pathlib.Path
            The written file path.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path = pathlib.Path(path)
        fmt = (format or path.suffix.lstrip(".").lower() or "png").lower()
        if fmt not in ("png", "pdf"):
            raise ValueError(f"Unsupported format: {fmt!r}. Use 'png' or 'pdf'.")

        data = self._data
        x = self._x if self._x is not None else np.arange(self.n_points, dtype=np.float32)

        fig, ax = plt.subplots(figsize=(6, 3.5), dpi=dpi)

        # Draw traces
        for t in range(self.n_traces):
            color = self.colors[t] if t < len(self.colors) else None
            label = self.labels[t] if t < len(self.labels) else f"Data {t + 1}"
            ax.plot(x, data[t], color=color, label=label, linewidth=self.line_width)

        # Peak markers
        if include_peaks and self.peak_markers:
            for pk in self.peak_markers:
                color = self.colors[pk["trace_idx"]] if pk["trace_idx"] < len(self.colors) else "C0"
                ax.plot(pk["x"], pk["y"], marker="^", color=color, markersize=6,
                        markeredgecolor="white", markeredgewidth=0.8, zorder=5)
                ax.annotate(
                    pk.get("label", ""),
                    (pk["x"], pk["y"]),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#333",
                )

        # Axis labels
        x_text = self.x_label or ""
        if self.x_unit:
            x_text += f" ({self.x_unit})" if x_text else self.x_unit
        if x_text:
            ax.set_xlabel(x_text)

        y_text = self.y_label or ""
        if self.y_unit:
            y_text += f" ({self.y_unit})" if y_text else self.y_unit
        if y_text:
            ax.set_ylabel(y_text)

        if self.title:
            ax.set_title(self.title, fontsize=11)

        if self.log_scale:
            ax.set_yscale("log")

        if self.auto_contrast and not self.log_scale:
            all_vals = self._data.ravel()
            vmin = float(np.percentile(all_vals, self.percentile_low))
            vmax = float(np.percentile(all_vals, self.percentile_high))
            if vmin < vmax:
                pad = (vmax - vmin) * 0.05
                ax.set_ylim(vmin - pad, vmax + pad)

        if self.show_grid:
            ax.grid(True, alpha=0.3, linestyle="--")

        if self.n_traces > 1 and self.show_legend:
            ax.legend(fontsize=8, framealpha=0.8)

        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), format=fmt, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def _compute_stats(self):
        means, mins, maxs, stds = [], [], [], []
        for i in range(self.n_traces):
            trace = self._data[i]
            means.append(float(np.mean(trace)))
            mins.append(float(np.min(trace)))
            maxs.append(float(np.max(trace)))
            stds.append(float(np.std(trace)))
        self.stats_mean = means
        self.stats_min = mins
        self.stats_max = maxs
        self.stats_std = stds

    def _compute_range_stats(self):
        if not self.x_range or len(self.x_range) != 2 or self.n_traces == 0:
            self.range_stats = []
            return
        x_lo, x_hi = self.x_range
        x_arr = self._x if self._x is not None else np.arange(self.n_points, dtype=np.float32)
        mask = (x_arr >= x_lo) & (x_arr <= x_hi)
        if not np.any(mask):
            self.range_stats = []
            return
        x_masked = x_arr[mask]
        results = []
        for i in range(self.n_traces):
            trace = self._data[i]
            y_masked = trace[mask]
            entry = {
                "mean": float(np.mean(y_masked)),
                "min": float(np.min(y_masked)),
                "max": float(np.max(y_masked)),
                "std": float(np.std(y_masked)),
                "integral": float(np.trapezoid(y_masked, x=x_masked)),
                "n_points": int(np.sum(mask)),
            }
            results.append(entry)
        self.range_stats = results

    # =========================================================================
    # Peak FWHM Methods
    # =========================================================================
    def _compute_peak_fwhm(self):
        if not self.selected_peaks or not self.peak_markers or self.n_traces == 0:
            self.peak_fwhm = []
            return

        x_arr = self._x if self._x is not None else np.arange(self.n_points, dtype=np.float32)
        results = []

        for peak_idx in self.selected_peaks:
            if peak_idx < 0 or peak_idx >= len(self.peak_markers):
                continue
            pk = self.peak_markers[peak_idx]
            trace_idx = pk["trace_idx"]
            if trace_idx < 0 or trace_idx >= self.n_traces:
                continue

            trace = self._data[trace_idx]
            if self._x is not None:
                center_idx = int(np.argmin(np.abs(self._x - pk["x"])))
            else:
                center_idx = int(round(pk["x"]))
                center_idx = max(0, min(self.n_points - 1, center_idx))

            radius = self.peak_search_radius
            lo = max(0, center_idx - radius)
            hi = min(self.n_points, center_idx + radius + 1)

            x_region = x_arr[lo:hi].astype(np.float64)
            y_region = trace[lo:hi].astype(np.float64)

            if len(x_region) < 4:
                results.append({"peak_idx": peak_idx, "fwhm": None, "error": "Too few points"})
                continue

            try:
                from scipy.optimize import curve_fit as _curve_fit

                A0 = float(np.max(y_region) - np.min(y_region))
                mu0 = float(x_arr[center_idx])
                sigma0 = max(float((x_region[-1] - x_region[0]) / 6), 1e-10)
                offset0 = float(np.min(y_region))

                def gaussian(x, A, mu, sigma, offset):
                    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset

                popt, _ = _curve_fit(gaussian, x_region, y_region, p0=[A0, mu0, sigma0, offset0], maxfev=5000)
                A, mu, sigma, offset = popt
                fwhm = 2.3548200450309493 * abs(sigma)

                y_pred = gaussian(x_region, *popt)
                ss_res = np.sum((y_region - y_pred) ** 2)
                ss_tot = np.sum((y_region - np.mean(y_region)) ** 2)
                r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

                results.append({
                    "peak_idx": peak_idx,
                    "fwhm": float(fwhm),
                    "center": float(mu),
                    "amplitude": float(A),
                    "sigma": float(sigma),
                    "offset": float(offset),
                    "fit_quality": r_squared,
                })
            except Exception as e:
                results.append({"peak_idx": peak_idx, "fwhm": None, "error": str(e)})

        self.peak_fwhm = results

    def measure_fwhm(self, peak_idx: int | None = None) -> Self:
        """Compute FWHM for selected peaks (or a specific peak).

        Parameters
        ----------
        peak_idx : int, optional
            If given, adds this peak to the selection before computing.
        """
        if peak_idx is not None:
            sel = list(self.selected_peaks)
            if peak_idx not in sel:
                sel.append(peak_idx)
                self.selected_peaks = sel
        self._compute_peak_fwhm()
        return self

    # =========================================================================
    # Observers
    # =========================================================================
    @traitlets.observe("x_range")
    def _on_x_range_change(self, change=None):
        self._compute_range_stats()

    @traitlets.observe("selected_peaks")
    def _on_selected_peaks_change(self, change=None):
        self._compute_peak_fwhm()

    # =========================================================================
    # State Persistence
    # =========================================================================
    def state_dict(self):
        return {
            "title": self.title,
            "labels": self.labels,
            "colors": self.colors,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "x_unit": self.x_unit,
            "y_unit": self.y_unit,
            "log_scale": self.log_scale,
            "auto_contrast": self.auto_contrast,
            "percentile_low": self.percentile_low,
            "percentile_high": self.percentile_high,
            "show_stats": self.show_stats,
            "show_legend": self.show_legend,
            "show_grid": self.show_grid,
            "show_controls": self.show_controls,
            "line_width": self.line_width,
            "focused_trace": self.focused_trace,
            "grid_density": self.grid_density,
            "x_range": list(self.x_range),
            "y_range": list(self.y_range),
            "peak_active": self.peak_active,
            "peak_search_radius": self.peak_search_radius,
            "peak_markers": list(self.peak_markers),
            "disabled_tools": self.disabled_tools,
            "hidden_tools": self.hidden_tools,
        }

    def save(self, path: str):
        save_state_file(path, "Show1D", self.state_dict())

    def load_state_dict(self, state):
        for key, val in state.items():
            if hasattr(self, key):
                setattr(self, key, val)

    def summary(self):
        lines = [self.title or "Show1D", "=" * 32]
        lines.append(f"Series:   {self.n_traces} x {self.n_points} points")
        if self.labels:
            lines.append(f"Labels:   {', '.join(self.labels)}")
        if self._x is not None:
            lines.append(f"X range:  {float(self._x[0]):.4g} - {float(self._x[-1]):.4g}")
        if self.x_label or self.x_unit:
            x_desc = self.x_label
            if self.x_unit:
                x_desc += f" ({self.x_unit})" if x_desc else self.x_unit
            lines.append(f"X axis:   {x_desc}")
        if self.y_label or self.y_unit:
            y_desc = self.y_label
            if self.y_unit:
                y_desc += f" ({self.y_unit})" if y_desc else self.y_unit
            lines.append(f"Y axis:   {y_desc}")
        for i in range(self.n_traces):
            if i < len(self.stats_mean):
                lines.append(
                    f"  [{i}] {self.labels[i] if i < len(self.labels) else ''}: "
                    f"mean={self.stats_mean[i]:.4g}  min={self.stats_min[i]:.4g}  "
                    f"max={self.stats_max[i]:.4g}  std={self.stats_std[i]:.4g}"
                )
        scale = "log" if self.log_scale else "linear"
        display = f"{scale}"
        if self.auto_contrast:
            display += f" | auto [{self.percentile_low:.1f}–{self.percentile_high:.1f}%]"
        if self.show_grid:
            display += " | grid"
            if self.grid_density != 10:
                display += f" ({self.grid_density})"
        lines.append(f"Display:  {display}")
        if self.peak_markers:
            lines.append(f"Peaks:    {len(self.peak_markers)} markers")
        if self.disabled_tools:
            lines.append(f"Locked:   {', '.join(self.disabled_tools)}")
        if self.hidden_tools:
            lines.append(f"Hidden:   {', '.join(self.hidden_tools)}")
        print("\n".join(lines))

    def __repr__(self) -> str:
        if self.n_traces == 1:
            return f"Show1D({self.n_points} points)"
        return f"Show1D({self.n_traces} traces x {self.n_points} points)"


bind_tool_runtime_api(Show1D, "Show1D")
