"""Calibrated scalar maps, distinct from spatial image viewers."""

import base64
import io
import math
from pathlib import Path

import anywidget
import matplotlib.pyplot as plt
import numpy as np
import traitlets
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from quantem.widget.colormap import colormap_lut, colormap_names
from quantem.widget.fallback import StaticFallbackMixin
from quantem.widget.state import SavedStateMixin
from quantem.widget.utils.array import _b64_safe, to_numpy


class Plot2D(SavedStateMixin, StaticFallbackMixin, anywidget.AnyWidget):
    """Inspect a scalar map with physical axes and a labeled color scale.

    Rows correspond to ``y`` and columns to ``x``. Both coordinates are bin
    centers on increasing uniform grids. The first row is shown at the bottom,
    as in Matplotlib's Cartesian plots. Source values and hover transport are
    float64; QuantEM's shared WebGPU colormap renderer uses float32 display data.
    Pan and zoom change only the viewport, never the scientific array.

    Parameters
    ----------
    data : numpy.ndarray, torch.Tensor or quantem Dataset2d
        Finite two-dimensional scalar values. A tensor may live on any device.
    x, y : numpy.ndarray
        Physical column and row bin centers, respectively.
    x_label, y_label, colorbar_label : str
        Scientific quantities including units where applicable.
    title : str
        Plot title.
    vmin, vmax : float or None
        Fixed color limits. When omitted, use the initial data range.
    width : int
        Initial plot width in CSS pixels.
    max_width : int
        Maximum width in CSS pixels, also bounded by the notebook container.
    save_state : bool, default False
        Embed the full float64 map in saved widget state when True. By default,
        save a static PNG preview and omit the array from full state snapshots.
        Targeted live updates always retain the original values.

    Examples
    --------
    >>> plot = Plot2D(g3, x=radii, y=angles, x_label="r02 (Å)",
    ...               y_label="Angle (°)", colorbar_label="Normalized G3")
    >>> plot.set_data(next_g3)
    >>> plot.figure().savefig("g3.png", dpi=180)
    """

    _esm = Path(__file__).parent / "static" / "plot2d.js"
    _static_fallback_jpeg = traitlets.Unicode("").tag(sync=True)
    _static_fallback_mime = traitlets.Unicode("image/png").tag(sync=True)
    _UNSAVED_HEAVY_KEYS = ("data_bytes",)
    data_bytes = traitlets.Bytes().tag(sync=True)
    grid = traitlets.Dict().tag(sync=True)
    title = traitlets.Unicode().tag(sync=True)
    x_label = traitlets.Unicode().tag(sync=True)
    y_label = traitlets.Unicode().tag(sync=True)
    colorbar_label = traitlets.Unicode().tag(sync=True)
    cmap = traitlets.Unicode("viridis").tag(sync=True)
    vmin = traitlets.Float(0).tag(sync=True)
    vmax = traitlets.Float(1).tag(sync=True)
    horizontal_line = traitlets.Float(None, allow_none=True).tag(sync=True)
    view_bounds = traitlets.List(traitlets.Float()).tag(sync=True)
    plot_height_px = traitlets.Int(350).tag(sync=True)
    max_width = traitlets.Int(600, min=260).tag(sync=True)

    data: np.ndarray | None = None

    def __init__(
        self,
        data,
        *,
        x: np.ndarray,
        y: np.ndarray,
        x_label: str = "",
        y_label: str = "",
        colorbar_label: str = "Value",
        title: str = "",
        vmin: float | None = None,
        vmax: float | None = None,
        width: int = 350,
        max_width: int = 600,
        save_state: bool = False,
    ) -> None:
        self._save_state = bool(save_state)
        self._configure_static_fallback(notebook_preview_format="png")
        super().__init__()
        self.x, self.y = np.array(x, dtype=float, copy=True), np.array(y, dtype=float, copy=True)
        edges = []
        for name, centers in (("x", self.x), ("y", self.y)):
            if centers.ndim != 1 or len(centers) < 2 or not np.isfinite(centers).all():
                raise ValueError(f"{name} needs at least two finite bin centers.")
            steps = np.diff(centers)
            if not (steps > 0).all() or not np.allclose(steps, steps[0], rtol=1e-5, atol=0.0):
                raise ValueError(
                    f"{name} must be an increasing uniform grid; got bin spacings "
                    f"from {steps.min():g} to {steps.max():g}. "
                    "Supply uniformly spaced bin centers in consistent units."
                )
            edges.extend([float(centers[0] - steps[0] / 2), float(centers[-1] + steps[0] / 2)])
        self.grid = {"rows": len(self.y), "cols": len(self.x), "bounds": edges}
        self.title, self.x_label, self.y_label, self.colorbar_label = title, x_label, y_label, colorbar_label
        self.max_width = max_width
        self.layout.width = f"{min(max_width, max(260, width))}px"
        self.layout.max_width = f"min(100%, {max_width}px)"
        self.set_data(data)
        self.vmin = float(self.data.min()) if vmin is None else float(vmin)
        self.vmax = float(self.data.max()) if vmax is None else float(vmax)
        if self.vmax == self.vmin and vmin is None and vmax is None:
            self.vmax = self.vmin + 1
        if not np.isfinite([self.vmin, self.vmax]).all() or self.vmax <= self.vmin:
            raise ValueError("Use finite color limits with vmax > vmin.")
        self.observe(self._invalidate_preview, names=["cmap", "vmin", "vmax", "title", "view_bounds", "horizontal_line"])

    def _invalidate_preview(self, change: dict | None = None) -> None:
        """Drop the cached preview so the next notebook save re-renders the current view."""
        self._static_fallback_jpeg = ""

    @traitlets.validate("cmap")
    def _validate_cmap(self, proposal: dict) -> str:
        """Refuse a colormap the shared renderer has no lookup table for."""
        value = proposal["value"]
        if value not in colormap_names():
            raise ValueError(f"Unknown colormap {value!r}; choose from {colormap_names()}.")
        return value

    def set_data(self, data) -> None:
        """Replace values on the existing grid, preserving limits and viewport.

        Parameters
        ----------
        data : numpy.ndarray, torch.Tensor or quantem Dataset2d
            Finite values with the original row/column shape.

        Examples
        --------
        >>> plot.set_data(predicted_g3[1])
        """
        data = to_numpy(data)
        if np.iscomplexobj(data):
            raise ValueError("Choose a real scalar quantity explicitly before plotting.")
        values = np.asarray(data, dtype=np.float64)
        if values.shape != (len(self.y), len(self.x)) or not np.isfinite(values).all():
            raise ValueError(f"Supply finite data shaped {(len(self.y), len(self.x))}; got {values.shape}.")
        self.data = values.copy()
        self.data_bytes = _b64_safe(np.ascontiguousarray(self.data).tobytes())
        self._invalidate_preview()

    def figure(self) -> Figure:
        """Return a closed Matplotlib figure of the current values and viewport.

        Returns
        -------
        matplotlib.figure.Figure
            Editable figure with physical axes and a labeled colorbar.

        Examples
        --------
        >>> plot.figure().savefig("correlation.svg")
        """
        return self._make_figure()

    def _static_png_b64(self, max_px: int = 512) -> str | None:
        """Render a bounded saved preview without modifying scientific data."""
        if self.data is None:
            return None
        buffer = io.BytesIO()
        self._make_figure(max_bins=max_px).savefig(buffer, format="png", dpi=100)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _make_figure(self, max_bins: int | None = None) -> Figure:
        """Draw exact data for export or a strided, bounded notebook preview."""
        size = (5, 4) if max_bins is None else (max_bins / 100, max_bins * 0.8 / 100)
        figure, axes = plt.subplots(figsize=size, layout="constrained")
        values = self.data
        x, y = self.x, self.y
        shading = "nearest"
        if max_bins is not None:
            rows, cols = values.shape
            row_step = max(1, math.ceil(rows / max_bins))
            col_step = max(1, math.ceil(cols / max_bins))
            # Explicit edges preserve calibration, including the last partial bin.
            col_edges = np.r_[np.arange(0, cols, col_step), cols]
            row_edges = np.r_[np.arange(0, rows, row_step), rows]
            left, right, bottom, top = self.grid["bounds"]
            x = left + col_edges * (right - left) / cols
            y = bottom + row_edges * (top - bottom) / rows
            values = values[::row_step, ::col_step]
            shading = "flat"
        mesh = axes.pcolormesh(
            x, y, values, shading=shading,
            cmap=ListedColormap(colormap_lut(self.cmap), name=self.cmap),
            vmin=self.vmin, vmax=self.vmax,
        )
        limits = self.view_bounds or self.grid["bounds"]
        axes.set(xlabel=self.x_label, ylabel=self.y_label, title=self.title, xlim=limits[:2], ylim=limits[2:])
        if self.horizontal_line is not None:
            axes.axhline(self.horizontal_line, color="#e649a0", linestyle=":")
        figure.colorbar(mesh, ax=axes, label=self.colorbar_label)
        plt.close(figure)
        return figure
