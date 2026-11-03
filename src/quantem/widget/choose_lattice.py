"""ChooseLattice: pick an origin and two lattice-vector points on an image.

A focused sibling of Show2D: one 2D image, wheel-zoom and drag-pan, and three
ordered clicks (origin, then the two lattice points) whose pixel coordinates in
the original un-zoomed image feed downstream lattice calculations as the
vectors ``u`` and ``v``.
"""

import io
import pathlib

import anywidget
import numpy as np
import traitlets
from PIL import Image

from quantem.widget.adapters import core as core_adapter
from quantem.widget.colormap import colorize
from quantem.widget.utils.array import to_numpy


class ChooseLattice(anywidget.AnyWidget):
    """Interactive picker for an ordered origin plus two lattice-vector points.

    Parameters
    ----------
    data : array_like or Dataset2d
        A single 2D image (NumPy, PyTorch or CuPy); a ``Dataset2d`` supplies
        its ``name`` as the default title.
    cmap : str, default "gray"
        Colormap used to render the image.
    title : str, default ""
        Title shown above the image.

    Notes
    -----
    Click on the image to place points in order; once 3 are placed, drag a
    point to adjust it, or press "Clear Points" to start over. Coordinates are
    always reported in the original image's ``(row, col)`` space regardless of
    zoom. The display range is the robust 1st to 99th percentile of the image.
    """

    _esm = pathlib.Path(__file__).parent / "static" / "chooselattice.js"

    height = traitlets.Int(1).tag(sync=True)
    width = traitlets.Int(1).tag(sync=True)
    frame_bytes = traitlets.Bytes(b"").tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    points = traitlets.List(trait=traitlets.List(traitlets.Float()), default_value=[]).tag(sync=True)

    def __init__(self, data, *, cmap: str = "gray", title: str = "") -> None:
        super().__init__()
        if core_adapter.is_dataset(data, ndim=2):
            title = title or data.name
            data = core_adapter.as_array(data)
        frame = to_numpy(data, dtype=np.float32)
        if frame.ndim != 2:
            raise ValueError(f"ChooseLattice expects a single 2D image, got array with shape {frame.shape!r}.")
        # percentile contrast so a few hot pixels do not black out the lattice; percentiles
        # and normalisation in float64 rather than at the float32 frame's precision
        values = frame.astype(np.float64, copy=False)
        low = float(np.nanpercentile(values, 1))
        high = float(np.nanpercentile(values, 99))
        if high <= low:
            high = low + 1.0
        rgb = colorize(np.clip((values - low) / (high - low), 0.0, 1.0), cmap)
        png = io.BytesIO()
        Image.fromarray(rgb, mode="RGB").save(png, format="PNG")
        with self.hold_sync():
            self.height = int(frame.shape[0])
            self.width = int(frame.shape[1])
            self.frame_bytes = png.getvalue()
            self.title = str(title)

    @traitlets.validate("points")
    def _validate_points(self, proposal):
        """At most three points, each clamped inside the image."""
        points = list(proposal["value"])
        if len(points) > 3:
            raise traitlets.TraitError(f"ChooseLattice supports at most 3 points, got {len(points)}.")
        return [
            [float(np.clip(row, 0, max(0, self.height - 1))), float(np.clip(col, 0, max(0, self.width - 1)))]
            for row, col in points
        ]

    @property
    def origin(self) -> tuple[float, float] | None:
        """First picked point ``(row, col)``, or None if not yet placed."""
        return tuple(self.points[0]) if self.points else None

    @property
    def u(self) -> tuple[float, float] | None:
        """Lattice vector from the origin to the second point, or None until both are placed."""
        if len(self.points) < 2:
            return None
        return (self.points[1][0] - self.points[0][0], self.points[1][1] - self.points[0][1])

    @property
    def v(self) -> tuple[float, float] | None:
        """Lattice vector from the origin to the third point, or None until both are placed."""
        if len(self.points) < 3:
            return None
        return (self.points[2][0] - self.points[0][0], self.points[2][1] - self.points[0][1])
