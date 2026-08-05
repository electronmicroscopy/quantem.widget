"""choose_lattice: pick an ordered origin + two lattice-vector points on an image.

A focused sibling of Show2D: displays a single 2D image, lets the user
wheel-zoom/drag-pan to inspect a region, and lets them click 3 ordered points
(origin, a1, a2 by default) whose pixel coordinates (in the ORIGINAL,
un-zoomed image) are exposed for downstream lattice calculations, along with
the derived lattice vectors ``u = a1 - origin`` and ``v = a2 - origin``.
"""

import base64
import pathlib
from typing import Any, Sequence

import anywidget
import matplotlib
import numpy as np
import traitlets
from quantem.widget.utils.array import to_numpy
from quantem.widget.utils.static_fallback import StaticFallbackMixin

_CORE_IMAGE_DATASET_IMPORT_ATTEMPTED = False
_CORE_IMAGE_DATASET_TYPES: tuple[type[Any], ...] = ()


def _core_image_dataset_types() -> tuple[type[Any], ...]:
    """Return core image dataset classes when quantem.core is importable."""
    global _CORE_IMAGE_DATASET_IMPORT_ATTEMPTED, _CORE_IMAGE_DATASET_TYPES
    if not _CORE_IMAGE_DATASET_IMPORT_ATTEMPTED:
        _CORE_IMAGE_DATASET_IMPORT_ATTEMPTED = True
        try:
            from quantem.core.datastructures import Dataset2d
        except Exception:
            _CORE_IMAGE_DATASET_TYPES = ()
        else:
            _CORE_IMAGE_DATASET_TYPES = (Dataset2d,)
    return _CORE_IMAGE_DATASET_TYPES


def _reject_unknown_kwargs(cls, kwargs: dict) -> None:
    """Raise TypeError for any kwarg that isn't a declared trait (catches typos)."""
    traits = set(cls.class_trait_names())
    unknown = [k for k in kwargs if k not in traits]
    if unknown:
        key = sorted(unknown)[0]
        raise TypeError(f"{cls.__name__}() got unexpected keyword argument {key!r}.")


def _frame_to_rgb(
    frame: np.ndarray,
    *,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    log_scale: bool,
) -> np.ndarray:
    """Colormap a 2D float frame into a uint8 (H, W, 3) RGB array."""
    values = frame.astype(np.float64, copy=False)
    if log_scale:
        values = np.log1p(np.clip(values - np.nanmin(values), 0, None))
    lo = float(np.nanpercentile(values, 1)) if vmin is None else float(vmin)
    hi = float(np.nanpercentile(values, 99)) if vmax is None else float(vmax)
    if hi <= lo:
        hi = lo + 1.0
    normalized = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    colormap = matplotlib.colormaps[cmap]
    rgba = colormap(normalized)
    return (rgba[..., :3] * 255).astype(np.uint8)


def _rgb_to_png_bytes(rgb: np.ndarray) -> bytes:
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


class ChooseLattice(StaticFallbackMixin, anywidget.AnyWidget):
    """Interactive picker for an ordered origin + two lattice-vector points.

    Parameters
    ----------
    data : array_like or quantem Dataset2d
        A single 2D image. NumPy, PyTorch, CuPy arrays, or a quantem
        ``Dataset2d`` (its ``.array``/``.name`` are auto-extracted).
    cmap : str, default "gray"
        Matplotlib colormap name used to render the image.
    vmin, vmax : float, optional
        Explicit display range. Defaults to a robust 1st/99th percentile
        auto-contrast when not given.
    log_scale : bool, default False
        Apply a log1p display stretch before contrast scaling.
    title : str, default ""
        Title shown above the image. Defaults to a quantem Dataset2d's
        ``.name`` when not given explicitly.
    point_labels : sequence of str, default ("Origin", "u", "v")
        Labels for the 3 points, in placement order. The first label is
        shown next to the raw origin point; the second and third are shown
        next to the derived lattice vectors (see ``u`` and ``v`` below),
        not the raw pixel positions of the 2nd/3rd clicks.
    save_state : bool, default False
        Embed full interactive state in the notebook so a cold reopen
        restores the picked points. See ``StaticFallbackMixin`` for the
        image-only fallback used otherwise.
    notebook_preview_format : {"jpeg", "webp", "png"} or None, default None
        Static preview format used when ``save_state=False``. Defaults to
        ``None`` (no fallback image): unlike Show2D/Show3D, ChooseLattice's
        live widget does not reliably hide the saved-notebook fallback
        sibling while interactive, so enabling it shows a redundant image
        alongside the live widget. Opt in explicitly if a cold-reopen
        preview is worth that tradeoff.
    notebook_preview_quality : int, default 88
        Lossy preview quality for JPEG/WebP, from 1 to 100. Ignored for PNG.
    notebook_preview_max_px : int, default 512
        Longest image side for the saved-notebook preview.

    Notes
    -----
    Click on the image to place points in order; once 3 are placed, click
    near an existing point and drag to adjust it. Use ``clear_points()`` (or
    the "Clear Points" button) to start over. Pixel coordinates are always
    reported in the ORIGINAL image's ``(row, col)`` space, regardless of the
    current zoom/pan. The ``u`` and ``v`` properties expose the lattice
    vectors ``a1 - origin`` and ``a2 - origin`` for downstream use.
    """

    _esm = pathlib.Path(__file__).parent / "static" / "chooselattice.js"

    widget_version = traitlets.Unicode("unknown").tag(sync=True)
    height = traitlets.Int(1).tag(sync=True)
    width = traitlets.Int(1).tag(sync=True)
    frame_bytes = traitlets.Bytes(b"").tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    point_labels = traitlets.List(
        traitlets.Unicode(), default_value=["Origin", "u", "v"]
    ).tag(sync=True)
    points = traitlets.List(
        trait=traitlets.List(traitlets.Float()), default_value=[]
    ).tag(sync=True)

    def __init__(
        self,
        data,
        *,
        cmap: str = "gray",
        vmin: float | None = None,
        vmax: float | None = None,
        log_scale: bool = False,
        title: str = "",
        point_labels: Sequence[str] = ("Origin", "u", "v"),
        points: Sequence[Sequence[float]] | None = None,
        save_state: bool = False,
        notebook_preview_format: str | None = None,
        notebook_preview_quality: int = 88,
        notebook_preview_max_px: int = 512,
        **kwargs,
    ) -> None:
        _reject_unknown_kwargs(type(self), kwargs)
        super().__init__(**kwargs)

        core_image_dataset_types = _core_image_dataset_types()
        if bool(core_image_dataset_types) and isinstance(data, core_image_dataset_types):
            if not title and getattr(data, "name", None):
                title = data.name
            data = data.array
        elif hasattr(data, "array") and hasattr(data, "name"):
            # Duck-typed Dataset2d fallback (quantem.core not importable here).
            if not title and getattr(data, "name", None):
                title = data.name
            data = data.array

        frame = to_numpy(data, dtype=np.float32)
        if frame.ndim != 2:
            raise ValueError(
                f"ChooseLattice expects a single 2D image, got array with shape {frame.shape!r}."
            )
        self._data = frame

        rgb = _frame_to_rgb(frame, cmap=cmap, vmin=vmin, vmax=vmax, log_scale=log_scale)
        png_bytes = _rgb_to_png_bytes(rgb)

        self._configure_static_fallback(
            notebook_preview_format=notebook_preview_format,
            notebook_preview_quality=notebook_preview_quality,
            notebook_preview_max_px=notebook_preview_max_px,
        )
        self._save_state = bool(save_state)

        with self.hold_sync():
            self.height = int(frame.shape[0])
            self.width = int(frame.shape[1])
            self.frame_bytes = png_bytes
            self.title = str(title)
            self.point_labels = list(point_labels)
            if points is not None:
                self.points = self._validate_points_value(points)

        try:
            from importlib.metadata import version
            self.widget_version = version("quantem-widget")
        except Exception:
            pass

    @traitlets.validate("points")
    def _validate_points(self, proposal):
        return self._validate_points_value(proposal["value"])

    def _validate_points_value(self, value) -> list[list[float]]:
        points = list(value)
        if len(points) > 3:
            raise traitlets.TraitError(
                f"ChooseLattice supports at most 3 points, got {len(points)}."
            )
        validated = []
        for point in points:
            row, col = point
            row = float(np.clip(row, 0, max(0, self.height - 1)))
            col = float(np.clip(col, 0, max(0, self.width - 1)))
            validated.append([row, col])
        return validated

    def set_points(self, points: Sequence[Sequence[float]]) -> None:
        """Programmatically set the picked points (up to 3, in order)."""
        self.points = [list(p) for p in points]

    def clear_points(self) -> None:
        """Remove all picked points."""
        self.points = []

    @property
    def points_array(self) -> np.ndarray:
        """Picked points as an ``(n, 2)`` array of ``(row, col)`` pixel coordinates."""
        return np.array(self.points, dtype=np.float64).reshape(-1, 2)

    def _point_at(self, index: int) -> tuple[float, float] | None:
        pts = self.points
        if index >= len(pts):
            return None
        return (pts[index][0], pts[index][1])

    @property
    def origin(self) -> tuple[float, float] | None:
        """First picked point ``(row, col)``, or None if not yet placed."""
        return self._point_at(0)

    @property
    def a1(self) -> tuple[float, float] | None:
        """Second picked point ``(row, col)``, or None if not yet placed."""
        return self._point_at(1)

    @property
    def a2(self) -> tuple[float, float] | None:
        """Third picked point ``(row, col)``, or None if not yet placed."""
        return self._point_at(2)

    @property
    def u(self) -> tuple[float, float] | None:
        """Lattice vector ``a1 - origin``, or None until both are placed."""
        origin, a1 = self.origin, self.a1
        if origin is None or a1 is None:
            return None
        return (a1[0] - origin[0], a1[1] - origin[1])

    @property
    def v(self) -> tuple[float, float] | None:
        """Lattice vector ``a2 - origin``, or None until both are placed."""
        origin, a2 = self.origin, self.a2
        if origin is None or a2 is None:
            return None
        return (a2[0] - origin[0], a2[1] - origin[1])

    def _static_png_b64(self, max_px: int = 512) -> str | None:
        if not self.frame_bytes:
            return None
        return base64.b64encode(bytes(self.frame_bytes)).decode("ascii")