"""Interactive 1D traces with a live reconstruction monitor.

``Show1D`` is the line companion to ``Show2D``: ordinary traces, a 2D image
line profile sampled into the trace view, and the live state a torch
reconstruction needs. Scalar histories are appended every iteration and image
snapshots are attached at checkpoints.
"""

import math
import pathlib
from collections.abc import Mapping, Sequence
from typing import Self

import anywidget
import numpy as np
import traitlets

from quantem.widget.units import axis_step, image_axes, per_axis, unit_entry
from quantem.widget.adapters import core as core_adapter
from quantem.widget.colormap import VALID_CMAPS
from quantem.widget.export import HtmlExportMixin
from quantem.widget.fallback import StaticFallbackMixin
from quantem.widget.show1d.export import Show1DExport
from quantem.widget.show1d.state import Show1DState
from quantem.widget.state import SavedStateMixin
from quantem.widget.utils.array import _b64_safe, to_numpy
from quantem.widget.utils.state_io import read_state
from quantem.widget.utils.ui import DisplayMode, reject_unknown_kwargs, resolve_mode

# Okabe-Ito palette: distinguishable for every common colour-vision deficiency
_DEFAULT_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#999999"]
_VALID_SNAPSHOT_CONTRAST_PRESETS = {"full", "0.5-99.5", "1-99", "2-98", "5-95"}
_VALID_OVERLAY_POSITIONS = {"top-left", "top-right", "bottom-left", "bottom-right"}
_UI_DEFAULTS = {
    "show_title": True,
    "show_stats": False,
    "show_legend": True,
    "show_grid": True,
    "show_controls": True,
    "controls_collapsed": False,
}


def sample_line_profile(image, line: Sequence[Sequence[float]], *, profile_width: int = 1) -> np.ndarray:
    """Values along ``line`` in ``(row, col)`` image coordinates, one sample per pixel of length.

    ``profile_width`` > 1 averages that many parallel lines spaced one pixel
    apart across the line, which suppresses noise on a lattice fringe profile
    without smoothing along the profile direction.
    """
    arr = np.asarray(to_numpy(image), dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {arr.shape}")
    if len(line) != 2 or len(line[0]) != 2 or len(line[1]) != 2:
        raise ValueError("line must be ((row0, col0), (row1, col1))")
    (row0, col0), (row1, col1) = line
    width = max(1, int(round(profile_width)))
    length = math.hypot(float(col1) - float(col0), float(row1) - float(row0))
    if width <= 1 or length < 1e-8:
        return _sample_single_line(arr, row0, col0, row1, col1)
    perp_row = -(float(col1) - float(col0)) / length
    perp_col = (float(row1) - float(row0)) / length
    half = (width - 1) / 2
    total = None
    for step in range(width):
        offset = -half + step
        values = _sample_single_line(
            arr,
            float(row0) + offset * perp_row,
            float(col0) + offset * perp_col,
            float(row1) + offset * perp_row,
            float(col1) + offset * perp_col,
        )
        total = values if total is None else total + values
    return total / width


class Show1D(Show1DState, Show1DExport, SavedStateMixin, HtmlExportMixin, StaticFallbackMixin, anywidget.AnyWidget):
    """Interactive 1D viewer for traces, line profiles, and live reconstruction.

    Parameters
    ----------
    data : array_like, mapping, Dataset, or None
        A 1D array, a 2D ``(n_traces, n_points)`` array, a list of 1D arrays, a
        mapping of ``label -> trace``, or ``None`` for an empty live monitor.
    x : array_like, optional
        Shared x positions. Defaults to point indices; an empty monitor fills
        them as :meth:`append` is called.
    labels : list of str, optional
        Per-trace labels.
    title, x_label, y_label : str, optional
        Plot text shown in the widget and exported figures.
    x_integer : bool, default False
        Put x ticks on whole numbers only, for counted quantities such as
        epochs where a tick at 2.5 has no meaning.
    log_scale : bool, default False
        Logarithmic y display; non-positive values are skipped in the plot.
    mode : {"interactive", "presentation", "report", "minimal"}, default "interactive"
        Shared viewer UI preset. Explicit ``show_*`` / ``controls_collapsed``
        keywords override the preset.
    plot_height_px, side_panel_width_px : int
        Initial plot height and snapshot side-panel width in pixels.
    image_cmap : str, default "viridis"
        Colormap for snapshot and profile images.
    snapshot_columns : int, default 0
        Columns of the snapshot image grid; 0 picks an automatic overview.
    snapshot_panel_width_px : int, default 0
        Snapshot tile width in pixels; 0 fits the available width.
    snapshot_histogram_width, snapshot_histogram_height : int, default 360, 52
        Size of the draggable snapshot contrast histogram in CSS pixels.
    profile_image, profile_line, profile_width
        A 2D image shown beside the trace, with the ``((row0, col0), (row1, col1))``
        line that the trace was sampled along; see :meth:`from_image`.
    sampling, units
        Pixel size and unit for the snapshot and profile scale bar. A sequence
        uses its last value, matching ``Show2D`` / ``Show3D``. A ``sampling``
        given without ``units`` is in angstrom.
    state : dict or path, optional
        Restore display state saved with :meth:`save`.
    save_state : bool, default False
        Embed the snapshot and profile image buffers in the saved notebook.
    **kwargs
        Any synced trait, for example ``show_snapshot_fft=True`` or
        ``snapshot_contrast_preset="1-99"``.

    Notes
    -----
    Live use is split between high-rate scalars and lower-rate images: call
    :meth:`append` every iteration and :meth:`snapshot` every ``N`` iterations
    so notebook comms stay responsive.
    """

    _esm = pathlib.Path(__file__).parent.parent / "static" / "show1d.js"
    # ipywidgets snapshots the full state while the comm opens, before the
    # constructor body runs; the preview renderer reads these two.
    _data: np.ndarray = np.empty((0, 0), dtype=np.float32)
    _x: np.ndarray | None = None
    # Synced traits the constructor sets itself, with the keyword that sets each one; a keyword
    # for the trait would be a second spelling of that keyword (see reject_unknown_kwargs).
    _CONSTRUCTOR_TRAITS = {"pixel_size": "sampling", "pixel_unit": "units"}

    y_bytes = traitlets.Bytes(b"").tag(sync=True)
    x_bytes = traitlets.Bytes(b"").tag(sync=True)
    n_traces = traitlets.Int(0).tag(sync=True)
    n_points = traitlets.Int(0).tag(sync=True)
    labels = traitlets.List(traitlets.Unicode()).tag(sync=True)
    colors = traitlets.List(traitlets.Unicode()).tag(sync=True)

    title = traitlets.Unicode("").tag(sync=True)
    x_label = traitlets.Unicode("").tag(sync=True)
    x_integer = traitlets.Bool(False).tag(sync=True)
    y_label = traitlets.Unicode("").tag(sync=True)
    x_unit = traitlets.Unicode("").tag(sync=True)
    y_unit = traitlets.Unicode("").tag(sync=True)
    log_scale = traitlets.Bool(False).tag(sync=True)
    show_title = traitlets.Bool(True).tag(sync=True)
    show_stats = traitlets.Bool(False).tag(sync=True)
    show_legend = traitlets.Bool(True).tag(sync=True)
    show_grid = traitlets.Bool(True).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)
    controls_collapsed = traitlets.Bool(False).tag(sync=True)
    line_width = traitlets.Float(1.5).tag(sync=True)
    plot_height_px = traitlets.Int(320).tag(sync=True)
    plot_width_px = traitlets.Int(400, min=0).tag(sync=True)
    max_width = traitlets.Int(0, min=0).tag(sync=True)
    side_panel_width_px = traitlets.Int(360).tag(sync=True)
    focused_trace = traitlets.Int(-1).tag(sync=True)
    x_range = traitlets.List(traitlets.Float()).tag(sync=True)
    y_range = traitlets.List(traitlets.Float()).tag(sync=True)

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
    bookmarked_snapshot_groups = traitlets.List(traitlets.Int()).tag(sync=True)

    hidden_snapshot_image_labels = traitlets.List(traitlets.Unicode()).tag(sync=True)

    show_snapshots = traitlets.Bool(True).tag(sync=True)
    show_snapshot_thumbnails = traitlets.Bool(True).tag(sync=True)
    snapshot_link_views = traitlets.Bool(False).tag(sync=True)
    show_snapshot_histogram = traitlets.Bool(True).tag(sync=True)
    show_snapshot_fft = traitlets.Bool(False).tag(sync=True)
    snapshot_fft_layout = traitlets.Unicode("overlay").tag(sync=True)
    snapshot_fft_window = traitlets.Bool(True).tag(sync=True)
    snapshot_fft_cmap = traitlets.Unicode("magma").tag(sync=True)
    show_snapshot_profile = traitlets.Bool(False).tag(sync=True)
    snapshot_profile_line = traitlets.List(traitlets.Dict(), default_value=[]).tag(sync=True)
    snapshot_profile_height = traitlets.Int(76).tag(sync=True)
    snapshot_histogram_width = traitlets.Int(360).tag(sync=True)
    snapshot_histogram_height = traitlets.Int(52).tag(sync=True)
    snapshot_contrast_preset = traitlets.Unicode("full").tag(sync=True)
    snapshot_panel_contrast_ranges = traitlets.Dict(default_value={}).tag(sync=True)
    snapshot_contrast_range = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    snapshot_thumbnail_size = traitlets.Int(48).tag(sync=True)
    snapshot_panel_width_px = traitlets.Int(0).tag(sync=True)
    snapshot_columns = traitlets.Int(0).tag(sync=True)
    snapshot_overlay_position = traitlets.Unicode("top-right").tag(sync=True)
    snapshot_real_space_zoom = traitlets.Float(1.0).tag(sync=True)
    snapshot_real_space_center = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    snapshot_fft_zoom = traitlets.Float(1.0).tag(sync=True)
    snapshot_fft_center = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    image_cmap = traitlets.Unicode("viridis").tag(sync=True)
    pixel_size = traitlets.Float(0.0).tag(sync=True)
    pixel_unit = traitlets.Unicode("px").tag(sync=True)
    show_scale_bar = traitlets.Bool(True).tag(sync=True)
    snapshot_playing = traitlets.Bool(False).tag(sync=True)
    snapshot_fps = traitlets.Int(2).tag(sync=True)
    snapshot_loop = traitlets.Bool(True).tag(sync=True)
    snapshot_bounce = traitlets.Bool(False).tag(sync=True)

    profile_image_bytes = traitlets.Bytes(b"").tag(sync=True)
    profile_image_height = traitlets.Int(0).tag(sync=True)
    profile_image_width = traitlets.Int(0).tag(sync=True)
    profile_line = traitlets.List(traitlets.Dict()).tag(sync=True)
    profile_width = traitlets.Int(1)  # read by the Python line sampler only, so it is not synced

    _export_light = traitlets.Bool(False).tag(sync=True)
    # Compact saved-notebook preview (see fallback.py) so a cold rehydrate
    # shows the plot even though the heavy buffers are trimmed.
    _static_fallback_image = traitlets.Unicode("").tag(sync=True)
    _static_fallback_mime = traitlets.Unicode("image/jpeg").tag(sync=True)

    def __init__(
        self,
        data=None,
        *,
        x=None,
        labels: Sequence[str] | None = None,
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        x_integer: bool = False,
        log_scale: bool = False,
        mode: DisplayMode = "interactive",
        plot_height_px: int = 320,
        side_panel_width_px: int = 360,
        image_cmap: str = "viridis",
        snapshot_columns: int = 0,
        snapshot_panel_width_px: int = 0,
        snapshot_histogram_width: int = 360,
        snapshot_histogram_height: int = 52,
        profile_image=None,
        profile_line: Sequence[Sequence[float]] | None = None,
        profile_width: int = 1,
        sampling: float | Sequence[float] | None = None,
        units: str | Sequence[str] | None = None,
        state: dict | str | pathlib.Path | None = None,
        save_state: bool = False,
        **kwargs,
    ) -> None:
        reject_unknown_kwargs(type(self), kwargs, owned=self._CONSTRUCTOR_TRAITS)
        overrides = {key: kwargs.pop(key, None) for key in _UI_DEFAULTS}
        kwargs.update(resolve_mode(mode, defaults=_UI_DEFAULTS, overrides=overrides))
        # a state this widget cannot load is refused before the model opens: a registered widget stays alive, holding its data
        state = None if state is None else read_state(state, "Show1D", self._STATE_KEYS)
        # Before super().__init__ so any get_state during comm-open sees it.
        self._save_state = bool(save_state)
        self._configure_static_fallback()
        super().__init__(**kwargs)
        self._static_fallback_mime = self._static_fallback_mime_type()
        self._data, inferred_labels, inferred_title = _normalize_data(data)
        self._x = _normalize_x(x, self._data.shape[1])
        self._snapshots: list[np.ndarray] = []
        self._profile_image: np.ndarray | None = None
        self.n_traces = int(self._data.shape[0])
        self.n_points = int(self._data.shape[1])
        self.labels = [str(label) for label in (labels or inferred_labels)]
        self.colors = _default_colors(self.n_traces)
        self.title = title or inferred_title
        self.x_label = x_label
        self.y_label = y_label
        self.x_integer = bool(x_integer)
        self.log_scale = bool(log_scale)
        self.plot_height_px = plot_height_px
        self.side_panel_width_px = side_panel_width_px
        self.image_cmap = image_cmap
        self.snapshot_columns = snapshot_columns
        self.snapshot_panel_width_px = snapshot_panel_width_px
        self.snapshot_histogram_width = snapshot_histogram_width
        self.snapshot_histogram_height = snapshot_histogram_height
        self.profile_width = max(1, int(profile_width))
        sampling, units = image_axes(sampling, units)
        if sampling is not None:
            self.pixel_size = float(per_axis(sampling, 2)[-1])
        if units is not None:
            self.pixel_unit = str(per_axis(units, 2)[-1])
        # one unit table for every widget: an unknown string raises, an uncalibrated axis is absent
        if unit_entry(self.pixel_unit)[0] == "pixels":
            self.pixel_size, self.pixel_unit = 0.0, "px"
        self._update_stats()
        self._update_data_bytes()
        if profile_image is not None:
            self._set_profile_image(profile_image, line=profile_line)
        if state is not None:
            self.load_state_dict(state)
        self.observe(self._on_export_request_change, names=["export_request"])

    # --- trait validation: every assignment, from Python or the browser, lands here

    @traitlets.validate("image_cmap", "snapshot_fft_cmap")
    def _validate_cmap(self, proposal: dict) -> str:
        """Reject a colormap the browser has no lookup table for, instead of painting with a silent fallback."""
        name = str(proposal["value"])
        if name not in VALID_CMAPS:
            raise ValueError(f"Unknown {proposal['trait'].name} {name!r}. Valid: {sorted(VALID_CMAPS)}")
        return name

    @traitlets.validate("snapshot_contrast_preset")
    def _validate_snapshot_contrast_preset(self, proposal: dict) -> str:
        """Accept only the percentile presets the snapshot histogram implements."""
        name = str(proposal["value"]).strip()
        if name not in _VALID_SNAPSHOT_CONTRAST_PRESETS:
            raise ValueError(f"Unknown snapshot_contrast_preset {name!r}. Valid: {sorted(_VALID_SNAPSHOT_CONTRAST_PRESETS)}")
        return name

    @traitlets.validate("snapshot_overlay_position")
    def _validate_snapshot_overlay_position(self, proposal: dict) -> str:
        """Normalise ``top_left`` style spellings to the corner names the browser places labels with."""
        name = str(proposal["value"]).strip().lower().replace("_", "-")
        if name not in _VALID_OVERLAY_POSITIONS:
            raise ValueError(f"snapshot_overlay_position must be one of {sorted(_VALID_OVERLAY_POSITIONS)}, got {proposal['value']!r}")
        return name

    @traitlets.validate("snapshot_fft_layout")
    def _validate_snapshot_fft_layout(self, proposal: dict) -> str:
        """The snapshot FFT is drawn over the image or below it; nothing else."""
        name = str(proposal["value"]).strip().lower()
        if name not in {"overlay", "below"}:
            raise ValueError(f"snapshot_fft_layout must be 'overlay' or 'below', got {proposal['value']!r}")
        return name

    @traitlets.validate("snapshot_contrast_range")
    def _validate_snapshot_contrast_range(self, proposal: dict) -> list[float]:
        """A manual contrast window: empty (automatic) or an increasing finite pair."""
        return _finite_pair(proposal["value"], "snapshot_contrast_range", increasing=True)

    @traitlets.validate("snapshot_real_space_center", "snapshot_fft_center")
    def _validate_snapshot_view_center(self, proposal: dict) -> list[float]:
        """A pan center: empty (centered) or a finite pair."""
        return _finite_pair(proposal["value"], proposal["trait"].name)

    @traitlets.validate("snapshot_real_space_zoom", "snapshot_fft_zoom")
    def _validate_snapshot_view_zoom(self, proposal: dict) -> float:
        """Clamp a snapshot zoom into the 1-32x range the viewer supports; NaN is rejected."""
        zoom = float(proposal["value"])
        if not math.isfinite(zoom):
            raise ValueError(f"{proposal['trait'].name} must be finite, got {proposal['value']!r}")
        return max(1.0, min(32.0, zoom))

    @traitlets.validate("pixel_size")
    def _validate_pixel_size(self, proposal: dict) -> float:
        """A pixel size must be finite and non-negative; 0 means uncalibrated."""
        value = float(proposal["value"])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"sampling/pixel_size must be finite and >= 0, got {value}")
        return value

    @traitlets.validate("snapshot_profile_line", "profile_line")
    def _validate_profile_line(self, proposal: dict) -> list[dict[str, float]]:
        """A profile line is at most two finite ``{row, col}`` points."""
        points = [{"row": float(point["row"]), "col": float(point["col"])} for point in proposal["value"]]
        if len(points) > 2:
            raise ValueError(f"{proposal['trait'].name} must contain at most two (row, col) points, got {points!r}")
        if not all(math.isfinite(point["row"]) and math.isfinite(point["col"]) for point in points):
            raise ValueError(f"{proposal['trait'].name} coordinates must be finite, got {points!r}")
        return points

    @traitlets.validate("bookmarked_snapshot_groups")
    def _validate_bookmarked_snapshot_groups(self, proposal: dict) -> list[int]:
        """Bookmarks as sorted unique group indices, so the timeline marks each group once."""
        indices = sorted({int(value) for value in proposal["value"]})
        if indices and indices[0] < 0:
            raise ValueError(f"bookmarked_snapshot_groups must contain non-negative indices, got {indices[0]}")
        return indices

    @traitlets.validate("hidden_snapshot_image_labels")
    def _validate_image_labels(self, proposal: dict) -> list[str]:
        """One entry per label key, so ``lambda_1`` and ``lambda 1`` are not starred twice."""
        return _unique_labels(proposal["value"])

    @traitlets.validate(
        "plot_height_px", "side_panel_width_px", "snapshot_thumbnail_size", "snapshot_panel_width_px",
        "snapshot_fps", "snapshot_columns", "snapshot_profile_height", "snapshot_histogram_width",
        "snapshot_histogram_height",
    )
    def _validate_clamped_int(self, proposal: dict) -> int:
        """Round and clamp a size or count into its ``_INT_RANGES`` bounds, whether Python or the browser set it."""
        low, high = _INT_RANGES[proposal["trait"].name]
        return max(low, min(high, int(round(float(proposal["value"])))))

    # --- constructors

    @classmethod
    def live(
        cls,
        traces: Sequence[str] | None = None,
        *,
        title: str = "Live Reconstruction",
        x_label: str = "iteration",
        y_label: str = "",
        log_scale: bool = True,
        **kwargs,
    ) -> Self:
        """An empty monitor for repeated :meth:`append` calls."""
        labels = [str(name) for name in (traces or [])]
        return cls(np.empty((len(labels), 0), dtype=np.float32), labels=labels, title=title, x_label=x_label, y_label=y_label, log_scale=log_scale, **kwargs)

    @classmethod
    def from_image(
        cls,
        image,
        *,
        line: Sequence[Sequence[float]],
        profile_width: int = 1,
        sampling: float | None = None,
        units: str | None = None,
        **kwargs,
    ) -> Self:
        """A line-profile viewer: the trace sampled along ``line`` with the 2D image beside it.

        ``line`` is ``((row0, col0), (row1, col1))`` in image pixels; ``sampling``
        turns the pixel distance along the line into ``units`` (angstrom when
        ``units`` is omitted). Without a ``sampling`` the distance is in pixels.
        """
        # the distance axis is in the image's unit, so the x_unit trait follows units= here
        reject_unknown_kwargs(cls, kwargs, owned={"x_unit": "units"})
        kwargs.setdefault("title", "Line Profile")
        kwargs.setdefault("y_label", "value")
        sampling, units = image_axes(sampling, units)
        return cls(
            profile_image=image,
            profile_line=line,
            profile_width=profile_width,
            x_label="distance",
            # a unit with no step, or an uncalibrated unit, leaves the distance in pixels
            x_unit="px" if axis_step(per_axis(sampling, 2)[-1], per_axis(units, 2)[-1]) is None else per_axis(units, 2)[-1],
            sampling=sampling,
            units=units,
            **kwargs,
        )

    # --- live updates

    def append(self, x: float | None = None, **values) -> Self:
        """Append one sample to named traces; new names start a trace back-filled with NaN."""
        if not values:
            raise ValueError("append requires at least one named value")
        return self.extend(None if x is None else [x], **{name: [value] for name, value in values.items()})

    def extend(self, x: Sequence[float] | np.ndarray | None = None, **values) -> Self:
        """Append a block of samples to named traces in one widget update.

        ``x`` defaults to continuing from the last x value. New names start a
        trace back-filled with NaN; existing traces missing from ``values`` get
        NaN over the appended span.
        """
        if not values:
            raise ValueError("extend requires at least one named value sequence")
        arrays = {str(name): np.asarray(to_numpy(raw), dtype=np.float32).ravel() for name, raw in values.items()}
        lengths = {array.size for array in arrays.values()}
        if len(lengths) != 1:
            raise ValueError(f"all extend value sequences must have the same length, got {sorted(lengths)}")
        n_new = lengths.pop()
        if n_new == 0:
            return self
        if x is None:
            start = float(self._x[-1] + 1) if self._x is not None and self._x.size else float(self.n_points)
            x_values = np.arange(start, start + n_new, dtype=np.float32)
        else:
            x_values = np.asarray(to_numpy(x), dtype=np.float32).ravel()
            if x_values.size != n_new:
                raise ValueError(f"x must have length {n_new}, got {x_values.size}")
        with self.hold_sync():
            labels = list(self.labels) + [name for name in arrays if name not in self.labels]
            block = np.full((len(labels), n_new), np.nan, dtype=np.float32)
            for row, label in enumerate(labels):
                if label in arrays:
                    block[row] = arrays[label]
            filler = np.full((len(labels) - self.n_traces, self.n_points), np.nan, dtype=np.float32)
            self._data = np.column_stack([np.vstack([self._data, filler]), block])
            self._x = x_values if self._x is None else np.concatenate([self._x, x_values])
            self.labels = labels
            self.colors = _default_colors(len(labels))
            self.n_traces = int(self._data.shape[0])
            self.n_points = int(self._data.shape[1])
            self._update_stats()
            self._update_data_bytes()
        return self

    def snapshot(self, iteration: float, *, label: str | None = None, **images) -> Self:
        """Attach named 2D images to an iteration as one snapshot group.

        Images passed together (``object=..., probe=...``) are one checkpoint:
        playback steps through groups and the plot marks the group's x value.
        """
        if not images:
            raise ValueError("snapshot requires one or more named images")
        group_idx = self.n_snapshot_groups
        first_image_idx = len(self._snapshots)
        for name, value in images.items():
            arr = np.ascontiguousarray(to_numpy(value), dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f"snapshot {name!r} must be 2D, got shape {arr.shape}")
            self._snapshots.append(arr)
        names = [str(name) for name in images]
        self.snapshot_iterations = [*self.snapshot_iterations, *[float(iteration)] * len(names)]
        self.snapshot_labels = [*self.snapshot_labels, *names]
        self.snapshot_image_labels = [*self.snapshot_image_labels, *names]
        self.snapshot_group_indices = [*self.snapshot_group_indices, *[group_idx] * len(names)]
        self.snapshot_group_iterations = [*self.snapshot_group_iterations, float(iteration)]
        self.snapshot_group_labels = [*self.snapshot_group_labels, label or f"iter {iteration:g}"]
        self.n_snapshot_groups = group_idx + 1
        self._update_snapshot_bytes()
        self.selected_snapshot_idx = first_image_idx
        self.selected_snapshot_group_idx = group_idx
        return self

    def set_data(self, data, *, x=None, labels: Sequence[str] | None = None) -> Self:
        """Replace the traces while keeping every display setting."""
        self._data, inferred_labels, _ = _normalize_data(data)
        self._x = _normalize_x(x, self._data.shape[1])
        self.n_traces = int(self._data.shape[0])
        self.n_points = int(self._data.shape[1])
        self.labels = [str(label) for label in (labels or inferred_labels)]
        self.colors = _default_colors(self.n_traces)
        self._update_stats()
        self._update_data_bytes()
        return self

    def star_snapshot_group(self, group: int | str | None = None) -> Self:
        """Star a snapshot group (index, label, or the selected one) in the playback timeline."""
        self.bookmarked_snapshot_groups = [*self.bookmarked_snapshot_groups, self._snapshot_group_index(group)]
        return self

    def __repr__(self) -> str:
        """Trace and point counts, e.g. ``Show1D(3 traces x 120 points)``."""
        if self.n_traces == 1:
            return f"Show1D({self.n_points} points)"
        return f"Show1D({self.n_traces} traces x {self.n_points} points)"

    # --- derived synced buffers

    def _snapshot_group_index(self, group: int | str | None) -> int:
        """A group index from an index, a group label (label-key match), or ``None`` for the selected group."""
        if self.n_snapshot_groups == 0:
            raise ValueError("Show1D has no snapshot groups")
        if group is None:
            return max(0, int(self.selected_snapshot_group_idx))
        if isinstance(group, str):
            for idx, label in enumerate(self.snapshot_group_labels):
                if _label_key(label) == _label_key(group):
                    return idx
            raise ValueError(f"unknown snapshot group {group!r}")
        if not 0 <= int(group) < self.n_snapshot_groups:
            raise ValueError(f"snapshot group index {group} out of range [0, {self.n_snapshot_groups})")
        return int(group)

    def _set_profile_image(self, image, *, line: Sequence[Sequence[float]] | None = None) -> None:
        """Attach the 2D context image; with ``line`` the trace is resampled along it."""
        arr = np.ascontiguousarray(to_numpy(image), dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"profile image must be 2D, got shape {arr.shape}")
        self._profile_image = arr
        self.profile_image_height = int(arr.shape[0])
        self.profile_image_width = int(arr.shape[1])
        self.profile_image_bytes = _b64_safe(arr.tobytes())
        if line is not None:
            self.profile_line = [{"row": float(row), "col": float(col)} for row, col in line]
            values = sample_line_profile(arr, line, profile_width=self.profile_width)
            (row0, col0), (row1, col1) = line
            length = math.hypot(float(row1) - float(row0), float(col1) - float(col0)) * (self.pixel_size or 1.0)
            self.set_data(values, x=np.linspace(0.0, length, values.size, dtype=np.float32), labels=["profile"])

    def _effective_x(self) -> np.ndarray:
        """The x positions the plot uses: the stored ones, or point indices when none were given."""
        return np.arange(self.n_points, dtype=np.float32) if self._x is None else self._x

    def _update_data_bytes(self) -> None:
        """Sync the traces and x positions as float32 bytes, the layout the browser reads."""
        self.y_bytes = _b64_safe(np.ascontiguousarray(self._data, dtype=np.float32).tobytes())
        x_values = self._effective_x()
        self.x_bytes = _b64_safe(np.ascontiguousarray(x_values, dtype=np.float32).tobytes()) if x_values.size else b""

    def _update_stats(self) -> None:
        """Per-trace mean, min, max and std for the stats row, ignoring NaN gaps from back-filled traces."""
        if self._data.size == 0:
            self.stats_mean, self.stats_min, self.stats_max, self.stats_std = [], [], [], []
            return
        self.stats_mean = np.nanmean(self._data, axis=1).tolist()
        self.stats_min = np.nanmin(self._data, axis=1).tolist()
        self.stats_max = np.nanmax(self._data, axis=1).tolist()
        self.stats_std = np.nanstd(self._data, axis=1).tolist()

    def _update_snapshot_bytes(self) -> None:
        """Pack every snapshot into one NaN-padded float32 stack; the browser crops each by its own size."""
        if not self._snapshots:
            self.snapshot_bytes = b""
            self.n_snapshots = self.snapshot_height = self.snapshot_width = 0
            self.snapshot_heights, self.snapshot_widths = [], []
            return
        heights = [int(snap.shape[0]) for snap in self._snapshots]
        widths = [int(snap.shape[1]) for snap in self._snapshots]
        stack = np.full((len(self._snapshots), max(heights), max(widths)), np.nan, dtype=np.float32)
        for idx, snap in enumerate(self._snapshots):
            stack[idx, : snap.shape[0], : snap.shape[1]] = snap
        self.n_snapshots, self.snapshot_height, self.snapshot_width = stack.shape
        self.snapshot_heights = heights
        self.snapshot_widths = widths
        self.snapshot_bytes = _b64_safe(stack.tobytes())


_INT_RANGES = {
    "plot_height_px": (220, 960),
    "side_panel_width_px": (300, 4096),
    "snapshot_thumbnail_size": (24, 112),
    "snapshot_panel_width_px": (0, 4096),
    "snapshot_fps": (1, 24),
    "snapshot_columns": (0, 8),
    "snapshot_profile_height": (44, 220),
    "snapshot_histogram_width": (110, 640),
    "snapshot_histogram_height": (36, 110),
}


def _default_colors(n_traces: int) -> list[str]:
    """One palette color per trace, cycling when there are more traces than colors."""
    return [_DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)] for idx in range(n_traces)]


def _normalize_data(data) -> tuple[np.ndarray, list[str], str]:
    """Traces as a float32 ``(n_traces, n_points)`` block with inferred labels and title."""
    title = ""
    if core_adapter.is_dataset(data):
        title = str(data.name or "")
        data = core_adapter.as_array(data)
    if data is None:
        return np.empty((0, 0), dtype=np.float32), [], title
    if isinstance(data, Mapping):
        return _stack_equal_length([np.asarray(to_numpy(value), dtype=np.float32).ravel() for value in data.values()]), [str(key) for key in data], title
    if isinstance(data, list) and not (data and all(np.asarray(to_numpy(value)).ndim == 0 for value in data)):
        arrays = [np.asarray(to_numpy(value), dtype=np.float32).ravel() for value in data]
        return _stack_equal_length(arrays), [f"Data {idx + 1}" for idx in range(len(arrays))], title
    arr = np.asarray(to_numpy(data), dtype=np.float32)
    if arr.ndim <= 1:
        return arr.reshape(1, -1), ["Data"], title
    if arr.ndim == 2:
        return np.ascontiguousarray(arr), [f"Data {idx + 1}" for idx in range(arr.shape[0])], title
    raise ValueError(f"Expected 1D or 2D data, got shape {arr.shape}")


def _stack_equal_length(arrays: list[np.ndarray]) -> np.ndarray:
    """Traces stacked as float32 ``(n_traces, n_points)``; unequal lengths raise because traces share one x axis."""
    if not arrays:
        return np.empty((0, 0), dtype=np.float32)
    for idx, arr in enumerate(arrays):
        if arr.size != arrays[0].size:
            raise ValueError(f"All traces must have the same length. Trace 0 has {arrays[0].size} points, trace {idx} has {arr.size}.")
    return np.ascontiguousarray(np.stack(arrays), dtype=np.float32)


def _normalize_x(x, n_points: int) -> np.ndarray | None:
    """x positions as float32 with one value per point; point indices when ``x`` is None, nothing for an empty trace."""
    if x is None:
        return None if n_points == 0 else np.arange(n_points, dtype=np.float32)
    arr = np.asarray(to_numpy(x), dtype=np.float32).ravel()
    if arr.size != n_points:
        raise ValueError(f"x has {arr.size} points but data has {n_points} points")
    return np.ascontiguousarray(arr)


def _label_key(label: str) -> str:
    """Case- and punctuation-insensitive key so ``lambda_10``, ``lambda 10`` and ``Lambda-10`` name one image."""
    return "".join(ch.lower() for ch in str(label) if ch.isalnum())


def _unique_labels(labels: Sequence[str]) -> list[str]:
    """Stripped labels, first spelling kept per label key, blanks dropped."""
    out: list[str] = []
    seen: set[str] = set()
    for label in labels:
        text = str(label).strip()
        key = _label_key(text)
        if key and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _finite_pair(value: Sequence[float], name: str, *, increasing: bool = False) -> list[float]:
    """``[]`` or two finite floats, optionally ``min < max``."""
    if len(value) == 0:
        return []
    if len(value) != 2:
        raise ValueError(f"{name} must be empty or contain exactly two values, got {value!r}")
    first, second = float(value[0]), float(value[1])
    if not math.isfinite(first) or not math.isfinite(second):
        raise ValueError(f"{name} values must be finite, got {value!r}")
    if increasing and second <= first:
        raise ValueError(f"{name} must be increasing (min < max), got {value!r}")
    return [first, second]


def _sample_single_line(image: np.ndarray, row0: float, col0: float, row1: float, col1: float) -> np.ndarray:
    """Bilinear samples along one line, one per pixel of line length, edges clamped."""
    height, width = image.shape
    delta_col = float(col1) - float(col0)
    delta_row = float(row1) - float(row0)
    n_samples = max(2, int(math.ceil(math.hypot(delta_col, delta_row))) + 1)
    out = np.empty(n_samples, dtype=np.float32)
    for index in range(n_samples):
        fraction = index / (n_samples - 1)
        col = float(col0) + fraction * delta_col
        row = float(row0) + fraction * delta_row
        col_floor = math.floor(col)
        row_floor = math.floor(row)
        col_frac = col - col_floor
        row_frac = row - row_floor
        col_lo = max(0, min(width - 1, col_floor))
        col_hi = max(0, min(width - 1, col_floor + 1))
        row_lo = max(0, min(height - 1, row_floor))
        row_hi = max(0, min(height - 1, row_floor + 1))
        out[index] = (
            image[row_lo, col_lo] * (1 - col_frac) * (1 - row_frac)
            + image[row_lo, col_hi] * col_frac * (1 - row_frac)
            + image[row_hi, col_lo] * (1 - col_frac) * row_frac
            + image[row_hi, col_hi] * col_frac * row_frac
        )
    return out

