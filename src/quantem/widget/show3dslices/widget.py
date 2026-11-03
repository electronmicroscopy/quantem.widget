"""Show3DSlices: a top slice plus one oblique vertical cut through a 3D volume.

The float32 volume is sent to the browser once; every slice, page and panel
change is re-sliced in JavaScript. Several volumes can share one widget as
panels (``(panel, slices, rows, cols)``) or as pages of comparable volumes
(``(page, panel, slices, rows, cols)`` or a 4D stack with ``page_labels``), all
sharing the same cut geometry, contrast and camera.
"""

import copy
import gc
import json
import math
import pathlib
from collections.abc import Sequence
from typing import Self

import anywidget
import numpy as np
import traitlets

from quantem.widget.adapters import core as core_adapter
from quantem.widget.colormap import VALID_CMAPS
from quantem.widget.export import HtmlExportMixin, export_slug
from quantem.widget.pages import resolve_page_labels
from quantem.widget.show3dslices.alignment import estimate_global_slice_alignment
from quantem.widget.state import StateFileMixin
from quantem.widget.units import axis_step, image_axes, per_axis, stated
from quantem.widget.utils.array import to_numpy
from quantem.widget.utils.state_io import read_state
from quantem.widget.utils.ui import validate_display_limits

_SLICE_ALIGNMENT_MODES = frozenset({"off", "auto", "manual"})


def _voxel_size_A(sampling: float | Sequence[float], units: str | Sequence[str]) -> list[float]:
    """Voxel sampling as ``[slice, row, col]`` in Angstrom; a scalar applies to every axis, an axis in frames or pixels gets 0 (no bar).

    The scale bar reads Angstrom: a length in another unit of ``quantem.widget.units`` is converted,
    and an axis calibrated in something that is not a length (mrad, 1/angstrom) raises.
    """
    samples = per_axis(sampling, 3)
    if len(samples) != 3:
        raise ValueError(f"sampling must be a scalar or 3 values (slice, row, col), got {len(samples)}")
    unit_list = per_axis(units, 3)
    if any(not math.isfinite(float(sample)) or float(sample) < 0 for sample in samples):
        raise ValueError(f"sampling must be finite and >= 0, got {samples}")
    axes = []
    for sample, unit in zip(samples, unit_list):
        step = axis_step(sample, unit)
        if step is not None and step[0] != "length":
            raise ValueError(f"Show3DSlices draws its scale bar in a length; the axis unit {unit!r} is a {step[0]}. "
                             "Use 'angstrom', 'nm' or 'pm', or 'frame' / 'pixels' for an axis without a length.")
        axes.append(0.0 if step is None else step[1])
    return axes


def _flatten_pages(
    data: np.ndarray,
    panel_titles: Sequence[str] | None,
    page_labels: Sequence[str | None] | None,
) -> tuple[np.ndarray, list[str], int, int, list[str]]:
    """Flatten pages into one ``(panel, slices, rows, cols)`` stack.

    5D input is ``(page, panel, slices, rows, cols)``; a 4D stack with ``page_labels``
    is one volume per page; a 4D stack without labels is a multi-panel volume
    and a 3D array is one panel. Returns the flattened stack, one title per
    flattened panel, ``n_pages``, ``panels_per_page`` (0 when unpaged) and the
    page labels.
    """
    if data.ndim == 3:
        data = data[None]
    if data.ndim == 4 and page_labels is not None:
        data = data[:, None]
    if data.ndim == 5:
        n_pages, panels_per_page = int(data.shape[0]), int(data.shape[1])
        labels = resolve_page_labels(page_labels, [f"Page {idx + 1}" for idx in range(n_pages)], n_pages)
        if panel_titles is None:
            slot_titles = ["Volume" if panels_per_page == 1 else f"Panel {idx + 1}" for idx in range(panels_per_page)]
            titles = slot_titles * n_pages
        elif len(panel_titles) == panels_per_page:
            titles = [str(value) for _ in range(n_pages) for value in panel_titles]
        elif len(panel_titles) == n_pages * panels_per_page:
            titles = [str(value) for value in panel_titles]
        else:
            raise ValueError(
                f"panel_titles for paged Show3DSlices must have length {panels_per_page} "
                f"or {n_pages * panels_per_page}, got {len(panel_titles)}"
            )
        return data.reshape(n_pages * panels_per_page, *data.shape[2:]), titles, n_pages, panels_per_page, labels
    if data.ndim != 4:
        raise ValueError(f"Show3DSlices requires 3D data or 4D panel data, got {data.ndim}D")
    n_panels = int(data.shape[0])
    if panel_titles is None:
        titles = [] if n_panels == 1 else [f"Panel {idx + 1}" for idx in range(n_panels)]
    else:
        titles = [str(value) for value in panel_titles]
        if len(titles) != n_panels:
            raise ValueError(f"panel_titles length {len(titles)} must match panel_count {n_panels}")
    return data, titles, 1, 0, []


class Show3DSlices(StateFileMixin, HtmlExportMixin, anywidget.AnyWidget):
    """Linked top-slice and oblique vertical-slice viewer for 3D volumes.

    Parameters
    ----------
    data : array_like or Dataset3d
        ``(slices, rows, cols)`` volume, ``(panel, slices, rows, cols)`` panels, or pages:
        ``(page, panel, slices, rows, cols)``, or ``(page, slices, rows, cols)`` together with
        ``page_labels``. A ``Dataset3d`` supplies title, sampling and units.
    title : str
        Title above the viewer.
    panel_titles : sequence of str, optional
        One title per panel (or per panel slot for pages).
    page_labels : sequence of str, optional
        One label per page. Page switching keeps the slice geometry, zoom and
        display settings so depths compare directly.
    show_title : bool
        Show the title row.
    cmap : str
        Colormap for every panel.
    sampling : float or (slice, row, col), optional
        Voxel sampling for the scale bars, paired with ``units``.
    units : str or sequence of str, optional
        Unit of ``sampling`` (``"A"`` or ``"nm"``); one value applies to every axis.
        Angstrom when omitted.
    show_scale_bar : bool
        Draw calibrated scale bars when a sampling is known.
    slice_stretch : float
        Display stretch of the depth axis, 1 to 50, for volumes with far fewer
        slices than lateral pixels.
    panel_width_px : int
        Display width per panel in CSS pixels; 0 uses the frontend default.
    show_fft : bool
        Show the power spectrum of the active plane.
    fft_window : bool
        Hann-window each plane before its FFT to suppress edge streaks.
    smooth : bool
        Bilinear interpolation when the canvas enlarges a slice. Default False
        paints every data pixel as a sharp block (nearest neighbour).
    auto_contrast : bool
        Percentile contrast (``image_vmin_pct`` to ``image_vmax_pct``); long-tailed
        phase histograms are crushed by the raw min/max. Applies only while
        ``vmin`` and ``vmax`` are both None.
    vmin, vmax : float, optional
        Manual contrast limits, used exactly as given: setting either one turns
        Auto off for the display, so the window is never narrowed to percentiles.
    image_vmin_pct, image_vmax_pct : float
        Percentiles for the automatic contrast window.
    slice_alignment : {"off", "auto", "manual"}
        Display-only global post-alignment. ``"auto"`` estimates one row/col
        shift per slice from adjacent-slice registration; the raw volume is
        never modified.
    dim_labels : list of str, optional
        Names of axes 0, 1, 2; default ``["slice", "row", "col"]``.
    offline : bool
        Pack the volume as uint8 against its global range (256 levels) for
        small standalone HTML, and print one line saying so; ``False`` keeps
        exact float32 in the browser.
    """

    _esm = pathlib.Path(__file__).parent.parent / "static" / "show3dslices.js"

    # [slices, rows, cols] of one panel's volume
    volume_shape = traitlets.List(traitlets.Int(), default_value=[1, 1, 1]).tag(sync=True)
    panel_count = traitlets.CInt(1).tag(sync=True)
    active_panel = traitlets.CInt(0).tag(sync=True)
    panel_titles = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    n_pages = traitlets.CInt(1).tag(sync=True)
    page_idx = traitlets.CInt(0).tag(sync=True)
    panels_per_page = traitlets.CInt(0).tag(sync=True)
    page_labels = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    col_index = traitlets.CInt(0).tag(sync=True)
    row_index = traitlets.CInt(0).tag(sync=True)
    slice_index = traitlets.CInt(0).tag(sync=True)
    oblique_angle = traitlets.Float(0.0).tag(sync=True)
    oblique_profile_line = traitlets.List(traitlets.Dict(), default_value=[]).tag(sync=True)
    volume_bytes = traitlets.Bytes(b"").tag(sync=True)
    offline = traitlets.Bool(False).tag(sync=True)
    # True only on the clone written by export_html: the standalone page renders
    # on a light background whatever the viewer's OS theme.
    _export_light = traitlets.Bool(False).tag(sync=True)
    _offline_min = traitlets.Float(0.0).tag(sync=True)
    _offline_max = traitlets.Float(1.0).tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    show_title = traitlets.Bool(True).tag(sync=True)
    cmap = traitlets.Unicode("plasma").tag(sync=True)
    log_scale = traitlets.Bool(False).tag(sync=True)
    auto_contrast = traitlets.Bool(True).tag(sync=True)
    vmin = traitlets.Float(None, allow_none=True).tag(sync=True)
    vmax = traitlets.Float(None, allow_none=True).tag(sync=True)
    # the [slice, row, col] sampling every panel's scale bar reads
    voxel_size_A = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0, 0.0]).tag(sync=True)
    show_scale_bar = traitlets.Bool(True).tag(sync=True)
    slice_stretch = traitlets.Float(30.0).tag(sync=True)
    panel_width_px = traitlets.Int(0).tag(sync=True)
    view_state = traitlets.Dict(default_value={}).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)
    controls_collapsed = traitlets.Bool(False).tag(sync=True)
    show_crosshair = traitlets.Bool(True).tag(sync=True)
    show_fft = traitlets.Bool(False).tag(sync=True)
    fft_window = traitlets.Bool(False).tag(sync=True)
    fft_colormap = traitlets.Unicode("inferno").tag(sync=True)
    fft_log_scale = traitlets.Bool(False).tag(sync=True)
    fft_auto = traitlets.Bool(True).tag(sync=True)
    orthographic = traitlets.Bool(False).tag(sync=True)
    smooth = traitlets.Bool(False).tag(sync=True)
    flip = traitlets.Bool(False).tag(sync=True)
    show_colorbar = traitlets.Bool(False).tag(sync=True)
    image_vmin_pct = traitlets.Float(0.0).tag(sync=True)
    image_vmax_pct = traitlets.Float(100.0).tag(sync=True)
    show_slice_planes = traitlets.Bool(True).tag(sync=True)
    plane_visibility = traitlets.List(traitlets.Bool(), default_value=[True, True]).tag(sync=True)
    slice_alignment = traitlets.Unicode("off").tag(sync=True)
    row_shift_px_per_slice = traitlets.Float(0.0).tag(sync=True)
    col_shift_px_per_slice = traitlets.Float(0.0).tag(sync=True)
    slice_alignment_cached = traitlets.Bool(False).tag(sync=True)
    slice_alignment_status = traitlets.Unicode("").tag(sync=True)
    _slice_alignment_request = traitlets.Unicode("").tag(sync=True)
    volume_opacity = traitlets.Float(0.5).tag(sync=True)
    slice_plane_opacity = traitlets.Float(0.35).tag(sync=True)
    dim_labels = traitlets.List(traitlets.Unicode(), default_value=["slice", "row", "col"]).tag(sync=True)
    playing = traitlets.Bool(False).tag(sync=True)
    reverse = traitlets.Bool(False).tag(sync=True)
    boomerang = traitlets.Bool(True).tag(sync=True)
    fps = traitlets.Float(30.0).tag(sync=True)
    loop = traitlets.Bool(True).tag(sync=True)
    play_axis = traitlets.Int(0).tag(sync=True)  # 0 slice, 1 oblique plane, 3 both

    _STATE_KEYS = (
        "title", "show_title", "page_idx", "page_labels",
        "active_panel", "panel_titles", "cmap", "log_scale", "auto_contrast", "vmin", "vmax",
        "show_controls", "controls_collapsed", "show_crosshair", "show_fft", "fft_window",
        "fft_colormap", "fft_log_scale", "fft_auto", "orthographic", "smooth", "flip",
        "show_colorbar", "image_vmin_pct", "image_vmax_pct", "show_slice_planes",
        "plane_visibility", "slice_alignment", "row_shift_px_per_slice", "col_shift_px_per_slice",
        "slice_alignment_cached", "volume_opacity", "slice_plane_opacity",
        "voxel_size_A", "show_scale_bar", "slice_stretch", "panel_width_px", "view_state",
        "col_index", "row_index", "slice_index", "oblique_angle", "oblique_profile_line", "fps", "loop",
        "reverse", "boomerang", "play_axis", "dim_labels",
    )

    @traitlets.validate("cmap", "fft_colormap")
    def _validate_cmap(self, proposal: dict) -> str:
        """Reject unknown colormap names so the JS LUT lookup never misses."""
        value = str(proposal["value"])
        if value not in VALID_CMAPS:
            raise traitlets.TraitError(f"Unknown {proposal['trait'].name} {value!r}. Valid: {sorted(VALID_CMAPS)}")
        return value

    @traitlets.validate("oblique_angle")
    def _validate_oblique_angle(self, proposal: dict) -> float:
        """A vertical plane at angle a is the plane at a + 180, so wrap to [0, 180)."""
        value = float(proposal["value"])
        if not math.isfinite(value):
            raise traitlets.TraitError(f"oblique_angle must be finite, got {value}")
        return value % 180.0

    @traitlets.validate("oblique_profile_line")
    def _validate_oblique_profile_line(self, proposal: dict) -> list[dict[str, float]]:
        """Empty, or exactly two finite ``{"row", "col"}`` endpoints."""
        value = list(proposal["value"])
        if not value:
            return []
        if len(value) != 2:
            raise traitlets.TraitError(f"oblique_profile_line must be [] or two endpoints, got {len(value)}")
        endpoints = [{"row": float(point["row"]), "col": float(point["col"])} for point in value]
        if not all(math.isfinite(point["row"]) and math.isfinite(point["col"]) for point in endpoints):
            raise traitlets.TraitError("oblique_profile_line endpoints must be finite")
        return endpoints

    @traitlets.validate("plane_visibility")
    def _validate_plane_visibility(self, proposal: dict) -> list[bool]:
        """Exactly two flags, ``[top, oblique]``."""
        value = [bool(visible) for visible in proposal["value"]]
        if len(value) != 2:
            raise traitlets.TraitError(f"plane_visibility must have length 2 [top, oblique], got {len(value)}")
        return value

    @traitlets.validate("slice_alignment")
    def _validate_slice_alignment(self, proposal: dict) -> str:
        """``"off"``, ``"auto"`` or ``"manual"`` in any case; an empty value means off."""
        value = str(proposal["value"] or "off").strip().lower()
        if value not in _SLICE_ALIGNMENT_MODES:
            raise traitlets.TraitError(f"slice_alignment must be one of {sorted(_SLICE_ALIGNMENT_MODES)}, got {proposal['value']!r}")
        return value

    @traitlets.validate("row_shift_px_per_slice", "col_shift_px_per_slice")
    def _validate_slice_alignment_shift(self, proposal: dict) -> float:
        """A per-slice display shift must be finite, or every shifted slice would vanish."""
        value = float(proposal["value"])
        if not math.isfinite(value):
            raise traitlets.TraitError(f"{proposal['trait'].name} must be finite, got {value}")
        return value

    @traitlets.validate("voxel_size_A")
    def _validate_voxel_size_A(self, proposal: dict) -> list[float]:
        """Three finite, non-negative samplings ``[slice, row, col]`` in Angstrom (0 means uncalibrated)."""
        value = [float(axis) for axis in proposal["value"]]
        if len(value) != 3 or any(not math.isfinite(axis) or axis < 0 for axis in value):
            raise traitlets.TraitError(f"voxel_size_A must be 3 finite values >= 0, got {value}")
        return value

    @traitlets.validate("slice_stretch")
    def _validate_slice_stretch(self, proposal: dict) -> float:
        """Clamp to [1, 50]: below 1 the depth panel collapses, above 50 it fills the page."""
        value = float(proposal["value"])
        if not math.isfinite(value):
            raise traitlets.TraitError(f"slice_stretch must be finite, got {value}")
        return max(1.0, min(value, 50.0))

    @traitlets.validate("active_panel")
    def _validate_active_panel(self, proposal: dict) -> int:
        """Clamp the active panel into the flattened panels."""
        return max(0, min(int(proposal["value"]), max(0, int(self.panel_count) - 1)))

    @traitlets.validate("page_idx")
    def _validate_page_idx(self, proposal: dict) -> int:
        """Clamp the page into the pages."""
        return max(0, min(int(proposal["value"]), max(0, int(self.n_pages) - 1)))

    @traitlets.validate("panel_titles")
    def _validate_panel_titles(self, proposal: dict) -> list[str]:
        """One title per flattened panel, or none."""
        value = [str(title) for title in proposal["value"]]
        if len(value) not in (0, int(self.panel_count)):
            raise traitlets.TraitError(f"panel_titles must be empty or length panel_count={self.panel_count}, got {len(value)}")
        return value

    @traitlets.validate("page_labels")
    def _validate_page_labels(self, proposal: dict) -> list[str]:
        """One label per page, or none."""
        value = [str(label) for label in proposal["value"]]
        if len(value) not in (0, int(self.n_pages)):
            raise traitlets.TraitError(f"page_labels must be empty or length n_pages={self.n_pages}, got {len(value)}")
        return value

    @traitlets.validate("dim_labels")
    def _validate_dim_labels(self, proposal: dict) -> list[str]:
        """Exactly 3 strings; a bare string would silently become single letters."""
        raw = proposal["value"]
        if isinstance(raw, (str, bytes)) or len(raw) != 3 or not all(isinstance(label, str) for label in raw):
            raise traitlets.TraitError(f"dim_labels must be a list of 3 strings, got {raw!r}")
        return list(raw)

    @traitlets.validate("slice_index", "row_index", "col_index")
    def _validate_cut_index(self, proposal: dict) -> int:
        """Clamp a cut index into its axis of ``volume_shape``, so a cut never leaves the volume."""
        size = self.volume_shape[("slice_index", "row_index", "col_index").index(proposal["trait"].name)]
        return max(0, min(int(proposal["value"]), max(0, int(size) - 1)))

    _validate_vmin_vmax = traitlets.validate("vmin", "vmax")(validate_display_limits)

    def __init__(
        self,
        data,
        *,
        title: str = "",
        panel_titles: Sequence[str] | None = None,
        page_labels: Sequence[str | None] | None = None,
        show_title: bool = True,
        cmap: str = "plasma",
        sampling: float | Sequence[float] | None = None,
        units: str | Sequence[str] | None = None,
        show_scale_bar: bool = True,
        slice_stretch: float = 30.0,
        panel_width_px: int = 0,
        show_fft: bool = False,
        fft_window: bool = False,
        smooth: bool = False,
        auto_contrast: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        image_vmin_pct: float = 0.0,
        image_vmax_pct: float = 100.0,
        slice_alignment: str = "off",
        dim_labels: Sequence[str] | None = None,
        offline: bool = False,
    ):
        """Flatten the volume to ``(panel, slices, rows, cols)`` float32 and send it with the display traits in one sync."""
        super().__init__()
        # free() and the export clone read these before any volume is stored
        self._data: np.ndarray | None = None
        self._slice_alignment_estimates: dict[int, dict] = {}
        self._syncing_plane_visibility = False
        self._syncing_pages = False
        is_dataset = core_adapter.is_dataset(data, ndim=3)
        recorded = _voxel_size_A(list(data.sampling)[-3:], list(data.units)[-3:]) if is_dataset else [0.0, 0.0, 0.0]
        if sampling is None and not any(recorded):
            # a unit stated with no step, for a volume that records none on any axis, raises as in every image widget
            image_axes(None, units)
        voxel_size_A = [0.0, 0.0, 0.0] if sampling is None else _voxel_size_A(sampling, "angstrom" if units is None else units)
        if is_dataset:
            title = title or data.name
            # the dataset is the source of its voxel size, axis by axis: a keyword gives the step of an
            # axis the dataset does not calibrate (the frame axis of a stack) and may repeat one it does
            voxel_size_A = [step or 0.0 for step in stated("sampling", [step or None for step in voxel_size_A], [step or None for step in recorded])]
            data = core_adapter.as_array(data)
        data = to_numpy(data)
        if 0 in data.shape:
            raise ValueError(f"Empty volume: shape {data.shape}. All dims must be >= 1.")
        if np.iscomplexobj(data):
            raise TypeError("Show3DSlices does not accept complex data. Pass np.abs(arr) or np.angle(arr).")
        data, panel_titles, n_pages, panels_per_page, page_labels = _flatten_pages(data, panel_titles, page_labels)
        with np.errstate(over="ignore", invalid="ignore"):
            volume = np.ascontiguousarray(data, dtype=np.float32)
        if not np.isfinite(volume).all():
            raise ValueError("Data contains NaN, inf, or values beyond float32 range. Clean with np.nan_to_num first.")
        self._data = volume
        with self.hold_sync():
            self.panel_count = volume.shape[0]
            self.volume_shape = list(volume.shape[1:])
            self.n_pages = n_pages
            self.panels_per_page = panels_per_page
            self.page_labels = page_labels
            self.panel_titles = panel_titles
            self.slice_index, self.row_index, self.col_index = (size // 2 for size in self.volume_shape)
            self.title = str(title)
            self.show_title = bool(show_title)
            self.cmap = cmap
            self.voxel_size_A = voxel_size_A
            self.show_scale_bar = bool(show_scale_bar)
            self.slice_stretch = float(slice_stretch)
            self.panel_width_px = int(panel_width_px)
            self.show_fft = bool(show_fft)
            self.fft_window = bool(fft_window)
            self.smooth = bool(smooth)
            self.auto_contrast = bool(auto_contrast)
            self.vmin = vmin
            self.vmax = vmax
            self.image_vmin_pct = float(image_vmin_pct)
            self.image_vmax_pct = float(image_vmax_pct)
            self.slice_alignment = slice_alignment
            self.slice_alignment_cached = self.slice_alignment != "off"
            if dim_labels is not None:
                self.dim_labels = list(dim_labels)
            self.offline = bool(offline)
            if self.offline:
                # uint8 against the global range: the browser rescales with _offline_min/_max
                print(f"Show3DSlices packs the {'x'.join(map(str, volume.shape[1:]))} volume as uint8 over its range "
                      "(256 levels); offline=False, or export_html(encoding='full'), keeps float32.")
                low, high = float(volume.min()), float(volume.max())
                self._offline_min, self._offline_max = low, high
                if high > low:
                    self.volume_bytes = np.clip(np.rint((volume - low) * (255.0 / (high - low))), 0, 255).astype(np.uint8).tobytes()
                else:
                    self.volume_bytes = np.zeros(volume.shape, dtype=np.uint8).tobytes()
            else:
                self.volume_bytes = volume.tobytes()
        if self.slice_alignment == "auto":
            self.estimate_slice_alignment()
        self.observe(self._on_page_change, names=["page_idx"])
        self.observe(self._on_active_panel_change, names=["active_panel"])
        self.observe(self._on_export_request_change, names=["export_request"])
        self.observe(self._on_show_slice_planes_change, names=["show_slice_planes"])
        self.observe(self._on_plane_visibility_change, names=["plane_visibility"])
        self.observe(self._on_slice_alignment_request_change, names=["_slice_alignment_request"])

    def __repr__(self) -> str:
        """Pages, panels, volume shape, active page and panel, cut indices and colormap on one line."""
        pages = f"{self.n_pages} pages × " if int(self.n_pages) > 1 else ""
        panels = f"{self.panels_per_page or self.panel_count} panels × " if int(self.panels_per_page or self.panel_count) > 1 else ""
        return (
            f"Show3DSlices({pages}{panels}{'×'.join(map(str, self.volume_shape))}, page={self.page_idx}, "
            f"active_panel={self.active_panel}, slices=({self.slice_index},{self.row_index},{self.col_index}), cmap={self.cmap})"
        )

    # --- slice alignment ----------------------------------------------------------

    def estimate_slice_alignment(self, *, panel: int | None = None, apply: bool = True, force: bool = False) -> dict:
        """Fit one global row/col display shift per slice for a panel.

        See ``alignment.estimate_global_slice_alignment`` for the method. The
        estimate is cached per panel so the Align toggle in the browser does not
        re-register the stack on every raw/aligned switch; ``force`` recomputes.
        With ``apply`` the slopes are written to ``row_shift_px_per_slice`` /
        ``col_shift_px_per_slice`` and ``slice_alignment`` becomes ``"auto"``.
        """
        if self._data is None:
            raise ValueError("Cannot estimate slice alignment after free(); rebuild the widget first.")
        panel_idx = max(0, min(int(self.active_panel if panel is None else panel), int(self.panel_count) - 1))
        cached = panel_idx in self._slice_alignment_estimates and not force
        if not cached:
            result = estimate_global_slice_alignment(self._data[panel_idx])
            result["panel"] = panel_idx
            self._slice_alignment_estimates[panel_idx] = result
        result = copy.deepcopy(self._slice_alignment_estimates[panel_idx])
        if apply:
            row, col = result["row_shift_px_per_slice"], result["col_shift_px_per_slice"]
            with self.hold_sync():
                self.row_shift_px_per_slice = row
                self.col_shift_px_per_slice = col
                self.slice_alignment = "auto"
                self.slice_alignment_cached = True
                self.slice_alignment_status = f"{'Cached' if cached else 'Aligned'} row {row:+.3f} px/slice, col {col:+.3f} px/slice"
        return result

    def reset_slice_alignment(self) -> Self:
        """Turn off post-alignment and discard the cached estimates."""
        self._slice_alignment_estimates.clear()
        with self.hold_sync():
            self.slice_alignment = "off"
            self.row_shift_px_per_slice = 0.0
            self.col_shift_px_per_slice = 0.0
            self.slice_alignment_cached = False
            self.slice_alignment_status = ""
        return self

    def _on_slice_alignment_request_change(self, change: dict) -> None:
        """Serve the browser's Align button: estimate for a panel, or reset."""
        raw = str(change.get("new") or "")
        if not raw:
            return
        try:
            payload = json.loads(raw)
            mode = str(payload.get("mode", "estimate")).lower()
            if mode == "reset":
                self.reset_slice_alignment()
            elif mode == "estimate":
                self.slice_alignment_status = "Estimating slice alignment..."
                self.estimate_slice_alignment(panel=payload.get("panel"), force=bool(payload.get("force", False)))
            elif mode != "clear":
                raise ValueError(f"unknown slice-alignment request mode {mode!r}")
        except (ValueError, TypeError, KeyError) as exc:
            # the toolbar shows the failure; raising would only reach the kernel log
            self.slice_alignment_status = f"Slice alignment failed: {exc}"
        finally:
            self._slice_alignment_request = ""

    # --- state --------------------------------------------------------------------

    def state_dict(self) -> dict:
        """JSON-serializable snapshot of every user-tunable trait (no pixels)."""
        state = {"_widget": "Show3DSlices"}
        for key in self._STATE_KEYS:
            value = getattr(self, key)
            state[key] = dict(value) if isinstance(value, dict) else list(value) if isinstance(value, list) else value
        return state

    def load_state_dict(self, state: dict | str | pathlib.Path) -> None:
        """Apply a ``state_dict`` snapshot, its envelope or a ``save`` file onto this widget's volume.

        Page topology comes from the data, so ``n_pages`` / ``panels_per_page``
        are not part of the state and labels of the wrong length are dropped.
        ``vmin`` / ``vmax`` are cleared first so either bound can land
        regardless of the current limits. A file of another ``schema`` or an
        unknown key raises before anything is applied.
        """
        state = dict(read_state(state, "Show3DSlices", ("_widget", *self._STATE_KEYS)))
        marker = state.pop("_widget", None)
        if marker is not None and marker != "Show3DSlices":
            raise ValueError(f"load_state_dict: state was saved from {marker!r}, not Show3DSlices.")
        if "page_labels" in state and len(state["page_labels"]) != int(self.n_pages):
            state.pop("page_labels")
        if "panel_titles" in state and len(state["panel_titles"]) not in (0, int(self.panel_count)):
            state.pop("panel_titles")
        if "vmin" in state or "vmax" in state:
            new_vmin = state.pop("vmin", self.vmin)
            new_vmax = state.pop("vmax", self.vmax)
            self.vmin = None
            self.vmax = None
            self.vmin = None if new_vmin is None else float(new_vmin)
            self.vmax = None if new_vmax is None else float(new_vmax)
        for key, value in state.items():
            setattr(self, key, value)

    def free(self) -> None:
        """Release the volume and its synced bytes.

        ``del widget`` alone does not free memory: traitlets observers pin the
        widget's refcount. Playback is stopped first so the browser's animation
        loop tears down before the buffer it reads goes away. Idempotent.
        """
        if self._data is None:
            return
        self.playing = False
        self._data = None
        self._slice_alignment_estimates.clear()
        self.slice_alignment_cached = False
        self.volume_bytes = b""
        gc.collect()

    # --- pages ----------------------------------------------------------------------

    def _on_page_change(self, change: dict) -> None:
        """Move the active flattened panel to the same slot on the new page."""
        if self._syncing_pages or int(self.n_pages) <= 1 or int(self.panels_per_page) <= 0:
            return
        per_page = int(self.panels_per_page)
        slot = max(0, min(int(self.active_panel) - int(change["old"]) * per_page, per_page - 1))
        self._syncing_pages = True
        self.active_panel = int(change["new"]) * per_page + slot
        self._syncing_pages = False

    def _on_active_panel_change(self, change: dict) -> None:
        """Keep ``page_idx`` in step when a caller selects an absolute panel."""
        if self._syncing_pages or int(self.n_pages) <= 1 or int(self.panels_per_page) <= 0:
            return
        target_page = int(change["new"]) // int(self.panels_per_page)
        if target_page == int(self.page_idx):
            return
        self._syncing_pages = True
        self.page_idx = target_page
        self._syncing_pages = False

    def _on_plane_visibility_change(self, change: dict) -> None:
        """The 3D view's master toggle mirrors "any plane visible"."""
        if self._syncing_plane_visibility:
            return
        visible = any(bool(plane) for plane in change["new"])
        if self.show_slice_planes == visible:
            return
        self._syncing_plane_visibility = True
        self.show_slice_planes = visible
        self._syncing_plane_visibility = False

    def _on_show_slice_planes_change(self, change: dict) -> None:
        """Turning the master toggle on restores both planes; off hides both."""
        if self._syncing_plane_visibility:
            return
        visible = bool(change["new"])
        if visible and any(bool(plane) for plane in self.plane_visibility):
            return
        self._syncing_plane_visibility = True
        self.plane_visibility = [visible, visible]
        self._syncing_plane_visibility = False

    # --- HTML export ----------------------------------------------------------------

    _HTML_ENCODINGS = ("full", "uint8")

    def export_html(self, path: str | pathlib.Path | None = None, *, title: str | None = None, encoding: str = "full") -> pathlib.Path:
        """Write a standalone HTML viewer that opens with no kernel and no network.

        ``encoding="full"`` embeds the exact float32 volume; ``"uint8"`` packs
        it against the global range for a file about a quarter the size. The
        default path is the kernel cwd with the title, volume shape and
        encoding in the name.
        """
        if self._data is None:
            raise ValueError("Cannot export HTML after free(); rebuild the widget first.")
        return self._write_export_and_report(path, title=title, **self._html_export_choice(encoding, None))

    def _default_html_export_path(self, quantized: bool) -> pathlib.Path:
        """File in the kernel cwd named after the title, volume shape and encoding (``HtmlExportMixin`` hook)."""
        slug = export_slug(self.title, "show3dslices")
        mode = "quantized" if quantized else "exact"
        return pathlib.Path.cwd() / f"{slug}_{'x'.join(map(str, self.volume_shape))}_{mode}.html"

    def _clone_for_html_export(self, *, quantized: bool) -> Self:
        """An export-only copy with the current display state and the requested packing."""
        clone_data = self._data
        if int(self.n_pages) > 1:
            clone_data = self._data.reshape(int(self.n_pages), int(self.panels_per_page), *self.volume_shape)
        clone = type(self)(
            clone_data,
            title=self.title,
            panel_titles=list(self.panel_titles) or None,
            page_labels=list(self.page_labels) if int(self.n_pages) > 1 else None,
            sampling=list(self.voxel_size_A),
            units="A",
            dim_labels=list(self.dim_labels),
            offline=quantized,
        )
        clone.load_state_dict(self.state_dict())
        clone.export_enabled = False
        clone._export_light = True
        return clone

    def _release_export_clone(self, clone) -> None:
        """Free the clone's volume copy (observers pin a widget, so ``close`` alone would keep it), then close its model."""
        clone.free()
        clone.close()
