"""ShowPDF: interactive pair distribution function analysis of 4D-STEM data.

The widget selects scan positions (a painted mask, a square probe or a line band) and shows
the curves that quantem's PairDistributionFunction computes for them.
"""

import base64
import csv
import datetime
import inspect
import io
import json
import math
import pathlib
import time
import warnings
import zipfile
from typing import Self

import anywidget
import numpy as np
import traitlets

from quantem.widget.export import HtmlExportMixin, ensure_mobile_viewport, export_slug
from quantem.widget.fallback import StaticFallbackMixin
from quantem.widget.state import SavedStateMixin
from quantem.widget.utils.array import to_numpy
from quantem.widget.utils.state_io import resolve_widget_version, unwrap_state_payload

_PDF_REQUIREMENT = (
    "ShowPDF needs quantem's PairDistributionFunction with calculate_Gr(density=..., "
    "max_iter=...), from electronmicroscopy/quantem pull request 248."
)

# Removed constructor keywords and what to use instead; passing one raises TypeError with
# the hint instead of being silently ignored by traitlets.
_REMOVED_KWARGS = {
    "density_mode": "the density is never estimated; pass density_value (atoms/Å³) instead",
    "line_max_bins": "renamed to linescan_max_bins",
    "_offline": "renamed to offline, which only exported HTML files set",
    "theme": "removed; the widget follows the notebook theme",
    "disabled_tools": "removed",
    "hidden_tools": "removed",
    "k_width": "removed; the k window's taper width is fixed",
    "r_min": "removed; the r grid always starts at 0",
}

# Status hints for an output that needs the number density, which is never estimated.
_DENSITY_HINT_DAMPING = "Set the density to apply damping and compute g(r)."
_DENSITY_HINT_GR = "Set the density to compute g(r)."
_DENSITY_HINTS = (_DENSITY_HINT_DAMPING, _DENSITY_HINT_GR)
# Status hint for a k window with no positive weight, where F(k) and G(r) are zero (g(r) = 1).
# It takes precedence over the density hints.
_WINDOW_HINT = (
    "The k window contains no k points; set its lower end below its upper end, "
    "more than one k step apart."
)
_LINE_WIDTH_ERROR = "line_width must be an odd number of scan pixels (1, 3, 5, ...), got {!r}."
# Bound on the line ends in scan pixels. No scan comes close; ends beyond about 1e154 would
# overflow the length of the perpendicular bisector.
_LINE_END_LIMIT_TEXT = "1e9"  # the spelling users see, as in the docs
_LINE_END_LIMIT = float(_LINE_END_LIMIT_TEXT)
_LINE_END_ERROR = f"{{}} must be a finite number within ±{_LINE_END_LIMIT_TEXT}, got {{!r}}."
# Each plot_mode's synced curve, as (x trait, y trait), and its axis labels.
_CURVE_TRAITS = {
    "Ik": ("ik_x_bytes", "ik_y_bytes"),
    "Fk": ("fk_x_bytes", "fk_y_bytes"),
    "Gr": ("gr_x_bytes", "gr_y_bytes"),
    "gr": ("pdf_x_bytes", "pdf_y_bytes"),
}
_CURVE_AXIS_LABELS = {"Ik": ("k (1/Å)", "I(k)"), "Fk": ("k (1/Å)", "F(k)"), "Gr": ("r (Å)", "G(r)"), "gr": ("r (Å)", "g(r)")}
# What the saved-notebook preview shows; a change to any of these redraws it at the next save.
_PREVIEW_TRAITS = (
    "title", "plot_mode", "cmap", "auto_contrast", "nav_image_bytes", "linescan_bytes",
    *(y for _, y in _CURVE_TRAITS.values()),
)


# =============================================================================
# Input and saved state
# =============================================================================
def _pdf_class():
    """quantem's PairDistributionFunction, imported on first use.

    quantem core is optional for quantem.widget (it does not install on an Intel Mac or
    Windows on ARM), so this module imports without it; ShowPDF raises ImportError naming
    the requirement.
    """
    try:
        from quantem.diffraction import PairDistributionFunction
    except ImportError as exc:
        raise ImportError(_PDF_REQUIREMENT) from exc
    gr_params = inspect.signature(PairDistributionFunction.calculate_Gr).parameters
    if not {"density", "max_iter"} <= set(gr_params):
        raise ImportError(_PDF_REQUIREMENT)
    return PairDistributionFunction


def _as_dataset(data):
    """Datasets pass through; 4D and 2D arrays are wrapped for PairDistributionFunction.from_data."""
    if hasattr(data, "array"):
        return data
    from quantem.core.datastructures import Dataset2d, Dataset4dstem

    arr = to_numpy(data)
    if arr.ndim == 4:
        return Dataset4dstem.from_array(arr)
    if arr.ndim == 2:
        return Dataset2d.from_array(arr)
    raise ValueError(f"ShowPDF expects 4D (4D-STEM) or 2D input; got {arr.ndim}D array.")


def _load_pdf(data, **from_data_kwargs):
    """``data`` as a PairDistributionFunction, and the title it carries ("" if none).

    Other inputs go through ``from_data`` with ``device`` "cpu" unless given: the library
    computes in float64, which MPS does not support and other GPU drivers handle unevenly,
    and the compute is light.
    """
    pdf_class = _pdf_class()
    if isinstance(data, pdf_class):
        return data, ""
    title = ""
    if hasattr(data, "title") and hasattr(data, "data"):  # an IOResult
        title = data.title or ""
        data = data.data
    if hasattr(data, "array") and hasattr(data, "name"):  # a Dataset
        title = title or (data.name or "")
    if from_data_kwargs.get("device") is None:
        from_data_kwargs["device"] = "cpu"
    return pdf_class.from_data(_as_dataset(data), **from_data_kwargs), title


def _read_state(state):
    """The state dict of a ``state=`` argument: a dict, or a file written by ``save``."""
    if isinstance(state, (str, pathlib.Path)):
        payload = json.loads(pathlib.Path(state).read_text())
        return unwrap_state_payload(payload, require_envelope=True, expected_widget="ShowPDF")
    return unwrap_state_payload(state, expected_widget="ShowPDF")


def _migrate_state(state):
    """A copy of a saved state with the values of earlier versions made loadable.

    Returns (state, notice): notice is the warning for a migrated even line_width, else None.
    """
    state = dict(state)
    if state.get("density_mode") == "estimated":
        # This density_value was an automatic estimate, never supplied by the user, so it
        # must not become the fixed density for g(r) and the damping.
        state.pop("density_value", None)
    density = state.get("density_value")
    if isinstance(density, float) and not math.isfinite(density):
        # NaN meant "not set" (save writes it as a NaN literal), so it loads as 0. The
        # validator rejects a non-finite density everywhere else.
        state["density_value"] = 0.0
    notice = None
    width = state.get("line_width")
    if isinstance(width, int):
        # Any width was allowed: an even width w selected w + 1 rows of a horizontal line and
        # a width below 1 acted as 1, so load the odd width that selects the same rows.
        if width < 1:
            state["line_width"] = 1
        elif width % 2 == 0:
            state["line_width"] = width + 1
            notice = (
                f"The saved line_width {width} is even; it loads as line_width {width + 1}, "
                "which selects the same positions of a horizontal or vertical line. "
                "A diagonal band may include more positions than before."
            )
    return state, notice


def _reject_unknown_kwargs(cls, kwargs):
    """Raise TypeError for a keyword that is not a trait, which traitlets would silently ignore."""
    traits = set(cls.class_trait_names())
    unknown = sorted(k for k in kwargs if k not in traits)
    if unknown:
        key = unknown[0]
        hint = _REMOVED_KWARGS.get(key, "check for typos or a renamed parameter")
        raise TypeError(f"{cls.__name__}() got unexpected keyword argument {key!r}: {hint}.")


def _is_whole_number(value):
    """True if int() converts ``value`` without truncating it (2.0 and np.int64(2), not 2.5).

    A value too large for a float is not a whole number here.
    """
    try:
        return float(value).is_integer()
    except OverflowError:
        return False


def _whole_number(name, value):
    """``value`` as an int for the Int trait ``name``; int() alone would truncate 2.5 to 2."""
    if not _is_whole_number(value):
        raise traitlets.TraitError(f"{name} must be a whole number, got {value!r}.")
    return int(value)


def _line_width_int(width):
    """``width`` as an int; a width that is not a whole number raises the validator's TraitError."""
    if not _is_whole_number(width):
        raise traitlets.TraitError(_LINE_WIDTH_ERROR.format(width))
    return int(width)


def _line_end_float(name, value):
    """``value`` as a float line end; NaN, infinity or a value beyond the limit raises ValueError."""
    try:
        end = float(value)
    except OverflowError:  # an int too large for a float is beyond the limit too
        raise ValueError(_LINE_END_ERROR.format(name, value)) from None
    if not math.isfinite(end) or abs(end) > _LINE_END_LIMIT:
        raise ValueError(_LINE_END_ERROR.format(name, end))
    return end


def _finite_pixel_size(pixel_size):
    """``pixel_size`` as a float, or None when it is not given.

    NaN or infinity raises ValueError: it is not valid JSON in the HTML export.
    """
    if pixel_size is None:
        return None
    value = float(pixel_size)
    if not math.isfinite(value):
        raise ValueError(f"pixel_size must be a finite number, got {value!r}.")
    return value


def _finite_k_axis(pdf):
    """The k axis of ``pdf`` as float32, the precision of the k traits.

    A NaN or infinite k (from a NaN or overflowing calibration) raises ValueError naming
    that cause, instead of rejecting a default fit range the user never passed.
    """
    qq = np.asarray(pdf.qq, dtype=np.float32)
    if not np.isfinite(qq).all():
        raise ValueError(
            "The dataset's k axis is not finite (or too large for float32); "
            "check the diffraction calibration."
        )
    return qq


# =============================================================================
# Arrays and line geometry
# =============================================================================
def _float32(values):
    """A NumPy, Torch or CuPy array as a new float32 NumPy array."""
    return to_numpy(values).astype(np.float32)


def _finite_range(img):
    """Min and max of the finite pixels of ``img``, or (0.0, 0.0) if it has none.

    NaN and infinity are not valid JSON in the HTML export; the frontend's findDataRange
    skips the same pixels.
    """
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return 0.0, 0.0
    return float(finite.min()), float(finite.max())


def _bf_nav_image(arr):
    """Bright-field virtual image of a 4D-STEM array for the navigation panel.

    The BF disk is found on the scan-summed pattern: threshold at mean + std, centroid for
    the centre and radius from the disk area (A = pi r^2), falling back to the detector
    centre with radius det_rows / 6 if nothing is above the threshold. Returns the mean
    intensity inside the disk at each scan position.
    """
    arr = np.asarray(arr, dtype=np.float32)
    det_rows, det_cols = arr.shape[-2], arr.shape[-1]
    summed = arr.reshape(-1, det_rows, det_cols).sum(axis=0)
    disk = summed > summed.mean() + summed.std()
    total = disk.sum()
    row_coords = np.arange(det_rows, dtype=np.float64).reshape(-1, 1)
    col_coords = np.arange(det_cols, dtype=np.float64).reshape(1, -1)
    if total > 0:
        center_row = float((row_coords * disk).sum() / total)
        center_col = float((col_coords * disk).sum() / total)
        bf_radius = float(np.sqrt(total / np.pi))
    else:
        center_row, center_col, bf_radius = det_rows / 2.0, det_cols / 2.0, det_rows / 6.0
    rad = np.sqrt((row_coords - center_row) ** 2 + (col_coords - center_col) ** 2)
    bf_mask = (rad < bf_radius).astype(np.float32)
    return (arr * bf_mask).mean(axis=(-2, -1)).astype(np.float32)


def _round_half_up(value):
    """Nearest whole scan coordinate, halves rounded up as Math.round does in the frontend."""
    return float(math.floor(value + 0.5))


def _perpendicular_bisector(r0, c0, r1, c1):
    """Ends of the segment's perpendicular bisector: same midpoint and length, turned 90°."""
    dr, dc = r1 - r0, c1 - c0
    # sqrt, not hypot, in the frontend's order of operations: both round the same, so an end
    # half way between scan positions snaps the same way in Python and in the browser.
    length = math.sqrt(dr * dr + dc * dc)
    if length > 0:
        mr, mc = (r0 + r1) / 2.0, (c0 + c1) / 2.0
        pr, pc = -dc / length, dr / length  # unit vector perpendicular to the segment
        half = length / 2.0
        return mr - pr * half, mc - pc * half, mr + pr * half, mc + pc * half
    return r0, c0, r1, c1  # a point has no direction


def _clip_to_scan(r0, c0, r1, c1, rows, cols):
    """Liang-Barsky clip of a segment to a rows x cols scan, in the frontend's arithmetic.

    The clipped ends are rounded to scan positions again; a segment wholly outside the
    scan collapses to the point (r0, c0).
    """
    dr, dc = r1 - r0, c1 - c0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dr, r0), (dr, rows - 1 - r0), (-dc, c0), (dc, cols - 1 - c0)):
        if p == 0:
            if q < 0:
                return r0, c0, r0, c0
        elif p < 0:
            t0 = max(t0, q / p)
        else:
            t1 = min(t1, q / p)
    if t0 > t1:
        return r0, c0, r0, c0
    return (
        _round_half_up(r0 + t0 * dr),
        _round_half_up(c0 + t0 * dc),
        _round_half_up(r0 + t1 * dr),
        _round_half_up(c0 + t1 * dc),
    )


def _nan_stack(n_rows, n_cols):
    """A float32 (n_rows, n_cols) stack of NaN: rows that get no curve stay NaN."""
    return np.full((n_rows, n_cols), np.nan, dtype=np.float32)


def _fill_row(stack, row, values):
    """Copy ``values`` into ``stack[row]``, cut to the stack's width."""
    n = min(len(values), stack.shape[1])
    stack[row, :n] = values[:n]


class ShowPDF(SavedStateMixin, HtmlExportMixin, StaticFallbackMixin, anywidget.AnyWidget):
    """Interactive pair distribution function (PDF) analysis of 4D-STEM data.

    Wraps :class:`quantem.diffraction.PairDistributionFunction`. The left panel
    shows a bright-field navigation image on which scan positions are selected
    (painted mask, square probe, or line band). The right panel shows the curves
    for the selection: I(k) with the fitted background B(k), the windowed reduced
    structure factor F(k), the reduced PDF G(r), and the PDF g(r).

    The k axis is in 1/Å as 1/d (the 2π enters inside the sine transform). The
    number density is never estimated: g(r) and the low-r damping need an
    explicit ``density_value``.

    Parameters
    ----------
    data : PairDistributionFunction, Dataset4dstem, Dataset2d, or array-like
        A prepared ``PairDistributionFunction``, or data it is built from with
        ``PairDistributionFunction.from_data``. Arrays must be 4D
        ``(scan_row, scan_col, det_row, det_col)`` or a single 2D pattern.
    nav_image : array-like, optional
        Navigation image of shape ``(scan_row, scan_col)``. Defaults to a
        bright-field virtual image of the input data.
    title : str, optional
        Widget title. Defaults to the dataset name, else ``"PDF"``.
    k_min_fit, k_max_fit : float, default 0.0
        k range (1/Å) of the background fit. 0 means the full data range.
    k_min_window, k_max_window : float, default 0.0
        k range (1/Å) of the window applied to F(k) before the transform.
        0 (or less) means "follow the fit range", as ``None`` in ``calculate_Gr``.
    k_lowpass, k_highpass : float, default 0.0
        Gaussian low-pass and high-pass filter widths on F(k), in 1/Å. 0 disables.
    r_max, r_step : float, default 20.0, 0.02
        Real-space grid (Å) of G(r) and g(r), which always starts at r = 0.
    damp_origin_oscillations : bool, default False
        Apply the low-r correction of ``PairDistributionFunction.damp_origin_oscillations``,
        which forces G(r) onto -4π ρ₀ r on [0, r_cut]. Needs ``density_value``.
    r_cut : float, default 0.8
        Upper end r (Å) of the low-r correction. Set it below the nearest-neighbour
        distance.
    max_iter : int, default 6
        Maximum number of iterations of the low-r correction.
    density_value : float, default 0.0
        Atomic number density ρ₀ (atoms/Å³) used for g(r) and the damping.
        0 means "not set": g(r) is left empty and damping is skipped.
    analysis_mode : {"mask", "probe", "line"}, default "mask"
        How scan positions are selected.
    probe_row, probe_col : int, optional
        Probe centre (row, col). Defaults to the scan centre.
    probe_size : int, default 1
        The probe selects a square of side ``2 * probe_size - 1`` positions.
    line_row0, line_col0, line_row1, line_col1 : float, default 0.0
        Line endpoints (row, col) in scan pixels; equal endpoints mean no line.
    line_width : int, default 3
        Width of the line band, an odd number of scan pixels (1, 3, 5, ...).
        A horizontal or vertical band of width w covers exactly w rows or
        columns (clipped at the scan edge).
    line_mode : {"averaged", "linescan"}, default "averaged"
        One PDF from the whole band, or one PDF per position along the line.
    line_perpendicular : bool, default False
        Sample along the perpendicular bisector of the drawn line instead.
    linescan_max_bins : int, default 256
        Maximum number of positions along the line in line-scan mode. The
        positions are evenly spaced, one per whole scan-pixel step along the
        line's longer axis, unless this cap is lower.
    linescan_view : {"heatmap", "waterfall"}, default "heatmap"
        Rendering of the line-scan stack.
    waterfall_count : int, default 10
        Number of curves shown in the waterfall view, evenly spaced along the
        line and ending at its tip; the shown count is the closest to this that
        keeps the spacing exact.
    plot_mode : {"Ik", "Fk", "Gr", "gr"}, default "Gr"
        Curve shown in the right panel.
    show_background, cmap, log_scale, auto_contrast, show_stats, show_controls
        Display options.
    find_origin, origin_row, origin_col, num_annular_bins, radial_min, radial_max, radial_step, two_fold_rotation_symmetry
        Passed to ``PairDistributionFunction.from_data`` when ``data`` is not
        already a ``PairDistributionFunction``.
    device : str, optional
        Torch device of the PDF computation. Defaults to ``"cpu"``.
    pixel_size : float, optional
        Scan pixel size in Å for the scale bar; overrides the dataset calibration.
    save_state : bool, default False
        Keep the scan image, mask and line-scan stack in a saved notebook, so that it
        reopens interactive without a kernel. By default a saved notebook keeps a static
        picture of the widget instead.
    state : dict, str, or pathlib.Path, optional
        State dictionary, or a file written by :meth:`save`, to restore.
    **kwargs
        Other widget traits. Unknown keywords raise ``TypeError``.
    """

    _esm = pathlib.Path(__file__).parent / "static" / "showpdf.js"
    # A notebook saved with save_state=False keeps this static picture instead of these traits.
    _static_fallback_jpeg = traitlets.Unicode("").tag(sync=True)
    _static_fallback_mime = traitlets.Unicode("image/png").tag(sync=True)
    _UNSAVED_HEAVY_KEYS = ("nav_image_bytes", "mask_b64", "mask_bytes", "linescan_bytes", "export_payload")

    # =========================================================================
    # Version
    # =========================================================================
    widget_version = traitlets.Unicode("unknown").tag(sync=True)

    # =========================================================================
    # Shape and data (read-only from JS)
    # =========================================================================
    title = traitlets.Unicode("PDF").tag(sync=True)
    scan_rows = traitlets.Int(1).tag(sync=True)
    scan_cols = traitlets.Int(1).tag(sync=True)
    nav_image_bytes = traitlets.Bytes(b"").tag(sync=True)
    nav_data_min = traitlets.Float(0.0).tag(sync=True)
    nav_data_max = traitlets.Float(1.0).tag(sync=True)
    # Scan calibration of the nav scale bar: the dataset's first axis in Å (nm converted),
    # else 1 px.
    nav_pixel_size = traitlets.Float(1.0).tag(sync=True)
    nav_unit = traitlets.Unicode("px").tag(sync=True)

    # =========================================================================
    # Painted mask (JS paints, Python reads): 1 includes a scan position and
    # 0 excludes it, as in calculate_radial_mean
    # =========================================================================
    mask_bytes = traitlets.Bytes(b"").tag(sync=True)  # raw uint8 copy of mask_b64 (Python only)
    mask_b64 = traitlets.Unicode("").tag(sync=True)  # the mask from JS, base64 uint8
    mask_version = traitlets.Int(0).tag(sync=True)  # JS increments it to trigger a recompute
    mask_tool = traitlets.Unicode("rectangle").tag(sync=True)
    mask_brush_size = traitlets.Int(5).tag(sync=True)
    mask_pixel_count = traitlets.Int(0).tag(sync=True)
    mask_fraction = traitlets.Float(0.0).tag(sync=True)

    # =========================================================================
    # Analysis mode: "mask" (the painted mask), "probe" (a square of side
    # 2 * probe_size - 1 around probe_row, probe_col) or "line" (a band)
    # =========================================================================
    analysis_mode = traitlets.Unicode("mask").tag(sync=True)
    probe_row = traitlets.Int(0).tag(sync=True)
    probe_col = traitlets.Int(0).tag(sync=True)
    probe_size = traitlets.Int(1).tag(sync=True)

    # =========================================================================
    # Line band: the scan positions within line_width / 2 of the segment
    # (line_row0, line_col0)-(line_row1, line_col1). line_mode is "averaged"
    # (one PDF from the band) or "linescan" (one PDF per position).
    # =========================================================================
    line_row0 = traitlets.Float(0.0).tag(sync=True)
    line_col0 = traitlets.Float(0.0).tag(sync=True)
    line_row1 = traitlets.Float(0.0).tag(sync=True)
    line_col1 = traitlets.Float(0.0).tag(sync=True)
    line_width = traitlets.Int(3).tag(sync=True)
    line_active = traitlets.Bool(False).tag(sync=True)  # False until a line is drawn
    line_mode = traitlets.Unicode("averaged").tag(sync=True)
    # JS bumps it once per edit of the ends, so a drag triggers a single recompute.
    line_version = traitlets.Int(0).tag(sync=True)
    # Sample along the perpendicular bisector of the drawn line, so a line traced along a
    # feature scans across it.
    line_perpendicular = traitlets.Bool(False).tag(sync=True)

    # Line-scan stack (line_mode="linescan"): float32 (n_linescan, linescan_n_points),
    # row-major by position, on the shared r or k axis of the plot_mode curve. Empty until
    # a line is drawn.
    linescan_bytes = traitlets.Bytes(b"").tag(sync=True)
    n_linescan = traitlets.Int(0).tag(sync=True)
    linescan_n_points = traitlets.Int(0).tag(sync=True)
    linescan_axis_bytes = traitlets.Bytes(b"").tag(sync=True)
    linescan_max_bins = traitlets.Int(256).tag(sync=True)
    linescan_view = traitlets.Unicode("heatmap").tag(sync=True)  # "heatmap" or "waterfall"
    # Curves stacked in the waterfall view, evenly spaced along the line; the heatmap shows all.
    waterfall_count = traitlets.Int(10).tag(sync=True)

    # =========================================================================
    # Curves (float32 bytes, all updated on every recompute)
    # =========================================================================
    # I(k) radial mean and its fitted background B(k)
    ik_x_bytes = traitlets.Bytes(b"").tag(sync=True)
    ik_y_bytes = traitlets.Bytes(b"").tag(sync=True)
    ik_bg_y_bytes = traitlets.Bytes(b"").tag(sync=True)
    n_points_ik = traitlets.Int(0).tag(sync=True)
    # F(k) after the k window, the curve the sine transform uses (damped when the low-r
    # correction ran)
    fk_x_bytes = traitlets.Bytes(b"").tag(sync=True)
    fk_y_bytes = traitlets.Bytes(b"").tag(sync=True)
    n_points_fk = traitlets.Int(0).tag(sync=True)
    # G(r) reduced PDF
    gr_x_bytes = traitlets.Bytes(b"").tag(sync=True)
    gr_y_bytes = traitlets.Bytes(b"").tag(sync=True)
    n_points_gr = traitlets.Int(0).tag(sync=True)
    # g(r) pair distribution function
    pdf_x_bytes = traitlets.Bytes(b"").tag(sync=True)
    pdf_y_bytes = traitlets.Bytes(b"").tag(sync=True)
    n_points_pdf = traitlets.Int(0).tag(sync=True)

    # =========================================================================
    # Standalone export data, set in the exported state only. offline_qq is the
    # exact k axis; with encoding="uint8" the per-position radial profiles
    # (mean over phi, uint8 per k bin) let the page recompute I(k) for any mask,
    # probe or line selection without a kernel.
    # =========================================================================
    offline_recompute = traitlets.Bool(False).tag(sync=True)
    offline_nk = traitlets.Int(0).tag(sync=True)
    offline_rows = traitlets.Int(0).tag(sync=True)
    offline_cols = traitlets.Int(0).tag(sync=True)
    offline_profiles = traitlets.Bytes(b"").tag(sync=True)  # uint8 (rows*cols*nk)
    offline_profiles_lo = traitlets.Bytes(b"").tag(sync=True)  # float32 (nk,)
    offline_profiles_hi = traitlets.Bytes(b"").tag(sync=True)  # float32 (nk,)
    offline_qq = traitlets.Bytes(b"").tag(sync=True)  # float64 (nk,) k axis

    # =========================================================================
    # PDF parameters (user-tunable, trigger recompute)
    # 0.0 sentinel means "auto" or "disabled"
    # =========================================================================
    k_min_fit = traitlets.Float(0.0).tag(sync=True)
    k_max_fit = traitlets.Float(0.0).tag(sync=True)
    k_min_window = traitlets.Float(0.0).tag(sync=True)
    k_max_window = traitlets.Float(0.0).tag(sync=True)
    k_lowpass = traitlets.Float(0.0).tag(sync=True)
    k_highpass = traitlets.Float(0.0).tag(sync=True)
    r_max = traitlets.Float(20.0).tag(sync=True)
    r_step = traitlets.Float(0.02).tag(sync=True)
    damp_origin_oscillations = traitlets.Bool(False).tag(sync=True)
    r_cut = traitlets.Float(0.8).tag(sync=True)
    max_iter = traitlets.Int(6).tag(sync=True)

    # Number density (atoms/Å³) for g(r) and the low-r damping; 0.0 means not set, since
    # PairDistributionFunction does not estimate it.
    density_value = traitlets.Float(0.0).tag(sync=True)

    # k range of the data (set from the data, used for the slider bounds)
    k_min_available = traitlets.Float(0.0).tag(sync=True)
    k_max_available = traitlets.Float(10.0).tag(sync=True)

    # =========================================================================
    # Display
    # =========================================================================
    plot_mode = traitlets.Unicode("Gr").tag(sync=True)
    show_background = traitlets.Bool(True).tag(sync=True)
    cmap = traitlets.Unicode("gray").tag(sync=True)
    log_scale = traitlets.Bool(False).tag(sync=True)
    auto_contrast = traitlets.Bool(True).tag(sync=True)
    show_stats = traitlets.Bool(True).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)

    # Status
    computing = traitlets.Bool(False).tag(sync=True)
    status_message = traitlets.Unicode("").tag(sync=True)

    # True only in the standalone HTML export: there is no kernel, so the k-space sliders
    # recompute the curves in JS from the baked I(k). The export traits come from
    # HtmlExportMixin.
    offline = traitlets.Bool(False).tag(sync=True)

    @traitlets.validate("linescan_max_bins")
    def _validate_linescan_max_bins(self, proposal):
        value = int(proposal["value"])
        if value < 1:
            raise traitlets.TraitError(f"linescan_max_bins must be at least 1, got {value!r}.")
        return value

    @traitlets.validate("linescan_view")
    def _validate_linescan_view(self, proposal):
        value = proposal["value"]
        if value not in ("heatmap", "waterfall"):
            raise traitlets.TraitError(f"linescan_view must be 'heatmap' or 'waterfall', got {value!r}.")
        return value

    @traitlets.validate("waterfall_count")
    def _validate_waterfall_count(self, proposal):
        value = proposal["value"]
        if value < 1:
            raise traitlets.TraitError(f"waterfall_count must be at least 1, got {value!r}.")
        return value

    @traitlets.validate("line_width")
    def _validate_line_width(self, proposal):
        value = int(proposal["value"])
        try:
            float(value)  # the band test divides line_width by 2.0
        except OverflowError:
            raise traitlets.TraitError(_LINE_WIDTH_ERROR.format(value)) from None
        if value < 1 or value % 2 == 0:
            raise traitlets.TraitError(_LINE_WIDTH_ERROR.format(value))
        return value

    @traitlets.validate("line_row0", "line_col0", "line_row1", "line_col1")
    def _validate_line_end(self, proposal):
        # A NaN or infinite end has no band and is not valid JSON in an export; the limit
        # keeps the perpendicular bisector finite.
        value = float(proposal["value"])
        if not math.isfinite(value) or abs(value) > _LINE_END_LIMIT:
            raise traitlets.TraitError(_LINE_END_ERROR.format(proposal["trait"].name, value))
        return value

    @traitlets.validate(
        "k_min_fit", "k_max_fit", "k_min_window", "k_max_window", "k_lowpass", "k_highpass",
        "r_max", "r_step", "r_cut", "density_value",
    )
    def _validate_finite(self, proposal):
        # NaN or infinity cannot be computed with and is not valid JSON in an export;
        # 0 still means unset.
        value = float(proposal["value"])
        if not math.isfinite(value):
            raise traitlets.TraitError(
                f"{proposal['trait'].name} must be a finite number, got {value!r}."
            )
        return value

    # =========================================================================
    # Constructor
    # =========================================================================
    def __init__(
        self,
        data,
        *,
        nav_image=None,
        title="",
        # PDF parameters
        k_min_fit=0.0,
        k_max_fit=0.0,
        k_min_window=0.0,
        k_max_window=0.0,
        k_lowpass=0.0,
        k_highpass=0.0,
        r_max=20.0,
        r_step=0.02,
        damp_origin_oscillations=False,
        r_cut=0.8,
        max_iter=6,
        # Number density (atoms/A^3) for g(r) and damping; 0.0 = not set
        density_value=0.0,
        # Analysis mode + probe geometry
        analysis_mode="mask",
        probe_row=None,
        probe_col=None,
        probe_size=1,
        # Line geometry (analysis_mode="line")
        line_row0=0.0,
        line_col0=0.0,
        line_row1=0.0,
        line_col1=0.0,
        line_width=3,
        line_mode="averaged",
        line_perpendicular=False,
        linescan_max_bins=256,
        linescan_view="heatmap",
        waterfall_count=10,
        # Display
        plot_mode="Gr",
        show_background=True,
        cmap="gray",
        log_scale=False,
        auto_contrast=True,
        show_stats=True,
        show_controls=True,
        # PDF construction params (only used when data is not already a PDF)
        find_origin=True,
        origin_row=None,
        origin_col=None,
        num_annular_bins=180,
        radial_min=0.0,
        radial_max=None,
        radial_step=1.0,
        two_fold_rotation_symmetry=False,
        device=None,
        # Real-space scale bar (overrides auto-detect from ds.sampling/units)
        pixel_size=None,
        # State
        save_state=False,
        state=None,
        **kwargs,
    ):
        _reject_unknown_kwargs(type(self), kwargs)
        # Set before the widget opens, since opening it takes a full state snapshot.
        self._save_state = bool(save_state)
        self._configure_static_fallback(notebook_preview_format="png")
        super().__init__(**kwargs)
        self._initializing = True
        self.widget_version = resolve_widget_version()
        t0 = time.perf_counter()

        self._pdf, input_title = _load_pdf(
            data,
            find_origin=find_origin,
            origin_row=origin_row,
            origin_col=origin_col,
            num_annular_bins=num_annular_bins,
            radial_min=radial_min,
            radial_max=radial_max,
            radial_step=radial_step,
            two_fold_rotation_symmetry=two_fold_rotation_symmetry,
            device=device,
        )
        self._reset_pdf_cache()
        # The dataset (counted by set_data) and the scan positions that the synced single
        # curves were computed for; None while there are none.
        self._data_version = 0
        self._curves_selection = None
        self._pixel_size_override = None

        # --- Scan shape and k range of the data ---
        polar_shape = self._pdf.polar.shape
        self.scan_rows = int(polar_shape[0])
        self.scan_cols = int(polar_shape[1])
        qq = _finite_k_axis(self._pdf)
        self.k_min_available = float(qq[0])
        self.k_max_available = float(qq[-1])
        # The fit range defaults to the full data range. The window keeps its 0.0 sentinel,
        # so calculate_Gr follows the fit range.
        if k_min_fit == 0.0:
            k_min_fit = float(qq[0])
        if k_max_fit == 0.0:
            k_max_fit = float(qq[-1])

        # --- Parameter traits ---
        self.title = title or input_title or "PDF"
        self.k_min_fit = k_min_fit
        self.k_max_fit = k_max_fit
        self.k_min_window = k_min_window
        self.k_max_window = k_max_window
        self.k_lowpass = k_lowpass
        self.k_highpass = k_highpass
        self.r_max = r_max
        self.r_step = r_step
        self.damp_origin_oscillations = damp_origin_oscillations
        self.r_cut = r_cut
        self.max_iter = _whole_number("max_iter", max_iter)
        self.density_value = float(density_value)
        self.analysis_mode = analysis_mode
        if probe_row is None:
            probe_row = self.scan_rows // 2
        if probe_col is None:
            probe_col = self.scan_cols // 2
        self.probe_row = _whole_number("probe_row", probe_row)
        self.probe_col = _whole_number("probe_col", probe_col)
        self.probe_size = _whole_number("probe_size", probe_size)
        self.line_row0 = float(line_row0)
        self.line_col0 = float(line_col0)
        self.line_row1 = float(line_row1)
        self.line_col1 = float(line_col1)
        self.line_width = _line_width_int(line_width)
        self.line_mode = line_mode
        self.line_perpendicular = bool(line_perpendicular)
        self.linescan_max_bins = _whole_number("linescan_max_bins", linescan_max_bins)
        self.linescan_view = linescan_view
        self.waterfall_count = _whole_number("waterfall_count", waterfall_count)
        # A line is active once its ends differ.
        self.line_active = (line_row0, line_col0) != (line_row1, line_col1)

        # --- Display traits ---
        self.plot_mode = plot_mode
        self.show_background = show_background
        self.cmap = cmap
        self.log_scale = log_scale
        self.auto_contrast = auto_contrast
        self.show_stats = show_stats
        self.show_controls = show_controls

        self._set_nav_image(nav_image, pixel_size)
        self._register_observers()

        # --- Initial computation ---
        self._initializing = False
        self._update_mask_stats()
        self._recompute_full()
        elapsed = time.perf_counter() - t0
        if not self.status_message:  # keep an error or hint the compute set
            self.status_message = f"Ready ({elapsed:.1f}s)"

        if state is not None:
            self.load_state_dict(_read_state(state))

    # =========================================================================
    # Dataset and navigation image
    # =========================================================================
    def _reset_pdf_cache(self):
        """Drop the results cached on the PairDistributionFunction.

        calculate_Gr reuses a cached radial mean when the cached mask matches, so a cache from
        preprocessing or another dataset could bypass this widget's selection.
        """
        for attr in (
            "Ik", "_Ik_mask", "bg", "f", "Sk", "Fk", "Fk_masked", "Fk_damped",
            "rho0", "_reduced_pdf", "_r", "_pdf", "reduced_pdf_damped",
        ):
            if hasattr(self._pdf, attr):
                setattr(self._pdf, attr, None)

    def _compute_nav_image(self):
        """Bright-field image of the 4D input data, else the polar data summed over angle and radius."""
        input_data = getattr(self._pdf, "input_data", None)
        if input_data is not None and getattr(input_data, "ndim", None) == 4:
            return _bf_nav_image(input_data.numpy())
        return self._pdf.polar.numpy().sum(axis=(-2, -1)).astype(np.float32)

    def _resolve_nav_calibration(self):
        """(pixel_size, unit) of the scan from the dataset: in Å for an Å or nm axis, else (1.0, "px")."""
        input_data = getattr(self._pdf, "input_data", None)
        sampling = getattr(input_data, "sampling", None)
        units = getattr(input_data, "units", None)
        if sampling is None or units is None or len(units) == 0:
            return 1.0, "px"
        unit0 = str(units[0]).lower()
        size0 = float(sampling[0])
        if not math.isfinite(size0):
            return 1.0, "px"  # the frontend's scale bar falls back to pixels too
        if unit0 in ("å", "a", "angstrom", "angstroms"):
            return size0, "Å"
        if unit0 == "nm" and math.isfinite(size0 * 10.0):  # the conversion can overflow
            return size0 * 10.0, "Å"
        return 1.0, "px"

    def _set_nav_image(self, nav_image, pixel_size):
        """Sync the navigation image (default: a bright-field image of the data) and its scale bar.

        A ``pixel_size`` (Å) overrides the dataset calibration from now on; None keeps the
        current override.
        """
        if nav_image is not None:
            nav_img = to_numpy(nav_image).astype(np.float32)
        else:
            nav_img = self._compute_nav_image()
        self._nav_image = nav_img
        self.nav_data_min, self.nav_data_max = _finite_range(nav_img)
        self.nav_image_bytes = nav_img.tobytes()
        self.nav_pixel_size, self.nav_unit = self._resolve_nav_calibration()
        if pixel_size is not None:
            self._pixel_size_override = _finite_pixel_size(pixel_size)
        if self._pixel_size_override is not None:
            self.nav_pixel_size = self._pixel_size_override
            self.nav_unit = "Å"

    def _carry_over_k_ranges(self, was_full_range):
        """Keep the fit range and window where they lie inside the new k axis; reset them otherwise.

        A fit range that spanned the old data's full range spans the new data's full range too.
        """
        # Compare with the float64 k axis, allowing half a step, so a value typed as e.g. 1.55
        # still counts as inside a float32 1.5499999.
        qq64 = np.asarray(self._pdf.qq, dtype=np.float64)
        tol = 0.5 * float(qq64[1] - qq64[0]) if qq64.size > 1 else 0.0

        def inside(k):
            return qq64[0] - tol <= k <= qq64[-1] + tol

        fit_inside = inside(self.k_min_fit) and inside(self.k_max_fit) and self.k_min_fit < self.k_max_fit
        if was_full_range or not fit_inside:
            self.k_min_fit = self.k_min_available
            self.k_max_fit = self.k_max_available
        if not inside(self.k_min_window):
            self.k_min_window = 0.0
        if not inside(self.k_max_window):
            self.k_max_window = 0.0

    # =========================================================================
    # Observers
    # =========================================================================
    def _register_observers(self):
        """Recompute when a parameter or the selection changes, from Python or from JS."""
        self.observe(self._on_mask_change, names=["mask_version"])
        self.observe(
            self._on_fit_params_change,
            names=["k_min_fit", "k_max_fit", "k_min_window", "k_max_window", "k_lowpass", "k_highpass"],
        )
        self.observe(
            self._on_output_params_change,
            names=["r_max", "r_step", "damp_origin_oscillations", "r_cut", "max_iter"],
        )
        self.observe(self._on_density_change, names=["density_value"])
        self.observe(self._on_probe_change, names=["analysis_mode", "probe_row", "probe_col", "probe_size"])
        # The line ends sync without observers: JS bumps line_version once per edit, so a drag
        # triggers a single recompute (not one per end).
        self.observe(
            self._on_line_change,
            names=["line_version", "line_width", "line_mode", "line_perpendicular", "linescan_max_bins"],
        )
        # In line-scan mode the shown curve decides which PDF the stack holds.
        self.observe(self._on_plot_mode_change, names=["plot_mode"])
        self.observe(self._on_export_request_change, names=["export_request"])
        self.observe(self._invalidate_preview, names=list(_PREVIEW_TRAITS))

    def _on_mask_change(self, change=None):
        if self._initializing:
            return
        self._pdf.Ik = None
        self._pdf.bg = None
        self._update_mask_stats()
        self._recompute_full()

    def _on_fit_params_change(self, change=None):
        if self._initializing:
            return
        self._pdf.bg = None
        self._recompute_full()

    def _on_output_params_change(self, change=None):
        if self._initializing:
            return
        self._pdf.bg = None
        self._recompute_full()

    def _on_density_change(self, change=None):
        if self._initializing:
            return
        if self._is_linescan():
            # The stack depends on the density only through g(r) or the damping.
            if self.plot_mode == "gr" or self.damp_origin_oscillations:
                self._compute_linescan()
        else:
            # Rerun calculate_Gr rather than only calculate_gr: the damped G(r) and F(k)
            # depend on the density, and a shared PairDistributionFunction may hold another
            # selection's G(r). The radial mean is reused.
            self._recompute_full()

    def _on_probe_change(self, change=None):
        if self._initializing:
            return
        # The probe defines a synthetic mask, so this is a mask change.
        self._pdf.Ik = None
        self._pdf.bg = None
        self._update_mask_stats()
        self._recompute_full()

    def _on_line_change(self, change=None):
        if self._initializing:
            return
        # The line defines a synthetic band mask, so this is a mask change.
        self._pdf.Ik = None
        self._pdf.bg = None
        self._update_mask_stats()
        self._recompute_full()

    def _on_plot_mode_change(self, change=None):
        if self._initializing:
            return
        # Only the line scan computes per plot_mode; otherwise JS redraws the synced curves.
        if self._is_linescan():
            self._compute_linescan()
        elif (
            self.status_message in ("", _WINDOW_HINT) + _DENSITY_HINTS
            or self.status_message.startswith("Ready")
        ):
            # The g(r) hint depends on which curve is shown; an error message stays.
            self.status_message = self._status_hint()

    # =========================================================================
    # Selection
    # =========================================================================
    def _is_linescan(self):
        """True in line-scan mode, which computes one PDF per position along the line."""
        return self.analysis_mode == "line" and self.line_mode == "linescan"

    def _effective_line(self):
        """Ends of the line actually sampled, on scan positions.

        The drawn line, or its perpendicular bisector when line_perpendicular is set (so
        tracing along a feature samples across it), with both ends rounded to scan positions
        and clipped to the scan, so the band, the line-scan samples and the line's tip all sit
        on positions that have data. Mirrors effectiveLineEnds in js/showpdf/linescan.ts.
        """
        ends = (self.line_row0, self.line_col0, self.line_row1, self.line_col1)
        if self.line_perpendicular:
            ends = _perpendicular_bisector(*ends)
        r0, c0, r1, c1 = [_round_half_up(v) for v in ends]
        return _clip_to_scan(r0, c0, r1, c1, self.scan_rows, self.scan_cols)

    def _line_band_mask(self):
        """Boolean mask of the scan positions within line_width / 2 of the sampled segment.

        None when no line is active, the segment has zero length or the band holds no position.
        """
        if not self.line_active:
            return None
        r0, c0, r1, c1 = self._effective_line()
        dr, dc = r1 - r0, c1 - c0
        seg_len_sq = dr * dr + dc * dc
        if seg_len_sq <= 0:
            return None
        length = np.sqrt(seg_len_sq)
        rows = np.arange(self.scan_rows, dtype=np.float64)[:, None]
        cols = np.arange(self.scan_cols, dtype=np.float64)[None, :]
        pr = rows - r0
        pc = cols - c0
        # Position along the segment (0 at its start, 1 at its end) and distance from its
        # infinite line: t in [0, 1] gives flat ends and the distance test a constant width,
        # so the band is a rectangle.
        t = (pr * dr + pc * dc) / seg_len_sq
        perp = np.abs(pr * dc - pc * dr) / length
        half_w = max(0.5, self.line_width / 2.0)
        mask = (t >= 0.0) & (t <= 1.0) & (perp <= half_w)
        if not mask.any():
            return None
        return mask

    def _get_mask(self):
        """Boolean mask of the selected scan positions; None leaves calculate_Gr on the full scan."""
        if self.analysis_mode == "line":
            return self._line_band_mask()
        if self.analysis_mode == "probe":
            return self._probe_mask()
        return self._painted_mask()

    def _probe_mask(self):
        """Square of side 2 * probe_size - 1 around the probe (the brush's sizing), clipped to the scan."""
        half = max(0, self.probe_size - 1)
        r0 = max(0, self.probe_row - half)
        r1 = min(self.scan_rows, self.probe_row + half + 1)
        c0 = max(0, self.probe_col - half)
        c1 = min(self.scan_cols, self.probe_col + half + 1)
        if r1 <= r0 or c1 <= c0:
            return None
        mask = np.zeros((self.scan_rows, self.scan_cols), dtype=bool)
        mask[r0:r1, c0:c1] = True
        return mask

    def _painted_mask(self):
        """The painted mask; None when nothing is painted, or all or none of the scan is."""
        if not self.mask_b64:
            return None
        raw = base64.b64decode(self.mask_b64)
        mask = np.frombuffer(raw, dtype=np.uint8).reshape(self.scan_rows, self.scan_cols)
        if mask.all():
            return None
        bool_mask = mask > 0
        if not bool_mask.any():
            return None
        return bool_mask

    def _update_mask_stats(self):
        """Sync the number and fraction of the scan positions the analysis averages."""
        mask = self._get_mask()
        total = self.scan_rows * self.scan_cols
        if mask is not None:
            count = int(mask.sum())
            self.mask_pixel_count = count
            self.mask_fraction = count / total
        elif self._is_linescan():
            # No line drawn: the line scan has no positions and no curves.
            self.mask_pixel_count = 0
            self.mask_fraction = 0.0
        elif self.analysis_mode in ("probe", "line") or not self.mask_b64:
            # No mask (probe outside the scan, no line, nothing painted): calculate_Gr
            # averages the full scan, so report that.
            self.mask_pixel_count = total
            self.mask_fraction = 1.0
        else:
            # A painted mask that selects all or none of the scan, which is no mask.
            count = int(np.frombuffer(base64.b64decode(self.mask_b64), dtype=np.uint8).sum())
            self.mask_pixel_count = count if count > 0 else total
            self.mask_fraction = count / total if count > 0 else 1.0

    # =========================================================================
    # Status hints
    # =========================================================================
    def _density(self):
        """The supplied number density (atoms/Å³), or None while it is unset (0)."""
        return self.density_value if self.density_value > 0 else None

    def _density_hint(self):
        """Status hint when a requested output needs the unset density, else ""."""
        if self._density() is not None:
            return ""
        if self.damp_origin_oscillations:
            return _DENSITY_HINT_DAMPING
        if self.plot_mode == "gr":
            return _DENSITY_HINT_GR
        return ""

    def _window_hint(self):
        """_WINDOW_HINT when the k window has no positive weight (F(k) and G(r) are zero), else "".

        The window ends resolve as in calculate_Gr (<= 0 follows the fit range) and the
        library's own window function weighs its float32 k axis, so this agrees with
        calculate_Gr at the edges.
        """
        import torch

        params = self._curve_params()
        k = torch.from_numpy(np.asarray(self._pdf.qq).astype(np.float32))
        k_min_fit = params["k_min_fit"] if params["k_min_fit"] is not None else float(k.min())
        k_max_fit = params["k_max_fit"] if params["k_max_fit"] is not None else float(k.max())
        k_min = params["k_min_window"] if params["k_min_window"] is not None else k_min_fit
        k_max = params["k_max_window"] if params["k_max_window"] is not None else k_max_fit
        wk = self._pdf._k_window(k, k_min, k_max)
        return "" if bool((wk > 0).any()) else _WINDOW_HINT

    def _status_hint(self):
        """The empty-window hint, else the density hint, else ""; an error message beats both."""
        return self._window_hint() or self._density_hint()

    # =========================================================================
    # Curves
    # =========================================================================
    def _curve_params(self):
        """Keyword arguments for every calculate_Gr call, with the 0.0 sentinels as None.

        The low-r damping needs the density, so it is only requested once density_value is
        set; _density_hint reports the skipped damping.
        """
        density = self._density()
        return dict(
            k_min_fit=self.k_min_fit if self.k_min_fit > 0 else None,
            k_max_fit=self.k_max_fit if self.k_max_fit > 0 else None,
            k_min_window=self.k_min_window if self.k_min_window > 0 else None,
            k_max_window=self.k_max_window if self.k_max_window > 0 else None,
            k_lowpass=self.k_lowpass if self.k_lowpass > 0 else None,
            k_highpass=self.k_highpass if self.k_highpass > 0 else None,
            r_max=self.r_max,
            r_step=self.r_step,
            damp_origin_oscillations=self.damp_origin_oscillations and density is not None,
            density=density,
            r_cut=self.r_cut,
            max_iter=int(self.max_iter),
        )

    def _refresh_pdf(self):
        """Recompute this widget's curves on the PDF object before reading them.

        The PairDistributionFunction may be shared (the object passed in, or ``pdf``), so its
        results can belong to another selection or parameter set. calculate_Gr reuses the
        radial mean of the same mask, so this is cheap. It raises if the current settings
        cannot be computed.
        """
        params = self._curve_params()
        self._pdf.calculate_Gr(mask_realspace=self._get_mask(), **params)
        if params["density"] is not None:
            self._pdf.calculate_gr(
                density=params["density"], r_cut=self.r_cut, max_iter=self.max_iter,
                set_pdf_positive=True,
            )

    def _displayed_fk(self):
        """F(k) the transform used: the damped one when the low-r correction ran."""
        if self._pdf.Fk_damped is not None:
            return self._pdf.Fk_damped
        return self._pdf.Fk_masked

    def _displayed_gr(self):
        """G(r) to show: the damped one when the low-r correction ran."""
        if self._pdf.reduced_pdf_damped is not None:
            return self._pdf.reduced_pdf_damped
        return self._pdf._reduced_pdf

    def _recompute_full(self):
        """Recompute and sync the curves (or the line-scan stack) of the current selection."""
        if self._is_linescan():
            self._compute_linescan()
            return
        self.computing = True
        self.status_message = "Computing..."
        selection = None
        try:
            mask = self._get_mask()
            selection = (self._data_version, None if mask is None else mask.tobytes())
            self._pdf.calculate_Gr(mask_realspace=mask, **self._curve_params())
            self._sync_curves_to_js()
            self._curves_selection = selection
            # A g(r) error takes precedence over the window and density hints.
            self.status_message = self._compute_gr() or self._status_hint()
        except Exception as e:
            self.status_message = f"Error: {e}"
            # Only a parameter change keeps the curves: those of another selection or dataset
            # would describe other positions (and a full export would recompute G(r) from
            # that I(k)).
            if selection != self._curves_selection:
                self._clear_curves()
        finally:
            self.computing = False

    def _clear_curves(self):
        """Empty every synced single curve: I(k), B(k), F(k), G(r) and g(r)."""
        self._curves_selection = None
        self.n_points_ik = 0
        self.ik_x_bytes = b""
        self.ik_y_bytes = b""
        self.ik_bg_y_bytes = b""
        self.n_points_fk = 0
        self.fk_x_bytes = b""
        self.fk_y_bytes = b""
        self.n_points_gr = 0
        self.gr_x_bytes = b""
        self.gr_y_bytes = b""
        self._clear_gr()

    def _compute_gr(self):
        """Compute g(r) with calculate_gr (negative values clamped to 0) and sync it.

        g(r) stays empty while density_value is unset, since the density is never estimated.
        Returns an error message for the status line, or "" on success.
        """
        density = self._density()
        if self._pdf._reduced_pdf is None or self._pdf._r is None or density is None:
            self._clear_gr()
            return ""
        try:
            self._pdf.calculate_gr(
                density=density, r_cut=self.r_cut, max_iter=self.max_iter, set_pdf_positive=True
            )
        except Exception as e:
            self._clear_gr()
            return f"g(r) error: {e}"
        r = _float32(self._pdf._r)
        gr = _float32(self._pdf._pdf)
        self.n_points_pdf = len(r)
        self.pdf_x_bytes = r.tobytes()
        self.pdf_y_bytes = gr.tobytes()
        return ""

    def _clear_gr(self):
        """Empty g(r) in JS and on the PDF object, so save_results keeps no g(r) of an earlier density."""
        self._pdf._pdf = None
        self.n_points_pdf = 0
        self.pdf_x_bytes = b""
        self.pdf_y_bytes = b""

    def _sync_curves_to_js(self):
        """Send I(k), B(k), F(k) and G(r) of the PDF object to JS."""
        qq = np.asarray(self._pdf.qq, dtype=np.float32)
        self.n_points_ik = len(qq)
        self.ik_x_bytes = qq.tobytes()
        self.ik_y_bytes = _float32(self._pdf.Ik).tobytes()
        if self._pdf.bg is not None:
            self.ik_bg_y_bytes = _float32(self._pdf.bg).tobytes()
        else:
            self.ik_bg_y_bytes = b""

        fk_t = self._displayed_fk()
        if fk_t is not None:
            fk = _float32(fk_t)
            self.n_points_fk = len(fk)
            self.fk_x_bytes = qq[: len(fk)].tobytes()
            self.fk_y_bytes = fk.tobytes()
        else:
            self.n_points_fk = 0
            self.fk_x_bytes = b""
            self.fk_y_bytes = b""

        if self._pdf._r is not None and self._pdf._reduced_pdf is not None:
            r = _float32(self._pdf._r)
            Gr = _float32(self._displayed_gr())
            self.n_points_gr = len(r)
            self.gr_x_bytes = r.tobytes()
            self.gr_y_bytes = Gr.tobytes()
        else:
            self.n_points_gr = 0
            self.gr_x_bytes = b""
            self.gr_y_bytes = b""

    # =========================================================================
    # Line scan
    # =========================================================================
    def _extract_plot_curve(self):
        """(y, x) float32 of the plot_mode curve from the computed PDF state.

        (None, None) if that curve has not been computed, or for g(r) without a density.
        """
        mode = self.plot_mode
        if mode == "Ik":
            x = np.asarray(self._pdf.qq, dtype=np.float32)
            y = _float32(self._pdf.Ik)
        elif mode == "Fk":
            fk = self._displayed_fk()
            if fk is None:
                return None, None
            qq = np.asarray(self._pdf.qq, dtype=np.float32)
            y = _float32(fk)
            x = qq[: len(y)]
        elif mode == "gr":
            density = self._density()
            if self._pdf._reduced_pdf is None or self._pdf._r is None or density is None:
                return None, None
            self._pdf.calculate_gr(
                density=density, r_cut=self.r_cut, max_iter=self.max_iter, set_pdf_positive=True
            )
            x = _float32(self._pdf._r)
            y = _float32(self._pdf._pdf)
        else:  # "Gr"
            if self._pdf._r is None or self._pdf._reduced_pdf is None:
                return None, None
            x = _float32(self._pdf._r)
            y = _float32(self._displayed_gr())
        return y, x

    def _clear_linescan(self):
        """Empty the synced line-scan stack."""
        self.n_linescan = 0
        self.linescan_n_points = 0
        self.linescan_bytes = b""
        self.linescan_axis_bytes = b""

    def _linescan_samples(self):
        """Evenly spaced samples along the line, and the sample each band position averages into.

        The samples run from the line start A (position 0) to its tip B (position L), one per
        scan step along the line's longer axis (capped by linescan_max_bins). The sampled line
        ends on scan positions (see _effective_line); without the cap no sample is empty. On a
        diagonal line the samples fall between scan positions. Each band position goes to its
        nearest sample; the two end samples cover half a step, because the band stops at A and
        B. Returns None without a line, else ``(band, sample_idx, n_samples, length, positions)``.
        """
        band = self._line_band_mask()
        if band is None:
            return None
        r0, c0, r1, c1 = self._effective_line()
        dr, dc = r1 - r0, c1 - c0
        seg_len_sq = dr * dr + dc * dc
        length = float(np.sqrt(seg_len_sq))
        n_steps = round(max(abs(dr), abs(dc)))
        n = max(1, min(self.linescan_max_bins, n_steps + 1))

        rows = np.arange(self.scan_rows, dtype=np.float64)[:, None]
        cols = np.arange(self.scan_cols, dtype=np.float64)[None, :]
        t = np.clip(((rows - r0) * dr + (cols - c0) * dc) / seg_len_sq, 0.0, 1.0)
        if n > 1:
            # Nearest sample. The offset is snapped to 1e-9 so exact ties are not split by
            # rounding error, then floor(x + 0.5) as in the frontend.
            x = t * (n - 1)
            x = np.floor(x * 1e9 + 0.5) / 1e9
            sample_idx = np.minimum(np.floor(x + 0.5), n - 1).astype(int)
            positions = np.arange(n, dtype=np.float64) * (length / (n - 1))
        else:
            sample_idx = np.zeros(t.shape, dtype=int)
            positions = np.array([length / 2.0])
        return band, sample_idx, n, length, positions

    def _linescan_bin_radial_means(self, band, bin_idx, n_bins):
        """Radial means I(k) of all line-scan samples in one chunked pass over the polar data.

        Equivalent to calculate_radial_mean per sample mask (same read path and output dtype),
        but reads the 4D polar array once instead of n_bins times. Returns (Ik_all, valid_bins):
        an (n_valid, Nk) float32 tensor on the PDF device and the matching non-empty samples.
        """
        import torch

        pol = self._pdf.polar
        if pol.array is None:
            polar = pol.tensor
        else:
            polar = torch.from_numpy(np.ascontiguousarray(pol.array))
        n_rows, n_k = polar.shape[0], polar.shape[3]  # polar is (scan row, scan col, phi, k)
        device = self._pdf.device
        labels = np.where(band, bin_idx, -1).astype(np.int64)  # sample of each position, -1 outside
        sums = torch.zeros(n_bins, n_k, dtype=torch.float64, device=device)
        counts = torch.zeros(n_bins, dtype=torch.float64, device=device)
        chunk = 16
        for row0 in range(0, n_rows, chunk):
            row1 = min(row0 + chunk, n_rows)
            arr = polar[row0:row1].to(device)
            # Mean over phi, scan positions flattened, accumulated into their samples.
            rad = arr.mean(dim=2).reshape(-1, n_k).double()
            lbl = torch.as_tensor(labels[row0:row1].reshape(-1), device=device)
            valid = lbl >= 0
            if bool(valid.any()):
                v = lbl[valid]
                sums.index_add_(0, v, rad[valid])
                ones = torch.ones(int(valid.sum()), dtype=torch.float64, device=device)
                counts.index_add_(0, v, ones)
        nonempty = counts > 0
        valid_bins = torch.nonzero(nonempty, as_tuple=False).flatten().tolist()
        Ik_all = (sums[nonempty] / counts[nonempty][:, None]).float()
        return Ik_all, valid_bins

    def _calculate_Gr_for_bin(self, Ik, bin_mask, params):
        """calculate_Gr for one line-scan sample from its precomputed radial mean.

        The I(k) is cached under the sample's mask, which calculate_Gr's cache check accepts,
        so the PDF object ends up as calculate_Gr(mask_realspace=bin_mask) would leave it.
        """
        self._pdf.Ik = Ik
        self._pdf._Ik_mask = bin_mask
        self._pdf.calculate_Gr(mask_realspace=bin_mask, **params)

    def _compute_linescan(self):
        """Compute and sync the plot_mode curve of every line-scan sample (see _linescan_samples).

        The radial means of all samples come from one pass over the polar data; each sample then
        runs calculate_Gr exactly as a single masked computation would. The curves are stacked
        into linescan_bytes, (n_linescan, linescan_n_points).
        """
        self.computing = True
        self.status_message = "Computing line-scan..."
        try:
            samples = self._linescan_samples()
            if samples is None:
                self._clear_linescan()
                self.status_message = self._status_hint()
                return
            band, bin_idx, n_bins, _length, _positions = samples

            Ik_all, valid_bins = self._linescan_bin_radial_means(band, bin_idx, n_bins)
            if not valid_bins:
                self._clear_linescan()
                self.status_message = ""
                return
            params = self._curve_params()

            stack = None
            axis = None
            for j, bin_i in enumerate(valid_bins):
                self._calculate_Gr_for_bin(Ik_all[j], band & (bin_idx == bin_i), params)
                y, x = self._extract_plot_curve()
                if y is None:
                    continue
                if stack is None:
                    axis = x
                    stack = _nan_stack(n_bins, len(y))
                _fill_row(stack, bin_i, y)

            if stack is None:
                # Nothing to stack, e.g. g(r) without a density.
                self._clear_linescan()
                self.status_message = self._status_hint()
                return
            self.n_linescan = n_bins
            self.linescan_n_points = stack.shape[1]
            self.linescan_bytes = stack.tobytes()
            self.linescan_axis_bytes = axis.tobytes()
            self.status_message = self._status_hint()
        except Exception as e:
            self.status_message = f"Line-scan error: {e}"
            self._clear_linescan()
        finally:
            self.computing = False

    def _compute_linescan_all_modes(self):
        """Per-sample I(k), B(k), F(k), G(r) and g(r) along the line, for save_results.

        Runs the samples as _compute_linescan does but keeps every curve, and leaves the synced
        single-mode stack alone. Returns None without a line or a non-empty sample, else a dict
        of NaN-filled (n_bins, n) stacks and their axes.
        """
        samples = self._linescan_samples()
        if samples is None:
            return None
        band, bin_idx, n_bins, length, positions = samples

        Ik_all, valid_bins = self._linescan_bin_radial_means(band, bin_idx, n_bins)
        if not valid_bins:
            return None
        params = self._curve_params()
        density = params["density"]

        k_axis = np.asarray(self._pdf.qq, dtype=np.float32)
        Ik_stack = _nan_stack(n_bins, k_axis.size)
        bg_stack = _nan_stack(n_bins, k_axis.size)
        # The F(k), G(r) and g(r) stacks are sized by the first sample that has the curve.
        Fk_stack = None
        Gr_stack = None
        gr_stack = None
        k_fk_axis = None
        r_axis = None

        for j, bin_i in enumerate(valid_bins):
            self._calculate_Gr_for_bin(Ik_all[j], band & (bin_idx == bin_i), params)
            Ik_stack[bin_i] = _float32(self._pdf.Ik)
            bg_stack[bin_i] = _float32(self._pdf.bg)

            fk_t = self._displayed_fk()
            if fk_t is not None:
                fk = _float32(fk_t)
                if Fk_stack is None:
                    k_fk_axis = k_axis[: fk.size]
                    Fk_stack = _nan_stack(n_bins, fk.size)
                _fill_row(Fk_stack, bin_i, fk)

            if self._pdf._r is not None and self._pdf._reduced_pdf is not None:
                Gr = _float32(self._displayed_gr())
                if Gr_stack is None:
                    r_axis = _float32(self._pdf._r)
                    Gr_stack = _nan_stack(n_bins, r_axis.size)
                _fill_row(Gr_stack, bin_i, Gr)

                if density is not None:
                    self._pdf.calculate_gr(
                        density=density, r_cut=self.r_cut, max_iter=self.max_iter,
                        set_pdf_positive=True,
                    )
                    gr = _float32(self._pdf._pdf)
                    if gr_stack is None:
                        gr_stack = _nan_stack(n_bins, Gr_stack.shape[1])
                    _fill_row(gr_stack, bin_i, gr)

        # Sample positions along the line in scan pixels (evenly spaced, from A).
        bin_positions_px = positions.astype(np.float32)

        return {
            "n_bins": n_bins,
            "bin_positions_px": bin_positions_px,
            "line_length_px": length,
            "k": k_axis,
            "Ik": Ik_stack,
            "bg": bg_stack,
            "k_fk": k_fk_axis if k_fk_axis is not None else np.zeros(0, dtype=np.float32),
            "Fk": Fk_stack if Fk_stack is not None else np.zeros((n_bins, 0), dtype=np.float32),
            "r": r_axis if r_axis is not None else np.zeros(0, dtype=np.float32),
            "Gr": Gr_stack if Gr_stack is not None else np.zeros((n_bins, 0), dtype=np.float32),
            "gr": gr_stack,  # None without a density
        }

    # =========================================================================
    # Public API
    # =========================================================================
    @property
    def pdf(self):
        """The PairDistributionFunction the curves are computed with."""
        return self._pdf

    @property
    def mask(self) -> np.ndarray:
        """Boolean mask of the selected scan positions (all True for the full scan)."""
        m = self._get_mask()
        if m is None:
            return np.ones((self.scan_rows, self.scan_cols), dtype=bool)
        return m

    def set_mask(self, mask) -> Self:
        """Set the painted scan mask and switch to mask mode.

        ``mask`` is a boolean array of shape ``(scan_rows, scan_cols)``; True
        includes a scan position.
        """
        mask_np = to_numpy(mask).astype(bool)
        if mask_np.shape != (self.scan_rows, self.scan_cols):
            raise ValueError(
                f"Mask shape {mask_np.shape} does not match scan shape "
                f"({self.scan_rows}, {self.scan_cols})"
            )
        raw = mask_np.astype(np.uint8).tobytes()
        self._initializing = True
        try:
            self.mask_bytes = raw
            # mask_b64 is what _get_mask() reads and the canvas draws.
            self.mask_b64 = base64.b64encode(raw).decode("ascii")
            self.analysis_mode = "mask"
        finally:
            self._initializing = False
        self._on_mask_change()
        return self

    def clear_mask(self) -> Self:
        """Clear the painted mask, which selects the full scan."""
        self.mask_bytes = b""
        self.mask_b64 = ""
        self._on_mask_change()
        return self

    def set_line(
        self,
        row0: float,
        col0: float,
        row1: float,
        col1: float,
        width: int | None = None,
    ) -> Self:
        """Set the line-band selection and switch to line mode.

        Endpoints are in scan (row, col) coordinates; the sampled line's ends
        snap to the nearest scan positions. ``width`` is the band thickness, an
        odd number of scan pixels (1, 3, 5, ...): a horizontal or vertical band
        of width w covers exactly w rows or columns (clipped at the scan edge).
        When omitted the current ``line_width`` is kept. A width that is not an
        odd integer raises ``TraitError`` and an endpoint that is NaN, infinite
        or beyond ±1e9 raises ``ValueError``; either leaves the widget unchanged.
        """
        # Check every argument first, so a bad one raises before anything changes.
        row0 = _line_end_float("row0", row0)
        col0 = _line_end_float("col0", col0)
        row1 = _line_end_float("row1", row1)
        col1 = _line_end_float("col1", col1)
        if width is not None:
            width = _line_width_int(width)
        self._initializing = True
        try:
            # Width first, so a rejected width raises before anything changes.
            if width is not None:
                self.line_width = width
            self.line_row0 = row0
            self.line_col0 = col0
            self.line_row1 = row1
            self.line_col1 = col1
            self.line_active = True
            self.analysis_mode = "line"
        finally:
            self._initializing = False
        self._on_line_change()
        return self

    def clear_line(self) -> Self:
        """Remove the line selection."""
        self._initializing = True
        try:
            self.line_active = False
            self.line_row0 = self.line_col0 = self.line_row1 = self.line_col1 = 0.0
        finally:
            self._initializing = False
        self._on_line_change()
        return self

    def set_data(
        self, data, *, nav_image=None, find_origin=True, pixel_size=None, **pdf_kwargs
    ) -> Self:
        """Replace the dataset, keeping the PDF parameters where they still fit.

        Accepts the same inputs as the constructor; ``pdf_kwargs`` go to
        ``PairDistributionFunction.from_data`` (``device`` defaults to
        ``"cpu"`` as in the constructor). The painted mask, line and probe
        are reset because they refer to the old scan. A fit range that spans
        the old data's full k range spans the new data's full range too. A
        narrower fit range and a set window are kept if they lie inside the
        new k range; otherwise the fit range is reset to the full range and the
        window to 0 (follow the fit range). ``pixel_size`` overrides the scan
        calibration as in the constructor; without it, an override given to
        the constructor (or an earlier ``set_data``) stays in effect.
        """
        pixel_size = _finite_pixel_size(pixel_size)  # raises before anything changes
        pdf, _title = _load_pdf(data, find_origin=find_origin, **pdf_kwargs)
        qq = _finite_k_axis(pdf)  # raises before anything changes
        self._pdf = pdf
        self._reset_pdf_cache()
        self._data_version += 1

        # Update traits without triggering one recompute per trait.
        self._initializing = True
        try:
            polar_shape = self._pdf.polar.shape
            self.scan_rows = int(polar_shape[0])
            self.scan_cols = int(polar_shape[1])
            was_full_range = (self.k_min_fit, self.k_max_fit) == (
                self.k_min_available, self.k_max_available
            )
            self.k_min_available = float(qq[0])
            self.k_max_available = float(qq[-1])
            self._carry_over_k_ranges(was_full_range)

            # Selections refer to the old scan grid.
            self.mask_bytes = b""
            self.mask_b64 = ""
            self.probe_row = self.scan_rows // 2
            self.probe_col = self.scan_cols // 2
            self.line_active = False
            self.line_row0 = self.line_col0 = self.line_row1 = self.line_col1 = 0.0
            self._clear_linescan()
        finally:
            self._initializing = False

        self._set_nav_image(nav_image, pixel_size)
        self._update_mask_stats()
        self._recompute_full()
        return self

    # =========================================================================
    # Saved-notebook preview
    # =========================================================================
    def _static_png_b64(self, max_px: int = 512) -> str | None:
        """The scan image beside the shown curve, or line-scan stack, as a base64 PNG max_px wide.

        A notebook saved with save_state=False keeps this picture instead of the scan image, mask
        and stack. It draws the synced traits and never recomputes; None while there is no data.
        """
        if getattr(self, "_pdf", None) is None or not self.nav_image_bytes:
            return None
        # A standalone Figure renders off screen without switching the notebook's backend.
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=(max_px / 100, max_px / 250), dpi=100)
        FigureCanvasAgg(fig)
        nav_ax, curve_ax = fig.subplots(1, 2, gridspec_kw={"width_ratios": (1, 2)})
        nav = self._nav_image
        finite = nav[np.isfinite(nav)]
        vmin, vmax = np.percentile(finite, (2, 98)) if finite.size and self.auto_contrast else (None, None)
        nav_ax.imshow(nav, cmap=self.cmap, vmin=vmin, vmax=vmax)
        nav_ax.set_axis_off()
        xlabel, ylabel = _CURVE_AXIS_LABELS[self.plot_mode]
        if self._is_linescan() and self.n_linescan:
            stack = np.frombuffer(self.linescan_bytes, np.float32).reshape(self.n_linescan, -1)
            axis = np.frombuffer(self.linescan_axis_bytes, np.float32)
            extent = (axis[0], axis[-1], 0, self.n_linescan - 1)
            curve_ax.imshow(stack, aspect="auto", origin="lower", cmap="inferno", extent=extent)
            ylabel = "line position"
        else:
            x_trait, y_trait = _CURVE_TRAITS[self.plot_mode]
            x = np.frombuffer(getattr(self, x_trait), np.float32)
            y = np.frombuffer(getattr(self, y_trait), np.float32)
            curve_ax.plot(x, y, color="steelblue", linewidth=1.0)
            if not y.size:  # g(r) without a density, or a failed compute
                curve_ax.text(0.5, 0.5, self.status_message, ha="center", va="center", transform=curve_ax.transAxes)
        curve_ax.set_xlabel(xlabel)
        curve_ax.set_ylabel(ylabel)
        if self.title:
            fig.suptitle(self.title, fontsize=9)
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _invalidate_preview(self, change=None):
        """Drop the stored preview, so the next notebook save draws the current view."""
        self._static_fallback_jpeg = ""

    # =========================================================================
    # Saving results
    # =========================================================================
    def save_image(
        self,
        path: str | pathlib.Path,
        *,
        plot_mode: str | None = None,
        format: str | None = None,
        dpi: int = 150,
    ) -> pathlib.Path:
        """Save the current PDF curve as PNG, PDF, or TIFF.

        Parameters
        ----------
        path : str or pathlib.Path
            Output file path.
        plot_mode : {"Ik", "Fk", "Gr", "gr"}, optional
            Curve to render. Defaults to the widget's current ``plot_mode``.
        format : {"png", "pdf", "tiff"}, optional
            Inferred from the file extension when omitted.
        dpi : int, default 150
            Output resolution.

        Returns
        -------
        pathlib.Path
            The written file path.
        """
        # A standalone Figure renders off screen without switching the notebook's
        # matplotlib backend (matplotlib.use("Agg") would).
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        path = pathlib.Path(path)
        fmt = (format or path.suffix.lstrip(".").lower() or "png").lower()
        if fmt == "tif":
            fmt = "tiff"
        if fmt not in ("png", "pdf", "tiff"):
            raise ValueError(f"Unsupported format: {fmt!r}. Use 'png', 'pdf', or 'tiff'.")

        mode = plot_mode if plot_mode is not None else self.plot_mode
        if mode not in ("Ik", "Fk", "Gr", "gr"):
            raise ValueError(f"Unknown plot_mode: {mode!r}. Use 'Ik', 'Fk', 'Gr', or 'gr'.")
        if self._is_linescan():
            raise ValueError(
                "save_image saves one curve; a line scan has one per position. "
                "Use save_results, or line_mode='averaged' for the band average."
            )

        self._refresh_pdf()
        bg = None
        if mode == "Ik":
            x = np.asarray(self._pdf.qq, dtype=np.float32)
            y = _float32(self._pdf.Ik)
            if self._pdf.bg is not None and self.show_background:
                bg = _float32(self._pdf.bg)
        elif mode == "Fk":
            y = _float32(self._displayed_fk())
            x = np.asarray(self._pdf.qq, dtype=np.float32)[: len(y)]
        elif mode == "Gr":
            x = _float32(self._pdf._r)
            y = _float32(self._displayed_gr())
        else:  # "gr"
            if self._pdf._pdf is None:
                raise ValueError("g(r) is not computed yet: set density_value (atoms/Å³).")
            x = _float32(self._pdf._r)
            y = _float32(self._pdf._pdf)
        xlabel, ylabel = _CURVE_AXIS_LABELS[mode]

        fig = Figure(figsize=(6, 3.5), dpi=dpi)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot()
        ax.plot(x, y, color="steelblue", linewidth=1.0, label=ylabel)
        if bg is not None:
            ax.plot(x, bg, color="orange", linewidth=1.0, linestyle="--", label="B(k)")
            ax.legend(fontsize=8, framealpha=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if self.title:
            ax.set_title(self.title, fontsize=11)
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.tight_layout()

        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), format=fmt, dpi=dpi, bbox_inches="tight")
        return path

    def save_results(
        self,
        path: str | pathlib.Path,
        *,
        overwrite: bool = False,
    ) -> pathlib.Path:
        """Save the current PDF results to a self-contained zip file.

        The zip holds a plain-text README, a plain-text metadata file and CSV
        curve files; only the mask (mask mode) is saved as ``mask.npy``, since a
        scan-shaped boolean CSV is unwieldy.

        Layout (always present):
            README.txt
            metadata.txt
            curves_k.csv         k, I(k), B(k), F(k) (selected region)
            curves_r.csv         r, G(r), g(r) (g(r) only if density_value is set)

        Extra files by mode:
            mask mode:       mask.npy
            line scan:       linescan_position.csv, linescan_Ik.csv,
                             linescan_Ik_bg.csv, linescan_Fk.csv,
                             linescan_Gr.csv, linescan_gr.csv (if g(r) computed)

        Parameters
        ----------
        path : str or pathlib.Path
            Output zip path. Should end in ``.zip``; the extension is added if
            missing.
        overwrite : bool, default False
            If False and the file already exists, raises FileExistsError.

        Returns
        -------
        pathlib.Path
            The written zip path.
        """
        path = pathlib.Path(path)
        if path.suffix.lower() != ".zip":
            path = path.with_suffix(path.suffix + ".zip" if path.suffix else ".zip")
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it.")

        is_linescan = self._is_linescan()
        linescan = None
        if is_linescan:
            linescan = self._compute_linescan_all_modes()
            if linescan is None:
                raise RuntimeError(
                    "Line scan has no valid bins yet — draw a line and let the "
                    "widget compute before saving."
                )
        # Recompute the selection first, since the PDF object may be shared. In a line scan the
        # curves_*.csv files hold the band average, and its I(k) stays cached under the band
        # mask, so the next recompute stays correct.
        self._refresh_pdf()
        k, Ik, bg, Fk, r, Gr, gr = self._extract_1d_curves()
        metadata = self._results_metadata(linescan)

        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("metadata.txt", _format_metadata_txt(metadata))
            zf.writestr("curves_k.csv", _columns_to_csv(["k", "Ik", "bg", "Fk"], [k, Ik, bg, Fk]))
            zf.writestr("curves_r.csv", _columns_to_csv(["r", "Gr", "gr"], [r, Gr, gr]))

            if self.analysis_mode == "mask":
                buf = io.BytesIO()
                np.save(buf, self.mask, allow_pickle=False)
                zf.writestr("mask.npy", buf.getvalue())

            if linescan is not None:
                k_axis, k_fk_axis, r_axis = linescan["k"], linescan["k_fk"], linescan["r"]
                zf.writestr("linescan_position.csv", _positions_to_csv(linescan["bin_positions_px"]))
                zf.writestr("linescan_Ik.csv", _stack_to_csv_with_axis(k_axis, linescan["Ik"]))
                zf.writestr("linescan_Ik_bg.csv", _stack_to_csv_with_axis(k_axis, linescan["bg"]))
                if linescan["Fk"].size:
                    zf.writestr("linescan_Fk.csv", _stack_to_csv_with_axis(k_fk_axis, linescan["Fk"]))
                if linescan["Gr"].size:
                    zf.writestr("linescan_Gr.csv", _stack_to_csv_with_axis(r_axis, linescan["Gr"]))
                if linescan["gr"] is not None and linescan["gr"].size:
                    zf.writestr("linescan_gr.csv", _stack_to_csv_with_axis(r_axis, linescan["gr"]))

            damped = self.damp_origin_oscillations and self._density() is not None
            zf.writestr("README.txt", _build_readme(is_linescan, self.analysis_mode, damped))

        return path.resolve()

    def _extract_1d_curves(self):
        """(k, I(k), B(k), F(k), r, G(r), g(r)) as float32 arrays; a curve not computed is empty.

        F(k) and G(r) are the curves the transform used (damped when the low-r correction
        ran); g(r) is empty without a density.
        """
        empty = np.zeros(0, dtype=np.float32)
        k = np.asarray(self._pdf.qq, dtype=np.float32)
        Ik = _float32(self._pdf.Ik) if self._pdf.Ik is not None else empty
        bg = _float32(self._pdf.bg) if self._pdf.bg is not None else empty
        fk_t = self._displayed_fk()
        Fk = _float32(fk_t) if fk_t is not None else empty
        if self._pdf._r is not None and self._pdf._reduced_pdf is not None:
            r = _float32(self._pdf._r)
            Gr = _float32(self._displayed_gr())
        else:
            r = empty
            Gr = empty
        if self._pdf._pdf is not None and r.size:
            gr = _float32(self._pdf._pdf)
        else:
            gr = empty
        return k, Ik, bg, Fk, r, Gr, gr

    def _results_metadata(self, linescan):
        """The metadata.txt content of save_results: mode, selection and fit parameters."""
        metadata = {
            "widget_version": self.widget_version,
            "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "title": self.title,
            "scan_shape": [self.scan_rows, self.scan_cols],
            "analysis_mode": self.analysis_mode,
            "line_mode": self.line_mode if self.analysis_mode == "line" else None,
            "plot_mode_at_save": self.plot_mode,
            "region": self._region_metadata(),
            "fit_parameters": {
                "k_min_fit": self.k_min_fit,
                "k_max_fit": self.k_max_fit,
                "k_min_window": self.k_min_window,
                "k_max_window": self.k_max_window,
                "k_lowpass": self.k_lowpass,
                "k_highpass": self.k_highpass,
                "r_max": self.r_max,
                "r_step": self.r_step,
                "damp_origin_oscillations": self.damp_origin_oscillations,
                "r_cut": self.r_cut,
                "max_iter": int(self.max_iter),
                "density_value": self.density_value,
            },
        }
        if linescan is not None:
            metadata["region"]["n_linescan_bins"] = linescan["n_bins"]
            metadata["region"]["line_length_px"] = linescan["line_length_px"]
        return metadata

    def _region_metadata(self):
        """The selected scan positions, for metadata.txt."""
        if self.analysis_mode == "mask":
            return {
                "kind": "mask",
                "mask_pixel_count": self.mask_pixel_count,
                "mask_fraction": self.mask_fraction,
                "mask_file": "mask.npy",
            }
        if self.analysis_mode == "probe":
            return {
                "kind": "probe",
                "probe_row": int(self.probe_row),
                "probe_col": int(self.probe_col),
                "probe_size": int(self.probe_size),
                "mask_pixel_count": self.mask_pixel_count,
                "mask_fraction": self.mask_fraction,
            }
        if self.analysis_mode == "line":
            return {
                "kind": "line",
                "line_row0": self.line_row0,
                "line_col0": self.line_col0,
                "line_row1": self.line_row1,
                "line_col1": self.line_col1,
                "line_width": self.line_width,
                "line_perpendicular": self.line_perpendicular,
                "sampled_line": list(self._effective_line()),
                "mask_pixel_count": self.mask_pixel_count,
                "mask_fraction": self.mask_fraction,
            }
        return {"kind": self.analysis_mode}

    # =========================================================================
    # State persistence
    # =========================================================================
    def state_dict(self) -> dict:
        """The parameters, selection and display settings, as restored by load_state_dict."""
        return {
            "title": self.title,
            "k_min_fit": self.k_min_fit,
            "k_max_fit": self.k_max_fit,
            "k_min_window": self.k_min_window,
            "k_max_window": self.k_max_window,
            "k_lowpass": self.k_lowpass,
            "k_highpass": self.k_highpass,
            "r_max": self.r_max,
            "r_step": self.r_step,
            "damp_origin_oscillations": self.damp_origin_oscillations,
            "r_cut": self.r_cut,
            "max_iter": self.max_iter,
            "density_value": self.density_value,
            "analysis_mode": self.analysis_mode,
            "probe_row": self.probe_row,
            "probe_col": self.probe_col,
            "probe_size": self.probe_size,
            "line_row0": self.line_row0,
            "line_col0": self.line_col0,
            "line_row1": self.line_row1,
            "line_col1": self.line_col1,
            "line_width": self.line_width,
            "line_active": self.line_active,
            "line_mode": self.line_mode,
            "line_perpendicular": self.line_perpendicular,
            "linescan_max_bins": self.linescan_max_bins,
            "linescan_view": self.linescan_view,
            "waterfall_count": self.waterfall_count,
            # The painted mask (base64 uint8, scan_rows * scan_cols); "" means the full scan.
            "mask_b64": self.mask_b64,
            "plot_mode": self.plot_mode,
            "show_background": self.show_background,
            "cmap": self.cmap,
            "log_scale": self.log_scale,
            "auto_contrast": self.auto_contrast,
            "show_stats": self.show_stats,
            "show_controls": self.show_controls,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore a state dict from state_dict and recompute; unknown keys are ignored."""
        state, width_notice = _migrate_state(state)
        self._initializing = True
        try:
            # All or nothing: when a value is rejected with TraitError,
            # hold_trait_notifications rolls back every value set before it.
            with self.hold_trait_notifications():
                allowed_keys = set(self.state_dict().keys())
                for key, val in state.items():
                    if key in allowed_keys and hasattr(self, key):
                        try:
                            setattr(self, key, val)
                        except OverflowError as e:
                            # An int too large for a Float trait: reject it as
                            # TraitError too, so the rollback applies.
                            raise traitlets.TraitError(f"{key}: {e}") from e
                self._restore_mask_bytes()
        finally:
            # A rejected value raises; the widget must still respond to later changes.
            self._initializing = False
        if width_notice:  # only now: a rejected load leaves the width unchanged
            warnings.warn(width_notice, UserWarning, stacklevel=2)
        self._update_mask_stats()
        self._recompute_full()

    def _restore_mask_bytes(self):
        """Mirror a restored mask_b64 into mask_bytes, or drop a mask saved for another scan shape."""
        if not self.mask_b64:
            return
        try:
            raw = base64.b64decode(self.mask_b64)
        except Exception:
            raw = b""
        if len(raw) == self.scan_rows * self.scan_cols:
            self.mask_bytes = raw
        else:
            self.mask_b64 = ""
            self.mask_bytes = b""

    # =========================================================================
    # Standalone interactive HTML export
    # =========================================================================
    def export_html(
        self,
        path: str | pathlib.Path | None = None,
        *,
        title: str | None = None,
        mode: str = "single",
        encoding: str = "full",
        downsample: int | None = None,
    ) -> pathlib.Path:
        """Write a standalone interactive HTML viewer for this widget.

        The file mounts the widget's JS bundle with its current state (the
        navigation image, painted mask, all curves, any line-scan stack and
        every setting), so it opens in any browser with no Jupyter kernel. The
        k-space, damping and density controls recompute the curves in the
        browser from the exported I(k). A line-scan stack needs the scan
        profiles to be recomputed, so with ``encoding="full"`` its controls
        are locked. A ``"full"`` export made when no curves were computed for
        the shown selection (its compute failed) keeps its parameter controls
        locked too and shows the baked error. A float trait that is NaN or
        infinite, which is not valid JSON, raises ``ValueError`` and no file
        is written.

        Only the synced traits are embedded, never the raw 4D-STEM dataset:
        ``encoding="full"`` holds the reduced products (a few MB) and
        ``encoding="uint8"`` adds one byte per scan position and k bin.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            Destination HTML path. Defaults to the current working directory
            with the widget title, scan shape and encoding ("exact" or
            "quantized") in the name.
        title : str, optional
            Browser page title. Defaults to the widget ``title`` or "ShowPDF".
        mode : {"single"}, default "single"
            File layout. Only a single self-contained file is supported.
        encoding : {"full", "uint8"}, default "full"
            ``"full"`` embeds the float32 curves only; the selection is fixed.
            ``"uint8"`` also embeds every scan position's radial profile,
            quantized to 8 bits per k bin, so the mask, probe and line can be
            changed in the exported file.
        downsample : None
            Not supported; accepted for the shared export interface.

        Returns
        -------
        pathlib.Path
            The written HTML file.
        """
        encoding = _export_encoding(mode, encoding, downsample)
        export_path = (
            pathlib.Path(path) if path is not None else self._default_html_export_path(encoding=encoding)
        )
        self._write_html_export(export_path, title=title, encoding=encoding)
        size_mb = export_path.stat().st_size / (1024 * 1024)
        self.export_status = (
            f"Exported {export_path.name} ({size_mb:.1f} MB, {self._export_mode_label(encoding)})"
        )
        return export_path

    def _html_export_options(self, payload: dict, mode: str) -> dict:
        """``export_html`` keywords from a toolbar request: the encoding."""
        encoding = payload.get("encoding") or "full"
        return {"encoding": _export_encoding(mode, encoding, payload.get("downsample"))}

    def _export_mode_label(self, encoding: str = "full") -> str:
        """What the page holds, for the toolbar status line."""
        return _export_label(encoding == "uint8")

    def _default_html_export_path(self, *, encoding: str = "full") -> pathlib.Path:
        """``<title>_<rows>x<cols>_<exact|quantized>.html`` in the working directory."""
        kind = "quantized" if encoding == "uint8" else "exact"
        slug = export_slug(self.title, "showpdf")
        return pathlib.Path.cwd() / f"{slug}_{self.scan_rows}x{self.scan_cols}_{kind}.html"

    def _write_html_export(
        self,
        path: str | pathlib.Path,
        *,
        title: str | None = None,
        encoding: str = "full",
    ) -> pathlib.Path:
        """Embed this widget's current state into a self-contained HTML file.

        ``self`` is embedded directly (no clone), so the multi-GB source data is never
        recomputed, and only the serialized state is edited, never the live traits: a
        temporary ``self.export_enabled = False`` would race the restore over the comm and
        could leave the live Export HTML button hidden in JupyterLab.
        """
        from ipywidgets.embed import dependency_state, embed_minimal_html

        export_path = pathlib.Path(path)
        page_title = title or self.title or "ShowPDF"
        # The page needs the scan image, mask and stack that a saved notebook leaves out.
        save_state, self._save_state = self._save_state, True
        try:
            state = dependency_state([self], drop_defaults=False)
        finally:
            self._save_state = save_state
        model_entry = state.get(self.model_id, {})
        if isinstance(model_entry.get("state"), dict):
            self._bake_standalone_state(model_entry, profiles=encoding == "uint8")
        # Only now, so a refused export leaves no new directories behind.
        export_path.parent.mkdir(parents=True, exist_ok=True)
        embed_minimal_html(
            str(export_path),
            views=[self],
            title=page_title,
            drop_defaults=False,
            state=state,
        )
        ensure_mobile_viewport(export_path)
        return export_path

    def _bake_standalone_state(self, model_entry, profiles):
        """Edit this widget's serialized state into the standalone page's.

        The page has no kernel, so its export button is disabled and no export request,
        status or payload is kept (a previous export's HTML must not be nested in this one).
        It gets the exact k axis and, with ``profiles``, the quantized per-position radial
        profiles that let it recompute mask, probe and line selections.
        """
        model_state = model_entry["state"]
        _refuse_non_finite_floats(model_state)
        model_state["export_enabled"] = False
        model_state["export_request"] = ""
        model_state["export_status"] = ""
        model_state["export_payload_id"] = ""
        model_state["export_filename"] = ""
        model_state["_static_fallback_jpeg"] = ""  # the page shows the widget, not a saved picture
        model_state["offline"] = True
        replaced = {"export_payload", "offline_qq"}
        if profiles:
            replaced |= {"offline_profiles", "offline_profiles_lo", "offline_profiles_hi"}
        # Bytes traits are serialized as buffers; export_payload stays as an empty buffer so
        # the frontend still sees a (zero-length) value.
        buffers = [
            b for b in model_entry.get("buffers", [])
            if (b.get("path") or [None])[0] not in replaced
        ]
        buffers.append(_base64_buffer("export_payload", b""))
        # The exact float64 k axis: the browser recompute must select the same k bins as the
        # library (the synced curve axes are float32).
        qq = np.asarray(self._pdf.qq, dtype=np.float64).tobytes()
        buffers.append(_base64_buffer("offline_qq", qq))
        if profiles:
            q, lo, hi, rows, cols, nk = self._offline_profiles()
            model_state["offline_recompute"] = True
            model_state["offline_nk"] = nk
            model_state["offline_rows"] = rows
            model_state["offline_cols"] = cols
            buffers.append(_base64_buffer("offline_profiles", q))
            buffers.append(_base64_buffer("offline_profiles_lo", lo))
            buffers.append(_base64_buffer("offline_profiles_hi", hi))
        model_entry["buffers"] = buffers

    def _offline_profiles(self):
        """Per-position radial profiles (mean over phi), quantized to uint8 per k bin.

        Each k bin keeps its own min and max, so the high-k tail is not crushed by the
        central beam. Averaging the profiles over the selected positions reproduces
        calculate_radial_mean, which lets the standalone page recompute I(k) for any
        selection. Returns (q_uint8, lo_f32, hi_f32, rows, cols, nk), the arrays as bytes.
        """
        pol = self._pdf.polar
        arr = pol.array if pol.array is not None else to_numpy(pol.tensor)
        arr = np.asarray(arr, dtype=np.float32)  # (rows, cols, phi, k)
        profiles = arr.mean(axis=2)  # (rows, cols, k)
        rows, cols, nk = profiles.shape
        flat = profiles.reshape(rows * cols, nk)
        lo = flat.min(axis=0)
        hi = flat.max(axis=0)
        rng = np.where(hi > lo, hi - lo, np.float32(1.0))
        q = np.clip(np.round((flat - lo) / rng * 255.0), 0, 255).astype(np.uint8)
        return q.tobytes(), lo.tobytes(), hi.tobytes(), rows, cols, nk

    # =========================================================================
    # Summary
    # =========================================================================
    def summary(self) -> None:
        """Print the scan, k ranges, selection and density settings."""
        name = self.title or "ShowPDF"
        lines = [name, "═" * 32]
        lines.append(f"Scan:     {self.scan_rows} × {self.scan_cols}")
        lines.append(f"k range:  [{self.k_min_available:.2f}, {self.k_max_available:.2f}] Å⁻¹")
        lines.append(f"Fit:      k=[{self.k_min_fit:.2f}, {self.k_max_fit:.2f}]")
        k_min_window = self.k_min_window if self.k_min_window > 0 else self.k_min_fit
        k_max_window = self.k_max_window if self.k_max_window > 0 else self.k_max_fit
        lines.append(f"Window:   k=[{k_min_window:.2f}, {k_max_window:.2f}]")
        lines.append(f"Output:   r=[0.00, {self.r_max:.2f}], step={self.r_step}")
        lines.append(f"Plot:     {self.plot_mode}")
        lines += self._summary_selection_lines()
        if self._density() is not None:
            lines.append(f"Density:  {self.density_value:g} atoms/Å³")
        else:
            lines.append("Density:  not set (g(r) and damping need it)")
        if self.damp_origin_oscillations:
            skipped = "" if self._density() is not None else ", skipped until the density is set"
            lines.append(f"Damping:  ON (r_cut={self.r_cut}, max_iter={self.max_iter}{skipped})")
        print("\n".join(lines))

    def _summary_selection_lines(self):
        """summary() lines for the line, probe or mask selection."""
        selected = f"{self.mask_pixel_count} px ({self.mask_fraction * 100:.1f}%)"
        if self.analysis_mode == "line":
            if not self.line_active:
                return ["Line:     none drawn"]
            er0, ec0, er1, ec1 = self._effective_line()
            length = float(np.hypot(er1 - er0, ec1 - ec0))
            perpendicular = ", perpendicular" if self.line_perpendicular else ""
            start = f"({self.line_row0:.1f}, {self.line_col0:.1f})"
            end = f"({self.line_row1:.1f}, {self.line_col1:.1f})"
            sampled = f"({er0:.0f}, {ec0:.0f}) → ({er1:.0f}, {ec1:.0f})"
            return [
                f"Line:     {start} → {end}",
                f"Sampled:  {sampled} (ends on scan positions{perpendicular})",
                f"          len={length:.1f} px, width={self.line_width} px, {selected}",
            ]
        if self.analysis_mode == "probe":
            side = 2 * max(1, self.probe_size) - 1
            return [f"Probe:    ({self.probe_row}, {self.probe_col}), {side} × {side}, {selected}"]
        if self.mask_b64:
            return [f"Mask:     {selected}"]
        return ["Mask:     full scan (no mask)"]

    def __repr__(self) -> str:
        mask_info = (
            f", mask={self.mask_pixel_count}px"
            if self.analysis_mode == "mask" and self.mask_b64
            else ""
        )
        return (
            f"ShowPDF(scan=({self.scan_rows}, {self.scan_cols}), "
            f"k=[{self.k_min_fit:.1f}, {self.k_max_fit:.1f}]{mask_info})"
        )


# =============================================================================
# HTML export helpers
# =============================================================================
def _export_encoding(mode, encoding, downsample):
    """Check the shared export options; return the encoding, "full" or "uint8"."""
    if str(mode or "single").strip().lower() != "single":
        raise ValueError(f"ShowPDF HTML export supports mode='single', got {mode!r}")
    if downsample not in (None, 1):
        raise NotImplementedError("ShowPDF HTML export does not support downsample")
    raw = str(encoding or "full").strip().lower()
    if raw in ("full", "float32"):
        return "full"
    if raw == "uint8":
        return "uint8"
    raise ValueError(f"unknown ShowPDF export encoding {encoding!r}; expected 'full' or 'uint8'")


def _export_label(profiles):
    """Export status suffix describing what the standalone HTML contains."""
    return "exact float32 curves + quantized uint8 profiles" if profiles else "exact float32 curves"


def _base64_buffer(name, data):
    """Embedded-state buffer holding ``data`` for the Bytes trait ``name``."""
    return {"encoding": "base64", "path": [name], "data": base64.b64encode(data).decode("ascii")}


def _refuse_non_finite_floats(model_state):
    """Raise ValueError for a NaN or infinite float trait, which is not valid JSON in the page.

    The validators keep them out of the parameters; this refuses one in any other float trait.
    """
    for name, value in model_state.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(
                f"Cannot export: {name} is {value!r}, which is not valid JSON "
                "in the exported page."
            )


# =============================================================================
# save_results files. The CSV files use comma separators, dot decimals and empty
# cells for missing values, so they open cleanly in Excel and pandas.
# =============================================================================
def _format_cell(v):
    """Float as a short decimal; NaN or a non-number as an empty cell."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    if math.isnan(f):
        return ""
    return f"{f:.7g}"


def _columns_to_csv(headers, columns):
    """Ragged 1D columns as CSV, shorter columns padded with empty cells (g(r) is empty without a density)."""
    n_rows = max((len(c) for c in columns), default=0)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(headers)
    for i in range(n_rows):
        w.writerow([_format_cell(col[i]) if i < len(col) else "" for col in columns])
    return buf.getvalue()


def _stack_to_csv_with_axis(axis, stack):
    """2D stack as CSV: the axis values as the header row, then one row per stack row.

    NaN cells are written empty (line-scan stacks are NaN for samples without a curve).
    """
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([_format_cell(v) for v in axis[: stack.shape[1]]])
    for row in stack:
        w.writerow([_format_cell(v) for v in row])
    return buf.getvalue()


def _positions_to_csv(positions):
    """linescan_position.csv: each sample's index and distance from the line start in scan pixels."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["bin_index", "position_px"])
    for i, p in enumerate(positions):
        w.writerow([i, f"{float(p):.6g}"])
    return buf.getvalue()


def _format_metadata_txt(meta):
    """metadata.txt: a title and underlined sections of ``key: value`` lines.

    The 0.0 sentinels read as "disabled" for the filters, as the fit range for the window
    and as "not set" for the density; the fit bounds are numbers.
    """

    def section(title):
        return [title, "-" * len(title)]

    def filter_value(v):
        return "disabled" if v == 0.0 else f"{v:g}"

    def window_value(v):
        return "fit range" if v <= 0.0 else f"{v:g}"

    def on_off(v):
        return "on" if v else "off"

    lines = [
        "quantem ShowPDF — exported results",
        "===================================",
        "",
        f"Saved:           {meta['saved_at']}",
        f"Widget version:  {meta['widget_version']}",
        f"Title:           {meta['title']}",
        "",
    ]

    lines += section("Scan")
    sr, sc = meta["scan_shape"]
    lines.append(f"  shape:           {sr} x {sc} pixels")
    lines.append("")

    lines += section("Analysis")
    lines.append(f"  mode:              {meta['analysis_mode']}")
    if meta["line_mode"]:
        lines.append(f"  line mode:         {meta['line_mode']}")
    lines.append(f"  plot mode at save: {meta['plot_mode_at_save']}")
    lines.append("")

    region = meta["region"]
    kind = region["kind"]
    lines += section(f"Region ({kind})")
    if kind == "mask":
        lines.append(
            f"  pixels selected: {region['mask_pixel_count']} "
            f"({region['mask_fraction'] * 100:.1f}% of scan)"
        )
        lines.append(f"  mask file:       {region['mask_file']}")
    elif kind == "probe":
        lines.append(f"  centre (row, col): ({region['probe_row']}, {region['probe_col']})")
        side = 2 * region["probe_size"] - 1
        lines.append(f"  size:              {side} x {side} px")
        lines.append(
            f"  pixels selected:   {region['mask_pixel_count']} "
            f"({region['mask_fraction'] * 100:.1f}% of scan)"
        )
    elif kind == "line":
        lines.append(f"  start (row, col):  ({region['line_row0']:.2f}, {region['line_col0']:.2f})")
        lines.append(f"  end (row, col):    ({region['line_row1']:.2f}, {region['line_col1']:.2f})")
        sr0, sc0, sr1, sc1 = region["sampled_line"]
        lines.append(
            f"  sampled line:      ({sr0:.0f}, {sc0:.0f}) to ({sr1:.0f}, {sc1:.0f}), "
            "ends on scan positions; position_px is measured from its start"
        )
        lines.append(f"  width:             {region['line_width']} px")
        lines.append(f"  perpendicular:     {'yes' if region['line_perpendicular'] else 'no'}")
        lines.append(
            f"  pixels selected:   {region['mask_pixel_count']} "
            f"({region['mask_fraction'] * 100:.1f}% of scan)"
        )
        if "n_linescan_bins" in region:
            lines.append(f"  linescan bins:     {region['n_linescan_bins']}")
            lines.append(f"  line length:       {region['line_length_px']:.2f} px")
    lines.append("")

    lines += section("Fit parameters")
    fp = meta["fit_parameters"]
    density = fp["density_value"]
    density_text = f"{density:g}" if density > 0 else "not set"
    damping = on_off(fp["damp_origin_oscillations"])
    # Same test as ShowPDF._density, so a negative density also counts as not set.
    if fp["damp_origin_oscillations"] and not density > 0:
        damping += " (not applied: density not set)"
    lines.append(f"  k_min_fit:               {fp['k_min_fit']:g}")
    lines.append(f"  k_max_fit:               {fp['k_max_fit']:g}")
    lines.append(f"  k_min_window:            {window_value(fp['k_min_window'])}")
    lines.append(f"  k_max_window:            {window_value(fp['k_max_window'])}")
    lines.append(f"  k_lowpass:               {filter_value(fp['k_lowpass'])}")
    lines.append(f"  k_highpass:              {filter_value(fp['k_highpass'])}")
    lines.append(f"  r_max:                   {fp['r_max']:g}")
    lines.append(f"  r_step:                  {fp['r_step']:g}")
    lines.append(f"  damp_origin_oscillations: {damping}")
    lines.append(f"  r_cut:                   {fp['r_cut']:g}")
    lines.append(f"  max_iter:                {fp['max_iter']}")
    lines.append(f"  density_value:           {density_text}")
    lines.append("")

    return "\n".join(lines)


def _build_readme(is_linescan, analysis_mode, damped):
    """README.txt of the results zip: what each file holds and how to load it."""
    if damped:
        fk_correction = "                  (with the low-r correction applied)."
    else:
        fk_correction = "                  (no low-r correction)."
    lines = [
        "quantem ShowPDF — exported results",
        "===================================",
        "",
        "Files in this archive:",
        "",
        "  metadata.txt    Mode, region selection, fit parameters, axis info,",
        "                  timestamp. Plain text — open in any editor.",
        "",
        "  curves_k.csv    Reciprocal-space curves for the selected region:",
        "                  columns are  k, I(k), B(k), F(k).",
        "                  Background B(k) is the fit; F(k) is the windowed",
        "                  reduced structure factor used for the transform",
        fk_correction,
        "",
        "  curves_r.csv    Real-space curves: columns are  r, G(r), g(r).",
        "                  g(r) is only present when a density was set.",
        "",
    ]
    if analysis_mode == "mask":
        lines += [
            "  mask.npy        Boolean 2D array (scan_rows × scan_cols) marking",
            "                  the scan positions used. Load with numpy:",
            "                      import numpy as np",
            "                      mask = np.load('mask.npy')",
            "",
        ]
    if is_linescan:
        lines += [
            "  linescan_position.csv",
            "                  Two columns: bin_index, position_px.",
            "                  position_px is the distance from the line start",
            "                  (A) in scan pixels; positions are evenly spaced.",
            "",
            "  linescan_Ik.csv, linescan_Ik_bg.csv, linescan_Fk.csv,",
            "  linescan_Gr.csv, linescan_gr.csv",
            "                  2D stacks of per-bin curves. The FIRST row of",
            "                  each file is the axis (k for Ik/Ik_bg/Fk; r for",
            "                  Gr/gr). Each subsequent row is one bin (same",
            "                  order as linescan_position.csv). Empty cells",
            "                  indicate bins with no scan positions inside.",
            "",
        ]
    lines += [
        "Quick load in Python:",
        "",
        "    import numpy as np",
        "",
        "    # 1D curves (Excel-style: skip the header row, blank cells → NaN)",
        "    curves_k = np.genfromtxt('curves_k.csv', delimiter=',',",
        "                             names=True, missing_values='', filling_values=np.nan)",
        "    k    = curves_k['k']",
        "    Ik   = curves_k['Ik']",
        "    bg   = curves_k['bg']",
        "    Fk   = curves_k['Fk']",
        "",
        "    # metadata.txt is plain text — read it with open() or just look at it",
        "    # in a text editor; the values are also visible in any browser.",
        "",
    ]
    if is_linescan:
        lines += [
            "    # 2D linescan stack — first row is the axis, rest is data",
            "    raw = np.genfromtxt('linescan_Gr.csv', delimiter=',',",
            "                        missing_values='', filling_values=np.nan)",
            "    r_axis = raw[0]",
            "    Gr_stack = raw[1:]   # shape (n_bins, n_r)",
            "",
        ]
    lines += [
        "Excel / Google Sheets: open any .csv file directly.",
        "",
    ]
    return "\n".join(lines)
