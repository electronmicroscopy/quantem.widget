"""Unit tests for the ShowPDF widget."""

import base64
import inspect
import io
import json
import math
import pathlib
import re
import warnings
import zipfile

import numpy as np
import pytest
import traitlets

# ShowPDF needs the PairDistributionFunction API of quantem PR #248 (calculate_Gr with density
# and max_iter); skip until the installed quantem has it.
_diffraction = pytest.importorskip("quantem.diffraction")
_PDF = getattr(_diffraction, "PairDistributionFunction", None)
if _PDF is None or not {"density", "max_iter"} <= set(inspect.signature(_PDF.calculate_Gr).parameters):
    pytest.skip(
        "ShowPDF needs PairDistributionFunction.calculate_Gr(density=..., max_iter=...)",
        allow_module_level=True,
    )

from quantem.core.datastructures.dataset4dstem import Dataset4dstem  # noqa: E402
from quantem.diffraction import PairDistributionFunction  # noqa: E402
import quantem.widget.showpdf as showpdf_module  # noqa: E402
from quantem.widget import ShowPDF  # noqa: E402
from quantem.widget.showpdf import (  # noqa: E402
    _DENSITY_HINT_DAMPING,
    _DENSITY_HINT_GR,
    _WINDOW_HINT,
    _bf_nav_image,
    _export_label,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCS_PAGE = REPO_ROOT / "docs" / "api" / "showpdf.md"
FRONTEND_SOURCE = REPO_ROOT / "js" / "showpdf" / "index.tsx"


# ---------------------------------------------------------------------------
# Test data and helpers
# ---------------------------------------------------------------------------

DET = 32  # detector size of the uncalibrated test patterns
DET_CENTER = (DET - 1) / 2.0
# from_data keywords for the uncalibrated patterns: the origin is fixed at the detector center
# (find_origin=False, for speed).
FROM_DATA_KWARGS = dict(
    find_origin=False, origin_row=DET_CENTER, origin_col=DET_CENTER, num_annular_bins=36, radial_step=1.0
)


def _detector_radius():
    """Distance of each pixel of the DET x DET detector from its center."""
    rows, cols = np.mgrid[0:DET, 0:DET]
    return np.sqrt((rows - DET_CENTER) ** 2 + (cols - DET_CENTER) ** 2)


def _noisy_scan(pattern, scan, noise, seed):
    """``pattern`` at every position of a ``scan`` grid, plus Gaussian noise of std ``noise``."""
    rng = np.random.RandomState(seed)
    data = np.empty((*scan, DET, DET), dtype=np.float32)
    for row in range(scan[0]):
        for col in range(scan[1]):
            data[row, col] = pattern + noise * rng.randn(DET, DET).astype(np.float32)
    return data


def _uncalibrated_pdf(data):
    """PairDistributionFunction of a DET x DET pattern array; k is in detector pixels."""
    return PairDistributionFunction.from_data(
        Dataset4dstem.from_array(data), device="cpu", **FROM_DATA_KWARGS
    )


@pytest.fixture
def small_4d_array():
    """4x4 scan of patterns with one broad ring and a little noise, as a plain ndarray."""
    ring = np.exp(-((_detector_radius() - 5) ** 2) / 4).astype(np.float32)
    return _noisy_scan(ring, scan=(4, 4), noise=0.01, seed=7)


@pytest.fixture
def small_pdf():
    """PairDistributionFunction of a 4x4 scan of patterns with two broad rings and a little noise."""
    radius = _detector_radius()
    rings = np.exp(-((radius - 5) ** 2) / 4) + 0.3 * np.exp(-((radius - 10) ** 2) / 8)
    rings = rings.astype(np.float32)
    return _uncalibrated_pdf(_noisy_scan(rings, scan=(4, 4), noise=0.01, seed=42))


def _calibrated_data(scan=(4, 4), det=64, seed=0):
    """Amorphous-like 4D-STEM array and its detector center.

    A direct beam plus three rings whose amplitudes change across the scan, so different scan
    selections give different I(k).
    """
    rr, cc = np.mgrid[0:det, 0:det].astype(np.float64)
    c = (det - 1) / 2.0
    rad = np.hypot(rr - c, cc - c)
    beam = 400.0 * np.exp(-(rad**2) / (2 * 2.0**2)) + 30.0 * np.exp(-rad / 9.0) + 0.2
    rings = (
        6.0 * np.exp(-((rad - 9.0) ** 2) / 5.0)
        + 2.5 * np.exp(-((rad - 17.0) ** 2) / 9.0)
        + 1.2 * np.exp(-((rad - 25.0) ** 2) / 12.0)
    )
    rng = np.random.default_rng(seed)
    data = np.empty((*scan, det, det), dtype=np.float32)
    for iy in range(scan[0]):
        for ix in range(scan[1]):
            amp = 1.0 + 0.15 * (iy + 2 * ix)
            data[iy, ix] = beam + amp * rings + 0.01 * rng.standard_normal((det, det))
    return data, c


def _calibrated_pdf(scan=(4, 4), det=64, seed=0, k_step=0.05, scan_step=1.0, scan_unit="A"):
    """PairDistributionFunction of _calibrated_data with ``k_step`` 1/Å per detector pixel."""
    data, c = _calibrated_data(scan=scan, det=det, seed=seed)
    ds = Dataset4dstem.from_array(
        data,
        sampling=(scan_step, scan_step, k_step, k_step),
        units=(scan_unit, scan_unit, "A^-1", "A^-1"),
    )
    return PairDistributionFunction.from_data(
        ds, find_origin=False, origin_row=c, origin_col=c, num_annular_bins=90, device="cpu"
    )


@pytest.fixture
def cal_pdf():
    """A 4x4 scan whose k axis runs from 0 to 1.55 1/Å in steps of 0.05."""
    return _calibrated_pdf()


def _f32(buffer):
    """The float32 values of a synced bytes trait."""
    return np.frombuffer(buffer, dtype=np.float32)


def _synced(w):
    """Every synced trait of ``w`` by name, to check that a rejected call changed nothing."""
    return {name: getattr(w, name) for name in w.traits(sync=True)}


def _read_zip(path):
    with zipfile.ZipFile(path) as zf:
        return {name: zf.read(name) for name in zf.namelist()}


def _read_csv(files, name):
    """CSV file ``name`` of a save_results zip as a structured array with named columns."""
    return np.genfromtxt(files[name].decode().splitlines(), delimiter=",", names=True)


# ---------------------------------------------------------------------------
# Construction and input types
# ---------------------------------------------------------------------------


# Each input type of the constructor, built from the same 4D array.
_WIDGET_FROM = {
    "PairDistributionFunction": lambda array: ShowPDF(_uncalibrated_pdf(array)),
    "ndarray": lambda array: ShowPDF(array, device="cpu", **FROM_DATA_KWARGS),
    "torch tensor": lambda array: ShowPDF(
        pytest.importorskip("torch").from_numpy(array), device="cpu", **FROM_DATA_KWARGS
    ),
    "Dataset4dstem": lambda array: ShowPDF(Dataset4dstem.from_array(array), device="cpu", **FROM_DATA_KWARGS),
}


@pytest.mark.parametrize("kind", list(_WIDGET_FROM))
def test_showpdf_accepts_each_input_type(small_4d_array, kind):
    w = _WIDGET_FROM[kind](small_4d_array)
    assert (w.scan_rows, w.scan_cols) == (4, 4)
    for name in ("ik_y_bytes", "ik_bg_y_bytes", "fk_y_bytes", "gr_y_bytes", "nav_image_bytes"):
        assert getattr(w, name), name
    # The navigation image is the bright field, also of a tensor-backed dataset (no .array).
    np.testing.assert_allclose(w._nav_image, _bf_nav_image(small_4d_array), rtol=1e-6)


def test_showpdf_accepts_a_single_pattern():
    # A 2D pattern is analyzed as a 1 x 1 scan.
    pattern = np.random.RandomState(1).rand(DET, DET).astype(np.float32)
    w = ShowPDF(pattern, device="cpu", **FROM_DATA_KWARGS)
    assert (w.scan_rows, w.scan_cols) == (1, 1)


def test_showpdf_rejects_unsupported_ndim():
    with pytest.raises(ValueError, match="4D.*or 2D"):
        ShowPDF(np.zeros((4, 4, 4), dtype=np.float32), device="cpu")


def test_showpdf_reports_missing_library_api(monkeypatch):
    # quantem.widget imports without quantem; constructing ShowPDF names the API it needs.
    import quantem.diffraction

    class OldPairDistributionFunction:
        def calculate_Gr(self, k_min_fit=None, damp_origin_oscillations=False):
            pass

    monkeypatch.setattr(quantem.diffraction, "PairDistributionFunction", OldPairDistributionFunction)
    with pytest.raises(ImportError, match="ShowPDF needs quantem's PairDistributionFunction"):
        ShowPDF(np.zeros((2, 2, 8, 8), dtype=np.float32))


def test_showpdf_lazy_export():
    import quantem.widget as qw

    assert "ShowPDF" in qw.__all__
    assert qw.ShowPDF is ShowPDF


# ---------------------------------------------------------------------------
# Constructor keywords and parameter validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, hint",
    [
        ({"density_mode": "manual"}, "never estimated"),
        ({"line_max_bins": 64}, "renamed to linescan_max_bins"),
        ({"theme": "dark"}, "follows the notebook theme"),
        ({"k_width": 0.25}, "taper width is fixed"),
        ({"r_min": 0.5}, "r grid always starts at 0"),
        ({"k_widht": 0.3}, "typos"),
        ({"_offline": True}, "renamed to offline"),
    ],
)
def test_showpdf_rejects_unknown_kwargs(small_pdf, kwargs, hint):
    # traitlets would silently ignore an unknown keyword; ShowPDF raises with a hint instead.
    with pytest.raises(TypeError, match=hint):
        ShowPDF(small_pdf, **kwargs)


@pytest.mark.parametrize(
    "name", ["max_iter", "probe_row", "probe_col", "probe_size", "linescan_max_bins", "waterfall_count"]
)
def test_showpdf_integer_options_must_be_whole_numbers(small_pdf, name):
    # The constructor rejects 2.5, as an assignment does, instead of truncating it to 2.
    with pytest.raises(traitlets.TraitError, match=rf"^{name} must be a whole number, got 2\.5\.$"):
        ShowPDF(small_pdf, **{name: 2.5})
    for whole in (2.0, np.int64(2)):
        value = getattr(ShowPDF(small_pdf, **{name: whole}), name)
        assert value == 2 and type(value) is int


# Float parameters that must be finite. Every finite value is accepted, and 0 keeps its meaning
# (full range, follow the fit range, off, or not set).
_FINITE_PARAMS = [
    "k_min_fit", "k_max_fit", "k_min_window", "k_max_window", "k_lowpass", "k_highpass",
    "r_max", "r_step", "r_cut", "density_value",
]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")], ids=repr)
@pytest.mark.parametrize("name", _FINITE_PARAMS)
def test_showpdf_parameter_must_be_finite(cal_pdf, name, bad):
    # NaN or infinity would make the export's state JSON invalid, so every way of setting one is
    # rejected and changes nothing.
    message = rf"^{name} must be a finite number, got {bad!r}\.$"
    # A saved non-finite density_value loads as not set (see test_showpdf_old_state_nonfinite_density).
    saved_rejected = name != "density_value"
    with pytest.raises(traitlets.TraitError, match=message):
        ShowPDF(cal_pdf, **{"k_min_fit": 0.3, name: bad})
    if saved_rejected:
        with pytest.raises(traitlets.TraitError, match=message):
            ShowPDF(cal_pdf, k_min_fit=0.3, state={name: bad})

    w = ShowPDF(cal_pdf, k_min_fit=0.3, density_value=0.05)
    before = _synced(w)
    with pytest.raises(traitlets.TraitError, match=message):
        setattr(w, name, bad)
    assert _synced(w) == before
    if saved_rejected:
        # The bad key first and last: the keys loaded before it are undone too.
        for state in (
            {name: bad, "analysis_mode": "probe", "probe_size": 2, "plot_mode": "Fk"},
            {"analysis_mode": "probe", "probe_size": 2, "plot_mode": "Fk", name: bad},
        ):
            with pytest.raises(traitlets.TraitError, match=message):
                w.load_state_dict(state)
            assert _synced(w) == before

    for value in (0.0, -1.0):  # finite values, zero and negative ones included, are accepted
        setattr(w, name, value)
        assert getattr(w, name) == value


@pytest.mark.parametrize("bad", [float("nan"), float("inf")], ids=repr)
def test_showpdf_pixel_size_must_be_finite(cal_pdf, bad):
    # pixel_size sets nav_pixel_size, which the export writes into its state JSON.
    message = rf"^pixel_size must be a finite number, got {bad!r}\.$"
    with pytest.raises(ValueError, match=message):
        ShowPDF(cal_pdf, k_min_fit=0.3, pixel_size=bad)
    w = ShowPDF(cal_pdf, k_min_fit=0.3, pixel_size=2.0)
    before = _synced(w)
    with pytest.raises(ValueError, match=message):
        w.set_data(_calibrated_pdf(scan=(6, 6)), pixel_size=bad)
    assert w.pdf is cal_pdf
    assert _synced(w) == before


@pytest.mark.parametrize("size, unit", [(float("nan"), "A"), (1e308, "nm")], ids=["nan", "nm overflow"])
def test_showpdf_nonfinite_scan_calibration_counts_as_none(tmp_path, size, unit):
    # A NaN scan sampling, or an nm sampling that overflows in Å, means no calibration (as for
    # the frontend's scale bar), so nav_pixel_size stays finite and the export works.
    w = ShowPDF(_calibrated_pdf(scan_step=size, scan_unit=unit), k_min_fit=0.3)
    assert (w.nav_pixel_size, w.nav_unit) == (1.0, "px")
    assert w.export_html(tmp_path / "uncalibrated.html").exists()


@pytest.mark.filterwarnings("ignore:(invalid value|overflow) encountered:RuntimeWarning")
@pytest.mark.parametrize("k_step", [float("nan"), float("inf"), 1e300], ids=repr)
def test_showpdf_nonfinite_k_axis_names_the_calibration(cal_pdf, k_step):
    # A k axis that is not finite in float32 (1e300 overflows it) raises an error naming the
    # calibration, not the default fit range; set_data raises it before anything changes.
    pdf = _calibrated_pdf(k_step=k_step)
    message = (
        r"^The dataset's k axis is not finite \(or too large for float32\); "
        r"check the diffraction calibration\.$"
    )
    for kwargs in ({}, {"k_min_fit": 0.3}, {"k_min_fit": 0.3, "k_max_fit": 1.5}):
        with pytest.raises(ValueError, match=message):
            ShowPDF(pdf, **kwargs)
    w = ShowPDF(cal_pdf, k_min_fit=0.3)
    before = _synced(w)
    with pytest.raises(ValueError, match=message):
        w.set_data(pdf)
    assert w.pdf is cal_pdf
    assert _synced(w) == before


# ---------------------------------------------------------------------------
# Curves: the widget shows exactly what calculate_Gr computes
# ---------------------------------------------------------------------------


def test_showpdf_curves_equal_library(cal_pdf):
    params = dict(
        k_min_fit=0.3, k_max_fit=1.5, k_min_window=0.35, k_max_window=1.45,
        k_lowpass=0.06, r_max=12.0, r_step=0.05, r_cut=0.9, max_iter=3,
    )
    w = ShowPDF(cal_pdf, damp_origin_oscillations=True, density_value=0.05, **params)
    ref = _calibrated_pdf()
    ref.calculate_Gr(damp_origin_oscillations=True, density=0.05, **params)
    ref.calculate_gr(
        density=0.05, r_cut=params["r_cut"], max_iter=params["max_iter"], set_pdf_positive=True
    )
    np.testing.assert_array_equal(_f32(w.ik_y_bytes), ref.Ik.numpy())
    np.testing.assert_array_equal(_f32(w.ik_bg_y_bytes), ref.bg.numpy())
    # F(k) and G(r) are the damped ones, which the transform used.
    np.testing.assert_array_equal(_f32(w.fk_y_bytes), ref.Fk_damped.numpy())
    np.testing.assert_array_equal(_f32(w.gr_y_bytes), ref.reduced_pdf_damped.numpy())
    np.testing.assert_array_equal(_f32(w.pdf_y_bytes), ref.pdf)


def test_showpdf_default_curves_equal_library_defaults(cal_pdf):
    w = ShowPDF(cal_pdf)
    ref = _calibrated_pdf()
    ref.calculate_Gr()
    np.testing.assert_array_equal(_f32(w.ik_bg_y_bytes), ref.bg.numpy())
    np.testing.assert_array_equal(_f32(w.gr_y_bytes), ref.reduced_pdf)


def test_showpdf_shared_pdf_object(cal_pdf, tmp_path):
    # The PairDistributionFunction may be shared: another widget's selection, or a direct library
    # call, must not leak into this widget's g(r) or saved results.
    w1 = ShowPDF(cal_pdf, k_min_fit=0.3, density_value=0.05)
    w2 = ShowPDF(cal_pdf, k_min_fit=0.3)
    column = np.zeros((4, 4), dtype=bool)
    column[:, 3] = True
    w2.set_mask(column)
    w1.density_value = 0.08
    ref = ShowPDF(_calibrated_pdf(), k_min_fit=0.3, density_value=0.08)
    np.testing.assert_array_equal(_f32(w1.pdf_y_bytes), _f32(ref.pdf_y_bytes))
    np.testing.assert_array_equal(_f32(w1.gr_y_bytes), _f32(ref.gr_y_bytes))

    cal_pdf.calculate_Gr(k_min_fit=0.6, k_max_fit=1.0)  # direct library call
    files = _read_zip(w1.save_results(tmp_path / "res.zip"))
    curves_r = _read_csv(files, "curves_r.csv")
    np.testing.assert_allclose(curves_r["Gr"], _f32(w1.gr_y_bytes), rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Mask and probe selection
# ---------------------------------------------------------------------------


def test_showpdf_mask_selection(small_pdf):
    w = ShowPDF(small_pdf)
    gr_full = _f32(w.gr_y_bytes).copy()
    w.analysis_mode = "probe"
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :2] = True
    w.set_mask(mask)  # switches back to mask mode
    assert w.analysis_mode == "mask"
    assert w.mask_pixel_count == 4
    assert w.mask_fraction == pytest.approx(0.25)
    assert not np.allclose(_f32(w.gr_y_bytes), gr_full, atol=1e-10)
    w.clear_mask()
    np.testing.assert_allclose(_f32(w.gr_y_bytes), gr_full, rtol=0, atol=1e-6)


def test_showpdf_probe_selection(small_pdf):
    w = ShowPDF(small_pdf, analysis_mode="probe", probe_row=1, probe_col=1, probe_size=2)
    assert w.mask_pixel_count == 9  # a square of side 2 * probe_size - 1 = 3: rows and cols 0..2
    w.probe_row = 10  # outside the scan: no square, so the full scan is computed and counted
    assert w.mask_pixel_count == 16


# ---------------------------------------------------------------------------
# Line selection: band geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("perp", [False, True])
@pytest.mark.parametrize("width", [1, 3, 5, 7])
def test_showpdf_line_band_covers_exactly_width_rows(width, perp):
    # An interior horizontal or vertical band of odd width w covers exactly w rows or columns;
    # with line_perpendicular, a drawn horizontal line samples a vertical one.
    w = ShowPDF(_calibrated_pdf(scan=(20, 20), det=32), k_min_fit=0.3, line_perpendicular=perp)
    half = width // 2
    for line, drawn_horizontal in (((10, 4, 10, 15), True), ((3, 9, 16, 9), False)):
        w.set_line(*line, width=width)
        assert w.analysis_mode == "line" and w.line_active
        r0, c0, r1, c1 = (int(v) for v in w._effective_line())
        expected = np.zeros((20, 20), dtype=bool)
        sampled_horizontal = drawn_horizontal != perp
        if sampled_horizontal:
            assert r0 == r1
            expected[r0 - half : r0 + half + 1, min(c0, c1) : max(c0, c1) + 1] = True
        else:
            assert c0 == c1
            expected[min(r0, r1) : max(r0, r1) + 1, c0 - half : c0 + half + 1] = True
        np.testing.assert_array_equal(w.mask, expected)
        assert w.mask_pixel_count == int(w._line_band_mask().sum()) == int(expected.sum())


@pytest.mark.parametrize("line_mode", ["averaged", "linescan"])
def test_showpdf_clear_line(small_pdf, line_mode):
    # Without a line, the averaged line mode computes (and counts) the full scan, and a line scan
    # computes nothing.
    w = ShowPDF(small_pdf, plot_mode="Gr", line_mode=line_mode)
    gr_full = _f32(w.gr_y_bytes).copy()
    w.set_line(1, 0, 1, 3, width=3)
    assert (w.n_linescan > 0) == (line_mode == "linescan")
    w.clear_line()
    assert not w.line_active
    if line_mode == "averaged":
        assert w.mask_pixel_count == 16
        np.testing.assert_allclose(_f32(w.gr_y_bytes), gr_full, rtol=0, atol=1e-6)
    else:
        assert (w.mask_pixel_count, w.n_linescan, w.linescan_bytes) == (0, 0, b"")


def test_showpdf_line_perpendicular_ends_round_as_in_the_frontend(monkeypatch):
    # This line's first perpendicular end is an ulp below column 1.5 with linescan.ts's length
    # sqrt(dr * dr + dc * dc), so it snaps to 1; a hypot one ulp short (as macOS np.hypot is
    # here) would snap it to 2. Both hypots are patched one ulp short to catch either on any OS.
    w = ShowPDF(_calibrated_pdf(scan=(8, 8)), k_min_fit=0.3)
    w.set_line(2.4097257447429, 4.714252327103168, 5.5902742552571, 1.466296183411032, width=1)
    w.line_perpendicular = True

    def hypot_one_ulp_short(dr, dc):
        return math.nextafter(math.sqrt(dr * dr + dc * dc), 0.0)

    monkeypatch.setattr(showpdf_module.np, "hypot", hypot_one_ulp_short)
    monkeypatch.setattr(showpdf_module.math, "hypot", hypot_one_ulp_short)
    assert w._effective_line() == (2.0, 1.0, 6.0, 5.0)


# ---------------------------------------------------------------------------
# Line width validation: an odd number of scan pixels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", [1, 3, 99, 3.0, np.float64(7.0), np.int64(5), np.int32(9)], ids=repr)
def test_showpdf_line_width_accepts_odd_whole_numbers(small_pdf, width):
    # The constructor and set_line take any odd whole number and store it as an int.
    w = ShowPDF(small_pdf, line_width=width)
    assert w.line_width == width and type(w.line_width) is int
    w = ShowPDF(small_pdf)
    w.set_line(1, 0, 1, 3, width=width)
    assert w.line_width == width and type(w.line_width) is int


@pytest.mark.parametrize("bad", [2, 4, 0, -1])
def test_showpdf_line_width_must_be_odd(small_pdf, bad):
    with pytest.raises(traitlets.TraitError, match=r"line_width must be an odd number of scan pixels"):
        ShowPDF(small_pdf, line_width=bad)
    w = ShowPDF(small_pdf)
    gr = w.gr_y_bytes
    with pytest.raises(traitlets.TraitError, match="line_width"):
        w.line_width = bad

    # A rejected set_line changes nothing: not the endpoints, the mode or the width.
    with pytest.raises(traitlets.TraitError, match="line_width"):
        w.set_line(1, 0, 1, 3, width=bad)
    assert (w.line_row0, w.line_col0, w.line_row1, w.line_col1) == (0.0, 0.0, 0.0, 0.0)
    assert not w.line_active
    assert w.analysis_mode == "mask"
    assert w.line_width == 3
    assert w.gr_y_bytes == gr

    # Nor with a line already set, in probe mode.
    w.set_line(0, 0, 3, 3, width=5)
    w.analysis_mode = "probe"
    with pytest.raises(traitlets.TraitError, match="line_width"):
        w.set_line(1, 0, 1, 3, width=bad)
    assert (w.line_row0, w.line_col0, w.line_row1, w.line_col1) == (0.0, 0.0, 3.0, 3.0)
    assert w.line_active
    assert w.analysis_mode == "probe"
    assert w.line_width == 5


@pytest.mark.parametrize(
    "bad",
    [5.5, 4.9, np.float64(2.5), float("nan"), float("inf"), 10**400 + 1, 10**400, -(10**400) - 1],
    ids=["5.5", "4.9", "np.float64(2.5)", "nan", "inf", "huge odd", "huge even", "huge negative"],
)
def test_showpdf_line_width_rejects_other_numbers(small_pdf, bad):
    # int() would truncate 5.5 to 5, and a width too large for a float would overflow the band
    # test's division by 2.0. The TraitError shows the width as given, and nothing changes.
    message = re.escape(f"line_width must be an odd number of scan pixels (1, 3, 5, ...), got {bad!r}.")
    with pytest.raises(traitlets.TraitError, match=message):
        ShowPDF(small_pdf, line_width=bad)
    w = ShowPDF(small_pdf)
    w.set_line(0, 0, 3, 3, width=5)
    before = _synced(w)
    with pytest.raises(traitlets.TraitError, match=message):
        w.set_line(1, 0, 1, 3, width=bad)
    with pytest.raises(traitlets.TraitError):
        w.line_width = bad
    assert _synced(w) == before
    w.line_width = 7  # the band works afterwards
    assert w.mask_pixel_count == int(w._line_band_mask().sum())


# ---------------------------------------------------------------------------
# Line end validation: finite and within ±1e9
# ---------------------------------------------------------------------------

_LINE_ENDS = ["line_row0", "line_col0", "line_row1", "line_col1"]


@pytest.mark.parametrize("line", [(None, 0, 1, 3), (1, "x", 1, 3), (1, 0, 1, None)])
def test_showpdf_set_line_bad_endpoint_changes_nothing(cal_pdf, line):
    # Every argument is converted before anything is assigned, so a bad endpoint next to a valid
    # new width leaves every synced trait unchanged.
    w = ShowPDF(cal_pdf, k_min_fit=0.3)
    w.set_line(1, 0, 1, 3, width=3)
    before = _synced(w)
    with pytest.raises((TypeError, ValueError)):
        w.set_line(*line, width=7)
    assert _synced(w) == before


@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), float("-inf"), 10**400, -(10**400)], ids=["nan", "inf", "-inf", "huge", "-huge"]
)
@pytest.mark.parametrize("position, name", enumerate(["row0", "col0", "row1", "col1"]))
def test_showpdf_set_line_end_must_be_finite_within_1e9(cal_pdf, position, name, bad):
    # float() accepts NaN and infinity, and an int too large for a float is beyond ±1e9 (a
    # ValueError, not an OverflowError): no band can be computed from them, so set_line rejects
    # them before assigning anything, the width included.
    w = ShowPDF(cal_pdf, k_min_fit=0.3)
    w.set_line(1, 0, 1, 3, width=3)
    before = _synced(w)
    line = [2, 0, 2, 3]
    line[position] = bad
    message = re.escape(f"{name} must be a finite number within ±1e9, got {bad!r}.")
    with pytest.raises(ValueError, match=message):
        w.set_line(*line, width=7)
    assert _synced(w) == before


@pytest.mark.parametrize("perp", [False, True])
@pytest.mark.parametrize(
    "line", [(0, 0, 1e200, 1e200), (1e308, 0, -1e308, 0), (0, 0, 1e154, 1e154), (0, 0, 5, 1e300)]
)
def test_showpdf_set_line_with_overflowing_ends_changes_nothing(cal_pdf, line, perp):
    # Finite ends this large would overflow the perpendicular bisector's length or an end
    # difference; set_line rejects them before switching to the line, so later recomputes work.
    w = ShowPDF(cal_pdf, k_min_fit=0.3, line_perpendicular=perp)
    before = _synced(w)
    with pytest.raises(ValueError, match=r"^(row|col)[01] must be a finite number within ±1e9, got "):
        w.set_line(*line)
    assert _synced(w) == before
    w.k_lowpass = 0.05
    assert not w.status_message.startswith("Error")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), 1e9 + 1, -1e154, 1e308], ids=repr)
@pytest.mark.parametrize("name", _LINE_ENDS)
def test_showpdf_line_end_trait_must_be_finite_within_1e9(cal_pdf, name, bad):
    # A NaN or infinite end would make the export's state JSON invalid, and a larger one could
    # overflow the band geometry. The constructor, assignment, load_state_dict and an update from
    # the frontend (the end with a line_version bump) each reject it and change nothing.
    message = re.escape(f"{name} must be a finite number within ±1e9, got {bad!r}.")
    with pytest.raises(traitlets.TraitError, match=message):
        ShowPDF(cal_pdf, k_min_fit=0.3, analysis_mode="line", **{name: bad})
    w = ShowPDF(cal_pdf, k_min_fit=0.3, line_perpendicular=True)
    w.set_line(1, 0, 1, 3)
    before = _synced(w)
    with pytest.raises(traitlets.TraitError, match=message):
        setattr(w, name, bad)
    for state in ({name: bad, "line_mode": "linescan"}, {"line_mode": "linescan", name: bad}):
        with pytest.raises(traitlets.TraitError, match=message):
            w.load_state_dict(state)
    with pytest.raises(traitlets.TraitError, match=message):
        w.set_state({name: bad, "line_active": True, "line_version": w.line_version + 1})
    assert _synced(w) == before
    # An end of exactly ±1e9 is accepted.
    setattr(w, name, math.copysign(1e9, bad))
    w.line_version += 1
    assert getattr(w, name) == math.copysign(1e9, bad)
    assert not w.status_message.startswith("Error")


def test_showpdf_line_end_error_states_the_limit():
    # The limit and its message come from one text, spelled "1e9" as in the docs and the
    # set_line docstring (a :g format would give "1e+09").
    text = showpdf_module._LINE_END_LIMIT_TEXT
    limit = showpdf_module._LINE_END_LIMIT
    assert limit == float(text)
    assert showpdf_module._LINE_END_ERROR.format("line_row0", 2 * limit) == (
        f"line_row0 must be a finite number within ±{text}, got {2 * limit!r}."
    )
    docs = DOCS_PAGE.read_text(encoding="utf-8")
    for source in (docs, ShowPDF.set_line.__doc__):
        assert f"±{text}" in source


# ---------------------------------------------------------------------------
# Line scan: one curve per sample along the line
# ---------------------------------------------------------------------------

# The synced y values of each plot_mode's curve.
_CURVE_Y_TRAIT = {"Ik": "ik_y_bytes", "Fk": "fk_y_bytes", "Gr": "gr_y_bytes", "gr": "pdf_y_bytes"}


def _line_bins(w, n_bins):
    """Each line-scan sample's band positions, reconstructed independently of _linescan_samples.

    n_bins evenly spaced samples from A to B; each band position goes to its nearest sample.
    """
    r0, c0, r1, c1 = w._effective_line()
    dr, dc = r1 - r0, c1 - c0
    seg = dr * dr + dc * dc
    length = np.sqrt(seg)
    rows = np.arange(w.scan_rows)[:, None].astype(float)
    cols = np.arange(w.scan_cols)[None, :].astype(float)
    pr, pc = rows - r0, cols - c0
    t = (pr * dr + pc * dc) / seg
    perp = np.abs(pr * dc - pc * dr) / length
    band = (t >= 0) & (t <= 1) & (perp <= max(0.5, w.line_width / 2))
    if n_bins > 1:
        bin_idx = np.minimum(np.floor(np.clip(t, 0, 1) * (n_bins - 1) + 0.5).astype(int), n_bins - 1)
    else:
        bin_idx = np.zeros(t.shape, dtype=int)
    return band, bin_idx


def _assert_linescan_matches_masked(pdf, w, **kwargs):
    """Every line-scan sample equals a ShowPDF computed on that sample's positions alone."""
    N, M = w.n_linescan, w.linescan_n_points
    stack = _f32(w.linescan_bytes).reshape(N, M)
    assert len(w.linescan_axis_bytes) == M * 4
    band, bin_idx = _line_bins(w, N)
    checked = 0
    for i in range(N):
        bin_mask = band & (bin_idx == i)
        if not bin_mask.any():
            continue
        ref_w = ShowPDF(pdf, plot_mode=w.plot_mode, **kwargs)
        ref_w.set_mask(bin_mask)
        ref = _f32(getattr(ref_w, _CURVE_Y_TRAIT[w.plot_mode]))[:M]
        # Each sample runs the same calculate_Gr as a masked compute; only the summation order of
        # the radial mean differs (float32 rounding).
        scale = max(float(np.abs(ref).max()), 1e-6)
        assert np.abs(stack[i] - ref).max() / scale < 1e-4, f"bin {i}"
        checked += 1
    assert checked > 0


def test_showpdf_linescan_bin_matches_averaged():
    # Well-conditioned data: a central beam keeps I(k) away from zero, so the background fit has
    # a unique minimum.
    radius = _detector_radius()
    pattern = (
        500.0 * np.exp(-(radius**2) / (2 * 2.5**2))
        + np.exp(-((radius - 5) ** 2) / 4)
        + 0.3 * np.exp(-((radius - 10) ** 2) / 8)
    ).astype(np.float32)
    pdf = _uncalibrated_pdf(_noisy_scan(pattern, scan=(4, 4), noise=0.05, seed=0))

    w = ShowPDF(pdf, plot_mode="Gr")
    w.set_line(0, 0, 3, 3, width=3)
    assert w.n_linescan == 0  # nothing is computed in averaged mode
    w.line_mode = "linescan"
    _assert_linescan_matches_masked(pdf, w)


def test_showpdf_linescan_after_masked_compute_uses_bin_data(cal_pdf):
    # A masked compute leaves the radial-mean cache keyed to its mask; every sample must still use
    # its own I(k). The data vary across the scan, so a stale I(k) would show.
    w = ShowPDF(cal_pdf, plot_mode="Ik")
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2] = True
    w.set_mask(mask)
    w.set_line(0, 0, 0, 3, width=1)
    w.line_mode = "linescan"
    _assert_linescan_matches_masked(cal_pdf, w)


def test_showpdf_linescan_leaves_the_band_average_consistent(cal_pdf, tmp_path):
    # After a line scan, and after saving one, the averaged line computes the band average, not
    # the last sample's I(k).
    w = ShowPDF(cal_pdf, k_min_fit=0.3, density_value=0.05)
    w.set_line(0, 0, 3, 3, width=3)
    ref = ShowPDF(_calibrated_pdf(), k_min_fit=0.3, density_value=0.05)
    ref.set_mask(w.mask)
    for save in (False, True):
        w.line_mode = "linescan"
        assert w.n_linescan > 0
        if save:
            w.save_results(tmp_path / "scan.zip")
        w.line_mode = "averaged"
        np.testing.assert_allclose(_f32(w.gr_y_bytes), _f32(ref.gr_y_bytes), rtol=0, atol=1e-5)


@pytest.mark.parametrize("plot_mode", ["Ik", "Fk", "Gr", "gr"])
def test_showpdf_linescan_with_damping_and_density(cal_pdf, plot_mode):
    kwargs = dict(k_min_fit=0.3, damp_origin_oscillations=True, density_value=0.05)
    w = ShowPDF(cal_pdf, plot_mode=plot_mode, **kwargs)
    w.set_line(0, 0, 3, 3, width=3)
    w.line_mode = "linescan"
    assert w.status_message == ""
    assert w.n_linescan > 0
    _assert_linescan_matches_masked(cal_pdf, w, **kwargs)


def test_showpdf_linescan_with_lowpass(cal_pdf):
    # Each sample's I(k) is float32 like calculate_radial_mean's, as the float32 Gaussian filter
    # kernel needs.
    w = ShowPDF(cal_pdf, k_min_fit=0.3, k_lowpass=0.05)
    w.set_line(0, 0, 3, 3, width=3)
    w.line_mode = "linescan"
    assert not w.status_message.startswith("Line-scan error"), w.status_message
    assert w.n_linescan > 0


def test_showpdf_linescan_plot_mode_recompute(small_pdf):
    w = ShowPDF(small_pdf, plot_mode="Gr")
    w.set_line(1, 0, 1, 3, width=3)
    w.line_mode = "linescan"
    m_gr = w.linescan_n_points
    assert m_gr > 0

    w.plot_mode = "Ik"
    assert w.n_linescan > 0
    assert w.linescan_n_points > 0
    # I(k) is sampled on the k grid, not the r grid, so the point count changes with the curve.
    assert w.linescan_n_points != m_gr


def test_showpdf_linescan_clips_line_to_scan(cal_pdf, tmp_path, capsys):
    # The perpendicular bisector of a line on the top edge leaves the scan; the sampled line stops
    # at the edge, so its tip has data, and summary and metadata.txt report the clipped ends.
    w = ShowPDF(cal_pdf, k_min_fit=0.3, line_perpendicular=True)
    w.set_line(0, 0, 0, 3, width=1)
    w.line_mode = "linescan"
    er0, ec0, er1, ec1 = w._effective_line()
    assert all(0 <= v <= 3 for v in (er0, ec0, er1, ec1))
    stack = _f32(w.linescan_bytes).reshape(w.n_linescan, w.linescan_n_points)
    assert np.isfinite(stack).any(axis=1).all()
    w.summary()
    assert f"Sampled:  ({er0:.0f}, {ec0:.0f}) → ({er1:.0f}, {ec1:.0f})" in capsys.readouterr().out
    meta = _read_zip(w.save_results(tmp_path / "s.zip"))["metadata.txt"].decode()
    assert f"sampled line:      ({er0:.0f}, {ec0:.0f}) to ({er1:.0f}, {ec1:.0f})" in meta


def test_showpdf_linescan_thin_diagonal_takes_one_position_per_sample(cal_pdf):
    w = ShowPDF(cal_pdf, k_min_fit=0.3, plot_mode="Ik")
    w.set_line(0, 0, 3, 3, width=1)
    w.line_mode = "linescan"
    band, idx, n, _length, _positions = w._linescan_samples()
    assert n == 4
    for i in range(n):
        assert list(zip(*np.nonzero(band & (idx == i)))) == [(i, i)]
    # Each row of the I(k) stack is exactly that scan position's I(k).
    stack = _f32(w.linescan_bytes).reshape(n, w.linescan_n_points)
    for i in range(n):
        one = np.zeros((4, 4), dtype=bool)
        one[i, i] = True
        ref = ShowPDF(_calibrated_pdf(), k_min_fit=0.3, plot_mode="Ik")
        ref.set_mask(one)
        np.testing.assert_allclose(stack[i], _f32(ref.ik_y_bytes), rtol=1e-6)


def test_showpdf_linescan_max_bins(small_pdf):
    w = ShowPDF(small_pdf)
    w.set_line(0, 0, 3, 3, width=3)
    w.line_mode = "linescan"
    assert w.n_linescan == 4  # one sample per scan step along the longer axis
    w.linescan_max_bins = 2
    assert w.n_linescan == 2
    restored = ShowPDF(small_pdf, state=w.state_dict())
    assert (restored.linescan_max_bins, restored.n_linescan) == (2, 2)
    for bad in (0, -3):
        with pytest.raises(traitlets.TraitError, match="^linescan_max_bins must be at least 1"):
            ShowPDF(small_pdf, linescan_max_bins=bad)


@pytest.mark.parametrize(
    "name, default, value, bad, message",
    [
        (
            "linescan_view", "heatmap", "waterfall", "bogus",
            "linescan_view must be 'heatmap' or 'waterfall', got 'bogus'",
        ),
        ("waterfall_count", 10, 5, 0, "waterfall_count must be at least 1, got 0"),
    ],
)
def test_showpdf_linescan_view_options(small_pdf, name, default, value, bad, message):
    w = ShowPDF(small_pdf)
    assert getattr(w, name) == default
    restored = ShowPDF(small_pdf, state=ShowPDF(small_pdf, **{name: value}).state_dict())
    assert getattr(restored, name) == value  # saved, and so exported
    with pytest.raises(traitlets.TraitError, match=message):
        ShowPDF(small_pdf, **{name: bad})
    with pytest.raises(traitlets.TraitError, match=message):  # an assignment is checked too
        setattr(w, name, bad)
    assert getattr(w, name) == default


# ---------------------------------------------------------------------------
# Line-scan sample positions
# ---------------------------------------------------------------------------


class _LineStub:
    """ShowPDF's line geometry methods on a scan of the given size, without a dataset."""

    _line_band_mask = ShowPDF._line_band_mask
    _effective_line = ShowPDF._effective_line
    _linescan_samples = ShowPDF._linescan_samples

    def __init__(self, line, width, perp, rows=40, cols=40, max_bins=256):
        self.scan_rows, self.scan_cols = rows, cols
        self.line_row0, self.line_col0, self.line_row1, self.line_col1 = line
        self.line_width, self.line_perpendicular = width, perp
        self.line_active, self.linescan_max_bins = True, max_bins


def _check_samples(band, idx, n, length, positions, eff, rows, cols):
    """Assert that the samples run evenly from A to the tip B and none is empty."""
    r0, c0, r1, c1 = eff
    assert all(float(v).is_integer() for v in eff)  # the sampled line ends on scan positions
    assert n == round(max(abs(r1 - r0), abs(c1 - c0))) + 1
    # evenly spaced from A (0) to the tip B (the line length)
    np.testing.assert_allclose(positions, np.linspace(0.0, length, n), rtol=0, atol=1e-12)
    assert positions[-1] == pytest.approx(np.hypot(r1 - r0, c1 - c0))
    # no empty samples; every band position goes to its nearest sample
    rr = np.arange(rows, dtype=float)[:, None]
    cc = np.arange(cols, dtype=float)[None, :]
    s = ((rr - r0) * (r1 - r0) + (cc - c0) * (c1 - c0)) / length
    assert (np.bincount(idx[band], minlength=n) > 0).all()
    assert (np.abs(s[band] - positions[idx[band]]) <= length / (n - 1) / 2 + 1e-9).all()


@pytest.mark.parametrize(
    "line, width, perp",
    [
        ((0, 0, 0, 10), 1, False),
        ((1.3, 0.4, 15.8, 12.1), 5, False),
        ((19, 2, 3, 15), 3, False),
        # the perpendicular bisector of a line with odd dr + dc has half-pixel ends
        ((5, 3, 12, 9), 1, True),
    ],
)
def test_showpdf_linescan_positions_are_evenly_spaced(line, width, perp):
    # The widget's own samples; the sweep below checks the geometry for many more lines.
    w = ShowPDF(_calibrated_pdf(scan=(20, 20), det=32), k_min_fit=0.3, line_perpendicular=perp)
    w.set_line(*line, width=width)
    w.line_mode = "linescan"
    band, idx, n, length, positions = w._linescan_samples()
    _check_samples(band, idx, n, length, positions, w._effective_line(), w.scan_rows, w.scan_cols)
    assert w.n_linescan == n
    stack = _f32(w.linescan_bytes).reshape(w.n_linescan, w.linescan_n_points)
    assert np.isfinite(stack).any(axis=1).all()


def test_showpdf_linescan_sweep_ends_at_the_tip_without_empty_samples():
    # Every line from (20, 20) with integer ends up to 12 away, random fractional lines and lines
    # that leave the 40x40 scan (clipped), as drawn and perpendicular, thin and wide.
    rng = np.random.default_rng(3)
    lines = [(20, 20, 20 + dr, 20 + dc) for dr in range(-12, 13) for dc in range(-12, 13) if dr or dc]
    lines += [tuple(v) for v in rng.uniform(4, 36, size=(200, 4))]
    lines += [tuple(v) for v in rng.uniform(-10, 50, size=(200, 4))]
    checked = 0
    for line in lines:
        for perp in (False, True):
            for width in (1, 3):
                stub = _LineStub(line, width, perp)
                samples = stub._linescan_samples()
                if samples is None or samples[2] < 2:  # no band, or a single sample
                    continue
                assert all(0 <= v <= 39 for v in stub._effective_line())  # clipped to the scan
                _check_samples(*samples, stub._effective_line(), 40, 40)
                checked += 1
    assert checked > 2000


# ---------------------------------------------------------------------------
# Density (never estimated) and the hints that ask for it
# ---------------------------------------------------------------------------


def test_showpdf_density_is_needed_for_gr_and_damping(small_pdf):
    # G(r) needs no density; g(r) and the low-r correction do, and the status line asks for it.
    w = ShowPDF(small_pdf)
    assert (w.density_value, w.n_points_pdf) == (0.0, 0)
    assert w.n_points_gr > 0
    assert w.status_message.startswith("Ready")
    w.plot_mode = "gr"
    assert w.status_message == _DENSITY_HINT_GR
    w.plot_mode = "Gr"
    assert w.status_message == ""
    w.damp_origin_oscillations = True
    assert small_pdf.reduced_pdf_damped is None
    assert w.status_message == _DENSITY_HINT_DAMPING
    # With a density, the correction runs and g(r) is computed, clamped at 0.
    w.density_value = 0.05
    assert small_pdf.reduced_pdf_damped is not None
    assert small_pdf.rho0 == 0.05
    assert w.n_points_pdf == w.n_points_gr
    assert _f32(w.pdf_y_bytes).min() >= 0.0
    assert w.status_message == ""
    # Unsetting it empties g(r) again.
    w.damp_origin_oscillations = False
    w.plot_mode = "gr"
    w.density_value = 0.0
    assert w.n_points_pdf == 0
    assert w.status_message == _DENSITY_HINT_GR


def test_showpdf_density_change_resyncs_damped_curves(cal_pdf):
    # With damping on, a new density changes G(r) and F(k), not just g(r).
    kwargs = dict(k_min_fit=0.3, damp_origin_oscillations=True)
    w = ShowPDF(cal_pdf, density_value=0.05, **kwargs)
    gr_05 = _f32(w.gr_y_bytes).copy()
    w.density_value = 0.08
    ref = ShowPDF(_calibrated_pdf(), density_value=0.08, **kwargs)
    assert not np.array_equal(_f32(w.gr_y_bytes), gr_05)
    np.testing.assert_array_equal(_f32(w.gr_y_bytes), _f32(ref.gr_y_bytes))
    np.testing.assert_array_equal(_f32(w.fk_y_bytes), _f32(ref.fk_y_bytes))
    np.testing.assert_array_equal(_f32(w.pdf_y_bytes), _f32(ref.pdf_y_bytes))


def test_showpdf_linescan_density_hints(cal_pdf):
    w = ShowPDF(cal_pdf, plot_mode="Gr", damp_origin_oscillations=True, analysis_mode="line", line_mode="linescan")
    assert w.n_linescan == 0  # no line yet
    assert w.status_message == _DENSITY_HINT_DAMPING
    w.set_line(0, 0, 3, 3, width=3)
    assert w.n_linescan > 0  # the undamped G(r); the hint says why
    assert w.status_message == _DENSITY_HINT_DAMPING
    w.damp_origin_oscillations = False
    w.plot_mode = "gr"  # a g(r) stack needs the density
    assert w.n_linescan == 0
    assert w.status_message == _DENSITY_HINT_GR
    w.density_value = 0.05
    assert w.n_linescan > 0
    assert w.status_message == ""


# ---------------------------------------------------------------------------
# k window and the empty-window hint
# ---------------------------------------------------------------------------


def test_showpdf_window_defaults_follow_fit_range(cal_pdf):
    w = ShowPDF(cal_pdf, k_min_fit=0.3, k_max_fit=1.4)
    assert w.k_min_window == 0.0 and w.k_max_window == 0.0
    assert cal_pdf.kmin_window == pytest.approx(0.3)
    assert cal_pdf.kmax_window == pytest.approx(1.4)
    w.k_min_fit = 0.4
    assert cal_pdf.kmin_window == pytest.approx(0.4)
    w.k_min_window = 0.5
    assert cal_pdf.kmin_window == pytest.approx(0.5)


# Windows with and without positive weight on cal_pdf's k axis (step 0.05 1/Å, ending at 1.55).
_EMPTY_WINDOWS = {
    "lower end above the upper end, which follows the fit range": dict(k_max_fit=0.8, k_min_window=1.0),
    "equal ends": dict(k_min_window=0.5, k_max_window=0.5),
    "narrower than one k step": dict(k_min_window=0.51, k_max_window=0.54),
    # a float64 hair outside two adjacent k bins: empty in the library's float32
    "ends on two adjacent k bins": dict(k_min_window=0.5 - 1e-9, k_max_window=0.55 + 1e-9),
}
_VALID_WINDOWS = {
    "following the fit range": dict(),
    "set inside the fit range": dict(k_min_window=0.35, k_max_window=1.45),
    "narrower than one k step around a k bin": dict(k_min_window=0.49, k_max_window=0.51),
}


@pytest.mark.parametrize(
    "window, empty",
    [(w, True) for w in _EMPTY_WINDOWS.values()] + [(w, False) for w in _VALID_WINDOWS.values()],
    ids=list(_EMPTY_WINDOWS) + list(_VALID_WINDOWS),
)
def test_showpdf_empty_window_hint(cal_pdf, window, empty):
    # An empty window leaves F(k) and G(r) zero, but for the float32 Lorch weight at a k bin on the
    # upper end (sin(pi) is about -8.7e-8); the hint uses the library's own window, so it agrees.
    w = ShowPDF(cal_pdf, k_min_fit=0.3, density_value=0.05, **window)
    fk, gr = _f32(w.fk_y_bytes), _f32(w.gr_y_bytes)
    assert (w.status_message == _WINDOW_HINT) == empty
    assert empty or w.status_message.startswith("Ready")
    w.plot_mode = "gr"
    assert (w.status_message == _WINDOW_HINT) == empty
    # A valid window clears the hint; setting the window again brings it back.
    w.k_max_fit, w.k_min_window, w.k_max_window = 1.5, 0.0, 0.0
    assert w.status_message == ""
    full_fk, full_gr = _f32(w.fk_y_bytes), _f32(w.gr_y_bytes)
    assert np.isfinite(full_fk).all()
    # Zero relative to the full window's curves; the one k bin inside the narrowest valid window
    # gives over 1e-3 of them.
    assert (np.abs(fk).max() < 1e-6 * np.abs(full_fk).max()) == empty
    assert (np.abs(gr).max() < 1e-6 * np.abs(full_gr).max()) == empty
    for name, value in window.items():
        setattr(w, name, value)
    assert w.status_message == (_WINDOW_HINT if empty else "")


def test_showpdf_window_hint_before_density_hint(cal_pdf):
    w = ShowPDF(cal_pdf, k_min_fit=0.3, damp_origin_oscillations=True)
    assert w.status_message == _DENSITY_HINT_DAMPING
    w.k_min_window = 1.6  # above the upper end, which follows the fit range
    assert w.status_message == _WINDOW_HINT
    w.plot_mode = "gr"
    w.damp_origin_oscillations = False
    assert w.status_message == _WINDOW_HINT
    # With the window valid again, the density hint is back.
    w.k_min_window = 0.0
    assert w.status_message == _DENSITY_HINT_GR
    w.damp_origin_oscillations = True
    assert w.status_message == _DENSITY_HINT_DAMPING


def test_showpdf_linescan_empty_window_hint(cal_pdf):
    w = ShowPDF(cal_pdf, k_min_fit=0.3, plot_mode="gr", analysis_mode="line", line_mode="linescan")
    assert w.status_message == _DENSITY_HINT_GR  # no line yet
    w.k_min_window = 1.6
    assert w.status_message == _WINDOW_HINT
    w.set_line(0, 0, 3, 3, width=3)
    assert w.status_message == _WINDOW_HINT
    w.k_min_window = 0.0
    assert w.status_message == _DENSITY_HINT_GR
    w.density_value = 0.05
    assert w.status_message == ""
    w.plot_mode = "Fk"
    full = _f32(w.linescan_bytes)
    w.k_max_window = 0.3  # equal to the lower end, which follows k_min_fit
    assert w.status_message == _WINDOW_HINT
    # Zero but for the library's float32 Lorch weight at k = 0.3 (about -3e-8).
    assert w.n_linescan > 0 and np.abs(_f32(w.linescan_bytes)).max() < 1e-6 * np.abs(full).max()
    w.k_max_window = 0.0
    assert w.status_message == ""


def test_showpdf_error_wins_over_window_hint(cal_pdf):
    w = ShowPDF(cal_pdf, k_min_fit=0.3, k_min_window=1.6)
    assert w.status_message == _WINDOW_HINT
    w.r_max = -1.0  # empty r grid: calculate_Gr raises
    assert w.status_message.startswith("Error")
    w.r_max = 20.0
    assert w.status_message == _WINDOW_HINT
    w.set_line(0, 0, 3, 3, width=3)
    w.line_mode = "linescan"
    assert w.status_message == _WINDOW_HINT
    w.r_max = -1.0
    assert w.status_message.startswith("Line-scan error")


# ---------------------------------------------------------------------------
# Failed computes: the shown curves always belong to the shown selection
# ---------------------------------------------------------------------------

_CURVE_TRAITS = (
    "n_points_ik", "ik_x_bytes", "ik_y_bytes", "ik_bg_y_bytes", "n_points_fk", "fk_x_bytes",
    "fk_y_bytes", "n_points_gr", "gr_x_bytes", "gr_y_bytes", "n_points_pdf", "pdf_x_bytes",
    "pdf_y_bytes",
)


def _curves(w):
    """The synced curve traits of ``w`` by name."""
    return {name: getattr(w, name) for name in _CURVE_TRAITS}


def _columns(*cols):
    """A 4x4 scan mask that selects the given columns."""
    mask = np.zeros((4, 4), dtype=bool)
    mask[:, list(cols)] = True
    return mask


# Each case sets up a selection, then selects other scan positions or data.
_SELECTION_CHANGES = {
    "mask": (lambda w: w.set_mask(_columns(0, 1)), lambda w: w.set_mask(_columns(2, 3))),
    "analysis mode": (lambda w: w.set_mask(_columns(0, 1)), lambda w: setattr(w, "analysis_mode", "probe")),
    "probe position": (lambda w: setattr(w, "analysis_mode", "probe"), lambda w: setattr(w, "probe_col", 0)),
    "probe size": (lambda w: setattr(w, "analysis_mode", "probe"), lambda w: setattr(w, "probe_size", 2)),
    "line": (lambda w: w.set_line(0, 0, 0, 3, width=1), lambda w: w.set_line(3, 0, 3, 3)),
    "line width": (lambda w: w.set_line(0, 0, 0, 3, width=1), lambda w: setattr(w, "line_width", 3)),
    "perpendicular": (
        lambda w: w.set_line(1, 0, 1, 3, width=1), lambda w: setattr(w, "line_perpendicular", True)
    ),
    "data": (lambda w: None, lambda w: w.set_data(_calibrated_pdf(seed=1))),
}


@pytest.mark.parametrize("change", list(_SELECTION_CHANGES))
def test_showpdf_failed_compute_clears_curves_of_another_selection(change):
    setup, select_other = _SELECTION_CHANGES[change]
    w = ShowPDF(_calibrated_pdf(), k_min_fit=0.3, density_value=0.05)
    setup(w)
    curves = _curves(w)
    assert all(curves.values())
    # A refused parameter keeps the curves: they still belong to the shown selection.
    w.r_max = -1.0
    assert w.status_message.startswith("Error")
    assert _curves(w) == curves
    # For other positions or data they would describe the previous selection (and a full export
    # would recompute G(r) from its I(k)), so they are cleared.
    select_other(w)
    assert w.status_message.startswith("Error")
    assert not any(_curves(w).values())
    # The next successful compute fills them for the new selection.
    w.r_max = 20.0
    ref = ShowPDF(w.pdf, state=w.state_dict())
    assert w.status_message == ref.status_message == ""
    assert _curves(w) == _curves(ref)


def test_showpdf_failed_compute_keeps_curves_of_the_same_positions(cal_pdf):
    # Both lines snap to the same scan positions, so the curves still describe the shown selection.
    w = ShowPDF(cal_pdf, k_min_fit=0.3)
    w.set_line(1, 0, 1, 3, width=1)
    w.r_max = -1.0
    curves = _curves(w)
    w.set_line(1.2, 0, 0.8, 3)
    assert w.status_message.startswith("Error")
    assert _curves(w) == curves


# ---------------------------------------------------------------------------
# set_data: replacing the dataset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", [{}, {"device": None}, {"device": "cpu"}], ids=["default", "None", "cpu"])
def test_showpdf_set_data_takes_an_array_on_the_cpu(small_pdf, small_4d_array, device):
    w = ShowPDF(small_pdf)
    w.set_data(small_4d_array[:2, :3], **device, **FROM_DATA_KWARGS)
    assert (w.scan_rows, w.scan_cols) == (2, 3)
    assert w.n_points_gr > 0
    assert str(w.pdf.device) == "cpu"


def test_showpdf_set_data_resets_selection(cal_pdf):
    # The mask, probe and line refer to the old scan grid, so they are reset (the 4x4 mask would
    # not even reshape to the new 2x2 scan).
    w = ShowPDF(cal_pdf, k_min_fit=0.3)
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2] = True
    w.set_mask(mask)
    w.probe_row, w.probe_col = 3, 3
    small_2x2 = _calibrated_pdf(scan=(2, 2))
    w.set_data(small_2x2)
    assert w.mask_b64 == "" and w.mask_bytes == b""
    assert w.mask_pixel_count == 4
    assert (w.probe_row, w.probe_col) == (1, 1)
    assert not w.line_active
    assert w.k_min_fit == 0.3  # still inside the new k range: kept
    assert w.status_message == "" or w.status_message.startswith("Ready")
    ref = ShowPDF(small_2x2, k_min_fit=0.3)
    np.testing.assert_array_equal(_f32(w.gr_y_bytes), _f32(ref.gr_y_bytes))


def test_showpdf_set_data_keeps_the_k_ranges_that_fit_the_new_data(small_pdf, cal_pdf):
    # A fit range outside the new k range is reset to the full range, and a window end outside it
    # to 0 (follow the fit range).
    w = ShowPDF(small_pdf, k_min_fit=2.0, k_max_fit=14.0, k_max_window=13.0)
    w.set_data(_calibrated_pdf())
    assert (w.k_min_fit, w.k_max_fit, w.k_max_window) == (w.k_min_available, w.k_max_available, 0.0)
    # Ends typed at the edges of the same calibration are kept.
    w = ShowPDF(cal_pdf, k_min_fit=0.3, k_max_fit=1.55, k_max_window=1.55)
    w.set_data(_calibrated_pdf())
    assert (w.k_min_fit, w.k_max_fit, w.k_max_window) == (0.3, 1.55, 1.55)
    # The default full range follows the new data.
    w = ShowPDF(_calibrated_pdf())
    w.set_data(_calibrated_pdf(k_step=0.08))
    assert (w.k_min_fit, w.k_max_fit) == (w.k_min_available, w.k_max_available)
    assert w.k_max_available > 2.0


def test_showpdf_set_data_keeps_pixel_size_override(small_4d_array):
    w = ShowPDF(small_4d_array, pixel_size=2.0, **FROM_DATA_KWARGS)
    assert (w.nav_pixel_size, w.nav_unit) == (2.0, "Å")
    w.set_data(small_4d_array, **FROM_DATA_KWARGS)
    assert (w.nav_pixel_size, w.nav_unit) == (2.0, "Å")
    w.set_data(small_4d_array, pixel_size=3.5, **FROM_DATA_KWARGS)
    assert (w.nav_pixel_size, w.nav_unit) == (3.5, "Å")


# ---------------------------------------------------------------------------
# save_image
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, kwargs",
    [
        ("out.png", {}),
        ("out.pdf", {}),
        ("out.tiff", {}),
        ("noext", {"format": "png"}),  # format= for a path without an extension
        ("Ik.png", {"plot_mode": "Ik"}),
        ("Fk.png", {"plot_mode": "Fk"}),
        ("Gr.png", {"plot_mode": "Gr"}),
        ("gr.png", {"plot_mode": "gr"}),
    ],
)
def test_showpdf_save_image(small_pdf, tmp_path, name, kwargs):
    w = ShowPDF(small_pdf, density_value=0.05)  # g(r) needs a density
    assert w.save_image(tmp_path / name, **kwargs).stat().st_size > 0


@pytest.mark.parametrize(
    "name, kwargs, error",
    [
        ("out.bmp", {}, "Unsupported format"),
        ("out.png", {"plot_mode": "bogus"}, "Unknown plot_mode"),
        ("gr.png", {"plot_mode": "gr"}, "density"),  # g(r) without a density
    ],
)
def test_showpdf_save_image_errors(small_pdf, tmp_path, name, kwargs, error):
    with pytest.raises(ValueError, match=error):
        ShowPDF(small_pdf).save_image(tmp_path / name, **kwargs)


def test_showpdf_save_image_rejects_linescan(small_pdf, tmp_path):
    w = ShowPDF(small_pdf)
    w.set_line(1, 0, 1, 3, width=3)
    w.line_mode = "linescan"
    with pytest.raises(ValueError, match="save_results"):
        w.save_image(tmp_path / "x.png")


def test_showpdf_save_image_leaves_matplotlib_backend_alone(small_pdf, tmp_path, monkeypatch):
    import matplotlib

    def no_switch(*args, **kwargs):
        raise AssertionError("save_image must not switch the matplotlib backend")

    monkeypatch.setattr(matplotlib, "use", no_switch)
    w = ShowPDF(small_pdf)
    assert w.save_image(tmp_path / "x.png").stat().st_size > 0


@pytest.mark.parametrize("view", ["g(r) without a density", "line scan"])
def test_showpdf_saved_notebook_preview(cal_pdf, view):
    # A notebook saved with save_state=False keeps a PNG of the scan image and the shown curve, also
    # of an empty g(r) and of a line scan (which save_image refuses); a new view clears it.
    from PIL import Image

    w = ShowPDF(cal_pdf, k_min_fit=0.3, plot_mode="gr")
    if view == "line scan":
        w.density_value = 0.05
        w.set_line(0, 0, 3, 3, width=3)
        w.line_mode = "linescan"
    assert "nav_image_bytes" not in w.get_state()  # a notebook save, which stores the preview
    assert w._static_fallback_mime == "image/png"
    image = np.asarray(Image.open(io.BytesIO(base64.b64decode(w._static_fallback_jpeg))))
    assert image.shape[1] == 512 and image.std() > 5
    w.plot_mode = "Gr"
    assert w._static_fallback_jpeg == ""


# ---------------------------------------------------------------------------
# save_results: a zip of the curves, metadata.txt and README.txt
# ---------------------------------------------------------------------------


def test_showpdf_save_results_mask(cal_pdf, tmp_path):
    w = ShowPDF(cal_pdf, k_min_fit=0.3, damp_origin_oscillations=True, density_value=0.05)
    files = _read_zip(w.save_results(tmp_path / "res"))
    assert {"README.txt", "metadata.txt", "curves_k.csv", "curves_r.csv", "mask.npy"} <= set(files)
    meta = files["metadata.txt"].decode()
    assert "k_width" not in meta
    assert "r_min" not in meta
    assert "k_min_window:            fit range" in meta
    assert "density_value:           0.05\n" in meta
    assert "density_mode" not in meta
    curves_k = _read_csv(files, "curves_k.csv")
    np.testing.assert_allclose(curves_k["Fk"], _f32(w.fk_y_bytes), rtol=1e-6)  # damped F(k)


def test_showpdf_save_results_negative_window_end_follows_fit_range(cal_pdf, tmp_path):
    # A window end at or below 0 follows the fit range, in the compute and in metadata.txt.
    w = ShowPDF(cal_pdf, k_min_fit=0.3, k_max_fit=1.2, k_min_window=-0.5)
    assert w._curve_params()["k_min_window"] is None
    meta = _read_zip(w.save_results(tmp_path / "res"))["metadata.txt"].decode()
    assert "k_min_window:            fit range" in meta


@pytest.mark.parametrize("density", [0.0, -0.05])
def test_showpdf_save_results_metadata_without_density(small_pdf, tmp_path, density):
    # A zero or negative density is not set, in metadata.txt and README.txt alike; once it is set,
    # README.txt says F(k) has the low-r correction.
    w = ShowPDF(small_pdf, damp_origin_oscillations=True, density_value=density)
    files = _read_zip(w.save_results(tmp_path / "res.zip"))
    meta = files["metadata.txt"].decode()
    assert "density_value:           not set" in meta
    assert "damp_origin_oscillations: on (not applied: density not set)" in meta
    assert "k_min_fit:               0\n" in meta  # a number, not "not set": the k axis starts at 0
    assert "k_lowpass:               disabled" in meta
    assert "(no low-r correction)" in files["README.txt"].decode()
    w.density_value = 0.05
    readme = _read_zip(w.save_results(tmp_path / "damped.zip"))["README.txt"].decode()
    assert "(with the low-r correction applied)" in readme


def test_showpdf_unset_density_clears_saved_gr(small_pdf, tmp_path):
    w = ShowPDF(small_pdf, density_value=0.07)
    assert small_pdf.pdf is not None
    w.density_value = 0.0
    assert small_pdf.pdf is None
    files = _read_zip(w.save_results(tmp_path / "res.zip"))
    curves_r = _read_csv(files, "curves_r.csv")
    assert np.isnan(curves_r["gr"]).all()


def test_showpdf_save_results_linescan(cal_pdf, tmp_path):
    w = ShowPDF(cal_pdf, k_min_fit=0.3, density_value=0.05)
    w.set_line(0, 0, 3, 2, width=3)
    w.line_mode = "linescan"
    files = _read_zip(w.save_results(tmp_path / "scan.zip"))
    for name in ("linescan_Ik.csv", "linescan_Ik_bg.csv", "linescan_Fk.csv", "linescan_Gr.csv", "linescan_gr.csv"):
        assert name in files, name
    # The positions run evenly from A to B.
    positions = _read_csv(files, "linescan_position.csv")["position_px"]
    np.testing.assert_allclose(positions, np.linspace(0, np.hypot(3, 2), 4), atol=1e-5)
    assert "evenly spaced" in files["README.txt"].decode()


def test_showpdf_save_refuses_after_compute_error(cal_pdf, tmp_path):
    w = ShowPDF(cal_pdf, k_min_fit=0.3)
    w.k_max_fit = 0.2  # below k_min_fit: the background fit fails
    assert w.status_message.startswith("Error")
    with pytest.raises(Exception):
        w.save_results(tmp_path / "res.zip")


# ---------------------------------------------------------------------------
# Saved state: state_dict, save and state=
# ---------------------------------------------------------------------------


def _select_mask(w):
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :2] = True
    w.set_mask(mask)


def _select_perpendicular_line(w):
    w.set_line(0, 1, 3, 2, width=3)
    w.line_perpendicular = True


@pytest.mark.parametrize("select", [_select_mask, _select_perpendicular_line], ids=["mask", "line"])
@pytest.mark.parametrize("via", ["state_dict", "file"])
def test_showpdf_state_roundtrip(small_pdf, tmp_path, select, via):
    w = ShowPDF(small_pdf, title="Test PDF", k_min_fit=1.5, r_max=15.0, plot_mode="Ik")
    select(w)
    state = w.state_dict()
    if via == "file":
        path = str(tmp_path / "state.json")
        w.save(path)
        envelope = json.loads(pathlib.Path(path).read_text())
        assert envelope["widget_name"] == "ShowPDF"
        assert {"metadata_version", "widget_version", "state"} <= set(envelope)
        state = path
    restored = ShowPDF(small_pdf, state=state)
    for key, value in w.state_dict().items():
        assert getattr(restored, key) == value, key
    # Restoring rebuilds both the analysis mask and the frontend's repaint bytes.
    np.testing.assert_array_equal(restored.mask, w.mask)
    assert restored.mask_pixel_count == w.mask_pixel_count
    assert restored.mask_bytes == w.mask_bytes


def test_showpdf_state_drops_mismatched_mask(small_pdf):
    w1 = ShowPDF(small_pdf)
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :2] = True
    w1.set_mask(mask)
    state = w1.state_dict()

    # Restore the 4x4 mask onto a 2x2 scan.
    scan_y, scan_x = 2, 2
    ring = np.exp(-((_detector_radius() - 5) ** 2) / 4).astype(np.float32)
    pdf2 = _uncalibrated_pdf(np.broadcast_to(ring, (scan_y, scan_x, DET, DET)).copy())
    w2 = ShowPDF(pdf2, state=state)
    # The mismatched mask is dropped, not applied: the full scan is included.
    assert w2.mask_b64 == ""
    assert w2.mask_pixel_count == scan_y * scan_x


def test_showpdf_state_rejects_other_widget_file(small_pdf, tmp_path):
    path = tmp_path / "other.json"
    path.write_text(json.dumps({
        "metadata_version": "1.0",
        "widget_name": "Show2D",
        "widget_version": "0",
        "state": {"title": "x"},
    }))
    with pytest.raises(ValueError):
        ShowPDF(small_pdf, state=str(path))


@pytest.mark.parametrize(
    "bad",
    [{"linescan_max_bins": 0}, {"line_width": 2.5}, {"k_min_fit": 10**400}, {"line_width": 10**400 + 1}],
    ids=["linescan_max_bins", "line_width", "k_min_fit too large", "line_width too large"],
)
def test_showpdf_rejected_state_changes_nothing(cal_pdf, bad):
    # The valid keys listed before the rejected one are not kept either, and the widget still
    # recomputes afterwards.
    w = ShowPDF(cal_pdf, k_min_fit=0.3)
    before = _synced(w)
    state = {
        "analysis_mode": "line", "line_col1": 3.0, "line_active": True, "line_width": 5,
        "density_value": 0.05, "mask_b64": base64.b64encode(bytes(16)).decode(), **bad,
    }
    with pytest.raises(traitlets.TraitError):
        w.load_state_dict(state)
    assert _synced(w) == before
    w.r_max = 7.0
    assert w.n_points_gr == int(round(7.0 / w.r_step))


# ---------------------------------------------------------------------------
# States saved by earlier versions
# ---------------------------------------------------------------------------


def _old_state_file(path, widget, **saved):
    """Save ``widget`` to ``path`` with the ``saved`` keys written into its state."""
    widget.save(str(path))
    envelope = json.loads(path.read_text())
    envelope["state"].update(saved)
    path.write_text(json.dumps(envelope))
    return path


@pytest.mark.parametrize("saved", [float("nan"), float("inf"), float("-inf")], ids=repr)
def test_showpdf_old_state_nonfinite_density(cal_pdf, tmp_path, saved):
    # Earlier versions saved an unset density as a NaN literal. A saved non-finite float density
    # loads as 0 (not set) along with the other keys; assigning one is still rejected.
    w = ShowPDF(cal_pdf, k_min_fit=0.3, density_value=0.05)
    before = _synced(w)
    # A saved string is rejected as any string is, and a mapped density is undone with the rest
    # of a rejected load.
    for state, rejected in (
        ({"density_value": str(saved)}, "'density_value' trait"),
        ({"density_value": saved, "linescan_max_bins": 0}, "^linescan_max_bins"),
    ):
        with pytest.raises(traitlets.TraitError, match=rejected):
            w.load_state_dict(state)
        assert _synced(w) == before
    w.load_state_dict({"density_value": saved, "r_max": 12.0})
    assert (w.density_value, w.r_max) == (0.0, 12.0)

    saved_widget = ShowPDF(cal_pdf, k_min_fit=0.3, r_max=12.0, plot_mode="Fk")
    path = _old_state_file(tmp_path / "old_state.json", saved_widget, density_value=saved)
    assert re.search(r'"density_value": -?(NaN|Infinity)', path.read_text())
    restored = ShowPDF(cal_pdf, state=str(path))
    assert (restored.density_value, restored.r_max, restored.plot_mode) == (0.0, 12.0, "Fk")
    with pytest.raises(traitlets.TraitError, match="^density_value must be a finite number"):
        restored.density_value = saved


@pytest.mark.parametrize("old_mode, expected", [("estimated", 0.0), ("manual", 0.06)])
def test_showpdf_old_state_density_mode(small_pdf, old_mode, expected):
    # An "estimated" density_value came from the removed estimator, not from the user, so it is
    # not adopted as the fixed density.
    w = ShowPDF(small_pdf)
    w.load_state_dict({"density_mode": old_mode, "density_value": 0.06})
    assert w.density_value == expected
    assert not hasattr(w, "density_mode")


@pytest.mark.parametrize("key, value", [("k_width", 1.0), ("r_min", 0.5)])
def test_showpdf_old_state_removed_keys_are_ignored(small_pdf, tmp_path, key, value):
    # The window's taper width is fixed and the r grid always starts at 0, so a saved k_width or
    # r_min is ignored, from a dict or from a file.
    ref = ShowPDF(small_pdf, r_max=12.0)
    assert _f32(ref.gr_x_bytes)[0] == 0.0
    w = ShowPDF(small_pdf)
    w.load_state_dict({key: value, "r_max": 12.0})
    path = _old_state_file(tmp_path / "old_state.json", ref, **{key: value})
    for restored in (w, ShowPDF(small_pdf, state={key: value, "r_max": 12.0}), ShowPDF(small_pdf, state=str(path))):
        assert restored.r_max == 12.0
        for curve in ("fk_y_bytes", "gr_x_bytes", "gr_y_bytes"):
            assert getattr(restored, curve) == getattr(ref, curve), curve
        assert not hasattr(restored, key) and key not in restored.state_dict()


@pytest.mark.parametrize("saved, expected", [(4, 5), (2, 3), (0, 1), (-3, 1), (1, 1), (5, 5)])
def test_showpdf_old_state_line_width(small_pdf, saved, expected):
    # Widths did not have to be odd: an even width w selected w + 1 rows of a horizontal line and
    # a width below 1 acted as 1, so they load as the odd width that selects the same rows. An
    # even width warns, since a diagonal band of width w + 1 can include more positions.
    line = (10, 4, 10, 15)
    np.testing.assert_array_equal(
        _LineStub(line, saved, False)._line_band_mask(), _LineStub(line, expected, False)._line_band_mask()
    )
    w = ShowPDF(small_pdf)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        w.load_state_dict({"line_width": saved})
        restored = ShowPDF(small_pdf, state={"line_width": saved})
    assert w.line_width == restored.line_width == expected
    notices = [c for c in caught if "saved line_width" in str(c.message)]
    if saved >= 2 and saved % 2 == 0:
        assert len(notices) == 2  # one per load
        assert notices[0].filename == __file__  # load_state_dict's points at the caller
        for notice in notices:
            assert re.search(
                rf"saved line_width {saved} is even; it loads as line_width {saved + 1}.*"
                r"diagonal band may include more positions than before",
                str(notice.message),
            )
    else:
        assert notices == []


def test_showpdf_rejected_state_with_even_line_width_does_not_warn(small_pdf):
    # A load that another key rejects keeps the old width, so it must not say the width loads
    # as w + 1.
    w = ShowPDF(small_pdf)
    state = {"line_width": 4, "linescan_max_bins": 0}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(traitlets.TraitError, match="^linescan_max_bins must be at least 1"):
            w.load_state_dict(state)
        with pytest.raises(traitlets.TraitError, match="^linescan_max_bins must be at least 1"):
            ShowPDF(small_pdf, state=state)
    assert not [c for c in caught if "saved line_width" in str(c.message)]
    assert w.line_width == 3
    with pytest.warns(UserWarning, match="saved line_width 4 is even"):
        w.load_state_dict({**state, "linescan_max_bins": 7})
    assert (w.line_width, w.linescan_max_bins) == (5, 7)


# ---------------------------------------------------------------------------
# summary and repr
# ---------------------------------------------------------------------------


def test_showpdf_summary(small_pdf, capsys):
    ShowPDF(small_pdf, title="My PDF", damp_origin_oscillations=True).summary()
    out = capsys.readouterr().out
    assert "My PDF" in out
    assert "4 × 4" in out  # scan shape
    assert "Mask:" in out
    assert "Window:" in out and "k_width" not in out
    assert "Output:   r=[0.00, 20.00]" in out
    assert "Density:  not set" in out
    assert "skipped until the density is set" in out
    ShowPDF(small_pdf, damp_origin_oscillations=True, density_value=0.05).summary()
    out = capsys.readouterr().out
    assert "Density:  0.05 atoms/Å³" in out
    assert "skipped" not in out


def test_showpdf_summary_and_repr_describe_the_selection(small_pdf, capsys):
    w = ShowPDF(small_pdf)
    assert "ShowPDF" in repr(w) and "scan=(4, 4)" in repr(w)
    # A mask painted in the frontend, which writes mask_b64 and then bumps mask_version.
    painted = np.ones((4, 4), dtype=np.uint8)
    painted[:2] = 0
    w.mask_b64 = base64.b64encode(painted.tobytes()).decode("ascii")
    w.mask_version += 1
    assert w.mask_pixel_count == 8
    w.summary()
    assert "Mask:     8 px (50.0%)" in capsys.readouterr().out
    assert "mask=8px" in repr(w)
    # In probe mode both describe the probe, not the painted mask.
    w.analysis_mode = "probe"
    w.probe_size = 2
    w.summary()
    out = capsys.readouterr().out
    assert "Probe:    (2, 2), 3 × 3, 9 px" in out
    assert "Mask:" not in out
    assert "mask=" not in repr(w)


# ---------------------------------------------------------------------------
# Text shared with the frontend and the docs
# ---------------------------------------------------------------------------


def test_showpdf_status_hints_match_frontend():
    source = FRONTEND_SOURCE.read_text(encoding="utf-8")
    assert f'"{_DENSITY_HINT_DAMPING}"' in source
    assert f'"{_DENSITY_HINT_GR}"' in source
    assert re.search(rf'\bWINDOW_HINT = "{re.escape(_WINDOW_HINT)}"', source)
    # The docs quote the window hint too, wrapped across lines.
    docs = DOCS_PAGE.read_text(encoding="utf-8")
    assert f'"{_WINDOW_HINT}"' in " ".join(docs.split())


def test_showpdf_export_labels_match_frontend():
    source = FRONTEND_SOURCE.read_text(encoding="utf-8")
    assert f'"{_export_label(False)}"' in source
    assert f'"{_export_label(True)}"' in source


def test_showpdf_docs_say_a_full_export_without_curves_is_locked():
    docs = DOCS_PAGE.read_text(encoding="utf-8")
    for source in (docs, ShowPDF.export_html.__doc__):
        text = " ".join(source.split()).replace("``", "`")
        assert (
            '`"full"` export made when no curves were computed for the shown selection '
            "(its compute failed) keeps its parameter controls locked too and shows the baked error."
        ) in text
