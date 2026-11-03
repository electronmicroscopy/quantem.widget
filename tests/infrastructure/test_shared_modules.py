"""The shared widget modules: one copy each, matching the browser rules."""

import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from ipywidgets.widgets.widget import _instances
from matplotlib import colormaps

from quantem.widget import Plot2D, Show1D, Show2D, Show3D, Show3DSlices, Show4DSTEM, ShowPtycho
from quantem.widget.adapters.core import make_dataset
from quantem.widget.colormap import VALID_CMAPS, Colormap, cmap_to_name, colorize, contrast_uint8, is_cmap_sequence
from quantem.widget.export import export_slug
from quantem.widget.pages import resolve_page_labels
from quantem.widget.panels import normalize_panel_refs, resolve_panel_ref
from quantem.widget.render.figure import format_scale_label, round_to_nice
from quantem.widget.units import UNIT_TABLE, axis_step, wrap_degrees

WIDGETS = Path(__file__).resolve().parents[2] / "src" / "quantem" / "widget"


@pytest.mark.parametrize(
    "value, unit, label",
    [
        (0.5, "nm", "5 Å"),  # sub-1 nm re-ladders to the unit that reads as an integer
        (13.8, "A", "1 nm"),  # 10 A reads as 1 nm
        (0.005, "nm", "5 pm"),
        (23.0, "um", "20 µm"),
        (20, "mrad", "20 mrad"),  # non-length units keep their unit
        (0.25, "mrad", "0.20 mrad"),
        (57, "pixels", "50 px"),  # an uncalibrated axis reads px
        (2.5, "ps", "2 ps"),
        # the cases of js/figure.test.ts: every length unit of the table rescales, meters included
        (20, "Å", "2 nm"),
        (5, "angstrom", "5 Å"),
        (12, "angstrom", "1 nm"),  # the unit strings a .qem file writes
        (0.2, "1/angstrom", "0.20 Å⁻¹"),
        (1500, "um", "2 mm"),
        (0.2, "μm", "200 nm"),
        (1e-9, "m", "1 nm"),
        (0.002, "mm", "2 µm"),
        (0.05, "µm", "50 nm"),
        (1e-7, "mm", "1 Å"),  # 1e-7 * 1e7 Å is exactly 1; a table in nm gave 0.0999... nm and printed "100 pm"
        (1e-4, "um", "1 Å"),  # the browser builds the power of ten from text, as exact as Python's 10 ** -4
        (1.2e-5, "um", "10 pm"),
    ],
)
def test_scale_label_matches_js_figure(value, unit, label):
    assert format_scale_label(value, unit) == label


def test_uncalibrated_axes_carry_the_unit_px():
    # the browser labels an uncalibrated axis px; the traits, and so a saved state, say the same
    image = np.zeros((8, 8), dtype=np.float32)
    assert Show2D(image).pixel_unit == "px"
    viewer = Show4DSTEM(np.ones((2, 2, 4, 4), dtype=np.uint16), precompute_virtual_images=False, verbose=False)
    # a 4D scan without a calibration has empty sampling lists: unknown is absent, not 1.0 px
    assert (viewer.scan_sampling_A, viewer.detector_sampling_mrad, viewer.detector_sampling_inv_A) == ([], [], [])
    viewer.close()
    profile = Show1D.from_image(image, line=((1, 1), (1, 6)))
    assert (profile.x_unit, profile.pixel_unit) == ("px", "px")


def test_round_to_nice_thresholds():
    assert [round_to_nice(v) for v in (1.4, 1.5, 3.4, 3.5, 7.4, 7.5, 0)] == [1, 2, 2, 5, 5, 10, 1.0]


def test_the_frontend_unit_table_holds_the_python_one():
    # one table per language: js/units.ts lists the same unit strings with the same kind, size and symbol
    source = (WIDGETS.parents[2] / "js" / "units.ts").read_text(encoding="utf-8")
    table = source[source.index("const UNIT_TABLE"):source.index("};", source.index("const UNIT_TABLE"))]
    entries = re.findall(r'(?:"([^"]+)"|\b(\w+)): \{ kind: "(\w+)", size: ([^,]+), symbol: "([^"]+)" \}', table)
    frontend = {quoted or bare: (kind, size, symbol) for quoted, bare, kind, size, symbol in entries}
    assert set(frontend) == set(UNIT_TABLE)
    for unit, (kind, size, symbol) in UNIT_TABLE.items():
        assert frontend[unit][0] == kind and frontend[unit][2] == symbol, unit
        if unit != "deg":  # written as (1e3 * Math.PI) / 180 there
            assert float(frontend[unit][1]) == size, unit


@pytest.mark.parametrize("sampling", [0.5, 1, np.float32(0.5), np.float64(0.5), np.array(0.5), (0.25, 0.5), [0.25, 0.5], np.array([0.25, 0.5])],
                         ids=["float", "int", "float32", "float64", "0-d array", "tuple", "list", "1-d array"])
def test_every_sampling_keyword_reads_a_numpy_scalar_as_one_step(sampling):
    # isinstance(sampling, (int, float)) sent np.float32(0.5) and a 0-d array down the sequence branch, which indexed a scalar
    row, col = (float(step) for step in np.broadcast_to(np.asarray(sampling, dtype=float), (2,)))
    image, stack = np.zeros((8, 8), np.float32), np.zeros((2, 8, 8), np.float32)
    for shown in (Show2D(image, sampling=sampling, units="nm", verbose=False), Show3D(stack, sampling=sampling, units="nm", verbose=False),
                  Show1D([1.0, 2.0], sampling=sampling, units="nm"), Show2D(make_dataset(image, sampling=(row / 10, col / 10), units=("nm", "nm")), sampling=sampling, verbose=False)):
        assert shown.pixel_size == col  # the column step, of the pair or of the one scalar
    assert ShowPtycho(image, sampling=sampling).object_sampling_A == [row, col]
    viewer = Show4DSTEM(np.ones((4, 4, 8, 8), np.uint16), scan_sampling_A=sampling, precompute_virtual_images=False)
    assert viewer.scan_sampling_A == [row, col]
    viewer.close()
    # three axes: the scalar is every axis, the pair grows a depth step in front, in the type it came in
    voxel = sampling if np.ndim(sampling) == 0 else type(sampling)([0.3, *sampling]) if not isinstance(sampling, np.ndarray) else np.array([0.3, *sampling])
    assert Show3DSlices(stack, sampling=voxel).voxel_size_A == [float(step) for step in np.broadcast_to(np.asarray(voxel, dtype=float), (3,))]


def test_a_unit_stated_for_an_input_without_a_step_raises():
    # units= with no sampling= vanished for an array (no bar, no message) and made the 1.0 of an uncalibrated dataset 1 nm
    image, stack = np.zeros((8, 8), np.float32), np.zeros((2, 8, 8), np.float32)
    bare = make_dataset(image, sampling=(1.0, 1.0), units=("pixels", "pixels"))
    for build in (lambda: Show1D.from_image(image, line=((2, 1), (2, 6)), units="nm"), lambda: Show2D(bare, units="nm", verbose=False),
                  lambda: Show2D(image, units="nm", verbose=False), lambda: Show1D([1.0, 2.0], units="nm"), lambda: Show3D(stack, units="nm", verbose=False),
                  lambda: Show3DSlices(stack, units="nm"), lambda: ShowPtycho(image, units="nm"), lambda: ShowPtycho(bare, units="nm")):
        with pytest.raises(ValueError, match="units='nm' names a unit and the input has no step to give it: a unit needs a step"):
            build()
    # an uncalibrated unit states no step, and a calibrated input has one
    for shown in (Show2D(image, units="px", verbose=False), Show2D(bare, units="px", verbose=False), Show3D(stack, units="px", verbose=False)):
        assert (shown.pixel_size, shown.pixel_unit) == (0.0, "px")
    calibrated = make_dataset(image, sampling=(0.2, 0.2), units=("nm", "nm"))
    assert Show2D(calibrated, units="nm", verbose=False).pixel_size == 0.2 and ShowPtycho(calibrated, units="nm").object_sampling_A == pytest.approx([2.0, 2.0])
    # a viewer without a step still builds its preview and its export clone (they pass the unit only with a step)
    plain = Show3D(stack, verbose=False)
    assert (plain.pixel_size, plain.pixel_unit) == (0.0, "A") and plain._static_fallback_png_b64()
    # no keyword is no stated unit: an nm dataset with a zero step opens uncalibrated, and a depth step is a step
    assert Show3D(make_dataset(stack, sampling=(1.0, 0.0, 0.0), units=("frame", "nm", "nm")), verbose=False).pixel_size == 0.0
    depth_only = make_dataset(stack, sampling=(0.5, 1.0, 1.0), units=("nm", "pixels", "pixels"))
    assert Show3DSlices(depth_only, units="nm").voxel_size_A == [5.0, 0.0, 0.0]
    # a sequence of one unit is the unit of both axes: the column step, not the row step, sets the bar
    anisotropic = make_dataset(image, sampling=(0.5, 0.25), units=("nm", "nm"))
    assert [Show2D(anisotropic, units=units, verbose=False).pixel_size for units in (["nm"], ("A",), "A")] == [0.25, 2.5, 2.5]


def test_a_half_calibrated_pair_takes_the_keyword_on_its_open_axis_only():
    # a pair with one calibrated axis counted as recording nothing, so a keyword replaced the recorded step unchecked
    half = make_dataset(np.zeros((8, 8), np.float32), sampling=(1.0, 0.05), units=("pixels", "nm"))  # the column step is 0.5 angstrom
    assert ShowPtycho(half, sampling=(0.4, 0.5)).object_sampling_A == pytest.approx([0.4, 0.5])
    assert ShowPtycho(half, sampling=0.5).object_sampling_A == pytest.approx([0.5, 0.5])
    assert ShowPtycho(half).object_sampling_A == []  # one open axis: no geometry to draw
    with pytest.raises(ValueError, match="contradicts the value the input records"):
        ShowPtycho(half, sampling=(0.4, 0.3))
    scan = make_dataset(np.ones((4, 4, 8, 8), np.uint16), sampling=(1.0, 0.06, 1.5, 1.5), units=("pixels", "nm", "mrad", "mrad"))
    viewer, bare = Show4DSTEM(scan, scan_sampling_A=(0.4, 0.6), precompute_virtual_images=False), Show4DSTEM(scan, precompute_virtual_images=False)
    assert viewer.scan_sampling_A == pytest.approx([0.4, 0.6]) and viewer.detector_sampling_mrad == [1.5, 1.5]
    assert bare.scan_sampling_A == [] and bare.detector_sampling_mrad == [1.5, 1.5]
    viewer.close()
    bare.close()
    with pytest.raises(ValueError, match="scan_sampling_A=.* contradicts the value the input records"):
        Show4DSTEM(scan, scan_sampling_A=(0.4, 0.7), precompute_virtual_images=False)


def test_an_angle_written_to_a_file_is_in_0_to_360():
    # value % 360.0 answers 360.0 for a tiny negative angle, outside the range the files state
    assert [wrap_degrees(value) for value in (-1e-15, -10.0, 360.0, 725.5, 21.3, 0.1)] == [0.0, 350.0, 0.0, 5.5, 21.3, 0.1]
    # js/units.ts wrapDegrees is the same computation: the remainder, shifted when negative, 360 itself taken as 0
    source = (WIDGETS.parents[2] / "js" / "units.ts").read_text(encoding="utf-8")
    assert "remainder < 0 ? remainder + 360 : remainder" in source and "wrapped === 360 ? 0 : wrapped" in source


def test_one_unit_table_reads_the_format_strings_and_refuses_unknown_text():
    # the three strings a .qem file writes, in the reference unit of their kind
    assert axis_step(0.4, "angstrom") == ("length", 0.4)
    assert axis_step(0.02, "1/angstrom") == ("reciprocal_length", 0.02)
    assert axis_step(0.5, "mrad") == ("angle", 0.5)
    assert axis_step(0.2, "nm") == ("length", 2.0)
    # an uncalibrated axis is absent whichever dataset type names it: None (quantem.gpu) or 1.0 "pixels" (quantem core)
    assert axis_step(None, None) is None and axis_step(1.0, "pixels") is None
    with pytest.raises(ValueError, match="Unknown unit 'furlong'"):
        axis_step(1.0, "furlong")
    with pytest.raises(ValueError, match="Unknown unit 'furlong'"):
        format_scale_label(57, "furlong")


def test_colormap_names_and_lut():
    assert {"inferno", "viridis", "magma", "plasma", "gray", "RdBu_r", "twilight_shifted"} <= VALID_CMAPS
    assert cmap_to_name(Colormap.MAGMA) == "magma" and cmap_to_name("hot") == "hot"
    assert is_cmap_sequence(["gray", "hot"]) and not is_cmap_sequence("gray")
    normalized = np.linspace(0, 1, 7, dtype=np.float32)[None, :]
    expected = (colormaps.get_cmap("viridis")(normalized)[..., :3] * 255).astype(np.uint8)
    np.testing.assert_array_equal(colorize(normalized, "viridis"), expected)
    assert colorize(normalized, "viridis").shape == (1, 7, 3)


def test_colorize_reads_the_level_position_of_256_and_510_entry_tables():
    # twilight and twilight_shifted are 510-entry matplotlib tables: level k reads entry 2k (509 at
    # the top), the same position along the colormap as entry k of a 256-entry table.
    levels = np.arange(256)
    for name, entries in (("inferno", levels), ("twilight", np.minimum(2 * levels, 509)), ("twilight_shifted", np.minimum(2 * levels, 509))):
        colormap = colormaps.get_cmap(name)
        table = (colormap(np.arange(colormap.N))[:, :3] * 255).astype(np.uint8)
        np.testing.assert_array_equal(colorize(levels / 255.0, name), table[entries])


def test_contrast_uint8_paints_nan_inf_and_an_empty_window_like_the_browser():
    # displayNormalize sends NaN and -inf to 0, +inf to 1 and every value of an empty window to 0.5;
    # applyColormap then reads entry trunc(n * 255). No platform-dependent NaN cast, no warning.
    values = np.array([np.nan, -np.inf, np.inf, -2.0, 0.25, 9.0], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        np.testing.assert_array_equal(contrast_uint8(values, 0.0, 1.0), [0, 0, 255, 0, 63, 255])
        np.testing.assert_array_equal(contrast_uint8(values, 1.0, 1.0), [0, 0, 255, 127, 127, 127])
        np.testing.assert_array_equal(contrast_uint8(values, float("nan"), 1.0), [0, 0, 255, 127, 127, 127])


def test_panel_refs_and_order():
    titles = ["raw", "filtered", "raw"].__getitem__
    assert resolve_panel_ref("filtered", 3, titles) == 1
    with pytest.raises(ValueError, match="not unique"):
        resolve_panel_ref("raw", 3, titles)
    with pytest.raises(ValueError, match="out of range"):
        resolve_panel_ref(3, 3, titles)
    assert normalize_panel_refs([2, "filtered", 2], 3, titles) == [2, 1]


def test_page_labels_and_export_slug():
    assert resolve_page_labels(None, ["Page 1", "Page 2"], 2) == ["Page 1", "Page 2"]
    assert resolve_page_labels(["a", None], ["x", "y"], 2) == ["a", ""]
    with pytest.raises(ValueError, match="page_labels length"):
        resolve_page_labels(["a"], ["x", "y"], 2)
    assert export_slug("Gold HAADF  frame", "show2d") == "gold_haadf_frame"
    assert export_slug("   ", "show2d") == "show2d"


SHARED_DEFINITIONS = (
    "def _with_initial_live_mount_state(",
    "def round_to_nice(",
    "def format_scale_label(",
    "class Colormap(",
    "def _on_export_request_change(",
    "def _html_export_choice(",
    "def _write_export_and_report(",
    "def widget_runtime_html(",
    "def read_state(",
    "def device_array(",
    "def _resolve_panel_ref(",
    "def _normalize_panel_refs(",
    "def _store_static_fallback_preview(",
    "def star_page(",
    "def _normalize_hidden_page_slots(",
    "class StaticFallbackMixin",
    "def int32_block_sum(",
    "def int32_block_max(",
    "def count_bound(",
)


def widget_sources(name: str) -> list[Path]:
    """A widget is one module (``show1d.py``) or one package (``show2d/``)."""
    package = WIDGETS / name
    return sorted(package.glob("*.py")) if package.is_dir() else [WIDGETS / f"{name}.py"]


def test_widget_modules_do_not_redefine_shared_helpers():
    """Every helper that was copy-pasted across widgets now has one home; this keeps it that way."""
    shared = {"colormap.py", "counts.py", "export.py", "fallback.py", "pages.py", "panels.py", "state.py"}
    offenders, homes = [], set()
    for path in sorted(WIDGETS.rglob("*.py")):
        relative = path.relative_to(WIDGETS).as_posix()
        text = path.read_text()
        found = {needle for needle in SHARED_DEFINITIONS if needle in text}
        if relative in shared or relative.startswith(("render/", "utils/")):
            homes |= found
        else:
            offenders += [f"{relative}: {needle}" for needle in sorted(found)]
    assert offenders == []
    # a needle with no home is stale (renamed helper) and would let a copy slip back in
    assert sorted(set(SHARED_DEFINITIONS) - homes) == []


def test_export_traits_come_from_the_mixin():
    for name in ("show2d", "show3d", "show4dstem", "show1d", "showdiffraction", "show3dslices", "diffraction_sim"):
        for path in widget_sources(name):
            text = path.read_text()
            assert not re.search(r"^    export_(request|status|enabled|payload|payload_id|filename) = traitlets", text, re.M), path.name



def test_every_widget_bundle_is_an_anywidget_default_export():
    # anywidget logs a deprecation warning for every view of a bundle that exports render by name
    entries = sorted((WIDGETS.parents[2] / "js").glob("*/index.tsx"))
    named = [entry.parent.name for entry in entries if re.search(r"^export (const|function) render\b", entry.read_text(), re.M)]
    missing = [entry.parent.name for entry in entries if not re.search(r"^export default \{", entry.read_text(), re.M)]
    assert len(entries) == 12 and named == [] and missing == [], (named, missing)

RNG = np.random.default_rng(0)
PREVIEW_WIDGETS = {
    "Show1D": lambda: Show1D(RNG.random(64)),
    "Show2D": lambda: Show2D(RNG.random((64, 64), dtype=np.float32)),
    "Show3D": lambda: Show3D(RNG.random((4, 64, 64), dtype=np.float32)),
    "Show4DSTEM": lambda: Show4DSTEM(RNG.random((8, 8, 12, 12), dtype=np.float32), device="cpu"),
    "Plot2D": lambda: Plot2D(RNG.random((6, 8)), x=np.arange(8.0), y=np.arange(6.0)),
}


@pytest.mark.parametrize("name", PREVIEW_WIDGETS)
def test_displaying_a_widget_leaves_no_widget_model_or_figure_open(name):
    # Every display renders the saved-notebook preview (fallback.py). A preview that leaves a widget
    # model open keeps a comm and a frame copy in the kernel and a model in the browser per display;
    # an open pyplot figure is drawn again under the cell by the inline backend.
    widget = PREVIEW_WIDGETS[name]()
    models, figures = set(_instances), plt.get_fignums()
    for _ in range(3):
        bundle = widget._repr_mimebundle_()
        assert any(mime.startswith("image/") for mime in (bundle[0] if isinstance(bundle, tuple) else bundle))
    assert set(_instances) == models
    assert plt.get_fignums() == figures
