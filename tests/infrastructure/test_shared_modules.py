"""The shared widget modules: one copy each, matching the browser rules."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from ipywidgets.widgets.widget import _instances
from matplotlib import colormaps

from quantem.widget import Plot2D, Show1D, Show2D, Show3D, Show4DSTEM
from quantem.widget.colormap import VALID_CMAPS, Colormap, cmap_to_name, colorize, is_cmap_sequence
from quantem.widget.export import export_slug
from quantem.widget.pages import resolve_page_labels
from quantem.widget.panels import normalize_panel_refs, resolve_panel_ref
from quantem.widget.render.figure import format_scale_label, round_to_nice, unit_symbol

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
        (57, "pixels", "50 pixels"),  # unknown units pass through unchanged
        (2.5, "ps", "2 ps"),
    ],
)
def test_scale_label_matches_js_figure(value, unit, label):
    assert format_scale_label(value, unit) == label


def test_round_to_nice_thresholds():
    assert [round_to_nice(v) for v in (1.4, 1.5, 3.4, 3.5, 7.4, 7.5, 0)] == [1, 2, 2, 5, 5, 10, 1.0]
    assert unit_symbol("angstrom") == "Å" and unit_symbol("micron") == "µm" and unit_symbol("px") == "px"


def test_colormap_names_and_lut():
    assert {"inferno", "viridis", "magma", "plasma", "gray", "RdBu_r", "twilight_shifted"} <= VALID_CMAPS
    assert cmap_to_name(Colormap.MAGMA) == "magma" and cmap_to_name("hot") == "hot"
    assert is_cmap_sequence(["gray", "hot"]) and not is_cmap_sequence("gray")
    normalized = np.linspace(0, 1, 7, dtype=np.float32)[None, :]
    expected = (colormaps.get_cmap("viridis")(normalized)[..., :3] * 255).astype(np.uint8)
    np.testing.assert_array_equal(colorize(normalized, "viridis"), expected)
    assert colorize(normalized, "viridis").shape == (1, 7, 3)


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
    "def _round_to_nice",
    "def _format_scale_label(",
    "class Colormap(",
    "def _on_export_request_change(",
    "def _html_export_bytes(",
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
    offenders = []
    for path in sorted(WIDGETS.rglob("*.py")):
        relative = path.relative_to(WIDGETS).as_posix()
        if relative in shared or relative.startswith(("render/", "utils/")):
            continue
        text = path.read_text()
        for needle in SHARED_DEFINITIONS:
            if needle in text:
                offenders.append(f"{relative}: {needle}")
    assert offenders == []


def test_export_traits_come_from_the_mixin():
    for name in ("show2d", "show3d", "show4dstem", "show1d", "showdiffraction", "show3dslices"):
        for path in widget_sources(name):
            text = path.read_text()
            assert not re.search(r"^    export_(request|status|enabled|payload|payload_id|filename) = traitlets", text, re.M), path.name


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
