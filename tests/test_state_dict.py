"""state_dict roundtrip tests for Show2D and Show4DSTEM.

For each widget:
1. Construct with default data.
2. Mutate every trait in state_dict() to a non-default value.
3. Get state_dict.
4. Construct a fresh widget and load_state_dict.
5. Assert every trait on the restored widget equals what we set.

Catches silent regressions when traits are added, renamed, or dropped without
updating the state_dict roundtrip path.
"""
import json

import numpy as np
import pytest
from quantem.widget import Show2D, Show3D, Show4DSTEM


def _flip_value(default):
    """Return a value distinct from `default` for the same type."""
    if isinstance(default, bool):
        return not default
    if isinstance(default, int):
        return int(default) + 7
    if isinstance(default, float):
        return float(default) + 0.123
    if isinstance(default, str):
        return default + "_x" if default else "x"
    if isinstance(default, list):
        return [_flip_value(default[0])] if default else [0]
    return default


def _mutate_state(state: dict) -> dict:
    """Build a new state dict with every key changed to a non-default value."""
    out = {}
    for k, v in state.items():
        # Skip values our flipper can't safely tweak (None defaults, nested dicts/lists-of-dicts, bytes).
        if v is None or isinstance(v, (dict, bytes)):
            out[k] = v
            continue
        # Lists hold structured items (dicts, tuples) for ROI / profile / labels;
        # mutating them generically is fragile. The roundtrip-defaults test already
        # covers list trait persistence — here we only mutate scalars.
        if isinstance(v, list):
            out[k] = v
            continue
        out[k] = _flip_value(v)
    return out


# ---------------------------------------------------------------------------
# Show4DSTEM
# ---------------------------------------------------------------------------

@pytest.fixture
def show4dstem_widget():
    data = np.random.default_rng(0).poisson(5, (8, 8, 16, 16)).astype(np.uint16)
    data[:, :, 6:10, 6:10] += 500  # synthetic BF disk
    return Show4DSTEM(data, verbose=False)


def test_show4dstem_state_dict_keys(show4dstem_widget):
    """state_dict returns a non-empty dict of public traits."""
    s = show4dstem_widget.state_dict()
    assert isinstance(s, dict)
    assert len(s) > 10
    # Required keys for the widget's user-facing display state
    for required in (
        "title",
        "dp_colormap",
        "vi_colormap",
        "roi_mode",
        "vi_roi_reduce",
        "show_title",
        "show_controls",
        "controls_collapsed",
        "show_stats",
        "show_scale_bar",
        "compare_group_mode",
    ):
        assert required in s, f"state_dict missing key {required!r}"


def test_show4dstem_state_dict_roundtrip_defaults(show4dstem_widget):
    """save → load on default widget preserves state."""
    original = show4dstem_widget.state_dict()
    data = np.random.default_rng(0).poisson(5, (8, 8, 16, 16)).astype(np.uint16)
    data[:, :, 6:10, 6:10] += 500
    fresh = Show4DSTEM(data, state=original, verbose=False)
    restored = fresh.state_dict()
    for k in original:
        assert restored[k] == original[k], f"{k}: {original[k]!r} -> {restored[k]!r}"


def test_show4dstem_state_dict_roundtrip_mutated(show4dstem_widget):
    """Mutating every trait then roundtripping preserves the mutations."""
    # Position / frame indices are clamped to valid range by trait validators
    # against the data dimensions; enum-like strings are validated choices.
    # Mutating those generically is meaningless here.
    skip = {
        "pos_row",
        "pos_col",
        "frame_idx",
        "path_index",
        "path_length",
        "vi_roi_center_row",
        "vi_roi_center_col",
        "view_mode",
        "compare_layout",
        "compare_dp_mode",
        "compare_group_mode",
    }
    original = show4dstem_widget.state_dict()
    mutated = _mutate_state(original)
    for key in skip:
        if key in mutated:
            mutated[key] = original[key]
    show4dstem_widget.load_state_dict(mutated)
    out = show4dstem_widget.state_dict()
    for k, v in mutated.items():
        if k in skip:
            continue
        if isinstance(v, float):
            assert abs(out[k] - v) < 1e-3, f"{k}: expected {v}, got {out[k]}"
        else:
            assert out[k] == v, f"{k}: expected {v!r}, got {out[k]!r}"


def test_show4dstem_save_and_load(tmp_path, show4dstem_widget):
    """save() writes a versioned envelope JSON, state= kwarg loads it."""
    show4dstem_widget.dp_colormap = "viridis"
    show4dstem_widget.vi_colormap = "magma"
    show4dstem_widget.show_fft = True
    path = tmp_path / "show4dstem_state.json"
    show4dstem_widget.save(str(path))

    payload = json.loads(path.read_text())
    assert payload["widget_name"] == "Show4DSTEM"
    assert "metadata_version" in payload
    assert "state" in payload

    data = np.random.default_rng(0).poisson(5, (8, 8, 16, 16)).astype(np.uint16)
    data[:, :, 6:10, 6:10] += 500
    fresh = Show4DSTEM(data, state=str(path), verbose=False)
    assert fresh.dp_colormap == "viridis"
    assert fresh.vi_colormap == "magma"
    assert fresh.show_fft is True


def test_show4dstem_controls_collapsed_roundtrips_state_and_html(tmp_path):
    """Controls can be collapsed reversibly and persist through standalone export."""
    rng = np.random.default_rng(1)
    data = rng.integers(0, 100, (4, 4, 8, 8), dtype=np.uint16)
    widget = Show4DSTEM(
        data,
        show_controls=True,
        controls_collapsed=True,
        verbose=False,
    )

    assert widget.controls_collapsed is True
    assert widget.expand_controls() is widget
    assert widget.controls_collapsed is False
    assert widget.collapse_controls() is widget
    assert widget.controls_collapsed is True
    assert widget.toggle_controls() is widget
    assert widget.controls_collapsed is False

    widget.collapse_controls()
    state = widget.state_dict()
    assert state["show_controls"] is True
    assert state["controls_collapsed"] is True

    restored = Show4DSTEM(data, state=state, verbose=False)
    assert restored.controls_collapsed is True

    clone = widget._clone_for_html_export(dtype="uint16", det_bin=1)
    try:
        assert clone.controls_collapsed is True
        assert clone.state_dict()["controls_collapsed"] is True
    finally:
        clone.close()

    out = widget.export_html(tmp_path / "show4dstem_controls_collapsed.html", encoding="full")
    html = out.read_text(encoding="utf-8")
    assert "controls_collapsed" in html
    assert "Hide controls" not in html
    assert "Show controls" not in html


def test_show4dstem_ui_mode_presets_and_overrides():
    rng = np.random.default_rng(2)
    data = rng.integers(0, 100, (4, 4, 8, 8), dtype=np.uint16)

    presentation = Show4DSTEM(data, ui_mode="presentation", verbose=False)
    assert presentation.show_title is True
    assert presentation.show_controls is True
    assert presentation.controls_collapsed is True
    assert presentation.show_stats is False
    assert presentation.show_scale_bar is True

    report = Show4DSTEM(data, ui_mode="report", verbose=False)
    assert report.show_title is True
    assert report.show_controls is False
    assert report.controls_collapsed is False
    assert report.show_stats is False
    assert report.show_scale_bar is True

    minimal = Show4DSTEM(data, ui_mode="minimal", verbose=False)
    assert minimal.show_title is False
    assert minimal.show_controls is False
    assert minimal.controls_collapsed is False
    assert minimal.show_stats is False
    assert minimal.show_scale_bar is False

    override = Show4DSTEM(
        data,
        ui_mode="minimal",
        show_title=True,
        show_controls=True,
        controls_collapsed=True,
        show_stats=True,
        show_scale_bar=True,
        verbose=False,
    )
    assert override.show_title is True
    assert override.show_controls is True
    assert override.controls_collapsed is True
    assert override.show_stats is True
    assert override.show_scale_bar is True


# ---------------------------------------------------------------------------
# Show2D
# ---------------------------------------------------------------------------

@pytest.fixture
def show2d_widget():
    return Show2D(np.random.default_rng(0).standard_normal((32, 32)).astype(np.float32), verbose=False)


def test_show2d_state_dict_keys(show2d_widget):
    s = show2d_widget.state_dict()
    assert isinstance(s, dict)
    assert len(s) > 5
    for required in ("cmap", "log_scale", "show_title", "show_controls", "controls_collapsed"):
        assert required in s, f"state_dict missing key {required!r}"


def test_show2d_state_dict_roundtrip_defaults(show2d_widget):
    original = show2d_widget.state_dict()
    fresh = Show2D(np.random.default_rng(0).standard_normal((32, 32)).astype(np.float32),
                   state=original, verbose=False)
    restored = fresh.state_dict()
    for k in original:
        if isinstance(original[k], float):
            assert abs(restored[k] - original[k]) < 1e-3, f"{k}: {original[k]} -> {restored[k]}"
        else:
            assert restored[k] == original[k], f"{k}: {original[k]!r} -> {restored[k]!r}"


def test_show2d_save_and_load(tmp_path, show2d_widget):
    show2d_widget.cmap = "viridis"
    show2d_widget.log_scale = True
    path = tmp_path / "show2d_state.json"
    show2d_widget.save(str(path))

    payload = json.loads(path.read_text())
    assert payload["widget_name"] == "Show2D"
    assert "state" in payload

    fresh = Show2D(np.random.default_rng(0).standard_normal((32, 32)).astype(np.float32),
                   state=str(path), verbose=False)
    assert fresh.cmap == "viridis"
    assert fresh.log_scale is True


def test_show2d_per_panel_cmap_state_roundtrip():
    data = np.random.default_rng(1).standard_normal((2, 16, 16)).astype(np.float32)
    widget = Show2D(data, cmap=["inferno", "viridis"], verbose=False)

    assert widget.cmap == "inferno"
    assert widget.panel_cmaps == ["inferno", "viridis"]

    fresh = Show2D(data, state=widget.state_dict(), verbose=False)
    assert fresh.cmap == "inferno"
    assert fresh.panel_cmaps == ["inferno", "viridis"]


def test_show2d_inset_plots_state_roundtrip():
    data = np.random.default_rng(3).standard_normal((2, 16, 16)).astype(np.float32)
    widget = Show2D(
        data,
        inset_plots=[
            {"x": [0, 1, 2], "y": [0.2, 0.8, 0.5], "point": (1, 0.8), "title": "ACF"},
            {"x": [0, 1, 2], "y": [0.5, 0.4, 0.7], "point": (2, 0.7), "title": "R"},
        ],
        verbose=False,
    )

    fresh = Show2D(data, state=widget.state_dict(), verbose=False)
    assert fresh.inset_plots == widget.inset_plots


def test_show2d_scale_bar_layout_state_roundtrip():
    data = np.random.default_rng(4).standard_normal((16, 16)).astype(np.float32)
    widget = Show2D(
        data,
        scale_bar_position="bottom-left",
        show_zoom_indicator=False,
        verbose=False,
    )

    state = widget.state_dict()
    fresh = Show2D(data, state=state, verbose=False)

    assert state["scale_bar_position"] == "bottom-left"
    assert state["show_zoom_indicator"] is False
    assert fresh.scale_bar_position == "bottom-left"
    assert fresh.show_zoom_indicator is False
    assert fresh._static_overlay_texts()[0][1] == ""


def test_show3d_per_panel_cmap_state_roundtrip():
    rng = np.random.default_rng(2)
    data_a = rng.standard_normal((3, 16, 16)).astype(np.float32)
    data_b = rng.standard_normal((3, 16, 16)).astype(np.float32)
    widget = Show3D(data_a, data_b, cmap=["inferno", "viridis"], verbose=False)

    assert widget.cmap == "inferno"
    assert widget.panel_cmaps == ["inferno", "viridis"]

    fresh = Show3D(data_a, data_b, state=widget.state_dict(), verbose=False)
    assert fresh.cmap == "inferno"
    assert fresh.panel_cmaps == ["inferno", "viridis"]
