"""Show2D current-view contract: capture the exact field of view you zoomed to.

The browser syncs the visible region (row0, row1, col0, col1, image pixel
coordinates) to the two-way ``view_box`` trait on every pan/zoom (debounced).
``Show2D.current_view`` exposes it Python-side so a user can record where they
were and reproduce the same crop later via ``Show2D(data, view_box=...)``.
These tests lock the trait sync direction, the calibrated-extent math, and the
construct-from-box round trip.
"""
import numpy as np
import pytest

from quantem.widget import Show2D


def _image(n=128):
    rng = np.random.default_rng(0)
    return rng.random((n, n)).astype(np.float32)


# ---------------------------------------------------------------------------
# Trait sync direction
# ---------------------------------------------------------------------------

def test_view_box_is_a_synced_trait():
    w = Show2D(_image(), verbose=False)
    assert w.trait_metadata("view_box", "sync") is True


def test_constructor_view_box_populates_trait_and_zoom():
    """Python -> JS: view_box= sugar must both center the zoom (legacy
    behavior) and seed the trait so current_view is correct before any
    browser interaction."""
    w = Show2D(_image(128), view_box=(32, 96, 16, 80), verbose=False)
    assert w.view_box == [32.0, 96.0, 16.0, 80.0]
    assert w.initial_zoom == pytest.approx(2.0)  # 128 / 64
    assert w.zoom_row == pytest.approx(64.0)
    assert w.zoom_col == pytest.approx(48.0)


def test_js_update_reflected_in_current_view():
    """JS -> Python: a pan/zoom in the browser writes the trait; current_view
    must report exactly that region."""
    w = Show2D(_image(128), verbose=False)
    w.view_box = [10.0, 74.0, 20.0, 84.0]  # what the frontend would send
    view = w.current_view
    assert (view["row0"], view["row1"]) == (10.0, 74.0)
    assert (view["col0"], view["col1"]) == (20.0, 84.0)
    assert view["height"] == pytest.approx(64.0)
    assert view["width"] == pytest.approx(64.0)
    assert view["box"] == (10.0, 74.0, 20.0, 84.0)


# ---------------------------------------------------------------------------
# Fallback derivation (no browser update yet)
# ---------------------------------------------------------------------------

def test_current_view_derives_from_zoom_before_any_js_update():
    w = Show2D(_image(128), zoom=2.0, verbose=False)
    view = w.current_view
    assert (view["row0"], view["row1"]) == (32.0, 96.0)
    assert (view["col0"], view["col1"]) == (32.0, 96.0)
    assert view["zoom"] == pytest.approx(2.0)
    assert "unit" not in view  # pixel_size 0 -> no calibrated extents


def test_current_view_default_is_full_image():
    w = Show2D(_image(128), verbose=False)
    view = w.current_view
    assert view["box"] == (0.0, 128.0, 0.0, 128.0)
    assert view["height"] == 128.0 and view["width"] == 128.0


def test_current_view_uses_zoom_center():
    w = Show2D(_image(128), zoom=4.0, zoom_row=16.0, zoom_col=112.0, verbose=False)
    view = w.current_view
    # Half extent = 128 / (2*4) = 16 px around the center, clamped to the image.
    assert (view["row0"], view["row1"]) == (0.0, 32.0)
    assert (view["col0"], view["col1"]) == (96.0, 128.0)


# ---------------------------------------------------------------------------
# Calibrated extents
# ---------------------------------------------------------------------------

def test_current_view_calibrated_extents():
    w = Show2D(_image(128), sampling=0.5, units="nm", verbose=False)
    w.view_box = [10.0, 74.0, 20.0, 84.0]
    view = w.current_view
    assert view["row0_cal"] == pytest.approx(5.0)
    assert view["row1_cal"] == pytest.approx(37.0)
    assert view["col0_cal"] == pytest.approx(10.0)
    assert view["col1_cal"] == pytest.approx(42.0)
    assert view["height_cal"] == pytest.approx(32.0)
    assert view["width_cal"] == pytest.approx(32.0)
    assert view["unit"] == "nm"


# ---------------------------------------------------------------------------
# Round trip: capture a view, reproduce the crop in a new widget
# ---------------------------------------------------------------------------

def test_round_trip_reproduces_the_same_crop():
    data = _image(128)
    w1 = Show2D(data, verbose=False)
    w1.view_box = [32.0, 96.0, 40.0, 104.0]  # user zoomed here in the browser
    captured = w1.current_view
    w2 = Show2D(data, view_box=captured["box"], verbose=False)
    view2 = w2.current_view
    assert view2["box"] == pytest.approx(captured["box"])
    assert w2.initial_zoom == pytest.approx(2.0)
    assert w2.zoom_row == pytest.approx(64.0)
    assert w2.zoom_col == pytest.approx(72.0)


def test_view_box_survives_state_dict_round_trip():
    data = _image(128)
    w1 = Show2D(data, verbose=False)
    w1.view_box = [8.0, 40.0, 16.0, 48.0]
    w2 = Show2D(data, verbose=False)
    w2.load_state_dict(w1.state_dict())
    assert w2.view_box == [8.0, 40.0, 16.0, 48.0]
    assert w2.current_view["box"] == (8.0, 40.0, 16.0, 48.0)
