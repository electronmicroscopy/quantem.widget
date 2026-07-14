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
from quantem.widget.utils.array import bin2d


def _image(n=128):
    rng = np.random.default_rng(0)
    return rng.random((n, n)).astype(np.float32)


# ---------------------------------------------------------------------------
# Trait sync direction
# ---------------------------------------------------------------------------

def test_view_box_is_a_synced_trait():
    w = Show2D(_image(), verbose=False)
    assert w.trait_metadata("view_box", "sync") is True


def test_presentation_mode_keeps_controls_recoverable():
    """C1: presentation starts clean but keeps a Controls affordance."""
    w = Show2D(_image(), ui_mode="presentation", verbose=False)

    assert w.show_controls is True
    assert w.controls_collapsed is True
    assert w.show_stats is False


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


def test_inset_plots_normalize_per_panel_and_static_png():
    """C1: scientist attaches one calibration curve per image panel, expect
    JSON-safe widget state and a saved-notebook PNG fallback that can render."""
    data = np.stack([_image(64), _image(64) * 0.5 + 0.2])
    x = np.linspace(0.0, 1.0, 8)
    plots = [
        {
            "x": x,
            "y": np.sin(x * np.pi),
            "point": (0.5, 1.0),
            "xlabel": "R",
            "box": (0.58, 0.53, 0.34, 0.22),
            "show_ticks": True,
            "xticks": (0, 1),
            "yticks": (0, 1),
            "legend": "ACF/R",
            "annotation": "R*=0.5",
            "tick_font_size": 8,
            "label_font_size": 9,
            "legend_font_size": 10,
            "border_color": "#ffffff",
            "border_width": 2,
            "background_alpha": 0.42,
            "text_color": "#f8f8ff",
            "tick_color": "#d0e8ff",
            "margin": (18, 24),
        },
        {"x": x, "y": np.cos(x * np.pi) ** 2, "point": (0.25, 0.5), "ylim": (0, 1)},
    ]
    w = Show2D(data, inset_plots=plots, marker_colors=["#2e7d32", "#d81b60"], verbose=False)

    assert len(w.inset_plots) == 2
    assert w.inset_plots[0]["x"] == pytest.approx(x.tolist())
    assert w.inset_plots[0]["point"] == pytest.approx([0.5, 1.0])
    assert w.inset_plots[0]["box"] == pytest.approx([0.58, 0.53, 0.34, 0.22])
    assert w.inset_plots[0]["show_ticks"] is True
    assert w.inset_plots[0]["xticks"] == pytest.approx([0.0, 1.0])
    assert w.inset_plots[0]["legend"] == "ACF/R"
    assert w.inset_plots[0]["annotation"] == "R*=0.5"
    assert w.inset_plots[0]["tick_font_size"] == pytest.approx(8.0)
    assert w.inset_plots[0]["border_color"] == "#ffffff"
    assert w.inset_plots[0]["border_width"] == pytest.approx(2.0)
    assert w.inset_plots[0]["background_alpha"] == pytest.approx(0.42)
    assert w.inset_plots[0]["text_color"] == "#f8f8ff"
    assert w.inset_plots[0]["tick_color"] == "#d0e8ff"
    assert w.inset_plots[0]["margin"] == pytest.approx([18.0, 24.0])
    assert w.inset_plots[1]["ylim"] == pytest.approx([0.0, 1.0])
    assert "inset_plots" in w.state_dict()
    assert w.state_dict()["show_inset_plots"] is True
    assert w._static_png_b64(max_px=220)


def test_inset_plot_can_broadcast_single_spec():
    data = np.stack([_image(32), _image(32), _image(32)])
    w = Show2D(data, inset_plots={"y": [0, 1, 0], "title": "score"}, verbose=False)
    assert len(w.inset_plots) == 3
    assert all(plot["title"] == "score" for plot in w.inset_plots)


def test_export_svg_writes_hybrid_figure(tmp_path):
    """C1: multipanel figure, expect SVG vector chrome with embedded images."""
    data = np.stack([_image(32), _image(32) * 0.5, _image(32) + 0.25])
    w = Show2D(
        data,
        labels=["raw", "filtered", "residual"],
        title="Show2D Figure",
        ncols=2,
        sampling=0.2,
        units="nm",
        marker_colors=["#2e7d32", "#c62828", "#1565c0"],
        verbose=False,
    )

    out = w.export_svg(tmp_path / "figure.svg")
    svg = out.read_text(encoding="utf-8")

    assert out.name == "figure.svg"
    assert 'data-show2d-svg-export="true"' in svg
    assert 'data-raster-scale="3"' in svg
    assert svg.count("<image ") == 3
    assert "data:image/png;base64," in svg
    assert ">Show2D Figure<" in svg
    assert ">raw<" in svg
    assert ">filtered<" in svg
    assert ">residual<" in svg
    assert "nm<" in svg
    assert 'fill="#2e7d32"' in svg


def test_export_svg_respects_hidden_panels_and_order(tmp_path):
    """C2: curated gallery state, expect SVG follows visible panel order."""
    data = np.stack([_image(16) + i for i in range(4)])
    w = Show2D(data, labels=["a", "b", "c", "d"], ncols=3, verbose=False)
    w.set_panel_order([2, 0, 3, 1])
    w.hide_panel(0)

    out = w.export_svg(tmp_path / "ordered.svg", scale=3, include_scale_bar=False)
    svg = out.read_text(encoding="utf-8")

    assert 'data-raster-scale="3"' in svg
    assert svg.count("<image ") == 3
    assert 'data-show2d-panel="2"' in svg
    assert 'data-show2d-panel="3"' in svg
    assert 'data-show2d-panel="1"' in svg
    assert 'data-show2d-panel="0"' not in svg
    assert ">c<" in svg
    assert ">d<" in svg
    assert ">b<" in svg


def test_export_svg_wraps_long_panel_labels(tmp_path):
    """C3: long panel label, expect exported SVG uses multiple text lines."""
    w = Show2D(
        _image(32),
        labels=["alpha beta gamma delta epsilon"],
        size=120,
        panel_title_font_size=10,
        verbose=False,
    )

    out = w.export_svg(tmp_path / "wrapped.svg")
    svg = out.read_text(encoding="utf-8")

    assert ">alpha beta<" in svg
    assert ">gamma delta<" in svg
    assert ">epsilon<" in svg
    assert ">alpha beta gamma delta epsilon<" not in svg


def test_export_svg_writes_publication_layers_as_vectors(tmp_path):
    """C4: publication callouts, expect editable SVG vector layers."""
    data = np.stack([_image(32), _image(32) * 0.5])
    w = Show2D(
        data,
        labels=["raw", "denoised"],
        panel_title_spans=[
            [{"math": r"\lambda=0.03", "color": "#60a5fa"}, {"text": " raw"}],
            [{"math": r"\chi^2"}, {"text": "/px"}],
        ],
        panel_annotations={
            "raw": {"math": r"\lambda", "position": "top-right", "variant": "outline"},
        },
        panel_overlays={
            "raw": {
                "shape": "circle",
                "center": (15, 16),
                "radius": 6,
                "stroke": "#facc15",
                "line_style": "dashed",
            },
            "denoised": {
                "shape": "rect",
                "box": (8, 9, 24, 26),
                "stroke": "#34d399",
                "dash": [5, 2, 1, 2],
            },
        },
        inset_plots={
            "x": [0, 1, 2],
            "y": [0.2, 0.6, 0.4],
            "legend": "ACF",
            "point": [1, 0.6],
        },
        row_markers={0: "#60a5fa"},
        col_markers={1: "#f87171"},
        ncols=2,
        verbose=False,
    )

    out = w.export_svg(tmp_path / "publication.svg", include_colorbar=True)
    svg = out.read_text(encoding="utf-8")

    assert 'data-show2d-vector-layer="true"' in svg
    assert 'data-show2d-panel-title-spans-svg="true"' in svg
    assert 'data-show2d-panel-overlay-svg="true"' in svg
    assert 'data-show2d-panel-annotation-svg="true"' in svg
    assert 'data-show2d-inset-plot-svg="true"' in svg
    assert 'data-show2d-colorbar-svg="true"' in svg
    assert 'data-show2d-group-marker-svg="row"' in svg
    assert 'data-show2d-group-marker-svg="col"' in svg
    assert "<circle " in svg
    assert "<polyline " in svg
    assert "stroke-dasharray" in svg
    assert "λ=0.03" in svg
    assert "χ^2" in svg
    assert "ACF" in svg


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


def test_show2d_identity_markers_histogram_preset_and_flips_round_trip():
    # C1: scientist/agent-readable identity markers, contrast preset state,
    # and display-only flips are lightweight view state and survive save/load.
    data = np.random.default_rng(169).random((3, 16, 16), dtype=np.float32)
    w1 = Show2D(
        data,
        marker_colors=["green", "red", "magenta"],
        marker_style="around",
        contrast_preset="2-98",
        show_histogram_advanced=True,
        image_flips_horizontal=[True, False, True],
        image_flips_vertical=[False, True, False],
        verbose=False,
    )
    w2 = Show2D(data, verbose=False)

    w2.load_state_dict(w1.state_dict())

    assert w2.marker_colors == ["green", "red", "magenta"]
    assert w2.marker_style == "around"
    assert w2.contrast_preset == "2-98"
    assert w2.show_histogram_advanced is True
    assert w2.image_flips_horizontal == [True, False, True]
    assert w2.image_flips_vertical == [False, True, False]
    assert w2._data.shape == data.shape


def test_binned_preview_detail_request_returns_full_resolution_crop():
    data = np.arange(2 * 96 * 128, dtype=np.float32).reshape(2, 96, 128)
    w = Show2D(data, display_bin=4, verbose=False)

    # Browser requests are expressed in preview pixels.  The Python side must
    # convert back to the full-resolution source, snap outward to the requested
    # tile bin, and return a compact float32 tile.
    request = {
        "id": "detail-1",
        "tiles": [
            {"panel": 1, "row0": 3.25, "row1": 15.75, "col0": 4.5, "col1": 19.25, "bin": 2}
        ],
    }
    w._detail_request = __import__("json").dumps(request)

    meta = __import__("json").loads(w._detail_meta)
    assert meta["id"] == "detail-1"
    assert len(meta["tiles"]) == 1

    tile_meta = meta["tiles"][0]
    assert tile_meta["panel"] == 1
    assert tile_meta["row0"] == 12
    assert tile_meta["col0"] == 18
    assert tile_meta["bin"] == 2

    rows = tile_meta["rows"]
    cols = tile_meta["cols"]
    byte_count = rows * cols * 4
    raw = bytes(w._detail_bytes[:byte_count])
    actual = np.frombuffer(raw, dtype=np.float32).reshape(rows, cols)
    expected = bin2d(
        data[1, tile_meta["row0"]:tile_meta["row0"] + rows * 2,
             tile_meta["col0"]:tile_meta["col0"] + cols * 2],
        factor=2,
        mode="mean",
    )
    assert np.array_equal(actual, expected)


def test_set_image_replaces_stack_in_place_and_resets_stale_view_state():
    w = Show2D(np.zeros((2, 8, 8), dtype=np.float32), labels=["old 0", "old 1"], offline=False, verbose=False)
    w.view_box = [1.0, 4.0, 2.0, 5.0]
    w._detail_meta = "stale"
    w._detail_bytes = b"stale"
    old_frame_bytes = bytes(w.frame_bytes)

    data = np.stack([
        np.full((6, 6), 2.0, dtype=np.float32),
        np.full((6, 6), 4.0, dtype=np.float32),
        np.full((6, 6), 6.0, dtype=np.float32),
    ])
    w.set_image(data, labels=["new 0", "new 1", "new 2"])

    assert w._data.shape == (3, 6, 6)
    assert w.n_images == 3
    assert w.height == 6
    assert w.width == 6
    assert w.labels == ["new 0", "new 1", "new 2"]
    assert w.starred == [0, 0, 0]
    assert w.hidden_panels == []
    assert w.view_box == []
    assert w._detail_meta == ""
    assert w._detail_bytes == b""
    assert w.stats_mean == [2.0, 4.0, 6.0]
    assert bytes(w.frame_bytes) != old_frame_bytes
    assert len(w.frame_bytes) == data.nbytes


# ---------------------------------------------------------------------------
# Reversible view ops: crop_to_view() + pad_ratio (display window, single panel)
# ---------------------------------------------------------------------------

def test_crop_to_view_commits_viewport_and_raw_data_stays():
    """Zoom into a feature with view_box, commit it with crop_to_view(): the
    displayed extent shrinks to the window while the stored array, and the
    stats computed from it, never change."""
    image = _image(256)
    kept = image.copy()
    w = Show2D(image, view_box=(64, 64, 96), verbose=False)
    stats_before = (list(w.stats_mean), list(w.stats_min), list(w.stats_max))
    w.crop_to_view()
    assert w.view_crop == [64, 160, 64, 160]
    assert (w.height, w.width) == (96, 96)
    sent = np.frombuffer(w.frame_bytes, dtype=np.float32, count=96 * 96).reshape(96, 96)
    np.testing.assert_array_equal(sent, image[64:160, 64:160])
    np.testing.assert_array_equal(w._data[0], kept)  # display-only: raw intact
    assert (list(w.stats_mean), list(w.stats_min), list(w.stats_max)) == stats_before
    assert "view: cropped to (64,64)-(160,160)" in w.view_banner
    # Cursor readouts stay full-image: JS adds this native-pixel offset.
    assert w._view_crop_offset == [64, 64]


def test_crop_applies_before_denoise():
    """Denoise operates on the cropped region: the packed frame equals
    filtering the cropped window, not cropping a full-frame filter result."""
    from quantem.widget.utils.display_filter import apply_display_filter

    image = _image(256)
    w = Show2D(image, denoise="gaussian", denoise_sigma=3, view_box=(32, 32, 64), verbose=False)
    w.crop_to_view()
    sent = np.frombuffer(w.frame_bytes, dtype=np.float32, count=64 * 64).reshape(64, 64)
    expected = apply_display_filter(image[32:96, 32:96], mode="gaussian", sigma=3.0)
    np.testing.assert_allclose(sent, expected, rtol=1e-6)


def test_pad_ratio_adds_min_valued_border():
    """pad_ratio=0.1 grows the packed frame by the border on each side and
    fills it with the image minimum so the colormap floor is unchanged."""
    image = _image(128)
    w = Show2D(image, pad_ratio=0.1, verbose=False)
    pad = round(0.1 * 128)
    assert (w.height, w.width) == (128 + 2 * pad, 128 + 2 * pad)
    n = w.height * w.width
    sent = np.frombuffer(w.frame_bytes, dtype=np.float32, count=n).reshape(w.height, w.width)
    np.testing.assert_array_equal(sent[pad:-pad, pad:-pad], image)
    assert sent[0, 0] == image.min()  # border keeps the colormap floor
    assert "pad 10% min" in w.view_banner


def test_pad_ratio_fill_modes_change_border_value():
    """A drift-reviewer can choose min, median, or mean border intensity."""
    image = _image(64)
    w = Show2D(image, pad_ratio=0.25, pad_fill_mode="median", verbose=False)
    pad = round(0.25 * 64)
    sent = np.frombuffer(
        w.frame_bytes,
        dtype=np.float32,
        count=w.height * w.width,
    ).reshape(w.height, w.width)
    assert sent[0, 0] == pytest.approx(float(np.median(image)))
    np.testing.assert_array_equal(sent[pad:-pad, pad:-pad], image)

    w.pad_fill_mode = "mean"
    sent = np.frombuffer(
        w.frame_bytes,
        dtype=np.float32,
        count=w.height * w.width,
    ).reshape(w.height, w.width)
    assert sent[0, 0] == pytest.approx(float(np.mean(image)))


def test_padding_keeps_fft_and_scale_bar_geometry_compatible():
    """Padding grows the same display frame that FFT and scale bars inspect.

    A drift reviewer should not get a stale scale bar or FFT dimensions after
    adding a margin: the packed frame, public width/height traits, static
    preview overlays, and FFT toggle all describe one displayed canvas.
    """
    image = _image(100)
    w = Show2D(
        image,
        pad_ratio=0.2,
        pad_fill_mode="median",
        sampling=0.2,
        units="nm",
        show_fft=True,
        verbose=False,
    )
    pad = round(0.2 * 100)
    assert (w.height, w.width) == (100 + 2 * pad, 100 + 2 * pad)
    assert w.show_fft is True

    sent = np.frombuffer(w.frame_bytes, dtype=np.float32, count=w.height * w.width).reshape(w.height, w.width)
    np.testing.assert_array_equal(sent[pad:-pad, pad:-pad], image)

    specs = w._static_panel_specs()
    assert specs[0]["frame"].shape == (w.height, w.width)
    _label, zoom_text, bar_text, bar_px = w._static_overlay_texts(specs)[0]
    assert zoom_text == "1.0×"
    assert bar_text.endswith("nm")
    assert bar_px > 0


def test_reset_view_ops_restores_bit_identical_frame():
    """Crop + pad, then reset_view_ops(): the frame bytes match the original
    pack exactly, proving both ops are reversible display windows."""
    image = _image(128)
    w = Show2D(image, verbose=False)
    original = bytes(w.frame_bytes)
    w.view_box = [32.0, 96.0, 32.0, 96.0]  # what a browser zoom would sync
    w.crop_to_view()
    w.pad_ratio = 0.1
    assert bytes(w.frame_bytes) != original
    w.reset_view_ops()
    assert bytes(w.frame_bytes) == original
    assert (w.height, w.width) == (128, 128)
    assert w.view_banner == ""
    assert w._view_crop_offset == [0, 0]
    assert w.pad_ratios == [0.0]
    assert w.pad_fill_modes == ["min"]


def test_view_ops_survive_state_round_trip():
    """Save a cropped + padded session, load it into a fresh widget on the
    same data: the committed window and border come back identically."""
    image = _image(128)
    w = Show2D(image, view_box=(16, 16, 64), verbose=False)
    w.crop_to_view()
    w.pad_ratio = 0.05
    state = w.state_dict()
    restored = Show2D(image, verbose=False)
    restored.load_state_dict(state)
    assert restored.view_crop == w.view_crop
    assert restored.pad_ratio == 0.05
    assert restored.pad_ratios == [0.05]
    assert restored.pad_fill_mode == "min"
    assert bytes(restored.frame_bytes) == bytes(w.frame_bytes)


def test_gallery_padding_can_apply_to_all_panels():
    """A gallery can grow its display canvas with a shared padding ratio."""
    a = _image(32)
    b = _image(32) + 100
    w = Show2D([a, b], pad_ratio=0.25, pad_fill_mode="median", verbose=False)
    pad = round(0.25 * 32)
    assert (w.height, w.width) == (32 + 2 * pad, 32 + 2 * pad)
    arr = np.frombuffer(w.frame_bytes, dtype=np.float32).reshape(2, w.height, w.width)
    np.testing.assert_array_equal(arr[0, pad:-pad, pad:-pad], a)
    np.testing.assert_array_equal(arr[1, pad:-pad, pad:-pad], b)
    assert arr[0, 0, 0] == pytest.approx(float(np.median(a)))
    assert arr[1, 0, 0] == pytest.approx(float(np.median(b)))


def test_gallery_padding_can_target_one_panel_with_common_canvas():
    """Per-panel padding lets a user compare one drift-margin choice at a time."""
    a = _image(32)
    b = _image(32) + 100
    w = Show2D([a, b], verbose=False)
    returned = w.set_padding(0.25, fill="mean", panels=[1])
    assert returned is w
    assert w.pad_scope == "panel"
    assert w.pad_ratios == [0.0, 0.25]
    assert w.pad_fill_modes == ["min", "mean"]
    pad = round(0.25 * 32)
    assert (w.height, w.width) == (32 + 2 * pad, 32 + 2 * pad)
    arr = np.frombuffer(w.frame_bytes, dtype=np.float32).reshape(2, w.height, w.width)
    # Both panels share the larger canvas; the selected panel's fill mode is
    # mean, while the untouched panel uses the default min fill in its margin.
    assert arr[0, 0, 0] == pytest.approx(float(np.min(a)))
    assert arr[1, 0, 0] == pytest.approx(float(np.mean(b)))
    np.testing.assert_array_equal(arr[1, pad:-pad, pad:-pad], b)

    restored = Show2D([a, b], verbose=False)
    restored.load_state_dict(w.state_dict())
    assert restored.pad_ratios == w.pad_ratios
    assert restored.pad_fill_modes == w.pad_fill_modes
    assert bytes(restored.frame_bytes) == bytes(w.frame_bytes)


def test_show2d_set_padding_accepts_all_alias_for_scientist_notebooks():
    """C1: user writes panels='all', expect every panel to update."""
    a = _image(16)
    b = _image(16) + 10
    w = Show2D([a, b], verbose=False)

    returned = w.set_padding(0.1, fill="median", panels="all")

    assert returned is w
    assert w.pad_scope == "all"
    assert w.pad_ratio == pytest.approx(0.1)
    assert w.pad_ratios == [pytest.approx(0.1), pytest.approx(0.1)]
    assert w.pad_fill_modes == ["median", "median"]


def test_crop_to_view_raises_for_galleries():
    """Crop-to-view is single panel only in this release; a gallery gets a
    clear NotImplementedError instead of a silently wrong window."""
    w = Show2D([_image(64), _image(64)], verbose=False)
    with pytest.raises(NotImplementedError, match="single panel"):
        w.crop_to_view()


def test_show2d_rich_panel_titles_keep_plain_labels_and_state():
    """C1: rich spans color status words while plain labels stay usable."""
    data = np.random.default_rng(12).random((2, 16, 16), dtype=np.float32)
    widget = Show2D(
        data,
        labels=[
            [
                {"text": "BF denoise  "},
                {"text": "low", "color": "#60a5fa"},
                {"text": "  χ²="},
                {"text": "0.5", "color": "#f59e0b"},
            ],
            "BF denoise mid",
        ],
        verbose=False,
    )

    assert widget.labels == ["BF denoise  low  χ²=0.5", "BF denoise mid"]
    assert widget.panel_title_spans[0][1] == {"text": "low", "color": "#60a5fa"}
    assert widget._resolve_panel_ref("BF denoise  low  χ²=0.5") == 0

    restored = Show2D(data, verbose=False)
    restored.load_state_dict(widget.state_dict())
    assert restored.labels == widget.labels
    assert restored.panel_title_spans == widget.panel_title_spans


def test_show2d_panel_title_style_and_group_markers_round_trip():
    """C1: panel title chrome and row/column markers survive saved state."""
    data = np.random.default_rng(22).random((4, 12, 12), dtype=np.float32)
    widget = Show2D(
        data,
        labels=["A", "B", "C", "D"],
        ncols=2,
        panel_title_style={
            "bg": "rgba(0,0,0,0.72)",
            "fg": "#ffffff",
            "border_color": "#60a5fa",
            "border_width": 1,
            "pad_x": 6,
            "pad_y": 2,
            "radius": 2,
            "max_width": "hug",
        },
        row_markers={0: "#60a5fa"},
        col_markers={1: "#f59e0b"},
        verbose=False,
    )

    restored = Show2D(data, verbose=False)
    restored.load_state_dict(widget.state_dict())

    assert restored.panel_title_style == widget.panel_title_style
    assert restored.row_markers == {"0": "#60a5fa"}
    assert restored.col_markers == {"1": "#f59e0b"}


def test_show2d_panel_annotations_accept_multiple_labels_per_panel():
    """C1: arbitrary panel annotations round-trip as JSON-safe state."""
    data = np.random.default_rng(24).random((3, 12, 12), dtype=np.float32)
    widget = Show2D(
        data,
        labels=["raw", "filtered", "residual"],
        panel_annotations={
            "raw": [
                {"text": "low dose", "position": "top-left", "variant": "pill"},
                {
                    "spans": [{"text": "ROI ", "color": "#fff"}, {"text": "A", "color": "#60a5fa"}],
                    "box": [0.20, 0.25, 0.32, 0.18],
                    "class_name": "roi-a-label",
                    "bg": "rgba(0,0,0,0.55)",
                    "font_size": 12,
                },
            ],
            2: {"text": "residual", "x": 0.5, "y": 0.85, "anchor": "bottom-center"},
        },
        verbose=False,
    )

    restored = Show2D(data, labels=["raw", "filtered", "residual"], verbose=False)
    restored.load_state_dict(widget.state_dict())

    assert len(restored.panel_annotations[0]) == 2
    assert restored.panel_annotations[0][0]["variant"] == "pill"
    assert restored.panel_annotations[0][1]["class_name"] == "roi-a-label"
    assert restored.panel_annotations[0][1]["box"] == [0.2, 0.25, 0.32, 0.18]
    assert restored.panel_annotations[2][0]["anchor"] == "bottom-center"

    single = Show2D(
        data[0],
        panel_annotations=[
            {"text": "first", "position": "top-left"},
            {"text": "second", "position": "bottom-right"},
        ],
        verbose=False,
    )
    assert [item["text"] for item in single.panel_annotations[0]] == ["first", "second"]


def test_show2d_labels_and_annotations_accept_math_spans():
    """C1: TeX-style symbols are preserved for frontend math rendering."""
    data = np.random.default_rng(26).random((2, 12, 12), dtype=np.float32)
    widget = Show2D(
        data,
        labels=[
            [{"math": r"\lambda=0.03"}, {"text": " raw"}],
            r"$\chi^2$/px residual",
        ],
        panel_annotations={
            0: {"math": r"\lambda", "position": "top-left"},
            1: {
                "spans": [
                    {"math": r"\chi^2"},
                    {"text": "/px"},
                ],
                "position": "top-right",
            },
        },
        verbose=False,
    )

    restored = Show2D(data, verbose=False)
    restored.load_state_dict(widget.state_dict())

    assert restored.panel_title_spans[0][0] == {"math": r"\lambda=0.03"}
    assert restored.labels[1] == r"$\chi^2$/px residual"
    assert restored.panel_annotations[0][0]["math"] == r"\lambda"
    assert restored.panel_annotations[1][0]["spans"][0] == {"math": r"\chi^2"}


def test_show2d_panel_overlays_support_global_and_per_panel_state():
    """C1: circle/rect overlays can broadcast globally or target panels."""
    data = np.random.default_rng(28).random((2, 12, 12), dtype=np.float32)
    circle = {
        "shape": "circle",
        "center": (4, 5),
        "radius": 2,
        "stroke": "white",
        "stroke_width": 3,
        "line_style": "dashed",
    }
    rect = {
        "shape": "rect",
        "box": (1, 2, 8, 9),
        "stroke": "#f87171",
        "fill": "#f87171",
        "fill_opacity": 0.2,
        "line_dash": [5, 2, 1, 2],
        "z_order": 2,
    }

    global_widget = Show2D(data, overlays=[circle, rect], verbose=False)
    assert [len(items) for items in global_widget.panel_overlays] == [2, 2]
    assert global_widget.panel_overlays[0][0]["shape"] == "circle"
    assert global_widget.panel_overlays[0][0]["line_style"] == "dashed"
    assert global_widget.panel_overlays[1][1]["fill"] == "#f87171"
    assert global_widget.panel_overlays[1][1]["dash"] == [5.0, 2.0, 1.0, 2.0]

    rect_default_fill = dict(rect)
    rect_default_fill.pop("fill_opacity")
    list_widget = Show2D(
        [data[0], data[1]],
        labels=["raw", "denoised"],
        panel_overlays={"denoised": rect_default_fill},
        verbose=False,
    )
    assert [len(items) for items in list_widget.panel_overlays] == [0, 1]
    assert list_widget.panel_overlays[1][0]["shape"] == "rect"
    assert list_widget.panel_overlays[1][0]["fill_opacity"] == 1.0

    per_panel = Show2D(
        data,
        labels=["raw", "denoised"],
        panel_overlays={
            "raw": circle,
            "denoised": [rect],
        },
        verbose=False,
    )
    restored = Show2D(data, labels=["raw", "denoised"], verbose=False)
    restored.load_state_dict(per_panel.state_dict())

    assert [len(items) for items in restored.panel_overlays] == [1, 1]
    assert restored.panel_overlays[0][0]["row"] == 4.0
    assert restored.panel_overlays[1][0]["row1"] == 8.0
    assert restored.state_dict()["panel_overlays"][1][0]["z_order"] == 2.0
