"""Show2D behaviour against references: input normalisation, state round trip,
exports, the static preview's colormap and scale-bar math, folder watching."""

import base64
import io
import json
import re
from pathlib import Path

import matplotlib.axes
import numpy as np
import pytest
from matplotlib import colormaps
from PIL import Image

from quantem.widget import Show2D
from quantem.widget.adapters.core import make_dataset
from quantem.widget.render.figure import format_scale_label


def _frame(seed: int = 0, shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    return np.random.default_rng(seed).random(shape).astype(np.float32)


def _decode_png(widget: Show2D) -> np.ndarray:
    png_b64 = widget._static_png_b64()
    assert png_b64
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(png_b64))).convert("RGB"))


def _embedded_state(path: Path) -> dict:
    match = re.search(r'<script type="application/vnd.jupyter.widget-state\+json">\n(.*?)\n</script>', path.read_text(), re.S)
    models = json.loads(match.group(1))["state"].values()
    return next(model["state"] for model in models if model["model_name"] == "AnyModel")


# --- input normalisation -----------------------------------------------------


def test_dataset_supplies_calibration_and_title():
    dataset = make_dataset(_frame(), sampling=(0.2, 0.2), units=("nm", "nm"), name="gold")
    widget = Show2D(dataset, verbose=False)
    assert (widget.title, widget.pixel_size, widget.pixel_unit) == ("gold", 0.2, "nm")
    assert widget.n_images == 1 and widget.labels == ["Image 1"]


def test_gallery_stats_match_numpy_per_panel():
    frames = [_frame(1), 10 * _frame(2), _frame(3) - 5]
    widget = Show2D(frames, labels=["a", "b", "c"], verbose=False)
    np.testing.assert_allclose(widget.stats_mean, [frame.mean() for frame in frames], rtol=1e-6)
    np.testing.assert_allclose(widget.stats_std, [frame.std() for frame in frames], rtol=1e-5)
    np.testing.assert_allclose(widget.stats_min, [frame.min() for frame in frames])
    np.testing.assert_allclose(widget.stats_max, [frame.max() for frame in frames])
    assert widget.link_zoom and widget.link_pan  # galleries link by default
    assert len(widget.frame_bytes) == 3 * 64 * 64 * 4  # one float32 block per panel


def test_pages_flatten_to_one_stack_with_page_labels():
    pages = np.random.default_rng(0).random((3, 4, 16, 16)).astype(np.float32)
    widget = Show2D(pages, labels=["raw", "filtered", "residual", "score"], page_labels=["a", "b", "c"], verbose=False)
    assert (widget.n_pages, widget.panels_per_page, widget.n_images) == (3, 4, 12)
    assert widget.labels[4:8] == ["raw", "filtered", "residual", "score"]
    assert widget._visible_panels() == [0, 1, 2, 3]
    widget.page_idx = 2
    assert widget._visible_panels() == [8, 9, 10, 11]
    widget.star_page(1)
    assert widget.page_starred == [0, 1, 0]
    with pytest.raises(ValueError):
        widget.star_page(3)


def test_stack_panels_keep_independent_frames():
    stacks = [np.stack([np.full((5, 6), offset + k, dtype=np.float32) for k in range(count)]) for count, offset in zip([2, 3, 4], [0.0, 10.0, 20.0])]
    widget = Show2D(stacks, labels=["baseline", "coarse z", "fine z"], panel_frame_indices=[0, 1, -1], verbose=False)
    assert widget.panel_frame_counts == [2, 3, 4]
    assert widget.panel_frame_indices == [0, 1, 3]
    np.testing.assert_allclose(widget.stats_mean, [0.0, 11.0, 23.0])
    widget.set_panel_frame("fine z", 2)
    np.testing.assert_allclose(widget.stats_mean, [0.0, 11.0, 22.0])
    with pytest.raises(IndexError):
        widget.set_panel_frame("baseline", 5)
    assert len(widget.panel_stack_bytes) == (2 + 3 + 4) * 5 * 6 * 4
    assert widget.panel_stack_offsets == [0, 2 * 30, 5 * 30]


def test_rgb_panels_bypass_colormap_and_feed_luminance_to_stats():
    rgb = np.zeros((32, 40, 3), dtype=np.float32)
    rgb[..., 0] = 1.0  # pure red
    gray = _frame(4, (32, 40))
    widget = Show2D([rgb, gray], verbose=False)
    assert widget.is_rgb == [True, False]
    assert widget.export_enabled is False
    np.testing.assert_allclose(widget.stats_mean[0], 0.2126, rtol=1e-5)  # Rec. 709 red weight
    assert len(widget.frame_bytes) - 1 == (3 + 1) * 32 * 40 * 4  # padded to a multiple of 3 for unpadded base64
    with pytest.raises(NotImplementedError):
        Show2D([rgb], offline=True, verbose=False)


def test_set_image_replaces_the_stack_in_place():
    widget = Show2D(_frame(), cmap="gray", verbose=False)
    widget.set_image([_frame(1, (32, 32)), _frame(2, (32, 32))], labels=["one", "two"])
    assert (widget.n_images, widget.height, widget.width) == (2, 32, 32)
    assert widget.labels == ["one", "two"] and widget.cmap == "gray"
    assert widget.starred == [0, 0] and widget.roi_list == []
    with pytest.raises(ValueError, match="labels length"):
        widget.set_image(_frame(), labels=["a", "b"])


def test_set_image_owns_its_pixels_like_the_constructor():
    image = _frame(5, (16, 16))
    widget = Show2D(_frame(), verbose=False)
    widget.set_image(image)
    image[:] = 100.0  # the caller reuses its buffer for the next acquisition
    assert widget._data.max() < 1.0 and widget.stats_max[0] < 1.0


def test_rotation_pads_a_mixed_shape_gallery_with_the_panel_median():
    wide = _frame(1, (20, 40))
    widget = Show2D([wide, wide + 10], verbose=False)
    widget.image_rotations = [0, 1]  # 20x40 next to 40x20: both pad to 40x40
    assert widget._data.shape == (2, 40, 40)
    # the same median fill as a mixed-size gallery at construction, never a black border
    assert widget._data[0, 0, 0] == np.float32(np.median(wide))
    assert widget._data[1, 0, 0] == np.float32(np.median(wide + 10))



def test_rotating_an_rgb_panel_turns_its_color_block_with_it():
    rgb = np.random.default_rng(2).random((20, 40, 3)).astype(np.float32)
    gray = _frame(3, (20, 40))
    widget = Show2D([rgb, gray], verbose=False)
    widget.image_rotations = [1, 0]  # a 40x20 RGB panel next to 20x40: both pad to 40x40
    assert (widget.height, widget.width) == (40, 40)
    # frame_bytes: the RGB panel's 40x40x3 block, then the gray panel's 40x40 plane
    floats = np.frombuffer(widget.frame_bytes[: (3 + 1) * 40 * 40 * 4], dtype=np.float32)
    color_block = floats[: 3 * 40 * 40].reshape(40, 40, 3)
    turned = np.rot90(rgb)
    np.testing.assert_array_equal(color_block[:, 10:30], turned)
    np.testing.assert_array_equal(color_block[0, 0], np.median(turned, axis=(0, 1)).astype(np.float32))
    np.testing.assert_allclose(color_block[:, 10:30] @ np.array([0.2126, 0.7152, 0.0722], np.float32),
                               widget._data[0][:, 10:30], rtol=1e-6)
    widget.image_rotations = [0, 0]
    np.testing.assert_array_equal(np.frombuffer(widget.frame_bytes[: 3 * 20 * 40 * 4], dtype=np.float32).reshape(20, 40, 3), rgb)

def test_ui_mode_presets_and_unknown_kwargs():
    report = Show2D(_frame(), ui_mode="report", verbose=False)
    assert (report.show_controls, report.show_stats, report.show_title) == (False, False, True)
    presentation = Show2D(_frame(), ui_mode="presentation", show_stats=True, verbose=False)
    assert presentation.show_controls is True and presentation.controls_collapsed is True
    assert presentation.show_stats is True  # explicit flags beat the preset
    with pytest.raises(TypeError, match="pixel_size_angstrom"):
        Show2D(_frame(), pixel_size_angstrom=0.5, verbose=False)



def test_drift_plot_keywords_center_and_gallery_gap_px(capsys):
    # quantem's drift plots: plot_combined(interactive=True) passes center=,
    # drift.show() passes gallery_gap_px=0 with pixel_size/pixel_unit traits.
    image = _frame(0, (256, 256))
    combined = Show2D([image, image], labels=["0 deg", "90 deg"], ncols=2, size=300, display_bin=1, zoom=4,
                      center=(40, 200), ui_mode="report")
    assert (combined.zoom_row, combined.zoom_col) == (40.0, 200.0)
    # center is in full-resolution pixels; the browser reads the binned preview's pixels
    binned = Show2D(image, zoom=4, center=(40, 200), display_bin=2, verbose=False)
    assert (binned.zoom_row, binned.zoom_col) == (20.0, 100.0)
    assert Show2D(image, center=None).zoom_row is None
    shown = Show2D([image] * 3, ncols=3, gallery_gap_px=0, size=200, pixel_size=0.1, pixel_unit="nm",
                   labels=["before", "affine", "nonrigid"], zoom=1, ui_mode="interactive")
    assert (shown.inter_panel_gap_px, shown.pixel_size, shown.pixel_unit) == (0, 0.1, "nm")
    assert Show2D([image] * 2, gallery_gap_px=6).inter_panel_gap_px == 6
    with pytest.raises(ValueError, match="not both"):
        Show2D(image, gallery_gap_px=1, inter_panel_gap_px=2)
    capsys.readouterr()


def test_denoise_keywords_switch_on_the_browser_filter():
    viewer = Show2D([_frame(0), _frame(1)], denoise="anscombe", denoise_sigma=8.0, show_denoise=True)
    assert viewer.denoise_enabled and viewer.show_denoise and viewer.denoise_scope == "all"
    assert (viewer.denoise_modes, viewer.denoise_sigmas, viewer.denoise_bins) == (["anscombe"] * 2, [8.0] * 2, [1, 1])
    assert (viewer.denoise, viewer.denoise_sigma) == ("anscombe", 8.0)  # the editor shows the first panel
    per_panel = Show2D([_frame(0), _frame(1)], denoise=["none", "Gaussian"], denoise_sigma=[4.0, 2.0])
    assert per_panel.denoise_modes == ["none", "gaussian"] and per_panel.denoise_sigmas == [4.0, 2.0]
    assert per_panel.denoise_scope == "panel" and per_panel.denoise_enabled and per_panel.show_denoise
    clean = Show2D(_frame(0))
    assert clean.denoise_modes == ["none"] and not clean.denoise_enabled and not clean.show_denoise
    with pytest.raises(ValueError, match="denoise must be one of"):
        Show2D(_frame(0), denoise="nlm")  # the browser has no such filter
    with pytest.raises(ValueError, match="panel count"):
        Show2D([_frame(0), _frame(1)], denoise=["anscombe"])
    with pytest.raises(ValueError, match="denoise_scope='all'"):
        Show2D([_frame(0), _frame(1)], denoise=["none", "anscombe"], denoise_scope="all")

def test_display_bin_auto_bins_large_input_and_serves_full_resolution_tiles(capsys):
    frame = np.random.default_rng(0).random((2048, 2100)).astype(np.float32)
    native = Show2D(frame, verbose=False)
    assert (native.height, native.width) == (2048, 2100) and capsys.readouterr().out == ""  # lossless default
    widget = Show2D(frame, display_bin="auto", verbose=False)
    # 17.2 MB float32 is over the 16 MiB wire budget: preview binned so the
    # long side is about 1024 px (2100 / 1024 -> 3), full-res tiles on zoom.
    assert "Show2D display bin 3x (auto, over the display budget): 2048x2100 -> 682x700" in capsys.readouterr().out
    assert widget._display_bin_factor == 3 and (widget.height, widget.width) == (682, 700)
    widget._detail_request = json.dumps({"id": "t1", "tiles": [{"panel": 0, "row0": 10, "row1": 20, "col0": 5, "col1": 15, "bin": 1}]})
    meta = json.loads(widget._detail_meta)
    tile = meta["tiles"][0]
    assert meta["id"] == "t1" and (tile["row0"], tile["col0"], tile["rows"], tile["cols"]) == (30, 15, 30, 30)
    decoded = np.frombuffer(widget._detail_bytes, dtype=np.float32).reshape(30, 30)
    np.testing.assert_array_equal(decoded, frame[30:60, 15:45])


# --- state round trip ---------------------------------------------------------


def test_state_dict_round_trips_through_memory_and_file(tmp_path: Path):
    data = np.random.default_rng(1).standard_normal((2, 16, 16)).astype(np.float32)
    widget = Show2D(data, cmap=["inferno", "viridis"], labels=["a", "b"], inset_plots=[{"x": [0, 1], "y": [1, 2]}, None], verbose=False)
    widget.log_scale = True
    widget.vmin, widget.vmax = 0.1, 0.9
    widget.roi_active = True
    widget.roi_list = [{"shape": "circle", "row": 4, "col": 5, "radius": 3}]
    state = widget.state_dict()
    fresh = Show2D(data, state=state, verbose=False)
    assert fresh.state_dict() == state
    assert fresh.panel_cmaps == ["inferno", "viridis"] and fresh.inset_plots[0]["x"] == [0.0, 1.0]
    path = widget.save(str(tmp_path / "view.json")) or tmp_path / "view.json"
    payload = json.loads(Path(path).read_text())
    assert payload["widget_name"] == "Show2D"
    loaded = Show2D(data, state=str(path), verbose=False)
    assert loaded.log_scale is True and loaded.roi_list[0]["radius"] == 3
    # a saved layout that does not fit the new data is skipped, not applied
    single = Show2D(data[0], state=state, verbose=False)
    assert single.panel_cmaps == [] and single.n_images == 1


def test_saved_notebook_state_drops_pixels_unless_save_state():
    widget = Show2D(_frame(), verbose=False)
    state = widget.get_state()
    assert "frame_bytes" not in state and state["_static_fallback_jpeg"]
    assert widget.get_state(key="frame_bytes")["frame_bytes"]
    assert "frame_bytes" in Show2D(_frame(), save_state=True, verbose=False).get_state()


# --- exports ------------------------------------------------------------------


def test_export_html_embeds_the_model_state(tmp_path: Path):
    widget = Show2D([_frame(1), _frame(2)], labels=["a", "b"], sampling=0.5, units="nm", title="pair", verbose=False)
    path = widget.export_html(tmp_path / "pair.html")
    state = _embedded_state(path)
    assert state["_anywidget_id"] == "quantem.widget.show2d.widget.Show2D"
    assert state["labels"] == ["a", "b"] and state["pixel_size"] == 0.5 and state["title"] == "pair"
    assert state["offline"] is False and state["_export_light"] is True and state["export_enabled"] is False
    assert widget.export_status.startswith("Exported pair.html")
    quantized = widget.export_html(tmp_path / "pair_u8.html", encoding="uint8")
    u8 = _embedded_state(quantized)
    assert u8["offline"] is True and len(u8["_offline_mins"]) == 2
    assert quantized.stat().st_size < path.stat().st_size
    with pytest.raises(ValueError, match="safe limit"):
        Show2D(np.zeros((8, 2048, 2048), dtype=np.float32), display_bin=1, verbose=False).export_html(tmp_path / "big.html")


def test_export_svg_writes_vector_chrome_around_png_panels(tmp_path: Path):
    widget = Show2D(
        [_frame(1), _frame(2)],
        labels=["raw", "denoised"],
        sampling=0.23,
        units="A",
        title="figure",
        marker_colors=["#ff0000", "#00ff00"],
        panel_annotations={"denoised": {"text": "A", "position": "top-left"}},
        overlays={"raw": {"shape": "circle", "row": 32, "col": 32, "radius": 8}},
        verbose=False,
    )
    path = widget.export_svg(tmp_path / "figure.svg", include_colorbar=True)
    svg = path.read_text()
    assert svg.startswith('<?xml version="1.0"') and svg.count('<image ') == 2
    assert svg.count("<g id=\"show2d-panel-") == 2 and ">figure</text>" in svg
    assert ">raw</text>" in svg and ">A</text>" in svg and "<circle " in svg
    assert svg.count("<linearGradient") == 2 and 'fill="#ff0000"' in svg
    assert ">2 Å</text>" in svg  # 60 css px on a 300 px gallery panel of 64 px -> 2.9 A -> nice 2 A
    assert Image.open(widget.save_image(tmp_path / "raw.png", scalebar=True, colorbar=True, title=True)).size[0] > 64


def test_save_image_scale_bar_labels_the_pixel_unit(tmp_path: Path, monkeypatch):
    drawn: list[str] = []
    draw_text = matplotlib.axes.Axes.text

    def record_text(self, x, y, text, *args, **kwargs):
        drawn.append(text)
        return draw_text(self, x, y, text, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "text", record_text)
    Show2D(_frame(), sampling=0.2, units="nm", verbose=False).save_image(tmp_path / "nm.png", scalebar=True)
    assert drawn == ["2 nm"]  # 20% of 64 px x 0.2 nm = 2.56 nm -> nice 2 nm, never "2 Å"


# --- static preview math (mirrors the browser) ------------------------------


def test_static_panel_rgb_is_the_matplotlib_lut():
    frame = _frame() * 100
    for cmap_name in ("gray", "inferno", "viridis"):
        widget = Show2D(frame, cmap=cmap_name, verbose=False)
        (vmin, vmax), = widget._resolve_panel_display_ranges([frame])
        expected = (colormaps.get_cmap(cmap_name)(np.clip((frame - vmin) / (vmax - vmin), 0, 1))[..., :3] * 255).astype(np.uint8)
        np.testing.assert_array_equal(widget._static_panel_rgb(frame, vmin, vmax, cmap_name), expected)
    gray_png = _decode_png(Show2D(frame, cmap="gray", verbose=False))
    spread = gray_png.max(axis=-1).astype(int) - gray_png.min(axis=-1).astype(int)
    assert (spread <= 3).mean() > 0.99


def test_display_ranges_follow_contrast_rules():
    dim, bright = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64), None
    bright = dim * 100
    assert Show2D([dim, bright], verbose=False)._resolve_panel_display_ranges([dim, bright]) == [(0.0, 100.0)] * 2
    unlinked = Show2D([dim, bright], link_contrast=False, verbose=False)._resolve_panel_display_ranges([dim, bright])
    assert unlinked == [(0.0, 1.0), (0.0, 100.0)]
    per_image = Show2D([dim, bright], vmin=[0.1, 1.0], vmax=[0.9, 9.0], verbose=False)._resolve_panel_display_ranges([dim, bright])
    assert per_image == [(pytest.approx(0.1), pytest.approx(0.9)), (pytest.approx(1.0), pytest.approx(9.0))]
    noisy = np.random.default_rng(3).normal(100, 25, (512, 512)).astype(np.float32)
    (lo, hi), = Show2D(noisy, auto_contrast=True, verbose=False)._resolve_panel_display_ranges([noisy])
    np.testing.assert_allclose([lo, hi], np.percentile(noisy, (2, 98)), rtol=1e-6)
    (lo, hi), = Show2D(noisy, log_scale=True, verbose=False)._resolve_panel_display_ranges([noisy])
    signed_log = lambda v: np.sign(v) * np.log1p(abs(v))  # the shader keeps the sign of negative counts
    assert lo == pytest.approx(signed_log(noisy.min())) and hi == pytest.approx(signed_log(noisy.max()))


def test_scale_bar_label_matches_the_browser_port():
    frame = np.zeros((512, 512), dtype=np.float32)
    effective_zoom = 500 / 512  # single image -> 500 css px canvas
    (label, zoom_text, bar_text, bar_px), = Show2D(frame, sampling=0.23, units="A", labels=["cal"], verbose=False)._static_overlay_texts()
    assert (label, zoom_text, bar_text) == ("cal", "", "1 nm")  # 14.1 A -> nice 10 A -> 1 nm
    assert bar_px == pytest.approx(10 / 0.23 * effective_zoom)
    (_, _, bar_text, bar_px), = Show2D(frame, verbose=False)._static_overlay_texts()
    assert (bar_text, bar_px) == ("50 px", pytest.approx(50 * effective_zoom))
    (_, zoom_text, bar_text, _), = Show2D(frame, zoom=1.8, show_zoom_indicator=True, verbose=False)._static_overlay_texts()
    assert (zoom_text, bar_text) == ("1.8×", "20 px")
    assert format_scale_label(0.5, "nm") == "5 Å"
    assert format_scale_label(13.8, "A") == "1 nm"
    assert format_scale_label(20, "mrad") == "20 mrad"
    fixed = Show2D(frame, sampling=0.23, units="A", scale_bar_length=2.0, scale_bar_label="2 nm", scale_bar_panels=[0], verbose=False)
    assert fixed._static_overlay_texts()[0][2] == "2 nm"


def test_static_png_shows_diff_panel_and_rgb_pixels():
    a, b = _frame(1), _frame(2)
    widget = Show2D([a, b], verbose=False)
    widget.diff_mode = True
    specs = widget._static_panel_specs()
    assert [spec["label"] for spec in specs] == ["Image 1", "Image 2", "Diff (A − B)"]
    assert specs[2]["cmap"] == "RdBu" and specs[2]["vmin"] == -specs[2]["vmax"]
    np.testing.assert_array_equal(specs[2]["frame"], a - b)
    rgb = np.zeros((32, 32, 3), dtype=np.float32)
    rgb[..., 1] = 1.0
    png = _decode_png(Show2D([rgb], verbose=False)).astype(int)
    green = (png[..., 1] > 200) & (png[..., 0] < 60) & (png[..., 2] < 60)
    assert green.mean() > 0.5


# --- folder watching ----------------------------------------------------------


def test_from_folder_pages_and_remaps_state_by_path(tmp_path: Path):
    for index in range(21):
        np.save(tmp_path / f"frame_{index:03d}.npy", np.full((6, 8), index, dtype=np.float32))
    widget = Show2D.from_folder(tmp_path, watch=False)
    assert widget.page_kind == "items" and widget.labels[:2] == ["frame_000", "frame_001"]
    assert (widget.n_images, widget.n_pages, widget.panels_per_page) == (21, 2, 20)
    assert widget.page_labels == ["Images 1–20", "Images 21–21"]
    widget.page_idx = 1
    assert widget._visible_panels() == [20]
    widget.selected_idx = 20
    widget.starred = [0] * 20 + [1]
    np.save(tmp_path / "frame_000b.npy", np.full((6, 8), 99, dtype=np.float32))
    assert widget.poll_folder() == []  # a new file must be stable for one poll
    assert widget.poll_folder() == [1]
    assert widget.labels[1] == "frame_000b" and widget.n_images == 22
    assert widget.selected_idx == 21 and widget.starred[21] == 1  # state follows the path
    assert widget.page_idx == 1 and len(widget.folder_paths) == 22
    with pytest.raises(TypeError, match="derives labels"):
        Show2D.from_folder(tmp_path, labels=["x"])
