"""Show2D behaviour against references: input normalisation, state round trip,
exports, the static preview's colormap and scale-bar math, folder watching."""

import base64
import io
import json
import re
from pathlib import Path

import matplotlib.axes
import matplotlib.figure
import matplotlib.colorbar
import numpy as np
import pytest
import traitlets
from matplotlib import colormaps
from PIL import Image

from quantem.widget import Show1D, Show2D, Show3D
from quantem.widget.adapters.core import make_dataset
from quantem.widget.colormap import display_window
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


def test_sampling_stated_without_units_is_in_angstrom():
    # units defaulted to "px", which the unit table reads as uncalibrated, so a stated step was dropped without a message
    stated, anisotropic = Show2D(_frame(), sampling=2.0, verbose=False), Show2D(_frame(), sampling=(0.5, 0.25), verbose=False)
    assert (stated.pixel_size, stated.pixel_unit) == (2.0, "A")
    assert (anisotropic.pixel_size, anisotropic.pixel_unit) == (0.25, "A")  # the column step sets the horizontal bar
    plain, pixels = Show2D(_frame(), verbose=False), Show2D(_frame(), sampling=2.0, units="px", verbose=False)
    assert (plain.pixel_size, plain.pixel_unit) == (pixels.pixel_size, pixels.pixel_unit) == (0.0, "px")
    # an uncalibrated dataset takes the keyword the same way, and a calibrated one checks it in angstrom
    bare = Show2D(make_dataset(_frame(), sampling=(1.0, 1.0), units=("pixels", "pixels")), sampling=2.0, verbose=False)
    assert (bare.pixel_size, bare.pixel_unit) == (2.0, "A")
    nanometre = make_dataset(_frame(), sampling=(0.2, 0.2), units=("nm", "nm"))
    assert Show2D(nanometre, sampling=2.0, verbose=False).pixel_size == 2.0
    with pytest.raises(ValueError, match="contradict the calibration the dataset records"):
        Show2D(nanometre, sampling=0.2, verbose=False)


def test_units_alone_shows_a_calibrated_dataset_in_that_unit():
    # one rule for every image widget (quantem.widget.units.image_axes): the dataset's step, converted
    nanometre = make_dataset(_frame(), sampling=(0.2, 0.2), units=("nm", "nm"))
    for units, expected in (("A", (2.0, "A")), ("nm", (0.2, "nm")), (None, (0.2, "nm"))):
        shown = Show2D(nanometre, units=units, verbose=False)
        assert (shown.pixel_size, shown.pixel_unit) == expected
    gallery = Show2D([nanometre, nanometre], units="A", verbose=False)
    assert (gallery.pixel_size, gallery.pixel_unit) == (2.0, "A")
    milliradian = make_dataset(_frame(), sampling=(0.5, 0.5), units=("mrad", "mrad"))
    assert Show2D(milliradian, units="rad", verbose=False).pixel_size == pytest.approx(0.0005)
    # the row axis of a kymograph is not a length: only the axis the bar measures is converted
    kymograph = make_dataset(_frame(), sampling=(0.5, 0.2), units=("K", "nm"))
    assert Show2D(kymograph, units="A", verbose=False).pixel_size == 2.0 and Show2D(kymograph, units="nm", verbose=False).pixel_size == 0.2
    for units in ("mrad", "px"):
        with pytest.raises(ValueError, match=f"units='{units}' cannot show a dataset calibrated in 'nm'"):
            Show2D(nanometre, units=units, verbose=False)
    with pytest.raises(ValueError, match="contradict the calibration the dataset records"):
        Show2D(nanometre, sampling=0.3, units="nm", verbose=False)


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


def test_drift_plot_keywords_center_and_panel_gap_px(capsys):
    # quantem's drift plots: plot_combined(interactive=True) passes center=,
    # drift.show() passes the panel gap with the calibration.
    image = _frame(0, (256, 256))
    combined = Show2D([image, image], labels=["0 deg", "90 deg"], ncols=2, panel_width_px=300, display_bin=1, zoom=4,
                      center=(40, 200), ui_mode="report")
    assert (combined.zoom_row, combined.zoom_col) == (40.0, 200.0)
    # center is in full-resolution pixels; the browser reads the binned preview's pixels
    binned = Show2D(image, zoom=4, center=(40, 200), display_bin=2, verbose=False)
    assert (binned.zoom_row, binned.zoom_col) == (20.0, 100.0)
    assert Show2D(image, center=None).zoom_row is None
    shown = Show2D([image] * 3, ncols=3, panel_gap_px=0, panel_width_px=200, sampling=0.1, units="nm",
                   labels=["before", "affine", "nonrigid"], zoom=1, ui_mode="interactive")
    assert (shown.panel_gap_px, shown.pixel_size, shown.pixel_unit) == (0, 0.1, "nm")
    assert Show2D([image] * 2, panel_gap_px=6).panel_gap_px == 6
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


def test_denoise_traits_take_only_the_names_the_browser_filter_runs():
    # the browser compares the trait to its filter names as written: another spelling would show raw pixels under a filter banner
    viewer = Show2D([_frame(0), _frame(1)])
    viewer.denoise = "anscombe"
    viewer.denoise_modes = ["gaussian", "none"]
    for spelling in ("off", "raw", "Gaussian", ""):
        with pytest.raises(traitlets.TraitError):
            viewer.denoise = spelling
        with pytest.raises(traitlets.TraitError):
            viewer.denoise_modes = ["none", spelling]
    assert (viewer.denoise, viewer.denoise_modes) == ("anscombe", ["gaussian", "none"])

def test_display_bin_auto_bins_large_input_and_serves_full_resolution_tiles(capsys):
    frame = np.random.default_rng(0).random((2048, 2100)).astype(np.float32)
    native = Show2D(frame, verbose=False)
    assert (native.height, native.width) == (2048, 2100) and capsys.readouterr().out == ""  # lossless default
    widget = Show2D(frame, display_bin="auto", verbose=False)
    # 17.2 MB float32 is over the 16 MiB wire budget: preview binned so the
    # long side is about 1024 px (2100 / 1024 -> 3), full-res tiles on zoom.
    assert "Show2D display bin 3x (auto, over the display budget): 2048x2100 -> 682x700" in capsys.readouterr().out
    assert widget.display_bin == 3 and (widget.height, widget.width) == (682, 700)
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
    assert "frame_bytes" not in state and state["_static_fallback_image"]
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
        panel_overlays={"raw": {"shape": "circle", "center": (32, 32), "radius": 8}},
        verbose=False,
    )
    path = widget.export_svg(tmp_path / "figure.svg", include_colorbar=True)
    svg = path.read_text()
    assert svg.startswith('<?xml version="1.0"') and svg.count('<image ') == 2
    assert svg.count("<g id=\"show2d-panel-") == 2 and ">figure</text>" in svg
    assert ">raw</text>" in svg and ">A</text>" in svg and "<circle " in svg
    assert svg.count("<linearGradient") == 2 and 'fill="#ff0000"' in svg
    assert ">2 Å</text>" in svg  # 60 css px on a 300 px gallery panel of 64 px -> 2.9 A -> nice 2 A
    assert Image.open(widget.save_image(tmp_path / "raw.png", show_scale_bar=True, colorbar=True, title=True)).size[0] > 64


def test_save_image_scale_bar_labels_the_pixel_unit(tmp_path: Path, monkeypatch):
    drawn: list[str] = []
    draw_text = matplotlib.axes.Axes.text

    def record_text(self, x, y, text, *args, **kwargs):
        drawn.append(text)
        return draw_text(self, x, y, text, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "text", record_text)
    Show2D(_frame(), sampling=0.2, units="nm", verbose=False).save_image(tmp_path / "nm.png", show_scale_bar=True)
    assert drawn == ["2 nm"]  # 20% of 64 px x 0.2 nm = 2.56 nm -> nice 2 nm, never "2 Å"


def test_save_image_log_colorbar_reads_log_values_marked_log(tmp_path: Path, monkeypatch):
    # The browser's colorbar labels a log-scale window with its signed log1p ends and the word "log".
    labels: dict[str, list[str]] = {"ticks": [], "label": []}
    set_ticklabels = matplotlib.colorbar.Colorbar.set_ticklabels
    set_label = matplotlib.colorbar.Colorbar.set_label

    def record_ticklabels(self, ticklabels, *args, **kwargs):
        labels["ticks"].extend(ticklabels)
        return set_ticklabels(self, ticklabels, *args, **kwargs)

    def record_label(self, label, *args, **kwargs):
        labels["label"].append(label)
        return set_label(self, label, *args, **kwargs)

    monkeypatch.setattr(matplotlib.colorbar.Colorbar, "set_ticklabels", record_ticklabels)
    monkeypatch.setattr(matplotlib.colorbar.Colorbar, "set_label", record_label)
    frame = np.linspace(-100, 1000, 64 * 64, dtype=np.float32).reshape(64, 64)
    Show2D(frame, log_scale=True, verbose=False).save_image(tmp_path / "log.png", colorbar=True)
    low, high = display_window(-100, 1000, True)
    assert labels["ticks"][0] == f"{low:.4g}" and labels["ticks"][-1] == f"{high:.4g}"
    assert labels["label"][-1] == "log"
    labels["ticks"].clear()
    labels["label"].clear()
    Show2D(frame, verbose=False).save_image(tmp_path / "linear.png", colorbar=True)
    assert labels["ticks"][-1] == "1000" and "log" not in labels["label"]


# --- static preview math (mirrors the browser) ------------------------------


def test_static_panel_rgb_is_the_matplotlib_lut():
    frame = _frame() * 100
    for cmap_name in ("gray", "inferno", "viridis"):
        widget = Show2D(frame, cmap=cmap_name, verbose=False)
        (vmin, vmax), = widget._resolve_panel_display_ranges([frame])
        # the browser's applyColormap entry: trunc(normalized * 255), not matplotlib's own floor(normalized * 256)
        entry = np.clip((frame - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
        lut = (colormaps.get_cmap(cmap_name)(np.arange(256))[:, :3] * 255).astype(np.uint8)
        np.testing.assert_array_equal(widget._static_panel_rgb(frame, vmin, vmax, cmap_name), lut[entry])
    gray_png = _decode_png(Show2D(frame, cmap="gray", verbose=False))
    spread = gray_png.max(axis=-1).astype(int) - gray_png.min(axis=-1).astype(int)
    assert (spread <= 3).mean() > 0.99


@pytest.mark.parametrize("window", [{"vmin": -40, "vmax": 40}, {}, {"auto_contrast": True}])
def test_save_image_log_scale_keeps_negative_values_like_the_preview(tmp_path: Path, window):
    # A signed diff under log scale: the browser and the preview map v < 0 to -log1p(-v); save_image
    # once clipped to log1p(max(v, 0)) and painted the whole negative half one flat level.
    frame = np.linspace(-50, 50, 64 * 64, dtype=np.float32).reshape(64, 64)
    widget = Show2D(frame, cmap="gray", log_scale=True, verbose=False, **window)
    saved = np.asarray(Image.open(widget.save_image(tmp_path / "diff.png")).convert("RGB"))
    (vmin, vmax), = widget._resolve_panel_display_ranges([widget._data[0]])
    np.testing.assert_array_equal(saved, widget._static_panel_rgb(widget._data[0], vmin, vmax, "gray"))
    assert len(np.unique(saved[:32, :, 0])) > 100


def test_save_image_and_preview_share_the_log_window_of_the_data_range(tmp_path: Path):
    # The window under log scale is signed_log1p of the data min/max (the browser's displayRange), not
    # the min/max of the float32 log frame: those differ by a float32 step and move whole levels.
    rng = np.random.default_rng(7)
    frames = [(rng.standard_normal((16, 16)) * 10.0 ** rng.uniform(-2, 4)).astype(np.float32) for _ in range(300)]
    widget = Show2D(frames, log_scale=True, link_contrast=False, verbose=False)
    ranges = widget._resolve_panel_display_ranges(list(widget._data))
    mismatched = []
    for idx, (vmin, vmax) in enumerate(ranges):
        saved = np.asarray(Image.open(widget.save_image(tmp_path / f"{idx}.png", idx=idx)).convert("RGB"))
        if not np.array_equal(saved, widget._static_panel_rgb(widget._data[idx], vmin, vmax, widget.cmap)):
            mismatched.append(idx)
    assert mismatched == []


@pytest.mark.parametrize("log_scale", [False, True])
def test_save_image_auto_contrast_of_sparse_data_is_not_flat(tmp_path: Path, log_scale):
    # A few bright spots on an empty field put both 2/98% percentiles on 0; the preview then
    # windows the full data range, and save_image must too instead of saving one flat level.
    frame = np.zeros((64, 64), np.float32)
    rows, cols = np.random.default_rng(3).integers(0, 64, (2, 12))
    frame[rows, cols] = np.linspace(5, 900, 12, dtype=np.float32)
    widget = Show2D(frame, cmap="gray", auto_contrast=True, log_scale=log_scale, verbose=False)
    saved = np.asarray(Image.open(widget.save_image(tmp_path / "sparse.png")).convert("RGB"))
    (vmin, vmax), = widget._resolve_panel_display_ranges([widget._data[0]])
    np.testing.assert_array_equal(saved, widget._static_panel_rgb(widget._data[0], vmin, vmax, "gray"))
    assert saved.max() == 255


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


# --- review regressions (each failed on rc41) ---------------------------------


def test_overlays_written_to_the_trait_are_normalized_like_the_keyword():
    # the browser draws the normalized spec only ("rect", sorted corners): a direct trait write gets the same pass as panel_overlays=
    stack = np.stack([_frame(0), _frame(1)])
    for widget in (Show2D(_frame(0)), Show3D(stack, verbose=False)):
        widget.panel_overlays = [[{"shape": "rect", "box": (10, 12, 2, 3), "color": "red"}]]
        (overlay,) = widget.panel_overlays[0]
        assert overlay["shape"] == "rect" and overlay["color"] == "red"
        assert overlay["box"] == [2.0, 3.0, 10.0, 12.0]
        # what the browser writes back after a drag is already normalized and is kept as written
        dragged = {**overlay, "box": [4.0, 3.0, 12.0, 12.0]}
        widget.panel_overlays = [[dragged]]
        assert widget.panel_overlays == [[dragged]]
        with pytest.raises(traitlets.TraitError, match="shape must be one of"):
            widget.panel_overlays = [[{"shape": "hexagon", "box": (0, 0, 4, 4)}]]
        # a refused frontend message leaves the trait as it was
        with pytest.raises(traitlets.TraitError, match="shape must be one of"):
            widget.set_state({"panel_overlays": [[{"shape": "hexagon", "box": [0, 0, 4, 4]}]]})
        assert widget.panel_overlays == [[dragged]]


@pytest.mark.parametrize("convert", [list, lambda frames: __import__("torch").from_numpy(np.stack(frames))], ids=["list", "torch"])
def test_annotations_overlays_and_scale_bar_panels_reach_every_panel(convert):
    # A-1: the panel count came from the unconverted input, so a list or tensor counted as one panel.
    frames = [_frame(1), _frame(2), _frame(3)]
    widget = Show2D(convert(frames), panel_annotations=["A", "B", "C"], scale_bar_panels=[2],
                    panel_overlays=[None, None, {"shape": "circle", "center": (8, 8), "radius": 2}], verbose=False)
    assert [[item["text"] for item in panel] for panel in widget.panel_annotations] == [["A"], ["B"], ["C"]]
    assert [len(panel) for panel in widget.panel_overlays] == [0, 0, 1]
    assert widget.scale_bar_panels == [2]
    with pytest.raises(ValueError, match="scale_bar_panels: panel index 5"):
        Show2D(convert(frames), scale_bar_panels=[5], verbose=False)


def test_one_annotation_with_panel_targets_only_that_panel():
    # A-10: {"text": ..., "panel": 1} was broadcast to every panel.
    widget = Show2D([_frame(1), _frame(2)], panel_annotations={"text": "B", "panel": 1}, verbose=False)
    assert [[item["text"] for item in panel] for panel in widget.panel_annotations] == [[], ["B"]]


def test_keywords_the_constructor_sets_itself_raise():
    # A-2: these traits were accepted and then silently overwritten in __init__.
    # the scale bar has one name, keyword and trait; the earlier trait name is refused
    assert Show2D(_frame(), show_scale_bar=False, verbose=False).show_scale_bar is False
    with pytest.raises(TypeError, match="scale_bar_visible"):
        Show2D(_frame(), scale_bar_visible=False, verbose=False)
    with pytest.raises(TypeError, match="image_rotations"):
        Show2D(_frame(), image_rotations=[1], verbose=False)
    assert Show2D(_frame(), show_title=False, verbose=False).show_title is False
    assert Show2D(_frame(), display_bin=np.int64(2), verbose=False).display_bin == 2
    for bad in ("2", 0, -3, 1.5):
        with pytest.raises(ValueError, match="display_bin"):
            Show2D(_frame(), display_bin=bad, verbose=False)


def test_labels_of_the_wrong_length_raise():
    # A-12: set_image checked the label count, the constructor did not.
    with pytest.raises(ValueError, match="labels length"):
        Show2D([_frame(1), _frame(2)], labels=["only one"], verbose=False)


def test_col_markers_are_normalized_like_row_markers():
    # A-15: col_markers was a bare trait, so {'x': None} reached int('x') in export_svg.
    widget = Show2D([_frame(1), _frame(2)], col_markers={1: "#ff0000"}, verbose=False)
    assert widget.col_markers == {"1": "#ff0000"}
    assert Show2D([_frame(1), _frame(2)], col_markers={"x": None}, verbose=False).col_markers == {}
    with pytest.raises(ValueError, match="col_markers"):
        Show2D([_frame(1), _frame(2)], col_markers={"x": "#ff0000"}, verbose=False)


def test_cupy_pages_open_like_numpy():
    # A-13: paged detection handled only NumPy and torch, so 4D CuPy crashed on the stats trait.
    cupy = pytest.importorskip("cupy")
    pages = np.random.default_rng(0).random((2, 3, 8, 8)).astype(np.float32)
    widget = Show2D(cupy.asarray(pages), verbose=False)
    assert (widget.n_pages, widget.panels_per_page, widget.n_images) == (2, 3, 6)
    np.testing.assert_array_equal(widget._data, pages.reshape(6, 8, 8))


def test_export_clone_downsample_scales_calibration_and_view_coordinates():
    # A-3: the clone's state restored the native pixel size and coordinates after downsampling.
    widget = Show2D(_frame(0, (64, 64)), sampling=0.5, units="nm", zoom=2, center=(20, 40), verbose=False)
    widget.roi_list = [{"shape": "circle", "row": 10.0, "col": 30.0, "radius": 4.0}]
    clone = widget._clone_for_html_export(quantized=True, image_bin=2)
    assert (clone.height, clone.width, clone.pixel_size) == (32, 32, 1.0)
    assert (clone.zoom_row, clone.zoom_col) == (10.0, 20.0)
    assert clone.roi_list[0]["row"] == 5.0 and clone.roi_list[0]["radius"] == 2.0


def test_export_clone_of_a_rotated_widget_is_rotated_once():
    # A-4: the clone received rotated pixels and then rotated them again from the restored state.
    widget = Show2D(_frame(0, (16, 32)), verbose=False)
    widget.image_rotations = [1]
    clone = widget._clone_for_html_export(quantized=False)
    assert clone._display_data.shape == widget._display_data.shape == (1, 32, 16)
    np.testing.assert_array_equal(clone._display_data, widget._display_data)
    assert clone.image_rotations == [0]


def test_state_from_a_binned_widget_keeps_the_native_calibration():
    # A-5: state_dict saved the binned pixel size, load_state_dict set display_bin without binning.
    image = _frame(0, (64, 64))
    binned = Show2D(image, sampling=0.5, units="nm", display_bin=2, zoom=2, center=(20, 40), verbose=False)
    native = Show2D(image, state=binned.state_dict(), verbose=False)
    assert native.pixel_size == 0.5 and native._display_bin == 1
    assert (native.zoom_row, native.zoom_col) == (20.0, 40.0)
    again = Show2D(image, display_bin=2, state=native.state_dict(), verbose=False)
    assert again.pixel_size == 1.0 and (again.zoom_row, again.zoom_col) == (10.0, 20.0)


def test_save_image_uses_the_panel_colormap_window_and_native_pixel_size(tmp_path: Path, monkeypatch):
    # A-6: save_image used the widget cmap and window for every panel and the binned pixel size for the bar.
    first, second = _frame(1), 10 * _frame(2)
    widget = Show2D([first, second], cmap=["gray", "viridis"], vmin=[0.0, 2.0], vmax=[1.0, 8.0], verbose=False)
    saved = np.asarray(Image.open(widget.save_image(tmp_path / "b.png", idx=1)).convert("RGB"))
    np.testing.assert_array_equal(saved, widget._static_panel_rgb(second, 2.0, 8.0, "viridis"))
    drawn: list[str] = []
    draw_text = matplotlib.axes.Axes.text
    monkeypatch.setattr(matplotlib.axes.Axes, "text", lambda self, x, y, text, *a, **k: drawn.append(text) or draw_text(self, x, y, text, *a, **k))
    Show2D(_frame(0, (64, 64)), sampling=0.2, units="nm", display_bin=2, verbose=False).save_image(tmp_path / "bar.png", show_scale_bar=True)
    assert drawn == ["2 nm"]  # 20% of 64 native px x 0.2 nm = 2.56 nm -> 2 nm; the binned 0.4 nm drew "5 nm"


def test_preview_and_svg_show_the_window_around_center(tmp_path: Path):
    # A-7: the preview cropped the image center whatever center= or the synced view said, and the SVG
    # stretched the part of the window inside the image over the whole panel.
    frame = np.zeros((256, 256), dtype=np.float32)
    frame[96:160, 216:256] = 1.0  # bright block around center (128, 250)
    widget = Show2D(frame, cmap="gray", zoom=4, center=(128, 250), verbose=False)
    assert widget._static_view_window(256, 256) == (96, 160, 218, 282)
    png = _decode_png(widget).max(axis=-1)
    middle = png[png.shape[0] // 2]
    assert middle[len(middle) // 3] > 200 and middle[int(len(middle) * 0.85)] < 30  # block, then past the edge
    svg = widget.export_svg(tmp_path / "window.svg").read_text()
    embedded = re.search(r'base64,([^"]+)"', svg).group(1)
    panel = np.asarray(Image.open(io.BytesIO(base64.b64decode(embedded))).convert("L"))
    assert panel[panel.shape[0] // 2, panel.shape[1] // 3] > 200 and panel[panel.shape[0] // 2, int(panel.shape[1] * 0.85)] < 30
    widget = Show2D(frame, verbose=False)
    widget.view_box = [0.0, 128.0, 0.0, 128.0]  # the browser zoomed into the top-left quadrant
    (_, _, bar_text, bar_px), = widget._static_overlay_texts()
    assert bar_px == pytest.approx(float(bar_text.split()[0]) * 500 / 128)


def test_static_paths_flip_like_the_browser_and_draw_inset_curves(tmp_path: Path):
    # A-11: flips were applied only in the browser; A-8: the SVG inset curve had stroke="" without
    # marker colors; A-9: the preview inset margin defaulted to 0.035 CSS px instead of 12.
    frame = _frame(0, (16, 24))
    widget = Show2D(frame, cmap="gray", inset_plots=[{"x": [0, 1], "y": [0, 1], "position": "top-left"}], verbose=False)
    widget.image_flips_horizontal = [True]
    saved = np.asarray(Image.open(widget.save_image(tmp_path / "flip.png")).convert("RGB"))
    (vmin, vmax), = widget._resolve_panel_display_ranges([frame])
    np.testing.assert_array_equal(saved, widget._static_panel_rgb(frame[:, ::-1], vmin, vmax, "gray"))
    np.testing.assert_array_equal(widget._static_panel_specs()[0]["frame"], frame[:, ::-1])
    assert 'stroke=""' not in widget.export_svg(tmp_path / "inset.svg").read_text()
    figure = matplotlib.figure.Figure()
    axes = figure.add_subplot()
    widget._draw_static_inset_plot(axes, widget.inset_plots[0], panel_index=0, line_color="#ffffff")
    assert axes.child_axes[0].get_position().x0 == pytest.approx(axes.get_position().x0 + 12 / 300 * axes.get_position().width)


def test_state_loads_from_a_file_and_refuses_another_widgets_envelope(tmp_path: Path):
    # A-14: Show2D.load_state_dict refused a path, and a Show3D envelope loaded without a word.
    widget = Show2D(_frame(), log_scale=True, verbose=False)
    widget.save(tmp_path / "view.json")
    fresh = Show2D(_frame(), verbose=False)
    fresh.load_state_dict(tmp_path / "view.json")
    assert fresh.log_scale is True
    with pytest.raises(ValueError, match="'Show3D'"):
        Show2D(_frame(), state={"schema": "quantem.widget-state/2", "widget_name": "Show3D", "state": {"log_scale": True}}, verbose=False)


def test_offline_quantization_announces_itself(capsys):
    # A-37: offline=True turned every panel into 256 levels without a word.
    widget = Show2D([_frame(1), _frame(2)], offline=True, verbose=False)
    assert len(widget._offline_mins) == 2
    assert "Show2D packs 2 panel(s) as uint8" in capsys.readouterr().out
    Show2D(_frame(1), verbose=False)
    assert capsys.readouterr().out == ""


def test_set_image_with_another_panel_count_resizes_every_per_panel_list():
    # A-45: set_image left per-panel lists at the old panel count.
    widget = Show2D([_frame(1), _frame(2)], cmap=["gray", "viridis"], denoise=["none", "gaussian"],
                    marker_colors=["#f00", "#0f0"], panel_annotations=["a", "b"], verbose=False)
    widget.image_flips_horizontal = [True, False]
    widget.set_image([_frame(3), _frame(4), _frame(5)])
    assert widget.panel_cmaps == [] and widget.marker_colors == [] and widget.panel_annotations == []
    assert widget.image_flips_horizontal == [False] * 3 and len(widget.denoise_modes) == 3
    assert len(widget.frequency_filter_cutoffs) == 3 and widget.denoise_modes == [widget.denoise] * 3


# --- one name per value (earlier spellings are refused) -----------------------


def test_the_panel_gap_has_one_name_in_both_gallery_widgets():
    # the gap was inter_panel_gap_px (trait), gallery_gap_px (Show2D keyword) and panel_gap (Show3D keyword)
    assert Show2D([_frame()] * 2, panel_gap_px=6, verbose=False).state_dict()["panel_gap_px"] == 6
    for earlier in ("gallery_gap_px", "inter_panel_gap_px"):
        with pytest.raises(TypeError, match=f"unexpected keyword argument '{earlier}'"):
            Show2D([_frame()] * 2, **{earlier: 6}, verbose=False)
    stack = np.zeros((2, 8, 8), dtype=np.float32)
    stacked = Show3D(stack, stack, panel_gap_px=3, verbose=False)
    assert stacked.state_dict()["panel_gap_px"] == 3
    stacked.free()
    for earlier in ("panel_gap", "inter_panel_gap_px"):
        with pytest.raises(TypeError, match=f"unexpected keyword argument '{earlier}'"):
            Show3D(stack, stack, **{earlier: 3}, verbose=False)


def test_the_show2d_panel_width_has_one_name():
    # Show2D took size and panel_width_px for one value and kept it in a trait called size
    assert Show2D(_frame(), panel_width_px=300, verbose=False).state_dict()["panel_width_px"] == 300
    with pytest.raises(TypeError, match="unexpected keyword argument 'size'"):
        Show2D(_frame(), size=300, verbose=False)


def test_the_calibration_keywords_are_sampling_and_units_not_the_trait_names():
    # every synced trait is a keyword, so pixel_size= and pixel_unit= worked beside sampling= and units=
    stack = np.zeros((2, 8, 8), dtype=np.float32)
    calibrated = Show2D(_frame(), sampling=0.1, units="nm", verbose=False)
    assert (calibrated.pixel_size, calibrated.pixel_unit) == (0.1, "nm")
    for build in (lambda **keywords: Show2D(_frame(), verbose=False, **keywords), lambda **keywords: Show1D(np.arange(8.0), **keywords)):
        with pytest.raises(TypeError, match="sets pixel_size itself; pass sampling= instead"):
            build(pixel_size=0.1)
        with pytest.raises(TypeError, match="sets pixel_unit itself; pass units= instead"):
            build(pixel_unit="nm")
    with pytest.raises(TypeError, match="unexpected keyword argument 'pixel_size'"):
        Show3D(stack, pixel_size=0.1, verbose=False)


def test_a_page_dict_carries_its_panels_under_images():
    # a page dict gave its panels as images, data or array; a key the page did not read was dropped without a word
    pages = [{"title": "iteration 10", "images": [_frame(1), _frame(2)]}, {"title": "iteration 20", "images": [_frame(3), _frame(4)]}]
    widget = Show2D(pages, verbose=False)
    assert (widget.n_pages, widget.panels_per_page, widget.page_labels) == (2, 2, ["iteration 10", "iteration 20"])
    np.testing.assert_array_equal(widget._data[3], _frame(4))
    for earlier in ("data", "array"):
        with pytest.raises(ValueError, match=rf"page 0 only accepts keys \['images', 'labels', 'title'\], got '{earlier}'"):
            Show2D([{"title": "iteration 10", earlier: [_frame(1), _frame(2)]}], verbose=False)
    # the page title is title; label was a second spelling of it
    with pytest.raises(ValueError, match="got 'label'"):
        Show2D([{"label": "iteration 10", "images": [_frame(1), _frame(2)]}], verbose=False)


def test_overlay_keys_have_one_spelling(tmp_path: Path):
    # an overlay took up to seven spellings per value (stroke, stroke_color, border_color or color; xyxy, xywh, region,
    # row0..col1 or box; ...) and nine line styles for four; the kept keys reach the state and the SVG export
    circle = {"shape": "circle", "center": (20, 30), "radius": 5, "color": "#123456", "line_width": 3, "line_dash": [6, 2]}
    rect = {"shape": "rect", "box": (40, 50, 10, 12), "fill_color": "#abcdef", "opacity": 0.5, "line_style": "dotted", "z_order": 1}
    widget = Show2D(_frame(0), display_bin=2, panel_overlays=[circle, rect], verbose=False)
    # the browser draws on the binned preview; the saved state keeps native pixels
    assert widget.panel_overlays[0][0]["center"] == [10.0, 15.0] and widget.panel_overlays[0][1]["box"] == [5.0, 6.0, 20.0, 25.0]
    native_circle, native_rect = widget.state_dict()["panel_overlays"][0]
    assert (native_circle["center"], native_circle["radius"], native_rect["box"]) == ([20.0, 30.0], 5.0, [10.0, 12.0, 40.0, 50.0])
    svg = widget.export_svg(tmp_path / "overlays.svg").read_text()
    assert 'stroke="#123456" stroke-width="3"' in svg and 'stroke-dasharray="6 2"' in svg and 'fill="#abcdef"' in svg
    removed = {"type": "rect", "kind": "rect", "coordinate_system": "data", "row": 1, "col": 1, "y": 1, "x": 1, "r": 2,
               "size": 4, "region": (0, 0, 4, 4), "xyxy": (0, 0, 4, 4), "xywh": (0, 0, 4, 4), "row0": 0, "col0": 0, "row1": 4,
               "col1": 4, "stroke": "red", "stroke_color": "red", "border_color": "red", "fill": "red", "stroke_width": 2,
               "border_width": 2, "stroke_style": "dashed", "dash": [2, 2], "alpha": 0.5, "order": 1}
    for key, value in removed.items():
        with pytest.raises(ValueError, match=f"panel_overlays entries only accept keys .*, got '{key}'"):
            Show2D(_frame(0), panel_overlays={"shape": "circle", "center": (8, 8), "radius": 2, key: value}, verbose=False)
    for shape in ("rectangle", "square", "Circle"):
        with pytest.raises(ValueError, match="shape must be one of"):
            Show2D(_frame(0), panel_overlays={"shape": shape, "box": (0, 0, 4, 4)}, verbose=False)
    for style in ("none", "dash", "dot", "dash-dot", "dash_dot", "dash-dot-dot", "Dashed"):
        with pytest.raises(ValueError, match=r"line_style must be one of \['solid', 'dashed', 'dotted', 'dashdot'\]"):
            Show2D(_frame(0), panel_overlays={"shape": "circle", "center": (8, 8), "radius": 2, "line_style": style}, verbose=False)
    # a spec spelled only in removed keys reads as a panel mapping; the error still names an entry's keys
    with pytest.raises(ValueError, match="one panel_overlays entry takes the keys"):
        Show2D(_frame(0), panel_overlays={"row": 8, "col": 8, "r": 2}, verbose=False)


def test_annotation_keys_have_one_spelling(tmp_path: Path):
    # annotation text came from text, label or title, the text color from fg or color, class from class or class_name,
    # the region from box or region
    widget = Show2D(_frame(0), panel_annotations={"text": "A", "color": "#ff4dff", "box": (0.1, 0.2, 0.3, 0.1), "class_name": "tag"}, verbose=False)
    (annotation,) = widget.panel_annotations[0]
    assert (annotation["text"], annotation["color"], annotation["class_name"], annotation["box"]) == ("A", "#ff4dff", "tag", [0.1, 0.2, 0.3, 0.1])
    svg = widget.export_svg(tmp_path / "annotations.svg").read_text()
    assert 'fill="#ff4dff">A</text>' in svg
    for key, value in {"label": "A", "title": "A", "fg": "#fff", "class": "tag", "region": (0.1, 0.2, 0.3, 0.1)}.items():
        with pytest.raises(ValueError, match=f"panel_annotations entries only accept keys .*, got '{key}'"):
            Show2D(_frame(0), panel_annotations={"text": "A", key: value}, verbose=False)
    with pytest.raises(ValueError, match="one panel_annotations entry takes the keys"):
        Show2D(_frame(0), panel_annotations={"label": "A"}, verbose=False)
