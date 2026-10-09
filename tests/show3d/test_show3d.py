"""Show3D: frame indexing, the embedded playback stack, exports, state and folders."""

import base64
import io
import json
import pathlib

import numpy as np
import pytest
from ipywidgets.widgets.widget import _instances
from PIL import Image

from quantem.widget import Show2D, Show3D, state
from quantem.widget.adapters.core import make_dataset


def _ramp(frames: int = 6, side: int = 64, offset: float = 0.0) -> np.ndarray:
    base = np.arange(frames * side * side, dtype=np.float32).reshape(frames, side, side)
    return base / base.max() + np.float32(offset)


def _decode_png(png_b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(png_b64))).convert("RGB"))


# --- frame indexing -------------------------------------------------------------


def test_frames_are_indexed_at_display_resolution_with_diff_modes(capsys):
    stack = _ramp()
    widget = Show3D(stack, display_bin=4, verbose=False)
    # an explicit bin is announced once with the way back to native pixels
    assert "Show3D display bin 4x: 64x64 -> 16x16 (mean); pass display_bin=1 for native pixels" in capsys.readouterr().out
    assert (widget.n_slices, widget.height, widget.width, widget.display_bin) == (6, 16, 16, 4)
    binned = stack.reshape(6, 16, 4, 16, 4).mean(axis=(2, 4))
    np.testing.assert_array_equal(widget._get_display_frame(3), binned[3])
    widget.diff_mode = "previous"
    np.testing.assert_array_equal(widget._get_display_frame(0), np.zeros((16, 16), dtype=np.float32))
    np.testing.assert_allclose(widget._get_display_frame(2), binned[2] - binned[1], atol=1e-6)
    assert widget.data_min <= 0.0 <= widget.data_max
    widget.diff_mode = "off"
    assert (widget.data_min, widget.data_max) == (float(stack.min()), float(stack.max()))
    widget.free()


def test_panels_slice_the_concatenated_frame_and_keep_native_sources():
    left, right = _ramp(), _ramp(offset=10.0)
    widget = Show3D(left, right, panel_titles=["a", "b"], display_bin=4, verbose=False)
    assert (widget.n_panels, widget.panel_width_px, widget.width) == (2, 16, 32)
    np.testing.assert_allclose(widget._get_display_panel_frame(1, 2), right[2].reshape(16, 4, 16, 4).mean(axis=(1, 3)), atol=1e-5)
    np.testing.assert_array_equal(widget._get_source_panel_frame(1, 2), right[2])
    assert widget.link_contrast is False
    assert len(widget.auto_vmins_per_panel) == 2 and widget.auto_vmins_per_panel[1] > widget.auto_vmins_per_panel[0]
    widget.free()


def test_paged_input_visits_one_page_of_panels_at_a_time():
    pages = np.stack([_ramp(5, 16, offset=10.0 * page) for page in range(3)]).reshape(3, 1, 5, 16, 16).repeat(4, axis=1)
    widget = Show3D(pages, panel_titles=["raw", "f", "r", "p"], page_labels=["a", "b", "c"], max_cols=4, verbose=False)
    assert (widget.n_pages, widget.panels_per_page, widget.n_panels, widget.n_slices) == (3, 4, 12, 5)
    assert widget.panel_titles[:4] == ["raw", "f", "r", "p"] and widget._visible_panels == [0, 1, 2, 3]
    widget.page_idx = 2
    widget.star_page(2)
    assert widget._visible_panels == [8, 9, 10, 11] and widget.page_starred == [0, 0, 1]
    widget.free()


# --- playback stack -------------------------------------------------------------


def test_playback_stack_holds_every_frame_exactly_native_by_default(capsys):
    stack = _ramp()
    native = Show3D(stack, verbose=False)
    assert native.display_bin == 1 and capsys.readouterr().out == ""  # lossless default, nothing to announce
    assert native.smooth is False  # each pixel drawn as a sharp block; bilinear Smooth is opt-in
    np.testing.assert_array_equal(np.frombuffer(native._offline_float_stack, dtype=np.float32).reshape(stack.shape), stack)
    widget = Show3D(stack, display_bin=4, verbose=False)
    sent = np.frombuffer(widget._offline_float_stack, dtype=np.float32).reshape(6, 16, 16)
    np.testing.assert_array_equal(sent, stack.reshape(6, 16, 4, 16, 4).mean(axis=(2, 4)))
    assert widget._offline_stack == b""
    widget.free()
    native.free()



def test_a_stack_too_large_for_one_browser_message_names_the_bin_that_fits(monkeypatch, capsys):
    # The default stays native; a payload the browser cannot receive says which bin would fit.
    monkeypatch.setattr(state, "BROWSER_MESSAGE_LIMIT_BYTES", 10_000)
    widget = Show3D(_ramp(), verbose=False)
    assert widget.display_bin == 1
    assert "Pass display_bin=4 (0 MB) for an interactive view." in capsys.readouterr().out
    widget.free()


def test_set_image_repacks_the_stack_and_clamps_frame_state():
    widget = Show3D(_ramp(), display_bin=4, verbose=False)
    widget.loop_end = 5
    widget.bookmarked_frames = [1, 5]
    seq = widget.frame_seq
    replacement = _ramp(3, 32)
    widget.set_image(replacement, labels=["x", "y", "z"])
    assert (widget.n_slices, widget.height, widget.width, widget.labels) == (3, 8, 8, ["x", "y", "z"])
    np.testing.assert_array_equal(
        np.frombuffer(widget._offline_float_stack, dtype=np.float32).reshape(3, 8, 8),
        replacement.reshape(3, 8, 4, 8, 4).mean(axis=(2, 4)),
    )
    assert widget.frame_seq == seq + 1 and widget.loop_end == -1 and widget.bookmarked_frames == [1]
    widget.free()


def test_rgb_stack_packs_true_color_and_keeps_a_luminance_plane():
    rgb = np.zeros((5, 16, 20, 3), dtype=np.float32)
    rgb[..., 0] = np.linspace(0, 1, 5)[:, None, None]
    widget = Show3D(rgb, verbose=False)
    assert widget.is_rgb and widget.display_bin == 1 and widget._data.shape == (5, 16, 20)
    np.testing.assert_array_equal(np.frombuffer(widget._offline_float_stack, dtype=np.float32).reshape(rgb.shape), rgb)
    np.testing.assert_allclose(widget._data[4], np.full((16, 20), 0.2126, dtype=np.float32), atol=1e-6)
    widget.free()


def test_dataset3d_supplies_title_and_a_display_pixel_scale_bar():
    dataset = make_dataset(_ramp(), name="gold stack", sampling=(1.0, 0.1, 0.1), units=("frame", "nm", "nm"))
    widget = Show3D(dataset, display_bin=4, verbose=False)
    assert widget.title == "gold stack"
    # 0.1 nm per native pixel becomes 1 A, times the 4x display bin.
    assert widget.pixel_size == pytest.approx(4.0) and widget.pixel_unit == "A"
    explicit = Show3D(_ramp(), sampling=0.03, units="nm", display_bin=4, verbose=False)
    assert explicit.pixel_size == pytest.approx(0.12) and explicit.pixel_unit == "nm"
    widget.free()
    explicit.free()


# --- exports --------------------------------------------------------------------


def test_gif_renders_are_byte_stable(tmp_path: pathlib.Path):
    rng = np.random.default_rng(3)
    raw = rng.random((4, 32, 32), dtype=np.float32)
    widget = Show3D(raw, raw + 1.0, panel_titles=["raw", "denoised"], fps=8, verbose=False)
    first = widget.save_gif(tmp_path / "a.gif", quality="low").read_bytes()
    second = widget.save_gif(tmp_path / "b.gif", quality="low").read_bytes()
    assert first == second and first.startswith(b"GIF89a")
    assert Image.open(tmp_path / "a.gif").n_frames == 4
    bounce = Image.open(widget.save_gif(tmp_path / "c.gif", quality="low", playback="bounce"))
    assert bounce.n_frames == 6
    # the toolbar's GIF request renders the same frames as the notebook call
    widget.export_request = json.dumps({"id": "gif", "download": True, "mode": "gif", "quality": "low", "playback": "bounce"})
    assert widget.export_payload == (tmp_path / "c.gif").read_bytes()
    widget.free()


def test_save_image_colorizes_the_current_frame(tmp_path: pathlib.Path):
    widget = Show3D(_ramp(), cmap="gray", verbose=False)
    path = widget.save_image(tmp_path / "frame.png", frame_idx=5)
    image = np.asarray(Image.open(path).convert("RGB"))
    assert image.shape == (64, 64, 3) and image[-1, -1, 0] > image[0, 0, 0]
    widget.free()


def test_export_html_writes_a_standalone_page_with_the_display_stack(tmp_path: pathlib.Path):
    widget = Show3D(_ramp(6, 128), title="Protocol Show3D", sampling=0.2, units="nm", verbose=False)
    exact = widget.export_html(tmp_path / "exact.html")
    html = exact.read_text()
    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in html
    assert 'id="quantem-widget-export-layout"' in html and "application/vnd.jupyter.widget-state+json" in html
    assert "Protocol Show3D" in html and "_offline_float_stack" in html
    assert widget.export_status.startswith("Exported exact.html")
    compact = widget.export_html(tmp_path / "compact.html", encoding="uint8", downsample=4)
    assert compact.stat().st_size < exact.stat().st_size
    with pytest.raises(ValueError, match="downsample"):
        widget.export_html(tmp_path / "bad.html", encoding="full", downsample=2)
    widget.free()


def test_export_clone_carries_view_state_and_embeds_pixels():
    widget = Show3D(_ramp(6, 128), sampling=0.2, units="nm", display_bin=4, verbose=False)
    widget.playing = True
    widget.cmap = "inferno"
    clone = widget._clone_for_html_export(quantized=True, downsample=4)
    assert (clone.n_slices, clone.height, clone.width) == (6, 8, 8)
    assert clone.pixel_size == pytest.approx(widget.pixel_size * 4)
    assert clone.playing is True and clone.cmap == "inferno" and clone._save_state is True
    state = clone.get_state()
    assert len(state["_offline_stack"]) == 6 * 8 * 8 and state["_offline_float_stack"] == b""
    clone.free()
    widget.free()


def test_export_html_leaves_no_clone_model_open(tmp_path: pathlib.Path):
    # The page is written from an export-only clone; a clone left open keeps a comm in the
    # kernel and a model in the browser for every export.
    widget = Show3D(_ramp(3, 16), verbose=False)
    models = set(_instances)
    widget.export_html(tmp_path / "stack.html")
    assert not [model for key, model in _instances.items() if key not in models and isinstance(model, Show3D)]
    widget.close()


def test_toolbar_export_request_returns_a_download_payload():
    widget = Show3D(_ramp(4, 32), title="Toolbar", verbose=False)
    widget.export_request = json.dumps({"id": "req-1", "filename": "view.html", "download": True, "mode": "single", "encoding": "uint8"})
    assert widget.export_payload_id == "req-1" and widget.export_filename == "view.html"
    assert b"Toolbar" in widget.export_payload and widget.export_status.startswith("Ready view.html")
    widget.export_request = json.dumps({"id": "req-2", "filename": "view.gif", "download": True, "mode": "gif", "quality": "low", "max_frames": 2})
    assert widget.export_payload.startswith(b"GIF89a") and "2 frames" in widget.export_status
    widget.free()


# --- state ----------------------------------------------------------------------


def test_state_round_trips_through_dict_and_file(tmp_path: pathlib.Path):
    widget = Show3D(_ramp(), labels=[f"t{i}" for i in range(6)], verbose=False)
    widget.cmap = "viridis"
    widget.vmin, widget.vmax = 0.1, 0.9
    widget.loop_end, widget.loop_start = 4, 2
    widget.fps = 12
    widget.slice_idx = 4
    state = widget.state_dict()
    assert state["_widget"] == "Show3D" and list(state).index("loop_end") < list(state).index("loop_start")
    fresh = Show3D(_ramp(), verbose=False)
    fresh.load_state_dict(state)
    assert (fresh.cmap, fresh.vmin, fresh.vmax, fresh.loop_start, fresh.loop_end, fresh.fps, fresh.slice_idx) == ("viridis", 0.1, 0.9, 2, 4, 12.0, 4)
    assert fresh.labels == widget.labels
    widget.save(tmp_path / "state.json")
    payload = json.loads((tmp_path / "state.json").read_text())
    other = Show3D(_ramp(3, 32), verbose=False)
    other.load_state_dict(payload["state"])
    assert other.cmap == "viridis" and other.labels == ["0", "1", "2"] and other.loop_end == 2
    with pytest.raises(ValueError, match="Show3DSlices"):
        other.load_state_dict({"_widget": "Show3DSlices"})
    widget.free()
    fresh.free()
    other.free()


def test_saved_notebook_state_keeps_a_preview_instead_of_pixels():
    widget = Show3D(_ramp(), title="save", verbose=False)
    state = widget.get_state()
    assert "_offline_float_stack" not in state
    assert state["_static_fallback_mime"] == "image/jpeg" and len(state["_static_fallback_jpeg"]) > 1000
    mount = widget._repr_mimebundle_()
    assert "application/vnd.jupyter.widget-view+json" in (mount[0] if isinstance(mount, tuple) else mount)
    persistent = Show3D(_ramp(), save_state=True, verbose=False)
    assert "_offline_float_stack" in persistent.get_state()
    widget.free()
    persistent.free()


def test_static_preview_matches_a_show2d_gallery_pixel_for_pixel():
    rng = np.random.default_rng(18)
    stacks = [rng.random((4, 96, 112), dtype=np.float32) + panel * 0.1 for panel in range(4)]
    widget = Show3D(*stacks, panel_titles=[f"P{p + 1}" for p in range(4)], max_cols=2, panel_gap=3, panel_width_px=180, sampling=0.12, units="nm", verbose=False)
    widget.slice_idx = 2
    widget.hidden_panels = [1]
    reference = Show2D(
        [widget._get_display_panel_frame(panel, 2) for panel in (0, 2, 3)],
        labels=[widget._static_panel_title(panel, 2) for panel in (0, 2, 3)],
        ncols=2,
        inter_panel_gap_px=3,
        size=180,
        sampling=widget.pixel_size,
        units="nm",
        cmap=widget.cmap,
        auto_contrast=True,
        link_contrast=False,
        panel_inner_border_px=0.0,
        panel_inner_border_color="#000000",
        show_stats=False,
        show_controls=False,
        verbose=False,
        save_state=False,
    )
    np.testing.assert_array_equal(_decode_png(widget._static_png_b64(max_px=220)), _decode_png(reference._static_png_b64(max_px=220)))
    widget.free()


# --- folders --------------------------------------------------------------------


def _save(path: pathlib.Path, value: float) -> None:
    np.save(path, np.full((6, 8), value, dtype=np.float32))


def test_from_folder_appends_files_and_remaps_frame_state(tmp_path: pathlib.Path):
    _save(tmp_path / "frame_2.npy", 2)
    _save(tmp_path / "frame_10.npy", 10)
    widget = Show3D.from_folder(tmp_path, watch=False, show_fft=True)
    try:
        assert widget.labels == ["frame_2", "frame_10"] and widget.title == tmp_path.name
        widget.slice_idx = 1
        widget.bookmarked_frames = [0, 1]
        widget.loop_end = 1
        widget.starred = [1]
        widget.playing = True
        _save(tmp_path / "frame_1.npy", 1)
        assert widget.poll_folder() == []
        assert widget.poll_folder() == [0]
        assert widget.labels == ["frame_1", "frame_2", "frame_10"]
        np.testing.assert_array_equal(widget._data[:, 0, 0], [1, 2, 10])
        assert (widget.slice_idx, widget.bookmarked_frames, widget.loop_end, widget.starred) == (2, [1, 2], 2, [2])
        assert widget.playing is True and widget.show_fft is True
    finally:
        widget.close()


def test_watched_folder_is_created_and_fills_on_the_first_stable_frame(tmp_path: pathlib.Path):
    folder = tmp_path / "session" / "frames"
    widget = Show3D.from_folder(folder, file_types=".PNG", watch=True, watch_interval=60)
    try:
        assert folder.is_dir() and widget.n_slices == 0 and widget.folder_waiting is True
        widget.stop_folder_watch()
        Image.fromarray(np.arange(48, dtype=np.uint8).reshape(6, 8)).save(folder / "frame_001.png")
        _save(folder / "ignored.npy", 3)
        assert widget.poll_folder() == []
        assert widget.poll_folder() == [0]
        assert widget.n_slices == 1 and widget.labels == ["frame_001"]
    finally:
        widget.close()
