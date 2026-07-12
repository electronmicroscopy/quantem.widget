"""Show3D data-replacement invariants for kernel-less viewing surfaces."""

import numpy as np
import pytest

from quantem.widget import Show3D

_RGB_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _color_stack(n: int, h: int = 8, w: int = 8) -> np.ndarray:
    """A stack whose frame k is a unique flat color: red fades out, blue in."""
    src = np.zeros((n, h, w, 3), dtype=np.float32)
    for k in range(n):
        t = k / max(1, n - 1)
        src[k] = [1.0 - t, 0.4, t]
    return src


def test_set_image_repacks_offline_stack():
    # Docs pages, exported HTML, and saved widget state slice frames straight
    # out of _offline_stack. set_image() must repack it, or any appended frame
    # index points past the old stack's end and renders blank on reopen.
    widget = Show3D(np.random.rand(4, 32, 32).astype("float32"))
    assert widget.offline
    assert len(widget._offline_stack) == 4 * 32 * 32

    widget.set_image(np.random.rand(8, 32, 32).astype("float32"))

    assert widget.n_slices == 8
    assert len(widget._offline_stack) == 8 * 32 * 32
    lo, hi = widget._offline_min, widget._offline_max
    assert hi > lo


def test_rgb_offline_stack_frames_decode_at_rgb_stride():
    # RGB stacks pack 3 bytes/px; frame k must live at k*3*H*W. A grayscale
    # stride here scrambles every frame after the first on kernel-less reopen
    # (the fig4 RGB export bug). This is the uint8-path parity lock.
    src = _color_stack(6, 12, 12)
    widget = Show3D(src)
    assert widget.is_rgb and widget.offline
    stack = np.frombuffer(widget._offline_stack, dtype=np.uint8).reshape(6, 12, 12, 3)
    for k in range(6):
        expected = np.clip(src[k] * 255.0, 0, 255).astype(np.uint8)
        assert np.array_equal(stack[k], expected), f"RGB frame {k} scrambled"


def test_set_image_gray_rgb_swap_keeps_color_content_and_luminance():
    # A scientist iterates a color figure in place: gray stack, then a color
    # composite, then back. set_image must reconfigure the whole contract each
    # time - color into _rgb_data (what the browser paints), a luminance plane
    # into _data (what stats/FFT/ROI read) - not leave a stale or 4-D _data.
    widget = Show3D(np.random.rand(4, 8, 8).astype("float32"))
    assert not widget.is_rgb and widget._rgb_data is None

    rgb = _color_stack(3)
    widget.set_image(rgb)
    assert widget.is_rgb and widget._rgb_data is not None
    assert widget._rgb_data.shape == (3, 8, 8, 3)
    assert widget._data.shape == (3, 8, 8)  # luminance, not 4-D color
    np.testing.assert_allclose(widget._rgb_data, rgb, atol=1e-6)
    np.testing.assert_allclose(widget._data, rgb @ _RGB_LUMA, atol=1e-5)
    # the offline stack the browser slices carries the color, not the luminance
    stack = np.frombuffer(widget._offline_stack, dtype=np.uint8).reshape(3, 8, 8, 3)
    np.testing.assert_allclose(stack[0] / 255.0, rgb[0], atol=1.5 / 255)

    widget.set_image(np.random.rand(5, 8, 8).astype("float32"))
    assert not widget.is_rgb and widget._rgb_data is None
    assert len(widget._offline_stack) == 5 * 8 * 8  # gray stride, color cleared


def test_rgb_export_source_and_raster_stay_true_color_after_set_image():
    # Exporting an RGB figure the scientist just updated in place: the HTML
    # export source, the offline stack, and a saved PNG must all reflect the
    # NEW color frames, never a stale stack or a colormapped luminance plane.
    widget = Show3D(_color_stack(6))
    grown = _color_stack(10)
    widget.set_image(grown)

    export_source = widget._offline_pack_source()
    assert export_source.shape == (10, 8, 8, 3)
    np.testing.assert_allclose(export_source[9], grown[9], atol=1e-6)

    preview = widget._static_show2d_preview(idx=5)
    assert preview.is_rgb == [True]  # cold-reopen preview is true color


def test_from_rgb_forces_color_the_heuristic_cannot_call():
    # A scientist has a short 2-frame color composite. The (N > 4) auto-detect
    # heuristic cannot tell it from a scalar stack, so from_rgb is the explicit
    # color path; rgb=False likewise forces gray on an ambiguous shape.
    two_frame = _color_stack(2)
    widget = Show3D.from_rgb(two_frame)
    assert widget.is_rgb
    assert widget._rgb_data.shape == (2, 8, 8, 3)
    assert widget._data.shape == (2, 8, 8)  # luminance plane for stats

    ambiguous_gray = np.random.rand(3, 8, 3).astype("float32")  # trailing 3
    gray = Show3D(ambiguous_gray, rgb=False)
    assert not gray.is_rgb and gray._data.shape == (3, 8, 3)


def test_rgb_config_rotation_keeps_true_color():
    # A scientist loads a color figure with a recon rotation. Rotation used to
    # be blocked for RGB; now each channel rotates together, so the figure keeps
    # its true color and a re-derived luminance plane drives stats.
    rgb = np.zeros((3, 20, 20, 3), dtype=np.float32)
    rgb[:, 5:15, 9:11, 0] = 1.0  # a red vertical bar
    widget = Show3D(rgb, rotation_deg=90, apply_config_transforms=True)
    assert widget.is_rgb
    assert widget._rgb_data.shape == (3, 20, 20, 3)
    assert widget._data.shape == (3, 20, 20)  # luminance, re-derived after rotation
    assert widget._rgb_data[0][..., 0].max() > 0.9  # red survived the rotation


def test_from_figure_gallery_labels_and_unifies_color():
    # A scientist collects candidate figures to pick the best. A dict keeps the
    # names; a mix of grayscale and color is unified to one true-color viewer.
    sweep = {
        "lambda 0.3": np.random.rand(16, 16).astype("float32"),
        "lambda 3": np.random.rand(16, 16).astype("float32"),
    }
    gallery = Show3D.from_figure_gallery(sweep, title="sweep")
    assert gallery.n_slices == 2
    assert list(gallery.labels) == ["lambda 0.3", "lambda 3"]

    gray_frame = np.random.rand(16, 16).astype("float32")
    color_frame = np.random.rand(16, 16, 3).astype("float32")
    mixed = Show3D.from_figure_gallery([gray_frame, color_frame], labels=["gray", "color"])
    assert mixed.is_rgb and mixed.n_slices == 2


def test_playback_dynamics_state_round_trips():
    # A time-series reviewer can use playback dynamics like microscope temporal
    # lenses: slow cadence, focus range, bounce, and held key frames.  These are
    # lightweight state, not resampled data.
    stack = np.random.default_rng(8).random((12, 16, 16), dtype=np.float32)
    widget = Show3D(stack, fps=6, verbose=False)
    widget.loop = True
    widget.boomerang = True
    widget.loop_start = 2
    widget.loop_end = 9
    widget.playback_path = [2, 3, 4, 4, 4, 5, 6, 7, 8, 9, 8, 7]
    widget.slice_idx = 4

    restored = Show3D(stack, verbose=False)
    restored.load_state_dict(widget.state_dict())

    assert restored.fps == pytest.approx(6.0)
    assert restored.loop is True
    assert restored.boomerang is True
    assert restored.loop_start == 2
    assert restored.loop_end == 9
    assert restored.playback_path == [2, 3, 4, 4, 4, 5, 6, 7, 8, 9, 8, 7]
    assert restored.slice_idx == 4
    assert restored._data.shape == stack.shape


def test_export_html_refuses_oversized_single_file(tmp_path):
    # A scientist exports a big float32 color movie. A single HTML that large
    # fails to open under Chrome file://, so export_html refuses and names the
    # smaller-encoding options instead of writing a doomed file.
    widget = Show3D(np.random.rand(20, 128, 128, 3).astype("float32"), verbose=False)
    with pytest.raises(ValueError, match="safe limit"):
        widget.export_html(tmp_path / "nope.html", max_mb=5)
    # the smaller uint8 encoding gets under the same limit and writes fine
    out = widget.export_html(tmp_path / "ok.html", encoding="uint8", max_mb=5)
    assert out.exists()


def test_multi_panel_rgb_packs_independent_color_panels():
    # A scientist compares two color results side by side (e.g. two channel
    # composites). Each panel is its own (N, H, W, 3) stack; Show3D concatenates
    # them into one wide color frame and the offline stack keeps each panel's
    # true color, sliced by panel_width_px.
    red = np.zeros((5, 12, 16, 3), dtype=np.float32)
    red[..., 0] = 1.0
    blue = np.zeros((5, 12, 16, 3), dtype=np.float32)
    blue[..., 2] = 1.0
    widget = Show3D(red, blue, panel_titles=["red", "blue"])
    assert widget.is_rgb and widget.n_panels == 2
    assert widget.panel_width_px == 16
    assert widget._rgb_data.shape == (5, 12, 32, 3)
    assert widget._data.shape == (5, 12, 32)  # luminance concat drives stats
    assert not widget.separate_panel_frames

    stack = np.frombuffer(widget._offline_stack, dtype=np.uint8).reshape(5, 12, 32, 3)
    assert (stack[0, :, :16, 0] == 255).all()   # left panel red
    assert (stack[0, :, 16:, 2] == 255).all()   # right panel blue


def test_multi_panel_rgb_export_clone_keeps_panels():
    # The HTML export clone must rebuild the same panel count, not collapse the
    # wide color concat into one panel.
    red = np.zeros((4, 16, 16, 3), dtype=np.float32)
    red[..., 0] = 1.0
    green = np.zeros((4, 16, 16, 3), dtype=np.float32)
    green[..., 1] = 1.0
    widget = Show3D(red, green)
    clone = widget._clone_for_html_export(quantized=True)
    try:
        assert clone.n_panels == 2 and clone.panel_width_px == 16 and clone.is_rgb
        cs = np.frombuffer(clone._offline_stack, dtype=np.uint8).reshape(4, 16, 32, 3)
        assert (cs[0, :, :16, 0] == 255).all() and (cs[0, :, 16:, 1] == 255).all()
    finally:
        clone.close()
