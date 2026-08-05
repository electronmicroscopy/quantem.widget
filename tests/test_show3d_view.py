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


def test_subpixel_alignment_state_round_trips():
    # A scientist can turn on browser-side drift alignment from the API, save
    # the notebook/widget state, and reopen with the same reference choice.
    # The raw stack shape stays unchanged; alignment is a display transform.
    stack = np.random.default_rng(166).random((6, 16, 16), dtype=np.float32)
    widget = Show3D(
        stack,
        subpixel_align=True,
        subpixel_align_reference=3,
        verbose=False,
    )

    restored = Show3D(stack, verbose=False)
    restored.load_state_dict(widget.state_dict())

    assert widget.subpixel_align_enabled is True
    assert widget.subpixel_align_reference == 3
    assert restored.subpixel_align_enabled is True
    assert restored.subpixel_align_reference == 3
    assert restored._data.shape == stack.shape


def test_show3d_rich_panel_titles_keep_plain_titles_and_state():
    """C1: Show3D panel title spans preserve plain title fallbacks."""
    stack_a = np.random.default_rng(13).random((3, 12, 12), dtype=np.float32)
    stack_b = np.random.default_rng(14).random((3, 12, 12), dtype=np.float32)
    widget = Show3D(
        stack_a,
        stack_b,
        panel_titles=[
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

    assert widget.panel_titles == ["BF denoise  low  χ²=0.5", "BF denoise mid"]
    assert widget.panel_title_spans[0][1] == {"text": "low", "color": "#60a5fa"}

    restored = Show3D(stack_a, stack_b, verbose=False)
    restored.load_state_dict(widget.state_dict())
    assert restored.panel_titles == widget.panel_titles
    assert restored.panel_title_spans == widget.panel_title_spans


def test_show3d_single_panel_title_spans_are_one_title():
    """C2: a single list of span dictionaries represents one rich title."""
    stack = np.random.default_rng(15).random((2, 10, 10), dtype=np.float32)
    widget = Show3D(
        stack,
        panel_title_spans=[
            {"text": "raw vs "},
            {"text": "denoise", "color": "#34d399"},
        ],
        verbose=False,
    )

    assert widget.panel_titles == ["raw vs denoise"]
    assert len(widget.panel_title_spans) == 1
    assert widget.panel_title_spans[0][1] == {"text": "denoise", "color": "#34d399"}


def test_show3d_panel_title_style_and_group_markers_round_trip():
    """C1: title chrome and group markers are portable Show3D state."""
    rng = np.random.default_rng(23)
    panels = [rng.random((3, 10, 10), dtype=np.float32) for _ in range(4)]
    widget = Show3D(
        *panels,
        panel_titles=["A", "B", "C", "D"],
        max_cols=2,
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
        panel_groups=[
            {"panels": [0, 1], "color": "#22c55e", "label": "raw"},
            {"start": 2, "end": 3, "color": "#d946ef", "label": "denoised"},
        ],
        verbose=False,
    )

    restored = Show3D(*panels, verbose=False)
    restored.load_state_dict(widget.state_dict())

    assert restored.panel_title_style == widget.panel_title_style
    assert restored.row_markers == {"0": "#60a5fa"}
    assert restored.col_markers == {"1": "#f59e0b"}
    assert restored.panel_groups == [
        {"panels": [0, 1], "color": "#22c55e", "label": "raw"},
        {"panels": [2, 3], "color": "#d946ef", "label": "denoised"},
    ]


def test_show3d_gallery_chrome_round_trip_and_panel_gap_alias():
    """C1: Show3D exposes explicit gallery gap/frame/border controls."""
    rng = np.random.default_rng(182)
    panels = [rng.random((3, 10, 10), dtype=np.float32) for _ in range(2)]
    widget = Show3D(
        *panels,
        panel_gap=5,
        inter_panel_gap_color="#111111",
        gallery_outer_border_px=3,
        gallery_outer_border_color="#000000",
        panel_inner_border_px=2,
        panel_inner_border_color="#ff00ff",
        verbose=False,
    )

    restored = Show3D(*panels, verbose=False)
    restored.load_state_dict(widget.state_dict())

    assert widget.inter_panel_gap_px == 5
    assert widget.panel_gap == 5
    assert restored.inter_panel_gap_px == 5
    assert restored.panel_gap == 5
    assert restored.inter_panel_gap_color == "#111111"
    assert restored.gallery_outer_border_px == 3
    assert restored.gallery_outer_border_color == "#000000"
    assert restored.panel_inner_border_px == pytest.approx(2)
    assert restored.panel_inner_border_color == "#ff00ff"


def test_show3d_panel_annotations_accept_flat_panel_targets():
    """C1: multi-panel stack annotations can target the same panel repeatedly."""
    rng = np.random.default_rng(25)
    panels = [rng.random((3, 10, 10), dtype=np.float32) for _ in range(3)]
    widget = Show3D(
        *panels,
        panel_titles=["raw", "filtered", "residual"],
        panel_annotations=[
            {"panel": "raw", "text": "input", "position": "top-left"},
            {"panel": 0, "text": "same panel", "position": "bottom-left", "variant": "outline"},
            {
                "panel": 2,
                "spans": [{"text": "χ² ", "color": "#fff"}, {"text": "high", "color": "#f87171"}],
                "x": 0.5,
                "y": 0.2,
                "anchor": "top-center",
                "class": "chi2-status",
            },
        ],
        verbose=False,
    )

    restored = Show3D(*panels, panel_titles=["raw", "filtered", "residual"], verbose=False)
    restored.load_state_dict(widget.state_dict())

    assert len(restored.panel_annotations[0]) == 2
    assert restored.panel_annotations[0][1]["variant"] == "outline"
    assert restored.panel_annotations[2][0]["text"] == "χ² high"
    assert restored.panel_annotations[2][0]["class_name"] == "chi2-status"


def test_show3d_titles_and_annotations_accept_math_spans():
    """C1: Show3D preserves math spans for browser-side label rendering."""
    rng = np.random.default_rng(27)
    panels = [rng.random((3, 10, 10), dtype=np.float32) for _ in range(2)]
    widget = Show3D(
        *panels,
        panel_titles=[
            [{"math": r"\lambda=0.01"}, {"text": " object"}],
            r"$\chi^2$/pixel",
        ],
        panel_annotations=[
            {"panel": 0, "math": r"\lambda", "position": "top-left"},
            {
                "panel": 1,
                "spans": [{"math": r"\chi^2"}, {"text": "/pixel"}],
                "position": "top-right",
            },
        ],
        verbose=False,
    )

    restored = Show3D(*panels, verbose=False)
    restored.load_state_dict(widget.state_dict())

    assert restored.panel_title_spans[0][0] == {"math": r"\lambda=0.01"}
    assert restored.panel_titles[1] == r"$\chi^2$/pixel"
    assert restored.panel_annotations[0][0]["math"] == r"\lambda"
    assert restored.panel_annotations[1][0]["spans"][0] == {"math": r"\chi^2"}


def test_show3d_panel_overlays_target_panel_titles_and_roundtrip():
    """C1: Show3D accepts the same per-panel overlay API as Show2D."""
    rng = np.random.default_rng(29)
    panels = [rng.random((3, 10, 10), dtype=np.float32) for _ in range(2)]
    circle = {
        "shape": "circle",
        "center": (4, 5),
        "radius": 2,
        "stroke": "#60a5fa",
        "stroke_style": "dotted",
    }
    rect = {
        "shape": "square",
        "center": (5, 5),
        "size": 4,
        "stroke": "#f87171",
        "fill": "#f87171",
        "fill_opacity": 0.15,
        "stroke_opacity": 0.9,
        "line_style": "dashdot",
        "z_order": 3,
    }

    widget = Show3D(
        *panels,
        panel_titles=["raw", "denoised"],
        panel_overlays={
            "raw": [circle],
            "denoised": rect,
        },
        verbose=False,
    )
    restored = Show3D(*panels, panel_titles=["raw", "denoised"], verbose=False)
    restored.load_state_dict(widget.state_dict())

    assert [len(items) for items in restored.panel_overlays] == [1, 1]
    assert restored.panel_overlays[0][0]["shape"] == "circle"
    assert restored.panel_overlays[0][0]["line_style"] == "dotted"
    assert restored.panel_overlays[1][0]["shape"] == "square"
    assert restored.panel_overlays[1][0]["row0"] == 3.0
    assert restored.panel_overlays[1][0]["fill_opacity"] == 0.15
    assert restored.panel_overlays[1][0]["line_style"] == "dashdot"


def test_show3d_panel_groups_validate_panel_indices():
    """C1: invalid rectangular panel groups fail before export."""
    rng = np.random.default_rng(24)
    panels = [rng.random((3, 10, 10), dtype=np.float32) for _ in range(2)]

    with pytest.raises(ValueError, match="outside the Show3D panel range"):
        Show3D(
            *panels,
            panel_groups=[{"panels": [0, 2], "color": "#22c55e"}],
            verbose=False,
        )


def test_show3d_playback_accepts_sixty_fps():
    # C1: a remote movie reviewer requests 60 fps playback, expect Show3D to
    # preserve that target instead of silently clamping to the old 30 fps cap.
    stack = np.random.default_rng(60).random((3, 8, 8), dtype=np.float32)
    widget = Show3D(stack, fps=60, verbose=False)
    too_high = Show3D(stack, fps=120, verbose=False)

    assert widget.fps == pytest.approx(60.0)
    assert too_high.fps == pytest.approx(60.0)


def test_show3d_moving_average_defaults_on_but_can_be_disabled():
    # C1: a microscopist reviewing a noisy time stack gets temporal averaging
    # immediately, while raw single-frame review remains an explicit option.
    stack = np.random.default_rng(63).random((5, 8, 8), dtype=np.float32)
    widget = Show3D(stack, verbose=False)
    raw = Show3D(stack, avg_window=1, verbose=False)

    assert widget.avg_window == 3
    assert widget.state_dict()["avg_window"] == 3
    assert raw.avg_window == 1


def test_show3d_compare_markers_histogram_and_flip_state_round_trip():
    # C1: point-defect comparison controls are public state, not only a
    # transient frontend menu. The raw stack remains unchanged.
    stack = np.random.default_rng(171).random((5, 12, 12), dtype=np.float32)
    widget = Show3D(
        stack,
        compare_mode="blink",
        compare_pair=(0, 4),
        blink_fps=4,
        diff_cmap="magenta-green",
        compare_background="dark",
        marker_colors=["green"],
        marker_style="around",
        rotation=90,
        rotations=[0, 90, 180, 270, 0],
        rotation_scope="frame",
        contrast_preset="1-99",
        show_histogram_advanced=True,
        flip_horizontal=True,
        flip_vertical=True,
        verbose=False,
    )

    restored = Show3D(stack, verbose=False)
    restored.load_state_dict(widget.state_dict())

    assert restored.compare_mode == "blink"
    assert restored.compare_pair == [0, 4]
    assert restored.blink_fps == pytest.approx(4)
    assert restored.diff_cmap == "magenta-green"
    assert restored.compare_background == "dark"
    assert restored.marker_colors == ["green"]
    assert restored.marker_style == "around"
    assert restored.image_rotation == 1
    assert restored.rotation_scope == "frame"
    assert restored.frame_rotations == [0, 1, 2, 3, 0]
    assert restored.contrast_preset == "1-99"
    assert restored.show_histogram_advanced is True
    assert restored.flip_horizontal is True
    assert restored.flip_vertical is True
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


def test_show3d_rich_panel_titles_keep_plain_titles_and_state():
    """C1: Show3D panel title spans preserve plain title fallbacks."""
    stack_a = np.random.default_rng(13).random((3, 12, 12), dtype=np.float32)
    stack_b = np.random.default_rng(14).random((3, 12, 12), dtype=np.float32)
    widget = Show3D(
        stack_a,
        stack_b,
        panel_titles=[
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

    assert widget.panel_titles == ["BF denoise  low  χ²=0.5", "BF denoise mid"]
    assert widget.panel_title_spans[0][1] == {"text": "low", "color": "#60a5fa"}

    restored = Show3D(stack_a, stack_b, verbose=False)
    restored.load_state_dict(widget.state_dict())
    assert restored.panel_titles == widget.panel_titles
    assert restored.panel_title_spans == widget.panel_title_spans


def test_show3d_single_panel_title_spans_are_one_title():
    """C2: a single list of span dictionaries represents one rich title."""
    stack = np.random.default_rng(15).random((2, 10, 10), dtype=np.float32)
    widget = Show3D(
        stack,
        panel_title_spans=[
            {"text": "raw vs "},
            {"text": "denoise", "color": "#34d399"},
        ],
        verbose=False,
    )

    assert widget.panel_titles == ["raw vs denoise"]
    assert len(widget.panel_title_spans) == 1
    assert widget.panel_title_spans[0][1] == {"text": "denoise", "color": "#34d399"}
