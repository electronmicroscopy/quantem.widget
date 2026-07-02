"""RGB panel support in Show2D: detection, packing, static PNG, guards.

RGB panels carry display-ready (H, W, 3) pixels that bypass the colormap and
contrast pipeline. The drift-correction green-magenta alignment overlays are
the driving use case: a mixed gallery of grayscale merges + RGB overlays.
"""

import io

import numpy as np
import pytest
from PIL import Image

from quantem.widget import Show2D


def _gray(h=32, w=40, seed=0):
    return np.random.default_rng(seed).random((h, w)).astype(np.float32) * 100.0


def _green_magenta(h=32, w=40, seed=1):
    """Synthetic overlay_pair-style composite: R == B (reference), G distinct."""
    rng = np.random.default_rng(seed)
    ref = rng.random((h, w)).astype(np.float32)
    mov = rng.random((h, w)).astype(np.float32)
    return np.stack([ref, mov, ref], axis=-1)


# --- input detection -----------------------------------------------------

def test_list_item_with_trailing_3_is_rgb():
    w = Show2D([_gray(), _green_magenta()], verbose=False)
    assert list(w.is_rgb) == [False, True]
    assert w.n_images == 2
    assert w._rgb_frames[0] is None
    assert w._rgb_frames[1].shape == (32, 40, 3)


def test_bare_3d_array_keeps_stack_semantics():
    # (3, H, W) stack whose leading dim is 3 must NOT flip to RGB
    w = Show2D(np.random.rand(3, 16, 16).astype(np.float32), verbose=False)
    assert w.n_images == 3
    assert list(w.is_rgb) == [False, False, False]


def test_bare_hw3_array_with_big_leading_dim_is_single_rgb():
    w = Show2D(_green_magenta(h=24, w=20), verbose=False)
    assert w.n_images == 1
    assert list(w.is_rgb) == [True]
    assert w.height == 24 and w.width == 20


def test_ambiguous_small_leading_dim_stays_stack():
    # (3, H, 3): leading dim <= 4 -> stack of three (H, 3) images, not RGB
    w = Show2D(np.random.rand(3, 16, 3).astype(np.float32), verbose=False)
    assert w.n_images == 3
    assert list(w.is_rgb) == [False, False, False]


def test_rgba_drops_alpha():
    rgba = np.concatenate([_green_magenta(), np.ones((32, 40, 1), dtype=np.float32)], axis=-1)
    w = Show2D([rgba], verbose=False)
    assert list(w.is_rgb) == [True]
    assert w._rgb_frames[0].shape == (32, 40, 3)


def test_uint8_rgb_scales_to_unit_range():
    rgb_u8 = (np.clip(_green_magenta(), 0, 1) * 255).astype(np.uint8)
    w = Show2D([rgb_u8], verbose=False)
    frame = w._rgb_frames[0]
    assert frame.dtype == np.float32
    assert 0.0 <= frame.min() and frame.max() <= 1.0
    np.testing.assert_allclose(frame, rgb_u8.astype(np.float32) / 255.0, atol=1e-6)


def test_float_rgb_clipped_to_unit_range():
    hot = _green_magenta() * 2.0 - 0.5
    w = Show2D([hot], verbose=False)
    assert w._rgb_frames[0].min() >= 0.0
    assert w._rgb_frames[0].max() <= 1.0


# --- frame packing + luminance -------------------------------------------

def test_mixed_gallery_frame_packing_layout():
    gray, overlay = _gray(), _green_magenta()
    w = Show2D([gray, overlay], verbose=False)
    per = 32 * 40
    expected_bytes = 4 * (per + 3 * per)
    # frame_bytes is padded to a multiple of 3 for base64 embedding (_b64_safe)
    assert expected_bytes <= len(w.frame_bytes) <= expected_bytes + 2
    floats = np.frombuffer(w.frame_bytes[:expected_bytes], dtype=np.float32)
    np.testing.assert_array_equal(floats[:per].reshape(32, 40), gray)
    packed_rgb = floats[per:].reshape(32, 40, 3)
    np.testing.assert_allclose(packed_rgb, np.clip(overlay, 0, 1), atol=1e-6)


def test_rgb_panel_luminance_plane_and_stats():
    overlay = _green_magenta()
    w = Show2D([_gray(), overlay], verbose=False)
    luma = np.clip(overlay, 0, 1) @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    np.testing.assert_allclose(w._data[1], luma, atol=1e-5)
    # stats row shows luminance stats for the RGB panel
    assert w.stats_mean[1] == pytest.approx(float(luma.mean()), abs=1e-4)
    assert w.stats_max[1] == pytest.approx(float(luma.max()), abs=1e-4)


def test_mixed_shapes_center_pad_rgb_and_gray():
    w = Show2D([_gray(h=32, w=40), _green_magenta(h=20, w=24)], verbose=False)
    assert w.height == 32 and w.width == 40
    assert w._rgb_frames[1].shape == (32, 40, 3)


# --- static PNG fallback ---------------------------------------------------

def _decode_static_png(widget):
    import base64
    png_b64 = widget._static_png_b64()
    assert png_b64
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(png_b64))).convert("RGB"), dtype=np.float32)


def test_static_png_preserves_green_magenta():
    # green-magenta overlay must stay channel-distinct through the PNG path:
    # a colormapped (corrupted) render would give it a LUT palette instead.
    overlay = _green_magenta(h=64, w=64)
    w = Show2D([overlay], verbose=False, scale_bar_visible=False, show_panel_titles=False)
    pixels = _decode_static_png(w)
    r, g, b = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    # reference is magenta (R == B): channels match closely across the panel
    assert float(np.abs(r - b).mean()) < 2.0
    # moving image is green: G is genuinely distinct from R
    assert float(np.abs(r - g).mean()) > 10.0


def test_static_png_mixed_gallery_renders_all_panels():
    w = Show2D([_gray(seed=3), _green_magenta(), _gray(seed=4)], verbose=False, ncols=3)
    specs = w._static_panel_specs()
    assert [s.get("rgb", False) for s in specs] == [False, True, False]
    pixels = _decode_static_png(w)
    assert pixels.ndim == 3 and pixels.shape[2] == 3


def test_rgb_panel_excluded_from_linked_contrast():
    gray_a = _gray(seed=5) + 1000.0  # counts-scaled panels
    gray_b = _gray(seed=6) + 1000.0
    w = Show2D([gray_a, gray_b, _green_magenta()], verbose=False, link_contrast=True)
    ranges = w._resolve_panel_display_ranges([w._data[i] for i in range(3)])
    # RGB panel pinned to [0, 1]; grayscale shared range unaffected by the overlay
    assert ranges[2] == (0.0, 1.0)
    assert ranges[0] == ranges[1]
    assert ranges[0][0] >= 1000.0


# --- save_image -------------------------------------------------------------

def test_save_image_rgb_bypasses_colormap(tmp_path):
    overlay = _green_magenta(h=48, w=48)
    w = Show2D([overlay], verbose=False)
    path = w.save_image(tmp_path / "overlay.png", idx=0)
    pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    np.testing.assert_array_equal(pixels[..., 0], pixels[..., 2])  # magenta: R == B exactly
    assert not np.allclose(pixels[..., 0], pixels[..., 1])


# --- overlay sugar -----------------------------------------------------------

def test_overlay_sugar_appends_green_magenta_panel():
    a, b = _gray(seed=10), _gray(seed=11)
    w = Show2D([a, b], overlay=True, verbose=False)
    assert w.n_images == 3
    assert list(w.is_rgb) == [False, False, True]
    assert w.labels == ["Image 1", "Image 2", "overlay (green-magenta)"]
    composed = w._rgb_frames[2]
    # green-magenta structure: R == B (reference), G is the moving image
    np.testing.assert_array_equal(composed[..., 0], composed[..., 2])
    assert not np.allclose(composed[..., 0], composed[..., 1])
    # aligned pixels read white: identical inputs give R == G == B everywhere
    aligned = Show2D([a, a], overlay=True, verbose=False)._rgb_frames[2]
    np.testing.assert_array_equal(aligned[..., 0], aligned[..., 1])
    np.testing.assert_array_equal(aligned[..., 1], aligned[..., 2])


def test_overlay_sugar_shared_percentile_scale():
    # parity with quantem drift overlay_pair: ONE 1/99 scale across BOTH images
    a, b = _gray(seed=12), _gray(seed=13) * 10.0
    composed = Show2D([a, b], overlay=True, verbose=False)._rgb_frames[2]
    lo = min(float(np.percentile(a, 1)), float(np.percentile(b, 1)))
    hi = max(float(np.percentile(a, 99)), float(np.percentile(b, 99)))
    expected_ref = np.clip((a - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    np.testing.assert_allclose(composed[..., 0], expected_ref, atol=1e-6)


def test_overlay_rgb_mode_red_green():
    w = Show2D([_gray(seed=14), _gray(seed=15)], overlay="rgb", verbose=False)
    assert w.labels[-1] == "overlay (rgb)"
    composed = w._rgb_frames[2]
    np.testing.assert_array_equal(composed[..., 2], 0.0)  # blue channel empty


def test_overlay_with_diff_mode_keeps_pair_diff_only():
    a, b = _gray(seed=16), _gray(seed=17)
    w = Show2D([a, b], overlay=True, diff_mode=True, verbose=False)
    assert w.diff_mode is True
    specs = w._static_panel_specs()
    # [a, b, overlay] image panels + ONE diff panel (a - b); the RGB overlay
    # panel is never diffed
    assert len(specs) == 4
    assert [s.get("rgb", False) for s in specs] == [False, False, True, False]
    assert specs[3]["cmap"] == "RdBu"


def test_overlay_labels_extend_user_labels():
    w = Show2D([_gray(), _gray(seed=2)], labels=["ref", "mov"], overlay=True, verbose=False)
    assert w.labels == ["ref", "mov", "overlay (green-magenta)"]


def test_overlay_requires_exactly_two_grayscale_images():
    with pytest.raises(ValueError, match="exactly 2"):
        Show2D([_gray(), _gray(seed=1), _gray(seed=2)], overlay=True, verbose=False)
    with pytest.raises(ValueError, match="exactly 2"):
        Show2D([_gray(), _green_magenta()], overlay=True, verbose=False)
    with pytest.raises(ValueError, match="overlay must be"):
        Show2D([_gray(), _gray(seed=1)], overlay="sepia", verbose=False)


# --- guards ------------------------------------------------------------------

def test_offline_with_rgb_raises():
    with pytest.raises(NotImplementedError, match="offline"):
        Show2D([_green_magenta()], offline=True, verbose=False)


def test_export_html_with_rgb_raises():
    w = Show2D([_gray(), _green_magenta()], verbose=False)
    assert w.export_enabled is False  # export menu hidden in the frontend
    with pytest.raises(NotImplementedError, match="RGB"):
        w.export_html("/tmp/should_not_exist.html")


def test_rotate_with_rgb_raises():
    w = Show2D([_gray(), _green_magenta()], verbose=False)
    with pytest.raises(NotImplementedError, match="RGB"):
        w.rotate(0, 90)


def test_grayscale_only_path_unchanged():
    # no-RGB galleries keep the legacy packing (one (N, H, W) float32 block)
    stack = np.random.rand(2, 16, 16).astype(np.float32)
    w = Show2D(stack, verbose=False)
    assert list(w.is_rgb) == [False, False]
    expected_bytes = 4 * 2 * 16 * 16
    floats = np.frombuffer(w.frame_bytes[:expected_bytes], dtype=np.float32)
    np.testing.assert_array_equal(floats.reshape(2, 16, 16), stack)
