import json
import numpy as np
import pytest
import torch
import traitlets

from quantem.widget import Show3D

try:
    import h5py  # type: ignore

    _HAS_H5PY = True
except Exception:
    h5py = None
    _HAS_H5PY = False

def test_show3d_numpy():
    """Create widget from numpy array."""
    data = np.random.rand(10, 32, 32).astype(np.float32)
    widget = Show3D(data)
    assert widget.n_slices == 10
    assert widget.height == 32
    assert widget.width == 32
    assert len(widget.frame_bytes) > 0

def test_show3d_torch():
    """Create widget from PyTorch tensor."""
    data = torch.rand(10, 32, 32)
    widget = Show3D(data)
    assert widget.n_slices == 10
    assert widget.height == 32
    assert widget.width == 32

def test_show3d_initial_slice():
    """Initial slice is at middle."""
    data = np.random.rand(20, 16, 16).astype(np.float32)
    widget = Show3D(data)
    assert widget.slice_idx == 10

def test_show3d_stats():
    """Statistics are computed for current slice."""
    data = np.zeros((5, 16, 16), dtype=np.float32)
    data[2, :, :] = 42.0
    widget = Show3D(data)
    widget.slice_idx = 2
    assert widget.stats_mean == pytest.approx(42.0)
    assert widget.stats_std == pytest.approx(0.0)

def test_show3d_labels():
    """Labels are set correctly."""
    data = np.random.rand(3, 16, 16).astype(np.float32)
    labels = ["Frame A", "Frame B", "Frame C"]
    widget = Show3D(data, labels=labels)
    assert widget.labels == labels

def test_show3d_default_labels():
    """Default labels are indices."""
    data = np.random.rand(3, 16, 16).astype(np.float32)
    widget = Show3D(data)
    assert widget.labels == ["0", "1", "2"]

def test_show3d_playback():
    """Playback methods work."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.play()
    assert widget.playing is True
    widget.pause()
    assert widget.playing is False
    widget.stop()
    assert widget.playing is False
    assert widget.slice_idx == 0

def test_show3d_roi():
    """ROI can be set and mean is computed."""
    data = np.ones((5, 32, 32), dtype=np.float32) * 10.0
    widget = Show3D(data)
    widget.set_roi(16, 16, radius=5)
    assert widget.roi_active is True
    assert widget.roi_stats["mean"] == pytest.approx(10.0)

def test_show3d_roi_defaults_include_visibility_and_lock():
    data = np.ones((3, 16, 16), dtype=np.float32)
    widget = Show3D(data)
    widget.add_roi(shape="circle")
    roi = widget.roi_list[widget.roi_selected_idx]
    assert roi["visible"] is True
    assert roi["locked"] is False

def test_show3d_roi_shapes_constant_mean():
    """Different ROI shapes work."""
    data = np.ones((5, 32, 32), dtype=np.float32) * 10.0
    widget = Show3D(data)
    for shape_fn in [widget.roi_circle, widget.roi_square, lambda r: widget.roi_rectangle(r * 2, r)]:
        widget.set_roi(16, 16, radius=5)
        shape_fn(5)
        assert widget.roi_stats["mean"] == pytest.approx(10.0)

def test_show3d_colormap():
    """Colormap option is applied."""
    data = np.random.rand(5, 16, 16).astype(np.float32)
    widget = Show3D(data, cmap="viridis")
    assert widget.cmap == "viridis"

def test_show3d_rejects_2d():
    """Raises error for 2D input."""
    data = np.random.rand(16, 16).astype(np.float32)
    with pytest.raises(ValueError, match="3D"):
        Show3D(data)

def test_show3d_timestamps():
    """Timestamps are stored."""
    data = np.random.rand(5, 16, 16).astype(np.float32)
    times = [0.0, 0.1, 0.2, 0.3, 0.4]
    widget = Show3D(data, timestamps=times, timestamp_unit="ms")
    assert widget.timestamps == times
    assert widget.timestamp_unit == "ms"

def test_show3d_display_options():
    """Log scale and auto contrast options work."""
    data = np.random.rand(5, 16, 16).astype(np.float32)
    widget = Show3D(data, log_scale=True, auto_contrast=True)
    assert widget.log_scale is True
    assert widget.auto_contrast is True

def test_show3d_boomerang():
    """Boomerang mode can be set."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.boomerang = True
    assert widget.boomerang is True

def test_show3d_bookmarks():
    """Bookmarked frames can be set and retrieved."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.bookmarked_frames = [0, 5, 9]
    assert widget.bookmarked_frames == [0, 5, 9]

def test_show3d_title():
    """Title parameter is stored."""
    data = np.random.rand(5, 16, 16).astype(np.float32)
    widget = Show3D(data, title="My Stack")
    assert widget.title == "My Stack"

def test_show3d_vmin_vmax():
    """Custom vmin/vmax stored."""
    data = np.random.rand(5, 16, 16).astype(np.float32)
    widget = Show3D(data, vmin=0.2, vmax=0.8)
    assert widget._vmin == pytest.approx(0.2)
    assert widget._vmax == pytest.approx(0.8)

def test_show3d_fps():
    """FPS parameter is stored."""
    data = np.random.rand(5, 16, 16).astype(np.float32)
    widget = Show3D(data, fps=30.0)
    assert widget.fps == pytest.approx(30.0)

def test_show3d_pixel_size():
    """Pixel size parameter is stored."""
    data = np.random.rand(5, 16, 16).astype(np.float32)
    widget = Show3D(data, pixel_size=5.0)
    assert widget.pixel_size == pytest.approx(5.0)

def test_show3d_scale_bar():
    """Scale bar visibility parameter is stored."""
    data = np.random.rand(5, 16, 16).astype(np.float32)
    widget = Show3D(data, scale_bar_visible=False)
    assert widget.scale_bar_visible is False

def test_show3d_loop_range():
    """Loop range parameters are stored."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.loop_start = 2
    widget.loop_end = 7
    assert widget.loop_start == 2
    assert widget.loop_end == 7

def test_show3d_slice_change_updates_stats():
    """Changing slice_idx updates statistics."""
    data = np.zeros((5, 16, 16), dtype=np.float32)
    data[0] = 10.0
    data[4] = 50.0
    widget = Show3D(data)
    widget.slice_idx = 0
    assert widget.stats_mean == pytest.approx(10.0)
    widget.slice_idx = 4
    assert widget.stats_mean == pytest.approx(50.0)

def test_show3d_roi_rectangle():
    """Rectangle ROI with roi_width/height computes mean."""
    data = np.ones((5, 32, 32), dtype=np.float32) * 7.0
    widget = Show3D(data)
    widget.set_roi(16, 16, radius=5)
    widget.roi_rectangle(10, 6)
    assert widget.roi_stats["mean"] == pytest.approx(7.0)

def test_show3d_roi_at_edge():
    """ROI at image edge doesn't crash."""
    data = np.ones((5, 32, 32), dtype=np.float32) * 5.0
    widget = Show3D(data)
    widget.set_roi(0, 0, radius=3)
    # Should not crash, and roi mean should be finite
    assert np.isfinite(widget.roi_stats["mean"])

def test_show3d_constant_data():
    """Constant data doesn't crash."""
    data = np.ones((5, 16, 16), dtype=np.float32) * 42.0
    widget = Show3D(data)
    assert widget.stats_mean == pytest.approx(42.0)
    assert len(widget.frame_bytes) > 0

def test_show3d_single_slice():
    """Single slice works."""
    data = np.random.rand(1, 16, 16).astype(np.float32)
    widget = Show3D(data)
    assert widget.n_slices == 1
    assert widget.slice_idx == 0

def test_show3d_image_width():
    """canvas_size parameter is stored."""
    data = np.random.rand(5, 16, 16).astype(np.float32)
    widget = Show3D(data, canvas_size=400)
    assert widget.canvas_size == 400

def test_show3d_current_timestamp():
    """Current timestamp updates with slice_idx."""
    data = np.random.rand(5, 16, 16).astype(np.float32)
    times = [0.0, 0.5, 1.0, 1.5, 2.0]
    widget = Show3D(data, timestamps=times)
    widget.slice_idx = 3
    assert widget.current_timestamp == pytest.approx(1.5)

def test_show3d_reverse_trait():
    """Reverse playback trait is stored."""
    data = np.random.rand(5, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.reverse = True
    assert widget.reverse is True

# --- Playback control tests ---

def test_show3d_loop_default_on():
    """Loop defaults to True."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    assert widget.loop is True

def test_show3d_boomerang_default_off():
    """Boomerang defaults to False."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    assert widget.boomerang is False

def test_show3d_loop_off_preserves_boomerang_off():
    """Turning loop off when boomerang is already off keeps it off."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.loop = False
    assert widget.boomerang is False

def test_show3d_boomerang_requires_loop_conceptually():
    """Boomerang can be set independently at trait level (JS enforces coupling)."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.boomerang = True
    assert widget.boomerang is True
    widget.loop = False
    # At Python trait level, boomerang stays set (JS toggle handles coupling)
    # This test documents that Python traits are independent
    assert widget.loop is False

def test_show3d_play_sets_playing():
    """play() sets playing to True."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.play()
    assert widget.playing is True

def test_show3d_pause_clears_playing():
    """pause() sets playing to False."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.play()
    widget.pause()
    assert widget.playing is False

def test_show3d_stop_resets_to_start():
    """stop() sets playing to False and resets to first frame."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.slice_idx = 7
    widget.play()
    widget.stop()
    assert widget.playing is False
    assert widget.slice_idx == 0

def test_show3d_stop_resets_to_zero():
    """stop() always resets to frame 0 (JS stop button respects loop_start)."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.loop_start = 3
    widget.slice_idx = 7
    widget.stop()
    assert widget.slice_idx == 0

def test_show3d_loop_range_set_and_get():
    """Loop range can be set and read back."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.loop_start = 2
    widget.loop_end = 8
    assert widget.loop_start == 2
    assert widget.loop_end == 8

def test_show3d_loop_range_default_full():
    """Default loop range covers entire stack."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    assert widget.loop_start == 0
    assert widget.loop_end == -1  # -1 means last frame

def test_show3d_loop_range_reset():
    """Loop range can be reset to full."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.loop_start = 3
    widget.loop_end = 7
    # Reset
    widget.loop_start = 0
    widget.loop_end = -1
    assert widget.loop_start == 0
    assert widget.loop_end == -1

def test_show3d_loop_range_clamp():
    """Loop range values are stored even if out of bounds (JS clamps)."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.loop_start = 5
    widget.loop_end = 5
    assert widget.loop_start == 5
    assert widget.loop_end == 5

def test_show3d_play_pause_toggle():
    """Repeated play/pause toggles work."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.play()
    assert widget.playing is True
    widget.pause()
    assert widget.playing is False
    widget.play()
    assert widget.playing is True
    widget.pause()
    assert widget.playing is False

def test_show3d_reverse_with_play():
    """Reverse can be set before or during playback."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.reverse = True
    widget.play()
    assert widget.playing is True
    assert widget.reverse is True

def test_show3d_fps_range():
    """FPS can be set to various valid values."""
    data = np.random.rand(5, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.fps = 1
    assert widget.fps == pytest.approx(1.0)
    widget.fps = 30
    assert widget.fps == pytest.approx(30.0)

def test_show3d_playing_default_false():
    """Widget starts with playing=False."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    assert widget.playing is False

def test_show3d_boomerang_with_loop():
    """Boomerang can be enabled alongside loop."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.loop = True
    widget.boomerang = True
    assert widget.loop is True
    assert widget.boomerang is True

def test_show3d_loop_range_with_boomerang():
    """Loop range works with boomerang mode."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.loop = True
    widget.boomerang = True
    widget.loop_start = 2
    widget.loop_end = 6
    assert widget.loop_start == 2
    assert widget.loop_end == 6
    assert widget.boomerang is True

def test_show3d_frame_bytes_float32_size():
    """frame_bytes has correct size (height * width * 4 bytes for float32)."""
    data = np.random.rand(5, 16, 16).astype(np.float32)
    widget = Show3D(data)
    assert len(widget.frame_bytes) == 16 * 16 * 4

def test_show3d_data_range():
    """data_min and data_max reflect global range across all frames."""
    data = np.zeros((5, 16, 16), dtype=np.float32)
    data[0] = -1.0
    data[4] = 10.0
    widget = Show3D(data)
    assert widget.data_min == pytest.approx(-1.0)
    assert widget.data_max == pytest.approx(10.0)

# =========================================================================
# Playback Buffer (sliding prefetch)
# =========================================================================

def test_show3d_default_buffer_size():
    """Default buffer_size is 64."""
    data = np.random.rand(100, 8, 8).astype(np.float32)
    widget = Show3D(data)
    assert widget._buffer_size == 64

def test_show3d_buffer_size_param():
    """buffer_size parameter is respected."""
    data = np.random.rand(100, 8, 8).astype(np.float32)
    widget = Show3D(data, buffer_size=32)
    assert widget._buffer_size == 32

def test_show3d_buffer_small_stack():
    """Stack smaller than buffer_size clamps to n_slices."""
    data = np.random.rand(5, 8, 8).astype(np.float32)
    widget = Show3D(data, buffer_size=64)
    assert widget._buffer_size == 5

def test_show3d_buffer_sent_on_play():
    """Buffer bytes are sent when playback starts."""
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.slice_idx = 3
    widget.playing = True
    assert len(widget._buffer_bytes) > 0
    assert widget._buffer_start == 3
    assert widget._buffer_count > 0

def test_show3d_buffer_data_correct():
    """Buffer contains correct float32 frame data."""
    data = np.zeros((10, 8, 8), dtype=np.float32)
    for i in range(10):
        data[i] = float(i)
    widget = Show3D(data)
    widget.slice_idx = 0
    widget.playing = True
    buf = np.frombuffer(widget._buffer_bytes, dtype=np.float32)
    frame_size = 8 * 8
    assert buf[:frame_size].mean() == pytest.approx(0.0)
    assert buf[frame_size : 2 * frame_size].mean() == pytest.approx(1.0)

def test_show3d_slice_change_skipped_during_playback():
    """Changing slice_idx during playback does NOT trigger _update_all."""
    data = np.zeros((10, 8, 8), dtype=np.float32)
    data[0] = 10.0
    data[5] = 50.0
    widget = Show3D(data)
    widget.slice_idx = 0
    assert widget.stats_mean == pytest.approx(10.0)
    widget.playing = True
    widget.slice_idx = 5
    # Stats should NOT have updated (still 10.0 from frame 0)
    assert widget.stats_mean == pytest.approx(10.0)

def test_show3d_stats_correct_after_stop():
    """After playback stops and slice_idx is set, stats are recomputed."""
    data = np.zeros((10, 8, 8), dtype=np.float32)
    data[5] = 42.0
    widget = Show3D(data)
    widget.slice_idx = 0
    widget.playing = True
    widget.playing = False
    widget.slice_idx = 5
    assert widget.stats_mean == pytest.approx(42.0)

def test_show3d_prefetch_triggers_buffer():
    """Setting _prefetch_request triggers new buffer send."""
    data = np.random.rand(100, 8, 8).astype(np.float32)
    widget = Show3D(data, buffer_size=32)
    widget.slice_idx = 0
    widget.playing = True
    assert widget._buffer_start == 0
    widget._prefetch_request = 32
    assert widget._buffer_start == 32
    assert widget._buffer_count == 32

def test_show3d_prefetch_ignored_when_not_playing():
    """Prefetch request is ignored when not playing."""
    data = np.random.rand(100, 8, 8).astype(np.float32)
    widget = Show3D(data, buffer_size=32)
    widget._prefetch_request = 50
    assert widget._buffer_start == 0
    assert widget._buffer_count == 0

def test_show3d_buffer_wraparound():
    """Buffer wraps around when start_idx + buffer_size > n_slices."""
    data = np.zeros((10, 4, 4), dtype=np.float32)
    for i in range(10):
        data[i] = float(i)
    widget = Show3D(data, buffer_size=8)
    widget.slice_idx = 5
    widget.playing = True
    assert widget._buffer_start == 5
    assert widget._buffer_count == 8
    buf = np.frombuffer(widget._buffer_bytes, dtype=np.float32)
    frame_size = 4 * 4
    # Frame at index 5 (first in buffer)
    assert buf[:frame_size].mean() == pytest.approx(5.0)
    # Frame at index 9 (5th in buffer, last before wrap)
    assert buf[4 * frame_size : 5 * frame_size].mean() == pytest.approx(9.0)
    # Frame at index 0 (6th in buffer, wrapped)
    assert buf[5 * frame_size : 6 * frame_size].mean() == pytest.approx(0.0)

def test_show3d_profile_set_and_clear():
    data = np.random.rand(5, 32, 32).astype(np.float32)
    widget = Show3D(data)
    widget.set_profile((0, 0), (31, 31))
    assert len(widget.profile_line) == 2
    assert widget.profile == [(0.0, 0.0), (31.0, 31.0)]
    widget.clear_profile()
    assert widget.profile_line == []
    assert widget.profile == []

def test_show3d_profile_values():
    data = np.ones((3, 16, 16), dtype=np.float32) * 5.0
    widget = Show3D(data)
    widget.set_profile((0, 0), (0, 15))
    vals = widget.profile_values
    assert vals is not None
    assert len(vals) >= 2
    assert vals.mean() == pytest.approx(5.0, abs=0.01)

def test_show3d_profile_distance_pixels():
    data = np.ones((2, 10, 10), dtype=np.float32)
    widget = Show3D(data)
    widget.set_profile((0, 0), (3, 4))
    assert widget.profile_distance == pytest.approx(5.0)

def test_show3d_profile_distance_calibrated():
    data = np.ones((2, 10, 10), dtype=np.float32)
    widget = Show3D(data, pixel_size=5.0)  # 5.0 Å/px
    widget.set_profile((0, 0), (3, 4))
    assert widget.profile_distance == pytest.approx(25.0)  # 5px * 5.0 Å/px

def test_show3d_profile_no_line():
    data = np.ones((2, 10, 10), dtype=np.float32)
    widget = Show3D(data)
    assert widget.profile_values is None
    assert widget.profile_distance is None

def test_show3d_profile_width_trait():
    data = np.ones((2, 16, 16), dtype=np.float32)
    widget = Show3D(data)
    assert widget.profile_width == 1
    widget.profile_width = 5
    assert widget.profile_width == 5

# =========================================================================
# Dimension Label
# =========================================================================

def test_show3d_dim_label_default():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    widget = Show3D(data)
    assert widget.dim_label == "Frame"

def test_show3d_dim_label_custom():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    widget = Show3D(data, dim_label="Defocus")
    assert widget.dim_label == "Defocus"

# =========================================================================
# Tool Lock/Hide Controls
# =========================================================================

def test_show3d_disabled_tools_default():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    widget = Show3D(data)
    assert widget.disabled_tools == []

def test_show3d_disabled_tools_custom_and_alias():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    widget = Show3D(data, disabled_tools=["display", "navigation", "display"])
    assert widget.disabled_tools == ["display", "playback"]

def test_show3d_disabled_tools_flags():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    widget = Show3D(
        data,
        disable_histogram=True,
        disable_playback=True,
        disable_roi=True,
    )
    assert widget.disabled_tools == ["histogram", "playback", "roi"]

def test_show3d_disabled_tools_navigation_flag_alias():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    widget = Show3D(data, disable_navigation=True)
    assert widget.disabled_tools == ["playback"]

def test_show3d_disabled_tools_disable_all():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    widget = Show3D(data, disabled_tools=["display"], disable_all=True)
    assert widget.disabled_tools == ["all"]

def test_show3d_disabled_tools_invalid():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    with pytest.raises(ValueError, match="Unknown tool group"):
        Show3D(data, disabled_tools=["unknown"])

def test_show3d_hidden_tools_default():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    widget = Show3D(data)
    assert widget.hidden_tools == []

def test_show3d_hidden_tools_custom_and_alias():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    widget = Show3D(data, hidden_tools=["stats", "navigation", "stats"])
    assert widget.hidden_tools == ["stats", "playback"]

def test_show3d_hidden_tools_flags():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    widget = Show3D(
        data,
        hide_display=True,
        hide_profile=True,
    )
    assert widget.hidden_tools == ["display", "profile"]

def test_show3d_hidden_tools_hide_all():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    widget = Show3D(data, hidden_tools=["display"], hide_all=True)
    assert widget.hidden_tools == ["all"]

def test_show3d_hidden_tools_invalid():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    with pytest.raises(ValueError, match="Unknown tool group"):
        Show3D(data, hidden_tools=["bad_group"])

# =========================================================================
# Method Chaining
# =========================================================================

def test_show3d_method_chaining():
    data = np.random.rand(10, 16, 16).astype(np.float32)
    w = Show3D(data)
    result = w.goto(3).play().pause().stop()
    assert result is w
    result = w.set_roi(5, 5, 10).roi_circle(8).roi_square(6).roi_rectangle(10, 5)
    assert result is w
    result = w.set_profile((0, 0), (10, 10)).clear_profile()
    assert result is w

def test_show3d_goto():
    data = np.random.rand(10, 8, 8).astype(np.float32)
    widget = Show3D(data)
    widget.goto(7)
    assert widget.slice_idx == 7
    widget.goto(12)  # wraps around
    assert widget.slice_idx == 2

# =========================================================================
# Annular ROI
# =========================================================================

def test_show3d_roi_annular():
    data = np.ones((5, 32, 32), dtype=np.float32) * 3.0
    widget = Show3D(data)
    widget.roi_annular(inner=5, outer=10)
    roi = widget.roi_list[widget.roi_selected_idx]
    assert roi["shape"] == "annular"
    assert roi["radius_inner"] == 5
    assert roi["radius"] == 10
    assert widget.roi_active is True

def test_show3d_roi_annular_mean():
    data = np.ones((3, 32, 32), dtype=np.float32) * 7.0
    widget = Show3D(data)
    widget.set_roi(16, 16, 10)
    widget.roi_annular(inner=3, outer=8)
    assert widget.roi_stats["mean"] == pytest.approx(7.0)

def test_show3d_roi_shapes():
    data = np.random.rand(3, 16, 16).astype(np.float32)
    widget = Show3D(data)
    widget.roi_circle(5)
    assert widget.roi_list[widget.roi_selected_idx]["shape"] == "circle"
    widget.roi_square(5)
    assert widget.roi_list[widget.roi_selected_idx]["shape"] == "square"
    widget.roi_rectangle(10, 5)
    assert widget.roi_list[widget.roi_selected_idx]["shape"] == "rectangle"
    widget.roi_annular(3, 8)
    assert widget.roi_list[widget.roi_selected_idx]["shape"] == "annular"

def test_show3d_duplicate_and_delete_selected_roi():
    data = np.ones((4, 24, 24), dtype=np.float32)
    widget = Show3D(data)
    widget.add_roi(row=8, col=9, shape="rectangle")
    assert len(widget.roi_list) == 1
    widget.duplicate_selected_roi(row_offset=2, col_offset=3)
    assert len(widget.roi_list) == 2
    assert widget.roi_selected_idx == 1
    assert widget.roi_list[1]["row"] == 10
    assert widget.roi_list[1]["col"] == 12
    assert widget.roi_list[1]["visible"] is True
    assert widget.roi_list[1]["locked"] is False
    widget.delete_selected_roi()
    assert len(widget.roi_list) == 1
    assert widget.roi_selected_idx == 0

# =========================================================================
# ROI Plot Data
# =========================================================================

def test_show3d_roi_plot_data():
    data = np.zeros((5, 8, 8), dtype=np.float32)
    for i in range(5):
        data[i] = float(i)
    widget = Show3D(data)
    widget.set_roi(4, 4, 3)
    assert len(widget.roi_plot_data) > 0
    plot = np.frombuffer(widget.roi_plot_data, dtype=np.float32)
    assert len(plot) == 5
    for i in range(5):
        assert plot[i] == pytest.approx(float(i))

def test_show3d_roi_plot_data_cleared():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    widget = Show3D(data)
    widget.set_roi(4, 4, 3)
    assert len(widget.roi_plot_data) > 0
    widget.roi_active = False
    assert widget.roi_plot_data == b""

# =========================================================================
# Playback Path
# =========================================================================

def test_show3d_playback_path():
    data = np.random.rand(10, 8, 8).astype(np.float32)
    widget = Show3D(data)
    widget.set_playback_path([0, 5, 3, 7])
    assert widget.playback_path == [0, 5, 3, 7]

def test_show3d_playback_path_clear():
    data = np.random.rand(10, 8, 8).astype(np.float32)
    widget = Show3D(data)
    widget.set_playback_path([0, 5, 3])
    widget.clear_playback_path()
    assert widget.playback_path == []

def test_show3d_playback_path_wraps():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    widget = Show3D(data)
    widget.set_playback_path([0, 7, 12])
    assert widget.playback_path == [0, 2, 2]  # 7%5=2, 12%5=2

def test_show3d_playback_path_chaining():
    data = np.random.rand(10, 8, 8).astype(np.float32)
    widget = Show3D(data)
    result = widget.set_playback_path([0, 1, 2]).clear_playback_path()
    assert result is widget

def test_show3d_gif_generation_sets_metadata():
    import json

    data = np.random.rand(6, 12, 12).astype(np.float32)
    widget = Show3D(data, cmap="viridis")
    widget.loop_start = 1
    widget.loop_end = 3
    widget.fps = 4.0

    widget._generate_gif()
    assert widget._gif_data[:3] == b"GIF"
    metadata = json.loads(widget._gif_metadata_json)
    assert metadata["widget_name"] == "Show3D"
    assert metadata["format"] == "gif"
    assert metadata["export_kind"] == "animated_frames"
    assert metadata["frame_range"] == {"start": 1, "end": 3}

def test_show3d_gif_generation_no_frames_clears_metadata():
    data = np.random.rand(4, 8, 8).astype(np.float32)
    widget = Show3D(data)
    widget.loop_start = 3
    widget.loop_end = 1
    widget._generate_gif()
    assert widget._gif_data == b""
    assert widget._gif_metadata_json == ""

def test_show3d_bundle_generation_contains_expected_files():
    import io
    import json
    import zipfile

    data = np.random.rand(5, 12, 12).astype(np.float32)
    widget = Show3D(data, cmap="viridis")
    widget.add_roi(row=6, col=6, shape="circle")
    widget._generate_bundle()

    assert len(widget._bundle_data) > 0
    with zipfile.ZipFile(io.BytesIO(widget._bundle_data), "r") as zf:
        names = set(zf.namelist())
        assert "roi_timeseries.csv" in names
        assert "state.json" in names
        assert any(name.startswith("frame_") and name.endswith(".png") for name in names)
        state_payload = json.loads(zf.read("state.json").decode("utf-8"))
        assert state_payload["widget_name"] == "Show3D"
        csv_text = zf.read("roi_timeseries.csv").decode("utf-8")
        assert "roi_1_mean" in csv_text

# ── State Protocol ────────────────────────────────────────────────────────

def test_show3d_state_dict_roundtrip():
    data = np.random.rand(10, 32, 32).astype(np.float32)
    w = Show3D(data, cmap="viridis", log_scale=True, auto_contrast=True,
               title="Stack", pixel_size=5.0, fps=15.0, show_fft=True,
               fft_window=False,
               disabled_tools=["display", "navigation"], hidden_tools=["stats"])
    w.roi_active = True
    w.roi_list = [{"row": 10, "col": 15, "shape": "circle", "radius": 10, "radius_inner": 5, "width": 20, "height": 20, "color": "#4fc3f7", "line_width": 2, "highlight": False}]
    w.roi_selected_idx = 0
    sd = w.state_dict()
    w2 = Show3D(data, state=sd)
    assert w2.cmap == "viridis"
    assert w2.log_scale is True
    assert w2.title == "Stack"
    assert w2.pixel_size == pytest.approx(5.0)
    assert w2.fps == pytest.approx(15.0)
    assert w2.show_fft is True
    assert w2.fft_window is False
    assert w2.roi_active is True
    assert w2.roi_list[0]["row"] == 10
    assert w2.disabled_tools == ["display", "playback"]
    assert w2.hidden_tools == ["stats"]

def test_show3d_state_dict_includes_tool_customization():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    w = Show3D(data, disabled_tools=["display"], hidden_tools=["playback"])
    sd = w.state_dict()
    assert sd["disabled_tools"] == ["display"]
    assert sd["hidden_tools"] == ["playback"]

def test_show3d_save_load_file(tmp_path):
    import json
    data = np.random.rand(5, 16, 16).astype(np.float32)
    w = Show3D(data, cmap="plasma", title="Saved")
    path = tmp_path / "show3d_state.json"
    w.save(str(path))
    assert path.exists()
    saved = json.loads(path.read_text())
    assert saved["metadata_version"] == "1.0"
    assert saved["widget_name"] == "Show3D"
    assert isinstance(saved["widget_version"], str)
    assert saved["state"]["cmap"] == "plasma"
    w2 = Show3D(data, state=str(path))
    assert w2.cmap == "plasma"
    assert w2.title == "Saved"

def test_show3d_summary(capsys):
    data = np.random.rand(10, 32, 32).astype(np.float32)
    w = Show3D(data, title="Focal Series", cmap="magma", pixel_size=2.5)
    w.summary()
    out = capsys.readouterr().out
    assert "Focal Series" in out
    assert "10×32×32" in out
    assert "magma" in out
    assert "2.5" in out

def test_show3d_summary_with_single_profile_point(capsys):
    data = np.random.rand(10, 32, 32).astype(np.float32)
    w = Show3D(data)
    w.profile_line = [{"row": 4.0, "col": 7.0}]
    w.summary()
    out = capsys.readouterr().out
    assert "Show3D" in out

def test_show3d_repr():
    data = np.random.rand(10, 32, 32).astype(np.float32)
    w = Show3D(data, cmap="magma")
    r = repr(w)
    assert "Show3D" in r
    assert "10×32×32" in r
    assert "magma" in r

def test_show3d_set_image():
    data = np.random.rand(10, 16, 16).astype(np.float32)
    widget = Show3D(data, cmap="gray", fps=12)
    assert widget.n_slices == 10

    new_data = np.random.rand(20, 32, 24).astype(np.float32)
    widget.set_image(new_data)
    assert widget.n_slices == 20
    assert widget.height == 32
    assert widget.width == 24
    assert widget.cmap == "gray"
    assert widget.fps == 12
    assert widget.data_min == pytest.approx(float(new_data.min()))
    assert widget.data_max == pytest.approx(float(new_data.max()))

# ── save_image ───────────────────────────────────────────────────────────

def test_show3d_save_image_png(tmp_path):
    stack = np.random.rand(10, 32, 32).astype(np.float32)
    w = Show3D(stack, cmap="viridis")
    out = w.save_image(tmp_path / "frame.png")
    assert out.exists()
    assert out.stat().st_size > 0
    from PIL import Image
    img = Image.open(out)
    assert img.size == (32, 32)

def test_show3d_save_image_pdf(tmp_path):
    stack = np.random.rand(10, 32, 32).astype(np.float32)
    w = Show3D(stack)
    out = w.save_image(tmp_path / "frame.pdf")
    assert out.exists()

def test_show3d_save_image_frame_idx(tmp_path):
    stack = np.random.rand(10, 32, 32).astype(np.float32)
    w = Show3D(stack)
    out = w.save_image(tmp_path / "f5.png", frame_idx=5)
    assert out.exists()

def test_show3d_save_image_bad_idx(tmp_path):
    stack = np.random.rand(10, 32, 32).astype(np.float32)
    w = Show3D(stack)
    with pytest.raises(IndexError):
        w.save_image(tmp_path / "out.png", frame_idx=20)

def test_show3d_save_image_bad_format(tmp_path):
    stack = np.random.rand(5, 16, 16).astype(np.float32)
    w = Show3D(stack)
    with pytest.raises(ValueError, match="Unsupported format"):
        w.save_image(tmp_path / "out.gif")

def test_show3d_widget_version_is_set():
    stack = np.random.rand(5, 16, 16).astype(np.float32)
    w = Show3D(stack)
    assert w.widget_version != "unknown"

def test_show3d_show_controls_default():
    stack = np.random.rand(5, 16, 16).astype(np.float32)
    w = Show3D(stack)
    assert w.show_controls is True


# =========================================================================
# profile_all_frames
# =========================================================================

def test_show3d_profile_all_frames_shape():
    data = np.random.rand(8, 16, 16).astype(np.float32)
    w = Show3D(data)
    w.set_profile((0, 0), (0, 15))
    result = w.profile_all_frames()
    assert result.shape[0] == 8
    assert result.shape[1] >= 2
    assert result.dtype == np.float32

def test_show3d_profile_all_frames_values():
    data = np.zeros((4, 10, 10), dtype=np.float32)
    for i in range(4):
        data[i] = float(i * 10)
    w = Show3D(data)
    w.set_profile((5, 0), (5, 9))
    result = w.profile_all_frames()
    for i in range(4):
        assert result[i].mean() == pytest.approx(float(i * 10), abs=0.5)

def test_show3d_profile_all_frames_explicit_endpoints():
    data = np.ones((3, 16, 16), dtype=np.float32) * 7.0
    w = Show3D(data)
    result = w.profile_all_frames(start=(0, 0), end=(0, 15))
    assert result.shape[0] == 3
    assert result[0].mean() == pytest.approx(7.0, abs=0.1)

def test_show3d_profile_all_frames_no_line_raises():
    data = np.random.rand(3, 8, 8).astype(np.float32)
    w = Show3D(data)
    with pytest.raises(ValueError, match="No profile line"):
        w.profile_all_frames()

def test_show3d_profile_all_frames_respects_width():
    data = np.ones((2, 20, 20), dtype=np.float32) * 3.0
    w = Show3D(data)
    w.profile_width = 5
    w.set_profile((10, 0), (10, 19))
    result = w.profile_all_frames()
    assert result.shape[0] == 2
    assert result[0].mean() == pytest.approx(3.0, abs=0.1)


# =========================================================================
# diff_mode
# =========================================================================

def test_show3d_diff_mode_default():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    w = Show3D(data)
    assert w.diff_mode == "off"

def test_show3d_diff_mode_previous():
    data = np.zeros((4, 8, 8), dtype=np.float32)
    for i in range(4):
        data[i] = float(i * 10)
    w = Show3D(data, diff_mode="previous")
    w.slice_idx = 2
    frame = w._get_display_frame(2)
    assert frame.mean() == pytest.approx(10.0)  # 20 - 10

def test_show3d_diff_mode_previous_frame_zero():
    data = np.ones((3, 8, 8), dtype=np.float32) * 5.0
    w = Show3D(data, diff_mode="previous")
    frame = w._get_display_frame(0)
    assert frame.mean() == pytest.approx(0.0)

def test_show3d_diff_mode_first():
    data = np.zeros((4, 8, 8), dtype=np.float32)
    for i in range(4):
        data[i] = float(i * 10)
    w = Show3D(data, diff_mode="first")
    frame = w._get_display_frame(3)
    assert frame.mean() == pytest.approx(30.0)  # 30 - 0

def test_show3d_diff_mode_off_normal():
    data = np.zeros((3, 8, 8), dtype=np.float32)
    data[1] = 42.0
    w = Show3D(data)
    frame = w._get_display_frame(1)
    assert frame.mean() == pytest.approx(42.0)

def test_show3d_diff_mode_invalid():
    data = np.random.rand(3, 8, 8).astype(np.float32)
    with pytest.raises(traitlets.TraitError):
        Show3D(data, diff_mode="bad")

def test_show3d_diff_mode_in_state_dict():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    w = Show3D(data, diff_mode="previous")
    sd = w.state_dict()
    assert sd["diff_mode"] == "previous"

def test_show3d_diff_mode_state_roundtrip():
    data = np.random.rand(5, 8, 8).astype(np.float32)
    w = Show3D(data, diff_mode="first")
    sd = w.state_dict()
    w2 = Show3D(data, state=sd)
    assert w2.diff_mode == "first"

def test_show3d_diff_mode_summary(capsys):
    data = np.random.rand(5, 8, 8).astype(np.float32)
    w = Show3D(data, diff_mode="previous")
    w.summary()
    out = capsys.readouterr().out
    assert "diff=previous" in out

def test_show3d_diff_mode_data_range_negative():
    data = np.zeros((3, 8, 8), dtype=np.float32)
    data[0] = 10.0
    data[1] = 5.0
    data[2] = 20.0
    w = Show3D(data)
    w.diff_mode = "previous"
    # frame 1 - frame 0 = -5, frame 2 - frame 1 = 15
    assert w.data_min < 0

def test_show3d_diff_mode_toggle_restores_range():
    data = np.zeros((3, 8, 8), dtype=np.float32)
    data[0] = 0.0
    data[1] = 50.0
    data[2] = 100.0
    w = Show3D(data)
    orig_min, orig_max = w.data_min, w.data_max
    w.diff_mode = "previous"
    assert w.data_min != orig_min or w.data_max != orig_max
    w.diff_mode = "off"
    assert w.data_min == pytest.approx(orig_min)
    assert w.data_max == pytest.approx(orig_max)
