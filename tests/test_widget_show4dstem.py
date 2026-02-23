import json
import pathlib

import numpy as np
import pytest
import quantem.widget
from quantem.widget import Show4DSTEM

def test_version_exists():
    assert hasattr(quantem.widget, "__version__")

def test_version_is_string():
    assert isinstance(quantem.widget.__version__, str)

def test_show4dstem_loads():
    """Widget can be created from mock 4D data."""
    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data)
    assert widget is not None

def test_show4dstem_flattened_scan_shape_mapping():
    """Test flattened 3D data with explicit scan shape."""
    data = np.zeros((6, 2, 2), dtype=np.float32)
    for idx in range(data.shape[0]):
        data[idx] = idx
    widget = Show4DSTEM(data, scan_shape=(2, 3))
    assert (widget.shape_rows, widget.shape_cols) == (2, 3)
    assert (widget.det_rows, widget.det_cols) == (2, 2)
    frame = widget._get_frame(1, 2)
    assert np.array_equal(frame, np.full((2, 2), 5, dtype=np.float32))

def test_show4dstem_dp_scale_mode():
    """DP scale mode is configurable through the canonical trait."""
    data = np.random.rand(2, 2, 8, 8).astype(np.float32) * 100 + 1
    widget = Show4DSTEM(data)
    assert widget.dp_scale_mode == "linear"
    widget.dp_scale_mode = "log"
    assert widget.dp_scale_mode == "log"

def test_show4dstem_auto_detect_center():
    """Test automatic center spot detection using centroid."""
    data = np.zeros((2, 2, 7, 7), dtype=np.float32)
    for i in range(7):
        for j in range(7):
            dist = np.sqrt((i - 3) ** 2 + (j - 3) ** 2)
            if dist <= 1.5:
                data[:, :, i, j] = 100.0
    widget = Show4DSTEM(data, precompute_virtual_images=False)
    widget.auto_detect_center()
    assert abs(widget.center_col - 3.0) < 0.5
    assert abs(widget.center_row - 3.0) < 0.5
    assert widget.bf_radius > 0

def test_show4dstem_adf_preset_cache():
    """Test that ADF preset cache works when precompute is enabled."""
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data, center=(8, 8), bf_radius=2, precompute_virtual_images=True)
    assert widget._cached_adf_virtual is not None
    widget.roi_mode = "annular"
    widget.roi_center_col = 8
    widget.roi_center_row = 8
    widget.roi_radius_inner = 2
    widget.roi_radius = 8
    cached = widget._get_cached_preset()
    assert cached == widget._cached_adf_virtual

def test_show4dstem_rectangular_scan_shape():
    """Test that rectangular (non-square) scans work correctly."""
    data = np.random.rand(4, 8, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data)
    assert widget.shape_rows == 4
    assert widget.shape_cols == 8
    assert widget.det_rows == 16
    assert widget.det_cols == 16
    frame_00 = widget._get_frame(0, 0)
    frame_37 = widget._get_frame(3, 7)
    assert frame_00.shape == (16, 16)
    assert frame_37.shape == (16, 16)

def test_show4dstem_hot_pixel_removal_uint16():
    """Test that saturated uint16 hot pixels are removed at init."""
    data = np.zeros((4, 4, 8, 8), dtype=np.uint16)
    data[:, :, :, :] = 100
    data[:, :, 3, 5] = 65535
    data[:, :, 1, 2] = 65535
    widget = Show4DSTEM(data)
    assert widget.dp_global_max < 65535
    assert widget.dp_global_max == 100.0
    frame = widget._get_frame(0, 0)
    assert frame[3, 5] == 0
    assert frame[1, 2] == 0
    assert frame[0, 0] == 100

def test_show4dstem_hot_pixel_removal_uint8():
    """Test that saturated uint8 hot pixels are removed at init."""
    data = np.zeros((4, 4, 8, 8), dtype=np.uint8)
    data[:, :, :, :] = 50
    data[:, :, 2, 3] = 255
    widget = Show4DSTEM(data)
    assert widget.dp_global_max == 50.0
    frame = widget._get_frame(0, 0)
    assert frame[2, 3] == 0

def test_show4dstem_no_hot_pixel_removal_float32():
    """Test that float32 data is not modified (no saturated value)."""
    data = np.ones((4, 4, 8, 8), dtype=np.float32) * 1000
    widget = Show4DSTEM(data)
    assert widget.dp_global_max == 1000.0

def test_show4dstem_roi_modes():
    """Test all ROI modes compute virtual images correctly."""
    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data, center=(8, 8), bf_radius=3)
    for mode in ["point", "circle", "square", "annular", "rect"]:
        widget.roi_mode = mode
        widget.roi_active = True
        assert len(widget.vi_stats) == 4
        assert widget.vi_stats[2] >= widget.vi_stats[1]

def test_show4dstem_virtual_image_excludes_hot_pixels():
    """Test that virtual images don't include hot pixel contributions."""
    data = np.ones((4, 4, 8, 8), dtype=np.uint16) * 10
    data[:, :, 4, 4] = 65535
    widget = Show4DSTEM(data, center=(4, 4), bf_radius=2)
    widget.roi_mode = "circle"
    widget.roi_center_col = 4
    widget.roi_center_row = 4
    widget.roi_radius = 3
    assert widget.vi_stats[2] < 1000

def test_show4dstem_torch_input():
    """PyTorch tensor input works."""
    import torch
    data = torch.rand(4, 4, 8, 8)
    widget = Show4DSTEM(data)
    assert widget.shape_rows == 4
    assert widget.shape_cols == 4
    assert widget.det_rows == 8
    assert widget.det_cols == 8

def test_show4dstem_position_property():
    """Position property get/set works."""
    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data)
    widget.position = (2, 3)
    assert widget.position == (2, 3)
    assert widget.pos_row == 2
    assert widget.pos_col == 3

def test_show4dstem_scan_shape_property():
    """scan_shape property returns correct tuple."""
    data = np.random.rand(4, 8, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data)
    assert widget.scan_shape == (4, 8)

def test_show4dstem_detector_shape_property():
    """detector_shape property returns correct tuple."""
    data = np.random.rand(4, 4, 12, 16).astype(np.float32)
    widget = Show4DSTEM(data)
    assert widget.detector_shape == (12, 16)

def test_show4dstem_initial_position():
    """Initial position is at scan center."""
    data = np.random.rand(8, 10, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data)
    assert widget.pos_row == 4
    assert widget.pos_col == 5

def test_show4dstem_frame_bytes_nonzero():
    """frame_bytes is non-empty after init."""
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    widget = Show4DSTEM(data)
    assert len(widget.frame_bytes) > 0
    assert len(widget.frame_bytes) == 8 * 8 * 4  # float32

def test_show4dstem_roi_circle_method():
    """roi_circle() sets mode and optional radius."""
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data)
    widget.roi_circle(5.0)
    assert widget.roi_mode == "circle"
    assert widget.roi_radius == 5.0

def test_show4dstem_roi_square_method():
    """roi_square() sets mode and optional half_size."""
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data)
    widget.roi_square(7.0)
    assert widget.roi_mode == "square"
    assert widget.roi_radius == 7.0

def test_show4dstem_roi_annular_method():
    """roi_annular() sets mode and inner/outer radii."""
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data)
    widget.roi_annular(3.0, 10.0)
    assert widget.roi_mode == "annular"
    assert widget.roi_radius_inner == 3.0
    assert widget.roi_radius == 10.0

def test_show4dstem_roi_rect_method():
    """roi_rect() sets mode and width/height."""
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data)
    widget.roi_rect(20.0, 15.0)
    assert widget.roi_mode == "rect"
    assert widget.roi_width == 20.0
    assert widget.roi_height == 15.0

def test_show4dstem_roi_point_method():
    """roi_point() sets mode to point."""
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data)
    widget.roi_point()
    assert widget.roi_mode == "point"

def test_show4dstem_method_chaining():
    """All ROI and playback methods return self for chaining."""
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data)
    assert widget.roi_circle(5) is widget
    assert widget.roi_point() is widget
    assert widget.roi_square(3) is widget
    assert widget.roi_annular(2, 8) is widget
    assert widget.roi_rect(10, 5) is widget
    assert widget.auto_detect_center() is widget
    assert widget.pause() is widget
    assert widget.stop() is widget

def test_show4dstem_path_animation():
    """set_path, play, pause, stop, goto work."""
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    widget = Show4DSTEM(data)
    points = [(0, 0), (1, 1), (2, 2), (3, 3)]
    widget.set_path(points, interval_ms=50, loop=True, autoplay=False)
    assert widget.path_length == 4
    assert widget.path_playing is False
    widget.play()
    assert widget.path_playing is True
    widget.pause()
    assert widget.path_playing is False
    widget.goto(2)
    assert widget.path_index == 2
    widget.stop()
    assert widget.path_index == 0

def test_show4dstem_raster():
    """raster() creates a path covering the scan."""
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    widget = Show4DSTEM(data)
    widget.raster(step=2, interval_ms=50, loop=False)
    assert widget.path_length > 0
    assert widget.path_playing is True  # autoplay by default

def test_show4dstem_calibration():
    """Calibration parameters are stored."""
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data, pixel_size=2.39, k_pixel_size=0.46)
    assert widget.pixel_size == 2.39
    assert widget.k_pixel_size == 0.46
    assert widget.k_calibrated is True

def test_show4dstem_dp_stats():
    """dp_stats has 4 values (mean, min, max, std)."""
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    widget = Show4DSTEM(data)
    assert len(widget.dp_stats) == 4

def test_show4dstem_vi_stats():
    """vi_stats has 4 values (mean, min, max, std)."""
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    widget = Show4DSTEM(data)
    assert len(widget.vi_stats) == 4

def test_show4dstem_rejects_2d():
    """2D input raises ValueError."""
    data = np.random.rand(16, 16).astype(np.float32)
    try:
        Show4DSTEM(data)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def test_show4dstem_rejects_6d():
    """6D input raises ValueError."""
    data = np.random.rand(2, 2, 2, 2, 8, 8).astype(np.float32)
    try:
        Show4DSTEM(data)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def test_show4dstem_non_square_flattened():
    """Non-perfect-square N without scan_shape raises ValueError."""
    data = np.random.rand(7, 8, 8).astype(np.float32)
    try:
        Show4DSTEM(data)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def test_show4dstem_repr():
    """__repr__ returns useful string."""
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data)
    r = repr(widget)
    assert "Show4DSTEM" in r
    assert "4" in r
    assert "16" in r

def test_show4dstem_virtual_image_bytes_nonzero():
    """Virtual image bytes are populated after ROI setup."""
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data)
    widget.roi_circle(5)
    assert len(widget.virtual_image_bytes) > 0

def test_show4dstem_center_explicit():
    """Explicit center and bf_radius are used."""
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    widget = Show4DSTEM(data, center=(5.0, 6.0), bf_radius=3.0)
    assert widget.center_row == 5.0
    assert widget.center_col == 6.0
    assert widget.bf_radius == 3.0

# ── Title ─────────────────────────────────────────────────────────────────

def test_show4dstem_title_default():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    assert w.title == ""

def test_show4dstem_title_set():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data, title="My Scan")
    assert w.title == "My Scan"

def test_show4dstem_title_in_state_dict():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data, title="Test Title")
    sd = w.state_dict()
    assert sd["title"] == "Test Title"
    w2 = Show4DSTEM(data, state=sd)
    assert w2.title == "Test Title"

def test_show4dstem_title_in_summary(capsys):
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data, title="Custom Name")
    w.summary()
    out = capsys.readouterr().out
    assert "Custom Name" in out

def test_show4dstem_title_in_repr():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data, title="My Title")
    r = repr(w)
    assert "My Title" in r

# ── State Protocol ────────────────────────────────────────────────────────

def test_show4dstem_state_dict_roundtrip():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data, center=(5.0, 6.0), bf_radius=3.0, fft_window=False)
    w.dp_scale_mode = "log"
    sd = w.state_dict()
    assert sd["dp_scale_mode"] == "log"
    assert sd["center_row"] == 5.0
    assert sd["center_col"] == 6.0
    assert sd["bf_radius"] == 3.0
    assert "disabled_tools" in sd
    assert "hidden_tools" in sd
    assert sd["fft_window"] is False
    w2 = Show4DSTEM(data, state=sd)
    assert w2.dp_scale_mode == "log"
    assert w2.bf_radius == 3.0
    assert w2.fft_window is False

def test_show4dstem_save_load_file(tmp_path):
    import json
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    w.dp_scale_mode = "log"
    path = tmp_path / "stem_state.json"
    w.save(str(path))
    assert path.exists()
    saved = json.loads(path.read_text())
    assert saved["metadata_version"] == "1.0"
    assert saved["widget_name"] == "Show4DSTEM"
    assert isinstance(saved["widget_version"], str)
    assert saved["state"]["dp_scale_mode"] == "log"
    w2 = Show4DSTEM(data, state=str(path))
    assert w2.dp_scale_mode == "log"

def test_show4dstem_summary(capsys):
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data, pixel_size=2.39, k_pixel_size=0.46)
    w.summary()
    out = capsys.readouterr().out
    assert "Show4DSTEM" in out
    assert "4×4" in out
    assert "16×16" in out
    assert "2.39" in out

def test_show4dstem_set_image():
    data = np.random.rand(4, 4, 32, 32).astype(np.float32)
    widget = Show4DSTEM(data)
    assert widget.shape_rows == 4

    new_data = np.random.rand(8, 8, 64, 64).astype(np.float32)
    widget.set_image(new_data)
    assert widget.shape_rows == 8
    assert widget.shape_cols == 8
    assert widget.det_rows == 64
    assert widget.det_cols == 64

# ── Line Profile ─────────────────────────────────────────────────────────

def test_show4dstem_profile_defaults():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    assert w.profile_line == []
    assert w.profile_width == 1
    assert w.profile == []
    assert w.profile_values is None
    assert w.profile_distance == 0.0

def test_show4dstem_set_profile():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    result = w.set_profile((0, 0), (15, 15))
    assert result is w
    assert len(w.profile_line) == 2
    assert w.profile_line[0] == {"row": 0.0, "col": 0.0}
    assert w.profile_line[1] == {"row": 15.0, "col": 15.0}

def test_show4dstem_clear_profile():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    w.set_profile((0, 0), (15, 15))
    assert len(w.profile_line) == 2
    result = w.clear_profile()
    assert result is w
    assert w.profile_line == []

def test_show4dstem_profile_property():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    w.set_profile((2.0, 3.0), (12.0, 8.0))
    pts = w.profile
    assert len(pts) == 2
    assert pts[0] == (2.0, 3.0)
    assert pts[1] == (12.0, 8.0)

def test_show4dstem_profile_values():
    data = np.ones((4, 4, 16, 16), dtype=np.float32) * 3.0
    w = Show4DSTEM(data)
    w.set_profile((0, 0), (15, 0))
    vals = w.profile_values
    assert vals is not None
    assert len(vals) >= 2
    assert np.allclose(vals, 3.0, atol=0.01)

def test_show4dstem_profile_distance_calibrated():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data, k_pixel_size=0.5)
    w.set_profile((0, 0), (3, 4))
    # pixel distance = 5, k-calibrated = 5 * 0.5 = 2.5
    assert abs(w.profile_distance - 2.5) < 0.01

def test_show4dstem_profile_distance_uncalibrated():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    w.set_profile((0, 0), (3, 4))
    # No k_pixel_size calibration → pixel distance = 5
    assert abs(w.profile_distance - 5.0) < 0.01

def test_show4dstem_profile_in_state_dict():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    w.set_profile((1, 2), (10, 12))
    w.profile_width = 5
    sd = w.state_dict()
    assert "profile_line" in sd
    assert "profile_width" in sd
    assert sd["profile_width"] == 5
    assert len(sd["profile_line"]) == 2

def test_show4dstem_profile_in_summary(capsys):
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    w.set_profile((0, 0), (15, 15))
    w.summary()
    out = capsys.readouterr().out
    assert "Profile:" in out

# ── GIF Export ──────────────────────────────────────────────────────────────

def test_show4dstem_gif_export_defaults():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    assert w._gif_export_requested is False
    assert w._gif_data == b""
    assert w._gif_metadata_json == ""

def test_show4dstem_gif_generation_with_path():
    import json

    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    w.set_path([(0, 0), (1, 1), (2, 2)], autoplay=False)
    w._generate_gif()
    assert len(w._gif_data) > 0
    assert w._gif_data[:3] == b"GIF"
    metadata = json.loads(w._gif_metadata_json)
    assert metadata["widget_name"] == "Show4DSTEM"
    assert metadata["format"] == "gif"
    assert metadata["export_kind"] == "path_animation"
    assert metadata["n_frames"] == 3

def test_show4dstem_gif_generation_no_path():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    w._generate_gif()
    assert w._gif_data == b""
    assert w._gif_metadata_json == ""

def test_show4dstem_normalize_frame():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    frame = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
    result = w._normalize_frame(frame)
    assert result.dtype == np.uint8
    assert result.shape == (2, 2)

# ── Programmatic Image Export ───────────────────────────────────────────────

def test_show4dstem_save_image_png_with_metadata(tmp_path):
    import json

    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    w.dp_colormap = "viridis"
    w.dp_scale_mode = "log"
    w.dp_vmin_pct = 5.0
    w.dp_vmax_pct = 95.0

    out = tmp_path / "dp_view.png"
    written = w.save_image(out, view="diffraction", include_metadata=True)
    assert written == out
    assert out.exists()
    assert out.stat().st_size > 0

    meta = out.with_suffix(".json")
    assert meta.exists()
    saved = json.loads(meta.read_text())
    assert saved["metadata_version"] == "1.0"
    assert saved["widget_name"] == "Show4DSTEM"
    assert isinstance(saved["widget_version"], str)
    assert saved["view"] == "diffraction"
    assert saved["display"]["diffraction"]["colormap"] == "viridis"
    assert saved["display"]["diffraction"]["scale_mode"] == "log"
    assert saved["calibration"]["pixel_size_unit"] == "Å/px"

def test_show4dstem_save_image_pdf_virtual(tmp_path):
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)

    out = tmp_path / "virtual.pdf"
    w.save_image(out, view="virtual", format="pdf", include_metadata=False)
    assert out.exists()
    assert out.stat().st_size > 0

def test_show4dstem_save_image_restore_state(tmp_path):
    data = np.random.rand(3, 4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    w.position = (1, 1)
    w.frame_idx = 0

    out = tmp_path / "frame_override.png"
    w.save_image(out, view="diffraction", position=(3, 2), frame_idx=2, include_metadata=False, restore_state=True)
    assert w.position == (1, 1)
    assert w.frame_idx == 0

def test_show4dstem_save_image_all_includes_fft_metadata(tmp_path):
    import json

    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    w.show_fft = True
    out = tmp_path / "all_panels.png"
    w.save_image(out, view="all")
    saved = json.loads(out.with_suffix(".json").read_text())
    assert "diffraction" in saved["display"]
    assert "virtual" in saved["display"]
    assert "fft" in saved["display"]

def test_show4dstem_save_image_rejects_unknown_format(tmp_path):
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    try:
        w.save_image(tmp_path / "bad.tiff", view="diffraction")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def test_show4dstem_save_image_rejects_view_aliases(tmp_path):
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)

    for alias in ("dp", "reconstruction", "vi", "virtual_image", "composite"):
        try:
            w.save_image(tmp_path / f"{alias}.png", view=alias, include_metadata=False)
            assert False, f"Should reject alias view '{alias}'"
        except ValueError:
            pass

def test_show4dstem_export_view_and_format_lists():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    assert w.list_export_views() == ("diffraction", "virtual", "fft", "all")
    assert w.list_export_formats() == ("png", "pdf")

def test_show4dstem_save_image_overlay_scalebar_flags_in_metadata(tmp_path):
    import json

    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    out = tmp_path / "flags.png"
    w.save_image(
        out,
        view="diffraction",
        include_metadata=True,
        include_overlays=False,
        include_scalebar=False,
    )
    metadata = json.loads(out.with_suffix(".json").read_text())
    assert metadata["include_overlays"] is False
    assert metadata["include_scalebar"] is False

def test_show4dstem_save_sequence_frames_manifest(tmp_path):
    import json

    data = np.random.rand(3, 4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    manifest = w.save_sequence(
        tmp_path / "seq",
        mode="frames",
        view="virtual",
        format="png",
        frame_indices=[0, 2],
        position=(1, 2),
    )
    assert manifest.exists()
    payload = json.loads(manifest.read_text())
    assert payload["export_kind"] == "sequence_batch"
    assert payload["mode"] == "frames"
    assert payload["n_exports"] == 2
    assert len(payload["exports"]) == 2
    for row in payload["exports"]:
        path = tmp_path / "seq" / pathlib.Path(row["path"]).name
        assert path.exists()
        assert "sha256" in row

def test_show4dstem_save_sequence_rejects_bad_mode(tmp_path):
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    try:
        w.save_sequence(tmp_path / "bad", mode="diagonal")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def test_show4dstem_preset_api_roundtrip(tmp_path):
    import json

    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data, center=(8.0, 8.0), bf_radius=3.0)

    w.apply_preset("adf")
    assert w.roi_mode == "annular"
    assert w.roi_radius > w.roi_radius_inner

    w.dp_colormap = "viridis"
    w.export_default_view = "all"
    w.save_preset("workflow_a")
    w.dp_colormap = "inferno"
    w.load_preset("workflow_a")
    assert w.dp_colormap == "viridis"
    assert w.export_default_view == "all"

    preset_path = tmp_path / "workflow_a.json"
    w.save_preset("workflow_a", path=preset_path)
    assert preset_path.exists()
    payload = json.loads(preset_path.read_text())
    assert payload["export_kind"] == "widget_preset"
    assert payload["preset_name"] == "workflow_a"

def test_show4dstem_save_figure_template(tmp_path):
    import json

    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    out = tmp_path / "figure.png"
    w.save_figure(
        out,
        template="publication_dp_vi_fft",
        include_metadata=True,
        annotations={"diffraction": "BF region"},
    )
    assert out.exists()
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["export_kind"] == "figure_template"
    assert meta["template"] == "publication_dp_vi_fft"
    assert meta["publication_style"] is True

def test_show4dstem_save_reproducibility_report(tmp_path):
    import json

    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    w.save_image(tmp_path / "one.png", view="diffraction")
    w.save_figure(tmp_path / "two.png", template="dp_vi")
    report_path = w.save_reproducibility_report(tmp_path / "report.json")
    payload = json.loads(report_path.read_text())
    assert payload["export_kind"] == "reproducibility_report"
    assert payload["n_exports"] >= 2
    assert len(payload["exports"]) >= 2
    assert all("session_id" in row for row in payload["exports"])

def test_show4dstem_suggest_adaptive_path_basic():
    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    result = w.suggest_adaptive_path(
        coarse_step=4,
        target_fraction=0.4,
        min_spacing=1,
        update_widget_path=True,
        autoplay=False,
    )
    assert result["coarse_count"] > 0
    assert result["path_count"] >= result["coarse_count"]
    assert len(result["path_points"]) == result["path_count"]
    assert len(set(result["path_points"])) == result["path_count"]
    assert w.path_length == result["path_count"]

def test_show4dstem_suggest_adaptive_path_rejects_small_target():
    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    try:
        w.suggest_adaptive_path(coarse_step=2, target_fraction=0.05)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def test_show4dstem_suggest_adaptive_path_respects_roi_mask():
    data = np.random.rand(10, 10, 16, 16).astype(np.float32)
    w = Show4DSTEM(data)
    mask = np.zeros((10, 10), dtype=bool)
    mask[0:5, 0:5] = True
    result = w.suggest_adaptive_path(
        coarse_step=5,
        target_fraction=0.1,
        include_coarse=False,
        roi_mask=mask,
        update_widget_path=False,
    )
    assert result["dense_count"] > 0
    for row, col in result["dense_points"]:
        assert mask[row, col]

def test_show4dstem_suggest_adaptive_path_return_maps():
    data = np.random.rand(6, 6, 8, 8).astype(np.float32)
    w = Show4DSTEM(data)
    result = w.suggest_adaptive_path(
        coarse_step=3,
        target_fraction=0.3,
        update_widget_path=False,
        return_maps=True,
    )
    assert "utility_map" in result
    assert result["utility_map"].shape == (6, 6)
    assert "utility_components" in result

# ── Sparse Ingest / Adaptive Loop APIs ─────────────────────────────────────

def test_show4dstem_ingest_scan_point_and_get_sparse_state():
    data = np.zeros((6, 6, 8, 8), dtype=np.float32)
    w = Show4DSTEM(data)
    dp = np.ones((8, 8), dtype=np.float32) * 7.0
    w.ingest_scan_point(2, 3, dp, dose=2.5)

    state = w.get_sparse_state()
    assert state["n_sampled"] == 1
    assert state["mask"].shape == (6, 6)
    assert state["mask"][2, 3]
    assert state["sampled_data"].shape == (1, 8, 8)
    assert np.allclose(state["sampled_data"][0], 7.0)
    assert state["total_dose"] == pytest.approx(2.5)
    assert np.allclose(w._get_frame(2, 3), 7.0)

def test_show4dstem_ingest_scan_block_and_set_sparse_state_roundtrip():
    data = np.zeros((5, 5, 8, 8), dtype=np.float32)
    w = Show4DSTEM(data)

    rows = [0, 2, 4]
    cols = [1, 2, 3]
    block = np.stack(
        [
            np.full((8, 8), 1.0, dtype=np.float32),
            np.full((8, 8), 2.0, dtype=np.float32),
            np.full((8, 8), 3.0, dtype=np.float32),
        ],
        axis=0,
    )
    w.ingest_scan_block(rows, cols, block)

    state = w.get_sparse_state()
    assert state["n_sampled"] == 3
    assert state["mask"].sum() == 3

    w2 = Show4DSTEM(np.zeros_like(data))
    w2.set_sparse_state(state["mask"], state["sampled_data"])
    state2 = w2.get_sparse_state()
    assert state2["n_sampled"] == 3
    assert np.array_equal(state2["mask"], state["mask"])
    assert np.allclose(w2._get_frame(0, 1), 1.0)
    assert np.allclose(w2._get_frame(2, 2), 2.0)
    assert np.allclose(w2._get_frame(4, 3), 3.0)

def test_show4dstem_propose_next_points_respects_budget_and_existing():
    data = np.random.rand(6, 6, 8, 8).astype(np.float32)
    w = Show4DSTEM(data)
    w.ingest_scan_point(0, 0, w._get_frame(0, 0))

    proposed = w.propose_next_points(
        5,
        strategy="adaptive",
        budget={"max_total_points": 3, "min_spacing": 1},
    )
    assert len(proposed) == 2  # Existing=1 and max_total_points=3
    assert (0, 0) not in proposed

    roi_mask = np.zeros((6, 6), dtype=bool)
    roi_mask[4:, 4:] = True
    proposed_random = w.propose_next_points(
        4,
        strategy="random",
        budget={"roi_mask": roi_mask, "seed": 42},
    )
    assert all(roi_mask[r, c] for r, c in proposed_random)

def test_show4dstem_evaluate_against_reference_outputs_metrics():
    data = np.random.rand(6, 6, 8, 8).astype(np.float32)
    w = Show4DSTEM(data)
    for row, col in [(0, 0), (0, 3), (2, 1), (3, 4), (5, 5), (4, 2), (1, 5), (5, 1)]:
        w.ingest_scan_point(row, col, w._get_frame(row, col))

    report = w.evaluate_against_reference(reference="full_raster")
    assert report["reference_kind"] == "full_raster"
    assert report["n_sampled"] == 8
    assert report["sampled_fraction"] > 0
    for key in ("rmse", "nrmse", "mae", "psnr"):
        assert key in report["metrics"]
        assert np.isfinite(report["metrics"][key])

def test_show4dstem_export_session_bundle(tmp_path):
    import json

    data = np.random.rand(5, 5, 8, 8).astype(np.float32)
    w = Show4DSTEM(data)
    w.ingest_scan_point(1, 1, w._get_frame(1, 1), dose=1.5)
    w.ingest_scan_point(3, 2, w._get_frame(3, 2), dose=0.5)

    manifest_path = w.export_session_bundle(tmp_path / "bundle")
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text())
    assert payload["export_kind"] == "session_bundle"
    assert payload["sparse_summary"]["n_sampled"] == 2
    for path in payload["files"].values():
        assert pathlib.Path(path).exists()

# ── 5D Time/Tilt Series ────────────────────────────────────────────────────

def test_show4dstem_5d_basic():
    """5D array creates widget with n_frames > 1."""
    data = np.random.rand(5, 4, 4, 8, 8).astype(np.float32)
    w = Show4DSTEM(data)
    assert w.n_frames == 5
    assert w.shape_rows == 4
    assert w.shape_cols == 4
    assert w.det_rows == 8
    assert w.det_cols == 8
    assert w.frame_idx == 0

def test_show4dstem_5d_frame_navigation():
    """Changing frame_idx updates the displayed frame."""
    data = np.zeros((3, 2, 2, 4, 4), dtype=np.float32)
    data[0] = 1.0
    data[1] = 2.0
    data[2] = 3.0
    w = Show4DSTEM(data)
    frame0 = w._get_frame(0, 0)
    assert np.allclose(frame0, 1.0)
    w.frame_idx = 1
    frame1 = w._get_frame(0, 0)
    assert np.allclose(frame1, 2.0)
    w.frame_idx = 2
    frame2 = w._get_frame(0, 0)
    assert np.allclose(frame2, 3.0)

def test_show4dstem_5d_frame_dim_label():
    """frame_dim_label is set from constructor param."""
    data = np.random.rand(3, 2, 2, 4, 4).astype(np.float32)
    w = Show4DSTEM(data, frame_dim_label="Tilt")
    assert w.frame_dim_label == "Tilt"
    w2 = Show4DSTEM(data)
    assert w2.frame_dim_label == "Frame"

def test_show4dstem_5d_virtual_image_per_frame():
    """Virtual image changes when frame_idx changes."""
    data = np.zeros((2, 4, 4, 8, 8), dtype=np.float32)
    data[0, :, :, 3:5, 3:5] = 10.0
    data[1, :, :, 3:5, 3:5] = 50.0
    w = Show4DSTEM(data, center=(4, 4), bf_radius=3)
    w.roi_mode = "circle"
    w.roi_center_row = 4.0
    w.roi_center_col = 4.0
    w.roi_radius = 3.0
    vi_bytes_0 = bytes(w.virtual_image_bytes)
    w.frame_idx = 1
    vi_bytes_1 = bytes(w.virtual_image_bytes)
    assert vi_bytes_0 != vi_bytes_1

def test_show4dstem_5d_global_range():
    """dp_global_min/max spans all frames."""
    data = np.zeros((3, 2, 2, 4, 4), dtype=np.float32)
    data[0] = 1.0
    data[1] = 5.0
    data[2] = 10.0
    w = Show4DSTEM(data)
    assert w.dp_global_min <= 1.0
    assert w.dp_global_max >= 10.0

def test_show4dstem_5d_set_image():
    """set_image works with 5D data."""
    data_4d = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Show4DSTEM(data_4d)
    assert w.n_frames == 1
    data_5d = np.random.rand(3, 4, 4, 8, 8).astype(np.float32)
    w.set_image(data_5d)
    assert w.n_frames == 3
    assert w.frame_idx == 0

def test_show4dstem_5d_state_dict():
    """state_dict includes frame traits."""
    data = np.random.rand(3, 2, 2, 4, 4).astype(np.float32)
    w = Show4DSTEM(data, frame_dim_label="Time")
    w.frame_idx = 1
    sd = w.state_dict()
    assert sd["frame_idx"] == 1
    assert sd["frame_dim_label"] == "Time"
    assert "frame_loop" in sd
    assert "frame_fps" in sd
    assert "frame_reverse" in sd
    assert "frame_boomerang" in sd

def test_show4dstem_5d_state_roundtrip():
    """State can be saved and restored for 5D widget."""
    data = np.random.rand(3, 2, 2, 4, 4).astype(np.float32)
    w = Show4DSTEM(data, frame_dim_label="Tilt")
    w.frame_idx = 2
    w.frame_fps = 10.0
    w.frame_reverse = True
    w.frame_boomerang = True
    sd = w.state_dict()
    w2 = Show4DSTEM(data, state=sd)
    assert w2.frame_idx == 2
    assert w2.frame_dim_label == "Tilt"
    assert w2.frame_fps == 10.0
    assert w2.frame_reverse is True
    assert w2.frame_boomerang is True

def test_show4dstem_5d_summary(capsys):
    """summary() shows frame info for 5D data."""
    data = np.random.rand(5, 2, 2, 4, 4).astype(np.float32)
    w = Show4DSTEM(data, frame_dim_label="Tilt")
    w.frame_idx = 2
    w.summary()
    out = capsys.readouterr().out
    assert "Frames:" in out
    assert "Tilt" in out

def test_show4dstem_5d_repr():
    """__repr__ includes frame info for 5D data."""
    data = np.random.rand(3, 2, 2, 4, 4).astype(np.float32)
    w = Show4DSTEM(data, frame_dim_label="Focus")
    r = repr(w)
    assert "3," in r  # n_frames in shape
    assert "focus=" in r  # frame_dim_label.lower()

def test_show4dstem_4d_no_frame_traits():
    """4D data keeps n_frames=1, no frame info in repr."""
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Show4DSTEM(data)
    assert w.n_frames == 1
    assert w.frame_idx == 0
    r = repr(w)
    assert "frame" not in r.lower() or "frame" not in r

def test_show4dstem_5d_torch_input():
    """5D PyTorch tensor input works."""
    import torch
    data = torch.rand(3, 4, 4, 8, 8)
    w = Show4DSTEM(data)
    assert w.n_frames == 3
    assert w.shape_rows == 4
    assert w.det_rows == 8

# ── Tool visibility / locking ────────────────────────────────────────────

@pytest.mark.parametrize("trait_name", ["disabled_tools", "hidden_tools"])
def test_show4dstem_tool_lists_default_empty(trait_name):
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data, scan_shape=(4, 4))
    assert getattr(w, trait_name) == []

@pytest.mark.parametrize(
    ("trait_name", "ctor_kwargs", "expected"),
    [
        ("disabled_tools", {"disabled_tools": ["display", "ROI", "histogram", "playback"]}, ["display", "roi", "histogram", "playback"]),
        ("hidden_tools", {"hidden_tools": ["display", "ROI", "histogram", "playback"]}, ["display", "roi", "histogram", "playback"]),
        ("disabled_tools", {"disable_display": True, "disable_fft": True, "disable_histogram": True, "disable_playback": True}, ["display", "histogram", "playback", "fft"]),
        ("hidden_tools", {"hide_display": True, "hide_fft": True, "hide_histogram": True, "hide_playback": True}, ["display", "histogram", "playback", "fft"]),
        ("disabled_tools", {"disable_all": True}, ["all"]),
        ("hidden_tools", {"hide_all": True}, ["all"]),
    ],
)
def test_show4dstem_tool_lists_constructor_behavior(trait_name, ctor_kwargs, expected):
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data, scan_shape=(4, 4), **ctor_kwargs)
    assert getattr(w, trait_name) == expected

@pytest.mark.parametrize("kwargs", [{"disabled_tools": ["not_real"]}, {"hidden_tools": ["not_real"]}])
def test_show4dstem_tool_lists_unknown_raises(kwargs):
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    with pytest.raises(ValueError, match="Unknown tool group"):
        Show4DSTEM(data, scan_shape=(4, 4), **kwargs)

@pytest.mark.parametrize("trait_name", ["disabled_tools", "hidden_tools"])
def test_show4dstem_tool_lists_normalizes(trait_name):
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data, scan_shape=(4, 4))
    setattr(w, trait_name, ["DISPLAY", "display", "roi"])
    assert getattr(w, trait_name) == ["display", "roi"]

def test_show4dstem_widget_version_is_set():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data, scan_shape=(4, 4))
    assert w.widget_version != "unknown"

def test_show4dstem_show_controls_default():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4DSTEM(data, scan_shape=(4, 4))
    assert w.show_controls is True
