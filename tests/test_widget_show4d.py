import json
import numpy as np
import pytest
import quantem.widget
from quantem.widget import Show4D

def test_version_exists():
    assert hasattr(quantem.widget, "__version__")

def test_show4d_loads():
    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    widget = Show4D(data)
    assert widget is not None

def test_show4d_shape_traits():
    data = np.random.rand(4, 8, 12, 16).astype(np.float32)
    widget = Show4D(data)
    assert widget.nav_rows == 4
    assert widget.nav_cols == 8
    assert widget.sig_rows == 12
    assert widget.sig_cols == 16

def test_show4d_initial_position():
    data = np.random.rand(8, 10, 16, 16).astype(np.float32)
    widget = Show4D(data)
    assert widget.pos_row == 4
    assert widget.pos_col == 5

def test_show4d_frame_bytes_nonzero():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    widget = Show4D(data)
    assert len(widget.frame_bytes) == 8 * 8 * 4  # float32

def test_show4d_nav_image_bytes_nonzero():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    widget = Show4D(data)
    assert len(widget.nav_image_bytes) == 4 * 4 * 4  # float32

def test_show4d_default_nav_image_is_mean():
    data = np.ones((4, 4, 8, 8), dtype=np.float32) * 42
    widget = Show4D(data)
    nav_arr = np.frombuffer(widget.nav_image_bytes, dtype=np.float32)
    assert np.allclose(nav_arr, 42.0)

def test_show4d_nav_image_override():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    nav = np.ones((4, 4), dtype=np.float32) * 99
    widget = Show4D(data, nav_image=nav)
    nav_arr = np.frombuffer(widget.nav_image_bytes, dtype=np.float32)
    assert np.allclose(nav_arr, 99.0)

def test_show4d_position_change_updates_frame():
    data = np.zeros((4, 4, 8, 8), dtype=np.float32)
    data[0, 0] = 1.0
    data[1, 1] = 2.0
    widget = Show4D(data)
    widget.pos_row = 0
    widget.pos_col = 0
    frame0 = bytes(widget.frame_bytes)
    widget.pos_row = 1
    widget.pos_col = 1
    frame1 = bytes(widget.frame_bytes)
    assert frame0 != frame1

def test_show4d_position_property():
    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    widget = Show4D(data)
    widget.position = (2, 3)
    assert widget.position == (2, 3)
    assert widget.pos_row == 2
    assert widget.pos_col == 3

def test_show4d_roi_circle():
    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    widget = Show4D(data)
    widget.roi_mode = "circle"
    widget.roi_center_row = 4
    widget.roi_center_col = 4
    widget.roi_radius = 3
    assert len(widget.frame_bytes) == 16 * 16 * 4
    assert len(widget.sig_stats) == 4

def test_show4d_roi_square():
    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    widget = Show4D(data)
    widget.roi_mode = "square"
    widget.roi_center_row = 4
    widget.roi_center_col = 4
    widget.roi_radius = 2
    assert len(widget.frame_bytes) > 0

def test_show4d_roi_rect():
    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    widget = Show4D(data)
    widget.roi_mode = "rect"
    widget.roi_center_row = 4
    widget.roi_center_col = 4
    widget.roi_width = 4
    widget.roi_height = 6
    assert len(widget.frame_bytes) > 0

def test_show4d_roi_off_falls_back():
    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    widget = Show4D(data)
    widget.roi_mode = "circle"
    widget.roi_mode = "off"
    assert len(widget.frame_bytes) == 16 * 16 * 4

def test_show4d_nav_stats():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    widget = Show4D(data)
    assert len(widget.nav_stats) == 4
    assert widget.nav_stats[2] >= widget.nav_stats[1]  # max >= min

def test_show4d_sig_stats():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    widget = Show4D(data)
    assert len(widget.sig_stats) == 4
    assert widget.sig_stats[2] >= widget.sig_stats[1]  # max >= min

def test_show4d_pixel_size():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    widget = Show4D(data, nav_pixel_size=2.5, sig_pixel_size=0.5, nav_pixel_unit="Å", sig_pixel_unit="mrad")
    assert widget.nav_pixel_size == 2.5
    assert widget.sig_pixel_size == 0.5
    assert widget.nav_pixel_unit == "Å"
    assert widget.sig_pixel_unit == "mrad"

def test_show4d_torch_input():
    import torch
    data = torch.rand(4, 4, 8, 8)
    widget = Show4D(data)
    assert widget.nav_rows == 4
    assert widget.sig_rows == 8

def test_show4d_rejects_2d():
    with pytest.raises(ValueError):
        Show4D(np.random.rand(16, 16).astype(np.float32))

def test_show4d_rejects_3d():
    with pytest.raises(ValueError):
        Show4D(np.random.rand(4, 16, 16).astype(np.float32))

def test_show4d_rejects_5d():
    with pytest.raises(ValueError):
        Show4D(np.random.rand(2, 2, 2, 8, 8).astype(np.float32))

def test_show4d_title():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    widget = Show4D(data, title="Test 4D")
    assert widget.title == "Test 4D"

def test_show4d_repr():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    widget = Show4D(data)
    r = repr(widget)
    assert "Show4D" in r
    assert "4" in r
    assert "16" in r

def test_show4d_nav_shape_property():
    data = np.random.rand(4, 8, 16, 16).astype(np.float32)
    widget = Show4D(data)
    assert widget.nav_shape == (4, 8)

def test_show4d_sig_shape_property():
    data = np.random.rand(4, 4, 12, 16).astype(np.float32)
    widget = Show4D(data)
    assert widget.sig_shape == (12, 16)

def test_show4d_data_ranges():
    data = np.ones((4, 4, 8, 8), dtype=np.float32) * 42
    widget = Show4D(data)
    assert widget.nav_data_min > 0
    assert widget.sig_data_min == 42.0
    assert widget.sig_data_max == 42.0

def test_show4d_roi_empty_mask_falls_back():
    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    widget = Show4D(data)
    widget.roi_mode = "circle"
    widget.roi_center_row = -100
    widget.roi_center_col = -100
    widget.roi_radius = 1
    # Should fall back to single position frame
    assert len(widget.frame_bytes) == 16 * 16 * 4

def test_show4d_roi_reduce_default():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    widget = Show4D(data)
    assert widget.roi_reduce == "mean"

def test_show4d_roi_reduce_mean():
    data = np.zeros((4, 4, 8, 8), dtype=np.float32)
    data[0, 0] = 2.0
    data[0, 1] = 4.0
    widget = Show4D(data)
    widget.roi_mode = "square"
    widget.roi_center_row = 0
    widget.roi_center_col = 0
    widget.roi_radius = 1
    widget.roi_reduce = "mean"
    frame = np.frombuffer(widget.frame_bytes, dtype=np.float32)
    # Mean of positions within radius 1 of (0,0) should be between 0 and 4
    assert frame.max() <= 4.0

def test_show4d_roi_reduce_max():
    data = np.zeros((4, 4, 8, 8), dtype=np.float32)
    data[1, 1] = 10.0
    data[2, 2] = 5.0
    widget = Show4D(data)
    widget.roi_mode = "circle"
    widget.roi_center_row = 1.5
    widget.roi_center_col = 1.5
    widget.roi_radius = 2
    widget.roi_reduce = "max"
    frame = np.frombuffer(widget.frame_bytes, dtype=np.float32)
    assert frame.max() == 10.0

def test_show4d_roi_reduce_min():
    data = np.ones((4, 4, 8, 8), dtype=np.float32) * 5.0
    data[1, 1] = 1.0
    widget = Show4D(data)
    widget.roi_mode = "square"
    widget.roi_center_row = 1
    widget.roi_center_col = 1
    widget.roi_radius = 1
    widget.roi_reduce = "min"
    frame = np.frombuffer(widget.frame_bytes, dtype=np.float32)
    assert frame.min() == 1.0

def test_show4d_roi_reduce_sum():
    data = np.ones((4, 4, 8, 8), dtype=np.float32)
    widget = Show4D(data)
    widget.roi_mode = "square"
    widget.roi_center_row = 1
    widget.roi_center_col = 1
    widget.roi_radius = 1
    widget.roi_reduce = "sum"
    frame = np.frombuffer(widget.frame_bytes, dtype=np.float32)
    # Sum of N positions, each with value 1.0 → frame values should be > 1
    assert frame.max() > 1.0

def test_show4d_roi_reduce_change_updates_frame():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    widget = Show4D(data)
    widget.roi_mode = "circle"
    widget.roi_center_row = 2
    widget.roi_center_col = 2
    widget.roi_radius = 2
    widget.roi_reduce = "mean"
    frame_mean = bytes(widget.frame_bytes)
    widget.roi_reduce = "max"
    frame_max = bytes(widget.frame_bytes)
    assert frame_mean != frame_max

def test_show4d_snap_defaults():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    widget = Show4D(data)
    assert widget.snap_enabled is False
    assert widget.snap_radius == 5

def test_show4d_snap_enabled():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    widget = Show4D(data, snap_enabled=True, snap_radius=10)
    assert widget.snap_enabled is True
    assert widget.snap_radius == 10

def test_show4d_snap_toggle():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    widget = Show4D(data)
    widget.snap_enabled = True
    assert widget.snap_enabled is True
    widget.snap_radius = 3
    assert widget.snap_radius == 3

# ── State Protocol ────────────────────────────────────────────────────────

def test_show4d_state_dict_roundtrip():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Show4D(data, cmap="viridis", log_scale=True, title="4D Data",
               snap_enabled=True, snap_radius=8, show_fft=True, fft_window=False,
               disabled_tools=["display"], hidden_tools=["roi"])
    sd = w.state_dict()
    assert "disabled_tools" in sd
    assert "hidden_tools" in sd
    assert "fft_window" in sd
    w2 = Show4D(data, state=sd)
    assert w2.cmap == "viridis"
    assert w2.log_scale is True
    assert w2.title == "4D Data"
    assert w2.snap_enabled is True
    assert w2.snap_radius == 8
    assert w2.fft_window is False
    assert w2.disabled_tools == ["display"]
    assert w2.hidden_tools == ["roi"]

def test_show4d_save_load_file(tmp_path):
    import json
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Show4D(data, cmap="plasma", title="Saved 4D")
    path = tmp_path / "show4d_state.json"
    w.save(str(path))
    assert path.exists()
    saved = json.loads(path.read_text())
    assert saved["metadata_version"] == "1.0"
    assert saved["widget_name"] == "Show4D"
    assert isinstance(saved["widget_version"], str)
    assert saved["state"]["cmap"] == "plasma"
    w2 = Show4D(data, state=str(path))
    assert w2.cmap == "plasma"
    assert w2.title == "Saved 4D"

def test_show4d_summary(capsys):
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Show4D(data, title="My 4D", cmap="inferno")
    w.summary()
    out = capsys.readouterr().out
    assert "My 4D" in out
    assert "4×4" in out
    assert "inferno" in out

def test_show4d_repr():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Show4D(data)
    r = repr(w)
    assert "Show4D" in r
    assert "4, 4, 8, 8" in r

def test_show4d_set_image():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    widget = Show4D(data, cmap="viridis")
    assert widget.nav_rows == 4

    new_data = np.random.rand(6, 6, 16, 16).astype(np.float32)
    widget.set_image(new_data)
    assert widget.nav_rows == 6
    assert widget.nav_cols == 6
    assert widget.sig_rows == 16
    assert widget.sig_cols == 16
    assert widget.cmap == "viridis"

# ── Line Profile ─────────────────────────────────────────────────────────

def test_show4d_profile_defaults():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Show4D(data)
    assert w.profile_line == []
    assert w.profile_width == 1
    assert w.profile == []
    assert w.profile_values is None
    assert w.profile_distance == 0.0

def test_show4d_set_profile():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Show4D(data)
    result = w.set_profile((0, 0), (7, 7))
    assert result is w  # method chaining
    assert len(w.profile_line) == 2
    assert w.profile_line[0] == {"row": 0.0, "col": 0.0}
    assert w.profile_line[1] == {"row": 7.0, "col": 7.0}

def test_show4d_clear_profile():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Show4D(data)
    w.set_profile((0, 0), (7, 7))
    assert len(w.profile_line) == 2
    result = w.clear_profile()
    assert result is w
    assert w.profile_line == []

def test_show4d_profile_property():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Show4D(data)
    w.set_profile((1.5, 2.5), (6.5, 3.5))
    pts = w.profile
    assert len(pts) == 2
    assert pts[0] == (1.5, 2.5)
    assert pts[1] == (6.5, 3.5)

def test_show4d_profile_values():
    data = np.ones((4, 4, 8, 8), dtype=np.float32) * 5.0
    w = Show4D(data)
    w.set_profile((0, 0), (7, 0))
    vals = w.profile_values
    assert vals is not None
    assert len(vals) >= 2
    assert np.allclose(vals, 5.0, atol=0.01)

def test_show4d_profile_distance():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Show4D(data, sig_pixel_size=2.0, sig_pixel_unit="Å")
    w.set_profile((0, 0), (3, 4))
    # pixel distance = sqrt(3^2 + 4^2) = 5, physical = 5 * 2.0 = 10.0
    assert abs(w.profile_distance - 10.0) < 0.01

def test_show4d_profile_in_state_dict():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Show4D(data)
    w.set_profile((1, 2), (5, 6))
    w.profile_width = 3
    sd = w.state_dict()
    assert "profile_line" in sd
    assert "profile_width" in sd
    assert sd["profile_width"] == 3
    assert len(sd["profile_line"]) == 2

def test_show4d_profile_in_summary(capsys):
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Show4D(data, title="Profile Test")
    w.set_profile((0, 0), (7, 7))
    w.summary()
    out = capsys.readouterr().out
    assert "Profile:" in out

# ── GIF Export ──────────────────────────────────────────────────────────────

def test_show4d_gif_export_defaults():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Show4D(data)
    assert w._gif_export_requested is False
    assert w._gif_data == b""
    assert w._gif_metadata_json == ""

def test_show4d_gif_generation_with_path():
    import json

    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Show4D(data, cmap="viridis")
    w.set_path([(0, 0), (1, 1), (2, 2)], autoplay=False)
    w._generate_gif()
    assert len(w._gif_data) > 0
    # GIF magic bytes
    assert w._gif_data[:3] == b"GIF"
    metadata = json.loads(w._gif_metadata_json)
    assert metadata["widget_name"] == "Show4D"
    assert metadata["format"] == "gif"
    assert metadata["export_kind"] == "path_animation"
    assert metadata["n_frames"] == 3

def test_show4d_gif_generation_no_path():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Show4D(data)
    w._generate_gif()
    assert w._gif_data == b""
    assert w._gif_metadata_json == ""

def test_show4d_normalize_frame():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Show4D(data)
    frame = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
    result = w._normalize_frame(frame)
    assert result.dtype == np.uint8
    assert result.shape == (2, 2)
    assert result.max() <= 255
    assert result.min() >= 0

# ── Tool Visibility / Locking ────────────────────────────────────────────

@pytest.mark.parametrize("trait_name", ["disabled_tools", "hidden_tools"])
def test_show4d_tool_lists_default_empty(trait_name):
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4D(data)
    assert getattr(w, trait_name) == []

@pytest.mark.parametrize(
    ("trait_name", "ctor_kwargs", "expected"),
    [
        ("disabled_tools", {"disabled_tools": ["display", "ROI", "playback", "FFT"]}, ["display", "roi", "playback", "fft"]),
        ("hidden_tools", {"hidden_tools": ["display", "ROI", "playback", "FFT"]}, ["display", "roi", "playback", "fft"]),
        ("disabled_tools", {"disable_display": True, "disable_roi": True, "disable_playback": True, "disable_fft": True}, ["display", "roi", "playback", "fft"]),
        ("hidden_tools", {"hide_display": True, "hide_roi": True, "hide_playback": True, "hide_fft": True}, ["display", "roi", "playback", "fft"]),
        ("disabled_tools", {"disable_all": True}, ["all"]),
        ("hidden_tools", {"hide_all": True}, ["all"]),
    ],
)
def test_show4d_tool_lists_constructor_behavior(trait_name, ctor_kwargs, expected):
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4D(data, **ctor_kwargs)
    assert getattr(w, trait_name) == expected

@pytest.mark.parametrize("kwargs", [{"disabled_tools": ["not_real"]}, {"hidden_tools": ["not_real"]}])
def test_show4d_tool_lists_unknown_raises(kwargs):
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    with pytest.raises(ValueError, match="Unknown tool group"):
        Show4D(data, **kwargs)

@pytest.mark.parametrize("trait_name", ["disabled_tools", "hidden_tools"])
def test_show4d_tool_lists_normalizes(trait_name):
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4D(data)
    setattr(w, trait_name, ["DISPLAY", "display", "roi"])
    assert getattr(w, trait_name) == ["display", "roi"]

# ── save_image ───────────────────────────────────────────────────────────

def test_show4d_save_image_signal(tmp_path):
    data = np.random.rand(4, 4, 32, 32).astype(np.float32)
    w = Show4D(data)
    out = w.save_image(tmp_path / "signal.png")
    assert out.exists()
    from PIL import Image
    img = Image.open(out)
    assert img.size == (32, 32)

def test_show4d_save_image_nav(tmp_path):
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4D(data)
    out = w.save_image(tmp_path / "nav.png", view="nav")
    assert out.exists()

def test_show4d_save_image_position(tmp_path):
    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    w = Show4D(data)
    out = w.save_image(tmp_path / "pos.png", position=(3, 5))
    assert out.exists()

def test_show4d_save_image_pdf(tmp_path):
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4D(data)
    out = w.save_image(tmp_path / "out.pdf")
    assert out.exists()

def test_show4d_save_image_bad_format(tmp_path):
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Show4D(data)
    with pytest.raises(ValueError, match="Unsupported format"):
        w.save_image(tmp_path / "out.bmp")

def test_show4d_widget_version_is_set():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Show4D(data)
    assert w.widget_version != "unknown"

def test_show4d_show_controls_default():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Show4D(data)
    assert w.show_controls is True
