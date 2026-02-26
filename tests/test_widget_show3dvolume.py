import numpy as np
import pytest
import torch

from quantem.widget import Show3DVolume

def test_show3dvolume_numpy():
    """Create widget from numpy array."""
    data = np.random.rand(16, 16, 16).astype(np.float32)
    widget = Show3DVolume(data)
    assert widget.nz == 16
    assert widget.ny == 16
    assert widget.nx == 16
    assert len(widget.volume_bytes) > 0

def test_show3dvolume_torch():
    """Create widget from PyTorch tensor."""
    data = torch.rand(16, 16, 16)
    widget = Show3DVolume(data)
    assert widget.nz == 16
    assert widget.ny == 16
    assert widget.nx == 16

def test_show3dvolume_initial_slices():
    """Initial slices are at middle."""
    data = np.random.rand(20, 30, 40).astype(np.float32)
    widget = Show3DVolume(data)
    assert widget.slice_z == 10
    assert widget.slice_y == 15
    assert widget.slice_x == 20

def test_show3dvolume_stats():
    """Statistics are computed for 3 orthogonal slices."""
    data = np.ones((16, 16, 16), dtype=np.float32) * 5.0
    widget = Show3DVolume(data)
    assert len(widget.stats_mean) == 3
    assert len(widget.stats_min) == 3
    assert len(widget.stats_max) == 3
    assert len(widget.stats_std) == 3
    for mean in widget.stats_mean:
        assert mean == pytest.approx(5.0)

def test_show3dvolume_rejects_2d():
    """Raises error for 2D input."""
    data = np.random.rand(16, 16).astype(np.float32)
    with pytest.raises(ValueError, match="3D"):
        Show3DVolume(data)

def test_show3dvolume_options():
    """Display options are applied."""
    data = np.random.rand(16, 16, 16).astype(np.float32)
    widget = Show3DVolume(
        data,
        title="Test Volume",
        cmap="viridis",
        log_scale=True,
        auto_contrast=True,
    )
    assert widget.title == "Test Volume"
    assert widget.cmap == "viridis"
    assert widget.log_scale is True
    assert widget.auto_contrast is True

def test_show3dvolume_non_cubic():
    """Non-cubic volumes work correctly."""
    data = np.random.rand(10, 20, 30).astype(np.float32)
    widget = Show3DVolume(data)
    assert widget.nz == 10
    assert widget.ny == 20
    assert widget.nx == 30

def test_show3dvolume_rejects_4d():
    """Raises error for 4D input."""
    data = np.random.rand(8, 8, 8, 8).astype(np.float32)
    with pytest.raises(ValueError, match="3D"):
        Show3DVolume(data)

def test_show3dvolume_slice_change_updates_stats():
    """Changing slice positions recomputes stats."""
    data = np.zeros((16, 16, 16), dtype=np.float32)
    data[0, :, :] = 10.0
    data[15, :, :] = 50.0
    widget = Show3DVolume(data)
    # Move Z slice to first plane (all 10s)
    widget.slice_z = 0
    assert widget.stats_mean[0] == pytest.approx(10.0)
    # Move Z slice to last plane (all 50s)
    widget.slice_z = 15
    assert widget.stats_mean[0] == pytest.approx(50.0)

def test_show3dvolume_stats_per_plane():
    """Stats are computed from correct orthogonal planes."""
    data = np.zeros((10, 20, 30), dtype=np.float32)
    # XY plane at slice_z=5: set to 1.0
    data[5, :, :] = 1.0
    # XZ plane at slice_y=10: set to 2.0
    data[:, 10, :] = 2.0
    # YZ plane at slice_x=15: set to 3.0
    data[:, :, 15] = 3.0
    widget = Show3DVolume(data)
    widget.slice_z = 5
    widget.slice_y = 10
    widget.slice_x = 15
    # XY plane mean should reflect data[5, :, :]
    # XZ plane mean should reflect data[:, 10, :]
    # YZ plane mean should reflect data[:, :, 15]
    assert widget.stats_mean[0] != widget.stats_mean[1]  # Different planes have different stats

def test_show3dvolume_crosshair():
    """show_crosshair default True, can be toggled."""
    data = np.random.rand(8, 8, 8).astype(np.float32)
    widget = Show3DVolume(data)
    assert widget.show_crosshair is True
    widget2 = Show3DVolume(data, show_crosshair=False)
    assert widget2.show_crosshair is False

def test_show3dvolume_show_controls():
    """show_controls default True."""
    data = np.random.rand(8, 8, 8).astype(np.float32)
    widget = Show3DVolume(data)
    assert widget.show_controls is True
    widget2 = Show3DVolume(data, show_controls=False)
    assert widget2.show_controls is False

def test_show3dvolume_show_fft():
    """show_fft default False."""
    data = np.random.rand(8, 8, 8).astype(np.float32)
    widget = Show3DVolume(data)
    assert widget.show_fft is False
    widget2 = Show3DVolume(data, show_fft=True)
    assert widget2.show_fft is True

@pytest.mark.parametrize("trait_name", ["disabled_tools", "hidden_tools"])
def test_show3dvolume_tool_lists_default_empty(trait_name):
    data = np.random.rand(8, 16, 16).astype(np.float32)
    w = Show3DVolume(data)
    assert getattr(w, trait_name) == []

@pytest.mark.parametrize(
    ("trait_name", "ctor_kwargs", "expected"),
    [
        ("disabled_tools", {"disabled_tools": ["display", "Histogram", "volume"]}, ["display", "histogram", "volume"]),
        ("hidden_tools", {"hidden_tools": ["display", "Histogram", "volume"]}, ["display", "histogram", "volume"]),
        ("disabled_tools", {"disable_display": True, "disable_fft": True, "disable_volume": True}, ["display", "fft", "volume"]),
        ("hidden_tools", {"hide_display": True, "hide_fft": True, "hide_volume": True}, ["display", "fft", "volume"]),
        ("disabled_tools", {"disable_all": True}, ["all"]),
        ("hidden_tools", {"hide_all": True}, ["all"]),
    ],
)
def test_show3dvolume_tool_lists_constructor_behavior(trait_name, ctor_kwargs, expected):
    data = np.random.rand(8, 16, 16).astype(np.float32)
    w = Show3DVolume(data, **ctor_kwargs)
    assert getattr(w, trait_name) == expected

@pytest.mark.parametrize("kwargs", [{"disabled_tools": ["not_real"]}, {"hidden_tools": ["not_real"]}])
def test_show3dvolume_tool_lists_unknown_raises(kwargs):
    data = np.random.rand(8, 16, 16).astype(np.float32)
    with pytest.raises(ValueError, match="Unknown tool group"):
        Show3DVolume(data, **kwargs)

@pytest.mark.parametrize("trait_name", ["disabled_tools", "hidden_tools"])
def test_show3dvolume_tool_lists_normalizes(trait_name):
    data = np.random.rand(8, 16, 16).astype(np.float32)
    w = Show3DVolume(data)
    setattr(w, trait_name, ["DISPLAY", "display", "fft"])
    assert getattr(w, trait_name) == ["display", "fft"]

def test_show3dvolume_scale_bar():
    """Scale bar parameters are stored."""
    data = np.random.rand(8, 8, 8).astype(np.float32)
    widget = Show3DVolume(data, pixel_size=2.5, scale_bar_visible=False)
    assert widget.pixel_size == pytest.approx(2.5)
    assert widget.scale_bar_visible is False

def test_show3dvolume_volume_bytes_size():
    """volume_bytes has correct size (nz * ny * nx * 4 bytes for float32)."""
    data = np.random.rand(8, 10, 12).astype(np.float32)
    widget = Show3DVolume(data)
    assert len(widget.volume_bytes) == 8 * 10 * 12 * 4

def test_show3dvolume_constant_data():
    """Constant volume doesn't crash."""
    data = np.ones((8, 8, 8), dtype=np.float32) * 42.0
    widget = Show3DVolume(data)
    for mean in widget.stats_mean:
        assert mean == pytest.approx(42.0)
    for std in widget.stats_std:
        assert std == pytest.approx(0.0)

def test_show3dvolume_single_voxel():
    """(1, 1, 1) volume works."""
    data = np.array([[[5.0]]], dtype=np.float32)
    widget = Show3DVolume(data)
    assert widget.nz == 1
    assert widget.ny == 1
    assert widget.nx == 1
    assert widget.slice_z == 0
    assert widget.slice_y == 0
    assert widget.slice_x == 0

def test_show3dvolume_asymmetric():
    """Very asymmetric dimensions work."""
    data = np.random.rand(5, 100, 10).astype(np.float32)
    widget = Show3DVolume(data)
    assert widget.nz == 5
    assert widget.ny == 100
    assert widget.nx == 10
    assert widget.slice_z == 2
    assert widget.slice_y == 50
    assert widget.slice_x == 5

def test_show3dvolume_playback_defaults():
    """Playback defaults match Show3D."""
    data = np.random.rand(8, 8, 8).astype(np.float32)
    widget = Show3DVolume(data)
    assert widget.playing is False
    assert widget.reverse is False
    assert widget.boomerang is False
    assert widget.fps == pytest.approx(5.0)
    assert widget.loop is True
    assert widget.play_axis == 0

def test_show3dvolume_play_pause_stop():
    """play/pause/stop methods work."""
    data = np.random.rand(8, 8, 8).astype(np.float32)
    widget = Show3DVolume(data)
    widget.play()
    assert widget.playing is True
    widget.pause()
    assert widget.playing is False
    widget.slice_z = 5
    widget.stop()
    assert widget.playing is False
    assert widget.slice_z == 4  # stop() resets to center slice

def test_show3dvolume_fps_parameter():
    """fps constructor parameter is applied."""
    data = np.random.rand(8, 8, 8).astype(np.float32)
    widget = Show3DVolume(data, fps=15.0)
    assert widget.fps == pytest.approx(15.0)

def test_show3dvolume_play_axis():
    """play_axis can be set."""
    data = np.random.rand(8, 8, 8).astype(np.float32)
    widget = Show3DVolume(data)
    widget.play_axis = 2
    assert widget.play_axis == 2
    widget.play_axis = 3  # "All"
    assert widget.play_axis == 3

def test_show3dvolume_gif_generation_sets_metadata():
    import json

    data = np.random.rand(8, 8, 8).astype(np.float32)
    widget = Show3DVolume(data, cmap="viridis")
    widget._export_axis = 2
    widget.fps = 5.0
    widget._generate_gif()
    assert widget._gif_data[:3] == b"GIF"
    metadata = json.loads(widget._gif_metadata_json)
    assert metadata["widget_name"] == "Show3DVolume"
    assert metadata["format"] == "gif"
    assert metadata["export_kind"] == "animated_slices"
    assert metadata["export_axis"] == 2

# ── State Protocol ────────────────────────────────────────────────────────

def test_show3dvolume_state_dict_roundtrip():
    data = np.random.rand(16, 16, 16).astype(np.float32)
    w = Show3DVolume(data, cmap="viridis", log_scale=True, title="Volume",
                     pixel_size=2.0, show_fft=True, fps=10.0,
                     disabled_tools=["display", "fft"],
                     hidden_tools=["stats"])
    w.slice_z = 5
    w.slice_y = 8
    w.slice_x = 12
    sd = w.state_dict()
    w2 = Show3DVolume(data, state=sd)
    assert w2.cmap == "viridis"
    assert w2.log_scale is True
    assert w2.title == "Volume"
    assert w2.pixel_size == pytest.approx(2.0)
    assert w2.show_fft is True
    assert w2.fps == pytest.approx(10.0)
    assert w2.slice_z == 5
    assert w2.slice_y == 8
    assert w2.slice_x == 12
    assert w2.disabled_tools == ["display", "fft"]
    assert w2.hidden_tools == ["stats"]

def test_show3dvolume_state_dict_includes_tool_customization():
    data = np.random.rand(8, 8, 8).astype(np.float32)
    w = Show3DVolume(data, disabled_tools=["display"], hidden_tools=["navigation"])
    sd = w.state_dict()
    assert "disabled_tools" in sd
    assert "hidden_tools" in sd
    assert sd["disabled_tools"] == ["display"]
    assert sd["hidden_tools"] == ["navigation"]

def test_show3dvolume_save_load_file(tmp_path):
    import json
    data = np.random.rand(8, 8, 8).astype(np.float32)
    w = Show3DVolume(data, cmap="plasma", title="Saved Vol")
    path = tmp_path / "vol_state.json"
    w.save(str(path))
    assert path.exists()
    saved = json.loads(path.read_text())
    assert saved["metadata_version"] == "1.0"
    assert saved["widget_name"] == "Show3DVolume"
    assert isinstance(saved["widget_version"], str)
    assert saved["state"]["cmap"] == "plasma"
    w2 = Show3DVolume(data, state=str(path))
    assert w2.cmap == "plasma"
    assert w2.title == "Saved Vol"

def test_show3dvolume_summary(capsys):
    data = np.random.rand(16, 16, 16).astype(np.float32)
    w = Show3DVolume(data, title="Nanoparticle", cmap="inferno")
    w.summary()
    out = capsys.readouterr().out
    assert "Nanoparticle" in out
    assert "16×16×16" in out
    assert "inferno" in out

def test_show3dvolume_repr():
    data = np.random.rand(16, 16, 16).astype(np.float32)
    w = Show3DVolume(data, cmap="inferno")
    r = repr(w)
    assert "Show3DVolume" in r
    assert "16×16×16" in r
    assert "inferno" in r

def test_show3dvolume_set_image():
    data = np.random.rand(16, 16, 16).astype(np.float32)
    widget = Show3DVolume(data, cmap="viridis")
    assert widget.nz == 16

    new_data = np.random.rand(32, 24, 20).astype(np.float32)
    widget.set_image(new_data)
    assert widget.nz == 32
    assert widget.ny == 24
    assert widget.nx == 20
    assert widget.cmap == "viridis"
    assert len(widget.volume_bytes) == 32 * 24 * 20 * 4

# ── save_image ───────────────────────────────────────────────────────────

def test_show3dvolume_save_image_xy(tmp_path):
    vol = np.random.rand(16, 16, 16).astype(np.float32)
    w = Show3DVolume(vol, cmap="viridis")
    out = w.save_image(tmp_path / "xy.png", plane="xy")
    assert out.exists()
    from PIL import Image
    img = Image.open(out)
    assert img.size == (16, 16)

def test_show3dvolume_save_image_xz(tmp_path):
    vol = np.random.rand(16, 20, 24).astype(np.float32)
    w = Show3DVolume(vol)
    out = w.save_image(tmp_path / "xz.png", plane="xz")
    assert out.exists()
    from PIL import Image
    img = Image.open(out)
    assert img.size == (24, 16)

def test_show3dvolume_save_image_yz(tmp_path):
    vol = np.random.rand(16, 20, 24).astype(np.float32)
    w = Show3DVolume(vol)
    out = w.save_image(tmp_path / "yz.png", plane="yz")
    assert out.exists()
    from PIL import Image
    img = Image.open(out)
    assert img.size == (20, 16)

def test_show3dvolume_save_image_pdf(tmp_path):
    vol = np.random.rand(8, 8, 8).astype(np.float32)
    w = Show3DVolume(vol)
    out = w.save_image(tmp_path / "slice.pdf")
    assert out.exists()

def test_show3dvolume_save_image_bad_plane(tmp_path):
    vol = np.random.rand(8, 8, 8).astype(np.float32)
    w = Show3DVolume(vol)
    with pytest.raises(ValueError, match="Unknown plane"):
        w.save_image(tmp_path / "out.png", plane="ab")

def test_show3dvolume_widget_version_is_set():
    vol = np.random.rand(8, 8, 8).astype(np.float32)
    w = Show3DVolume(vol)
    assert w.widget_version != "unknown"


# ── Dual-Volume Comparison Mode ──────────────────────────────────────────

def test_show3dvolume_dual_mode_basic():
    """Two volumes: dual_mode True, both volume_bytes populated."""
    a = np.random.rand(16, 16, 16).astype(np.float32)
    b = np.random.rand(16, 16, 16).astype(np.float32)
    w = Show3DVolume(a, data_b=b, title="A", title_b="B")
    assert w.dual_mode is True
    assert w.title == "A"
    assert w.title_b == "B"
    assert len(w.volume_bytes) == 16 * 16 * 16 * 4
    assert len(w.volume_bytes_b) == 16 * 16 * 16 * 4


def test_show3dvolume_dual_mode_shape_mismatch():
    """ValueError when data_b shape differs from data."""
    a = np.random.rand(16, 16, 16).astype(np.float32)
    b = np.random.rand(8, 8, 8).astype(np.float32)
    with pytest.raises(ValueError, match="must match"):
        Show3DVolume(a, data_b=b)


def test_show3dvolume_dual_mode_b_must_be_3d():
    """ValueError when data_b is 2D."""
    a = np.random.rand(16, 16, 16).astype(np.float32)
    b = np.random.rand(16, 16).astype(np.float32)
    with pytest.raises(ValueError, match="3D"):
        Show3DVolume(a, data_b=b)


def test_show3dvolume_single_mode_unchanged():
    """Backward compat: single volume → dual_mode False, volume_bytes_b empty."""
    a = np.random.rand(8, 8, 8).astype(np.float32)
    w = Show3DVolume(a)
    assert w.dual_mode is False
    assert w.volume_bytes_b == b""
    assert w.title_b == ""


def test_show3dvolume_dual_mode_stats_b():
    """Independent stats for volume B."""
    a = np.ones((8, 8, 8), dtype=np.float32) * 5.0
    b = np.ones((8, 8, 8), dtype=np.float32) * 10.0
    w = Show3DVolume(a, data_b=b)
    for mean_a in w.stats_mean:
        assert mean_a == pytest.approx(5.0)
    for mean_b in w.stats_mean_b:
        assert mean_b == pytest.approx(10.0)


def test_show3dvolume_dual_mode_stats_update_on_slice_change():
    """Both A and B stats update when slice changes."""
    a = np.zeros((16, 16, 16), dtype=np.float32)
    b = np.zeros((16, 16, 16), dtype=np.float32)
    a[0, :, :] = 1.0
    b[0, :, :] = 99.0
    w = Show3DVolume(a, data_b=b)
    w.slice_z = 0
    assert w.stats_mean[0] == pytest.approx(1.0)
    assert w.stats_mean_b[0] == pytest.approx(99.0)


def test_show3dvolume_dual_mode_torch():
    """PyTorch tensor input for both volumes."""
    a = torch.rand(8, 8, 8)
    b = torch.rand(8, 8, 8)
    w = Show3DVolume(a, data_b=b)
    assert w.dual_mode is True
    assert w.nz == 8


def test_show3dvolume_dual_mode_ioresult():
    """IOResult duck-typing for data_b."""
    from quantem.widget.io import IOResult
    a = np.random.rand(8, 8, 8).astype(np.float32)
    b_result = IOResult(
        data=np.random.rand(8, 8, 8).astype(np.float32),
        pixel_size=2.5,
        units="Å",
        title="Ground Truth",
    )
    w = Show3DVolume(a, data_b=b_result)
    assert w.dual_mode is True
    assert w.title_b == "Ground Truth"


def test_show3dvolume_dual_mode_state_dict():
    """state_dict includes dual_mode, title_b."""
    a = np.random.rand(8, 8, 8).astype(np.float32)
    b = np.random.rand(8, 8, 8).astype(np.float32)
    w = Show3DVolume(a, data_b=b, title_b="GT")
    sd = w.state_dict()
    assert sd["dual_mode"] is True
    assert sd["title_b"] == "GT"


def test_show3dvolume_dual_mode_summary(capsys):
    """summary shows Volume B info."""
    a = np.random.rand(8, 8, 8).astype(np.float32)
    b = np.random.rand(8, 8, 8).astype(np.float32)
    w = Show3DVolume(a, data_b=b, title="Phantom", title_b="Ground Truth")
    w.summary()
    out = capsys.readouterr().out
    assert "Volume B" in out or "Ground Truth" in out


def test_show3dvolume_dual_mode_repr():
    """repr contains dual=True."""
    a = np.random.rand(8, 8, 8).astype(np.float32)
    b = np.random.rand(8, 8, 8).astype(np.float32)
    w = Show3DVolume(a, data_b=b)
    assert "dual=True" in repr(w)


def test_show3dvolume_set_image_dual():
    """set_image(a2, b2) updates both volumes."""
    a = np.random.rand(8, 8, 8).astype(np.float32)
    b = np.random.rand(8, 8, 8).astype(np.float32)
    w = Show3DVolume(a, data_b=b)
    a2 = np.ones((10, 10, 10), dtype=np.float32) * 2.0
    b2 = np.ones((10, 10, 10), dtype=np.float32) * 3.0
    w.set_image(a2, data_b=b2)
    assert w.nz == 10
    assert w.dual_mode is True
    assert len(w.volume_bytes_b) == 10 * 10 * 10 * 4
    for m in w.stats_mean:
        assert m == pytest.approx(2.0)
    for m in w.stats_mean_b:
        assert m == pytest.approx(3.0)


def test_show3dvolume_set_image_drops_b_on_shape_mismatch():
    """set_image(larger_a) drops old B when shapes no longer match."""
    a = np.random.rand(8, 8, 8).astype(np.float32)
    b = np.random.rand(8, 8, 8).astype(np.float32)
    w = Show3DVolume(a, data_b=b)
    assert w.dual_mode is True
    # New A has different shape → B should be dropped
    a2 = np.random.rand(10, 10, 10).astype(np.float32)
    w.set_image(a2)
    assert w.dual_mode is False
    assert w.volume_bytes_b == b""


# ── Diff View ─────────────────────────────────────────────────────────────

def test_show3dvolume_show_diff_default():
    """show_diff defaults to False."""
    a = np.random.rand(8, 8, 8).astype(np.float32)
    w = Show3DVolume(a)
    assert w.show_diff is False

def test_show3dvolume_show_diff_state_dict():
    """show_diff is included in state_dict and restored via state param."""
    a = np.random.rand(8, 8, 8).astype(np.float32)
    b = np.random.rand(8, 8, 8).astype(np.float32)
    w = Show3DVolume(a, data_b=b, show_diff=True)
    assert w.show_diff is True
    sd = w.state_dict()
    assert sd["show_diff"] is True
    w2 = Show3DVolume(a, data_b=b, state=sd)
    assert w2.show_diff is True

def test_show3dvolume_show_diff_single_mode():
    """show_diff has no effect in single-volume mode (trait still stored)."""
    a = np.random.rand(8, 8, 8).astype(np.float32)
    w = Show3DVolume(a, show_diff=True)
    assert w.show_diff is True
    assert w.dual_mode is False

