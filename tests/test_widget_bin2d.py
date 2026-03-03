import json

import numpy as np
import pytest

from quantem.widget import Bin2D


def test_bin2d_loads_2d_array():
    data = np.random.rand(64, 64).astype(np.float32)
    w = Bin2D(data)
    assert w.height == 64
    assert w.width == 64
    assert w.n_images == 1
    assert w.bin_factor == 2
    assert w.binned_height == 32
    assert w.binned_width == 32

def test_bin2d_loads_3d_stack():
    data = np.random.rand(5, 32, 48).astype(np.float32)
    w = Bin2D(data)
    assert w.n_images == 5
    assert w.height == 32
    assert w.width == 48
    assert w.binned_height == 16
    assert w.binned_width == 24

def test_bin2d_loads_list_of_images():
    imgs = [np.random.rand(40, 40).astype(np.float32) for _ in range(3)]
    w = Bin2D(imgs, labels=["A", "B", "C"])
    assert w.n_images == 3
    assert w.labels == ["A", "B", "C"]

def test_bin2d_bin_factor_changes_shape():
    data = np.random.rand(64, 64).astype(np.float32)
    w = Bin2D(data, bin_factor=4)
    assert w.binned_height == 16
    assert w.binned_width == 16
    w.bin_factor = 8
    assert w.binned_height == 8
    assert w.binned_width == 8

def test_bin2d_mean_vs_sum_mode():
    data = np.ones((16, 16), dtype=np.float32) * 3.0
    w_mean = Bin2D(data, bin_factor=2, bin_mode="mean")
    w_sum = Bin2D(data, bin_factor=2, bin_mode="sum")
    mean_arr = w_mean.get_binned_image()
    sum_arr = w_sum.get_binned_image()
    np.testing.assert_allclose(mean_arr, 3.0, atol=1e-6)
    np.testing.assert_allclose(sum_arr, 12.0, atol=1e-6)

def test_bin2d_crop_vs_pad_edge_mode():
    data = np.ones((65, 65), dtype=np.float32)
    w_crop = Bin2D(data, bin_factor=4, edge_mode="crop")
    w_pad = Bin2D(data, bin_factor=4, edge_mode="pad")
    assert w_crop.binned_height == 16
    assert w_crop.binned_width == 16
    assert w_pad.binned_height == 17
    assert w_pad.binned_width == 17

def test_bin2d_calibration_scaling():
    data = np.random.rand(64, 64).astype(np.float32)
    w = Bin2D(data, bin_factor=4, pixel_size=1.5)
    assert w.pixel_size == 1.5
    assert w.binned_pixel_size == pytest.approx(6.0)

def test_bin2d_get_binned_data():
    data = np.random.rand(3, 32, 32).astype(np.float32)
    w = Bin2D(data, bin_factor=2)
    result = w.get_binned_data()
    assert result.shape == (3, 16, 16)
    assert result.dtype == np.float32

def test_bin2d_get_binned_image():
    data = np.random.rand(3, 32, 32).astype(np.float32)
    w = Bin2D(data, bin_factor=2)
    img = w.get_binned_image(idx=1)
    assert img.shape == (16, 16)

def test_bin2d_state_dict_roundtrip():
    data = np.random.rand(32, 32).astype(np.float32)
    w = Bin2D(data, bin_factor=4, bin_mode="sum", edge_mode="pad", cmap="viridis", log_scale=True, title="Test")
    sd = w.state_dict()
    assert sd["bin_factor"] == 4
    assert sd["bin_mode"] == "sum"
    assert sd["edge_mode"] == "pad"
    assert sd["cmap"] == "viridis"
    assert sd["log_scale"] is True
    assert sd["title"] == "Test"
    w2 = Bin2D(np.random.rand(32, 32).astype(np.float32))
    w2.load_state_dict(sd)
    assert w2.bin_factor == 4
    assert w2.bin_mode == "sum"
    assert w2.cmap == "viridis"

def test_bin2d_save_load_file(tmp_path):
    data = np.random.rand(32, 32).astype(np.float32)
    w = Bin2D(data, bin_factor=3, title="Save Test")
    path = tmp_path / "bin2d_state.json"
    w.save(str(path))
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["metadata_version"] == "1.0"
    assert payload["widget_name"] == "Bin2D"
    assert isinstance(payload["widget_version"], str)
    assert payload["state"]["bin_factor"] == 3
    w2 = Bin2D(np.random.rand(32, 32).astype(np.float32), state=str(path))
    assert w2.bin_factor == 3
    assert w2.title == "Save Test"

def test_bin2d_summary(capsys):
    data = np.random.rand(64, 64).astype(np.float32)
    w = Bin2D(data, bin_factor=4, title="My Image", pixel_size=2.0)
    w.summary()
    out = capsys.readouterr().out
    assert "Bin2D" in out
    assert "My Image" in out
    assert "64" in out
    assert "4" in out

def test_bin2d_repr():
    data = np.random.rand(64, 64).astype(np.float32)
    w = Bin2D(data, bin_factor=2, title="T")
    r = repr(w)
    assert "Bin2D(" in r
    assert "64" in r
    assert "T" in r

def test_bin2d_set_image():
    data1 = np.random.rand(64, 64).astype(np.float32)
    w = Bin2D(data1, bin_factor=4, cmap="viridis")
    assert w.binned_height == 16
    data2 = np.random.rand(128, 128).astype(np.float32)
    result = w.set_image(data2)
    assert result is w
    assert w.height == 128
    assert w.width == 128
    assert w.binned_height == 32
    assert w.cmap == "viridis"

def test_bin2d_torch_input():
    torch = pytest.importorskip("torch")
    t = torch.rand(32, 32)
    w = Bin2D(t)
    assert w.height == 32
    assert w.width == 32

def test_bin2d_ioresult_input():
    from quantem.widget import IOResult
    data = np.random.rand(64, 64).astype(np.float32)
    result = IOResult(data=data, pixel_size=2.5, title="From IO")
    w = Bin2D(result)
    assert w.title == "From IO"
    assert w.pixel_size == 2.5

def test_bin2d_original_bytes_populated():
    data = np.random.rand(32, 32).astype(np.float32)
    w = Bin2D(data)
    assert len(w.original_bytes) == 32 * 32 * 4
    assert len(w.binned_bytes) == 16 * 16 * 4

def test_bin2d_stats_populated():
    data = np.ones((16, 16), dtype=np.float32) * 5.0
    w = Bin2D(data, bin_factor=2)
    assert len(w.original_stats) == 4
    assert w.original_stats[0] == pytest.approx(5.0)
    assert len(w.binned_stats) == 4
    assert w.binned_stats[0] == pytest.approx(5.0)

def test_bin2d_selected_idx_updates_bytes():
    data = np.zeros((3, 16, 16), dtype=np.float32)
    data[0] = 1.0
    data[1] = 2.0
    data[2] = 3.0
    w = Bin2D(data)
    assert w.original_stats[0] == pytest.approx(1.0)
    w.selected_idx = 1
    assert w.original_stats[0] == pytest.approx(2.0)

# --- Tool lock/hide tests ---

@pytest.mark.parametrize("trait_name", ["disabled_tools", "hidden_tools"])
def test_bin2d_tool_default_empty(trait_name):
    data = np.random.rand(32, 32).astype(np.float32)
    w = Bin2D(data)
    assert getattr(w, trait_name) == []

@pytest.mark.parametrize(
    ("trait_name", "kwargs", "expected"),
    [
        ("disabled_tools", {"disabled_tools": ["display", "Binning"]}, ["display", "binning"]),
        ("hidden_tools", {"hidden_tools": ["display", "Binning"]}, ["display", "binning"]),
        ("disabled_tools", {"disable_display": True, "disable_binning": True}, ["display", "binning"]),
        ("hidden_tools", {"hide_display": True, "hide_binning": True}, ["display", "binning"]),
        ("disabled_tools", {"disable_all": True}, ["all"]),
        ("hidden_tools", {"hide_all": True}, ["all"]),
    ],
)
def test_bin2d_tool_lock_hide(trait_name, kwargs, expected):
    data = np.random.rand(32, 32).astype(np.float32)
    w = Bin2D(data, **kwargs)
    assert getattr(w, trait_name) == expected

@pytest.mark.parametrize("kwargs", [{"disabled_tools": ["not_real"]}, {"hidden_tools": ["not_real"]}])
def test_bin2d_tool_invalid_key_raises(kwargs):
    data = np.random.rand(32, 32).astype(np.float32)
    with pytest.raises(ValueError):
        Bin2D(data, **kwargs)

def test_bin2d_tool_runtime_api():
    data = np.random.rand(32, 32).astype(np.float32)
    w = Bin2D(data)
    assert w.lock_tool("display") is w
    assert "display" in w.disabled_tools
    assert w.unlock_tool("display") is w
    assert "display" not in w.disabled_tools
    assert w.hide_tool("binning") is w
    assert "binning" in w.hidden_tools
    assert w.show_tool("binning") is w
    assert "binning" not in w.hidden_tools

def test_bin2d_widget_version():
    data = np.random.rand(32, 32).astype(np.float32)
    w = Bin2D(data)
    assert w.widget_version != "unknown"

def test_bin2d_save_image(tmp_path):
    pytest.importorskip("PIL")
    pytest.importorskip("matplotlib")
    data = np.random.rand(32, 32).astype(np.float32)
    w = Bin2D(data, bin_factor=2)
    path = tmp_path / "test.png"
    out = w.save_image(path)
    assert out == path
    assert path.exists()

def test_bin2d_save_image_grid(tmp_path):
    pytest.importorskip("PIL")
    pytest.importorskip("matplotlib")
    data = np.random.rand(32, 32).astype(np.float32)
    w = Bin2D(data, bin_factor=2)
    path = tmp_path / "grid.png"
    out = w.save_image(path, view="grid")
    assert out == path
    assert path.exists()
    meta = path.with_suffix(".json")
    assert meta.exists()
    payload = json.loads(meta.read_text())
    assert payload["widget_name"] == "Bin2D"
    assert payload["render"]["view"] == "grid"
