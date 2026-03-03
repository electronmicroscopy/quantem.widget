import json
import zipfile

import numpy as np
import pytest

from quantem.widget import Bin4D

def test_bin_widget_loads_from_4d_array():
    data = np.random.rand(8, 10, 16, 18).astype(np.float32)
    widget = Bin4D(data)

    assert widget is not None
    assert (widget.scan_rows, widget.scan_cols, widget.det_rows, widget.det_cols) == (8, 10, 16, 18)
    assert (widget.binned_scan_rows, widget.binned_scan_cols, widget.binned_det_rows, widget.binned_det_cols) == (8, 10, 16, 18)
    assert isinstance(widget.device, str)
    assert len(widget.original_bf_bytes) == 8 * 10 * 4
    assert len(widget.binned_adf_bytes) == 8 * 10 * 4

def test_bin_widget_updates_shape_and_calibration():
    data = np.random.rand(12, 8, 20, 24).astype(np.float32)
    widget = Bin4D(data, pixel_size=(0.5, 0.25), k_pixel_size=(1.0, 2.0))

    widget.scan_bin_row = 3
    widget.scan_bin_col = 2
    widget.det_bin_row = 4
    widget.det_bin_col = 3

    assert (widget.binned_scan_rows, widget.binned_scan_cols) == (4, 4)
    assert (widget.binned_det_rows, widget.binned_det_cols) == (5, 8)

    assert widget.binned_pixel_size_row == 1.5
    assert widget.binned_pixel_size_col == 0.5
    assert widget.binned_k_pixel_size_row == 4.0
    assert widget.binned_k_pixel_size_col == 6.0

def test_bin_widget_crop_vs_pad_behavior():
    data = np.ones((9, 9, 9, 9), dtype=np.float32)
    widget = Bin4D(data, edge_mode="crop")

    widget.scan_bin_row = 4
    widget.scan_bin_col = 4
    widget.det_bin_row = 4
    widget.det_bin_col = 4
    assert (widget.binned_scan_rows, widget.binned_scan_cols, widget.binned_det_rows, widget.binned_det_cols) == (2, 2, 2, 2)

    widget.edge_mode = "pad"
    assert (widget.binned_scan_rows, widget.binned_scan_cols, widget.binned_det_rows, widget.binned_det_cols) == (3, 3, 3, 3)

def test_bin_widget_flattened_3d_input_with_scan_shape():
    data = np.zeros((6, 4, 4), dtype=np.float32)
    for i in range(6):
        data[i] = i

    widget = Bin4D(data, scan_shape=(2, 3))
    assert (widget.scan_rows, widget.scan_cols, widget.det_rows, widget.det_cols) == (2, 3, 4, 4)

    widget.scan_bin_row = 2
    widget.scan_bin_col = 3
    out = widget.get_binned_data(copy=False)
    assert out.shape == (1, 1, 4, 4)

def test_bin_widget_error_mode_reports_status():
    data = np.random.rand(7, 7, 7, 7).astype(np.float32)
    widget = Bin4D(data, edge_mode="crop")
    original_shape = widget.get_binned_data().shape

    widget.edge_mode = "error"
    widget.scan_bin_row = 3  # 7 not divisible by 3 -> status error

    assert widget.status_level == "error"
    assert "not divisible" in widget.status_message
    assert widget.get_binned_data().shape == original_shape


def test_bin_widget_exports_image_with_metadata(tmp_path):
    pytest.importorskip("PIL")
    pytest.importorskip("matplotlib")

    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    widget = Bin4D(data, device="cpu")

    image_path = tmp_path / "bin_grid.png"
    meta_path = tmp_path / "bin_grid_meta.json"
    out = widget.save_image(image_path, view="grid", include_metadata=True, metadata_path=meta_path)

    assert out == image_path
    assert image_path.exists()
    assert meta_path.exists()
    payload = json.loads(meta_path.read_text())
    assert payload["metadata_version"] == "1.0"
    assert payload["widget_name"] == "Bin4D"
    assert isinstance(payload["widget_version"], str)
    assert payload["format"] == "png"
    assert payload["export_kind"] == "single_view_image"
    assert payload["render"]["view"] == "grid"

def test_bin_widget_exports_zip_bundle(tmp_path):
    pytest.importorskip("PIL")
    pytest.importorskip("matplotlib")

    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    widget = Bin4D(data, device="cpu")

    zip_path = tmp_path / "bin_bundle.zip"
    out = widget.save_zip(zip_path, include_arrays=True)

    assert out == zip_path
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())

    expected = {
        "original_bf.png",
        "original_adf.png",
        "binned_bf.png",
        "binned_adf.png",
        "grid.png",
        "original_bf.npy",
        "original_adf.npy",
        "binned_bf.npy",
        "binned_adf.npy",
        "metadata.json",
    }
    assert expected.issubset(names)

    with zipfile.ZipFile(zip_path, "r") as zf:
        metadata = json.loads(zf.read("metadata.json").decode("utf-8"))
    assert metadata["metadata_version"] == "1.0"
    assert metadata["widget_name"] == "Bin4D"
    assert isinstance(metadata["widget_version"], str)
    assert metadata["format"] == "zip"
    assert metadata["export_kind"] == "multi_panel_bundle"

def test_bin_widget_exports_gif(tmp_path):
    pytest.importorskip("PIL")
    pytest.importorskip("matplotlib")

    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    widget = Bin4D(data, device="cpu")

    gif_path = tmp_path / "bin_compare.gif"
    out = widget.save_gif(gif_path, channel="bf")

    assert out == gif_path
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0

    meta = gif_path.with_suffix(".json")
    assert meta.exists()
    payload = json.loads(meta.read_text())
    assert payload["metadata_version"] == "1.0"
    assert payload["widget_name"] == "Bin4D"
    assert isinstance(payload["widget_version"], str)
    assert payload["format"] == "gif"
    assert payload["export_kind"] == "before_after_animation"

def test_bin_widget_version_is_set():
    data = np.random.rand(8, 10, 16, 18).astype(np.float32)
    w = Bin4D(data)
    assert w.widget_version != "unknown"

# --- Title tests ---

def test_bin_title_default():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Bin4D(data)
    assert w.title == ""

def test_bin_title_set():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Bin4D(data, title="My Binning")
    assert w.title == "My Binning"

def test_bin_title_in_state_dict():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Bin4D(data, title="Test Title")
    sd = w.state_dict()
    assert sd["title"] == "Test Title"
    w2 = Bin4D(np.random.rand(4, 4, 8, 8).astype(np.float32))
    w2.load_state_dict(sd)
    assert w2.title == "Test Title"

def test_bin_title_in_summary(capsys):
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Bin4D(data, title="Custom Bin Name")
    w.summary()
    out = capsys.readouterr().out
    assert "Custom Bin Name" in out

def test_bin_title_in_repr():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Bin4D(data, title="My Title")
    r = repr(w)
    assert "My Title" in r

# --- Display trait sync tests ---

def test_bin_state_dict_includes_display_traits():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Bin4D(data, cmap="viridis", log_scale=True, show_controls=False)

    sd = w.state_dict()
    assert sd["cmap"] == "viridis"
    assert sd["log_scale"] is True
    assert sd["show_controls"] is False

    w2 = Bin4D(np.random.rand(4, 4, 8, 8).astype(np.float32))
    w2.load_state_dict(sd)
    assert w2.cmap == "viridis"
    assert w2.log_scale is True
    assert w2.show_controls is False

def test_bin_set_data_replaces_array():
    data1 = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Bin4D(data1, cmap="viridis", log_scale=True)

    assert (w.scan_rows, w.scan_cols, w.det_rows, w.det_cols) == (4, 4, 8, 8)

    data2 = np.random.rand(6, 5, 16, 12).astype(np.float32)
    result = w.set_data(data2)

    assert result is w
    assert (w.scan_rows, w.scan_cols, w.det_rows, w.det_cols) == (6, 5, 16, 12)
    assert w.cmap == "viridis"
    assert w.log_scale is True

# --- Tool lock/hide tests ---

@pytest.mark.parametrize("trait_name", ["disabled_tools", "hidden_tools"])
def test_bin_tool_default_empty(trait_name):
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Bin4D(data)
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
def test_bin_tool_lock_hide(trait_name, kwargs, expected):
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Bin4D(data, **kwargs)
    assert getattr(w, trait_name) == expected

@pytest.mark.parametrize("kwargs", [{"disabled_tools": ["not_real"]}, {"hidden_tools": ["not_real"]}])
def test_bin_tool_invalid_key_raises(kwargs):
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    with pytest.raises(ValueError):
        Bin4D(data, **kwargs)

@pytest.mark.parametrize("trait_name", ["disabled_tools", "hidden_tools"])
def test_bin_tool_trait_assignment_normalizes(trait_name):
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Bin4D(data)
    setattr(w, trait_name, ["DISPLAY", "display", "binning"])
    assert getattr(w, trait_name) == ["display", "binning"]

def test_bin_tool_runtime_api():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Bin4D(data)

    assert w.lock_tool("display") is w
    assert "display" in w.disabled_tools
    assert w.unlock_tool("display") is w
    assert "display" not in w.disabled_tools

    assert w.hide_tool("binning") is w
    assert "binning" in w.hidden_tools
    assert w.show_tool("binning") is w
    assert "binning" not in w.hidden_tools

def test_bin_state_dict_roundtrip_with_tools():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Bin4D(data, disabled_tools=["display"], hidden_tools=["stats"])

    sd = w.state_dict()
    assert sd["disabled_tools"] == ["display"]
    assert sd["hidden_tools"] == ["stats"]

    w2 = Bin4D(np.random.rand(4, 4, 8, 8).astype(np.float32))
    w2.load_state_dict(sd)
    assert w2.disabled_tools == ["display"]
    assert w2.hidden_tools == ["stats"]

def test_bin_summary_includes_locked_hidden(capsys):
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Bin4D(data, disabled_tools=["display"], hidden_tools=["binning"])
    w.summary()
    output = capsys.readouterr().out
    assert "Locked" in output
    assert "display" in output
    assert "Hidden" in output
    assert "binning" in output

# --- Detector preview (mean DP) tests ---

def test_bin_mean_dp_bytes_populated():
    data = np.random.rand(4, 6, 16, 20).astype(np.float32)
    w = Bin4D(data)
    assert len(w.original_mean_dp_bytes) == 16 * 20 * 4
    assert len(w.binned_mean_dp_bytes) == 16 * 20 * 4

def test_bin_mean_dp_updates_after_binning():
    data = np.random.rand(8, 8, 16, 16).astype(np.float32)
    w = Bin4D(data)
    assert len(w.original_mean_dp_bytes) == 16 * 16 * 4

    w.det_bin_row = 2
    w.det_bin_col = 2
    # Binned detector is 8x8
    assert len(w.binned_mean_dp_bytes) == 8 * 8 * 4
    # Original stays at original detector size
    assert len(w.original_mean_dp_bytes) == 16 * 16 * 4

def test_bin_binned_center_traits():
    data = np.random.rand(4, 4, 16, 16).astype(np.float32)
    w = Bin4D(data, center=(8.0, 8.0))
    assert w.center_row == 8.0
    assert w.center_col == 8.0

    w.det_bin_row = 2
    w.det_bin_col = 2
    assert w.binned_center_row == 4.0
    assert w.binned_center_col == 4.0

def test_bin_position_dp_bytes():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Bin4D(data)
    assert w._position_dp_bytes == b""
    assert w._binned_position_dp_bytes == b""
    w._scan_position = [2, 3]
    assert len(w._position_dp_bytes) == 8 * 8 * 4
    assert len(w._binned_position_dp_bytes) == 8 * 8 * 4
    dp = np.frombuffer(w._position_dp_bytes, dtype=np.float32).reshape(8, 8)
    np.testing.assert_allclose(dp, data[2, 3], atol=1e-6)

def test_bin_position_dp_clears_on_negative():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Bin4D(data)
    w._scan_position = [1, 1]
    assert len(w._position_dp_bytes) > 0
    w._scan_position = [-1, -1]
    assert w._position_dp_bytes == b""
    assert w._binned_position_dp_bytes == b""

def test_bin_position_dp_updates_on_rebin():
    data = np.random.rand(4, 4, 8, 8).astype(np.float32)
    w = Bin4D(data)
    w._scan_position = [0, 0]
    dp_before = w._binned_position_dp_bytes
    w.scan_bin_row = 2
    dp_after = w._binned_position_dp_bytes
    assert dp_after != dp_before or len(dp_after) != len(dp_before)

def test_bin_deprecated_alias():
    from quantem.widget import Bin
    assert Bin is Bin4D
