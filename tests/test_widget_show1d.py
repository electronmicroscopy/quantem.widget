import json

import numpy as np
import pytest
import torch

from quantem.widget import Show1D

# =========================================================================
# Basic construction
# =========================================================================
def test_show1d_single_numpy():
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    w = Show1D(data)
    assert w.n_traces == 1
    assert w.n_points == 4
    assert len(w.y_bytes) == 4 * 4

def test_show1d_single_torch():
    data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    w = Show1D(data)
    assert w.n_traces == 1
    assert w.n_points == 3

def test_show1d_multiple_traces_2d():
    data = np.random.randn(3, 100).astype(np.float32)
    w = Show1D(data)
    assert w.n_traces == 3
    assert w.n_points == 100

def test_show1d_multiple_traces_list():
    t1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    w = Show1D([t1, t2], labels=["A", "B"])
    assert w.n_traces == 2
    assert w.n_points == 3
    assert w.labels == ["A", "B"]

def test_show1d_with_x_data():
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    w = Show1D(y, x=x)
    assert len(w.x_bytes) == 3 * 4

def test_show1d_x_length_mismatch_raises():
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = np.array([0.0, 1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="points"):
        Show1D(y, x=x)

def test_show1d_list_length_mismatch_raises():
    t1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t2 = np.array([4.0, 5.0], dtype=np.float32)
    with pytest.raises(ValueError, match="same length"):
        Show1D([t1, t2])

def test_show1d_3d_raises():
    data = np.zeros((2, 3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="1D or 2D"):
        Show1D(data)

# =========================================================================
# Statistics
# =========================================================================
def test_show1d_stats_single():
    data = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    w = Show1D(data)
    assert w.stats_mean[0] == pytest.approx(20.0)
    assert w.stats_min[0] == pytest.approx(10.0)
    assert w.stats_max[0] == pytest.approx(30.0)

def test_show1d_stats_multiple():
    t1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t2 = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    w = Show1D([t1, t2])
    assert len(w.stats_mean) == 2
    assert w.stats_mean[0] == pytest.approx(2.0)
    assert w.stats_mean[1] == pytest.approx(20.0)

# =========================================================================
# Display options
# =========================================================================
def test_show1d_title():
    w = Show1D(np.ones(5, dtype=np.float32), title="My Plot")
    assert w.title == "My Plot"

def test_show1d_labels_default():
    w = Show1D(np.ones(5, dtype=np.float32))
    assert w.labels == ["Data"]

def test_show1d_labels_multi_default():
    w = Show1D(np.ones((3, 5), dtype=np.float32))
    assert w.labels == ["Data 1", "Data 2", "Data 3"]

def test_show1d_colors():
    w = Show1D(np.ones(5, dtype=np.float32), colors=["#ff0000"])
    assert w.colors == ["#ff0000"]

def test_show1d_colors_default():
    w = Show1D(np.ones((2, 5), dtype=np.float32))
    assert len(w.colors) == 2
    assert w.colors[0] == "#4fc3f7"
    assert w.colors[1] == "#81c784"

def test_show1d_axis_labels():
    w = Show1D(np.ones(5, dtype=np.float32), x_label="Energy", y_label="Counts", x_unit="eV", y_unit="a.u.")
    assert w.x_label == "Energy"
    assert w.y_label == "Counts"
    assert w.x_unit == "eV"
    assert w.y_unit == "a.u."

def test_show1d_log_scale():
    w = Show1D(np.ones(5, dtype=np.float32), log_scale=True)
    assert w.log_scale is True

def test_show1d_show_grid():
    w = Show1D(np.ones(5, dtype=np.float32), show_grid=False)
    assert w.show_grid is False

def test_show1d_show_legend():
    w = Show1D(np.ones(5, dtype=np.float32), show_legend=False)
    assert w.show_legend is False

def test_show1d_show_stats():
    w = Show1D(np.ones(5, dtype=np.float32), show_stats=False)
    assert w.show_stats is False

def test_show1d_line_width():
    w = Show1D(np.ones(5, dtype=np.float32), line_width=2.5)
    assert w.line_width == pytest.approx(2.5)

# =========================================================================
# Mutation methods
# =========================================================================
def test_show1d_set_data():
    w = Show1D(np.array([1.0, 2.0, 3.0], dtype=np.float32), title="Keep", log_scale=True)
    w.set_data(np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32))
    assert w.n_points == 4
    assert w.n_traces == 1
    assert w.title == "Keep"
    assert w.log_scale is True

def test_show1d_set_data_with_x():
    w = Show1D(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    x = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    w.set_data(np.array([4.0, 5.0, 6.0], dtype=np.float32), x=x)
    assert len(w.x_bytes) == 3 * 4

def test_show1d_add_trace():
    w = Show1D(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert w.n_traces == 1
    w.add_trace(np.array([4.0, 5.0, 6.0], dtype=np.float32), label="New")
    assert w.n_traces == 2
    assert w.labels[-1] == "New"

def test_show1d_add_trace_length_mismatch():
    w = Show1D(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    with pytest.raises(ValueError, match="points"):
        w.add_trace(np.array([4.0, 5.0], dtype=np.float32))

def test_show1d_remove_trace():
    w = Show1D(np.ones((3, 5), dtype=np.float32), labels=["A", "B", "C"])
    w.remove_trace(1)
    assert w.n_traces == 2
    assert w.labels == ["A", "C"]

def test_show1d_remove_trace_out_of_range():
    w = Show1D(np.ones(5, dtype=np.float32))
    with pytest.raises(IndexError):
        w.remove_trace(5)

def test_show1d_clear():
    w = Show1D(np.ones((3, 5), dtype=np.float32))
    w.clear()
    assert w.n_traces == 0
    assert w.n_points == 0
    assert w.labels == []
    assert w.colors == []
    assert w.y_bytes == b""

# =========================================================================
# State persistence (required 3 tests)
# =========================================================================
def test_show1d_state_dict_roundtrip():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(
        data,
        title="Test",
        x_label="Energy",
        y_label="Counts",
        x_unit="eV",
        log_scale=True,
        show_grid=False,
        show_legend=False,
        line_width=2.0,
        grid_density=20,
        peak_active=True,
    )
    sd = w.state_dict()
    assert "grid_density" in sd
    assert "peak_active" in sd
    assert sd["grid_density"] == 20
    assert sd["peak_active"] is True
    w2 = Show1D(data, state=sd)
    assert w2.title == "Test"
    assert w2.x_label == "Energy"
    assert w2.y_label == "Counts"
    assert w2.x_unit == "eV"
    assert w2.log_scale is True
    assert w2.show_grid is False
    assert w2.show_legend is False
    assert w2.line_width == pytest.approx(2.0)
    assert w2.grid_density == 20
    assert w2.peak_active is True

def test_show1d_save_load_file(tmp_path):
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data, title="Save Test", log_scale=True)
    path = tmp_path / "show1d_state.json"
    w.save(str(path))
    assert path.exists()
    saved = json.loads(path.read_text())
    assert saved["metadata_version"] == "1.0"
    assert saved["widget_name"] == "Show1D"
    assert isinstance(saved["widget_version"], str)
    assert saved["state"]["title"] == "Save Test"
    w2 = Show1D(data, state=str(path))
    assert w2.title == "Save Test"
    assert w2.log_scale is True

def test_show1d_summary(capsys):
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data, title="My Plot", x_label="Distance", x_unit="nm")
    w.summary()
    out = capsys.readouterr().out
    assert "My Plot" in out
    assert "3" in out
    assert "Distance" in out

# =========================================================================
# Repr
# =========================================================================
def test_show1d_repr_single():
    w = Show1D(np.ones(100, dtype=np.float32))
    r = repr(w)
    assert "Show1D" in r
    assert "100" in r

def test_show1d_repr_multi():
    w = Show1D(np.ones((3, 50), dtype=np.float32))
    r = repr(w)
    assert "Show1D" in r
    assert "3" in r
    assert "50" in r

# =========================================================================
# Edge cases
# =========================================================================
def test_show1d_constant_trace():
    data = np.full(100, 5.0, dtype=np.float32)
    w = Show1D(data)
    assert w.stats_mean[0] == pytest.approx(5.0)
    assert w.stats_std[0] == pytest.approx(0.0)

def test_show1d_single_point():
    data = np.array([42.0], dtype=np.float32)
    w = Show1D(data)
    assert w.n_points == 1
    assert w.stats_mean[0] == pytest.approx(42.0)

def test_show1d_dataset_duck_typing():
    class MockDataset:
        def __init__(self):
            self.array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            self.name = "Test Spectrum"

    ds = MockDataset()
    w = Show1D(ds)
    assert w.title == "Test Spectrum"
    assert w.n_points == 3

def test_show1d_set_data_returns_self():
    w = Show1D(np.ones(5, dtype=np.float32))
    result = w.set_data(np.ones(3, dtype=np.float32))
    assert result is w

def test_show1d_add_trace_returns_self():
    w = Show1D(np.ones(5, dtype=np.float32))
    result = w.add_trace(np.ones(5, dtype=np.float32))
    assert result is w

def test_show1d_remove_trace_returns_self():
    w = Show1D(np.ones((2, 5), dtype=np.float32))
    result = w.remove_trace(0)
    assert result is w

def test_show1d_clear_returns_self():
    w = Show1D(np.ones(5, dtype=np.float32))
    result = w.clear()
    assert result is w

def test_show1d_data_bytes_match():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data)
    recovered = np.frombuffer(w.y_bytes, dtype=np.float32)
    np.testing.assert_array_equal(recovered, data)

# =========================================================================
# Widget version / show_controls defaults
# =========================================================================
def test_show1d_widget_version_is_set():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data)
    assert w.widget_version != "unknown"

def test_show1d_show_controls_default():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data)
    assert w.show_controls is True


# =========================================================================
# Peak markers
# =========================================================================
def test_show1d_add_peak():
    data = np.array([1.0, 3.0, 5.0, 4.0, 2.0], dtype=np.float32)
    w = Show1D(data)
    w.add_peak(2.0)  # near index 2, which is the max
    assert len(w.peak_markers) == 1
    assert w.peak_markers[0]["x"] == pytest.approx(2.0)
    assert w.peak_markers[0]["y"] == pytest.approx(5.0)
    assert w.peak_markers[0]["trace_idx"] == 0
    assert w.peak_markers[0]["type"] == "peak"

def test_show1d_add_peak_local_max_search():
    # Peak is at index 5 (value 10), click near index 3
    data = np.array([1, 2, 3, 4, 6, 10, 7, 3, 1, 0], dtype=np.float32)
    w = Show1D(data)
    w.add_peak(3.0)  # near index 3, should find max at index 5
    assert w.peak_markers[0]["x"] == pytest.approx(5.0)
    assert w.peak_markers[0]["y"] == pytest.approx(10.0)

def test_show1d_add_peak_with_x():
    data = np.array([1.0, 5.0, 3.0], dtype=np.float32)
    x = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    w = Show1D(data, x=x)
    w.add_peak(150.0)  # nearest to x=200, which is the max
    assert w.peak_markers[0]["x"] == pytest.approx(200.0)
    assert w.peak_markers[0]["y"] == pytest.approx(5.0)

def test_show1d_remove_peak():
    data = np.array([1.0, 5.0, 3.0, 7.0, 2.0], dtype=np.float32)
    w = Show1D(data)
    w.add_peak(1.0)
    w.add_peak(3.0)
    assert len(w.peak_markers) == 2
    w.remove_peak()  # remove last
    assert len(w.peak_markers) == 1

def test_show1d_clear_peaks():
    data = np.array([1.0, 5.0, 3.0], dtype=np.float32)
    w = Show1D(data)
    w.add_peak(1.0)
    w.add_peak(0.0)
    w.clear_peaks()
    assert len(w.peak_markers) == 0

def test_show1d_peaks_property():
    data = np.array([1.0, 5.0, 3.0], dtype=np.float32)
    w = Show1D(data)
    w.add_peak(1.0)
    peaks = w.peaks
    assert len(peaks) == 1
    assert isinstance(peaks, list)

def test_show1d_add_peak_returns_self():
    data = np.array([1.0, 5.0, 3.0], dtype=np.float32)
    w = Show1D(data)
    result = w.add_peak(1.0)
    assert result is w

def test_show1d_add_peak_invalid_trace():
    data = np.array([1.0, 5.0, 3.0], dtype=np.float32)
    w = Show1D(data)
    with pytest.raises(IndexError):
        w.add_peak(1.0, trace_idx=5)

def test_show1d_peak_markers_in_state():
    data = np.array([1.0, 5.0, 3.0], dtype=np.float32)
    w = Show1D(data)
    w.add_peak(1.0)
    sd = w.state_dict()
    assert len(sd["peak_markers"]) == 1
    w2 = Show1D(data, state=sd)
    assert len(w2.peak_markers) == 1


# =========================================================================
# find_peaks (scipy-based auto-detection)
# =========================================================================
def test_show1d_find_peaks_basic():
    # Create data with clear peaks at indices 10, 30, 50
    x = np.linspace(0, 100, 200, dtype=np.float32)
    y = np.zeros(200, dtype=np.float32)
    for peak_idx in [40, 100, 160]:
        y += 5.0 * np.exp(-0.5 * ((np.arange(200) - peak_idx) / 3.0) ** 2).astype(np.float32)
    w = Show1D(y, x=x)
    w.find_peaks(prominence=1.0)
    assert len(w.peak_markers) == 3

def test_show1d_find_peaks_with_height():
    y = np.array([0, 1, 0, 3, 0, 2, 0], dtype=np.float32)
    w = Show1D(y)
    w.find_peaks(height=2.5, prominence=0.5)
    assert len(w.peak_markers) == 1
    assert w.peak_markers[0]["y"] == pytest.approx(3.0)
    assert w.peak_markers[0]["type"] == "peak"

def test_show1d_find_peaks_appends():
    y = np.array([0, 5, 0, 3, 0], dtype=np.float32)
    w = Show1D(y)
    w.add_peak(1.0)
    w.find_peaks(prominence=0.5)
    # Should have the manual peak + auto-detected peaks
    assert len(w.peak_markers) >= 2

def test_show1d_find_peaks_returns_self():
    y = np.array([0, 5, 0], dtype=np.float32)
    w = Show1D(y)
    result = w.find_peaks(prominence=0.5)
    assert result is w

def test_show1d_find_peaks_invalid_trace():
    y = np.array([0, 5, 0], dtype=np.float32)
    w = Show1D(y)
    with pytest.raises(IndexError):
        w.find_peaks(trace_idx=5)

def test_show1d_find_peaks_with_x_values():
    x = np.array([100, 200, 300, 400, 500], dtype=np.float32)
    y = np.array([0, 5, 0, 3, 0], dtype=np.float32)
    w = Show1D(y, x=x)
    w.find_peaks(prominence=0.5)
    # Peak at index 1 should have x=200
    peak_xs = [m["x"] for m in w.peak_markers]
    assert 200.0 in peak_xs


# =========================================================================
# export_peaks
# =========================================================================
def test_show1d_export_peaks_csv(tmp_path):
    y = np.array([0, 5, 0, 3, 0], dtype=np.float32)
    w = Show1D(y)
    w.add_peak(1.0)
    w.add_peak(3.0)
    path = tmp_path / "peaks.csv"
    result = w.export_peaks(str(path))
    assert result == path
    assert path.exists()
    content = path.read_text()
    assert "x,y,trace_idx,label,type" in content
    lines = content.strip().split("\n")
    assert len(lines) == 3  # header + 2 peaks
    assert "peak" in lines[1]

def test_show1d_export_peaks_json(tmp_path):
    y = np.array([0, 5, 0, 3, 0], dtype=np.float32)
    w = Show1D(y)
    w.add_peak(1.0)
    path = tmp_path / "peaks.json"
    result = w.export_peaks(str(path))
    assert result == path
    data = json.loads(path.read_text())
    assert isinstance(data, list)
    assert len(data) == 1
    assert "x" in data[0]
    assert "y" in data[0]

def test_show1d_export_peaks_empty_raises():
    y = np.array([0, 5, 0], dtype=np.float32)
    w = Show1D(y)
    with pytest.raises(ValueError, match="No peak markers"):
        w.export_peaks("peaks.csv")

def test_show1d_export_peaks_bad_format():
    y = np.array([0, 5, 0], dtype=np.float32)
    w = Show1D(y)
    w.add_peak(1.0)
    with pytest.raises(ValueError, match="Unsupported format"):
        w.export_peaks("peaks.txt")


# =========================================================================
# save_image
# =========================================================================
def test_show1d_save_image_png(tmp_path):
    data = np.array([1.0, 3.0, 5.0, 3.0, 1.0], dtype=np.float32)
    w = Show1D(data, title="Test", x_label="Energy", x_unit="eV", y_label="Counts")
    path = tmp_path / "test.png"
    result = w.save_image(path)
    assert result == path
    assert path.exists()
    assert path.stat().st_size > 0

def test_show1d_save_image_pdf(tmp_path):
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data)
    path = tmp_path / "test.pdf"
    result = w.save_image(path)
    assert result == path
    assert path.exists()

def test_show1d_save_image_with_peaks(tmp_path):
    data = np.array([0, 5, 0, 1, 5, 0], dtype=np.float32)
    w = Show1D(data)
    w.add_peak(1.0)
    w.add_peak(4.0)
    path = tmp_path / "peaks.png"
    result = w.save_image(path, include_peaks=True)
    assert result == path
    assert path.exists()

def test_show1d_save_image_multi_trace(tmp_path):
    data = np.random.randn(3, 50).astype(np.float32)
    w = Show1D(data, labels=["A", "B", "C"])
    path = tmp_path / "multi.png"
    result = w.save_image(path)
    assert result == path
    assert path.exists()

def test_show1d_save_image_log_scale(tmp_path):
    data = np.array([1, 10, 100, 1000, 100, 10, 1], dtype=np.float32)
    w = Show1D(data, log_scale=True)
    path = tmp_path / "log.png"
    result = w.save_image(path)
    assert result == path
    assert path.exists()

def test_show1d_save_image_bad_format():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data)
    with pytest.raises(ValueError, match="Unsupported format"):
        w.save_image("test.bmp")


# =========================================================================
# Tool lock/hide
# =========================================================================
def test_show1d_disabled_tools_default():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data)
    assert w.disabled_tools == []

def test_show1d_disabled_tools_custom():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data, disabled_tools=["display", "PEAKS"])
    assert w.disabled_tools == ["display", "peaks"]

def test_show1d_disabled_tools_flags():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data, disable_display=True, disable_stats=True)
    assert w.disabled_tools == ["display", "stats"]

def test_show1d_disabled_tools_disable_all():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data, disable_all=True, disable_display=True)
    assert w.disabled_tools == ["all"]

def test_show1d_disabled_tools_unknown_raises():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(ValueError, match="Unknown tool group"):
        Show1D(data, disabled_tools=["not_real"])

def test_show1d_disabled_tools_trait_assignment_normalizes():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data)
    w.disabled_tools = ["DISPLAY", "display", "peaks"]
    assert w.disabled_tools == ["display", "peaks"]

def test_show1d_hidden_tools_default():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data)
    assert w.hidden_tools == []

def test_show1d_hidden_tools_custom():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data, hidden_tools=["display", "EXPORT"])
    assert w.hidden_tools == ["display", "export"]

def test_show1d_hidden_tools_flags():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data, hide_peaks=True, hide_export=True)
    assert w.hidden_tools == ["peaks", "export"]

def test_show1d_hidden_tools_hide_all():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data, hide_all=True, hide_display=True)
    assert w.hidden_tools == ["all"]

def test_show1d_hidden_tools_unknown_raises():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(ValueError, match="Unknown tool group"):
        Show1D(data, hidden_tools=["not_real"])

def test_show1d_hidden_tools_trait_assignment_normalizes():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data)
    w.hidden_tools = ["DISPLAY", "display", "export"]
    assert w.hidden_tools == ["display", "export"]

def test_show1d_tool_lock_hide_state_dict_roundtrip():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data, disabled_tools=["display"], hidden_tools=["stats"])
    state = w.state_dict()
    assert state["disabled_tools"] == ["display"]
    assert state["hidden_tools"] == ["stats"]
    w2 = Show1D(data, state=state)
    assert w2.disabled_tools == ["display"]
    assert w2.hidden_tools == ["stats"]

def test_show1d_tool_lock_hide_runtime_api():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data)
    w.lock_tool("display")
    assert "display" in w.disabled_tools
    w.unlock_tool("display")
    assert "display" not in w.disabled_tools
    w.hide_tool("stats")
    assert "stats" in w.hidden_tools
    w.show_tool("stats")
    assert "stats" not in w.hidden_tools

def test_show1d_tool_lock_hide_summary(capsys):
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data, disabled_tools=["display"], hidden_tools=["peaks"])
    w.summary()
    out = capsys.readouterr().out
    assert "Locked:" in out
    assert "display" in out
    assert "Hidden:" in out
    assert "peaks" in out


# =========================================================================
# Grid density
# =========================================================================
def test_show1d_grid_density_default():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data)
    assert w.grid_density == 10

def test_show1d_grid_density_in_state():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data, grid_density=25)
    sd = w.state_dict()
    assert sd["grid_density"] == 25
    w2 = Show1D(data, state=sd)
    assert w2.grid_density == 25


# =========================================================================
# Peak active
# =========================================================================
def test_show1d_peak_active_default():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data)
    assert w.peak_active is False

def test_show1d_peak_active_in_state():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data, peak_active=True)
    sd = w.state_dict()
    assert sd["peak_active"] is True
    w2 = Show1D(data, state=sd)
    assert w2.peak_active is True


def test_show1d_peak_search_radius_default():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data)
    assert w.peak_search_radius == 20


def test_show1d_peak_search_radius_custom():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data, peak_search_radius=50)
    assert w.peak_search_radius == 50


def test_show1d_peak_search_radius_in_state():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data, peak_search_radius=35)
    sd = w.state_dict()
    assert sd["peak_search_radius"] == 35
    w2 = Show1D(data, state=sd)
    assert w2.peak_search_radius == 35


# =========================================================================
# Selected peaks
# =========================================================================
def test_show1d_selected_peaks_default():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data)
    assert w.selected_peaks == []

def test_show1d_selected_peaks_sync():
    data = np.array([1.0, 5.0, 3.0, 7.0, 2.0], dtype=np.float32)
    w = Show1D(data)
    w.add_peak(1.0)
    w.add_peak(3.0)
    w.selected_peaks = [0, 1]
    result = w.selected_peak_data
    assert len(result) == 2
    assert result[0] == w.peak_markers[0]
    assert result[1] == w.peak_markers[1]

def test_show1d_selected_peaks_out_of_range():
    data = np.array([1.0, 5.0, 3.0], dtype=np.float32)
    w = Show1D(data)
    w.add_peak(1.0)
    w.selected_peaks = [0, 5, 99]
    result = w.selected_peak_data
    assert len(result) == 1

def test_show1d_clear_peaks_clears_selection():
    data = np.array([1.0, 5.0, 3.0], dtype=np.float32)
    w = Show1D(data)
    w.add_peak(1.0)
    w.selected_peaks = [0]
    w.clear_peaks()
    assert w.selected_peaks == []
    assert w.peak_markers == []

def test_show1d_summary_with_peaks(capsys):
    data = np.array([1.0, 5.0, 3.0], dtype=np.float32)
    w = Show1D(data)
    w.add_peak(1.0)
    w.summary()
    out = capsys.readouterr().out
    assert "1 markers" in out


def test_show1d_x_range_default():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data)
    assert w.x_range == []
    assert w.y_range == []


def test_show1d_x_range_set():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data)
    w.x_range = [0.5, 2.5]
    assert w.x_range == [0.5, 2.5]


def test_show1d_y_range_set():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data)
    w.y_range = [1.0, 2.0]
    assert w.y_range == [1.0, 2.0]


def test_show1d_range_in_state_dict():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data)
    w.x_range = [10.0, 50.0]
    w.y_range = [0.0, 100.0]
    s = w.state_dict()
    assert s["x_range"] == [10.0, 50.0]
    assert s["y_range"] == [0.0, 100.0]

    w2 = Show1D(data, state=s)
    assert w2.x_range == [10.0, 50.0]
    assert w2.y_range == [0.0, 100.0]


def test_show1d_range_unlock():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data)
    w.x_range = [0.5, 2.5]
    assert w.x_range == [0.5, 2.5]
    w.x_range = []
    assert w.x_range == []


# =========================================================================
# Range Stats (Feature 1+4)
# =========================================================================
def test_show1d_range_stats_basic():
    data = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    w = Show1D(data, x=x)
    w.x_range = [1.0, 3.0]
    assert len(w.range_stats) == 1
    rs = w.range_stats[0]
    assert rs["mean"] == pytest.approx(30.0)
    assert rs["min"] == pytest.approx(20.0)
    assert rs["max"] == pytest.approx(40.0)
    assert rs["n_points"] == 3


def test_show1d_range_stats_integral():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    w = Show1D(data, x=x)
    w.x_range = [0.0, 4.0]
    rs = w.range_stats[0]
    expected = float(np.trapezoid(data, x=x))
    assert rs["integral"] == pytest.approx(expected, rel=1e-5)


def test_show1d_range_stats_cleared_when_empty():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data)
    w.x_range = [0.0, 2.0]
    assert len(w.range_stats) == 1
    w.x_range = []
    assert w.range_stats == []


def test_show1d_range_stats_with_custom_x():
    data = np.array([1.0, 5.0, 3.0, 7.0, 2.0], dtype=np.float32)
    x = np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float32)
    w = Show1D(data, x=x)
    w.x_range = [200.0, 400.0]
    rs = w.range_stats[0]
    assert rs["n_points"] == 3
    assert rs["mean"] == pytest.approx(5.0)


def test_show1d_range_stats_updated_on_data_change():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    w = Show1D(data, x=x)
    w.x_range = [0.0, 4.0]
    assert w.range_stats[0]["mean"] == pytest.approx(3.0)
    w.set_data(np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32), x=x)
    w.x_range = [0.0, 4.0]
    assert w.range_stats[0]["mean"] == pytest.approx(30.0)


# =========================================================================
# Peak FWHM (Feature 5)
# =========================================================================
def test_show1d_fwhm_known_gaussian():
    # Create a Gaussian with known sigma → FWHM = 2.3548 * sigma
    sigma = 5.0
    x = np.linspace(-50, 50, 1001, dtype=np.float32)
    y = (10.0 * np.exp(-0.5 * (x / sigma) ** 2) + 1.0).astype(np.float32)
    w = Show1D(y, x=x)
    w.add_peak(0.0)
    w.selected_peaks = [0]
    assert len(w.peak_fwhm) == 1
    f = w.peak_fwhm[0]
    assert f["fwhm"] is not None
    expected_fwhm = 2.3548200450309493 * sigma
    assert f["fwhm"] == pytest.approx(expected_fwhm, rel=0.05)


def test_show1d_fwhm_no_selection():
    data = np.array([1.0, 5.0, 3.0], dtype=np.float32)
    w = Show1D(data)
    w.add_peak(1.0)
    # No selection → no FWHM
    assert w.peak_fwhm == []


def test_show1d_fwhm_fit_failure():
    # Flat data → Gaussian fit should fail or give poor result
    data = np.full(100, 5.0, dtype=np.float32)
    w = Show1D(data)
    w.add_peak(50.0)
    w.selected_peaks = [0]
    # Should produce a result (possibly with error or None fwhm)
    assert len(w.peak_fwhm) == 1


def test_show1d_measure_fwhm_method():
    sigma = 5.0
    x = np.linspace(-50, 50, 1001, dtype=np.float32)
    y = (10.0 * np.exp(-0.5 * (x / sigma) ** 2) + 1.0).astype(np.float32)
    w = Show1D(y, x=x)
    w.add_peak(0.0)
    result = w.measure_fwhm(peak_idx=0)
    assert result is w
    assert len(w.peak_fwhm) == 1
    assert 0 in w.selected_peaks


# =========================================================================
# Auto-contrast / percentile clipping
# =========================================================================
def test_show1d_auto_contrast_default():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data)
    assert w.auto_contrast is False


def test_show1d_auto_contrast_state_dict():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data, auto_contrast=True, percentile_low=5.0, percentile_high=95.0)
    sd = w.state_dict()
    assert sd["auto_contrast"] is True
    assert sd["percentile_low"] == 5.0
    assert sd["percentile_high"] == 95.0
    w2 = Show1D(data, state=sd)
    assert w2.auto_contrast is True
    assert w2.percentile_low == pytest.approx(5.0)
    assert w2.percentile_high == pytest.approx(95.0)


def test_show1d_percentile_traits():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    w = Show1D(data)
    assert w.percentile_low == pytest.approx(2.0)
    assert w.percentile_high == pytest.approx(98.0)
    w2 = Show1D(data, percentile_low=10.0, percentile_high=90.0)
    assert w2.percentile_low == pytest.approx(10.0)
    assert w2.percentile_high == pytest.approx(90.0)


def test_show1d_save_image_auto_contrast(tmp_path):
    # Spectrum with a huge outlier — auto-contrast should still export fine
    data = np.array([1.0, 2.0, 3.0, 100.0, 2.0, 1.0], dtype=np.float32)
    w = Show1D(data, auto_contrast=True, percentile_low=10.0, percentile_high=90.0)
    path = tmp_path / "auto.png"
    result = w.save_image(path)
    assert result == path
    assert path.exists()
    assert path.stat().st_size > 0
