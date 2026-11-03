"""Review a calibrated correlation map, change its window, and export it."""

import base64
import json

import matplotlib.pyplot as plt
import numpy as np
import pytest
from ipywidgets import Widget
from ipywidgets.embed import embed_data

from quantem.widget import Plot2D, Show1D, Show2D


def test_calibrated_map_export_and_saved_state(tmp_path):
    radius = (np.arange(100) + 0.5) * 0.1
    angles = (np.arange(36) + 0.5) * 5
    truth = np.cos(np.deg2rad(angles[:, None])) ** 2 * radius[None, :]
    plot = Plot2D(
        truth,
        x=radius,
        y=angles,
        x_label="r02 (Å)",
        y_label="Angle (°)",
        colorbar_label="Normalized G3",
        width=1200,
        max_width=420,
        save_state=True,
    )
    assert plot.layout.width == "420px"
    assert plot.layout.max_width == "min(100%, 420px)"
    original = truth.copy()
    np.testing.assert_allclose(plot.grid["bounds"], [0, 10, 0, 180], atol=1e-14)
    plot.view_bounds = [1, 8, 30, 150]
    plot.horizontal_line = 92.5
    plot.set_data(truth * 0.75)
    fixed_limits = (plot.vmin, plot.vmax)
    plot.cmap = "magma"
    figures_before = plt.get_fignums()
    figure = plot.figure()
    assert plt.get_fignums() == figures_before
    axes = figure.axes[0]
    np.testing.assert_array_equal(axes.collections[0].get_array(), truth * 0.75)
    np.testing.assert_array_equal(axes.get_xlim(), [1, 8])
    np.testing.assert_array_equal(axes.get_ylim(), [30, 150])
    assert axes.get_xlabel() == "r02 (Å)"
    assert figure.axes[1].get_ylabel() == "Normalized G3"
    assert axes.collections[0].cmap.name == "magma"
    assert axes.collections[0].get_clim() == fixed_limits
    figure.savefig(tmp_path / "g3.svg")
    own_state = Widget.get_manager_state(widgets=[plot, plot.layout])["state"]
    state = embed_data(views=[plot], state=own_state)["manager_state"]["state"][
        plot.model_id
    ]
    assert state["state"]["horizontal_line"] == 92.5
    buffer = next(item for item in state["buffers"] if item["path"] == ["data_bytes"])
    restored = np.frombuffer(
        base64.b64decode(buffer["data"]), dtype=np.float64, count=truth.size
    )
    np.testing.assert_array_equal(restored, (truth * 0.75).ravel())
    np.testing.assert_array_equal(truth, original)


def test_asymmetric_map_retains_row_column_calibration():
    """Export a nonsquare, strided map without swapping axes or changing values."""
    radius = np.array([1.0, 1.5, 2.0])
    angle = np.array([30.0, 90.0])
    values = np.arange(12, dtype=np.float64).reshape(2, 6)[:, ::2]
    plot = Plot2D(values, x=radius, y=angle, vmin=-1, vmax=12)
    figure = plot.figure()
    mesh = figure.axes[0].collections[0]
    expected_column_edges = [0.75, 1.25, 1.75, 2.25]
    expected_row_edges = [0.0, 60.0, 120.0]
    np.testing.assert_array_equal(
        mesh.get_coordinates()[0, :, 0], expected_column_edges
    )
    np.testing.assert_array_equal(mesh.get_coordinates()[:, 0, 1], expected_row_edges)
    np.testing.assert_array_equal(mesh.get_array(), values)
    # Reusing caller-owned input arrays cannot change the already-created map.
    values[:] = -1
    radius[:] = 0
    angle[:] = 0
    np.testing.assert_array_equal(
        plot.figure().axes[0].collections[0].get_array(), [[0, 2, 4], [6, 8, 10]]
    )


def test_meter_calibration_preserves_uniform_grid_requirement():
    """Changing distance units to meters must not allow misleading axes."""
    distances_m = np.array([1, 2, 3]) * 1e-10
    angles = np.array([30, 90])
    values = np.arange(6).reshape(2, 3)
    plot = Plot2D(values, x=distances_m, y=angles, x_label="Distance (m)")
    np.testing.assert_allclose(
        plot.grid["bounds"][:2], [0.5e-10, 3.5e-10], rtol=1e-14, atol=0
    )
    for invalid in ([0, 1e-10, 5e-11], [0, 1e-10, 1e-10], [0, 1e-10, 5e-9]):
        with pytest.raises(ValueError, match="uniformly spaced bin centers"):
            Plot2D(values, x=np.array(invalid), y=angles)


def test_lightweight_snapshot_preserves_live_values_and_current_preview():
    """Save a calibrated view without embedding its source array by default."""
    values = np.arange(24, dtype=float).reshape(4, 6)
    plot = Plot2D(values, x=np.arange(6), y=np.arange(4))
    first = plot.get_state()
    assert "data_bytes" not in first
    assert first["_static_fallback_mime"] == "image/png"
    assert base64.b64decode(first["_static_fallback_image"]).startswith(b"\x89PNG")
    plot.view_bounds = [1, 4, 0, 2]
    plot.horizontal_line = 1.5
    plot.set_data(values[::-1])
    saved = plot.get_state()
    assert saved["_static_fallback_image"] != first["_static_fallback_image"]
    assert saved["view_bounds"] == [1, 4, 0, 2]
    assert saved["horizontal_line"] == 1.5
    assert "data_bytes" not in saved
    live = plot.get_state({"data_bytes", "grid"})
    np.testing.assert_array_equal(
        np.frombuffer(live["data_bytes"], dtype=np.float64, count=values.size),
        values[::-1].ravel(),
    )
    plot.close()


def test_saved_view_loads_onto_a_new_map(tmp_path):
    """Save the color window, labels and viewport, then restore them onto a fresh map of the same grid."""
    values = np.arange(24, dtype=float).reshape(4, 6)
    plot = Plot2D(values, x=np.arange(6), y=np.arange(4), title="G3", x_label="r02 (Å)")
    plot.cmap = "magma"
    plot.vmin, plot.vmax = 2.0, 20.0
    plot.view_bounds = [1, 4, 0, 2]
    plot.horizontal_line = 1.5
    path = tmp_path / "plot2d_view.json"
    plot.save(path)
    assert json.loads(path.read_text())["widget_name"] == "Plot2D"
    restored = Plot2D(values, x=np.arange(6), y=np.arange(4))
    restored.load_state_dict(path)
    assert restored.state_dict() == plot.state_dict()
    assert Plot2D(values, x=np.arange(6), y=np.arange(4), state=path).state_dict() == plot.state_dict()
    axes = restored.figure().axes[0]
    assert axes.collections[0].cmap.name == "magma"
    assert axes.collections[0].get_clim() == (2.0, 20.0)
    np.testing.assert_array_equal(axes.get_xlim(), [1, 4])


def test_restored_view_is_checked_like_a_new_one(tmp_path):
    """Refuse another widget's file and unmappable color limits; a restored width shows in the notebook box."""
    values = np.arange(24, dtype=float).reshape(4, 6)
    plot = Plot2D(values, x=np.arange(6), y=np.arange(4), width=500, max_width=600)
    Show1D(np.arange(8.0)).save(tmp_path / "show1d.json")
    Show2D(np.ones((4, 4), np.float32), verbose=False).save(tmp_path / "show2d.json")
    for path in (tmp_path / "show1d.json", tmp_path / "show2d.json"):
        with pytest.raises(ValueError, match="cannot load into 'Plot2D'"):
            plot.load_state_dict(path)
        with pytest.raises(ValueError, match="cannot load into 'Plot2D'"):
            plot.load_state_dict(json.loads(path.read_text()))
    for limits in ({"vmin": 5.0, "vmax": 1.0}, {"vmin": float("nan")}, {"vmax": float("inf")}):
        with pytest.raises(ValueError, match="finite color limits"):
            plot.load_state_dict({**limits, "cmap": "magma"})
    assert (plot.vmin, plot.vmax, plot.cmap) == (0.0, 23.0, "viridis")
    plot.load_state_dict({"max_width": 300})
    assert (plot.layout.width, plot.layout.max_width) == ("300px", "min(100%, 300px)")


def test_device_tensor_bin_centers_are_accepted():
    # A-16: x and y went through np.array, which refuses a CUDA tensor.
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    centers = torch.arange(4, dtype=torch.float32, device="cuda")
    plot = Plot2D(np.ones((4, 4)), x=centers, y=centers * 2)
    np.testing.assert_array_equal(plot.x, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(plot.y, [0.0, 2.0, 4.0, 6.0])
