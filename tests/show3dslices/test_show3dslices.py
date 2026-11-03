"""Show3DSlices: the browser gets the exact volume, pages share one cut, alignment finds the drift."""

import json

import numpy as np
import pytest
import traitlets
from ipywidgets.widgets.widget import _instances

from quantem.widget import Show3DSlices
from quantem.widget.adapters.core import make_dataset
from quantem.widget.show3dslices import estimate_global_slice_alignment


def column_volume(depth: int = 6, size: int = 48) -> np.ndarray:
    """Gaussian columns that sway with depth, like the tutorial volume."""
    row, col = np.mgrid[0:size, 0:size]
    sites = np.arange(6, size, 12)
    volume = np.zeros((depth, size, size), dtype=np.float32)
    for slice_index in range(depth):
        shift = 3.0 * np.sin(2 * np.pi * slice_index / depth)
        for r0 in sites:
            for c0 in sites:
                volume[slice_index] += np.exp(-((row - r0 - shift) ** 2 + (col - c0) ** 2) / (2 * 2.5**2))
    return volume


def drifting_stack(row_drift: float, col_drift: float, slices: int = 7, size: int = 64) -> np.ndarray:
    """A textured slice Fourier-shifted by ``idx * drift`` per slice (exact subpixel drift)."""
    row, col = np.mgrid[:size, :size].astype(np.float32)
    base = np.sin(row / 3.1) + 0.7 * np.cos(col / 4.7) + 0.35 * np.sin((row + col) / 5.3)
    base += 4.0 * np.exp(-((row - size * 0.35) ** 2 + (col - size * 0.42) ** 2) / 80.0)
    row_freq = np.fft.fftfreq(size)[:, None]
    col_freq = np.fft.fftfreq(size)[None, :]
    spectrum = np.fft.fft2(base)
    frames = [
        np.fft.ifft2(spectrum * np.exp(-2j * np.pi * (row_freq * idx * row_drift + col_freq * idx * col_drift))).real
        for idx in range(slices)
    ]
    return np.stack(frames).astype(np.float32)


# --- volume transport -----------------------------------------------------------


def test_browser_volume_matches_numpy_slices():
    volume = column_volume()
    widget = Show3DSlices(volume, sampling=(0.3, 0.18, 0.18), units="nm", title="columns")
    sent = np.frombuffer(widget.volume_bytes, dtype=np.float32).reshape(1, *volume.shape)
    np.testing.assert_array_equal(sent[0], volume)
    np.testing.assert_array_equal(sent[0, widget.slice_index], volume[volume.shape[0] // 2])
    np.testing.assert_array_equal(sent[0, :, widget.row_index, :], volume[:, volume.shape[1] // 2, :])
    assert widget.volume_shape == list(volume.shape)
    assert widget.voxel_size_A == pytest.approx([3.0, 1.8, 1.8])
    # the three-axis sampling is the one calibration the browser reads: no lateral-mean copy beside it
    assert "pixel_size" not in widget.get_state() and "pixel_size" not in widget.state_dict()
    assert widget.title == "columns"
    assert widget.smooth is False  # each voxel drawn as a sharp block; bilinear Smooth is opt-in


def test_offline_pack_is_uint8_against_the_global_range(capsys):
    volume = column_volume()
    widget = Show3DSlices(volume, offline=True)
    # D-R1: the uint8 pack was silent
    printed = capsys.readouterr().out
    assert "Show3DSlices packs the 6x48x48 volume as uint8" in printed and "keeps float32" in printed
    packed = np.frombuffer(widget.volume_bytes, dtype=np.uint8).reshape(volume.shape)
    restored = widget._offline_min + packed * (widget._offline_max - widget._offline_min) / 255.0
    assert widget._offline_min == pytest.approx(float(volume.min()))
    assert widget._offline_max == pytest.approx(float(volume.max()))
    np.testing.assert_allclose(restored, volume, atol=(volume.max() - volume.min()) / 255.0)


def test_dataset3d_supplies_title_and_sampling_in_angstrom():
    dataset = make_dataset(column_volume(), name="phase", sampling=[0.3, 0.18, 0.18], units=["nm", "nm", "nm"])
    widget = Show3DSlices(dataset)
    assert widget.title == "phase"
    assert widget.voxel_size_A == pytest.approx([3.0, 1.8, 1.8])


def test_a_frame_axis_from_read_image_stack_opens_without_a_depth_bar():
    # rc41: read_image_stack labels its first axis "frame", which Show3DSlices refused.
    dataset = make_dataset(column_volume(), name="stack", sampling=[1.0, 0.18, 0.18], units=["frame", "nm", "nm"])
    assert Show3DSlices(dataset).voxel_size_A == pytest.approx([0.0, 1.8, 1.8])
    with pytest.raises(ValueError, match="'furlong'"):
        Show3DSlices(column_volume(), sampling=1.0, units="furlong")


def test_sampling_keyword_fills_the_axis_a_dataset_does_not_calibrate():
    # the keyword was compared with the whole recorded vector, where the frame axis is 0, so a depth step raised "contradicts"
    stack = make_dataset(column_volume(), name="stack", sampling=[1.0, 0.186, 0.186], units=["frame", "nm", "nm"])
    assert Show3DSlices(stack, sampling=(3.0, 0.186, 0.186), units="nm").voxel_size_A == pytest.approx([30.0, 1.86, 1.86])
    assert Show3DSlices(stack, sampling=(3.0, 1.86, 1.86)).voxel_size_A == pytest.approx([3.0, 1.86, 1.86])  # no units: angstrom
    assert Show3DSlices(stack, sampling=(3.0, 0.0, 0.0)).voxel_size_A == pytest.approx([3.0, 1.86, 1.86])  # 0 states no step
    # an axis the dataset calibrates still has one answer
    with pytest.raises(ValueError, match="contradicts the value the input records"):
        Show3DSlices(stack, sampling=(3.0, 0.2, 0.186), units="nm")
    with pytest.raises(ValueError, match="contradicts the value the input records"):
        Show3DSlices(stack, sampling=(3.0, 0.186, 0.186))  # 0.186 angstrom is not the recorded 0.186 nm


def test_rejects_data_the_browser_cannot_show():
    with pytest.raises(TypeError, match="complex"):
        Show3DSlices(np.ones((2, 3, 3), dtype=np.complex64))
    with pytest.raises(ValueError, match="NaN"):
        Show3DSlices(np.full((2, 3, 3), np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="panel_titles"):
        Show3DSlices(np.zeros((2, 2, 3, 3), dtype=np.float32), panel_titles=["one"])


# --- slice alignment ------------------------------------------------------------


def test_alignment_recovers_known_subpixel_drift():
    stack = drifting_stack(row_drift=0.75, col_drift=-0.4)
    result = estimate_global_slice_alignment(stack)
    # the display correction is the opposite of the measured drift
    assert result["row_shift_px_per_slice"] == pytest.approx(-0.75, abs=0.1)
    assert result["col_shift_px_per_slice"] == pytest.approx(0.4, abs=0.1)
    assert result["fit_r2"]["row"] > 0.99 and result["fit_r2"]["col"] > 0.99
    assert len(result["adjacent_shift_apply_px"]) == stack.shape[0] - 1
    np.testing.assert_array_equal(stack, drifting_stack(row_drift=0.75, col_drift=-0.4))


def test_widget_alignment_applies_caches_and_resets():
    widget = Show3DSlices(drifting_stack(row_drift=0.75, col_drift=-0.4, slices=6))
    first = widget.estimate_slice_alignment()
    assert widget.slice_alignment == "auto"
    assert widget.slice_alignment_cached is True
    assert widget.row_shift_px_per_slice == pytest.approx(-0.75, abs=0.1)
    assert widget.col_shift_px_per_slice == pytest.approx(0.4, abs=0.1)
    assert widget.slice_alignment_status.startswith("Aligned row")
    widget.slice_alignment = "off"
    second = widget.estimate_slice_alignment()
    assert second == first
    assert widget.slice_alignment_status.startswith("Cached row")
    widget.reset_slice_alignment()
    assert (widget.slice_alignment, widget.row_shift_px_per_slice, widget.slice_alignment_cached) == ("off", 0.0, False)
    # the browser's Align button goes through the request trait
    widget._slice_alignment_request = json.dumps({"mode": "estimate", "panel": 0, "id": "test"})
    assert widget._slice_alignment_request == ""
    assert widget.slice_alignment == "auto"
    with pytest.raises(traitlets.TraitError, match="slice_alignment"):
        widget.slice_alignment = "physical_tilt"


# --- pages ------------------------------------------------------------------------


def test_pages_keep_the_cut_and_move_the_active_panel():
    volume = column_volume(depth=3, size=16)
    data = np.stack([volume, volume + 100])
    widget = Show3DSlices(data, page_labels=["raw", "corrected"], panel_titles=["object"])
    assert (widget.n_pages, widget.panels_per_page, widget.panel_count) == (2, 1, 2)
    assert widget.panel_titles == ["object", "object"]
    widget.slice_index, widget.row_index, widget.col_index = 1, 5, 7
    widget.page_idx = 1
    assert widget.active_panel == 1
    assert (widget.slice_index, widget.row_index, widget.col_index) == (1, 5, 7)
    paged = Show3DSlices(np.stack([data, data + 1]), page_labels=["single", "multi"], panel_titles=["object", "error"])
    assert paged.panel_titles == ["object", "error", "object", "error"]
    paged.active_panel = 1
    paged.page_idx = 1
    assert paged.active_panel == 3
    paged.active_panel = 0
    assert paged.page_idx == 0
    sent = np.frombuffer(paged.volume_bytes, dtype=np.float32).reshape(4, *volume.shape)
    np.testing.assert_array_equal(sent[3], volume + 101)


# --- state --------------------------------------------------------------------------


def test_state_round_trip_onto_a_second_widget():
    data = np.stack([column_volume(depth=3, size=16)] * 2)
    widget = Show3DSlices(data, page_labels=["a", "b"], cmap="viridis", vmin=0.1, vmax=0.9)
    widget.page_idx = 1
    widget.oblique_angle = 200.0
    restored = Show3DSlices(data, page_labels=["a", "b"])
    restored.load_state_dict(widget.state_dict())
    assert restored.page_idx == 1 and restored.active_panel == 1
    assert restored.cmap == "viridis"
    assert (restored.vmin, restored.vmax) == (0.1, 0.9)
    assert restored.oblique_angle == pytest.approx(20.0)
    assert restored.state_dict() == widget.state_dict()
    with pytest.raises(ValueError, match="not_a_trait"):
        restored.load_state_dict({"not_a_trait": 1})
    with pytest.raises(ValueError, match="Show3D"):
        restored.load_state_dict({"_widget": "Show3D"})


def test_the_x_y_z_names_of_earlier_releases_are_not_accepted():
    volume = column_volume(depth=3, size=16)
    with pytest.raises(TypeError, match="z_stretch"):
        Show3DSlices(volume, z_stretch=2.0)
    widget = Show3DSlices(volume, slice_stretch=2.0)
    assert widget.slice_stretch == 2.0
    # a state saved under the earlier names is not converted: its keys are named and nothing is applied
    with pytest.raises(ValueError, match="slice_z"):
        widget.load_state_dict({"slice_index": 0, "slice_z": 2, "pixel_size_axes": [1.0, 1.0, 1.0]})
    assert widget.slice_index == 1 and widget.voxel_size_A == [0.0, 0.0, 0.0]


# --- export -------------------------------------------------------------------------


def test_export_html_embeds_the_viewer_and_the_requested_packing(tmp_path):
    volume = column_volume(depth=3, size=16)
    widget = Show3DSlices(volume, title="depth comparison", sampling=(0.3, 0.18, 0.18), units="nm")
    widget.slice_index = 2
    exact = widget.export_html(tmp_path / "exact.html")
    html = exact.read_text(encoding="utf-8")
    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in html
    assert "application/vnd.jupyter.widget-state+json" in html
    assert "depth comparison" in html and '"slice_index": 2' in html and '"_export_light": true' in html
    assert '"offline": false' in html
    small = widget.export_html(tmp_path / "small.html", encoding="uint8")
    assert '"offline": true' in small.read_text(encoding="utf-8")
    assert small.stat().st_size < exact.stat().st_size
    assert widget.export_status.startswith("Exported small.html")
    assert widget._default_html_export_path(False).name == "depth_comparison_3x16x16_exact.html"
    # the toolbar download path answers with the bytes and the file name
    widget.export_request = json.dumps({"encoding": "uint8", "id": "req-1", "filename": "volume.html", "download": True})
    assert widget.export_filename == "volume.html" and widget.export_payload_id == "req-1" and widget.export_status.endswith(", uint8)")
    assert len(widget.export_payload) > 0
    with pytest.raises(ValueError, match="encoding"):
        widget.export_html(tmp_path / "bad.html", encoding="gzip")
    widget.free()
    widget.free()
    assert widget.volume_bytes == b""
    with pytest.raises(ValueError, match="free"):
        widget.export_html(tmp_path / "after_free.html")


def test_export_html_leaves_no_clone_model_open(tmp_path):
    # The page is written from an export-only clone; a clone left open keeps a comm in the
    # kernel and a model in the browser for every export.
    widget = Show3DSlices(column_volume(depth=3, size=16))
    models = set(_instances)
    widget.export_html(tmp_path / "volume.html")
    assert not [model for key, model in _instances.items() if key not in models and isinstance(model, Show3DSlices)]
    widget.close()
