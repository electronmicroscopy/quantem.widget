import json

import numpy as np
import pytest
import torch

from quantem.widget import ShowDiffraction
from quantem.widget.io import LoadResult


def test_showdiffraction_2d_single_frame():
    dp = np.random.rand(32, 48).astype(np.float32)
    w = ShowDiffraction(dp, verbose=False)
    assert w.n_frames == 1
    assert w.frame_idx == 0
    assert (w.det_rows, w.det_cols) == (32, 48)
    assert w.detector_shape == (32, 48)
    assert len(w.frame_bytes) == 32 * 48 * 4
    assert w.dp_scale_mode == "log"
    assert w.panel_width_px == 384


def test_showdiffraction_panel_width_hint():
    dp = np.random.rand(32, 48).astype(np.float32)
    w = ShowDiffraction(dp, panel_width_px=480, verbose=False)
    assert w.panel_width_px == 480


def test_showdiffraction_3d_stack():
    data = np.random.rand(5, 16, 16).astype(np.float32)
    w = ShowDiffraction(data, verbose=False)
    assert w.n_frames == 5
    assert (w.det_rows, w.det_cols) == (16, 16)


def test_showdiffraction_frame_idx_changes_frame():
    data = np.zeros((4, 8, 8), dtype=np.float32)
    for i in range(4):
        data[i] = float(i + 1)  # each frame a distinct constant
    w = ShowDiffraction(data, verbose=False)
    assert w.n_frames == 4
    f0 = np.frombuffer(w.frame_bytes, dtype=np.float32).copy()
    assert np.allclose(f0, 1.0)
    w.frame_idx = 2
    f2 = np.frombuffer(w.frame_bytes, dtype=np.float32)
    assert np.allclose(f2, 3.0)
    # out-of-range frame index is clamped into [0, n_frames)
    w.frame_idx = 99
    assert w.frame_idx == 3


def test_showdiffraction_offline_frames_baked():
    data = np.zeros((4, 8, 8), dtype=np.float32)
    for i in range(4):
        data[i] = float(i + 1)
    frame_len = 8 * 8
    # offline multi-frame: whole stack baked so the kernel-less HTML can scrub it
    w = ShowDiffraction(data, offline=True, verbose=False)
    assert len(w.offline_frames) == 4 * frame_len * 4
    baked = np.frombuffer(w.offline_frames, dtype=np.float32).reshape(4, 8, 8)
    assert np.allclose(baked[2], 3.0)
    # live widget stays empty (frames stream through frame_bytes per frame)
    live = ShowDiffraction(data, offline=False, verbose=False)
    assert live.offline_frames == b""
    # toggling offline bakes / clears
    live.offline = True
    assert len(live.offline_frames) == 4 * frame_len * 4
    live.offline = False
    assert live.offline_frames == b""
    # single pattern never bakes: nothing to scrub
    single = ShowDiffraction(np.ones((8, 8), dtype=np.float32), offline=True, verbose=False)
    assert single.offline_frames == b""


def test_showdiffraction_4d_raises():
    with pytest.raises(ValueError, match="Show4DSTEM"):
        ShowDiffraction(np.random.rand(4, 4, 16, 16).astype(np.float32), verbose=False)


def test_showdiffraction_wrong_ndim_raises():
    with pytest.raises(ValueError, match="Expected a 2D or 3D"):
        ShowDiffraction(np.zeros((4,), dtype=np.float32), verbose=False)
    with pytest.raises(ValueError, match="Expected a 2D or 3D"):
        ShowDiffraction(np.zeros((2, 2, 4, 4, 4), dtype=np.float32), verbose=False)


def test_showdiffraction_auto_detect_center():
    data = np.zeros((3, 7, 7), dtype=np.float32)
    for i in range(7):
        for j in range(7):
            if np.sqrt((i - 3) ** 2 + (j - 3) ** 2) <= 1.5:
                data[:, i, j] = 100.0
    w = ShowDiffraction(data, verbose=False)
    assert abs(w.center_row - 3.0) < 0.5
    assert abs(w.center_col - 3.0) < 0.5
    assert w.bf_radius > 0
    assert w.auto_detect_center() is w


def test_showdiffraction_manual_center():
    data = np.random.rand(16, 16).astype(np.float32)
    w = ShowDiffraction(data, center=(5.0, 6.0), bf_radius=3.0, verbose=False)
    assert w.center_row == 5.0
    assert w.center_col == 6.0
    assert w.bf_radius == 3.0
    w.set_center(7.0, 8.0)
    assert (w.center_row, w.center_col) == (7.0, 8.0)
    assert w.center_mode == "manual"


def test_showdiffraction_add_spot_calibrated():
    data = np.random.rand(32, 32).astype(np.float32)
    w = ShowDiffraction(
        data, k_pixel_size=0.1, spot_refine=False, center=(16, 16), bf_radius=5, verbose=False
    )
    w.add_spot(16, 26)
    spot = w.spots[0]
    assert spot["id"] == 1
    assert abs(spot["r_pixels"] - 10.0) < 0.01
    assert abs(spot["g_magnitude"] - 1.0) < 0.01
    assert abs(spot["d_spacing"] - 1.0) < 0.01


def test_showdiffraction_add_spot_uncalibrated():
    data = np.random.rand(32, 32).astype(np.float32)
    w = ShowDiffraction(data, center=(16, 16), bf_radius=5, verbose=False)
    w.add_spot(16, 26)
    assert w.spots[0]["d_spacing"] is None
    assert w.spots[0]["g_magnitude"] is None


def test_showdiffraction_spot_at_center():
    data = np.random.rand(16, 16).astype(np.float32)
    w = ShowDiffraction(
        data, k_pixel_size=0.1, spot_refine=False, center=(8, 8), bf_radius=3, verbose=False
    )
    w.add_spot(8, 8)
    assert w.spots[0]["r_pixels"] == pytest.approx(0.0)
    assert w.spots[0]["d_spacing"] is None


def test_showdiffraction_snap_to_peak():
    data = np.zeros((16, 16), dtype=np.float32)
    data[5, 8] = 100.0
    w = ShowDiffraction(
        data, snap_enabled=True, spot_refine=False, snap_radius=3,
        center=(8, 8), bf_radius=3, verbose=False,
    )
    w.add_spot(6, 7)
    assert w.spots[0]["row"] == 5.0
    assert w.spots[0]["col"] == 8.0
    assert w.spots[0]["raw_row"] == 6.0


def test_showdiffraction_undo_clear():
    data = np.random.rand(16, 16).astype(np.float32)
    w = ShowDiffraction(data, center=(8, 8), bf_radius=3, verbose=False)
    w.add_spot(5, 5).add_spot(10, 10)
    assert len(w.spots) == 2
    w.undo_spot()
    assert len(w.spots) == 1
    w.clear_spots()
    assert len(w.spots) == 0
    w.undo_spot()
    assert len(w.spots) == 0


def test_showdiffraction_remove_spot():
    data = np.random.rand(16, 16).astype(np.float32)
    w = ShowDiffraction(data, center=(8, 8), bf_radius=3, verbose=False)
    w.add_spot(5, 5).add_spot(10, 10)
    sid = w.spots[0]["id"]
    w.remove_spot(sid)
    assert len(w.spots) == 1
    assert all(s["id"] != sid for s in w.spots)


def test_showdiffraction_state_dict_roundtrip():
    data = np.random.rand(16, 16).astype(np.float32)
    w = ShowDiffraction(data, center=(5.0, 6.0), bf_radius=3.0, k_pixel_size=0.1, verbose=False)
    w.dp_scale_mode = "linear"
    w.dp_colormap = "viridis"
    w.snap_enabled = True
    w.add_spot(8, 8)
    sd = w.state_dict()
    assert sd["dp_scale_mode"] == "linear"
    assert sd["dp_colormap"] == "viridis"
    assert sd["center_row"] == 5.0
    assert sd["k_pixel_size"] == pytest.approx(0.1)
    assert sd["snap_enabled"] is True
    assert "frame_idx" in sd
    assert len(sd["spots"]) == 1
    w2 = ShowDiffraction(data, state=sd, verbose=False)
    assert w2.dp_scale_mode == "linear"
    assert w2.dp_colormap == "viridis"
    assert w2.bf_radius == 3.0
    assert w2.snap_enabled is True
    assert len(w2.spots) == 1


def test_showdiffraction_ui_mode_presets_and_overrides():
    data = np.random.rand(16, 16).astype(np.float32)

    presentation = ShowDiffraction(data, ui_mode="presentation", verbose=False)
    assert presentation.show_title is True
    assert presentation.show_controls is True
    assert presentation.controls_collapsed is True
    assert presentation.show_stats is False

    report = ShowDiffraction(data, ui_mode="report", verbose=False)
    assert report.show_title is True
    assert report.show_controls is False
    assert report.controls_collapsed is False
    assert report.show_stats is False

    minimal = ShowDiffraction(data, ui_mode="minimal", verbose=False)
    assert minimal.show_title is False
    assert minimal.show_controls is False
    assert minimal.controls_collapsed is False
    assert minimal.show_stats is False

    override = ShowDiffraction(
        data,
        ui_mode="minimal",
        show_title=True,
        show_controls=True,
        controls_collapsed=True,
        show_stats=True,
        verbose=False,
    )
    assert override.show_title is True
    assert override.show_controls is True
    assert override.controls_collapsed is True
    assert override.show_stats is True
    assert override.expand_controls() is override
    assert override.controls_collapsed is False
    assert override.collapse_controls() is override
    assert override.controls_collapsed is True
    assert override.toggle_controls() is override
    assert override.controls_collapsed is False


def test_showdiffraction_save_load_file(tmp_path):
    data = np.random.rand(16, 16).astype(np.float32)
    w = ShowDiffraction(data, verbose=False)
    w.dp_colormap = "viridis"
    path = tmp_path / "diff_state.json"
    w.save(str(path))
    saved = json.loads(path.read_text())
    assert saved["metadata_version"] == "1.0"
    assert saved["widget_name"] == "ShowDiffraction"
    assert "widget_version" in saved
    assert saved["state"]["dp_colormap"] == "viridis"
    w2 = ShowDiffraction(data, state=str(path), verbose=False)
    assert w2.dp_colormap == "viridis"


def test_showdiffraction_summary(capsys):
    data = np.random.rand(5, 16, 16).astype(np.float32)
    w = ShowDiffraction(data, pixel_size=2.39, k_pixel_size=0.1, verbose=False)
    w.add_spot(5, 5)
    w.summary()
    out = capsys.readouterr().out
    assert "Frames:" in out
    assert "Detector:" in out
    assert "Spots:" in out


def test_showdiffraction_set_image():
    data = np.random.rand(32, 32).astype(np.float32)
    w = ShowDiffraction(data, verbose=False)
    w.add_spot(10, 10)
    new_data = np.random.rand(8, 64, 64).astype(np.float32)
    w.set_image(new_data)
    assert w.n_frames == 8
    assert w.det_rows == 64
    assert len(w.spots) == 0


def test_showdiffraction_set_image_loadresult():
    data = np.random.rand(16, 16).astype(np.float32)
    w = ShowDiffraction(data, verbose=False)
    result = LoadResult(
        data=np.random.rand(8, 32, 32).astype(np.float32),
        metadata={"pixel_size": 3.0},
    )
    w.set_image(result)
    assert w.pixel_size == 3.0


def test_showdiffraction_accepts_torch():
    w = ShowDiffraction(torch.rand(4, 16, 16), verbose=False)
    assert w.n_frames == 4


def test_showdiffraction_accepts_loadresult():
    result = LoadResult(
        data=np.random.rand(4, 16, 16).astype(np.float32),
        metadata={"pixel_size": 2.0},
    )
    w = ShowDiffraction(result, verbose=False)
    assert w.pixel_size == 2.0


def test_showdiffraction_hot_pixel_removal():
    data = np.ones((4, 32, 32), dtype=np.uint16) * 100
    data[0, 3, 5] = 65535
    w = ShowDiffraction(data, verbose=False)
    assert w._get_frame(0)[3, 5] == 0


def test_showdiffraction_repr():
    w = ShowDiffraction(np.random.rand(4, 16, 16).astype(np.float32), k_pixel_size=0.1, verbose=False)
    r = repr(w)
    assert "ShowDiffraction" in r
    assert "sampling=" in r
    assert "frame=" in r


def test_showdiffraction_free():
    w = ShowDiffraction(np.random.rand(4, 16, 16).astype(np.float32), verbose=False)
    w.free()
    assert not hasattr(w, "_data")


def _disk_dp(size=64, center=(32, 30), radius=6):
    rows = np.arange(size)[:, None]
    cols = np.arange(size)[None, :]
    r2 = (rows - center[0]) ** 2 + (cols - center[1]) ** 2
    return np.exp(-r2 / (2 * radius**2)).astype(np.float32)


def test_showdiffraction_calibration_recomputes():
    w = ShowDiffraction(_disk_dp(), spot_refine=False, verbose=False)
    w.set_center(32, 32)
    w.add_spot(32, 42)
    assert w.spots[0]["d_spacing"] is None
    w.calibrate_from_ring(10.0, 2.0)  # r=10 px -> d=2.0 A -> k=0.05
    assert w.k_calibrated and abs(w.k_pixel_size - 0.05) < 1e-9
    assert abs(w.spots[0]["d_spacing"] - 2.0) < 1e-4
    with pytest.raises(ValueError):
        w.calibrate_from_ring(-1, 2.0)


def test_showdiffraction_ring_picking():
    w = ShowDiffraction(_disk_dp(), k_pixel_size=0.05, verbose=False)
    w.set_center(32, 32)
    w.add_ring(10.0)  # g = 10*0.05 -> d = 2.0 A
    assert abs(w.rings[0]["d_spacing"] - 2.0) < 1e-4
    w.add_ring(20.0)
    w.undo_ring()
    assert len(w.rings) == 1


def _two_spot_dp(size=64, center=(32, 32), spot=(28, 44), sigma=2.0):
    rows = np.arange(size)[:, None]
    cols = np.arange(size)[None, :]
    beam = np.exp(-((rows - center[0]) ** 2 + (cols - center[1]) ** 2) / (2 * 2.0**2))
    blob = np.exp(-((rows - spot[0]) ** 2 + (cols - spot[1]) ** 2) / (2 * sigma**2))
    return (50.0 * beam + 40.0 * blob).astype(np.float32)


def test_showdiffraction_gaussian_spot_refine():
    spot = (28, 44)
    w = ShowDiffraction(
        _two_spot_dp(spot=spot), k_pixel_size=0.05, center=(32, 32), bf_radius=3, verbose=False
    )
    w.add_spot(spot[0] + 1.4, spot[1] - 1.2)  # click ~2 px off the true spot
    s = w.spots[0]
    assert abs(s["row"] - spot[0]) < 0.5 and abs(s["col"] - spot[1]) < 0.5  # refined to the centroid
    assert s["raw_row"] == pytest.approx(spot[0] + 1.4)
    assert s["fit_quality"] > 0.9
    assert s["row_err"] is not None and s["d_spacing_err"] is not None and s["d_spacing_err"] >= 0


def test_showdiffraction_interplanar_angle():
    w = ShowDiffraction(_disk_dp(), spot_refine=False, center=(32, 32), bf_radius=3, verbose=False)
    w.set_center(32, 32)
    w.add_spot(32, 42)
    w.add_spot(42, 32)
    # Angles are measured relative to the first spot.
    assert w.spots[0]["angle_deg"] == pytest.approx(0.0, abs=1e-6)
    assert w.spots[1]["angle_deg"] == pytest.approx(90.0, abs=1e-6)


def test_showdiffraction_calibration_provenance():
    w = ShowDiffraction(_disk_dp(), spot_refine=False, center=(32, 32), bf_radius=3, verbose=False)
    w.set_center(32, 32)
    assert w.calibration_source == "none"
    w.calibrate_from_spot(32, 42, 2.0)  # r=10 px, d=2 A -> k=0.05
    assert w.calibration_source == "from_spot"
    assert w.calibration_ref_d == pytest.approx(2.0)
    assert w.calibration_ref_radius == pytest.approx(10.0)


def test_showdiffraction_export(tmp_path):
    w = ShowDiffraction(
        _disk_dp(), k_pixel_size=0.05, spot_refine=False, center=(32, 32), bf_radius=3, verbose=False
    )
    w.set_center(32, 32)
    w.add_spot(32, 42)
    w.add_ring(20.0)

    csv_text = w.export_measurements(tmp_path / "m.csv").read_text()
    assert "g_inv_angstrom" in csv_text
    assert csv_text.strip().count("\n") >= 2

    payload = json.loads(w.export_measurements(tmp_path / "m.json").read_text())
    assert payload["metadata"]["calibration_source"] == "manual"
    assert len(payload["measurements"]) == 2


def test_showdiffraction_measurements_from_state(tmp_path):
    # The saved state holds every spot and ring, so the measurement table is
    # rebuildable from it alone -- no separate export file needs to be kept.
    w = ShowDiffraction(
        _disk_dp(), k_pixel_size=0.05, spot_refine=False, center=(32, 32), bf_radius=3, verbose=False
    )
    w.set_center(32, 32)
    w.add_spot(32, 42)
    w.add_ring(20.0)

    state_path = tmp_path / "state.json"
    w.save(state_path)

    records = ShowDiffraction.measurements_from_state(state_path)
    assert [r["kind"] for r in records] == ["spot", "ring"]
    assert records == w._measurement_records()

    csv_path = ShowDiffraction.measurements_from_state(state_path, tmp_path / "from_state.csv")
    assert csv_path.read_text() == w.export_measurements(tmp_path / "live.csv").read_text()


def test_showdiffraction_center_mode_validator():
    w = ShowDiffraction(_disk_dp(), verbose=False)
    w.center_mode = "manual"
    assert w.center_mode == "manual"
    with pytest.raises(ValueError):
        w.center_mode = "midpoint"


def test_showdiffraction_detect_spots():
    M, cen, G = 128, (64, 64), 24.0
    rows = np.arange(M)[:, None]
    cols = np.arange(M)[None, :]
    def blob(r, c, a, s):
        return a * np.exp(-(((rows - r) ** 2 + (cols - c) ** 2) / (2 * s * s)))
    dp = blob(*cen, 300, 4)
    truth = [(cen[0], cen[1] + G), (cen[0], cen[1] - G), (cen[0] + G, cen[1]), (cen[0] - G, cen[1])]
    for r, c in truth:
        dp = dp + blob(r, c, 40, 2.0)
    dp = dp.astype(np.float32)
    w = ShowDiffraction(dp, center=cen, bf_radius=6, k_pixel_size=1 / (2.099 * G), verbose=False)
    w.detect_spots(max_spots=6)
    assert 4 <= len(w.spots) <= 6  # found the spots, beam excluded
    on_spot = sum(any(abs(s["row"] - r) < 2 and abs(s["col"] - c) < 2 for r, c in truth) for s in w.spots)
    assert on_spot >= 4


def test_showdiffraction_detect_rings():
    M, cen = 256, (128, 128)
    rows = np.arange(M)[:, None]
    cols = np.arange(M)[None, :]
    r = np.hypot(rows - cen[0], cols - cen[1])
    dp = 200.0 * np.exp(-(r**2) / (2 * 5.0**2))
    ring_radii = [40.0, 70.0, 100.0]
    for rr in ring_radii:
        dp = dp + 30.0 * np.exp(-((r - rr) ** 2) / (2 * 2.5**2))
    dp = dp.astype(np.float32)
    w = ShowDiffraction(dp, center=cen, bf_radius=15, k_pixel_size=0.02, verbose=False)
    w.detect_rings(max_rings=6)
    found = sorted(rng["radius_px"] for rng in w.rings)
    assert len(found) >= 3
    for target in ring_radii:
        assert any(abs(f - target) < 3 for f in found)
