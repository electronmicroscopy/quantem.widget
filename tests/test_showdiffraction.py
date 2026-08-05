import json
import math
import sys
from collections import namedtuple

import numpy as np
import pytest
import torch

from quantem.widget import ShowDiffraction
from quantem.widget import Phase
from quantem.widget.showdiffraction import build_measurement_records, measurement_metadata

LoadResult = namedtuple("LoadResult", ("data", "metadata"))


def _disk_dp(size=64, center=(32, 30), radius=6):
    rows = np.arange(size)[:, None]
    cols = np.arange(size)[None, :]
    r2 = (rows - center[0]) ** 2 + (cols - center[1]) ** 2
    return np.exp(-r2 / (2 * radius**2)).astype(np.float32)


def _two_spot_dp(size=64, center=(32, 32), spot=(28, 44), sigma=2.0):
    rows = np.arange(size)[:, None]
    cols = np.arange(size)[None, :]
    beam = np.exp(-((rows - center[0]) ** 2 + (cols - center[1]) ** 2) / (2 * 2.0**2))
    blob = np.exp(-((rows - spot[0]) ** 2 + (cols - spot[1]) ** 2) / (2 * sigma**2))
    return (50.0 * beam + 40.0 * blob).astype(np.float32)


def test_showdiffraction_construction_and_ingest():
    # 2D single frame
    dp = np.random.rand(32, 48).astype(np.float32)
    w = ShowDiffraction(dp, verbose=False)
    assert w.n_frames == 1
    assert w.frame_idx == 0
    assert (w.det_rows, w.det_cols) == (32, 48)
    assert w.detector_shape == (32, 48)
    assert len(w.frame_bytes) == 32 * 48 * 4
    assert w.dp_scale_mode == "log"
    assert w.panel_width_px == 384
    # 3D stack
    data = np.random.rand(5, 16, 16).astype(np.float32)
    w = ShowDiffraction(data, verbose=False)
    assert w.n_frames == 5
    assert (w.det_rows, w.det_cols) == (16, 16)
    # 4D and wrong ndim raise
    with pytest.raises(ValueError, match="4D input"):
        ShowDiffraction(np.random.rand(4, 4, 16, 16).astype(np.float32), verbose=False)
    with pytest.raises(ValueError, match="Expected a 2D or 3D"):
        ShowDiffraction(np.zeros((4,), dtype=np.float32), verbose=False)
    with pytest.raises(ValueError, match="Expected a 2D or 3D"):
        ShowDiffraction(np.zeros((2, 2, 4, 4, 4), dtype=np.float32), verbose=False)
    # torch tensor input
    w = ShowDiffraction(torch.rand(4, 16, 16), verbose=False)
    assert w.n_frames == 4
    # LoadResult input applies metadata
    result = LoadResult(
        data=np.random.rand(4, 16, 16).astype(np.float32),
        metadata={"pixel_size": 2.0},
    )
    w = ShowDiffraction(result, verbose=False)
    assert w.pixel_size == 2.0
    # set_image replaces the stack and clears spots
    w = ShowDiffraction(np.random.rand(32, 32).astype(np.float32), verbose=False)
    w.add_spot(10, 10)
    new_data = np.random.rand(8, 64, 64).astype(np.float32)
    w.set_image(new_data)
    assert w.n_frames == 8
    assert w.det_rows == 64
    assert len(w.spots) == 0
    # set_image with a LoadResult applies metadata
    result = LoadResult(
        data=np.random.rand(8, 32, 32).astype(np.float32),
        metadata={"pixel_size": 3.0},
    )
    w.set_image(result)
    assert w.pixel_size == 3.0


def test_showdiffraction_frames_offline_and_hot_pixels():
    data = np.zeros((4, 8, 8), dtype=np.float32)
    for i in range(4):
        data[i] = float(i + 1)  # distinct frames
    # frame_idx switches the streamed frame
    w = ShowDiffraction(data, verbose=False)
    assert w.n_frames == 4
    f0 = np.frombuffer(w.frame_bytes, dtype=np.float32).copy()
    assert np.allclose(f0, 1.0)
    w.frame_idx = 2
    f2 = np.frombuffer(w.frame_bytes, dtype=np.float32)
    assert np.allclose(f2, 3.0)
    # bounds clamp
    w.frame_idx = 99
    assert w.frame_idx == 3
    # offline stack bakes every frame
    frame_len = 8 * 8
    w = ShowDiffraction(data, offline=True, verbose=False)
    assert len(w.offline_frames) == 4 * frame_len * 4
    baked = np.frombuffer(w.offline_frames, dtype=np.float32).reshape(4, 8, 8)
    assert np.allclose(baked[2], 3.0)
    # live frame stream
    live = ShowDiffraction(data, offline=False, verbose=False)
    assert live.offline_frames == b""
    # offline toggle
    live.offline = True
    assert len(live.offline_frames) == 4 * frame_len * 4
    live.offline = False
    assert live.offline_frames == b""
    # single frame
    single = ShowDiffraction(np.ones((8, 8), dtype=np.float32), offline=True, verbose=False)
    assert single.offline_frames == b""
    # hot pixels zeroed on load
    hot = np.ones((4, 32, 32), dtype=np.uint16) * 100
    hot[0, 3, 5] = 65535
    w = ShowDiffraction(hot, verbose=False)
    assert w._get_frame(0)[3, 5] == 0


def test_ingest_sparse_counting_frame():
    # sparse electron counting: the hot-pixel clamp must not zero real counts
    rng = np.random.default_rng(0)
    frame = np.zeros((256, 256), dtype=np.uint16)
    pixels = rng.choice(frame.size, size=40, replace=False)
    frame.flat[pixels] = rng.integers(1, 4, size=40)
    w = ShowDiffraction(frame, verbose=False)
    assert float(w._get_frame(0).sum()) == float(frame.sum())


def test_showdiffraction_center_and_modes():
    # auto-detect finds the bright disk
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
    # manual center via kwarg and set_center
    data = np.random.rand(16, 16).astype(np.float32)
    w = ShowDiffraction(data, center=(5.0, 6.0), bf_radius=3.0, verbose=False)
    assert w.center_row == 5.0
    assert w.center_col == 6.0
    assert w.bf_radius == 3.0
    w.set_center(7.0, 8.0)
    assert (w.center_row, w.center_col) == (7.0, 8.0)
    assert w.center_mode == "manual"
    # center_mode accepts known modes only
    w = ShowDiffraction(_disk_dp(), verbose=False)
    w.center_mode = "manual"
    assert w.center_mode == "manual"
    with pytest.raises(ValueError):
        w.center_mode = "midpoint"


def test_showdiffraction_spot_picking_and_refine():
    # calibrated spot gets r/g/d
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
    # uncalibrated spot has no g/d
    w = ShowDiffraction(data, center=(16, 16), bf_radius=5, verbose=False)
    w.add_spot(16, 26)
    assert w.spots[0]["d_spacing"] is None
    assert w.spots[0]["g_magnitude"] is None
    # spot at the center
    data = np.random.rand(16, 16).astype(np.float32)
    w = ShowDiffraction(
        data, k_pixel_size=0.1, spot_refine=False, center=(8, 8), bf_radius=3, verbose=False
    )
    w.add_spot(8, 8)
    assert w.spots[0]["r_pixels"] == pytest.approx(0.0)
    assert w.spots[0]["d_spacing"] is None
    # snap to local peak
    data = np.zeros((16, 16), dtype=np.float32)
    data[5, 8] = 100.0
    w = ShowDiffraction(
        data,
        snap_enabled=True,
        spot_refine=False,
        snap_radius=3,
        center=(8, 8),
        bf_radius=3,
        verbose=False,
    )
    w.add_spot(6, 7)
    assert w.spots[0]["row"] == 5.0
    assert w.spots[0]["col"] == 8.0
    assert w.spots[0]["raw_row"] == 6.0
    # Gaussian refine pulls the click onto the true spot
    spot = (28, 44)
    w = ShowDiffraction(
        _two_spot_dp(spot=spot), k_pixel_size=0.05, center=(32, 32), bf_radius=3, verbose=False
    )
    w.add_spot(spot[0] + 1.4, spot[1] - 1.2)  # click ~2 px off the true spot
    s = w.spots[0]
    assert (
        abs(s["row"] - spot[0]) < 0.5 and abs(s["col"] - spot[1]) < 0.5
    )  # refined to the centroid
    assert s["raw_row"] == pytest.approx(spot[0] + 1.4)
    assert s["fit_quality"] > 0.9
    assert s["row_err"] is not None and s["d_spacing_err"] is not None and s["d_spacing_err"] >= 0
    # interplanar angles are measured relative to the first spot
    w = ShowDiffraction(_disk_dp(), spot_refine=False, center=(32, 32), bf_radius=3, verbose=False)
    w.set_center(32, 32)
    w.add_spot(32, 42)
    w.add_spot(42, 32)
    assert w.spots[0]["angle_deg"] == pytest.approx(0.0, abs=1e-6)
    assert w.spots[1]["angle_deg"] == pytest.approx(90.0, abs=1e-6)


def test_showdiffraction_spot_editing():
    # undo and clear
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
    # remove by id
    w.add_spot(5, 5).add_spot(10, 10)
    sid = w.spots[0]["id"]
    w.remove_spot(sid)
    assert len(w.spots) == 1
    assert all(s["id"] != sid for s in w.spots)


def test_showdiffraction_state_dict_and_file(tmp_path):
    # state_dict -> state kwarg
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
    # save to file and load via state=path
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


def test_state_envelope_rejects_other_widget(tmp_path):
    # a Show2D envelope must not load into ShowDiffraction
    data = np.random.rand(16, 16).astype(np.float32)
    w = ShowDiffraction(data, verbose=False)
    envelope = {"metadata_version": "1.0", "widget_name": "Show2D", "state": w.state_dict()}
    path = tmp_path / "show2d_state.json"
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="Show2D"):
        ShowDiffraction(data, state=str(path), verbose=False)
    with pytest.raises(ValueError, match="Show2D"):
        ShowDiffraction(data, state=envelope, verbose=False)
    with pytest.raises(ValueError, match="Show2D"):
        ShowDiffraction.measurements_from_state(str(path))
    with pytest.raises(ValueError, match="Show2D"):
        ShowDiffraction.measurements_from_state(envelope)


def test_showdiffraction_ui_modes_and_summary(capsys):
    data = np.random.rand(16, 16).astype(np.float32)
    # presets
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
    # explicit flags override the preset; toggle helpers chain
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
    # summary prints key stats
    data = np.random.rand(5, 16, 16).astype(np.float32)
    w = ShowDiffraction(data, pixel_size=2.39, k_pixel_size=0.1, verbose=False)
    w.add_spot(5, 5)
    w.summary()
    out = capsys.readouterr().out
    assert "Frames:" in out
    assert "Detector:" in out
    assert "Spots:" in out


def test_summary_uncalibrated_labels(capsys):
    # uncalibrated: no fake units on the detector or spot lines
    w = ShowDiffraction(_disk_dp(), spot_refine=False, verbose=False)
    w.set_center(32, 32)
    w.add_spot(32, 42)
    w.summary()
    out = capsys.readouterr().out
    assert "(uncalibrated)" in out
    assert "px/px" not in out
    assert "r=10.0 px" in out
    assert "d=" not in out
    # calibrated: units restored
    w.calibrate_from_ring(10.0, 2.0)
    w.summary()
    out = capsys.readouterr().out
    assert "1/Å/px" in out
    assert "d=2.000" in out


def test_showdiffraction_ring_calibration_and_picking():
    # calibration recomputes existing spots
    w = ShowDiffraction(_disk_dp(), spot_refine=False, verbose=False)
    w.set_center(32, 32)
    w.add_spot(32, 42)
    assert w.spots[0]["d_spacing"] is None
    w.calibrate_from_ring(10.0, 2.0)  # r=10 px -> d=2.0 A -> k=0.05
    assert w.k_calibrated and abs(w.k_pixel_size - 0.05) < 1e-9
    assert abs(w.spots[0]["d_spacing"] - 2.0) < 1e-4
    with pytest.raises(ValueError):
        w.calibrate_from_ring(-1, 2.0)
    # ring picking and undo
    w = ShowDiffraction(_disk_dp(), k_pixel_size=0.05, verbose=False)
    w.set_center(32, 32)
    w.add_ring(10.0)  # g = 10*0.05 -> d = 2.0 A
    assert abs(w.rings[0]["d_spacing"] - 2.0) < 1e-4
    w.add_ring(20.0)
    w.undo_ring()
    assert len(w.rings) == 1


def test_calibrate_request_channels_report_failure():
    # at-center spot and bad ring inputs surface a status instead of silence
    w = ShowDiffraction(_disk_dp(), spot_refine=False, verbose=False)
    w.set_center(32, 32)
    w._calibrate_from_spot_request = [32.0, 32.0, 2.0]
    assert w.analysis_status.startswith("Calibrate failed")
    assert w._calibrate_from_spot_request == []
    w.analysis_status = ""
    w._calibrate_from_ring_request = [10.0, -1.0]
    assert w.analysis_status.startswith("Calibrate failed")
    assert w._calibrate_from_ring_request == []


def test_showdiffraction_export_and_measurements(tmp_path):
    w = ShowDiffraction(
        _disk_dp(),
        k_pixel_size=0.05,
        spot_refine=False,
        center=(32, 32),
        bf_radius=3,
        verbose=False,
    )
    w.set_center(32, 32)
    w.add_spot(32, 42)
    w.add_ring(20.0)
    # CSV and JSON export
    csv_text = w.export_measurements(tmp_path / "m.csv").read_text()
    assert "g_inv_angstrom" in csv_text
    assert csv_text.strip().count("\n") >= 2
    payload = json.loads(w.export_measurements(tmp_path / "m.json").read_text())
    assert payload["metadata"]["calibration_source"] == "manual"
    assert len(payload["measurements"]) == 2
    # measurements straight from a saved state file
    state_path = tmp_path / "state.json"
    w.save(state_path)
    records = ShowDiffraction.measurements_from_state(state_path)
    assert [r["kind"] for r in records] == ["spot", "ring"]
    assert records == build_measurement_records(w.spots, w.rings)
    csv_path = ShowDiffraction.measurements_from_state(state_path, tmp_path / "from_state.csv")
    assert csv_path.read_text() == w.export_measurements(tmp_path / "live.csv").read_text()


def test_export_ring_hkl_candidates(tmp_path):
    # near-degenerate assignments stay auditable in the export
    card = Phase.from_dspacings("card", [(2.00, "a"), (1.97, "b")])
    w = ShowDiffraction(
        np.zeros((256, 256), np.float32), center=(128, 128), k_pixel_size=0.01, verbose=False
    )
    w.add_ring(1.0 / (1.985 * 0.01))
    w.index_rings(card)
    csv_text = w.export_measurements(tmp_path / "m.csv").read_text()
    assert "hkl_candidates" in csv_text
    assert "a|b" in csv_text
    payload = json.loads(w.export_measurements(tmp_path / "m.json").read_text())
    assert payload["measurements"][0]["hkl_candidates"] == "a|b"


def test_showdiffraction_detect_spots_and_rings():
    # spot detection
    M, cen, G = 128, (64, 64), 24.0
    rows = np.arange(M)[:, None]
    cols = np.arange(M)[None, :]

    def blob(r, c, a, s):
        return a * np.exp(-(((rows - r) ** 2 + (cols - c) ** 2) / (2 * s * s)))

    dp = blob(*cen, 300, 4)
    truth = [
        (cen[0], cen[1] + G),
        (cen[0], cen[1] - G),
        (cen[0] + G, cen[1]),
        (cen[0] - G, cen[1]),
    ]
    for r, c in truth:
        dp = dp + blob(r, c, 40, 2.0)
    dp = dp.astype(np.float32)
    w = ShowDiffraction(dp, center=cen, bf_radius=6, k_pixel_size=1 / (2.099 * G), verbose=False)
    w.detect_spots(max_spots=6)
    assert 4 <= len(w.spots) <= 6  # found the spots, beam excluded
    on_spot = sum(
        any(abs(s["row"] - r) < 2 and abs(s["col"] - c) < 2 for r, c in truth) for s in w.spots
    )
    assert on_spot >= 4
    # ring detection
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


def test_detect_rings_without_scipy_keeps_manual_rings(monkeypatch):
    # a scipy-less detect must not wipe manually added rings
    dp, cen = _ring_dp([60.0], background="power")
    w = ShowDiffraction(dp, verbose=False)
    w.set_center(*cen)
    w.add_ring(60.0)
    monkeypatch.setitem(sys.modules, "scipy.signal", None)
    w.detect_rings()
    assert any(abs(r["radius_px"] - 60.0) < 1.0 for r in w.rings)


def _dp(
    radii=(60.0,),
    *,
    center=None,
    background="exp",
    arc=None,
    ellipse=None,
    sigma=2.5,
    size=256,
    amp=30.0,
):
    cen = center or (size // 2, size // 2)
    rows = np.arange(size, dtype=np.float64)[:, None]
    cols = np.arange(size, dtype=np.float64)[None, :]
    d_row, d_col = rows - cen[0], cols - cen[1]
    r = np.hypot(d_row, d_col)
    dp = 500.0 * (r + 1.0) ** -1.5 if background == "power" else 1000.0 * np.exp(-r / 30.0)
    r_eff, mask = r, 1.0
    if ellipse is not None:
        ratio, phi = ellipse[0], math.radians(ellipse[1])
        major = d_col * math.cos(phi) + d_row * math.sin(phi)
        minor = -d_col * math.sin(phi) + d_row * math.cos(phi)
        r_eff = np.hypot(major / ratio, minor)
    if arc is not None:
        theta = np.degrees(np.arctan2(d_row, d_col)) % 360
        mask = (theta >= arc[0]) & (theta <= arc[1])
    for rr in radii:
        dp = dp + amp * np.exp(-((r_eff - rr) ** 2) / (2 * sigma**2)) * mask
    return dp.astype(np.float32), cen


def _ring_dp(radii, sigma=2.5, size=256, amp=30.0, background="exp"):
    return _dp(radii, background=background, sigma=sigma, size=size, amp=amp)


def _off_center_ring_dp(center, radii=(60.0,), sigma=2.5, size=256, amp=30.0):
    return _dp(radii, center=center, background="power", sigma=sigma, size=size, amp=amp)[0]


def _arc_ring_dp(r0=60.0, theta_lo=10.0, theta_hi=70.0, sigma=2.5, size=256, amp=30.0):
    return _dp(
        (r0,), background="power", arc=(theta_lo, theta_hi), sigma=sigma, size=size, amp=amp
    )


def _elliptical_ring_dp(r0=60.0, ratio=1.2, angle_deg=30.0, sigma=2.5, size=256, amp=30.0):
    return _dp(
        (r0,), background="power", ellipse=(ratio, angle_deg), sigma=sigma, size=size, amp=amp
    )


def test_radial_profile_units_wedges_and_background():
    # axes and units
    dp, cen = _ring_dp([60.0], background="power")
    w = ShowDiffraction(dp, center=cen, verbose=False)
    radii, intensity = w.radial_profile()
    assert radii.shape == intensity.shape
    assert np.all(np.diff(radii) > 0)
    assert abs(radii[40:][np.argmax(intensity[40:])] - 60.0) < 2.0
    assert len(w.radial_profile(n_bins=50)[0]) == 50
    with pytest.raises(ValueError, match="calibrat"):
        w.radial_profile(units="d")
    with pytest.raises(ValueError, match="units"):
        w.radial_profile(units="nm")
    w2 = ShowDiffraction(dp, center=cen, k_pixel_size=0.02, verbose=False)
    r_px, _ = w2.radial_profile(units="px")
    g, _ = w2.radial_profile(units="g")
    q_alias, _ = w2.radial_profile(units="q")
    assert np.allclose(g, r_px * 0.02, rtol=1e-6)
    assert np.allclose(q_alias, g)
    assert np.allclose(w2.radial_profile()[0], g)
    # azimuthal wedge: a ring painted in one sector only shows up when that sector is integrated
    size, cen, r0 = 256, 128, 60.0
    rows = np.arange(size, dtype=np.float64)[:, None]
    cols = np.arange(size, dtype=np.float64)[None, :]
    r = np.hypot(rows - cen, cols - cen)
    theta = np.degrees(np.arctan2(rows - cen, cols - cen)) % 360
    dp = 30.0 * np.exp(-((r - r0) ** 2) / (2 * 2.5**2)) * ((theta >= 10) & (theta <= 80))
    w = ShowDiffraction(dp.astype(np.float32), center=(cen, cen), verbose=False)
    radii, inside = w.radial_profile(angular_range=(10, 80))
    _, outside = w.radial_profile(angular_range=(180, 250))
    band = (radii > 55) & (radii < 65)
    assert inside[band].max() > 10.0
    assert outside[band].max() < 1.0
    # a wedge spanning the full circle matches the unrestricted profile
    dp, cen = _arc_ring_dp(r0=60.0, theta_lo=10.0, theta_hi=70.0)
    w = ShowDiffraction(dp, center=cen, verbose=False)
    _, unrestricted = w.radial_profile()
    for full_span in ((0, 360), (90, 450)):
        _, profile = w.radial_profile(angular_range=full_span)
        assert np.allclose(profile, unrestricted)
    _, wedge = w.radial_profile(angular_range=(0, 90))
    assert not np.allclose(wedge, unrestricted)
    # background: power-law fit flattens the falloff off-peak while preserving ring peaks
    dp, cen = _ring_dp([60.0, 90.0], background="power")
    w = ShowDiffraction(dp, center=cen, bf_radius=10, verbose=False)
    for rr in (60.0, 90.0):
        w.add_ring(rr)
    radii, intensity = w.radial_profile(units="px")
    _, background = w.radial_background()
    resid = intensity - background
    off_peak = (radii > 20) & (np.abs(radii - 60) > 10) & (np.abs(radii - 90) > 10) & (radii < 110)
    assert np.abs(resid[off_peak]).max() < 3.0
    at_peak = int(np.argmin(np.abs(radii - 60.0)))
    assert resid[at_peak] > 15.0
    _, net = w.radial_profile(units="px", subtract_background=True)
    assert np.allclose(net, intensity - background, atol=1e-5)
    with pytest.raises(ValueError, match="method"):
        w.radial_background(method="spline")


def test_ingest_non_finite_pixels():
    # a NaN or inf pixel must not poison ring detection
    dp, cen = _ring_dp([60.0, 90.0], background="power")
    clean = ShowDiffraction(dp, center=cen, bf_radius=10, verbose=False)
    clean.detect_rings(max_rings=2)
    dirty_dp = dp.copy()
    dirty_dp[10, 10] = np.nan
    dirty_dp[200, 200] = np.inf
    dirty = ShowDiffraction(dirty_dp, center=cen, bf_radius=10, verbose=False)
    dirty.detect_rings(max_rings=2)
    assert len(clean.rings) == 2
    assert [r["radius_px"] for r in dirty.rings] == [r["radius_px"] for r in clean.rings]


def test_refine_center_offsets_and_methods():
    # recovers known offsets (second case is far off-midpoint: wrap candidates)
    for true, start, radii in [
        ((121.3, 134.7), (118.0, 138.0), (50.0, 75.0)),
        ((60.2, 190.6), (63.0, 187.0), (40.0,)),
    ]:
        dp = _off_center_ring_dp(true, radii=radii)
        w = ShowDiffraction(dp, center=start, verbose=False)
        w.refine_center()
        assert abs(w.center_row - true[0]) < 0.3
        assert abs(w.center_col - true[1]) < 0.3
        assert w.center_mode == "auto"
    # auto_detect_center(refine=True) improves on the plain centroid estimate
    true = (124.4, 132.9)
    dp = _off_center_ring_dp(true, radii=(50.0, 75.0))
    plain = ShowDiffraction(dp, verbose=False).auto_detect_center()
    refined = ShowDiffraction(dp, verbose=False).auto_detect_center(refine=True)
    err_plain = math.hypot(plain.center_row - true[0], plain.center_col - true[1])
    err_refined = math.hypot(refined.center_row - true[0], refined.center_col - true[1])
    assert err_refined <= err_plain
    assert err_refined < 0.3
    assert refined.center_mode == "auto"
    # phase_corr and the auto cascade both recover the offset
    true = (121.3, 134.7)
    dp = _off_center_ring_dp(true, radii=(50.0, 75.0))
    w = ShowDiffraction(dp, center=(118.0, 138.0), verbose=False)
    w.refine_center(method="phase_corr")
    assert abs(w.center_row - true[0]) < 0.5 and abs(w.center_col - true[1]) < 0.5
    assert w.center_method == "phase_corr"
    w2 = ShowDiffraction(dp, center=(118.0, 138.0), verbose=False)
    w2.refine_center(method="auto")
    assert abs(w2.center_row - true[0]) < 0.5 and abs(w2.center_col - true[1]) < 0.5
    with pytest.raises(ValueError, match="method"):
        w.refine_center(method="centroid")


def test_fit_ring_profile():
    # a ring picked ~2 px off refines onto the profile peak with width, quality, d, size
    dp, cen = _ring_dp([60.0], background="power")
    w = ShowDiffraction(dp, center=cen, bf_radius=10, k_pixel_size=0.02, verbose=False)
    w.add_ring(62.0)
    d_before = w.rings[0]["d_spacing"]
    w.fit_ring_profile()
    ring = w.rings[0]
    assert abs(ring["radius_px"] - 60.0) < 0.3
    assert ring["raw_radius_px"] == 62.0
    assert abs(ring["fwhm_px"] - 2.3548 * 2.5) < 0.8
    assert ring["fit_quality"] > 0.9
    d_true = 1.0 / (60.0 * 0.02)
    assert abs(ring["d_spacing"] - d_true) < abs(d_before - d_true)
    # integrated intensity scales with ring amplitude
    sizes = {}
    for amp in (30.0, 60.0):
        dp, cen = _ring_dp([60.0], amp=amp, background="power")
        w = ShowDiffraction(dp, center=cen, bf_radius=10, verbose=False)
        w.add_ring(60.0)
        w.fit_ring_profile()
        sizes[amp] = w.rings[0]["intensity_integrated"]
    assert sizes[30.0] > 0
    assert abs(sizes[60.0] / sizes[30.0] - 2.0) < 0.2


def test_fit_ring_peaks_close_pair():
    # neighboring rings 7.1 px apart: the fit window shrinks below the neighbor
    from quantem.widget.showdiffraction import fit_ring_peaks

    radii = np.arange(0.25, 120.0, 0.5)
    intensity = 300.0 * np.exp(-0.5 * ((radii - 60.0) / 1.5) ** 2)
    intensity += 450.0 * np.exp(-0.5 * ((radii - 67.1) / 1.5) ** 2)
    rings = [{"radius_px": 60.0}, {"radius_px": 67.1}]
    for update, amp in zip(fit_ring_peaks(radii, intensity, rings), (300.0, 450.0)):
        assert abs(update["intensity"] - amp) / amp < 0.08
        assert update["fit_quality"] > 0.97


def test_texture_arc_vs_uniform():
    dp, cen = _arc_ring_dp(theta_lo=10.0, theta_hi=70.0)
    w = ShowDiffraction(dp, center=cen, verbose=False)
    w.add_ring(60.0)
    theta, intensity = w.azimuthal_profile(n_theta=90)
    assert theta.shape == intensity.shape == (90,)
    assert theta[0] >= 0 and theta[-1] < 360
    report = w.texture()
    assert report["strength"] > 0.5
    delta = abs(report["angle_deg"] - 40.0) % 180
    assert min(delta, 180 - delta) < 5.0
    dp2, cen2 = _ring_dp([60.0], background="power")
    w2 = ShowDiffraction(dp2, center=cen2, verbose=False)
    w2.add_ring(60.0)
    assert w2.texture()["strength"] < 0.05
    # fit/texture/ellipse analysis all require a picked ring
    dp3, cen3 = _ring_dp([60.0])
    w3 = ShowDiffraction(dp3, center=cen3, verbose=False)
    for method in (w3.fit_ring_profile, w3.texture, w3.fit_ellipse):
        with pytest.raises(ValueError):
            method()


def test_texture_masked_wedge():
    # isotropic ring with a masked 90 deg wedge reads untextured
    from quantem.widget.showdiffraction import texture_from_profile

    theta = (np.arange(72) + 0.5) * 5.0
    rng = np.random.default_rng(0)
    intensity = 100.0 + rng.normal(0.0, 1.0, 72)
    intensity[(theta > 45.0) & (theta < 135.0)] = 0.0
    report = texture_from_profile(theta.astype(np.float32), intensity.astype(np.float32))
    assert report["strength"] < 0.05


def test_texture_low_coverage_zeroed_and_bounded():
    # a few clustered live sectors cannot constrain the harmonic fit
    from quantem.widget.showdiffraction import texture_from_profile

    theta = ((np.arange(72) + 0.5) * 5.0).astype(np.float32)
    rng = np.random.default_rng(3)
    for n_live in (3, 4, 6):
        intensity = np.zeros(72, np.float32)
        intensity[:n_live] = 100.0 + rng.normal(0.0, 2.0, n_live)
        report = texture_from_profile(theta, intensity)
        assert report["strength"] == 0.0
        assert report["angle_deg"] == 0.0
    # strength is a fractional modulation, never above 1
    intensity = np.zeros(72, np.float32)
    intensity[:24] = 1.0
    intensity[0] = 1000.0
    report = texture_from_profile(theta, intensity)
    assert 0.0 <= report["strength"] <= 1.0


def test_fit_ellipse_center_error_partial_coverage():
    # a 1 px center error on a circle must not alias into ellipticity
    # when a 90 deg wedge is missing
    from quantem.widget.showdiffraction import fit_ellipse_from_sectors

    theta_centers = (np.arange(36) + 0.5) * 10.0
    radii = 60.0 - np.cos(np.radians(theta_centers))
    counts = np.full(36, 50.0)
    weight_sum = np.ones(36)
    wedge = theta_centers > 270.0
    counts[wedge] = 0.0
    weight_sum[wedge] = 0.0
    report = fit_ellipse_from_sectors(theta_centers, counts, weight_sum, weight_sum * radii)
    assert abs(report["ratio"] - 1.0) < 0.005


def test_fit_ellipse_short_arc_rejected():
    # a short live arc leaves the order-1/order-2 harmonics collinear
    from quantem.widget.showdiffraction import fit_ellipse_from_sectors

    theta_centers = (np.arange(180) + 0.5) * 2.0
    radii = np.full(180, 200.0)
    for n_live in (8, 20, 30):  # 16 / 40 / 60 degree arcs
        counts = np.zeros(180)
        weight_sum = np.zeros(180)
        counts[:n_live] = 50.0
        weight_sum[:n_live] = 1.0
        with pytest.raises(ValueError, match="span"):
            fit_ellipse_from_sectors(theta_centers, counts, weight_sum, weight_sum * radii)
    # >= 120 degrees of coverage still fits
    counts = np.zeros(180)
    weight_sum = np.zeros(180)
    counts[:65] = 50.0
    weight_sum[:65] = 1.0
    report = fit_ellipse_from_sectors(theta_centers, counts, weight_sum, weight_sum * radii)
    assert abs(report["ratio"] - 1.0) < 0.01


def test_fit_ellipse_and_correction():
    # detects elongation; a circular ring fits ratio 1
    dp, cen = _elliptical_ring_dp(r0=60.0, ratio=1.2, angle_deg=30.0)
    w = ShowDiffraction(dp, center=cen, bf_radius=10, verbose=False)
    w.add_ring(60.0)
    report = w.fit_ellipse()
    assert abs(report["ratio"] - 1.2) < 0.03
    assert abs(report["angle_deg"] - 30.0) < 5.0
    assert w.ellipse_ratio == report["ratio"]
    dp2, cen2 = _ring_dp([60.0], background="power")
    w2 = ShowDiffraction(dp2, center=cen2, bf_radius=10, verbose=False)
    w2.add_ring(60.0)
    assert abs(w2.fit_ellipse()["ratio"] - 1.0) < 0.01
    # noise plus a flat pedestal must not dilute the fitted ellipse amplitude
    rng = np.random.default_rng(0)
    dp, cen = _elliptical_ring_dp(r0=60.0, ratio=1.10, angle_deg=30.0, sigma=6.0, amp=30.0)
    noisy = (dp + 100.0 + rng.normal(0.0, 10.0, dp.shape)).astype(np.float32)
    w = ShowDiffraction(noisy, center=cen, bf_radius=10, verbose=False)
    w.add_ring(60.0)
    report = w.fit_ellipse()
    assert abs(report["ratio"] - 1.10) < 0.02
    assert abs(report["angle_deg"] - 30.0) < 5.0
    clean, cen2 = _elliptical_ring_dp(r0=60.0, ratio=1.0, angle_deg=0.0, sigma=6.0)
    w2 = ShowDiffraction(clean, center=cen2, bf_radius=10, verbose=False)
    w2.add_ring(60.0)
    assert abs(w2.fit_ellipse()["ratio"] - 1.0) < 0.01
    # spots on the major and minor axes disagree in d until the correction is applied
    dp, cen = _elliptical_ring_dp(r0=60.0, ratio=1.2, angle_deg=0.0)
    w = ShowDiffraction(
        dp, center=cen, bf_radius=10, k_pixel_size=0.01, spot_refine=False, verbose=False
    )
    w.add_spot(cen[0], cen[1] + 60.0 * 1.2)  # major axis (+col)
    w.add_spot(cen[0] + 60.0, cen[1])  # minor axis (+row)
    d_major, d_minor = (s["d_spacing"] for s in w.spots)
    assert abs(d_major - d_minor) / d_minor > 0.1
    w.add_ring(60.0)
    w.fit_ellipse()
    w.apply_ellipse_correction()
    d_major, d_minor = (s["d_spacing"] for s in w.spots)
    assert abs(d_major - d_minor) / d_minor < 0.02
    # mean-preserving: both axes map to the mean radius sqrt(A*B) = 60*sqrt(1.2),
    # so the correction does not silently rescale an existing calibration
    d_mean = 1.0 / (60.0 * math.sqrt(1.2) * 0.01)
    assert abs(d_major - d_mean) / d_mean < 0.03
    assert abs(d_minor - d_mean) / d_mean < 0.03


def _fe3o4_dp():
    # real SAED pattern, pending the tutorial dataset upload
    from quantem.widget.data.tutorials import _FE3O4_PACKAGED_PATH

    if not _FE3O4_PACKAGED_PATH.is_file():
        pytest.skip("real Fe3O4 SAED pattern unavailable")
    return np.load(_FE3O4_PACKAGED_PATH)


def test_fit_ellipse_default_width_real_data():
    # the default annulus stays on the requested ring instead of spanning neighbors
    w = ShowDiffraction(_fe3o4_dp(), verbose=False)
    w.run_auto(max_rings=5, exclude_radius=70)
    for ring in w.rings:
        report = w.fit_ellipse(ring_id=ring["id"])
        assert abs(report["r_mean"] - ring["radius_px"]) < 5.0


def test_ellipse_corrected_spot_angles():
    # inter-spot angles use the same ellipse transform as radii
    ratio, cen = 1.06, (128.0, 128.0)
    w = ShowDiffraction(
        np.zeros((256, 256), np.float32),
        center=cen,
        k_pixel_size=0.01,
        spot_refine=False,
        verbose=False,
    )
    root = math.sqrt(ratio)
    for theta in (52.5, 112.5):  # 60 degrees apart on the circularized pattern
        a = math.radians(theta)
        w.add_spot(cen[0] + 80.0 * math.sin(a) / root, cen[1] + 80.0 * math.cos(a) * root)
    w.ellipse_ratio = ratio
    w.apply_ellipse_correction()
    assert abs(w._measured_angle(w.spots[0], w.spots[1]) - 60.0) < 0.3
    assert abs(w.spots[1]["angle_deg"] - 60.0) < 0.3


def _au():
    return Phase.from_cubic("Au", 4.078, absences="fcc")


_AU_HKLS = {"111": (1, 1, 1), "200": (2, 0, 0), "220": (2, 2, 0), "311": (3, 1, 1)}


def _au_ring_widget(k=0.01, size=256):
    # place rings at the exact Au d-spacings for a k-space sampling of k 1/Å per px
    au, cen = _au(), (size // 2, size // 2)
    w = ShowDiffraction(
        np.zeros((size, size), np.float32), center=cen, k_pixel_size=k, verbose=False
    )
    for hkl in _AU_HKLS.values():
        w.add_ring(1.0 / (au.d_spacing(hkl) * k))
    return w


def test_calibration_from_spot_and_phase():
    # calibrate_from_spot records source and reference values
    w = ShowDiffraction(_disk_dp(), spot_refine=False, center=(32, 32), bf_radius=3, verbose=False)
    w.set_center(32, 32)
    assert w.calibration_source == "none"
    w.calibrate_from_spot(32, 42, 2.0)  # r=10 px, d=2 A -> k=0.05
    assert w.calibration_source == "from_spot"
    assert w.calibration_ref_d == pytest.approx(2.0)
    assert w.calibration_ref_radius == pytest.approx(10.0)
    # multi-ring regression recovers the camera constant from unindexed Au rings
    k_true, au = 0.01, _au()
    w = ShowDiffraction(np.zeros((256, 256), np.float32), center=(128, 128), verbose=False)
    for hkl in _AU_HKLS.values():
        w.add_ring(1.0 / (au.d_spacing(hkl) * k_true))
    w.calibrate_from_phase(au)
    assert abs(w.k_pixel_size - k_true) / k_true < 0.01
    assert [rng["hkl"] for rng in w.rings] == list(_AU_HKLS)
    assert w.calibration_rms_px < 0.5
    assert all(rng["radius_resid_px"] is not None for rng in w.rings)
    assert w.calibration_source == "from_phase"
    assert w.k_calibrated is True
    w2 = ShowDiffraction(np.zeros((256, 256), np.float32), center=(128, 128), verbose=False)
    w2.add_ring(49.0)
    with pytest.raises(ValueError, match="2"):
        w2.calibrate_from_phase(au)
    # regression over all rings averages out per-ring picking error
    rng, k_true, au = np.random.default_rng(7), 0.01, _au()
    w = ShowDiffraction(np.zeros((256, 256), np.float32), center=(128, 128), verbose=False)
    radii = [1.0 / (au.d_spacing(h) * k_true) for h in _AU_HKLS.values()]
    jitter = rng.uniform(-0.6, 0.6, len(radii))
    for r, j in zip(radii, jitter):
        w.add_ring(r + j)
    w.calibrate_from_phase(au)
    k_multi = w.k_pixel_size
    k_single = 1.0 / (au.d_spacing((1, 1, 1)) * (radii[0] + jitter[0]))
    assert abs(k_multi - k_true) <= abs(k_single - k_true)


def test_calibrate_from_phase_prefers_exact_assignment():
    # rings at exactly Au 220/311 must not lose to a lower-order near-miss
    k_true, au = 0.01, _au()
    w = ShowDiffraction(np.zeros((256, 256), np.float32), center=(128, 128), verbose=False)
    for hkl in ((2, 2, 0), (3, 1, 1)):
        w.add_ring(1.0 / (au.d_spacing(hkl) * k_true))
    w.calibrate_from_phase(au)
    assert [rng["hkl"] for rng in w.rings] == ["220", "311"]
    assert abs(w.k_pixel_size - k_true) / k_true < 1e-6


def test_calibrate_from_phase_tolerates_ring_jitter():
    # sub-pixel ring jitter must not flip the assignment to a high-order match
    from quantem.widget.showdiffraction import library_phase

    spinel = library_phase("MgAl2O4")
    k_true = 0.0025
    w = ShowDiffraction(np.zeros((512, 512), np.float32), center=(256, 256), verbose=False)
    for radius in (139.97, 164.38):  # 220/311 at k=0.0025 with ~0.2 px jitter
        w.add_ring(radius)
    w.calibrate_from_phase(spinel)
    assert [rng["hkl"] for rng in w.rings] == ["220", "311"]
    assert abs(w.k_pixel_size - k_true) / k_true < 0.01


def test_non_positive_ring_rejected():
    # add_ring rejects non-positive radii
    au = _au()
    w = ShowDiffraction(np.zeros((256, 256), np.float32), center=(128, 128), verbose=False)
    with pytest.raises(ValueError):
        w.add_ring(-30.0)
    with pytest.raises(ValueError):
        w.add_ring(0.0)
    assert w.rings == []
    # a corrupt negative radius never anchors a calibration
    w.add_ring(1.0 / (au.d_spacing((1, 1, 1)) * 0.01))
    w.rings = list(w.rings) + [{**w.rings[0], "id": 2, "radius_px": -30.0}]
    with pytest.raises(ValueError):
        w.calibrate_from_phase(au)
    assert w.k_calibrated is False


def test_index_rings_and_identify_phase():
    # index_rings labels every ring against the phase
    w = _au_ring_widget()
    w.index_rings(_au())
    assert [rng["hkl"] for rng in w.rings] == list(_AU_HKLS)
    assert all(rng["d_ref"] is not None and rng["d_error"] < 0.01 for rng in w.rings)
    assert "Au" in w.phase_match
    w2 = ShowDiffraction(np.zeros((64, 64), np.float32), verbose=False)
    w2.add_ring(10.0)
    with pytest.raises(ValueError):
        w2.index_rings(_au())
    # identify_phase ranks candidates from ring d-spacings
    w = ShowDiffraction(
        np.zeros((256, 256), np.float32), center=(128, 128), k_pixel_size=0.01, verbose=False
    )
    for d in (2.53, 1.71, 1.44):
        w.add_ring(1.0 / (d * 0.01))
    magnetite = Phase.from_dspacings(
        "Fe3O4", [(2.97, "220"), (2.53, "311"), (2.10, "400"), (1.71, "422"), (1.44, "440")]
    )
    other = Phase.from_dspacings("Other", [(2.80, "x"), (1.90, "y"), (1.20, "z")])
    ranked = w.identify_phase([magnetite, other])
    assert ranked[0]["name"] == "Fe3O4"
    assert "Fe3O4" in w.phase_match
    assert "few measured lines" in w.phase_match
    # Search only
    assert all(rng["hkl"] == "" for rng in w.rings)
    w2 = ShowDiffraction(np.zeros((64, 64), np.float32), verbose=False)
    w2.add_ring(10.0)
    with pytest.raises(ValueError):
        w2.identify_phase([magnetite])
    # with no rings, identify_phase ranks against spot d-spacings; spots stay unlabeled
    w = ShowDiffraction(
        np.zeros((256, 256), np.float32),
        center=(128, 128),
        k_pixel_size=0.01,
        spot_refine=False,
        verbose=False,
    )
    for d, ang in ((2.53, 0), (1.71, 50), (1.44, 100)):
        r, a = 1.0 / (d * 0.01), math.radians(ang)
        w.add_spot(128 + r * math.sin(a), 128 + r * math.cos(a))
    ranked = w.identify_phase([magnetite, other])
    assert ranked[0]["name"] == "Fe3O4"
    assert "Fe3O4" in w.phase_match
    assert all(s["hkl"] == "" for s in w.spots)


def test_identify_lines_unknown_intensity():
    # lattice-based phases have no intensities; lines carry None, not 0
    w = _au_ring_widget()
    reports = w.identify_phase([_au()])
    assert all(row["i_rel"] is None for row in reports[0]["lines"])


def test_identify_summary_tie():
    # a runner-up matching the winner's count is annotated
    w = ShowDiffraction(np.zeros((64, 64), np.float32), verbose=False)
    reports = [
        {"name": "Fe3O4", "matched": 6, "n_obs": 6},
        {"name": "ZnFe2O4", "matched": 6, "n_obs": 6},
        {"name": "Au", "matched": 3, "n_obs": 6},
    ]
    assert w._identify_summary(reports) == "Fe3O4: 6/6 lines; next: ZnFe2O4 (also 6/6), Au"


def _au_spot_widget(specs, k=0.01, size=256):
    # specs: [(hkl, angle_deg)] -> spot at radius 1/(d(hkl)*k) px, angle from +col axis
    au, cen = _au(), (size // 2, size // 2)
    w = ShowDiffraction(
        np.zeros((size, size), np.float32),
        center=cen,
        k_pixel_size=k,
        spot_refine=False,
        verbose=False,
    )
    for hkl, ang in specs:
        r = 1.0 / (au.d_spacing(hkl) * k)
        a = math.radians(ang)
        w.add_spot(cen[0] + r * math.sin(a), cen[1] + r * math.cos(a))
    return w


def test_index_spots():
    # a (200)/(220) anchor pair at the measured 45 degrees solves zone axis [001]
    w = _au_spot_widget([((2, 0, 0), 0), ((0, 2, 0), 90), ((2, 2, 0), 45)])
    w.index_spots(_au())
    assert w.zone_axis == "[001]"
    assert [s["hkl"] for s in w.spots] == ["200", "200", "220"]
    assert all(s["d_ref"] is not None and s["d_error"] < 1e-3 for s in w.spots)
    assert "Au" in w.phase_match
    # no anchor pair or collinear spots: labels fall back to best d, no zone axis
    w = _au_spot_widget([((2, 0, 0), 0), ((0, 2, 0), 90)])  # same family
    w.index_spots(_au())
    assert w.zone_axis == "" and w.phase_match == ""
    assert [s["hkl"] for s in w.spots] == ["200", "200"]
    assert all(s["d_ref"] is not None for s in w.spots)
    w2 = _au_spot_widget([((2, 0, 0), 0), ((4, 0, 0), 0)])  # collinear
    w2.index_spots(_au())
    assert w2.zone_axis == ""
    assert [s["hkl"] for s in w2.spots] == ["200", "400"]
    # error paths: uncalibrated widget and a d-spacing-only card
    w = ShowDiffraction(np.zeros((64, 64), np.float32), spot_refine=False, verbose=False)
    w.add_spot(20.0, 40.0)
    with pytest.raises(ValueError):
        w.index_spots(_au())
    w2 = _au_spot_widget([((2, 0, 0), 0)])
    with pytest.raises(ValueError, match="lattice-based"):
        w2.index_spots(Phase.from_dspacings("card", [(2.0, "a"), (1.5, "b")]))


def _parse_zone(label):
    values, sign = [], 1
    for ch in label.strip("[]"):
        if ch == "-":
            sign = -1
        else:
            values.append(sign * int(ch))
            sign = 1
    return tuple(values)


def test_index_spots_zone_variants():
    # a [011]-type pattern: the canonical anchor angle matches, the variant differs
    from quantem.widget.showdiffraction import _parse_hkl_label

    w = _au_spot_widget(
        [
            ((2, 0, 0), 0.0),
            ((1, 1, -1), 54.7356),
            ((0, 2, -2), 90.0),
            ((1, 1, 1), 20.0),  # no 111 variant sits at 20 degrees from (200)
        ]
    )
    w.index_spots(_au())
    assert w.zone_axis != ""
    axis = _parse_zone(w.zone_axis)
    labels = [s["hkl"] for s in w.spots]
    assert labels[0] and labels[1] and labels[2]
    assert labels[3] == ""
    for label in labels[:3]:
        h, k, ell = _parse_hkl_label(label)
        assert h * axis[0] + k * axis[1] + ell * axis[2] == 0


def test_index_spots_hexagonal_family_closure():
    # hexagonal {110} images at 90 degrees are not signed permutations of (110)
    from quantem.widget.showdiffraction import library_phase

    zno, k, cen = library_phase("ZnO"), 0.01, (128, 128)
    w = ShowDiffraction(
        np.zeros((256, 256), np.float32),
        center=cen,
        k_pixel_size=k,
        spot_refine=False,
        verbose=False,
    )
    d100, d110 = zno.d_spacing((1, 0, 0)), zno.d_spacing((1, 1, 0))
    for d, start in ((d100, 0), (d110, 30)):
        for ang in range(start, start + 360, 60):
            r, a = 1.0 / (d * k), math.radians(ang)
            w.add_spot(cen[0] + r * math.sin(a), cen[1] + r * math.cos(a))
    w.index_spots(zno)
    assert w.zone_axis == "[001]"
    assert all(s["hkl"] for s in w.spots)


def test_analysis_request_channels():
    # the Refine button channel runs refine_center and resets itself
    true = (124.0, 130.5)
    dp = _off_center_ring_dp(true, radii=(50.0, 75.0))
    w = ShowDiffraction(dp, center=(121.0, 133.0), verbose=False)
    w._refine_center_request = True
    assert abs(w.center_row - true[0]) < 1.0 and abs(w.center_col - true[1]) < 1.0
    assert w._refine_center_request is False
    assert w.analysis_status.startswith("Center")
    # the Refine channel runs the method picked in the dropdown and reports the grade
    dp = _off_center_ring_dp((124.0, 130.5), radii=(50.0, 75.0))
    w = ShowDiffraction(dp, center=(121.0, 133.0), verbose=False)
    w.refine_method = "phase_corr"
    w._refine_center_request = True
    assert "phase_corr" in w.analysis_status
    assert w._refine_center_request is False
    # fit-rings and fit-ellipse channels
    dp, cen = _ring_dp([60.0], background="power")
    w = ShowDiffraction(dp, center=cen, bf_radius=10, verbose=False)
    w.add_ring(60.0)
    w._fit_rings_request = True
    assert all("fwhm_px" in r for r in w.rings)
    assert w._fit_rings_request is False
    assert w.analysis_status.startswith("Fitted")
    assert "texture" in w.analysis_status
    dp2, cen2 = _elliptical_ring_dp(r0=60.0, ratio=1.2, angle_deg=30.0)
    w2 = ShowDiffraction(dp2, center=cen2, bf_radius=10, verbose=False)
    w2.add_ring(60.0)
    w2._fit_ellipse_request = True
    assert w2.ellipse_ratio == pytest.approx(1.2, rel=0.05)
    assert "Ellipse" in w2.analysis_status
    assert w2._fit_ellipse_request is False
    # either channel with no rings picked sets a failed status
    for trait in ("_fit_rings_request", "_fit_ellipse_request"):
        dp, cen = _ring_dp([60.0])
        w = ShowDiffraction(dp, center=cen, verbose=False)
        setattr(w, trait, True)
        assert "failed" in w.analysis_status
        assert getattr(w, trait) is False


def _blob_grid_dp(size=128, spacing=22):
    cen = (size // 2, size // 2)
    rows = np.arange(size)[:, None]
    cols = np.arange(size)[None, :]
    dp = 300.0 * np.exp(-(((rows - cen[0]) ** 2 + (cols - cen[1]) ** 2) / (2 * 4.0**2)))
    coords = [20 + spacing * i for i in range(5)]
    truth = [(r, c) for r in coords for c in coords if (r, c) != cen]
    for r, c in truth:
        dp = dp + 40.0 * np.exp(-(((rows - r) ** 2 + (cols - c) ** 2) / (2 * 2.0**2)))
    return dp.astype(np.float32), cen, truth


def test_detect_spots_uncapped_and_rejection():
    # every blob found with no count cap; caps and relative bar still apply
    dp, cen, truth = _blob_grid_dp()
    w = ShowDiffraction(dp, center=cen, bf_radius=6, verbose=False)
    w.detect_spots()
    assert len(w.spots) == 24  # every blob, no count cap
    assert all(
        any(abs(s["row"] - r) < 2 and abs(s["col"] - c) < 2 for r, c in truth) for s in w.spots
    )
    w.detect_spots(max_spots=5)
    assert len(w.spots) == 5
    w.detect_spots(min_relative=2.0)  # nothing is twice the strongest peak
    assert len(w.spots) == 0
    # low-contrast lattice detected; ring crests rejected
    size = 256
    center = (size - 1) / 2
    rows, cols = np.mgrid[0:size, 0:size]
    lattice = np.zeros((size, size), np.float64)
    for h in range(-2, 3):
        for k in range(-2, 3):
            amp = 6.0 if h == k == 0 else 1.0 / (1 + 0.4 * (h * h + k * k))
            lattice += amp * np.exp(
                -((rows - (center + h * 28.0)) ** 2 + (cols - (center + k * 28.0)) ** 2) / 8.0
            )
    w = ShowDiffraction(
        lattice.astype(np.float32), center=(center, center), bf_radius=14, verbose=False
    )
    w.detect_spots()
    assert len(w.spots) >= 20  # sub-unity intensities: bar is relative, not absolute
    dp, cen = _dp((60.0, 90.0), background="power")
    w2 = ShowDiffraction(dp, center=cen, bf_radius=10, spot_refine=False, verbose=False)
    w2.detect_spots()
    assert len(w2.spots) <= 2  # ring crests are not spots
    # request value -1 means keep all, for spots and rings alike
    dp, cen, _ = _blob_grid_dp()
    w = ShowDiffraction(dp, center=cen, bf_radius=6, verbose=False)
    w._detect_spots_request = -1
    assert len(w.spots) >= 21
    assert w._detect_spots_request == 0
    dp2, cen2 = _dp(tuple(range(40, 265, 25)), size=512, background="power")
    w2 = ShowDiffraction(dp2, center=cen2, bf_radius=15, verbose=False)
    w2._detect_rings_request = -1
    assert len(w2.rings) >= 9  # all 9 rings, not the old top-8
    assert w2._detect_rings_request == 0


def test_detect_spots_close_pair_keeps_strongest():
    # peaks closer than min_distance: the strongest survives, not neither
    size = 128
    rows = np.arange(size, dtype=np.float64)[:, None]
    cols = np.arange(size, dtype=np.float64)[None, :]
    for sep in (4.0, 5.0, 6.0):
        dp = np.zeros((size, size))
        for r0, c0, amp in ((40.0, 40.0, 100.0), (40.0, 40.0 + sep, 60.0), (90.0, 90.0, 80.0)):
            dp += amp * np.exp(-0.5 * (((rows - r0) / 1.5) ** 2 + ((cols - c0) / 1.5) ** 2))
        w = ShowDiffraction(
            dp.astype(np.float32), center=(64, 64), spot_refine=False, verbose=False
        )
        w.detect_spots(min_distance=6, exclude_radius=5)
        positions = [(round(s["row"]), round(s["col"])) for s in w.spots]
        assert positions.count((40, 40)) == 1  # strongest of the pair
        assert (40, 40 + sep) not in positions
        assert (90, 90) in positions  # isolated control
        assert len(w.spots) == 2


def test_move_spot_repicks_and_resets_index():
    size, cen = 128, (64, 64)
    rows = np.arange(size)[:, None]
    cols = np.arange(size)[None, :]

    def blob(r, c, a, s):
        return a * np.exp(-(((rows - r) ** 2 + (cols - c) ** 2) / (2 * s * s)))

    dp = (blob(*cen, 300, 4) + blob(40, 64, 60, 2) + blob(88, 80, 60, 2)).astype(np.float32)
    w = ShowDiffraction(dp, center=cen, bf_radius=6, k_pixel_size=0.01, verbose=False)
    w.add_spot(40.0, 64.0)
    spot = w.spots[0]
    d_before = spot["d_spacing"]
    w.spots = [{**spot, "hkl": "200"}]
    w.move_spot(spot["id"], 87.0, 79.0)  # fit re-picks the nearby blob
    moved = w.spots[0]
    assert moved["id"] == spot["id"]
    assert abs(moved["row"] - 88.0) < 1.0 and abs(moved["col"] - 80.0) < 1.0
    assert moved["hkl"] == ""
    assert moved["d_spacing"] != d_before
    w._spot_move_request = [float(spot["id"]), 40.0, 64.0]
    assert abs(w.spots[0]["row"] - 40.0) < 1.0
    assert w._spot_move_request == []
    w.move_spot(999, 10.0, 10.0)
    assert len(w.spots) == 1


def test_profile_panel(monkeypatch):
    # populates on show_profile with sorted radii peaking at the ring
    dp, cen = _ring_dp([60.0], background="power")
    w = ShowDiffraction(dp, center=cen, bf_radius=10, verbose=False)
    assert w._profile_data == b""
    w.show_profile = True
    arr = np.frombuffer(w._profile_data, np.float32)
    assert arr.size > 0 and arr.size % 2 == 0
    n = arr.size // 2
    radii, intensity = arr[:n], arr[n:]
    assert np.all(np.diff(radii) > 0)
    outside = radii > 20
    assert abs(radii[outside][np.argmax(intensity[outside])] - 60.0) < 3.0
    # subtract-background toggle changes intensities but not radii
    dp, cen = _ring_dp([60.0], background="power")
    w = ShowDiffraction(dp, center=cen, bf_radius=10, verbose=False)
    w.show_profile = True
    raw = np.frombuffer(w._profile_data, np.float32).copy()
    w.profile_subtract_background = True
    net = np.frombuffer(w._profile_data, np.float32)
    n = raw.size // 2
    assert np.allclose(raw[:n], net[:n])
    assert not np.allclose(raw[n:], net[n:])

    # background failure falls back to the raw profile with a failed status
    def boom(**kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(w, "radial_background", boom)
    w.profile_subtract_background = False
    w.profile_subtract_background = True
    assert len(w._profile_data) > 0
    assert "failed" in w.analysis_status
    # the profile recomputes on frame/center changes and clears when hidden
    stack = np.stack(
        [_ring_dp([60.0], background="power")[0], _ring_dp([90.0], background="power")[0]]
    )
    w = ShowDiffraction(stack, center=(128, 128), verbose=False)
    w.frame_idx = 1
    assert w._profile_data == b""  # hidden: nothing computed
    w.show_profile = True
    first = bytes(w._profile_data)
    assert len(first) > 0
    w.frame_idx = 0
    second = bytes(w._profile_data)
    assert second != first
    w.center_row = w.center_row + 4.0
    assert bytes(w._profile_data) != second
    w.show_profile = False
    assert w._profile_data == b""
    # theta limits restrict the profile to the azimuthal sector
    dp, cen = _arc_ring_dp(theta_lo=10.0, theta_hi=70.0)
    w = ShowDiffraction(dp, center=cen, verbose=False)
    w.show_profile = True
    w.profile_theta_min, w.profile_theta_max = 10.0, 70.0
    arr = np.frombuffer(w._profile_data, np.float32)
    n = arr.size // 2
    in_sector = arr[n:][(arr[:n] > 55) & (arr[:n] < 65)].max()
    w.profile_theta_min, w.profile_theta_max = 180.0, 250.0
    arr2 = np.frombuffer(w._profile_data, np.float32)
    out_sector = arr2[n:][(arr2[:n] > 55) & (arr2[:n] < 65)].max()
    assert in_sector > 10.0
    assert out_sector < 2.0


def _library_ring_widget(name="Au", k=0.01, size=256, uncalibrated=False):
    from quantem.widget import library_phase

    ph, cen = library_phase(name), (size // 2, size // 2)
    kwargs = {} if uncalibrated else {"k_pixel_size": k}
    w = ShowDiffraction(np.zeros((size, size), np.float32), center=cen, verbose=False, **kwargs)
    for r in ph.reflections(d_min=1.4)[:5]:
        w.add_ring(1.0 / (r["d"] * k))
    return w


def test_phase_and_indexing_request_channels():
    # calibrate channel: no phase fails, library phase calibrates, custom phase calibrates
    w = _library_ring_widget("Au", uncalibrated=True)
    assert {"Au", "Si", "Fe3O4"} <= {p["name"] for p in w._phase_library}
    w._calibrate_phase_request = True  # no phase selected
    assert "failed" in w.analysis_status and "phase" in w.analysis_status.lower()
    w.phase_name = "Au"
    w._calibrate_phase_request = True
    assert abs(w.k_pixel_size - 0.01) / 0.01 < 0.01
    assert w.calibration_source == "from_phase"
    assert w._calibrate_phase_request is False
    w2 = _library_ring_widget("Au", uncalibrated=True)
    w2.custom_phases = [{"name": "MyAu", "a": 4.0782, "absences": "fcc"}]
    w2.phase_name = "MyAu"
    w2._calibrate_phase_request = True
    assert abs(w2.k_pixel_size - 0.01) / 0.01 < 0.01
    # index-rings and index-spots channels
    w = _library_ring_widget("Au")
    w.phase_name = "Au"
    w._index_rings_request = True
    assert all(r["hkl"] for r in w.rings)
    assert "Au" in w.phase_match
    w2 = _au_spot_widget([((2, 0, 0), 0), ((0, 2, 0), 90), ((2, 2, 0), 45)])
    w2.phase_name = "Au"
    w2._index_spots_request = True
    assert w2.zone_axis == "[001]"


def test_search_phases_and_identify_request():
    # identify channel ranks the spinel family from Fe3O4 rings
    from quantem.widget import library_phase

    ph = library_phase("Fe3O4")
    w = ShowDiffraction(
        np.zeros((512, 512), np.float32), center=(256, 256), k_pixel_size=0.004, verbose=False
    )
    for r in ph.reflections(d_min=1.4)[:6]:
        w.add_ring(1.0 / (r["d"] * 0.004))
    w._identify_request = True
    assert "Fe3O4" in w.analysis_status or "Fe2O3" in w.analysis_status
    assert len(w._identify_results) > 0
    assert w._identify_request is False
    ranked = w.identify_phase(w._all_phases())
    assert ranked[0]["name"] == "Fe3O4"
    # the a = 8.35-8.44 spinel family is near-degenerate on ring positions alone
    spinels = {"Fe3O4", "CoFe2O4", "NiFe2O4", "ZnFe2O4", "γ-Fe2O3"}
    assert {rep["name"] for rep in ranked[:5]} == spinels
    # library search matches Au from its own rings, with a per-line match table
    w = _library_ring_widget("Au")
    reports = w.search_phases()
    from quantem.widget.showdiffraction import match_sort_key

    keys = [match_sort_key(rep) for rep in reports]
    assert keys == sorted(keys)
    for key in ("matched", "n_obs", "mean_err", "n_missing_strong", "lines"):
        assert key in reports[0]
    au = next(rep for rep in reports if rep["name"] == "Au")
    assert au["matched"] == au["n_obs"] == len(w.rings)
    matched = [ln for ln in au["lines"] if ln["obs_d"] is not None and ln["ref_d"] is not None]
    assert len(matched) == len(w.rings)
    assert len(w._identify_results) > 0
    json.dumps(w._identify_results)  # synced trait must be JSON-serializable
    # element filter: Au lacks Fe/O, so it drops out of the candidate list
    filtered = w.search_phases(elements="Fe,O")
    assert "Au" not in {rep["name"] for rep in filtered}
    with pytest.raises(ValueError, match="calibrat"):
        _library_ring_widget("Au", uncalibrated=True).search_phases()
    # on a calibrated pattern the phase matching at scale 1.0 should rank first
    assert w.search_phases()[0]["name"] == "Au"


def test_search_phases_absurd_calibration_bounded():
    # a wrong calibration must not hang the reflection enumeration
    import time

    w = ShowDiffraction(
        np.zeros((256, 256), np.float32), center=(128, 128), k_pixel_size=10.0, verbose=False
    )
    w.add_ring(30.0)
    w.add_ring(50.0)
    t0 = time.perf_counter()
    reports = w.search_phases()
    assert time.perf_counter() - t0 < 120.0
    assert reports


def test_search_phases_skips_degenerate_candidate():
    # one broken candidate must not abort ranking of the rest
    w = _library_ring_widget("Au")
    bad = Phase.from_cubic("Bad", 4.0)
    bad._g_star = -np.eye(3)  # every d_spacing raises "invalid reflection"
    reports = w.search_phases(extra=[bad])
    assert reports[0]["name"] == "Au"
    assert "Bad" not in {rep["name"] for rep in reports}


def test_custom_phases():
    # entries accept a full lattice, defaulting b=a, c=a, angles 90
    from quantem.widget import library_phase

    w = _library_ring_widget("Zr", uncalibrated=True)
    w.custom_phases = [
        {"name": "MyZr", "a": 3.2320, "c": 5.1470, "gamma": 120.0, "absences": "hcp"},
        {"name": "Tet", "a": 3.0, "c": 5.0},
    ]
    w.phase_name = "MyZr"
    zr = w._selected_phase()
    assert zr.lattice == (3.2320, 3.2320, 5.1470, 90.0, 90.0, 120.0)
    assert zr.d_spacing((1, 0, 1)) == pytest.approx(library_phase("Zr").d_spacing((1, 0, 1)))
    w.phase_name = "Tet"
    assert w._selected_phase().lattice == (3.0, 3.0, 5.0, 90.0, 90.0, 90.0)
    # a hexagonal custom phase calibrates and indexes rings like its library twin
    w = _library_ring_widget("Zr", uncalibrated=True)
    w.custom_phases = [
        {"name": "MyZr", "a": 3.2320, "c": 5.1470, "gamma": 120.0, "absences": "hcp"}
    ]
    w.phase_name = "MyZr"
    w._calibrate_phase_request = True
    assert abs(w.k_pixel_size - 0.01) / 0.01 < 0.01
    w._index_rings_request = True
    assert all(r["hkl"] for r in w.rings)
    assert "MyZr" in w.phase_match
    # identify_custom_only ranks only user candidates, not the library
    w = _library_ring_widget("Au")
    w.custom_phases = [{"name": "MyAu", "a": 4.0782, "absences": "fcc"}]
    w.identify_custom_only = True
    w._identify_request = True
    assert {rep["name"] for rep in w._identify_results} == {"MyAu"}
    assert w.search_phases(custom_only=False)[0]["name"] == "Au"
    reports = w.search_phases(custom_only=True, extra=[Phase.from_cubic("ExtraAu", 4.0782)])
    assert {rep["name"] for rep in reports} == {"MyAu", "ExtraAu"}
    assert "identify_custom_only" in w.state_dict()
    w.custom_phases = []
    w._identify_request = True
    assert "Identify failed" in w.analysis_status


def test_malformed_custom_phase_request_resets_flag():
    # KeyError from a malformed entry must not leave the request flag stuck
    w = _library_ring_widget("Au", uncalibrated=True)
    w.custom_phases = [{"name": "Broken"}]  # missing "a"
    w.phase_name = "Broken"
    w._calibrate_phase_request = True
    assert w._calibrate_phase_request is False
    assert "Phase calibration failed" in w.analysis_status
    w.custom_phases = [{"name": "Broken", "a": None}]  # float(None) raises TypeError
    w._index_rings_request = True
    assert w._index_rings_request is False
    assert "Ring indexing failed" in w.analysis_status


def test_export_html_includes_identify_and_quality(tmp_path):
    # identify table and quality panel survive into the offline export
    w = _library_ring_widget("Au")
    w.search_phases()
    w.quality_report()
    assert w._identify_results and w._quality
    clone = w._clone_for_html_export()
    try:
        assert clone._identify_results == w._identify_results
        assert clone._quality == w._quality
    finally:
        clone.close()
    path = w.export_html(tmp_path / "dp.html")
    html = path.read_text()
    assert w._identify_results[0]["name"] in html


def _au_saed_pattern(size=512, cenpx=255.5, k=0.004):
    from quantem.widget import library_phase

    au = library_phase("Au")
    rows = np.arange(size, dtype=np.float64)[:, None]
    cols = np.arange(size, dtype=np.float64)[None, :]
    r = np.hypot(rows - cenpx, cols - cenpx)
    pat = 300.0 * np.exp(-(r**2) / (2 * 8.0**2)) + 20.0 * np.exp(-r / 40.0)
    for refl in au.reflections(d_min=1.2):
        pat += 30.0 * np.exp(-((r - 1.0 / (refl["d"] * k)) ** 2) / (2 * 2.5**2))
    return pat.astype(np.float32)


def test_run_auto():
    # full pipeline in one call
    w = ShowDiffraction(_au_saed_pattern(), verbose=False)
    w.phase_name = "Au"
    w.run_auto(max_rings=4)
    assert abs(w.k_pixel_size - 0.004) / 0.004 < 0.02
    assert all(ring["hkl"] for ring in w.rings)
    assert all(ring.get("fwhm_px") is not None for ring in w.rings)
    assert w.analysis_status == ""
    # Auto button channel
    dp, _ = _ring_dp([60.0, 90.0], background="power")
    w2 = ShowDiffraction(dp, verbose=False)
    w2._auto_request = True
    assert w2._auto_request is False
    assert w2.analysis_status == ""
    # stays silent on success and only surfaces failed steps
    flat = np.full((256, 256), 5.0, dtype=np.float32)
    w = ShowDiffraction(flat, verbose=False)
    w.run_auto()
    assert w.rings == []
    assert w.analysis_status == "Auto: ring detection failed (no rings found)"
    # a selected phase with too few rings surfaces a calibration failure
    dp, _ = _ring_dp([60.0], background="power")
    w = ShowDiffraction(dp, verbose=False)
    w.phase_name = "Au"
    w.run_auto(max_rings=1)
    assert len(w.rings) == 1
    assert "calibration failed" in w.analysis_status
    assert not w.k_calibrated
    # a phase_name that matches no library or custom phase is reported
    dp, _ = _ring_dp([60.0, 90.0], background="power")
    w = ShowDiffraction(dp, verbose=False)
    w.phase_name = "NotAPhase"
    w.run_auto()
    assert 'calibration failed (phase "NotAPhase" not found)' in w.analysis_status


def test_run_auto_calibrates_fitted_radii():
    # calibration uses profile-refined radii: replaying the least-squares scale
    # on the stored rings reproduces k exactly
    w = ShowDiffraction(_au_saed_pattern(), verbose=False)
    w.phase_name = "Au"
    w.run_auto(max_rings=4)
    pairs = [(rng["radius_px"], 1.0 / rng["d_ref"]) for rng in w.rings if rng["hkl"]]
    scale = sum(rp * q for rp, q in pairs) / sum(q * q for _, q in pairs)
    assert w.k_pixel_size == pytest.approx(1.0 / scale, rel=1e-9)
    assert all(rng.get("fwhm_inv_angstrom") is not None for rng in w.rings)


def test_run_auto_reports_unindexed_rings():
    # a default run anchored on unexcluded artifacts leaves rings unmatched
    w = ShowDiffraction(_fe3o4_dp(), verbose=False)
    w.phase_name = "Fe3O4"
    w.run_auto()
    assert "rings unindexed" in w.analysis_status


def _drifted_stack(seed=11):
    # same ring pattern rolled by known shifts, plus one pure-noise garbage frame
    base = _dp((60.0,), background="power")[0]
    shifts = [(0, 0), (2, -1), (-3, 2), (1, 3)]
    frames = [np.roll(base, s, axis=(0, 1)) for s in shifts]
    rng = np.random.default_rng(seed)
    frames.append(rng.random(base.shape).astype(np.float32))
    return np.stack(frames)


def test_merge_frames():
    w = ShowDiffraction(_drifted_stack(), center=(128, 128), verbose=False)
    w.add_ring(60.0)
    report = w.merge_frames()
    assert report["n_frames"] == 5
    assert report["n_used"] >= 4
    assert report["used"][4] is False  # garbage frame gated out
    assert len(report["shifts"]) == 5
    assert {"cv", "coverage", "snr"} <= set(report["before"])
    assert {"cv", "coverage", "snr"} <= set(report["after"])
    assert w.n_frames == 6 and w.frame_idx == 5  # merged pattern appended and shown
    single = ShowDiffraction(_dp((60.0,))[0], verbose=False)
    with pytest.raises(ValueError, match="multi-frame"):
        single.merge_frames()
    with pytest.raises(ValueError, match="statistic"):
        w.merge_frames(statistic="sum")
    # Merge button channel
    w = ShowDiffraction(_drifted_stack(), center=(128, 128), verbose=False)
    w._merge_request = True
    assert w.analysis_status.startswith("Merged")
    assert w._merge_request is False


def test_merge_frames_repeat_uses_originals():
    # a second merge must not fold the appended merged frame back in
    w = ShowDiffraction(_drifted_stack(), center=(128, 128), verbose=False)
    w.add_ring(60.0)
    r1 = w.merge_frames(align=False)
    first = w._data.cpu().numpy()[-1].copy()
    r2 = w.merge_frames(align=False)
    second = w._data.cpu().numpy()[-1].copy()
    assert r1["n_frames"] == r2["n_frames"] == 5
    assert w.n_frames == 6  # merged frame replaced, not stacked
    np.testing.assert_allclose(second, first)
    # replacing the data resets merge provenance
    w.set_image(_drifted_stack())
    assert w.merge_frames(align=False)["n_frames"] == 5


def _staggered_ring_stack(size=128, radii=(30.0, 40.0, 50.0), scales=None):
    rows = np.arange(size, dtype=np.float64)[:, None]
    cols = np.arange(size, dtype=np.float64)[None, :]
    r = np.hypot(rows - size / 2, cols - size / 2)
    scales = scales or [1.0] * len(radii)
    return np.stack(
        [s * 1000.0 * np.exp(-((r - rr) ** 2) / (2 * 1.5**2)) for rr, s in zip(radii, scales)]
    ).astype(np.float32)


def test_measurements_keep_source_frame():
    # records tag their frame; geometry changes resample from that frame
    stack = _staggered_ring_stack()
    w = ShowDiffraction(stack, center=(64, 64), spot_refine=False, verbose=False)
    w.add_ring(30.0)
    w.add_spot(64.0, 94.0)  # on the frame-0 ring
    ring_i, spot_i = w.rings[0]["intensity"], w.spots[0]["intensity"]
    assert w.rings[0]["frame_idx"] == 0
    assert w.spots[0]["frame_idx"] == 0
    assert spot_i > 100.0
    w.frame_idx = 2  # scrub: values persist
    w.center_row = w.center_row + 1e-9  # geometry nudge must not switch frames
    # bin-edge pixels may flip bins on the nudge; frame 2 would read ~0 here
    assert w.rings[0]["intensity"] == pytest.approx(ring_i, rel=0.02)
    assert w.spots[0]["intensity"] == pytest.approx(spot_i, rel=1e-6)
    # state roundtrip: saved values survive a restore with non-default geometry
    w2 = ShowDiffraction(stack, center=(64, 64), spot_refine=False, verbose=False)
    w2.add_ring(30.0)
    w2.add_spot(64.0, 94.0)
    w2.ellipse_ratio = 1.05  # trailing geometry field in the saved state
    saved_ring_i = w2.rings[0]["intensity"]
    w2.frame_idx = 2
    state = json.loads(json.dumps(w2.state_dict()))
    w3 = ShowDiffraction(stack, center=(64, 64), spot_refine=False, verbose=False)
    w3.load_state_dict(state)
    assert w3.frame_idx == 2
    assert w3.rings[0]["intensity"] == pytest.approx(saved_ring_i, rel=1e-6)


def test_merge_repeat_refreshes_profile_panes():
    # a repeat merge leaves frame_idx unchanged; the panes must still refresh
    stack = _staggered_ring_stack(radii=(30.0, 30.0, 30.0), scales=(1.0, 2.0, 3.0))
    w = ShowDiffraction(stack, center=(64, 64), verbose=False)
    w.add_ring(30.0)
    w.show_profile = True
    w.show_azimuthal = True
    w.merge_frames(align=False)
    profile_mean, azimuthal_mean = w._profile_data, w._azimuthal_data
    w.merge_frames(statistic="max", align=False)
    assert w._profile_data != profile_mean
    assert w._azimuthal_data != azimuthal_mean


def test_state_dict_merge_provenance_and_clamp_warning():
    stack = _staggered_ring_stack()
    w = ShowDiffraction(stack, center=(64, 64), verbose=False)
    w.merge_frames(align=False)
    state = json.loads(json.dumps(w.state_dict()))
    assert state["n_source_frames"] == 3
    # restored onto source+merged data, a re-merge keeps 3 sources
    data4 = w._data.cpu().numpy()
    w2 = ShowDiffraction(data4, center=(64, 64), verbose=False)
    w2.load_state_dict(state)
    assert w2.merge_frames(align=False)["n_frames"] == 3
    assert w2.n_frames == 4
    # restoring onto fewer frames clamps frame_idx and says so
    w3 = ShowDiffraction(stack, center=(64, 64), verbose=False)
    w3.load_state_dict(state)
    assert w3.frame_idx == 2
    assert "clamp" in w3.analysis_status.lower()


def test_set_image_refreshes_profile_panes():
    # new data with an unchanged auto-detected center must refresh the profile
    stack = _staggered_ring_stack()
    w = ShowDiffraction(stack[0], verbose=False)
    w.show_profile = True
    before = w._profile_data
    w.set_image(np.zeros((128, 128), np.float32))  # empty mask: center unchanged
    after = w._profile_data
    assert after != before
    w._update_profile()
    assert w._profile_data == after  # already fresh


def test_quality_report_frame_attribution():
    # ring_snr follows the ring's source frame; the report names its frame
    stack = _staggered_ring_stack()
    w = ShowDiffraction(stack, center=(64, 64), verbose=False)
    w.add_ring(30.0)
    snr0 = w.quality_report()["ring_snr"]
    w.frame_idx = 2
    report = w.quality_report()
    assert report["frame_idx"] == 2
    assert report["ring_snr"] == snr0


def test_constructor_frame_idx_kwarg():
    # frame_idx applies after ingest, not against the default n_frames=1
    stack = _staggered_ring_stack()
    w = ShowDiffraction(stack, center=(64, 64), frame_idx=2, verbose=False)
    assert w.frame_idx == 2
    np.testing.assert_array_equal(
        np.frombuffer(w.frame_bytes, dtype=np.float32).reshape(128, 128), stack[2]
    )


def test_quality_report():
    dp, cen = _ring_dp([60.0, 90.0], background="power")
    w = ShowDiffraction(dp, center=cen, bf_radius=10, k_pixel_size=0.02, verbose=False)
    for rr in (60.0, 90.0):
        w.add_ring(rr)
    w.fit_ring_profile()
    w.mask_regions = [{"kind": "wedge", "start_deg": 0.0, "end_deg": 90.0}]
    report = w.quality_report()
    for key in (
        "center",
        "calibration",
        "ellipse",
        "rings",
        "n_unexplained_rings",
        "mask_coverage_pct",
    ):
        assert key in report
    assert report["center"]["method"]
    assert report["mask_coverage_pct"] > 0
    assert len(report["rings"]) == 2
    assert w._quality == report  # synced trait
    json.dumps(w._quality)
    w2 = ShowDiffraction(dp, center=cen, bf_radius=10, verbose=False)
    w2.add_ring(60.0)
    assert w2._quality == {}
    w2._fit_rings_request = True  # QC refresh
    assert w2._quality != {}
    assert "center" in w2._quality
    # QC button channel refreshes the snapshot
    dp, cen = _ring_dp([60.0], background="power")
    w = ShowDiffraction(dp, center=cen, bf_radius=10, verbose=False)
    w.add_ring(60.0)
    assert w._quality == {}
    w._quality_request = True
    assert w._quality_request is False
    assert "center" in w._quality
    assert w.analysis_status == "Quality updated"


def test_quality_report_ring_snr_respects_mask():
    # dead-wedge pattern: masking the wedge lifts ring coverage
    dp, cen = _ring_dp([60.0])
    rows, cols = np.indices(dp.shape, dtype=np.float64)
    theta = np.degrees(np.arctan2(rows - cen[0], cols - cen[1])) % 360.0
    dp = np.where((theta >= 0.0) & (theta <= 90.0), 0.0, dp).astype(np.float32)
    w = ShowDiffraction(dp, center=cen, verbose=False)
    w.add_ring(60.0)
    before = w.quality_report()["ring_snr"]
    w.mask_regions = [{"kind": "wedge", "start_deg": 0.0, "end_deg": 90.0}]
    after = w.quality_report()["ring_snr"]
    assert after["coverage"] > before["coverage"]
    assert after["cv"] < before["cv"]


def test_mask_regions():
    # masks exclude from the radial profile
    dp, cen = _arc_ring_dp(theta_lo=10.0, theta_hi=70.0)
    w = ShowDiffraction(dp, center=cen, verbose=False)
    radii, before = w.radial_profile(units="px")
    band = (radii > 55) & (radii < 65)
    w.mask_regions = [{"kind": "wedge", "start_deg": 0.0, "end_deg": 90.0}]
    _, after = w.radial_profile(units="px")
    assert before[band].max() > 4.0
    assert after[band].max() < 2.0
    dp2, cen2 = _ring_dp([60.0], background="power")
    w2 = ShowDiffraction(dp2, center=cen2, verbose=False)
    radii2, before2 = w2.radial_profile(units="px")
    w2.mask_regions = [{"kind": "disk", "row": cen2[0], "col": cen2[1] + 60.0, "radius": 15.0}]
    _, after2 = w2.radial_profile(units="px")
    band2 = (radii2 > 55) & (radii2 < 65)
    assert after2[band2].max() < before2[band2].max()
    # masks respected by detect_spots
    dp = np.zeros((128, 128), np.float32)
    dp[40, 40] = 100.0
    dp[90, 90] = 100.0
    w = ShowDiffraction(dp, center=(64, 64), spot_refine=False, verbose=False)
    w.mask_regions = [{"kind": "disk", "row": 40.0, "col": 40.0, "radius": 8.0}]
    w.detect_spots(max_spots=5, exclude_radius=5)
    positions = [(round(s["row"]), round(s["col"])) for s in w.spots]
    assert (90, 90) in positions
    assert (40, 40) not in positions


def test_full_circle_wedge_masks_all():
    # a wedge spanning >= 360 degrees masks every pixel
    from quantem.widget.showdiffraction import build_analysis_mask

    shape, cen = (64, 64), (31.5, 31.5)
    for start, end in ((0.0, 360.0), (90.0, 450.0)):
        mask = build_analysis_mask(
            shape, [{"kind": "wedge", "start_deg": start, "end_deg": end}], cen
        )
        assert mask.all()
    quarter = build_analysis_mask(
        shape, [{"kind": "wedge", "start_deg": 0.0, "end_deg": 90.0}], cen
    )
    assert abs(quarter.mean() - 0.25) < 0.02
    wrap = build_analysis_mask(
        shape, [{"kind": "wedge", "start_deg": 350.0, "end_deg": 20.0}], cen
    )
    assert 0.06 < wrap.mean() < 0.11


def test_azimuthal_panel_populates():
    dp, cen = _arc_ring_dp(theta_lo=10.0, theta_hi=70.0)
    w = ShowDiffraction(dp, center=cen, verbose=False)
    w.add_ring(60.0)
    w.show_azimuthal = True
    arr = np.frombuffer(w._azimuthal_data, np.float32)
    n = arr.size // 2
    theta, intensity = arr[:n], arr[n:]
    assert 0 <= theta[np.argmax(intensity)] <= 90
    w.show_azimuthal = False
    assert w._azimuthal_data == b""
    w2 = ShowDiffraction(_ring_dp([60.0])[0], center=(128, 128), verbose=False)
    w2.show_azimuthal = True  # no rings
    assert w2._azimuthal_data == b""
    assert "failed" in w2.analysis_status


def test_analysis_state_roundtrip():
    dp, cen = _elliptical_ring_dp()
    w = ShowDiffraction(dp, center=cen, bf_radius=10, verbose=False)
    w.add_ring(60.0)
    w.fit_ellipse()
    w.apply_ellipse_correction()
    w.mask_regions = [{"kind": "wedge", "start_deg": 350.0, "end_deg": 20.0}]
    w.show_profile = True
    w.profile_log = False
    w.profile_subtract_background = True
    sd = w.state_dict()
    assert "analysis_status" not in sd and "_profile_data" not in sd
    w2 = ShowDiffraction(np.zeros((256, 256), np.float32), verbose=False)
    w2.load_state_dict(json.loads(json.dumps(sd)))
    assert w2.ellipse_ratio == w.ellipse_ratio
    assert w2.ellipse_angle == w.ellipse_angle
    assert w2.ellipse_corrected is True
    assert w2.mask_regions[0]["kind"] == "wedge"
    assert w2.show_profile is True and w2.profile_log is False
    assert w2.profile_subtract_background is True
    assert len(w2._profile_data) > 0
    w3 = _au_ring_widget()
    w3.index_rings(_au())
    state = json.loads(json.dumps(w3.state_dict()))
    w4 = ShowDiffraction(np.zeros((256, 256), np.float32), verbose=False)
    w4.load_state_dict(state)
    assert w4.phase_match == w3.phase_match
    assert [rng["hkl"] for rng in w4.rings] == [rng["hkl"] for rng in w3.rings]
    # refine method, center method, and element filter roundtrip to the export header
    dp = _off_center_ring_dp((124.0, 130.5), radii=(50.0, 75.0))
    w = ShowDiffraction(dp, center=(121.0, 133.0), verbose=False)
    w.refine_method = "phase_corr"
    w.identify_elements = "Fe,O"
    w.refine_center(method="phase_corr")
    w.mask_regions = [{"kind": "wedge", "start_deg": 0.0, "end_deg": 90.0}]
    w.profile_subtract_background = True
    sd = w.state_dict()
    for key in ("refine_method", "center_method", "identify_elements"):
        assert key in sd
    w2 = ShowDiffraction(np.zeros((256, 256), np.float32), verbose=False)
    w2.load_state_dict(json.loads(json.dumps(sd)))
    assert w2.refine_method == "phase_corr"
    assert w2.center_method == "phase_corr"
    assert w2.identify_elements == "Fe,O"
    meta = measurement_metadata(w.state_dict())
    assert meta["center_method"] == "phase_corr"
    assert meta["mask_regions"] == [{"kind": "wedge", "start_deg": 0.0, "end_deg": 90.0}]
    assert meta["background_subtracted"] is True
