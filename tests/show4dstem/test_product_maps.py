"""Show4DSTEM precomputed image-source maps."""

import json
import pathlib
import sys
import time
import types

import numpy as np
import pytest

from quantem.widget import Show4DSTEM
from quantem.widget.show4dstem import Show4DSTEM as Show4DSTEMBase


def _data(shape=(4, 4, 8, 8)):
    rng = np.random.default_rng(7)
    data = rng.poisson(4, shape).astype(np.uint16)
    data[..., 3:5, 3:5] += 100
    return data


def test_show4dstem_accepts_dpc_and_ssb_product_maps_without_resending_bytes():
    """C1: static DPC/SSB maps, expect one product payload and source-only switches."""
    scan = (4, 4)
    dpc_r = np.linspace(-1, 1, np.prod(scan), dtype=np.float32).reshape(scan)
    dpc_c = dpc_r.T.copy()
    ssb = np.sin(np.linspace(0, np.pi, np.prod(scan), dtype=np.float32)).reshape(scan)

    widget = Show4DSTEM(_data(), DPC_row=dpc_r, DPC_col=dpc_c, SSB=ssb, verbose=False)
    roi_payload = widget.virtual_image_bytes
    product_payload = widget.vi_product_maps_bytes

    assert widget.vi_product_labels == ["DPC_row", "DPC_col", "SSB"]
    assert len(product_payload) == 3 * np.prod(scan) * 4

    widget.vi_source = "DPC_row"
    assert widget.vi_source == "DPC_row"
    assert widget.vi_colormap == "inferno"
    assert widget.virtual_image_bytes == roi_payload
    assert widget.vi_product_maps_bytes == product_payload
    np.testing.assert_allclose(widget._get_virtual_image_array(), dpc_r)

    widget.vi_source = "SSB"
    assert widget.vi_source == "SSB"
    assert widget.vi_colormap == "inferno"
    assert widget.virtual_image_bytes == roi_payload
    assert widget.vi_product_maps_bytes == product_payload
    np.testing.assert_allclose(widget._get_virtual_image_array(), ssb)


def test_show4dstem_product_maps_follow_frame_index_without_resending_vi_bytes():
    """C2: 5D product maps, expect frame_idx to choose the displayed product frame."""
    ssb = np.stack(
        [
            np.full((4, 4), -0.25, dtype=np.float32),
            np.full((4, 4), 0.75, dtype=np.float32),
        ],
        axis=0,
    )
    widget = Show4DSTEM(_data((2, 4, 4, 8, 8)), SSB=ssb, vi_source="SSB", verbose=False)
    product_payload = widget.vi_product_maps_bytes
    vi_payload = widget.virtual_image_bytes

    np.testing.assert_allclose(widget._get_virtual_image_array(), ssb[0])
    widget.frame_idx = 1
    np.testing.assert_allclose(widget._get_virtual_image_array(), ssb[1])
    assert widget.vi_product_maps_bytes == product_payload
    assert widget.virtual_image_bytes == vi_payload


def test_show4dstem_product_source_switches_keep_global_colormap():
    """C2b: source switches must not mutate the shared VI color/scale controls."""
    ssb = np.linspace(-0.5, 0.5, 16, dtype=np.float32).reshape(4, 4)

    widget = Show4DSTEM(_data(), SSB=ssb, verbose=False)

    widget.vi_source = "SSB"
    assert widget.vi_colormap == "inferno"
    assert widget.vi_scale_mode == "linear"

    widget.apply_preset("bf")
    assert widget.vi_source == "roi"
    assert widget.vi_colormap == "inferno"
    assert widget.vi_scale_mode == "linear"

    widget.vi_colormap = "viridis"
    widget.vi_scale_mode = "log"
    widget.vi_source = "SSB"
    assert widget.vi_colormap == "viridis"
    assert widget.vi_scale_mode == "log"

    widget.vi_colormap = "gray"
    widget.apply_preset("abf")
    assert widget.vi_source == "roi"
    assert widget.vi_colormap == "gray"
    assert widget.vi_scale_mode == "log"


def test_show4dstem_initial_product_source_keeps_global_colormap():
    """C2c: SSB-first widgets should open with the shared VI color."""
    ssb = np.ones((4, 4), dtype=np.float32)
    widget = Show4DSTEM(_data(), SSB=ssb, vi_source="SSB", verbose=False)

    assert widget.vi_source == "SSB"
    assert widget.vi_colormap == "inferno"
    assert widget.vi_scale_mode == "linear"


def test_show4dstem_compare_product_maps_update_indices_not_image_payload():
    """C3: multiple view product maps, expect static maps plus small index/status updates."""
    dpc_r = np.stack([np.full((4, 4), idx - 1, dtype=np.float32) for idx in range(3)])
    dpc_c = dpc_r + 10
    ssb = dpc_r + 20
    widget = Show4DSTEM(
        _data((3, 4, 4, 8, 8)),
        DPC_row=dpc_r,
        DPC_col=dpc_c,
        SSB=ssb,
        view_mode="multiple",
        verbose=False,
    )
    compare_payload = widget.compare_virtual_image_bytes
    product_payload = widget.vi_product_maps_bytes

    for source, expected in [("DPC_row", dpc_r), ("DPC_col", dpc_c), ("SSB", ssb)]:
        widget.vi_source = source
        assert widget.compare_panel_count == 3
        assert widget.compare_panel_indices == [0, 1, 2]
        assert widget.compare_virtual_image_bytes == compare_payload
        assert widget.vi_product_maps_bytes == product_payload
        np.testing.assert_allclose(widget._get_virtual_image_array(), expected[0])


def test_show4dstem_rejects_complex_ssb_map():
    """C4: ambiguous complex SSB, expect a corrective error."""
    ssb = np.ones((4, 4), dtype=np.complex64)
    with pytest.raises(ValueError, match="real-valued map"):
        Show4DSTEM(_data(), SSB=ssb, verbose=False)


def test_show4dstem_rejects_product_map_shape_mismatch():
    """C5: wrong scan shape, expect a clear shape error."""
    with pytest.raises(ValueError, match="does not match the scan shape"):
        Show4DSTEM(_data(), DPC_row=np.ones((3, 4), dtype=np.float32), verbose=False)


def test_show4dstem_compute_ssb_attaches_static_product_map(monkeypatch):
    """C6: kernel SSB result becomes the normal one-transfer SSB product map."""
    scan = (4, 4)
    dpc_r = np.arange(np.prod(scan), dtype=np.float32).reshape(scan)
    phase = np.linspace(-0.5, 0.5, np.prod(scan), dtype=np.float32).reshape(scan)
    seen = {}

    def fake_compute(self, **kwargs):
        seen.update(kwargs)
        return phase, phase * 0.1, phase * 0.2

    monkeypatch.setattr(Show4DSTEMBase, "_compute_ssb_phase", fake_compute)
    widget = Show4DSTEM(_data(), DPC_row=dpc_r, verbose=False)
    before_vi_payload = widget.virtual_image_bytes

    result = widget.compute_ssb(n_trials=0, refine=False, set_source=True)

    np.testing.assert_allclose(result, phase)
    assert seen["n_trials"] == 0
    assert seen["refine"] is False
    assert widget.vi_source == "SSB"
    assert widget.vi_colormap == "inferno"
    assert widget.vi_product_labels == ["DPC_row", "DPC_col", "SSB"]
    # compute_ssb refreshes DPC alongside the phase (CoM comes for free)
    np.testing.assert_allclose(widget._vi_product_maps["DPC_row"][0], phase * 0.1)
    np.testing.assert_allclose(widget._vi_product_maps["DPC_col"][0], phase * 0.2)
    assert widget.virtual_image_bytes == before_vi_payload
    np.testing.assert_allclose(widget._get_virtual_image_array(), phase)


def test_show4dstem_ssb_compute_request_runs_in_background(monkeypatch):
    """C7: frontend request trait computes SSB and consumes the trigger."""
    phase = np.full((4, 4), 1.25, dtype=np.float32)

    def fake_compute(self, **kwargs):
        assert kwargs["n_trials"] == 0
        assert kwargs["refine"] is False
        assert kwargs["bf_subsample"] == 0.3
        return phase, phase * 0.1, phase * 0.2

    monkeypatch.setattr(Show4DSTEMBase, "_compute_ssb_phase", fake_compute)
    widget = Show4DSTEM(_data(), verbose=False)

    widget.ssb_compute_request = json.dumps(
        {"action": "compute_ssb", "n_trials": 0, "refine": False, "bf_subsample": 0.3}
    )

    for _ in range(100):
        if widget.ssb_compute_status.startswith("SSB ready"):
            break
        time.sleep(0.01)

    assert widget.ssb_compute_request == ""
    assert widget.ssb_compute_busy is False
    assert widget.ssb_compute_status.startswith("SSB ready")
    assert widget.ssb_compute_n_trials == 0
    assert widget.ssb_compute_refine is False
    assert widget.ssb_compute_bf_subsample == 0.3
    assert widget.vi_source == "SSB"
    assert widget.vi_product_labels == ["DPC_row", "DPC_col", "SSB"]
    np.testing.assert_allclose(widget._get_virtual_image_array(), phase)


def test_show4dstem_ssb_advanced_defaults_are_synced():
    """C7b: More-menu defaults use the production SSB search policy."""
    widget = Show4DSTEM(_data(), verbose=False)

    assert widget.ssb_compute_n_trials == 200
    assert widget.ssb_compute_refine is True
    assert widget.ssb_compute_bf_subsample == 1.0
    assert widget.ssb_compute_manual_aberrations is False


def test_show4dstem_ssb_advanced_overrides_are_synced():
    """C7c: More-menu overrides mirror constructor SSB controls."""
    widget = Show4DSTEM(
        _data(),
        ssb_n_trials=50,
        ssb_refine=False,
        ssb_bf_subsample=0.5,
        verbose=False,
    )

    assert widget.ssb_compute_n_trials == 50
    assert widget.ssb_compute_refine is False
    assert widget.ssb_compute_bf_subsample == 0.5


def test_show4dstem_frontend_more_menu_direct_ptycho_contract():
    """C7d: More menu, expect one-click phase compute plus advanced controls."""
    source = pathlib.Path("js/show4dstem/index.tsx").read_text()
    request = source[
        source.index("const requestSsbCompute = React.useCallback("):
        source.index("const requestSsbManualReconstruct", source.index("const requestSsbCompute = React.useCallback("))
    ]
    more_menu = source[
        source.index("{ssbComputeEnabled && <Button"):
        source.index("{exportEnabled && (localHtmlExportStatus || exportStatus)", source.index("{ssbComputeEnabled && <Button"))
    ]
    calibration_panel = source[
        source.index("{showSsbCalibrationPanel && ("):
        source.index("{effectiveShowFft && (", source.index("{showSsbCalibrationPanel && ("))
    ]

    assert "title=\"More actions\"" in more_menu
    assert "More" in more_menu
    assert "Calculate Phase" in more_menu
    assert "Compute SSB" not in more_menu
    assert "Trials" in more_menu
    assert "<MenuItem value={200}>200</MenuItem>" in more_menu
    assert "Refine" in more_menu
    assert "BF ratio" in more_menu
    assert "<MenuItem value={0.3}>0.3</MenuItem>" in more_menu
    assert "<MenuItem value={0.5}>0.5</MenuItem>" in more_menu
    assert "<MenuItem value={1}>1.0</MenuItem>" in more_menu
    assert "Lock C10" in more_menu
    assert "Lock C12" in more_menu
    assert "Default is 200 trials, refine on, BF ratio 1.0" in more_menu
    assert "Download calibration JSON" in more_menu
    assert "Running SSB..." in source

    assert 'action: "compute_ssb"' in request
    assert "n_trials: nTrials" in request
    assert "refine: Boolean(ssbComputeRefine)" in request
    assert "bf_subsample: bfSubsample" in request
    assert "manual_aberrations: manualAberrations" in request
    assert "model.set(\"ssb_compute_request\", payload);" in request
    assert "model.save_changes();" in request

    assert "SSB calibration" in calibration_panel
    assert "Download JSON" in calibration_panel
    assert "scheduleSsbTuneCommit" in calibration_panel
    assert "commitSsbTuneNow" in calibration_panel
    assert "C10" in calibration_panel
    assert "C12" in calibration_panel
    assert "φ12" in calibration_panel
    assert "Rotation" in calibration_panel


def test_show4dstem_ssb_manual_coeff_request_passes_solver_kwargs(monkeypatch):
    """C7e: manual SSB coefficients are opt-in and passed in solver units."""
    phase = np.full((4, 4), -0.5, dtype=np.float32)
    seen = {}

    def fake_compute(self, **kwargs):
        seen.update(kwargs)
        return phase, phase * 0.1, phase * 0.2

    monkeypatch.setattr(Show4DSTEMBase, "_compute_ssb_phase", fake_compute)
    widget = Show4DSTEM(_data(), verbose=False)

    widget.ssb_compute_request = json.dumps(
        {
            "action": "compute_ssb",
            "n_trials": 0,
            "refine": False,
            "bf_subsample": 1.0,
            "manual_aberrations": True,
            "c10_nm": 42.0,
            "c12_nm": 13.0,
            "phi12_deg": 30.0,
            "rotation_angle_deg": 73.5,
        }
    )

    for _ in range(100):
        if widget.ssb_compute_status.startswith("SSB ready"):
            break
        time.sleep(0.01)

    assert widget.ssb_compute_manual_aberrations is True
    assert widget.ssb_compute_c10_nm == 42.0
    assert widget.ssb_compute_c12_nm == 13.0
    assert widget.ssb_compute_phi12_deg == 30.0
    assert widget.ssb_compute_rotation_angle_deg == 73.5
    assert seen["bf_subsample"] == 1.0
    assert seen["lock_aberrations"] is True
    assert seen["rotation_angle_deg"] == 73.5
    assert seen["aberrations"]["C10"] == 42.0
    assert seen["aberrations"]["C12"] == 13.0
    assert seen["aberrations"]["phi12"] == pytest.approx(np.deg2rad(30.0))


def test_show4dstem_html_export_clone_disables_live_ssb_compute():
    """C8: standalone HTML can show SSB maps, but cannot advertise live compute."""
    ssb = np.ones((4, 4), dtype=np.float32)
    widget = Show4DSTEM(_data(), SSB=ssb, vi_source="SSB", verbose=False)

    clone = widget._clone_for_html_export(dtype="uint8", det_bin=1)
    try:
        assert clone.ssb_compute_enabled is False
        assert clone.ssb_compute_status == ""
        assert clone.ssb_compute_calibration_json == ""
        assert clone.ssb_compute_calibration_filename == ""
        assert clone.vi_source == "SSB"
        assert clone.vi_product_labels == ["SSB"]
        np.testing.assert_allclose(clone._get_virtual_image_array(), ssb)
    finally:
        clone.close()


def test_show4dstem_compute_ssb_phase_passes_solver_arguments(monkeypatch):
    """C9: real SSB plumbing resolves metadata and calls quantem.gpu.ssb.SSB."""
    phase = np.full((128, 128), 0.25, dtype=np.float32)
    calls = {}

    class FakeData:
        ndim = 4

    class FakeResult:
        def __init__(self, phase):
            self.phase = phase
            self.aberrations = {"C10": 12.0, "C12": 4.0, "phi12": np.deg2rad(15.0)}
            self.rotation_angle_deg = 2.5
            self.loss = 0.125
            self.elapsed = 1.5
            self.refine_method = "fake-nmead"
            self.refine_nfev = 7
            self.refine_elapsed = 0.4

    class FakeSSB:
        def __init__(self, data, **kwargs):
            calls["data"] = data
            calls["init"] = kwargs
            self.bf_inds_row = np.arange(120, dtype=np.int32)

        def optimize(self, **kwargs):
            calls["optimize"] = kwargs

        def refine(self, **kwargs):
            calls["refine"] = kwargs

        def result(self):
            calls["result"] = True
            return FakeResult(phase)

    widget = Show4DSTEM(
        _data((128, 128, 4, 4)),
        sampling=(0.5, 0.5, 0.25, 0.25),
        units=("A", "A", "mrad", "mrad"),
        bf_radius=20,
        ssb_voltage_kV=300,
        verbose=False,
    )
    fake_data = FakeData()
    gpu_module = types.ModuleType("quantem.gpu")
    ssb_module = types.ModuleType("quantem.gpu.ssb")
    ssb_module.SSB = FakeSSB
    dpc_module = types.ModuleType("quantem.gpu.dpc")
    fake_com_row = np.linspace(-1.0, 1.0, 128 * 128).astype(np.float32).reshape(128, 128)
    fake_com_col = np.linspace(1.0, -1.0, 128 * 128).astype(np.float32).reshape(128, 128)
    dpc_module.center_of_mass = lambda data, **kw: (fake_com_row, fake_com_col)
    monkeypatch.setitem(sys.modules, "quantem.gpu", gpu_module)
    monkeypatch.setitem(sys.modules, "quantem.gpu.ssb", ssb_module)
    monkeypatch.setitem(sys.modules, "quantem.gpu.dpc", dpc_module)
    monkeypatch.setattr(widget, "_ssb_cupy_frame", lambda: (fake_data, object()))

    out, dpc_row, dpc_col = widget._compute_ssb_phase(
        n_trials=2, refine=True, seed=11, bf_subsample=0.5
    )

    np.testing.assert_allclose(out, phase)
    theta = np.deg2rad(2.5)
    expected_row = np.cos(theta) * fake_com_row - np.sin(theta) * fake_com_col
    expected_col = np.sin(theta) * fake_com_row + np.cos(theta) * fake_com_col
    np.testing.assert_allclose(dpc_row, expected_row, atol=1e-6)
    np.testing.assert_allclose(dpc_col, expected_col, atol=1e-6)
    assert calls["data"] is fake_data
    assert calls["init"]["semiangle"] == pytest.approx(5.0)
    assert calls["init"]["scan_sampling"] == (0.5, 0.5)
    assert calls["init"]["det_sampling"] == (0.25, 0.25)
    assert calls["init"]["voltage_kV"] == 300.0
    assert calls["init"]["scan_shape"] is None
    assert calls["init"]["bf_intensity_threshold"] == 0.5
    assert widget.ssb_compute_bf_pixels == 120
    assert widget.ssb_compute_bf_selected_pixels == 60
    assert calls["optimize"]["n_trials"] == 2
    assert calls["optimize"]["seed"] == 11
    assert calls["optimize"]["bf_subsample"] == 0.5
    assert calls["refine"]["bf_subsample"] == 0.5
    assert calls["result"] is True
    calibration = json.loads(widget.ssb_compute_calibration_json)
    assert calibration["schema"] == "quantem.ssb.calibration.v1"
    assert calibration["aberrations"]["C10"] == 12.0
    assert calibration["aberrations"]["C12"] == 4.0
    assert calibration["aberrations"]["phi12"] == pytest.approx(np.deg2rad(15.0))
    assert calibration["rotation_angle_deg"] == 2.5
    assert calibration["calibration"]["bf_pixels"] == 120
    assert calibration["calibration"]["bf_selected_pixels"] == 60
    assert calibration["run"]["n_trials"] == 2
    assert calibration["run"]["loss"] == 0.125
    assert widget.ssb_compute_calibration_filename.endswith("_ssb_calibration.json")


def test_show4dstem_compute_ssb_phase_locks_manual_coefficients(monkeypatch):
    """C10: manual coefficients reconstruct directly instead of drifting."""
    phase = np.full((128, 128), -0.125, dtype=np.float32)
    calls = {}

    class FakeData:
        ndim = 4

    class FakeResult:
        def __init__(self, phase, aberrations, rotation_angle_deg):
            self.phase = phase
            self.aberrations = dict(aberrations)
            self.rotation_angle_deg = rotation_angle_deg
            self.loss = None
            self.elapsed = None

    class FakeSSB:
        def __init__(self, data, **kwargs):
            calls["data"] = data
            calls["init"] = kwargs
            self.bf_inds_row = np.arange(120, dtype=np.int32)
            self.aberrations = dict(kwargs.get("aberrations") or {})
            self.rotation_angle_deg = float(kwargs.get("rotation_angle_deg", 0.0))

        def optimize(self, **kwargs):
            calls["optimize"] = kwargs

        def refine(self, **kwargs):
            calls["refine"] = kwargs

        def result(self):
            calls["result"] = True
            return FakeResult(phase, self.aberrations, self.rotation_angle_deg)

    widget = Show4DSTEM(
        _data((128, 128, 4, 4)),
        sampling=(0.5, 0.5, 0.25, 0.25),
        units=("A", "A", "mrad", "mrad"),
        bf_radius=20,
        ssb_voltage_kV=300,
        verbose=False,
    )
    fake_data = FakeData()
    gpu_module = types.ModuleType("quantem.gpu")
    ssb_module = types.ModuleType("quantem.gpu.ssb")
    ssb_module.SSB = FakeSSB
    dpc_module = types.ModuleType("quantem.gpu.dpc")
    fake_com_row = np.linspace(-1.0, 1.0, 128 * 128).astype(np.float32).reshape(128, 128)
    fake_com_col = np.linspace(1.0, -1.0, 128 * 128).astype(np.float32).reshape(128, 128)
    dpc_module.center_of_mass = lambda data, **kw: (fake_com_row, fake_com_col)
    monkeypatch.setitem(sys.modules, "quantem.gpu", gpu_module)
    monkeypatch.setitem(sys.modules, "quantem.gpu.ssb", ssb_module)
    monkeypatch.setitem(sys.modules, "quantem.gpu.dpc", dpc_module)
    monkeypatch.setattr(widget, "_ssb_cupy_frame", lambda: (fake_data, object()))

    out = widget._compute_ssb_phase(
        n_trials=200,
        refine=True,
        seed=11,
        bf_subsample=0.3,
        aberrations={"C10": 42.0, "C12": 13.0, "phi12": np.deg2rad(30.0)},
        rotation_angle_deg=73.5,
        lock_aberrations=True,
    )
    out, dpc_row, dpc_col = out

    np.testing.assert_allclose(out, phase)
    theta = np.deg2rad(73.5)
    expected_row = np.cos(theta) * fake_com_row - np.sin(theta) * fake_com_col
    np.testing.assert_allclose(dpc_row, expected_row, atol=1e-6)
    assert calls["init"]["aberrations"]["C10"] == 42.0
    assert calls["init"]["aberrations"]["C12"] == 13.0
    assert calls["init"]["aberrations"]["phi12"] == pytest.approx(np.deg2rad(30.0))
    assert calls["init"]["rotation_angle_deg"] == 73.5
    assert "optimize" not in calls
    assert "refine" not in calls
    assert widget.ssb_compute_bf_pixels == 120
    assert widget.ssb_compute_bf_selected_pixels == 40
    assert calls["result"] is True
    calibration = json.loads(widget.ssb_compute_calibration_json)
    assert calibration["aberrations"]["C10"] == 42.0
    assert calibration["aberrations"]["C12"] == 13.0
    assert calibration["aberrations"]["phi12"] == pytest.approx(np.deg2rad(30.0))
    assert calibration["rotation_angle_deg"] == 73.5
    assert calibration["run"]["manual_locked"] is True
    assert calibration["calibration"]["bf_selected_pixels"] == 40
