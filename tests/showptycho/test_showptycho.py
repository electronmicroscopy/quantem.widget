"""ShowPtycho: a reconstruction array on any computer, and a real quantem.gpu SSB session.

Arrays (NumPy or torch on any device, real phase or complex object wave) must show exactly the
values given, to float32, with no quantem.gpu. For the session, a synthetic 128 x 128 scan of a
bright-field disk that shifts with a smooth phase object is fitted nowhere; the session carries
starting aberrations and the widget must show exactly the phase ``SSB.preview`` returns for them,
follow slider requests, write the calibration and star files, and export the browser folder with
exact bright-field counts.
"""

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from quantem.widget import ShowPtycho
from quantem.widget.adapters import gpu as gpu_adapter
from quantem.widget.device import INSTALL_CUDA, INSTALL_MPS, NO_GPU_PATH
from quantem.widget.showptycho import widget as showptycho_module

SCAN, DET = 128, 16
START = {"C10": 3.0, "C12": 1.0, "phi12": 0.2}
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else []) + (["mps"] if torch.backends.mps.is_available() else [])


def _object_wave(dtype=np.complex64) -> np.ndarray:
    """A weak-phase lattice (0.3 rad peaks) with 10 % amplitude contrast, 96 x 80 so rows and columns differ."""
    rows, cols = np.mgrid[0:96, 0:80].astype(np.float64)
    phase = 0.3 * np.cos(2 * np.pi * rows / 12.0) * np.cos(2 * np.pi * cols / 10.0)
    amplitude = 1.0 - 0.1 * (np.sin(2 * np.pi * rows / 24.0) ** 2)
    return (amplitude * np.exp(1j * phase)).astype(dtype)


def _shown(widget, trait: str = "phase_bytes") -> np.ndarray:
    """The float32 image the browser receives in ``trait``, shaped like the displayed phase."""
    return np.frombuffer(getattr(widget, trait), np.float32).reshape(widget.phase_height, widget.phase_width)


# --- Arrays: every computer ---
def test_a_phase_array_is_shown_as_given_with_the_session_controls_explained():
    phase = np.angle(_object_wave())
    widget = ShowPtycho(phase, sampling=0.05, units="nm")
    np.testing.assert_array_equal(_shown(widget), phase)
    assert widget.amplitude_bytes == b"" and (widget.phase_height, widget.phase_width) == (96, 80)
    assert widget.pixel_size == pytest.approx(0.5)                          # 0.05 nm drawn in Angstrom
    assert not widget.ssb_session and "needs an SSB session" in widget.session_note
    assert repr(widget) == "ShowPtycho(phase 96x80, 0.5 Å/px)"
    assert ShowPtycho(phase, sampling=(0.4, 0.5)).pixel_size == 0.5        # the column sampling sets the scale bar
    with pytest.raises(ValueError, match="export"):
        widget.export()


@pytest.mark.parametrize("installed, hint, says, never", [
    (True, INSTALL_CUDA, "quantem.gpu.SSB.open(", "pip install"),         # quantem.gpu here: open a session
    (False, INSTALL_CUDA, INSTALL_CUDA, INSTALL_MPS),                      # NVIDIA machine without it
    (False, INSTALL_MPS, INSTALL_MPS, INSTALL_CUDA),                       # Apple silicon without it
    (False, None, NO_GPU_PATH, "pip install"),                             # no GPU route on this machine
])
def test_the_session_note_names_what_this_machine_needs(monkeypatch, installed, hint, says, never):
    monkeypatch.setattr(gpu_adapter, "available", lambda: installed)
    monkeypatch.setattr(showptycho_module, "gpu_path_hint", lambda: hint, raising=False)
    note = ShowPtycho(np.angle(_object_wave())).session_note
    assert says in note and never not in note


def test_a_complex_object_wave_shows_its_phase_and_amplitude(capsys):
    wave = _object_wave(np.complex64)
    widget = ShowPtycho(wave)
    np.testing.assert_array_equal(_shown(widget), np.angle(wave))
    np.testing.assert_array_equal(_shown(widget, "amplitude_bytes"), np.abs(wave))
    assert capsys.readouterr().out == ""                                    # complex64 is drawn as it is
    wide = _object_wave(np.complex128)
    widget = ShowPtycho(wide)
    np.testing.assert_array_equal(_shown(widget), np.angle(wide).astype(np.float32))
    np.testing.assert_array_equal(_shown(widget, "amplitude_bytes"), np.abs(wide).astype(np.float32))
    printed = capsys.readouterr().out
    assert printed.startswith("ShowPtycho: complex128 input is drawn in float32; largest change: phase ")
    assert repr(widget) == "ShowPtycho(object wave 96x80, no sampling)"


@pytest.mark.parametrize("device", DEVICES)
def test_a_torch_tensor_on_any_device_shows_the_same_numbers(device):
    wave = _object_wave(np.complex64)
    widget = ShowPtycho(torch.from_numpy(wave).to(device), sampling=0.2)
    expected = torch.from_numpy(wave).to(device)
    np.testing.assert_array_equal(_shown(widget), expected.angle().cpu().numpy())
    np.testing.assert_array_equal(_shown(widget, "amplitude_bytes"), expected.abs().cpu().numpy())
    phase = torch.from_numpy(np.angle(wave)).to(device)
    np.testing.assert_array_equal(_shown(ShowPtycho(phase)), np.angle(wave))


def test_only_a_2d_array_or_an_ssb_session_opens():
    with pytest.raises(ValueError, match="one 2D reconstruction"):
        ShowPtycho(np.zeros((3, 16, 16), np.float32))
    with pytest.raises(ValueError, match="units must be one of"):
        ShowPtycho(np.zeros((16, 16), np.float32), sampling=1.0, units="inch")
    with pytest.raises(TypeError, match="2D phase or complex object-wave array"):
        ShowPtycho([[0.0, 1.0]])


# --- SSB session: quantem.gpu with CUDA or Metal ---


def _counts(scan: int = SCAN) -> np.ndarray:
    """uint16 counts of a bright-field disk shifted per scan position by a smooth phase object, plus 0-2 counts of noise."""
    rows, cols = np.indices((DET, DET))
    disk = ((rows - 7.5) ** 2 + (cols - 7.5) ** 2) <= 4.0**2
    phase_object = np.sin(np.linspace(0, 6 * np.pi, scan))[:, None] * np.cos(np.linspace(0, 4 * np.pi, scan))[None, :]
    counts = np.zeros((scan, scan, DET, DET), dtype=np.uint16)
    counts[..., disk] = 40
    shift = np.round(3 * phase_object).astype(int)
    for row in range(scan):
        for col in range(scan):
            counts[row, col] = np.roll(counts[row, col], shift[row, col], axis=1)
    return counts + np.random.default_rng(0).integers(0, 3, counts.shape, dtype=np.uint16)


def _session(scan: int = SCAN):
    """A real quantem.gpu SSB session at the START aberrations, or a skip on machines without CUDA or MPS."""
    gpu = pytest.importorskip("quantem.gpu", reason="an SSB session needs quantem.gpu: pip install quantem.widget[cuda] or [mps]")
    try:
        pytest.importorskip("quantem.gpu.device").detect()
    except RuntimeError as exc:
        pytest.skip(f"SSB needs CUDA or MPS: {exc}")
    return gpu.SSB(_counts(scan), voltage_kV=300.0, semiangle_mrad=20.0, scan_sampling_A=0.5, det_sampling=5.0, aberrations=dict(START), rotation_angle_deg=10.0)


@pytest.fixture(scope="module")
def ssb():
    """One session shared by the module: building it dominates the test time."""
    return _session()


@pytest.fixture
def widget(ssb, tmp_path):
    """A fresh widget over the shared session, which the previous test may have rotated or re-aberrated."""
    ssb.set_rotation(10.0)
    ssb.aberrations, ssb.best_loss = dict(START), float("inf")
    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"master")
    return ShowPtycho(ssb, fft_on=True, source_file=str(master), save_dir=tmp_path)


def test_opens_on_the_session_fit_and_shows_its_preview(widget, ssb):
    assert (widget.auto_c10, widget.auto_c12, widget.auto_phi12_deg) == (3.0, 1.0, pytest.approx(math.degrees(0.2)))
    assert widget.rotation_deg == widget.auto_rotation_deg == 10.0
    assert (widget.pixel_size, widget.scan_rows, widget.scan_cols) == (0.5, SCAN, SCAN)
    assert widget.total_bf == widget.drag_bf == ssb.num_bf
    assert widget.initial_fft_on and widget.crop_refit_available and widget.trials_json == ""
    # Python reconstructs every live frame: no browser SSB runs, so there is no browser status to show
    assert widget.webgpu_preview_status == "" and not widget.webgpu_preview_enabled
    expected, loss = ssb.preview({"C10": 3.0, "C12": 1.0, "phi12": 0.2}, compute_loss=True)
    np.testing.assert_array_equal(_shown(widget), expected)
    result = json.loads(widget.result_json)
    assert result == {"id": 0, "C10": 3.0, "upsample": 1, "C12": 1.0, "phi12_deg": pytest.approx(math.degrees(0.2)), "loss": pytest.approx(loss)}
    assert repr(widget) == f"ShowPtycho(0 pinned, drag_bf={ssb.num_bf})"


def test_requests_flip_and_rotation_follow_the_session(widget, ssb):
    widget.request_json = json.dumps({"id": 7, "c10": 2.0, "c12": 0.5, "phi12_deg": 30.0, "committed": True})
    expected, loss = ssb.preview({"C10": 2.0, "C12": 0.5, "phi12": math.radians(30.0)}, compute_loss=True)
    np.testing.assert_array_equal(_shown(widget), expected)
    assert json.loads(widget.result_json)["loss"] == pytest.approx(loss)
    widget.request_json = json.dumps({"id": 8, "c10": 2.5, "c12": 0.5, "phi12_deg": 30.0, "committed": False})
    assert json.loads(widget.result_json)["loss"] is None                     # drags skip the variance pass
    widget.flip_phase = True
    np.testing.assert_array_equal(_shown(widget), -ssb.preview({"C10": 2.5, "C12": 0.5, "phi12": math.radians(30.0)}, compute_loss=False)[0])
    widget.rotation_deg = 25.0
    assert ssb.rotation_angle_deg == 25.0
    expected, _ = ssb.preview({"C10": 2.5, "C12": 0.5, "phi12": math.radians(30.0)}, compute_loss=True)
    np.testing.assert_array_equal(_shown(widget), -expected)
    assert json.loads(widget.result_json)["C10"] == 2.5                    # the rotation keeps the displayed aberrations


def test_higher_order_panel_switches_to_the_full_kernel():
    """The CUDA higher-order loss kernel starts at 256 x 256 scans."""
    ssb = _session(256)
    widget = ShowPtycho(ssb)
    widget.higher_order_json = json.dumps({"C21_mag": 5.0, "C21_angle": 45.0, "C30": 2.0})
    mags = np.zeros(14, dtype=np.float32)
    angles = np.zeros(14, dtype=np.float32)
    mags[0], mags[1], angles[1] = 3.0, 1.0, 0.2
    mags[2], angles[2], mags[4] = 5.0, math.radians(45.0), 2.0
    expected, _ = ssb.preview({"C10": 3.0, "C12": 1.0, "phi12": 0.2}, compute_loss=True, higher_order_magnitudes=mags, higher_order_angles=angles)
    np.testing.assert_array_equal(_shown(widget), expected)
    with pytest.raises(ValueError, match="upsample=1"):
        widget.upsample = 2
    widget.higher_order_json = "{}"
    if ssb.backend == "cuda":
        widget.upsample = 2
        assert _shown(widget).shape == (512, 512)
        with pytest.raises(ValueError, match="Set upsample=1"):
            widget.higher_order_json = json.dumps({"C30": 1.0})
    with pytest.raises(ValueError, match="1, 2, 4, or 8"):
        widget.upsample = 3


def test_sample_tilt_panel_uses_the_thick_sample_model(widget, ssb):
    assert widget.sample_available
    widget.sample_json = json.dumps({"tilt_row_mrad": 3.0, "tilt_col_mrad": -4.0, "thickness_nm": 12.0})
    expected, _ = ssb.preview({"C10": 3.0, "C12": 1.0, "phi12": 0.2}, compute_loss=True, tilt_mrad=(3.0, -4.0), depth_spread_nm=12.0)
    np.testing.assert_array_equal(_shown(widget), expected)
    widget.save_trigger += 1
    saved = json.loads(Path(widget.calibration_path).read_text())
    assert saved["tilt_mrad"] == [3.0, -4.0] and saved["depth_spread_nm"] == 12.0
    rot = math.radians(10.0)
    assert saved["tilt_object_mrad"] == pytest.approx([3.0 * math.cos(rot) + (-4.0) * math.sin(rot), -3.0 * math.sin(rot) + (-4.0) * math.cos(rot)])


def test_save_writes_the_calibration_and_mirrors_it_onto_the_session(widget, ssb, tmp_path):
    widget.request_json = json.dumps({"id": 1, "c10": -4.0, "c12": 0.25, "phi12_deg": 12.0, "committed": True})
    widget.notes = "lamella A"
    widget.save_trigger += 1
    assert widget.calibration_path == str(tmp_path / "calibration.json") and widget.calibration_saved_at
    saved = json.loads((tmp_path / "calibration.json").read_text())
    assert saved["aberration_unit"] == "nm" and saved["schema_version"] == 3
    assert saved["aberrations"] == {"C10": -4.0, "C12": 0.25, "phi12": pytest.approx(math.radians(12.0))}
    assert (saved["rotation_angle_deg"], saved["flip_phase"], saved["notes"], saved["scan_region"]) == (10.0, False, "lamella A", [0, SCAN, 0, SCAN])
    assert (saved["voltage_kV"], saved["semiangle_mrad"], saved["scan_sampling_A"]) == (300.0, 20.0, 0.5)
    assert saved["source_file"] == str(tmp_path / "scan_master.h5") and saved["loss"] == json.loads(widget.result_json)["loss"]
    assert saved["tilt_mrad"] is None and saved["timestamp"] == widget.calibration_saved_at
    assert ssb.aberrations == {"C10": -4.0, "C12": 0.25, "phi12": pytest.approx(math.radians(12.0))} and ssb.best_loss == saved["loss"]


def test_pins_star_view_and_unpin(widget, tmp_path):
    first = _shown(widget).copy()
    widget.pin_json = json.dumps({"action": "pin", "id": "a", "C10": 3.0, "C12": 1.0, "phi12_deg": 11.0, "loss": 0.1})
    widget.request_json = json.dumps({"id": 2, "c10": 0.0, "c12": 0.0, "phi12_deg": 0.0, "committed": True})
    widget.pin_json = json.dumps({"action": "pin", "id": "b", "C10": 0.0, "C12": 0.0, "phi12_deg": 0.0, "loss": 0.2})
    assert repr(widget).startswith("ShowPtycho(2 pinned")
    widget.pin_json = json.dumps({"action": "star", "id": "a"})
    stars = json.loads((tmp_path / "showptycho_stars.json").read_text())
    assert [s["id"] for s in stars] == ["a"]
    assert stars[0]["aberrations"] == {"C10": 3.0, "C12": 1.0, "phi12": pytest.approx(math.radians(11.0))}
    assert (stars[0]["rotation_angle_deg"], stars[0]["loss"], stars[0]["aberration_unit"], stars[0]["source_file"]) == (10.0, 0.1, "nm", str(tmp_path / "scan_master.h5"))
    widget.pin_json = json.dumps({"action": "view", "id": "a"})
    np.testing.assert_array_equal(_shown(widget), first)
    assert json.loads(widget.result_json)["C10"] == 3.0
    widget.pin_json = json.dumps({"action": "unpin", "id": "a"})
    assert json.loads((tmp_path / "showptycho_stars.json").read_text()) == []
    assert repr(widget).startswith("ShowPtycho(1 pinned")


def test_export_writes_the_browser_folder_from_exact_bf_counts(widget, ssb, tmp_path):
    out = widget.export(tmp_path / "review", title="my sample SSB")
    assert {p.name for p in out.iterdir()} == {"index.html", "source", "snapshots", ".viewer", "ShowPtycho.command"}
    suffix = "qem" if ssb.backend == "cuda" else "u16"
    assert sorted(p.name for p in (out / "source").iterdir()) == [f"bf_columns.{suffix}"]
    cal = json.loads((out / "snapshots" / "cal.json").read_text())
    assert cal["kind"] == "showptycho_webgpu_folder" and cal["source_file"] == "redacted_local_source"
    assert (cal["num_bf"], cal["g_shape"], cal["phase_shape"], cal["detector_shape"]) == (ssb.num_bf, [ssb.num_bf, SCAN, SCAN], [SCAN, SCAN], [DET, DET])
    assert cal["aberrations"] == {"C10": 3.0, "C12": 1.0, "phi12": pytest.approx(0.2), "phi12_deg": pytest.approx(math.degrees(0.2))}
    assert (cal["rotation_angle_deg"], cal["scan_sampling_A"], cal["voltage_kV"], cal["gpu_memory_gb"]) == (10.0, 0.5, 300.0, 4.5)
    assert cal["bf_column_companion_path"] == f"source/bf_columns.{suffix}" and cal["persistent_bf_cache"] is False
    manifest = json.loads((out / "snapshots" / "manifest.json").read_text())
    assert manifest["title"] == "my sample SSB" and manifest["source"]["bf_columns"]["scan_shape"] == [SCAN, SCAN]
    assert json.loads((out / "snapshots" / "snapshots.json").read_text()) == []
    html = (out / "index.html").read_text()
    embedded = json.loads(re.search(r'<script type="application/vnd.jupyter.widget-state\+json">(.*?)</script>', html, re.S).group(1))
    exported = next(model["state"] for model in embedded["state"].values() if "webgpu_standalone" in model["state"])
    assert exported["webgpu_standalone"] and exported["webgpu_preview_enabled"] and not exported["crop_refit_available"]
    assert exported["webgpu_preview_status"].startswith("WebGPU folder ready") and widget.webgpu_preview_status == ""
    assert json.loads(exported["webgpu_h5_source_json"])["kind"] == "bf_columns" and exported["stars_path"] == "snapshots/snapshots.json"
    assert str(tmp_path) not in html and "scan_master.h5" not in html
    assert widget.crop_refit_available and not widget.webgpu_standalone and len(widget.phase_bytes) == 4 * SCAN * SCAN
    counts_file = (out / "source" / f"bf_columns.{suffix}").read_bytes()
    widget.export(out)                                                     # a re-export reuses its own counts
    assert (out / "source" / f"bf_columns.{suffix}").read_bytes() == counts_file


def test_crop_refit_validates_the_region_and_a_session_keeps_its_own_sampling(widget, ssb):
    widget.crop_refit_request_json = json.dumps({"scan_region": [0, 16, 0, 16], "n_trials": 1})
    assert widget.crop_refit_status.startswith("Crop refit failed: SSB refit crops must be at least 32x32")
    widget.crop_refit_request_json = json.dumps({"scan_region": [0, 64, 100, 164], "n_trials": 1})
    assert "outside 128x128 data" in widget.crop_refit_status
    assert widget.ssb_session and widget.session_note == "" and widget.amplitude_bytes == b""
    with pytest.raises(ValueError, match="scan_sampling_A"):
        ShowPtycho(ssb, sampling=1.0)
