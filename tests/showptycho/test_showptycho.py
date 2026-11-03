"""ShowPtycho: a reconstruction array on any computer, and a real quantem.gpu SSB session.

Arrays (NumPy or torch on any device, real phase or complex object wave) must show exactly the
values given, to float32, with no quantem.gpu. For the session, a synthetic 128 x 128 scan of a
bright-field disk that shifts with a smooth phase object is fitted nowhere; the session carries
starting aberrations and the widget must show exactly the phase ``SSB.preview`` returns for them,
follow slider requests, write the calibration and star files, and export the browser folder with
exact bright-field counts.
"""

import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from quantem.widget import ShowPtycho
from quantem.widget.adapters import gpu as gpu_adapter
from quantem.widget.device import INSTALL_CUDA, INSTALL_MPS, NO_GPU_PATH
from quantem.widget.showptycho import export as showptycho_export
from quantem.widget.showptycho import widget as showptycho_module

SCAN, DET = 128, 16
START = {"C10_nm": 3.0, "C12_nm": 1.0, "phi12_deg": math.degrees(0.2)}
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
    assert widget.object_sampling_A == pytest.approx([0.5, 0.5])            # 0.05 nm drawn in Angstrom
    assert not widget.ssb_session and "needs an SSB session" in widget.session_note
    assert repr(widget) == "ShowPtycho(phase 96x80, 0.5 Å/px)"
    assert ShowPtycho(phase, sampling=(0.4, 0.5)).object_sampling_A == [0.4, 0.5]  # [row, col]; the column step sets the scale bar
    with pytest.raises(ValueError, match="export"):
        widget.export()


@pytest.mark.parametrize("installed, hint, says, never", [
    # quantem.gpu here: open a session (the note inspects quantem.gpu's session type, so the package must be present)
    pytest.param(True, INSTALL_CUDA, "quantem.gpu.SSB(data)", "pip install",
                 marks=pytest.mark.skipif(importlib.util.find_spec("quantem.gpu") is None, reason="needs quantem.gpu installed")),
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


def test_any_2d_array_like_opens_and_other_shapes_are_refused():
    with pytest.raises(ValueError, match="one 2D reconstruction"):
        ShowPtycho(np.zeros((3, 16, 16), np.float32))
    with pytest.raises(ValueError, match="Unknown unit 'inch'"):
        ShowPtycho(np.zeros((16, 16), np.float32), sampling=1.0, units="inch")
    # C-1: a list (or a CuPy array) raised TypeError although the widgets accept any array-like
    widget = ShowPtycho([[0.0, 1.0], [2.0, 3.0]])
    np.testing.assert_array_equal(np.frombuffer(widget.phase_bytes, np.float32), [0.0, 1.0, 2.0, 3.0])
    cupy = pytest.importorskip("cupy")
    phase = np.linspace(-1, 1, 64, dtype=np.float32).reshape(8, 8)
    np.testing.assert_array_equal(np.frombuffer(ShowPtycho(cupy.asarray(phase)).phase_bytes, np.float32).reshape(8, 8), phase)


# --- SSB session: quantem.gpu with CUDA or Metal ---


def _counts(scan: int = SCAN, scan_cols: int | None = None) -> np.ndarray:
    """uint16 counts of a bright-field disk shifted per scan position by a smooth phase object, plus 0-2 counts of noise.

    ``scan`` rows by ``scan_cols`` columns (square when omitted).
    """
    scan_cols = scan if scan_cols is None else scan_cols
    rows, cols = np.indices((DET, DET))
    disk = ((rows - 7.5) ** 2 + (cols - 7.5) ** 2) <= 4.0**2
    phase_object = np.sin(np.linspace(0, 6 * np.pi, scan))[:, None] * np.cos(np.linspace(0, 4 * np.pi, scan_cols))[None, :]
    counts = np.zeros((scan, scan_cols, DET, DET), dtype=np.uint16)
    counts[..., disk] = 40
    shift = np.round(3 * phase_object).astype(int)
    for row in range(scan):
        for col in range(scan_cols):
            counts[row, col] = np.roll(counts[row, col], shift[row, col], axis=1)
    return counts + np.random.default_rng(0).integers(0, 3, counts.shape, dtype=np.uint16)


def _session(scan: int = SCAN):
    """A real quantem.gpu SSB session at the START aberrations, or a skip on machines without CUDA or MPS."""
    gpu = pytest.importorskip("quantem.gpu", reason="an SSB session needs quantem.gpu: pip install quantem.widget[cuda] or [mps]")
    try:
        pytest.importorskip("quantem.gpu.device").detect()
    except RuntimeError as exc:
        pytest.skip(f"SSB needs CUDA or MPS: {exc}")
    return gpu.SSB(_counts(scan), voltage_kV=300.0, semiangle_mrad=20.0, scan_sampling_A=(0.5, 0.5), detector_sampling_mrad=(5.0, 5.0),
                   aberrations=dict(START), scan_detector_rotation_deg=10.0)


@pytest.fixture(scope="module")
def ssb():
    """One session shared by the module: building it dominates the test time."""
    return _session()


def test_small_scan_calibration_and_region_bounds_exclude_fft_padding(tmp_path, monkeypatch):
    """A 96 x 80 scan is centred in a 128 x 128 FFT grid on CUDA and MPS alike.

    Region images, regional previews, restored regions and the calibration phase
    all show measured positions: the native bounds are shifted by the padding.
    """
    gpu = pytest.importorskip("quantem.gpu")
    if not gpu_adapter.accelerator_ready():
        pytest.skip("Needs CUDA or MPS")
    source = tmp_path / "scan.npy"
    np.save(source, _counts(96, 80))
    options = dict(voltage_kV=300, semiangle_mrad=20, scan_sampling_A=(0.5, 0.5), detector_sampling_mrad=(5, 5))
    top, left = (128 - 96) // 2, (128 - 80) // 2

    def measured(phase, bounds, factor):
        """Native ``bounds`` of a padded full-scan phase."""
        phase = phase.get() if hasattr(phase, "get") else np.asarray(phase)
        r0, r1, c0, c1 = bounds
        return phase[(r0 + top) * factor:(r1 + top) * factor, (c0 + left) * factor:(c1 + left) * factor]

    with gpu.SSB.open(str(source), aberrations=START, **options) as session:
        viewer = ShowPtycho(session, save_dir=tmp_path, local_tilt=True)
        factor = viewer.auto_upsampling_factor
        assert session.input_scan_shape == (96, 80) and session.scan_shape == (128, 128)
        assert viewer.regions[0]["bounds"] == [0, 96, 0, 80]
        record = viewer.calibration(include_phase=True)
        assert record["scan_shape"] == [96, 80] and record["scan_region"] == [0, 96, 0, 80]
        full, _ = session.preview(dict(START), upsampling_factor=factor,
                                  phase_estimator="phase_of_mean", compute_loss=False)
        np.testing.assert_array_equal(record["ssb_phase"], measured(full, (0, 96, 0, 80), factor))

        bounds = [8, 72, 4, 68]
        viewer.regions = [{"id": "region-1", "label": "Region 1", "bounds": bounds}]
        fit, = viewer.fit_regions()
        assert fit["reconstruction"]["scan_offset"] == [top, left]
        model = dict(sample_tilt_mrad=fit["sample_tilt_mrad"], depth_spread_nm=fit["depth_spread_nm"],
                     upsampling_factor=factor, phase_estimator="phase_of_mean", compute_loss=False)
        phase, _ = session.preview(fit["aberrations"], **model)
        np.testing.assert_array_equal(viewer.region_images["Region 1 · SSB"], measured(phase, bounds, factor))

        adjusted = {**fit["aberrations"], "C10_nm": fit["aberrations"]["C10_nm"] + 1}
        viewer.request_json = json.dumps({
            "id": 200, "region_id": "region-1", "region_bounds": bounds, **adjusted,
            "sample": {"sample_tilt_mrad": fit["sample_tilt_mrad"], "depth_spread_nm": fit["depth_spread_nm"]},
            "committed": False,
        })
        assert "error" not in json.loads(viewer.local_tilt_preview_json)
        phase, _ = session.preview(adjusted, **model)
        expected = measured(phase, bounds, factor)
        np.testing.assert_array_equal(viewer.region_images["Region 1 · SSB"], expected)
        viewer.save_trigger += 1
        viewer.close()

    def no_refit(*args, **kwargs):
        raise AssertionError("Saved regions should reconstruct without another search.")
    monkeypatch.setattr(gpu.SSB, "find_aberrations", no_refit)
    with gpu.SSB.open(str(source), **options) as reopened:
        viewer = ShowPtycho(reopened, save_dir=tmp_path, local_tilt=True)
        np.testing.assert_array_equal(viewer.region_images["Region 1 · SSB"], expected)
        viewer.close()


@pytest.fixture
def widget(ssb, tmp_path):
    """A fresh widget over the shared session, which the previous test may have rotated or re-aberrated."""
    ssb.set_rotation(10.0)
    ssb.aberrations, ssb.best_loss = dict(START), float("inf")
    ssb.sample_tilt_mrad, ssb.depth_spread_nm = None, None
    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"master")
    ssb.source_path = str(master)  # the session names its file; the widget reads it from there
    view = ShowPtycho(ssb, show_fft=True, save_dir=tmp_path)
    view.upsampling_factor = 1  # Explicit native-grid comparisons below.
    return view


def test_opens_on_the_session_fit_and_shows_its_preview(widget, ssb):
    widget.upsampling_factor = "auto"
    widget.request_json = json.dumps({"id": 0, **START, "committed": True})
    assert (widget.fitted_C10_nm, widget.fitted_C12_nm, widget.fitted_phi12_deg) == (3.0, 1.0, pytest.approx(math.degrees(0.2)))
    assert widget.scan_detector_rotation_deg == widget.fitted_scan_detector_rotation_deg == 10.0
    assert (widget.scan_sampling_A, widget.object_sampling_A, widget.scan_rows, widget.scan_cols) == ([0.5, 0.5], [], SCAN, SCAN)
    assert widget.num_bright_field_pixels == widget.num_drag_bright_field_pixels == ssb.num_bright_field_pixels
    assert widget.initial_fft_on and widget.crop_refit_available and widget.trials_json == ""
    # Python reconstructs every live frame: no browser SSB runs, so there is no browser status to show
    assert widget.webgpu_preview_status == "" and not widget.webgpu_preview_enabled
    expected, loss = ssb.preview(dict(START), upsampling_factor=widget.auto_upsampling_factor, compute_loss=True)
    np.testing.assert_array_equal(_shown(widget), expected)
    result = json.loads(widget.result_json)
    assert result == {"id": 0, "C10_nm": 3.0, "upsampling_factor": 3, "C12_nm": 1.0, "phi12_deg": pytest.approx(math.degrees(0.2)), "tilt_correction": True, "loss": pytest.approx(loss)}
    assert repr(widget) == f"ShowPtycho(0 pinned, {ssb.num_bright_field_pixels} bright-field pixels)"


def test_native_regions_fit_joint_models_without_replacing_the_overview(tmp_path, monkeypatch):
    """Move a rectangle, refit, and compare its view with full-scan SSB."""
    gpu = pytest.importorskip("quantem.gpu")
    if not gpu_adapter.accelerator_ready():
        pytest.skip("Needs CUDA or MPS")
    source = tmp_path / "scan.npy"
    np.save(source, _counts())
    with gpu.SSB.open(
        str(source), voltage_kV=300, semiangle_mrad=20,
        scan_sampling_A=(0.5, 0.5), detector_sampling_mrad=(5, 5),
        aberrations=START,
    ) as ssb:
        viewer = ShowPtycho(ssb, local_tilt=True, save_dir=tmp_path)
        overview = _shown(viewer).copy()
        reference = viewer.ssb_parameters
        progress = []
        viewer.observe(lambda change: progress.append(json.loads(change["new"])),
                       names="local_tilt_reply_json")
        for bounds in ([0, 64, 0, 64], [16, 96, 8, 80]):
            progress.clear()
            viewer.regions = [{"id": "region-1", "label": "Region 1", "bounds": bounds}]
            fit, = viewer.fit_regions()
            assert viewer.ssb_parameters == reference
            np.testing.assert_array_equal(_shown(viewer), overview)
            assert len(fit["fit_record"]["fits"]) == 5
            assert all(run["num_completed_trials"] == 200 and run["refinement"] == "newton"
                       for run in fit["fit_record"]["fits"])
            assert [event["repeat"] for event in progress if event.get("stage") == "joint_fit"] == [1, 2, 3, 4, 5]
            assert progress[-1]["status"] == "complete"
            assert all(set(run["aberrations"]) == set(START) for run in fit["fit_record"]["fits"])
            phase, _ = ssb.preview(
                fit["aberrations"], sample_tilt_mrad=fit["sample_tilt_mrad"],
                depth_spread_nm=fit["depth_spread_nm"],
                upsampling_factor=viewer.auto_upsampling_factor,
                phase_estimator="phase_of_mean", compute_loss=False,
            )
            if hasattr(phase, "get"):
                phase = phase.get()
            row_start, row_stop, col_start, col_stop = (
                value * viewer.auto_upsampling_factor for value in bounds)
            np.testing.assert_array_equal(
                viewer.region_images["Region 1 · SSB"],
                phase[row_start:row_stop, col_start:col_stop],
            )
        viewer.save_trigger += 1
        saved = json.loads(Path(viewer.calibration_path).read_text())
        assert saved["local_tilt"]["regions"] == viewer.regions
        assert saved["local_tilt"]["fits"] == viewer.region_fits
        reference_sample = viewer.sample_json
        before = viewer.region_images["Region 1 · SSB"].copy()
        viewer.request_json = json.dumps({
            "id": 100, "region_id": "region-1", "region_bounds": bounds,
            **fit["aberrations"], "C10_nm": fit["aberrations"]["C10_nm"] + 1,
            "sample": {"sample_tilt_mrad": fit["sample_tilt_mrad"],
                       "depth_spread_nm": fit["depth_spread_nm"]}, "committed": True,
        })
        assert "error" not in json.loads(viewer.local_tilt_preview_json)
        assert viewer.ssb_parameters == reference and viewer.sample_json == reference_sample
        np.testing.assert_array_equal(_shown(viewer), overview)
        assert not np.array_equal(before, viewer.region_images["Region 1 · SSB"])
        assert viewer.region_fits[0]["adjusted"]
        assert viewer.region_fits[0]["fitted_parameters"]["aberrations"] == fit["aberrations"]
        # Auto preserves an unchanged region, including its deliberate manual correction.
        viewer.local_tilt_request_json = json.dumps({
            "id": 101, "action": "fit", "regions": viewer.regions, "only_changed": True})
        assert json.loads(viewer.local_tilt_json)["fitted_region_count"] == 0
        assert viewer.region_fits[0]["adjusted"]
        # Fitting a later region preserves the comparison order and earlier correction.
        first_fit = viewer.region_fits[0]
        first_image = viewer.region_images["Region 1 · SSB"].copy()
        viewer.regions = [*viewer.regions,
                          {"id": "region-2", "label": "Region 2", "bounds": [0, 64, 0, 64]}]
        viewer.local_tilt_request_json = json.dumps({
            "id": 102, "action": "fit", "regions": viewer.regions, "only_changed": True})
        assert list(viewer.region_images) == ["Region 1 · SSB", "Region 2 · SSB"]
        assert viewer.region_fits[0] == first_fit
        np.testing.assert_array_equal(viewer.region_images["Region 1 · SSB"], first_image)
        # Each selected region keeps its own optics and pixels when the other changes.
        for index in (0, 1):
            other = 1 - index
            other_fit = viewer.region_fits[other]
            other_image = viewer.region_images[f"{other_fit['label']} · SSB"]
            selected = viewer.region_fits[index]
            viewer.request_json = json.dumps({
                "id": 103 + index, "region_id": selected["id"],
                "region_bounds": selected["bounds"], **selected["aberrations"],
                "C10_nm": selected["aberrations"]["C10_nm"] + 1,
                "sample": {"sample_tilt_mrad": selected["sample_tilt_mrad"],
                           "depth_spread_nm": selected["depth_spread_nm"]},
                "committed": True,
            })
            assert "error" not in json.loads(viewer.local_tilt_preview_json)
            assert viewer.region_fits[other] == other_fit
            np.testing.assert_array_equal(
                viewer.region_images[f"{other_fit['label']} · SSB"], other_image)
            np.testing.assert_array_equal(_shown(viewer), overview)
        second_fit = viewer.region_fits[1]
        second_image = viewer.region_images["Region 2 · SSB"]
        viewer.local_tilt_request_json = json.dumps({
            "id": 105, "action": "fit", "regions": viewer.regions,
            "only_changed": True, "force_region": "region-1"})
        assert json.loads(viewer.local_tilt_json)["fitted_region_count"] == 1
        assert viewer.region_fits[1] == second_fit
        np.testing.assert_array_equal(viewer.region_images["Region 2 · SSB"], second_image)
        expected_fits, expected_images = viewer.region_fits, viewer.region_images
        viewer.save_trigger += 1
        viewer.close()
    def no_refit(*args, **kwargs):
        raise AssertionError("Saved regions should reconstruct without another search.")
    monkeypatch.setattr(gpu.SSB, "find_aberrations", no_refit)
    with gpu.SSB.open(str(source), voltage_kV=300, semiangle_mrad=20,
                      scan_sampling_A=(0.5, 0.5), detector_sampling_mrad=(5, 5)) as reopened:
        viewer = ShowPtycho(reopened, local_tilt=True, save_dir=tmp_path)
        assert viewer.region_fits == expected_fits
        for label, expected in expected_images.items():
            np.testing.assert_array_equal(viewer.region_images[label], expected)
        np.testing.assert_array_equal(_shown(viewer), overview)
        viewer.regions = viewer.regions[1:]
        assert viewer.region_fits == expected_fits[1:]
        np.testing.assert_array_equal(viewer.region_images["Region 2 · SSB"], expected_images["Region 2 · SSB"])
        viewer.regions = []
        viewer.save_trigger += 1
        viewer.close()
    with gpu.SSB.open(str(source), voltage_kV=300, semiangle_mrad=20,
                      scan_sampling_A=(0.5, 0.5), detector_sampling_mrad=(5, 5)) as reopened:
        viewer = ShowPtycho(reopened, local_tilt=True, save_dir=tmp_path)
        assert viewer.regions == [] and viewer.region_images == {}
        np.testing.assert_array_equal(_shown(viewer), overview)
        viewer.close()


def test_requests_flip_and_rotation_follow_the_session(widget, ssb):
    widget.request_json = json.dumps({"id": 7, "C10_nm": 2.0, "C12_nm": 0.5, "phi12_deg": 30.0, "committed": True})
    expected, loss = ssb.preview({"C10_nm": 2.0, "C12_nm": 0.5, "phi12_deg": 30.0}, compute_loss=True)
    np.testing.assert_array_equal(_shown(widget), expected)
    assert json.loads(widget.result_json)["loss"] == pytest.approx(loss)
    widget.request_json = json.dumps({"id": 8, "C10_nm": 2.5, "C12_nm": 0.5, "phi12_deg": 30.0, "committed": False})
    assert json.loads(widget.result_json)["loss"] is None                     # drags skip the variance pass
    widget.display_negate_phase = True
    np.testing.assert_array_equal(_shown(widget), -ssb.preview({"C10_nm": 2.5, "C12_nm": 0.5, "phi12_deg": 30.0}, compute_loss=False)[0])
    widget.scan_detector_rotation_deg = 25.0
    assert ssb.scan_detector_rotation_deg == 25.0
    expected, _ = ssb.preview({"C10_nm": 2.5, "C12_nm": 0.5, "phi12_deg": 30.0}, compute_loss=True)
    np.testing.assert_array_equal(_shown(widget), -expected)
    assert json.loads(widget.result_json)["C10_nm"] == 2.5                 # the rotation keeps the displayed aberrations
    assert json.loads(widget.result_json)["loss"] is not None              # a rotation set from Python computes the loss
    widget.set_state({"scan_detector_rotation_deg": 26.0})                 # a slider drag from the browser
    assert ssb.scan_detector_rotation_deg == 26.0
    expected, _ = ssb.preview({"C10_nm": 2.5, "C12_nm": 0.5, "phi12_deg": 30.0}, compute_loss=False)
    np.testing.assert_array_equal(_shown(widget), -expected)
    assert json.loads(widget.result_json)["loss"] is None                     # drags skip the variance pass
    widget.request_json = json.dumps({"id": 10, "C10_nm": 2.5, "C12_nm": 0.5, "phi12_deg": 30.0, "committed": True})
    assert json.loads(widget.result_json)["loss"] is not None              # the release computes it
    # the request keys of the earlier release are not read under a second spelling
    with pytest.raises(KeyError, match="C10_nm"):
        widget.request_json = json.dumps({"id": 9, "c10": 2.0, "c12": 0.5, "phi12_deg": 30.0, "committed": True})


def test_higher_order_panel_switches_to_the_full_kernel():
    """The CUDA higher-order loss kernel starts at 256 x 256 scans."""
    ssb = _session(256)
    widget = ShowPtycho(ssb)
    widget.upsampling_factor = 1
    widget.higher_order_aberrations_json = json.dumps({"C21_nm": 5.0, "phi21_deg": 45.0, "C30_nm": 2.0})
    expected, _ = ssb.preview({**START, "C21_nm": 5.0, "phi21_deg": 45.0, "C30_nm": 2.0}, compute_loss=True)
    np.testing.assert_array_equal(_shown(widget), expected)
    widget.upsampling_factor = "auto"
    expected, _ = ssb.preview(
        {**START, "C21_nm": 5.0, "phi21_deg": 45.0, "C30_nm": 2.0},
        upsampling_factor=widget.auto_upsampling_factor,
    )
    np.testing.assert_array_equal(_shown(widget), expected)
    widget.upsampling_factor = 1
    widget.higher_order_aberrations_json = "{}"
    # a panel left at zero keeps the two-term phase: a named term, even at zero, makes quantem.gpu hold C10, C12 and
    # phi12 in float32, which changed the last bits of most pixels for coefficients float32 cannot hold exactly
    widget.higher_order_aberrations_json = json.dumps({"C21_nm": 0.0, "phi21_deg": 45.0})
    displayed = {"C10_nm": -26.259025251007465, "C12_nm": 6.41328169139375, "phi12_deg": 63.473910926518215}
    widget.request_json = json.dumps({"id": 3, **displayed, "committed": True})
    np.testing.assert_array_equal(_shown(widget), ssb.preview(displayed, compute_loss=True)[0])
    if ssb.backend == "cuda":
        widget.upsampling_factor = "auto"
        assert _shown(widget).shape == (256 * widget.auto_upsampling_factor,) * 2
        widget.higher_order_aberrations_json = json.dumps({"C30_nm": 1.0})
        expected, _ = ssb.preview(
            {**displayed, "C30_nm": 1.0}, upsampling_factor=widget.auto_upsampling_factor,
        )
        np.testing.assert_array_equal(_shown(widget), expected)
    with pytest.raises(ValueError, match="1, 2, 3, 4, or 8"):
        widget.upsampling_factor = 5


def test_higher_order_tilt_uses_automatic_sampling_and_saves_the_same_model(tmp_path):
    """Tune higher orders without changing sampling or losing the sample correction."""
    with _session(256) as ssb:
        widget = ShowPtycho(ssb, save_dir=tmp_path)
        sample = {"sample_tilt_mrad": [3.0, -4.0], "depth_spread_nm": 12.0}
        higher = {"C21_nm": 40.0, "phi21_deg": 45.0, "C30_nm": 1500.0}
        widget.sample_json = json.dumps(sample)
        baseline = _shown(widget).copy()
        widget.higher_order_aberrations_json = json.dumps(higher)
        expected, _ = ssb.preview(
            {**START, **higher}, upsampling_factor=widget.auto_upsampling_factor,
            phase_estimator="phase_of_mean", compute_loss=True, **sample,
        )
        np.testing.assert_array_equal(_shown(widget), expected)
        assert not np.array_equal(baseline, expected)
        assert widget.upsampling_factor == "auto"
        assert expected.shape == (768, 768)
        widget.save_trigger += 1
        saved = json.loads(Path(widget.calibration_path).read_text())
        assert saved["phase_estimator"] == "phase_of_mean"
        assert saved["upsampling_factor"] == 3
        assert saved["upsampling_mode"] == "auto"
        assert saved["sample_tilt_mrad"] == sample["sample_tilt_mrad"]
        assert saved["aberrations"]["C30_nm"] == higher["C30_nm"]
        widget.higher_order_aberrations_json = "{}"
        np.testing.assert_array_equal(_shown(widget), baseline)
        # A zero magnitude with a nonzero angle retains the primary correction.
        widget.higher_order_aberrations_json = json.dumps({"C21_nm": 0.0, "phi21_deg": 45.0})
        np.testing.assert_array_equal(_shown(widget), baseline)
        widget.close()


def test_sample_tilt_panel_uses_the_thick_sample_model(widget, ssb):
    assert widget.sample_available
    widget.sample_json = json.dumps({"sample_tilt_mrad": [3.0, -4.0], "depth_spread_nm": 12.0})
    expected, _ = ssb.preview(dict(START), compute_loss=True, sample_tilt_mrad=(3.0, -4.0), depth_spread_nm=12.0)
    np.testing.assert_array_equal(_shown(widget), expected)
    widget.save_trigger += 1
    saved = json.loads(Path(widget.calibration_path).read_text())
    # the panel's depth is the model's depth spread: it is stored under that name and never as a thickness
    assert saved["sample_tilt_mrad"] == [3.0, -4.0] and saved["depth_spread_nm"] == 12.0 and "thickness_nm" not in saved
    rot = math.radians(10.0)
    assert saved["propagator_tilt_mrad"] == pytest.approx([3.0 * math.cos(rot) + (-4.0) * math.sin(rot), -3.0 * math.sin(rot) + (-4.0) * math.cos(rot)])


def test_joint_fit_reference_survives_manual_optics_and_sample_adjustments(widget):
    """Return from manual tuning to the complete fitted model, without zeroing the sample."""
    widget.sample_fit_request += 1
    fitted = json.loads(widget.sample_fit_json)
    assert widget.fitted_loss == pytest.approx(widget._ssb.state()["loss"])
    reference = {name: fitted[name] for name in START}
    sample = {name: fitted[name] for name in ("sample_tilt_mrad", "depth_spread_nm")}
    assert json.loads(widget.fitted_sample_json) == sample
    assert (widget.fitted_C10_nm, widget.fitted_C12_nm, widget.fitted_phi12_deg) == (
        reference["C10_nm"], reference["C12_nm"], reference["phi12_deg"])
    widget.request_json = json.dumps({"id": 20, **reference, "sample": sample, "committed": True})
    fitted_phase = _shown(widget).copy()
    widget.request_json = json.dumps({
        "id": 21, **reference, "C10_nm": reference["C10_nm"] + 2,
        "sample": {"sample_tilt_mrad": [5, -3], "depth_spread_nm": 9}, "committed": True,
    })
    assert json.loads(widget.fitted_sample_json) == sample
    assert not np.array_equal(_shown(widget), fitted_phase)
    widget.request_json = json.dumps({"id": 22, **reference, "sample": sample, "committed": True})
    np.testing.assert_array_equal(_shown(widget), fitted_phase)


def test_save_writes_the_calibration_and_mirrors_it_onto_the_session(widget, ssb, tmp_path):
    widget.request_json = json.dumps({"id": 1, "C10_nm": -4.0, "C12_nm": 0.25, "phi12_deg": 12.0, "committed": True})
    widget.notes = "lamella A"
    widget.save_trigger += 1
    assert widget.calibration_path == str(tmp_path / "calibration.json") and widget.calibration_saved_at
    saved = json.loads((tmp_path / "calibration.json").read_text())
    assert saved["schema"] == "quantem.ssb-calibration/6"
    # nm and degrees under names that say so: phi12 was radians beside higher-order angles in degrees
    assert saved["aberrations"] == {"C10_nm": -4.0, "C12_nm": 0.25, "phi12_deg": 12.0}
    assert (saved["scan_detector_rotation_deg"], saved["scan_detector_rotation_flipped"], saved["display_negate_phase"]) == (10.0, False, False)
    assert (saved["notes"], saved["scan_region"]) == ("lamella A", [0, SCAN, 0, SCAN])
    # one [row, col] pair for the scan step and one spelling for the detector step, filled from the session
    assert (saved["voltage_kV"], saved["semiangle_mrad"], saved["scan_sampling_A"], saved["detector_sampling_mrad"]) == (300.0, 20.0, [0.5, 0.5], [5.0, 5.0])
    # the acquisition by name and by its place under the save folder, never by absolute path
    assert saved["source"] == {"file_name": "scan_master.h5", "relative_path": "scan_master.h5"}
    assert saved["loss"] == json.loads(widget.result_json)["loss"] and "source_path" not in saved
    assert saved["sample_tilt_mrad"] is None and saved["generated_at_utc"] == widget.calibration_saved_at and saved["generated_at_utc"].endswith("Z")
    assert saved["upsampling_factor"] == widget.upsampling_factor
    retired = {"schema_version", "version", "aberration_unit", "rotation_angle_deg", "higher_order", "det_sampling_mrad_px",
               "source_file", "source_stem", "label", "timestamp", "tilt_mrad", "tilt_object_mrad"}
    assert not retired & set(saved)
    assert ssb.aberrations == {"C10_nm": -4.0, "C12_nm": 0.25, "phi12_deg": 12.0} and ssb.best_loss == saved["loss"]


def test_save_calibration_writes_what_the_save_button_writes(widget, ssb, tmp_path):
    # a notebook had no public Save: it had to bump the button's trait, viewer.save_trigger += 1
    widget.request_json = json.dumps({"id": 1, "C10_nm": -4.0, "C12_nm": 0.25, "phi12_deg": 12.0, "committed": True})
    path = widget.save_calibration()
    assert path == (tmp_path / "calibration.json").resolve() and widget.calibration_status == "Saved calibration"
    by_call = json.loads(path.read_text())
    assert by_call["aberrations"] == {"C10_nm": -4.0, "C12_nm": 0.25, "phi12_deg": 12.0}
    assert ssb.aberrations == by_call["aberrations"]
    widget.save_trigger += 1
    by_button = json.loads(path.read_text())
    assert by_button["id"] != by_call["id"]
    assert {key: value for key, value in by_button.items() if key not in ("id", "generated_at_utc")} == {
        key: value for key, value in by_call.items() if key not in ("id", "generated_at_utc")}
    assert not hasattr(widget, "save")  # ShowPtycho has no display-state file; save_calibration names what it writes
    with pytest.raises(ValueError, match="SSB session"):
        ShowPtycho(np.zeros((8, 8), np.float32)).save_calibration()


def test_tilt_comparison_preserves_the_saved_reconstruction(widget, ssb):
    """Compare zero tilt, save while viewing it, then recover the corrected image."""
    sample = {"sample_tilt_mrad": [3.0, -4.0], "depth_spread_nm": 12.0}
    widget.sample_json = json.dumps(sample)
    corrected = _shown(widget).copy()
    corrected_loss = json.loads(widget.result_json)["loss"]

    widget.tilt_correction = False
    expected, _ = ssb.preview(dict(START), sample_tilt_mrad=(0, 0), depth_spread_nm=12)
    np.testing.assert_array_equal(_shown(widget), expected)
    assert not np.array_equal(expected, corrected)
    assert json.loads(widget.sample_json) == sample

    widget.save_trigger += 1
    saved = json.loads(Path(widget.calibration_path).read_text())
    assert saved["sample_tilt_mrad"] == [3.0, -4.0]
    assert saved["depth_spread_nm"] == ssb.depth_spread_nm == 12.0
    assert ssb.sample_tilt_mrad == (3.0, -4.0)
    assert saved["loss"] == corrected_loss
    assert saved["display"]["tilt_correction"] is False
    assert not widget.tilt_correction
    np.testing.assert_array_equal(_shown(widget), expected)

    widget.tilt_correction = True
    np.testing.assert_array_equal(_shown(widget), corrected)


def test_higher_order_tilt_and_sampling_match_the_gpu_session(widget, ssb):
    """Review and save the same complete correction supported by the backend."""
    sample = {"sample_tilt_mrad": [3.0, -4.0], "depth_spread_nm": 12.0}
    widget.sample_json = json.dumps(sample)
    widget.higher_order_aberrations_json = json.dumps({"C30_nm": 2.0})
    optics = {**START, "C30_nm": 2.0}
    for factor in (1, 2, "auto"):
        widget.upsampling_factor = factor
        resolved = widget.auto_upsampling_factor if factor == "auto" else factor
        expected, _ = ssb.preview(optics, upsampling_factor=resolved, **sample)
        np.testing.assert_array_equal(_shown(widget), expected)
    widget.save_trigger += 1
    saved = json.loads(Path(widget.calibration_path).read_text())
    assert saved["sample_tilt_mrad"] == sample["sample_tilt_mrad"]
    assert ssb.aberrations["C30_nm"] == 2.0
    widget.tilt_correction = False
    expected, _ = ssb.preview(
        optics, upsampling_factor=resolved,
        sample_tilt_mrad=(0.0, 0.0), depth_spread_nm=12.0,
    )
    np.testing.assert_array_equal(_shown(widget), expected)


def test_opening_a_fitted_session_preserves_higher_order_aberrations(ssb, tmp_path, monkeypatch):
    """Opening and saving a fitted session must not silently drop its C30 term."""
    original = dict(ssb.aberrations)
    preview = ssb.preview
    calls = []

    def measured_preview(*args, **kwargs):
        calls.append((args, kwargs))
        return preview(*args, **kwargs)

    monkeypatch.setattr(ssb, "preview", measured_preview)
    try:
        ssb.aberrations = {**START, "C30_nm": 2.0}
        viewer = ShowPtycho(ssb, save_dir=tmp_path)
        assert len(calls) == 1  # Adopt the complete model before its first reconstruction.
        assert json.loads(viewer.higher_order_aberrations_json) == {"C30_nm": 2.0}
        viewer.save_trigger += 1
        assert ssb.aberrations["C30_nm"] == 2.0
        expected, _ = ssb.preview(dict(ssb.aberrations),
                                  upsampling_factor=viewer.auto_upsampling_factor,
                                  sample_tilt_mrad=ssb.sample_tilt_mrad or (0.0, 0.0),
                                  depth_spread_nm=ssb.depth_spread_nm or 0.0)
        np.testing.assert_array_equal(_shown(viewer), expected)
    finally:
        ssb.aberrations = original


def test_freezing_keeps_the_saved_phase_without_owning_the_session(widget, ssb):
    widget.upsampling_factor = "auto"
    phase = _shown(widget).copy()
    widget.freeze()
    assert not widget.ssb_session and not widget.crop_refit_available
    assert widget.object_sampling_A == [0.5 / widget.auto_upsampling_factor] * 2
    np.testing.assert_array_equal(_shown(widget), phase)
    widget.display_negate_phase = True
    np.testing.assert_array_equal(_shown(widget), -phase)
    # A second consumer can still use the session: freeze detaches the viewer.
    expected, _ = ssb.preview(dict(ssb.aberrations), upsampling_factor=widget.auto_upsampling_factor)
    np.testing.assert_array_equal(phase, expected)


def test_pins_star_view_and_unpin(widget, tmp_path):
    first = _shown(widget).copy()
    widget.pin_json = json.dumps({"action": "pin", "id": "a", "C10_nm": 3.0, "C12_nm": 1.0, "phi12_deg": 11.0, "loss": 0.1})
    widget.request_json = json.dumps({"id": 2, "C10_nm": 0.0, "C12_nm": 0.0, "phi12_deg": 0.0, "committed": True})
    widget.pin_json = json.dumps({"action": "pin", "id": "b", "C10_nm": 0.0, "C12_nm": 0.0, "phi12_deg": 0.0, "loss": 0.2})
    assert repr(widget).startswith("ShowPtycho(2 pinned")
    widget.pin_json = json.dumps({"action": "star", "id": "a"})
    stars = json.loads((tmp_path / "showptycho_stars.json").read_text())
    assert stars["schema"] == "quantem.ssb-stars/3" and [star["id"] for star in stars["stars"]] == ["a"]
    star = stars["stars"][0]
    # a star is the calibration file's record: the same keys, units and physical rotation
    assert star["aberrations"] == {"C10_nm": 3.0, "C12_nm": 1.0, "phi12_deg": 11.0}
    assert (star["scan_detector_rotation_deg"], star["loss"], star["source"]) == (
        10.0, 0.1, {"file_name": "scan_master.h5", "relative_path": "scan_master.h5"})
    assert (star["scan_sampling_A"], star["detector_sampling_mrad"]) == ([0.5, 0.5], [5.0, 5.0])
    widget.pin_json = json.dumps({"action": "view", "id": "a"})
    np.testing.assert_array_equal(_shown(widget), first)
    assert json.loads(widget.result_json)["C10_nm"] == 3.0
    widget.pin_json = json.dumps({"action": "unpin", "id": "a"})
    assert json.loads((tmp_path / "showptycho_stars.json").read_text()) == {"schema": "quantem.ssb-stars/3", "stars": []}
    assert repr(widget).startswith("ShowPtycho(1 pinned")


def test_export_writes_the_browser_folder_from_exact_bf_counts(widget, ssb, tmp_path, capsys):
    capsys.readouterr()
    out = widget.export(tmp_path / "review", title="my sample SSB")
    assert capsys.readouterr().out == ""                                   # an explicit 1x is exported as chosen
    assert {p.name for p in out.iterdir()} == {"index.html", "source", "snapshots", ".viewer", "ShowPtycho.command"}
    suffix = "qem" if ssb.backend == "cuda" else "u8"  # all fixture counts fit uint8 exactly
    assert sorted(p.name for p in (out / "source").iterdir()) == [f"bf_columns.{suffix}"]
    cal = json.loads((out / "snapshots" / "cal.json").read_text())
    assert cal["schema"] == "quantem.ssb-browser-calibration/3" and cal["source_path"] == "redacted_local_source"
    assert cal["display_negate_phase"] is False and "flip_phase" not in cal
    assert cal["precision"] == {"real_dtype": "float32", "complex_dtype": "complex64"}  # the browser kernels refuse any other
    assert (cal["num_bright_field_pixels"], cal["g_shape"], cal["phase_shape"], cal["detector_shape"]) == (ssb.num_bright_field_pixels, [ssb.num_bright_field_pixels, SCAN, SCAN], [SCAN, SCAN], [DET, DET])
    # each value once, in the vocabulary's unit: no second copy in radians, no folded angle
    assert cal["aberrations"] == {"C10_nm": 3.0, "C12_nm": 1.0, "phi12_deg": pytest.approx(math.degrees(0.2))}
    assert (cal["scan_detector_rotation_deg"], cal["scan_detector_rotation_flipped"], cal["scan_sampling_A"], cal["voltage_kV"], cal["gpu_memory_gb"]) == (10.0, False, [0.5, 0.5], 300.0, 4.5)
    assert cal["detector_sampling_mrad"] == pytest.approx([5.0, 5.0]) and cal["scan_region"] == [0, SCAN, 0, SCAN]
    state = ssb.browser_state()
    # the reciprocal arrays keep their values under axis names: quantem.gpu's x is the scan row axis
    np.testing.assert_array_equal(np.asarray(cal["bright_field_k_row"], np.float32), np.asarray(state.kx_bf, np.float32))
    np.testing.assert_array_equal(np.asarray(cal["scan_q_col"], np.float32), np.asarray(state.qy_1d, np.float32))
    retired = {"schema_version", "kind", "source_file", "num_bf", "bf_center", "bf_radius_px", "kx_bf", "ky_bf", "qx_1d", "qy_1d",
               "semiangle_rad", "sampling_A", "angular_sampling_rad", "det_sampling_mrad_px", "rotation_angle_deg", "rotation_angle_rad"}
    assert not retired & set(cal)
    assert cal["bf_column_companion_path"] == f"source/bf_columns.{suffix}" and cal["persistent_bf_cache"] is False
    manifest = json.loads((out / "snapshots" / "manifest.json").read_text())
    assert manifest["schema"] == "quantem.ssb-browser-folder/3" and manifest["title"] == "my sample SSB"
    assert manifest["source"]["bf_columns"]["scan_shape"] == [SCAN, SCAN]
    assert json.loads((out / "snapshots" / "snapshots.json").read_text()) == {"schema": "quantem.ssb-snapshots/2", "snapshots": []}
    html = (out / "index.html").read_text()
    embedded = json.loads(re.search(r'<script type="application/vnd.jupyter.widget-state\+json">(.*?)</script>', html, re.S).group(1))
    exported = next(model["state"] for model in embedded["state"].values() if "webgpu_standalone" in model["state"])
    assert exported["webgpu_standalone"] and exported["webgpu_preview_enabled"] and not exported["crop_refit_available"]
    assert exported["auto_upsampling_factor"] == 4 and widget.auto_upsampling_factor == 3
    assert exported["webgpu_preview_status"].startswith("WebGPU folder ready") and widget.webgpu_preview_status == ""
    source = json.loads(exported["webgpu_h5_source_json"])
    assert source["kind"] == "bf_columns" and exported["stars_path"] == "snapshots/snapshots.json"
    assert (source["num_bright_field_pixels"], source["scan_shape"]) == (ssb.num_bright_field_pixels, [SCAN, SCAN]) and "numBf" not in source
    assert str(tmp_path) not in html and "scan_master.h5" not in html
    assert widget.crop_refit_available and not widget.webgpu_standalone and len(widget.phase_bytes) == 4 * SCAN * SCAN
    counts_file = (out / "source" / f"bf_columns.{suffix}").read_bytes()
    if ssb.backend == "mps":
        expected = _counts()[..., state.brightfield.rows, state.brightfield.cols].transpose(2, 0, 1)
        exported_counts = np.frombuffer(counts_file, np.uint8).reshape(expected.shape)
        np.testing.assert_array_equal(exported_counts, expected)
    widget.export(out)                                                     # a re-export reuses its own counts
    assert (out / "source" / f"bf_columns.{suffix}").read_bytes() == counts_file
    # WebGPU FFTs take powers of two: the automatic 3x is exported at 4x, and one line says so
    widget.upsampling_factor = "auto"
    capsys.readouterr()
    widget.export(tmp_path / "auto")
    assert capsys.readouterr().out == ("ShowPtycho export: automatic sampling is 3x here; the browser's WebGPU FFT "
                                       "takes 1, 2, 4 or 8x, so the exported review shows 4x.\n")


@pytest.mark.parametrize("transposed", [False, True])
def test_qem_export_reuses_the_acquisition_and_preserves_the_live_phase(ssb, tmp_path, monkeypatch, transposed):
    """A scientist can export an unchanged QEM session without another encode."""
    from quantem.gpu import SSB, io

    acquisition_dir = tmp_path / "acquisition"
    acquisition_dir.mkdir()
    path = acquisition_dir / "measured.qem"
    io.save(path, _counts(), backend="cpu")  # portable synthetic reference file
    original_bytes = path.read_bytes()
    session = SSB.open(
        str(path), voltage_kV=300.0, semiangle_mrad=20.0,
        scan_sampling_A=(0.5, 0.5), detector_sampling_mrad=(5.0, 5.0),
        bright_field_radius_px=4.0, bright_field_center_px=(7.5, 7.5),
        aberrations=dict(START), scan_detector_rotation_deg=10.0,
        scan_detector_rotation_flipped=transposed,
    )
    try:
        viewer = ShowPtycho(session, save_dir=tmp_path / "live")
        viewer.upsampling_factor = 1
        phase = _shown(viewer).copy()
        qem = session.browser_state().qem_source
        assert qem is not None

        def unnecessary_companion(*args, **kwargs):
            pytest.fail("An unchanged QEM acquisition must not be re-encoded as BF columns")

        export_companion = viewer._ssb.export_bright_field
        monkeypatch.setattr(viewer._ssb, "export_bright_field", unnecessary_companion)
        review = tmp_path / "review"
        review.mkdir()
        (review / "source").symlink_to(acquisition_dir, target_is_directory=True)
        out = viewer.export(review)
        assert not (out / "source").is_symlink()
        linked = out / "source" / "acquisition.qem"
        assert linked.samefile(path) and not linked.is_symlink()
        assert [p.name for p in linked.parent.iterdir()] == ["acquisition.qem"]
        manifest = json.loads((out / "snapshots" / "manifest.json").read_text())
        source = manifest["source"]["full_detector"]
        assert manifest["source"]["preferred_browser_source"] == "full_detector"
        assert source["kind"] == "full_detector" and source["encoding"] == "qem"
        assert source["scan_shape"] == [SCAN, SCAN] and source["detector_shape"] == [DET, DET]
        assert source["detector_origin_px"] == list(qem["detector_origin_px"])
        assert source["scan_detector_rotation_flipped"] is transposed
        assert source["header_sha256"] == qem["header_sha256"]
        assert re.fullmatch(r"[0-9a-f]{64}", source["header_sha256"])
        assert source["dtype"] == "uint16" and source["link"] == "hardlink"
        assert source["num_bright_field_pixels"] == session.num_bright_field_pixels
        cal = json.loads((out / "snapshots" / "cal.json").read_text())
        assert cal["source_transport"] == "full_detector" and not cal["bf_column_companion"]
        assert cal["scan_detector_rotation_flipped"] is transposed
        assert "bf_column_companion_path" not in cal
        html = (out / "index.html").read_text()
        embedded = json.loads(re.search(r'<script type="application/vnd.jupyter.widget-state\+json">(.*?)</script>', html, re.S).group(1))
        exported = next(model["state"] for model in embedded["state"].values() if "webgpu_standalone" in model["state"])
        browser = json.loads(exported["webgpu_h5_source_json"])
        assert browser["kind"] == "full_detector" and browser["url"] == "source/acquisition.qem"
        assert browser["detector_origin_px"] == source["detector_origin_px"]
        assert browser["scan_detector_rotation_flipped"] is transposed
        assert browser["header_sha256"] == source["header_sha256"]
        assert str(path) not in html
        viewer.export(out)
        assert linked.samefile(path) and path.read_bytes() == original_bytes
        np.testing.assert_array_equal(_shown(viewer), phase)
        # If the original is moved, prepared counts still export exactly as BF
        # columns. The existing hardlinked review remains self-contained.
        moved = path.with_name("moved.qem")
        path.rename(moved)
        monkeypatch.setattr(viewer._ssb, "export_bright_field", export_companion)
        fallback = viewer.export(tmp_path / "fallback")
        fallback_manifest = json.loads((fallback / "snapshots" / "manifest.json").read_text())
        assert fallback_manifest["source"]["kind"] == "bf_columns"
        assert fallback_manifest["source"]["bf_columns"]["num_bright_field_pixels"] == session.num_bright_field_pixels
        assert linked.read_bytes() == moved.read_bytes() == original_bytes
        np.testing.assert_array_equal(_shown(viewer), phase)
    finally:
        session.close()


def test_replacing_an_export_link_keeps_both_external_sources(tmp_path):
    """Replacing a stale link must unlink the link, not its external target."""
    old = tmp_path / "old.qem"
    new = tmp_path / "new.qem"
    old.write_bytes(b"older exact source")
    new.write_bytes(b"current exact source")
    target = tmp_path / "review" / "source" / "acquisition.qem"
    target.parent.mkdir(parents=True)
    target.symlink_to(old)
    assert showptycho_export._link_counts(new, target) == "hardlink"
    assert old.read_bytes() == b"older exact source"
    assert new.read_bytes() == target.read_bytes() == b"current exact source"
    assert target.samefile(new) and not target.is_symlink()


@pytest.mark.parametrize("replace_when", ["before session", "after session"])
def test_replaced_qem_header_falls_back_even_when_size_and_mtime_match(ssb, tmp_path, replace_when):
    """A reused pathname must not silently bind a loaded session to another source."""
    from quantem.gpu import SSB, io

    source, replacement = tmp_path / "scan.qem", tmp_path / "replacement.qem"
    counts = _counts()
    io.save(source, counts, backend="cpu", metadata={"annotation": "alpha"})
    io.save(replacement, counts, backend="cpu", metadata={"annotation": "bravo"})
    original_stat = source.stat()
    assert replacement.stat().st_size == original_stat.st_size
    assert source.read_bytes() != replacement.read_bytes()

    def replace_source():
        replacement.replace(source)
        os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    with io.load(source) as data:
        if replace_when == "before session":
            replace_source()
        with SSB(data, voltage_kV=300.0, semiangle_mrad=20.0,
                 scan_sampling_A=(0.5, 0.5), detector_sampling_mrad=(5.0, 5.0),
                 aberrations=dict(START), scan_detector_rotation_deg=10.0) as session:
            if replace_when == "after session":
                assert session.browser_state().qem_source is not None
                replace_source()
            assert session.browser_state().qem_source is None
            viewer = ShowPtycho(session, save_dir=tmp_path / "live")
            viewer.upsampling_factor = 1
            phase = _shown(viewer).copy()
            out = viewer.export(tmp_path / "review")
            manifest = json.loads((out / "snapshots" / "manifest.json").read_text())
            assert manifest["source"]["kind"] == "bf_columns"
            assert manifest["source"]["bf_columns"]["num_bright_field_pixels"] == session.num_bright_field_pixels
            np.testing.assert_array_equal(_shown(viewer), phase)


def test_browser_memory_budget_fits_every_bf_pixel_at_full_scan_resolution():
    """A full detector selection must never be reduced to fit the old 4.5 GiB cap."""
    rows, cols, num_bf = 512, 512, 8900
    budget = showptycho_export._resident_spectra_budget(num_bf, (rows, cols))
    per_bf_bytes = rows * (cols // 2 + 1) * np.dtype(np.complex64).itemsize
    assert budget > 4.5
    assert int(budget * 2**30) // per_bf_bytes >= num_bf
    assert showptycho_export._resident_spectra_budget(52, (128, 128)) == 4.5


def test_crop_refit_validates_the_region_and_a_session_keeps_its_own_sampling(widget, ssb):
    widget.crop_refit_request_json = json.dumps({"scan_region": [0, 16, 0, 16], "num_trials": 1})
    assert widget.crop_refit_status.startswith("Crop refit failed: SSB refit crops must be at least 32x32")
    widget.crop_refit_request_json = json.dumps({"scan_region": [0, 64, 100, 164], "num_trials": 1})
    assert "outside the 128x128 scan" in widget.crop_refit_status
    assert widget.ssb_session and widget.session_note == "" and widget.amplitude_bytes == b""
    with pytest.raises(ValueError, match="scan_sampling_A"):
        ShowPtycho(ssb, sampling=1.0)
    # the session is the source of its file: the keyword that named it again is gone
    with pytest.raises(TypeError, match="source_file"):
        ShowPtycho(ssb, source_file="scan_master.h5")


def test_crop_and_refit_keeps_the_acquisition_and_reopens_as_a_scan_region_session(tmp_path, monkeypatch):
    """The rebuilt crop session was an "array" with no identity, no bad-pixel provenance, no flip, a
    bright-field disk re-detected on the crop and the parent's saved identity; its save never reopened."""
    gpu = pytest.importorskip("quantem.gpu")
    if not gpu_adapter.accelerator_ready():
        pytest.skip("Needs CUDA or MPS")
    source = tmp_path / "scan.npy"
    np.save(source, _counts())
    calibration = dict(voltage_kV=300, semiangle_mrad=20, scan_sampling_A=(0.5, 0.5), detector_sampling_mrad=(5, 5),
                       scan_detector_rotation_deg=10.0, scan_detector_rotation_flipped=True)
    region = (16, 112, 8, 104)
    with gpu.io.load(str(source)) as data:
        with gpu.SSB(data, aberrations=START, **calibration) as ssb:
            viewer = ShowPtycho(ssb, save_dir=tmp_path)
            assert viewer.crop_refit_available
            viewer.crop_refit_request_json = json.dumps({"scan_region": list(region), "num_trials": 2})
            assert viewer.crop_refit_status.startswith("Refit complete: [16:112, 8:104]")
            crop = viewer._ssb.ssb
            assert crop is not ssb and crop.scan_region == region and crop.input_scan_shape == (96, 96)
            assert (crop.source_kind, crop.source_path, crop.source_provenance) == (ssb.source_kind, ssb.source_path, ssb.source_provenance)
            assert crop.source_kind == "resident" and crop.scan_detector_rotation_flipped
            assert (crop.bright_field_center_px, crop.bright_field_radius_px, crop.detector_sampling_mrad) == (
                ssb.bright_field_center_px, ssb.bright_field_radius_px, ssb.detector_sampling_mrad)
            record = viewer.calibration()
            assert record["scan_region"] == list(region)
            assert record["session_identity"]["data"]["scan_region"] == list(region)
            viewer.upsampling_factor = 1
            viewer.request_json = json.dumps({"id": 1, **START, "committed": True})
            expected = _shown(viewer).copy()
            crop_tilt = tuple(json.loads(viewer.sample_json)["sample_tilt_mrad"])
            viewer.save_trigger += 1
            viewer.close()
            assert ssb.scan_region is None  # the caller's session is unchanged

    def no_search(*args, **kwargs):
        raise AssertionError("Restoring a saved model must not run optimization.")

    monkeypatch.setattr(gpu.SSB, "find_aberrations", no_search)
    with gpu.io.load(str(source)) as data:
        with gpu.SSB(data, scan_region=region, **calibration) as reopened:
            restored = ShowPtycho(reopened, save_dir=tmp_path)
            assert restored.calibration_status == "Restored saved fit of scan region [16, 112, 8, 104]"
            np.testing.assert_array_equal(_shown(restored), expected)
            result = reopened.reconstruct()  # no refit: find_aberrations would raise
            assert result.aberrations == START and result.sample_tilt_mrad == crop_tilt
            restored.close()


def test_session_scale_bar_uses_the_column_scan_step():
    # D-1 (ShowPtycho): an SSB session gave the scale bar its row step; the array path uses the column.
    gpu = pytest.importorskip("quantem.gpu")
    try:
        pytest.importorskip("quantem.gpu.device").detect()
    except RuntimeError as exc:
        pytest.skip(f"SSB needs CUDA or MPS: {exc}")
    session = gpu.SSB(_counts(SCAN), voltage_kV=300.0, semiangle_mrad=20.0, scan_sampling_A=(0.5, 0.25), detector_sampling_mrad=(5.0, 5.0),
                      aberrations=dict(START), scan_detector_rotation_deg=10.0)
    # the trait holds both steps; the browser's horizontal bar reads the column one
    assert ShowPtycho(session).scan_sampling_A == [0.5, 0.25]


def test_a_dataset_in_the_format_unit_draws_its_scale_bar():
    # a Dataset2d whose unit is "angstrom" (what a .qem file writes) opened with no scale bar and no message
    from quantem.widget.adapters.core import make_dataset

    phase = np.zeros((16, 16), np.float32)
    assert ShowPtycho(make_dataset(phase, sampling=(0.2, 0.3), units=("angstrom", "angstrom"))).object_sampling_A == pytest.approx([0.2, 0.3])
    assert ShowPtycho(make_dataset(phase, sampling=(0.05, 0.05), units=("nm", "nm"))).object_sampling_A == pytest.approx([0.5, 0.5])
    assert ShowPtycho(make_dataset(phase)).object_sampling_A == []  # an uncalibrated dataset is absent
    # the dataset is the source: a keyword may repeat its pixel size, never contradict it
    assert ShowPtycho(make_dataset(phase, sampling=(0.2, 0.2), units=("angstrom", "angstrom")), sampling=0.02, units="nm").object_sampling_A == pytest.approx([0.2, 0.2])
    with pytest.raises(ValueError, match="contradicts the value the input records"):
        ShowPtycho(make_dataset(phase, sampling=(0.2, 0.2), units=("angstrom", "angstrom")), sampling=0.3)


def test_the_rewrite_script_carries_earlier_calibration_files_over(tmp_path):
    pytest.importorskip("quantem.gpu", reason="the rewrite script reads quantem.gpu's session identities")
    # the package reads one layout; a calibration a person chose is carried over once by scripts/rewrite_ssb_calibration.py
    script = Path(__file__).resolve().parents[2] / "scripts" / "rewrite_ssb_calibration.py"
    saved = tmp_path / "calibration.json"
    saved.write_text(json.dumps({
        "schema_version": 3, "version": "2.0", "aberration_unit": "nm", "rotation_angle_deg": 345.5,
        "aberrations": {"C10": -4.0, "C12": 0.25, "phi12": math.radians(12.0), "C21_mag": 5.0, "C21_angle": 45.0, "C30": 2.0},
        "flip_phase": True, "voltage_kV": 300.0, "semiangle_mrad": 20.0, "scan_sampling_A": 0.5, "loss": 0.1,
        "source_file": "scan_master.h5", "scan_region": [0, 128, 0, 128], "notes": "lamella A", "id": "abc",
        "tilt_mrad": [3.0, -4.0], "tilt_object_mrad": [2.3, -4.5], "depth_spread_nm": 12.0, "timestamp": "2026-10-01T10:00:00",
    }))
    fitted = tmp_path / "scan_128x128x16x16_ssb_calibration.json"
    fitted.write_text(json.dumps({
        "schema": "quantem.ssb.calibration.v1", "created_utc": "2026-10-01T10:00:00Z",
        "source": {"widget": "Show4DSTEM", "title": "scan", "frame_idx": 0, "scan_shape": [128, 128], "detector_shape": [16, 16]},
        "aberrations": {"C10": 12.0, "C12": 3.0, "phi12": math.pi / 6}, "aberration_units": {"C10": "nm", "C12": "nm", "phi12": "rad"},
        "rotation_angle_deg": 165.5, "com_reversed": True,
        "calibration": {"voltage_kV": 300.0, "semiangle_mrad": 20.0, "scan_sampling_A": [0.4, 0.6], "det_sampling_mrad": [1.5, 1.5],
                        "bf_radius": None, "bf_intensity_threshold": 0.5, "bf_pixels": 52},
        "run": {"n_trials": 200, "refine": True, "manual_locked": False, "seed": 42, "loss": 0.25, "elapsed_s": 1.5},
    }))
    before = saved.read_text()
    dry = subprocess.run([sys.executable, str(script), "--dry-run", str(saved)], capture_output=True, text=True, check=True)
    assert saved.read_text() == before and "would be rewritten" in dry.stdout and "row step 0.5" in dry.stdout
    subprocess.run([sys.executable, str(script), str(saved), str(fitted)], capture_output=True, text=True, check=True)
    current = json.loads(saved.read_text())
    assert current["schema"] == "quantem.ssb-calibration/6" and (tmp_path / "calibration.json.before").read_text() == before
    assert current["aberrations"] == pytest.approx({"C10_nm": -4.0, "C12_nm": 0.25, "phi12_deg": 12.0, "C21_nm": 5.0, "phi21_deg": 45.0, "C30_nm": 2.0})
    assert (current["scan_detector_rotation_deg"], current["scan_sampling_A"], current["source"]) == (
        345.5, [0.5, 0.5], {"file_name": "scan_master.h5", "relative_path": None})
    assert (current["sample_tilt_mrad"], current["propagator_tilt_mrad"], current["depth_spread_nm"]) == ([3.0, -4.0], [2.3, -4.5], 12.0)
    current = json.loads(fitted.read_text())
    assert current["schema"] == "quantem.show4dstem-ssb-calibration/3"  # Show4DSTEM's record, not ShowPtycho's
    assert current["refinement"] == "nelder-mead" and "refine" not in current
    # Compute SSB fitted at tilt 0 until /3; the tilt-corrected fit is its default now
    tilt_zero = {"tilt": False, "sample_tilt_mrad": None, "depth_spread_nm": None, "tilt_fit_gain": None}
    assert {key: current[key] for key in tilt_zero} == tilt_zero
    layout_1 = tmp_path / "layout_1_ssb_calibration.json"
    layout_1.write_text(json.dumps({"schema": "quantem.show4dstem-ssb-calibration/1", "refine": False, "loss": 0.2}))
    layout_2 = tmp_path / "layout_2_ssb_calibration.json"
    layout_2.write_text(json.dumps({"schema": "quantem.show4dstem-ssb-calibration/2", "refinement": "nelder-mead", "loss": 0.3}))
    subprocess.run([sys.executable, str(script), str(layout_1), str(layout_2)], capture_output=True, text=True, check=True)
    assert json.loads(layout_1.read_text()) == {"schema": "quantem.show4dstem-ssb-calibration/3", "refinement": None, "loss": 0.2, **tilt_zero}
    assert json.loads(layout_2.read_text()) == {"schema": "quantem.show4dstem-ssb-calibration/3", "refinement": "nelder-mead", "loss": 0.3, **tilt_zero}
    # the angle below 180 degrees and its reversal flag become the one physical angle
    assert (current["scan_detector_rotation_deg"], current["detector_sampling_mrad"], current["num_bright_field_pixels"]) == (345.5, [1.5, 1.5], 52)
    assert current["aberrations"] == pytest.approx({"C10_nm": 12.0, "C12_nm": 3.0, "phi12_deg": 30.0}) and current["source"]["acquisition_index"] == 0
    again = subprocess.run([sys.executable, str(script), str(saved)], capture_output=True, text=True, check=True)
    assert "already in the current layout" in again.stdout
    # before the nm marker the numbers were Angstrom under an nm label: such a file is refused, not guessed
    unmarked = tmp_path / "older.json"
    unmarked.write_text(json.dumps({"schema_version": 3, "rotation_angle_deg": 10.0, "aberrations": {"C10": -40.0}}))
    refused = subprocess.run([sys.executable, str(script), str(unmarked)], capture_output=True, text=True)
    assert refused.returncode != 0 and "unit of C10 and C12 in this file is not known" in refused.stderr


def test_the_rewrite_script_carries_a_path_identity_over_with_its_data(tmp_path):
    pytest.importorskip("quantem.gpu", reason="the rewrite script reads quantem.gpu's session identities")
    """Calibration /5 named the data by absolute path and matched it by inode and times; /6 recomputes a portable identity."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "rewrite_ssb_calibration.py"
    data = tmp_path / "scan.npy"
    np.save(data, np.zeros((2, 2, 4, 4), np.uint16))
    hot = {"method": "median", "pixel_count": 1, "coordinates_row_col": [[1, 1]], "applied": True, "kernel_size": 3}
    record = {"schema": "quantem.ssb-calibration/5", "source_path": "/elsewhere/scan.npy", "scan_shape": [2, 2],
              "session_identity": {"source": {"detector": {"path": "/elsewhere/scan.npy", "inode": 7}, "storage": None,
                                              "calibration": None, "manifest": None},
                                   "data": {"kind": "detector", "provenance": {"hot_pixel_correction": hot}},
                                   "geometry": {"voltage_kV": 300.0}}}
    saved = tmp_path / "calibration.json"
    saved.write_text(json.dumps(record))
    subprocess.run([sys.executable, str(script), str(saved)], capture_output=True, text=True, check=True)
    current = json.loads(saved.read_text())
    assert current["schema"] == "quantem.ssb-calibration/6" and "source_path" not in current
    assert current["source"] == {"file_name": "scan.npy", "relative_path": "scan.npy"}
    identity = current["session_identity"]
    assert identity["data"] == {"kind": "detector"} and identity["geometry"] == {"voltage_kV": 300.0}
    assert identity["source"]["detector"]["file_name"] == "scan.npy"
    assert identity["source"]["detector"]["hot_pixel_correction"] == {
        "method": "median", "pixel_count": 1, "applied": True, "largest_window_px": 3}
    assert _absolute_paths(current) == []
    # without the data beside it, the identity cannot be computed and the file is left as it was
    away = tmp_path / "away"
    away.mkdir()
    (away / "calibration.json").write_text(json.dumps(record))
    refused = subprocess.run([sys.executable, str(script), str(away / "calibration.json")], capture_output=True, text=True)
    assert refused.returncode != 0 and "pass --data" in refused.stderr
    subprocess.run([sys.executable, str(script), "--data", str(data), str(away / "calibration.json")],
                   capture_output=True, text=True, check=True)
    assert json.loads((away / "calibration.json").read_text())["source"] == {"file_name": "scan.npy", "relative_path": None}


def test_the_rewrite_script_carries_layout_4_records_to_the_canonical_names(tmp_path):
    pytest.importorskip("quantem.gpu", reason="the rewrite script reads quantem.gpu's session identities")
    # calibration /4, stars /1 and snapshots /1 named the display sign flip_phase and the regional
    # tilt, score and trial count tilt_mrad, agreement and completed_trials; ShowPtycho now refuses them
    from quantem.widget.showptycho.calibration import read_calibration
    from quantem.widget.showptycho.widget import CALIBRATION_SCHEMA

    script = Path(__file__).resolve().parents[2] / "scripts" / "rewrite_ssb_calibration.py"
    run = {"seed": 11, "tilt_mrad": [1.0, 2.0], "agreement": 0.9, "completed_trials": 200}
    files = {
        "calibration.json": {"schema": "quantem.ssb-calibration/4", "flip_phase": True, "loss": 0.1,
                             "local_tilt": {"fits": [{"id": "a", "tilt_mrad": [1.0, 2.0], "agreement": 0.9,
                                                      "fit_record": {"fits": [run]}}]}},
        "showptycho_stars.json": {"schema": "quantem.ssb-stars/1", "stars": [{"id": "a", "flip_phase": False}]},
        "snapshots.json": {"schema": "quantem.ssb-snapshots/1", "snapshots": [{"id": "s", "flip_phase": True}]},
        "cal.json": {"schema": "quantem.ssb-browser-calibration/2", "flip_phase": True, "scan_shape": [128, 128]},
    }
    for name, record in files.items():
        (tmp_path / name).write_text(json.dumps(record))
    with pytest.raises(ValueError, match="rewrite it once with scripts/rewrite_ssb_calibration.py"):
        read_calibration(tmp_path / "calibration.json", {}, (0, 1, 0, 1), None, CALIBRATION_SCHEMA)
    subprocess.run([sys.executable, str(script), *(str(tmp_path / name) for name in files)], capture_output=True, text=True, check=True)
    calibration = json.loads((tmp_path / "calibration.json").read_text())
    assert (calibration["schema"], calibration["display_negate_phase"], calibration["loss"]) == ("quantem.ssb-calibration/6", True, 0.1)
    fit = calibration["local_tilt"]["fits"][0]
    assert (fit["sample_tilt_mrad"], fit["sample_agreement"]) == ([1.0, 2.0], 0.9)
    assert fit["fit_record"]["fits"][0] == {"seed": 11, "sample_tilt_mrad": [1.0, 2.0], "sample_agreement": 0.9,
                                            "num_completed_trials": 200}
    assert json.loads((tmp_path / "showptycho_stars.json").read_text()) == {
        "schema": "quantem.ssb-stars/3", "stars": [{"id": "a", "display_negate_phase": False, "source": None}]}
    assert json.loads((tmp_path / "snapshots.json").read_text()) == {
        "schema": "quantem.ssb-snapshots/2", "snapshots": [{"id": "s", "display_negate_phase": True}]}
    assert json.loads((tmp_path / "cal.json").read_text()) == {
        "schema": "quantem.ssb-browser-calibration/3", "display_negate_phase": True, "scan_shape": [128, 128]}
    again = subprocess.run([sys.executable, str(script), str(tmp_path / "calibration.json")], capture_output=True, text=True, check=True)
    assert "already in the current layout" in again.stdout


def test_auto_sampling_matches_core_and_save_keeps_resolved_factor(widget, ssb, monkeypatch):
    """Native -> Auto -> native preserves optics, physical extent and saved sampling."""
    from quantem.gpu.optics.physics import ssb_upsampling_factor

    factor = ssb_upsampling_factor(voltage_kV=ssb.voltage_kV,
                                   semiangle_mrad=ssb.semiangle_mrad,
                                   scan_sampling_A=ssb.scan_sampling_A)
    widget.sample_json = json.dumps({"sample_tilt_mrad": [3.0, -4.0], "depth_spread_nm": 12.0})
    native = _shown(widget).copy()
    packets = []
    send_state = widget.send_state

    def record_state(key=None):
        packets.append({key} if isinstance(key, str) else set(key or ()))
        return send_state(key)

    monkeypatch.setattr(widget, "send_state", record_state)
    widget.upsampling_factor = "auto"
    image_keys = {"phase_height", "phase_width", "phase_bytes", "result_json"}
    assert any(image_keys <= keys for keys in packets)
    assert not any(keys & image_keys and not image_keys <= keys for keys in packets)
    assert widget.auto_upsampling_factor == factor == 3
    expected, _ = ssb.preview(dict(START), upsampling_factor=factor,
                              sample_tilt_mrad=(3.0, -4.0), depth_spread_nm=12.0,
                              compute_loss=False)
    np.testing.assert_array_equal(_shown(widget), expected)
    assert _shown(widget).shape == tuple(value * factor for value in native.shape)
    # Drag and release also carry the exact phase and completion in one packet,
    # so a notebook can start its next frame without waiting for another message.
    for request_id, committed in [(101, False), (102, True)]:
        packets.clear()
        aberrations = {**START, "C10_nm": 2.0 + request_id / 100}
        widget._handle_custom_msg({"type": "ssb-preview",
                                   "request": {"id": request_id, **aberrations, "committed": committed}}, [])
        assert any({"phase_bytes", "result_json"} <= keys for keys in packets)
        assert not any(keys & {"phase_bytes", "result_json"}
                       and not {"phase_bytes", "result_json"} <= keys for keys in packets)
        expected, loss = ssb.preview(aberrations, upsampling_factor=factor,
                                    sample_tilt_mrad=(3.0, -4.0), depth_spread_nm=12.0,
                                    compute_loss=committed)
        np.testing.assert_array_equal(_shown(widget), expected)
        assert json.loads(widget.result_json)["loss"] == loss
    widget.request_json = json.dumps({"id": 103, **START, "committed": True})
    widget.save_trigger += 1
    saved = json.loads(Path(widget.calibration_path).read_text())
    assert saved["upsampling_factor"] == factor and saved["upsampling_mode"] == "auto"
    assert saved["scan_sampling_A"] == list(ssb.scan_sampling_A)
    widget.upsampling_factor = 1
    np.testing.assert_array_equal(_shown(widget), native)
    widget.save_trigger += 1
    saved = json.loads(Path(widget.calibration_path).read_text())
    assert saved["upsampling_factor"] == 1 and saved["upsampling_mode"] == "explicit"


@pytest.mark.parametrize("estimator", ["mean_phase", "phase_of_mean"])
def test_explicit_estimator_is_displayed_and_saved(ssb, tmp_path, estimator):
    """A saved initialization must describe the numerical phase reviewed on either GPU."""
    view = ShowPtycho(ssb, save_dir=tmp_path, phase_estimator=estimator)
    try:
        view.upsampling_factor = "auto"
        expected, _ = ssb.preview(
            dict(ssb.aberrations), upsampling_factor=view.auto_upsampling_factor,
            phase_estimator=estimator, compute_loss=True,
            sample_tilt_mrad=ssb.sample_tilt_mrad or (0.0, 0.0),
            depth_spread_nm=ssb.depth_spread_nm or 0.0,
        )
        np.testing.assert_array_equal(_shown(view), expected)
        view.display_negate_phase = True
        view.save_trigger += 1
        record = json.loads(Path(view.calibration_path).read_text())
        with pytest.raises(ValueError, match="does not yet preserve"):
            view.export(tmp_path / "browser")
        assert not (tmp_path / "browser").exists()
        assert record["phase_estimator"] == estimator
        assert record["upsampling_factor"] == view.auto_upsampling_factor
        np.testing.assert_array_equal(_shown(view), -expected)
    finally:
        view.close()


def test_estimator_requires_a_supported_name_and_an_ssb_session():
    with pytest.raises(ValueError, match="phase_estimator must"):
        ShowPtycho(np.ones((8, 8)), phase_estimator="unknown")
    with pytest.raises(ValueError, match="needs an SSB session"):
        ShowPtycho(np.ones((8, 8)), phase_estimator="mean_phase")


def test_local_ssb_results_follow_current_optics_without_replacing_the_session(ssb, tmp_path):
    regions = [{'id': 'a', 'label': 'A', 'bounds': [16, 48, 24, 56]}]
    received = []
    def fit(selected):
        received.append(viewer.ssb_parameters)
        return {'regions': selected, 'fits': [{**selected[0], 'sample_tilt_mrad': [1., 2.]}],
                'images': {'A': np.ones((32, 32), dtype=np.float32)}, 'sampling_A': .5}
    viewer = ShowPtycho(ssb, save_dir=tmp_path,
                       local_tilt={'scan_shape': [128, 128], 'regions': regions}, fit_regions=fit)
    assert viewer.ssb_session and not viewer.local_tilt_enabled
    viewer.fit_regions()
    before = viewer.phase_bytes
    parameters = viewer.ssb_parameters
    changed = {**parameters['aberrations'], 'C10_nm': parameters['aberrations']['C10_nm'] + 2}
    viewer.request_json = json.dumps({'id': 918, **changed, 'committed': True})
    assert viewer.phase_bytes != before
    assert viewer.ssb_parameters['aberrations'] == changed
    assert viewer.local_tilt_stale
    with pytest.raises(ValueError, match='Shared optics changed'):
        viewer.region_fits
    with pytest.raises(ValueError, match='Shared optics changed'):
        viewer.region_images
    viewer.local_tilt_enabled = True
    viewer.fit_regions()
    assert received[-1]['aberrations'] == changed
    assert not viewer.local_tilt_stale
    assert viewer.region_fits[0]['sample_tilt_mrad'] == [1., 2.]
    viewer.local_tilt_enabled = False
    np.testing.assert_array_equal(viewer.region_images['A'], 1)
    assert viewer.ssb_session


def test_interaction_uses_all_bf_pixels_even_with_a_legacy_drag_count(ssb, tmp_path, monkeypatch):
    def no_subset(*args, **kwargs):
        raise AssertionError('Full-BF interaction must not allocate a reduced preview context.')

    monkeypatch.setattr(ssb, 'preview_context', no_subset)
    viewer = ShowPtycho(ssb, save_dir=tmp_path)
    viewer.num_drag_bright_field_pixels = max(1, ssb.num_bright_field_pixels // 2)
    assert viewer.num_drag_bright_field_pixels == ssb.num_bright_field_pixels
    parameters = {**viewer.ssb_parameters['aberrations'], 'C10_nm': 5.0}
    viewer.request_json = json.dumps({'id': 919, **parameters, 'committed': False})
    expected, _ = ssb.preview(parameters, upsampling_factor=viewer.auto_upsampling_factor,
                              compute_loss=False, **json.loads(viewer.sample_json))
    np.testing.assert_array_equal(_shown(viewer), expected)


def test_tilt_drag_and_release_reconstruct_one_complete_state(widget, ssb, monkeypatch):
    """Dragging changes the phase; release computes and saves the same full correction."""
    preview = ssb.preview
    calls = []

    def measured_preview(*args, **kwargs):
        calls.append(kwargs)
        return preview(*args, **kwargs)

    monkeypatch.setattr(ssb, "preview", measured_preview)
    sample = {"sample_tilt_mrad": [3.0, -4.0], "depth_spread_nm": 12.0}
    higher = {"C30_nm": 100.0}
    for request_id, committed in enumerate((False, True), start=1):
        widget.request_json = json.dumps({
            "id": request_id, **START, "sample": sample,
            "higher_order": higher, "committed": committed,
        })
        assert len(calls) == request_id
        expected, loss = preview({**START, **higher}, compute_loss=committed, **sample)
        np.testing.assert_array_equal(_shown(widget), expected)
        assert json.loads(widget.result_json)["loss"] == loss
    assert json.loads(widget.sample_json) == sample
    assert json.loads(widget.higher_order_aberrations_json) == higher


def test_launch_table_is_displayed_once_before_widget(widget, monkeypatch):
    """Opening ShowPtycho shows calibration without an extra notebook cell."""
    import IPython.display

    displayed = []
    monkeypatch.setattr(IPython.display, "display", displayed.append)
    bundle, _ = widget._repr_mimebundle_()
    assert "application/vnd.jupyter.widget-view+json" in bundle
    assert len(displayed) == 1
    table = displayed[0].data
    assert list(table.columns) == ["Value used", "Unit", "Source"]
    labels = pytest.importorskip("quantem.gpu.io.metadata_display")  # quantem.gpu's one calibration label set
    assert table.loc[labels.CALIBRATION_LABELS["scan_detector_rotation_deg"], "Source"] == labels.EXPLICIT_OVERRIDE
    widget._repr_mimebundle_()
    assert len(displayed) == 1


def test_launch_table_only_repeats_file_values_when_overridden(widget, monkeypatch):
    """Review an overridden detector step without repeating unchanged calibration."""
    import IPython.display

    displayed = []
    monkeypatch.setattr(IPython.display, "display", displayed.append)
    labels = pytest.importorskip("quantem.gpu.io.metadata_display")  # quantem.gpu's one calibration label set
    scan, step, voltage = (labels.CALIBRATION_LABELS[name] for name in ("scan_sampling_A", "detector_sampling_mrad", "voltage_kV"))
    calibration = widget._launch_calibration
    calibration["File value"] = calibration["Value used"].copy()
    calibration["File unit"] = calibration["Unit used"].copy()
    calibration.at[scan, "Value used"] = (0.5, 0.5)
    calibration.at[scan, "File value"] = (0.5, 0.5)
    calibration.at[step, "File value"] = (0.8, 0.8)
    calibration.at[step, "Value used"] = (0.81, 0.82)
    widget._repr_mimebundle_()
    table = displayed[0].data
    assert list(table.columns) == ["Value used", "Unit", "File value", "Source"]
    assert table.loc[voltage, "File value"] == ""
    assert table.loc[step, "File value"] == (0.8, 0.8)
    html = displayed[0].to_html()
    assert ">0.5</td>" in html and "(0.5, 0.5)" not in html
    assert ">(0.81, 0.82)</td>" in html


def test_automatic_detector_sampling_is_retained_in_saved_calibration(tmp_path):
    """Fit the disk automatically and save the calibration actually used."""
    gpu = pytest.importorskip("quantem.gpu")
    with gpu.SSB(_counts(), voltage_kV=300.0, semiangle_mrad=20.0,
                 scan_sampling_A=(0.5, 0.5), scan_detector_rotation_deg=0.0) as ssb:
        viewer = ShowPtycho(ssb, save_dir=tmp_path)
        sampling = ssb.detector_sampling_mrad
        assert sampling is not None and all(value > 0 for value in sampling)
        labels = pytest.importorskip("quantem.gpu.io.metadata_display")
        assert viewer._launch_calibration.loc[labels.CALIBRATION_LABELS["detector_sampling_mrad"], "Source"] == labels.ESTIMATED
        viewer.save_trigger += 1
        saved = json.loads((tmp_path / "calibration.json").read_text())
        assert tuple(saved["detector_sampling_mrad"]) == sampling


def test_restart_restores_the_complete_saved_full_field_without_refitting(tmp_path, monkeypatch):
    """Save a reviewed correction, reopen the data, and recover identical pixels."""
    gpu = pytest.importorskip("quantem.gpu")
    source = tmp_path / "scan.npy"
    np.save(source, _counts())
    calibration = dict(voltage_kV=300, semiangle_mrad=20,
                       scan_sampling_A=(0.5, 0.5), detector_sampling_mrad=(5, 5))
    with gpu.SSB.open(str(source), **calibration) as ssb:
        viewer = ShowPtycho(ssb, save_dir=tmp_path)
        viewer.upsampling_factor = 2
        viewer.scan_detector_rotation_deg = 12
        viewer.request_json = json.dumps({
            "id": 1, **START, "sample": {"sample_tilt_mrad": [3, -4], "depth_spread_nm": 12},
            "higher_order": {"C30_nm": 2}, "committed": True,
        })
        viewer.display_negate_phase = True
        viewer.notes = "Reviewed full field"
        viewer.local_tilt_max_columns = 2
        expected = _shown(viewer).copy()
        viewer.save_trigger += 1
        viewer.close()

    def no_search(*args, **kwargs):
        raise AssertionError("Restoring a saved model must not run optimization.")

    monkeypatch.setattr(gpu.SSB, "find_aberrations", no_search)
    with gpu.SSB.open(str(source), **calibration) as reopened:
        restored = ShowPtycho(reopened, save_dir=tmp_path)
        np.testing.assert_array_equal(_shown(restored), expected)
        assert restored.calibration_status == "Restored saved full-field fit"
        assert reopened.aberrations == {**START, "C30_nm": 2}
        assert restored.scan_detector_rotation_deg == restored.fitted_scan_detector_rotation_deg == 12
        assert json.loads(restored.sample_json) == json.loads(restored.fitted_sample_json) == {
            "sample_tilt_mrad": [3, -4], "depth_spread_nm": 12,
        }
        assert restored.upsampling_factor == 2 and restored.display_negate_phase
        assert restored.notes == "Reviewed full field"
        assert restored.local_tilt_max_columns == 2
        restored.save_trigger += 1
        restored.close()
    # Saving a restored, rotated model must retain its original launch identity.
    with gpu.SSB.open(str(source), **calibration) as reopened:
        restored = ShowPtycho(reopened, save_dir=tmp_path)
        np.testing.assert_array_equal(_shown(restored), expected)
        restored.close()


def _absolute_paths(value, found=None):
    """Every string in a JSON value that is an absolute path or names the home directory."""
    found = [] if found is None else found
    if isinstance(value, dict):
        for item in value.values():
            _absolute_paths(item, found)
    elif isinstance(value, list):
        for item in value:
            _absolute_paths(item, found)
    elif isinstance(value, str) and (value.startswith("/") or str(Path.home()) in value):
        found.append(value)
    return found


def test_a_saved_fit_restores_after_its_data_folder_moves_and_names_no_absolute_path(tmp_path):
    """calibration.json and the stars name the data by file and relative path and match it by content.

    The folder moved whole (data, calibration, stars) restores the fit; another acquisition saved under
    the same file name is refused.
    """
    import shutil

    gpu = pytest.importorskip("quantem.gpu")
    folder = tmp_path / "session"
    folder.mkdir()
    np.save(folder / "scan.npy", _counts())
    calibration = dict(voltage_kV=300, semiangle_mrad=20, scan_sampling_A=(0.5, 0.5), detector_sampling_mrad=(5, 5))
    with gpu.SSB.open(str(folder / "scan.npy"), **calibration) as ssb:
        viewer = ShowPtycho(ssb, save_dir=folder)
        viewer.upsampling_factor = 1
        viewer.request_json = json.dumps({"id": 1, **START, "committed": True})
        expected = _shown(viewer).copy()
        viewer.save_trigger += 1
        viewer.pin_json = json.dumps({"action": "pin", "id": "a", **START, "loss": 0.1})
        viewer.pin_json = json.dumps({"action": "star", "id": "a"})
        viewer.close()
    saved = json.loads((folder / "calibration.json").read_text())
    stars = json.loads((folder / "showptycho_stars.json").read_text())
    assert saved["schema"] == "quantem.ssb-calibration/6"
    assert saved["source"] == stars["stars"][0]["source"] == {"file_name": "scan.npy", "relative_path": "scan.npy"}
    assert _absolute_paths(saved) == [] and _absolute_paths(stars) == []
    assert saved["session_identity"]["source"]["detector"]["file_name"] == "scan.npy"

    moved = tmp_path / "elsewhere" / "session"
    moved.parent.mkdir()
    shutil.move(folder, moved)
    with gpu.SSB.open(str(moved / "scan.npy"), **calibration) as reopened:
        restored = ShowPtycho(reopened, save_dir=moved)
        assert restored.calibration_status == "Restored saved full-field fit"
        np.testing.assert_array_equal(_shown(restored), expected)
        restored.close()
    np.save(moved / "scan.npy", _counts() + 1)
    with gpu.SSB.open(str(moved / "scan.npy"), **calibration) as other:
        viewer = ShowPtycho(other, save_dir=moved)
        assert "acquisition or launch calibration has changed" in viewer.calibration_status
        viewer.close()


def test_a_flipped_session_records_its_flip_in_every_file_and_restores(tmp_path, monkeypatch):
    """calibration.json, stars, calibration() and the export say the session is flipped, and it reopens."""
    gpu = pytest.importorskip("quantem.gpu")
    if not gpu_adapter.accelerator_ready():
        pytest.skip("Needs CUDA or MPS")
    source = tmp_path / "scan.npy"
    np.save(source, _counts())
    calibration = dict(voltage_kV=300, semiangle_mrad=20, scan_sampling_A=(0.5, 0.5),
                       detector_sampling_mrad=(5, 5), scan_detector_rotation_deg=10.0,
                       scan_detector_rotation_flipped=True)
    with gpu.SSB.open(str(source), aberrations=START, **calibration) as ssb:
        assert ssb.scan_detector_rotation_flipped
        viewer = ShowPtycho(ssb, save_dir=tmp_path)
        viewer.upsampling_factor = 1
        viewer.scan_detector_rotation_deg = 12
        expected = _shown(viewer).copy()
        assert viewer.calibration()["scan_detector_rotation_flipped"] is True
        viewer.save_trigger += 1
        saved = json.loads((tmp_path / "calibration.json").read_text())
        assert saved["scan_detector_rotation_flipped"] is True
        viewer.pin_json = json.dumps({"action": "pin", "id": "a", **START, "loss": 0.1})
        viewer.pin_json = json.dumps({"action": "star", "id": "a"})
        star, = json.loads((tmp_path / "showptycho_stars.json").read_text())["stars"]
        assert star["scan_detector_rotation_flipped"] is True
        out = viewer.export(tmp_path / "review")
        assert json.loads((out / "snapshots" / "cal.json").read_text())["scan_detector_rotation_flipped"] is True
        viewer.close()

    def no_search(*args, **kwargs):
        raise AssertionError("Restoring a saved model must not run optimization.")

    monkeypatch.setattr(gpu.SSB, "find_aberrations", no_search)
    with gpu.SSB.open(str(source), **calibration) as reopened:
        restored = ShowPtycho(reopened, save_dir=tmp_path)
        assert restored.calibration_status == "Restored saved full-field fit"
        assert reopened.scan_detector_rotation_flipped and restored.scan_detector_rotation_deg == 12
        assert restored.upsampling_factor == 1
        np.testing.assert_array_equal(_shown(restored), expected)
        restored.close()
    # The same counts opened unflipped are another geometry: the flipped fit is not reused.
    with gpu.SSB.open(str(source), **{**calibration, "scan_detector_rotation_flipped": False}) as other:
        viewer = ShowPtycho(other, save_dir=tmp_path)
        assert "Saved fit not restored" in viewer.calibration_status
        viewer.close()


def test_a_save_made_after_a_180_degree_move_reopens_at_the_launch_rotation(tmp_path, monkeypatch):
    """The fit's rotation check moved the session to the other branch before ShowPtycho opened.

    The identity stored the moved rotation and compared it exactly, so the next
    session, opened at the launch rotation, never restored the save.
    """
    gpu = pytest.importorskip("quantem.gpu")
    if not gpu_adapter.accelerator_ready():
        pytest.skip("Needs CUDA or MPS")
    from quantem.gpu.geometry import opposite_rotation_deg

    source = tmp_path / "scan.npy"
    np.save(source, _counts())
    calibration = dict(voltage_kV=300, semiangle_mrad=20, scan_sampling_A=(0.5, 0.5), detector_sampling_mrad=(5, 5))
    with gpu.SSB.open(str(source), **calibration) as ssb:
        launch = ssb.scan_detector_rotation_deg  # estimated from the centre-of-mass curl
        moved = opposite_rotation_deg(launch)
        ssb.set_rotation(moved)  # where find_aberrations' 180-degree check leaves it
        viewer = ShowPtycho(ssb, save_dir=tmp_path)
        viewer.upsampling_factor = 1
        viewer.request_json = json.dumps({"id": 1, **START, "committed": True})
        expected = _shown(viewer).copy()
        viewer.save_trigger += 1
        viewer.close()

    def no_search(*args, **kwargs):
        raise AssertionError("Restoring a saved model must not run optimization.")

    monkeypatch.setattr(gpu.SSB, "find_aberrations", no_search)
    with gpu.SSB.open(str(source), **calibration) as reopened:
        assert reopened.scan_detector_rotation_deg == launch
        restored = ShowPtycho(reopened, save_dir=tmp_path)
        assert restored.calibration_status == "Restored saved full-field fit"
        assert restored.scan_detector_rotation_deg == reopened.scan_detector_rotation_deg == pytest.approx(moved)
        np.testing.assert_array_equal(_shown(restored), expected)
        # reconstruct() applies the restored model (find_aberrations would raise here)
        result = reopened.reconstruct()
        assert result.aberrations == START and result.scan_detector_rotation_deg == pytest.approx(moved)
        restored.close()
    # a launch rotation that differs by anything but 180 degrees is another geometry
    with gpu.SSB.open(str(source), **calibration, scan_detector_rotation_deg=launch + 1.0) as other:
        viewer = ShowPtycho(other, save_dir=tmp_path)
        assert "acquisition or launch calibration has changed" in viewer.calibration_status
        viewer.close()


def test_a_scan_region_session_saves_a_calibration_it_reopens(tmp_path, monkeypatch):
    """SSB(data, scan_region=R) saved scan_region in full-scan positions but checked it against [0, rows, 0, cols]."""
    gpu = pytest.importorskip("quantem.gpu")
    if not gpu_adapter.accelerator_ready():
        pytest.skip("Needs CUDA or MPS")
    source = tmp_path / "scan.npy"
    np.save(source, _counts())
    calibration = dict(voltage_kV=300, semiangle_mrad=20, scan_sampling_A=(0.5, 0.5), detector_sampling_mrad=(5, 5),
                       scan_detector_rotation_deg=10.0)
    region = (16, 112, 8, 104)
    with gpu.io.load(str(source)) as data:
        with gpu.SSB(data, scan_region=region, **calibration) as ssb:
            assert ssb.input_scan_shape == (96, 96)
            viewer = ShowPtycho(ssb, save_dir=tmp_path)
            viewer.upsampling_factor = 1
            viewer.request_json = json.dumps({"id": 1, **START, "committed": True})
            expected = _shown(viewer).copy()
            record = viewer.calibration()
            assert (record["scan_region"], record["scan_shape"]) == (list(region), [96, 96])
            viewer.save_trigger += 1
            viewer.close()

    def no_search(*args, **kwargs):
        raise AssertionError("Restoring a saved model must not run optimization.")

    monkeypatch.setattr(gpu.SSB, "find_aberrations", no_search)
    with gpu.io.load(str(source)) as data:
        with gpu.SSB(data, scan_region=region, **calibration) as reopened:
            restored = ShowPtycho(reopened, save_dir=tmp_path)
            assert restored.calibration_status == "Restored saved fit of scan region [16, 112, 8, 104]"
            assert reopened.aberrations == START
            np.testing.assert_array_equal(_shown(restored), expected)
            assert reopened.reconstruct().aberrations == START  # no refit: find_aberrations would raise
            restored.close()
        # the whole scan, or another region, is another session
        with gpu.SSB(data, **calibration) as whole:
            viewer = ShowPtycho(whole, save_dir=tmp_path)
            assert "Saved fit not restored" in viewer.calibration_status
            viewer.close()


def test_the_propagator_tilt_of_a_flipped_session_follows_the_detector_axes(tmp_path):
    """propagator_frame_tilt rotated the scan-frame tilt but ignored the flip, so a flipped session saved the
    detector-axis components in the wrong order."""
    gpu = pytest.importorskip("quantem.gpu")
    if not gpu_adapter.accelerator_ready():
        pytest.skip("Needs CUDA or MPS")
    sample = {"sample_tilt_mrad": [3.0, -4.0], "depth_spread_nm": 12.0}
    rotation = math.radians(10.0)
    unflipped = [3.0 * math.cos(rotation) - 4.0 * math.sin(rotation), -3.0 * math.sin(rotation) - 4.0 * math.cos(rotation)]
    for flipped, expected in ((False, unflipped), (True, unflipped[::-1])):
        with gpu.SSB(_counts(), voltage_kV=300.0, semiangle_mrad=20.0, scan_sampling_A=(0.5, 0.5),
                     detector_sampling_mrad=(5.0, 5.0), aberrations=dict(START), scan_detector_rotation_deg=10.0,
                     scan_detector_rotation_flipped=flipped) as ssb:
            viewer = ShowPtycho(ssb, save_dir=tmp_path)
            viewer.upsampling_factor = 1
            viewer.request_json = json.dumps({"id": 1, **START, "sample": sample, "committed": True})
            assert viewer.calibration()["propagator_tilt_mrad"] == pytest.approx(expected)
            viewer.save_trigger += 1
            saved = json.loads((tmp_path / "calibration.json").read_text())
            assert saved["scan_detector_rotation_flipped"] is flipped
            assert saved["propagator_tilt_mrad"] == pytest.approx(expected)
            viewer.close()


def test_calibration_returns_the_record_save_writes(tmp_path):
    """calibration() and calibration.json shared a schema but not their keys: calibration() lacked
    upsampling_mode, display and id, and only it held scan_shape and bright_field_radius_px."""
    gpu = pytest.importorskip("quantem.gpu")
    if not gpu_adapter.accelerator_ready():
        pytest.skip("Needs CUDA or MPS")
    from quantem.widget.showptycho.calibration import read_calibration

    source = tmp_path / "scan.npy"
    np.save(source, _counts())
    with gpu.SSB.open(str(source), voltage_kV=300, semiangle_mrad=20, scan_sampling_A=(0.5, 0.5),
                      detector_sampling_mrad=(5, 5), aberrations=START, scan_detector_rotation_deg=10.0) as ssb:
        viewer = ShowPtycho(ssb, save_dir=tmp_path)
        viewer.request_json = json.dumps({"id": 1, **START, "sample": {"sample_tilt_mrad": [3, -4], "depth_spread_nm": 12},
                                          "committed": True})
        record = viewer.calibration()
        viewer.save_trigger += 1
        saved = json.loads((tmp_path / "calibration.json").read_text())
        assert set(record) == set(saved)
        assert (record["id"], record["generated_at_utc"]) == (None, None) and saved["id"] and saved["generated_at_utc"]
        assert {key: value for key, value in record.items() if key not in ("id", "generated_at_utc")} == {
            key: value for key, value in saved.items() if key not in ("id", "generated_at_utc")}
        assert (saved["scan_shape"], saved["upsampling_mode"]) == ([128, 128], "auto")
        assert saved["bright_field_radius_px"] == ssb.bright_field_radius_px
        # a downstream reader takes the snapshot exactly as it takes the file
        snapshot = tmp_path / "snapshot.json"
        snapshot.write_text(json.dumps(record))
        read_calibration(snapshot, viewer._calibration_identity, (0, 128, 0, 128), None, record["schema"])
        viewer.close()


def test_the_tutorial_order_reopens_the_save_and_reconstruct_keeps_it(tmp_path, monkeypatch, capsys):
    """SSB(data); ssb.reconstruct(); ssb.show(save_dir=d), then the same cells after a restart.

    reconstruct() fits the new session; show() restores the save over that fit and says so, and
    a later reconstruct() applies the restored values instead of fitting again. A session that only
    calls show() restores too, and so does one whose Save stored the values without a fit.
    """
    gpu = pytest.importorskip("quantem.gpu")
    if not gpu_adapter.accelerator_ready():
        pytest.skip("Needs CUDA or MPS")
    source = tmp_path / "scan.npy"
    np.save(source, _counts())
    calibration = dict(voltage_kV=300, semiangle_mrad=20, scan_sampling_A=(0.5, 0.5), detector_sampling_mrad=(5, 5))
    reviewed = {"C10_nm": 3.0, "C12_nm": 1.0, "phi12_deg": 11.0}
    with gpu.io.load(str(source)) as data, gpu.SSB(data, **calibration) as ssb:
        fitted = dict(ssb.reconstruct(verbose=False).aberrations)
        viewer = ssb.show(save_dir=tmp_path)
        viewer.upsampling_factor = 1
        viewer.request_json = json.dumps({"id": 1, **reviewed, "committed": True})
        expected = _shown(viewer).copy()
        viewer.save_trigger += 1
        viewer.close()
    with gpu.io.load(str(source)) as data, gpu.SSB(data, **calibration) as ssb:
        assert dict(ssb.reconstruct(verbose=False).aberrations) == fitted  # the restart fits again
        searches = []
        monkeypatch.setattr(gpu.SSB, "find_aberrations", lambda *args, **kwargs: searches.append(1))
        capsys.readouterr()
        viewer = ssb.show(save_dir=tmp_path)
        assert viewer.calibration_status == "Restored saved full-field fit"
        assert "replacing the session's C10" in capsys.readouterr().out
        np.testing.assert_array_equal(_shown(viewer), expected)
        assert dict(ssb.reconstruct().aberrations) == reviewed and not searches
        viewer.close()
    with gpu.io.load(str(source)) as data, gpu.SSB(data, **calibration) as ssb:
        viewer = ssb.show(save_dir=tmp_path)  # no reconstruct() first
        assert viewer.calibration_status == "Restored saved full-field fit"
        assert dict(ssb.reconstruct().aberrations) == reviewed and not searches
        viewer.close()
    (tmp_path / "calibration.json").unlink()
    with gpu.io.load(str(source)) as data, gpu.SSB(data, **calibration) as ssb:
        viewer = ssb.show(save_dir=tmp_path)  # nothing saved yet, nothing fitted
        viewer.request_json = json.dumps({"id": 2, **reviewed, "committed": True})
        viewer.save_trigger += 1
        assert dict(ssb.reconstruct().aberrations) == reviewed and not searches  # Save stated them as known
        viewer.close()


def test_changed_acquisition_or_calibration_does_not_restore_an_old_fit(tmp_path):
    """A revised detector scale or replaced acquisition starts its own analysis."""
    gpu = pytest.importorskip("quantem.gpu")
    source = tmp_path / "scan.npy"
    np.save(source, _counts())
    calibration = dict(voltage_kV=300, semiangle_mrad=20,
                       scan_sampling_A=(0.5, 0.5), detector_sampling_mrad=(5, 5))
    with gpu.SSB.open(str(source), aberrations=START, **calibration) as ssb:
        viewer = ShowPtycho(ssb, save_dir=tmp_path)
        viewer.save_trigger += 1
        viewer.close()
    for changed in (True, False):
        settings = {**calibration, "detector_sampling_mrad": (5.1, 5.1)} if changed else calibration
        if not changed:
            np.save(source, _counts() + 1)
        with gpu.SSB.open(str(source), **settings) as ssb:
            viewer = ShowPtycho(ssb, save_dir=tmp_path)
            assert viewer.fitted_C10_nm == 0
            assert "acquisition or launch calibration has changed" in viewer.calibration_status
            assert "Joint fit" in viewer.calibration_status
            viewer.close()


def test_a_matching_save_replaces_a_supplied_model_unless_restore_is_false(tmp_path, capsys):
    """reconstruct() then show() is the documented reopen: the save wins over the fresh fit, and says so.

    A fitted or manually initialized session skipped restore, so that order refit and never restored.
    """
    gpu = pytest.importorskip("quantem.gpu")
    source = tmp_path / "scan.npy"
    np.save(source, _counts())
    calibration = dict(voltage_kV=300, semiangle_mrad=20,
                       scan_sampling_A=(0.5, 0.5), detector_sampling_mrad=(5, 5))
    with gpu.SSB.open(str(source), aberrations=START, **calibration) as ssb:
        viewer = ShowPtycho(ssb, save_dir=tmp_path)
        viewer.save_trigger += 1
        viewer.close()
    with gpu.SSB.open(str(source), aberrations={**START, "C10_nm": 7}, **calibration) as ssb:
        capsys.readouterr()
        viewer = ShowPtycho(ssb, save_dir=tmp_path)
        assert viewer.fitted_C10_nm == 3 and viewer.calibration_status == "Restored saved full-field fit"
        assert ssb.aberrations == START
        printed = capsys.readouterr().out.strip().splitlines()
        assert len(printed) == 1 and printed[0].startswith("ShowPtycho: restored calibration.json saved ")
        assert re.search(r"\(C10 3\.00 nm, C12 1\.00 nm, phi12 11\.5°, rotation [0-9.]+°\), "
                         r"replacing the session's C10 7\.00 nm, C12 1\.00 nm", printed[0])
        assert printed[0].endswith("Pass restore=False to keep the session's values.")
        viewer.close()
    with gpu.SSB.open(str(source), aberrations={**START, "C10_nm": 7}, **calibration) as ssb:
        viewer = ShowPtycho(ssb, save_dir=tmp_path, restore=False)
        assert viewer.fitted_C10_nm == 7 and viewer.calibration_status == ""
        viewer.close()


@pytest.mark.parametrize("saved_kind", ["crop", "older save"])
def test_full_field_launch_does_not_reinterpret_a_crop_or_unidentified_save(tmp_path, saved_kind):
    """A full-field notebook explains why an incompatible project cannot reopen."""
    gpu = pytest.importorskip("quantem.gpu")
    source = tmp_path / "scan.npy"
    np.save(source, _counts())
    calibration = dict(voltage_kV=300, semiangle_mrad=20,
                       scan_sampling_A=(0.5, 0.5), detector_sampling_mrad=(5, 5))
    with gpu.SSB.open(str(source), aberrations=START, **calibration) as ssb:
        viewer = ShowPtycho(ssb, save_dir=tmp_path)
        viewer.save_trigger += 1
        viewer.close()
    path = tmp_path / "calibration.json"
    saved = json.loads(path.read_text())
    if saved_kind == "crop":
        saved["scan_region"] = [0, 64, 0, 64]
    else:
        saved.pop("session_identity")
    path.write_text(json.dumps(saved))
    with gpu.SSB.open(str(source), **calibration) as ssb:
        viewer = ShowPtycho(ssb, save_dir=tmp_path)
        assert "Saved fit not restored" in viewer.calibration_status
        assert viewer.scan_rows == viewer.scan_cols == SCAN
        assert viewer.fitted_C10_nm == 0
        viewer.close()


def test_calibration_snapshot_selects_current_joint_region_independently(ssb, tmp_path):
    """A downstream cell receives committed regional controls, not the active tab."""
    fits = [
        {"id": "a", "label": "Upper", "bounds": [0, 64, 0, 64],
         "aberrations": dict(START), "sample_tilt_mrad": [4., -2.], "depth_spread_nm": 12.},
        {"id": "b", "label": "Lower", "bounds": [64, 128, 64, 128],
         "aberrations": dict(START, C10_nm=7.), "sample_tilt_mrad": [9., 3.], "depth_spread_nm": 20.},
    ]
    regions = [{key: fit[key] for key in ("id", "label", "bounds")} for fit in fits]
    viewer = ShowPtycho(ssb, save_dir=tmp_path, local_tilt={
        "scan_shape": [128, 128], "regions": regions, "fits": fits})
    overview = viewer.calibration()
    lower = viewer.calibration("Lower")
    viewer.local_tilt_active_region = "b"
    viewer.request_json = json.dumps({
        "id": 701, "region_id": "a", "region_bounds": fits[0]["bounds"],
        **START, "C10_nm": 5., "sample": {
            "sample_tilt_mrad": [4., -2.], "depth_spread_nm": 12.}, "committed": True})
    upper = viewer.calibration("a")
    assert upper["aberrations"]["C10_nm"] == 5.
    assert upper["sample_tilt_mrad"] == [4., -2.]
    assert upper["region_fit"]["adjusted"]
    # the overview model is unchanged; its regional views (local_tilt, as Save writes them) follow the region
    def model(record: dict) -> dict:
        return {key: value for key, value in record.items() if key != "local_tilt"}

    current = viewer.calibration()
    assert model(current) == model(overview)
    assert current["local_tilt"]["fits"][0]["aberrations"]["C10_nm"] == 5.
    assert model(viewer.calibration("b")) == model(lower)
    expected, _ = ssb.preview(
        upper["aberrations"], sample_tilt_mrad=upper["sample_tilt_mrad"],
        depth_spread_nm=upper["depth_spread_nm"],
        upsampling_factor=upper["upsampling_factor"],
        phase_estimator=upper["phase_estimator"], compute_loss=False,
    )
    viewer.tilt_correction = False
    snapshot = viewer.calibration("a", include_phase=True)
    np.testing.assert_array_equal(snapshot["ssb_phase"], expected)
    snapshot["ssb_phase"][:] = 0
    np.testing.assert_array_equal(
        viewer.calibration("a", include_phase=True)["ssb_phase"], expected,
    )
    assert "ssb_phase" not in viewer.calibration("a")
    upper["aberrations"]["C10_nm"] = 999
    upper["region_fit"]["bounds"][0] = 30
    assert viewer.calibration("a")["aberrations"]["C10_nm"] == 5.
    assert viewer.regions[0]["bounds"][0] == 0
    viewer.regions = [{**regions[0], "bounds": [8, 72, 0, 64]}, regions[1]]
    with pytest.raises(ValueError, match="unfitted"):
        viewer.calibration("a")
    viewer.close()


def _fitted_regions():
    """Two regions with supplied joint fits, as a restored workspace holds them."""
    fits = [
        {"id": "a", "label": "Upper", "bounds": [0, 64, 0, 64],
         "aberrations": dict(START), "sample_tilt_mrad": [4., -2.], "depth_spread_nm": 12.},
        {"id": "b", "label": "Lower", "bounds": [64, 128, 64, 128],
         "aberrations": dict(START, C10_nm=7.), "sample_tilt_mrad": [9., 3.], "depth_spread_nm": 20.},
    ]
    regions = [{key: fit[key] for key in ("id", "label", "bounds")} for fit in fits]
    return {"scan_shape": [SCAN, SCAN], "regions": regions, "fits": fits}


def test_a_scan_too_coarse_for_automatic_sampling_opens_at_native_sampling(capsys):
    """3 Å steps at 300 kV and 20 mrad need about 12x output; the viewer still opens."""
    _session()  # skips without quantem.gpu or an accelerator
    from quantem.gpu import SSB

    coarse = SSB(_counts(), voltage_kV=300.0, semiangle_mrad=20.0, scan_sampling_A=(3.0, 3.0),
                 detector_sampling_mrad=(5.0, 5.0), aberrations=dict(START))
    viewer = ShowPtycho(coarse)
    assert viewer.upsampling_factor == 1 and _shown(viewer).shape == (SCAN, SCAN)
    assert "more than 8" in viewer.auto_sampling_note and "upsampling_factor=1" in viewer.auto_sampling_note
    assert capsys.readouterr().out.count("Automatic SSB output sampling is unavailable") == 1
    with pytest.raises(ValueError, match="more than 8"):
        viewer.upsampling_factor = "auto"
    viewer.upsampling_factor = 2  # an explicit, limited output remains available
    assert _shown(viewer).shape == (2 * SCAN, 2 * SCAN)
    viewer.close()


def test_output_sampling_changes_make_region_images_stale(ssb, tmp_path):
    """Region images keep one sampling_A; another factor would mix two grids."""
    ssb.aberrations, ssb.best_loss = dict(START), float("inf")
    viewer = ShowPtycho(ssb, save_dir=tmp_path, local_tilt=_fitted_regions())
    assert viewer.upsampling_factor == "auto" and not viewer.local_tilt_stale
    viewer.upsampling_factor = 1
    assert viewer.local_tilt_stale
    with pytest.raises(ValueError, match="output sampling"):
        viewer.region_fits
    viewer.upsampling_factor = "auto"
    assert not viewer.local_tilt_stale
    viewer.close()


def test_calibration_loss_is_the_saved_accepted_model_loss(widget):
    """Neither the zero-tilt comparison nor a drag without a loss changes the handoff loss."""
    widget.sample_json = json.dumps({"sample_tilt_mrad": [3.0, -4.0], "depth_spread_nm": 12.0})
    widget.tilt_correction = False
    widget.request_json = json.dumps({"id": 40, **START, "C10_nm": 2.5, "committed": False})
    assert json.loads(widget.result_json)["loss"] is None
    record = widget.calibration()
    widget.save_trigger += 1
    saved = json.loads(Path(widget.calibration_path).read_text())
    assert record["loss"] is not None and record["loss"] == saved["loss"]


def test_freezing_detaches_previews_and_regional_fits(ssb, tmp_path):
    ssb.aberrations, ssb.best_loss = dict(START), float("inf")
    viewer = ShowPtycho(ssb, save_dir=tmp_path, local_tilt=_fitted_regions())
    viewer.freeze()
    assert viewer._on_preview_message not in viewer._msg_callbacks.callbacks
    phase = viewer.phase_bytes
    viewer._handle_custom_msg({"type": "ssb-preview", "request": {"id": 50, **START, "committed": True}}, [])
    assert viewer.phase_bytes == phase
    viewer.scan_detector_rotation_deg = 30.0
    assert not viewer.local_tilt_stale
    assert not viewer.local_tilt_can_fit and not viewer.local_tilt_can_reconstruct
    viewer.local_tilt_request_json = json.dumps({"id": 51, "action": "fit", "regions": viewer.regions})
    reply = json.loads(viewer.local_tilt_reply_json)
    assert reply["status"] == "error" and "no live callback" in reply["message"]
    viewer.close()


def test_zero_higher_order_magnitudes_leave_the_three_term_model(widget):
    """An azimuth without its magnitude is not a correction and must not block Fit Regions."""
    widget.higher_order_aberrations_json = json.dumps({"C21_nm": 0.0, "phi21_deg": 45.0, "C30_nm": 0.0})
    assert set(widget.ssb_parameters["aberrations"]) == set(START)
    widget.higher_order_aberrations_json = json.dumps({"C21_nm": 5.0, "phi21_deg": 45.0, "C30_nm": 0.0})
    assert set(widget.ssb_parameters["aberrations"]) == {*START, "C21_nm", "phi21_deg"}


def test_the_trial_strip_lists_the_default_fits_trials_with_their_objective():
    """The strip kept only phase-variance trials, so the default fit (model agreement) showed none.

    Every trial is listed with quantem.gpu's name for its objective and its own sample model; losses rank within
    one objective (an agreement trial's loss is its negative agreement), best first.
    """
    from types import SimpleNamespace

    from quantem.widget.adapters.gpu import SsbSession

    def record(C10_nm, loss, objective, **sample):
        return {"params": {"C10_nm": C10_nm, "C12_nm": 1.0, "phi12_deg": 20.0}, "loss": loss,
                "objective": objective, **sample}

    session = SsbSession.__new__(SsbSession)
    session.ssb = SimpleNamespace(trial_history=[
        record(4.0, -5.0, "negative_thick_agreement", sample_tilt_mrad=[1.0, -2.0], depth_spread_nm=8.0),
        record(5.0, -9.0, "negative_thick_agreement", sample_tilt_mrad=[0.5, -1.5], depth_spread_nm=6.0),
        record(6.0, float("nan"), "negative_thick_agreement", sample_tilt_mrad=[0.0, 0.0], depth_spread_nm=0.0),
        {"params": {"C10_nm": 7.0, "C12_nm": 2.0, "phi12_deg": 30.0}, "loss": 0.3},  # a record before objectives
    ])
    trials = json.loads(ShowPtycho._trials_payload(SimpleNamespace(_ssb=session)))
    assert [(trial["rank"], trial["C10_nm"], trial["objective"]) for trial in trials] == [
        (0, 5.0, "agreement"), (1, 4.0, "agreement"), (2, 7.0, "phase_variance")]
    assert (trials[0]["sample_tilt_mrad"], trials[0]["depth_spread_nm"], trials[0]["loss"]) == ([0.5, -1.5], 6.0, -9.0)
    assert (trials[2]["sample_tilt_mrad"], trials[2]["depth_spread_nm"]) == ([0.0, 0.0], 0.0)
