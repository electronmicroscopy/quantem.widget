"""Slider drags in ShowPtycho use quantem.gpu's drag windows; releases stay exact.

A drag frame from a window is ``preview``'s image to a few float32 ulps of the
largest phase (quantem.gpu tests/hardware/cuda/test_direct_drag_window.py:
below 4e-7 measured, tolerance 2e-6 there and here). Frames before a window is
built, and every release, are ``preview`` itself, bit for bit.
"""

import json

import numpy as np
import pytest
import torch

from quantem.widget import ShowPtycho

ABERRATIONS = {"C10_nm": 12.0, "C12_nm": 3.0, "phi12_deg": 40.0}
SAMPLE = {"sample_tilt_mrad": (1.5, -0.8), "depth_spread_nm": 8.0}


@pytest.fixture(scope="module")
def ptycho():
    """A 128 x 128 scan of a weak, varying specimen on a 64 x 64 detector, on CUDA."""
    gpu = pytest.importorskip("quantem.gpu")
    if not torch.cuda.is_available():
        pytest.skip("drag windows need CUDA")
    import cupy as cp

    rng = np.random.default_rng(7)
    row, col = cp.indices((64, 64))
    disk = ((row - 31.5) ** 2 + (col - 31.5) ** 2 < 20.5**2).astype(cp.float32)
    scan_row, scan_col = cp.indices((128, 128), dtype=cp.float32)
    pattern = 1.0
    for wave_row, wave_col in rng.normal(size=(6, 2)):
        pattern = pattern + 0.02 * cp.cos(0.3 * (wave_row * scan_row + wave_col * scan_col))[:, :, None, None] * cp.cos(
            0.2 * (wave_row * row + wave_col * col)
        )
    counts = cp.random.poisson(400 * disk * pattern).astype(cp.float32)
    session = gpu.DirectPtycho(
        counts, backend="cuda", voltage_kV=300.0, semiangle_mrad=25.0, scan_sampling_A=0.4,
        detector_sampling_mrad=1.25, bright_field_center_px=(31.5, 31.5), bright_field_radius_px=20.5,
        scan_detector_rotation_deg=0.0, aberrations=ABERRATIONS,
    )
    yield session
    session.close()


def _shown(widget) -> np.ndarray:
    return np.frombuffer(widget.phase_bytes, np.float32).reshape(widget.phase_height, widget.phase_width)


@pytest.mark.parametrize("method", ["phase", "phase_of_mean"])
def test_one_slider_drag_uses_its_window_and_a_release_prepares_the_next(ptycho, tmp_path, method):
    viewer = ShowPtycho(ptycho, method=method, save_dir=tmp_path, restore=False)
    viewer.upsample = 3
    viewer.sample_json = json.dumps({"sample_tilt_mrad": list(SAMPLE["sample_tilt_mrad"]),
                                     "depth_spread_nm": SAMPLE["depth_spread_nm"]})
    strengths = {"phase_damp": 0.03} if method == "phase" else {}
    expected_options = dict(upsample=3, method=method, compute_loss=False, **SAMPLE, **strengths)

    def request(index, C10_nm, committed=False):
        viewer._on_request({"id": index, **ABERRATIONS, "C10_nm": C10_nm, "method": method,
                                               **strengths, "committed": committed})
        return ptycho.preview({**ABERRATIONS, "C10_nm": C10_nm}, **expected_options)[0]

    ptycho._direct_windows = {}
    # the first frame has nothing to compare with; the second moves C10 before its window exists
    for index, value in enumerate((12.0, 12.05), start=1):
        expected = request(index, value)
        np.testing.assert_array_equal(_shown(viewer), expected)
    while not viewer._direct.advance():
        pass
    expected = request(3, 12.2)
    assert np.abs(_shown(viewer) - expected).max() <= 2e-6 * np.abs(expected).max()
    assert ptycho._direct_windows["C10_nm"][1][0].series.ready
    expected = request(4, 12.2, committed=True)
    np.testing.assert_array_equal(_shown(viewer), expected)
    assert set(ptycho._direct_windows) == {"C10_nm", "C12_nm", "phi12_deg"}


def test_a_rotation_drag_never_touches_the_drag_windows(ptycho, tmp_path, monkeypatch):
    viewer = ShowPtycho(ptycho, save_dir=tmp_path, restore=False)
    viewer.upsample = 3

    def request(index, rotation, C10_nm=12.0, committed=False):
        viewer._on_request({"id": index, **ABERRATIONS, "C10_nm": C10_nm,
                                               "scan_detector_rotation_deg": rotation, "committed": committed})

    request(1, 0.0, committed=True)
    request(2, 0.0, C10_nm=12.05)                                          # an unreleased C10 drag
    calls = []
    for name in ("drag", "advance", "prepare"):
        monkeypatch.setattr(viewer._direct, name, lambda *args, _name=name, **kwargs: calls.append(_name))
    for index, rotation in enumerate((0.5, 1.0, 1.5), start=3):            # then the rotation slider
        request(index, rotation, C10_nm=12.05)
        expected = ptycho.preview({**ABERRATIONS, "C10_nm": 12.05}, upsample=3, compute_loss=False)[0]
        np.testing.assert_array_equal(_shown(viewer), expected)
    assert calls == []
    request(6, 1.5, C10_nm=12.05, committed=True)                          # its release prepares the next grab
    assert calls == ["prepare"]
    viewer.close()
