from __future__ import annotations

import json
import math
import pathlib
import re
import sys
import types
from dataclasses import replace

import numpy as np

from quantem.gpu import SSB


def _webgpu_source(name: str) -> str:
    root = pathlib.Path(__file__).resolve().parents[1]
    return (root / "js" / ".generated" / "engine" / name).read_text(
        encoding="utf-8"
    )


class _FakeCuPy(types.SimpleNamespace):
    ndarray = np.ndarray
    float32 = np.float32
    int64 = np.int64
    complex64 = np.complex64

    @staticmethod
    def zeros(shape, dtype=np.float32):
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def empty(shape, dtype=np.float32):
        return np.empty(shape, dtype=dtype)

    @staticmethod
    def arange(*args, **kwargs):
        return np.arange(*args, **kwargs)

    @staticmethod
    def ascontiguousarray(array):
        return np.ascontiguousarray(array)

    @staticmethod
    def asnumpy(array):
        return np.asarray(array)


class _FakeAccel:
    backend = "cuda"

    def __init__(self):
        self._cache = {
            "num_bf": 8,
            "ny": 3,
            "nx": 4,
            "kx_bf": np.linspace(-1.0, 1.0, 8, dtype=np.float32),
            "ky_bf": np.linspace(1.0, -1.0, 8, dtype=np.float32),
            "qx_1d": np.linspace(-0.5, 0.5, 4, dtype=np.float32),
            "qy_1d": np.linspace(-0.5, 0.5, 3, dtype=np.float32),
            "aperture_k_1d": np.ones(8, dtype=np.float32),
            "alpha_k2_1d": np.ones(8, dtype=np.float32),
            "cos2phi_k_1d": np.zeros(8, dtype=np.float32),
            "sin2phi_k_1d": np.zeros(8, dtype=np.float32),
            "semiangle_rad": np.float32(0.0219),
            "ang_y_rad": np.float32(0.001),
            "ang_x_rad": np.float32(0.001),
        }
        self._rotations: list[float] = []
        self._pk_buffer = None
        self._result_buffer = None
        self._mean_phase_buffer = None
        self._sumsq_buffer = None
        self.G_qk = np.ones((8, 3, 4), dtype=np.complex64)
        self.gpts = (16, 16)
        self.bf_center = (7.5, 7.5)
        self.bf_inds_row = np.arange(8, dtype=np.int32)
        self.bf_inds_col = np.arange(8, dtype=np.int32)
        self.wavelength = np.float32(0.025)
        self.sampling = (np.float32(0.5), np.float32(0.5))
        self._dc_value_host = np.complex64(1.0 + 0.0j)
        self.reconstruct_calls: list[tuple[str, float, float, float]] = []

    @property
    def scan_shape(self):
        return int(self._cache["ny"]), int(self._cache["nx"])

    @property
    def detector_shape(self):
        return self.gpts

    @property
    def num_bf(self):
        return int(self._cache["num_bf"])

    def preview_context(self, num_bf: int):
        return _FakePreparedSubset(num_bf)

    @staticmethod
    def phase_to_numpy(phase):
        return np.asarray(phase, dtype=np.float32)

    def browser_state(self):
        from quantem.gpu.ssb.compute.protocol import SSBExportState
        from quantem.gpu.ssb.bf_selector import BrightfieldDisk

        selection = BrightfieldDisk(
            rows=self.bf_inds_row,
            cols=self.bf_inds_col,
            center_row_col=self.bf_center,
            radius_px=2.0,
            detected_radius_px=2.0,
            detector_shape=self.detector_shape,
        )
        return SSBExportState(
            backend=self.backend,
            scan_shape=self.scan_shape,
            brightfield=selection,
            kx_bf=self._cache["kx_bf"],
            ky_bf=self._cache["ky_bf"],
            qx_1d=self._cache["qx_1d"],
            qy_1d=self._cache["qy_1d"],
            aperture_k=self._cache["aperture_k_1d"],
            alpha_k2=self._cache["alpha_k2_1d"],
            cos2phi_k=self._cache["cos2phi_k_1d"],
            sin2phi_k=self._cache["sin2phi_k_1d"],
            wavelength_A=float(self.wavelength),
            semiangle_rad=float(self._cache["semiangle_rad"]),
            angular_sampling_rad=(
                float(self._cache["ang_y_rad"]),
                float(self._cache["ang_x_rad"]),
            ),
            sampling_A=tuple(float(value) for value in self.sampling),
            dc_value=complex(self._dc_value_host),
        )

    def cache_rotation(self, rotation_rad: float) -> None:
        self._rotations.append(float(rotation_rad))

    def reconstruct_with_loss(self, c10: float, c12: float, phi12: float):
        self.reconstruct_calls.append(("loss", float(c10), float(c12), float(phi12)))
        value = np.float32(c10 + c12 + phi12)
        phase = np.full((3, 4), value, dtype=np.float32)
        self._mean_phase_buffer = phase
        self._sumsq_buffer = (phase ** 2 + np.float32(0.25)) * self._cache["num_bf"]
        return phase, 0.125

    def reconstruct(self, c10: float, c12: float, phi12: float):
        self.reconstruct_calls.append(("phase", float(c10), float(c12), float(phi12)))
        value = np.float32(10.0 + c10 + c12 + phi12)
        return np.full((3, 4), value, dtype=np.float32)

    def reconstruct_full_with_loss(self, mags_m, angles_rad):
        self.reconstruct_calls.append(("full_loss", float(np.asarray(mags_m)[0]), float(np.asarray(mags_m)[1]), float(np.asarray(angles_rad)[1])))
        value = np.float32(np.asarray(mags_m).sum() + np.asarray(angles_rad).sum())
        return np.full((3, 4), value, dtype=np.float32), 0.25

    def reconstruct_full(self, mags_m, angles_rad):
        self.reconstruct_calls.append(("full_phase", float(np.asarray(mags_m)[0]), float(np.asarray(mags_m)[1]), float(np.asarray(angles_rad)[1])))
        value = np.float32(20.0 + np.asarray(mags_m).sum() + np.asarray(angles_rad).sum())
        return np.full((3, 4), value, dtype=np.float32)

    def preview(
        self,
        aberrations,
        *,
        compute_loss,
        higher_order_magnitudes,
        higher_order_angles,
    ):
        if higher_order_magnitudes is not None:
            if compute_loss:
                return self.reconstruct_full_with_loss(
                    higher_order_magnitudes, higher_order_angles
                )
            return self.reconstruct_full(
                higher_order_magnitudes, higher_order_angles
            ), None
        if compute_loss:
            return self.reconstruct_with_loss(
                aberrations["C10"], aberrations["C12"], aberrations["phi12"]
            )
        return self.reconstruct(
            aberrations["C10"], aberrations["C12"], aberrations["phi12"]
        ), None

    def reconstruct_object(self, c10: float, c12: float, phi12: float):
        value = np.complex64(1.0 + 0.5j + np.float32(c10 + c12 + phi12) * 0j)
        return np.full((3, 4), value, dtype=np.complex64)


class _FakeMpsAccel(_FakeAccel):
    backend = "mps"
    capabilities = ()


class _FakePreparedSubset:
    def __init__(self, num_bf: int):
        self.num_bf = int(num_bf)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def close(self):
        return None


class _FakeWebGPUAccel(_FakeAccel):
    def __init__(self, n: int = 128):
        super().__init__()
        self._cache.update(
            {
                "num_bf": 1,
                "ny": n,
                "nx": n,
                "kx_bf": np.array([0.0], dtype=np.float32),
                "ky_bf": np.array([0.0], dtype=np.float32),
                "qx_1d": np.linspace(-0.5, 0.5, n, dtype=np.float32),
                "qy_1d": np.linspace(-0.5, 0.5, n, dtype=np.float32),
                "aperture_k_1d": np.ones(1, dtype=np.float32),
                "alpha_k2_1d": np.ones(1, dtype=np.float32),
                "cos2phi_k_1d": np.ones(1, dtype=np.float32),
                "sin2phi_k_1d": np.zeros(1, dtype=np.float32),
            }
        )
        self.G_qk = np.zeros((1, n, n), dtype=np.complex64)
        self.bf_inds_row = np.array([64], dtype=np.int32)
        self.bf_inds_col = np.array([64], dtype=np.int32)
        self.gpts = (192, 192)
        self.bf_center = (96.0, 96.0)
        self._n = n

    def reconstruct_with_loss(self, c10: float, c12: float, phi12: float):
        phase = np.zeros((self._n, self._n), dtype=np.float32)
        self._mean_phase_buffer = phase
        self._sumsq_buffer = phase
        return phase, 0.125

    def reconstruct(self, c10: float, c12: float, phi12: float):
        return np.zeros((self._n, self._n), dtype=np.float32)


class _FakeColumnWebGPUAccel(_FakeWebGPUAccel):
    def __init__(self, n: int = 128):
        super().__init__(n)
        self._cache.update(
            {
                "num_bf": 2,
                "kx_bf": np.array([0.0, 0.1], dtype=np.float32),
                "ky_bf": np.array([0.0, -0.1], dtype=np.float32),
                "aperture_k_1d": np.ones(2, dtype=np.float32),
                "alpha_k2_1d": np.ones(2, dtype=np.float32),
                "cos2phi_k_1d": np.ones(2, dtype=np.float32),
                "sin2phi_k_1d": np.zeros(2, dtype=np.float32),
            }
        )
        self.G_qk = np.zeros((2, n, n), dtype=np.complex64)
        self.bf_inds_row = np.array([1, 2], dtype=np.int32)
        self.bf_inds_col = np.array([2, 1], dtype=np.int32)
        self.gpts = (4, 4)
        self.bf_center = (1.5, 1.5)


class _FakeMetadataOnlyMpsAccel(_FakeColumnWebGPUAccel):
    backend = "mps"

    def __init__(self, n: int = 128):
        super().__init__(n)
        self._frames = object()
        self.g_shape = tuple(int(v) for v in self.G_qk.shape)
        delattr(self, "G_qk")
        self.heavy_sync_calls = 0

    def _sync_webgpu_export_state(self):
        self.heavy_sync_calls += 1
        raise AssertionError("compressed-source folder export should not sync G_qk")

    def browser_state(self):
        return super().browser_state()


class _FakeSSB(SSB):
    def __init__(self, accel=None):
        self.aberrations = {"C10": 1.0, "C12": 2.0, "phi12": 0.1}
        self.rotation_angle_deg = math.degrees(0.2)
        self.best_loss = float("inf")
        self.trial_history = []
        self.scan_sampling_A = (0.5, 0.5)
        self.angular_sampling = (1.0, 1.0)
        self.bf_intensity_threshold = 0.5
        self.voltage_kV = 300.0
        self.semiangle_mrad = 21.9
        self._accel = _FakeAccel() if accel is None else accel
        self.backend = self._accel.backend

    @property
    def _backend_protocol(self):
        return self._accel


def test_ssb_crop_accepts_in_bounds_rectangles_with_minimum_span():
    """C1: rectangular crop requests are valid when both spans are usable."""
    from quantem.widget.showptycho import _validate_ssb_scan_region

    assert _validate_ssb_scan_region((0, 128, 128, 320), (512, 512)) == (
        0,
        128,
        128,
        320,
    )
    for region in ((0, 31, 0, 128), (0, 128, 0, 31), (-1, 127, 0, 128)):
        try:
            _validate_ssb_scan_region(region, (512, 512))
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid SSB crop {region}")


def test_showptycho_crop_request_uses_global_source_coordinates(monkeypatch, tmp_path):
    """C2: nested crop selection, expect source-relative SSB refit coordinates."""
    from quantem.widget import ShowPtycho

    monkeypatch.setitem(sys.modules, "cupy", _FakeCuPy())
    master = tmp_path / "scan_master.h5"
    master.write_bytes(b"master")
    ssb = _FakeSSB()
    ssb._accel._cache["ny"] = 1024
    ssb._accel._cache["nx"] = 1024
    widget = ShowPtycho(ssb, source_file=str(master))
    widget._scan_region = (128, 640, 256, 768)
    received: dict[str, object] = {}

    def fake_refit(region, *, n_trials):
        received["region"] = region
        received["n_trials"] = n_trials

    monkeypatch.setattr(widget, "_refit_ssb_crop", fake_refit)
    widget.crop_refit_request_json = json.dumps(
        {"id": 1, "scan_region": [64, 192, 128, 256], "n_trials": 200}
    )

    assert received == {"region": (192, 320, 384, 512), "n_trials": 200}


def test_showptycho_from_ssb_uses_widget_contract(monkeypatch):
    """C1: prepared SSB input, expect a ready widget without quantem.gpu import."""
    from quantem.widget import ShowPtycho

    monkeypatch.setitem(sys.modules, "cupy", _FakeCuPy())
    ssb = _FakeSSB()
    widget = ShowPtycho(ssb, size=320, fft_on=True)

    assert widget.__class__.__name__ == "_ShowPtychoWidget"
    assert repr(widget) == "ShowPtycho(0 pinned, drag_bf=8)"
    assert widget.phase_width == 4
    assert widget.phase_height == 3
    assert len(widget.phase_bytes) == 3 * 4 * 4
    assert widget.pixel_size == 0.5
    assert widget.initial_panel_size == 320
    assert widget.initial_fft_on is True
    assert widget.total_bf == 8
    assert widget.drag_bf == 8
    assert widget.c10_min == -300.0
    assert widget.c10_max == 300.0
    result = json.loads(widget.result_json)
    assert result["loss"] == 0.125
    assert not hasattr(ssb, "_showptycho_widget")


def test_showptycho_rejects_conflicting_prepared_backend(monkeypatch):
    """C1a: prepared CUDA session plus MPS request, expect deterministic error."""
    from quantem.widget import ShowPtycho

    monkeypatch.setitem(sys.modules, "cupy", _FakeCuPy())
    try:
        ShowPtycho(_FakeSSB(), backend="mps")
    except ValueError as exc:
        assert "prepared 'cuda' session" in str(exc)
    else:
        raise AssertionError("ShowPtycho silently changed a prepared backend")


def test_showptycho_python_uses_only_public_session_state() -> None:
    """C1b: frontend boundary, expect no CUDA optimizer private fields."""
    source = pathlib.Path("src/quantem/widget/showptycho.py").read_text()

    for private_name in (
        "_rotation_angle_rad",
        "_best_loss",
        "_optuna_trials",
        "_showptycho_widget",
        "_get_accelerator",
    ):
        assert private_name not in source


def test_showptycho_embedded_state_is_strict_json_with_unknown_loss(monkeypatch):
    """C1a: unknown/non-finite losses must not emit NaN into exported HTML."""
    from ipywidgets.embed import dependency_state

    from quantem.widget import ShowPtycho

    monkeypatch.setitem(sys.modules, "cupy", _FakeCuPy())
    ssb = _FakeSSB()
    ssb.trial_history = [
        {
            "loss": float("nan"),
            "params": {"C10_nm": 1.0, "C12_nm": 2.0, "phi12_deg": 3.0},
        },
        {
            "loss": 0.25,
            "params": {"C10_nm": 4.0, "C12_nm": 5.0, "phi12_deg": 6.0},
        },
    ]

    widget = ShowPtycho(ssb)
    state = dependency_state([widget], drop_defaults=False)

    json.dumps(state, allow_nan=False)
    assert widget.auto_loss == 0.0
    assert json.loads(widget.trials_json) == [
        {"rank": 0, "C10": 4.0, "C12": 5.0, "phi12_deg": 6.0, "loss": 0.25},
    ]


def test_showptycho_default_c10_range_includes_outlier_auto(monkeypatch):
    """C1a: default C10 range is -300..300 unless auto C10 sits outside it."""
    from quantem.widget import ShowPtycho

    monkeypatch.setitem(sys.modules, "cupy", _FakeCuPy())
    ssb = _FakeSSB()
    ssb.aberrations["C10"] = 383.3
    widget = ShowPtycho(ssb)

    assert widget.c10_min == -300.0
    assert widget.c10_max == 383.3


def test_showptycho_drag_bf_fraction_one_means_full_bf(monkeypatch):
    """C1b: drag_bf fraction 1.0, expect full BF count rather than one pixel."""
    from quantem.widget import ShowPtycho

    monkeypatch.setitem(sys.modules, "cupy", _FakeCuPy())
    ssb = _FakeSSB()
    widget = ShowPtycho(ssb, drag_bf=1.0)

    assert widget.total_bf == 8
    assert widget.drag_bf == 8


def test_showptycho_drag_bf_integer_one_still_means_one_bf(monkeypatch):
    """C1c: drag_bf integer 1, expect explicit one-BF preview request."""
    from quantem.widget import ShowPtycho

    monkeypatch.setitem(sys.modules, "cupy", _FakeCuPy())
    ssb = _FakeSSB()
    widget = ShowPtycho(ssb, drag_bf=1)

    assert widget.total_bf == 8
    assert widget.drag_bf == 1


def test_showptycho_save_calibration_is_widget_owned(monkeypatch, tmp_path):
    """C2: save from widget, expect calibration JSON without quantem.live."""
    from quantem.widget import ShowPtycho

    monkeypatch.setitem(sys.modules, "cupy", _FakeCuPy())
    ssb = _FakeSSB()
    widget = ShowPtycho(ssb, save_dir=tmp_path)
    widget.notes = "manual tune"
    widget._on_save({"new": 1})

    saved = tmp_path / "calibration.json"
    assert saved.exists()
    payload = json.loads(saved.read_text())
    assert payload["aberrations"]["C10"] == 1.0
    assert payload["aberrations"]["C12"] == 2.0
    assert math.isclose(payload["aberrations"]["phi12"], 0.1, abs_tol=1e-4)
    assert payload["notes"] == "manual tune"
    assert payload["voltage_kV"] == 300.0


def test_showptycho_calibration_seed_restores_higher_order(monkeypatch, tmp_path):
    """C3: saved calibration input, expect SSB and HO widget controls seeded."""
    from quantem.widget import PtychoCalibration, ShowPtycho
    from quantem.widget.showptycho import save_ptycho_calibration

    monkeypatch.setitem(sys.modules, "cupy", _FakeCuPy())
    path = tmp_path / "calibration.json"
    save_ptycho_calibration(
        PtychoCalibration(
            rotation_angle_deg=30.0,
            aberrations={
                "C10": 12.0,
                "C12": 5.0,
                "phi12": math.radians(10.0),
                "C30": 800.0,
                "C32": 400.0,
                "phi32": math.radians(45.0),
            },
            flip_phase=True,
            source_file="example_master.h5",
        ),
        path,
    )

    ssb = _FakeSSB()
    widget = ShowPtycho(ssb, calibration=path)

    assert ssb.aberrations["C10"] == 12.0
    assert ssb.aberrations["C12"] == 5.0
    assert math.isclose(ssb.aberrations["phi12"], math.radians(10.0))
    assert math.isclose(ssb.rotation_angle_deg, 30.0)
    assert widget.flip_phase is True
    assert widget._source_file == "example_master.h5"
    higher = json.loads(widget.higher_order_json)
    assert higher["C30"] == 800.0
    assert higher["C32_mag"] == 400.0
    assert math.isclose(higher["C32_angle"], 45.0)


def test_showptycho_calibration_seed_reuses_saved_loss(monkeypatch, tmp_path):
    """C3b: saved calibration loss, expect no duplicate full-loss pass at open."""
    from quantem.widget import PtychoCalibration, ShowPtycho
    from quantem.widget.showptycho import save_ptycho_calibration

    monkeypatch.setitem(sys.modules, "cupy", _FakeCuPy())
    path = tmp_path / "calibration.json"
    save_ptycho_calibration(
        PtychoCalibration(
            rotation_angle_deg=0.0,
            aberrations={
                "C10": 12.0,
                "C12": 5.0,
                "phi12": math.radians(10.0),
            },
            loss=0.03125,
        ),
        path,
    )

    ssb = _FakeSSB()
    widget = ShowPtycho(ssb, calibration=path)

    result = json.loads(widget.result_json)
    assert result["loss"] == 0.03125
    assert ssb._accel.reconstruct_calls == [
        ("phase", 12.0, 5.0, math.radians(10.0)),
    ]


def test_showptycho_calibration_seed_preserves_full_precision(monkeypatch, tmp_path):
    """C3b1: calibration values retain precision in synchronized state."""
    from quantem.widget import PtychoCalibration, ShowPtycho
    from quantem.widget.showptycho import save_ptycho_calibration

    monkeypatch.setitem(sys.modules, "cupy", _FakeCuPy())
    path = tmp_path / "calibration.json"
    expected = {
        "C10": 12.1234567890123,
        "C12": 5.9876543210987,
        "phi12_deg": 10.1234567890123,
    }
    save_ptycho_calibration(
        PtychoCalibration(
            rotation_angle_deg=0.0,
            aberrations={
                "C10": expected["C10"],
                "C12": expected["C12"],
                "phi12": math.radians(expected["phi12_deg"]),
            },
            loss=0.03125,
        ),
        path,
    )

    widget = ShowPtycho(_FakeSSB(), calibration=path)
    result = json.loads(widget.result_json)

    assert result["C10"] == expected["C10"]
    assert result["C12"] == expected["C12"]
    assert result["phi12_deg"] == expected["phi12_deg"]


def test_showptycho_calibration_without_loss_still_skips_loss_pass(
    monkeypatch,
    tmp_path,
):
    """C3c: calibration without saved loss, expect first draw without loss pass."""
    from quantem.widget import PtychoCalibration, ShowPtycho
    from quantem.widget.showptycho import save_ptycho_calibration

    monkeypatch.setitem(sys.modules, "cupy", _FakeCuPy())
    path = tmp_path / "calibration.json"
    save_ptycho_calibration(
        PtychoCalibration(
            rotation_angle_deg=0.0,
            aberrations={
                "C10": 12.0,
                "C12": 5.0,
                "phi12": math.radians(10.0),
            },
        ),
        path,
    )

    ssb = _FakeSSB()
    widget = ShowPtycho(ssb, calibration=path)

    result = json.loads(widget.result_json)
    assert result["loss"] is None
    assert ssb._accel.reconstruct_calls == [
        ("phase", 12.0, 5.0, math.radians(10.0)),
    ]


def test_showptycho_data_requires_geometry():
    """C4: raw data without microscope geometry, expect corrective error."""
    from quantem.widget import ShowPtycho

    try:
        ShowPtycho(np.zeros((2, 2, 4, 4), dtype=np.float32))
    except ValueError as exc:
        assert "voltage_kV, semiangle_mrad, and scan_sampling_A" in str(exc)
    else:
        raise AssertionError("ShowPtycho accepted raw data without geometry")


def test_showptycho_mps_accel_does_not_require_cupy(monkeypatch):
    """MPS-backed ShowPtycho interactions must not import CuPy."""
    from quantem.widget.showptycho import _ShowPtychoWidget

    monkeypatch.delitem(sys.modules, "cupy", raising=False)
    ssb = _FakeSSB(_FakeMpsAccel())
    widget = _ShowPtychoWidget(
        accel=ssb,
        rotation_rad=0.0,
        auto_aberrations={"C10": 1.0, "C12": 2.0, "phi12": 0.1},
        auto_loss_val=0.125,
        c10_range=(-10.0, 10.0),
        c12_range=(0.0, 10.0),
        phi12_range=(-90.0, 90.0),
        ssb_ref=ssb,
        source_file=__file__,
    )

    assert widget.phase_width == 4
    assert widget.phase_height == 3
    assert len(widget.phase_bytes) == 3 * 4 * 4
    assert json.loads(widget.result_json)["loss"] == 0.125
    assert widget.crop_refit_available is True
    assert "Ready to refit" in widget.crop_refit_status
    assert "cupy" not in sys.modules


def test_showptycho_mps_accel_uses_phase_only_reconstruct(monkeypatch):
    """MPS ShowPtycho should hit the fused phase/loss path, not object-wave work."""
    import quantem.gpu.ssb.compute.mps.engine as mps
    import quantem.gpu.ssb.compute.mps.backend as mps_engine
    from quantem.gpu.ssb.compute.mps.backend import MpsSSBBackend

    class FakePrepared:
        num_bf = 2
        scan_shape = (4, 4)
        kx_np = np.array([-0.02], dtype=np.float32)
        ky_np = np.array([0.02], dtype=np.float32)
        bf_storage_indices_np = np.array([0], dtype=np.int32)
        wavelength = 0.025
        semiangle_rad = 0.02
        ang_y_rad = 0.001
        ang_x_rad = 0.001
        dc_value = 1.0 + 0.0j
        g_qk = np.ones((1, 4, 3), dtype=np.complex64)

    calls: list[dict] = []
    fake_prepared = FakePrepared()

    monkeypatch.setattr(mps_engine, "_as_chunked_frames", lambda data: data)
    monkeypatch.setattr(mps_engine, "_scan_shape", lambda frames: (4, 4))
    monkeypatch.setattr(
        mps_engine,
        "_as_sampling",
        lambda value: (float(value), float(value)),
    )
    monkeypatch.setattr(
        mps_engine,
        "_resolve_bf_selection",
        lambda *_args, **_kwargs: mps.BrightfieldDisk(
            rows=np.array([1, 2], dtype=np.int32),
            cols=np.array([2, 1], dtype=np.int32),
            center_row_col=(1.5, 1.5),
            radius_px=2.0,
            detected_radius_px=2.0,
            detector_shape=(4, 4),
        ),
    )
    monkeypatch.setattr(mps_engine, "_default_object_redraw_chunk_bf", lambda: 16)
    monkeypatch.setattr(mps_engine, "_default_object_setup_chunk_bf", lambda: 16)
    monkeypatch.setattr(
        mps_engine,
        "_prepare_selection",
        lambda *_args, **_kwargs: fake_prepared,
    )

    def fake_reconstruct(_prepared, **kwargs):
        calls.append(dict(kwargs))
        return None, 0.25 if kwargs["compute_loss"] else None, np.zeros((4, 4), dtype=np.float32)

    monkeypatch.setattr(mps_engine, "_reconstruct_prepared", fake_reconstruct)

    accel = MpsSSBBackend(
        types.SimpleNamespace(shape=(16, 4, 4), detector_sum=None),
        voltage_kV=200.0,
        semiangle_mrad=21.4,
        scan_sampling=0.5,
        det_sampling=None,
        bf_intensity_threshold=0.5,
        bf_center=None,
        bf_radius=2,
        rotation_angle_deg=0.0,
    )
    accel.reconstruct_with_loss(1.0, 2.0, 0.3)
    accel.reconstruct(1.0, 2.0, 0.3)

    assert [call["compute_loss"] for call in calls] == [True, False]
    assert [call["compute_object"] for call in calls] == [False, False]
    state = accel.browser_state()
    from quantem.gpu.ssb.compute import SSBProtocol

    assert isinstance(accel, SSBProtocol)
    assert state.kx_bf.shape == (2,)
    assert state.ky_bf.shape == (2,)
    assert state.kx_bf[1] != 0.0


def test_showssb_is_not_public_api():
    """The public widget name is ShowPtycho."""
    import quantem.widget as qw

    assert not hasattr(qw, "ShowSSB")


def test_showptycho_exports_webgpu_folder_as_same_widget_ui(monkeypatch, tmp_path):
    """Widget-owned WebGPU folder export embeds the same ShowPtycho UI."""
    import h5py

    from quantem.widget.showptycho import _ShowPtychoWidget

    monkeypatch.setitem(sys.modules, "cupy", _FakeCuPy())
    master = tmp_path / "scan_master.h5"
    data_file = tmp_path / "scan_data_000001.h5"
    master.write_bytes(b"master")
    with h5py.File(data_file, "w") as handle:
        entry = handle.create_group("entry")
        group = entry.create_group("data")
        data = np.zeros((128 * 128, 4, 4), dtype=np.uint16)
        data[:, 1, 2] = 3
        data[:, 2, 1] = 7
        group.create_dataset("data", data=data, chunks=(512, 4, 4))
    widget = _ShowPtychoWidget(
        accel=_FakeSSB(_FakeColumnWebGPUAccel(128)),
        rotation_rad=0.0,
        auto_aberrations={"C10": 1.0, "C12": 2.0, "phi12": 0.1},
        auto_loss_val=0.125,
        c10_range=(-10.0, 10.0),
        c12_range=(0.0, 10.0),
        phi12_range=(-90.0, 90.0),
        size=320,
        fft_on=True,
        source_file=str(master),
    )

    out_dir = widget.export(tmp_path / "folder")

    assert (out_dir / "index.html").exists()
    assert not (out_dir / "g_bf.c64").exists()
    assert not (out_dir / "ref_phase.f32").exists()
    assert not (out_dir / "ref_fft.f32").exists()
    assert not (out_dir / "ref_products.npz").exists()
    assert not (out_dir / "source" / master.name).exists()
    assert not (out_dir / "source" / data_file.name).exists()
    assert (out_dir / "source" / "bf_columns.u8").exists()
    assert not (out_dir / "saves").exists()
    assert json.loads((out_dir / "snapshots" / "snapshots.json").read_text()) == []
    assert (out_dir / "snapshots" / "cal.json").exists()
    assert (out_dir / "snapshots" / "manifest.json").exists()
    assert not (out_dir / "cal.json").exists()
    assert not (out_dir / "manifest.json").exists()
    top_level = {p.name for p in out_dir.iterdir()}
    assert top_level == {"index.html", "source", "snapshots", ".viewer", "ShowPtycho.command"}
    cal = json.loads((out_dir / "snapshots" / "cal.json").read_text())
    assert cal["kind"] == "showptycho_webgpu_folder"
    assert cal["source_file"] == "redacted_local_source"
    assert cal["source_transport"] == "bf_columns"
    assert cal["bf_column_companion"] is True
    assert cal["bf_column_companion_path"] == "source/bf_columns.u8"
    assert cal["persistent_bf_cache"] is False
    assert cal["source_calibration"] == "redacted_local_calibration"
    assert cal["num_bf"] == 2
    assert cal["g_shape"] == [2, 128, 128]
    assert cal["precision"] == {
        "real_dtype": "float32",
        "complex_dtype": "complex64",
    }
    manifest = json.loads((out_dir / "snapshots" / "manifest.json").read_text())
    assert manifest["format"] == "quantem.showptycho.webgpu.folder.v2"
    assert manifest["arrays"] == {}
    assert manifest["source"]["kind"] == "bf_columns"
    assert manifest["calibration"] == "snapshots/cal.json"
    assert manifest["source"]["preferred_browser_source"] == "bf_columns"
    assert manifest["source"]["bf_columns"]["path"] == "source/bf_columns.u8"
    assert manifest["source"]["bf_columns"]["encoding"] == "uint8"
    assert manifest["source"]["bf_columns"]["num_bf"] == 2
    assert manifest["source"]["bf_columns"]["scan_shape"] == [128, 128]
    assert manifest["source"]["bf_columns"]["plane"] == 128 * 128
    assert manifest["persistent_arrays"] == []
    from quantem.widget.cli import (
        _is_showptycho_folder_export,
        _showptycho_folder,
        _showptycho_manifest,
    )

    assert _is_showptycho_folder_export(out_dir)
    assert _showptycho_folder(out_dir) == out_dir
    assert _showptycho_manifest(out_dir)["calibration"] == "snapshots/cal.json"
    html = (out_dir / "index.html").read_text()
    assert "application/vnd.jupyter.widget-state+json" in html
    assert "webgpu_standalone" in html
    assert "webgpu_h5_source_json" in html
    assert "bf_columns.u8" in html
    assert "g_bf.c64" not in html
    assert "ShowPtycho WebGPU Sidecar" not in html
    exported_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in out_dir.rglob("*")
        if path.is_file() and path.suffix in {".html", ".json", ".md"}
    )
    assert master.name not in exported_text
    assert data_file.name not in exported_text
    assert str(tmp_path) not in exported_text


def test_showptycho_export_reuses_matching_exact_bf_companion(
    tmp_path,
    monkeypatch,
):
    """C4: matching MPS BF source, expect a link without HDF5 re-extraction."""
    from quantem.gpu.ssb.bf_selector import BrightfieldDisk
    from quantem.gpu.ssb.compute.mps.engine import MpsBfColumnFrames

    from quantem.widget.showptycho_webgpu_export import _reuse_bf_column_source

    source = tmp_path / "existing" / "bf_columns.u8"
    source.parent.mkdir()
    source.write_bytes(np.arange(32, dtype=np.uint8).tobytes())
    selection = BrightfieldDisk(
        rows=np.asarray([1, 2], dtype=np.int32),
        cols=np.asarray([2, 1], dtype=np.int32),
        center_row_col=(1.5, 1.5),
        radius_px=1.0,
        detected_radius_px=1.0,
        detector_shape=(4, 4),
    )
    monkeypatch.setattr(
        MpsBfColumnFrames,
        "_detector_sum_mps",
        lambda self: np.zeros(self.det_shape, dtype=np.uint64),
    )
    frames = MpsBfColumnFrames(
        source,
        selection=selection,
        scan_shape=(4, 4),
        dtype=np.uint8,
        max_value=31,
    )
    out = tmp_path / "out"
    out.mkdir()
    calibration = {
        "bf_rows": [1, 2],
        "bf_cols": [2, 1],
        "scan_region": {"shape": [4, 4]},
    }

    state = replace(
        _FakeColumnWebGPUAccel(4).browser_state(),
        brightfield=selection,
        bf_source_path=frames.source_path,
        bf_source_dtype=frames.dtype,
        bf_source_max_value=frames.max_value,
    )
    metadata = _reuse_bf_column_source(state, out, calibration)

    assert metadata is not None
    assert metadata["encoding"] == "uint8"
    assert metadata["max_value"] == 31
    assert metadata["num_bf"] == 2
    assert metadata["plane"] == 16
    assert (out / "source" / "bf_columns.u8").read_bytes() == source.read_bytes()


def test_showptycho_folder_ui_reads_only_current_snapshot_state() -> None:
    """C1: snapshot loading has one current manifest and no compatibility merge."""
    ui_source = pathlib.Path("js/showptycho/index.tsx").read_text(encoding="utf-8")

    assert 'readSSBFolderJson<FolderSaveRecord[]>("snapshots/snapshots.json")' in ui_source
    assert 'readSSBFolderJson<FolderSaveRecord[]>("saves/saves.json")' not in ui_source
    assert "byId.set(record.id, record)" not in ui_source


def test_showptycho_folder_save_persists_without_automatic_download() -> None:
    """C1: standalone Save, expect silent snapshot JSON write without browser prompt."""
    ui_source = pathlib.Path("js/showptycho/index.tsx").read_text(encoding="utf-8")
    save_block = ui_source[
        ui_source.index("const doFolderSave = React.useCallback(async () => {"):
        ui_source.index("const loadFolderSave = React.useCallback", ui_source.index("const doFolderSave"))
    ]

    assert 'writeSSBFolderFile(record.image, jpeg)' in save_block
    assert 'writeSSBFolderFile("snapshots/snapshots.json", JSON.stringify(next, null, 2))' in save_block
    assert 'setFolderSaveStatus(`Snapshot saved (${next.length})`)' in save_block
    assert "downloadBlob(" not in save_block
    assert "+ downloaded" not in save_block


def test_showptycho_save_copy_is_project_agnostic() -> None:
    """C1: Save copy, expect the public project workflow, not a private notebook."""
    ui_source = pathlib.Path("js/showptycho/index.tsx").read_text(encoding="utf-8")
    webgpu_source = pathlib.Path(
        "js/.generated/engine/ssb/compute/webgpu/backend.ts"
    ).read_text(encoding="utf-8")

    assert "Save calibration.json in this ShowPtycho project" in ui_source
    assert "3_live.ipynb" not in ui_source
    assert "open this project with its .command launcher" in webgpu_source
    assert "serve_sidecar_range.py" not in webgpu_source


def test_showptycho_exports_wrapper_master_external_hdf5_source_fallback(monkeypatch, tmp_path):
    """C5: wrapper master, expect external HDF5 source kept compressed."""
    import h5py

    from quantem.widget.showptycho import _ShowPtychoWidget

    monkeypatch.setitem(sys.modules, "cupy", _FakeCuPy())
    data_file = tmp_path / "berk_inline_000001.h5"
    data_file_2 = tmp_path / "berk_inline_000002.h5"
    master = tmp_path / "berk_master_wrapper.h5"
    for data_path in (data_file, data_file_2):
        with h5py.File(data_path, "w") as handle:
            entry = handle.create_group("entry")
            data = entry.create_group("data")
            data.create_dataset(
                "data",
                data=np.zeros((1, 2, 2), dtype=np.uint16),
                chunks=(1, 2, 2),
            )
    with h5py.File(master, "w") as handle:
        entry = handle.create_group("entry")
        data = entry.create_group("data")
        data["data_000001"] = h5py.ExternalLink(str(data_file), "/entry/data/data")
        data["data_000002"] = h5py.ExternalLink(str(data_file_2), "/entry/data/data")

    widget = _ShowPtychoWidget(
        accel=_FakeSSB(_FakeWebGPUAccel(128)),
        rotation_rad=0.0,
        auto_aberrations={"C10": 1.0, "C12": 2.0, "phi12": 0.1},
        auto_loss_val=0.125,
        c10_range=(-10.0, 10.0),
        c12_range=(0.0, 10.0),
        phi12_range=(-90.0, 90.0),
        source_file=str(master),
    )

    out_dir = widget.export(tmp_path / "wrapper-folder", decode_dtype="uint8", webgpu_source="hdf5")

    assert (out_dir / "source" / master.name).exists()
    assert (out_dir / "source" / data_file.name).exists()
    assert (out_dir / "source" / data_file_2.name).exists()
    sidecar = out_dir / "source" / "chunks.u64"
    assert sidecar.exists()
    assert list((out_dir / "source").glob("*.chunks.u64")) == []
    assert not (out_dir / "g_bf.c64").exists()
    cal = json.loads((out_dir / "snapshots" / "cal.json").read_text())
    assert cal["source_file"] == master.name
    assert cal["source_decode_dtype"] == "uint8"
    assert cal["source_transport"] == "compressed_hdf5"
    assert cal["bf_column_companion"] is False
    manifest = json.loads((out_dir / "snapshots" / "manifest.json").read_text())
    assert manifest["source"]["master"] == f"source/{master.name}"
    assert manifest["source"]["data_files"] == [
        f"source/{data_file.name}",
        f"source/{data_file_2.name}",
    ]
    assert manifest["source"]["decode_dtype"] == "uint8"
    assert "bf_columns" not in manifest["source"]
    assert "preferred_browser_source" not in manifest["source"]
    indexes = manifest["source"]["chunk_indexes"]
    assert len(indexes) == 2
    assert {index["path"] for index in indexes} == {"source/chunks.u64"}
    assert [index["byte_offset"] for index in indexes] == [0, indexes[0]["bytes"]]
    assert sidecar.stat().st_size == sum(index["bytes"] for index in indexes)
    for index in indexes:
        assert index["frames"] == 1
        assert index["detector_shape"] == [2, 2]
        assert index["dtype"] == "uint16"
        assert index["chunk_shape"] == [1, 2, 2]
        assert index["record"] == "u64le_offset,u64le_size"
    assert manifest["source"]["files"][1]["chunk_index"] == "source/chunks.u64"
    assert manifest["source"]["files"][1]["chunk_index_byte_offset"] == 0
    assert manifest["source"]["files"][1]["chunk_index_bytes"] == indexes[0]["bytes"]
    assert manifest["source"]["files"][2]["chunk_index"] == "source/chunks.u64"
    assert manifest["source"]["files"][2]["chunk_index_byte_offset"] == indexes[0]["bytes"]
    assert manifest["source"]["files"][2]["chunk_index_bytes"] == indexes[1]["bytes"]
    assert manifest["persistent_arrays"] == []


def test_showptycho_webgpu_folder_uses_mps_metadata_without_gqk_sync(
    monkeypatch,
    tmp_path,
):
    """C5c: MPS metadata-only state, expect folder export without host G_qk copy."""
    import h5py

    from quantem.widget.showptycho import _ShowPtychoWidget

    monkeypatch.setitem(sys.modules, "cupy", _FakeCuPy())
    frames = 128 * 128
    data = np.zeros((frames, 4, 4), dtype=np.uint16)
    data[:, 1, 2] = np.arange(frames, dtype=np.uint16) % 16
    data[:, 2, 1] = (np.arange(frames, dtype=np.uint16) * 5) % 16
    data_file = tmp_path / "mps_data_000001.h5"
    master = tmp_path / "mps_master_wrapper.h5"
    with h5py.File(data_file, "w") as handle:
        entry = handle.create_group("entry")
        group = entry.create_group("data")
        group.create_dataset("data", data=data)
    with h5py.File(master, "w") as handle:
        entry = handle.create_group("entry")
        group = entry.create_group("data")
        group["data_000001"] = h5py.ExternalLink(str(data_file), "/entry/data/data")

    accel = _FakeMetadataOnlyMpsAccel(128)
    widget = _ShowPtychoWidget(
        accel=_FakeSSB(accel),
        rotation_rad=0.0,
        auto_aberrations={"C10": 1.0, "C12": 2.0, "phi12": 0.1},
        auto_loss_val=0.125,
        c10_range=(-10.0, 10.0),
        c12_range=(0.0, 10.0),
        phi12_range=(-90.0, 90.0),
        source_file=str(master),
    )

    widget.rotation_deg = 5.0
    out_dir = widget.export(tmp_path / "mps-folder", decode_dtype="uint8")

    cal = json.loads((out_dir / "snapshots" / "cal.json").read_text())
    assert cal["g_shape"] == [2, 128, 128]
    assert cal["source_transport"] == "bf_columns"
    assert cal["bf_column_companion"] is True
    assert accel.heavy_sync_calls == 0
    assert not hasattr(accel, "G_qk")
    assert not (out_dir / "g_bf.c64").exists()
    assert list((out_dir / "source").glob("bf_columns.*"))


def test_showptycho_webgpu_kernel_source_has_128_256_512_1024_specializations():
    """C6: frontend SSB code keeps explicit 128/256/512/1024 WGSL support."""
    source = _webgpu_source("ssb/compute/webgpu/backend.ts")
    source += _webgpu_source("ssb/compute/webgpu/protocol.ts")
    ui_source = pathlib.Path("js/showptycho/index.tsx").read_text()

    registry = _webgpu_source("ssb/compute/webgpu/kernels/index.ts")
    assert "SUPPORTED_SSB_SIZES = [128, 256, 512, 1024]" in registry
    assert "const workgroupSize = Math.min(n, 256)" in source
    assert "@compute @workgroup_size(${workgroupSize})" in source
    assert "for (var off = 0u; off < ${n}u; off = off + ${workgroupSize}u)" in source
    assert "for (var butterfly = tid; butterfly < ${n / 2}u; butterfly = butterfly + ${workgroupSize}u)" in source
    assert "LAN-IP HTTP pages cannot use WebGPU" in source
    assert "makeFftStages(n" in source
    assert "fn reducePartialGroups" in source
    assert "fn finalizePartialGroups" in source
    assert "bfGeom" in source
    assert "bfTrig" in source
    assert "collectActiveBfIndices" in source
    assert "packGeometry(this.cal, activeIndices, rotationDeg)" in source
    assert "fetchPackedBfColumnBytes" in source
    assert "active_bf: u32" in source
    assert "if (bg.w == 0.0) { return; }" in source
    assert "if (bf >= params.active_bf) { return; }" in source
    assert "accumSum" not in source
    assert "accumSumSq" not in source
    assert "gqkChunks" in source
    assert "paramsChunks" in source
    assert "chooseChunkCapacity" in source
    assert "GQK_GPU_BUDGET_BYTES" in source
    assert "function clippedBslz4Frames" in source
    assert "const fileFramesToRead = Math.min(index.frames, Math.max(0, plane - sourceFrames))" in source
    assert "const frameStop = Math.min(fileFramesToRead, frameStart + framesPerChunk)" in source
    assert "prepareBfCount" in source
    assert "bfCount?: number" in source
    assert "computeLoss?: boolean" in source
    assert "compute_loss: u32" in source
    assert "params.compute_loss != 0u" in source
    assert "computeLoss: false" in ui_source
    assert "computeLoss: isFull" in ui_source
    assert "if (isFull || showFFTRef.current)" in ui_source
    assert "publishShowPtychoTestFFT" in ui_source
    assert "__QUANTEM_SHOWPTYCHO_LAST_FFT__" in ui_source
    assert "rotationDeg?: number" in source
    assert "updateGeometryRotation" in source
    assert "packGeometry(this.cal, buffers.activeSourceIndices, requested)" in source
    assert "rotationDeg: rotationVal" in ui_source
    assert "sendDrag(sliderVals.current.c10, sliderVals.current.c12, sliderVals.current.phi12, v)" in ui_source
    assert "full_aberration: u32" in source
    assert "function packAberrations" in source
    assert "@group(0) @binding(12) var<storage, read> abr" in source
    assert "{ binding: 12, resource: { buffer: buffers.aberrations } }" in source
    assert "higherOrder?: Record<string, number>" in source
    assert "higherOrder: higherOrderRef.current" in ui_source
    assert "if (!engine || hoActiveRef.current) return false" not in ui_source
    assert "if (!webgpuStandalone || !engine || hoActiveRef.current) return false" not in ui_source
    assert "MAX_BF_WORKGROUPS_PER_SUBMIT = 256" in source
    assert "bf_offset: u32" in source
    assert "chunk_bf: u32" in source
    assert "activeBfCount" in source
    assert "local_bf + params.bf_offset" in source
    assert "bfOffset = 0" in source
    assert "for (let bfOffset = 0; bfOffset < buffers.activeBfCount; bfOffset += buffers.dispatchChunkCapacity)" in source
    assert "buffers.fullStack" not in source
    assert "pass.setPipeline(pipelines.reducePartial)" in source
    assert "pass.setPipeline(pipelines.finalizeGroups)" in source
    assert "pipelines.reducePartial" in source
    assert "pipelines.finalizeGroups" in source
    assert "setupPromise: Promise<void> | null" in source
    assert "operationQueue: Promise<void>" in source
    assert "runExclusive(async () =>" in source
    assert "WebGPU SSB buffers are not ready after setup" in source
    assert "if (this.setupPromise)" in source
    assert "readSourceBytes" in source
    assert "fetchRangeBytes" in source
    assert "headers: { Range:" in source
    assert "bfColumnsToGqk" in source
    assert "BF_COLUMN_UNPACK_WORKGROUP_X = 32" in source
    assert "BF_COLUMN_UNPACK_WORKGROUP_Y = 8" in source
    assert "Math.ceil(plane / BF_COLUMN_UNPACK_WORKGROUP_X)" in source
    assert "Math.ceil(chunkBf / BF_COLUMN_UNPACK_WORKGROUP_Y)" in source
    assert "pass.dispatchWorkgroups(Math.ceil(plane / 16)" not in source
    assert "mode == 3u" in source
    assert "__ssbBfColumnProfile" in source
    assert "loadedBfCount" in source
    assert "setup(requiredBfCount" in source
    assert "DEFAULT_BF_FRACTION = 1.0" in ui_source
    assert "defaultBfCount(total)" in ui_source
    assert "dragBfRef.current = count" in ui_source
    assert "setLocalDragBf(count)" in ui_source
    assert "exported WebGPU HTML has no kernel observer" in ui_source
    assert "renderAllRef.current(flipped, p.w, p.h)" in ui_source
    assert "}, [flipPhase]);" in ui_source
    assert "__QUANTEM_SHOWPTYCHO_CAPTURE__" in ui_source
    assert "__QUANTEM_SHOWPTYCHO_LAST_PHASE__" in ui_source
    assert "full BF phase" in ui_source
    assert "full BF + loss" in ui_source
    assert "loop {" not in source
    assert "var len: u32" not in source
    assert "bf + 7u < params.chunk_bf" not in source
    assert "bf + 7u < params.num_bf" not in source
    assert "const REDUCE_BF_GROUP = 32" in source
    assert "groups_in_chunk = (params.chunk_bf + ${REDUCE_BF_GROUP - 1}u)" in source
    assert "global_group = start_bf / ${REDUCE_BF_GROUP}u" in source


def test_showptycho_phase_contrast_coalesces_gpu_updates_without_thumb_swaps():
    """C7: rapid phase contrast drags, expect latest-only GPU paints."""
    ui_source = pathlib.Path("js/showptycho/index.tsx").read_text()

    assert "disableSwap" in ui_source
    assert "gpuCmapBusyRef" in ui_source
    assert "gpuCmapPendingRef" in ui_source
    assert "if (gpuCmapBusyRef.current[slot])" in ui_source
    assert "gpuCmapPendingRef.current[slot] = launch" in ui_source
    assert "pending?.()" in ui_source
    assert "engine.applySingle(slot, vmin, vmax, false).then(rgba =>" in ui_source
    assert "image.data.set(rgba)" in ui_source
    assert "if (!rawPhaseRef.current || contrastRafRef.current !== null) return" in ui_source
    assert "renderRealDisplayRef.current(latest.data, latest.w, latest.h)" in ui_source
    assert 'aria-pressed={extraRealViews.amp}' in ui_source
    assert 'aria-pressed={extraRealViews.complex}' in ui_source


def test_showptycho_webgpu_reuses_resident_bf_columns_after_prepare():
    """C7: prepared browser BF columns, expect sliders/FFT not to refetch them."""
    source = _webgpu_source("ssb/compute/webgpu/backend.ts")
    ui_source = pathlib.Path("js/showptycho/index.tsx").read_text()

    setup = source[
        source.index("private async setup("):
        source.index("private async runExclusive", source.index("private async setup("))
    ]
    setup_once = source[
        source.index("private async setupOnce("):
        source.index("private clampBfCount", source.index("private async setupOnce("))
    ]
    reconstruct = source[
        source.index("async reconstruct("):
        source.index("\n  }\n}", source.index("async reconstruct("))
    ]
    commit_bf = ui_source[
        ui_source.index("const commitDragBf = React.useCallback("):
        ui_source.index("/* --- Slider values ref", ui_source.index("const commitDragBf = React.useCallback("))
    ]
    frontend_full = ui_source[
        ui_source.index("const runFrontendFull = React.useCallback("):
        ui_source.index("frontendFullRef.current = runFrontendFull", ui_source.index("const runFrontendFull = React.useCallback("))
    ]

    resident_guard = (
        "if (this.initialized && this.loadedBfCount >= capacity) {\n"
        "      this.updateGeometryRotation(rotationDeg);\n"
        "      return;\n"
        "    }"
    )
    assert resident_guard in setup
    assert setup.index(resident_guard) < setup.index("this.setupPromise = this.setupOnce")
    assert "await this.setupPromise;" in setup
    assert setup.count("this.updateGeometryRotation(rotationDeg);") >= 3
    assert re.search(r"this\.setupPromise = this\.setupOnce\(capacity, rotationDeg\);", setup)

    assert "if (this.initialized && this.loadedBfCount >= capacity) return;" in setup_once
    assert "buildBfColumnGqkChunks(" in setup_once
    assert "buildH5GqkChunks(" in setup_once
    assert "this.loadedBfCount = nbf;" in setup_once
    assert "this.loadedBfCount = 0;" in source

    assert "await this.setup(bfCount, rotationDeg);" in reconstruct
    assert "computeLoss = options.computeLoss ?? bfCount === this.cal.num_bf" in reconstruct
    assert "await readF32(device, buffers.phase, this.plane)" in reconstruct

    assert ui_source.count("phi12Val * Math.PI / 180") == 2
    assert "frame.phi12 * Math.PI / 180" in ui_source

    assert "engine.prepareBfCount(count).then(prepared =>" in commit_bf
    assert "const fn = prepared >= total ? frontendFullRef.current : frontendPreviewRef.current;" in commit_bf
    assert "fn?.(current.c10, current.c12, current.phi12, rotationDegRef.current);" in commit_bf
    assert "computeLoss: isFull" in frontend_full
    assert "const modeLabel = isFull ? \"full BF + loss\" : \"selected BF\";" in frontend_full


def test_showptycho_webgpu_folder_uses_full_logical_bf_total():
    """C7b: standalone BF slider, expect the complete coordinate prefix."""
    source = _webgpu_source("ssb/compute/webgpu/backend.ts")
    ui_source = pathlib.Path("js/showptycho/index.tsx").read_text()

    active_selector = source[
        source.index("function collectActiveBfIndices("):
        source.index("function packGeometry", source.index("function collectActiveBfIndices("))
    ]

    assert "for (let i = 0; i < cal.num_bf; i++)" in active_selector
    assert "if (aperture !== 0) active.push(i)" in active_selector
    assert "const count = Math.max(1, Math.min(active.length, Math.round(numBf)))" in active_selector
    assert "for (let i = 0; i < count; i++)" in active_selector
    assert "const webgpuCalLogicalBf = React.useMemo(() =>" in ui_source
    assert "const effectiveTotalBf = webgpuStandalone && webgpuCalLogicalBf > 0" in ui_source
    assert "return Number.isFinite(numBf) && numBf > 0 ? Math.round(numBf) : 0" in ui_source
    assert "const seedStandaloneDefault = webgpuStandalone && !standaloneBfSeededRef.current && total > 0" in ui_source
    assert "raw <= 0 || seedStandaloneDefault" in ui_source
    assert "max={effectiveTotalBf > 0 ? effectiveTotalBf : 0}" in ui_source
