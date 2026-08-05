#!/usr/bin/env python3
"""Generate standalone HTML exports for every export-capable widget."""

from __future__ import annotations

import argparse
import html
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from quantem.widget.export import supports_html_export
from quantem.widget.show2d import Show2D
from quantem.widget.show3d import Show3D
from quantem.widget.show3dslices import Show3DSlices
from quantem.widget.show4dstem import Show4DSTEM
from quantem.widget.showdiffraction import ShowDiffraction
from quantem.widget.showeds import ShowEDS
from quantem.widget.showfolder import ShowFolder
from quantem.widget.showptycho import _ShowPtychoWidget


def _metadata() -> np.ndarray:
    payload = json.dumps({
        "Scan": {"ScanRotation": "0"},
        "BinaryResult": {"PixelSize": {"height": 1e-9, "width": 1e-9}},
    }).encode()
    arr = np.zeros((len(payload) + 1, 1), dtype=np.uint8)
    arr[: len(payload), 0] = np.frombuffer(payload, dtype=np.uint8)
    return arr


def _image_emd(path: Path) -> None:
    with h5py.File(path, "w") as h5:
        group = h5.create_group("Data/Image/uid")
        group.create_dataset("Data", data=np.arange(16 * 16, dtype=np.float32).reshape(16, 16))
        group.create_dataset("Metadata", data=_metadata())


class _SmokePtychoAccel:
    """Tiny MPS-shaped SSB accelerator for a real ShowPtycho frontend smoke."""

    backend = "mps"

    def __init__(self, n: int = 128) -> None:
        self._frames = None
        q = np.fft.fftfreq(n, d=0.5).astype(np.float32)
        self._cache = {
            "num_bf": 2,
            "ny": n,
            "nx": n,
            "kx_bf": np.array([-0.08, 0.08], dtype=np.float32),
            "ky_bf": np.array([0.08, -0.08], dtype=np.float32),
            "qx_1d": q,
            "qy_1d": q,
            "aperture_k_1d": np.ones(2, dtype=np.float32),
            "alpha_k2_1d": np.array([0.002, 0.002], dtype=np.float32),
            "cos2phi_k_1d": np.array([1.0, -1.0], dtype=np.float32),
            "sin2phi_k_1d": np.array([0.0, 0.0], dtype=np.float32),
            "semiangle_rad": np.float32(0.03),
            "ang_y_rad": np.float32(0.001),
            "ang_x_rad": np.float32(0.001),
        }
        self.g_shape = (2, n, n)
        self.gpts = (4, 4)
        self.bf_center = (1.5, 1.5)
        self.bf_inds_row = np.array([1, 2], dtype=np.int32)
        self.bf_inds_col = np.array([2, 1], dtype=np.int32)
        self.wavelength = np.float32(0.0197)
        self.sampling = (np.float32(0.5), np.float32(0.5))
        self._dc_value_host = np.complex64(1.0 + 0.0j)
        self._phase_template = _ptycho_phase_template(n)
        self._g_qk = np.ones((2, n, n), dtype=np.complex64)

    @property
    def scan_shape(self) -> tuple[int, int]:
        return int(self._cache["ny"]), int(self._cache["nx"])

    @property
    def detector_shape(self) -> tuple[int, int]:
        return self.gpts

    @property
    def num_bf(self) -> int:
        return int(self._cache["num_bf"])

    @staticmethod
    def phase_to_numpy(phase) -> np.ndarray:
        return np.asarray(phase, dtype=np.float32)

    def browser_state(self):
        from quantem.gpu.ssb.compute.protocol import SSBExportState
        from quantem.gpu.ssb.bf_selector import BrightfieldDisk

        selection = BrightfieldDisk(
            rows=self.bf_inds_row,
            cols=self.bf_inds_col,
            center_row_col=self.bf_center,
            radius_px=1.0,
            detected_radius_px=1.0,
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
        self._rotation_rad = float(rotation_rad)

    def preview_context(self, num_bf: int):
        del num_bf
        return None

    def reconstruct_with_loss(self, c10: float, c12: float, phi12: float):
        phase = self.reconstruct(c10, c12, phi12)
        return phase, float(np.var(phase))

    def reconstruct(self, c10: float, c12: float, phi12: float):
        phase = self._phase_template.copy()
        phase += np.float32(0.0005 * c10 + 0.0003 * c12)
        phase += np.float32(0.03) * np.sin(
            np.linspace(0, np.pi * 2, phase.shape[1], dtype=np.float32)
            + np.float32(phi12)
        )[None, :]
        return phase.astype(np.float32, copy=False)

    def reconstruct_full_with_loss(self, mags_m, angles_rad):
        mags = np.asarray(mags_m, dtype=np.float32)
        angles = np.asarray(angles_rad, dtype=np.float32)
        if np.any(mags[2:] != 0):
            raise NotImplementedError("smoke ptycho accelerator implements the 3-parameter path only")
        return self.reconstruct_with_loss(float(mags[0]), float(mags[1]), float(angles[1]))

    def reconstruct_full(self, mags_m, angles_rad):
        mags = np.asarray(mags_m, dtype=np.float32)
        angles = np.asarray(angles_rad, dtype=np.float32)
        if np.any(mags[2:] != 0):
            raise NotImplementedError("smoke ptycho accelerator implements the 3-parameter path only")
        return self.reconstruct(float(mags[0]), float(mags[1]), float(angles[1]))


class _SmokePtychoState:
    def __init__(self, accel: _SmokePtychoAccel) -> None:
        self.backend = accel.backend
        self.aberrations = {"C10": 1.0, "C12": 2.0, "phi12": 0.1}
        self.rotation_angle_deg = 0.0
        self.trial_history = [
            {"loss": 0.18, "params": {"C10_nm": 1.0, "C12_nm": 2.0, "phi12_deg": 5.0}},
            {"loss": 0.22, "params": {"C10_nm": -8.0, "C12_nm": 6.0, "phi12_deg": 15.0}},
        ]
        self.best_loss = 0.18
        self.voltage_kV = 300.0
        self.semiangle_mrad = 30.0
        self.scan_sampling_A = 0.5
        self.angular_sampling = (1.0, 1.0)
        self._accel = accel

    @property
    def _backend_protocol(self):
        return self._accel

    @property
    def scan_shape(self) -> tuple[int, int]:
        return self._accel.scan_shape

    @property
    def num_bf(self) -> int:
        return self._accel.num_bf

    def set_rotation(self, rotation_angle_deg: float) -> None:
        self.rotation_angle_deg = float(rotation_angle_deg)
        self._accel.cache_rotation(np.deg2rad(self.rotation_angle_deg))

    def preview(
        self,
        aberrations: dict[str, float],
        *,
        compute_loss: bool = True,
        higher_order_magnitudes=None,
        higher_order_angles=None,
    ):
        if higher_order_magnitudes is not None:
            if compute_loss:
                phase, loss = self._accel.reconstruct_full_with_loss(
                    higher_order_magnitudes,
                    higher_order_angles,
                )
            else:
                phase = self._accel.reconstruct_full(
                    higher_order_magnitudes,
                    higher_order_angles,
                )
                loss = None
        elif compute_loss:
            phase, loss = self._accel.reconstruct_with_loss(
                aberrations["C10"],
                aberrations["C12"],
                aberrations["phi12"],
            )
        else:
            phase = self._accel.reconstruct(
                aberrations["C10"],
                aberrations["C12"],
                aberrations["phi12"],
            )
            loss = None
        return np.asarray(phase, dtype=np.float32), loss

    def preview_context(self, num_bf: int):
        return self._accel.preview_context(num_bf)

    def browser_state(self):
        return self._accel.browser_state()


class _ShowPtychoFolderSmoke:
    def __init__(self, widget: _ShowPtychoWidget) -> None:
        self.widget = widget

    def export_html(
        self,
        path: str | Path | None = None,
        *,
        title: str | None = None,
        **options: object,
    ) -> Path:
        target = Path(path or "showptycho-webgpu-folder.html")
        out_dir = target.with_suffix("")
        return self.widget.export(out_dir, title=title, **options) / "index.html"


def _ptycho_phase_template(n: int) -> np.ndarray:
    y, x = np.mgrid[:n, :n].astype(np.float32)
    phase = np.full((n, n), -0.12, dtype=np.float32)
    spacing = n / 7.5
    sigma = max(1.2, spacing * 0.13)
    for row in range(-1, 9):
        for col in range(-1, 9):
            cx = n * 0.08 + col * spacing + (row % 2) * spacing * 0.5
            cy = n * 0.10 + row * spacing * 0.86
            if -3 * sigma <= cx < n + 3 * sigma and -3 * sigma <= cy < n + 3 * sigma:
                phase += np.float32(0.8) * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    phase -= float(phase.mean())
    return phase.astype(np.float32, copy=False)


def _showptycho_master(folder_root: Path, rng: np.random.Generator) -> Path:
    try:
        import hdf5plugin
    except Exception as exc:  # pragma: no cover - environment problem
        raise RuntimeError(
            "ShowPtycho compressed-HDF5 smoke requires hdf5plugin so the "
            "fixture uses real bitshuffle-LZ4 detector chunks."
        ) from exc
    frames = 128 * 128
    data = np.zeros((frames, 4, 4), dtype=np.uint16)
    data[:, 1, 2] = rng.integers(0, 16, size=frames, dtype=np.uint16)
    data[:, 2, 1] = rng.integers(0, 16, size=frames, dtype=np.uint16)
    data_file = folder_root / "showptycho_data_000001.h5"
    master = folder_root / "showptycho_master.h5"
    with h5py.File(data_file, "w") as handle:
        group = handle.create_group("entry/data")
        group.create_dataset(
            "data",
            data=data,
            chunks=(1, 4, 4),
            **hdf5plugin.Bitshuffle(cname="lz4"),
        )
    with h5py.File(master, "w") as handle:
        group = handle.create_group("entry/data")
        group["data_000001"] = h5py.ExternalLink(data_file.name, "/entry/data/data")
    return master


def _showptycho_smoke(folder_root: Path, rng: np.random.Generator) -> _ShowPtychoFolderSmoke:
    accel = _SmokePtychoAccel(128)
    ssb = _SmokePtychoState(accel)
    widget = _ShowPtychoWidget(
        accel=ssb,
        rotation_rad=0.0,
        auto_aberrations=ssb.aberrations,
        auto_loss_val=ssb.best_loss,
        c10_range=(-20.0, 20.0),
        c12_range=(0.0, 20.0),
        phi12_range=(-45.0, 45.0),
        drag_bf=1.0,
        ssb_ref=ssb,
        pixel_size=0.5,
        source_file=str(_showptycho_master(folder_root, rng)),
        size=320,
        fft_on=True,
    )
    return _ShowPtychoFolderSmoke(widget)


def _mos2_lattice_stack(rng: np.random.Generator, frames: int, rows: int, cols: int) -> np.ndarray:
    """Small 1H-MoS2-like HAADF phantom for human-readable smoke reports.

    The smoke must stay tiny enough for CI, but the visual report should still
    read like atomic-resolution microscopy data. A projected 1H-MoS2 HAADF view
    has a honeycomb-like pair of column sites: bright Mo columns and dimmer
    projected S2 columns. Keeping sulfur as one projected column avoids the
    generic three-dot motif that does not look like MoS2.
    """
    y, x = np.mgrid[:rows, :cols].astype(np.float32)
    spacing = max(6.5, min(rows, cols) / 8.5)
    sigma_mo = max(0.65, spacing * 0.115)
    sigma_s2 = max(0.72, spacing * 0.135)
    angle = np.deg2rad(8.0)
    a1 = spacing * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
    a2 = spacing * np.array([np.cos(angle + np.pi / 3.0), np.sin(angle + np.pi / 3.0)], dtype=np.float32)
    # Approximate HAADF Z contrast: Mo is dominant, projected S2 is visible but
    # much dimmer. The exact exponent is not important for this smoke; the visual
    # contract is the strong Mo/S2 contrast and honeycomb geometry.
    basis = [
        (np.array([0.0, 0.0], dtype=np.float32), 1.00, sigma_mo),
        ((a1 + a2) / 3.0, 0.42, sigma_s2),
    ]

    stack = []
    for idx in range(frames):
        frame = np.full((rows, cols), 0.035, dtype=np.float32)
        drift = np.array([0.16 * idx, -0.11 * idx], dtype=np.float32)
        origin = np.array([cols * 0.08, rows * 0.10], dtype=np.float32) + drift
        for row_idx in range(-2, int(rows / spacing) + 4):
            for col_idx in range(-2, int(cols / spacing) + 4):
                cell = origin + col_idx * a1 + row_idx * a2
                for offset, amp, sigma in basis:
                    cx, cy = cell + offset
                    if (
                        -3 * sigma <= cx < cols + 3 * sigma
                        and -3 * sigma <= cy < rows + 3 * sigma
                    ):
                        r2 = (x - cx) ** 2 + (y - cy) ** 2
                        frame += amp * np.exp(-r2 / (2.0 * sigma**2)).astype(np.float32)
        scan_modulation = 1.0 + 0.035 * np.sin((y + 0.7 * idx) / max(rows, 1) * 2.0 * np.pi)
        thickness = 0.88 + 0.12 * np.sin((x + y * 0.35) / max(cols, 1) * 2.0 * np.pi)
        noise = rng.normal(0, 0.014, size=(rows, cols)).astype(np.float32)
        frame = frame * scan_modulation.astype(np.float32) * thickness.astype(np.float32) + noise
        frame -= float(frame.min())
        frame /= float(frame.max()) + 1e-6
        stack.append(frame.astype(np.float32))
    return np.stack(stack, axis=0)


def _cases(folder_root: Path) -> list[tuple[str, str, object, dict[str, object], str]]:
    rng = np.random.default_rng(0)
    showfolder_dir = folder_root / "showfolder-session"
    showfolder_dir.mkdir(parents=True, exist_ok=True)
    _image_emd(showfolder_dir / "0010 - HAADF 15Mx Nano.emd")
    _image_emd(showfolder_dir / "0011 - HAADF 15Mx Nano.emd")

    show2d_single = _mos2_lattice_stack(rng, 1, 160, 192)[0]
    show2d_gallery3 = _mos2_lattice_stack(rng, 3, 160, 192)
    show2d_gallery6 = _mos2_lattice_stack(rng, 6, 144, 168)
    show2d_gallery8 = _mos2_lattice_stack(rng, 8, 128, 144)
    show3d_stack = _mos2_lattice_stack(rng, 10, 160, 192)
    show3d_short = _mos2_lattice_stack(rng, 5, 144, 168)
    show3d_panel_a = _mos2_lattice_stack(rng, 8, 144, 168)
    show3d_panel_b = _mos2_lattice_stack(rng, 8, 144, 168) * 0.7
    show3d_panel_c = _mos2_lattice_stack(rng, 8, 144, 168) + 0.2
    show3d_panel_d = _mos2_lattice_stack(rng, 8, 144, 168)

    cases: list[tuple[str, str, object, dict[str, object], str]] = [
        (
            "show2d",
            "show2d-single",
            Show2D(show2d_single, title="Smoke Show2D Single", sampling=0.2, units="nm", verbose=False),
            {"encoding": "uint8"},
            "Smoke Show2D Single",
        ),
        (
            "show2d",
            "show2d-gallery-3",
            Show2D(
                show2d_gallery3,
                labels=["original", "shifted", "noisy"],
                title="Smoke Show2D Gallery 3",
                ncols=3,
                sampling=0.2,
                units="nm",
                panel_order=["noisy", "original", "shifted"],
                verbose=False,
            ),
            {"encoding": "uint8"},
            "Smoke Show2D Gallery 3",
        ),
        (
            "show2d",
            "show2d-gallery-6-fft",
            Show2D(
                show2d_gallery6,
                labels=[f"panel {idx + 1}" for idx in range(6)],
                title="Smoke Show2D Gallery 6 FFT",
                ncols=3,
                show_fft=True,
                link_zoom=True,
                link_pan=True,
                link_contrast=True,
                verbose=False,
            ),
            {"encoding": "uint8"},
            "Smoke Show2D Gallery 6 FFT",
        ),
        (
            "show2d",
            "show2d-hidden-starred",
            Show2D(
                show2d_gallery6,
                labels=[f"panel {idx + 1}" for idx in range(6)],
                title="Smoke Show2D Hidden Starred",
                ncols=3,
                hidden_panels=[1, "panel 5"],
                starred=[0, "panel 3"],
                verbose=False,
            ),
            {"encoding": "uint8"},
            "Smoke Show2D Hidden Starred",
        ),
        (
            "show2d",
            "show2d-compact-no-titles",
            Show2D(
                show2d_gallery8,
                labels=[f"compact {idx + 1}" for idx in range(8)],
                title="Smoke Show2D Compact No Titles",
                ncols=4,
                show_panel_titles=False,
                show_stats=False,
                display_bin=2,
                verbose=False,
            ),
            {"encoding": "uint8"},
            "Smoke Show2D Compact No Titles",
        ),
        (
            "show3d",
            "show3d-single-stack",
            Show3D(show3d_stack, title="Smoke Show3D Single Stack", sampling=0.2, units="nm"),
            {"encoding": "uint8"},
            "Smoke Show3D Single Stack",
        ),
        (
            "show3d",
            "show3d-single-fft-bottom",
            Show3D(show3d_short, title="Smoke Show3D FFT Bottom", show_fft=True, fft_layout="bottom", fps=12),
            {"encoding": "uint8"},
            "Smoke Show3D FFT Bottom",
        ),
        (
            "show3d",
            "show3d-single-fft-overlay",
            Show3D(
                show3d_short,
                title="Smoke Show3D FFT Overlay",
                show_fft=True,
                fft_layout="overlay",
                fft_overlay_position="bottom-right",
                fps=12,
            ),
            {"encoding": "uint8"},
            "Smoke Show3D FFT Overlay",
        ),
        (
            "show3d",
            "show3d-three-panels",
            Show3D(
                show3d_panel_a,
                show3d_panel_b,
                show3d_panel_c,
                title="Smoke Show3D Three Panels",
                panel_titles=["SSB reconstruction", "Mean DP", "Probe"],
                panel_order=["Probe", "SSB reconstruction", "Mean DP"],
                max_cols=3,
                hideable=True,
            ),
            {"encoding": "uint8"},
            "Smoke Show3D Three Panels",
        ),
        (
            "show3d",
            "show3d-hidden-panel",
            Show3D(
                show3d_panel_a,
                show3d_panel_b,
                show3d_panel_c,
                title="Smoke Show3D Hidden Panel",
                panel_titles=["SSB reconstruction", "Mean DP", "Probe"],
                hidden_panels=["Mean DP"],
                hideable=True,
                max_cols=3,
                show_stats=False,
            ),
            {"encoding": "uint8"},
            "Smoke Show3D Hidden Panel",
        ),
        (
            "show3d",
            "show3d-four-panel-downsample",
            Show3D(
                show3d_panel_a,
                show3d_panel_b,
                show3d_panel_c,
                show3d_panel_d,
                title="Smoke Show3D Four Panel Downsample",
                panel_titles=["A", "B", "C", "D"],
                max_cols=4,
                avg_window=2,
                fps=18,
            ),
            {"encoding": "uint8", "downsample": 2},
            "Smoke Show3D Four Panel Downsample",
        ),
        (
            "show3dslices",
            "show3dslices",
            Show3DSlices(rng.random((8, 32, 32), dtype=np.float32), title="Smoke Show3DSlices"),
            {"encoding": "uint8"},
            "Smoke Show3DSlices",
        ),
        (
            "show4dstem",
            "show4dstem",
            Show4DSTEM(rng.integers(0, 64, size=(4, 4, 8, 8), dtype=np.uint16), title="Smoke Show4DSTEM", verbose=False),
            {"encoding": "uint8", "downsample": 1},
            "Smoke Show4DSTEM",
        ),
        (
            "show4dstem",
            "show4dstem-compare",
            Show4DSTEM(
                rng.integers(0, 64, size=(14, 4, 4, 8, 8), dtype=np.uint16),
                title="Smoke Show4DSTEM Compare",
                frame_dim_label="Dataset",
                frame_labels=[f"scan-{idx}" for idx in range(14)],
                view_mode="multiple",
                compare_cols=4,
                compare_max_panels=14,
                verbose=False,
            ),
            {"encoding": "uint8", "downsample": 1},
            "Smoke Show4DSTEM Compare",
        ),
        (
            "showptycho",
            "showptycho-webgpu-folder",
            _showptycho_smoke(folder_root, rng),
            {"decode_dtype": "uint8"},
            "Smoke showptycho-webgpu-folder",
        ),
        (
            "showeds",
            "showeds",
            ShowEDS(rng.integers(0, 32, size=(5, 6, 12), dtype=np.uint16), title="Smoke ShowEDS", band=(2, 8), roi=(1, 1, 3, 3)),
            {"mode": "single", "encoding": "full"},
            "Smoke ShowEDS",
        ),
        (
            "showdiffraction",
            "showdiffraction",
            ShowDiffraction(rng.random((48, 48), dtype=np.float32), title="Smoke ShowDiffraction", verbose=False),
            {"encoding": "full"},
            "Smoke ShowDiffraction",
        ),
        (
            "showfolder",
            "showfolder",
            ShowFolder(showfolder_dir, thumb=8, group_by="none", cache_dir=folder_root / "cache"),
            {},
            "0010",
        ),
    ]
    return cases


def _relative_export_path(artifact_dir: Path, path: str | Path) -> str:
    try:
        return Path(path).resolve().relative_to(artifact_dir.resolve()).as_posix()
    except ValueError:
        return Path(path).name


def _write_browser_plan(artifact_dir: Path, report: dict[str, Any]) -> None:
    pages = [
        {
            "widget": item["widget"],
            "variant": item["variant"],
            "file": _relative_export_path(artifact_dir, str(item["path"])),
            "url_path": _relative_export_path(artifact_dir, str(item["path"])),
            "required_interactions": [
                "open the page and confirm the widget renders",
                "click or drag the primary image/canvas where available",
                "toggle FFT, profile, ROI, or related toolbar controls where available",
                "open Export and confirm the downloaded HTML path still works",
            ],
        }
        for item in report["exports"]
    ]
    plan = {
        "version": 1,
        "description": "Open these small standalone exports in the in-app browser for visual HTML export signoff.",
        "artifact_dir": str(artifact_dir),
        "pages": pages,
    }
    (artifact_dir / "browser-plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")


def _write_html_report(artifact_dir: Path, report: dict[str, Any]) -> None:
    rows = "\n".join(
        (
            export_path := _relative_export_path(artifact_dir, str(item["path"])),
            "<tr>"
            f"<td>{html.escape(str(item['widget']))}</td>"
            f"<td>{html.escape(str(item['variant']))}</td>"
            f"<td>{html.escape(str(item['seconds']))}</td>"
            f"<td>{html.escape(format(float(item['size_mb']), '.3f'))}</td>"
            f"<td><a href='{html.escape(export_path)}'>"
            f"{html.escape(export_path)}</a></td>"
            "</tr>",
        )[1]
        for item in report["exports"]
    )
    report_json = html.escape(json.dumps(report, indent=2))
    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>quantem.widget HTML export smoke</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #18202a; line-height: 1.45; }}
    table {{ border-collapse: collapse; margin-top: 12px; min-width: 640px; }}
    th, td {{ border: 1px solid #ccd3db; padding: 6px 8px; text-align: left; }}
    th {{ background: #f3f5f7; }}
    code, pre {{ background: #f5f7f9; border-radius: 4px; }}
    code {{ padding: 2px 4px; }}
    pre {{ overflow: auto; padding: 12px; max-width: 1100px; }}
  </style>
</head>
<body>
  <h1>quantem.widget HTML export smoke</h1>
  <p>This report is generated by <code>scripts/widget_html_smoke.py</code>. Open
  each linked export in the in-app browser and follow <code>browser-plan.json</code>
  when a visual signoff is needed.</p>
  <p><strong>Show2D</strong> and <strong>Show3D</strong> examples use a small
  synthetic MoS2-like HAADF lattice so CI stays lightweight while the visual
  checks still show microscopy-style atomic contrast and FFT peaks.</p>
  <p><strong>ShowPtycho</strong> uses a tiny WebGPU folder export with a local
  HDF5 source folder so the browser smoke opens the same sidecar shape used for
  collaborator handoff.</p>
  <p>Total export size: <strong>{html.escape(f'{float(report["total_size_mb"]):.3f} MB')}</strong></p>
  <table>
    <thead><tr><th>Widget</th><th>Variant</th><th>Export seconds</th><th>Size MB</th><th>Standalone HTML</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <h2>Machine-readable report</h2>
  <p><a href="report.json">report.json</a> · <a href="browser-plan.json">browser-plan.json</a></p>
  <pre>{report_json}</pre>
</body>
</html>
"""
    (artifact_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--max-total-mb", type=float, default=25.0)
    args = parser.parse_args()

    if args.artifact_dir is None:
        artifact_dir = Path(tempfile.mkdtemp(prefix="quantem-widget-html-smoke-"))
    else:
        artifact_dir = args.artifact_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)

    report_rows: list[dict[str, object]] = []
    total_size = 0
    for widget_name, variant, widget, kwargs, marker in _cases(artifact_dir):
        if not supports_html_export(widget):
            raise RuntimeError(f"{variant} does not satisfy supports_html_export")
        start = time.perf_counter()
        out = widget.export_html(artifact_dir / f"{variant}.html", title=f"Smoke {variant}", **kwargs)
        elapsed = time.perf_counter() - start
        text = out.read_text(encoding="utf-8")
        size = out.stat().st_size
        total_size += size
        required = [
            "application/vnd.jupyter.widget-state+json",
            "quantem-widget-export-layout",
            str(marker),
        ]
        missing = [item for item in required if item not in text]
        if missing:
            raise RuntimeError(f"{variant} export missing markers: {missing}")
        report_rows.append({
            "widget": widget_name,
            "variant": variant,
            "path": str(out),
            "seconds": round(elapsed, 3),
            "size_mb": round(size / 1024 / 1024, 3),
            "options": kwargs,
        })

    max_total = args.max_total_mb * 1024 * 1024
    report = {
        "artifact_dir": str(artifact_dir),
        "total_size_mb": round(total_size / 1024 / 1024, 3),
        "exports": report_rows,
    }
    (artifact_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_browser_plan(artifact_dir, report)
    _write_html_report(artifact_dir, report)
    print(json.dumps(report, indent=2))
    print(f"HTML smoke report: {artifact_dir / 'index.html'}")
    if total_size > max_total:
        raise RuntimeError(
            f"HTML smoke exports total {total_size / 1024 / 1024:.2f} MB "
            f"> {args.max_total_mb:.2f} MB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
