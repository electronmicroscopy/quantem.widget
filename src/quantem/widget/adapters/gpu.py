"""quantem.gpu acquisitions, detector sessions and SSB, whether or not quantem.gpu is installed.

quantem.gpu is optional (``quantem.widget[cuda]`` or ``quantem.widget[mps]``). This is the only
module that imports it: encoded acquisitions from ``quantem.gpu.io.load``, their
detector sessions, master discovery and inspection, SSB, centre of mass and the
release of cached GPU memory.
Without quantem.gpu every name below is None, ``available()`` is False and a
function that needs it raises ``ImportError`` with the install command; the
widgets then read files densely with ``show4dstem.reader`` and reduce with
their own torch code.
"""

import math
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from quantem.widget.device import device_name, gpu_path_hint, no_gpu_path, print_once

try:
    from quantem.gpu import SSB
    from quantem.gpu import io as gpu_io
    from quantem.gpu.detector import disk_edge_radius as gpu_disk_edge_radius
    from quantem.gpu.detector import fit_bright_field_disk as gpu_fit_bright_field_disk
    from quantem.gpu.detector import prepare
    from quantem.gpu.formats.qem.metadata import BRIGHT_FIELD_DISK_KEY, BRIGHT_FIELD_DISK_SCHEMA
    from quantem.gpu.device import release_cached_memory as gpu_release_cached_memory
    from quantem.gpu.device import runtime_notice as gpu_runtime_notice
    from quantem.gpu.dpc import run as gpu_dpc_run
    from quantem.gpu.io import Dataset4dstemGPU
except ImportError:
    SSB = gpu_io = prepare = gpu_disk_edge_radius = gpu_fit_bright_field_disk = gpu_release_cached_memory = None
    gpu_runtime_notice = gpu_dpc_run = Dataset4dstemGPU = BRIGHT_FIELD_DISK_KEY = BRIGHT_FIELD_DISK_SCHEMA = None


def available() -> bool:
    """Whether quantem.gpu imported."""
    return gpu_io is not None


def require(feature: str) -> None:
    """Raise ``ImportError`` naming ``feature`` and the install command when quantem.gpu is missing."""
    if not available():
        raise ImportError(f"{feature} needs quantem.gpu, which is not installed. {gpu_path_hint() or no_gpu_path()}.")


def accelerator_ready() -> bool:
    """Whether quantem.gpu can load files on this machine: installed, and a CUDA or Metal backend works."""
    if not available():
        return False
    from quantem.gpu.device import detect

    try:
        detect()
    except RuntimeError:
        return False
    return True


def is_acquisition(obj) -> bool:
    """Whether ``obj`` is an acquisition from ``quantem.gpu.io.load``."""
    return available() and isinstance(obj, Dataset4dstemGPU)


def is_ssb(obj) -> bool:
    """Whether ``obj`` is a ``quantem.gpu.SSB`` session."""
    return available() and isinstance(obj, SSB)


def announce_hot_pixels(record: dict | None) -> None:
    """The one line a widget load prints about the detector pixels its file flags (``hot_pixel_correction``)."""
    if not record or not record.get("pixel_count"):
        return
    count, method = int(record["pixel_count"]), record.get("method")
    if record.get("applied"):
        window = record.get("largest_window_px") or 3
        replaced = ("with the median of their unflagged 3x3 neighbours"
                    + (f" (windows up to {window}x{window} inside flagged clusters)" if window > 3 else "")
                    if method == "median" else "with 0")
        print_once(f"quantem.widget: replaced {count} flagged detector pixels {replaced} "
                   f'(hot_pixel_correction="{method}"); pass hot_pixel_correction="none" to keep the stored counts.')
    else:
        print_once(f"quantem.widget: {count} detector pixels are flagged in the file and left as stored "
                   '(hot_pixel_correction="none"); pass hot_pixel_correction="median" to replace them.')


def load_acquisition(path, device: str | None = None, *, backend: str = "auto", scan_shape=None,
                     hot_pixel_correction: str = "median"):
    """Load one master (or a list) into encoded GPU storage with ``quantem.gpu.io.load``.

    ``device`` is a ``resolve_device`` string passed through: ``"cuda:N"``
    selects that GPU, ``"mps"`` the Apple GPU, None (``device="auto"``) lets
    quantem.gpu choose, and the choice is printed once, as ``resolve_device``
    prints it for arrays. ``hot_pixel_correction`` (``"median"``, ``"zero"`` or
    ``"none"``) treats the pixels the file flags, as ``read_dense`` does, and
    one line says how many were replaced or left. Saved QEM sources retain the
    correction recorded in their metadata.
    """
    require("Loading an acquisition on the GPU")
    loaded = gpu_io.load(
        path, backend=backend, device=device, scan_shape=scan_shape,
        hot_pixel_correction=hot_pixel_correction, verbose=False,
    )
    first = loaded[0] if isinstance(loaded, list) else loaded
    for acquisition in loaded if isinstance(loaded, list) else [loaded]:
        announce_hot_pixels(acquisition.metadata.get("hot_pixel_correction"))
    if device is None:
        print_once(f'quantem.widget: device="auto" selected {device_name(torch.device(first.device))}.')
    return loaded


def acquisition_from_tensor(values_t: torch.Tensor, metadata: dict):
    """A ``Dataset4dstemGPU`` over a 4D tensor already on the GPU, with ``metadata`` as its calibration and name.

    quantem.gpu holds such a tensor as a dense acquisition: indexing, ``read``
    and its detector sessions work as on an encoded one, so a tensor computed
    on the GPU (a strided preview of a loaded file) keeps the GPU dataset type.
    """
    require("A GPU acquisition")
    return Dataset4dstemGPU(values_t, dict(metadata))


def discover_masters(folder, *, pattern: str = "*_master.h5", recursive: bool = True, scan_shape=None) -> list[str]:
    """Readable candidate masters under ``folder`` (``quantem.gpu.io.discover``)."""
    require("Master discovery")
    return list(gpu_io.discover(str(folder), pattern=pattern, recursive=recursive, scan_shape=scan_shape, verbose=False))


def inspect_master(path, *, scan_shape=None):
    """Readiness report of one master (``quantem.gpu.io.inspect``): shapes, dtype, reason, action."""
    require("Master inspection")
    return gpu_io.inspect(str(path), scan_shape=scan_shape)


def header_calibration(metadata: dict) -> dict:
    """The calibration in the header metadata ``inspect_master`` returns, under the ``.qem`` flat names.

    quantem.gpu's header metadata holds the format's flat keys
    ``scan_sampling_A``, one of ``detector_sampling_mrad`` and
    ``detector_sampling_inv_A`` (each ``[row, col]``), ``voltage_kV`` and
    ``semiangle_mrad``; an absent one is None here. A ``.qem`` viewer and the
    command-line export read the header this way, so their scale bars carry
    the file's calibration and not ``1.0 px``.
    """
    return {name: metadata.get(name)
            for name in ("scan_sampling_A", "detector_sampling_mrad", "detector_sampling_inv_A", "voltage_kV", "semiangle_mrad")}


def prepare_session(source):
    """A ``quantem.gpu`` detector session over an acquisition view."""
    require("Encoded detector reductions")
    return prepare(source)


def disk_edge_radius(mean_pattern) -> float:
    """Radius in detector pixels where the bright-field disk falls to half its plateau (``quantem.gpu.detector``).

    quantem.gpu's one rule linking the disk to the convergence semiangle:
    ``semiangle_mrad = disk_edge_radius * detector_sampling_mrad``.
    """
    require("The bright-field disk edge")
    return float(gpu_disk_edge_radius(np.asarray(mean_pattern, dtype=np.float64)))


def fit_bright_field_disk(mean_pattern) -> tuple[tuple[float, float], float]:
    """Bright-field disk ``((center_row, center_col), edge_radius_px)`` of a mean pattern (``quantem.gpu.detector``).

    quantem.gpu's one disk rule, the one ``data.calibration.fit_detector_sampling()`` records and SSB reads.
    """
    require("The bright-field disk fit")
    fit = gpu_fit_bright_field_disk(np.asarray(mean_pattern, dtype=np.float64))
    return fit.bright_field_center_px, fit.bright_field_edge_radius_px


def recorded_bright_field_disk(metadata) -> tuple[tuple[float, float], float] | None:
    """The bright-field disk ``data.calibration.fit_detector_sampling()`` recorded: ``((row, col), edge_radius_px)``, or None."""
    disk = (metadata or {}).get(BRIGHT_FIELD_DISK_KEY) if available() else None
    if not isinstance(disk, dict) or disk.get("schema") != BRIGHT_FIELD_DISK_SCHEMA:
        return None
    center, radius = disk.get("bright_field_center_px"), disk.get("bright_field_edge_radius_px")
    if center is None or radius is None:
        return None
    return (float(center[0]), float(center[1])), float(radius)


def detector_sampling_mrad_from_inv_A(detector_sampling_inv_A, voltage_kV: float) -> tuple[float, float]:
    """A ``(row, col)`` detector step in reciprocal angstrom per pixel as mrad per pixel at ``voltage_kV``.

    The angle of a spatial frequency is the frequency times the electron
    wavelength (``quantem.gpu.optics.wavelength_A_from_kV``).
    """
    require("Converting the detector step to mrad")
    from quantem.gpu.optics import wavelength_A_from_kV

    wavelength_A = wavelength_A_from_kV(float(voltage_kV))
    return tuple(float(step) * wavelength_A * 1e3 for step in detector_sampling_inv_A)


def can_query_together(sources) -> bool:
    """Whether complete count acquisitions share one supported query layout."""
    return bool(sources) and all(
        is_acquisition(source)
        and source.representation.value in {"encoded", "paired"}
        and source.dtype in (np.dtype("uint8"), np.dtype("uint16"))
        for source in sources
    ) and len({source.representation.value for source in sources}) == 1


def aligned_center_of_mass(source, scan_shape, *, scan_detector_rotation_deg: float,
                           scan_detector_rotation_flipped: bool) -> tuple[np.ndarray, np.ndarray]:
    """Centre of mass ``(row, col)`` of every scan position in the scan frame (``quantem.gpu.dpc.run``).

    The mean-subtracted detector centre of mass, its axes swapped first when
    the rotation is flipped, then rotated by the physical scan-detector
    rotation: quantem.gpu's aligned DPC fields at that rotation.
    """
    require("Centre of mass")
    result = gpu_dpc_run(source, scan_shape=scan_shape, scan_detector_rotation_deg=float(scan_detector_rotation_deg),
                         scan_detector_rotation_flipped=bool(scan_detector_rotation_flipped))
    return result.com_row_aligned, result.com_col_aligned


# Fit Regions starts quantem.gpu's joint optics, tilt and depth fit from these seeds and keeps the best;
# the tilt limit is passed to it, so bound_hit tests the limit the fit used.
REGION_FIT_SEEDS = (11, 29, 47, 71, 101)
REGION_TILT_LIMIT_MRAD = 25.0


class SsbSession:
    """A ``quantem.gpu.SSB`` session as the SSB widgets read and drive it.

    quantem.gpu and the widgets use one name per concept
    (``scan_detector_rotation_deg``, ``detector_sampling_mrad``, ``C10_nm`` /
    ``C12_nm`` / ``phi12_deg``, ``bright_field_radius_px``,
    ``num_bright_field_pixels``, ``sample_tilt_mrad``), so arguments reach
    quantem.gpu as given. This class collects what the widgets read from a
    session into plain dicts (``state``, ``trials``, ``browser_state``,
    ``result``); every file and message of the SSB widgets is built from them.
    """

    def __init__(self, ssb):
        self.ssb = ssb

    @classmethod
    def open(cls, source, **parameters):
        """A new session ``quantem.gpu.SSB(source, **parameters)``; use it as a context manager."""
        require("Compute SSB")
        return cls(SSB(source, **parameters))

    def __enter__(self):
        self.ssb.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.ssb.__exit__(exc_type, exc_value, traceback)

    @property
    def num_bright_field_pixels(self) -> int:
        """Pixels of the detected bright-field disk; the first read prepares the session."""
        return int(self.ssb.num_bright_field_pixels)

    @property
    def bright_field_radius_px(self) -> float:
        """The bright-field radius the session selects with, in detector pixels, given or detected.

        A radius quantem.gpu detected is read from its prepared backend
        (``browser_state``); the first read prepares the session.
        """
        radius = self.ssb.bright_field_radius_px
        if radius is None:
            radius = self.ssb.browser_state().brightfield.bright_field_radius_px
        return float(radius)

    @property
    def detector_sampling_mrad(self) -> list[float]:
        """The detector step ``[row, col]`` in mrad per pixel along the original detector axes, given or estimated."""
        sampling = self.ssb.detector_sampling_mrad
        if sampling is None:
            # the backend holds the estimate in the axes it computes in, swapped when the rotation is flipped
            sampling = [float(step) * 1e3 for step in self.ssb.browser_state().angular_sampling_rad]
            if self.ssb.scan_detector_rotation_flipped:
                sampling = sampling[::-1]
        return [float(step) for step in sampling]

    @property
    def bright_field_intensity_threshold(self) -> float:
        """Fraction of the mean-pattern maximum a bright-field pixel exceeds, as the session uses it."""
        return float(self.ssb.bright_field_intensity_threshold)

    @property
    def scan_detector_rotation_flipped(self) -> bool:
        """Whether the session swaps the detector axes before rotating them onto the scan."""
        return bool(self.ssb.scan_detector_rotation_flipped)

    @property
    def scan_offset(self) -> tuple[int, int]:
        """``(row, col)`` of the first measured position in the prepared Fourier grid.

        quantem.gpu centres a scan its FFT cannot take (CUDA: not 128, 256,
        512 or 1024 square; MPS: rectangular) in the next supported square,
        ``(target - size) // 2`` from each edge. ``scan_shape`` is that padded
        grid and ``input_scan_shape`` the measured one; (0, 0) without padding.
        """
        padded, measured = self.ssb.scan_shape, self.ssb.input_scan_shape
        return tuple(int(full - native) // 2 for full, native in zip(padded, measured))

    def select_region(self, scan_region, *, aberrations: dict, scan_detector_rotation_deg: float) -> "SsbSession":
        """A separately owned session over ``scan_region`` of this scan (``quantem.gpu.SSB.select_region``).

        ``scan_region`` is relative to this session's scan; the new session's
        ``scan_region`` is in full-scan positions. Close it with ``.ssb.close()``.
        """
        return SsbSession(self.ssb.select_region(
            tuple(int(value) for value in scan_region), aberrations=dict(aberrations),
            scan_detector_rotation_deg=float(scan_detector_rotation_deg)))

    def crop_native(self, phase, bounds, upsampling_factor: int) -> np.ndarray:
        """A host copy of ``bounds`` (native scan positions) from a full-scan ``phase``.

        ``phase`` covers the padded grid at ``upsampling_factor``; the bounds are
        shifted by ``scan_offset`` so a region never shows padding.
        """
        row_offset, col_offset = self.scan_offset
        row_start, row_stop, col_start, col_stop = (
            int(value) * upsampling_factor for value in (
                bounds[0] + row_offset, bounds[1] + row_offset,
                bounds[2] + col_offset, bounds[3] + col_offset))
        crop = phase[row_start:row_stop, col_start:col_stop]
        return crop.get() if hasattr(crop, "get") else np.array(crop, copy=True)

    def state(self) -> dict:
        """Geometry, optics and current fit of the session.

        ``scan_sampling_A`` and ``detector_sampling_mrad`` are ``[row, col]``
        lists and the rotation is the physical angle in [0, 360). Every file and
        message of the SSB widgets is built from this dict, so they cannot
        disagree on a name or a unit.
        """
        ssb = self.ssb
        return {
            "backend": ssb.backend, "scan_shape": tuple(int(size) for size in ssb.scan_shape), "source_path": ssb.source_path,
            "input_scan_shape": ssb.input_scan_shape,
            "voltage_kV": ssb.voltage_kV, "semiangle_mrad": ssb.semiangle_mrad, "scan_sampling_A": list(ssb.scan_sampling_A),
            "detector_sampling_mrad": None if ssb.detector_sampling_mrad is None else list(ssb.detector_sampling_mrad),
            "scan_detector_rotation_deg": ssb.scan_detector_rotation_deg, "scan_detector_rotation_flipped": ssb.scan_detector_rotation_flipped,
            "aberrations": dict(ssb.aberrations),
            "loss": float(ssb.best_loss) if math.isfinite(ssb.best_loss) else None,
            "bright_field_radius_px": ssb.bright_field_radius_px,
            "bright_field_intensity_threshold": ssb.bright_field_intensity_threshold,
            "num_bright_field_pixels": self.num_bright_field_pixels,
            "sample_tilt_mrad": None if ssb.sample_tilt_mrad is None else list(ssb.sample_tilt_mrad),
            "depth_spread_nm": ssb.depth_spread_nm,
        }

    def trials(self) -> list[dict]:
        """The fit's trials, in the order they ran, as ``{C10_nm, C12_nm, phi12_deg, sample_tilt_mrad,
        depth_spread_nm, objective, loss}``.

        ``objective`` is quantem.gpu's name for what the trial was scored on: ``"agreement"`` (the default
        model-based fit, with or without tilt; its ``loss`` is the negative agreement, lower is better) or
        ``"phase_variance"`` (the backup search). ``sample_tilt_mrad`` (scan frame) and ``depth_spread_nm`` are the
        trial's sample model, 0 for a tilt-0 trial. Losses compare only within one objective.
        """
        return [{"C10_nm": float(trial["params"].get("C10_nm", 0.0)), "C12_nm": float(trial["params"].get("C12_nm", 0.0)),
                 "phi12_deg": float(trial["params"].get("phi12_deg", 0.0)),
                 "sample_tilt_mrad": [float(value) for value in trial.get("sample_tilt_mrad") or (0.0, 0.0)],
                 "depth_spread_nm": float(trial.get("depth_spread_nm") or 0.0),
                 # quantem.gpu tags the agreement trials "negative_thick_agreement" (the loss is its negative)
                 "objective": "phase_variance" if trial.get("objective", "phase_variance") == "phase_variance" else "agreement",
                 "loss": trial.get("loss")}
                for trial in self.ssb.trial_history]

    def calibration_identity(self) -> dict | None:
        """Identify the loaded evidence and launch geometry without copying counts or naming a path.

        The source is quantem.gpu's portable ``source_identity`` (file names,
        sizes, SHA-256 of each file's first MiB, scan and detector shapes, how
        flagged pixels were filled), so a saved calibration is restored after
        its data folder is moved or copied and refused for another acquisition.
        Array inputs have no persistent identity: a filename attached to a
        modified array does not make it the acquisition on disk.
        """
        from quantem.gpu.ssb.persistence import json_value, source_identity

        ssb = self.ssb
        if ssb.source_kind == "array" or ssb.source_path is None:
            return None
        state = self.state()  # Resolve automatic probe calibration first.
        return json_value({
            "source": source_identity(
                ssb.source_path, scan_shape=ssb.input_scan_shape,
                hot_pixel_correction=(ssb.source_provenance or {}).get("hot_pixel_correction"),
                storage_path=ssb.source_storage_path,
                calibration_path=ssb.calibration_path,
                manifest_path=ssb.source_manifest_path,
            ),
            "data": {"kind": ssb.source_kind, "dtype": ssb.source_dtype,
                     "bytes": ssb.source_bytes, "scan_shape": ssb.scan_shape,
                     "scan_region": ssb.scan_region,
                     "detector_bin": ssb.source_detector_bin},
            "geometry": {
                **{name: state[name] for name in (
                    "voltage_kV", "semiangle_mrad", "scan_sampling_A",
                    "detector_sampling_mrad", "scan_detector_rotation_deg",
                    "scan_detector_rotation_flipped", "bright_field_radius_px",
                    "bright_field_intensity_threshold", "num_bright_field_pixels",
                )},
                "bright_field_center_px": ssb.bright_field_center_px,
            },
        })

    @staticmethod
    def same_calibration_identity(saved: dict, current: dict) -> bool:
        """Whether a saved ``calibration_identity`` names this session's evidence and launch geometry.

        Every entry must be equal except the scan-detector rotation, which is
        compared modulo 180 degrees (``quantem.gpu.geometry.rotation_gap_deg``):
        the fit's rotation check moves a session to the other 180-degree
        branch, so a session opened before that move must find a save made
        after it, and the other way round. Restore then applies the saved
        rotation, branch included.
        """
        from quantem.gpu.geometry import rotation_gap_deg

        name = "scan_detector_rotation_deg"
        saved_geometry, geometry = dict(saved.get("geometry") or {}), dict(current.get("geometry") or {})
        saved_rotation, rotation = saved_geometry.pop(name, None), geometry.pop(name, None)
        if {**saved, "geometry": saved_geometry} != {**current, "geometry": geometry}:
            return False
        if saved_rotation is None or rotation is None:
            return saved_rotation is None and rotation is None
        return rotation_gap_deg(float(saved_rotation), float(rotation), period_deg=180.0) < 1e-6

    def restore_calibration(self, saved: dict) -> None:
        """Adopt a validated complete correction without running a new search.

        Assigning ``ssb.aberrations`` states them as known (quantem.gpu), so a
        later ``ssb.reconstruct()`` applies them instead of fitting.
        """
        from quantem.gpu.ssb.units import validate_aberrations

        aberrations = validate_aberrations(saved["aberrations"], higher_order=True)
        self.set_scan_detector_rotation(saved["scan_detector_rotation_deg"])
        self.ssb.aberrations = aberrations  # the saved model replaces every term, higher orders included
        self.store(aberrations, saved["loss"], sample=(
            {"sample_tilt_mrad": saved["sample_tilt_mrad"],
             "depth_spread_nm": saved["depth_spread_nm"]}
            if saved["depth_spread_nm"] is not None else None
        ))

    def propagator_tilt_mrad(self, sample_tilt_mrad, scan_detector_rotation_deg: float) -> list[float]:
        """A scan-frame sample tilt along the detector axes, the ptychographic propagator's frame.

        ``quantem.gpu.geometry.sample_to_propagator_tilt`` at
        ``scan_detector_rotation_deg`` and this session's flip.
        """
        from quantem.gpu.geometry import sample_to_propagator_tilt

        return list(sample_to_propagator_tilt(
            tuple(sample_tilt_mrad), scan_detector_rotation_deg=float(scan_detector_rotation_deg),
            scan_detector_rotation_flipped=self.scan_detector_rotation_flipped))

    def set_scan_detector_rotation(self, scan_detector_rotation_deg: float) -> None:
        """Set the physical scan-to-detector rotation and refresh the session's geometry."""
        self.ssb.set_rotation(float(scan_detector_rotation_deg))

    def store(self, aberrations: dict, loss: float | None, *, sample: dict | None = None) -> None:
        """Keep reviewed optics and sample correction for the next reconstruction.

        The whole mapping is assigned, quantem.gpu's way to state known
        aberrations: an in-place update left an unfitted session to fit again
        on ``reconstruct()``, discarding the reviewed values.
        """
        self.ssb.aberrations = {**self.ssb.aberrations, **aberrations}
        self.ssb.sample_tilt_mrad = None if sample is None else tuple(sample["sample_tilt_mrad"])
        self.ssb.depth_spread_nm = None if sample is None else float(sample["depth_spread_nm"])
        if loss is not None:
            self.ssb.best_loss = float(loss)

    def preview_context(self, num_bright_field_pixels: int):
        """A reduced bright-field context for slider drags (``quantem.gpu.SSB.preview_context``)."""
        return self.ssb.preview_context(int(num_bright_field_pixels))

    def preview(self, aberrations: dict, **options):
        """Transient ``(phase, loss)`` at ``aberrations`` (``quantem.gpu.SSB.preview(aberrations, **options)``).

        ``aberrations`` holds ``C10_nm``, ``C12_nm``, ``phi12_deg`` and the
        higher-order panel's terms (``C21_nm`` with ``phi21_deg``; ``C30_nm`` and
        ``C50_nm`` have no azimuth). The higher orders reach quantem.gpu only
        while one of their magnitudes is non-zero: a named term, even at zero,
        selects quantem.gpu's 14-slot path, which holds C10, C12 and phi12 in
        float32, so a panel left at zero changed the last bits of most pixels of
        the two-term phase.
        """
        higher_order = any(value != 0.0 for name, value in aberrations.items() if name.endswith("_nm") and name not in ("C10_nm", "C12_nm"))
        primary = aberrations if higher_order else {name: aberrations[name] for name in ("C10_nm", "C12_nm", "phi12_deg")}
        return self.ssb.preview(primary, **options)

    def browser_state(self) -> dict:
        """The compact state a browser SSB engine rebuilds its reducers from.

        quantem.gpu's ``SSBExportState``: the bright-field pixel list
        (``bright_field_rows`` / ``_cols``), their reciprocal coordinates in the
        scan frame (``bright_field_k_row`` / ``_k_col``, quantem.gpu's ``kx_bf``
        / ``ky_bf``: its x is the scan row axis), the scan frequencies
        (``scan_q_row`` / ``scan_q_col``, its ``qx_1d`` / ``qy_1d``) and the
        aperture terms, with the detector step in mrad. ``ShowPtycho.export``
        writes ``snapshots/cal.json`` from this dict alone.
        """
        state = self.ssb.browser_state()
        return {
            "scan_shape": tuple(int(size) for size in state.scan_shape),
            "num_bright_field_pixels": int(state.num_bright_field_pixels),
            "bright_field_radius_px": state.brightfield.bright_field_radius_px,
            "bright_field_center_px": list(state.brightfield.bright_field_center_px),
            "detector_shape": list(state.brightfield.detector_shape),
            "bright_field_rows": state.brightfield.rows, "bright_field_cols": state.brightfield.cols,
            "bright_field_k_row": state.kx_bf, "bright_field_k_col": state.ky_bf,
            "scan_q_row": state.qx_1d, "scan_q_col": state.qy_1d,
            "aperture_k": state.aperture_k, "alpha_k2": state.alpha_k2,
            "cos2phi_k": state.cos2phi_k, "sin2phi_k": state.sin2phi_k,
            "wavelength_A": state.wavelength_A,
            "detector_sampling_mrad": [float(step) * 1e3 for step in state.angular_sampling_rad],
            "dc_value": state.dc_value,
            "bright_field_source_path": state.bf_source_path, "bright_field_source_dtype": state.bf_source_dtype,
            "bright_field_source_max_value": state.bf_source_max_value,
            "qem_source": state.qem_source,
        }

    def export_bright_field(self, path) -> None:
        """Write the exact bright-field counts beside an exported page (``quantem.gpu.SSB.export_brightfield``)."""
        self.ssb.export_brightfield(path)

    def reconstruct(self) -> dict:
        """Reconstruct at the session's aberrations (``quantem.gpu.SSB.reconstruct``); see ``result`` for the keys."""
        return self.result(self.ssb.reconstruct())

    def find_aberrations(self, **options) -> dict:
        """Fit the aberrations (``quantem.gpu.SSB.find_aberrations(**options)``); see ``result`` for the keys."""
        return self.result(self.ssb.find_aberrations(**options))

    def fit_regions(self, regions: list[dict], parameters: dict, *,
                    upsampling_factor: int, phase_estimator: str,
                    num_trials: int = 200, refinement: str | None = "newton",
                    progress: Callable[[dict], None] | None = None) -> dict:
        """Jointly fit regional models and crop their full-scan reconstructions.

        All numerical fitting and reconstruction use the ordinary SSB backend.
        Independent regional sessions leave the overview calibration intact.
        Detector geometry and rotation are common; aberrations, both tilt
        components and depth spread are optimized together for each region,
        ``quantem.gpu.SSB.find_aberrations(tilt=True, num_trials=, refinement=)``
        from each of ``REGION_FIT_SEEDS``, keeping the best sample agreement.
        A fit within 0.01 of the tilt limit or of quantem.gpu's depth-spread
        search range (``quantem.gpu.ssb.DEPTH_SPREAD_RANGE_NM``) is flagged ``bound_hit``.
        """
        ssb = self.ssb
        if ssb.scan_region is not None or not ssb.source_path:
            raise ValueError(
                "Local fitting needs the full acquisition. Open it with "
                "SSB.open(path, ...) or pass source_path to SSB(data, ...).")
        optics = parameters["aberrations"]
        if set(optics) != {"C10_nm", "C12_nm", "phi12_deg"}:
            raise ValueError(
                "Joint regional fits optimize C10, C12, tilt and depth. "
                "Set higher-order coefficients to zero before Fit Regions.")
        if ssb.scan_sampling_A[0] != ssb.scan_sampling_A[1]:
            raise ValueError("Regional display currently needs equal row and column scan sampling.")
        started = time.perf_counter()
        fits, images = [], {}
        seeds = REGION_FIT_SEEDS
        from quantem.gpu.ssb import DEPTH_SPREAD_RANGE_NM as depth_range_nm  # the joint fit's own depth search range
        factor = int(upsampling_factor)
        geometry = {
            "backend": ssb.backend, "voltage_kV": ssb.voltage_kV,
            "semiangle_mrad": ssb.semiangle_mrad,
            "scan_sampling_A": ssb.scan_sampling_A,
            "detector_sampling_mrad": ssb.detector_sampling_mrad,
            "bright_field_center_px": ssb.bright_field_center_px,
            "bright_field_radius_px": ssb.bright_field_radius_px,
            "bright_field_intensity_threshold": ssb.bright_field_intensity_threshold,
            "scan_detector_rotation_flipped": ssb.scan_detector_rotation_flipped,
            "scan_detector_rotation_deg": parameters["scan_detector_rotation_deg"],
        }
        for region_index, region in enumerate(regions, start=1):
            region_started = time.perf_counter()
            bounds = tuple(region["bounds"])
            def report(stage: str, repeat: int) -> None:
                if progress is not None:
                    progress({"label": region["label"], "region_id": region["id"],
                              "region_index": region_index, "region_count": len(regions),
                              "stage": stage, "repeat": repeat, "repeats": len(seeds),
                              "num_trials": int(num_trials), "refinement": refinement,
                              "elapsed_s": time.perf_counter() - started})
            report("preparing", 0)
            with ssb.select_region(
                bounds, aberrations=optics,
                scan_detector_rotation_deg=parameters["scan_detector_rotation_deg"],
            ) as regional:
                runs = []
                for repeat, seed in enumerate(seeds, start=1):
                    report("joint_fit", repeat)
                    fitted = regional.find_aberrations(
                        tilt=True, num_trials=int(num_trials), refinement=refinement,
                        tilt_limit_mrad=REGION_TILT_LIMIT_MRAD,
                        check_rotation=False, compare_standard=False,
                        seed=seed, verbose=False)
                    runs.append({
                        "seed": seed, "aberrations": dict(fitted.aberrations),
                        "sample_tilt_mrad": list(fitted.sample_tilt_mrad),
                        "depth_spread_nm": float(fitted.depth_spread_nm),
                        "sample_agreement": float(fitted.sample_agreement),
                        "num_completed_trials": len(fitted.trial_records or ()),
                        "refinement": fitted.refinement,
                        "optimizer": dict(getattr(fitted, "optimizer", {})),
                        "timings": dict(getattr(fitted, "timings", {})),
                    })
            # A NaN agreement compares false both ways, so max() would keep a
            # nonfinite first seed over every finite one.
            finite = [run for run in runs if math.isfinite(run["sample_agreement"])]
            if not finite:
                raise ValueError(f"{region['label']}: nonfinite fit. Select another region and refit.")
            best = max(finite, key=lambda run: run["sample_agreement"])
            report("reconstructing", len(seeds))
            phase, _ = self.preview(
                best["aberrations"], sample_tilt_mrad=best["sample_tilt_mrad"],
                depth_spread_nm=best["depth_spread_nm"],
                upsampling_factor=factor, phase_estimator=phase_estimator,
                compute_loss=False)
            if tuple(phase.shape) != tuple(size * factor for size in ssb.scan_shape):
                raise ValueError("Local correction requires a full-scan reconstruction before cropping.")
            images[f"{region['label']} · SSB"] = self.crop_native(phase, bounds, factor)
            tilts = np.asarray([run["sample_tilt_mrad"] for run in runs])
            fits.append({
                **region, **best,
                "seed_sd_mrad": float(np.sqrt(np.var(tilts, axis=0, ddof=1).sum())),
                "fit_seconds": time.perf_counter() - region_started,
                "bound_hit": (max(abs(v) for v in best["sample_tilt_mrad"]) > REGION_TILT_LIMIT_MRAD - 0.01
                              or best["depth_spread_nm"] < depth_range_nm[0] + 0.01
                              or best["depth_spread_nm"] > depth_range_nm[1] - 0.01),
                "fit_record": {"method": "joint optics, tilt and depth", "fits": runs,
                               "backend": ssb.backend, "num_trials": int(num_trials),
                               "refinement": refinement},
                "reconstruction": {
                    "scan_shape": list(ssb.scan_shape), "scan_offset": list(self.scan_offset),
                    "scan_region": None,
                    "display_bounds": list(bounds), "upsampling_factor": factor,
                    "phase_estimator": phase_estimator,
                    "num_bright_field_pixels": self.num_bright_field_pixels,
                },
            })
        return {
            "regions": regions, "fits": fits, "images": images,
            "sampling_A": ssb.scan_sampling_A[0] / factor,
            "result_kind": "Joint optics, tilt and depth · full-scan SSB",
            "reference_parameters": parameters, "shared_geometry": geometry,
            "map": fits, "last_fit_seconds": time.perf_counter() - started,
        }

    @staticmethod
    def result(result) -> dict:
        """The values of a quantem.gpu ``SSBResult`` the widgets show and store.

        ``phase``, ``aberrations`` (``C10_nm``, ``C12_nm``, ``phi12_deg``),
        ``scan_detector_rotation_deg`` (the physical angle in [0, 360)),
        ``loss``, ``elapsed_s``, ``refinement`` (the optimizer a fit used, None
        without one), ``objective`` (what a fit optimized, None without one),
        and with a sample model ``sample_tilt_mrad``,
        ``depth_spread_nm`` and ``tilt_fit_gain``.
        """
        fitted = {
            "phase": result.phase, "aberrations": dict(result.aberrations),
            "scan_detector_rotation_deg": result.scan_detector_rotation_deg,
            "loss": None if result.loss is None else float(result.loss),
            "elapsed_s": None if result.elapsed_s is None else float(result.elapsed_s),
            "refinement": getattr(result, "refinement", None),
            "objective": getattr(result, "objective", None),
        }
        if result.sample_tilt_mrad is not None:
            fitted.update(sample_tilt_mrad=[float(value) for value in result.sample_tilt_mrad],
                          depth_spread_nm=float(result.depth_spread_nm),
                          tilt_fit_gain=None if result.tilt_fit_gain is None else float(result.tilt_fit_gain))
        return fitted


class AcquisitionView:
    """One acquisition, or a scan region of it, read in bounded windows.

    ``source`` is a ``Dataset4dstemGPU`` from ``io.load`` or a 4D tensor on the
    GPU. ``quantem.gpu.detector.prepare`` recognises the marker attributes and
    sums virtual detectors on the encoded counts; reads return float32 with the
    detector pixels the source still flags (an acquisition loaded with
    ``hot_pixel_correction="none"``) set to zero, which one line announces.
    """

    _bounded_detector_source = True

    def __init__(self, source, region=None):
        self.source = source
        self.region = region or (0, source.shape[0], 0, source.shape[1])
        row_start, row_stop, col_start, col_stop = self.region
        if not (0 <= row_start < row_stop <= source.shape[0] and 0 <= col_start < col_stop <= source.shape[1]):
            raise ValueError(f"Scan region {self.region} is outside {tuple(source.shape[:2])}.")
        self.shape = (row_stop - row_start, col_stop - col_start, *source.shape[2:])
        device = torch.device(source.device)
        self.device = torch.device(device.type, device.index or 0)
        self._detector_source = source
        self._detector_region = self.region
        self.valid = None
        if is_acquisition(source):
            session = prepare(source)
            validity = session.detector_validity
            session.close()
            if validity is not None:
                self.valid = torch.as_tensor(validity, device=self.device).reshape(self.shape[-2:])
                flagged = int((~self.valid).sum())
                if flagged:
                    print_once(f"quantem.widget: {flagged} detector pixels the acquisition flags are shown as 0; load it "
                               "with quantem.gpu.io.load's default hot_pixel_correction (\"median\") to replace them.")

    def read(self, *, scan_region):
        """``scan_region`` relative to this view, as a float32 tensor on the source GPU."""
        row_start, row_stop, col_start, col_stop = scan_region
        row_offset, _, col_offset, _ = self.region
        region = (row_start + row_offset, row_stop + row_offset, col_start + col_offset, col_stop + col_offset)
        if torch.is_tensor(self.source):
            return self.source[region[0] : region[1], region[2] : region[3]]
        values_t = self.source.read(scan_region=region).float()
        return values_t if self.valid is None else values_t.masked_fill(~self.valid, 0)


class AcquisitionSeries:
    """Several acquisitions of one scan and detector geometry as a growing series."""

    def __init__(self, sources):
        self.frames = [AcquisitionView(source) for source in sources]
        first = self.frames[0]
        if any(frame.shape != first.shape or frame.device != first.device for frame in self.frames):
            raise ValueError("Comparison sources must share scan/detector shape and device.")
        self.device = first.device

    @property
    def shape(self) -> tuple[int, ...]:
        """``(n_acquisitions, scan_rows, scan_cols, det_rows, det_cols)``."""
        return (len(self.frames), *self.frames[0].shape)

    def append(self, source) -> None:
        """Add one acquisition that arrived while a folder is watched."""
        frame = AcquisitionView(source)
        if frame.shape != self.frames[0].shape or frame.device != self.device:
            raise ValueError(
                f"Acquisition shape {frame.shape} on {frame.device} does not match "
                f"{self.frames[0].shape} on {self.device}."
            )
        self.frames.append(frame)
