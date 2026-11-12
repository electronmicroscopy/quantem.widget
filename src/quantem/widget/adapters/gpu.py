"""quantem.gpu acquisitions, detector sessions and direct ptychography, whether or not quantem.gpu is installed.

quantem.gpu is optional (``quantem.widget[cuda]`` or ``quantem.widget[mps]``). This is the only
module that imports it: encoded acquisitions from ``quantem.gpu.io.load``, their
detector sessions, master discovery and inspection, direct ptychography, centre of mass and the
release of cached GPU memory.
Without quantem.gpu every name below is None, ``available()`` is False and a
function that needs it raises ``ImportError`` with the install command; the
widgets then read files densely with ``show4dstem.reader`` and reduce with
their own torch code. An installed quantem.gpu is imported unguarded: a name it
renamed or a module it cannot import fails here, loudly, instead of reading as
"not installed" and silently routing a session through the array path.
"""

import math
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from quantem.widget.device import device_name, gpu_path_hint, module_installed, no_gpu_path, print_once

if module_installed("quantem.gpu"):
    from quantem.gpu import DirectPtycho
    from quantem.gpu import io as gpu_io
    from quantem.gpu.detector import bright_field_disk as gpu_bright_field_disk
    from quantem.gpu.detector import disk_edge_radius as gpu_disk_edge_radius
    from quantem.gpu.detector import prepare
    from quantem.gpu.formats.qem.metadata import BRIGHT_FIELD_DISK_KEY, BRIGHT_FIELD_DISK_SCHEMA
    from quantem.gpu.device import release_cached_memory as gpu_release_cached_memory
    from quantem.gpu.device import runtime_notice as gpu_runtime_notice
    from quantem.gpu.dpc import run as gpu_dpc_run
    from quantem.gpu.io import Dataset4dstem
else:
    DirectPtycho = gpu_io = prepare = gpu_disk_edge_radius = gpu_bright_field_disk = gpu_release_cached_memory = None
    gpu_runtime_notice = gpu_dpc_run = Dataset4dstem = BRIGHT_FIELD_DISK_KEY = BRIGHT_FIELD_DISK_SCHEMA = None


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
    from quantem.gpu.device import resolve

    try:
        resolve("auto")
    except RuntimeError:
        return False
    return True


def is_acquisition(obj) -> bool:
    """Whether ``obj`` is an acquisition from ``quantem.gpu.io.load``."""
    return available() and isinstance(obj, Dataset4dstem)


def is_direct_ptycho(obj) -> bool:
    """Whether ``obj`` is a ``quantem.gpu.DirectPtycho`` session."""
    return available() and isinstance(obj, DirectPtycho)


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


def load_acquisition(path, device: str | None = None, *, scan_shape=None, hot_pixel_correction: str = "median"):
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
        path, device=device, scan_shape=scan_shape,
        hot_pixel_correction=hot_pixel_correction, verbose=False,
    )
    first = loaded[0] if isinstance(loaded, list) else loaded
    for acquisition in loaded if isinstance(loaded, list) else [loaded]:
        announce_hot_pixels(acquisition.metadata.get("hot_pixel_correction"))
    if device is None:
        print_once(f'quantem.widget: device="auto" selected {device_name(torch.device(first.device))}.')
    return loaded


def acquisition_from_tensor(values_t: torch.Tensor, metadata: dict):
    """A ``Dataset4dstem`` over a 4D tensor already on the GPU, with ``metadata`` as its calibration and name.

    quantem.gpu holds such a tensor as a dense acquisition: indexing, ``read``
    and its detector sessions work as on an encoded one, so a tensor computed
    on the GPU (a strided preview of a loaded file) keeps the GPU dataset type.
    """
    require("A GPU acquisition")
    return Dataset4dstem(values_t, dict(metadata))


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


def bright_field_disk(mean_pattern) -> tuple[tuple[float, float], float]:
    """Bright-field disk ``((center_row, center_col), edge_radius_px)`` of a mean pattern (``quantem.gpu.detector``).

    quantem.gpu's one disk rule, the one ``data.calibration.detector_sampling()`` records and direct ptychography reads.
    """
    require("The bright-field disk fit")
    fit = gpu_bright_field_disk(np.asarray(mean_pattern, dtype=np.float64))
    return fit.bright_field_center_px, fit.bright_field_edge_radius_px


def recorded_bright_field_disk(metadata) -> tuple[tuple[float, float], float] | None:
    """The bright-field disk ``data.calibration.detector_sampling()`` recorded: ``((row, col), edge_radius_px)``, or None."""
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


# Fit Regions uses the same fixed seed as quantem.gpu's ordinary joint fit;
# the tilt limit is passed to it, so bound_hit tests the limit the fit used.
REGION_FIT_SEED = 42
REGION_TILT_LIMIT_MRAD = 25.0


def _finite_or_none(value) -> float | None:
    return float(value) if value is not None and math.isfinite(float(value)) else None


class DirectSession:
    """A ``quantem.gpu.DirectPtycho`` session as the direct ptychography widgets read and drive it.

    quantem.gpu and the widgets use one name per concept
    (``scan_detector_rotation_deg``, ``detector_sampling_mrad``, ``C10_nm`` /
    ``C12_nm`` / ``phi12_deg``, ``bright_field_radius_px``,
    ``num_bright_field_pixels``, ``sample_tilt_mrad``), so arguments reach
    quantem.gpu as given. This class collects what the widgets read from a
    session into plain dicts (``state``, ``trials``, ``browser_state``,
    ``result``); every file and message of the direct ptychography widgets is built from them.
    """

    def __init__(self, direct):
        self.direct = direct

    @classmethod
    def open(cls, source, **parameters):
        """A new session ``quantem.gpu.DirectPtycho(source, **parameters)``; use it as a context manager."""
        require("Direct ptychography")
        return cls(DirectPtycho(source, **parameters))

    def __enter__(self):
        self.direct.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.direct.__exit__(exc_type, exc_value, traceback)

    @property
    def num_bright_field_pixels(self) -> int:
        """Pixels of the detected bright-field disk; the first read prepares the session."""
        return int(self.direct.num_bright_field_pixels)

    @property
    def bright_field_radius_px(self) -> float:
        """The bright-field radius the session selects with, in detector pixels, given or detected.

        A radius quantem.gpu detected is read from its prepared backend
        (``browser_state``); the first read prepares the session.
        """
        radius = self.direct.bright_field_radius_px
        if radius is None:
            radius = self.direct.browser_state().brightfield.bright_field_radius_px
        return float(radius)

    @property
    def detector_sampling_mrad(self) -> list[float]:
        """The detector step ``[row, col]`` in mrad per pixel along the original detector axes, given or estimated."""
        sampling = self.direct.detector_sampling_mrad
        if sampling is None:
            # the backend holds the estimate in the axes it computes in, swapped when the rotation is flipped
            sampling = [float(step) * 1e3 for step in self.direct.browser_state().angular_sampling_rad]
            if self.direct.scan_detector_rotation_flipped:
                sampling = sampling[::-1]
        return [float(step) for step in sampling]

    @property
    def bright_field_intensity_threshold(self) -> float:
        """Fraction of the mean-pattern maximum a bright-field pixel exceeds, as the session uses it."""
        return float(self.direct.bright_field_intensity_threshold)

    @property
    def scan_detector_rotation_flipped(self) -> bool:
        """Whether the session swaps the detector axes before rotating them onto the scan."""
        return bool(self.direct.scan_detector_rotation_flipped)

    @property
    def scan_offset(self) -> tuple[int, int]:
        """``(row, col)`` of the first measured position in the prepared Fourier grid.

        quantem.gpu centres a scan its FFT cannot take (CUDA: not 128, 256,
        512 or 1024 square; MPS: rectangular) in the next supported square,
        ``(target - size) // 2`` from each edge. ``scan_shape`` is that padded
        grid and ``input_scan_shape`` the measured one; (0, 0) without padding.
        """
        padded, measured = self.direct.scan_shape, self.direct.input_scan_shape
        return tuple(int(full - native) // 2 for full, native in zip(padded, measured))

    def select_region(self, scan_region, *, aberrations: dict, scan_detector_rotation_deg: float) -> "DirectSession":
        """A separately owned session over ``scan_region`` of this scan (``quantem.gpu.DirectPtycho.select_region``).

        ``scan_region`` is relative to this session's scan; the new session's
        ``scan_region`` is in full-scan positions. Close it with ``.direct.close()``.
        """
        return DirectSession(self.direct.select_region(
            tuple(int(value) for value in scan_region), aberrations=dict(aberrations),
            scan_detector_rotation_deg=float(scan_detector_rotation_deg)))

    def crop_native(self, phase, bounds, upsample: int) -> np.ndarray:
        """A host copy of ``bounds`` (native scan positions) from a full-scan ``phase``.

        ``phase`` covers the padded grid at ``upsample``; the bounds are
        shifted by ``scan_offset`` so a region never shows padding.
        """
        row_offset, col_offset = self.scan_offset
        row_start, row_stop, col_start, col_stop = (
            int(value) * upsample for value in (
                bounds[0] + row_offset, bounds[1] + row_offset,
                bounds[2] + col_offset, bounds[3] + col_offset))
        crop = phase[row_start:row_stop, col_start:col_stop]
        return crop.get() if hasattr(crop, "get") else np.array(crop, copy=True)

    def state(self) -> dict:
        """Geometry, optics and current fit of the session.

        ``scan_sampling_A`` and ``detector_sampling_mrad`` are ``[row, col]``
        lists and the rotation is the physical angle in [0, 360). Every file and
        message of the direct ptychography widgets is built from this dict, so they cannot
        disagree on a name or a unit.
        """
        direct = self.direct
        fit = getattr(direct, "_aberration_search", None)
        return {
            "backend": direct.backend, "scan_shape": tuple(int(size) for size in direct.scan_shape), "source_path": direct.source_path,
            "input_scan_shape": direct.input_scan_shape,
            "voltage_kV": direct.voltage_kV, "semiangle_mrad": direct.semiangle_mrad, "scan_sampling_A": list(direct.scan_sampling_A),
            "detector_sampling_mrad": None if direct.detector_sampling_mrad is None else list(direct.detector_sampling_mrad),
            "scan_detector_rotation_deg": direct.scan_detector_rotation_deg, "scan_detector_rotation_flipped": direct.scan_detector_rotation_flipped,
            "aberrations": dict(direct.known_aberrations),
            "loss": float(direct.best_loss) if math.isfinite(direct.best_loss) else None,
            "bright_field_radius_px": direct.bright_field_radius_px,
            "bright_field_intensity_threshold": direct.bright_field_intensity_threshold,
            "num_bright_field_pixels": self.num_bright_field_pixels,
            "sample_tilt_mrad": None if direct.sample_tilt_mrad is None else list(direct.sample_tilt_mrad),
            "depth_spread_nm": direct.depth_spread_nm,
            # the kept fit's model agreement (higher is better) and its gain over standard direct ptychography; None before a fit
            "sample_agreement": _finite_or_none(getattr(fit, "sample_agreement", None)),
            "tilt_fit_gain": _finite_or_none(getattr(fit, "tilt_fit_gain", None)),
        }

    def trials(self) -> list[dict]:
        """The fit's trials, in the order they ran, as ``{C10_nm, C12_nm, phi12_deg, sample_tilt_mrad,
        depth_spread_nm, scan_detector_rotation_deg, higher_order, objective, loss}``.

        ``objective`` is quantem.gpu's name for what the trial was scored on: ``"agreement"`` (the default
        model-based fit, with or without tilt; its ``loss`` is the negative agreement, lower is better) or
        ``"phase_variance"`` (quantem.gpu's phase-variance search, for a session fitted with it). ``sample_tilt_mrad`` (scan frame) and ``depth_spread_nm`` are the
        trial's sample model, 0 for a tilt-0 trial. Losses compare only within one objective.
        """
        return [{"C10_nm": float(trial["params"].get("C10_nm", 0.0)), "C12_nm": float(trial["params"].get("C12_nm", 0.0)),
                 "phi12_deg": float(trial["params"].get("phi12_deg", 0.0)),
                 "higher_order": {key: float(value) for key, value in trial["params"].items()
                                  if key not in {"C10_nm", "C12_nm", "phi12_deg"}
                                  and (key.startswith("C") and key.endswith("_nm")
                                       or key.startswith("phi") and key.endswith("_deg"))},
                 "scan_detector_rotation_deg": trial.get("scan_detector_rotation_deg"),
                 "sample_tilt_mrad": [float(value) for value in trial.get("sample_tilt_mrad") or (0.0, 0.0)],
                 "depth_spread_nm": float(trial.get("depth_spread_nm") or 0.0),
                 # quantem.gpu tags the agreement trials "negative_thick_agreement" (the loss is its negative)
                 "objective": "phase_variance" if trial.get("objective", "phase_variance") == "phase_variance" else "agreement",
                 "loss": trial.get("loss")}
                for trial in self.direct.trial_history]

    def calibration_identity(self) -> dict | None:
        """Identify the loaded evidence and launch geometry without copying counts or naming a path.

        The source is quantem.gpu's portable ``source_identity`` (file names,
        sizes, SHA-256 of each file's first MiB, scan and detector shapes, how
        flagged pixels were filled), so a saved calibration is restored after
        its data folder is moved or copied and refused for another acquisition.
        Array inputs have no persistent identity: a filename attached to a
        modified array does not make it the acquisition on disk.
        """
        from quantem.gpu.direct_ptycho.persistence import json_value, source_identity

        direct = self.direct
        if direct.source_kind == "array" or direct.source_path is None:
            return None
        state = self.state()  # Resolve automatic probe calibration first.
        return json_value({
            "source": source_identity(
                direct.source_path, scan_shape=direct.input_scan_shape,
                hot_pixel_correction=(direct.source_provenance or {}).get("hot_pixel_correction"),
                storage_path=direct.source_storage_path,
                calibration_path=direct.calibration_path,
                manifest_path=direct.source_manifest_path,
            ),
            "data": {"kind": direct.source_kind, "dtype": direct.source_dtype,
                     "bytes": direct.source_bytes, "scan_shape": direct.scan_shape,
                     "scan_region": direct.scan_region,
                     "detector_bin": direct.source_detector_bin},
            "geometry": {
                **{name: state[name] for name in (
                    "voltage_kV", "semiangle_mrad", "scan_sampling_A",
                    "detector_sampling_mrad", "scan_detector_rotation_deg",
                    "scan_detector_rotation_flipped", "bright_field_radius_px",
                    "bright_field_intensity_threshold", "num_bright_field_pixels",
                )},
                "bright_field_center_px": direct.bright_field_center_px,
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

        Assigning ``direct.known_aberrations`` states them as known (quantem.gpu), so a
        later ``ptycho.run()`` applies them instead of fitting.
        """
        from quantem.gpu.direct_ptycho.units import validate_aberrations

        aberrations = validate_aberrations(saved["aberrations"], higher_order=True)
        self.set_scan_detector_rotation(saved["scan_detector_rotation_deg"])
        self.direct.known_aberrations = aberrations  # the saved model replaces every term, higher orders included
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
        self.direct.set_rotation(float(scan_detector_rotation_deg))

    def store(self, aberrations: dict, loss: float | None, *, sample: dict | None = None) -> None:
        """Keep reviewed optics and sample correction for the next reconstruction.

        The whole mapping is assigned, quantem.gpu's way to state known
        aberrations: an in-place update left an unfitted session to fit again
        on ``run()``, discarding the reviewed values.
        """
        self.direct.known_aberrations = {**self.direct.known_aberrations, **aberrations}
        self.direct.sample_tilt_mrad = None if sample is None else tuple(sample["sample_tilt_mrad"])
        self.direct.depth_spread_nm = None if sample is None else float(sample["depth_spread_nm"])
        if loss is not None:
            self.direct.best_loss = float(loss)

    def preview(self, aberrations: dict, **options):
        """Transient ``(phase, loss)`` at ``aberrations`` (``quantem.gpu.DirectPtycho.preview(aberrations, **options)``).

        ``aberrations`` holds ``C10_nm``, ``C12_nm``, ``phi12_deg`` and the
        higher-order panel's terms (``C21_nm`` with ``phi21_deg``; ``C30_nm`` and
        ``C50_nm`` have no azimuth). The higher orders reach quantem.gpu only
        while one of their magnitudes is non-zero: a named term, even at zero,
        selects quantem.gpu's 14-slot path, which holds C10, C12 and phi12 in
        float32, so a panel left at zero changed the last bits of most pixels of
        the two-term phase.
        """
        return self.direct.preview(_named_orders(aberrations), **options)

    @property
    def drag_windows(self) -> bool:
        """Whether ``drag`` / ``prepare`` / ``advance`` apply: quantem.gpu builds drag windows on CUDA only."""
        return self.direct.backend == "cuda"

    def drag(self, control: str, aberrations: dict, **options):
        """Transient float32 phase while one slider ``control`` (``C10_nm``, ``phi12_deg``, ...) is dragged.

        ``quantem.gpu.direct_ptycho.direct_workflow.drag``: ``preview``'s image to a few
        float32 ulps, served from a Chebyshev window along that control once
        ``advance`` has built it (1 ms phase, 3 ms complex, 7 ms direct ptychography at 3x on a
        512 x 512 scan with a 192 x 192 detector), exact frames until then. ``options`` are ``preview``'s
        except ``compute_loss``; committed frames use ``preview``.
        """
        from quantem.gpu.direct_ptycho import direct_workflow

        return direct_workflow.drag(self.direct, control, *self._drag_inputs(aberrations, **options)).get()

    def prepare(self, controls, aberrations: dict, **options) -> None:
        """Ask for drag windows along ``controls`` at these optics, built by ``advance`` while the sliders are idle."""
        from quantem.gpu.direct_ptycho import direct_workflow

        direct_workflow.prepare(self.direct, tuple(controls), *self._drag_inputs(aberrations, **options))

    def advance(self) -> bool:
        """Build one more node of a pending drag window; True when none is left."""
        from quantem.gpu.direct_ptycho import direct_workflow

        return direct_workflow.advance(self.direct)

    def _drag_inputs(self, aberrations: dict, *, upsample, method: str = "phase_of_mean", sample_tilt_mrad=(0.0, 0.0),
                     depth_spread_nm: float = 0.0, **strengths):
        """``preview``'s options as ``direct_workflow``'s (aberrations, tilt, depth, factor, policy)."""
        from quantem.gpu.direct_ptycho import direct_workflow
        from quantem.gpu.direct_ptycho.units import validate_aberrations

        policy = direct_workflow.settings(method, *(strengths.get(name) for name in (
            "phase_damp", "amplitude_damp", "smoothness_inv_A", "detector_weighting")))
        return (validate_aberrations(_named_orders(aberrations), higher_order=True), tuple(sample_tilt_mrad),
                float(depth_spread_nm), self.direct._resolve_upsample(upsample), policy)

    def browser_state(self) -> dict:
        """The compact state a browser direct ptychography engine rebuilds its reducers from.

        quantem.gpu's ``ExportState``: the bright-field pixel list
        (``bright_field_rows`` / ``_cols``), their reciprocal coordinates in the
        scan frame (``bright_field_k_row`` / ``_k_col``, quantem.gpu's ``kx_bf``
        / ``ky_bf``: its x is the scan row axis), the scan frequencies
        (``scan_q_row`` / ``scan_q_col``, its ``qx_1d`` / ``qy_1d``) and the
        aperture terms, with the detector step in mrad. ``ShowPtycho.export``
        writes ``snapshots/cal.json`` from this dict alone.
        """
        state = self.direct.browser_state()
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
        """Write the exact bright-field counts beside an exported page (``quantem.gpu.DirectPtycho.export_brightfield``)."""
        self.direct.export_brightfield(path)

    def run(self, **options) -> dict:
        """Run direct reconstruction at the reviewed optics (``quantem.gpu.DirectPtycho.run``); see ``result`` for the keys."""
        return self.result(self.direct.run(**options))

    def aberrations(self, **options) -> dict:
        """Fit the aberrations (``quantem.gpu.DirectPtycho.aberrations(**options)``); see ``result`` for the keys."""
        options.setdefault("search", "standard")
        options.setdefault("show", False)
        return self.result(self.direct.aberrations(**options))

    def fit_regions(self, regions: list[dict], parameters: dict, *,
                    upsample: int,
                    trials: int = 200,
                    progress: Callable[[dict], None] | None = None) -> dict:
        """Jointly fit regional models and crop their full-scan reconstructions.

        All numerical fitting and reconstruction use the ordinary direct ptychography backend.
        Independent regional sessions leave the overview calibration intact.
        Detector geometry and rotation are common; aberrations, both tilt
        components and depth spread are optimized together for each region,
        ``quantem.gpu.DirectPtycho.aberrations(tilt=True, trials=)``
        once with ``REGION_FIT_SEED`` and quantem.gpu's default refinement.
        The separate phase-variance diagnostic is not needed for this fit,
        so ``compute_loss=False`` omits it without changing any fitted value.
        A fit within 0.01 of the tilt limit or of quantem.gpu's depth-spread
        search range (``quantem.gpu.direct_ptycho.DEPTH_SPREAD_RANGE_NM``) is flagged ``bound_hit``.
        """
        direct = self.direct
        if direct.scan_region is not None or not direct.source_path:
            raise ValueError(
                "Local fitting needs the full acquisition. Open it with "
                "DirectPtycho.open(path, ...) or pass source_path to DirectPtycho(data, ...).")
        optics = parameters["aberrations"]
        if direct.scan_sampling_A[0] != direct.scan_sampling_A[1]:
            raise ValueError("Regional display currently needs equal row and column scan sampling.")
        started = time.perf_counter()
        fits, images = [], {}
        from quantem.gpu.direct_ptycho import DEPTH_SPREAD_RANGE_NM as depth_range_nm  # the joint fit's own depth search range
        factor = int(upsample)
        geometry = {
            "backend": direct.backend, "voltage_kV": direct.voltage_kV,
            "semiangle_mrad": direct.semiangle_mrad,
            "scan_sampling_A": direct.scan_sampling_A,
            "detector_sampling_mrad": direct.detector_sampling_mrad,
            "bright_field_center_px": direct.bright_field_center_px,
            "bright_field_radius_px": direct.bright_field_radius_px,
            "bright_field_intensity_threshold": direct.bright_field_intensity_threshold,
            "scan_detector_rotation_flipped": direct.scan_detector_rotation_flipped,
            "scan_detector_rotation_deg": parameters["scan_detector_rotation_deg"],
        }
        for region_index, region in enumerate(regions, start=1):
            region_started = time.perf_counter()
            bounds = tuple(region["bounds"])
            def report(stage: str, repeat: int) -> None:
                if progress is not None:
                    progress({"label": region["label"], "region_id": region["id"],
                              "region_index": region_index, "region_count": len(regions),
                              "stage": stage, "repeat": repeat, "repeats": 1,
                              "trials": int(trials),
                              "elapsed_s": time.perf_counter() - started})
            report("preparing", 0)
            with direct.select_region(
                bounds, aberrations=optics,
                scan_detector_rotation_deg=parameters["scan_detector_rotation_deg"],
            ) as regional:
                report("joint_fit", 1)
                fitted = regional.aberrations(
                    search="standard",
                    tilt=True, trials=int(trials),
                    tilt_limit_mrad=REGION_TILT_LIMIT_MRAD,
                    check_rotation=False, compare_standard=False,
                    compute_loss=False, seed=REGION_FIT_SEED, verbose=False, show=False)
                fit = {
                    "seed": REGION_FIT_SEED, "aberrations": dict(fitted.aberrations),
                    "sample_tilt_mrad": list(fitted.sample_tilt_mrad),
                    "depth_spread_nm": float(fitted.depth_spread_nm),
                    "sample_agreement": float(fitted.sample_agreement),
                    "num_completed_trials": len(fitted.trial_records or ()),
                    "refinement": fitted.refinement,
                    "optimizer": dict(getattr(fitted, "optimizer", {})),
                    "timings": dict(getattr(fitted, "timings", {})),
                }
            if not math.isfinite(fit["sample_agreement"]):
                raise ValueError(f"{region['label']}: nonfinite fit. Select another region and refit.")
            report("reconstructing", 1)
            phase, _ = self.preview(
                fit["aberrations"], sample_tilt_mrad=fit["sample_tilt_mrad"],
                depth_spread_nm=fit["depth_spread_nm"],
                upsample=factor, compute_loss=False)
            if tuple(phase.shape) != tuple(size * factor for size in direct.scan_shape):
                raise ValueError("Local correction requires a full-scan reconstruction before cropping.")
            images[f"{region['label']} · Direct"] = self.crop_native(phase, bounds, factor)
            coordinates = np.asarray(fit["optimizer"].get("coordinates", []))
            fit_bounds = np.asarray(fit["optimizer"].get("bounds", []))
            if coordinates.size and fit_bounds.shape == (coordinates.size, 2):
                bound_hit = np.any(np.minimum(coordinates - fit_bounds[:, 0],
                                               fit_bounds[:, 1] - coordinates) < 1e-3)
            else:
                bound_hit = (max(abs(v) for v in fit["sample_tilt_mrad"]) > REGION_TILT_LIMIT_MRAD - 0.01
                             or fit["depth_spread_nm"] < depth_range_nm[0] + 0.01
                             or fit["depth_spread_nm"] > depth_range_nm[1] - 0.01)
            fits.append({
                **region, **fit,
                "seed_sd_mrad": None,
                "fit_seconds": time.perf_counter() - region_started,
                "bound_hit": bool(bound_hit),
                "fit_record": {"method": "joint optics, tilt and depth", "fits": [fit],
                               "backend": direct.backend, "trials": int(trials)},
                "reconstruction": {
                    "scan_shape": list(direct.scan_shape), "scan_offset": list(self.scan_offset),
                    "scan_region": None,
                    "display_bounds": list(bounds), "upsample": factor,
                    "num_bright_field_pixels": self.num_bright_field_pixels,
                },
            })
        return {
            "regions": regions, "fits": fits, "images": images,
            "sampling_A": direct.scan_sampling_A[0] / factor,
            "result_kind": "Joint optics, tilt and depth · full-scan direct ptychography",
            "reference_parameters": parameters, "shared_geometry": geometry,
            "map": fits, "last_fit_seconds": time.perf_counter() - started,
        }

    @staticmethod
    def result(result) -> dict:
        """The values of a quantem.gpu ``DirectPtychoResult`` the widgets show and store.

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

    ``source`` is a ``Dataset4dstem`` from ``io.load`` or a 4D tensor on the
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
        """Add one acquisition that joined the comparison after it opened (``Show4DSTEM.append``, a stream)."""
        frame = AcquisitionView(source)
        if frame.shape != self.frames[0].shape or frame.device != self.device:
            raise ValueError(
                f"Acquisition shape {frame.shape} on {frame.device} does not match "
                f"{self.frames[0].shape} on {self.device}."
            )
        self.frames.append(frame)


def _named_orders(aberrations: dict) -> dict:
    """The terms that reach quantem.gpu: the higher orders only while one magnitude is non-zero (see ``DirectSession.preview``)."""
    higher_order = any(value != 0.0 for name, value in aberrations.items() if name.endswith("_nm") and name not in ("C10_nm", "C12_nm"))
    return aberrations if higher_order else {name: aberrations[name] for name in ("C10_nm", "C12_nm", "phi12_deg")}
