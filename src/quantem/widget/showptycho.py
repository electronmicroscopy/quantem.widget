"""
ShowPtycho - interactive anywidget for ptychography aberration exploration.

Renders phase and optional FFT while tuning aberrations.  The default uses the
full selected bright-field disk for the authoritative reconstruction; a smaller
BF subset can still be requested explicitly for exploratory drag previews.

Usage::

    from quantem.widget import ShowPtycho

    ssb.fit()
    del data
    ShowPtycho(ssb)   # opens the widget
"""

import datetime
import json
import math
import numbers
import pathlib
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import anywidget
import numpy as np
import traitlets

from quantem.gpu import SSB


# Default path for auto-saved starred aberration snapshots.  Users can pass
# ``save_dir`` to override; otherwise we drop the JSON next to wherever the
# notebook is executing so the "next cell" can read it back.
_DEFAULT_STARS_FILENAME = "showptycho_stars.json"
_CALIBRATION_SCHEMA_VERSION = 1
_DEFAULT_DRAG_BF_FRACTION = 1.0
_MIN_SSB_CROP_SPAN = 32


def _resolve_export_dir(
    widget: Any,
    path: "str | pathlib.Path | None",
    data: "str | pathlib.Path | None",
) -> pathlib.Path:
    """Pick the viewer output folder for :meth:`ShowPtycho.export`.

    ``path`` wins when given. Otherwise the location follows ``data`` so the
    viewer lands next to the source and the raw HDF5 is never duplicated:
    ``"in-place"`` uses the source file's own directory; an explicit path uses
    that directory; ``None`` defaults to a ``<name>_showptycho`` folder beside
    the source (or the current directory if the source path is unknown).
    """
    if data not in (None, "in-place"):
        raise ValueError(
            f"data={data!r} is not supported; use None (bundle) or 'in-place'."
        )
    if path is not None:
        return pathlib.Path(path).expanduser()
    source = widget._source_file
    if data == "in-place" and source:
        return pathlib.Path(source).expanduser().resolve().parent
    if source:
        base = pathlib.Path(source).expanduser().resolve()
        stem = (
            base.name[: -len("_master.h5")]
            if base.name.endswith("_master.h5")
            else base.stem
        )
        return base.parent / f"{stem}_showptycho"
    return pathlib.Path.cwd() / "showptycho_export"


@dataclass
class PtychoCalibration:
    """Locked SSB calibration parameters saved by ``ShowPtycho``.

    Parameters
    ----------
    rotation_angle_deg : float
        Scan-detector rotation angle in degrees.
    aberrations : dict[str, float]
        Aberration coefficients. ``C10`` and ``C12`` are in nm, ``phi12`` is in
        radians. Higher-order magnitudes are stored in nm.
    flip_phase : bool
        Whether the displayed phase sign was flipped.
    """

    rotation_angle_deg: float
    aberrations: dict[str, float]
    higher_order: dict[str, float] = field(default_factory=dict)
    flip_phase: bool = False
    voltage_kV: float | None = None
    semiangle_mrad: float | None = None
    scan_sampling_A: float | None = None
    det_sampling_mrad_px: float | None = None
    loss: float | None = None
    source_file: str | None = None
    scan_region: tuple[int, int, int, int] | None = None
    source_stem: str | None = None
    label: str | None = None
    notes: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now().isoformat(timespec="seconds")
    )


def _atomic_write_json(path: pathlib.Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(path)


def _finite_float_or_none(value: object) -> float | None:
    """Return a JSON-safe finite float, or ``None`` for missing/non-finite values."""

    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _calibration_from_mapping(data: dict[str, Any]) -> PtychoCalibration:
    return PtychoCalibration(
        rotation_angle_deg=float(data["rotation_angle_deg"]),
        aberrations={
            str(k): float(v) for k, v in (data.get("aberrations") or {}).items()
        },
        higher_order={
            str(k): float(v) for k, v in (data.get("higher_order") or {}).items()
        },
        flip_phase=bool(data.get("flip_phase", False)),
        voltage_kV=data.get("voltage_kV"),
        semiangle_mrad=data.get("semiangle_mrad"),
        scan_sampling_A=data.get("scan_sampling_A"),
        det_sampling_mrad_px=data.get("det_sampling_mrad_px"),
        loss=data.get("loss"),
        source_file=data.get("source_file"),
        scan_region=(
            tuple(int(v) for v in data["scan_region"])
            if data.get("scan_region") is not None
            else None
        ),
        source_stem=data.get("source_stem"),
        label=data.get("label"),
        notes=str(data.get("notes", "")),
        id=str(data.get("id", uuid.uuid4().hex[:12])),
        timestamp=str(
            data.get(
                "timestamp",
                datetime.datetime.now().isoformat(timespec="seconds"),
            )
        ),
    )


def load_ptycho_calibration(path: str | pathlib.Path) -> PtychoCalibration:
    """Load a single ``ShowPtycho`` calibration JSON file."""

    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    data = json.loads(path.read_text())
    if isinstance(data, list):
        raise ValueError(f"{path} is a calibration list, expected one object")
    return _calibration_from_mapping(data)


def save_ptycho_calibration(
    calibration: PtychoCalibration,
    path: str | pathlib.Path,
) -> pathlib.Path:
    """Save one ``ShowPtycho`` calibration JSON file."""

    path = pathlib.Path(path)
    payload = {
        "schema_version": _CALIBRATION_SCHEMA_VERSION,
        "version": "2.0",
        **asdict(calibration),
    }
    _atomic_write_json(path, payload)
    return path


def _coerce_calibration(calibration: object) -> PtychoCalibration:
    if isinstance(calibration, PtychoCalibration):
        return calibration
    if isinstance(calibration, (str, pathlib.Path)):
        return load_ptycho_calibration(calibration)
    raise TypeError(
        "calibration must be a PtychoCalibration or calibration JSON path; "
        f"got {type(calibration).__name__}."
    )


def _higher_order_widget_payload(calibration: PtychoCalibration) -> dict[str, float]:
    """Translate saved calibration keys into the widget panel convention."""

    payload: dict[str, float] = {}
    source = {**calibration.higher_order, **calibration.aberrations}
    radial = {"C30", "C50"}
    for key, value in source.items():
        if key in {"C10", "C12", "phi12"}:
            continue
        value = float(value)
        if key.endswith("_angle"):
            payload[key] = value
        elif key.startswith("phi") and len(key) >= 5:
            payload[f"C{key[3:]}_angle"] = math.degrees(value)
        elif key in radial:
            payload[key] = value
        elif key.startswith("C"):
            payload[f"{key}_mag"] = value
    return payload


def _resolve_drag_bf_count(
    drag_bf: int | float | None,
    total_bf: int,
) -> int:
    """Resolve a user BF preview request to a positive detector-pixel count."""

    total = max(1, int(total_bf))
    if drag_bf is None:
        return max(1, min(total, int(round(total * _DEFAULT_DRAG_BF_FRACTION))))
    if isinstance(drag_bf, numbers.Integral) and not isinstance(drag_bf, bool):
        return max(1, min(total, int(drag_bf)))
    value = float(drag_bf)
    if not math.isfinite(value) or value <= 0:
        value = _DEFAULT_DRAG_BF_FRACTION
    if 0 < value <= 1:
        return max(1, min(total, int(round(total * value))))
    return max(1, min(total, int(round(value))))


def _validate_ssb_scan_region(
    scan_region: object,
    full_scan_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Validate one in-bounds real-space scan crop for a fresh SSB fit.

    A crop is reconstruction input, not a display view.  The SSB kernels use
    native square scan sizes, so accepting an arbitrary rectangle here would
    cause the backend to pad it silently and blur the provenance of the fit.
    """

    if not isinstance(scan_region, (tuple, list)) or len(scan_region) != 4:
        raise TypeError(
            "scan_region must be (row_start, row_stop, col_start, col_stop)"
        )
    r0, r1, c0, c1 = (int(v) for v in scan_region)
    rows, cols = int(full_scan_shape[0]), int(full_scan_shape[1])
    if not (0 <= r0 < r1 <= rows and 0 <= c0 < c1 <= cols):
        raise ValueError(
            f"scan crop [{r0}:{r1}, {c0}:{c1}] is outside {rows}x{cols} data"
        )
    height, width = r1 - r0, c1 - c0
    if height < _MIN_SSB_CROP_SPAN or width < _MIN_SSB_CROP_SPAN:
        raise ValueError(
            "SSB refit crops must be at least "
            f"{_MIN_SSB_CROP_SPAN}x{_MIN_SSB_CROP_SPAN}; got {height}x{width}."
        )
    return r0, r1, c0, c1


class _ShowPtychoWidget(anywidget.AnyWidget):
    """ShowPtycho anywidget with GPU-accelerated reconstruction.

    During slider drag, uses the full selected BF disk by default. A smaller
    deterministic BF-pixel subset can be requested explicitly for responsive
    exploratory previews.

    The widget exclusively owns the ``accel`` backend - no other code
    should call methods on it while the widget is active.

    Parameters
    ----------
    accel : SSB
        Prepared backend-neutral SSB session.
    rotation_rad : float
        Rotation angle in radians.
    auto_aberrations : dict
        Auto-optimized aberration values (C10, C12, phi12 in radians).
    auto_loss_val : float
        Loss from automatic optimization.
    c10_range, c12_range, phi12_range : tuple[float, float]
        Slider ranges.
    drag_bf : int
        Number of BF pixels for preview.  Float values in ``(0, 1]`` are
        interpreted as a fraction of the detected BF disk; the default is full
        selected BF (``1.0``).
    save_dir : str or Path, optional
        Directory for saving results.
    ssb_ref : SSB, optional
        High-level SSB instance for applying aberrations.
    """

    _esm = pathlib.Path(__file__).with_name("static") / "showptycho.js"

    # -- Slider ranges (Python → JS, set once) --
    c10_min = traitlets.Float(-400.0).tag(sync=True)
    c10_max = traitlets.Float(400.0).tag(sync=True)
    c12_min = traitlets.Float(-100.0).tag(sync=True)
    c12_max = traitlets.Float(100.0).tag(sync=True)
    phi12_min = traitlets.Float(-90.0).tag(sync=True)
    phi12_max = traitlets.Float(90.0).tag(sync=True)
    rotation_min = traitlets.Float(-180.0).tag(sync=True)
    rotation_max = traitlets.Float(180.0).tag(sync=True)

    # -- Current rotation (JS ↔ Python).  Starts at the value passed to
    # __init__ and can be swept to find the best orientation of the BF mask. --
    rotation_deg = traitlets.Float(0.0).tag(sync=True)

    # -- Flip phase sign (JS → Python).  Phase is inherently ambiguous by sign
    # in SSB; this toggle lets the user pick the convention that matches
    # expected sample contrast without re-optimizing. --
    flip_phase = traitlets.Bool(False).tag(sync=True)

    # -- Auto reference (Python → JS, set once) --
    auto_c10 = traitlets.Float(0.0).tag(sync=True)
    auto_c12 = traitlets.Float(0.0).tag(sync=True)
    auto_phi12_deg = traitlets.Float(0.0).tag(sync=True)
    auto_loss = traitlets.Float(0.0).tag(sync=True)
    # Initial rotation angle at mount time - used by the Reset button so it
    # returns the rotation slider to where the user entered the widget.
    auto_rotation_deg = traitlets.Float(0.0).tag(sync=True)

    # -- Request from JS → Python --
    request_json = traitlets.Unicode("").tag(sync=True)

    # -- Response from Python → JS --
    phase_bytes = traitlets.Bytes(b"").tag(sync=True)
    phase_width = traitlets.Int(0).tag(sync=True)
    phase_height = traitlets.Int(0).tag(sync=True)
    result_json = traitlets.Unicode("").tag(sync=True)

    # -- Pixel size (Å) for scale bar --
    pixel_size = traitlets.Float(0.0).tag(sync=True)

    # -- Initial panel width and FFT visibility (Python → JS, set once at mount).
    # JS reads these as one-shot seeds for its local panel/showFFT state - the
    # user can still resize via the corner handle and toggle FFT via the switch. --
    initial_panel_size = traitlets.Int(800).tag(sync=True)
    initial_fft_on = traitlets.Bool(False).tag(sync=True)

    # -- Save/Apply trigger (JS → Python) --
    save_trigger = traitlets.Int(0).tag(sync=True)

    # -- User-editable notes persisted into calibration.json (JS ↔ Python) --
    notes = traitlets.Unicode("").tag(sync=True)

    # -- Pin event (JS → Python) --
    pin_json = traitlets.Unicode("").tag(sync=True)

    # -- Where starred snapshots are auto-saved (Python → JS, displayed in UI) --
    stars_path = traitlets.Unicode("").tag(sync=True)

    # -- Where calibration.json was last written (Python → JS, empty until save) --
    calibration_path = traitlets.Unicode("").tag(sync=True)
    calibration_saved_at = traitlets.Unicode("").tag(sync=True)

    # -- Optuna trial history (Python → JS, set once at init).
    # JSON list of {rank, C10, C12, phi12_deg, loss} sorted by ascending loss.
    # Empty string when the SSB instance hasn't been optimized yet. --
    trials_json = traitlets.Unicode("").tag(sync=True)

    # -- Preview BF count (JS → Python).
    # Positive count used while exploring aberrations. The full BF count is the
    # right edge of the slider rather than a separate "off" state. --
    drag_bf = traitlets.Int(0).tag(sync=True)
    total_bf = traitlets.Int(0).tag(sync=True)

    # -- Higher-order aberration panel (JS → Python).
    # JSON-encoded dict with keys a subset of
    #   {C21_mag, C21_angle, C23_mag, C23_angle, C30, C32_mag, C32_angle,
    #    C34_mag, C34_angle, C41_mag, C41_angle, C43_mag, C43_angle,
    #    C45_mag, C45_angle, C50, C52_mag, C52_angle, C54_mag, C54_angle,
    #    C56_mag, C56_angle}
    # All values default to 0 when absent.  Magnitudes in nm (displayed unit);
    # angles in degrees (displayed unit).  When any magnitude is non-zero the
    # reconstruct path switches from the fast 2-term kernel to the 14-coef
    # chi_full kernel via SSBEngine.reconstruct_full. --
    higher_order_json = traitlets.Unicode("{}").tag(sync=True)

    # -- Browser-side WebGPU SSB source (Python → JS).
    # Folder exports point the browser at exact BF columns or compressed HDF5
    # and let WebGPU build transient BF reducers on the device.
    webgpu_preview_enabled = traitlets.Bool(False).tag(sync=True)
    webgpu_cal_json = traitlets.Unicode("").tag(sync=True)
    webgpu_h5_source_json = traitlets.Unicode("").tag(sync=True)
    webgpu_preview_status = traitlets.Unicode("WebGPU preview not initialized.").tag(
        sync=True
    )
    webgpu_standalone = traitlets.Bool(False).tag(sync=True)

    # -- Crop-and-refit SSB (JS ↔ Python).  This deliberately rebuilds SSB
    # from raw 4D-STEM input; it is not a display crop of the current phase. --
    scan_rows = traitlets.Int(0).tag(sync=True)
    scan_cols = traitlets.Int(0).tag(sync=True)
    scan_region_json = traitlets.Unicode("").tag(sync=True)
    crop_refit_available = traitlets.Bool(False).tag(sync=True)
    crop_refit_status = traitlets.Unicode("").tag(sync=True)
    crop_refit_request_json = traitlets.Unicode("").tag(sync=True)

    def __init__(
        self,
        accel: SSB,
        rotation_rad: float,
        auto_aberrations: dict,
        auto_loss_val: float,
        c10_range: tuple[float, float],
        c12_range: tuple[float, float],
        phi12_range: tuple[float, float],
        rotation_range: "tuple[float, float] | None" = None,
        drag_bf: int | float | None = _DEFAULT_DRAG_BF_FRACTION,
        save_dir: "str | pathlib.Path | None" = None,
        ssb_ref: SSB | None = None,
        pixel_size: float = 0.0,
        source_file: "str | None" = None,
        size: int = 800,
        fft_on: bool = False,
        initial_compute_loss: bool = True,
        initial_loss_val: float | None = None,
        initial_flip_phase: bool = False,
        initial_higher_order: dict[str, float] | None = None,
        scan_region: tuple[int, int, int, int] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._accel = accel
        self._rotation_rad = rotation_rad
        self._ssb_ref = ssb_ref
        self._save_dir = pathlib.Path(save_dir) if save_dir else None
        # Identifies which 4D-STEM file this SSB was built from.  Persisted into
        # calibration.json and every starred entry so time-series `Live.watch(...,
        # calibrations=...)` can match a star to the right file by stem.
        self._source_file = str(source_file) if source_file else None
        self._scan_shape = accel.scan_shape
        self._source_scan_shape = self._scan_shape
        self._scan_region = (
            _validate_ssb_scan_region(scan_region, self._scan_shape)
            if scan_region is not None
            else (0, self._scan_shape[0], 0, self._scan_shape[1])
        )
        self._pinned: list[dict] = []
        self._last_phase_np: np.ndarray | None = None
        self._inflight_id: int = -1
        self._drag_state: Any | None = None
        # Cached decode of result_json so `_current_c10/c12/phi12_deg()` skip
        # JSON parse on every higher-order observer tick (3 parses/frame
        # → 0 parses/frame).  Refreshed at the end of _do_reconstruct.
        self._last_result: dict = {}
        self._webgpu_asset_dir: pathlib.Path | None = None
        # Backend-neutral host coefficient buffers. Device conversion, when
        # needed, belongs to the compute backend rather than the widget.
        self._ho_mags_buf = np.zeros(14, dtype=np.float32)
        self._ho_angs_buf = np.zeros(14, dtype=np.float32)
        # Starred snapshots are the time-series payload: every star/unstar is
        # atomically written to ``self._stars_path`` so downstream cells can
        # re-open the file and ingest the full defocus/C12/phi/rotation history.
        stars_dir = self._save_dir if self._save_dir is not None else pathlib.Path.cwd()
        self._stars_path = stars_dir / _DEFAULT_STARS_FILENAME
        self.stars_path = str(self._stars_path.resolve())
        # Canonical location for the project calibration. Downstream workflows
        # read this stable filename with ``load_calibration()``.
        self._calibration_path = stars_dir / "calibration.json"

        self.scan_rows, self.scan_cols = self._scan_shape
        self.scan_region_json = json.dumps(self._scan_region)
        self.crop_refit_available = bool(
            self._source_file
            and pathlib.Path(self._source_file).is_file()
        )
        self.crop_refit_status = (
            "Ready to refit a native square scan crop."
            if self.crop_refit_available
            else "Crop refit needs a re-loadable master file."
        )

        # Cache rotation
        accel.set_rotation(math.degrees(rotation_rad))

        # Set ranges
        self.c10_min, self.c10_max = c10_range
        self.c12_min, self.c12_max = c12_range
        self.phi12_min, self.phi12_max = phi12_range
        # Rotation range: default ±180° so the slider covers negative rotations
        # directly without the user mentally adding 180.  Technically scan/detector
        # rotation is mod-180, but the extra width costs nothing and matches
        # microscope conventions that sometimes report negative values.
        start_deg = math.degrees(rotation_rad)
        if rotation_range is None:
            rotation_range = (-180.0, 180.0)
        self.rotation_min, self.rotation_max = rotation_range
        # Set current rotation without firing the observer (guard against premature reconstruct
        # before _inflight_id and accel state are initialized).
        self._rotation_deg_init = True
        self.rotation_deg = start_deg
        self._rotation_deg_init = False

        # Set auto reference
        self.auto_c10 = auto_aberrations.get("C10", 0.0)
        self.auto_c12 = auto_aberrations.get("C12", 0.0)
        self.auto_phi12_deg = math.degrees(auto_aberrations.get("phi12", 0.0))
        self.auto_loss = _finite_float_or_none(auto_loss_val) or 0.0
        self.auto_rotation_deg = start_deg

        # Set pixel size for scale bar
        self.pixel_size = pixel_size

        # Seed the UI's initial panel size + FFT visibility.  JS reads these
        # once at mount; user interaction with the resize handle / FFT switch
        # takes over from there.
        self.initial_panel_size = int(size)
        self.initial_fft_on = bool(fft_on)

        # Seed saved-calibration state before observers are connected and
        # before the first phase image is reconstructed.  Otherwise loading a
        # higher-order calibration first draws the 3-parameter image, then
        # immediately triggers a second full reconstruction when the
        # higher-order JSON trait is assigned.
        self.flip_phase = bool(initial_flip_phase)
        if initial_higher_order:
            self.higher_order_json = json.dumps(initial_higher_order)

        # Listen for events
        self.observe(self._on_request, names=["request_json"])
        self.observe(self._on_save, names=["save_trigger"])
        self.observe(self._on_pin, names=["pin_json"])
        self.observe(self._on_drag_bf_change, names=["drag_bf"])
        self.observe(self._on_rotation_change, names=["rotation_deg"])
        self.observe(self._on_flip_change, names=["flip_phase"])
        self.observe(self._on_higher_order_change, names=["higher_order_json"])
        self.observe(self._on_crop_refit_request, names=["crop_refit_request_json"])

        # Publish total BF count so the UI can clamp user input to valid range
        self.total_bf = accel.num_bf

        # Publish Optuna trial history (sorted by ascending loss) so the widget
        # can render a browsable trials strip.  Capped at 50 for payload size.
        self.trials_json = self._build_trials_payload()

        # Initial reconstruction. Saved calibrations already contain the loss, so
        # they only need the phase image for first display.
        self._inflight_id = 0
        self._do_reconstruct(
            0,
            self.auto_c10,
            self.auto_c12,
            self.auto_phi12_deg,
            compute_loss=bool(initial_compute_loss),
            loss_override=initial_loss_val,
        )

        # Start with an interactive BF fraction. Full BF is still available by
        # moving the slider to the total count.
        self.drag_bf = _resolve_drag_bf_count(drag_bf, self.total_bf)

    # ------------------------------------------------------------------
    #  BF-subset drag preview state
    # ------------------------------------------------------------------

    def _build_drag_state(self, drag_bf: int) -> None:
        """Ask the compute backend to prepare a reusable BF subset."""

        self._drag_state = self._accel.preview_context(drag_bf)

    def _enter_drag(self) -> None:
        """Enter the backend-owned BF-subset context."""

        self._drag_state.__enter__()

    def _exit_drag(self) -> None:
        """Restore full-BF state through the backend-owned context."""

        self._drag_state.__exit__(None, None, None)

    def free_drag_state(self) -> None:
        """Explicitly release drag preview VRAM."""
        if self._drag_state is not None:
            self._drag_state.close()
        self._drag_state = None

    def _on_drag_bf_change(self, change):
        """Rebuild or drop the BF-subset drag state when the user changes count."""
        new_count = int(change["new"])
        if new_count <= 0:
            self.free_drag_state()
            return
        # Clamp to valid range and rebuild. Existing state is dropped first to release VRAM.
        num_bf = self._accel.num_bf
        count = max(1, min(new_count, num_bf))
        self.free_drag_state()
        self._build_drag_state(count)

    def _on_rotation_change(self, change):
        """Re-cache the rotation on the engine and re-reconstruct with current aberrations."""
        if self._rotation_deg_init:
            return  # initial trait assignment during __init__; skip
        new_deg = float(change["new"])
        new_rad = math.radians(new_deg)
        self._rotation_rad = new_rad
        # cache_rotation invalidates the drag BF subset (indices into kx/ky_bf change),
        # so drop drag state and rebuild on next drag if caller enabled it.
        drag_count = int(self.drag_bf)
        if drag_count > 0:
            self.free_drag_state()
        self._accel.set_rotation(new_deg)
        if drag_count > 0:
            self._build_drag_state(drag_count)
        # Re-reconstruct with the last-committed aberrations.
        self._inflight_id += 1
        self._do_reconstruct(
            self._inflight_id,
            self.auto_c10 if self._last_phase_np is None else self._current_c10(),
            self.auto_c12 if self._last_phase_np is None else self._current_c12(),
            self.auto_phi12_deg if self._last_phase_np is None else self._current_phi12_deg(),
        )

    def _on_higher_order_change(self, change):
        """Re-reconstruct when the higher-order panel values change.

        Triggered by JS writing the ``higher_order_json`` trait.  We reuse
        the currently-displayed C10/C12/phi12 (from the last request) and
        just re-run the pipeline - the 14-coef kernel picks up whatever is
        now in the JSON.
        """
        if self._last_phase_np is None:
            return
        self._inflight_id += 1
        self._do_reconstruct(
            self._inflight_id,
            self._current_c10(),
            self._current_c12(),
            self._current_phi12_deg(),
            compute_loss=True,
        )

    def _on_flip_change(self, change):
        """Re-send the current phase with the new sign; no GPU recompute needed.

        Toggling flip is an involution: every toggle event means "negate the
        currently-displayed phase".  The _do_reconstruct path always starts
        from raw GPU output and re-applies flip_phase, so the cache state
        stays consistent across reconstruct + toggle events.
        """
        if self._last_phase_np is None:
            return
        self._last_phase_np = -self._last_phase_np
        self.phase_bytes = self._last_phase_np.astype(np.float32, copy=False).tobytes()

    def _current_c10(self) -> float:
        return float(self._last_result.get("C10", self.auto_c10))

    def _current_c12(self) -> float:
        return float(self._last_result.get("C12", self.auto_c12))

    def _current_phi12_deg(self) -> float:
        return float(self._last_result.get("phi12_deg", self.auto_phi12_deg))

    # ------------------------------------------------------------------
    #  Event handlers
    # ------------------------------------------------------------------

    @property
    def pinned(self) -> list[dict]:
        """All pinned snapshots (params + loss)."""
        return self._pinned

    @property
    def starred(self) -> list[dict]:
        """Starred pinned snapshots - the time-series payload.

        Each entry includes the aberration values, rotation, flip sign,
        loss, and an ISO-8601 timestamp.  Same content as
        ``self._stars_path`` on disk, minus the raw phase array.
        """
        return [
            {k: v for k, v in p.items() if k not in ("phase",)}
            for p in self._pinned
            if p.get("starred")
        ]

    def _build_trials_payload(self, max_trials: int = 50) -> str:
        """Serialize the session trial history for the widget's trials panel.

        The SSB optimizer stores trials as
        ``list[{"loss": float, "params": {"C10_nm", "C12_nm", "phi12_deg"}}]``.
        We sort by ascending loss (best first), rename to the widget's
        ``C10 / C12 / phi12_deg`` convention, and cap at ``max_trials`` so a
        200-trial Optuna run doesn't bloat the initial trait payload.
        """
        ssb = self._ssb_ref
        raw = ssb.trial_history if ssb is not None else None
        if not raw:
            return ""
        finite_trials = []
        for trial in raw:
            loss = _finite_float_or_none(trial.get("loss"))
            if loss is None:
                continue
            finite_trials.append((loss, trial))
        sorted_trials = [
            trial for _loss, trial in sorted(finite_trials, key=lambda item: item[0])
        ][:max_trials]
        payload = []
        for rank, trial in enumerate(sorted_trials):
            p = trial.get("params", {})
            loss = _finite_float_or_none(trial.get("loss"))
            if loss is None:
                continue
            payload.append({
                "rank": rank,
                "C10": float(p.get("C10_nm", 0.0)),
                "C12": float(p.get("C12_nm", 0.0)),
                "phi12_deg": float(p.get("phi12_deg", 0.0)),
                "loss": loss,
            })
        return json.dumps(payload)

    def _write_stars_file(self) -> None:
        """Atomically dump starred entries to ``self._stars_path``.

        Each entry is written in the canonical Calibration shape (nested
        ``aberrations`` dict, phi12 in radians, ``rotation_angle_deg``,
        microscope metadata pulled off ``self._ssb_ref``) plus the three
        widget extras ``id`` / ``timestamp`` / ``starred``.  This means
        ``load_calibrations(stars_path)`` returns a ``list[Calibration]``
        directly, providing a stable bridge for downstream time-series work.
        """
        self._stars_path.parent.mkdir(parents=True, exist_ok=True)

        ssb = self._ssb_ref
        voltage_kV = ssb.voltage_kV if ssb is not None else None
        semiangle_mrad = ssb.semiangle_mrad if ssb is not None else None
        scan_sampling = ssb.scan_sampling_A if ssb is not None else None
        if isinstance(scan_sampling, (tuple, list)):
            scan_sampling_A = float(scan_sampling[0])
        elif scan_sampling is not None:
            scan_sampling_A = float(scan_sampling)
        else:
            scan_sampling_A = None

        # Layout for unpacking the panel JSON into canonical names/angles.
        ho_layout = [
            ("C21",  True),  ("C23", True),
            ("C30",  False), ("C32", True), ("C34", True),
            ("C41",  True),  ("C43", True), ("C45", True),
            ("C50", False),  ("C52", True), ("C54", True), ("C56", True),
        ]

        payload = []
        for p in self._pinned:
            if not p.get("starred"):
                continue
            aberr = {
                "C10": float(p.get("C10", 0.0)),
                "C12": float(p.get("C12", 0.0)),
                "phi12": math.radians(float(p.get("phi12_deg", 0.0))),
            }
            # Merge starred higher-order coefs into the aberrations dict using
            # the canonical {name: value} / {"phi<n><m>": rad} shape so that
            # ``load_calibrations(stars_path)`` and ``SSB.reconstruct_full`` can
            # consume the file without translation.  Missing/zero values are
            # omitted to keep the JSON small.
            ho = p.get("higher_order", {}) or {}
            for name, has_angle in ho_layout:
                if has_angle:
                    mag = float(ho.get(f"{name}_mag", 0.0))
                    ang_deg = float(ho.get(f"{name}_angle", 0.0))
                else:
                    mag = float(ho.get(name, 0.0))
                    ang_deg = 0.0
                if mag == 0.0:
                    continue
                aberr[name] = mag
                if has_angle:
                    # Canonical name: phi21, phi23, phi32, ... (drop the 'C').
                    aberr[f"phi{name[1:]}"] = math.radians(ang_deg)
            payload.append({
                "id": p.get("id"),
                "timestamp": p.get("timestamp"),
                "starred": True,
                "rotation_angle_deg": float(p.get("rotation_deg", math.degrees(self._rotation_rad))),
                "aberrations": aberr,
                "flip_phase": bool(p.get("flip_phase", False)),
                "voltage_kV": voltage_kV,
                "semiangle_mrad": semiangle_mrad,
                "scan_sampling_A": scan_sampling_A,
                "loss": float(p["loss"]) if p.get("loss") is not None else None,
                "source_file": p.get("source_file"),
                "notes": None,
            })

        tmp = self._stars_path.with_suffix(self._stars_path.suffix + ".tmp")
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        tmp.replace(self._stars_path)

    def _on_request(self, change):
        """Handle reconstruction request from JS."""
        raw = change["new"]
        if not raw:
            return
        req = json.loads(raw)
        self._inflight_id = req["id"]
        self._do_reconstruct(
            req["id"], req["c10"], req["c12"], req["phi12_deg"],
            compute_loss=req.get("committed", False),
        )

    def _on_crop_refit_request(self, change):
        """Rebuild SSB from a selected scan crop and refit aberrations.

        The current SSB object retains only its BF-indexed Fourier cache, not
        raw diffraction frames.  A crop therefore has to reload the selected
        scan region from the original master before a scientifically valid
        optimization can start.
        """

        raw = change["new"]
        if not raw:
            return
        try:
            request = json.loads(raw)
            local_region = _validate_ssb_scan_region(
                request["scan_region"], self._scan_shape
            )
            n_trials = int(request.get("n_trials", 200))
            if n_trials < 1:
                raise ValueError("SSB refit needs at least one optimization trial.")
            if not self.crop_refit_available or not self._source_file:
                raise RuntimeError(
                    "Crop refit is unavailable. Open ShowPtycho with "
                    "source_file=<master.h5>."
                )

            origin_r, _origin_r1, origin_c, _origin_c1 = self._scan_region
            global_region = (
                origin_r + local_region[0],
                origin_r + local_region[1],
                origin_c + local_region[2],
                origin_c + local_region[3],
            )
            self.crop_refit_status = (
                f"Loading [{global_region[0]}:{global_region[1]}, "
                f"{global_region[2]}:{global_region[3]}] and fitting SSB "
                f"({n_trials} trials)..."
            )
            self._refit_ssb_crop(global_region, n_trials=n_trials)
        except Exception as exc:
            self.crop_refit_status = f"Crop refit failed: {exc}"

    def _refit_ssb_crop(
        self,
        scan_region: tuple[int, int, int, int],
        *,
        n_trials: int,
    ) -> None:
        """Load one raw scan crop, optimize SSB, and replace widget state."""

        if self._ssb_ref is None or not self._source_file:
            raise RuntimeError("Crop refit requires a source-backed SSB session.")
        scan_region = _validate_ssb_scan_region(scan_region, self._source_scan_shape)

        from quantem.gpu.io import load

        previous = self._ssb_ref
        loaded = load(
            self._source_file,
            scan_region=scan_region,
            backend=self._accel.backend,
            scan_shape=self._source_scan_shape,
            verbose=False,
        )
        rebuilt = SSB.from_array(
            loaded.data,
            backend=previous.backend,
            voltage_kV=previous.voltage_kV,
            semiangle_mrad=previous.semiangle_mrad,
            scan_sampling_A=previous.scan_sampling_A,
            det_sampling=previous.det_sampling,
            aberrations={
                "C10": float(
                    self._last_result.get(
                        "C10",
                        previous.aberrations.get("C10", 0.0),
                    )
                ),
                "C12": float(
                    self._last_result.get(
                        "C12",
                        previous.aberrations.get("C12", 0.0),
                    )
                ),
                "phi12": math.radians(
                    float(
                        self._last_result.get(
                            "phi12_deg",
                            math.degrees(previous.aberrations.get("phi12", 0.0)),
                        )
                    )
                ),
            },
            rotation_angle_deg=math.degrees(self._rotation_rad),
            bf_intensity_threshold=previous.bf_intensity_threshold,
            bf_radius=previous.bf_radius,
            source_path=previous.source_path,
        )
        rebuilt.fit(
            trials=n_trials,
            refinement="nelder-mead",
            verbose=False,
        )

        self._ssb_ref = rebuilt
        self._accel = rebuilt
        self._scan_shape = rebuilt.scan_shape
        self._scan_region = scan_region
        self.scan_rows, self.scan_cols = self._scan_shape
        self.scan_region_json = json.dumps(scan_region)
        self.total_bf = self._accel.num_bf
        self.drag_bf = self.total_bf
        self._rotation_rad = math.radians(rebuilt.rotation_angle_deg)
        self.rotation_deg = math.degrees(self._rotation_rad)
        self.auto_rotation_deg = self.rotation_deg
        self.auto_c10 = float(rebuilt.aberrations["C10"])
        self.auto_c12 = float(rebuilt.aberrations["C12"])
        self.auto_phi12_deg = math.degrees(float(rebuilt.aberrations["phi12"]))
        self.auto_loss = _finite_float_or_none(rebuilt.best_loss) or 0.0
        self.higher_order_json = "{}"
        self.trials_json = self._build_trials_payload()
        self._inflight_id += 1
        self._do_reconstruct(
            self._inflight_id,
            self.auto_c10,
            self.auto_c12,
            self.auto_phi12_deg,
            compute_loss=True,
        )
        self.crop_refit_status = (
            f"Refit complete: [{scan_region[0]}:{scan_region[1]}, "
            f"{scan_region[2]}:{scan_region[3]}], {n_trials} trials."
        )

    def _higher_order_arrays(
        self, c10: float, c12: float, phi12_deg: float,
    ) -> "tuple[object, object, bool]":
        """Read ``higher_order_json`` and pack all 14 Krivanek coefs into
        backend coefficient arrays ready for ``reconstruct_full``.

        The first two slots carry ``(C10, C12, phi12)`` from the main
        three-slider UI. Returns
        ``(mags_m, angles_rad, any_higher_order_active)``.

        mags are in the engine's magnitude convention (nm-valued numeric -
        the engine's 2-term kernel treats the nm float as the coefficient,
        so the 14-coef kernel must do the same to match).
        """
        # Reuse scratch buffers allocated in __init__.  fill(0) is a single
        # memset, far cheaper than allocating two fresh (14,) arrays per frame.
        mags = self._ho_mags_buf
        angs = self._ho_angs_buf
        mags.fill(0)
        angs.fill(0)
        # C10, C12, phi12 come from the main sliders (nm, nm, rad).
        mags[0] = np.float32(c10)
        mags[1] = np.float32(c12)
        angs[1] = np.float32(math.radians(phi12_deg))

        # Higher-order values come from the panel JSON.
        ho = json.loads(self.higher_order_json or "{}")
        any_active = False
        # (name, index, has_angle) for the 11 higher-order coefficients
        layout = [
            ("C21",  2, True), ("C23", 3, True),
            ("C30",  4, False), ("C32", 5, True), ("C34", 6, True),
            ("C41",  7, True),  ("C43", 8, True), ("C45", 9, True),
            ("C50", 10, False), ("C52", 11, True),
            ("C54", 12, True),  ("C56", 13, True),
        ]
        for name, idx, has_angle in layout:
            if has_angle:
                mag = float(ho.get(f"{name}_mag", 0.0))
                ang_deg = float(ho.get(f"{name}_angle", 0.0))
            else:
                mag = float(ho.get(name, 0.0))
                ang_deg = 0.0
            if mag != 0.0:
                any_active = True
            mags[idx] = np.float32(mag)
            angs[idx] = np.float32(math.radians(ang_deg))
        return mags, angs, any_active

    def _do_reconstruct(
        self,
        rid: int,
        c10: float,
        c12: float,
        phi12_deg: float,
        compute_loss: bool = True,
        loss_override: float | None = None,
    ):
        """Run GPU reconstruction and push result to JS.

        Default (no higher-order active) uses the fast 2-term
        ``reconstruct_with_loss`` path (78 ms on 512×512).  When any
        higher-order slider is non-zero, routes through
        ``SSBEngine.reconstruct_full_with_loss`` with the full 14-coef
        Krivanek polynomial.  The variance loss is the same BF-pixel
        phase-variance metric the 3-param optimizer uses, so it is
        directly comparable across both paths and the user can watch it
        drop as they tune higher-order sliders.
        """
        phi12_rad = math.radians(phi12_deg)
        t0 = time.perf_counter()
        use_drag = not compute_loss and self._drag_state is not None
        mags_m, angles_rad, any_ho = self._higher_order_arrays(c10, c12, phi12_deg)

        # Higher-order path uses the full 14-coef kernel; the 3-param fast path
        # is the common case. During drag (compute_loss=False) we skip the
        # variance pass so slider frames stay under the drag budget.
        if use_drag:
            self._enter_drag()
        try:
            phase_np, loss = self._accel.preview(
                {"C10": c10, "C12": c12, "phi12": phi12_rad},
                compute_loss=compute_loss,
                higher_order_magnitudes=mags_m if any_ho else None,
                higher_order_angles=angles_rad if any_ho else None,
            )
            if not compute_loss and loss_override is not None:
                loss = float(loss_override)
            t_gpu = time.perf_counter()
            t_d2h = time.perf_counter()
        finally:
            if use_drag:
                self._exit_drag()

        if rid != self._inflight_id:
            return

        # Apply flip-phase sign convention BEFORE caching/sending.  SSB's phase
        # has an inherent ± ambiguity; we let the user pick the convention that
        # matches expected sample contrast.  Cached value is the displayed one.
        if bool(self.flip_phase):
            phase_np = -phase_np
        self._last_phase_np = phase_np

        # tobytes copies the ndarray into a Python bytes object. For 512×512 float32
        # = 1 MB; expected ~1-2 ms on a modern CPU.
        payload = phase_np.tobytes()
        t_bytes = time.perf_counter()

        h, w = phase_np.shape
        self.phase_height = h
        self.phase_width = w
        # This assignment triggers traitlets sync → Comm message to the frontend.
        # The work measured here is Python-side only (serialization + queue enqueue);
        # the actual wire time is Comm and lives in (UI − GPU − JS).
        self.phase_bytes = payload
        t_trait = time.perf_counter()

        entry = {
            "id": rid,
            # Keep scientific parameters at their optimizer precision. The
            # frontend formats labels for readability; rounding the synced
            # state here would silently perturb saved/exported calibration.
            "C10": float(c10),
            "C12": float(c12),
            "phi12_deg": float(phi12_deg),
            "loss": loss,
            "time_ms": round((t_gpu - t0) * 1000, 1),        # GPU kernel only
            "d2h_ms":  round((t_d2h - t_gpu) * 1000, 1),     # cp.asnumpy + dtype
            "bytes_ms": round((t_bytes - t_d2h) * 1000, 1),   # ndarray.tobytes
            "trait_ms": round((t_trait - t_bytes) * 1000, 1), # Comm enqueue + sync broadcast
            "py_total_ms": round((t_trait - t0) * 1000, 1),   # everything on Python side
        }
        # Cache the dict so observers can read current slider values without
        # re-parsing result_json on every tick.
        self._last_result = entry
        self.result_json = json.dumps(entry)

    def _on_pin(self, change):
        """Handle pin/unpin event from JS."""
        raw = change["new"]
        if not raw:
            return
        evt = json.loads(raw)

        action = evt.get("action")
        if action == "pin":
            # source_file is captured AT PIN TIME so stars written later still
            # know which file this snapshot came from, even if the widget is
            # reused against a different dataset in the same kernel.
            # Capture the higher-order panel state at pin time.  A session
            # can pin multiple snapshots with different higher-order tunings,
            # and each one must remember the exact 14-coef configuration it
            # was taken with - otherwise re-viewing an old star would silently
            # fall back to whatever higher_order_json currently holds.
            higher_order_snapshot = json.loads(self.higher_order_json or "{}")
            pin_entry = {
                "id": evt.get("id"),
                "C10": evt.get("C10"),
                "C12": evt.get("C12"),
                "phi12_deg": evt.get("phi12_deg"),
                "rotation_deg": evt.get("rotation_deg", math.degrees(self._rotation_rad)),
                "flip_phase": bool(evt.get("flip_phase", self.flip_phase)),
                "loss": evt.get("loss"),
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "starred": False,
                "source_file": self._source_file,
                "higher_order": higher_order_snapshot,
            }
            if self._last_phase_np is not None:
                pin_entry["phase"] = self._last_phase_np.copy()
            self._pinned.append(pin_entry)
        elif action == "star" or action == "unstar":
            pin_id = evt.get("id")
            want_starred = (action == "star")
            for pin_entry in self._pinned:
                if pin_entry.get("id") != pin_id:
                    continue
                pin_entry["starred"] = want_starred
                # Refresh the timestamp on star so stars reflect when the user
                # actually decided to keep the snapshot (not when it was pinned).
                if want_starred:
                    pin_entry["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
                break
            self._write_stars_file()
        elif action == "view":
            pin_id = evt.get("id")
            for pin_entry in self._pinned:
                if pin_entry.get("id") != pin_id:
                    continue
                phase = pin_entry.get("phase")
                if phase is not None:
                    self._last_phase_np = np.array(phase, copy=True)
                    h, w = self._last_phase_np.shape
                    self.phase_height = h
                    self.phase_width = w
                    self.phase_bytes = self._last_phase_np.astype(np.float32).tobytes()
                self.result_json = json.dumps({
                    "id": pin_entry.get("id"),
                    "C10": float(pin_entry.get("C10", 0.0)),
                    "C12": float(pin_entry.get("C12", 0.0)),
                    "phi12_deg": float(pin_entry.get("phi12_deg", 0.0)),
                    "loss": pin_entry.get("loss"),
                    "time_ms": None,
                })
                break
        elif action == "unpin":
            pin_id = evt.get("id")
            removed_starred = any(
                p.get("id") == pin_id and p.get("starred") for p in self._pinned
            )
            self._pinned = [p for p in self._pinned if p.get("id") != pin_id]
            if removed_starred:
                self._write_stars_file()

    def _on_save(self, change):
        """Write the currently-viewed aberrations as calibration.json.

        Mirrors the chosen aberrations onto ``self._ssb_ref.aberrations`` so
        any Python code still holding the SSB instance sees the update.
        """
        if self._last_phase_np is None or not self.result_json:
            return

        latest = json.loads(self.result_json)
        c10 = float(latest.get("C10", latest.get("c10", 0.0)))
        c12 = float(latest.get("C12", latest.get("c12", 0.0)))
        phi12_deg = float(latest.get("phi12_deg", 0.0))
        phi12_rad = math.radians(phi12_deg)
        loss = latest.get("loss")

        if self._ssb_ref is not None:
            self._ssb_ref.aberrations["C10"] = c10
            self._ssb_ref.aberrations["C12"] = c12
            self._ssb_ref.aberrations["phi12"] = phi12_rad
            if loss is not None:
                self._ssb_ref.best_loss = float(loss)

        # Pull microscope metadata off the SSB instance - single source of
        # truth, avoids duplicating voltage/semiangle/scan_sampling in the
        # widget's constructor.
        ssb = self._ssb_ref
        voltage_kV = ssb.voltage_kV if ssb is not None else None
        semiangle_mrad = ssb.semiangle_mrad if ssb is not None else None
        scan_sampling = ssb.scan_sampling_A if ssb is not None else None
        if isinstance(scan_sampling, (tuple, list)):
            scan_sampling_A = float(scan_sampling[0])
        elif scan_sampling is not None:
            scan_sampling_A = float(scan_sampling)
        else:
            scan_sampling_A = None
        source_file = self._source_file

        # Merge in every non-zero higher-order coefficient from the HO panel.
        # The widget stores magnitudes in nm and angles in degrees; we keep
        # that convention in calibration.json so load-then-restore is a plain
        # copy of the dict.  phi12 remains in radians for historical reasons
        # (matches the existing Calibration contract downstream code expects).
        aberrations = {"C10": c10, "C12": c12, "phi12": phi12_rad}
        ho_dict = json.loads(self.higher_order_json or "{}")
        for k, v in ho_dict.items():
            fv = float(v)
            if k.endswith("_angle"):
                aberrations[k] = fv            # degrees
            elif abs(fv) > 0:
                aberrations[k] = fv            # nm (magnitude or single-coef like C30)

        cal_kwargs = dict(
            rotation_angle_deg=math.degrees(self._rotation_rad),
            aberrations=aberrations,
            flip_phase=bool(self.flip_phase),
            voltage_kV=voltage_kV,
            semiangle_mrad=semiangle_mrad,
            scan_sampling_A=scan_sampling_A,
            loss=float(loss) if loss is not None else None,
            source_file=source_file,
            scan_region=self._scan_region,
        )
        if (self.notes or None) is not None:
            cal_kwargs["notes"] = self.notes
        cal = PtychoCalibration(**cal_kwargs)
        saved = save_ptycho_calibration(cal, self._calibration_path)
        self.calibration_path = str(saved.resolve())
        self.calibration_saved_at = datetime.datetime.now().isoformat(timespec="seconds")

    def __repr__(self):
        n = len(self._pinned)
        drag = ""
        if self._drag_state is not None:
            bf = self._drag_state.num_bf
            drag = f", drag_bf={bf}"
        return f"ShowPtycho({n} pinned{drag})"

    def export(
        self,
        path: str | pathlib.Path | None = None,
        *,
        backend: str = "webgpu",
        data: str | pathlib.Path | None = None,
        title: str | None = None,
        overwrite: bool = True,
        decode_dtype: str = "uint16",
        webgpu_source: str = "bf_columns",
    ) -> pathlib.Path:
        """Export a shareable interactive viewer folder for the current state.

        This is the standard export verb across all widgets. It writes an
        ``index.html`` viewer plus a double-click ``ShowPtycho.command`` launcher,
        so the reconstruction can be reopened later with no kernel.

        Parameters
        ----------
        path : str or Path, optional
            Where to write the viewer folder. If ``None``, defaults to the
            directory of ``data`` (in-place) when given, else a folder named from
            the source file next to it.
        backend : str, default "webgpu"
            Compute backend the exported viewer uses. Only ``"webgpu"`` is
            supported today; the argument exists so other backends can be added
            without a new method.
        data : {None, "in-place"}, optional
            How the HDF5 source is handled. ``None`` (default) bundles the source
            into the viewer folder, hard-linked when on the same filesystem so
            there is no real copy - the folder is self-contained and portable.
            ``"in-place"`` writes the viewer next to the existing source and
            serves it there, so nothing is copied at all (the viewer is tied to
            that data folder).
        webgpu_source : {"bf_columns", "hdf5"}, default "bf_columns"
            Browser source layout. ``"bf_columns"`` writes exact detector
            bright-field columns and uses them on open, avoiding compressed HDF5
            decode unless a fallback is explicitly requested. ``"hdf5"`` keeps
            the older compressed-source browser path.
        """
        if backend != "webgpu":
            raise ValueError(
                f"backend={backend!r} is not supported; only 'webgpu' today."
            )
        from quantem.widget.showptycho_webgpu_export import (
            export_showptycho_webgpu_folder,
        )

        out_dir = _resolve_export_dir(self, path, data)
        return export_showptycho_webgpu_folder(
            self,
            out_dir,
            title=title,
            overwrite=overwrite,
            decode_dtype=decode_dtype,
            webgpu_source=webgpu_source,
        )


def _is_ssb_like(obj: object) -> bool:
    return isinstance(obj, SSB)


def _scan_sampling_scalar(ssb: SSB) -> float:
    scan_sampling = ssb.scan_sampling_A
    if isinstance(scan_sampling, (tuple, list)):
        return float(scan_sampling[0])
    return float(scan_sampling or 0.0)


def _apply_calibration(
    ssb: SSB,
    calibration: object,
    source_file: str | None,
) -> tuple[bool, dict[str, float], float | None, str | None]:
    cal = _coerce_calibration(calibration)
    primary = {
        "C10": float(cal.aberrations.get("C10", 0.0)),
        "C12": float(cal.aberrations.get("C12", 0.0)),
        "phi12": float(cal.aberrations.get("phi12", 0.0)),
    }
    for key, value in {**cal.higher_order, **cal.aberrations}.items():
        if key not in primary:
            primary[str(key)] = float(value)
    ssb.aberrations = primary
    ssb.set_rotation(float(cal.rotation_angle_deg))
    if cal.loss is not None:
        ssb.best_loss = float(cal.loss)
    return (
        bool(cal.flip_phase),
        _higher_order_widget_payload(cal),
        None if cal.loss is None else float(cal.loss),
        source_file or cal.source_file,
    )


def _show_ptycho_from_ssb(
    ssb: SSB,
    *,
    c10_range: tuple[float, float] | None,
    c12_range: tuple[float, float] | None,
    phi12_range: tuple[float, float] | None,
    rotation_range: tuple[float, float] | None,
    drag_bf: int | float | None,
    save_dir: str | pathlib.Path | None,
    source_file: str | None,
    size: int,
    fft_on: bool,
    calibration: object | None,
) -> _ShowPtychoWidget:
    flip_from_cal: bool | None = None
    ho_from_cal: dict[str, float] | None = None
    loss_from_cal: float | None = None
    if calibration is not None:
        flip_from_cal, ho_from_cal, loss_from_cal, source_file = _apply_calibration(
            ssb, calibration, source_file,
        )

    aberrations = dict(ssb.aberrations)
    auto_c10 = float(aberrations.get("C10", 0.0))

    if c10_range is None:
        c10_range = (min(-300.0, auto_c10), max(300.0, auto_c10))
    if c12_range is None:
        c12_range = (-100.0, 100.0)
    if phi12_range is None:
        phi12_range = (-90.0, 90.0)

    accel = ssb
    rotation_rad = math.radians(float(ssb.rotation_angle_deg))
    accel.set_rotation(math.degrees(rotation_rad))
    auto_loss_val = (
        float(loss_from_cal)
        if loss_from_cal is not None and math.isfinite(float(loss_from_cal))
        else float("nan")
    )
    initial_compute_loss = calibration is None and not math.isfinite(auto_loss_val)

    widget = _ShowPtychoWidget(
        accel=accel,
        rotation_rad=rotation_rad,
        auto_aberrations=aberrations,
        auto_loss_val=auto_loss_val,
        c10_range=c10_range,
        c12_range=c12_range,
        phi12_range=phi12_range,
        rotation_range=rotation_range,
        drag_bf=drag_bf,
        save_dir=save_dir,
        ssb_ref=ssb,
        pixel_size=_scan_sampling_scalar(ssb),
        source_file=source_file,
        size=size,
        fft_on=fft_on,
        initial_compute_loss=initial_compute_loss,
        initial_loss_val=auto_loss_val if math.isfinite(auto_loss_val) else None,
        initial_flip_phase=bool(flip_from_cal) if flip_from_cal is not None else False,
        initial_higher_order=ho_from_cal,
    )

    return widget


def ShowPtycho(
    data_or_ssb: object,
    *,
    backend: Literal["auto", "cuda", "mps"] = "auto",
    semiangle_mrad: float | None = None,
    scan_sampling_A: float | tuple[float, float] | None = None,
    det_sampling: float | tuple[float, float] | None = None,
    voltage_kV: float | None = None,
    scan_shape: tuple[int, int] | None = None,
    bf_intensity_threshold: float = 0.5,
    bf_radius: int | None = None,
    aberrations: dict[str, float] | None = None,
    rotation_angle_deg: float = 0.0,
    c10_range: tuple[float, float] | None = None,
    c12_range: tuple[float, float] | None = None,
    phi12_range: tuple[float, float] | None = None,
    rotation_range: tuple[float, float] | None = None,
    drag_bf: int | float | None = _DEFAULT_DRAG_BF_FRACTION,
    save_dir: str | pathlib.Path | None = None,
    source_file: str | None = None,
    size: int = 800,
    fft_on: bool = False,
    calibration: object | None = None,
) -> _ShowPtychoWidget:
    """Open an interactive ptychography aberration explorer.

    Parameters
    ----------
    data_or_ssb : object
        Either a prepared ``quantem.gpu.SSB`` instance or a 4D-STEM array.
        Passing an SSB instance is the preferred path because it reuses the
        existing GPU-resident preprocessing buffers.
    backend : {"auto", "cuda", "mps"}, default "auto"
        Compute backend used when ``data_or_ssb`` is raw detector data.
        Prepared sessions already own their backend; a conflicting explicit
        value raises an error rather than silently moving evidence.
    semiangle_mrad : float, optional
        Probe semi-convergence angle in mrad. Required when passing raw data.
    scan_sampling_A : float or tuple[float, float], optional
        Scan sampling in Angstroms. Required when passing raw data.
    det_sampling : float or tuple[float, float], optional
        Detector angular sampling in mrad per pixel.
    voltage_kV : float, optional
        Accelerating voltage in kV. Required when passing raw data.
    source_file : str, optional
        Path to the raw 4D-STEM master HDF5 file. When this path is available,
        the toolbar offers native square real-space crops and can rebuild/refit
        SSB from the selected detector data with 200 optimization trials.
    calibration : path or object, optional
        Previously saved calibration used to seed aberrations, rotation, phase
        flip, and higher-order controls.
    Returns
    -------
    anywidget.AnyWidget
        ShowPtycho widget instance backed by the ``quantem.gpu`` SSB engine.
    """

    calibration_seed = _coerce_calibration(calibration) if calibration is not None else None
    prepared_rotation_angle_deg = float(rotation_angle_deg)
    prepared_aberrations = aberrations
    if calibration_seed is not None:
        prepared_rotation_angle_deg = float(calibration_seed.rotation_angle_deg)
        prepared_aberrations = {
            **(aberrations or {}),
            **calibration_seed.higher_order,
            **calibration_seed.aberrations,
        }

    if _is_ssb_like(data_or_ssb):
        ssb = data_or_ssb
        prepared_backend = ssb.backend
        if backend != "auto" and backend != prepared_backend:
            raise ValueError(
                f"ShowPtycho received a prepared {prepared_backend!r} session "
                f"with backend={backend!r}. Use backend='auto' or "
                f"backend={prepared_backend!r}."
            )
    else:
        if semiangle_mrad is None or scan_sampling_A is None or voltage_kV is None:
            raise ValueError(
                "ShowPtycho(data, ...) requires voltage_kV, semiangle_mrad, "
                "and scan_sampling_A. Pass a prepared quantem.gpu.SSB object to reuse an "
                "existing GPU-resident reconstruction."
            )
        ssb = SSB.from_array(
            data_or_ssb,
            backend=backend,
            voltage_kV=float(voltage_kV),
            semiangle_mrad=float(semiangle_mrad),
            scan_sampling_A=scan_sampling_A,
            det_sampling=det_sampling,
            scan_shape=scan_shape,
            bf_intensity_threshold=bf_intensity_threshold,
            bf_radius=bf_radius,
            aberrations=prepared_aberrations,
            rotation_angle_deg=prepared_rotation_angle_deg,
        )

    return _show_ptycho_from_ssb(
        ssb,
        c10_range=c10_range,
        c12_range=c12_range,
        phi12_range=phi12_range,
        rotation_range=rotation_range,
        drag_bf=drag_bf,
        save_dir=save_dir,
        source_file=source_file,
        size=size,
        fft_on=fft_on,
        calibration=calibration,
    )
