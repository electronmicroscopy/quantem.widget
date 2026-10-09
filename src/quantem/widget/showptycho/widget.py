"""ShowPtycho: view a ptychographic reconstruction, and tune SSB aberrations on a quantem.gpu session.

Given an array (a phase image, or a complex object wave), the widget shows the phase, the
amplitude of a complex wave, their FFT and histograms with a calibrated scale bar, on any
computer. Given a fitted ``quantem.gpu.SSB`` session it owns the session: every slider move asks
it for a transient ``preview`` (C10, C12, phi12, scan rotation, optional higher-order terms,
optional thick-sample tilt) and ships the float32 phase to the browser, which draws it and its
FFT. Save writes the chosen aberrations to ``calibration.json``; starring a pinned snapshot
appends it to ``showptycho_stars.json``.

Usage::

    from quantem.widget import ShowPtycho

    ShowPtycho(object_wave, sampling=0.2, units="Å")   # NumPy or torch, any device

    from quantem.gpu import SSB

    ssb = SSB.open("scan_master.h5", voltage_kV=300, semiangle_mrad=30, scan_sampling_A=0.264)
    ssb.find_aberrations()
    ShowPtycho(ssb, fft_on=True)
"""

import contextlib
import datetime
import json
import math
import pathlib
import uuid

import anywidget
import numpy as np
import torch
import traitlets

from quantem.widget.adapters import core as core_adapter
from quantem.widget.adapters import gpu as gpu_adapter
from quantem.widget.device import gpu_path_hint, no_gpu_path
from quantem.widget.showptycho.export import write_showptycho_folder

STARS_FILENAME = "showptycho_stars.json"
CALIBRATION_FILENAME = "calibration.json"
CALIBRATION_SCHEMA_VERSION = 3
# Saved magnitudes are nm (quantem.gpu SSB reported Angstrom under an nm label before 2026-09-24).
ABERRATION_UNIT = "nm"
MIN_CROP_SPAN = 32
# length units an array's sampling may be given in, as Angstrom per unit (the widget draws in Angstrom)
ANGSTROM_PER_UNIT = {"Å": 1.0, "A": 1.0, "nm": 10.0, "pm": 0.01}
# (name, slot in the 14-coefficient Krivanek vector, carries an azimuth) for the higher-order panel;
# slots 0 and 1 are C10 and C12 / phi12 from the main sliders.
HIGHER_ORDER_LAYOUT = (
    ("C21", 2, True), ("C23", 3, True),
    ("C30", 4, False), ("C32", 5, True), ("C34", 6, True),
    ("C41", 7, True), ("C43", 8, True), ("C45", 9, True),
    ("C50", 10, False), ("C52", 11, True), ("C54", 12, True), ("C56", 13, True),
)


def session_note() -> str:
    """The line an array input shows in place of the aberration controls, true for this machine.

    With quantem.gpu installed the missing piece is a session, so the line says how to open one;
    without it, the install command for this machine's GPU, or why this machine has no GPU route.
    A fixed line naming both extras told users with quantem.gpu installed to install it again.
    """
    if gpu_adapter.available():
        return (
            "Aberration tuning (C10, C12, rotation, Save) needs an SSB session: "
            'ssb = quantem.gpu.SSB.open("scan_master.h5", voltage_kV=..., semiangle_mrad=..., scan_sampling_A=...), '
            "ssb.find_aberrations(), then ShowPtycho(ssb)."
        )
    hint = gpu_path_hint()
    if hint:
        return f"Aberration tuning (C10, C12, rotation, Save) needs an SSB session from quantem.gpu: {hint}."
    return f"Aberration tuning (C10, C12, rotation, Save) needs an SSB session from quantem.gpu. {no_gpu_path()}."


def finite_or_none(value: object) -> float | None:
    """A finite float for JSON, or None for missing, non-numeric or non-finite values."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def scan_sampling_scalar(ssb) -> float:
    """The row scan step in Angstrom; ``SSB.scan_sampling_A`` may be one number or (row, col)."""
    sampling = ssb.scan_sampling_A
    if isinstance(sampling, (tuple, list)):
        return float(sampling[0])
    return float(sampling or 0.0)


def object_frame_tilt(tilt_row_mrad: float, tilt_col_mrad: float, rotation_deg: float) -> list[float]:
    """SSB tilt (scan frame) in the ptychography object frame: ``[[cos, sin], [-sin, cos]]`` of the scan-detector rotation.

    quantem.thick places scan positions in the object frame by the same rotation with no sign change, so this value
    seeds its reconstruction directly (swap the components only if that reconstruction transposed the scan axes).
    Verified on a logic-device dataset (rotation -8.6 deg): SSB (-10.3, +4.7) -> (-10.9, +3.1) mrad against
    ptychography's learned (-11.6, +2.8), 2.4 deg apart.
    """
    rotation = math.radians(float(rotation_deg))
    return [
        tilt_row_mrad * math.cos(rotation) + tilt_col_mrad * math.sin(rotation),
        -tilt_row_mrad * math.sin(rotation) + tilt_col_mrad * math.cos(rotation),
    ]


def write_json(path: pathlib.Path, payload: object) -> None:
    """Write JSON through a sibling temp file so a reader never sees a half-written calibration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".tmp")
    partial.write_text(json.dumps(payload, indent=2, default=str))
    partial.replace(path)


def has_higher_order(panel_json: str) -> bool:
    """Whether the higher-order panel JSON holds a non-zero magnitude (its angles do not count).

    Upsampled output carries C10/C12 only, so the two validators use it to refuse a
    higher-order term and upsampling together.
    """
    return any(value != 0 for key, value in json.loads(panel_json or "{}").items() if not key.endswith("_angle"))


class ShowPtycho(anywidget.AnyWidget):
    """Ptychographic reconstruction viewer, and SSB aberration explorer over a ``quantem.gpu.SSB`` session.

    Parameters
    ----------
    data : numpy.ndarray, torch.Tensor, quantem Dataset2d or quantem.gpu.SSB
        A 2D reconstruction or a prepared SSB session. A ``Dataset2d`` gives its array, and its
        sampling when its units are a length. A real array is the phase (rad); a complex
        array is the object wave, shown as its phase (``angle``) and amplitude (``abs``). Tensors
        may live on any device. Arrays need no quantem.gpu; the aberration controls then stay
        hidden and one line says what they need. An SSB session, normally after
        ``ssb.find_aberrations()``, is reused for its resident bright-field evidence and
        exclusively driven while the widget is open.
    sampling : float or (row, col), optional
        Pixel size of an array in ``units``; it draws the scale bar and calibrates the FFT. An SSB
        session carries its own scan sampling.
    units : str, default "Å"
        Length unit of ``sampling``: ``"Å"``, ``"nm"`` or ``"pm"``.
    source_file : str, optional
        SSB session: path of the raw ``*_master.h5``. With a readable master the toolbar offers
        Crop and Refit SSB, which reloads only the selected scan region and fits it afresh; the
        path is also recorded in every saved calibration and star.
    save_dir : str or Path, optional
        SSB session: where ``calibration.json`` and ``showptycho_stars.json`` are written; the
        working directory by default.
    fft_on : bool, default False
        Open with the FFT panel visible.
    """

    _esm = pathlib.Path(__file__).parent.parent / "static" / "showptycho.js"

    # slider ranges (nm, nm, degrees, degrees): the physical span of the SSB fit's default search
    c10_min = traitlets.Float(-40.0).tag(sync=True)
    c10_max = traitlets.Float(40.0).tag(sync=True)
    rotation_min = traitlets.Float(-180.0).tag(sync=True)
    rotation_max = traitlets.Float(180.0).tag(sync=True)
    rotation_deg = traitlets.Float(0.0).tag(sync=True)
    # SSB phase is ambiguous by sign; the user picks the convention matching the expected contrast
    flip_phase = traitlets.Bool(False).tag(sync=True)
    # the fitted reference the Reset button returns to
    auto_c10 = traitlets.Float(0.0).tag(sync=True)
    auto_c12 = traitlets.Float(0.0).tag(sync=True)
    auto_phi12_deg = traitlets.Float(0.0).tag(sync=True)
    auto_loss = traitlets.Float(0.0).tag(sync=True)
    auto_rotation_deg = traitlets.Float(0.0).tag(sync=True)
    # browser -> Python reconstruction request {"id", "c10", "c12", "phi12_deg", "committed"}
    request_json = traitlets.Unicode("").tag(sync=True)
    phase_bytes = traitlets.Bytes(b"").tag(sync=True)
    phase_width = traitlets.Int(0).tag(sync=True)
    phase_height = traitlets.Int(0).tag(sync=True)
    result_json = traitlets.Unicode("").tag(sync=True)
    pixel_size = traitlets.Float(0.0).tag(sync=True)
    # one-shot seed the browser reads at mount; the FFT switch takes over
    initial_fft_on = traitlets.Bool(False).tag(sync=True)
    # SSB alias-order output sampling (CUDA sessions), not image interpolation
    upsample = traitlets.Int(1).tag(sync=True)
    upsampling_available = traitlets.Bool(False).tag(sync=True)
    save_trigger = traitlets.Int(0).tag(sync=True)
    notes = traitlets.Unicode("").tag(sync=True)
    pin_json = traitlets.Unicode("").tag(sync=True)
    stars_path = traitlets.Unicode("").tag(sync=True)
    calibration_path = traitlets.Unicode("").tag(sync=True)
    calibration_saved_at = traitlets.Unicode("").tag(sync=True)
    # fit history for the trials strip: JSON list of {rank, C10, C12, phi12_deg, loss}, best first
    trials_json = traitlets.Unicode("").tag(sync=True)
    # bright-field pixels used while dragging; the full count is the right edge of the slider
    drag_bf = traitlets.Int(0).tag(sync=True)
    total_bf = traitlets.Int(0).tag(sync=True)
    # higher-order panel: {C21_mag, C21_angle, ..., C30, ..., C50, ...}, magnitudes nm, angles degrees
    higher_order_json = traitlets.Unicode("{}").tag(sync=True)
    # browser-side WebGPU SSB source, set by the folder export
    webgpu_preview_enabled = traitlets.Bool(False).tag(sync=True)
    webgpu_cal_json = traitlets.Unicode("").tag(sync=True)
    webgpu_h5_source_json = traitlets.Unicode("").tag(sync=True)
    # browser SSB status; only an exported folder runs SSB in the browser (a live notebook asks Python)
    webgpu_preview_status = traitlets.Unicode("").tag(sync=True)
    webgpu_standalone = traitlets.Bool(False).tag(sync=True)
    # crop and refit: rebuilds SSB from the raw scan region, not a display crop
    scan_rows = traitlets.Int(0).tag(sync=True)
    scan_cols = traitlets.Int(0).tag(sync=True)
    crop_refit_available = traitlets.Bool(False).tag(sync=True)
    crop_refit_status = traitlets.Unicode("").tag(sync=True)
    crop_refit_request_json = traitlets.Unicode("").tag(sync=True)
    # thick-sample SSB: {"tilt_row_mrad", "tilt_col_mrad", "thickness_nm"}; thickness 0 is standard SSB
    sample_json = traitlets.Unicode("{}").tag(sync=True)
    sample_available = traitlets.Bool(False).tag(sync=True)
    sample_fit_request = traitlets.Int(0).tag(sync=True)
    sample_fit_status = traitlets.Unicode("").tag(sync=True)
    sample_fit_json = traitlets.Unicode("").tag(sync=True)
    # an array input has no session: the browser hides the aberration controls and shows session_note
    ssb_session = traitlets.Bool(True).tag(sync=True)
    session_note = traitlets.Unicode("").tag(sync=True)
    # float32 amplitude of a complex object wave, row-major like phase_bytes; empty for a phase
    amplitude_bytes = traitlets.Bytes(b"").tag(sync=True)

    def __init__(
        self,
        data,
        *,
        sampling: float | tuple[float, float] | None = None,
        units: str = "Å",
        source_file: str | None = None,
        save_dir: str | pathlib.Path | None = None,
        fft_on: bool = False,
    ) -> None:
        if core_adapter.is_dataset(data, ndim=2):
            # a calibrated reconstruction: a length unit on its axes draws the scale bar
            if sampling is None and str(data.units[-1]) in ANGSTROM_PER_UNIT:
                sampling, units = tuple(float(value) for value in data.sampling), str(data.units[-1])
            data = core_adapter.as_array(data)
        is_array = isinstance(data, (np.ndarray, torch.Tensor))
        if not is_array and not gpu_adapter.is_ssb(data):
            needs = "" if gpu_adapter.available() else f"; a session needs quantem.gpu ({gpu_path_hint() or no_gpu_path()})"
            raise TypeError(
                "ShowPtycho takes a 2D phase or complex object-wave array (NumPy, torch or a quantem Dataset2d) or a prepared "
                f"quantem.gpu.SSB session{needs}; got {type(data).__name__}."
            )
        if not is_array and sampling is not None:
            raise ValueError("sampling describes an array; an SSB session carries its own scan_sampling_A.")
        super().__init__()
        self._ssb = None if is_array else data
        self._pinned: list[dict] = []
        self._last_phase: np.ndarray | None = None
        self._last_result: dict = {}
        self._inflight_id = -1
        self._drag_context = None
        self.initial_fft_on = bool(fft_on)
        self.observe(self._on_flip_change, names=["flip_phase"])
        if is_array:
            # angle and abs run where the data lives (a CUDA or MPS tensor stays there until the 2D
            # result is copied for the browser), in the input's precision; the browser draws float32,
            # so wider input is rounded once and the largest change is printed, never silently
            if data.ndim != 2:
                raise ValueError(f"ShowPtycho shows one 2D reconstruction; got shape {tuple(data.shape)}. Pass one slice.")
            if units not in ANGSTROM_PER_UNIT:
                raise ValueError(f"units must be one of {sorted(ANGSTROM_PER_UNIT)}; got {units!r}")
            is_tensor = isinstance(data, torch.Tensor)
            data = data.detach() if is_tensor else data
            if data.is_complex() if is_tensor else np.iscomplexobj(data):
                channels = {"phase": data.angle() if is_tensor else np.angle(data), "amplitude": data.abs() if is_tensor else np.abs(data)}
            else:
                channels = {"phase": data}
            drawn, changes = {}, []
            for name, values in channels.items():
                if is_tensor:
                    rounded = values.to(torch.float32)
                    change = float((rounded.to(values.dtype) - values).abs().max())
                    drawn[name] = np.ascontiguousarray(rounded.cpu().numpy())
                else:
                    rounded = np.ascontiguousarray(values, dtype=np.float32)
                    change = float(np.max(np.abs(rounded.astype(np.float64) - values.astype(np.float64))))
                    drawn[name] = rounded
                if change > 0.0:
                    changes.append(f"{name} {change:.3g}")
            if changes:
                print(f"ShowPtycho: {data.dtype} input is drawn in float32; largest change: {', '.join(changes)}.")
            self._last_phase = drawn["phase"]
            self.phase_height, self.phase_width = self._last_phase.shape
            self.phase_bytes = self._last_phase.tobytes()
            if "amplitude" in drawn:
                self.amplitude_bytes = drawn["amplitude"].tobytes()
            if sampling is not None:
                column_sampling = float(sampling) if isinstance(sampling, (int, float)) else float(sampling[-1])
                self.pixel_size = column_sampling * ANGSTROM_PER_UNIT[units]
            self.ssb_session = False
            self.session_note = session_note()
            return
        ssb = data
        self._source_file = str(source_file) if source_file else None
        self._source_scan_shape = ssb.scan_shape
        self._scan_region = (0, ssb.scan_shape[0], 0, ssb.scan_shape[1])
        save_root = pathlib.Path(save_dir) if save_dir else pathlib.Path.cwd()
        self._stars_path = save_root / STARS_FILENAME
        self._calibration_path = save_root / CALIBRATION_FILENAME
        self.stars_path = str(self._stars_path.resolve())
        self.upsampling_available = ssb.backend == "cuda"
        self.sample_available = ssb.supports_tilt
        self.pixel_size = scan_sampling_scalar(ssb)
        self.crop_refit_available = bool(self._source_file and pathlib.Path(self._source_file).is_file())
        self.crop_refit_status = (
            "Ready to refit a native square scan crop." if self.crop_refit_available
            else "Crop refit needs a re-loadable master file."
        )
        self.observe(self._on_request, names=["request_json"])
        self.observe(self._on_upsample, names=["upsample"])
        self.observe(self._on_save, names=["save_trigger"])
        self.observe(self._on_pin, names=["pin_json"])
        self.observe(self._on_drag_bf_change, names=["drag_bf"])
        self.observe(self._on_rotation_change, names=["rotation_deg"])
        self.observe(self._on_panel_change, names=["higher_order_json", "sample_json"])
        self.observe(self._on_crop_refit_request, names=["crop_refit_request_json"])
        self.observe(self._on_sample_fit_request, names=["sample_fit_request"])
        self._adopt_session(initial=True)
        # the Sample panel opens on the session's latest find_aberrations(tilt=True)
        if self.sample_available and ssb.tilt_mrad is not None and ssb.depth_spread_nm:
            self.sample_json = json.dumps({"tilt_row_mrad": float(ssb.tilt_mrad[0]), "tilt_col_mrad": float(ssb.tilt_mrad[1]), "thickness_nm": float(ssb.depth_spread_nm)})

    def _adopt_session(self, *, initial: bool) -> None:
        """Publish the session's geometry and fit as the widget's reference, then reconstruct at that fit.

        Runs once at construction and again after a crop refit replaces the session. The rotation slider
        spans at least +-180 deg so negative microscope conventions need no mental +180; the C10 range
        widens to include an outlier fit so the reference is always reachable.
        """
        ssb = self._ssb
        self.scan_rows, self.scan_cols = ssb.scan_shape
        rotation = float(ssb.physical_rotation_deg)
        fitted_c10 = float(ssb.aberrations.get("C10", 0.0))
        ssb.set_rotation(rotation)
        if initial:
            self.c10_min, self.c10_max = min(-30.0, fitted_c10), max(30.0, fitted_c10)
            self.rotation_min, self.rotation_max = min(-180.0, rotation), max(180.0, rotation)
        # the session is already at this rotation: the observer must not rebuild and reconstruct
        self._adopting = True
        self.rotation_deg = rotation
        self._adopting = False
        self.auto_rotation_deg = rotation
        self.auto_c10 = fitted_c10
        self.auto_c12 = float(ssb.aberrations.get("C12", 0.0))
        self.auto_phi12_deg = math.degrees(float(ssb.aberrations.get("phi12", 0.0)))
        self.auto_loss = finite_or_none(ssb.best_loss) or 0.0
        self.higher_order_json = "{}"
        self.trials_json = self._trials_payload()
        self.total_bf = ssb.num_bf
        self._inflight_id += 1
        self._reconstruct(self._inflight_id, self.auto_c10, self.auto_c12, self.auto_phi12_deg, compute_loss=True)
        # drags start on the full disk; the slider's left side trades pixels for speed
        self._free_drag_context()
        self.drag_bf = self.total_bf
        self._drag_context = ssb.preview_context(self.total_bf)

    def _trials_payload(self, max_trials: int = 50) -> str:
        """The session's fit trials for the browser strip, best loss first, capped so a 200-trial run stays light."""
        finite = [(loss, trial) for trial in self._ssb.trial_history if (loss := finite_or_none(trial.get("loss"))) is not None]
        payload = [
            {"rank": rank, "C10": float(trial["params"].get("C10_nm", 0.0)), "C12": float(trial["params"].get("C12_nm", 0.0)),
             "phi12_deg": float(trial["params"].get("phi12_deg", 0.0)), "loss": loss}
            for rank, (loss, trial) in enumerate(sorted(finite, key=lambda item: item[0])[:max_trials])
        ]
        return json.dumps(payload) if payload else ""

    def _current(self) -> tuple[float, float, float]:
        """(C10 nm, C12 nm, phi12 deg) of the displayed phase, the fitted reference before the first request."""
        return (
            float(self._last_result.get("C10", self.auto_c10)),
            float(self._last_result.get("C12", self.auto_c12)),
            float(self._last_result.get("phi12_deg", self.auto_phi12_deg)),
        )

    def _tilt(self) -> dict | None:
        """The Sample panel as ``SSB.preview`` arguments, or None for standard SSB (depth spread 0)."""
        values = json.loads(self.sample_json or "{}")
        depth_spread_nm = float(values.get("thickness_nm", 0.0))
        if depth_spread_nm <= 0.0:
            return None
        return {"tilt_mrad": (float(values.get("tilt_row_mrad", 0.0)), float(values.get("tilt_col_mrad", 0.0))), "depth_spread_nm": depth_spread_nm}

    def _free_drag_context(self) -> None:
        """Release the reduced bright-field subset the session built for drags before a new one
        replaces it, so two subsets never hold device memory at once."""
        if self._drag_context is not None:
            self._drag_context.close()
        self._drag_context = None

    def _reconstruct(self, request_id: int, c10: float, c12: float, phi12_deg: float, *, compute_loss: bool) -> None:
        """Ask the session for the phase at these aberrations and push it to the browser.

        The three main sliders fill slots 0 and 1 of the 14-coefficient Krivanek vector; a non-zero
        higher-order panel switches the session to its full chi kernel. Drags (``compute_loss=False``)
        run inside the reduced bright-field context and skip the variance loss so slider frames stay
        under budget; a committed request computes the same phase-variance loss the optimizer minimised.
        A stale request (another one was issued meanwhile) is dropped after the compute.
        """
        magnitudes = np.zeros(14, dtype=np.float32)
        angles = np.zeros(14, dtype=np.float32)
        magnitudes[0], magnitudes[1], angles[1] = c10, c12, math.radians(phi12_deg)
        higher_order = json.loads(self.higher_order_json or "{}")
        any_higher_order = False
        for name, slot, has_angle in HIGHER_ORDER_LAYOUT:
            magnitude = float(higher_order.get(f"{name}_mag" if has_angle else name, 0.0))
            any_higher_order |= magnitude != 0.0
            magnitudes[slot] = magnitude
            angles[slot] = math.radians(float(higher_order.get(f"{name}_angle", 0.0))) if has_angle else 0.0
        tilt = self._tilt() if self.sample_available else None
        if tilt is not None and any_higher_order:
            # the thick-sample kernel carries C10/C12/phi12 only; higher-order wins and the panel says so
            self.sample_fit_status = "Sample tilt is ignored while higher-order aberrations are non-zero."
            tilt = None
        use_drag = not compute_loss and self._drag_context is not None
        with self._drag_context if use_drag else contextlib.nullcontext():
            phase, loss = self._ssb.preview(
                {"C10": c10, "C12": c12, "phi12": math.radians(phi12_deg)},
                compute_loss=compute_loss,
                upsampling_factor=self.upsample,
                higher_order_magnitudes=magnitudes if any_higher_order else None,
                higher_order_angles=angles if any_higher_order else None,
                **(tilt or {}),
            )
        if request_id != self._inflight_id:
            return
        if self.flip_phase:
            phase = -phase
        self._last_phase = phase
        self.phase_height, self.phase_width = phase.shape
        self.phase_bytes = phase.tobytes()
        # optimizer precision: the browser formats labels, rounding here would perturb saved calibrations
        self._last_result = {"id": request_id, "C10": float(c10), "upsample": self.upsample, "C12": float(c12), "phi12_deg": float(phi12_deg), "loss": loss}
        self.result_json = json.dumps(self._last_result)

    def _reconstruct_displayed(self, *, compute_loss: bool) -> None:
        """Reconstruct the displayed aberrations again, under a new request id, after a setting
        outside the three sliders (rotation, sampling, a panel) changed the phase."""
        self._inflight_id += 1
        self._reconstruct(self._inflight_id, *self._current(), compute_loss=compute_loss)

    def _on_request(self, change) -> None:
        """A slider request from the browser: reconstruct at its aberrations; committed requests add the loss."""
        if not change["new"]:
            return
        request = json.loads(change["new"])
        self._inflight_id = request["id"]
        self._reconstruct(request["id"], request["c10"], request["c12"], request["phi12_deg"], compute_loss=request.get("committed", False))

    def _on_upsample(self, change) -> None:
        """Output sampling changed: show the displayed aberrations at the new sampling."""
        self._reconstruct_displayed(compute_loss=False)

    @traitlets.validate("upsample")
    def _validate_upsample(self, proposal):
        """Refuse a factor the session cannot produce: 1, 2, 4 or 8, above 1 only on CUDA and
        only while every higher-order term is zero."""
        factor = proposal["value"]
        if factor not in (1, 2, 4, 8):
            raise ValueError("ShowPtycho upsample must be 1, 2, 4, or 8.")
        if factor > 1 and self._ssb.backend != "cuda":
            raise ValueError("This session supports native sampling only; use upsample=1.")
        if factor > 1 and has_higher_order(self.higher_order_json):
            raise ValueError("Higher-order aberrations require upsample=1; reset them before upsampling.")
        return factor

    @traitlets.validate("higher_order_json")
    def _validate_higher_order_sampling(self, proposal):
        """Refuse a non-zero higher-order term while the output is upsampled."""
        if self.upsample > 1 and has_higher_order(proposal["value"]):
            raise ValueError("Set upsample=1 before changing higher-order aberrations.")
        return proposal["value"]

    def _on_drag_bf_change(self, change) -> None:
        """Rebuild the reduced bright-field context the session uses while the user drags."""
        self._free_drag_context()
        if int(change["new"]) > 0:
            self._drag_context = self._ssb.preview_context(min(int(change["new"]), self._ssb.num_bf))

    def _on_rotation_change(self, change) -> None:
        """A new scan-detector rotation re-indexes the bright-field geometry, so the drag context is rebuilt too."""
        if self._adopting:
            return
        self._free_drag_context()
        self._ssb.set_rotation(float(change["new"]))
        if self.drag_bf > 0:
            self._drag_context = self._ssb.preview_context(int(self.drag_bf))
        self._reconstruct_displayed(compute_loss=True)

    def _on_panel_change(self, change) -> None:
        """Higher-order or Sample panel changed: reconstruct the displayed aberrations with it.

        The first adoption resets the panel before any phase exists; there is nothing to redraw then.
        """
        if self._last_phase is None:
            return
        self._reconstruct_displayed(compute_loss=True)

    def _on_flip_change(self, change) -> None:
        """Negate the displayed phase; no reconstruction, the sign is a display convention."""
        if self._last_phase is None:
            return
        self._last_phase = -self._last_phase
        self.phase_bytes = self._last_phase.tobytes()

    def _on_sample_fit_request(self, change) -> None:
        """Fit C10, C12, phi12, sample tilt and depth spread together and hand the result to the sliders.

        The browser applies ``sample_fit_json`` to the aberration sliders and the Sample panel, which
        triggers the reconstruction. C10 comes back as the defocus at mid-depth, not the thin-sheet optimum.
        """
        if not change["new"]:
            return
        if not self.sample_available:
            self.sample_fit_status = "Tilt fit failed: needs a CUDA or MPS SSB session."
            return
        self.sample_fit_status = "Fitting defocus, astigmatism, sample tilt and thickness..."
        try:
            result = self._ssb.find_aberrations(tilt=True, verbose=False)
        except (RuntimeError, ValueError, NotImplementedError, MemoryError) as exc:
            self.sample_fit_status = f"Tilt fit failed: {exc}"
            return
        tilt_row, tilt_col = (float(value) for value in result.tilt_mrad)
        tilt_object = object_frame_tilt(tilt_row, tilt_col, self.rotation_deg)
        self.sample_fit_json = json.dumps({
            "C10": float(result.aberrations["C10"]), "C12": float(result.aberrations["C12"]),
            "phi12_deg": math.degrees(float(result.aberrations["phi12"])),
            "tilt_row_mrad": tilt_row, "tilt_col_mrad": tilt_col, "tilt_object_mrad": tilt_object,
            "thickness_nm": float(result.depth_spread_nm), "gain": float(result.tilt_fit_gain),
        })
        self.sample_fit_status = (
            f"Tilt fit: ({tilt_row:+.1f}, {tilt_col:+.1f}) mrad scan frame = ({tilt_object[0]:+.1f}, {tilt_object[1]:+.1f}) mrad "
            f"ptychography frame, depth spread {float(result.depth_spread_nm):.1f} nm, fit x{float(result.tilt_fit_gain):.2f} over standard SSB."
        )

    def _on_crop_refit_request(self, change) -> None:
        """Reload one raw scan crop from the master, fit SSB on it and make it the widget's session.

        The session keeps only its bright-field Fourier cache, so a scientifically valid fit on a region
        needs the raw frames again: the acquisition is opened encoded and only the crop is decoded. The
        browser's region is relative to the current crop; it is mapped back to the full scan. A crop is
        reconstruction input, not a display view: the SSB kernels take native square scans, so a
        rectangle must span at least 32 positions each way.
        """
        if not change["new"]:
            return
        try:
            request = json.loads(change["new"])
            region = request["scan_region"]
            if not isinstance(region, (tuple, list)) or len(region) != 4:
                raise TypeError("scan_region must be (row_start, row_stop, col_start, col_stop)")
            n_trials = int(request.get("n_trials", 200))
            if n_trials < 1:
                raise ValueError("SSB refit needs at least one optimization trial.")
            if not self.crop_refit_available or not self._source_file:
                raise RuntimeError("Crop refit is unavailable. Open ShowPtycho with source_file=<master.h5>.")
            origin_row, _, origin_col, _ = self._scan_region
            row_start, row_stop, col_start, col_stop = (int(value) for value in region)
            region = (origin_row + row_start, origin_row + row_stop, origin_col + col_start, origin_col + col_stop)
            source_rows, source_cols = self._source_scan_shape
            if not (0 <= region[0] < region[1] <= source_rows and 0 <= region[2] < region[3] <= source_cols):
                raise ValueError(f"scan crop [{region[0]}:{region[1]}, {region[2]}:{region[3]}] is outside {source_rows}x{source_cols} data")
            if row_stop - row_start < MIN_CROP_SPAN or col_stop - col_start < MIN_CROP_SPAN:
                raise ValueError(
                    f"SSB refit crops must be at least {MIN_CROP_SPAN}x{MIN_CROP_SPAN}; got {row_stop - row_start}x{col_stop - col_start}."
                )
            self.crop_refit_status = f"Loading [{region[0]}:{region[1]}, {region[2]}:{region[3]}] and fitting SSB ({n_trials} trials)..."
            previous = self._ssb
            with gpu_adapter.load_acquisition(self._source_file, backend=previous.backend, scan_shape=self._source_scan_shape) as acquisition:
                crop = acquisition.read(scan_region=region)
            c10, c12, phi12_deg = self._current()
            rebuilt = gpu_adapter.ssb_session(
                crop, backend=previous.backend, voltage_kV=previous.voltage_kV, semiangle_mrad=previous.semiangle_mrad,
                scan_sampling_A=previous.scan_sampling_A, det_sampling=previous.det_sampling,
                aberrations={"C10": c10, "C12": c12, "phi12": math.radians(phi12_deg)},
                rotation_angle_deg=self.rotation_deg, bf_intensity_threshold=previous.bf_intensity_threshold,
                bf_radius=previous.bf_radius, source_path=previous.source_path,
            )
            rebuilt.find_aberrations(trials=n_trials, refinement="nelder-mead", verbose=False)
        except (RuntimeError, ValueError, TypeError, KeyError, OSError, MemoryError) as exc:
            self.crop_refit_status = f"Crop refit failed: {exc}"
            return
        self._free_drag_context()
        self._ssb = rebuilt
        self._scan_region = region
        self.sample_available = rebuilt.supports_tilt
        self._adopt_session(initial=False)
        self.crop_refit_status = f"Refit complete: [{region[0]}:{region[1]}, {region[2]}:{region[3]}], {n_trials} trials."

    def _on_pin(self, change) -> None:
        """Pin, star, view or unpin a snapshot of the displayed aberrations.

        A pin captures the higher-order panel and the source file with it, so an old snapshot stays
        reproducible when the panel or dataset changes later; starring writes the stars file.
        """
        if not change["new"]:
            return
        event = json.loads(change["new"])
        action, pin_id = event.get("action"), event.get("id")
        if action == "pin":
            entry = {
                "id": pin_id, "C10": event.get("C10"), "C12": event.get("C12"), "phi12_deg": event.get("phi12_deg"),
                "rotation_deg": event.get("rotation_deg", self.rotation_deg), "flip_phase": bool(event.get("flip_phase", self.flip_phase)),
                "loss": event.get("loss"), "timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "starred": False,
                "source_file": self._source_file, "higher_order": json.loads(self.higher_order_json or "{}"),
            }
            if self._last_phase is not None:
                entry["phase"] = self._last_phase.copy()
            self._pinned.append(entry)
        elif action in ("star", "unstar"):
            for entry in self._pinned:
                if entry["id"] == pin_id:
                    entry["starred"] = action == "star"
                    if action == "star":
                        # the star time, not the pin time, is when the user decided to keep it
                        entry["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
            self._write_stars()
        elif action == "view":
            for entry in self._pinned:
                if entry["id"] == pin_id:
                    if "phase" in entry:
                        self._last_phase = entry["phase"].copy()
                        self.phase_height, self.phase_width = self._last_phase.shape
                        self.phase_bytes = self._last_phase.tobytes()
                    # the viewed snapshot becomes the displayed state, so Save and later slider moves start from it
                    self._last_result = {"id": pin_id, "C10": float(entry.get("C10", 0.0)), "C12": float(entry.get("C12", 0.0)),
                                         "phi12_deg": float(entry.get("phi12_deg", 0.0)), "loss": entry.get("loss")}
                    self.result_json = json.dumps(self._last_result)
        elif action == "unpin":
            starred = any(entry["id"] == pin_id and entry["starred"] for entry in self._pinned)
            self._pinned = [entry for entry in self._pinned if entry["id"] != pin_id]
            if starred:
                self._write_stars()

    def _write_stars(self) -> None:
        """Write the starred snapshots as a list of calibrations (nested ``aberrations`` in nm and radians).

        Higher-order coefficients are merged in under their canonical names (``C21`` with ``phi21`` in
        radians) so ``SSB.reconstruct`` can consume an entry without translation; zero magnitudes are omitted.
        """
        ssb = self._ssb
        payload = []
        for entry in self._pinned:
            if not entry["starred"]:
                continue
            aberrations = {"C10": float(entry.get("C10", 0.0)), "C12": float(entry.get("C12", 0.0)), "phi12": math.radians(float(entry.get("phi12_deg", 0.0)))}
            higher_order = entry["higher_order"]
            for name, _, has_angle in HIGHER_ORDER_LAYOUT:
                magnitude = float(higher_order.get(f"{name}_mag" if has_angle else name, 0.0))
                if magnitude == 0.0:
                    continue
                aberrations[name] = magnitude
                if has_angle:
                    aberrations[f"phi{name[1:]}"] = math.radians(float(higher_order.get(f"{name}_angle", 0.0)))
            payload.append({
                "id": entry["id"], "timestamp": entry["timestamp"], "starred": True,
                "rotation_angle_deg": float(entry["rotation_deg"]), "aberrations": aberrations, "aberration_unit": ABERRATION_UNIT,
                "flip_phase": entry["flip_phase"], "voltage_kV": ssb.voltage_kV, "semiangle_mrad": ssb.semiangle_mrad,
                "scan_sampling_A": scan_sampling_scalar(ssb), "loss": float(entry["loss"]) if entry.get("loss") is not None else None,
                "source_file": entry["source_file"], "notes": None,
            })
        write_json(self._stars_path, payload)

    def _on_save(self, change) -> None:
        """Write the displayed aberrations as ``calibration.json`` and mirror them onto the session.

        Magnitudes are nm and panel angles degrees as the widget shows them; phi12 stays in radians,
        the convention downstream readers of the file expect. The session's ``aberrations`` are updated
        so Python code still holding it sees the user's choice.
        """
        if self._last_phase is None or not self.result_json:
            return
        c10, c12, phi12_deg = self._current()
        loss = self._last_result.get("loss")
        ssb = self._ssb
        ssb.aberrations["C10"], ssb.aberrations["C12"], ssb.aberrations["phi12"] = c10, c12, math.radians(phi12_deg)
        if loss is not None:
            ssb.best_loss = float(loss)
        aberrations = {"C10": c10, "C12": c12, "phi12": math.radians(phi12_deg)}
        for key, value in json.loads(self.higher_order_json or "{}").items():
            if key.endswith("_angle") or abs(float(value)) > 0:
                aberrations[key] = float(value)
        tilt = self._tilt() if self.sample_available else None
        payload = {
            "schema_version": CALIBRATION_SCHEMA_VERSION, "version": "2.0", "aberration_unit": ABERRATION_UNIT,
            "rotation_angle_deg": float(self.rotation_deg), "aberrations": aberrations, "higher_order": {}, "flip_phase": bool(self.flip_phase),
            "voltage_kV": ssb.voltage_kV, "semiangle_mrad": ssb.semiangle_mrad, "scan_sampling_A": scan_sampling_scalar(ssb),
            "det_sampling_mrad_px": None, "loss": float(loss) if loss is not None else None, "source_file": self._source_file,
            "scan_region": self._scan_region, "source_stem": None, "label": None, "notes": self.notes, "id": uuid.uuid4().hex[:12],
            "tilt_mrad": tilt["tilt_mrad"] if tilt else None,
            "tilt_object_mrad": tuple(object_frame_tilt(*tilt["tilt_mrad"], self.rotation_deg)) if tilt else None,
            "depth_spread_nm": tilt["depth_spread_nm"] if tilt else None,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        write_json(self._calibration_path, payload)
        self.calibration_path = str(self._calibration_path.resolve())
        self.calibration_saved_at = payload["timestamp"]

    def export(self, path: str | pathlib.Path | None = None, *, title: str | None = None) -> pathlib.Path:
        """Write a browser folder that reviews this SSB with no kernel: ``index.html``, exact bright-field
        counts under ``source/``, the calibration under ``snapshots/`` and a double-click ``ShowPtycho.command``.

        ``path`` defaults to ``<stem>_showptycho`` beside the source master (or ``showptycho_export`` in the
        working directory). The browser rebuilds the SSB reducers on its GPU from the stored counts, so
        C10, C12, phi12, rotation and the BF count all move live without Python.
        """
        if self._ssb is None:
            raise ValueError("export() writes an SSB review folder from a session's bright-field counts; this widget shows an array.")
        return write_showptycho_folder(self, path, title)

    def __repr__(self) -> str:
        """What the widget shows (array kind, size, sampling) or, for a session, the pins and drag subset."""
        if self._ssb is None:
            kind = "object wave" if self.amplitude_bytes else "phase"
            sampling = f"{self.pixel_size:g} Å/px" if self.pixel_size > 0 else "no sampling"
            return f"ShowPtycho({kind} {self.phase_height}x{self.phase_width}, {sampling})"
        drag = f", drag_bf={self._drag_context.num_bf}" if self._drag_context is not None else ""
        return f"ShowPtycho({len(self._pinned)} pinned{drag})"
