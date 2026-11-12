"""ShowPtycho: view a ptychographic reconstruction, and tune direct ptychography aberrations on a quantem.gpu session.

Given an array (a phase image, or a complex object wave), the widget shows the phase, the
amplitude of a complex wave, their FFT and histograms with a calibrated scale bar, on any
computer. Given a fitted ``quantem.gpu.DirectPtycho`` session it owns the session: every slider move asks
it for a transient ``preview`` (C10, C12, phi12, scan rotation, optional higher-order terms,
optional thick-sample tilt) and ships the float32 phase to the browser, which draws it and its
FFT. Save writes the chosen aberrations to ``calibration.json``; starring a pinned snapshot
appends it to ``showptycho_stars.json``.

Usage::

    from quantem.widget import ShowPtycho

    ShowPtycho(object_wave, sampling=0.2, units="Å")   # NumPy or torch, any device

    from quantem.gpu import DirectPtycho, io

    data = io.load("scan_master.h5")
    data.calibration.voltage_kV = 300
    data.calibration.semiangle_mrad = 30
    data.calibration.scan_sampling_A = 0.264
    ptycho = DirectPtycho(data)
    ptycho.run()
    ptycho.show(show_fft=True)                              # ShowPtycho(ptycho, show_fft=True)
"""

import datetime
import json
import math
import pathlib
import uuid
from collections.abc import Callable

import anywidget
import numpy as np
import torch
import traitlets

from quantem.widget.adapters import core as core_adapter
from quantem.widget.adapters import gpu as gpu_adapter
from quantem.widget.colormap import colorize, contrast_uint8
from quantem.widget.device import gpu_path_hint, no_gpu_path
from quantem.widget.showptycho.export import write_showptycho_folder
from quantem.widget.units import axis_pair, image_axes, per_axis, stated, wrap_degrees
from quantem.widget.utils.array import device_array

STARS_FILENAME = "showptycho_stars.json"
CALIBRATION_FILENAME = "calibration.json"
CALIBRATION_IMAGE_FILENAME = "calibration_phase.png"
# every stored record names its layout; a reader takes the current value only
CALIBRATION_SCHEMA = "quantem.direct-calibration/9"
STARS_SCHEMA = "quantem.direct-stars/6"
MIN_CROP_SPAN = 32
# Damping strengths a slider drags: the browser sends them and shows them itself.
_BROWSER_STRENGTHS = ("phase_damp", "amplitude_damp", "smoothness_inv_A")


def session_note() -> str:
    """The line an array input shows in place of the aberration controls, true for this machine.

    With quantem.gpu installed the missing piece is a session, so the line says how to open one;
    without it, the install command for this machine's GPU, or why this machine has no GPU route.
    A fixed line naming both extras told users with quantem.gpu installed to install it again.
    """
    if gpu_adapter.available():
        return (
            "Aberration tuning (C10, C12, rotation, Save) needs a DirectPtycho session: "
            'data = quantem.gpu.io.load("scan_master.h5"), set data.calibration (voltage_kV, semiangle_mrad, scan_sampling_A), '
            "then ptycho = quantem.gpu.DirectPtycho(data), ptycho.run(), ptycho.show()."
        )
    hint = gpu_path_hint()
    if hint:
        return f"Aberration tuning (C10, C12, rotation, Save) needs a DirectPtycho session from quantem.gpu: {hint}."
    return f"Aberration tuning (C10, C12, rotation, Save) needs a DirectPtycho session from quantem.gpu. {no_gpu_path()}."


def finite_or_none(value: object) -> float | None:
    """A finite float for JSON, or None for missing, non-numeric or non-finite values."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def utc_now() -> str:
    """The present time as ISO 8601 UTC ending in Z: the ``generated_at_utc`` every file of this widget carries.

    A local time without a zone, written before, could not be compared across machines.
    """
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: pathlib.Path, payload: object) -> None:
    """Write JSON through a sibling temp file so a reader never sees a half-written calibration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".tmp")
    partial.write_text(json.dumps(payload, indent=2, default=str))
    partial.replace(path)


def nonzero_higher_order(aberrations: dict) -> dict:
    """The higher-order terms that change the phase: each non-zero magnitude with its azimuth.

    An azimuth (``phi21_deg``) whose magnitude (``C21_nm``) is zero is dropped, so a
    panel left at zero names no higher order and Fit Regions keeps its three-term model.
    """
    def magnitude(name: str) -> str:
        return f"C{name[3:-4]}_nm" if name.startswith("phi") else name
    return {name: value for name, value in aberrations.items()
            if float(aberrations.get(magnitude(name), 0.0)) != 0.0}


class ShowPtycho(anywidget.AnyWidget):
    """Ptychographic reconstruction viewer, and direct ptychography aberration explorer over a ``quantem.gpu.DirectPtycho`` session.

    Parameters
    ----------
    data : array_like, quantem Dataset2d or quantem.gpu.DirectPtycho
        A 2D reconstruction (NumPy, torch, CuPy or anything ``np.asarray`` takes) or a prepared
        Direct ptychography session. A ``Dataset2d`` gives its array and its sampling (a keyword that
        contradicts it raises). A real array is the phase (rad); a complex array is the object wave, shown as
        its phase (``angle``) and amplitude (``abs``). Tensors and CuPy arrays stay on their
        device until the 2D result is drawn. Arrays need no quantem.gpu; the aberration controls then stay
        hidden and one line says what they need. A direct ptychography session, normally after
        ``ptycho.aberrations()`` (quantem's default fit, model-based and tilt-corrected), is
        reused for its resident bright-field evidence and exclusively driven while the widget is
        open. Joint fit, Refit and Fit Regions use the same default fifth-order fit on both
        CUDA and MPS. A supplied broad fit retains its higher-order correction.
    sampling : float or (row, col), optional
        Pixel size of an array in ``units``; it draws the scale bar and calibrates the FFT. A direct ptychography
        session carries its own scan sampling.
    units : str, optional
        Length unit of ``sampling``: any length of ``quantem.widget.units``
        (``"angstrom"``, ``"Å"``, ``"A"``, ``"nm"``, ``"pm"``, ...); angstrom when omitted.
    save_dir : str or Path, optional
        Direct ptychography session: where ``calibration.json`` and ``showptycho_stars.json`` are written; the
        working directory by default. A saved correction of the same
        acquisition, scan region and launch calibration is restored before the
        first preview.
    restore : bool, default True
        Restore the complete saved optics, tilt, depth and display settings
        when the acquisition and launch calibration match, also over a
        session that is already fitted (``ptycho.run()`` before
        ``ptycho.show()``); one line says what was restored and what it replaced.
        False keeps the supplied session even when a saved correction exists.
    compute_loss : bool, default False
        Calculate the optional reconstruction diagnostic when controls settle
        or calibration is saved. False keeps interactive review free of this
        extra score calculation. The aberration fitting objective is unchanged.
    show_fft : bool, default False
        Open with the FFT panel visible.
    method : {"phase_of_mean", "phase", "complex"} or None, optional
        Direct reconstruction method. None restores a compatible saved method,
        otherwise uses ``"phase_of_mean"``, conventional direct ptychography (the phase of the
        mean wave). An explicit method is never replaced by a save.
        Phase and complex are experimental CUDA methods using the same fitted
        optics; complex estimates log-amplitude and phase but displays phase.
    phase_damp, amplitude_damp : float, optional
        Independent positive stabilization strengths. Omitted values restore the
        saved strengths or use 0.03. Explicit values override saved strengths.
        Tikhonov damping of the direct least-squares solve. Amplitude damping
        is used only for complex reconstruction.
        These controls use a cached fixed-optics solve, not display smoothing.
    smoothness_inv_A : float, optional
        Frequency scale of the smoothness prior in inverse angstroms; 1.0 by
        default when no saved value exists. Explicit values override a save.
        Use 0 for a frequency-independent regularizer. Saved with
        the method and strengths so the reconstruction is reproducible.
    detector_weighting : {"uniform", "inverse_mean"}, optional
        Detector weighting for regularized methods, uniform by default.
        Omitted values restore the saved choice; explicit values override it.

    upsample : {"auto", 1, 2, 3, 4, 8}, optional
        Output multiplier along each scan axis. Auto stops at 4×; 8× is
        explicit only. Omitted, preserve a saved explicit choice or use Auto.
        An explicit value overrides saved sampling, retaining the joint fit.
    local_tilt : bool or dict, optional
        True adds region selection to the existing viewer. Fit Regions jointly
        optimizes aberrations, tilt and depth with the direct ptychography backend, then crops
        each region from its full-scan reconstruction. Detector geometry and
        rotation stay common. A dict can supply ``scan_shape`` and ``regions``
        (id, label and bounds in row-start, row-stop, col-start, col-stop order).
        Read selections, fits and phase arrays through ``regions``,
        ``region_fits`` and ``region_images`` in later notebook cells.

    Attributes
    ----------
    tilt_correction : bool
        Whether the preview includes the fitted sample tilt, True by default.
        False compares zero tilt at the same aberrations and depth spread.
        Saving retains the fitted tilt regardless of this display setting.

    A session opened from its master (``DirectPtycho.open("scan_master.h5", ...)``) records the file as its
    ``source_path``: the toolbar then offers Crop and Refit, which reloads only the selected
    scan region and fits it afresh, and every saved calibration and star names the file.
    """

    compute_loss = traitlets.Bool(False).tag(sync=True)

    local_tilt_selection_json = traitlets.Unicode("").tag(sync=True)
    local_tilt_active_region = traitlets.Unicode("").tag(sync=True)
    local_tilt_preview_json = traitlets.Unicode("").tag(sync=True)
    local_tilt_json = traitlets.Unicode("").tag(sync=True)
    local_tilt_request_json = traitlets.Unicode("").tag(sync=True)
    local_tilt_reply_json = traitlets.Unicode("").tag(sync=True)
    local_tilt_images_json = traitlets.Unicode("[]").tag(sync=True)
    local_tilt_bytes = traitlets.Bytes(b"").tag(sync=True)
    local_tilt_enabled = traitlets.Bool(False).tag(sync=True)
    local_tilt_max_columns = traitlets.Int(3, min=1, max=8).tag(sync=True)
    local_tilt_stale = traitlets.Bool(False).tag(sync=True)
    local_tilt_can_fit = traitlets.Bool(False).tag(sync=True)
    local_tilt_can_reconstruct = traitlets.Bool(False)

    _esm = pathlib.Path(__file__).parent.parent / "static" / "showptycho.js"

    # slider ranges: the physical span of the direct ptychography fit's default search
    C10_min_nm = traitlets.Float(-40.0).tag(sync=True)
    C10_max_nm = traitlets.Float(40.0).tag(sync=True)
    scan_detector_rotation_min_deg = traitlets.Float(-180.0).tag(sync=True)
    scan_detector_rotation_max_deg = traitlets.Float(180.0).tag(sync=True)
    # the physical scan-to-detector rotation as the slider holds it; files store it in [0, 360)
    scan_detector_rotation_deg = traitlets.Float(0.0).tag(sync=True)
    # Direct ptychography phase is ambiguous by sign; the user picks the convention matching the expected contrast.
    # Display only: never part of a numerical handoff.
    display_negate_phase = traitlets.Bool(False).tag(sync=True)
    # The phase panel's colormap and contrast window (rad), kept in step by the browser so Save
    # writes the image as shown; None before a browser has drawn it.
    cmap = traitlets.Unicode("inferno").tag(sync=True)
    vmin = traitlets.Float(None, allow_none=True).tag(sync=True)
    vmax = traitlets.Float(None, allow_none=True).tag(sync=True)
    # the fitted reference the Reset button returns to
    fitted_C10_nm = traitlets.Float(0.0).tag(sync=True)
    fitted_C12_nm = traitlets.Float(0.0).tag(sync=True)
    fitted_phi12_deg = traitlets.Float(0.0).tag(sync=True)
    fitted_loss = traitlets.Float(None, allow_none=True).tag(sync=True)
    # The kept fit's model agreement (higher is better) and its gain over standard direct ptychography, None before a fit.
    fitted_sample_agreement = traitlets.Float(None, allow_none=True).tag(sync=True)
    fitted_tilt_fit_gain = traitlets.Float(None, allow_none=True).tag(sync=True)
    fitted_scan_detector_rotation_deg = traitlets.Float(0.0).tag(sync=True)
    fitted_sample_json = traitlets.Unicode("{}").tag(sync=True)
    fitted_higher_order_aberrations_json = traitlets.Unicode("{}").tag(sync=True)
    phase_bytes = traitlets.Bytes(b"").tag(sync=True)
    phase_width = traitlets.Int(0).tag(sync=True)
    phase_height = traitlets.Int(0).tag(sync=True)
    result_json = traitlets.Unicode("").tag(sync=True)
    # an array: the pixel size [row, col] of its image in angstrom. A session: the scan step [row, col] in angstrom,
    # which the browser shares out over the output columns (upsample). [] when unknown or not that input.
    object_sampling_A = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    scan_sampling_A = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    # one-shot seed the browser reads at mount; the FFT switch takes over
    initial_fft_on = traitlets.Bool(False).tag(sync=True)
    # Direct ptychography alias-order output sampling, not image interpolation
    upsample = traitlets.Union(
        [traitlets.Int(), traitlets.Enum(["auto"])], default_value="auto",
    ).tag(sync=True)
    auto_upsample = traitlets.Int(1).tag(sync=True)
    auto_sampling_note = traitlets.Unicode("").tag(sync=True)
    sampling_limit_note = traitlets.Unicode("").tag(sync=True)
    upsample_available = traitlets.Bool(False).tag(sync=True)
    method = traitlets.Enum(["phase_of_mean", "phase", "complex"], default_value="phase_of_mean").tag(sync=True)
    phase_damp = traitlets.Float(0.03, min=0.0).tag(sync=True)
    amplitude_damp = traitlets.Float(0.03, min=0.0).tag(sync=True)
    smoothness_inv_A = traitlets.Float(1.0, min=0.0).tag(sync=True)
    detector_weighting = traitlets.Enum(["uniform", "inverse_mean"], default_value="uniform").tag(sync=True)
    regularization_available = traitlets.Bool(False).tag(sync=True)
    save_trigger = traitlets.Int(0).tag(sync=True)
    notes = traitlets.Unicode("").tag(sync=True)
    pin_json = traitlets.Unicode("").tag(sync=True)
    stars_path = traitlets.Unicode("").tag(sync=True)
    calibration_path = traitlets.Unicode("").tag(sync=True)
    calibration_saved_at = traitlets.Unicode("").tag(sync=True)
    calibration_status = traitlets.Unicode("").tag(sync=True)
    calibration_image_path = traitlets.Unicode("")
    # fit history for the trials strip: JSON list of {rank, C10_nm, C12_nm, phi12_deg, loss}, best first
    trials_json = traitlets.Unicode("").tag(sync=True)
    num_bright_field_pixels = traitlets.Int(0).tag(sync=True)
    # higher-order panel: {C21_nm, phi21_deg, ..., C30_nm, ..., C50_nm, ...}
    higher_order_aberrations_json = traitlets.Unicode("{}").tag(sync=True)
    # browser-side WebGPU direct ptychography source, set by the folder export
    webgpu_preview_enabled = traitlets.Bool(False).tag(sync=True)
    webgpu_cal_json = traitlets.Unicode("").tag(sync=True)
    webgpu_h5_source_json = traitlets.Unicode("").tag(sync=True)
    # browser direct ptychography status; only an exported folder runs direct ptychography in the browser (a live notebook asks Python)
    webgpu_preview_status = traitlets.Unicode("").tag(sync=True)
    webgpu_standalone = traitlets.Bool(False).tag(sync=True)
    # crop and refit: rebuilds direct ptychography from the raw scan region, not a display crop
    scan_rows = traitlets.Int(0).tag(sync=True)
    scan_cols = traitlets.Int(0).tag(sync=True)
    crop_refit_available = traitlets.Bool(False).tag(sync=True)
    crop_refit_status = traitlets.Unicode("").tag(sync=True)
    crop_refit_request_json = traitlets.Unicode("").tag(sync=True)
    # thick-sample direct ptychography: {"sample_tilt_mrad": [row, col], "depth_spread_nm"}; depth spread 0 is standard direct ptychography
    sample_json = traitlets.Unicode("{}").tag(sync=True)
    # Preview only: keep the fitted tilt when comparing against zero-tilt direct ptychography.
    tilt_correction = traitlets.Bool(True).tag(sync=True)
    sample_available = traitlets.Bool(False).tag(sync=True)
    sample_fit_request = traitlets.Int(0).tag(sync=True)
    sample_fit_status = traitlets.Unicode("").tag(sync=True)
    sample_fit_json = traitlets.Unicode("").tag(sync=True)
    # an array input has no session: the browser hides the aberration controls and shows session_note
    ptycho_session = traitlets.Bool(True).tag(sync=True)
    session_note = traitlets.Unicode("").tag(sync=True)
    # float32 amplitude of a complex object wave, row-major like phase_bytes; empty for a phase
    amplitude_bytes = traitlets.Bytes(b"").tag(sync=True)

    @property
    def ptycho_parameters(self) -> dict:
        """Displayed shared optics for subsequent direct ptychography calculations.

        Sample tilt and depth spread belong to individual fits, so they are
        excluded. Coefficients use nm and degrees; rotation is in degrees.

        Examples
        --------
        >>> parameters = viewer.ptycho_parameters
        >>> parameters["aberrations"]["C10_nm"]
        """
        if self._direct is None:
            return {}
        higher = json.loads(self.higher_order_aberrations_json or "{}")
        return {
            "aberrations": {**self._current(), **nonzero_higher_order(higher)},
            "scan_detector_rotation_deg": float(self.scan_detector_rotation_deg),
        }

    @property
    def source_path(self) -> str | None:
        """The direct ptychography session's acquisition file on this machine, or None for an array session.

        Held in memory for a handoff that loads the same data (myptycho's ``from_viewer``); the
        saved records name it portably instead (``source``: file name and relative path).
        """
        return getattr(self, "_source_path", None)

    def calibration(self, region: str | None = None, *, include_phase: bool = False) -> dict:
        """Snapshot a complete numerical model for downstream reconstruction.

        Parameters
        ----------
        region : str or None, optional
            Region label or stable ID. None selects the overview, independently
            of which controls are visible. Finish fitting and release sliders
            before taking a snapshot.
        include_phase : bool, default False
            Reconstruct the full phase at this model using the prepared session.
            The ``direct_phase`` entry is an independent NumPy array in radians
            covering the measured ``scan_shape`` times ``upsample``:
            positions quantem.gpu padded for its FFT are removed. No second direct ptychography
            session or aberration search is created.

        Returns
        -------
        dict
            Independent calibration record with the keys and units of the
            ``calibration.json`` Save writes; ``id`` and ``generated_at_utc``
            are None until saved. Regional records contain their own jointly fitted
            aberrations, scan-frame tilt, depth spread and full-acquisition scan bounds.
            ``region_fit["bounds"]`` remains relative to the direct ptychography session.
            Display contrast, zoom and the tilt-comparison toggle do not change
            this model. No file is written and no optimization runs. Without
            ``include_phase``, every entry is JSON serializable.

        Examples
        --------
        >>> model = viewer.calibration(region="Region 2")
        >>> model["sample_tilt_mrad"]
        """
        if self._direct is None:
            raise ValueError(
                "Open ShowPtycho on a direct ptychography session to reuse its calibration."
            )
        if self._region_busy:
            raise RuntimeError(
                "Regional fitting is running. Wait for completion before handoff."
            )
        fit = None
        if region is not None and self.method != "phase_of_mean":
            raise ValueError("Regional calibration currently supports method='phase_of_mean'; use the full-session regularized calibration.")
        if region is not None:
            matches = [
                fit for fit in self.region_fits if region in (fit["id"], fit["label"])
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Region {region!r} is not unique or fitted. Choose a current region label or ID."
                )
            fit = matches[0]
        state = self._direct.state()
        sample = self._sample() if self.sample_available else None
        aberrations = self.ptycho_parameters["aberrations"]
        # The overview reports the loss Save stores: the accepted model with its
        # fitted tilt, whatever the tilt-comparison view or the last drag showed.
        loss = None if fit is not None else self._accepted_loss(aberrations, sample)
        if fit is not None:
            aberrations = fit["aberrations"]
            sample = {
                "sample_tilt_mrad": fit["sample_tilt_mrad"],
                "depth_spread_nm": fit["depth_spread_nm"],
            }
        record = self._calibration_record(aberrations, sample, loss)
        if fit is not None:
            # Selectors use session coordinates; handoff uses acquisition coordinates.
            r0, r1, c0, c1 = fit["bounds"]
            row, _, col, _ = self._scan_region
            record.update(
                scan_region=[r0 + row, r1 + row, c0 + col, c1 + col], region_fit=fit
            )
        record = json.loads(json.dumps(record, allow_nan=False))
        if include_phase:
            phase, _ = self._direct.preview(
                record["aberrations"],
                sample_tilt_mrad=record["sample_tilt_mrad"] or (0.0, 0.0),
                depth_spread_nm=record["depth_spread_nm"] or 0.0,
                upsample=record["upsample"],
                **self._reconstruction_options(),
                compute_loss=False,
            )
            rows, cols = state["input_scan_shape"]
            record["direct_phase"] = self._direct.crop_native(
                phase, (0, rows, 0, cols), record["upsample"])
        return record

    def _region_reference(self) -> dict:
        """Shared optics and output sampling that regional images were computed at."""
        if self._direct is None:
            return {}
        return {**self.ptycho_parameters, "upsample": self._output_factor}

    def _on_region_optics_change(self, change) -> None:
        """Invalidate local results when their shared optics or output sampling change."""
        self.local_tilt_stale = self._region_reference() != self._region_fit_parameters

    @property
    def regions(self) -> list[dict]:
        """Current rectangles, with half-open ``[r0, r1, c0, c1]`` bounds.

        Read this property again after editing the widget. Returned records are
        independent copies; assigning a new list updates the visible selection.

        Examples
        --------
        >>> regions = viewer.regions
        >>> viewer.regions = regions
        """
        from quantem.widget.showptycho.local_tilt import _regions

        state = json.loads(self.local_tilt_json or "{}")
        if not state:
            return []
        selected = json.loads(self.local_tilt_selection_json or "null")
        return _regions(selected if selected is not None else state["regions"],
                        tuple(state["scan_shape"]))

    @regions.setter
    def regions(self, regions: list[dict]) -> None:
        from quantem.widget.showptycho.local_tilt import _regions

        state = json.loads(self.local_tilt_json or "{}")
        if not state:
            raise ValueError("Configure local_tilt when constructing ShowPtycho before selecting regions.")
        self.local_tilt_selection_json = json.dumps(
            _regions(regions, tuple(state["scan_shape"])), allow_nan=False)

    @property
    def region_fits(self) -> list[dict]:
        """Current direct ptychography fit records, including ``sample_tilt_mrad`` in (row, col).

        Raises
        ------
        ValueError
            Any selected rectangle has no matching fit. Use Fit Regions in the
            widget or call ``viewer.fit_regions()`` before reading these records.

        Examples
        --------
        >>> sample_tilt_mrad = [fit["sample_tilt_mrad"] for fit in viewer.region_fits]
        """
        if self.local_tilt_stale:
            raise ValueError("Shared optics changed (optics, rotation or output sampling). Fit Regions again before reading local fits.")
        state = json.loads(self.local_tilt_json or "{}")
        fits = []
        for region in self.regions:
            fit = next((f for f in state.get("fits", [])
                        if f["id"] == region["id"] and f["bounds"] == region["bounds"]), None)
            if fit is None:
                raise ValueError("Selection has unfitted regions. Click Fit Regions or call viewer.fit_regions().")
            fits.append(fit)
        return fits

    def _on_region_selection(self, change) -> None:
        """Remove deleted views without refitting the remaining regions."""
        state = json.loads(self.local_tilt_json or "{}")
        if not state:
            return
        previous = json.loads(change["old"] or "null")
        previous = state["regions"] if previous is None else previous
        selected = self.regions
        if len(selected) >= len(previous) or any(region not in previous for region in selected):
            return
        retained = {region["id"] for region in selected}
        state["regions"] = selected
        state["fits"] = [fit for fit in state.get("fits", []) if fit["id"] in retained]
        state["map"] = [fit for fit in state.get("map", []) if fit["id"] in retained]
        labels = {name for region in selected for name in (region["label"], f"{region['label']} · Direct")}
        metadata = json.loads(self.local_tilt_images_json)
        sampling = {entry["label"]: entry["sampling_A"] for entry in metadata}
        with self.hold_sync():
            self.set_region_images({name: image for name, image in self._region_images.items() if name in labels},
                                   sampling_A=sampling)
            self.local_tilt_json = json.dumps(state, allow_nan=False)
            self.local_tilt_reply_json = ""
            if self.local_tilt_active_region not in retained:
                self.local_tilt_active_region = ""

    @property
    def region_images(self) -> dict[str, np.ndarray]:
        """Copies of the current named phase arrays at their original dtype.

        Native direct ptychography results are cropped from full-scan reconstructions using
        each region's jointly fitted optics, tilt and depth. Custom callbacks
        may supply their own phases. Arrays precede float32 display transport.
        Changing the selection makes this property raise until images for that
        selection are supplied. Reread it after fitting or reconstruction.

        Examples
        --------
        >>> phases = viewer.region_images
        """
        if self.local_tilt_stale:
            raise ValueError("Shared optics changed (optics, rotation or output sampling). Fit Regions again before reading region images.")
        if self.regions != self._region_image_regions:
            raise ValueError("Region images belong to another selection. Fit the current regions first.")
        return {name: phase.copy() for name, phase in self._region_images.items()}

    def fit_regions(
        self, *, trials: int = 200,
    ) -> list[dict]:
        """Jointly fit optics, tilt and depth in the current rectangles.

        With a live direct ptychography session this uses QuantEM directly, keeps the overview
        calibration unchanged and displays crops of full-scan reconstructions.
        A custom callback may override the fitting for supplied image arrays.

        Parameters
        ----------
        trials : int, default 200
            Search trials of the single joint fit per region (seed 42), then
            quantem.gpu's exact-derivative refinement.

        The widget's Fit Regions button uses the trials of the last call.

        Returns
        -------
        list[dict]
            Current direct ptychography fit records with local tilt in (row, col) mrad.

        Examples
        --------
        >>> fits = viewer.fit_regions()
        """
        if int(trials) < 1:
            raise ValueError("Fit Regions needs at least one trial per seed.")
        self._region_fit_options = {"trials": int(trials)}
        self._run_region_callback("fit")
        return self.region_fits

    def reconstruct_regions(self) -> dict[str, np.ndarray]:
        """Reconstruct the current rectangles using the configured callback.

        Returns
        -------
        dict[str, numpy.ndarray]
            Named regional phases, including the reconstructed comparisons.

        Examples
        --------
        >>> images = viewer.reconstruct_regions()
        """
        self._run_region_callback("reconstruct")
        return self.region_images

    def _run_region_callback(self, action: str) -> None:
        if self._region_busy:
            raise RuntimeError("A regional calculation is running. Wait for it to complete.")
        # Use the same observer path as the live widget and propagate its error
        # to a notebook caller instead of silently returning previous results.
        regions = self.regions
        self.local_tilt_request_json = json.dumps({
            "id": uuid.uuid4().hex, "action": action, "regions": regions,
            "signature": json.dumps(regions, separators=(",", ":"), ensure_ascii=False),
        })
        reply = json.loads(self.local_tilt_reply_json)
        if reply["status"] == "error":
            raise RuntimeError(reply["message"])

    def set_region_images(self, images: dict, *, sampling_A: float | dict[str, float]) -> None:
        """Show named regional phases within this viewer.

        Parameters
        ----------
        images : dict[str, array_like]
            Named two-dimensional phase images, displayed in float32.
        sampling_A : float or dict[str, float]
            Image sampling in angstrom, common or keyed by image name.
            Per-image sampling preserves native grids in a comparison.

        Examples
        --------
        >>> viewer.set_region_images({"Region 1": phase}, sampling_A=0.2)
        """
        from quantem.widget.showptycho.local_tilt import _images
        metadata, pixels = _images(images, sampling_A)
        self._region_images = {name: np.array(value, copy=True) for name, value in images.items()}
        self._region_image_regions = self.regions
        with self.hold_sync():
            self.local_tilt_images_json = metadata
            self.local_tilt_bytes = pixels

    def _on_local_tilt_request(self, change):
        """Run the backend or supplied callback for the requested selection."""
        from quantem.widget.showptycho.local_tilt import _regions
        request = json.loads(change["new"] or "{}")
        if not request or self._region_busy:
            return
        request_id = request.get("id")
        reply = {"id": request_id, "signature": request.get("signature")}
        try:
            if self.method != "phase_of_mean":
                raise ValueError("Regional fitting currently supports method='phase_of_mean'; use a full-session review for regularization.")
            state = json.loads(self.local_tilt_json)
            regions = _regions(request["regions"], tuple(state["scan_shape"]))
            if not regions:
                raise ValueError("Draw a region before fitting.")
            action = request["action"]
            callback = self._region_callbacks.get(action)
            if callback is None:
                raise ValueError('This saved view has no live callback. Run the workspace cell to reconnect.')
            self._region_busy = True
            self.local_tilt_reply_json = json.dumps({**reply, "status": "running", "action": action})
            reference = self._region_reference()
            if callback == self._fit_native_regions:
                def progress(update: dict) -> None:
                    self.local_tilt_reply_json = json.dumps({
                        **reply, "status": "running", "action": action, **update})
                result = callback(regions, only_changed=request.get("only_changed", False),
                                  force_region=request.get("force_region"), progress=progress)
            else:
                result = callback(regions)
            if self._region_reference() != reference:
                raise ValueError("Shared optics changed during fitting. Fit Regions again at the current settings.")
            if result.get("regions") != regions:
                raise ValueError('Callback returned another region selection; retain the requested bounds.')
            state.update({k: v for k, v in result.items() if k not in ('images', 'sampling_A')})
            serialized_state = json.dumps(state, allow_nan=False)
            with self.hold_sync():
                if 'images' in result:
                    self.set_region_images(result['images'], sampling_A=result['sampling_A'])
                    self._region_image_regions = regions
                self._region_fit_parameters = reference
                self.local_tilt_stale = False
                self.local_tilt_json = serialized_state
                self.local_tilt_reply_json = json.dumps({**reply, "status": "complete", "action": action})
        except Exception as exc:
            self.local_tilt_reply_json = json.dumps({**reply, "status": "error", "message": str(exc)})
        finally:
            self._region_busy = False

    def __init__(
        self,
        data,
        *,
        sampling: float | tuple[float, float] | None = None,
        units: str | None = None,
        save_dir: str | pathlib.Path | None = None,
        restore: bool = True,
        show_fft: bool = False,
        compute_loss: bool = False,
        upsample: str | int | None = None,
        method: str | None = None,
        phase_damp: float | None = None,
        amplitude_damp: float | None = None,
        smoothness_inv_A: float | None = None,
        detector_weighting: str | None = None,
        local_tilt: bool | dict | None = None,
        fit_regions: Callable[[list[dict]], dict] | None = None,
        reconstruct_regions: Callable[[list[dict]], dict] | None = None,
    ) -> None:
        self._method_explicit = method is not None
        method = "phase_of_mean" if method is None else method
        is_array = not gpu_adapter.is_direct_ptycho(data)
        if method != "phase_of_mean" and is_array:
            raise ValueError("Reconstruction methods need a DirectPtycho session; an array is already reconstructed.")
        if method != "phase_of_mean" and (local_tilt or fit_regions or reconstruct_regions):
            raise ValueError("Regional fitting currently supports method='phase_of_mean'. Review regularized methods on the full session.")

        if upsample is not None:
            valid = (isinstance(upsample, str) and upsample == "auto") or (
                isinstance(upsample, (int, np.integer))
                and not isinstance(upsample, (bool, np.bool_))
                and upsample in (1, 2, 3, 4, 8)
            )
            if not valid:
                raise ValueError("upsample must be 'auto' or an integer from 1, 2, 3, 4, 8; use upsample='auto'.")
            if is_array:
                raise ValueError("upsample needs a direct ptychography session; an array is already reconstructed.")
        restore_regions = local_tilt is True
        if local_tilt is True:
            if is_array:
                raise ValueError("local_tilt=True needs a live direct ptychography session. Pass ShowPtycho(ptycho, local_tilt=True).")
            rows, cols = data.input_scan_shape
            local_tilt = {
                "scan_shape": [rows, cols],
                "regions": [{"id": "region-1", "label": "Region 1",
                             "bounds": [0, rows, 0, cols]}],
                "fits": [], "map": [],
            }
        elif local_tilt is False:
            local_tilt = None
        # an array's pixel size [row, col] in angstrom, read through the one unit table (the widget draws in angstrom)
        if is_array and sampling is None:
            # a unit stated with no step, for an image that records none, raises as in every image widget
            image_axes(None, units, data if core_adapter.is_dataset(data, ndim=2) else None)
        given = None if sampling is None or not is_array else axis_pair(
            per_axis(sampling, 2), per_axis("Å" if units is None else units, 2), ("length",), "image")
        if core_adapter.is_dataset(data, ndim=2):
            # a calibrated reconstruction is the source of its pixel size, axis by axis: the keyword may
            # repeat a recorded step and gives the step of an axis the dataset leaves open
            recorded = axis_pair(data.sampling, data.units, ("length",), "image")
            given = stated("sampling", None if given is None else given[1], None if recorded is None else recorded[1])
        elif given is not None:
            given = given[1]
        if is_array:
            data = device_array(data)  # a tensor or CuPy array stays on its GPU until the 2D result is drawn
        if not is_array and sampling is not None:
            raise ValueError("sampling describes an array; a direct ptychography session carries its own scan_sampling_A.")
        super().__init__()
        self.method = method
        self.compute_loss = compute_loss
        self._explicit_reconstruction_options = {
            name: value for name, value in {
                "phase_damp": phase_damp,
                "amplitude_damp": amplitude_damp,
                "smoothness_inv_A": smoothness_inv_A,
                "detector_weighting": detector_weighting,
            }.items() if value is not None
        }
        self.phase_damp = 0.03 if phase_damp is None else phase_damp
        self.amplitude_damp = 0.03 if amplitude_damp is None else amplitude_damp
        self.smoothness_inv_A = 1.0 if smoothness_inv_A is None else smoothness_inv_A
        self.detector_weighting = "uniform" if detector_weighting is None else detector_weighting
        self._launch_calibration = None
        self._launch_calibration_shown = False
        self._direct = None if is_array else data
        self._region_fit_parameters: dict = {}
        self._region_fit_options: dict = {"trials": 200}
        self._pinned: list[dict] = []
        self._region_callbacks = {"fit": fit_regions, "reconstruct": reconstruct_regions}
        self.local_tilt_can_fit = callable(fit_regions)
        self.local_tilt_can_reconstruct = callable(reconstruct_regions)
        self._region_busy = False
        self._region_images: dict[str, np.ndarray] = {}
        self._region_image_regions: list[dict] = []
        if local_tilt is not None:
            from quantem.widget.showptycho.local_tilt import _regions
            state = dict(local_tilt)
            state["regions"] = _regions(state["regions"], tuple(state["scan_shape"]))
            self.local_tilt_json = json.dumps(state, allow_nan=False)
        self.observe(self._on_local_tilt_request, names=["local_tilt_request_json"])
        self.observe(self._on_region_selection, names=["local_tilt_selection_json"])
        self._last_phase: np.ndarray | None = None
        self._last_result: dict = {}
        self._inflight_id = -1
        # Kernel-initiated redraws count down from -2, browser requests up from 1: the ids never collide.
        self._redraw_id = -1
        self.initial_fft_on = bool(show_fft)
        self.observe(self._on_flip_change, names=["display_negate_phase"])
        if is_array:
            # angle and abs run where the data lives (a CUDA or MPS tensor stays there until the 2D
            # result is copied for the browser), in the input's precision; the browser draws float32,
            # so wider input is rounded once and the largest change is printed, never silently
            if data.ndim != 2:
                needs = "" if gpu_adapter.available() else f" (a session needs quantem.gpu: {gpu_path_hint() or no_gpu_path()})"
                raise ValueError(f"ShowPtycho shows one 2D reconstruction or a quantem.gpu DirectPtycho session{needs}; "
                                 f"got an array of shape {tuple(data.shape)}. Pass one 2D slice.")
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
            if given is not None and None not in given:  # both steps known: one open axis leaves the image uncalibrated
                self.object_sampling_A = [float(step) for step in given]
            self.ptycho_session = False
            self.session_note = session_note()
            return
        self._direct = gpu_adapter.DirectSession(data)
        self._owned_crop_session = None  # the session Crop and Refit made, closed when the next crop replaces it
        if local_tilt is not None and fit_regions is None and reconstruct_regions is None:
            self._region_callbacks = {
                "fit": self._fit_native_regions, "reconstruct": self._fit_native_regions}
            self.local_tilt_can_fit = True
            self.local_tilt_can_reconstruct = True
        state = self._direct.state()
        self.regularization_available = state["backend"] == "cuda"
        self._check_method()
        self._source_path = state["source_path"]
        self._source_scan_shape = state["input_scan_shape"]
        self._scan_region = data.scan_region or (0, self._source_scan_shape[0], 0, self._source_scan_shape[1])
        save_root = pathlib.Path(save_dir) if save_dir else pathlib.Path.cwd()
        self._stars_path = save_root / STARS_FILENAME
        self._calibration_path = save_root / CALIBRATION_FILENAME
        self._calibration_image_path = save_root / CALIBRATION_IMAGE_FILENAME
        self._calibration_identity = self._direct.calibration_identity()
        self.stars_path = str(self._stars_path.resolve())
        self.upsample_available = state["backend"] in ("cuda", "mps")
        from quantem.gpu.optics.physics import direct_auto_upsample, direct_upsample_requirement

        try:
            self.auto_upsample = direct_auto_upsample(
                voltage_kV=state["voltage_kV"], semiangle_mrad=state["semiangle_mrad"],
                scan_sampling_A=state["scan_sampling_A"],
            )
            required = direct_upsample_requirement(
                voltage_kV=state["voltage_kV"], semiangle_mrad=state["semiangle_mrad"],
                scan_sampling_A=state["scan_sampling_A"],
            )
            if required > 4 + 1e-12:
                # Auto never goes above 4×; 8× stays an explicit choice.
                self.sampling_limit_note = f"Auto 4× (sampling supports 8×; nominal {required:.2f}×)"
        except ValueError as exc:
            # Invalid calibration still opens at native sampling.
            self.auto_sampling_note = (
                f"Automatic direct ptychography output sampling is unavailable: {exc} "
                "Showing native sampling (upsample=1); set upsample to 2, 3, 4 or 8 for finer output.")
            self.upsample = 1
            print(f"ShowPtycho: {self.auto_sampling_note}")
        self.sample_available = True  # CUDA and MPS, the direct ptychography backends, both fit the thick-sample model
        self.scan_sampling_A = state["scan_sampling_A"]
        restored = None
        if restore and self._calibration_path.is_file():
            restored = self._restore_calibration(state)
        if upsample is not None:
            # Override output density before the first preview, retaining the saved fit.
            self.upsample = upsample if isinstance(upsample, str) else int(upsample)
        self.crop_refit_available = bool(self._source_path and pathlib.Path(self._source_path).is_file())
        self.crop_refit_status = (
            "Ready to refit a native square scan crop." if self.crop_refit_available
            else "Crop refit needs a re-loadable master file."
        )
        self.on_msg(self._on_preview_message)
        self.observe(self._on_upsample, names=["upsample"])
        self.observe(self._on_save, names=["save_trigger"])
        self.observe(self._on_pin, names=["pin_json"])
        self.observe(self._on_rotation_change, names=["scan_detector_rotation_deg"])
        self.observe(self._on_panel_change, names=["higher_order_aberrations_json", "sample_json"])
        self.observe(self._on_panel_change, names=["tilt_correction"])
        self.observe(self._on_panel_change, names=[
            "method", "phase_damp", "amplitude_damp", "smoothness_inv_A", "detector_weighting"])
        self.observe(self._on_crop_refit_request, names=["crop_refit_request_json"])
        self.observe(self._on_sample_fit_request, names=["sample_fit_request"])
        self._adopt_session(initial=True)
        self._launch_calibration = data.calibration()

        self._region_fit_parameters = self._region_reference()
        if restore_regions and restored and restored.get("local_tilt"):
            self._restore_regions(restored["local_tilt"])
        self.observe(self._on_region_optics_change, names=[
            "result_json", "scan_detector_rotation_deg", "higher_order_aberrations_json",
            "upsample"])

    def _restore_calibration(self, state: dict) -> dict | None:
        """Restore one matching correction of the session's scan before registering observers."""
        from quantem.widget.showptycho.calibration import read_calibration

        def model(aberrations: dict, rotation_deg: float, tilt, depth_spread_nm) -> str:
            described = (f"C10 {aberrations.get('C10_nm', 0.0):.2f} nm, C12 {aberrations.get('C12_nm', 0.0):.2f} nm, "
                         f"phi12 {aberrations.get('phi12_deg', 0.0):.1f}°, rotation {wrap_degrees(rotation_deg):.1f}°")
            if depth_spread_nm:
                described += f", tilt ({tilt[0]:+.1f}, {tilt[1]:+.1f}) mrad, depth spread {depth_spread_nm:.1f} nm"
            return described

        supplied = (state["loss"] is not None or state["sample_tilt_mrad"] is not None
                    or any(state["aberrations"].values()))
        before = model(state["aberrations"], state["scan_detector_rotation_deg"],
                       state["sample_tilt_mrad"], state["depth_spread_nm"])
        try:
            saved = read_calibration(
                self._calibration_path, self._calibration_identity,
                self._scan_region, CALIBRATION_SCHEMA,
            )
            if saved["upsample_mode"] == "auto" and self.auto_sampling_note:
                raise ValueError(self.auto_sampling_note)
            # Output density is independent of the joint fit. Re-resolve saved
            # Auto under the current policy; retain explicit user choices.
            reconstruction = saved["reconstruction"]
            saved_method = reconstruction["method"]
            if self._method_explicit and saved_method != self.method:
                raise ValueError("the requested reconstruction method differs from the saved review")
            self._check_method(saved_method)
            self._direct.restore_calibration(saved)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.calibration_status = f"Saved fit not restored: {exc}. Use Joint fit, then Save."
            return
        self.method = saved_method
        for name in ("phase_damp", "amplitude_damp", "smoothness_inv_A", "detector_weighting"):
            if name in reconstruction:
                setattr(self, name, reconstruction[name])
        for name, value in self._explicit_reconstruction_options.items():
            setattr(self, name, value)
        self.upsample = "auto" if saved["upsample_mode"] == "auto" else saved["upsample"]
        self.display_negate_phase = saved["display_negate_phase"]
        self.tilt_correction = saved["display"]["tilt_correction"]
        self.local_tilt_max_columns = saved["display"].get("max_columns", 3)
        self.notes = saved.get("notes", "")
        self.calibration_path = str(self._calibration_path.resolve())
        self.calibration_saved_at = saved.get("generated_at_utc", "")
        rows, cols = self._source_scan_shape
        self.calibration_status = ("Restored saved full-field fit" if self._scan_region == (0, rows, 0, cols)
                                   else f"Restored saved fit of scan region {list(self._scan_region)}")
        self.sample_fit_status = self.calibration_status
        restored = model(saved["aberrations"], saved["scan_detector_rotation_deg"],
                         saved["sample_tilt_mrad"], saved["depth_spread_nm"])
        print(f"ShowPtycho: restored {self._calibration_path.name} saved {self.calibration_saved_at or 'earlier'} "
              f"({restored})" + (f", replacing the session's {before}" if supplied else "")
              + ". Pass restore=False to keep the session's values.")
        return saved

    def _restore_regions(self, saved: dict) -> None:
        """Rebuild saved regional views from the full scan without fitting again."""
        from quantem.widget.showptycho.local_tilt import _regions

        try:
            regions = _regions(saved["regions"], tuple(self._source_scan_shape))
            fits = [fit for fit in saved["fits"] if any(
                all(fit[key] == region[key] for key in ("id", "label", "bounds"))
                for region in regions
            )]
            images = {}
            for fit in fits:
                phase, _ = self._direct.preview(
                    fit["aberrations"], sample_tilt_mrad=fit["sample_tilt_mrad"],
                    depth_spread_nm=fit["depth_spread_nm"],
                    upsample=self._output_factor,
                    compute_loss=False,
                )
                images[f"{fit['label']} · Direct"] = self._direct.crop_native(
                    phase, fit["bounds"], self._output_factor)
            self.local_tilt_json = json.dumps({
                "scan_shape": list(self._source_scan_shape), "regions": regions,
                "fits": fits, "map": fits,
                "reference_parameters": self.ptycho_parameters,
            }, allow_nan=False)
            if images:
                self.set_region_images(images, sampling_A=self.scan_sampling_A[0] / self._output_factor)
            if len(images) == len(regions):
                self._region_image_regions = regions
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            self.calibration_status += f"; regional views not restored: {exc}"

    def _fit_native_regions(self, regions: list[dict], *, only_changed: bool = False,
                            force_region: str | None = None,
                            progress: Callable[[dict], None] | None = None) -> dict:
        """Use the current full-scan direct ptychography backend without notebook callbacks."""
        state = json.loads(self.local_tilt_json)
        previous = {fit["id"]: fit for fit in state.get("fits", [])}
        retained = {
            region["id"]: previous[region["id"]] for region in regions
            if only_changed and not self.local_tilt_stale
            and region["id"] != force_region and region["id"] in previous
            and previous[region["id"]]["bounds"] == region["bounds"]
            and previous[region["id"]]["label"] == region["label"]
            and f"{region['label']} · Direct" in self._region_images
        }
        changed = [region for region in regions if region["id"] not in retained]
        result = self._direct.fit_regions(
            changed, self.ptycho_parameters,
            upsample=self._output_factor,
            progress=progress, **self._region_fit_options)
        updated = {fit["id"]: fit for fit in result["fits"]}
        result["fits"] = [updated.get(region["id"], retained.get(region["id"])) for region in regions]
        result["images"].update({f"{fit['label']} · Direct": self._region_images[f"{fit['label']} · Direct"]
                                 for fit in retained.values()})
        result["images"] = {
            f"{region['label']} · Direct": result["images"][f"{region['label']} · Direct"]
            for region in regions
        }
        result.update(regions=regions, map=result["fits"], fitted_region_count=len(changed))
        return result

    def _preview_region(self, request: dict) -> None:
        """Tune one regional model without replacing overview pixels or optics."""
        reply = {"id": request["id"], "region_id": request["region_id"]}
        try:
            fit = next(fit for fit in self.region_fits if fit["id"] == request["region_id"])
            if fit["bounds"] != request["region_bounds"]:
                raise ValueError("Region moved. Refit it before adjusting its optics.")
            sample = request["sample"]
            higher = request.get("higher_order", {
                key: value for key, value in fit["aberrations"].items()
                if key not in {"C10_nm", "C12_nm", "phi12_deg"}
            })
            optics = {**higher,
                      **{name: float(request[name]) for name in ("C10_nm", "C12_nm", "phi12_deg")}}
            if not all(np.isfinite(v) for v in [*optics.values(), *sample["sample_tilt_mrad"], sample["depth_spread_nm"]]):
                raise ValueError("Use finite regional optical and sample parameters.")
            model = sample if sample["depth_spread_nm"] > 0 else {}
            phase, loss = self._direct.preview(
                optics, **model, upsample=self._output_factor,
                compute_loss=self.compute_loss and bool(request.get("committed")))
            crop = self._direct.crop_native(phase, fit["bounds"], self._output_factor)
            fitted = fit.get("fitted_parameters", {
                "aberrations": fit["aberrations"], "sample_tilt_mrad": fit["sample_tilt_mrad"],
                "depth_spread_nm": fit["depth_spread_nm"],
            })
            fit.update(aberrations=optics, sample_tilt_mrad=list(sample["sample_tilt_mrad"]),
                       depth_spread_nm=float(sample["depth_spread_nm"]), fitted_parameters=fitted)
            fit["adjusted"] = any(fit[key] != value for key, value in fitted.items())
            state = json.loads(self.local_tilt_json)
            state["fits"] = [fit if item["id"] == fit["id"] else item for item in state["fits"]]
            state["map"] = state["fits"]
            images = dict(self._region_images)
            images[f"{fit['label']} · Direct"] = crop
            with self.hold_sync():
                self.set_region_images(images, sampling_A=self.scan_sampling_A[0] / self._output_factor)
                self.local_tilt_json = json.dumps(state, allow_nan=False)
                self.local_tilt_preview_json = json.dumps({**reply, "loss": loss})
        except (ValueError, RuntimeError, StopIteration) as exc:
            self.local_tilt_preview_json = json.dumps({**reply, "error": str(exc) or "Fit the selected region first."})

    def _adopt_session(self, *, initial: bool, compute_loss: bool = True, restore_tilt: bool = False) -> None:
        """Publish the session's geometry and fit as the widget's reference, then reconstruct at that fit.

        Runs once at construction and again after a crop refit replaces the session. The rotation slider
        spans at least +-180 deg so negative microscope conventions need no mental +180; the C10 range
        widens to include an outlier fit so the reference is always reachable.
        """
        state = self._direct.state()
        rotation = state["scan_detector_rotation_deg"]
        fitted = state["aberrations"]
        self._direct.set_scan_detector_rotation(rotation)
        # Publish one complete model before any observer can request a preview.
        # This also covers initial sample adoption and crop-refit replacements.
        self._adopting = True
        try:
            with self.hold_sync(), self.hold_trait_notifications():
                if restore_tilt:
                    self.tilt_correction = True
                self.sample_json = json.dumps({
                    "sample_tilt_mrad": state["sample_tilt_mrad"] or [0.0, 0.0],
                    "depth_spread_nm": float(state["depth_spread_nm"] or 0.0),
                })
                self.fitted_sample_json = self.sample_json
                self.scan_rows, self.scan_cols = state["scan_shape"]
                if initial:
                    self.C10_min_nm, self.C10_max_nm = min(-30.0, fitted["C10_nm"]), max(30.0, fitted["C10_nm"])
                    self.scan_detector_rotation_min_deg, self.scan_detector_rotation_max_deg = min(-180.0, rotation), max(180.0, rotation)
                self.scan_detector_rotation_deg = rotation
                self.fitted_scan_detector_rotation_deg = rotation
                self.fitted_C10_nm, self.fitted_C12_nm, self.fitted_phi12_deg = fitted["C10_nm"], fitted["C12_nm"], fitted["phi12_deg"]
                self.fitted_loss = state["loss"]
                self.fitted_sample_agreement = state.get("sample_agreement")
                self.fitted_tilt_fit_gain = state.get("tilt_fit_gain")
                self.higher_order_aberrations_json = json.dumps({
                    key: value for key, value in fitted.items()
                    if key not in {"C10_nm", "C12_nm", "phi12_deg"}
                })
                self.fitted_higher_order_aberrations_json = self.higher_order_aberrations_json
                self.trials_json = self._trials_payload()
                self.num_bright_field_pixels = state["num_bright_field_pixels"]
        finally:
            self._adopting = False
        self._inflight_id = self._next_redraw_id()
        self._reconstruct(self._inflight_id, fitted, compute_loss=compute_loss)

    def _trials_payload(self, max_trials: int = 50) -> str:
        """The session's fit trials for the browser strip, capped so a 200-trial run stays light.

        Every trial is listed with the ``objective`` it was scored on: the default fit's model agreement (with its
        sample tilt and depth spread) or quantem.gpu's phase variance, for a session fitted with it. Losses rank
        only within one objective, so the trials are ordered by objective, in the order the fit ran them, and best
        first within each.
        """
        finite = [trial for trial in self._direct.trials() if finite_or_none(trial["loss"]) is not None]
        objectives = list(dict.fromkeys(trial["objective"] for trial in finite))
        ranked = sorted(finite, key=lambda trial: (objectives.index(trial["objective"]), float(trial["loss"])))
        payload = [{"rank": rank, **trial, "loss": float(trial["loss"])} for rank, trial in enumerate(ranked[:max_trials])]
        return json.dumps(payload) if payload else ""

    def _current(self) -> dict:
        """``{C10_nm, C12_nm, phi12_deg}`` of the displayed phase, the fitted reference before the first request."""
        fitted = {"C10_nm": self.fitted_C10_nm, "C12_nm": self.fitted_C12_nm, "phi12_deg": self.fitted_phi12_deg}
        return {name: float(self._last_result.get(name, value)) for name, value in fitted.items()}

    def _sample(self) -> dict | None:
        """The Sample panel as ``{sample_tilt_mrad, depth_spread_nm}``, or None for standard direct ptychography (depth spread 0)."""
        values = json.loads(self.sample_json or "{}")
        depth_spread_nm = float(values.get("depth_spread_nm", 0.0))
        if depth_spread_nm <= 0.0:
            return None
        tilt_row, tilt_col = values.get("sample_tilt_mrad", (0.0, 0.0))
        return {"sample_tilt_mrad": (float(tilt_row), float(tilt_col)), "depth_spread_nm": depth_spread_nm}

    def _check_method(self, method: str | None = None) -> None:
        """Keep experimental methods on their supported backend and full scan."""
        if (self.method if method is None else method) == "phase_of_mean":
            return
        if not self.regularization_available:
            raise ValueError("Regularized direct reconstruction currently requires CUDA; use method='phase_of_mean' on MPS.")
        if self.local_tilt_json:
            raise ValueError("Regional fitting currently supports method='phase_of_mean'; use a separate full-session review for regularization.")

    def _reconstruction_options(self) -> dict:
        """Pass only parameters used by the selected numerical method."""
        self._check_method()
        if self.method == "phase_of_mean":
            return {"method": "phase_of_mean"}
        options = {"method": self.method, "phase_damp": self.phase_damp,
                   "smoothness_inv_A": self.smoothness_inv_A, "detector_weighting": self.detector_weighting}
        if self.method == "complex":
            options["amplitude_damp"] = self.amplitude_damp
        return options

    def _reconstruct(self, request_id: int, aberrations: dict, *, compute_loss: bool) -> None:
        """Ask the session for the phase at ``aberrations`` (C10_nm, C12_nm, phi12_deg) and push it to the browser.

        The higher-order panel joins them under its own keys (``C21_nm``, ``phi21_deg``, ...); a non-zero
        one switches the session to its full chi kernel. Drags use the full bright-field set and skip
        the variance loss; committed requests compute it only when explicitly enabled. A stale request
        (another one was issued meanwhile) is dropped after the compute.
        """
        higher_order = json.loads(self.higher_order_aberrations_json or "{}")
        sample = self._sample() if self.sample_available else None
        if sample is not None and not self.tilt_correction:
            sample = {**sample, "sample_tilt_mrad": (0.0, 0.0)}
        model = {**higher_order, **aberrations}
        options = dict(upsample=self._output_factor, **self._reconstruction_options(), **(sample or {}))
        # One slider moving (no loss asked, so not a release): its drag window, the same image to a few
        # float32 ulps in 1-7 ms instead of 30-50. A drag frame that repeats the last values (the browser
        # coalesces moves) stays on that window. Releases, other settings and first frames stay exact.
        previous = getattr(self, "_drag_model", None)
        # the rotation is part of the key: a rotation drag is never a window drag, and never builds or drops windows
        options_key = json.dumps({**options, "rotation": self.scan_detector_rotation_deg}, sort_keys=True, default=str)
        same_options = options_key == getattr(self, "_drag_options_key", None)
        moved = [] if previous is None else [name for name in model if previous.get(name, model[name]) != model[name]]
        if not moved and same_options and getattr(self, "_drag_control", None):
            moved = [self._drag_control]
        dragging = not compute_loss and same_options and len(moved) == 1 and getattr(self._direct, "drag_windows", False)
        if dragging:
            phase, loss = self._direct.drag(moved[0], model, **options), None
        else:
            phase, loss = self._direct.preview(model, compute_loss=compute_loss and self.compute_loss, **options)
        rotating = self.scan_detector_rotation_deg != getattr(self, "_drag_rotation", self.scan_detector_rotation_deg)
        self._drag_model, self._drag_options_key = model, options_key
        self._drag_rotation = self.scan_detector_rotation_deg
        self._drag_control = moved[0] if dragging else None
        if request_id != self._inflight_id:
            return
        if self.display_negate_phase:
            phase = -phase
        self._last_phase = phase
        # optimizer precision: the browser formats labels, rounding here would perturb saved calibrations
        self._last_result = {"id": request_id, "C10_nm": float(aberrations["C10_nm"]), "C12_nm": float(aberrations["C12_nm"]),
                             "phi12_deg": float(aberrations["phi12_deg"]), "upsample": self._output_factor,
                             "tilt_correction": self.tilt_correction, "loss": loss,
                             **self._reconstruction_options()}
        # Deliver pixels, dimensions and completion together. The browser can
        # request the next frame immediately without a second comm round trip.
        with self.hold_sync():
            self.phase_height, self.phase_width = phase.shape
            self.phase_bytes = phase.tobytes()
            self.result_json = json.dumps(self._last_result)
        if dragging:
            # one window node per frame, after the frame has left
            self._direct.advance()
        elif (compute_loss or not (same_options or rotating)) and getattr(self._direct, "drag_windows", False):
            # a release, the first frame, a method switch or other settings (not a rotation drag frame): windows
            # along the slider just used and the three main ones build while the sliders rest, so the next grab
            # starts on window frames (no work once they exist for these inputs)
            controls = dict.fromkeys(moved + ["C10_nm", "C12_nm", "phi12_deg"])
            self._direct.prepare(controls, model, **options)
            self._schedule_window_build()

    def _schedule_window_build(self) -> None:
        """Build drag windows one node per IOLoop tick while the sliders are idle; a request runs between ticks."""
        from tornado.ioloop import IOLoop

        def build():
            self._window_build = not self._direct.advance()
            if self._window_build:
                IOLoop.current().call_later(0.0, build)

        if not getattr(self, "_window_build", False):
            self._window_build = True
            IOLoop.current().call_later(0.0, build)

    def _reconstruct_displayed(self, *, compute_loss: bool) -> None:
        """Reconstruct the displayed aberrations again, under a new request id, after a setting
        outside the three sliders (rotation, sampling, a panel) changed the phase."""
        self._inflight_id = self._next_redraw_id()
        self._reconstruct(self._inflight_id, self._current(), compute_loss=compute_loss)

    def _next_redraw_id(self) -> int:
        """A new id for a redraw the kernel starts itself: negative, so it never equals a browser request id."""
        self._redraw_id -= 1
        return self._redraw_id

    def _on_preview_message(self, widget, content: dict, buffers) -> None:
        """Receive a bounded preview command without the generic trait throttle."""
        if content.get("type") == "direct-preview":
            self._on_request(content["request"])

    def _on_request(self, request: dict) -> None:
        """A slider request from the browser {"id", "C10_nm", "C12_nm", "phi12_deg", "committed", ...}: reconstruct at
        its aberrations; committed requests add the loss."""
        if request.get("region_id") and request.get("method", self.method) != "phase_of_mean":
            self.local_tilt_preview_json = json.dumps({"id": request["id"], "region_id": request["region_id"],
                "error": "Regional previews support method='phase_of_mean'; use a full-session review for regularization."})
            return
        if request.get("region_id"):
            self._preview_region(request)
            return
        # One complete slider state gives one reconstruction. Direct notebook
        # edits to either panel trait retain their normal observer behavior.
        # A dragged strength came from the browser, which already shows it: it is not echoed back,
        # so a drag sends one message per frame (the phase) and no extra widget render.
        previous_lock = self._property_lock
        echo = {name: request[name] for name in _BROWSER_STRENGTHS if name in request}
        # A dragged rotation came from the browser too; it adopts the value itself on release.
        if request.get("scan_detector_rotation_deg") is not None:
            echo["scan_detector_rotation_deg"] = float(request["scan_detector_rotation_deg"])
        self._property_lock = {**previous_lock, **echo}
        self._adopting = True
        try:
            with self.hold_sync():
                for name in ("method", "phase_damp", "amplitude_damp", "smoothness_inv_A", "detector_weighting"):
                    if name in request:
                        setattr(self, name, request[name])
                if "sample" in request:
                    self.sample_json = json.dumps(request["sample"])
                if "higher_order" in request:
                    self.higher_order_aberrations_json = json.dumps(request["higher_order"])
                if "tilt_correction" in request:
                    self.tilt_correction = bool(request["tilt_correction"])
                rotation = request.get("scan_detector_rotation_deg")
                if rotation is not None and rotation != self.scan_detector_rotation_deg:
                    self.scan_detector_rotation_deg = float(rotation)
                    self._direct.set_scan_detector_rotation(float(rotation))
        finally:
            self._adopting = False
            self._property_lock = previous_lock
        if request["id"] > 0 and request["id"] < getattr(self, "_browser_id", 0):
            return  # an older browser request arriving late: a newer one already ran
        self._browser_id = request["id"]
        self._inflight_id = request["id"]
        try:
            self._reconstruct(request["id"], {name: request[name] for name in ("C10_nm", "C12_nm", "phi12_deg")},
                              compute_loss=request.get("committed", False))
        except (ValueError, RuntimeError) as exc:
            self.result_json = json.dumps({"id": request["id"], "error": str(exc)})

    @property
    def _output_factor(self) -> int:
        """Resolve Auto using the same support criterion as DirectPtycho.run."""
        return self.auto_upsample if self.upsample == "auto" else self.upsample

    def _on_upsample(self, change) -> None:
        """Output sampling changed: show the displayed aberrations at the new sampling."""
        self._reconstruct_displayed(compute_loss=False)

    @traitlets.validate("upsample")
    def _validate_upsample(self, proposal):
        """Resolve Auto and refuse output the active correction cannot produce."""
        requested = proposal["value"]
        if requested == "auto" and self.auto_sampling_note:
            raise ValueError(self.auto_sampling_note)
        factor = self.auto_upsample if requested == "auto" else requested
        if factor not in (1, 2, 3, 4, 8):
            raise ValueError("ShowPtycho upsample must be 'auto', 1, 2, 3, 4, or 8.")
        if factor > 1 and not self.upsample_available:
            raise ValueError("This session supports native sampling only; use upsample=1.")
        return requested

    def _on_rotation_change(self, change) -> None:
        """Re-index the full bright-field geometry at the new scan-detector rotation.

        A rotation slider drag arrives from the browser as trait updates and
        skips the variance loss like every other drag; releasing the slider
        sends a committed request, which computes it. Rotations set from
        Python still compute the loss at once.
        """
        if self._adopting:
            return
        self._direct.set_scan_detector_rotation(float(change["new"]))
        from_browser = change["name"] in self._property_lock
        self._reconstruct_displayed(compute_loss=not from_browser)

    def _on_panel_change(self, change) -> None:
        """Higher-order or Sample panel changed: reconstruct the displayed aberrations with it.

        The first adoption resets the panel before any phase exists; there is nothing to redraw then.
        """
        if self._adopting or self._last_phase is None:
            return
        self._reconstruct_displayed(compute_loss=True)

    def _on_flip_change(self, change) -> None:
        """Negate the displayed phase; no reconstruction, the sign is a display convention."""
        if self._last_phase is None:
            return
        self._last_phase = -self._last_phase
        self.phase_bytes = self._last_phase.tobytes()

    def _on_sample_fit_request(self, change) -> None:
        """Initialize optics, tilt and depth, then jointly refine through fifth order.

        Publish initial and final Auto phase without a duplicate browser request.
        C10 is the fitted mid-depth defocus.
        """
        if not change["new"]:
            return
        if not self.sample_available:
            self.sample_fit_status = "Tilt fit failed: needs a CUDA or MPS direct ptychography session."
            return
        self.sample_fit_status = "Fitting initial optics, tilt and depth..."
        def show_initial(result):
            self._adopt_session(initial=False, compute_loss=False, restore_tilt=True)
            self.sample_fit_status = "Initial phase ready · refining through fifth order..."
        try:
            result = self._direct.aberrations(
                search="standard",
                tilt=True, check_rotation=False, compare_standard=False, verbose=False,
                on_initial_fit=show_initial)
        except (RuntimeError, ValueError, NotImplementedError, MemoryError) as exc:
            self.sample_fit_status = f"Tilt fit failed: {exc}"
            return
        tilt_row, tilt_col = result["sample_tilt_mrad"]
        self._adopt_session(initial=False, compute_loss=False, restore_tilt=True)
        with self.hold_sync():
            self.fitted_C10_nm = result["aberrations"]["C10_nm"]
            self.fitted_C12_nm = result["aberrations"]["C12_nm"]
            self.fitted_phi12_deg = result["aberrations"]["phi12_deg"]
            self.fitted_loss = result["loss"]
            self.fitted_sample_json = json.dumps({
                "sample_tilt_mrad": [tilt_row, tilt_col],
                "depth_spread_nm": result["depth_spread_nm"],
            })
        propagator_tilt = self._direct.propagator_tilt_mrad((tilt_row, tilt_col), self.scan_detector_rotation_deg)
        self.sample_fit_json = json.dumps({
            **result["aberrations"], "sample_tilt_mrad": [tilt_row, tilt_col], "propagator_tilt_mrad": propagator_tilt,
            "depth_spread_nm": result["depth_spread_nm"], "tilt_fit_gain": result["tilt_fit_gain"],
            "phase_ready": True,
        })
        self.sample_fit_status = (
            f"Fifth-order fit: tilt ({tilt_row:+.1f}, {tilt_col:+.1f}) mrad · "
            f"depth spread {result['depth_spread_nm']:.1f} nm."
        )

    def _on_crop_refit_request(self, change) -> None:
        """Fit direct ptychography on one scan crop and make it the widget's session.

        ``quantem.gpu.DirectPtycho.select_region`` builds the crop session from the current session's evidence
        (borrowed resident counts, or the crop decoded from the source), with the same acquisition
        identity, preprocessing provenance, flip, detector calibration and bright-field disk. The
        browser's region is relative to the current crop; the new session's region is in full-scan
        positions, and the saved identity is refreshed to it. A crop is reconstruction input, not a
        display view: a rectangle must span at least 32 positions each way.
        """
        if not change["new"]:
            return
        try:
            request = json.loads(change["new"])
            region = request.get("scan_region")
            if not isinstance(region, (tuple, list)) or len(region) != 4:
                raise ValueError("scan_region must be (row_start, row_stop, col_start, col_stop)")
            trials = int(request.get("trials", 200))
            if trials < 1:
                raise ValueError("Direct ptychography refit needs at least one optimization trial.")
            if not self.crop_refit_available or not self._source_path:
                raise RuntimeError("Crop refit is unavailable: open the session from its master, DirectPtycho.open(<master.h5>, ...).")
            origin_row, _, origin_col, _ = self._scan_region
            row_start, row_stop, col_start, col_stop = (int(value) for value in region)
            region = (origin_row + row_start, origin_row + row_stop, origin_col + col_start, origin_col + col_stop)
            rows, cols = self._direct.state()["input_scan_shape"]
            if not (0 <= row_start < row_stop <= rows and 0 <= col_start < col_stop <= cols):
                raise ValueError(f"scan crop [{row_start}:{row_stop}, {col_start}:{col_stop}] is outside the {rows}x{cols} scan")
            if row_stop - row_start < MIN_CROP_SPAN or col_stop - col_start < MIN_CROP_SPAN:
                raise ValueError(
                    f"Direct ptychography refit crops must be at least {MIN_CROP_SPAN}x{MIN_CROP_SPAN}; got {row_stop - row_start}x{col_stop - col_start}."
                )
            self.crop_refit_status = f"Loading [{region[0]}:{region[1]}, {region[2]}:{region[3]}] and fitting direct ptychography ({trials} trials)..."
            # quantem.gpu.DirectPtycho.select_region keeps the acquisition identity, preprocessing provenance, flip,
            # detector calibration and the whole scan's bright-field disk; only the scan positions change
            rebuilt = self._direct.select_region(
                (row_start, row_stop, col_start, col_stop), aberrations=self._current(),
                scan_detector_rotation_deg=self.scan_detector_rotation_deg)
            try:
                rebuilt.aberrations(search="standard",
                                    tilt=True, check_rotation=False, trials=trials, verbose=False)
            except BaseException:
                rebuilt.direct.close()
                raise
        except (RuntimeError, ValueError, OSError, MemoryError) as exc:
            # a refused request, an unreadable file or a solver failure; a code bug still raises
            self.crop_refit_status = f"Crop refit failed: {exc}"
            return
        if self._owned_crop_session is not None:
            self._owned_crop_session.direct.close()  # an earlier crop this widget made; the caller's session stays open
        self._direct = self._owned_crop_session = rebuilt
        self._scan_region = region
        self._calibration_identity = self._direct.calibration_identity()
        self._adopt_session(initial=False)
        self.crop_refit_status = f"Refit complete: [{region[0]}:{region[1]}, {region[2]}:{region[3]}], {trials} trials."

    def _on_pin(self, change) -> None:
        """Pin, star, view or unpin a snapshot of the displayed aberrations.

        A pin captures the higher-order panel with it, so an old snapshot stays reproducible when the
        panel changes later; starring writes the stars file.
        """
        if not change["new"]:
            return
        event = json.loads(change["new"])
        action, pin_id = event.get("action"), event.get("id")
        if action == "pin":
            entry = {
                "id": pin_id,
                "aberrations": {**json.loads(self.higher_order_aberrations_json or "{}"),
                                **{name: float(event[name]) for name in ("C10_nm", "C12_nm", "phi12_deg")}},
                "scan_detector_rotation_deg": float(event.get("scan_detector_rotation_deg", self.scan_detector_rotation_deg)),
                "display_negate_phase": bool(event.get("display_negate_phase", self.display_negate_phase)), "loss": event.get("loss"),
                "generated_at_utc": utc_now(), "starred": False,
                "reconstruction": self._reconstruction_options(),
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
                        entry["generated_at_utc"] = utc_now()
            self._write_stars()
        elif action == "view":
            for entry in self._pinned:
                if entry["id"] == pin_id:
                    if "phase" in entry:
                        self._last_phase = entry["phase"].copy()
                        self.phase_height, self.phase_width = self._last_phase.shape
                        self.phase_bytes = self._last_phase.tobytes()
                    # the viewed snapshot becomes the displayed state, so Save and later slider moves start from it
                    self._last_result = {"id": pin_id, **{name: entry["aberrations"][name] for name in ("C10_nm", "C12_nm", "phi12_deg")},
                                         "loss": entry.get("loss")}
                    self.result_json = json.dumps(self._last_result)
        elif action == "unpin":
            starred = any(entry["id"] == pin_id and entry["starred"] for entry in self._pinned)
            self._pinned = [entry for entry in self._pinned if entry["id"] != pin_id]
            if starred:
                self._write_stars()

    def _calibration(self, aberrations: dict, scan_detector_rotation_deg: float, display_negate_phase: bool, loss) -> dict:
        """One direct ptychography calibration under the vocabulary's names: what ``calibration.json`` and every star hold.

        Aberrations are nm and degrees (``C10_nm``, ``C12_nm``, ``phi12_deg`` and the non-zero higher
        orders ``C21_nm`` / ``phi21_deg`` ...), the rotation is the physical angle in [0, 360) with
        ``scan_detector_rotation_flipped`` beside it, ``scan_sampling_A`` and ``detector_sampling_mrad`` are
        ``[row, col]``, and the optics are the session's. ``source`` names the acquisition without an
        absolute path (quantem.gpu's ``portable_source``): its file name, and its path relative to the
        folder of the files when the data lives under it, so the folder can be moved or shared whole. Both
        files are built here, so a reader of one reads the other.
        """
        from quantem.gpu.direct_ptycho.persistence import portable_source

        state = self._direct.state()
        return {
            "aberrations": {name: float(value) for name, value in aberrations.items()
                            # a higher-order term is kept with its azimuth only while its magnitude is non-zero
                            if name in ("C10_nm", "C12_nm", "phi12_deg") or name in nonzero_higher_order(aberrations)},
            "scan_detector_rotation_deg": wrap_degrees(scan_detector_rotation_deg),
            "scan_detector_rotation_flipped": state["scan_detector_rotation_flipped"],
            "display_negate_phase": bool(display_negate_phase), "voltage_kV": state["voltage_kV"], "semiangle_mrad": state["semiangle_mrad"],
            "scan_sampling_A": state["scan_sampling_A"], "detector_sampling_mrad": state["detector_sampling_mrad"],
            "loss": None if loss is None else float(loss),
            "reconstruction": self._reconstruction_options(),
            "source": portable_source(self._source_path, self._calibration_path.parent), "scan_region": list(self._scan_region),
        }

    def _calibration_record(self, aberrations: dict, sample: dict | None, loss) -> dict:
        """The one ``quantem.direct-calibration`` record: what ``calibration()`` returns and Save writes.

        ``_calibration``'s optics and geometry with the Sample panel beside them (``sample_tilt_mrad``
        in the scan frame, ``propagator_tilt_mrad`` along the detector axes, ``depth_spread_nm``), the
        output sampling, display settings, the measured ``scan_shape`` of ``scan_region``, the
        bright-field radius, the regional views and the acquisition identity. ``id`` and
        ``generated_at_utc`` are None until Save writes the record.
        """
        state = self._direct.state()
        local_tilt = None
        if self.local_tilt_json:
            try:
                regional_fits = self.region_fits
            except ValueError:
                regional_fits = []
            local_tilt = {"regions": self.regions, "fits": regional_fits, "scan_shape": list(self._source_scan_shape)}
        return {
            "schema": CALIBRATION_SCHEMA, "id": None, "generated_at_utc": None, "notes": self.notes,
            **self._calibration(aberrations, self.scan_detector_rotation_deg, self.display_negate_phase, loss),
            "sample_tilt_mrad": list(sample["sample_tilt_mrad"]) if sample else None,
            "propagator_tilt_mrad": self._direct.propagator_tilt_mrad(sample["sample_tilt_mrad"], self.scan_detector_rotation_deg)
            if sample else None,
            "depth_spread_nm": sample["depth_spread_nm"] if sample else None,
            "upsample": self._output_factor,
            "upsample_mode": "auto" if self.upsample == "auto" else "explicit",
            "reconstruction": self._reconstruction_options(),
            "display": {"tilt_correction": self.tilt_correction, "max_columns": self.local_tilt_max_columns},
            "scan_shape": list(state["input_scan_shape"]),
            "bright_field_radius_px": state["bright_field_radius_px"],
            "session_identity": self._calibration_identity,
            "local_tilt": local_tilt,
        }

    def _accepted_loss(self, aberrations: dict, sample: dict | None) -> float | None:
        """Phase-variance loss of the accepted model, the value Save stores.

        It includes the fitted sample tilt even while the view compares zero
        tilt. When ``compute_loss`` is enabled, it is computed afresh; otherwise
        the optional diagnostic remains ``None``. Preview is transient and does
        not replace the displayed image.
        """
        if not self.compute_loss:
            return None
        _, loss = self._direct.preview(
            aberrations, compute_loss=True,
            upsample=self._output_factor,
            **self._reconstruction_options(),
            **(sample or {}),
        )
        return loss

    def _write_stars(self) -> None:
        """Write the starred snapshots: ``{"schema", "stars": [calibration, ...]}``, each as ``_calibration`` builds it."""
        stars = [{"id": entry["id"], "generated_at_utc": entry["generated_at_utc"],
                  **self._calibration(entry["aberrations"], entry["scan_detector_rotation_deg"], entry["display_negate_phase"], entry.get("loss")),
                  "reconstruction": entry.get("reconstruction", {"method": "phase_of_mean"})}
                 for entry in self._pinned if entry["starred"]]
        write_json(self._stars_path, {"schema": STARS_SCHEMA, "stars": stars})

    def _on_save(self, change) -> None:
        """The Save button: ``save_calibration()`` once the first preview has drawn."""
        if self._direct is None or self._last_phase is None or not self.result_json:
            return
        self.save_calibration()

    def save_calibration(self) -> pathlib.Path:
        """Write the reviewed model to ``calibration.json`` and the displayed phase to
        ``calibration_phase.png`` in ``save_dir``, as the Save button does.

        The file is the record ``calibration()`` returns, with its ``id`` and
        ``generated_at_utc``: the displayed aberrations, the Sample panel, the
        rotation, the output sampling, the display settings and the
        acquisition identity a later session restores from. The session's
        aberrations are set to the saved ones, as known values, so
        ``ptycho.run()`` applies them. Display state (``state_dict``) is
        not written here.

        Returns
        -------
        pathlib.Path
            The written ``calibration.json``.

        Examples
        --------
        >>> path = viewer.save_calibration()
        """
        if self._direct is None:
            raise ValueError("save_calibration() writes a direct ptychography session's model; open ShowPtycho on a direct ptychography session.")
        if self._region_busy:
            raise RuntimeError("Regional fitting is running. Wait for completion before saving.")
        aberrations = self._current()
        sample = self._sample() if self.sample_available else None
        higher_order = json.loads(self.higher_order_aberrations_json or "{}")
        loss = self._accepted_loss({**higher_order, **aberrations}, sample)
        self._direct.store({**higher_order, **aberrations}, loss, sample=sample)
        payload = {**self._calibration_record({**higher_order, **aberrations}, sample, loss),
                   "id": uuid.uuid4().hex[:12], "generated_at_utc": utc_now()}
        write_json(self._calibration_path, payload)
        self.calibration_path = str(self._calibration_path.resolve())
        self.calibration_saved_at = payload["generated_at_utc"]
        self._write_phase_image()
        self.calibration_status = "Saved calibration and phase PNG"
        return self._calibration_path.resolve()

    def _write_phase_image(self) -> None:
        """Write the displayed phase as ``calibration_phase.png``, the reviewed image beside its calibration.

        ``rgb = colorize(contrast_uint8(phase, vmin, vmax) / 255, cmap)`` at the phase's native
        size, with the display sign, colormap and window the browser shows (``cmap``, ``vmin``,
        ``vmax``). Without a browser window it takes ShowPtycho's default 1-99 % percentile
        window. Without it a saved calibration has no record of the image it was judged on.
        """
        from PIL import Image

        phase = self._last_phase
        if self.vmin is not None and self.vmax is not None:
            low, high = self.vmin, self.vmax
        else:
            low, high = (float(value) for value in np.percentile(phase[np.isfinite(phase)], (1, 99)))
        pixels = colorize(contrast_uint8(phase, low, high) / 255.0, self.cmap)
        Image.fromarray(pixels, mode="RGB").save(self._calibration_image_path)
        self.calibration_image_path = str(self._calibration_image_path.resolve())

    def freeze(self):
        """Keep the displayed phase visible while detaching its live direct ptychography controls.

        The session's owner can then close it and release GPU data. This does
        not close a shared direct ptychography session or change the displayed scientific array.
        Contrast, zoom and FFT inspection remain available.

        Returns
        -------
        ShowPtycho
            This viewer, now showing a fixed phase array.

        Examples
        --------
        >>> viewer.freeze()
        >>> ptycho.close()
        """
        if self._direct is None:
            return self
        for callback, names in (
            (self._on_upsample, ["upsample"]),
            (self._on_save, ["save_trigger"]),
            (self._on_pin, ["pin_json"]),
            (self._on_rotation_change, ["scan_detector_rotation_deg"]),
            (self._on_panel_change, ["higher_order_aberrations_json", "sample_json"]),
            (self._on_panel_change, ["tilt_correction"]),
            (self._on_panel_change, ["method", "phase_damp", "amplitude_damp", "smoothness_inv_A", "detector_weighting"]),
            (self._on_crop_refit_request, ["crop_refit_request_json"]),
            (self._on_sample_fit_request, ["sample_fit_request"]),
            (self._on_region_optics_change, ["result_json", "scan_detector_rotation_deg",
                                             "higher_order_aberrations_json", "upsample"]),
        ):
            self.unobserve(callback, names=names)
        self.on_msg(self._on_preview_message, remove=True)
        if self._region_callbacks["fit"] == self._fit_native_regions:
            # Native regional fits need the session; a frozen view keeps its images only.
            self._region_callbacks = {"fit": None, "reconstruct": None}
            self.local_tilt_can_fit = self.local_tilt_can_reconstruct = False
        self.object_sampling_A = [step / self._output_factor for step in self.scan_sampling_A]
        self._direct = None
        self.ptycho_session = self.sample_available = self.upsample_available = False
        self.regularization_available = False
        self.crop_refit_available = False
        self.session_note = "Saved direct ptychography phase. Open a new direct ptychography review to edit aberrations."
        return self

    def export(self, path: str | pathlib.Path | None = None, *, title: str | None = None) -> pathlib.Path:
        """Write a browser folder that reviews this direct ptychography with no kernel: ``index.html``, exact bright-field
        counts under ``source/``, the calibration under ``snapshots/`` and a double-click ``ShowPtycho.command``.

        ``path`` defaults to ``<stem>_showptycho`` beside the source master (or ``showptycho_export`` in the
        working directory). The browser rebuilds the direct ptychography reducers on its GPU from the stored counts, so
        C10, C12, phi12, rotation and the BF count all move live without Python.
        An unchanged whole-scan QEM source is reused without re-encoding; other sessions export
        exact BF columns. Sources are hardlinked when possible, otherwise symlinked or copied.
        A symlinked folder still needs its external source; ``snapshots/manifest.json`` records the link.
        The browser's WebGPU FFT takes 1, 2, 4 or 8x: an automatic 3x is exported at 4x, and one line says so.
        """
        if self._direct is None:
            raise ValueError("export() writes a direct ptychography review folder from a session's bright-field counts; this widget shows an array.")
        if self.method != "phase_of_mean":
            raise ValueError("Regularized reconstruction needs a live CUDA session. Use save_calibration() to save its method and strengths; standalone browser export supports method='phase_of_mean' only.")
        return write_showptycho_folder(self, path, title)

    def _repr_mimebundle_(self, **kwargs):
        """Show the launch calibration once, immediately before the live widget."""
        if self._launch_calibration is not None and not self._launch_calibration_shown:
            from IPython.display import display

            table = self._launch_calibration[["Value", "Source"]].copy()
            display(table.rename_axis("Parameter").style.set_caption("DirectPtycho calibration"))

            self._launch_calibration_shown = True
        return super()._repr_mimebundle_(**kwargs)

    def __repr__(self) -> str:
        """What the widget shows (array kind, size, sampling) or, for a session, the pins and full BF count."""
        if self._direct is None:
            kind = "object wave" if self.amplitude_bytes else "phase"
            sampling = f"{self.object_sampling_A[-1]:g} Å/px" if self.object_sampling_A else "no sampling"
            return f"ShowPtycho({kind} {self.phase_height}x{self.phase_width}, {sampling})"
        return f"ShowPtycho({len(self._pinned)} pinned, {self.num_bright_field_pixels} bright-field pixels)"
