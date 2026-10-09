"""ShowDiffraction: interactive d-spacing analysis for 2D/3D diffraction patterns.

The widget holds the frame stack, the center and k calibration, the picked spots and
rings and the display traits; every measurement and search is a call into
:mod:`~quantem.widget.showdiffraction.lattice` with phases from
:mod:`~quantem.widget.showdiffraction.phases`.
"""

import csv
import gc
import json
import math
import pathlib
import re
from typing import Self

import anywidget
import numpy as np
import torch
import traitlets

from quantem.widget.adapters import core as core_adapter
from quantem.widget.counts import exact_sum
from quantem.widget.device import resolve_device
from quantem.widget.export import HtmlExportMixin, export_slug
from quantem.widget.showdiffraction import lattice
from quantem.widget.showdiffraction.phases import (
    PHASE_LIBRARY,
    Phase,
    library_phase,
    phase_from_entry,
)
from quantem.widget.state import StateFileMixin
from quantem.widget.utils.array import to_numpy
from quantem.widget.utils.display_filter import apply_display_filter
from quantem.widget.utils.state_io import unwrap_state_payload

BF_RADIUS_FRACTION = 0.125
# a significant peak this close to a reflection the calibrated phase predicts is recovered as a ring
RECOVERY_TOL_PX = 4.0
CALIBRATION_TRAITS = (
    "k_pixel_size", "k_calibrated", "calibration_source", "calibration_ref_d", "calibration_ref_radius",
    "calibration_rms_px",
)
MEASUREMENT_COLUMNS = (
    "id", "kind", "raw_row", "raw_col", "row", "col", "row_err", "col_err", "r_pixels",
    "r_pixels_err", "g_inv_angstrom", "g_inv_angstrom_err", "d_angstrom", "d_angstrom_err",
    "angle_deg", "angle_deg_err", "intensity", "fit_quality", "fwhm_px", "fwhm_inv_angstrom",
    "intensity_integrated", "hkl", "hkl_candidates", "note",
)


def measurement_record(kind: str, item: dict) -> dict:
    """One export row (the :data:`MEASUREMENT_COLUMNS`) for a spot or ring record."""
    return {
        "id": item.get("id"),
        "kind": kind,
        "raw_row": item.get("raw_row"),
        "raw_col": item.get("raw_col"),
        "row": item.get("row"),
        "col": item.get("col"),
        "row_err": item.get("row_err"),
        "col_err": item.get("col_err"),
        "r_pixels": item.get("r_pixels") if kind == "spot" else item.get("radius_px"),
        "r_pixels_err": item.get("r_pixels_err"),
        "g_inv_angstrom": item.get("g_magnitude"),
        "g_inv_angstrom_err": item.get("g_magnitude_err"),
        "d_angstrom": item.get("d_spacing"),
        "d_angstrom_err": item.get("d_spacing_err"),
        "angle_deg": item.get("angle_deg"),
        "angle_deg_err": item.get("angle_deg_err"),
        "intensity": item.get("intensity"),
        "fit_quality": item.get("fit_quality"),
        "fwhm_px": item.get("fwhm_px"),
        "fwhm_inv_angstrom": item.get("fwhm_inv_angstrom"),
        "intensity_integrated": item.get("intensity_integrated"),
        "hkl": item.get("hkl", ""),
        "hkl_candidates": "|".join(item.get("hkl_candidates") or []),
        "note": item.get("note", ""),
    }


def index_fields(candidate: dict | None) -> dict:
    """Indexing fields of a spot or ring record for a matched reflection (or none)."""
    if candidate is None:
        return {"hkl": "", "d_ref": None, "d_error": None}
    return {"hkl": candidate["hkl_str"], "d_ref": candidate["d"], "d_error": candidate["d_error"]}


def phase_match_line(phase: Phase, d_errors: list[float], n_rings: int) -> str:
    """The ``phase_match`` status for rings labelled from ``phase``: rings matched out of
    ``n_rings`` and the mean relative d error. Calibrate-from-phase and Index Rings both
    write it, so the browser shows one format whichever step labelled the rings."""
    return (
        f"{phase.name} ({phase.absences}): {len(d_errors)}/{n_rings} matched, "
        f"{100.0 * float(np.mean(d_errors)):.2f}% mean error"
    )


class ShowDiffraction(StateFileMixin, HtmlExportMixin, anywidget.AnyWidget):
    """
    Interactive d-spacing analysis for 2D/3D diffraction patterns.

    Pick Bragg spots and rings on the diffraction pattern to measure d-spacings,
    g-vectors, and inter-spot angles, with sub-pixel Gaussian refinement. Works with
    a single 2D pattern (SAED) or a 3D stack of patterns, and accepts NumPy arrays,
    PyTorch tensors, or ``Dataset2d`` / ``Dataset3d``. 4D input is not supported.

    Parameters
    ----------
    data : np.ndarray, torch.Tensor, Dataset2d or Dataset3d
        2D ``(det_rows, det_cols)`` single pattern or 3D
        ``(n_frames, det_rows, det_cols)`` stack of patterns. A dataset supplies the
        title and, when its last axis is in ``1/Å``, the k calibration.
    k_pixel_size : float, optional
        k-space sampling in 1/Å per pixel. Marks the pattern calibrated.
    center : tuple[float, float], optional
        (row, col) of the diffraction center in pixels. Defaults to the detector
        center, then auto-detected from the bright-field disk if also no radius.
    bf_radius : float, optional
        Bright-field disk radius in pixels. Defaults to 1/8 of the detector size.
    title : str, default ""
        Title displayed above the widget.
    panel_width_px : int, optional
        Initial diffraction canvas width in CSS pixels.
    offline : bool, default False
        Pack every frame into the page so an exported HTML scrubs without a kernel.
    device : str, default "auto"
        Where the frames are reduced: ``"auto"`` (CUDA, then MPS, then CPU;
        printed once), ``"cuda"``, ``"cuda:N"``, ``"mps"`` or ``"cpu"``. A
        tensor stays on its device unless this names another.
    remove_hot_pixels : bool, default False
        Zero integer counts above three times the 99.9th percentile when the
        frame maximum exceeds five times it (hot pixels on counting detectors).
        Off by default so the frames hold the recorded counts; when such pixels
        are present a one-line notice says how many.
    verbose : bool, default True
        Print the loaded shape and device.
    **kwargs
        Any synced trait, for example ``dp_scale_mode="linear"`` or
        ``show_controls=False``.

    Examples
    --------
    >>> import numpy as np
    >>> from quantem.widget import ShowDiffraction
    >>> ShowDiffraction(np.random.rand(256, 256))
    >>> ShowDiffraction(np.random.rand(20, 128, 128), k_pixel_size=0.012)
    """

    _esm = pathlib.Path(__file__).parent.parent / "static" / "showdiffraction.js"
    _LIST_STATE_FIELDS = {"spots", "rings", "custom_phases", "mask_regions"}
    _STATE_FIELDS = (
        "title", "frame_idx", "panel_width_px", "k_pixel_size", "k_calibrated", "center_row",
        "center_col", "bf_radius", "spots", "rings", "zone_axis", "phase_match", "show_hkl",
        "snap_enabled", "snap_radius", "spot_refine", "detect_denoise", "denoise", "center_mode",
        "calibration_source", "calibration_ref_d", "calibration_ref_radius", "calibration_rms_px",
        "ellipse_ratio", "ellipse_angle", "ellipse_corrected", "dp_colormap", "dp_scale_mode",
        "dp_invert", "dp_vmin_pct", "dp_vmax_pct", "show_title", "show_stats", "show_controls",
        "controls_collapsed", "show_profile", "profile_log", "profile_subtract_background",
        "phase_name", "custom_phases", "mask_regions", "show_mask", "profile_theta_min",
        "profile_theta_max", "show_azimuthal", "refine_method", "center_method",
        "identify_elements", "identify_custom_only",
    )
    _GEOMETRY_TRAITS = (
        "center_row", "center_col", "k_pixel_size", "k_calibrated", "ellipse_ratio",
        "ellipse_angle", "ellipse_corrected", "mask_regions",
    )
    _PROFILE_TRAITS = (
        "show_profile", "profile_subtract_background", "profile_theta_min", "profile_theta_max",
        "frame_idx",
    )

    title = traitlets.Unicode("").tag(sync=True)
    n_frames = traitlets.Int(1).tag(sync=True)
    frame_idx = traitlets.Int(0).tag(sync=True)
    det_rows = traitlets.Int(1).tag(sync=True)
    det_cols = traitlets.Int(1).tag(sync=True)
    frame_bytes = traitlets.Bytes(b"").tag(sync=True)
    offline_frames = traitlets.Bytes(b"").tag(sync=True)
    offline = traitlets.Bool(False).tag(sync=True)

    # Center and calibration
    center_row = traitlets.Float(0.0).tag(sync=True)
    center_col = traitlets.Float(0.0).tag(sync=True)
    bf_radius = traitlets.Float(0.0).tag(sync=True)
    k_pixel_size = traitlets.Float(0.0).tag(sync=True)
    k_calibrated = traitlets.Bool(False).tag(sync=True)
    center_mode = traitlets.Enum(("auto", "manual"), default_value="auto").tag(sync=True)
    calibration_source = traitlets.Unicode("none").tag(sync=True)
    calibration_ref_d = traitlets.Float(0.0).tag(sync=True)
    calibration_ref_radius = traitlets.Float(0.0).tag(sync=True)
    calibration_rms_px = traitlets.Float(0.0).tag(sync=True)
    refine_method = traitlets.Unicode("auto").tag(sync=True)
    center_method = traitlets.Unicode("").tag(sync=True)
    ellipse_ratio = traitlets.Float(1.0).tag(sync=True)
    ellipse_angle = traitlets.Float(0.0).tag(sync=True)
    ellipse_corrected = traitlets.Bool(False).tag(sync=True)

    # Spots and rings
    spots = traitlets.List(traitlets.Dict()).tag(sync=True)
    rings = traitlets.List(traitlets.Dict()).tag(sync=True)
    snap_enabled = traitlets.Bool(False).tag(sync=True)
    snap_radius = traitlets.Int(5).tag(sync=True)
    spot_refine = traitlets.Bool(True).tag(sync=True)
    detect_denoise = traitlets.Enum(("auto", "none", "gaussian", "anscombe"), default_value="auto").tag(sync=True)
    denoise = traitlets.Enum(("none", "gaussian", "anscombe", "nlm"), default_value="none").tag(sync=True)
    zone_axis = traitlets.Unicode("").tag(sync=True)
    phase_match = traitlets.Unicode("").tag(sync=True)
    show_hkl = traitlets.Bool(True).tag(sync=True)
    selected_ring_id = traitlets.Int(0).tag(sync=True)

    # Browser request: {"action": name, "args": [...], "seq": n}; Python answers in
    # analysis_status and clears it
    _request = traitlets.Dict().tag(sync=True)
    analysis_status = traitlets.Unicode("").tag(sync=True)
    _quality = traitlets.Dict().tag(sync=True)

    # Analysis mask
    mask_regions = traitlets.List(traitlets.Dict()).tag(sync=True)
    show_mask = traitlets.Bool(True).tag(sync=True)

    # Phases
    phase_name = traitlets.Unicode("").tag(sync=True)
    custom_phases = traitlets.List(traitlets.Dict()).tag(sync=True)
    _phase_library = traitlets.List(traitlets.Dict()).tag(sync=True)
    identify_elements = traitlets.Unicode("").tag(sync=True)
    identify_custom_only = traitlets.Bool(False).tag(sync=True)
    _identify_results = traitlets.List(traitlets.Dict()).tag(sync=True)

    # Display
    dp_colormap = traitlets.Unicode("inferno").tag(sync=True)
    dp_scale_mode = traitlets.Enum(("linear", "log", "sqrt"), default_value="log").tag(sync=True)
    dp_invert = traitlets.Bool(False).tag(sync=True)
    dp_vmin_pct = traitlets.Float(0.0).tag(sync=True)
    dp_vmax_pct = traitlets.Float(100.0).tag(sync=True)
    dp_stats = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0, 0.0, 0.0]).tag(sync=True)

    # Profiles
    show_profile = traitlets.Bool(False).tag(sync=True)
    profile_log = traitlets.Bool(True).tag(sync=True)
    profile_subtract_background = traitlets.Bool(False).tag(sync=True)
    profile_theta_min = traitlets.Float(0.0).tag(sync=True)
    profile_theta_max = traitlets.Float(360.0).tag(sync=True)
    _profile_data = traitlets.Bytes(b"").tag(sync=True)  # float32 radii then intensities
    show_azimuthal = traitlets.Bool(False).tag(sync=True)
    _azimuthal_data = traitlets.Bytes(b"").tag(sync=True)  # float32 angles then intensities

    # UI
    show_title = traitlets.Bool(True).tag(sync=True)
    show_stats = traitlets.Bool(True).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)
    controls_collapsed = traitlets.Bool(False).tag(sync=True)
    panel_width_px = traitlets.Int(384).tag(sync=True)

    @traitlets.validate("frame_idx")
    def _validate_frame_idx(self, proposal):
        """Clamp a new frame index into the stack: a saved state may index more frames than
        the current stack has."""
        return max(0, min(int(proposal["value"]), max(1, int(self.n_frames)) - 1))

    def __init__(
        self,
        data,
        *,
        k_pixel_size: float | None = None,
        center: tuple[float, float] | None = None,
        bf_radius: float | None = None,
        title: str = "",
        panel_width_px: int | None = None,
        offline: bool = False,
        device: str = "auto",
        remove_hot_pixels: bool = False,
        verbose: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._remove_hot_pixels = remove_hot_pixels
        calibration_source = "manual"
        if core_adapter.is_dataset(data, ndim=2) or core_adapter.is_dataset(data, ndim=3):
            if not title and data.name:
                title = str(data.name)
            if k_pixel_size is None and data.units[-1] in ("1/Å", "1/A"):
                k_pixel_size = float(data.sampling[-1])
                calibration_source = "metadata"
            data = core_adapter.as_array(data)
        self._device = torch.device(resolve_device(device, data))
        # frame count before a merge appended the combined pattern
        self._n_source_frames: int | None = None
        # geometry the spots, rings and profiles were last derived for (see _on_geometry_change)
        self._derived_geometry: list | None = None
        self._ingest(data)
        if k_pixel_size is not None and k_pixel_size > 0:
            self.k_pixel_size = float(k_pixel_size)
            self.k_calibrated = True
            self.calibration_source = calibration_source
        self.title = title
        if panel_width_px is not None:
            self.panel_width_px = int(panel_width_px)
        self.offline = offline
        self.center_row = float(center[0]) if center is not None else self.det_rows / 2
        self.center_col = float(center[1]) if center is not None else self.det_cols / 2
        self.bf_radius = (
            float(bf_radius) if bf_radius is not None else min(self.det_rows, self.det_cols) * BF_RADIUS_FRACTION
        )
        if center is None and bf_radius is None:
            self._auto_detect_center()
        self._update_frame()
        self._bake_offline_frames()
        self._phase_library = [{"name": name, **entry} for name, entry in PHASE_LIBRARY.items()]
        self.observe(self._update_frame, names=["frame_idx", "denoise"])
        self.observe(self._bake_offline_frames, names=["offline"])
        self.observe(self._on_geometry_change, names=list(self._GEOMETRY_TRAITS))
        self.observe(self._on_request, names=["_request"])
        self.observe(self._update_profile, names=list(self._PROFILE_TRAITS))
        self.observe(self._update_azimuthal, names=["show_azimuthal", "rings", "frame_idx"])
        self.observe(self._on_export_request_change, names=["export_request"])
        if verbose:
            frames = "frame" if self.n_frames == 1 else "frames"
            print(
                f"ShowDiffraction: {self.n_frames} {frames} {self.det_rows}x{self.det_cols} "
                f"({self._data.nelement() * 4 / 1e6:.1f} MB float32) on {self._device}"
            )

    # --- Tutorial API ---
    def detect_spots(
        self,
        max_spots: int | None = None,
        min_distance: int = 6,
        min_relative: float = 0.1,
        exclude_radius: float | None = None,
        noise_sigma: float = 5.0,
    ) -> Self:
        """Replace the spots with detected Bragg peaks, strongest first.

        Candidates come from the ``detect_denoise`` view; each is then picked on the raw
        frame like a click (Gaussian-refined when ``spot_refine``). ``min_relative`` is the
        contrast floor relative to the strongest peak, ``noise_sigma`` the shot-noise
        floor in robust sigma units; lower it on frames whose background structure
        inflates the estimate.
        """
        if exclude_radius is None:
            exclude_radius = max(self.bf_radius, 2.0 * min_distance)
        coords = lattice.detect_spot_coords(
            self._detection_frame(),
            center=(self.center_row, self.center_col),
            exclude_radius=exclude_radius,
            min_distance=min_distance,
            min_relative=min_relative,
            noise_sigma=noise_sigma,
            mask=self._analysis_mask(),
            max_spots=max_spots,
        )
        self.spots = []
        for row, col in coords:
            self._add_spot(float(row), float(col))
        return self

    def index_spots(self, phase: Phase, tol: float = 0.03, angle_tol: float = 3.0) -> Self:
        """Index the spots against ``phase`` and solve the zone axis.

        Needs a calibrated pattern and a lattice-based phase (``Phase.from_cubic`` or the
        full constructor): the zone axis comes from the inter-spot angles, which a
        d-spacing card does not have. See :func:`lattice.index_spot_vectors`.
        """
        self._require_calibrated()
        if phase.lattice is None:
            raise ValueError(
                "index_spots needs a lattice-based Phase (from_cubic / full constructor) "
                "for the inter-spot angle check; a d-spacing card has no angles"
            )
        d_values = [spot.get("d_spacing") for spot in self.spots]
        assignments, candidate_lists, zone = lattice.index_spot_vectors(
            phase, [self._spot_vector(spot) for spot in self.spots], d_values, tol=tol, angle_tol=angle_tol
        )
        self.spots = self._with_angles(
            [
                {**spot, "hkl_candidates": [candidate["hkl_str"] for candidate in candidates], **index_fields(chosen)}
                for spot, candidates, chosen in zip(self.spots, candidate_lists, assignments)
            ]
        )
        self.zone_axis = zone or ""
        if zone is not None:
            self.phase_match = lattice.match_report(phase, d_values, tol)
        return self

    def run_auto(
        self,
        phase: Phase | None = None,
        *,
        max_rings: int = 8,
        exclude_radius: float | None = None,
    ) -> Self:
        """Center finding, ring detection, ring fitting, k calibration from ``phase``
        (default: the selected ``phase_name``), recovery of predicted rings and indexing.

        ``max_rings`` keeps the most significant ring peaks (standard errors above the
        profile envelope, :func:`lattice.detect_ring_radii`). The calibration is verified
        scale-free, ring-radius ratios against the phase's d-spacing ratios
        (:func:`lattice.calibration_scale_from_phase`), and is not applied when the rings
        do not decide it. ``analysis_status`` reports every failed step and the rings the
        phase does not explain; it is empty on a clean result.
        """
        phase = phase or self._selected_phase()
        problems = []
        self.phase_match = ""
        self._auto_detect_center(refine=True)
        candidates = self._detect_rings(max_rings=max_rings, exclude_radius=exclude_radius)
        if not self.rings:
            problems.append("ring detection failed (no rings found)")
        else:
            self._fit_rings()
            if all(ring.get("fit_quality") is None for ring in self.rings):
                problems.append("ring fit failed")
        if phase is None and self.phase_name:
            problems.append(f'calibration failed (phase "{self.phase_name}" not found)')
        if phase is not None and self.rings:
            calibration = {name: getattr(self, name) for name in CALIBRATION_TRAITS}
            try:
                self._calibrate_phase(phase)
                # significant peaks past the max_rings budget that sit on a reflection the
                # verified scale predicts
                predicted = [1.0 / (reflection["d"] * self.k_pixel_size) for reflection in phase.reflections()]
                held = [ring["radius_px"] for ring in self.rings]
                recovered = [
                    radius for radius in candidates[max_rings:]
                    if any(abs(radius - predicted_radius) <= RECOVERY_TOL_PX for predicted_radius in predicted)
                    and all(abs(radius - held_radius) > RECOVERY_TOL_PX for held_radius in held)
                ]
                for radius in recovered:
                    self._add_ring(radius)
                if recovered:
                    self._fit_rings()
                    self._calibrate_phase(phase)
            except ValueError as exc:
                # an unverified scale is not applied: the calibration from before Auto stays
                problems.append(f"calibration failed ({exc})")
                for name, value in calibration.items():
                    setattr(self, name, value)
                self.rings = [{**ring, "hkl_candidates": [], "radius_resid_px": None, **index_fields(None)} for ring in self.rings]
                self.phase_match = ""
            unexplained = [ring["radius_px"] for ring in self.rings if not ring.get("hkl")]
            if unexplained and len(unexplained) < len(self.rings):
                radii = ", ".join(f"{radius:.1f}" for radius in sorted(unexplained))
                problems.append(
                    f"{phase.name} explains {len(self.rings) - len(unexplained)} of {len(self.rings)} rings; "
                    f"not at r = {radii} px (another phase or an artefact?)"
                )
        self.analysis_status = "Auto: " + ", ".join(problems) if problems else ""
        return self

    def identify_phase(self, database, tol: float = 0.03) -> list[dict]:
        """Rank an explicit list of candidate phases against the measured d-spacings.

        This is the primary verification workflow: build the candidates you expect
        (:func:`~quantem.widget.library_phase`, :meth:`Phase.from_cubic`,
        :meth:`Phase.from_dspacings`) and rank only those. Use :meth:`search_phases`
        when you have no candidates in mind.
        """
        self._require_calibrated()
        phases = list(database)
        reports = lattice.rank_phases(self._observed_d(), phases, tol, max(len(phases), 1))
        self._identify_results = reports[:10]
        self.phase_match = lattice.identify_summary(reports)
        return reports

    def search_phases(
        self,
        *,
        tol: float = 0.03,
        elements=None,
        custom_only: bool | None = None,
        top_n: int = 10,
    ) -> list[dict]:
        """Rank the built-in library plus the custom phases against the measured d-spacings.

        ``elements`` (default: the ``identify_elements`` trait, for example ``"Fe, O"``)
        keeps only phases whose formula uses those elements; with ``custom_only``
        (default: the ``identify_custom_only`` trait) the library is skipped.
        """
        self._require_calibrated()
        observed = self._observed_d()
        if custom_only is None:
            custom_only = self.identify_custom_only
        allowed = elements if elements is not None else self.identify_elements
        if isinstance(allowed, str):
            allowed = re.split(r"[,\s]+", allowed.strip())
        # an empty field ("" splits to [""]) means no element filter, not an empty element set
        allowed = {symbol.strip().capitalize() for symbol in allowed if symbol.strip()} or None
        candidates = [] if custom_only else [library_phase(name) for name in PHASE_LIBRARY]
        for entry in self.custom_phases:
            # a half-edited custom phase in the menu must not abort the search
            try:
                candidates.append(phase_from_entry(entry))
            except (KeyError, ValueError, TypeError):
                continue
        if custom_only and not candidates:
            raise ValueError("no candidate phases; add custom phases first")
        if allowed is not None:
            candidates = [
                phase
                for phase in candidates
                if not set(re.findall(r"[A-Z][a-z]?", phase.name)) - allowed
            ]
        reports = lattice.rank_phases(observed, candidates, tol, top_n)
        self._identify_results = reports[:10]
        self.phase_match = lattice.identify_summary(reports)
        return reports

    @classmethod
    def measurements_from_state(cls, state, path: str | pathlib.Path | None = None):
        """Measurement rows (:data:`MEASUREMENT_COLUMNS`) from a saved state file or dict;
        with ``path`` they are written as CSV (or JSON with the calibration metadata)."""
        if isinstance(state, (str, pathlib.Path)):
            state = unwrap_state_payload(
                json.loads(pathlib.Path(state).read_text()),
                require_envelope=True,
                expected_widget="ShowDiffraction",
            )
        else:
            state = unwrap_state_payload(state, expected_widget="ShowDiffraction")
        records = [measurement_record("spot", spot) for spot in state.get("spots", [])]
        records += [measurement_record("ring", ring) for ring in state.get("rings", [])]
        if path is None:
            return records
        path = pathlib.Path(path)
        if path.suffix.lower() == ".json":
            metadata = {
                "widget_name": "ShowDiffraction",
                "center_row": state.get("center_row"),
                "center_col": state.get("center_col"),
                "center_method": state.get("center_method", ""),
                "k_pixel_size_inv_angstrom_per_px": state.get("k_pixel_size"),
                "calibrated": bool(state.get("k_calibrated")),
                "calibration_source": state.get("calibration_source", "none"),
                "calibration_ref_d_angstrom": state.get("calibration_ref_d", 0.0),
                "calibration_ref_radius_px": state.get("calibration_ref_radius", 0.0),
                "mask_regions": state.get("mask_regions", []),
                "background_subtracted": bool(state.get("profile_subtract_background")),
            }
            path.write_text(json.dumps({"metadata": metadata, "measurements": records}, indent=2))
        else:
            with open(path, "w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=MEASUREMENT_COLUMNS)
                writer.writeheader()
                writer.writerows(records)
        return path

    def export_html(
        self, path: str | pathlib.Path | None = None, *, title: str | None = None
    ) -> pathlib.Path:
        """Write a standalone HTML viewer with exact float32 frames."""
        export_path = pathlib.Path(path) if path is not None else self._default_html_export_path()
        # write_widget_html already adds the standalone viewport and module paths
        self._write_html_export(export_path, title=title)
        size_mb = export_path.stat().st_size / (1024 * 1024)
        self.export_status = f"Exported {export_path.name} ({size_mb:.1f} MB, full float32)"
        return export_path

    def state_dict(self) -> dict:
        """The persistable widget state (calibration, picks, display) as a plain dict."""
        return {
            field: list(getattr(self, field)) if field in self._LIST_STATE_FIELDS else getattr(self, field)
            for field in self._STATE_FIELDS
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore widget state from a dict; unknown keys are ignored."""
        # measurement records last, so the geometry restores do not resample them
        for key in self._STATE_FIELDS:
            if key in state and key not in ("spots", "rings", "frame_idx"):
                setattr(self, key, state[key])
        if "frame_idx" in state:
            requested = int(state["frame_idx"])
            self.frame_idx = requested
            if self.frame_idx != requested:
                self.analysis_status = (
                    f"Saved frame_idx {requested} clamped to {self.frame_idx}: "
                    f"data has {self.n_frames} frames"
                )
        for key in ("spots", "rings"):
            if key in state:
                setattr(self, key, state[key])

    def summary(self) -> None:
        """Print calibration, center, spots, rings, zone axis and phase match."""
        lines = [self.title or "ShowDiffraction"]
        lines.append(f"Frames:   {self.n_frames} (showing #{self.frame_idx})")
        k_info = f"{self.k_pixel_size:.4f} 1/Å/px" if self.k_calibrated else "uncalibrated"
        lines.append(f"Detector: {self.det_rows}x{self.det_cols} ({k_info})")
        if self.k_calibrated:
            source = {"from_phase": "phase", "from_ring": "ring", "from_spot": "spot"}.get(
                self.calibration_source, self.calibration_source
            )
            calibration_line = f"Calibration: {source}"
            if self.calibration_ref_d > 0:
                calibration_line += f" (d={self.calibration_ref_d:.3f} Å at r={self.calibration_ref_radius:.1f} px)"
            elif self.calibration_source == "from_phase":
                calibration_line += f" (rms {self.calibration_rms_px:.2f} px)"
            lines.append(calibration_line)
        if self.ellipse_ratio != 1.0:
            state = "corrected" if self.ellipse_corrected else "not corrected"
            lines.append(f"Ellipse:  a/b={self.ellipse_ratio:.3f} at {self.ellipse_angle:.1f}° ({state})")
        lines.append(f"Center:   ({self.center_row:.1f}, {self.center_col:.1f})  BF r={self.bf_radius:.1f} px")
        lines.append(f"Spots:    {len(self.spots)}")
        for spot in self.spots[:5]:
            if spot.get("d_spacing"):
                d_err = spot.get("d_spacing_err")
                d_text = f"{spot['d_spacing']:.3f}±{d_err:.3f} Å" if d_err else f"{spot['d_spacing']:.3f} Å"
                label = f"d={d_text}"
            else:
                label = f"r={spot['r_pixels']:.1f} px"
            angle = f"  angle={spot['angle_deg']:.1f}°" if spot.get("angle_deg") is not None else ""
            hkl = f"  {spot['hkl']}" if spot.get("hkl") else ""
            lines.append(f"  #{spot['id']} ({spot['row']:.1f}, {spot['col']:.1f}) {label}{angle}{hkl}")
        if len(self.spots) > 5:
            lines.append(f"  ... +{len(self.spots) - 5} more")
        lines.append(f"Rings:    {len(self.rings)}")
        if self.zone_axis:
            lines.append(f"Zone:     {self.zone_axis}")
        if self.phase_match:
            lines.append(f"Phase:    {self.phase_match}")
        lines.append(f"Display:  {self.dp_colormap} | {self.dp_scale_mode}")
        if self.snap_enabled:
            lines.append(f"Snap:     radius={self.snap_radius}")
        print("\n".join(lines))

    def __repr__(self) -> str:
        """Shape, calibration, frame and spot count on one line, for a bare widget name in a cell."""
        k_info = f", k_pixel_size={self.k_pixel_size} 1/Å" if self.k_calibrated else ""
        title_info = f", title='{self.title}'" if self.title else ""
        spots_info = f", spots={len(self.spots)}" if self.spots else ""
        return (
            f"ShowDiffraction(shape=({self.n_frames}, {self.det_rows}, {self.det_cols}){k_info}, "
            f"frame={self.frame_idx}/{self.n_frames}{spots_info}{title_info})"
        )

    def free(self) -> None:
        """Release the frame stack. ``del widget`` alone does not free it because the
        traitlets observers pin the widget."""
        self._data = None
        gc.collect()
        if self._device.type == "mps":
            torch.mps.empty_cache()
        elif self._device.type == "cuda":
            torch.cuda.empty_cache()

    # --- Frames ---
    def _ingest(self, data) -> None:
        """Store ``data`` as the float32 frame stack on the widget's device and set the shape traits.

        Non-finite values become 0 so one NaN cannot blank the display range or a fit. Integer
        (counting-detector) input is checked for hot pixels: when the maximum exceeds five times
        the 99.9th percentile, pixels above three times it are counted, and zeroed only with
        ``remove_hot_pixels``. A stack too large for one MPS tensor stays on the CPU.
        """
        array = to_numpy(data)
        is_integer = np.issubdtype(array.dtype, np.integer)
        # one float32 stack serves display and analysis; integer counts up to 2**24 stay exact
        array = np.nan_to_num(array.astype(np.float32), copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if array.size > 2**31 - 1 and self._device.type == "mps":
            # one MPS tensor holds at most 2**31 - 1 elements
            print(f"ShowDiffraction: {array.size} values exceed the 2**31 - 1 elements one MPS tensor holds; "
                  "the frames stay on the CPU.")
            self._device = torch.device("cpu")
        if is_integer:
            # hot pixels on counting detectors; sparse frames (99.9th percentile 0) are left alone
            global_max = float(array.max())
            percentile_99_9 = float(np.percentile(array, 99.9))
            if percentile_99_9 > 0 and global_max > percentile_99_9 * 5:
                threshold = percentile_99_9 * 3
                hot_pixels = array > threshold
                if self._remove_hot_pixels:
                    array[hot_pixels] = 0
                    print(f"ShowDiffraction: zeroed {int(hot_pixels.sum())} hot pixels above {threshold:.0f} counts "
                          "(3x the 99.9th percentile); pass remove_hot_pixels=False to keep the recorded counts.")
                else:
                    print(f"ShowDiffraction: {int(hot_pixels.sum())} pixels exceed {threshold:.0f} counts (3x the 99.9th "
                          "percentile, likely hot pixels); pass remove_hot_pixels=True to zero them.")
        if array.ndim == 2:
            array = array[None, ...]
        elif array.ndim == 4:
            raise ValueError("ShowDiffraction is for 2D/3D diffraction patterns; 4D input is not supported.")
        elif array.ndim != 3:
            raise ValueError(f"Expected a 2D or 3D array, got {array.ndim}D")
        self._data = torch.from_numpy(np.ascontiguousarray(array)).to(self._device)
        self.n_frames = int(array.shape[0])
        self.det_rows = int(array.shape[1])
        self.det_cols = int(array.shape[2])

    def _frame_index(self, frame_idx: int | None = None) -> int:
        """The displayed frame, or ``frame_idx`` clamped into the stack: a restored spot or
        ring record may index a frame this stack does not have."""
        return self.frame_idx if frame_idx is None else max(0, min(int(frame_idx), self.n_frames - 1))

    def _frame(self, frame_idx: int | None = None) -> np.ndarray:
        """One frame (see :meth:`_frame_index`) as a NumPy copy."""
        # a copy: on the CPU .numpy() shares the stored stack, which no caller may edit
        return self._data[self._frame_index(frame_idx)].cpu().numpy().copy()

    def _detection_frame(self) -> np.ndarray:
        """The displayed frame denoised per ``detect_denoise``; detection runs on it,
        every measurement on the raw frame so smoothing cannot bias positions."""
        frame = self._frame()
        mode = lattice.pick_detect_denoise(frame) if self.detect_denoise == "auto" else self.detect_denoise
        return frame if mode == "none" else apply_display_filter(frame, mode=mode, sigma=2.0)

    def _update_frame(self, change=None) -> None:
        """``frame_idx`` or ``denoise`` changed: send the displayed frame, display-denoised, and its stats."""
        frame = self._frame()
        if self.denoise != "none":
            frame = apply_display_filter(frame, mode=self.denoise, sigma=2.0)
        self.dp_stats = [float(frame.mean()), float(frame.min()), float(frame.max()), float(frame.std())]
        self.frame_bytes = frame.tobytes()

    def _bake_offline_frames(self, change=None) -> None:
        """``offline`` changed: pack every float32 frame into the page so an exported stack scrubs
        without a kernel; a single frame already travels in ``frame_bytes``."""
        if self.offline and self.n_frames > 1:
            self.offline_frames = self._data.cpu().numpy().tobytes()
        else:
            self.offline_frames = b""

    def _analysis_mask(self) -> np.ndarray | None:
        """Excluded pixels from ``mask_regions`` (beam stop, detector gaps), or None; every
        profile, fit and detection skips them so the hardware does not read as a ring or spot."""
        return lattice.build_analysis_mask(
            (self.det_rows, self.det_cols), self.mask_regions, (self.center_row, self.center_col)
        )

    def _radial_profile(
        self,
        *,
        frame: np.ndarray | None = None,
        frame_idx: int | None = None,
        angular_range: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Radii, azimuthal mean and its standard error of ``frame`` (default: frame
        ``frame_idx``, else the displayed one) about the widget's center, with its mask and
        ellipse correction, so every ring measurement sees the same geometry. A stored frame
        is reduced where the stack lives, without a copy back."""
        return lattice.radial_profile_px(
            self._data[self._frame_index(frame_idx)] if frame is None else frame,
            center=(self.center_row, self.center_col),
            mask=self._analysis_mask(),
            angular_range=angular_range,
            ellipse_ratio=self.ellipse_ratio,
            ellipse_angle=self.ellipse_angle,
            ellipse_corrected=self.ellipse_corrected,
        )

    def _radial_background(self, radii_px: np.ndarray, intensity: np.ndarray) -> np.ndarray:
        """Power-law background under the displayed frame's radial profile ``(radii_px,
        intensity)``, excluding the central beam and a window of one FWHM (6 px when unfitted)
        around each ring. The caller passes the profile it already holds."""
        peak_windows = [
            (ring["radius_px"] - (ring.get("fwhm_px") or 6.0), ring["radius_px"] + (ring.get("fwhm_px") or 6.0))
            for ring in self.rings
        ]
        return lattice.fit_radial_background(
            radii_px, intensity, peak_windows=peak_windows, exclude_radius=self.bf_radius
        )

    def _update_profile(self, change=None, profile: tuple | None = None) -> None:
        """Profile panel, sector, background switch or frame changed: send the radial profile.

        ``profile`` is the displayed frame's full-circle profile when the caller (a geometry
        change) already measured it, so a center click profiles each frame once."""
        if not self.show_profile:
            self._profile_data = b""
            return
        sector = (self.profile_theta_min, self.profile_theta_max)
        if profile is None and (sector == (0.0, 360.0) or self.profile_subtract_background):
            profile = self._radial_profile()
        radii, intensity, _ = profile if sector == (0.0, 360.0) else self._radial_profile(angular_range=sector)
        if self.profile_subtract_background:
            try:
                # the background is fitted under the full-circle profile, also for a sector
                intensity = intensity - self._radial_background(*profile[:2])
            except ValueError as exc:
                self.analysis_status = f"Background subtract failed: {exc}"
        # the browser reads float32
        self._profile_data = np.concatenate([radii, intensity]).astype(np.float32).tobytes()

    def _outer_ring_radius(self) -> float:
        """Radius of the outermost ring: the azimuthal profile, the ellipse fit and the merge
        QC use it because a given distortion or misalignment moves the largest ring the most."""
        if not self.rings:
            raise ValueError("no ring to analyze; call detect_rings / add_ring first")
        return float(max(ring["radius_px"] for ring in self.rings))

    def _ring_half_width(self, radius_px: float) -> float:
        """Annulus half-width that keeps neighbouring rings out.

        Half the gap to the nearest other ring, clamped to 3 to 6 px; a lone ring gets
        ``max(6, radius / 4)``.
        """
        gaps = [abs(radius_px - ring["radius_px"]) for ring in self.rings if ring["radius_px"] != radius_px]
        return min(6.0, max(3.0, min(gaps) / 2.0)) if gaps else max(6.0, 0.25 * radius_px)

    def _azimuthal_profile(self) -> tuple[np.ndarray, np.ndarray]:
        """I(theta) around the outermost ring."""
        radius = self._outer_ring_radius()
        return lattice.azimuthal_profile_from_frame(
            self._frame(),
            center=(self.center_row, self.center_col),
            radius_px=radius,
            half_width=self._ring_half_width(radius),
            n_theta=180,
            mask=self._analysis_mask(),
            ellipse_ratio=self.ellipse_ratio,
            ellipse_angle=self.ellipse_angle,
            ellipse_corrected=self.ellipse_corrected,
        )

    def _update_azimuthal(self, change=None) -> None:
        """Azimuthal panel, rings or frame changed: send I(theta) around the outermost ring."""
        if not self.show_azimuthal:
            self._azimuthal_data = b""
            return
        try:
            # the browser reads float32
            self._azimuthal_data = np.concatenate(self._azimuthal_profile()).astype(np.float32).tobytes()
        except ValueError as exc:
            self.analysis_status = f"Azimuthal profile failed: {exc}"
            self._azimuthal_data = b""

    # --- Center ---
    def _auto_detect_center(self, refine: bool = False) -> None:
        """Center and radius of the bright-field disk from the summed stack: the centroid
        of the pixels above mean + std, then the connected component under it."""
        summed = self._data.sum(dim=0)
        bright = summed > summed.mean() + summed.std()
        bright_count = int(exact_sum(bright.flatten(), dim=0))  # torch counts bools in int64; Apple MPS needs int32 blocks
        if bright_count == 0:
            return
        row_coords = torch.arange(self.det_rows, device=self._device, dtype=torch.float32)[:, None]
        col_coords = torch.arange(self.det_cols, device=self._device, dtype=torch.float32)[None, :]
        with self.hold_trait_notifications():
            self.center_row = float((row_coords * bright).sum() / bright_count)
            self.center_col = float((col_coords * bright).sum() / bright_count)
        self.bf_radius = lattice.central_beam_radius(bright.cpu().numpy(), self.center_row, self.center_col)
        self.center_mode = "auto"
        if refine:
            self._refine_center("symmetry")

    def _refine_center(self, method: str | None = None) -> str:
        """Refine Center action: re-estimate the center on the detection view with ``method``
        (default ``refine_method``) and return the status line."""
        picked = lattice.pick_center(
            self._detection_frame(),
            method=method or self.refine_method,
            mask=self._analysis_mask(),
            guess=(self.center_row, self.center_col),
        )
        with self.hold_trait_notifications():
            self.center_row, self.center_col = picked["row"], picked["col"]
        self.center_mode = "auto"
        self.center_method = picked["method"]
        return f"Center ({self.center_row:.1f}, {self.center_col:.1f}) via {self.center_method}"

    def _on_geometry_change(self, change=None) -> None:
        """Center, calibration, ellipse or mask changed: re-derive every measurement.

        The page sends a center click as ``center_row`` and ``center_col`` in one message, and
        the analysis steps set related traits under ``hold_trait_notifications``; traitlets
        then notifies once per trait with every value already set, so the first notification
        derives the final geometry and the others find it derived. Each source frame's radial
        profile is measured once and shared by its rings and the profile panel.
        """
        geometry = [getattr(self, name) for name in self._GEOMETRY_TRAITS]
        if geometry == self._derived_geometry:
            return
        self._derived_geometry = geometry
        profiles = {}
        rings_changed = False
        if self.spots:
            self.spots = self._with_angles(
                [
                    {
                        **spot,
                        **self._spot_info(
                            spot["row"], spot["col"], spot.get("row_err", 0.0), spot.get("col_err", 0.0), spot.get("frame_idx")
                        ),
                    }
                    for spot in self.spots
                ]
            )
        if self.rings:
            calibrated = self.k_calibrated and self.k_pixel_size > 0
            rings = []
            for record in self.rings:
                frame_idx = self._frame_index(record.get("frame_idx"))
                if frame_idx not in profiles:
                    profiles[frame_idx] = self._radial_profile(frame_idx=frame_idx)
                ring = {**record, **self._ring_info(record["radius_px"], profiles[frame_idx])}
                if ring.get("fwhm_px") is not None:
                    ring["fwhm_inv_angstrom"] = ring["fwhm_px"] * self.k_pixel_size if calibrated else None
                rings.append(ring)
            rings_changed = rings != self.rings
            self.rings = rings
        self._update_profile(profile=profiles.get(self.frame_idx))
        if not rings_changed:
            # new ring records re-measure the azimuthal profile through their own observer
            self._update_azimuthal()

    # --- Spots ---
    def _spot_vector(self, spot: dict) -> tuple[float, float]:
        """Spot g-vector in pixels from the center, ellipse-corrected like its radius, so the
        inter-spot angles and the zone-axis solve see undistorted geometry."""
        return lattice.corrected_vector(
            spot["row"] - self.center_row,
            spot["col"] - self.center_col,
            ellipse_ratio=self.ellipse_ratio,
            ellipse_angle=self.ellipse_angle,
            ellipse_corrected=self.ellipse_corrected,
        )

    def _spot_info(
        self, row: float, col: float, row_err: float = 0.0, col_err: float = 0.0, frame_idx: int | None = None
    ) -> dict:
        """Radius, g, d (with propagated errors) and intensity of a spot position.

        The radius error is the position error projected on the radial direction,
        ``hypot(d_row / r * row_err, d_col / r * col_err)``; g = r k and d = 1 / g carry the
        same relative error.
        """
        d_row = row - self.center_row
        d_col = col - self.center_col
        r_pixels = self._radius_px(row, col)
        if r_pixels > 0:
            r_pixels_err = math.hypot((d_row / r_pixels) * row_err, (d_col / r_pixels) * col_err)
        else:
            r_pixels_err = math.hypot(row_err, col_err)
        pixel_row = max(0, min(self.det_rows - 1, round(row)))
        pixel_col = max(0, min(self.det_cols - 1, round(col)))
        if self.k_calibrated and self.k_pixel_size > 0 and r_pixels > 0:
            g_magnitude = r_pixels * self.k_pixel_size
            d_spacing = 1.0 / g_magnitude
            # relative radius error propagates unchanged to g and d
            relative_err = r_pixels_err / r_pixels
            g_err = g_magnitude * relative_err
            d_err = d_spacing * relative_err
        else:
            g_magnitude = d_spacing = g_err = d_err = None
        return {
            "d_spacing": d_spacing,
            "d_spacing_err": d_err,
            "g_magnitude": g_magnitude,
            "g_magnitude_err": g_err,
            "r_pixels": r_pixels,
            "r_pixels_err": r_pixels_err,
            "intensity": float(self._data[self._frame_index(frame_idx), pixel_row, pixel_col]),
        }

    def _radius_px(self, row: float, col: float) -> float:
        """Distance of a detector position from the center in pixels, ellipse-corrected when
        ``ellipse_corrected``; spot measurements and calibrate-from-spot must measure alike."""
        return float(
            lattice.corrected_radius(
                row - self.center_row,
                col - self.center_col,
                ellipse_ratio=self.ellipse_ratio,
                ellipse_angle=self.ellipse_angle,
                ellipse_corrected=self.ellipse_corrected,
            )
        )

    def _with_angles(self, spots: list) -> list:
        """Angle of every spot to the first one, with the propagated position error."""
        if not spots:
            return spots
        reference = spots[0]
        ref_vector = self._spot_vector(reference)
        ref_radius = math.hypot(*ref_vector)
        ref_error = math.hypot(reference.get("row_err", 0.0), reference.get("col_err", 0.0))
        with_angles = []
        for spot in spots:
            vector = self._spot_vector(spot)
            radius = math.hypot(*vector)
            if ref_radius > 0 and radius > 0:
                angle = lattice.vector_angle(ref_vector, vector)
                spot_error = math.hypot(spot.get("row_err", 0.0), spot.get("col_err", 0.0))
                angle_err = math.degrees(math.hypot(spot_error / radius, ref_error / ref_radius))
            else:
                angle = angle_err = None
            with_angles.append({**spot, "angle_deg": angle, "angle_deg_err": angle_err})
        return with_angles

    def _pick_spot(self, row: float, col: float) -> dict:
        """Position and measurement fields of a pick on the displayed frame: Gaussian-fitted
        when ``spot_refine``, snapped to the local maximum when ``snap_enabled``, else exact."""
        raw_row, raw_col = float(row), float(col)
        row_err = col_err = 0.0
        fit_quality = None
        frame = self._frame()
        if self.spot_refine:
            fit = lattice.fit_gaussian_spot(frame, raw_row, raw_col, half_window=self.snap_radius)
            if fit is not None:
                row, col = fit["row"], fit["col"]
                row_err, col_err = fit["row_err"], fit["col_err"]
                fit_quality = fit["fit_quality"]
        elif self.snap_enabled:
            pixel_row, pixel_col = round(row), round(col)
            row_start, col_start = max(0, pixel_row - self.snap_radius), max(0, pixel_col - self.snap_radius)
            region = frame[row_start : pixel_row + self.snap_radius + 1, col_start : pixel_col + self.snap_radius + 1]
            if region.size:
                peak = np.unravel_index(region.argmax(), region.shape)
                row, col = float(row_start + peak[0]), float(col_start + peak[1])
        return {
            "row": float(row),
            "col": float(col),
            "raw_row": raw_row,
            "raw_col": raw_col,
            "row_err": float(row_err),
            "col_err": float(col_err),
            "fit_quality": fit_quality,
            "frame_idx": int(self.frame_idx),
            "hkl": "",
            "hkl_candidates": [],
            "d_ref": None,
            "d_error": None,
            "note": "",
            **self._spot_info(row, col, row_err=row_err, col_err=col_err),
        }

    def _add_spot(self, row: float, col: float) -> None:
        """Add Spot action (a click): pick a new spot with the next free id."""
        spot = {
            "id": max((int(spot["id"]) for spot in self.spots), default=0) + 1,
            "angle_deg": None,
            "angle_deg_err": None,
            **self._pick_spot(row, col),
        }
        self.spots = self._with_angles(list(self.spots) + [spot])

    def _move_spot(self, spot_id: int, row: float, col: float) -> None:
        """Move Spot action (a drag): re-pick spot ``spot_id`` at the new position."""
        spots = [{**spot, **self._pick_spot(row, col)} if spot["id"] == spot_id else spot for spot in self.spots]
        self.spots = self._with_angles(spots)

    def _remove_spot(self, spot_id: int) -> None:
        """Remove Spot action; the angles are re-referenced when the first spot goes."""
        remaining = [spot for spot in self.spots if spot["id"] != spot_id]
        if len(remaining) != len(self.spots):
            self.spots = self._with_angles(remaining)

    def _undo_spot(self) -> None:
        """Undo Spot action: drop the newest spot (the angle reference stays)."""
        self.spots = list(self.spots[:-1])

    def _clear_spots(self) -> None:
        """Clear Spots action."""
        self.spots = []

    # --- Rings ---
    def _ring_info(self, radius_px: float, profile: tuple) -> dict:
        """g, d and intensity of a ring radius, read from ``profile``, the radial profile of
        the ring's source frame (the caller measures it once for all its rings)."""
        if self.k_calibrated and self.k_pixel_size > 0:
            g_magnitude = float(radius_px) * self.k_pixel_size
            d_spacing = 1.0 / g_magnitude if g_magnitude > 0 else None
        else:
            g_magnitude = d_spacing = None
        radii_px, intensity, _ = profile
        ring_intensity = float(intensity[np.argmin(np.abs(radii_px - radius_px))]) if radii_px.size else 0.0
        return {
            "radius_px": float(radius_px),
            "g_magnitude": g_magnitude,
            "d_spacing": d_spacing,
            "intensity": ring_intensity,
        }

    def _add_ring(self, radius_px: float, profile: tuple | None = None) -> None:
        """Add Ring action: a ring at ``radius_px`` on the displayed frame with the next free id;
        ``profile`` is the displayed frame's radial profile when the caller already has it."""
        if radius_px <= 0:
            raise ValueError(f"radius_px must be positive, got {radius_px}")
        ring = {
            "id": max((int(ring["id"]) for ring in self.rings), default=0) + 1,
            "frame_idx": int(self.frame_idx),
            "hkl": "",
            "hkl_candidates": [],
            "d_ref": None,
            "d_error": None,
            "note": "",
            **self._ring_info(radius_px, self._radial_profile() if profile is None else profile),
        }
        self.rings = list(self.rings) + [ring]

    def _remove_ring(self, ring_id: int) -> None:
        """Remove Ring action; an unknown id leaves the rings (and their observers) untouched."""
        remaining = [ring for ring in self.rings if ring["id"] != ring_id]
        if len(remaining) != len(self.rings):
            self.rings = remaining

    def _undo_ring(self) -> None:
        """Undo Ring action: drop the newest ring."""
        self.rings = list(self.rings[:-1])

    def _clear_rings(self) -> None:
        """Clear Rings action."""
        self.rings = []

    def _detect_rings(self, max_rings: int | None = None, exclude_radius: float | None = None) -> list[float]:
        """Replace the rings with the ``max_rings`` most significant radial-profile peaks of
        the ``detect_denoise`` view and return every significant peak, most significant
        first. Significance is measured against the standard error of the raw frame's
        azimuthal mean: the denoised view only locates the peaks."""
        radii_px, intensity, _ = self._radial_profile(frame=self._detection_frame())
        raw_profile = self._radial_profile()
        candidates = lattice.detect_ring_radii(
            radii_px,
            intensity,
            raw_profile[2],
            exclude_radius=self.bf_radius if exclude_radius is None else exclude_radius,
        )
        self.rings = []
        for radius in sorted(candidates[:max_rings]):
            self._add_ring(radius, raw_profile)
        return candidates

    def _fit_rings(self) -> str:
        """Fit every ring peak on the background-subtracted raw profile and store the
        refined radius, FWHM, integrated intensity and R^2."""
        if not self.rings:
            raise ValueError("no rings to fit; call detect_rings or add_ring first")
        raw_profile = self._radial_profile()
        radii_px, intensity, _ = raw_profile
        # the background is removed in float64 so the subtraction does not round the peaks
        profile = intensity.astype(np.float64)
        try:
            profile = profile - self._radial_background(radii_px, intensity)
        except ValueError:
            # too few background bins (rings fill the profile): fit the raw profile
            pass
        calibrated = self.k_calibrated and self.k_pixel_size > 0
        rings = []
        for ring, update in zip(self.rings, lattice.fit_ring_peaks(radii_px, profile, self.rings)):
            ring = dict(ring)
            if update is None:
                ring["fit_quality"] = None
                rings.append(ring)
                continue
            ring.setdefault("raw_radius_px", update.pop("raw_radius_px"))
            # the fit re-measures on the displayed frame
            ring["frame_idx"] = int(self.frame_idx)
            ring.update(self._ring_info(update["radius_px"], raw_profile))
            ring.update(update)
            ring["fwhm_inv_angstrom"] = ring["fwhm_px"] * self.k_pixel_size if calibrated else None
            rings.append(ring)
        self.rings = rings
        n_fitted = sum(1 for ring in self.rings if ring.get("fit_quality") is not None)
        status = f"Fitted {n_fitted}/{len(self.rings)} rings"
        texture = lattice.texture_from_profile(*self._azimuthal_profile())
        return status + f", texture {texture['strength']:.2f} at {texture['angle_deg']:.0f}°"

    def _fit_ellipse(self) -> str:
        """Elliptical distortion from the outermost ring's radius vs azimuth."""
        radius = self._outer_ring_radius()
        theta_centers, counts, _, weight_sum, weighted_radius_sum = lattice.ring_sectors(
            self._frame(),
            center=(self.center_row, self.center_col),
            radius_px=radius,
            half_width=self._ring_half_width(radius),
            n_theta=180,
            mask=self._analysis_mask(),
            use_corrected_radius=False,
        )
        report = lattice.fit_ellipse_from_sectors(theta_centers, counts, weight_sum, weighted_radius_sum)
        with self.hold_trait_notifications():
            self.ellipse_ratio = report["ratio"]
            self.ellipse_angle = report["angle_deg"]
        return f"Ellipse ratio {report['ratio']:.3f} at {report['angle_deg']:.1f}°"

    def _merge(self) -> str:
        """Align the source frames by phase correlation and append their mean as a new frame."""
        if self.n_frames < 2:
            raise ValueError("merge needs a multi-frame stack")
        # alignment and the mean run in float64; the merged stack is stored as float32 again
        frames = self._data.cpu().numpy().astype(np.float64)
        # a repeated merge re-aligns the source frames, not the earlier merged pattern
        if self._n_source_frames is not None:
            frames = frames[: self._n_source_frames]
        aligned, _, used = lattice.align_frames(frames)
        if not any(used):
            raise ValueError("no frames survived alignment")
        merged = np.mean([frame for frame, usable in zip(aligned, used) if usable], axis=0)
        status = f"Merged {sum(used)}/{len(frames)} frames"
        if self.rings:
            outer_radius = self._outer_ring_radius()
            center = (self.center_row, self.center_col)
            mask = self._analysis_mask()
            before = lattice.ring_uniformity(frames[min(self.frame_idx, len(frames) - 1)], center, outer_radius, mask=mask)
            after = lattice.ring_uniformity(merged, center, outer_radius, mask=mask)
            status += f", ring coverage {before['coverage']:.2f} to {after['coverage']:.2f}"
        self._ingest(np.concatenate([frames, merged[None]], axis=0).astype(np.float32))
        self._n_source_frames = len(frames)
        self.frame_idx = self.n_frames - 1
        self._update_frame()
        self._bake_offline_frames()
        self._update_profile()
        self._update_azimuthal()
        return status

    # --- Calibration and phases ---
    def _require_calibrated(self) -> None:
        """Stop a d-spacing step on an uncalibrated pattern, where every d would be None."""
        if not (self.k_calibrated and self.k_pixel_size > 0):
            raise ValueError("pattern is uncalibrated; calibrate from a ring, spot or phase first")

    def _selected_phase(self) -> Phase | None:
        """The Phase menu choice: a library phase, else the custom phase of that name, else None."""
        if self.phase_name in PHASE_LIBRARY:
            return library_phase(self.phase_name)
        for entry in self.custom_phases:
            if entry.get("name") == self.phase_name:
                return phase_from_entry(entry)
        return None

    def _require_phase(self) -> Phase:
        """The selected phase, or a ValueError the browser shows when none is selected."""
        phase = self._selected_phase()
        if phase is None:
            raise ValueError("no phase selected; set phase_name or add a custom phase")
        return phase

    def _observed_d(self) -> list[float]:
        """Measured d-spacings to rank phases against, sorted: the rings, else the spots."""
        source = self.rings if self.rings else self.spots
        observed = sorted(d for d in (record.get("d_spacing") for record in source) if d and d > 0)
        if not observed:
            raise ValueError("no measured d-spacings; add rings or spots first")
        return observed

    def _calibrate(self, d_known: float, r_pixels: float, source: str) -> None:
        """Set ``k_pixel_size = 1 / (d_known * r_pixels)`` from one reflection of known d at
        ``r_pixels`` and record where the calibration came from."""
        if d_known <= 0:
            raise ValueError(f"d_known must be positive, got {d_known}")
        if r_pixels <= 0:
            raise ValueError("calibration point is at the center; no g-vector")
        with self.hold_trait_notifications():
            self.k_pixel_size = 1.0 / (d_known * r_pixels)
            self.k_calibrated = True
        self.calibration_source = source
        self.calibration_ref_d = float(d_known)
        self.calibration_ref_radius = float(r_pixels)

    def _calibrate_from_spot(self, row: float, col: float, d_known: float) -> None:
        """Calibrate from Spot action: the spot at (row, col) has spacing ``d_known``."""
        self._calibrate(d_known, self._radius_px(row, col), "from_spot")

    def _calibrate_from_ring(self, radius_px: float, d_known: float) -> None:
        """Calibrate from Ring action: the ring at ``radius_px`` has spacing ``d_known``."""
        self._calibrate(d_known, radius_px, "from_ring")

    def _calibrate_phase(self, phase: Phase | None = None) -> str:
        """Fit ``k_pixel_size`` from the rings of ``phase`` (default: the selected phase)
        and label the rings it explains.

        The scale is verified scale-free before it is applied: ring-radius ratios must
        match the phase's d-spacing ratios within :data:`lattice.RATIO_TOL`, and a second
        scale that explains the rings about as well, or more unexplained than explained
        rings, raise ``ValueError`` with the calibration left untouched (see
        :func:`lattice.calibration_scale_from_phase`). Rings outside the ratio test stay
        unlabelled; ``phase_match`` counts only the verified ones.
        """
        phase = phase or self._require_phase()
        if len(self.rings) < 2:
            raise ValueError("calibrate_from_phase needs >= 2 rings; calibrate from one ring instead")
        reflections = phase.reflections(d_min=0.5)
        if not reflections:
            raise ValueError(f"{phase.name} has no reflections above d_min=0.5")
        radii = [float(r["radius_px"]) for r in self.rings]
        solution = lattice.calibration_scale_from_phase(radii, [1.0 / ref["d"] for ref in reflections])
        scale = solution["scale"]
        with self.hold_trait_notifications():
            self.k_pixel_size = 1.0 / scale
            self.k_calibrated = True
        self.calibration_source = "from_phase"
        self.calibration_ref_d = 0.0
        self.calibration_ref_radius = 0.0
        residuals, d_errors, rings = [], [], []
        for ring, radius_px, reflection_index in zip(self.rings, radii, solution["assigned"]):
            ring = dict(ring)
            if reflection_index is None:
                ring.update({"hkl_candidates": [], "radius_resid_px": None, **index_fields(None)})
                rings.append(ring)
                continue
            reflection = reflections[reflection_index]
            d_error = abs(1.0 / (radius_px * self.k_pixel_size) - reflection["d"]) / reflection["d"]
            ring["hkl_candidates"] = [reflection["hkl_str"]]
            ring.update(index_fields({"hkl_str": reflection["hkl_str"], "d": reflection["d"], "d_error": d_error}))
            ring["radius_resid_px"] = radius_px - scale / reflection["d"]
            residuals.append(ring["radius_resid_px"])
            d_errors.append(d_error)
            rings.append(ring)
        self.rings = rings
        self.calibration_rms_px = float(np.sqrt(np.mean(np.square(residuals))))
        self.phase_match = phase_match_line(phase, d_errors, len(rings))
        return (
            f"Calibrated from {phase.name}: k={self.k_pixel_size:.5f} 1/Å/px, "
            f"{solution['n_explained']}/{len(rings)} rings fit one scale (rms {self.calibration_rms_px:.2f} px)"
        )

    def _index_rings(self, phase: Phase | None = None, tol: float = 0.03) -> str:
        """Label the rings with reflections of ``phase`` at the current calibration, verified
        by ring-radius ratios.

        Matching each d to the nearest reflection within ``tol`` is not evidence on its own:
        above a few 1/Å the reflections of a large cell are closer than 3 %, so any ring has
        a candidate. A label is applied only when the scale-free assignment of
        :func:`lattice.calibration_scale_from_phase` exists and its scale agrees with the
        current k within ``tol``; each ring that assignment explains takes its reflection.
        A ring whose d has a candidate within ``tol`` but which the ratios do not explain
        keeps the candidates in ``hkl_candidates`` with an empty ``hkl`` (the table shows
        it as unverified), and the returned status names those rings and the reason.
        The calibration itself is not changed.
        """
        self._require_calibrated()
        phase = phase or self._require_phase()
        reflections = phase.reflections(d_min=0.5)
        verified, reason = [None] * len(self.rings), ""
        try:
            solution = lattice.calibration_scale_from_phase(
                [float(r["radius_px"]) for r in self.rings], [1.0 / ref["d"] for ref in reflections]
            )
            # scale (px per 1/Å) times k (1/Å per px) is 1 when the ratios and the calibration agree
            offset = solution["scale"] * self.k_pixel_size - 1.0
            if abs(offset) <= tol:
                verified = solution["assigned"]
            else:
                reason = (
                    f"the ring ratios fit {phase.name} at k = {1.0 / solution['scale']:.5f} 1/Å/px, "
                    f"{100.0 * offset:+.1f} % from the current calibration"
                )
        except ValueError as exc:
            reason = str(exc)
        rings, d_errors, unverified = [], [], []
        for ring, reflection_index in zip(self.rings, verified):
            d = ring.get("d_spacing")
            candidates = phase.match_d(d, tol) if d else []
            ring = {**ring, "hkl_candidates": [c["hkl_str"] for c in candidates]}
            if reflection_index is None:
                ring.update(index_fields(None))
                if candidates:
                    unverified.append(ring["radius_px"])
            else:
                reflection = reflections[reflection_index]
                d_error = abs(d - reflection["d"]) / reflection["d"]
                ring.update(index_fields({"hkl_str": reflection["hkl_str"], "d": reflection["d"], "d_error": d_error}))
                d_errors.append(d_error)
            rings.append(ring)
        self.rings = rings
        self.phase_match = phase_match_line(phase, d_errors, len(rings)) if d_errors else ""
        status = f"Indexed {len(d_errors)}/{len(rings)} rings against {phase.name}, verified by ring ratios"
        if unverified:
            radii = ", ".join(f"{radius:.1f}" for radius in sorted(unverified))
            status += f"; unverified (d within {100 * tol:.0f}% only) at r = {radii} px"
        return status + (f" ({reason})" if reason else "")

    def _index_spots(self) -> str:
        """Index Spots action: :meth:`index_spots` against the selected phase, as a status line."""
        phase = self._require_phase()
        self.index_spots(phase)
        zone = f", zone {self.zone_axis}" if self.zone_axis else ""
        return f"Indexed spots against {phase.name}{zone}"

    def _identify(self) -> str:
        """Identify action: :meth:`search_phases` with the menu's filters, as a status line."""
        return lattice.identify_summary(self.search_phases())

    def _auto(self) -> str:
        """Auto action: :meth:`run_auto`; its ``analysis_status`` is the status line."""
        self.run_auto()
        return self.analysis_status

    def _quality_report(self) -> str:
        """QC snapshot for the browser panel: center method, calibration, ellipse, ring
        fits, unexplained rings, mask coverage and the outermost ring's uniformity."""
        mask = self._analysis_mask()
        indexed = [r for r in self.rings if r.get("hkl")]
        report = {
            "frame_idx": int(self.frame_idx),
            "center": {"method": self.center_method or self.center_mode},
            "calibration": {
                "source": self.calibration_source,
                "k_pixel_size": float(self.k_pixel_size),
                "rms_px": float(self.calibration_rms_px),
            },
            "ellipse": {
                "ratio": float(self.ellipse_ratio),
                "angle_deg": float(self.ellipse_angle),
                "corrected": bool(self.ellipse_corrected),
            },
            "rings": [{"id": r["id"], "fit_quality": r.get("fit_quality")} for r in self.rings],
            # with a phase selected every ring it does not index counts, a failed calibration included
            "n_unexplained_rings": len(self.rings) - len(indexed) if self.phase_name and self.rings else None,
            "mask_coverage_pct": float(mask.mean() * 100.0) if mask is not None else 0.0,
        }
        if self.rings:
            outermost = max(self.rings, key=lambda r: r["radius_px"])
            # SNR on the frame the ring was measured on
            frame = self._frame(outermost.get("frame_idx")).astype(np.float64)
            report["ring_snr"] = lattice.ring_uniformity(
                frame, (self.center_row, self.center_col), float(outermost["radius_px"]), mask=mask
            )
        self._quality = report
        return "Quality updated"

    # --- Browser requests ---
    _ACTIONS = {
        "add_spot": _add_spot,
        "move_spot": _move_spot,
        "remove_spot": _remove_spot,
        "undo_spot": _undo_spot,
        "clear_spots": _clear_spots,
        "detect_spots": detect_spots,
        "add_ring": _add_ring,
        "remove_ring": _remove_ring,
        "undo_ring": _undo_ring,
        "clear_rings": _clear_rings,
        "detect_rings": _detect_rings,
        "calibrate_from_spot": _calibrate_from_spot,
        "calibrate_from_ring": _calibrate_from_ring,
        "refine_center": _refine_center,
        "fit_rings": _fit_rings,
        "fit_ellipse": _fit_ellipse,
        "calibrate_phase": _calibrate_phase,
        "index_rings": _index_rings,
        "index_spots": _index_spots,
        "identify": _identify,
        "auto": _auto,
        "merge": _merge,
        "quality": _quality_report,
    }
    # analysis steps whose result the QC panel shows
    _QUALITY_ACTIONS = frozenset(
        {"refine_center", "fit_rings", "fit_ellipse", "calibrate_phase", "index_rings", "index_spots", "identify", "auto", "merge"}
    )

    def _on_request(self, change: dict) -> None:
        """Run the browser's ``_request`` action, report its status (or failure) in
        ``analysis_status``, refresh the QC panel after analysis steps and clear the request."""
        request = change["new"]
        if not request:
            return
        action = str(request.get("action", ""))
        try:
            # KeyError/TypeError also cover malformed custom_phases entries and bad args
            status = self._ACTIONS[action](self, *request.get("args", []))
        except (ValueError, KeyError, TypeError) as exc:
            status = f"{action.replace('_', ' ').capitalize()} failed: {exc}"
        if isinstance(status, str):
            self.analysis_status = status
        if action in self._QUALITY_ACTIONS:
            # the QC panel is a side view: its failure must not replace the action's status
            try:
                self._quality_report()
            except ValueError:
                pass
        self._request = {}

    # --- Export ---
    def _default_html_export_path(self) -> pathlib.Path:
        """``<title slug>_<frames>x<rows>x<cols>.html`` in the working directory."""
        slug = export_slug(self.title, "showdiffraction")
        return pathlib.Path.cwd() / f"{slug}_{self.n_frames}x{self.det_rows}x{self.det_cols}.html"

    def _clone_for_html_export(self) -> Self:
        """An offline copy of this widget with its full state, which the HTML export embeds so
        the page scrubs every frame without a kernel and the live widget is not modified."""
        if self._data is None:
            raise ValueError("Cannot export HTML after free(); rebuild the widget first.")
        clone = type(self)(
            to_numpy(self._data),
            center=(self.center_row, self.center_col),
            bf_radius=self.bf_radius,
            title=self.title,
            offline=True,
            verbose=False,
        )
        clone.load_state_dict(self.state_dict())
        # derived panels are not state fields
        clone._identify_results = list(self._identify_results)
        clone._quality = dict(self._quality)
        clone.export_enabled = False
        clone._update_frame()
        return clone
