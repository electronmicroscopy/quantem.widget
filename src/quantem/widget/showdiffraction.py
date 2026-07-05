"""
showdiffraction: Interactive d-spacing analysis for 2D/3D diffraction patterns.
"""

import json
import math
import pathlib
import tempfile
import time
from collections.abc import Sequence
from typing import Self

import anywidget
import numpy as np
import torch
import traitlets

from quantem.widget.crystal import PHASE_LIBRARY, Phase, library_phase
from quantem.widget.diffraction import (
    BF_RADIUS_FRACTION,
    azimuthal_profile_from_frame,
    build_analysis_mask,
    build_measurement_records,
    corrected_radius,
    element_symbols,
    empty_index_fields,
    fit_ellipse_from_sectors,
    fit_gaussian_spot,
    fit_radial_background,
    fit_ring_peaks,
    format_zone_axis,
    index_assignment,
    measurement_metadata,
    next_record_id,
    normalize_data_input,
    pack_float32_halves,
    parse_elements,
    radial_profile_px,
    ring_sectors,
    texture_from_profile,
    write_measurement_file,
)
from quantem.widget.export import ensure_mobile_viewport
from quantem.widget.utils.array import to_numpy
from quantem.widget.utils.state_io import resolve_widget_version, save_state_file, unwrap_state_payload
from quantem.widget.utils.ui import UiMode, resolve_ui_mode


class ShowDiffraction(anywidget.AnyWidget):
    """
    Interactive d-spacing analysis for 2D/3D diffraction patterns.

    Pick Bragg spots and rings on the diffraction pattern to measure d-spacings,
    g-vectors, and inter-spot angles, with optional sub-pixel Gaussian refinement.
    Works with a single 2D pattern (SAED) or a 3D stack of patterns, and accepts
    NumPy arrays, PyTorch tensors, or quantem datasets. 4D input is not supported.

    Parameters
    ----------
    data : np.ndarray or torch.Tensor
        2D ``(det_rows, det_cols)`` single pattern or 3D
        ``(n_frames, det_rows, det_cols)`` stack of patterns. A quantem dataset
        or io ``LoadResult`` is also accepted and unwrapped. 4D input raises.
    k_pixel_size : float, optional
        k-space sampling in 1/Å per pixel. Marks the pattern calibrated.
    pixel_size : float, optional
        Real-space pixel size in Å.
    center : tuple[float, float], optional
        (row, col) of the diffraction center in pixels. Defaults to the detector
        center, then auto-detected from the bright-field disk if also no radius.
    bf_radius : float, optional
        Bright-field disk radius in pixels. Defaults to 1/8 of the detector size.
    title : str, default ""
        Title displayed above the widget.
    snap_enabled : bool, default False
        Snap clicked spots to the local intensity maximum.
    snap_radius : int, default 5
        Search radius in pixels for snapping / Gaussian refinement.
    spot_refine : bool, default True
        Sub-pixel refine spots with a 2D Gaussian fit on add.
    dp_scale_mode : str, default "log"
        Diffraction display scaling ("linear", "log", "sqrt").
    ui_mode : {"interactive", "presentation", "report", "minimal"}, default "interactive"
        Shared viewer UI preset. Explicit ``show_*`` keyword arguments override
        preset values.
    show_title : bool, default True
        Show the top title row.
    show_stats : bool, default True
        Show statistics (mean, min, max, std).
    show_controls : bool, default True
        Show the control panel.
    controls_collapsed : bool, default False
        Start with controls hidden while keeping a recoverable ``Controls``
        button in the frontend.
    panel_width_px : int, optional
        Initial diffraction canvas width in CSS pixels. The frontend still lets
        users resize the panel interactively.
    verbose : bool, default True
        Print load timing on construction.
    state : str, pathlib.Path, or dict, optional
        Saved state to restore after construction.

    Examples
    --------
    >>> import numpy as np
    >>> from quantem.widget.showdiffraction import ShowDiffraction

    Single 2D diffraction pattern:

    >>> ShowDiffraction(np.random.rand(256, 256))

    Calibrated stack of diffraction patterns:

    >>> ShowDiffraction(np.random.rand(20, 128, 128), k_pixel_size=0.012)
    """

    _esm = pathlib.Path(__file__).parent / "static" / "showdiffraction.js"
    _CENTER_MODES = ("auto", "manual")
    _CENTER_METHODS = ("symmetry", "auto", "phase_corr")
    _SCALE_MODES = ("linear", "log", "sqrt")
    _LIST_STATE_FIELDS = {"spots", "rings", "custom_phases", "mask_regions"}
    _STATE_FIELDS = (
        "title",
        "frame_idx",
        "pixel_size",
        "k_pixel_size",
        "k_calibrated",
        "center_row",
        "center_col",
        "bf_radius",
        "spots",
        "rings",
        "zone_axis",
        "phase_match",
        "show_hkl",
        "snap_enabled",
        "snap_radius",
        "spot_refine",
        "center_mode",
        "calibration_source",
        "calibration_ref_d",
        "calibration_ref_radius",
        "calibration_rms_px",
        "ellipse_ratio",
        "ellipse_angle",
        "ellipse_corrected",
        "dp_colormap",
        "dp_scale_mode",
        "dp_invert",
        "dp_vmin_pct",
        "dp_vmax_pct",
        "show_title",
        "show_stats",
        "show_controls",
        "controls_collapsed",
        "show_profile",
        "profile_log",
        "profile_subtract_background",
        "phase_name",
        "custom_phases",
        "mask_regions",
        "show_mask",
        "profile_theta_min",
        "profile_theta_max",
        "show_azimuthal",
        "refine_method",
        "center_method",
        "identify_elements",
        "identify_custom_only",
    )
    _GEOMETRY_TRAITS = (
        "center_row",
        "center_col",
        "k_pixel_size",
        "k_calibrated",
        "ellipse_ratio",
        "ellipse_angle",
        "ellipse_corrected",
        "mask_regions",
    )
    _PROFILE_TRAITS = (
        "show_profile",
        "profile_subtract_background",
        "profile_theta_min",
        "profile_theta_max",
        "frame_idx",
    )

    # Core state
    widget_version = traitlets.Unicode("unknown").tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    n_frames = traitlets.Int(1).tag(sync=True)
    frame_idx = traitlets.Int(0).tag(sync=True)
    det_rows = traitlets.Int(1).tag(sync=True)
    det_cols = traitlets.Int(1).tag(sync=True)

    frame_bytes = traitlets.Bytes(b"").tag(sync=True)
    # Offline frame stack
    offline_frames = traitlets.Bytes(b"").tag(sync=True)

    # Offline render mode
    offline = traitlets.Bool(False).tag(sync=True)

    # HTML export bridge
    export_request = traitlets.Unicode("").tag(sync=True)
    export_status = traitlets.Unicode("").tag(sync=True)
    export_enabled = traitlets.Bool(True).tag(sync=True)
    export_payload = traitlets.Bytes(b"").tag(sync=True)
    export_payload_id = traitlets.Unicode("").tag(sync=True)
    export_filename = traitlets.Unicode("").tag(sync=True)

    # Detector calibration
    center_row = traitlets.Float(0.0).tag(sync=True)
    center_col = traitlets.Float(0.0).tag(sync=True)
    bf_radius = traitlets.Float(0.0).tag(sync=True)
    pixel_size = traitlets.Float(1.0).tag(sync=True)
    k_pixel_size = traitlets.Float(0.0).tag(sync=True)
    k_calibrated = traitlets.Bool(False).tag(sync=True)

    center_mode = traitlets.Unicode("auto").tag(sync=True)

    calibration_source = traitlets.Unicode("none").tag(sync=True)
    calibration_ref_d = traitlets.Float(0.0).tag(sync=True)
    calibration_ref_radius = traitlets.Float(0.0).tag(sync=True)
    calibration_rms_px = traitlets.Float(0.0).tag(sync=True)

    refine_method = traitlets.Unicode("auto").tag(sync=True)
    center_method = traitlets.Unicode("").tag(sync=True)

    # Ellipse correction
    ellipse_ratio = traitlets.Float(1.0).tag(sync=True)
    ellipse_angle = traitlets.Float(0.0).tag(sync=True)
    ellipse_corrected = traitlets.Bool(False).tag(sync=True)

    # Spots and rings
    spots = traitlets.List(traitlets.Dict()).tag(sync=True)
    snap_enabled = traitlets.Bool(False).tag(sync=True)
    snap_radius = traitlets.Int(5).tag(sync=True)

    rings = traitlets.List(traitlets.Dict()).tag(sync=True)

    spot_refine = traitlets.Bool(True).tag(sync=True)

    # Indexing
    zone_axis = traitlets.Unicode("").tag(sync=True)
    phase_match = traitlets.Unicode("").tag(sync=True)
    show_hkl = traitlets.Bool(True).tag(sync=True)

    # Frontend requests
    _spot_add_request = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    _spot_undo_request = traitlets.Bool(False).tag(sync=True)
    _spot_clear_request = traitlets.Bool(False).tag(sync=True)
    _ring_add_request = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    _ring_undo_request = traitlets.Bool(False).tag(sync=True)
    _ring_clear_request = traitlets.Bool(False).tag(sync=True)
    _calibrate_from_ring_request = traitlets.List(traitlets.Float(), default_value=[]).tag(
        sync=True
    )
    _calibrate_from_spot_request = traitlets.List(traitlets.Float(), default_value=[]).tag(
        sync=True
    )
    _detect_spots_request = traitlets.Int(0).tag(sync=True)  # max_spots, -1 = all
    _detect_rings_request = traitlets.Int(0).tag(sync=True)  # max_rings, -1 = all
    _spot_remove_request = traitlets.Int(0).tag(sync=True)  # spot id
    _spot_move_request = traitlets.List(traitlets.Float(), default_value=[]).tag(
        sync=True
    )  # id, row, col
    _ring_remove_request = traitlets.Int(0).tag(sync=True)  # ring id
    _refine_center_request = traitlets.Bool(False).tag(sync=True)
    _fit_rings_request = traitlets.Bool(False).tag(sync=True)
    _fit_ellipse_request = traitlets.Bool(False).tag(sync=True)
    _calibrate_phase_request = traitlets.Bool(False).tag(sync=True)
    _index_rings_request = traitlets.Bool(False).tag(sync=True)
    _index_spots_request = traitlets.Bool(False).tag(sync=True)
    _identify_request = traitlets.Bool(False).tag(sync=True)
    _auto_request = traitlets.Bool(False).tag(sync=True)
    _merge_request = traitlets.Bool(False).tag(sync=True)
    _quality_request = traitlets.Bool(False).tag(sync=True)
    analysis_status = traitlets.Unicode("").tag(sync=True)
    _quality = traitlets.Dict().tag(sync=True)
    selected_ring_id = traitlets.Int(0).tag(sync=True)

    # Analysis mask
    mask_regions = traitlets.List(traitlets.Dict()).tag(sync=True)
    show_mask = traitlets.Bool(True).tag(sync=True)

    # Phase workbench
    phase_name = traitlets.Unicode("").tag(sync=True)
    custom_phases = traitlets.List(traitlets.Dict()).tag(sync=True)
    _phase_library = traitlets.List(traitlets.Dict()).tag(sync=True)
    identify_elements = traitlets.Unicode("").tag(sync=True)
    identify_custom_only = traitlets.Bool(False).tag(sync=True)
    _identify_results = traitlets.List(traitlets.Dict()).tag(sync=True)

    # Display
    dp_colormap = traitlets.Unicode("inferno").tag(sync=True)
    dp_scale_mode = traitlets.Unicode("log").tag(sync=True)
    dp_invert = traitlets.Bool(False).tag(sync=True)
    dp_vmin_pct = traitlets.Float(0.0).tag(sync=True)
    dp_vmax_pct = traitlets.Float(100.0).tag(sync=True)

    # Profiles
    show_profile = traitlets.Bool(False).tag(sync=True)
    profile_log = traitlets.Bool(True).tag(sync=True)
    profile_subtract_background = traitlets.Bool(False).tag(sync=True)
    profile_theta_min = traitlets.Float(0.0).tag(sync=True)
    profile_theta_max = traitlets.Float(360.0).tag(sync=True)
    _profile_data = traitlets.Bytes(b"").tag(sync=True)  # float32 pairs
    show_azimuthal = traitlets.Bool(False).tag(sync=True)
    _azimuthal_data = traitlets.Bytes(b"").tag(sync=True)  # float32 pairs

    # Statistics
    dp_stats = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0, 0.0, 0.0]).tag(sync=True)

    # =========================================================================
    # UI Visibility
    # =========================================================================
    show_title = traitlets.Bool(True).tag(sync=True)
    show_stats = traitlets.Bool(True).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)
    controls_collapsed = traitlets.Bool(False).tag(sync=True)
    panel_width_px = traitlets.Int(384).tag(sync=True)

    @traitlets.validate("center_mode")
    def _validate_center_mode(self, proposal):
        val = proposal["value"]
        if val not in self._CENTER_MODES:
            raise ValueError(f"center_mode must be one of {self._CENTER_MODES}, got {val!r}")
        return val

    @traitlets.validate("frame_idx")
    def _validate_frame_idx(self, proposal):
        # Saved-state bounds
        val = int(proposal["value"])
        n = max(1, int(self.n_frames))
        return max(0, min(val, n - 1))

    @traitlets.validate("dp_scale_mode")
    def _validate_dp_scale_mode(self, proposal):
        val = proposal["value"]
        if val not in self._SCALE_MODES:
            raise ValueError(f"dp_scale_mode must be one of {self._SCALE_MODES}, got {val!r}")
        return val

    def __init__(
        self,
        data: np.ndarray | torch.Tensor,
        k_pixel_size: float | None = None,
        pixel_size: float | None = None,
        center: tuple[float, float] | None = None,
        bf_radius: float | None = None,
        title: str = "",
        snap_enabled: bool = False,
        snap_radius: int = 5,
        spot_refine: bool = True,
        dp_scale_mode: str = "log",
        ui_mode: UiMode = "interactive",
        show_title: bool | None = None,
        show_stats: bool | None = None,
        show_controls: bool | None = None,
        controls_collapsed: bool | None = None,
        panel_width_px: int | None = None,
        offline: bool = False,
        verbose: bool = True,
        state=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        t_start = time.perf_counter()
        self.widget_version = resolve_widget_version()
        user_k_pixel_size = k_pixel_size is not None
        data, title, pixel_size, k_pixel_size, metadata_calibrated = normalize_data_input(
            data,
            title=title,
            pixel_size=pixel_size,
            k_pixel_size=k_pixel_size,
        )

        self._device = self._best_device()
        self._ingest_data(data)
        self._set_initial_calibration(
            pixel_size,
            k_pixel_size,
            metadata_calibrated=metadata_calibrated,
            user_k_pixel_size=user_k_pixel_size,
        )

        self.title = title
        self.dp_scale_mode = dp_scale_mode
        self.snap_enabled = snap_enabled
        self.snap_radius = snap_radius
        self.spot_refine = spot_refine
        ui = resolve_ui_mode(
            ui_mode,
            defaults={
                "show_title": True,
                "show_stats": True,
                "show_controls": True,
                "controls_collapsed": False,
            },
            overrides={
                "show_title": show_title,
                "show_stats": show_stats,
                "show_controls": show_controls,
                "controls_collapsed": controls_collapsed,
            },
        )
        self.show_title = bool(ui["show_title"])
        self.show_stats = bool(ui["show_stats"])
        self.show_controls = bool(ui["show_controls"])
        self.controls_collapsed = bool(ui["controls_collapsed"])
        if panel_width_px is not None:
            self.panel_width_px = int(panel_width_px)
        self.offline = offline

        self._set_initial_geometry(center, bf_radius)

        self._update_frame()
        self._bake_offline_frames()
        self._phase_library = [{"name": name, **entry} for name, entry in PHASE_LIBRARY.items()]
        self._observe_traits()

        if verbose:
            mem_mb = self._data.nelement() * 4 / 1e6
            print(f"  to {self._device}: {time.perf_counter() - t_start:.2f}s ({mem_mb:.1f} MB)")

        self._load_initial_state(state)

    @staticmethod
    def _best_device() -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _set_initial_calibration(
        self,
        pixel_size: float | None,
        k_pixel_size: float | None,
        *,
        metadata_calibrated: bool,
        user_k_pixel_size: bool,
    ) -> None:
        if pixel_size is not None:
            self.pixel_size = float(pixel_size)
        if k_pixel_size is not None and k_pixel_size > 0:
            self.k_pixel_size = float(k_pixel_size)
            self.k_calibrated = True
            self.calibration_source = "manual" if user_k_pixel_size else "metadata"
        elif metadata_calibrated:
            self.k_calibrated = True
            self.calibration_source = "metadata"

    def _set_initial_geometry(
        self,
        center: tuple[float, float] | None,
        bf_radius: float | None,
    ) -> None:
        if center is None:
            self.center_row = float(self.det_rows / 2)
            self.center_col = float(self.det_cols / 2)
        else:
            self.center_row = float(center[0])
            self.center_col = float(center[1])

        self.bf_radius = (
            float(bf_radius)
            if bf_radius is not None
            else min(self.det_rows, self.det_cols) * BF_RADIUS_FRACTION
        )
        if center is None and bf_radius is None:
            self.auto_detect_center()

    def _observe_traits(self) -> None:
        self.observe(self._update_frame, names=["frame_idx"])
        self.observe(self._bake_offline_frames, names=["offline"])
        self.observe(self._on_spot_add_request, names=["_spot_add_request"])
        self.observe(self._on_spot_undo_request, names=["_spot_undo_request"])
        self.observe(self._on_spot_clear_request, names=["_spot_clear_request"])
        self.observe(self._on_ring_add_request, names=["_ring_add_request"])
        self.observe(self._on_ring_undo_request, names=["_ring_undo_request"])
        self.observe(self._on_ring_clear_request, names=["_ring_clear_request"])
        self.observe(self._on_calibrate_from_ring_request, names=["_calibrate_from_ring_request"])
        self.observe(self._on_calibrate_from_spot_request, names=["_calibrate_from_spot_request"])
        self.observe(self._on_geometry_change, names=list(self._GEOMETRY_TRAITS))
        self.observe(self._on_detect_spots_request, names=["_detect_spots_request"])
        self.observe(self._on_detect_rings_request, names=["_detect_rings_request"])
        self.observe(self._on_spot_remove_request, names=["_spot_remove_request"])
        self.observe(self._on_spot_move_request, names=["_spot_move_request"])
        self.observe(self._on_ring_remove_request, names=["_ring_remove_request"])
        self.observe(self._on_status_request, names=list(self._STATUS_REQUESTS))
        self.observe(self._on_quality_request, names=["_quality_request"])
        self.observe(self._update_profile, names=list(self._PROFILE_TRAITS))
        self.observe(self._update_azimuthal, names=["show_azimuthal", "rings", "frame_idx"])
        self.observe(self._on_export_request_change, names=["export_request"])

    def _load_initial_state(self, state) -> None:
        if state is None:
            return
        if isinstance(state, (str, pathlib.Path)):
            state = unwrap_state_payload(
                json.loads(pathlib.Path(state).read_text()),
                require_envelope=True,
            )
        else:
            state = unwrap_state_payload(state)
        self.load_state_dict(state)

    def _ingest_data(self, data):
        array = to_numpy(data)
        is_integer = np.issubdtype(array.dtype, np.integer)
        array = array.astype(np.float32)
        if array.size > 2**31 - 1 and self._device.type == "mps":
            self._device = torch.device("cpu")
        if is_integer:
            global_max = float(array.max())
            p999 = float(np.percentile(array, 99.9))
            if global_max > p999 * 5:
                array[array > p999 * 3] = 0
        ndim = array.ndim
        if ndim == 2:
            array = array[None, ...]
        elif ndim == 4:
            raise ValueError(
                "ShowDiffraction is for 2D/3D diffraction patterns; 4D input is not supported."
            )
        elif ndim != 3:
            raise ValueError(f"Expected a 2D or 3D array, got {ndim}D")
        self._det_shape = (array.shape[1], array.shape[2])
        self._data = torch.from_numpy(np.ascontiguousarray(array)).to(self._device)
        self.n_frames = int(array.shape[0])
        self.det_rows = self._det_shape[0]
        self.det_cols = self._det_shape[1]

    @property
    def detector_shape(self) -> tuple[int, int]:
        """Detector shape as ``(rows, cols)``."""
        return self._det_shape

    def auto_detect_center(self, *, refine: bool = False) -> Self:
        """Find the BF disk center/radius from the summed stack."""
        summed_dp = self._data.sum(dim=0)

        threshold = summed_dp.mean() + summed_dp.std()
        mask = summed_dp > threshold

        total = mask.sum()
        if total == 0:
            return self

        row_coords = torch.arange(self.det_rows, device=self._device, dtype=torch.float32)[:, None]
        col_coords = torch.arange(self.det_cols, device=self._device, dtype=torch.float32)[None, :]
        self.center_row = float((row_coords * mask).sum() / total)
        self.center_col = float((col_coords * mask).sum() / total)
        # Central component
        self.bf_radius = self._central_beam_radius(mask, self.center_row, self.center_col)
        self.center_mode = "auto"
        if refine:
            self.refine_center()
        return self

    def refine_center(self, *, method: str = "symmetry", search_radius: float = 8.0) -> Self:
        """Refine the center with symmetry, phase correlation, or auto."""
        if method not in self._CENTER_METHODS:
            raise ValueError(f"unknown refine method {method!r}")

        from quantem.widget import centering

        picked = centering.pick_center(
            self._displayed_frame().astype(np.float64),
            method=method,
            mask=self._analysis_mask(),
            guess=(self.center_row, self.center_col),
            search_radius=search_radius,
        )
        self.center_row, self.center_col = float(picked["row"]), float(picked["col"])
        self.center_mode = "auto"
        self.center_method = picked["method"]
        return self

    def _central_beam_radius(self, mask, center_row: float, center_col: float) -> float:
        mask_np = mask.detach().cpu().numpy()
        try:
            from scipy.ndimage import label
        except Exception:
            return float(np.sqrt(float(mask_np.sum()) / np.pi))
        labels, n_labels = label(mask_np)
        if n_labels == 0:
            return 0.0
        row_idx = int(min(max(round(center_row), 0), mask_np.shape[0] - 1))
        col_idx = int(min(max(round(center_col), 0), mask_np.shape[1] - 1))
        central_label = int(labels[row_idx, col_idx])
        if central_label == 0:
            # Beam stop
            comp_rows, comp_cols = np.nonzero(labels)
            nearest = int(np.argmin((comp_rows - center_row) ** 2 + (comp_cols - center_col) ** 2))
            central_label = int(labels[comp_rows[nearest], comp_cols[nearest]])
        area = float((labels == central_label).sum())
        return float(np.sqrt(area / np.pi))

    def set_center(self, row: float, col: float) -> Self:
        """Set the diffraction center to (row, col) and mark the mode manual."""
        self.center_row = float(row)
        self.center_col = float(col)
        self.center_mode = "manual"
        return self

    def _get_frame(self, idx: int) -> np.ndarray:
        idx = max(0, min(int(idx), self.n_frames - 1))
        return self._data[idx].cpu().numpy().astype(np.float32)

    def _displayed_frame(self) -> np.ndarray:
        return self._get_frame(self.frame_idx)

    def _update_frame(self, change=None):
        frame = self._displayed_frame()
        self.dp_stats = [
            float(frame.mean()),
            float(frame.min()),
            float(frame.max()),
            float(frame.std()),
        ]
        self.frame_bytes = frame.tobytes()

    def _bake_offline_frames(self, change=None) -> None:
        # Offline stack
        if self.offline and self.n_frames > 1 and getattr(self, "_data", None) is not None:
            frames = self._data.cpu().numpy().astype(np.float32)
            self.offline_frames = np.ascontiguousarray(frames).tobytes()
        else:
            self.offline_frames = b""

    def _compute_spot_info(
        self, row: float, col: float, row_err: float = 0.0, col_err: float = 0.0
    ) -> dict:
        d_row = row - self.center_row
        d_col = col - self.center_col
        r_pixels = float(
            corrected_radius(
                d_row,
                d_col,
                ellipse_ratio=self.ellipse_ratio,
                ellipse_angle=self.ellipse_angle,
                ellipse_corrected=self.ellipse_corrected,
            )
        )

        # Radial uncertainty
        if r_pixels > 0:
            r_err = math.hypot((d_row / r_pixels) * row_err, (d_col / r_pixels) * col_err)
        else:
            r_err = math.hypot(row_err, col_err)

        frame = self._displayed_frame()
        r_int = max(0, min(self.det_rows - 1, int(round(row))))
        c_int = max(0, min(self.det_cols - 1, int(round(col))))
        intensity = float(frame[r_int, c_int])

        if self.k_calibrated and self.k_pixel_size > 0 and r_pixels > 0:
            g_magnitude = r_pixels * self.k_pixel_size
            d_spacing = 1.0 / g_magnitude
            # Propagated d error
            frac = r_err / r_pixels
            g_err = g_magnitude * frac
            d_err = d_spacing * frac
        else:
            g_magnitude = d_spacing = g_err = d_err = None

        return {
            "d_spacing": d_spacing,
            "d_spacing_err": d_err,
            "g_magnitude": g_magnitude,
            "g_magnitude_err": g_err,
            "r_pixels": r_pixels,
            "r_pixels_err": r_err,
            "intensity": intensity,
        }

    def _with_angles(self, spots) -> list:
        if not spots:
            return spots
        reference = spots[0]
        center_row, center_col = self.center_row, self.center_col
        ref_row = reference["row"] - center_row
        ref_col = reference["col"] - center_col
        ref_radius = math.hypot(ref_row, ref_col)
        ref_error = math.hypot(reference.get("row_err", 0.0), reference.get("col_err", 0.0))
        with_angles = []
        for spot in spots:
            delta_row = spot["row"] - center_row
            delta_col = spot["col"] - center_col
            radius = math.hypot(delta_row, delta_col)
            if ref_radius > 0 and radius > 0:
                cos_a = max(
                    -1.0,
                    min(1.0, (ref_row * delta_row + ref_col * delta_col) / (ref_radius * radius)),
                )
                angle = math.degrees(math.acos(cos_a))
                spot_error = math.hypot(spot.get("row_err", 0.0), spot.get("col_err", 0.0))
                angle_err = math.degrees(math.hypot(spot_error / radius, ref_error / ref_radius))
            else:
                angle = None
                angle_err = None
            with_angles.append({**spot, "angle_deg": angle, "angle_deg_err": angle_err})
        return with_angles

    def detect_spots(
        self,
        max_spots: int | None = None,
        min_distance: int = 6,
        min_relative: float = 0.1,
        exclude_radius: float | None = None,
        replace: bool = True,
    ) -> Self:
        """Detect Bragg spots with contrast at least ``min_relative`` of the strongest peak."""
        frame = self._displayed_frame().astype(np.float64)
        n_rows, n_cols = frame.shape
        if exclude_radius is None:
            exclude_radius = max(self.bf_radius, 2.0 * float(min_distance))
        try:
            from scipy.ndimage import gaussian_filter, maximum_filter
        except Exception:
            return self
        work = np.log1p(np.clip(frame - frame.min(), 0.0, None))
        work = work - gaussian_filter(work, sigma=max(2.0, float(min_distance)))

        size = max(3, int(min_distance) | 1)  # odd window
        local_max = maximum_filter(work, size=size) == work
        rows = np.arange(n_rows)[:, None]
        cols = np.arange(n_cols)[None, :]
        radius = np.hypot(rows - self.center_row, cols - self.center_col)
        local_max &= radius > float(exclude_radius)
        exclusion = self._analysis_mask()
        if exclusion is not None:
            local_max &= ~exclusion
        local_max[0, :] = local_max[-1, :] = False
        local_max[:, 0] = local_max[:, -1] = False
        coords = np.argwhere(local_max)
        if replace:
            self.clear_spots()
        if coords.size == 0:
            return self
        prominence = work[coords[:, 0], coords[:, 1]]
        # contrast relative to the strongest peak, with a noise floor on noisy data
        contrast = np.expm1(prominence)
        sigma = 1.4826 * float(np.median(np.abs(work - np.median(work))))
        level = max(min_relative * float(contrast.max()), float(np.expm1(5.0 * sigma)))
        keep = (prominence > 0) & (contrast >= level)
        coords, prominence = coords[keep], prominence[keep]
        if coords.size:
            # isolated peaks only: a circle around a spot drops on all sides,
            # around a ring-crest bump it stays high along the ring
            angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
            ring_r = float(max(3, min_distance))
            rr = np.clip(coords[:, :1] + ring_r * np.sin(angles), 0, n_rows - 1).astype(int)
            cc = np.clip(coords[:, 1:] + ring_r * np.cos(angles), 0, n_cols - 1).astype(int)
            ring_vals = work[rr, cc]
            peak = work[coords[:, 0], coords[:, 1]]
            hi = np.percentile(ring_vals, 90, axis=1)
            lo = np.percentile(ring_vals, 10, axis=1)
            isolated = (peak - hi) >= 0.5 * np.maximum(peak - lo, 1e-9)
            coords, prominence = coords[isolated], prominence[isolated]
        order = np.argsort(-prominence)
        if max_spots is not None:
            order = order[: int(max_spots)]
        for r0, c0 in coords[order]:
            self.add_spot(float(r0), float(c0))
        return self

    def detect_rings(
        self,
        max_rings: int | None = None,
        prominence_rel: float = 0.05,
        min_separation: int = 5,
        exclude_radius: float | None = None,
        replace: bool = True,
    ) -> Self:
        """Detect Debye-Scherrer rings from radial profile peaks (max_rings=None keeps all)."""
        try:
            radii_px, intensity = self._radial_profile()
        except Exception:
            return self
        y = np.asarray(intensity, dtype=np.float64)
        if replace:
            self.clear_rings()
        if y.size < 5:
            return self
        try:
            from scipy.ndimage import gaussian_filter1d
            from scipy.signal import find_peaks
        except Exception:
            return self
        if exclude_radius is None:
            exclude_radius = self.bf_radius
        y_log = np.log1p(np.clip(y - y.min(), 0.0, None))
        detrended = y_log - gaussian_filter1d(y_log, sigma=max(3.0, y_log.size / 20.0))
        span = float(detrended.max() - detrended.min())
        prominence = prominence_rel * span if span > 0 else None
        peaks, props = find_peaks(
            detrended, prominence=prominence, distance=max(1, int(min_separation))
        )
        if peaks.size == 0:
            return self
        outside_beam = radii_px[peaks] > float(exclude_radius)
        peaks = peaks[outside_beam]
        prominences = props["prominences"][outside_beam]
        if peaks.size == 0:
            return self
        strongest = np.argsort(prominences)[::-1]
        if max_rings is not None:
            strongest = strongest[: int(max_rings)]
        for p in sorted(peaks[strongest]):
            self.add_ring(float(radii_px[p]))
        return self

    def _on_detect_spots_request(self, change=None):
        n = self._detect_spots_request
        if n:
            self.detect_spots(max_spots=int(n) if n > 0 else None)
            self._detect_spots_request = 0

    def _on_detect_rings_request(self, change=None):
        n = self._detect_rings_request
        if n:
            self.detect_rings(max_rings=int(n) if n > 0 else None)
            self._detect_rings_request = 0

    def _snap_to_peak(self, row: float, col: float) -> tuple[float, float]:
        frame = self._displayed_frame()
        r, c = int(round(row)), int(round(col))
        radius = int(self.snap_radius)
        r0 = max(0, r - radius)
        r1 = min(self.det_rows, r + radius + 1)
        c0 = max(0, c - radius)
        c1 = min(self.det_cols, c + radius + 1)
        region = frame[r0:r1, c0:c1]
        if region.size == 0:
            return float(row), float(col)
        idx = np.unravel_index(region.argmax(), region.shape)
        return float(r0 + idx[0]), float(c0 + idx[1])

    def _pick_spot_fields(self, row: float, col: float) -> dict:
        # Position + measurement fields per the current pick mode (fit/snap/exact)
        raw_row, raw_col = float(row), float(col)
        row_err = col_err = 0.0
        fit_quality = None
        if self.spot_refine:
            fit = fit_gaussian_spot(
                self._displayed_frame(),
                raw_row,
                raw_col,
                half_window=self.snap_radius,
            )
            if fit is not None:
                row, col = fit["row"], fit["col"]
                row_err, col_err = fit["row_err"], fit["col_err"]
                fit_quality = fit["fit_quality"]
        elif self.snap_enabled:
            row, col = self._snap_to_peak(raw_row, raw_col)
        info = self._compute_spot_info(row, col, row_err=row_err, col_err=col_err)
        return {
            "row": float(row),
            "col": float(col),
            "raw_row": raw_row,
            "raw_col": raw_col,
            "row_err": float(row_err),
            "col_err": float(col_err),
            "fit_quality": fit_quality,
            **empty_index_fields(),
            **info,
        }

    def add_spot(self, row: float, col: float) -> Self:
        """Add a spot, optionally refining or snapping it."""
        spot = {
            "id": next_record_id(self.spots),
            "angle_deg": None,
            "angle_deg_err": None,
            **self._pick_spot_fields(row, col),
        }
        self.spots = self._with_angles(list(self.spots) + [spot])
        return self

    def move_spot(self, spot_id: int, row: float, col: float) -> Self:
        """Move the spot with id ``spot_id``, re-picking it at the new position."""
        idx = next((i for i, s in enumerate(self.spots) if s["id"] == spot_id), None)
        if idx is None:
            return self
        spots = list(self.spots)
        spots[idx] = {**spots[idx], **self._pick_spot_fields(row, col)}
        self.spots = self._with_angles(spots)
        return self

    def clear_spots(self) -> Self:
        """Remove all spots."""
        self.spots = []
        return self

    def undo_spot(self) -> Self:
        """Remove the most recently added spot."""
        if self.spots:
            self.spots = list(self.spots[:-1])
        return self

    def remove_spot(self, spot_id: int) -> Self:
        """Remove the spot with id ``spot_id`` (no-op if not present)."""
        remaining = [s for s in self.spots if s["id"] != spot_id]
        if len(remaining) != len(self.spots):
            self.spots = self._with_angles(remaining)
        return self

    def _on_spot_add_request(self, change=None):
        val = self._spot_add_request
        if val and len(val) == 2:
            self.add_spot(val[0], val[1])
            self._spot_add_request = []

    def _on_spot_undo_request(self, change=None):
        if self._spot_undo_request:
            self.undo_spot()
            self._spot_undo_request = False

    def _on_spot_clear_request(self, change=None):
        if self._spot_clear_request:
            self.clear_spots()
            self._spot_clear_request = False

    def _on_spot_remove_request(self, change=None):
        if self._spot_remove_request:
            self.remove_spot(int(self._spot_remove_request))
            self._spot_remove_request = 0

    def _on_spot_move_request(self, change=None):
        val = self._spot_move_request
        if val and len(val) == 3:
            self.move_spot(int(val[0]), val[1], val[2])
            self._spot_move_request = []

    def _recompute_spots(self):
        if not self.spots:
            return
        spots = [
            {
                **s,
                **self._compute_spot_info(
                    s["row"], s["col"], s.get("row_err", 0.0), s.get("col_err", 0.0)
                ),
            }
            for s in self.spots
        ]
        self.spots = self._with_angles(spots)

    def _on_geometry_change(self, change=None):
        # Derived geometry
        self._recompute_spots()
        self._recompute_rings()
        self._update_profile()
        self._update_azimuthal()

    def _compute_ring_info(self, radius_px: float) -> dict:
        if self.k_calibrated and self.k_pixel_size > 0:
            g_magnitude = float(radius_px) * self.k_pixel_size
            d_spacing = 1.0 / g_magnitude if g_magnitude > 0 else None
        else:
            g_magnitude = d_spacing = None
        radii_px, intensity = self._radial_profile()
        ring_intensity = (
            float(intensity[int(np.argmin(np.abs(radii_px - radius_px)))])
            if radii_px.size
            else 0.0
        )
        return {
            "radius_px": float(radius_px),
            "g_magnitude": g_magnitude,
            "d_spacing": d_spacing,
            "intensity": ring_intensity,
        }

    def add_ring(self, radius_px: float) -> Self:
        """Add a ring at radius_px from the center (polycrystalline d-spacing pick)."""
        ring = {
            "id": next_record_id(self.rings),
            **empty_index_fields(),
            **self._compute_ring_info(radius_px),
        }
        self.rings = list(self.rings) + [ring]
        return self

    def clear_rings(self) -> Self:
        """Remove all rings."""
        self.rings = []
        return self

    def undo_ring(self) -> Self:
        """Remove the most recently added ring."""
        if self.rings:
            self.rings = list(self.rings[:-1])
        return self

    def remove_ring(self, ring_id: int) -> Self:
        """Remove the ring with id ``ring_id`` (no-op if not present)."""
        remaining = [r for r in self.rings if r["id"] != ring_id]
        if len(remaining) != len(self.rings):
            self.rings = remaining
        return self

    def _recompute_rings(self):
        if not self.rings:
            return
        self.rings = [{**r, **self._compute_ring_info(r["radius_px"])} for r in self.rings]

    def fit_ring_profile(
        self,
        *,
        window: float | None = None,
        model: str = "gaussian",
        subtract_background: bool = True,
    ) -> Self:
        """Fit each ring peak and store refined radius, width, area, and quality."""
        if not self.rings:
            raise ValueError("no rings to fit; call add_ring or detect_rings first")

        radii_px, intensity = self._radial_profile()
        profile = intensity.astype(np.float64)
        if subtract_background:
            try:
                _, background = self.radial_background()
                profile = profile - background.astype(np.float64)
            except ValueError:
                pass

        calibrated = self.k_calibrated and self.k_pixel_size > 0
        updates = fit_ring_peaks(radii_px, profile, self.rings, model=model, window=window)
        rings = []
        for ring, update in zip(self.rings, updates):
            ring = dict(ring)
            if update is None:
                ring["fit_quality"] = None
                rings.append(ring)
                continue
            raw_radius = update.pop("raw_radius_px")
            ring.setdefault("raw_radius_px", raw_radius)
            ring.update(self._compute_ring_info(update["radius_px"]))
            ring.update(update)
            ring["fwhm_inv_angstrom"] = ring["fwhm_px"] * self.k_pixel_size if calibrated else None
            rings.append(ring)
        self.rings = rings
        return self

    def _on_ring_add_request(self, change=None):
        val = self._ring_add_request
        if val and len(val) == 1:
            self.add_ring(val[0])
            self._ring_add_request = []

    def _on_ring_undo_request(self, change=None):
        if self._ring_undo_request:
            self.undo_ring()
            self._ring_undo_request = False

    def _on_ring_clear_request(self, change=None):
        if self._ring_clear_request:
            self.clear_rings()
            self._ring_clear_request = False

    def _on_ring_remove_request(self, change=None):
        if self._ring_remove_request:
            self.remove_ring(int(self._ring_remove_request))
            self._ring_remove_request = 0

    # Request dispatch
    _STATUS_REQUESTS = {
        "_refine_center_request": ("Refine", "_do_refine_center"),
        "_fit_rings_request": ("Ring fit", "_do_fit_rings"),
        "_fit_ellipse_request": ("Ellipse", "_do_fit_ellipse"),
        "_calibrate_phase_request": ("Phase calibration", "_do_calibrate_phase"),
        "_index_rings_request": ("Ring indexing", "_do_index_rings"),
        "_index_spots_request": ("Spot indexing", "_do_index_spots"),
        "_identify_request": ("Identify", "_do_identify"),
        "_auto_request": ("Auto", "_do_auto"),
        "_merge_request": ("Merge", "_do_merge"),
    }

    def _on_status_request(self, change):
        if not change["new"]:
            return
        prefix, method = self._STATUS_REQUESTS[change["name"]]
        try:
            self.analysis_status = getattr(self, method)()
        except (ValueError, ImportError) as exc:
            self.analysis_status = f"{prefix} failed: {exc}"
        setattr(self, change["name"], False)
        try:
            self.quality_report()
        except (ValueError, ImportError):
            pass

    def _on_quality_request(self, change=None):
        if not self._quality_request:
            return
        try:
            self.quality_report()
            self.analysis_status = "Quality updated"
        except (ValueError, ImportError) as exc:
            self.analysis_status = f"Quality failed: {exc}"
        self._quality_request = False

    def _do_refine_center(self) -> str:
        self.refine_center(method=self.refine_method)
        return f"Center ({self.center_row:.1f}, {self.center_col:.1f}) via {self.center_method}"

    def _do_fit_rings(self) -> str:
        self.fit_ring_profile()
        n_ok = sum(1 for r in self.rings if r.get("fit_quality") is not None)
        status = f"Fitted {n_ok}/{len(self.rings)} rings"
        try:
            tex = self.texture()
            status += f", texture {tex['strength']:.2f} at {tex['angle_deg']:.0f}°"
        except (ValueError, ImportError):
            pass
        return status

    def _do_fit_ellipse(self) -> str:
        report = self.fit_ellipse()
        return f"Ellipse ratio {report['ratio']:.3f} at {report['angle_deg']:.1f}°"

    def _do_calibrate_phase(self) -> str:
        phase = self._require_phase()
        self.calibrate_from_phase(phase)
        return (
            f"Calibrated from {phase.name}: k={self.k_pixel_size:.5f} 1/Å/px "
            f"(rms {self.calibration_rms_px:.2f} px)"
        )

    def _do_index_rings(self) -> str:
        phase = self._require_phase()
        self.index_rings(phase)
        n = sum(1 for r in self.rings if r.get("hkl"))
        return f"Indexed {n}/{len(self.rings)} rings against {phase.name}"

    def _do_index_spots(self) -> str:
        phase = self._require_phase()
        self.index_spots(phase)
        zone = f", zone {self.zone_axis}" if self.zone_axis else ""
        return f"Indexed spots against {phase.name}{zone}"

    def _do_identify(self) -> str:
        ranked = self.search_phases()
        return self._identify_summary(ranked)

    def _do_auto(self) -> str:
        self.run_auto()
        return self.analysis_status

    def _do_merge(self) -> str:
        report = self.merge_frames()
        status = f"Merged {report['n_used']}/{report['n_frames']} frames"
        if "after" in report:
            status += (
                f", ring coverage {report['before']['coverage']:.2f} to "
                f"{report['after']['coverage']:.2f}"
            )
        return status

    def _selected_phase(self) -> Phase | None:
        if not self.phase_name:
            return None
        if self.phase_name in PHASE_LIBRARY:
            return library_phase(self.phase_name)
        for entry in self.custom_phases:
            if entry.get("name") == self.phase_name:
                return self._custom_phase(entry)
        return None

    @staticmethod
    def _custom_phase(entry: dict) -> Phase:
        a = float(entry["a"])
        return Phase(
            entry["name"],
            a,
            float(entry.get("b", a)),
            float(entry.get("c", a)),
            float(entry.get("alpha", 90.0)),
            float(entry.get("beta", 90.0)),
            float(entry.get("gamma", 90.0)),
            absences=entry.get("absences", "none"),
        )

    def _require_phase(self) -> Phase:
        phase = self._selected_phase()
        if phase is None:
            raise ValueError("no phase selected; set phase_name or add a custom phase")
        return phase

    def _all_phases(self, custom_only: bool = False) -> list[Phase]:
        phases = [] if custom_only else [library_phase(name) for name in PHASE_LIBRARY]
        for entry in self.custom_phases:
            try:
                phases.append(self._custom_phase(entry))
            except (KeyError, ValueError, TypeError):
                continue
        return phases

    def run_auto(
        self,
        phase: Phase | None = None,
        *,
        max_rings: int = 8,
        exclude_radius: float | None = None,
    ) -> Self:
        """Run center finding, ring detection, calibration, fitting, and indexing.

        Silent on success; ``analysis_status`` only reports steps that failed.
        """
        phase = phase or self._selected_phase()
        problems = []
        self.auto_detect_center(refine=True)
        self.detect_rings(max_rings=max_rings, exclude_radius=exclude_radius)
        if not self.rings:
            problems.append("ring detection failed (no rings found)")
        if phase is None and self.phase_name:
            problems.append(f'calibration failed (phase "{self.phase_name}" not found)')
        if phase is not None and self.rings:
            try:
                self.calibrate_from_phase(phase)
            except ValueError as exc:
                problems.append(f"calibration failed ({exc})")
                phase = None
        if self.rings:
            try:
                self.fit_ring_profile()
                if all(r.get("fit_quality") is None for r in self.rings):
                    problems.append("ring fit failed")
            except (ValueError, ImportError):
                problems.append("ring fit failed")
        if phase is not None and self.k_calibrated:
            try:
                self.index_rings(phase)
                if not any(r.get("hkl") for r in self.rings):
                    problems.append("indexing failed (no rings matched)")
            except ValueError as exc:
                problems.append(f"indexing failed ({exc})")
        self.analysis_status = "Auto: " + ", ".join(problems) if problems else ""
        return self

    def merge_frames(
        self, *, statistic: str = "mean", align: bool = True, max_shift: float = 8.0
    ) -> dict:
        """Align the stack and append the combined pattern as a new frame."""
        if self.n_frames < 2:
            raise ValueError("merge_frames needs a multi-frame stack")
        if statistic not in ("mean", "median", "max"):
            raise ValueError(f"statistic must be mean, median or max, got {statistic!r}")
        from quantem.widget import centering

        frames = self._data.cpu().numpy().astype(np.float64)
        if align:
            aligned, shifts, used = centering.align_frames(frames, max_shift=max_shift)
        else:
            aligned, shifts, used = frames, [(0.0, 0.0)] * len(frames), [True] * len(frames)
        if not any(used):
            raise ValueError("no frames survived alignment; raise max_shift or set align=False")
        stack = np.asarray([f for f, u in zip(aligned, used) if u])
        merged = getattr(np, statistic)(stack, axis=0)
        report = {
            "n_frames": int(len(frames)),
            "n_used": int(len(stack)),
            "shifts": [(float(s[0]), float(s[1])) for s in shifts],
            "used": [bool(u) for u in used],
        }
        if self.rings:
            r0 = max(r["radius_px"] for r in self.rings)
            center = (self.center_row, self.center_col)
            report["before"] = centering.ring_uniformity(frames[self.frame_idx], center, r0)
            report["after"] = centering.ring_uniformity(merged, center, r0)
        self._ingest_data(np.concatenate([frames, merged[None]], axis=0).astype(np.float32))
        self.frame_idx = self.n_frames - 1
        self._update_frame()
        self._bake_offline_frames()
        return report

    def quality_report(self) -> dict:
        """QC snapshot: center method, calibration, ellipse, ring fits,
        unexplained rings, mask coverage, and outermost-ring SNR.
        """
        frame = self._displayed_frame().astype(np.float64)
        mask = self._analysis_mask()
        indexed = [r for r in self.rings if r.get("hkl")]
        report = {
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
            "n_unexplained_rings": (len(self.rings) - len(indexed)) if indexed else 0,
            "mask_coverage_pct": float(mask.mean() * 100.0) if mask is not None else 0.0,
        }
        if self.rings:
            try:
                from quantem.widget import centering

                r0 = max(r["radius_px"] for r in self.rings)
                report["ring_snr"] = centering.ring_uniformity(
                    frame, (self.center_row, self.center_col), r0
                )
            except ImportError:
                pass
        self._quality = report
        return report

    def _update_profile(self, change=None):
        if not self.show_profile:
            self._profile_data = b""
            return
        sector = (self.profile_theta_min, self.profile_theta_max)
        sector = None if sector == (0.0, 360.0) else sector
        try:
            radii, intensity = self.radial_profile(
                units="px",
                subtract_background=self.profile_subtract_background,
                angular_range=sector,
            )
        except ValueError as exc:
            self.analysis_status = f"Background subtract failed: {exc}"
            radii, intensity = self.radial_profile(units="px", angular_range=sector)
        self._profile_data = pack_float32_halves(radii, intensity)

    def _update_azimuthal(self, change=None):
        if not self.show_azimuthal:
            self._azimuthal_data = b""
            return
        try:
            self._azimuthal_data = pack_float32_halves(*self.azimuthal_profile())
        except ValueError as exc:
            self.analysis_status = f"Azimuthal profile failed: {exc}"
            self._azimuthal_data = b""

    def _radial_profile(
        self,
        *,
        n_bins: int | None = None,
        max_radius: float | None = None,
        center: tuple[float, float] | None = None,
        angular_range: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return radial_profile_px(
            self._displayed_frame(),
            center=center or (self.center_row, self.center_col),
            n_bins=n_bins,
            max_radius=max_radius,
            mask=self._analysis_mask(),
            angular_range=angular_range,
            ellipse_ratio=self.ellipse_ratio,
            ellipse_angle=self.ellipse_angle,
            ellipse_corrected=self.ellipse_corrected,
        )

    def _analysis_mask(self) -> "np.ndarray | None":
        return build_analysis_mask(
            (self.det_rows, self.det_cols),
            self.mask_regions,
            (self.center_row, self.center_col),
        )

    def radial_profile(
        self,
        *,
        n_bins: int | None = None,
        max_radius: float | None = None,
        center: tuple[float, float] | None = None,
        units: str = "auto",
        angular_range: tuple[float, float] | None = None,
        subtract_background: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Azimuthally averaged profile in px, q, or d units."""
        if units not in ("auto", "px", "q", "d"):
            raise ValueError(f"units must be 'auto', 'px', 'q' or 'd', got {units!r}")
        if n_bins is not None and n_bins <= 0:
            raise ValueError(f"n_bins must be positive, got {n_bins}")
        if max_radius is not None and max_radius <= 0:
            raise ValueError(f"max_radius must be positive, got {max_radius}")
        if angular_range is not None and len(angular_range) != 2:
            raise ValueError("angular_range must be a (start_deg, end_deg) pair")
        calibrated = self.k_calibrated and self.k_pixel_size > 0
        if units in ("q", "d") and not calibrated:
            raise ValueError(
                f"radial_profile(units={units!r}) needs a calibrated pattern; call "
                "calibrate_from_ring / calibrate_from_spot / calibrate_from_phase first"
            )
        radii_px, intensity = self._radial_profile(
            n_bins=n_bins, max_radius=max_radius, center=center, angular_range=angular_range
        )
        if subtract_background:
            _, background = self.radial_background(
                n_bins=n_bins, max_radius=max_radius, center=center
            )
            intensity = intensity - background
        if units == "auto":
            units = "q" if calibrated else "px"
        if units == "px":
            return radii_px, intensity
        if units == "q":
            return (radii_px * self.k_pixel_size).astype(np.float32), intensity
        keep = radii_px > 0
        d_axis = (1.0 / (radii_px[keep] * self.k_pixel_size)).astype(np.float32)
        return d_axis, intensity[keep]

    def _ring_radius_for(self, ring_id: int | None, radius_px: float | None) -> float:
        if radius_px is not None:
            if radius_px <= 0:
                raise ValueError(f"radius_px must be positive, got {radius_px}")
            return float(radius_px)
        if not self.rings:
            raise ValueError("no ring to analyze; call detect_rings / add_ring or pass radius_px")
        if ring_id is None:
            return float(max(self.rings, key=lambda r: r["radius_px"])["radius_px"])
        matches = [r for r in self.rings if r["id"] == ring_id]
        if not matches:
            raise ValueError(f"no ring with id {ring_id}; have {[r['id'] for r in self.rings]}")
        return float(matches[0]["radius_px"])

    def azimuthal_profile(
        self,
        *,
        ring_id: int | None = None,
        radius_px: float | None = None,
        width: float | None = None,
        n_theta: int = 180,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Intensity vs azimuth around a ring."""
        radius = self._ring_radius_for(ring_id, radius_px)
        half_width = float(width) if width is not None else max(6.0, 0.25 * radius)
        return azimuthal_profile_from_frame(
            self._displayed_frame(),
            center=(self.center_row, self.center_col),
            radius_px=radius,
            half_width=half_width,
            n_theta=n_theta,
            mask=self._analysis_mask(),
            ellipse_ratio=self.ellipse_ratio,
            ellipse_angle=self.ellipse_angle,
            ellipse_corrected=self.ellipse_corrected,
        )

    def texture(
        self,
        *,
        ring_id: int | None = None,
        radius_px: float | None = None,
        width: float | None = None,
        n_theta: int = 180,
        return_profile: bool = False,
    ) -> dict:
        """Order-2 ring texture: strength in [0, 1] and 180-degree angle."""
        theta_deg, intensity = self.azimuthal_profile(
            ring_id=ring_id, radius_px=radius_px, width=width, n_theta=n_theta
        )
        return texture_from_profile(theta_deg, intensity, return_profile=return_profile)

    def fit_ellipse(self, ring_id: int | None = None, *, n_theta: int = 180) -> dict:
        """Fit ellipse distortion from ring radius vs azimuth."""
        radius = self._ring_radius_for(ring_id, None)
        half_width = max(6.0, 0.25 * radius)
        theta_centers, counts, _, weight_sum, weighted_radius_sum = ring_sectors(
            self._displayed_frame(),
            center=(self.center_row, self.center_col),
            radius_px=radius,
            half_width=half_width,
            n_theta=n_theta,
            mask=self._analysis_mask(),
            use_corrected_radius=False,
        )
        report = fit_ellipse_from_sectors(theta_centers, counts, weight_sum, weighted_radius_sum)
        self.ellipse_ratio = report["ratio"]
        self.ellipse_angle = report["angle_deg"]
        return report

    def apply_ellipse_correction(self, *, enable: bool = True) -> Self:
        """Enable or disable radius circularization by the fitted ellipse."""
        self.ellipse_corrected = bool(enable)
        return self

    def radial_background(
        self,
        *,
        n_bins: int | None = None,
        max_radius: float | None = None,
        center: tuple[float, float] | None = None,
        method: str = "power",
        poly_order: int = 3,
        peak_windows: list[tuple[float, float]] | None = None,
        exclude_radius: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit a smooth radial background while excluding peaks."""
        radii_px, intensity = self._radial_profile(
            n_bins=n_bins, max_radius=max_radius, center=center
        )
        if exclude_radius is None:
            exclude_radius = self.bf_radius
        if peak_windows is None:
            peak_windows = []
            for ring in self.rings:
                half = ring.get("fwhm_px") or 6.0
                peak_windows.append((ring["radius_px"] - half, ring["radius_px"] + half))
        return radii_px, fit_radial_background(
            radii_px,
            intensity,
            peak_windows=peak_windows,
            exclude_radius=exclude_radius,
            method=method,
            poly_order=poly_order,
        )

    # Indexing and phase identification
    def _require_calibrated(self) -> None:
        if not (self.k_calibrated and self.k_pixel_size > 0):
            raise ValueError(
                "pattern is uncalibrated; call calibrate_from_ring / calibrate_from_spot first"
            )

    def _match_report(self, phase: Phase, d_values: Sequence[float | None], tol: float) -> dict:
        errors = []
        for d in d_values:
            if d and d > 0:
                cands = phase.match_d(d, tol)
                if cands:
                    errors.append(cands[0]["d_error"])
        n_total = sum(1 for d in d_values if d and d > 0)
        mean_err = float(np.mean(errors)) if errors else 0.0
        return {
            "name": phase.name,
            "n_matched": len(errors),
            "n_total": n_total,
            "mean_error": mean_err,
        }

    def _set_phase_match(self, report: dict, absences: str) -> None:
        pct = 100.0 * report["mean_error"]
        self.phase_match = (
            f"{report['name']} ({absences}): "
            f"{report['n_matched']}/{report['n_total']} matched, {pct:.1f}% mean error"
        )

    def index_rings(self, phase: Phase, tol: float = 0.03, replace: bool = True) -> Self:
        """Label rings by d-spacing match against a calibrated phase."""
        self._require_calibrated()
        rings = [dict(r) for r in self.rings]
        for r in rings:
            if not replace and r.get("hkl"):
                continue
            d = r.get("d_spacing")
            cands = phase.match_d(d, tol) if d else []
            r["hkl_candidates"] = [c["hkl_str"] for c in cands]
            r.update(index_assignment(cands[0] if cands else None))
        self.rings = rings
        self._set_phase_match(
            self._match_report(phase, [r.get("d_spacing") for r in rings], tol), phase.absences
        )
        return self

    def identify_phase(self, database, tol: float = 0.03) -> list[dict]:
        """Rank an explicit list of candidate phases against measured d-spacings.

        This is the primary verification workflow: build the candidates you
        expect (:meth:`Phase.from_cif`, :meth:`Phase.from_cubic`,
        :meth:`Phase.from_dspacings`, ...) and rank only those. Use
        :meth:`search_phases` when you have no candidates in mind.
        """
        self._require_calibrated()
        phases = list(database)
        reports = self._rank_phases(self._observed_d(), phases, tol, max(len(phases), 1))
        self._set_identify_results(reports)
        return reports

    def search_phases(
        self,
        *,
        tol: float = 0.03,
        elements=None,
        exclude_elements=None,
        extra=None,
        custom_only: bool | None = None,
        top_n: int = 10,
    ) -> list[dict]:
        """Rank library, custom, and extra phases against measured d-spacings.

        With ``custom_only`` (default: the ``identify_custom_only`` trait) the
        library is skipped and only user candidates — custom phases plus
        ``extra`` — are ranked.
        """
        self._require_calibrated()
        observed = self._observed_d()
        if custom_only is None:
            custom_only = self.identify_custom_only
        allowed = parse_elements(elements if elements is not None else self.identify_elements)
        excluded = parse_elements(exclude_elements)
        candidates = list(self._all_phases(custom_only=custom_only)) + list(extra or [])
        if custom_only and not candidates:
            raise ValueError("no candidate phases; add custom phases or pass extra")
        phases = []
        for phase in candidates:
            els = element_symbols(phase.name)
            if allowed is not None and els and not els <= allowed:
                continue
            if excluded and els & excluded:
                continue
            phases.append(phase)
        reports = self._rank_phases(observed, phases, tol, top_n)
        self._set_identify_results(reports)
        return reports

    def _observed_d(self) -> list[float]:
        source = self.rings if self.rings else self.spots
        observed = sorted(d for d in (x.get("d_spacing") for x in source) if d and d > 0)
        if not observed:
            raise ValueError("no measured d-spacings; add rings or spots first")
        return observed

    @staticmethod
    def _phase_lines(phase: Phase, d_min: float) -> list[dict]:
        return [
            {
                "d": reflection["d"],
                "hkl": reflection["hkl_str"],
                "i_rel": reflection["intensity"] or 0.0,
            }
            for reflection in phase.reflections(d_min=d_min)
        ]

    def _rank_phases(self, observed, phases, tol, top_n) -> list[dict]:
        from quantem.widget.phasedb import match_candidate, match_sort_key

        reports = []
        d_min = min(observed) * 0.8
        for phase in phases:
            lines = self._phase_lines(phase, d_min=d_min)
            if len(lines) < 2:
                continue
            report = match_candidate(observed, lines, tol=tol)
            report.update(
                {
                    "phase_id": f"phase-{phase.name}",
                    "name": phase.name,
                    "formula": "",
                    "spacegroup": "",
                    "crystal_system": "",
                    "lines": self._match_lines(observed, lines, report),
                }
            )
            report.pop("assignments", None)
            reports.append(report)
        if not reports:
            raise ValueError("no candidate phases pass the filters")
        reports.sort(key=match_sort_key)
        return reports[: int(top_n)]

    def _identify_summary(self, reports: list[dict]) -> str:
        top = reports[0]
        status = f"{top['name']}: {top['matched']}/{top['n_obs']} lines"
        if top["n_obs"] < 4:
            status += "; few measured lines"
        runners = ", ".join(report["name"] for report in reports[1:3])
        return f"{status}; next: {runners}" if runners else status

    def _set_identify_results(self, reports: list[dict]) -> None:
        self._identify_results = reports[:10]
        self.phase_match = self._identify_summary(reports)

    @staticmethod
    def _match_lines(observed: list[float], lines: list[dict], report: dict) -> list[dict]:
        rows = []
        assignments = dict(report.get("assignments") or [])
        used_refs = set()
        for obs_index, measured_d in enumerate(observed):
            ref_index = assignments.get(obs_index)
            if ref_index is None:
                rows.append(
                    {
                        "obs_d": float(measured_d),
                        "ref_d": None,
                        "hkl": "",
                        "err": None,
                        "i_rel": None,
                    }
                )
            else:
                ref = lines[ref_index]
                used_refs.add(ref_index)
                rows.append(
                    {
                        "obs_d": float(measured_d),
                        "ref_d": float(ref["d"]),
                        "hkl": ref.get("hkl", ""),
                        "err": abs(ref["d"] - measured_d) / ref["d"],
                        "i_rel": ref.get("i_rel"),
                    }
                )
        lo, hi = min(observed), max(observed)
        missing = [
            (j, ref)
            for j, ref in enumerate(lines)
            if j not in used_refs and lo <= ref["d"] <= hi and (ref.get("i_rel") or 0) >= 25
        ]
        missing.sort(key=lambda x: -(x[1].get("i_rel") or 0))
        for _, ref in missing[:5]:
            rows.append(
                {
                    "obs_d": None,
                    "ref_d": float(ref["d"]),
                    "hkl": ref.get("hkl", ""),
                    "err": None,
                    "i_rel": ref.get("i_rel"),
                }
            )
        return rows

    def _spot_vector(self, spot: dict) -> tuple[float, float]:
        return spot["row"] - self.center_row, spot["col"] - self.center_col

    def _measured_angle(self, s1: dict, s2: dict) -> float:
        dr1, dc1 = self._spot_vector(s1)
        dr2, dc2 = self._spot_vector(s2)
        r1, r2 = math.hypot(dr1, dc1), math.hypot(dr2, dc2)
        if r1 == 0 or r2 == 0:
            return 0.0
        cos_a = max(-1.0, min(1.0, (dr1 * dr2 + dc1 * dc2) / (r1 * r2)))
        return math.degrees(math.acos(cos_a))

    def _find_anchor_pair(
        self,
        phase: Phase,
        spots: list[dict],
        cand_lists: list[list[dict]],
        angle_tol: float,
    ) -> tuple[int, int, dict, dict] | None:
        """First non-collinear spot pair that matches phase angle geometry."""
        for i in range(len(spots)):
            for j in range(i + 1, len(spots)):
                if not cand_lists[i] or not cand_lists[j]:
                    continue
                measured = self._measured_angle(spots[i], spots[j])
                if measured < 1e-6:
                    continue
                best = None
                for ci in cand_lists[i]:
                    for cj in cand_lists[j]:
                        err = abs(phase.plane_angle(ci["hkl"], cj["hkl"]) - measured)
                        if err <= angle_tol and (best is None or err < best[0]):
                            best = (err, ci, cj)
                if best is not None:
                    return i, j, best[1], best[2]
        return None

    def index_spots(self, phase: Phase, tol: float = 0.03, angle_tol: float = 3.0) -> Self:
        """Index spots and solve the zone axis from an angle-consistent anchor pair."""
        self._require_calibrated()
        if phase.lattice is None:
            raise ValueError(
                "index_spots needs a lattice-based Phase (from_cubic / full constructor) "
                "for the inter-spot angle check; a d-spacing card has no angles"
            )
        spots = [dict(s) for s in self.spots]
        cand_lists = []
        for s in spots:
            d = s.get("d_spacing")
            cands = phase.match_d(d, tol) if d else []
            s["hkl_candidates"] = [c["hkl_str"] for c in cands]
            cand_lists.append(cands)

        anchors = self._find_anchor_pair(phase, spots, cand_lists, angle_tol)
        if anchors is None:
            for s, cands in zip(spots, cand_lists):
                s.update(index_assignment(cands[0] if cands else None))
            self.spots = self._with_angles(spots)
            self.zone_axis = ""
            return self

        i, j, ci, cj = anchors
        ref = {i: ci, j: cj}
        for idx, (s, cands) in enumerate(zip(spots, cand_lists)):
            if idx in ref:
                chosen = ref[idx]
            elif cands:
                measured = self._measured_angle(spots[i], s)
                chosen = min(
                    cands, key=lambda c: abs(phase.plane_angle(ci["hkl"], c["hkl"]) - measured)
                )
            else:
                chosen = None
            s.update(index_assignment(chosen))

        self.spots = self._with_angles(spots)
        self.zone_axis = format_zone_axis(ci["hkl"], cj["hkl"])
        self._set_phase_match(
            self._match_report(phase, [s.get("d_spacing") for s in spots], tol), phase.absences
        )
        return self

    def calibrate_from_spot(self, row: float, col: float, d_known: float) -> Self:
        """Calibrate ``k_pixel_size`` from a spot of known d-spacing."""
        if d_known <= 0:
            raise ValueError(f"d_known must be positive, got {d_known}")
        r_pixels = float(
            corrected_radius(
                row - self.center_row,
                col - self.center_col,
                ellipse_ratio=self.ellipse_ratio,
                ellipse_angle=self.ellipse_angle,
                ellipse_corrected=self.ellipse_corrected,
            )
        )
        if r_pixels <= 0:
            raise ValueError("calibration point is at the center; no g-vector")
        self.k_pixel_size = 1.0 / (d_known * r_pixels)
        self.k_calibrated = True
        self.calibration_source = "from_spot"
        self.calibration_ref_d = float(d_known)
        self.calibration_ref_radius = float(r_pixels)
        return self

    def calibrate_from_ring(self, radius_px: float, d_known: float) -> Self:
        """Calibrate ``k_pixel_size`` from a ring of known d-spacing."""
        if d_known <= 0:
            raise ValueError(f"d_known must be positive, got {d_known}")
        if radius_px <= 0:
            raise ValueError(f"radius_px must be positive, got {radius_px}")
        self.k_pixel_size = 1.0 / (d_known * radius_px)
        self.k_calibrated = True
        self.calibration_source = "from_ring"
        self.calibration_ref_d = float(d_known)
        self.calibration_ref_radius = float(radius_px)
        return self

    def calibrate_from_phase(self, phase: Phase, *, tol: float = 0.03, d_min: float = 0.5) -> Self:
        """Fit ``k_pixel_size`` by assigning ring-radius ratios to a known phase."""
        if len(self.rings) < 2:
            raise ValueError(
                "calibrate_from_phase needs >= 2 rings; use calibrate_from_ring for a single ring"
            )
        refl = phase.reflections(d_min=d_min)
        if not refl:
            raise ValueError(f"{phase.name} has no reflections above d_min={d_min}")
        radii = [float(r["radius_px"]) for r in self.rings]
        inv_d = [1.0 / rf["d"] for rf in refl]

        r_inner = min(radii)
        best = None
        for x0 in inv_d:
            scale = r_inner / x0
            assigned, errs = [], []
            for r in radii:
                x_pred = r / scale
                nearest = min(range(len(inv_d)), key=lambda i: abs(inv_d[i] - x_pred))
                err = abs(inv_d[nearest] - x_pred) / x_pred
                assigned.append(nearest if err <= tol else None)
                errs.append(err if err <= tol else None)
            used = [a for a in assigned if a is not None]
            n_ok = len(used)
            if n_ok < 2:
                continue
            mean_err = float(np.mean([e for e in errs if e is not None]))
            candidate_key = (n_ok, -sum(inv_d[a] for a in used), -mean_err)
            if best is None or candidate_key > best[0]:
                best = (candidate_key, assigned)
        if best is None:
            raise ValueError(
                f"could not assign >= 2 rings to {phase.name} reflections within tol={tol}; "
                "check the phase or calibrate_from_ring manually"
            )
        _, assigned = best

        pairs = [
            (radius_px, inv_d[reflection_index])
            for radius_px, reflection_index in zip(radii, assigned)
            if reflection_index is not None
        ]
        scale = sum(radius_px * q for radius_px, q in pairs) / sum(q * q for _, q in pairs)
        self.k_pixel_size = 1.0 / scale
        self.k_calibrated = True
        self.calibration_source = "from_phase"
        self.calibration_ref_d = 0.0
        self.calibration_ref_radius = 0.0

        resids = []
        rings = [dict(r) for r in self.rings]
        for ring, radius_px, reflection_index in zip(rings, radii, assigned):
            if reflection_index is None:
                ring["hkl_candidates"] = []
                ring.update(index_assignment(None))
                ring["radius_resid_px"] = None
                continue
            reflection = refl[reflection_index]
            measured_d = 1.0 / (radius_px * self.k_pixel_size)
            assignment = {
                "hkl_str": reflection["hkl_str"],
                "d": reflection["d"],
                "d_error": abs(measured_d - reflection["d"]) / reflection["d"],
            }
            ring["hkl_candidates"] = [reflection["hkl_str"]]
            ring.update(index_assignment(assignment))
            residual_px = radius_px - scale * inv_d[reflection_index]
            ring["radius_resid_px"] = residual_px
            resids.append(residual_px)
        self.rings = rings
        self.calibration_rms_px = float(np.sqrt(np.mean(np.square(resids))))
        return self

    def _on_calibrate_from_ring_request(self, change=None):
        val = self._calibrate_from_ring_request
        if val and len(val) == 2:
            try:
                self.calibrate_from_ring(val[0], val[1])
            except ValueError:
                pass
            self._calibrate_from_ring_request = []

    def _on_calibrate_from_spot_request(self, change=None):
        val = self._calibrate_from_spot_request
        if val and len(val) == 3:
            try:
                self.calibrate_from_spot(val[0], val[1], val[2])
            except ValueError:
                pass
            self._calibrate_from_spot_request = []

    def export_measurements(self, path: str) -> pathlib.Path:
        """Export spot and ring measurements as CSV or JSON."""
        return write_measurement_file(
            path,
            build_measurement_records(self.spots, self.rings),
            measurement_metadata(self.state_dict()),
        )

    @classmethod
    def measurements_from_state(cls, state, path=None):
        """Rebuild the measurement table from a saved state."""
        if isinstance(state, (str, pathlib.Path)):
            state = unwrap_state_payload(
                json.loads(pathlib.Path(state).read_text()), require_envelope=True
            )
        else:
            state = unwrap_state_payload(state)
        records = build_measurement_records(state.get("spots", []), state.get("rings", []))
        if path is None:
            return records
        return write_measurement_file(path, records, measurement_metadata(state))

    def export_html(
        self,
        path: str | pathlib.Path | None = None,
        *,
        title: str | None = None,
        **options,
    ) -> pathlib.Path:
        """Write a standalone HTML viewer with exact float32 frames."""
        if not hasattr(self, "_data") or self._data is None:
            raise ValueError("Cannot export HTML after free(); rebuild the widget first.")
        export_path = pathlib.Path(path) if path is not None else self._default_html_export_path()
        self._write_html_export(export_path, title=title)
        ensure_mobile_viewport(export_path)
        size_mb = export_path.stat().st_size / (1024 * 1024)
        self.export_status = f"Exported {export_path.name} ({size_mb:.1f} MB, full float32)"
        return export_path

    def _on_export_request_change(self, change: dict) -> None:
        raw = str(change.get("new") or "")
        if not raw:
            return
        try:
            payload = json.loads(raw)
            mode = str(payload.get("mode", "single"))
            if mode == "clear":
                self.export_payload = b""
                self.export_payload_id = ""
                self.export_filename = ""
                return
            if payload.get("download"):
                filename = str(payload.get("filename") or self._default_html_export_path().name)
                request_id = str(payload.get("id") or "")
                self.export_status = f"Preparing {filename}..."
                html = self._html_export_bytes()
                self.export_filename = filename
                self.export_payload = html
                self.export_payload_id = request_id
                size_mb = len(html) / (1024 * 1024)
                self.export_status = f"Ready {filename} ({size_mb:.1f} MB, full float32)"
            else:
                self.export_status = "Exporting HTML..."
                self.export_html()
        except Exception as exc:
            self.export_status = f"Export failed: {exc}"

    def _default_html_export_path(self) -> pathlib.Path:
        label = self.title.strip() or "showdiffraction"
        slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        if not slug:
            slug = "showdiffraction"
        shape = f"{self.n_frames}x{self.det_rows}x{self.det_cols}"
        return pathlib.Path.cwd() / f"{slug}_{shape}.html"

    def _write_html_export(
        self,
        path: str | pathlib.Path,
        *,
        title: str | None = None,
    ) -> pathlib.Path:
        from ipywidgets.embed import dependency_state, embed_minimal_html

        export_path = pathlib.Path(path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        page_title = title or self.title or "ShowDiffraction"
        export_widget = self._clone_for_html_export()
        try:
            state = dependency_state([export_widget], drop_defaults=False)
            embed_minimal_html(
                str(export_path),
                views=[export_widget],
                title=page_title,
                drop_defaults=False,
                state=state,
            )
        finally:
            export_widget.close()
        return export_path

    def _html_export_bytes(self) -> bytes:
        with tempfile.TemporaryDirectory(prefix="showdiffraction-export-") as tmp:
            path = pathlib.Path(tmp) / self._default_html_export_path().name
            self._write_html_export(path)
            ensure_mobile_viewport(path)
            return path.read_bytes()

    def _clone_for_html_export(self) -> Self:
        if not hasattr(self, "_data") or self._data is None:
            raise ValueError("Cannot export HTML after free(); rebuild the widget first.")
        clone = type(self)(to_numpy(self._data), state=self.state_dict(), verbose=False)
        clone.offline = True
        clone.export_enabled = False
        clone.export_status = ""
        clone.export_payload = b""
        clone.export_payload_id = ""
        clone.export_filename = ""
        clone._update_frame()
        return clone

    def set_image(self, data) -> Self:
        """Replace data. Preserves display settings, clears spots and rings."""
        data, title, pixel_size, k_pixel_size, metadata_calibrated = normalize_data_input(
            data,
            title=self.title,
            replace_title=True,
        )
        self.title = title
        if pixel_size is not None:
            self.pixel_size = float(pixel_size)
        if k_pixel_size is not None and k_pixel_size > 0:
            self.k_pixel_size = float(k_pixel_size)
            self.k_calibrated = True
            if metadata_calibrated:
                self.calibration_source = "metadata"
        self._ingest_data(data)
        self.frame_idx = min(self.frame_idx, self.n_frames - 1)
        self.spots = []
        self.rings = []
        self.auto_detect_center()
        self._update_frame()
        self._bake_offline_frames()
        return self

    def state_dict(self):
        """Return the persistable widget state as a plain dict."""
        state = {}
        for field in self._STATE_FIELDS:
            value = getattr(self, field)
            state[field] = list(value) if field in self._LIST_STATE_FIELDS else value
        return state

    def save(self, path: str):
        """Write the widget state to a JSON file."""
        save_state_file(path, "ShowDiffraction", self.state_dict())

    def collapse_controls(self) -> Self:
        """Collapse controls behind the frontend ``Controls`` button."""
        self.controls_collapsed = True
        return self

    def expand_controls(self) -> Self:
        """Expand frontend controls when ``show_controls`` is enabled."""
        self.controls_collapsed = False
        return self

    def toggle_controls(self) -> Self:
        """Toggle whether frontend controls start collapsed."""
        self.controls_collapsed = not bool(self.controls_collapsed)
        return self

    def load_state_dict(self, state):
        """Restore widget state from a dict; unknown keys are ignored."""
        for key, val in state.items():
            # State restore
            if key in self._STATE_FIELDS:
                setattr(self, key, val)

    def summary(self):
        """Print a text summary of calibration, spots, rings, and indexing."""
        lines = [self.title or "ShowDiffraction", "═" * 32]
        lines.append(f"Frames:   {self.n_frames} (showing #{self.frame_idx})")
        k_unit = "1/Å" if self.k_calibrated else "px"
        k_val = f"{self.k_pixel_size:.4f}" if self.k_calibrated else "uncalibrated"
        lines.append(f"Detector: {self.det_rows}×{self.det_cols} ({k_val} {k_unit}/px)")
        if self.k_calibrated:
            source = {
                "from_phase": "phase",
                "from_ring": "ring",
                "from_spot": "spot",
            }.get(self.calibration_source, self.calibration_source)
            cal = f"Calibration: {source}"
            if self.calibration_ref_d > 0:
                cal += (
                    f" (d={self.calibration_ref_d:.3f} Å @ r={self.calibration_ref_radius:.1f} px)"
                )
            elif self.calibration_source == "from_phase":
                cal += f" (rms {self.calibration_rms_px:.2f} px)"
            lines.append(cal)
        if self.ellipse_ratio != 1.0:
            state = "corrected" if self.ellipse_corrected else "not corrected"
            lines.append(
                f"Ellipse:  a/b={self.ellipse_ratio:.3f} @ {self.ellipse_angle:.1f}° ({state})"
            )
        lines.append(
            f"Center:   ({self.center_row:.1f}, {self.center_col:.1f})  "
            f"BF r={self.bf_radius:.1f} px"
        )
        lines.append(f"Spots:    {len(self.spots)}")
        if self.spots:
            for s in self.spots[:5]:
                if s.get("d_spacing"):
                    derr = s.get("d_spacing_err")
                    d = f"{s['d_spacing']:.3f}±{derr:.3f} Å" if derr else f"{s['d_spacing']:.3f} Å"
                else:
                    d = f"{s['r_pixels']:.1f} px"
                ang = f"  ∠={s['angle_deg']:.1f}°" if s.get("angle_deg") is not None else ""
                hkl = f"  {s['hkl']}" if s.get("hkl") else ""
                lines.append(f"  #{s['id']} ({s['row']:.1f}, {s['col']:.1f}) d={d}{ang}{hkl}")
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
        k_unit = "1/Å" if self.k_calibrated else "px"
        shape = f"({self.n_frames}, {self.det_rows}, {self.det_cols})"
        title_info = f", title='{self.title}'" if self.title else ""
        spots_info = f", spots={len(self.spots)}" if self.spots else ""
        return (
            f"ShowDiffraction(shape={shape}, "
            f"sampling=({self.pixel_size} Å, {self.k_pixel_size} {k_unit}), "
            f"frame={self.frame_idx}/{self.n_frames}{spots_info}{title_info})"
        )

    def free(self):
        """Free GPU memory held by this widget."""
        if hasattr(self, "_data"):
            del self._data
        import gc

        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
