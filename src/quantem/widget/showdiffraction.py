"""
showdiffraction: Interactive d-spacing analysis for 2D/3D diffraction patterns.

Pick Bragg spots and Debye-Scherrer rings on a diffraction pattern to read off
d-spacings and inter-spot angles, with sub-pixel Gaussian peak refinement. The
detector center can be set manually or auto-detected from the bright-field disk,
and k-space calibration (1/Å per pixel) is taken from metadata or solved from a
spot/ring of known d-spacing.

A single 2D pattern is shown as a one-frame stack; a 3D ``(N, H, W)`` array is a
simple N-frame stack scrubbed with ``frame_idx``. 4D-STEM data is not handled
here -- use the separate Show4DSTEM widget for that.
"""

import csv
import json
import math
import pathlib
import tempfile
import time
from typing import Self

import anywidget
import numpy as np
import torch
import traitlets

from quantem.widget.utils.array import to_numpy
from quantem.widget.export import ensure_mobile_viewport
from quantem.widget.utils.state_io import resolve_widget_version, save_state_file, unwrap_state_payload

# ============================================================================
# Constants
# ============================================================================
DEFAULT_BF_RATIO = 0.125  # BF disk radius as fraction of detector size (1/8)


class ShowDiffraction(anywidget.AnyWidget):
    """
    Interactive d-spacing analysis for 2D/3D diffraction patterns.

    Pick Bragg spots and rings on the diffraction pattern to measure d-spacings,
    g-vectors, and inter-spot angles, with optional sub-pixel Gaussian refinement.
    Works with a single 2D pattern (SAED) or a 3D stack of patterns, and accepts
    NumPy arrays, PyTorch tensors, or quantem datasets. 4D-STEM stacks are not
    supported here; use Show4DSTEM instead.

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
    show_stats : bool, default True
        Show statistics (mean, min, max, std).
    show_controls : bool, default True
        Show the control panel.
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

    # =========================================================================
    # Core State / Frame Stack + Detector
    # =========================================================================
    widget_version = traitlets.Unicode("unknown").tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    n_frames = traitlets.Int(1).tag(sync=True)
    frame_idx = traitlets.Int(0).tag(sync=True)
    det_rows = traitlets.Int(1).tag(sync=True)
    det_cols = traitlets.Int(1).tag(sync=True)

    frame_bytes = traitlets.Bytes(b"").tag(sync=True)
    # whole stack as float32, baked only when offline so the kernel-less HTML can scrub
    # frames client-side (live widgets stay empty and stream frame_bytes per frame).
    offline_frames = traitlets.Bytes(b"").tag(sync=True)

    # Offline/export render flag. The frontend forces a light background when set
    # so standalone HTML exports read on any OS theme. Frames are always embedded
    # as exact float32.
    offline = traitlets.Bool(False).tag(sync=True)

    # =========================================================================
    # Standalone HTML export bridge (see quantem.widget.export protocol)
    # =========================================================================
    export_request = traitlets.Unicode("").tag(sync=True)
    export_status = traitlets.Unicode("").tag(sync=True)
    export_enabled = traitlets.Bool(True).tag(sync=True)
    export_payload = traitlets.Bytes(b"").tag(sync=True)
    export_payload_id = traitlets.Unicode("").tag(sync=True)
    export_filename = traitlets.Unicode("").tag(sync=True)

    # =========================================================================
    # Detector Calibration
    # =========================================================================
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

    # =========================================================================
    # Spots & Rings
    # =========================================================================
    spots = traitlets.List(traitlets.Dict()).tag(sync=True)
    snap_enabled = traitlets.Bool(False).tag(sync=True)
    snap_radius = traitlets.Int(5).tag(sync=True)

    rings = traitlets.List(traitlets.Dict()).tag(sync=True)

    spot_refine = traitlets.Bool(True).tag(sync=True)

    # =========================================================================
    # Frontend request channel
    # =========================================================================
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
    _detect_spots_request = traitlets.Int(0).tag(sync=True)  # carries max_spots
    _detect_rings_request = traitlets.Int(0).tag(sync=True)  # carries max_rings
    _spot_remove_request = traitlets.Int(0).tag(sync=True)  # carries spot id (0 = none)
    _ring_remove_request = traitlets.Int(0).tag(sync=True)  # carries ring id (0 = none)

    # =========================================================================
    # Display
    # =========================================================================
    dp_colormap = traitlets.Unicode("inferno").tag(sync=True)
    dp_scale_mode = traitlets.Unicode("log").tag(sync=True)
    dp_invert = traitlets.Bool(False).tag(sync=True)
    dp_vmin_pct = traitlets.Float(0.0).tag(sync=True)
    dp_vmax_pct = traitlets.Float(100.0).tag(sync=True)

    # =========================================================================
    # Statistics
    # =========================================================================
    dp_stats = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0, 0.0, 0.0]).tag(
        sync=True
    )

    # =========================================================================
    # UI Visibility
    # =========================================================================
    show_stats = traitlets.Bool(True).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)
    panel_width_px = traitlets.Int(384).tag(sync=True)

    @traitlets.validate("center_mode")
    def _validate_center_mode(self, proposal):
        val = proposal["value"]
        allowed = ("auto", "manual")
        if val not in allowed:
            raise ValueError(f"center_mode must be one of {allowed}, got {val!r}")
        return val

    @traitlets.validate("frame_idx")
    def _validate_frame_idx(self, proposal):
        # Clamp to [0, n_frames) so stale indices don't IndexError on reload.
        val = int(proposal["value"])
        n = max(1, int(self.n_frames))
        return max(0, min(val, n - 1))

    @traitlets.validate("dp_scale_mode")
    def _validate_dp_scale_mode(self, proposal):
        val = proposal["value"]
        allowed = ("linear", "log", "sqrt")
        if val not in allowed:
            raise ValueError(f"dp_scale_mode must be one of {allowed}, got {val!r}")
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
        show_stats: bool = True,
        show_controls: bool = True,
        panel_width_px: int | None = None,
        offline: bool = False,
        verbose: bool = True,
        state=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        t_start = time.perf_counter()
        self.widget_version = resolve_widget_version()

        if hasattr(data, "_fields") and "data" in getattr(data, "_fields", ()):
            meta = data.metadata or {}
            if pixel_size is None and meta.get("pixel_size") is not None:
                pixel_size = meta.get("pixel_size")
            data = data.data

        k_calibrated = False
        if hasattr(data, "sampling") and hasattr(data, "array"):
            if not title and hasattr(data, "name") and data.name:
                title = str(data.name)
            units = list(getattr(data, "units", ["pixels"] * 4))
            if pixel_size is None and units and units[0] in ("Å", "angstrom", "A", "nm"):
                pixel_size = float(data.sampling[0])
                if units[0] == "nm":
                    pixel_size *= 10
            if k_pixel_size is None and len(units) > 2 and units[2] in ("1/Å", "1/A"):
                k_pixel_size = float(data.sampling[2])
                k_calibrated = True
            data = data.array

        self._device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self._ingest_data(data)

        if pixel_size is not None:
            self.pixel_size = float(pixel_size)
        if k_pixel_size is not None and k_pixel_size > 0:
            self.k_pixel_size = float(k_pixel_size)
            self.k_calibrated = True
            self.calibration_source = "manual"
        elif k_calibrated:
            self.k_calibrated = True
            self.calibration_source = "metadata"

        self.title = title
        self.dp_scale_mode = dp_scale_mode
        self.snap_enabled = snap_enabled
        self.snap_radius = snap_radius
        self.spot_refine = spot_refine
        self.show_stats = show_stats
        self.show_controls = show_controls
        if panel_width_px is not None:
            self.panel_width_px = int(panel_width_px)
        self.offline = offline

        if center is not None:
            self.center_row = float(center[0])
            self.center_col = float(center[1])
        else:
            self.center_row = float(self.det_rows / 2)
            self.center_col = float(self.det_cols / 2)

        if bf_radius is not None:
            self.bf_radius = float(bf_radius)
        else:
            self.bf_radius = min(self.det_rows, self.det_cols) * DEFAULT_BF_RATIO

        if center is None and bf_radius is None:
            self.auto_detect_center()

        self._update_frame()
        self._bake_offline_frames()

        self.observe(self._update_frame, names=["frame_idx"])
        self.observe(self._bake_offline_frames, names=["offline"])
        self.observe(self._on_spot_add_request, names=["_spot_add_request"])
        self.observe(self._on_spot_undo_request, names=["_spot_undo_request"])
        self.observe(self._on_spot_clear_request, names=["_spot_clear_request"])
        self.observe(self._on_ring_add_request, names=["_ring_add_request"])
        self.observe(self._on_ring_undo_request, names=["_ring_undo_request"])
        self.observe(self._on_ring_clear_request, names=["_ring_clear_request"])
        self.observe(
            self._on_calibrate_from_ring_request, names=["_calibrate_from_ring_request"]
        )
        self.observe(
            self._on_calibrate_from_spot_request, names=["_calibrate_from_spot_request"]
        )
        # Recompute derived quantities when center / calibration change.
        self.observe(
            self._on_geometry_change,
            names=["center_row", "center_col", "k_pixel_size", "k_calibrated"],
        )
        self.observe(self._on_detect_spots_request, names=["_detect_spots_request"])
        self.observe(self._on_detect_rings_request, names=["_detect_rings_request"])
        self.observe(self._on_spot_remove_request, names=["_spot_remove_request"])
        self.observe(self._on_ring_remove_request, names=["_ring_remove_request"])
        self.observe(self._on_export_request_change, names=["export_request"])

        if verbose:
            mem_mb = self._data.nelement() * 4 / 1e6
            print(f"  to {self._device}: {time.perf_counter() - t_start:.2f}s ({mem_mb:.1f} MB)")

        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(
                    json.loads(pathlib.Path(state).read_text()),
                    require_envelope=True,
                )
            else:
                state = unwrap_state_payload(state)
            self.load_state_dict(state)

    def _ingest_data(self, data):
        data_np = to_numpy(data)
        is_integer = np.issubdtype(data_np.dtype, np.integer)
        data_np = data_np.astype(np.float32)
        if data_np.size > 2**31 - 1 and self._device.type == "mps":
            self._device = torch.device("cpu")
        if is_integer:
            global_max = float(data_np.max())
            p999 = float(np.percentile(data_np, 99.9))
            if global_max > p999 * 5:
                data_np[data_np > p999 * 3] = 0
        ndim = data_np.ndim
        if ndim == 2:
            # Single 2D pattern (SAED): a one-frame stack.
            data_np = data_np[None, ...]
        elif ndim == 3:
            # A simple N-frame stack of patterns.
            pass
        elif ndim == 4:
            raise ValueError(
                "ShowDiffraction is for 2D/3D diffraction patterns; "
                "use Show4DSTEM for 4D-STEM data."
            )
        else:
            raise ValueError(f"Expected a 2D or 3D array, got {ndim}D")
        self._det_shape = (data_np.shape[1], data_np.shape[2])
        self._data = torch.from_numpy(np.ascontiguousarray(data_np)).to(self._device)
        self.n_frames = int(data_np.shape[0])
        self.det_rows = self._det_shape[0]
        self.det_cols = self._det_shape[1]

    @property
    def detector_shape(self) -> tuple[int, int]:
        return self._det_shape

    def auto_detect_center(self) -> Self:
        """Auto-detect BF disk center and radius from the summed diffraction stack."""
        summed_dp = self._data.sum(dim=0)

        threshold = summed_dp.mean() + summed_dp.std()
        mask = summed_dp > threshold

        total = mask.sum()
        if total == 0:
            return self

        row_coords = torch.arange(self.det_rows, device=self._device, dtype=torch.float32)[
            :, None
        ]
        col_coords = torch.arange(self.det_cols, device=self._device, dtype=torch.float32)[
            None, :
        ]
        self.center_row = float((row_coords * mask).sum() / total)
        self.center_col = float((col_coords * mask).sum() / total)
        # central beam only: rings are also above threshold, so whole-mask area would
        # overestimate the radius and later mask out the inner rings in detect_rings.
        self.bf_radius = self._central_beam_radius(mask, self.center_row, self.center_col)
        self.center_mode = "auto"
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
            # center is dark (beam stop): use the nearest bright component
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
        # skip single frames and live widgets (which stream per frame) to avoid bloat
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
        r_pixels = math.hypot(d_row, d_col)

        # Project the centroid uncertainty onto the radial direction.
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
            # d = 1/(k r) ⇒ σ_d/d = σ_g/g = σ_r/r (calibration treated as exact).
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

    def _fit_gaussian_2d(self, row: float, col: float) -> dict | None:
        frame = self._displayed_frame()
        half = max(4, int(self.snap_radius))
        r0, c0 = int(round(row)), int(round(col))
        r_lo, r_hi = max(0, r0 - half), min(self.det_rows, r0 + half + 1)
        c_lo, c_hi = max(0, c0 - half), min(self.det_cols, c0 + half + 1)
        patch = frame[r_lo:r_hi, c_lo:c_hi].astype(np.float64)
        if patch.shape[0] < 5 or patch.shape[1] < 5:
            return None
        try:
            from scipy.optimize import OptimizeWarning, curve_fit
        except Exception:
            return None

        ny, nx = patch.shape
        rr, cc = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")

        def gauss2d(coords, amp, fr, fc, sr, sc, off):
            r, c = coords
            return (amp * np.exp(-0.5 * (((r - fr) / sr) ** 2 + ((c - fc) / sc) ** 2)) + off).ravel()

        peak = np.unravel_index(int(np.argmax(patch)), patch.shape)
        p0 = (
            float(patch.max() - patch.min()),
            float(peak[0]),
            float(peak[1]),
            2.0,
            2.0,
            float(patch.min()),
        )
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                popt, pcov = curve_fit(gauss2d, (rr, cc), patch.ravel(), p0=p0, maxfev=5000)
        except Exception:
            return None
        _, fr, fc, sigma_row, sigma_col, _ = popt
        if not (0 <= fr < ny and 0 <= fc < nx):
            return None

        perr = np.sqrt(np.abs(np.diag(pcov)))
        residual = patch.ravel() - gauss2d((rr, cc), *popt)
        ss_res = float(np.sum(residual**2))
        ss_tot = float(np.sum((patch.ravel() - patch.mean()) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            "row": float(r_lo + fr),
            "col": float(c_lo + fc),
            "row_err": float(perr[1]) if np.isfinite(perr[1]) else 0.0,
            "col_err": float(perr[2]) if np.isfinite(perr[2]) else 0.0,
            "sigma_row": float(abs(sigma_row)),
            "sigma_col": float(abs(sigma_col)),
            "fit_quality": float(r_squared),
        }

    def _with_angles(self, spots) -> list:
        # Angles are measured relative to the first spot.
        if not spots:
            return spots
        ref = spots[0]
        cr, cc = self.center_row, self.center_col
        ref_dr, ref_dc = ref["row"] - cr, ref["col"] - cc
        ref_r = math.hypot(ref_dr, ref_dc)
        ref_perp = math.hypot(ref.get("row_err", 0.0), ref.get("col_err", 0.0))
        out = []
        for s in spots:
            d_row, d_col = s["row"] - cr, s["col"] - cc
            r = math.hypot(d_row, d_col)
            if ref_r > 0 and r > 0:
                cos_a = max(-1.0, min(1.0, (ref_dr * d_row + ref_dc * d_col) / (ref_r * r)))
                angle = math.degrees(math.acos(cos_a))
                perp = math.hypot(s.get("row_err", 0.0), s.get("col_err", 0.0))
                angle_err = math.degrees(math.hypot(perp / r, ref_perp / ref_r))
            else:
                angle = None
                angle_err = None
            out.append({**s, "angle_deg": angle, "angle_deg_err": angle_err})
        return out

    def detect_spots(
        self,
        max_spots: int = 20,
        min_distance: int = 6,
        threshold_rel: float = 0.15,
        exclude_radius: float | None = None,
        replace: bool = True,
    ) -> Self:
        """Auto-detect Bragg spots as local maxima in the current frame.

        The saturated central beam is log-compressed and high-passed so its halo
        does not dominate, peaks within ``exclude_radius`` of the center are
        dropped, and the strongest remaining peaks are kept.

        Parameters
        ----------
        max_spots : int, default 20
            Maximum number of spots to keep, ordered by prominence.
        min_distance : int, default 6
            Minimum separation in pixels between detected peaks.
        threshold_rel : float, default 0.15
            Relative prominence threshold; higher keeps fewer, stronger peaks.
        exclude_radius : float, optional
            Radius in pixels around the center to ignore. Defaults to the larger
            of ``bf_radius`` and ``2 * min_distance``.
        replace : bool, default True
            Clear existing spots before adding the detected ones.

        Returns
        -------
        Self
            The widget, for chaining.
        """
        frame = self._displayed_frame().astype(np.float64)
        n_rows, n_cols = frame.shape
        if exclude_radius is None:
            exclude_radius = max(self.bf_radius, 2.0 * float(min_distance))
        try:
            from scipy.ndimage import gaussian_filter, maximum_filter
        except Exception:
            return self
        # Log-compress the saturated beam, then high-pass to flatten its halo so spots stand out.
        work = np.log1p(np.clip(frame - frame.min(), 0.0, None))
        work = work - gaussian_filter(work, sigma=max(2.0, float(min_distance)))

        size = max(3, int(min_distance) | 1)  # odd window enforces a min separation
        local_max = maximum_filter(work, size=size) == work
        rows = np.arange(n_rows)[:, None]
        cols = np.arange(n_cols)[None, :]
        radius = np.hypot(rows - self.center_row, cols - self.center_col)
        local_max &= radius > float(exclude_radius)
        local_max[0, :] = local_max[-1, :] = False
        local_max[:, 0] = local_max[:, -1] = False
        coords = np.argwhere(local_max)
        if replace:
            self.clear_spots()
        if coords.size == 0:
            return self
        # Threshold on prominence (high-passed value), robust to the saturated beam.
        prominence = work[coords[:, 0], coords[:, 1]]
        positive = prominence[prominence > 0]
        if positive.size:
            level = float(positive.mean() + (threshold_rel / 0.15) * positive.std())
            keep = prominence >= level
            coords, prominence = coords[keep], prominence[keep]
        order = np.argsort(-prominence)[: int(max_spots)]
        for r0, c0 in coords[order]:
            self.add_spot(float(r0), float(c0))
        return self

    def detect_rings(
        self,
        max_rings: int = 10,
        prominence_rel: float = 0.05,
        min_separation: int = 5,
        exclude_radius: float | None = None,
        replace: bool = True,
    ) -> Self:
        """Auto-detect Debye-Scherrer rings as peaks in the radial profile.

        The radial profile is log-compressed and detrended so rings read as
        peaks, peaks inside ``exclude_radius`` are dropped, and the innermost
        (low-order) rings are kept. Use this instead of ``detect_spots`` for
        polycrystalline / powder patterns.

        Parameters
        ----------
        max_rings : int, default 10
            Maximum number of rings to keep.
        prominence_rel : float, default 0.05
            Peak prominence as a fraction of the detrended profile span.
        min_separation : int, default 5
            Minimum separation in radial bins between detected peaks.
        exclude_radius : float, optional
            Radius in pixels around the center to ignore. Defaults to ``bf_radius``.
        replace : bool, default True
            Clear existing rings before adding the detected ones.

        Returns
        -------
        Self
            The widget, for chaining.
        """
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
        # Log-compress the steep central-beam falloff, then detrend so rings read as peaks.
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
        # strongest, not innermost: a noise bump in a dark gap can sit inside a real ring
        # but has far lower prominence. Keep the top max_rings, then order inner -> outer.
        strongest = np.argsort(prominences)[::-1][: int(max_rings)]
        for p in sorted(peaks[strongest]):
            self.add_ring(float(radii_px[p]))
        return self

    def _on_detect_spots_request(self, change=None):
        n = self._detect_spots_request
        if n and n > 0:
            self.detect_spots(max_spots=int(n))
            self._detect_spots_request = 0

    def _on_detect_rings_request(self, change=None):
        n = self._detect_rings_request
        if n and n > 0:
            self.detect_rings(max_rings=int(n))
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

    def add_spot(self, row: float, col: float) -> Self:
        """Add a spot at (row, col). Sub-pixel Gaussian refine if spot_refine, else snap if enabled."""
        raw_row, raw_col = float(row), float(col)
        row_err = col_err = 0.0
        fit_quality = None
        if self.spot_refine:
            fit = self._fit_gaussian_2d(raw_row, raw_col)
            if fit is not None:
                row, col = fit["row"], fit["col"]
                row_err, col_err = fit["row_err"], fit["col_err"]
                fit_quality = fit["fit_quality"]
        elif self.snap_enabled:
            row, col = self._snap_to_peak(raw_row, raw_col)
        info = self._compute_spot_info(row, col, row_err=row_err, col_err=col_err)
        spot = {
            "id": (max(s["id"] for s in self.spots) + 1) if self.spots else 1,
            "row": float(row),
            "col": float(col),
            "raw_row": raw_row,
            "raw_col": raw_col,
            "row_err": float(row_err),
            "col_err": float(col_err),
            "fit_quality": fit_quality,
            "angle_deg": None,
            "angle_deg_err": None,
            "hkl": "",
            "note": "",
            **info,
        }
        self.spots = self._with_angles(list(self.spots) + [spot])
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
        # Center/calibration moved → existing spot and ring d-spacings are stale.
        self._recompute_spots()
        self._recompute_rings()

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
            "id": (max(r["id"] for r in self.rings) + 1) if self.rings else 1,
            "hkl": "",
            "note": "",
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

    def _radial_profile(
        self,
        *,
        n_bins: int | None = None,
        max_radius: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Azimuthally averaged radial intensity profile of the current frame.

        Internal helper for ring detection and ring intensity readout. Bins the
        displayed frame by distance (in pixels) from the center and averages the
        intensity in each radial bin.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(radii_px, intensity)`` as float32 arrays, with the radial axis in
            pixels.
        """
        frame = self._displayed_frame()
        n_rows, n_cols = frame.shape
        center_row = float(self.center_row)
        center_col = float(self.center_col)

        if max_radius is None:
            max_radius = float(
                min(center_row, center_col, (n_rows - 1) - center_row, (n_cols - 1) - center_col)
            )
        max_radius = float(max(1.0, max_radius))

        if n_bins is None:
            n_bins = max(1, int(round(max_radius)))
        n_bins = int(max(1, n_bins))

        rows = np.arange(n_rows, dtype=np.float64)[:, None]
        cols = np.arange(n_cols, dtype=np.float64)[None, :]
        radii = np.sqrt((rows - center_row) ** 2 + (cols - center_col) ** 2)
        flat_r = radii.ravel()
        flat_i = frame.astype(np.float64).ravel()

        edges = np.linspace(0.0, max_radius, n_bins + 1)
        idx = np.digitize(flat_r, edges) - 1
        inside = (idx >= 0) & (idx < n_bins)
        idx = idx[inside]
        vals = flat_i[inside]

        counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
        sums = np.bincount(idx, weights=vals, minlength=n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            intensity = np.where(counts > 0, sums / counts, 0.0)

        bin_centers_px = 0.5 * (edges[:-1] + edges[1:])
        return bin_centers_px.astype(np.float32), intensity.astype(np.float32)

    def calibrate_from_spot(self, row: float, col: float, d_known: float) -> Self:
        """Calibrate ``k_pixel_size`` from a spot of known d-spacing.

        Sets the k-space sampling so the spot at (row, col), measured from the
        current center, corresponds to a d-spacing of ``d_known``.

        Parameters
        ----------
        row, col : float
            Spot position in detector pixels.
        d_known : float
            Known d-spacing in Å (must be positive).

        Returns
        -------
        Self
            The widget, for chaining.

        Raises
        ------
        ValueError
            If ``d_known`` is not positive or the spot lies at the center.
        """
        if d_known <= 0:
            raise ValueError(f"d_known must be positive, got {d_known}")
        r_pixels = math.hypot(row - self.center_row, col - self.center_col)
        if r_pixels <= 0:
            raise ValueError("calibration point is at the center; no g-vector")
        self.k_pixel_size = 1.0 / (d_known * r_pixels)
        self.k_calibrated = True
        self.calibration_source = "from_spot"
        self.calibration_ref_d = float(d_known)
        self.calibration_ref_radius = float(r_pixels)
        return self

    def calibrate_from_ring(self, radius_px: float, d_known: float) -> Self:
        """Calibrate ``k_pixel_size`` from a ring of known d-spacing.

        Sets the k-space sampling so a ring at ``radius_px`` from the center
        corresponds to a d-spacing of ``d_known``.

        Parameters
        ----------
        radius_px : float
            Ring radius in detector pixels (must be positive).
        d_known : float
            Known d-spacing in Å (must be positive).

        Returns
        -------
        Self
            The widget, for chaining.

        Raises
        ------
        ValueError
            If ``d_known`` or ``radius_px`` is not positive.
        """
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

    _MEASUREMENT_COLUMNS = [
        "id", "kind", "raw_row", "raw_col", "row", "col", "row_err", "col_err",
        "r_pixels", "r_pixels_err", "g_inv_angstrom", "g_inv_angstrom_err",
        "d_angstrom", "d_angstrom_err", "angle_deg", "angle_deg_err",
        "intensity", "fit_quality", "hkl", "note",
    ]

    @staticmethod
    def _build_measurement_records(spots, rings) -> list:
        # Flatten spots and rings into unified measurement rows.
        records = []
        for s in spots:
            records.append({
                "id": s.get("id"),
                "kind": "spot",
                "raw_row": s.get("raw_row"),
                "raw_col": s.get("raw_col"),
                "row": s.get("row"),
                "col": s.get("col"),
                "row_err": s.get("row_err"),
                "col_err": s.get("col_err"),
                "r_pixels": s.get("r_pixels"),
                "r_pixels_err": s.get("r_pixels_err"),
                "g_inv_angstrom": s.get("g_magnitude"),
                "g_inv_angstrom_err": s.get("g_magnitude_err"),
                "d_angstrom": s.get("d_spacing"),
                "d_angstrom_err": s.get("d_spacing_err"),
                "angle_deg": s.get("angle_deg"),
                "angle_deg_err": s.get("angle_deg_err"),
                "intensity": s.get("intensity"),
                "fit_quality": s.get("fit_quality"),
                "hkl": s.get("hkl", ""),
                "note": s.get("note", ""),
            })
        for r in rings:
            records.append({
                "id": r.get("id"),
                "kind": "ring",
                "raw_row": None,
                "raw_col": None,
                "row": None,
                "col": None,
                "row_err": None,
                "col_err": None,
                "r_pixels": r.get("radius_px"),
                "r_pixels_err": None,
                "g_inv_angstrom": r.get("g_magnitude"),
                "g_inv_angstrom_err": None,
                "d_angstrom": r.get("d_spacing"),
                "d_angstrom_err": None,
                "angle_deg": None,
                "angle_deg_err": None,
                "intensity": r.get("intensity"),
                "fit_quality": None,
                "hkl": r.get("hkl", ""),
                "note": r.get("note", ""),
            })
        return records

    def _measurement_records(self) -> list:
        return self._build_measurement_records(self.spots, self.rings)

    @staticmethod
    def _measurement_metadata(state) -> dict:
        # Provenance header for the measurement table.
        return {
            "widget_name": "ShowDiffraction",
            "center_row": state.get("center_row"),
            "center_col": state.get("center_col"),
            "k_pixel_size_inv_angstrom_per_px": state.get("k_pixel_size"),
            "calibrated": bool(state.get("k_calibrated")),
            "calibration_source": state.get("calibration_source", "none"),
            "calibration_ref_d_angstrom": state.get("calibration_ref_d", 0.0),
            "calibration_ref_radius_px": state.get("calibration_ref_radius", 0.0),
        }

    @staticmethod
    def _write_measurement_file(path, records, meta) -> pathlib.Path:
        # .json writes a {"metadata", "measurements"} document; anything else CSV.
        p = pathlib.Path(path)
        if p.suffix.lower() == ".json":
            p.write_text(json.dumps({"metadata": meta, "measurements": records}, indent=2))
        else:
            with open(p, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=ShowDiffraction._MEASUREMENT_COLUMNS)
                writer.writeheader()
                writer.writerows(records)
        return p

    def export_measurements(self, path: str) -> pathlib.Path:
        """Export the spot and ring measurements to a CSV or JSON file.

        The format is inferred from the file extension: ``.json`` writes a
        ``{"metadata": ..., "measurements": ...}`` document, anything else
        writes CSV with the columns in ``_MEASUREMENT_COLUMNS``. This table is
        fully contained in the saved state, so you do not need to keep it as a
        separate file -- ``measurements_from_state`` rebuilds it on demand.

        Parameters
        ----------
        path : str
            Output file path. A ``.json`` suffix selects JSON; otherwise CSV.

        Returns
        -------
        pathlib.Path
            The written file path.
        """
        return self._write_measurement_file(
            path, self._measurement_records(), self._measurement_metadata(self.state_dict())
        )

    @classmethod
    def measurements_from_state(cls, state, path=None):
        """Rebuild the spot/ring measurement table from a saved state.

        The saved state already holds every spot and ring, so the measurement
        table is derived from it -- there is no need to keep a separate CSV/JSON
        export next to the state file. Regenerate it whenever needed, without
        loading the image data.

        Parameters
        ----------
        state : dict, str, or pathlib.Path
            A saved state file path, or an already-loaded state dict/envelope.
        path : str or pathlib.Path, optional
            Where to write the table (``.json`` selects JSON, otherwise CSV). If
            omitted, the list of measurement records is returned instead.

        Returns
        -------
        list[dict] or pathlib.Path
            The measurement records, or the written file path when ``path`` is set.
        """
        if isinstance(state, (str, pathlib.Path)):
            state = unwrap_state_payload(
                json.loads(pathlib.Path(state).read_text()), require_envelope=True
            )
        else:
            state = unwrap_state_payload(state)
        records = cls._build_measurement_records(
            state.get("spots", []), state.get("rings", [])
        )
        if path is None:
            return records
        return cls._write_measurement_file(path, records, cls._measurement_metadata(state))

    def export_html(
        self,
        path: str | pathlib.Path | None = None,
        *,
        title: str | None = None,
        **options,
    ) -> pathlib.Path:
        """Write a standalone HTML viewer for this widget.

        The exported file mounts the live anywidget JS bundle with the current
        widget state (frames, center, calibration, spots, rings, display
        settings) and opens in any browser without a Jupyter kernel.

        ShowDiffraction always embeds the exact ``full`` float32 frames -- there
        is no gallery and no reduced encoding. The ``mode``/``encoding``/
        ``downsample`` keys are accepted via ``**options`` for cross-widget API
        compatibility but are treated as no-ops.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            Destination HTML path. Defaults to a slug derived from the title.
        title : str, optional
            Browser page title. Defaults to the widget ``title`` or
            ``"ShowDiffraction"``.
        **options
            Accepted and ignored (compatibility with the HTML export protocol).

        Returns
        -------
        pathlib.Path
            The written file path.
        """
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
        if hasattr(data, "_fields") and "data" in getattr(data, "_fields", ()):
            meta = data.metadata or {}
            if meta.get("pixel_size") is not None:
                self.pixel_size = float(meta.get("pixel_size"))
            data = data.data
        if hasattr(data, "sampling") and hasattr(data, "array"):
            units = list(getattr(data, "units", ["pixels"] * 4))
            if units and units[0] in ("Å", "angstrom", "A", "nm"):
                px = float(data.sampling[0])
                if units[0] == "nm":
                    px *= 10
                self.pixel_size = px
            if len(units) > 2 and units[2] in ("1/Å", "1/A"):
                self.k_pixel_size = float(data.sampling[2])
                self.k_calibrated = True
            if hasattr(data, "name") and data.name:
                self.title = str(data.name)
            data = data.array
        self._ingest_data(data)
        self.frame_idx = min(self.frame_idx, self.n_frames - 1)
        self.spots = []
        self.rings = []
        self.auto_detect_center()
        self._update_frame()
        self._bake_offline_frames()
        return self

    def state_dict(self):
        return {
            "title": self.title,
            "frame_idx": self.frame_idx,
            "pixel_size": self.pixel_size,
            "k_pixel_size": self.k_pixel_size,
            "k_calibrated": self.k_calibrated,
            "center_row": self.center_row,
            "center_col": self.center_col,
            "bf_radius": self.bf_radius,
            "spots": list(self.spots),
            "rings": list(self.rings),
            "snap_enabled": self.snap_enabled,
            "snap_radius": self.snap_radius,
            "spot_refine": self.spot_refine,
            "center_mode": self.center_mode,
            "calibration_source": self.calibration_source,
            "calibration_ref_d": self.calibration_ref_d,
            "calibration_ref_radius": self.calibration_ref_radius,
            "dp_colormap": self.dp_colormap,
            "dp_scale_mode": self.dp_scale_mode,
            "dp_invert": self.dp_invert,
            "dp_vmin_pct": self.dp_vmin_pct,
            "dp_vmax_pct": self.dp_vmax_pct,
            "show_stats": self.show_stats,
            "show_controls": self.show_controls,
        }

    def save(self, path: str):
        save_state_file(path, "ShowDiffraction", self.state_dict())

    def load_state_dict(self, state):
        allowed_keys = set(self.state_dict().keys())
        for key, val in state.items():
            # frame_idx is clamped by its validator against the current stack.
            if key in allowed_keys:
                setattr(self, key, val)

    def summary(self):
        lines = [self.title or "ShowDiffraction", "═" * 32]
        lines.append(f"Frames:   {self.n_frames} (showing #{self.frame_idx})")
        k_unit = "1/Å" if self.k_calibrated else "px"
        k_val = f"{self.k_pixel_size:.4f}" if self.k_calibrated else "uncalibrated"
        lines.append(f"Detector: {self.det_rows}×{self.det_cols} ({k_val} {k_unit}/px)")
        if self.k_calibrated:
            cal = f"Calib:    {self.calibration_source}"
            if self.calibration_ref_d > 0:
                cal += f" (d={self.calibration_ref_d:.3f} Å @ r={self.calibration_ref_radius:.1f} px)"
            lines.append(cal)
        lines.append(
            f"Center:   ({self.center_row:.1f}, {self.center_col:.1f})  BF r={self.bf_radius:.1f} px"
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
        if hasattr(self, "_data"):
            del self._data
        import gc

        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
