"""showdiffraction: Interactive d-spacing analysis for 2D/3D/4D diffraction patterns."""

import csv
import json
import math
import pathlib
import time
from typing import List, Optional, Self

import anywidget
import numpy as np
import torch
import traitlets

from quantem.widget.array_utils import to_numpy
from quantem.widget.io import IOResult
from quantem.widget.json_state import resolve_widget_version, save_state_file, unwrap_state_payload
from quantem.widget.tool_parity import (
    bind_tool_runtime_api,
    build_tool_groups,
    normalize_tool_groups,
)

DEFAULT_BF_RATIO = 0.125

def circle_center_from_3pts(p1, p2, p3, eps: float = 1e-9) -> tuple[float, float]:
    """Center (row, col) of the circle through three (row, col) points; used for ring centering."""
    r1, c1 = float(p1[0]), float(p1[1])
    r2, c2 = float(p2[0]), float(p2[1])
    r3, c3 = float(p3[0]), float(p3[1])
    d = 2.0 * (r1 * (c2 - c3) + r2 * (c3 - c1) + r3 * (c1 - c2))
    if abs(d) < eps:
        raise ValueError("three points are nearly collinear; cannot fit a ring center")
    s1 = r1 * r1 + c1 * c1
    s2 = r2 * r2 + c2 * c2
    s3 = r3 * r3 + c3 * c3
    center_row = (s1 * (c2 - c3) + s2 * (c3 - c1) + s3 * (c1 - c2)) / d
    center_col = (s1 * (r3 - r2) + s2 * (r1 - r3) + s3 * (r2 - r1)) / d
    return float(center_row), float(center_col)

class ShowDiffraction(anywidget.AnyWidget):
    """Interactive d-spacing analysis for 2D/3D/4D diffraction patterns."""

    _esm = pathlib.Path(__file__).parent / "static" / "showdiffraction.js"
    _css = pathlib.Path(__file__).parent / "static" / "showdiffraction.css"

    widget_version = traitlets.Unicode("unknown").tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    pos_row = traitlets.Int(0).tag(sync=True)
    pos_col = traitlets.Int(0).tag(sync=True)
    shape_rows = traitlets.Int(1).tag(sync=True)
    shape_cols = traitlets.Int(1).tag(sync=True)
    det_rows = traitlets.Int(1).tag(sync=True)
    det_cols = traitlets.Int(1).tag(sync=True)
    is_2d = traitlets.Bool(False).tag(sync=True)

    frame_bytes = traitlets.Bytes(b"").tag(sync=True)
    virtual_image_bytes = traitlets.Bytes(b"").tag(sync=True)

    center_row = traitlets.Float(0.0).tag(sync=True)
    center_col = traitlets.Float(0.0).tag(sync=True)
    bf_radius = traitlets.Float(0.0).tag(sync=True)
    pixel_size = traitlets.Float(1.0).tag(sync=True)
    k_pixel_size = traitlets.Float(0.0).tag(sync=True)
    k_calibrated = traitlets.Bool(False).tag(sync=True)

    dp_global_min = traitlets.Float(0.0).tag(sync=True)
    dp_global_max = traitlets.Float(1.0).tag(sync=True)

    spots = traitlets.List(traitlets.Dict()).tag(sync=True)
    snap_enabled = traitlets.Bool(False).tag(sync=True)
    snap_radius = traitlets.Int(5).tag(sync=True)

    rings = traitlets.List(traitlets.Dict()).tag(sync=True)

    spot_refine = traitlets.Bool(True).tag(sync=True)
    reference_spot_id = traitlets.Int(-1).tag(sync=True)  # -1 → first spot

    center_mode = traitlets.Unicode("auto").tag(sync=True)

    calibration_source = traitlets.Unicode("none").tag(sync=True)
    calibration_ref_d = traitlets.Float(0.0).tag(sync=True)
    calibration_ref_radius = traitlets.Float(0.0).tag(sync=True)

    show_radial = traitlets.Bool(False).tag(sync=True)
    radial_g_bytes = traitlets.Bytes(b"").tag(sync=True)
    radial_i_bytes = traitlets.Bytes(b"").tag(sync=True)
    radial_n_bins = traitlets.Int(0).tag(sync=True)
    radial_calibrated = traitlets.Bool(False).tag(sync=True)

    _spot_add_request = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    _spot_undo_request = traitlets.Bool(False).tag(sync=True)
    _spot_clear_request = traitlets.Bool(False).tag(sync=True)
    _ring_add_request = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    _ring_undo_request = traitlets.Bool(False).tag(sync=True)
    _ring_clear_request = traitlets.Bool(False).tag(sync=True)
    _center_from_points_request = traitlets.List(traitlets.Float(), default_value=[]).tag(
        sync=True
    )
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

    dp_colormap = traitlets.Unicode("inferno").tag(sync=True)
    dp_scale_mode = traitlets.Unicode("log").tag(sync=True)
    dp_invert = traitlets.Bool(False).tag(sync=True)
    dp_rotation = traitlets.Int(0).tag(sync=True)  # display-only, degrees CCW
    dp_vmin_pct = traitlets.Float(0.0).tag(sync=True)
    dp_vmax_pct = traitlets.Float(100.0).tag(sync=True)
    vi_colormap = traitlets.Unicode("inferno").tag(sync=True)
    vi_vmin_pct = traitlets.Float(0.0).tag(sync=True)
    vi_vmax_pct = traitlets.Float(100.0).tag(sync=True)

    dp_stats = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0, 0.0, 0.0]).tag(
        sync=True
    )
    vi_stats = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0, 0.0, 0.0]).tag(
        sync=True
    )

    show_stats = traitlets.Bool(True).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)

    disabled_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)
    hidden_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)

    @classmethod
    def _normalize_tool_groups(cls, tool_groups) -> List[str]:
        return normalize_tool_groups("ShowDiffraction", tool_groups)

    @classmethod
    def _build_disabled_tools(
        cls,
        disabled_tools=None,
        disable_display: bool = False,
        disable_histogram: bool = False,
        disable_stats: bool = False,
        disable_navigation: bool = False,
        disable_view: bool = False,
        disable_export: bool = False,
        disable_spots: bool = False,
        disable_all: bool = False,
    ) -> List[str]:
        return build_tool_groups(
            "ShowDiffraction",
            tool_groups=disabled_tools,
            all_flag=disable_all,
            flag_map={
                "display": disable_display,
                "histogram": disable_histogram,
                "stats": disable_stats,
                "navigation": disable_navigation,
                "view": disable_view,
                "export": disable_export,
                "spots": disable_spots,
            },
        )

    @classmethod
    def _build_hidden_tools(
        cls,
        hidden_tools=None,
        hide_display: bool = False,
        hide_histogram: bool = False,
        hide_stats: bool = False,
        hide_navigation: bool = False,
        hide_view: bool = False,
        hide_export: bool = False,
        hide_spots: bool = False,
        hide_all: bool = False,
    ) -> List[str]:
        return build_tool_groups(
            "ShowDiffraction",
            tool_groups=hidden_tools,
            all_flag=hide_all,
            flag_map={
                "display": hide_display,
                "histogram": hide_histogram,
                "stats": hide_stats,
                "navigation": hide_navigation,
                "view": hide_view,
                "export": hide_export,
                "spots": hide_spots,
            },
        )

    @traitlets.validate("disabled_tools")
    def _validate_disabled_tools(self, proposal):
        return self._normalize_tool_groups(proposal["value"])

    @traitlets.validate("hidden_tools")
    def _validate_hidden_tools(self, proposal):
        return self._normalize_tool_groups(proposal["value"])

    @traitlets.validate("center_mode")
    def _validate_center_mode(self, proposal):
        val = proposal["value"]
        allowed = ("auto", "midpoint", "ring", "manual")
        if val not in allowed:
            raise ValueError(f"center_mode must be one of {allowed}, got {val!r}")
        return val

    @traitlets.validate("dp_scale_mode")
    def _validate_dp_scale_mode(self, proposal):
        val = proposal["value"]
        allowed = ("linear", "log", "sqrt", "power")
        if val not in allowed:
            raise ValueError(f"dp_scale_mode must be one of {allowed}, got {val!r}")
        return val

    @traitlets.validate("dp_rotation")
    def _validate_dp_rotation(self, proposal):
        val = int(proposal["value"]) % 360
        if val not in (0, 90, 180, 270):
            raise ValueError(f"dp_rotation must be one of 0/90/180/270, got {val}")
        return val

    def __init__(
        self,
        data,
        scan_shape: tuple[int, int] | None = None,
        k_pixel_size: float | None = None,
        pixel_size: float | None = None,
        center: tuple[float, float] | None = None,
        bf_radius: float | None = None,
        title: str = "",
        snap_enabled: bool = False,
        snap_radius: int = 5,
        spot_refine: bool = True,
        show_radial: bool = False,
        dp_scale_mode: str = "log",
        show_stats: bool = True,
        show_controls: bool = True,
        disabled_tools: Optional[List[str]] = None,
        disable_display: bool = False,
        disable_histogram: bool = False,
        disable_stats: bool = False,
        disable_navigation: bool = False,
        disable_view: bool = False,
        disable_export: bool = False,
        disable_spots: bool = False,
        disable_all: bool = False,
        hidden_tools: Optional[List[str]] = None,
        hide_display: bool = False,
        hide_histogram: bool = False,
        hide_stats: bool = False,
        hide_navigation: bool = False,
        hide_view: bool = False,
        hide_export: bool = False,
        hide_spots: bool = False,
        hide_all: bool = False,
        verbose: bool = True,
        state=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        t_start = time.perf_counter()
        self.widget_version = resolve_widget_version()

        if isinstance(data, IOResult):
            if not title and data.title:
                title = data.title
            if pixel_size is None and data.pixel_size is not None:
                pixel_size = data.pixel_size
            data = data.data

        k_calibrated = False
        if hasattr(data, "sampling") and hasattr(data, "array"):
            if not title and hasattr(data, "name") and data.name:
                title = str(data.name)
            units = getattr(data, "units", ["pixels"] * 4)
            if pixel_size is None and units[0] in ("Å", "angstrom", "A", "nm"):
                pixel_size = float(data.sampling[0])
                if units[0] == "nm":
                    pixel_size *= 10
            if k_pixel_size is None and units[2] in ("1/Å", "1/A"):
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
        self._ingest_data(data, scan_shape)

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
        self.show_radial = show_radial
        self.show_stats = show_stats
        self.show_controls = show_controls

        self.disabled_tools = self._build_disabled_tools(
            disabled_tools=disabled_tools,
            disable_display=disable_display,
            disable_histogram=disable_histogram,
            disable_stats=disable_stats,
            disable_navigation=disable_navigation,
            disable_view=disable_view,
            disable_export=disable_export,
            disable_spots=disable_spots,
            disable_all=disable_all,
        )
        self.hidden_tools = self._build_hidden_tools(
            hidden_tools=hidden_tools,
            hide_display=hide_display,
            hide_histogram=hide_histogram,
            hide_stats=hide_stats,
            hide_navigation=hide_navigation,
            hide_view=hide_view,
            hide_export=hide_export,
            hide_spots=hide_spots,
            hide_all=hide_all,
        )

        det_size = min(self.det_rows, self.det_cols)
        if center is not None:
            self.center_row = float(center[0])
            self.center_col = float(center[1])
        else:
            self.center_row = float(self.det_rows / 2)
            self.center_col = float(self.det_cols / 2)

        if bf_radius is not None:
            self.bf_radius = float(bf_radius)
        else:
            self.bf_radius = det_size * DEFAULT_BF_RATIO

        if center is None and bf_radius is None:
            self.auto_detect_center()

        self.pos_row = self._scan_shape[0] // 2
        self.pos_col = self._scan_shape[1] // 2

        self._compute_virtual_image()

        self._update_frame()
        self._update_radial()

        self.observe(self._on_position_change, names=["pos_row", "pos_col"])
        self.observe(self._on_spot_add_request, names=["_spot_add_request"])
        self.observe(self._on_spot_undo_request, names=["_spot_undo_request"])
        self.observe(self._on_spot_clear_request, names=["_spot_clear_request"])
        self.observe(self._on_ring_add_request, names=["_ring_add_request"])
        self.observe(self._on_ring_undo_request, names=["_ring_undo_request"])
        self.observe(self._on_ring_clear_request, names=["_ring_clear_request"])
        self.observe(
            self._on_center_from_points_request, names=["_center_from_points_request"]
        )
        self.observe(
            self._on_calibrate_from_ring_request, names=["_calibrate_from_ring_request"]
        )
        self.observe(
            self._on_calibrate_from_spot_request, names=["_calibrate_from_spot_request"]
        )
        # Recompute derived quantities when center / calibration / radial toggle change.
        self.observe(
            self._on_geometry_change,
            names=["center_row", "center_col", "k_pixel_size", "k_calibrated", "show_radial"],
        )
        self.observe(self._on_reference_spot_change, names=["reference_spot_id"])
        self.observe(self._on_detect_spots_request, names=["_detect_spots_request"])
        self.observe(self._on_detect_rings_request, names=["_detect_rings_request"])
        self.observe(self._on_spot_remove_request, names=["_spot_remove_request"])
        self.observe(self._on_ring_remove_request, names=["_ring_remove_request"])

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

    def _ingest_data(self, data, scan_shape=None):
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
            # Single 2D pattern (SAED): a 1×1 scan with one detector frame.
            self._scan_shape = (1, 1)
            self._det_shape = (data_np.shape[0], data_np.shape[1])
        elif ndim == 3:
            if scan_shape is not None:
                self._scan_shape = scan_shape
            else:
                n = data_np.shape[0]
                side = int(n**0.5)
                if side * side != n:
                    raise ValueError(
                        f"Cannot infer square scan_shape from N={n}. "
                        f"Provide scan_shape explicitly."
                    )
                self._scan_shape = (side, side)
            self._det_shape = (data_np.shape[1], data_np.shape[2])
        elif ndim == 4:
            self._scan_shape = (data_np.shape[0], data_np.shape[1])
            self._det_shape = (data_np.shape[2], data_np.shape[3])
        else:
            raise ValueError(f"Expected 2D, 3D, or 4D array, got {ndim}D")
        reshaped = data_np.reshape(
            self._scan_shape[0], self._scan_shape[1], self._det_shape[0], self._det_shape[1]
        )
        self._data = torch.from_numpy(reshaped).to(self._device)
        self.is_2d = self._scan_shape == (1, 1)
        self.shape_rows = self._scan_shape[0]
        self.shape_cols = self._scan_shape[1]
        self.det_rows = self._det_shape[0]
        self.det_cols = self._det_shape[1]
        self.dp_global_min = float(self._data.min().item())
        self.dp_global_max = float(self._data.max().item())

    @property
    def position(self) -> tuple[int, int]:
        return (self.pos_row, self.pos_col)

    @position.setter
    def position(self, value: tuple[int, int]):
        self.pos_row = int(max(0, min(value[0], self.shape_rows - 1)))
        self.pos_col = int(max(0, min(value[1], self.shape_cols - 1)))

    @property
    def scan_shape(self) -> tuple[int, int]:
        return self._scan_shape

    @property
    def detector_shape(self) -> tuple[int, int]:
        return self._det_shape

    def auto_detect_center(self) -> Self:
        """Auto-detect BF disk center and radius from summed diffraction pattern."""
        summed_dp = self._data.sum(dim=(0, 1))

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
        self.bf_radius = float(torch.sqrt(total / torch.pi))
        self.center_mode = "auto"
        return self

    def set_center(self, row: float, col: float) -> Self:
        self.center_row = float(row)
        self.center_col = float(col)
        self.center_mode = "manual"
        return self

    def center_from_midpoint(
        self, p1: tuple[float, float], p2: tuple[float, float]
    ) -> Self:
        """Center as the midpoint of a Friedel pair (single-crystal)."""
        self.center_row = 0.5 * (float(p1[0]) + float(p2[0]))
        self.center_col = 0.5 * (float(p1[1]) + float(p2[1]))
        self.center_mode = "midpoint"
        return self

    def center_from_ring(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
        p3: tuple[float, float],
    ) -> Self:
        """Center as the circle through three points on a ring (polycrystal)."""
        self.center_row, self.center_col = circle_center_from_3pts(p1, p2, p3)
        self.center_mode = "ring"
        return self

    def set_center_from_points(self, points) -> Self:
        """Center from 2 points (midpoint) or 3 points (ring fit), each (row, col)."""
        pts = [(float(p[0]), float(p[1])) for p in points]
        if len(pts) == 2:
            return self.center_from_midpoint(pts[0], pts[1])
        if len(pts) == 3:
            return self.center_from_ring(pts[0], pts[1], pts[2])
        raise ValueError(f"need 2 (midpoint) or 3 (ring) points, got {len(pts)}")

    def _on_center_from_points_request(self, change=None):
        flat = self._center_from_points_request
        if not flat:
            return
        if len(flat) % 2 != 0:
            self._center_from_points_request = []
            return
        points = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
        try:
            self.set_center_from_points(points)
        except ValueError:
            pass
        self._center_from_points_request = []

    def _get_frame(self, row: int, col: int) -> np.ndarray:
        row = max(0, min(row, self._scan_shape[0] - 1))
        col = max(0, min(col, self._scan_shape[1] - 1))
        return self._data[row, col].cpu().numpy().astype(np.float32)

    def _update_frame(self, change=None):
        frame = self._get_frame(self.pos_row, self.pos_col)
        self.dp_stats = [
            float(frame.mean()),
            float(frame.min()),
            float(frame.max()),
            float(frame.std()),
        ]
        self.frame_bytes = frame.tobytes()

    def _on_position_change(self, change=None):
        self._update_frame()
        self._update_radial()

    def _compute_virtual_image(self):
        row_coords = torch.arange(self.det_rows, device=self._device, dtype=torch.float32)[
            :, None
        ]
        col_coords = torch.arange(self.det_cols, device=self._device, dtype=torch.float32)[
            None, :
        ]
        r2 = (row_coords - self.center_row) ** 2 + (col_coords - self.center_col) ** 2
        mask = (r2 <= self.bf_radius**2).float()

        data_4d = self._data
        vi = torch.tensordot(data_4d, mask, dims=([2, 3], [0, 1]))

        vi_np = vi.cpu().numpy().astype(np.float32)
        self.vi_stats = [
            float(vi_np.mean()),
            float(vi_np.min()),
            float(vi_np.max()),
            float(vi_np.std()),
        ]
        self.virtual_image_bytes = vi_np.tobytes()

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

        frame = self._get_frame(self.pos_row, self.pos_col)
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
            g_magnitude = None
            d_spacing = None
            g_err = None
            d_err = None

        return {
            "d_spacing": d_spacing,
            "d_spacing_err": d_err,
            "g_magnitude": g_magnitude,
            "g_magnitude_err": g_err,
            "r_pixels": r_pixels,
            "r_pixels_err": r_err,
            "intensity": intensity,
        }

    def _fit_gaussian_2d(self, row: float, col: float) -> Optional[dict]:
        frame = self._get_frame(self.pos_row, self.pos_col)
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
        amp, fr, fc, sigma_row, sigma_col, off = popt
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

    def _reference_spot(self, spots) -> Optional[dict]:
        if not spots:
            return None
        for s in spots:
            if s["id"] == self.reference_spot_id:
                return s
        return spots[0]

    def _with_angles(self, spots) -> list:
        ref = self._reference_spot(spots)
        if ref is None:
            return spots
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
        exclude_radius: Optional[float] = None,
        replace: bool = True,
    ) -> Self:
        """Auto-detect spots (local maxima) in the current frame, excluding the central beam."""
        frame = self._get_frame(self.pos_row, self.pos_col).astype(np.float64)
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
        exclude_radius: Optional[float] = None,
        replace: bool = True,
    ) -> Self:
        """Auto-detect rings as peaks in the radial profile (use instead of detect_spots for rings)."""
        try:
            radii_px, intensity = self.radial_profile(use_calibration=False)
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
        peaks = peaks[radii_px[peaks] > float(exclude_radius)]  # drop the beam region
        if peaks.size == 0:
            return self
        # find_peaks already filtered by prominence; keep the innermost (strong low-order) rings.
        for p in sorted(peaks)[: int(max_rings)]:
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
        frame = self._get_frame(self.pos_row, self.pos_col)
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

    def label_spot(self, spot_id: int, hkl: str | None = None, note: str | None = None) -> Self:
        spots = []
        for s in self.spots:
            if s["id"] == spot_id:
                s = {**s}
                if hkl is not None:
                    s["hkl"] = str(hkl)
                if note is not None:
                    s["note"] = str(note)
            spots.append(s)
        self.spots = spots
        return self

    def set_reference_spot(self, spot_id: int) -> Self:
        self.reference_spot_id = int(spot_id)
        return self

    def clear_spots(self) -> Self:
        self.spots = []
        return self

    def undo_spot(self) -> Self:
        if self.spots:
            self.spots = list(self.spots[:-1])
        return self

    def remove_spot(self, spot_id: int) -> Self:
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

    def _on_reference_spot_change(self, change=None):
        if self.spots:
            self.spots = self._with_angles(list(self.spots))

    def _on_geometry_change(self, change=None):
        # Center/calibration moved → existing spot and ring d-spacings are stale.
        self._recompute_spots()
        self._recompute_rings()
        self._update_radial()

    def _compute_ring_info(self, radius_px: float) -> dict:
        if self.k_calibrated and self.k_pixel_size > 0:
            g_magnitude = float(radius_px) * self.k_pixel_size
            d_spacing = 1.0 / g_magnitude if g_magnitude > 0 else None
        else:
            g_magnitude = None
            d_spacing = None
        radii_px, intensity = self.radial_profile(use_calibration=False)
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
        self.rings = []
        return self

    def undo_ring(self) -> Self:
        if self.rings:
            self.rings = list(self.rings[:-1])
        return self

    def remove_ring(self, ring_id: int) -> Self:
        remaining = [r for r in self.rings if r["id"] != ring_id]
        if len(remaining) != len(self.rings):
            self.rings = remaining
        return self

    def _recompute_rings(self):
        if not self.rings:
            return
        self.rings = [
            {**r, **self._compute_ring_info(r["radius_px"])} for r in self.rings
        ]

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

    def radial_profile(
        self,
        *,
        n_bins: int | None = None,
        max_radius: float | None = None,
        use_calibration: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Azimuthally-averaged I(q): returns (x, intensity), x in 1/Å if calibrated else px."""
        frame = self._get_frame(self.pos_row, self.pos_col)
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
        if use_calibration and self.k_calibrated and self.k_pixel_size > 0:
            x = bin_centers_px * float(self.k_pixel_size)
        else:
            x = bin_centers_px
        return x.astype(np.float32), intensity.astype(np.float32)

    def _update_radial(self, change=None):
        if not self.show_radial:
            return
        try:
            x, intensity = self.radial_profile()
        except Exception:
            return
        self.radial_g_bytes = np.ascontiguousarray(x, dtype=np.float32).tobytes()
        self.radial_i_bytes = np.ascontiguousarray(intensity, dtype=np.float32).tobytes()
        self.radial_n_bins = int(x.size)
        self.radial_calibrated = bool(self.k_calibrated and self.k_pixel_size > 0)

    def calibrate_from_spot(self, row: float, col: float, d_known: float) -> Self:
        """Calibrate k_pixel_size so the spot at (row, col) has d-spacing d_known (Å)."""
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
        """Calibrate k_pixel_size so a ring at radius_px has d-spacing d_known (Å)."""
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

    def _measurement_records(self) -> list:
        records = []
        for s in self.spots:
            records.append({
                "id": s.get("id"), "kind": "spot",
                "raw_row": s.get("raw_row"), "raw_col": s.get("raw_col"),
                "row": s.get("row"), "col": s.get("col"),
                "row_err": s.get("row_err"), "col_err": s.get("col_err"),
                "r_pixels": s.get("r_pixels"), "r_pixels_err": s.get("r_pixels_err"),
                "g_inv_angstrom": s.get("g_magnitude"), "g_inv_angstrom_err": s.get("g_magnitude_err"),
                "d_angstrom": s.get("d_spacing"), "d_angstrom_err": s.get("d_spacing_err"),
                "angle_deg": s.get("angle_deg"), "angle_deg_err": s.get("angle_deg_err"),
                "intensity": s.get("intensity"), "fit_quality": s.get("fit_quality"),
                "hkl": s.get("hkl", ""), "note": s.get("note", ""),
            })
        for r in self.rings:
            records.append({
                "id": r.get("id"), "kind": "ring",
                "raw_row": None, "raw_col": None, "row": None, "col": None,
                "row_err": None, "col_err": None,
                "r_pixels": r.get("radius_px"), "r_pixels_err": None,
                "g_inv_angstrom": r.get("g_magnitude"), "g_inv_angstrom_err": None,
                "d_angstrom": r.get("d_spacing"), "d_angstrom_err": None,
                "angle_deg": None, "angle_deg_err": None,
                "intensity": r.get("intensity"), "fit_quality": None,
                "hkl": r.get("hkl", ""), "note": r.get("note", ""),
            })
        return records

    def export_measurements(self, path: str) -> pathlib.Path:
        """Export spots + rings to CSV or JSON (format inferred from extension)."""
        p = pathlib.Path(path)
        records = self._measurement_records()
        meta = {
            "widget_name": "ShowDiffraction",
            "widget_version": self.widget_version,
            "center_row": self.center_row,
            "center_col": self.center_col,
            "k_pixel_size_inv_angstrom_per_px": self.k_pixel_size,
            "calibrated": bool(self.k_calibrated),
            "calibration_source": self.calibration_source,
            "calibration_ref_d_angstrom": self.calibration_ref_d,
            "calibration_ref_radius_px": self.calibration_ref_radius,
            "reference_spot_id": self.reference_spot_id,
        }
        if p.suffix.lower() == ".json":
            p.write_text(json.dumps({"metadata": meta, "measurements": records}, indent=2))
        else:
            with open(p, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._MEASUREMENT_COLUMNS)
                writer.writeheader()
                writer.writerows(records)
        return p

    def set_image(self, data, scan_shape: tuple[int, int] | None = None) -> Self:
        """Replace data. Preserves display settings, clears spots."""
        if isinstance(data, IOResult):
            if data.pixel_size is not None:
                self.pixel_size = float(data.pixel_size)
            if data.title:
                self.title = data.title
            data = data.data
        if hasattr(data, "sampling") and hasattr(data, "array"):
            units = getattr(data, "units", ["pixels"] * 4)
            if units[0] in ("Å", "angstrom", "A", "nm"):
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
        self._ingest_data(data, scan_shape)
        self.pos_row = min(self.pos_row, self.shape_rows - 1)
        self.pos_col = min(self.pos_col, self.shape_cols - 1)
        self.spots = []
        self.rings = []
        self.auto_detect_center()
        self._compute_virtual_image()
        self._update_frame()
        self._update_radial()
        return self

    def state_dict(self):
        return {
            "title": self.title,
            "pos_row": self.pos_row,
            "pos_col": self.pos_col,
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
            "reference_spot_id": self.reference_spot_id,
            "center_mode": self.center_mode,
            "calibration_source": self.calibration_source,
            "calibration_ref_d": self.calibration_ref_d,
            "calibration_ref_radius": self.calibration_ref_radius,
            "show_radial": self.show_radial,
            "dp_colormap": self.dp_colormap,
            "dp_scale_mode": self.dp_scale_mode,
            "dp_invert": self.dp_invert,
            "dp_rotation": self.dp_rotation,
            "dp_vmin_pct": self.dp_vmin_pct,
            "dp_vmax_pct": self.dp_vmax_pct,
            "vi_colormap": self.vi_colormap,
            "vi_vmin_pct": self.vi_vmin_pct,
            "vi_vmax_pct": self.vi_vmax_pct,
            "show_stats": self.show_stats,
            "show_controls": self.show_controls,
            "disabled_tools": self.disabled_tools,
            "hidden_tools": self.hidden_tools,
        }

    def save(self, path: str):
        save_state_file(path, "ShowDiffraction", self.state_dict())

    def load_state_dict(self, state):
        allowed_keys = set(self.state_dict().keys())
        pending_pos_row = state.get("pos_row", None)
        pending_pos_col = state.get("pos_col", None)
        for key, val in state.items():
            if key in {"pos_row", "pos_col"}:
                continue
            if key in allowed_keys:
                setattr(self, key, val)
        if pending_pos_row is not None or pending_pos_col is not None:
            row = int(self.pos_row if pending_pos_row is None else pending_pos_row)
            col = int(self.pos_col if pending_pos_col is None else pending_pos_col)
            self.pos_row = int(max(0, min(row, self.shape_rows - 1)))
            self.pos_col = int(max(0, min(col, self.shape_cols - 1)))

    def summary(self):
        name = self.title if self.title else "ShowDiffraction"
        lines = [name, "═" * 32]
        lines.append(f"Scan:     {self.shape_rows}×{self.shape_cols} ({self.pixel_size:.2f} Å/px)")
        k_unit = "1/Å" if self.k_calibrated else "px"
        k_val = f"{self.k_pixel_size:.4f}" if self.k_calibrated else "uncalibrated"
        lines.append(f"Detector: {self.det_rows}×{self.det_cols} ({k_val} {k_unit}/px)")
        if self.k_calibrated:
            cal = f"Calib:    {self.calibration_source}"
            if self.calibration_ref_d > 0:
                cal += f" (d={self.calibration_ref_d:.3f} Å @ r={self.calibration_ref_radius:.1f} px)"
            lines.append(cal)
        lines.append(f"Position: ({self.pos_row}, {self.pos_col})")
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
        lines.append(f"Display:  {self.dp_colormap} | {self.dp_scale_mode}")
        if self.snap_enabled:
            lines.append(f"Snap:     radius={self.snap_radius}")
        print("\n".join(lines))

    def __repr__(self) -> str:
        k_unit = "1/Å" if self.k_calibrated else "px"
        shape = f"({self.shape_rows}, {self.shape_cols}, {self.det_rows}, {self.det_cols})"
        title_info = f", title='{self.title}'" if self.title else ""
        spots_info = f", spots={len(self.spots)}" if self.spots else ""
        return (
            f"ShowDiffraction(shape={shape}, "
            f"sampling=({self.pixel_size} Å, {self.k_pixel_size} {k_unit}), "
            f"pos=({self.pos_row}, {self.pos_col}){spots_info}{title_info})"
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

bind_tool_runtime_api(ShowDiffraction, "ShowDiffraction")
