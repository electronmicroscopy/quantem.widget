"""
Bin: Interactive calibration-aware 4D-STEM binning widget.

This widget is designed as a preprocessing + quality-control step before
`Show4DSTEM` analysis. It lets you interactively choose binning extent for
scan and detector axes, preview resulting shapes/calibration, and compare
BF/ADF virtual images before/after binning.
"""

from __future__ import annotations

import json
import math
import pathlib
import time

from typing import Self

import anywidget
import numpy as np
import traitlets

from quantem.widget.array_utils import to_numpy
from quantem.widget.io import IOResult
from quantem.widget.json_state import build_json_header, resolve_widget_version, save_state_file, unwrap_state_payload
from quantem.widget.bin_batch import _bin_axis_torch as _bin_axis_standalone, _binned_axis_shape
from quantem.widget.tool_parity import (
    bind_tool_runtime_api,
    build_tool_groups,
    normalize_tool_groups,
)

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import h5py

    _HAS_H5PY = True
except ImportError:
    h5py = None  # type: ignore[assignment]
    _HAS_H5PY = False

try:
    import hdf5plugin  # noqa: F401 — registers bitshuffle/LZ4 HDF5 filters
except ImportError:
    pass

try:
    from quantem.core.config import validate_device

    _HAS_VALIDATE_DEVICE = True
except Exception:
    _HAS_VALIDATE_DEVICE = False


# Supported units for calibration extraction from quantem Dataset objects
_REAL_UNITS = {"Å", "angstrom", "A", "nm"}
_K_UNITS = {"mrad", "1/Å", "1/A"}
_BIN_ESM = pathlib.Path(__file__).parent / "static" / "bin4d.js"
_BIN_CSS = pathlib.Path(__file__).parent / "static" / "bin4d.css"


def _as_pair(value: float | tuple[float, float] | list[float] | None, default: float) -> tuple[float, float]:
    """Normalize scalar/pair value to a `(row, col)` pair."""
    if value is None:
        return (float(default), float(default))
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("Expected a scalar or a 2-tuple/list")
        return (float(value[0]), float(value[1]))
    return (float(value), float(value))


def _qc_stats_torch(image) -> list[float]:
    """Compute compact quality metrics for a 2D torch tensor.

    Returns [mean, min, max, std, snr, contrast_1_99].
    """
    if image.numel() == 0:
        return [0.0] * 6

    arr = image.float()
    mean = float(arr.mean())
    amin = float(arr.min())
    amax = float(arr.max())
    std = float(arr.std(unbiased=False))
    snr = float(mean / (std + 1e-12))
    q = torch.quantile(arr.flatten(), torch.tensor([0.01, 0.99], device=arr.device))
    p1 = float(q[0])
    p99 = float(q[1])
    contrast = float((p99 - p1) / (abs(p99) + abs(p1) + 1e-12))
    return [mean, amin, amax, std, snr, contrast]


def _discover_arina_chunks(master_path):
    """Parse a Dectris arina master HDF5 and return chunk metadata.

    Parameters
    ----------
    master_path : str or pathlib.Path
        Path to the ``*_master.h5`` file.

    Returns
    -------
    dict
        Keys: ``chunks`` (list of (file_path, dataset_path, n_frames)),
        ``total_frames``, ``det_rows``, ``det_cols``,
        ``beam_center_x``, ``beam_center_y``, ``ntrigger``.
    """
    master_path = pathlib.Path(master_path).resolve()
    master_dir = master_path.parent
    chunks = []
    total_frames = 0
    det_rows = None
    det_cols = None
    with h5py.File(master_path, "r") as f:
        data_group = f["/entry/data"]
        for key in sorted(data_group.keys()):
            link = data_group.get(key, getlink=True)
            if isinstance(link, h5py.ExternalLink):
                chunk_file = str(master_dir / link.filename)
                chunk_ds_path = link.path
            else:
                chunk_file = str(master_path)
                chunk_ds_path = f"/entry/data/{key}"
            with h5py.File(chunk_file, "r") as cf:
                ds = cf[chunk_ds_path]
                n_frames = int(ds.shape[0])
                if det_rows is None and ds.ndim >= 3:
                    det_rows = int(ds.shape[1])
                    det_cols = int(ds.shape[2])
            total_frames += n_frames
            chunks.append((chunk_file, chunk_ds_path, n_frames))
        beam_center_x = None
        beam_center_y = None
        ntrigger = None
        det = f.get("/entry/instrument/detector")
        if det is not None:
            if "beam_center_x" in det:
                beam_center_x = float(np.asarray(det["beam_center_x"]))
            if "beam_center_y" in det:
                beam_center_y = float(np.asarray(det["beam_center_y"]))
            spec = det.get("detectorSpecific")
            if spec is not None and "ntrigger" in spec:
                ntrigger = int(np.asarray(spec["ntrigger"]))
    return {
        "chunks": chunks,
        "total_frames": total_frames,
        "det_rows": det_rows,
        "det_cols": det_cols,
        "beam_center_x": beam_center_x,
        "beam_center_y": beam_center_y,
        "ntrigger": ntrigger,
    }


def _is_arina_master(h5f):
    """Return True if the open HDF5 file looks like a Dectris arina master."""
    if "/entry/data" not in h5f:
        return False
    data_group = h5f["/entry/data"]
    for key in data_group.keys():
        link = data_group.get(key, getlink=True)
        if isinstance(link, h5py.ExternalLink):
            return True
    return False


class Bin4D(anywidget.AnyWidget):
    """
    Interactive 4D-STEM binning widget with calibration tracking and BF/ADF QC.

    Parameters
    ----------
    data : Dataset4dstem or array_like
        4D array `(scan_rows, scan_cols, det_rows, det_cols)` or flattened 3D
        `(N, det_rows, det_cols)` with explicit `scan_shape`.
        If a quantem dataset object is provided, calibration is auto-extracted.
    scan_shape : tuple[int, int], optional
        Required for flattened 3D input.
    pixel_size : float or tuple[float, float], optional
        Real-space sampling in Å/px for `(row, col)`.
    k_pixel_size : float or tuple[float, float], optional
        Detector sampling in mrad/px (or reciprocal-space units) for `(row, col)`.
    center : tuple[float, float], optional
        Detector center `(row, col)` used for BF/ADF preview masks.
    bin_mode : {"mean", "sum"}, default "mean"
        Reduction mode for block binning.
    edge_mode : {"crop", "pad", "error"}, default "crop"
        How non-divisible dimensions are handled.
    bf_radius_ratio : float, default 0.125
        BF disk radius as fraction of `min(det_rows, det_cols)`.
    adf_inner_ratio : float, default 0.30
        ADF annulus inner radius as fraction of detector size.
    adf_outer_ratio : float, default 0.45
        ADF annulus outer radius as fraction of detector size.

    Notes
    -----
    - Real-space calibration is multiplied by scan bin factors.
    - Detector-space calibration is multiplied by detector bin factors.
    - BF/ADF previews are recomputed after every parameter change.
    """

    _esm = _BIN_ESM if _BIN_ESM.exists() else "export function render() {}"
    _css = _BIN_CSS if _BIN_CSS.exists() else ""

    # Original shape
    scan_rows = traitlets.Int(1).tag(sync=True)
    scan_cols = traitlets.Int(1).tag(sync=True)
    det_rows = traitlets.Int(1).tag(sync=True)
    det_cols = traitlets.Int(1).tag(sync=True)

    # Current bin factors
    scan_bin_row = traitlets.Int(1).tag(sync=True)
    scan_bin_col = traitlets.Int(1).tag(sync=True)
    det_bin_row = traitlets.Int(1).tag(sync=True)
    det_bin_col = traitlets.Int(1).tag(sync=True)

    # UI hint maxima
    max_scan_bin_row = traitlets.Int(1).tag(sync=True)
    max_scan_bin_col = traitlets.Int(1).tag(sync=True)
    max_det_bin_row = traitlets.Int(1).tag(sync=True)
    max_det_bin_col = traitlets.Int(1).tag(sync=True)

    # Binning behavior
    bin_mode = traitlets.Unicode("mean").tag(sync=True)  # "mean" | "sum"
    edge_mode = traitlets.Unicode("crop").tag(sync=True)  # "crop" | "pad" | "error"
    device = traitlets.Unicode("cpu").tag(sync=True)

    # Binned shape
    binned_scan_rows = traitlets.Int(1).tag(sync=True)
    binned_scan_cols = traitlets.Int(1).tag(sync=True)
    binned_det_rows = traitlets.Int(1).tag(sync=True)
    binned_det_cols = traitlets.Int(1).tag(sync=True)

    # Calibration (original)
    pixel_size_row = traitlets.Float(1.0).tag(sync=True)
    pixel_size_col = traitlets.Float(1.0).tag(sync=True)
    pixel_unit = traitlets.Unicode("px").tag(sync=True)
    pixel_calibrated = traitlets.Bool(False).tag(sync=True)

    k_pixel_size_row = traitlets.Float(1.0).tag(sync=True)
    k_pixel_size_col = traitlets.Float(1.0).tag(sync=True)
    k_unit = traitlets.Unicode("px").tag(sync=True)
    k_calibrated = traitlets.Bool(False).tag(sync=True)

    # Calibration (binned)
    binned_pixel_size_row = traitlets.Float(1.0).tag(sync=True)
    binned_pixel_size_col = traitlets.Float(1.0).tag(sync=True)
    binned_k_pixel_size_row = traitlets.Float(1.0).tag(sync=True)
    binned_k_pixel_size_col = traitlets.Float(1.0).tag(sync=True)

    # Detector center for BF/ADF preview masks
    center_row = traitlets.Float(0.0).tag(sync=True)
    center_col = traitlets.Float(0.0).tag(sync=True)

    # BF/ADF mask settings (fractions of detector size)
    bf_radius_ratio = traitlets.Float(0.125).tag(sync=True)
    adf_inner_ratio = traitlets.Float(0.30).tag(sync=True)
    adf_outer_ratio = traitlets.Float(0.45).tag(sync=True)

    # Scan position for DP exploration — compound [row, col], [-1, -1] = mean DP
    _scan_position = traitlets.List(traitlets.Int(), default_value=[-1, -1]).tag(sync=True)

    # Preview data as float32 bytes
    original_bf_bytes = traitlets.Bytes(b"").tag(sync=True)
    original_adf_bytes = traitlets.Bytes(b"").tag(sync=True)
    binned_bf_bytes = traitlets.Bytes(b"").tag(sync=True)
    binned_adf_bytes = traitlets.Bytes(b"").tag(sync=True)

    # Mean diffraction pattern (detector-space preview)
    original_mean_dp_bytes = traitlets.Bytes(b"").tag(sync=True)
    binned_mean_dp_bytes = traitlets.Bytes(b"").tag(sync=True)

    # Per-position diffraction pattern bytes (sent when _scan_position >= 0)
    _position_dp_bytes = traitlets.Bytes(b"").tag(sync=True)
    _binned_position_dp_bytes = traitlets.Bytes(b"").tag(sync=True)

    # Binned detector center (for JS overlay positioning)
    binned_center_row = traitlets.Float(0.0).tag(sync=True)
    binned_center_col = traitlets.Float(0.0).tag(sync=True)

    # Preview stats [mean, min, max, std, snr, contrast]
    original_bf_stats = traitlets.List(traitlets.Float(), default_value=[0.0] * 6).tag(sync=True)
    original_adf_stats = traitlets.List(traitlets.Float(), default_value=[0.0] * 6).tag(sync=True)
    binned_bf_stats = traitlets.List(traitlets.Float(), default_value=[0.0] * 6).tag(sync=True)
    binned_adf_stats = traitlets.List(traitlets.Float(), default_value=[0.0] * 6).tag(sync=True)

    # Status
    status_message = traitlets.Unicode("").tag(sync=True)
    status_level = traitlets.Unicode("ok").tag(sync=True)  # "ok" | "warn" | "error"

    # Display
    title = traitlets.Unicode("").tag(sync=True)
    cmap = traitlets.Unicode("inferno").tag(sync=True)
    log_scale = traitlets.Bool(False).tag(sync=True)
    auto_contrast = traitlets.Bool(False).tag(sync=True)
    show_fft = traitlets.Bool(False).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)
    disabled_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)
    hidden_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)

    # Export (trait-triggered .npy download)
    _npy_export_requested = traitlets.Bool(False).tag(sync=True)
    _npy_export_data = traitlets.Bytes(b"").tag(sync=True)

    @classmethod
    def _normalize_tool_groups(cls, tool_groups):
        return normalize_tool_groups("Bin4D", tool_groups)

    @classmethod
    def _build_disabled_tools(
        cls,
        disabled_tools=None,
        disable_display: bool = False,
        disable_binning: bool = False,
        disable_mask: bool = False,
        disable_preview: bool = False,
        disable_stats: bool = False,
        disable_export: bool = False,
        disable_all: bool = False,
    ):
        return build_tool_groups(
            "Bin4D",
            tool_groups=disabled_tools,
            all_flag=disable_all,
            flag_map={
                "display": disable_display,
                "binning": disable_binning,
                "mask": disable_mask,
                "preview": disable_preview,
                "stats": disable_stats,
                "export": disable_export,
            },
        )

    @classmethod
    def _build_hidden_tools(
        cls,
        hidden_tools=None,
        hide_display: bool = False,
        hide_binning: bool = False,
        hide_mask: bool = False,
        hide_preview: bool = False,
        hide_stats: bool = False,
        hide_export: bool = False,
        hide_all: bool = False,
    ):
        return build_tool_groups(
            "Bin4D",
            tool_groups=hidden_tools,
            all_flag=hide_all,
            flag_map={
                "display": hide_display,
                "binning": hide_binning,
                "mask": hide_mask,
                "preview": hide_preview,
                "stats": hide_stats,
                "export": hide_export,
            },
        )

    @traitlets.validate("disabled_tools")
    def _validate_disabled_tools(self, proposal):
        return self._normalize_tool_groups(proposal["value"])

    @traitlets.validate("hidden_tools")
    def _validate_hidden_tools(self, proposal):
        return self._normalize_tool_groups(proposal["value"])

    @traitlets.validate("scan_bin_row", "scan_bin_col", "det_bin_row", "det_bin_col")
    def _validate_bin_factor(self, proposal):
        value = int(proposal["value"])
        if value < 1:
            raise traitlets.TraitError("Binning factors must be >= 1")
        return value

    @traitlets.validate("bin_mode")
    def _validate_bin_mode(self, proposal):
        value = str(proposal["value"]).lower()
        if value not in {"mean", "sum"}:
            raise traitlets.TraitError("bin_mode must be 'mean' or 'sum'")
        return value

    @traitlets.validate("edge_mode")
    def _validate_edge_mode(self, proposal):
        value = str(proposal["value"]).lower()
        if value not in {"crop", "pad", "error"}:
            raise traitlets.TraitError("edge_mode must be 'crop', 'pad', or 'error'")
        return value

    @traitlets.validate("device")
    def _validate_device_name(self, proposal):
        value = str(proposal["value"]).strip().lower()
        if not value:
            raise traitlets.TraitError("device must be a non-empty string")
        return value

    @traitlets.validate("bf_radius_ratio", "adf_inner_ratio", "adf_outer_ratio")
    def _validate_ratio(self, proposal):
        value = float(proposal["value"])
        if value < 0:
            raise traitlets.TraitError("Ratios must be >= 0")
        return value

    def __init__(
        self,
        data,
        scan_shape: tuple[int, int] | None = None,
        pixel_size: float | tuple[float, float] | None = None,
        k_pixel_size: float | tuple[float, float] | None = None,
        center: tuple[float, float] | None = None,
        bin_mode: str = "mean",
        edge_mode: str = "crop",
        bf_radius_ratio: float = 0.125,
        adf_inner_ratio: float = 0.30,
        adf_outer_ratio: float = 0.45,
        title: str = "",
        cmap: str = "inferno",
        log_scale: bool = False,
        show_controls: bool = True,
        disabled_tools: list[str] | None = None,
        hidden_tools: list[str] | None = None,
        disable_display: bool = False,
        disable_binning: bool = False,
        disable_mask: bool = False,
        disable_preview: bool = False,
        disable_stats: bool = False,
        disable_export: bool = False,
        disable_all: bool = False,
        hide_display: bool = False,
        hide_binning: bool = False,
        hide_mask: bool = False,
        hide_preview: bool = False,
        hide_stats: bool = False,
        hide_export: bool = False,
        hide_all: bool = False,
        device: str | None = None,
        state: dict | str | pathlib.Path | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.widget_version = resolve_widget_version()

        self.bin_mode = bin_mode
        self.edge_mode = edge_mode
        self.cmap = cmap
        self.log_scale = log_scale
        self.show_controls = show_controls
        self.bf_radius_ratio = float(bf_radius_ratio)
        self.adf_inner_ratio = float(adf_inner_ratio)
        self.adf_outer_ratio = float(adf_outer_ratio)

        # Check if data is an IOResult and extract metadata
        if isinstance(data, IOResult):
            if not title and data.title:
                title = data.title
            data = data.data

        # Dataset-like duck typing: `array`, `sampling`, `units`
        dataset_pixel: float | tuple[float, float] | None = None
        dataset_k: float | tuple[float, float] | None = None
        dataset_pixel_unit = "px"
        dataset_k_unit = "px"
        pixel_calibrated = False
        k_calibrated = False

        if hasattr(data, "sampling") and hasattr(data, "array"):
            if not title and hasattr(data, "name") and data.name:
                title = str(data.name)
            units = list(getattr(data, "units", ["pixels"] * 4))
            sampling = list(getattr(data, "sampling", [1.0] * 4))

            if len(units) >= 2 and len(sampling) >= 2 and units[0] in _REAL_UNITS:
                sy = float(sampling[0])
                sx = float(sampling[1])
                if units[0] == "nm":
                    sy *= 10.0
                if units[1] == "nm":
                    sx *= 10.0
                dataset_pixel = (sy, sx)
                dataset_pixel_unit = "Å"
                pixel_calibrated = True

            if len(units) >= 4 and len(sampling) >= 4 and units[2] in _K_UNITS:
                ky = float(sampling[2])
                kx = float(sampling[3])
                dataset_k = (ky, kx)
                dataset_k_unit = units[2]
                k_calibrated = True

            data = data.array

        self.title = title

        # Manual kwargs override extracted calibration
        p_row, p_col = _as_pair(pixel_size if pixel_size is not None else dataset_pixel, 1.0)
        k_row, k_col = _as_pair(k_pixel_size if k_pixel_size is not None else dataset_k, 1.0)

        if pixel_size is not None:
            pixel_calibrated = True
            dataset_pixel_unit = "Å"
        if k_pixel_size is not None:
            k_calibrated = True
            dataset_k_unit = "mrad"

        self.pixel_size_row = p_row
        self.pixel_size_col = p_col
        self.pixel_unit = dataset_pixel_unit if pixel_calibrated else "px"
        self.pixel_calibrated = pixel_calibrated

        self.k_pixel_size_row = k_row
        self.k_pixel_size_col = k_col
        self.k_unit = dataset_k_unit if k_calibrated else "px"
        self.k_calibrated = k_calibrated

        # Normalize input to float32 4D NumPy, then move to torch (compute is torch-only).
        data_np = to_numpy(data, dtype=np.float32)
        if data_np.ndim == 4:
            scan_r, scan_c, det_r, det_c = data_np.shape
            data4d = data_np
        elif data_np.ndim == 3:
            if scan_shape is None:
                n = int(data_np.shape[0])
                side = int(math.isqrt(n))
                if side * side != n:
                    raise ValueError(
                        f"Cannot infer square scan_shape from flattened N={n}. Provide scan_shape=(rows, cols)."
                    )
                scan_shape = (side, side)
            if int(scan_shape[0]) * int(scan_shape[1]) != int(data_np.shape[0]):
                raise ValueError(
                    f"scan_shape={scan_shape} does not match flattened length {data_np.shape[0]}"
                )
            scan_r, scan_c = int(scan_shape[0]), int(scan_shape[1])
            det_r, det_c = int(data_np.shape[1]), int(data_np.shape[2])
            data4d = data_np.reshape(scan_r, scan_c, det_r, det_c)
        else:
            raise ValueError(
                f"Expected 4D array (scan_rows, scan_cols, det_rows, det_cols) or flattened 3D (N, det_rows, det_cols), got {data_np.ndim}D"
            )

        self.scan_rows = scan_r
        self.scan_cols = scan_c
        self.det_rows = det_r
        self.det_cols = det_c

        if not _HAS_TORCH:
            raise ImportError("Bin requires torch. Install PyTorch to use this widget.")

        device_str = self._resolve_torch_device(requested=device, numel=int(data_np.size))
        if device_str is None:
            requested = "auto" if device is None else str(device)
            raise ValueError(f"Unable to initialize torch device '{requested}'")

        self._device = torch.device(device_str)
        data4d_writable = np.array(data4d, dtype=np.float32, copy=True)
        self._data_torch = torch.from_numpy(data4d_writable).to(self._device)
        self.device = device_str

        # Slider maxima (UI hint only)
        self.max_scan_bin_row = max(1, scan_r)
        self.max_scan_bin_col = max(1, scan_c)
        self.max_det_bin_row = max(1, det_r)
        self.max_det_bin_col = max(1, det_c)

        if center is None:
            self.center_row = det_r / 2.0
            self.center_col = det_c / 2.0
        else:
            self.center_row = float(center[0])
            self.center_col = float(center[1])

        self._binned_data_torch = self._data_torch
        self._original_bf_torch = torch.zeros((self.scan_rows, self.scan_cols), dtype=torch.float32)
        self._original_adf_torch = torch.zeros((self.scan_rows, self.scan_cols), dtype=torch.float32)
        self._binned_bf_torch = torch.zeros((self.scan_rows, self.scan_cols), dtype=torch.float32)
        self._binned_adf_torch = torch.zeros((self.scan_rows, self.scan_cols), dtype=torch.float32)

        self.observe(
            self._on_params_changed,
            names=[
                "scan_bin_row",
                "scan_bin_col",
                "det_bin_row",
                "det_bin_col",
                "bin_mode",
                "edge_mode",
                "center_row",
                "center_col",
                "bf_radius_ratio",
                "adf_inner_ratio",
                "adf_outer_ratio",
            ],
        )
        self.observe(self._on_position_changed, names=["_scan_position"])
        self.observe(self._on_npy_export, names=["_npy_export_requested"])

        self._recompute_previews()

        self.disabled_tools = self._build_disabled_tools(
            disabled_tools=disabled_tools,
            disable_display=disable_display,
            disable_binning=disable_binning,
            disable_mask=disable_mask,
            disable_preview=disable_preview,
            disable_stats=disable_stats,
            disable_export=disable_export,
            disable_all=disable_all,
        )
        self.hidden_tools = self._build_hidden_tools(
            hidden_tools=hidden_tools,
            hide_display=hide_display,
            hide_binning=hide_binning,
            hide_mask=hide_mask,
            hide_preview=hide_preview,
            hide_stats=hide_stats,
            hide_export=hide_export,
            hide_all=hide_all,
        )

        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(
                    json.loads(pathlib.Path(state).read_text()),
                    require_envelope=True,
                )
            else:
                state = unwrap_state_payload(state)
            self.load_state_dict(state)

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def file_info(cls, path, det_bin_row=2, det_bin_col=2, edge_mode="crop"):
        """Print file summary without loading data.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to a master or saved HDF5 file.
        det_bin_row : int
            Detector row binning factor (for estimating binned size).
        det_bin_col : int
            Detector column binning factor (for estimating binned size).
        edge_mode : {"crop", "pad", "error"}
            Edge handling (for estimating binned size).
        """
        if not _HAS_H5PY:
            raise ImportError("h5py is required. Install: pip install h5py")
        path = pathlib.Path(path).resolve()
        disk_bytes = path.stat().st_size
        disk_gb = disk_bytes / 1_000_000_000.0
        with h5py.File(path, "r") as f:
            is_arina = _is_arina_master(f)
        if is_arina:
            info = _discover_arina_chunks(path)
            total = info["total_frames"]
            det_r, det_c = info["det_rows"], info["det_cols"]
            side = int(math.isqrt(total))
            scan_r, scan_c = (side, side) if side * side == total else (total, 1)
            raw_gb = total * det_r * det_c * 2 / 1e9  # uint16
            binned_det_r, _ = _binned_axis_shape(det_r, det_bin_row, edge_mode, axis=0)
            binned_det_c, _ = _binned_axis_shape(det_c, det_bin_col, edge_mode, axis=1)
            mem_gb = scan_r * scan_c * binned_det_r * binned_det_c * 4 / 1e9
            # sum chunk file sizes (master file itself is tiny)
            chunk_paths = {c[0] for c in info["chunks"]}
            disk_gb = sum(pathlib.Path(p).stat().st_size for p in chunk_paths) / 1e9
            lines = [
                f"File:      {path.name}",
                f"Format:    Dectris arina master ({len(info['chunks'])} chunks)",
                f"Disk:      {disk_gb:.2f} GB (bitshuffle+LZ4, {len(chunk_paths)} files)",
                f"Raw:       {raw_gb:.1f} GB (uint16 uncompressed)",
                f"Scan:      {scan_r} x {scan_c} ({total:,} frames)",
                f"Detector:  {det_r} x {det_c}",
                f"Binned:    {scan_r} x {scan_c} x {binned_det_r} x {binned_det_c} (det {det_bin_row}x{det_bin_col})",
                f"Memory:    {mem_gb:.1f} GB (float32 after binning)",
            ]
        else:
            with h5py.File(path, "r") as f:
                ds = f["data"]
                shape = tuple(int(v) for v in ds.shape)
                mem_gb = np.prod(shape) * 4 / 1e9
            lines = [
                f"File:      {path.name}",
                f"Format:    Saved binned HDF5",
                f"Disk:      {disk_gb:.2f} GB (bitshuffle+LZ4)",
                f"Shape:     {shape}",
                f"Memory:    {mem_gb:.1f} GB (float32)",
            ]
        print("\n".join(lines))

    @classmethod
    def from_file(
        cls,
        path,
        scan_shape=None,
        det_bin_row=2,
        det_bin_col=2,
        bin_mode="mean",
        edge_mode="crop",
        frames_per_batch=1000,
        pixel_size=None,
        k_pixel_size=None,
        center=None,
        title="",
        device=None,
        **kwargs,
    ) -> "Bin":
        """Load an HDF5 file into a Bin widget.

        Auto-detects the file format:

        - **Dectris arina master** (``*_master.h5`` with external-link chunks):
          streams frames in small batches, bins detector axes on the fly, and
          assembles only the binned result in memory.
        - **Saved binned file** (written by :meth:`save_h5`): loads the 4D
          dataset and calibration directly — no re-binning needed.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the HDF5 file.
        scan_shape : tuple[int, int], optional
            Scan grid ``(rows, cols)``. Inferred as square if omitted.
            Ignored when loading a saved binned file.
        det_bin_row : int
            Detector row binning factor (arina path only).
        det_bin_col : int
            Detector column binning factor (arina path only).
        bin_mode : {"mean", "sum"}
            Reduction mode for binning (arina path only).
        edge_mode : {"crop", "pad", "error"}
            How non-divisible detector edges are handled (arina path only).
        frames_per_batch : int
            Frames read per I/O batch (arina path only).
        pixel_size : float or tuple[float, float], optional
            Real-space sampling in Å/px. Overrides saved calibration.
        k_pixel_size : float or tuple[float, float], optional
            Detector sampling in mrad/px. Overrides saved calibration.
        center : tuple[float, float], optional
            Detector center ``(row, col)``. Overrides saved calibration.
        title : str
            Widget title.
        device : str, optional
            Torch device.
        **kwargs
            Forwarded to widget init (``cmap``, ``log_scale``, etc.).

        Returns
        -------
        Bin
            Widget ready for interactive exploration.
        """
        if not _HAS_H5PY:
            raise ImportError("h5py is required for Bin.from_file(). Install: pip install h5py")
        if not _HAS_TORCH:
            raise ImportError("torch is required for Bin.from_file()")

        path = pathlib.Path(path).resolve()

        # -- auto-detect format ------------------------------------------------
        with h5py.File(path, "r") as probe:
            is_arina = _is_arina_master(probe)

        if is_arina:
            return cls._from_arina(
                path,
                scan_shape=scan_shape,
                det_bin_row=det_bin_row,
                det_bin_col=det_bin_col,
                bin_mode=bin_mode,
                edge_mode=edge_mode,
                frames_per_batch=frames_per_batch,
                pixel_size=pixel_size,
                k_pixel_size=k_pixel_size,
                center=center,
                title=title,
                device=device,
                **kwargs,
            )
        return cls._from_binned_h5(
            path,
            pixel_size=pixel_size,
            k_pixel_size=k_pixel_size,
            center=center,
            title=title,
            device=device,
            **kwargs,
        )

    @classmethod
    def _from_binned_h5(cls, path, pixel_size=None, k_pixel_size=None,
                        center=None, title="", device=None, **kwargs):
        """Load a saved binned HDF5 (written by :meth:`save_h5`).

        Uses the same zero-copy construction as ``_from_arina`` to avoid
        doubling memory for large datasets.
        """
        t0 = time.time()
        with h5py.File(path, "r") as f:
            ds = f["data"]
            data_np = np.asarray(ds, dtype=np.float32)
            attrs = dict(ds.attrs)
        elapsed = time.time() - t0
        print(f"  Loaded {path.name}: {data_np.shape} in {elapsed:.1f}s")

        scan_r, scan_c = int(data_np.shape[0]), int(data_np.shape[1])
        det_r, det_c = int(data_np.shape[2]), int(data_np.shape[3])

        # calibration from file, overridable by explicit kwargs
        if pixel_size is None and "pixel_size_row" in attrs:
            pixel_size = (float(attrs["pixel_size_row"]), float(attrs["pixel_size_col"]))
        if k_pixel_size is None and "k_pixel_size_row" in attrs:
            k_pixel_size = (float(attrs["k_pixel_size_row"]), float(attrs["k_pixel_size_col"]))
        if center is None and "center_row" in attrs:
            center = (float(attrs["center_row"]), float(attrs["center_col"]))
        if not title and "title" in attrs:
            title = str(attrs["title"])

        pixel_unit = str(attrs.get("pixel_unit", "px"))
        k_unit = str(attrs.get("k_unit", "px"))

        # resolve device
        numel = int(data_np.size)
        if device is not None:
            device_str = str(device).strip().lower()
        elif _HAS_VALIDATE_DEVICE:
            device_str, _ = validate_device(None)
        else:
            device_str = (
                "mps"
                if torch.backends.mps.is_available()
                else "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        if device_str == "mps" and numel > 2**31 - 1:
            device_str = "cpu"
        torch.zeros(1, device=torch.device(device_str))

        # numpy → torch without an extra copy (from_numpy shares memory on cpu)
        data_torch = torch.from_numpy(data_np)
        if device_str != "cpu":
            data_torch = data_torch.to(torch.device(device_str))

        # build widget directly — same pattern as _from_arina
        cmap = kwargs.pop("cmap", "inferno")
        log_scale = kwargs.pop("log_scale", False)
        show_controls = kwargs.pop("show_controls", True)
        bf_radius_ratio = float(kwargs.pop("bf_radius_ratio", 0.125))
        adf_inner_ratio = float(kwargs.pop("adf_inner_ratio", 0.30))
        adf_outer_ratio = float(kwargs.pop("adf_outer_ratio", 0.45))

        inst = cls.__new__(cls)
        anywidget.AnyWidget.__init__(inst, **kwargs)
        inst.widget_version = resolve_widget_version()
        inst.bin_mode = "mean"
        inst.edge_mode = "crop"
        inst.title = title
        inst.cmap = cmap
        inst.log_scale = log_scale
        inst.show_controls = show_controls
        inst.bf_radius_ratio = bf_radius_ratio
        inst.adf_inner_ratio = adf_inner_ratio
        inst.adf_outer_ratio = adf_outer_ratio

        # calibration
        p_row, p_col = _as_pair(pixel_size, 1.0)
        k_row, k_col = _as_pair(k_pixel_size, 1.0)
        inst.pixel_size_row = p_row
        inst.pixel_size_col = p_col
        inst.pixel_unit = pixel_unit if pixel_unit != "px" else ("Å" if pixel_size is not None else "px")
        inst.pixel_calibrated = pixel_unit != "px" or pixel_size is not None
        inst.k_pixel_size_row = k_row
        inst.k_pixel_size_col = k_col
        inst.k_unit = k_unit if k_unit != "px" else ("mrad" if k_pixel_size is not None else "px")
        inst.k_calibrated = k_unit != "px" or k_pixel_size is not None

        inst.scan_rows = scan_r
        inst.scan_cols = scan_c
        inst.det_rows = det_r
        inst.det_cols = det_c

        inst._device = torch.device(device_str)
        inst._data_torch = data_torch
        inst.device = device_str

        inst.max_scan_bin_row = max(1, scan_r)
        inst.max_scan_bin_col = max(1, scan_c)
        inst.max_det_bin_row = max(1, det_r)
        inst.max_det_bin_col = max(1, det_c)

        if center is not None:
            inst.center_row = float(center[0])
            inst.center_col = float(center[1])
        else:
            inst.center_row = det_r / 2.0
            inst.center_col = det_c / 2.0

        inst._binned_data_torch = inst._data_torch
        inst._original_bf_torch = torch.zeros((scan_r, scan_c), dtype=torch.float32)
        inst._original_adf_torch = torch.zeros((scan_r, scan_c), dtype=torch.float32)
        inst._binned_bf_torch = torch.zeros((scan_r, scan_c), dtype=torch.float32)
        inst._binned_adf_torch = torch.zeros((scan_r, scan_c), dtype=torch.float32)

        inst.observe(
            inst._on_params_changed,
            names=[
                "scan_bin_row", "scan_bin_col", "det_bin_row", "det_bin_col",
                "bin_mode", "edge_mode", "center_row", "center_col",
                "bf_radius_ratio", "adf_inner_ratio", "adf_outer_ratio",
            ],
        )
        inst.observe(inst._on_position_changed, names=["_scan_position"])
        inst.observe(inst._on_npy_export, names=["_npy_export_requested"])

        inst.disabled_tools = cls._build_disabled_tools()
        inst.hidden_tools = cls._build_hidden_tools()
        inst._recompute_previews()
        return inst

    @classmethod
    def _from_arina(cls, path, scan_shape=None, det_bin_row=2, det_bin_col=2,
                    bin_mode="mean", edge_mode="crop", frames_per_batch=1000,
                    pixel_size=None, k_pixel_size=None, center=None,
                    title="", device=None, **kwargs):
        """Stream arina master HDF5, bin detector axes on the fly."""
        # -- discover chunks --------------------------------------------------
        info = _discover_arina_chunks(path)
        chunks = info["chunks"]
        total_frames = info["total_frames"]
        det_r = info["det_rows"]
        det_c = info["det_cols"]

        # -- scan shape --------------------------------------------------------
        if scan_shape is None:
            side = int(math.isqrt(total_frames))
            if side * side != total_frames:
                raise ValueError(
                    f"Cannot infer square scan_shape from {total_frames} frames. "
                    f"Provide scan_shape=(rows, cols)."
                )
            scan_shape = (side, side)
        scan_r, scan_c = int(scan_shape[0]), int(scan_shape[1])

        # -- binned detector shape ---------------------------------------------
        binned_det_r, _ = _binned_axis_shape(det_r, det_bin_row, edge_mode, axis=0)
        binned_det_c, _ = _binned_axis_shape(det_c, det_bin_col, edge_mode, axis=1)

        # -- resolve torch device ----------------------------------------------
        numel = scan_r * scan_c * binned_det_r * binned_det_c
        if device is not None:
            device_str = str(device).strip().lower()
        elif _HAS_VALIDATE_DEVICE:
            device_str, _ = validate_device(None)
        else:
            device_str = (
                "mps"
                if torch.backends.mps.is_available()
                else "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        if device_str == "mps" and numel > 2**31 - 1:
            device_str = "cpu"
        torch.zeros(1, device=torch.device(device_str))

        # -- pre-allocate output -----------------------------------------------
        output = torch.zeros(
            (scan_r, scan_c, binned_det_r, binned_det_c),
            dtype=torch.float32,
            device=torch.device(device_str),
        )

        # -- stream chunks & bin detector axes ---------------------------------
        frame_idx = 0
        t0 = time.time()
        for chunk_file, chunk_ds_path, chunk_n_frames in chunks:
            with h5py.File(chunk_file, "r") as cf:
                ds = cf[chunk_ds_path]
                for batch_start in range(0, chunk_n_frames, frames_per_batch):
                    batch_end = min(batch_start + frames_per_batch, chunk_n_frames)
                    batch = torch.from_numpy(
                        np.asarray(ds[batch_start:batch_end], dtype=np.float32)
                    )
                    # bin detector rows (axis 1) then cols (axis 2)
                    binned = _bin_axis_standalone(batch, axis=1, factor=det_bin_row, mode=bin_mode, edge=edge_mode)
                    binned = _bin_axis_standalone(binned, axis=2, factor=det_bin_col, mode=bin_mode, edge=edge_mode)
                    # scatter into output using vectorized indexing
                    n_batch = binned.shape[0]
                    gi = torch.arange(frame_idx, frame_idx + n_batch, dtype=torch.long)
                    rows = gi // scan_c
                    cols = gi % scan_c
                    mask = rows < scan_r
                    if mask.any():
                        output[rows[mask], cols[mask]] = binned[mask].to(device=output.device)
                    frame_idx += n_batch
                    elapsed = time.time() - t0
                    pct = frame_idx / total_frames * 100
                    fps = frame_idx / max(elapsed, 1e-6)
                    remaining = (total_frames - frame_idx) / max(fps, 1e-6)
                    print(
                        f"\r  Loading: {frame_idx:,}/{total_frames:,} frames "
                        f"({pct:.1f}%) {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining",
                        end="",
                        flush=True,
                    )
        elapsed_total = time.time() - t0
        print(f"\n  Done: {total_frames:,} frames in {elapsed_total:.1f}s")

        # -- build Bin instance without copying data ---------------------------
        cmap = kwargs.pop("cmap", "inferno")
        log_scale = kwargs.pop("log_scale", False)
        show_controls = kwargs.pop("show_controls", True)
        bf_radius_ratio = float(kwargs.pop("bf_radius_ratio", 0.125))
        adf_inner_ratio = float(kwargs.pop("adf_inner_ratio", 0.30))
        adf_outer_ratio = float(kwargs.pop("adf_outer_ratio", 0.45))

        inst = cls.__new__(cls)
        anywidget.AnyWidget.__init__(inst, **kwargs)
        inst.widget_version = resolve_widget_version()
        inst.bin_mode = bin_mode
        inst.edge_mode = edge_mode
        inst.title = title
        inst.cmap = cmap
        inst.log_scale = log_scale
        inst.show_controls = show_controls
        inst.bf_radius_ratio = bf_radius_ratio
        inst.adf_inner_ratio = adf_inner_ratio
        inst.adf_outer_ratio = adf_outer_ratio

        # calibration
        p_row, p_col = _as_pair(pixel_size, 1.0)
        k_row, k_col = _as_pair(k_pixel_size, 1.0)
        inst.pixel_size_row = p_row
        inst.pixel_size_col = p_col
        inst.pixel_unit = "Å" if pixel_size is not None else "px"
        inst.pixel_calibrated = pixel_size is not None
        inst.k_pixel_size_row = k_row
        inst.k_pixel_size_col = k_col
        inst.k_unit = "mrad" if k_pixel_size is not None else "px"
        inst.k_calibrated = k_pixel_size is not None

        # shape — the widget sees the already-binned detector dimensions
        inst.scan_rows = scan_r
        inst.scan_cols = scan_c
        inst.det_rows = binned_det_r
        inst.det_cols = binned_det_c

        # torch data
        inst._device = torch.device(device_str)
        inst._data_torch = output
        inst.device = device_str

        # slider maxima
        inst.max_scan_bin_row = max(1, scan_r)
        inst.max_scan_bin_col = max(1, scan_c)
        inst.max_det_bin_row = max(1, binned_det_r)
        inst.max_det_bin_col = max(1, binned_det_c)

        # detector center
        if center is not None:
            inst.center_row = float(center[0])
            inst.center_col = float(center[1])
        elif info["beam_center_y"] is not None and info["beam_center_x"] is not None:
            inst.center_row = float(info["beam_center_y"]) / det_bin_row
            inst.center_col = float(info["beam_center_x"]) / det_bin_col
        else:
            inst.center_row = binned_det_r / 2.0
            inst.center_col = binned_det_c / 2.0

        # internal preview tensors
        inst._binned_data_torch = inst._data_torch
        inst._original_bf_torch = torch.zeros((scan_r, scan_c), dtype=torch.float32)
        inst._original_adf_torch = torch.zeros((scan_r, scan_c), dtype=torch.float32)
        inst._binned_bf_torch = torch.zeros((scan_r, scan_c), dtype=torch.float32)
        inst._binned_adf_torch = torch.zeros((scan_r, scan_c), dtype=torch.float32)

        # observers
        inst.observe(
            inst._on_params_changed,
            names=[
                "scan_bin_row", "scan_bin_col", "det_bin_row", "det_bin_col",
                "bin_mode", "edge_mode", "center_row", "center_col",
                "bf_radius_ratio", "adf_inner_ratio", "adf_outer_ratio",
            ],
        )
        inst.observe(inst._on_position_changed, names=["_scan_position"])
        inst.observe(inst._on_npy_export, names=["_npy_export_requested"])

        inst.disabled_tools = cls._build_disabled_tools()
        inst.hidden_tools = cls._build_hidden_tools()
        inst._recompute_previews()
        return inst

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "title": self.title,
            "scan_bin_row": self.scan_bin_row,
            "scan_bin_col": self.scan_bin_col,
            "det_bin_row": self.det_bin_row,
            "det_bin_col": self.det_bin_col,
            "bin_mode": self.bin_mode,
            "edge_mode": self.edge_mode,
            "center_row": self.center_row,
            "center_col": self.center_col,
            "bf_radius_ratio": self.bf_radius_ratio,
            "adf_inner_ratio": self.adf_inner_ratio,
            "adf_outer_ratio": self.adf_outer_ratio,
            "cmap": self.cmap,
            "log_scale": self.log_scale,
            "auto_contrast": self.auto_contrast,
            "show_fft": self.show_fft,
            "show_controls": self.show_controls,
            "disabled_tools": list(self.disabled_tools),
            "hidden_tools": list(self.hidden_tools),
        }

    def save(self, path: str | pathlib.Path) -> None:
        save_state_file(path, "Bin4D", self.state_dict())

    def save_h5(
        self,
        path: str | pathlib.Path,
        source_file: str = "",
    ) -> pathlib.Path:
        """Save the current binned 4D data to HDF5 with bitshuffle + LZ4.

        The file can be reloaded with ``Bin.from_file(path)`` — calibration,
        center, and provenance metadata are stored as dataset attributes.

        Parameters
        ----------
        path : str or pathlib.Path
            Output ``.h5`` file path.
        source_file : str
            Optional provenance string (e.g. original master file path).

        Returns
        -------
        pathlib.Path
            The written file path.
        """
        if not _HAS_H5PY:
            raise ImportError("h5py is required for save_h5(). Install: pip install h5py")
        output_path = pathlib.Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        arr = self._binned_data_torch.detach().cpu().numpy().astype(np.float32, copy=False)
        compression_kwargs = {}
        try:
            compression_kwargs = dict(hdf5plugin.Bitshuffle(cname="lz4"))
        except Exception:
            pass
        t0 = time.time()
        with h5py.File(output_path, "w") as f:
            ds = f.create_dataset(
                "data",
                data=arr,
                chunks=True,
                **compression_kwargs,
            )
            ds.attrs["pixel_size_row"] = float(self.binned_pixel_size_row)
            ds.attrs["pixel_size_col"] = float(self.binned_pixel_size_col)
            ds.attrs["pixel_unit"] = self.pixel_unit
            ds.attrs["k_pixel_size_row"] = float(self.binned_k_pixel_size_row)
            ds.attrs["k_pixel_size_col"] = float(self.binned_k_pixel_size_col)
            ds.attrs["k_unit"] = self.k_unit
            ds.attrs["center_row"] = float(self.binned_center_row)
            ds.attrs["center_col"] = float(self.binned_center_col)
            ds.attrs["title"] = self.title
            ds.attrs["bin_factors"] = [
                int(self.scan_bin_row),
                int(self.scan_bin_col),
                int(self.det_bin_row),
                int(self.det_bin_col),
            ]
            if source_file:
                ds.attrs["source_file"] = str(source_file)
        elapsed = time.time() - t0
        size_gb = float(output_path.stat().st_size) / 1_000_000_000.0
        print(f"  Saved {output_path.name}: {arr.shape} in {elapsed:.1f}s ({size_gb:.2f} GB)")
        return output_path

    def load_state_dict(self, state: dict) -> None:
        for key, value in state.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @property
    def result(self):
        """Current binned data as torch tensor."""
        return self._binned_data_torch

    def get_binned_data(self, copy: bool = True, as_numpy: bool = False):
        """Return current binned data (torch by default)."""
        if as_numpy:
            arr = self._binned_data_torch.detach().cpu().numpy().astype(np.float32, copy=False)
            return arr.copy() if copy else arr
        return self._binned_data_torch.clone() if copy else self._binned_data_torch

    def set_data(
        self,
        data,
        scan_shape: tuple[int, int] | None = None,
        pixel_size: float | tuple[float, float] | None = None,
        k_pixel_size: float | tuple[float, float] | None = None,
        center: tuple[float, float] | None = None,
    ) -> Self:
        """Replace the 4D data while preserving display settings."""
        dataset_pixel: float | tuple[float, float] | None = None
        dataset_k: float | tuple[float, float] | None = None
        dataset_pixel_unit = "px"
        dataset_k_unit = "px"
        pixel_calibrated = False
        k_calibrated = False

        if hasattr(data, "sampling") and hasattr(data, "array"):
            units = list(getattr(data, "units", ["pixels"] * 4))
            sampling = list(getattr(data, "sampling", [1.0] * 4))
            if len(units) >= 2 and len(sampling) >= 2 and units[0] in _REAL_UNITS:
                sy, sx = float(sampling[0]), float(sampling[1])
                if units[0] == "nm":
                    sy *= 10.0
                if units[1] == "nm":
                    sx *= 10.0
                dataset_pixel = (sy, sx)
                dataset_pixel_unit = "Å"
                pixel_calibrated = True
            if len(units) >= 4 and len(sampling) >= 4 and units[2] in _K_UNITS:
                ky, kx = float(sampling[2]), float(sampling[3])
                dataset_k = (ky, kx)
                dataset_k_unit = units[2]
                k_calibrated = True
            data = data.array

        p_row, p_col = _as_pair(pixel_size if pixel_size is not None else dataset_pixel, 1.0)
        k_row, k_col = _as_pair(k_pixel_size if k_pixel_size is not None else dataset_k, 1.0)
        if pixel_size is not None:
            pixel_calibrated = True
            dataset_pixel_unit = "Å"
        if k_pixel_size is not None:
            k_calibrated = True
            dataset_k_unit = "mrad"

        self.pixel_size_row = p_row
        self.pixel_size_col = p_col
        self.pixel_unit = dataset_pixel_unit if pixel_calibrated else "px"
        self.pixel_calibrated = pixel_calibrated
        self.k_pixel_size_row = k_row
        self.k_pixel_size_col = k_col
        self.k_unit = dataset_k_unit if k_calibrated else "px"
        self.k_calibrated = k_calibrated

        data_np = to_numpy(data, dtype=np.float32)
        if data_np.ndim == 4:
            scan_r, scan_c, det_r, det_c = data_np.shape
            data4d = data_np
        elif data_np.ndim == 3:
            if scan_shape is None:
                n = int(data_np.shape[0])
                side = int(math.isqrt(n))
                if side * side != n:
                    raise ValueError(
                        f"Cannot infer square scan_shape from flattened N={n}. Provide scan_shape=(rows, cols)."
                    )
                scan_shape = (side, side)
            if int(scan_shape[0]) * int(scan_shape[1]) != int(data_np.shape[0]):
                raise ValueError(f"scan_shape={scan_shape} does not match flattened length {data_np.shape[0]}")
            scan_r, scan_c = int(scan_shape[0]), int(scan_shape[1])
            det_r, det_c = int(data_np.shape[1]), int(data_np.shape[2])
            data4d = data_np.reshape(scan_r, scan_c, det_r, det_c)
        else:
            raise ValueError(f"Expected 4D or flattened 3D array, got {data_np.ndim}D")

        self.scan_rows = scan_r
        self.scan_cols = scan_c
        self.det_rows = det_r
        self.det_cols = det_c
        self.max_scan_bin_row = max(1, scan_r)
        self.max_scan_bin_col = max(1, scan_c)
        self.max_det_bin_row = max(1, det_r)
        self.max_det_bin_col = max(1, det_c)

        data4d_writable = np.array(data4d, dtype=np.float32, copy=True)
        self._data_torch = torch.from_numpy(data4d_writable).to(self._device)

        if center is not None:
            self.center_row = float(center[0])
            self.center_col = float(center[1])
        else:
            self.center_row = det_r / 2.0
            self.center_col = det_c / 2.0

        self.scan_bin_row = 1
        self.scan_bin_col = 1
        self.det_bin_row = 1
        self.det_bin_col = 1
        self._binned_data_torch = self._data_torch
        self._original_bf_torch = torch.zeros((scan_r, scan_c), dtype=torch.float32)
        self._original_adf_torch = torch.zeros((scan_r, scan_c), dtype=torch.float32)
        self._binned_bf_torch = torch.zeros((scan_r, scan_c), dtype=torch.float32)
        self._binned_adf_torch = torch.zeros((scan_r, scan_c), dtype=torch.float32)
        self.original_mean_dp_bytes = b""
        self.binned_mean_dp_bytes = b""
        self._recompute_previews()
        return self

    def to_show4dstem(self, **kwargs):
        """Create a `Show4DSTEM` instance from the current binned data.

        Notes
        -----
        `Show4DSTEM` currently accepts scalar calibrations; row/col calibrations
        are collapsed using arithmetic mean for compatibility.
        """
        from quantem.widget.show4dstem import Show4DSTEM

        pixel_size = kwargs.pop(
            "pixel_size",
            float((self.binned_pixel_size_row + self.binned_pixel_size_col) / 2.0),
        )

        if self.k_calibrated:
            k_pixel_size = kwargs.pop(
                "k_pixel_size",
                float((self.binned_k_pixel_size_row + self.binned_k_pixel_size_col) / 2.0),
            )
        else:
            k_pixel_size = kwargs.pop("k_pixel_size", None)

        center = kwargs.pop(
            "center",
            (
                float(self.center_row / self.det_bin_row),
                float(self.center_col / self.det_bin_col),
            ),
        )

        bf_radius = kwargs.pop(
            "bf_radius",
            float(self.bf_radius_ratio * min(self.binned_det_rows, self.binned_det_cols)),
        )

        return Show4DSTEM(
            self.get_binned_data(copy=False),
            pixel_size=pixel_size,
            k_pixel_size=k_pixel_size,
            center=center,
            bf_radius=bf_radius,
            **kwargs,
        )

    def save_image(
        self,
        path: str | pathlib.Path,
        view: str = "binned_bf",
        cmap: str = "inferno",
        scale_mode: str = "linear",
        format: str | None = None,
        include_metadata: bool = True,
        metadata_path: str | pathlib.Path | None = None,
        dpi: int = 300,
    ) -> pathlib.Path:
        """Save one Bin preview view (or 2x2 grid) as PNG/PDF with metadata."""
        from PIL import Image

        output_path = pathlib.Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fmt = self._resolve_export_format(output_path, format)

        image, render_meta = self._render_view_image(view=view, cmap=cmap, scale_mode=scale_mode)
        if fmt == "png":
            image.save(output_path, format="PNG", dpi=(dpi, dpi))
        else:
            image.save(output_path, format="PDF", resolution=dpi)

        if include_metadata:
            meta_path = (
                pathlib.Path(metadata_path)
                if metadata_path is not None
                else output_path.with_suffix(".json")
            )
            metadata = {
                **build_json_header("Bin4D"),
                "view": view,
                "format": fmt,
                "export_kind": "single_view_image",
                "path": str(output_path),
                "bin_factors": {
                    "scan_row": int(self.scan_bin_row),
                    "scan_col": int(self.scan_bin_col),
                    "det_row": int(self.det_bin_row),
                    "det_col": int(self.det_bin_col),
                },
                "bin_mode": self.bin_mode,
                "edge_mode": self.edge_mode,
                "shape": {
                    "input": [int(self.scan_rows), int(self.scan_cols), int(self.det_rows), int(self.det_cols)],
                    "output": [
                        int(self.binned_scan_rows),
                        int(self.binned_scan_cols),
                        int(self.binned_det_rows),
                        int(self.binned_det_cols),
                    ],
                },
                "calibration": {
                    "pixel_size": [float(self.pixel_size_row), float(self.pixel_size_col)],
                    "pixel_size_binned": [
                        float(self.binned_pixel_size_row),
                        float(self.binned_pixel_size_col),
                    ],
                    "k_pixel_size": [float(self.k_pixel_size_row), float(self.k_pixel_size_col)],
                    "k_pixel_size_binned": [
                        float(self.binned_k_pixel_size_row),
                        float(self.binned_k_pixel_size_col),
                    ],
                },
                "render": render_meta,
            }
            meta_path.write_text(json.dumps(metadata, indent=2))

        return output_path

    def save_zip(
        self,
        path: str | pathlib.Path,
        cmap: str = "inferno",
        scale_mode: str = "linear",
        include_arrays: bool = False,
    ) -> pathlib.Path:
        """Export all Bin previews + metadata in a ZIP bundle."""
        import io
        import zipfile

        zip_path = pathlib.Path(path)
        zip_path.parent.mkdir(parents=True, exist_ok=True)

        panels = ["original_bf", "original_adf", "binned_bf", "binned_adf", "grid"]
        metadata = {
            **build_json_header("Bin4D"),
            "format": "zip",
            "export_kind": "multi_panel_bundle",
            "include_arrays": bool(include_arrays),
            "bin_factors": {
                "scan_row": int(self.scan_bin_row),
                "scan_col": int(self.scan_bin_col),
                "det_row": int(self.det_bin_row),
                "det_col": int(self.det_bin_col),
            },
            "panels": {},
        }

        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for panel in panels:
                image, render_meta = self._render_view_image(view=panel, cmap=cmap, scale_mode=scale_mode)
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                zf.writestr(f"{panel}.png", buf.getvalue())
                metadata["panels"][panel] = render_meta

            if include_arrays:
                arrays = {
                    "original_bf.npy": self._original_bf_torch,
                    "original_adf.npy": self._original_adf_torch,
                    "binned_bf.npy": self._binned_bf_torch,
                    "binned_adf.npy": self._binned_adf_torch,
                }
                for name, tensor in arrays.items():
                    arr_buf = io.BytesIO()
                    np.save(arr_buf, tensor.detach().cpu().numpy().astype(np.float32, copy=False))
                    zf.writestr(name, arr_buf.getvalue())

            zf.writestr("metadata.json", json.dumps(metadata, indent=2))

        return zip_path

    def save_gif(
        self,
        path: str | pathlib.Path,
        channel: str = "bf",
        cmap: str = "inferno",
        scale_mode: str = "linear",
        duration_ms: int = 800,
        loop: int = 0,
        include_metadata: bool = True,
        metadata_path: str | pathlib.Path | None = None,
    ) -> pathlib.Path:
        """Save a two-frame GIF comparing original vs binned BF/ADF previews."""
        from PIL import Image

        channel_key = str(channel).strip().lower()
        if channel_key not in {"bf", "adf"}:
            raise ValueError("channel must be 'bf' or 'adf'")

        left = "original_bf" if channel_key == "bf" else "original_adf"
        right = "binned_bf" if channel_key == "bf" else "binned_adf"
        img_left, left_meta = self._render_view_image(view=left, cmap=cmap, scale_mode=scale_mode)
        img_right, right_meta = self._render_view_image(view=right, cmap=cmap, scale_mode=scale_mode)

        gif_path = pathlib.Path(path)
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        img_left.save(
            gif_path,
            format="GIF",
            save_all=True,
            append_images=[img_right],
            duration=max(10, int(duration_ms)),
            loop=max(0, int(loop)),
        )

        if include_metadata:
            meta_path = pathlib.Path(metadata_path) if metadata_path is not None else gif_path.with_suffix(".json")
            metadata = {
                **build_json_header("Bin4D"),
                "format": "gif",
                "export_kind": "before_after_animation",
                "path": str(gif_path),
                "channel": channel_key,
                "duration_ms": int(max(10, int(duration_ms))),
                "loop": int(max(0, int(loop))),
                "display": {
                    "left": left_meta,
                    "right": right_meta,
                },
                "bin_factors": {
                    "scan_row": int(self.scan_bin_row),
                    "scan_col": int(self.scan_bin_col),
                    "det_row": int(self.det_bin_row),
                    "det_col": int(self.det_bin_col),
                },
                "bin_mode": self.bin_mode,
                "edge_mode": self.edge_mode,
            }
            meta_path.write_text(json.dumps(metadata, indent=2))
        return gif_path

    def summary(self) -> None:
        """Print compact binning + calibration summary."""
        name = self.title if self.title else "Bin"
        lines = [name, "═" * 32]
        lines.append(f"Device:   {self.device}")
        lines.append(
            f"Shape:    ({self.scan_rows}, {self.scan_cols}, {self.det_rows}, {self.det_cols})"
            f" -> ({self.binned_scan_rows}, {self.binned_scan_cols}, {self.binned_det_rows}, {self.binned_det_cols})"
        )
        lines.append(
            "Factors:  "
            f"scan=({self.scan_bin_row}, {self.scan_bin_col}), "
            f"det=({self.det_bin_row}, {self.det_bin_col}), "
            f"mode={self.bin_mode}, edge={self.edge_mode}"
        )
        lines.append(
            f"Real cal: ({self.pixel_size_row:.4g}, {self.pixel_size_col:.4g}) {self.pixel_unit}/px"
            f" -> ({self.binned_pixel_size_row:.4g}, {self.binned_pixel_size_col:.4g})"
        )
        lines.append(
            f"K cal:    ({self.k_pixel_size_row:.4g}, {self.k_pixel_size_col:.4g}) {self.k_unit}/px"
            f" -> ({self.binned_k_pixel_size_row:.4g}, {self.binned_k_pixel_size_col:.4g})"
        )
        if self.disabled_tools:
            lines.append(f"Locked:   {', '.join(self.disabled_tools)}")
        if self.hidden_tools:
            lines.append(f"Hidden:   {', '.join(self.hidden_tools)}")
        if self.status_message:
            lines.append(f"Status:   {self.status_level.upper()} - {self.status_message}")
        print("\n".join(lines))

    # ------------------------------------------------------------------
    # Internal compute
    # ------------------------------------------------------------------

    def _on_params_changed(self, change=None):
        self._recompute_previews()

    def _on_position_changed(self, change=None):
        pos = self._scan_position
        if len(pos) != 2 or pos[0] < 0 or pos[1] < 0:
            self._position_dp_bytes = b""
            self._binned_position_dp_bytes = b""
            return
        r = min(max(0, pos[0]), self.scan_rows - 1)
        c = min(max(0, pos[1]), self.scan_cols - 1)
        dp = self._data_torch[r, c, :, :]
        self._position_dp_bytes = dp.detach().cpu().contiguous().float().numpy().astype(np.float32, copy=False).tobytes()
        br = min(r // max(1, self.scan_bin_row), self.binned_scan_rows - 1)
        bc = min(c // max(1, self.scan_bin_col), self.binned_scan_cols - 1)
        dp_binned = self._binned_data_torch[br, bc, :, :]
        self._binned_position_dp_bytes = dp_binned.detach().cpu().contiguous().float().numpy().astype(np.float32, copy=False).tobytes()

    def _on_npy_export(self, change=None):
        import io

        if not self._npy_export_requested:
            return
        self._npy_export_requested = False
        arr = self._binned_data_torch.detach().cpu().numpy().astype(np.float32, copy=False)
        buf = io.BytesIO()
        np.save(buf, arr)
        self._npy_export_data = buf.getvalue()

    def _resolve_torch_device(self, requested: str | None, numel: int) -> str | None:
        """Pick a valid torch device from user request or quantem config."""
        if not _HAS_TORCH:
            return None

        if requested is not None:
            device_str = str(requested).strip().lower()
        elif _HAS_VALIDATE_DEVICE:
            device_str, _ = validate_device(None)
        else:
            device_str = (
                "mps"
                if torch.backends.mps.is_available()
                else "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        # MPS has tensor-size limitations similar to Show4DSTEM.
        if device_str == "mps" and numel > 2**31 - 1:
            device_str = "cpu"

        try:
            torch.zeros(1, device=torch.device(device_str))
        except Exception:
            return None

        return device_str

    def _resolve_export_format(
        self,
        path: pathlib.Path,
        fmt: str | None,
    ) -> str:
        if fmt is not None and str(fmt).strip():
            resolved = str(fmt).strip().lower()
        else:
            resolved = path.suffix.lstrip(".").lower() or "png"
        if resolved not in {"png", "pdf"}:
            raise ValueError(f"Unsupported format '{resolved}'. Supported: png, pdf")
        return resolved

    def _get_panel_tensor(self, view: str):
        key = str(view).strip().lower()
        mapping = {
            "original_bf": self._original_bf_torch,
            "original_adf": self._original_adf_torch,
            "binned_bf": self._binned_bf_torch,
            "binned_adf": self._binned_adf_torch,
        }
        if key not in mapping:
            raise ValueError(
                "view must be one of: original_bf, original_adf, binned_bf, binned_adf, grid"
            )
        return mapping[key]

    def _tensor_to_rgb(self, tensor, cmap: str, scale_mode: str) -> np.ndarray:
        from matplotlib import colormaps

        arr = tensor.detach().cpu().float()
        if scale_mode == "log":
            arr = torch.log1p(torch.clamp_min(arr, 0.0))
        elif scale_mode == "power":
            arr = torch.sqrt(torch.clamp_min(arr, 0.0))
        elif scale_mode != "linear":
            raise ValueError("scale_mode must be 'linear', 'log', or 'power'")

        if arr.numel() == 0:
            dmin = 0.0
            dmax = 0.0
        else:
            dmin = float(torch.min(arr))
            dmax = float(torch.max(arr))

        if dmax <= dmin:
            normalized = torch.zeros_like(arr, dtype=torch.float32)
        else:
            normalized = torch.clamp((arr - dmin) / (dmax - dmin), 0.0, 1.0)
        rgba = colormaps.get_cmap(cmap)(normalized.numpy())
        return (rgba[..., :3] * 255).astype(np.uint8)

    def _render_view_image(self, view: str, cmap: str, scale_mode: str):
        from PIL import Image

        view_key = str(view).strip().lower()
        if view_key == "grid":
            panels = ["original_bf", "original_adf", "binned_bf", "binned_adf"]
            rgbs = [self._tensor_to_rgb(self._get_panel_tensor(v), cmap=cmap, scale_mode=scale_mode) for v in panels]
            h0 = max(int(rgbs[0].shape[0]), int(rgbs[1].shape[0]))
            h1 = max(int(rgbs[2].shape[0]), int(rgbs[3].shape[0]))
            w0 = max(int(rgbs[0].shape[1]), int(rgbs[2].shape[1]))
            w1 = max(int(rgbs[1].shape[1]), int(rgbs[3].shape[1]))
            grid = np.zeros((h0 + h1, w0 + w1, 3), dtype=np.uint8)
            grid[: rgbs[0].shape[0], : rgbs[0].shape[1]] = rgbs[0]
            grid[: rgbs[1].shape[0], w0 : w0 + rgbs[1].shape[1]] = rgbs[1]
            grid[h0 : h0 + rgbs[2].shape[0], : rgbs[2].shape[1]] = rgbs[2]
            grid[h0 : h0 + rgbs[3].shape[0], w0 : w0 + rgbs[3].shape[1]] = rgbs[3]
            meta = {"view": "grid", "panels": panels, "colormap": cmap, "scale_mode": scale_mode}
            return Image.fromarray(grid, mode="RGB"), meta

        panel = self._get_panel_tensor(view_key)
        rgb = self._tensor_to_rgb(panel, cmap=cmap, scale_mode=scale_mode)
        meta = {
            "view": view_key,
            "shape": [int(panel.shape[0]), int(panel.shape[1])],
            "colormap": cmap,
            "scale_mode": scale_mode,
        }
        return Image.fromarray(rgb, mode="RGB"), meta

    def _recompute_previews(self) -> None:
        # Ensure annulus is valid
        if self.adf_outer_ratio <= self.adf_inner_ratio:
            self.status_level = "warn"
            self.status_message = "ADF outer ratio must be greater than inner ratio."
            self.adf_outer_ratio = float(self.adf_inner_ratio + 1e-3)
            return

        # Torch-only compute path
        try:
            binned_t = self._bin_4d_torch(
                self._data_torch,
                factors=(self.scan_bin_row, self.scan_bin_col, self.det_bin_row, self.det_bin_col),
                mode=self.bin_mode,
                edge=self.edge_mode,
            )

            orig_bf_t, orig_adf_t = self._virtual_images_torch(
                self._data_torch,
                center=(self.center_row, self.center_col),
                bf_ratio=self.bf_radius_ratio,
                adf_inner_ratio=self.adf_inner_ratio,
                adf_outer_ratio=self.adf_outer_ratio,
            )
        except ValueError as exc:
            self.status_level = "error"
            self.status_message = str(exc)
            return
        except Exception as exc:
            self.status_level = "error"
            self.status_message = str(exc)
            return

        self._binned_data_torch = binned_t
        self.binned_scan_rows = int(binned_t.shape[0])
        self.binned_scan_cols = int(binned_t.shape[1])
        self.binned_det_rows = int(binned_t.shape[2])
        self.binned_det_cols = int(binned_t.shape[3])

        self.binned_pixel_size_row = float(self.pixel_size_row * self.scan_bin_row)
        self.binned_pixel_size_col = float(self.pixel_size_col * self.scan_bin_col)
        self.binned_k_pixel_size_row = float(self.k_pixel_size_row * self.det_bin_row)
        self.binned_k_pixel_size_col = float(self.k_pixel_size_col * self.det_bin_col)

        # Detector center maps with detector bin factors (clamped to bounds)
        self.binned_center_row = float(
            max(0.0, min(self.binned_det_rows - 1, self.center_row / self.det_bin_row))
        )
        self.binned_center_col = float(
            max(0.0, min(self.binned_det_cols - 1, self.center_col / self.det_bin_col))
        )

        binned_bf_t, binned_adf_t = self._virtual_images_torch(
            binned_t,
            center=(self.binned_center_row, self.binned_center_col),
            bf_ratio=self.bf_radius_ratio,
            adf_inner_ratio=self.adf_inner_ratio,
            adf_outer_ratio=self.adf_outer_ratio,
        )

        self._original_bf_torch = orig_bf_t.detach().cpu().float()
        self._original_adf_torch = orig_adf_t.detach().cpu().float()
        self._binned_bf_torch = binned_bf_t.detach().cpu().float()
        self._binned_adf_torch = binned_adf_t.detach().cpu().float()

        self.original_bf_bytes = (
            orig_bf_t.detach().cpu().contiguous().numpy().astype(np.float32, copy=False).tobytes()
        )
        self.original_adf_bytes = (
            orig_adf_t.detach().cpu().contiguous().numpy().astype(np.float32, copy=False).tobytes()
        )
        self.binned_bf_bytes = (
            binned_bf_t.detach().cpu().contiguous().numpy().astype(np.float32, copy=False).tobytes()
        )
        self.binned_adf_bytes = (
            binned_adf_t.detach().cpu().contiguous().numpy().astype(np.float32, copy=False).tobytes()
        )

        # Mean diffraction patterns (detector-space previews)
        mean_dp_orig = self._data_torch.mean(dim=(0, 1))
        self.original_mean_dp_bytes = (
            mean_dp_orig.detach().cpu().contiguous().float().numpy().astype(np.float32, copy=False).tobytes()
        )
        mean_dp_binned = binned_t.mean(dim=(0, 1))
        self.binned_mean_dp_bytes = (
            mean_dp_binned.detach().cpu().contiguous().float().numpy().astype(np.float32, copy=False).tobytes()
        )

        self.original_bf_stats = _qc_stats_torch(orig_bf_t)
        self.original_adf_stats = _qc_stats_torch(orig_adf_t)
        self.binned_bf_stats = _qc_stats_torch(binned_bf_t)
        self.binned_adf_stats = _qc_stats_torch(binned_adf_t)

        self.status_level = "ok"
        self.status_message = (
            f"Preview updated on torch/{self.device}: "
            f"({self.scan_rows}×{self.scan_cols}×{self.det_rows}×{self.det_cols})"
            f" -> ({self.binned_scan_rows}×{self.binned_scan_cols}×{self.binned_det_rows}×{self.binned_det_cols})"
        )

        # Refresh per-position DP if a position is selected
        self._on_position_changed()

    def _virtual_images_torch(
        self,
        data4d,
        center: tuple[float, float],
        bf_ratio: float,
        adf_inner_ratio: float,
        adf_outer_ratio: float,
    ):
        """Compute BF/ADF virtual images using torch tensors."""
        det_rows, det_cols = int(data4d.shape[2]), int(data4d.shape[3])
        center_row, center_col = float(center[0]), float(center[1])
        det_size = float(min(det_rows, det_cols))

        bf_radius = max(1e-6, bf_ratio * det_size)
        adf_inner = max(0.0, adf_inner_ratio * det_size)
        adf_outer = max(adf_inner + 1e-6, adf_outer_ratio * det_size)

        rr = torch.arange(det_rows, device=data4d.device, dtype=torch.float32)[:, None]
        cc = torch.arange(det_cols, device=data4d.device, dtype=torch.float32)[None, :]
        dist2 = (rr - center_row) ** 2 + (cc - center_col) ** 2

        bf_mask = (dist2 <= bf_radius**2).float()
        adf_mask = ((dist2 >= adf_inner**2) & (dist2 <= adf_outer**2)).float()

        bf = torch.tensordot(data4d, bf_mask, dims=([2, 3], [0, 1]))
        adf = torch.tensordot(data4d, adf_mask, dims=([2, 3], [0, 1]))
        return bf, adf

    def _bin_axis_torch(self, data, axis: int, factor: int, mode: str, edge: str):
        """Torch equivalent of `_bin_axis`."""
        if factor == 1:
            return data

        n = int(data.shape[axis])

        if edge == "crop":
            n_used = (n // factor) * factor
            if n_used <= 0:
                raise ValueError(
                    f"crop mode: factor {factor} is larger than axis size {n} for axis {axis}"
                )
            trimmed = data.narrow(axis, 0, n_used)
        elif edge == "pad":
            n_used = int(math.ceil(n / factor) * factor)
            pad_amount = n_used - n
            if pad_amount > 0:
                pad_shape = list(data.shape)
                pad_shape[axis] = pad_amount
                pad_block = torch.zeros(
                    pad_shape,
                    dtype=data.dtype,
                    device=data.device,
                )
                trimmed = torch.cat([data, pad_block], dim=axis)
            else:
                trimmed = data
        else:  # edge == "error"
            if n % factor != 0:
                raise ValueError(
                    f"error mode: axis size {n} is not divisible by factor {factor} (axis {axis})"
                )
            n_used = n
            trimmed = data

        new_shape = (
            tuple(trimmed.shape[:axis])
            + (n_used // factor, factor)
            + tuple(trimmed.shape[axis + 1 :])
        )
        reshaped = trimmed.reshape(new_shape)
        reduce_axis = axis + 1

        if mode == "sum":
            return reshaped.sum(dim=reduce_axis)
        return reshaped.mean(dim=reduce_axis)

    def _bin_4d_torch(
        self,
        data4d,
        factors: tuple[int, int, int, int],
        mode: str,
        edge: str,
    ):
        out = data4d
        for axis, factor in enumerate(factors):
            out = self._bin_axis_torch(out, axis=axis, factor=int(factor), mode=mode, edge=edge)
        return out.float()

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        title_info = f", title='{self.title}'" if self.title else ""
        return (
            "Bin4D("
            f"shape=({self.scan_rows}, {self.scan_cols}, {self.det_rows}, {self.det_cols}), "
            f"bin=({self.scan_bin_row}, {self.scan_bin_col}, {self.det_bin_row}, {self.det_bin_col}), "
            f"binned_shape=({self.binned_scan_rows}, {self.binned_scan_cols}, {self.binned_det_rows}, {self.binned_det_cols}), "
            f"mode={self.bin_mode}, edge={self.edge_mode}, device={self.device}"
            f"{title_info})"
        )


bind_tool_runtime_api(Bin4D, "Bin4D")

# Backwards-compatible alias
Bin = Bin4D
