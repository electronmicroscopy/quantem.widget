"""Show4DSTEM: scan position, virtual detector and diffraction pattern of a 4D-STEM scan.

One class opens every input the tutorials use: a file path (read with
``read_4dstem``), a ``quantem.gpu.io.Dataset4dstemGPU`` (kept on its GPU, read
in bounded windows), a quantem core ``Dataset4dstem``, a list of either (one
virtual image per dataset under a shared detector) or ``from_folder``, a NumPy
array or a Torch tensor (dense, on the tensor's device), or a folder of linked
HDF5 masters read by the browser itself (``h5_urls``, the CLI WebGPU export).

quantem.gpu and quantem core are reached only through
``quantem.widget.adapters``. A file becomes a ``Dataset4dstemGPU`` when
quantem.gpu is installed and a GPU is present, else ``show4dstem.reader``
reads it densely onto the compute device (``read_4dstem`` gives the same
counts as a quantem core ``Dataset4dstem``); the viewer shows the same numbers
for either. Dense data reduce in ``show4dstem.dense`` (the
widget's definition of the math), GPU acquisitions in quantem.gpu's detector
sessions; this file owns the traits the frontend syncs and the bookkeeping
between them.
"""

import json
import math
import os
import pathlib
import threading
import warnings
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Self

import anywidget
import numpy as np
import torch
import traitlets
from tornado.ioloop import IOLoop

from quantem.widget.adapters import core as core_adapter
from quantem.widget.adapters import gpu as gpu_adapter
from quantem.widget.adapters.gpu import AcquisitionSeries, AcquisitionView
from quantem.widget.counts import counts_tensor, detector_bin_mean, host_dtype_for_torch, tensor_dtype
from quantem.widget.device import gpu_notice, gpu_path_hint, no_gpu_path, print_once, resolve_device
from quantem.widget.folder_watch_status import FOLDER_WATCH_STATE_VALUES, set_folder_watch_status
from quantem.widget.export import HtmlExportMixin
from quantem.widget.fallback import StaticFallbackMixin
from quantem.widget.show4dstem import export as html_export
from quantem.widget.show4dstem import reader
from quantem.widget.show4dstem.reader import acquisition_name
from quantem.widget.show4dstem.dense import DenseSession
from quantem.widget.show4dstem.detector import PRESET_RADII, fit_disk, preset_mask, roi_mask, scan_indices
from quantem.widget.show4dstem.preview import static_png_b64
from quantem.widget.state import SavedStateMixin
from quantem.widget.utils.state_io import unwrap_state_payload
from quantem.widget.utils.ui import UiMode, resolve_ui_mode

DEFAULT_BF_RATIO = 0.125  # bright-field radius as a fraction of the detector side before fitting
MIN_LOG_VALUE = 1e-10  # keeps log scaling finite on an all-zero pattern
DEFAULT_VI_ROI_RATIO = 0.15  # scan ROI radius as a fraction of the scan side
PRODUCT_LABELS = ("DPC_row", "DPC_col", "SSB")
# Soft guide for the inline offline pack: the browser's JSON parse of the embedded
# base64 is the real ceiling, and gzip keeps a 2 GB uint8 stack under it.
OFFLINE_BUDGET_BYTES = 2000 * 1024 * 1024


def dataset_name(dataset) -> str:
    """The name a dataset carries: a GPU acquisition's metadata ``name`` or its file, else the dataset's ``name``.

    ``read_4dstem`` names both kinds after the file, so a path opens with the
    same title and labels whichever kind this machine reads it into.
    """
    if gpu_adapter.is_acquisition(dataset):
        return str(dataset.metadata.get("name") or acquisition_name(dataset.metadata.get("source_path", "")))
    return str(dataset.name)


def master_contract(master) -> dict:
    """Scan shape, detector shape, frame count and dtype of a complete master.

    A master still being written, or one whose header lacks a shape, cannot
    join a comparison; the error carries the loader's corrective action.
    """
    report = gpu_adapter.inspect_master(master)
    required = {
        "scan_shape": report.scan_shape,
        "detector_shape": report.detector_shape,
        "n_frames": report.actual_frames,
        "dtype": report.dtype,
    }
    missing = [name for name, value in required.items() if value is None]
    if not report.ready or missing:
        problems = [] if report.ready else [report.reason or "the source is not ready"]
        if missing:
            problems.append("inspection did not provide " + ", ".join(missing))
        action = report.action or "Verify that the master and external data are complete."
        raise ValueError(f"Cannot open 4D-STEM master {str(master)!r}: {'; '.join(problems)}. {action}")
    required["dtype"] = np.dtype(report.dtype).str
    return required


def readiness_is_waiting(report) -> bool:
    """Whether a not-ready master is still being written (wait) rather than broken (error)."""
    if report.actual_frames is not None and report.expected_frames is not None:
        return int(report.actual_frames) < int(report.expected_frames)
    reason = str(report.reason or "").casefold()
    action = str(report.action or "").strip().casefold()
    permanent = ("inconsistent detector", "inconsistent dtype", "inconsistent scan_shape", "incompatible",
                 "does not match", "conflicting", "expected at least (frame, det_row, det_col)")
    if any(marker in reason for marker in permanent) or action.startswith(("fix ", "repair ", "use ", "pass ")):
        return False
    if action.startswith("wait "):
        return True
    transient = ("missing", "empty", "zero stored frames", "not readable hdf5", "cannot be inspected",
                 "changed during readiness inspection", "headers are incomplete", "header to finish writing",
                 "no entry/data", "has no entry/data/data")
    return any(marker in reason for marker in transient)


class Show4DSTEM(SavedStateMixin, HtmlExportMixin, StaticFallbackMixin, anywidget.AnyWidget):
    """Interactive 4D-STEM viewer.

    Drag over the scan to pick a position and see its diffraction pattern; drag
    or resize the detector ROI on the pattern and the virtual image follows.
    BF, ABF, ADF and HAADF presets place the ROI around the fitted bright-field
    disk. Several datasets of one geometry open as a comparison grid with one
    shared detector. ``export_html`` writes a kernel-less page (the counts run on
    browser WebGPU) or a static report.

    Parameters
    ----------
    data
        A file path (``*_master.h5``, HDF5 or ``.npy``), a dataset from
        ``read_4dstem`` (a ``Dataset4dstemGPU`` or a quantem core
        ``Dataset4dstem``), a list of datasets, a 4D ``(scan_row, scan_col,
        det_row, det_col)`` or 5D ``(n_datasets, ...)`` array or tensor, or a
        3D ``(positions, det_row, det_col)`` flat scan with ``scan_shape``. A
        path opens as ``read_4dstem(path, device)`` reads it: on the GPU
        through quantem.gpu when it is installed and a GPU is present, else
        densely, block by block onto ``device`` (one printed line; refused
        above a memory ceiling, never reduced).
    device
        Where a NumPy array or a file is placed and reduced: ``"auto"`` (CUDA,
        then MPS, then CPU; printed once), ``"cuda"``, ``"cuda:N"``, ``"mps"``
        or ``"cpu"``. A tensor stays on its device unless this names another.
    scan_region
        ``(row_start, row_stop, col_start, col_stop)`` of one 4D scan to view,
        exclusive stops: a GPU acquisition is read only there, a dense scan is
        cut before it moves to its device.
    sampling, units
        Pixel sizes and units for the four axes; taken from the dataset when it
        carries a calibration.
    center, bf_radius
        Bright-field disk geometry in detector pixels; fitted from the mean
        pattern when omitted.
    precompute_virtual_images
        Compute the four preset images up front so preset clicks are instant.
    ssb_voltage_kV, ssb_semiangle_mrad, ssb_scan_sampling_A, ssb_det_sampling_mrad
        Physical calibration for ``compute_ssb`` when the dataset lacks it.
    view_mode, compare_cols, compare_grid_width_px, compare_max_panels,
    compare_group_mode, compare_dp_mode
        Comparison grid layout for several datasets: ``"multiple"`` shows one
        virtual image per dataset, ``compare_max_panels`` per page, with the
        diffraction pattern of the ``"selected"`` dataset or the ``"average"``.
    offline, offline_dtype
        Pack the counts for the browser so the widget works without a kernel
        (``None`` packs automatically under a 2 GB budget). A GPU acquisition
        is read into the pack in small scan windows.
    h5_urls, scan_shape, detector_shape, backend
        Browser WebGPU source over linked HDF5 masters, written by the CLI.
    """

    _esm = pathlib.Path(__file__).parent.parent / "static" / "show4dstem.js"

    title = traitlets.Unicode("").tag(sync=True)
    show_title = traitlets.Bool(True).tag(sync=True)
    folder_watch_state = traitlets.Enum(FOLDER_WATCH_STATE_VALUES, default_value="hidden").tag(sync=True)
    folder_watch_detail = traitlets.Unicode("").tag(sync=True)
    pos_row = traitlets.Int(0).tag(sync=True)
    pos_col = traitlets.Int(0).tag(sync=True)
    shape_rows = traitlets.Int(1).tag(sync=True)
    shape_cols = traitlets.Int(1).tag(sync=True)
    det_rows = traitlets.Int(1).tag(sync=True)
    det_cols = traitlets.Int(1).tag(sync=True)
    frame_bytes = traitlets.Bytes(b"").tag(sync=True)
    dp_global_min = traitlets.Float(0.0).tag(sync=True)
    dp_global_max = traitlets.Float(1.0).tag(sync=True)
    center_col = traitlets.Float(0.0).tag(sync=True)
    center_row = traitlets.Float(0.0).tag(sync=True)
    bf_radius = traitlets.Float(0.0).tag(sync=True)
    roi_active = traitlets.Bool(False).tag(sync=True)
    roi_mode = traitlets.Unicode("point").tag(sync=True)
    roi_center_col = traitlets.Float(0.0).tag(sync=True)
    roi_center_row = traitlets.Float(0.0).tag(sync=True)
    roi_center = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0]).tag(sync=True)
    roi_radius = traitlets.Float(10.0).tag(sync=True)
    roi_radius_inner = traitlets.Float(5.0).tag(sync=True)
    roi_width = traitlets.Float(20.0).tag(sync=True)
    roi_height = traitlets.Float(10.0).tag(sync=True)
    virtual_image_bytes = traitlets.Bytes(b"").tag(sync=True)
    vi_source = traitlets.Unicode("roi").tag(sync=True)
    vi_product_labels = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    vi_product_map_frames = traitlets.Int(0).tag(sync=True)
    vi_product_maps_bytes = traitlets.Bytes(b"").tag(sync=True)
    offline = traitlets.Bool(False).tag(sync=True)
    _offline_stack = traitlets.Bytes(b"").tag(sync=True)
    _h5_urls = traitlets.Unicode("").tag(sync=True)
    _offline_bad_px = traitlets.Unicode("").tag(sync=True)
    ssb_compute_request = traitlets.Unicode("").tag(sync=True)
    ssb_compute_status = traitlets.Unicode("").tag(sync=True)
    ssb_compute_busy = traitlets.Bool(False).tag(sync=True)
    ssb_compute_enabled = traitlets.Bool(True).tag(sync=True)
    ssb_compute_n_trials = traitlets.Int(200).tag(sync=True)
    ssb_compute_refine = traitlets.Bool(True).tag(sync=True)
    ssb_compute_bf_pixels = traitlets.Int(0).tag(sync=True)
    ssb_compute_lock_c10 = traitlets.Bool(False).tag(sync=True)
    ssb_compute_lock_c12 = traitlets.Bool(False).tag(sync=True)
    ssb_compute_c10_nm = traitlets.Float(0.0).tag(sync=True)
    ssb_compute_c12_nm = traitlets.Float(0.0).tag(sync=True)
    ssb_compute_phi12_deg = traitlets.Float(0.0).tag(sync=True)
    ssb_compute_rotation_angle_deg = traitlets.Float(0.0).tag(sync=True)
    ssb_compute_calibration_json = traitlets.Unicode("").tag(sync=True)
    ssb_compute_calibration_filename = traitlets.Unicode("").tag(sync=True)
    vi_roi_mode = traitlets.Unicode("off").tag(sync=True)
    vi_roi_center_row = traitlets.Float(0.0).tag(sync=True)
    vi_roi_center_col = traitlets.Float(0.0).tag(sync=True)
    vi_roi_center = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0]).tag(sync=True)
    vi_roi_radius = traitlets.Float(5.0).tag(sync=True)
    vi_roi_width = traitlets.Float(10.0).tag(sync=True)
    vi_roi_height = traitlets.Float(10.0).tag(sync=True)
    vi_roi_reduce = traitlets.Unicode("mean").tag(sync=True)
    vi_roi_dp_bytes = traitlets.Bytes(b"").tag(sync=True)
    vi_roi_receipt = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    pixel_size = traitlets.Float(1.0).tag(sync=True)
    pixel_unit = traitlets.Unicode("pixels").tag(sync=True)
    k_pixel_size = traitlets.Float(1.0).tag(sync=True)
    k_pixel_unit = traitlets.Unicode("pixels").tag(sync=True)
    k_calibrated = traitlets.Bool(False).tag(sync=True)
    _static_fallback_jpeg = traitlets.Unicode("").tag(sync=True)
    _static_fallback_mime = traitlets.Unicode("image/jpeg").tag(sync=True)
    dp_colormap = traitlets.Unicode("inferno").tag(sync=True)
    vi_colormap = traitlets.Unicode("inferno").tag(sync=True)
    fft_colormap = traitlets.Unicode("inferno").tag(sync=True)
    dp_scale_mode = traitlets.Unicode("log").tag(sync=True)
    vi_scale_mode = traitlets.Unicode("linear").tag(sync=True)
    fft_scale_mode = traitlets.Unicode("linear").tag(sync=True)
    dp_vmin_pct = traitlets.Float(0.0).tag(sync=True)
    dp_vmax_pct = traitlets.Float(100.0).tag(sync=True)
    vi_vmin_pct = traitlets.Float(0.0).tag(sync=True)
    vi_vmax_pct = traitlets.Float(100.0).tag(sync=True)
    fft_vmin_pct = traitlets.Float(0.0).tag(sync=True)
    fft_vmax_pct = traitlets.Float(100.0).tag(sync=True)
    dp_vmin = traitlets.Float(None, allow_none=True).tag(sync=True)
    dp_vmax = traitlets.Float(None, allow_none=True).tag(sync=True)
    vi_vmin = traitlets.Float(None, allow_none=True).tag(sync=True)
    vi_vmax = traitlets.Float(None, allow_none=True).tag(sync=True)
    fft_auto = traitlets.Bool(True).tag(sync=True)
    show_fft = traitlets.Bool(False).tag(sync=True)
    _preset_request = traitlets.Unicode("").tag(sync=True)
    fft_window = traitlets.Bool(True).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)
    controls_collapsed = traitlets.Bool(False).tag(sync=True)
    show_stats = traitlets.Bool(True).tag(sync=True)
    show_scale_bar = traitlets.Bool(True).tag(sync=True)
    panel_width_px = traitlets.Int(0).tag(sync=True)
    dp_show_colorbar = traitlets.Bool(False).tag(sync=True)
    vi_auto_contrast = traitlets.Bool(False).tag(sync=True)
    vi_smooth = traitlets.Bool(False).tag(sync=True)
    frame_idx = traitlets.Int(0).tag(sync=True)
    n_frames = traitlets.Int(1).tag(sync=True)
    frame_dim_label = traitlets.Unicode("Frame").tag(sync=True)
    frame_labels = traitlets.List(traitlets.Unicode(), []).tag(sync=True)
    frame_playing = traitlets.Bool(False).tag(sync=True)
    frame_loop = traitlets.Bool(True).tag(sync=True)
    frame_fps = traitlets.Float(5.0).tag(sync=True)
    frame_reverse = traitlets.Bool(False).tag(sync=True)
    frame_boomerang = traitlets.Bool(False).tag(sync=True)
    view_mode = traitlets.Unicode("single").tag(sync=True)
    compare_cols = traitlets.Int(0).tag(sync=True)
    compare_grid_width_px = traitlets.Int(0).tag(sync=True)
    compare_max_panels = traitlets.Int(12).tag(sync=True)
    compare_group_mode = traitlets.Unicode("paged").tag(sync=True)
    compare_page_idx = traitlets.Int(0).tag(sync=True)
    compare_page_count = traitlets.Int(1).tag(sync=True)
    compare_dp_mode = traitlets.Unicode("average").tag(sync=True)
    compare_diffraction_bytes = traitlets.Bytes(b"").tag(sync=True)
    compare_diffraction_indices = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    compare_panel_order = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    compare_hidden_panels = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    compare_starred_panels = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    compare_virtual_image_bytes = traitlets.Bytes(b"").tag(sync=True)
    compare_panel_count = traitlets.Int(0).tag(sync=True)
    compare_panel_indices = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    compare_status = traitlets.Unicode("").tag(sync=True)
    profile_line = traitlets.List(traitlets.Dict()).tag(sync=True)
    profile_width = traitlets.Int(1).tag(sync=True)

    # Bulk payload traits left out of the saved notebook unless save_state=True.
    _UNSAVED_HEAVY_KEYS = ("_offline_stack", "export_payload")

    def __init__(
        self,
        data,
        *,
        device: str = "auto",
        scan_region: tuple[int, int, int, int] | None = None,
        scan_shape: tuple[int, int] | None = None,
        detector_shape: tuple[int, int] | None = None,
        sampling: tuple[float, ...] | list[float] | float | None = None,
        units: list[str] | tuple[str, ...] | str | None = None,
        center: tuple[float, float] | None = None,
        bf_radius: float | None = None,
        precompute_virtual_images: bool | None = None,
        ssb_voltage_kV: float | None = None,
        ssb_semiangle_mrad: float | None = None,
        ssb_scan_sampling_A: float | tuple[float, float] | None = None,
        ssb_det_sampling_mrad: float | tuple[float, float] | None = None,
        frame_dim_label: str | None = None,
        frame_labels: list[str] | None = None,
        view_mode: str | None = None,
        compare_cols: int = 0,
        compare_grid_width_px: int = 0,
        compare_max_panels: int = 12,
        compare_group_mode: str = "paged",
        compare_dp_mode: str | None = None,
        title: str = "",
        ui_mode: UiMode = "interactive",
        show_title: bool | None = None,
        show_controls: bool | None = None,
        controls_collapsed: bool | None = None,
        show_stats: bool | None = None,
        show_scale_bar: bool | None = None,
        offline: bool | None = False,
        offline_dtype: str = "uint8",
        h5_urls: Sequence[str] | None = None,
        backend: str | None = None,
        show_fft: bool = False,
        panel_width_px: int = 0,
        verbose: bool = False,
        state=None,
        save_state: bool = False,
        notebook_preview_format: str | None = "jpeg",
        notebook_preview_quality: int = 88,
        notebook_preview_max_px: int = 512,
        **kwargs,
    ):
        """Validate the arguments, place the data on its compute device and set the traits (see the class docstring)."""
        # save_state=False keeps the packed 4D stack out of the notebook's saved
        # widget metadata; a static preview stands in for a cold reopen.
        self._save_state = bool(save_state)
        self._configure_static_fallback(
            notebook_preview_format=notebook_preview_format,
            notebook_preview_quality=notebook_preview_quality,
            notebook_preview_max_px=notebook_preview_max_px,
        )
        ui = resolve_ui_mode(
            ui_mode,
            defaults={"show_title": True, "show_controls": True, "controls_collapsed": False,
                      "show_stats": True, "show_scale_bar": True},
            overrides={"show_title": show_title, "show_controls": show_controls,
                       "controls_collapsed": controls_collapsed, "show_stats": show_stats,
                       "show_scale_bar": show_scale_bar},
        )
        self._data = None  # get_state runs inside open(); the data joins once the model exists
        self._verbose = bool(verbose)
        self._sessions: dict[int, object] = {}  # DenseSession or a quantem.gpu detector session per dataset
        self._vi_product_maps: dict[str, np.ndarray] = {}
        self._preset_images: dict[str, bytes] = {}
        self._suppress_roi_recompute = False
        self._suppress_compare_recompute = False
        self._folder_source = None
        self._owned_acquisitions = []
        self._folder_known_masters: set[str] = set()
        self._folder_ready_probation: dict[str, str] = {}
        self._folder_fill_thread = None
        self._folder_fill_stop = threading.Event()
        self._folder_watch_stop = None
        self._folder_watch_thread = None
        self._folder_watch_started = False
        self._folder_poll_lock = threading.Lock()
        self._compare_lock = threading.RLock()
        # Every check and the file read run before super().__init__ opens the widget model, so a
        # refused input (a bad argument, a file above the memory ceiling) leaves no model behind.
        if backend not in (None, "webgpu"):
            raise ValueError(
                f"backend must be 'webgpu' or None, got {backend!r}. "
                "Native CUDA or MPS compute is selected from the data payload."
            )
        offline_dtype = {"u8": "uint8", "u16": "uint16"}.get(str(offline_dtype).lower(), str(offline_dtype).lower())
        if offline_dtype not in ("uint8", "uint16"):
            raise ValueError(
                f"offline_dtype must be 'uint8' or 'uint16', got {offline_dtype!r}. "
                "Use 'uint8' for compact browse data or 'uint16' to preserve detector counts."
            )
        for name, value in (("panel_width_px", panel_width_px), ("compare_cols", compare_cols),
                            ("compare_grid_width_px", compare_grid_width_px)):
            if int(value) < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if int(compare_max_panels) < 1:
            raise ValueError(f"compare_max_panels must be >= 1, got {compare_max_panels}")
        if h5_urls and (scan_shape is None or detector_shape is None):
            raise ValueError("Show4DSTEM(..., h5_urls=...) requires scan_shape=(rows, cols) and detector_shape=(rows, cols).")

        # Route the input to its storage: bounded views over GPU acquisitions,
        # a dense tensor on its device, or no data at all for a browser source.
        if isinstance(data, (str, os.PathLike)) and reader.gpu_route(device):
            data = reader.read_4dstem(data, device, scan_shape=scan_shape)
            if gpu_adapter.is_acquisition(data):
                self._owned_acquisitions = [data]
        elif isinstance(data, (str, os.PathLike)):
            # read_4dstem's dense read, with the counts going straight onto the compute device block by block
            gpu_notice()
            title = title or acquisition_name(data)
            data = reader.read_dense(data, resolve_device(device), scan_shape=scan_shape)
        listed = isinstance(data, (list, tuple)) and len(data) > 0
        series = listed and any(gpu_adapter.is_acquisition(item) for item in data)
        bounded = gpu_adapter.is_acquisition(data) or series
        datasets = listed and all(core_adapter.is_dataset(item, ndim=4) for item in data)
        if bounded and h5_urls:
            raise ValueError("h5_urls= opens files in the browser; pass the acquisition without it, or the URLs alone.")
        if bounded or datasets or core_adapter.is_dataset(data, ndim=4):
            # A dataset (or a list of them) carries its name and calibration.
            first = data[0] if listed else data
            if listed:
                names = [dataset_name(item) for item in data] if datasets or all(map(gpu_adapter.is_acquisition, data)) else []
                if frame_labels is None and names and all(names):
                    frame_labels = names
                frame_dim_label = "Dataset" if frame_dim_label is None else frame_dim_label
                view_mode = "multiple" if view_mode is None else view_mode
                compare_dp_mode = "selected" if compare_dp_mode is None else compare_dp_mode
            elif not title:
                title = dataset_name(first)
            # the dataset's calibration, unless the caller gives one; an acquisition reports an unknown axis as None
            axes = list(zip(first.sampling, first.units))
            if sampling is None and units is None and any(spacing is not None and unit is not None for spacing, unit in axes):
                sampling = tuple(1.0 if spacing is None or unit is None else float(spacing) for spacing, unit in axes)
                units = ["pixels" if spacing is None or unit is None else str(unit) for spacing, unit in axes]
        if bounded:
            if series:
                if scan_region is not None:
                    raise ValueError("Select the same source regions before multi-source comparison.")
                data = AcquisitionSeries(list(data))
            else:
                data = AcquisitionView(data, scan_region)
            precompute_virtual_images = False if precompute_virtual_images is None else precompute_virtual_images
        elif listed:
            data = [core_adapter.as_array(item) if core_adapter.is_dataset(item) else item for item in data]
            # host arrays stack on the host, so torch 2.2 never meets a uint16 array
            data = np.stack(data) if all(isinstance(item, np.ndarray) for item in data) else torch.stack([torch.as_tensor(item) for item in data])
        elif core_adapter.is_dataset(data, ndim=4):
            data = core_adapter.as_array(data)
        if scan_region is not None and not bounded:
            # a dense scan is cut to the region before it moves to its device, as a GPU acquisition is
            row_start, row_stop, col_start, col_stop = scan_region
            if listed or len(data.shape) != 4 or not (0 <= row_start < row_stop <= data.shape[0]
                                                       and 0 <= col_start < col_stop <= data.shape[1]):
                raise ValueError(f"scan_region {tuple(scan_region)} must lie inside the scan of one 4D dataset of shape "
                                 f"{tuple(data.shape)}.")
            data = data[row_start:row_stop, col_start:col_stop]
        precompute_virtual_images = True if precompute_virtual_images is None else bool(precompute_virtual_images)
        view_mode = self._normalise_view_mode("single" if view_mode is None else view_mode)
        compare_dp_mode = self._normalise_compare_dp_mode("average" if compare_dp_mode is None else compare_dp_mode)
        compare_group_mode = self._normalise_compare_group_mode(compare_group_mode)
        if h5_urls:
            urls = list(dict.fromkeys(str(url) for url in h5_urls))
            data = None
            data_device = torch.device("cpu")
            shape = (len(urls), *scan_shape, *detector_shape) if len(urls) > 1 else (*scan_shape, *detector_shape)
        else:
            if type(data).__module__.split(".")[0] == "cupy":
                data = torch.from_dlpack(data)  # zero-copy: the array stays on its GPU
            if isinstance(data, (AcquisitionView, AcquisitionSeries)):
                data_device = data.device
            elif isinstance(data, torch.Tensor):
                data_device = torch.device(resolve_device(device, data))
                data = data.to(data_device)
            else:
                data_device = torch.device(resolve_device(device, data))
                data = np.asarray(data)
                # a copy on a GPU, or a wider copy on the host (uint16 on torch 2.2), is refused rather than reduced
                needed = data.size * host_dtype_for_torch(data.dtype).itemsize
                limit = reader.ceiling(data_device)
                if (data_device.type != "cpu" or host_dtype_for_torch(data.dtype) != data.dtype) and needed > limit:
                    raise MemoryError(
                        f"The {tuple(data.shape)} {data.dtype} scan needs {needed / 1e9:.1f} GB on {data_device}, above the "
                        f"{limit / 1e9:.1f} GB ceiling ({reader.MEMORY_FRACTION:.0%} of the memory available there). Nothing "
                        "was binned or cropped. Free memory, pass scan_region= or another device=, or open the file on a GPU "
                        f"with quantem.gpu: {gpu_path_hint() or no_gpu_path()}.")
                data = counts_tensor(data, data_device)
            if isinstance(data, torch.Tensor) and data.ndim == 3:
                if scan_shape is None:
                    side = int(data.shape[0] ** 0.5)
                    if side * side != data.shape[0]:
                        raise ValueError(f"Cannot infer square scan_shape from N={data.shape[0]}. Provide scan_shape explicitly.")
                    scan_shape = (side, side)
                data = data.reshape(*scan_shape, *data.shape[1:])
            shape = tuple(data.shape)
            if len(shape) not in (4, 5):
                raise ValueError(
                    "Show4DSTEM expects a 3D (positions, det_row, det_col) flat scan, a 4D "
                    f"(scan_row, scan_col, det_row, det_col) scan, or a 5D (n_datasets, ...) series; got {len(shape)}D."
                )
        super().__init__(**kwargs)
        self._static_fallback_mime = self._static_fallback_mime_type()
        self._offline_dtype = offline_dtype
        self._device = data_device
        self._data = data
        self.view_mode = view_mode
        self.compare_dp_mode = compare_dp_mode
        self.compare_group_mode = compare_group_mode
        self.panel_width_px, self.compare_cols = int(panel_width_px), int(compare_cols)
        self.compare_grid_width_px, self.compare_max_panels = int(compare_grid_width_px), int(compare_max_panels)

        if sampling is None:
            sampling = (1.0, 1.0, 1.0, 1.0)
        elif isinstance(sampling, (int, float)):
            sampling = (float(sampling),) * 4
        if units is None:
            units = ["pixels"] * 4
        elif isinstance(units, str):
            units = [units] * 4
        self._axis_sampling = tuple(float(spacing) for spacing in sampling)
        self._axis_units = [str(unit) for unit in units]
        self.title = title
        self.show_title = bool(ui["show_title"])
        self.show_controls = bool(ui["show_controls"])
        self.controls_collapsed = bool(ui["controls_collapsed"])
        self.show_stats = bool(ui["show_stats"])
        self.show_scale_bar = bool(ui["show_scale_bar"])
        self.pixel_size = self._axis_sampling[1]  # scan column axis sets the horizontal scale bar
        self.pixel_unit = self._axis_units[1]
        self.k_pixel_size = self._axis_sampling[3] if len(self._axis_sampling) > 3 else 1.0
        self.k_pixel_unit = self._axis_units[3] if len(self._axis_units) > 3 else "pixels"
        self.k_calibrated = self.k_pixel_unit not in ("pixels", "")
        self._ssb_calibration = {
            "voltage_kV": ssb_voltage_kV,
            "semiangle_mrad": ssb_semiangle_mrad,
            "scan_sampling_A": ssb_scan_sampling_A,
            "det_sampling_mrad": ssb_det_sampling_mrad,
        }
        self.show_fft = bool(show_fft)
        self.frame_dim_label = "Frame" if frame_dim_label is None else str(frame_dim_label)

        self.n_frames = shape[0] if len(shape) == 5 else 1
        self.shape_rows, self.shape_cols = int(shape[-4]), int(shape[-3])
        self.det_rows, self.det_cols = int(shape[-2]), int(shape[-1])
        self.pos_row = self.shape_rows // 2
        self.pos_col = self.shape_cols // 2
        if frame_labels:
            self.frame_labels = [str(label) for label in frame_labels]
        self.vi_roi_center_row = self.shape_rows / 2
        self.vi_roi_center_col = self.shape_cols / 2
        self.vi_roi_center = [self.vi_roi_center_row, self.vi_roi_center_col]
        roi_size = max(3, min(self.shape_rows, self.shape_cols) * DEFAULT_VI_ROI_RATIO)
        self.vi_roi_radius = float(roi_size)
        self.vi_roi_width = float(roi_size * 2)
        self.vi_roi_height = float(roi_size)
        self.center_row = float(self.det_rows / 2) if center is None else float(center[0])
        self.center_col = float(self.det_cols / 2) if center is None else float(center[1])
        self.bf_radius = min(self.det_rows, self.det_cols) * DEFAULT_BF_RATIO if bf_radius is None else float(bf_radius)

        if h5_urls:
            self._h5_urls = json.dumps(urls)
            self.offline = True
            self.ssb_compute_enabled = False
            self.dp_global_min = MIN_LOG_VALUE
            self.dp_global_max = 1.0
            self._set_roi_to_disk()
            self.virtual_image_bytes = np.zeros(self.shape_rows * self.shape_cols, dtype=np.float32).tobytes()
            self.frame_bytes = np.zeros(self.det_rows * self.det_cols, dtype=np.float32).tobytes()
            if self._multiple_view_active():
                visible = min(self.n_frames, self.compare_max_panels)
                self.compare_status = f"Loading {visible}/{self.n_frames} browser WebGPU panels"
            self.compare_page_count = max(1, math.ceil(self.n_frames / self.compare_max_panels))
            return

        # The first pattern bounds the histogram axis; the browser clips percentiles per frame. The
        # bounds are taken in float32 because that is the precision frame_bytes sends the browser.
        first_pattern = np.asarray(self._session(0).frame(0), dtype=np.float32)
        self.dp_global_min = max(float(first_pattern.min()), MIN_LOG_VALUE)
        self.dp_global_max = float(first_pattern.max())
        if center is None and bf_radius is None:
            self.center_row, self.center_col, self.bf_radius = fit_disk(
                self._session(0).mean_dp(), (self.det_rows, self.det_cols), DEFAULT_BF_RATIO
            )
        if precompute_virtual_images and self.n_frames == 1:
            self._precompute_presets()
        self.on_msg(self._handle_frontend_ready_msg)
        self.observe(self._update_frame, names=["pos_row", "pos_col"])
        self.observe(self._on_roi_change, names=["roi_center_col", "roi_center_row", "roi_radius", "roi_radius_inner",
                                                 "roi_active", "roi_mode", "roi_width", "roi_height"])
        self.observe(self._on_roi_center_change, names=["roi_center"])
        self.observe(self._on_vi_source_change, names=["vi_source"])
        self.observe(self._on_calibration_change, names=["center_row", "center_col", "bf_radius"])
        # The first render shows the bright-field image: a point detector paints near black.
        self._set_roi_to_disk()
        self._compute_virtual_image_from_roi()
        self._refresh_compare_virtual_images()
        self._update_frame()
        self.observe(self._on_export_request_change, names=["export_request"])
        self.observe(self._on_ssb_compute_request_change, names=["ssb_compute_request"])
        self.observe(self._on_frame_idx_change, names=["frame_idx"])
        self.observe(self._on_preset_request, names=["_preset_request"])
        self.observe(self._on_compare_config_change, names=["view_mode", "compare_max_panels", "n_frames",
                                                            "compare_group_mode", "compare_page_idx",
                                                            "compare_panel_order", "compare_hidden_panels"])
        self.observe(self._on_compare_dp_mode_change, names=["compare_dp_mode"])
        self.observe(self._on_vi_roi_change, names=["vi_roi_mode", "vi_roi_center_row", "vi_roi_center_col",
                                                    "vi_roi_radius", "vi_roi_width", "vi_roi_height", "vi_roi_reduce"])
        self.observe(self._on_vi_roi_center_change, names=["vi_roi_center"])
        self._schedule_initial_view_sync()
        self._pack_offline(offline)
        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(json.loads(pathlib.Path(state).read_text()), require_envelope=True)
            else:
                state = unwrap_state_payload(state)
            self.load_state_dict(state)
        if self._verbose:
            print(f"Show4DSTEM {'x'.join(str(s) for s in shape)} on {self._device}")

    # ------------------------------------------------------------------
    # Opening a folder of masters
    # ------------------------------------------------------------------

    @classmethod
    def from_folder(
        cls,
        folder,
        *,
        pattern: str = "*_master.h5",
        recursive: bool = True,
        scan_size: int | None = None,
        max_masters: int | None = None,
        min_masters: int | None = None,
        ready_only: bool = True,
        backend: str = "auto",
        device: str = "auto",
        view_mode: str = "multiple",
        columns: int | None = None,
        page_size: int | None = None,
        watch: bool = True,
        watch_interval: float = 2.0,
        verbose: bool = False,
        **viewer_kwargs,
    ) -> Self:
        """Open every ready ``*_master.h5`` in a folder as one live comparison viewer.

        With quantem.gpu and a GPU, each acquisition is loaded with
        ``quantem.gpu.io.load`` into encoded storage at full detector
        resolution, so the whole folder stays resident. Without them every
        master is read densely (``show4dstem.reader``) into one 5D tensor on
        ``device``; that path cannot watch the folder and refuses a folder
        larger than the memory ceiling.
        Masters still being written are skipped (``ready_only``), and when the
        folder mixes geometries the largest group sharing one scan and detector
        shape is shown. The viewer opens once the first master is loaded; the
        others join in the background (``wait_for_folder()`` blocks until they
        have). With ``watch=True`` a master that completes later is appended once
        its headers are unchanged on two polls, every ``watch_interval`` seconds.
        ``columns`` sets the grid width and ``page_size`` the datasets per page.

        Examples
        --------
        >>> viewer = Show4DSTEM.from_folder("/data/session", scan_size=512)  # doctest: +SKIP
        """
        folder_path = pathlib.Path(folder).expanduser().resolve()
        scan_shape = (int(scan_size), int(scan_size)) if scan_size else None
        if not gpu_adapter.accelerator_ready() or str(device).strip().lower() == "cpu":
            # Dense path: every compatible master read whole into one 5D tensor on the device.
            gpu_notice()
            groups: dict[tuple, list] = {}
            for master in reader.find_masters(folder_path, pattern=pattern, recursive=recursive):
                layout = reader.master_layout(master)  # None while a linked data file is missing
                if layout is not None and (scan_shape is None or layout[0] == scan_shape[0] * scan_shape[1]):
                    groups.setdefault(layout[:2], []).append(master)
            if not groups:
                raise ValueError(f"No complete {pattern!r} files found in {folder_path}.")
            masters = max(groups.values(), key=len)[:max_masters]
            if min_masters is not None and len(masters) < min_masters:
                raise ValueError(f"Show4DSTEM.from_folder requires at least {min_masters} compatible master(s), but found {len(masters)}.")
            if watch:
                hint = gpu_path_hint()
                live = f"For live folders: {hint}" if hint else f"{no_gpu_path()}."
                print_once(f"quantem.widget: watching a folder for new masters needs quantem.gpu; showing the "
                           f"{len(masters)} present now. {live}")
            target = torch.device(resolve_device(device))
            limit = reader.ceiling(target)
            series_t = None
            for index, master in enumerate(masters):
                values_t = reader.read_dense(master, target, scan_shape=scan_shape, verbose=verbose)
                if series_t is None:
                    needed = len(masters) * values_t.numel() * values_t.element_size()
                    if needed > limit:
                        hint = gpu_path_hint()
                        advice = f"keep them encoded on a GPU: {hint}" if hint else "use a machine with more memory."
                        raise MemoryError(
                            f"{len(masters)} masters need {needed / 1e9:.1f} GB as one dense series, above the "
                            f"{limit / 1e9:.1f} GB ceiling on {target}. Nothing was binned or cropped; pass "
                            f"max_masters= or {advice}")
                    series_t = torch.empty((len(masters), *values_t.shape), dtype=values_t.dtype, device=target)
                else:
                    # A later master can hold larger counts or fractions: the series widens to the narrowest
                    # dtype holding both exactly (NumPy's promotion rules; torch's refuses uint16).
                    held = [torch.empty(0, dtype=tensor_t.dtype).numpy().dtype for tensor_t in (series_t, values_t)]
                    wide = tensor_dtype(np.promote_types(*held))
                    if wide != series_t.dtype:
                        series_t = reader.promote(series_t, wide, index, limit, master)
                series_t[index] = values_t.to(series_t.dtype)
            return cls(series_t, frame_labels=[acquisition_name(master) for master in masters], frame_dim_label="Dataset",
                       view_mode=view_mode, compare_cols=3 if columns is None else int(columns),
                       compare_max_panels=12 if page_size is None else int(page_size), compare_dp_mode="selected",
                       verbose=verbose, **viewer_kwargs)
        masters = gpu_adapter.discover_masters(folder_path, pattern=pattern, recursive=recursive, scan_shape=scan_shape)
        if ready_only:
            ready = [master for master in masters if gpu_adapter.inspect_master(master).ready]
            if verbose and len(ready) < len(masters):
                names = ", ".join(acquisition_name(master) for master in masters if master not in ready)
                warnings.warn(f"Show4DSTEM.from_folder skipped incomplete master files: {names}.",
                              RuntimeWarning, stacklevel=2)
            masters = ready
        if not masters:
            raise ValueError(
                f"No {'ready ' if ready_only else ''}{pattern!r} files found in {folder_path}. Wait for linked "
                "data files to finish writing, or pass ready_only=False if you know the masters are complete."
            )
        if len(masters) > 1:
            # A comparison needs one geometry; a folder that also holds test scans
            # of another size would otherwise fail to open at all.
            groups: dict[tuple, list] = {}
            for master in masters:
                report = gpu_adapter.inspect_master(master)
                groups.setdefault((report.scan_shape, report.detector_shape, report.actual_frames), []).append(master)
            if len(groups) > 1:
                key, compatible = max(groups.items(), key=lambda item: len(item[1]))
                if verbose:
                    warnings.warn(
                        "Show4DSTEM.from_folder found mixed 4D-STEM shapes in the folder; using the largest "
                        f"compatible group ({len(compatible)}/{len(masters)}) with scan_shape={key[0]}, "
                        f"detector_shape={key[1]}. Skipped {len(masters) - len(compatible)} master file(s). "
                        "Use scan_size= or a narrower pattern= to select a different group.",
                        RuntimeWarning, stacklevel=2)
                masters = compatible
        if min_masters is not None and len(masters) < min_masters:
            raise ValueError(f"Show4DSTEM.from_folder requires at least {min_masters} compatible master(s), but found {len(masters)}.")
        if max_masters is not None:
            masters = masters[:max_masters]
        expected = master_contract(masters[0])

        def validate_master(master) -> None:
            """Reject a watched master whose geometry cannot join the comparison."""
            contract = master_contract(master)
            # dtype is not compared: io.load narrows uint32 counts that fit into uint16.
            mismatches = [name for name in ("scan_shape", "detector_shape", "n_frames") if contract[name] != expected[name]]
            if mismatches:
                observed = ", ".join(f"{name}={contract[name]!r}" for name in mismatches)
                wanted = ", ".join(f"{name}={expected[name]!r}" for name in mismatches)
                raise ValueError(
                    f"Incompatible 4D-STEM master {acquisition_name(master)!r}: {observed}; expected {wanted}. "
                    "Use scan_size= or a narrower pattern= for a uniform folder."
                )

        target = None if str(device).strip().lower() == "auto" else resolve_device(device)

        def load_master(master):
            """Load one master into encoded GPU storage on the requested device (also used by the watcher)."""
            return gpu_adapter.load_acquisition(master, target, backend=backend)

        viewer = cls(
            [load_master(masters[0])],
            view_mode=view_mode,
            compare_cols=3 if columns is None else int(columns),
            compare_max_panels=12 if page_size is None else int(page_size),
            verbose=verbose,
            **viewer_kwargs,
        )
        viewer._folder_source = {
            "folder": folder_path, "pattern": str(pattern), "recursive": bool(recursive),
            "scan_shape": scan_shape, "load_master": load_master, "validate_master": validate_master,
        }
        viewer._folder_known_masters = {viewer._master_key(master) for master in masters}
        # from_folder loaded these acquisitions itself, so free() closes them.
        viewer._owned_acquisitions = [frame.source for frame in viewer._data.frames]
        set_folder_watch_status(viewer, "hidden", "")
        viewer._fill_folder(masters[1:])
        if watch:
            viewer.watch_folder(interval=watch_interval)
        return viewer

    # ------------------------------------------------------------------
    # Public API used by the tutorials, the CLI and quantem.live
    # ------------------------------------------------------------------

    def export_html(
        self,
        path: str | pathlib.Path | None = None,
        *,
        title: str | None = None,
        dtype: str = "uint8",
        det_bin: int = 1,
        scan_bin: int = 1,
        export_kind: str = "interactive",
        dataset_scope: str = "unhidden",
    ) -> pathlib.Path:
        """Write a standalone HTML viewer.

        ``export_kind="interactive"`` packs the counts (``dtype`` ``uint8`` or
        ``uint16``, mean-binned by ``det_bin`` on the detector and ``scan_bin``
        on the scan) into a page whose detector ROI runs on browser WebGPU.
        Packed counts above what one page can load (2 GiB) are refused before
        any frame is read; the error names the least binning that fits.
        ``export_kind="report"`` writes PNG virtual-image pages for the datasets
        in ``dataset_scope`` (``unhidden``, ``current_page``, ``starred`` or
        ``all``) with no raw 4D payload.
        """
        if self._data is None and not self._h5_urls:
            raise ValueError("Cannot export HTML after free(); rebuild the widget first.")
        dtype, det_bin, scan_bin = html_export.export_options(self, dtype, det_bin, scan_bin)
        kind = html_export.export_kind(export_kind)
        out = pathlib.Path(path) if path is not None else html_export.default_export_path(self, dtype, det_bin, scan_bin, kind)
        self._write_html_export(out, title=title, dtype=dtype, det_bin=det_bin, scan_bin=scan_bin,
                                export_kind=kind, dataset_scope=dataset_scope)
        if kind == "report":
            how = "self-contained HTML - double-click to open, no server needed"
        else:
            # The page fetches its packed data over HTTP (file:// blocks it), so a
            # double-click launcher serves the folder and opens the page.
            from quantem.widget.command_launcher import write_command_launcher

            write_command_launcher(out.parent, "Show4DSTEM", viewer_html=out.name)
            how = (f"WebGPU folder - reads its data file over HTTP; double-click Show4DSTEM.command in "
                   f"{out.parent.name}/ to open (a bare double-click of the HTML will not load the data)")
        size_mb = out.stat().st_size / (1024 * 1024)
        self.export_status = f"Exported {out.name} ({size_mb:.1f} MB, {html_export.mode_label(dtype, det_bin, scan_bin, kind)}) - {how}"
        return out

    def _html_export_options(self, payload: dict, mode: str) -> dict:
        """The ``export_html`` keywords of a toolbar request, validated as a Python call is (``HtmlExportMixin`` hook)."""
        dtype, det_bin, scan_bin = html_export.export_options(
            self, payload.get("dtype", "uint8"), payload.get("det_bin", 1), payload.get("scan_bin", 1))
        return {"dtype": dtype, "det_bin": det_bin, "scan_bin": scan_bin,
                "export_kind": html_export.export_kind(payload.get("export_kind", "interactive")),
                "dataset_scope": str(payload.get("dataset_scope", "unhidden"))}

    def _default_html_export_path(self, *, dtype: str, det_bin: int, scan_bin: int, export_kind: str,
                                  dataset_scope: str = "unhidden") -> pathlib.Path:
        """File in the kernel cwd that a toolbar export writes, named after the packing (``HtmlExportMixin`` hook)."""
        return html_export.default_export_path(self, dtype, det_bin, scan_bin, export_kind)

    def _export_mode_label(self, *, dtype: str, det_bin: int, scan_bin: int, export_kind: str,
                           dataset_scope: str = "unhidden") -> str:
        """Packing summary for the export status line (``HtmlExportMixin`` hook)."""
        return html_export.mode_label(dtype, det_bin, scan_bin, export_kind)

    def _write_html_export(self, path, *, title: str | None = None, dtype: str, det_bin: int, scan_bin: int,
                           export_kind: str, dataset_scope: str = "unhidden") -> pathlib.Path:
        """Write a report or an interactive page.

        Replaces the mixin's single export-clone path because Show4DSTEM has two
        export kinds, and a report needs no clone of the counts at all.
        """
        if export_kind == "report":
            return html_export.write_report_html(self, path, dtype=dtype, det_bin=det_bin, scan_bin=scan_bin,
                                                 dataset_scope=dataset_scope, title=title)
        return html_export.write_interactive_html(self, path, dtype=dtype, det_bin=det_bin, scan_bin=scan_bin, title=title)

    def compute_ssb(self, *, set_source: bool = True, verbose: bool = False, **kwargs) -> np.ndarray:
        """Reconstruct the SSB phase of the current dataset and show it as the image.

        ``kwargs`` go to the solver: ``n_trials``, ``refine``, ``seed``,
        ``aberrations``, ``rotation_angle_deg``, ``bf_radius``,
        ``bf_intensity_threshold``, ``lock_c10``, ``lock_c12`` and the
        calibration overrides ``voltage_kV``, ``semiangle_mrad``,
        ``scan_sampling_A``, ``det_sampling_mrad``. The aligned DPC maps from
        the same data are attached beside the phase.
        """
        if self.ssb_compute_busy:
            raise RuntimeError("Compute SSB is already running.")
        self.ssb_compute_busy = True
        self.ssb_compute_status = "Computing SSB..."
        self.ssb_compute_calibration_json = ""
        self.ssb_compute_calibration_filename = ""
        try:
            phase, dpc_row, dpc_col = self._ssb_phase(verbose=verbose, **kwargs)
            self._set_vi_product_map("DPC_row", dpc_row)
            self._set_vi_product_map("DPC_col", dpc_col)
            self._set_vi_product_map("SSB", phase)
            if set_source:
                self.vi_source = "SSB"
            self.ssb_compute_status = f"SSB ready ({phase.shape[0]}x{phase.shape[1]})"
            return phase
        except (ValueError, RuntimeError, MemoryError) as exc:
            self.ssb_compute_status = f"SSB failed: {exc}"
            raise
        finally:
            self.ssb_compute_busy = False

    def apply_preset(self, name: str) -> Self:
        """Place the detector ROI on a named preset: ``bf``, ``abf``, ``adf`` or ``haadf``."""
        preset = str(name).strip().lower()
        if preset not in PRESET_RADII:
            raise ValueError(f"Unknown preset {name!r}. Choices: 'bf', 'abf', 'adf', 'haadf'.")
        bf_radius = self.bf_radius
        # Every ROI trait commits before one recompute; an intermediate state (mode
        # switched, radii stale) would paint a wrong image first.
        self._suppress_roi_recompute = True
        with self.hold_trait_notifications():
            self.vi_source = "roi"
            self.roi_active = True
            self.roi_mode = "circle" if preset == "bf" else "annular"
            self.roi_center_row = float(self.center_row)
            self.roi_center_col = float(self.center_col)
            if preset == "bf":
                self.roi_radius = float(max(1.0, bf_radius))
            elif preset == "abf":
                self.roi_radius_inner = float(max(0.5, bf_radius * 0.5))
                self.roi_radius = float(max(1.0, bf_radius))
            elif preset == "adf":
                self.roi_radius_inner = float(max(1.0, bf_radius))
                self.roi_radius = float(max(bf_radius + 1.0, bf_radius * 2.0))
            else:
                self.roi_radius_inner = float(max(1.0, bf_radius * 2.0))
                self.roi_radius = float(max(bf_radius * 2.0 + 1.0, bf_radius * 4.0))
        self._suppress_roi_recompute = False
        if not self._multiple_view_active():
            self._compute_virtual_image_from_roi()
        self._refresh_compare_virtual_images()
        return self

    def state_dict(self) -> dict:
        """The display state (cursor, ROI, colormaps, layout); no pixel data."""
        keys = (
            "title", "show_title", "pos_row", "pos_col", "pixel_size", "pixel_unit", "k_pixel_size", "k_pixel_unit",
            "k_calibrated", "center_row", "center_col", "bf_radius", "roi_active", "roi_mode", "roi_center_row",
            "roi_center_col", "roi_radius", "roi_radius_inner", "roi_width", "roi_height", "vi_roi_mode",
            "vi_roi_center_row", "vi_roi_center_col", "vi_roi_radius", "vi_roi_width", "vi_roi_height",
            "vi_roi_reduce", "vi_source", "dp_colormap", "vi_colormap", "fft_colormap", "dp_scale_mode",
            "vi_scale_mode", "fft_scale_mode", "dp_vmin_pct", "dp_vmax_pct", "vi_vmin_pct", "vi_vmax_pct",
            "fft_vmin_pct", "fft_vmax_pct", "dp_vmin", "dp_vmax", "vi_vmin", "vi_vmax", "fft_auto", "show_fft",
            "fft_window", "show_controls", "controls_collapsed", "show_stats", "show_scale_bar", "panel_width_px",
            "dp_show_colorbar", "vi_auto_contrast", "vi_smooth", "view_mode", "compare_cols",
            "compare_grid_width_px", "compare_max_panels", "compare_group_mode", "compare_page_idx",
            "compare_dp_mode", "profile_width", "frame_idx", "frame_dim_label", "frame_loop", "frame_fps",
            "frame_reverse", "frame_boomerang",
        )
        state = {key: getattr(self, key) for key in keys}
        for key in ("compare_panel_order", "compare_hidden_panels", "compare_starred_panels", "profile_line", "frame_labels"):
            state[key] = list(getattr(self, key))
        return state

    def load_state_dict(self, state) -> None:
        """Restore a ``state_dict``; positions and the dataset index are clamped to this data."""
        allowed = set(self.state_dict())
        normalisers = {
            "view_mode": self._normalise_view_mode,
            "compare_dp_mode": self._normalise_compare_dp_mode,
            "compare_group_mode": self._normalise_compare_group_mode,
            "vi_source": self._normalise_vi_source,
        }
        for key, value in state.items():
            if key in {"pos_row", "pos_col", "frame_idx"} or key not in allowed:
                continue
            if key in normalisers:
                value = normalisers[key](value)
            if key == "vi_source" and value != "roi" and value not in self._vi_product_maps:
                value = "roi"
            setattr(self, key, value)
        if state.get("frame_idx") is not None:
            self.frame_idx = int(max(0, min(int(state["frame_idx"]), self.n_frames - 1)))
        for key in ("pos_row", "pos_col"):
            if state.get(key) is not None:
                setattr(self, key, int(state[key]))
        self._refresh_compare_virtual_images()
        self._update_frame()

    def free(self) -> None:
        """Release the data held by this widget so the stack leaves GPU memory.

        Detector sessions cache views of the data, so they close first; the
        allocator caches of every device the data touched are flushed; the
        acquisitions ``from_folder`` loaded are closed. Call before loading a
        new dataset into the same kernel. The packed browser copy of the counts
        (up to the 2 GB offline budget) and the last export go too.
        """
        self._release_owned_acquisitions()
        data = self._data
        nbytes = data.nbytes if isinstance(data, torch.Tensor) else 0
        self._close_sessions()
        self._data = None
        self._preset_images = {}
        self.offline = False
        self._offline_stack = b""
        self.export_payload = b""
        self._clear_compare_virtual_images("Show4DSTEM data was freed. Re-run the load/display cell to restore the multiple grid.")
        if self._device.type == "cuda":
            with torch.cuda.device(self._device):
                torch.cuda.empty_cache()
        elif self._device.type == "mps":
            torch.mps.empty_cache()
        if nbytes:
            print(f"freed {nbytes / 1e9:.2f} GB ({self._device})")

    def close(self) -> None:
        """Stop background work, release sessions and close the widget comm."""
        self._release_owned_acquisitions()
        self._close_sessions()
        super().close()

    # ------------------------------------------------------------------
    # Watching a folder
    # ------------------------------------------------------------------

    def poll_folder(self) -> list[int]:
        """Append newly completed folder masters to the comparison.

        Only ``from_folder`` widgets watch a folder. A new master must report the
        same complete header signature on two consecutive polls before it is
        loaded, so a file still being written is never read. Returns the dataset
        indices this poll appended.
        """
        if self._folder_source is None:
            raise RuntimeError("poll_folder() is available only on Show4DSTEM.from_folder(...) widgets.")
        source = self._folder_source
        watch_active = self._folder_watch_is_alive()
        waiting: list[str] = []
        errors: list[str] = []
        with self._folder_poll_lock:
            if watch_active:
                set_folder_watch_status(self, "updating", "Checking the folder for new 4D-STEM data.")
            # A file listing, not quantem.gpu's discover, which raises on an empty match: a watched
            # folder with no master yet is an ordinary state. Incomplete candidates stay visible to
            # the readiness protocol so the badge can say why a matching master has not appeared yet.
            masters = reader.find_masters(source["folder"], pattern=source["pattern"], recursive=source["recursive"])
            known = self._folder_known_masters
            discovered = {self._master_key(master) for master in masters}
            probation = {key: signature for key, signature in self._folder_ready_probation.items()
                         if key in discovered and key not in known}
            accepted = []
            for master in masters:
                key = self._master_key(master)
                if key in known:
                    continue
                try:
                    report = gpu_adapter.inspect_master(master, scan_shape=source["scan_shape"])
                except (OSError, ValueError, KeyError) as exc:
                    probation.pop(key, None)
                    errors.append(self._folder_issue_detail(
                        master, f"readiness inspection failed ({type(exc).__name__}: {exc})",
                        "Check file permissions and HDF5 integrity, then retry."))
                    continue
                if not report.ready:
                    probation.pop(key, None)
                    detail = self._folder_issue_detail(master, report.reason, report.action)
                    (waiting if readiness_is_waiting(report) else errors).append(detail)
                    continue
                signature = json.dumps(report.source_signature, sort_keys=True, separators=(",", ":"), default=str)
                if probation.get(key) != signature:
                    probation[key] = signature
                    waiting.append(self._folder_issue_detail(
                        master, "complete headers found; waiting for one unchanged follow-up readiness poll",
                        "Keep the master and linked detector files in place."))
                    continue
                try:
                    source["validate_master"](master)
                except ValueError as exc:
                    # The confirmed signature stays so a corrected file retries at once.
                    errors.append(self._folder_issue_detail(
                        master, f"incompatible master ({exc})",
                        "Use a matching scan and detector shape, or move this file out of the watched folder."))
                    continue
                probation.pop(key, None)
                accepted.append(master)
            self._folder_ready_probation = probation
            added, load_errors = self._append_folder_masters(accepted)
            errors.extend(load_errors)
            if not masters and not known:
                waiting.append("No matching 4D-STEM master has arrived yet. Keep the watcher running or check the folder and filename pattern.")
            if self._folder_watch_is_alive():
                if errors:
                    set_folder_watch_status(self, "error", errors[0])
                elif waiting:
                    set_folder_watch_status(self, "waiting", waiting[0])
                else:
                    set_folder_watch_status(self, "watching", "")
            return added

    def wait_for_folder(self, timeout: float | None = None) -> Self:
        """Block until the masters found when ``from_folder`` opened are loaded."""
        if self._folder_fill_thread is not None:
            self._folder_fill_thread.join(timeout)
        return self

    def watch_folder(self, *, interval: float = 2.0) -> Self:
        """Poll the attached folder in the background and append ready masters."""
        if self._folder_source is None:
            raise RuntimeError("watch_folder() is available only on Show4DSTEM.from_folder(...) widgets.")
        interval = float(interval)
        if not np.isfinite(interval) or interval <= 0:
            raise ValueError(f"watch interval must be a finite value > 0 seconds, got {interval!r}")
        self.stop_folder_watch()
        stop = threading.Event()
        self._folder_watch_stop = stop
        self._folder_watch_started = True

        def worker() -> None:
            """Poll every ``interval`` seconds until stopped, then say on the badge why the watcher ended."""
            try:
                while not stop.wait(interval):
                    try:
                        self.poll_folder()
                    except (OSError, ValueError, RuntimeError) as exc:
                        # A folder that briefly vanishes (network share, renamed
                        # session) must not kill the viewer: report and retry.
                        set_folder_watch_status(self, "error", self._compact_folder_watch_detail(
                            f"{type(exc).__name__}: {exc}. The watcher is still alive and will retry; "
                            "check the folder, storage, and file permissions."))
            finally:
                if self._folder_watch_stop is stop:
                    self._folder_watch_stop = None
                if self._folder_watch_thread is threading.current_thread():
                    self._folder_watch_thread = None
                if stop.is_set():
                    set_folder_watch_status(self, "stopped", "Folder watcher stopped.")
                else:
                    set_folder_watch_status(self, "error", "Folder watch worker stopped unexpectedly. Call watch_folder() to restart it.")

        thread = threading.Thread(target=worker, name="Show4DSTEM-folder-watch", daemon=True)
        self._folder_watch_thread = thread
        thread.start()
        set_folder_watch_status(self, "watching", "")
        return self

    def stop_folder_watch(self) -> None:
        """Stop the background folder watcher, if one was started."""
        stop, thread = self._folder_watch_stop, self._folder_watch_thread
        if stop is not None:
            stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        if self._folder_watch_stop is stop:
            self._folder_watch_stop = None
        if self._folder_watch_thread is thread:
            self._folder_watch_thread = None
        if self._folder_watch_started:
            set_folder_watch_status(self, "stopped", "Folder watcher stopped.")

    @staticmethod
    def _master_key(master) -> str:
        """Absolute resolved path of a master, so a relative path or a symlink is not loaded twice."""
        return str(pathlib.Path(master).expanduser().resolve())

    def _folder_watch_is_alive(self) -> bool:
        """Whether the watcher thread runs; only a running watcher owns the folder badge."""
        return self._folder_watch_thread is not None and self._folder_watch_thread.is_alive()

    def _compact_folder_watch_detail(self, detail: str, *, limit: int = 480) -> str:
        """Bounded status text without the watched root path or any host filesystem layout."""
        text = " ".join(str(detail).split())
        if self._folder_source is not None:
            root = str(pathlib.Path(self._folder_source["folder"]))
            variants = {root, os.path.realpath(root)}
            variants.update(path[len("/private"):] for path in tuple(variants) if path.startswith("/private/"))
            for path in sorted(variants, key=len, reverse=True):
                text = text.replace(f"{path}{os.sep}", "")
                text = text.replace(path, pathlib.Path(path).name or "watched folder")
        tokens = []
        for token in text.split(" "):
            leading = token[: len(token) - len(token.lstrip("([{<"))]
            candidate = token[len(leading):]
            trailing = candidate[len(candidate.rstrip(".,;:)]}>")):]
            if trailing:
                candidate = candidate[: len(candidate) - len(trailing)]
            if candidate.startswith(os.sep) and os.sep in candidate[1:]:
                candidate = pathlib.Path(candidate).name or "source file"
            tokens.append(f"{leading}{candidate}{trailing}")
        text = " ".join(tokens)
        if len(text) > limit:
            text = f"{text[: max(0, limit - 1)].rstrip()}…"
        return text

    def _folder_issue_detail(self, master, reason: str, action: str = "") -> str:
        """Badge text ``"<file>: <reason>. <action>"`` for one master, compacted so no host path reaches the UI."""
        name = pathlib.Path(master).name or str(master)
        reason = str(reason).strip().rstrip(".")
        detail = f"{name}: {reason}." if reason else f"{name}: not ready."
        return self._compact_folder_watch_detail(f"{detail} {str(action).strip()}".strip())

    def _append_folder_masters(self, masters) -> tuple[list[int], list[str]]:
        """Load ready masters into the series and publish them in one update.

        Returns the appended dataset indices and, per master that failed to load,
        the corrective badge text; a failed master is forgotten so the next poll
        retries it. Callers hold the folder lock.
        """
        old_order = list(self.compare_panel_order)
        custom_order = sorted(old_order) == list(range(self.n_frames))
        labels = list(self.frame_labels)
        added: list[int] = []
        errors: list[str] = []
        for master in masters:
            try:
                acquisition = self._folder_source["load_master"](master)
            except (OSError, ValueError, RuntimeError, MemoryError) as exc:
                self._folder_known_masters.discard(self._master_key(master))
                errors.append(self._folder_issue_detail(
                    master, f"could not be loaded ({type(exc).__name__}: {exc})",
                    "Check that the master and its linked detector files are complete, or move it out of the watched folder."))
                continue
            self._data.append(acquisition)
            self._owned_acquisitions.append(acquisition)
            self._folder_known_masters.add(self._master_key(master))
            added.append(len(self._data.frames) - 1)
            labels.append(acquisition_name(master))
        if added:
            # n_frames refreshes the comparison grid once the held changes land.
            with self.hold_trait_notifications():
                self.n_frames = len(self._data.frames)
                self.frame_labels = labels
                if custom_order:
                    self.compare_panel_order = [*old_order, *added]
        return added, errors

    def _fill_folder(self, masters) -> None:
        """Load the rest of the opening folder in the background, one master at a time."""
        stop = self._folder_fill_stop

        def worker() -> None:
            """Append the opening masters one at a time, stopping early once the widget is freed."""
            errors: list[str] = []
            for master in masters:
                if stop.is_set():
                    return
                with self._folder_poll_lock:
                    errors.extend(self._append_folder_masters([master])[1])
            if errors:
                set_folder_watch_status(self, "error", errors[0])

        self._folder_fill_thread = threading.Thread(target=worker, name="Show4DSTEM-folder-fill", daemon=True)
        self._folder_fill_thread.start()

    def _release_owned_acquisitions(self) -> None:
        """Stop folder work and close the acquisitions this widget loaded itself (a path or ``from_folder``)."""
        self.stop_folder_watch()
        if self._folder_source is not None:
            self._folder_fill_stop.set()
            self.wait_for_folder()
        for acquisition in self._owned_acquisitions:
            acquisition.close()
        self._owned_acquisitions = []

    # ------------------------------------------------------------------
    # Detector sessions and reductions
    # ------------------------------------------------------------------

    def _frame_source(self, frame_idx: int):
        """The 4D source of one dataset: a bounded view or a tensor."""
        if isinstance(self._data, AcquisitionSeries):
            return self._data.frames[frame_idx]
        if isinstance(self._data, torch.Tensor) and self._data.ndim == 5:
            return self._data[frame_idx]
        return self._data

    def _session(self, frame_idx: int):
        """The detector session of one dataset, created on first use.

        A dense tensor reduces in the widget's own torch code on its device; an
        encoded acquisition in quantem.gpu's session over its compressed counts.
        """
        if frame_idx not in self._sessions:
            source = self._frame_source(frame_idx)
            self._sessions[frame_idx] = (DenseSession(source) if isinstance(source, torch.Tensor)
                                         else gpu_adapter.prepare_session(source))
        return self._sessions[frame_idx]

    def _close_sessions(self) -> None:
        """Close every detector session: sessions hold views of the data, so the memory is released only after this."""
        for session in self._sessions.values():
            session.close()
        self._sessions = {}

    @property
    def _compute(self):
        """The detector session of the current dataset."""
        return self._session(int(self.frame_idx))

    def pattern(self, row: int, col: int) -> np.ndarray:
        """The diffraction pattern at scan position ``(row, col)`` of the current dataset as float32."""
        if self._data is None:
            return np.zeros((self.det_rows, self.det_cols), dtype=np.float32)
        # the flat index would wrap a column past the edge into the next row's pattern
        if not (0 <= row < self.shape_rows and 0 <= col < self.shape_cols):
            raise IndexError(f"scan position ({row}, {col}) is outside the {self.shape_rows} x {self.shape_cols} scan.")
        return np.asarray(self._compute.frame(row * self.shape_cols + col), dtype=np.float32)

    def virtual_image(self) -> np.ndarray:
        """The image panel as float32: the product map in view, else the detector ROI sum."""
        product = self._vi_product_maps.get(self.vi_source)
        if product is not None:
            frame = 0 if product.shape[0] == 1 else max(0, min(int(self.frame_idx), product.shape[0] - 1))
            return np.array(product[frame], dtype=np.float32)
        if len(self.virtual_image_bytes) != self.shape_rows * self.shape_cols * 4:
            return np.zeros((self.shape_rows, self.shape_cols), dtype=np.float32)
        return np.frombuffer(self.virtual_image_bytes, dtype=np.float32).reshape(self.shape_rows, self.shape_cols).copy()

    def vi_roi_pattern(self) -> np.ndarray:
        """The pattern reduced over the scan ROI, as the diffraction panel shows it."""
        self._compute_vi_roi_dp()
        if len(self.vi_roi_dp_bytes) != self.det_rows * self.det_cols * 4:
            return self.pattern(self.pos_row, self.pos_col)
        return np.frombuffer(self.vi_roi_dp_bytes, dtype=np.float32).reshape(self.det_rows, self.det_cols).copy()

    def _frame_array(self, frame_idx: int, det_bin: int) -> np.ndarray:
        """One dataset as a host float32 array, mean-binned on the detector (counts summed exactly).

        Tensors come back in scan-row chunks and bounded views in 32 MiB scan
        windows, so an export never materialises a dense cube on the GPU.
        """
        source = self._frame_source(frame_idx)
        det_rows, det_cols = self.det_rows // det_bin, self.det_cols // det_bin
        out = np.empty((self.shape_rows, self.shape_cols, det_rows, det_cols), np.float32)
        if isinstance(source, AcquisitionView):
            window_cols = max(1, min(self.shape_cols, (32 << 20) // (self.det_rows * self.det_cols * 4)))
            for row in range(self.shape_rows):
                for col in range(0, self.shape_cols, window_cols):
                    stop = min(col + window_cols, self.shape_cols)
                    out[row : row + 1, col:stop] = detector_bin_mean(source.read(scan_region=(row, row + 1, col, stop)), det_bin)
            return out
        chunk_rows = max(1, (600 << 20) // max(1, self.shape_cols * self.det_rows * self.det_cols * 4))
        for row in range(0, self.shape_rows, chunk_rows):
            out[row : row + chunk_rows] = detector_bin_mean(source[row : row + chunk_rows], det_bin)
        return out

    def _current_detector_mask(self) -> np.ndarray:
        """Boolean detector mask of the ROI traits, shared by the single view and the comparison grid."""
        return roi_mask(self.roi_mode, (self.det_rows, self.det_cols), center_row=self.roi_center_row,
                        center_col=self.roi_center_col, radius=self.roi_radius, radius_inner=self.roi_radius_inner,
                        width=self.roi_width, height=self.roi_height)

    def _compute_virtual_image_from_roi(self) -> None:
        """Sum the current dataset over the detector ROI into ``virtual_image_bytes``."""
        if self._data is None or self.vi_source != "roi":
            return
        mask = self._current_detector_mask()
        preset = self._preset_name(mask)
        if preset in self._preset_images:
            self.virtual_image_bytes = self._preset_images[preset]
            return
        image = self._compute.masked_sum(mask)
        self.virtual_image_bytes = np.ascontiguousarray(image, dtype=np.float32).tobytes()

    def _precompute_presets(self) -> None:
        """Cache the four preset images so a preset click paints without a reduction."""
        center = (self.center_row, self.center_col)
        self._preset_images = {
            name: np.ascontiguousarray(
                self._compute.masked_sum(preset_mask(name, (self.det_rows, self.det_cols), center, self.bf_radius)),
                dtype=np.float32).tobytes()
            for name in PRESET_RADII
        }

    def _preset_name(self, mask: np.ndarray) -> str | None:
        """The preset whose detector mask equals ``mask`` pixel for pixel, if any.

        A cached preset image is the masked sum of that exact mask, so it may only
        stand in for an ROI that selects the same detector pixels; a radius within a
        pixel of the preset can still select a different set and a different image.
        """
        center = (self.center_row, self.center_col)
        for name in PRESET_RADII:
            if np.array_equal(mask, preset_mask(name, (self.det_rows, self.det_cols), center, self.bf_radius)):
                return name
        return None

    def _compare_images(self, indices: Sequence[int], mask: np.ndarray) -> list[np.ndarray]:
        """Virtual images of several datasets under one mask, per detector pixel.

        Panels are previews across datasets, so each is divided by the mask area:
        switching from BF to ADF then does not flatten them only because the mask
        covers more pixels. The single-view image stays a plain sum.
        """
        area = max(1.0, float(np.count_nonzero(mask)))
        return [np.ascontiguousarray(self._session(int(idx)).masked_sum(mask), dtype=np.float32) / area for idx in indices]

    def _compute_vi_roi_dp(self) -> None:
        """Reduce the patterns inside the scan ROI (mean, sum or max) into ``vi_roi_dp_bytes``."""
        if self._data is None or self.vi_roi_mode not in ("circle", "square", "rect"):
            return
        indices = scan_indices(self.vi_roi_mode, (self.shape_rows, self.shape_cols), center_row=self.vi_roi_center_row,
                               center_col=self.vi_roi_center_col, radius=self.vi_roi_radius, width=self.vi_roi_width,
                               height=self.vi_roi_height)
        if indices.size == 0:
            self.vi_roi_dp_bytes = b""
            self._clear_all_diffraction()
            return
        pattern = self._compute.reduce_frames(indices, self.vi_roi_reduce)
        self.vi_roi_dp_bytes = np.ascontiguousarray(pattern, dtype=np.float32).tobytes()
        self._publish_all_diffraction(indices)

    @traitlets.validate("pos_row", "pos_col")
    def _clamp_scan_position(self, proposal) -> int:
        """Stop a scan position at the scan edge.

        The pattern is read at the flat index ``row * shape_cols + col``: a
        column past the edge would show the next row's pattern and a row past
        it no pattern at all, so every assignment (Python, the browser, a
        saved state) is clamped to ``[0, shape - 1]``.
        """
        size = self.shape_rows if proposal["trait"].name == "pos_row" else self.shape_cols
        return max(0, min(int(proposal["value"]), size - 1))

    def _update_frame(self, change=None) -> None:
        """Send the diffraction pattern of the cursor (or the compare average) to the frontend."""
        if self._data is None:
            return
        flat = self.pos_row * self.shape_cols + self.pos_col
        if self._multiple_view_active() and self.compare_dp_mode == "average":
            page = self._page_indices() or [int(self.frame_idx)]
            frame = np.mean([np.asarray(self._session(idx).frame(flat), dtype=np.float32) for idx in page], axis=0)
        else:
            frame = np.asarray(self._compute.frame(flat), dtype=np.float32)
        self.frame_bytes = frame.tobytes()
        if self.vi_roi_mode != "off":
            self._compute_vi_roi_dp()
        else:
            self._publish_all_diffraction()

    def _publish_all_diffraction(self, indices=None) -> None:
        """In ``compare_dp_mode="all"`` send every visible dataset's pattern for the one selection."""
        if self._multiple_view_active() and self.compare_dp_mode == "all":
            panels = list(self.compare_panel_indices)
            flat = self.pos_row * self.shape_cols + self.pos_col
            frames = [
                self._session(idx).frame(flat) if indices is None else self._session(idx).reduce_frames(indices, self.vi_roi_reduce)
                for idx in panels
            ]
            if frames:
                with self.hold_sync():
                    self.compare_diffraction_indices = panels
                    self.compare_diffraction_bytes = np.stack([np.asarray(frame, dtype=np.float32) for frame in frames]).tobytes()
                return
        self._clear_all_diffraction()

    def _clear_all_diffraction(self) -> None:
        """Empty the per-dataset patterns of ``compare_dp_mode="all"`` in one frontend message."""
        with self.hold_sync():
            self.compare_diffraction_bytes = b""
            self.compare_diffraction_indices = []

    # ------------------------------------------------------------------
    # Product maps (DPC and SSB phase)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_vi_source(value: str | None) -> str:
        """Canonical image source (``roi``, ``DPC_row``, ``DPC_col``, ``SSB``) from the aliases callers and states send."""
        source = str(value or "roi").strip()
        key = source.lower().replace("-", "_").replace(" ", "_")
        aliases = {"": "roi", "roi": "roi", "virtual": "roi", "virtual_image": "roi", "bf": "roi",
                   "dpc_row": "DPC_row", "dpc_col": "DPC_col", "ssb": "SSB", "ssb_phase": "SSB", "phase": "SSB"}
        return aliases.get(key, source)

    def _set_vi_product_map(self, label: str, value) -> None:
        """Attach or replace one static map (2D, or 3D with one frame per dataset) and resend the stack."""
        label = self._normalise_vi_source(label)
        if label not in PRODUCT_LABELS:
            raise ValueError(f"Unsupported virtual-image product {label!r}. Use one of {PRODUCT_LABELS}.")
        product = np.asarray(value)
        if np.iscomplexobj(product):
            raise ValueError(f"{label} must be a real-valued map. Pass the SSB phase map rather than a complex reconstruction.")
        if product.ndim == 2:
            product = product[None]
        if product.ndim != 3 or product.shape[1:] != (self.shape_rows, self.shape_cols) or product.shape[0] not in (1, self.n_frames):
            raise ValueError(
                f"{label} must have shape (scan_rows, scan_cols) or (n_frames, scan_rows, scan_cols) matching "
                f"{self.n_frames}x{self.shape_rows}x{self.shape_cols}; got {product.shape}."
            )
        maps = dict(self._vi_product_maps)
        maps[label] = np.ascontiguousarray(product, dtype=np.float32)
        frame_count = max(stored.shape[0] for stored in maps.values())
        stacked = [np.broadcast_to(stored, (frame_count, self.shape_rows, self.shape_cols)) for stored in maps.values()]
        self._vi_product_maps = maps
        self.vi_product_labels = list(maps)
        self.vi_product_map_frames = int(frame_count)
        self.vi_product_maps_bytes = np.stack(stacked).tobytes()
        if self.vi_source == label:
            self._refresh_compare_virtual_images()

    def _on_vi_source_change(self, change=None) -> None:
        """React to an image-source switch: fall back to the ROI image when that product map does not exist, then repaint."""
        source = self._normalise_vi_source(self.vi_source)
        if source != "roi" and source not in self._vi_product_maps:
            source = "roi"
        if source != self.vi_source:
            self.vi_source = source  # re-enters this observer with a valid source
            return
        if source == "roi" and not self._multiple_view_active():
            self._compute_virtual_image_from_roi()
        self._refresh_compare_virtual_images()

    # ------------------------------------------------------------------
    # SSB
    # ------------------------------------------------------------------

    def _ssb_source(self):
        """The current dataset in the form quantem.gpu SSB reads without a dense copy."""
        frame = self._frame_source(int(self.frame_idx))
        if not isinstance(frame, AcquisitionView):
            return frame
        if frame.region == (0, frame.source.shape[0], 0, frame.source.shape[1]):
            return frame.source
        return frame.read(scan_region=(0, frame.shape[0], 0, frame.shape[1]))

    def _ssb_scan_sampling_A(self, value) -> tuple[float, float]:
        """Scan sampling ``(row, col)`` in Angstrom: ``value``, else the scan axes' calibration converted from A, nm or pm.

        SSB needs the physical real-space step; a scan calibrated only in pixels
        is refused with the two ways to supply it.
        """
        if value is not None:
            return self._positive_pair(value, "scan_sampling_A")
        factors = {"a": 1.0, "ang": 1.0, "nm": 10.0, "pm": 0.01}
        pair = []
        for axis in (0, 1):
            key = self._unit_key(self._axis_units[axis])
            if key not in factors or self._axis_sampling[axis] <= 0:
                raise ValueError(
                    "Compute SSB needs real-space scan sampling in Angstroms. Pass ssb_scan_sampling_A=(row_A, col_A), "
                    "or construct Show4DSTEM with sampling=(scan_row, scan_col, ..., ...) and units=('A', 'A', ..., ...)."
                )
            pair.append(self._axis_sampling[axis] * factors[key])
        return (pair[0], pair[1])

    def _ssb_det_sampling_mrad(self, value) -> tuple[float, float] | None:
        """Detector sampling ``(row, col)`` in mrad per pixel: ``value``, else the detector axes' calibration, else None.

        None (detector axes without an angular unit) is allowed: the semiangle
        is then required explicitly.
        """
        if value is not None:
            return self._positive_pair(value, "det_sampling_mrad")
        factors = {"mrad": 1.0, "rad": 1000.0, "urad": 0.001}
        keys = [self._unit_key(unit) for unit in self._axis_units[2:4]]
        if len(keys) < 2 or any(key not in factors for key in keys) or any(s <= 0 for s in self._axis_sampling[2:4]):
            return None
        return (self._axis_sampling[2] * factors[keys[0]], self._axis_sampling[3] * factors[keys[1]])

    @staticmethod
    def _unit_key(unit: str) -> str:
        """Unit spelling as a lookup key (``"Å/pixel"`` -> ``"a"``, ``"µrad"`` -> ``"urad"``), so any spelling converts."""
        key = str(unit or "").strip().lower().replace("µ", "u").replace("μ", "u").replace("å", "a")
        key = key.replace("angstroms", "angstrom").replace("angstrom", "a")
        key = key.replace(" per pixel", "").replace("/pixel", "").replace("/px", "")
        return key.replace(" ", "").replace("_", "")

    @staticmethod
    def _positive_pair(value, name: str) -> tuple[float, float]:
        """A positive ``(row, col)`` pair from a scalar or a length-2 sequence; zero or negative sampling has no SSB geometry."""
        pair = (float(value), float(value)) if np.isscalar(value) else tuple(float(component) for component in value)
        if len(pair) != 2 or pair[0] <= 0 or pair[1] <= 0:
            raise ValueError(f"{name} must be a positive scalar or length-2 pair, got {value!r}.")
        return pair

    def _ssb_phase(
        self,
        *,
        voltage_kV: float | None = None,
        semiangle_mrad: float | None = None,
        scan_sampling_A=None,
        det_sampling_mrad=None,
        bf_radius: float | None = None,
        aberrations: dict[str, float] | None = None,
        rotation_angle_deg: float = 0.0,
        bf_intensity_threshold: float = 0.5,
        n_trials: int | None = None,
        refine: bool | None = None,
        lock_aberrations: bool = False,
        lock_c10: bool = False,
        lock_c12: bool = False,
        seed: int = 42,
        verbose: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run quantem.gpu SSB on the current dataset; returns ``(phase, dpc_row, dpc_col)``.

        The DPC centre-of-mass maps are rotated by the solved physical scan
        rotation so they follow the aligned-DPC convention.
        """
        if self.n_frames != 1:
            raise ValueError("Compute SSB runs on a single 4D dataset. For a series, open the target dataset as a 4D widget.")
        if (self.shape_rows, self.shape_cols) not in ((128, 128), (256, 256), (512, 512)):
            raise ValueError(f"Compute SSB supports square 128x128, 256x256, or 512x512 scan grids; got {self.shape_rows}x{self.shape_cols}.")
        calibration = self._ssb_calibration
        scan_sampling = self._ssb_scan_sampling_A(scan_sampling_A if scan_sampling_A is not None else calibration["scan_sampling_A"])
        det_sampling = self._ssb_det_sampling_mrad(det_sampling_mrad if det_sampling_mrad is not None else calibration["det_sampling_mrad"])
        semiangle = semiangle_mrad if semiangle_mrad is not None else calibration["semiangle_mrad"]
        if semiangle is None:
            if det_sampling is None or self.bf_radius <= 0:
                raise ValueError("Compute SSB needs ssb_semiangle_mrad, or calibrated detector sampling in mrad/pixel plus a detected BF radius.")
            semiangle = self.bf_radius * det_sampling[1]
        if semiangle <= 0:
            raise ValueError(f"semiangle_mrad must be positive, got {semiangle}.")
        voltage = voltage_kV if voltage_kV is not None else calibration["voltage_kV"]
        if voltage is None:
            raise ValueError("Compute SSB needs ssb_voltage_kV. Example: Show4DSTEM(data, ssb_voltage_kV=300).")
        trials = int(self.ssb_compute_n_trials if n_trials is None else n_trials)
        if trials < 0:
            raise ValueError(f"n_trials must be >= 0, got {trials}.")
        do_refine = bool(self.ssb_compute_refine if refine is None else refine)
        if aberrations is not None:
            phi12 = math.radians(float(aberrations["phi12_deg"])) if "phi12_deg" in aberrations else float(aberrations.get("phi12", 0.0))
            aberrations = {"C10": float(aberrations.get("C10", aberrations.get("C10_nm", 0.0))),
                           "C12": float(aberrations.get("C12", aberrations.get("C12_nm", 0.0))), "phi12": phi12}
        search_ranges = refine_lock = None
        if lock_c10 or lock_c12:
            base = aberrations or {}
            locked_phi12_deg = math.degrees(float(base["phi12"])) if "phi12" in base else float(self.ssb_compute_phi12_deg)
            # A scalar pins a coefficient in the trial search and refine_lock holds it
            # through Nelder-Mead; locking C12 pins phi12 too (magnitude and angle pair).
            search_ranges = {
                "C10_nm": float(base.get("C10", self.ssb_compute_c10_nm)) if lock_c10 else (-40.0, 40.0),
                "C12_nm": float(base.get("C12", self.ssb_compute_c12_nm)) if lock_c12 else (0.0, 10.0),
                "phi12_deg": locked_phi12_deg if lock_c12 else (-90.0, 90.0),
            }
            refine_lock = (["C10"] if lock_c10 else []) + (["C12", "phi12"] if lock_c12 else [])
        self.ssb_compute_status = "Preparing SSB data..."
        source = self._ssb_source()
        manual = bool(lock_aberrations and aberrations is not None)
        with gpu_adapter.ssb_session(
            source,
            voltage_kV=float(voltage),
            semiangle_mrad=float(semiangle),
            scan_sampling_A=scan_sampling,
            det_sampling=det_sampling,
            scan_shape=(self.shape_rows, self.shape_cols) if source.ndim == 3 else None,
            aberrations=aberrations,
            rotation_angle_deg=float(rotation_angle_deg),
            bf_intensity_threshold=float(bf_intensity_threshold),
            bf_radius=None if bf_radius is None else float(bf_radius),
        ) as ssb:
            num_bf = int(ssb.num_bf)
            self.ssb_compute_bf_pixels = num_bf
            if manual or (trials == 0 and not do_refine):
                self.ssb_compute_status = f"Reconstructing SSB with the given coefficients ({num_bf} BF pixels)..."
                result = ssb.reconstruct()
            else:
                refinement = "nelder-mead" if do_refine else None
                self.ssb_compute_status = f"Fitting SSB aberrations ({trials} trials{', Nelder-Mead' if refinement else ''}, {num_bf} BF pixels)..."
                result = ssb.find_aberrations(trials=trials, refinement=refinement, search_ranges=search_ranges,
                                              refine_lock=refine_lock if refinement else None, seed=int(seed), verbose=verbose)
            physical_rotation = ssb.physical_rotation_deg
        phase = np.ascontiguousarray(torch.as_tensor(result.phase).cpu().numpy(), dtype=np.float32)
        if phase.shape != (self.shape_rows, self.shape_cols):
            raise ValueError(f"SSB result shape {phase.shape} does not match the widget scan shape {self.shape_rows}x{self.shape_cols}.")
        self.ssb_compute_c10_nm = float(result.aberrations["C10"])
        self.ssb_compute_c12_nm = float(result.aberrations["C12"])
        self.ssb_compute_phi12_deg = math.degrees(float(result.aberrations["phi12"]))
        self.ssb_compute_rotation_angle_deg = float(result.rotation_angle_deg)
        self.ssb_compute_calibration_json = json.dumps({
            "schema": "quantem.ssb.calibration.v1",
            "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "source": {"widget": "Show4DSTEM", "title": self.title, "frame_idx": int(self.frame_idx),
                       "scan_shape": [self.shape_rows, self.shape_cols], "detector_shape": [self.det_rows, self.det_cols]},
            "aberrations": {"C10": self.ssb_compute_c10_nm, "C12": self.ssb_compute_c12_nm, "phi12": float(result.aberrations["phi12"])},
            "aberration_units": {"C10": "nm", "C12": "nm", "phi12": "rad"},
            "rotation_angle_deg": float(result.rotation_angle_deg),
            "com_reversed": bool(result.com_reversed),
            "calibration": {"voltage_kV": float(voltage), "semiangle_mrad": float(semiangle),
                            "scan_sampling_A": [float(step) for step in scan_sampling],
                            "det_sampling_mrad": None if det_sampling is None else [float(step) for step in det_sampling],
                            "bf_radius": None if bf_radius is None else float(bf_radius),
                            "bf_intensity_threshold": float(bf_intensity_threshold), "bf_pixels": num_bf},
            "run": {"n_trials": 0 if manual else trials, "refine": bool(do_refine and not manual), "manual_locked": manual,
                    "seed": int(seed), "loss": None if result.loss is None else float(result.loss),
                    "elapsed_s": None if result.elapsed is None else float(result.elapsed)},
        }, indent=2)
        self.ssb_compute_calibration_filename = (
            f"{html_export.slug(self.title)}_{self.shape_rows}x{self.shape_cols}x{self.det_rows}x{self.det_cols}_ssb_calibration.json"
        )
        com_row, com_col = gpu_adapter.center_of_mass(source, scan_shape=(self.shape_rows, self.shape_cols))
        theta = math.radians(physical_rotation)
        dpc_row = np.ascontiguousarray(math.cos(theta) * com_row - math.sin(theta) * com_col, dtype=np.float32)
        dpc_col = np.ascontiguousarray(math.sin(theta) * com_row + math.cos(theta) * com_col, dtype=np.float32)
        return phase, dpc_row, dpc_col

    def _on_ssb_compute_request_change(self, change) -> None:
        """The frontend's Compute SSB button: run the solver on a worker thread."""
        raw = str(change.get("new") or "")
        if not raw:
            return
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"action": raw}
        action = str(payload.get("action", "compute_ssb")).strip().lower()
        self.ssb_compute_request = ""  # consume the trigger before any long work
        if action in {"clear", "reset"}:
            self.ssb_compute_status = ""
            return
        if action not in {"compute_ssb", "ssb", "compute"}:
            self.ssb_compute_status = f"SSB failed: unknown request {action!r}"
            return
        if not self.ssb_compute_enabled:
            self.ssb_compute_status = "Compute SSB is available only in a live Python kernel. Precompute SSB before exporting standalone HTML."
            return
        if self.ssb_compute_busy:
            self.ssb_compute_status = "SSB compute is already running."
            return
        kwargs = {key: payload[key] for key in ("semiangle_mrad", "scan_sampling_A", "det_sampling_mrad", "voltage_kV",
                                                "bf_radius", "bf_intensity_threshold", "rotation_angle_deg", "n_trials",
                                                "refine", "seed", "lock_aberrations") if key in payload}
        if "n_trials" in kwargs:
            self.ssb_compute_n_trials = int(kwargs["n_trials"])
        if "refine" in kwargs:
            self.ssb_compute_refine = bool(kwargs["refine"])
        manual = bool(payload.get("manual_aberrations", False))
        if manual:
            self.ssb_compute_c10_nm = float(payload.get("c10_nm", self.ssb_compute_c10_nm))
            self.ssb_compute_c12_nm = float(payload.get("c12_nm", self.ssb_compute_c12_nm))
            self.ssb_compute_phi12_deg = float(payload.get("phi12_deg", self.ssb_compute_phi12_deg))
            self.ssb_compute_rotation_angle_deg = float(payload.get("rotation_angle_deg", self.ssb_compute_rotation_angle_deg))
            kwargs["aberrations"] = {"C10": self.ssb_compute_c10_nm, "C12": self.ssb_compute_c12_nm,
                                     "phi12": math.radians(self.ssb_compute_phi12_deg)}
            kwargs["rotation_angle_deg"] = self.ssb_compute_rotation_angle_deg
            kwargs["lock_aberrations"] = True
        self.ssb_compute_lock_c10 = bool(payload.get("lock_c10", self.ssb_compute_lock_c10))
        self.ssb_compute_lock_c12 = bool(payload.get("lock_c12", self.ssb_compute_lock_c12))
        if not manual:
            kwargs["lock_c10"] = self.ssb_compute_lock_c10
            kwargs["lock_c12"] = self.ssb_compute_lock_c12

        def worker() -> None:
            """Run the solver off the kernel thread so the frontend stays responsive while it fits."""
            try:
                self.compute_ssb(verbose=bool(payload.get("verbose", False)), **kwargs)
            except (ValueError, RuntimeError, MemoryError):
                return  # compute_ssb already published the failure string

        threading.Thread(target=worker, name="Show4DSTEM-compute-SSB", daemon=True).start()

    # ------------------------------------------------------------------
    # Offline packing
    # ------------------------------------------------------------------

    def _pack_offline(self, offline: bool | None) -> None:
        """Gzip the counts into ``_offline_stack`` so the browser computes without a kernel.

        ``uint8`` clips counts to 255 for compact browse data; ``offline_dtype="uint16"``
        preserves detector counts. ``offline=None`` packs when the stack is under
        the budget, ``True`` warns and skips above it, ``False`` does nothing.
        """
        bytes_per_pixel = 2 if self._offline_dtype == "uint16" else 1
        n_bytes = math.prod(self._data.shape) * bytes_per_pixel
        if offline is None:
            offline = n_bytes <= OFFLINE_BUDGET_BYTES
        if not offline:
            return
        if n_bytes > OFFLINE_BUDGET_BYTES:
            print(f"  offline browser mode skipped: stack is {n_bytes / 1e6:.0f} MB > {OFFLINE_BUDGET_BYTES / 1e6:.0f} MB budget; the kernel still works")
            return
        html_export.pack_inline(self, self._offline_dtype)

    # ------------------------------------------------------------------
    # Display handshake and saved state
    # ------------------------------------------------------------------

    def _schedule_initial_view_sync(self) -> None:
        """Re-send the view buffers after the frontend connects.

        A heavy ``Bytes`` trait set during ``__init__`` is missed when the
        frontend mounts a tick later and the virtual image stays black until the
        first interaction. The ready message is the deterministic path; the two
        delays cover frontends that do not send it. No-op outside a kernel.
        """
        loop = IOLoop.current(instance=False)
        if loop is None:
            return
        for delay in (0.3, 1.5):
            loop.call_later(delay, self._resend_view_state)

    def _resend_view_state(self) -> None:
        """Re-send the view buffers so a frontend that mounted after ``__init__`` still paints them."""
        for name in ("virtual_image_bytes", "vi_product_labels", "vi_product_map_frames", "vi_product_maps_bytes",
                     "frame_bytes", "compare_diffraction_bytes", "compare_diffraction_indices", "compare_virtual_image_bytes"):
            self.send_state(name)

    def _handle_frontend_ready_msg(self, _widget, content, _buffers) -> None:
        """Answer the frontend's ready message with the view buffers (``_schedule_initial_view_sync``)."""
        if isinstance(content, dict) and content.get("type") == "show4dstem_frontend_ready" and content.get("version") == 1:
            self._resend_view_state()

    def __repr__(self) -> str:
        """Shape, calibration, cursor and title on one line."""
        shape = (f"({self.n_frames}, {self.shape_rows}, {self.shape_cols}, {self.det_rows}, {self.det_cols})"
                 if self.n_frames > 1 else f"({self.shape_rows}, {self.shape_cols}, {self.det_rows}, {self.det_cols})")
        frame = f", {self.frame_dim_label.lower()}={self.frame_idx}" if self.n_frames > 1 else ""
        title = f", title='{self.title}'" if self.title else ""
        return (f"Show4DSTEM(shape={shape}, sampling=({self.pixel_size} {self.pixel_unit}, "
                f"{self.k_pixel_size} {self.k_pixel_unit}), pos=({self.pos_row}, {self.pos_col}){frame}{title})")

    _static_png_b64 = static_png_b64  # the saved-notebook preview StaticFallbackMixin renders

    # ------------------------------------------------------------------
    # Observers
    # ------------------------------------------------------------------

    def _set_roi_to_disk(self) -> None:
        """Place a circular ROI on the fitted bright-field disk without firing recomputes."""
        self._suppress_roi_recompute = True
        with self.hold_trait_notifications():
            self.roi_mode = "circle"
            self.roi_center_col = self.center_col
            self.roi_center_row = self.center_row
            self.roi_center = [self.center_row, self.center_col]
            self.roi_radius = float(max(1.0, self.bf_radius))
            self.roi_active = True
        self._suppress_roi_recompute = False

    def _on_preset_request(self, change) -> None:
        """React to a frontend preset button: apply it, then clear the request so the same preset can be clicked again."""
        name = (change.get("new") or "").strip().lower()
        if name in PRESET_RADII:
            self.apply_preset(name)
            self._preset_request = ""

    def _on_roi_change(self, change=None) -> None:
        """React to a detector ROI edit: recompute the image in view unless a batched update or a product map is showing."""
        if not self.roi_active or self._suppress_roi_recompute or self.vi_source != "roi":
            return
        if not self._multiple_view_active():
            self._compute_virtual_image_from_roi()
        self._refresh_compare_virtual_images()

    def _on_roi_center_change(self, change=None) -> None:
        """Drag fast path: the frontend sends ``[row, col]`` as one trait per mouse move."""
        if not self.roi_active or self._suppress_roi_recompute:
            return
        if change and "new" in change:
            row, col = change["new"]
            self.unobserve(self._on_roi_change, names=["roi_center_col", "roi_center_row"])
            self.roi_center_row = row
            self.roi_center_col = col
            self.observe(self._on_roi_change, names=["roi_center_col", "roi_center_row"])
        self._on_roi_change()

    def _on_calibration_change(self, change=None) -> None:
        """Drop the cached preset images when the disk centre or radius moves: they were summed over the old masks."""
        self._preset_images = {}

    def _on_frame_idx_change(self, change=None) -> None:
        """React to a dataset switch: the cached presets belong to the old dataset, so drop them and repaint."""
        if self.n_frames <= 1:
            return
        self._preset_images = {}
        if self._multiple_view_active():
            self._sync_compare_page_to_frame_idx()
        else:
            self._compute_virtual_image_from_roi()
        self._update_frame()

    def _on_vi_roi_center_change(self, change=None) -> None:
        """Apply the compound ``[row, col]`` update atomically and acknowledge it with a receipt."""
        if change and "new" in change:
            row, col = change["new"]
            self.unobserve(self._on_vi_roi_change, names=["vi_roi_center_row", "vi_roi_center_col"])
            self.vi_roi_center_row = float(row)
            self.vi_roi_center_col = float(col)
            self.observe(self._on_vi_roi_change, names=["vi_roi_center_row", "vi_roi_center_col"])
        self._on_vi_roi_change()
        revision = self.vi_roi_receipt[2] + 1 if self.vi_roi_receipt else 1
        self.vi_roi_receipt = [self.vi_roi_center_row, self.vi_roi_center_col, float(revision)]

    def _on_vi_roi_change(self, change=None) -> None:
        """React to a scan-ROI edit: reduce the patterns inside it, or return the panel to the cursor pattern when it is off."""
        if self.vi_roi_mode == "off":
            self.vi_roi_dp_bytes = b""
            self._publish_all_diffraction()
            return
        self._compute_vi_roi_dp()

    def _on_compare_config_change(self, change=None) -> None:
        """React to a grid layout change: normalise it, keep one panel visible, rebuild the grid, keep the dataset in view."""
        if change and change.get("name") == "view_mode":
            self.view_mode = self._normalise_view_mode(change.get("new", "single"))
        if change and change.get("name") == "compare_group_mode":
            self.compare_group_mode = self._normalise_compare_group_mode(change.get("new", "paged"))
        if change and change.get("name") == "compare_hidden_panels":
            hidden = self._panel_set(change.get("new", []))
            if len(hidden) >= self.n_frames:
                # Hiding every panel would leave the grid blank; keep one visible.
                keep = next((idx for idx in self._panel_order() if idx not in self._panel_set(change.get("old", []))), 0)
                hidden.discard(keep)
                self.compare_hidden_panels = sorted(hidden)
        self._refresh_compare_virtual_images()
        if self._multiple_view_active():
            visible = self._page_indices()
            if visible and int(self.frame_idx) not in visible:
                self.frame_idx = int(visible[0])
        else:
            self._compute_virtual_image_from_roi()
        self._update_frame()

    def _on_compare_dp_mode_change(self, change=None) -> None:
        """React to the comparison pattern mode (average, selected, all): normalise it and resend the pattern."""
        if change:
            self.compare_dp_mode = self._normalise_compare_dp_mode(change.get("new", "average"))
        if self._multiple_view_active():
            self._update_frame()

    # ------------------------------------------------------------------
    # Comparison grid
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_view_mode(value: str) -> str:
        """``"single"`` or ``"multiple"`` from a caller, frontend or saved-state value; anything else is refused."""
        mode = str(value or "single").strip().lower().replace("-", "_")
        if mode not in {"single", "multiple"}:
            raise ValueError(f"view_mode must be 'single' or 'multiple', got {value!r}")
        return mode

    @staticmethod
    def _normalise_compare_dp_mode(value: str) -> str:
        """``"average"``, ``"selected"`` or ``"all"``, accepting short aliases; anything else is refused."""
        mode = str(value or "average").strip().lower().replace("-", "_")
        mode = {"avg": "average", "mean": "average", "current": "selected"}.get(mode, mode)
        if mode not in {"average", "selected", "all"}:
            raise ValueError(f"compare_dp_mode must be 'average', 'selected', or 'all', got {value!r}")
        return mode

    @staticmethod
    def _normalise_compare_group_mode(value: str) -> str:
        """``"paged"`` or ``"all"``, accepting aliases; anything else is refused."""
        mode = str(value or "paged").strip().lower().replace("-", "_")
        mode = {"page": "paged", "pages": "paged", "group": "paged", "groups": "paged",
                "collapse": "all", "collapsed": "all", "single": "all"}.get(mode, mode)
        if mode not in {"paged", "all"}:
            raise ValueError(f"compare_group_mode must be 'paged' or 'all', got {value!r}")
        return mode

    def _multiple_view_active(self) -> bool:
        """Whether the comparison grid shows: multiple view requested and more than one dataset to compare."""
        return self.view_mode == "multiple" and self.n_frames > 1

    def _panel_title(self, panel: int) -> str:
        """Grid caption of a dataset: its label, else ``"<frame_dim_label> <n>"``."""
        if 0 <= panel < len(self.frame_labels) and self.frame_labels[panel]:
            return str(self.frame_labels[panel])
        return f"{self.frame_dim_label} {panel + 1}"

    def _panel_set(self, values) -> set[int]:
        """Validated dataset indices from a frontend list trait."""
        return {int(idx) for idx in values if not isinstance(idx, bool) and 0 <= int(idx) < self.n_frames}

    def _panel_order(self) -> list[int]:
        """Dataset indices in display order: the user's order when it is a full permutation."""
        order = [int(idx) for idx in self.compare_panel_order]
        if sorted(order) == list(range(self.n_frames)):
            return order
        return list(range(self.n_frames))

    def _page_indices(self) -> list[int]:
        """Visible datasets of the active page (every visible one when groups are collapsed)."""
        ordered = self._panel_order()
        hidden = self._panel_set(self.compare_hidden_panels)
        page_size = max(1, int(self.compare_max_panels))
        page_count = max(1, math.ceil(len(ordered) / page_size))
        if int(self.compare_page_count) != page_count:
            self.compare_page_count = page_count
        page_idx = max(0, min(int(self.compare_page_idx), page_count - 1))
        if int(self.compare_page_idx) != page_idx:
            self.compare_page_idx = page_idx
        if self.compare_group_mode == "all":
            visible = [idx for idx in ordered if idx not in hidden]
            if not visible and ordered:
                # A state load must not leave the grid blank: unhide the first panel.
                self.compare_hidden_panels = sorted(hidden - {ordered[0]})
                visible = [ordered[0]]
            return visible
        page = ordered[page_idx * page_size : (page_idx + 1) * page_size]
        return [idx for idx in page if idx not in hidden]

    def _sync_compare_page_to_frame_idx(self) -> None:
        """Show the page that contains the active dataset."""
        ordered = self._panel_order()
        frame_idx = int(self.frame_idx)
        if frame_idx in self._panel_set(self.compare_hidden_panels):
            return
        page_idx = ordered.index(frame_idx) // max(1, int(self.compare_max_panels))
        if int(self.compare_page_idx) != page_idx:
            self.compare_page_idx = page_idx

    def _compare_status(self, shown: Sequence[int]) -> str:
        """Grid status line: shown panels out of the unhidden total, with the page when the grid is paged."""
        label = self.frame_dim_label.lower()
        total = self.n_frames - len(self._panel_set(self.compare_hidden_panels))
        if self.compare_group_mode == "all":
            return f"{len(shown)}/{total} {label} panels{' · all groups' if self.compare_page_count > 1 else ''}"
        page = f" · page {self.compare_page_idx + 1}/{self.compare_page_count}" if self.compare_page_count > 1 else ""
        return f"{len(shown)}/{total} {label} panels{page}"

    def _clear_compare_virtual_images(self, status: str = "") -> None:
        """Empty the comparison grid and show ``status`` (after ``free``, or when the grid has nothing to draw)."""
        with self.hold_trait_notifications():
            self.compare_virtual_image_bytes = b""
            self.compare_panel_count = 0
            self.compare_panel_indices = []
            self.compare_status = status

    def _refresh_compare_virtual_images(self) -> None:
        """Build the virtual-image stack of the visible datasets for the comparison grid."""
        if self._data is None:
            self._clear_compare_virtual_images()
            return
        if not self._multiple_view_active():
            if self.compare_panel_count or self.compare_virtual_image_bytes:
                self._clear_compare_virtual_images()
            return
        if self._suppress_compare_recompute:
            return
        self._suppress_compare_recompute = True
        try:
            indices = self._page_indices()
            if not indices:
                self._clear_compare_virtual_images(f"0/{self.n_frames} {self.frame_dim_label.lower()} panels · hidden")
                return
            if self.vi_source != "roi":
                # Product maps travel once in vi_product_maps_bytes; the grid only needs the indices.
                with self.hold_trait_notifications():
                    self.compare_panel_count = len(indices)
                    self.compare_panel_indices = indices
                    self.compare_status = self._compare_status(indices)
                return
            with self._compare_lock:
                images = self._compare_images(indices, self._current_detector_mask())
            with self.hold_trait_notifications():
                self.compare_virtual_image_bytes = np.stack(images).tobytes()
                self.compare_panel_count = len(indices)
                self.compare_panel_indices = indices
                self.compare_status = self._compare_status(indices)
        except (RuntimeError, ValueError, MemoryError) as exc:
            self._clear_compare_virtual_images(f"Multiple grid unavailable: {exc}")
        finally:
            self._suppress_compare_recompute = False

