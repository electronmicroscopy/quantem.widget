"""Show4DSTEM: scan position, virtual detector and diffraction pattern of a 4D-STEM scan.

One class opens every input the tutorials use: a file path (read with
``read_4dstem``), a ``quantem.gpu.io.Dataset4dstem`` (kept on its GPU, read
in bounded windows), a quantem core ``Dataset4dstem``, a list of either (one
virtual image per dataset under a shared detector), a NumPy
array or a Torch tensor (dense, on the tensor's device), or a folder of linked
HDF5 masters or .qem acquisitions read by the browser itself (``h5_urls`` or
``qem_urls``, the WebGPU folder exports).

quantem.gpu and quantem core are reached only through
``quantem.widget.adapters``. A file becomes a ``Dataset4dstem`` when
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
import time
from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from typing import Self

import anywidget
import numpy as np
import torch
import traitlets
from ipywidgets.widgets.trait_types import CByteMemoryView
from tornado.ioloop import IOLoop

from quantem.widget.adapters import core as core_adapter
from quantem.widget.adapters import gpu as gpu_adapter
from quantem.widget.adapters.gpu import AcquisitionSeries, AcquisitionView
from quantem.widget.counts import (
    counts_tensor,
    detector_bin_mean,
    host_dtype_for_torch,
)
from quantem.widget.device import (
    gpu_notice,
    gpu_path_hint,
    no_gpu_path,
    resolve_device,
)
from quantem.widget.export import HtmlExportMixin, export_slug
from quantem.widget.fallback import StaticFallbackMixin
from quantem.widget.show4dstem import export as html_export
from quantem.widget.show4dstem import reader
from quantem.widget.show4dstem.dense import DenseSession
from quantem.widget.show4dstem.detector import (
    PRESET_RADII,
    detector_reach_px,
    fit_disk,
    preset_mask,
    preset_radii,
    roi_mask,
    scan_indices,
)
from quantem.widget.show4dstem.preview import static_png_b64
from quantem.widget.show4dstem.reader import acquisition_name
from quantem.widget.show4dstem.session import DetectorSessionSource, SessionSeries
from quantem.widget.show4dstem.streaming import StreamFollower
from quantem.widget.state import SavedStateMixin
from quantem.widget.units import acquisition_calibration, per_axis, stated, wrap_degrees
from quantem.widget.utils.array import device_array
from quantem.widget.utils.state_io import read_state
from quantem.widget.utils.ui import (
    DisplayMode, ViewerSize, reject_unknown_kwargs, resolve_panel_width, resolve_mode,
)

BRIGHT_FIELD_RADIUS_FRACTION = 0.125  # bright-field radius as a fraction of the detector side before fitting
DEFAULT_VI_ROI_RATIO = 0.15  # scan ROI radius as a fraction of the scan side
PRODUCT_LABELS = ("DPC_row", "DPC_col", "Direct")
# the calibration a 4D acquisition carries, under the .qem format's flat names
# direct_phase()'s downloadable record; ShowPtycho's calibration.json has another shape and schema
DIRECT_CALIBRATION_SCHEMA = "quantem.show4dstem-direct-calibration/5"
FROM_TRAIT = object()  # a direct_phase() setting left to its direct_* trait (None is a value of its own)
CALIBRATION_KEYS = ("scan_sampling_A", "detector_sampling_mrad", "detector_sampling_inv_A", "voltage_kV", "semiangle_mrad")
# Soft guide for the inline offline pack: the browser's JSON parse of the embedded
# base64 is the real ceiling, and gzip keeps a 2 GB uint8 stack under it.
OFFLINE_BUDGET_BYTES = 2000 * 1024 * 1024


class _ImageMemoryView(CByteMemoryView):
    """Publish a new immutable image buffer without comparing every byte.

    Each computed image owns its buffer. Retained views remain valid through
    transport; callers must not mutate a buffer after publishing it.
    """

    def set(self, obj: traitlets.HasTraits, value: bytes | memoryview) -> None:
        value = self._validate(obj, value)
        assert self.name is not None
        old = obj._trait_values.get(self.name, self.default_value)
        obj._trait_values[self.name] = value
        if old is not value:
            obj._notify_trait(self.name, old, value)


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
        ``read_4dstem`` (a ``Dataset4dstem`` or a quantem core
        ``Dataset4dstem``), a list of datasets, a 4D ``(scan_row, scan_col,
        det_row, det_col)`` or 5D ``(n_datasets, ...)`` array or tensor, or a
        3D ``(positions, det_row, det_col)`` flat scan with ``scan_shape``. A
        path opens as ``read_4dstem(path, device)`` reads it: on the GPU
        through quantem.gpu when it is installed and a GPU is present, else
        densely, block by block onto ``device`` (one printed line; refused
        above a memory ceiling, never reduced). None with ``h5_urls`` or
        ``qem_urls``, whose data the browser reads.
        A prepared ``DetectorSessionSource`` keeps its acquisition resident
        and supplies bounded detector queries, including distributed CUDA
        sessions. The widget closes that borrowed session on ``free()``;
        close its source acquisition separately. Such sessions require
        ``offline=False`` and do not expose direct ptychography or source cropping here.
        A loaded scan with ``detector_session()`` prepares that session
        automatically; pass the scan directly to ``Show4DSTEM``.
    device
        Where a NumPy array or a file is placed and reduced: ``"auto"`` (CUDA,
        then MPS, then CPU; printed once), ``"cuda"``, ``"cuda:N"``, ``"mps"``
        or ``"cpu"``. A tensor stays on its device unless this names another.
    scan_region
        ``(row_start, row_stop, col_start, col_stop)`` of one 4D scan to view,
        exclusive stops: a GPU acquisition is read only there, a dense scan is
        cut before it moves to its device.
    hot_pixel_correction
        A file path only: ``"median"`` (default) replaces the detector pixels
        the master flags with the median of their unflagged 3x3 neighbours (a larger
        window inside a flagged cluster, half to even, as quantem.gpu's io.load),
        ``"zero"`` sets them to 0 and ``"none"`` keeps the stored values; one
        line says how many pixels were replaced or left.
    scan_sampling_A, detector_sampling_mrad, detector_sampling_inv_A, voltage_kV, semiangle_mrad
        Calibration of an input that records none (an array, an uncalibrated
        file): the scan step ``(row, col)`` in angstrom, the detector step
        ``(row, col)`` in mrad or in 1/angstrom per pixel (one of the two),
        the accelerating voltage in kV and the convergence semiangle in mrad.
        A dataset or file is the source of its own calibration: these are read
        from it, and a keyword that contradicts a recorded value raises.
    bright_field_center_px, bright_field_radius_px
        Bright-field disk centre ``(row, col)`` and edge radius in detector pixels. Omitted, the disk a
        dataset recorded with ``data.calibration.detector_sampling()``, else that same fit
        (``quantem.gpu.detector.bright_field_disk``) of the mean pattern: the BF, ABF, ADF and
        HAADF presets are multiples of this radius.
    precompute_virtual_images
        Compute the four preset images up front so preset clicks are instant.
    size, dp_size, image_size
        Named display widths: ``"small"``, ``"medium"``, ``"large"`` or
        ``"xlarge"``. ``size`` sets both panels; ``dp_size`` and ``image_size``
        override diffraction and virtual images independently. Omitted, the
        diffraction pattern shows at 300 px and the virtual image (or scan grid)
        fills the rest of the row. Source pixels and scientific precision do not change.
    panel_width_px, dp_panel_width_px, vi_panel_width_px
        Display widths in CSS pixels. The diffraction and virtual-image widths
        override ``panel_width_px`` independently; zero uses the shared default.
        Resizing changes the display only, never the data or detector sampling.
    sync_resize
        Resize both panels together when dragging a corner. Defaults to True,
        or False when ``dp_size`` or ``image_size`` is supplied.
    view_mode, compare_columns, compare_grid_width_px, compare_max_panels, compare_group_mode, compare_dp_mode
        Comparison grid layout for several datasets: ``"multiple"`` shows one
        virtual image per dataset, ``compare_max_panels`` per page, with the
        diffraction pattern of the ``"selected"`` dataset or the ``"average"``.
    offline, offline_dtype
        Pack the counts for the browser so the widget works without a kernel
        (``None`` packs automatically under a 2 GB budget). A GPU acquisition
        is read into the pack in small scan windows.
    h5_urls, qem_urls, scan_shape, detector_shape
        Browser WebGPU source over linked HDF5 masters or .qem acquisitions,
        written by the CLI and ``show4dstem.export.write_qem_bundle``.
    """

    _esm = pathlib.Path(__file__).parent.parent / "static" / "show4dstem.js"

    title = traitlets.Unicode("").tag(sync=True)
    show_title = traitlets.Bool(True).tag(sync=True)
    # Live acquisition: the diffraction pattern follows the newest arrived pattern, except while the user
    # interacts and for STREAM_FOLLOW_PAUSE_S after (drag the probe, let go, and it returns to the newest).
    stream_follow = traitlets.Bool(True)
    pos_row = traitlets.Int(0).tag(sync=True)
    pos_col = traitlets.Int(0).tag(sync=True)
    shape_rows = traitlets.Int(1).tag(sync=True)
    shape_cols = traitlets.Int(1).tag(sync=True)
    det_rows = traitlets.Int(1).tag(sync=True)
    det_cols = traitlets.Int(1).tag(sync=True)
    frame_bytes = traitlets.Bytes(b"").tag(sync=True)
    bright_field_center_col_px = traitlets.Float(0.0).tag(sync=True)
    bright_field_center_row_px = traitlets.Float(0.0).tag(sync=True)
    bright_field_radius_px = traitlets.Float(0.0).tag(sync=True)
    roi_active = traitlets.Bool(False).tag(sync=True)
    roi_mode = traitlets.Unicode("point").tag(sync=True)
    roi_center_col = traitlets.Float(0.0).tag(sync=True)
    roi_center_row = traitlets.Float(0.0).tag(sync=True)
    roi_center = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0]).tag(sync=True)
    roi_radius = traitlets.Float(10.0).tag(sync=True)
    roi_radius_inner = traitlets.Float(5.0).tag(sync=True)
    roi_width = traitlets.Float(20.0).tag(sync=True)
    roi_height = traitlets.Float(10.0).tag(sync=True)
    virtual_image_bytes = _ImageMemoryView(b"").tag(sync=True)
    vi_source = traitlets.Unicode("roi").tag(sync=True)
    vi_product_labels = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    vi_product_map_frames = traitlets.Int(0).tag(sync=True)
    vi_product_maps_bytes = traitlets.Bytes(b"").tag(sync=True)
    offline = traitlets.Bool(False).tag(sync=True)
    _offline_stack = traitlets.Bytes(b"").tag(sync=True)
    _h5_urls = traitlets.Unicode("").tag(sync=True)
    _qem_urls = traitlets.Unicode("").tag(sync=True)
    _offline_bad_px = traitlets.Unicode("").tag(sync=True)
    direct_request = traitlets.Unicode("").tag(sync=True)
    direct_status = traitlets.Unicode("").tag(sync=True)
    direct_busy = traitlets.Bool(False).tag(sync=True)
    direct_enabled = traitlets.Bool(True).tag(sync=True)
    direct_trials = traitlets.Int(200).tag(sync=True)
    # quantem.gpu's default direct ptychography: the model fit with the sample tilt and depth spread; False fits at tilt 0
    direct_tilt = traitlets.Bool(True).tag(sync=True)
    # local refinement after the trial search: "auto", quantem.gpu's exact-derivative refinement, or None for none
    direct_refinement = traitlets.Enum(["auto"], default_value="auto", allow_none=True).tag(sync=True)
    direct_bright_field_pixels = traitlets.Int(0).tag(sync=True)
    direct_lock_C10 = traitlets.Bool(False).tag(sync=True)
    direct_lock_C12 = traitlets.Bool(False).tag(sync=True)
    direct_C10_nm = traitlets.Float(0.0).tag(sync=True)
    direct_C12_nm = traitlets.Float(0.0).tag(sync=True)
    direct_phi12_deg = traitlets.Float(0.0).tag(sync=True)
    # the physical scan-to-detector rotation in [0, 360): the reversal of the centre of mass is part of the angle
    direct_scan_detector_rotation_deg = traitlets.Float(0.0).tag(sync=True)
    direct_calibration_json = traitlets.Unicode("").tag(sync=True)
    direct_calibration_filename = traitlets.Unicode("").tag(sync=True)
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
    # calibration under the .qem flat names, each [row, col]; an empty list is an uncalibrated axis pair,
    # and at most one of the two detector lists is filled
    scan_sampling_A = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    detector_sampling_mrad = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    detector_sampling_inv_A = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    _static_fallback_image = traitlets.Unicode("").tag(sync=True)
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
    dp_panel_width_px = traitlets.Int(0).tag(sync=True)
    vi_panel_width_px = traitlets.Int(0).tag(sync=True)
    sync_resize = traitlets.Bool(True).tag(sync=True)
    dp_show_colorbar = traitlets.Bool(False).tag(sync=True)
    vi_auto_contrast = traitlets.Bool(False).tag(sync=True)
    vi_smooth = traitlets.Bool(False).tag(sync=True)
    frame_idx = traitlets.Int(0).tag(sync=True)
    n_frames = traitlets.Int(1).tag(sync=True)
    frame_dim_label = traitlets.Unicode("Frame").tag(sync=True)
    frame_labels = traitlets.List(traitlets.Unicode(), []).tag(sync=True)
    _stream_source_ids = traitlets.List(traitlets.Unicode(), []).tag(sync=True)
    _stream_selected_source = traitlets.Unicode("").tag(sync=True)
    frame_playing = traitlets.Bool(False).tag(sync=True)
    frame_loop = traitlets.Bool(True).tag(sync=True)
    frame_fps = traitlets.Float(5.0).tag(sync=True)
    frame_reverse = traitlets.Bool(False).tag(sync=True)
    frame_boomerang = traitlets.Bool(False).tag(sync=True)
    view_mode = traitlets.Unicode("single").tag(sync=True)
    compare_columns = traitlets.Int(0).tag(sync=True)
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
    compare_virtual_image_bytes = _ImageMemoryView().tag(sync=True)
    compare_image_revision = traitlets.Int(0).tag(sync=True)
    _stream_panel_update = traitlets.Dict().tag(sync=True)
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
        scan_sampling_A: float | tuple[float, float] | None = None,
        detector_sampling_mrad: float | tuple[float, float] | None = None,
        detector_sampling_inv_A: float | tuple[float, float] | None = None,
        voltage_kV: float | None = None,
        semiangle_mrad: float | None = None,
        bright_field_center_px: tuple[float, float] | None = None,
        bright_field_radius_px: float | None = None,
        hot_pixel_correction: str = "median",
        precompute_virtual_images: bool | None = None,
        frame_dim_label: str | None = None,
        frame_labels: list[str] | None = None,
        view_mode: str | None = None,
        compare_columns: int = 0,
        compare_grid_width_px: int = 0,
        compare_max_panels: int = 12,
        compare_group_mode: str = "paged",
        compare_dp_mode: str | None = None,
        title: str = "",
        mode: DisplayMode = "interactive",
        show_title: bool | None = None,
        show_controls: bool | None = None,
        controls_collapsed: bool | None = None,
        show_stats: bool | None = None,
        show_scale_bar: bool | None = None,
        offline: bool | None = False,
        offline_dtype: str = "uint8",
        h5_urls: Sequence[str] | None = None,
        qem_urls: Sequence[str] | None = None,
        show_fft: bool = False,
        size: ViewerSize | None = None,
        dp_size: ViewerSize | None = None,
        image_size: ViewerSize | None = None,
        panel_width_px: int = 0,
        dp_panel_width_px: int = 0,
        vi_panel_width_px: int = 0,
        sync_resize: bool | None = None,
        verbose: bool = False,
        state=None,
        save_state: bool = False,
        notebook_preview_format: str | None = "jpeg",
        notebook_preview_quality: int = 88,
        notebook_preview_max_px: int = 512,
        **kwargs,
    ):
        """Validate the arguments, place the data on its compute device and set the traits (see the class docstring)."""
        # close() runs even when a check below refuses the call (ipywidgets closes in __del__), so its state exists first
        self._data = None  # get_state runs inside open(); the data joins once the model exists
        self._verbose = bool(verbose)
        self._sessions: dict[int, object] = {}  # DenseSession or a quantem.gpu detector session per dataset
        self._compare_session = None
        self._compare_session_indices: tuple[int, ...] = ()
        self._stream_completed_sources: dict[int, object] = {}
        self._vi_product_maps: dict[str, np.ndarray] = {}
        self._preset_images: dict[str, bytes | memoryview] = {}
        self._suppress_roi_recompute = False
        self._roi_seen: tuple | None = None  # the detector ROI at the last ROI notification
        self._position_seen: tuple | None = None  # the scan position at the last position notification
        self._last_interaction_at = 0.0
        self._suppress_compare_recompute = False
        self._owned_acquisitions = []
        self._stream_lock = threading.RLock()
        self._stream_follower = None
        self._stream_source_serial = 0
        self._stream_display_view = None
        self._stream_display_views: dict[str, bool] = {}
        self._stream_display_negotiated = False
        self._stream_display_delta = False
        self._stream_layout_views: set[str] = set()
        self._stream_display_probe_seq = None
        self._stream_display_probe_at = 0.0
        self._stream_display_seq = 0
        self._stream_display_acknowledged = 0
        self._stream_display_credits = 2
        self._stream_display_pending = 0
        self._stream_display_revision = 0
        self._stream_display_sent_at = 0.0
        self._stream_cell_running = False
        self._stream_cell_hooks = None
        self._stream_defer_compute = False
        self._stream_deferred_state: set[str] = set()
        self._stream_deferred_snapshot: dict | None = None
        self._compare_lock = threading.RLock()
        self._session_lock = threading.RLock()
        self._direct_flipped: bool | None = None  # the last direct ptychography fit's flip, None before one
        # the last tilt-corrected fit's {sample_tilt_mrad, depth_spread_nm}: a manual recompute keeps the sample model
        self._direct_sample: dict | None = None
        # the bright-field disk the caller gave (direct_phase() uses it); the widget's own disk fit is display only
        self._given_bright_field = {"bright_field_center_px": None if bright_field_center_px is None else
                                    tuple(float(value) for value in bright_field_center_px),
                                    "bright_field_radius_px": None if bright_field_radius_px is None else float(bright_field_radius_px)}
        reject_unknown_kwargs(type(self), kwargs)
        # save_state=False keeps the packed 4D stack out of the notebook's saved
        # widget metadata; a static preview stands in for a cold reopen.
        self._save_state = bool(save_state)
        self._configure_static_fallback(
            notebook_preview_format=notebook_preview_format,
            notebook_preview_quality=notebook_preview_quality,
            notebook_preview_max_px=notebook_preview_max_px,
        )
        ui = resolve_mode(
            mode,
            defaults={"show_title": True, "show_controls": True, "controls_collapsed": False,
                      "show_stats": True, "show_scale_bar": True},
            overrides={"show_title": show_title, "show_controls": show_controls,
                       "controls_collapsed": controls_collapsed, "show_stats": show_stats,
                       "show_scale_bar": show_scale_bar},
        )
        # Every check and the file read run before super().__init__ opens the widget model, so a
        # refused input (a bad argument, a file above the memory ceiling, a state of another schema)
        # leaves no model behind.
        state = None if state is None else read_state(state, "Show4DSTEM", (*self._STATE_KEYS, *self._LIST_STATE_KEYS))
        if offline_dtype not in ("uint8", "uint16"):
            raise ValueError(
                f"offline_dtype must be 'uint8' or 'uint16', got {offline_dtype!r}. "
                "Use 'uint8' for compact browse data or 'uint16' to preserve detector counts."
            )
        for preset, width, name in ((size, panel_width_px, "size"),
                                    (dp_size, dp_panel_width_px, "dp_size"),
                                    (image_size, vi_panel_width_px, "image_size")):
            if preset is not None and width:
                raise ValueError(f"Use {name} or its explicit pixel width, not both.")
        if size is not None:
            panel_width_px = resolve_panel_width(size)
        if dp_size is not None:
            dp_panel_width_px = resolve_panel_width(dp_size)
        if image_size is not None:
            vi_panel_width_px = resolve_panel_width(image_size)
        if sync_resize is None:
            sync_resize = dp_size is None and image_size is None
        for name, value in (("panel_width_px", panel_width_px), ("dp_panel_width_px", dp_panel_width_px),
                            ("vi_panel_width_px", vi_panel_width_px), ("compare_columns", compare_columns),
                            ("compare_grid_width_px", compare_grid_width_px)):
            if int(value) < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if int(compare_max_panels) < 1:
            raise ValueError(f"compare_max_panels must be >= 1, got {compare_max_panels}")
        if h5_urls and qem_urls:
            raise ValueError("Pass h5_urls= or qem_urls=, not both: one viewer reads one kind of browser source.")
        browser_urls = h5_urls or qem_urls
        waiting = data is None and not browser_urls
        if browser_urls and (scan_shape is None or detector_shape is None):
            raise ValueError("Show4DSTEM(..., h5_urls= or qem_urls=...) requires scan_shape=(rows, cols) and detector_shape=(rows, cols).")

        # Route the input to its storage: bounded views over GPU acquisitions,
        # a dense tensor on its device, or no data at all for a browser source.
        if not isinstance(data, (str, os.PathLike)) and hot_pixel_correction != "median":
            raise ValueError("hot_pixel_correction applies when Show4DSTEM reads a file; load the data with the "
                             "correction you want (quantem.gpu.io.load or read_4dstem) and pass it.")
        if isinstance(data, (str, os.PathLike)) and reader.gpu_route(device):
            data = reader.read_4dstem(data, device, scan_shape=scan_shape, hot_pixel_correction=hot_pixel_correction)
            if gpu_adapter.is_acquisition(data):
                self._owned_acquisitions = [data]
        elif isinstance(data, (str, os.PathLike)):
            # read_4dstem's dense read, with the counts going straight onto the compute device block by block
            gpu_notice()
            title = title or acquisition_name(data)
            data = reader.read_dense(data, resolve_device(device), scan_shape=scan_shape,
                                     hot_pixel_correction=hot_pixel_correction)
        make_session = getattr(data, "detector_session", None)
        if callable(make_session):
            if scan_region is not None or browser_urls or offline is not False:
                raise ValueError("A prepared detector session is already scoped to its scan; use offline=False without scan_region or browser URLs.")
            data = make_session()
            if not isinstance(data, DetectorSessionSource):
                raise TypeError("detector_session() must return a DetectorSessionSource with bounded detector queries.")
        prepared = isinstance(data, DetectorSessionSource)
        listed = isinstance(data, (list, tuple)) and len(data) > 0
        prepared_series = listed and all(isinstance(item, DetectorSessionSource)
                                        or callable(getattr(item, "detector_session", None)) for item in data)
        if (prepared or prepared_series or waiting) and (scan_region is not None or browser_urls or offline is not False):
            raise ValueError("A prepared detector session is already scoped to its scan; use offline=False without scan_region or browser URLs.")
        series = listed and any(gpu_adapter.is_acquisition(item) for item in data)
        bounded = gpu_adapter.is_acquisition(data) or series
        datasets = listed and all(core_adapter.is_dataset(item, ndim=4) for item in data)
        recorded = dict.fromkeys(CALIBRATION_KEYS)
        recorded_disk = None  # the bright-field disk data.calibration.detector_sampling() recorded
        if prepared:
            recorded = {key: data.metadata.get(key) for key in CALIBRATION_KEYS}
            recorded_disk = gpu_adapter.recorded_bright_field_disk(data.metadata)
            precompute_virtual_images = False if precompute_virtual_images is None else precompute_virtual_images
        if bounded and browser_urls:
            raise ValueError("h5_urls= and qem_urls= open files in the browser; pass the acquisition without them, or the URLs alone.")
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
            # the dataset is the source of its calibration, read the same way for a GPU acquisition
            # (an unknown axis is None) and a quantem core dataset (an unknown axis is 1.0 "pixels")
            recorded = acquisition_calibration(first.sampling, first.units, first.metadata)
            recorded_disk = gpu_adapter.recorded_bright_field_disk(first.metadata)
        if prepared_series or waiting:
            if prepared_series:
                data = SessionSeries(data)
                recorded = {key: data.frames[0].metadata.get(key) for key in CALIBRATION_KEYS}
            precompute_virtual_images = False
            frame_dim_label = "Dataset" if frame_dim_label is None else frame_dim_label
            view_mode = "multiple" if view_mode is None else view_mode
            compare_dp_mode = "selected" if compare_dp_mode is None else compare_dp_mode
        elif bounded:
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
        view_mode = self._normalize_view_mode("single" if view_mode is None else view_mode)
        compare_dp_mode = self._normalize_compare_dp_mode("average" if compare_dp_mode is None else compare_dp_mode)
        compare_group_mode = self._normalize_compare_group_mode(compare_group_mode)
        if detector_sampling_mrad is not None and detector_sampling_inv_A is not None:
            raise ValueError("Pass detector_sampling_mrad= or detector_sampling_inv_A=, not both: converting one into "
                             "the other needs the voltage, and one detector has one calibration.")
        keywords = {"scan_sampling_A": scan_sampling_A, "detector_sampling_mrad": detector_sampling_mrad,
                    "detector_sampling_inv_A": detector_sampling_inv_A, "voltage_kV": voltage_kV,
                    "semiangle_mrad": semiangle_mrad}
        calibration = {name: stated(name, keywords[name], recorded[name]) for name in CALIBRATION_KEYS}
        if calibration["detector_sampling_mrad"] is not None and calibration["detector_sampling_inv_A"] is not None:
            raise ValueError("The input records its detector step in one unit and the keyword gives the other; "
                             "leave the keyword out.")
        for name in ("scan_sampling_A", "detector_sampling_mrad", "detector_sampling_inv_A"):
            # a pair with an axis still open (recorded on one axis, no keyword for the other) has no geometry to draw
            open_axis = calibration[name] is None or None in per_axis(calibration[name], 2)
            calibration[name] = [] if open_axis else list(self._positive_pair(calibration[name], name))
        if prepared:
            data_device = torch.device(data.device)
            shape = (*data.scan_shape, *data.detector_shape)
        elif waiting:
            data_device, shape = torch.device("cpu"), (0, 1, 1, 1, 1)
        elif browser_urls:
            urls = list(dict.fromkeys(str(url) for url in browser_urls))
            data = None
            data_device = torch.device("cpu")
            shape = (len(urls), *scan_shape, *detector_shape) if len(urls) > 1 else (*scan_shape, *detector_shape)
        else:
            if not isinstance(data, (AcquisitionView, AcquisitionSeries, SessionSeries)):
                data = device_array(data)  # a CuPy array becomes a tensor on its GPU, zero-copy
            if isinstance(data, (AcquisitionView, AcquisitionSeries, SessionSeries)):
                data_device = torch.device(data.device)
            elif isinstance(data, torch.Tensor):
                data_device = torch.device(resolve_device(device, data))
                data = data.to(data_device)
            else:
                data_device = torch.device(resolve_device(device, data))
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
        if prepared or prepared_series or waiting:
            self.direct_enabled = False
        self.view_mode = view_mode
        self.compare_dp_mode = compare_dp_mode
        self.compare_group_mode = compare_group_mode
        self.panel_width_px, self.compare_columns = int(panel_width_px), int(compare_columns)
        self.dp_panel_width_px, self.vi_panel_width_px = int(dp_panel_width_px), int(vi_panel_width_px)
        self.sync_resize = bool(sync_resize)
        self.compare_grid_width_px, self.compare_max_panels = int(compare_grid_width_px), int(compare_max_panels)

        self.scan_sampling_A = calibration["scan_sampling_A"]
        self.detector_sampling_mrad = calibration["detector_sampling_mrad"]
        self.detector_sampling_inv_A = calibration["detector_sampling_inv_A"]
        self.voltage_kV = None if calibration["voltage_kV"] is None else float(calibration["voltage_kV"])
        self.semiangle_mrad = None if calibration["semiangle_mrad"] is None else float(calibration["semiangle_mrad"])
        self.title = title
        self.show_title = bool(ui["show_title"])
        self.show_controls = bool(ui["show_controls"])
        self.controls_collapsed = bool(ui["controls_collapsed"])
        self.show_stats = bool(ui["show_stats"])
        self.show_scale_bar = bool(ui["show_scale_bar"])
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
        self.bright_field_center_row_px = float(self.det_rows / 2) if bright_field_center_px is None else float(bright_field_center_px[0])
        self.bright_field_center_col_px = float(self.det_cols / 2) if bright_field_center_px is None else float(bright_field_center_px[1])
        self.bright_field_radius_px = min(self.det_rows, self.det_cols) * BRIGHT_FIELD_RADIUS_FRACTION if bright_field_radius_px is None else float(bright_field_radius_px)

        if browser_urls:
            if h5_urls:
                self._h5_urls = json.dumps(urls)
            else:
                self._qem_urls = json.dumps(urls)
            self.offline = True
            self.direct_enabled = False
            self._set_roi_to_disk()
            self.virtual_image_bytes = np.zeros(self.shape_rows * self.shape_cols, dtype=np.float32).tobytes()
            self.frame_bytes = np.zeros(self.det_rows * self.det_cols, dtype=np.float32).tobytes()
            if self._multiple_view_active():
                visible = min(self.n_frames, self.compare_max_panels)
                self.compare_status = f"Loading {visible}/{self.n_frames} browser WebGPU panels"
            self.compare_page_count = max(1, math.ceil(self.n_frames / self.compare_max_panels))
            if state is not None:  # the browser computes; the display state still applies
                self.load_state_dict(state)
            return

        if not waiting and bright_field_center_px is None and bright_field_radius_px is None:
            if recorded_disk is not None:
                (self.bright_field_center_row_px, self.bright_field_center_col_px), self.bright_field_radius_px = recorded_disk
            else:
                self.bright_field_center_row_px, self.bright_field_center_col_px, self.bright_field_radius_px = fit_disk(
                    self._session(0).mean_pattern(), (self.det_rows, self.det_cols), BRIGHT_FIELD_RADIUS_FRACTION
                )
        if precompute_virtual_images and self.n_frames == 1:
            self._precompute_presets()
        self.on_msg(self._handle_frontend_ready_msg)
        self.on_msg(self._handle_detector_request)
        self.on_msg(self._handle_detector_radius_request)
        self.on_msg(self._handle_position_request)
        self.observe(self._on_scan_position_change, names=["pos_row", "pos_col"])
        self.observe(self._on_roi_change, names=["roi_center_col", "roi_center_row", "roi_radius", "roi_radius_inner",
                                                 "roi_active", "roi_mode", "roi_width", "roi_height"])
        self.observe(self._on_roi_center_change, names=["roi_center"])
        self.observe(self._on_vi_source_change, names=["vi_source"])
        self.observe(self._on_calibration_change, names=["bright_field_center_row_px", "bright_field_center_col_px", "bright_field_radius_px"])
        # The first render shows the bright-field image: a point detector paints near black.
        self._set_roi_to_disk()
        self._compute_virtual_image_from_roi()
        self._refresh_compare_virtual_images()
        self._update_frame()
        self.observe(self._on_export_request_change, names=["export_request"])
        self.observe(self._on_direct_request_change, names=["direct_request"])
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
            self.load_state_dict(state)
        if self._verbose:
            print(f"Show4DSTEM {'x'.join(str(s) for s in shape)} on {self._device}")

    # ------------------------------------------------------------------
    # Public API used by the tutorials, the CLI and quantem.live
    # ------------------------------------------------------------------

    def export_html(
        self,
        path: str | pathlib.Path | None = None,
        *,
        title: str | None = None,
        dtype: str = "uint8",
        detector_bin: int = 1,
        scan_bin: int = 1,
        export_kind: str = "interactive",
        dataset_scope: str = "unhidden",
    ) -> pathlib.Path:
        """Write a standalone HTML viewer.

        ``export_kind="interactive"`` packs the counts (``dtype`` ``uint8`` or
        ``uint16``, mean-binned by ``detector_bin`` on the detector and ``scan_bin``
        on the scan) into a page whose detector ROI runs on browser WebGPU.
        Packed counts above what one page can load (2 GiB) are refused before
        any frame is read; the error names the least binning that fits.
        ``export_kind="report"`` writes PNG virtual-image pages for the datasets
        in ``dataset_scope`` (``unhidden``, ``current_page``, ``starred`` or
        ``all``) with no raw 4D payload.
        """
        if self._data is None and not (self._h5_urls or self._qem_urls):
            raise ValueError("Cannot export HTML after free(); rebuild the widget first.")
        dtype, detector_bin, scan_bin = html_export.export_options(self, dtype, detector_bin, scan_bin)
        kind = html_export.export_kind(export_kind)
        out = pathlib.Path(path) if path is not None else html_export.default_export_path(self, dtype, detector_bin, scan_bin, kind)
        self._write_html_export(out, title=title, dtype=dtype, detector_bin=detector_bin, scan_bin=scan_bin,
                                export_kind=kind, dataset_scope=dataset_scope)
        if kind == "report" or not (self._h5_urls or self._qem_urls):
            how = "self-contained HTML - double-click to open, no server needed"
        else:
            # A browser source reads its HDF5 or .qem files over HTTP (file:// blocks it), so a
            # double-click launcher serves the folder and opens the page.
            from quantem.widget.command_launcher import write_command_launcher

            write_command_launcher(out.parent, "Show4DSTEM", viewer_html=out.name)
            how = (f"WebGPU folder - reads its data files over HTTP; double-click Show4DSTEM.command in "
                   f"{out.parent.name}/ to open (a bare double-click of the HTML will not load the data)")
        size_mb = out.stat().st_size / (1024 * 1024)
        self.export_status = f"Exported {out.name} ({size_mb:.1f} MB, {html_export.mode_label(dtype, detector_bin, scan_bin, kind)}) - {how}"
        return out

    def _html_export_options(self, payload: dict, mode: str = "") -> dict:
        """The ``export_html`` keywords of a toolbar request, validated as a Python call is (``HtmlExportMixin`` hook)."""
        dtype, detector_bin, scan_bin = html_export.export_options(
            self, payload.get("dtype", "uint8"), payload.get("detector_bin", 1), payload.get("scan_bin", 1))
        return {"dtype": dtype, "detector_bin": detector_bin, "scan_bin": scan_bin,
                "export_kind": html_export.export_kind(payload.get("export_kind", "interactive")),
                "dataset_scope": str(payload.get("dataset_scope", "unhidden"))}

    def _default_html_export_path(self, *, dtype: str, detector_bin: int, scan_bin: int, export_kind: str,
                                  dataset_scope: str = "unhidden") -> pathlib.Path:
        """File in the kernel cwd that a toolbar export writes, named after the packing (``HtmlExportMixin`` hook)."""
        return html_export.default_export_path(self, dtype, detector_bin, scan_bin, export_kind)

    def _export_mode_label(self, *, dtype: str, detector_bin: int, scan_bin: int, export_kind: str,
                           dataset_scope: str = "unhidden") -> str:
        """Packing summary for the export status line (``HtmlExportMixin`` hook)."""
        return html_export.mode_label(dtype, detector_bin, scan_bin, export_kind)

    def _write_html_export(self, path, *, title: str | None = None, dtype: str, detector_bin: int, scan_bin: int,
                           export_kind: str, dataset_scope: str = "unhidden") -> pathlib.Path:
        """Write a report or an interactive page.

        Replaces the mixin's single export-clone path because Show4DSTEM has two
        export kinds, and a report needs no clone of the counts at all.
        """
        if export_kind == "report":
            return html_export.write_report_html(self, path, dtype=dtype, detector_bin=detector_bin, scan_bin=scan_bin,
                                                 dataset_scope=dataset_scope, title=title)
        return html_export.write_interactive_html(self, path, dtype=dtype, detector_bin=detector_bin, scan_bin=scan_bin, title=title)

    def direct_phase(self, *, set_source: bool = True, verbose: bool = False, **kwargs) -> np.ndarray:
        """Reconstruct the direct ptychography phase of the current dataset and show it as the image.

        The fit is quantem.gpu's default ``DirectPtycho.aberrations()``, model-based
        and tilt-corrected: the direct ptychography model fitted to the data together with the
        sample tilt and depth spread (C10 is then the defocus at mid-depth).
        ``direct_phase(tilt=False)`` fits the same model with the tilt held at 0.

        ``kwargs`` go to the solver: ``tilt`` (the ``direct_tilt`` trait,
        True by default), ``trials`` (the ``direct_trials`` trait, 200 by default),
        ``refinement`` (``"auto"``, the default, quantem.gpu's exact-derivative
        refinement, or None for the best trial without one), ``seed``,
        ``aberrations`` (a dict of ``C10_nm``, ``C12_nm`` and ``phi12_deg``; a
        missing key is 0), ``scan_detector_rotation_deg`` (the physical angle
        in [0, 360); omitted, the rotation the acquisition records, else the
        centre-of-mass estimate, as quantem.gpu.DirectPtycho chooses it),
        ``scan_detector_rotation_flipped``, ``bright_field_center_px`` and
        ``bright_field_radius_px`` (omitted, the constructor's, else the disk
        quantem.gpu detects), ``bright_field_intensity_threshold``
        (quantem.gpu's default when omitted), ``lock_aberrations``,
        ``lock_C10`` and ``lock_C12``. In the tilt-corrected fit the locks are
        quantem.gpu's ``fixed_aberrations``: C10 alone, C12 at 0 alone, or both,
        which hold all three coefficients and fit only the tilt and depth
        spread. ``lock_aberrations`` reconstructs the given coefficients with
        the sample tilt and depth spread of the last tilt-corrected fit (none
        before one, or with ``tilt=False``). The voltage,
        semiangle and sampling are the widget's calibration: what the dataset
        or file records, or the constructor keywords for an input that records
        none. The aligned DPC maps from the same data are attached beside the
        phase.
        """
        if self.direct_busy:
            raise RuntimeError("Direct ptycho is already running.")
        self.direct_busy = True
        self.direct_status = "Computing direct ptycho..."
        self.direct_calibration_json = ""
        self.direct_calibration_filename = ""
        try:
            phase, dpc_row, dpc_col = self._direct_phase_maps(verbose=verbose, **kwargs)
            self._set_vi_product_map("DPC_row", dpc_row)
            self._set_vi_product_map("DPC_col", dpc_col)
            self._set_vi_product_map("Direct", phase)
            if set_source:
                self.vi_source = "Direct"
            tilt = json.loads(self.direct_calibration_json or "{}").get("sample_tilt_mrad")
            sample = "" if tilt is None else f", tilt-corrected: ({tilt[0]:+.1f}, {tilt[1]:+.1f}) mrad"
            self.direct_status = f"Direct ptycho ready ({phase.shape[0]}x{phase.shape[1]}{sample})"
            return phase
        except (ValueError, RuntimeError, MemoryError) as exc:
            self.direct_status = f"Direct ptycho failed: {exc}"
            raise
        finally:
            self.direct_busy = False

    def apply_preset(self, name: str) -> Self:
        """Place the detector ROI on a named preset: ``bf``, ``abf``, ``adf`` or ``haadf``.

        Annuli are multiples of the bright-field radius (ADF 1 to 2, HAADF 2
        to 4). A ring ends at the detector edge nearest the bright-field
        centre, and one line says so; a preset whose ring would start beyond
        that edge is refused with one line and the ROI is left unchanged.
        """
        preset = str(name).strip().lower()
        if preset not in PRESET_RADII:
            raise ValueError(f"Unknown preset {name!r}. Choices: 'bf', 'abf', 'adf', 'haadf'.")
        bright_field_radius_px = self.bright_field_radius_px
        center = (self.bright_field_center_row_px, self.bright_field_center_col_px)
        radii = preset_radii(preset, (self.det_rows, self.det_cols), center, bright_field_radius_px)
        if radii is None:
            print(f"Show4DSTEM: {preset.upper()} starts {PRESET_RADII[preset][0] * max(1.0, bright_field_radius_px):.1f} px "
                  f"from the bright-field centre, but this detector reaches {detector_reach_px((self.det_rows, self.det_cols), center):.1f} px; "
                  "the detector ROI is unchanged. Choose ADF, or set roi_radius_inner and roi_radius.")
            return self
        if radii[1] < radii[2]:
            print(f"Show4DSTEM: {preset.upper()} outer radius is {radii[1]:.1f} px, the detector edge, not "
                  f"{radii[2]:.1f} px ({PRESET_RADII[preset][1]:g}x the bright-field radius): a larger ring is cut by the edges.")
        # Every ROI trait commits before one recompute; an intermediate state (mode
        # switched, radii stale) would paint a wrong image first.
        self._suppress_roi_recompute = True
        with self.hold_trait_notifications():
            self.vi_source = "roi"
            self.roi_active = True
            self.roi_mode = "circle" if preset == "bf" else "annular"
            self.roi_center_row = float(self.bright_field_center_row_px)
            self.roi_center_col = float(self.bright_field_center_col_px)
            if preset == "bf":
                self.roi_radius_inner = 0.0
                self.roi_radius = float(max(1.0, bright_field_radius_px))
            elif preset == "abf":
                self.roi_radius_inner = float(max(0.5, bright_field_radius_px * 0.5))
                self.roi_radius = float(max(1.0, bright_field_radius_px))
            else:
                # preset_radii: the multiples of the disk radius, the outer one ending at the detector edge
                self.roi_radius_inner, self.roi_radius = float(radii[0]), float(radii[1])
        self._suppress_roi_recompute = False
        if not self._multiple_view_active():
            self._compute_virtual_image_from_roi()
        self._refresh_compare_virtual_images()
        return self

    # The traits of the display state: scalars, then the lists state_dict copies.
    _STATE_KEYS = (
        "title", "show_title", "pos_row", "pos_col", "bright_field_center_row_px", "bright_field_center_col_px", "bright_field_radius_px", "roi_active", "roi_mode", "roi_center_row",
        "roi_center_col", "roi_radius", "roi_radius_inner", "roi_width", "roi_height", "vi_roi_mode",
        "vi_roi_center_row", "vi_roi_center_col", "vi_roi_radius", "vi_roi_width", "vi_roi_height",
        "vi_roi_reduce", "vi_source", "dp_colormap", "vi_colormap", "fft_colormap", "dp_scale_mode",
        "vi_scale_mode", "fft_scale_mode", "dp_vmin_pct", "dp_vmax_pct", "vi_vmin_pct", "vi_vmax_pct",
        "fft_vmin_pct", "fft_vmax_pct", "dp_vmin", "dp_vmax", "vi_vmin", "vi_vmax", "fft_auto", "show_fft",
        "fft_window", "show_controls", "controls_collapsed", "show_stats", "show_scale_bar", "panel_width_px",
        "dp_panel_width_px", "vi_panel_width_px", "sync_resize",
        "dp_show_colorbar", "vi_auto_contrast", "vi_smooth", "view_mode", "compare_columns",
        "compare_grid_width_px", "compare_max_panels", "compare_group_mode", "compare_page_idx",
        "compare_dp_mode", "profile_width", "frame_idx", "frame_dim_label", "frame_loop", "frame_fps",
        "frame_reverse", "frame_boomerang",
    )
    _LIST_STATE_KEYS = ("compare_panel_order", "compare_hidden_panels", "compare_starred_panels", "profile_line", "frame_labels")

    def state_dict(self) -> dict:
        """The display state (cursor, ROI, colormaps, layout); no pixel data."""
        state = {key: getattr(self, key) for key in self._STATE_KEYS}
        for key in self._LIST_STATE_KEYS:
            state[key] = list(getattr(self, key))
        return state

    def load_state_dict(self, state) -> None:
        """Restore a ``state_dict``, its envelope or a ``save`` file; positions and the dataset index are clamped to this data.

        A file of another ``schema`` or a key ``state_dict`` does not write
        raises before anything is applied.
        """
        state = read_state(state, "Show4DSTEM", (*self._STATE_KEYS, *self._LIST_STATE_KEYS))
        normalisers = {
            "view_mode": self._normalize_view_mode,
            "compare_dp_mode": self._normalize_compare_dp_mode,
            "compare_group_mode": self._normalize_compare_group_mode,
        }
        for key, value in state.items():
            if key in {"pos_row", "pos_col", "frame_idx"}:
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
        allocator caches of every device the data touched are flushed; an
        acquisition the widget opened from a path is closed. Call before loading a
        new dataset into the same kernel. The packed browser copy of the counts
        (up to the 2 GB offline budget) and the last export go too.
        """
        self._stop_stream()
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
        self._stop_stream()
        self._release_owned_acquisitions()
        with self._stream_lock:
            self._close_sessions()
            self._data = None
            self._preset_images.clear()
            super().close()

    def _stop_stream(self) -> None:
        if self._stream_follower is not None:
            self._stream_follower.stop()
            self._stream_follower = None

    def _publish_appended(self, added, labels, old_order, custom_order) -> None:
        """Show acquisitions appended to the series in one update; n_frames refreshes the comparison grid."""
        if added:
            with self.hold_trait_notifications():
                if hasattr(self, "_stream_stop"):
                    self._stream_source_ids = [*self._stream_source_ids, *self._new_stream_source_ids(len(added))]
                self.n_frames = len(self._data.frames)
                self.frame_labels = labels
                if getattr(self, "_stream_newest_first", False):
                    order = old_order if custom_order else list(reversed(range(added[0])))
                    self.compare_panel_order = [*reversed(added), *order]
                    self.compare_page_idx = 0
                elif custom_order:
                    self.compare_panel_order = [*old_order, *added]

    def append(self, acquisitions, labels=None) -> list[int]:
        """Add resident acquisitions (quantem.gpu datasets of the same scan and detector) to the comparison.

        Works on a viewer of several acquisitions (``Show4DSTEM([a, b])``, ``stream``). The
        caller keeps ownership: ``free`` does not close them. Returns the new dataset indices.
        """
        if not isinstance(self._data, (AcquisitionSeries, SessionSeries)):
            raise ValueError("append adds to a viewer of several acquisitions: open it with a list, "
                             "Show4DSTEM([first, ...]), or with Show4DSTEM.stream(recording).")
        acquisitions = list(acquisitions) if isinstance(acquisitions, (list, tuple)) else [acquisitions]
        # missing labels fall back to each dataset's own name
        labels = ([] if labels is None else list(labels)) + [None] * len(acquisitions)
        old_order = list(self.compare_panel_order)
        custom_order = sorted(old_order) == list(range(self.n_frames))
        names, added = list(self.frame_labels), []
        for acquisition, label in zip(acquisitions, labels):
            self._data.append(acquisition)
            added.append(len(self._data.frames) - 1)
            name = ((acquisition.metadata.get("name") or acquisition.metadata.get("series")) if isinstance(self._data, SessionSeries)
                    else dataset_name(acquisition))
            names.append(str(label or name or f"acquisition {added[-1]}"))
        self._publish_appended(added, names, old_order, custom_order)
        return added

    def refresh(self) -> None:
        """Refresh arriving counts while preserving the detector, cursor and zoom.

        Use with a mutable prepared session. Completed acquisitions need no
        refresh; changing a detector already recomputes their images.
        """
        if self._data is None:
            return
        self._preset_images.clear()
        with self.hold_sync():
            if not self._multiple_view_active():
                self._compute_virtual_image_from_roi()
            self._refresh_compare_virtual_images()
            self._update_frame()

    @classmethod
    def stream(cls, recording, *, refresh_s: float = 1 / 60, labels=None, first_timeout_s: float = 0.0,
               newest_first: bool = True,
               **viewer_kwargs) -> Self:
        """Show arriving scans from a GPU recording, before a whole scan completes.

        The recording retains ANS counts. Missing positions are NaN, not measured
        zeros. Detector changes query all arrived positions. ``refresh_s`` limits
        display updates only (at most 60 updates/s by default); no incoming
        patterns are dropped. At most two display updates await browser
        acknowledgment; later previews coalesce. During blocking user
        cells, previews are limited to one every two seconds; interactions wait
        until the cell returns. Background image
        transfers coalesce during dragging so cursor requests stay responsive. Older recordings
        that expose only completed ``datasets`` still append on completion.

        Returns a waiting viewer immediately when the receiver is idle. It does
        not arm or trigger acquisition. Pass ``first_timeout_s=120`` to wait for
        the first pattern before returning, as in earlier versions.
        New scans appear at the top left, three per row (``compare_columns``); the selected scan stays selected.
        The diffraction pattern and the virtual image resize independently (``sync_resize=False``):
        drag either corner, or pass ``dp_size``/``image_size``.
        Set ``newest_first=False`` for chronological display. Data and dataset
        indices always retain acquisition order.

        Examples
        --------
        >>> viewer = Show4DSTEM.stream(recording)  # doctest: +SKIP
        """
        if not math.isfinite(refresh_s) or refresh_s <= 0:
            raise ValueError("refresh_s must be finite and positive.")
        if not math.isfinite(first_timeout_s) or first_timeout_s < 0:
            raise ValueError("first_timeout_s must be finite and nonnegative.")
        viewer_kwargs.setdefault("compare_columns", 3)   # a steady grid while scans arrive; 0 chooses automatically
        viewer_kwargs.setdefault("sync_resize", False)   # the pattern and the image each keep the size dragged to
        partial = hasattr(recording, "acquisitions")
        lock = getattr(recording, "lock", nullcontext())

        def snapshot():
            return list(recording.acquisitions if partial else recording.datasets)

        pause, waited = threading.Event(), 0.0
        labels = None if labels is None else list(labels)
        while True:
            with lock:
                sources = snapshot()
                if sources and (not partial or sources[0].received):
                    if labels:
                        first_label = labels[0]
                    elif partial:
                        first_label = (sources[0].metadata.get("name")
                                       or sources[0].metadata.get("series") or "acquisition 0")
                    else:
                        first_label = dataset_name(sources[0]) or "acquisition 0"
                    viewer = cls([sources[0]], frame_labels=[str(first_label)], **viewer_kwargs)
                    break
            if recording.error is not None:
                raise recording.error
            if first_timeout_s == 0 and recording.receiving:
                viewer = cls(None, **viewer_kwargs)
                viewer._replace_stream_sources([], [])
                viewer.title = "Waiting for acquisition"
                sources = []
                break
            if waited >= first_timeout_s or not recording.receiving:
                raise TimeoutError(f"No acquisition arrived within {waited:.0f} s.")
            pause.wait(0.01)
            waited += 0.01
        viewer._stream_newest_first = newest_first
        if partial:
            viewer.pos_row = viewer.pos_col = 0
        viewer._stream_stop = threading.Event()
        viewer._stream_source_ids = viewer._new_stream_source_ids(len(sources[:1]))
        viewer._stream_selected_source = viewer._stream_source_ids[0] if sources else ""
        viewer._stream_thread = None
        viewer.stream_error = None
        viewer._stream_follower = StreamFollower(
            viewer, recording, sources, labels=labels, refresh_s=refresh_s, options=viewer_kwargs)
        try:
            viewer._stream_follower.start()
        except Exception:
            viewer.close()
            raise
        return viewer

    def refresh_stream(self, scan_id: str | None = None) -> None:
        """Refresh a stream after unloading scans, optionally selecting a stable scan ID.

        Examples
        --------
        >>> recording.unload(scan_id)
        >>> viewer.refresh_stream()
        """
        with self._stream_lock:
            update = getattr(self, "_stream_update", None)
            if update is None:
                raise ValueError("This viewer is not streaming; use Show4DSTEM.stream(recording).")
            update(force=True)
            if scan_id is not None:
                with self._stream_recording.lock:
                    sources = list(self._stream_recording.acquisitions)
                    index = next((i for i, source in enumerate(sources) if source.identifier == scan_id), None)
                    if index is None:
                        raise ValueError("This scan is no longer available; refresh the scan list.")
                    self.frame_idx = index

    def _initialize_stream_geometry(self, source, options) -> None:
        """Adopt the first real scan's geometry and calibration after an idle start."""
        self._device = torch.device(self._data.device)
        self.shape_rows, self.shape_cols, self.det_rows, self.det_cols = self._data.shape[-4:]
        self.pos_row = self.pos_col = 0
        self.vi_roi_center_row, self.vi_roi_center_col = self.shape_rows / 2, self.shape_cols / 2
        self.vi_roi_center = [self.vi_roi_center_row, self.vi_roi_center_col]
        roi_size = max(3, min(self.shape_rows, self.shape_cols) * DEFAULT_VI_ROI_RATIO)
        self.vi_roi_radius, self.vi_roi_width, self.vi_roi_height = roi_size, roi_size * 2, roi_size
        recorded = (acquisition_calibration(source.sampling, source.units, source.metadata)
                    if gpu_adapter.is_acquisition(source) else source.metadata)
        calibration = {name: stated(name, options.get(name), recorded.get(name)) for name in CALIBRATION_KEYS}
        if calibration["detector_sampling_mrad"] is not None and calibration["detector_sampling_inv_A"] is not None:
            raise ValueError("The input and viewer specify different detector units; leave the calibration keyword out.")
        for name, value in calibration.items():
            if name in {"scan_sampling_A", "detector_sampling_mrad", "detector_sampling_inv_A"}:
                value = [] if value is None or None in per_axis(value, 2) else list(self._positive_pair(value, name))
            setattr(self, name, value)
        center = self._given_bright_field["bright_field_center_px"]
        radius = self._given_bright_field["bright_field_radius_px"]
        if center is None and radius is None:
            disk = gpu_adapter.recorded_bright_field_disk(source.metadata)
            if disk is not None:
                center, radius = disk
            else:
                row, col, radius = fit_disk(self._session(0).mean_pattern(),
                                            (self.det_rows, self.det_cols), BRIGHT_FIELD_RADIUS_FRACTION)
                center = (row, col)
        self.bright_field_center_row_px, self.bright_field_center_col_px = (
            center if center is not None else (self.det_rows / 2, self.det_cols / 2))
        self.bright_field_radius_px = (radius if radius is not None else
                                      min(self.det_rows, self.det_cols) * BRIGHT_FIELD_RADIUS_FRACTION)
        self._set_roi_to_disk()

    @contextmanager
    def _defer_stream_observers(self):
        """Prepare one source topology, then let the caller refresh it once."""
        previous = self._stream_defer_compute
        self._stream_defer_compute = True
        try:
            yield
        finally:
            self._stream_defer_compute = previous

    def _new_stream_source_ids(self, count: int) -> list[str]:
        """Allocate viewer-local identities that never refer to a replacement scan."""
        start = self._stream_source_serial
        self._stream_source_serial += count
        return [str(index) for index in range(start + 1, start + count + 1)]

    def _replace_stream_sources(self, previous, sources, labels=None) -> None:
        """Retire old live panels without resetting the detector or surviving query sessions."""
        with self._compare_lock, self._session_lock:
            self._stream_completed_sources.clear()
            if not sources:
                self._close_sessions()
                self._data = None
                self._preset_images.clear()
                with self.hold_sync():
                    self.n_frames = 0
                    self._stream_source_ids = []
                    self._stream_selected_source = ""
                    self.frame_labels = []
                    self.frame_bytes = np.full((self.det_rows, self.det_cols), np.nan, np.float32).tobytes()
                    self.virtual_image_bytes = np.full((self.shape_rows, self.shape_cols), np.nan, np.float32).tobytes()
                    self.compare_diffraction_bytes = b""
                    self._clear_compare_virtual_images("No retained scans. Waiting for another acquisition.")
                    self.title = "No retained scans"
                return
            if not isinstance(self._data, SessionSeries):
                self._close_sessions()
                sessions = {}
            else:
                sessions = dict(zip(previous, self._data.frames))
            selected = previous[min(int(self.frame_idx), len(previous) - 1)] if previous else None
            names = dict(zip(previous, self.frame_labels))
            identities = dict(zip(previous, self._stream_source_ids))
            hidden = [previous[i] for i in self.compare_hidden_panels if i < len(previous)]
            starred = [previous[i] for i in self.compare_starred_panels if i < len(previous)]
            data = (AcquisitionSeries(sources) if gpu_adapter.is_acquisition(sources[0]) else
                    SessionSeries([sessions.get(source, source) for source in sources]))
            if self._compare_session is not None:
                self._compare_session.close()
                self._compare_session = None
                self._compare_session_indices = ()
            for source, session in sessions.items():
                if source not in sources:
                    session.close()
            self._sessions = {}
            self._data = data
            self._preset_images.clear()
            with self.hold_sync(), self.hold_trait_notifications():
                self._stream_source_ids = [identities[source] if source in identities else
                                           self._new_stream_source_ids(1)[0] for source in sources]
                self.n_frames = len(sources)
                self.frame_idx = sources.index(selected) if selected in sources else 0
                self._stream_selected_source = self._stream_source_ids[self.frame_idx]
                self.frame_labels = [str(labels[i]) if labels and i < len(labels) else
                                     names.get(source, str(source.metadata.get("name") or
                                                           source.metadata.get("series", f"acquisition {i}")))
                                     for i, source in enumerate(sources)]
                self.compare_panel_order = (list(reversed(range(len(sources))))
                                            if getattr(self, "_stream_newest_first", False)
                                            else list(range(len(sources))))
                self.compare_hidden_panels = [i for i, source in enumerate(sources) if source in hidden]
                self.compare_starred_panels = [i for i, source in enumerate(sources) if source in starred]
                self.compare_page_idx = 0

    def _release_owned_acquisitions(self) -> None:
        """Close the acquisition this widget loaded itself from a path; acquisitions the caller passed stay open."""
        for acquisition in self._owned_acquisitions:
            acquisition.close()
        self._owned_acquisitions = []

    # ------------------------------------------------------------------
    # Detector sessions and reductions
    # ------------------------------------------------------------------

    def _frame_source(self, frame_idx: int):
        """The 4D source of one dataset: a bounded view or a tensor."""
        if isinstance(self._data, (AcquisitionSeries, SessionSeries)):
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
            self._sessions[frame_idx] = (source if isinstance(source, DetectorSessionSource)
                                         else DenseSession(source) if isinstance(source, torch.Tensor)
                                         else gpu_adapter.prepare_session(source))
        return self._sessions[frame_idx]

    def _close_sessions(self) -> None:
        """Close every detector session: sessions hold views of the data, so the memory is released only after this."""
        with self._compare_lock, self._session_lock:
            self._stream_completed_sources.clear()
            if self._compare_session is not None:
                self._compare_session.close()
                self._compare_session = None
                self._compare_session_indices = ()
            sessions = list(self._sessions.values())
            if isinstance(self._data, SessionSeries):
                sessions.extend(self._data.frames)
            for session in {id(session): session for session in sessions}.values():
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
        with self._session_lock:
            return np.asarray(self._compute.frame(row * self.shape_cols + col), dtype=np.float32)

    def virtual_image(self) -> np.ndarray:
        """The image panel as float32: the product map in view, else the detector ROI sum."""
        product = self._vi_product_maps.get(self.vi_source)
        if product is not None:
            frame = 0 if product.shape[0] == 1 else max(0, min(int(self.frame_idx), product.shape[0] - 1))
            return np.array(product[frame], dtype=np.float32)
        if self._data is not None and self._multiple_view_active():
            # Multiple view transports the panel stack, not a duplicate hidden
            # image. An explicit Python query still returns current counts.
            return np.array(self._compute.masked_sum(self._current_detector_mask()), dtype=np.float32)
        if len(self.virtual_image_bytes) != self.shape_rows * self.shape_cols * 4:
            return np.zeros((self.shape_rows, self.shape_cols), dtype=np.float32)
        return np.frombuffer(self.virtual_image_bytes, dtype=np.float32).reshape(self.shape_rows, self.shape_cols).copy()

    def vi_roi_pattern(self) -> np.ndarray:
        """The pattern reduced over the scan ROI, as the diffraction panel shows it."""
        self._compute_vi_roi_dp()
        if len(self.vi_roi_dp_bytes) != self.det_rows * self.det_cols * 4:
            return self.pattern(self.pos_row, self.pos_col)
        return np.frombuffer(self.vi_roi_dp_bytes, dtype=np.float32).reshape(self.det_rows, self.det_cols).copy()

    def _frame_array(self, frame_idx: int, detector_bin: int) -> np.ndarray:
        """One dataset as a host float32 array, mean-binned on the detector (counts summed exactly).

        Tensors come back in scan-row chunks and bounded views in 32 MiB scan
        windows, so an export never materialises a dense cube on the GPU.
        """
        source = self._frame_source(frame_idx)
        det_rows, det_cols = self.det_rows // detector_bin, self.det_cols // detector_bin
        out = np.empty((self.shape_rows, self.shape_cols, det_rows, det_cols), np.float32)
        if isinstance(source, AcquisitionView):
            window_cols = max(1, min(self.shape_cols, (32 << 20) // (self.det_rows * self.det_cols * 4)))
            for row in range(self.shape_rows):
                for col in range(0, self.shape_cols, window_cols):
                    stop = min(col + window_cols, self.shape_cols)
                    out[row : row + 1, col:stop] = detector_bin_mean(source.read(scan_region=(row, row + 1, col, stop)), detector_bin)
            return out
        chunk_rows = max(1, (600 << 20) // max(1, self.shape_cols * self.det_rows * self.det_cols * 4))
        for row in range(0, self.shape_rows, chunk_rows):
            out[row : row + chunk_rows] = detector_bin_mean(source[row : row + chunk_rows], detector_bin)
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
        # Keep the result buffer alive through the comm without copying a
        # full image into Python bytes. Each query owns a fresh result.
        image_bytes = memoryview(np.ascontiguousarray(image, dtype=np.float32)).cast("B")
        if preset is not None:
            # A preset image computed once is kept as a precomputed one is (and
            # dropped with them), so clicking back to that preset sends it at once.
            self._preset_images[preset] = image_bytes
        self.virtual_image_bytes = image_bytes

    def _precompute_presets(self) -> None:
        """Cache the four preset images so a preset click paints without a reduction."""
        center = (self.bright_field_center_row_px, self.bright_field_center_col_px)
        masks = {name: preset_mask(name, (self.det_rows, self.det_cols), center, self.bright_field_radius_px) for name in PRESET_RADII}
        self._preset_images = {
            name: np.ascontiguousarray(self._compute.masked_sum(mask), dtype=np.float32).tobytes()
            for name, mask in masks.items() if mask is not None  # a preset the detector cannot hold has no image
        }

    def _preset_name(self, mask: np.ndarray) -> str | None:
        """The preset whose detector mask equals ``mask`` pixel for pixel, if any.

        A cached preset image is the masked sum of that exact mask, so it may only
        stand in for an ROI that selects the same detector pixels; a radius within a
        pixel of the preset can still select a different set and a different image.
        """
        center = (self.bright_field_center_row_px, self.bright_field_center_col_px)
        for name in PRESET_RADII:
            preset = preset_mask(name, (self.det_rows, self.det_cols), center, self.bright_field_radius_px)
            if preset is not None and np.array_equal(mask, preset):
                return name
        return None

    def _joint_session(self, indices: Sequence[int]):
        """Reuse one batched query session for the visible compatible acquisitions."""
        selected = tuple(int(idx) for idx in indices)
        sources = ([self._data.frames[idx].source for idx in selected]
                   if isinstance(self._data, AcquisitionSeries) else
                   [self._stream_completed_sources[idx] for idx in selected])
        if gpu_adapter.can_query_together(sources):
            if selected != self._compare_session_indices:
                if self._compare_session is not None:
                    self._compare_session.close()
                self._compare_session = None
                self._compare_session_indices = ()
                self._compare_session = gpu_adapter.prepare_session(sources)
                self._compare_session_indices = selected
            return self._compare_session
        return None

    def _compare_query(self, indices: Sequence[int], method: str, *args) -> np.ndarray:
        """Batch immutable counts, querying any arriving scans in their original panel order."""
        with self._compare_lock:
            completed = (list(indices) if isinstance(self._data, AcquisitionSeries) else
                         [idx for idx in indices if idx in self._stream_completed_sources])
            session = self._joint_session(completed) if completed else None
            if session is None:
                return np.stack([np.asarray(getattr(self._session(idx), method)(*args), dtype=np.float32)
                                 for idx in indices])
            results = np.asarray(getattr(session, method)(*args), dtype=np.float32)
            if len(completed) == len(indices):
                return results
            ready = dict(zip(completed, results))
            return np.stack([ready[idx] if idx in ready else
                             np.asarray(getattr(self._session(idx), method)(*args), dtype=np.float32)
                             for idx in indices])

    def _compare_frames(self, indices: Sequence[int], flat: int) -> np.ndarray:
        """Read one position across visible acquisitions in a single supported query."""
        return self._compare_query(indices, "frame", flat)

    def _compare_images(self, indices: Sequence[int], mask: np.ndarray) -> np.ndarray:
        """Virtual images of several datasets under one mask, per detector pixel.

        Panels are previews across datasets, so each is divided by the mask area:
        switching from BF to ADF then does not flatten them only because the mask
        covers more pixels. The single-view image stays a plain sum.
        """
        area = max(1.0, float(np.count_nonzero(mask)))
        images = np.ascontiguousarray(self._compare_query(indices, "masked_sum", mask))
        images /= area
        return images

    def selected_scan_region(self) -> tuple[int, int, int, int]:
        """Return the scan rectangle selected in the virtual image.

        Choose **Rect** in the virtual-image ROI controls, then call this
        method to use the same integer scan positions in an analysis. With
        the ROI off, the complete displayed scan is returned. Detector ROI
        controls do not affect this selection.

        Returns
        -------
        tuple[int, int, int, int]
            ``(row_start, row_stop, col_start, col_stop)`` with exclusive stop
            indices, clipped to the displayed scan. Integer pixel centers on
            the rectangle boundary are included, matching the viewer's ROI.
            Coordinates belong to the input shown by this viewer: a cropped
            input starts at zero and a strided preview uses preview indices.
            No acquisition crop origin or stride is added automatically.

        Raises
        ------
        ValueError
            If a nonrectangular ROI is active or the rectangle contains no
            scan positions. Select **Rect**, or turn the ROI off.

        Examples
        --------
        After selecting a rectangle in a live widget:

        >>> row_start, row_stop, col_start, col_stop = viewer.selected_scan_region()
        >>> region = data[row_start:row_stop, col_start:col_stop]
        """
        if self.vi_roi_mode == "off":
            return (0, self.shape_rows, 0, self.shape_cols)
        if self.vi_roi_mode != "rect":
            raise ValueError(
                f"The active scan ROI is {self.vi_roi_mode!r}. Select Rect "
                "for rectangular analysis bounds, or turn the scan ROI off "
                "to use the full displayed scan."
            )
        row_start = max(0, math.ceil(self.vi_roi_center_row - self.vi_roi_height / 2))
        row_stop = min(self.shape_rows, math.floor(self.vi_roi_center_row + self.vi_roi_height / 2) + 1)
        col_start = max(0, math.ceil(self.vi_roi_center_col - self.vi_roi_width / 2))
        col_stop = min(self.shape_cols, math.floor(self.vi_roi_center_col + self.vi_roi_width / 2) + 1)
        if row_start >= row_stop or col_start >= col_stop:
            raise ValueError(
                "The scan rectangle contains no displayed scan positions. "
                "Move or enlarge Rect inside the virtual image."
            )
        return (row_start, row_stop, col_start, col_stop)

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
        frames = None
        with self._compare_lock, self._session_lock:
            if self._multiple_view_active() and self.compare_dp_mode == "average":
                page = self._page_indices() or [int(self.frame_idx)]
                frames = self._compare_frames(page, flat)
                # A live scan returns an all-NaN pattern until this position
                # arrives. Average acquired patterns; retain NaN if none exist.
                acquired = [frame for frame in frames if not np.isnan(frame).all()]
                frame = (np.mean(acquired, axis=0) if acquired else
                         np.full((self.det_rows, self.det_cols), np.nan, np.float32))
            elif (self._multiple_view_active() and self.compare_dp_mode == "all"
                  and self.vi_roi_mode == "off" and self.compare_panel_indices):
                panels = list(self.compare_panel_indices)
                frames = self._compare_frames(panels, flat)
                frame = (frames[panels.index(self.frame_idx)] if self.frame_idx in panels else
                         np.asarray(self._compute.frame(flat), dtype=np.float32))
            else:
                frame = np.asarray(self._compute.frame(flat), dtype=np.float32)
        self.frame_bytes = frame.tobytes()
        if self.vi_roi_mode != "off":
            self._compute_vi_roi_dp()
        else:
            self._publish_all_diffraction(frames=frames)

    def _publish_all_diffraction(self, indices=None, *, frames=None) -> None:
        """In ``compare_dp_mode="all"`` send every visible dataset's pattern for the one selection."""
        if self._multiple_view_active() and self.compare_dp_mode == "all":
            panels = list(self.compare_panel_indices)
            flat = self.pos_row * self.shape_cols + self.pos_col
            if panels:
                if frames is None:
                    frames = (self._compare_frames(panels, flat) if indices is None else
                              np.stack([self._session(idx).reduce_frames(indices, self.vi_roi_reduce) for idx in panels]))
                with self.hold_sync():
                    self.compare_diffraction_indices = panels
                    self.compare_diffraction_bytes = np.ascontiguousarray(frames, dtype=np.float32).tobytes()
                return
        self._clear_all_diffraction()

    def _clear_all_diffraction(self) -> None:
        """Empty the per-dataset patterns of ``compare_dp_mode="all"`` in one frontend message."""
        with self.hold_sync():
            self.compare_diffraction_bytes = b""
            self.compare_diffraction_indices = []

    # ------------------------------------------------------------------
    # Product maps (DPC and direct ptychography phase)
    # ------------------------------------------------------------------

    def _set_vi_product_map(self, label: str, value) -> None:
        """Attach or replace one static map (2D, or 3D with one frame per dataset) and resend the stack."""
        if label not in PRODUCT_LABELS:
            raise ValueError(f"Unsupported virtual-image product {label!r}. Use one of {PRODUCT_LABELS}.")
        product = np.asarray(value)
        if np.iscomplexobj(product):
            raise ValueError(f"{label} must be a real-valued map. Pass the direct ptychography phase map rather than a complex reconstruction.")
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
        source = self.vi_source
        if source != "roi" and source not in self._vi_product_maps:
            source = "roi"
        if source != self.vi_source:
            self.vi_source = source  # re-enters this observer with a valid source
            return
        if source == "roi" and not self._multiple_view_active():
            self._compute_virtual_image_from_roi()
        self._refresh_compare_virtual_images()

    # ------------------------------------------------------------------
    # Direct ptychography
    # ------------------------------------------------------------------

    def _direct_source(self):
        """The current dataset in the form quantem.gpu direct ptychography reads without a dense copy."""
        frame = self._frame_source(int(self.frame_idx))
        if not isinstance(frame, AcquisitionView):
            return frame
        if frame.region == (0, frame.source.shape[0], 0, frame.source.shape[1]):
            return frame.source
        return frame.read(scan_region=(0, frame.shape[0], 0, frame.shape[1]))

    @staticmethod
    def _positive_pair(value, name: str) -> tuple[float, float]:
        """A positive ``(row, col)`` pair from a scalar or a length-2 sequence; zero or negative sampling has no direct ptychography geometry."""
        pair = tuple(float(component) for component in per_axis(value, 2))
        if len(pair) != 2 or pair[0] <= 0 or pair[1] <= 0:
            raise ValueError(f"{name} must be a positive scalar or length-2 pair, got {value!r}.")
        return pair

    def _direct_phase_maps(
        self,
        *,
        bright_field_radius_px: float | None = None,
        bright_field_center_px: tuple[float, float] | None = None,
        aberrations: dict[str, float] | None = None,
        scan_detector_rotation_deg: float | None = None,
        scan_detector_rotation_flipped: bool | None = None,
        bright_field_intensity_threshold: float | None = None,
        trials: int | None = None,
        tilt: bool | object = FROM_TRAIT,
        refinement: str | None | object = FROM_TRAIT,
        lock_aberrations: bool = False,
        lock_C10: bool = False,
        lock_C12: bool = False,
        seed: int = 42,
        verbose: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run quantem.gpu direct ptychography on the current dataset; returns ``(phase, dpc_row, dpc_col)``.

        The DPC centre-of-mass maps are quantem.gpu's aligned fields
        (``dpc.run``) at the solved physical rotation and flip. Without
        ``scan_detector_rotation_deg`` quantem.gpu's rule applies: the
        rotation the acquisition records, else the centre-of-mass estimate,
        with the 180-degree choice settled by the fit.
        """
        if self.n_frames != 1:
            raise ValueError("Direct ptycho runs on a single 4D dataset. For a series, open the target dataset as a 4D widget.")
        # the calibration the dataset or file records (or, for an input that records none, the constructor keywords)
        if not self.scan_sampling_A:
            raise ValueError("Direct ptycho needs the scan step in angstrom and the data record none. For an array or an "
                             "uncalibrated file, construct Show4DSTEM(data, scan_sampling_A=(row_A, col_A)).")
        if self.voltage_kV is None:
            raise ValueError("Direct ptycho needs the accelerating voltage and the data record none. For an array or an "
                             "uncalibrated file, construct Show4DSTEM(data, voltage_kV=300).")
        detector_sampling_mrad = tuple(self.detector_sampling_mrad) or None
        if detector_sampling_mrad is None and self.detector_sampling_inv_A:
            # Direct ptychography takes the detector step in mrad; a step recorded in reciprocal angstrom converts at this voltage
            detector_sampling_mrad = gpu_adapter.detector_sampling_mrad_from_inv_A(self.detector_sampling_inv_A, self.voltage_kV)
        semiangle = self.semiangle_mrad
        if semiangle is None:
            if detector_sampling_mrad is None:
                raise ValueError("Direct ptycho needs the convergence semiangle and the data record none. Construct "
                                 "Show4DSTEM(data, semiangle_mrad=...), or give the detector step in mrad per pixel "
                                 "(detector_sampling_mrad=) so it follows from the bright-field disk edge.")
            # quantem.gpu's rule: the disk edge sits at the semiangle. The equal-area edge radius
            # of an elliptical disk takes the geometric mean of the two detector steps.
            edge_radius_px = gpu_adapter.disk_edge_radius(self._session(int(self.frame_idx)).mean_pattern())
            semiangle = edge_radius_px * math.sqrt(detector_sampling_mrad[0] * detector_sampling_mrad[1])
            print(f"Show4DSTEM: semiangle_mrad is not recorded; direct ptycho uses {semiangle:.3f} mrad, the bright-field "
                  f"disk edge ({edge_radius_px:.2f} px, quantem.gpu.detector.disk_edge_radius) times the detector step. "
                  "Pass Show4DSTEM(data, semiangle_mrad=...) to set it.")
        if semiangle <= 0:
            raise ValueError(f"semiangle_mrad must be positive, got {semiangle}.")
        trials = int(self.direct_trials if trials is None else trials)
        if trials < 0:
            raise ValueError(f"trials must be >= 0, got {trials}.")
        tilt = bool(self.direct_tilt if tilt is FROM_TRAIT else tilt)
        refinement = self.direct_refinement if refinement is FROM_TRAIT else refinement
        if refinement not in ("auto", None):
            raise ValueError(f"Direct ptycho refinement must be 'auto' or None, got {refinement!r}.")
        do_refine = refinement is not None
        fit_options: dict = {}
        if lock_C10 or lock_C12:
            held = {name: float(value) for name, value in {
                "C10_nm": self.direct_C10_nm, "C12_nm": self.direct_C12_nm, "phi12_deg": self.direct_phi12_deg,
                **(aberrations or {})}.items()}
            if tilt:
                # quantem.gpu's joint fit holds C10 alone, C12 = 0 alone, or all three (then only tilt and depth are fitted)
                if lock_C12 and not lock_C10 and held["C12_nm"] != 0.0:
                    raise ValueError(
                        f"The tilt-corrected fit holds C12 alone only at 0 (no astigmatism); it is {held['C12_nm']:g} nm. "
                        "Lock C10 as well to hold all three coefficients, or fit with tilt=False.")
                fixed = {"C10_nm": held["C10_nm"]} if lock_C10 else {}
                if lock_C12:
                    fixed.update({"C12_nm": held["C12_nm"], "phi12_deg": held["phi12_deg"]} if lock_C10 else {"C12_nm": 0.0})
                fit_options["fixed_aberrations"] = fixed
                if len(fixed) == 3:
                    fit_options["check_rotation"] = False  # the optics are fixed: there is no 180-degree branch to choose
            else:
                # A scalar pins a coefficient in the trial search and the lock holds it through the
                # refinement; locking C12 pins phi12 too (magnitude and angle pair).
                fit_options["search_ranges"] = {
                    "C10_nm": held["C10_nm"] if lock_C10 else (-40.0, 40.0),
                    "C12_nm": held["C12_nm"] if lock_C12 else (0.0, 10.0),
                    "phi12_deg": held["phi12_deg"] if lock_C12 else (-90.0, 90.0),
                }
                locked = (["C10_nm"] if lock_C10 else []) + (["C12_nm", "phi12_deg"] if lock_C12 else [])
                fit_options["refine_lock"] = locked if do_refine else None
        # omitted, the disk given to the constructor; else quantem.gpu detects it
        if bright_field_radius_px is None:
            bright_field_radius_px = self._given_bright_field["bright_field_radius_px"]
        if bright_field_center_px is None:
            bright_field_center_px = self._given_bright_field["bright_field_center_px"]
        self.direct_status = "Preparing direct ptycho data..."
        source = self._direct_source()
        manual = bool(lock_aberrations and aberrations is not None)
        with gpu_adapter.DirectSession.open(
            source,
            voltage_kV=self.voltage_kV,
            semiangle_mrad=float(semiangle),
            scan_sampling_A=tuple(self.scan_sampling_A),
            detector_sampling_mrad=detector_sampling_mrad,
            scan_shape=(self.shape_rows, self.shape_cols) if source.ndim == 3 else None,
            # a missing coefficient is 0, as direct_phase documents; quantem.gpu takes all three and refuses any other key
            aberrations=None if aberrations is None else {"C10_nm": 0.0, "C12_nm": 0.0, "phi12_deg": 0.0, **aberrations},
            scan_detector_rotation_deg=None if scan_detector_rotation_deg is None else float(scan_detector_rotation_deg),
            scan_detector_rotation_flipped=scan_detector_rotation_flipped,
            # omitted, quantem.gpu's own default threshold applies
            **({} if bright_field_intensity_threshold is None
               else {"bright_field_intensity_threshold": float(bright_field_intensity_threshold)}),
            bright_field_radius_px=None if bright_field_radius_px is None else float(bright_field_radius_px),
            bright_field_center_px=None if bright_field_center_px is None else tuple(float(value) for value in bright_field_center_px),
        ) as direct:
            num_bright_field_pixels = direct.num_bright_field_pixels
            bright_field_intensity_threshold = direct.bright_field_intensity_threshold
            # the values the session used, also when it detected or estimated them
            bright_field_radius_px, detector_sampling_mrad = direct.bright_field_radius_px, direct.detector_sampling_mrad
            scan_detector_rotation_flipped = direct.scan_detector_rotation_flipped
            self.direct_bright_field_pixels = num_bright_field_pixels
            if tilt and self._direct_sample is not None:
                # a manual recompute, or a refinement without trials, starts from the last tilt-corrected fit's sample
                direct.direct.sample_tilt_mrad = tuple(self._direct_sample["sample_tilt_mrad"])
                direct.direct.depth_spread_nm = float(self._direct_sample["depth_spread_nm"])
            if manual or (trials == 0 and not do_refine):
                self.direct_status = f"Reconstructing direct ptycho with the given coefficients ({num_bright_field_pixels} BF pixels)..."
                result = direct.run()
            else:
                method = "with sample tilt" if tilt else "at tilt 0"
                self.direct_status = (f"Fitting aberrations {method} ({trials} trials"
                                           f"{', refined' if do_refine else ''}, {num_bright_field_pixels} BF pixels)...")
                search = "standard" if tilt and not fit_options and trials > 0 and refinement is not None else "primary"
                result = direct.aberrations(search=search,
                                          tilt=tilt, trials=trials, refinement=refinement,
                                          seed=int(seed), verbose=verbose, show=False, **fit_options)
                self._direct_sample = None if result.get("sample_tilt_mrad") is None else {
                    "sample_tilt_mrad": result["sample_tilt_mrad"], "depth_spread_nm": result["depth_spread_nm"]}
            # quantem.gpu centres a scan its FFT cannot take in a padded grid; keep the measured positions
            phase = result["phase"]
            if tuple(phase.shape) != (self.shape_rows, self.shape_cols):
                phase = direct.crop_native(phase, (0, self.shape_rows, 0, self.shape_cols), 1)
        phase = np.ascontiguousarray(torch.as_tensor(phase).cpu().numpy(), dtype=np.float32)
        if phase.shape != (self.shape_rows, self.shape_cols):
            raise ValueError(f"Direct ptychography result shape {phase.shape} does not match the widget scan shape {self.shape_rows}x{self.shape_cols}.")
        self.direct_C10_nm = result["aberrations"]["C10_nm"]
        self.direct_C12_nm = result["aberrations"]["C12_nm"]
        self.direct_phi12_deg = result["aberrations"]["phi12_deg"]
        # the physical angle: a manual recompute sends it back whole, where the folded angle alone lost the reversal
        self.direct_scan_detector_rotation_deg = result["scan_detector_rotation_deg"]
        # a manual recompute sends the angle back; the flip of the same fit goes with it
        self._direct_flipped = scan_detector_rotation_flipped
        self.direct_calibration_json = json.dumps({
            "schema": DIRECT_CALIBRATION_SCHEMA,
            "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "source": {"widget": "Show4DSTEM", "title": self.title, "acquisition_index": int(self.frame_idx),
                       "scan_shape": [self.shape_rows, self.shape_cols], "detector_shape": [self.det_rows, self.det_cols]},
            "aberrations": result["aberrations"],
            "scan_detector_rotation_deg": result["scan_detector_rotation_deg"],
            "scan_detector_rotation_flipped": scan_detector_rotation_flipped,
            "voltage_kV": self.voltage_kV, "semiangle_mrad": float(semiangle),
            "scan_sampling_A": list(self.scan_sampling_A),
            "detector_sampling_mrad": None if detector_sampling_mrad is None else list(detector_sampling_mrad),
            "bright_field_radius_px": None if bright_field_radius_px is None else float(bright_field_radius_px),
            "bright_field_intensity_threshold": float(bright_field_intensity_threshold),
            "num_bright_field_pixels": num_bright_field_pixels,
            "trials": 0 if manual else trials, "refinement": None if manual else result.get("refinement", refinement),
            "manual_locked": manual,
            # the sample model of the phase: the tilt-corrected fit's (scan frame, mrad) or None at tilt 0
            "tilt": result.get("sample_tilt_mrad") is not None, "objective": None if manual else result.get("objective"),
            "sample_tilt_mrad": result.get("sample_tilt_mrad"), "depth_spread_nm": result.get("depth_spread_nm"),
            "tilt_fit_gain": result.get("tilt_fit_gain"),
            "seed": int(seed), "loss": result["loss"], "elapsed_s": result["elapsed_s"],
        }, indent=2)
        self.direct_calibration_filename = (
            f"{export_slug(self.title, 'show4dstem')}_{self.shape_rows}x{self.shape_cols}x{self.det_rows}x{self.det_cols}_direct_calibration.json"
        )
        dpc_row, dpc_col = gpu_adapter.aligned_center_of_mass(
            source, (self.shape_rows, self.shape_cols),
            scan_detector_rotation_deg=result["scan_detector_rotation_deg"],
            scan_detector_rotation_flipped=scan_detector_rotation_flipped)
        return phase, np.ascontiguousarray(dpc_row, dtype=np.float32), np.ascontiguousarray(dpc_col, dtype=np.float32)

    def _on_direct_request_change(self, change) -> None:
        """The frontend's Direct ptycho Calculate Phase button: run the solver on a worker thread."""
        raw = str(change.get("new") or "")
        if not raw:
            return
        payload = json.loads(raw)  # the frontend sends {"action": "direct_phase", ...}
        self.direct_request = ""  # consume the trigger before any long work
        if payload.get("action") != "direct_phase":
            self.direct_status = f"Direct ptycho failed: unknown request {payload.get('action')!r}"
            return
        if not self.direct_enabled:
            self.direct_status = "Direct ptycho is available only in a live Python kernel. Precompute the direct phase before exporting standalone HTML."
            return
        if self.direct_busy:
            self.direct_status = "Direct ptycho is already running."
            return
        kwargs = {key: payload[key] for key in ("trials", "tilt", "refinement", "lock_aberrations")
                  if key in payload}
        if "trials" in kwargs:
            self.direct_trials = int(kwargs["trials"])
        if "tilt" in kwargs:
            self.direct_tilt = bool(kwargs["tilt"])
        if "refinement" in kwargs:
            self.direct_refinement = kwargs["refinement"]
        manual = bool(kwargs.get("lock_aberrations", False))
        if manual:
            # the sliders' aberrations and the physical rotation, reconstructed as given
            aberrations = payload.get("aberrations", {})
            self.direct_C10_nm = float(aberrations.get("C10_nm", self.direct_C10_nm))
            self.direct_C12_nm = float(aberrations.get("C12_nm", self.direct_C12_nm))
            self.direct_phi12_deg = float(aberrations.get("phi12_deg", self.direct_phi12_deg))
            self.direct_scan_detector_rotation_deg = wrap_degrees(
                payload.get("scan_detector_rotation_deg", self.direct_scan_detector_rotation_deg))
            kwargs["aberrations"] = {"C10_nm": self.direct_C10_nm, "C12_nm": self.direct_C12_nm,
                                     "phi12_deg": self.direct_phi12_deg}
            kwargs["scan_detector_rotation_deg"] = self.direct_scan_detector_rotation_deg
            kwargs["scan_detector_rotation_flipped"] = self._direct_flipped
        self.direct_lock_C10 = bool(payload.get("lock_C10", self.direct_lock_C10))
        self.direct_lock_C12 = bool(payload.get("lock_C12", self.direct_lock_C12))
        if not manual:
            kwargs["lock_C10"] = self.direct_lock_C10
            kwargs["lock_C12"] = self.direct_lock_C12

        def worker() -> None:
            """Run the solver off the kernel thread so the frontend stays responsive while it fits."""
            try:
                self.direct_phase(verbose=bool(payload.get("verbose", False)), **kwargs)
            except (ValueError, RuntimeError, MemoryError):
                return  # direct_phase already published the failure string

        threading.Thread(target=worker, name="Show4DSTEM-direct-phase", daemon=True).start()

    # ------------------------------------------------------------------
    # Offline packing
    # ------------------------------------------------------------------

    def _pack_offline(self, offline: bool | None) -> None:
        """Gzip the counts into ``_offline_stack`` so the browser computes without a kernel.

        ``uint8`` clips counts to 255 for compact browse data; ``offline_dtype="uint16"``
        preserves detector counts. ``offline=None`` packs when the stack is under
        the budget, ``True`` warns and skips above it, ``False`` does nothing.
        """
        if offline is False:
            return
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
        with self._stream_lock:
            if self._stream_display_pending or (self._stream_display_negotiated
                                                and self._stream_display_view is None):
                return
            with self.hold_sync():
                # Jupyter deep-compares DataViews. A fresh revision guarantees
                # a delivery event even when the full image bytes are identical.
                self.compare_image_revision += 1
                self._stream_panel_update = {}
                self._states_to_send.update({
                    "virtual_image_bytes", "vi_product_labels", "vi_product_map_frames", "vi_product_maps_bytes",
                    "frame_bytes", "compare_diffraction_bytes", "compare_diffraction_indices",
                    "compare_virtual_image_bytes", "compare_image_revision"})
            self._stream_display_barrier()

    def _stream_display_barrier(self) -> None:
        """Mark the ordered end of an image update; this is transport, not GPU presentation."""
        if self._stream_display_view is None or not hasattr(self, "_stream_stop"):
            return
        self._stream_display_seq += 1
        self._stream_display_pending = self._stream_display_seq
        self._stream_display_revision = self.compare_image_revision
        self._stream_display_sent_at = time.monotonic()
        self._stream_display_probe_seq = None
        self._send_stream_display_marker()

    def _send_stream_display_marker(self) -> None:
        """Transfer a pending receipt to another view without resending images."""
        self.send({"type": "show4dstem_stream_update", "version": 1,
                   "view": self._stream_display_view, "seq": self._stream_display_pending,
                   "revision": self._stream_display_revision})

    def _stream_compare_delta(self, previous, revision):
        """Send changed panels exactly; retain the complete immutable export snapshot."""
        current = self.compare_virtual_image_bytes
        pixels = self.shape_rows * self.shape_cols
        count = self.compare_panel_count
        if (self.compare_image_revision == revision or not count
                or len(previous) != len(current) or len(current) != count * pixels * 4):
            return None
        old = np.frombuffer(previous, np.uint32).reshape(count, pixels)
        new = np.frombuffer(current, np.uint32).reshape(count, pixels)
        slots = np.flatnonzero(np.any(old != new, axis=1)).tolist()
        if len(slots) == count:
            return None
        # Model traits still hold the full snapshot for export and new views.
        # Only this ordered background publication omits the unchanged panels.
        self._stream_panel_update = {"snapshot_pending": True}
        self._states_to_send.difference_update({"compare_virtual_image_bytes", "compare_image_revision"})
        message = {"type": "show4dstem_stream_panels", "version": 1, "base": revision,
                   "revision": self.compare_image_revision, "slots": slots,
                   "panels": count, "rows": self.shape_rows, "cols": self.shape_cols}
        buffers = [current[slot * pixels * 4:(slot + 1) * pixels * 4] for slot in slots]
        return message, buffers

    def _stream_compare_layout(self, previous, revision, previous_ids):
        """Reuse bit-identical panels by source identity during append or retirement."""
        pixels = self.shape_rows * self.shape_cols
        current = self.compare_virtual_image_bytes
        identities = [self._stream_source_ids[index] for index in self.compare_panel_indices]
        if (not identities or not previous_ids or self.compare_image_revision == revision
                or len(previous) != len(previous_ids) * pixels * 4
                or len(current) != len(identities) * pixels * 4):
            return
        old = np.frombuffer(previous, np.uint32).reshape(len(previous_ids), pixels)
        new = np.frombuffer(current, np.uint32).reshape(len(identities), pixels)
        lookup = {identity: index for index, identity in enumerate(previous_ids)}
        reuse = [lookup[identity] if identity in lookup and np.array_equal(old[lookup[identity]], new[index])
                 else -1 for index, identity in enumerate(identities)]
        if all(index < 0 for index in reuse):
            return
        slots = [index for index, source in enumerate(reuse) if source < 0]
        # This trait and all changed labels/indices share one remote-state
        # transaction. New metadata can never be paired with old panel slots.
        self._stream_panel_update = {
            "snapshot_pending": True,
            "base": revision, "revision": self.compare_image_revision,
            "previous_ids": previous_ids, "source_ids": identities, "reuse": reuse,
            "rows": self.shape_rows, "cols": self.shape_cols, "slots": slots,
            "buffers": [current[slot * pixels * 4:(slot + 1) * pixels * 4] for slot in slots],
        }
        self._states_to_send.difference_update({"compare_virtual_image_bytes", "compare_image_revision"})

    def get_state(self, key=None, drop_defaults=False):
        """Keep full snapshots coherent without changing saved-state or targeted-sync policy."""
        if key is not None:
            # Workers can publish while holding a session lock. Do not invert
            # the refresh order (stream lock, then session lock) here.
            return super().get_state(key=key, drop_defaults=drop_defaults)
        with self._stream_lock:
            state = super().get_state(drop_defaults=drop_defaults if self._stream_deferred_snapshot is None else False)
            # Full mounts and saved notebooks already contain canonical images;
            # a transport delta must not be replayed or embedded beside them.
            state.pop("_stream_panel_update", None)
            if self._stream_deferred_snapshot is None:
                return state
            # The shared mixin owns heavy-buffer trimming and previews. Only replace deferred display fields it retained.
            for name in self._stream_deferred_state.intersection(state):
                state[name] = self._stream_deferred_snapshot[name]
            if drop_defaults:
                traits = self.traits()
                state = {name: value for name, value in state.items()
                         if not self._compare(value, traits[name].default_value)}
            return state

    def set_state(self, sync_data) -> None:
        """Resolve browser selection by acquisition identity before applying traits."""
        with self._stream_lock:
            if self._stream_deferred_state:
                self._stream_update(force=True)
            if hasattr(self, "_stream_stop") and ("frame_idx" in sync_data or "_stream_selected_source" in sync_data):
                source = sync_data.get("_stream_selected_source")
                if source not in self._stream_source_ids:
                    self.send_state(["_stream_source_ids", "_stream_selected_source", "frame_idx"])
                    self.send({"type": "show4dstem_source_receipt", "status": "rejected", "source": source,
                               "error": "This scan was released. Select a retained scan."})
                    return
                sync_data = dict(sync_data, frame_idx=self._stream_source_ids.index(source))
            super().set_state(sync_data)

    def _interaction_context(self, keys: str) -> list:
        """A live acquisition keeps its identity when retention changes its index."""
        return [(self._stream_selected_source or self.frame_idx) if key == "frame_idx" else getattr(self, key)
                for key in keys.split()]

    def _handle_msg(self, msg) -> None:
        """Serialize browser edits with arriving scan images and session changes."""
        # Signal priority before waiting: the current refresh may hold the lock.
        content = msg.get("content", {}).get("data", {}).get("content", {})
        display_control = content.get("type") in {
            "show4dstem_stream_ack", "show4dstem_frontend_ready",
            "show4dstem_frontend_closed", "show4dstem_stream_alive", "show4dstem_stream_resync"}
        if not display_control:
            self._last_interaction_at = time.monotonic()
        with self._stream_lock:
            if self._stream_deferred_state and not display_control:
                self._stream_update(force=True)
            super()._handle_msg(msg)

    def _handle_frontend_ready_msg(self, _widget, content, _buffers) -> None:
        """Answer the frontend's ready message with the view buffers (``_schedule_initial_view_sync``)."""
        if not isinstance(content, dict) or content.get("version") != 1:
            return
        kind, view = content.get("type"), content.get("view")
        if kind == "show4dstem_frontend_ready":
            if content.get("stream_ack") is True and isinstance(content.get("view"), str):
                self._stream_display_negotiated = True
                self._stream_display_views[view] = content.get("stream_delta") is True
                if content.get("stream_layout") is True:
                    self._stream_layout_views.add(view)
                else:
                    self._stream_layout_views.discard(view)
                self._stream_display_delta = all(self._stream_display_views.values())
                self._stream_display_view = view
                self._stream_display_acknowledged = self._stream_display_seq
                self._stream_display_pending = 0
                self._stream_display_probe_seq = None
            if self._stream_deferred_state:
                self._stream_update(force=True)
            self._resend_view_state()
        elif (kind == "show4dstem_stream_ack" and view == self._stream_display_view
              and type(content.get("seq")) is int
              and self._stream_display_acknowledged < content["seq"] <= self._stream_display_seq):
            # The frontend applies ordered deltas before cumulatively acknowledging
            # the newest revision. Coalescing receipts never discards a sent delta.
            self._stream_display_acknowledged = content["seq"]
            if content["seq"] == self._stream_display_seq:
                self._stream_display_pending = 0
            self._stream_display_probe_seq = None
        elif kind == "show4dstem_frontend_closed" and isinstance(view, str):
            self._stream_display_views.pop(view, None)
            self._stream_layout_views.discard(view)
            self._stream_display_delta = bool(self._stream_display_views) and all(self._stream_display_views.values())
            if view == self._stream_display_view:
                self._stream_display_view = next(reversed(self._stream_display_views), None)
                if self._stream_display_view is not None and self._stream_display_pending:
                    self._send_stream_display_marker()
        elif (kind == "show4dstem_stream_alive" and isinstance(view, str)
              and view in self._stream_display_views and self._stream_display_probe_seq is not None
              and content.get("seq") == self._stream_display_probe_seq):
            self._stream_display_probe_seq = None
            self._stream_display_view = view
            if self._stream_display_pending:
                self._send_stream_display_marker()
        elif (kind == "show4dstem_stream_resync" and isinstance(view, str)
              and view in self._stream_display_views):
            self._stream_display_view = view
            self._stream_display_acknowledged = self._stream_display_seq
            self._stream_display_pending = 0
            self._resend_view_state()

    def _handle_position_request(self, _widget, content, _buffers) -> None:
        """Pipeline cursor requests without waiting for the kernel's idle notification."""
        if not isinstance(content, dict) or content.get("type") != "show4dstem_position":
            return
        receipt = {key: content.get(key) for key in ("view", "epoch", "seq")}
        receipt["type"] = "show4dstem_position_receipt"
        try:
            if content.get("version") != 1 or self._data is None or self.comm is None:
                raise ValueError("The position request is stale; reopen the viewer.")
            if content.get("context") != self._interaction_context("frame_idx vi_roi_mode compare_dp_mode"):
                raise ValueError("The selected acquisition or diffraction mode changed.")
            row, col = content["center"]
            if not (isinstance(row, (int, float)) and isinstance(col, (int, float))
                    and math.isfinite(row) and math.isfinite(col)
                    and 0 <= row < self.shape_rows and 0 <= col < self.shape_cols):
                raise ValueError("Select a position inside the scan.")
            self.send(dict(receipt, status="accepted"))
            self._last_interaction_at = time.monotonic()
            with self.hold_sync(), self.hold_trait_notifications():
                self.pos_row, self.pos_col = int(row), int(col)
            self._on_scan_position_change()
        except (KeyError, TypeError, ValueError, RuntimeError, MemoryError) as error:
            self._position_seen = None
            self.send(dict(receipt, status="rejected", error=str(error)))
        else:
            self.send(dict(receipt, status="complete"))

    def _handle_detector_request(self, _widget, content, _buffers) -> None:
        """Accept one center request before its synchronous image computation."""
        if not isinstance(content, dict) or content.get("type") != "show4dstem_detector":
            return
        receipt = {key: content.get(key) for key in ("view", "epoch", "seq")}
        receipt["type"] = "show4dstem_detector_receipt"
        try:
            context = "frame_idx vi_source view_mode compare_page_idx roi_mode roi_radius roi_radius_inner roi_width roi_height"
            if content.get("version") != 1 or self._data is None or self.comm is None:
                raise ValueError("The detector request is stale; reopen the viewer.")
            if content.get("context") != self._interaction_context(context):
                raise ValueError("Detector settings changed before this request was accepted.")
            row, col = map(float, content["center"])
            if not (math.isfinite(row) and math.isfinite(col)
                    and 0 <= row < self.det_rows and 0 <= col < self.det_cols):
                raise ValueError("The detector center must lie inside the diffraction pattern.")
            self.send(dict(receipt, status="accepted"))
            self._last_interaction_at = time.monotonic()
            with self.hold_sync(), self.hold_trait_notifications():
                self.roi_active = True
                self.roi_center = [row, col]
            self._on_roi_change()
            if self._multiple_view_active() and not self.compare_virtual_image_bytes:
                raise RuntimeError(self.compare_status)
        except (KeyError, TypeError, ValueError, RuntimeError, MemoryError) as error:
            self._roi_seen = None
            self.send(dict(receipt, status="rejected", error=str(error)))
        else:
            self.send(dict(receipt, status="complete"))

    def _handle_detector_radius_request(self, _widget, content, _buffers) -> None:
        """Acknowledge a detector resize before computing its complete image."""
        if not isinstance(content, dict) or content.get("type") != "show4dstem_detector_radius":
            return
        receipt = {key: content.get(key) for key in ("view", "epoch", "seq")}
        receipt["type"] = "show4dstem_detector_radius_receipt"
        try:
            context = (
                "frame_idx vi_source view_mode compare_page_idx roi_mode "
                "roi_center_row roi_center_col"
            )
            if content.get("version") != 1 or self._data is None or self.comm is None:
                raise ValueError("The radius request is stale; reopen the viewer.")
            if content.get("context") != self._interaction_context(context):
                raise ValueError("Detector settings changed before this radius request was accepted.")
            outer, inner = map(float, content["radii"])
            if not (math.isfinite(outer) and math.isfinite(inner) and 0 <= inner < outer):
                raise ValueError(
                    f"Use finite detector radii with 0 <= inner < outer; got {inner}, {outer}."
                )
            self.send(dict(receipt, status="accepted"))
            self._last_interaction_at = time.monotonic()
            with self.hold_sync(), self.hold_trait_notifications():
                self.roi_radius, self.roi_radius_inner = outer, inner
                self.roi_active = True
            self._on_roi_change()
            if self._multiple_view_active() and not self.compare_virtual_image_bytes:
                raise RuntimeError(self.compare_status)
        except (KeyError, TypeError, ValueError, OverflowError, RuntimeError, MemoryError) as error:
            self._roi_seen = None
            self.send(dict(receipt, status="rejected", error=str(error)))
        else:
            self.send(dict(receipt, status="complete"))

    def __repr__(self) -> str:
        """Shape, calibration, cursor and title on one line."""
        shape = (f"({self.n_frames}, {self.shape_rows}, {self.shape_cols}, {self.det_rows}, {self.det_cols})"
                 if self.n_frames > 1 else f"({self.shape_rows}, {self.shape_cols}, {self.det_rows}, {self.det_cols})")
        frame = f", {self.frame_dim_label.lower()}={self.frame_idx}" if self.n_frames > 1 else ""
        title = f", title='{self.title}'" if self.title else ""
        scan = f"scan_sampling_A={self.scan_sampling_A}" if self.scan_sampling_A else "scan uncalibrated"
        detector = (f"detector_sampling_mrad={self.detector_sampling_mrad}" if self.detector_sampling_mrad else
                    f"detector_sampling_inv_A={self.detector_sampling_inv_A}" if self.detector_sampling_inv_A else "detector uncalibrated")
        return f"Show4DSTEM(shape={shape}, {scan}, {detector}, pos=({self.pos_row}, {self.pos_col}){frame}{title})"

    _static_png_b64 = static_png_b64  # the saved-notebook preview StaticFallbackMixin renders

    # ------------------------------------------------------------------
    # Observers
    # ------------------------------------------------------------------

    def _set_roi_to_disk(self) -> None:
        """Place a circular ROI on the fitted bright-field disk without firing recomputes."""
        self._suppress_roi_recompute = True
        with self.hold_trait_notifications():
            self.roi_mode = "circle"
            self.roi_center_col = self.bright_field_center_col_px
            self.roi_center_row = self.bright_field_center_row_px
            self.roi_center = [self.bright_field_center_row_px, self.bright_field_center_col_px]
            self.roi_radius = float(max(1.0, self.bright_field_radius_px))
            self.roi_active = True
        self._suppress_roi_recompute = False

    def _on_preset_request(self, change) -> None:
        """React to a frontend preset button: apply it, then clear the request so the same preset can be clicked again.

        The detector traits, the new image and the cleared request reach the
        frontend in one message, which it renders once.
        """
        name = (change.get("new") or "").strip().lower()
        if name in PRESET_RADII:
            with self.hold_sync():
                self.apply_preset(name)
                self._preset_request = ""

    def _on_roi_change(self, change=None) -> None:
        """React to a detector ROI edit: recompute the image in view unless a batched update or a product map is showing.

        One frontend message can change several ROI traits (a drag writes the
        row, the column and the ``[row, col]`` pair; an annulus edit writes both
        radii). traitlets applies them all and then calls this once per changed
        trait, so every call of one message sees the same detector: the first
        reduces the image and the rest return here, one reduction per message
        instead of one per trait.
        """
        roi = (self.roi_active, self.roi_mode, self.roi_center_row, self.roi_center_col, self.roi_radius,
               self.roi_radius_inner, self.roi_width, self.roi_height)
        if roi == self._roi_seen:
            return
        self._roi_seen = roi
        if not self.roi_active or self._suppress_roi_recompute or self.vi_source != "roi":
            return
        if not self._multiple_view_active():
            self._compute_virtual_image_from_roi()
        self._refresh_compare_virtual_images()

    def _on_scan_position_change(self, change=None) -> None:
        """React to a cursor move: send the pattern of the new scan position once.

        A frontend move writes the row and the column in one message and
        traitlets notifies each changed trait after both are set, so the second
        call sees the position the first one already sent.
        """
        position = (self.pos_row, self.pos_col)
        if position == self._position_seen:
            return
        self._position_seen = position
        self._update_frame()

    def _on_roi_center_change(self, change=None) -> None:
        """Split the ``[row, col]`` pair of a detector move into the row and column traits, whose observer reduces the image."""
        with self.hold_trait_notifications():  # both halves land first: never the new row with the old column
            self.roi_center_row, self.roi_center_col = self.roi_center

    def _on_calibration_change(self, change=None) -> None:
        """Drop the cached preset images when the disk centre or radius moves: they were summed over the old masks."""
        self._preset_images = {}

    def _on_frame_idx_change(self, change=None) -> None:
        """React to a dataset switch: the cached presets belong to the old dataset, so drop them and repaint."""
        if self._stream_source_ids:
            self._stream_selected_source = self._stream_source_ids[self.frame_idx]
        if self._stream_defer_compute:
            return
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
        if self._stream_defer_compute:
            return
        if change and change.get("name") == "view_mode":
            self.view_mode = self._normalize_view_mode(change.get("new", "single"))
        if change and change.get("name") == "compare_group_mode":
            self.compare_group_mode = self._normalize_compare_group_mode(change.get("new", "paged"))
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
            self.compare_dp_mode = self._normalize_compare_dp_mode(change.get("new", "average"))
        if self._multiple_view_active():
            self._update_frame()

    # ------------------------------------------------------------------
    # Comparison grid
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_view_mode(value: str) -> str:
        """``"single"`` or ``"multiple"`` from a caller, frontend or saved state; anything else is refused."""
        if value not in ("single", "multiple"):
            raise ValueError(f"view_mode must be 'single' or 'multiple', got {value!r}")
        return value

    @staticmethod
    def _normalize_compare_dp_mode(value: str) -> str:
        """``"average"``, ``"selected"`` or ``"all"``; anything else is refused."""
        if value not in ("average", "selected", "all"):
            raise ValueError(f"compare_dp_mode must be 'average', 'selected', or 'all', got {value!r}")
        return value

    @staticmethod
    def _normalize_compare_group_mode(value: str) -> str:
        """``"paged"`` or ``"all"``; anything else is refused."""
        if value not in ("paged", "all"):
            raise ValueError(f"compare_group_mode must be 'paged' or 'all', got {value!r}")
        return value

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
        with self.hold_sync(), self.hold_trait_notifications():
            self.compare_virtual_image_bytes = b""
            self._stream_panel_update = {}
            self.compare_image_revision += 1
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
            with self.hold_sync(), self.hold_trait_notifications():
                self.compare_virtual_image_bytes = memoryview(images).cast("B")
                self._stream_panel_update = {}
                self.compare_image_revision += 1
                self.compare_panel_count = len(indices)
                self.compare_panel_indices = indices
                self.compare_status = self._compare_status(indices)
        except (RuntimeError, ValueError, MemoryError) as exc:
            self._clear_compare_virtual_images(f"Multiple grid unavailable: {exc}")
        finally:
            self._suppress_compare_recompute = False
