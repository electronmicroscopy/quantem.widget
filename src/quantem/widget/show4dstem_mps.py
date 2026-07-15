"""MacBook raw-Metal widget adapter for the public Show4DSTEM factory.

Normal users should call ``quantem.widget.Show4DSTEM(load(..., backend="mps"))``.
The names in this module stay importable for compatibility and for backend
tests, but they are implementation details behind the single public widget API.

This file is deliberately one cohesive adapter rather than many small MPS UI
modules. The responsibilities split across layers like this:

* ``show4dstem_factory`` decides whether public ``Show4DSTEM(...)`` should route
  to the base viewer or this MPS adapter.
* ``kernels/io/mps.py`` owns HDF5/Metal decode and detector binning.
* ``quantem.gpu.compute.backends.MetalRawBackend`` owns masked-sum compute,
  fast-sidecar/radial-cache lifecycles, and lazy multi-dataset backend state.
* ``Show4DSTEMMPS`` owns only widget-facing traitlets, observers, preset caches,
  ROI mask translation, and status updates needed by the frontend.

See ``quantem.gpu.compute.backend`` for the protocol and
``quantem.gpu.compute.backends.MetalRawBackend`` for the compute implementation.
"""
from __future__ import annotations

import gc
import threading
import time

import numpy as np
import traitlets

from quantem.widget.show4dstem import Show4DSTEM
from quantem.widget.detector import detector_mask
from quantem.gpu.compute.mps import (
    ChunkedFrames,
    _DEFAULT_COMPACT_TARGET_BYTES,
    _bin_mask,
    _upsample_bin_dp,
)


def _drop_cached_decompressor():
    """Release decoder scratch buffers before allocating interaction sidecars."""
    from quantem.widget.io import clear_mps_cache
    clear_mps_cache()


class Show4DSTEMMPS(Show4DSTEM):
    """Show4DSTEM widget adapter over raw-Metal frame buffers.

    Lifecycle (fast_vi sidecar / radial cache / multi-dataset) is delegated to
    ``MetalRawBackend``; this class drives it through the ``self._compute``
    property. The MPS-specific traits below are observed and synced to the JS
    widget for status indicators.
    """

    fast_interaction = traitlets.Bool(False).tag(sync=True)
    fast_interaction_ready = traitlets.Bool(False).tag(sync=True)
    fast_interaction_building = traitlets.Bool(False).tag(sync=True)
    radial_interaction_ready = traitlets.Bool(False).tag(sync=True)
    radial_interaction_building = traitlets.Bool(False).tag(sync=True)

    # ----------------------------------------------------------------- setup

    def __init__(
        self,
        *args,
        fast_interaction: bool = False,
        fast_interaction_verbose: bool = True,
        fast_interaction_async: bool = False,
        full_resolution_interaction: bool = False,
        auto_detect_frames: int | None = 64,
        initial_preset: str | None = "BF",
        **kwargs,
    ):
        verbose = bool(kwargs.pop("verbose", True))
        if full_resolution_interaction:
            raise ValueError(
                "full_resolution_interaction has been disabled on the MacBook "
                "viewer path. Use load(...) plus show_4dstem_mps(...) for the "
                "supported real-time fast path."
            )
        self._fast_interaction_verbose = bool(fast_interaction_verbose)
        self._fast_interaction_async = bool(fast_interaction_async)
        self._fast_interaction_thread: threading.Thread | None = None
        self._fast_interaction_error: str | None = None
        self._mps_folder_shutdown = False
        self.auto_detect_frames = auto_detect_frames
        self._suppress_fast_interaction_observer = False
        self._mps_initializing = True
        kwargs.setdefault("precompute_virtual_images", False)
        t0 = time.perf_counter()
        try:
            super().__init__(*args, verbose=False, **kwargs)
        finally:
            self._mps_initializing = False
        self._det_row_coords_np = np.arange(self.det_rows, dtype=np.float32)[:, None]
        self._det_col_coords_np = np.arange(self.det_cols, dtype=np.float32)[None, :]
        self._wire_multi_dataset()
        self.observe(self._on_fast_interaction_change, names=["fast_interaction"])
        pre_binned_fast = (
            isinstance(self._data, ChunkedFrames)
            and int(getattr(self._data, "det_bin", 1)) > 1
        )
        fused_fast = (
            isinstance(self._data, ChunkedFrames)
            and getattr(self._data, "fast_vi", None) is not None
        )
        if pre_binned_fast:
            fast_interaction = False
            self.fast_interaction_ready = True
        elif fused_fast:
            self.fast_interaction_ready = True
        if initial_preset is not None:
            self._mps_initializing = True
            try:
                self.apply_preset(initial_preset)
            finally:
                self._mps_initializing = False
        if fast_interaction:
            self.set_fast_interaction(True, wait=not self._fast_interaction_async)
            if fused_fast:
                self._cache_fast_presets()
        else:
            self._clear_virtual_image_caches()
            self._compute_virtual_image_from_roi()
        if verbose:
            det_bin = int(getattr(self._data, "det_bin", 1)) if isinstance(
                self._data, ChunkedFrames
            ) else 1
            fb = int(getattr(self._data, "fast_bin", 2)) if isinstance(
                self._data, ChunkedFrames) else 2
            mode = (
                f"fast detector-bin{det_bin}"
                if det_bin > 1 else
                f"fast bin{fb} ready" if fast_interaction and self.fast_interaction_ready else
                f"fast bin{fb} async" if fast_interaction and fast_interaction_async else
                f"fast bin{fb}" if fast_interaction else "full 192x192 exact"
            )
            shape = f"{self.shape_rows}x{self.shape_cols}x{self.det_rows}x{self.det_cols}"
            print(
                f"Ready MPS viewer in {time.perf_counter() - t0:.2f}s "
                f"({shape}, Raw Metal, {mode})"
            )

    # ----------------------------------------------------------------- multi-dataset status
    # Lazy 5D multi-file proxy — the backend owns the underlying MultiChunkedFrames;
    # this widget only wires the n_frames trait + title to its on_ready callback.

    def _multi_source(self):
        """Return the live multi-dataset proxy, if this viewer wraps one."""
        data = getattr(self, "_data", None)
        if (
            hasattr(data, "datasets")
            and hasattr(data, "set_active")
            and hasattr(data, "on_ready")
        ):
            return data
        return None

    def _wire_multi_dataset(self):
        multi = self._multi_source()
        if multi is None:
            self._multi = None
            return
        self._multi = multi
        self._multi_total = len(multi.datasets)
        self.frame_dim_label = "Dataset"
        names = list(getattr(multi, "names", []) or [])
        self._frame_labels = names
        self.frame_labels = names
        # Keep the full dataset axis visible while background decode fills slots.
        self.n_frames = max(1, self._multi_total)
        self._refresh_multi_title()
        try:
            from tornado.ioloop import IOLoop
            self._ioloop = IOLoop.current()
        except Exception:
            self._ioloop = None
        multi.on_ready = self._on_multi_dataset_ready

    def _refresh_multi_title(self):
        multi = getattr(self, "_multi", None)
        if multi is None:
            return
        names = list(getattr(multi, "names", []) or [])
        active_idx = int(getattr(multi, "active_idx", 0))
        name = names[active_idx] if 0 <= active_idx < len(names) else f"dataset {active_idx}"
        n_ready = int(getattr(multi, "n_ready", 1))
        n_total = self._multi_total
        self.title = name if n_ready >= n_total else f"{name}  -  loading {n_ready}/{n_total}"

    def _on_multi_dataset_ready(self, idx: int):
        if self._mps_folder_shutdown:
            return
        multi = getattr(self, "_multi", None)
        if multi is None:
            return

        def _apply():
            if self._mps_folder_shutdown:
                return
            self._multi_total = len(getattr(multi, "datasets", []) or [])
            self.n_frames = max(1, self._multi_total)
            labels = list(getattr(multi, "names", []) or [])
            self._frame_labels = labels
            self.frame_labels = labels
            self._refresh_multi_title()
            self._refresh_compare_virtual_images()

        loop_running = False
        if self._ioloop is not None:
            try:
                loop_running = bool(self._ioloop.asyncio_loop.is_running())
            except Exception:
                loop_running = False
        if self._ioloop is not None and loop_running:
            self._ioloop.add_callback(_apply)
        else:
            _apply()

    def _on_frame_idx_change(self, change=None):
        multi = getattr(self, "_multi", None)
        if multi is not None:
            multi.set_active(int(self.frame_idx))
            self._compute_backend = None
            self._compute_for = None
            self._refresh_multi_title()
        return super()._on_frame_idx_change(change)

    # ----------------------------------------------------------------- frame and detector compute
    # Frame access + masked_sum already route through self._compute in the parent,
    # so we only need to handle the ROI-specific virtual image (preset caching,
    # radial path, scan-position column fallback).

    def _fast_masked_sum(self, mask):
        """Override to keep ChunkedFrames-only logic; falls back to parent torch
        path for any other data type."""
        import torch
        data = self._data
        if not isinstance(data, ChunkedFrames):
            return super()._fast_masked_sum(mask)
        mask_np = mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
        b = self._compute
        if self.fast_interaction and self.fast_interaction_ready and b.has_fast:
            self._ensure_fast_interaction_ready()
        vi = b.masked_sum(mask_np)  # routes through fast_vi when ready
        return torch.from_numpy(vi).reshape(self._scan_shape)

    def _detector_mask_np(self) -> np.ndarray | None:
        """ROI -> detector mask (numpy). Shared across all MPS code paths."""
        cx = float(self.roi_center_col)
        cy = float(self.roi_center_row)
        rows = self._det_row_coords_np
        cols = self._det_col_coords_np

        if self.roi_mode == "point":
            return (np.abs(cols - cx) < 0.5) & (np.abs(rows - cy) < 0.5)
        if self.roi_mode == "circle" and self.roi_radius > 0:
            return detector_mask((cy, cx), 0.0, float(self.roi_radius),
                                 (self.det_rows, self.det_cols))
        if self.roi_mode == "square" and self.roi_radius > 0:
            half_size = float(self.roi_radius)
            return (np.abs(cols - cx) <= half_size) & (np.abs(rows - cy) <= half_size)
        if self.roi_mode == "annular" and self.roi_radius > 0:
            return detector_mask((cy, cx), float(self.roi_radius_inner),
                                 float(self.roi_radius), (self.det_rows, self.det_cols))
        if self.roi_mode == "rect" and self.roi_width > 0 and self.roi_height > 0:
            half_w = float(self.roi_width) / 2.0
            half_h = float(self.roi_height) / 2.0
            return (np.abs(cols - cx) <= half_w) & (np.abs(rows - cy) <= half_h)
        return None

    def _get_frame(self, row: int, col: int) -> np.ndarray:
        data = self._data
        if not isinstance(data, ChunkedFrames):
            return super()._get_frame(row, col)
        return self._compute.frame(row * self.shape_cols + col)

    # ----------------------------------------------------------------- auto_detect_center
    def auto_detect_center(self, update_roi: bool = True):
        data = self._data
        if not isinstance(data, ChunkedFrames):
            return super().auto_detect_center(update_roi)
        sample = self.auto_detect_frames
        if sample is not None and int(sample) > 0 and int(sample) < data._n:
            sample = int(sample)
            first = data.chunks[0]
            if sample <= int(first.shape[0]) and not data.vi.row_prefix_enabled:
                mean_dp = np.asarray(first[:sample], dtype=np.float32).mean(axis=0)
            else:
                indices = np.arange(sample, dtype=np.uint32)
                mean_dp = self._compute.reduce_frames(indices, "mean")
        else:
            mean_dp = self._compute.mean_dp()
        mean_dp = np.asarray(mean_dp, dtype=np.float32)
        threshold = float(mean_dp.mean()) + float(mean_dp.std())
        mask = mean_dp > threshold
        total = int(mask.sum())
        if total == 0:
            return self
        rows = np.arange(mean_dp.shape[0], dtype=np.float32)[:, None]
        cols = np.arange(mean_dp.shape[1], dtype=np.float32)[None, :]
        cx = float((cols * mask).sum() / total)
        cy = float((rows * mask).sum() / total)
        radius = float(round(np.sqrt(total / np.pi)))
        self.center_col, self.center_row, self.bf_radius = cx, cy, radius
        if update_roi:
            self.roi_center_col = cx
            self.roi_center_row = cy
            if self.fast_interaction and self.fast_interaction_ready:
                self._clear_virtual_image_caches()
                self._compute_virtual_image_from_roi()
        return self

    # ----------------------------------------------------------------- detector ROI -> virtual image
    def _set_virtual_image_bytes_np(self, vi: np.ndarray):
        arr = np.asarray(vi).reshape(self._scan_shape)
        arr = np.asarray(arr, dtype=np.float32, order="C")
        self.virtual_image_bytes = arr.tobytes()

    def _set_virtual_image_startup_preview(self):
        rows = np.linspace(0.0, 1.0, self.shape_rows, dtype=np.float32)[:, None]
        cols = np.linspace(0.0, 1.0, self.shape_cols, dtype=np.float32)[None, :]
        try:
            frame = self._get_frame(self.shape_rows // 2, self.shape_cols // 2)
            level = float(np.asarray(frame, dtype=np.float32).mean())
        except Exception:
            level = 1.0
        scale = max(level, 1.0)
        preview = scale * (0.95 + 0.05 * (rows + cols))
        self._set_virtual_image_bytes_np(preview)

    def _compute_virtual_image_from_roi(self):
        data = self._data
        if getattr(self, "_mps_initializing", False):
            self.virtual_image_bytes = b""
            return
        if not isinstance(data, ChunkedFrames):
            return super()._compute_virtual_image_from_roi()
        cached = self._get_cached_preset()
        if cached is not None:
            self.virtual_image_bytes = cached
            return
        b = self._compute
        if (self.fast_interaction and self._fast_interaction_async
                and not self.fast_interaction_ready):
            self._set_virtual_image_startup_preview()
            return
        if (self.fast_interaction and not self.fast_interaction_ready
                and not self._fast_interaction_async):
            self._ensure_fast_interaction_ready()

        # Radial-cache exact path (full no-bin BF/ADF on circular/annular ROIs).
        start_radial_background = False
        if (not self.fast_interaction
                and self.roi_mode in ("circle", "annular")
                and float(self.roi_radius) > 0
                and "radial_cache" in b.capabilities):
            inner = float(self.roi_radius_inner) if self.roi_mode == "annular" else 0.0
            radial_vi = b.radial_masked_sum(
                center_row=float(self.roi_center_row),
                center_col=float(self.roi_center_col),
                outer_radius=float(self.roi_radius),
                inner_radius=inner,
                build=False,
            )
            if radial_vi is not None:
                self._set_virtual_image_bytes_np(radial_vi)
                return
            start_radial_background = True

        mask = self._detector_mask_np()
        if mask is None:
            # Point/no-mask ROI: read one scan-position column (single detector
            # pixel under the marker).
            row = int(max(0, min(round(float(self.roi_center_row)), self.det_rows - 1)))
            col = int(max(0, min(round(float(self.roi_center_col)), self.det_cols - 1)))
            if self.fast_interaction and self.fast_interaction_ready and b.has_fast:
                self._ensure_fast_interaction_ready()
                fast_vi = data.fast_vi
                fast_rows, fast_cols = fast_vi.det
                scale_r = self.det_rows / fast_rows
                scale_c = self.det_cols / fast_cols
                fast_row = int(max(0, min(round(row / scale_r), fast_rows - 1)))
                fast_col = int(max(0, min(round(col / scale_c), fast_cols - 1)))
                fast_mask = np.zeros((fast_rows, fast_cols), dtype=bool)
                fast_mask[fast_row, fast_col] = True
                self._set_virtual_image_bytes_np(fast_vi.masked_sum(fast_mask))
            else:
                self._set_virtual_image_bytes_np(data.column(row, col))
            if start_radial_background:
                self._kick_radial_background()
            return

        vi = b.masked_sum(mask)  # routes through fast_vi when ready
        self._set_virtual_image_bytes_np(vi)
        if start_radial_background:
            self._kick_radial_background()

    # ----------------------------------------------------------------- detector-preset cache
    def _clear_virtual_image_caches(self):
        self._cached_bf_virtual = None
        self._cached_abf_virtual = None
        self._cached_adf_virtual = None
        self._cached_haadf_virtual = None

    def _get_cached_preset(self):
        if abs(self.roi_center_col - self.center_col) >= 1:
            return None
        if abs(self.roi_center_row - self.center_row) >= 1:
            return None
        bf = float(self.bf_radius)
        if self.roi_mode == "circle" and abs(self.roi_radius - bf) < 1:
            return self._cached_bf_virtual
        if (self.roi_mode == "annular"
                and abs(self.roi_radius_inner - bf * 0.5) < 1
                and abs(self.roi_radius - bf) < 1):
            return self._cached_abf_virtual
        if (self.roi_mode == "annular"
                and abs(self.roi_radius_inner - bf) < 1
                and abs(self.roi_radius - bf * 2.0) < 1):
            return self._cached_adf_virtual
        if (self.roi_mode == "annular"
                and abs(self.roi_radius_inner - bf * 2.0) < 1
                and abs(self.roi_radius - bf * 4.0) < 1):
            return self._cached_haadf_virtual
        return None

    def _preset_mask_np(self, name: str) -> np.ndarray | None:
        bf = float(max(1.0, self.bf_radius))
        bands = {"bf": (0.0, bf), "abf": (0.5 * bf, bf),
                 "adf": (bf, 2.0 * bf), "haadf": (2.0 * bf, 4.0 * bf)}
        band = bands.get(str(name).strip().lower())
        if band is None:
            return None
        return detector_mask((float(self.center_row), float(self.center_col)),
                             band[0], band[1], (self.det_rows, self.det_cols))

    def _cache_fast_presets(self):
        b = self._compute
        if "fast_sidecar" not in b.capabilities or not b.has_fast:
            return
        if not self.fast_interaction_ready:
            return
        masks = {}
        for name in ("bf", "abf", "adf", "haadf"):
            m = self._preset_mask_np(name)
            if m is not None:
                masks[name] = m
        cached = b.cache_fast_presets(masks)
        attr_map = {
            "bf": "_cached_bf_virtual",
            "abf": "_cached_abf_virtual",
            "adf": "_cached_adf_virtual",
            "haadf": "_cached_haadf_virtual",
        }
        for name, arr in cached.items():
            setattr(self, attr_map[name], arr.tobytes())

    # ----------------------------------------------------------------- fast-sidecar lifecycle
    def set_fast_interaction(self, enabled: bool = True, *, wait: bool = True):
        """Toggle bin2 fast interaction for BF/DF/ADF virtual images."""
        enabled = bool(enabled)
        if enabled and wait:
            self._ensure_fast_interaction_ready()
        self._suppress_fast_interaction_observer = True
        try:
            self.fast_interaction = enabled
        finally:
            self._suppress_fast_interaction_observer = False
        self._clear_virtual_image_caches()
        self._compute_virtual_image_from_roi()
        if enabled and not wait:
            self._start_fast_interaction_background()
        return self

    def _ensure_fast_interaction_ready(self) -> bool:
        b = self._compute
        if "fast_sidecar" not in b.capabilities:
            return False
        ok = b.ensure_fast_sidecar(verbose=self._fast_interaction_verbose)
        if ok:
            self.fast_interaction_ready = True
        return ok

    def _on_fast_interaction_change(self, change=None):
        if getattr(self, "_suppress_fast_interaction_observer", False):
            return
        if self.fast_interaction and self._fast_interaction_async:
            self._start_fast_interaction_background()
        elif self.fast_interaction:
            self._ensure_fast_interaction_ready()
        self._clear_virtual_image_caches()
        self._compute_virtual_image_from_roi()

    def _start_fast_interaction_background(self):
        if self.fast_interaction_ready or self.fast_interaction_building:
            return
        b = self._compute
        if "fast_sidecar" not in b.capabilities:
            return
        self.fast_interaction_building = True
        self._fast_interaction_error = None

        def _build():
            if self._fast_interaction_async:
                time.sleep(0.05)
            try:
                ok = b.ensure_fast_sidecar(verbose=self._fast_interaction_verbose)
                if ok:
                    self.fast_interaction_ready = True
                    self._clear_virtual_image_caches()
                    self._cache_fast_presets()
                    if self.fast_interaction:
                        self._compute_virtual_image_from_roi()
                        if getattr(self, "vi_roi_mode", "off") != "off":
                            self._compute_vi_roi_dp()
            except Exception as exc:  # pragma: no cover
                self._fast_interaction_error = repr(exc)
            finally:
                self.fast_interaction_building = False

        self._fast_interaction_thread = threading.Thread(
            target=_build, name="Show4DSTEMMPS-fast-interaction", daemon=True,
        )
        self._fast_interaction_thread.start()

    def wait_for_fast_interaction(self, timeout: float | None = None) -> bool:
        thread = self._fast_interaction_thread
        if thread is not None:
            thread.join(timeout)
        if self._fast_interaction_error is not None:
            raise RuntimeError(self._fast_interaction_error)
        return bool(self.fast_interaction_ready)

    # ----------------------------------------------------------------- radial-cache lifecycle
    def _kick_radial_background(self):
        """Ask the backend to build the radial cache at the current ROI center."""
        b = self._compute
        if "radial_cache" not in b.capabilities:
            return
        center_row = float(self.roi_center_row)
        center_col = float(self.roi_center_col)
        if b.radial_cache_ready(center_row, center_col):
            self.radial_interaction_ready = True
            return
        self.radial_interaction_ready = False
        self.radial_interaction_building = True
        b.ensure_radial_cache(center_row, center_col)

        # Poll for completion on a worker thread; flips traits when done.
        def _watch():
            while b.radial_building:
                time.sleep(0.1)
            if b.radial_error is None and b.radial_cache_ready(center_row, center_col):
                self.radial_interaction_ready = True
            self.radial_interaction_building = False

        threading.Thread(target=_watch, name="Show4DSTEMMPS-radial-watch",
                         daemon=True).start()

    def wait_for_radial_interaction(self, timeout: float | None = None) -> bool:
        b = self._compute
        if "radial_cache" not in b.capabilities:
            return False
        # Wait for backend's internal thread to settle.
        deadline = None if timeout is None else (time.perf_counter() + timeout)
        while b.radial_building:
            time.sleep(0.05)
            if deadline is not None and time.perf_counter() >= deadline:
                break
        if b.radial_error is not None:
            raise RuntimeError(b.radial_error)
        return bool(self.radial_interaction_ready)

    # ----------------------------------------------------------------- ROI observer guards
    def _on_roi_change(self, change=None):
        if getattr(self, "_mps_initializing", False):
            return
        return super()._on_roi_change(change)

    def _on_roi_center_change(self, change=None):
        if getattr(self, "_mps_initializing", False):
            return
        return super()._on_roi_center_change(change)

    # ----------------------------------------------------------------- scan-ROI -> diffraction pattern
    def _clear_vi_roi_dp(self):
        if hasattr(self, "vi_roi_dp_bytes"):
            self.vi_roi_dp_bytes = b""
        if hasattr(self, "summed_dp_bytes"):
            self.summed_dp_bytes = b""
        if hasattr(self, "summed_dp_count"):
            self.summed_dp_count = 0

    def _set_vi_roi_dp(self, dp: np.ndarray, n_positions: int):
        payload = np.asarray(dp, dtype=np.float32, order="C").tobytes()
        if hasattr(self, "vi_roi_dp_bytes"):
            self.vi_roi_dp_bytes = payload
        if hasattr(self, "summed_dp_bytes"):
            self.summed_dp_bytes = payload
        if hasattr(self, "summed_dp_count"):
            self.summed_dp_count = int(n_positions)

    def _vi_roi_indices_np(self) -> np.ndarray:
        rows = np.arange(self.shape_rows, dtype=np.float32)[:, None]
        cols = np.arange(self.shape_cols, dtype=np.float32)[None, :]
        center_row = float(self.vi_roi_center_row)
        center_col = float(self.vi_roi_center_col)
        if self.vi_roi_mode == "point":
            mask = (np.abs(rows - center_row) < 0.5) & (np.abs(cols - center_col) < 0.5)
        elif self.vi_roi_mode == "circle":
            radius = float(self.vi_roi_radius)
            mask = (rows - center_row) ** 2 + (cols - center_col) ** 2 <= radius ** 2
        elif self.vi_roi_mode == "square":
            half_size = float(self.vi_roi_radius)
            mask = ((np.abs(rows - center_row) <= half_size)
                    & (np.abs(cols - center_col) <= half_size))
        elif self.vi_roi_mode == "rect":
            half_w = float(self.vi_roi_width) / 2.0
            half_h = float(self.vi_roi_height) / 2.0
            mask = ((np.abs(rows - center_row) <= half_h)
                    & (np.abs(cols - center_col) <= half_w))
        else:
            return np.empty(0, dtype=np.uint32)
        return np.flatnonzero(mask.reshape(-1)).astype(np.uint32, copy=False)

    def _compute_summed_dp_from_vi_roi(self):
        data = self._data
        if not isinstance(data, ChunkedFrames):
            return super()._compute_summed_dp_from_vi_roi()
        if self.vi_roi_mode == "off":
            self._clear_vi_roi_dp()
            return
        indices = self._vi_roi_indices_np()
        n_positions = int(indices.size)
        if n_positions == 0:
            self._clear_vi_roi_dp()
            return
        dp = self._compute.reduce_frames(indices, "mean")
        b = self._compute
        if self.fast_interaction and self.fast_interaction_ready and b.has_fast:
            # backend returned a bin2 DP; upsample to full det
            dp = _upsample_bin_dp(dp, (self.det_rows, self.det_cols), b.fast_bin)
        self._set_vi_roi_dp(dp, n_positions)

    def _compute_vi_roi_dp(self):
        data = self._data
        if not isinstance(data, ChunkedFrames):
            return super()._compute_vi_roi_dp()
        if self.vi_roi_mode == "off":
            self._clear_vi_roi_dp()
            return
        indices = self._vi_roi_indices_np()
        n_positions = int(indices.size)
        if n_positions == 0:
            self._clear_vi_roi_dp()
            return
        reduce = getattr(self, "vi_roi_reduce", "mean")
        b = self._compute
        if reduce in ("mean", "sum"):
            dp = b.reduce_frames(indices, "mean")
            if reduce == "sum":
                dp = dp * float(n_positions)
            if self.fast_interaction and self.fast_interaction_ready and b.has_fast:
                dp = _upsample_bin_dp(dp, (self.det_rows, self.det_cols), b.fast_bin)
        elif reduce == "max":
            dp = b.reduce_frames(indices, "max")
        else:
            return
        self._set_vi_roi_dp(dp, n_positions)

    # ----------------------------------------------------------------- folder lifecycle

    def _publish_mps_folder_watch_status(self, state: str, detail: str = "") -> None:
        """Marshal an MPS watcher state change onto the notebook event loop."""
        from quantem.widget._folder_watch_status import set_folder_watch_status

        def _apply() -> None:
            if self._mps_folder_shutdown:
                return
            set_folder_watch_status(self, state, detail)

        loop = getattr(self, "_ioloop", None)
        loop_running = False
        if loop is not None:
            try:
                loop_running = bool(loop.asyncio_loop.is_running())
            except Exception:
                loop_running = False
        if loop is not None and loop_running:
            loop.add_callback(_apply)
        else:
            _apply()

    def _shutdown_mps_folder_live(self) -> None:
        """Join folder work and detach callbacks before data cleanup."""
        if self._mps_folder_shutdown:
            return
        live = getattr(self, "_mps_folder_live", None)
        if live is not None:
            watch_started = bool(getattr(live, "_watch_started", False))
            shutdown = getattr(live, "shutdown", None)
            if callable(shutdown):
                shutdown()
            else:
                stop = getattr(live, "stop_watch", None)
                if callable(stop):
                    stop()
            from quantem.widget._folder_watch_status import set_folder_watch_status

            if watch_started:
                set_folder_watch_status(
                    self,
                    "stopped",
                    "Folder watching has stopped.",
                )
            else:
                set_folder_watch_status(self, "hidden", "")
        self._mps_folder_shutdown = True
        multi = getattr(self, "_multi", None)
        if multi is not None:
            multi.on_ready = None

    def free(self):
        """Join MPS folder workers before releasing Metal-backed data."""
        self._shutdown_mps_folder_live()
        return super().free()

    def close(self) -> None:
        """Join MPS folder workers before closing the widget comm."""
        self._shutdown_mps_folder_live()
        super().close()


# =====================================================================
# Back-compat factories. Preferred path is ``Show4DSTEM(load(...))``.
# =====================================================================

def load_4dstem_mps(
    master_path: str,
    *,
    scan_shape=None,
    fast_interaction: bool = True,
    fast_interaction_async: bool = True,
    full_resolution_interaction: bool = False,
    det_bin: int = 1,
    compact: bool = False,
    compact_target_gb: float = _DEFAULT_COMPACT_TARGET_BYTES / 1e9,
    auto_detect_frames: int | None = 64,
    initial_preset: str | None = "BF",
    skip_mps_memory_check: bool | None = None,
    **kwargs,
):
    """Deprecated combo loader+viewer. Use ``Show4DSTEM(load(path, backend='mps'))``."""
    import warnings as _w
    _w.warn(
        "load_4dstem_mps is deprecated. Use "
        "Show4DSTEM(load(path, backend='mps', det_bin=...)) instead.",
        DeprecationWarning, stacklevel=2,
    )
    verbose = kwargs.pop("verbose", True)
    if full_resolution_interaction:
        raise ValueError(
            "full_resolution_interaction has been disabled. Use load(...) and "
            "show_4dstem_mps(...) for fast interaction."
        )
    from quantem.widget.io import load_mps_4dstem

    data = load_mps_4dstem(
        master_path,
        scan_shape=scan_shape,
        verbose=verbose,
        compact=compact,
        compact_target_gb=compact_target_gb,
        det_bin=det_bin,
        skip_mps_memory_check=skip_mps_memory_check,
    )
    return _build_mps_viewer(
        data,
        scan_shape=scan_shape,
        fast_interaction=fast_interaction,
        fast_interaction_async=fast_interaction_async,
        full_resolution_interaction=full_resolution_interaction,
        fast_interaction_verbose=verbose,
        auto_detect_frames=auto_detect_frames,
        initial_preset=initial_preset,
        verbose=verbose,
        **kwargs,
    )


def _build_mps_viewer(
    data,
    *,
    scan_shape=None,
    fast_interaction: bool = True,
    fast_interaction_async: bool = True,
    full_resolution_interaction: bool = False,
    fast_interaction_verbose: bool = True,
    auto_detect_frames: int | None = 64,
    initial_preset: str | None = "BF",
    verbose: bool = True,
    **kwargs,
):
    """Internal MPS viewer builder. Public callers should use
    ``Show4DSTEM(load(path, backend='mps'))`` — this function is used by the
    quantem.widget.__init__ factory + the legacy show_4dstem_mps shim."""
    if full_resolution_interaction:
        raise ValueError(
            "full_resolution_interaction has been disabled. Use load(...) and "
            "show_4dstem_mps(...) for fast interaction."
        )
    if isinstance(data, ChunkedFrames):
        frames = data
    else:
        if hasattr(data, "scan_shape") and scan_shape is None:
            scan_shape = data.scan_shape
        row_prefix = bool(
            getattr(data, "row_prefix", False)
            or getattr(data, "metadata", {}).get("row_prefix", False)
        )
        frames = ChunkedFrames(data, row_prefix=row_prefix)
    if scan_shape is None and hasattr(data, "metadata"):
        scan_shape = data.metadata.get("scan_shape")
    return Show4DSTEMMPS(
        frames,
        scan_shape=scan_shape,
        fast_interaction=fast_interaction,
        fast_interaction_async=fast_interaction_async,
        full_resolution_interaction=full_resolution_interaction,
        fast_interaction_verbose=fast_interaction_verbose,
        auto_detect_frames=auto_detect_frames,
        initial_preset=initial_preset,
        verbose=verbose,
        **kwargs,
    )


def show_4dstem_mps(*args, **kwargs):
    """Deprecated. Use ``Show4DSTEM(load(path, backend='mps'))`` —
    the unified factory auto-detects ChunkedFrames and routes to MetalRawBackend."""
    import warnings as _w
    _w.warn(
        "show_4dstem_mps is deprecated. Use "
        "Show4DSTEM(load(path, backend='mps')) instead.",
        DeprecationWarning, stacklevel=2,
    )
    return _build_mps_viewer(*args, **kwargs)


def _meta_number(meta: dict, *keys: str):
    for key in keys:
        if key not in meta:
            continue
        value = meta[key]
        try:
            arr = np.asarray(value)
            if arr.size == 1:
                return float(arr.reshape(-1)[0])
        except Exception:
            try:
                return float(value)
            except Exception:
                pass
    return None


def Show4DSTEM_MACBOOK(
    data,
    meta: dict | None = None,
    *,
    scan_sampling_A: float | None = None,
    det_sampling_mrad_per_px: float | None = None,
    semiangle_mrad: float | None = None,
    sampling=None,
    units=None,
    **kwargs,
):
    """Build the raw-Metal backend viewer used by ``quantem.widget.Show4DSTEM``."""
    combined_meta = {}
    if hasattr(data, "metadata"):
        combined_meta.update(getattr(data, "metadata", {}) or {})
    if meta:
        combined_meta.update(meta)
    if scan_sampling_A is None:
        scan_sampling_A = _meta_number(combined_meta, "scan_sampling_A",
                                       "scan_sampling", "pixel_size_A")
    if det_sampling_mrad_per_px is None:
        det_sampling_mrad_per_px = _meta_number(
            combined_meta, "det_sampling_mrad_per_px",
            "detector_sampling_mrad_per_px", "k_pixel_size",
        )
    if semiangle_mrad is None:
        semiangle_mrad = _meta_number(combined_meta, "semiangle_mrad", "semiangle")
    if sampling is None and (scan_sampling_A is not None or det_sampling_mrad_per_px is not None):
        sampling = (
            float(scan_sampling_A) if scan_sampling_A is not None else 1.0,
            float(scan_sampling_A) if scan_sampling_A is not None else 1.0,
            float(det_sampling_mrad_per_px) if det_sampling_mrad_per_px is not None else 1.0,
            float(det_sampling_mrad_per_px) if det_sampling_mrad_per_px is not None else 1.0,
        )
    if units is None and sampling is not None:
        units = (
            "Å" if scan_sampling_A is not None else "pixels",
            "Å" if scan_sampling_A is not None else "pixels",
            "mrad" if det_sampling_mrad_per_px is not None else "pixels",
            "mrad" if det_sampling_mrad_per_px is not None else "pixels",
        )
    verbose = bool(kwargs.get("verbose", True))
    viewer = _build_mps_viewer(data, sampling=sampling, units=units, **kwargs)
    inferred_det_sampling = False
    if det_sampling_mrad_per_px is None and semiangle_mrad is not None:
        bf_radius = float(getattr(viewer, "bf_radius", 0) or 0)
        if bf_radius > 0:
            det_sampling_mrad_per_px = float(semiangle_mrad) / bf_radius
            viewer.k_pixel_size = float(det_sampling_mrad_per_px)
            viewer.k_pixel_unit = "mrad"
            inferred_det_sampling = True
    elif det_sampling_mrad_per_px is not None:
        viewer.k_pixel_size = float(det_sampling_mrad_per_px)
        viewer.k_pixel_unit = "mrad"
    if scan_sampling_A is not None:
        viewer.pixel_size = float(scan_sampling_A)
        viewer.pixel_unit = "Å"
    viewer.macbook_sampling = {
        "scan_sampling_A": scan_sampling_A,
        "det_sampling_mrad_per_px": det_sampling_mrad_per_px,
        "semiangle_mrad": semiangle_mrad,
        "det_sampling_inferred_from_bf_radius": inferred_det_sampling,
    }
    if verbose and (scan_sampling_A is not None or det_sampling_mrad_per_px is not None):
        parts = []
        if scan_sampling_A is not None:
            parts.append(f"scan {float(scan_sampling_A):.4g} Å/px")
        if det_sampling_mrad_per_px is not None:
            source = " from BF radius" if inferred_det_sampling else ""
            parts.append(f"detector {float(det_sampling_mrad_per_px):.4g} mrad/px{source}")
        print(f"MacBook sampling: {', '.join(parts)}")
    return viewer


__all__ = [
    "Show4DSTEMMPS",
    "Show4DSTEM_MACBOOK",
    "show_4dstem_mps",
    "load_4dstem_mps",
]
