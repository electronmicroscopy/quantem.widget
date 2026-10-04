"""Show4DSTEM controls over an explicitly attached, pre-existing CUDA owner.

The source executor owns CUDA scheduling. This widget uses CPU control tensors
and complete exact reduced batches; it never constructs a substitute 4D array.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import time

import numpy as np

from quantem.gpu.detector import prepare
from quantem.gpu.io._resident import CudaResidentSource
from quantem.widget.show4dstem import Show4DSTEM


def _owned_payload(array: np.ndarray) -> bytes:
    """Reuse the immutable complete host result instead of copying it again."""
    owner = array
    while isinstance(owner, np.ndarray):
        owner = owner.base
    if isinstance(owner, bytes) and len(owner) == array.nbytes:
        return owner
    return array.tobytes()


class _ResidentShow4DSTEM(Show4DSTEM):
    """Private routing target; scientists continue to call Show4DSTEM(data)."""

    def __init__(self, data: CudaResidentSource, stream: bool = True, **kwargs):
        if not isinstance(data, CudaResidentSource):
            raise TypeError("Open the complete resident source, not a materialized acquisition view.")
        if kwargs.get("backend") is not None:
            raise ValueError("The resident source already owns its CUDA backend; omit backend=.")
        if kwargs.get("offline") not in (None, False) or kwargs.get("data_url"):
            raise ValueError("A compact CUDA owner requires a live session; offline source export is not implemented.")
        self._resident_source = data
        self._resident_session = prepare(data)
        self._resident_images: np.ndarray | None = None
        self._resident_patterns: np.ndarray | None = None
        self._resident_mask_key: tuple | None = None
        self._resident_scan: int | None = None
        self._resident_sequence = 0
        self._resident_area = 0
        kwargs.setdefault("view_mode", "multiple")
        kwargs.setdefault("compare_max_panels", data.shape[0])
        kwargs.setdefault("compare_group_mode", "all")
        kwargs.setdefault("compare_dp_mode", "selected")
        kwargs.setdefault("center", tuple((size - 1) / 2 for size in data.shape[-2:]))
        kwargs.setdefault("bf_radius", min(data.shape[-2:]) / 8)
        kwargs["precompute_virtual_images"] = False
        kwargs["offline"] = False
        self._resident_stream = None
        super().__init__(data, **kwargs)
        if stream:
            # Complete batches bypass the notebook comm (5 -> 60 paints/s for seven
            # 512x512 panels on one machine). Trait delivery remains the fallback
            # until the worker connects.
            from quantem.widget.show4dstem_resident_stream import ResidentBatchStream
            self._resident_stream = ResidentBatchStream(self)
            self.resident_stream = dict(enabled=True, url=self._resident_stream.url, generation=data.generation)

    def _current_detector_mask(self):
        rows, cols = np.indices(self._resident_source.shape[-2:], dtype=np.float64)
        row, col = float(self.roi_center_row), float(self.roi_center_col)
        outer, inner = float(self.roi_radius), float(self.roi_radius_inner)
        if not np.isfinite([row, col, outer, inner, self.roi_width, self.roi_height]).all():
            raise ValueError("Use finite detector coordinates and radii.")
        if self.roi_mode in ("circle", "annular"):
            inner = inner if self.roi_mode == "annular" else 0.0
            if not 0 <= inner <= outer:
                raise ValueError("Use detector radii with 0 <= inner <= outer.")
            # Native owner semantics use inclusive float64 hypot, including
            # translated/fractional/empty/boundary annuli.
            distance = np.hypot(rows - row, cols - col)
            value = (distance >= inner) & (distance <= outer)
        elif self.roi_mode == "square":
            if outer < 0:
                raise ValueError("Use a nonnegative square half-width.")
            value = (np.abs(rows-row) <= outer) & (np.abs(cols-col) <= outer)
        elif self.roi_mode == "rect":
            if self.roi_width < 0 or self.roi_height < 0:
                raise ValueError("Use nonnegative rectangle dimensions.")
            value = (np.abs(rows-row) <= self.roi_height/2) & (np.abs(cols-col) <= self.roi_width/2)
        elif self.roi_mode == "point":
            value = np.zeros(rows.shape, np.bool_)
            value[int(np.clip(round(row), 0, rows.shape[0]-1)), int(np.clip(round(col), 0, rows.shape[1]-1))] = True
        else:
            raise NotImplementedError(f"The resident owner does not support ROI mode {self.roi_mode!r}.")
        return self._resident_source.mask(value)

    def _ensure_resident_images(self):
        mask = self._current_detector_mask()
        key = mask.tobytes()
        request_key = (key, self.roi_mode, float(self.roi_center_row), float(self.roi_center_col),
                       float(self.roi_radius_inner), float(self.roi_radius),
                       float(self.roi_width), float(self.roi_height), self._resident_source.generation)
        if request_key != self._resident_mask_key:
            if self._resident_sequence >= 2**53 - 1:
                raise RuntimeError("Start a new display generation before exhausting browser request IDs.")
            before = time.perf_counter()
            images = self._resident_session.masked_sum_batch_exact(mask)
            sequence = self._resident_sequence + 1
            metadata = dict(
                dtype="<u4", shape=list(images.shape), bytes=images.nbytes,
                request_id=sequence, generation=self._resident_source.generation,
                **self._resident_source.storage_metadata,
                mask_area=int(mask.sum()), detector_mask_sha256=hashlib.sha256(key).hexdigest(),
                center=[float(self.roi_center_row), float(self.roi_center_col)],
                inner=float(self.roi_radius_inner) if self.roi_mode == "annular" else 0.0,
                outer=float(self.roi_radius), producer_wall_ms=(time.perf_counter()-before)*1000,
            )
            remote = getattr(self._resident_source, "remote_owner_metadata", None)
            if remote is not None:
                metadata["source_location"] = {
                    key: remote[key] for key in
                    ("pid", "hostname", "device", "physical_gpu_uuid", "physical_gpu_index", "source_location")
                }
            # Publish only after the complete exact source result succeeds.
            self._resident_images = images
            self._resident_mask_key, self._resident_sequence = request_key, sequence
            self._resident_area = metadata["mask_area"]
            stream = self._resident_stream
            if stream is not None and stream.connected:
                panels = images.shape[0]
                if self.compare_panel_count != panels or list(self.compare_panel_indices) != list(range(panels)):
                    with self.hold_sync():
                        self.compare_panel_count = panels
                        self.compare_panel_indices = list(range(panels))
                        self.compare_status = f"{panels} exact resident acquisitions"
                stream.publish(metadata, _owned_payload(images))
            else:
                with self.hold_sync(), self.hold_trait_notifications():
                    self.compare_virtual_image_bytes = _owned_payload(images)
                    self.resident_batch_info = metadata
                    self.compare_panel_count = images.shape[0]
                    self.compare_panel_indices = list(range(images.shape[0]))
                    self.compare_status = f"{images.shape[0]} exact resident acquisitions"
        return self._resident_images

    def _selected_display(self):
        images = self._ensure_resident_images()
        if self._resident_area == 0:
            return np.zeros(images.shape[1:], np.float32)
        return (images[int(self.frame_idx)].astype(np.float64) / self._resident_area).astype(np.float32)

    def _compute_virtual_image_from_roi(self):
        if self.vi_source != "roi":
            raise NotImplementedError("Computed product maps are not attached to this compact owner.")
        self.virtual_image_bytes = self._selected_display().tobytes()

    def _refresh_compare_virtual_images(self):
        if getattr(self, "_suppress_roi_recompute", False):
            return
        self._ensure_resident_images()
        # The inspection panel uses derived float display values. The complete
        # raw batch remains available for integer readouts and scientific export.
        self.virtual_image_bytes = self._selected_display().tobytes()

    def _ensure_resident_patterns(self, index: int | None = None):
        if index is None:
            index = int(self.pos_row) * self.shape_cols + int(self.pos_col)
        if self._resident_scan != index:
            patterns = self._resident_session.frame_batch(index)
            self._resident_patterns, self._resident_scan = patterns, index
        return self._resident_patterns

    def _diffraction_frame_for_index(self, frame_idx: int):
        return self._ensure_resident_patterns()[int(frame_idx)]

    def _get_frame(self, row: int, col: int):
        return self._ensure_resident_patterns(row*self.shape_cols+col)[int(self.frame_idx)]

    def _average_compare_diffraction_frame(self):
        patterns = self._ensure_resident_patterns()
        indices = self._compare_current_page_indices()
        if not indices:
            raise ValueError("Select at least one acquisition to average its point DP.")
        return (patterns[np.asarray(indices)].sum(axis=0, dtype=np.uint64) / len(indices)).astype(np.float32)

    def _update_frame(self, change=None):
        patterns = self._ensure_resident_patterns()
        if self.view_mode == "multiple" and self.compare_dp_mode == "average":
            frame = self._average_compare_diffraction_frame()
        else:
            frame = patterns[int(self.frame_idx)]
        # The displayed pattern excludes the authenticated invalid detector pixels
        # (hardware sentinel 65535), as every displayed product already does;
        # otherwise four sentinels set the colour range and the beam is invisible.
        # frame_batch still returns the raw counts.
        frame = np.where(self._resident_source.valid_pixels, frame, 0)
        self.frame_bytes = np.ascontiguousarray(frame, dtype=np.float32).tobytes()

    def _virtual_image_for_frame(self, frame_idx):
        return self._ensure_resident_images()[int(frame_idx)]

    def _fast_masked_sum(self, mask):
        raise NotImplementedError("Use the complete resident batch operation; per-acquisition compute is disabled.")

    def _compute_vi_roi_dp(self):
        raise NotImplementedError("Multi-position DP reductions are not implemented for this compact owner.")

    def auto_detect_center(self, update_roi=True):
        raise NotImplementedError("Full-scan auto-probe reduction is not implemented for this compact owner; set center and bf_radius explicitly.")

    def _format_gpu_memory_label(self):
        source = self._resident_source
        remote = getattr(source, "remote_owner_metadata", None)
        if remote is not None:
            return (f"{source.shape[0]} acquisitions | {source.nbytes/2**30:.2f} GiB compact | "
                    f"remote {remote['hostname']} GPU{remote['physical_gpu_index']} "
                    f"({source.device}), PID {remote['pid']}")
        return f"{source.shape[0]} acquisitions | {source.nbytes/2**30:.2f} GiB compact | {source.device}"

    def _pack_offline(self, offline, data_url=None):
        if offline or data_url:
            raise NotImplementedError("Save the compact source archive; full offline materialization is unsupported.")

    def _export_data_array(self, **kwargs):
        raise NotImplementedError("Full compact-source materialization is unsupported. Export exact reduced images with save_virtual_images().")

    def save_virtual_images(self, path: str | Path) -> Path:
        """Save the current complete exact uint32 batch as a NumPy .npy file.

        The saved values are counts, before mask-area normalization, contrast,
        logarithms or color mapping. This is not a source-data archive.

        Parameters
        ----------
        path
            New ``.npy`` file to create. Existing files are never overwritten.

        Returns
        -------
        pathlib.Path
            Path to the complete current acquisition batch.

        Examples
        --------
        >>> viewer.save_virtual_images("adf_counts.npy")
        """
        path = Path(path)
        if path.suffix != ".npy":
            raise ValueError("Use a .npy path for exact uint32 virtual-image export.")
        with path.open("xb") as stream:
            np.save(stream, self._ensure_resident_images(), allow_pickle=False)
        return path
