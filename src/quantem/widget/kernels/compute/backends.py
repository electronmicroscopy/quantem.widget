"""4D-STEM compute backends — an Apple Silicon MacBook (raw Metal) + universal torch.

ONE compute layer consumed by BOTH the Jupyter widget (``show4dstem``) AND the
web Browse (``server/routers/browse.py``). The same masked-sum math is
implemented three ways across the repo (torch tensordot, raw Metal, CuPy
RawKernel); this module is the single interface they collapse into.

Backends conform to the ``ComputeBackend`` protocol (see ``backend.py``):

    TorchBackend       — torch tensor on CUDA / MPS / CPU — universal default
    MetalRawBackend    — ChunkedFrames + MetalVirtualImage — a 19.3 GB
                         large-no-bin-class no-bin path where torch.MPS overflows.
                         Owns fast_vi sidecar, radial cache, multi-dataset
                         proxy lifecycle (see capabilities tuple).
    CudaKernelCompute  — placeholder; web Browse RawKernel will collapse here.

``TorchCompute`` / ``MetalCompute`` are kept as aliases for one release.

``compute_backend(data)`` duck-types the data source and returns the right
backend so callers (widget + web Browse) never branch on hardware themselves.
"""
from __future__ import annotations

import threading

import numpy as np

from quantem.widget.kernels.compute.backend import ComputeBackend  # noqa: F401

# Cap transient float32 memory per reduction chunk (matches the widget budget).
_CHUNK_BYTE_BUDGET = 600 * 1024 * 1024


def compute_backend(data):
    """Return the compute backend for ``data``, duck-typed on its type.

    torch tensor / numpy / Dataset wrapping a tensor -> TorchBackend (any
        torch device: CUDA / MPS-binned / CPU). This is the GENERAL path.
    ChunkedFrames / anything with ``_is_gpu_frames`` -> MetalRawBackend
        (raw Metal). The device-specific path, used ONLY for MPS no-bin
        (where torch can't hold the >2^31-element stack).
    cupy ndarray -> converted to a torch CUDA tensor (zero-copy dlpack) and
        run on TorchBackend. The widget compute path is torch, never cupy;
        cupy lives only in the io decode + the parity-test reference.

    One selection point here means callers (widget + web Browse) never branch
    on hardware themselves.
    """
    if getattr(data, "_is_gpu_frames", False):
        return MetalRawBackend(data)
    cls_name = type(data).__module__.split(".")[0]
    if cls_name == "cupy":
        import torch
        return TorchBackend(torch.from_dlpack(data))  # cupy -> torch CUDA, no cupy compute
    try:
        import torch
        if isinstance(data, torch.Tensor):
            return TorchBackend(data)
    except ImportError:
        pass
    # Dataset-like: unwrap a torch tensor if present, else hand to TorchBackend to
    # numpy-ify (it owns the conversion so the widget doesn't have to).
    return TorchBackend(data)


# ---


class TorchBackend:
    """Torch backend - one chunked path on CUDA / MPS / CPU.

    Ports the widget's `_fast_masked_sum` / `auto_detect_center` / `_compute_vi_roi_dp`
    math verbatim (chunked tensordot, int64 mean-DP, einsum/amax reduce) so the
    universal-device path is identical to today. uint16 stays integer until the
    small reduced output; the per-chunk float32 cast is bounded by the byte budget.

    Conforms to ``ComputeBackend`` protocol. No optional capabilities — torch
    runs the universal path.
    """

    capabilities: tuple[str, ...] = ()

    def __init__(self, data, *, scan_shape=None, det_shape=None, device=None):
        import torch
        self.torch = torch
        tensor = data if isinstance(data, torch.Tensor) else torch.as_tensor(np.asarray(data))
        if device is not None:
            tensor = tensor.to(device)
        self._t = tensor
        if tensor.ndim == 4:
            sr, sc, dr, dc = tensor.shape
        elif tensor.ndim == 3:
            n, dr, dc = tensor.shape
            if scan_shape is None:
                sr = int(round(n ** 0.5))
                sc = n // sr
            else:
                sr, sc = scan_shape
        else:
            raise ValueError(f"expected 3D/4D tensor, got {tuple(tensor.shape)}")
        self.scan_shape = (int(sr), int(sc))
        self.det_shape = (int(dr), int(dc))
        self.n_frames = int(sr) * int(sc)
        self.device = tensor.device
        self._4d = tensor.reshape(self.scan_shape[0], self.scan_shape[1], dr, dc)
        self._flat = tensor.reshape(-1, dr, dc)
        self._row = torch.arange(dr, device=self.device, dtype=torch.float32)[:, None]
        self._col = torch.arange(dc, device=self.device, dtype=torch.float32)[None, :]

    def _chunk_rows(self) -> int:
        bytes_per_row = self.scan_shape[1] * self.det_shape[0] * self.det_shape[1] * 4
        return max(1, _CHUNK_BYTE_BUDGET // max(1, bytes_per_row))

    def frame(self, idx: int) -> np.ndarray:
        return self._flat[int(idx)].cpu().numpy()

    def masked_sum(self, det_mask: np.ndarray) -> np.ndarray:
        """Virtual image: sum masked detector pixels per scan position (chunked)."""
        torch = self.torch
        mask = torch.as_tensor(np.ascontiguousarray(det_mask), device=self.device).float()
        out = torch.zeros(self.scan_shape, dtype=torch.float32, device=self.device)
        step = self._chunk_rows()
        for i in range(0, self.scan_shape[0], step):
            chunk = self._4d[i:i + step]
            if chunk.dtype != torch.float32:
                chunk = chunk.float()   # int OR float64 -> float32; never compute in 64-bit (mask is float32)
            out[i:i + step] = torch.tensordot(chunk, mask, dims=([2, 3], [0, 1]))
        return out.cpu().numpy()

    def mean_dp(self) -> np.ndarray:
        """Mean DP over all scan positions - int64 accumulate, float only at output.

        The chunk size accounts for the int64 dtype cast that ``.sum(dtype=int64)``
        materializes internally. Otherwise a uint16 chunk expands 4× (to int64)
        during the sum, holding an 8+ GB transient on 192² detectors that
        oversubscribes 24 GB cards.
        """
        torch = self.torch
        acc = torch.zeros(self.det_shape, dtype=torch.int64, device=self.device)
        # Budget 128 MB of int64 transient per chunk (was 1 GB base, but the
        # int64 cast during sum() multiplies by 8/element vs uint16's 2).
        det_pixels = max(1, self.det_shape[0] * self.det_shape[1])
        step = max(1, (1 << 27) // (det_pixels * 8))
        for i in range(0, self.n_frames, step):
            acc += self._flat[i:i + step].sum(dim=0, dtype=torch.int64)
        return (acc.float() / self.n_frames).cpu().numpy()

    def reduce_frames(self, scan_indices: np.ndarray, reduce: str = "mean") -> np.ndarray:
        """Summed / mean / max DP over a set of scan positions (flat indices)."""
        torch = self.torch
        idx = torch.as_tensor(np.asarray(scan_indices, dtype=np.int64), device=self.device)
        frames = self._flat.index_select(0, idx).float()
        if reduce == "sum":
            dp = frames.sum(dim=0)
        elif reduce == "max":
            dp = frames.amax(dim=0)
        else:
            dp = frames.mean(dim=0)
        return dp.cpu().numpy()

    def center_of_mass(self, det_mask: np.ndarray | None = None):
        """Per-scan-position CoM over the (masked) detector - the DPC vector field.

        Returns ``(com_col, com_row)`` each ``(N,)`` float32 in absolute detector
        coordinates (col = Sum col*I / Sum I, row = Sum row*I / Sum I), matching
        ``MetalVirtualImage.center_of_mass`` so DPC is single-source across
        CUDA / MPS / CPU. ``det_mask`` None means the full detector. Chunked by the
        same byte budget as ``masked_sum``; integer frames stay int until the small
        per-chunk float reduce.
        """
        torch = self.torch
        mask = None
        if det_mask is not None:
            mask = torch.as_tensor(np.ascontiguousarray(det_mask), device=self.device).float()
        com_col = torch.zeros(self.n_frames, dtype=torch.float32, device=self.device)
        com_row = torch.zeros(self.n_frames, dtype=torch.float32, device=self.device)
        sc = self.scan_shape[1]
        step = self._chunk_rows()
        for i in range(0, self.scan_shape[0], step):
            chunk = self._4d[i:i + step]
            if chunk.dtype != torch.float32:
                chunk = chunk.float()   # int OR float64 -> float32; never compute in 64-bit (mask is float32)
            if mask is not None:
                chunk = chunk * mask
            denom = chunk.sum(dim=(2, 3))
            sum_row = (chunk * self._row).sum(dim=(2, 3))
            sum_col = (chunk * self._col).sum(dim=(2, 3))
            safe = denom.clamp(min=1e-12)  # empty / masked-out frames -> CoM 0, no div0
            lo = i * sc
            com_row[lo:lo + sum_row.numel()] = (sum_row / safe).reshape(-1)
            com_col[lo:lo + sum_col.numel()] = (sum_col / safe).reshape(-1)
        return com_col.cpu().numpy(), com_row.cpu().numpy()


# ---


class MetalRawBackend:
    """Raw-Metal backend - wraps ``MetalVirtualImage`` over ``ChunkedFrames``.

    Owns the MPS lifecycle hooks that the Show4DSTEMMPS widget subclass used to
    drive directly. Capabilities: fast_sidecar (bin2 fast_vi), radial_cache
    (exact no-bin row-prefix BF/ADF), multi_dataset (lazy multi-file proxy).
    See ``backend.py`` for the protocol; ``Show4DSTEMMPS`` reads
    ``backend.capabilities`` and calls the corresponding methods only when the
    feature is supported.

    VIRTUAL-IMAGE BINNING CONTRACT (MPS) — the design, stated plainly:
      - det_bin == 1 (NO-BIN): detector stays full-res (e.g. 192x192) so a single
        diffraction frame (CBED) keeps full angular resolution. The VIRTUAL IMAGE
        is a masked_sum over ALL frames; at full-res that is bandwidth-bound
        (~40 GB/s scattered uint16 -> ~8-10 fps). So we AUTO-build a bin2 (96x96)
        copy of the frames in the background -- `fast_vi`, a.k.a. the "sidecar" --
        and compute the virtual image on it: 4x fewer pixels to read => real-time.
        Full-res `vi` is still used for the single-frame CBED. The bin2 buffer is
        NOT optional for speed: binning the mask alone still reads all 192x192; the
        speedup comes only from reading the 4x-smaller bin2 buffer.
      - det_bin >= 2: loaded data is ALREADY binned (e.g. 96x96), so the VI
        masked_sum is fast directly and NO sidecar is built (`_auto_fast` is gated
        on det_bin == 1). Simpler path: bin at load = fast VI + no extra buffer, at
        the cost of CBED angular detail.
    Net: no-bin auto-bins by 2 for the VIRTUAL IMAGE only; det_bin=2 needs none.
    (Also preserves row-prefix exact reductions + the lazy multi-dataset container.)
    """

    @property
    def capabilities(self) -> tuple[str, ...]:
        """Capabilities advertised by this MetalRaw backend.

        - ``fast_sidecar`` always (bin2 sidecar available when det_bin==1, or
          immediately ready for already-binned data).
        - ``radial_cache`` only when the underlying ChunkedFrames had row_prefix
          enabled at load time (``data.vi.row_prefix_enabled``).
        - ``multi_dataset`` only when the data source is a ``MultiChunkedFrames``
          proxy (has ``set_active`` + ``on_ready``).
        """
        caps = ["fast_sidecar"]
        if getattr(self._cf, "vi", None) is not None and getattr(
            self._cf.vi, "row_prefix_enabled", False
        ):
            caps.append("radial_cache")
            caps.append("row_prefix_exact")
        if hasattr(self._cf, "set_active") and hasattr(self._cf, "on_ready"):
            caps.append("multi_dataset")
        return tuple(caps)

    def __init__(self, frames):
        self._cf = frames  # ChunkedFrames (or MultiChunkedFrames, duck-types the same)
        det = tuple(int(x) for x in self._cf.vi.det)
        n = int(self._cf._n)
        sr = int(round(n ** 0.5))
        self.scan_shape = (sr, n // sr)
        self.det_shape = det
        self.n_frames = n
        self.device = "mps"
        self.det_bin = int(getattr(self._cf, "det_bin", 1))
        # Auto fast-mode: on a big NO-BIN detector, full-res masked_sum is ~8-10 fps
        # (40 GB/s scattered uint16). Build a bin2 sidecar (96x96) in the background
        # so interaction jumps to real-time once ready; serve full-res until then.
        # Already-binned data (det_bin>1) is small enough - no sidecar.
        self._com_cache = None  # full-detector CoM (com_col, com_row), eager-built below
        self._auto_fast = (self.det_bin == 1 and det[0] >= 96
                           and hasattr(self._cf, "ensure_fast_interaction"))
        # Background radial-cache lifecycle. Only matters when row_prefix is on.
        self._radial_thread: threading.Thread | None = None
        self._radial_pending: tuple[float, float] | None = None
        self._radial_request = 0
        self._radial_building = False
        self._radial_error: str | None = None
        if self._auto_fast and getattr(self._cf, "fast_vi", None) is None:
            threading.Thread(target=self._build_fast, daemon=True).start()

    def _build_fast(self):
        try:
            self._cf.ensure_fast_interaction(verbose=False)
            # Eager-cache the full-detector CoM on the bin2 sidecar so the FIRST DPC
            # click is instant (cached), the same way BF rides the prebuilt sidecar.
            self._com_cache = self.center_of_mass()
        except Exception:
            pass  # fall back to full-res; interaction just stays at the no-bin rate

    @property
    def has_fast(self) -> bool:
        return getattr(self._cf, "fast_vi", None) is not None

    def frame(self, idx: int) -> np.ndarray:
        return self._cf.frame(int(idx))

    def masked_sum(self, det_mask: np.ndarray) -> np.ndarray:
        cf = self._cf
        fv = getattr(cf, "fast_vi", None)
        if self._auto_fast and fv is not None:
            # bin2 sidecar ready: downsample the detector mask + reduce on 96x96
            # (4x fewer pixels = real-time). The scan-space output shape is unchanged.
            from quantem.widget.kernels.compute.mps import _bin2_mask
            vi = np.asarray(fv.masked_sum(_bin2_mask(np.ascontiguousarray(det_mask))))
        else:
            vi = np.asarray(cf.vi.masked_sum(np.ascontiguousarray(det_mask)))
        return vi.reshape(self.scan_shape).astype(np.float32, copy=False)

    def mean_dp(self) -> np.ndarray:
        return np.asarray(self._cf.vi.detector_sum(), dtype=np.float32) / self.n_frames

    def reduce_frames(self, scan_indices: np.ndarray, reduce: str = "mean") -> np.ndarray:
        idx = np.asarray(scan_indices, dtype=np.uint32)
        if reduce == "mean":
            return np.asarray(self._cf.vi.mean_frames(idx), dtype=np.float32)
        # sum / max: mean_frames gives the average; scale for sum, fall back to
        # per-frame max (rare path - the widget's max accumulates frames directly).
        if reduce == "sum":
            return np.asarray(self._cf.vi.mean_frames(idx), dtype=np.float32) * len(idx)
        dp = None
        for i in idx:
            f = np.asarray(self._cf.frame(int(i)), dtype=np.float32)
            dp = f if dp is None else np.maximum(dp, f)
        return dp if dp is not None else np.zeros(self.det_shape, dtype=np.float32)

    def center_of_mass(self, det_mask: np.ndarray | None = None):
        """Per-scan-position CoM (DPC vector field) on the raw Metal kernel.

        Uses the bin2 sidecar (``fast_vi``) when ready - same fast path as
        ``masked_sum`` - so no-bin DPC is real-time instead of an ~8 s full-res 192^2
        pass; result is eager-cached in ``_build_fast`` so the first DPC click is
        instant. The bin2 detector halves the CoM coordinate scale, so multiply by 2
        to return absolute full-res detector px; the constant half-pixel bin offset
        cancels under the DPC zero-mean. Falls back to full-res ``vi`` until the
        sidecar builds. Returns ``(com_col, com_row)`` flat ``(N,)`` float32 - the
        same contract as ``TorchCompute`` so DPC is single-source across backends.
        """
        if det_mask is None and self._com_cache is not None:
            return self._com_cache  # eager-built in _build_fast -> instant DPC
        cf = self._cf
        fv = getattr(cf, "fast_vi", None)
        if self._auto_fast and fv is not None:
            from quantem.widget.kernels.compute.mps import _bin2_mask
            m = None if det_mask is None else _bin2_mask(np.ascontiguousarray(det_mask))
            cc, cr = fv.center_of_mass(m)
            return cc * 2.0, cr * 2.0  # bin2 px -> full-res detector px
        mask = None if det_mask is None else np.ascontiguousarray(det_mask)
        return cf.vi.center_of_mass(mask)

    # ---------------------------------------------------------------- fast_sidecar
    # bin2 sidecar (``fast_vi``) — accelerates BF/DF/ADF masked_sum 4x by
    # downsampling the detector once at sidecar-build time, then reading the
    # 4x-smaller buffer on every reduction. Auto-built for no-bin data; already
    # ready for det_bin>=2 data. Show4DSTEMMPS used to drive this; now the
    # backend owns it.

    @property
    def fast_bin(self) -> int:
        return int(getattr(self._cf, "fast_bin", 2))

    def ensure_fast_sidecar(self, verbose: bool = False) -> bool:
        """Block until the bin2 fast_vi sidecar is ready. Returns True if
        ready or no sidecar is needed (already-binned data)."""
        cf = self._cf
        if int(getattr(cf, "det_bin", 1)) > 1:
            return True  # already binned at load — no sidecar needed
        if not hasattr(cf, "ensure_fast_interaction"):
            return False
        cf.ensure_fast_interaction(verbose=verbose)
        return getattr(cf, "fast_vi", None) is not None

    def cache_fast_presets(self, masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Pre-compute virtual images on the fast sidecar for a dict of named
        detector masks (typ. {"bf": mask, "abf": mask, ...}). Returns a dict
        of `(scan_r, scan_c) float32` arrays. Caller stores bytes."""
        cf = self._cf
        fv = getattr(cf, "fast_vi", None)
        if fv is None:
            return {}
        from quantem.widget.kernels.compute.mps import _bin_mask
        out: dict[str, np.ndarray] = {}
        for name, mask in masks.items():
            vi = fv.masked_sum(_bin_mask(np.ascontiguousarray(mask), self.fast_bin))
            out[name] = np.asarray(vi).reshape(self.scan_shape).astype(np.float32, copy=False)
        return out

    # ---------------------------------------------------------------- radial_cache
    # Exact no-bin BF/ADF on circular/annular detector masks via row-prefix
    # cumulative sums. Only available when ChunkedFrames was loaded with
    # ``row_prefix=True``. Cache is per-(center_row, center_col); building
    # touches the whole 19 GB stack, so we serialize requests + cancel stale
    # ones (operator drags the center, we only build for the last position).

    def radial_cache_ready(self, center_row: float, center_col: float) -> bool:
        vi = getattr(self._cf, "vi", None)
        if vi is None or not getattr(vi, "row_prefix_enabled", False):
            return False
        return vi.radial_cache_ready(float(center_row), float(center_col))

    def radial_masked_sum(
        self,
        *,
        center_row: float,
        center_col: float,
        outer_radius: float,
        inner_radius: float = 0.0,
        build: bool = False,
    ) -> np.ndarray | None:
        """Returns a (scan_r, scan_c) virtual image or None if the cache isn't
        ready (and ``build`` is False)."""
        vi = getattr(self._cf, "vi", None)
        if vi is None or not getattr(vi, "row_prefix_enabled", False):
            return None
        return vi.radial_masked_sum(
            center_row=float(center_row),
            center_col=float(center_col),
            outer_radius=float(outer_radius),
            inner_radius=float(inner_radius),
            build=bool(build),
        )

    def ensure_radial_cache(self, center_row: float, center_col: float,
                            *, idle_delay_s: float = 0.75) -> None:
        """Schedule a background build of the radial cache at (row, col).

        Cancels any prior pending build for a different center. Cheap if the
        cache is already ready at this center.
        """
        vi = getattr(self._cf, "vi", None)
        if vi is None or not getattr(vi, "row_prefix_enabled", False):
            return
        if vi.radial_cache_ready(float(center_row), float(center_col)):
            self._radial_pending = None
            return
        self._radial_request += 1
        self._radial_pending = (float(center_row), float(center_col))
        if self._radial_building:
            return  # an existing thread will pick up the new pending center
        self._radial_building = True
        self._radial_error = None
        import time as _time

        def _build():
            try:
                while True:
                    request = self._radial_request
                    center = self._radial_pending
                    if center is None:
                        return
                    _time.sleep(idle_delay_s)
                    # If the request changed during the idle wait, restart on the new center.
                    if request != self._radial_request or center != self._radial_pending:
                        continue
                    vi._ensure_radial_cache(center[0], center[1])
                    if request == self._radial_request and center == self._radial_pending:
                        self._radial_pending = None
                        return
            except Exception as exc:  # pragma: no cover
                self._radial_error = repr(exc)
            finally:
                self._radial_building = False

        self._radial_thread = threading.Thread(
            target=_build, name="MetalRawBackend-radial", daemon=True,
        )
        self._radial_thread.start()

    @property
    def radial_building(self) -> bool:
        return self._radial_building

    @property
    def radial_error(self) -> str | None:
        return self._radial_error

    # ---------------------------------------------------------------- multi_dataset
    # Lazy multi-file proxy (MultiChunkedFrames) — set_active(idx) points the
    # backend at one of N decoded datasets; on_ready(idx) fires when the
    # background decoder finishes a dataset. Show4DSTEMMPS uses these to drive
    # the n_frames slider for time/tilt series.

    def set_active_dataset(self, idx: int) -> None:
        if hasattr(self._cf, "set_active"):
            self._cf.set_active(int(idx))

    @property
    def multi_n_ready(self) -> int:
        return int(getattr(self._cf, "n_ready", 1))

    @property
    def multi_names(self) -> list[str]:
        return list(getattr(self._cf, "names", []) or [])

    @property
    def multi_active_idx(self) -> int:
        return int(getattr(self._cf, "active_idx", 0))

    def multi_total(self) -> int:
        datasets = getattr(self._cf, "datasets", None)
        return len(datasets) if datasets is not None else 1

    def set_multi_ready_callback(self, cb) -> None:
        if hasattr(self._cf, "on_ready"):
            self._cf.on_ready = cb


# Back-compat aliases (one-release deprecation).
TorchCompute = TorchBackend
MetalCompute = MetalRawBackend


# ---


class CudaKernelCompute:
    """CuPy backend - the web Browse fused RawKernel path (designed-for, not yet wired).

    Placeholder so `compute_backend(cupy_array)` resolves; the real implementation
    moves `server/routers/browse.py:_vi_mask_kernel` here so the web Browse and the
    widget share one CUDA masked-sum. Until then this raises a clear error.
    """

    def __init__(self, data):
        self._data = data
        raise NotImplementedError(
            "CudaKernelCompute is reserved for folding the web Browse RawKernel into "
            "the shared compute layer (see docs/2026-06-01-show4dstem-compute-backends.md). "
            "Use TorchCompute for CUDA today."
        )
