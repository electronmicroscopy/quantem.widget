"""MPS (Apple Metal GPU) bitshuffle+LZ4 decompress backend for arina 4D-STEM.

The Apple-GPU view/screen path for ``quantem.widget.io.load(backend='mps')`` on a
Mac (every modern MacBook has an Apple GPU, so this — not cpu — is the real
off-NVIDIA fallback). Metal compute shaders decompress directly into
unified-memory numpy arrays.

Ported from quantem.widget's tested ``kernels/arina_mps.py`` (the Metal kernel
+ MPSDecompressor are verbatim). quantem.live deliberately differs in ONE place:
detector binning. The widget GPU-bins to float32 (mean); quantem.live keeps the
native **uint16** dtype and integer-sum bins host-side via the shared cpu helper,
so an mps load matches a cuda load bit-for-bit (same dtype, same reduction).

This module imports ``Metal`` + ``numba`` at top — Mac-only deps. It is imported
lazily by ``load()`` only after the backend resolves to 'mps', so importing
quantem.widget.io on Linux never touches it.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
import warnings
import Metal
import h5py
import hdf5plugin  # noqa: F401 - registers bitshuffle filter
import numpy as np
from numba import njit, prange
from tqdm.auto import tqdm

__all__ = [
    "MPSChunked4DSTEM",
    "MPSMasterPlan",
    "clear_mps_cache",
    "load_arina",
    "load_master",
    "load_master_chunked",
    "load_master_torch",
    "load_mps_4dstem",
    "plan_master",
]


@dataclass(frozen=True)
class MPSMasterPlan:
    """Resolved Arina master layout for the Metal chunked loader."""

    master_path: str
    detector_shape: tuple[int, int]
    dtype: np.dtype
    ntrigger: int
    chunk_files: tuple[str, ...]
    chunk_n_frames: tuple[int, ...]

    @property
    def frame_bytes(self) -> int:
        return int(np.prod(self.detector_shape) * self.dtype.itemsize)

    @property
    def elem_size(self) -> int:
        return int(self.dtype.itemsize)

    @property
    def n_blocks_per_frame(self) -> int:
        return (self.frame_bytes + 8191) // 8192

    @property
    def total_frames(self) -> int:
        return int(sum(self.chunk_n_frames))

    @property
    def total_bytes(self) -> int:
        return self.total_frames * self.frame_bytes


@dataclass
class MPSChunked4DSTEM:
    """Zero-copy MPS IO result for full no-bin 4D-STEM viewing.

    ``chunks`` are numpy views over unified-memory Metal buffers. Widget code
    consumes them directly; callers should not concatenate them unless they
    intentionally want a host-side copy.
    """

    chunks: list
    metadata: dict
    master_path: str
    scan_shape: tuple[int, int] | None = None
    row_prefix: bool = False
    det_bin: int = 1
    fast_chunks: list | None = None
    fast_det_bin: int | None = None

    @property
    def detector_shape(self) -> tuple[int, int]:
        return tuple(int(x) for x in self.chunks[0].shape[1:])

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.chunks[0].dtype)

    @property
    def n_frames(self) -> int:
        return int(sum(int(c.shape[0]) for c in self.chunks))

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.n_frames, *self.detector_shape)

    @property
    def nbytes(self) -> int:
        return int(sum(int(c.nbytes) for c in self.chunks))


@dataclass(frozen=True)
class _ChunkReadPlan:
    n_frames: int
    frame_shape: tuple[int, int]
    dtype: np.dtype
    file_offsets: np.ndarray
    sizes: np.ndarray
    out_offsets: np.ndarray
    run_start: np.ndarray
    run_end: np.ndarray
    total_bytes: int


_master_plan_cache: dict[tuple[str, int, int], MPSMasterPlan] = {}
_chunk_read_plan_cache: dict[tuple[str, int, int], _ChunkReadPlan] = {}
_cached_dec = None
_cached_dec_key = None
_MPS_SAFE_WORKING_SET_FRACTION = 0.70
_MPS_WARN_WORKING_SET_FRACTION = 0.50
_MPS_SKIP_MEMORY_CHECK_ENV = "QUANTEM_WIDGET_MPS_SKIP_MEMORY_CHECK"


def _file_cache_key(path: str) -> tuple[str, int, int]:
    path = os.path.abspath(os.fspath(path))
    st = os.stat(path)
    return path, int(st.st_mtime_ns), int(st.st_size)


def _read_pixel_mask(master_path: str) -> np.ndarray | None:
    """Read the Arina dead-pixel mask without importing the full HDF5 loader."""
    key = "entry/instrument/detector/detectorSpecific/pixel_mask"
    try:
        with h5py.File(master_path, "r") as f:
            if key not in f:
                return None
            return f[key][:]
    except (OSError, KeyError):
        return None


def clear_mps_cache() -> None:
    """Drop reusable MPS decoder buffers while keeping cheap chunk-layout plans."""
    global _cached_dec, _cached_dec_key
    _cached_dec = None
    _cached_dec_key = None


def _format_gib(nbytes: int | float | None) -> str:
    if nbytes is None:
        return "unknown"
    return f"{float(nbytes) / (1 << 30):.1f} GiB"


def _mps_recommended_working_set_bytes() -> int | None:
    fn = getattr(_device, "recommendedMaxWorkingSetSize", None)
    if not callable(fn):
        return None
    try:
        value = int(fn())
    except Exception:
        return None
    return value if value > 0 else None


def _mps_max_buffer_bytes() -> int | None:
    fn = getattr(_device, "maxBufferLength", None)
    if not callable(fn):
        return None
    try:
        value = int(fn())
    except Exception:
        return None
    return value if value > 0 else None


def _mps_output_bytes(plan: MPSMasterPlan, det_bin: int) -> int:
    det_bin = int(det_bin)
    if det_bin < 1:
        raise ValueError("det_bin must be >= 1.")
    det_row, det_col = (int(x) for x in plan.detector_shape)
    if det_row % det_bin or det_col % det_bin:
        raise ValueError(
            f"Detector dims {(det_row, det_col)} are not divisible by det_bin={det_bin}."
        )
    return (
        int(plan.total_frames)
        * (det_row // det_bin)
        * (det_col // det_bin)
        * int(plan.dtype.itemsize)
    )


def _mps_recommended_det_bin(plan: MPSMasterPlan, limit_bytes: int) -> tuple[int, int] | None:
    for factor in (2, 4, 8, 16):
        det_row, det_col = (int(x) for x in plan.detector_shape)
        if det_row % factor or det_col % factor:
            continue
        output_bytes = _mps_output_bytes(plan, factor)
        if output_bytes <= int(limit_bytes):
            return factor, output_bytes
    return None


def _mps_skip_memory_check_requested(skip_memory_check: bool | None) -> bool:
    if skip_memory_check is not None:
        return bool(skip_memory_check)
    value = os.environ.get(_MPS_SKIP_MEMORY_CHECK_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _check_mps_memory_guard(
    plan: MPSMasterPlan,
    *,
    det_bin: int,
    skip_memory_check: bool | None = None,
) -> None:
    """Fail early before an MPS load can pressure unified memory.

    The guard is intentionally metadata-only: it runs before Metal output
    buffers are allocated. It never silently changes ``det_bin`` because binning
    changes the data; instead it recommends the smallest safer bin factor.
    """
    det_bin = int(det_bin)
    recommended = _mps_recommended_working_set_bytes()
    if recommended is None:
        return
    output_bytes = _mps_output_bytes(plan, det_bin)
    safe_limit = int(recommended * _MPS_SAFE_WORKING_SET_FRACTION)
    warn_limit = int(recommended * _MPS_WARN_WORKING_SET_FRACTION)
    max_buffer = _mps_max_buffer_bytes()
    hard_limit = safe_limit
    if det_bin > 1 and max_buffer is not None:
        # The binned MPS path uses one output Metal buffer, so maxBufferLength is
        # a hard per-allocation cap in addition to total unified-memory pressure.
        hard_limit = min(hard_limit, int(max_buffer * 0.95))

    if output_bytes <= warn_limit:
        return

    recommendation = _mps_recommended_det_bin(plan, hard_limit)
    if recommendation is None:
        rec_text = "Use a larger det_bin or load a smaller scan region."
    else:
        rec_bin, rec_bytes = recommendation
        rec_text = (
            f"Use det_bin={rec_bin} "
            f"(estimated output {_format_gib(rec_bytes)}) for browsing."
        )
    message = (
        "MPS load memory check: "
        f"det_bin={det_bin} would materialize {_format_gib(output_bytes)} "
        f"for {os.path.basename(plan.master_path)}. This Mac reports a Metal "
        f"recommended working set of {_format_gib(recommended)}; quantem.widget "
        f"uses a conservative {_MPS_SAFE_WORKING_SET_FRACTION:.0%} limit "
        f"({_format_gib(safe_limit)}) to avoid freezing the laptop. {rec_text} "
        "MPS is still selected automatically; only the large allocation is blocked."
    )
    if output_bytes <= hard_limit:
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        return

    if _mps_skip_memory_check_requested(skip_memory_check):
        warnings.warn(
            message
            + f" Proceeding because skip_mps_memory_check=True or {_MPS_SKIP_MEMORY_CHECK_ENV}=1.",
            RuntimeWarning,
            stacklevel=3,
        )
        return
    raise MemoryError(
        message
        + " To bypass this memory check and force the no-bin/large MPS load, pass "
        "skip_mps_memory_check=True or set QUANTEM_WIDGET_MPS_SKIP_MEMORY_CHECK=1."
    )


def _get_cached_decompressor(frame_bytes: int, max_frames: int) -> "MPSDecompressor":
    """Reuse one decompressor so repeated notebook loads do not leak buffers."""
    global _cached_dec, _cached_dec_key
    key = (int(frame_bytes), int(max_frames))
    if _cached_dec is None or _cached_dec_key != key:
        _cached_dec = MPSDecompressor(frame_bytes=frame_bytes, max_frames=max_frames)
        _cached_dec_key = key
    return _cached_dec


def _get_chunk_read_plan(filepath: str) -> _ChunkReadPlan:
    """Return cached HDF5 chunk byte layout for one Arina data file."""
    key = _file_cache_key(filepath)
    cached = _chunk_read_plan_cache.get(key)
    if cached is not None:
        return cached

    with h5py.File(filepath, "r") as f:
        ds = f["entry/data/data"]
        n_frames = int(ds.shape[0])
        frame_shape = tuple(int(x) for x in ds.shape[1:])
        dtype = np.dtype(ds.dtype)
        file_offsets = np.empty(n_frames, dtype=np.int64)
        sizes = np.empty(n_frames, dtype=np.int64)

        def _collect(info):
            frame_idx = int(info.chunk_offset[0])
            file_offsets[frame_idx] = int(info.byte_offset)
            sizes[frame_idx] = int(info.size)

        ds.id.chunk_iter(_collect)

    out_offsets = np.empty(n_frames, dtype=np.int64)
    out_offsets[0] = 0
    np.cumsum(sizes[:-1], out=out_offsets[1:])
    total = int(out_offsets[-1] + sizes[-1])
    run_break = np.flatnonzero(
        file_offsets[1:] != file_offsets[:-1] + sizes[:-1]
    ) + 1
    run_start = np.concatenate(([0], run_break)).astype(np.int64, copy=False)
    run_end = np.concatenate((run_break, [n_frames])).astype(np.int64, copy=False)
    plan = _ChunkReadPlan(
        n_frames=n_frames,
        frame_shape=frame_shape,
        dtype=dtype,
        file_offsets=file_offsets,
        sizes=sizes,
        out_offsets=out_offsets,
        run_start=run_start,
        run_end=run_end,
        total_bytes=total,
    )
    _chunk_read_plan_cache[key] = plan
    return plan


def load_master(
    filepath: str,
    *,
    det_bin: int = 1,
    pixel_mask: "np.ndarray | None" = None,
    verbose: bool = True,
) -> np.ndarray:
    """Decompress an arina master to numpy ``(n_frames, det_row, det_col)`` on
    the Apple GPU. Same signature + contract as the cpu backend's load_master:
    native uint16 dtype, dead-pixel mask applied BEFORE binning, integer-sum
    detector binning (host-side) when det_bin > 1.

    No-bin uses the widget full-stack decompressor (needs the stack in RAM).
    det_bin > 1 uses a fused GPU LZ4+bitshuffle+integer-sum-bin (mask-aware,
    uint16, double-buffered) that keeps only the binned result in memory and
    matches the cuda integer-sum bin bit-for-bit.
    """
    det_shape, dtype, _ntrigger, chunk_files, chunk_n_frames = _parse_master(filepath)
    det_row, det_col = det_shape

    if det_bin <= 1:
        # No-bin: the verbatim widget full-stack decompressor. Native uint16,
        # bit-identical to cuda. Needs the whole stack in RAM (e.g. 19.3 GB),
        # so on a memory-constrained Mac this is for small stacks; use det_bin
        # > 1 for big ones (the streaming path below).
        frame_bytes = int(np.prod(det_shape) * np.dtype(dtype).itemsize)
        dec = _get_decompressor(frame_bytes, max_frames=max(chunk_n_frames))
        out = dec.load_master(filepath)
        if pixel_mask is not None:
            bad = np.asarray(pixel_mask) != 0
            if bad.shape == out.shape[1:]:
                out[:, bad] = 0  # zero dead pixels (raw frames, matches cuda)
        return out

    # det_bin > 1: fused GPU LZ4+bitshuffle+integer-sum-bin, dead-pixel masked,
    # native uint16, double-buffered. Keeps only the 4.8 GB binned result in
    # memory (fits a 24 GB Mac) and runs at the GPU decompress floor. Bit-
    # identical to a cuda integer-sum bin (mask applied pre-bin in the kernel).
    frame_bytes = int(np.prod(det_shape) * np.dtype(dtype).itemsize)
    dec = _get_decompressor(frame_bytes, max_frames=max(chunk_n_frames))
    return dec.load_binned_masked(filepath, det_bin, mask=pixel_mask, verbose=verbose)

# ---------------------------------------------------------------------------
# Metal Shading Language kernels
# ---------------------------------------------------------------------------
import pathlib as _pathlib
_METAL_SOURCE = (_pathlib.Path(__file__).parent / 'metal' / 'bslz4.msl').read_text()

# ---------------------------------------------------------------------------
# Compile Metal kernels at import time
# ---------------------------------------------------------------------------
# LZ4 occupancy knob: compressed 8 KB bitshuffle blocks packed per threadgroup
# (more SIMD groups → better latency hiding on the M5). 8 is the most stable
# no-bin setting; larger groups can be tested with QT_MPS_LZ4_Y. The
# threadgroup input cache is sized to exactly _LZ4_Y via compile-time token
# substitution — an oversized fixed buffer hurts occupancy and erases the win.
_LZ4_Y = int(os.environ.get("QT_MPS_LZ4_Y", "8"))
_device = Metal.MTLCreateSystemDefaultDevice()
_options = Metal.MTLCompileOptions.alloc().init()
_library, _compile_error = _device.newLibraryWithSource_options_error_(
    _METAL_SOURCE.replace("LZ4_BLOCKS_PER_TG", str(_LZ4_Y)), _options, None
)
if _compile_error:
    raise RuntimeError(f"Metal shader compile error: {_compile_error}")
_h5lz4dc_fn = _library.newFunctionWithName_("h5lz4dc_batched")
_shuf32_fn = _library.newFunctionWithName_("shuf_8192_32_batched")
_shuf16_fn = _library.newFunctionWithName_("shuf_8192_16_batched")
_bin_u16_fn = _library.newFunctionWithName_("bin_sum_u16")
_bin_u32_fn = _library.newFunctionWithName_("bin_sum_u32")
_bin_tiled_u16_fn = _library.newFunctionWithName_("bin_sum_tiled_u16")
_zero_bad_u16_fn = _library.newFunctionWithName_("zero_bad_pixels_u16")
_row_prefix_masked_u16_fn = _library.newFunctionWithName_("row_prefix_masked_u16")
_row_prefix_u16_fn = _library.newFunctionWithName_("row_prefix_u16")
_h5lz4dc_pipeline, _ = _device.newComputePipelineStateWithFunction_error_(_h5lz4dc_fn, None)
_shuf32_pipeline, _ = _device.newComputePipelineStateWithFunction_error_(_shuf32_fn, None)
_shuf16_pipeline, _ = _device.newComputePipelineStateWithFunction_error_(_shuf16_fn, None)
_bin_u16_pipeline, _ = _device.newComputePipelineStateWithFunction_error_(_bin_u16_fn, None)
_bin_u32_pipeline, _ = _device.newComputePipelineStateWithFunction_error_(_bin_u32_fn, None)
_bin_tiled_u16_pipeline, _ = _device.newComputePipelineStateWithFunction_error_(_bin_tiled_u16_fn, None)
_zero_bad_u16_pipeline, _ = _device.newComputePipelineStateWithFunction_error_(_zero_bad_u16_fn, None)
_row_prefix_masked_u16_pipeline, _ = _device.newComputePipelineStateWithFunction_error_(
    _row_prefix_masked_u16_fn, None
)
_row_prefix_u16_pipeline, _ = _device.newComputePipelineStateWithFunction_error_(
    _row_prefix_u16_fn, None
)
_queue = _device.newCommandQueue()


# ---------------------------------------------------------------------------
# Header parser (numba, runs on CPU in parallel)
# ---------------------------------------------------------------------------
@njit(parallel=True)
def _bin_mean(src, dst, brow, bcol):
    """Bin a 3D array (n, rows, cols) by averaging brow×bcol blocks."""
    nf, dr, dc = dst.shape
    scale = np.float32(1.0 / (brow * bcol))
    for i in prange(nf):
        for r in range(dr):
            rb = r * brow
            for c in range(dc):
                cb = c * bcol
                s = np.float32(0.0)
                for br in range(brow):
                    for bc in range(bcol):
                        s += np.float32(src[i, rb + br, cb + bc])
                dst[i, r, c] = s * scale


@njit(cache=True, parallel=True)
def _parse_headers(
    buffer, chunk_sizes, chunk_offsets,
    block_starts_out, block_counts_out,
    n_frames, n_blocks_per_frame,
):
    """Parse bitshuffle+LZ4 chunk headers in parallel."""
    for i in prange(n_frames):
        offset = chunk_offsets[i]
        chunk = buffer[offset : offset + chunk_sizes[i]]
        uncomp_size = (
            int(chunk[0]) << 56 | int(chunk[1]) << 48
            | int(chunk[2]) << 40 | int(chunk[3]) << 32
            | int(chunk[4]) << 24 | int(chunk[5]) << 16
            | int(chunk[6]) << 8  | int(chunk[7])
        )
        block_size = (
            int(chunk[8]) << 24 | int(chunk[9]) << 16
            | int(chunk[10]) << 8 | int(chunk[11])
        )
        n_blocks = (uncomp_size + block_size - 1) // block_size
        block_counts_out[i] = n_blocks
        pos = 12
        base_idx = i * n_blocks_per_frame
        for b in range(n_blocks):
            block_starts_out[base_idx + b] = pos
            comp_size = (
                int(chunk[pos]) << 24 | int(chunk[pos + 1]) << 16
                | int(chunk[pos + 2]) << 8 | int(chunk[pos + 3])
            )
            pos += 4 + comp_size


def _metal_buffer_alloc(nbytes):
    """Allocate an MTLBuffer of given size (shared memory)."""
    buf = _device.newBufferWithLength_options_(
        nbytes, Metal.MTLResourceStorageModeShared
    )
    if buf is None:
        gb = nbytes / 1e9
        raise MemoryError(
            f"Metal buffer allocation failed ({gb:.1f} GB). "
            f"Try a larger det_bin to reduce output size."
        )
    return buf


def _numpy_view(mtl_buf, dtype, count):
    """Get a writable numpy view of a Metal buffer (zero-copy, unified memory)."""
    mv = mtl_buf.contents().as_buffer(mtl_buf.length())
    return np.frombuffer(mv, dtype=dtype, count=count)


class _MtlArray(np.ndarray):
    """ndarray that keeps its backing Metal buffer alive via ``_mtl``.

    The binned load returns a zero-copy view into a Metal unified-memory
    buffer (no host memcpy). This subclass holds a reference to that buffer so
    it is not freed while the array is in use; the buffer is allocated fresh
    per load so the view can never be aliased by a later decompress.
    """
    _mtl = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class MPSDecompressor:
    """MPS-accelerated decompressor for bitshuffle+LZ4 HDF5 datasets.

    Uses Metal compute shaders on Apple Silicon. All buffers are pre-allocated
    in unified memory and reused across calls — no per-call allocation or
    CPU-GPU transfers.

    Parameters
    ----------
    max_compressed_bytes : int, optional
        Maximum total compressed data per load call, by default 150 MB.
    max_frames : int, optional
        Maximum number of frames per load call, by default 11000.
    frame_bytes : int, optional
        Decompressed bytes per frame, by default 192*192*2 (uint16).
    n_blocks_per_frame : int, optional
        LZ4 blocks per frame, by default 9 for 192x192 uint16.
    """

    def __init__(
        self,
        max_compressed_bytes: int = 150 * 1024 * 1024,
        max_frames: int = 11_000,
        frame_bytes: int = 192 * 192 * 2,
        n_blocks_per_frame: int = 9,
        gpu_batch: int | None = None,
    ):
        self.max_frames = max_frames
        self.frame_bytes = frame_bytes
        self.n_blocks_per_frame = n_blocks_per_frame
        # gpu_batch controls _lz4/_shuf sizing (smaller = less GPU memory)
        self.gpu_batch = gpu_batch or max_frames
        # Pre-allocate Metal buffers with numpy views (unified memory)
        self._comp_mtl = _metal_buffer_alloc(max_compressed_bytes)
        self._comp_np = _numpy_view(self._comp_mtl, np.uint8, max_compressed_bytes)
        gpu_output = self.gpu_batch * frame_bytes
        self._lz4_mtl = _metal_buffer_alloc(gpu_output)
        # Second LZ4 scratch for the D=2 chunked pipeline (two command buffers
        # in flight need separate scratch). Allocated lazily on first chunked use.
        self._lz4_mtl_b = None
        self._lz4_mtl_extra: list = []
        self._shuf_mtl = _metal_buffer_alloc(gpu_output)
        self._result_np = _numpy_view(self._shuf_mtl, np.uint8, gpu_output)
        # Pre-allocate metadata Metal buffers with numpy views
        self._co_mtl = _metal_buffer_alloc(max_frames * 4)
        self._co_np = _numpy_view(self._co_mtl, np.uint32, max_frames)
        max_blocks = max_frames * n_blocks_per_frame
        self._bs_mtl = _metal_buffer_alloc(max_blocks * 4)
        self._bs_np = _numpy_view(self._bs_mtl, np.uint32, max_blocks)
        self._bc_mtl = _metal_buffer_alloc(max_frames * 4)
        self._bc_np = _numpy_view(self._bc_mtl, np.uint32, max_frames)
        self._bo_mtl = _metal_buffer_alloc((max_frames + 1) * 4)
        self._bo_np = _numpy_view(self._bo_mtl, np.uint32, max_frames + 1)
        # CPU-side arrays for chunk reading
        self._chunk_sizes = np.zeros(max_frames, dtype=np.uint32)
        # Buffer B (for double-buffering in load_master)
        self._comp_mtl_b = _metal_buffer_alloc(max_compressed_bytes)
        self._comp_np_b = _numpy_view(self._comp_mtl_b, np.uint8, max_compressed_bytes)
        self._co_mtl_b = _metal_buffer_alloc(max_frames * 4)
        self._co_np_b = _numpy_view(self._co_mtl_b, np.uint32, max_frames)
        self._bs_mtl_b = _metal_buffer_alloc(max_blocks * 4)
        self._bs_np_b = _numpy_view(self._bs_mtl_b, np.uint32, max_blocks)
        self._bc_mtl_b = _metal_buffer_alloc(max_frames * 4)
        self._bc_np_b = _numpy_view(self._bc_mtl_b, np.uint32, max_frames)
        self._bo_mtl_b = _metal_buffer_alloc((max_frames + 1) * 4)
        self._bo_np_b = _numpy_view(self._bo_mtl_b, np.uint32, max_frames + 1)
        self._chunk_sizes_b = np.zeros(max_frames, dtype=np.uint32)
        # Optional third compressed-input slot for chunked no-bin loads. This is
        # much smaller than a third LZ4 scratch buffer, but lets the CPU read and
        # parse chunk N+2 before GPU scratch slot N is free.
        self._comp_mtl_c = None
        self._comp_np_c = None
        self._co_mtl_c = None
        self._co_np_c = None
        self._bs_mtl_c = None
        self._bs_np_c = None
        self._bc_mtl_c = None
        self._bc_np_c = None
        self._bo_mtl_c = None
        self._bo_np_c = None
        self._chunk_sizes_c = None
        # Large output buffer for load_master() — allocated on first use
        self._out_mtl = None
        self._out_np = None
        self._out_nbytes = 0
        # Reusable per-chunk output buffer pool for load_master_chunked()
        self._chunk_out_pool: list = []
        self._chunk_fast_pool: list = []
        self._bad_idx_mtl = None
        self._bad_idx_np = None
        self._bad_idx_capacity = 0
        self._bad_idx_count = 0
        self._prefix_overflow_mtl = _metal_buffer_alloc(4)
        self._prefix_overflow_np = _numpy_view(
            self._prefix_overflow_mtl, np.uint32, 1
        )

    def _read_ahead_buffer(self):
        """Return a third compressed-input metadata slot for load pipelining."""
        if self._comp_mtl_c is None:
            max_compressed_bytes = int(self._comp_mtl.length())
            max_blocks = int(self._bs_np.shape[0])
            self._comp_mtl_c = _metal_buffer_alloc(max_compressed_bytes)
            self._comp_np_c = _numpy_view(
                self._comp_mtl_c, np.uint8, max_compressed_bytes
            )
            self._co_mtl_c = _metal_buffer_alloc(self.max_frames * 4)
            self._co_np_c = _numpy_view(self._co_mtl_c, np.uint32, self.max_frames)
            self._bs_mtl_c = _metal_buffer_alloc(max_blocks * 4)
            self._bs_np_c = _numpy_view(self._bs_mtl_c, np.uint32, max_blocks)
            self._bc_mtl_c = _metal_buffer_alloc(self.max_frames * 4)
            self._bc_np_c = _numpy_view(self._bc_mtl_c, np.uint32, self.max_frames)
            self._bo_mtl_c = _metal_buffer_alloc((self.max_frames + 1) * 4)
            self._bo_np_c = _numpy_view(
                self._bo_mtl_c, np.uint32, self.max_frames + 1
            )
            self._chunk_sizes_c = np.zeros(self.max_frames, dtype=np.uint32)
        return (
            self._comp_np_c, self._co_np_c, self._bs_np_c, self._bc_np_c,
            self._bo_np_c, self._chunk_sizes_c,
            self._comp_mtl_c, self._co_mtl_c, self._bs_mtl_c, self._bc_mtl_c,
            self._bo_mtl_c,
        )

    def _ensure_output_buffer(self, nbytes):
        """Allocate (or reuse) a large Metal output buffer for all frames."""
        if nbytes <= self._out_nbytes:
            return
        self._out_mtl = _metal_buffer_alloc(nbytes)
        self._out_np = _numpy_view(self._out_mtl, np.uint8, nbytes)
        self._out_nbytes = nbytes

    def _read_chunk(self, filepath, comp_np, co_np, chunk_sizes):
        """Read raw compressed HDF5 chunks into pre-allocated buffers.

        Each Arina frame is one HDF5 chunk, so the naive per-frame
        ``read_direct_chunk`` loop costs 10000 calls/file (262144 total). The
        data chunks in an Arina file are nearly contiguous, with small HDF5
        metadata gaps every few dozen frames. Reading the whole byte span once
        cuts the a 512² file from 176 syscalls/data-file to 1 while still
        avoiding a Python copy: chunk offsets point into the span, and the Metal
        LZ4 kernel ignores the tiny gaps.
        """
        plan = _get_chunk_read_plan(filepath)
        n_frames = plan.n_frames
        chunk_sizes[:n_frames] = plan.sizes
        first = int(plan.file_offsets.min())
        last = int((plan.file_offsets + plan.sizes).max())
        span_bytes = last - first
        fd = os.open(filepath, os.O_RDONLY)
        try:
            if (
                os.environ.get("QT_MPS_SPAN_READ", "1") != "0"
                and span_bytes <= int(comp_np.size)
                and span_bytes <= int(plan.total_bytes * 1.25) + 8 * 1024 * 1024
            ):
                co_np[:n_frames] = plan.file_offsets - first
                dest = memoryview(comp_np[:span_bytes])
                os.preadv(fd, [dest], first)
            else:
                co_np[:n_frames] = plan.out_offsets
                dest = memoryview(comp_np[:plan.total_bytes])
                for start, end in zip(plan.run_start.tolist(), plan.run_end.tolist()):
                    run_bytes = int(
                        plan.out_offsets[end - 1]
                        + plan.sizes[end - 1]
                        - plan.out_offsets[start]
                    )
                    dest_pos = int(plan.out_offsets[start])
                    os.preadv(
                        fd,
                        [dest[dest_pos : dest_pos + run_bytes]],
                        int(plan.file_offsets[start]),
                    )
        finally:
            os.close(fd)
        return n_frames, plan.frame_shape, plan.dtype

    def _submit_gpu(self, n_frames, frame_bytes, elem_size, out_byte_offset,
                    comp_mtl, co_mtl, bs_mtl, bc_mtl, bo_mtl, max_blocks,
                    out_mtl=None, lz4_mtl=None, zero_bad: bool = False,
                    row_prefix: bool = False,
                    det_shape: tuple[int, int] | None = None,
                    fast_out_mtl=None,
                    fast_out_byte_offset: int = 0,
                    fast_det_bin: int | None = None):
        """Submit LZ4 + bitshuffle GPU work, return uncommitted command buffer.

        out_mtl: destination buffer for the bitshuffle output. Defaults to the
        single big self._out_mtl (load_master). Pass a per-chunk buffer (with
        out_byte_offset=0) for the chunked no-bin path that dodges the 14.3 GB
        maxBufferLength cap.
        lz4_mtl: the LZ4 intermediate scratch. Defaults to self._lz4_mtl. Pass a
        distinct buffer per in-flight chunk so a 2-deep pipeline (D=2) can run
        two command buffers concurrently without the second clobbering the
        first's scratch — recovers the ~0.6s per-chunk CPU<->GPU drain gap.
        """
        if out_mtl is None:
            out_mtl = self._out_mtl
        if lz4_mtl is None:
            lz4_mtl = self._lz4_mtl
        cmd = _queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        # LZ4 — n_frames in X (unlimited), per-frame blocks in Z (small)
        enc.setComputePipelineState_(_h5lz4dc_pipeline)
        enc.setBuffer_offset_atIndex_(comp_mtl, 0, 0)
        enc.setBuffer_offset_atIndex_(co_mtl, 0, 1)
        enc.setBuffer_offset_atIndex_(bs_mtl, 0, 2)
        enc.setBuffer_offset_atIndex_(bc_mtl, 0, 3)
        enc.setBuffer_offset_atIndex_(bo_mtl, 0, 4)
        enc.setBytes_length_atIndex_(
            np.array([8192], dtype=np.uint32).tobytes(), 4, 5
        )
        enc.setBytes_length_atIndex_(
            np.array([frame_bytes], dtype=np.uint32).tobytes(), 4, 6
        )
        enc.setBuffer_offset_atIndex_(lz4_mtl, 0, 7)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(n_frames, 1, (max_blocks + _LZ4_Y - 1) // _LZ4_Y),
            Metal.MTLSizeMake(32, _LZ4_Y, 1),
        )
        enc.memoryBarrierWithScope_(Metal.MTLBarrierScopeBuffers)
        # Bitshuffle — n_frames in X, tg_count in Z
        n_8kb = frame_bytes // 8192
        if elem_size == 2:
            groups_per_block = 8192 // (elem_size * 32)
            groups_per_frame = n_8kb * groups_per_block
            frame_elems = frame_bytes // 2
            enc.setComputePipelineState_(_shuf16_pipeline)
            enc.setBuffer_offset_atIndex_(lz4_mtl, 0, 0)
            enc.setBuffer_offset_atIndex_(out_mtl, out_byte_offset, 1)
            enc.setBytes_length_atIndex_(
                np.array([frame_elems], dtype=np.uint32).tobytes(), 4, 2
            )
            enc.setBytes_length_atIndex_(
                np.array([groups_per_block], dtype=np.uint32).tobytes(), 4, 3
            )
            enc.setBytes_length_atIndex_(
                np.array([groups_per_frame], dtype=np.uint32).tobytes(), 4, 4
            )
            tg_count = (groups_per_frame + 31) // 32
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSizeMake(n_frames, 1, tg_count),
                Metal.MTLSizeMake(32, 32, 1),
            )
        else:
            groups_per_block = 2048 // 32
            groups_per_frame = n_8kb * groups_per_block
            frame_elems = frame_bytes // 4
            enc.setComputePipelineState_(_shuf32_pipeline)
            enc.setBuffer_offset_atIndex_(lz4_mtl, 0, 0)
            enc.setBuffer_offset_atIndex_(out_mtl, out_byte_offset, 1)
            enc.setBytes_length_atIndex_(
                np.array([frame_elems], dtype=np.uint32).tobytes(), 4, 2
            )
            enc.setBytes_length_atIndex_(
                np.array([groups_per_block], dtype=np.uint32).tobytes(), 4, 3
            )
            enc.setBytes_length_atIndex_(
                np.array([groups_per_frame], dtype=np.uint32).tobytes(), 4, 4
            )
            tg_count = (groups_per_frame + 31) // 32
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSizeMake(n_frames, 1, tg_count),
                Metal.MTLSizeMake(32, 32, 1),
            )
        if row_prefix:
            if elem_size != 2:
                raise ValueError("row_prefix=True requires uint16 data.")
            if det_shape is None:
                raise ValueError("row_prefix=True requires det_shape.")
            detrows, detcols = (int(det_shape[0]), int(det_shape[1]))
            enc.memoryBarrierWithScope_(Metal.MTLBarrierScopeBuffers)
            if zero_bad and self._bad_idx_count:
                nbad = int(self._bad_idx_count)
                ndet = int(frame_bytes // elem_size)
                enc.setComputePipelineState_(_zero_bad_u16_pipeline)
                enc.setBuffer_offset_atIndex_(out_mtl, out_byte_offset, 0)
                enc.setBuffer_offset_atIndex_(self._bad_idx_mtl, 0, 1)
                enc.setBytes_length_atIndex_(
                    np.array([ndet], dtype=np.uint32).tobytes(), 4, 2
                )
                enc.setBytes_length_atIndex_(
                    np.array([nbad], dtype=np.uint32).tobytes(), 4, 3
                )
                enc.setBytes_length_atIndex_(
                    np.array([n_frames], dtype=np.uint32).tobytes(), 4, 4
                )
                total = n_frames * nbad
                enc.dispatchThreadgroups_threadsPerThreadgroup_(
                    Metal.MTLSizeMake((total + 255) // 256, 1, 1),
                    Metal.MTLSizeMake(256, 1, 1),
                )
                enc.memoryBarrierWithScope_(Metal.MTLBarrierScopeBuffers)
            enc.setComputePipelineState_(_row_prefix_u16_pipeline)
            enc.setBuffer_offset_atIndex_(out_mtl, out_byte_offset, 0)
            enc.setBuffer_offset_atIndex_(self._prefix_overflow_mtl, 0, 1)
            enc.setBytes_length_atIndex_(
                np.array([detcols], dtype=np.uint32).tobytes(), 4, 2
            )
            enc.setBytes_length_atIndex_(
                np.array([detrows], dtype=np.uint32).tobytes(), 4, 3
            )
            enc.setBytes_length_atIndex_(
                np.array([n_frames], dtype=np.uint32).tobytes(), 4, 4
            )
            total_rows = n_frames * detrows
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSizeMake((total_rows + 255) // 256, 1, 1),
                Metal.MTLSizeMake(256, 1, 1),
            )
        elif zero_bad and self._bad_idx_count:
            enc.memoryBarrierWithScope_(Metal.MTLBarrierScopeBuffers)
            nbad = int(self._bad_idx_count)
            ndet = int(frame_bytes // elem_size)
            enc.setComputePipelineState_(_zero_bad_u16_pipeline)
            enc.setBuffer_offset_atIndex_(out_mtl, out_byte_offset, 0)
            enc.setBuffer_offset_atIndex_(self._bad_idx_mtl, 0, 1)
            enc.setBytes_length_atIndex_(
                np.array([ndet], dtype=np.uint32).tobytes(), 4, 2
            )
            enc.setBytes_length_atIndex_(
                np.array([nbad], dtype=np.uint32).tobytes(), 4, 3
            )
            enc.setBytes_length_atIndex_(
                np.array([n_frames], dtype=np.uint32).tobytes(), 4, 4
            )
            total = n_frames * nbad
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSizeMake((total + 255) // 256, 1, 1),
                Metal.MTLSizeMake(256, 1, 1),
            )
        if fast_out_mtl is not None and fast_det_bin is not None:
            if row_prefix:
                raise ValueError("fast_det_bin cannot be fused with row_prefix=True.")
            if det_shape is None:
                raise ValueError("fast_det_bin requires det_shape.")
            det_row, det_col = (int(det_shape[0]), int(det_shape[1]))
            bin_factor = int(fast_det_bin)
            if det_row % bin_factor or det_col % bin_factor:
                raise ValueError(
                    f"Detector shape {(det_row, det_col)} is not divisible by "
                    f"fast_det_bin={bin_factor}."
                )
            enc.memoryBarrierWithScope_(Metal.MTLBarrierScopeBuffers)
            out_det_row = det_row // bin_factor
            out_det_col = det_col // bin_factor
            out_frame_elems = out_det_row * out_det_col
            in_frame_elems = det_row * det_col
            bin_pipeline = _bin_u16_pipeline if elem_size == 2 else _bin_u32_pipeline
            enc.setComputePipelineState_(bin_pipeline)
            enc.setBuffer_offset_atIndex_(out_mtl, out_byte_offset, 0)
            enc.setBuffer_offset_atIndex_(fast_out_mtl, fast_out_byte_offset, 1)
            enc.setBytes_length_atIndex_(
                np.array([det_col], dtype=np.uint32).tobytes(), 4, 2
            )
            enc.setBytes_length_atIndex_(
                np.array([in_frame_elems], dtype=np.uint32).tobytes(), 4, 3
            )
            enc.setBytes_length_atIndex_(
                np.array([out_det_col], dtype=np.uint32).tobytes(), 4, 4
            )
            enc.setBytes_length_atIndex_(
                np.array([out_frame_elems], dtype=np.uint32).tobytes(), 4, 5
            )
            enc.setBytes_length_atIndex_(
                np.array([bin_factor], dtype=np.uint32).tobytes(), 4, 6
            )
            enc.setBuffer_offset_atIndex_(self._mask_mtl, 0, 7)
            grid_x = (out_det_col + 15) // 16
            grid_y = (out_det_row + 15) // 16
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSizeMake(n_frames, grid_y, grid_x),
                Metal.MTLSizeMake(1, 16, 16),
            )
        enc.endEncoding()
        cmd.commit()
        return cmd

    def _submit_gpu_binned(self, n_frames, frame_bytes, elem_size,
                           out_byte_offset, det_row, det_col, bin_factor,
                           comp_mtl, co_mtl, bs_mtl, bc_mtl, bo_mtl,
                           max_blocks, meta_frame_offset=0):
        """Submit LZ4 + bitshuffle + bin GPU work. Returns command buffer.

        meta_frame_offset: offset into metadata buffers (co, bc, bo) for
        sub-batch processing. bs (block_starts) uses absolute indexing.
        """
        meta_off = meta_frame_offset * 4  # bytes (uint32 arrays)
        cmd = _queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        # LZ4 — n_frames in X
        enc.setComputePipelineState_(_h5lz4dc_pipeline)
        enc.setBuffer_offset_atIndex_(comp_mtl, 0, 0)
        enc.setBuffer_offset_atIndex_(co_mtl, meta_off, 1)
        enc.setBuffer_offset_atIndex_(bs_mtl, 0, 2)
        enc.setBuffer_offset_atIndex_(bc_mtl, meta_off, 3)
        enc.setBuffer_offset_atIndex_(bo_mtl, meta_off, 4)
        enc.setBytes_length_atIndex_(
            np.array([8192], dtype=np.uint32).tobytes(), 4, 5
        )
        enc.setBytes_length_atIndex_(
            np.array([frame_bytes], dtype=np.uint32).tobytes(), 4, 6
        )
        enc.setBuffer_offset_atIndex_(self._lz4_mtl, 0, 7)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(n_frames, 1, (max_blocks + _LZ4_Y - 1) // _LZ4_Y),
            Metal.MTLSizeMake(32, _LZ4_Y, 1),
        )
        enc.memoryBarrierWithScope_(Metal.MTLBarrierScopeBuffers)
        # Bitshuffle → _shuf_mtl (temporary) — n_frames in X
        n_8kb = frame_bytes // 8192
        if elem_size == 2:
            groups_per_block = 8192 // (elem_size * 32)
            groups_per_frame = n_8kb * groups_per_block
            frame_elems = frame_bytes // 2
            enc.setComputePipelineState_(_shuf16_pipeline)
            enc.setBuffer_offset_atIndex_(self._lz4_mtl, 0, 0)
            enc.setBuffer_offset_atIndex_(self._shuf_mtl, 0, 1)
            enc.setBytes_length_atIndex_(
                np.array([frame_elems], dtype=np.uint32).tobytes(), 4, 2
            )
            enc.setBytes_length_atIndex_(
                np.array([groups_per_block], dtype=np.uint32).tobytes(), 4, 3
            )
            enc.setBytes_length_atIndex_(
                np.array([groups_per_frame], dtype=np.uint32).tobytes(), 4, 4
            )
            tg_count = (groups_per_frame + 31) // 32
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSizeMake(n_frames, 1, tg_count),
                Metal.MTLSizeMake(32, 32, 1),
            )
        else:
            groups_per_block = 2048 // 32
            groups_per_frame = n_8kb * groups_per_block
            frame_elems = frame_bytes // 4
            enc.setComputePipelineState_(_shuf32_pipeline)
            enc.setBuffer_offset_atIndex_(self._lz4_mtl, 0, 0)
            enc.setBuffer_offset_atIndex_(self._shuf_mtl, 0, 1)
            enc.setBytes_length_atIndex_(
                np.array([frame_elems], dtype=np.uint32).tobytes(), 4, 2
            )
            enc.setBytes_length_atIndex_(
                np.array([groups_per_block], dtype=np.uint32).tobytes(), 4, 3
            )
            enc.setBytes_length_atIndex_(
                np.array([groups_per_frame], dtype=np.uint32).tobytes(), 4, 4
            )
            tg_count = (groups_per_frame + 31) // 32
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSizeMake(n_frames, 1, tg_count),
                Metal.MTLSizeMake(32, 32, 1),
            )
        enc.memoryBarrierWithScope_(Metal.MTLBarrierScopeBuffers)
        # Bin: _shuf_mtl → _out_mtl at offset — n_frames in X
        out_det_row = det_row // bin_factor
        out_det_col = det_col // bin_factor
        out_frame_elems = out_det_row * out_det_col
        in_frame_elems = det_row * det_col
        # uint16: scalar sum kernel. (A threadgroup-tiled bin was tried and
        # regressed on M5 — barrier + reduced occupancy beat the coalescing
        # win; Apple's cache absorbs the strided 2x2 gather fine.)
        tiled = False
        bin_pipeline = (_bin_tiled_u16_pipeline if tiled
                        else (_bin_u16_pipeline if elem_size == 2 else _bin_u32_pipeline))
        enc.setComputePipelineState_(bin_pipeline)
        enc.setBuffer_offset_atIndex_(self._shuf_mtl, 0, 0)
        enc.setBuffer_offset_atIndex_(self._out_mtl, out_byte_offset, 1)
        enc.setBytes_length_atIndex_(
            np.array([det_col], dtype=np.uint32).tobytes(), 4, 2
        )
        enc.setBytes_length_atIndex_(
            np.array([in_frame_elems], dtype=np.uint32).tobytes(), 4, 3
        )
        enc.setBytes_length_atIndex_(
            np.array([out_det_col], dtype=np.uint32).tobytes(), 4, 4
        )
        enc.setBytes_length_atIndex_(
            np.array([out_frame_elems], dtype=np.uint32).tobytes(), 4, 5
        )
        enc.setBytes_length_atIndex_(
            np.array([bin_factor], dtype=np.uint32).tobytes(), 4, 6
        )
        # Dead-pixel mask (buffer 7): one detector frame, uchar, nonzero = dead.
        enc.setBuffer_offset_atIndex_(self._mask_mtl, 0, 7)
        if tiled:
            enc.setBytes_length_atIndex_(
                np.array([out_det_row], dtype=np.uint32).tobytes(), 4, 8
            )
        grid_x = (out_det_col + 15) // 16
        grid_y = (out_det_row + 15) // 16
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(n_frames, grid_y, grid_x),
            Metal.MTLSizeMake(1, 16, 16),
        )
        enc.endEncoding()
        cmd.commit()
        return cmd

    def _set_mask(self, mask, det_row, det_col):
        """Upload the dead-pixel mask (nonzero = dead) to a Metal buffer for the
        bin kernel. A None mask becomes all-zero (nothing dead)."""
        n = det_row * det_col
        if getattr(self, "_mask_mtl", None) is None or self._mask_np.size != n:
            self._mask_mtl = _metal_buffer_alloc(n)
            self._mask_np = _numpy_view(self._mask_mtl, np.uint8, n)
        if mask is None:
            self._mask_np[:] = 0
        else:
            bad = (np.asarray(mask) != 0).astype(np.uint8).ravel()
            self._mask_np[:] = bad if bad.size == n else 0

    def _set_bad_pixels(self, pixel_mask, frame_shape) -> None:
        if pixel_mask is None:
            self._bad_idx_count = 0
            return
        mask = np.asarray(pixel_mask) != 0
        if mask.shape != tuple(frame_shape):
            self._bad_idx_count = 0
            return
        bad = np.flatnonzero(mask.reshape(-1)).astype(np.uint32, copy=False)
        nbad = int(bad.size)
        self._bad_idx_count = nbad
        if nbad == 0:
            return
        if nbad > self._bad_idx_capacity:
            self._bad_idx_capacity = max(16, nbad)
            self._bad_idx_mtl = _metal_buffer_alloc(self._bad_idx_capacity * 4)
            self._bad_idx_np = _numpy_view(
                self._bad_idx_mtl, np.uint32, self._bad_idx_capacity
            )
        self._bad_idx_np[:nbad] = bad

    def load_binned_masked(self, master_path, det_bin, mask=None, verbose=False):
        """Fast det_bin>1 path: GPU LZ4+bitshuffle+integer-sum-bin, dead-pixel
        masked, native uint16, double-buffered (read next chunk while the GPU
        bins the current one). Writes binned uint16 straight into one full
        output buffer at each frame offset — no per-batch host copy — so it
        runs at the GPU decompress floor instead of paying a float32 memcpy
        tax. Bit-identical to a cuda integer-sum bin.
        """
        det_shape, dtype, _nt, chunk_files, chunk_n_frames = _parse_master(master_path)
        det_row, det_col = det_shape
        frame_bytes = int(np.prod(det_shape) * np.dtype(dtype).itemsize)
        elem_size = np.dtype(dtype).itemsize
        n_blocks_per_frame = (frame_bytes + 8191) // 8192
        out_row, out_col = det_row // det_bin, det_col // det_bin
        out_frame_bytes = out_row * out_col * elem_size  # uint16 binned
        total_frames = sum(chunk_n_frames)
        self._set_mask(mask, det_row, det_col)
        # Fresh output buffer per call (NOT the reused _ensure_output_buffer
        # pool) so the zero-copy view we return below can't be aliased by a
        # later load. The returned array keeps this buffer alive via _mtl.
        out_total_bytes = total_frames * out_frame_bytes
        self._out_mtl = _metal_buffer_alloc(out_total_bytes)
        self._out_np = _numpy_view(self._out_mtl, np.uint8, out_total_bytes)
        self._out_nbytes = out_total_bytes
        bufs = [
            (self._comp_np, self._co_np, self._bs_np, self._bc_np, self._bo_np,
             self._chunk_sizes, self._comp_mtl, self._co_mtl, self._bs_mtl,
             self._bc_mtl, self._bo_mtl),
            (self._comp_np_b, self._co_np_b, self._bs_np_b, self._bc_np_b,
             self._bo_np_b, self._chunk_sizes_b, self._comp_mtl_b, self._co_mtl_b,
             self._bs_mtl_b, self._bc_mtl_b, self._bo_mtl_b),
        ]

        def _read_parse(buf_idx, ci):
            comp_np, co_np, bs_np, bc_np, bo_np, csizes, *_ = bufs[buf_idx]
            self._read_chunk(chunk_files[ci], comp_np, co_np, csizes)
            nf = chunk_n_frames[ci]
            _parse_headers(comp_np, csizes, co_np, bs_np, bc_np, nf, n_blocks_per_frame)
            bo_np[0] = 0
            bo_np[1 : nf + 1] = np.cumsum(bc_np[:nf])

        import time as _t
        _tr = _t.perf_counter()
        _read_parse(0, 0)
        t_read = _t.perf_counter() - _tr
        t_gpu = 0.0
        frame_offset = 0
        n_chunks = len(chunk_files)
        gpu_batch = self.gpu_batch
        iterable = range(n_chunks)
        if verbose:
            iterable = tqdm(iterable, desc="mps", leave=False)
        for ci in iterable:
            comp_np, co_np, bs_np, bc_np, bo_np, csizes, comp_mtl, co_mtl, \
                bs_mtl, bc_mtl, bo_mtl = bufs[ci % 2]
            nf = chunk_n_frames[ci]
            read_next_done = False
            for s in range(0, nf, gpu_batch):
                e = min(s + gpu_batch, nf)
                nb = e - s
                max_blk = int(bc_np[s:e].max())
                _tg = _t.perf_counter()
                cmd = self._submit_gpu_binned(
                    nb, frame_bytes, elem_size,
                    (frame_offset + s) * out_frame_bytes,
                    det_row, det_col, det_bin,
                    comp_mtl, co_mtl, bs_mtl, bc_mtl, bo_mtl,
                    max_blk, meta_frame_offset=s,
                )
                # Overlap: read+parse next chunk while the last sub-batch runs.
                if not read_next_done and e >= nf and ci + 1 < n_chunks:
                    _tr = _t.perf_counter()
                    _read_parse((ci + 1) % 2, ci + 1)
                    t_read += _t.perf_counter() - _tr
                cmd.waitUntilCompleted()
                t_gpu += _t.perf_counter() - _tg
                if not read_next_done and e >= nf:
                    read_next_done = True
            frame_offset += nf
        # Zero-copy: return a view straight into the Metal unified-memory output
        # buffer (no 4.8 GB host memcpy). _MtlArray holds a reference to the
        # Metal buffer so it stays alive as long as the array does; we allocated
        # a fresh buffer above, so nothing else can overwrite it.
        view = self._out_np[:out_total_bytes].view(dtype).reshape(
            total_frames, out_row, out_col
        )
        out = view.view(_MtlArray)
        out._mtl = self._out_mtl
        # Drop our refs so the buffer's lifetime is owned by the returned array.
        self._out_mtl = None
        self._out_np = None
        self._out_nbytes = 0
        if verbose:
            print(f"[mps] read(non-overlapped) {t_read:.2f}s  gpu+wait {t_gpu:.2f}s  "
                  f"(zero-copy out)")
        return out

    def load_master(self, master_path: str) -> np.ndarray:
        """Load all chunks from an arina master file via MPS.

        Uses double-buffering: reads chunk N+1 while GPU processes chunk N.
        Writes each chunk's result directly at the correct offset in a single
        output buffer — no intermediate copies.

        Parameters
        ----------
        master_path : str
            Path to the arina master HDF5 file.

        Returns
        -------
        np.ndarray
            Numpy array with shape (total_frames, det_rows, det_cols).
        """
        t0 = time.perf_counter()
        plan = plan_master(master_path)
        frame_shape = plan.detector_shape
        dtype = plan.dtype
        frame_bytes = plan.frame_bytes
        elem_size = plan.elem_size
        chunk_files = list(plan.chunk_files)
        chunk_n_frames = list(plan.chunk_n_frames)
        total_frames = sum(chunk_n_frames)
        total_bytes = total_frames * frame_bytes
        self._ensure_output_buffer(total_bytes)
        n_blocks_per_frame = plan.n_blocks_per_frame
        # Double-buffer sets: A (primary) and B
        bufs = [
            (self._comp_np, self._co_np, self._bs_np, self._bc_np,
             self._bo_np, self._chunk_sizes,
             self._comp_mtl, self._co_mtl, self._bs_mtl, self._bc_mtl,
             self._bo_mtl),
            (self._comp_np_b, self._co_np_b, self._bs_np_b, self._bc_np_b,
             self._bo_np_b, self._chunk_sizes_b,
             self._comp_mtl_b, self._co_mtl_b, self._bs_mtl_b, self._bc_mtl_b,
             self._bo_mtl_b),
        ]
        t_setup = time.perf_counter()
        # Read first chunk into buffer A
        comp_np, co_np, bs_np, bc_np, bo_np, csizes, \
            comp_mtl, co_mtl, bs_mtl, bc_mtl, bo_mtl = bufs[0]
        self._read_chunk(chunk_files[0], comp_np, co_np, csizes)
        _parse_headers(comp_np, csizes, co_np, bs_np, bc_np,
                       chunk_n_frames[0], n_blocks_per_frame)
        bo_np[0] = 0
        bo_np[1 : chunk_n_frames[0] + 1] = np.cumsum(bc_np[:chunk_n_frames[0]])
        max_blk = int(bc_np[:chunk_n_frames[0]].max())
        frame_offset = 0
        n_chunks = len(chunk_files)
        chunk_range = range(n_chunks)
        if n_chunks > 1:
            chunk_range = tqdm(chunk_range, desc="GPU chunks", leave=False)
        for ci in chunk_range:
            cur = bufs[ci % 2]
            comp_np, co_np, bs_np, bc_np, bo_np, csizes, \
                comp_mtl, co_mtl, bs_mtl, bc_mtl, bo_mtl = cur
            n_frames = chunk_n_frames[ci]
            out_byte_offset = frame_offset * frame_bytes
            # Submit GPU (async — returns immediately)
            cmd = self._submit_gpu(
                n_frames, frame_bytes, elem_size, out_byte_offset,
                comp_mtl, co_mtl, bs_mtl, bc_mtl, bo_mtl, max_blk,
            )
            # While GPU runs, read + parse next chunk into the other buffer
            if ci + 1 < n_chunks:
                nxt = bufs[(ci + 1) % 2]
                comp_np_n, co_np_n, bs_np_n, bc_np_n, bo_np_n, csizes_n, \
                    *_ = nxt
                self._read_chunk(chunk_files[ci + 1], comp_np_n, co_np_n,
                                 csizes_n)
                nf_next = chunk_n_frames[ci + 1]
                _parse_headers(comp_np_n, csizes_n, co_np_n, bs_np_n,
                               bc_np_n, nf_next, n_blocks_per_frame)
                bo_np_n[0] = 0
                bo_np_n[1 : nf_next + 1] = np.cumsum(bc_np_n[:nf_next])
                max_blk = int(bc_np_n[:nf_next].max())
            # Wait for current GPU to finish
            cmd.waitUntilCompleted()
            frame_offset += n_frames
        t_total = time.perf_counter()
        result = self._out_np[:total_bytes].view(dtype).reshape(
            (total_frames,) + frame_shape
        )
        print(
            f"MPSDecompressor.load_master: {total_frames} frames, "
            f"{total_bytes / 1e9:.2f} GB, "
            f"{t_total - t0:.3f}s"
        )
        return result

    def load_master_chunked(self, master_path: str,
                            pixel_mask: "np.ndarray | None" = None,
                            verbose: bool = True,
                            target_bytes: int | None = None,
                            row_prefix: bool = False,
                            fast_det_bin: int | None = None) -> list:
        """Zero-copy no-bin decompress returning a LIST of per-chunk arrays.

        Same zero-copy unified-memory path as load_master (disk reads straight
        into a shared Metal buffer, the GPU decodes in place — NO host->device
        copy), but the bitshuffle output goes to a fresh per-chunk Metal buffer
        (~0.7 GB each) instead of one 19.3 GB buffer. This dodges the M5
        14.3 GB maxBufferLength cap so full no-bin 19.3 GB loads on a 24 GB Mac,
        and avoids the ~2s H2D tensor-copy tax the torch path pays. Each entry
        is a zero-copy _MtlArray view (its Metal buffer kept alive on ._mtl).

        pixel_mask : (det_row, det_col) array, optional
            Dead-pixel mask (nonzero = dead). Each chunk's frames are zeroed at
            the masked pixels right after decode, matching the cuda raw-frame
            contract (dead pixels contribute nothing downstream). Applied as an
            in-place write into the chunk's unified output buffer; it overlaps
            the next chunk's read so it costs ~nothing.
        row_prefix : bool, optional
            Store each detector row as an in-place uint16 prefix sum during
            load. This is the exact no-bin virtual-image interaction layout; it
            avoids a second full-stack conversion when the viewer opens.
        fast_det_bin : int, optional
            Also write a detector-binned sidecar from the decoded raw frames in
            the same Metal command buffer. This avoids the later second
            compressed read/decompress pass used by fast interaction mode.
        """
        t0 = time.perf_counter()
        plan = plan_master(master_path)
        frame_shape = plan.detector_shape
        dtype = plan.dtype
        frame_bytes = plan.frame_bytes
        elem_size = plan.elem_size
        fast_det_bin = int(fast_det_bin or 0)
        if fast_det_bin:
            if row_prefix:
                raise ValueError("fast_det_bin cannot be combined with row_prefix=True.")
            if target_bytes is not None:
                raise ValueError("fast_det_bin is only supported for non-compact loads.")
            if frame_shape[0] % fast_det_bin or frame_shape[1] % fast_det_bin:
                raise ValueError(
                    f"Detector shape {frame_shape} is not divisible by "
                    f"fast_det_bin={fast_det_bin}."
                )
        if row_prefix and elem_size != 2:
            raise ValueError("row_prefix=True requires uint16 detector data.")
        chunk_files = list(plan.chunk_files)
        chunk_n_frames = list(plan.chunk_n_frames)
        n_blocks_per_frame = plan.n_blocks_per_frame
        if row_prefix:
            self._set_bad_pixels(pixel_mask, frame_shape)
            self._prefix_overflow_np[0] = 0
            zero_bad = bool(self._bad_idx_count and elem_size == 2)
        else:
            self._set_bad_pixels(pixel_mask, frame_shape)
            zero_bad = bool(self._bad_idx_count and elem_size == 2)
        if fast_det_bin:
            self._set_mask(pixel_mask, int(frame_shape[0]), int(frame_shape[1]))
        bufs = [
            (self._comp_np, self._co_np, self._bs_np, self._bc_np,
             self._bo_np, self._chunk_sizes,
             self._comp_mtl, self._co_mtl, self._bs_mtl, self._bc_mtl, self._bo_mtl),
            (
                self._comp_np_b, self._co_np_b, self._bs_np_b, self._bc_np_b,
                self._bo_np_b, self._chunk_sizes_b,
                self._comp_mtl_b, self._co_mtl_b, self._bs_mtl_b,
                self._bc_mtl_b, self._bo_mtl_b,
            ),
        ]
        n_chunks = len(chunk_files)
        if n_chunks >= 3 and os.environ.get("QT_MPS_READAHEAD", "1") != "0":
            bufs.append(self._read_ahead_buffer())
        comp_depth = len(bufs)
        # D=2 keeps two command buffers in flight with separate LZ4 scratch
        # buffers. D=3 regressed on the 24 GB Mac because the extra 0.8 GB
        # scratch buffer pushed the full 19.3 GB output into memory pressure.
        # Keep D=2 for GPU scratch, but use a third compressed-input slot so
        # read+parse for chunk N+2 can happen before scratch slot N is free.
        gpu_depth = int(os.environ.get("QT_MPS_GPU_DEPTH", "2"))
        D = min(max(1, gpu_depth), n_chunks)
        if D >= 2 and self._lz4_mtl_b is None:
            self._lz4_mtl_b = _metal_buffer_alloc(self.gpu_batch * frame_bytes)
        while len(self._lz4_mtl_extra) < max(0, D - 2):
            self._lz4_mtl_extra.append(_metal_buffer_alloc(self.gpu_batch * frame_bytes))
        lz4s = [self._lz4_mtl]
        if D >= 2:
            lz4s.append(self._lz4_mtl_b)
        if D > 2:
            lz4s.extend(self._lz4_mtl_extra[: D - 2])
        # Persistent per-chunk output buffer pool: allocate once, reuse every
        # load. Avoids the ~19.3 GB alloc/free churn (page re-zeroing) that made
        # repeated loads 4s instead of 2.6s. Trade-off: a prior load's arrays
        # alias the pool, so the caller must finish with one dataset before
        # loading the next — the GUI shows one dataset at a time, so this is fine.
        compact = target_bytes is not None and int(target_bytes) > frame_bytes
        if compact:
            self._chunk_out_pool = []
            max_frames_per_out = max(1, int(target_bytes) // frame_bytes)
            output_group_for_chunk: list[int] = []
            output_frame_offset: list[int] = []
            output_n_frames: list[int] = []
            current_frames = 0
            current_group = -1
            for nf in chunk_n_frames:
                if current_group < 0 or current_frames + nf > max_frames_per_out:
                    current_group += 1
                    output_n_frames.append(0)
                    current_frames = 0
                output_group_for_chunk.append(current_group)
                output_frame_offset.append(current_frames)
                current_frames += nf
                output_n_frames[current_group] = current_frames
            out_views: list = [None] * len(output_n_frames)
            out_mtls: list = [
                _metal_buffer_alloc(nf * frame_bytes)
                for nf in output_n_frames
            ]
        else:
            pool = self._chunk_out_pool
            fast_pool = self._chunk_fast_pool
            output_group_for_chunk = list(range(n_chunks))
            output_frame_offset = [0] * n_chunks
            output_n_frames = chunk_n_frames
            out_views: list = [None] * n_chunks
            out_mtls: list = [None] * n_chunks
            fast_frame_shape = (
                int(frame_shape[0]) // fast_det_bin,
                int(frame_shape[1]) // fast_det_bin,
            ) if fast_det_bin else None
            fast_frame_bytes = (
                fast_frame_shape[0] * fast_frame_shape[1] * elem_size
            ) if fast_frame_shape else 0
            fast_out_views: list = [None] * n_chunks if fast_det_bin else []
            fast_out_mtls: list = [None] * n_chunks if fast_det_bin else []
        cmds: list = [None] * n_chunks
        mblk: list = [0] * n_chunks

        def _read_parse(ci):
            comp_np, co_np, bs_np, bc_np, bo_np, csizes, *_ = bufs[ci % comp_depth]
            nf = chunk_n_frames[ci]
            self._read_chunk(chunk_files[ci], comp_np, co_np, csizes)
            _parse_headers(comp_np, csizes, co_np, bs_np, bc_np, nf,
                           n_blocks_per_frame)
            bo_np[0] = 0
            bo_np[1 : nf + 1] = np.cumsum(bc_np[:nf])
            mblk[ci] = int(bc_np[:nf].max())

        def _finalize_output(oi):
            nf = output_n_frames[oi]
            view = _numpy_view(out_mtls[oi], dtype, nf * (frame_bytes // elem_size))
            arr = view.reshape((nf,) + frame_shape).view(_MtlArray)
            arr._mtl = out_mtls[oi]  # keep the unified buffer alive on the array
            arr._row_prefix = bool(row_prefix)
            out_views[oi] = arr

        def _finalize_fast_output(oi):
            nf = output_n_frames[oi]
            count = nf * (fast_frame_bytes // elem_size)
            view = _numpy_view(fast_out_mtls[oi], dtype, count)
            arr = view.reshape((nf,) + fast_frame_shape).view(_MtlArray)
            arr._mtl = fast_out_mtls[oi]
            arr._row_prefix = False
            fast_out_views[oi] = arr

        for ci in range(n_chunks):
            if ci >= comp_depth:  # free compressed-input slot before reuse
                cmds[ci - comp_depth].waitUntilCompleted()
            _read_parse(ci)
            nf = chunk_n_frames[ci]
            need = nf * frame_bytes
            if compact:
                oi = output_group_for_chunk[ci]
                out_mtl = out_mtls[oi]
                out_byte_offset = output_frame_offset[ci] * frame_bytes
            else:
                if ci < len(pool) and pool[ci].length() >= need:
                    out_mtl = pool[ci]
                else:
                    out_mtl = _metal_buffer_alloc(need)
                    if ci < len(pool):
                        pool[ci] = out_mtl
                    else:
                        pool.append(out_mtl)
                out_mtls[ci] = out_mtl
                out_byte_offset = 0
            fast_out_mtl = None
            fast_out_byte_offset = 0
            if fast_det_bin:
                fast_need = nf * fast_frame_bytes
                if ci < len(fast_pool) and fast_pool[ci].length() >= fast_need:
                    fast_out_mtl = fast_pool[ci]
                else:
                    fast_out_mtl = _metal_buffer_alloc(fast_need)
                    if ci < len(fast_pool):
                        fast_pool[ci] = fast_out_mtl
                    else:
                        fast_pool.append(fast_out_mtl)
                fast_out_mtls[ci] = fast_out_mtl
            if ci >= D:  # free LZ4 scratch slot before reuse
                cmds[ci - D].waitUntilCompleted()
            comp_mtl, co_mtl, bs_mtl, bc_mtl, bo_mtl = bufs[ci % comp_depth][6:11]
            cmds[ci] = self._submit_gpu(
                nf, frame_bytes, elem_size, out_byte_offset,
                comp_mtl, co_mtl, bs_mtl, bc_mtl, bo_mtl, mblk[ci],
                out_mtl=out_mtl, lz4_mtl=lz4s[ci % D], zero_bad=zero_bad,
                row_prefix=row_prefix, det_shape=frame_shape,
                fast_out_mtl=fast_out_mtl,
                fast_out_byte_offset=fast_out_byte_offset,
                fast_det_bin=fast_det_bin if fast_det_bin else None,
            )
        for ci in range(max(0, n_chunks - D), n_chunks):  # drain the in-flight tail
            cmds[ci].waitUntilCompleted()
        if row_prefix and int(self._prefix_overflow_np[0]) != 0:
            raise RuntimeError(
                "row_prefix=True overflowed uint16 row-prefix storage. "
                "Use the raw full-resolution path for this dataset."
            )
        for oi in range(len(out_views)):  # finalize after ALL GPU done (batch coherency)
            _finalize_output(oi)
            if fast_det_bin:
                _finalize_fast_output(oi)
        if verbose:
            total = sum(chunk_n_frames)
            elapsed = time.perf_counter() - t0
            gbps = total * frame_bytes / elapsed / 1e9 if elapsed > 0 else float("inf")
            label = (
                f"fused bin{fast_det_bin} " if fast_det_bin else
                "row-prefix " if row_prefix else ""
            )
            print(
                f"Loaded {label}chunked MPS data in {elapsed:.2f}s "
                f"({total} frames, {total * frame_bytes / 1e9:.2f} GB, "
                f"{len(out_views)} arrays, {gbps:.1f} GB/s)"
            )
        if fast_det_bin:
            return out_views, fast_out_views
        return out_views

    def load(
        self,
        filepath: str,
        dataset_path: str = "entry/data/data",
        n_frames: int | None = None,
        verbose: bool = True,
    ) -> np.ndarray:
        """Load and decompress a bitshuffle+LZ4 HDF5 dataset via MPS.

        Returns a zero-copy view into the pre-allocated unified memory buffer.
        The view is overwritten on the next load() call.

        Parameters
        ----------
        filepath : str
            Path to the HDF5 file.
        dataset_path : str, optional
            Path to the dataset within the HDF5 file.
        n_frames : int, optional
            Number of frames to load. If None, loads all frames.

        Returns
        -------
        np.ndarray
            Numpy array with shape (n_frames, height, width).
        """
        t0 = time.perf_counter()

        # ---- Read raw chunks directly into pre-allocated Metal buffer ----
        with h5py.File(filepath, "r") as f:
            ds = f[dataset_path]
            total_in_file = ds.shape[0]
            n_frames = min(n_frames, total_in_file) if n_frames else total_in_file
            frame_shape = ds.shape[1:]
            dtype = ds.dtype
            frame_bytes = int(np.prod(frame_shape) * np.dtype(dtype).itemsize)
            offset = 0
            for i in range(n_frames):
                _, raw = ds.id.read_direct_chunk((i, 0, 0))
                chunk_len = len(raw)
                self._co_np[i] = offset
                self._chunk_sizes[i] = chunk_len
                self._comp_np[offset : offset + chunk_len] = np.frombuffer(
                    raw, dtype=np.uint8
                )
                offset += chunk_len
            total_compressed = offset
        t_read = time.perf_counter()

        # ---- Parse headers directly into pre-allocated Metal buffers ----
        # n_blocks_per_frame matches actual block count, so block_starts
        # layout matches block_offsets indexing — no repack needed
        n_blocks_per_frame = (frame_bytes + 8191) // 8192
        _parse_headers(
            self._comp_np, self._chunk_sizes, self._co_np,
            self._bs_np, self._bc_np, n_frames, n_blocks_per_frame,
        )
        self._bo_np[0] = 0
        self._bo_np[1 : n_frames + 1] = np.cumsum(self._bc_np[:n_frames])
        max_blocks_per_frame = int(self._bc_np[:n_frames].max())
        t_parse = time.perf_counter()

        # ---- Single command buffer: LZ4 + barrier + bitshuffle ----
        cmd = _queue.commandBuffer()
        enc = cmd.computeCommandEncoder()

        # LZ4 decompression
        enc.setComputePipelineState_(_h5lz4dc_pipeline)
        enc.setBuffer_offset_atIndex_(self._comp_mtl, 0, 0)
        enc.setBuffer_offset_atIndex_(self._co_mtl, 0, 1)
        enc.setBuffer_offset_atIndex_(self._bs_mtl, 0, 2)
        enc.setBuffer_offset_atIndex_(self._bc_mtl, 0, 3)
        enc.setBuffer_offset_atIndex_(self._bo_mtl, 0, 4)
        enc.setBytes_length_atIndex_(
            np.array([8192], dtype=np.uint32).tobytes(), 4, 5
        )
        enc.setBytes_length_atIndex_(
            np.array([frame_bytes], dtype=np.uint32).tobytes(), 4, 6
        )
        enc.setBuffer_offset_atIndex_(self._lz4_mtl, 0, 7)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(n_frames, 1, (max_blocks_per_frame + 1) // 2),
            Metal.MTLSizeMake(32, 2, 1),
        )

        # Memory barrier between LZ4 output and bitshuffle input
        enc.memoryBarrierWithScope_(Metal.MTLBarrierScopeBuffers)

        # Bitshuffle unshuffle — n_frames in X
        elem_size = np.dtype(dtype).itemsize
        n_8kb = frame_bytes // 8192
        if elem_size == 2:
            groups_per_block = 8192 // (elem_size * 32)  # 128
            groups_per_frame = n_8kb * groups_per_block   # 1152
            frame_u16s = frame_bytes // 2
            enc.setComputePipelineState_(_shuf16_pipeline)
            enc.setBuffer_offset_atIndex_(self._lz4_mtl, 0, 0)
            enc.setBuffer_offset_atIndex_(self._shuf_mtl, 0, 1)
            enc.setBytes_length_atIndex_(
                np.array([frame_u16s], dtype=np.uint32).tobytes(), 4, 2
            )
            enc.setBytes_length_atIndex_(
                np.array([groups_per_block], dtype=np.uint32).tobytes(), 4, 3
            )
            enc.setBytes_length_atIndex_(
                np.array([groups_per_frame], dtype=np.uint32).tobytes(), 4, 4
            )
            # 32 SIMD groups per threadgroup → 32x fewer launches
            tg_count = (groups_per_frame + 31) // 32
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSizeMake(n_frames, 1, tg_count),
                Metal.MTLSizeMake(32, 32, 1),
            )
        else:
            groups_per_block = 2048 // 32  # 64
            groups_per_frame = n_8kb * groups_per_block
            frame_u32s = frame_bytes // 4
            enc.setComputePipelineState_(_shuf32_pipeline)
            enc.setBuffer_offset_atIndex_(self._lz4_mtl, 0, 0)
            enc.setBuffer_offset_atIndex_(self._shuf_mtl, 0, 1)
            enc.setBytes_length_atIndex_(
                np.array([frame_u32s], dtype=np.uint32).tobytes(), 4, 2
            )
            enc.setBytes_length_atIndex_(
                np.array([groups_per_block], dtype=np.uint32).tobytes(), 4, 3
            )
            enc.setBytes_length_atIndex_(
                np.array([groups_per_frame], dtype=np.uint32).tobytes(), 4, 4
            )
            tg_count = (groups_per_frame + 31) // 32
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSizeMake(n_frames, 1, tg_count),
                Metal.MTLSizeMake(32, 32, 1),
            )

        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        t_gpu = time.perf_counter()

        # ---- Zero-copy result via unified memory ----
        total_bytes = n_frames * frame_bytes
        result = self._result_np[:total_bytes].view(dtype).reshape(
            (n_frames,) + frame_shape
        )
        t_total = time.perf_counter()

        if verbose:
            print(
                f"MPSDecompressor.load: {n_frames} frames, "
                f"{total_compressed / 1e6:.0f} MB compressed → "
                f"{total_bytes / 1e6:.0f} MB decompressed"
            )
            print(
                f"  read: {t_read - t0:.3f}s | "
                f"parse: {t_parse - t_read:.3f}s | "
                f"gpu: {t_gpu - t_parse:.3f}s | "
                f"total: {t_total - t0:.3f}s"
            )
        return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_master_chunked(
    master_path: str,
    *,
    pixel_mask: np.ndarray | None = None,
    apply_mask: bool = True,
    verbose: bool = True,
    target_bytes: int | None = None,
    row_prefix: bool = False,
    fast_det_bin: int | None = None,
) -> list:
    """Explicit zero-copy MPS no-bin IO step returning Metal-backed chunks."""
    plan = plan_master(master_path)
    if pixel_mask is None and apply_mask:
        pixel_mask = _read_pixel_mask(plan.master_path)
    dec = _get_cached_decompressor(plan.frame_bytes, max(plan.chunk_n_frames))
    return dec.load_master_chunked(
        plan.master_path,
        pixel_mask=pixel_mask,
        verbose=verbose,
        target_bytes=target_bytes,
        row_prefix=row_prefix,
        fast_det_bin=fast_det_bin,
    )


def load_mps_4dstem(
    master_path: str,
    *,
    scan_shape: tuple[int, int] | None = None,
    apply_mask: bool = True,
    pixel_mask: np.ndarray | None = None,
    verbose: bool = True,
    compact: bool = False,
    compact_target_gb: float = 3.6,
    row_prefix: bool = False,
    det_bin: int = 1,
    fast_det_bin: int | None = None,
    skip_mps_memory_check: bool | None = None,
) -> MPSChunked4DSTEM:
    """Load full no-bin Arina data as zero-copy MPS chunks for viewing.

    This is the explicit IO half of the Apple-Silicon viewer path. It performs
    disk IO + Metal decompression only, returning chunk arrays that remain in
    unified-memory Metal buffers. Pass the result to
    ``quantem.widget.show4dstem_mps.show_4dstem_mps`` to display it
    without copying.

    ``fast_det_bin`` defaults to ``None`` so a plain no-bin load sits at the
    theoretical floor — exactly the data (19.3 GB for 512x512x192x192), with no
    bin2 viewer sidecar. The viewer builds that sidecar itself on first scrub
    (``ChunkedFrames.ensure_fast_interaction``), so compute paths (dpc/virtual)
    never pay for it and a 24 GB Mac no longer goes into swap on load. Pass
    ``fast_det_bin=2`` to eagerly fuse the sidecar in the same decode pass when
    you know a viewer is about to open and want zero first-scrub latency.
    """
    t0 = time.perf_counter()
    plan = plan_master(master_path)
    det_bin = int(det_bin)
    if det_bin < 1:
        raise ValueError("det_bin must be >= 1.")
    if det_bin > 1 and row_prefix:
        raise ValueError("row_prefix=True is only valid for det_bin=1.")
    if det_bin > 1 or row_prefix:
        fast_det_bin = None
    elif fast_det_bin is not None:
        fast_det_bin = int(fast_det_bin)
        if fast_det_bin <= 1:
            fast_det_bin = None
    _check_mps_memory_guard(
        plan,
        det_bin=det_bin,
        skip_memory_check=skip_mps_memory_check,
    )
    target_bytes = int(float(compact_target_gb) * 1e9) if compact else None
    if verbose:
        layout = (
            f"detector-bin{det_bin} "
            if det_bin > 1 else
            f"full+bin{fast_det_bin} "
            if fast_det_bin else
            "row-prefix " if row_prefix else ""
        )
        print(f"Loading {layout}MPS chunks from {os.path.basename(plan.master_path)}")
    fast_chunks = None
    if det_bin > 1:
        if pixel_mask is None and apply_mask:
            pixel_mask = _read_pixel_mask(plan.master_path)
        arr = load_master(
            plan.master_path,
            det_bin=det_bin,
            pixel_mask=pixel_mask,
            verbose=False,
        )
        chunks = [arr]
    else:
        result = load_master_chunked(
            plan.master_path,
            pixel_mask=pixel_mask,
            apply_mask=apply_mask,
            verbose=False,
            target_bytes=target_bytes,
            row_prefix=row_prefix,
            fast_det_bin=fast_det_bin,
        )
        if fast_det_bin:
            chunks, fast_chunks = result
        else:
            chunks = result
            fast_chunks = None
    # Drop the cached decompressor once the result is built. The returned arrays
    # each own their Metal output buffer via ``arr._mtl``, so they survive the
    # clear; only the decoder's reusable scratch (lz4 + compressed staging +
    # read-ahead) and its pool list lose their Python refs. This lets a later
    # ``del result`` actually release the data instead of the cached decompressor
    # pinning it forever, and keeps the no-bin phys_footprint at ~20.2 GB on a
    # device 512x512x192x192 load (19.3 GB data + ~0.9 GB decode working set).
    # The headline footprint win is ``fast_det_bin=None`` above (no +4.8 GB bin2
    # sidecar); the clear is the second-order hygiene that stops the decompressor
    # from holding a second reference to the whole pool. Only cost: the next load
    # re-zeroes the 19.3 GB pool (~1.4 s) instead of reusing it.
    clear_mps_cache()
    import gc

    gc.collect()
    inferred_scan = scan_shape
    if inferred_scan is None:
        root = int(round(plan.ntrigger ** 0.5))
        if root * root == plan.ntrigger:
            inferred_scan = (root, root)
    metadata = {
        "backend": "mps",
        "master_path": plan.master_path,
        "scan_shape": inferred_scan,
        "detector_shape": tuple(int(x) // det_bin for x in plan.detector_shape),
        "raw_detector_shape": plan.detector_shape,
        "det_bin": det_bin,
        "dtype": str(plan.dtype),
        "n_frames": plan.total_frames,
        "chunk_n_frames": list(plan.chunk_n_frames),
        "n_chunks": len(chunks),
        "nbytes": int(sum(int(c.nbytes) for c in chunks)),
        "fast_det_bin": int(fast_det_bin) if fast_det_bin else None,
        "fast_detector_shape": (
            tuple(int(x) // int(fast_det_bin) for x in plan.detector_shape)
            if fast_det_bin else None
        ),
        "fast_nbytes": (
            int(sum(int(c.nbytes) for c in fast_chunks))
            if fast_det_bin and fast_chunks is not None else 0
        ),
        "zero_copy": True,
        "row_prefix": bool(row_prefix),
    }
    data = MPSChunked4DSTEM(
        chunks=chunks,
        metadata=metadata,
        master_path=plan.master_path,
        scan_shape=inferred_scan,
        row_prefix=bool(row_prefix),
        det_bin=det_bin,
        fast_chunks=fast_chunks if det_bin == 1 else None,
        fast_det_bin=int(fast_det_bin) if fast_det_bin else None,
    )
    if verbose:
        elapsed = time.perf_counter() - t0
        gbps = data.nbytes / elapsed / 1e9 if elapsed > 0 else float("inf")
        layout = (
            f" detector-bin{det_bin}"
            if det_bin > 1 else
            f" full+bin{fast_det_bin}"
            if fast_det_bin else
            " row-prefix" if row_prefix else ""
        )
        total_bytes = data.nbytes + int(metadata.get("fast_nbytes") or 0)
        print(
            f"Loaded MPS{layout} chunks in {elapsed:.2f}s "
            f"({data.n_frames} frames, {total_bytes / 1e9:.2f} GB, "
            f"{gbps:.1f} GB/s raw)"
        )
    return data


_decompressor_cache: dict[int, MPSDecompressor] = {}

# GPU sub-batch sizing.
#
# Each chunk may contain 100K+ frames (e.g. a 100K-frame dataset: 100,352 frames
# × 73,728 bytes/frame = 7.4 GB per chunk). Allocating full-chunk _lz4 +
# _shuf intermediate buffers (2 × 7.4 GB = 14.8 GB) plus the output buffer
# exceeds 24 GB, causing Metal to swap to SSD and GPU time to spike from
# ~3s to 30s+.
#
# Solution: process each chunk in sub-batches of ~7K frames (0.5 GB target
# per intermediate buffer, ~2 GB total Metal). Benchmarked on M5 24 GB:
#   batch=5000  → 4.9s total  (optimal — fits L2/SLC well)
#   batch=10000 → 5.5s        (slight pressure)
#   batch=20000 → 6.5s
#   batch=40000 → 9.3s        (memory pressure begins)
#   batch=100K  → 13.1s       (significant GPU stalls)
#
# After loading, _out_mtl is freed but the decompressor is cached (keeps
# _lz4/_shuf + metadata buffers ≈ 1.5 GB). Warm loads skip shader
# compilation and buffer allocation: ~0.7s for 65K frames on local NVMe.
_GPU_BATCH_TARGET_GB = 0.5


def _get_decompressor(frame_bytes, max_frames=11_000):
    """Get or create a decompressor sized for the given frame byte size."""
    cache_key = (frame_bytes, max_frames)
    if cache_key not in _decompressor_cache:
        n_blocks = (frame_bytes + 8191) // 8192
        # Scale compressed buffer: worst observed bitshuffle+LZ4 ratio ~7:1,
        # use //4 for headroom (386 MB for uint32, 256 MB for uint16)
        max_comp = max(256 * 1024 * 1024, max_frames * frame_bytes // 4)
        # Cap _lz4/_shuf buffers to ~3 GB each to avoid Metal memory pressure
        gpu_batch = min(max_frames,
                        int(_GPU_BATCH_TARGET_GB * 1e9 / frame_bytes))
        _decompressor_cache[cache_key] = MPSDecompressor(
            max_compressed_bytes=max_comp,
            max_frames=max_frames,
            frame_bytes=frame_bytes,
            n_blocks_per_frame=n_blocks,
            gpu_batch=gpu_batch,
        )
    return _decompressor_cache[cache_key]


def _parse_master_uncached(master_path: str) -> MPSMasterPlan:
    """Read metadata and chunk file list from an Arina master file."""
    master_path = os.path.abspath(os.fspath(master_path))
    master_dir = os.path.dirname(master_path)
    prefix = os.path.basename(master_path).replace("_master.h5", "")
    with h5py.File(master_path, "r") as f:
        chunk_keys = sorted(f["entry/data"].keys())
        ds0 = f[f"entry/data/{chunk_keys[0]}"]
        det_shape = tuple(int(x) for x in ds0.shape[1:])
        dtype = np.dtype(ds0.dtype)
        spec = f["entry/instrument/detector/detectorSpecific"]
        ntrigger = int(spec["ntrigger"][()])
    chunk_files = []
    chunk_n_frames = []
    missing = []
    for k in chunk_keys:
        suffix = k.split("_")[-1]
        cf = os.path.join(master_dir, f"{prefix}_data_{suffix}.h5")
        if not os.path.exists(cf):
            missing.append(os.path.basename(cf))
            continue
        chunk_files.append(cf)
        with h5py.File(cf, "r") as f:
            chunk_n_frames.append(f["entry/data/data"].shape[0])
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)}/{len(chunk_keys)} chunk files for "
            f"{os.path.basename(master_path)}: {missing}"
        )
    return MPSMasterPlan(
        master_path=master_path,
        detector_shape=det_shape,
        dtype=dtype,
        ntrigger=ntrigger,
        chunk_files=tuple(chunk_files),
        chunk_n_frames=tuple(int(n) for n in chunk_n_frames),
    )


def plan_master(master_path: str) -> MPSMasterPlan:
    """Return the cached MPS load plan for an Arina master."""
    key = _file_cache_key(master_path)
    cached = _master_plan_cache.get(key)
    if cached is not None:
        return cached
    plan = _parse_master_uncached(master_path)
    _master_plan_cache[key] = plan
    return plan


def _parse_master(master_path):
    """Compatibility tuple for older callers."""
    plan = plan_master(master_path)
    return (
        plan.detector_shape,
        plan.dtype,
        plan.ntrigger,
        list(plan.chunk_files),
        list(plan.chunk_n_frames),
    )


def load_arina(
    master_path: str,
    det_bin: int = 1,
    scan_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Load an arina 4D-STEM dataset with Metal GPU decompression.

    Decompresses bitshuffle+LZ4 data on the GPU and optionally bins the
    detector axes on the fly. With binning, only the smaller binned result
    is kept in memory, so datasets larger than RAM can be loaded.

    Parameters
    ----------
    master_path : str
        Path to the arina master HDF5 file.
    det_bin : int, optional
        Detector binning factor (applied to both axes), by default 1.
    scan_shape : tuple of (int, int), optional
        Reshape into (scan_rows, scan_cols, det_rows, det_cols). If None
        and det_bin > 1, inferred as (sqrt(n), sqrt(n)) from ntrigger.

    Returns
    -------
    np.ndarray
        If scan_shape: (scan_rows, scan_cols, det_rows, det_cols) float32.
        Otherwise: (n_frames, det_rows, det_cols) in original dtype.

    Examples
    --------
    >>> data = load_arina("SnMoS2s_001_master.h5")
    >>> data.shape
    (262144, 192, 192)

    >>> data = load_arina("SnMoS2s_001_master.h5", det_bin=2)
    >>> data.shape
    (512, 512, 96, 96)
    """
    t0 = time.perf_counter()
    det_shape, dtype, ntrigger, chunk_files, chunk_n_frames = _parse_master(
        master_path
    )
    total_frames = sum(chunk_n_frames)
    det_row, det_col = det_shape
    elem_size = np.dtype(dtype).itemsize
    frame_bytes = int(np.prod(det_shape) * elem_size)
    n_blocks_per_frame = (frame_bytes + 8191) // 8192
    max_chunk_frames = max(chunk_n_frames)
    dec = _get_decompressor(frame_bytes, max_frames=max_chunk_frames)
    gpu_batch = dec.gpu_batch
    # Compute output shape
    if det_bin > 1:
        out_det_row = det_row // det_bin
        out_det_col = det_col // det_bin
        out_dtype = np.float32
    else:
        out_det_row, out_det_col = det_row, det_col
        out_dtype = dtype
    # Warmup numba JIT (skip if already warmed)
    if not getattr(dec, '_jit_warm', False):
        dec.load(chunk_files[0], n_frames=1, verbose=False)
        dec._jit_warm = True
    if det_bin > 1:
        # GPU pipeline: LZ4 → bitshuffle → bin, sub-batched to fit in GPU mem
        out_frame_bytes = out_det_row * out_det_col * 4  # float32
        # Batch-sized Metal output buffer (not total — copy to numpy per batch)
        batch_out_bytes = gpu_batch * out_frame_bytes
        dec._ensure_output_buffer(batch_out_bytes)
        # Allocate numpy output array
        output = np.empty((total_frames, out_det_row, out_det_col),
                          dtype=np.float32)
        # Warmup binned GPU pipeline (triggers Metal lazy buffer mapping)
        if not getattr(dec, '_bin_warm', False):
            bufs_0 = (dec._comp_np, dec._co_np, dec._bs_np, dec._bc_np,
                       dec._bo_np, dec._chunk_sizes,
                       dec._comp_mtl, dec._co_mtl, dec._bs_mtl, dec._bc_mtl,
                       dec._bo_mtl)
            comp_np_w, co_np_w, bs_np_w, bc_np_w, bo_np_w, csizes_w, \
                comp_mtl_w, co_mtl_w, bs_mtl_w, bc_mtl_w, bo_mtl_w = bufs_0
            dec._read_chunk(chunk_files[0], comp_np_w, co_np_w, csizes_w)
            _parse_headers(comp_np_w, csizes_w, co_np_w, bs_np_w, bc_np_w,
                           1, n_blocks_per_frame)
            bo_np_w[0] = 0
            bo_np_w[1] = int(bc_np_w[0])
            cmd_w = dec._submit_gpu_binned(
                1, frame_bytes, elem_size, 0,
                det_row, det_col, det_bin,
                comp_mtl_w, co_mtl_w, bs_mtl_w, bc_mtl_w, bo_mtl_w,
                int(bc_np_w[0]),
            )
            cmd_w.waitUntilCompleted()
            dec._bin_warm = True
        bufs = [
            (dec._comp_np, dec._co_np, dec._bs_np, dec._bc_np,
             dec._bo_np, dec._chunk_sizes,
             dec._comp_mtl, dec._co_mtl, dec._bs_mtl, dec._bc_mtl,
             dec._bo_mtl),
            (dec._comp_np_b, dec._co_np_b, dec._bs_np_b, dec._bc_np_b,
             dec._bo_np_b, dec._chunk_sizes_b,
             dec._comp_mtl_b, dec._co_mtl_b, dec._bs_mtl_b, dec._bc_mtl_b,
             dec._bo_mtl_b),
        ]
        # Read first chunk
        comp_np, co_np, bs_np, bc_np, bo_np, csizes, \
            comp_mtl, co_mtl, bs_mtl, bc_mtl, bo_mtl = bufs[0]
        dec._read_chunk(chunk_files[0], comp_np, co_np, csizes)
        _parse_headers(comp_np, csizes, co_np, bs_np, bc_np,
                       chunk_n_frames[0], n_blocks_per_frame)
        bo_np[0] = 0
        bo_np[1 : chunk_n_frames[0] + 1] = np.cumsum(bc_np[:chunk_n_frames[0]])
        frame_offset = 0
        n_chunks = len(chunk_files)
        total_batches = sum((nf + gpu_batch - 1) // gpu_batch
                            for nf in chunk_n_frames)
        pbar = tqdm(total=total_batches, desc="GPU", leave=False) \
            if total_batches > 1 else None
        for ci in range(n_chunks):
            cur = bufs[ci % 2]
            comp_np, co_np, bs_np, bc_np, bo_np, csizes, \
                comp_mtl, co_mtl, bs_mtl, bc_mtl, bo_mtl = cur
            nf = chunk_n_frames[ci]
            max_blk = int(bc_np[:nf].max())
            # Process chunk in sub-batches of gpu_batch
            for b_start in range(0, nf, gpu_batch):
                b_end = min(b_start + gpu_batch, nf)
                nb = b_end - b_start
                cmd = dec._submit_gpu_binned(
                    nb, frame_bytes, elem_size, 0,
                    det_row, det_col, det_bin,
                    comp_mtl, co_mtl, bs_mtl, bc_mtl, bo_mtl,
                    max_blk, meta_frame_offset=b_start,
                )
                # Overlap: read next chunk while last batch of current runs
                if b_start + gpu_batch >= nf and ci + 1 < n_chunks:
                    nxt = bufs[(ci + 1) % 2]
                    comp_np_n, co_np_n, bs_np_n, bc_np_n, bo_np_n, \
                        csizes_n, *_ = nxt
                    dec._read_chunk(chunk_files[ci + 1], comp_np_n, co_np_n,
                                    csizes_n)
                    nf_next = chunk_n_frames[ci + 1]
                    _parse_headers(comp_np_n, csizes_n, co_np_n, bs_np_n,
                                   bc_np_n, nf_next, n_blocks_per_frame)
                    bo_np_n = nxt[4]
                    bo_np_n[0] = 0
                    bo_np_n[1 : nf_next + 1] = np.cumsum(bc_np_n[:nf_next])
                cmd.waitUntilCompleted()
                # Copy batch result from Metal buffer to numpy output
                src = dec._out_np[:nb * out_frame_bytes]
                output[frame_offset:frame_offset + nb] = (
                    src.view(np.float32).reshape(nb, out_det_row, out_det_col)
                )
                frame_offset += nb
                if pbar:
                    pbar.update(1)
        if pbar:
            pbar.close()
    else:
        # No binning — use load_master for double-buffered raw decompression
        output = dec.load_master(chunk_files[0].replace(
            "_data_000001.h5", "_master.h5"
        ))
    # Free the batch output Metal buffer (kept buffers are small: ~1.5 GB)
    dec._out_mtl = None
    dec._out_np = None
    dec._out_nbytes = 0
    t_total = time.perf_counter()
    # Infer scan shape
    if scan_shape is None and det_bin > 1:
        side = int(total_frames ** 0.5)
        if side * side == total_frames:
            scan_shape = (side, side)
    if scan_shape is not None:
        output = output.reshape(*scan_shape, out_det_row, out_det_col)
    print(
        f"load_arina: {total_frames} frames, "
        f"det ({det_row},{det_col}) → ({out_det_row},{out_det_col}), "
        f"{t_total - t0:.2f}s"
    )
    return output


# ---------------------------------------------------------------------------
# Torch-native MPS decompressor (no-bin path)
# ---------------------------------------------------------------------------
# Dispatches the SAME Metal kernels (_METAL_SOURCE) through
# torch.mps.compile_shader with torch MPS tensors as buffers, instead of the
# PyObjC command-buffer path above. The output is a LIST of per-chunk torch MPS
# uint16 tensors (one MPS tensor cannot exceed ~14.3 GB on a 24 GB Mac, and the
# full no-bin stack is 19.3 GB, so a single tensor is impossible — the list is
# the contract). Bit-identical to h5py ground truth.
#
# Performance (a 512²×192² logic dataset, 262144x192x192 uint16 = 19.3 GB, warm cache,
# M5 24 GB): ~2.6 s end-to-end vs ~4.6 s for the serial PyObjC path. The win
# comes from two structural changes mirrored from the CUDA _load_master_pipelined:
#   1. disk read ‖ GPU decode — ONE producer thread reads + header-parses chunk
#      N+1 (CPU/disk, GIL released) into a free host slot while the main thread
#      GPU-decodes chunk N in order, so wall ≈ read(chunk0) + max(rest_read,
#      all_gpu) instead of the serial sum.
#   2. zero-copy host→GPU — the producer reads compressed bytes straight into an
#      MPS unified-memory tensor's numpy view (Apple unified memory), so the
#      compressed payload (the big buffer) never pays a separate .to("mps") copy.
# A SINGLE producer (not a pool) is deliberate: numba's parallel=True
# _parse_headers is NOT threadsafe when called from multiple Python threads
# ("The workqueue threading layer is not threadsafe"), so the read+parse stage
# must be serialized. One producer suffices: read+parse projects to ~0.8 s for
# all 27 chunks, comfortably under the ~1.7 s GPU floor, so it hides fully.
# The remaining floor is GPU compute (~1.7 s warm: LZ4 ~1.1 s + bitshuffle
# ~0.6 s). Driving below ~1.7 s requires LZ4-kernel surgery (two-phase block
# parse / LZ4+bitshuffle fusion) in _METAL_SOURCE; the structural wins above
# are independent of that and bit-exact today.

_torch_lib_cache: dict[str, object] = {}


def _torch_shader_lib():
    """Compile (once) the Metal kernels via torch.mps.compile_shader.

    Cached per LZ4_Y substitution. Reuses the exact _METAL_SOURCE the PyObjC
    path uses, so the torch-dispatched kernels are byte-identical.
    """
    import torch.mps as _tmps
    key = str(_LZ4_Y)
    if key not in _torch_lib_cache:
        _torch_lib_cache[key] = _tmps.compile_shader(
            _METAL_SOURCE.replace("LZ4_BLOCKS_PER_TG", key)
        )
    return _torch_lib_cache[key]


def load_master_torch(
    master_path: str,
    *,
    pixel_mask: "np.ndarray | None" = None,
    nbuf: int = 5,
    verbose: bool = True,
) -> "list":
    """Decompress an arina master to a LIST of per-chunk torch MPS uint16
    tensors on the Apple GPU, no detector binning.

    Each list element is a ``torch.Tensor`` on device ``mps`` with shape
    ``(chunk_frames, det_row, det_col)`` and native uint16 dtype. Concatenate
    on the caller side only if the result fits (19.3 GB exceeds the 14.3 GB
    single-tensor limit on a 24 GB Mac, which is why this returns a list).

    Mirrors the CUDA ``_load_master_pipelined`` design: ONE producer thread
    reads + header-parses chunk N+1 (CPU/disk, GIL released) into a free host
    slot whose compressed buffer is an MPS unified-memory tensor's numpy view
    (zero host→GPU copy), while the main thread GPU-decodes chunk N in order.
    Bit-exact to h5py ``entry/data/data``.

    A single producer (not a pool) is required because numba's ``parallel=True``
    ``_parse_headers`` is not threadsafe across Python threads. One producer
    suffices: read+parse hides fully under the GPU decode.

    Parameters
    ----------
    master_path : str
        Path to the arina master HDF5 file.
    pixel_mask : np.ndarray or None, optional
        Dead-pixel mask (nonzero = dead). When given, dead pixels are zeroed in
        every frame (matches the cuda raw-frame contract).
    nbuf : int, optional
        Host buffer slots, by default 3 (1 GPU-decoding + 1 reading + 1 ready).
    verbose : bool, optional
        Print wall time, by default True.
    """
    import threading
    import queue
    import torch
    import torch.mps

    t0 = time.perf_counter()
    lib = _torch_shader_lib()
    det_shape, dtype, _ntrigger, chunk_files, chunk_n_frames = _parse_master(master_path)
    det_row, det_col = det_shape
    elem_size = int(np.dtype(dtype).itemsize)
    frame_bytes = int(np.prod(det_shape) * elem_size)
    frame_u16s = frame_bytes // 2
    n_blocks_per_frame = (frame_bytes + 8191) // 8192
    groups_per_block = 8192 // (elem_size * 32)
    groups_per_frame = n_blocks_per_frame * groups_per_block
    tg_count = (groups_per_frame + 31) // 32
    dev = torch.device("mps")
    n_chunks = len(chunk_files)
    max_frames = max(chunk_n_frames)
    nbuf = max(2, min(nbuf, n_chunks))
    dec = _get_decompressor(frame_bytes, max_frames=max_frames)
    comp_cap = dec._comp_np.shape[0]

    class _Slot:
        __slots__ = ("comp_t", "comp_np", "co", "cs", "bs", "bc", "bo")

        def __init__(self):
            # CPU staging buffer the producer reads disk into; copied to an MPS
            # tensor per chunk in _gpu. MPS tensors are NOT host-writable, so a
            # true zero-copy numpy view (comp_t.numpy()) is impossible — the
            # per-chunk H2D copy (~0.39s) is hidden under the GPU decode by the
            # producer/consumer overlap.
            self.comp_np = np.empty(comp_cap, dtype=np.uint8)
            self.co = np.empty(max_frames, np.uint32)
            self.cs = np.empty(max_frames, np.uint32)
            self.bs = np.empty(max_frames * n_blocks_per_frame, np.uint32)
            self.bc = np.empty(max_frames, np.uint32)
            self.bo = np.empty(max_frames + 1, np.uint32)

    slots = [_Slot() for _ in range(nbuf)]

    def _read_parse(ci, s):
        nf = chunk_n_frames[ci]
        dec._read_chunk(chunk_files[ci], s.comp_np, s.co, s.cs)
        _parse_headers(s.comp_np, s.cs, s.co, s.bs, s.bc, nf, n_blocks_per_frame)
        s.bo[0] = 0
        s.bo[1 : nf + 1] = np.cumsum(s.bc[:nf])
        return nf, int(s.co[nf - 1] + s.cs[nf - 1]), int(s.bc[:nf].max())

    def _gpu_enqueue(nf, total_comp, max_blk, s, bad_t):
        # Enqueue (async) the H2D copies + LZ4 + bitshuffle for one chunk. NO
        # per-chunk sync: MPS commands stay ordered on the queue and the GPU runs
        # them back-to-back while the read+parse threads fill ahead. The slot's
        # CPU staging buffers must stay un-reused until a later wave sync.
        comp_t = torch.from_numpy(s.comp_np[:total_comp]).to(dev)
        co_t = torch.from_numpy(s.co[:nf]).to(dev)
        bs_t = torch.from_numpy(s.bs[: nf * n_blocks_per_frame]).to(dev)
        bc_t = torch.from_numpy(s.bc[:nf]).to(dev)
        bo_t = torch.from_numpy(s.bo[: nf + 1]).to(dev)
        lz4 = torch.empty(nf * frame_bytes, dtype=torch.uint8, device=dev)
        out = torch.empty(nf * frame_u16s, dtype=torch.uint16, device=dev)
        zblk = (max_blk + _LZ4_Y - 1) // _LZ4_Y
        lib.h5lz4dc_batched(
            comp_t, co_t, bs_t, bc_t, bo_t, 8192, frame_bytes, lz4,
            threads=(nf * 32, _LZ4_Y, zblk), group_size=(32, _LZ4_Y, 1),
        )
        lib.shuf_8192_16_batched(
            lz4.view(torch.uint16), out, frame_u16s, groups_per_block,
            groups_per_frame,
            threads=(nf * 32, 32, tg_count), group_size=(32, 32, 1),
        )
        out = out.view(nf, det_row, det_col)
        if bad_t is not None:
            out[:, bad_t] = 0  # zero dead pixels (enqueued, matches cuda)
        return out

    # Three-stage pipeline: read thread -> parse thread -> GPU consumer. read(N+1)
    # overlaps parse(N) overlaps GPU(N-1), so the producer wall = max(read, parse)
    # instead of read+parse, and it all hides under the GPU decode floor.
    free_q: "queue.Queue" = queue.Queue()
    for i in range(nbuf):
        free_q.put(i)
    parse_q: "queue.Queue" = queue.Queue()
    ready_q: "queue.Queue" = queue.Queue()
    _SENT = object()

    prof = {"read": 0.0, "read_wait": 0.0, "parse": 0.0, "parse_wait": 0.0,
            "gpu": 0.0, "sync": 0.0, "cons_wait": 0.0}

    def _read_thread():
        for ci in range(n_chunks):
            _w = time.perf_counter()
            slot_idx = free_q.get()
            prof["read_wait"] += time.perf_counter() - _w
            s = slots[slot_idx]
            nf = chunk_n_frames[ci]
            _t = time.perf_counter()
            dec._read_chunk(chunk_files[ci], s.comp_np, s.co, s.cs)
            prof["read"] += time.perf_counter() - _t
            parse_q.put((ci, slot_idx, nf))
        parse_q.put(_SENT)

    def _parse_thread():
        while True:
            _w = time.perf_counter()
            item = parse_q.get()
            prof["parse_wait"] += time.perf_counter() - _w
            if item is _SENT:
                ready_q.put(_SENT)
                return
            ci, slot_idx, nf = item
            s = slots[slot_idx]
            _t = time.perf_counter()
            _parse_headers(s.comp_np, s.cs, s.co, s.bs, s.bc, nf, n_blocks_per_frame)
            s.bo[0] = 0
            s.bo[1 : nf + 1] = np.cumsum(s.bc[:nf])
            prof["parse"] += time.perf_counter() - _t
            ready_q.put((ci, slot_idx, nf, int(s.co[nf - 1] + s.cs[nf - 1]),
                         int(s.bc[:nf].max())))

    threading.Thread(target=_read_thread, daemon=True).start()
    threading.Thread(target=_parse_thread, daemon=True).start()

    if pixel_mask is not None:
        bad = np.asarray(pixel_mask) != 0
        bad_t = (
            torch.from_numpy(bad).to(dev)
            if bad.shape == (det_row, det_col)
            else None
        )
    else:
        bad_t = None

    # Wave sync: hold up to nbuf-1 chunks' GPU work in flight, then sync once and
    # release that whole wave of slots. GPU saturated across the wave; one sync
    # per nbuf-1 chunks instead of per chunk.
    outputs: "list" = []
    wave: "list" = []  # slot_idx held until next sync
    while True:
        _w = time.perf_counter()
        item = ready_q.get()
        prof["cons_wait"] += time.perf_counter() - _w
        if item is _SENT:
            break
        ci, slot_idx, nf, total_comp, max_blk = item
        _t = time.perf_counter()
        outputs.append(_gpu_enqueue(nf, total_comp, max_blk, slots[slot_idx], bad_t))
        prof["gpu"] += time.perf_counter() - _t
        wave.append(slot_idx)
        if len(wave) >= nbuf - 1:
            _t = time.perf_counter()
            torch.mps.synchronize()  # GPU done reading this wave's staging
            prof["sync"] += time.perf_counter() - _t
            for si in wave:
                free_q.put(si)
            wave = []
    _t = time.perf_counter()
    torch.mps.synchronize()  # drain final wave
    prof["sync"] += time.perf_counter() - _t
    for si in wave:
        free_q.put(si)
    if verbose:
        total_frames = sum(chunk_n_frames)
        gb = total_frames * frame_bytes / 1e9
        print(
            f"load_master_torch: {total_frames} frames, {gb:.1f} GB, "
            f"{len(outputs)} MPS tensors, {time.perf_counter() - t0:.2f}s"
        )
        print("  PROF " + "  ".join(f"{k}={v:.2f}" for k, v in prof.items()))
    return outputs
