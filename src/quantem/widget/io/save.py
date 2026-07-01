"""CUDA bitshuffle+LZ4 saving for 4D-STEM HDF5 files."""
from __future__ import annotations

import queue
import threading
from pathlib import Path

# cupy guarded so `import quantem.widget.io` (which pulls this module) works on a
# non-CUDA box. Writing compressed HDF5 is a cuda-only operation; on a Mac this
# module imports fine and only errors if save() is actually called.
try:
    import cupy as cp
except ImportError:  # pragma: no cover - exercised only on non-CUDA hosts
    cp = None
import h5py
import hdf5plugin  # noqa: F401 - registers bitshuffle filter
import numpy as np
from numba import njit, prange

from .constants import BLOCK_SIZE


# Lazy bitshuffle+LZ4 forward/compress kernel proxies (see hdf5.py for the
# rationale). Compile only on first call, inside the cuda save path.
def _lazy_kernel(_name):
    _cache = []

    def _call(*args, **kwargs):
        if not _cache:
            from . import bitshuffle as _bs
            _cache.append(getattr(_bs, _name))
        return _cache[0](*args, **kwargs)

    return _call


_bitshuffle_fwd_kernel = _lazy_kernel("bitshuffle_fwd_kernel")
_bitshuffle_fwd_kernel_u16 = _lazy_kernel("bitshuffle_fwd_kernel_u16")
_bitshuffle_fwd_tail_kernel_u16 = _lazy_kernel("bitshuffle_fwd_tail_kernel_u16")
_bitshuffle_fwd_tail_kernel_u32 = _lazy_kernel("bitshuffle_fwd_tail_kernel_u32")
_lz4_compress_kernel = _lazy_kernel("lz4_compress_kernel")
_lz4_compress_var_kernel = _lazy_kernel("lz4_compress_var_kernel")
_pack_h5_chunks_kernel = _lazy_kernel("pack_h5_chunks_kernel")


# Compression codecs available to save() / H5Writer. Default 'lz4' uses the
# GPU bitshuffle+LZ4 fast path that streams compressed bytes via
# write_direct_chunk. The other codecs let HDF5's CPU plugin pipeline run
# (bigger ratio, slower) — pick them for archival writes where size matters
# more than write throughput.
_COMPRESSION_CODECS = ("lz4", "zstd", "blosc2_zstd")
_DEFAULT_CLEVELS = {"zstd": 3, "blosc2_zstd": 5}


# =========================================================================
#  GPU HDF5 saving
# =========================================================================

_SAVE_DTYPES = (
    np.dtype(np.uint16),
    np.dtype(np.uint32),
    # float16 intentionally NOT supported: 10-bit mantissa makes the
    # smallest representable step at value V equal to V / 1024. For typical
    # detector counts up to ~3000, that's a step of ~3 counts — WORSE than
    # uint16 round-quantize (max error 0.5). Use uint16 for lossy or
    # float32 for lossless.
    np.dtype(np.float32),
)
_DTYPE_ALIASES = {
    "u16": np.uint16,
    "uint16": np.uint16,
    "u32": np.uint32,
    "uint32": np.uint32,
    "f32": np.float32,
    "float32": np.float32,
    "f16": np.float16,
    "float16": np.float16,
}
_write_queue = queue.Queue()
_write_thread = None
_write_thread_lock = threading.Lock()
_write_error = None


@njit(cache=True, parallel=True)
def _pack_chunks_numba(packed, chunk_starts, compact, sizes, offsets,
                       header_bytes, n_frames, n_8kb):
    """Pack bitshuffle+LZ4 blocks into HDF5 chunk byte buffers."""
    for i in prange(n_frames):
        dst = chunk_starts[i]
        for h in range(12):
            packed[dst + h] = header_bytes[h]
        pos = dst + 12
        base = i * n_8kb
        for b in range(n_8kb):
            idx = base + b
            sz = sizes[idx]
            packed[pos] = (sz >> 24) & 0xFF
            packed[pos + 1] = (sz >> 16) & 0xFF
            packed[pos + 2] = (sz >> 8) & 0xFF
            packed[pos + 3] = sz & 0xFF
            pos += 4
            off = offsets[idx]
            for j in range(sz):
                packed[pos + j] = compact[off + j]
            pos += sz


def _pack_chunks(compact_np, sizes_np, offsets_np, n_frames, n_8kb,
                 frame_bytes):
    """Return packed HDF5 chunk bytes, per-frame starts, and sizes."""
    import struct

    sizes_2d = sizes_np.reshape(n_frames, n_8kb)
    frame_comp_sizes = sizes_2d.sum(axis=1).astype(np.int64)
    header_overhead = 12 + n_8kb * 4
    chunk_sizes = header_overhead + frame_comp_sizes

    chunk_starts = np.zeros(n_frames + 1, dtype=np.int64)
    np.cumsum(chunk_sizes, out=chunk_starts[1:])
    packed = np.empty(int(chunk_starts[n_frames]), dtype=np.uint8)
    header_bytes = np.frombuffer(
        struct.pack(">QI", frame_bytes, BLOCK_SIZE), dtype=np.uint8
    ).copy()

    _pack_chunks_numba(
        packed, chunk_starts, compact_np,
        sizes_np.astype(np.int64), offsets_np.astype(np.int64),
        header_bytes, n_frames, n_8kb,
    )
    return packed, chunk_starts[:n_frames], chunk_sizes


def _pack_chunks_gpu(comp_buf, sizes_gpu, n_frames, n_8kb, frame_bytes, max_out):
    """Pack bitshuffle+LZ4 blocks into HDF5 chunk bytes on GPU."""
    sizes_2d = sizes_gpu.reshape(n_frames, n_8kb)
    frame_comp_sizes = sizes_2d.sum(axis=1, dtype=cp.uint64)
    header_overhead = np.uint64(12 + n_8kb * 4)
    chunk_sizes_gpu = frame_comp_sizes + header_overhead
    chunk_starts_gpu = cp.empty(n_frames + 1, dtype=cp.uint64)
    chunk_starts_gpu[0] = 0
    chunk_starts_gpu[1:] = cp.cumsum(chunk_sizes_gpu)
    packed_bytes = int(chunk_starts_gpu[-1].get())
    packed_gpu = cp.empty(packed_bytes, dtype=cp.uint8)
    _pack_h5_chunks_kernel(
        (n_frames,), (256,),
        (
            comp_buf,
            sizes_gpu,
            chunk_starts_gpu,
            packed_gpu,
            np.uint32(n_frames),
            np.uint32(n_8kb),
            np.uint32(max_out),
            np.uint64(frame_bytes),
            np.uint32(BLOCK_SIZE),
        ),
    )
    cp.cuda.Stream.null.synchronize()
    packed = packed_gpu.get()
    chunk_starts = chunk_starts_gpu[:-1].get().astype(np.int64, copy=False)
    chunk_sizes = chunk_sizes_gpu.get().astype(np.int64, copy=False)
    del packed_gpu, chunk_starts_gpu, chunk_sizes_gpu, frame_comp_sizes
    return packed, chunk_starts, chunk_sizes


def _write_batch_to_h5(ds, packed, chunk_starts, chunk_sizes, frame_offset,
                       n_frames):
    """Write packed compressed chunks to an open flattened HDF5 dataset."""
    packed_mv = memoryview(packed)
    for i in range(n_frames):
        start = int(chunk_starts[i])
        size = int(chunk_sizes[i])
        ds.id.write_direct_chunk(
            (frame_offset + i, 0, 0), packed_mv[start:start + size]
        )


def _write_batch_via_filter(ds, host_frames, frame_offset, n_frames):
    """Write a CPU host-side frame batch through the dataset's filter pipeline.

    Used for non-LZ4 codecs (zstd, blosc2_zstd) where compression runs on CPU
    inside HDF5 instead of in our GPU pipeline. ``host_frames`` is a contiguous
    numpy array of shape (n_frames, det_row, det_col).
    """
    ds[frame_offset:frame_offset + n_frames] = host_frames


def _normalize_compression(compression, compression_level):
    if compression is None:
        compression = "lz4"
    compression = str(compression).lower()
    if compression not in _COMPRESSION_CODECS:
        raise ValueError(
            f"Unsupported compression {compression!r}; choose from "
            f"{_COMPRESSION_CODECS}."
        )
    if compression == "lz4":
        # LZ4 path is GPU bitshuffle+LZ4; level is fixed in the kernel.
        if compression_level not in (None, 0):
            raise ValueError(
                "compression_level is only meaningful for zstd / blosc2_zstd; "
                "leave it 0 for the default LZ4 path."
            )
        return compression, 0
    level = compression_level if compression_level else _DEFAULT_CLEVELS[compression]
    return compression, int(level)


def _hdf5_filter(compression, compression_level):
    """Return the hdf5plugin filter mapping for a given codec+level."""
    if compression == "lz4":
        return hdf5plugin.Bitshuffle(cname="lz4")
    if compression == "zstd":
        return hdf5plugin.Bitshuffle(cname="zstd", clevel=compression_level)
    if compression == "blosc2_zstd":
        return hdf5plugin.Blosc2(
            cname="zstd",
            clevel=compression_level,
            filters=hdf5plugin.Blosc2.BITSHUFFLE,
        )
    raise ValueError(f"Unsupported compression {compression!r}")


def _compress_batch(data_gpu, n_8kb, frame_bytes):
    """Compress a contiguous 16-bit or 32-bit frame batch on GPU."""
    max_out = BLOCK_SIZE * 2
    cuda_max_z = 65535
    n = int(data_gpu.shape[0])
    n_blocks = n * n_8kb
    itemsize = int(data_gpu.dtype.itemsize)
    n_full_8kb = frame_bytes // BLOCK_SIZE
    tail_bytes = frame_bytes % BLOCK_SIZE
    if tail_bytes:
        tail_items = tail_bytes // itemsize
        if tail_bytes % itemsize or tail_items % 8:
            raise ValueError(
                "GPU bitshuffle/LZ4 save supports partial final blocks only "
                "when the partial detector frame contains a multiple of 8 "
                f"elements; got frame_bytes={frame_bytes}, dtype={data_gpu.dtype}."
            )

    shuffled = cp.empty(n * frame_bytes, dtype=cp.uint8)
    if itemsize == 2:
        data_u16 = data_gpu.reshape(n, -1).view(cp.uint16)
        for start in range(0, n, cuda_max_z):
            end = min(start + cuda_max_z, n)
            batch_n = end - start
            out = shuffled[start * frame_bytes:end * frame_bytes]
            if n_full_8kb:
                _bitshuffle_fwd_kernel_u16(
                    (n_full_8kb, 16, batch_n), (256, 1, 1),
                    (
                        data_u16[start:end],
                        out,
                        np.uint32(frame_bytes),
                    ),
                )
            if tail_bytes:
                tail_bitplane_bytes = (tail_bytes // itemsize) // 8
                _bitshuffle_fwd_tail_kernel_u16(
                    ((tail_bitplane_bytes + 127) // 128, 16, batch_n),
                    (128, 1, 1),
                    (
                        data_u16[start:end],
                        out,
                        np.uint32(frame_bytes),
                    ),
                )
    elif itemsize == 4:
        frame_u32s = frame_bytes // 4
        data_u32 = data_gpu.reshape(n, -1).view(cp.uint32)
        for start in range(0, n, cuda_max_z):
            end = min(start + cuda_max_z, n)
            batch_n = end - start
            out = shuffled[start * frame_bytes:end * frame_bytes]
            if n_full_8kb:
                _bitshuffle_fwd_kernel(
                    (n_full_8kb, 2, batch_n), (32, 32, 1),
                    (
                        data_u32[start:end],
                        out.view(cp.uint32),
                        np.uint32(frame_u32s),
                    ),
                )
            if tail_bytes:
                tail_bitplane_bytes = (tail_bytes // itemsize) // 8
                _bitshuffle_fwd_tail_kernel_u32(
                    ((tail_bitplane_bytes + 127) // 128, 32, batch_n),
                    (128, 1, 1),
                    (
                        data_u32[start:end],
                        out,
                        np.uint32(frame_bytes),
                    ),
                )
    else:
        raise TypeError(f"Unsupported save dtype itemsize: {itemsize}")

    comp_buf = cp.empty(n_blocks * max_out, dtype=cp.uint8)
    sizes_gpu = cp.empty(n_blocks, dtype=cp.uint32)
    if tail_bytes:
        _lz4_compress_var_kernel(
            (n_blocks,), (32,),
            (
                shuffled,
                comp_buf,
                sizes_gpu,
                np.uint32(frame_bytes),
                np.uint32(BLOCK_SIZE),
                np.uint32(max_out),
                np.uint32(n_8kb),
                np.uint32(n_blocks),
            ),
        )
    else:
        _lz4_compress_kernel(
            (n_blocks,), (32,),
            (shuffled, comp_buf, sizes_gpu, np.uint32(BLOCK_SIZE), np.uint32(n_blocks)),
        )
    del shuffled

    packed, chunk_starts, chunk_sizes = _pack_chunks_gpu(
        comp_buf, sizes_gpu, n, n_8kb, frame_bytes, max_out
    )
    del comp_buf, sizes_gpu
    return packed, chunk_starts, chunk_sizes


def _raise_write_error():
    global _write_error
    if _write_error is None:
        return
    err = _write_error
    _write_error = None
    raise RuntimeError("Background HDF5 write failed") from err


def _writer_loop():
    global _write_error
    while True:
        job = _write_queue.get()
        try:
            if job is None:
                return
            func, args = job
            func(*args)
        except BaseException as exc:  # propagate on wait_for_saves()
            _write_error = exc
        finally:
            _write_queue.task_done()


def _ensure_writer_thread():
    global _write_thread
    with _write_thread_lock:
        if _write_thread is None or not _write_thread.is_alive():
            _write_thread = threading.Thread(target=_writer_loop, daemon=True)
            _write_thread.start()


def wait_for_saves() -> None:
    """Block until queued HDF5 write jobs finish, then raise write errors."""
    _write_queue.join()
    _raise_write_error()


def _normalize_save_dtype(dtype):
    if isinstance(dtype, str):
        dtype = _DTYPE_ALIASES.get(dtype.lower(), dtype)
    dtype = np.dtype(dtype)
    if dtype not in _SAVE_DTYPES:
        raise ValueError(
            "save() supports float32 for canonical drift-corrected 4D-STEM "
            "archives, with integer dtypes reserved for explicit raw-data or "
            "compatibility exports."
        )
    return dtype



def _default_save_dtype(data_dtype):
    data_dtype = np.dtype(data_dtype)
    if np.issubdtype(data_dtype, np.floating):
        return np.dtype(np.float32)
    if data_dtype == np.dtype(np.uint16):
        return np.dtype(np.uint16)
    if data_dtype == np.dtype(np.uint32):
        return np.dtype(np.uint32)
    raise ValueError(
        f"Unsupported input dtype {data_dtype}; pass dtype=np.float32, "
        "np.uint16, or np.uint32 explicitly."
    )

def _metadata_attrs(f, metadata):
    if metadata is None:
        return
    for key, val in metadata.items():
        if val is None:
            continue
        try:
            f.attrs[key] = val
        except (TypeError, ValueError):
            f.attrs[key] = str(val)


def _output_file_size(master_path):
    master_path = Path(master_path)
    prefix = _master_prefix(master_path)
    total = master_path.stat().st_size if master_path.exists() else 0
    for data_path in master_path.parent.glob(f"{prefix}_data_*.h5"):
        total += data_path.stat().st_size
    return total


def _master_prefix(master_path):
    stem = master_path.stem
    return stem[:-7] if stem.endswith("_master") else stem


def _copy_master_shell(source_master, output_master):
    """Copy source Arina master metadata, excluding entry/data links."""
    with h5py.File(source_master, "r") as src, h5py.File(output_master, "w") as dst:
        for key, val in src.attrs.items():
            dst.attrs[key] = val
        for name in src.keys():
            if name != "entry":
                src.copy(name, dst, name=name, expand_external=False)
        if "entry" in src:
            src_entry = src["entry"]
            dst_entry = dst.require_group("entry")
            for key, val in src_entry.attrs.items():
                dst_entry.attrs[key] = val
            for name in src_entry.keys():
                if name == "data":
                    continue
                src.copy(src_entry[name], dst_entry, name=name, expand_external=False)
        else:
            dst.require_group("entry")


def _write_master_file(
    master_path,
    data_files,
    frame_ranges,
    scan_shape,
    det_shape,
    dtype,
    metadata,
    source_master,
):
    master_path = Path(master_path)
    if source_master is not None:
        _copy_master_shell(source_master, master_path)
        mode = "r+"
    else:
        mode = "w"

    with h5py.File(master_path, mode) as f:
        entry = f.require_group("entry")
        if "data" in entry:
            del entry["data"]
        data_group = entry.create_group("data")
        if source_master is not None:
            with h5py.File(source_master, "r") as src:
                src_data = src.get("entry/data")
                if src_data is not None:
                    for key, val in src_data.attrs.items():
                        data_group.attrs[key] = val
        else:
            data_group.attrs["NX_class"] = np.bytes_(b"NXdata")
        data_group.attrs["signal"] = np.bytes_(b"data_000001")
        for index, data_file in enumerate(data_files, start=1):
            data_group[f"data_{index:06d}"] = h5py.ExternalLink(
                Path(data_file).name, "/entry/data/data"
            )
        if scan_shape is not None:
            data_group.attrs["scan_shape"] = tuple(int(x) for x in scan_shape)
            f.attrs["scan_shape"] = tuple(int(x) for x in scan_shape)
        data_group.attrs["det_shape"] = tuple(int(x) for x in det_shape)
        data_group.attrs["dtype"] = str(np.dtype(dtype))
        data_group.attrs["n_frames"] = int(sum(hi - lo + 1 for lo, hi in frame_ranges))
        data_group.attrs["data_file_frame_ranges"] = np.asarray(frame_ranges, dtype=np.uint64)
        f.attrs["detector_shape"] = tuple(int(x) for x in det_shape)
        f.attrs["dtype"] = str(np.dtype(dtype))
        f.attrs["n_frames"] = int(sum(hi - lo + 1 for lo, hi in frame_ranges))
        _metadata_attrs(f, metadata)


class H5Writer:
    """Streaming Arina-style GPU writer for bitshuffle+LZ4 4D-STEM files.

    The output is a master HDF5 file with external data files:
    ``name_master.h5`` plus ``name_data_000001.h5``, etc. Each external
    data file contains ``entry/data/data`` with shape
    ``(n_frames_in_file, detector_row, detector_col)`` and HDF5 chunks
    ``(1, detector_row, detector_col)``. This matches the row/column
    Arina/Dectris layout closely enough for existing chunked readers.

    ``float32`` is the intended dtype for canonical drift-corrected 4D-STEM
    archives because bilinear correction creates fractional detector values.
    Pass a different dtype only when intentionally making a non-default export.
    """

    def __init__(self, filepath, n_frames, det_shape, scan_shape=None,
                 metadata=None, dtype=np.float32, source_master=None,
                 frames_per_file=32768, compression="lz4",
                 compression_level=0):
        self._filepath = Path(filepath)
        self._n_frames = int(n_frames)
        self._det_row, self._det_col = (int(det_shape[0]), int(det_shape[1]))
        self._dtype = _normalize_save_dtype(dtype)
        self._compression, self._compression_level = _normalize_compression(
            compression, compression_level
        )
        self._frame_bytes = self._det_row * self._det_col * self._dtype.itemsize
        self._n_8kb = (self._frame_bytes + BLOCK_SIZE - 1) // BLOCK_SIZE
        self._scan_shape = None if scan_shape is None else tuple(int(x) for x in scan_shape)
        self._metadata = metadata
        self._source_master = source_master
        self._frames_per_file = int(frames_per_file)
        if self._frames_per_file <= 0:
            raise ValueError("frames_per_file must be positive")
        self._frame_offset = 0
        self._file_index = 0
        self._current_file = None
        self._current_ds = None
        self._current_file_n = 0
        self._current_file_start = 0
        self._current_file_offset = 0
        self._data_files = []
        self._frame_ranges = []
        self._closed = False
        self._prefix = _master_prefix(self._filepath)
        self._filepath.parent.mkdir(parents=True, exist_ok=True)

        wait_for_saves()
        _ensure_writer_thread()

    def _open_data_file(self):
        self._file_index += 1
        self._current_file_start = self._frame_offset
        remaining = self._n_frames - self._frame_offset
        self._current_file_n = min(self._frames_per_file, remaining)
        self._current_file_offset = 0
        data_path = self._filepath.with_name(f"{self._prefix}_data_{self._file_index:06d}.h5")
        self._data_files.append(data_path)
        lo = self._current_file_start + 1
        hi = self._current_file_start + self._current_file_n
        self._frame_ranges.append((lo, hi))

        self._current_file = h5py.File(data_path, "w")
        self._current_ds = self._current_file.create_dataset(
            "entry/data/data",
            shape=(self._current_file_n, self._det_row, self._det_col),
            dtype=self._dtype,
            chunks=(1, self._det_row, self._det_col),
            **_hdf5_filter(self._compression, self._compression_level),
        )
        self._current_ds.attrs["image_nr_low"] = np.uint64(lo)
        self._current_ds.attrs["image_nr_high"] = np.uint64(hi)

    def _close_data_file(self):
        if self._current_file is None:
            return
        wait_for_saves()
        self._current_file.close()
        self._current_file = None
        self._current_ds = None
        self._current_file_n = 0
        self._current_file_offset = 0

    def write(self, data_gpu):
        """Compress and queue a frame batch.

        ``data_gpu`` may be a CuPy or NumPy array with shape
        ``(n_batch, detector_row, detector_col)``. The dtype is cast to the
        writer dtype before compression.
        """
        if self._closed:
            raise RuntimeError("H5Writer is closed")
        _raise_write_error()
        if not isinstance(data_gpu, cp.ndarray):
            data_gpu = cp.asarray(np.asarray(data_gpu))
        if data_gpu.dtype != self._dtype:
            # Float→integer cast: round to nearest BEFORE casting so bilinear-merged
            # 4D-STEM keeps max-error 0.5 counts (sub-noise-floor) instead of the 1.0
            # max-error you get from truncation. Numpy/CuPy default float->uint cast
            # truncates fractional parts.
            if (np.issubdtype(data_gpu.dtype, np.floating)
                    and np.issubdtype(self._dtype, np.integer)):
                lo, hi = (int(np.iinfo(self._dtype).min),
                          int(np.iinfo(self._dtype).max))
                data_gpu = cp.clip(cp.rint(data_gpu), lo, hi).astype(self._dtype)
            else:
                data_gpu = data_gpu.astype(self._dtype)
        if data_gpu.ndim != 3 or data_gpu.shape[1:] != (self._det_row, self._det_col):
            raise ValueError(
                f"Expected batch shape (n, {self._det_row}, {self._det_col}), "
                f"got {tuple(data_gpu.shape)}"
            )
        if self._frame_offset + int(data_gpu.shape[0]) > self._n_frames:
            raise ValueError("Batch would exceed declared n_frames")

        data_gpu = cp.ascontiguousarray(data_gpu)
        batch_start = 0
        batch_n = int(data_gpu.shape[0])
        while batch_start < batch_n:
            if self._current_file is None:
                self._open_data_file()
            room = self._current_file_n - self._current_file_offset
            n_part = min(room, batch_n - batch_start)
            part = data_gpu[batch_start:batch_start + n_part]
            if self._compression == "lz4":
                packed, starts, sizes = _compress_batch(
                    part, self._n_8kb, self._frame_bytes
                )
                _write_queue.put((
                    _write_batch_to_h5,
                    (self._current_ds, packed, starts, sizes,
                     self._current_file_offset, n_part),
                ))
            else:
                # Non-LZ4 codecs run inside HDF5's filter pipeline on CPU.
                # Pull the batch to host once, queue the filtered write so
                # GPU work continues while compression happens on a worker.
                host_frames = cp.asnumpy(part)
                _write_queue.put((
                    _write_batch_via_filter,
                    (self._current_ds, host_frames,
                     self._current_file_offset, n_part),
                ))
            self._current_file_offset += n_part
            self._frame_offset += n_part
            batch_start += n_part
            if self._current_file_offset == self._current_file_n:
                self._close_data_file()

    def close(self, wait: bool = False):
        """Finalize external data files and write the master file."""
        if self._closed:
            if wait:
                wait_for_saves()
            return
        self._closed = True
        self._close_data_file()
        if self._frame_offset != self._n_frames:
            raise RuntimeError(f"H5Writer wrote {self._frame_offset} of {self._n_frames} frames")
        _write_master_file(
            self._filepath,
            self._data_files,
            self._frame_ranges,
            self._scan_shape,
            (self._det_row, self._det_col),
            self._dtype,
            self._metadata,
            self._source_master,
        )
        if wait:
            wait_for_saves()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close(wait=True)
        return False

    def __del__(self):
        if not getattr(self, "_closed", True):
            self.close(wait=False)


def _prepare_save_data(data, dtype, scan_shape):
    input_dtype = data.dtype if isinstance(data, cp.ndarray) else np.asarray(data).dtype
    dtype = _normalize_save_dtype(dtype if dtype is not None else _default_save_dtype(input_dtype))
    data_gpu = data if isinstance(data, cp.ndarray) else cp.asarray(np.asarray(data))
    if data_gpu.dtype != dtype:
        if (np.issubdtype(data_gpu.dtype, np.floating)
                and np.issubdtype(dtype, np.integer)):
            lo, hi = int(np.iinfo(dtype).min), int(np.iinfo(dtype).max)
            data_gpu = cp.clip(cp.rint(data_gpu), lo, hi).astype(dtype)
        else:
            data_gpu = data_gpu.astype(dtype)
    if data_gpu.ndim == 4:
        inferred_scan = tuple(int(x) for x in data_gpu.shape[:2])
        if scan_shape is not None and tuple(scan_shape) != inferred_scan:
            raise ValueError(f"scan_shape={scan_shape} does not match data shape {inferred_scan}")
        scan_shape = inferred_scan
        data_gpu = data_gpu.reshape(-1, data_gpu.shape[-2], data_gpu.shape[-1])
    elif data_gpu.ndim != 3:
        raise ValueError("save() expects 3D frames or 4D-STEM data")
    data_gpu = cp.ascontiguousarray(data_gpu)
    return data_gpu, dtype, scan_shape


def save(
    filepath: str,
    data: "np.ndarray | cp.ndarray",
    scan_shape: tuple[int, int] | None = None,
    metadata: dict | None = None,
    dtype: type | np.dtype | None = None,
    batch_size: int = 4096,
    wait: bool = True,
    verbose: bool = False,
    source_master: str | None = None,
    frames_per_file: int = 32768,
    compression: str = "lz4",
    compression_level: int = 0,
) -> None:
    """Save 4D-STEM data as an Arina-style bitshuffle+LZ4 HDF5 set.

    Output: a master HDF5 file pointing to ``*_data_NNNNNN.h5`` external files
    with per-frame HDF5 chunks. Matches Arina row/column native chunking. By
    default uses GPU bitshuffle+LZ4 (the fastest path); pass ``compression=``
    to switch codecs.

    Drift-correction recipe (the canonical use case)
    ------------------------------------------------
    Bilinear merging of a 0°/+90° pair produces a float32 4D-STEM where every
    detector cell holds a weighted average of two integer counts. Lossless
    float32 + LZ4 compresses these fractional values to ~2× ratio (huge files,
    slow). Quantizing back to ``uint16`` recovers the integer-count statistics
    of the underlying detector, gives 10× better compression, runs ~4× faster,
    and keeps max error 0.5 counts which is far below the detector's own
    Poisson noise (~√N counts at signal level N)::

        # Drift-corrected merged float32 → save as uint16 (recommended):
        save("corrected_master.h5", merged_f32,
             scan_shape=(512, 512),
             dtype="u16")            # ← the single line that matters

    Float→integer casts use ``cp.rint`` (round-half-to-even) followed by
    ``cp.clip`` to the dtype range, NOT truncation. Max error is exactly half
    a count for any value in range; truncation would double it.

    Performance - a 512^2 x 192^2 float32 bilinear-merged stack
    -----------------------------------------------------------
    Measured on RTX PRO 6000 Blackwell (workstation), real bilinear-merged data:

        =========================== ============= ============= ====== ========
        dtype                       wall          file size     ratio  GB/s in
        =========================== ============= ============= ====== ========
        ``float32`` (lossless)      ~49 s         19.5 GB       1.98x  0.78
        ``uint16`` (round-quantize) ~41 s         3.4 GB        11.24x 0.94
        =========================== ============= ============= ====== ========

    For smaller (256²) scans the wall scales linearly. Pure synthetic data
    compresses 22× rather than 2× because random integer counts have heavy
    bit-level repetition; bilinear-merged real data is the realistic ceiling.

    Quality — what "max_err = 0.5 counts" means
    -------------------------------------------
    The detector measures integer photon counts. Poisson noise floor at signal
    level N counts is √N, so:

        =========== =============== =====================================
        Mean signal Noise floor (σ)  uint16 quant error / σ
        =========== =============== =====================================
        100 counts  10 counts        0.5 / 10 = 5%
        1000        ~32              0.5 / 32 = 1.6%
        4000        ~63              0.5 / 63 = 0.8%
        =========== =============== =====================================

    Quantization is well below the data's own statistical noise. For
    ptychography, drift correction, virtual imaging, etc., this is
    indistinguishable from lossless. If 0.5 counts still feels too coarse,
    scale up before quantizing — store ``round(merged * 10)`` as uint16,
    divide by 10 on read; max error becomes 0.05 counts at ~30% ratio cost.

    Why ``uint16`` not ``int16``
    -----------------------------
    Detector counts are non-negative by physics. Unsigned uses the full
    [0, 65535] range; signed wastes a bit on the negative half and risks
    clipping bright Bragg spots > 32767. Use ``np.uint16``.

    ``float16`` is intentionally NOT supported
    -------------------------------------------
    10-bit mantissa makes the smallest representable step at value V equal
    to V / 1024. For typical STEM detector counts up to ~3000, that's a step
    of ~3 counts — *worse* than uint16's 0.5 max error AND larger than the
    Poisson noise floor at low signals. float16 saves are pure noise, never
    use them. Calls with ``dtype=np.float16`` raise ``ValueError``.

    Drift metadata co-saved with the 4D-STEM
    ----------------------------------------
    Save BOTH the spline knot positions (compact, model-of-record) and the
    dense per-position offsets (ready for ptycho without re-evaluation). Pass
    them via ``metadata=`` (root attrs) or write into the master file::

        save("corrected_master.h5", merged_f32, dtype=np.uint16, metadata={
            "drift_model": "spline_n16",
            "drift_knots": knots,                     # (n_imgs, 2, n_knots)
            "drift_probe_positions_px": probe_pos,    # (N_scan_pos, 2)
        })

    Knots = small (kilobytes), regenerable, source of truth.
    Probe positions = dense (~2 MB at 512²), consumed directly by ptycho.

    Parameters
    ----------
    filepath : str
        Output master HDF5 path. External data files are written next to it
        with the same prefix.
    data : np.ndarray | cp.ndarray
        4D-STEM data. Shape (N, det_row, det_col) or (scan_y, scan_x,
        det_row, det_col). CuPy arrays save without a host copy.
    scan_shape : tuple[int, int] | None
        Scan grid shape. Required for 3D inputs; inferred from 4D inputs.
    dtype : str or np.dtype or None
        Output dtype. ``None`` uses input dtype. Short aliases such as
        ``"u16"`` and ``"f32"`` are accepted. **For drift-corrected
        bilinear-merged float32 inputs, pass "u16" explicitly** (10×
        smaller, 4× faster, sub-noise-floor error).
    batch_size : int
        Frames compressed per GPU pass. 4096 is the sweet spot for 192² det.
    compression : {"lz4", "zstd", "blosc2_zstd"}
        ``lz4`` (default, GPU pipeline) is fastest. ``zstd``/``blosc2_zstd``
        run on CPU, give a few extra % ratio at 5-20× the wall time.
    compression_level : int
        Codec level. 0 = codec default. Ignored for LZ4.
    metadata : dict | None
        Saved as root attributes on the master file. Use for drift knots,
        probe positions, calibration.

    See also
    --------
    quantem.widget.io.load : Round-trip read of these files; bit-exact for
        lossless dtypes; near-lossless (≤0.5 count) for uint16-quantized.
    """
    import time

    t0 = time.perf_counter()
    data_gpu, dtype, scan_shape = _prepare_save_data(data, dtype, scan_shape)
    n_frames, det_row, det_col = (int(x) for x in data_gpu.shape)

    writer = H5Writer(
        filepath,
        n_frames=n_frames,
        det_shape=(det_row, det_col),
        scan_shape=scan_shape,
        metadata=metadata,
        dtype=dtype,
        source_master=source_master,
        frames_per_file=frames_per_file,
        compression=compression,
        compression_level=compression_level,
    )
    try:
        for start in range(0, n_frames, int(batch_size)):
            writer.write(data_gpu[start:start + int(batch_size)])
    finally:
        writer.close(wait=wait)

    if verbose:
        elapsed = time.perf_counter() - t0
        file_size = _output_file_size(filepath)
        raw = n_frames * det_row * det_col * dtype.itemsize
        codec = writer._compression
        if codec != "lz4":
            codec = f"{codec}@{writer._compression_level}"
        print(
            f"Saved {filepath} [{codec}]: {raw / 1e9:.2f} GB -> "
            f"{file_size / 1e9:.2f} GB ({raw / file_size:.1f}x) in {elapsed:.2f}s"
        )
