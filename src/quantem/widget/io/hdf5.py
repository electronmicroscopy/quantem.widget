"""
GPU-accelerated HDF5 loading for 4D-STEM diffraction data.

This module provides high-performance bitshuffle+LZ4 decompression
using CUDA kernels, achieving 4-8x speedup over CPU.

Public API
----------
load : Load HDF5 data to GPU with auto-detection of file format.
bin : Bin data on GPU along detector, scan, or all axes.

Examples
--------
>>> from quantem.live import load
>>> from quantem.widget.io import bin
>>> data = load('/path/to/file.h5').data
>>> binned = bin(data, factor=2)
"""

from __future__ import annotations

from typing import Literal, NamedTuple

# cupy is the CUDA toolkit, absent on a Mac / plain laptop. Guard it so this
# module imports anywhere and the view/screen path (backend='cpu'/'mps') works
# without CUDA. Every `cp.<x>` use below sits inside a function that only runs
# on the cuda backend, so on an NVIDIA box `cp` is the real module (zero
# overhead) and on a non-CUDA box those functions are never reached. The
# `from __future__ import annotations` line keeps `cp.ndarray` annotations as
# strings so they never evaluate at import.
try:
    import cupy as cp
except ImportError:  # pragma: no cover - exercised only on non-CUDA hosts
    cp = None
import h5py
import hdf5plugin  # noqa: F401 - registers bitshuffle filter
import numpy as np
from numba import njit, prange

from .constants import BLOCK_SIZE
from .save import H5Writer, save, wait_for_saves


# Lazy bitshuffle+LZ4 kernel proxies. The kernels compile only on first CALL
# (inside the cuda decompress path), so importing this module never touches
# cupy. Each proxy resolves + caches its real kernel on first use, so the
# per-launch cost after that is one list read.
def _lazy_kernel(_name):
    _cache = []

    def _call(*args, **kwargs):
        if not _cache:
            from . import bitshuffle as _bs
            _cache.append(getattr(_bs, _name))
        return _cache[0](*args, **kwargs)

    return _call


_h5lz4dc_kernel = _lazy_kernel("h5lz4dc_kernel")
_bitshuffle_kernel = _lazy_kernel("bitshuffle_kernel")
_bitshuffle_kernel_u16 = _lazy_kernel("bitshuffle_kernel_u16")
_bitshuffle_tail_kernel_u16 = _lazy_kernel("bitshuffle_tail_kernel_u16")
_bitshuffle_tail_kernel_u32 = _lazy_kernel("bitshuffle_tail_kernel_u32")

__version__ = "0.0.3"
__all__ = [
    "load", "load_parallel", "disk_of", "group_by_disk", "save", "H5Writer", "wait_for_saves", "bin",
    "discover_masters", "is_master_ready", "read_pixel_mask", "__version__",
]


def read_pixel_mask(filepath):
    """Return the Arina pixel_mask array from a master HDF5.

    The Arina detector writes a 2-D `pixel_mask` dataset under
    `entry/instrument/detector/detectorSpecific/` enumerating hardware
    dead pixels (>0 = bad). This is the ONLY sanctioned reader - other
    modules must go through here instead of opening h5py directly, so
    the Arina schema stays in one place.

    Parameters
    ----------
    filepath : str or Path
        Path to an Arina master HDF5 file.

    Returns
    -------
    np.ndarray or None
        Raw (H, W) mask array as stored in the HDF5, or None if the
        file is missing/unreadable or has no `pixel_mask` dataset.
    """
    from pathlib import Path
    try:
        with h5py.File(str(Path(filepath)), "r") as f:
            key = "entry/instrument/detector/detectorSpecific/pixel_mask"
            if key not in f:
                return None
            return f[key][:]
    except (OSError, KeyError):
        return None


# =========================================================================
#  CPU helper (Numba JIT)
# =========================================================================

class LoadResult(NamedTuple):
    """Result from load() containing data and metadata.

    Attributes
    ----------
    data : cp.ndarray
        The loaded data as a CuPy array on GPU. Shape is 4D
        ``(scan_r, scan_c, det_r, det_c)`` when ``scan_shape`` is known
        (auto-derived from the HDF5 file or passed explicitly), else 3D
        ``(n_frames, det_r, det_c)``.
    metadata : dict
        Acquisition and detector metadata from the HDF5 file. See
        :func:`get_metadata` for the full spec. The dict mixes two layers:

        **Derived, named fields** (always present; value is ``None`` when
        the source field is missing):

        - ``scan_shape`` : ``(H, W)`` or ``None``
            Auto-derived from ``ntrigger`` assuming a square scan.
        - ``n_frames`` : ``int`` or ``None``
            Total frame count.
        - ``dwell_time_us`` : ``float`` or ``None``
            Per-frame dwell in microseconds.
        - ``detector_shape`` : ``(H, W)`` or ``None``
            Detector pixel count.
        - ``detector_name`` : ``str`` or ``None``
            Human-readable detector description.
        - ``saturation`` : ``int`` or ``None``
            ADU ceiling before the detector saturates.

        **Raw HDF5 scalars**: every scalar dataset in the file keyed by its
        full HDF5 path (e.g. ``metadata["entry/instrument/detector/count_time"]``),
        as an escape hatch for fields not in the derived layer.

        .. note::

            Scope-side parameters (``voltage_kV``, ``semiangle``,
            ``scan_sampling``, ``camera_length``, ``rotation``) are NOT in
            the h5 master - pass them to ``ssb()`` explicitly.

    Examples
    --------
    >>> data, meta = load('gold_master.h5')
    >>> data.shape
    (512, 512, 192, 192)
    >>> meta['scan_shape']
    (512, 512)
    >>> meta['dwell_time_us']
    99.6
    >>> meta['detector_name']
    'Dectris ARINA Si'
    """

    data: cp.ndarray
    metadata: dict


def _apply_scan_shape(
    data: "cp.ndarray",
    explicit: tuple[int, int] | None,
    meta: dict,
) -> "cp.ndarray":
    """Reshape 3D ``(N, det_r, det_c)`` → 4D ``(scan_r, scan_c, det_r, det_c)``.

    Uses ``explicit`` when the caller passed ``scan_shape=``, else
    ``meta["scan_shape"]`` (auto-derived from ``ntrigger``). No-op when
    data is already 4D or no shape is available.
    """
    shape = explicit if explicit is not None else meta.get("scan_shape")
    if shape is None or data.ndim != 3:
        return data
    scan_r, scan_c = shape
    if scan_r * scan_c != data.shape[0]:
        raise ValueError(
            f"scan_shape {shape} incompatible with frame count {data.shape[0]}"
        )
    dr, dc = data.shape[-2:]
    return data.reshape(scan_r, scan_c, dr, dc)


def get_metadata(filepath: str) -> dict:
    """Read all scalar metadata from an HDF5 master file.

    Returns a flat dict that mixes two layers:

    **Derived, named fields** (always present as keys; value is ``None`` when
    the source field is missing from the file):

    - ``scan_shape`` : tuple[int, int] or None
        Scan grid as ``(height, width)``. Derived from ``ntrigger`` assuming
        a square scan. If ``ntrigger`` is not a perfect square, this is
        ``None`` and the caller must pass ``scan_shape=`` to ``load()``
        explicitly.
    - ``n_frames`` : int or None
        Total frame count (``ntrigger``).
    - ``dwell_time_us`` : float or None
        Per-frame dwell in microseconds (``frame_time * 1e6``).
    - ``detector_shape`` : tuple[int, int] or None
        Detector pixel count as ``(height, width)``.
    - ``detector_name`` : str or None
        Human-readable detector description, e.g. ``"Dectris ARINA Si"``.
    - ``saturation`` : int or None
        ADU ceiling before the detector saturates.

    **Raw HDF5 scalars** (schema-agnostic): every scalar dataset in the file
    keyed by its full HDF5 path, e.g.
    ``metadata["entry/instrument/detector/frame_time"]``. Arrays of more
    than 100 elements are skipped. This is the escape hatch when you need a
    field the derived layer does not cover.

    .. note::

        Scope-side parameters (``voltage_kV``, ``semiangle``,
        ``scan_sampling``, ``camera_length``, ``rotation``) are NOT in the
        h5 master - they must be passed to ``ssb()`` explicitly or loaded
        from a site config. If a field is in this dict, it came from the
        file.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 master file.

    Returns
    -------
    dict
        Mixed dict of derived named fields and raw h5-path scalars.

    Examples
    --------
    >>> m = get_metadata('gold_master.h5')
    >>> m['scan_shape']
    (512, 512)
    >>> m['dwell_time_us']
    49.8
    >>> m['detector_name']
    'Dectris ARINA Si'
    >>> # Raw path also available for anything not in the derived layer
    >>> m['entry/instrument/detector/count_time']
    9.95e-05
    """
    metadata: dict = {}
    with h5py.File(filepath, "r") as f:
        def _visit(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            if obj.size > 100:
                return  # skip large arrays (flatfield, pixel_mask, etc.)
            if "data_" in name:
                return  # skip data chunk links
            try:
                val = obj[()]
                if isinstance(val, bytes):
                    val = val.decode()
                elif isinstance(val, np.ndarray) and val.ndim == 0:
                    val = val.item()
                metadata[name] = val
            except (TypeError, ValueError, OSError, UnicodeDecodeError):
                return  # Skip non-scalar/non-readable datasets
        f.visititems(_visit)

        def _copy_attrs(attrs):
            for key, val in attrs.items():
                if isinstance(val, bytes):
                    val = val.decode()
                elif isinstance(val, np.ndarray) and val.ndim == 0:
                    val = val.item()
                metadata.setdefault(key, val)

        _copy_attrs(f.attrs)
        data_group = f.get("entry/data")
        if data_group is not None:
            _copy_attrs(data_group.attrs)

        data_ds = f.get("entry/data/data")
        if data_ds is None and data_group is not None:
            for key in sorted(data_group.keys()):
                if key.startswith("data_"):
                    try:
                        data_ds = data_group[key]
                    except (OSError, KeyError):
                        data_ds = None
                    break
        if data_ds is not None:
            if "scan_shape" in data_ds.attrs:
                metadata.setdefault("scan_shape", tuple(int(x) for x in data_ds.attrs["scan_shape"]))
            if "det_shape" in data_ds.attrs:
                metadata.setdefault("detector_shape", tuple(int(x) for x in data_ds.attrs["det_shape"]))
            if data_ds.ndim >= 3:
                metadata.setdefault("n_frames", int(np.prod(data_ds.shape[:-2])))
    _derive_fields(metadata)
    return metadata


def _derive_fields(metadata: dict) -> None:
    """Promote raw h5-path scalars into named fields on the metadata dict.

    Every derived field is set unconditionally - missing sources land as
    ``None`` so the key is always present and code can do ``meta["scan_shape"]``
    without defensive ``.get()`` calls.
    """
    import math

    ntrigger = metadata.get("entry/instrument/detector/detectorSpecific/ntrigger")
    n_frames = int(ntrigger) if ntrigger is not None else metadata.get("n_frames")
    n_frames = int(n_frames) if n_frames is not None else None

    scan_shape = metadata.get("scan_shape")
    if scan_shape is not None:
        scan_shape = tuple(int(x) for x in scan_shape)
    elif n_frames is not None:
        side = math.isqrt(n_frames)
        scan_shape = (side, side) if side * side == n_frames else None

    frame_time = metadata.get("entry/instrument/detector/frame_time")
    dwell_time_us = float(frame_time) * 1e6 if frame_time is not None else None

    y_pix = metadata.get("entry/instrument/detector/detectorSpecific/y_pixels_in_detector")
    x_pix = metadata.get("entry/instrument/detector/detectorSpecific/x_pixels_in_detector")
    detector_shape = metadata.get("detector_shape")
    if detector_shape is not None:
        detector_shape = tuple(int(x) for x in detector_shape)
    elif y_pix is not None and x_pix is not None:
        detector_shape = (int(y_pix), int(x_pix))

    detector_name = metadata.get("entry/instrument/detector/description")
    saturation_raw = metadata.get("entry/instrument/detector/saturation_value")
    saturation = int(saturation_raw) if saturation_raw is not None else None

    metadata["scan_shape"] = scan_shape
    metadata["n_frames"] = n_frames
    metadata["dwell_time_us"] = dwell_time_us
    metadata["detector_shape"] = detector_shape
    metadata["detector_name"] = detector_name
    metadata["saturation"] = saturation


# =============================================================================
# Velox EMD metadata (#178)
# =============================================================================
#
# Velox (.emd) files contain a per-frame JSON blob at
# ``Data/Image/<hash>/Metadata`` holding microscope-side parameters the
# Arina master never captures: StemMagnification, FullScanFieldOfView,
# AccelerationVoltage, ConvergenceSemiAngle. When a collaborator exports
# the same scan through Velox and drops the EMD next to the master, we
# surface those fields in config.json so the screener can auto-derive
# scan_step_A and show magnification in the list view without anyone
# hand-typing them.

def read_emd_metadata(emd_path) -> dict:
    """Extract scope-side fields from a Velox EMD file.

    Reads the first image's ``Data/Image/<hash>/Metadata`` JSON (Velox
    stores metadata as a uint8 byte vector per frame). Returns a dict
    with whichever of the following keys were found; missing keys are
    omitted so callers can merge via ``dict.update`` without clobbering:

    - ``stem_magnification``    : float, e.g. 5_100_000 for 5.1 Mx
    - ``field_of_view_nm``      : float, FullScanFieldOfView.x in nm
    - ``voltage_kV``            : float, AccelerationVoltage / 1000
    - ``semi_angle_mrad``       : float, probe semiangle when exposed

    Returns ``{}`` on any parse failure so callers can always `.update()`
    the result into an existing config dict without guarding. The EMD
    format version varies across microscope builds, so missing-field
    handling is the common path, not the edge case.
    """
    from pathlib import Path as _Path
    import json as _json
    path = _Path(emd_path)
    if not path.is_file():
        return {}
    try:
        with h5py.File(path, "r") as f:
            if "Data/Image" not in f:
                return {}
            image_group = f["Data/Image"]
            first_hash = next(iter(image_group.keys()), None)
            if first_hash is None:
                return {}
            meta_ds = image_group[first_hash].get("Metadata")
            if meta_ds is None:
                return {}
            # Velox stores metadata as a (nbytes, nframes) uint8 JSON buffer.
            # Frame 0 is sufficient; per-frame blobs are near-identical.
            raw = meta_ds[:, 0] if meta_ds.ndim == 2 else meta_ds[()]
            raw_bytes = bytes(np.asarray(raw).tolist()).rstrip(b"\x00")
            doc = _json.loads(raw_bytes)
    except (OSError, ValueError, KeyError, _json.JSONDecodeError):
        return {}

    out: dict = {}
    optics = doc.get("Optics") or {}
    custom = doc.get("CustomProperties") or {}

    # Velox wraps most scalars as {"type": "double", "value": "5100000"};
    # AccelerationVoltage is historically a bare string. Accept both.
    def _as_float(v):
        if isinstance(v, dict):
            v = v.get("value")
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    mag = _as_float(custom.get("StemMagnification"))
    if mag is not None:
        out["stem_magnification"] = mag

    fov = optics.get("FullScanFieldOfView")
    if isinstance(fov, dict):
        fov_x = _as_float(fov.get("x"))
        if fov_x is not None:
            # Velox reports FOV in metres; screener works in nm.
            out["field_of_view_nm"] = fov_x * 1e9

    voltage = _as_float(optics.get("AccelerationVoltage"))
    if voltage is not None:
        out["voltage_kV"] = voltage / 1000.0

    semi = _as_float(optics.get("ConvergenceSemiAngle") or optics.get("SemiConvergenceAngle"))
    if semi is not None:
        # Velox stores the convergence angle in radians.
        out["semi_angle_mrad"] = semi * 1000.0
    return out


def find_emd_sibling(master_path) -> "Path | None":
    """Locate a Velox EMD next to an Arina master file.

    Arina writes ``<stem>_master.h5`` alongside data chunk files; when
    the operator also exports the scan to Velox, the EMD usually lands
    in the same folder. Strategy:

    1. Prefer a file named ``<stem>.emd`` (strict match).
    2. Fall back to any ``*.emd`` in the same directory - Dectris
       operators often batch-rename after the fact.

    Returns ``None`` when no EMD sibling is found.
    """
    from pathlib import Path as _Path
    master = _Path(master_path)
    folder = master.parent
    stem = master.stem
    if stem.endswith("_master"):
        stem = stem[:-7]
    candidates = list(folder.glob(f"{stem}.emd")) + list(folder.glob(f"{stem}*.emd"))
    if candidates:
        return candidates[0]
    others = list(folder.glob("*.emd"))
    return others[0] if len(others) == 1 else None


# =============================================================================
# GPU CLASSES AND FUNCTIONS
# =============================================================================
#
# Kernels are imported from bitshuffle.py (compiled at import time)
#

# NOTE: CUDA kernel source (~500 lines) lives in bitshuffle.py
# See bitshuffle._CUDA_LZ4_SOURCE for the raw CUDA C++ code


class GPUDecompressor:
    """GPU-accelerated decompressor for bitshuffle+LZ4 HDF5 datasets.

    Uses pinned memory and CUDA kernels for maximum throughput.
    CUDA kernels are compiled at module import time.
    """

    def __init__(
        self,
        max_compressed_bytes: int = 1024 * 1024 * 1024,
        max_frames: int = 100000,
        max_frame_bytes: int = 192 * 192 * 4,
        n_blocks_per_frame: int = 18,
    ):
        """Initialize the decompressor with pre-allocated buffers.

        Parameters
        ----------
        max_compressed_bytes : int, optional
            Maximum size of compressed data, by default 1GB.
        max_frames : int, optional
            Maximum number of frames to support, by default 100000.
        max_frame_bytes : int, optional
            Maximum bytes per frame, by default 147456 (192x192 uint32).
        n_blocks_per_frame : int, optional
            LZ4 blocks per frame, by default 18 for 192x192 uint32.
        """
        self.max_compressed_bytes = max_compressed_bytes
        self.max_frames = max_frames
        self.max_frame_bytes = max_frame_bytes
        self.n_blocks_per_frame = n_blocks_per_frame
        self._h5lz4dc = _h5lz4dc_kernel
        self._shuf = _bitshuffle_kernel
        # Pinned memory for fast CPU->GPU transfers
        self._pinned_mem = cp.cuda.alloc_pinned_memory(max_compressed_bytes)
        self._pinned_buffer = np.frombuffer(
            self._pinned_mem, dtype=np.uint8, count=max_compressed_bytes
        )
        # Pre-allocated metadata arrays
        self._chunk_sizes = np.zeros(max_frames, dtype=np.uint32)
        # uint64: absolute byte offsets into the compressed read buffer
        # can exceed 4 GB on dense 4D-STEM scans (Arina gold etc.)
        self._chunk_offsets = np.zeros(max_frames, dtype=np.uint64)
        self._block_counts = np.zeros(max_frames, dtype=np.uint32)
        self._block_starts_flat = np.zeros(max_frames * n_blocks_per_frame, dtype=np.uint32)
        self._block_offsets = np.zeros(max_frames + 1, dtype=np.uint32)
        # Pre-allocate all GPU buffers for fast first load()
        self._concat_gpu = cp.empty(max_compressed_bytes, dtype=cp.uint8)
        total_output_bytes = max_frames * max_frame_bytes
        self._lz4_output = cp.empty(total_output_bytes, dtype=cp.uint8)
        self._shuffled_output = cp.empty(total_output_bytes, dtype=cp.uint8)

    def load(
        self,
        filepath: str,
        dataset_path: str = "entry/data/data",
    ) -> cp.ndarray:
        """Load and decompress a bitshuffle+LZ4 HDF5 dataset to GPU.

        Parameters
        ----------
        filepath : str
            Path to the HDF5 file.
        dataset_path : str, optional
            Path to the dataset within the HDF5 file, by default "entry/data/data".

        Returns
        -------
        cp.ndarray
            CuPy array on GPU with shape (n_frames, height, width).
        """
        with h5py.File(filepath, "r") as f:
            ds = f[dataset_path]
            n_frames = ds.shape[0]
            frame_shape = ds.shape[1:]
            dtype = ds.dtype
            frame_bytes = int(np.prod(frame_shape) * np.dtype(dtype).itemsize)

            # Reallocate output GPU buffers if dataset exceeds pre-allocated size
            total_needed = n_frames * frame_bytes
            if total_needed > len(self._lz4_output):
                self._lz4_output = cp.empty(total_needed, dtype=cp.uint8)
                self._shuffled_output = cp.empty(total_needed, dtype=cp.uint8)
            # Reallocate metadata arrays if frame count exceeds capacity
            if n_frames > self.max_frames:
                self.max_frames = n_frames
                self._chunk_sizes = np.zeros(n_frames, dtype=np.uint32)
                self._chunk_offsets = np.zeros(n_frames, dtype=np.uint64)
                self._block_counts = np.zeros(n_frames, dtype=np.uint32)
                self._block_starts_flat = np.zeros(
                    n_frames * self.n_blocks_per_frame, dtype=np.uint32
                )
                self._block_offsets = np.zeros(n_frames + 1, dtype=np.uint32)
            # Read chunks into pinned memory, reallocating if compressed data exceeds buffer
            offset = 0
            for i in range(n_frames):
                _, raw = ds.id.read_direct_chunk((i, 0, 0))
                chunk_len = len(raw)
                # Grow pinned buffer if needed
                if offset + chunk_len > self.max_compressed_bytes:
                    new_size = max(
                        self.max_compressed_bytes * 2,
                        offset + chunk_len + 256 * 1024 * 1024,
                    )
                    new_pinned_mem = cp.cuda.alloc_pinned_memory(new_size)
                    new_pinned_buffer = np.frombuffer(
                        new_pinned_mem, dtype=np.uint8, count=new_size
                    )
                    new_pinned_buffer[:offset] = self._pinned_buffer[:offset]
                    self._pinned_mem = new_pinned_mem
                    self._pinned_buffer = new_pinned_buffer
                    self.max_compressed_bytes = new_size
                    self._concat_gpu = cp.empty(new_size, dtype=cp.uint8)
                self._chunk_offsets[i] = offset
                self._chunk_sizes[i] = chunk_len
                self._pinned_buffer[offset : offset + chunk_len] = np.frombuffer(
                    raw, dtype=np.uint8
                )
                offset += chunk_len
            total_compressed = offset
        # Parse headers
        _parse_headers(
            self._pinned_buffer,
            self._chunk_sizes,
            self._chunk_offsets,
            self._block_starts_flat,
            self._block_counts,
            n_frames,
            self.n_blocks_per_frame,
        )
        # Compute block offsets
        self._block_offsets[1 : n_frames + 1] = np.cumsum(self._block_counts[:n_frames])
        total_blocks = int(self._block_offsets[n_frames])
        # Transfer to GPU
        self._concat_gpu[:total_compressed].set(self._pinned_buffer[:total_compressed])
        chunk_offsets_gpu = cp.asarray(self._chunk_offsets[:n_frames])
        block_starts_gpu = cp.asarray(self._block_starts_flat[:total_blocks])
        block_counts_gpu = cp.asarray(self._block_counts[:n_frames])
        block_offsets_gpu = cp.asarray(self._block_offsets[: n_frames + 1])
        # LZ4 decompress
        max_blocks = int(self._block_counts[:n_frames].max())
        max_batch = 10000
        for start in range(0, n_frames, max_batch):
            end = min(start + max_batch, n_frames)
            batch_n = end - start
            byte_offset = start * frame_bytes
            self._h5lz4dc(
                ((max_blocks + 1) // 2, 1, batch_n),
                (32, 2, 1),
                (
                    self._concat_gpu,
                    chunk_offsets_gpu[start:],
                    block_starts_gpu,
                    block_counts_gpu[start:],
                    block_offsets_gpu[start:],
                    np.uint32(BLOCK_SIZE),
                    np.uint32(frame_bytes),
                    self._lz4_output[byte_offset:],
                ),
            )
        # Bitshuffle - use different kernel based on element size
        n_full_8kb = frame_bytes // BLOCK_SIZE
        tail_bytes = frame_bytes % BLOCK_SIZE
        elem_size = np.dtype(dtype).itemsize

        if elem_size == 2:
            # uint16: use optimized shared memory kernel
            for start in range(0, n_frames, max_batch):
                end = min(start + max_batch, n_frames)
                batch_n = end - start
                byte_offset = start * frame_bytes
                if n_full_8kb:
                    _bitshuffle_kernel_u16(
                        (n_full_8kb, 1, batch_n),
                        (256, 1, 1),
                        (
                            self._lz4_output[byte_offset:],
                            self._shuffled_output[byte_offset:].view(cp.uint16),
                            np.uint32(frame_bytes),
                        ),
                    )
                if tail_bytes:
                    tail_elems = tail_bytes // elem_size
                    if tail_bytes % elem_size or tail_elems % 8:
                        raise ValueError(
                            "GPU bitshuffle/LZ4 load supports partial final "
                            "blocks only when the partial detector frame "
                            f"contains a multiple of 8 elements; got {frame_shape}."
                        )
                    _bitshuffle_tail_kernel_u16(
                        ((tail_elems + 255) // 256, 1, batch_n),
                        (256, 1, 1),
                        (
                            self._lz4_output[byte_offset:],
                            self._shuffled_output[byte_offset:].view(cp.uint16),
                            np.uint32(frame_bytes),
                        ),
                    )
        else:
            # uint32: use optimized ballot-based kernel
            frame_u32s = frame_bytes // 4
            for start in range(0, n_frames, max_batch):
                end = min(start + max_batch, n_frames)
                batch_n = end - start
                byte_offset = start * frame_bytes
                if n_full_8kb:
                    self._shuf(
                        (n_full_8kb, 2, batch_n),
                        (32, 32, 1),
                        (
                            self._lz4_output[byte_offset:].view(cp.uint32),
                            self._shuffled_output[byte_offset:].view(cp.uint32),
                            np.uint32(frame_u32s),
                        ),
                    )
                if tail_bytes:
                    tail_elems = tail_bytes // elem_size
                    if tail_bytes % elem_size or tail_elems % 8:
                        raise ValueError(
                            "GPU bitshuffle/LZ4 load supports partial final "
                            "blocks only when the partial detector frame "
                            f"contains a multiple of 8 elements; got {frame_shape}."
                        )
                    _bitshuffle_tail_kernel_u32(
                        ((tail_elems + 255) // 256, 1, batch_n),
                        (256, 1, 1),
                        (
                            self._lz4_output[byte_offset:],
                            self._shuffled_output[byte_offset:].view(cp.uint32),
                            np.uint32(frame_bytes),
                        ),
                    )
        cp.cuda.Device().synchronize()
        total_bytes = n_frames * frame_bytes
        # Return an independent copy - the view into _shuffled_output would
        # keep the entire oversized pre-allocated buffer alive, preventing
        # the caller from releasing the raw block via `del data`.
        return self._shuffled_output[:total_bytes].view(dtype).reshape(
            (n_frames,) + frame_shape
        ).copy()


@njit(cache=True, parallel=True)
def _parse_headers(
    pinned_buffer,
    chunk_sizes,
    chunk_offsets,
    block_starts_out,
    block_counts_out,
    n_frames,
    n_blocks_per_frame,
):
    """Parse bitshuffle+LZ4 chunk headers in parallel."""
    for i in prange(n_frames):
        offset = chunk_offsets[i]
        chunk = pinned_buffer[offset : offset + chunk_sizes[i]]

        # Parse header (first 12 bytes)
        uncomp_size = (
            int(chunk[0]) << 56
            | int(chunk[1]) << 48
            | int(chunk[2]) << 40
            | int(chunk[3]) << 32
            | int(chunk[4]) << 24
            | int(chunk[5]) << 16
            | int(chunk[6]) << 8
            | int(chunk[7])
        )
        block_size = (
            int(chunk[8]) << 24
            | int(chunk[9]) << 16
            | int(chunk[10]) << 8
            | int(chunk[11])
        )
        n_blocks = (uncomp_size + block_size - 1) // block_size
        block_counts_out[i] = n_blocks
        pos = 12
        base_idx = i * n_blocks_per_frame
        for b in range(n_blocks):
            block_starts_out[base_idx + b] = pos
            comp_size = (
                int(chunk[pos]) << 24
                | int(chunk[pos + 1]) << 16
                | int(chunk[pos + 2]) << 8
                | int(chunk[pos + 3])
            )
            pos += 4 + comp_size


_parse_headers_bulk = _parse_headers  # Same function, works with uint64 offsets

# Lazy-initialized decompressor (not at import time to save GPU memory)
_default_decompressor = None


_LIBC = None
_POSIX_FADV_SEQUENTIAL = 2
_POSIX_FADV_WILLNEED = 3


def _get_libc():
    """Lazy-load libc for posix_fadvise. None on non-Linux platforms."""
    global _LIBC
    if _LIBC is None:
        import ctypes
        import ctypes.util
        lib_name = ctypes.util.find_library("c")
        if lib_name is None:
            _LIBC = False
        else:
            try:
                _LIBC = ctypes.CDLL(lib_name, use_errno=True)
            except OSError:
                _LIBC = False
    return _LIBC if _LIBC is not False else None


# Persistent pinned (page-locked) host memory pool. The compressed read_buffer
# is allocated from it so (a) the subsequent H2D upload runs at full PCIe
# bandwidth (~25 GiB/s vs ~3-13 GiB/s pageable) and (b) the page-lock cost is
# paid once then amortized — freed blocks are reused across loads, unlike a
# fresh cp.cuda.alloc_pinned_memory per call (which page-locks from scratch).
# Guarded: the pinned-memory pool is a CUDA resource. On a non-CUDA box `cp` is
# None and there is no pinned pool to set up; the cuda decompress path (the only
# user) never runs there.
if cp is not None:
    _PINNED_POOL = cp.cuda.PinnedMemoryPool()
    cp.cuda.set_pinned_memory_allocator(_PINNED_POOL.malloc)
else:
    _PINNED_POOL = None

# Fast pinned host buffers for the compressed read_buffer. cudaHostAlloc
# page-locks 4.5 GiB in ~2.0 s because it zeros every page first; an anonymous
# mmap (page-aligned, lazily faulted) + cudaHostRegister does the SAME lock in
# ~1.2 s with no zeroing — ~0.8 s saved on the first load of a process
# (measured 2026-05-24 on the gold masters). The mlock path is serial in the
# kernel so threading the register does not help. Registered buffers are kept
# for the process and reused from a free list, so a session loading same-size
# masters pays the lock once and later masters reuse the page-lock for free.
_PINNED_BUFS: list[dict] = []
_PINNED_BUFS_LOCK = None


def _pinned_lock():
    global _PINNED_BUFS_LOCK
    if _PINNED_BUFS_LOCK is None:
        import threading
        _PINNED_BUFS_LOCK = threading.Lock()
    return _PINNED_BUFS_LOCK


def _alloc_pinned_fast(nbytes: int) -> np.ndarray:
    """Return a page-locked uint8 host buffer of length >= nbytes, view[:nbytes].

    Reuses a registered buffer from the free list when one fits (size within
    1.5x, so a 1024-scan buffer is not wasted on a 512-scan load); otherwise
    mmaps a page-aligned anonymous region and cudaHostRegisters it once. The
    page-lock is what makes the downstream H2D run at full PCIe Gen4
    (~25 GiB/s vs ~3-13 GiB/s pageable). Without this the first load of a
    process eats ~2 s in cudaHostAlloc; this trims that to ~1.2 s and to ~0 on
    every subsequent same-size master.
    """
    with _pinned_lock():
        for entry in _PINNED_BUFS:
            if entry["free"] and nbytes <= entry["size"] <= int(nbytes * 1.5):
                entry["free"] = False
                return entry["arr"][:nbytes]
    import ctypes
    import mmap
    from cuda.bindings import runtime as cudart
    region = mmap.mmap(-1, nbytes)  # anonymous → page-aligned base
    addr = ctypes.addressof(ctypes.c_char.from_buffer(region))
    err = cudart.cudaHostRegister(addr, nbytes, 0)
    if int(err[0]) != 0:
        raise RuntimeError(f"cudaHostRegister failed: {int(err[0])}")
    arr = np.frombuffer(region, dtype=np.uint8)
    with _pinned_lock():
        _PINNED_BUFS.append(
            {"region": region, "addr": addr, "arr": arr, "size": nbytes, "free": False}
        )
    return arr[:nbytes]


def _release_pinned(view: np.ndarray) -> None:
    """Mark a buffer from :func:`_alloc_pinned_fast` reusable.

    Keeps it registered (re-registering costs the lock again); the free list is
    bounded by the in-flight master count (~2-3 in the pipeline) so locked host
    RSS stays small. ``view`` is the sliced array; its ``.base`` is the full
    registered array we cached.
    """
    base = view.base if view.base is not None else view
    with _pinned_lock():
        for entry in _PINNED_BUFS:
            if entry["arr"] is base:
                entry["free"] = True
                return


def _prepare_master(
    filepath: str,
    chunk_names: list[str],
    apply_mask: bool = True,
) -> dict:
    """CPU-only phase: read compressed bytes from disk + parse headers.

    Returns a dict with everything needed for GPU decompression. This runs
    entirely on CPU threads - no GPU memory or kernels used. Call
    _decompress_prepared() to finish on GPU.

    Each read worker issues posix_fadvise(SEQUENTIAL|WILLNEED) on its
    fd before pulling bytes, so the kernel kicks off readahead before
    the first os.readv lands. On Linux/NVMe Gen4 this trims ~300 ms
    off the cold disk_read phase on a 12 GB / 105-file Arina master
    and ~300 ms off warm (page cache served at higher effective BW
    once SEQUENTIAL doubles the readahead window).

    Read and chunk_iter still share one pool: a 12 GB read is fast
    enough that h5py.File reopen + chunk_iter on the SAME thread (~30
    ms per file) hides behind the per-file disk wait without needing
    a second pool.

    Typical time on 1024 scan Arina master (12 GB / 105 chunk files):
    ~1.1 s warm / ~2.2 s cold on NVMe Gen4 (NVMe-bound).
    """
    import ctypes
    import os
    from concurrent.futures import ThreadPoolExecutor

    master_dir = os.path.dirname(os.path.abspath(filepath))
    with h5py.File(filepath, "r") as f:
        data_group = f["entry/data"]
        data_paths = []
        for chunk_name in chunk_names:
            link = data_group.get(chunk_name, getlink=True)
            if isinstance(link, h5py.ExternalLink):
                data_paths.append(os.path.join(master_dir, link.filename))
            else:
                data_paths.append(data_group[chunk_name].file.filename)
        pixel_mask = None
        if apply_mask:
            mask_path = "entry/instrument/detector/detectorSpecific/pixel_mask"
            if mask_path in f:
                pixel_mask = f[mask_path][:]

    file_sizes = [os.path.getsize(p) for p in data_paths]
    total_compressed_est = sum(file_sizes)
    # Page-locked host buffer (parallel cudaHostRegister, reused from free list)
    # → full-PCIe H2D downstream without the ~2 s serial cudaHostAlloc lock.
    # Buffered readv (not O_DIRECT) on purpose: it populates the page cache so
    # the second load of the same data is served from RAM (warm 4.6 s vs 13 s
    # cold). O_DIRECT was tried 2026-05-24 — it bypasses the cache (warm
    # collapsed to 10 s) and in the per-master pipeline it did not reach the
    # concurrency that made it fast in a synthetic all-files bench; net loss.
    read_buffer = _alloc_pinned_fast(total_compressed_est)
    file_offsets = np.cumsum([0] + file_sizes[:-1]).tolist()

    libc = _get_libc()

    def read_and_index(args):
        data_path, buf_offset, file_size = args
        fd = os.open(data_path, os.O_RDONLY)
        try:
            if libc is not None:
                # SEQUENTIAL doubles the kernel readahead window;
                # WILLNEED kicks off immediate prefetch.
                libc.posix_fadvise(fd, ctypes.c_long(0), ctypes.c_long(0), _POSIX_FADV_SEQUENTIAL)
                libc.posix_fadvise(fd, ctypes.c_long(0), ctypes.c_long(0), _POSIX_FADV_WILLNEED)
            mv = memoryview(read_buffer)[buf_offset:buf_offset + file_size]
            remaining = file_size
            view_off = 0
            while remaining > 0:
                got = os.readv(fd, [mv[view_off:view_off + remaining]])
                if got == 0:
                    break
                view_off += got
                remaining -= got
        finally:
            os.close(fd)
        with h5py.File(data_path, "r") as df:
            ds = df["entry/data/data"]
            n_frames = ds.shape[0]
            frame_shape = ds.shape[1:]
            dtype = ds.dtype
            chunk_infos = []
            ds.id.chunk_iter(lambda info: chunk_infos.append(
                (info.byte_offset, info.size)
            ))
        return {
            "n_frames": n_frames,
            "frame_shape": frame_shape,
            "dtype": dtype,
            "chunk_infos": chunk_infos,
        }

    # 12 threads saturates the NVMe random-file bandwidth on Arina-style
    # masters; past 16 hurts on host queue depth, 8 underfeeds it.
    with ThreadPoolExecutor(max_workers=12) as pool:
        file_infos = list(pool.map(
            read_and_index,
            zip(data_paths, file_offsets, file_sizes),
        ))

    total_frames = sum(fi["n_frames"] for fi in file_infos)
    frame_shape = file_infos[0]["frame_shape"]
    dtype = file_infos[0]["dtype"]
    frame_bytes = int(np.prod(frame_shape) * np.dtype(dtype).itemsize)
    n_blocks_per_frame = (frame_bytes + BLOCK_SIZE - 1) // BLOCK_SIZE

    # uint64 chunk_offsets: the absolute byte offset of each frame's
    # compressed chunk inside `read_buffer` can exceed 4 GB on dense
    # multi-file scans, so uint32 silently wraps and the GPU kernel
    # reads garbage. (chunk_sizes stays uint32 - per-frame compressed
    # payload is only ~23 KB.)
    chunk_offsets_arr = np.empty(total_frames, dtype=np.uint64)
    chunk_sizes_arr = np.empty(total_frames, dtype=np.uint32)
    frame_idx = 0
    for fi, buf_offset in zip(file_infos, file_offsets):
        for byte_offset, size in fi["chunk_infos"]:
            chunk_offsets_arr[frame_idx] = buf_offset + byte_offset
            chunk_sizes_arr[frame_idx] = size
            frame_idx += 1

    block_starts_flat = np.zeros(total_frames * n_blocks_per_frame, dtype=np.uint32)
    block_counts = np.zeros(total_frames, dtype=np.uint32)
    block_offsets_arr = np.zeros(total_frames + 1, dtype=np.uint32)
    _parse_headers_bulk(
        read_buffer, chunk_sizes_arr, chunk_offsets_arr,
        block_starts_flat, block_counts,
        total_frames, n_blocks_per_frame,
    )
    block_offsets_arr[1:total_frames + 1] = np.cumsum(block_counts[:total_frames])
    total_blocks = int(block_offsets_arr[total_frames])
    total_used = int(max(chunk_offsets_arr + chunk_sizes_arr))

    return {
        "read_buffer": read_buffer[:total_used],
        "chunk_offsets": chunk_offsets_arr,
        "block_starts": block_starts_flat[:total_blocks],
        "block_counts": block_counts,
        "block_offsets": block_offsets_arr,
        "total_frames": total_frames,
        "frame_shape": frame_shape,
        "frame_bytes": frame_bytes,
        "dtype": dtype,
        "pixel_mask": pixel_mask,
        "n_chunk_files": len(chunk_names),
    }


def _decompress_prepared(
    prepared: dict,
    verbose: bool = False,
    auto_narrow: bool = True,
    batch_bytes_target: int = 1 << 28,  # 256 MB per scratch buffer
    det_bin: int = 1,
    streaming_bin: bool = False,
    output_dtype: type | np.dtype | None = None,
    streaming_upload: bool | None = None,
) -> cp.ndarray:
    """GPU phase: transfer compressed bytes and decompress on GPU.

    Chunked implementation: processes frames in ~256 MB batches using two
    small per-batch scratch buffers (reused across iterations) and writes
    directly into a pre-allocated final result buffer. Peak transient
    VRAM is ``compressed + 2*batch + final`` instead of the old
    ``compressed + 2*full_uncompressed + final``, which halves or
    better the loader's peak memory on large scans.

    Optional ``auto_narrow`` (default True) casts uint32 detector data
    down to uint16 on the fly when every observed value fits. Arina
    writes uint32 when its auto-config predicts counts might exceed
    65535, but actual counts in 4D-STEM rarely exceed a few thousand,
    so ``auto_narrow`` halves the final buffer for free on the common
    case. If any batch has a real value >= 65536 it raises
    ``ValueError`` so the caller can retry with ``auto_narrow=False``.

    The pixel mask (Arina's dead-pixel map) is applied PER BATCH inside
    the loop rather than at the end, because dead pixels contain
    0xFFFFFFFF sentinels that would otherwise trip the narrow check.

    Parameters
    ----------
    prepared
        Output of :func:`_prepare_master`.
    verbose
        Print timing + final dtype decision.
    auto_narrow
        If True and source is uint32 and all values fit uint16, return
        a uint16 array. Default True - Arina's uint32 is almost always
        over-allocated in practice, so narrowing is a free memory win.
    batch_bytes_target
        Target size of each per-batch scratch buffer. Default 256 MiB,
        empirically the sweet spot on Blackwell (L2 ≈ 96 MB): speed is
        flat from 128 MB to 4 GB (within ±3%), 256 MB gives the best
        mix of low peak VRAM and low kernel-launch overhead. Smaller
        = lower peak, more launches. Larger = fewer launches, higher
        peak.
    """
    import time
    t0 = time.perf_counter()

    read_buffer = prepared["read_buffer"]
    chunk_offsets_arr = prepared["chunk_offsets"]
    block_starts_flat = prepared["block_starts"]
    block_counts = prepared["block_counts"]
    block_offsets_arr = prepared["block_offsets"]
    total_frames = prepared["total_frames"]
    frame_shape = prepared["frame_shape"]
    frame_bytes = prepared["frame_bytes"]
    source_dtype = prepared["dtype"]
    pixel_mask = prepared["pixel_mask"]

    source_itemsize = int(np.dtype(source_dtype).itemsize)

    # Auto-pick streaming vs full upload: streaming only helps when the
    # compressed file is big enough that holding it all on GPU costs more
    # than per-batch refill overhead saves. Measured 2026-05-14:
    #   * 12 GiB compressed (1024² Arina): streaming -0.11s, fits 24 GB ✓
    #   * 3 GiB compressed (512² Arina): streaming +0.02s (marginal slower)
    # Threshold 6 GiB picks streaming only when the file actually needs it.
    _streaming_auto = streaming_upload is None

    # --- Decide final dtype -------------------------------------------------
    if output_dtype is not None:
        # Honor the project's browse vocabulary: "u8"/"uint8" mean 8-BIT unsigned
        # (the screening dtype the public load(dtype="u8") advertises). numpy's
        # np.dtype("u8") is uint64 (8 BYTES) — passing the browse token straight
        # to np.dtype would silently 8× the output and OOM. Map it explicitly.
        if isinstance(output_dtype, str) and output_dtype in ("u8", "uint8"):
            final_dtype = np.dtype(np.uint8)
        else:
            final_dtype = np.dtype(output_dtype)
        narrow_mode = False
    else:
        narrow_mode = bool(
            auto_narrow
            and np.dtype(source_dtype) == np.dtype(np.uint32)
        )
        final_dtype = np.uint16 if narrow_mode else source_dtype

    # --- Pick batch size ----------------------------------------------------
    # Cap at max_batch=10000 to match the kernel's old launch characteristics
    # and stay well under any plausible single-batch scratch allocation.
    max_batch_from_target = max(1, int(batch_bytes_target // frame_bytes))
    max_batch = min(10000, max_batch_from_target, total_frames)

    if _streaming_auto:
        # Stream + async-overlap the H2D with the kernels whenever there is
        # more than one batch (a single-batch file cannot overlap). Async
        # upload of batch N+1 hides behind batch N's LZ4/bitshuffle work, so
        # the per-file H2D (~116 ms at 512²) no longer runs serially before
        # the kernels. Single-batch files fall back to one full upload.
        streaming_upload = (total_frames + max_batch - 1) // max_batch > 1

    # --- Upload compressed + metadata to GPU -------------------------------
    # streaming_upload=True (default): per-batch compressed slice (~120 MB)
    # instead of full file (~12 GB for 1024² Arina). Bit-equivalent output,
    # peak VRAM drops by ~sizeof(read_buffer), wall time same-or-faster
    # because per-batch H2D overlaps with prior batch's kernel work.
    t_xfer0 = time.perf_counter()
    block_starts_gpu = cp.asarray(block_starts_flat)
    block_counts_gpu = cp.asarray(block_counts)
    block_offsets_gpu = cp.asarray(block_offsets_arr)
    if streaming_upload:
        # Precompute every batch's compressed slice + rebased chunk offsets
        # once. block_starts are chunk-relative so they need no rebasing.
        batch_slices = []
        all_rebased = np.empty(total_frames, dtype=np.uint64)
        max_batch_compressed = 0
        for _s in range(0, total_frames, max_batch):
            _e = min(_s + max_batch, total_frames)
            _bs = int(chunk_offsets_arr[_s])
            _be = len(read_buffer) if _e == total_frames else int(chunk_offsets_arr[_e])
            all_rebased[_s:_e] = chunk_offsets_arr[_s:_e] - np.uint64(_bs)
            max_batch_compressed = max(max_batch_compressed, _be - _bs)
            batch_slices.append((_s, _e, _bs, _be))
        all_rebased_gpu = cp.asarray(all_rebased)
        # Double-buffered async H2D: upload batch N+1 into the spare buffer on
        # a copy stream while batch N's kernels run on the main stream. The
        # read_buffer is page-locked so .set(stream=) is a true async DMA;
        # events keep the copy stream from overwriting a buffer whose LZ4
        # kernel has not finished reading it. This overlaps the ~116 ms/file
        # H2D with the ~95 ms/file kernels instead of running them serially.
        comp_bufs = [cp.empty(max_batch_compressed, dtype=cp.uint8) for _ in range(2)]
        copy_stream = cp.cuda.Stream(non_blocking=True)
        copy_done = [cp.cuda.Event() for _ in range(2)]
        kernel_done = [cp.cuda.Event() for _ in range(2)]
        main_stream = cp.cuda.get_current_stream()
        def _upload_batch(batch_idx, slot):
            _s, _e, _bs, _be = batch_slices[batch_idx]
            comp_bufs[slot][: _be - _bs].set(read_buffer[_bs:_be], stream=copy_stream)
            copy_stream.record(copy_done[slot])
        _upload_batch(0, 0)
        compressed_gpu = None
        chunk_offsets_gpu = None
    else:
        compressed_gpu = cp.empty(len(read_buffer), dtype=cp.uint8)
        compressed_gpu.set(read_buffer)
        chunk_offsets_gpu = cp.asarray(chunk_offsets_arr)
    t_xfer = time.perf_counter() - t_xfer0

    # --- Per-batch scratch buffers (reused across iterations) --------------
    batch_scratch_bytes = max_batch * frame_bytes
    lz4_scratch = cp.empty(batch_scratch_bytes, dtype=cp.uint8)
    shuf_scratch = cp.empty(batch_scratch_bytes, dtype=cp.uint8)

    # --- Pre-allocate final result (possibly narrowed) ---------------------
    # Streaming bin: if det_bin > 1, allocate the final BINNED buffer
    # directly. Per-batch loop bins each batch and writes to the final
    # buffer so the full unbinned (det_bin² × bigger) tensor never lives
    # in VRAM. At 512²×192² with det_bin=2 this is 19.3 GB → 4.83 GB
    # final buffer.
    if streaming_bin and det_bin > 1:
        if frame_shape[-2] % det_bin != 0 or frame_shape[-1] % det_bin != 0:
            raise ValueError(
                f"Detector dims {frame_shape[-2:]} not divisible by det_bin={det_bin}"
            )
        binned_shape = (frame_shape[-2] // det_bin, frame_shape[-1] // det_bin)
        result = cp.empty((total_frames,) + binned_shape, dtype=final_dtype)
    else:
        result = cp.empty((total_frames,) + frame_shape, dtype=final_dtype)

    max_blocks_val = int(block_counts.max())
    n_full_8kb = frame_bytes // BLOCK_SIZE
    tail_bytes = frame_bytes % BLOCK_SIZE

    # Precompute pixel-mask indices on GPU once (shared across batches).
    if pixel_mask is not None:
        _bad_row_np, _bad_col_np = np.where(pixel_mask > 0)
        _has_mask = len(_bad_row_np) > 0
        if _has_mask:
            _bad_row = cp.asarray(_bad_row_np)
            _bad_col = cp.asarray(_bad_col_np)
    else:
        _has_mask = False

    t_decomp0 = time.perf_counter()
    n_batches = (total_frames + max_batch - 1) // max_batch
    for batch_idx, start in enumerate(range(0, total_frames, max_batch)):
        end = min(start + max_batch, total_frames)
        batch_n = end - start
        batch_bytes = batch_n * frame_bytes

        # Streaming upload (double-buffered async): this batch's bytes are
        # already in flight on the copy stream; wait for them, then prefetch
        # the next batch into the spare buffer so its H2D overlaps this
        # batch's kernels. block_starts are chunk-relative (no rebasing);
        # chunk_offsets were rebased per-batch into all_rebased_gpu.
        if streaming_upload:
            slot = batch_idx % 2
            main_stream.wait_event(copy_done[slot])
            if batch_idx + 1 < n_batches:
                next_slot = (batch_idx + 1) % 2
                if batch_idx >= 1:
                    copy_stream.wait_event(kernel_done[next_slot])
                _upload_batch(batch_idx + 1, next_slot)
            cur_compressed = comp_bufs[slot]
            batch_chunk_offsets_gpu = all_rebased_gpu[start:]
        else:
            cur_compressed = compressed_gpu
            batch_chunk_offsets_gpu = chunk_offsets_gpu[start:]

        # 1. LZ4 decompress this batch into lz4_scratch (from offset 0).
        _h5lz4dc_kernel(
            ((max_blocks_val + 1) // 2, 1, batch_n),
            (32, 2, 1),
            (
                cur_compressed,
                batch_chunk_offsets_gpu,
                block_starts_gpu,
                block_counts_gpu[start:],
                block_offsets_gpu[start:],
                np.uint32(BLOCK_SIZE),
                np.uint32(frame_bytes),
                lz4_scratch,
            ),
        )
        # LZ4 is the only consumer of the compressed buffer; once it has run
        # the copy stream may refill this slot for batch_idx+2.
        if streaming_upload:
            main_stream.record(kernel_done[slot])

        # 2. Bitshuffle this batch into shuf_scratch. Pass the full
        #    scratch buffers - the kernel uses batch_n to bound work,
        #    and slicing a uint8 buffer then .view()ing into a wider
        #    dtype can leave CuPy confused about strides.
        if source_itemsize == 2:
            if n_full_8kb:
                _bitshuffle_kernel_u16(
                    (n_full_8kb, 1, batch_n),
                    (256, 1, 1),
                    (
                        lz4_scratch,
                        shuf_scratch.view(cp.uint16),
                        np.uint32(frame_bytes),
                    ),
                )
            if tail_bytes:
                tail_elems = tail_bytes // source_itemsize
                if tail_bytes % source_itemsize or tail_elems % 8:
                    raise ValueError(
                        "GPU bitshuffle/LZ4 load supports partial final blocks "
                        "only when the partial detector frame contains a "
                        f"multiple of 8 elements; got frame_shape={frame_shape}."
                    )
                _bitshuffle_tail_kernel_u16(
                    ((tail_elems + 255) // 256, 1, batch_n),
                    (256, 1, 1),
                    (
                        lz4_scratch,
                        shuf_scratch.view(cp.uint16),
                        np.uint32(frame_bytes),
                    ),
                )
        else:
            frame_u32s = frame_bytes // 4
            if n_full_8kb:
                _bitshuffle_kernel(
                    (n_full_8kb, 2, batch_n),
                    (32, 32, 1),
                    (
                        lz4_scratch.view(cp.uint32),
                        shuf_scratch.view(cp.uint32),
                        np.uint32(frame_u32s),
                    ),
                )
            if tail_bytes:
                tail_elems = tail_bytes // source_itemsize
                if tail_bytes % source_itemsize or tail_elems % 8:
                    raise ValueError(
                        "GPU bitshuffle/LZ4 load supports partial final blocks "
                        "only when the partial detector frame contains a "
                        f"multiple of 8 elements; got frame_shape={frame_shape}."
                    )
                _bitshuffle_tail_kernel_u32(
                    ((tail_elems + 255) // 256, 1, batch_n),
                    (256, 1, 1),
                    (
                        lz4_scratch,
                        shuf_scratch.view(cp.uint32),
                        np.uint32(frame_bytes),
                    ),
                )

        # 3. View the batch prefix of shuf_scratch as source dtype +
        #    batch shape. View the full uint8 scratch first THEN slice
        #    (doing it the other way can silently reinterpret strides).
        n_src_per_frame = frame_bytes // source_itemsize
        batch_view = (
            shuf_scratch.view(source_dtype)[: batch_n * n_src_per_frame]
            .reshape((batch_n,) + frame_shape)
        )

        # 4. Apply pixel_mask to this batch first. Arina writes sentinels
        #    (0xFFFFFFFF for uint32, 0xFFFF for uint16) at dead pixels
        #    and the current narrow check needs to see them as 0 so the
        #    sentinels don't trip it. Same end result as zeroing at the
        #    very end; just moves the mask write inside the batch loop.
        if _has_mask:
            batch_view[:, _bad_row, _bad_col] = 0

        # 5. If narrowing: verify the (masked) batch values fit uint16.
        if narrow_mode:
            batch_max = int(batch_view.max().get())
            if batch_max >= 65536:
                # Rollback: clean up and raise so the caller can retry.
                del lz4_scratch, shuf_scratch, result
                del compressed_gpu, chunk_offsets_gpu, block_starts_gpu
                del block_counts_gpu, block_offsets_gpu
                cp.get_default_memory_pool().free_all_blocks()
                raise ValueError(
                    f"auto_narrow=True but batch frames [{start}, {end}) "
                    f"have max value {batch_max} >= 65536; uint16 cannot "
                    f"represent this data. Retry with auto_narrow=False "
                    f"(requires ~2× more final-buffer VRAM)."
                )

        # 6. Bin batch in-place (streaming) and/or copy with dtype cast
        #    into the final buffer.
        if streaming_bin and det_bin > 1:
            new_dr = frame_shape[-2] // det_bin
            new_dc = frame_shape[-1] // det_bin
            # Reshape (B, det_r, det_c) → (B, new_dr, det_bin, new_dc, det_bin)
            # then sum over binning axes. Integer accumulation, bit-exact
            # to bin_4dstem on the same data.
            binned_batch = batch_view.reshape(
                batch_n, new_dr, det_bin, new_dc, det_bin
            ).sum(axis=(2, 4))
            if final_dtype == source_dtype:
                result[start:end] = binned_batch
            elif final_dtype == np.uint8:
                # browse uint8: clip@255 per batch into the uint8 output, so the
                # full uint16 block is never materialized (peak = uint8 out + one
                # batch + scratch). clip keeps it linear -> virtual-image sums correct.
                result[start:end] = cp.minimum(binned_batch, 255).astype(cp.uint8)
            else:
                result[start:end] = binned_batch.astype(final_dtype)
            del binned_batch
        elif final_dtype == source_dtype:
            result[start:end] = batch_view
        elif final_dtype == np.uint8:
            result[start:end] = cp.minimum(batch_view, 255).astype(cp.uint8)
        else:
            result[start:end] = batch_view.astype(final_dtype)

    cp.cuda.Device().synchronize()
    t_decomp = time.perf_counter() - t_decomp0

    # --- Release scratches + compressed; keep only result ------------------
    del lz4_scratch, shuf_scratch
    del compressed_gpu, chunk_offsets_gpu, block_starts_gpu
    del block_counts_gpu, block_offsets_gpu
    cp.get_default_memory_pool().free_all_blocks()
    # Host read_buffer is fully uploaded; return it to the pinned free list so
    # the next master reuses the page-lock instead of paying it again.
    _release_pinned(read_buffer)

    # pixel_mask already applied per-batch above; no final touch needed.

    t_total = time.perf_counter() - t0
    if verbose:
        total_output = total_frames * frame_bytes
        throughput = total_output / t_total / 1e9
        final_label = (
            f"uint32 → uint16 (auto_narrow)"
            if narrow_mode
            else str(np.dtype(source_dtype))
        )
        print(
            f"  Decompressed {total_frames} frames as {final_label} "
            f"in {t_total:.2f}s ({throughput:.1f} GB/s)"
        )
    return result


def _discover_chunk_names(filepath: str) -> list[str]:
    """Get chunk dataset names (data_000001, etc.) from a master file."""
    import re
    with h5py.File(filepath, "r") as f:
        data_group = f.get("entry/data")
        if data_group is None:
            return []
        return sorted([
            name for name in data_group.keys()
            if re.match(r"data_\d{6}", name)
        ])


def is_master_ready(filepath: str) -> bool:
    """Check if a master H5 file and all its data files are present on disk.

    The master file links to ``data_000001.h5``, ``data_000002.h5``, etc.
    This function checks that all linked data files exist - without reading
    any actual data. Use this before calling :func:`load` on files that may
    still be in the process of being written by the detector.

    Parameters
    ----------
    filepath : str
        Path to the master H5 file.

    Returns
    -------
    bool
        True if the master and all linked data files exist.
    """
    import os

    if not os.path.exists(filepath):
        return False
    try:
        chunk_names = _discover_chunk_names(filepath)
        if not chunk_names:
            return False
        master_dir = os.path.dirname(os.path.abspath(filepath))
        with h5py.File(filepath, "r") as f:
            data_group = f.get("entry/data")
            if data_group is None:
                return False
            for chunk_name in chunk_names:
                link = data_group.get(chunk_name, getlink=True)
                if isinstance(link, h5py.ExternalLink):
                    data_path = os.path.join(master_dir, link.filename)
                else:
                    data_path = data_group[chunk_name].file.filename
                if not os.path.exists(data_path) or os.path.getsize(data_path) == 0:
                    return False
        return True
    except (OSError, KeyError):
        return False


def _load_master_pipelined(
    filepath: str,
    chunk_names: list[str],
    *,
    apply_mask: bool = True,
    auto_narrow: bool = True,
    det_bin: int = 1,
    streaming_bin: bool = False,
    output_dtype: type | np.dtype | None = None,
    n_groups: int = 3,
):
    """Single-master load with disk‖GPU overlap across chunk-file groups.

    Splits the master's chunk files into ``n_groups`` contiguous groups. A
    producer thread reads + header-parses group N+1 from disk
    (``_prepare_master``, pure CPU/disk, GIL released during I/O) while the main
    thread runs the GPU decompress on group N. Wall drops from disk + gpu
    (serial) toward read(G0) + max(rest_disk, all_gpu).

    No concat: the per-group outputs are written into one preallocated output
    array at the right frame offset (each contiguous chunk group → contiguous
    frame range), so peak VRAM is output + one group's transient, not 2× output.
    Returns (data, pixel_mask).
    """
    import queue
    import threading

    n = len(chunk_names)
    n_groups = max(1, min(n_groups, n))
    bounds = [round(i * n / n_groups) for i in range(n_groups + 1)]
    groups = [chunk_names[bounds[i]:bounds[i + 1]] for i in range(n_groups)
              if bounds[i + 1] > bounds[i]]

    q: "queue.Queue" = queue.Queue(maxsize=1)  # 1 in-flight group while GPU works
    _SENT = object()

    def producer():
        for g in groups:
            try:
                q.put((_prepare_master(filepath, g, apply_mask), None))
            except (FileNotFoundError, OSError, ValueError) as e:
                q.put((None, e))
                return
        q.put(_SENT)

    threading.Thread(target=producer, daemon=True).start()

    out = None
    pixel_mask = None
    cursor = 0
    while True:
        item = q.get()
        if item is _SENT:
            break
        prepared, err = item
        if err is not None:
            raise err
        d = _decompress_prepared(
            prepared, verbose=False, auto_narrow=auto_narrow,
            det_bin=det_bin, streaming_bin=streaming_bin,
            output_dtype=output_dtype)
        g_frames = d.shape[0]
        if out is None:
            # First group reveals the post-bin detector shape; total frames
            # comes from the master's ntrigger (one header read, not 105 chunk
            # opens). Preallocate the full flat output once, then write each
            # group into its frame slice (no concat).
            det_shape = d.shape[1:]
            total = _master_total_frames(filepath, chunk_names)
            out = cp.empty((total, *det_shape), dtype=d.dtype)
            pixel_mask = prepared.get("pixel_mask")
        out[cursor:cursor + g_frames] = d
        cursor += g_frames
        del d
        cp.get_default_memory_pool().free_all_blocks()
    if out is None:
        raise FileNotFoundError(f"{filepath}: no data decompressed")
    if cursor < out.shape[0]:
        out = out[:cursor]
    return out, pixel_mask


def _master_total_frames(filepath: str, chunk_names: list[str]) -> int:
    """Total frame count for the master — from ntrigger (one header read)."""
    with h5py.File(filepath, "r") as f:
        nt = f.get("entry/instrument/detector/detectorSpecific/ntrigger")
        if nt is not None:
            return int(nt[()])
        # Fallback: sum chunk-dataset shapes (rare — no ntrigger field).
        import os
        master_dir = os.path.dirname(os.path.abspath(filepath))
        dg = f["entry/data"]
        total = 0
        for cn in chunk_names:
            link = dg.get(cn, getlink=True)
            if isinstance(link, h5py.ExternalLink):
                with h5py.File(os.path.join(master_dir, link.filename), "r") as df:
                    total += df["entry/data/data"].shape[0]
            else:
                total += dg[cn].shape[0]
        return total


def _load_master_optimized(
    filepath: str,
    chunk_names: list[str],
    apply_mask: bool = True,
    verbose: bool = False,
    auto_narrow: bool = True,
    det_bin: int = 1,
    streaming_bin: bool = False,
    output_dtype: type | np.dtype | None = None,
):
    """Bulk-read loader for Dectris master files.

    Two-phase pipeline: _prepare_master() reads compressed bytes from disk
    and parses headers on CPU, _decompress_prepared() transfers and
    decompresses on GPU. Combined here for the single-file load() path.

    Typical speedup: 5s → <0.5s for 262K frames.

    Returns
    -------
    data : cp.ndarray
    pixel_mask : np.ndarray | None
        The Arina pixel_mask read by `_prepare_master` (already used to
        zero dead pixels in-place). Surfaced so callers can thread it to
        downstream consumers without re-opening the HDF5.
    """
    import time
    t0 = time.perf_counter()
    # Enough chunk files to overlap disk read with GPU decompress (group
    # pipeline). Below 8, the single bulk read is already fast and the per-group
    # split isn't worth it.
    if len(chunk_names) >= 8:
        data, pixel_mask = _load_master_pipelined(
            filepath, chunk_names, apply_mask=apply_mask,
            auto_narrow=auto_narrow, det_bin=det_bin,
            streaming_bin=streaming_bin, output_dtype=output_dtype)
        if verbose:
            t_total = time.perf_counter() - t0
            size_gb = data.size * data.dtype.itemsize / 1e9
            print(f"  Loaded {data.shape[0]} frames ({size_gb:.1f} GB) "
                  f"in {t_total:.2f}s (disk‖gpu group pipeline)")
        return data, pixel_mask
    prepared = _prepare_master(filepath, chunk_names, apply_mask)
    t_cpu = time.perf_counter() - t0
    result = _decompress_prepared(
        prepared, verbose=False, auto_narrow=auto_narrow,
        det_bin=det_bin, streaming_bin=streaming_bin,
        output_dtype=output_dtype,
    )
    t_total = time.perf_counter() - t0
    if verbose:
        total_output = result.size * result.dtype.itemsize
        t_gpu = t_total - t_cpu
        throughput = total_output / t_total / 1e9
        size_gb = total_output / 1e9
        narrowed = (
            result.dtype != prepared["dtype"]
            and np.dtype(prepared["dtype"]) == np.dtype(np.uint32)
        )
        narrow_note = "  (uint32 → uint16 auto-narrowed)" if narrowed else ""
        print(
            f"  Loaded {prepared['total_frames']} frames ({size_gb:.1f} GB) "
            f"in {t_total:.2f}s ({throughput:.1f} GB/s){narrow_note}"
        )
    return result, prepared.get("pixel_mask")


def _load_sharded(
    filepaths: list[str],
    devices: list[int] | str,
    *,
    dataset_path=None, apply_mask=True, scan_shape=None,
    det_bin=1, verbose=True, auto_narrow=True, output_dtype=None,
) -> LoadResult:
    """Sharded multi-GPU load — files split across GPUs, each kept on its card.

    Round-robins the file list across ``devices``; one thread per device loads
    its subset (each thread pinned to its GPU via ``cp.cuda.Device``) and stacks
    them into one per-device array. No cross-GPU gather, no host bounce — the
    only way a stack exceeding one card's VRAM fits, and faster than gather.

    Returns ``LoadResult`` whose ``.data`` is ``{device: stacked_array}`` (each
    array resident on that device) and ``.metadata["device_map"] = {file_idx:
    device}`` plus ``["shard_order"] = {device: [file_idx, ...]}``.
    """
    import concurrent.futures
    import time

    if devices == "all":
        devices = list(range(cp.cuda.runtime.getDeviceCount()))
    devices = [int(d) for d in devices]
    n_files = len(filepaths)
    assign = {d: [i for i in range(n_files) if devices[i % len(devices)] == d]
              for d in devices}
    if verbose:
        bin_str = f", det_bin={det_bin}" if det_bin > 1 else ""
        print(f"Loading {n_files} files sharded across GPUs {devices}{bin_str}")

    shards: dict[int, cp.ndarray] = {}
    shard_order: dict[int, list[int]] = {}
    meta_box: dict[int, dict] = {}
    skipped: list[int] = []

    def worker(dev: int):
        idxs = assign[dev]
        if not idxs:
            return
        with cp.cuda.Device(dev):
            stacked = None
            order = []
            for idx in idxs:
                try:
                    r = load(filepaths[idx], dataset_path=dataset_path,
                             apply_mask=apply_mask, scan_shape=scan_shape,
                             det_bin=det_bin, verbose=False,
                             auto_narrow=auto_narrow, output_dtype=output_dtype)
                except (FileNotFoundError, OSError, ValueError) as e:
                    if verbose:
                        print(f"  gpu{dev} [{idx+1}/{n_files}] SKIPPED: {e}")
                    skipped.append(idx)
                    continue
                d = r.data
                meta_box.setdefault(dev, r.metadata)
                # Pre-allocate the per-device stack on the first file, then copy
                # each file into its slot and free the temp — peak is stack +
                # one transient file, not stack + all files (which cp.stack does).
                if stacked is None:
                    stacked = cp.empty((len(idxs), *d.shape), dtype=d.dtype)
                    anchor = d.shape
                if d.shape != anchor:
                    if verbose:
                        print(f"  gpu{dev} [{idx+1}/{n_files}] SKIPPED: shape mismatch")
                    del d, r
                    cp.get_default_memory_pool().free_all_blocks()
                    skipped.append(idx)
                    continue
                stacked[len(order)] = d
                order.append(idx)
                del d, r
                cp.get_default_memory_pool().free_all_blocks()
            if order:
                shards[dev] = stacked[:len(order)] if len(order) < len(idxs) else stacked
                shard_order[dev] = order

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        list(pool.map(worker, devices))

    if not shards:
        raise FileNotFoundError(f"All {n_files} files failed to load")

    device_map = {idx: dev for dev, idxs in shard_order.items() for idx in idxs}
    meta = dict(next(iter(meta_box.values())))
    meta["device_map"] = device_map
    meta["shard_order"] = shard_order
    meta["sharded"] = True
    if verbose:
        dt = time.perf_counter() - t0
        total_gib = sum(s.nbytes for s in shards.values()) / (1 << 30)
        per = " ".join(f"gpu{d}:{shards[d].shape[0]}f/{shards[d].nbytes/(1<<30):.0f}GiB"
                       for d in sorted(shards))
        skip = f" (skipped {len(skipped)})" if skipped else ""
        print(f"  Done: {len(device_map)} files{skip} sharded [{per}] "
              f"total {total_gib:.1f} GiB in {dt:.2f}s")
    return LoadResult(shards, meta)


def _load_as_dataset5dstem(
    filepath, *, dataset_path, apply_mask, scan_shape, det_bin, verbose,
    auto_narrow, output_dtype, devices, series_type, series, sampling, units,
):
    """Load (born-sharded across ``devices`` when given) and wrap into one
    ``Dataset5dstem`` - a multi-tilt / time series presented as a single logical
    dataset whose frames may live on different GPUs.

    The per-frame cupy arrays are handed to torch via ``from_dlpack`` (zero-copy,
    each stays on its card), so the series occupies the same VRAM the raw load
    did, just wrapped. ``Dataset5dstem.from_frames`` keeps them as a series of
    frames when they span devices, or stacks them when they share one.
    """
    try:
        import torch
        # Dataset5dstem is vendored in quantem.live for now (self-contained torch
        # container; migrates to quantem core once #228/#231 land there).
        from quantem.widget.datastructures import Dataset5dstem
    except ImportError as exc:
        raise ImportError("series_type= needs torch installed.") from exc

    result = load(
        filepath, dataset_path=dataset_path, apply_mask=apply_mask,
        scan_shape=scan_shape, det_bin=det_bin, verbose=verbose,
        auto_narrow=auto_narrow, output_dtype=output_dtype, devices=devices,
    )
    data, meta = result.data, result.metadata

    if isinstance(data, dict):  # born-sharded: {device: stacked cupy}
        shard_order = meta["shard_order"]
        n_total = sum(len(v) for v in shard_order.values())
        frames: list = [None] * n_total
        for dev, global_indices in shard_order.items():
            stack = data[dev]
            for local, global_idx in enumerate(global_indices):
                frames[global_idx] = torch.from_dlpack(stack[local])
    elif data.ndim == 5:  # multi-file, one device: (n_files, scan, scan, k, k)
        frames = [torch.from_dlpack(data[i]) for i in range(data.shape[0])]
    else:  # single 4D acquisition → a length-1 series
        frames = [torch.from_dlpack(data)]

    return Dataset5dstem.from_frames(
        frames, sampling=sampling, units=units, series_type=series_type, series=series,
    )


def _load_view(
    filepath,
    backend: str,
    *,
    dataset_path=None,
    apply_mask: bool = True,
    scan_shape=None,
    det_bin: int = 1,
    verbose: bool = True,
    auto_narrow: bool = True,
    output_dtype=None,
    row_prefix: bool = False,
):
    """View/screen load path for the non-cuda backends (cpu, mps).

    Decompresses to a numpy array via the chosen backend's ``load_master``,
    then runs the SAME post-processing the cuda path applies — pixel mask,
    auto_narrow (uint32→uint16), output_dtype cast, scan-shape unflatten — so
    the returned LoadResult is shape/metadata-identical to a cuda load, just
    numpy instead of cupy. The MPS no-bin path is the exception: it returns a
    zero-copy ``MPSChunked4DSTEM`` object because a full 512x512x192x192 stack
    cannot be one Metal buffer on 24 GB Apple Silicon.
    """
    import time

    if backend == "cpu":
        from .backends import cpu as _be
    elif backend == "mps":
        from .backends import mps as _be
    else:  # pragma: no cover - guarded upstream
        raise ValueError(f"_load_view does not handle backend={backend!r}")

    def _one(path):
        meta = get_metadata(str(path))
        if (
            backend == "mps"
            and dataset_path is None
            and output_dtype is None
        ):
            data = _be.load_mps_4dstem(
                str(path),
                scan_shape=scan_shape,
                apply_mask=apply_mask,
                verbose=verbose,
                row_prefix=row_prefix,
                det_bin=det_bin,
            )
            meta.update(data.metadata)
            if apply_mask:
                mask = read_pixel_mask(str(path))
                if mask is not None:
                    meta["pixel_mask"] = mask
            return data, meta

        # Read the dead-pixel mask up front and hand it to the backend so dead
        # pixels are zeroed BEFORE binning (the 65535-sentinel-into-a-bin bug),
        # matching the cuda path. Mask is detector-pixel-resolution, so it only
        # records into meta at det_bin == 1.
        mask = read_pixel_mask(str(path)) if apply_mask else None
        data = _be.load_master(
            str(path), det_bin=det_bin, pixel_mask=mask, verbose=verbose
        )
        if mask is not None and det_bin == 1:
            meta["pixel_mask"] = mask
        if auto_narrow and data.dtype == np.uint32 and int(data.max()) < 65536:
            data = data.astype(np.uint16)
        if output_dtype is not None:
            out_dtype = np.dtype(output_dtype)
            if out_dtype == np.dtype(np.uint8) and data.dtype != np.uint8:
                data = np.minimum(data, 255).astype(np.uint8)
            else:
                data = data.astype(out_dtype)
        data = _apply_scan_shape(data, scan_shape, meta)
        if backend == "mps":
            # Torch MPS tensor is the first-class GPU citizen on Apple, the peer
            # of cupy on cuda. Show4DSTEM consumes it directly and runs BF/DF +
            # virtual images on-GPU from here — no numpy hop (numpy BF/DF is
            # ~2000ms = 0 fps; torch MPS GEMM is ~38ms). The decode already ran
            # on the GPU; this is the one unavoidable H2D of the result
            # (torch uses a private Metal heap, ~0.17s for a 4.8 GB binned load).
            import torch
            data = torch.from_numpy(np.ascontiguousarray(data)).to("mps")
        return data, meta

    paths = list(filepath) if isinstance(filepath, (list, tuple)) else None
    if backend == "mps" and det_bin == 1 and paths is not None:
        raise ValueError(
            "MPS no-bin load returns zero-copy chunks and currently supports "
            "one master at a time. Pass one path, or use det_bin>1 for the "
            "stacked view path."
        )
    if row_prefix:
        if backend != "mps":
            raise ValueError("row_prefix=True is only supported with backend='mps'.")
        if det_bin != 1:
            raise ValueError("row_prefix=True is an exact no-bin MPS layout; use det_bin=1.")
        if dataset_path is not None or output_dtype is not None:
            raise ValueError(
                "row_prefix=True is only supported for full master-file MPS loads."
            )
    t0 = time.perf_counter()
    if paths is None:
        data, meta = _one(filepath)
        if verbose and not (
            backend == "mps" and getattr(data, "chunks", None) is not None
        ):
            nbytes = data.nbytes if hasattr(data, "nbytes") else (
                data.element_size() * data.nelement())  # torch tensor
            print(f"  Loaded {tuple(data.shape)} ({nbytes / 1e9:.1f} GB) in "
                  f"{time.perf_counter() - t0:.2f}s ({backend} backend)")
        return LoadResult(data, meta)
    # Multi-file: stack with a leading file axis (matches cuda multi-file).
    first, meta = _one(paths[0])
    out = np.empty((len(paths), *first.shape), dtype=first.dtype)
    out[0] = first
    for i, path in enumerate(paths[1:], start=1):
        arr, _ = _one(path)
        out[i] = arr
    meta["n_files"] = len(paths)
    if verbose:
        gb = out.nbytes / 1e9
        print(f"  Loaded {len(paths)} files {out.shape} ({gb:.1f} GB) in "
              f"{time.perf_counter() - t0:.2f}s ({backend} backend)")
    return LoadResult(out, meta)


def _browse_dtype_advise_and_cast(data, dtype, verbose):
    """Recommend / apply the smallest lossless integer dtype for BROWSING.

    Browsing is a visual call, and Arina counts are usually low, so uint8 (half
    the memory) is often lossless. This inspects the real count range and:
      - always prints a recommendation (verbose),
      - ``dtype='u8'``: clip at 255 + cast to uint8 (linear, so virtual-image
        sums stay correct), printing how many pixels clipped,
      - ``dtype='auto'``: pick uint8 only if it is lossless (max <= 255), else
        keep the native dtype,
      - ``dtype='u16'`` / ``None``: keep native (None still prints advice).
    Raw uint16 stays the source for reconstruction; uint8 is screening-only.
    """
    sel = (dtype or "").lower()
    if data.dtype != np.uint8 and data.dtype.kind == "u":
        try:
            # Estimate the count range from a strided ~4M-element SAMPLE, not a
            # full-block reduction: cupy max()/mean() run at ~3 GB/s on this card
            # (sm_120), so a full pass over 19 GB adds ~15 s. The sample is plenty
            # for a recommendation; the dtype='u8' decode-direct path counts the
            # real clips exactly anyway.
            flat = data.reshape(-1)
            step = max(1, int(flat.size) // 4_000_000)
            sample = flat[::step]
            mx = int(sample.max())
            pct255 = float((sample > 255).mean()) * 100.0
        except (RuntimeError, MemoryError, ValueError):
            return data
        want_u8 = sel in ("u8", "uint8") or (sel == "auto" and mx <= 255)
        if want_u8 and data.dtype.itemsize > 1:
            xp = type(data).__module__.split(".")[0]
            clip = data  # clip at 255 keeps it linear; only the >255 tail is lost
            data = (data if mx <= 255 else
                    (clip.clip(0, 255) if xp != "cupy" else clip.clip(0, 255))).astype(np.uint8)
            if verbose:
                if pct255 == 0.0:
                    print(f"  Loaded this in uint8 to save you memory - your brightest pixel is "
                          f"only {mx} counts, so nothing was lost and you're using "
                          f"{data.nbytes/1e9:.1f} GB instead of {data.nbytes*2/1e9:.1f} GB.")
                else:
                    print(f"  Loaded this in uint8 for browsing - I clipped {pct255:.2f}% of pixels "
                          f"at 255 (your saturated bright spot, where counts reach {mx}). That's fine "
                          f"for looking at the data; reconstruction always uses the raw uint16.")
        elif verbose:
            if mx <= 255:
                print(f"  Heads up: your brightest pixel is only {mx} counts (well under 255), so you "
                      f"could browse this in uint8 with zero loss and use half the memory "
                      f"({data.nbytes/2/1e9:.1f} GB instead of {data.nbytes/1e9:.1f} GB). "
                      f"Just pass dtype='u8' when you load.")
            else:
                print(f"  Heads up: this dataset has some bright pixels (counts up to {mx}; {pct255:.2f}% "
                      f"sit above 255). uint16 keeps everything exact. If you want to browse lighter, "
                      f"dtype='u8' halves the memory and clips only that {pct255:.2f}% - fine for screening, "
                      f"and reconstruction still uses the raw data.")
    return data


def load(filepath, *args, dtype: str | None = None, gpus=None, stack: bool = True,
         max_concurrent=None, **kwargs):
    """Load 4D-STEM data — one master, or many.

    * ``load(master)`` → one ``LoadResult``.
    * ``load([masters])`` → the masters **stacked** into one 5D dataset (the
      series/viewer case).
    * ``load([masters], gpus=[0, 1])`` (or ``stack=False``) → a **list** of separate
      ``LoadResult``, **read in parallel across disks** and **placed across GPUs** —
      the joint-reconstruction path (``gpus``: ``None`` current device / ``int``
      all-that-GPU / ``list`` per-master round-robin). Decode is serial (concurrent
      in-process CUDA decode corrupts the device); reads overlap across disks so
      bandwidth adds.

    Also recommends / applies the smallest lossless browse dtype: ``dtype=None``
    prints the recommendation; ``dtype='u8'`` clips@255 + casts; ``dtype='auto'``
    picks uint8 only if lossless.
    """
    is_seq = isinstance(filepath, (list, tuple))
    if is_seq and (gpus is not None or not stack):
        # N separate GPU-placed datasets (parallel read, serial decode).
        return _load_many_parallel(list(filepath), gpus=gpus, max_concurrent=max_concurrent,
                                   verbose=kwargs.pop("verbose", False), **kwargs)
    verbose = kwargs.get("verbose", True)
    sel = (dtype or "").lower()
    if sel in ("u8", "uint8") and kwargs.get("output_dtype") is None and not isinstance(filepath, (list, tuple)):
        # decode-DIRECT to uint8: the batched decoder clips@255 into a uint8
        # output, so the full uint16 block is never materialized (peak ~ uint8
        # out + one batch + scratch, not uint16+uint8). The laptop browse path.
        kwargs["output_dtype"] = np.uint8
        result = _load_impl(filepath, *args, **kwargs)
        d = getattr(result, "data", None)
        if verbose and d is not None and hasattr(d, "nbytes"):
            print(f"  Loaded in uint8 for browsing - using {d.nbytes/1e9:.1f} GB, half of uint16 "
                  f"(decoded straight to uint8, so peak memory stayed low). Reconstruction uses raw uint16.")
        return result
    result = _load_impl(filepath, *args, **kwargs)
    data = getattr(result, "data", None)
    if (data is not None and hasattr(data, "max") and hasattr(data, "dtype")
            and getattr(data, "ndim", 0) >= 3):
        new = _browse_dtype_advise_and_cast(data, dtype, verbose)
        if new is not data:
            result = LoadResult(new, result.metadata)
    return result


def disk_of(path) -> str:
    """The physical disk a path lives on, e.g. ``"nvme1n1"``.

    Two paths with the same ``disk_of`` share one drive (loading them in parallel
    gives no gain — the disk is the floor); different ``disk_of`` = independent
    disks whose read bandwidth ADDS under :func:`load_parallel`. Resolves the file's
    ``st_dev`` to its block device via ``/sys/dev/block`` and strips the partition.
    Returns ``"?"`` if the path is unreadable.
    """
    import os as _os
    try:
        st = _os.stat(str(path)).st_dev
        real = _os.path.realpath(f"/sys/dev/block/{_os.major(st)}:{_os.minor(st)}")
        parent = _os.path.basename(_os.path.dirname(real))  # partition -> whole disk
        return parent if parent and parent != "block" else _os.path.basename(real)
    except OSError:
        return "?"


def group_by_disk(paths) -> dict:
    """``{disk: [paths on it]}`` — the cross-disk layout at a glance.

    More distinct disks = more aggregate read bandwidth available to
    :func:`load_parallel`. Spreading hot datasets across the keys unlocks parallel IO.
    """
    out: dict = {}
    for p in paths:
        out.setdefault(disk_of(p), []).append(str(p))
    return out


def _load_many_parallel(masters, *, gpus=None, max_concurrent=None, verbose=False, **load_kwargs):
    """Load many masters with concurrent READS + SERIAL GPU decode, placing each
    master on a chosen GPU. The data-feeding path for joint reconstruction.

    Reached via ``load([masters], gpus=...)`` (or ``stack=False``); ``load_parallel``
    is a thin back-compat alias. See :func:`load`.

    A producer pool reads + header-parses masters concurrently (host/IO only, the
    GIL is released during ``os.readv``); a single consumer decodes them one at a
    time on the GPU. This gives two speedups, both safe:

    * **cross-disk** — readers on different physical disks overlap, bandwidth ADDS.
    * **same-disk** — master N+1 is read *while* master N decodes (read-ahead), so
      the decode hides under the next read.

    GPU decode is SERIAL on purpose: concurrent in-process CUDA decode shares the
    CuPy kernel/plan caches + pinned pool and raises ``cudaErrorIllegalAddress``.
    Serial decode is safe on ANY GPU (it is just a normal single decode) and loses
    no throughput, because one disk feeds slower than one GPU decodes.

    Parameters
    ----------
    masters : list[str]
        Master ``.h5`` paths.
    gpus : None | int | list[int]
        Which GPU each master decodes onto. ``None`` = the current device; an ``int``
        = all masters to that GPU; a ``list`` = per-master (round-robin if shorter),
        e.g. ``gpus=[0, 1]`` places alternating masters on GPU 0 and GPU 1 — the
        placement a multi-GPU joint solver wants. (Placement is serial, never
        concurrent, so it is safe even on a GPU that is also serving the dashboard.)
    max_concurrent : int, optional
        Parallel READERS. Default = max(2, #distinct disks) so there is always a
        read in flight to overlap the decode.

    Returns
    -------
    list[LoadResult]  — one per input master, in input order; each ``.data`` lives
    on its assigned GPU.
    """
    import queue
    import threading
    from contextlib import nullcontext
    from concurrent.futures import ThreadPoolExecutor
    import cupy as cp

    masters = list(masters)
    n = len(masters)
    if gpus is None:
        dev = [None] * n
    elif isinstance(gpus, int):
        dev = [int(gpus)] * n
    else:
        gl = [int(g) for g in gpus]
        dev = [gl[i % len(gl)] for i in range(n)]

    disks = [disk_of(m) for m in masters]
    n_disks = len({d for d in disks if d != "?"})
    n_read = int(max_concurrent) if max_concurrent else max(1, n_disks)
    n_read = max(n_read, 2) if n > 1 else 1  # >=2 so a read is always queued ahead

    # Disk-interleaved read order: hit different disks first for peak parallel BW.
    order, buckets = [], {}
    for i, dk in enumerate(disks):
        buckets.setdefault(dk, []).append(i)
    ql = [list(q) for q in buckets.values()]
    while any(ql):
        for q in ql:
            if q:
                order.append(q.pop(0))

    # Producer: concurrent read+prepare (host) into a bounded queue. The pinned-
    # buffer pool is lock-guarded, so concurrent reads are thread-safe; the queue
    # bound caps in-flight host buffers.
    q: "queue.Queue" = queue.Queue(maxsize=n_read + 1)
    _SENT = object()

    def _producer():
        def _prep(i):
            q.put((i, _prepare_master(masters[i], _discover_chunk_names(masters[i]), True)))
        try:
            with ThreadPoolExecutor(max_workers=n_read) as pool:
                list(pool.map(_prep, order))
        finally:
            q.put(_SENT)

    threading.Thread(target=_producer, daemon=True).start()

    # Consumer: serial decode, each master onto its assigned GPU.
    decode_kw = {k: load_kwargs[k] for k in ("output_dtype", "det_bin", "auto_narrow")
                 if k in load_kwargs}
    if decode_kw.get("det_bin", 1) > 1:
        decode_kw["streaming_bin"] = True
    results: list = [None] * n
    while True:
        item = q.get()
        if item is _SENT:
            break
        i, prepared = item
        d = dev[i]
        with (cp.cuda.Device(d) if d is not None else nullcontext()):
            data = _decompress_prepared(prepared, **decode_kw)
            nf = int(data.shape[0])
            side = int(nf ** 0.5)
            if data.ndim == 3 and side * side == nf:  # (frames,k,k) -> (scan,scan,k,k)
                data = data.reshape(side, side, *data.shape[1:])
        results[i] = LoadResult(data, get_metadata(masters[i]))
    return results


def load_parallel(masters, *, gpus=None, max_concurrent=None, verbose=False, **load_kwargs):
    """Back-compat alias for ``load(masters, gpus=..., stack=False)``.

    Prefer the single entry point: ``load([masters], gpus=[0, 1])`` returns the same
    list of GPU-placed datasets. Kept so existing call sites keep working.
    """
    return load(list(masters), gpus=gpus, stack=False, max_concurrent=max_concurrent,
                verbose=verbose, **load_kwargs)


def _load_impl(
    filepath: str | list[str],
    dataset_path: str | None = None,
    apply_mask: bool = True,
    scan_shape: tuple[int, int] | None = None,
    det_bin: int = 1,
    verbose: bool = True,
    auto_narrow: bool = True,
    output_dtype: type | np.dtype | None = None,
    device: int | str | None = None,
    devices: list[int] | str | None = None,
    series_type: str | None = None,
    series=None,
    sampling=None,
    units=None,
    backend: str = "auto",
    row_prefix: bool = False,
) -> "LoadResult":
    """Load bitshuffle+LZ4 compressed HDF5 data directly to GPU.

    Automatically detects file format:
    - Master files (*_master.h5): Auto-discovers data chunks (data_000001, etc.)
    - Single data files: Uses entry/data/data or specified dataset_path

    When a list of file paths is provided, loads each file sequentially and
    stacks the results into a single array with an extra leading dimension.
    The scan dimension is unflattened automatically from metadata so the
    result is ready for ``Show4DSTEM`` (e.g. 5 files → ``(5, 256, 256, 96, 96)``).

    Parameters
    ----------
    filepath : str or list[str]
        Path to the HDF5 file, or a list of paths to load and stack.
    dataset_path : str, optional
        Path to dataset within HDF5 file. If None, auto-detects.
    apply_mask : bool, optional
        Apply pixel mask to zero out bad pixels (master files only), by default True.
    scan_shape : tuple[int, int], optional
        Scan grid shape ``(scan_row, scan_col)``. By default this is
        **auto-derived** from the h5 ``ntrigger`` field assuming a square
        scan, so users rarely need to pass it. Pass explicitly for
        non-square scans, or to override the derived value.
        When provided (or derived), the scan dimension is unflattened:
        ``(N, det_r, det_c)`` → ``(scan_r, scan_c, det_r, det_c)``; for
        multi-file loads: ``(n_files, scan_r, scan_c, det_r, det_c)``.
    det_bin : int, optional
        Detector binning factor (default 1 = no binning). Applied immediately
        after loading each file, before copying into the output array. Reduces
        VRAM by ``det_bin**2`` (e.g. ``det_bin=2`` quarters detector pixels).
    verbose : bool, optional
        Print progress information (default True).
    auto_narrow : bool, optional
        For master files with uint32 data, cast the final array down to
        uint16 when every observed value fits (< 65536). Arina's uint32
        output is almost always over-allocated in 4D-STEM (actual counts
        rarely exceed a few thousand), so this halves the returned
        array's memory for free. Raises ``ValueError`` if the data
        genuinely contains a value >= 65536 - caller should retry with
        ``auto_narrow=False`` in that case. Default True.
    output_dtype : dtype, optional
        Cast the returned GPU array during load. This is useful for corrected
        4D-STEM archives saved as ``float32``: callers can request
        ``output_dtype=np.float16`` and/or ``det_bin=2`` to work with a much
        smaller GPU array while keeping the on-disk archive high precision.
    device : int or str, optional
        Pin every allocation of a single-target load to this GPU
        (``device=1`` or ``"cuda:1"``). Default None = current device.
    devices : list[int], optional
        **Sharded multi-GPU load** (lists of files only). Split the files
        round-robin across these GPUs; each card decompresses + keeps its own
        subset, with NO gather to one card. This is how a stack larger than a
        single card fits - e.g. ``load(six_512_masters, devices=[0, 1])`` holds
        108 GiB (6× 512²×192² no-bin) across two 96 GB cards. The result's
        ``.data`` is a ``{device: array}`` dict (not one array), and
        ``metadata["device_map"]`` records which file landed on which GPU.
        Sharding is for CAPACITY, not speed: cold load is disk-bound and both
        cards share the one NVMe, so 16× sharded ≈ 16× single-GPU in wall time.

    Returns
    -------
    LoadResult
        Named tuple with ``data`` (cupy.ndarray) and ``metadata`` (dict).
        See :class:`LoadResult` for the full metadata field list, including
        the derived fields (``scan_shape``, ``n_frames``, ``dwell_time_us``,
        ``detector_shape``, ``detector_name``, ``saturation``). Can be
        unpacked: ``data, meta = load(path)``.

    Examples
    --------
    >>> from quantem.live import load
    >>> # scan_shape auto-derived from h5 metadata - no need to type it
    >>> data, meta = load('gold_master.h5')
    >>> data.shape
    (512, 512, 192, 192)
    >>> meta['dwell_time_us']
    99.6

    >>> # Multiple files, scan shape still auto-derived
    >>> data, meta = load(masters[:5])
    >>> data.shape
    (5, 256, 256, 192, 192)

    >>> # Override for non-square scans
    >>> data, meta = load('rectangular_master.h5', scan_shape=(128, 256))

    >>> # Load a float32 corrected archive as a smaller working array
    >>> data, meta = load('corrected_master.h5', det_bin=2, output_dtype=np.float16)

    Performance
    -----------
    Two regimes, because the page cache changes everything:

    - **Cold** (first load of a dataset, bytes not in RAM): disk-bound. The
      compressed bytes must come off the NVMe, and that read dominates - the
      GPU decompress runs hidden in its shadow. Wall time scales with the
      *compressed* size, NOT ``det_bin`` (binning happens after decompress, so
      the same bytes are read either way; ``det_bin`` only shrinks the output
      array / VRAM).
    - **Warm** (data already in RAM cache from a prior load): the read is
      served from RAM at ~5x the NVMe rate, exposing the GPU phase. ~2-3x
      faster than cold. Requires the working set to fit free RAM, else the
      cache churns and warm degrades toward cold.

    Measured 2026-05-24 on real Arina data (512²/1024² scan, 192² detector,
    one RTX PRO 6000, data on a WD_BLACK SN850X). COLD = cache evicted,
    WARM = min of 3 with cache hot:

    ====================================  ========  ========  ==========
    case                                  COLD (s)  WARM (s)  peak VRAM
    ====================================  ========  ========  ==========
    single 512²  det_bin=1 (18 GiB out)     0.75      0.35     24.8 GiB
    single 512²  det_bin=2                   0.75      0.36      6.9 GiB
    single 1024² det_bin=2                   2.54      1.15     24.9 GiB
    single 1024² det_bin=4                   2.52      1.12      6.7 GiB
    6x  512²     det_bin=2 -> 1 GPU          3.87      1.69     32.3 GiB
    6x  512²     det_bin=4 -> 1 GPU          3.73      1.63      8.5 GiB
    10x 512²     det_bin=4 -> 1 GPU          6.25      2.63     13.0 GiB
    16x 512²     det_bin=4 -> 1 GPU         10.05      4.16     19.8 GiB
    ====================================  ========  ========  ==========

    Notes:
    - ``det_bin`` barely changes load time (compare 6x det_bin=2 vs 4) - it
      trades VRAM, not wall time. Use it to fit memory, not to go faster.
    - The compressed read is page-locked (parallel ``cudaHostRegister``) and
      its H2D overlaps the LZ4+bitshuffle kernels on a copy stream, so the GPU
      phase is largely hidden. The remaining cold cost is pure NVMe read.
    - Multi-GPU ``devices=[0, 1]`` shards files across cards for *capacity*
      (a stack larger than one card), not speed - cold load is disk-bound and
      both cards share the one NVMe.

    Sharded multi-GPU (``devices=[0, 1]``, 2x 96 GB cards, same setup):

    ====================================  ========  ========  ==========
    case                                  COLD (s)  WARM (s)  total VRAM
    ====================================  ========  ========  ==========
    6x  512² no-bin     (108 GiB)            4.07      2.02     108 GiB
    8x  512² no-bin     (144 GiB)            OOM       -        ceiling
    6x  512² det_bin=2  (27 GiB)             4.21      2.04      27 GiB
    16x 512² det_bin=2  (72 GiB)            10.03      5.10      72 GiB
    19x 512² det_bin=2  (86 GiB)            12.50      6.14      86 GiB
    16x 512² det_bin=4  (18 GiB)            10.43      5.24      18 GiB
    ====================================  ========  ========  ==========

    Capacity rule (per-tilt 512²x192² uint16): no-bin = 18 GiB, det_bin=2 =
    4.5 GiB, det_bin=4 = 1.1 GiB. The usable VRAM is NOT the full 190 GiB -
    each file's decompress transient (compressed + scratch + the growing
    output stack) stacks ~20 GiB on top per card, so:

    - **no-bin caps at ~6 files** on 2 cards (8x = 144 GiB OOMs on the
      transient, not the final size).
    - **det_bin=2 fits ~30 files**, **det_bin=4 fits ~150** (tiny per-file
      transient at bin=4). E.g. a 70-tilt det_bin=4 series = 70 x 1.1 ~= 79 GiB,
      fits two cards (or even one) with room to spare.
    """
    import os
    import re
    import time

    # Resolve the decompress backend ("auto" → cuda on an NVIDIA box, else mps
    # on Apple Silicon, else cpu). The cuda path below is unchanged. cpu/mps are
    # view/screen-only and land in later steps; until then they raise clearly so
    # a non-cuda box gets an honest error instead of a cupy ImportError crash.
    from .backends import resolve_backend
    backend = resolve_backend(backend)
    if row_prefix and backend != "mps":
        raise ValueError("row_prefix=True is only supported with backend='mps'.")
    if series_type is not None and series_type != "generic" and series is None:
        raise ValueError(
            f"series= is required for series_type={series_type!r} (Arina h5 does not store the "
            f"{series_type} axis); pass the per-frame coordinate, e.g. series=[0, 5, 12, 30]."
        )
    if backend != "cuda":
        # These features are intrinsically CUDA (multi-GPU sharding, GPU
        # pinning, the torch-from-dlpack 5D dataset wrap). Name the fix.
        if series_type is not None:
            raise ValueError(
                f"series_type= requires backend='cuda'; got {backend!r}."
            )
        if device is not None or devices is not None:
            raise ValueError(
                f"device=/devices= multi-GPU requires backend='cuda'; got {backend!r}."
            )
        # MPS multi-dataset: a 4-5 dataset 5D Metal stack is 12s+ to decode and
        # may not fit 24 GB unified memory, so eager stacking is the wrong model
        # on Apple Silicon. Return a lazy handle - dataset 0 decoded now, 1..N
        # filled in the background once Show4DSTEM(handle) builds the viewer.
        # (CUDA stacks eagerly below; big VRAM gives instant dataset switch.)
        if backend == "mps" and (
            isinstance(filepath, (list, tuple))
            or (isinstance(filepath, (str, os.PathLike))
                and os.path.isdir(os.path.expanduser(str(filepath))))
        ):
            from quantem.widget.multidataset_mps import load_macbook_datasets
            return load_macbook_datasets(
                filepath, det_bin=det_bin, scan_size=None, verbose=verbose,
            )
        return _load_view(
            filepath, backend, dataset_path=dataset_path, apply_mask=apply_mask,
            scan_shape=scan_shape, det_bin=det_bin, verbose=verbose,
            auto_narrow=auto_narrow, output_dtype=output_dtype,
            row_prefix=row_prefix,
        )

    # series_type set → return a Dataset5dstem (a multi-tilt / time series),
    # not a raw LoadResult. Load normally (born-sharded across `devices` when
    # given), then wrap the per-frame tensors into one logical 5D dataset. Bare
    # load (series_type=None) returns LoadResult exactly as before, so no
    # existing caller changes.
    if series_type is not None:
        # Arina h5 does NOT store the tilt/time axis, so the per-frame coordinate
        # has to come from the caller. Without it a non-'generic' series would have
        # no axis to plot against - fail early with a copy-paste fix.
        return _load_as_dataset5dstem(
            filepath, dataset_path=dataset_path, apply_mask=apply_mask,
            scan_shape=scan_shape, det_bin=det_bin, verbose=verbose,
            auto_narrow=auto_narrow, output_dtype=output_dtype, devices=devices,
            series_type=series_type, series=series, sampling=sampling, units=units,
        )

    # Pin all cupy allocations to `device` for this call. Recurse without
    # device so the wrap is paid once even on multi-file loads.
    if device is not None:
        device_idx = int(device.split(":")[1]) if isinstance(device, str) else int(device)
        with cp.cuda.Device(device_idx):
            return _load_impl(
                filepath, dataset_path=dataset_path, apply_mask=apply_mask,
                scan_shape=scan_shape, det_bin=det_bin, verbose=verbose,
                auto_narrow=auto_narrow, output_dtype=output_dtype,
            )

    # Sharded multi-GPU: split files across `devices`, each device loads + keeps
    # its own subset (NO gather to a single card). The only way a stack larger
    # than one GPU fits (e.g. 6× 512² no-bin = 108 GiB across 2× 96 GB), and
    # avoids the host-bounce penalty that made gather-mode slower than serial.
    # Returns LoadResult.data = {device: stacked_array_on_that_device}.
    if devices is not None and isinstance(filepath, (list, tuple)):
        return _load_sharded(
            list(filepath), devices, dataset_path=dataset_path,
            apply_mask=apply_mask, scan_shape=scan_shape, det_bin=det_bin,
            verbose=verbose, auto_narrow=auto_narrow, output_dtype=output_dtype,
        )

    # Multi-file: load first to get shape, pre-allocate, copy in-place.
    # Mixed scan shapes: explicit `scan_shape` wins; else first successful
    # file anchors the stack and any later file with a different shape is
    # skipped (same skipped-list pattern as missing data files).
    if isinstance(filepath, (list, tuple)):
        if len(filepath) == 0:
            raise ValueError("Empty file list")
        import queue
        import threading
        n_files = len(filepath)
        if verbose:
            bin_str = f", det_bin={det_bin}" if det_bin > 1 else ""
            print(f"Loading {n_files} files{bin_str} (disk‖gpu pipeline)")

        # Disk‖GPU pipeline: a producer thread reads + header-parses each
        # master's compressed bytes into host memory (_prepare_master, pure
        # CPU/disk, releases the GIL during I/O) while the main thread runs the
        # GPU decompress (_decompress_prepared) on the previous file. Wall time
        # drops from sum(disk+gpu) to ~max(total_disk, total_gpu). A bounded
        # queue (2 slots) caps host memory to ~2 files' compressed bytes.
        # Files that aren't resolvable chunked masters fall back to a full
        # serial load() in the consumer.
        prep_q: "queue.Queue" = queue.Queue(maxsize=2)
        _SENTINEL = object()

        def producer():
            for i, fp in enumerate(filepath):
                try:
                    chunk_names = _discover_chunk_names(fp)
                    if not chunk_names:
                        prep_q.put((i, fp, None, None))  # fallback to serial
                        continue
                    prepared = _prepare_master(fp, chunk_names, apply_mask)
                    prep_q.put((i, fp, prepared, None))
                except (FileNotFoundError, OSError, ValueError) as e:
                    prep_q.put((i, fp, None, e))
            prep_q.put(_SENTINEL)

        threading.Thread(target=producer, daemon=True).start()

        meta = None
        out = None
        n_loaded = 0
        skipped = []
        effective_shape = scan_shape
        t_multi_start = time.perf_counter()
        while True:
            item = prep_q.get()
            if item is _SENTINEL:
                break
            i, fp, prepared, err = item
            if err is not None:
                if verbose:
                    print(f"  [{i+1}/{n_files}] SKIPPED: {err}")
                skipped.append(i)
                continue
            try:
                if prepared is None:
                    # Fallback: not a chunked master — full serial load.
                    r = load(fp, dataset_path=dataset_path, apply_mask=apply_mask,
                             scan_shape=effective_shape, det_bin=det_bin,
                             verbose=False, auto_narrow=auto_narrow,
                             output_dtype=output_dtype)
                    fmeta, d = r.metadata, r.data
                else:
                    d = _decompress_prepared(
                        prepared, verbose=False, auto_narrow=auto_narrow,
                        det_bin=det_bin, streaming_bin=(det_bin > 1),
                        output_dtype=output_dtype)
                    fmeta = get_metadata(fp)
                    if prepared.get("pixel_mask") is not None:
                        fmeta["pixel_mask"] = prepared["pixel_mask"]
                    d = _apply_scan_shape(d, effective_shape, fmeta)
            except (FileNotFoundError, OSError, ValueError) as e:
                if verbose:
                    print(f"  [{i+1}/{n_files}] SKIPPED: {e}")
                skipped.append(i)
                continue
            if out is None:
                meta = fmeta
                out = cp.empty((n_files, *d.shape), dtype=d.dtype)
                if effective_shape is None:
                    effective_shape = meta.get("scan_shape")
            if d.shape != out.shape[1:]:
                if verbose:
                    print(f"  [{i+1}/{n_files}] SKIPPED: shape {tuple(d.shape)} "
                          f"differs from anchor {tuple(out.shape[1:])}")
                del d
                cp.get_default_memory_pool().free_all_blocks()
                skipped.append(i)
                continue
            out[n_loaded] = d
            del d
            cp.get_default_memory_pool().free_all_blocks()
            n_loaded += 1
        if out is None:
            raise FileNotFoundError(
                f"All {n_files} files failed to load (missing data files)"
            )
        if n_loaded < n_files:
            out = out[:n_loaded]
        # Record the per-dataset names (loaded order, skips dropped) so the viewer
        # can label the dataset slider with each source file instead of an index.
        skipped_set = set(skipped)
        loaded_names = [
            os.path.basename(str(filepath[i]))[:-len("_master.h5")]
            if str(filepath[i]).endswith("_master.h5") else os.path.basename(str(filepath[i]))
            for i in range(n_files) if i not in skipped_set
        ]
        meta["file_names"] = loaded_names
        meta["n_files"] = n_loaded
        if verbose:
            t_multi = time.perf_counter() - t_multi_start
            size_gb = out.nbytes / 1e9 if out is not None else 0
            skip_msg = f" (skipped {len(skipped)})" if skipped else ""
            print(f"  Done: {n_loaded} files{skip_msg} → {tuple(out.shape)} ({size_gb:.1f} GB) in {t_multi:.2f}s")
        return LoadResult(out, meta)

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"HDF5 file not found: {filepath}")

    t0 = time.perf_counter()
    global _default_decompressor

    with h5py.File(filepath, "r") as f:
        data_group = f.get("entry/data")
        if data_group is not None:
            # Check for Dectris-style chunked data (data_000001, etc.)
            chunk_names = sorted([
                name for name in data_group.keys()
                if re.match(r"data_\d{6}", name)
            ])
            # Two layouts ride the chunk-name prefix:
            #   1. classic Dectris master with external _data_NNNNNN.h5 siblings
            #   2. self-contained master where the data lives inline (rare;
            #      seen in some Velox-exported sets) or where the sibling
            #      files are missing entirely (common for half-copied
            #      datasets, e.g. sample_master.h5)
            # If chunk_names exist AND every external link resolves, take the
            # bulk-loader path. Otherwise drop to the inline `entry/data/data`
            # path and raise a clean error if neither layout has any data.
            chunks_resolvable = False
            if chunk_names:
                master_dir = os.path.dirname(os.path.abspath(filepath))
                for cn in chunk_names:
                    link = data_group.get(cn, getlink=True)
                    if isinstance(link, h5py.ExternalLink):
                        chunk_path = os.path.join(master_dir, link.filename)
                        if not os.path.exists(chunk_path):
                            break
                    # Internal/soft links resolve in the same file, so they're
                    # always present by definition.
                else:
                    # Every external link points at an existing file.
                    chunks_resolvable = True
            if chunk_names and chunks_resolvable:
                # Master file with external data links (single or multi-chunk).
                # When det_bin > 1, request streaming bin so the full
                # unbinned tensor never lives in VRAM (4× memory savings
                # at det_bin=2).
                data, pixel_mask = _load_master_optimized(
                    filepath, chunk_names, apply_mask=apply_mask,
                    verbose=verbose, auto_narrow=auto_narrow,
                    det_bin=det_bin, streaming_bin=(det_bin > 1),
                    output_dtype=output_dtype,
                )
                meta = get_metadata(filepath)
                if pixel_mask is not None:
                    meta["pixel_mask"] = pixel_mask
                data = _apply_scan_shape(data, scan_shape, meta)
                return LoadResult(data, meta)
            if "data" in data_group:
                # Self-contained master OR a master whose sibling chunk
                # files weren't found - try inline entry/data/data.
                dataset_path = "entry/data/data"
            elif chunk_names and not chunks_resolvable:
                # Master listed external chunks but none of the sibling files
                # exist on disk and there's no inline fallback. This is the
                # half-copied-dataset case; raise a clean FileNotFoundError so
                # callers (e.g. browse router) return 4xx instead of 500.
                raise FileNotFoundError(
                    f"{os.path.basename(filepath)}: external _data_NNNNNN.h5 "
                    "siblings missing and no inline entry/data/data dataset. "
                    "The master.h5 is incomplete; copy the sibling files."
                )

    # Use default path if not set
    if dataset_path is None:
        dataset_path = "entry/data/data"

    # Get dataset info for decompressor initialization
    with h5py.File(filepath, "r") as f:
        if dataset_path not in f:
            raise ValueError(f"Dataset '{dataset_path}' not found in {filepath}")
        ds = f[dataset_path]
        shape = ds.shape
        dtype = ds.dtype

        # Check if data has any filters (bitshuffle, gzip, etc.)
        dcpl = ds.id.get_create_plist()
        n_filters = dcpl.get_nfilters()

        # Calculate frame shape based on dimensionality
        # 4D/5D data (from save()): frame_shape is last 2 dims
        # 3D data (raw Dectris): frame_shape is last 2 dims
        if len(shape) == 5:
            frame_shape = shape[3:]
        elif len(shape) == 4:
            frame_shape = shape[2:]
        else:  # 3D
            frame_shape = shape[1:]

        frame_bytes = int(np.prod(frame_shape) * np.dtype(dtype).itemsize)
        n_blocks_per_frame = (frame_bytes + BLOCK_SIZE - 1) // BLOCK_SIZE

        # Get pixel mask if available
        pixel_mask = None
        if apply_mask:
            mask_path = "entry/instrument/detector/detectorSpecific/pixel_mask"
            if mask_path in f:
                pixel_mask = f[mask_path][:]

        # For uncompressed data (no filters), just read with h5py and transfer to GPU
        if n_filters == 0:
            raw_data = ds[:]
            data = cp.asarray(raw_data)
            if output_dtype is not None:
                data = data.astype(output_dtype)
            t1 = time.perf_counter()
            if verbose:
                size_gb = data.nbytes / 1e9
                print(f"  {os.path.basename(filepath)}  {tuple(data.shape)}  {size_gb:.1f} GB  {t1-t0:.2f}s")
            meta = get_metadata(filepath)
            if pixel_mask is not None:
                meta["pixel_mask"] = pixel_mask
            data = _apply_scan_shape(data, scan_shape, meta)
            return LoadResult(data, meta)

    # For 4D/5D compressed data, use the dedicated loader
    if len(shape) >= 4:
        data = _load_gpu_decompressed(filepath, dataset_path, shape, dtype, verbose)
        if output_dtype is not None:
            data = data.astype(output_dtype)
        t1 = time.perf_counter()
        if verbose:
            size_gb = data.nbytes / 1e9
            print(f"  {os.path.basename(filepath)}  {tuple(data.shape)}  {size_gb:.1f} GB  {t1-t0:.2f}s")
        meta = get_metadata(filepath)
        if pixel_mask is not None:
            meta["pixel_mask"] = pixel_mask
        data = _apply_scan_shape(data, scan_shape, meta)
        return LoadResult(data, meta)

    # For 3D data, use cached GPUDecompressor
    if (
        _default_decompressor is None
        or frame_bytes > _default_decompressor.max_frame_bytes
        or n_blocks_per_frame > _default_decompressor.n_blocks_per_frame
    ):
        _default_decompressor = GPUDecompressor(
            max_compressed_bytes=1024 * 1024 * 1024,
            max_frames=70000,
            max_frame_bytes=frame_bytes,
            n_blocks_per_frame=n_blocks_per_frame,
        )

    data = _default_decompressor.load(filepath, dataset_path)
    if output_dtype is not None:
        data = data.astype(output_dtype)

    # Free decompressor buffers - they hold ~12 GB of GPU memory
    # and are only needed during decompression.
    _default_decompressor = None
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    # Apply pixel mask if present
    if pixel_mask is not None:
        bad_row, bad_col = np.where(pixel_mask > 0)
        if len(bad_row) > 0:
            data[:, bad_row, bad_col] = 0

    t1 = time.perf_counter()
    if verbose:
        size_gb = data.nbytes / 1e9
        throughput = data.nbytes / (t1 - t0) / 1e9 if (t1 - t0) > 0 else 0
        print(f"  {os.path.basename(filepath)}  {tuple(data.shape)}  {size_gb:.1f} GB  {t1-t0:.2f}s  {throughput:.1f} GB/s")

    # Bin detector if requested
    if det_bin > 1:
        data = bin(data, factor=det_bin)

    # Unflatten scan dimension - uses explicit scan_shape if passed,
    # else auto-derives from metadata (ntrigger for square scans).
    meta = get_metadata(filepath)
    if pixel_mask is not None:
        meta["pixel_mask"] = pixel_mask
    data = _apply_scan_shape(data, scan_shape, meta)

    return LoadResult(data, meta)


def _load_gpu_decompressed(
    filepath: str,
    dataset_path: str,
    shape: tuple,
    dtype: np.dtype,
    verbose: bool = False,
) -> cp.ndarray:
    """Load HDF5 dataset using GPU decompression.

    Works with files saved using our GPU bitshuffle format.
    Uses the same kernel interface as GPUDecompressor.
    """
    import time

    t0 = time.perf_counter()

    # Calculate frame layout
    if len(shape) == 5:
        n_frames = shape[0] * shape[1] * shape[2]
        frame_shape = shape[3:]
    elif len(shape) == 4:
        n_frames = shape[0] * shape[1]
        frame_shape = shape[2:]
    else:  # 3D
        n_frames = shape[0]
        frame_shape = shape[1:]

    frame_bytes = int(np.prod(frame_shape)) * np.dtype(dtype).itemsize
    n_blocks_per_frame = (frame_bytes + BLOCK_SIZE - 1) // BLOCK_SIZE

    if verbose:
        print(f"  Loading {os.path.basename(filepath)}: {n_frames} frames, {shape}")

    # Pre-allocate metadata arrays
    # uint64 chunk_offsets: see _prepare_master for rationale
    chunk_offsets = np.zeros(n_frames, dtype=np.uint64)
    block_counts = np.zeros(n_frames, dtype=np.uint32)
    block_starts_flat = np.zeros(n_frames * n_blocks_per_frame, dtype=np.uint32)
    block_offsets = np.zeros(n_frames + 1, dtype=np.uint32)

    # First pass: read all chunks and calculate total size
    raw_chunks = []
    with h5py.File(filepath, "r") as f:
        ds = f[dataset_path]

        for frame_idx in range(n_frames):
            # Calculate chunk index based on shape
            if len(shape) == 5:
                idx0 = frame_idx // (shape[1] * shape[2])
                rem = frame_idx % (shape[1] * shape[2])
                idx1 = rem // shape[2]
                idx2 = rem % shape[2]
                chunk_idx = (idx0, idx1, idx2, 0, 0)
            elif len(shape) == 4:
                idx0 = frame_idx // shape[1]
                idx1 = frame_idx % shape[1]
                chunk_idx = (idx0, idx1, 0, 0)
            else:
                chunk_idx = (frame_idx, 0, 0)

            # Read raw chunk
            _, raw = ds.id.read_direct_chunk(chunk_idx)
            raw_chunks.append(raw)

    # Calculate total compressed size and build metadata
    total_compressed = sum(len(r) for r in raw_chunks)

    # Use pinned memory for fast CPU→GPU transfer (26 GB/s vs 11 GB/s)
    # Allocation is slow (~0.5s) but transfer is 2.5x faster
    pinned_mem = cp.cuda.alloc_pinned_memory(total_compressed)
    buffer = np.frombuffer(pinned_mem, dtype=np.uint8, count=total_compressed)

    # Copy chunks to buffer and parse headers
    offset = 0
    for frame_idx, raw in enumerate(raw_chunks):
        chunk_len = len(raw)
        chunk_offsets[frame_idx] = offset
        buffer[offset:offset + chunk_len] = np.frombuffer(raw, dtype=np.uint8)

        # Parse bitshuffle header and block sizes
        # Header: 8 bytes uncompressed size + 4 bytes block size = 12 bytes
        pos = 12
        block_counts[frame_idx] = n_blocks_per_frame
        block_base = frame_idx * n_blocks_per_frame

        for b in range(n_blocks_per_frame):
            # Each LZ4 block has 4-byte big-endian size prefix
            block_starts_flat[block_base + b] = pos
            lz4_size = int.from_bytes(raw[pos:pos+4], 'big')
            pos += 4 + lz4_size

        offset += chunk_len

    # Free raw_chunks to save memory
    del raw_chunks

    # Compute cumulative block offsets
    block_offsets[1:n_frames + 1] = np.cumsum(block_counts[:n_frames])
    total_blocks = int(block_offsets[n_frames])

    t_read = time.perf_counter() - t0

    # Transfer to GPU using pinned memory for maximum throughput
    t0 = time.perf_counter()
    compressed_gpu = cp.empty(total_compressed, dtype=cp.uint8)
    compressed_gpu.set(buffer)

    chunk_offsets_gpu = cp.asarray(chunk_offsets)
    block_starts_gpu = cp.asarray(block_starts_flat[:total_blocks])
    block_counts_gpu = cp.asarray(block_counts)
    block_offsets_gpu = cp.asarray(block_offsets)

    # Allocate output buffers
    total_output_bytes = n_frames * frame_bytes
    lz4_output = cp.empty(total_output_bytes, dtype=cp.uint8)

    # LZ4 decompress with batching
    max_blocks = int(block_counts.max())
    max_batch = 10000

    for start in range(0, n_frames, max_batch):
        end = min(start + max_batch, n_frames)
        batch_n = end - start
        byte_offset = start * frame_bytes

        _h5lz4dc_kernel(
            ((max_blocks + 1) // 2, 1, batch_n),
            (32, 2, 1),
            (
                compressed_gpu,
                chunk_offsets_gpu[start:],
                block_starts_gpu,
                block_counts_gpu[start:],
                block_offsets_gpu[start:],
                np.uint32(BLOCK_SIZE),
                np.uint32(frame_bytes),
                lz4_output[byte_offset:],
            ),
        )

    # Inverse bitshuffle - use different kernel based on element size
    n_full_8kb = frame_bytes // BLOCK_SIZE
    tail_bytes = frame_bytes % BLOCK_SIZE
    elem_size = np.dtype(dtype).itemsize
    result_flat = cp.empty(total_output_bytes, dtype=cp.uint8)

    if elem_size == 2:
        # uint16: use optimized shared memory kernel
        for start in range(0, n_frames, max_batch):
            end = min(start + max_batch, n_frames)
            batch_n = end - start
            byte_offset = start * frame_bytes
            if n_full_8kb:
                _bitshuffle_kernel_u16(
                    (n_full_8kb, 1, batch_n),
                    (256, 1, 1),
                    (
                        lz4_output[byte_offset:],
                        result_flat[byte_offset:].view(cp.uint16),
                        np.uint32(frame_bytes),
                    ),
                )
            if tail_bytes:
                tail_elems = tail_bytes // elem_size
                if tail_bytes % elem_size or tail_elems % 8:
                    raise ValueError(
                        "GPU bitshuffle/LZ4 load supports partial final blocks "
                        "only when the partial detector frame contains a "
                        f"multiple of 8 elements; got frame_shape={frame_shape}."
                    )
                _bitshuffle_tail_kernel_u16(
                    ((tail_elems + 255) // 256, 1, batch_n),
                    (256, 1, 1),
                    (
                        lz4_output[byte_offset:],
                        result_flat[byte_offset:].view(cp.uint16),
                        np.uint32(frame_bytes),
                    ),
                )
    else:
        # uint32: use optimized ballot-based kernel
        frame_u32s = frame_bytes // 4
        for start in range(0, n_frames, max_batch):
            end = min(start + max_batch, n_frames)
            batch_n = end - start
            byte_offset = start * frame_bytes
            if n_full_8kb:
                _bitshuffle_kernel(
                    (n_full_8kb, 2, batch_n),
                    (32, 32, 1),
                    (
                        lz4_output[byte_offset:].view(cp.uint32),
                        result_flat[byte_offset:].view(cp.uint32),
                        np.uint32(frame_u32s),
                    ),
                )
            if tail_bytes:
                tail_elems = tail_bytes // elem_size
                if tail_bytes % elem_size or tail_elems % 8:
                    raise ValueError(
                        "GPU bitshuffle/LZ4 load supports partial final blocks "
                        "only when the partial detector frame contains a "
                        f"multiple of 8 elements; got frame_shape={frame_shape}."
                    )
                _bitshuffle_tail_kernel_u32(
                    ((tail_elems + 255) // 256, 1, batch_n),
                    (256, 1, 1),
                    (
                        lz4_output[byte_offset:],
                        result_flat[byte_offset:].view(cp.uint32),
                        np.uint32(frame_bytes),
                    ),
                )

    cp.cuda.Device().synchronize()

    # Reshape to original shape
    result = result_flat.view(dtype).reshape(shape)

    t_decomp = time.perf_counter() - t0

    if verbose:
        print(f"  Read: {t_read:.2f}s, decompress: {t_decomp:.2f}s")

    return result



def bin(
    data: cp.ndarray,
    factor: int = 2,
    axes: Literal["detector", "k", "scan", "r", "all"] = "detector",
    dtype: type | np.dtype | None = None,
    reduction: Literal["sum", "mean"] = "sum",
) -> cp.ndarray:
    """Apply spatial binning to 4D-STEM data on GPU.

    Parameters
    ----------
    data : cp.ndarray
        CuPy array with shape:
        - 4D: (scan_row, scan_col, k_row, k_col) - full 4D-STEM
        - 3D: (n_frames, k_row, k_col) - flattened scan
        - 2D: (k_row, k_col) - single diffraction pattern
    factor : int, optional
        Binning factor (2 for 2x2, 4 for 4x4, etc.), by default 2.
    axes : str, optional
        Which axes to bin:
        - 'detector' or 'k': bin k_row, k_col (last 2 dims) - default
        - 'scan' or 'r': bin scan_row, scan_col (first 2 dims of 4D data)
        - 'all': bin all 4 dimensions (4D data only)
    dtype : type or np.dtype, optional
        Output dtype. If None, uses uint32 for int input (sum), float32 for mean.
    reduction : str, optional
        Reduction method - 'sum' (default) or 'mean'.

    Returns
    -------
    cp.ndarray
        Binned CuPy array with reduced dimensions.

    Examples
    --------
    >>> # Bin detector (k_row, k_col)
    >>> data_4d = data.reshape(256, 256, 192, 192)  # (scan_row, scan_col, k_row, k_col)
    >>> binned = bin(data_4d, factor=2, axes='detector')  # (256, 256, 96, 96)

    >>> # Bin scan (scan_row, scan_col)
    >>> binned = bin(data_4d, factor=2, axes='scan')  # (128, 128, 192, 192)

    >>> # 3D data (flattened scan) - bins detector by default
    >>> binned = bin(data_3d, factor=2)  # (65536, 96, 96)
    """
    if reduction not in ("sum", "mean"):
        raise ValueError(f"reduction must be 'sum' or 'mean', got '{reduction}'")
    if factor == 1:
        return data

    axes = axes.lower()
    if axes in ("detector", "diffraction", "q", "k"):
        axes = "detector"
    elif axes in ("scan", "real", "r"):
        axes = "scan"
    elif axes == "all":
        axes = "all"
    else:
        raise ValueError(f"axes must be 'detector', 'scan', or 'all', got '{axes}'")

    # Determine output dtype
    if dtype is None:
        dtype = cp.float32 if reduction == "mean" else (
            cp.uint32 if cp.issubdtype(data.dtype, cp.integer) else cp.float32
        )

    # Handle different input dimensions
    if data.ndim == 2:
        # Single 2D image (k_row, k_col)
        if axes == "scan":
            raise ValueError("Cannot bin scan axes on 2D data")
        h, w = data.shape
        if h % factor != 0 or w % factor != 0:
            raise ValueError(f"Dimensions ({h}, {w}) not divisible by factor {factor}")
        reshaped = data.reshape(h // factor, factor, w // factor, factor)
        if reduction == "mean":
            return reshaped.mean(axis=(1, 3), dtype=dtype)
        return reshaped.sum(axis=(1, 3), dtype=dtype)

    elif data.ndim == 3:
        # 3D stack (n_frames, k_row, k_col)
        if axes == "scan":
            raise ValueError("Cannot bin scan axes on 3D data. Reshape to 4D first: data.reshape(Ry, Rx, k_row, k_col)")
        n, h, w = data.shape
        if h % factor != 0 or w % factor != 0:
            raise ValueError(f"Dimensions ({h}, {w}) not divisible by factor {factor}")
        reshaped = data.reshape(n, h // factor, factor, w // factor, factor)
        if reduction == "mean":
            return reshaped.mean(axis=(2, 4), dtype=dtype)
        return reshaped.sum(axis=(2, 4), dtype=dtype)

    elif data.ndim == 4:
        # Full 4D-STEM (scan_row, scan_col, k_row, k_col)
        sr, sc, kr, kc = data.shape

        if axes == "detector":
            if kr % factor != 0 or kc % factor != 0:
                raise ValueError(f"Detector dims ({kr}, {kc}) not divisible by factor {factor}")
            reshaped = data.reshape(sr, sc, kr // factor, factor, kc // factor, factor)
            if reduction == "mean":
                return reshaped.mean(axis=(3, 5), dtype=dtype)
            return reshaped.sum(axis=(3, 5), dtype=dtype)

        elif axes == "scan":
            if sr % factor != 0 or sc % factor != 0:
                raise ValueError(f"Scan dims ({sr}, {sc}) not divisible by factor {factor}")
            reshaped = data.reshape(sr // factor, factor, sc // factor, factor, kr, kc)
            if reduction == "mean":
                return reshaped.mean(axis=(1, 3), dtype=dtype)
            return reshaped.sum(axis=(1, 3), dtype=dtype)

        else:  # all
            if sr % factor != 0 or sc % factor != 0:
                raise ValueError(f"Scan dims ({sr}, {sc}) not divisible by factor {factor}")
            if kr % factor != 0 or kc % factor != 0:
                raise ValueError(f"Detector dims ({kr}, {kc}) not divisible by factor {factor}")
            # Bin all 4 dimensions
            reshaped = data.reshape(
                sr // factor, factor, sc // factor, factor,
                kr // factor, factor, kc // factor, factor
            )
            if reduction == "mean":
                return reshaped.mean(axis=(1, 3, 5, 7), dtype=dtype)
            return reshaped.sum(axis=(1, 3, 5, 7), dtype=dtype)

    else:
        raise ValueError(f"Expected 2D, 3D, or 4D array, got {data.ndim}D. "
                         f"For multi-file data, use load(..., det_bin=2) instead.")


def _clear_memory() -> None:
    """Release GPU memory pools and decompressor buffers (internal use)."""
    global _default_decompressor
    _default_decompressor = None
    # free_all_blocks() raises MemoryError / RuntimeError on genuine GPU
    # failures (OOM, CUDA driver error). Let those propagate so callers can
    # detect a dirty GPU state and stop processing rather than cascading into
    # every subsequent file (#130). Only swallow AttributeError in case CuPy
    # was never initialised (no-GPU environment).
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except AttributeError:
        pass
bin2d = bin


def _read_frame_count(filepath: str) -> int | None:
    """Read total frame count from a master HDF5 without decompressing.

    Only reads the first data file's nimages field from the master,
    which is the total frame count for virtual-dataset masters. Falls
    back to summing per-chunk shapes if nimages is unavailable.
    """
    try:
        with h5py.File(filepath, "r") as f:
            # Arina: ntrigger * nimages = total frames
            det = "entry/instrument/detector/detectorSpecific/"
            nimages = int(f[det + "nimages"][()]) if det + "nimages" in f else 1
            ntrigger = int(f[det + "ntrigger"][()]) if det + "ntrigger" in f else None
            if ntrigger is not None:
                return nimages * ntrigger
            # Fallback: sum shapes from data chunks
            import os
            chunk_names = _discover_chunk_names(filepath)
            if not chunk_names:
                return None
            master_dir = os.path.dirname(os.path.abspath(filepath))
            total = 0
            data_group = f["entry/data"]
            for name in chunk_names:
                link = data_group.get(name, getlink=True)
                if isinstance(link, h5py.ExternalLink):
                    data_path = os.path.join(master_dir, link.filename)
                else:
                    data_path = data_group[name].file.filename
                with h5py.File(data_path, "r") as df:
                    total += df["entry/data/data"].shape[0]
            return total
    except (OSError, KeyError, ValueError, TypeError):
        return None


def discover_masters(
    folder: str,
    pattern: str = "*_master.h5",
    recursive: bool = True,
    scan_shape: tuple[int, int] | None = None,
    verbose: bool = True,
) -> list[str]:
    """Find all master HDF5 files in a folder, sorted by path.

    Parameters
    ----------
    folder : str
        Root directory to search.
    pattern : str
        Glob pattern for matching filenames (default ``*_master.h5``).
    recursive : bool
        Search subdirectories recursively (default True).
    scan_shape : tuple[int, int], optional
        If set, only return files whose frame count matches
        ``scan_shape[0] * scan_shape[1]``. Reads HDF5 headers only,
        no decompression. Useful when a folder contains mixed scan sizes.
    verbose : bool
        Print indexed file list (default True).

    Returns
    -------
    list[str]
        Sorted list of absolute file paths.

    Raises
    ------
    FileNotFoundError
        If *folder* does not exist.
    ValueError
        If no files match the pattern.
    """
    import pathlib
    root = pathlib.Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    glob_method = root.rglob if recursive else root.glob
    paths = sorted(str(p) for p in glob_method(pattern))
    if not paths:
        raise ValueError(
            f"No files matching '{pattern}' found in {folder}"
        )
    if scan_shape is not None:
        expected_frames = scan_shape[0] * scan_shape[1]
        filtered = []
        skipped = 0
        for p in paths:
            n = _read_frame_count(p)
            if n == expected_frames:
                filtered.append(p)
            else:
                skipped += 1
        paths = filtered
        if verbose and skipped > 0:
            print(f"  Filtered: {len(paths)} files matching {scan_shape[0]}x{scan_shape[1]} "
                  f"(skipped {skipped})")
    if verbose:
        for i, p in enumerate(paths):
            print(f"  [{i:>2}] {p.split('/')[-1]}")
        print(f"\nFound {len(paths)} files in {root.name}/")
    return paths
