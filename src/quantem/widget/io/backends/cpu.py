"""CPU decompress backend for arina 4D-STEM masters.

Pure h5py + hdf5plugin: the bitshuffle+LZ4 HDF5 filter is registered at import
(hdf5plugin), so slicing a dataset transparently decompresses on the CPU — no
custom kernel, works on any platform with no GPU. Returns a numpy array.

This is the universal fallback for the view/screen path on a non-CUDA box
(a Mac without Metal, a plain laptop). Slower than the cuda/mps kernels but
produces bit-identical raw frames (it reads the same compressed chunks).

Ported from quantem.widget's `_load_arina_cpu`, with one deliberate change:
detector binning uses an **integer sum** (uint64 accumulator) to match the
cuda backend's reduction, not widget's float mean — so a binned cpu load and a
binned cuda load agree numerically.
"""
from __future__ import annotations

import os

import h5py
import hdf5plugin  # noqa: F401 - registers the bitshuffle+LZ4 filter
import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **kwargs):
        return it


def _bin_sum(frames: np.ndarray, factor: int) -> np.ndarray:
    """Integer-sum bin the detector axes of (n, rows, cols) by `factor`.

    uint64 accumulator (zero rounding, per the integer-for-sums rule), then
    cast back to the input dtype — matching the cuda loader, which sums binned
    pixels and keeps the native integer dtype. Trailing pixels that don't fill
    a full bin are trimmed (same as cuda).
    """
    n, rows, cols = frames.shape
    out_rows, out_cols = rows // factor, cols // factor
    trimmed = frames[:, : out_rows * factor, : out_cols * factor]
    summed = (
        trimmed.reshape(n, out_rows, factor, out_cols, factor)
        .sum(axis=(2, 4), dtype=np.uint64)
    )
    return summed.astype(frames.dtype)


def load_master(
    filepath: str,
    *,
    det_bin: int = 1,
    pixel_mask: "np.ndarray | None" = None,
    verbose: bool = True,
) -> np.ndarray:
    """Decompress an arina master (+ its external chunk files) to a numpy 3D
    array ``(n_frames, det_row, det_col)``. The ``load()`` wrapper owns the
    scan-shape unflatten, auto_narrow and metadata, exactly as for cuda.

    Returns native dtype at det_bin=1; integer-sum binned (same dtype) when
    det_bin > 1.

    ``pixel_mask`` (the raw detector mask, nonzero = dead) is applied — dead
    pixels zeroed — to each chunk BEFORE binning, matching the cuda path. This
    matters at det_bin > 1: an unmasked Arina dead pixel carries a 65535
    sentinel that would otherwise dominate its bin (a 2× total-intensity error
    from one pixel).
    """
    master_dir = os.path.dirname(os.path.abspath(filepath))
    with h5py.File(filepath, "r") as f:
        data_group = f.get("entry/data")
        if data_group is None:
            raise ValueError(f"{filepath}: no entry/data group")
        chunk_keys = sorted(k for k in data_group.keys() if k.startswith("data_"))
        if not chunk_keys:
            raise ValueError(f"{filepath}: no data_NNNNNN chunks")
        ds0 = data_group[chunk_keys[0]]
        det_row, det_col = ds0.shape[1:]
        dtype = ds0.dtype

    prefix = os.path.basename(filepath).replace("_master.h5", "")
    chunk_files, chunk_n_frames = [], []
    for key in chunk_keys:
        suffix = key.split("_")[-1]
        chunk_file = os.path.join(master_dir, f"{prefix}_data_{suffix}.h5")
        if not os.path.exists(chunk_file):
            raise FileNotFoundError(f"Missing chunk file: {os.path.basename(chunk_file)}")
        chunk_files.append(chunk_file)
        with h5py.File(chunk_file, "r") as f:
            chunk_n_frames.append(f["entry/data/data"].shape[0])

    total_frames = sum(chunk_n_frames)
    if det_bin > 1:
        out_row, out_col = det_row // det_bin, det_col // det_bin
    else:
        out_row, out_col = det_row, det_col
    output = np.empty((total_frames, out_row, out_col), dtype=dtype)

    bad = None
    if pixel_mask is not None:
        bad = np.asarray(pixel_mask) != 0
        if bad.shape != (det_row, det_col):
            bad = None  # mask shape mismatch → skip rather than corrupt

    offset = 0
    chunk_iter = chunk_files
    if verbose and len(chunk_files) > 1:
        chunk_iter = tqdm(chunk_files, desc="chunks", leave=False)
    for chunk_file in chunk_iter:
        with h5py.File(chunk_file, "r") as f:
            raw = f["entry/data/data"][:]  # hdf5plugin decompresses here
        if bad is not None:
            raw[:, bad] = 0  # zero dead pixels BEFORE binning (matches cuda)
        n = raw.shape[0]
        output[offset : offset + n] = _bin_sum(raw, det_bin) if det_bin > 1 else raw
        offset += n
    return output
