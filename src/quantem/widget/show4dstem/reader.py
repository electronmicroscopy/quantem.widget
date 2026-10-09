"""Read a 4D-STEM file into the dataset type of this machine.

``read_4dstem`` loads a file through quantem.gpu when it is installed and a
GPU works: a ``quantem.gpu.io.Dataset4dstemGPU`` whose counts stay encoded on
the GPU and load in seconds. Everywhere else (a CPU-only machine, an Intel Mac,
a GPU box without the [cuda] or [mps] extra, or ``device="cpu"``) ``read_dense``
reads the file with h5py and hdf5plugin: into host memory for ``read_4dstem``,
whose counts come back as a quantem core ``Dataset4dstem`` (the widget's
stand-in where quantem core cannot install), or block by block onto the device
a viewer computes on. That read is slower and holds every count, so it prints
one line with the time taken and what quantem.gpu gains, and it refuses,
rather than bins or crops, when the dense array would not fit under the memory
ceiling.

hdf5plugin decodes the detector's bitshuffle-LZ4 chunks. It is imported only when
an HDF5 file is opened (``quantem.widget.io.hdf5_family.open_hdf5``), and it is
optional: Windows on ARM has no hdf5plugin wheel, and there ``.npy`` files and
HDF5 files without its filters (uncompressed or gzip) still read. A file that
needs it raises ImportError naming the filter and the install route.

Formats: an Arina/Dectris ``*_master.h5`` whose ``entry/data/data_NNNNNN``
links point at external data files, an HDF5 file with one 3D or 4D dataset at
``entry/data/data``, and ``.npy``. Detector pixels the master flags in
``pixel_mask`` are replaced by the median of their unflagged 3x3 neighbours,
the rule quantem.gpu's loader applies, so both paths show the same counts.
Integer counts keep uint8 or uint16, wider counts become uint16 when every
count fits and int64 otherwise: lossless, and the dtype quantem.gpu stores.
"""

import contextlib
import math
import os
import pathlib
import sys
import time

import numpy as np
import psutil
import torch

from quantem.widget.adapters import gpu as gpu_adapter
from quantem.widget.adapters.core import make_dataset
from quantem.widget.counts import counts_tensor, host_dtype_for_torch, tensor_dtype
from quantem.widget.device import gpu_notice, gpu_path_hint, no_gpu_path, resolve_device
from quantem.widget.io.hdf5_family import open_hdf5

MEMORY_FRACTION = 0.8  # share of the available host memory a dense read may take
BLOCK_BYTES = 256 << 20  # raw bytes read from the file per step
PIXEL_MASK_KEY = "entry/instrument/detector/detectorSpecific/pixel_mask"
BIT_DEPTH_KEY = "entry/instrument/detector/bit_depth_image"  # Dectris: the largest count is 2**bits - 1
# (largest count, stored dtype) from narrow to wide
INTEGER_LADDER = ((255, np.uint8), (65535, np.uint16), (2**63 - 1, np.int64))
FLOAT_TYPES = (np.dtype(np.float16), np.dtype(np.float32), np.dtype(np.float64))


def read_4dstem(path, device: str = "auto", *, scan_shape: tuple[int, int] | None = None,
                max_bytes: int | None = None, verbose: bool = True):
    """Read a 4D-STEM file: a ``Dataset4dstemGPU`` on a GPU, a quantem core ``Dataset4dstem`` otherwise.

    With quantem.gpu installed (``quantem.widget[cuda]`` or ``[mps]``) and a
    CUDA or Metal GPU working, ``quantem.gpu.io.load`` keeps the counts
    encoded on the GPU. Without that route the file is read whole into host
    memory (``read_dense``) and returned as a quantem core ``Dataset4dstem``;
    on an Intel Mac or Windows on ARM, which cannot install quantem core, as
    the widget's stand-in with the same ``array``, ``name``, ``sampling``,
    ``units``, ``metadata``, ``shape`` and ``dtype``. ``Show4DSTEM`` and the
    detector math give the same numbers for either.

    Parameters
    ----------
    path
        ``*_master.h5``, an HDF5 file with ``entry/data/data``, ``.npy``, or a
        list of them (a list of datasets is returned).
    device
        ``"auto"`` (quantem.gpu picks CUDA, then Metal, and prints which),
        ``"cuda"``, ``"cuda:N"`` or ``"mps"`` place an acquisition on that
        GPU; ``"cpu"`` reads densely into host memory even where quantem.gpu
        runs. A dense dataset stays in host memory; a widget computes on it
        where its own ``device=`` says.
    scan_shape
        ``(rows, cols)`` of a flat ``(positions, det_row, det_col)`` file; a
        square scan is inferred when omitted.
    max_bytes
        Largest dense array to allocate. Default: ``MEMORY_FRACTION`` of the
        available host memory.
    verbose
        Print the dense read's one-line summary.

    Raises
    ------
    MemoryError
        When a dense array would exceed ``max_bytes``. Nothing is reduced instead.

    Examples
    --------
    >>> from quantem.widget import Show4DSTEM, read_4dstem  # doctest: +SKIP
    >>> data = read_4dstem("scan_master.h5")  # doctest: +SKIP
    >>> Show4DSTEM(data)  # doctest: +SKIP
    """
    if not isinstance(path, (str, os.PathLike)):
        return [read_4dstem(item, device, scan_shape=scan_shape, max_bytes=max_bytes, verbose=verbose) for item in path]
    path = pathlib.Path(path).expanduser()
    if gpu_route(device):
        target = None if str(device).strip().lower() == "auto" else resolve_device(device)
        try:
            return gpu_adapter.load_acquisition(path, target, scan_shape=scan_shape)
        except (NotImplementedError, ValueError) as exc:
            # quantem.gpu holds a NumPy file only as exact 4D uint8, uint16 or float32 counts
            if path.suffix != ".npy":
                raise
            print(f"quantem.widget: quantem.gpu cannot hold {path.name} on the GPU ({exc}); it is read densely instead.")
    else:
        gpu_notice()
    values = read_dense(path, scan_shape=scan_shape, max_bytes=max_bytes, verbose=verbose)
    return make_dataset(values, name=acquisition_name(path))


def gpu_route(device) -> bool:
    """Whether a file read with ``device`` loads through quantem.gpu: installed, a GPU working, and not ``device="cpu"``."""
    return str(device).strip().lower() != "cpu" and gpu_adapter.accelerator_ready()


def read_dense(
    path,
    device=None,
    *,
    scan_shape: tuple[int, int] | None = None,
    hot_pixel_correction: str = "median",
    max_bytes: int | None = None,
    verbose: bool = True,
):
    """Read ``path`` into a dense ``(scan_row, scan_col, det_row, det_col)`` array of exact counts.

    Parameters
    ----------
    path
        ``*_master.h5``, an HDF5 file with ``entry/data/data``, or ``.npy``.
    device
        None returns a host NumPy array, what ``read_4dstem`` wraps in a
        dataset. A torch device returns a tensor there, filled block by block
        so the host never holds more than one block: what a viewer computing
        on that device reads (in the dtype ``counts.tensor_dtype`` gives, int32
        for uint16 counts on torch 2.2).
    scan_shape
        ``(rows, cols)`` of a flat ``(positions, det_row, det_col)`` file; a
        square scan is inferred when omitted.
    hot_pixel_correction
        ``"median"`` (default, as quantem.gpu), ``"zero"`` or ``"none"`` for the
        pixels the master's ``pixel_mask`` flags.
    max_bytes
        Largest dense array to allocate. Default: ``MEMORY_FRACTION`` of the
        available host memory, and of the free memory of a CUDA ``device``.
    verbose
        Print the one-line summary: shape, dtype, size, seconds, the quantem.gpu hint.

    Raises
    ------
    MemoryError
        When the array would exceed ``max_bytes``. Nothing is reduced instead.
    """
    path = pathlib.Path(path).expanduser()
    target = None if device is None else torch.device(device)
    limit = ceiling(target) if max_bytes is None else int(max_bytes)
    started = time.perf_counter()
    npy = path.suffix == ".npy"
    with contextlib.nullcontext() if npy else open_hdf5(path) as handle:
        datasets = [np.load(path, mmap_mode="r")] if npy else detector_datasets(handle)
        frames, det_shape, source_dtype = frame_layout(datasets)
        scan_shape = infer_scan_shape(frames, scan_shape, datasets)
        integer = source_dtype.kind in "ui"
        ladder = INTEGER_LADDER if integer else ((None, source_dtype if source_dtype in FLOAT_TYPES else np.dtype(np.float32)),)
        # Counts keep uint8 or uint16, and wider counts become uint16 when they fit, as quantem.gpu
        # stores them, so a file has one dtype on every machine. Larger or negative counts, known only
        # after reading, take int64; the file's dtype and the detector bit depth bound the largest.
        rung = 1 if integer and source_dtype.itemsize > 1 else 0
        bits = int(handle[BIT_DEPTH_KEY][()]) if not npy and BIT_DEPTH_KEY in handle else None
        bit_depth_note = "" if bits is None else f", {bits}-bit counts"
        if integer:
            info = np.iinfo(source_dtype)
            largest = info.max if bits is None else min(info.max, 2**bits - 1)
            fits = [i for i, step in enumerate(ladder) if i >= rung and largest <= step[0]]
            top = len(ladder) - 1 if info.min < 0 or not fits else fits[0]
        else:
            top = 0
        # a tensor stores each rung as torch reads it exactly (uint16 as int32 on torch 2.2)
        stored = [np.dtype(step[1]) if target is None else host_dtype_for_torch(step[1]) for step in ladder[rung : top + 1]]
        sizes = [(frames * math.prod(det_shape) * dtype.itemsize, dtype.name) for dtype in stored]
        smallest = sizes[0][0]
        if smallest > limit:
            hint = gpu_path_hint()
            if hint:
                advice = (f"Free memory, pass max_bytes= (--max-gb for the quantem command) to read it anyway, "
                          f"or keep it encoded on a GPU: {hint}")
            else:
                advice = (f"The file is read on the CPU and every count is held in memory: {no_gpu_path()}. Free "
                          "memory, pass max_bytes= (--max-gb for the quantem command) to read it anyway, open a "
                          "smaller dataset or part of the scan as an array, or use a machine with more memory.")
            options = [f"{nbytes / 1e9:.1f} GB as {name}" for nbytes, name in sizes]
            need = options[0] if len(sizes) == 1 else f"between {smallest / 1e9:.1f} GB and {sizes[-1][0] / 1e9:.1f} GB"
            stored_note = "" if len(sizes) == 1 else (
                f" It is stored as the narrowest type that holds every count: {', '.join(options[:-1])} or "
                f"{options[-1]} for this file ({source_dtype}{bit_depth_note}); the largest count is known only after reading.")
            raise MemoryError(
                f"{path.name}: the dense {scan_shape + det_shape} array needs {need}, above the {limit / 1e9:.1f} GB "
                f"ceiling ({MEMORY_FRACTION:.0%} of the available memory).{stored_note} "
                f"Nothing was binned or cropped. {advice}"
            )
        valid = None if npy else read_pixel_mask(path)
        valid = None if valid is None else np.asarray(valid) == 0
        data = None
        for start, block in frame_blocks(datasets):
            if integer and valid is not None:
                block = correct_flagged_pixels(block, valid, hot_pixel_correction)
            peak = int(block.max()) if integer and block.size else 0
            # negative counts skip the unsigned rungs
            negative = integer and block.size and int(block.min()) < 0
            while integer and (peak > ladder[rung][0] or (negative and rung < len(ladder) - 1)):
                rung += 1
                if data is not None:
                    data = promote(data, ladder[rung][1] if target is None else tensor_dtype(ladder[rung][1]), start, limit, path)
            if data is None:
                data = (np.empty((frames, *det_shape), dtype=ladder[rung][1]) if target is None else
                        torch.empty((frames, *det_shape), dtype=tensor_dtype(ladder[rung][1]), device=target))
            # every value fits the rung, so the cast is exact
            block = block.astype(ladder[rung][1], copy=False)
            data[start : start + len(block)] = block if target is None else counts_tensor(block, target)
    data = data.reshape(*scan_shape, *det_shape)
    if verbose:
        hint = gpu_path_hint()
        if gpu_adapter.available():
            hint = " quantem.gpu, installed here, keeps it encoded on a GPU when it is read on one."
        elif hint:
            hint = f" quantem.gpu keeps it encoded on a GPU and loads it faster: {hint}."
        else:
            hint = ""  # no GPU quantem.gpu runs on (an Intel Mac, no NVIDIA GPU): nothing to offer
        where = "" if target is None else f" into {target}"
        dtype_name = str(data.dtype).removeprefix("torch.")
        print(
            f"quantem.widget: read {path.name} densely on the CPU{where}: {tuple(data.shape)} {dtype_name}, "
            f"{math.prod(data.shape) * data.itemsize / 1e9:.2f} GB, {time.perf_counter() - started:.1f} s.{hint}"
        )
    return data


def acquisition_name(path) -> str:
    """Name an acquisition by its file: a master without ``_master.h5``, another file without its extension."""
    name = pathlib.Path(str(path)).name
    return name.removesuffix("_master.h5") if name.endswith("_master.h5") else pathlib.Path(name).stem


def find_masters(folder, *, pattern: str = "*_master.h5", recursive: bool = True) -> list[str]:
    """Sorted files matching ``pattern`` under ``folder``; ``master_layout`` tells which are complete."""
    folder = pathlib.Path(folder).expanduser()
    candidates = folder.rglob(pattern) if recursive else folder.glob(pattern)
    return [str(path) for path in sorted(candidates) if path.is_file()]


def master_layout(path) -> tuple[int, tuple[int, int], np.dtype] | None:
    """``(frames, detector_shape, dtype)`` of a complete master, or None while a linked file is missing."""
    try:
        with open_hdf5(path) as handle:
            return frame_layout(detector_datasets(handle))
    except (OSError, KeyError, ValueError):
        # a master whose data files are still being written cannot be read yet
        return None


def read_pixel_mask(path) -> np.ndarray | None:
    """The master's ``pixel_mask`` (nonzero = flagged detector pixel), or None when it has none."""
    with open_hdf5(path) as handle:
        return handle[PIXEL_MASK_KEY][()] if PIXEL_MASK_KEY in handle else None


def ceiling(device=None) -> int:
    """``MEMORY_FRACTION`` of the memory a dense array may take.

    The available host memory (the CPU, and Apple MPS, which shares it), and
    no more than the free memory of a CUDA ``device``.
    """
    limit = psutil.virtual_memory().available
    if device is not None and torch.device(device).type == "cuda":
        limit = min(limit, torch.cuda.mem_get_info(device)[0])
    return int(limit * MEMORY_FRACTION)


def correct_flagged_pixels(block: np.ndarray, valid: np.ndarray, method: str) -> np.ndarray:
    """Replace flagged detector pixels in a ``(frames, det_row, det_col)`` block of counts.

    ``"median"``: the median of the unflagged pixels of the 3x3 neighbourhood,
    the mean of the two middle values rounded down when their count is even,
    0 when no neighbour is valid; quantem.gpu's hot-pixel kernel computes the
    same. ``"zero"`` writes 0; ``"none"`` keeps the stored values.
    """
    if method not in ("median", "zero", "none"):
        raise ValueError("hot_pixel_correction must be 'median', 'zero', or 'none'.")
    if method == "none" or valid.all():
        return block
    block = np.array(block, copy=True)
    rows, cols = valid.shape
    for row, col in np.argwhere(~valid):
        neighbours = [
            (row + d_row, col + d_col)
            for d_row in (-1, 0, 1)
            for d_col in (-1, 0, 1)
            if (d_row or d_col) and 0 <= row + d_row < rows and 0 <= col + d_col < cols and valid[row + d_row, col + d_col]
        ]
        if method == "zero" or not neighbours:
            block[:, row, col] = 0
            continue
        neighbour_counts = np.stack([block[:, near_row, near_col] for near_row, near_col in neighbours], axis=1)
        # int64 so the sum of the two middle counts cannot overflow the stored dtype
        values = np.sort(neighbour_counts.astype(np.int64), axis=1)
        middle = len(neighbours) // 2
        block[:, row, col] = values[:, middle] if len(neighbours) % 2 else (values[:, middle - 1] + values[:, middle]) // 2
    return block


def promote(data, dtype, filled: int, limit: int, path):
    """Copy the ``filled`` leading entries read so far into a wider ``dtype`` once a block exceeds the current one.

    ``data`` is a host array with a NumPy ``dtype``, or a tensor with a torch
    ``dtype``. The narrow and the wide array exist together until the copy
    ends, so both count against the ceiling.
    """
    is_tensor = isinstance(data, torch.Tensor)
    itemsize = torch.empty((), dtype=dtype).element_size() if is_tensor else np.dtype(dtype).itemsize
    needed = math.prod(data.shape) * itemsize
    current = math.prod(data.shape) * data.itemsize
    if needed + current > limit:
        hint = gpu_path_hint()
        advice = f"keep it encoded on a GPU: {hint}" if hint else "use a machine with more memory."
        name = str(dtype).removeprefix("torch.") if is_tensor else np.dtype(dtype).name
        raise MemoryError(
            f"{pathlib.Path(path).name}: counts above the narrower type need {needed / 1e9:.1f} GB as {name} while "
            f"{current / 1e9:.1f} GB are held, above the {limit / 1e9:.1f} GB ceiling. Nothing was clipped. "
            f"Pass max_bytes= (--max-gb for the quantem command) or {advice}"
        )
    wider = torch.empty(data.shape, dtype=dtype, device=data.device) if is_tensor else np.empty(data.shape, dtype=dtype)
    wider[:filled] = data[:filled]
    return wider


# ---------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------


def detector_datasets(handle) -> list:
    """The detector datasets of an open HDF5 file in frame order: ``data_NNNNNN`` links, else ``data``.

    Raises ImportError when the data is compressed with a filter h5py cannot
    decode because hdf5plugin did not load. Without this check h5py fails at
    the first read with "can't open directory", which names neither the
    filter nor hdf5plugin.
    """
    from h5py import h5z

    group = handle.get("entry/data")
    if group is None:
        raise KeyError(f"{handle.filename} has no entry/data group; it is not a 4D-STEM master.")
    names = sorted(name for name in group if name.startswith("data_"))
    if names:
        datasets = [group[name] for name in names]
    elif "data" in group:
        datasets = [group["data"]]
    else:
        raise KeyError(f"{handle.filename} has no entry/data/data or entry/data/data_NNNNNN datasets.")
    pipeline = datasets[0].id.get_create_plist()
    for index in range(pipeline.get_nfilters()):
        code, _, _, name = pipeline.get_filter(index)
        if not h5z.filter_avail(code) and sys.modules.get("hdf5plugin") is None:
            raise ImportError(
                f"{pathlib.Path(handle.filename).name}: the detector data is compressed with the "
                f"{name.decode(errors='replace').split(';')[0]} filter (HDF5 filter {code}), which h5py reads "
                "through hdf5plugin, and hdf5plugin is not installed or did not import. Install it with "
                "pip install hdf5plugin. Windows on ARM has no hdf5plugin wheel for ARM64 Python: use x64 Python, "
                "which Windows runs under emulation, and pip install hdf5plugin there."
            )
    return datasets


def frame_layout(datasets) -> tuple[int, tuple[int, int], np.dtype]:
    """``(frames, detector_shape, dtype)`` of 3D frame stacks or one 4D scan."""
    first = datasets[0]
    if first.ndim not in (3, 4):
        raise ValueError(f"Expected 3D (frames, det_row, det_col) or 4D detector data, got shape {first.shape}.")
    det_shape = (int(first.shape[-2]), int(first.shape[-1]))
    if any(tuple(dataset.shape[-2:]) != det_shape or dataset.dtype != first.dtype for dataset in datasets):
        raise ValueError("The linked data files disagree on detector shape or dtype.")
    frames = sum(math.prod(dataset.shape[:-2]) for dataset in datasets)
    return int(frames), det_shape, np.dtype(first.dtype)


def infer_scan_shape(frames: int, scan_shape, datasets) -> tuple[int, int]:
    """``scan_shape`` if given, the scan axes of a single 4D dataset, else the square raster."""
    if scan_shape is not None:
        scan_shape = (int(scan_shape[0]), int(scan_shape[1]))
        if scan_shape[0] * scan_shape[1] != frames:
            raise ValueError(f"scan_shape={scan_shape} does not match the {frames} frames in the file.")
        return scan_shape
    if len(datasets) == 1 and datasets[0].ndim == 4:
        return (int(datasets[0].shape[0]), int(datasets[0].shape[1]))
    side = math.isqrt(frames)
    if side * side != frames:
        raise ValueError(f"The file holds {frames} frames, not a square scan; pass scan_shape=(rows, cols).")
    return (side, side)


def frame_blocks(datasets):
    """``(first_frame, block)`` pairs of ``(frames, det_row, det_col)`` blocks of about ``BLOCK_BYTES``."""
    start = 0
    for dataset in datasets:
        frame_bytes = math.prod(dataset.shape[-2:]) * dataset.dtype.itemsize
        if dataset.ndim == 3:
            step = max(1, BLOCK_BYTES // frame_bytes)
            for offset in range(0, dataset.shape[0], step):
                block = np.asarray(dataset[offset : offset + step])
                yield start + offset, block
            start += int(dataset.shape[0])
            continue
        # a 4D scan is read whole scan rows at a time
        cols = int(dataset.shape[1])
        step = max(1, BLOCK_BYTES // (cols * frame_bytes))
        for row in range(0, dataset.shape[0], step):
            block = np.asarray(dataset[row : row + step]).reshape(-1, *dataset.shape[-2:])
            yield start + row * cols, block
        start += int(dataset.shape[0]) * cols
