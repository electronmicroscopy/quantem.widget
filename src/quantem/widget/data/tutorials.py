"""Built-in tutorial datasets for documentation and Colab notebooks."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from quantem.core.datastructures import Dataset2d, Dataset3d, Dataset4dstem

from quantem.widget.io import download


def load_tutorial_show2d(*, stride: int = 8, verbose: bool = True) -> Dataset2d:
    """Load the real gold HAADF image used by the Show2D tutorial.

    Parameters
    ----------
    stride
        Pixel stride used to make a calibrated preview. Use ``stride=1`` for the
        full image.
    verbose
        If ``True``, print a short dataset summary.

    Returns
    -------
    Dataset2d
        Calibrated HAADF image preview.
    """

    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    data_dir = Path(download("gold_haadf_npy", verbose=False))
    meta = json.loads((data_dir / "meta.json").read_text())
    full_image = np.load(data_dir / "data.npy", mmap_mode="r")

    image = np.asarray(full_image[::stride, ::stride], dtype=np.float32)
    sampling = tuple(float(v) * stride for v in meta["sampling"])
    units = tuple(meta["units"])
    dataset = Dataset2d.from_array(
        image,
        sampling=sampling,
        units=units,
        name="Gold HAADF preview",
    )
    if verbose:
        print(f"Source: {meta['name']} from Hugging Face")
        print(f"Full image: {full_image.shape[0]} x {full_image.shape[1]} {full_image.dtype}")
        print(f"Preview: {image.shape[0]} x {image.shape[1]}, pixel size {sampling[0]:.4f} {units[0]}")
    return dataset


def load_tutorial_show3d(
    *,
    n_frames: int = 32,
    stride: int = 8,
    crop_size: int = 256,
    verbose: bool = True,
) -> Dataset3d:
    """Load a real HAADF image stack used by the Show3D tutorial.

    The stack is built from moving calibrated crops of the public gold HAADF
    image. This keeps the tutorial real-data based while opening quickly in
    rendered documentation and Colab.

    Parameters
    ----------
    n_frames
        Number of frames in the stack.
    stride
        Pixel stride used before cropping. Use ``stride=1`` for full image
        sampling.
    crop_size
        Spatial size of each square frame after striding.
    verbose
        If ``True``, print a short dataset summary.

    Returns
    -------
    Dataset3d
        Calibrated stack of related real HAADF views.
    """

    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if crop_size < 16:
        raise ValueError(f"crop_size must be >= 16, got {crop_size}")

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    data_dir = Path(download("gold_haadf_npy", verbose=False))
    meta = json.loads((data_dir / "meta.json").read_text())
    full_image = np.load(data_dir / "data.npy", mmap_mode="r")
    preview = np.asarray(full_image[::stride, ::stride], dtype=np.float32)
    if crop_size > min(preview.shape):
        raise ValueError(
            f"crop_size={crop_size} is larger than the strided preview shape {preview.shape}"
        )

    max_row = preview.shape[0] - crop_size
    max_col = preview.shape[1] - crop_size
    rows = np.linspace(0, max_row, n_frames)
    cols = np.linspace(max_col, 0, n_frames)
    stack = np.empty((n_frames, crop_size, crop_size), dtype=np.float32)
    for idx, (row, col) in enumerate(zip(rows, cols)):
        r0 = int(round(float(row)))
        c0 = int(round(float(col)))
        stack[idx] = preview[r0 : r0 + crop_size, c0 : c0 + crop_size]

    pixel_sampling = float(meta["sampling"][0]) * stride
    units = tuple(meta["units"])
    dataset = Dataset3d.from_array(
        stack,
        sampling=(1.0, pixel_sampling, pixel_sampling),
        units=("frame", units[0], units[1]),
        name="Gold HAADF moving-crop stack",
    )
    if verbose:
        print(f"Source: {meta['name']} from Hugging Face")
        print(f"Full image: {full_image.shape[0]} x {full_image.shape[1]} {full_image.dtype}")
        print(f"Stack: {stack.shape}, stride {stride}, pixel size {pixel_sampling:.4f} {units[0]}")
    return dataset


def load_tutorial_show4dstem(*, scan_stride: int = 2, verbose: bool = True) -> Dataset4dstem:
    """Load the real binned gold 4D-STEM stack used by the Show4DSTEM tutorial.

    Parameters
    ----------
    scan_stride
        Scan-axis stride used to make a calibrated real-data preview. Use
        ``scan_stride=1`` for the full 128 by 128 scan.
    verbose
        If ``True``, print a short dataset summary.

    Returns
    -------
    Dataset4dstem
        Calibrated gold 4D-STEM scan with 24 by 24 detector frames.
    """

    if scan_stride < 1:
        raise ValueError(f"scan_stride must be >= 1, got {scan_stride}")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    data_dir = Path(download("gold_128_npy_bin8", verbose=False))
    meta = json.loads((data_dir / "meta.json").read_text())
    full_stack = np.load(data_dir / "data.npy", mmap_mode="r")
    stack = np.asarray(full_stack[::scan_stride, ::scan_stride], dtype=np.uint16)
    sampling = tuple(
        float(v) * scan_stride if axis < 2 else float(v)
        for axis, v in enumerate(meta["sampling"])
    )
    dataset = Dataset4dstem.from_array(
        stack,
        sampling=sampling,
        units=tuple(meta["units"]),
        name="Gold 4D-STEM bin8 preview",
    )
    if verbose:
        print(f"Source: {meta['name']} from Hugging Face")
        print(f"Full stack: {full_stack.shape} {full_stack.dtype}")
        print(f"Preview: {stack.shape}, scan stride {scan_stride}")
        print(f"Sampling: {dataset.sampling} {dataset.units}")
        print(f"Processing: {meta.get('processing', 'none')}")
    return dataset
