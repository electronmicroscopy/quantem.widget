"""Tutorial datasets: small real microscopy data downloaded once from Hugging Face.

Every loader takes a friendly ``size`` name and returns a calibrated quantem
dataset (or a folder of files), so the tutorials never ship data in git: a
quantem core ``Dataset2d``/``Dataset3d`` for images and stacks, and for the
4D-STEM scan a ``quantem.gpu.io.Dataset4dstemGPU`` on a GPU or a quantem core
``Dataset4dstem`` elsewhere (the widget's stand-in where quantem core cannot
install).
"""

import json
import os
from pathlib import Path

import numpy as np

from quantem.widget.adapters import gpu as gpu_adapter
from quantem.widget.adapters.core import as_array, make_dataset
from quantem.widget.show4dstem.reader import read_4dstem

snapshot_download = None
TUTORIAL_DATA_REPO_ID = "bobleesj/quantem-data"
TUTORIAL_DATA_ROOT = "widget-tutorials"
TUTORIAL_SIZES = ("small", "medium", "large", "full")
_SHOW2D_STRIDE_BY_SIZE = {
    "small": 8,
    "medium": 4,
    "large": 2,
    "full": 1,
}
_SHOW3D_PARAMS_BY_SIZE = {
    "small": {"n_frames": 32, "stride": 8, "crop_size": 256},
    "medium": {"n_frames": 48, "stride": 4, "crop_size": 384},
    "large": {"n_frames": 64, "stride": 2, "crop_size": 512},
    "full": {"n_frames": 64, "stride": 1, "crop_size": 512},
}
_SHOW4DSTEM_SCAN_STRIDE_BY_SIZE = {
    "small": 4,
    "medium": 2,
    "large": 1,
    "full": 1,
}
_GOLD_HAADF_VIEWER = "shared"
_GOLD_HAADF_NAME = "gold-haadf"
_GOLD_HAADF_SOURCE_SIZE = "full"
_GOLD_4DSTEM_VIEWER = "show4dstem"
_GOLD_4DSTEM_NAME = "gold-128-bin8"
_GOLD_4DSTEM_SOURCE_SIZE = "full"
_FE3O4_SAED_VIEWER = "showdiffraction"
_FE3O4_SAED_NAME = "fe3o4-saed"


def _snapshot_download_dataset(**kwargs) -> Path:
    """``huggingface_hub.snapshot_download`` as a Path, imported on first use.

    Importing the loaders stays light, and the module-level
    ``snapshot_download`` is the one name tests replace to avoid the network.
    """
    global snapshot_download
    if snapshot_download is None:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
    return Path(snapshot_download(**kwargs))


def _normalise_tutorial_size(size: str) -> str:
    """Lowercase tutorial size name; anything outside ``TUTORIAL_SIZES`` is a ValueError naming the valid ones."""
    value = str(size).strip().lower()
    if value not in TUTORIAL_SIZES:
        valid = ", ".join(TUTORIAL_SIZES)
        raise ValueError(f"size must be one of {valid}; got {size!r}")
    return value


def _download_widget_tutorial_folder(
    viewer: str,
    name: str,
    *,
    size: str = "small",
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    force_download: bool = False,
) -> Path:
    """Download one ``widget-tutorials/<viewer>/<name>/<size>`` folder and return its local path.

    Only that folder is fetched (``allow_patterns``), so a small tutorial
    never pulls the full-size payloads stored next to it.
    """
    size = _normalise_tutorial_size(size)
    path_in_repo = f"{TUTORIAL_DATA_ROOT}/{viewer}/{name}/{size}"
    kwargs = {
        "repo_id": TUTORIAL_DATA_REPO_ID,
        "repo_type": "dataset",
        "allow_patterns": [f"{path_in_repo}/*"],
        "force_download": bool(force_download),
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if revision is not None:
        kwargs["revision"] = str(revision)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    root = _snapshot_download_dataset(**kwargs)
    folder = root / path_in_repo
    if not folder.is_dir():
        raise FileNotFoundError(
            f"downloaded tutorial folder is missing: {folder}. "
            f"Expected Hugging Face path {TUTORIAL_DATA_REPO_ID}/{path_in_repo}"
        )
    return folder


def show1d_ducky(
    *,
    size: str = "small",
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    force_download: bool = False,
    verbose: bool = True,
) -> Path:
    """Download the real ducky joint-time ptychography Show1D tutorial run.

    The returned folder contains ``show1d_monitor.jsonl`` and snapshot ``.npy``
    files. Open it with :meth:`quantem.widget.Show1D.from_monitor_file`.

    Parameters
    ----------
    size
        Tutorial payload size. Valid values are ``"small"``, ``"medium"``,
        ``"large"``, and ``"full"``. The current public upload provides the
        ``"small"`` payload; larger sizes are reserved for future higher
        resolution snapshots.
    cache_dir
        Optional Hugging Face cache directory.
    revision
        Optional Hugging Face dataset revision.
    force_download
        If ``True``, ask Hugging Face Hub to refresh the cached files.
    verbose
        If ``True``, print a short dataset summary.

    Returns
    -------
    Path
        Local folder containing the Show1D monitor run.
    """

    folder = _download_widget_tutorial_folder(
        "show1d",
        "ducky",
        size=size,
        cache_dir=cache_dir,
        revision=revision,
        force_download=force_download,
    )
    monitor = folder / "show1d_monitor.jsonl"
    if not monitor.is_file():
        raise FileNotFoundError(f"Show1D ducky tutorial monitor is missing: {monitor}")
    if verbose:
        events = sum(1 for line in monitor.read_text(encoding="utf-8").splitlines() if line.strip())
        print(f"Show1D ducky tutorial run: {folder}")
        print(f"Monitor events: {events}")
    return folder


def show2d_gold(
    *,
    size: str = "small",
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    force_download: bool = False,
    verbose: bool = True,
):
    """Load the gold HAADF Show2D tutorial dataset by friendly size name.

    The source image is downloaded once from
    ``widget-tutorials/shared/gold-haadf/full`` and the requested size controls
    the preview stride. The same source is reused by :func:`show3d_gold`.
    """

    size = _normalise_tutorial_size(size)
    folder = _download_widget_tutorial_folder(
        _GOLD_HAADF_VIEWER,
        _GOLD_HAADF_NAME,
        size=_GOLD_HAADF_SOURCE_SIZE,
        cache_dir=cache_dir,
        revision=revision,
        force_download=force_download,
    )
    return _gold_haadf_2d_from_folder(
        folder,
        stride=_SHOW2D_STRIDE_BY_SIZE[size],
        verbose=verbose,
    )


def show3d_gold(
    *,
    size: str = "small",
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    force_download: bool = False,
    verbose: bool = True,
):
    """Load the gold HAADF Show3D tutorial stack by friendly size name.

    The stack is built from moving crops of the shared gold HAADF tutorial
    source instead of storing a second copy of the image for Show3D.
    """

    size = _normalise_tutorial_size(size)
    folder = _download_widget_tutorial_folder(
        _GOLD_HAADF_VIEWER,
        _GOLD_HAADF_NAME,
        size=_GOLD_HAADF_SOURCE_SIZE,
        cache_dir=cache_dir,
        revision=revision,
        force_download=force_download,
    )
    params = _SHOW3D_PARAMS_BY_SIZE[size]
    return _gold_haadf_3d_from_folder(folder, **params, verbose=verbose)


def show4dstem_gold(
    *,
    size: str = "small",
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    force_download: bool = False,
    verbose: bool = True,
):
    """Load the gold 4D-STEM Show4DSTEM tutorial scan by friendly size name.

    The source scan is stored once under
    ``widget-tutorials/show4dstem/gold-128-bin8/full`` and the requested size
    controls the scan-axis stride. Returns a ``quantem.gpu.io.Dataset4dstemGPU``
    when quantem.gpu is installed and a GPU is present, else a quantem core
    ``Dataset4dstem``, as ``read_4dstem`` does for any 4D-STEM file.
    """

    size = _normalise_tutorial_size(size)
    folder = _download_widget_tutorial_folder(
        _GOLD_4DSTEM_VIEWER,
        _GOLD_4DSTEM_NAME,
        size=_GOLD_4DSTEM_SOURCE_SIZE,
        cache_dir=cache_dir,
        revision=revision,
        force_download=force_download,
    )
    scan_stride = _SHOW4DSTEM_SCAN_STRIDE_BY_SIZE[size]
    return _gold_4dstem_from_folder(folder, scan_stride=scan_stride, verbose=verbose)


def gold_session(
    *, size: str = "small", cache_dir: str | Path | None = None,
    revision: str | None = None, force_download: bool = False,
    verbose: bool = True,
) -> Path:
    """Download the public gold HAADF session for image-reader tutorials.

    Parameters
    ----------
    size
        Tutorial size, normally ``"small"``.
    cache_dir, revision, force_download
        Download cache, dataset revision and refresh controls.
    verbose
        Print the number of downloaded EMD files.

    Returns
    -------
    Path
        Cached folder containing the public EMD images.

    Examples
    --------
    >>> folder = gold_session()
    >>> images = sorted(folder.glob("*.emd"))
    """
    size = _normalise_tutorial_size(size)
    folder = _download_widget_tutorial_folder(
        "showfolder", "gold-haadf-session", size=size, cache_dir=cache_dir,
        revision=revision, force_download=force_download,
    )
    if verbose:
        print(f"Gold image session: {len(list(folder.glob('*.emd')))} EMD files")
    return folder


def showdiffraction_fe3o4(
    *,
    size: str = "small",
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    force_download: bool = False,
    verbose: bool = True,
):
    """Load the real Fe3O4 nanoparticle SAED pattern used by the ShowDiffraction tutorial.

    The pattern is downloaded from
    ``widget-tutorials/showdiffraction/fe3o4-saed``. It is uncalibrated by
    design; the tutorial calibrates it against the Fe3O4 phase.

    Parameters
    ----------
    size
        Tutorial payload size. Valid values are ``"small"``, ``"medium"``,
        ``"large"``, and ``"full"``. The public upload provides the ``"small"``
        payload.
    cache_dir
        Optional Hugging Face cache directory.
    revision
        Optional Hugging Face dataset revision.
    force_download
        If ``True``, ask Hugging Face Hub to refresh the cached files.
    verbose
        If ``True``, print a short dataset summary.

    Returns
    -------
    Dataset2d
        The 512 by 512 float32 diffraction pattern (``.array``), a quantem core
        ``Dataset2d`` like every 2D tutorial loader returns.
    """

    size = _normalise_tutorial_size(size)
    folder = _download_widget_tutorial_folder(
        _FE3O4_SAED_VIEWER,
        _FE3O4_SAED_NAME,
        size=size,
        cache_dir=cache_dir,
        revision=revision,
        force_download=force_download,
    )
    source = folder / "data.npy"
    if not source.is_file():
        raise FileNotFoundError(f"tutorial pattern is missing: {source}")
    pattern = np.asarray(np.load(source), dtype=np.float32)
    if verbose:
        print(f"Fe3O4 SAED tutorial pattern: {source}")
        print(f"Pattern: {pattern.shape[0]} x {pattern.shape[1]} {pattern.dtype}")
    return make_dataset(pattern, name="Fe3O4 SAED")


def _gold_haadf_2d_from_folder(folder: Path, *, stride: int, verbose: bool = True):
    """Build a calibrated 2D HAADF preview from the shared tutorial source."""

    metadata = json.loads((folder / "meta.json").read_text())
    full_image = np.load(folder / "data.npy", mmap_mode="r")

    image = np.asarray(full_image[::stride, ::stride], dtype=np.float32)
    sampling = tuple(float(value) * stride for value in metadata["sampling"])
    units = tuple(metadata["units"])
    dataset = make_dataset(
        image,
        sampling=sampling,
        units=units,
        name="Gold HAADF preview",
    )
    if verbose:
        print(f"Source: {metadata['name']} from Hugging Face")
        print(f"Full image: {full_image.shape[0]} x {full_image.shape[1]} {full_image.dtype}")
        print(f"Preview: {image.shape[0]} x {image.shape[1]}, pixel size {sampling[0]:.4f} {units[0]}")
    return dataset


def _gold_haadf_3d_from_folder(
    folder: Path,
    *,
    n_frames: int,
    stride: int,
    crop_size: int,
    verbose: bool = True,
):
    """Build a moving-crop stack from the shared HAADF tutorial source."""

    metadata = json.loads((folder / "meta.json").read_text())
    full_image = np.load(folder / "data.npy", mmap_mode="r")
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
        row_start = int(round(float(row)))
        col_start = int(round(float(col)))
        stack[idx] = preview[row_start : row_start + crop_size, col_start : col_start + crop_size]

    pixel_sampling = float(metadata["sampling"][0]) * stride
    units = tuple(metadata["units"])
    dataset = make_dataset(
        stack,
        sampling=(1.0, pixel_sampling, pixel_sampling),
        units=("frame", units[0], units[1]),
        name="Gold HAADF moving-crop stack",
    )
    if verbose:
        print(f"Source: {metadata['name']} from Hugging Face")
        print(f"Full image: {full_image.shape[0]} x {full_image.shape[1]} {full_image.dtype}")
        print(f"Stack: {stack.shape}, stride {stride}, pixel size {pixel_sampling:.4f} {units[0]}")
    return dataset


def _gold_4dstem_from_folder(
    folder: Path,
    *,
    scan_stride: int = 2,
    verbose: bool = True,
):
    """Build a calibrated 4D-STEM preview from the widget tutorial source.

    The file is read as every 4D-STEM file is (``read_4dstem``): on a GPU the
    preview is taken from the GPU acquisition and stays there as a
    ``Dataset4dstemGPU``; otherwise it is a quantem core ``Dataset4dstem``.
    """

    metadata = json.loads((folder / "meta.json").read_text())
    source = folder / "data.npy"
    loaded = read_4dstem(source, verbose=verbose)
    name = "Gold 4D-STEM bin8 preview"
    sampling = [float(value) * scan_stride if axis < 2 else float(value) for axis, value in enumerate(metadata["sampling"])]
    units = [str(unit) for unit in metadata["units"]]
    if gpu_adapter.is_acquisition(loaded):
        with loaded:
            preview_t = loaded[::scan_stride, ::scan_stride].contiguous()
        dataset = gpu_adapter.acquisition_from_tensor(
            preview_t, {"name": name, "sampling": sampling, "units": units, "source_path": str(source)})
    else:
        stack = np.ascontiguousarray(as_array(loaded)[::scan_stride, ::scan_stride])
        dataset = make_dataset(stack, sampling=sampling, units=units, name=name)
    if verbose:
        print(f"Source: {metadata['name']} from Hugging Face")
        print(f"Full stack: {tuple(loaded.shape)} {loaded.dtype}")
        print(f"Preview: {tuple(dataset.shape)}, scan stride {scan_stride}")
        print(f"Sampling: {[float(value) for value in dataset.sampling]} {[str(unit) for unit in dataset.units]}")
        print(f"Processing: {metadata.get('processing', 'none')}")
    return dataset


__all__ = [
    "gold_session",
    "show1d_ducky",
    "show2d_gold",
    "show3d_gold",
    "show4dstem_gold",
    "showdiffraction_fe3o4",
]
