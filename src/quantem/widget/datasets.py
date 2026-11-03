"""Tutorial inputs: real acquisitions and synthetic reconstruction examples.

Image and scan loaders take a friendly ``size`` name and return a calibrated quantem
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

# Tests replace this (or huggingface_hub.snapshot_download) to avoid the network.
snapshot_download = None
TUTORIAL_DATA_REPO_ID = "bobleesj/quantem-data"
TUTORIAL_DATA_REVISION = "5925e4b49b2ca28ff82d6d5a6b9a8bfea9edfb20"
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
# ShowPDF's simulated Ta scan is pinned at the dataset commit that adds it,
# the next commit after TUTORIAL_DATA_REVISION.
_TA_SIM_PATH = "4dstem/ta_amorphous_sim"
_TA_SIM_REVISION = "fe907499a3a2c1f827c8827c0b3472c5b08f47bf"


def _normalize_tutorial_size(size: str) -> str:
    """Lowercase tutorial size name; anything outside ``TUTORIAL_SIZES`` is a ValueError naming the valid ones."""
    value = str(size).strip().lower()
    if value not in TUTORIAL_SIZES:
        valid = ", ".join(TUTORIAL_SIZES)
        raise ValueError(f"size must be one of {valid}; got {size!r}")
    return value


def _download_tutorial_folder(
    path_in_repo: str,
    *,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    force_download: bool = False,
) -> Path:
    """Fetch a complete input family without mixing repository revisions."""
    selected_revision = str(revision) if revision is not None else TUTORIAL_DATA_REVISION
    kwargs = {
        "repo_id": TUTORIAL_DATA_REPO_ID,
        "repo_type": "dataset",
        "force_download": bool(force_download),
        "revision": selected_revision,
        "allow_patterns": [f"{path_in_repo}/*"],
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    download = snapshot_download
    if download is None:
        from huggingface_hub import snapshot_download as download  # imported only on download

    folder = Path(download(**kwargs)) / path_in_repo
    if folder.is_dir():
        return folder
    raise FileNotFoundError(
        f"Tutorial input {path_in_repo!r} is unavailable at revision {selected_revision!r}. "
        "Use the documented default revision or your own data. Historical "
        "folder layouts require their matching software version. "
        f"Expected Hugging Face path {TUTORIAL_DATA_REPO_ID}/{path_in_repo}"
    )


def show1d_ducky(
    *,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    force_download: bool = False,
    verbose: bool = True,
) -> Path:
    """Download reconstruction snapshots for the synthetic Ducky example.

    The returned folder contains ``show1d_monitor.jsonl`` and snapshot ``.npy``
    files. Open it with :meth:`quantem.widget.Show1D.from_monitor_file`.

    The original ``Trademark ducky dataset`` simulation tutorial is by
    Georgios Varnavides (October 23, 2025). These files contain subsequent
    reconstruction results, not measured detector data.

    Parameters
    ----------
    cache_dir
        Optional Hugging Face cache directory.
    revision
        Hugging Face dataset revision. The default pins the documented inputs;
        an explicit revision is never mixed with files from another revision.
    force_download
        If ``True``, ask Hugging Face Hub to refresh the cached files.
    verbose
        If ``True``, print a short dataset summary.

    Returns
    -------
    Path
        Local folder containing the Show1D monitor run.
    """

    folder = _download_tutorial_folder(
        "widget/ducky",
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
    """Load the gold HAADF Show2D tutorial image by friendly size name.

    The source image is downloaded once from
    ``haadf/gold_haadf_npy``; every size but ``"full"``
    keeps every Nth pixel of it (stride 8, 4, 2) and says so in one line.
    The same source is reused by :func:`show3d_gold`.
    """

    size = _normalize_tutorial_size(size)
    folder = _download_tutorial_folder(
        "haadf/gold_haadf_npy", cache_dir=cache_dir,
        revision=revision, force_download=force_download,
    )
    metadata = json.loads((folder / "meta.json").read_text())
    full_image = np.load(folder / "data.npy", mmap_mode="r")
    stride = _SHOW2D_STRIDE_BY_SIZE[size]
    image = np.asarray(full_image[::stride, ::stride], dtype=np.float32)
    sampling = tuple(float(value) * stride for value in metadata["sampling"])
    units = tuple(metadata["units"])
    if stride > 1:
        print(f"show2d_gold(size={size!r}) keeps every {stride}th pixel of the {full_image.shape[0]}x{full_image.shape[1]} "
              f"image ({image.shape[0]}x{image.shape[1]}); size='full' loads native pixels.")
    if verbose:
        print(f"Source: {metadata['name']} from Hugging Face")
        print(f"Full image: {full_image.shape[0]} x {full_image.shape[1]} {full_image.dtype}")
        print(f"Preview: {image.shape[0]} x {image.shape[1]}, pixel size {sampling[0]:.4f} {units[0]}")
    return make_dataset(image, sampling=sampling, units=units, name="Gold HAADF preview")


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
    source instead of storing a second copy of the image for Show3D. Every
    size but ``"full"`` crops from every Nth pixel and says so in one line.
    """

    size = _normalize_tutorial_size(size)
    folder = _download_tutorial_folder(
        "haadf/gold_haadf_npy", cache_dir=cache_dir,
        revision=revision, force_download=force_download,
    )
    n_frames, stride, crop_size = (_SHOW3D_PARAMS_BY_SIZE[size][key] for key in ("n_frames", "stride", "crop_size"))
    metadata = json.loads((folder / "meta.json").read_text())
    full_image = np.load(folder / "data.npy", mmap_mode="r")
    preview = np.asarray(full_image[::stride, ::stride], dtype=np.float32)
    # the crop window walks from the top right to the bottom left, one frame per step
    rows = np.linspace(0, preview.shape[0] - crop_size, n_frames)
    cols = np.linspace(preview.shape[1] - crop_size, 0, n_frames)
    stack = np.empty((n_frames, crop_size, crop_size), dtype=np.float32)
    for idx, (row, col) in enumerate(zip(rows, cols)):
        row_start, col_start = int(round(float(row))), int(round(float(col)))
        stack[idx] = preview[row_start : row_start + crop_size, col_start : col_start + crop_size]
    pixel_sampling = float(metadata["sampling"][0]) * stride
    units = tuple(metadata["units"])
    if stride > 1:
        print(f"show3d_gold(size={size!r}) crops from every {stride}th pixel of the {full_image.shape[0]}x{full_image.shape[1]} "
              "image; size='full' crops native pixels.")
    if verbose:
        print(f"Source: {metadata['name']} from Hugging Face")
        print(f"Full image: {full_image.shape[0]} x {full_image.shape[1]} {full_image.dtype}")
        print(f"Stack: {stack.shape}, stride {stride}, pixel size {pixel_sampling:.4f} {units[0]}")
    return make_dataset(
        stack,
        sampling=(1.0, pixel_sampling, pixel_sampling),
        units=("frame", units[0], units[1]),
        name="Gold HAADF moving-crop stack",
    )


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
    ``4dstem/gold_128_npy_bin8``; ``"small"`` and
    ``"medium"`` keep every 4th or 2nd scan position and say so in one line.
    The file is read as every 4D-STEM file is (``read_4dstem``): a
    ``quantem.gpu.io.Dataset4dstemGPU`` that stays on the GPU when quantem.gpu
    and a GPU are present, else a quantem core ``Dataset4dstem``.
    """

    size = _normalize_tutorial_size(size)
    folder = _download_tutorial_folder(
        "4dstem/gold_128_npy_bin8", cache_dir=cache_dir,
        revision=revision, force_download=force_download,
    )
    scan_stride = _SHOW4DSTEM_SCAN_STRIDE_BY_SIZE[size]
    metadata = json.loads((folder / "meta.json").read_text())
    source = folder / "data.npy"
    loaded = read_4dstem(source, verbose=verbose)
    name = "Gold 4D-STEM bin8 preview"
    sampling = [float(value) * scan_stride if axis < 2 else float(value) for axis, value in enumerate(metadata["sampling"])]
    units = [str(unit) for unit in metadata["units"]]
    # the sidecar's optics travel with the dataset under the .qem flat names, so Compute SSB reads them from it
    optics = {"voltage_kV": metadata.get("voltage_kV"), "semiangle_mrad": metadata.get("probe_semiangle_mrad")}
    if gpu_adapter.is_acquisition(loaded):
        with loaded:
            preview_t = loaded[::scan_stride, ::scan_stride].contiguous()
        dataset = gpu_adapter.acquisition_from_tensor(
            preview_t, {"name": name, "sampling": sampling, "units": units, "source_path": str(source), **optics})
    else:
        stack = np.ascontiguousarray(as_array(loaded)[::scan_stride, ::scan_stride])
        dataset = make_dataset(stack, sampling=sampling, units=units, name=name)
        dataset.metadata.update(optics)
    if scan_stride > 1:
        print(f"show4dstem_gold(size={size!r}) keeps every {scan_stride}th scan position of the "
              f"{loaded.shape[0]}x{loaded.shape[1]} scan; size='large' or 'full' loads every position.")
    if verbose:
        print(f"Source: {metadata['name']} from Hugging Face")
        print(f"Full stack: {tuple(loaded.shape)} {loaded.dtype}")
        print(f"Preview: {tuple(dataset.shape)}, scan stride {scan_stride}")
        print(f"Sampling: {[float(value) for value in dataset.sampling]} {[str(unit) for unit in dataset.units]}")
        print(f"Processing: {metadata.get('processing', 'none')}")
    return dataset


def gold_session(
    *, size: str = "small", cache_dir: str | Path | None = None,
    revision: str | None = None, force_download: bool = False,
    verbose: bool = True,
) -> Path:
    """Download the public gold HAADF session for image-reader tutorials.

    The pinned default uses ``widget/gold_session``. The one measured session
    is downloaded intact; ``size`` is retained for existing notebook calls.

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
    size = _normalize_tutorial_size(size)
    if size != "small":
        raise ValueError(f"Only one gold session is available; use size='small', got {size!r}.")
    folder = _download_tutorial_folder(
        "widget/gold_session", cache_dir=cache_dir,
        revision=revision, force_download=force_download,
    )
    if verbose:
        print(f"Gold image session: {len(list(folder.glob('*.emd')))} EMD files")
    return folder


def showpdf_ta(
    *,
    cache_dir: str | Path | None = None,
    revision: str | None = _TA_SIM_REVISION,
    force_download: bool = False,
    verbose: bool = True,
):
    """Load the simulated amorphous Ta 4D-STEM scan used by the ShowPDF tutorial.

    The scan is downloaded from ``4dstem/ta_amorphous_sim``: a 20 x 20 scan of
    307 x 307 float32 patterns simulated from a 1,024,000-atom amorphous Ta
    model, with the detector calibrated in 1/Å (k = 1/d). The model's number
    density is 1,024,000 / 268.6053³ atoms/Å³.

    Parameters
    ----------
    cache_dir
        Optional Hugging Face cache directory.
    revision
        Hugging Face dataset revision; defaults to the pinned one.
    force_download
        If ``True``, ask Hugging Face Hub to refresh the cached files.
    verbose
        If ``True``, print a short dataset summary.

    Returns
    -------
    Dataset4dstem
        The calibrated scan, a quantem core ``Dataset4dstem`` (the widget's
        stand-in where quantem core cannot install).
    """
    folder = _download_tutorial_folder(
        _TA_SIM_PATH, cache_dir=cache_dir, revision=revision, force_download=force_download
    )
    metadata = json.loads((folder / "meta.json").read_text())
    data = np.load(folder / "data.npy")
    dataset = make_dataset(
        data, sampling=metadata["sampling"], units=metadata["units"], name="Simulated amorphous Ta"
    )
    if verbose:
        print(f"Source: {metadata['name']} from Hugging Face")
        print(f"Scan: {data.shape} {data.dtype}")
        print(f"Sampling: {[float(value) for value in dataset.sampling]} {list(dataset.units)}")
    return dataset


def showpdf_ta_ground_truth(
    *,
    cache_dir: str | Path | None = None,
    revision: str | None = _TA_SIM_REVISION,
    force_download: bool = False,
):
    """g(r) of the atomic model behind ``showpdf_ta``: ``(r, g)`` arrays, r in Å."""
    folder = _download_tutorial_folder(
        _TA_SIM_PATH, cache_dir=cache_dir, revision=revision, force_download=force_download
    )
    with np.load(folder / "ground_truth_gr.npz") as truth:
        return truth["r"], truth["gr"]


__all__ = [
    "gold_session",
    "show1d_ducky",
    "show2d_gold",
    "show3d_gold",
    "show4dstem_gold",
    "showpdf_ta",
    "showpdf_ta_ground_truth",
]
