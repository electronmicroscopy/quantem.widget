"""Built-in tutorial datasets for documentation and Colab notebooks."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
from quantem.core.datastructures import Dataset2d, Dataset3d, Dataset4dstem

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
_SHOWFOLDER_GOLD_VIEWER = "showfolder"
_SHOWFOLDER_GOLD_NAME = "gold-haadf-session"


def _snapshot_download_dataset(**kwargs) -> Path:
    global snapshot_download
    if snapshot_download is None:
        try:
            from huggingface_hub import snapshot_download as _snapshot_download  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "tutorial dataset downloads require huggingface_hub. "
                "Install it with `pip install huggingface_hub`."
            ) from exc

        snapshot_download = _snapshot_download
    return Path(snapshot_download(**kwargs))


def _normalise_tutorial_size(size: str) -> str:
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
    files. Use it with :meth:`quantem.widget.Show1D.from_monitor_file`, or call
    ``Show1D.from_example("ducky")`` when a one-line widget is preferred.

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
) -> Dataset2d:
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
) -> Dataset3d:
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
) -> Dataset4dstem:
    """Load the gold 4D-STEM Show4DSTEM tutorial scan by friendly size name.

    The source scan is stored once under
    ``widget-tutorials/show4dstem/gold-128-bin8/full`` and the requested size
    controls the scan-axis stride.
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


def showfolder_gold(
    *,
    size: str = "small",
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    force_download: bool = False,
    verbose: bool = True,
    allow_fallback: bool = True,
) -> Path:
    """Download the compact gold HAADF ShowFolder tutorial session.

    The returned folder contains the public ``.emd`` files used by the
    ShowFolder tutorial under
    ``widget-tutorials/showfolder/gold-haadf-session/small``.
    """

    _normalise_tutorial_size(size)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        folder = _download_widget_tutorial_folder(
            _SHOWFOLDER_GOLD_VIEWER,
            _SHOWFOLDER_GOLD_NAME,
            size=size,
            cache_dir=cache_dir,
            revision=revision,
            force_download=force_download,
        )
    except Exception:
        if not allow_fallback:
            raise
        folder = create_tutorial_showfolder_folder()

    if verbose:
        files = sorted(path.name for path in folder.glob("*.emd"))
        print(f"Tutorial ShowFolder folder: {folder}")
        print(f"Files: {len(files)} EMD")
    return folder


def _gold_haadf_2d_from_folder(folder: Path, *, stride: int, verbose: bool = True) -> Dataset2d:
    """Build a calibrated 2D HAADF preview from the shared tutorial source."""

    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    meta = json.loads((folder / "meta.json").read_text())
    full_image = np.load(folder / "data.npy", mmap_mode="r")

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


def _gold_haadf_3d_from_folder(
    folder: Path,
    *,
    n_frames: int,
    stride: int,
    crop_size: int,
    verbose: bool = True,
) -> Dataset3d:
    """Build a moving-crop stack from the shared HAADF tutorial source."""

    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if crop_size < 16:
        raise ValueError(f"crop_size must be >= 16, got {crop_size}")

    meta = json.loads((folder / "meta.json").read_text())
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


def _gold_4dstem_from_folder(
    folder: Path,
    *,
    scan_stride: int = 2,
    verbose: bool = True,
) -> Dataset4dstem:
    """Build a calibrated 4D-STEM preview from the widget tutorial source."""

    if scan_stride < 1:
        raise ValueError(f"scan_stride must be >= 1, got {scan_stride}")
    meta = json.loads((folder / "meta.json").read_text())
    full_stack = np.load(folder / "data.npy", mmap_mode="r")
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


def load_tutorial_showfolder_folder(*, verbose: bool = True, allow_fallback: bool = True) -> Path:
    """Download the real HAADF EMD folder used by the ShowFolder tutorial.

    Returns a folder containing a compact 26-file Velox ``.emd`` session from
    the public ``bobleesj/quantem-data`` Hugging Face dataset. The folder is
    small enough for documentation but still exercises the real
    ``ShowFolder(folder)`` path: Velox EMD image loading, metadata parsing, labels,
    repeated field-of-view grouping, thumbnails, scale bars, and the inventory
    table.

    Parameters
    ----------
    verbose
        If ``True``, print a short folder summary.
    allow_fallback
        If ``True``, create a tiny offline EMD folder when the Hugging Face
        download is unavailable. Documentation notebooks set this to ``False``
        so rendered pages always use the real public dataset.

    Returns
    -------
    Path
        Folder containing tutorial ``.emd`` files.
    """

    return showfolder_gold(verbose=verbose, allow_fallback=allow_fallback)


def create_tutorial_showfolder_folder(path: str | Path | None = None) -> Path:
    """Create a tiny Velox-like EMD folder for offline ShowFolder tests.

    The generated folder mimics a microscope session: two HAADF images from the
    same field of view, one matching EDS spectrum-image file, and one lower-mag
    overview image. Files are deliberately small so documentation notebooks can
    embed the ShowFolder widgets without making the site heavy.

    Parameters
    ----------
    path
        Optional destination folder. If omitted, a stable folder under the
        system temporary directory is used.

    Returns
    -------
    Path
        Folder containing the generated ``.emd`` files.
    """

    root = Path(path) if path is not None else Path(tempfile.gettempdir()) / "quantem-widget-showfolder-demo"
    root.mkdir(parents=True, exist_ok=True)
    for old in root.glob("*.emd"):
        old.unlink()

    _write_tutorial_image_emd(root / "0010 - HAADF 15Mx Nano.emd", rotation_deg=0.0, seed=2)
    _write_tutorial_image_emd(root / "0011 - HAADF 15Mx Nano 90deg.emd", rotation_deg=90.0, seed=3)
    _write_tutorial_eds_emd(root / "0012 - HAADF 15Mx Nano EDS.emd")
    _write_tutorial_image_emd(
        root / "0020 - HAADF 3.7Mx Nano overview.emd",
        rotation_deg=0.0,
        seed=8,
        stage=(1.4e-6, 2.4e-6, 3e-6),
        fov_nm=(150.0, 150.0),
    )
    return root


def _tutorial_showfolder_metadata(
    rotation_deg: float,
    *,
    stage: tuple[float, float, float] = (1e-6, 2e-6, 3e-6),
    fov_nm: tuple[float, float] = (36.0, 36.0),
    pixel_nm: float = 0.28,
) -> np.ndarray:
    meta = {
        "Scan": {
            "ScanRotation": str(np.deg2rad(rotation_deg)),
            "ScanSize": {"height": 96, "width": 96},
        },
        "BinaryResult": {"PixelSize": {"height": pixel_nm * 1e-9, "width": pixel_nm * 1e-9}},
        "Stage": {"Position": {"x": stage[0], "y": stage[1], "z": stage[2]}},
        "Optics": {
            "FullScanFieldOfView": {
                "height": fov_nm[0] * 1e-9,
                "width": fov_nm[1] * 1e-9,
            },
        },
    }
    text = json.dumps(meta).encode("utf-8")
    out = np.zeros((len(text) + 1, 1), dtype=np.uint8)
    out[: len(text), 0] = np.frombuffer(text, dtype=np.uint8)
    return out


def _tutorial_showfolder_image(shape: tuple[int, int] = (96, 96), *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[-1:1:complex(shape[0]), -1:1:complex(shape[1])]
    particles = (
        1.3 * np.exp(-((x + 0.32) ** 2 + (y - 0.18) ** 2) / 0.035)
        + 0.9 * np.exp(-((x - 0.18) ** 2 + (y + 0.12) ** 2) / 0.018)
        + 0.55 * np.exp(-((x - 0.42) ** 2 + (y - 0.38) ** 2) / 0.012)
    )
    scan_texture = 0.08 * np.sin(18 * x + 3 * y) + 0.04 * rng.normal(size=shape)
    return (particles + scan_texture).astype(np.float32)


def _write_tutorial_image_emd(
    path: Path,
    *,
    rotation_deg: float,
    seed: int,
    stage: tuple[float, float, float] = (1e-6, 2e-6, 3e-6),
    fov_nm: tuple[float, float] = (36.0, 36.0),
) -> None:
    import h5py  # noqa: PLC0415

    with h5py.File(path, "w") as h:
        group = h.create_group("Data/Image/uid")
        group.create_dataset("Data", data=_tutorial_showfolder_image(seed=seed))
        group.create_dataset(
            "Metadata",
            data=_tutorial_showfolder_metadata(rotation_deg, stage=stage, fov_nm=fov_nm),
        )


def _write_tutorial_eds_emd(
    path: Path,
    *,
    rotation_deg: float = 90.0,
    stage: tuple[float, float, float] = (1e-6, 2e-6, 3e-6),
    fov_nm: tuple[float, float] = (36.0, 36.0),
) -> None:
    import h5py  # noqa: PLC0415

    with h5py.File(path, "w") as h:
        group = h.create_group("Data/SpectrumImage/uid")
        group.create_dataset("Data", data=np.zeros((24, 24, 16), dtype=np.uint16))
        group.create_dataset(
            "Metadata",
            data=_tutorial_showfolder_metadata(rotation_deg, stage=stage, fov_nm=fov_nm),
        )
        h.create_group("Data/SpectrumStream")


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
    data_dir = _download_widget_tutorial_folder(
        _GOLD_HAADF_VIEWER, _GOLD_HAADF_NAME, size=_GOLD_HAADF_SOURCE_SIZE
    )
    return _gold_haadf_2d_from_folder(data_dir, stride=stride, verbose=verbose)


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

    data_dir = _download_widget_tutorial_folder(
        _GOLD_HAADF_VIEWER, _GOLD_HAADF_NAME, size=_GOLD_HAADF_SOURCE_SIZE
    )
    return _gold_haadf_3d_from_folder(
        data_dir,
        n_frames=n_frames,
        stride=stride,
        crop_size=crop_size,
        verbose=verbose,
    )


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
    data_dir = _download_widget_tutorial_folder(
        _GOLD_4DSTEM_VIEWER, _GOLD_4DSTEM_NAME, size=_GOLD_4DSTEM_SOURCE_SIZE
    )
    return _gold_4dstem_from_folder(data_dir, scan_stride=scan_stride, verbose=verbose)
