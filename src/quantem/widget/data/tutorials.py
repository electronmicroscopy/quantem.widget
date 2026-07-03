"""Built-in tutorial datasets for documentation and Colab notebooks."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
from quantem.core.datastructures import Dataset2d, Dataset3d, Dataset4dstem

download = None
snapshot_download = None
_SURVEY_HF_PATTERNS = (
    "survey/gold_haadf_session/*.emd",
)


def _download_dataset(name: str, *, verbose: bool = False) -> Path:
    global download
    if download is None:
        from quantem.widget.io import download as _download  # noqa: PLC0415

        download = _download
    return Path(download(name, verbose=verbose))


def _snapshot_download_dataset(**kwargs) -> Path:
    global snapshot_download
    if snapshot_download is None:
        from huggingface_hub import snapshot_download as _snapshot_download  # noqa: PLC0415

        snapshot_download = _snapshot_download
    return Path(snapshot_download(**kwargs))


def load_tutorial_survey_folder(*, verbose: bool = True, allow_fallback: bool = True) -> Path:
    """Download the real HAADF EMD folder used by the survey tutorials.

    Returns a folder containing a compact 26-file Velox ``.emd`` session from
    the public ``bobleesj/quantem-data`` Hugging Face dataset. The folder is
    small enough for documentation but still exercises the real
    ``survey(folder)`` path: Velox EMD image loading, metadata parsing, labels,
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

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        root = _snapshot_download_dataset(
            repo_id="bobleesj/quantem-data",
            repo_type="dataset",
            allow_patterns=list(_SURVEY_HF_PATTERNS),
        )
        folder = root / "survey" / "gold_haadf_session"
        if not folder.is_dir():
            raise FileNotFoundError(f"downloaded tutorial folder is missing: {folder}")
    except Exception:
        if not allow_fallback:
            raise
        folder = create_tutorial_survey_folder()

    if verbose:
        files = sorted(path.name for path in folder.glob("*.emd"))
        print(f"Tutorial survey folder: {folder}")
        print(f"Files: {len(files)} EMD")
    return folder


def create_tutorial_survey_folder(path: str | Path | None = None) -> Path:
    """Create a tiny Velox-like EMD folder for offline survey tests.

    The generated folder mimics a microscope session: two HAADF images from the
    same field of view, one matching EDS spectrum-image file, and one lower-mag
    overview image. Files are deliberately small so documentation notebooks can
    embed the survey widgets without making the site heavy.

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

    root = Path(path) if path is not None else Path(tempfile.gettempdir()) / "quantem-widget-survey-demo"
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


def _tutorial_survey_metadata(
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


def _tutorial_survey_image(shape: tuple[int, int] = (96, 96), *, seed: int = 0) -> np.ndarray:
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
        group.create_dataset("Data", data=_tutorial_survey_image(seed=seed))
        group.create_dataset(
            "Metadata",
            data=_tutorial_survey_metadata(rotation_deg, stage=stage, fov_nm=fov_nm),
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
            data=_tutorial_survey_metadata(rotation_deg, stage=stage, fov_nm=fov_nm),
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
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    data_dir = _download_dataset("gold_haadf_npy", verbose=False)
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
    data_dir = _download_dataset("gold_haadf_npy", verbose=False)
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
    data_dir = _download_dataset("gold_128_npy_bin8", verbose=False)
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
