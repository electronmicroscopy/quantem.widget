"""Read 2D survey images and folders of frames into quantem datasets.

``io.load`` is the iterative-4D-STEM loader (Arina/Dectris HDF5). A plain 2D
survey image - a HAADF saved by Velox as ``.emd``, or a ``.npy`` - needs a
different, tiny reader. :func:`read_image` is it, returning a :class:`Dataset2d`
that carries the pixel size + raw metadata, so ``Show2D(io.read_image(path))``
draws a real scale bar (in nm) with no extra arguments.

:func:`read_image_stack` is the folder analog: a directory of PNG/TIFF frames
(an in-situ time series, a tilt series, a reconstruction sweep) decoded in
parallel into a :class:`Dataset3d` for ``Show3D``. PIL/tifffile release the GIL
during the C-level decode, so a thread pool gives near-linear speedup until I/O
or memory bandwidth saturates (~8 workers optimal): ~90 fps vs ~24 fps serial on
2048x2048 16-bit PNGs.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from quantem.core.datastructures import Dataset3d
from quantem.widget.datastructures import Dataset2d


def read_image(path: str | Path) -> Dataset2d:
    """Return a :class:`Dataset2d` from a Velox ``.emd`` HAADF or a ``.npy`` file.

    Velox stores the image under ``Data/Image/<hash>/Data`` shaped (H, W, N) and
    a JSON metadata blob alongside; we take the first frame and pull the pixel
    size into ``sampling`` (nm) + keep the full metadata dict. ``.npy`` loads the
    array with no calibration (sampling defaults to pixels).
    """
    p = Path(path)
    if p.suffix == ".npy":
        return Dataset2d(np.load(p), name=p.stem)
    if p.suffix == ".emd":
        import h5py  # noqa: PLC0415  (lazy: keep importing quantem.widget.io cheap)
        with h5py.File(p, "r") as f:
            group = next(iter(f["Data/Image"]))  # one image signal per HAADF emd
            arr = f[f"Data/Image/{group}/Data"][...]
            meta = _read_velox_metadata(f, group)
        image = arr[:, :, 0] if arr.ndim == 3 else arr
        sampling, units = _velox_sampling(meta)
        return Dataset2d(image, sampling=sampling, units=units, name=p.stem, metadata=meta)
    raise ValueError(f"read_image: unsupported extension {p.suffix!r} (use .emd or .npy)")


def _read_velox_metadata(f, group) -> dict:
    """Decode the per-image Velox JSON metadata blob (null-padded uint8)."""
    raw = bytes(f[f"Data/Image/{group}/Metadata"][:, 0].tobytes()).split(b"\x00", 1)[0]
    if not raw:
        return {}
    try:
        return json.loads(raw.decode("utf-8", "ignore"))
    except json.JSONDecodeError:
        return {}


def _velox_sampling(meta: dict):
    """Pixel size (row, col) in nm + units from Velox ``BinaryResult.PixelSize`` (meters)."""
    px = meta.get("BinaryResult", {}).get("PixelSize")
    if not px:
        return None, None
    height_nm = float(px["height"]) * 1e9
    width_nm = float(px["width"]) * 1e9
    return (height_nm, width_nm), ["nm", "nm"]


# --- folder-of-frames stack reader (parallel decode) ----------------------

_IMAGE_EXTS = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}
_TIFF_EXTS = {".tif", ".tiff"}
_NATURAL_RE = re.compile(r"(\d+)")


def read_image_stack(
    path: str | Path,
    *,
    file_type: str | None = None,
    pattern: str | None = None,
    workers: int = 8,
    progress: bool = True,
) -> Dataset3d:
    """Decode a folder of image frames into a :class:`Dataset3d` in parallel.

    A directory of PNG/TIFF frames - an in-situ time series, a tilt series, a
    reconstruction sweep - is read with a thread pool into one contiguous
    ``(N, H, W)`` float32 array, then wrapped so ``Show3D(read_image_stack(dir))``
    scrubs the frames with no extra arguments. Frames are sorted naturally
    (``frame_2`` before ``frame_10``). Decode is threaded because PIL/tifffile
    release the GIL during the C decode, so N threads give near-linear speedup
    until I/O or memory bandwidth saturates; ~8 workers is optimal on most disks.

    Parameters
    ----------
    path : str or Path
        Folder containing the image frames.
    file_type : str, optional
        Extension filter (e.g. ``"png"``, ``"tif"``). When omitted every common
        image extension in the folder is taken.
    pattern : str, optional
        Glob within the folder (e.g. ``"frame_*.png"``); overrides ``file_type``.
    workers : int, default 8
        Thread count for parallel decompression.
    progress : bool, default True
        Show a tqdm bar while decoding.

    Returns
    -------
    Dataset3d
        Shape ``(N, H, W)``, dtype float32. Sampling defaults to pixels since a
        bare image folder carries no calibration.
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Not a directory: {path}")
    files = _collect_frames(path, file_type, pattern)
    if not files:
        raise FileNotFoundError(f"No image frames in {path}")
    first = _read_frame(files[0])
    if first.ndim != 2:
        raise ValueError(f"Expected 2D frames, got shape {first.shape} from {files[0].name}")
    height, width = first.shape
    count = len(files)
    stack = np.empty((count, height, width), dtype=np.float32)
    stack[0] = first
    if count > 1:
        tasks = [(idx, files[idx], stack) for idx in range(1, count)]
        with ThreadPoolExecutor(max_workers=min(workers, count - 1)) as pool:
            results = pool.map(_read_frame_into, tasks)
            if progress:
                try:
                    from tqdm import tqdm
                    list(tqdm(results, desc=f"Decoding {count} frames",
                              initial=1, total=count - 1, leave=False, unit="frame"))
                except ImportError:
                    for _ in results:
                        pass
            else:
                for _ in results:
                    pass
    return Dataset3d.from_array(stack, name=path.name)


def _collect_frames(path: Path, file_type: str | None, pattern: str | None) -> list[Path]:
    """Sorted list of frame files in a folder, ordered naturally for scrubbing."""
    if pattern:
        return sorted(path.glob(pattern), key=_natural_key)
    if file_type:
        ext = file_type.lower().lstrip(".")
        return sorted(path.glob(f"*.{ext}"), key=_natural_key)
    return sorted(
        (f for f in path.iterdir()
         if f.is_file() and f.suffix.lower() in _IMAGE_EXTS and not f.name.startswith(".")),
        key=_natural_key,
    )


def _natural_key(p: Path) -> list:
    """Natural sort key so ``frame_2`` precedes ``frame_10`` instead of after it."""
    return [int(s) if s.isdigit() else s.lower() for s in _NATURAL_RE.split(p.stem)]


def _read_frame(path: Path) -> np.ndarray:
    """Decode a single frame to a float32 array (tifffile for TIFF, PIL otherwise)."""
    if path.suffix.lower() in _TIFF_EXTS:
        import tifffile
        return tifffile.imread(str(path)).astype(np.float32)
    from PIL import Image
    with Image.open(path) as img:
        return np.array(img, dtype=np.float32)


def _read_frame_into(args: tuple[int, Path, np.ndarray]) -> None:
    """Decode one frame straight into its pre-allocated slot (no per-thread copy)."""
    idx, path, out = args
    out[idx] = _read_frame(path)
