"""Read 2D survey images and folders of frames into quantem datasets.

``io.load`` is the optimized 4D-STEM loader (Arina/Dectris HDF5). A plain 2D
survey image - a HAADF saved by Velox as ``.emd``, or a ``.npy`` - needs a
different, tiny reader. :func:`read_image` is it, returning a :class:`Dataset2d`
that carries the pixel size + raw metadata, so ``Show2D(io.read_image(path))``
draws a real scale bar (in nm) with no extra arguments.

:func:`read_images` is for a folder of independent survey images, including EMD,
PNG, TIFF, DM, and NPY files. It keeps per-image calibration and can use a thread
pool for large sessions.

:func:`read_image_stack` is the folder analog: a directory of PNG/TIFF frames
(an in-situ time series, a tilt series, a reconstruction sweep) decoded in
parallel into a :class:`Dataset3d` for ``Show3D``. PIL/tifffile release the GIL
during the C-level decode, so a thread pool gives near-linear speedup until I/O
or memory bandwidth saturates (~8 workers optimal): ~90 fps vs ~24 fps serial on
2048x2048 16-bit PNGs.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from quantem.core.datastructures import Dataset2d, Dataset3d


@dataclass
class _ImageDataset:
    """Dataset-like image container used when quantem.core cannot import."""

    array: np.ndarray
    name: str = ""
    sampling: tuple[float, ...] = (1.0, 1.0)
    units: tuple[str, ...] = ("pixels", "pixels")
    signal_units: str = "arb. units"

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.array.shape)

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype

    @property
    def ndim(self) -> int:
        return self.array.ndim

    def __array__(self, dtype: object | None = None) -> np.ndarray:
        return np.asarray(self.array, dtype=dtype)


def _is_quantem_core_circular_import(exc: BaseException) -> bool:
    msg = str(exc)
    return "partially initialized module 'quantem.core" in msg or (
        "cannot import name 'Dataset'" in msg and "quantem.core" in msg
    )


def _core_dataset_classes():
    """Return core dataset classes when a reader actually needs them."""
    try:
        from quantem.core.datastructures import Dataset2d, Dataset3d
    except Exception as exc:
        if _is_quantem_core_circular_import(exc):
            return None, None
        raise ImportError(
            "quantem.widget.io image readers require quantem.core dataset "
            "classes. Check that quantem.core imports cleanly in this Python "
            "environment."
        ) from exc
    return Dataset2d, Dataset3d


def _dataset2d_from_array(
    arr: np.ndarray,
    *,
    name: str = "",
    sampling: tuple[float, float] | None = None,
    units: tuple[str, str] | list[str] | None = None,
):
    Dataset2d, _ = _core_dataset_classes()
    sampling_tuple = (1.0, 1.0) if sampling is None else tuple(float(v) for v in sampling)
    units_tuple = ("pixels", "pixels") if units is None else tuple(str(v) for v in units)
    if Dataset2d is None:
        return _ImageDataset(
            np.asarray(arr),
            name=str(name),
            sampling=sampling_tuple,
            units=units_tuple,
        )
    return Dataset2d.from_array(arr, sampling=sampling_tuple, units=units_tuple, name=name)


def _dataset3d_from_array(
    arr: np.ndarray,
    *,
    name: str = "",
    sampling: tuple[float, float, float] | None = None,
    units: tuple[str, str, str] | list[str] | None = None,
):
    _, Dataset3d = _core_dataset_classes()
    sampling_tuple = (1.0, 1.0, 1.0) if sampling is None else tuple(float(v) for v in sampling)
    units_tuple = ("frame", "pixels", "pixels") if units is None else tuple(str(v) for v in units)
    if Dataset3d is None:
        return _ImageDataset(
            np.asarray(arr),
            name=str(name),
            sampling=sampling_tuple,
            units=units_tuple,
        )
    return Dataset3d.from_array(arr, sampling=sampling_tuple, units=units_tuple, name=name)


_IMAGE_SUFFIXES = (".npy", ".emd", ".tif", ".tiff", ".png",
                   ".jpg", ".jpeg", ".bmp", ".gif", ".dm3", ".dm4")


def read_images(
    folder: str | Path,
    *,
    workers: int = 1,
    progress: bool = False,
) -> list[Dataset2d]:
    """Read every image in a folder into a list of :class:`Dataset2d`.

    The folder analog of :func:`read_image` for a *mixed* set of survey images -
    different formats and different sizes that cannot stack into one cube (use
    :func:`read_image_stack` for a folder of same-size frames). Files are sorted
    by name; every supported extension is read, anything else is skipped. Lets a
    gallery be one line: ``Show2D([d.array for d in io.read_images(folder)])``.

    Parameters
    ----------
    folder : str or Path
        Folder containing supported 2D image files.
    workers : int, default 1
        Thread count for reading many files. Use ``workers=8`` for large folders
        of independent EMD/TIFF/PNG survey images.
    progress : bool, default False
        Show a tqdm bar while reading.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a directory: {folder}")
    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
                   and not p.name.startswith("."))
    if not files:
        raise FileNotFoundError(f"No supported images in {folder}")
    if workers <= 1 or len(files) == 1:
        iterable = files
        if progress:
            try:
                from tqdm import tqdm  # noqa: PLC0415
                iterable = tqdm(files, desc=f"Reading {len(files)} images", unit="image")
            except ImportError:
                pass
        return [read_image(p) for p in iterable]

    with ThreadPoolExecutor(max_workers=min(workers, len(files))) as pool:
        results = pool.map(read_image, files)
        if progress:
            try:
                from tqdm import tqdm  # noqa: PLC0415
                results = tqdm(results, desc=f"Reading {len(files)} images",
                               total=len(files), unit="image")
            except ImportError:
                pass
        return list(results)


def read_image(path: str | Path) -> Dataset2d | RgbImage:
    """Return a single image from disk (grayscale or true-color RGB).

    One reader for every survey-image format the lab produces:

    - ``.npy`` - raw array, no calibration.
    - ``.emd`` - Velox HAADF (image under ``Data/Image/<hash>/Data`` with a JSON
      metadata blob carrying the pixel size); falls back to the largest 2D
      dataset for non-Velox EMD layouts (e.g. a ``data/drift/data`` series).
    - ``.tif`` / ``.tiff`` / ``.png`` / ``.jpg`` / ``.bmp`` / ``.gif`` - via Pillow.
      **Color PNG/JPEG/TIFF keep RGB** (``RgbImage`` with shape ``(H, W, 3)``);
      they are not converted to a single gray channel. Pass the result to
      ``Show2D`` or ``Show3D`` to display true color.
    - ``.dm3`` / ``.dm4`` - Gatan, via ncempy.

    Grayscale results are :class:`Dataset2d`. Color results are :class:`RgbImage`
    (duck-types the ``.array`` / ``.name`` / ``.sampling`` surface Show2D already
    unwraps). A multi-frame container is reduced to its first frame.

    Examples
    --------
    >>> from quantem.widget import Show2D, io  # doctest: +SKIP
    >>> Show2D(io.read_image("figure_rgb.png"))  # true color, not gray  # doctest: +SKIP
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".npy":
        return _wrap_image_array(_normalize_image_array(np.load(p)), name=p.stem)
    if ext == ".emd":
        return _read_emd(p)
    if ext == ".gif":
        ds = read_gif(p)
        return _dataset2d_from_array(ds.array[0], name=p.stem)
    if ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"):
        from PIL import Image  # noqa: PLC0415  (lazy: keep io import cheap)
        with Image.open(p) as img:
            arr = _pil_to_array(img)
        return _wrap_image_array(_normalize_image_array(arr), name=p.stem)
    if ext in (".dm3", ".dm4"):
        from ncempy.io import dm  # noqa: PLC0415
        arr = np.asarray(dm.dmReader(str(p))["data"])
        return _wrap_image_array(_normalize_image_array(arr), name=p.stem)
    raise ValueError(
        f"read_image: unsupported extension {ext!r} "
        "(use .npy, .emd, .tif/.tiff, .png, .jpg, .bmp, .gif, .dm3/.dm4)")


class RgbImage:
    """True-color image from disk: display-ready ``(H, W, 3)`` for Show2D/Show3D.

    Duck-types the ``Dataset2d`` surface (``.array``, ``.name``, ``.sampling``,
    ``.units``) so ``Show2D(io.read_image("color.png"))`` works without an
    extra conversion. Grayscale EM formats still return ``Dataset2d``.
    """

    def __init__(
        self,
        array: np.ndarray,
        *,
        name: str = "",
        sampling: tuple[float, float] = (1.0, 1.0),
        units: tuple[str, str] = ("pixels", "pixels"),
    ) -> None:
        arr = np.asarray(array)
        if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
            raise ValueError(
                f"RgbImage expects shape (H, W, 3) or (H, W, 4); got {arr.shape}"
            )
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        self.array = np.ascontiguousarray(arr)
        self.name = str(name)
        self.sampling = (float(sampling[0]), float(sampling[1]))
        self.units = (str(units[0]), str(units[1]))

    def __array__(self, dtype=None):
        return np.asarray(self.array, dtype=dtype)


def _is_rgb_array(arr: np.ndarray) -> bool:
    """True for a single color image with channel-last RGB(A)."""
    return arr.ndim == 3 and arr.shape[-1] in (3, 4)


def _is_rgb_stack(arr: np.ndarray) -> bool:
    """True for a stack of color frames ``(N, H, W, 3/4)``."""
    return arr.ndim == 4 and arr.shape[-1] in (3, 4)


def _pil_to_array(img) -> np.ndarray:
    """Decode a PIL image, preserving RGB instead of collapsing to gray."""
    mode = img.mode
    if mode in ("P", "PA"):
        # Palette files often encode true color; expand before asarray.
        img = img.convert("RGBA" if "A" in mode else "RGB")
    elif mode in ("CMYK", "YCbCr", "LAB", "HSV"):
        img = img.convert("RGB")
    elif mode == "1":
        img = img.convert("L")
    return np.asarray(img)


def _normalize_image_array(arr: np.ndarray) -> np.ndarray:
    """Keep RGB color; reduce multi-frame containers to the first frame.

    Historical bug: ``_first_frame`` treated ``(H, W, 3)`` as a 3-frame stack and
    returned the first *row* as ``(W, 3)``, destroying color PNGs. Channel-last
    RGB(A) is now detected and preserved.
    """
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if _is_rgb_array(arr):
        return arr[..., :3]
    if _is_rgb_stack(arr):
        # Multi-page color TIFF / multi-frame container → first color frame.
        return arr[0, ..., :3]
    if arr.ndim == 3:
        # Grayscale multi-frame (N, H, W) → first frame.
        return arr[0]
    if arr.ndim == 4:
        # Unusual (N, H, W, C) already handled; other 4D → first plane.
        return arr[0]
    raise ValueError(
        f"Unsupported image array shape {arr.shape}; expected (H, W), "
        "(H, W, 3/4), (N, H, W), or (N, H, W, 3/4)."
    )


def _wrap_image_array(
    arr: np.ndarray,
    *,
    name: str = "",
    sampling: tuple[float, float] = (1.0, 1.0),
    units: tuple[str, str] = ("pixels", "pixels"),
) -> Dataset2d | RgbImage:
    """Wrap a normalized array as Dataset2d (gray) or RgbImage (color)."""
    if _is_rgb_array(arr):
        return RgbImage(arr, name=name, sampling=sampling, units=units)
    return _dataset2d_from_array(arr, name=name, sampling=sampling, units=units)


def read_gif(path: str | Path) -> Dataset3d:
    """Read a static or animated GIF as a grayscale :class:`Dataset3d`.

    GIF is useful as a lightweight interchange format for time-series previews
    and denoising comparisons. The reader always returns an ``(N, H, W)``
    float32 stack so the same object works with ``Show3D`` for playback and
    ``Show2D`` for a flat contact sheet. Static GIFs become a one-frame stack.

    Parameters
    ----------
    path : str or Path
        GIF file to read.

    Returns
    -------
    Dataset3d
        Grayscale frame stack named from the file stem.
    """
    _, Dataset3d = _core_dataset_classes()
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"GIF file not found: {p}")
    if p.suffix.lower() != ".gif":
        raise ValueError(f"read_gif expects a .gif file, got {p.suffix!r}")

    from PIL import Image, ImageSequence  # noqa: PLC0415

    frames: list[np.ndarray] = []
    with Image.open(p) as img:
        for frame in ImageSequence.Iterator(img):
            frames.append(np.asarray(frame.convert("L"), dtype=np.float32))
    if not frames:
        raise ValueError(f"GIF contains no frames: {p}")

    shape = frames[0].shape
    for idx, frame in enumerate(frames[1:], 1):
        if frame.shape != shape:
            raise ValueError(
                f"GIF frame {idx} has shape {frame.shape}, expected {shape}. "
                "Use a GIF with same-size frames."
            )
    return _dataset3d_from_array(np.stack(frames, axis=0), name=p.stem)


def _first_frame(arr: np.ndarray) -> np.ndarray:
    """Reduce multi-frame arrays to the first frame; preserve RGB(A).

    Prefer :func:`_normalize_image_array` for new code. Kept for callers that
    already import ``_first_frame``.
    """
    return _normalize_image_array(arr)


def _read_emd(p: Path) -> Dataset2d:
    """Read an EMD image: Velox HAADF layout if present, else the largest 2D dataset."""
    import h5py  # noqa: PLC0415
    with h5py.File(p, "r") as f:
        if "Data/Image" in f:                       # Velox HAADF
            group = next(iter(f["Data/Image"]))
            arr = f[f"Data/Image/{group}/Data"][...]
            meta = _read_velox_metadata(f, group)
            image = arr[:, :, 0] if arr.ndim == 3 else arr
            sampling, units = _velox_sampling(meta)
            ds = _dataset2d_from_array(image, sampling=sampling, units=units, name=p.stem)
            ds._metadata = meta
            ds.scan_rotation_deg = _velox_scan_rotation_deg(meta)
            return ds
        biggest = []                                 # non-Velox: largest >=2D dataset
        f.visititems(lambda name, obj: biggest.append(obj)
                     if isinstance(obj, h5py.Dataset) and obj.ndim >= 2 else None)
        arr = max(biggest, key=lambda obj: obj.size)[()]
    ds = _dataset2d_from_array(_first_frame(arr).astype(np.float32), name=p.stem)
    ds.scan_rotation_deg = None                       # no Velox metadata to read the angle from
    return ds


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


def _velox_scan_rotation_deg(meta: dict):
    """Scan rotation in degrees from Velox ``Scan.ScanRotation`` (stored in radians).

    Returns ``None`` when the field is absent so callers can tell a genuine 0°
    scan from a file that simply carries no rotation metadata. Drift correction
    needs this angle to orient a 0°/90° pair; without it the merge is blind.
    """
    sr = meta.get("Scan", {}).get("ScanRotation")
    if sr is None:
        return None
    return float(sr) * 180.0 / np.pi


# --- folder-of-frames stack reader (parallel decode) ----------------------

_IMAGE_EXTS = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg", ".emd", ".dm3", ".dm4", ".npy"}
_TIFF_EXTS = {".tif", ".tiff"}
# Formats PIL cannot open; routed through read_image (which also carries their
# calibration metadata).
_METADATA_EXTS = {".emd", ".dm3", ".dm4", ".npy"}
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

    A directory of PNG/TIFF/EMD/DM/NPY frames - an in-situ time series, a tilt
    series, a reconstruction sweep - is read with a thread pool into one
    contiguous ``(N, H, W)`` float32 array, then wrapped so
    ``Show3D(read_image_stack(dir))`` scrubs the frames with no extra
    arguments. Frames are sorted naturally (``frame_2`` before ``frame_10``).
    Decode is threaded because PIL/tifffile release the GIL during the C
    decode, so N threads give near-linear speedup until I/O or memory
    bandwidth saturates; ~8 workers is optimal on most disks. When the first
    frame is a calibrated format (EMD/DM), its pixel sampling and units carry
    onto the stack's spatial axes so ``Show3D`` draws a physical scale bar.

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
    count = len(files)
    if _is_rgb_array(first):
        # Color frame stack for Show3D true-color scrubbing: (N, H, W, 3).
        height, width = int(first.shape[0]), int(first.shape[1])
        stack = np.empty((count, height, width, 3), dtype=np.float32)
        stack[0] = first[..., :3]
        if count > 1:
            for idx in range(1, count):
                frame = _read_frame(files[idx])
                if not _is_rgb_array(frame):
                    raise ValueError(
                        f"Mixed gray/RGB frames in {path}: {files[0].name} is RGB "
                        f"but {files[idx].name} has shape {frame.shape}."
                    )
                if frame.shape[:2] != (height, width):
                    raise ValueError(
                        f"Frame {files[idx].name} spatial shape {frame.shape[:2]} "
                        f"!= {(height, width)} from {files[0].name}."
                    )
                stack[idx] = frame[..., :3]
        # Dataset3d is gray-only; return a bare RGB stack. Show3D accepts it.
        return stack  # type: ignore[return-value]
    if first.ndim != 2:
        raise ValueError(f"Expected 2D frames, got shape {first.shape} from {files[0].name}")
    height, width = first.shape
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
    if files[0].suffix.lower() in _METADATA_EXTS:
        first_dataset = read_image(files[0])
        sampling = getattr(first_dataset, "sampling", None)
        units = getattr(first_dataset, "units", None)
        if sampling is not None and units is not None and len(sampling) >= 2:
            return _dataset3d_from_array(
                stack,
                sampling=(1.0, float(sampling[-2]), float(sampling[-1])),
                units=("frame", str(units[-2]), str(units[-1])),
                name=path.name,
            )
    return _dataset3d_from_array(stack, name=path.name)


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
    """Decode a single frame (gray ``(H,W)`` or RGB ``(H,W,3)``) as float32."""
    if path.suffix.lower() in _TIFF_EXTS:
        import tifffile
        arr = np.asarray(tifffile.imread(str(path)))
    elif path.suffix.lower() in _METADATA_EXTS:
        arr = np.asarray(read_image(path).array)
    else:
        from PIL import Image
        with Image.open(path) as img:
            arr = _pil_to_array(img)
    arr = _normalize_image_array(arr)
    if _is_rgb_array(arr):
        # uint8 color PNGs → unit-range float; float color stays clipped later in Show*.
        out = arr.astype(np.float32, copy=False)
        if arr.dtype == np.uint8:
            out = out / 255.0
        return out
    return arr.astype(np.float32, copy=False)


def _read_frame_into(args: tuple[int, Path, np.ndarray]) -> None:
    """Decode one frame straight into its pre-allocated slot (no per-thread copy)."""
    idx, path, out = args
    out[idx] = _read_frame(path)
