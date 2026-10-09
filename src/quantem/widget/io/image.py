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

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from quantem.widget.adapters.core import make_dataset
from quantem.widget.io.hdf5_family import open_hdf5

_IMAGE_SUFFIXES = (".npy", ".emd", ".tif", ".tiff", ".png",
                   ".jpg", ".jpeg", ".bmp", ".gif", ".dm3", ".dm4")


class RgbImage:
    """True-color image from disk: display-ready ``(H, W, 3)`` for Show2D/Show3D.

    quantem core has no color-image dataset, so this is the one reader result
    that is not a quantem dataset. It carries the ``array``, ``name``,
    ``sampling`` and ``units`` a ``Dataset2d`` carries, so
    ``Show2D(io.read_image("color.png"))`` works without a conversion.
    Grayscale formats return ``Dataset2d``.
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
        """The pixels, so ``np.asarray(rgb_image)`` and ``to_numpy`` read it like an array."""
        return np.asarray(self.array, dtype=dtype)


def read_images(
    folder: str | Path,
    *,
    workers: int = 1,
    progress: bool = False,
):
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
    files = sorted(file for file in folder.iterdir()
                   if file.is_file() and file.suffix.lower() in _IMAGE_SUFFIXES
                   and not file.name.startswith("."))
    if not files:
        raise FileNotFoundError(f"No supported images in {folder}")
    if workers <= 1 or len(files) == 1:
        return [read_image(file) for file in _with_progress(
            files, progress, desc=f"Reading {len(files)} images", unit="image")]

    with ThreadPoolExecutor(max_workers=min(workers, len(files))) as pool:
        results = pool.map(read_image, files)
        return list(_with_progress(
            results, progress, desc=f"Reading {len(files)} images", total=len(files), unit="image"))


def read_image(path: str | Path):
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

    Grayscale results are a quantem core ``Dataset2d`` (on an Intel Mac or
    Windows on ARM, which cannot install quantem core, the widget's stand-in
    with the same ``array``, ``name``, ``sampling``, ``units`` and
    ``metadata``). Color results are :class:`RgbImage`. A multi-frame
    container is reduced to its first frame. A Velox EMD keeps its metadata in
    ``metadata`` and its scan rotation in ``scan_rotation_deg``.

    Examples
    --------
    >>> from quantem.widget import Show2D, io  # doctest: +SKIP
    >>> Show2D(io.read_image("figure_rgb.png"))  # true color, not gray  # doctest: +SKIP
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _wrap_image_array(_normalize_image_array(np.load(path)), name=path.stem)
    if suffix == ".emd":
        return _read_emd(path)
    if suffix == ".gif":
        from PIL import Image  # noqa: PLC0415
        with Image.open(path) as image:  # the first frame, as gray
            return make_dataset(np.asarray(image.convert("L"), dtype=np.float32), name=path.stem)
    if suffix in (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"):
        from PIL import Image  # noqa: PLC0415  (lazy: keep io import cheap)
        with Image.open(path) as image:
            arr = _pil_to_array(image)
        return _wrap_image_array(_normalize_image_array(arr), name=path.stem)
    if suffix in (".dm3", ".dm4"):
        from ncempy.io import dm  # noqa: PLC0415
        arr = np.asarray(dm.dmReader(str(path))["data"])
        return _wrap_image_array(_normalize_image_array(arr), name=path.stem)
    raise ValueError(
        f"read_image: unsupported extension {suffix!r} "
        "(use .npy, .emd, .tif/.tiff, .png, .jpg, .bmp, .gif, .dm3/.dm4)")


def _with_progress(iterable, show: bool, **tqdm_options):
    """``iterable`` behind a tqdm bar when ``show`` is set; tqdm is optional, so without it the bar is skipped."""
    if not show:
        return iterable
    try:
        from tqdm import tqdm  # noqa: PLC0415
    except ImportError:
        return iterable
    return tqdm(iterable, **tqdm_options)


def _is_rgb_array(arr: np.ndarray) -> bool:
    """True for a single color image with channel-last RGB(A)."""
    return arr.ndim == 3 and arr.shape[-1] in (3, 4)


def _pil_to_array(image) -> np.ndarray:
    """Decode a PIL image, preserving RGB instead of collapsing to gray."""
    mode = image.mode
    if mode in ("P", "PA"):
        # Palette files often encode true color; expand before asarray.
        image = image.convert("RGBA" if "A" in mode else "RGB")
    elif mode in ("CMYK", "YCbCr", "LAB", "HSV"):
        image = image.convert("RGB")
    elif mode == "1":
        image = image.convert("L")
    return np.asarray(image)


def _normalize_image_array(arr: np.ndarray) -> np.ndarray:
    """Keep RGB color; reduce multi-frame containers to the first frame.

    Channel-last RGB(A) is detected and kept, so an ``(H, W, 3)`` color image is
    never mistaken for a 3-frame stack.
    """
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if _is_rgb_array(arr):
        return arr[..., :3]
    if arr.ndim == 4 and arr.shape[-1] in (3, 4):
        # Multi-page color TIFF / multi-frame container → first color frame.
        return arr[0, ..., :3]
    if arr.ndim in (3, 4):
        # Grayscale multi-frame (N, H, W), or another 4D layout → first plane.
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
):
    """Wrap a normalized array as Dataset2d (gray) or RgbImage (color)."""
    if _is_rgb_array(arr):
        return RgbImage(arr, name=name, sampling=sampling, units=units)
    return make_dataset(arr, name=name, sampling=sampling, units=units)


def _read_emd(path: Path):
    """Read an EMD image: Velox HAADF layout if present, else the largest 2D dataset."""
    import h5py  # noqa: PLC0415
    with open_hdf5(path) as handle:
        if "Data/Image" in handle:                  # Velox HAADF
            group = next(iter(handle["Data/Image"]))
            arr = handle[f"Data/Image/{group}/Data"][...]
            metadata = _read_velox_metadata(handle, group)
            image = arr[:, :, 0] if arr.ndim == 3 else arr
            sampling, units = _velox_sampling(metadata)
            dataset = make_dataset(image, sampling=sampling, units=units, name=path.stem)
            dataset.metadata.update(metadata)
            dataset.scan_rotation_deg = _velox_scan_rotation_deg(metadata)
            return dataset
        candidates = []                              # non-Velox: largest >=2D dataset
        handle.visititems(lambda name, obj: candidates.append(obj)
                          if isinstance(obj, h5py.Dataset) and obj.ndim >= 2 else None)
        arr = max(candidates, key=lambda obj: obj.size)[()]
    dataset = make_dataset(_normalize_image_array(arr).astype(np.float32), name=path.stem)
    dataset.scan_rotation_deg = None                 # no Velox metadata to read the angle from
    return dataset


def _read_velox_metadata(handle, group) -> dict:
    """Decode the per-image Velox JSON metadata blob (null-padded uint8)."""
    raw = bytes(handle[f"Data/Image/{group}/Metadata"][:, 0].tobytes()).split(b"\x00", 1)[0]
    if not raw:
        return {}
    try:
        return json.loads(raw.decode("utf-8", "ignore"))
    except json.JSONDecodeError:
        return {}


def _velox_sampling(metadata: dict):
    """Pixel size (row, col) in nm + units from Velox ``BinaryResult.PixelSize`` (meters)."""
    pixel_size_m = metadata.get("BinaryResult", {}).get("PixelSize")
    if not pixel_size_m:
        return None, None
    height_nm = float(pixel_size_m["height"]) * 1e9
    width_nm = float(pixel_size_m["width"]) * 1e9
    return (height_nm, width_nm), ["nm", "nm"]


def _velox_scan_rotation_deg(metadata: dict):
    """Scan rotation in degrees from Velox ``Scan.ScanRotation`` (stored in radians).

    Returns ``None`` when the field is absent so callers can tell a genuine 0°
    scan from a file that simply carries no rotation metadata. Drift correction
    needs this angle to orient a 0°/90° pair; without it the merge is blind.
    """
    rotation_rad = metadata.get("Scan", {}).get("ScanRotation")
    if rotation_rad is None:
        return None
    return float(rotation_rad) * 180.0 / np.pi


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
):
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
        Shape ``(N, H, W)``, dtype float32 (the widget's stand-in where quantem
        core cannot install). Sampling defaults to pixels since a bare image
        folder carries no calibration. A folder of color frames returns an
        ``(N, H, W, 3)`` array: quantem core has no color-image dataset.
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
        return stack
    if first.ndim != 2:
        raise ValueError(f"Expected 2D frames, got shape {first.shape} from {files[0].name}")
    height, width = first.shape
    stack = np.empty((count, height, width), dtype=np.float32)
    stack[0] = first
    if count > 1:
        tasks = [(idx, files[idx], stack) for idx in range(1, count)]
        with ThreadPoolExecutor(max_workers=min(workers, count - 1)) as pool:
            results = pool.map(_read_frame_into, tasks)
            # draining the iterator waits for every frame and re-raises a failed decode
            list(_with_progress(results, progress, desc=f"Decoding {count} frames",
                                initial=1, total=count - 1, leave=False, unit="frame"))
    if files[0].suffix.lower() in _METADATA_EXTS:
        first_dataset = read_image(files[0])
        return make_dataset(
            stack,
            sampling=(1.0, float(first_dataset.sampling[-2]), float(first_dataset.sampling[-1])),
            units=("frame", str(first_dataset.units[-2]), str(first_dataset.units[-1])),
            name=path.name,
        )
    return make_dataset(stack, name=path.name)


def _collect_frames(path: Path, file_type: str | None, pattern: str | None) -> list[Path]:
    """Sorted list of frame files in a folder, ordered naturally for scrubbing."""
    if pattern:
        return sorted(path.glob(pattern), key=_natural_key)
    if file_type:
        ext = file_type.lower().lstrip(".")
        return sorted(path.glob(f"*.{ext}"), key=_natural_key)
    return sorted(
        (file for file in path.iterdir()
         if file.is_file() and file.suffix.lower() in _IMAGE_EXTS and not file.name.startswith(".")),
        key=_natural_key,
    )


def _natural_key(path: Path) -> list:
    """Natural sort key so ``frame_2`` precedes ``frame_10`` instead of after it."""
    return [int(part) if part.isdigit() else part.lower() for part in _NATURAL_RE.split(path.stem)]


def _read_frame(path: Path) -> np.ndarray:
    """Decode a single frame (gray ``(H,W)`` or RGB ``(H,W,3)``) as float32."""
    if path.suffix.lower() in _TIFF_EXTS:
        import tifffile
        arr = np.asarray(tifffile.imread(str(path)))
    elif path.suffix.lower() in _METADATA_EXTS:
        arr = np.asarray(read_image(path).array)
    else:
        from PIL import Image
        with Image.open(path) as image:
            arr = _pil_to_array(image)
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
    idx, path, stack = args
    stack[idx] = _read_frame(path)
