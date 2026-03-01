"""
io: Unified file loading for quantem.widget.

Reads common electron microscopy formats (PNG, TIFF, EMD, DM3/DM4, MRC, SER,
NPY/NPZ) into a lightweight IOResult that any widget accepts via duck typing.
GPU-accelerated loaders (e.g. ``IO.arina()``) auto-detect the best backend.
"""

import importlib
import os
import pathlib
from dataclasses import dataclass, field

import numpy as np
from tqdm.auto import tqdm

try:
    import h5py  # type: ignore

    _HAS_H5PY = True
except Exception:
    h5py = None  # type: ignore[assignment]
    _HAS_H5PY = False

try:
    import hdf5plugin  # type: ignore  # noqa: F401 — registers HDF5 filters (bitshuffle, lz4, zstd, …)
except Exception:
    pass


def _format_memory(nbytes: int) -> str:
    if nbytes >= 1 << 30:
        return f"{nbytes / (1 << 30):.1f} GB"
    if nbytes >= 1 << 20:
        return f"{nbytes / (1 << 20):.0f} MB"
    if nbytes >= 1 << 10:
        return f"{nbytes / (1 << 10):.0f} KB"
    return f"{nbytes} B"


@dataclass
class IOResult:
    """Result of reading a file or folder.

    Attributes
    ----------
    data : np.ndarray
        Float32 array, 2D (H, W) or 3D (N, H, W).
    pixel_size : float or None
        Pixel size in angstroms, extracted from file metadata.
    units : str or None
        Unit string (e.g. "Å", "nm").
    title : str
        Title derived from filename stem.
    labels : list of str
        One label per frame for stacks.
    metadata : dict
        Raw metadata tree from file.
    frame_metadata : list of dict
        Per-frame metadata for 5D stacks. One dict per frame, extracted
        from each source file (e.g. defocus, tilt angle, timestamp).
    """

    data: np.ndarray
    pixel_size: float | None = None
    units: str | None = None
    title: str = ""
    labels: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    frame_metadata: list[dict] = field(default_factory=list)

    @property
    def array(self):
        return self.data

    @property
    def name(self):
        return self.title

    def __getattr__(self, name):
        return getattr(self.data, name)

    def __repr__(self):
        shape = "x".join(str(s) for s in self.data.shape)
        mem = _format_memory(self.data.nbytes)
        parts = [f"IOResult({shape} {self.data.dtype}, {mem}"]
        if self.title:
            parts[0] += f", title={self.title!r}"
        if self.pixel_size is not None:
            parts[0] += f", pixel_size={self.pixel_size:.4f}"
        if self.units:
            parts[0] += f" {self.units}"
        if self.labels:
            parts[0] += f", {len(self.labels)} labels"
        parts[0] += ")"
        return parts[0]

    def __str__(self):
        shape = " x ".join(str(s) for s in self.data.shape)
        mem = _format_memory(self.data.nbytes)
        lines = [f"IOResult"]
        lines.append(f"  shape:      {shape}")
        lines.append(f"  dtype:      {self.data.dtype}")
        lines.append(f"  memory:     {mem}")
        if self.title:
            lines.append(f"  title:      {self.title}")
        if self.pixel_size is not None:
            unit = f" {self.units}" if self.units else ""
            lines.append(f"  pixel_size: {self.pixel_size:.4f}{unit}")
        if self.labels:
            if len(self.labels) <= 4:
                lines.append(f"  labels:     {self.labels}")
            else:
                shown = ", ".join(repr(l) for l in self.labels[:3])
                lines.append(f"  labels:     [{shown}, ...] ({len(self.labels)} total)")
        if self.frame_metadata:
            n_keys = len(self.frame_metadata[0]) if self.frame_metadata[0] else 0
            lines.append(f"  frames:     {len(self.frame_metadata)} with metadata ({n_keys} fields each)")
        if self.metadata:
            lines.append(f"  metadata:   {len(self.metadata)} fields")
        return "\n".join(lines)

    def describe(self, keys: list[str] | None = None, diff: bool = True):
        """Print a per-frame metadata table.

        Parameters
        ----------
        keys : list of str, optional
            Metadata keys to show (short names, matched against the end
            of full HDF5 paths). If None, shows all keys (filtered by
            ``diff`` when multiple frames exist).
        diff : bool, default True
            Only show columns where values differ across frames. Set
            ``diff=False`` to show all columns including constants.

        Examples
        --------
        >>> result = IO.arina_folder("/data/session/", det_bin=8)
        >>> result.describe()                    # diff columns only
        >>> result.describe(diff=False)          # all columns
        >>> result.describe(keys=["count_time", "photon_energy"])
        """
        # Single-file metadata (no frame_metadata but has metadata)
        if not self.frame_metadata and self.metadata:
            for k, v in self.metadata.items():
                short = k.rsplit("/", 1)[-1] if "/" in k else k
                if isinstance(v, float):
                    v = f"{v:.6g}"
                print(f"  {short}: {v}")
            return
        if not self.frame_metadata:
            print("No per-frame metadata available.")
            return
        # Resolve keys → full HDF5 paths
        all_paths = list(self.frame_metadata[0].keys()) if self.frame_metadata[0] else []
        if keys is None:
            # Use all keys — let diff filtering select the interesting ones
            resolved = [(fp.rsplit("/", 1)[-1], fp) for fp in all_paths]
        else:
            resolved: list[tuple[str, str]] = []  # (short_name, full_path)
            for k in keys:
                for fp in all_paths:
                    if fp.endswith("/" + k) or fp == k:
                        short = k if "/" not in k else k.rsplit("/", 1)[-1]
                        resolved.append((short, fp))
                        break
        if not resolved:
            print("No matching metadata keys found.")
            print(f"Available: {[p.rsplit('/', 1)[-1] for p in all_paths[:20]]}")
            return
        # Filter to diff-only columns when requested
        if diff and len(self.frame_metadata) > 1:
            diff_resolved = []
            for short, fp in resolved:
                values = [fm.get(fp) for fm in self.frame_metadata]
                if len(set(str(v) for v in values)) > 1:
                    diff_resolved.append((short, fp))
            if not diff_resolved:
                print("All selected metadata fields are identical across frames.")
                print("Use describe(diff=False) to see all fields.")
                return
            resolved = diff_resolved
        # Format a value for display
        def _fmt(v):
            if isinstance(v, float):
                return f"{v:.6g}"
            return str(v) if v != "" else ""
        # Vertical layout when many columns (>10) — one key per line, frames as columns
        if len(resolved) > 10:
            labels = [self.labels[i] if i < len(self.labels) else str(i) for i in range(len(self.frame_metadata))]
            key_w = max(len(short) for short, _ in resolved)
            col_w = max((len(l) for l in labels), default=6)
            # Widen columns to fit values
            for short, fp in resolved:
                for fm in self.frame_metadata:
                    col_w = max(col_w, len(_fmt(fm.get(fp, ""))))
            # Header row: key + frame labels
            header = "  " + "".ljust(key_w) + "  "
            header += "  ".join(l.ljust(col_w) for l in labels)
            print(header)
            print("  " + "─" * (len(header) - 2))
            # One row per key
            for short, fp in resolved:
                row = "  " + short.ljust(key_w) + "  "
                row += "  ".join(_fmt(self.frame_metadata[i].get(fp, "")).ljust(col_w) for i in range(len(self.frame_metadata)))
                print(row)
            return
        # Horizontal table (few columns): frames as rows, keys as columns
        col_widths = [max(len(short), 6) for short, _ in resolved]
        label_w = max((len(l) for l in self.labels), default=5) if self.labels else 5
        # Header
        header = "  " + "label".ljust(label_w) + "  "
        header += "  ".join(short.ljust(w) for (short, _), w in zip(resolved, col_widths))
        print(header)
        print("  " + "─" * (len(header) - 2))
        # Rows
        for i, fm in enumerate(self.frame_metadata):
            label = self.labels[i] if i < len(self.labels) else str(i)
            row = "  " + label.ljust(label_w) + "  "
            vals = []
            for (short, fp), w in zip(resolved, col_widths):
                vals.append(_fmt(fm.get(fp, "")).ljust(w))
            row += "  ".join(vals)
            print(row)


# =========================================================================
# Native loaders (no rsciio dependency)
# =========================================================================


def _load_image_2d(path: pathlib.Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as img:
        return np.asarray(img.convert("F"), dtype=np.float32)


def _load_tiff_stack(path: pathlib.Path) -> tuple[np.ndarray, list[str]]:
    from PIL import Image, ImageSequence

    frames: list[np.ndarray] = []
    labels: list[str] = []
    with Image.open(path) as img:
        for i, page in enumerate(ImageSequence.Iterator(img)):
            frame = np.asarray(page.convert("F"), dtype=np.float32)
            frames.append(frame)
            labels.append(f"{path.stem}[{i}]")
    if not frames:
        raise ValueError(f"No readable frames found in TIFF file: {path}")
    shape0 = frames[0].shape
    for i, frame in enumerate(frames[1:], start=1):
        if frame.shape != shape0:
            raise ValueError(
                f"Inconsistent TIFF frame shapes in {path}: "
                f"frame 0={shape0}, frame {i}={frame.shape}"
            )
    return np.stack(frames, axis=0).astype(np.float32), labels


def _find_best_h5_dataset(h5f, *, prefer_ndim: int = 2):
    candidates: list[tuple[int, str, object]] = []

    def _walk(group, prefix: str = ""):
        for key, item in group.items():
            item_path = f"{prefix}/{key}" if prefix else key
            if hasattr(item, "shape") and hasattr(item, "dtype"):
                try:
                    ndim = int(item.ndim)
                    size = int(item.size)
                    shape = tuple(int(s) for s in item.shape)
                    dtype_kind = str(item.dtype.kind)
                except Exception:
                    continue
                if size <= 0 or ndim < 2 or dtype_kind not in {"i", "u", "f", "c"}:
                    continue
                # Effective ndim ignores trailing size-1 dims so that
                # e.g. (2048, 2048, 1) is treated as 2D and (60000, 1) as 1D.
                eff_ndim = len([s for s in shape if s > 1])
                score = size
                if eff_ndim == prefer_ndim:
                    score += 10**15
                elif eff_ndim == prefer_ndim + 1:
                    score += 9 * 10**14
                elif eff_ndim == prefer_ndim - 1:
                    score += 9 * 10**14
                elif eff_ndim == 4:
                    score += 8 * 10**14
                elif eff_ndim == 5:
                    score += 7 * 10**14
                # Penalize degenerate shapes (e.g. (60000, 1)) — these are
                # metadata or lookup tables, not images.
                if eff_ndim < 2:
                    score -= 2 * 10**15
                lower_path = item_path.lower()
                for token in ["image", "stack", "series", "frame", "signal", "data"]:
                    if token in lower_path:
                        score += 10**12
                for token in ["preview", "thumb", "mask", "meta", "label", "calib"]:
                    if token in lower_path:
                        score -= 10**12
                candidates.append((score, item_path, item))
            elif hasattr(item, "items"):
                _walk(item, item_path)

    _walk(h5f)
    if not candidates:
        return "", None
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, ds_path, ds = candidates[0]
    return ds_path, ds


def _load_emd(
    path: pathlib.Path,
    *,
    dataset_path: str | None = None,
    prefer_ndim: int = 2,
) -> np.ndarray:
    if not _HAS_H5PY:
        raise RuntimeError(
            "h5py is required to read .emd files. Install with: pip install h5py"
        )
    with h5py.File(path, "r") as h5f:
        if dataset_path is not None:
            ds_path = str(dataset_path).strip()
            if not ds_path:
                raise ValueError("dataset_path must be a non-empty string when provided.")
            if ds_path not in h5f and ds_path.startswith("/") and ds_path[1:] in h5f:
                ds_path = ds_path[1:]
            if ds_path not in h5f:
                raise ValueError(f"dataset_path '{dataset_path}' not found in EMD file: {path}")
            ds = h5f[ds_path]
        else:
            ds_path, ds = _find_best_h5_dataset(h5f, prefer_ndim=prefer_ndim)
        if ds is None:
            raise ValueError(f"No array-like dataset found in EMD file: {path}")
        if not hasattr(ds, "shape") or not hasattr(ds, "dtype"):
            raise ValueError(f"dataset_path '{ds_path}' is not an array dataset in EMD file: {path}")
        arr = np.asarray(ds)
    return arr


def _load_npy(path: pathlib.Path) -> np.ndarray:
    return np.load(path)


def _load_npz(path: pathlib.Path) -> np.ndarray:
    npz = np.load(path)
    keys = list(npz.keys())
    if not keys:
        raise ValueError(f"No arrays found in .npz file: {path}")
    return npz[keys[0]]


# =========================================================================
# rsciio loader
# =========================================================================


def _rsciio_ext_map() -> dict[str, str]:
    """Build extension → rsciio API module path mapping lazily."""
    try:
        from rsciio import IO_PLUGINS  # type: ignore
    except ImportError:
        return {}
    mapping: dict[str, str] = {}
    for plugin in IO_PLUGINS:
        api = plugin.get("api", "")
        if not api:
            continue
        for ext in plugin.get("file_extensions", []):
            ext_lower = ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            mapping[ext_lower] = api
    return mapping


def _load_rsciio(path: pathlib.Path) -> IOResult:
    """Load any file via rsciio and extract metadata."""
    try:
        from rsciio import IO_PLUGINS  # type: ignore  # noqa: F401
        import rsciio  # type: ignore
    except ImportError:
        raise ImportError(
            f"rsciio is required to read '{path.suffix}' files. "
            "Install with: pip install rosettasciio"
        )
    ext = path.suffix.lower()
    ext_map = _rsciio_ext_map()
    api_path = ext_map.get(ext)
    if api_path is None:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported by rsciio: {sorted(ext_map.keys())}"
        )
    try:
        reader_mod = importlib.import_module(api_path)
    except ImportError:
        raise ValueError(f"rsciio module '{api_path}' not found for extension {ext}")
    results = reader_mod.file_reader(str(path))
    if not results:
        raise ValueError(f"rsciio returned no data for: {path}")
    entry = results[0]
    arr = np.asarray(entry["data"])
    arr = _coerce_to_float32(arr)
    pixel_size = None
    units = None
    raw_metadata = entry.get("metadata", {})
    axes = entry.get("axes", [])
    if len(axes) >= 2:
        spatial_axes = axes[-2:]
        scale = spatial_axes[-1].get("scale", None)
        unit = spatial_axes[-1].get("units", None)
        if scale is not None and scale > 0:
            pixel_size = float(scale)
            if unit:
                units = str(unit)
                if units.lower() in ("nm", "nanometer", "nanometers"):
                    pixel_size = pixel_size * 10  # nm → Å
                    units = "Å"
                elif units.lower() in ("å", "angstrom", "angstroms", "a"):
                    units = "Å"
    title_from_meta = ""
    general = raw_metadata.get("General", {})
    if general.get("title"):
        title_from_meta = str(general["title"])
    title = title_from_meta or path.stem
    labels = []
    if arr.ndim == 3:
        labels = [f"{path.stem}[{i}]" for i in range(arr.shape[0])]
    return IOResult(
        data=arr,
        pixel_size=pixel_size,
        units=units,
        title=title,
        labels=labels,
        metadata=raw_metadata,
    )


# =========================================================================
# Helpers
# =========================================================================


def _coerce_to_float32(arr: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    return np.asarray(arr, dtype=np.float32)


def _coerce_to_stack_3d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim < 2:
        raise ValueError(f"Expected at least 2D image data, got {arr.ndim}D")
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim == 3:
        return arr
    return arr.reshape((-1, arr.shape[-2], arr.shape[-1]))


# =========================================================================
# Native format extensions
# =========================================================================

_NATIVE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".emd", ".h5", ".hdf5", ".npy", ".npz"}

_FOLDER_ALIASES = {"tif": "tiff", "hdf5": "h5"}
_FOLDER_EXT_MAP = {
    "png": {".png"},
    "tiff": {".tif", ".tiff"},
    "emd": {".emd"},
    "h5": {".h5", ".hdf5"},
    "dm3": {".dm3"},
    "dm4": {".dm4"},
    "mrc": {".mrc"},
    "npy": {".npy"},
    "npz": {".npz"},
}


def _all_supported_exts() -> set[str]:
    """All extensions we can read (native + rsciio), with leading dot."""
    exts = set(_NATIVE_EXTS)
    for ext in _rsciio_ext_map():
        exts.add(ext if ext.startswith(".") else f".{ext}")
    return exts


def _collect_files(
    folder: pathlib.Path,
    *,
    exts: set[str],
    recursive: bool,
) -> list[pathlib.Path]:
    if recursive:
        files = sorted(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in exts
        )
    else:
        files = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        )
    return files


def _detect_folder_type(files: list[pathlib.Path]) -> str | None:
    """Auto-detect a single file type from a list of files.

    Returns the normalized type key (e.g. "dm4", "png", "tiff") or None if
    the folder contains mixed types.
    """
    ext_groups: dict[str, int] = {}
    # Reverse map: extension → type key
    ext_to_key: dict[str, str] = {}
    for key, ext_set in _FOLDER_EXT_MAP.items():
        for ext in ext_set:
            ext_to_key[ext] = key
    # Also add native image exts that aren't in _FOLDER_EXT_MAP
    for ext in (".jpg", ".jpeg", ".bmp"):
        ext_to_key[ext] = "png"  # treat as image type

    for f in files:
        ext = f.suffix.lower()
        key = ext_to_key.get(ext, ext.lstrip("."))
        ext_groups[key] = ext_groups.get(key, 0) + 1

    if len(ext_groups) == 1:
        return next(iter(ext_groups))
    return None


def _apply_bin_factor(result: IOResult, bin_factor: int) -> IOResult:
    from quantem.widget.array_utils import bin2d
    binned = bin2d(result.data, factor=bin_factor)
    pixel_size = result.pixel_size * bin_factor if result.pixel_size else None
    return IOResult(
        data=binned,
        pixel_size=pixel_size,
        units=result.units,
        title=result.title,
        labels=result.labels,
        metadata=result.metadata,
        frame_metadata=result.frame_metadata,
    )


def _format_shape_summary(files_by_shape: dict[tuple, list[str]]) -> str:
    """Format a shape→filenames mapping into a human-readable summary."""
    lines = []
    for s, names in sorted(files_by_shape.items(), key=lambda x: -len(x[1])):
        preview = ", ".join(names[:3])
        if len(names) > 3:
            preview += f", ... ({len(names)} total)"
        lines.append(f"  {s[0]}×{s[1]}: {preview}")
    return "\n".join(lines)


def _filter_by_shape(
    loaded: list[tuple[pathlib.Path, IOResult]],
    shape: tuple[int, int],
) -> list[tuple[pathlib.Path, IOResult]]:
    """Filter loaded results by image shape, with informative errors."""
    filtered = [(p, r) for p, r in loaded if r.data.shape[-2:] == tuple(shape)]
    if not filtered:
        files_by_shape: dict[tuple, list[str]] = {}
        for p, r in loaded:
            s = r.data.shape[-2:]
            files_by_shape.setdefault(s, []).append(p.name)
        most_common = max(files_by_shape, key=lambda s: len(files_by_shape[s]))
        raise ValueError(
            f"No files match shape={shape}. Available sizes:\n"
            f"{_format_shape_summary(files_by_shape)}\n\n"
            f"Suggestion: shape={most_common}"
        )
    n_dropped = len(loaded) - len(filtered)
    if n_dropped:
        print(f"Filtered: kept {len(filtered)}/{len(loaded)} files matching shape={shape}")
    return filtered


def _stack_results(
    results: list[IOResult],
    *,
    title: str = "",
    paths: list[pathlib.Path] | None = None,
) -> IOResult:
    """Stack multiple IOResults into a single (N,H,W) IOResult."""
    frames: list[np.ndarray] = []
    labels: list[str] = []
    pixel_size = None
    units = None
    metadata: dict = {}
    for idx, r in enumerate(results):
        if pixel_size is None and r.pixel_size is not None:
            pixel_size = r.pixel_size
            units = r.units
        if not metadata and r.metadata:
            metadata = r.metadata
        name = paths[idx].name if paths and idx < len(paths) else None
        arr = r.data
        if arr.ndim == 2:
            frames.append(arr)
            labels.append(name or r.title or f"frame_{len(frames) - 1}")
        elif arr.ndim == 3:
            for i in range(arr.shape[0]):
                frames.append(arr[i])
                if name and arr.shape[0] > 1:
                    labels.append(f"{name}[{i}]")
                elif name:
                    labels.append(name)
                elif i < len(r.labels):
                    labels.append(r.labels[i])
                else:
                    labels.append(f"{r.title}[{i}]")
        else:
            raise ValueError(f"Unexpected array shape: {arr.shape}")
    if not frames:
        raise ValueError("No frames to stack.")
    shape0 = frames[0].shape
    if any(f.shape != shape0 for f in frames[1:]):
        files_by_shape: dict[tuple, list[str]] = {}
        for i, frame in enumerate(frames):
            files_by_shape.setdefault(frame.shape, []).append(labels[i])
        most_common = max(files_by_shape, key=lambda s: len(files_by_shape[s]))
        raise ValueError(
            f"Images have different sizes:\n"
            f"{_format_shape_summary(files_by_shape)}\n\n"
            f"Use shape= to select one size, e.g. shape={most_common}"
        )
    stack = np.stack(frames, axis=0).astype(np.float32)
    return IOResult(
        data=stack,
        pixel_size=pixel_size,
        units=units,
        title=title,
        labels=labels,
        metadata=metadata,
    )


# =========================================================================
# GPU backend detection + arina helpers
# =========================================================================


def _detect_gpu_backend() -> str | None:
    try:
        import Metal  # noqa: F401

        return "mps"
    except ImportError:
        pass
    try:
        import cupy  # noqa: F401

        return "cuda"
    except ImportError:
        pass
    return None


# =========================================================================
# CPU fallback for arina loading
# =========================================================================


def _cpu_bin_mean(src: np.ndarray, bin_factor: int) -> np.ndarray:
    """Bin detector axes of a 3D array (n, rows, cols) by averaging.

    Uses numpy reshape+mean — no numba needed, fast enough for CPU path.
    """
    n, rows, cols = src.shape
    out_rows = rows // bin_factor
    out_cols = cols // bin_factor
    trimmed = src[:, : out_rows * bin_factor, : out_cols * bin_factor]
    return (
        trimmed.reshape(n, out_rows, bin_factor, out_cols, bin_factor)
        .mean(axis=(2, 4))
        .astype(np.float32)
    )


def _load_arina_cpu(
    master_path: str,
    det_bin: int = 1,
    scan_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Load arina 4D-STEM data using CPU (h5py + hdf5plugin transparent decompression).

    hdf5plugin registers the bitshuffle+LZ4 HDF5 filter at import time,
    so h5py decompresses chunks transparently — no custom kernels needed.
    Slower than MPS/CUDA but works on any platform.
    """
    import time

    if not _HAS_H5PY:
        raise RuntimeError("h5py is required for arina loading")

    t0 = time.perf_counter()

    # Parse master file structure
    master_dir = os.path.dirname(os.path.abspath(master_path))
    with h5py.File(master_path, "r") as f:
        chunk_keys = sorted(f["entry/data"].keys())
        ds0 = f[f"entry/data/{chunk_keys[0]}"]
        det_shape = ds0.shape[1:]
        dtype = ds0.dtype

    det_row, det_col = det_shape
    prefix = os.path.basename(master_path).replace("_master.h5", "")

    # Discover chunk files and frame counts
    chunk_files = []
    chunk_n_frames = []
    for k in chunk_keys:
        suffix = k.split("_")[-1]
        cf = os.path.join(master_dir, f"{prefix}_data_{suffix}.h5")
        if not os.path.exists(cf):
            raise FileNotFoundError(f"Missing chunk file: {os.path.basename(cf)}")
        chunk_files.append(cf)
        with h5py.File(cf, "r") as f:
            chunk_n_frames.append(f["entry/data/data"].shape[0])

    total_frames = sum(chunk_n_frames)

    # Compute output shape
    if det_bin > 1:
        out_det_row = det_row // det_bin
        out_det_col = det_col // det_bin
        output = np.empty((total_frames, out_det_row, out_det_col), dtype=np.float32)
    else:
        out_det_row, out_det_col = det_row, det_col
        output = np.empty((total_frames, out_det_row, out_det_col), dtype=dtype)

    # Read and decompress chunk by chunk (hdf5plugin handles bitshuffle+LZ4)
    frame_offset = 0
    # Large single-chunk files: read in batches with per-frame progress
    if len(chunk_files) == 1 and total_frames > 1000:
        cf = chunk_files[0]
        batch = 256
        with h5py.File(cf, "r") as f:
            ds = f["entry/data/data"]
            for start in tqdm(
                range(0, total_frames, batch),
                desc="frames",
                total=(total_frames + batch - 1) // batch,
            ):
                end = min(start + batch, total_frames)
                raw = ds[start:end]
                if det_bin > 1:
                    output[start:end] = _cpu_bin_mean(
                        raw.astype(np.float32) if raw.dtype != np.float32 else raw,
                        det_bin,
                    )
                else:
                    output[start:end] = raw
    else:
        chunk_iter = enumerate(chunk_files)
        if len(chunk_files) > 1:
            chunk_iter = tqdm(
                chunk_iter, total=len(chunk_files), desc="chunks", leave=False,
            )
        for ci, cf in chunk_iter:
            nf = chunk_n_frames[ci]
            with h5py.File(cf, "r") as f:
                raw = f["entry/data/data"][:nf]  # hdf5plugin decompresses here

            if det_bin > 1:
                output[frame_offset : frame_offset + nf] = _cpu_bin_mean(
                    raw.astype(np.float32) if raw.dtype != np.float32 else raw,
                    det_bin,
                )
            else:
                output[frame_offset : frame_offset + nf] = raw

            frame_offset += nf

    t_total = time.perf_counter() - t0

    # Infer scan shape
    if scan_shape is None and det_bin > 1:
        side = int(total_frames**0.5)
        if side * side == total_frames:
            scan_shape = (side, side)
    if scan_shape is not None:
        output = output.reshape(*scan_shape, out_det_row, out_det_col)

    tqdm.write(
        f"load_arina (cpu): {total_frames} frames, "
        f"det ({det_row},{det_col}) → ({out_det_row},{out_det_col}), "
        f"{t_total:.2f}s"
    )
    return output


def _get_available_memory() -> int:
    try:
        import psutil

        return psutil.virtual_memory().available
    except ImportError:
        pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        total = pages * page_size
        return int(total * 0.5)
    except (ValueError, OSError, AttributeError):
        return 16 * 1024**3


def _parse_arina_info(master_path: str) -> tuple[int, int, int, np.dtype]:
    if not _HAS_H5PY:
        raise RuntimeError("h5py is required for arina loading")
    with h5py.File(master_path, "r") as f:
        chunk_keys = sorted(f["entry/data"].keys())
        ds0 = f[f"entry/data/{chunk_keys[0]}"]
        det_rows, det_cols = ds0.shape[1], ds0.shape[2]
        dtype = ds0.dtype
        n_frames = 0
        master_dir = os.path.dirname(master_path)
        prefix = os.path.basename(master_path).replace("_master.h5", "")
        for k in chunk_keys:
            suffix = k.split("_")[-1]
            cf = os.path.join(master_dir, f"{prefix}_data_{suffix}.h5")
            with h5py.File(cf, "r") as cf_f:
                n_frames += cf_f["entry/data/data"].shape[0]
    return n_frames, det_rows, det_cols, dtype


def _auto_det_bin(master_path: str) -> int:
    n_frames, det_rows, det_cols, dtype = _parse_arina_info(master_path)
    available = _get_available_memory()
    budget = int(available * 0.6)

    for b in [1, 2, 4, 8]:
        out_r = det_rows // b
        out_c = det_cols // b
        if b == 1:
            out_bytes = n_frames * det_rows * det_cols * dtype.itemsize
        else:
            out_bytes = n_frames * out_r * out_c * 4
        if out_bytes <= budget:
            print(
                f"auto det_bin={b}: {n_frames} frames, "
                f"det ({det_rows},{det_cols}) → ({out_r},{out_c}), "
                f"{out_bytes / 1024**3:.1f} GB "
                f"(budget {budget / 1024**3:.1f} GB)"
            )
            return b

    print("auto det_bin=8: dataset too large for RAM, using maximum binning")
    return 8


def _filter_hot_pixels(
    data: np.ndarray,
    threshold_sigma: float = 5.0,
    n_sample: int = 1000,
) -> np.ndarray:
    if data.ndim == 4:
        sr, sc = data.shape[:2]
        rng = np.random.default_rng(42)
        rows = rng.integers(0, sr, n_sample)
        cols = rng.integers(0, sc, n_sample)
        mean_dp = data[rows, cols].mean(axis=0)
    elif data.ndim == 3:
        n = min(n_sample, data.shape[0])
        idx = np.random.default_rng(42).choice(data.shape[0], n, replace=False)
        mean_dp = data[idx].mean(axis=0)
    else:
        return data

    med = np.median(mean_dp)
    std = np.std(mean_dp)
    hot_mask = mean_dp > (med + threshold_sigma * std)

    n_hot = int(hot_mask.sum())
    if n_hot == 0:
        return data

    print(f"hot pixel filter: {n_hot} pixels zeroed ({threshold_sigma}σ threshold)")
    if data.ndim == 4:
        data[:, :, hot_mask] = 0
    else:
        data[:, hot_mask] = 0
    return data


def _extract_arina_metadata(master_path: str) -> dict:
    """Extract all scalar metadata from an arina master HDF5 (cheap, no decompression).

    Walks the HDF5 tree and stores every scalar dataset (number, string, short
    array) keyed by its full path. Large arrays (flatfield, pixel_mask, lookup
    tables, raw data) are skipped.
    """
    meta: dict = {}
    if not _HAS_H5PY:
        return meta
    try:
        with h5py.File(master_path, "r") as f:
            def _visit(name, obj):
                if not hasattr(obj, "shape"):
                    return  # group, not dataset
                if obj.size > 100:
                    return  # skip large arrays (flatfield, masks, LUTs)
                if "data_" in name:
                    return  # skip data chunk links
                try:
                    val = obj[()]
                    if hasattr(val, "decode"):
                        val = val.decode()
                    elif hasattr(val, "item"):
                        val = val.item()
                    meta[name] = val
                except Exception:
                    pass
            f.visititems(_visit)
    except Exception:
        pass
    return meta


def _apply_scan_bin(data: np.ndarray, scan_bin: int) -> np.ndarray:
    if data.ndim != 4:
        raise ValueError(
            f"scan_bin requires 4D data (scan_rows, scan_cols, det_rows, det_cols), "
            f"got shape {data.shape}"
        )
    sr, sc, dr, dc = data.shape
    new_sr = sr // scan_bin
    new_sc = sc // scan_bin
    trimmed = data[: new_sr * scan_bin, : new_sc * scan_bin]
    binned = trimmed.reshape(new_sr, scan_bin, new_sc, scan_bin, dr, dc)
    return binned.mean(axis=(1, 3)).astype(np.float32)


# =========================================================================
# IO class
# =========================================================================


class IO:
    """Unified file reader for electron microscopy data."""

    @staticmethod
    def _read_single_file(
        path: pathlib.Path,
        *,
        dataset_path: str | None = None,
    ) -> IOResult:
        """Read a single file into an IOResult (internal)."""
        ext = path.suffix.lower()
        if ext in {".png", ".jpg", ".jpeg", ".bmp"}:
            arr = _load_image_2d(path)
            return IOResult(data=arr, title=path.stem)
        if ext in {".tif", ".tiff"}:
            stack, labels = _load_tiff_stack(path)
            if stack.shape[0] == 1:
                return IOResult(data=stack[0], title=path.stem, labels=labels)
            return IOResult(data=stack, title=path.stem, labels=labels)
        if ext in {".emd", ".h5", ".hdf5"}:
            # Try rsciio first (better metadata extraction for Velox EMD)
            try:
                return _load_rsciio(path)
            except Exception:
                pass
            # Fall back to native h5py loader
            arr = _load_emd(path, dataset_path=dataset_path)
            arr = _coerce_to_float32(arr)
            # Squeeze trailing singleton dims (e.g. Velox EMD: (H, W, 1) → (H, W))
            while arr.ndim > 2 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            labels = []
            if arr.ndim == 3:
                labels = [f"{path.stem}[{i}]" for i in range(arr.shape[0])]
            return IOResult(data=arr, title=path.stem, labels=labels)
        if ext == ".npy":
            arr = _coerce_to_float32(_load_npy(path))
            labels = []
            if arr.ndim == 3:
                labels = [f"{path.stem}[{i}]" for i in range(arr.shape[0])]
            return IOResult(data=arr, title=path.stem, labels=labels)
        if ext == ".npz":
            arr = _coerce_to_float32(_load_npz(path))
            labels = []
            if arr.ndim == 3:
                labels = [f"{path.stem}[{i}]" for i in range(arr.shape[0])]
            return IOResult(data=arr, title=path.stem, labels=labels)
        # Fall back to rsciio for DM3, DM4, MRC, SER, etc.
        return _load_rsciio(path)

    @staticmethod
    def file(
        source: "str | pathlib.Path | list[str | pathlib.Path]",
        *,
        dataset_path: str | None = None,
        file_type: str | None = None,
        recursive: bool = False,
        bin_factor: int | None = None,
        shape: "tuple[int, int] | None" = None,
    ) -> IOResult:
        """Read one or more files.

        Parameters
        ----------
        source : str, pathlib.Path, or list
            Single file path or list of file/folder paths.
        dataset_path : str, optional
            Explicit HDF dataset path for ``.emd`` files.
        file_type : str, optional
            Filter for folders in a list (passed to ``IO.folder()``).
        recursive : bool, default False
            For folders in a list (passed to ``IO.folder()``).
        bin_factor : int, optional
            Spatial bin factor applied after loading (mean pooling).
            Pixel size is scaled accordingly.
        shape : tuple of (int, int), optional
            Only include images matching this (height, width). When the
            file list contains images of different sizes, the error
            message shows available shapes and suggests the ``shape=`` fix.

        Returns
        -------
        IOResult

        Examples
        --------
        >>> result = IO.file("gold.dm4")

        >>> result = IO.file(["a.emd", "b.emd", "c.emd"])

        Filter mixed-size files:

        >>> result = IO.file(["overview.emd", "detail.emd"], shape=(1024, 1024))
        """
        if isinstance(source, list):
            if not source:
                raise ValueError("Empty path list.")
            loaded: list[tuple[pathlib.Path, IOResult]] = []
            items = source
            if len(source) > 1:
                items = tqdm(source, desc="files", leave=False)
            for p in items:
                p = pathlib.Path(p)
                if p.is_dir():
                    folder_result = IO.folder(
                        p,
                        file_type=file_type,
                        recursive=recursive,
                        dataset_path=dataset_path,
                    )
                    loaded.append((p, folder_result))
                elif p.is_file():
                    loaded.append((p, IO._read_single_file(p, dataset_path=dataset_path)))
                else:
                    raise ValueError(f"Path does not exist: {p}")
            if shape is not None:
                loaded = _filter_by_shape(loaded, shape)
            if len(loaded) == 1:
                result = loaded[0][1]
            else:
                results = [r for _, r in loaded]
                paths = [p for p, _ in loaded]
                title = pathlib.Path(source[0]).parent.name or "files"
                result = _stack_results(results, title=title, paths=paths)
            if bin_factor is not None and bin_factor > 1:
                result = _apply_bin_factor(result, bin_factor)
            return result

        path = pathlib.Path(source)
        if not path.is_file():
            raise ValueError(f"Path does not exist: {path}")
        result = IO._read_single_file(path, dataset_path=dataset_path)
        if bin_factor is not None and bin_factor > 1:
            result = _apply_bin_factor(result, bin_factor)
        return result

    @staticmethod
    def folder(
        folder: str | pathlib.Path | list,
        *,
        file_type: str | None = None,
        recursive: bool = False,
        dataset_path: str | None = None,
        bin_factor: int | None = None,
        shape: tuple[int, int] | None = None,
    ) -> IOResult:
        """Read a folder of files into a stacked IOResult.

        Parameters
        ----------
        folder : str, pathlib.Path, or list
            Folder path, or list of folder paths to merge into one stack.
        file_type : str, optional
            File type to select (e.g. ``"png"``, ``"tiff"``, ``"dm4"``).
            If omitted, auto-detects from the folder contents.
        recursive : bool, default False
            Include files in subdirectories.
        dataset_path : str, optional
            Explicit HDF dataset path for ``.emd`` files.
        bin_factor : int, optional
            Spatial bin factor applied after loading (mean pooling).
            Pixel size is scaled accordingly.
        shape : tuple of (int, int), optional
            Only include images matching this (height, width). When a
            folder contains images of different sizes, the error message
            shows available shapes and suggests the ``shape=`` fix.

        Returns
        -------
        IOResult

        Examples
        --------
        >>> result = IO.folder("/data/session/", file_type="dm4")

        >>> result = IO.folder("/data/session/")  # auto-detect type

        Filter mixed-size images:

        >>> result = IO.folder("/data/", file_type="emd", shape=(1024, 1024))

        Merge multiple folders:

        >>> result = IO.folder(["/session1/", "/session2/"], file_type="dm3")
        """
        if isinstance(folder, list):
            folder_iter = folder
            if len(folder) > 1:
                folder_iter = tqdm(folder, desc="folders", leave=False)
            loaded = [
                (pathlib.Path(f), IO.folder(
                    f,
                    file_type=file_type,
                    recursive=recursive,
                    dataset_path=dataset_path,
                    shape=shape,
                ))
                for f in folder_iter
            ]
            results = [r for _, r in loaded]
            paths = [p for p, _ in loaded]
            result = _stack_results(results, paths=paths)
            if bin_factor is not None and bin_factor > 1:
                result = _apply_bin_factor(result, bin_factor)
            return result

        folder = pathlib.Path(folder)
        if not folder.is_dir():
            raise ValueError(f"Folder does not exist: {folder}")

        if file_type is not None:
            ft = str(file_type).strip().lower()
            ft = _FOLDER_ALIASES.get(ft, ft)
            exts = _FOLDER_EXT_MAP.get(ft)
            if exts is None:
                supported = sorted(_FOLDER_EXT_MAP.keys())
                raise ValueError(
                    f"Unknown file_type '{file_type}'. Supported: {supported}"
                )
            files = _collect_files(folder, exts=exts, recursive=recursive)
            if not files:
                raise ValueError(f"No {ft.upper()} files found in {folder}.")
        else:
            all_exts = _all_supported_exts()
            files = _collect_files(folder, exts=all_exts, recursive=recursive)
            if not files:
                raise ValueError(f"No supported files found in {folder}.")
            detected = _detect_folder_type(files)
            if detected is None:
                by_ext: dict[str, int] = {}
                for f in files:
                    e = f.suffix.lower().lstrip(".")
                    by_ext[e] = by_ext.get(e, 0) + 1
                summary = ", ".join(f"{e}: {n}" for e, n in sorted(by_ext.items()))
                raise ValueError(
                    f"Folder contains mixed file types ({summary}). "
                    f"Specify file_type= to select one."
                )
            detected_exts = _FOLDER_EXT_MAP.get(detected)
            if detected_exts is not None:
                files = [f for f in files if f.suffix.lower() in detected_exts]

        file_iter = files
        if len(files) > 1:
            file_iter = tqdm(files, desc="files", leave=False)
        loaded = [
            (p, IO._read_single_file(p, dataset_path=dataset_path))
            for p in file_iter
        ]
        if shape is not None:
            loaded = _filter_by_shape(loaded, shape)
        results = [r for _, r in loaded]
        paths = [p for p, _ in loaded]
        result = _stack_results(results, title=folder.name, paths=paths)
        if bin_factor is not None and bin_factor > 1:
            result = _apply_bin_factor(result, bin_factor)
        return result

    @staticmethod
    def supported_formats() -> list[str]:
        """Return sorted list of supported file extensions (without dots)."""
        native = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "emd", "npy", "npz"}
        rsciio_exts = set()
        ext_map = _rsciio_ext_map()
        for ext in ext_map:
            rsciio_exts.add(ext.lstrip("."))
        return sorted(native | rsciio_exts)

    @staticmethod
    def benchmark(path: str | pathlib.Path, *, sample_mb: int = 64) -> None:
        """Measure filesystem read throughput for a file or folder.

        Useful for checking if a network mount (e.g. HPC server) is fast
        enough before starting a long load.

        Parameters
        ----------
        path : str or pathlib.Path
            File or folder to benchmark. For folders, auto-discovers the
            first supported file. For arina masters, targets the first
            data chunk file.
        sample_mb : int, default 64
            Maximum megabytes to read for the speed test.
        """
        import time

        p = pathlib.Path(path)
        if not p.exists():
            raise ValueError(f"Path does not exist: {p}")
        # Folder → find first supported file
        if p.is_dir():
            target = None
            # Try arina masters first
            masters = sorted(p.glob("*_master.h5"))
            if masters:
                target = masters[0]
            else:
                all_exts = _all_supported_exts()
                candidates = _collect_files(p, exts=all_exts, recursive=False)
                if candidates:
                    target = candidates[0]
            if target is None:
                raise ValueError(f"No supported files found in {p}")
            p = target
        # Arina master → resolve to first data chunk
        if p.name.endswith("_master.h5"):
            prefix = p.name.replace("_master.h5", "")
            chunk = p.parent / f"{prefix}_data_000001.h5"
            if chunk.is_file():
                p = chunk
        file_size = p.stat().st_size
        sample_bytes = min(file_size, sample_mb * 1024 * 1024)
        t0 = time.perf_counter()
        with open(p, "rb") as f:
            f.read(sample_bytes)
        elapsed = time.perf_counter() - t0
        throughput = (sample_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else float("inf")
        estimated_full = (file_size / (1024 * 1024)) / throughput if throughput > 0 else float("inf")
        print(
            f"Benchmark: {p.name}\n"
            f"  File size:  {_format_memory(file_size)}\n"
            f"  Sampled:    {_format_memory(sample_bytes)} in {elapsed:.3f}s\n"
            f"  Throughput: {throughput:.1f} MB/s\n"
            f"  Est. full:  {estimated_full:.2f}s"
        )

    @staticmethod
    def _arina_single(
        master_path: str,
        *,
        det_bin: int,
        scan_bin: int,
        scan_shape: tuple[int, int] | None,
        hot_pixel_filter: bool,
        backend: str,
    ) -> IOResult:
        """Load a single arina master file (internal)."""
        if backend in ("mps", "cuda"):
            try:
                if backend == "mps":
                    from quantem.widget.kernels.arina_mps import load_arina
                else:
                    from quantem.widget.kernels.arina_cuda import load_arina
                output = load_arina(
                    master_path, det_bin=det_bin, scan_shape=scan_shape
                )
            except Exception as exc:
                print(
                    f"WARNING: {backend.upper()} backend failed: {exc}\n"
                    f"Falling back to CPU backend. This is slower but produces "
                    f"identical results."
                )
                output = _load_arina_cpu(
                    master_path, det_bin=det_bin, scan_shape=scan_shape
                )
        elif backend == "cpu":
            output = _load_arina_cpu(
                master_path, det_bin=det_bin, scan_shape=scan_shape
            )
        else:
            raise ValueError(
                f"Unknown backend: {backend!r}. Use 'auto', 'mps', 'cuda', or 'cpu'."
            )
        # hot_pixel_filter and scan_bin require numpy arrays
        if hot_pixel_filter or scan_bin > 1:
            from quantem.widget.array_utils import to_numpy

            output = to_numpy(output)
        if hot_pixel_filter:
            output = _filter_hot_pixels(output)
        if scan_bin > 1:
            output = _apply_scan_bin(output, scan_bin)
        title = pathlib.Path(master_path).stem.replace("_master", "")
        meta = _extract_arina_metadata(master_path)
        return IOResult(data=output, title=title, metadata=meta)

    @staticmethod
    def _resolve_arina_args(
        det_bin: int | str,
        backend: str,
        master_path: str | None = None,
    ) -> tuple[int, str]:
        """Resolve auto det_bin and auto backend (internal)."""
        if det_bin == "auto":
            if master_path is None:
                raise ValueError("det_bin='auto' requires a master_path for size estimation")
            det_bin = _auto_det_bin(master_path)
        if not isinstance(det_bin, int) or det_bin < 1:
            raise ValueError(f"det_bin must be a positive int or 'auto', got {det_bin!r}")
        if backend == "auto":
            backend = _detect_gpu_backend()
            if backend is None:
                backend = "cpu"
        return det_bin, backend

    @staticmethod
    def arina_file(
        master_path: "str | list[str]",
        det_bin: int | str = 1,
        scan_bin: int = 1,
        scan_shape: tuple[int, int] | None = None,
        hot_pixel_filter: bool = True,
        backend: str = "auto",
    ) -> IOResult:
        """Load arina 4D-STEM data with GPU-accelerated decompression.

        Accepts a single master file path (returns 4D) or a list of paths
        (returns 5D stacked along a new leading axis).

        Parameters
        ----------
        master_path : str or list of str
            Path to one arina master HDF5 file, or a list of paths.
            A list of two or more files produces a 5D result.
        det_bin : int or ``"auto"``, optional
            Detector binning factor (applied to both axes). ``"auto"`` picks
            the smallest factor that fits in available RAM. Default 1.
        scan_bin : int, optional
            Scan (navigation) binning factor. Applied after loading as a
            mean over ``scan_bin x scan_bin`` neighborhoods. Requires 4D
            output (i.e. ``scan_shape`` must be set or inferred). Default 1.
        scan_shape : tuple of (int, int), optional
            Reshape into ``(scan_rows, scan_cols, det_rows, det_cols)``. If
            None and ``det_bin > 1``, inferred as ``(sqrt(n), sqrt(n))``.
        hot_pixel_filter : bool, optional
            Zero out hot pixels on the detector (pixels > 5σ above median
            in the mean diffraction pattern). Default True.
        backend : str, optional
            ``"auto"`` (MPS on macOS, CPU elsewhere), ``"mps"`` (Apple Metal),
            or ``"cpu"`` (h5py, works on all platforms). Default ``"auto"``.

        Returns
        -------
        IOResult
            Single file: ``.data`` shape ``(scan_r, scan_c, det_r, det_c)``.
            Multiple files: ``.data`` shape ``(n_files, scan_r, scan_c, det_r, det_c)``.

        Examples
        --------
        Single file → 4D:

        >>> result = IO.arina_file("SnMoS2s_001_master.h5", det_bin=2)
        >>> result.data.shape
        (512, 512, 96, 96)

        Auto-detect smallest bin that fits in RAM:

        >>> result = IO.arina_file("master.h5", det_bin="auto")

        Cherry-pick specific files → 5D:

        >>> result = IO.arina_file([
        ...     "scan_00_master.h5",
        ...     "scan_03_master.h5",
        ...     "scan_07_master.h5",
        ... ], det_bin=4)
        >>> result.data.shape
        (3, 256, 256, 48, 48)

        Free GPU memory when done (important for large datasets):

        >>> del widget           # free MPS tensor held by Show4DSTEM
        >>> del result           # free numpy array from IOResult
        >>> import torch, gc
        >>> gc.collect()
        >>> torch.mps.empty_cache()  # release MPS allocator cache
        """
        # --- Handle list of paths ---
        if isinstance(master_path, list):
            if not master_path:
                raise ValueError("Empty file list.")
            if len(master_path) == 1:
                master_path = master_path[0]
            else:
                det_bin, backend = IO._resolve_arina_args(
                    det_bin, backend, master_path=master_path[0]
                )
                labels: list[str] = []
                frame_meta: list[dict] = []
                stack: np.ndarray | None = None
                expected_shape: tuple | None = None
                mp_iter = enumerate(master_path)
                if len(master_path) > 1:
                    mp_iter = tqdm(
                        mp_iter, total=len(master_path), desc="arina files",
                    )
                for i, mp in mp_iter:
                    r = IO._arina_single(
                        str(mp),
                        det_bin=det_bin,
                        scan_bin=scan_bin,
                        scan_shape=scan_shape,
                        hot_pixel_filter=hot_pixel_filter,
                        backend=backend,
                    )
                    if stack is None:
                        expected_shape = r.data.shape
                        stack = np.empty(
                            (len(master_path), *expected_shape), dtype=r.data.dtype
                        )
                    elif r.data.shape != expected_shape:
                        raise ValueError(
                            f"Cannot stack files with different shapes into a "
                            f"5D array.\n"
                            f"  File 0: {expected_shape}\n"
                            f"  File {i} ({pathlib.Path(mp).name}): {r.data.shape}\n"
                            f"All files must have the same scan and detector "
                            f"dimensions. Load them separately with "
                            f"IO.arina_file() instead."
                        )
                    stack[i] = r.data
                    labels.append(pathlib.Path(mp).stem.replace("_master", ""))
                    frame_meta.append(r.metadata)
                return IOResult(
                    data=stack, labels=labels, title=labels[0],
                    frame_metadata=frame_meta,
                )

        # --- Single file ---
        det_bin, backend = IO._resolve_arina_args(
            det_bin, backend, master_path=master_path
        )
        return IO._arina_single(
            master_path,
            det_bin=det_bin,
            scan_bin=scan_bin,
            scan_shape=scan_shape,
            hot_pixel_filter=hot_pixel_filter,
            backend=backend,
        )

    @staticmethod
    def _collect_masters(
        folder: pathlib.Path,
        *,
        recursive: bool,
        pattern: str | None,
    ) -> list[pathlib.Path]:
        """Find *_master.h5 files in a folder, with optional filtering."""
        if recursive:
            masters = sorted(folder.rglob("*_master.h5"))
        else:
            masters = sorted(folder.glob("*_master.h5"))
        if pattern is not None:
            pat = pattern.lower()
            masters = [m for m in masters if pat in m.stem.lower()]
        return masters

    @staticmethod
    def arina_folder(
        folder: "str | pathlib.Path | list[str | pathlib.Path]",
        det_bin: int | str = 1,
        scan_bin: int = 1,
        scan_shape: tuple[int, int] | None = None,
        hot_pixel_filter: bool = True,
        backend: str = "auto",
        max_files: int = 50,
        recursive: bool = False,
        pattern: str | None = None,
        shape: "tuple[int, int] | None" = None,
    ) -> IOResult:
        """Load arina master files from one or more folders into a 5D stack.

        Finds every ``*_master.h5``, loads each with :meth:`IO.arina_file`,
        and stacks them along a new leading axis.

        Parameters
        ----------
        folder : str, pathlib.Path, or list
            Directory (or list of directories) containing ``*_master.h5`` files.
        det_bin, scan_bin, scan_shape, hot_pixel_filter, backend
            Forwarded to :meth:`IO.arina_file` for each file.
        max_files : int, default 50
            Maximum number of master files to load. Prevents accidentally
            loading hundreds of files into RAM. Set to 0 for no limit.
        recursive : bool, default False
            Search subdirectories for ``*_master.h5`` files.
        pattern : str, optional
            Only load files whose stem contains this string (case-insensitive).
            E.g. ``pattern="SnMoS2"`` loads only files with "SnMoS2" in the name.
        shape : tuple of (int, int), optional
            Only include scans whose scan dimensions match ``(scan_rows, scan_cols)``.
            Scans with different scan resolutions are skipped with a message.

        Returns
        -------
        IOResult
            ``.data`` has shape ``(n_files, scan_rows, scan_cols, det_rows, det_cols)``
            (5D) when each file produces 4D output. ``.labels`` contains the
            stem of each master file.

        Examples
        --------
        Load all scans in a session folder:

        >>> result = IO.arina_folder("/data/20260208/", det_bin=4)
        >>> result.data.shape
        (10, 256, 256, 48, 48)

        Filter by sample name:

        >>> result = IO.arina_folder("/data/", pattern="SnMoS2", det_bin=2)

        Merge scans from multiple session folders:

        >>> result = IO.arina_folder(["/data/day1/", "/data/day2/"], det_bin=8)

        Filter by scan resolution (keep only 256x256 scans):

        >>> result = IO.arina_folder("/data/", det_bin=8, shape=(256, 256))

        Search subdirectories:

        >>> result = IO.arina_folder("/data/", recursive=True, pattern="focal", det_bin=4)

        Free GPU memory when done (important for large datasets):

        >>> del widget           # free MPS tensor held by Show4DSTEM
        >>> del result           # free numpy array from IOResult
        >>> import torch, gc
        >>> gc.collect()
        >>> torch.mps.empty_cache()  # release MPS allocator cache
        """
        # --- Handle list of folders ---
        if isinstance(folder, list):
            all_masters: list[pathlib.Path] = []
            for f in folder:
                f = pathlib.Path(f)
                if not f.is_dir():
                    raise ValueError(f"Folder does not exist: {f}")
                all_masters.extend(
                    IO._collect_masters(f, recursive=recursive, pattern=pattern)
                )
            all_masters.sort()
            masters = all_masters
            title = pathlib.Path(folder[0]).name if folder else "folders"
        else:
            folder = pathlib.Path(folder)
            if not folder.is_dir():
                raise ValueError(f"Folder does not exist: {folder}")
            masters = IO._collect_masters(folder, recursive=recursive, pattern=pattern)
            title = folder.name

        # --- Apply max_files cap ---
        if max_files > 0 and len(masters) > max_files:
            print(
                f"Found {len(masters)} master files, loading first {max_files} "
                f"(set max_files=0 to load all)"
            )
            masters = masters[:max_files]
        if not masters:
            filter_msg = f" matching pattern={pattern!r}" if pattern else ""
            raise ValueError(f"No *_master.h5 files found{filter_msg}")

        det_bin, backend = IO._resolve_arina_args(
            det_bin, backend, master_path=str(masters[0])
        )
        labels: list[str] = []
        frame_meta: list[dict] = []
        skipped: list[str] = []
        files_by_shape: dict[tuple, list[str]] = {}
        stack: np.ndarray | None = None
        expected_shape: tuple | None = None
        slot = 0
        master_iter = enumerate(masters)
        if len(masters) > 1:
            master_iter = tqdm(
                master_iter, total=len(masters), desc="arina files",
            )
        for i, mp in master_iter:
            try:
                r = IO._arina_single(
                    str(mp),
                    det_bin=det_bin,
                    scan_bin=scan_bin,
                    scan_shape=scan_shape,
                    hot_pixel_filter=hot_pixel_filter,
                    backend=backend,
                )
            except (FileNotFoundError, ValueError, OSError) as exc:
                skipped.append(mp.name)
                tqdm.write(f"  SKIPPED {mp.name}: {exc}")
                continue
            scan_dims = r.data.shape[:2]
            files_by_shape.setdefault(scan_dims, []).append(mp.name)
            if shape is not None and scan_dims != tuple(shape):
                skipped.append(mp.name)
                tqdm.write(
                    f"  SKIPPED {mp.name}: scan shape {scan_dims} "
                    f"!= shape={shape}"
                )
                continue
            if stack is None:
                expected_shape = r.data.shape
                n_good = len(masters) - len(skipped)
                stack = np.empty((n_good, *expected_shape), dtype=r.data.dtype)
            elif r.data.shape != expected_shape:
                skipped.append(mp.name)
                tqdm.write(
                    f"  SKIPPED {mp.name}: shape {r.data.shape} != expected "
                    f"{expected_shape}. All files must have the same scan and "
                    f"detector dimensions."
                )
                continue
            stack[slot] = r.data
            slot += 1
            labels.append(mp.stem.replace("_master", ""))
            frame_meta.append(r.metadata)
        if skipped and shape is not None:
            print(f"Filtered: kept {slot}/{len(masters)} files matching shape={shape}")
        elif skipped:
            tqdm.write(f"Skipped {len(skipped)}/{len(masters)} files: {skipped}")
        if stack is None:
            if shape is not None and files_by_shape:
                # Strip _master.h5 from names for readability
                clean = {
                    s: [n.replace("_master.h5", "") for n in names]
                    for s, names in files_by_shape.items()
                }
                most_common = max(clean, key=lambda s: len(clean[s]))
                raise ValueError(
                    f"No scans match shape={shape}. Available scan shapes:\n"
                    f"{_format_shape_summary(clean)}\n\n"
                    f"Suggestion: shape={most_common}"
                )
            raise ValueError(
                f"All {len(masters)} master files failed to load"
            )
        if slot < stack.shape[0]:
            stack = stack[:slot]
        # Single file → squeeze to 4D (consistent with arina_file behaviour)
        if stack.shape[0] == 1:
            stack = stack[0]
        return IOResult(
            data=stack, title=title, labels=labels, frame_metadata=frame_meta,
        )

    # Alias: IO.arina() → IO.arina_file() for backwards compatibility
    arina = arina_file
