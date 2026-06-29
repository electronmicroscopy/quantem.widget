"""Interactive EDS/EELS spectrum-image explorer widget."""

from __future__ import annotations

import json
import pathlib
import tempfile
import time
from functools import lru_cache
from typing import Any
from urllib.parse import quote, unquote

import anywidget
import numpy as np
import traitlets

from quantem.widget.utils.array import to_numpy
from quantem.widget.utils.state_io import (
    resolve_widget_version,
    save_state_file,
    unwrap_state_payload,
)


DEFAULT_MAX_SHOWEDS_SIDECAR_BYTES = 12 * 1024**3


_FALLBACK_LINES: tuple[dict[str, Any], ...] = (
    {"element": "C", "line": "Ka1", "energy_keV": 0.277, "intensity": 1.0},
    {"element": "N", "line": "Ka1", "energy_keV": 0.392, "intensity": 1.0},
    {"element": "O", "line": "Ka1", "energy_keV": 0.525, "intensity": 1.0},
    {"element": "F", "line": "Ka1", "energy_keV": 0.677, "intensity": 1.0},
    {"element": "Na", "line": "Ka1", "energy_keV": 1.041, "intensity": 1.0},
    {"element": "Mg", "line": "Ka1", "energy_keV": 1.254, "intensity": 1.0},
    {"element": "Al", "line": "Ka1", "energy_keV": 1.487, "intensity": 1.0},
    {"element": "Si", "line": "Ka1", "energy_keV": 1.740, "intensity": 1.0},
    {"element": "P", "line": "Ka1", "energy_keV": 2.014, "intensity": 1.0},
    {"element": "S", "line": "Ka1", "energy_keV": 2.308, "intensity": 1.0},
    {"element": "Cl", "line": "Ka1", "energy_keV": 2.622, "intensity": 1.0},
    {"element": "K", "line": "Ka1", "energy_keV": 3.314, "intensity": 1.0},
    {"element": "Ca", "line": "Ka1", "energy_keV": 3.692, "intensity": 1.0},
    {"element": "Fe", "line": "Ka1", "energy_keV": 6.404, "intensity": 1.0},
    {"element": "Cu", "line": "Ka1", "energy_keV": 8.047, "intensity": 1.0},
    {"element": "Au", "line": "Ma", "energy_keV": 2.118, "intensity": 1.0},
    {"element": "Au", "line": "Mb", "energy_keV": 2.203, "intensity": 1.0},
    {"element": "Au", "line": "La1", "energy_keV": 9.713, "intensity": 0.7},
)


def _line_family(line: str) -> str:
    if not line:
        return ""
    head = line[0].upper()
    if head in {"K", "L", "M"}:
        return head
    return ""


def _line_priority(line: str) -> int:
    if line in {"Ka1", "La1", "Ma"}:
        return 0
    if line in {"Ka2", "La2", "Mb"}:
        return 1
    if line in {"Kb1", "Lb1", "Mg"}:
        return 2
    return 3


@lru_cache(maxsize=1)
def all_eds_lines() -> tuple[dict[str, Any], ...]:
    """Return characteristic X-ray lines as JSON-serializable dictionaries."""

    try:
        import xraydb
    except Exception:
        return _FALLBACK_LINES

    lines: list[dict[str, Any]] = []
    for z in range(4, 93):
        symbol = xraydb.atomic_symbol(z)
        for name, line in xraydb.xray_lines(symbol).items():
            energy_kev = float(line.energy) / 1000.0
            intensity = float(line.intensity)
            family = _line_family(name)
            if family not in {"K", "L", "M"}:
                continue
            if energy_kev < 0.08 or energy_kev > 30.0 or intensity < 0.015:
                continue
            lines.append(
                {
                    "element": symbol,
                    "line": name,
                    "family": family,
                    "energy_keV": round(energy_kev, 5),
                    "intensity": round(intensity, 6),
                }
            )
    return tuple(sorted(lines, key=lambda item: (item["energy_keV"], item["element"], _line_priority(item["line"]))))


def eds_line_hints(
    energy_min: float,
    energy_max: float,
    *,
    elements: list[str] | tuple[str, ...] | None = None,
    max_lines: int = 600,
) -> list[dict[str, Any]]:
    """Return line hints inside an energy range for frontend display."""

    allowed = {element.strip().capitalize() for element in elements or [] if element.strip()}
    out: list[dict[str, Any]] = []
    for line in all_eds_lines():
        energy = float(line["energy_keV"])
        if energy < energy_min or energy > energy_max:
            continue
        if allowed and line["element"] not in allowed:
            continue
        out.append(dict(line))
    out.sort(key=lambda item: (float(item["energy_keV"]), item["element"], _line_priority(item["line"])))
    return out[:max_lines]


def _normalise_element_symbols(elements: Any) -> list[str]:
    """Return unique element symbols with user-friendly capitalization."""

    if elements is None:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in elements:
        symbol = str(raw).strip()
        if not symbol:
            continue
        symbol = symbol[:1].upper() + symbol[1:].lower()
        if symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def _format_bytes(n_bytes: int) -> str:
    value = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{n_bytes} B"


def _directory_size(path: pathlib.Path) -> int:
    total = 0
    try:
        for child in path.rglob("*"):
            if child.is_file():
                total += child.stat().st_size
    except OSError:
        return 0
    return total


def _estimate_spectrum_image_sidecar_bytes(
    shape: tuple[int, int, int],
    *,
    include_base_image: bool = True,
) -> int:
    """Estimate bytes written by the exact ShowEDS prefix-cache data folder."""

    rows, cols, n_energy = (int(v) for v in shape)
    energy_prefix = (n_energy + 1) * rows * cols * np.dtype("<u4").itemsize
    spatial_prefix = (rows + 1) * (cols + 1) * n_energy * np.dtype("<u4").itemsize
    base = rows * cols * np.dtype("<f4").itemsize if include_base_image else 0
    return int(energy_prefix + spatial_prefix + base)


def _estimate_spectrum_stream_sidecar_bytes(n_events: int, rows: int, cols: int, n_energy: int) -> int:
    """Estimate bytes written by the sparse Velox stream data folder."""

    return int(
        (n_energy + 1) * np.dtype("<u4").itemsize
        + n_events * np.dtype("<u4").itemsize
        + (rows * cols + 1) * np.dtype("<u4").itemsize
        + n_events * np.dtype("<u2").itemsize
        + rows * cols * np.dtype("<f4").itemsize
    )


def _binned_spectrum_image_shape(
    shape: tuple[int, int, int],
    *,
    spatial_bin: int = 1,
    energy_bin: int = 1,
) -> tuple[int, int, int]:
    """Return the sum-binned cube shape after trimming incomplete edge bins."""

    rows, cols, n_energy = (int(v) for v in shape)
    sb = int(max(1, spatial_bin))
    eb = int(max(1, energy_bin))
    return rows // sb, cols // sb, n_energy // eb


def _bin_axis_sum(data: np.ndarray, axis: int, factor: int) -> np.ndarray:
    factor = int(factor)
    if factor <= 1:
        return np.asarray(data)
    usable = data.shape[axis] // factor * factor
    if usable <= 0:
        raise ValueError(f"bin factor {factor} is larger than axis {axis} with length {data.shape[axis]}")
    if usable != data.shape[axis]:
        slicer = [slice(None)] * data.ndim
        slicer[axis] = slice(0, usable)
        data = data[tuple(slicer)]
    shape = list(data.shape)
    shape[axis : axis + 1] = [usable // factor, factor]
    return data.reshape(shape).sum(axis=axis + 1)


def bin_spectrum_image(
    cube: np.ndarray,
    energy_keV: np.ndarray | list[float] | None = None,
    *,
    base_image: np.ndarray | None = None,
    spatial_bin: int = 1,
    energy_bin: int = 1,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Sum-bin an EDS/EELS cube before sending it to the browser.

    ``spatial_bin`` bins rows and columns; ``energy_bin`` bins adjacent energy
    channels. Summing preserves counts, which is usually the right behavior for
    spectrum-image exploration.
    """

    data = np.asarray(to_numpy(cube))
    if data.ndim != 3:
        raise ValueError(f"bin_spectrum_image expects a 3D cube, got {data.ndim}D")
    spatial_bin = int(max(1, spatial_bin))
    energy_bin = int(max(1, energy_bin))
    out = _bin_axis_sum(data, 0, spatial_bin)
    out = _bin_axis_sum(out, 1, spatial_bin)
    out = _bin_axis_sum(out, 2, energy_bin)

    axis_out = None
    if energy_keV is not None:
        axis = np.asarray(energy_keV, dtype=np.float32)
        if axis.ndim != 1 or axis.size != data.shape[2]:
            raise ValueError(f"energy_keV must be length {data.shape[2]}, got shape {axis.shape}")
        if energy_bin > 1:
            usable = axis.size // energy_bin * energy_bin
            axis = axis[:usable].reshape(-1, energy_bin).mean(axis=1)
        axis_out = axis.astype(np.float32, copy=False)

    base_out = None
    if base_image is not None:
        base = np.asarray(to_numpy(base_image))
        if base.shape != data.shape[:2]:
            raise ValueError(f"base_image must have shape {data.shape[:2]}, got {base.shape}")
        base_out = _bin_axis_sum(_bin_axis_sum(base, 0, spatial_bin), 1, spatial_bin)

    return np.asarray(out, dtype=np.float32), axis_out, None if base_out is None else np.asarray(base_out, dtype=np.float32)


def _sum_bin_spectrum_image_lazy(
    cube: Any,
    energy_keV: np.ndarray | list[float] | None = None,
    *,
    base_image: Any | None = None,
    spatial_bin: int = 1,
    energy_bin: int = 1,
) -> tuple[Any, np.ndarray | None, Any | None]:
    """Sum-bin a lazy spectrum image without materializing the raw cube."""

    spatial_bin = int(max(1, spatial_bin))
    energy_bin = int(max(1, energy_bin))
    if spatial_bin <= 1 and energy_bin <= 1:
        axis = None if energy_keV is None else np.asarray(energy_keV, dtype=np.float32)
        return cube, axis, base_image
    try:
        import dask.array as da
    except ImportError:
        if hasattr(cube, "compute"):
            raise ImportError(
                "Lazy ShowEDS binning requires dask. Install dask, choose a smaller eager array, "
                "or keep native EMD data in the exact lazy query backend."
            )
        return bin_spectrum_image(
            np.asarray(cube),
            energy_keV,
            base_image=None if base_image is None else np.asarray(base_image),
            spatial_bin=spatial_bin,
            energy_bin=energy_bin,
        )

    if not isinstance(cube, da.Array):
        if hasattr(cube, "compute"):
            raise TypeError(
                "Lazy ShowEDS binning expected a dask.array.Array. Refusing to materialize a lazy "
                "native EDS source before binning."
            )
        return bin_spectrum_image(
            np.asarray(cube),
            energy_keV,
            base_image=None if base_image is None else np.asarray(base_image),
            spatial_bin=spatial_bin,
            energy_bin=energy_bin,
        )

    rows, cols, n_energy = (int(v) for v in cube.shape)
    out_rows, out_cols, out_energy = _binned_spectrum_image_shape(
        (rows, cols, n_energy),
        spatial_bin=spatial_bin,
        energy_bin=energy_bin,
    )
    if min(out_rows, out_cols, out_energy) <= 0:
        raise ValueError(
            f"bin factors spatial_bin={spatial_bin}, energy_bin={energy_bin} are larger than cube shape {cube.shape}"
        )
    cropped = cube[: out_rows * spatial_bin, : out_cols * spatial_bin, : out_energy * energy_bin]
    factors: dict[int, int] = {}
    if spatial_bin > 1:
        factors[0] = spatial_bin
        factors[1] = spatial_bin
    if energy_bin > 1:
        factors[2] = energy_bin
    binned = da.coarsen(np.sum, cropped, factors, trim_excess=False)

    axis_out = None
    if energy_keV is not None:
        axis = np.asarray(energy_keV, dtype=np.float32)
        if axis.ndim != 1 or axis.size != n_energy:
            raise ValueError(f"energy_keV must be length {n_energy}, got shape {axis.shape}")
        if energy_bin > 1:
            axis = axis[: out_energy * energy_bin].reshape(out_energy, energy_bin).mean(axis=1)
        else:
            axis = axis[:out_energy]
        axis_out = axis.astype(np.float32, copy=False)

    base_out = None
    if base_image is not None:
        base = base_image if isinstance(base_image, da.Array) else da.from_array(base_image)
        if tuple(int(v) for v in base.shape) != (rows, cols):
            raise ValueError(f"base_image must have shape {(rows, cols)}, got {base.shape}")
        base_crop = base[: out_rows * spatial_bin, : out_cols * spatial_bin]
        if spatial_bin > 1:
            base_out = da.coarsen(np.sum, base_crop, {0: spatial_bin, 1: spatial_bin}, trim_excess=False)
        else:
            base_out = base_crop

    return binned, axis_out, base_out


def prepare_spectrum_image_sidecar(
    cube: np.ndarray,
    energy_keV: np.ndarray | list[float],
    out_dir: str | pathlib.Path,
    *,
    base_image: np.ndarray | None = None,
    energy_chunk: int = 256,
    max_sidecar_bytes: int | None = DEFAULT_MAX_SHOWEDS_SIDECAR_BYTES,
) -> pathlib.Path:
    """Write an exact ShowEDS data folder for fast EDS/EELS interaction.

    The data folder keeps the raw count precision but stores summaries that make
    browser interaction proportional to screen size or energy channels, not to
    the full ``row * col * energy`` cube on every drag. Files are written as
    little-endian ``uint32``:

    - ``energy_prefix_u32.bin``: cumulative energy planes, shape
      ``(n_energy + 1, rows, cols)``. Band maps are exact plane differences.
    - ``spatial_prefix_u32.bin``: summed-area spectra, shape
      ``(rows + 1, cols + 1, n_energy)``. ROI spectra fetch four contiguous
      spectra and subtract them exactly.

    This is intentionally a preprocessing step. Once the files exist, the live
    widget can fetch them from the frontend without Python round trips.

    This prefix-cache format is meant for small or deliberately spatial-binned
    portable viewers. Native vendor EDS files should usually remain the query
    source for no-bin work: the widget only needs the current energy window,
    ROI spectrum, or visible preview, not a fully expanded browser artifact.
    ``max_sidecar_bytes`` protects against accidentally creating a portable
    cache that is not appropriate for the source.
    """

    data = cube
    if not hasattr(data, "ndim") or not hasattr(data, "shape") or not hasattr(data, "dtype"):
        data = np.asarray(to_numpy(cube))
    if int(data.ndim) != 3:
        raise ValueError(f"prepare_spectrum_image_sidecar expects a 3D cube, got {data.ndim}D")
    axis = np.asarray(energy_keV, dtype=np.float32)
    rows, cols, n_energy = (int(v) for v in data.shape)
    if axis.ndim != 1 or axis.size != n_energy:
        raise ValueError(f"energy_keV must be length {n_energy}, got shape {axis.shape}")
    if not np.issubdtype(np.dtype(data.dtype), np.integer):
        raise ValueError("prepare_spectrum_image_sidecar currently expects integer count data")
    estimated_bytes = _estimate_spectrum_image_sidecar_bytes(
        (rows, cols, n_energy),
        include_base_image=base_image is not None,
    )
    if max_sidecar_bytes is not None and estimated_bytes > int(max_sidecar_bytes):
        raise ValueError(
            "ShowEDS exact prefix-cache data folder would be too large for this source. "
            f"The requested cache shape {(rows, cols, n_energy)} is estimated to write "
            f"{_format_bytes(estimated_bytes)}, above the {_format_bytes(int(max_sidecar_bytes))} "
            "safety limit. Keep the native EMD file as the exact query source for no-bin work, "
            "or choose an explicit spatial_bin for a portable sharing cache."
        )
    if not np.issubdtype(np.dtype(data.dtype), np.unsignedinteger):
        min_value = data.min().compute() if hasattr(data.min(), "compute") else np.nanmin(data)
        if min_value < 0:
            raise ValueError("prepare_spectrum_image_sidecar expects non-negative counts")

    out = pathlib.Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    prefix_path = out / "energy_prefix_u32.bin"
    prefix = np.memmap(prefix_path, dtype="<u4", mode="w+", shape=(n_energy + 1, rows, cols))
    prefix[0] = 0
    spatial_path = out / "spatial_prefix_u32.bin"
    spatial = np.memmap(spatial_path, dtype="<u4", mode="w+", shape=(rows + 1, cols + 1, n_energy))
    running = np.zeros((rows, cols), dtype=np.uint64)
    chunk_size = int(max(1, energy_chunk))
    max_u32 = np.iinfo(np.uint32).max

    for start in range(0, n_energy, chunk_size):
        end = min(n_energy, start + chunk_size)
        block = data[:, :, start:end]
        if hasattr(block, "compute"):
            block = block.compute()
        block = np.asarray(block)
        sat = np.zeros((rows + 1, cols + 1, end - start), dtype=np.uint32)
        for j, e in enumerate(range(start, end)):
            plane = np.asarray(block[:, :, j], dtype=np.uint64)
            running += plane
            if running.max(initial=0) > max_u32:
                raise OverflowError("energy prefix exceeds uint32; split counts are needed for exact storage")
            prefix[e + 1] = running.astype(np.uint32)

            integral = plane.cumsum(axis=0, dtype=np.uint64).cumsum(axis=1, dtype=np.uint64)
            if integral.max(initial=0) > max_u32:
                raise OverflowError("spatial integral exceeds uint32; split counts are needed for exact storage")
            sat[1:, 1:, j] = integral.astype(np.uint32)
        spatial[:, :, start:end] = sat
    prefix.flush()
    spatial.flush()

    base = None
    if base_image is not None:
        base = np.asarray(to_numpy(base_image), dtype=np.float32)
        if base.shape != (rows, cols):
            raise ValueError(f"base_image must have shape {(rows, cols)}, got {base.shape}")
        (out / "base_f32.bin").write_bytes(np.ascontiguousarray(base, dtype="<f4").tobytes())

    meta = {
        "format": "quantem.widget.showeds.sidecar.v1",
        "rows": rows,
        "cols": cols,
        "n_energy": n_energy,
        "energy_keV": [float(v) for v in axis],
        "energy_prefix": "energy_prefix_u32.bin",
        "energy_prefix_shape": [n_energy + 1, rows, cols],
        "spatial_prefix": "spatial_prefix_u32.bin",
        "spatial_prefix_shape": [rows + 1, cols + 1, n_energy],
        "base_image": "base_f32.bin" if base is not None else "",
        "build_seconds": round(time.perf_counter() - t0, 3),
    }
    (out / "meta.json").write_text(json.dumps(meta))
    return out


def _stream_index_kernels():
    try:
        from numba import njit
    except Exception:  # pragma: no cover - fallback used only without numba
        return None, None

    @njit(cache=True)
    def count_stream_events(streams, rows, cols, n_energy):  # type: ignore[no-untyped-def]
        n_pixels = rows * cols
        channel_counts = np.zeros(n_energy, np.uint32)
        pixel_counts = np.zeros(n_pixels, np.uint32)
        total = 0
        for si in range(len(streams)):
            nav = 0
            stream = streams[si]
            for value in stream:
                if value == 65535:
                    nav += 1
                    if nav >= n_pixels:
                        break
                elif value < n_energy:
                    channel_counts[value] += 1
                    pixel_counts[nav] += 1
                    total += 1
        return channel_counts, pixel_counts, total

    @njit(cache=True)
    def fill_stream_indexes(  # type: ignore[no-untyped-def]
        streams,
        rows,
        cols,
        n_energy,
        channel_offsets,
        pixel_offsets,
        channel_pixels,
        pixel_channels,
    ):
        n_pixels = rows * cols
        channel_pos = channel_offsets[:-1].copy()
        pixel_pos = pixel_offsets[:-1].copy()
        for si in range(len(streams)):
            nav = 0
            stream = streams[si]
            for value in stream:
                if value == 65535:
                    nav += 1
                    if nav >= n_pixels:
                        break
                elif value < n_energy:
                    cp = channel_pos[value]
                    channel_pixels[cp] = nav
                    channel_pos[value] = cp + 1
                    pp = pixel_pos[nav]
                    pixel_channels[pp] = value
                    pixel_pos[nav] = pp + 1

    return count_stream_events, fill_stream_indexes


def _stream_index_fallback(streams: list[np.ndarray], rows: int, cols: int, n_energy: int):
    n_pixels = rows * cols
    channel_counts = np.zeros(n_energy, dtype=np.uint32)
    pixel_counts = np.zeros(n_pixels, dtype=np.uint32)
    total = 0
    for stream in streams:
        nav = 0
        for value in stream:
            v = int(value)
            if v == 65535:
                nav += 1
                if nav >= n_pixels:
                    break
            elif v < n_energy:
                channel_counts[v] += 1
                pixel_counts[nav] += 1
                total += 1
    return channel_counts, pixel_counts, total


def _fill_stream_index_fallback(
    streams: list[np.ndarray],
    rows: int,
    cols: int,
    n_energy: int,
    channel_offsets: np.ndarray,
    pixel_offsets: np.ndarray,
    channel_pixels: np.ndarray,
    pixel_channels: np.ndarray,
) -> None:
    n_pixels = rows * cols
    channel_pos = channel_offsets[:-1].copy()
    pixel_pos = pixel_offsets[:-1].copy()
    for stream in streams:
        nav = 0
        for value in stream:
            v = int(value)
            if v == 65535:
                nav += 1
                if nav >= n_pixels:
                    break
            elif v < n_energy:
                cp = int(channel_pos[v])
                channel_pixels[cp] = nav
                channel_pos[v] = cp + 1
                pp = int(pixel_pos[nav])
                pixel_channels[pp] = v
                pixel_pos[nav] = pp + 1


def _velox_stream_groups(path: pathlib.Path) -> tuple[list[np.ndarray], int, int, int]:
    import h5py

    with h5py.File(path, "r") as handle:
        stream_root = handle.get("Data/SpectrumStream")
        if stream_root is None:
            raise ValueError(f"No Velox SpectrumStream group found in {path}")
        keys = sorted(stream_root.keys())
        if not keys:
            raise ValueError(f"Velox SpectrumStream group in {path} is empty")
        streams: list[np.ndarray] = []
        rows = cols = n_energy = 0
        for key in keys:
            group = stream_root[key]
            settings = json.loads(group["AcquisitionSettings"][0].decode("utf-8"))
            raster = settings.get("RasterScanDefinition", {})
            cols = int(raster.get("Width", cols))
            rows = int(raster.get("Height", rows))
            n_energy = int(settings.get("bincount", n_energy))
            streams.append(np.asarray(group["Data"][:, 0], dtype=np.uint16))
    if rows <= 0 or cols <= 0 or n_energy <= 0:
        raise ValueError(f"Could not read Velox SpectrumStream dimensions from {path}")
    return streams, rows, cols, n_energy


def prepare_spectrum_stream_sidecar(
    path: str | pathlib.Path,
    out_dir: str | pathlib.Path,
    *,
    base_image: np.ndarray | None = None,
    energy_keV: np.ndarray | None = None,
    max_sidecar_bytes: int | None = DEFAULT_MAX_SHOWEDS_SIDECAR_BYTES,
) -> pathlib.Path:
    """Build an exact sparse data folder from a Velox EDS spectrum stream.

    The folder is optimized for interactive exploration: band maps use a
    channel-sorted pixel index, and ROI spectra use a pixel-sorted channel
    index. No dense ``(row, col, energy)`` cube is materialized.
    """

    source = pathlib.Path(path).expanduser()
    out = pathlib.Path(out_dir).expanduser()
    streams, rows, cols, n_energy = _velox_stream_groups(source)
    if energy_keV is None or base_image is None:
        loaded = _read_emd_spectrum_image(source, lazy=True)
        energy_keV = np.asarray(loaded["energy_keV"], dtype=np.float32) if energy_keV is None else energy_keV
        if base_image is None and loaded.get("base_image") is not None:
            base_image = _compute_numpy(loaded["base_image"], dtype=np.float32)
    axis = np.asarray(energy_keV if energy_keV is not None else np.arange(n_energy), dtype=np.float32)
    if axis.shape != (n_energy,):
        raise ValueError(f"energy_keV must have length {n_energy}, got shape {axis.shape}")
    base = np.zeros((rows, cols), dtype=np.float32) if base_image is None else np.asarray(to_numpy(base_image), dtype=np.float32)
    if base.shape != (rows, cols):
        raise ValueError(f"base_image must have shape {(rows, cols)}, got {base.shape}")

    try:
        from numba.typed import List as NumbaList

        typed_streams = NumbaList()
        for stream in streams:
            typed_streams.append(stream)
        count_kernel, fill_kernel = _stream_index_kernels()
    except Exception:
        typed_streams = None
        count_kernel = fill_kernel = None

    if count_kernel is not None and fill_kernel is not None and typed_streams is not None:
        channel_counts, pixel_counts, total_events = count_kernel(typed_streams, rows, cols, n_energy)
    else:  # pragma: no cover - numba is expected in development
        channel_counts, pixel_counts, total_events = _stream_index_fallback(streams, rows, cols, n_energy)

    estimated_bytes = _estimate_spectrum_stream_sidecar_bytes(int(total_events), rows, cols, n_energy)
    if max_sidecar_bytes is not None and estimated_bytes > int(max_sidecar_bytes):
        raise ValueError(
            "ShowEDS sparse stream folder would be "
            f"{_format_bytes(estimated_bytes)}, above the {_format_bytes(int(max_sidecar_bytes))} safety limit."
        )

    channel_offsets = np.empty(n_energy + 1, dtype=np.uint32)
    channel_offsets[0] = 0
    channel_offsets[1:] = np.cumsum(channel_counts, dtype=np.uint64).astype(np.uint32)
    pixel_offsets = np.empty(rows * cols + 1, dtype=np.uint32)
    pixel_offsets[0] = 0
    pixel_offsets[1:] = np.cumsum(pixel_counts, dtype=np.uint64).astype(np.uint32)
    channel_pixels = np.empty(int(total_events), dtype=np.uint32)
    pixel_channels = np.empty(int(total_events), dtype=np.uint16)

    if fill_kernel is not None and typed_streams is not None:
        fill_kernel(typed_streams, rows, cols, n_energy, channel_offsets, pixel_offsets, channel_pixels, pixel_channels)
    else:  # pragma: no cover
        _fill_stream_index_fallback(
            streams,
            rows,
            cols,
            n_energy,
            channel_offsets,
            pixel_offsets,
            channel_pixels,
            pixel_channels,
        )

    out.mkdir(parents=True, exist_ok=True)
    (out / "channel_offsets_u32.bin").write_bytes(channel_offsets.astype("<u4", copy=False).tobytes())
    (out / "channel_pixels_u32.bin").write_bytes(channel_pixels.astype("<u4", copy=False).tobytes())
    (out / "pixel_offsets_u32.bin").write_bytes(pixel_offsets.astype("<u4", copy=False).tobytes())
    (out / "pixel_channels_u16.bin").write_bytes(pixel_channels.astype("<u2", copy=False).tobytes())
    (out / "base_f32.bin").write_bytes(base.astype("<f4", copy=False).tobytes())
    meta = {
        "format": "quantem.widget.showeds.stream-sidecar.v1",
        "source": str(source),
        "rows": int(rows),
        "cols": int(cols),
        "n_energy": int(n_energy),
        "n_events": int(total_events),
        "energy_keV": axis.astype(float).tolist(),
        "channel_offsets": "channel_offsets_u32.bin",
        "channel_pixels": "channel_pixels_u32.bin",
        "pixel_offsets": "pixel_offsets_u32.bin",
        "pixel_channels": "pixel_channels_u16.bin",
        "base_image": "base_f32.bin",
        "sidecar_bytes": int(estimated_bytes),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return out


def _normalise_sidecar_url(sidecar_url: str | None) -> str:
    url = str(sidecar_url or "")
    if url and not url.endswith("/"):
        url += "/"
    return url


def _sidecar_path_from_url(sidecar_url: str) -> pathlib.Path | None:
    url = str(sidecar_url or "")
    if url.startswith("/files/"):
        return pathlib.Path.cwd() / unquote(url[len("/files/") :].strip("/"))
    if url.startswith("files/"):
        return pathlib.Path.cwd() / unquote(url[len("files/") :].strip("/"))
    return None


def _sidecar_url_from_path(sidecar_path: str | pathlib.Path) -> str:
    path = pathlib.Path(sidecar_path).expanduser()
    try:
        rel = path.resolve().relative_to(pathlib.Path.cwd().resolve())
    except ValueError as exc:
        raise ValueError(
            "sidecar_url is required when sidecar_dir is outside the current Jupyter files root"
        ) from exc
    return "/files/" + "/".join(quote(part) for part in rel.parts) + "/"


def _resolve_band_indices(
    axis: np.ndarray,
    *,
    energy: float | None = None,
    width: float | None = None,
    band: tuple[float, float] | tuple[int, int] | None = None,
) -> tuple[int, int]:
    n_energy = int(axis.size)
    if n_energy <= 0:
        return 0, 1
    if band is not None:
        a, b = band
        if isinstance(a, int) and isinstance(b, int):
            start = int(min(a, b))
            end = int(max(a, b))
        else:
            lo = float(min(a, b))
            hi = float(max(a, b))
            start = int(np.searchsorted(axis, lo, side="left"))
            end = int(np.searchsorted(axis, hi, side="right"))
    elif energy is not None:
        span = float(np.ptp(axis))
        half = float(width if width is not None else max(span / max(n_energy, 1) * 8, 0.1)) / 2
        start = int(np.searchsorted(axis, float(energy) - half, side="left"))
        end = int(np.searchsorted(axis, float(energy) + half, side="right"))
    else:
        start = max(0, int(n_energy * 0.14))
        end = min(n_energy, max(start + 1, int(n_energy * 0.17)))
    start = int(max(0, min(n_energy - 1, start)))
    end = int(max(start + 1, min(n_energy, end)))
    return start, end


def _normalise_roi_shape(value: Any) -> str:
    shape = str(value).strip().lower()
    if shape == "circle":
        return "circle"
    if shape in {"ellipse", "oval"}:
        return "ellipse"
    return "rect"


def _normalise_roi(
    rows: int,
    cols: int,
    roi: tuple[int, int, int, int] | None = None,
    *,
    roi_shape: str = "rect",
) -> tuple[int, int, int, int]:
    if roi is None:
        height = max(8, rows // 4)
        width = max(8, cols // 4)
        row = max(0, rows // 2 - height // 2)
        col = max(0, cols // 2 - width // 2)
    else:
        row, col, height, width = (int(v) for v in roi)
    row = int(max(0, min(rows - 1, row)))
    col = int(max(0, min(cols - 1, col)))
    if _normalise_roi_shape(roi_shape) == "circle":
        diameter = int(max(1, max(height, width)))
        diameter = int(min(diameter, rows, cols))
        row = int(max(0, min(rows - diameter, row)))
        col = int(max(0, min(cols - diameter, col)))
        return row, col, diameter, diameter
    height = int(max(1, min(rows - row, height)))
    width = int(max(1, min(cols - col, width)))
    return row, col, height, width


def _round_col_segment(
    row: int,
    roi_row: int,
    roi_col: int,
    height: int,
    width: int,
    *,
    roi_shape: str = "circle",
) -> tuple[int, int] | None:
    r0 = roi_row
    r1 = roi_row + height
    c0 = roi_col
    c1 = roi_col + width
    cx = (c0 + c1) * 0.5
    cy = (r0 + r1) * 0.5
    dy = row + 0.5 - cy
    if _normalise_roi_shape(roi_shape) == "circle":
        radius = max(height, width) * 0.5
        if abs(dy) > radius:
            return None
        half = float(np.sqrt(max(0.0, radius * radius - dy * dy)))
    else:
        ry = max(0.5, height * 0.5)
        rx = max(0.5, width * 0.5)
        norm_y = dy / ry
        if abs(norm_y) > 1:
            return None
        half = float(rx * np.sqrt(max(0.0, 1.0 - norm_y * norm_y)))
    start = max(c0, int(np.ceil(cx - half - 0.5)))
    end = min(c1, int(np.floor(cx + half - 0.5)) + 1)
    return (start, end) if end > start else None


def _spectrum_from_spatial_prefix(
    spatial: np.ndarray,
    row: int,
    col: int,
    height: int,
    width: int,
    *,
    roi_shape: str = "rect",
) -> np.ndarray:
    shape = _normalise_roi_shape(roi_shape)
    if shape == "rect":
        r1 = row + height
        c1 = col + width
        return (
            spatial[r1, c1, :]
            - spatial[row, c1, :]
            - spatial[r1, col, :]
            + spatial[row, col, :]
        ).astype(np.float32)

    out = np.zeros(int(spatial.shape[2]), dtype=np.int64)
    for r in range(row, row + height):
        segment = _round_col_segment(r, row, col, height, width, roi_shape=shape)
        if segment is None:
            continue
        c0, c1 = segment
        out += (
            spatial[r + 1, c1, :].astype(np.int64)
            - spatial[r, c1, :].astype(np.int64)
            - spatial[r + 1, c0, :].astype(np.int64)
            + spatial[r, c0, :].astype(np.int64)
        )
    return out.astype(np.float32)


def load_spectrum_image_sidecar(
    sidecar_dir: str | pathlib.Path,
    *,
    energy: float | None = None,
    width: float | None = None,
    band: tuple[float, float] | tuple[int, int] | None = None,
    roi: tuple[int, int, int, int] | None = None,
    roi_shape: str = "rect",
) -> dict[str, Any]:
    """Load the tiny startup state needed by a browser-only EDS data-folder widget."""

    sidecar = pathlib.Path(sidecar_dir).expanduser()
    meta_path = sidecar / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"ShowEDS data-folder metadata was not found: {meta_path}")
    meta = json.loads(meta_path.read_text())
    if meta.get("format") == "quantem.widget.showeds.stream-sidecar.v1":
        return load_spectrum_stream_sidecar(
            sidecar,
            energy=energy,
            width=width,
            band=band,
            roi=roi,
            roi_shape=roi_shape,
        )
    rows = int(meta["rows"])
    cols = int(meta["cols"])
    n_energy = int(meta["n_energy"])
    axis = np.asarray(meta["energy_keV"], dtype=np.float32)
    if axis.ndim != 1 or axis.size != n_energy:
        raise ValueError(f"data-folder energy axis must be length {n_energy}, got shape {axis.shape}")

    band_start, band_end = _resolve_band_indices(axis, energy=energy, width=width, band=band)
    row, col, height, roi_width = _normalise_roi(rows, cols, roi, roi_shape=roi_shape)

    prefix = np.memmap(sidecar / meta["energy_prefix"], dtype="<u4", mode="r", shape=(n_energy + 1, rows, cols))
    initial_map = (prefix[band_end] - prefix[band_start]).astype(np.float32)

    spatial = np.memmap(sidecar / meta["spatial_prefix"], dtype="<u4", mode="r", shape=(rows + 1, cols + 1, n_energy))
    initial_spectrum = _spectrum_from_spatial_prefix(
        spatial,
        row,
        col,
        height,
        roi_width,
        roi_shape=roi_shape,
    )

    base_name = str(meta.get("base_image", "") or "")
    if base_name:
        base_image = np.asarray(
            np.memmap(sidecar / base_name, dtype="<f4", mode="r", shape=(rows, cols)),
            dtype=np.float32,
        )
    else:
        base_image = initial_map

    return {
        "sidecar_dir": sidecar,
        "meta": meta,
        "energy_keV": axis,
        "initial_map": initial_map,
        "initial_spectrum": initial_spectrum,
        "base_image": base_image,
        "roi": (row, col, height, roi_width),
        "band": (band_start, band_end),
    }


def load_spectrum_stream_sidecar(
    sidecar_dir: str | pathlib.Path,
    *,
    energy: float | None = None,
    width: float | None = None,
    band: tuple[float, float] | tuple[int, int] | None = None,
    roi: tuple[int, int, int, int] | None = None,
    roi_shape: str = "rect",
) -> dict[str, Any]:
    """Load startup state from an exact sparse Velox stream data folder."""

    sidecar = pathlib.Path(sidecar_dir).expanduser()
    meta = json.loads((sidecar / "meta.json").read_text())
    rows = int(meta["rows"])
    cols = int(meta["cols"])
    n_energy = int(meta["n_energy"])
    n_events = int(meta["n_events"])
    axis = np.asarray(meta["energy_keV"], dtype=np.float32)
    if axis.ndim != 1 or axis.size != n_energy:
        raise ValueError(f"stream data-folder energy axis must be length {n_energy}, got shape {axis.shape}")

    band_start, band_end = _resolve_band_indices(axis, energy=energy, width=width, band=band)
    normalised_shape = _normalise_roi_shape(roi_shape)
    row, col, height, roi_width = _normalise_roi(rows, cols, roi, roi_shape=normalised_shape)

    channel_offsets = np.memmap(sidecar / meta["channel_offsets"], dtype="<u4", mode="r", shape=(n_energy + 1,))
    channel_pixels = np.memmap(sidecar / meta["channel_pixels"], dtype="<u4", mode="r", shape=(n_events,))
    pixel_offsets = np.memmap(sidecar / meta["pixel_offsets"], dtype="<u4", mode="r", shape=(rows * cols + 1,))
    pixel_channels = np.memmap(sidecar / meta["pixel_channels"], dtype="<u2", mode="r", shape=(n_events,))
    base_image = np.asarray(np.memmap(sidecar / meta["base_image"], dtype="<f4", mode="r", shape=(rows, cols)), dtype=np.float32)

    initial_map_flat = np.zeros(rows * cols, dtype=np.float32)
    p0 = int(channel_offsets[band_start])
    p1 = int(channel_offsets[band_end])
    np.add.at(initial_map_flat, np.asarray(channel_pixels[p0:p1], dtype=np.intp), 1)
    initial_map = initial_map_flat.reshape(rows, cols)

    initial_spectrum = np.zeros(n_energy, dtype=np.float32)
    for r in range(row, row + height):
        if normalised_shape == "rect":
            segments = [(col, col + roi_width)]
        else:
            segment = _round_col_segment(r, row, col, height, roi_width, roi_shape=normalised_shape)
            segments = [] if segment is None else [segment]
        for c0, c1 in segments:
            for c in range(c0, c1):
                pixel = r * cols + c
                s0 = int(pixel_offsets[pixel])
                s1 = int(pixel_offsets[pixel + 1])
                np.add.at(initial_spectrum, np.asarray(pixel_channels[s0:s1], dtype=np.intp), 1)

    return {
        "sidecar_dir": sidecar,
        "meta": meta,
        "energy_keV": axis,
        "initial_map": initial_map,
        "initial_spectrum": initial_spectrum,
        "base_image": base_image,
        "roi": (row, col, height, roi_width),
        "roi_shape": normalised_shape,
        "band": (band_start, band_end),
    }


def _sum_bin_sidecar_cube(
    sidecar_dir: str | pathlib.Path,
    *,
    spatial_bin: int,
    energy_bin: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Build a portable sum-binned cube from exact data-folder prefix files."""

    sidecar = pathlib.Path(sidecar_dir).expanduser()
    meta = json.loads((sidecar / "meta.json").read_text())
    rows = int(meta["rows"])
    cols = int(meta["cols"])
    n_energy = int(meta["n_energy"])
    spatial_bin = int(max(1, spatial_bin))
    energy_bin = int(max(1, energy_bin))
    out_rows = rows // spatial_bin
    out_cols = cols // spatial_bin
    out_energy = n_energy // energy_bin
    if out_rows <= 0 or out_cols <= 0 or out_energy <= 0:
        raise ValueError(
            f"binning {spatial_bin}x spatial and {energy_bin}x energy is too large for "
            f"shape {(rows, cols, n_energy)}"
        )

    axis = np.asarray(meta["energy_keV"], dtype=np.float32)
    binned_axis = axis[: out_energy * energy_bin].reshape(out_energy, energy_bin).mean(axis=1).astype(np.float32)
    spatial = np.memmap(sidecar / meta["spatial_prefix"], dtype="<u4", mode="r", shape=(rows + 1, cols + 1, n_energy))
    cube = np.empty((out_rows, out_cols, out_energy), dtype=np.uint32)
    c0 = np.arange(out_cols, dtype=np.int64) * spatial_bin
    c1 = c0 + spatial_bin
    max_u32 = np.iinfo(np.uint32).max
    usable_energy = out_energy * energy_bin

    for rr in range(out_rows):
        r0 = rr * spatial_bin
        r1 = r0 + spatial_bin
        block = (
            spatial[r1, c1, :usable_energy].astype(np.uint64)
            - spatial[r0, c1, :usable_energy].astype(np.uint64)
            - spatial[r1, c0, :usable_energy].astype(np.uint64)
            + spatial[r0, c0, :usable_energy].astype(np.uint64)
        )
        if energy_bin > 1:
            block = block.reshape(out_cols, out_energy, energy_bin).sum(axis=2, dtype=np.uint64)
        if block.max(initial=0) > max_u32:
            raise OverflowError("binned EDS export exceeds uint32; smaller bins are required")
        cube[rr] = block.astype(np.uint32)

    base_name = str(meta.get("base_image", "") or "")
    if base_name:
        base = np.asarray(np.memmap(sidecar / base_name, dtype="<f4", mode="r", shape=(rows, cols)), dtype=np.float32)
        base = base[: out_rows * spatial_bin, : out_cols * spatial_bin]
        base = base.reshape(out_rows, spatial_bin, out_cols, spatial_bin).sum(axis=(1, 3), dtype=np.float32)
    else:
        base = cube.sum(axis=2, dtype=np.uint64).astype(np.float32)
    return cube, binned_axis, base


def _dataset_title(dataset: dict[str, Any]) -> str:
    return str(dataset.get("metadata", {}).get("General", {}).get("title", "") or "")


def _energy_axis_from_dataset(dataset: dict[str, Any], n_energy: int) -> np.ndarray | None:
    axes = dataset.get("axes") or []
    best: dict[str, Any] | None = None
    for axis in axes:
        if int(axis.get("size", -1)) != int(n_energy):
            continue
        name = str(axis.get("name", "")).lower()
        units = str(axis.get("units", "")).lower()
        if "energy" in name or "ev" in units:
            best = axis
            break
        if best is None:
            best = axis
    if best is None:
        return None
    scale = float(best.get("scale", 1.0))
    offset = float(best.get("offset", 0.0))
    units = str(best.get("units", "")).lower()
    axis = offset + scale * np.arange(n_energy, dtype=np.float32)
    if units == "ev" or units.endswith(" ev"):
        axis = axis / 1000.0
    return axis.astype(np.float32, copy=False)


def _read_emd_spectrum_image(
    path: str | pathlib.Path,
    *,
    lazy: bool = False,
    candidate_elements: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    try:
        from rsciio.emd import file_reader
    except Exception as exc:  # pragma: no cover - dependency is optional at import time.
        raise ImportError("ShowEDS EMD loading requires rsciio") from exc

    source = pathlib.Path(path).expanduser()
    try:
        datasets = file_reader(str(source), lazy=lazy)
    except TypeError:
        datasets = file_reader(str(source))

    cube_ds: dict[str, Any] | None = None
    base_ds: dict[str, Any] | None = None
    for dataset in datasets:
        data = dataset.get("data")
        ndim = getattr(data, "ndim", None)
        shape = getattr(data, "shape", None)
        if ndim == 3 and (cube_ds is None or int(np.prod(shape)) > int(np.prod(getattr(cube_ds["data"], "shape")))):
            cube_ds = dataset
        title = _dataset_title(dataset).lower()
        if ndim == 2 and base_ds is None and "haadf" in title:
            base_ds = dataset
    if cube_ds is None:
        raise ValueError(f"No 3D spectrum image dataset found in {source}")

    cube = cube_ds["data"]
    source_shape = tuple(int(v) for v in cube.shape)
    if cube.shape[0] > cube.shape[1] and cube.shape[0] > cube.shape[2]:
        cube = np.moveaxis(cube, 0, -1)
    energy_axis = _energy_axis_from_dataset(cube_ds, cube.shape[2])
    if energy_axis is None:
        energy_axis = np.arange(cube.shape[2], dtype=np.float32)
    base = None if base_ds is None else base_ds["data"]
    if base is not None and tuple(int(v) for v in base.shape) != tuple(int(v) for v in cube.shape[:2]):
        base = None

    return {
        "cube": cube,
        "energy_keV": energy_axis,
        "base_image": base,
        "title": _dataset_title(cube_ds) or source.stem,
        "candidate_elements": list(candidate_elements or ["O", "Si", "Ca", "Au"]),
        "source_shape": source_shape,
        "path": str(source),
    }


def load_emd_spectrum_image(
    path: str | pathlib.Path,
    *,
    spatial_bin: int = 1,
    energy_bin: int = 1,
    candidate_elements: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Load a Velox/RSCIIO EDS spectrum image for :class:`ShowEDS`.

    Returns a dictionary containing ``cube``, ``energy_keV``, ``base_image``,
    ``title``, ``candidate_elements``, and ``source_shape``. The returned cube is
    always ``(row, col, energy)``. Optional ``spatial_bin`` and ``energy_bin``
    perform count-preserving sum binning when explicitly requested.
    """

    loaded = _read_emd_spectrum_image(path, lazy=False, candidate_elements=candidate_elements)
    cube = np.asarray(loaded["cube"])
    energy_axis = np.asarray(loaded["energy_keV"], dtype=np.float32)
    base = None if loaded["base_image"] is None else np.asarray(loaded["base_image"])

    binned_cube, binned_energy, binned_base = bin_spectrum_image(
        cube,
        energy_axis,
        base_image=base,
        spatial_bin=spatial_bin,
        energy_bin=energy_bin,
    )
    return {
        "cube": binned_cube,
        "energy_keV": binned_energy,
        "base_image": binned_base,
        "title": loaded["title"],
        "candidate_elements": loaded["candidate_elements"],
        "source_shape": loaded["source_shape"],
        "binned_shape": tuple(int(v) for v in binned_cube.shape),
        "spatial_bin": int(max(1, spatial_bin)),
        "energy_bin": int(max(1, energy_bin)),
        "path": loaded["path"],
    }


def _normalise_emd_backend(value: str | None) -> str:
    raw = str(value or "auto").strip().lower().replace("_", "-")
    aliases = {
        "": "auto",
        "data-folder": "sidecar",
        "folder": "sidecar",
        "linked-folder": "sidecar",
        "lazy": "kernel",
        "native": "kernel",
    }
    backend = aliases.get(raw, raw)
    if backend not in {"auto", "sidecar", "kernel"}:
        raise ValueError("ShowEDS.from_emd backend must be 'auto', 'sidecar', or 'kernel'")
    return backend


def _compute_numpy(value: Any, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    if hasattr(value, "compute"):
        value = value.compute()
    return np.asarray(to_numpy(value), dtype=dtype)


def _masked_lazy_spectrum(
    cube: Any,
    row: int,
    col: int,
    height: int,
    width: int,
    *,
    roi_shape: str,
) -> Any:
    shape = _normalise_roi_shape(roi_shape)
    subset = cube[row : row + height, col : col + width, :]
    if shape == "rect":
        return subset.sum(axis=(0, 1))

    mask = np.zeros((height, width), dtype=np.float32)
    for rr in range(height):
        segment = _round_col_segment(rr + row, row, col, height, width, roi_shape=shape)
        if segment is None:
            continue
        c0, c1 = segment
        mask[rr, c0 - col : c1 - col] = 1
    return (subset * mask[:, :, None]).sum(axis=(0, 1))


def _initial_lazy_emd_state(
    cube: Any,
    energy_keV: np.ndarray,
    base_image: Any | None,
    *,
    energy: float | None,
    width: float | None,
    band: tuple[float, float] | tuple[int, int] | None,
    roi: tuple[int, int, int, int] | None,
    roi_shape: str,
) -> dict[str, Any]:
    rows, cols, n_energy = (int(v) for v in cube.shape)
    axis = np.asarray(energy_keV, dtype=np.float32)
    if axis.ndim != 1 or axis.size != n_energy:
        raise ValueError(f"energy_keV must be length {n_energy}, got shape {axis.shape}")

    band_start, band_end = _resolve_band_indices(axis, energy=energy, width=width, band=band)
    normalised_shape = _normalise_roi_shape(roi_shape)
    row, col, roi_height, roi_width = _normalise_roi(rows, cols, roi, roi_shape=normalised_shape)

    initial_map = _compute_numpy(cube[:, :, band_start:band_end].sum(axis=2), dtype=np.float32)
    spectrum = _masked_lazy_spectrum(
        cube,
        row,
        col,
        roi_height,
        roi_width,
        roi_shape=normalised_shape,
    )
    initial_spectrum = _compute_numpy(spectrum, dtype=np.float32)

    if base_image is None:
        base = initial_map
    else:
        base = _compute_numpy(base_image, dtype=np.float32)
        if base.shape != (rows, cols):
            base = initial_map

    return {
        "initial_map": initial_map,
        "initial_spectrum": initial_spectrum,
        "base_image": base,
        "band": (band_start, band_end),
        "roi": (row, col, roi_height, roi_width),
        "roi_shape": normalised_shape,
    }


class ShowEDS(anywidget.AnyWidget):
    """Explore an EDS/EELS spectrum image ``(row, col, energy)``.

    For browser-backed widgets, the frontend keeps the spectrum cube resident in
    WebGPU. ``ShowEDS.from_emd(...)`` keeps native EMD files lazy by default for
    exact no-bin work: the source file remains the query backend, and only the
    current map or ROI spectrum is read.
    """

    _esm = pathlib.Path(__file__).parent / "static" / "showeds.js"

    DEFAULT_MAX_STATE_BYTES = 512 * 1024**2

    widget_version = traitlets.Unicode("unknown").tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    n_rows = traitlets.Int(1).tag(sync=True)
    n_cols = traitlets.Int(1).tag(sync=True)
    n_energy = traitlets.Int(1).tag(sync=True)
    cube_bytes = traitlets.Bytes(b"").tag(sync=True)
    cube_dtype = traitlets.Unicode("float32").tag(sync=True)
    base_image_bytes = traitlets.Bytes(b"").tag(sync=True)
    initial_map_bytes = traitlets.Bytes(b"").tag(sync=True)
    initial_spectrum_bytes = traitlets.Bytes(b"").tag(sync=True)
    energy_keV = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    compute_backend = traitlets.Unicode("browser").tag(sync=True)
    sidecar_url = traitlets.Unicode("").tag(sync=True)

    band_start = traitlets.Int(0).tag(sync=True)
    band_end = traitlets.Int(1).tag(sync=True)
    roi_row = traitlets.Int(0).tag(sync=True)
    roi_col = traitlets.Int(0).tag(sync=True)
    roi_height = traitlets.Int(1).tag(sync=True)
    roi_width = traitlets.Int(1).tag(sync=True)
    roi_shape = traitlets.Unicode("rect").tag(sync=True)

    panel_width_px = traitlets.Int(420).tag(sync=True)
    spectrum_width_px = traitlets.Int(640).tag(sync=True)
    spectrum_height_px = traitlets.Int(250).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)
    log_spectrum = traitlets.Bool(False).tag(sync=True)
    smooth = traitlets.Bool(False).tag(sync=True)
    pixel_size = traitlets.Float(0.0).tag(sync=True)
    pixel_unit = traitlets.Unicode("px").tag(sync=True)
    scale_bar_visible = traitlets.Bool(True).tag(sync=True)
    map_vmin_pct = traitlets.Float(2.0).tag(sync=True)
    map_vmax_pct = traitlets.Float(98.0).tag(sync=True)
    overlay_opacity = traitlets.Float(0.65).tag(sync=True)
    map_zoom = traitlets.Float(1.0).tag(sync=True)
    map_view_row = traitlets.Float(0.0).tag(sync=True)
    map_view_col = traitlets.Float(0.0).tag(sync=True)
    spectrum_view_start = traitlets.Float(0.0).tag(sync=True)
    spectrum_view_end = traitlets.Float(1.0).tag(sync=True)
    element_label = traitlets.Unicode("").tag(sync=True)
    show_line_hints = traitlets.Bool(True).tag(sync=True)
    line_hints = traitlets.List(traitlets.Dict(), default_value=[]).tag(sync=True)
    selected_elements = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    auto_identify = traitlets.Bool(True).tag(sync=True)
    show_debug = traitlets.Bool(False).tag(sync=True)
    saved_rois = traitlets.List(traitlets.Dict(), default_value=[]).tag(sync=True)
    saved_bands = traitlets.List(traitlets.Dict(), default_value=[]).tag(sync=True)
    export_presets = traitlets.List(traitlets.Dict(), default_value=[]).tag(sync=True)
    export_request = traitlets.Unicode("").tag(sync=True)
    export_status = traitlets.Unicode("").tag(sync=True)
    export_enabled = traitlets.Bool(True).tag(sync=True)
    export_payload = traitlets.Bytes(b"").tag(sync=True)
    export_payload_id = traitlets.Unicode("").tag(sync=True)
    export_filename = traitlets.Unicode("").tag(sync=True)
    export_sidecar_bytes = traitlets.Int(0).tag(sync=True)
    # True only while writing standalone HTML: exported files have no host
    # notebook theme, so match Show2D/Show3D and force readable light chrome.
    _export_light = traitlets.Bool(False).tag(sync=True)

    def __init__(
        self,
        cube: np.ndarray | None,
        energy_keV: np.ndarray | list[float] | None = None,
        *,
        title: str = "",
        base_image: np.ndarray | None = None,
        energy: float | None = None,
        width: float | None = None,
        band: tuple[float, float] | tuple[int, int] | None = None,
        roi: tuple[int, int, int, int] | None = None,
        roi_shape: str = "rect",
        panel_width_px: int = 420,
        spectrum_width_px: int | None = None,
        spectrum_height_px: int | None = None,
        show_controls: bool = True,
        log_spectrum: bool = False,
        smooth: bool = False,
        pixel_size: float | None = None,
        pixel_unit: str = "px",
        sampling: float | tuple[float, float] | list[float] | None = None,
        units: str | list[str] | None = None,
        scale_bar_visible: bool = True,
        map_vmin_pct: float = 2.0,
        map_vmax_pct: float = 98.0,
        overlay_opacity: float = 0.65,
        element_label: str = "",
        show_line_hints: bool = True,
        line_hints: list[dict[str, Any]] | None = None,
        selected_elements: list[str] | tuple[str, ...] | None = None,
        auto_identify: bool = True,
        show_debug: bool = False,
        saved_rois: list[dict[str, Any]] | None = None,
        saved_bands: list[dict[str, Any]] | None = None,
        export_presets: list[dict[str, Any]] | None = None,
        candidate_elements: list[str] | tuple[str, ...] | None = None,
        state: dict[str, Any] | str | pathlib.Path | None = None,
        max_state_bytes: int | None = DEFAULT_MAX_STATE_BYTES,
        initial_map: np.ndarray | None = None,
        initial_spectrum: np.ndarray | None = None,
        lazy_path: str | pathlib.Path | None = None,
        sidecar_url: str = "",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.widget_version = resolve_widget_version()
        self._lazy_path = pathlib.Path(lazy_path).expanduser() if lazy_path is not None else None
        self._lazy_cube: Any | None = None
        self._lazy_loading = False
        self._sidecar_dir: pathlib.Path | None = _sidecar_path_from_url(sidecar_url) if sidecar_url else None

        data: np.ndarray | None = None
        if cube is None:
            if initial_map is None or initial_spectrum is None:
                raise ValueError("ShowEDS requires cube, or initial_map and initial_spectrum for kernel-backed lazy mode")
            initial_map_arr = np.asarray(to_numpy(initial_map), dtype=np.float32)
            initial_spectrum_arr = np.asarray(to_numpy(initial_spectrum), dtype=np.float32).reshape(-1)
            if initial_map_arr.ndim != 2:
                raise ValueError(f"initial_map must be 2D, got {initial_map_arr.ndim}D")
            rows, cols = initial_map_arr.shape
            n_energy = int(initial_spectrum_arr.size)
            self.compute_backend = "sidecar" if sidecar_url else "kernel"
            self.sidecar_url = sidecar_url
        else:
            raw_data = np.asarray(to_numpy(cube))
            if raw_data.ndim != 3:
                raise ValueError(f"ShowEDS expects a 3D cube (row, col, energy), got {raw_data.ndim}D")
            if np.issubdtype(raw_data.dtype, np.integer) and np.nanmin(raw_data) >= 0:
                max_count = np.nanmax(raw_data)
                if max_count <= np.iinfo(np.uint16).max:
                    cube_storage_dtype = np.dtype(np.uint16)
                    self.cube_dtype = "uint16"
                elif max_count <= np.iinfo(np.uint32).max:
                    cube_storage_dtype = np.dtype(np.uint32)
                    self.cube_dtype = "uint32"
                else:
                    cube_storage_dtype = np.dtype(np.float32)
                    self.cube_dtype = "float32"
            else:
                cube_storage_dtype = np.dtype(np.float32)
                self.cube_dtype = "float32"
            outgoing_cube_bytes = int(raw_data.size * cube_storage_dtype.itemsize)
            if max_state_bytes is not None and outgoing_cube_bytes > int(max_state_bytes):
                raise ValueError(
                    "ShowEDS browser mode embeds an in-browser copy of the spectrum image. "
                    f"The requested cube shape {tuple(int(v) for v in raw_data.shape)} would send "
                    f"{_format_bytes(outgoing_cube_bytes)} as a widget buffer, above the "
                    f"{_format_bytes(int(max_state_bytes))} safety limit. Use ShowEDS.from_emd(...) "
                    "for native EMD data so the file stays as the exact lazy query source. Use an "
                    "explicit spatial_bin only when you intentionally want a portable sharing cache, "
                    "or pass max_state_bytes=None only to bypass this guard for debugging."
                )
            data = np.asarray(raw_data, dtype=cube_storage_dtype)
            rows, cols, n_energy = data.shape
            initial_map_arr = None
            initial_spectrum_arr = None
            self.compute_backend = "browser"
        self.n_rows = int(rows)
        self.n_cols = int(cols)
        self.n_energy = int(n_energy)
        self.title = title
        self.show_controls = bool(show_controls)
        self.log_spectrum = bool(log_spectrum)
        self.smooth = bool(smooth)
        if pixel_size is None:
            if sampling is None:
                self.pixel_size = 0.0
            elif isinstance(sampling, (int, float)):
                self.pixel_size = float(sampling)
            else:
                self.pixel_size = float(sampling[-1])
        else:
            self.pixel_size = float(max(0.0, pixel_size))
        if units is not None:
            self.pixel_unit = units if isinstance(units, str) else str(units[-1])
        else:
            self.pixel_unit = str(pixel_unit)
        self.scale_bar_visible = bool(scale_bar_visible)
        self.map_vmin_pct = float(max(0.0, min(100.0, map_vmin_pct)))
        self.map_vmax_pct = float(max(self.map_vmin_pct, min(100.0, map_vmax_pct)))
        self.overlay_opacity = float(max(0.0, min(1.0, overlay_opacity)))
        self.element_label = str(element_label)
        self.show_line_hints = bool(show_line_hints)
        self.selected_elements = _normalise_element_symbols(selected_elements or candidate_elements or [])
        self.auto_identify = bool(auto_identify)
        self.show_debug = bool(show_debug)
        self.roi_shape = _normalise_roi_shape(roi_shape)
        self.saved_rois = [{**dict(item), "shape": _normalise_roi_shape(dict(item).get("shape", "rect"))} for item in (saved_rois or [])]
        self.saved_bands = [dict(item) for item in (saved_bands or [])]
        self.export_presets = [dict(item) for item in (export_presets or [])]

        if energy_keV is None:
            axis = np.arange(n_energy, dtype=np.float32)
        else:
            axis = np.asarray(energy_keV, dtype=np.float32)
            if axis.ndim != 1 or axis.size != n_energy:
                raise ValueError(f"energy_keV must be length {n_energy}, got shape {axis.shape}")
        self.energy_keV = [float(v) for v in axis]
        energy_min = float(np.nanmin(axis)) if axis.size else 0.0
        energy_max = float(np.nanmax(axis)) if axis.size else float(n_energy - 1)
        self.line_hints = (
            [dict(item) for item in line_hints]
            if line_hints is not None
            else eds_line_hints(energy_min, energy_max, elements=candidate_elements)
        )

        if base_image is None and data is not None:
            base = data.sum(axis=2)
        elif base_image is None:
            base = initial_map_arr
        else:
            base = np.asarray(to_numpy(base_image), dtype=np.float32)
            if base.shape != (rows, cols):
                raise ValueError(f"base_image must have shape {(rows, cols)}, got {base.shape}")

        if data is None:
            self.cube_bytes = b""
        elif self.cube_dtype == "uint16":
            self.cube_bytes = np.ascontiguousarray(data, dtype=np.uint16).tobytes()
        elif self.cube_dtype == "uint32":
            self.cube_bytes = np.ascontiguousarray(data, dtype=np.uint32).tobytes()
        else:
            self.cube_bytes = np.ascontiguousarray(data, dtype=np.float32).tobytes()
        self.base_image_bytes = np.ascontiguousarray(base, dtype=np.float32).tobytes()
        if initial_map_arr is not None:
            self.initial_map_bytes = np.ascontiguousarray(initial_map_arr, dtype=np.float32).tobytes()
        if initial_spectrum_arr is not None:
            self.initial_spectrum_bytes = np.ascontiguousarray(initial_spectrum_arr, dtype=np.float32).tobytes()
        if self._sidecar_dir is not None and self._sidecar_dir.exists():
            self.export_sidecar_bytes = _directory_size(self._sidecar_dir)

        self.panel_width_px = int(max(180, panel_width_px))
        if spectrum_width_px is None:
            spectrum_width_px = self.panel_width_px + 220
        self.spectrum_width_px = int(max(320, spectrum_width_px))
        if spectrum_height_px is None:
            spectrum_height_px = int(round(self.panel_width_px * 0.58))
        self.spectrum_height_px = int(max(140, spectrum_height_px))
        self.map_zoom = 1.0
        self.map_view_row = 0.0
        self.map_view_col = 0.0
        self.spectrum_view_start = 0.0
        self.spectrum_view_end = float(n_energy)
        if band is not None:
            start, end = self._resolve_band(axis, band)
        elif energy is not None:
            span = float(np.ptp(axis))
            half = float(width if width is not None else max(span / max(n_energy, 1) * 8, 0.1)) / 2
            start, end = self._resolve_band(axis, (float(energy) - half, float(energy) + half))
        else:
            start = max(0, int(n_energy * 0.14))
            end = min(n_energy, max(start + 1, int(n_energy * 0.17)))
        self.band_start = int(start)
        self.band_end = int(max(start + 1, min(n_energy, end)))

        if roi is None:
            h = max(8, rows // 4)
            w = max(8, cols // 4)
            r = max(0, rows // 2 - h // 2)
            c = max(0, cols // 2 - w // 2)
        else:
            r, c, h, w = (int(v) for v in roi)
        self.roi_row, self.roi_col, self.roi_height, self.roi_width = _normalise_roi(
            rows,
            cols,
            (r, c, h, w),
            roi_shape=self.roi_shape,
        )

        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(json.loads(pathlib.Path(state).read_text()), require_envelope=True)
            else:
                state = unwrap_state_payload(state)
            self.load_state_dict(state)

        if self.compute_backend == "kernel":
            self.on_msg(self._handle_kernel_compute_msg)
        self.observe(self._on_export_request_change, names=["export_request"])

    @classmethod
    def from_sidecar(
        cls,
        sidecar_url: str,
        *,
        sidecar_dir: str | pathlib.Path | None = None,
        initial_map: np.ndarray | None = None,
        initial_spectrum: np.ndarray | None = None,
        energy_keV: np.ndarray | list[float] | None = None,
        title: str = "",
        base_image: np.ndarray | None = None,
        energy: float | None = None,
        width: float | None = None,
        element_label: str = "",
        candidate_elements: list[str] | tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> "ShowEDS":
        """Create a browser-only widget from a prepared data-folder URL."""

        url = _normalise_sidecar_url(sidecar_url)
        sidecar_path = pathlib.Path(sidecar_dir).expanduser() if sidecar_dir is not None else _sidecar_path_from_url(url)
        if initial_map is None or initial_spectrum is None or energy_keV is None:
            if sidecar_path is None:
                raise ValueError(
                    "sidecar_dir is required when initial_map, initial_spectrum, energy_keV, or base_image "
                    "are not supplied and sidecar_url is not a Jupyter /files/ data-folder URL"
                )
            startup = load_spectrum_image_sidecar(
                sidecar_path,
                energy=energy,
                width=width,
                band=kwargs.get("band"),
                roi=kwargs.get("roi"),
                roi_shape=kwargs.get("roi_shape", "rect"),
            )
            if initial_map is None:
                initial_map = startup["initial_map"]
            if initial_spectrum is None:
                initial_spectrum = startup["initial_spectrum"]
            if energy_keV is None:
                energy_keV = startup["energy_keV"]
            if base_image is None:
                base_image = startup["base_image"]
            kwargs.setdefault("roi", startup["roi"])
            kwargs.setdefault("roi_shape", startup.get("roi_shape", kwargs.get("roi_shape", "rect")))
        widget = cls(
            None,
            energy_keV,
            title=title,
            base_image=base_image,
            initial_map=initial_map,
            initial_spectrum=initial_spectrum,
            sidecar_url=url,
            energy=energy,
            width=width,
            element_label=element_label,
            candidate_elements=candidate_elements,
            **kwargs,
        )
        widget._sidecar_dir = sidecar_path
        if sidecar_path is not None and sidecar_path.exists():
            widget.export_sidecar_bytes = _directory_size(sidecar_path)
        return widget

    @classmethod
    def from_emd(
        cls,
        path: str | pathlib.Path,
        *,
        backend: str = "auto",
        sidecar_dir: str | pathlib.Path | None = None,
        sidecar_url: str | None = None,
        rebuild_sidecar: bool = False,
        energy_chunk: int = 256,
        spatial_bin: int = 1,
        energy_bin: int = 1,
        max_sidecar_bytes: int | None = DEFAULT_MAX_SHOWEDS_SIDECAR_BYTES,
        title: str | None = None,
        energy: float = 8.04,
        width: float = 0.24,
        element_label: str = "Cu K",
        candidate_elements: list[str] | tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> "ShowEDS":
        """Open a Velox/RSCIIO EMD spectrum image.

        ``backend="auto"`` uses an existing data folder if one is present.
        Otherwise exact no-bin EMD stays native/lazy: the EMD file remains the
        query source, and the widget reads only the current band map or ROI
        spectrum. A prefix-cache data folder is built only when the caller asks
        for ``backend="sidecar"`` or explicit binning for a portable viewer.
        """

        source = pathlib.Path(path).expanduser()
        backend_mode = _normalise_emd_backend(backend)
        spatial_bin_value = int(max(1, spatial_bin))
        energy_bin_value = int(max(1, energy_bin))
        wants_binned_cache = spatial_bin_value > 1 or energy_bin_value > 1
        sidecar_path = (
            pathlib.Path(sidecar_dir).expanduser()
            if sidecar_dir is not None
            else pathlib.Path.cwd() / "sidecars" / source.stem
        )
        meta_path = sidecar_path / "meta.json"
        has_existing_cache = (not rebuild_sidecar) and meta_path.exists()
        use_native_query = backend_mode == "kernel" or (
            backend_mode == "auto" and not has_existing_cache and not wants_binned_cache
        )

        if backend_mode == "kernel" and wants_binned_cache:
            raise ValueError(
                "ShowEDS.from_emd backend='kernel' keeps the native EMD exact and currently "
                "does not apply spatial_bin or energy_bin. Use backend='sidecar' for a portable "
                "sum-binned data folder."
            )

        if use_native_query:
            loaded = _read_emd_spectrum_image(source, lazy=True, candidate_elements=candidate_elements)
            cube = loaded["cube"]
            axis = loaded["energy_keV"]
            base = loaded["base_image"]
            startup = _initial_lazy_emd_state(
                cube,
                axis,
                base,
                energy=energy,
                width=width,
                band=kwargs.get("band"),
                roi=kwargs.get("roi"),
                roi_shape=kwargs.get("roi_shape", "rect"),
            )
            widget_kwargs = dict(kwargs)
            widget_kwargs["band"] = startup["band"]
            widget_kwargs["roi"] = startup["roi"]
            widget_kwargs["roi_shape"] = startup["roi_shape"]
            return cls(
                None,
                axis,
                title=title or loaded["title"] or source.stem,
                base_image=startup["base_image"],
                initial_map=startup["initial_map"],
                initial_spectrum=startup["initial_spectrum"],
                lazy_path=source,
                element_label=element_label,
                candidate_elements=list(candidate_elements or ["O", "Si", "Ca", "Cu", "Au"]),
                **widget_kwargs,
            )

        if rebuild_sidecar or not meta_path.exists():
            loaded = _read_emd_spectrum_image(source, lazy=True, candidate_elements=candidate_elements)
            cube = loaded["cube"]
            axis = loaded["energy_keV"]
            base = loaded["base_image"]
            sidecar_shape = _binned_spectrum_image_shape(
                tuple(int(v) for v in cube.shape),
                spatial_bin=spatial_bin_value,
                energy_bin=energy_bin_value,
            )
            estimated_bytes = _estimate_spectrum_image_sidecar_bytes(
                sidecar_shape,
                include_base_image=base is not None,
            )
            sidecar_too_large = max_sidecar_bytes is not None and estimated_bytes > int(max_sidecar_bytes)
            if sidecar_too_large:
                raise ValueError(
                    "ShowEDS.from_emd cannot create the requested exact prefix-cache data folder within "
                    "the configured safety limit. "
                    f"Source shape {tuple(int(v) for v in cube.shape)} with spatial_bin={spatial_bin_value} "
                    f"and energy_bin={energy_bin_value} would create cache shape {sidecar_shape}, estimated at "
                    f"{_format_bytes(estimated_bytes)}, above the {_format_bytes(int(max_sidecar_bytes))} safety limit. "
                    "Use the native lazy backend for exact no-bin analysis, increase spatial_bin for a smaller "
                    "portable cache, or pass max_sidecar_bytes=None only for an intentional local experiment."
                )
            if wants_binned_cache:
                binned_cube, binned_axis, binned_base = _sum_bin_spectrum_image_lazy(
                    cube,
                    axis,
                    base_image=base,
                    spatial_bin=spatial_bin_value,
                    energy_bin=energy_bin_value,
                )
                cube = binned_cube
                axis = binned_axis
                base = binned_base
            prepare_spectrum_image_sidecar(
                cube,
                axis,
                sidecar_path,
                base_image=None if base is None else base,
                energy_chunk=energy_chunk,
                max_sidecar_bytes=max_sidecar_bytes,
            )

        url = _normalise_sidecar_url(sidecar_url) if sidecar_url is not None else _sidecar_url_from_path(sidecar_path)
        return cls.from_sidecar(
            url,
            sidecar_dir=sidecar_path,
            title=title or source.stem,
            energy=energy,
            width=width,
            element_label=element_label,
            candidate_elements=list(candidate_elements or ["O", "Si", "Ca", "Cu", "Au"]),
            **kwargs,
        )

    @staticmethod
    def _resolve_band(axis: np.ndarray, band: tuple[float, float] | tuple[int, int]) -> tuple[int, int]:
        a, b = band
        if isinstance(a, int) and isinstance(b, int):
            return int(min(a, b)), int(max(a, b))
        lo = float(min(a, b))
        hi = float(max(a, b))
        start = int(np.searchsorted(axis, lo, side="left"))
        end = int(np.searchsorted(axis, hi, side="right"))
        return start, end

    def _ensure_lazy_cube(self) -> Any:
        if self._lazy_cube is not None:
            return self._lazy_cube
        if self._lazy_path is None:
            raise RuntimeError("No lazy EDS source is attached to this widget")
        if self._lazy_loading:
            raise RuntimeError("Lazy EDS source is already loading")
        self._lazy_loading = True
        try:
            self._lazy_cube = _read_emd_spectrum_image(self._lazy_path, lazy=True)["cube"]
            return self._lazy_cube
        finally:
            self._lazy_loading = False

    def _handle_kernel_compute_msg(self, _widget: Any, content: dict[str, Any], _buffers: list[Any]) -> None:
        msg_type = content.get("type")
        request_id = content.get("request_id")
        try:
            cube = self._ensure_lazy_cube()
            if msg_type == "compute_map":
                start = int(max(0, min(self.n_energy - 1, round(float(content.get("start", self.band_start))))))
                end = int(max(start + 1, min(self.n_energy, round(float(content.get("end", self.band_end))))))
                arr = np.asarray(cube[:, :, start:end].sum(axis=2).compute(), dtype=np.float32)
                payload_type = "map"
            elif msg_type == "compute_spectrum":
                row = int(max(0, min(self.n_rows - 1, round(float(content.get("row", self.roi_row))))))
                col = int(max(0, min(self.n_cols - 1, round(float(content.get("col", self.roi_col))))))
                height = int(max(1, min(self.n_rows - row, round(float(content.get("height", self.roi_height))))))
                width = int(max(1, min(self.n_cols - col, round(float(content.get("width", self.roi_width))))))
                roi_shape = _normalise_roi_shape(content.get("shape", self.roi_shape))
                subset = cube[row : row + height, col : col + width, :]
                if roi_shape != "rect":
                    yy, xx = np.ogrid[:height, :width]
                    cy = height * 0.5
                    cx = width * 0.5
                    if roi_shape == "circle":
                        radius = max(height, width) * 0.5
                        mask = ((yy + 0.5 - cy) ** 2 + (xx + 0.5 - cx) ** 2) <= radius**2
                    else:
                        ry = max(0.5, height * 0.5)
                        rx = max(0.5, width * 0.5)
                        mask = (((yy + 0.5 - cy) / ry) ** 2 + ((xx + 0.5 - cx) / rx) ** 2) <= 1.0
                    reduced = (subset * mask[:, :, None]).sum(axis=(0, 1))
                else:
                    reduced = subset.sum(axis=(0, 1))
                if hasattr(reduced, "compute"):
                    reduced = reduced.compute()
                arr = np.asarray(reduced, dtype=np.float32)
                payload_type = "spectrum"
            else:
                return
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            self.send(
                {"type": payload_type, "request_id": request_id, "shape": list(arr.shape)},
                buffers=[memoryview(arr).cast("B")],
            )
        except Exception as exc:
            self.send({"type": "error", "request_id": request_id, "message": str(exc)})

    def state_dict(self) -> dict[str, Any]:
        return {
            "_widget": "ShowEDS",
            "title": self.title,
            "band_start": self.band_start,
            "band_end": self.band_end,
            "roi_row": self.roi_row,
            "roi_col": self.roi_col,
            "roi_height": self.roi_height,
            "roi_width": self.roi_width,
            "roi_shape": self.roi_shape,
            "panel_width_px": self.panel_width_px,
            "spectrum_width_px": self.spectrum_width_px,
            "spectrum_height_px": self.spectrum_height_px,
            "show_controls": self.show_controls,
            "log_spectrum": self.log_spectrum,
            "smooth": self.smooth,
            "pixel_size": self.pixel_size,
            "pixel_unit": self.pixel_unit,
            "scale_bar_visible": self.scale_bar_visible,
            "map_vmin_pct": self.map_vmin_pct,
            "map_vmax_pct": self.map_vmax_pct,
            "overlay_opacity": self.overlay_opacity,
            "map_zoom": self.map_zoom,
            "map_view_row": self.map_view_row,
            "map_view_col": self.map_view_col,
            "spectrum_view_start": self.spectrum_view_start,
            "spectrum_view_end": self.spectrum_view_end,
            "element_label": self.element_label,
            "show_line_hints": self.show_line_hints,
            "selected_elements": list(self.selected_elements),
            "auto_identify": self.auto_identify,
            "show_debug": self.show_debug,
            "saved_rois": [dict(item) for item in self.saved_rois],
            "saved_bands": [dict(item) for item in self.saved_bands],
            "export_presets": [dict(item) for item in self.export_presets],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        state = dict(state)
        state.pop("_widget", None)
        allowed = set(self.state_dict().keys())
        for key, value in state.items():
            if key in allowed and self.has_trait(key):
                setattr(self, key, value)
        self.band_start = int(max(0, min(self.n_energy - 1, self.band_start)))
        self.band_end = int(max(self.band_start + 1, min(self.n_energy, self.band_end)))
        self.roi_shape = _normalise_roi_shape(self.roi_shape)
        self.roi_row, self.roi_col, self.roi_height, self.roi_width = _normalise_roi(
            self.n_rows,
            self.n_cols,
            (self.roi_row, self.roi_col, self.roi_height, self.roi_width),
            roi_shape=self.roi_shape,
        )
        self.pixel_size = float(max(0.0, self.pixel_size))
        self.pixel_unit = str(self.pixel_unit or "px")
        self.scale_bar_visible = bool(self.scale_bar_visible)
        self.smooth = bool(self.smooth)
        self.map_zoom = float(max(1.0, min(32.0, self.map_zoom)))
        view_rows = self.n_rows / self.map_zoom
        view_cols = self.n_cols / self.map_zoom
        self.map_view_row = float(max(0.0, min(max(0.0, self.n_rows - view_rows), self.map_view_row)))
        self.map_view_col = float(max(0.0, min(max(0.0, self.n_cols - view_cols), self.map_view_col)))
        min_span = min(8.0, float(max(1, self.n_energy)))
        self.spectrum_view_start = float(max(0.0, min(max(0.0, self.n_energy - min_span), self.spectrum_view_start)))
        self.spectrum_view_end = float(max(self.spectrum_view_start + min_span, min(float(self.n_energy), self.spectrum_view_end)))
        self.saved_rois = [{**dict(item), "shape": _normalise_roi_shape(dict(item).get("shape", "rect"))} for item in self.saved_rois]
        self.saved_bands = [dict(item) for item in self.saved_bands]
        self.export_presets = [dict(item) for item in self.export_presets]
        self.selected_elements = _normalise_element_symbols(self.selected_elements)
        self.auto_identify = bool(self.auto_identify)

    def save(self, path: str) -> None:
        save_state_file(path, "ShowEDS", self.state_dict())

    def export_html(
        self,
        path: str | pathlib.Path | None = None,
        *,
        title: str | None = None,
        mode: str = "single",
        encoding: str = "full",
        downsample: int | None = None,
        binning: int | None = None,
    ) -> pathlib.Path:
        """Write a standalone HTML explorer for this EDS/EELS widget.

        ``mode="single"`` writes one HTML file. ``mode="folder"`` writes an
        HTML file that references an existing exact data folder. Use
        ``downsample=2`` or ``downsample=4`` for a smaller one-file export that
        sum-bins rows, columns, and energy channels. ``binning`` is kept as a
        compatibility alias for older notebooks.
        """
        export_path = pathlib.Path(path) if path is not None else self._default_html_export_path()
        widget, mode_label = self._export_widget_for_mode(
            mode,
            encoding=encoding,
            downsample=downsample,
            binning=binning,
        )
        self._write_html_export(export_path, title=title, widget=widget)
        size_mb = export_path.stat().st_size / (1024 * 1024)
        self.export_status = f"Exported {export_path.name} ({size_mb:.1f} MB, {mode_label})"
        return export_path

    def _write_html_export(
        self,
        path: str | pathlib.Path,
        *,
        title: str | None = None,
        widget: "ShowEDS" | None = None,
    ) -> pathlib.Path:
        from ipywidgets.embed import dependency_state, embed_minimal_html

        from .export import ensure_mobile_viewport

        export_widget = widget or self
        export_path = pathlib.Path(path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        previous_export_light = export_widget._export_light
        export_widget._export_light = True
        try:
            embed_minimal_html(
                str(export_path),
                views=[export_widget],
                title=title or export_widget.title or self.title or "ShowEDS",
                drop_defaults=False,
                state=dependency_state([export_widget], drop_defaults=False),
            )
        finally:
            export_widget._export_light = previous_export_light
        ensure_mobile_viewport(export_path)
        return export_path

    def _html_export_bytes(
        self,
        *,
        mode: str = "single",
        encoding: str = "full",
        downsample: int | str | None = None,
        binning: int | str | None = None,
    ) -> tuple[bytes, str]:
        widget, mode_label = self._export_widget_for_mode(
            mode,
            encoding=encoding,
            downsample=downsample,
            binning=binning,
        )
        with tempfile.TemporaryDirectory(prefix="showeds-export-") as tmp:
            path = pathlib.Path(tmp) / self._default_html_export_path().name
            self._write_html_export(path, widget=widget)
            return path.read_bytes(), mode_label

    def _normalise_export_request(
        self,
        mode: str | None,
        *,
        encoding: str = "full",
        downsample: int | str | None = None,
        binning: int | str | None = None,
    ) -> tuple[str, int | None]:
        raw_mode = str(mode or "single").strip().lower().replace("_", "-")
        if downsample not in (None, "", 0, "0") and binning not in (None, "", 0, "0"):
            if int(downsample) != int(binning):
                raise ValueError("Specify either downsample or binning, not conflicting values")
        requested_downsample = downsample if downsample not in (None, "", 0, "0") else binning
        requested_binning = None if requested_downsample in (None, "", 0, "0") else int(requested_downsample)
        if raw_mode.startswith("binned-"):
            requested_binning = int(raw_mode.split("-", 1)[1])
            raw_mode = "single"
        raw_encoding = str(encoding or "full").strip().lower().replace("_", "-")
        if raw_encoding not in {"full", "exact", "auto"}:
            raise ValueError("ShowEDS HTML export supports encoding='full' or encoding='auto'")
        mode_aliases = {
            "": "single",
            "auto": "single",
            "state": "single",
            "embedded": "single",
            "single-file": "single",
            "linked-folder": "folder",
            "linked-data-folder": "folder",
            "sidecar": "folder",
        }
        normalised_mode = mode_aliases.get(raw_mode, raw_mode)
        if normalised_mode not in {"single", "folder"}:
            raise ValueError(f"unknown export mode {mode!r}; expected 'single' or 'folder'")
        if requested_binning is not None and requested_binning not in (2, 4):
            raise ValueError(
                f"downsample must be 2 or 4 for ShowEDS, got {requested_binning} "
                "(binning is a compatibility alias)"
            )
        return normalised_mode, requested_binning

    def _export_widget_for_mode(
        self,
        mode: str,
        *,
        encoding: str = "full",
        downsample: int | str | None = None,
        binning: int | str | None = None,
    ) -> tuple["ShowEDS", str]:
        mode, downsample_factor = self._normalise_export_request(
            mode,
            encoding=encoding,
            downsample=downsample,
            binning=binning,
        )
        if downsample_factor is not None:
            return (
                self._make_binned_export_widget(spatial_bin=downsample_factor, energy_bin=downsample_factor),
                f"single sum-binned {downsample_factor}x",
            )
        if mode == "folder":
            if self.compute_backend != "sidecar":
                raise ValueError("folder export is only available for ShowEDS widgets with a data folder")
            return self, "folder exact"
        if mode == "single":
            if self.compute_backend == "kernel" and not self.cube_bytes:
                raise ValueError(
                    "single exact HTML export is not available for native lazy EMD widgets because "
                    "the exported page has no Python kernel or local EMD query backend. Use "
                    "downsample=2/4 for a portable single-file preview, or backend='sidecar' with "
                    "an explicit spatial_bin for a shareable data-folder cache."
                )
            if self.compute_backend == "sidecar":
                raise ValueError(
                    "single exact export is not available for this folder-backed ShowEDS widget; "
                    "use mode='folder' or set downsample=2/4"
                )
            label = "single exact"
            return self, label
        raise AssertionError(f"unhandled export mode {mode!r}")

    def _make_binned_export_widget(self, *, spatial_bin: int, energy_bin: int) -> "ShowEDS":
        if self.compute_backend == "sidecar":
            sidecar = self._sidecar_dir or _sidecar_path_from_url(self.sidecar_url)
            if sidecar is None or not sidecar.exists():
                raise ValueError("binned single export needs an accessible ShowEDS data folder")
            cube, axis, base = _sum_bin_sidecar_cube(sidecar, spatial_bin=spatial_bin, energy_bin=energy_bin)
        elif self.cube_bytes:
            dtype = np.uint16 if self.cube_dtype == "uint16" else np.uint32 if self.cube_dtype == "uint32" else np.float32
            cube = np.frombuffer(self.cube_bytes, dtype=dtype).reshape(self.n_rows, self.n_cols, self.n_energy)
            base = np.frombuffer(self.base_image_bytes, dtype=np.float32).reshape(self.n_rows, self.n_cols)
            cube, axis, base = bin_spectrum_image(
                cube,
                np.asarray(self.energy_keV, dtype=np.float32),
                base_image=base,
                spatial_bin=spatial_bin,
                energy_bin=energy_bin,
            )
        else:
            raise ValueError("binned single export is only available for browser or folder-backed ShowEDS widgets")

        row0 = self.roi_row // spatial_bin
        col0 = self.roi_col // spatial_bin
        row1 = max(row0 + 1, int(np.ceil((self.roi_row + self.roi_height) / spatial_bin)))
        col1 = max(col0 + 1, int(np.ceil((self.roi_col + self.roi_width) / spatial_bin)))
        band_start = self.band_start // energy_bin
        band_end = max(band_start + 1, int(np.ceil(self.band_end / energy_bin)))
        saved_rois = []
        for item in self.saved_rois:
            shape = _normalise_roi_shape(item.get("shape", "rect"))
            row = int(item.get("row", 0)) // spatial_bin
            col = int(item.get("col", 0)) // spatial_bin
            height = max(1, int(np.ceil(int(item.get("height", 1)) / spatial_bin)))
            width = max(1, int(np.ceil(int(item.get("width", 1)) / spatial_bin)))
            row_clamped = max(0, min(int(cube.shape[0]) - 1, row))
            col_clamped = max(0, min(int(cube.shape[1]) - 1, col))
            if shape == "circle":
                diameter = min(
                    max(height, width),
                    int(cube.shape[0]) - row_clamped,
                    int(cube.shape[1]) - col_clamped,
                )
                height = diameter
                width = diameter
            saved_rois.append(
                {
                    **dict(item),
                    "shape": shape,
                    "row": row_clamped,
                    "col": col_clamped,
                    "height": max(1, min(int(cube.shape[0]) - row_clamped, height)),
                    "width": max(1, min(int(cube.shape[1]) - col_clamped, width)),
                }
            )
        saved_bands = []
        for item in self.saved_bands:
            start = int(item.get("start", 0)) // energy_bin
            end = max(start + 1, int(np.ceil(int(item.get("end", start + 1)) / energy_bin)))
            saved_bands.append(
                {
                    **dict(item),
                    "start": max(0, min(int(axis.size) - 1, start)),
                    "end": max(1, min(int(axis.size), end)),
                }
            )
        title = self.title or "ShowEDS"
        widget = ShowEDS(
            cube,
            axis,
            title=f"{title} sum-binned {spatial_bin}x/{energy_bin}x",
            base_image=base,
            band=(band_start, min(int(axis.size), band_end)),
            roi=(
                max(0, min(int(cube.shape[0]) - 1, row0)),
                max(0, min(int(cube.shape[1]) - 1, col0)),
                max(1, min(int(cube.shape[0]) - row0, row1 - row0)),
                max(1, min(int(cube.shape[1]) - col0, col1 - col0)),
            ),
            roi_shape=self.roi_shape,
            panel_width_px=self.panel_width_px,
            spectrum_width_px=self.spectrum_width_px,
            spectrum_height_px=self.spectrum_height_px,
            show_controls=self.show_controls,
            log_spectrum=self.log_spectrum,
            smooth=self.smooth,
            pixel_size=self.pixel_size * spatial_bin if self.pixel_size > 0 else 0.0,
            pixel_unit=self.pixel_unit,
            scale_bar_visible=self.scale_bar_visible,
            map_vmin_pct=self.map_vmin_pct,
            map_vmax_pct=self.map_vmax_pct,
            overlay_opacity=self.overlay_opacity,
            element_label=self.element_label,
            show_line_hints=self.show_line_hints,
            line_hints=[dict(item) for item in self.line_hints],
            selected_elements=list(self.selected_elements),
            auto_identify=self.auto_identify,
            show_debug=self.show_debug,
            saved_rois=saved_rois,
            saved_bands=saved_bands,
            export_presets=[dict(item) for item in self.export_presets],
            max_state_bytes=None,
        )
        widget.map_zoom = self.map_zoom
        widget.map_view_row = self.map_view_row / spatial_bin
        widget.map_view_col = self.map_view_col / spatial_bin
        widget.spectrum_view_start = self.spectrum_view_start / energy_bin
        widget.spectrum_view_end = max(widget.spectrum_view_start + 1, self.spectrum_view_end / energy_bin)
        widget.load_state_dict(widget.state_dict())
        widget._export_light = True
        return widget

    def _on_export_request_change(self, change: dict) -> None:
        raw = str(change.get("new") or "")
        if not raw:
            return
        try:
            payload = json.loads(raw)
            mode = str(payload.get("mode", "single"))
            encoding = str(payload.get("encoding", "full"))
            downsample = payload.get("downsample")
            binning = payload.get("binning")
            if mode == "clear":
                self.export_payload = b""
                self.export_payload_id = ""
                self.export_filename = ""
                return
            self._normalise_export_request(mode, encoding=encoding, downsample=downsample, binning=binning)
            if payload.get("download"):
                filename = str(payload.get("filename") or self._default_html_export_path().name)
                request_id = str(payload.get("id") or "")
                self.export_status = f"Preparing {filename}..."
                html, mode_label = self._html_export_bytes(
                    mode=mode,
                    encoding=encoding,
                    downsample=downsample,
                    binning=binning,
                )
                self.export_filename = filename
                self.export_payload = html
                self.export_payload_id = request_id
                size_mb = len(html) / (1024 * 1024)
                self.export_status = f"Ready {filename} ({size_mb:.1f} MB, {mode_label})"
            else:
                self.export_status = "Exporting HTML..."
                self.export_html(mode=mode, encoding=encoding, downsample=downsample, binning=binning)
        except Exception as exc:
            self.export_status = f"Export failed: {exc}"

    def _default_html_export_path(self) -> pathlib.Path:
        label = self.title.strip() or "showeds"
        slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        if not slug:
            slug = "showeds"
        return pathlib.Path.cwd() / f"{slug}_{self.n_rows}x{self.n_cols}x{self.n_energy}.html"
