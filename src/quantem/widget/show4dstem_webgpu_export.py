"""Local-server Show4DSTEM WebGPU bundle export.

Mirrors the ShowPtycho handoff protocol: the recipient double-clicks one
``Show4DSTEM.command``, a local range-capable HTTP server starts over the data
folder, and Chrome opens a fully vendored viewer page. The normal CLI path keeps
the compressed HDF5 family on disk and lets the browser range-fetch and
decompress detector chunks directly. No Python package install, no network, and
no folder-grant click are required at view time. Everything the page needs
(require.js, the Jupyter widget manager, anywidget, the server script) ships
from this package's ``static/vendor``.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import pathlib
import re
import struct
from typing import Any, Sequence

import numpy as np

from quantem.widget.command_launcher import write_command_launcher
from quantem.widget.io.hdf5_family import collect_hdf5_family

_VENDOR = pathlib.Path(__file__).parent / "vendor"
# CDN references embed_minimal_html emits; each is replaced by a vendored copy so
# the bundle works with no network at all (conference wifi is not a dependency).
# Patterns, not exact URLs: the emitted version specifiers drift across
# ipywidgets/anywidget releases (e.g. anywidget@0.11.0 vs anywidget@~0.11.*).
_CDN_REWRITES = (
    (re.compile(r"https://cdnjs\.cloudflare\.com/[^\"']*/require(\.min)?\.js"), "./require.min.js"),
    (re.compile(r"https://cdn\.jsdelivr\.net/[^\"']*html-manager[^\"']*/embed-amd\.js"), "./embed-amd.js"),
    (re.compile(r"\"https://cdn\.jsdelivr\.net/npm/anywidget@[^\"]*\""), '"./anywidget.min"'),
)


def _write_vendor_asset(name: str, viewer: pathlib.Path) -> None:
    """Expand a compressed browser-manager asset into the export viewer."""
    source = _VENDOR / f"{name}.gz"
    if not source.is_file():
        raise FileNotFoundError(
            f"missing vendored Show4DSTEM browser asset: {source}; "
            "rebuild the package with src/quantem/widget/vendor included"
        )
    with gzip.open(source, "rb") as src, (viewer / name).open("wb") as dst:
        dst.write(src.read())


# Promoted WebGPU decode configuration. Native uint16 is the conservative
# default; audited uint8 browse sources can use the low8-only kernel to skip the
# high bitplanes that only hold masked detector sentinels.
def _tuning(*, h5_decode_dtype: str, h5_uint8_lossless: bool) -> str:
    dtype = "uint8" if h5_uint8_lossless or h5_decode_dtype in {"u8", "uint8"} else "u2"
    low8 = "true" if h5_uint8_lossless else "false"
    return (
        "<script>\n"
        f'globalThis.__QT_H5_DECODE_DTYPE ??= "{dtype}";\n'
        f"globalThis.__QT_H5_FORCE_LOW8 ??= {low8};\n"
        f"globalThis.__BSLZ4_LOW8_ONLY ??= {low8};\n"
        "globalThis.__BSLZ4_FRAME_WG ??= 64;\n"
        "globalThis.__BSLZ4_PIPELINE_STAGING ??= false;\n"
        "globalThis.__QT_H5_FETCH_WINDOW ??= 8;\n"
        "globalThis.__QT_H5_DECODE_QUEUE ??= 8;\n"
        "globalThis.__QT_H5_PRELOAD_WINDOW ??= 1;\n"
        "globalThis.__QT_H5_LOCAL_GROUP ??= 8;\n"
        "globalThis.__QT_H5_LOCAL_WORKERS ??= 8;\n"
        "</script>\n"
    )


def export_show4dstem_webgpu_bundle(
    widget: Any,
    out_dir: str | pathlib.Path,
    *,
    port: int = 8794,
    title: str | None = None,
    h5_decode_dtype: str = "uint16",
) -> pathlib.Path:
    """Write a double-clickable Show4DSTEM WebGPU bundle into ``out_dir``.

    ``out_dir`` must be the folder holding the linked ``*_master.h5`` family the
    widget references. Produces ``Show4DSTEM.command`` at the root and a hidden
    ``.viewer/`` with the
    vendored page and the range-capable server. Returns the path to the
    launcher. Without this bundle the recipient needs Python, the CDNs, and a
    folder-grant click; with it the demo is one double-click.
    """
    root = pathlib.Path(out_dir)
    if not root.is_dir():
        raise ValueError(f"bundle out_dir must be an existing data folder: {root}")
    masters = sorted(root.rglob("*_master.h5"))
    masters.extend(sorted(root.rglob("*_master_wrapper.h5")))
    if not masters and not (root / "rans" / "manifest.json").is_file():
        raise ValueError(f"no *_master.h5 files or rans/manifest.json in {root}; the bundle serves the data folder itself")
    viewer = root / ".viewer"
    viewer.mkdir(exist_ok=True)
    html = viewer / "Show4DSTEM.html"
    widget._write_html_export(html, dtype="uint16", det_bin=1, scan_bin=1, title=title)
    text = html.read_text(encoding="utf-8")
    text = text.replace(
        "<head>",
        "<head>\n"
        "<script>\n"
        "if (globalThis.location?.protocol === \"file:\") {\n"
        "  globalThis.__QT_REQUIRE_LOCAL_H5_FILES = true;\n"
        "}\n"
        "</script>\n"
        + _tuning(
            h5_decode_dtype=str(h5_decode_dtype).lower(),
            h5_uint8_lossless=bool(getattr(widget, "_h5_uint8_lossless", False)),
        ),
        1,
    )
    for pattern, local in _CDN_REWRITES:
        text = pattern.sub(local, text)
    html.write_text(text, encoding="utf-8")
    for name in ("require.min.js", "embed-amd.js", "anywidget.min.js"):
        _write_vendor_asset(name, viewer)
    (root / "index.html").write_text(
        """<!doctype html>
<html><head><meta charset="utf-8"><title>Show4DSTEM</title></head>
<body>
<script>
if (window.location.protocol === "file:") {
  window.location.replace(".viewer/Show4DSTEM.html" + window.location.search + window.location.hash);
} else {
  window.location.replace(".viewer/Show4DSTEM.html" + window.location.search + window.location.hash);
}
</script>
</body></html>
""",
        encoding="utf-8",
    )
    return write_command_launcher(
        root,
        "Show4DSTEM",
        viewer_html="index.html",
        port=int(port),
    )


def bundle_master_urls(
    folder: str | pathlib.Path,
    names: Sequence[str] | None = None,
    *,
    viewer_prefix: str = "..",
) -> list[str]:
    """Return viewer-relative URLs for masters in a bundle folder.

    The viewer page lives one level down in ``.viewer/``, so data references
    must climb back to the served root; a bare basename would resolve inside
    ``.viewer/`` and 404. ``names`` filters by substring, preserving its order.
    """
    folder = pathlib.Path(folder)
    masters = sorted(p.name for p in folder.glob("*_master.h5"))
    if names:
        picked = []
        for token in names:
            hits = [m for m in masters if token in m]
            if not hits:
                raise ValueError(f"no master matches {token!r} in {folder}")
            picked.append(hits[0])
        masters = picked
    return [f"{viewer_prefix}/{name}" for name in masters]


def _link_read_only(source: pathlib.Path, target: pathlib.Path) -> str:
    """Link one source file without ever falling back to a physical copy."""

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        try:
            target.symlink_to(source)
            return "symlink"
        except OSError as exc:
            raise ValueError(
                "Show4DSTEM needs a hard link or symbolic link to keep the raw "
                f"source read-only without copying it: {source}"
            ) from exc


def export_show4dstem_hdf5_viewer(
    master: str | pathlib.Path,
    out_dir: str | pathlib.Path,
    *,
    scan_shape: tuple[int, int],
    detector_shape: tuple[int, int],
    title: str,
    target_stem: str | None = None,
) -> pathlib.Path:
    """Write a direct WebGPU viewer linked to one read-only HDF5 family.

    The output contains hard links when source and project share a filesystem,
    otherwise symbolic links. Raw detector bytes are never copied.
    """

    from quantem.widget import Show4DSTEM

    source_master = pathlib.Path(master).expanduser().resolve()
    root = pathlib.Path(out_dir).expanduser().resolve()
    if root == source_master.parent:
        raise ValueError(
            "Show4DSTEM project output must differ from the source HDF5 folder"
        )
    root.mkdir(parents=True, exist_ok=True)
    # This function owns the raw-family aliases in ``root``. Remove aliases
    # from an earlier export so --force cannot leave acquisition names or
    # obsolete detector parts behind. The source files remain untouched.
    for pattern in ("*_master.h5", "*_data_*.h5"):
        for stale in root.glob(pattern):
            if stale.is_file() or stale.is_symlink():
                stale.unlink()

    family = collect_hdf5_family(source_master)
    source_stem = family[0].name.removesuffix("_master.h5")
    linked: list[pathlib.Path] = []
    for source in family:
        target_name = source.name
        if target_stem is not None:
            if source == family[0]:
                if not source.name.endswith("_master.h5"):
                    raise ValueError(
                        "an anonymous Show4DSTEM companion requires a native "
                        "*_master.h5 family"
                    )
                target_name = f"{target_stem}_master.h5"
            else:
                prefix = f"{source_stem}_data_"
                if not source.name.startswith(prefix):
                    raise ValueError(
                        "an anonymous Show4DSTEM companion requires detector "
                        "files named <source>_data_*.h5"
                    )
                target_name = f"{target_stem}_data_{source.name[len(prefix):]}"
        target = root / target_name
        _link_read_only(source, target)
        linked.append(target)

    widget = Show4DSTEM(
        np.zeros((1, 1, 1, 1), dtype=np.uint8),
        h5_urls=[f"../{linked[0].name}"],
        h5_uint8_lossless=False,
        scan_shape=tuple(int(value) for value in scan_shape),
        detector_shape=tuple(int(value) for value in detector_shape),
        backend="webgpu",
        precompute_virtual_images=False,
        verbose=False,
    )
    try:
        export_show4dstem_webgpu_bundle(widget, root, title=title)
    finally:
        widget.close()
    return root / ".viewer" / "Show4DSTEM.html"


def _read_h5_bad_pixel_indices(path: pathlib.Path, detector_size: int) -> tuple[list[int], bool]:
    """Return detector pixels from HDF5 metadata and whether a mask existed."""
    try:
        from quantem.gpu.io import inspect

        mask = inspect(path).pixel_mask
    except Exception:
        return [], False
    if mask is None:
        return [], False
    bad = np.flatnonzero(np.asarray(mask).reshape(-1) > 0)
    bad = bad[(bad >= 0) & (bad < detector_size)]
    return bad.astype(int).tolist(), True


def _bslz4_payload_size_at(
    handle,
    offset: int,
    file_size: int,
    *,
    expected_uncompressed_bytes: int,
) -> int | None:
    """Return one raw BSLZ4 chunk payload size from a file offset."""
    if offset < 0 or offset + 12 > file_size:
        return None
    handle.seek(offset)
    header = handle.read(12)
    if len(header) != 12:
        return None
    _, uncompressed_bytes, block_bytes = struct.unpack(">III", header)
    if uncompressed_bytes != expected_uncompressed_bytes or block_bytes <= 0:
        return None
    n_blocks = math.ceil(uncompressed_bytes / block_bytes)
    size = 12
    for _ in range(n_blocks):
        raw = handle.read(4)
        if len(raw) != 4:
            return None
        (compressed_size,) = struct.unpack(">I", raw)
        if handle.tell() + compressed_size > file_size:
            return None
        handle.seek(compressed_size, 1)
        size += 4 + compressed_size
    return size


def _find_next_bslz4_payload_offset(
    handle,
    start_offset: int,
    file_size: int,
    *,
    expected_uncompressed_bytes: int,
    search_window: int = 1024 * 1024,
) -> int | None:
    """Find the next BSLZ4 chunk header after an HDF5 metadata gap."""
    if start_offset < 0 or start_offset >= file_size:
        return None
    handle.seek(start_offset)
    data = handle.read(min(search_window, file_size - start_offset))
    needle = struct.pack(">II", 0, int(expected_uncompressed_bytes))
    pos = data.find(needle)
    while pos >= 0:
        candidate = start_offset + pos
        size = _bslz4_payload_size_at(
            handle,
            candidate,
            file_size,
            expected_uncompressed_bytes=expected_uncompressed_bytes,
        )
        if size is not None:
            return candidate
        pos = data.find(needle, pos + 1)
    return None


def _contiguous_bslz4_frame_index(
    data_file: pathlib.Path,
    *,
    first_offset: int,
    n_frames: int,
    uncompressed_bytes: int,
) -> list[tuple[int, int, int]] | None:
    """Derive frame byte ranges when HDF5 stores raw chunks contiguously."""
    file_size = data_file.stat().st_size
    out: list[tuple[int, int, int]] = []
    offset = int(first_offset)
    with data_file.open("rb") as handle:
        for frame in range(int(n_frames)):
            size = _bslz4_payload_size_at(
                handle,
                offset,
                file_size,
                expected_uncompressed_bytes=uncompressed_bytes,
            )
            if size is None:
                next_offset = _find_next_bslz4_payload_offset(
                    handle,
                    offset,
                    file_size,
                    expected_uncompressed_bytes=uncompressed_bytes,
                )
                if next_offset is None:
                    return None
                offset = next_offset
                size = _bslz4_payload_size_at(
                    handle,
                    offset,
                    file_size,
                    expected_uncompressed_bytes=uncompressed_bytes,
                )
            if size is None:
                return None
            out.append((frame, offset, size))
            offset += size
    return out


def build_lazy_show4dstem_sidecar(
    folder: str | pathlib.Path,
    *,
    label: str,
    scan_shape: tuple[int, int],
    detector_shape: tuple[int, int],
) -> str:
    """Build a lazy WebGPU sidecar for an anonymous HDF5 family.

    The sidecar is intentionally small compared with the raw HDF5 family:
    ``profile.bin`` stores radial detector-bin sums per scan position,
    ``index.bin`` stores one range-fetch pointer per scan frame, and ``com.bin``
    stores a full-detector center-of-mass field. Raw detector frames stay in the
    linked ``*_data_*.h5`` files and are fetched on demand by byte range.
    """

    import h5py

    root = pathlib.Path(folder)
    data_files = sorted(root.glob(f"{label}_data_*.h5"))
    if not data_files:
        raise ValueError(f"no linked HDF5 data files found for {label!r} in {root}")
    scan_rows, scan_cols = (int(scan_shape[0]), int(scan_shape[1]))
    det_rows, det_cols = (int(detector_shape[0]), int(detector_shape[1]))
    if det_rows != det_cols:
        raise ValueError(
            "Show4DSTEM lazy WebGPU sidecars currently require a square detector; "
            f"got detector_shape={detector_shape!r}."
        )
    scan_count = scan_rows * scan_cols
    detector_size = det_rows * det_cols
    nbins = max(1, det_rows // 2)
    lazy_dir = root / f"{label}_lazy"
    lazy_dir.mkdir(parents=True, exist_ok=True)
    bad_pixels, has_h5_pixel_mask = _read_h5_bad_pixel_indices(
        root / f"{label}_master.h5",
        detector_size,
    )
    bad_mask = np.zeros(detector_size, dtype=bool)
    if bad_pixels:
        bad_mask[np.asarray(bad_pixels, dtype=np.int64)] = True

    rows = np.arange(det_rows, dtype=np.float32)[:, None]
    cols = np.arange(det_cols, dtype=np.float32)[None, :]
    radial_bins = np.floor(
        np.hypot(rows - det_rows / 2, cols - det_cols / 2)
    ).astype(np.int32)
    radial_bins = np.clip(radial_bins.reshape(-1), 0, nbins - 1)
    radial_order = np.argsort(radial_bins)
    radial_sorted_bins = radial_bins[radial_order]
    radial_unique_bins, radial_starts = np.unique(radial_sorted_bins, return_index=True)
    row_coords = np.broadcast_to(
        np.arange(det_rows, dtype=np.float32)[:, None], (det_rows, det_cols)
    ).reshape(-1)
    col_coords = np.broadcast_to(
        np.arange(det_cols, dtype=np.float32)[None, :], (det_rows, det_cols)
    ).reshape(-1)
    batch = 512
    source_dtype: str | None = None
    if not has_h5_pixel_mask:
        for data_file in data_files:
            with h5py.File(data_file, "r") as handle:
                dataset = handle.get("entry/data/data")
                if dataset is None:
                    raise ValueError(f"{data_file.name} has no entry/data/data dataset")
                if tuple(int(value) for value in dataset.shape[-2:]) != (det_rows, det_cols):
                    raise ValueError(
                        f"{data_file.name} detector shape {dataset.shape[-2:]} does not "
                        f"match {detector_shape!r}."
                    )
                dtype_name = np.dtype(dataset.dtype).name
                if source_dtype is None:
                    source_dtype = dtype_name
                elif source_dtype != dtype_name:
                    raise ValueError(
                        f"{label!r} mixes HDF5 source dtypes {source_dtype!r} and "
                        f"{dtype_name!r}; WebGPU lazy sidecars require one dtype."
                    )
                if not np.issubdtype(dataset.dtype, np.integer):
                    continue
                saturated_value = np.iinfo(dataset.dtype).max
                for start in range(0, int(dataset.shape[0]), batch):
                    stop = min(int(dataset.shape[0]), start + batch)
                    frames = np.asarray(dataset[start:stop]).reshape(
                        stop - start, detector_size
                    )
                    bad_mask |= (frames >= saturated_value).any(axis=0)

    profile_path = lazy_dir / "profile.bin"
    index_path = lazy_dir / "index.bin"
    com_path = lazy_dir / "com.bin"
    profile = np.memmap(profile_path, mode="w+", dtype=np.float32, shape=(scan_count, nbins))
    frame_index = np.memmap(index_path, mode="w+", dtype=np.uint32, shape=(scan_count, 3))
    com = np.memmap(com_path, mode="w+", dtype=np.float32, shape=(2, scan_count))

    frame_cursor = 0
    for file_index, data_file in enumerate(data_files):
        with h5py.File(data_file, "r") as handle:
            dataset = handle.get("entry/data/data")
            if dataset is None:
                raise ValueError(f"{data_file.name} has no entry/data/data dataset")
            if tuple(int(value) for value in dataset.shape[-2:]) != (det_rows, det_cols):
                raise ValueError(
                    f"{data_file.name} detector shape {dataset.shape[-2:]} does not "
                    f"match {detector_shape!r}."
                )
            dtype_name = np.dtype(dataset.dtype).name
            if source_dtype is None:
                source_dtype = dtype_name
            elif source_dtype != dtype_name:
                raise ValueError(
                    f"{label!r} mixes HDF5 source dtypes {source_dtype!r} and "
                    f"{dtype_name!r}; WebGPU lazy sidecars require one dtype."
                )
            n_frames = int(dataset.shape[0])
            if frame_cursor + n_frames > scan_count:
                raise ValueError(
                    f"{label!r} has more frames than scan_shape={scan_shape!r}."
                )
            start_scan = frame_cursor
            indexed = False
            first_info = dataset.id.get_chunk_info_by_coord((0, 0, 0))
            contiguous_infos = _contiguous_bslz4_frame_index(
                data_file,
                first_offset=int(first_info.byte_offset),
                n_frames=n_frames,
                uncompressed_bytes=int(np.dtype(dataset.dtype).itemsize) * detector_size,
            )
            if (
                contiguous_infos is not None
                and len(contiguous_infos) == n_frames
                and contiguous_infos[0][2] == int(first_info.size)
            ):
                for frame, byte_offset, size in contiguous_infos:
                    frame_index[start_scan + frame] = (file_index, byte_offset, size)
                indexed = True
            if (
                not indexed
                and hasattr(dataset.id, "get_num_chunks")
                and hasattr(dataset.id, "get_chunk_info")
            ):
                infos = []
                for chunk_index in range(int(dataset.id.get_num_chunks())):
                    info = dataset.id.get_chunk_info(chunk_index)
                    offset = tuple(int(value) for value in info.chunk_offset)
                    if len(offset) >= 1 and 0 <= offset[0] < n_frames:
                        infos.append((offset[0], int(info.byte_offset), int(info.size)))
                if len(infos) == n_frames:
                    for frame, byte_offset, size in sorted(infos):
                        frame_index[start_scan + frame] = (file_index, byte_offset, size)
                    indexed = True
            if not indexed:
                for frame in range(n_frames):
                    info = dataset.id.get_chunk_info_by_coord((frame, 0, 0))
                    frame_index[start_scan + frame] = (
                        file_index,
                        int(info.byte_offset),
                        int(info.size),
                    )
            frame_cursor += n_frames
            for start in range(0, n_frames, batch):
                stop = min(n_frames, start + batch)
                frames = np.asarray(dataset[start:stop], dtype=np.float32).reshape(
                    stop - start, detector_size
                )
                out_slice = slice(start_scan + start, start_scan + stop)
                if bad_mask.any():
                    frames[:, bad_mask] = 0
                radial_sums = np.zeros((stop - start, nbins), dtype=np.float32)
                radial_sums[:, radial_unique_bins] = np.add.reduceat(
                    frames[:, radial_order],
                    radial_starts,
                    axis=1,
                )
                profile[out_slice, :] = radial_sums
                totals = frames.sum(axis=1)
                safe_totals = np.where(totals > 0, totals, 1.0)
                com[0, out_slice] = (frames @ row_coords) / safe_totals
                com[1, out_slice] = (frames @ col_coords) / safe_totals

    if frame_cursor != scan_count:
        raise ValueError(
            f"{label!r} has {frame_cursor} frames, expected {scan_count} from "
            f"scan_shape={scan_shape!r}."
        )
    profile.flush()
    frame_index.flush()
    com.flush()
    meta = {
        "SR": scan_rows,
        "SC": scan_cols,
        "D": det_rows,
        "NB": nbins,
        "nFrames": scan_count,
        "files": [f"../{path.name}" for path in data_files],
        "sourceDtype": source_dtype or "uint16",
        "badPixels": np.flatnonzero(bad_mask).astype(int).tolist(),
    }
    (lazy_dir / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    return f"{label}_lazy/"


# --- browser-resident lossless (rANS) series -------------------------------------------------

def export_show4dstem_rans_viewer(
    build_result: str | pathlib.Path | Sequence[str | pathlib.Path],
    out_dir: str | pathlib.Path,
    *,
    tilts: Sequence[int] | None = None,
    title: str = "Show4DSTEM",
    frame_labels: Sequence[str] | None = None,
    valid_pixels: np.ndarray | None = None,
    debug: bool = False,
) -> pathlib.Path:
    """Write a WebGPU viewer whose source is an encoded rANS series decoded in the browser.

    ``build_result`` is the JSON written by the detector-rANS encoder (one record per
    acquisition with ``artifacts`` folder, ``scan_block``, ``model_frames``,
    ``scale_bits``, ``models`` and ``shape``). The encoded payload files are linked
    read-only into ``out_dir/rans`` and read by HTTP byte range, so nothing large is
    copied. Decode tables (packed symbol entries, a 256-bucket slot lookup per column,
    column metadata) are derived once here and written next to the links.

    An integer QEM file, or an ordered sequence of those files, uses the
    same exporter. Geometry and native dtype are admitted from the validated
    container metadata without decoding counts. Each container is linked into
    the export; the browser requires an explicit local file grant instead of
    fetching a whole container through HTTP. All selected acquisitions must
    share geometry and dtype. ``tilts`` selects indices from the supplied order.

    Parameters
    ----------
    build_result
        Legacy encoder JSON, one integer QEM file, or an ordered
        sequence of QEM files.
    out_dir
        Destination bundle folder. Source containers are linked, never copied
        or modified.
    tilts
        Optional indices selecting acquisitions from the supplied order.
    frame_labels
        One label per selected acquisition.
    valid_pixels
        Optional Boolean native detector mask, shared by the acquisitions.

    Returns
    -------
    pathlib.Path
        Standalone HTML viewer path inside the bundle.

    Examples
    --------
    >>> export_show4dstem_rans_viewer(["tilt-0.qem", "tilt-1.qem"], "viewer")
    PosixPath('.../.viewer/Show4DSTEM.html')
    """

    from quantem.widget import Show4DSTEM

    if isinstance(build_result, (str, pathlib.Path)):
        source_path = pathlib.Path(build_result).expanduser().resolve()
        with source_path.open("rb") as stream:
            canonical = stream.read(8) == b"QEMDATA1"
        canonical_paths = [source_path] if canonical else None
        build_result = source_path
    else:
        canonical_paths = [pathlib.Path(path).expanduser().resolve() for path in build_result]
    if canonical_paths is not None:
        return _export_qem_viewer(
            canonical_paths, out_dir, tilts=tilts, title=title,
            frame_labels=frame_labels, valid_pixels=valid_pixels, debug=debug,
        )

    records = json.loads(pathlib.Path(build_result).read_text())["tilts"]
    chosen = list(tilts) if tilts is not None else list(range(len(records)))
    root = pathlib.Path(out_dir).expanduser().resolve()
    rans = root / "rans"
    rans.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"schema": "quantem.show4dstem-rans-browser/v1", "tilts": []}
    shape = None
    for ordinal, tilt in enumerate(chosen):
        record = records[tilt]
        artifacts = pathlib.Path(record["artifacts"])
        det_rows, det_cols = int(record["shape"][2]), int(record["shape"][3])
        K = det_rows * det_cols
        frames = int(record["scan_block"])
        blocks = int(record["shape"][0]) * int(record["shape"][1]) // frames
        shape = (int(record["shape"][0]), int(record["shape"][1]), det_rows, det_cols)
        offsets = np.load(artifacts / "offsets.npy")
        starts = np.load(artifacts / "block_starts.npy")
        payload_name = f"t{ordinal}-payload.bin"
        _link_read_only(artifacts / "payload.bin", rans / payload_name)
        blocks_meta = []
        for b in range(blocks):
            np.ascontiguousarray(offsets[b], dtype=np.uint32).tofile(rans / f"t{ordinal}-offsets-{b:02d}.u32")
            blocks_meta.append(dict(index=b, byte_start=int(starts[b]), byte_end=int(starts[b + 1]),
                                    bytes=int(starts[b + 1] - starts[b]), model=(b * frames) // int(record["model_frames"])))
        models = []
        for mi, m in enumerate(record["models"]):
            with np.load(artifacts / m["path"]) as z:
                ctx, sym, cum, freq, lit = (z[k] for k in ("context_offsets", "symbols", "cumulative", "frequencies", "literal"))
            entries = np.empty((sym.size, 2), dtype=np.uint32)
            entries[:, 0] = sym.astype(np.uint32) | (cum.astype(np.uint32) << 16)
            entries[:, 1] = freq.astype(np.uint32)
            lut = np.zeros((K, 256), dtype=np.uint8)
            for k in range(K):
                lo, hi = int(ctx[k]), int(ctx[k + 1])
                if hi > lo:
                    lut[k] = np.clip(np.searchsorted(cum[lo:hi].astype(np.int64), np.arange(256) << 7, side="right") - 1, 0, hi - lo - 1)
            entries.tofile(rans / f"t{ordinal}-entries-{mi}.u32")
            lut.tofile(rans / f"t{ordinal}-lut-{mi}.u8")
            ctx.astype(np.uint32).tofile(rans / f"t{ordinal}-ctx-{mi}.u32")
            lit.astype(np.uint8).tofile(rans / f"t{ordinal}-literal-{mi}.u8")
            models.append(dict(index=mi, symbols=int(sym.size)))
        manifest["tilts"].append(dict(tilt=ordinal, source_tilt=int(tilt), K=K, frames=frames, blocks=blocks,
                                      scale=int(record["scale_bits"]), model_frames=int(record["model_frames"]),
                                      payload_url=payload_name, payload_sha256=record.get("payload_sha256"),
                                      blocks_meta=blocks_meta, models=models))
    if shape is None:
        raise ValueError("no acquisitions selected")
    if valid_pixels is not None:
        bad = np.flatnonzero(~np.asarray(valid_pixels, dtype=bool).reshape(-1))
        manifest["bad_pixels"] = [int(v) for v in bad]
    manifest["scan_shape"] = [shape[0], shape[1]]
    manifest["detector_shape"] = [shape[2], shape[3]]
    (rans / "manifest.json").write_text(json.dumps(manifest, indent=1))
    widget = Show4DSTEM(
        np.zeros((1, 1, 1, 1), dtype=np.uint8),
        rans_url="../rans/",
        rans_count=len(chosen),
        scan_shape=(shape[0], shape[1]),
        detector_shape=(shape[2], shape[3]),
        frame_labels=list(frame_labels) if frame_labels else None,
        backend="webgpu",
        view_mode="multiple" if len(chosen) > 1 else "single",
        compare_max_panels=max(3, len(chosen)),
        compare_group_mode="all",
        compare_dp_mode="selected",
        precompute_virtual_images=False,
        debug=debug,
        verbose=False,
    )
    try:
        export_show4dstem_webgpu_bundle(widget, root, title=title)
    finally:
        widget.close()
    return root / ".viewer" / "Show4DSTEM.html"


def _export_qem_viewer(
    paths, out_dir, *, tilts, title, frame_labels, valid_pixels, debug,
):
    """Admit canonical containers and serialize a local-grant-only viewer."""
    from quantem.gpu.io._qem_reference import read_envelope
    from quantem.gpu.io.qem_validation import validate_qem
    from quantem.widget import Show4DSTEM

    chosen = list(tilts) if tilts is not None else list(range(len(paths)))
    if not chosen or any(type(index) is not int or not 0 <= index < len(paths) for index in chosen):
        raise ValueError("Select at least one valid acquisition index from the QEM files.")
    if frame_labels is not None and len(frame_labels) != len(chosen):
        raise ValueError("Provide one frame label per selected QEM acquisition.")
    sources = []
    shape = dtype = None
    for ordinal, index in enumerate(chosen):
        integrity = validate_qem(paths[index])
        header, _ = read_envelope(paths[index])
        if header["codec"] != "runtime-column-rans-spatial-v2" or header["dtype"] not in ("uint8", "uint16"):
            raise ValueError("Browser QEM export needs exact uint8/uint16 QEM counts; use Python GPU for float32 or scaled data.")
        current_shape, current_dtype = tuple(header["shape"]), header["dtype"]
        if shape is None:
            shape, dtype = current_shape, current_dtype
        elif current_shape != shape or current_dtype != dtype:
            raise ValueError("QEM acquisitions must share native scan/detector geometry and dtype.")
        sources.append({
            "url": f"t{ordinal}-counts.qem",
            "source_index": index,
            "shape": list(current_shape),
            "dtype": current_dtype,
            "file_bytes": integrity["file_bytes"],
            "scientific_metadata": header["scientific_metadata"],
        })
    bad_pixels = []
    if valid_pixels is not None:
        valid = np.asarray(valid_pixels, dtype=bool)
        if valid.shape != shape[2:]:
            raise ValueError(f"valid_pixels must have native detector shape {shape[2:]}.")
        bad_pixels = np.flatnonzero(~valid.reshape(-1)).astype(int).tolist()
    root = pathlib.Path(out_dir).expanduser().resolve()
    rans = root / "rans"
    rans.mkdir(parents=True, exist_ok=True)
    for index, record in zip(chosen, sources):
        target = rans / record["url"]
        if target.exists() and target.samefile(paths[index]):
            record["link"] = "existing"
        else:
            record["link"] = _link_read_only(paths[index], target)
    manifest = {
        "schema": "quantem.show4dstem-qem-browser/v1",
        "source_format": "qem-v1",
        "local_grant_required": True,
        "sources": sources,
        "scan_shape": list(shape[:2]),
        "detector_shape": list(shape[2:]),
        "dtype": dtype,
        "bad_pixels": bad_pixels,
    }
    (rans / "manifest.json").write_text(json.dumps(manifest, indent=1))
    widget = Show4DSTEM(
        np.zeros((1, 1, 1, 1), dtype=np.uint8),
        rans_url="../rans/", rans_count=len(chosen),
        rans_format="qem-v1", rans_files=[record["url"] for record in sources],
        rans_dtype=dtype, scan_shape=shape[:2], detector_shape=shape[2:],
        offline_dtype=dtype, frame_labels=list(frame_labels) if frame_labels else None,
        backend="webgpu", view_mode="multiple" if len(chosen) > 1 else "single",
        compare_max_panels=max(3, len(chosen)), compare_group_mode="all",
        compare_dp_mode="selected", precompute_virtual_images=False,
        debug=debug, verbose=False,
    )
    widget._offline_bad_px = json.dumps(bad_pixels)
    try:
        export_show4dstem_webgpu_bundle(widget, root, title=title)
    finally:
        widget.close()
    return root / ".viewer" / "Show4DSTEM.html"
