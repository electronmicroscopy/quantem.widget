from __future__ import annotations

import json
import math
import os
import pathlib
import shutil
from typing import Any

import numpy as np

from quantem.widget.io.hdf5_family import collect_hdf5_family

def _showptycho_fft_mag(array: np.ndarray) -> np.ndarray:
    """Match the ShowPtycho frontend FFT display path."""

    phase = array.astype(np.float32, copy=False)
    centered = (phase - np.float32(phase.mean())).astype(np.float32)
    h, w = centered.shape
    window = (
        np.hanning(h).astype(np.float32)[:, None]
        * np.hanning(w).astype(np.float32)[None, :]
    )
    padded_h = 1 << int(math.ceil(math.log2(h)))
    padded_w = 1 << int(math.ceil(math.log2(w)))
    padded = np.zeros((padded_h, padded_w), dtype=np.complex64)
    padded[:h, :w] = (centered * window).astype(np.complex64)
    fft = np.fft.fft2(padded).astype(np.complex64)
    mag = np.abs(fft[:h, :w]).astype(np.float32)
    return np.log1p(np.fft.fftshift(mag)).astype(np.float32)


def _write_empty_snapshots_manifest(out_path: pathlib.Path) -> None:
    """Seed optional standalone folder snapshots so the UI has no startup 404."""

    snapshots_dir = out_path / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    snapshots_json = snapshots_dir / "snapshots.json"
    if not snapshots_json.exists():
        snapshots_json.write_text("[]\n", encoding="utf-8")


def _jsonable_float_list(array: object) -> list[float]:
    return np.asarray(array).astype(np.float32, copy=False).tolist()


def _jsonable_int_list(array: object) -> list[int]:
    return np.asarray(array).astype(np.int32, copy=False).tolist()


def _folder_calibration(widget: Any, state: Any) -> dict[str, Any]:
    ny, nx = state.scan_shape
    g_shape = (state.num_bf, ny, nx)
    c10 = float(widget._current_c10())
    c12 = float(widget._current_c12())
    phi12_deg = float(widget._current_phi12_deg())
    phi12 = math.radians(phi12_deg)
    scan_region = widget._scan_region
    if scan_region is None:
        scan_region_payload = {
            "row_start": 0,
            "row_stop": ny,
            "col_start": 0,
            "col_stop": nx,
            "shape": [ny, nx],
        }
    else:
        row_start, row_stop, col_start, col_stop = scan_region
        scan_region_payload = {
            "row_start": int(row_start),
            "row_stop": int(row_stop),
            "col_start": int(col_start),
            "col_stop": int(col_stop),
            "shape": [int(row_stop - row_start), int(col_stop - col_start)],
        }
    ssb = widget._ssb_ref
    scan_sampling = ssb.scan_sampling_A if ssb is not None else None
    if isinstance(scan_sampling, (tuple, list)):
        scan_sampling_A = float(scan_sampling[0])
    elif scan_sampling is not None:
        scan_sampling_A = float(scan_sampling)
    else:
        scan_sampling_A = float(widget.pixel_size or 0.0)
    return {
        "schema_version": 1,
        "kind": "showptycho_webgpu_folder",
        "source_file": "redacted_local_source",
        "source_calibration": "redacted_local_calibration",
        "scan_region": scan_region_payload,
        "backend_reference": "quantem.gpu SSBProtocol.reconstruct_with_loss",
        "precision": {
            "real_dtype": state.precision.real_dtype,
            "complex_dtype": state.precision.complex_dtype,
        },
        "bf_radius_px": state.brightfield.radius_px,
        "num_bf": state.num_bf,
        "g_shape": [int(g_shape[0]), int(g_shape[1]), int(g_shape[2])],
        "g_dtype": "complex64_interleaved_re_im_native_le",
        "phase_shape": [ny, nx],
        "phase_dtype": "float32_native_le",
        "detector_shape": list(state.brightfield.detector_shape),
        "bf_center": list(state.brightfield.center_row_col),
        "bf_rows": _jsonable_int_list(state.brightfield.rows),
        "bf_cols": _jsonable_int_list(state.brightfield.cols),
        "kx_bf": _jsonable_float_list(state.kx_bf),
        "ky_bf": _jsonable_float_list(state.ky_bf),
        "qx_1d": _jsonable_float_list(state.qx_1d),
        "qy_1d": _jsonable_float_list(state.qy_1d),
        "aperture_k": _jsonable_float_list(state.aperture_k),
        "alpha_k2": _jsonable_float_list(state.alpha_k2),
        "cos2phi_k": _jsonable_float_list(state.cos2phi_k),
        "sin2phi_k": _jsonable_float_list(state.sin2phi_k),
        "wavelength_A": state.wavelength_A,
        "semiangle_mrad": state.semiangle_rad * 1e3,
        "semiangle_rad": state.semiangle_rad,
        "scan_sampling_A": scan_sampling_A,
        "voltage_kV": float(ssb.voltage_kV if ssb is not None else 0.0),
        "det_sampling_mrad_px": [
            value * 1e3 for value in state.angular_sampling_rad
        ],
        "sampling_A": list(state.sampling_A),
        "angular_sampling_rad": list(state.angular_sampling_rad),
        "rotation_angle_deg": float(widget.rotation_deg),
        "rotation_angle_rad": math.radians(float(widget.rotation_deg)),
        "aberrations": {
            "C10": c10,
            "C12": c12,
            "phi12": phi12,
            "phi12_deg": phi12_deg,
        },
        "flip_phase": bool(widget.flip_phase),
        "dc_value": [float(state.dc_value.real), float(state.dc_value.imag)],
    }


def _ensure_supported_webgpu_shape(state: Any) -> tuple[int, int]:
    """Validate the specialized browser SSB kernels can open this crop."""

    ny, nx = state.scan_shape
    supported = {128, 256, 512, 1024}
    if ny != nx or ny not in supported:
        raise NotImplementedError(
            "ShowPtycho WebGPU folder export supports square 128, 256, 512, or 1024 "
            f"crops; got {ny}x{nx}."
        )
    return ny, nx


def _write_embedded_widget_html(
    widget: Any,
    html_path: pathlib.Path,
    *,
    title: str,
    calibration: dict[str, Any],
    h5_source: dict[str, Any] | None = None,
) -> None:
    """Write the same ShowPtycho widget UI with folder-local WebGPU data."""

    from ipywidgets.embed import dependency_state, embed_minimal_html

    from .export import ensure_mobile_viewport

    old_state = (
        widget.webgpu_preview_enabled,
        widget.webgpu_standalone,
        widget.webgpu_cal_json,
        widget.webgpu_h5_source_json,
        widget.webgpu_preview_status,
        widget.phase_bytes,
        widget.phase_width,
        widget.phase_height,
        widget.stars_path,
        widget.calibration_path,
    )
    try:
        widget.webgpu_preview_enabled = True
        widget.webgpu_standalone = True
        widget.webgpu_cal_json = json.dumps(calibration)
        widget.webgpu_h5_source_json = json.dumps(h5_source or {})
        widget.stars_path = "snapshots/snapshots.json"
        widget.calibration_path = "snapshots/calibration.json"
        if h5_source:
            widget.phase_bytes = b""
            widget.phase_width = 0
            widget.phase_height = 0
        if h5_source and h5_source.get("kind") == "bf_columns":
            widget.webgpu_preview_status = (
                "WebGPU folder ready: browser range-reads exact BF columns "
                "and builds reducers transiently."
            )
        elif h5_source:
            widget.webgpu_preview_status = (
                "WebGPU folder ready: browser reads compressed HDF5 source "
                "and builds BF reducers transiently."
            )
        else:
            raise ValueError("WebGPU folder export requires an exact detector source.")
        state = dependency_state([widget], drop_defaults=False)
        embed_minimal_html(
            str(html_path),
            views=[widget],
            title=title,
            drop_defaults=False,
            state=state,
        )
    finally:
        (
            widget.webgpu_preview_enabled,
            widget.webgpu_standalone,
            widget.webgpu_cal_json,
            widget.webgpu_h5_source_json,
            widget.webgpu_preview_status,
            widget.phase_bytes,
            widget.phase_width,
            widget.phase_height,
            widget.stars_path,
            widget.calibration_path,
        ) = old_state
    ensure_mobile_viewport(html_path)


def _link_or_copy(src: pathlib.Path, dst: pathlib.Path) -> str:
    """Make ``dst`` refer to ``src`` without duplicating bytes when possible."""

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        try:
            dst.symlink_to(src)
            return "symlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy"


def _find_hdf5_stack(handle: Any) -> Any:
    """Return the 3D detector stack dataset inside an HDF5 data file."""

    candidates = ("entry/data/data", "entry/data", "data")
    for path in candidates:
        try:
            dataset = handle[path]
        except KeyError:
            continue
        if getattr(dataset, "shape", None) and len(dataset.shape) == 3:
            return dataset
    for _, item in handle.items():
        if getattr(item, "shape", None) and len(item.shape) == 3:
            return item
        if hasattr(item, "visititems"):
            found: list[Any] = []

            def visitor(_name: str, obj: Any) -> None:
                if not found and getattr(obj, "shape", None) and len(obj.shape) == 3:
                    found.append(obj)

            item.visititems(visitor)
            if found:
                return found[0]
    raise ValueError("no 3D detector-stack dataset found in HDF5 data file")


def _hdf5_chunk_index_records(
    src: pathlib.Path,
) -> tuple[dict[str, Any], np.ndarray] | None:
    """Return compact raw-chunk offset metadata for browser range reads."""

    try:
        import h5py
    except ImportError:
        return None

    try:
        with h5py.File(src, "r") as handle:
            dataset = _find_hdf5_stack(handle)
            shape = tuple(int(v) for v in dataset.shape)
            if len(shape) != 3 or not getattr(dataset, "chunks", None):
                return None
            dtype_name = str(dataset.dtype)
            chunk_shape = [int(v) for v in dataset.chunks]
            n_frames = shape[0]
            records = np.zeros((n_frames, 2), dtype="<u8")

            def collect(info: Any) -> None:
                frame = int(info.chunk_offset[0])
                if 0 <= frame < n_frames:
                    records[frame, 0] = int(info.byte_offset)
                    records[frame, 1] = int(info.size)

            dataset.id.chunk_iter(collect)
    except Exception:
        return None
    if not np.all(records[:, 1] > 0):
        return None

    return {
        "frames": int(shape[0]),
        "detector_shape": [int(shape[1]), int(shape[2])],
        "dtype": dtype_name,
        "chunk_shape": chunk_shape,
        "record": "u64le_offset,u64le_size",
    }, records


def _write_hdf5_chunk_index(src: pathlib.Path, dst: pathlib.Path) -> dict[str, Any] | None:
    """Write a compact single-file raw-chunk index for browser range reads."""

    built = _hdf5_chunk_index_records(src)
    if built is None:
        return None
    index, records = built
    dst.parent.mkdir(parents=True, exist_ok=True)
    records.tofile(dst)
    index["path"] = dst.name
    index["bytes"] = int(dst.stat().st_size)
    return index


def _prepare_hdf5_source_folder(
    master: pathlib.Path,
    out_path: pathlib.Path,
    *,
    files: list[pathlib.Path] | None = None,
) -> dict[str, Any]:
    """Expose compressed HDF5 source files inside the review folder."""

    files = files or collect_hdf5_family(master)
    links = []
    chunk_indexes = []
    chunk_index_payloads: list[tuple[dict[str, Any], dict[str, Any], np.ndarray]] = []
    for src in files:
        rel = pathlib.Path("source") / src.name
        mode = _link_or_copy(src, out_path / rel)
        entry = {
            "path": rel.as_posix(),
            "name": src.name,
            "bytes": int(src.stat().st_size),
            "link": mode,
        }
        if src != files[0]:
            built = _hdf5_chunk_index_records(src)
            if built is not None:
                index, records = built
                chunk_index_payloads.append((entry, index, records))
        links.append(entry)
    if chunk_index_payloads and len(chunk_index_payloads) == max(0, len(files) - 1):
        index_rel = pathlib.Path("source") / "chunks.u64"
        index_path = out_path / index_rel
        index_path.parent.mkdir(parents=True, exist_ok=True)
        offset = 0
        with index_path.open("wb") as handle:
            for entry, index, records in chunk_index_payloads:
                payload = records.astype("<u8", copy=False).tobytes(order="C")
                handle.write(payload)
                index = dict(index)
                index["path"] = index_rel.as_posix()
                index["byte_offset"] = offset
                index["bytes"] = len(payload)
                entry["chunk_index"] = index["path"]
                entry["chunk_index_byte_offset"] = offset
                entry["chunk_index_bytes"] = len(payload)
                chunk_indexes.append(index)
                offset += len(payload)
    master_rel = pathlib.Path("source") / files[0].name
    return {
        "kind": "hdf5",
        "master": master_rel.as_posix(),
        "data_files": [entry["path"] for entry in links[1:]],
        "chunk_indexes": chunk_indexes,
        "link_mode": sorted({entry["link"] for entry in links}),
        "files": links,
        "note": (
            "Compressed HDF5 source files are served directly; no persistent "
            "float32 or complex64 BF reducer is stored in this folder."
        ),
    }


def _source_stack_files(files: list[pathlib.Path]) -> list[pathlib.Path]:
    """Return HDF5 files that hold scan-frame detector stacks."""

    if len(files) > 1:
        return files[1:]
    return files


def _detector_stack_shape(src: pathlib.Path) -> tuple[int, int, int, np.dtype]:
    """Return ``(frames, detector_rows, detector_cols, dtype)`` for ``src``."""

    import h5py

    with h5py.File(src, "r") as handle:
        dataset = _find_hdf5_stack(handle)
        frames, det_rows, det_cols = (int(v) for v in dataset.shape)
        return frames, det_rows, det_cols, np.dtype(dataset.dtype)


def _write_bf_column_source(
    files: list[pathlib.Path],
    out_path: pathlib.Path,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    """Write exact detector BF columns for browser-first ShowPtycho loading."""

    import h5py

    stack_files = _source_stack_files(files)
    if not stack_files:
        raise ValueError("ShowPtycho BF-column export found no HDF5 detector stack files.")

    bf_rows = np.asarray(calibration["bf_rows"], dtype=np.int64)
    bf_cols = np.asarray(calibration["bf_cols"], dtype=np.int64)
    if bf_rows.shape != bf_cols.shape or bf_rows.ndim != 1:
        raise ValueError("ShowPtycho BF-column export needs 1D bf_rows and bf_cols.")
    num_bf = int(bf_rows.size)
    if num_bf <= 0:
        raise ValueError("ShowPtycho BF-column export found no BF detector pixels.")

    shapes = [_detector_stack_shape(src) for src in stack_files]
    detector_shapes = {(det_rows, det_cols) for _, det_rows, det_cols, _ in shapes}
    if len(detector_shapes) != 1:
        raise ValueError(f"ShowPtycho source files have inconsistent detector shapes: {sorted(detector_shapes)!r}")
    det_rows, det_cols = next(iter(detector_shapes))
    if int(bf_rows.max()) >= det_rows or int(bf_cols.max()) >= det_cols:
        raise ValueError(
            "ShowPtycho BF mask is outside the source detector shape: "
            f"max row/col {(int(bf_rows.max()), int(bf_cols.max()))}, detector {(det_rows, det_cols)}."
        )
    plane = int(sum(frames for frames, _, _, _ in shapes))
    scan_shape = list(calibration.get("scan_region", {}).get("shape") or calibration.get("phase_shape") or [])

    source_dir = out_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = source_dir / "bf_columns.tmp.u16"
    if tmp_path.exists():
        tmp_path.unlink()
    columns_u16 = np.memmap(tmp_path, dtype="<u2", mode="w+", shape=(num_bf, plane))
    offset = 0
    max_value = 0
    for src, (frames, _det_rows, _det_cols, _dtype) in zip(stack_files, shapes, strict=True):
        with h5py.File(src, "r") as handle:
            dataset = _find_hdf5_stack(handle)
            for frame0 in range(0, frames, 1024):
                frame1 = min(frames, frame0 + 1024)
                frames_block = np.asarray(dataset[frame0:frame1])
                block = frames_block[:, bf_rows, bf_cols]
                if block.ndim != 2 or block.shape[1] != num_bf:
                    raise ValueError(
                        f"ShowPtycho BF-column slice from {src} returned {block.shape}, "
                        f"expected ({frame1 - frame0}, {num_bf})."
                    )
                if block.size:
                    max_value = max(max_value, int(np.max(block)))
                columns_u16[:, offset + frame0 : offset + frame1] = block.T.astype("<u2", copy=False)
        offset += frames
    columns_u16.flush()

    if max_value <= 255:
        rel = pathlib.Path("source") / "bf_columns.u8"
        final_path = out_path / rel
        if final_path.exists():
            final_path.unlink()
        columns_u8 = np.memmap(final_path, dtype="u1", mode="w+", shape=(num_bf, plane))
        for bf0 in range(0, num_bf, 256):
            bf1 = min(num_bf, bf0 + 256)
            columns_u8[bf0:bf1, :] = columns_u16[bf0:bf1, :].astype("u1", copy=False)
        columns_u8.flush()
        del columns_u8
        del columns_u16
        tmp_path.unlink(missing_ok=True)
        dtype = "uint8"
        bytes_per_value = 1
    else:
        rel = pathlib.Path("source") / "bf_columns.u16"
        final_path = out_path / rel
        if final_path.exists():
            final_path.unlink()
        del columns_u16
        tmp_path.replace(final_path)
        dtype = "uint16"
        bytes_per_value = 2

    return {
        "kind": "bf_columns",
        "path": rel.as_posix(),
        "url": rel.as_posix(),
        "dtype": dtype,
        "encoding": dtype,
        "num_bf": num_bf,
        "scan_shape": [int(v) for v in scan_shape],
        "plane": plane,
        "bytes_per_bf": int(plane * bytes_per_value),
        "bits_per_value": int(bytes_per_value * 8),
        "bytes": int(final_path.stat().st_size),
        "max_value": int(max_value),
        "note": "Exact detector BF columns; browser range-reads only the BF evidence needed on open.",
    }


def _reuse_bf_column_source(
    state: Any,
    out_path: pathlib.Path,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    """Link an existing exact MPS BF companion when geometry is unchanged."""

    source_path = state.bf_source_path
    if source_path is None or state.bf_source_dtype is None:
        raise TypeError("SSB export state does not contain an exact BF source.")
    if not source_path.is_file():
        raise FileNotFoundError(f"Exact BF companion not found: {source_path}")
    bf_rows = np.asarray(calibration["bf_rows"], dtype=np.int32)
    bf_cols = np.asarray(calibration["bf_cols"], dtype=np.int32)
    if not (
        np.array_equal(bf_rows, state.brightfield.rows)
        and np.array_equal(bf_cols, state.brightfield.cols)
    ):
        raise ValueError(
            "Export BF coordinates do not match the exact source selection."
        )
    scan_shape = tuple(
        int(value)
        for value in (
            calibration.get("scan_region", {}).get("shape")
            or calibration.get("phase_shape")
        )
    )
    if scan_shape != state.scan_shape:
        raise ValueError(
            f"Export scan shape {scan_shape} does not match exact source "
            f"shape {state.scan_shape}."
        )
    dtype = state.bf_source_dtype
    if dtype == np.dtype(np.uint8):
        encoding = "uint8"
        suffix = "u8"
    elif dtype == np.dtype(np.uint16):
        encoding = "uint16"
        suffix = "u16"
    else:
        raise TypeError(f"Unsupported exact BF source dtype: {dtype}.")
    rel = pathlib.Path("source") / f"bf_columns.{suffix}"
    final_path = (out_path / rel).resolve()
    try:
        source_path.relative_to(out_path.resolve())
    except ValueError:
        pass
    else:
        raise ValueError(
            "Cannot overwrite a ShowPtycho folder while reusing its own BF "
            "companion; choose a different --out folder."
        )
    link_mode = _link_or_copy(source_path, final_path)
    plane = int(np.prod(scan_shape))
    num_bf = int(bf_rows.size)
    return {
        "kind": "bf_columns",
        "path": rel.as_posix(),
        "url": rel.as_posix(),
        "dtype": encoding,
        "encoding": encoding,
        "num_bf": num_bf,
        "scan_shape": list(scan_shape),
        "plane": plane,
        "bytes_per_bf": int(plane * dtype.itemsize),
        "bits_per_value": int(dtype.itemsize * 8),
        "bytes": int(final_path.stat().st_size),
        "max_value": state.bf_source_max_value,
        "link": link_mode,
        "note": (
            "Reused exact detector BF columns; browser range-reads only the "
            "BF evidence needed on open."
        ),
    }


def export_showptycho_webgpu_folder(
    widget: Any,
    out_dir: str | pathlib.Path,
    *,
    title: str | None = None,
    overwrite: bool = True,
    source_master: str | pathlib.Path | None = None,
    decode_dtype: str = "uint16",
    webgpu_source: str = "bf_columns",
) -> pathlib.Path:
    """Export a ShowPtycho WebGPU folder backed by browser-ready source files."""

    if decode_dtype not in {"uint8", "uint16", "float32"}:
        raise ValueError(
            "decode_dtype must be 'uint8', 'uint16', or 'float32'; "
            f"got {decode_dtype!r}"
        )
    if webgpu_source not in {"bf_columns", "hdf5"}:
        raise ValueError(
            "webgpu_source must be 'bf_columns' or 'hdf5'; "
            f"got {webgpu_source!r}"
        )
    accel = widget._accel
    accel.set_rotation(float(widget.rotation_deg))
    state = accel.browser_state()
    _ensure_supported_webgpu_shape(state)

    out_path = pathlib.Path(out_dir).expanduser()
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} already exists")
    out_path.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for stale in (
            "ref_phase.f32",
            "ref_fft.f32",
            "ref_phase_variance.f32",
            "ref_amplitude.f32",
            "ref_phase.npy",
            "ref_fft.npy",
            "ref_products.npz",
            "PARITY.md",
            "cal.json",
            "manifest.json",
            "README.md",
            "serve_range.py",
        ):
            (out_path / stale).unlink(missing_ok=True)
        source_dir = out_path / "source"
        if source_dir.exists() or source_dir.is_symlink():
            if source_dir.is_dir() and not source_dir.is_symlink():
                shutil.rmtree(source_dir)
            else:
                source_dir.unlink()
        saves_dir = out_path / "saves"
        if saves_dir.exists() or saves_dir.is_symlink():
            if saves_dir.is_dir() and not saves_dir.is_symlink():
                shutil.rmtree(saves_dir)
            else:
                saves_dir.unlink()
        snapshots_dir = out_path / "snapshots"
        if snapshots_dir.exists() and snapshots_dir.is_dir():
            for stale in ("cal.json", "manifest.json", "README.md"):
                (snapshots_dir / stale).unlink(missing_ok=True)

    raw_source = source_master if source_master is not None else widget._source_file
    if not raw_source:
        raise ValueError(
            "ShowPtycho compressed-source export needs the original *_master.h5 path. "
            "Construct the widget with source_file=... or pass source_master=..."
        )
    master = pathlib.Path(raw_source).expanduser()
    master = master.resolve()
    cal = _folder_calibration(widget, state)
    if webgpu_source == "bf_columns":
        if state.bf_source_path is not None:
            bf_columns = _reuse_bf_column_source(
                state,
                out_path,
                cal,
            )
        else:
            source_files = collect_hdf5_family(master)
            bf_columns = _write_bf_column_source(source_files, out_path, cal)
        source = {
            "kind": "bf_columns",
            "bf_columns": bf_columns,
            "preferred_browser_source": "bf_columns",
            "note": (
                "Only exact bright-field detector columns are bundled; the "
                "private raw HDF5 acquisition is not included."
            ),
        }
        cal["source_file"] = "redacted_local_source"
        cal["source_transport"] = "bf_columns"
        cal["source_files"] = [bf_columns["path"]]
        cal["source_decode_dtype"] = bf_columns["dtype"]
        cal["persistent_bf_cache"] = False
        cal["bf_column_companion"] = True
        cal["bf_column_companion_path"] = bf_columns["path"]
        cal["bf_column_encoding"] = bf_columns["encoding"]
        cal["webgpu_source_policy"] = "bf_columns_preferred_exact"
    else:
        source_files = collect_hdf5_family(master)
        source = _prepare_hdf5_source_folder(
            master,
            out_path,
            files=source_files,
        )
        source["decode_dtype"] = decode_dtype
        cal["source_file"] = pathlib.Path(source["master"]).name
        cal["source_transport"] = "compressed_hdf5"
        cal["source_files"] = source["data_files"]
        cal["source_decode_dtype"] = decode_dtype
        cal["persistent_bf_cache"] = False
        cal["bf_column_companion"] = False
        cal["webgpu_source_policy"] = "compressed_hdf5_fallback"

    snapshots_dir = out_path / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    (snapshots_dir / "cal.json").write_text(json.dumps(cal, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 2,
        "format": "quantem.showptycho.webgpu.folder.v2",
        "title": title or "ShowPtycho",
        "index": "index.html",
        "calibration": "snapshots/cal.json",
        "source": source,
        "arrays": {},
        "persistent_arrays": [],
        "non_goals": [
            "no persistent BF-G cache",
            "no reference float32 image payloads",
            "no detector binning",
        ],
    }
    (snapshots_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    _write_empty_snapshots_manifest(out_path)
    if webgpu_source == "bf_columns":
        bf_columns = source["bf_columns"]
        browser_source = {
            "kind": "bf_columns",
            "url": bf_columns["url"],
            "dtype": bf_columns["dtype"],
            "encoding": bf_columns["encoding"],
            "numBf": bf_columns["num_bf"],
            "plane": bf_columns["plane"],
            "scanShape": bf_columns["scan_shape"],
            "bytesPerBf": bf_columns["bytes_per_bf"],
            "bitsPerValue": bf_columns["bits_per_value"],
        }
    else:
        browser_source = {
            "kind": "hdf5",
            "masterUrl": source["master"],
            "dataUrls": source["data_files"],
            "chunkIndexes": source.get("chunk_indexes", []),
            "decodeDtype": decode_dtype,
        }
    _write_embedded_widget_html(
        widget,
        out_path / "index.html",
        title=str(manifest["title"]),
        calibration=cal,
        h5_source=browser_source,
    )
    if webgpu_source == "bf_columns":
        source_note = (
            "The browser range-reads exact bright-field detector columns from "
            "`source/bf_columns.*` by default, so opening the viewer does not "
            "decode the compressed HDF5 stack unless a fallback path is needed."
        )
    else:
        source_note = (
            "The browser reads the original compressed HDF5 master/data files under "
            "`source/`, decompresses the selected BF evidence with WebGPU, and builds "
            "BF reducers transiently in GPU memory."
        )
    (snapshots_dir / "README.md").write_text(
        "# ShowPtycho WebGPU Folder\n\n"
        "Two ways to open this review - no install needed for the first:\n\n"
        "1. **Double-click** `ShowPtycho.command` (macOS) - it serves this folder "
        "and opens the viewer in Chrome. Or double-click `index.html`, click "
        "**Open data folder**, and select this folder.\n"
        "2. **CLI**: `quantem show <this folder>` serves it and opens the browser "
        "automatically (needs the `quantem-widget` package). Any other Range-capable "
        "static server works too.\n\n"
        f"{source_note} The folder stores detector evidence rather than derived "
        "Fourier caches, reference images, or detector-binned data.\n",
        encoding="utf-8",
    )
    # Double-click launcher (see quantem.widget.command_launcher).
    from quantem.widget.command_launcher import write_command_launcher

    write_command_launcher(out_path, "ShowPtycho")
    return out_path
