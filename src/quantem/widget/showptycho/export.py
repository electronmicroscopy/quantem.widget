"""Browser folder for a ShowPtycho: the same widget UI over exact bright-field counts, no kernel.

The folder holds ``index.html`` (the widget with its state embedded and WebGPU switched on),
``source/bf_columns.*`` (the exact detector counts of the bright-field disk, ANS-coded ``.qem`` from a
CUDA session or the ``.u8`` / ``.u16`` companion an MPS session opened), ``snapshots/`` (the calibration
the browser kernels start from, a manifest and the saved aberration states) and a double-click
``ShowPtycho.command`` that serves the folder. No float32 image, complex reducer or copy of the raw
HDF5 acquisition is written: the browser rebuilds the SSB reducers from the counts on its own GPU.
"""

import json
import math
import os
import pathlib
import shutil

import numpy as np

from quantem.widget.command_launcher import write_command_launcher
from quantem.widget.export import write_widget_html

# the browser kernels are specialised per native scan size
BROWSER_SCAN_SIZES = (128, 256, 512, 1024)
# resident scan spectra the browser may keep, in GiB; raising it never drops detector pixels
BROWSER_GPU_MEMORY_GB = 4.5
# live-session traits the embedded page overrides; they are restored after the page is written
LIVE_TRAITS = (
    "webgpu_preview_enabled", "webgpu_standalone", "webgpu_cal_json", "webgpu_h5_source_json", "webgpu_preview_status",
    "phase_bytes", "phase_width", "phase_height", "stars_path", "calibration_path", "crop_refit_available",
)


def write_showptycho_folder(widget, path: str | pathlib.Path | None, title: str | None) -> pathlib.Path:
    """Write the review folder for ``widget`` and return its path (see the module docstring for the layout)."""
    ssb = widget._ssb
    ssb.set_rotation(float(widget.rotation_deg))
    state = ssb.browser_state()
    rows, cols = state.scan_shape
    if rows != cols or rows not in BROWSER_SCAN_SIZES:
        raise NotImplementedError(f"ShowPtycho WebGPU folder export supports square 128, 256, 512, or 1024 crops; got {rows}x{cols}.")
    if path is not None:
        folder = pathlib.Path(path).expanduser()
    elif widget._source_file:
        master = pathlib.Path(widget._source_file).expanduser().resolve()
        stem = master.name[: -len("_master.h5")] if master.name.endswith("_master.h5") else master.stem
        folder = master.parent / f"{stem}_showptycho"
    else:
        folder = pathlib.Path.cwd() / "showptycho_export"
    folder.mkdir(parents=True, exist_ok=True)
    source_dir = folder / "source"
    # a re-export reuses the counts it wrote last time instead of re-encoding them
    own_source = state.bf_source_path is not None and state.bf_source_path.is_file() and state.bf_source_path.resolve().parent == source_dir.resolve()
    if not own_source and (source_dir.exists() or source_dir.is_symlink()):
        if source_dir.is_dir() and not source_dir.is_symlink():
            shutil.rmtree(source_dir)
        else:
            source_dir.unlink()
    if ssb.backend == "cuda" and not (state.bf_source_path is not None and state.bf_source_path.suffix == ".qem" and state.bf_source_path.exists()):
        ssb.export_brightfield(source_dir / "bf_columns.qem")
        state = ssb.browser_state()
    if state.bf_source_path is None or state.bf_source_dtype is None:
        raise RuntimeError("ShowPtycho export needs exact bright-field counts: open the acquisition with SSB.open(...) or use a CUDA session.")

    c10, c12, phi12_deg = widget._current()
    row_start, row_stop, col_start, col_stop = widget._scan_region
    # geometry arrays are stored in the browser kernels' int32 / float32, the values they compute with
    calibration = {
        "schema_version": 1,
        "kind": "showptycho_webgpu_folder",
        "source_file": "redacted_local_source",
        "source_calibration": "redacted_local_calibration",
        "scan_region": {"row_start": row_start, "row_stop": row_stop, "col_start": col_start, "col_stop": col_stop,
                        "shape": [row_stop - row_start, col_stop - col_start], "source_shape": list(widget._source_scan_shape)},
        "backend_reference": "quantem.gpu SSBProtocol.reconstruct_with_loss",
        "precision": {"real_dtype": state.precision.real_dtype, "complex_dtype": state.precision.complex_dtype},
        "bf_radius_px": state.brightfield.radius_px,
        "num_bf": state.num_bf,
        "g_shape": [state.num_bf, rows, cols],
        "g_dtype": "complex64_interleaved_re_im_native_le",
        "phase_shape": [rows, cols],
        "phase_dtype": "float32_native_le",
        "detector_shape": list(state.brightfield.detector_shape),
        "bf_center": list(state.brightfield.center_row_col),
        "bf_rows": np.asarray(state.brightfield.rows).astype(np.int32).tolist(),
        "bf_cols": np.asarray(state.brightfield.cols).astype(np.int32).tolist(),
        "kx_bf": np.asarray(state.kx_bf).astype(np.float32).tolist(),
        "ky_bf": np.asarray(state.ky_bf).astype(np.float32).tolist(),
        "qx_1d": np.asarray(state.qx_1d).astype(np.float32).tolist(),
        "qy_1d": np.asarray(state.qy_1d).astype(np.float32).tolist(),
        "aperture_k": np.asarray(state.aperture_k).astype(np.float32).tolist(),
        "alpha_k2": np.asarray(state.alpha_k2).astype(np.float32).tolist(),
        "cos2phi_k": np.asarray(state.cos2phi_k).astype(np.float32).tolist(),
        "sin2phi_k": np.asarray(state.sin2phi_k).astype(np.float32).tolist(),
        "wavelength_A": state.wavelength_A,
        "semiangle_mrad": state.semiangle_rad * 1e3,
        "semiangle_rad": state.semiangle_rad,
        "scan_sampling_A": float(widget.pixel_size),
        "voltage_kV": float(ssb.voltage_kV),
        "det_sampling_mrad_px": [value * 1e3 for value in state.angular_sampling_rad],
        "sampling_A": list(state.sampling_A),
        "angular_sampling_rad": list(state.angular_sampling_rad),
        "rotation_angle_deg": float(widget.rotation_deg),
        "rotation_angle_rad": math.radians(float(widget.rotation_deg)),
        "aberrations": {"C10": c10, "C12": c12, "phi12": math.radians(phi12_deg), "phi12_deg": phi12_deg},
        "flip_phase": bool(widget.flip_phase),
        "dc_value": [float(state.dc_value.real), float(state.dc_value.imag)],
        "gpu_memory_gb": BROWSER_GPU_MEMORY_GB,
    }

    # the counts file: link the session's companion into source/ (hardlink, else symlink, else copy)
    dtype = state.bf_source_dtype
    if state.bf_source_path.suffix == ".qem":
        encoding, suffix = "qem", "qem"
    elif dtype == np.dtype(np.uint8):
        encoding, suffix = "uint8", "u8"
    else:
        encoding, suffix = "uint16", "u16"
    relative_path = pathlib.Path("source") / f"bf_columns.{suffix}"
    target = (folder / relative_path).resolve()
    if state.bf_source_path.resolve() == target:
        link_mode = "encoded_here"
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        try:
            os.link(state.bf_source_path, target)
            link_mode = "hardlink"
        except OSError:
            try:
                target.symlink_to(state.bf_source_path)
                link_mode = "symlink"
            except OSError:
                shutil.copy2(state.bf_source_path, target)
                link_mode = "copy"
    bf_columns = {
        "kind": "bf_columns", "path": relative_path.as_posix(), "url": relative_path.as_posix(), "dtype": dtype.name, "encoding": encoding,
        "num_bf": state.num_bf, "scan_shape": [rows, cols], "plane": rows * cols, "bytes_per_bf": rows * cols * dtype.itemsize,
        "bits_per_value": dtype.itemsize * 8, "bytes": target.stat().st_size, "max_value": state.bf_source_max_value, "link": link_mode,
        "note": ("Lossless ANS counts decoded on the GPU; changing output sampling reuses the resident native scan spectra."
                 if encoding == "qem" else "Exact detector BF columns, read once and retained on the GPU."),
    }
    calibration.update({
        "source_transport": "bf_columns", "source_files": [bf_columns["path"]], "source_decode_dtype": bf_columns["dtype"],
        "persistent_bf_cache": False, "bf_column_companion": True, "bf_column_companion_path": bf_columns["path"],
        "bf_column_encoding": encoding, "webgpu_source_policy": "bf_columns_preferred_exact",
    })

    snapshots = folder / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    (snapshots / "cal.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 2,
        "format": "quantem.showptycho.webgpu.folder.v2",
        "title": title or "ShowPtycho",
        "index": "index.html",
        "calibration": "snapshots/cal.json",
        "source": {"kind": "bf_columns", "bf_columns": bf_columns, "preferred_browser_source": "bf_columns",
                   "note": "Only exact bright-field detector columns are bundled; the private raw HDF5 acquisition is not included."},
        "arrays": {},
        "persistent_arrays": [],
        "non_goals": ["no persistent BF-G cache", "no reference float32 image payloads", "no detector binning"],
    }
    (snapshots / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not (snapshots / "snapshots.json").exists():
        (snapshots / "snapshots.json").write_text("[]\n", encoding="utf-8")
    (snapshots / "README.md").write_text(
        "# ShowPtycho WebGPU Folder\n\n"
        "Two ways to open this review - no install needed for the first:\n\n"
        "1. **Double-click** `ShowPtycho.command` (macOS) - it serves this folder "
        "and opens the viewer in Chrome. Or double-click `index.html`, click "
        "**Open data folder**, and select this folder.\n"
        "2. **Serve it**: run `python -m http.server` in this folder (or any other "
        "Range-capable static server) and open `index.html`.\n\n"
        "The browser loads exact bright-field detector counts from "
        "`source/bf_columns.*` by default, so opening the viewer does not "
        "decode the compressed HDF5 stack unless a fallback path is needed. The folder stores detector evidence rather than derived "
        "Fourier caches, reference images, or detector-binned data.\n",
        encoding="utf-8",
    )

    # embed the widget with the folder source switched on, then restore the live-session traits
    browser_source = {"kind": "bf_columns", "url": bf_columns["url"], "dtype": bf_columns["dtype"], "encoding": encoding, "numBf": state.num_bf,
                      "plane": rows * cols, "scanShape": [rows, cols], "bytesPerBf": bf_columns["bytes_per_bf"], "bitsPerValue": bf_columns["bits_per_value"]}
    live_values = {name: getattr(widget, name) for name in LIVE_TRAITS}
    try:
        widget.webgpu_preview_enabled = True
        widget.webgpu_standalone = True
        widget.crop_refit_available = False  # refitting needs the live Python session
        widget.webgpu_cal_json = json.dumps(calibration)
        widget.webgpu_h5_source_json = json.dumps(browser_source)
        widget.stars_path = "snapshots/snapshots.json"
        widget.calibration_path = "snapshots/calibration.json"
        widget.phase_bytes, widget.phase_width, widget.phase_height = b"", 0, 0
        widget.webgpu_preview_status = "WebGPU folder ready: browser loads exact BF counts and builds reducers transiently."
        write_widget_html(folder / "index.html", widget, str(manifest["title"]))
    finally:
        for name, value in live_values.items():
            setattr(widget, name, value)
    write_command_launcher(folder, "ShowPtycho")
    return folder
