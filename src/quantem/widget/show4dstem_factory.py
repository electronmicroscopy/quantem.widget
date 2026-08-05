"""Public Show4DSTEM factory.

``quantem.widget.Show4DSTEM`` is intentionally one user-facing API. This module
keeps the backend dispatch out of ``quantem.widget.__init__`` so the package
initializer stays a public export list instead of carrying backend policy.

The raw-Metal/MPS implementation lives in ``show4dstem_mps`` because it has
backend-specific trait/status wiring. Users reach it only through
``Show4DSTEM(quantem.gpu.io.load(..., backend="mps"))``.
"""

from __future__ import annotations

import gc
import math
import os
import pathlib
import types
import warnings
from typing import Any

from quantem.gpu.io.load import LoadResult
from quantem.widget.show4dstem import Show4DSTEM as _Show4DSTEMBase


def _payload(value: Any) -> Any:
    """Unwrap ``io.load`` output to the object used for backend routing."""
    return value.data if isinstance(value, LoadResult) else value


def is_mps_show4dstem_payload(value: Any) -> bool:
    """Return True when ``value`` should use the raw-Metal Show4DSTEM path.

    MPS HDF5 loads expose chunked buffers. Already-wrapped Metal frame proxies
    expose ``_is_gpu_frames``; those should only be treated as MPS when their
    device string says MPS. CUDA 5D dataset wrappers are left for the base
    viewer.
    """
    payload = _payload(value)
    payload_device = str(getattr(payload, "device", ""))
    is_mps_frames = (
        bool(getattr(payload, "_is_gpu_frames", False)) and "mps" in payload_device
    )
    return hasattr(payload, "chunks") or is_mps_frames


def show4dstem_backend_kind(value: Any) -> str:
    """Classify the backend family the public factory will select."""
    payload = _payload(value)
    if is_mps_show4dstem_payload(payload):
        return "mps"
    return "base"


def _build_mps_viewer(data: Any, **kwargs: Any) -> Any:
    """Build the sampling-aware MPS viewer lazily to keep normal imports light."""
    from quantem.widget.show4dstem_mps import _show4dstem_mps

    return _show4dstem_mps(data, **kwargs)


def _apply_loadresult_labels(data: Any, payload: Any, kwargs: dict[str, Any]) -> None:
    """Label a CUDA 5D ``load([...])`` stack as datasets."""
    if not isinstance(data, LoadResult) or getattr(payload, "ndim", 0) != 5:
        return
    meta = getattr(data, "metadata", {}) or {}
    kwargs.setdefault("frame_dim_label", "Dataset")
    names = meta.get("file_names")
    if names is not None:
        kwargs.setdefault("frame_labels", list(names))


def Show4DSTEM(data: Any, **kwargs: Any) -> Any:
    """Open a 4D-STEM viewer over ``load(...)`` output, on any backend.

    Canonical examples::

        from quantem.gpu.io import load
        from quantem.widget import Show4DSTEM

        Show4DSTEM(load("a.h5"))                         # auto: CUDA / MPS
        Show4DSTEM(load("a.h5", backend="mps"))          # explicit Apple Metal load
        Show4DSTEM(load(["a.h5", "b.h5"], det_bin=4))    # many datasets, one slider
        Show4DSTEM(load("a.h5"), backend="webgpu")       # browser WebGPU compute

        w = Show4DSTEM(load("a.h5"), backend="webgpu", offline_codec="bslz4",
                       data_url="show4dstem-data")
        w.export_html("show4dstem.html")

    Dispatch is automatic from what ``load`` returns:
      - Apple Silicon MPS single-file loads use the raw-Metal real-time viewer.
      - Apple Silicon MPS multi-file loads use a lazy handle; dataset 0 shows
        immediately and later datasets fill behind the dataset slider.
      - CUDA CuPy loads use the base viewer with ``quantem.gpu`` RawKernel
        reducers for BF/DF/ADF interaction.
      - Browser WebGPU performs detector reductions in the browser.
    """
    payload = _payload(data)
    if is_mps_show4dstem_payload(payload):
        return _build_mps_viewer(payload, **kwargs)

    _apply_loadresult_labels(data, payload, kwargs)
    return _Show4DSTEMBase(payload, **kwargs)


def _normalise_gpus(gpus) -> list[int] | None:
    if gpus is None:
        return None
    if isinstance(gpus, str):
        if gpus.strip().lower() != "all":
            raise ValueError(
                "gpus must be None, 'all', an int, or a non-empty sequence of GPU ids."
            )
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise ValueError("gpus='all' requires at least one visible CUDA device.")
        return list(range(torch.cuda.device_count()))
    if isinstance(gpus, int):
        return [int(gpus)]
    out = [int(gpu) for gpu in gpus]
    if not out:
        raise ValueError(
            "gpus must be None, 'all', an int, or a non-empty sequence of GPU ids."
        )
    return out


def _master_label(master) -> str:
    name = os.path.basename(str(master))
    return name[: -len("_master.h5")] if name.endswith("_master.h5") else name


def _master_file_contract(master: Any) -> dict[str, Any]:
    """Read the raw shape and dtype needed to validate a watched master."""
    import numpy as np
    from quantem.gpu.io import inspect

    report = inspect(str(master))
    required = {
        "scan_shape": report.scan_shape,
        "detector_shape": report.detector_shape,
        "n_frames": report.actual_frames,
        "dtype": report.dtype,
    }
    missing = [name for name, value in required.items() if value is None]
    if not report.ready or missing:
        problems: list[str] = []
        if not report.ready:
            problems.append(report.reason or "the source is not ready")
        if missing:
            problems.append(
                "inspection did not provide " + ", ".join(missing)
            )
        action = report.action or (
            "Verify that the master and external data are complete."
        )
        raise ValueError(
            f"Cannot open 4D-STEM master {str(master)!r}: "
            f"{'; '.join(problems)}. {action}"
        )
    return {
        "scan_shape": report.scan_shape,
        "detector_shape": report.detector_shape,
        "n_frames": report.actual_frames,
        "dtype": np.dtype(report.dtype).str,
    }


def _largest_compatible_master_group(
    masters: list[Any],
    *,
    verbose: bool = False,
) -> list[Any]:
    """Return the largest group sharing scan/detector shape metadata."""
    if len(masters) <= 1:
        return masters
    from quantem.gpu.io import inspect

    groups: dict[tuple[Any, Any, Any], list[Any]] = {}
    for master in masters:
        try:
            report = inspect(str(master))
            key = (
                report.scan_shape,
                report.detector_shape,
                report.actual_frames,
            )
        except Exception:
            key = (None, None, None)
        groups.setdefault(key, []).append(master)
    if len(groups) <= 1:
        return masters
    key, compatible = max(groups.items(), key=lambda item: len(item[1]))
    skipped = len(masters) - len(compatible)
    if verbose:
        warnings.warn(
            "Show4DSTEM.from_folder found mixed 4D-STEM shapes in the folder; "
            f"using the largest compatible group ({len(compatible)}/{len(masters)}) "
            f"with scan_shape={key[0]}, detector_shape={key[1]}. "
            f"Skipped {skipped} master file{'s' if skipped != 1 else ''}. "
            "Use scan_size= or a narrower pattern= to select a different group.",
            RuntimeWarning,
            stacklevel=3,
        )
    return compatible


def _filter_ready_masters(
    masters: list[Any],
    ready_check: Any,
    *,
    verbose: bool,
) -> list[Any]:
    """Return browse-ready masters, warning once for skipped files."""
    ready: list[Any] = []
    skipped: list[Any] = []
    for master in masters:
        if ready_check(master):
            ready.append(master)
        else:
            skipped.append(master)
    if skipped and verbose:
        names = ", ".join(_master_label(master) for master in skipped[:12])
        if len(skipped) > 12:
            names += f", ... (+{len(skipped) - 12} more)"
        warnings.warn(
            "Show4DSTEM.from_folder skipped "
            f"{len(skipped)} incomplete or unreadable master file"
            f"{'s' if len(skipped) != 1 else ''}: {names}.",
            RuntimeWarning,
            stacklevel=3,
        )
    return ready


def _multiple_view_requested(view_mode: Any) -> bool:
    value = str(view_mode or "").strip().lower().replace("-", "_")
    return value in {"multiple", "multi", "compare", "grid"}


def _dtype_token(dtype: Any) -> str | None:
    """Normalize public browse dtype spellings for backend capability checks."""
    if dtype is None:
        return None
    return str(dtype).strip().lower().replace("_", "")


def _mps_output_dtype(dtype: Any) -> str | None:
    """Return the raw-Metal folder output dtype requested by the public API."""
    token = _dtype_token(dtype)
    if token in {None, "", "auto", "native", "full", "exact"}:
        return None
    if token in {"u8", "uint8"}:
        return "u8"
    if token in {"u16", "uint16"}:
        return "u16"
    raise ValueError(
        "Show4DSTEM.from_folder(..., backend='mps') currently supports "
        f"dtype='auto', 'u8', or 'u16'; got dtype={dtype!r}."
    )


def _is_recoverable_allocation_error(exc: BaseException) -> bool:
    """True when a loader failed from memory pressure, not bad input data."""
    if isinstance(exc, MemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message or "oom" in message


def _format_memory(nbytes: int) -> str:
    if nbytes >= 1 << 30:
        return f"{nbytes / (1 << 30):.1f} GB"
    if nbytes >= 1 << 20:
        return f"{nbytes / (1 << 20):.0f} MB"
    if nbytes >= 1 << 10:
        return f"{nbytes / (1 << 10):.0f} KB"
    return f"{nbytes} B"


def _cuda_memory_label(torch_module: Any, gpu_ids: list[int] | None) -> str:
    """Return compact CUDA memory text for a failed folder load."""
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return ""
    if gpu_ids is None:
        devices = list(range(cuda.device_count()))
    else:
        devices = list(gpu_ids)
    parts: list[str] = []
    for idx in devices:
        try:
            with cuda.device(int(idx)):
                free, total = cuda.mem_get_info(int(idx))
            used = int(total - free)
            parts.append(
                f"cuda:{int(idx)} {_format_memory(used)}/{_format_memory(int(total))}"
            )
        except Exception:
            continue
    return "GPU " + " | ".join(parts) if parts else ""


def _reclaim_failed_load_memory(torch_module: Any, gpu_ids: list[int] | None) -> None:
    """Release unreferenced CUDA/CuPy pools after a failed folder load."""
    gc.collect()
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and cuda.is_available():
        devices = list(range(cuda.device_count())) if gpu_ids is None else list(gpu_ids)
        for idx in devices:
            try:
                with cuda.device(int(idx)):
                    cuda.empty_cache()
                    if hasattr(cuda, "ipc_collect"):
                        cuda.ipc_collect()
            except Exception:
                pass
    try:
        import cupy as cp

        count = cp.cuda.runtime.getDeviceCount()
        devices = list(range(count)) if gpu_ids is None else list(gpu_ids)
        for idx in devices:
            try:
                with cp.cuda.Device(int(idx)):
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass
    except Exception:
        pass


def _folder_memory_warning(
    *,
    folder_path: pathlib.Path,
    memory_label: str,
) -> str:
    memory_text = f" Current memory: {memory_label}." if memory_label else ""
    return (
        "GPU memory limited: Show4DSTEM.from_folder could not load the first "
        f"dataset from {folder_path}.{memory_text} Nothing was loaded. Free memory "
        "with cleanup(w) or w.free() for existing widgets, close other GPU jobs, "
        "restart the kernel, or reopen with fewer panels/lower page_budget. For "
        "shareable reports, export with detector/real-space binning and uint8."
    )


def _memory_limited_folder_widget(
    *,
    folder_path: pathlib.Path,
    memory_label: str,
    warning: str,
) -> Any:
    """Return a tiny placeholder widget carrying a visible memory warning."""
    import torch

    widget = _Show4DSTEMBase(
        torch.zeros((2, 2, 2, 2), dtype=torch.uint8, device="cpu"),
        title=f"Show4DSTEM folder: {folder_path.name} (not loaded)",
        precompute_virtual_images=False,
        verbose=False,
        view_mode="single",
    )
    widget.gpu_memory_label = memory_label
    widget.memory_warning = warning
    widget.compare_status = "GPU memory limited: folder data was not loaded."
    return widget


def _warn_mps_from_folder_limits(
    *,
    dtype: Any,
    page_budget: int | str | None,
    page_max_vram_fraction: float,
    page_reserve_vram_bytes: int | None,
    page_max_vram_bytes: int | dict | None,
    preload_initial_page: bool | int,
) -> None:
    """Warn for public from_folder knobs the raw-Metal MPS path cannot honor."""
    messages: list[str] = []
    token = _dtype_token(dtype)
    if token in {"u8", "uint8"}:
        messages.append(
            "MPS folder dtype='u8' uses browse clipping before raw-Metal "
            "interaction. Use dtype='auto' for native detector counts."
        )
    else:
        _mps_output_dtype(dtype)

    page_options_changed = (
        page_budget is not None
        or float(page_max_vram_fraction) != 0.98
        or page_reserve_vram_bytes is not None
        or page_max_vram_bytes is not None
        or preload_initial_page is not True
    )
    if page_options_changed:
        messages.append(
            "Dataset5dstem paging/preload options are ignored; the raw-Metal MPS "
            "folder viewer uses LazyMacbookDatasets/MultiChunkedFrames rather "
            "than Dataset5dstem.page()."
        )
    if messages:
        warnings.warn(" ".join(messages), RuntimeWarning, stacklevel=3)


def _attach_mps_folder_methods(
    viewer: Any,
    live: Any,
    *,
    folder: pathlib.Path,
    pattern: str,
    recursive: bool,
    scan_size: int | None,
    ready_only: bool,
) -> Any:
    """Expose the from_folder polling API on an MPS lazy viewer instance."""
    from quantem.widget._folder_watch_status import set_folder_watch_status

    viewer._mps_folder_live = live

    def publish_status(state: str, detail: str = "") -> None:
        publisher = getattr(viewer, "_publish_mps_folder_watch_status", None)
        if callable(publisher):
            publisher(state, detail)
        else:
            set_folder_watch_status(viewer, state, detail)

    status_setter = getattr(live, "set_status_callback", None)
    if callable(status_setter):
        status_setter(publish_status)
    else:
        set_folder_watch_status(viewer, "hidden", "")

    def poll_folder(self, *, async_: bool = False) -> list[int]:
        scan_shape = (int(scan_size), int(scan_size)) if scan_size else None
        return live.poll(
            folder,
            pattern=pattern,
            recursive=recursive,
            scan_shape=scan_shape,
            ready_only=ready_only,
            async_=async_,
            require_stable=True,
        )

    def watch_folder(self, *, interval: float = 2.0) -> Any:
        scan_shape = (int(scan_size), int(scan_size)) if scan_size else None
        live.watch(
            folder,
            interval=interval,
            pattern=pattern,
            recursive=recursive,
            scan_shape=scan_shape,
            ready_only=ready_only,
        )
        return self

    def stop_folder_watch(self) -> None:
        live.stop()

    viewer.poll_folder = types.MethodType(poll_folder, viewer)
    viewer.watch_folder = types.MethodType(watch_folder, viewer)
    viewer.stop_folder_watch = types.MethodType(stop_folder_watch, viewer)
    return viewer


def from_folder(
    folder,
    *,
    pattern: str = "*_master.h5",
    recursive: bool = True,
    scan_size: int | None = None,
    max_masters: int | None = None,
    min_masters: int | None = None,
    ready_only: bool = True,
    gpus=None,
    page_budget: int | str | None = "auto",
    det_bin: int = 4,
    dtype: str = "auto",
    backend: str | None = None,
    load_kwargs: dict[str, Any] | None = None,
    watch: bool = True,
    watch_interval: float = 2.0,
    page_max_vram_fraction: float = 0.98,
    page_reserve_vram_bytes: int | None = None,
    page_max_vram_bytes: int | dict | None = None,
    view_mode: str = "multiple",
    columns: int | None = None,
    page_size: int | None = None,
    preload_all_if_fits: bool = True,
    warm_cache: bool = False,
    preview_cache: bool | str = "auto",
    preview_cache_dir: str | pathlib.Path | None = None,
    preview_cache_max_bytes: int | None = 4 << 30,
    rebuild_preview_cache: bool = False,
    verbose: bool = False,
    **viewer_kwargs: Any,
) -> Any:
    """Open a lazy, folder-backed Show4DSTEM over ready ``*_master.h5`` files.

    This is the direct many-master browse API. It differs from ``load(folder)``
    by keeping the folder as lazy slots instead of materializing every master.
    CUDA multiple views load the initial visible page for first paint, then
    ``preload_all_if_fits=True`` fills every unhidden dataset in the background
    when the complete shape/dtype footprint fits the selected GPU budget. Larger
    series keep full-resolution lazy paging. Folder watching is enabled by
    default. On the CUDA ``Dataset5dstem`` path, new ready masters are
    appended without rebuilding the widget, then join complete-series preload
    when the updated footprint still fits. Otherwise they remain cold until a
    page needs them. The MPS backend keeps its separate live polling architecture
    and does not share this reduced-page cache lifecycle. Set ``watch=False`` for
    a fixed folder snapshot.

    ``columns`` controls the grid width and ``page_size`` controls how many
    datasets are shown and preloaded together. Set
    ``preview_cache="auto"`` persists exact reduced BF/ABF/ADF/HAADF previews
    in the user cache so a later widget can paint them while current raw data
    refreshes. ``warm_cache=True`` proactively fills all four standard presets
    in bounded background batches while keeping raw 4D data lazy.
    """
    import torch
    from quantem.gpu import io as gpu_io
    from quantem.gpu.device import resolve
    from quantem.widget.data import Dataset5dstem

    if backend in (None, "auto") and gpus is not None:
        backend = "cuda"
    else:
        backend = resolve(backend)

    preload_initial_page = viewer_kwargs.pop("preload_initial_page", True)
    compare_cols = int(3 if columns is None else columns)
    compare_max_panels = int(12 if page_size is None else page_size)
    if compare_cols < 0:
        raise ValueError(f"columns must be >= 0, got {compare_cols}")
    if compare_max_panels < 1:
        raise ValueError(f"page_size must be >= 1, got {compare_max_panels}")
    if preview_cache_max_bytes is not None and int(preview_cache_max_bytes) < 0:
        raise ValueError(
            "preview_cache_max_bytes must be >= 0 or None, got "
            f"{preview_cache_max_bytes}"
        )
    if max_masters is not None:
        max_masters = int(max_masters)
        if max_masters < 1:
            raise ValueError(f"max_masters must be >= 1, got {max_masters}")
    if min_masters is not None:
        min_masters = int(min_masters)
        if min_masters < 1:
            raise ValueError(f"min_masters must be >= 1, got {min_masters}")
    if page_size is not None and preload_initial_page is True:
        preload_initial_page = compare_max_panels

    if backend == "mps":
        from quantem.gpu import io as gpu_io

        if gpus is not None:
            raise ValueError(
                "Show4DSTEM.from_folder(..., backend='mps') does not accept gpus=. "
                "gpus= selects CUDA devices; omit it for Apple MPS."
            )
        if load_kwargs:
            raise ValueError(
                "Show4DSTEM.from_folder(..., backend='mps') does not accept "
                "load_kwargs; use the typed folder options directly."
            )
        _warn_mps_from_folder_limits(
            dtype=dtype,
            page_budget=page_budget,
            page_max_vram_fraction=page_max_vram_fraction,
            page_reserve_vram_bytes=page_reserve_vram_bytes,
            page_max_vram_bytes=page_max_vram_bytes,
            preload_initial_page=preload_initial_page,
        )
        folder_path = pathlib.Path(folder).expanduser().resolve()
        scan_shape = (int(scan_size), int(scan_size)) if scan_size else None
        masters = gpu_io.discover(
            str(folder_path),
            pattern=pattern,
            recursive=recursive,
            scan_shape=scan_shape,
            verbose=False,
        )
        if ready_only:
            masters = _filter_ready_masters(
                list(masters),
                lambda master: gpu_io.inspect(master).ready,
                verbose=bool(verbose),
            )
        if not masters:
            state = "ready " if ready_only else ""
            raise ValueError(
                f"No {state}{pattern!r} files found in {folder_path}. "
                "Wait for linked data files to finish writing, or pass "
                "ready_only=False if you know the masters are complete."
            )
        masters = _largest_compatible_master_group(
            list(masters),
            verbose=bool(verbose),
        )
        if min_masters is not None and len(masters) < min_masters:
            raise ValueError(
                f"Show4DSTEM.from_folder requires at least {min_masters} "
                f"compatible master(s), but found {len(masters)}."
            )
        if max_masters is not None:
            masters = masters[:max_masters]
        expected_contract = _master_file_contract(masters[0])

        def validate_mps_master(master: Any) -> None:
            candidate = _master_file_contract(master)
            # The MPS loader normalizes supported Arina browse dtypes to
            # uint16 chunk-backed data, so a uint32/uint16 source mix is valid
            # as long as the geometry and frame count match.
            mismatches = [
                name
                for name in ("scan_shape", "detector_shape", "n_frames")
                if candidate.get(name) != expected_contract.get(name)
            ]
            if mismatches:
                observed = ", ".join(
                    f"{name}={candidate.get(name)!r}" for name in mismatches
                )
                expected = ", ".join(
                    f"{name}={expected_contract.get(name)!r}" for name in mismatches
                )
                raise ValueError(
                    f"Incompatible 4D-STEM master {_master_label(master)!r}: "
                    f"{observed}; expected {expected}. Use scan_size= or a "
                    "narrower pattern= for a uniform folder."
                )

        compatible_masters: list[Any] = []
        skipped_contracts: list[Any] = []
        for master in masters:
            try:
                validate_mps_master(master)
            except ValueError:
                skipped_contracts.append(master)
            else:
                compatible_masters.append(master)
        if skipped_contracts and verbose:
            warnings.warn(
                "Show4DSTEM.from_folder skipped "
                f"{len(skipped_contracts)} MPS master file"
                f"{'s' if len(skipped_contracts) != 1 else ''} whose scan, detector, "
                "or frame-count contract differs from the first ready master.",
                RuntimeWarning,
                stacklevel=2,
            )
        masters = compatible_masters
        if warm_cache:
            viewer_kwargs.setdefault(
                "compare_cache_pages",
                max(1, math.ceil(len(masters) / compare_max_panels)) * 4,
            )

        loaded = gpu_io.load(
            masters,
            backend="mps",
            det_bin=det_bin,
            scan_shape=scan_shape,
            dtype=_mps_output_dtype(dtype),
            verbose=verbose,
        )
        live = loaded.data
        viewer = Show4DSTEM(
            live,
            view_mode=view_mode,
            compare_cols=compare_cols,
            compare_max_panels=compare_max_panels,
            page_budget=page_budget,
            page_device=None,
            page_max_vram_fraction=page_max_vram_fraction,
            page_reserve_vram_bytes=page_reserve_vram_bytes,
            page_max_vram_bytes=page_max_vram_bytes,
            verbose=verbose,
            **viewer_kwargs,
        )
        viewer = _attach_mps_folder_methods(
            viewer,
            live,
            folder=folder_path,
            pattern=pattern,
            recursive=recursive,
            scan_size=scan_size,
            ready_only=ready_only,
        )
        if watch:
            viewer.watch_folder(interval=watch_interval)
        if warm_cache and callable(getattr(viewer, "warm_compare_cache", None)):
            viewer.warm_compare_cache(background=True)
        return viewer

    gpu_ids = _normalise_gpus(gpus)
    folder_path = pathlib.Path(folder).expanduser().resolve()
    scan_shape = (int(scan_size), int(scan_size)) if scan_size else None
    masters = gpu_io.discover(
        str(folder_path),
        pattern=pattern,
        recursive=recursive,
        scan_shape=scan_shape,
        verbose=False,
    )
    if ready_only:
        masters = _filter_ready_masters(
            list(masters),
            lambda master: gpu_io.inspect(master).ready,
            verbose=bool(verbose),
        )
    if not masters:
        raise ValueError(
            f"No ready {pattern!r} files found in {folder_path}. "
            "Wait for linked data files to finish writing, or pass ready_only=False "
            "if you know the masters are complete."
        )
    masters = _largest_compatible_master_group(
        list(masters),
        verbose=bool(verbose),
    )
    if min_masters is not None and len(masters) < min_masters:
        raise ValueError(
            f"Show4DSTEM.from_folder requires at least {min_masters} "
            f"compatible master(s), but found {len(masters)}."
        )
    if max_masters is not None:
        masters = masters[:max_masters]
    if warm_cache:
        viewer_kwargs.setdefault(
            "compare_cache_pages",
            max(1, math.ceil(len(masters) / compare_max_panels)) * 4,
        )

    load_options = dict(load_kwargs or {})
    placement_conflicts = sorted(
        key for key in ("device", "devices", "gpus") if key in load_options
    )
    if gpu_ids is not None and placement_conflicts:
        raise ValueError(
            "gpus= controls Show4DSTEM.from_folder CUDA placement; remove "
            f"{placement_conflicts!r} from load_kwargs so each worker decodes "
            "on its capacity-planned GPU."
        )
    if backend is None and gpu_ids is not None:
        backend = "cuda"
    if backend is not None:
        load_options.setdefault("backend", backend)
    load_dtype = "u16" if _dtype_token(dtype) == "auto" else dtype
    from quantem.widget.show4dstem_preview_cache import Show4DSTEMPreviewCache

    loaded_source_signatures: dict[int, dict[str, Any]] = {}
    preview_cache_enabled = bool(
        preview_cache is not False
        and str(preview_cache).strip().lower() not in {"false", "off", "none", "0"}
        and preview_cache_max_bytes != 0
    )

    # Lazy loaders are created before Dataset5dstem configures paging, so keep a
    # late-bound series reference. Once page("auto") selects capacity-aware
    # targets, cold reloads follow that authoritative placement instead of
    # recomputing a fixed idx % n_gpus assignment in the factory.
    series_ref: dict[str, Any] = {}

    def target_gpu(idx: int) -> int | None:
        if gpu_ids is None:
            return None
        series = series_ref.get("series")
        page_devices = list(getattr(series, "_page_devices", []) or [])
        if idx < len(page_devices):
            device = torch.device(page_devices[idx])
            if device.type == "cuda":
                return int(0 if device.index is None else device.index)
        return int(gpu_ids[idx % len(gpu_ids)])

    def load_master(master, idx: int) -> torch.Tensor:
        signature_before = Show4DSTEMPreviewCache.source_signature(master)
        gpu = target_gpu(idx)
        single_options = dict(load_options)
        if gpu is not None:
            single_options.setdefault("device", gpu)
        result = gpu_io.load(
            master,
            det_bin=det_bin,
            dtype=load_dtype,
            verbose=False,
            **single_options,
        )
        data = result.data
        tensor = data if isinstance(data, torch.Tensor) else torch.from_dlpack(data)
        if gpu is not None:
            tensor = tensor.to(f"cuda:{gpu}")
        signature_after = Show4DSTEMPreviewCache.source_signature(master)
        if signature_after != signature_before:
            raise RuntimeError(
                f"{master!s} changed while Show4DSTEM was loading it. "
                "Wait for the master and linked chunks to become stable, then retry."
            )
        loaded_source_signatures[int(idx)] = signature_after
        return tensor

    def load_independent_batch(items) -> dict[int, torch.Tensor]:
        """Load independent frames concurrently across, but not within, GPUs.

        One worker owns each selected CUDA device and decodes that device's
        files serially. Different cards can make progress together, while every
        returned frame owns an independent allocation that LRU eviction can
        actually reclaim. This also keeps the capacity planner's per-index
        target authoritative for folders spread across multiple disks.
        """
        from concurrent.futures import ThreadPoolExecutor

        grouped: dict[int, list[tuple[int, Any]]] = {}
        for idx, master in items:
            gpu = target_gpu(int(idx))
            if gpu is None:
                return {}
            grouped.setdefault(gpu, []).append((int(idx), master))

        def worker(group: list[tuple[int, Any]]) -> dict[int, torch.Tensor]:
            return {idx: load_master(master, idx) for idx, master in group}

        if len(grouped) == 1:
            return worker(next(iter(grouped.values())))
        loaded: dict[int, torch.Tensor] = {}
        with ThreadPoolExecutor(max_workers=max(1, len(grouped))) as pool:
            for result in pool.map(worker, grouped.values()):
                loaded.update(result)
        return loaded

    def _initial_page_target() -> int:
        if not preload_initial_page or not _multiple_view_requested(view_mode):
            return 1
        # A durable preview cache can paint every visible BF/DF tile after the
        # unavoidable first-master calibration load.  Loading a larger raw page
        # before the cache is consulted defeats fast repeat-open behavior.
        if preview_cache_enabled:
            return 1
        # One unbinned 512x512x192x192 uint16 master is about 18 GiB. Load the
        # first master safely, let Dataset5dstem establish per-GPU byte budgets,
        # then materialize the visible page through memory-aware sub-batches.
        if int(det_bin) == 1:
            return 1
        if isinstance(preload_initial_page, int) and not isinstance(
            preload_initial_page, bool
        ):
            target = int(preload_initial_page)
        elif isinstance(page_budget, int) and not isinstance(page_budget, bool):
            target = int(page_budget)
        elif int(compare_cols or 0) > 0:
            target = int(compare_cols)
        else:
            target = 4
        return max(1, min(len(masters), int(compare_max_panels), target))

    def _try_load_initial_page() -> dict[int, torch.Tensor]:
        target = _initial_page_target()
        if target <= 1 or gpu_ids is None:
            return {}
        backend_name = str(load_options.get("backend", "")).lower()
        if backend_name not in {"cuda", "cupy"}:
            return {}
        load_failed = False
        try:
            initial = load_independent_batch(
                [(idx, master) for idx, master in enumerate(masters[:target])]
            )
        except BaseException as exc:
            if _is_recoverable_allocation_error(exc):
                load_failed = True
            else:
                return {}
        if load_failed:
            # Reclaim after leaving the except block so the exception traceback
            # no longer holds partial CuPy arrays from the failed batch.
            _reclaim_failed_load_memory(torch, gpu_ids)
            return {}
        if 0 not in initial:
            return {}
        return initial

    initial_frames = _try_load_initial_page()
    first_tensor = None
    first_master = None
    first_original_idx = 0
    if initial_frames:
        first_tensor = initial_frames[0]
        first_master = masters[0]
    else:
        for original_idx, master in enumerate(masters):
            try:
                first_tensor = load_master(master, original_idx)
            except BaseException as exc:
                if _is_recoverable_allocation_error(exc):
                    _reclaim_failed_load_memory(torch, gpu_ids)
                    memory_label = _cuda_memory_label(torch, gpu_ids)
                    warning = _folder_memory_warning(
                        folder_path=folder_path,
                        memory_label=memory_label,
                    )
                    return _memory_limited_folder_widget(
                        folder_path=folder_path,
                        memory_label=memory_label,
                        warning=warning,
                    )
                if (
                    isinstance(exc, (FileNotFoundError, ValueError, RuntimeError))
                    and ready_only
                ):
                    continue
                raise
            first_master = master
            first_original_idx = original_idx
            initial_frames = {0: first_tensor}
            break
    if first_tensor is None or first_master is None:
        raise ValueError(f"No readable {pattern!r} files found in {folder_path}.")

    kept_masters = (
        masters
        if first_original_idx == 0
        else [first_master, *masters[first_original_idx + 1 :]]
    )
    folder_masters = list(kept_masters)
    validate_folder_contracts = True
    try:
        initial_contract = _master_file_contract(first_master)
    except Exception:
        validate_folder_contracts = False
        initial_contract = {
            "scan_shape": tuple(int(value) for value in first_tensor.shape[:2]),
            "detector_shape": tuple(
                int(value) * int(det_bin) for value in first_tensor.shape[-2:]
            ),
            "n_frames": int(math.prod(first_tensor.shape[:2])),
            "dtype": None,
        }
    expected_post_shape = tuple(int(value) for value in first_tensor.shape)
    expected_scan_shape = expected_post_shape[:2]
    initial_detector_shape = initial_contract.get("detector_shape") or tuple(
        int(value) * int(det_bin) for value in first_tensor.shape[-2:]
    )
    expected_detector_shape = tuple(int(value) for value in initial_detector_shape)
    expected_source_dtype = initial_contract.get("dtype")

    def validate_master(master) -> None:
        if not validate_folder_contracts:
            return
        contract = _master_file_contract(master)
        candidate_scan = scan_shape or contract.get("scan_shape")
        detector_shape = contract.get("detector_shape")
        n_frames = contract.get("n_frames")
        if candidate_scan is None or detector_shape is None or n_frames is None:
            raise ValueError(
                f"{master!s} does not yet expose complete scan/detector metadata."
            )
        candidate_scan = tuple(int(value) for value in candidate_scan)
        detector_shape = tuple(int(value) for value in detector_shape)
        if int(math.prod(candidate_scan)) != int(n_frames):
            raise ValueError(
                f"{master!s} has {n_frames} frames, incompatible with "
                f"scan_shape={candidate_scan}."
            )
        if candidate_scan != expected_scan_shape:
            raise ValueError(
                f"{master!s} has scan_shape={candidate_scan}; expected "
                f"{expected_scan_shape}."
            )
        if detector_shape != expected_detector_shape:
            raise ValueError(
                f"{master!s} has detector_shape={detector_shape}; expected "
                f"{expected_detector_shape}."
            )
        if any(value % int(det_bin) for value in detector_shape):
            raise ValueError(
                f"{master!s} detector_shape={detector_shape} is not divisible by "
                f"det_bin={det_bin}."
            )
        post_shape = (
            *candidate_scan,
            detector_shape[0] // int(det_bin),
            detector_shape[1] // int(det_bin),
        )
        if post_shape != expected_post_shape:
            raise ValueError(
                f"{master!s} would load as shape={post_shape}; expected "
                f"{expected_post_shape}."
            )
        if (
            expected_source_dtype is not None
            and contract.get("dtype") != expected_source_dtype
        ):
            raise ValueError(
                f"{master!s} has dtype={contract.get('dtype')}; expected "
                f"{expected_source_dtype}."
            )

    def register_master(master, idx: int) -> None:
        if int(idx) != len(folder_masters):
            raise RuntimeError(
                "Watched Show4DSTEM master registration lost index alignment."
            )
        folder_masters.append(master)

    def load_master_batch(indices) -> dict[int, torch.Tensor]:
        requested = [int(idx) for idx in indices]
        selected_masters = [folder_masters[idx] for idx in requested]
        if len(requested) <= 1 or gpu_ids is None:
            return {}
        backend_name = str(load_options.get("backend", "")).lower()
        if backend_name not in {"cuda", "cupy"}:
            return {}
        return load_independent_batch(zip(requested, selected_masters, strict=True))

    def make_loader(master, idx: int):
        label = _master_label(master)

        def _loader(path=master, load_idx=idx):
            return load_master(path, load_idx)

        return label, _loader

    names: list[str] = []
    loaders = []
    for idx, master in enumerate(kept_masters):
        label, loader = make_loader(master, idx)
        names.append(label)
        loaders.append(loader)

    series = Dataset5dstem.from_lazy_loaders(
        loaders,
        shape=(len(loaders), *tuple(first_tensor.shape)),
        dtype=first_tensor.dtype,
        initial_frames=initial_frames,
        # Each device worker decodes serially into independent allocations, so
        # exact no-bin masters can use the same multi-GPU progressive waves
        # without a shared page-sized CuPy stack blocking LRU reclamation.
        batch_loader=load_master_batch,
        name=f"Show4DSTEM folder: {folder_path.name}",
    )
    def cache_contract_value(value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (list, tuple)):
            return [cache_contract_value(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): cache_contract_value(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        return repr(value)

    value_options = {
        str(key): cache_contract_value(value)
        for key, value in sorted(load_options.items())
        if str(key) not in {"backend", "device", "verbose"}
    }
    preview_store = Show4DSTEMPreviewCache(
        folder_path,
        cache=preview_cache if preview_cache_enabled else False,
        cache_dir=preview_cache_dir,
        max_bytes=preview_cache_max_bytes,
        rebuild=rebuild_preview_cache and preview_cache_enabled,
        load_contract={
            "det_bin": int(det_bin),
            "dtype": str(load_dtype),
            "scan_shape": None
            if scan_shape is None
            else [int(value) for value in scan_shape],
            "options": value_options,
        },
    )
    # Internal handoff lets cache-enabled construction consult disk previews
    # before _attach_folder_source(), and lets the host page cache retain source
    # provenance even when persistent writes are explicitly disabled.
    series._compare_preview_cache = preview_store
    series._compare_preview_sources = folder_masters
    series._compare_loaded_source_signatures = loaded_source_signatures
    series_ref["series"] = series
    viewer_kwargs.setdefault("frame_dim_label", "Dataset")
    viewer_kwargs.setdefault("frame_labels", names)
    viewer_kwargs.setdefault("view_mode", view_mode)
    viewer_kwargs.setdefault("compare_cols", compare_cols)
    viewer_kwargs.setdefault("compare_max_panels", compare_max_panels)
    viewer_kwargs.setdefault("verbose", verbose)
    page_devices = gpu_ids
    if page_devices is None and first_tensor.device.type == "cuda":
        page_devices = [first_tensor.device]
    widget = _Show4DSTEMBase(
        series,
        page_budget=page_budget,
        page_device=page_devices,
        page_max_vram_fraction=page_max_vram_fraction,
        page_reserve_vram_bytes=page_reserve_vram_bytes,
        page_max_vram_bytes=page_max_vram_bytes,
        **viewer_kwargs,
    )
    widget._attach_folder_source(
        folder=folder_path,
        pattern=pattern,
        recursive=recursive,
        scan_shape=scan_shape,
        ready_only=ready_only,
        known_masters=kept_masters,
        make_loader=make_loader,
        validate_master=validate_master,
        register_master=register_master,
        preload_all_if_fits=preload_all_if_fits,
        warm_cache=warm_cache,
    )
    if watch:
        widget.watch_folder(interval=watch_interval)
    initial_page_thread = getattr(widget, "_compare_page_thread", None)
    if initial_page_thread is None or not initial_page_thread.is_alive():
        # If the constructor's cache/progressive page worker is active, its
        # _resume_folder_compare_maintenance() path starts these sequentially
        # after visible work. Launching here would race the same CUDA contexts.
        if preload_all_if_fits:
            widget.preload_all_datasets(background=not warm_cache)
        if warm_cache:
            widget.warm_compare_cache(background=True)
    return widget


Show4DSTEM.from_folder = from_folder  # type: ignore[attr-defined]


__all__ = [
    "Show4DSTEM",
    "from_folder",
    "is_mps_show4dstem_payload",
    "show4dstem_backend_kind",
]
