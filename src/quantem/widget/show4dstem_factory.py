"""Public Show4DSTEM factory.

``quantem.widget.Show4DSTEM`` is intentionally one user-facing API. This module
keeps the backend dispatch out of ``quantem.widget.__init__`` so the package
initializer stays a public export list instead of carrying backend policy.

The raw-Metal/MPS implementation still lives in ``show4dstem_mps`` for
compatibility and because it has backend-specific trait/status wiring. Users
should normally reach it through ``Show4DSTEM(load(..., backend="mps"))``.
"""
from __future__ import annotations

import os
import pathlib
from typing import Any

from quantem.widget.show4dstem import Show4DSTEM as _Show4DSTEMBase


def _is_load_result(value: Any) -> bool:
    """Return True for the ``io.load`` namedtuple shape without importing IO."""
    return hasattr(value, "_fields") and "data" in getattr(value, "_fields", ())


def _payload(value: Any) -> Any:
    """Unwrap ``io.load`` output to the object used for backend routing."""
    return value.data if _is_load_result(value) else value


def is_mps_show4dstem_payload(value: Any) -> bool:
    """Return True when ``value`` should use the raw-Metal Show4DSTEM path.

    MPS HDF5 loads expose chunked buffers. Already-wrapped Metal frame proxies
    expose ``_is_gpu_frames``; those should only be treated as MPS when their
    device string says MPS. CUDA/CPU 5D dataset wrappers are left for the base
    viewer.
    """
    payload = _payload(value)
    payload_device = str(getattr(payload, "device", ""))
    is_mps_frames = (
        bool(getattr(payload, "_is_gpu_frames", False))
        and "mps" in payload_device
    )
    return hasattr(payload, "chunks") or is_mps_frames


def show4dstem_backend_kind(value: Any) -> str:
    """Classify the backend family the public factory will select."""
    from quantem.widget.multidataset_mps import LazyMacbookDatasets

    payload = _payload(value)
    if isinstance(value, LazyMacbookDatasets) or isinstance(payload, LazyMacbookDatasets):
        return "mps"
    if is_mps_show4dstem_payload(payload):
        return "mps"
    return "base"


def _build_mps_viewer(data: Any, **kwargs: Any) -> Any:
    """Build the sampling-aware MPS viewer lazily to keep normal imports light."""
    from quantem.widget.show4dstem_mps import Show4DSTEM_MACBOOK

    return Show4DSTEM_MACBOOK(data, **kwargs)


def _apply_loadresult_labels(data: Any, payload: Any, kwargs: dict[str, Any]) -> None:
    """Label a CUDA/CPU 5D ``load([...])`` stack as datasets."""
    if not _is_load_result(data) or getattr(payload, "ndim", 0) != 5:
        return
    meta = getattr(data, "metadata", {}) or {}
    kwargs.setdefault("frame_dim_label", "Dataset")
    names = meta.get("file_names")
    if names is not None:
        kwargs.setdefault("frame_labels", list(names))


def Show4DSTEM(data: Any, **kwargs: Any) -> Any:
    """Open a 4D-STEM viewer over ``load(...)`` output, on any backend.

    Canonical examples::

        from quantem.widget import load, Show4DSTEM

        Show4DSTEM(load("a.h5"))                         # auto: CUDA / MPS / CPU
        Show4DSTEM(load("a.h5", backend="mps"))          # explicit Apple Metal load
        Show4DSTEM(load(["a.h5", "b.h5"], det_bin=4))    # many datasets, one slider
        Show4DSTEM(load("a.h5"), backend="web")          # browser WebGPU compute

        w = Show4DSTEM(load("a.h5"), backend="web", offline_codec="bslz4",
                       data_url="show4dstem-data")
        w.export_html("show4dstem.html")

    Dispatch is automatic from what ``load`` returns:
      - Apple Silicon MPS single-file loads use the raw-Metal real-time viewer.
      - Apple Silicon MPS multi-file loads use a lazy handle; dataset 0 shows
        immediately and later datasets fill behind the dataset slider.
      - CUDA / CPU single or multi-file loads use the universal torch viewer.

    Web aliases ``backend="browser"``, ``backend="webgpu"``, and
    ``offline=True`` are accepted by the base viewer for compatibility.
    """
    from quantem.widget.multidataset_mps import LazyMacbookDatasets

    payload = _payload(data)
    if isinstance(data, LazyMacbookDatasets):
        return data.build_viewer(**kwargs)
    if isinstance(payload, LazyMacbookDatasets):
        return payload.build_viewer(**kwargs)
    if is_mps_show4dstem_payload(payload):
        return _build_mps_viewer(payload, **kwargs)

    _apply_loadresult_labels(data, payload, kwargs)
    return _Show4DSTEMBase(data, **kwargs)


def _normalise_gpus(gpus) -> list[int] | None:
    if gpus is None:
        return None
    if isinstance(gpus, int):
        return [int(gpus)]
    out = [int(gpu) for gpu in gpus]
    if not out:
        raise ValueError("gpus must be None, an int, or a non-empty sequence of GPU ids.")
    return out


def _master_label(master) -> str:
    name = os.path.basename(str(master))
    return name[:-len("_master.h5")] if name.endswith("_master.h5") else name


def from_folder(
    folder,
    *,
    pattern: str = "*_master.h5",
    recursive: bool = True,
    scan_size: int | None = None,
    ready_only: bool = True,
    gpus=None,
    page_budget: int | str | None = "auto",
    det_bin: int = 4,
    dtype: str = "u8",
    backend: str | None = None,
    load_kwargs: dict[str, Any] | None = None,
    watch: bool = False,
    watch_interval: float = 2.0,
    page_max_vram_fraction: float = 0.75,
    page_reserve_vram_bytes: int | None = None,
    page_max_vram_bytes: int | dict | None = None,
    view_mode: str = "multiple",
    compare_cols: int = 3,
    compare_max_panels: int = 12,
    verbose: bool = True,
    **viewer_kwargs: Any,
) -> Any:
    """Open a lazy, folder-backed Show4DSTEM over ready ``*_master.h5`` files.

    This is the direct many-master browse API. It differs from ``load(folder)``
    by loading only the first ready master for shape and first paint; remaining
    masters are lazy slots and allocate GPU memory only when selected or shown in
    the visible multiple grid. ``poll_folder()`` / ``watch_folder()`` append new
    ready masters as lazy slots without rebuilding the widget.
    """
    import torch
    from quantem.widget.data import Dataset5dstem
    from quantem.widget.io import discover_masters, is_master_ready
    from quantem.widget import load

    if backend == "mps":
        from quantem.widget.multidataset_mps import load_macbook_datasets

        live = load_macbook_datasets(
            folder,
            det_bin=det_bin,
            scan_size=scan_size,
            verbose=verbose,
            skip_mps_memory_check=(load_kwargs or {}).get("skip_mps_memory_check"),
        )
        viewer = Show4DSTEM(
            live,
            view_mode=view_mode,
            compare_cols=compare_cols,
            compare_max_panels=compare_max_panels,
            verbose=verbose,
            **viewer_kwargs,
        )
        if watch:
            live.watch_master_folder(folder, interval=watch_interval, scan_size=scan_size)
        return viewer

    gpu_ids = _normalise_gpus(gpus)
    folder_path = pathlib.Path(folder).expanduser().resolve()
    scan_shape = (int(scan_size), int(scan_size)) if scan_size else None
    masters = discover_masters(
        str(folder_path),
        pattern=pattern,
        recursive=recursive,
        scan_shape=scan_shape,
        verbose=False,
    )
    if ready_only:
        masters = [master for master in masters if is_master_ready(master)]
    if not masters:
        raise ValueError(
            f"No ready {pattern!r} files found in {folder_path}. "
            "Wait for linked data files to finish writing, or pass ready_only=False "
            "if you know the masters are complete."
        )

    load_options = dict(load_kwargs or {})
    if backend is not None:
        load_options.setdefault("backend", backend)

    def load_master(master, idx: int) -> torch.Tensor:
        result = load(
            master,
            det_bin=det_bin,
            dtype=dtype,
            verbose=False,
            **load_options,
        )
        data = result.data if hasattr(result, "_fields") and "data" in result._fields else result
        tensor = data if isinstance(data, torch.Tensor) else torch.from_dlpack(data)
        if gpu_ids is not None:
            tensor = tensor.to(f"cuda:{gpu_ids[idx % len(gpu_ids)]}")
        return tensor

    first_tensor = None
    first_master = None
    first_original_idx = 0
    for original_idx, master in enumerate(masters):
        try:
            first_tensor = load_master(master, original_idx)
        except (FileNotFoundError, ValueError, RuntimeError):
            if ready_only:
                continue
            raise
        first_master = master
        first_original_idx = original_idx
        break
    if first_tensor is None or first_master is None:
        raise ValueError(f"No readable {pattern!r} files found in {folder_path}.")

    kept_masters = [first_master, *masters[first_original_idx + 1:]]

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
        initial_frames={0: first_tensor},
        name=f"Show4DSTEM folder: {folder_path.name}",
    )
    viewer_kwargs.setdefault("frame_dim_label", "Dataset")
    viewer_kwargs.setdefault("frame_labels", names)
    viewer_kwargs.setdefault("view_mode", view_mode)
    viewer_kwargs.setdefault("compare_cols", compare_cols)
    viewer_kwargs.setdefault("compare_max_panels", compare_max_panels)
    viewer_kwargs.setdefault("verbose", verbose)
    widget = _Show4DSTEMBase(
        series,
        page_budget=page_budget,
        page_device=gpu_ids,
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
    )
    if watch:
        widget.watch_folder(interval=watch_interval)
    return widget


Show4DSTEM.from_folder = from_folder  # type: ignore[attr-defined]


__all__ = [
    "Show4DSTEM",
    "from_folder",
    "is_mps_show4dstem_payload",
    "show4dstem_backend_kind",
]
