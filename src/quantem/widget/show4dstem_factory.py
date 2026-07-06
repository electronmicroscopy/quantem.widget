"""Public Show4DSTEM factory.

``quantem.widget.Show4DSTEM`` is intentionally one user-facing API. This module
keeps the backend dispatch out of ``quantem.widget.__init__`` so the package
initializer stays a public export list instead of carrying backend policy.

The raw-Metal/MPS implementation still lives in ``show4dstem_mps`` for
compatibility and because it has backend-specific trait/status wiring. Users
should normally reach it through ``Show4DSTEM(load(..., backend="mps"))``.
"""
from __future__ import annotations

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


__all__ = [
    "Show4DSTEM",
    "is_mps_show4dstem_payload",
    "show4dstem_backend_kind",
]
