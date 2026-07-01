"""Compatibility adapter for the shared :mod:`quantem.data` dataset API.

Historically, widget notebooks imported dataset helpers from
``quantem.widget.io.hub``. The implementation now belongs to ``quantem.data``,
but local/dev installs may expose it under one of a few API shapes:

- ``quantem.data.hub``: original consolidated module name.
- ``quantem.data.huggingface``: current raw-data sharing API.
- ``quantem.data``: registry-style package facade with ``available`` / ``info``.

This module preserves the old widget-facing import path and maps the common
helpers onto the installed ``quantem.data`` implementation.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any


DEFAULT_REPO = "bobleesj/quantem-data"
_SOURCE_CANDIDATES = ("quantem.data.hub", "quantem.data.huggingface", "quantem.data")
_SKIP_GLOBALS = {
    "__name__",
    "__file__",
    "__doc__",
    "__loader__",
    "__spec__",
    "__package__",
    "__builtins__",
    "__cached__",
}


def _resolve_source():
    errors: list[str] = []
    for module_name in _SOURCE_CANDIDATES:
        try:
            return import_module(module_name)
        except ModuleNotFoundError as exc:
            errors.append(f"{module_name}: {exc}")
    joined = "\n  - ".join(errors)
    raise ImportError(
        "quantem.widget.io.hub needs quantem.data. Install quantem.data or add its "
        f"source tree to PYTHONPATH. Tried:\n  - {joined}"
    )


_src = _resolve_source()
_SOURCE_NAME = _src.__name__
globals().update({k: v for k, v in vars(_src).items() if k not in _SKIP_GLOBALS})


def _call(name: str, *args: Any, **kwargs: Any) -> Any:
    fn = getattr(_src, name, None)
    if fn is None:
        raise AttributeError(f"{_SOURCE_NAME} does not provide {name!r}")
    return fn(*args, **kwargs)


def list_datasets(*args: Any, **kwargs: Any) -> list[str]:
    """List datasets through the installed ``quantem.data`` implementation."""

    if hasattr(_src, "list_datasets"):
        return _call("list_datasets", *args, **kwargs)
    if hasattr(_src, "available"):
        technique = kwargs.pop("technique", None)
        if args:
            technique = args[0]
        return _call("available", technique=technique)
    raise AttributeError(f"{_SOURCE_NAME} does not provide list_datasets/available")


def download(name: str, *args: Any, **kwargs: Any) -> Path | str:
    """Download a dataset and return the local file/folder path when supported."""

    if hasattr(_src, "download"):
        return _call("download", name, *args, **kwargs)
    folder = _find_folder_style_dataset(name)
    if folder is not None:
        return _download_folder_style_dataset(folder, **kwargs)
    if hasattr(_src, "load_raw"):
        if args or kwargs:
            raise TypeError("registry-style quantem.data.load_raw accepts only the dataset name")
        return Path(_call("load_raw", name))
    raise AttributeError(
        f"{_SOURCE_NAME} does not provide download/load_raw; use quantem.data.load(...) "
        "for array datasets."
    )


def _repo_id() -> str:
    """Return the Hugging Face dataset repo used by folder-style datasets."""

    return str(getattr(_src, "REPO_ID", None) or DEFAULT_REPO)


def _find_folder_style_dataset(name: str) -> str | None:
    """Return ``bucket/name`` for folder-style HF datasets, if present."""

    if not hasattr(_src, "list_files"):
        return None
    try:
        files = _call("list_files")
    except Exception:
        return None
    folders: set[str] = set()
    for item in files:
        path = str(item.get("path", ""))
        parts = path.split("/")
        if len(parts) >= 3 and parts[1] == name:
            folders.add(f"{parts[0]}/{name}")
    if not folders:
        return None
    if len(folders) > 1:
        raise ValueError(f"{name!r} is ambiguous in {_repo_id()}: {sorted(folders)}")
    return folders.pop()


def _download_folder_style_dataset(
    folder: str,
    *,
    repo: str | None = None,
    out: str | Path | None = None,
    verbose: bool = True,
    **kwargs: Any,
) -> Path:
    """Download ``bucket/name/*`` from the shared Hugging Face dataset repo."""

    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(f"unsupported download option(s) for folder-style dataset: {names}")
    hub = import_module("huggingface_hub")
    root = hub.snapshot_download(
        repo_id=repo or _repo_id(),
        repo_type="dataset",
        allow_patterns=f"{folder}/*",
        local_dir=str(out) if out is not None else None,
    )
    if verbose:
        print(f"Downloaded '{folder.split('/')[-1]}' to {Path(root) / folder}")
    return Path(root) / folder


def read_meta(name: str, *args: Any, **kwargs: Any) -> dict | None:
    """Return dataset metadata through either ``read_meta`` or ``info``."""

    if hasattr(_src, "read_meta"):
        return _call("read_meta", name, *args, **kwargs)
    if hasattr(_src, "info"):
        if args or kwargs:
            raise TypeError("registry-style quantem.data.info accepts only the dataset name")
        return _call("info", name)
    raise AttributeError(f"{_SOURCE_NAME} does not provide read_meta/info")


def status(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Return a lightweight repository status snapshot."""

    if hasattr(_src, "status"):
        return _call("status", *args, **kwargs)
    if hasattr(_src, "list_files"):
        if args:
            raise TypeError("registry-style quantem.data.list_files accepts keyword arguments only")
        files = _call("list_files", **kwargs)
        total_mb = sum(float(item.get("size_mb", 0.0)) for item in files)
        datasets = [
            item
            for item in files
            if str(item.get("type", "")) == "data" or str(item.get("path", "")).endswith(".npy")
        ]
        return {"datasets": datasets, "total_mb": total_mb, "files": files}
    raise AttributeError(f"{_SOURCE_NAME} does not provide status/list_files")


def __getattr__(name: str) -> Any:
    return getattr(_src, name)


_source_all = list(getattr(_src, "__all__", []))
__all__ = sorted(set(_source_all + ["download", "list_datasets", "read_meta", "status"]))


del _SKIP_GLOBALS
