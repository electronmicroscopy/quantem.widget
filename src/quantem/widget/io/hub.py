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

import json
import os
import tempfile
import time
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


def _resolve_repo(repo: str | None) -> str:
    """Pick the shared dataset repo: explicit arg, env, source, then default."""

    return (
        repo
        or os.environ.get("QUANTEM_DATA_REPO")
        or str(getattr(_src, "REPO_ID", None) or DEFAULT_REPO)
    )


def _hub():
    """Import ``huggingface_hub`` lazily with a clear install hint."""

    try:
        return import_module("huggingface_hub")
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to share datasets. "
            "Install it with `pip install huggingface_hub`."
        ) from exc


def _derive_4dstem_shapes(folder: Path) -> dict:
    """Read detector and square scan shape from an Arina master, best-effort."""

    try:
        import math  # noqa: PLC0415

        import h5py  # noqa: PLC0415

        masters = sorted(folder.glob("*_master.h5"))
        if not masters:
            return {}
        with h5py.File(masters[0], "r") as f:
            spec = f["entry/instrument/detector/detectorSpecific"]
            out: dict = {
                "det_shape": [
                    int(spec["y_pixels_in_detector"][()]),
                    int(spec["x_pixels_in_detector"][()]),
                ]
            }
            data = f.get("entry/data/data")
            if data is not None and data.ndim >= 1:
                n_frames = int(data.shape[0])
                side = math.isqrt(n_frames)
                if side * side == n_frames:
                    out["scan_shape"] = [side, side]
            return out
    except (OSError, KeyError, ValueError, ImportError):
        return {}


def _build_meta(src: Path, meta: dict | None) -> dict:
    """Merge auto-derived folder metadata under explicit operator metadata."""

    out: dict = {}
    if src.is_dir():
        out.update(_derive_4dstem_shapes(src))
    if meta:
        out.update({k: v for k, v in meta.items() if v is not None})
    return out


def _upload_meta(
    hub,
    repo_id: str,
    folder: str,
    name: str,
    sidecar: dict,
    *,
    is_dir: bool,
) -> None:
    """Write the calibration sidecar next to the dataset."""

    path_in_repo = (
        f"{folder}/{name}/quantem_meta.json" if is_dir else f"{folder}/{name}.json"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(sidecar, fh, indent=2)
        tmp = fh.name
    try:
        hub.upload_file(
            path_or_fileobj=tmp,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
        )
    finally:
        os.unlink(tmp)


def _upload_path_dataset(
    path: str | Path,
    name: str | None = None,
    *,
    folder: str | None = None,
    repo: str | None = None,
    meta: dict | None = None,
) -> str:
    """Upload a file/folder dataset using the legacy live-data convention."""

    hub = _hub()
    src = Path(path)
    repo_id = _resolve_repo(repo)
    if name is None:
        name = src.stem if src.is_file() else src.name
    if folder is None:
        folder = "haadf" if src.is_file() else "4dstem"
    if src.is_dir():
        info = hub.upload_folder(
            folder_path=str(src),
            path_in_repo=f"{folder}/{name}",
            repo_id=repo_id,
            repo_type="dataset",
        )
    else:
        suffix = "".join(src.suffixes)
        info = hub.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=f"{folder}/{name}{suffix}",
            repo_id=repo_id,
            repo_type="dataset",
        )
    sidecar = _build_meta(src, meta)
    if sidecar:
        _upload_meta(hub, repo_id, folder, name, sidecar, is_dir=src.is_dir())
    return getattr(info, "commit_url", info)


def _list_hf_datasets(*, repo: str | None = None) -> list[str]:
    """List folder/file datasets directly from the shared Hugging Face repo."""

    hub = _hub()
    repo_id = _resolve_repo(repo)
    names = set()
    for f in hub.list_repo_files(repo_id=repo_id, repo_type="dataset"):
        parts = f.split("/")
        if len(parts) < 2 or parts[1].startswith("placeholder_"):
            continue
        if len(parts) >= 3:
            names.add(f"{parts[0]}/{parts[1]}")
        elif not parts[1].endswith(".json"):
            names.add(f"{parts[0]}/{Path(parts[1]).stem}")
    return sorted(names)


def _download_hf_dataset(
    name: str,
    *,
    repo: str | None = None,
    out: str | Path | None = None,
    verbose: bool = True,
) -> Path:
    """Download one shared folder/file dataset by flat name."""

    hub = _hub()
    repo_id = _resolve_repo(repo)
    files = hub.list_repo_files(repo_id=repo_id, repo_type="dataset")
    candidates: dict[str, str] = {}
    for f in files:
        parts = f.split("/")
        if len(parts) >= 3 and parts[1] == name:
            candidates[f"{parts[0]}/{name}"] = "dir"
        elif len(parts) == 2 and Path(parts[1]).stem == name and not f.endswith(".json"):
            candidates[f] = "file"
    if not candidates:
        raise FileNotFoundError(
            f"{name!r} not found in {repo_id}. Run `quantem data list` "
            "or `live data list`."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"{name!r} is ambiguous in {repo_id}: {sorted(candidates)}. "
            "Rename one, or set repo= to a repo where it is unique."
        )
    target_rel, kind = next(iter(candidates.items()))
    pattern = f"{target_rel}/*" if kind == "dir" else target_rel
    if verbose:
        print(
            f"Downloading '{name}' from Hugging Face ({repo_id}) over the internet - "
            "speed depends on your connection, not your computer ...",
            flush=True,
        )
    t0 = time.perf_counter()
    root = hub.snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=pattern,
        local_dir=str(out) if out is not None else None,
    )
    result = Path(root) / target_rel
    if verbose:
        dt = time.perf_counter() - t0
        gb = (
            sum(f.stat().st_size for f in result.rglob("*") if f.is_file())
            if result.is_dir()
            else result.stat().st_size
        ) / 1e9
        if dt < 1.0:
            print(
                f"'{name}' ({gb:.2f} GB) is already cached on disk - no re-download.\n"
                f"  cached at: {result}",
                flush=True,
            )
        else:
            print(
                f"Downloaded '{name}' ({gb:.2f} GB) in {dt:.0f}s "
                f"({gb * 1000 / dt:.0f} MB/s from Hugging Face).\n"
                "  cached on disk - future loads are instant, no re-download.\n"
                f"  cached at: {result}",
                flush=True,
            )
    return result


def _read_hf_meta(name: str, *, repo: str | None = None) -> dict | None:
    """Return a folder/file dataset sidecar, or ``None`` if absent."""

    hub = _hub()
    repo_id = _resolve_repo(repo)
    target = None
    for f in hub.list_repo_files(repo_id=repo_id, repo_type="dataset"):
        parts = f.split("/")
        if len(parts) == 3 and parts[1] == name and parts[2] == "quantem_meta.json":
            target = f
            break
        if len(parts) == 2 and f.endswith(".json") and Path(parts[1]).stem == name:
            target = f
            break
    if target is None:
        return None
    local = hub.hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=target)
    return json.loads(Path(local).read_text())


def _delete_hf_dataset(name: str, *, repo: str | None = None) -> list[str]:
    """Delete a shared folder/file dataset by flat name."""

    hub = _hub()
    repo_id = _resolve_repo(repo)
    dir_locs: set[str] = set()
    file_groups: dict[str, list[str]] = {}
    for f in hub.list_repo_files(repo_id=repo_id, repo_type="dataset"):
        parts = f.split("/")
        if len(parts) >= 3 and parts[1] == name:
            dir_locs.add(f"{parts[0]}/{name}")
        elif len(parts) == 2 and Path(parts[1]).stem == name:
            file_groups.setdefault(parts[0], []).append(f)
    locations = list(dir_locs) + [f"{bucket}/{name}" for bucket in file_groups]
    if not locations:
        raise FileNotFoundError(
            f"{name!r} not found in {repo_id}. Run `quantem data list` "
            "or `live data list`."
        )
    if len(locations) > 1:
        raise ValueError(
            f"{name!r} is ambiguous in {repo_id}: {sorted(locations)}. "
            "Delete one explicitly."
        )
    deleted = []
    if dir_locs:
        loc = next(iter(dir_locs))
        hub.delete_folder(path_in_repo=loc, repo_id=repo_id, repo_type="dataset")
        deleted.append(f"{loc}/")
    else:
        for f in next(iter(file_groups.values())):
            hub.delete_file(path_in_repo=f, repo_id=repo_id, repo_type="dataset")
            deleted.append(f)
    return deleted


def _status_hf(*, repo: str | None = None) -> dict:
    """Snapshot repo auth, shared folder/file datasets, and local cache size."""

    hub = _hub()
    repo_id = _resolve_repo(repo)
    api = hub.HfApi()
    token = hub.get_token()
    user = None
    if token:
        try:
            user = api.whoami(token=token).get("name")
        except hub.errors.HfHubHTTPError:
            user = None
    sizes: dict[str, int] = {}
    counts: dict[str, int] = {}
    for entry in api.list_repo_tree(repo_id, repo_type="dataset", recursive=True):
        size = getattr(entry, "size", None)
        if size is None:
            continue
        parts = entry.path.split("/")
        if len(parts) < 2 or parts[1].startswith("placeholder_"):
            continue
        if len(parts) >= 3:
            key = f"{parts[0]}/{parts[1]}"
        elif entry.path.endswith(".json"):
            continue
        else:
            key = f"{parts[0]}/{Path(parts[1]).stem}"
        sizes[key] = sizes.get(key, 0) + size
        counts[key] = counts.get(key, 0) + 1
    datasets = [
        {"name": key, "files": counts[key], "size_mb": sizes[key] / 1e6}
        for key in sorted(sizes)
    ]
    cache_dir = Path(hub.constants.HF_HUB_CACHE) / f"datasets--{repo_id.replace('/', '--')}"
    cached_mb = 0.0
    if cache_dir.exists():
        cached_mb = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()) / 1e6
    return {
        "repo": repo_id,
        "logged_in_as": user,
        "datasets": datasets,
        "total_mb": sum(sizes.values()) / 1e6,
        "cache_dir": str(cache_dir),
        "cached_mb": cached_mb,
    }


def _status_from_file_entries(files: list[dict[str, Any]]) -> dict[str, Any]:
    """Normalize ``quantem.data.list_files()`` output to the live-data shape."""

    sizes: dict[str, float] = {}
    counts: dict[str, int] = {}
    total_mb = 0.0
    for item in files:
        path = str(item.get("path", ""))
        size_mb = float(item.get("size_mb", 0.0))
        total_mb += size_mb
        parts = path.split("/")
        if len(parts) < 2 or parts[1].startswith("placeholder_"):
            continue
        if len(parts) >= 3:
            key = f"{parts[0]}/{parts[1]}"
        elif path.endswith(".json"):
            continue
        else:
            key = f"{parts[0]}/{Path(parts[1]).stem}"
        sizes[key] = sizes.get(key, 0.0) + size_mb
        counts[key] = counts.get(key, 0) + 1
    datasets = [
        {"name": key, "files": counts[key], "size_mb": sizes[key]}
        for key in sorted(sizes)
    ]
    return {
        "repo": _repo_id(),
        "logged_in_as": None,
        "datasets": datasets,
        "total_mb": total_mb,
        "cache_dir": "",
        "cached_mb": 0.0,
        "files": files,
    }


def list_datasets(*args: Any, **kwargs: Any) -> list[str]:
    """List datasets through the installed ``quantem.data`` implementation."""

    if hasattr(_src, "list_datasets"):
        return _call("list_datasets", *args, **kwargs)
    repo = kwargs.pop("repo", None)
    if repo is None and not args and not kwargs:
        try:
            return _list_hf_datasets()
        except Exception:
            pass
    elif repo is not None and not args and not kwargs:
        return _list_hf_datasets(repo=repo)
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
    if not args:
        try:
            return _download_hf_dataset(name, **kwargs)
        except (AttributeError, ImportError, FileNotFoundError):
            pass
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
        if args:
            raise TypeError("registry-style quantem.data.info accepts only the dataset name")
        try:
            if kwargs:
                raise TypeError("registry-style quantem.data.info accepts only the dataset name")
            return _call("info", name)
        except (AttributeError, FileNotFoundError, KeyError, TypeError):
            pass
    if not args:
        return _read_hf_meta(name, **kwargs)
    raise AttributeError(f"{_SOURCE_NAME} does not provide read_meta/info")


def status(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Return a lightweight repository status snapshot."""

    if hasattr(_src, "status"):
        return _call("status", *args, **kwargs)
    if hasattr(_src, "list_files"):
        if args:
            raise TypeError("registry-style quantem.data.list_files accepts keyword arguments only")
        if "repo" not in kwargs:
            files = _call("list_files", **kwargs)
            return _status_from_file_entries(files)
    if not args:
        return _status_hf(**kwargs)
    raise AttributeError(f"{_SOURCE_NAME} does not provide status/list_files")


def upload(path_or_data: Any, name: str | None = None, *args: Any, **kwargs: Any) -> Any:
    """Upload either a widget array dataset or a file/folder raw dataset.

    ``quantem.data`` owns array uploads using ``technique=...``. The legacy
    live-data convention uploads raw files/folders under ``haadf/`` or
    ``4dstem/`` using ``folder=...`` and optional ``meta=...`` sidecars.
    """

    array_upload_keys = {
        "technique",
        "metadata",
        "description",
        "contributor",
        "license",
        "token",
        "create_pr",
    }
    if args or array_upload_keys.intersection(kwargs):
        if hasattr(_src, "upload"):
            if name is None:
                return _call("upload", path_or_data, *args, **kwargs)
            return _call("upload", path_or_data, name=name, *args, **kwargs)
        raise AttributeError(f"{_SOURCE_NAME} does not provide upload")
    repo = kwargs.pop("repo", None)
    folder = kwargs.pop("folder", None)
    meta = kwargs.pop("meta", None)
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(f"unsupported upload option(s): {names}")
    return _upload_path_dataset(
        path_or_data,
        name=name,
        folder=folder,
        repo=repo,
        meta=meta,
    )


def delete(name: str, *args: Any, **kwargs: Any) -> list[str]:
    """Delete a shared folder/file dataset by flat name."""

    if hasattr(_src, "delete"):
        return _call("delete", name, *args, **kwargs)
    if args:
        raise TypeError("delete accepts only a dataset name and keyword options")
    return _delete_hf_dataset(name, **kwargs)


def __getattr__(name: str) -> Any:
    return getattr(_src, name)


_source_all = list(getattr(_src, "__all__", []))
__all__ = sorted(set(_source_all + [
    "DEFAULT_REPO",
    "delete",
    "download",
    "list_datasets",
    "read_meta",
    "status",
    "upload",
]))


del _SKIP_GLOBALS
