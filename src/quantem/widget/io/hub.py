"""Share raw 4D-STEM / HAADF datasets through one Hugging Face dataset repo.

Collaborators with no instrument data need a one-line way to pull a reference
acquisition; the data owner needs a one-line way to publish one. Both sides
install ``quantem.live`` once and never touch ``huggingface_hub`` directly.

A single dataset repo (default ``bobleesj/quantem-data``, override with the
``QUANTEM_DATA_REPO`` env var or a ``repo=`` argument) is the storage backend.
Keep it simple: two top-level buckets, ``4dstem/`` for acquisitions and
``haadf/`` for images, each holding one folder/file per dataset (``4dstem/gold_512/``,
``haadf/gold_haadf.tif``). An Arina acquisition keeps its master + ``_data_*.h5``
chunk siblings together inside its folder, so ``download`` returns a directory
``discover_masters`` can read. ``download`` takes a flat name and finds its bucket.

The verbs are plain English for a microscopist audience: ``upload``, ``download``,
``list_datasets``, ``delete``, ``status``. ``huggingface_hub`` is imported lazily
so importing ``quantem.widget.io`` on a CUDA-less laptop stays cheap; it is a regular
install dependency so a single ``pip install`` gives both the loader and this path.
"""
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

DEFAULT_REPO = "bobleesj/quantem-data"


def _resolve_repo(repo: str | None) -> str:
    """Pick the dataset repo: explicit arg, else env, else the project default."""
    return repo or os.environ.get("QUANTEM_DATA_REPO") or DEFAULT_REPO


def _hub():
    """Import huggingface_hub lazily with a clear install hint when missing."""
    # Our datasets are PUBLIC - no token needed. Silence huggingface_hub's
    # "HF_TOKEN secret does not exist" nudge (it fires on every download in Colab
    # and confuses users into thinking auth is required).
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    warnings.filterwarnings("ignore", message=r"(?s).*HF_TOKEN.*")
    try:
        import huggingface_hub  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to share datasets. "
            "Install it with `pip install huggingface_hub`."
        ) from exc
    return huggingface_hub


def upload(path: str | Path, name: str | None = None, *,
           folder: str | None = None, repo: str | None = None,
           meta: dict | None = None) -> str:
    """Upload a file or folder to the shared repo under ``<folder>/<name>``.

    ``folder`` is the top-level bucket; it defaults to ``haadf`` for a single
    file and ``4dstem`` for an acquisition folder, so HAADF images and 4D-STEM
    land in the right place with no thought. A folder uploads its whole contents
    (Arina master + chunk siblings stay together) so ``download`` returns a loadable
    dir. Returns the commit URL. Needs a write token on this machine
    (``hf auth login`` or ``HF_TOKEN``); publishing is an explicit owner action.

    ``meta`` carries the calibration the raw Arina master can NOT store itself
    (scan sampling, FOV, voltage, semiangle): the detector h5 only knows detector
    pixels, so a collaborator who downloads the data would otherwise have no FOV.
    When given, it is merged with auto-derived ``det_shape``/``scan_shape`` (read
    from the master) and written as a ``quantem_meta.json`` sidecar travelling with
    the dataset; ``read_meta`` returns it on the other side.
    """
    hub = _hub()
    src = Path(path)
    repo_id = _resolve_repo(repo)
    if name is None:
        name = src.stem if src.is_file() else src.name
    if folder is None:
        folder = "haadf" if src.is_file() else "4dstem"
    if src.is_dir():
        info = hub.upload_folder(
            folder_path=str(src), path_in_repo=f"{folder}/{name}",
            repo_id=repo_id, repo_type="dataset",
        )
    else:
        suffix = "".join(src.suffixes)  # keep multi-part extensions like .nii.gz
        info = hub.upload_file(
            path_or_fileobj=str(src), path_in_repo=f"{folder}/{name}{suffix}",
            repo_id=repo_id, repo_type="dataset",
        )
    sidecar = _build_meta(src, meta)
    if sidecar:
        _upload_meta(hub, repo_id, folder, name, sidecar, is_dir=src.is_dir())
    return getattr(info, "commit_url", info)  # CommitInfo in modern hub, str in old


def _derive_4dstem_shapes(folder: Path) -> dict:
    """Read det_shape (+ square scan_shape) from an Arina master, best-effort.

    The detector h5 stores its own pixel count even though it knows no scan FOV;
    surfacing det_shape/scan_shape in the sidecar saves the collaborator from
    re-deriving them. Returns ``{}`` if no master or the read fails - never blocks
    the upload over a metadata convenience.
    """
    try:
        import h5py  # noqa: PLC0415
        import math
        masters = sorted(folder.glob("*_master.h5"))
        if not masters:
            return {}
        with h5py.File(masters[0], "r") as f:
            spec = f["entry/instrument/detector/detectorSpecific"]
            out: dict = {"det_shape": [int(spec["y_pixels_in_detector"][()]),
                                       int(spec["x_pixels_in_detector"][()])]}
            data = f.get("entry/data/data")
            if data is not None and data.ndim >= 1:
                n = int(data.shape[0])
                side = math.isqrt(n)
                if side * side == n:  # square scan - the common case
                    out["scan_shape"] = [side, side]
            return out
    except (OSError, KeyError, ValueError, ImportError):
        return {}


def _build_meta(src: Path, meta: dict | None) -> dict:
    """Merge auto-derived shapes (4D-STEM folder) under explicit operator meta."""
    out: dict = {}
    if src.is_dir():
        out.update(_derive_4dstem_shapes(src))
    if meta:
        out.update({k: v for k, v in meta.items() if v is not None})
    return out


def _upload_meta(hub, repo_id: str, folder: str, name: str,
                 sidecar: dict, *, is_dir: bool) -> None:
    """Write the calibration sidecar next to the dataset.

    Folder dataset -> ``<bucket>/<name>/quantem_meta.json`` inside the folder
    (download returns the dir, so it rides along). File dataset -> a sibling
    ``<bucket>/<name>.json`` (the same stem ``delete`` already removes).
    """
    import json  # noqa: PLC0415
    import tempfile
    path_in_repo = (f"{folder}/{name}/quantem_meta.json" if is_dir
                    else f"{folder}/{name}.json")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(sidecar, fh, indent=2)
        tmp = fh.name
    try:
        hub.upload_file(path_or_fileobj=tmp, path_in_repo=path_in_repo,
                        repo_id=repo_id, repo_type="dataset")
    finally:
        os.unlink(tmp)


def read_meta(name: str, *, repo: str | None = None) -> dict | None:
    """Return a dataset's calibration sidecar, or ``None`` if it has none.

    The counterpart to ``upload(..., meta=...)``: a collaborator who downloads a
    4D-STEM acquisition gets back the scan sampling / FOV / voltage / semiangle the
    raw detector master cannot carry. Public repos need no token.
    """
    import json  # noqa: PLC0415
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


def download(name: str, *, repo: str | None = None, out: str | Path | None = None,
             verbose: bool = True) -> Path:
    """Download one shared dataset by flat name and return its local path.

    The collaborator names just the dataset (``"gold_512"``); this searches
    every bucket to find where it lives, so they never need to know it is under
    ``4dstem/`` or ``haadf/``. Returns a directory for a multi-file acquisition
    (ready for ``discover_masters`` / ``load``) or the file path for a single-file
    dataset. Public repos need no token.

    ``verbose`` (default) frames Hugging Face's own per-file byte progress bars
    with a clear "downloading from the internet" header and a size/speed summary,
    so the user can tell the wait is the network, not our code. A second call on
    the same dataset is a local cache hit and prints "(already downloaded)".
    """
    hub = _hub()
    repo_id = _resolve_repo(repo)
    files = hub.list_repo_files(repo_id=repo_id, repo_type="dataset")
    candidates: dict[str, str] = {}  # target_rel -> "dir" | "file"
    for f in files:
        parts = f.split("/")
        if len(parts) >= 3 and parts[1] == name:
            candidates[f"{parts[0]}/{name}"] = "dir"
        elif len(parts) == 2 and Path(parts[1]).stem == name and not f.endswith(".json"):
            candidates[f] = "file"  # .json is a sidecar of the data file, not a rival dataset
    if not candidates:
        raise FileNotFoundError(f"{name!r} not found in {repo_id}. Run `live data list`.")
    if len(candidates) > 1:
        raise ValueError(
            f"{name!r} is ambiguous in {repo_id}: {sorted(candidates)}. "
            "Rename one, or set --repo to a repo where it is unique."
        )
    target_rel, kind = next(iter(candidates.items()))
    pattern = f"{target_rel}/*" if kind == "dir" else target_rel
    if verbose:
        print(f"Downloading '{name}' from Hugging Face ({repo_id}) over the internet - "
              "speed depends on your connection, not your computer ...", flush=True)
    t0 = time.perf_counter()
    root = hub.snapshot_download(
        repo_id=repo_id, repo_type="dataset",
        allow_patterns=pattern,
        local_dir=str(out) if out is not None else None,
    )
    result = Path(root) / target_rel
    if verbose:
        dt = time.perf_counter() - t0
        gb = (sum(f.stat().st_size for f in result.rglob("*") if f.is_file())
              if result.is_dir() else result.stat().st_size) / 1e9
        if dt < 1.0:
            print(f"'{name}' ({gb:.2f} GB) is already cached on disk - no re-download.\n"
                  f"  cached at: {result}", flush=True)
        else:
            print(f"Downloaded '{name}' ({gb:.2f} GB) in {dt:.0f}s "
                  f"({gb * 1000 / dt:.0f} MB/s from Hugging Face).\n"
                  f"  cached on disk - future loads are instant, no re-download.\n"
                  f"  cached at: {result}", flush=True)
    return result


def list_datasets(*, repo: str | None = None) -> list[str]:
    """List shared datasets as ``<bucket>/<name>`` (skips placeholders/docs)."""
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


def delete(name: str, *, repo: str | None = None) -> list[str]:
    """Delete a shared dataset by flat name; returns the repo paths removed.

    A folder dataset removes the whole folder; a file dataset removes the data
    file and its ``.json`` sidecar (same stem). Refuses to act when the name
    matches more than one dataset so a delete never nukes the wrong bucket. The
    CLI adds a re-type-to-confirm prompt; this function deletes immediately.
    """
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
    locations = list(dir_locs) + [f"{b}/{name}" for b in file_groups]
    if not locations:
        raise FileNotFoundError(f"{name!r} not found in {repo_id}. Run `live data list`.")
    if len(locations) > 1:
        raise ValueError(f"{name!r} is ambiguous in {repo_id}: {sorted(locations)}. Delete one explicitly.")
    deleted = []
    if dir_locs:
        loc = next(iter(dir_locs))
        hub.delete_folder(path_in_repo=loc, repo_id=repo_id, repo_type="dataset")
        deleted.append(f"{loc}/")
    else:
        for f in next(iter(file_groups.values())):  # data file + its .json sidecar
            hub.delete_file(path_in_repo=f, repo_id=repo_id, repo_type="dataset")
            deleted.append(f)
    return deleted


def status(*, repo: str | None = None) -> dict:
    """Snapshot of the shared repo: auth, datasets + sizes, and local cache size.

    Answers the operator's "where does my data live, can I upload, what is shared,
    what do I already have locally" in one call. No token needed for the dataset
    listing; auth is reported as whoever is logged in (or ``None`` = download-only).
    """
    hub = _hub()
    repo_id = _resolve_repo(repo)
    api = hub.HfApi()
    token = hub.get_token()
    user = None
    if token:
        try:
            user = api.whoami(token=token).get("name")
        except hub.errors.HfHubHTTPError:
            user = None  # stale/invalid token -> treat as download-only
    sizes: dict[str, int] = {}
    counts: dict[str, int] = {}
    for entry in api.list_repo_tree(repo_id, repo_type="dataset", recursive=True):
        size = getattr(entry, "size", None)
        if size is None:
            continue  # folder entry, not a file
        parts = entry.path.split("/")
        if len(parts) < 2 or parts[1].startswith("placeholder_"):
            continue
        if len(parts) >= 3:
            key = f"{parts[0]}/{parts[1]}"
        elif entry.path.endswith(".json"):
            continue  # top-level sidecar, folded into its data file's dataset
        else:
            key = f"{parts[0]}/{Path(parts[1]).stem}"
        sizes[key] = sizes.get(key, 0) + size
        counts[key] = counts.get(key, 0) + 1
    datasets = [{"name": k, "files": counts[k], "size_mb": sizes[k] / 1e6} for k in sorted(sizes)]
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
