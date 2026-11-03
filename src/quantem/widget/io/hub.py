"""Download one shared dataset from the public Hugging Face hub."""

import inspect
import os
import time
from pathlib import Path, PurePosixPath

DEFAULT_REPO = "bobleesj/quantem-data"
# The buckets, data suffixes and sidecar suffixes of quantem.data's resolver
# (quantem.data.huggingface): for one listing the fallback names the same datasets.
_DATA_BUCKETS = ("4dstem", "haadf", "denoise", "widget")
_DATA_SUFFIXES = {
    ".h5", ".hdf5", ".qem", ".emd", ".npy", ".npz", ".tif", ".tiff",
    ".mrc", ".mrcs", ".dm3", ".dm4", ".raw", ".bin", ".png", ".jpg",
    ".jpeg",
}
_SIDECAR_SUFFIXES = (".yaml", ".yml", ".json")


def _shared_download(name: str, **kwargs) -> Path | None:
    """Use the shared resolver when the compatible optional package is installed."""
    try:
        from quantem.data.huggingface import download as shared_download
    except ImportError:
        return None
    if not {"revision", "cache_dir", "force_download"}.issubset(
        inspect.signature(shared_download).parameters
    ):
        return None
    return Path(shared_download(name, **kwargs))


def download(
    name: str,
    *,
    repo: str | None = None,
    out: str | Path | None = None,
    verbose: bool = True,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> Path:
    """Download the folder or file dataset called ``name`` and return its local path.

    Prefer a qualified name such as ``"4dstem/gold_512_npy_bin4"``. A short
    name is accepted only when unique. Files already in the Hub cache are
    reused. The compatible optional ``quantem.data`` package owns resolution;
    this small fallback keeps widget-only installations working and resolves
    a name as ``quantem.data`` does: a dataset is a folder or a data file in
    the ``4dstem``, ``haadf``, ``denoise`` or ``widget`` bucket whose name
    does not start with ``placeholder_``.

    Parameters
    ----------
    name
        Qualified dataset ID, the path of a single-file dataset, or unique short name.
    repo
        Dataset repository. Defaults to ``bobleesj/quantem-data``.
    out
        Optional local destination, retaining the repository-relative paths.
    verbose
        Print download and cache information.
    revision
        Optional immutable repository commit for reproducible retrieval.
    cache_dir, force_download
        Hub cache location and explicit refresh control.

    Returns
    -------
    Path
        Downloaded dataset file or folder, including its metadata companions.

    Examples
    --------
    >>> folder = download("4dstem/gold_128_npy_bin8", revision="5925e4b49b2ca28ff82d6d5a6b9a8bfea9edfb20")
    """
    from huggingface_hub import list_repo_files, snapshot_download

    repo_id = repo or os.environ.get("QUANTEM_DATA_REPO") or DEFAULT_REPO
    if repo_id == DEFAULT_REPO and name == "gold_haadf_session":
        name = "widget/gold_session"
    shared = _shared_download(
        name, repo=repo_id, out=out, verbose=verbose, revision=revision,
        cache_dir=cache_dir, force_download=force_download,
    )
    if shared is not None:
        return shared
    revision_options = {"revision": revision} if revision is not None else {}
    files = list_repo_files(repo_id=repo_id, repo_type="dataset", **revision_options)
    candidates: dict[str, str] = {}
    for path in files:
        parts = path.split("/")
        if len(parts) < 2 or parts[0] not in _DATA_BUCKETS or parts[1].startswith("placeholder_"):
            continue
        # repository paths always use "/": Path answers haadf\gold.yaml on Windows, which matches nothing in a listing
        suffix = PurePosixPath(path).suffix.lower()
        is_monitor = parts[0] == "widget" and suffix == ".jsonl"
        if suffix not in _DATA_SUFFIXES and not is_monitor:
            continue
        # a payload under <bucket>/<name>/ declares a folder dataset; a file in a bucket is one, shown by its stem
        target = "/".join(parts[:2])
        key = target if len(parts) >= 3 else f"{parts[0]}/{PurePosixPath(parts[1]).stem}"
        if name in (key, target, key.split("/", 1)[1]):
            candidates[target] = "dir" if len(parts) >= 3 else "file"
    if not candidates:
        raise FileNotFoundError(
            f"{name!r} not found in {repo_id}; browse https://huggingface.co/datasets/{repo_id}"
        )
    # a folder and a file can share a name without the extension; the exact repository path still names one
    if name not in candidates and len(candidates) > 1:
        raise ValueError(
            f"{name!r} is ambiguous in {repo_id}: {sorted(candidates)}. "
            "Use one of these qualified dataset paths."
        )
    target_rel = name if name in candidates else next(iter(candidates))
    kind = candidates[target_rel]
    pattern = f"{target_rel}/*" if kind == "dir" else target_rel
    if kind == "file":
        sidecars = [str(PurePosixPath(target_rel).with_suffix(suffix)) for suffix in _SIDECAR_SUFFIXES]
        selected = [path for path in files if path == target_rel or path in sidecars]
        pattern = selected if len(selected) > 1 else target_rel
    if verbose:
        print(
            f"Downloading '{name}' from Hugging Face ({repo_id}) over the internet - "
            "speed depends on your connection, not your computer ...",
            flush=True,
        )
    start = time.perf_counter()
    root = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=pattern,
        local_dir=str(out) if out is not None else None,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        force_download=force_download,
        **revision_options,
    )
    result = Path(root) / target_rel
    if verbose:
        elapsed_s = time.perf_counter() - start
        size_gb = (
            sum(file.stat().st_size for file in result.rglob("*") if file.is_file())
            if result.is_dir()
            else result.stat().st_size
        ) / 1e9
        if elapsed_s < 1.0:
            print(f"'{name}' ({size_gb:.2f} GB) is already cached on disk - no re-download.\n  cached at: {result}", flush=True)
        else:
            print(
                f"Downloaded '{name}' ({size_gb:.2f} GB) in {elapsed_s:.0f}s ({size_gb * 1000 / elapsed_s:.0f} MB/s from Hugging Face).\n"
                f"  cached on disk - future loads are instant, no re-download.\n  cached at: {result}",
                flush=True,
            )
    return result
