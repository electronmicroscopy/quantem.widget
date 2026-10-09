"""Download one shared dataset from the public Hugging Face hub."""

import os
import time
from pathlib import Path

DEFAULT_REPO = "bobleesj/quantem-data"


def download(
    name: str,
    *,
    repo: str | None = None,
    out: str | Path | None = None,
    verbose: bool = True,
) -> Path:
    """Download the folder or file dataset called ``name`` and return its local path.

    ``name`` is the flat dataset name (for example ``"gold_512_npy_bin4"``);
    the hub layout ``<bucket>/<name>/...`` is resolved here so tutorials never
    spell out repository paths. Files already in the Hugging Face cache are
    not downloaded again.
    """
    from huggingface_hub import list_repo_files, snapshot_download

    repo_id = repo or os.environ.get("QUANTEM_DATA_REPO") or DEFAULT_REPO
    candidates: dict[str, str] = {}
    for path in list_repo_files(repo_id=repo_id, repo_type="dataset"):
        parts = path.split("/")
        if len(parts) >= 3 and parts[1] == name:
            candidates[f"{parts[0]}/{name}"] = "dir"
        elif len(parts) == 2 and Path(parts[1]).stem == name and not path.endswith(".json"):
            candidates[path] = "file"
    if not candidates:
        raise FileNotFoundError(
            f"{name!r} not found in {repo_id}; browse https://huggingface.co/datasets/{repo_id}"
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
    start = time.perf_counter()
    root = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=pattern,
        local_dir=str(out) if out is not None else None,
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
