#!/usr/bin/env python3
"""Guard the main branch against accidental large data artifacts."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DATA_SUFFIXES = {
    ".dm3",
    ".dm4",
    ".emd",
    ".gif",
    ".h5",
    ".hdf5",
    ".html",
    ".mrc",
    ".npy",
    ".npz",
    ".ser",
    ".tif",
    ".tiff",
}

# This public README demo intentionally trades repository size for enough
# spatial and temporal resolution to show live Show4DSTEM interaction.
DATA_MAX_MB_EXCEPTIONS = {
    "docs/_static/show4dstem-serin-gold.gif": 20.0,
}


def _tracked_files() -> list[Path]:
    out = subprocess.check_output(["git", "ls-files"], text=True)
    return [Path(line) for line in out.splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Optional files to check. Defaults to all git-tracked files.",
    )
    parser.add_argument("--max-mb", type=float, default=25.0)
    parser.add_argument("--data-max-mb", type=float, default=5.0)
    args = parser.parse_args()

    max_bytes = int(args.max_mb * 1024 * 1024)
    data_max_bytes = int(args.data_max_mb * 1024 * 1024)
    failures: list[str] = []

    print("Tracked file size guard")
    paths = args.paths if args.paths else _tracked_files()
    for path in paths:
        if not path.exists():
            continue
        size = path.stat().st_size
        suffix = path.suffix.lower()
        if size > max_bytes:
            failures.append(f"{path} is {size / 1024 / 1024:.2f} MB > {args.max_mb:.2f} MB")
        data_limit_mb = DATA_MAX_MB_EXCEPTIONS.get(path.as_posix(), args.data_max_mb)
        data_limit_bytes = int(data_limit_mb * 1024 * 1024)
        if suffix in DATA_SUFFIXES and size > data_limit_bytes:
            failures.append(
                f"{path} is a data/rendered artifact ({suffix}) and "
                f"{size / 1024 / 1024:.2f} MB > {data_limit_mb:.2f} MB"
            )
        elif (
            suffix in DATA_SUFFIXES
            and data_limit_mb != args.data_max_mb
            and size > data_max_bytes
        ):
            print(
                f"Approved size exception: {path} is {size / 1024 / 1024:.2f} MB "
                f"<= {data_limit_mb:.2f} MB"
            )

    if failures:
        print("\nTracked file size guard failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Tracked file size guard passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
