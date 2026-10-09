#!/usr/bin/env python3
"""Stage AnyWidget's AMD runtime for offline Jupyter Book pages.

The ipywidgets HTML manager resolves the ``anywidget`` model relative to each
rendered page before trying a public CDN. All executable widget tutorials live
under ``tutorials/``, so the book must publish ``tutorials/anywidget.js``. The
source is copied from the installed dependency at build time; no third-party
bundle is vendored in the repository.
"""

import argparse
import importlib.util
import shutil
from pathlib import Path


def _runtime_path() -> Path:
    """The installed AnyWidget AMD runtime (``nbextension/index.js``); RuntimeError when it is missing."""
    spec = importlib.util.find_spec("anywidget")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "AnyWidget is not installed. Install the documentation dependencies "
            "before building quantem.widget tutorials."
        )
    runtime = Path(spec.origin).resolve().parent / "nbextension" / "index.js"
    if not runtime.is_file():
        raise RuntimeError(
            f"AnyWidget's AMD runtime was not found at {runtime}. "
            "Reinstall the supported AnyWidget package."
        )
    return runtime


def main() -> int:
    """Copy the runtime to ``--output``, where the book build publishes it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/_extra/tutorials/anywidget.js"),
        help="Staged runtime path copied into the built book.",
    )
    args = parser.parse_args()

    source = _runtime_path()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output)
    print(f"Staged AnyWidget runtime: {source} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
