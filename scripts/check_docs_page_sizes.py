#!/usr/bin/env python
"""Guard the BUILT docs pages against runaway widget-state weight.

`scripts/check_notebook_sizes.py` guards the committed notebooks, but since
the docs build executes tutorials and bakes widget state into the HTML, a
page can quietly grow to tens of MB (a widget syncing a bulk buffer, a
tutorial loading too much data) without any committed file changing. This
checks every built tutorial page and, when one exceeds the budget, names the
heaviest widget buffers inside its baked state so the offending trait is
obvious.

Run after `jupyter-book build docs`:

    python scripts/check_docs_page_sizes.py
    python scripts/check_docs_page_sizes.py --max-mb 25 --html-dir docs/_build/html
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

STATE_RE = re.compile(
    r'<script type="application/vnd\.jupyter\.widget-state\+json">(.*?)</script>',
    re.S,
)


def heaviest_buffers(html: str, top: int = 5) -> list[str]:
    """Names + sizes of the largest widget buffers baked into the page."""
    match = STATE_RE.search(html)
    if not match:
        return []
    try:
        state = json.loads(match.group(1)).get("state", {})
    except json.JSONDecodeError:
        return []
    rows: list[tuple[int, str]] = []
    for entry in state.values():
        for buffer in entry.get("buffers", []):
            path = "/".join(str(p) for p in buffer.get("path", []))
            rows.append((len(buffer.get("data", "")), path or "?"))
    rows.sort(reverse=True)
    return [f"{size / 1e6:.1f} MB {path}" for size, path in rows[:top] if size > 100_000]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html-dir", type=Path, default=Path("docs/_build/html"))
    parser.add_argument("--max-mb", type=float, default=50.0)
    args = parser.parse_args()

    pages = sorted((args.html_dir / "tutorials").glob("*.html"))
    if not pages:
        print(f"No built tutorial pages under {args.html_dir}; run jupyter-book build docs first.")
        return 1

    max_bytes = int(args.max_mb * 1024 * 1024)
    failures: list[str] = []
    for page in pages:
        size = page.stat().st_size
        print(f"- {page.relative_to(args.html_dir)}: {size / 1024 / 1024:.1f} MB")
        if size > max_bytes:
            detail = heaviest_buffers(page.read_text(errors="replace"))
            lines = "".join(f"\n    {row}" for row in detail) or "\n    (no widget-state blob found)"
            failures.append(
                f"{page.relative_to(args.html_dir)} is {size / 1024 / 1024:.1f} MB "
                f"> {args.max_mb:.1f} MB budget. Heaviest baked widget buffers:{lines}"
            )

    if failures:
        print("\nDocs page size guard FAILED:")
        for failure in failures:
            print(f"- {failure}")
        print(
            "\nShrink the tutorial's demo payload explicitly (visible stride/"
            "subset in the cell), or add the offending trait to that widget's "
            "_UNSAVED_HEAVY_KEYS if it should never be baked."
        )
        return 1
    print("Docs page size guard passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
