#!/usr/bin/env python3
"""Fail when tutorial notebooks become too large for the main branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _string_bytes(value: object) -> int:
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, list):
        return sum(_string_bytes(item) for item in value)
    if isinstance(value, dict):
        return sum(_string_bytes(item) for item in value.values())
    return 0


def _output_bytes(notebook: dict) -> int:
    total = 0
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            total += _string_bytes(output.get("data", {}))
            total += _string_bytes(output.get("text", ""))
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=["docs/tutorials"])
    parser.add_argument("--max-mb", type=float, default=10.0)
    parser.add_argument("--max-output-mb", type=float, default=9.0)
    args = parser.parse_args()

    max_bytes = int(args.max_mb * 1024 * 1024)
    max_output_bytes = int(args.max_output_mb * 1024 * 1024)
    notebooks: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            notebooks.extend(sorted(path.rglob("*.ipynb")))
        elif path.suffix == ".ipynb":
            notebooks.append(path)

    failures: list[str] = []
    print("Notebook size guard")
    for path in notebooks:
        size = path.stat().st_size
        notebook = json.loads(path.read_text(encoding="utf-8"))
        outputs = _output_bytes(notebook)
        widget_state = _string_bytes(notebook.get("metadata", {}).get("widgets", {}))
        print(f"- {path}: {size / 1024 / 1024:.2f} MB, outputs {outputs / 1024 / 1024:.2f} MB")
        if widget_state:
            # The docs CI re-executes every tutorial (execute_notebooks: force)
            # and bakes fresh widget state into the published HTML, so committed
            # state is never used; it only grows git history on every save.
            failures.append(
                f"{path} carries {widget_state / 1024 / 1024:.2f} MB of baked widget "
                f"state (metadata.widgets); strip it with "
                f"jq 'del(.metadata.widgets)' {path} > tmp && mv tmp {path}"
            )
        if size > max_bytes:
            failures.append(f"{path} is {size / 1024 / 1024:.2f} MB > {args.max_mb:.2f} MB")
        if outputs > max_output_bytes:
            failures.append(
                f"{path} embeds {outputs / 1024 / 1024:.2f} MB of outputs "
                f"> {args.max_output_mb:.2f} MB"
            )

    if failures:
        print("\nNotebook size guard failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Notebook size guard passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
