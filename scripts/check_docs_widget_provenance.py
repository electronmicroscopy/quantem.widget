#!/usr/bin/env python3
"""Verify that baked tutorial widgets use this checkout's frontend bundles."""

import argparse
import hashlib
import json
import re
from pathlib import Path


STATE_RE = re.compile(
    r'<script type="application/vnd\.jupyter\.widget-state\+json">(.*?)</script>',
    re.DOTALL,
)


def _digest(text: str) -> str:
    """sha256 of a bundle, short enough to compare a stale page against the build by eye."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    """Compare every baked AnyWidget ``_esm`` with this checkout's bundles; exit 1 on a mismatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html_dir", nargs="?", type=Path, default=Path("docs/_build/html"))
    parser.add_argument(
        "--static-dir",
        type=Path,
        default=Path("src/quantem/widget/static"),
        help="Frontend bundles built from the checkout under test.",
    )
    args = parser.parse_args()

    bundles = {
        path.read_text(encoding="utf-8"): path.name
        for path in sorted(args.static_dir.glob("*.js"))
    }
    if not bundles:
        raise SystemExit(
            f"No frontend bundles found under {args.static_dir}; run npm run build first."
        )

    checked = 0
    mismatches: list[str] = []
    for page in sorted((args.html_dir / "tutorials").glob("*.html")):
        match = STATE_RE.search(page.read_text(encoding="utf-8", errors="replace"))
        if match is None:
            continue
        state = json.loads(match.group(1)).get("state", {})
        for model in state.values():
            if model.get("model_module") != "anywidget":
                continue
            esm = model.get("state", {}).get("_esm")
            if not isinstance(esm, str):
                mismatches.append(f"{page.name}: AnyWidget model has no embedded _esm bundle")
                continue
            checked += 1
            if esm not in bundles:
                widget_id = model.get("state", {}).get("_anywidget_id", "unknown widget")
                mismatches.append(
                    f"{page.name}: {widget_id} bundle sha256={_digest(esm)} "
                    "does not match this checkout's npm build"
                )

    if not checked:
        raise SystemExit(f"No baked AnyWidget models found under {args.html_dir / 'tutorials'}.")
    if mismatches:
        print("Docs widget provenance check failed:")
        for mismatch in mismatches:
            print(f"- {mismatch}")
        return 1
    print(
        f"Docs widget provenance check passed: {checked} baked model(s) match "
        f"{len(bundles)} checkout bundle(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
