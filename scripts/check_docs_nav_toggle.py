#!/usr/bin/env python3
"""Guard the docs mobile hamburger against sphinx theme drift.

Both sphinx-book-theme and pydata-sphinx-theme wire the sidebar drawer to
``document.querySelector('.primary-toggle')``, the FIRST matching element.
pydata-sphinx-theme >= 0.17 renders its own extra (display: none) header
button with that class ahead of the visible sphinx-book-theme hamburger, so
every handler binds to the invisible button and the visible hamburger goes
dead (found live on 2026-07-28: the phone nav could not be opened at all).

``docs/_static/nav-toggle-fix.js`` compensates by forwarding clicks from the
unwired buttons to the wired one, so the themes stay UNPINNED. This check
fails the build when a page carries multiple ``primary-toggle`` buttons
WITHOUT that shim loaded (the state where the hamburger is dead), or no
toggle at all. Run after ``jupyter-book build docs``:

    python scripts/check_docs_nav_toggle.py [docs/_build/html]
"""

import re
import sys
from pathlib import Path

TOGGLE_RE = re.compile(r"<(?:button|label)[^>]*class=\"[^\"]*\bprimary-toggle\b", re.I)
SHIM_RE = re.compile(r"nav-toggle-fix\.js", re.I)


def main() -> int:
    """Check every built theme page; exit 1 when one has no toggle or several without the shim."""
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/_build/html")
    pages = [
        page
        for page in root.rglob("*.html")
        if "_sources" not in page.parts and "_static" not in page.parts
    ]
    if not pages:
        print(f"check_docs_nav_toggle: no built pages under {root}", file=sys.stderr)
        return 1
    broken = []
    checked = 0
    for page in pages:
        text = page.read_text(encoding="utf-8", errors="replace")
        if "bd-main" not in text:
            continue  # redirect stub or bare page without the theme scaffold
        checked += 1
        toggles = len(TOGGLE_RE.findall(text))
        if toggles == 0:
            broken.append((page.relative_to(root), "no primary-toggle button"))
        elif toggles > 1 and not SHIM_RE.search(text):
            broken.append(
                (
                    page.relative_to(root),
                    f"{toggles} primary-toggle buttons and nav-toggle-fix.js "
                    "is not loaded, so the visible hamburger is dead",
                )
            )
    if broken:
        print(
            "check_docs_nav_toggle: FAIL: theme drift broke the mobile nav "
            "(handlers bind to the first .primary-toggle, which newer pydata "
            "themes render hidden). Ensure docs/_static/nav-toggle-fix.js is "
            "present and loaded on every page:",
            file=sys.stderr,
        )
        for relative, reason in broken[:20]:
            print(f"  {relative}: {reason}", file=sys.stderr)
        return 1
    if not checked:
        print("check_docs_nav_toggle: no theme pages found to check", file=sys.stderr)
        return 1
    print(f"check_docs_nav_toggle: OK: {checked} pages have a working nav toggle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
