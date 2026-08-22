"""Private environment diagnostics used by :func:`quantem.widget.profile`."""

from __future__ import annotations

import json
from importlib.metadata import distribution
from pathlib import Path
import subprocess
from urllib.parse import unquote, urlparse
import urllib.request

from packaging.version import Version


def _editable_source(distribution_name: str) -> Path | None:
    """Return the source checkout for a PEP 610 editable installation."""
    try:
        raw = distribution(distribution_name).read_text("direct_url.json")
        if not raw:
            return None
        direct_url = json.loads(raw)
        url = direct_url.get("url", "")
        if not direct_url.get("dir_info", {}).get("editable"):
            return None
        parsed = urlparse(url)
        if parsed.scheme != "file":
            return None
        return Path(unquote(parsed.path)).resolve()
    except Exception:
        # profile() is diagnostic: malformed optional metadata must not stop a
        # notebook from reporting the rest of its environment.
        return None


def _git(source: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(source), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=8,
    ).stdout.strip()


def _print_checkout_status(source: Path) -> None:
    try:
        head = _git(source, "rev-parse", "--short=12", "HEAD")
        branch = _git(source, "branch", "--show-current") or "detached"
        tracking = _git(source, "rev-parse", "--abbrev-ref", "@{upstream}")

        try:
            remote, remote_branch = tracking.split("/", 1)
            _git(source, "fetch", "--quiet", remote, remote_branch)
        except Exception as exc:
            print(f"  note          remote refresh unavailable ({exc})")

        ahead, behind = map(
            int,
            _git(
                source,
                "rev-list",
                "--left-right",
                "--count",
                f"HEAD...{tracking}",
            ).split(),
        )
        if behind:
            state = f"BEHIND by {behind} commit(s)"
        elif ahead:
            state = f"ahead by {ahead} commit(s)"
        else:
            state = "current"
        print(f"  checkout      {branch} @ {head}; {state} vs {tracking}")

        if _git(source, "status", "--porcelain"):
            print("  note          working tree has local changes (preserved)")
    except Exception as exc:
        print(f"  checkout      status check failed ({exc})")


def _latest_testpypi_version(distribution_name: str) -> str:
    package = distribution_name.replace(".", "-")
    request = urllib.request.Request(
        f"https://test.pypi.org/pypi/{package}/json",
        headers={"User-Agent": "quantem.widget profile()"},
    )
    with urllib.request.urlopen(request, timeout=4) as response:
        return json.load(response)["info"]["version"]


def _print_release_status(
    distribution_name: str,
    installed: str,
    *,
    development_mode: bool,
) -> None:
    try:
        latest = _latest_testpypi_version(distribution_name)
        installed_version = Version(installed)
        latest_version = Version(latest)
        print(f"  TestPyPI      latest {latest}")

        if installed_version < latest_version:
            subject = "metadata" if development_mode else "installed version"
            action = (
                "editable source may contain newer code"
                if development_mode
                else "upgrade before relying on new APIs"
            )
            print(
                f"  WARNING       {subject} {installed} trails latest TestPyPI "
                f"{latest}; {action}"
            )
        elif installed_version > latest_version:
            print("  note          development version is newer than TestPyPI")
        else:
            print("  release       current")
    except Exception as exc:
        print(f"  release       update check unavailable ({exc})")


def print_distribution_status(distribution_name: str, installed: str) -> None:
    """Print install mode, source freshness, and published-release status."""
    source = _editable_source(distribution_name)
    development_mode = source is not None

    if development_mode:
        print("  install       DEVELOPMENT MODE (editable)")
        print(f"  source        {source}")
        _print_checkout_status(source)
    else:
        print("  install       published package")

    _print_release_status(
        distribution_name,
        installed,
        development_mode=development_mode,
    )
