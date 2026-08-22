"""Private helpers for :func:`quantem.widget.profile`."""

import json
import subprocess
from importlib.metadata import PackageNotFoundError, distribution
from importlib.util import find_spec
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from packaging.version import Version


def _editable_source(distribution_name: str) -> Path | None:
    """Return the source recorded for a PEP 610 editable installation."""
    try:
        raw = distribution(distribution_name).read_text("direct_url.json")
    except (PackageNotFoundError, OSError):
        return None
    if not raw:
        return None

    try:
        direct_url = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(direct_url, dict):
        return None
    if not direct_url.get("dir_info", {}).get("editable"):
        return None

    parsed = urlparse(direct_url.get("url", ""))
    if parsed.scheme != "file":
        return None
    return Path(unquote(parsed.path)).resolve()


def _loaded_path(module_name: str) -> Path | None:
    """Return the path of the module Python will actually import."""
    try:
        spec = find_spec(module_name)
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).resolve()


def _git(source: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(source), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _print_checkout(source: Path) -> None:
    head = _git(source, "rev-parse", "--short=12", "HEAD")
    branch = _git(source, "branch", "--show-current")
    changes = _git(source, "status", "--porcelain")
    if head is None or branch is None or changes is None:
        return

    state = "; local changes" if changes else ""
    print(f"  checkout      {branch or 'detached'} @ {head}{state}")


def _latest_testpypi_version(distribution_name: str) -> str:
    package = distribution_name.replace(".", "-")
    request = Request(
        f"https://test.pypi.org/pypi/{package}/json",
        headers={"User-Agent": "quantem.widget profile()"},
    )
    with urlopen(request, timeout=4) as response:
        return json.load(response)["info"]["version"]


def _print_update(distribution_name: str, installed: str) -> None:
    try:
        latest = _latest_testpypi_version(distribution_name)
        installed_version = Version(installed)
        latest_version = Version(latest)
    except (KeyError, OSError, ValueError):
        print("  release       update check unavailable")
        return

    print(f"  TestPyPI      latest {latest}")
    if installed_version < latest_version:
        print(f"  WARNING       installed metadata {installed} trails {latest}")
    elif installed_version > latest_version:
        print("  release       newer than TestPyPI")
    else:
        print("  release       current")


def print_distribution_status(
    distribution_name: str,
    installed: str,
    *,
    check_updates: bool,
) -> None:
    """Print install mode and optional TestPyPI status."""
    source = _editable_source(distribution_name)
    loaded = _loaded_path(distribution_name)

    if source is None:
        print("  install       published package")
    elif loaded is not None and not loaded.is_relative_to(source):
        print("  install       source override (differs from installed metadata)")
    else:
        print("  install       editable checkout")
        _print_checkout(source)

    if check_updates:
        _print_update(distribution_name, installed)
