"""Install the newest QuantEM release candidates in Google Colab."""

from __future__ import annotations

import json
import subprocess
import sys
from importlib.metadata import version
from urllib.request import urlopen


def _latest_testpypi_wheel_url(project: str) -> str:
    """Return the newest non-yanked TestPyPI wheel URL with its hash."""
    with urlopen(
        f"https://test.pypi.org/pypi/{project}/json", timeout=30
    ) as response:
        releases = json.load(response)["releases"]
    wheels = [
        file
        for files in releases.values()
        for file in files
        if file["filename"].endswith(".whl") and not file.get("yanked", False)
    ]
    if not wheels:
        raise RuntimeError(
            f"TestPyPI has no installable wheel for {project}. "
            "Ask the QuantEM maintainers to publish a new release candidate."
        )
    wheel = max(wheels, key=lambda file: file["upload_time_iso_8601"])
    digest = wheel.get("digests", {}).get("sha256")
    return f'{wheel["url"]}#sha256={digest}' if digest else wheel["url"]


def install_latest_rc() -> None:
    """Install current widget/GPU RCs without using TestPyPI for dependencies."""
    import numpy as np
    from google.colab import output

    output.enable_custom_widget_manager()
    numba_version = version("numba")
    widget_wheel = _latest_testpypi_wheel_url("quantem.widget")
    gpu_wheel = _latest_testpypi_wheel_url("quantem.gpu")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            f"numpy=={np.__version__}",
            f"numba=={numba_version}",
            f"quantem.gpu[movie] @ {gpu_wheel}",
            widget_wheel,
        ],
        check=True,
    )
    from quantem.widget import profile

    print("QuantEM ready")
    profile()


if __name__ == "__main__":
    install_latest_rc()
