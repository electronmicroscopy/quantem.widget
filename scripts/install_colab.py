"""Install the newest QuantEM release candidates in Google Colab."""

import json
import subprocess
import sys
from importlib.metadata import version
from urllib.request import urlopen

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet


def _testpypi_wheels(project: str) -> list[dict]:
    """Return every non-yanked TestPyPI wheel of a project, each with the version of its release."""
    with urlopen(
        f"https://test.pypi.org/pypi/{project}/json", timeout=30
    ) as response:
        releases = json.load(response)["releases"]
    wheels = [
        {**file, "version": release}
        for release, files in releases.items()
        for file in files
        if file["filename"].endswith(".whl") and not file.get("yanked", False)
    ]
    if not wheels:
        raise RuntimeError(
            f"TestPyPI has no installable wheel for {project}. "
            "Ask the QuantEM maintainers to publish a new release candidate."
        )
    return wheels


def _newest(wheels: list[dict]) -> dict:
    """Return the wheel uploaded last."""
    return max(wheels, key=lambda file: file["upload_time_iso_8601"])


def _wheel_url(wheel: dict) -> str:
    """Return a wheel's URL with its sha256, so pip verifies the download."""
    digest = wheel.get("digests", {}).get("sha256")
    return f'{wheel["url"]}#sha256={digest}' if digest else wheel["url"]


def _gpu_versions_for(widget_version: str) -> SpecifierSet:
    """Return the quantem.gpu versions a widget release was built and tested with.

    Each widget release bundles the browser kernels of one quantem.gpu release
    and pins it in its ``[cuda]`` extra. Taking the newest quantem.gpu instead
    would pair the widget with a release it was never run against whenever
    quantem.gpu is published first.
    """
    with urlopen(
        f"https://test.pypi.org/pypi/quantem.widget/{widget_version}/json", timeout=30
    ) as response:
        requirements = json.load(response)["info"]["requires_dist"] or []
    for text in requirements:
        requirement = Requirement(text)
        if requirement.name.replace("-", ".") == "quantem.gpu" and "cuda" in requirement.extras:
            return requirement.specifier
    raise RuntimeError(
        f"quantem.widget {widget_version} on TestPyPI names no quantem.gpu version. "
        "Ask the QuantEM maintainers to publish a new release candidate."
    )


def install_latest_rc() -> None:
    """Install current widget/GPU RCs without using TestPyPI for dependencies."""
    import numpy as np
    from google.colab import output

    output.enable_custom_widget_manager()
    # quantem.gpu requires numba; pinning the numpy and numba Colab already
    # loaded keeps pip from replacing compiled modules in the running kernel.
    numba_version = version("numba")
    widget = _newest(_testpypi_wheels("quantem.widget"))
    pinned = _gpu_versions_for(widget["version"])
    matching = [wheel for wheel in _testpypi_wheels("quantem.gpu") if pinned.contains(wheel["version"], prereleases=True)]
    if not matching:
        raise RuntimeError(
            f'quantem.widget {widget["version"]} needs quantem.gpu{pinned}, which TestPyPI does not have. '
            "Ask the QuantEM maintainers to publish it."
        )
    widget_wheel, gpu_wheel = _wheel_url(widget), _wheel_url(_newest(matching))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            f"numpy=={np.__version__}",
            f"numba=={numba_version}",
            f"quantem.gpu @ {gpu_wheel}",
            widget_wheel,
        ],
        check=True,
    )
    from quantem.widget import profile

    print("QuantEM ready")
    profile()


if __name__ == "__main__":
    install_latest_rc()
