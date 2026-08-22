"""Human-readable environment information for notebooks."""

import json
import os
import platform
import subprocess
from datetime import datetime
from importlib.metadata import PackageNotFoundError, distribution, version
from importlib.util import find_spec
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from packaging.version import Version


def profile(*, check_updates: bool = False) -> None:
    """Print the installed QuantEM stack and active compute environment.

    Use this single report in notebooks and bug reports instead of printing
    individual package versions. The default report is local and does not
    contact package indexes or Git remotes.

    Parameters
    ----------
    check_updates : bool, default False
        Compare installed widget and GPU metadata with TestPyPI. This opt-in
        check needs network access.

    Examples
    --------
    >>> import quantem.widget as qw
    >>> qw.profile()
    """
    import quantem.widget as qw

    def editable_source(distribution_name: str) -> Path | None:
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
        directory = direct_url.get("dir_info")
        source_url = direct_url.get("url")
        if not isinstance(directory, dict) or not directory.get("editable"):
            return None
        if not isinstance(source_url, str):
            return None

        parsed = urlparse(source_url)
        if parsed.scheme != "file":
            return None
        return Path(unquote(parsed.path)).resolve()

    def print_update(distribution_name: str, installed: str) -> None:
        package = distribution_name.replace(".", "-")
        request = Request(
            f"https://test.pypi.org/pypi/{package}/json",
            headers={"User-Agent": "quantem.widget profile()"},
        )
        try:
            with urlopen(request, timeout=4) as response:
                latest = json.load(response)["info"]["version"]
            installed_version = Version(installed)
            latest_version = Version(latest)
        except (KeyError, OSError, TypeError, ValueError):
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
    ) -> None:
        source = editable_source(distribution_name)
        try:
            spec = find_spec(distribution_name)
        except (ImportError, ValueError):
            spec = None
        loaded = (
            Path(spec.origin).resolve()
            if spec is not None and spec.origin is not None
            else None
        )

        if source is None:
            print("  install       published package")
        elif loaded is not None and not loaded.is_relative_to(source):
            print("  install       source override (differs from installed metadata)")
        else:
            print("  install       editable checkout")

        if check_updates:
            print_update(distribution_name, installed)

    print(f"quantem.widget  {qw.__version__}")
    print_distribution_status("quantem.widget", qw.__version__)
    try:
        gpu_version = version("quantem.gpu")
        print(f"quantem.gpu     {gpu_version}")
        print_distribution_status("quantem.gpu", gpu_version)
    except PackageNotFoundError:
        print("quantem.gpu     (not installed)")
    try:
        import quantem

        print(f"quantem         {getattr(quantem, '__version__', '?')}")
    except ImportError:
        print("quantem         (not importable)")
    try:
        import torch

        if torch.cuda.is_available():
            device = f"cuda ({torch.cuda.get_device_name(0)})"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps (Apple)"
        else:
            device = "cpu"
        print(f"torch           {torch.__version__}  device={device}")
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
            print(f"GPUs            {count} visible (CUDA_VISIBLE_DEVICES={visible})")
            for index in range(count):
                free, total = torch.cuda.mem_get_info(index)
                print(
                    f"  GPU{index}          {(total - free) / 1e9:5.1f} used / "
                    f"{total / 1e9:.0f} GB  ({free / 1e9:.0f} free)"
                )
            print(
                f"  torch pool    {torch.cuda.memory_allocated() / 1e9:.1f} live / "
                f"{torch.cuda.memory_reserved() / 1e9:.1f} reserved GB"
            )
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            current = (
                torch.mps.current_allocated_memory() / 1e9
                if hasattr(torch.mps, "current_allocated_memory")
                else 0.0
            )
            driver = (
                torch.mps.driver_allocated_memory() / 1e9
                if hasattr(torch.mps, "driver_allocated_memory")
                else 0.0
            )
            print(f"VRAM (MPS)      {current:.1f} live / {driver:.1f} driver GB")
    except ImportError:
        print("torch           (not importable)")
    print(f"python          {platform.python_version()}")


def device_info(verbose: bool = True) -> dict[str, str]:
    """Return and optionally print the active device information."""
    from quantem.gpu.device import detect

    import quantem.widget as qw

    backend = detect()
    report = {
        "widget_version": qw.__version__,
        "date": str(datetime.now().astimezone().date()),
        "backend": backend,
        "device": "CPU",
    }
    if backend == "mps":

        def sysctl(key: str) -> str:
            try:
                result = subprocess.run(
                    ["sysctl", "-n", key],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
            except (OSError, subprocess.SubprocessError):
                return ""
            return result.stdout.strip()

        chip = sysctl("machdep.cpu.brand_string") or "Apple Silicon"
        memory = sysctl("hw.memsize")
        memory_gb = f"{int(memory) // (1024**3)} GB" if memory.isdigit() else "?"
        report["device"] = f"Apple Metal (MPS) - {chip}, {memory_gb} unified memory"
    elif backend == "cuda":
        try:
            import torch

            report["device"] = f"CUDA - {torch.cuda.get_device_name(0)}"
        except (AssertionError, ImportError, RuntimeError):
            report["device"] = "CUDA"
    if verbose:
        print(f"quantem.widget {report['widget_version']}   |   {report['date']}")
        print(f"compute: {report['device']}")
    return report
