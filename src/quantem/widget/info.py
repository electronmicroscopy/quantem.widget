"""Human-readable environment information for notebooks."""

import json
import os
import platform
import re
import subprocess
from datetime import datetime
from importlib.metadata import PackageNotFoundError, distribution, version
from importlib.util import find_spec
from pathlib import Path
from urllib.parse import unquote, urlparse

import torch


def profile() -> None:
    """Print the installed QuantEM stack and active compute environment.

    Use this single report in notebooks and bug reports instead of printing
    individual package versions. The report is local and does not contact
    package indexes or Git remotes.

    Examples
    --------
    >>> import quantem.widget as qw
    >>> qw.profile()
    """
    import quantem.widget as qw

    def editable_source(distribution_name: str) -> Path | None:
        """The checkout an editable install points at, from its ``direct_url.json``; None for a published wheel."""
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

    def print_distribution_status(distribution_name: str) -> None:
        """Print whether the package is published, editable, or imported from a path that overrides its metadata.

        A source override (``PYTHONPATH`` ahead of an editable install) is the usual
        reason a bug report does not match the code the reporter thinks they run.
        """
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

    from quantem.widget.adapters import core as core_adapter
    from quantem.widget.adapters import gpu as gpu_adapter

    print(f"quantem.widget  {qw.__version__}")
    print_distribution_status("quantem.widget")
    for name, importable in (("quantem.gpu", gpu_adapter.available()), ("quantem", core_adapter.available())):
        try:
            installed = version(name)
        except PackageNotFoundError:
            print(f"{name:<15} (not installed)")
            continue
        print(f"{name:<15} {installed}{'' if importable else '  (installed but not importable)'}")
        print_distribution_status(name)
    if torch.cuda.is_available():
        # the model family names the card; the edition suffix only lengthens the line
        cuda_name = torch.cuda.get_device_name(0).strip()
        model_family = re.match(r"^(NVIDIA RTX PRO \d+)", cuda_name)
        device = f"cuda ({model_family.group(1) if model_family else cuda_name})"
    elif torch.backends.mps.is_available():
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
    elif torch.backends.mps.is_available():
        current = torch.mps.current_allocated_memory() / 1e9
        driver = torch.mps.driver_allocated_memory() / 1e9
        print(f"VRAM (MPS)      {current:.1f} live / {driver:.1f} driver GB")
    print(f"python          {platform.python_version()}")


def device_info(verbose: bool = True) -> dict[str, str]:
    """Return and optionally print where the widgets compute and which GPU path is installed.

    ``backend`` is the device ``device="auto"`` picks (``cuda``, ``mps`` or
    ``cpu``); ``gpu_path`` says whether quantem.gpu (encoded acquisitions and GPU
    file loading) is importable, and ``quantem_core`` the same for quantem core.
    """
    import quantem.widget as qw
    from quantem.widget.adapters import core as core_adapter
    from quantem.widget.adapters import gpu as gpu_adapter
    from quantem.widget.device import gpu_notice_text, mps_available

    backend = "cuda" if torch.cuda.is_available() else "mps" if mps_available() else "cpu"
    report = {
        "widget_version": qw.__version__,
        "date": str(datetime.now().astimezone().date()),
        "backend": backend,
        "device": "CPU",
        "torch": torch.__version__,
        "gpu_path": "quantem.gpu" if gpu_adapter.available() else "not installed",
        "quantem_core": "installed" if core_adapter.available() else "not installed",
    }
    if backend == "mps":

        def sysctl(key: str) -> str:
            """One macOS ``sysctl`` value, or "" where the command is missing or slow."""
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

        chip = sysctl("machdep.cpu.brand_string") or "Apple"
        memory = sysctl("hw.memsize")
        memory_gb = f"{int(memory) // (1024**3)} GB" if memory.isdigit() else "?"
        report["device"] = f"Apple Metal (MPS) - {chip}, {memory_gb} memory"
    elif backend == "cuda":
        report["device"] = f"CUDA - {torch.cuda.get_device_name(0)}"
    if verbose:
        print(f"quantem.widget {report['widget_version']}   |   {report['date']}")
        print(f"compute: {report['device']} (torch {report['torch']})")
        print(f"GPU path: {report['gpu_path']}   |   quantem core: {report['quantem_core']}")
        notice = gpu_notice_text()
        if notice is not None:
            print(notice)
    return report
