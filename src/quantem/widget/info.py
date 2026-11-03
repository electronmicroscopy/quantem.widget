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

# Written by the frontend build: the quantem.gpu ref and commit the bundled WebGPU science kernels came from.
ENGINE_MANIFEST = Path(__file__).parent / "static" / "engine-manifest.json"


def profile() -> None:
    """Print the installed QuantEM stack and active compute environment.

    Use this single report in notebooks and bug reports instead of printing
    individual package versions. The report is local and does not contact
    package indexes or Git remotes.

    Installation labels describe local provenance, not publication status.
    Commits come from the imported checkout, VCS install metadata, or an
    explicit ``+g<commit>`` version label; unavailable commits stay unknown.

    Examples
    --------
    >>> import quantem.widget as qw
    >>> qw.profile()
    """
    import quantem.widget as qw

    def direct_install(distribution_name: str) -> dict:
        """Read pip's local provenance without exposing source URLs or paths."""
        try:
            raw = distribution(distribution_name).read_text("direct_url.json")
        except (PackageNotFoundError, OSError):
            return {}
        if not raw:
            return {}

        try:
            metadata = json.loads(raw)
        except ValueError:
            return {}
        return metadata if isinstance(metadata, dict) else {}

    def checkout_commit(loaded: Path | None) -> str | None:
        """Identify tracked source, never a repository surrounding site-packages."""
        if loaded is None or "site-packages" in loaded.parts:
            return None
        root = next(
            (parent for parent in loaded.parents if (parent / ".git").exists()),
            None,
        )
        if root is None:
            return None

        def git(*args: str) -> str:
            return subprocess.check_output(
                ["git", "-C", str(root), *args],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2,
            ).strip()

        try:
            git("ls-files", "--error-unmatch", str(loaded.relative_to(root)))
            commit = git("rev-parse", "--short=12", "HEAD")
            dirty = bool(git("status", "--porcelain", "--untracked-files=normal"))
        except (OSError, subprocess.SubprocessError):
            return None
        return commit + (" (dirty checkout)" if dirty else "")

    def print_distribution_status(distribution_name: str) -> None:
        """Print installation provenance and the associated source commit.

        A source override (``PYTHONPATH`` ahead of an editable install) is the usual
        reason a bug report does not match the code the reporter thinks they run.
        """
        metadata = direct_install(distribution_name)
        directory = metadata.get("dir_info") or {}
        source_url = metadata.get("url")
        parsed = urlparse(source_url if isinstance(source_url, str) else "")
        editable = isinstance(directory, dict) and bool(directory.get("editable"))
        source = (
            Path(unquote(parsed.path)).resolve()
            if editable and parsed.scheme == "file" else None
        )
        try:
            spec = find_spec(distribution_name)
        except (ImportError, ValueError):
            spec = None
        loaded = (
            Path(spec.origin).resolve()
            if spec is not None and spec.origin is not None
            else None
        )

        checkout = checkout_commit(loaded)
        override = (
            source is not None and loaded is not None
            and not loaded.is_relative_to(source)
        )
        vcs = metadata.get("vcs_info") or {}
        vcs_commit = vcs.get("commit_id") if isinstance(vcs, dict) else None
        try:
            installed = version(distribution_name)
        except PackageNotFoundError:
            installed = "unknown"
        local_version = installed.partition("+")[2]
        version_commit = re.search(r"(?:^|\.)g([0-9a-f]{7,40})(?:\.|$)", local_version)
        commit = "not recorded"
        if override or (checkout is not None and not editable):
            status = "source override (differs from installed metadata)"
            commit = checkout or commit
        elif editable:
            status = "editable checkout"
            commit = checkout or commit
        else:
            if vcs:
                status = "VCS install"
            elif parsed.scheme == "file":
                status = "local build"
            elif local_version or ".dev" in installed:
                status = "development build"
            elif metadata:
                status = "direct install"
            else:
                status = "installed package"
            if isinstance(vcs_commit, str) and re.fullmatch(r"[0-9a-fA-F]{7,64}", vcs_commit):
                commit = f"{vcs_commit} (VCS metadata)"
            elif version_commit:
                commit = f"{version_commit[1]} (version metadata)"
        print(f"  install       {status}")
        print(f"  commit        {commit}")

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
    # The bundles carry their own copy of the quantem.gpu WebGPU kernels, independent of the installed quantem.gpu.
    if ENGINE_MANIFEST.is_file():
        engine = json.loads(ENGINE_MANIFEST.read_text(encoding="utf-8"))["quantem_gpu"]
        source = engine["ref"] or f"quantem.gpu {engine['version']}"
        commit = f" ({engine['git_sha'][:12]})" if engine["git_sha"] else ""
        pin = "" if source == engine["pinned_ref"] else f", pinned {engine['pinned_ref']}"
        print(f"webgpu kernels  {source}{commit}{pin}")
    else:
        print("webgpu kernels  (frontend not built)")
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
    from quantem.widget.device import gpu_notice, host_platform, mps_available

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

        if host_platform()[1] == "arm64":
            # Apple silicon: the chip is the GPU and its memory is shared with it
            chip = sysctl("machdep.cpu.brand_string") or "Apple"
            memory = sysctl("hw.memsize")
            memory_gb = f"{int(memory) // (1024**3)} GB" if memory.isdigit() else "?"
            report["device"] = f"Apple Metal (MPS) - {chip}, {memory_gb} unified memory"
        else:
            # an Intel Mac's sysctl names its CPU, not the GPU MPS runs on
            report["device"] = "Apple Metal (MPS) - the GPU of this Intel Mac"
    elif backend == "cuda":
        report["device"] = f"CUDA - {torch.cuda.get_device_name(0)}"
    if verbose:
        print(f"quantem.widget {report['widget_version']}   |   {report['date']}")
        print(f"compute: {report['device']} (torch {report['torch']})")
        print(f"GPU path: {report['gpu_path']}   |   quantem core: {report['quantem_core']}")
        gpu_notice()  # once per process: a later device="auto" choice does not print it again
    return report
