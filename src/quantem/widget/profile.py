import importlib.metadata
import os
import platform
import shutil


class _ProfileResult(dict):
    """Dict subclass with empty repr so Jupyter doesn't display the raw dict."""

    def __repr__(self) -> str:
        return ""

    def _repr_html_(self) -> str:
        return ""


def profile() -> dict:
    info: dict = {}

    # quantem-widget version
    try:
        info["quantem_widget_version"] = importlib.metadata.version("quantem-widget")
    except importlib.metadata.PackageNotFoundError:
        info["quantem_widget_version"] = "unknown"

    # quantem version
    try:
        info["quantem_version"] = importlib.metadata.version("quantem")
    except importlib.metadata.PackageNotFoundError:
        info["quantem_version"] = "not installed"

    # Python
    info["python_version"] = (
        f"{platform.python_version()} ({platform.python_implementation()})"
    )

    # NumPy
    try:
        import numpy as np
        info["numpy_version"] = np.__version__
    except ImportError:
        info["numpy_version"] = "not installed"

    # PyTorch
    try:
        import torch
        info["pytorch_version"] = torch.__version__
    except ImportError:
        torch = None  # type: ignore[assignment]
        info["pytorch_version"] = "not installed"

    # Compute device
    device_str = "cpu"
    gpu_name = ""
    if torch is not None:
        try:
            from quantem.core.config import validate_device
            device_str, _ = validate_device(None)
        except Exception:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_str = "mps"
            elif torch.cuda.is_available():
                device_str = "cuda"
        if device_str == "mps":
            gpu_name = platform.processor() or "Apple Silicon"
        elif device_str == "cuda":
            try:
                props = torch.cuda.get_device_properties(torch.cuda.current_device())
                gpu_name = props.name
            except Exception:
                gpu_name = "unknown"
    info["compute_device"] = device_str
    info["gpu_name"] = gpu_name

    # System RAM
    import psutil
    vm = psutil.virtual_memory()
    info["system_ram_gb"] = round(vm.total / (1024**3), 1)
    info["system_ram_available_gb"] = round(vm.available / (1024**3), 1)

    # VRAM
    if device_str == "cuda" and torch is not None:
        try:
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            total = props.total_memory
            allocated = torch.cuda.memory_allocated(idx)
            info["vram_gb"] = round(total / (1024**3), 1)
            info["vram_available_gb"] = round((total - allocated) / (1024**3), 1)
        except Exception:
            info["vram_gb"] = "unknown"
            info["vram_available_gb"] = "unknown"
    elif device_str == "mps":
        info["vram_gb"] = "shared"
        info["vram_available_gb"] = "shared"
    else:
        info["vram_gb"] = "N/A"
        info["vram_available_gb"] = "N/A"

    # Jupyter — check both JupyterLab and notebook
    jupyter_parts = []
    try:
        jupyter_parts.append(f"Lab {importlib.metadata.version('jupyterlab')}")
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        jupyter_parts.append(f"notebook {importlib.metadata.version('notebook')}")
    except importlib.metadata.PackageNotFoundError:
        pass
    info["jupyter_version"] = ", ".join(jupyter_parts) if jupyter_parts else "not installed"

    # anywidget version
    try:
        info["anywidget_version"] = importlib.metadata.version("anywidget")
    except importlib.metadata.PackageNotFoundError:
        info["anywidget_version"] = "not installed"

    # Platform
    system = platform.system()
    if system == "Darwin":
        system = "macOS"
    release = platform.mac_ver()[0] if platform.system() == "Darwin" else platform.release()
    info["platform_info"] = f"{system} {release} {platform.machine()}"

    # Disk space — check cwd and common data directories
    cwd = os.getcwd()
    try:
        usage = shutil.disk_usage(cwd)
        info["disk_total_gb"] = round(usage.total / (1024**3), 1)
        info["disk_free_gb"] = round(usage.free / (1024**3), 1)
    except OSError:
        info["disk_total_gb"] = "unknown"
        info["disk_free_gb"] = "unknown"

    # Print formatted output
    compute_label = info["compute_device"]
    if info["gpu_name"]:
        compute_label = f"{info['compute_device']} ({info['gpu_name']})"

    ram_label = "unknown"
    if isinstance(info["system_ram_gb"], (int, float)):
        ram_label = f"{info['system_ram_gb']} GB"
        if isinstance(info["system_ram_available_gb"], (int, float)):
            ram_label += f" ({info['system_ram_available_gb']} GB available)"

    vram_label = str(info["vram_gb"])
    if isinstance(info["vram_gb"], (int, float)):
        vram_label = f"{info['vram_gb']} GB"
        if isinstance(info["vram_available_gb"], (int, float)):
            vram_label += f" ({info['vram_available_gb']} GB available)"
    elif info["vram_gb"] == "shared":
        vram_label = "shared (unified memory)"

    disk_label = "unknown"
    if isinstance(info["disk_total_gb"], (int, float)):
        disk_label = f"{info['disk_free_gb']} GB free / {info['disk_total_gb']} GB"

    rows = [
        ("quantem.widget", info["quantem_widget_version"]),
        ("quantem", info["quantem_version"]),
        ("Python", info["python_version"]),
        ("NumPy", info["numpy_version"]),
        ("PyTorch", info["pytorch_version"]),
        ("Compute", compute_label),
        ("System RAM", ram_label),
        ("VRAM", vram_label),
        ("Disk", disk_label),
        ("Jupyter", info["jupyter_version"]),
        ("anywidget", info["anywidget_version"]),
        ("Platform", info["platform_info"]),
    ]
    width = max(len(r[0]) for r in rows)
    for label, value in rows:
        print(f"{label:<{width}}  {value}")

    return _ProfileResult(info)
