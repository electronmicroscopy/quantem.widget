"""Where widget computations run: one ``device=`` rule for every widget, and the GPU notice.

``resolve_device`` gives ``device=`` the same meaning everywhere, and the same
strings as ``quantem.gpu.device.resolve_device``: ``"auto"`` (CUDA, then Apple
MPS, then CPU), ``"cuda"``, ``"cuda:N"``, ``"mps"`` or ``"cpu"``. Data that is
already a torch tensor stays where it is unless an explicit device moves it.
Every choice the user did not spell out is printed once, so CPU is never picked
silently on a machine that has a GPU.

``gpu_notice`` names, once per process, a GPU this Python cannot use for the
quantem.gpu path and the pip command that fixes it: ``quantem.widget[cuda]`` for
an NVIDIA GPU, ``quantem.widget[mps]`` for an Apple-silicon GPU. An Intel Mac
has no quantem.gpu path; the line says so and names the torch device used.
"""

import importlib.util
import os
import platform
import shutil
import sys

import torch

INSTALL_CUDA = 'pip install "quantem.widget[cuda]"'
INSTALL_MPS = 'pip install "quantem.widget[mps]"'
NO_GPU_PATH_INTEL_MAC = (
    "quantem.gpu, which keeps files encoded on a GPU, does not run on Intel Macs: it needs torch 2.3 or newer, "
    "and PyTorch publishes torch 2.2 at most for them"
)
NO_GPU_PATH = (
    "quantem.gpu, which keeps files encoded on a GPU, runs with CUDA on an NVIDIA GPU or with Metal on an Apple "
    "silicon Mac, and neither is available to this Python"
)
# Each line prints once per process.
_printed: set[str] = set()


def resolve_device(device: str | torch.device | None = "auto", data=None) -> str:
    """Resolve ``device=`` to the device a widget computes on, as a torch device string.

    The strings, order and announcement are those of
    ``quantem.gpu.device.resolve_device``, in torch alone so they hold without
    quantem.gpu. ``"auto"`` (or ``None``) picks ``"cuda:0"``, then ``"mps"``,
    then ``"cpu"`` and prints one line naming the choice, once per process.
    ``"cuda"``, ``"cuda:N"``, ``"mps"`` and ``"cpu"`` select that device quietly
    and raise when it is not available. A torch tensor in ``data`` keeps its
    device under ``"auto"``; an explicit device that differs moves it, and that
    move is printed once.

    Parameters
    ----------
    device : str, torch.device or None, default "auto"
        ``"auto"``, ``"cuda"``, ``"cuda:N"``, ``"mps"`` or ``"cpu"``.
    data : optional
        The widget input. Only a torch tensor changes the outcome.

    Returns
    -------
    str
        ``"cuda:N"``, ``"mps"`` or ``"cpu"``; ``torch.device`` accepts it.

    Examples
    --------
    >>> resolve_device("cpu")
    'cpu'
    """
    requested = "auto" if device is None else str(device).strip().lower()
    if isinstance(data, torch.Tensor) and requested in ("", "auto"):
        return device_name(data.device)
    if requested in ("", "auto"):
        gpu_notice()
        resolved = "cuda:0" if torch.cuda.is_available() else "mps" if mps_available() else "cpu"
        print_once(f'quantem.widget: device="auto" selected {resolved}.')
        return resolved
    if requested == "cpu":
        resolved = "cpu"
    elif requested == "mps":
        if not mps_available():
            raise RuntimeError('MPS device is unavailable; use device="auto" for automatic selection.')
        resolved = "mps"
    elif requested == "cuda" or requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA device is unavailable; use device="auto" for automatic selection.')
        try:
            index = 0 if requested == "cuda" else int(requested.removeprefix("cuda:"))
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device {device!r}; use 'cuda' or 'cuda:N', for example 'cuda:1'.") from exc
        count = torch.cuda.device_count()
        if not 0 <= index < count:
            raise ValueError(
                f"CUDA device {device!r} is unavailable: {count} CUDA device(s) are visible. "
                f"Choose an index from 0 to {count - 1}."
            )
        resolved = f"cuda:{index}"
    else:
        raise ValueError(f"Unknown device {device!r}; use 'auto', 'cuda:N', 'mps', or 'cpu'.")
    if isinstance(data, torch.Tensor) and device_name(data.device) != resolved:
        print_once(f'quantem.widget: device="{requested}" moves the data from {device_name(data.device)} to {resolved}.')
    return resolved


def device_name(device: torch.device) -> str:
    """``"cuda:N"``, ``"mps"`` or ``"cpu"`` for a torch device; ``cuda`` without an index is ``cuda:0``."""
    return f"cuda:{device.index or 0}" if device.type == "cuda" else device.type


def mps_available() -> bool:
    """Whether torch can run on an Apple GPU (Apple silicon, or an AMD GPU of an Intel Mac)."""
    return bool(torch.backends.mps.is_available())


def gpu_path_hint() -> str | None:
    """The ``pip`` command that adds the quantem.gpu path on this machine, or None when it has no GPU for it.

    ``[cuda]`` only when an NVIDIA driver is present or torch already runs on
    CUDA, ``[mps]`` only on an Apple silicon Mac. On other machines (an Intel
    Mac, Windows on ARM, any machine without an NVIDIA GPU) neither extra
    would help, so callers name what does work instead and ``no_gpu_path``
    says why; no message offers a route that cannot work there.
    """
    system, machine = host_platform()
    if system == "darwin":
        return INSTALL_MPS if machine == "arm64" else None
    return INSTALL_CUDA if nvidia_present() or torch.cuda.is_available() else None


def no_gpu_path() -> str:
    """Why this machine has no quantem.gpu path, for the messages where ``gpu_path_hint`` is None."""
    system, machine = host_platform()
    return NO_GPU_PATH_INTEL_MAC if system == "darwin" and machine != "arm64" else NO_GPU_PATH


def gpu_notice() -> str | None:
    """Print, once per process, a GPU the quantem.gpu path cannot use here and how to fix it.

    The widget computes on any torch device by itself; quantem.gpu adds encoded
    acquisitions and GPU file loading. On a machine with an NVIDIA or Apple GPU
    but without quantem.gpu (or without its CuPy, Metal or MLX runtime), files
    are read densely on the CPU instead. Without this line that slower path
    would be taken silently. Returns the line, or None when nothing is missing.
    """
    line = gpu_notice_text()
    print_once(line)
    return line


def gpu_notice_text() -> str | None:
    """The notice line for this machine, or None. Split from ``gpu_notice`` so tests read it without printing."""
    system, machine = host_platform()
    if system == "darwin" and machine != "arm64":
        device = "torch MPS on its GPU" if mps_available() else "the CPU"
        return (
            f"quantem.widget: on this Intel Mac the widgets compute with {device}, and Show4DSTEM reads files "
            f"on the CPU and holds every count in memory. {NO_GPU_PATH_INTEL_MAC}."
        )
    if module_installed("quantem.gpu"):
        from quantem.widget.adapters import gpu as gpu_adapter

        if gpu_adapter.gpu_runtime_notice is not None:
            # quantem.gpu names its own missing runtime (CuPy, Metal, MLX)
            return gpu_adapter.gpu_runtime_notice()
    if system == "darwin":
        missing = [name for name in ("quantem.gpu", "Metal", "mlx") if not module_installed(name)]
        gpu, extra = "Apple GPU", "mps"
    else:
        if not nvidia_present():
            return None
        if not torch.cuda.is_available() and torch.version.cuda is None:
            return (
                "quantem.widget: NVIDIA GPU found but this torch build has no CUDA, so widget compute runs on CPU. "
                "Install a CUDA build of torch (https://pytorch.org/get-started/locally/), then "
                f"{INSTALL_CUDA} for the GPU path."
            )
        missing = [name for name in ("quantem.gpu", "cupy") if not module_installed(name)]
        gpu, extra = "NVIDIA GPU", "cuda"
    if not missing:
        return None
    return (
        f"quantem.widget: {gpu} found but quantem.gpu[{extra}] is not installed ({', '.join(missing)} missing); "
        f"Show4DSTEM file loading runs on CPU. For the GPU path: pip install \"quantem.widget[{extra}]\""
    )


def host_platform() -> tuple[str, str]:
    """``(sys.platform, machine)``, for example ``("darwin", "arm64")`` or ``("linux", "x86_64")``."""
    return sys.platform, platform.machine()


def nvidia_present() -> bool:
    """Whether an NVIDIA driver is installed: ``nvidia-smi`` on the path or a loaded kernel module."""
    return shutil.which("nvidia-smi") is not None or os.path.exists("/proc/driver/nvidia")


def module_installed(name: str) -> bool:
    """Whether ``name`` can be imported, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        # a parent package that is missing or broken means the module is not usable
        return False


def print_once(line: str | None) -> None:
    """Print ``line`` the first time it occurs in this process."""
    if line is not None and line not in _printed:
        _printed.add(line)
        print(line)
