"""Human-readable device / version info for notebooks - which GPU, which machine.

``device_info()`` prints (and returns) a tidy one-block summary so a shared
notebook records what it ran on: widget version, date, compute backend (Apple
Metal / CUDA / CPU), the GPU or Mac chip, and memory. Useful at the top of any
demo so results are reproducible.
"""
from __future__ import annotations

import datetime
import subprocess


def _mac_chip_mem():
    def _sysctl(key):
        try:
            return subprocess.run(["sysctl", "-n", key], capture_output=True,
                                  text=True, timeout=3).stdout.strip()
        except Exception:
            return ""
    chip = _sysctl("machdep.cpu.brand_string") or "Apple Silicon"
    mem = _sysctl("hw.memsize")
    gb = f"{int(mem) // (1024 ** 3)} GB" if mem.isdigit() else "?"
    return chip, gb


def device_info(verbose: bool = True) -> dict:
    """Return (and by default print) version + backend + hardware for this machine."""
    import quantem.widget
    from quantem.gpu.device import detect

    backend = detect()
    info = {
        "widget_version": quantem.widget.__version__,
        "date": str(datetime.date.today()),
        "backend": backend,
    }
    if backend == "mps":
        chip, mem = _mac_chip_mem()
        info["device"] = f"Apple Metal (MPS) - {chip}, {mem} unified memory"
    elif backend == "cuda":
        try:
            import torch
            info["device"] = f"CUDA - {torch.cuda.get_device_name(0)}"
        except Exception:
            info["device"] = "CUDA"
    if verbose:
        print(f"quantem.widget {info['widget_version']}   |   {info['date']}")
        print(f"compute: {info['device']}")
    return info
