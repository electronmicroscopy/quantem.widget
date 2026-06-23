"""One-call memory profiler: per-GPU VRAM + system RAM (+ optional disk staging).

Run ``io.memory()`` anytime to see what you have and what is resident. VRAM is
shown first (it is the scarce tier), then RAM. Disk staging (HF cache, /tmp) is
off by default - pass ``disk=True`` when you care about download/upload room.

    io.memory()             # VRAM + RAM (default)
    io.memory(ram=False)    # VRAM only
    io.memory(vram=False)   # RAM only
    io.memory(disk=True)    # also show disk staging
    io.memory(dset)         # + the dataset's per-device VRAM footprint

Prints a readable block and returns a dict for programmatic use.
"""
from __future__ import annotations


def memory(dataset=None, *, vram: bool = True, ram: bool = True,
           disk: bool = False, disk_paths=None, verbose: bool = True) -> dict:
    """VRAM (first) + RAM (+ optional disk); optional Dataset5dstem footprint."""
    report: dict = {}

    if vram:
        import torch  # noqa: PLC0415
        cp = None
        if torch.cuda.is_available():
            try:
                import cupy as cp  # noqa: PLC0415
            except ImportError:
                cp = None
            report["gpu"] = []
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                torch_alloc = torch.cuda.memory_allocated(i)
                cupy_used = 0
                if cp is not None:
                    with cp.cuda.Device(i):
                        cupy_used = cp.get_default_memory_pool().used_bytes()
                report["gpu"].append({
                    "id": i, "name": torch.cuda.get_device_name(i),
                    "total_gb": total / 1e9, "used_gb": (total - free) / 1e9, "free_gb": free / 1e9,
                    "torch_alloc_gb": torch_alloc / 1e9, "cupy_pool_gb": cupy_used / 1e9,
                })
                if verbose:
                    print(f"VRAM GPU{i}  {(total - free) / 1e9:6.1f} / {total / 1e9:6.1f} GB used   "
                          f"({free / 1e9:.1f} free)   [torch {torch_alloc / 1e9:.1f}, cupy {cupy_used / 1e9:.1f}]   "
                          f"{torch.cuda.get_device_name(i)}")

    if ram:
        import psutil  # noqa: PLC0415
        vm = psutil.virtual_memory()
        report["ram_gb"] = {"total": vm.total / 1e9, "used": vm.used / 1e9, "free": vm.available / 1e9}
        if verbose:
            print(f"RAM        {vm.used / 1e9:6.1f} / {vm.total / 1e9:6.1f} GB used   ({vm.available / 1e9:.1f} free)")

    if disk:
        import shutil  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415
        if disk_paths is None:
            disk_paths = [Path.home() / ".cache/huggingface", "/tmp"]
        report["disk_gb"] = {}
        for path in disk_paths:
            p = Path(path)
            if not p.exists():
                continue
            usage = shutil.disk_usage(p)
            report["disk_gb"][str(p)] = {"total": usage.total / 1e9, "used": usage.used / 1e9, "free": usage.free / 1e9}
            if verbose:
                print(f"DISK       {usage.used / 1e9:6.1f} / {usage.total / 1e9:6.1f} GB used   "
                      f"({usage.free / 1e9:.1f} free)   {p}")

    if dataset is not None and hasattr(dataset, "summary"):
        if verbose:
            print("--- dataset footprint (per device) ---")
        report["dataset_gb_per_device"] = dataset.summary()
    return report
