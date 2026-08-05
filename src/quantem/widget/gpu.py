"""
GPU VRAM profiling utilities.

Without visibility into GPU memory usage, it's hard to know whether
a reconstruction is close to running out of VRAM, or why a second
reconstruction OOMs when the first succeeded. These utilities make
the invisible visible.

Public API
----------
gpu_info : Print a snapshot of GPU VRAM and CuPy pool usage.
vram_status : Return a compact CUDA memory-status string for live error paths.

Examples
--------
>>> from quantem.widget import gpu_info
>>> gpu_info()
NVIDIA RTX PRO 6000 (95.0 GB)
  VRAM: 82.6 GB available of 95.0 GB
  CuPy pool: 8.2 GB in use, 3.1 GB cached

Use :func:`quantem.widget.free_gpu` for cross-backend cache release.
"""

# CuPy is imported lazily inside gpu_info so a widget install without a
# CUDA runtime (e.g. Mac, or a lightweight CI env) doesn't fail at import time.


def _format_bytes(n_bytes: int) -> str:
    """Format bytes as a human-readable string (GB, MB, or KB)."""
    if n_bytes >= 1 << 30:
        return f"{n_bytes / (1 << 30):.1f} GB"
    if n_bytes >= 1 << 20:
        return f"{n_bytes / (1 << 20):.1f} MB"
    return f"{n_bytes / (1 << 10):.1f} KB"


def gpu_info(device_id: int | None = None) -> None:
    """Print a snapshot of GPU VRAM and CuPy memory pool usage.

    Shows three layers of the GPU memory stack:
    1. **Physical VRAM** - total and used memory on the GPU hardware
    2. **CuPy memory pool** - software cache that holds freed GPU allocations
       for reuse (allocated = in-use + cached)
    3. **CuPy pinned pool** - page-locked host memory for fast CPU↔GPU transfers

    Parameters
    ----------
    device_id : int or None
        GPU device index. If None, uses the current device.
    """
    import cupy as cp
    if device_id is not None:
        dev = cp.cuda.Device(device_id)
    else:
        dev = cp.cuda.Device()

    with dev:
        # Physical VRAM
        vram_free, vram_total = cp.cuda.runtime.memGetInfo()
        # CuPy device memory pool
        pool = cp.get_default_memory_pool()
        pool_total = pool.total_bytes()
        pool_used = pool.used_bytes()
        pool_cached = pool_total - pool_used

        # GPU name
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        gpu_name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]

    vram_free_gb = vram_free / (1 << 30)
    vram_total_gb = vram_total / (1 << 30)

    print(f"{gpu_name} ({vram_total_gb:.1f} GB)")
    print(f"  VRAM: {vram_free_gb:.1f} GB available of {vram_total_gb:.1f} GB")
    print(f"  CuPy pool: {_format_bytes(pool_used)} in use, "
          f"{_format_bytes(pool_cached)} cached")


def vram_status() -> str:
    """Return a compact VRAM status for the current CUDA device."""
    try:
        import cupy as cp

        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        used_bytes = total_bytes - free_bytes
        return (
            f"GPU {cp.cuda.runtime.getDevice()}: "
            f"{_format_bytes(used_bytes)} used / {_format_bytes(free_bytes)} free "
            f"(of {_format_bytes(total_bytes)})"
        )
    except (RuntimeError, AttributeError, ImportError) as exc:
        return f"GPU status unavailable: {exc}"
