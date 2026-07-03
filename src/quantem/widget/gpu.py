"""
GPU VRAM profiling utilities.

Without visibility into GPU memory usage, it's hard to know whether
a reconstruction is close to running out of VRAM, or why a second
reconstruction OOMs when the first succeeded. These utilities make
the invisible visible.

Public API
----------
gpu_info : Print a snapshot of GPU VRAM and CuPy pool usage.
free_gpu : Free all CuPy memory pool blocks to release GPU VRAM.

Examples
--------
>>> from quantem.widget.utils.gpu_info import gpu_info, free_gpu
>>> gpu_info()
NVIDIA RTX PRO 6000 (95.0 GB)
  VRAM: 82.6 GB available of 95.0 GB
  CuPy pool: 8.2 GB in use, 3.1 GB cached
>>> free_gpu()
>>> gpu_info()
  CuPy pool: 0 bytes in use, 0 bytes cached
"""

import gc

# cupy imported lazily inside gpu_info/free_gpu so a widget install without a
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


def free_gpu() -> None:
    """Free all CuPy memory pool blocks to release GPU VRAM.

    Clears Python references, CuPy FFT plan cache, IO decompressor buffers,
    and both device and pinned memory pools.
    """
    import cupy as cp
    gc.collect()
    # Clear CuPy FFT plan cache (holds GPU workspace buffers)
    try:
        cp.fft.config.get_plan_cache().clear()
    except (AttributeError, RuntimeError):
        pass
    # Clear IO decompressor GPU buffers
    try:
        from quantem.live.io import _clear_memory
        _clear_memory()
    except (AttributeError, RuntimeError):
        pass
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def vram_status() -> str:
    """Human-readable VRAM state for the current CuPy device.

    Used by the live batch loop after a failure to surface whether cleanup
    actually recovered memory - previously cleanup warnings scrolled off and
    the next file OOM'd with no context (#130).
    """
    import cupy as cp
    try:
        free_b, total_b = cp.cuda.runtime.memGetInfo()
        used_b = total_b - free_b
        return (f"GPU {cp.cuda.runtime.getDevice()}: "
                f"{_format_bytes(used_b)} used / {_format_bytes(free_b)} free "
                f"(of {_format_bytes(total_b)})")
    except (RuntimeError, AttributeError, ImportError) as e:
        return f"GPU status unavailable: {e}"
