import os as _os
import warnings as _warnings
from importlib.metadata import PackageNotFoundError, version

# Silence two noisy-but-harmless warnings at import, BEFORE anything imports cupy
# or huggingface_hub (the (?s) flag is required - both messages start with a
# newline, so a plain `.*` would not match):
#   - cupy "multiple CuPy packages" (cuda12x + cuda13x): on a host/Colab runtime
#     that already shipped a cupy, ours is redundant; we no longer pin one, but a
#     runtime contaminated by an older release can still have two. The check is
#     advisory; the working cupy still loads.
#   - huggingface_hub "HF_TOKEN secret does not exist": our datasets are PUBLIC,
#     no token needed. The nudge wrongly implies auth is required.
_os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
_warnings.filterwarnings("ignore", message=r"(?s).*multiple CuPy packages.*")
_warnings.filterwarnings("ignore", message=r"(?s).*HF_TOKEN.*")

from quantem.widget.show1d import Show1D
from quantem.widget.show2d import Show2D
from quantem.widget.show3d import Show3D
from quantem.widget.show3dslices import Show3DSlices
from quantem.widget.showeds import ShowEDS, SpectrumImage, bin_spectrum_image, load_eds, load_emd_spectrum_image
from quantem.widget.show4dstem_factory import Show4DSTEM
from quantem.widget.showdiffraction import ShowDiffraction
from quantem.widget.showfolder import ShowFolder, prebuild_showfolder_cache, show_folder
from quantem.widget.io import load, load_scan_region, read_gif, read_image, read_image_stack, read_images
from . import movie
from quantem.widget.paths import first_existing
from quantem.widget.backend import detect_backend, resolve_backend
from quantem.widget.gpu import gpu_info
from quantem.widget.detector import detect_bf_radius, dp_mean, virtual_image
from quantem.widget.folder_picker import FolderPicker, pick_folder
from quantem.widget.multidataset_mps import load_4dstem_macbook
from quantem.widget.export import (
    HTML_EXPORT_TRAITS,
    SupportsFrontendHtmlExport,
    SupportsHtmlExport,
    supports_html_export,
)
from quantem.widget.dpc import idpc, com
from quantem.widget.info import device_info
from quantem.widget.detector import bf, adf, df
from quantem.widget._timing import (
    WidgetProfile,
    format_timing_table,
    format_widget_render_timing,
    profile_widget,
    widget_timing_report,
)


try:
    __version__ = version("quantem.widget")
except PackageNotFoundError:
    # Source-tree imports (e.g. `PYTHONPATH=src pytest`) skip pip install.
    __version__ = "0.0.0+local"


def profile() -> None:
    """Print the runtime environment: quantem + quantem.widget versions, where quantem is
    imported from, the torch device, and Python. Call it at the top of a notebook so the
    reader (and any bug report or shared HTML) records exactly what produced the results -
    versions drift, and "which build / which branch" is the first question every time."""
    import platform
    print(f"quantem.widget  {__version__}")
    try:
        import quantem
        print(f"quantem         {getattr(quantem, '__version__', '?')}")
        print(f"  loaded from   {quantem.__file__}")
    except ImportError:
        print("quantem         (not importable)")
    try:
        import torch
        if torch.cuda.is_available():
            dev = f"cuda ({torch.cuda.get_device_name(0)})"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dev = "mps (Apple)"
        else:
            dev = "cpu"
        print(f"torch           {torch.__version__}  device={dev}")
        if torch.cuda.is_available():
            # Show EVERY visible GPU + how many are visible, so the reader knows up front
            # whether the next merge / recon fits and on WHICH card - no surprise mid-run.
            # torch live-vs-reserved is the leak signal: if "live" climbs across repeated
            # calls, refs are still pinned (del them, then free_gpu() returns the pool).
            import os
            n = torch.cuda.device_count()
            print(f"GPUs            {n} visible (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')})")
            for i in range(n):
                free, total = torch.cuda.mem_get_info(i)
                print(f"  GPU{i}          {(total - free) / 1e9:5.1f} used / {total / 1e9:.0f} GB  ({free / 1e9:.0f} free)  <- run free_gpu() if low")
            print(f"  torch pool    {torch.cuda.memory_allocated() / 1e9:.1f} live / {torch.cuda.memory_reserved() / 1e9:.1f} reserved GB")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            cur = torch.mps.current_allocated_memory() / 1e9 if hasattr(torch.mps, "current_allocated_memory") else 0.0
            drv = torch.mps.driver_allocated_memory() / 1e9 if hasattr(torch.mps, "driver_allocated_memory") else 0.0
            print(f"VRAM (MPS)      {cur:.1f} live / {drv:.1f} driver GB")
    except ImportError:
        print("torch           (not importable)")
    print(f"python          {platform.python_version()}")


def free_gpu(verbose: bool = True) -> float:
    """Release cached GPU memory back to the driver, on CUDA (torch + cupy pools) or Apple
    MPS. Call AFTER ``del``-ing your big objects (the merged 4D stack, the widget): this
    hands the allocator's cached-but-unused blocks back to the device - it cannot drop
    references you still hold, so ``del`` first. Returns GB released.

    Why it is needed: torch (and cupy) keep a caching allocator. After ``del`` of a 38 GB
    merge the pool still PINS those blocks - ``nvidia-smi`` shows them used and the next
    load OOMs. ``empty_cache`` + cupy ``free_all_blocks`` return them. MPS caches the same
    way; ``torch.mps.empty_cache`` is the equivalent. Backend is auto-detected.

    >>> del widget, merged          # drop every reference first
    >>> free_gpu()
    freed 38.6 GB  (40.5 -> 1.8)
    """
    import gc
    gc.collect()
    try:
        import torch
    except ImportError:
        if verbose:
            print("torch not importable - nothing to free")
        return 0.0
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        used = lambda: sum(total - free for free, total in (torch.cuda.mem_get_info(i) for i in range(n))) / 1e9
        before = used()
        try:
            import cupy as cp
        except ImportError:
            cp = None
        for i in range(n):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
            if cp is not None:
                cp.cuda.Device(i).use()
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
        if cp is not None:
            cp.cuda.Device(0).use()   # leave the default device on GPU0 so the next load lands where it expects
        after = used()
        if verbose:
            for i in range(n):
                free, total = torch.cuda.mem_get_info(i)
                print(f"GPU{i}: {(total - free) / 1e9:5.1f} GB used  ({free / 1e9:.0f} GB free)")
        return before - after
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if mps:
        cur = torch.mps.current_allocated_memory() if hasattr(torch.mps, "current_allocated_memory") else 0
        torch.mps.empty_cache()
        post = torch.mps.current_allocated_memory() if hasattr(torch.mps, "current_allocated_memory") else 0
        if verbose:
            print(f"freed {(cur - post) / 1e9:.1f} GB (MPS)")
        return (cur - post) / 1e9
    if verbose:
        print("no GPU - nothing to free")
    return 0.0


__all__ = [
    "Show1D",
    "Show2D",
    "Show3D",
    "Show3DSlices",
    "Show4DSTEM",
    "ShowDiffraction",
    "ShowEDS",
    "ShowFolder",
    "prebuild_showfolder_cache",
    "SpectrumImage",
    "bin_spectrum_image",
    "load_eds",
    "load_emd_spectrum_image",
    "load",
    "load_scan_region",
    "show_folder",
    "read_gif",
    "read_image",
    "read_image_stack",
    "read_images",
    "movie",
    "HTML_EXPORT_TRAITS",
    "SupportsFrontendHtmlExport",
    "SupportsHtmlExport",
    "supports_html_export",
    "idpc",
    "com",
    "device_info",
    "bf",
    "adf",
    "df",
    "profile",
    "free_gpu",
]
