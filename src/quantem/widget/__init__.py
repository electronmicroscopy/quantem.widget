import os as _os
import warnings as _warnings
from importlib import import_module as _import_module
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

_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    "ChooseLattice": ("quantem.widget.choose_lattice", "ChooseLattice"),
    "Show1D": ("quantem.widget.show1d", "Show1D"),
    "Show2D": ("quantem.widget.show2d", "Show2D"),
    "Show3D": ("quantem.widget.show3d", "Show3D"),
    "Show3DSlices": ("quantem.widget.show3dslices", "Show3DSlices"),
    "Show4DSTEM": ("quantem.widget.show4dstem_factory", "Show4DSTEM"),
    "ShowDiffraction": ("quantem.widget.showdiffraction", "ShowDiffraction"),
    "ShowEDS": ("quantem.widget.showeds", "ShowEDS"),
    "ShowFolder": ("quantem.widget.showfolder", "ShowFolder"),
    "ShowPtycho": ("quantem.widget.showptycho", "ShowPtycho"),
    "PtychoCalibration": ("quantem.widget.showptycho", "PtychoCalibration"),
    "load_ptycho_calibration": (
        "quantem.widget.showptycho",
        "load_ptycho_calibration",
    ),
    "prebuild_showfolder_cache": (
        "quantem.widget.showfolder",
        "prebuild_showfolder_cache",
    ),
    "show_folder": ("quantem.widget.showfolder", "show_folder"),
    "SpectrumImage": ("quantem.widget.showeds", "SpectrumImage"),
    "bin_spectrum_image": ("quantem.widget.showeds", "bin_spectrum_image"),
    "load_eds": ("quantem.widget.showeds", "load_eds"),
    "load_emd_spectrum_image": (
        "quantem.widget.showeds",
        "load_emd_spectrum_image",
    ),
    "read_gif": ("quantem.widget.io.image", "read_gif"),
    "read_image": ("quantem.widget.io.image", "read_image"),
    "read_image_stack": ("quantem.widget.io.image", "read_image_stack"),
    "read_images": ("quantem.widget.io.image", "read_images"),
    "movie": ("quantem.widget.movie", None),
    "first_existing": ("quantem.widget.paths", "first_existing"),
    "gpu_info": ("quantem.widget.gpu", "gpu_info"),
    "FolderPicker": ("quantem.widget.folder_picker", "FolderPicker"),
    "pick_folder": ("quantem.widget.folder_picker", "pick_folder"),
    "HTML_EXPORT_TRAITS": ("quantem.widget.export", "HTML_EXPORT_TRAITS"),
    "SupportsFrontendHtmlExport": (
        "quantem.widget.export",
        "SupportsFrontendHtmlExport",
    ),
    "SupportsHtmlExport": ("quantem.widget.export", "SupportsHtmlExport"),
    "supports_html_export": ("quantem.widget.export", "supports_html_export"),
    "device_info": ("quantem.widget.info", "device_info"),
    "WidgetProfile": ("quantem.widget._timing", "WidgetProfile"),
    "format_timing_table": ("quantem.widget._timing", "format_timing_table"),
    "format_widget_render_timing": (
        "quantem.widget._timing",
        "format_widget_render_timing",
    ),
    "profile_widget": ("quantem.widget._timing", "profile_widget"),
    "widget_timing_report": ("quantem.widget._timing", "widget_timing_report"),
}


try:
    __version__ = version("quantem.widget")
except PackageNotFoundError:
    # Source-tree imports (e.g. `PYTHONPATH=src pytest`) skip pip install.
    __version__ = "0.0.0+local"


def __getattr__(name: str):
    """Load one explicitly declared public export on first use."""

    export = _LAZY_EXPORTS.get(name)
    if export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = export
    module = _import_module(module_name)
    value = module if attribute_name is None else module.__dict__[attribute_name]
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include explicit lazy exports in interactive discovery."""

    return sorted(set(globals()) | set(_LAZY_EXPORTS))


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
    "ChooseLattice",
    "Show1D",
    "Show2D",
    "Show3D",
    "Show3DSlices",
    "Show4DSTEM",
    "ShowDiffraction",
    "ShowEDS",
    "ShowFolder",
    "ShowPtycho",
    "PtychoCalibration",
    "load_ptycho_calibration",
    "prebuild_showfolder_cache",
    "SpectrumImage",
    "bin_spectrum_image",
    "load_eds",
    "load_emd_spectrum_image",
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
    "device_info",
    "profile",
    "free_gpu",
]
