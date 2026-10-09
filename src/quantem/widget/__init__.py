import os
import warnings
from importlib import import_module
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
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
warnings.filterwarnings("ignore", message=r"(?s).*multiple CuPy packages.*")
warnings.filterwarnings("ignore", message=r"(?s).*HF_TOKEN.*")

_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    "ChooseLattice": ("quantem.widget.choose_lattice", "ChooseLattice"),
    "ShowCIF": ("quantem.widget.showcif", "ShowCIF"),
    "Show1D": ("quantem.widget.show1d", "Show1D"),
    "Plot2D": ("quantem.widget.plot2d", "Plot2D"),
    "Show2D": ("quantem.widget.show2d", "Show2D"),
    "Show3D": ("quantem.widget.show3d", "Show3D"),
    "Show3DSlices": ("quantem.widget.show3dslices", "Show3DSlices"),
    "Show4DSTEM": ("quantem.widget.show4dstem", "Show4DSTEM"),
    "ShowDiffraction": ("quantem.widget.showdiffraction", "ShowDiffraction"),
    "Phase": ("quantem.widget.showdiffraction", "Phase"),
    "library_phase": ("quantem.widget.showdiffraction", "library_phase"),
    "ShowPtycho": ("quantem.widget.showptycho", "ShowPtycho"),
    "read_4dstem": ("quantem.widget.show4dstem.reader", "read_4dstem"),
    "read_image": ("quantem.widget.io.image", "read_image"),
    "read_image_stack": ("quantem.widget.io.image", "read_image_stack"),
    "read_images": ("quantem.widget.io.image", "read_images"),
    "gpu_info": ("quantem.widget.gpu", "gpu_info"),
    "FolderPicker": ("quantem.widget.folder_picker", "FolderPicker"),
    "pick_folder": ("quantem.widget.folder_picker", "pick_folder"),
    "device_info": ("quantem.widget.info", "device_info"),
    "profile": ("quantem.widget.info", "profile"),
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
    module = import_module(module_name)
    value = module if attribute_name is None else module.__dict__[attribute_name]
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include explicit lazy exports in interactive discovery.

    The stdlib modules this file imports for its own setup (``os``,
    ``warnings``, ``import_module``) are left out, so tab completion on
    ``quantem.widget.`` lists quantem names only.
    """

    return sorted((set(globals()) - {"os", "warnings", "import_module"}) | set(_LAZY_EXPORTS))


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

    import torch

    gc.collect()
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()

        def used_gb() -> float:
            """Memory in use on every visible GPU, in GB, as the driver reports it."""
            return sum(total - free for free, total in map(torch.cuda.mem_get_info, range(device_count))) / 1e9

        before = used_gb()
        try:
            import cupy as cp
        except ImportError:
            cp = None
        for index in range(device_count):
            with torch.cuda.device(index):
                torch.cuda.empty_cache()
            if cp is not None:
                cp.cuda.Device(index).use()
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
        if cp is not None:
            cp.cuda.Device(0).use()   # leave the default device on GPU0 so the next load lands where it expects
        after = used_gb()
        if verbose:
            for index in range(device_count):
                free, total = torch.cuda.mem_get_info(index)
                print(f"GPU{index}: {(total - free) / 1e9:5.1f} GB used  ({free / 1e9:.0f} GB free)")
        return before - after
    if torch.backends.mps.is_available():
        allocated_before = torch.mps.current_allocated_memory()
        torch.mps.empty_cache()
        allocated_after = torch.mps.current_allocated_memory()
        if verbose:
            print(f"freed {(allocated_before - allocated_after) / 1e9:.1f} GB (MPS)")
        return (allocated_before - allocated_after) / 1e9
    if verbose:
        print("no GPU - nothing to free")
    return 0.0


__all__ = [
    "ChooseLattice",
    "Show1D",
    "Plot2D",
    "Show2D",
    "Show3D",
    "Show3DSlices",
    "Show4DSTEM",
    "ShowCIF",
    "Phase",
    "ShowDiffraction",
    "library_phase",
    "ShowPtycho",
    "read_4dstem",
    "read_image",
    "read_image_stack",
    "read_images",
    "device_info",
    "profile",
    "free_gpu",
]
