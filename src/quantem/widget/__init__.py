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
    "DiffractionSim": ("quantem.widget.diffraction_sim", "DiffractionSim"),
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
    "ShowPDF": ("quantem.widget.showpdf", "ShowPDF"),
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
    """Return cached GPU memory to the driver after you ``del`` your big objects; returns the GB released.

    With quantem.gpu installed (``quantem.widget[cuda]`` or ``[mps]``) this is
    ``quantem.gpu.device.release_cached_memory``: it collects garbage, empties the Torch, CuPy and MLX
    caches on every GPU the process has opened and returns what this process handed back to the driver.
    Without quantem.gpu (``[cpu]``, an Intel Mac) the widgets compute with Torch alone, so garbage is
    collected and Torch's cache emptied, and the result is the drop in the memory Torch holds from the
    driver: its reserved bytes on every CUDA GPU, or the Metal driver's total.

    Why it is needed: the allocators keep freed blocks for reuse, so after ``del`` the memory still shows
    as used and the next load runs out of it. It cannot drop references you still hold, so ``del`` first.

    >>> del widget, merged          # drop every reference first
    >>> free_gpu()
    GPU 0: released 2.1 GB, 32.0 GB in use, 69.9 GB free
    2.147483648
    """
    from quantem.widget.adapters import gpu as gpu_adapter

    if gpu_adapter.available():
        return gpu_adapter.gpu_release_cached_memory(verbose=verbose)
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        devices = range(torch.cuda.device_count())
        reserved = sum(torch.cuda.memory_reserved(index) for index in devices)
        torch.cuda.empty_cache()
        freed = (reserved - sum(torch.cuda.memory_reserved(index) for index in devices)) / 1e9
    elif torch.backends.mps.is_available():
        # the Metal driver's total holds the cached blocks empty_cache returns; the tensor
        # allocation counter never includes them, so it always read 0 GB freed
        allocated = torch.mps.driver_allocated_memory()
        torch.mps.empty_cache()
        freed = (allocated - torch.mps.driver_allocated_memory()) / 1e9
    else:
        freed = 0.0
    if verbose:
        print(f"freed {freed:.1f} GB (Torch cache)")
    return freed


__all__ = [
    "ChooseLattice",
    "DiffractionSim",
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
    "ShowPDF",
    "ShowPtycho",
    "read_4dstem",
    "read_image",
    "read_image_stack",
    "read_images",
    "device_info",
    "profile",
    "free_gpu",
]
