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

from quantem.widget.show2d import Show2D
from quantem.widget.show3d import Show3D
from quantem.widget.show3dslices import Show3DSlices
from quantem.widget.show4dstem import Show4DSTEM as _Show4DSTEMBase
from quantem.widget.io import load
from quantem.widget.dpc import idpc, com
from quantem.widget.info import device_info
from quantem.widget.detector import bf, adf, df


def Show4DSTEM(data, **kwargs):
    """Open a 4D-STEM viewer over ``load(...)`` output, on any backend.

    Canonical examples::

        from quantem.widget import load, Show4DSTEM

        Show4DSTEM(load("a.h5"))                         # auto: CUDA / MPS / CPU
        Show4DSTEM(load("a.h5", backend="mps"))          # explicit Apple Metal load
        Show4DSTEM(load(["a.h5", "b.h5"], det_bin=4))    # many datasets, one slider
        Show4DSTEM(load("a.h5"), backend="web")          # browser WebGPU compute

        w = Show4DSTEM(load("a.h5"), backend="web", offline_codec="bslz4",
                       data_url="show4dstem-data")
        w.export_html("show4dstem.html")

    Dispatch is automatic from what ``load`` returns:
      - MacBook (MPS) single -> the raw-Metal real-time viewer (full-res CBED +
        bin2 virtual-image fast path). torch.mps is not fast enough on Apple
        Silicon, which is why the dedicated Metal path exists.
      - MacBook (MPS) many -> a lazy handle; dataset 0 shows now, 1..N fill in the
        background behind the dataset slider.
      - CUDA / CPU single or many -> the universal torch viewer (a 5D array gives
        an instant dataset slider on big-VRAM boxes).

    Web aliases ``backend="browser"``, ``backend="webgpu"``, and
    ``offline=True`` are accepted for compatibility. Large backendless exports
    should use ``offline_codec="bslz4"`` plus a ``data_url`` companion directory
    instead of embedding the full stack in the HTML.
    """
    # MacBook lazy multi-dataset handle -> build the viewer + start background fill.
    from quantem.widget.multidataset_mps import LazyMacbookDatasets
    if isinstance(data, LazyMacbookDatasets):
        return data.build_viewer(**kwargs)
    # MacBook single raw-Metal load (LoadResult wrapping MPSChunked4DSTEM, or an
    # already-wrapped ChunkedFrames) -> the specialized Metal viewer with sampling
    # pulled from metadata. CUDA/CPU loads fall through to the universal viewer.
    is_loadresult = hasattr(data, "_fields") and "data" in getattr(data, "_fields", ())
    payload = data.data if is_loadresult else data
    # Route to the raw-Metal viewer only for MPS data: an MPSChunked4DSTEM (has
    # ``chunks``) is always MPS, and an ``_is_gpu_frames`` stack goes to Metal only
    # when its frames live on MPS. A CUDA ``_is_gpu_frames`` stack (e.g. a sharded
    # multi-GPU Dataset5dstem - 7 tilts across 2 cards) is NOT MPS and falls through
    # to the universal torch viewer, which has its own no-gather frame path.
    _payload_dev = str(getattr(payload, "device", ""))
    _is_mps_frames = getattr(payload, "_is_gpu_frames", False) and "mps" in _payload_dev
    if hasattr(payload, "chunks") or _is_mps_frames:
        # Show4DSTEM_MACBOOK = sampling-aware MPS viewer factory. The factory
        # itself doesn't warn (it's the natural API for Mac users); only direct
        # imports of show_4dstem_mps / load_4dstem_mps warn.
        from quantem.widget.show4dstem_mps import Show4DSTEM_MACBOOK
        return Show4DSTEM_MACBOOK(payload, **kwargs)
    # CUDA/CPU multi-dataset stack (a list load gives a 5D array): label the slider
    # "Dataset" and name each slot from the source files, so it reads as a list of
    # datasets rather than a generic frame axis.
    if is_loadresult and getattr(payload, "ndim", 0) == 5:
        meta = getattr(data, "metadata", {}) or {}
        kwargs.setdefault("frame_dim_label", "Dataset")
        names = meta.get("file_names")
        if names is not None:
            kwargs.setdefault("frame_labels", list(names))
    return _Show4DSTEMBase(data, **kwargs)


try:
    __version__ = version("quantem.widget")
except PackageNotFoundError:
    # Source-tree imports (e.g. `PYTHONPATH=src pytest`) skip pip install.
    __version__ = "0.0.0+local"

__all__ = ["Show2D", "Show3D", "Show3DSlices", "Show4DSTEM", "load", "idpc", "com", "device_info", "bf", "adf", "df"]
