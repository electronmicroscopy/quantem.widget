"""``ComputeBackend`` protocol — the formal contract every Show4DSTEM backend
implements.

This is the **backend** side of the backend-vs-backendless split. A backend
owns the 4D-STEM data on the Python side and exposes a fixed set of compute
primitives the widget calls. Backends today:

  TorchBackend       — universal: CUDA / MPS-binned / CPU via torch
  MetalRawBackend    — raw Metal for a 19.3 GB large-no-bin-class no-bin stack
                       where torch.MPS hits the >2^31-element buffer limit.
                       Also owns MPS lifecycle: fast_vi (bin2 sidecar), radial
                       cache, multi-dataset proxy.

Backendless paths (WebGPU online / offline) live elsewhere — there is no
Python-side data, just a browser-side render. See ``show4dstem.export_html``.

Method surface (every backend implements these exact shapes):

    backend.scan_shape   -> (rows, cols)
    backend.det_shape    -> (rows, cols)
    backend.n_frames     -> int
    backend.device       -> str / torch.device — "cuda" / "mps" / "cpu" etc.

    backend.frame(idx)                     -> np.ndarray (det_r, det_c)
    backend.masked_sum(det_mask)           -> np.ndarray (scan_r, scan_c) f32
    backend.mean_dp()                      -> np.ndarray (det_r, det_c) f32
    backend.reduce_frames(scan_idx, mode)  -> np.ndarray (det_r, det_c) f32
    backend.center_of_mass(det_mask=None)  -> (com_col, com_row) — flat (N,) f32

MetalRawBackend additionally implements the OPTIONAL fast-sidecar /
radial-cache / multi-dataset hooks (declared in ``MPSCapabilities``); other
backends report ``capabilities = ()`` and the widget skips the related
controls.
"""
from __future__ import annotations

from typing import Iterable, Literal, Protocol, runtime_checkable

import numpy as np


ReduceMode = Literal["mean", "sum", "max"]
ComputeCapability = Literal[
    "fast_sidecar",      # ``ensure_fast_sidecar()`` + ``has_fast`` + ``fast_bin``
    "radial_cache",      # ``ensure_radial_cache(center)`` + ``radial_masked_sum(...)``
    "multi_dataset",     # ``set_active_dataset(idx)`` + ``on_ready``
    "row_prefix_exact",  # exact no-bin reductions via row-prefix
]


@runtime_checkable
class ComputeBackend(Protocol):
    """Protocol every Show4DSTEM compute backend conforms to.

    Implementations: ``TorchBackend`` (universal torch), ``MetalRawBackend``
    (raw Metal for big-no-bin MPS), ``CudaKernelCompute`` (future cupy
    RawKernel for web Browse).

    Only the REQUIRED primitives + the ``capabilities`` tuple are declared
    here. Optional features (``fast_sidecar``, ``radial_cache``,
    ``multi_dataset``) are documented in the module docstring and accessed via
    ``"<cap>" in backend.capabilities`` followed by direct method calls — they
    are not part of the duck-typed Protocol so ``isinstance`` works for any
    backend that implements just the required surface.
    """

    # --- Shape + device metadata ---
    scan_shape: tuple[int, int]
    det_shape: tuple[int, int]
    n_frames: int
    device: object  # str | torch.device

    # --- Optional capability list. Tuple of ComputeCapability strings. ---
    capabilities: tuple[ComputeCapability, ...]

    # --- Required primitives ---
    def frame(self, idx: int) -> np.ndarray: ...
    def masked_sum(self, det_mask: np.ndarray) -> np.ndarray: ...
    def mean_dp(self) -> np.ndarray: ...
    def reduce_frames(
        self, scan_indices: Iterable[int], reduce: ReduceMode = "mean"
    ) -> np.ndarray: ...
    def center_of_mass(
        self, det_mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]: ...


# Optional capability methods (NOT part of the runtime Protocol — checked via
# ``"<cap>" in backend.capabilities``). Documented here for IDE discoverability.
#
#   fast_sidecar:
#       backend.ensure_fast_sidecar(verbose=False) -> bool
#       backend.fast_bin -> int
#       backend.has_fast -> bool
#       backend.cache_fast_presets(masks: dict[str, np.ndarray]) -> dict
#
#   radial_cache (exact no-bin BF/ADF on circular / annular masks):
#       backend.ensure_radial_cache(center_row, center_col, *, idle_delay_s=0.75) -> None
#       backend.radial_cache_ready(center_row, center_col) -> bool
#       backend.radial_masked_sum(*, center_row, center_col, outer_radius,
#                                 inner_radius=0.0, build=False) -> np.ndarray | None
#
#   multi_dataset:
#       backend.set_active_dataset(idx) -> None
#       backend.multi_n_ready -> int
#       backend.multi_names -> list[str]
#       backend.multi_active_idx -> int
#       backend.multi_total() -> int
#       backend.set_multi_ready_callback(cb) -> None
