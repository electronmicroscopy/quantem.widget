"""Standalone torch 5D-STEM series for quantem.live.

TEMPORARY home. This belongs in quantem core (PR electronmicroscopy/quantem#231,
on top of the torch-native Dataset work #228), but core PR review takes time and
that torch stack is not on quantem core main yet (its ``Dataset4dstem`` is still
array-backed). So this is a self-contained torch container - it does NOT depend
on quantem core - and ships with quantem.live today.

Migrate back to ``quantem.core.datastructures.Dataset5dstem`` once #228/#231
merge: re-point the import in ``quantem.widget.io.hdf5`` and delete this file. The
public surface (from_tensor / from_frames / shape / devices / summary / free /
to / offload / numpy / frames / is_sharded / indexing) mirrors the core version to keep that
swap mechanical. See the migration GitHub issue.

Model: a series (axis 0 = tilt / time / dose / focus / energy) of 4D-STEM
acquisitions ``(N, scan_row, scan_col, k_row, k_col)``. Two backings, one logical
view:
- single tensor (one device) - the common case.
- series of frames (multi-device) - each frame is its own 4D torch tensor on its
  own card, so a series larger than one GPU fits across several. Each frame is an
  independent acquisition: placement is per-frame, freeing VRAM is per-frame.
"""

import gc
from collections.abc import Callable, Sequence
from typing import Iterator, Self

import numpy as np
import torch
from numpy.typing import NDArray

_SERIES_TYPES = ("time", "tilt", "energy", "dose", "focus", "generic")
_GiB = 1 << 30
_MiB = 1 << 20

# Show4DSTEM reductions already cap their largest temporary chunks near 600 MiB.
# Keep one such chunk plus allocator/context headroom, while allowing predictable
# fixed-shape series to use almost all otherwise available VRAM.
_DEFAULT_MAX_VRAM_FRACTION = 0.98
_MAX_VRAM_FRACTION = 0.995
_MIN_WORKSPACE_RESERVE_BYTES = 768 * _MiB
_MAX_WORKSPACE_RESERVE_BYTES = 2 * _GiB


def _is_recoverable_allocation_error(exc: BaseException) -> bool:
    """True for backend allocation failures where a smaller retry can work."""
    if isinstance(exc, MemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message or "oom" in message


def _validate_4(value, default, name: str) -> NDArray:
    if value is None:
        return np.asarray(default, dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.shape != (4,):
        raise ValueError(f"{name} must be length 4 (scan_row, scan_col, k_row, k_col), got {arr.shape}.")
    return arr


class Dataset5dstem:
    def __init__(
        self,
        *,
        tensor: torch.Tensor | None = None,
        frames: list[torch.Tensor] | None = None,
        lazy_loaders: list[Callable[[], torch.Tensor]] | None = None,
        lazy_batch_loader: Callable[[Sequence[int]], dict[int, torch.Tensor]] | None = None,
        lazy_shape: tuple[int, int, int, int, int] | None = None,
        lazy_dtype: torch.dtype | None = None,
        initial_frames: dict[int, torch.Tensor] | None = None,
        name: str = "",
        sampling=None,
        units=None,
        series_type: str = "generic",
        series=None,
    ):
        backings = sum(x is not None for x in (tensor, frames, lazy_loaders))
        if backings != 1:
            raise ValueError("provide exactly one of tensor=, frames=, or lazy_loaders=.")
        if series_type not in _SERIES_TYPES:
            raise ValueError(f"series_type must be one of {_SERIES_TYPES}, got {series_type!r}.")
        self._tensor = tensor          # 5D torch tensor, or None
        self._frames = frames          # list of 4D torch tensors (or lazy slots), or None
        self._lazy_loaders = lazy_loaders
        self._lazy_batch_loader = lazy_batch_loader
        self._lazy_shape = None if lazy_shape is None else tuple(int(x) for x in lazy_shape)
        self._lazy_dtype = lazy_dtype
        if lazy_loaders is not None:
            if not lazy_loaders:
                raise ValueError("lazy_loaders needs at least one loader.")
            if self._lazy_shape is None or len(self._lazy_shape) != 5:
                raise ValueError("lazy_shape must be a 5D shape (N, scan_row, scan_col, k_row, k_col).")
            if int(self._lazy_shape[0]) != len(lazy_loaders):
                raise ValueError(
                    f"lazy_shape first dimension must match loader count; "
                    f"got shape {self._lazy_shape} and {len(lazy_loaders)} loaders."
                )
            if lazy_dtype is None:
                raise ValueError("lazy_dtype is required for lazy_loaders.")
            self._frames = [None] * len(lazy_loaders)
            for idx, frame in (initial_frames or {}).items():
                idx = int(idx)
                if idx < 0 or idx >= len(lazy_loaders):
                    raise ValueError(f"initial frame index {idx} is out of range for {len(lazy_loaders)} loaders.")
                self._validate_lazy_frame(frame, idx)
                self._frames[idx] = frame
        self.name = name
        self.series_type = series_type
        self.sampling = sampling
        self.units = units
        self.series = series

    # --- constructors ---
    @classmethod
    def from_tensor(
        cls, tensor: torch.Tensor, name: str | None = None,
        sampling=None, units=None, series_type: str = "generic", series=None,
    ) -> Self:
        """Wrap a single 5D torch tensor ``(N, scan, scan, k, k)``."""
        if tensor.ndim != 5:
            raise ValueError(
                f"from_tensor needs a 5D tensor (N, scan, scan, k, k), got {tuple(tensor.shape)}."
            )
        return cls(tensor=tensor, name=name or "5D-STEM series (torch)",
                   sampling=sampling, units=units, series_type=series_type, series=series)

    @classmethod
    def from_frames(
        cls, frames: list[torch.Tensor], name: str | None = None,
        sampling=None, units=None, series_type: str = "generic", series=None,
        stack_same_device: bool = True,
    ) -> Self:
        """Build a series from per-frame 4D tensors (each may be on its own device).

        Same-device frames stack into one compact 5D tensor; frames spanning
        DIFFERENT devices stay a per-frame list (each on its card), so a series
        larger than one GPU just works. Invariant: a frame list is kept ONLY when
        the frames genuinely span devices, so ``is_sharded`` is reliable.

        Set ``stack_same_device=False`` for folder browsers and other out-of-core
        workflows where each frame must remain independently pageable even when
        every hot frame currently lives on the same GPU.
        """
        if not frames:
            raise ValueError("from_frames needs at least one 4D tensor.")
        base_shape = tuple(frames[0].shape)
        base_dtype = frames[0].dtype
        for i, f in enumerate(frames):
            if f.ndim != 4:
                raise ValueError(f"frame {i} must be 4D (scan, scan, k, k), got {tuple(f.shape)}.")
            if tuple(f.shape) != base_shape:
                raise ValueError(f"all frames must share shape; frame 0 is {base_shape}, frame {i} is {tuple(f.shape)}.")
            if f.dtype != base_dtype:
                raise ValueError(f"all frames must share dtype; frame 0 is {base_dtype}, frame {i} is {f.dtype}.")
        if stack_same_device and len({str(f.device) for f in frames}) == 1:
            return cls.from_tensor(torch.stack(list(frames), dim=0), name=name,
                                   sampling=sampling, units=units, series_type=series_type, series=series)
        return cls(frames=list(frames), name=name or "5D-STEM series (torch)",
                   sampling=sampling, units=units, series_type=series_type, series=series)

    # Alias: "tensors" reads more naturally than "frames" at the io.read5dstem call site.
    from_tensors = from_frames

    @classmethod
    def from_lazy_loaders(
        cls,
        loaders: list[Callable[[], torch.Tensor]],
        *,
        shape: tuple[int, int, int, int, int],
        dtype: torch.dtype,
        initial_frames: dict[int, torch.Tensor] | None = None,
        batch_loader: Callable[[Sequence[int]], dict[int, torch.Tensor]] | None = None,
        name: str | None = None,
        sampling=None,
        units=None,
        series_type: str = "generic",
        series=None,
    ) -> Self:
        """Build a file-backed 5D series whose frames load only on demand."""
        return cls(
            lazy_loaders=list(loaders),
            lazy_batch_loader=batch_loader,
            lazy_shape=shape,
            lazy_dtype=dtype,
            initial_frames=initial_frames,
            name=name or "lazy 5D-STEM series",
            sampling=sampling,
            units=units,
            series_type=series_type,
            series=series,
        )

    @property
    def is_lazy(self) -> bool:
        """True when file-backed frames can be released and reloaded later."""
        return self._lazy_loaders is not None

    def loaded_indices(self) -> list[int]:
        """Indices whose frame tensor is currently resident in memory."""
        if self._frames is None:
            return list(range(len(self))) if self._tensor is not None else []
        return [i for i, frame in enumerate(self._frames) if frame is not None]

    def append_lazy_frame(
        self,
        loader: Callable[[], torch.Tensor],
        *,
        frame: torch.Tensor | None = None,
        series_value: float | None = None,
    ) -> int:
        """Append one lazy frame slot without loading it immediately.

        This keeps a live folder-backed Show4DSTEM viewer stable while newly
        completed masters become available. The frame is loaded only when the
        viewer selects or computes that panel.
        """
        if self._lazy_loaders is None or self._lazy_shape is None or self._lazy_dtype is None:
            raise RuntimeError("append_lazy_frame requires a lazy Dataset5dstem.")
        if self._frames is None:
            raise RuntimeError("Dataset5dstem has been freed; re-load to use it again.")
        idx = len(self._lazy_loaders)
        self._lazy_loaders.append(loader)
        self._frames.append(None)
        self._lazy_shape = (idx + 1, *self._lazy_shape[1:])
        if frame is not None:
            self._validate_lazy_frame(frame, idx)
            self._frames[idx] = frame
        if self._series is not None:
            value = np.nan if series_value is None else float(series_value)
            self._series = np.concatenate([self._series, np.asarray([value], dtype=float)])
        if hasattr(self, "_page_devices"):
            page_devices = list(getattr(self, "_page_devices", []))
            cycle = list(getattr(self, "_page_device_cycle", []) or [])
            target = (
                cycle[idx % len(cycle)]
                if cycle
                else page_devices[idx % len(page_devices)]
                if page_devices
                else torch.device("cpu")
            )
            self._page_devices.append(target)
        return idx

    def preload(self, indices: Sequence[int] | int) -> list[int]:
        """Load a group of lazy frames, using the batch loader when available.

        Compare-page browsing commonly needs several adjacent 4D-STEM masters at
        once. Loading them one-by-one loses the optimized multi-file CUDA path,
        so a folder-backed series can provide ``lazy_batch_loader`` to materialize
        the whole visible page in one operation. Existing resident frames in that
        page are preserved; lazy frames outside the requested group are released
        before the batch arrives to avoid transient VRAM spikes.
        """
        frames = self._materialize_frames()
        wanted = self._indices(indices)
        if not wanted:
            return []
        missing = [idx for idx in wanted if frames[idx] is None]
        if not missing:
            self._lru = [idx for idx in getattr(self, "_lru", []) if idx not in wanted] + list(wanted)
            self._evict_to_page_limits(current=wanted[-1])
            return []
        if self._lazy_loaders is None:
            return [idx for idx in wanted if self.frame(idx) is not None]

        loaded: list[int] = []
        if self._lazy_batch_loader is not None and len(missing) > 1:
            # Page-level batch loads can allocate several very large frames at
            # once. Drop non-page lazy frames first so the transient remains
            # bounded by the visible page rather than old page + new page.
            self.release(idx=[idx for idx in self.loaded_indices() if idx not in set(wanted)])
            frames = self._materialize_frames()
            gc.collect()
            batch_failed = False
            try:
                batch = self._lazy_batch_loader(tuple(missing))
            except BaseException as exc:
                if not _is_recoverable_allocation_error(exc):
                    raise
                batch_failed = True

            if batch_failed:
                # Run this after leaving the ``except`` block. CUDA/CuPy loaders
                # can leave partial arrays reachable from the exception traceback;
                # reclaiming while that traceback is still alive is ineffective.
                self._reclaim(self._known_reclaim_devices())
                gc.collect()
            else:
                for idx in missing:
                    frame = batch.get(idx)
                    if frame is None:
                        continue
                    self._validate_lazy_frame(frame, idx)
                    frames[idx] = frame
                    loaded.append(idx)

        for idx in missing:
            if frames[idx] is None:
                try:
                    self.frame(idx)
                except BaseException as exc:
                    if not _is_recoverable_allocation_error(exc):
                        raise
                    self._reclaim(self._known_reclaim_devices())
                    gc.collect()
                    break
                loaded.append(idx)

        resident_wanted = [idx for idx in wanted if frames[idx] is not None]
        self._lru = [idx for idx in getattr(self, "_lru", []) if idx not in wanted] + resident_wanted
        self._evict_to_page_limits(current=wanted[-1])
        return loaded

    def preload_batches(self, indices: Sequence[int] | int) -> list[list[int]]:
        """Split lazy indices into groups that fit the configured page budget.

        A display page can contain more panels than the raw 4D data that fit in
        VRAM. This method keeps that UI page intact while returning smaller load
        groups based on the fixed frame budget or per-device byte budgets set by
        :meth:`page`.
        """
        wanted = self._indices(indices)
        if not wanted:
            return []

        fixed_budget = getattr(self, "_page_budget", None)
        byte_budgets = getattr(self, "_page_max_vram_bytes", None) or {}
        page_devices = list(getattr(self, "_page_devices", []) or [])
        if fixed_budget is None and not byte_budgets:
            return [wanted]

        batches: list[list[int]] = []
        current: list[int] = []
        used: dict[torch.device, int] = {}

        def target_device(idx: int) -> torch.device:
            if page_devices:
                device = self._as_device(page_devices[idx % len(page_devices)])
            else:
                frame = self._materialize_frames()[idx]
                device = torch.device("cpu") if frame is None else frame.device
            if device.type == "cuda":
                return torch.device(f"cuda:{0 if device.index is None else device.index}")
            return device

        def fits(idx: int) -> bool:
            if fixed_budget is not None and len(current) >= int(fixed_budget):
                return False
            device = target_device(idx)
            budget = byte_budgets.get(device)
            if budget is None:
                return True
            nbytes = self._frame_nbytes_for_index(idx)
            return not current or used.get(device, 0) + nbytes <= int(budget)

        for idx in wanted:
            if current and not fits(idx):
                batches.append(current)
                current = []
                used = {}
            device = target_device(idx)
            current.append(idx)
            used[device] = used.get(device, 0) + self._frame_nbytes_for_index(idx)
        if current:
            batches.append(current)
        return batches

    def _validate_lazy_frame(self, frame: torch.Tensor, i: int) -> None:
        if self._lazy_shape is None or self._lazy_dtype is None:
            raise RuntimeError("lazy frame validation requires lazy shape and dtype metadata.")
        if frame.ndim != 4:
            raise ValueError(f"lazy frame {i} must load a 4D tensor, got {tuple(frame.shape)}.")
        expected_shape = tuple(self._lazy_shape[1:])
        if tuple(frame.shape) != expected_shape:
            raise ValueError(
                f"lazy frame {i} loaded shape {tuple(frame.shape)}, expected {expected_shape}."
            )
        if frame.dtype != self._lazy_dtype:
            raise ValueError(
                f"lazy frame {i} loaded dtype {frame.dtype}, expected {self._lazy_dtype}."
            )

    # --- calibration (4-length: scan + k; series axis is separate) ---
    @property
    def sampling(self) -> NDArray: return self._sampling

    @sampling.setter
    def sampling(self, value) -> None:
        self._sampling = _validate_4(value, [1, 1, 1, 1], "sampling")

    @property
    def units(self) -> list[str]: return self._units

    @units.setter
    def units(self, value) -> None:
        if value is None:
            self._units = ["pixels"] * 4
        else:
            u = [str(x) for x in value]
            if len(u) != 4:
                raise ValueError(f"units must be length 4, got {len(u)}.")
            self._units = u

    # --- series metadata ---
    @property
    def series(self) -> NDArray | None: return self._series

    @series.setter
    def series(self, value) -> None:
        if value is None:
            self._series = None
            return
        arr = np.asarray(value, dtype=float)
        if arr.ndim != 1 or len(arr) != len(self):
            raise ValueError(f"series must be 1D length {len(self)}, got shape {arr.shape}.")
        self._series = arr

    # --- logical 5D view ---
    @property
    def is_sharded(self) -> bool:
        """True when the series spans more than one device."""
        if self._frames is None:
            return False
        return len({str(t.device) for t in self._frames if t is not None}) > 1

    @property
    def shape(self) -> tuple[int, ...]:
        if self._lazy_shape is not None:
            return self._lazy_shape
        if self._frames is not None:
            return (len(self._frames), *tuple(self._frames[0].shape))
        if self._tensor is None:
            raise RuntimeError("Dataset5dstem has been freed; re-load to use it again.")
        return tuple(self._tensor.shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)  # 5

    @property
    def tensor(self) -> "torch.Tensor":
        """The 5D ``(N, scan, scan, k, k)`` torch tensor on GPU — for viewers/solvers.

        Single-device series return the compact stacked tensor as-is (no copy).
        A multi-device (sharded) series is stacked onto its first frame's device
        (a copy) so the result is one contiguous 5D tensor.
        """
        if self._tensor is not None:
            return self._tensor
        if self._frames is None:
            raise RuntimeError("Dataset5dstem has been freed; re-load to use it again.")
        frames = [self.frame(i) for i in range(len(self))]
        dev = frames[0].device
        return torch.stack([f.to(dev) for f in frames])

    @property
    def dtype(self):
        if self._lazy_dtype is not None:
            return self._lazy_dtype
        return self._frames[0].dtype if self._frames is not None else self._tensor.dtype

    @property
    def device(self):
        """Device for the first frame.

        A sharded series can span multiple devices; use :attr:`devices` when
        placement matters for every frame. The first-frame device is enough for
        viewer initialization and small coordinate tensors.
        """
        if self._frames is not None:
            for frame in self._frames:
                if frame is not None:
                    return frame.device
            page_devices = getattr(self, "_page_devices", None)
            if page_devices:
                return self._as_device(page_devices[0])
            return torch.device("cpu")
        if self._tensor is None:
            raise RuntimeError("Dataset5dstem has been freed; re-load to use it again.")
        return self._tensor.device

    @property
    def devices(self) -> list[str]:
        """Device of each frame, in series order."""
        if self._frames is not None:
            page_devices = getattr(self, "_page_devices", None) or []
            out = []
            for i, frame in enumerate(self._frames):
                if frame is not None:
                    out.append(str(frame.device))
                elif page_devices:
                    out.append(str(self._as_device(page_devices[i % len(page_devices)])))
                else:
                    out.append("cpu")
            return out
        return [str(self._tensor.device)] * len(self)

    @property
    def nbytes(self) -> int:
        """Total logical bytes across the full series."""
        if self._lazy_shape is not None:
            return int(self.element_size() * self.numel())
        if self._frames is not None:
            return int(sum(f.element_size() * f.nelement() for f in self._frames))
        if self._tensor is None:
            raise RuntimeError("Dataset5dstem has been freed; re-load to use it again.")
        return int(self._tensor.element_size() * self._tensor.nelement())

    @property
    def resident_nbytes(self) -> int:
        """Bytes currently backed by live tensors.

        For lazy file-backed series this is the loaded/cache footprint, not the
        logical size of every master on disk.
        """
        if self._frames is not None:
            return int(sum(f.element_size() * f.nelement() for f in self._frames if f is not None))
        if self._tensor is None:
            return 0
        return int(self._tensor.element_size() * self._tensor.nelement())

    def numel(self) -> int:
        """Total logical element count, matching ``torch.Tensor.numel()``."""
        total = 1
        for value in self.shape:
            total *= int(value)
        return int(total)

    def element_size(self) -> int:
        """Bytes per element, matching ``torch.Tensor.element_size()``."""
        if self._lazy_dtype is not None:
            return int(torch.empty((), dtype=self._lazy_dtype).element_size())
        if self._frames is not None:
            return int(self._frames[0].element_size())
        if self._tensor is None:
            raise RuntimeError("Dataset5dstem has been freed; re-load to use it again.")
        return int(self._tensor.element_size())

    @property
    def frames(self) -> list[torch.Tensor]:
        """Per-frame 4D torch tensors, in series order, each on its device.

        The plain-torch view a viewer consumes: ``Show4DSTEM(dset.frames)``.
        """
        if self._frames is not None:
            return [self.frame(i) for i in range(len(self))]
        return [self._tensor[i] for i in range(len(self))]

    def numpy(self) -> NDArray:
        """Gather the whole series to ONE host numpy array ``(N, scan, scan, k, k)``.

        Pulls every frame off its GPU and stacks on the host - the full 5D must
        fit RAM (a 108 GiB no-bin series will not; bin first, or pull per frame
        via ``dset[i].cpu().numpy()``).
        """
        if self._frames is not None:
            return np.stack([self.frame(i).detach().cpu().numpy() for i in range(len(self))], axis=0)
        if self._tensor is None:
            raise RuntimeError("Dataset5dstem has been freed; re-load to use it again.")
        return self._tensor.detach().cpu().numpy()

    def summary(self) -> dict[str, float]:
        """Print a frame | device | GiB | dtype table; return per-device GiB totals."""
        frames = [self._frames[i] if self._frames is not None else self[i] for i in range(len(self))]
        per_device: dict[str, float] = {}
        sr, sc = self.shape[1], self.shape[2]
        kr, kc = self.shape[3], self.shape[4]
        print(f"{self.name}  ({self.series_type} series, {len(self)} frames)")
        print(f"  scan {sr}x{sc}  detector {kr}x{kc}  sampling {tuple(self._sampling)} {self._units}")
        if self._series is not None:
            print(f"  series: {self.series_type} {list(self._series)}")
        print(f"{'frame':>5}  {'device':>8}  {'GiB':>6}  dtype")
        for i, f in enumerate(frames):
            if f is None:
                print(f"{i:>5}  {'lazy':>8}  {0:>6.2f}  {self.dtype}")
                continue
            gib = f.element_size() * f.nelement() / _GiB
            dev = str(f.device)
            per_device[dev] = per_device.get(dev, 0.0) + gib
            print(f"{i:>5}  {dev:>8}  {gib:>6.2f}  {f.dtype}")
        for dev, gib in sorted(per_device.items()):
            print(f"  total {dev}: {gib:.2f} GiB")
        return per_device

    # --- placement + lifecycle (per-frame VRAM control) ---
    @staticmethod
    def _as_device(device) -> torch.device:
        """One spelling for placement args: int -> cuda:int; str/torch.device pass through."""
        if isinstance(device, torch.device):
            return device
        if isinstance(device, int):
            return torch.device(f"cuda:{device}")
        return torch.device(device)

    def _materialize_frames(self) -> list[torch.Tensor | None]:
        """Force the per-frame list backing so frames can be placed/freed independently.

        A single 5D tensor is split into independent per-frame tensors (own storage via
        ``clone``) and the 5D backing dropped, so moving or freeing one frame does not pin
        the rest. No-op when already frame-backed (the multi-device load path)."""
        if self._frames is None:
            if self._tensor is None:
                raise RuntimeError("Dataset5dstem has been freed; re-load to use it again.")
            self._frames = [self._tensor[i].clone() for i in range(self._tensor.shape[0])]
            self._tensor = None
        return self._frames

    def _ensure_frame_loaded(self, i: int) -> torch.Tensor:
        """Load lazy frame ``i`` once, preserving already-loaded/cached frames."""
        if self._frames is None:
            raise RuntimeError("Dataset5dstem has been freed; re-load to use it again.")
        i = i % len(self)
        frame = self._frames[i]
        if frame is not None:
            return frame
        if self._lazy_loaders is None or self._lazy_shape is None or self._lazy_dtype is None:
            raise RuntimeError(f"frame {i} is missing and no lazy loader is available.")
        loaded = self._lazy_loaders[i]()
        self._validate_lazy_frame(loaded, i)
        self._frames[i] = loaded
        return loaded

    def _indices(self, idx) -> list[int]:
        """Normalize idx (None=all, int, or iterable) to a sorted unique in-range index list."""
        if idx is None:
            return list(range(len(self)))
        if isinstance(idx, int):
            idx = [idx]
        return sorted({i % len(self) for i in idx})

    def _known_reclaim_devices(self) -> set[torch.device]:
        """CUDA devices that may have unreferenced cached blocks after a failed load."""
        devs: set[torch.device] = set()
        if self._frames is not None:
            devs.update(frame.device for frame in self._frames if frame is not None)
        for device in getattr(self, "_page_devices", []) or []:
            dev = self._as_device(device)
            if dev.type == "cuda":
                devs.add(torch.device(f"cuda:{0 if dev.index is None else dev.index}"))
        return devs

    @staticmethod
    def _reclaim(devs) -> None:
        """Empty torch AND cupy caching pools on each device so freed VRAM returns.

        Frames are cupy-backed (io.load -> from_dlpack), so released memory sits in cupy's
        pool and ``torch.cuda.empty_cache()`` alone does NOT return it. Safe on a device that
        still hosts live frames: both calls only release unreferenced blocks."""
        cuda_devs = {d for d in devs if d.type == "cuda"}
        for d in cuda_devs:
            with torch.cuda.device(d):
                torch.cuda.empty_cache()
        try:
            import cupy as cp  # noqa: PLC0415  (lazy: keep torch-only import on a CUDA-less laptop)
        except ImportError:
            return
        for d in cuda_devs:
            with cp.cuda.Device(0 if d.index is None else d.index):
                cp.get_default_memory_pool().free_all_blocks()

    def to(self, device, idx=None) -> Self:
        """Move frame(s) to a device, in place; return self.

        ``device``: int (cuda:N), str/torch.device, or a LIST to round-robin the WHOLE
        series across cards (``idx`` must be None when spreading). ``idx`` (int or iterable)
        moves a subset and leaves the rest put, so the series spans devices afterward.
        Source-card VRAM is reclaimed once the old tensors drop. Needed to consolidate a
        series onto one card, spread it across several, or rebalance per frame."""
        if isinstance(device, (list, tuple)):
            if idx is not None:
                raise ValueError("cannot pass idx= when spreading across a device list; spread moves the whole series.")
            targets = [self._as_device(d) for d in device]
            old = self._materialize_frames()
            src = {f.device for f in old if f is not None}
            self._frames = [
                None if f is None else f.to(targets[i % len(targets)])
                for i, f in enumerate(old)
            ]
            del old
            self._reclaim(src)
            return self
        target = self._as_device(device)
        if idx is None and self._frames is None:
            if self._tensor is None:
                raise RuntimeError("Dataset5dstem has been freed; re-load to use it again.")
            src = self._tensor.device
            if src != target:
                self._tensor = self._tensor.to(target)
                self._reclaim({src})
            return self
        old = self._materialize_frames()
        move = set(self._indices(idx))
        src = {old[i].device for i in move if old[i] is not None}
        self._frames = [
            f.to(target) if i in move and f is not None else f
            for i, f in enumerate(old)
        ]
        del old
        self._reclaim(src)
        return self

    def offload(self, idx=None) -> Self:
        """Spill frame(s) to host RAM (``to('cpu')``), keeping them in the series; return self.

        Reclaims a card's VRAM without losing the data; bring it back with ``.to(device, idx)``.
        The non-destructive counterpart to ``.free`` - use this when a series is bigger than
        total VRAM and you want to page frames in and out."""
        return self.to("cpu", idx)

    def release(self, idx=None, device=None) -> list[int]:
        """Release resident frame tensors while preserving frame indices.

        For lazy file-backed series, released frames become unloaded slots and
        reload from their original loader on the next access. This is the right
        primitive for UI hide/curation: hidden datasets stop occupying GPU
        memory, but labels, ordering, and saved hidden-state indices remain
        stable. For non-lazy frame-backed series this offloads selected GPU
        frames to CPU without removing them.
        """
        frames = self._materialize_frames()
        drop = set(self._indices(idx)) if idx is not None else set()
        if device is not None:
            dev = self._as_device(device)
            drop |= {i for i, frame in enumerate(frames) if frame is not None and frame.device == dev}
        if not drop:
            return []

        reclaimed: set[torch.device] = set()
        released: list[int] = []
        for i in sorted(drop):
            frame = frames[i]
            if frame is None:
                continue
            dev = frame.device
            if self._lazy_loaders is not None:
                frames[i] = None
                released.append(i)
            elif dev.type != "cpu":
                frames[i] = frame.to("cpu")
                released.append(i)
            if dev.type != "cpu":
                reclaimed.add(dev)

        if hasattr(self, "_lru"):
            self._lru = [i for i in self._lru if i not in set(released)]
        # The loop variable otherwise keeps the final released tensor alive
        # during _reclaim(), leaving its CUDA/CuPy block in the allocator pool.
        del frame
        if reclaimed:
            self._reclaim(reclaimed)
        return released

    def free(self, idx=None, device=None) -> None:
        """Release frame VRAM, reclaiming the torch + cupy pools.

        No args -> free the WHOLE series (spent afterward; accessing it raises). ``idx``
        (int/iterable) or ``device`` (int/str) frees a SUBSET: those frames are removed from
        the series and their card's VRAM reclaimed, while the remaining frames stay usable.
        Destructive (the data is gone) - use ``.offload`` to keep it on CPU instead."""
        if idx is None and device is None:
            devs = set()
            if self._frames is not None:
                devs = {t.device for t in self._frames if t is not None}
            elif self._tensor is not None:
                devs = {self._tensor.device}
            self._frames = None
            self._tensor = None
            self._lazy_loaders = None
            self._lazy_batch_loader = None
            self._lazy_shape = None
            self._lazy_dtype = None
            self._series = None
            self._reclaim(devs)
            return
        old = self._materialize_frames()
        drop = set(self._indices(idx)) if idx is not None else set()
        if device is not None:
            dev = self._as_device(device)
            drop |= {i for i, f in enumerate(old) if f is not None and f.device == dev}
        if not drop:
            return
        src = {old[i].device for i in drop if old[i] is not None}
        keep = [i for i in range(len(old)) if i not in drop]
        self._frames = [old[i] for i in keep]
        if self._lazy_loaders is not None:
            self._lazy_loaders = [self._lazy_loaders[i] for i in keep]
            if self._lazy_shape is not None:
                self._lazy_shape = (len(keep), *self._lazy_shape[1:])
        if hasattr(self, "_page_devices"):
            self._page_devices = [self._page_devices[i] for i in keep]
        if hasattr(self, "_lru"):
            old_to_new = {old_i: new_i for new_i, old_i in enumerate(keep)}
            self._lru = [old_to_new[i] for i in self._lru if i in old_to_new]
        if self._series is not None:
            self._series = self._series[keep]
        del old
        if not self._frames:
            self._frames = None
            self._tensor = None
            self._lazy_loaders = None
            self._lazy_shape = None
            self._lazy_dtype = None
            self._series = None
        self._reclaim(src)

    # --- auto-swap / out-of-core paging ---
    @staticmethod
    def _frame_nbytes(frame: torch.Tensor) -> int:
        return int(frame.element_size() * frame.nelement())

    def _frame_nbytes_for_index(self, idx: int) -> int:
        frames = self._materialize_frames()
        frame = frames[idx]
        if frame is not None:
            return self._frame_nbytes(frame)
        if self._lazy_shape is not None:
            n = 1
            for value in self._lazy_shape[1:]:
                n *= int(value)
            return int(n * self.element_size())
        return 0

    @staticmethod
    def _canonical_cuda_device(device: torch.device) -> torch.device:
        if device.type != "cuda":
            return device
        return torch.device(f"cuda:{0 if device.index is None else device.index}")

    def _page_target_device(self, idx: int) -> torch.device:
        frames = self._materialize_frames()
        frame = frames[idx]
        if frame is not None and frame.device.type == "cuda":
            return self._canonical_cuda_device(frame.device)
        page_devices = list(getattr(self, "_page_devices", []) or [])
        if not page_devices:
            return torch.device("cpu") if frame is None else frame.device
        return self._canonical_cuda_device(
            self._as_device(page_devices[idx % len(page_devices)])
        )

    def _managed_cuda_nbytes(self) -> dict[torch.device, int]:
        managed: dict[torch.device, int] = {}
        for frame in self._materialize_frames():
            if frame is None or frame.device.type != "cuda":
                continue
            device = self._canonical_cuda_device(frame.device)
            managed[device] = managed.get(device, 0) + self._frame_nbytes(frame)
        return managed

    def _largest_frame_nbytes_by_device(
        self, devices: Sequence[torch.device]
    ) -> dict[torch.device, int]:
        largest = {self._canonical_cuda_device(device): 0 for device in devices}
        for idx in range(len(self)):
            device = self._page_target_device(idx)
            if device.type != "cuda":
                continue
            largest[device] = max(
                largest.get(device, 0), self._frame_nbytes_for_index(idx)
            )
        return largest

    @staticmethod
    def _adaptive_workspace_reserve(total: int, largest_frame: int) -> int:
        """Return bounded CUDA headroom for reductions and allocator metadata."""
        requested = max(
            _MIN_WORKSPACE_RESERVE_BYTES,
            int(total) // 100,
            int(largest_frame) // 8,
        )
        return min(requested, _MAX_WORKSPACE_RESERVE_BYTES)

    def _evict_for_incoming_lazy_frame(
        self,
        *,
        current: int,
        target_device: torch.device,
    ) -> None:
        """Make room before a lazy loader allocates the next frame."""
        if self._lazy_loaders is None:
            return
        frames = self._materialize_frames()
        self._lru = [i for i in getattr(self, "_lru", []) if i < len(frames)]
        reclaimed: set[torch.device] = set()

        def evict(idx: int) -> None:
            frame = frames[idx]
            if frame is None:
                self._lru[:] = [x for x in self._lru if x != idx]
                return
            dev = frame.device
            frames[idx] = None
            if dev.type != "cpu":
                reclaimed.add(dev)
            self._lru[:] = [x for x in self._lru if x != idx]

        budget = getattr(self, "_page_budget", None)
        while budget is not None:
            resident = [
                i for i in self._lru
                if frames[i] is not None
                and (self._lazy_loaders is not None or frames[i].device.type != "cpu")
            ]
            if len(resident) < int(budget):
                break
            candidates = [i for i in resident if i != current] or resident
            evict(candidates[0])

        byte_budgets = getattr(self, "_page_max_vram_bytes", None) or {}
        target = self._as_device(target_device)
        if target.type == "cuda":
            target = torch.device(f"cuda:{0 if target.index is None else target.index}")
        max_bytes = int(byte_budgets.get(target, 0))
        incoming = self._frame_nbytes_for_index(current)
        while target.type == "cuda" and max_bytes > 0:
            resident = [
                i for i in self._lru
                if frames[i] is not None and frames[i].device == target
            ]
            used = sum(self._frame_nbytes(frames[i]) for i in resident if frames[i] is not None)
            if used + incoming <= max_bytes or not resident:
                break
            candidates = [i for i in resident if i != current] or resident
            evict(candidates[0])

        if reclaimed:
            self._reclaim(reclaimed)

    def _auto_vram_budgets(
        self,
        devices,
        *,
        max_vram_fraction: float = _DEFAULT_MAX_VRAM_FRACTION,
        reserve_vram_bytes: int | None = None,
        max_vram_bytes: int | dict | None = None,
    ) -> dict[torch.device, int]:
        """Return per-CUDA-device budgets for this series' resident frames.

        The automatic budget uses current free memory plus bytes already owned by
        this Dataset5dstem. Other tensors in the process remain accounted for as
        unavailable, so loading a predictable full series cannot evict unrelated
        work merely because it belongs to the same Python process.
        """
        targets = [self._as_device(d) for d in devices]
        cuda_targets = sorted(
            {
                d.index if d.index is not None else 0
                for d in targets
                if d.type == "cuda"
            }
        )
        if not cuda_targets:
            return {}
        if max_vram_bytes is not None:
            if isinstance(max_vram_bytes, dict):
                out = {}
                for key, value in max_vram_bytes.items():
                    dev = self._as_device(key)
                    if dev.type == "cuda":
                        target = torch.device(
                            f"cuda:{0 if dev.index is None else dev.index}"
                        )
                        out[target] = int(value)
                return out
            return {
                torch.device(f"cuda:{idx}"): int(max_vram_bytes)
                for idx in cuda_targets
            }
        fraction = float(
            max(0.05, min(float(max_vram_fraction), _MAX_VRAM_FRACTION))
        )
        managed = self._managed_cuda_nbytes()
        cuda_devices = [torch.device(f"cuda:{idx}") for idx in cuda_targets]
        largest = self._largest_frame_nbytes_by_device(cuda_devices)
        budgets: dict[torch.device, int] = {}
        for idx in cuda_targets:
            device = torch.device(f"cuda:{idx}")
            with torch.cuda.device(idx):
                free, total = torch.cuda.mem_get_info(idx)
            reserve = (
                self._adaptive_workspace_reserve(total, largest.get(device, 0))
                if reserve_vram_bytes is None
                else max(0, int(reserve_vram_bytes))
            )
            total_budget = int(total * fraction)
            series_bytes = int(managed.get(device, 0))
            available_to_series = max(0, int(free) + series_bytes - reserve)
            # Never invalidate a frame that is already live merely because
            # another process consumed memory after it was loaded.
            budget = max(series_bytes, min(total_budget, available_to_series))
            budgets[device] = budget
        return budgets

    def _refresh_auto_vram_budgets(self) -> None:
        config = getattr(self, "_page_auto_config", None)
        if not config:
            return
        self._page_max_vram_bytes = self._auto_vram_budgets(
            self._page_devices,
            max_vram_fraction=config["max_vram_fraction"],
            reserve_vram_bytes=config["reserve_vram_bytes"],
            max_vram_bytes=config["max_vram_bytes"],
        )

    def residency_plan(
        self, indices: Sequence[int] | int | None = None
    ) -> dict[str, object]:
        """Plan whether selected lazy frames can all remain in GPU memory.

        Frame shapes and dtypes are known before file-backed frames load, so this
        calculation does not read detector data. The returned per-device byte
        totals are suitable for status displays and deterministic tests.
        """
        wanted = self._indices(indices)
        self._refresh_auto_vram_budgets()
        required: dict[torch.device, int] = {}
        for idx in wanted:
            device = self._page_target_device(idx)
            required[device] = (
                required.get(device, 0) + self._frame_nbytes_for_index(idx)
            )

        budgets = dict(getattr(self, "_page_max_vram_bytes", None) or {})
        fixed_budget = getattr(self, "_page_budget", None)
        cuda_only = bool(required) and all(
            device.type == "cuda" for device in required
        )
        count_fits = fixed_budget is None or len(wanted) <= int(fixed_budget)
        bytes_fit = bool(budgets) and all(
            device in budgets and int(nbytes) <= int(budgets[device])
            for device, nbytes in required.items()
        )
        fits = bool(cuda_only and count_fits and bytes_fit)
        resident = set(self.vram_resident())
        return {
            "fits": fits,
            "indices": list(wanted),
            "required_bytes": required,
            "budget_bytes": budgets,
            "total_required_bytes": int(sum(required.values())),
            "resident_count": sum(idx in resident for idx in wanted),
        }

    def preload_all_if_fits(
        self,
        indices: Sequence[int] | int | None = None,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> dict[str, object]:
        """Load every selected lazy frame only when the residency plan fits.

        A memory change between planning and allocation is handled as a normal
        paging fallback: frames that fit remain cached and the rest stay lazy.
        Backend allocation errors are reclaimed and are not surfaced to users as
        raw CUDA/CuPy out-of-memory traces.
        """
        plan = self.residency_plan(indices)
        wanted = list(plan["indices"])
        if not bool(plan["fits"]):
            plan["resident"] = False
            self._last_residency_plan = plan
            return plan

        # Keep each master in an independent allocation. A single stacked batch
        # would make all frame views share one storage block, so hiding one panel
        # could not return its share of VRAM until every sibling was released.
        frames = self._materialize_frames()
        for idx in wanted:
            if cancelled is not None and cancelled():
                break
            if frames[idx] is not None:
                continue
            try:
                self.frame(idx)
            except BaseException as exc:
                if not _is_recoverable_allocation_error(exc):
                    raise
                self._reclaim(self._known_reclaim_devices())
                gc.collect()
                break
        resident = set(self.vram_resident())
        complete = all(idx in resident for idx in wanted)
        if not complete:
            self._refresh_auto_vram_budgets()
            self._evict_to_page_limits()
            resident = set(self.vram_resident())
        plan["resident"] = complete
        plan["resident_count"] = sum(idx in resident for idx in wanted)
        self._last_residency_plan = plan
        return plan

    def _evict_to_page_limits(self, *, current: int | None = None) -> None:
        frames = self._materialize_frames()
        budget = getattr(self, "_page_budget", None)
        byte_budgets = getattr(self, "_page_max_vram_bytes", None) or {}
        if budget is None and not byte_budgets:
            return
        self._lru = [i for i in getattr(self, "_lru", []) if i < len(frames)]
        for i, frame in enumerate(frames):
            if (
                frame is not None
                and (self._lazy_loaders is not None or frame.device.type != "cpu")
                and i not in self._lru
            ):
                self._lru.insert(0, i)

        reclaimed: set[torch.device] = set()

        def evict(idx: int) -> None:
            frame = frames[idx]
            if frame is None:
                self._lru[:] = [x for x in self._lru if x != idx]
                return
            dev = frame.device
            frames[idx] = None if self._lazy_loaders is not None else frame.to("cpu")
            reclaimed.add(dev)
            self._lru[:] = [x for x in self._lru if x != idx]

        while budget is not None:
            resident = [
                i for i in self._lru
                if frames[i] is not None
                and (self._lazy_loaders is not None or frames[i].device.type != "cpu")
            ]
            if len(resident) <= int(budget):
                break
            candidates = [i for i in resident if i != current] or resident
            evict(candidates[0])

        for dev, max_bytes in byte_budgets.items():
            if dev.type != "cuda" or max_bytes <= 0:
                continue
            while True:
                resident = [
                    i
                    for i in self._lru
                    if frames[i] is not None and frames[i].device == dev
                ]
                used = sum(
                    self._frame_nbytes(frames[i])
                    for i in resident
                    if frames[i] is not None
                )
                if used <= max_bytes or not resident:
                    break
                candidates = [i for i in resident if i != current]
                if not candidates:
                    break
                evict(candidates[0])

        if reclaimed:
            self._reclaim(reclaimed)

    def page(
        self,
        vram_frames: int | str,
        device=None,
        *,
        max_vram_fraction: float = _DEFAULT_MAX_VRAM_FRACTION,
        reserve_vram_bytes: int | None = None,
        max_vram_bytes: int | dict | None = None,
    ) -> Self:
        """Enable out-of-core paging.

        ``vram_frames`` may be an integer fixed count, or ``"auto"`` to keep as
        many frames resident as the current CUDA memory budget allows. Remaining
        frames live in host RAM and are paged in on access via :meth:`frame`,
        evicting least-recently-used frames when over budget.

        Lets you scrub or jointly reconstruct a series LARGER than VRAM — only the
        active window sits on the GPU. (RAM tier today; a disk tier for series bigger
        than RAM is the next step.) ``device=None`` preserves each frame's current
        device before offload, so a round-robin multi-GPU series pages frame ``i``
        back to its original card. Pass one device to page every frame to the same
        card, or a list/tuple to define a round-robin paging target. Returns self.
        """
        frames = self._materialize_frames()
        auto = isinstance(vram_frames, str) and vram_frames.lower() == "auto"
        self._page_budget = None if auto else max(1, int(vram_frames))
        if device is None:
            self._page_devices = [
                f.device if f is not None else torch.device("cpu")
                for f in frames
            ]
            self._page_device_cycle = []
        elif isinstance(device, (list, tuple)):
            targets = [self._as_device(d) for d in device]
            if not targets:
                raise ValueError("device list for page() must not be empty.")
            self._page_devices = [
                targets[i % len(targets)] for i in range(len(frames))
            ]
            self._page_device_cycle = list(targets)
        else:
            target = self._as_device(device)
            self._page_devices = [target for _ in frames]
            self._page_device_cycle = [target]
        if auto:
            self._page_auto_config = {
                "max_vram_fraction": float(max_vram_fraction),
                "reserve_vram_bytes": reserve_vram_bytes,
                "max_vram_bytes": max_vram_bytes,
            }
            self._page_max_vram_bytes = self._auto_vram_budgets(
                self._page_devices,
                max_vram_fraction=max_vram_fraction,
                reserve_vram_bytes=reserve_vram_bytes,
                max_vram_bytes=max_vram_bytes,
            )
            self._lru = [i for i, f in enumerate(frames) if f is not None and f.device.type != "cpu"]
            self._evict_to_page_limits()
        else:
            self._page_auto_config = None
            self._page_max_vram_bytes = {}
            self._lru = [i for i, f in enumerate(frames) if f is not None and f.device.type != "cpu"]
            if self._lazy_loaders is None:
                self.offload()  # fixed-count mode starts cold for a flat footprint
            else:
                self._evict_to_page_limits()
        return self

    def frame(self, i: int) -> "torch.Tensor":
        """Frame ``i`` as a GPU tensor, AUTO-PAGING when :meth:`page` is enabled:
        bring ``i`` into VRAM from RAM, and evict the least-recently-used VRAM frame
        if that exceeds the budget. Without :meth:`page` it is just ``self[i]``.
        """
        if getattr(self, "_page_budget", None) is None and not getattr(self, "_page_max_vram_bytes", None):
            return self._ensure_frame_loaded(i) if self._lazy_loaders is not None else self[i]
        frames = self._materialize_frames()
        i = i % len(self)
        page_devices = getattr(self, "_page_devices", None)
        if self._lazy_loaders is not None and frames[i] is None:
            if not page_devices:
                raise RuntimeError("Dataset5dstem paging was not initialized correctly.")
            self._evict_for_incoming_lazy_frame(
                current=i,
                target_device=self._as_device(page_devices[i % len(page_devices)]),
            )
        frame = self._ensure_frame_loaded(i)
        if frame.device.type == "cpu":
            if not page_devices:
                raise RuntimeError("Dataset5dstem paging was not initialized correctly.")
            self._frames[i] = frame.to(page_devices[i % len(page_devices)])
        self._lru = [x for x in self._lru if x != i] + [i]  # most-recent last
        self._evict_to_page_limits(current=i)
        return self._frames[i]

    def vram_resident(self) -> list[int]:
        """Indices of frames currently in VRAM (the rest are paged out to RAM)."""
        if self._frames is None:
            return list(range(len(self))) if self._tensor is not None else []
        return [i for i, f in enumerate(self._frames) if f is not None and f.device.type != "cpu"]

    # --- frame access ---
    def __len__(self) -> int:
        if self._frames is not None:
            return len(self._frames)
        if self._tensor is None:
            raise RuntimeError("Dataset5dstem has been freed; re-load to use it again.")
        return int(self._tensor.shape[0])

    def minibatch_rounds(self, batch_size: int, *, shuffle: bool = True, seed=None):
        """Yield synchronized minibatch ROUNDS for data-parallel multi-GPU ptycho.

        For tilt / time-series joint reconstruction across several GPUs. Frames
        (tilts) are grouped by the device they live on (place them with ``.to([0,1,
        2,3])`` first). Each round is a LIST of one batch PER device — every batch's
        DPs are **resident on that device** (a slice, no PCIe transfer), so each GPU
        works on its local tilts while the solver all-reduces the shared object/probe
        gradient between rounds.

        Each batch is ``(device, frame_idx, scan_idx, dp_batch)``:
        ``dp_batch`` = ``(B, k, k)`` DPs on ``device``; ``scan_idx`` their flat scan
        positions in frame ``frame_idx`` (use to index that tilt's object). Within a
        frame, scan positions are chunked into ``batch_size``. Rounds run until the
        device with the MOST batches is exhausted; shorter devices cycle (so every
        GPU has work every round). One device = ordinary single-GPU minibatch SGD.

        Consume it once per ptycho iteration::

            for it in range(n_iters):
                for rnd in ds.minibatch_rounds(512):
                    for dev, fi, sidx, dps in rnd:        # GPUs in parallel
                        accumulate_grad(obj[fi], probe, dps, sidx)   # on `dev`
                    all_reduce(probe_grad); step()
        """
        import torch
        self._materialize_frames()
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(int(seed))
        # Per-device list of (frame_idx, scan_idx chunk).
        by_dev: dict = {}
        for fi in range(len(self)):
            f = self.frame(fi)
            dev = str(f.device)
            n_scan = int(f.shape[0]) * int(f.shape[1])
            order = (torch.randperm(n_scan, generator=gen) if shuffle
                     else torch.arange(n_scan))
            for c in range(0, n_scan, batch_size):
                by_dev.setdefault(dev, []).append((fi, order[c:c + batch_size]))
        if not by_dev:
            return
        n_rounds = max(len(v) for v in by_dev.values())
        for r in range(n_rounds):
            rnd = []
            for dev, batches in by_dev.items():
                fi, scan_idx = batches[r % len(batches)]
                f = self.frame(fi)
                flat = f.reshape(-1, *f.shape[2:])          # (n_scan, k, k)
                dp = flat[scan_idx.to(f.device)]            # on-device slice, no copy
                rnd.append((dev, fi, scan_idx, dp))
            yield rnd

    def __getitem__(self, index: int | slice) -> torch.Tensor | Self:
        if isinstance(index, int):
            return self.frame(index) if self._lazy_loaders is not None else (
                self._frames[index] if self._frames is not None else self._tensor[index]
            )
        sub_series = None if self._series is None else self._series[index]
        if self._frames is not None:
            if self._lazy_loaders is not None:
                selected = list(range(len(self)))[index]
                initial = {
                    new_i: self._frames[old_i]
                    for new_i, old_i in enumerate(selected)
                    if self._frames[old_i] is not None
                }
                return Dataset5dstem.from_lazy_loaders(
                    [self._lazy_loaders[i] for i in selected],
                    shape=(len(selected), *self.shape[1:]),
                    dtype=self._lazy_dtype,
                    initial_frames=initial,
                    name=self.name,
                    sampling=self._sampling,
                    units=self._units,
                    series_type=self.series_type,
                    series=sub_series,
                )
            return Dataset5dstem.from_frames(
                self._frames[index], name=self.name, sampling=self._sampling,
                units=self._units, series_type=self.series_type, series=sub_series)
        return Dataset5dstem.from_tensor(
            self._tensor[index], name=self.name, sampling=self._sampling,
            units=self._units, series_type=self.series_type, series=sub_series)

    def __iter__(self) -> Iterator[torch.Tensor]:
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        try:
            shp = self.shape
        except RuntimeError:
            return "Dataset5dstem(freed)"
        sharded = ", sharded" if self.is_sharded else ""
        return (f"Dataset5dstem(shape={shp}, {self.series_type} series, "
                f"dtype={self.dtype}{sharded}, name={self.name!r})")
