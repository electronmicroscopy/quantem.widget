"""Focused CPU-safe tests for capacity-aware Dataset5dstem paging."""

from __future__ import annotations

import pytest
import torch

from quantem.widget.data.dataset5dstem import Dataset5dstem


def _lazy_series(
    count: int,
    *,
    shape: tuple[int, int, int, int] = (1, 2, 3, 4),
    batch_loader=None,
) -> Dataset5dstem:
    loaders = [
        lambda value=value: torch.full(shape, value, dtype=torch.uint8)
        for value in range(count)
    ]
    return Dataset5dstem.from_lazy_loaders(
        loaders,
        shape=(count, *shape),
        dtype=torch.uint8,
        batch_loader=batch_loader,
    )


def test_auto_page_uses_capacity_weighting_and_equal_budget_round_robin():
    frame_bytes = 1 * 2 * 3 * 4
    weighted = _lazy_series(8)
    weighted.page(
        "auto",
        device=["cuda:0", "cuda:1"],
        max_vram_bytes={
            "cuda:0": 3 * frame_bytes,
            "cuda:1": frame_bytes,
        },
    )

    assert weighted.devices.count("cuda:0") == 6
    assert weighted.devices.count("cuda:1") == 2

    equal = _lazy_series(6)
    equal.page(
        "auto",
        device=["cuda:0", "cuda:1"],
        max_vram_bytes=3 * frame_bytes,
    )

    assert equal.devices == [
        "cuda:0",
        "cuda:1",
        "cuda:0",
        "cuda:1",
        "cuda:0",
        "cuda:1",
    ]


def test_auto_page_preserves_loaded_cuda_device_and_weights_new_slots():
    class FakeCudaFrame:
        ndim = 4
        shape = (1, 2, 3, 4)
        dtype = torch.uint8
        device = torch.device("cuda:1")

        @staticmethod
        def element_size() -> int:
            return 1

        @staticmethod
        def nelement() -> int:
            return 24

    data = _lazy_series(4)
    data._frames[2] = FakeCudaFrame()
    data.page(
        "auto",
        device=["cuda:0", "cuda:1"],
        max_vram_bytes={"cuda:0": 96, "cuda:1": 96},
    )

    assert data.devices[2] == "cuda:1"
    assert data.devices == ["cuda:0", "cuda:1", "cuda:1", "cuda:0"]


def test_appended_lazy_frames_continue_capacity_aware_policy():
    data = _lazy_series(8)
    data.page(
        "auto",
        device=["cuda:0", "cuda:1"],
        max_vram_bytes={"cuda:0": 72, "cuda:1": 24},
    )

    data.append_lazy_frame(lambda: torch.zeros((1, 2, 3, 4), dtype=torch.uint8))
    data.append_lazy_frame(lambda: torch.zeros((1, 2, 3, 4), dtype=torch.uint8))

    assert data.devices[-2:] == ["cuda:1", "cuda:0"]
    assert data.devices.count("cuda:0") == 7
    assert data.devices.count("cuda:1") == 3


def test_progressive_batches_limit_each_cold_device_to_one_per_wave():
    data = _lazy_series(6)
    data.page(
        "auto",
        device=["cuda:0", "cuda:1"],
        max_vram_bytes=1_000,
    )

    waves = data.progressive_batches([0, 2, 1, 3, 4])

    assert waves == [[0, 1], [2, 3], [4]]
    for wave in waves:
        targets = [data.devices[idx] for idx in wave]
        assert len(targets) == len(set(targets))


def test_batch_preload_evicts_only_enough_lru_frames_for_incoming_data():
    observed_loaded: list[list[int]] = []
    data = None

    def batch_loader(indices):
        observed_loaded.append(data.loaded_indices())
        return {
            idx: torch.full((1, 2, 3, 4), idx, dtype=torch.uint8)
            for idx in indices
        }

    loaders = [
        lambda value=value: torch.full((1, 2, 3, 4), value, dtype=torch.uint8)
        for value in range(5)
    ]
    data = Dataset5dstem.from_lazy_loaders(
        loaders,
        shape=(5, 1, 2, 3, 4),
        dtype=torch.uint8,
        initial_frames={idx: loaders[idx]() for idx in range(3)},
        batch_loader=batch_loader,
    )
    data.page(3, device="cpu")
    data.frame(0)
    data.frame(1)

    assert data.preload([3, 4]) == [3, 4]
    assert observed_loaded == [[1]]
    assert data.loaded_indices() == [1, 3, 4]


def test_batch_preload_keeps_byte_bounded_lru_cache_without_real_cuda():
    class FakeCudaFrame:
        ndim = 4
        shape = (1, 2, 3, 4)
        dtype = torch.uint8
        device = torch.device("cuda:0")

        @staticmethod
        def element_size() -> int:
            return 1

        @staticmethod
        def nelement() -> int:
            return 24

    observed_loaded: list[list[int]] = []
    data = None

    def batch_loader(indices):
        observed_loaded.append(data.loaded_indices())
        return {idx: FakeCudaFrame() for idx in indices}

    data = Dataset5dstem.from_lazy_loaders(
        [lambda: FakeCudaFrame() for _ in range(5)],
        shape=(5, 1, 2, 3, 4),
        dtype=torch.uint8,
        initial_frames={idx: FakeCudaFrame() for idx in range(3)},
        batch_loader=batch_loader,
    )
    data._reclaim = lambda devices: None
    data.page("auto", device=["cuda:0"], max_vram_bytes=3 * 24)
    data.frame(0)
    data.frame(1)

    assert data.preload([3, 4]) == [3, 4]
    assert observed_loaded == [[1]]
    assert data.loaded_indices() == [1, 3, 4]


def test_auto_workspace_reserve_rejects_near_capacity_full_series(monkeypatch):
    gib = 1 << 30
    frame_shape = (1, 1, 32768, 32768)
    data = Dataset5dstem.from_lazy_loaders(
        [lambda: None for _ in range(89)],
        shape=(89, *frame_shape),
        dtype=torch.uint8,
    )

    class CudaDeviceContext:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        torch.cuda,
        "device",
        lambda *args, **kwargs: CudaDeviceContext(),
    )
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda *args, **kwargs: (90 * gib, 100 * gib),
    )

    data.page("auto", device=["cuda:0"])
    plan = data.residency_plan()

    assert plan["budget_bytes"] == {torch.device("cuda:0"): 88 * gib}
    assert plan["total_required_bytes"] == 89 * gib
    assert plan["fits"] is False


def test_auto_page_excludes_devices_that_cannot_hold_one_frame():
    # C1: one selected GPU is below one-frame capacity, expect every cold frame
    # to be assigned to the capable GPU and a corrective error when none fit.
    frame_bytes = 1 * 2 * 3 * 4
    data = _lazy_series(4)
    data.page(
        "auto",
        device=["cuda:0", "cuda:1"],
        max_vram_bytes={
            "cuda:0": frame_bytes - 1,
            "cuda:1": 4 * frame_bytes,
        },
    )

    assert data.devices == ["cuda:1"] * 4

    blocked = _lazy_series(2)
    with pytest.raises(MemoryError, match="No selected CUDA device can hold one") as exc:
        blocked.page(
            "auto",
            device=["cuda:0", "cuda:1"],
            max_vram_bytes=frame_bytes - 1,
        )
    assert "increase detector binning" in str(exc.value)
    assert "choose a smaller dtype" in str(exc.value)


def test_foreground_work_refreshes_capacity_and_avoids_oversized_batch():
    # C1: free memory shifts from two GPUs to one single-frame budget, expect
    # progressive placement to refresh and preload to serialize allocations.
    shape = (1, 2, 3, 4)
    frame_bytes = 1 * 2 * 3 * 4
    data = None
    load_targets: list[str] = []
    batch_calls: list[tuple[int, ...]] = []

    class FakeCudaFrame:
        ndim = 4
        dtype = torch.uint8

        def __init__(self, device: str):
            self.shape = shape
            self.device = torch.device(device)

        @staticmethod
        def element_size() -> int:
            return 1

        @staticmethod
        def nelement() -> int:
            return frame_bytes

        def to(self, device):
            raise AssertionError(f"unexpected transfer from {self.device} to {device}")

    def loader(index: int):
        def load():
            target = data.devices[index]
            load_targets.append(target)
            return FakeCudaFrame(target)

        return load

    def batch_loader(indices):
        batch_calls.append(tuple(indices))
        return {idx: FakeCudaFrame(data.devices[idx]) for idx in indices}

    data = Dataset5dstem.from_lazy_loaders(
        [loader(index) for index in range(3)],
        shape=(3, *shape),
        dtype=torch.uint8,
        batch_loader=batch_loader,
    )
    data._reclaim = lambda devices: None
    data.page(
        "auto",
        device=["cuda:0", "cuda:1"],
        max_vram_bytes=2 * frame_bytes,
    )
    assert data.devices == ["cuda:0", "cuda:1", "cuda:0"]

    def constrained_budgets(devices, **kwargs):
        return {
            torch.device("cuda:0"): frame_bytes - 1,
            torch.device("cuda:1"): frame_bytes,
        }

    data._auto_vram_budgets = constrained_budgets

    assert data.progressive_batches([0, 1, 2]) == [[0], [1], [2]]
    assert data.devices == ["cuda:1"] * 3

    assert data.preload([0, 1]) == [0, 1]
    assert batch_calls == []
    assert load_targets == ["cuda:1", "cuda:1"]
    assert data.loaded_indices() == [1]


def test_fixed_count_cuda_page_keeps_safe_batch_loading_enabled():
    # C1: fixed-count paging has no byte-budget map, expect a within-count CUDA
    # batch to remain eligible for the existing optimized batch loader.
    data = _lazy_series(2)
    data.page(2, device="cuda:0")

    assert data._incoming_lazy_frames_fit_page_limits([0, 1]) is True
