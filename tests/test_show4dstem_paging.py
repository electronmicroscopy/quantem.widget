"""Show4DSTEM dataset-slider paging (out-of-core multi-dataset VRAM control).

A ShowFolder of many 4D masters loads into one Show4DSTEM behind a dataset
slider. Keeping every master resident fills VRAM; ``page_budget`` controls how
many stay on the GPU and switching the slider pages the target in while evicting
least-recently-used datasets to RAM. This guards both fixed-count paging and
``page_budget="auto"`` memory-sized caching.
"""

from __future__ import annotations

from collections import namedtuple
import threading
import time
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from quantem.widget import Show4DSTEM  # noqa: E402
from quantem.widget.data.dataset5dstem import Dataset5dstem  # noqa: E402

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="paging test needs a CUDA device"
)
LoadResult = namedtuple("LoadResult", ["data", "metadata"])


def _wait_until(predicate, *, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("timed out waiting for background Show4DSTEM work")


def _series(n=4, scan=32, det=48):
    """n independent 4D uint8 datasets on cuda:0 — a synthetic 'n-master folder'."""
    frames = [
        torch.randint(
            0, 100, (scan, scan, det, det), dtype=torch.uint8, device="cuda:0"
        )
        for _ in range(n)
    ]
    return Dataset5dstem.from_frames(frames)


@cuda_required
def test_no_page_budget_keeps_all_datasets_resident():
    ds = _series(4)
    w = Show4DSTEM(ds, verbose=False)
    assert w.n_frames == 4
    # default: every dataset stays on the GPU (instant switch, current behavior)
    assert sorted(ds.vram_resident()) == [0, 1, 2, 3]


@cuda_required
def test_page_budget_caps_residency_and_evicts_lru_on_switch():
    ds = _series(4)
    w = Show4DSTEM(ds, page_budget=1, verbose=False)
    # after mount only the active dataset (frame 0) is resident
    assert ds.vram_resident() == [0]

    baseline = torch.cuda.memory_allocated(0)
    for target in (2, 3, 1, 0):
        w.frame_idx = target
        _ = w._frame_data  # the access that pages target in + evicts the LRU
        resident = ds.vram_resident()
        assert resident == [target], f"budget=1 but resident={resident}"
        # VRAM must not grow across switches — the old dataset is evicted, not leaked
        assert torch.cuda.memory_allocated(0) <= baseline * 1.5


@cuda_required
def test_page_budget_two_keeps_two_resident():
    ds = _series(4)
    w = Show4DSTEM(ds, page_budget=2, verbose=False)
    w.frame_idx = 1
    _ = w._frame_data
    w.frame_idx = 3
    _ = w._frame_data
    resident = set(ds.vram_resident())
    assert len(resident) == 2 and 3 in resident  # newest always resident, ≤ budget


def test_lazy_preload_batches_respect_fixed_page_budget():
    loaders = [
        lambda value=value: torch.full((3, 3, 6, 6), value, dtype=torch.uint8)
        for value in range(5)
    ]
    ds = Dataset5dstem.from_lazy_loaders(
        loaders,
        shape=(5, 3, 3, 6, 6),
        dtype=torch.uint8,
        initial_frames={0: loaders[0]()},
    )
    ds.page(2, device="cpu")

    assert ds.preload_batches(range(5)) == [[0, 1], [2, 3], [4]]


@cuda_required
def test_page_budget_auto_keeps_hot_frames_until_byte_budget():
    frames = [
        torch.randint(0, 100, (16, 16, 24, 24), dtype=torch.uint8, device="cuda:0")
        for _ in range(4)
    ]
    frame_bytes = frames[0].element_size() * frames[0].nelement()
    ds = Dataset5dstem.from_frames(frames, stack_same_device=False)

    ds.page("auto", device=[0], max_vram_bytes=frame_bytes * 2 + 1)

    assert len(ds.vram_resident()) <= 2
    for target in (3, 2, 1, 0):
        frame = ds.frame(target)
        assert frame.device.type == "cuda"
        del frame
        resident = ds.vram_resident()
        assert target in resident
        assert len(resident) <= 2


def test_page_budget_preserves_existing_frame_devices_on_cpu():
    frames = [torch.full((2, 2, 3, 3), i, dtype=torch.uint8) for i in range(3)]
    ds = Dataset5dstem(frames=frames)

    ds.page(1, device=None)
    assert [str(device) for device in ds._page_devices] == ["cpu", "cpu", "cpu"]
    assert ds.vram_resident() == []
    assert ds.frame(2).device.type == "cpu"


def test_page_budget_accepts_explicit_device_sequence_on_cpu():
    frames = [torch.full((2, 2, 3, 3), i, dtype=torch.uint8) for i in range(3)]
    ds = Dataset5dstem(frames=frames)

    ds.page(1, device=["cpu"])

    assert [str(device) for device in ds._page_devices] == ["cpu", "cpu", "cpu"]
    assert ds.frame(1).device.type == "cpu"


def test_lazy_dataset_loads_only_requested_frame_on_cpu():
    calls = []

    def make_loader(i):
        def load():
            calls.append(i)
            return torch.full((2, 2, 3, 3), i, dtype=torch.uint8)

        return load

    ds = Dataset5dstem.from_lazy_loaders(
        [make_loader(i) for i in range(4)],
        shape=(4, 2, 2, 3, 3),
        dtype=torch.uint8,
    )

    assert ds.shape == (4, 2, 2, 3, 3)
    assert ds.dtype == torch.uint8
    assert calls == []

    assert int(ds.frame(2)[0, 0, 0, 0]) == 2
    assert calls == [2]
    assert int(ds.frame(2)[0, 0, 0, 0]) == 2
    assert calls == [2]
    assert int(ds.frame(0)[0, 0, 0, 0]) == 0
    assert calls == [2, 0]


def test_lazy_dataset_release_preserves_index_and_reloads_on_demand():
    calls = []

    def make_loader(i):
        def load():
            calls.append(i)
            return torch.full((2, 2, 3, 3), i, dtype=torch.uint8)

        return load

    ds = Dataset5dstem.from_lazy_loaders(
        [make_loader(i) for i in range(3)],
        shape=(3, 2, 2, 3, 3),
        dtype=torch.uint8,
    )

    assert int(ds.frame(1)[0, 0, 0, 0]) == 1
    assert ds.loaded_indices() == [1]
    released = ds.release(idx=[1])

    assert released == [1]
    assert ds.shape == (3, 2, 2, 3, 3)
    assert ds.loaded_indices() == []
    assert int(ds.frame(1)[0, 0, 0, 0]) == 1
    assert calls == [1, 1]


def test_lazy_preload_oom_keeps_partial_page_loaded():
    """preload should return the prefix that fits instead of raising OOM."""
    calls = []

    def make_loader(i):
        def load():
            calls.append(i)
            if i == 1:
                raise RuntimeError("Out of memory allocating 123 bytes")
            return torch.full((2, 2, 3, 3), i, dtype=torch.uint8)

        return load

    ds = Dataset5dstem.from_lazy_loaders(
        [make_loader(i) for i in range(3)],
        shape=(3, 2, 2, 3, 3),
        dtype=torch.uint8,
    )

    loaded = ds.preload([0, 1, 2])

    assert loaded == [0]
    assert calls == [0, 1]
    assert ds.loaded_indices() == [0]


def test_lazy_page_budget_evicts_before_loading_next_frame():
    calls = []
    observed_loaded_during_load = []
    ds = None

    def make_loader(i):
        def load():
            calls.append(i)
            if i == 2:
                observed_loaded_during_load.append(ds.loaded_indices())
            return torch.full((2, 2, 3, 3), i, dtype=torch.uint8)

        return load

    ds = Dataset5dstem.from_lazy_loaders(
        [make_loader(i) for i in range(3)],
        shape=(3, 2, 2, 3, 3),
        dtype=torch.uint8,
    )
    ds.page(2, device="cpu")

    assert int(ds.frame(0)[0, 0, 0, 0]) == 0
    assert int(ds.frame(1)[0, 0, 0, 0]) == 1
    assert ds.loaded_indices() == [0, 1]

    assert int(ds.frame(2)[0, 0, 0, 0]) == 2

    assert calls == [0, 1, 2]
    assert observed_loaded_during_load == [[1]]
    assert ds.loaded_indices() == [1, 2]


def test_lazy_dataset_validates_metadata_and_initial_frames():
    def load():
        return torch.zeros((2, 2, 3, 3), dtype=torch.uint8)

    with pytest.raises(ValueError, match="at least one"):
        Dataset5dstem.from_lazy_loaders([], shape=(0, 2, 2, 3, 3), dtype=torch.uint8)

    with pytest.raises(ValueError, match="out of range"):
        Dataset5dstem.from_lazy_loaders(
            [load],
            shape=(1, 2, 2, 3, 3),
            dtype=torch.uint8,
            initial_frames={1: load()},
        )

    with pytest.raises(ValueError, match="expected"):
        Dataset5dstem.from_lazy_loaders(
            [load],
            shape=(1, 2, 2, 3, 3),
            dtype=torch.uint8,
            initial_frames={0: torch.zeros((2, 2, 4, 4), dtype=torch.uint8)},
        )


def test_lazy_dataset_slice_and_subset_free_keep_lazy_metadata():
    calls = []

    def make_loader(i):
        def load():
            calls.append(i)
            return torch.full((2, 2, 3, 3), i, dtype=torch.uint8)

        return load

    ds = Dataset5dstem.from_lazy_loaders(
        [make_loader(i) for i in range(4)],
        shape=(4, 2, 2, 3, 3),
        dtype=torch.uint8,
        initial_frames={0: torch.zeros((2, 2, 3, 3), dtype=torch.uint8)},
    )

    sub = ds[1:4:2]

    assert sub.shape == (2, 2, 2, 3, 3)
    assert calls == []
    assert int(sub.frame(1)[0, 0, 0, 0]) == 3
    assert calls == [3]

    ds.free(idx=[0, 2])

    assert ds.shape == (2, 2, 2, 3, 3)
    assert len(ds._lazy_loaders) == 2
    assert int(ds.frame(1)[0, 0, 0, 0]) == 3
    assert calls == [3, 3]


def test_show4dstem_free_releases_dataset5dstem_on_cpu():
    """free() must release the Dataset5dstem owner, not just the widget ref."""
    frames = [torch.full((2, 2, 3, 3), i, dtype=torch.uint8) for i in range(2)]
    ds = Dataset5dstem(frames=frames)
    widget = Show4DSTEM(ds, verbose=False)

    widget.free()

    assert widget._data is None
    with pytest.raises(RuntimeError, match="Dataset5dstem has been freed"):
        len(ds)


def test_show4dstem_hiding_lazy_compare_panel_releases_and_skips_recompute():
    calls = []

    def make_loader(i):
        def load():
            calls.append(i)
            return torch.full((3, 3, 6, 6), i + 1, dtype=torch.uint8)

        return load

    ds = Dataset5dstem.from_lazy_loaders(
        [make_loader(i) for i in range(4)],
        shape=(4, 3, 3, 6, 6),
        dtype=torch.uint8,
        initial_frames={0: torch.ones((3, 3, 6, 6), dtype=torch.uint8)},
    )
    widget = Show4DSTEM(
        ds,
        view_mode="multiple",
        compare_max_panels=3,
        precompute_virtual_images=False,
        verbose=False,
    )

    try:
        assert widget.compare_panel_indices == [0, 1, 2]
        assert calls == [1, 2]
        assert ds.loaded_indices() == [0, 1, 2]

        widget.hide_compare_panel(1)

        assert widget.compare_hidden_panels == [1]
        assert 1 not in ds.loaded_indices()
        assert widget.compare_panel_indices == [0, 2]
        assert calls == [1, 2]

        widget.apply_preset("adf")

        assert 1 not in calls[3:]
        assert 3 not in calls
        assert widget.compare_panel_indices == [0, 2]
    finally:
        widget.close()


def test_show4dstem_lazy_compare_pages_load_only_active_page():
    calls = []

    def make_loader(i):
        def load():
            calls.append(i)
            return torch.full((3, 3, 6, 6), i + 1, dtype=torch.uint8)

        return load

    ds = Dataset5dstem.from_lazy_loaders(
        [make_loader(i) for i in range(7)],
        shape=(7, 3, 3, 6, 6),
        dtype=torch.uint8,
        initial_frames={0: torch.ones((3, 3, 6, 6), dtype=torch.uint8)},
    )
    widget = Show4DSTEM(
        ds,
        view_mode="multiple",
        compare_max_panels=3,
        page_budget=3,
        precompute_virtual_images=False,
        verbose=False,
    )

    try:
        assert widget.compare_panel_indices == [0, 1, 2]
        assert calls == [1, 2]
        assert ds.loaded_indices() == [0, 1, 2]

        widget.hide_compare_panel(1)

        assert widget.compare_panel_indices == [0, 2]
        assert calls == [1, 2]
        assert ds.loaded_indices() == [0, 2]
        assert "hidden" in widget.compare_status

        widget.set_compare_page(1)

        assert widget.compare_panel_indices == [3, 4, 5]
        assert calls == [1, 2, 3, 4, 5]
        assert ds.loaded_indices() == [3, 4, 5]

        widget.next_compare_page()

        assert widget.compare_panel_indices == [6]
        assert calls == [1, 2, 3, 4, 5, 6]
        assert ds.loaded_indices() == [4, 5, 6]
    finally:
        widget.close()


def test_show4dstem_compare_all_groups_batches_lazy_pages():
    """Collapsed compare groups show all tiles without keeping all raw frames loaded."""
    single_calls = []
    batch_calls = []

    def make_loader(i):
        def load():
            single_calls.append(i)
            return torch.full((3, 3, 6, 6), i + 1, dtype=torch.uint8)

        return load

    def batch_loader(indices):
        batch_calls.append(tuple(int(idx) for idx in indices))
        return {
            int(idx): torch.full((3, 3, 6, 6), int(idx) + 1, dtype=torch.uint8)
            for idx in indices
        }

    ds = Dataset5dstem.from_lazy_loaders(
        [make_loader(i) for i in range(7)],
        shape=(7, 3, 3, 6, 6),
        dtype=torch.uint8,
        initial_frames={0: torch.ones((3, 3, 6, 6), dtype=torch.uint8)},
        batch_loader=batch_loader,
    )
    widget = Show4DSTEM(
        ds,
        view_mode="multiple",
        compare_dp_mode="selected",
        compare_group_mode="all",
        compare_max_panels=3,
        compare_cache_pages=8,
        page_budget=3,
        precompute_virtual_images=False,
        verbose=False,
    )

    try:
        assert widget.compare_group_mode == "all"
        assert widget.compare_page_count == 3
        assert widget.compare_panel_count == 7
        assert widget.compare_panel_indices == list(range(7))
        assert "all groups" in widget.compare_status
        assert batch_calls
        assert all(len(call) <= 3 for call in batch_calls)
        assert len(ds.loaded_indices()) <= 3

        widget.show_compare_paged_groups()

        assert widget.compare_group_mode == "paged"
        assert widget.compare_panel_indices == [0, 1, 2]
        assert len(ds.loaded_indices()) <= 3
    finally:
        widget.close()


def test_show4dstem_compare_page_cache_reuses_reduced_virtual_images():
    """Returning to a page should reuse cached VI tiles, not reload every panel."""
    calls = []

    def make_loader(i):
        def load():
            calls.append(i)
            return torch.full((3, 3, 6, 6), i + 1, dtype=torch.uint8)

        return load

    ds = Dataset5dstem.from_lazy_loaders(
        [make_loader(i) for i in range(6)],
        shape=(6, 3, 3, 6, 6),
        dtype=torch.uint8,
        initial_frames={0: torch.ones((3, 3, 6, 6), dtype=torch.uint8)},
    )
    widget = Show4DSTEM(
        ds,
        view_mode="multiple",
        compare_dp_mode="selected",
        compare_max_panels=3,
        compare_cache_pages=4,
        page_budget=3,
        precompute_virtual_images=False,
        verbose=False,
    )

    try:
        assert widget.compare_panel_indices == [0, 1, 2]
        assert calls == [1, 2]
        assert len(widget._compare_virtual_page_cache) == 1

        widget.set_compare_page(1)

        assert widget.compare_panel_indices == [3, 4, 5]
        assert calls == [1, 2, 3, 4, 5]
        assert len(widget._compare_virtual_page_cache) == 2

        before_back = list(calls)
        widget.set_compare_page(0)

        assert widget.compare_panel_indices == [0, 1, 2]
        # The selected DP may reload panel 0, but cached VI tiles mean the page
        # does not reload/reduce panels 1 and 2 just to redraw thumbnails.
        assert all(idx not in calls[len(before_back) :] for idx in (1, 2))
    finally:
        widget.close()


def test_show4dstem_compare_average_dp_uses_cached_page_without_reload():
    """Cached compare pages should not reload full frames just to redraw average DP."""
    calls = []

    def make_loader(i):
        def load():
            calls.append(i)
            return torch.full((3, 3, 6, 6), i + 1, dtype=torch.uint8)

        return load

    ds = Dataset5dstem.from_lazy_loaders(
        [make_loader(i) for i in range(6)],
        shape=(6, 3, 3, 6, 6),
        dtype=torch.uint8,
        initial_frames={0: torch.ones((3, 3, 6, 6), dtype=torch.uint8)},
    )
    widget = Show4DSTEM(
        ds,
        view_mode="multiple",
        compare_dp_mode="average",
        compare_max_panels=3,
        compare_cache_pages=4,
        page_budget=3,
        precompute_virtual_images=False,
        verbose=False,
    )

    try:
        assert widget.compare_panel_indices == [0, 1, 2]
        assert calls == [1, 2]

        widget.set_compare_page(1)

        assert widget.compare_panel_indices == [3, 4, 5]
        assert calls == [1, 2, 3, 4, 5]

        before_back = list(calls)
        widget.set_compare_page(0)

        assert widget.compare_panel_indices == [0, 1, 2]
        assert calls == before_back
    finally:
        widget.close()


def test_show4dstem_uncached_compare_page_uses_lazy_batch_loader():
    """First visits to a compare page should load the visible group together."""
    single_calls = []
    batch_calls = []

    def make_loader(i):
        def load():
            single_calls.append(i)
            return torch.full((3, 3, 6, 6), i + 1, dtype=torch.uint8)

        return load

    def batch_loader(indices):
        batch_calls.append(tuple(indices))
        return {
            int(idx): torch.full((3, 3, 6, 6), int(idx) + 1, dtype=torch.uint8)
            for idx in indices
        }

    ds = Dataset5dstem.from_lazy_loaders(
        [make_loader(i) for i in range(6)],
        shape=(6, 3, 3, 6, 6),
        dtype=torch.uint8,
        initial_frames={0: torch.ones((3, 3, 6, 6), dtype=torch.uint8)},
        batch_loader=batch_loader,
    )
    widget = Show4DSTEM(
        ds,
        view_mode="multiple",
        compare_dp_mode="average",
        compare_max_panels=3,
        compare_cache_pages=4,
        page_budget=3,
        precompute_virtual_images=False,
        verbose=False,
    )

    try:
        assert widget.compare_panel_indices == [0, 1, 2]
        assert batch_calls == [(1, 2)]
        assert single_calls == []

        widget.set_compare_page(1)

        assert widget.compare_panel_indices == [3, 4, 5]
        assert batch_calls == [(1, 2), (3, 4, 5)]
        assert single_calls == []
        assert ds.loaded_indices() == [3, 4, 5]
    finally:
        widget.close()


def test_show4dstem_large_visible_page_uses_memory_bounded_sub_batches():
    batch_calls = []

    def make_loader(i):
        return lambda: torch.full((3, 3, 6, 6), i + 1, dtype=torch.uint8)

    def batch_loader(indices):
        batch_calls.append(tuple(indices))
        return {
            int(idx): torch.full((3, 3, 6, 6), int(idx) + 1, dtype=torch.uint8)
            for idx in indices
        }

    ds = Dataset5dstem.from_lazy_loaders(
        [make_loader(i) for i in range(5)],
        shape=(5, 3, 3, 6, 6),
        dtype=torch.uint8,
        initial_frames={0: torch.ones((3, 3, 6, 6), dtype=torch.uint8)},
        batch_loader=batch_loader,
    )
    widget = Show4DSTEM(
        ds,
        view_mode="multiple",
        compare_dp_mode="selected",
        compare_max_panels=5,
        page_budget=2,
        precompute_virtual_images=False,
        verbose=False,
    )

    try:
        assert widget.compare_panel_indices == [0, 1, 2, 3, 4]
        assert batch_calls == [(2, 3)]
        assert len(ds.loaded_indices()) <= 2
    finally:
        widget.close()


def test_show4dstem_warms_standard_presets_without_retaining_raw_pages():
    calls = []

    def make_loader(i):
        def load():
            calls.append(i)
            return torch.full((3, 3, 6, 6), i + 1, dtype=torch.uint8)

        return load

    ds = Dataset5dstem.from_lazy_loaders(
        [make_loader(i) for i in range(6)],
        shape=(6, 3, 3, 6, 6),
        dtype=torch.uint8,
        initial_frames={0: torch.ones((3, 3, 6, 6), dtype=torch.uint8)},
    )
    widget = Show4DSTEM(
        ds,
        view_mode="multiple",
        compare_dp_mode="selected",
        compare_max_panels=3,
        compare_cache_pages=4,
        page_budget=3,
        precompute_virtual_images=False,
        verbose=False,
    )

    try:
        widget.warm_compare_cache(("bf", "adf"), background=False)

        assert widget._compare_cache_warm_status == "ready"
        for page in ([0, 1, 2], [3, 4, 5]):
            assert widget._get_cached_compare_preset(
                page, preset_name="bf"
            ) is not None
            assert widget._get_cached_compare_preset(
                page, preset_name="adf"
            ) is not None
        assert len(ds.loaded_indices()) <= 3
    finally:
        widget.close()


def test_show4dstem_lazy_batch_oom_falls_back_to_single_loads():
    """Page changes should recover when a large backend batch exceeds VRAM."""
    single_calls = []
    batch_calls = []

    def make_loader(i):
        def load():
            single_calls.append(i)
            return torch.full((3, 3, 6, 6), i + 1, dtype=torch.uint8)

        return load

    def batch_loader(indices):
        batch_calls.append(tuple(indices))
        raise RuntimeError("Out of memory allocating 123 bytes")

    ds = Dataset5dstem.from_lazy_loaders(
        [make_loader(i) for i in range(6)],
        shape=(6, 3, 3, 6, 6),
        dtype=torch.uint8,
        initial_frames={0: torch.ones((3, 3, 6, 6), dtype=torch.uint8)},
        batch_loader=batch_loader,
    )
    widget = Show4DSTEM(
        ds,
        view_mode="multiple",
        compare_dp_mode="average",
        compare_max_panels=3,
        compare_cache_pages=4,
        page_budget=3,
        precompute_virtual_images=False,
        verbose=False,
    )

    try:
        assert widget.compare_panel_indices == [0, 1, 2]
        assert batch_calls == [(1, 2)]
        assert single_calls == [1, 2]

        widget.set_compare_page(1)

        assert widget.compare_panel_indices == [3, 4, 5]
        assert batch_calls == [(1, 2), (3, 4, 5)]
        assert single_calls == [1, 2, 3, 4, 5]
        assert ds.loaded_indices() == [3, 4, 5]
        assert "unavailable" not in widget.compare_status.lower()
    finally:
        widget.close()


def test_show4dstem_average_dp_oom_uses_cached_partial_frame():
    """Average DP mode should not crash the widget when a raw DP load hits OOM."""
    data = torch.ones((3, 3, 3, 4, 4), dtype=torch.uint8)
    widget = Show4DSTEM(
        data,
        view_mode="multiple",
        compare_dp_mode="average",
        compare_max_panels=3,
        precompute_virtual_images=False,
        verbose=False,
    )

    try:
        cached = np.full((4, 4), 7, dtype=np.float32)
        widget._get_cached_compare_diffraction_frame = (
            lambda idx: cached if int(idx) == 0 else None
        )

        def fail_load(idx):
            raise RuntimeError("Out of memory allocating 123 bytes")

        widget._diffraction_frame_for_index = fail_load

        frame = widget._average_compare_diffraction_frame()

        assert np.allclose(frame, cached)
        assert "GPU memory limited" in widget.memory_warning
        assert "Out of memory" not in widget.memory_warning
        assert "cleanup(w)" in widget.memory_warning
    finally:
        widget.close()


def test_show4dstem_average_dp_oom_without_cache_returns_blank_frame():
    """If no DP can be loaded, average mode should still provide a valid frame."""
    data = torch.ones((3, 3, 3, 4, 4), dtype=torch.uint8)
    widget = Show4DSTEM(
        data,
        view_mode="multiple",
        compare_dp_mode="average",
        compare_max_panels=3,
        precompute_virtual_images=False,
        verbose=False,
    )

    try:
        widget._get_cached_compare_diffraction_frame = lambda idx: None

        def fail_load(idx):
            raise RuntimeError("Out of memory allocating 123 bytes")

        widget._diffraction_frame_for_index = fail_load

        frame = widget._average_compare_diffraction_frame()

        assert frame.shape == (4, 4)
        assert frame.dtype == np.float32
        assert np.count_nonzero(frame) == 0
        assert "GPU memory limited" in widget.memory_warning
        assert "Out of memory" not in widget.memory_warning
        assert "cleanup(w)" in widget.memory_warning
    finally:
        widget.close()


def test_show4dstem_compare_grid_oom_keeps_partial_page():
    """A memory-limited compare page should show computed prefix panels."""
    data = torch.ones((3, 3, 3, 4, 4), dtype=torch.uint8)
    widget = Show4DSTEM(
        data,
        view_mode="multiple",
        compare_dp_mode="average",
        compare_max_panels=3,
        precompute_virtual_images=False,
        verbose=False,
    )

    try:
        widget._clear_compare_virtual_page_cache()

        def partial_tensor_image(idx, mask):
            if int(idx) == 0:
                return torch.ones((3, 3), dtype=torch.float32)
            raise RuntimeError("Out of memory allocating 123 bytes")

        widget._compare_virtual_image_tensor_for_frame = partial_tensor_image

        widget._refresh_compare_virtual_images()

        assert widget.compare_panel_indices == [0]
        assert widget.compare_panel_count == 1
        assert widget.compare_virtual_image_bytes
        assert "unavailable" not in widget.compare_status.lower()
        assert "memory limited" in widget.compare_status
        assert "Out of memory" not in widget.compare_status
        assert "GPU memory limited" in widget.memory_warning
        assert "Out of memory" not in widget.memory_warning
        assert "cleanup(w)" in widget.memory_warning
    finally:
        widget.close()


def test_show4dstem_compare_grid_oom_reports_gracefully_without_raw_trace():
    """A fully memory-limited compare page should show corrective user guidance."""
    data = torch.ones((3, 3, 3, 4, 4), dtype=torch.uint8)
    widget = Show4DSTEM(
        data,
        view_mode="multiple",
        compare_dp_mode="average",
        compare_max_panels=3,
        precompute_virtual_images=False,
        verbose=False,
    )

    try:
        widget._clear_compare_virtual_page_cache()

        def fail_tensor_image(idx, mask):
            raise RuntimeError("Out of memory allocating 123 bytes")

        widget._compare_virtual_image_tensor_for_frame = fail_tensor_image

        widget._refresh_compare_virtual_images()

        assert widget.compare_panel_indices == []
        assert widget.compare_panel_count == 0
        assert widget.compare_virtual_image_bytes == b""
        assert (
            widget.compare_status
            == "GPU memory limited: multiple grid page was not computed."
        )
        assert "Out of memory" not in widget.compare_status
        assert "GPU memory limited" in widget.memory_warning
        assert "Out of memory" not in widget.memory_warning
        assert "cleanup(w)" in widget.memory_warning
    finally:
        widget.close()


# --- ShowFolder -> Show4DSTEM handoff -----------------------------------------


def _stub_browser(folder):
    from quantem.widget.showfolder_core import ShowFolderBrowser

    class _Stub(ShowFolderBrowser):
        def __init__(self, f):
            self.folder = f

    return _Stub(folder)


def test_showfolder_open_show4dstem_preserves_loader_device_when_gpus_none(
    monkeypatch, tmp_path
):
    """open_show4dstem(gpus=None) must not force CUDA on CPU/MPS loaders."""
    import quantem.widget.io as wio
    import quantem.widget as qw

    fake_masters = [str(tmp_path / f"scan_{i:02d}_master.h5") for i in range(2)]

    def fake_discover(folder, *, scan_shape=None, verbose=False, **kw):
        return list(fake_masters)

    def fake_ready(path):
        return True

    class _Result:
        def __init__(self, path):
            # distinct value per master so frames are not aliased
            v = int(path.split("scan_")[1][:2])
            self.data = torch.full((4, 4, 6, 6), v, dtype=torch.uint8)

    def fake_load(path, *, det_bin=4, dtype="u8", verbose=False, **kw):
        return _Result(path)

    monkeypatch.setattr(wio, "discover_masters", fake_discover)
    monkeypatch.setattr(wio, "is_master_ready", fake_ready)
    monkeypatch.setattr(qw, "load", fake_load)

    sf = _stub_browser(tmp_path)
    w = sf.open_show4dstem(gpus=None, page_budget=1, det_bin=4, dtype="u8")
    assert w is not None
    ds = w._data
    assert w.n_frames == 2
    assert list(w.frame_labels) == [f"scan_{i:02d}" for i in range(2)]
    assert [frame.device.type for frame in ds.frames] == ["cpu", "cpu"]
    w.frame_idx = 1
    _ = w._frame_data
    assert [frame.device.type for frame in ds.frames] == ["cpu", "cpu"]


def test_showfolder_open_show4dstem_is_lazy_after_initial_frame(monkeypatch, tmp_path):
    """Opening a master folder must not load every dataset hot."""
    import quantem.widget.io as wio
    import quantem.widget as qw

    fake_masters = [str(tmp_path / f"scan_{i:02d}_master.h5") for i in range(4)]
    calls = []

    def fake_discover(folder, *, scan_shape=None, verbose=False, **kw):
        return list(fake_masters)

    def fake_ready(path):
        return True

    class _Result:
        def __init__(self, path):
            v = int(path.split("scan_")[1][:2])
            self.data = torch.full((4, 4, 6, 6), v, dtype=torch.uint8)

    def fake_load(path, *, det_bin=4, dtype="u8", verbose=False, **kw):
        calls.append(path)
        return _Result(path)

    monkeypatch.setattr(wio, "discover_masters", fake_discover)
    monkeypatch.setattr(wio, "is_master_ready", fake_ready)
    monkeypatch.setattr(qw, "load", fake_load)

    sf = _stub_browser(tmp_path)
    w = sf.open_show4dstem(gpus=None, page_budget="auto", det_bin=4, dtype="u8")

    assert w is not None
    assert w.n_frames == 4
    assert calls == [fake_masters[0]]
    loaded = [idx for idx, frame in enumerate(w._data._frames) if frame is not None]
    assert loaded == [0]

    w.frame_idx = 3
    _ = w._frame_data

    assert calls == [fake_masters[0], fake_masters[3]]
    loaded = [idx for idx, frame in enumerate(w._data._frames) if frame is not None]
    assert loaded == [0, 3]


def test_show4dstem_from_folder_builds_lazy_widget_and_poll_appends(
    monkeypatch, tmp_path
):
    import quantem.widget as qw
    import quantem.widget.io as wio

    masters = [str(tmp_path / f"scan_{i:02d}_master.h5") for i in range(2)]
    calls = []
    discovered = [masters[0]]

    def fake_discover(
        folder,
        *,
        pattern="*_master.h5",
        recursive=True,
        scan_shape=None,
        verbose=False,
        **kw,
    ):
        return list(discovered)

    def fake_ready(path):
        return True

    def fake_load(path, *, det_bin=4, dtype="u8", verbose=False, **kw):
        calls.append(path)
        value = int(path.split("scan_")[1][:2]) + 1
        return LoadResult(torch.full((3, 3, 6, 6), value, dtype=torch.uint8), {})

    monkeypatch.setattr(wio, "discover_masters", fake_discover)
    monkeypatch.setattr(wio, "is_master_ready", fake_ready)
    monkeypatch.setattr(
        wio,
        "get_metadata",
        lambda path: {
            "scan_shape": (3, 3),
            "detector_shape": (24, 24),
            "n_frames": 9,
            "dtype": "uint8",
        },
    )
    monkeypatch.setattr(qw, "load", fake_load)

    widget = Show4DSTEM.from_folder(
        tmp_path,
        gpus=None,
        page_budget=1,
        det_bin=4,
        dtype="u8",
        view_mode="single",
        precompute_virtual_images=False,
        verbose=False,
    )

    try:
        assert widget.n_frames == 1
        assert list(widget.frame_labels) == ["scan_00"]
        assert calls == [masters[0]]

        discovered.append(masters[1])
        added = widget.poll_folder()

        assert added == [1]
        assert widget.n_frames == 2
        assert list(widget.frame_labels) == ["scan_00", "scan_01"]
        assert calls == [masters[0]]

        widget.frame_idx = 1
        assert int(widget._frame_data[0, 0, 0, 0]) == 2
        assert calls == [masters[0], masters[1]]
    finally:
        widget.stop_folder_watch()
        widget.close()


def test_show4dstem_from_folder_watches_by_default_and_appends_cold(
    monkeypatch, tmp_path
):
    import quantem.widget as qw
    import quantem.widget.io as wio

    masters = [str(tmp_path / f"scan_{idx:02d}_master.h5") for idx in range(3)]
    discovered = [masters[0]]
    calls = []

    monkeypatch.setattr(
        wio, "discover_masters", lambda *args, **kwargs: list(discovered)
    )
    monkeypatch.setattr(wio, "is_master_ready", lambda path: True)
    monkeypatch.setattr(
        wio,
        "get_metadata",
        lambda path: {
            "scan_shape": (3, 3),
            "detector_shape": (24, 24),
            "n_frames": 9,
            "dtype": "uint8",
        },
    )

    def fake_load(path, *, det_bin=4, dtype="u8", verbose=False, **kwargs):
        calls.append(str(path))
        value = int(str(path).split("scan_")[1][:2]) + 1
        return LoadResult(torch.full((3, 3, 6, 6), value, dtype=torch.uint8), {})

    monkeypatch.setattr(qw, "load", fake_load)
    widget = Show4DSTEM.from_folder(
        tmp_path,
        gpus=None,
        page_budget=1,
        det_bin=4,
        dtype="u8",
        view_mode="single",
        preload_all_if_fits=False,
        watch_interval=0.01,
        precompute_virtual_images=False,
    )
    watcher = widget._folder_watch_thread

    try:
        assert watcher is not None and watcher.is_alive()
        assert calls == [masters[0]]
        assert widget._data.loaded_indices() == [0]

        discovered.append(masters[1])
        _wait_until(lambda: widget.n_frames == 2)

        assert calls == [masters[0]]
        assert widget._data.loaded_indices() == [0]

        widget.frame_idx = 1
        _ = widget._frame_data
        assert calls == [masters[0], masters[1]]
        assert widget._data.loaded_indices() == [1]

        discovered.append(masters[2])
        _wait_until(lambda: widget.n_frames == 3)
        assert calls == [masters[0], masters[1]]

        widget.frame_idx = 2
        _ = widget._frame_data
        assert calls == masters
        assert widget._data.loaded_indices() == [2]
    finally:
        widget.free()
        assert widget._folder_watch_thread is None
        widget.close()

    assert not watcher.is_alive()


def test_show4dstem_watched_master_contract_retries_and_registers_batch_paths(
    monkeypatch, tmp_path
):
    import quantem.widget as qw
    import quantem.widget.io as wio

    masters = [str(tmp_path / f"scan_{idx:02d}_master.h5") for idx in range(4)]
    discovered = [masters[0]]
    calls = []
    contracts = {
        master: {
            "scan_shape": (3, 3),
            "detector_shape": (24, 24),
            "n_frames": 9,
            "dtype": "uint8",
        }
        for master in masters
    }

    monkeypatch.setattr(
        wio, "discover_masters", lambda *args, **kwargs: list(discovered)
    )
    monkeypatch.setattr(wio, "is_master_ready", lambda path: True)
    monkeypatch.setattr(wio, "get_metadata", lambda path: dict(contracts[str(path)]))

    def fake_load(path, *, det_bin=4, dtype="u8", verbose=False, **kwargs):
        calls.append(str(path))
        value = int(str(path).split("scan_")[1][:2]) + 1
        return LoadResult(torch.full((3, 3, 6, 6), value, dtype=torch.uint8), {})

    monkeypatch.setattr(qw, "load", fake_load)
    widget = Show4DSTEM.from_folder(
        tmp_path,
        gpus=None,
        page_budget=2,
        det_bin=4,
        dtype="u8",
        view_mode="single",
        watch=False,
        preload_all_if_fits=False,
        precompute_virtual_images=False,
    )

    try:
        discovered.extend(masters[1:3])
        assert widget.poll_folder() == [1, 2]
        assert widget.n_frames == 3
        assert calls == [masters[0]]

        # The optimized batch callback must resolve watched indices against the
        # extended master list before falling back to CPU single-frame loaders.
        widget._data.preload([1, 2])
        assert calls == masters[:3]
        assert len(widget._data.loaded_indices()) <= 2

        contracts[masters[3]]["detector_shape"] = (20, 24)
        discovered.append(masters[3])
        labels_before = list(widget.frame_labels)
        known_before = set(widget._folder_known_masters)

        assert widget.poll_folder() == []
        assert widget.n_frames == 3
        assert list(widget.frame_labels) == labels_before
        assert widget._folder_known_masters == known_before

        contracts[masters[3]]["detector_shape"] = (24, 24)
        assert widget.poll_folder() == [3]
        assert widget.n_frames == 4
        assert list(widget.frame_labels)[-1] == "scan_03"
        assert calls == masters[:3]
    finally:
        widget.close()


def test_show4dstem_watcher_warms_only_new_pages_without_blocking_cached_page(
    monkeypatch, tmp_path
):
    import quantem.widget as qw
    import quantem.widget.io as wio

    masters = [str(tmp_path / f"scan_{idx:02d}_master.h5") for idx in range(4)]
    discovered = list(masters[:2])
    calls = []
    block_new = threading.Event()
    new_load_started = threading.Event()
    release_new_load = threading.Event()

    monkeypatch.setattr(
        wio, "discover_masters", lambda *args, **kwargs: list(discovered)
    )
    monkeypatch.setattr(wio, "is_master_ready", lambda path: True)
    monkeypatch.setattr(
        wio,
        "get_metadata",
        lambda path: {
            "scan_shape": (3, 3),
            "detector_shape": (24, 24),
            "n_frames": 9,
            "dtype": "uint8",
        },
    )

    def fake_load(path, *, det_bin=4, dtype="u8", verbose=False, **kwargs):
        path = str(path)
        calls.append(path)
        if path == masters[2] and block_new.is_set():
            new_load_started.set()
            if not release_new_load.wait(timeout=3):
                raise RuntimeError("test did not release watched master load")
        value = int(path.split("scan_")[1][:2]) + 1
        return LoadResult(torch.full((3, 3, 6, 6), value, dtype=torch.uint8), {})

    monkeypatch.setattr(qw, "load", fake_load)
    widget = Show4DSTEM.from_folder(
        tmp_path,
        gpus=None,
        page_budget=2,
        det_bin=4,
        dtype="u8",
        page_size=2,
        compare_dp_mode="selected",
        warm_cache=True,
        preload_all_if_fits=False,
        watch_interval=0.01,
        precompute_virtual_images=False,
    )

    page0 = [0, 1]
    presets = ("bf", "abf", "adf", "haadf")
    try:
        _wait_until(
            lambda: all(
                widget._get_cached_compare_preset(page0, preset_name=name)
                is not None
                for name in presets
            )
        )
        old_payloads = {
            name: widget._get_cached_compare_preset(page0, preset_name=name)[0]
            for name in presets
        }
        old_call_counts = {master: calls.count(master) for master in masters[:2]}

        block_new.set()
        discovered.extend(masters[2:])
        _wait_until(lambda: widget.n_frames == 4)
        assert new_load_started.wait(timeout=3)

        started = time.monotonic()
        widget._refresh_compare_virtual_images()
        assert time.monotonic() - started < 0.25
        assert widget.compare_panel_indices == page0
        for name, payload in old_payloads.items():
            cached = widget._get_cached_compare_preset(page0, preset_name=name)
            assert cached is not None and cached[0] == payload

        release_new_load.set()
        page1 = [2, 3]
        _wait_until(
            lambda: all(
                widget._get_cached_compare_preset(page1, preset_name=name)
                is not None
                for name in presets
            )
        )
        _wait_until(lambda: widget._compare_cache_warm_status == "ready")

        assert {master: calls.count(master) for master in masters[:2]} == old_call_counts
        assert len(widget._data.loaded_indices()) <= 2
    finally:
        release_new_load.set()
        widget.close()


def test_show4dstem_from_folder_auto_dtype_uses_stable_u16(monkeypatch, tmp_path):
    import quantem.widget as qw
    import quantem.widget.io as wio

    masters = [str(tmp_path / f"scan_{i:02d}_master.h5") for i in range(2)]
    dtypes = []

    def fake_discover(
        folder,
        *,
        pattern="*_master.h5",
        recursive=True,
        scan_shape=None,
        verbose=False,
        **kw,
    ):
        return list(masters)

    def fake_ready(path):
        return True

    def fake_load(path, *, det_bin=4, dtype="u8", verbose=False, **kw):
        dtypes.append(dtype)
        value = int(path.split("scan_")[1][:2]) + 1
        return LoadResult(torch.full((3, 3, 6, 6), value, dtype=torch.uint16), {})

    monkeypatch.setattr(wio, "discover_masters", fake_discover)
    monkeypatch.setattr(wio, "is_master_ready", fake_ready)
    monkeypatch.setattr(qw, "load", fake_load)

    widget = Show4DSTEM.from_folder(
        tmp_path,
        gpus=None,
        page_budget=1,
        det_bin=4,
        dtype="auto",
        view_mode="single",
        precompute_virtual_images=False,
        verbose=False,
    )

    try:
        assert dtypes == ["u16"]
        widget.frame_idx = 1
        assert int(widget._frame_data[0, 0, 0, 0]) == 2
        assert dtypes == ["u16", "u16"]
    finally:
        widget.stop_folder_watch()
        widget.close()


def test_show4dstem_from_folder_skips_unreadable_masters(monkeypatch, tmp_path):
    import quantem.widget as qw
    import quantem.widget.io as wio

    masters = [str(tmp_path / f"scan_{i:02d}_master.h5") for i in range(3)]
    calls = []

    def fake_discover(
        folder,
        *,
        pattern="*_master.h5",
        recursive=True,
        scan_shape=None,
        verbose=False,
        **kw,
    ):
        return list(masters)

    def fake_ready(path):
        return path != masters[1]

    def fake_load(path, *, det_bin=4, dtype="u8", verbose=False, **kw):
        calls.append(path)
        value = int(path.split("scan_")[1][:2]) + 1
        return LoadResult(torch.full((3, 3, 6, 6), value, dtype=torch.uint16), {})

    monkeypatch.setattr(wio, "discover_masters", fake_discover)
    monkeypatch.setattr(wio, "is_master_ready", fake_ready)
    monkeypatch.setattr(qw, "load", fake_load)

    with pytest.warns(
        RuntimeWarning, match="skipped 1 incomplete or unreadable master file"
    ):
        widget = Show4DSTEM.from_folder(
            tmp_path,
            gpus=None,
            page_budget=1,
            det_bin=4,
            dtype="auto",
            view_mode="single",
            precompute_virtual_images=False,
            verbose=True,
        )

    try:
        assert widget.n_frames == 2
        assert list(widget.frame_labels) == ["scan_00", "scan_02"]
        assert calls == [masters[0]]
        widget.frame_idx = 1
        assert int(widget._frame_data[0, 0, 0, 0]) == 3
        assert calls == [masters[0], masters[2]]
    finally:
        widget.stop_folder_watch()
        widget.close()


def test_show4dstem_from_folder_is_quiet_by_default(monkeypatch, tmp_path, capsys):
    import quantem.widget as qw
    import quantem.widget.io as wio

    masters = [str(tmp_path / f"scan_{i:02d}_master.h5") for i in range(3)]

    monkeypatch.setattr(wio, "discover_masters", lambda *args, **kwargs: list(masters))
    monkeypatch.setattr(wio, "is_master_ready", lambda path: path != masters[1])
    monkeypatch.setattr(
        qw,
        "load",
        lambda *args, **kwargs: LoadResult(
            torch.ones((3, 3, 6, 6), dtype=torch.uint16),
            {},
        ),
    )

    with warnings.catch_warnings(record=True) as caught:
        widget = Show4DSTEM.from_folder(
            tmp_path,
            gpus=None,
            view_mode="single",
            precompute_virtual_images=False,
        )

    try:
        assert caught == []
        assert capsys.readouterr().out == ""
        assert widget.n_frames == 2
    finally:
        widget.stop_folder_watch()
        widget.close()


def test_show4dstem_from_folder_accepts_simple_grid_names(monkeypatch, tmp_path):
    import quantem.widget as qw
    import quantem.widget.io as wio

    masters = [str(tmp_path / f"scan_{i:02d}_master.h5") for i in range(3)]

    monkeypatch.setattr(wio, "discover_masters", lambda *args, **kwargs: list(masters))
    monkeypatch.setattr(wio, "is_master_ready", lambda path: True)
    monkeypatch.setattr(
        qw,
        "load",
        lambda *args, **kwargs: LoadResult(
            torch.ones((3, 3, 6, 6), dtype=torch.uint16),
            {},
        ),
    )

    widget = Show4DSTEM.from_folder(
        tmp_path,
        gpus=None,
        columns=2,
        page_size=2,
        warm_cache=True,
        compare_dp_mode="selected",
        precompute_virtual_images=False,
    )

    try:
        thread = widget._compare_cache_warm_thread
        if thread is not None:
            thread.join(timeout=5)
        assert widget.compare_cols == 2
        assert widget.compare_max_panels == 2
        assert widget._compare_cache_warm_status == "ready"
    finally:
        widget.stop_folder_watch()
        widget.close()


def test_show4dstem_from_folder_uses_largest_compatible_metadata_group(
    monkeypatch, tmp_path
):
    import quantem.widget as qw
    import quantem.widget.io as wio

    masters = [str(tmp_path / f"scan_{i:02d}_master.h5") for i in range(5)]
    calls = []

    def fake_discover(
        folder,
        *,
        pattern="*_master.h5",
        recursive=True,
        scan_shape=None,
        verbose=False,
        **kw,
    ):
        return list(masters)

    def fake_ready(path):
        return True

    def fake_metadata(path):
        idx = int(path.split("scan_")[1][:2])
        if idx < 3:
            return {
                "scan_shape": (512, 512),
                "detector_shape": (192, 192),
                "n_frames": 512 * 512,
            }
        return {
            "scan_shape": (256, 256),
            "detector_shape": (192, 192),
            "n_frames": 256 * 256,
        }

    def fake_load(path, *, det_bin=4, dtype="u8", verbose=False, **kw):
        calls.append(path)
        value = int(path.split("scan_")[1][:2]) + 1
        return LoadResult(torch.full((3, 3, 6, 6), value, dtype=torch.uint16), {})

    monkeypatch.setattr(wio, "discover_masters", fake_discover)
    monkeypatch.setattr(wio, "is_master_ready", fake_ready)
    monkeypatch.setattr(wio, "get_metadata", fake_metadata)
    monkeypatch.setattr(qw, "load", fake_load)

    with pytest.warns(RuntimeWarning, match="mixed 4D-STEM shapes"):
        widget = Show4DSTEM.from_folder(
            tmp_path,
            gpus=None,
            page_budget=1,
            det_bin=4,
            dtype="auto",
            view_mode="single",
            precompute_virtual_images=False,
            verbose=True,
        )

    try:
        assert widget.n_frames == 3
        assert list(widget.frame_labels) == ["scan_00", "scan_01", "scan_02"]
        assert calls == [masters[0]]
    finally:
        widget.stop_folder_watch()
        widget.close()


def test_show4dstem_from_folder_first_load_oom_returns_warning_widget(
    monkeypatch, tmp_path
):
    import quantem.widget as qw
    import quantem.widget.io as wio

    class FakeCudaOom(BaseException):
        pass

    masters = [str(tmp_path / "scan_00_master.h5")]

    def fake_discover(
        folder,
        *,
        pattern="*_master.h5",
        recursive=True,
        scan_shape=None,
        verbose=False,
        **kw,
    ):
        return list(masters)

    def fake_ready(path):
        return True

    def fake_load(path, *, det_bin=4, dtype="u8", verbose=False, **kw):
        raise FakeCudaOom("Out of memory allocating 123 bytes")

    monkeypatch.setattr(wio, "discover_masters", fake_discover)
    monkeypatch.setattr(wio, "is_master_ready", fake_ready)
    monkeypatch.setattr(qw, "load", fake_load)

    widget = Show4DSTEM.from_folder(
        tmp_path,
        gpus=None,
        page_budget=1,
        det_bin=4,
        dtype="u8",
        view_mode="multiple",
        precompute_virtual_images=False,
        verbose=False,
    )

    try:
        assert widget.n_frames == 1
        assert widget.title.endswith("(not loaded)")
        assert (
            widget.compare_status == "GPU memory limited: folder data was not loaded."
        )
        assert "GPU memory limited" in widget.memory_warning
        assert "Out of memory" not in widget.memory_warning
        assert "cleanup(w)" in widget.memory_warning
    finally:
        widget.close()


def test_lazy_residency_plan_uses_known_shape_across_all_page_devices():
    frame_shape = (2, 3, 4, 5)
    frame_bytes = 2 * 3 * 4 * 5
    loaders = [lambda: (_ for _ in ()).throw(AssertionError("must stay lazy")) for _ in range(6)]
    data = Dataset5dstem.from_lazy_loaders(
        loaders,
        shape=(6, *frame_shape),
        dtype=torch.uint8,
    )
    data.page(
        "auto",
        device=["cuda:0", "cuda:1"],
        max_vram_bytes={
            "cuda:0": 3 * frame_bytes,
            "cuda:1": 3 * frame_bytes,
        },
    )

    plan = data.residency_plan()

    assert plan["fits"] is True
    assert plan["total_required_bytes"] == 6 * frame_bytes
    assert plan["required_bytes"] == {
        torch.device("cuda:0"): 3 * frame_bytes,
        torch.device("cuda:1"): 3 * frame_bytes,
    }
    assert data.loaded_indices() == []


def test_lazy_residency_plan_keeps_paging_when_one_gpu_is_over_budget():
    frame_shape = (2, 3, 4, 5)
    frame_bytes = 2 * 3 * 4 * 5
    data = Dataset5dstem.from_lazy_loaders(
        [lambda: torch.zeros(frame_shape, dtype=torch.uint8) for _ in range(6)],
        shape=(6, *frame_shape),
        dtype=torch.uint8,
    )
    data.page(
        "auto",
        device=["cuda:0", "cuda:1"],
        max_vram_bytes={
            "cuda:0": 3 * frame_bytes,
            "cuda:1": 3 * frame_bytes - 1,
        },
    )

    assert data.residency_plan()["fits"] is False


def test_auto_vram_budget_uses_adaptive_margin_instead_of_fixed_four_gib(
    monkeypatch,
):
    frame_shape = (1, 1, 1024, 1024)
    data = Dataset5dstem.from_lazy_loaders(
        [lambda: torch.zeros(frame_shape, dtype=torch.uint8)],
        shape=(1, *frame_shape),
        dtype=torch.uint8,
    )
    total = 100 * 1024**3
    free = 90 * 1024**3

    class _CudaDeviceContext:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(torch.cuda, "device", lambda *args, **kwargs: _CudaDeviceContext())
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args, **kwargs: (free, total))

    data.page("auto", device=["cuda:0"])

    budget = data._page_max_vram_bytes[torch.device("cuda:0")]
    assert budget == free - 1024**3
    assert budget > free - 4 * 1024**3


def test_append_lazy_frame_preserves_round_robin_page_devices():
    frame_shape = (2, 3, 4, 5)
    data = Dataset5dstem.from_lazy_loaders(
        [lambda: torch.zeros(frame_shape, dtype=torch.uint8) for _ in range(2)],
        shape=(2, *frame_shape),
        dtype=torch.uint8,
    )
    data.page(
        "auto",
        device=["cuda:0", "cuda:1"],
        max_vram_bytes=10_000,
    )

    data.append_lazy_frame(lambda: torch.zeros(frame_shape, dtype=torch.uint8))
    data.append_lazy_frame(lambda: torch.zeros(frame_shape, dtype=torch.uint8))

    assert data.devices == ["cuda:0", "cuda:1", "cuda:0", "cuda:1"]


@cuda_required
def test_release_returns_independent_lazy_frame_vram_immediately():
    frame_shape = (64, 1024, 1024, 1)
    data = Dataset5dstem.from_lazy_loaders(
        [lambda: torch.empty(frame_shape, dtype=torch.uint8, device="cuda:0")],
        shape=(1, *frame_shape),
        dtype=torch.uint8,
        initial_frames={
            0: torch.empty(frame_shape, dtype=torch.uint8, device="cuda:0")
        },
    )
    torch.cuda.synchronize(0)
    free_before, _ = torch.cuda.mem_get_info(0)

    assert data.release(idx=0) == [0]
    torch.cuda.synchronize(0)
    free_after, _ = torch.cuda.mem_get_info(0)

    assert free_after - free_before >= 48 * 1024**2


@cuda_required
def test_show4dstem_from_folder_preloads_complete_series_when_it_fits(
    monkeypatch, tmp_path
):
    import quantem.widget as qw
    import quantem.widget.io as wio

    masters = [str(tmp_path / f"scan_{idx:02d}_master.h5") for idx in range(4)]
    calls = []
    frame_shape = (8, 8, 12, 12)
    frame_bytes = int(np.prod(frame_shape))

    monkeypatch.setattr(wio, "discover_masters", lambda *args, **kwargs: list(masters))
    monkeypatch.setattr(wio, "is_master_ready", lambda path: True)

    def fake_load(path, *, det_bin=4, dtype="u8", verbose=False, **kwargs):
        calls.append(path)
        paths = list(path) if isinstance(path, list) else [path]
        frames = [
            torch.full(
                frame_shape,
                int(item.split("scan_")[1][:2]) + 1,
                dtype=torch.uint8,
                device="cuda:0",
            )
            for item in paths
        ]
        data = torch.stack(frames) if isinstance(path, list) else frames[0]
        return LoadResult(data, {})

    monkeypatch.setattr(qw, "load", fake_load)

    widget = Show4DSTEM.from_folder(
        tmp_path,
        gpus=[0],
        page_budget="auto",
        page_max_vram_bytes=frame_bytes * len(masters),
        det_bin=4,
        dtype="u8",
        view_mode="single",
        preload_initial_page=False,
        precompute_virtual_images=False,
    )
    try:
        widget.wait_for_dataset_preload(timeout=10)
        assert widget._data.vram_resident() == list(range(len(masters)))
        assert widget._raw_preload_status == "resident"
        assert "raw 4/4 resident" in widget.gpu_memory_label
        assert calls == masters
    finally:
        widget.close()


@cuda_required
def test_show4dstem_from_folder_does_not_preload_series_over_budget(
    monkeypatch, tmp_path
):
    import quantem.widget as qw
    import quantem.widget.io as wio

    masters = [str(tmp_path / f"scan_{idx:02d}_master.h5") for idx in range(4)]
    calls = []
    frame_shape = (8, 8, 12, 12)
    frame_bytes = int(np.prod(frame_shape))

    monkeypatch.setattr(wio, "discover_masters", lambda *args, **kwargs: list(masters))
    monkeypatch.setattr(wio, "is_master_ready", lambda path: True)

    def fake_load(path, *, det_bin=4, dtype="u8", verbose=False, **kwargs):
        calls.append(path)
        value = int(path.split("scan_")[1][:2]) + 1
        return LoadResult(
            torch.full(frame_shape, value, dtype=torch.uint8, device="cuda:0"),
            {},
        )

    monkeypatch.setattr(qw, "load", fake_load)

    widget = Show4DSTEM.from_folder(
        tmp_path,
        gpus=[0],
        page_budget="auto",
        page_max_vram_bytes=frame_bytes * 2,
        det_bin=4,
        dtype="u8",
        view_mode="single",
        preload_initial_page=False,
        precompute_virtual_images=False,
    )
    try:
        widget.wait_for_dataset_preload(timeout=10)
        assert widget._raw_preload_status == "paged"
        assert widget._data.vram_resident() == [0]
        assert calls == [masters[0]]
    finally:
        widget.close()


@cuda_required
def test_showfolder_open_show4dstem_builds_paged_multimaster_on_explicit_cuda(
    monkeypatch, tmp_path
):
    """open_show4dstem(gpus=[...]) pages lazy masters into one Show4DSTEM."""
    import quantem.widget.io as wio
    import quantem.widget as qw

    fake_masters = [str(tmp_path / f"scan_{i:02d}_master.h5") for i in range(4)]
    calls = []

    def fake_discover(folder, *, scan_shape=None, verbose=False, **kw):
        return list(fake_masters)

    def fake_ready(path):
        return True

    class _Result:
        def __init__(self, path):
            v = int(path.split("scan_")[1][:2])
            self.data = torch.full(
                (16, 16, 24, 24), v, dtype=torch.uint8, device="cuda:0"
            )

    def fake_load(path, *, det_bin=4, dtype="u8", verbose=False, **kw):
        calls.append(path)
        return _Result(path)

    monkeypatch.setattr(wio, "discover_masters", fake_discover)
    monkeypatch.setattr(wio, "is_master_ready", fake_ready)
    monkeypatch.setattr(qw, "load", fake_load)

    sf = _stub_browser(tmp_path)
    w = sf.open_show4dstem(gpus=[0], page_budget=1, det_bin=4, dtype="u8")
    assert w is not None
    ds = w._data
    assert w.n_frames == 4
    assert list(w.frame_labels) == [f"scan_{i:02d}" for i in range(4)]
    assert calls == [fake_masters[0]]
    assert ds.vram_resident() == [0]
    w.frame_idx = 3
    _ = w._frame_data
    assert calls == [fake_masters[0], fake_masters[3]]
    assert ds.vram_resident() == [3]
    assert ds._frames[0] is None
    w.frame_idx = 0
    _ = w._frame_data
    assert calls == [fake_masters[0], fake_masters[3], fake_masters[0]]
    assert ds.vram_resident() == [0]


@cuda_required
def test_showfolder_open_show4dstem_auto_uses_gpu_sized_cache(monkeypatch, tmp_path):
    """Auto paging keeps hot datasets until the byte budget, then LRU-evicts."""
    import quantem.widget as qw
    import quantem.widget.io as wio

    fake_masters = [str(tmp_path / f"scan_{i:02d}_master.h5") for i in range(5)]

    def fake_discover(folder, *, scan_shape=None, verbose=False, **kw):
        return list(fake_masters)

    def fake_ready(path):
        return True

    class _Result:
        def __init__(self, path):
            v = int(path.split("scan_")[1][:2])
            self.data = torch.full(
                (16, 16, 24, 24), v, dtype=torch.uint8, device="cuda:0"
            )

    def fake_load(path, *, det_bin=4, dtype="u8", verbose=False, **kw):
        return _Result(path)

    monkeypatch.setattr(wio, "discover_masters", fake_discover)
    monkeypatch.setattr(wio, "is_master_ready", fake_ready)
    monkeypatch.setattr(qw, "load", fake_load)

    frame_bytes = 16 * 16 * 24 * 24
    sf = _stub_browser(tmp_path)
    w = sf.open_show4dstem(
        gpus=[0],
        page_budget="auto",
        page_max_vram_bytes=frame_bytes * 2 + 1,
        det_bin=4,
        dtype="u8",
    )

    assert w is not None
    ds = w._data
    assert w.n_frames == 5
    assert 0 in ds.vram_resident()
    assert len(ds.vram_resident()) <= 2
    w.frame_idx = 4
    frame = w._frame_data
    assert frame.device.type == "cuda"
    del frame
    assert 4 in ds.vram_resident()
    assert len(ds.vram_resident()) <= 2


@cuda_required
def test_showfolder_open_show4dstem_preloads_complete_series_when_it_fits(
    monkeypatch, tmp_path
):
    import quantem.widget as qw
    import quantem.widget.io as wio

    fake_masters = [str(tmp_path / f"scan_{idx:02d}_master.h5") for idx in range(4)]
    calls = []
    frame_shape = (16, 16, 24, 24)
    frame_bytes = int(np.prod(frame_shape))

    monkeypatch.setattr(wio, "discover_masters", lambda *args, **kwargs: list(fake_masters))
    monkeypatch.setattr(wio, "is_master_ready", lambda path: True)

    class _Result:
        def __init__(self, path):
            value = int(path.split("scan_")[1][:2])
            self.data = torch.full(
                frame_shape,
                value,
                dtype=torch.uint8,
                device="cuda:0",
            )

    def fake_load(path, *, det_bin=4, dtype="u8", verbose=False, **kwargs):
        calls.append(path)
        return _Result(path)

    monkeypatch.setattr(qw, "load", fake_load)

    browser = _stub_browser(tmp_path)
    widget = browser.open_show4dstem(
        gpus=[0],
        page_budget="auto",
        page_max_vram_bytes=frame_bytes * len(fake_masters),
        det_bin=4,
        dtype="u8",
        preload_all_if_fits=True,
    )
    try:
        widget.wait_for_dataset_preload(timeout=10)
        assert widget._data.vram_resident() == list(range(len(fake_masters)))
        assert widget._raw_preload_status == "resident"
        assert calls == fake_masters
    finally:
        widget.close()


def test_showfolder_open_show4dstem_no_masters_returns_none(monkeypatch, tmp_path):
    import quantem.widget.io as wio

    monkeypatch.setattr(wio, "discover_masters", lambda *a, **k: [])
    sf = _stub_browser(tmp_path)
    assert sf.open_show4dstem() is None


def test_showfolder_open_show4dstem_with_selection_panel_builds_once(
    monkeypatch, tmp_path
):
    """A live selection panel must not build the heavy Show4DSTEM twice."""
    sf = _stub_browser(tmp_path)
    sf._selection_viewer_output = object()
    calls = []

    def fake_apply():
        calls.append("apply")
        sf._selected_show4dstem_widget = "widget"
        return "widget"

    def fake_refresh():
        return fake_apply()

    monkeypatch.setattr(sf, "_apply_selected_show4dstem", fake_apply)
    monkeypatch.setattr(sf, "_refresh_selected_viewers", fake_refresh)

    assert sf.open_show4dstem(gpus=[0], page_budget=1) == "widget"
    assert calls == ["apply"]


def test_showfolder_inherits_live_show4dstem_config_and_releases_old_widget(
    monkeypatch,
    tmp_path,
):
    """A watched folder rebuild keeps 4D paging options without leaking the old widget."""

    class _FakeShow4DSTEM:
        freed = False

        def free(self):
            self.freed = True

    previous = _stub_browser(tmp_path)
    old_widget = _FakeShow4DSTEM()
    previous._active_selected_modes = {"show4dstem"}
    previous._show4dstem_config = {
        "gpus": [0],
        "page_budget": "auto",
        "det_bin": 8,
        "dtype": "u8",
        "scan_size": 128,
    }
    previous._selected_show4dstem_widget = old_widget

    current = _stub_browser(tmp_path)
    refresh_calls = []
    monkeypatch.setattr(
        current,
        "_refresh_selected_viewers",
        lambda: refresh_calls.append("refresh"),
    )

    current.inherit_selected_viewers_from(previous)

    assert current._active_selected_modes == {"show4dstem"}
    assert current._show4dstem_config == previous._show4dstem_config
    assert getattr(current, "_selected_show4dstem_widget", None) is None
    assert getattr(previous, "_selected_show4dstem_widget", None) is None
    assert old_widget.freed is True
    assert refresh_calls == ["refresh"]


@cuda_required
def test_showfolder_open_show4dstem_drops_staging_frames_on_second_gpu(
    monkeypatch, tmp_path
):
    """Explicit multi-GPU staging must not pin offloaded frames outside Dataset5dstem."""
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA devices")
    import gc

    import quantem.widget as qw
    import quantem.widget.io as wio

    fake_masters = [str(tmp_path / f"scan_{i:02d}_master.h5") for i in range(4)]

    def fake_discover(folder, *, scan_shape=None, verbose=False, **kw):
        return list(fake_masters)

    def fake_ready(path):
        return True

    class _Result:
        def __init__(self, path):
            v = int(path.split("scan_")[1][:2])
            self.data = torch.full(
                (16, 16, 24, 24), v, dtype=torch.uint8, device="cuda:0"
            )

    def fake_load(path, *, det_bin=4, dtype="u8", verbose=False, **kw):
        return _Result(path)

    monkeypatch.setattr(wio, "discover_masters", fake_discover)
    monkeypatch.setattr(wio, "is_master_ready", fake_ready)
    monkeypatch.setattr(qw, "load", fake_load)

    sf = _stub_browser(tmp_path)
    w = sf.open_show4dstem(gpus=[0, 1], page_budget=1, det_bin=4, dtype="u8")
    gc.collect()
    with torch.cuda.device(1):
        torch.cuda.empty_cache()

    assert w._data.vram_resident() == [0]
    leaked = []
    for obj in gc.get_objects():
        try:
            if (
                isinstance(obj, torch.Tensor)
                and obj.device.type == "cuda"
                and obj.device.index == 1
                and tuple(obj.shape) == (16, 16, 24, 24)
            ):
                leaked.append(obj)
        except Exception:
            pass
    assert leaked == []

    w.frame_idx = 1
    frame = w._frame_data
    torch.cuda.synchronize(frame.device.index)
    del frame
    w.free()
    gc.collect()
    for device in (0, 1):
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    leaked_after_free = []
    for obj in gc.get_objects():
        try:
            if (
                isinstance(obj, torch.Tensor)
                and obj.device.type == "cuda"
                and obj.device.index == 1
                and tuple(obj.shape) == (16, 16, 24, 24)
            ):
                leaked_after_free.append(obj)
        except Exception:
            pass
    assert leaked_after_free == []
