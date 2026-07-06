"""Show4DSTEM dataset-slider paging (out-of-core multi-dataset VRAM control).

A ShowFolder of many 4D masters loads into one Show4DSTEM behind a dataset
slider. Keeping every master resident fills VRAM; ``page_budget`` controls how
many stay on the GPU and switching the slider pages the target in while evicting
least-recently-used datasets to RAM. This guards both fixed-count paging and
``page_budget="auto"`` memory-sized caching.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from quantem.widget import Show4DSTEM
from quantem.widget.data.dataset5dstem import Dataset5dstem

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="paging test needs a CUDA device")


def _series(n=4, scan=32, det=48):
    """n independent 4D uint8 datasets on cuda:0 — a synthetic 'n-master folder'."""
    frames = [
        torch.randint(0, 100, (scan, scan, det, det), dtype=torch.uint8, device="cuda:0")
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
    assert len(resident) == 2 and 3 in resident   # newest always resident, ≤ budget


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


def test_lazy_dataset_validates_metadata_and_initial_frames():
    def load():
        return torch.zeros((2, 2, 3, 3), dtype=torch.uint8)

    with pytest.raises(ValueError, match="at least one"):
        Dataset5dstem.from_lazy_loaders([], shape=(0, 2, 2, 3, 3), dtype=torch.uint8)

    with pytest.raises(ValueError, match="out of range"):
        Dataset5dstem.from_lazy_loaders(
            [load], shape=(1, 2, 2, 3, 3), dtype=torch.uint8,
            initial_frames={1: load()},
        )

    with pytest.raises(ValueError, match="expected"):
        Dataset5dstem.from_lazy_loaders(
            [load], shape=(1, 2, 2, 3, 3), dtype=torch.uint8,
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


# --- ShowFolder -> Show4DSTEM handoff -----------------------------------------

def _stub_browser(folder):
    from quantem.widget.showfolder_core import ShowFolderBrowser

    class _Stub(ShowFolderBrowser):
        def __init__(self, f):
            self.folder = f

    return _Stub(folder)


def test_showfolder_open_show4dstem_preserves_loader_device_when_gpus_none(monkeypatch, tmp_path):
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


@cuda_required
def test_showfolder_open_show4dstem_builds_paged_multimaster_on_explicit_cuda(monkeypatch, tmp_path):
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
            self.data = torch.full((16, 16, 24, 24), v, dtype=torch.uint8, device="cuda:0")

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
            self.data = torch.full((16, 16, 24, 24), v, dtype=torch.uint8, device="cuda:0")

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


def test_showfolder_open_show4dstem_no_masters_returns_none(monkeypatch, tmp_path):
    import quantem.widget.io as wio
    monkeypatch.setattr(wio, "discover_masters", lambda *a, **k: [])
    sf = _stub_browser(tmp_path)
    assert sf.open_show4dstem() is None


def test_showfolder_open_show4dstem_with_selection_panel_builds_once(monkeypatch, tmp_path):
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


@cuda_required
def test_showfolder_open_show4dstem_drops_staging_frames_on_second_gpu(monkeypatch, tmp_path):
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
            self.data = torch.full((16, 16, 24, 24), v, dtype=torch.uint8, device="cuda:0")

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
