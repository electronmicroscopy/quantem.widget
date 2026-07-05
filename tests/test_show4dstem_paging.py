"""Show4DSTEM dataset-slider paging (out-of-core multi-dataset VRAM control).

A ShowFolder of many 4D masters loads into one Show4DSTEM behind a dataset
slider. Keeping every master resident fills VRAM; ``page_budget`` caps how many
stay on the GPU and switching the slider pages the target in while evicting the
least-recently-used dataset to RAM. This guards that wiring: without the budget
every dataset is resident (unchanged default); with it, only ``page_budget``
datasets sit in VRAM and the footprint stays flat across switches (no leak).
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


@cuda_required
def test_showfolder_open_show4dstem_builds_paged_multimaster_on_explicit_cuda(monkeypatch, tmp_path):
    """open_show4dstem(gpus=[...]) loads every folder master into one paged Show4DSTEM."""
    import quantem.widget.io as wio
    import quantem.widget as qw

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
    w = sf.open_show4dstem(gpus=[0], page_budget=1, det_bin=4, dtype="u8")
    assert w is not None
    ds = w._data
    assert w.n_frames == 4
    assert list(w.frame_labels) == [f"scan_{i:02d}" for i in range(4)]
    assert ds.vram_resident() == [0]
    w.frame_idx = 3
    _ = w._frame_data
    assert ds.vram_resident() == [3]


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
