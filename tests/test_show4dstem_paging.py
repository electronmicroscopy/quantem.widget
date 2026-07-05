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

if not torch.cuda.is_available():
    pytest.skip("paging test needs a CUDA device", allow_module_level=True)

from quantem.widget import Show4DSTEM
from quantem.widget.data.dataset5dstem import Dataset5dstem


def _series(n=4, scan=32, det=48):
    """n independent 4D uint8 datasets on cuda:0 — a synthetic 'n-master folder'."""
    frames = [
        torch.randint(0, 100, (scan, scan, det, det), dtype=torch.uint8, device="cuda:0")
        for _ in range(n)
    ]
    return Dataset5dstem.from_frames(frames)


def test_no_page_budget_keeps_all_datasets_resident():
    ds = _series(4)
    w = Show4DSTEM(ds, verbose=False)
    assert w.n_frames == 4
    # default: every dataset stays on the GPU (instant switch, current behavior)
    assert sorted(ds.vram_resident()) == [0, 1, 2, 3]


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


def test_page_budget_two_keeps_two_resident():
    ds = _series(4)
    w = Show4DSTEM(ds, page_budget=2, verbose=False)
    w.frame_idx = 1
    _ = w._frame_data
    w.frame_idx = 3
    _ = w._frame_data
    resident = set(ds.vram_resident())
    assert len(resident) == 2 and 3 in resident   # newest always resident, ≤ budget


# --- ShowFolder -> Show4DSTEM handoff -----------------------------------------

def _stub_browser(folder):
    from quantem.widget.showfolder_core import ShowFolderBrowser

    class _Stub(ShowFolderBrowser):
        def __init__(self, f):
            self.folder = f

    return _Stub(folder)


def test_showfolder_open_show4dstem_builds_paged_multimaster(monkeypatch, tmp_path):
    """open_show4dstem loads every folder master into one paged Show4DSTEM."""
    import quantem.widget.io as wio
    import quantem.widget as qw

    fake_masters = [str(tmp_path / f"scan_{i:02d}_master.h5") for i in range(4)]

    def fake_discover(folder, *, scan_shape=None, verbose=False, **kw):
        return list(fake_masters)

    def fake_ready(path):
        return True

    class _Result:
        def __init__(self, path):
            # distinct value per master so frames are not aliased
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
    assert ds.vram_resident() == [0]           # only the active dataset resident
    w.frame_idx = 3
    _ = w._frame_data
    assert ds.vram_resident() == [3]           # switching evicted the LRU


def test_showfolder_open_show4dstem_no_masters_returns_none(monkeypatch, tmp_path):
    import quantem.widget.io as wio
    monkeypatch.setattr(wio, "discover_masters", lambda *a, **k: [])
    sf = _stub_browser(tmp_path)
    assert sf.open_show4dstem() is None
