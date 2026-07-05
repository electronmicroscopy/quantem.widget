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
