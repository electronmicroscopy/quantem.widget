from __future__ import annotations

import numpy as np

from quantem.widget.kernels.compute.mps import MultiChunkedFrames
from quantem.widget.multidataset_mps import LazyMacbookDatasets


class _DummyFrames:
    def __init__(self, *, scan: int = 4, det: tuple[int, int] = (8, 8), value: int = 0):
        self._det = det
        self._frame_elems = det[0] * det[1]
        self.det_bin = 1
        self._torch = None
        self._n = scan * scan
        self.dtype = np.dtype("uint16")
        self.device = "cpu"
        self.metadata = {}
        self.value = value

    def frame(self, idx: int) -> np.ndarray:
        return np.full(self._det, self.value + int(idx), dtype=np.uint16)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.frame(idx)


def test_multi_chunked_frames_appends_dataset_and_notifies_viewer() -> None:
    first = _DummyFrames(value=1)
    second = _DummyFrames(value=2)
    multi = MultiChunkedFrames([first], names=["first"])
    ready: list[int] = []
    multi.on_ready = ready.append

    idx = multi.append_dataset(second, name="second")

    assert idx == 1
    assert multi.shape == (2, 4, 4, 8, 8)
    assert multi.names == ["first", "second"]
    assert multi.n_ready == 2
    assert ready == [1]
    multi.set_active(1)
    assert multi.active_idx == 1
    assert np.all(multi[0, 0] == 2)


def test_multi_chunked_frames_rejects_incompatible_append() -> None:
    multi = MultiChunkedFrames([_DummyFrames(det=(8, 8))])

    with np.testing.assert_raises(ValueError):
        multi.append_dataset(_DummyFrames(det=(16, 16)))


def test_lazy_macbook_datasets_append_master_sync() -> None:
    appended: list[tuple[object, str | None]] = []

    class _FakeMulti:
        def append_dataset(self, frames, name=None):
            appended.append((frames, name))
            return len(appended)

    decoded = object()
    lazy = LazyMacbookDatasets(
        masters=["first_master.h5"],
        det_bin=4,
        names=["first"],
        multi=_FakeMulti(),
        decode=lambda path: decoded,
        verbose=False,
    )

    idx = lazy.append_master("/tmp/second_master.h5", async_=False)

    assert idx == 1
    assert lazy.masters == ["first_master.h5", "/tmp/second_master.h5"]
    assert lazy.names == ["first", "second"]
    assert appended == [(decoded, "second")]
