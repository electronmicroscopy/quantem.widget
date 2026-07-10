from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

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
        self.chunks = [np.stack([self.frame(i) for i in range(self._n)], axis=0)]

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


def test_multi_chunked_frames_export_materializes_all_ready_datasets() -> None:
    from quantem.widget.show4dstem import Show4DSTEM

    multi = MultiChunkedFrames([_DummyFrames(value=1)], names=["first"])
    multi.append_dataset(_DummyFrames(value=20), name="second")

    class _ExportView:
        _data = multi
        n_frames = 2
        shape_rows = 4
        shape_cols = 4
        det_rows = 8
        det_cols = 8

    arr = Show4DSTEM._export_data_array(_ExportView(), dtype="uint8", det_bin=2)

    assert arr.shape == (2, 4, 4, 4, 4)
    assert arr.dtype == np.uint8
    assert arr[0, 0, 0, 0, 0] == 1
    assert arr[1, 0, 0, 0, 0] == 20


def test_show4dstem_keeps_dataset5dstem_frame_backing_for_export() -> None:
    import torch

    from quantem.widget.data import Dataset5dstem
    from quantem.widget.show4dstem import Show4DSTEM

    frames = [
        torch.full((4, 4, 4, 4), 10, dtype=torch.uint16),
        torch.full((4, 4, 4, 4), 20, dtype=torch.uint16),
    ]
    data = Dataset5dstem(frames=frames, name="frame-backed test")

    widget = Show4DSTEM(
        data,
        center=(2, 2),
        bf_radius=1,
        precompute_virtual_images=False,
        verbose=False,
    )

    assert widget._data is data
    assert widget.n_frames == 2
    assert widget.frame_dim_label == "Frame"

    arr = widget._export_data_array(dtype="uint8", det_bin=2)

    assert arr.shape == (2, 4, 4, 2, 2)
    assert arr.dtype == np.uint8
    assert np.all(arr[0] == 10)
    assert np.all(arr[1] == 20)

    vi = widget._virtual_image_for_frame(1)

    assert vi.shape == (4, 4)
    assert np.all(vi == 100)


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


def test_lazy_macbook_datasets_append_new_masters_skips_existing(tmp_path) -> None:
    first = tmp_path / "first_master.h5"
    second = tmp_path / "second_master.h5"
    calls: list[tuple[str, bool]] = []

    lazy = LazyMacbookDatasets(
        masters=[str(first)],
        det_bin=4,
        names=["first"],
        multi=object(),
        decode=lambda path: object(),
        verbose=False,
    )

    def _append_master(path, *, name=None, async_=True):
        calls.append((path, async_))
        lazy.masters.append(path)
        return len(calls)

    lazy.append_master = _append_master  # type: ignore[method-assign]

    added = lazy.append_new_masters([first, second, second], async_=False)

    assert added == [1]
    assert calls == [(str(second), False)]


def test_lazy_macbook_datasets_poll_master_folder_filters_ready(monkeypatch, tmp_path) -> None:
    first = tmp_path / "first_master.h5"
    ready = tmp_path / "ready_master.h5"
    partial = tmp_path / "partial_master.h5"

    lazy = LazyMacbookDatasets(
        masters=[str(first)],
        det_bin=4,
        names=["first"],
        multi=object(),
        decode=lambda path: object(),
        verbose=False,
    )
    calls: list[str] = []

    def _append_master(path, *, name=None, async_=True):
        calls.append(path)
        lazy.masters.append(path)
        return len(calls)

    lazy.append_master = _append_master  # type: ignore[method-assign]

    import quantem.widget.io as widget_io

    monkeypatch.setattr(widget_io, "discover_masters", lambda *args, **kwargs: [
        str(first),
        str(ready),
        str(partial),
    ])
    monkeypatch.setattr(widget_io, "is_master_ready", lambda path: path != str(partial))

    added = lazy.poll_master_folder(tmp_path, async_=False)

    assert added == [1]
    assert calls == [str(ready)]


def test_show4dstem_from_folder_mps_honors_public_folder_options(
    monkeypatch,
    tmp_path,
) -> None:
    import quantem.widget.io as widget_io
    import quantem.widget.multidataset_mps as multidataset_mps
    import quantem.widget.show4dstem_factory as factory

    masters = [str(tmp_path / f"scan_{idx}_master.h5") for idx in range(3)]
    calls: dict[str, object] = {}

    def fake_discover(
        folder,
        *,
        pattern="*_master.h5",
        recursive=True,
        scan_shape=None,
        verbose=False,
        **kwargs,
    ):
        calls["discover"] = {
            "folder": folder,
            "pattern": pattern,
            "recursive": recursive,
            "scan_shape": scan_shape,
            "verbose": verbose,
            "kwargs": kwargs,
        }
        return list(masters)

    def fake_ready(path):
        return path != masters[1]

    class _FakeLive:
        def poll_master_folder(self, folder, **kwargs):
            calls["poll"] = {"folder": folder, **kwargs}
            return [7]

        def watch_master_folder(self, folder, **kwargs):
            calls["watch"] = {"folder": folder, **kwargs}
            return self

        def stop_watch(self):
            calls["stop"] = True

    live = _FakeLive()

    def fake_load_macbook_datasets(
        paths,
        *,
        det_bin=4,
        scan_size=None,
        verbose=True,
        skip_mps_memory_check=None,
    ):
        calls["loader"] = {
            "paths": list(paths),
            "det_bin": det_bin,
            "scan_size": scan_size,
            "verbose": verbose,
            "skip_mps_memory_check": skip_mps_memory_check,
        }
        return live

    def fake_show4dstem(data, **kwargs):
        calls["viewer"] = {"data": data, "kwargs": kwargs}
        return SimpleNamespace()

    monkeypatch.setattr(widget_io, "discover_masters", fake_discover)
    monkeypatch.setattr(widget_io, "is_master_ready", fake_ready)
    monkeypatch.setattr(
        multidataset_mps,
        "load_macbook_datasets",
        fake_load_macbook_datasets,
    )
    monkeypatch.setattr(factory, "Show4DSTEM", fake_show4dstem)

    with pytest.warns(RuntimeWarning, match="dtype='u8'.*paging/preload"):
        viewer = factory.from_folder(
            tmp_path,
            backend="mps",
            pattern="scan_*_master.h5",
            recursive=False,
            scan_size=256,
            ready_only=True,
            det_bin=8,
            dtype="u8",
            load_kwargs={"skip_mps_memory_check": True},
            page_budget=2,
            page_max_vram_fraction=0.5,
            preload_initial_page=False,
            view_mode="single",
            compare_cols=2,
            compare_max_panels=5,
            watch=True,
            watch_interval=0.25,
            verbose=False,
            custom_option="kept",
        )

    assert calls["discover"] == {
        "folder": str(tmp_path.resolve()),
        "pattern": "scan_*_master.h5",
        "recursive": False,
        "scan_shape": (256, 256),
        "verbose": False,
        "kwargs": {},
    }
    assert calls["loader"] == {
        "paths": [masters[0], masters[2]],
        "det_bin": 8,
        "scan_size": 256,
        "verbose": False,
        "skip_mps_memory_check": True,
    }
    assert calls["viewer"] == {
        "data": live,
        "kwargs": {
            "view_mode": "single",
            "compare_cols": 2,
            "compare_max_panels": 5,
            "page_budget": 2,
            "page_device": None,
            "page_max_vram_fraction": 0.5,
            "page_reserve_vram_bytes": None,
            "page_max_vram_bytes": None,
            "verbose": False,
            "custom_option": "kept",
        },
    }
    assert calls["watch"] == {
        "folder": tmp_path.resolve(),
        "interval": 0.25,
        "pattern": "scan_*_master.h5",
        "recursive": False,
        "scan_size": 256,
        "ready_only": True,
    }

    assert viewer.poll_folder(async_=False) == [7]
    assert calls["poll"] == {
        "folder": tmp_path.resolve(),
        "pattern": "scan_*_master.h5",
        "recursive": False,
        "scan_size": 256,
        "ready_only": True,
        "async_": False,
    }
    viewer.stop_folder_watch()
    assert calls["stop"] is True


def test_show4dstem_from_folder_mps_rejects_unsupported_load_kwargs(tmp_path) -> None:
    import quantem.widget.show4dstem_factory as factory

    with pytest.raises(ValueError, match="apply_mask"):
        factory.from_folder(
            tmp_path,
            backend="mps",
            dtype="u16",
            load_kwargs={"apply_mask": False},
        )
