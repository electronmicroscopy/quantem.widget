from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from quantem.gpu.compute.mps import MultiChunkedFrames
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
    lazy.shutdown()


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
        lazy._successful_master_keys.add(lazy._master_key(path))
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
        lazy._successful_master_keys.add(lazy._master_key(path))
        return len(calls)

    lazy.append_master = _append_master  # type: ignore[method-assign]

    import quantem.gpu.io.hdf5 as widget_io

    monkeypatch.setattr(widget_io, "discover_masters", lambda *args, **kwargs: [
        str(first),
        str(ready),
        str(partial),
    ])
    monkeypatch.setattr(widget_io, "is_master_ready", lambda path: path != str(partial))

    added = lazy.poll_master_folder(tmp_path, async_=False)

    assert added == [1]
    assert calls == [str(ready)]


def test_lazy_macbook_datasets_serializes_fifo_decodes(tmp_path) -> None:
    decode_order: list[str] = []
    active = 0
    max_active = 0
    lock = threading.Lock()

    class _FakeMulti:
        def __init__(self) -> None:
            self.datasets = [object()]
            self.names = ["first"]
            self.on_ready = None

        def append_dataset(self, frames, name=None):
            self.datasets.append(frames)
            self.names.append(str(name))
            return len(self.datasets) - 1

    def decode(path):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        decode_order.append(str(path))
        time.sleep(0.01)
        with lock:
            active -= 1
        return object()

    first = tmp_path / "first_master.h5"
    arrivals = [tmp_path / f"arrival_{idx}_master.h5" for idx in range(3)]
    multi = _FakeMulti()
    lazy = LazyMacbookDatasets(
        masters=[str(first)],
        det_bin=4,
        names=["first"],
        multi=multi,
        decode=decode,
        verbose=False,
    )

    # C1: three asynchronous arrivals share one worker, expect FIFO and max
    # concurrency 1.
    added = lazy.append_new_masters(arrivals, async_=True)
    assert added == [1, 2, 3]
    assert lazy.wait_for_decodes(timeout=2.0)
    assert decode_order == [lazy._master_key(path) for path in arrivals]
    assert max_active == 1
    assert multi.names == ["first", "arrival_0", "arrival_1", "arrival_2"]
    assert lazy.names == multi.names
    lazy.stop_watch()


def test_lazy_macbook_initial_fill_uses_owned_worker(
    monkeypatch,
    tmp_path,
) -> None:
    decoded: list[str] = []
    filled: list[tuple[int, object]] = []

    class _FakeMulti:
        def __init__(self) -> None:
            self.datasets = [object(), None, None]
            self.names = ["first", "second", "third"]
            self.on_ready = None

        def set_dataset(self, idx, frames):
            self.datasets[idx] = frames
            filled.append((idx, frames))

    viewer = SimpleNamespace()
    import quantem.widget.show4dstem_mps as show4dstem_mps

    monkeypatch.setattr(
        show4dstem_mps,
        "Show4DSTEM_MACBOOK",
        lambda data, **kwargs: viewer,
    )
    masters = [tmp_path / f"scan_{idx}_master.h5" for idx in range(3)]

    def decode(path):
        decoded.append(path)
        return object()

    lazy = LazyMacbookDatasets(
        masters=masters,
        det_bin=4,
        names=["first", "second", "third"],
        multi=_FakeMulti(),
        decode=decode,
        verbose=False,
    )

    # C1: initial slots 1..N use the tracked FIFO worker, expect lifecycle ownership.
    assert lazy.build_viewer() is viewer
    assert viewer._mps_folder_live is lazy
    assert lazy.wait_for_decodes(timeout=1.0)
    assert decoded == [lazy._master_key(path) for path in masters[1:]]
    assert [idx for idx, _ in filled] == [1, 2]
    lazy.shutdown()
    assert lazy._decode_thread is None


def test_lazy_macbook_datasets_failed_decode_remains_retryable(tmp_path) -> None:
    attempts = 0

    class _FakeMulti:
        datasets = [object()]
        names = ["first"]
        on_ready = None

        def append_dataset(self, frames, name=None):
            self.datasets.append(frames)
            self.names.append(str(name))
            return len(self.datasets) - 1

    def decode(path):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("linked chunk is still open")
        return object()

    first = tmp_path / "first_master.h5"
    arrival = tmp_path / "arrival_master.h5"
    lazy = LazyMacbookDatasets(
        masters=[str(first)],
        det_bin=4,
        names=["first"],
        multi=_FakeMulti(),
        decode=decode,
        verbose=False,
    )

    # C1: a decode failure is neither pending nor successful, expect retryability.
    with pytest.raises(OSError, match="still open"):
        lazy.append_master(arrival, async_=False)
    assert lazy.pending_masters == ()
    assert lazy._master_key(arrival) not in lazy.successful_masters
    assert lazy.masters == [str(first)]

    # C2: the unchanged path decodes on retry, expect one appended dataset/label.
    assert lazy.append_master(arrival, async_=False) == 1
    assert attempts == 2
    assert lazy.masters == [str(first), lazy._master_key(arrival)]
    assert lazy.names == ["first", "arrival"]
    lazy.stop_watch()


def test_lazy_macbook_datasets_stop_joins_and_discards_late_decode(tmp_path) -> None:
    entered = threading.Event()
    release = threading.Event()
    appended: list[object] = []
    statuses: list[tuple[str, str]] = []

    class _FakeMulti:
        datasets = [object()]
        names = ["first"]
        on_ready = None

        def append_dataset(self, frames, name=None):
            appended.append(frames)
            return 1

    def decode(path):
        entered.set()
        release.wait(2.0)
        return object()

    lazy = LazyMacbookDatasets(
        masters=[str(tmp_path / "first_master.h5")],
        det_bin=4,
        names=["first"],
        multi=_FakeMulti(),
        decode=decode,
        verbose=False,
    )
    lazy.set_status_callback(lambda state, detail: statuses.append((state, detail)))
    lazy.watch_master_folder(tmp_path, interval=100.0)
    lazy.append_master(tmp_path / "late_master.h5", async_=True)
    assert entered.wait(1.0)

    # C1: stop during decode waits, expect its late result/callback to be rejected.
    stopper = threading.Thread(target=lazy.stop_watch)
    stopper.start()
    time.sleep(0.02)
    assert stopper.is_alive()
    release.set()
    stopper.join(1.0)
    assert not stopper.is_alive()
    assert appended == []
    assert lazy.pending_masters == ()
    assert lazy._decode_thread is None
    assert statuses[-1][0] == "stopped"


@pytest.mark.parametrize("interval", [0.0, -1.0, float("nan"), float("inf")])
def test_lazy_macbook_watch_rejects_invalid_interval(tmp_path, interval) -> None:
    lazy = LazyMacbookDatasets(
        masters=[str(tmp_path / "first_master.h5")],
        det_bin=4,
        names=["first"],
        multi=SimpleNamespace(),
        decode=lambda path: object(),
        verbose=False,
    )

    lazy.watch_master_folder(tmp_path, interval=100.0)
    healthy_thread = lazy._watch_thread
    try:
        # C1: an invalid restart is rejected before lifecycle mutation, expect
        # the existing healthy watcher and truthful green state to survive.
        with pytest.raises(ValueError, match="finite positive"):
            lazy.watch_master_folder(tmp_path, interval=interval)
        assert lazy._watch_thread is healthy_thread
        assert healthy_thread is not None and healthy_thread.is_alive()
        assert lazy._status_state == "watching"
    finally:
        lazy.stop_watch()


def test_lazy_macbook_overlapping_poll_returns_without_blocking(tmp_path) -> None:
    lazy = LazyMacbookDatasets(
        masters=[str(tmp_path / "first_master.h5")],
        det_bin=4,
        names=["first"],
        multi=SimpleNamespace(),
        decode=lambda path: object(),
        verbose=False,
    )
    lazy._poll_lock.acquire()
    started = time.perf_counter()
    try:
        # C1: a manual poll overlaps the background scanner, expect a bounded
        # no-op rather than duplicate probation/decode work or UI blocking.
        assert lazy.poll_master_folder(tmp_path) == []
    finally:
        lazy._poll_lock.release()
    assert time.perf_counter() - started < 0.5


def test_lazy_macbook_watch_worker_failure_is_not_green(monkeypatch, tmp_path) -> None:
    statuses: list[tuple[str, str]] = []
    lazy = LazyMacbookDatasets(
        masters=[str(tmp_path / "first_master.h5")],
        det_bin=4,
        names=["first"],
        multi=SimpleNamespace(),
        decode=lambda path: object(),
        verbose=False,
    )
    lazy.set_status_callback(lambda state, detail: statuses.append((state, detail)))

    import quantem.gpu.io.hdf5 as widget_io

    def fail_discovery(*args, **kwargs):
        raise SystemExit("simulated worker termination")

    monkeypatch.setattr(widget_io, "discover_masters", fail_discovery)

    # C1: an unexpected worker exit clears liveness, expect red rather than green.
    lazy.watch_master_folder(tmp_path, interval=0.01)
    deadline = time.monotonic() + 1.0
    while statuses[-1][0] != "error" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert statuses[-1][0] == "error"
    assert "stopped unexpectedly" in statuses[-1][1]
    assert lazy._watch_thread is None
    assert lazy._watch_stop is None
    assert not lazy._watching
    lazy._refresh_activity_status()
    assert statuses[-1][0] == "error"
    lazy.stop_watch()


def test_lazy_macbook_watch_requires_two_identical_ready_signatures(
    monkeypatch,
    tmp_path,
) -> None:
    first = tmp_path / "first_master.h5"
    arrival = tmp_path / "arrival_master.h5"
    statuses: list[tuple[str, str]] = []

    class _FakeMulti:
        def __init__(self) -> None:
            self.datasets = [object()]
            self.names = ["first"]
            self.on_ready = None

        def append_dataset(self, frames, name=None):
            self.datasets.append(frames)
            self.names.append(str(name))
            return len(self.datasets) - 1

    lazy = LazyMacbookDatasets(
        masters=[str(first)],
        det_bin=4,
        names=["first"],
        multi=_FakeMulti(),
        decode=lambda path: object(),
        verbose=False,
    )
    lazy.set_status_callback(lambda state, detail: statuses.append((state, detail)))

    import quantem.gpu.io.hdf5 as widget_io

    discovered = [str(first), str(arrival)]
    monkeypatch.setattr(
        widget_io,
        "discover_masters",
        lambda *args, **kwargs: list(discovered),
    )
    report = SimpleNamespace(
        ready=True,
        reason="complete",
        action="ready",
        source_signature={"files": [{"size": 123, "mtime_ns": 456}]},
    )
    monkeypatch.setattr(widget_io, "inspect_master_readiness", lambda *a, **k: report)
    lazy.watch_master_folder(tmp_path, interval=100.0)

    # C1: the first complete signature is probationary, expect discovery to
    # transition through Updating before amber Waiting.
    statuses.clear()
    assert lazy.poll_master_folder(
        tmp_path,
        async_=False,
        require_stable=True,
    ) == []
    assert statuses[-1][0] == "waiting"
    assert "updating" in [state for state, _ in statuses]
    assert "unchanged completion signature" in statuses[-1][1]

    # C2: the candidate disappears for a poll, expect Updating discovery,
    # probation pruning, and then idle Watching.
    discovered[:] = [str(first)]
    statuses.clear()
    assert lazy.poll_master_folder(
        tmp_path,
        async_=False,
        require_stable=True,
    ) == []
    assert statuses[-1][0] == "watching"
    assert "updating" in [state for state, _ in statuses]

    # C3: the same path reappears, expect a fresh first probation observation
    # instead of immediate acceptance from the stale pre-disappearance signature.
    discovered.append(str(arrival))
    statuses.clear()
    assert lazy.poll_master_folder(
        tmp_path,
        async_=False,
        require_stable=True,
    ) == []
    assert statuses[-1][0] == "waiting"
    assert "updating" in [state for state, _ in statuses]

    # C4: the next unchanged signature is accepted, expect Updating only for the
    # real decode/append operation and then green Watching.
    statuses.clear()
    assert lazy.poll_master_folder(
        tmp_path,
        async_=False,
        require_stable=True,
    ) == [1]
    assert lazy.names == ["first", "arrival"]
    assert "updating" in [state for state, _ in statuses]
    assert statuses[-1][0] == "watching"
    lazy.stop_watch()


def test_lazy_macbook_watch_classifies_readiness_waiting_and_error(
    monkeypatch,
    tmp_path,
) -> None:
    first = tmp_path / "first_master.h5"
    candidate = tmp_path / "candidate_master.h5"
    statuses: list[tuple[str, str]] = []
    current = {
        "report": SimpleNamespace(
            ready=False,
            reason="detector sources have inconsistent dtypes: uint8, uint16",
            action="Repair or reacquire the acquisition.",
            actual_frames=16,
            expected_frames=16,
            source_signature={"revision": "bad"},
        )
    }
    lazy = LazyMacbookDatasets(
        masters=[str(first)],
        det_bin=4,
        names=["first"],
        multi=SimpleNamespace(),
        decode=lambda path: object(),
        verbose=False,
    )
    lazy.set_status_callback(lambda state, detail: statuses.append((state, detail)))

    import quantem.gpu.io.hdf5 as widget_io

    monkeypatch.setattr(
        widget_io,
        "discover_masters",
        lambda *args, **kwargs: [str(first), str(candidate)],
    )
    monkeypatch.setattr(
        widget_io,
        "inspect_master_readiness",
        lambda *args, **kwargs: current["report"],
    )
    lazy.watch_master_folder(tmp_path, interval=100.0)
    try:
        # C1: a permanent source-contract defect is not a writer delay, expect a
        # red error while the live worker remains available for a corrected file.
        assert lazy.poll_master_folder(
            tmp_path,
            async_=False,
            require_stable=True,
        ) == []
        assert statuses[-1][0] == "error"
        assert "inconsistent dtypes" in statuses[-1][1]

        # C2: the candidate becomes incomplete, expect Updating discovery then
        # amber Waiting without a decode.
        current["report"] = SimpleNamespace(
            ready=False,
            reason="stored frame count is 8; expected 16",
            action="Wait for detector chunks to finish writing.",
            actual_frames=8,
            expected_frames=16,
            source_signature={"revision": "partial"},
        )
        statuses.clear()
        assert lazy.poll_master_folder(
            tmp_path,
            async_=False,
            require_stable=True,
        ) == []
        assert statuses[-1][0] == "waiting"
        assert "updating" in [state for state, _ in statuses]
    finally:
        lazy.stop_watch()


def test_lazy_macbook_watch_distinguishes_empty_discovery_from_bad_request(
    monkeypatch,
    tmp_path,
) -> None:
    statuses: list[tuple[str, str]] = []
    mode = {"value": "empty"}
    lazy = LazyMacbookDatasets(
        masters=[str(tmp_path / "first_master.h5")],
        det_bin=4,
        names=["first"],
        multi=SimpleNamespace(),
        decode=lambda path: object(),
        verbose=False,
    )
    lazy.set_status_callback(lambda state, detail: statuses.append((state, detail)))

    import quantem.gpu.io.hdf5 as widget_io

    def discover(*args, **kwargs):
        if mode["value"] == "empty":
            raise ValueError("No files matching '*_master.h5' in acquisition")
        raise ValueError("scan_shape must contain two positive integers")

    monkeypatch.setattr(widget_io, "discover_masters", discover)
    lazy.watch_master_folder(tmp_path, interval=100.0)
    try:
        # C1: no acquisition has arrived yet, expect Updating discovery then
        # an idle green watcher.
        statuses.clear()
        assert lazy.poll_master_folder(tmp_path, require_stable=True) == []
        assert statuses[-1][0] == "watching"
        assert "updating" in [state for state, _ in statuses]

        # C2: a malformed discovery request is not an empty folder, expect the
        # error to propagate and publish a red corrective watcher state.
        mode["value"] = "invalid"
        with pytest.raises(ValueError, match="scan_shape"):
            lazy.poll_master_folder(tmp_path, require_stable=True)
        assert statuses[-1][0] == "error"
        assert "scan_shape" in statuses[-1][1]
    finally:
        lazy.stop_watch()


def test_lazy_macbook_poll_reports_contract_error_without_blocking_valid_master(
    monkeypatch,
    tmp_path,
) -> None:
    first = tmp_path / "first_master.h5"
    bad = tmp_path / "bad_master.h5"
    good = tmp_path / "good_master.h5"
    statuses: list[tuple[str, str]] = []

    class _FakeMulti:
        def __init__(self) -> None:
            self.datasets = [object()]
            self.names = ["first"]
            self.on_ready = None

        def append_dataset(self, frames, name=None):
            self.datasets.append(frames)
            self.names.append(str(name))
            return len(self.datasets) - 1

    def validate(path: str) -> None:
        if path == LazyMacbookDatasets._master_key(bad):
            raise ValueError("detector_shape=(96, 96); expected (192, 192)")

    lazy = LazyMacbookDatasets(
        masters=[str(first)],
        det_bin=4,
        names=["first"],
        multi=_FakeMulti(),
        decode=lambda path: object(),
        verbose=False,
        validate_master=validate,
    )
    lazy.set_status_callback(lambda state, detail: statuses.append((state, detail)))

    import quantem.gpu.io.hdf5 as widget_io

    monkeypatch.setattr(
        widget_io,
        "discover_masters",
        lambda *args, **kwargs: [str(first), str(bad), str(good)],
    )
    monkeypatch.setattr(widget_io, "is_master_ready", lambda path: True)
    lazy.watch_master_folder(tmp_path, interval=100.0)

    # C1: an incompatible master does not block a later valid one, expect red + append.
    assert lazy.poll_master_folder(tmp_path, async_=False) == [1]
    assert lazy.names == ["first", "good"]
    assert statuses[-1][0] == "error"
    assert "detector_shape" in statuses[-1][1]
    assert lazy._master_key(bad) not in lazy.successful_masters
    lazy.stop_watch()


def test_lazy_macbook_poll_decode_failure_does_not_block_later_master(
    monkeypatch,
    tmp_path,
) -> None:
    first = tmp_path / "first_master.h5"
    bad = tmp_path / "bad_master.h5"
    good = tmp_path / "good_master.h5"
    attempts: dict[str, int] = {}

    class _FakeMulti:
        def __init__(self) -> None:
            self.datasets = [object()]
            self.names = ["first"]
            self.on_ready = None

        def append_dataset(self, frames, name=None):
            self.datasets.append(frames)
            self.names.append(str(name))
            return len(self.datasets) - 1

    def decode(path):
        attempts[path] = attempts.get(path, 0) + 1
        if path == LazyMacbookDatasets._master_key(bad) and attempts[path] == 1:
            raise OSError("chunk changed during decode")
        return object()

    lazy = LazyMacbookDatasets(
        masters=[str(first)],
        det_bin=4,
        names=["first"],
        multi=_FakeMulti(),
        decode=decode,
        verbose=False,
    )
    import quantem.gpu.io.hdf5 as widget_io

    monkeypatch.setattr(
        widget_io,
        "discover_masters",
        lambda *args, **kwargs: [str(first), str(bad), str(good)],
    )
    monkeypatch.setattr(widget_io, "is_master_ready", lambda path: True)
    lazy.watch_master_folder(tmp_path, interval=100.0)

    # C1: one synchronous decode fails, expect the later ready master to append.
    assert lazy.poll_master_folder(tmp_path, async_=False) == [1]
    assert lazy.names == ["first", "good"]
    assert lazy._status_state == "error"
    assert "retryable" in lazy._last_error_detail
    assert "poll again" in lazy._last_error_detail
    assert lazy._master_key(bad) not in lazy.successful_masters

    # C2: the failed path remains unknown, expect one retry append on the next poll.
    assert lazy.poll_master_folder(tmp_path, async_=False) == [2]
    assert lazy.names == ["first", "good", "bad"]
    assert attempts[lazy._master_key(bad)] == 2
    lazy.stop_watch()


def test_mps_ready_callback_synchronizes_frame_labels() -> None:
    from quantem.widget.show4dstem_mps import Show4DSTEMMPS

    multi = SimpleNamespace(
        datasets=[object(), object()],
        names=["first", "arrival"],
    )
    viewer = SimpleNamespace(
        _mps_folder_shutdown=False,
        _multi=multi,
        _multi_total=1,
        _ioloop=None,
        n_frames=1,
        frame_labels=["first"],
        _frame_labels=["first"],
        _refresh_multi_title=lambda: None,
        _refresh_compare_virtual_images=lambda: None,
    )

    # C1: a decoded append callback publishes multi.names, expect exact slider labels.
    Show4DSTEMMPS._on_multi_dataset_ready(viewer, 1)
    assert viewer.n_frames == 2
    assert viewer.frame_labels == ["first", "arrival"]
    assert viewer._frame_labels == viewer.frame_labels

    # C2: a callback queued before cleanup cannot mutate the released viewer.
    viewer._mps_folder_shutdown = True
    multi.datasets.append(object())
    multi.names.append("late")
    Show4DSTEMMPS._on_multi_dataset_ready(viewer, 2)
    assert viewer.n_frames == 2
    assert viewer.frame_labels == ["first", "arrival"]


def test_mps_status_callback_marshals_to_mounted_viewer_loop() -> None:
    from quantem.widget.show4dstem_mps import Show4DSTEMMPS

    callbacks: list[object] = []
    loop = SimpleNamespace(
        asyncio_loop=SimpleNamespace(is_running=lambda: True),
        add_callback=lambda callback: callbacks.append(callback),
    )
    viewer = SimpleNamespace(
        _mps_folder_shutdown=False,
        _ioloop=loop,
        folder_watch_state="hidden",
        folder_watch_detail="",
    )

    # C1: an MPS worker publishes off the notebook thread, expect one queued
    # callback and no direct trait mutation before the event loop runs it.
    Show4DSTEMMPS._publish_mps_folder_watch_status(
        viewer,
        "watching",
        "Watching for completed masters.",
    )
    assert viewer.folder_watch_state == "hidden"
    assert len(callbacks) == 1
    callbacks.pop()()
    assert viewer.folder_watch_state == "watching"

    # C2: cleanup wins a race with an already queued callback, expect the late
    # state to be discarded rather than resurrecting a false green badge.
    Show4DSTEMMPS._publish_mps_folder_watch_status(
        viewer,
        "error",
        "late failure",
    )
    viewer._mps_folder_shutdown = True
    callbacks.pop()()
    assert viewer.folder_watch_state == "watching"


@pytest.mark.parametrize(
    ("watch_started", "expected_state"),
    [(False, "hidden"), (True, "stopped")],
)
def test_mps_shutdown_detaches_callbacks_and_preserves_snapshot_status(
    watch_started,
    expected_state,
) -> None:
    from quantem.widget.show4dstem_mps import Show4DSTEMMPS

    calls: list[str] = []
    multi = SimpleNamespace(on_ready=lambda idx: None)
    live = SimpleNamespace(
        _watch_started=watch_started,
        shutdown=lambda: calls.append("shutdown"),
    )
    viewer = SimpleNamespace(
        _mps_folder_shutdown=False,
        _mps_folder_live=live,
        _multi=multi,
        folder_watch_state="hidden",
        folder_watch_detail="",
    )

    # C1: cleanup detaches callbacks; watched viewers stop while snapshots stay hidden.
    Show4DSTEMMPS._shutdown_mps_folder_live(viewer)
    assert calls == ["shutdown"]
    assert viewer._mps_folder_shutdown is True
    assert multi.on_ready is None
    assert viewer.folder_watch_state == expected_state


def test_show4dstem_from_folder_mps_honors_public_folder_options(
    monkeypatch,
    tmp_path,
) -> None:
    import quantem.gpu.io.hdf5 as widget_io
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
        def set_status_callback(self, callback):
            calls["status_callback"] = callback
            callback("hidden", "")

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
        validate_master=None,
    ):
        calls["loader"] = {
            "paths": list(paths),
            "det_bin": det_bin,
            "scan_size": scan_size,
            "verbose": verbose,
            "skip_mps_memory_check": skip_mps_memory_check,
            "validate_master": validate_master,
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
    monkeypatch.setattr(
        factory,
        "_master_file_contract",
        lambda path: {
            "scan_shape": (256, 256),
            "detector_shape": (192, 192),
            "n_frames": 256 * 256,
            "dtype": "<u2",
        },
    )

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
        "validate_master": calls["loader"]["validate_master"],
    }
    assert callable(calls["loader"]["validate_master"])
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

    # C1: the live handle's callback is the mounted viewer's status bridge,
    # expect MPS lifecycle text to reach the same frontend traits as CUDA/CPU.
    status_callback = calls["status_callback"]
    assert callable(status_callback)
    status_callback("watching", "Watching for completed masters.")
    assert viewer.folder_watch_state == "watching"
    status_callback("waiting", "scan_004 is incomplete")
    assert viewer.folder_watch_state == "waiting"
    assert viewer.folder_watch_detail == "scan_004 is incomplete"

    assert viewer.poll_folder(async_=False) == [7]
    assert calls["poll"] == {
        "folder": tmp_path.resolve(),
        "pattern": "scan_*_master.h5",
        "recursive": False,
        "scan_size": 256,
        "ready_only": True,
        "async_": False,
        "require_stable": True,
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
