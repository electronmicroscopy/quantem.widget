import threading
import time
from pathlib import Path

import numpy as np
import pytest
import traitlets

from quantem.widget import Show2D, Show3D
from quantem.widget import io as widget_io
from quantem.widget.image_folder import WatchedImageFolder


def _save(path: Path, value: float, shape: tuple[int, int] = (6, 8)) -> None:
    np.save(path, np.full(shape, value, dtype=np.float32))


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached before the watcher timeout")


@pytest.mark.parametrize(
    ("viewer", "count_attr"),
    [(Show2D, "n_images"), (Show3D, "n_slices")],
)
def test_folder_background_watcher_appends_full_resolution_images(
    tmp_path: Path,
    viewer,
    count_attr: str,
) -> None:
    _save(tmp_path / "frame_1.npy", 1, shape=(17, 23))
    widget = viewer.from_folder(tmp_path, watch_interval=0.02)
    source = widget._folder_source
    thread = source._watch_thread
    try:
        assert thread is not None and thread.is_alive()
        _save(tmp_path / "frame_2.npy", 2, shape=(17, 23))
        _wait_until(lambda: int(getattr(widget, count_attr)) == 2)
        assert widget._data.shape[-2:] == (17, 23)

        _save(tmp_path / "frame_2.npy", 7, shape=(17, 23))
        time.sleep(0.08)
        assert float(widget._data[1, 0, 0]) == 2.0
    finally:
        widget.stop_folder_watch()
        widget.stop_folder_watch()
        assert not thread.is_alive()
        assert source._watch_thread is None
        widget.close()
        widget.close()


@pytest.mark.parametrize("viewer", [Show2D, Show3D])
def test_folder_poll_reports_updating_only_while_discovering(
    tmp_path: Path,
    viewer,
) -> None:
    _save(tmp_path / "frame_1.npy", 1)
    widget = viewer.from_folder(tmp_path, watch=True, watch_interval=60)
    source = widget._folder_source
    transitions: list[str] = []
    widget.observe(
        lambda change: transitions.append(change["new"]),
        names="folder_watch_state",
    )
    try:
        # Confirm the candidate that existed when the live mount opened, then
        # isolate the idle-poll transition contract below.
        assert widget.poll_folder() == [0]
        transitions.clear()

        # C1: an idle caller-owned scan reports Updating during discovery and
        # returns to Watching; a rejected concurrent scan emits nothing.
        assert widget.poll_folder() == []
        source._poll_lock.acquire()
        started = time.perf_counter()
        try:
            assert widget.poll_folder() == []
        finally:
            source._poll_lock.release()
        assert time.perf_counter() - started < 0.5
        assert transitions == ["updating", "watching"]
        assert widget.folder_watch_state == "watching"

        # C2: a new readable path first enters probation, then its confirming
        # decode performs one real apply bracketed by Updating and Watching.
        _save(tmp_path / "frame_2.npy", 2)
        assert widget.poll_folder() == []
        assert transitions[-2:] == ["updating", "waiting"]
        assert widget.poll_folder() == [1]
        assert transitions[-2:] == ["updating", "watching"]
    finally:
        widget.close()


@pytest.mark.parametrize("viewer", [Show2D, Show3D])
def test_folder_readable_candidate_must_stop_growing_before_append(
    tmp_path: Path,
    viewer,
) -> None:
    _save(tmp_path / "frame_1.npy", 1)
    widget = viewer.from_folder(tmp_path, watch=False)
    source = widget._folder_source
    arrival = tmp_path / "frame_2.npy"
    canonical = arrival.resolve()
    try:
        # C1: a newly readable file has only one decoded fingerprint, expect a
        # probationary no-op rather than immediate scientific-data exposure.
        _save(arrival, 2)
        assert widget.poll_folder() == []
        first_fingerprint = source._ready_probation[canonical]
        assert widget.labels == ["frame_1"]

        # C2: the readable file grows before confirmation, expect probation to
        # restart from its new fingerprint and still no append.
        with arrival.open("ab") as stream:
            stream.write(b"acquisition still growing")
        assert widget.poll_folder() == []
        assert source._ready_probation[canonical] != first_fingerprint
        assert widget.labels == ["frame_1"]

        # C3: a probationary path disappears, expect its state to be pruned;
        # recreating it then requires two new unchanged decodes before append.
        arrival.unlink()
        assert widget.poll_folder() == []
        assert canonical not in source._ready_probation
        _save(arrival, 3)
        assert widget.poll_folder() == []
        assert widget.poll_folder() == [1]
        assert widget.labels == ["frame_1", "frame_2"]
        np.testing.assert_array_equal(widget._data[:, 0, 0], [1, 3])
    finally:
        widget.close()


@pytest.mark.parametrize(("viewer", "count_attr"), [
    (Show2D, "n_images"),
    (Show3D, "n_slices"),
])
def test_folder_live_launch_probates_existing_readable_file(
    tmp_path: Path,
    viewer,
    count_attr: str,
) -> None:
    initial = tmp_path / "frame_1.npy"
    _save(initial, 1)
    widget = viewer.from_folder(tmp_path, watch=True, watch_interval=60)
    source = widget._folder_source
    canonical = initial.resolve()
    try:
        # C1: an image is readable when the live mount opens, expect the viewer
        # to mount immediately but keep it probationary until a later decode.
        assert int(getattr(widget, count_attr)) == 0
        assert canonical in source._ready_probation
        assert widget.folder_watch_state == "waiting"

        # C2: the writer changes the file after launch, expect the first later
        # decode to restart probation instead of freezing the paused content.
        first_fingerprint = source._ready_probation[canonical]
        _save(initial, 2)
        assert widget.poll_folder() == []
        assert source._ready_probation[canonical] != first_fingerprint
        assert int(getattr(widget, count_attr)) == 0

        # C3: only an unchanged follow-up decode exposes the completed image.
        assert widget.poll_folder() == [0]
        assert int(getattr(widget, count_attr)) == 1
        np.testing.assert_array_equal(widget._data[:, 0, 0], [2])
        assert widget.folder_watch_state == "watching"
    finally:
        widget.close()


@pytest.mark.parametrize("viewer", [Show2D, Show3D])
@pytest.mark.parametrize(("failure", "named"), [(RuntimeError, False), (KeyError, False), (AttributeError, True), (SystemExit, False)])
def test_a_watcher_that_dies_turns_the_badge_red(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, viewer, failure, named) -> None:
    # A per-poll error the watcher expects keeps it alive; anything else ends it with a red badge
    # (naming an Exception) and its traceback, instead of a stale green "watching".
    _save(tmp_path / "frame_1.npy", 1)
    widget = viewer.from_folder(tmp_path, watch=False)
    source = widget._folder_source
    tracebacks = []
    monkeypatch.setattr(threading, "excepthook", tracebacks.append)

    def failing_poll(_widget) -> list[int]:
        raise failure("decoder runtime aborted")

    monkeypatch.setattr(source, "poll", failing_poll)
    try:
        widget.watch_folder(interval=0.01)
        thread = source._watch_thread
        if failure in (RuntimeError, KeyError):
            _wait_until(lambda: "decoder runtime aborted" in widget.folder_watch_detail)
            assert thread.is_alive() and "still alive" in widget.folder_watch_detail
            return
        _wait_until(lambda: source._watch_thread is None and not thread.is_alive())
        assert source._watch_stop is None
        assert widget.folder_watch_state == "error"
        assert "stopped unexpectedly" in widget.folder_watch_detail
        assert ("AttributeError: decoder runtime aborted" in widget.folder_watch_detail) is named
        assert len(tracebacks) == 1
    finally:
        widget.close()


@pytest.mark.parametrize(
    ("viewer", "count_attr"),
    [(Show2D, "n_images"), (Show3D, "n_slices")],
)
def test_folder_empty_launch_arrival_restart_and_close_lifecycle(
    tmp_path: Path,
    viewer,
    count_attr: str,
    request,
) -> None:
    # C1: a watched acquisition starts empty, expect a mounted zero-record
    # model and a visible-status trait instead of a fake scientific frame.
    widget = viewer.from_folder(tmp_path, watch=True, watch_interval=0.02)
    request.addfinalizer(widget.close)
    source = widget._folder_source
    first_thread = source._watch_thread
    model_id = widget.model_id
    assert int(getattr(widget, count_attr)) == 0
    assert widget._data.shape == (0, 0, 0)
    assert widget.folder_waiting is True
    assert "Waiting for the first stable" in widget.folder_status
    assert widget.folder_watch_state == "watching"
    assert first_thread is not None and first_thread.is_alive()

    # C2: the first path is initially unreadable and later completes, expect
    # exactly one append on the same widget model while the watcher survives.
    first = tmp_path / "frame_1.npy"
    first.write_bytes(b"incomplete npy")
    _wait_until(
        lambda: (
            first.resolve() in widget._folder_source.errors
            and widget.folder_watch_state == "waiting"
        )
    )
    assert int(getattr(widget, count_attr)) == 0
    assert widget.folder_watch_state == "waiting"
    assert "pending file" in widget.folder_watch_detail
    _save(first, 1, shape=(17, 23))
    _wait_until(lambda: (
        int(getattr(widget, count_attr)) == 1
        and not widget.folder_waiting
        and widget.folder_watch_state == "watching"
    ))
    assert widget.model_id == model_id
    assert widget.labels == ["frame_1"]
    assert widget._data.shape == (1, 17, 23)

    # C3: stop is idempotent and blocks mutation; restarting uses a new thread
    # and catches up, while close prevents any later file from mutating state.
    widget.stop_folder_watch()
    widget.stop_folder_watch()
    assert widget.folder_watch_state == "stopped"
    assert first_thread is not None and not first_thread.is_alive()
    _save(tmp_path / "frame_2.npy", 2, shape=(17, 23))
    time.sleep(0.06)
    assert int(getattr(widget, count_attr)) == 1
    widget.watch_folder(interval=0.02)
    second_thread = source._watch_thread
    assert second_thread is not None and second_thread is not first_thread
    assert widget.folder_watch_state == "watching"
    _wait_until(lambda: int(getattr(widget, count_attr)) == 2)
    widget.close()
    assert widget.folder_watch_state == "stopped"
    assert source._watch_thread is None
    _save(tmp_path / "frame_3.npy", 3, shape=(17, 23))
    time.sleep(0.06)
    assert int(getattr(widget, count_attr)) == 2


@pytest.mark.parametrize("viewer", [Show2D, Show3D])
def test_folder_empty_snapshot_without_watch_still_raises(
    tmp_path: Path,
    viewer,
) -> None:
    # C1: a fixed empty snapshot has no future arrival path, expect the legacy
    # actionable error instead of returning a permanently empty viewer.
    with pytest.raises(FileNotFoundError, match="No readable 2D"):
        viewer.from_folder(tmp_path, watch=False)


@pytest.mark.parametrize("viewer", [Show2D, Show3D])
def test_folder_fixed_snapshot_has_no_watch_badge(
    tmp_path: Path,
    viewer,
) -> None:
    # C1: a fixed populated snapshot never starts a worker, expect the shared
    # frontend protocol to remain hidden rather than showing a false green dot.
    _save(tmp_path / "frame_1.npy", 1)
    widget = viewer.from_folder(tmp_path, watch=False)
    try:
        assert widget.folder_watch_state == "hidden"
        assert widget.folder_watch_detail == ""
        assert widget._folder_source._watch_thread is None
    finally:
        widget.close()


@pytest.mark.parametrize("viewer", [Show2D, Show3D])
def test_folder_live_saved_state_never_serializes_false_green(
    tmp_path: Path,
    viewer,
) -> None:
    _save(tmp_path / "frame_1.npy", 1)
    widget = viewer.from_folder(tmp_path, watch=True, watch_interval=60)
    try:
        assert widget.poll_folder() == [0]
        assert widget.folder_watch_state == "watching"

        # C1: the mounted model still has a real worker, but a saved/exported
        # model cannot retain that daemon and must reopen as a gray snapshot.
        snapshot = widget.get_state()
        assert widget.folder_watch_state == "watching"
        assert snapshot["folder_watch_state"] == "stopped"
        assert "Re-run the cell" in snapshot["folder_watch_detail"]

        # C2: the initial comm-open handshake represents the actually running
        # model, while arbitrary assignments outside the protocol are rejected.
        widget._initial_live_mount_state = True
        try:
            assert widget.get_state()["folder_watch_state"] == "watching"
        finally:
            widget._initial_live_mount_state = False
        with pytest.raises(traitlets.TraitError):
            widget.folder_watch_state = "false-green"
    finally:
        widget.close()


@pytest.mark.parametrize("viewer", [Show2D, Show3D])
def test_folder_failed_and_mismatched_files_remain_retryable(
    tmp_path: Path,
    viewer,
) -> None:
    _save(tmp_path / "frame_1.npy", 1)
    widget = viewer.from_folder(tmp_path, watch=False)
    try:
        # C1: an unreadable new file stays pending, expect no partial append.
        partial = tmp_path / "frame_2.npy"
        partial.write_bytes(b"not a complete npy file")
        assert widget.poll_folder() == []
        assert partial.resolve() in widget._folder_source.errors
        assert [path.name for path in widget.folder_paths] == ["frame_1.npy"]

        _save(partial, 2)
        assert widget.poll_folder() == []
        assert widget.poll_folder() == [1]

        # C2: a persistent mismatch precedes a compatible file, expect the
        # compatible file to append without resizing or removing the mismatch.
        mismatch = tmp_path / "frame_3.npy"
        _save(mismatch, 3, shape=(3, 4))
        _save(tmp_path / "frame_4.npy", 4)
        assert widget.poll_folder() == []
        assert widget.poll_folder() == [2]
        assert mismatch.resolve() in widget._folder_source.errors
        assert "expected (6, 8), got (3, 4)" in widget._folder_source.errors[mismatch.resolve()]
        assert [path.name for path in widget.folder_paths] == [
            "frame_1.npy",
            "frame_2.npy",
            "frame_4.npy",
        ]
        assert "1 pending file" in widget.folder_status
        assert widget.poll_folder() == []

        # C3: the mismatch is corrected in place, expect one naturally ordered
        # append and no duplicate of the already represented later file.
        _save(mismatch, 3)
        assert widget.poll_folder() == []
        assert widget.poll_folder() == [2]
        assert [path.name for path in widget.folder_paths] == [
            "frame_1.npy",
            "frame_2.npy",
            "frame_3.npy",
            "frame_4.npy",
        ]
        np.testing.assert_array_equal(widget._data[:, 0, 0], [1, 2, 3, 4])
        assert widget.folder_status == ("4 ready panels" if viewer is Show2D else "4 ready frames")
    finally:
        widget.close()


@pytest.mark.parametrize("viewer", [Show2D, Show3D])
def test_folder_live_shape_error_badge_is_red_protocol_state(
    tmp_path: Path,
    viewer,
) -> None:
    _save(tmp_path / "frame_1.npy", 1)
    widget = viewer.from_folder(tmp_path, watch=True, watch_interval=10)
    try:
        assert widget.poll_folder() == [0]
        # C1: an active watcher sees an incompatible scientific shape, expect
        # a corrective error state while the still-live worker never looks green.
        mismatch = tmp_path / "frame_2.npy"
        _save(mismatch, 2, shape=(3, 4))
        assert widget.poll_folder() == []
        assert widget.folder_watch_state == "error"
        assert "Incompatible image shape" in widget.folder_watch_detail
        assert widget._folder_source._watch_thread.is_alive()

        # C2: correcting the file succeeds on the same worker, expect the
        # protocol to wait for one unchanged decode, then return to Watching.
        _save(mismatch, 2)
        assert widget.poll_folder() == []
        assert widget.folder_watch_state == "waiting"
        assert widget.poll_folder() == [1]
        assert widget.folder_watch_state == "watching"
    finally:
        widget.close()


@pytest.mark.parametrize("viewer", [Show2D, Show3D])
def test_folder_initial_mismatch_does_not_block_later_valid_file(
    tmp_path: Path,
    viewer,
) -> None:
    # C1: an initial folder contains valid, mismatched, then valid files,
    # expect construction to retain both compatible files and report the other.
    _save(tmp_path / "frame_1.npy", 1)
    mismatch = tmp_path / "frame_2.npy"
    _save(mismatch, 2, shape=(3, 4))
    _save(tmp_path / "frame_3.npy", 3)

    widget = viewer.from_folder(tmp_path, watch=False)
    try:
        assert widget.labels == ["frame_1", "frame_3"]
        np.testing.assert_array_equal(widget._data[:, 0, 0], [1, 3])
        assert mismatch.resolve() in widget._folder_source.errors

        # C2: correcting the skipped file later, expect insertion at its natural
        # index without rebuilding the widget object.
        widget_id = id(widget)
        _save(mismatch, 2)
        assert widget.poll_folder() == []
        assert widget.poll_folder() == [1]
        assert id(widget) == widget_id
        assert widget.labels == ["frame_1", "frame_2", "frame_3"]
        np.testing.assert_array_equal(widget._data[:, 0, 0], [1, 2, 3])
    finally:
        widget.close()


def test_watched_folder_empty_opt_in_anchors_shape_and_prunes_errors(
    tmp_path: Path,
) -> None:
    # C1: the initial read of an empty folder fails, while the watched empty
    # start opens a waiting widget with no scale bar.
    with pytest.raises(FileNotFoundError, match="No readable 2D"):
        WatchedImageFolder(tmp_path, interval=0.02, mode="panels").read_initial()
    widget = Show2D.from_folder(tmp_path, watch=True, watch_interval=60)
    source = widget._folder_source
    try:
        assert widget.folder_waiting is True
        assert widget.pixel_size == 0.0
        assert widget.show_scale_bar is False
        assert "waiting" in source.calibration_status

        # C2: the first poll sees compatible files around one mismatch, expect
        # the first valid file to anchor shape and the later valid file to pass.
        _save(tmp_path / "frame_1.npy", 1, shape=(7, 9))
        mismatch = tmp_path / "frame_2.npy"
        _save(mismatch, 2, shape=(3, 4))
        _save(tmp_path / "frame_3.npy", 3, shape=(7, 9))
        assert widget.poll_folder() == []
        assert widget.poll_folder() == [0, 1]
        assert source.expected_shape == (7, 9)
        assert [path.name for path in source.paths] == ["frame_1.npy", "frame_3.npy"]
        assert mismatch.resolve() in source.errors
        assert widget.folder_waiting is False
        assert widget.show_scale_bar is True

        # C3: a rejected file disappears, expect its stale diagnostic to be pruned
        # without changing represented records.
        mismatch.unlink()
        assert widget.poll_folder() == []
        assert source.errors == {}
        assert widget.folder_status == "2 ready panels"
    finally:
        widget.close()


def test_watched_folder_empty_start_accepts_rgb_files(tmp_path: Path) -> None:
    widget = Show3D.from_folder(tmp_path, watch=True, watch_interval=60)
    source = widget._folder_source
    try:
        # C1: an RGB file arrives in a folder watched from empty, expect it to
        # anchor the spatial (rows, cols) shape and appear after its confirming
        # poll instead of failing the shape check against its own channel axis.
        np.save(tmp_path / "frame_1.npy", np.ones((6, 8, 3), dtype=np.uint8))
        assert widget.poll_folder() == []
        assert widget.poll_folder() == [0]
        assert source.expected_shape == (6, 8)
        assert source.errors == {} and widget.is_rgb
    finally:
        widget.close()


def test_watched_folder_natural_order_has_exact_path_tie_breaker(
    tmp_path: Path,
) -> None:
    # C1: numeric-equivalent names are created in reverse lexical order, expect
    # the exact relative path to provide a deterministic final tie breaker.
    _save(tmp_path / "frame_1.npy", 1)
    _save(tmp_path / "frame_01.npy", 1)
    source = WatchedImageFolder(tmp_path, mode="frames")

    assert [path.name for path in source.discover()] == [
        "frame_01.npy",
        "frame_1.npy",
    ]


def test_watched_folder_invalid_restart_preserves_thread(tmp_path: Path) -> None:
    widget = Show2D.from_folder(tmp_path, watch=True, watch_interval=0.02)
    source = widget._folder_source
    thread = source._watch_thread
    try:
        assert thread is not None and thread.is_alive()
        # an invalid interval fails validation before the healthy thread is stopped
        with pytest.raises(ValueError, match="finite value > 0"):
            widget.watch_folder(interval=0)
        assert source._watch_thread is thread and thread.is_alive()
    finally:
        widget.close()
    assert source._watch_thread is None and not thread.is_alive()


@pytest.mark.parametrize("viewer", [Show2D, Show3D])
def test_folder_widget_append_preserves_display_bin_and_calibration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    viewer,
) -> None:
    class Dataset:
        def __init__(self, value: int) -> None:
            self.array = np.full((8, 10), value, dtype=np.uint16)
            self.sampling = (0.25, 0.5)
            self.units = ("nm", "nm")

    monkeypatch.setattr(
        widget_io,
        "read_image",
        lambda path: Dataset(int(Path(path).name[6])),
    )
    (tmp_path / "frame_1.fake.npy").write_bytes(b"first")
    widget = viewer.from_folder(tmp_path, watch=False, display_bin=2)
    try:
        # C1: a calibrated uint16 source uses a 2x display preview, expect
        # native data/value parity and calibration in display-pixel units.
        assert widget._data.shape == (1, 8, 10)
        assert widget._data.dtype == np.float32
        assert float(widget._data[0, 0, 0]) == 1.0
        assert widget._display_data.shape == (1, 4, 5)
        assert widget.pixel_size == pytest.approx(1.0)

        # C2: a watched append uses set_image in place, expect the explicit
        # display factor and scale-bar calibration to remain unchanged.
        (tmp_path / "frame_2.fake.npy").write_bytes(b"second")
        assert widget.poll_folder() == []
        assert widget.poll_folder() == [1]
        assert widget._data.shape == (2, 8, 10)
        np.testing.assert_array_equal(widget._data[:, 0, 0], [1, 2])
        assert widget._display_data.shape == (2, 4, 5)
        assert widget.pixel_size == pytest.approx(1.0)
        assert widget.show_scale_bar is True
        if viewer is Show2D:
            assert widget.display_bin == 2
        else:
            assert widget._display_bin == 2
    finally:
        widget.close()


def test_show2d_empty_first_recomputes_auto_display_bin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # C1: the bootstrap 1x1 waiting state cannot choose a useful auto preview;
    # expect the first real frame to recompute the wire-budget factor.
    monkeypatch.setattr(Show2D, "_WIRE_BUDGET_BYTES_PER_PANEL", 100)
    widget = Show2D.from_folder(tmp_path, watch=True, watch_interval=60, display_bin="auto")
    try:
        assert widget.n_images == 0
        assert widget.display_bin == 1
        _save(tmp_path / "frame_1.npy", 1, shape=(16, 16))
        assert widget.poll_folder() == []
        assert widget.poll_folder() == [0]
        assert widget.display_bin == 4
        assert widget._data.shape == (1, 16, 16)
        assert widget._display_data.shape == (1, 4, 4)
    finally:
        widget.close()


@pytest.mark.parametrize("viewer", [Show2D, Show3D])
def test_folder_calibration_is_preserved_until_files_disagree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    viewer,
) -> None:
    class Dataset:
        def __init__(self, value: float, sampling: float) -> None:
            self.array = np.full((5, 7), value, dtype=np.float32)
            self.sampling = (sampling, sampling)
            self.units = ("nm", "nm")

    sampling_by_name = {"frame_1.fake.npy": 0.2, "frame_2.fake.npy": 0.2}

    def read_image(path: Path) -> Dataset:
        return Dataset(float(path.name[6]), sampling_by_name[path.name])

    monkeypatch.setattr(widget_io, "read_image", read_image)
    for name in sampling_by_name:
        (tmp_path / name).write_bytes(name.encode())

    widget = viewer.from_folder(tmp_path, watch=False)
    try:
        assert widget.pixel_size == pytest.approx(0.2)
        assert widget.pixel_unit == "nm"
        assert widget.show_scale_bar is True

        sampling_by_name["frame_3.fake.npy"] = 0.3
        (tmp_path / "frame_3.fake.npy").write_bytes(b"new mixed calibration")
        assert widget.poll_folder() == []
        assert widget.poll_folder() == [2]

        assert widget.pixel_size == 0.0
        assert widget.show_scale_bar is False
        assert "different sampling or units" in widget._folder_source.calibration_status
    finally:
        widget.close()


def test_folder_reads_symlinked_files_with_extensionless_targets(
    tmp_path: Path,
) -> None:
    # Hugging Face hub cache folders store files as symlinks to extension-less
    # blobs; format dispatch must use the symlink's own name, not its target.
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    named = blobs / "blob.npy"
    _save(named, 2.5)
    blob = blobs / "abc123"
    named.rename(blob)
    session = tmp_path / "session"
    session.mkdir()
    (session / "frame_000.npy").symlink_to(blob)

    widget = Show2D.from_folder(session, watch=False)
    assert widget.labels == ["frame_000"]
    assert widget._data.shape == (1, 6, 8)
