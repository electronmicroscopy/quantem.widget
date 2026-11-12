"""Idle microscope listeners adopt arriving scans without owning their counts."""

import asyncio
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from quantem.widget import Show4DSTEM
from quantem.widget.show4dstem import widget as widget_module
from quantem.widget.show4dstem.dense import DenseSession


class LiveScan:
    """A CPU source implementing the same borrowed-session contract as live counts."""

    def __init__(self, series, value=1):
        self.counts = torch.full((3, 4, 8, 10), value, dtype=torch.float32)
        self.metadata = {"series": series, "scan_sampling_A": [0.4, 0.6], "voltage_kV": 300.0}
        self.received = self.total = 12
        self.sessions = []

    def detector_session(self):
        session = DenseSession(self.counts)
        session.metadata = self.metadata
        self.sessions.append(session)
        return session


@pytest.fixture
def recording(monkeypatch):
    # Keep polling deterministic; refresh_stream is the notebook's public refresh.
    monkeypatch.setattr(widget_module.threading.Thread, "start", lambda self: None)
    monkeypatch.setattr(widget_module.threading.Thread, "join", lambda self, **kwargs: None)
    return SimpleNamespace(lock=threading.RLock(), acquisitions=[], datasets=[], receiving=True, error=None)


def test_idle_listener_receives_unloads_and_receives_again(recording):
    viewer = Show4DSTEM.stream(recording)
    try:
        assert viewer.n_frames == 0 and viewer._data is None
        assert viewer.title == "Waiting for acquisition"
        assert np.isnan(np.frombuffer(viewer.frame_bytes, np.float32)).all()
        viewer.refresh_stream()
        assert viewer.stream_error is None

        first = LiveScan("first")
        first.received = 0
        recording.acquisitions.append(first)
        viewer.refresh_stream()
        assert viewer.n_frames == 0 and not first.sessions
        first.received = first.total
        viewer.refresh_stream()
        assert viewer.n_frames == 1 and viewer.frame_labels == ["first"]
        assert (viewer.shape_rows, viewer.shape_cols, viewer.det_rows, viewer.det_cols) == (3, 4, 8, 10)
        assert viewer.scan_sampling_A == [0.4, 0.6] and viewer.voltage_kV == 300.0
        np.testing.assert_array_equal(viewer.pattern(1, 2), first.counts[1, 2].numpy())
        viewer.apply_preset("adf")
        viewer.pos_row, viewer.pos_col = 1, 2
        detector = (viewer.roi_center, viewer.roi_radius_inner, viewer.roi_radius)

        recording.acquisitions.clear()
        viewer.refresh_stream()
        viewer.refresh_stream()
        assert viewer.n_frames == 0 and viewer._data is None and viewer.stream_error is None
        assert first.sessions[0]._flat_t is None
        assert np.isnan(np.frombuffer(viewer.virtual_image_bytes, np.float32)).all()

        second = LiveScan("second", value=2)
        recording.acquisitions.append(second)
        viewer.refresh_stream()
        assert viewer.n_frames == 1 and viewer.frame_labels == ["second"]
        assert (viewer.pos_row, viewer.pos_col) == (1, 2)
        assert (viewer.roi_center, viewer.roi_radius_inner, viewer.roi_radius) == detector
        expected = second.counts.numpy()[..., viewer._current_detector_mask()].sum(-1)
        np.testing.assert_array_equal(viewer.virtual_image(), expected)
        np.testing.assert_array_equal(viewer.pattern(1, 2), second.counts[1, 2].numpy())
    finally:
        viewer.close()
    assert second.sessions[0]._flat_t is None
    assert second.counts.sum() > 0


def test_idle_first_arrival_matches_opening_the_same_scan(recording):
    options = {"labels": ["stated label"], "bright_field_center_px": (3.0, 4.0),
               "bright_field_radius_px": 2.0, "title": "Microscope"}
    waiting = Show4DSTEM.stream(recording, **options)
    scan = LiveScan("metadata label")
    recording.acquisitions.append(scan)
    opened = Show4DSTEM.stream(recording, first_timeout_s=1, **options)
    try:
        waiting.refresh_stream()
        opened.refresh_stream()
        for name in ("frame_labels", "scan_sampling_A", "voltage_kV", "roi_center", "roi_radius",
                     "frame_dim_label", "view_mode", "compare_dp_mode", "title"):
            assert getattr(waiting, name) == getattr(opened, name)
        np.testing.assert_array_equal(waiting.virtual_image(), opened.virtual_image())
        np.testing.assert_array_equal(waiting.pattern(0, 0), opened.pattern(0, 0))
    finally:
        waiting.close()
        opened.close()


def test_idle_listener_reports_receiver_failure_and_closes(recording):
    viewer = Show4DSTEM.stream(recording)
    failure = RuntimeError("Receiver disconnected")
    recording.error = failure
    try:
        assert viewer._stream_update() is False
        assert viewer.stream_error is failure
        assert viewer.title == "Live acquisition stopped: Receiver disconnected"
        assert viewer.n_frames == 0
    finally:
        viewer.close()
    assert viewer._stream_stop.is_set()


def test_live_counts_remain_exact_through_completion_and_eviction(recording):
    """Completed scans batch while an arriving scan still exposes missing positions."""
    if not torch.cuda.is_available():
        pytest.skip("live ANS encoding requires CUDA")
    cp = pytest.importorskip("cupy")
    from quantem.gpu.streaming import LiveEncoder

    counts = np.random.default_rng(5).integers(0, 20, (3, 32, 16, 16, 16), dtype=np.uint16)
    acquisitions = [LiveEncoder((32, 16), (16, 16), hot_pixel_correction="none") for _ in counts]
    viewer = Show4DSTEM.stream(recording, compare_dp_mode="all")
    try:
        for index, acquisition in enumerate(acquisitions):
            acquisition.add(0, cp.asarray(counts[index].reshape(-1, 16, 16)[:256 if index == 2 else 512]))
            recording.acquisitions.append(acquisition)
            if acquisition.complete:
                recording.datasets.append(acquisition.finish())
        viewer.refresh_stream()
        with viewer.hold_trait_notifications():
            viewer.pos_row, viewer.pos_col = 20, 7
        panels = viewer.compare_panel_indices
        actual = np.frombuffer(viewer.compare_diffraction_bytes, np.float32).reshape(-1, 16, 16)
        for index, frame in zip(panels, actual):
            if index == 2:
                assert np.isnan(frame).all()
            else:
                np.testing.assert_array_equal(frame, counts[index, 20, 7])
        viewer.compare_dp_mode = "average"
        np.testing.assert_array_equal(np.frombuffer(viewer.frame_bytes, np.float32).reshape(16, 16),
                                      counts[:2, 20, 7].astype(np.float32).mean(0))

        recording.acquisitions.pop(0)
        recording.datasets.pop(0)
        acquisitions[0]._retire()
        assert not acquisitions[0].source.is_released  # viewer still borrows this scan
        viewer.refresh_stream()
        assert acquisitions[0].source.is_released
        acquisitions[2].add(256, cp.asarray(counts[2].reshape(-1, 16, 16)[256:]))
        recording.datasets.append(acquisitions[2].finish())
        viewer.refresh_stream()
        viewer.compare_dp_mode = "all"
        viewer.roi_center = [6.5, 7.5]
        mask = viewer._current_detector_mask()
        panels = viewer.compare_panel_indices
        images = np.frombuffer(viewer.compare_virtual_image_bytes, np.float32).reshape(-1, 32, 16)
        for index, image in zip(panels, images):
            expected = counts[index + 1][..., mask].sum(-1).astype(np.float32) / np.float32(mask.sum())
            np.testing.assert_array_equal(image, expected)
        assert recording.receiving  # this optimization must not depend on stopping the receiver
    finally:
        viewer.close()
        for acquisition in acquisitions:
            acquisition.release()


def test_stream_updates_while_notebook_waits():
    """Arriving counts still reach the viewer while a cell blocks its event loop."""
    async def notebook():
        recording = SimpleNamespace(lock=threading.RLock(), acquisitions=[], datasets=[],
                                    receiving=True, error=None)
        viewer = Show4DSTEM.stream(recording, refresh_s=0.01)
        try:
            scan = LiveScan("7")
            scan.received = 4
            recording.acquisitions.append(scan)
            deadline = time.monotonic() + 5
            # Deliberately synchronous, just like replay.wait()/arina_stream.wait().
            while "4/12" not in viewer.title and time.monotonic() < deadline:
                time.sleep(0.01)
            assert "Live · 7 · 4/12 · " in viewer.title
            scan.counts.fill_(2)
            scan.received = scan.total
            while "✓" not in viewer.title and time.monotonic() < deadline:
                time.sleep(0.01)
            assert "Live · 7 · 12/12 ✓" in viewer.title
            np.testing.assert_array_equal(viewer.pattern(1, 2), scan.counts[1, 2].numpy())
            assert viewer.stream_error is None
        finally:
            viewer.close()
        assert not viewer._stream_thread.is_alive()

    # A notebook thread of its own: a browser test earlier in the run (Playwright) leaves an event
    # loop running on this one, where asyncio.run() refuses to start.
    errors = []

    def kernel():
        try:
            asyncio.run(notebook())
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=kernel)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]


def test_stream_coalesces_until_browser_receipt_and_delivers_final_counts(recording, monkeypatch):
    viewer = Show4DSTEM.stream(recording)
    viewer._stream_display_credits = 1  # Exercise a full window with one pending update.
    messages = []
    monkeypatch.setattr(viewer, "send", lambda message: messages.append(message))

    def acknowledge(message=None):
        message = messages[-1] if message is None else message
        viewer._handle_frontend_ready_msg(None, dict(message, type="show4dstem_stream_ack"), None)

    try:
        viewer._handle_frontend_ready_msg(None, {"type": "show4dstem_frontend_ready", "version": 1,
                                                "stream_ack": True, "view": "browser"}, None)
        acknowledge()
        scan = LiveScan("live")
        scan.received = 2
        recording.acquisitions.append(scan)
        assert viewer._stream_update()
        first = messages[-1]
        for received in (4, 6, 8):
            scan.received = received
            assert viewer._stream_update()
        assert messages[-1] is first and "2/12" in viewer.title
        acknowledge(dict(first, view="old-browser"))
        assert viewer._stream_display_pending
        acknowledge(first)
        assert viewer._stream_update() and "8/12" in viewer.title
        assert len(messages) == 3  # ready, first counts, latest counts

        scan.received = scan.total
        scan.counts.fill_(5)
        recording.receiving = False
        assert viewer._stream_update() and "8/12" in viewer.title
        acknowledge()
        assert viewer._stream_update() and "12/12" in viewer.title
        np.testing.assert_array_equal(viewer.pattern(1, 2), scan.counts[1, 2].numpy())
        acknowledge(first)  # A delayed ACK cannot release a newer final update.
        assert viewer._stream_update()
        acknowledge()
        assert viewer._stream_update() is False
        assert viewer.stream_error is None
    finally:
        viewer.close()


def test_busy_cell_preview_is_limited_and_idle_cell_never_bypasses_ack(recording, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(widget_module.time, "monotonic", lambda: clock[0])
    viewer = Show4DSTEM.stream(recording)
    viewer._stream_display_credits = 1  # Exercise a full window with one pending update.
    messages = []
    monkeypatch.setattr(viewer, "send", lambda message: messages.append(message))
    try:
        viewer._handle_frontend_ready_msg(None, {"type": "show4dstem_frontend_ready", "version": 1,
                                                "stream_ack": True, "view": "browser"}, None)
        scan = LiveScan("live")
        recording.acquisitions.append(scan)
        viewer._stream_display_sent_at = time.monotonic() - 3
        assert viewer._stream_update() and viewer.n_frames == 0  # Idle: no timeout bypass.
        viewer._stream_cell_running = True
        assert viewer._stream_update() and viewer.n_frames == 1
        first = messages[-1]
        scan.received = 11
        assert viewer._stream_update() and messages[-1] is first
        clock[0] += 2.1
        assert viewer._stream_update() and messages[-1]["seq"] > first["seq"]
        assert "11/12" in viewer.title
    finally:
        viewer.close()


@pytest.mark.parametrize("credits", [2, 4])
def test_stream_credits_preserve_delta_order_retirement_reconnect_and_final_state(recording, monkeypatch, credits):
    scans = [LiveScan(str(index), value=index + 1) for index in range(4)]
    scans[2].received = scans[3].received = 1
    recording.acquisitions[:] = scans[:3]
    viewer = Show4DSTEM.stream(recording, view_mode="multiple", compare_group_mode="all",
                              device="cpu", save_state=True, notebook_preview_format=None)
    messages, states = [], []
    monkeypatch.setattr(viewer, "send", lambda message, buffers=None: messages.append((message, buffers)))
    monkeypatch.setattr(viewer, "send_state", lambda key=None: states.append(viewer.get_state(key)) if key else None)

    def message(kind, view="browser", **fields):
        viewer._handle_frontend_ready_msg(None, {"type": f"show4dstem_{kind}", "version": 1,
                                                "view": view, **fields}, None)

    try:
        assert viewer._stream_display_credits == 2
        viewer._stream_display_credits = credits
        viewer.refresh_stream()
        message("frontend_ready", stream_ack=True, stream_delta=True, stream_layout=True)
        message("stream_ack", seq=viewer._stream_display_seq)
        initial_revision = viewer.compare_image_revision
        messages.clear()
        for received in range(2, 2 + credits):
            scans[2].received = received
            scans[2].counts.fill_(received + 3)
            assert viewer._stream_update()
        deltas = [value for value, _ in messages if value["type"] == "show4dstem_stream_panels"]
        assert len(deltas) == credits
        assert deltas[0]["base"] == initial_revision
        assert all(right["base"] == left["revision"] for left, right in zip(deltas, deltas[1:]))
        first, last = viewer._stream_display_seq - credits + 1, viewer._stream_display_seq
        published = bytes(viewer.compare_virtual_image_bytes)

        for seq, view in [(last + 1, "browser"), (last, "old"), (0, "browser"), (True, "browser"), (str(last), "browser")]:
            message("stream_ack", view, seq=seq)
        assert viewer._stream_display_seq - viewer._stream_display_acknowledged == credits
        states.clear()
        recording.acquisitions[:] = scans[1:]
        assert viewer._stream_update()  # Release retired sources even with all credits occupied.
        assert not states and scans[0].sessions[-1]._flat_t is None
        message("stream_ack", seq=first)
        assert viewer._stream_update()
        patch = states[-1]["_stream_panel_update"]
        assert patch["base"] == deltas[-1]["revision"]
        size = viewer.shape_rows * viewer.shape_cols * 4
        rebuilt = bytearray(len(patch["reuse"]) * size)
        for slot, source in enumerate(patch["reuse"]):
            if source >= 0:
                rebuilt[slot * size:(slot + 1) * size] = published[source * size:(source + 1) * size]
        for slot, data in zip(patch["slots"], patch["buffers"]):
            rebuilt[slot * size:(slot + 1) * size] = data
        assert rebuilt == bytes(viewer.compare_virtual_image_bytes)
        message("stream_ack", seq=first)  # Duplicate cannot release the newer updates.
        assert viewer._stream_display_seq - viewer._stream_display_acknowledged == credits

        recording.receiving = False
        scans[3].counts.fill_(19)
        scans[3].received = scans[3].total
        assert viewer._stream_update()
        message("stream_ack", seq=viewer._stream_display_seq)  # One cumulative receipt releases all.
        assert viewer._stream_update()
        saved = viewer.get_state()
        assert bytes(saved["compare_virtual_image_bytes"]) == bytes(viewer.compare_virtual_image_bytes)
        assert "_stream_panel_update" not in saved
        old_seq = viewer._stream_display_seq
        message("frontend_ready", "reopened", stream_ack=True, stream_delta=True, stream_layout=True)
        assert viewer._stream_display_acknowledged == old_seq
        assert viewer._stream_panel_update == {} and "compare_virtual_image_bytes" in states[-1]
        message("stream_ack", seq=old_seq)
        assert viewer._stream_display_pending
        message("stream_ack", "reopened", seq=viewer._stream_display_seq)
        assert viewer._stream_update() is False
        assert viewer.stream_error is None
    finally:
        viewer.close()
    assert viewer._stream_stop.is_set()


def test_initial_busy_cell_limits_previews_before_frontend_handshake(recording, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(widget_module.time, "monotonic", lambda: clock[0])
    viewer = Show4DSTEM.stream(recording)
    try:
        viewer._stream_cell_running = True
        scan = LiveScan("live")
        scan.received = 2
        recording.acquisitions.append(scan)
        assert viewer._stream_update() and "2/12" in viewer.title
        scan.received = 4
        clock[0] += 0.1
        assert viewer._stream_update() and "2/12" in viewer.title
        clock[0] += 2
        assert viewer._stream_update() and "4/12" in viewer.title
    finally:
        viewer.close()


def test_duplicate_views_recover_after_unmount_or_hard_close_without_bulk_retries(recording, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(widget_module.time, "monotonic", lambda: clock[0])
    viewer = Show4DSTEM.stream(recording)
    messages, states = [], []
    monkeypatch.setattr(viewer, "send", messages.append)
    monkeypatch.setattr(viewer, "send_state", lambda key=None: states.append(key) if key else None)

    def message(kind, view, **fields):
        viewer._handle_frontend_ready_msg(None, {"type": f"show4dstem_{kind}", "version": 1,
                                                "view": view, **fields}, None)

    try:
        for view in ("first", "second"):
            message("frontend_ready", view, stream_ack=True)
        pending = viewer._stream_display_pending
        bulk_count = len(states)
        message("frontend_closed", "second")
        assert messages[-1] == {"type": "show4dstem_stream_update", "version": 1, "view": "first",
                                "seq": pending, "revision": viewer._stream_display_revision}
        message("stream_ack", "second", seq=pending)
        assert viewer._stream_display_pending == pending
        message("stream_ack", "first", seq=pending)
        assert viewer._stream_display_pending == 0 and len(states) == bulk_count

        message("frontend_ready", "second", stream_ack=True)
        pending = viewer._stream_display_pending
        bulk_count = len(states)
        clock[0] += 2.1  # The second tab was killed, so no cleanup message arrived.
        assert viewer._stream_update()
        assert messages[-1] == {"type": "show4dstem_stream_probe", "version": 1, "seq": pending}
        assert len(states) == bulk_count
        message("stream_alive", "first", seq=pending)
        assert messages[-1]["view"] == "first"
        message("stream_alive", "second", seq=pending)  # Only the first probe response takes ownership.
        assert viewer._stream_display_view == "first"
        message("stream_ack", "first", seq=pending)
        assert viewer._stream_display_pending == 0

        message("frontend_closed", "second")
        message("frontend_closed", "first")
        viewer._stream_cell_running = True
        recording.acquisitions.append(LiveScan("new"))
        clock[0] += 3
        assert viewer._stream_update() and viewer.n_frames == 0
        assert len(states) == bulk_count  # No live display: even busy cells cannot queue images.
        message("frontend_ready", "reopened", stream_ack=True)
        message("stream_ack", "reopened", seq=viewer._stream_display_pending)
        viewer._stream_cell_running = False
        assert viewer._stream_update() and viewer.frame_labels == ["new"]
    finally:
        viewer.close()


def test_retired_sessions_release_without_browser_ack_and_reopen_shows_latest(recording, monkeypatch):
    first = LiveScan("first")
    recording.acquisitions.append(first)
    viewer = Show4DSTEM.stream(recording)
    viewer._stream_display_credits = 1  # Exercise a full window with one pending update.
    messages, states = [], []
    monkeypatch.setattr(viewer, "send", lambda message: messages.append(message))
    monkeypatch.setattr(viewer, "send_state", lambda key=None: states.append(key) if key else None)
    try:
        ready = {"type": "show4dstem_frontend_ready", "version": 1, "stream_ack": True, "view": "old"}
        viewer._handle_frontend_ready_msg(None, ready, None)
        original = viewer.virtual_image_bytes
        for index in range(2, 6):
            previous = recording.acquisitions[0]
            current = LiveScan(str(index), value=index)
            recording.acquisitions[:] = [current]
            assert viewer._stream_update()
            assert previous.sessions[-1]._flat_t is None
            assert len(messages) == len(states) == 1  # No bulk traffic or metadata-only mismatches.
            assert viewer.virtual_image_bytes == original
            np.testing.assert_array_equal(viewer.pattern(1, 2), current.counts[1, 2].numpy())
        viewer._handle_frontend_ready_msg(None, dict(ready, view="reopened"), None)
        assert len(messages) == len(states) == 2
        assert messages[-1]["view"] == "reopened"
        assert not viewer._stream_deferred_state
        assert viewer.frame_labels == ["5"]
        expected = current.counts.numpy()[..., viewer._current_detector_mask()].sum(-1)
        np.testing.assert_array_equal(viewer.virtual_image(), expected)
        recording.error = RuntimeError("Disconnected")
        assert viewer._stream_update() is False
        assert viewer.stream_error is recording.error
    finally:
        viewer.close()


def test_stream_topology_refreshes_visible_images_once(recording, monkeypatch):
    recording.acquisitions.append(LiveScan("first"))
    viewer = Show4DSTEM.stream(recording, view_mode="multiple", compare_group_mode="all")
    calls = []
    compare = viewer._compare_images

    def counted(indices, mask):
        calls.append(tuple(indices))
        return compare(indices, mask)

    monkeypatch.setattr(viewer, "_compare_images", counted)
    try:
        for index in range(2, 8):
            calls.clear()
            recording.acquisitions.append(LiveScan(str(index), value=index))
            viewer.refresh_stream()
            assert calls == [tuple(reversed(range(index)))]
        selected = recording.acquisitions[3]
        viewer.frame_idx = 3
        calls.clear()
        recording.acquisitions[:] = [*recording.acquisitions[1:], LiveScan("eighth", value=8)]
        viewer.refresh_stream()
        assert calls == [tuple(reversed(range(7)))]
        assert recording.acquisitions[viewer.frame_idx] is selected
        np.testing.assert_array_equal(viewer.pattern(1, 2), selected.counts[1, 2].numpy())
    finally:
        viewer.close()


def test_stream_sends_exact_changed_panels_but_full_topology_and_detector_updates(recording, monkeypatch):
    scans = [LiveScan(str(index), value=index + 1) for index in range(3)]
    scans[-1].received = 4
    recording.acquisitions.extend(scans)
    viewer = Show4DSTEM.stream(recording, view_mode="multiple", compare_group_mode="all")
    viewer._stream_display_credits = 1  # Exercise retirement with the window already full.
    messages, states = [], []
    monkeypatch.setattr(viewer, "send", lambda message, buffers=None: messages.append((message, buffers)))
    monkeypatch.setattr(viewer, "send_state", lambda key=None: states.append(set(key)) if key else None)

    def acknowledge():
        viewer._handle_frontend_ready_msg(None, {"type": "show4dstem_stream_ack", "version": 1,
                                                "view": "browser", "seq": viewer._stream_display_pending}, None)

    try:
        viewer._handle_frontend_ready_msg(None, {"type": "show4dstem_frontend_ready", "version": 1,
                                                "stream_ack": True, "stream_delta": True, "view": "browser"}, None)
        acknowledge()
        viewer._stream_update()
        acknowledge()
        snapshot = bytes(viewer.compare_virtual_image_bytes)
        messages.clear()
        states.clear()
        scans[-1].counts.fill_(9)
        scans[-1].received = 8
        assert viewer._stream_update()
        delta, buffers = messages[0]
        assert delta["type"] == "show4dstem_stream_panels" and delta["slots"] == [0]
        assert len(buffers) == 1 and len(buffers[0]) == 3 * 4 * 4
        reconstructed = bytearray(snapshot)
        reconstructed[:len(buffers[0])] = buffers[0]
        assert reconstructed == bytes(viewer.compare_virtual_image_bytes)
        assert all("compare_virtual_image_bytes" not in state for state in states)
        assert viewer._stream_panel_update == {"snapshot_pending": True}
        assert "_stream_panel_update" not in viewer._with_initial_live_mount_state(viewer.get_state)
        acknowledge()

        # Progress without a count change advances the revision without sending pixels.
        messages.clear()
        scans[-1].received = 10
        assert viewer._stream_update()
        assert messages[0][0]["slots"] == [] and messages[0][1] == []
        acknowledge()
        messages.clear()
        states.clear()
        scans[-1].counts.fill_(10)
        scans[-1].received = 11
        assert viewer._stream_update()
        assert messages[0][0]["type"] == "show4dstem_stream_panels"
        messages.clear()
        states.clear()
        recording.acquisitions[:] = [*scans[1:], LiveScan("new")]
        assert viewer._stream_update()  # Retire a scan while the old delta is still pending.
        assert scans[0].sessions[-1]._flat_t is None and not states
        acknowledge()
        assert viewer._stream_update()
        assert not any(message["type"] == "show4dstem_stream_panels" for message, _ in messages)
        assert any("compare_virtual_image_bytes" in state for state in states)
        acknowledge()
        messages.clear()
        states.clear()
        viewer.apply_preset("adf")
        assert any("compare_virtual_image_bytes" in state for state in states)
        assert not any(message["type"] == "show4dstem_stream_panels" for message, _ in messages)
        states.clear()
        snapshot, revision = bytes(viewer.compare_virtual_image_bytes), viewer.compare_image_revision
        viewer._handle_frontend_ready_msg(None, {"type": "show4dstem_stream_resync", "version": 1,
                                                "view": "browser"}, None)
        assert {"compare_virtual_image_bytes", "compare_image_revision"} <= states[-1]
        assert bytes(viewer.compare_virtual_image_bytes) == snapshot
        assert viewer.compare_image_revision > revision  # Identical resends still produce a model event.
        assert messages[-1][0]["revision"] == viewer.compare_image_revision
    finally:
        viewer.close()


@pytest.mark.parametrize("newest_first", [True, False])
def test_stream_order_preserves_selected_scan_across_arrival_and_eviction(recording, newest_first):
    scans = [LiveScan(str(i), value=i + 1) for i in range(4)]
    viewer = Show4DSTEM.stream(recording, newest_first=newest_first, view_mode="multiple")
    try:
        recording.acquisitions.extend(scans[:2])
        viewer.refresh_stream()
        viewer.frame_idx = 1
        selected = viewer.pattern(1, 2).copy()
        recording.acquisitions.append(scans[2])
        viewer.refresh_stream()
        assert viewer.compare_panel_order == ([2, 1, 0] if newest_first else [0, 1, 2])
        assert viewer.frame_labels[viewer.frame_idx] == "1"
        np.testing.assert_array_equal(viewer.pattern(1, 2), selected)
        recording.acquisitions[:] = scans[1:]
        viewer.refresh_stream()
        assert viewer.compare_panel_order == ([2, 1, 0] if newest_first else [0, 1, 2])
        assert viewer.frame_idx == 0 and viewer.frame_labels[0] == "1"
        np.testing.assert_array_equal(viewer.pattern(1, 2), selected)
        assert viewer.stream_error is None
    finally:
        viewer.close()


def test_queued_live_selection_and_cursor_follow_identity_after_eviction(recording):
    scans = [LiveScan(str(i), value=i + 1) for i in range(4)]
    recording.acquisitions[:] = scans[:3]
    viewer = Show4DSTEM.stream(recording, view_mode="multiple", compare_dp_mode="selected", device="cpu")
    try:
        viewer.refresh_stream()
        viewer.frame_idx = 1
        identities = list(viewer._stream_source_ids)
        context = viewer._interaction_context("frame_idx vi_roi_mode compare_dp_mode")
        recording.acquisitions[:] = scans[1:]
        viewer.refresh_stream()
        assert viewer._stream_source_ids[:2] == identities[1:]
        assert viewer._stream_selected_source == identities[1] and viewer.frame_idx == 0
        messages = []
        viewer.send = lambda message, buffers=None: messages.append(message)
        viewer._handle_position_request(viewer, {
            "type": "show4dstem_position", "version": 1, "view": "browser", "epoch": 0,
            "seq": 1, "center": [1, 2], "context": context}, [])
        assert [message["status"] for message in messages] == ["accepted", "complete"]
        assert (viewer.pos_row, viewer.pos_col) == (1, 2)
        np.testing.assert_array_equal(np.frombuffer(viewer.frame_bytes, np.float32), 2)
        # This click was queued when source 2 occupied slot 2. Slot 2 now holds source 3.
        viewer.set_state({"_stream_selected_source": identities[2], "frame_idx": 2})
        assert viewer.frame_idx == 1 and viewer.frame_labels[1] == "2"
        np.testing.assert_array_equal(np.frombuffer(viewer.frame_bytes, np.float32), 3)
        for stale in ({"_stream_selected_source": identities[0], "frame_idx": 0}, {"frame_idx": 2}):
            viewer.set_state(stale)
            assert viewer.frame_idx == 1 and viewer._stream_selected_source == identities[2]
            assert messages[-1]["type"] == "show4dstem_source_receipt"
            assert messages[-1]["status"] == "rejected"
        recording.acquisitions.clear()
        viewer.refresh_stream()
        assert viewer._stream_source_ids == [] and viewer._stream_selected_source == ""
        recording.acquisitions.append(LiveScan("next"))
        viewer.refresh_stream()
        assert viewer._stream_source_ids[0] not in identities
    finally:
        viewer.close()


@pytest.mark.parametrize("request_kind", ["show4dstem_detector", "show4dstem_detector_radius"])
def test_queued_detector_geometry_survives_selected_scan_reindex(recording, request_kind):
    scans = [LiveScan(str(i), value=i + 1) for i in range(3)]
    recording.acquisitions[:] = scans[:2]
    viewer = Show4DSTEM.stream(recording, view_mode="multiple", device="cpu")
    try:
        viewer.refresh_stream()
        viewer.frame_idx = 1
        center = request_kind == "show4dstem_detector"
        keys = ("frame_idx vi_source view_mode compare_page_idx roi_mode "
                + ("roi_radius roi_radius_inner roi_width roi_height" if center else "roi_center_row roi_center_col"))
        request = {"type": request_kind, "version": 1, "view": "browser", "epoch": 0,
                   "seq": 1, "context": viewer._interaction_context(keys)}
        request["center" if center else "radii"] = [3, 4] if center else [3.5, 0]
        recording.acquisitions[:] = scans[1:]
        viewer.refresh_stream()
        messages = []
        viewer.send = lambda message, buffers=None: messages.append(message)
        handler = viewer._handle_detector_request if center else viewer._handle_detector_radius_request
        handler(viewer, request, [])
        assert [message["status"] for message in messages] == ["accepted", "complete"]
        assert viewer.roi_center == [3, 4] if center else viewer.roi_radius == 3.5
    finally:
        viewer.close()


def test_deferred_retirement_keeps_full_snapshot_identity_and_pixels_atomic(recording, monkeypatch):
    scans = [LiveScan(str(i), value=i + 1) for i in range(4)]
    recording.acquisitions[:] = scans[:3]
    viewer = Show4DSTEM.stream(recording, view_mode="multiple", device="cpu")
    viewer._stream_display_credits = 1  # Exercise a full window with one pending update.
    try:
        viewer.refresh_stream()
        viewer._handle_frontend_ready_msg(None, {
            "type": "show4dstem_frontend_ready", "version": 1, "stream_ack": True, "view": "browser"}, None)
        published = viewer.get_state()
        states = []
        monkeypatch.setattr(viewer, "send_state", lambda key=None: states.append(viewer.get_state(key)) if key else None)
        recording.acquisitions[:] = scans[1:]
        viewer._stream_update()
        assert scans[0].sessions[-1]._flat_t is None
        assert not states  # No new identity/labels without their new pixels.
        during = viewer.get_state()
        for key in ("_stream_source_ids", "_stream_selected_source", "frame_labels", "compare_panel_indices"):
            assert during[key] == published[key]
        assert bytes(during["compare_virtual_image_bytes"]) == bytes(published["compare_virtual_image_bytes"])
        assert viewer._stream_source_ids != published["_stream_source_ids"]  # Current sources already retired.
        viewer._handle_frontend_ready_msg(None, {
            "type": "show4dstem_stream_ack", "version": 1, "view": "browser",
            "seq": viewer._stream_display_pending}, None)
        viewer._stream_update()
        assert viewer._stream_deferred_snapshot is None
        assert len(states) == 1
        assert {"_stream_source_ids", "frame_labels", "compare_virtual_image_bytes", "compare_image_revision"} <= states[0].keys()
        assert states[0]["_stream_source_ids"] == viewer._stream_source_ids
        assert bytes(states[0]["compare_virtual_image_bytes"]) == bytes(viewer.compare_virtual_image_bytes)
    finally:
        viewer.close()


@pytest.mark.parametrize("save_state, preview", [(False, "jpeg"), (False, None), (True, "jpeg")])
def test_deferred_snapshot_preserves_saved_state_policy(recording, save_state, preview):
    scans = [LiveScan(str(i), value=i + 1) for i in range(3)]
    recording.acquisitions[:] = scans[:2]
    viewer = Show4DSTEM.stream(recording, device="cpu", save_state=save_state, notebook_preview_format=preview)
    viewer._stream_display_credits = 1  # Exercise a full window with one pending update.
    try:
        viewer.refresh_stream()
        viewer.frame_idx = 1
        viewer._offline_stack = b"test-only-heavy-buffer"
        viewer.export_payload = b"test-only-export-buffer"
        viewer._handle_frontend_ready_msg(None, {
            "type": "show4dstem_frontend_ready", "version": 1, "stream_ack": True, "view": "browser"}, None)
        recording.acquisitions[:] = scans[1:]
        viewer._stream_update()
        assert viewer.frame_idx == 0 and viewer._stream_deferred_snapshot["frame_idx"] == 1
        saved = viewer.get_state(drop_defaults=True)
        assert saved["frame_idx"] == 1  # Filter defaults after restoring the published selection.
        assert all((name in saved) == save_state for name in viewer._UNSAVED_HEAVY_KEYS)
        if not save_state:
            assert bool(saved.get("_static_fallback_image")) == bool(preview)
        live = viewer._with_initial_live_mount_state(viewer.get_state)
        assert live["frame_idx"] == 1 and live["frame_labels"] == ["0", "1"]
        assert all(name in live for name in viewer._UNSAVED_HEAVY_KEYS)
        assert viewer.get_state("_offline_stack")["_offline_stack"] == b"test-only-heavy-buffer"
    finally:
        viewer.close()


def test_targeted_session_publication_does_not_wait_for_stream_lock():
    viewer = Show4DSTEM(torch.ones((2, 3, 4, 5)), device="cpu", notebook_preview_format=None)
    published = threading.Event()

    def publish():
        with viewer._session_lock:
            assert viewer.get_state("frame_bytes")["frame_bytes"]
            published.set()

    worker = threading.Thread(target=publish)
    try:
        with viewer._stream_lock:
            worker.start()
            assert published.wait(1), "Targeted publication acquired the stream lock after the session lock."
    finally:
        worker.join(timeout=2)
        viewer.close()


def test_structural_stream_updates_reuse_exact_panels_and_keep_metadata_atomic(recording, monkeypatch):
    scans = [LiveScan(str(index), value=index + 1) for index in range(5)]
    recording.acquisitions[:] = scans[:3]
    viewer = Show4DSTEM.stream(recording, view_mode="multiple", compare_group_mode="all", device="cpu")
    viewer._stream_display_credits = 1  # Exercise retirement with the window already full.
    states = []
    monkeypatch.setattr(viewer, "send_state", lambda key=None: states.append(viewer.get_state(key)) if key else None)
    monkeypatch.setattr(viewer, "send", lambda *args, **kwargs: None)

    def acknowledge():
        viewer._handle_frontend_ready_msg(None, {"type": "show4dstem_stream_ack", "version": 1,
                                                "view": "browser", "seq": viewer._stream_display_pending}, None)

    def reconstruct(previous, update):
        size = viewer.shape_rows * viewer.shape_cols * 4
        result = bytearray(len(update["reuse"]) * size)
        for slot, source in enumerate(update["reuse"]):
            if source >= 0:
                result[slot * size:(slot + 1) * size] = previous[source * size:(source + 1) * size]
        for slot, buffer in zip(update["slots"], update["buffers"]):
            result[slot * size:(slot + 1) * size] = buffer
        assert result == bytes(viewer.compare_virtual_image_bytes)

    try:
        viewer.refresh_stream()
        viewer._handle_frontend_ready_msg(None, {"type": "show4dstem_frontend_ready", "version": 1,
            "stream_ack": True, "stream_delta": True, "stream_layout": True, "view": "browser"}, None)
        acknowledge()
        previous = bytes(viewer.compare_virtual_image_bytes)
        states.clear()
        recording.acquisitions.append(scans[3])
        viewer._stream_update()
        assert len(states) == 1
        state = states[-1]
        update = state["_stream_panel_update"]
        assert update["snapshot_pending"] is True
        assert update["reuse"] == [-1, 0, 1, 2]
        assert update["slots"] == [0]
        assert {"frame_labels", "_stream_source_ids", "compare_panel_indices"} <= state.keys()
        assert "compare_virtual_image_bytes" not in state and "compare_image_revision" not in state
        reconstruct(previous, update)
        previous = bytes(viewer.compare_virtual_image_bytes)
        states.clear()
        recording.acquisitions[:] = scans[1:]
        viewer._stream_update()  # Retirement still proceeds while the append receipt is pending.
        assert not states and scans[0].sessions[-1]._flat_t is None
        acknowledge()
        viewer._stream_update()
        assert len(states) == 1
        update = states[-1]["_stream_panel_update"]
        assert update["reuse"] == [-1, 0, 1, 2]
        reconstruct(previous, update)
        assert "_stream_panel_update" not in viewer.get_state()
        assert "_stream_panel_update" not in viewer._with_initial_live_mount_state(viewer.get_state)
        acknowledge()
        viewer._resend_view_state()
        assert "compare_virtual_image_bytes" in states[-1]
        assert viewer._stream_panel_update == {}
    finally:
        viewer.close()


def test_final_partial_credit_window_recovers_a_hard_closed_owner(recording, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(widget_module.time, "monotonic", lambda: clock[0])
    recording.acquisitions.append(LiveScan("final"))
    viewer = Show4DSTEM.stream(recording, device="cpu")
    messages, states = [], []
    monkeypatch.setattr(viewer, "send", messages.append)
    monkeypatch.setattr(viewer, "send_state", lambda key=None: states.append(key) if key else None)

    def message(kind, view, **fields):
        viewer._handle_frontend_ready_msg(None, {"type": f"show4dstem_{kind}", "version": 1,
                                                "view": view, **fields}, None)

    try:
        for view in ("surviving", "owner"):
            message("frontend_ready", view, stream_ack=True)
        message("stream_ack", "owner", seq=viewer._stream_display_seq)
        recording.receiving = False
        viewer.refresh_stream()
        final = viewer._stream_display_seq
        assert viewer._stream_display_credits == 2
        assert final - viewer._stream_display_acknowledged == 1
        bulk_count = len(states)
        clock[0] += 2.1  # Owner disappeared without an unmount message.
        assert viewer._stream_update()
        assert messages[-1] == {"type": "show4dstem_stream_probe", "version": 1, "seq": final}
        control_count = len(messages)
        clock[0] += 0.1
        assert viewer._stream_update() and len(messages) == control_count
        assert len(states) == bulk_count  # No image retry, even for final delivery.
        message("stream_alive", "surviving", seq=final)
        message("stream_ack", "owner", seq=final)
        assert viewer._stream_display_pending == final
        message("stream_ack", "surviving", seq=final)
        assert viewer._stream_update() is False
    finally:
        viewer.close()


@pytest.mark.parametrize('options', [
    {'refresh_s': float('nan')}, {'refresh_s': float('inf')}, {'refresh_s': 0},
    {'first_timeout_s': -1}, {'first_timeout_s': float('nan')},
    {'first_timeout_s': float('inf')},
])
def test_invalid_stream_timing_fails_before_starting(recording, options):
    with pytest.raises(ValueError, match='must be finite'):
        Show4DSTEM.stream(recording, **options)


def test_closed_stream_does_not_retain_recording_counts():
    import gc
    import weakref

    scan = LiveScan('complete')
    scan_ref = weakref.ref(scan)
    recording = SimpleNamespace(lock=threading.RLock(), acquisitions=[scan], datasets=[],
                                receiving=False, error=None)
    viewer = Show4DSTEM.stream(recording)
    viewer._stream_thread.join(5)
    assert not viewer._stream_thread.is_alive()
    try:
        viewer.close()
        viewer.close()
        assert scan.counts.sum() > 0  # The viewer only releases its own references.
        del scan, recording
        gc.collect()
        assert scan_ref() is None
        with pytest.raises(ValueError, match='not streaming'):
            viewer.refresh_stream()
    finally:
        viewer.close()


def test_close_waits_for_running_query_and_can_be_retried(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    recording = SimpleNamespace(lock=threading.RLock(), acquisitions=[], datasets=[],
                                receiving=True, error=None)
    original = Show4DSTEM.refresh

    def blocked_refresh(viewer):
        entered.set()
        assert release.wait(5)
        original(viewer)

    monkeypatch.setattr(Show4DSTEM, 'refresh', blocked_refresh)
    viewer = Show4DSTEM.stream(recording, refresh_s=0.01)
    scan = LiveScan('arriving')
    recording.acquisitions.append(scan)
    try:
        assert entered.wait(5)
        thread = viewer._stream_thread
        join = thread.join
        monkeypatch.setattr(thread, 'join', lambda timeout=None: join(0.01))
        with pytest.raises(RuntimeError, match='still refreshing'):
            viewer.close()
        assert viewer._data is not None
        assert scan.sessions[0]._flat_t is not None
        assert viewer.comm is not None
        release.set()
        join(5)
        viewer.close()
        assert viewer._data is None and viewer.comm is None
        assert scan.sessions[0]._flat_t is None
    finally:
        release.set()
        viewer._stream_thread.join(5)
        viewer.close()


def test_background_transport_failure_is_reported(monkeypatch):
    recording = SimpleNamespace(lock=threading.RLock(), acquisitions=[], datasets=[],
                                receiving=True, error=None)
    viewer = Show4DSTEM.stream(recording, refresh_s=0.01)
    failure = RuntimeError('Display transport disconnected')

    def fail_delivery():
        raise failure

    monkeypatch.setattr(viewer, '_stream_display_barrier', fail_delivery)
    try:
        recording.acquisitions.append(LiveScan('arriving'))
        viewer._stream_thread.join(5)
        assert not viewer._stream_thread.is_alive()
        assert viewer.stream_error is failure
        assert viewer.title == 'Live acquisition stopped: Display transport disconnected'
    finally:
        viewer.close()


@pytest.mark.parametrize('kind', ['plain', 'chained', 'grouped'])
def test_failed_stream_releases_counts_but_preserves_error_details(monkeypatch, kind):
    import gc
    import traceback
    import weakref

    recording = SimpleNamespace(lock=threading.RLock(), acquisitions=[], datasets=[],
                                receiving=True, error=None)
    viewer = Show4DSTEM.stream(recording, refresh_s=0.01)

    def fail_refresh(widget):
        counts = widget._session(0)._flat_t
        try:
            raise ValueError(f'Cannot publish {counts.shape[0]} patterns')
        except ValueError as error:
            if kind == 'chained':
                raise RuntimeError('Display failed') from error
            if kind == 'grouped':
                raise ExceptionGroup('Display failed', [error])
            raise

    monkeypatch.setattr(Show4DSTEM, 'refresh', fail_refresh)
    scan = LiveScan('arriving')
    scan_ref, counts_ref = weakref.ref(scan), weakref.ref(scan.counts)
    recording.acquisitions.append(scan)
    try:
        viewer._stream_thread.join(5)
        assert not viewer._stream_thread.is_alive()
        error = viewer.stream_error
        stack = traceback.format_exception(error)
        assert 'Cannot publish 12 patterns' in ''.join(stack)
        viewer.close()
        del scan, recording
        gc.collect()
        assert scan_ref() is None and counts_ref() is None
        assert viewer.stream_error is error
        assert traceback.format_exception(error) == stack
    finally:
        viewer.close()


def test_stream_resizes_the_pattern_and_the_image_independently(recording):
    """While scans arrive, dragging one panel never resizes the other; dp_size and image_size set each."""
    viewer = Show4DSTEM.stream(recording)
    try:
        assert viewer.sync_resize is False
    finally:
        viewer.close()
    viewer = Show4DSTEM.stream(recording, dp_size="small", image_size="large")
    try:
        assert (viewer.sync_resize, viewer.dp_panel_width_px, viewer.vi_panel_width_px) == (False, 300, 600)
    finally:
        viewer.close()
    viewer = Show4DSTEM.stream(recording, sync_resize=True)
    try:
        assert viewer.sync_resize is True
    finally:
        viewer.close()
