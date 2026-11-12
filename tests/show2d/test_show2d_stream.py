"""Show2D.stream: changed-panel transfers, coalescing behind acknowledgments and background-thread updates."""

import threading

import numpy as np
import pytest

from quantem.widget import Show2D


def _capture(widget):
    """Record every comm message the widget sends as (method, content-or-state, buffer sizes)."""
    sent = []
    original = widget._send

    def record(msg, buffers=None):
        body = msg.get("content") if msg.get("method") == "custom" else msg.get("state")
        sent.append((msg.get("method"), body, [memoryview(buffer).nbytes for buffer in buffers or []]))
        return original(msg, buffers)

    widget._send = record
    return sent


def _patches(sent):
    return [(body, sizes) for method, body, sizes in sent if method == "custom" and body["type"] == "show2d_stream_panels"]


def _full_updates(sent):
    return [sizes for method, _, sizes in sent if method == "update" and sizes]


def _ready(widget):
    widget._handle_stream_msg(widget, {"type": "show2d_frontend_ready", "version": 1,
                                       "revision": int(widget._stream_revision), "bytes": len(widget.frame_bytes)}, [])


def _images(*values, shape=(8, 6)):
    return [np.full(shape, value, dtype=np.float32) for value in values]


def test_only_changed_panels_are_sent():
    view = Show2D.stream(_images(0, 1, 2), labels=["a", "b", "c"])
    sent = _capture(view)
    view.update(_images(0, 5, 2), labels=["a", "b", "c"])
    assert view._stream_wait_sent(5)
    (body, sizes), = _patches(sent)
    assert body["slots"] == [1] and sizes == [8 * 6 * 4]
    assert not _full_updates(sent)  # the frame_bytes trait is current in Python but not resent
    np.testing.assert_array_equal(np.frombuffer(view.frame_bytes, np.float32, 3 * 48).reshape(3, 8, 6)[1], 5)
    # unchanged pixels with new labels: no image bytes at all
    sent.clear()
    view.update(_images(0, 5, 2), labels=["a", "b", "new"])
    assert view._stream_wait_sent(5)
    assert not _patches(sent) and not _full_updates(sent) and view.labels == ["a", "b", "new"]
    view.close()


def test_a_new_layout_sends_the_full_images_once_with_a_new_revision():
    view = Show2D.stream()
    assert view.folder_waiting
    sent = _capture(view)
    view.update(_images(1, 2))
    assert view._stream_wait_sent(5)
    revision = view._stream_revision
    view.update(_images(1, 2, 3))
    assert view._stream_wait_sent(5)
    assert len(_full_updates(sent)) == 2 and view._stream_revision == revision + 1
    assert (view.n_images, view.folder_waiting) == (3, False)
    view.close()


def test_updates_coalesce_while_two_transfers_await_acknowledgment():
    view = Show2D.stream(_images(0, 0))
    sent = _capture(view)
    _ready(view)
    view._stream_credits.stalled_ack_s = 60.0
    for value in (1, 2):
        view.update(_images(value, 0))
        assert view._stream_wait_sent(5)
    for value in range(3, 9):
        view.update(_images(value, 0))
    assert not view._stream_wait_sent(0.3)
    assert len(_patches(sent)) == 2
    view._handle_stream_msg(view, {"type": "show2d_stream_ack", "version": 1, "seq": 2}, [])
    assert view._stream_wait_sent(5)
    # the latest call wins: one more transfer, carrying the last update only
    assert len(_patches(sent)) == 3
    np.testing.assert_array_equal(view._data[0], 8)
    view.close()


def test_updates_from_a_background_thread_while_the_user_interacts():
    view = Show2D.stream(_images(0, 0, 0))
    sent = _capture(view)

    def process():
        for value in range(1, 31):
            view.update(_images(value, value, 0), labels=["x", "y", "z"])

    worker = threading.Thread(target=process)
    worker.start()
    for _ in range(50):
        view.set_state({"cmap": "gray"})
    worker.join()
    assert view._stream_wait_sent(5)
    np.testing.assert_array_equal(view._data[0], 30)
    assert all(body["slots"] == [0, 1] for body, _ in _patches(sent))
    assert 1 <= len(_patches(sent)) <= 30
    view.close()


def test_update_validates_before_returning_and_needs_a_stream():
    view = Show2D.stream()
    with pytest.raises(ValueError, match="NaN"):
        view.update(np.full((4, 4), np.nan))
    with pytest.raises(ValueError, match="labels length"):
        view.update(_images(1, 2), labels=["only one"])
    view.close()
    plain = Show2D(np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(RuntimeError, match="Show2D.stream"):
        plain.update(np.zeros((4, 4)))
    plain.close()
