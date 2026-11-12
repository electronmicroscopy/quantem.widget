"""Custom detector requests preserve canonical state and exact detector products."""

import pytest
import torch

from quantem.widget import Show4DSTEM

CONTEXT = "frame_idx vi_source view_mode compare_page_idx roi_mode roi_radius roi_radius_inner roi_width roi_height"
RADIUS_CONTEXT = "frame_idx vi_source view_mode compare_page_idx roi_mode roi_center_row roi_center_col"


def request(viewer, center, seq=1):
    return {"type": "show4dstem_detector", "version": 1, "view": "first", "epoch": 0,
            "seq": seq, "center": center, "context": [getattr(viewer, key) for key in CONTEXT.split()]}


def radius_request(viewer, radii, seq=1):
    return {"type": "show4dstem_detector_radius", "version": 1, "view": "first", "epoch": 0,
            "seq": seq, "radii": radii, "context": [getattr(viewer, key) for key in RADIUS_CONTEXT.split()]}


@pytest.fixture
def viewer():
    values_t = torch.arange(2 * 5 * 7 * 16 * 16, dtype=torch.float32).reshape(2, 5, 7, 16, 16)
    widget = Show4DSTEM(values_t, device="cpu", view_mode="multiple", show_fft=False,
                       bright_field_center_px=(7.5, 7.5), bright_field_radius_px=3)
    messages = []
    widget.send = lambda content, buffers=None: messages.append(content)
    try:
        yield widget, values_t, messages
    finally:
        widget.free()
        widget.close()


def test_center_request_acknowledges_before_one_exact_computation(viewer, monkeypatch):
    widget, values_t, messages = viewer
    compute = widget._compare_images
    queries = []

    def observed(indices, mask):
        assert messages[-1]["status"] == "accepted"
        queries.append(tuple(indices))
        return compute(indices, mask)

    monkeypatch.setattr(widget, "_compare_images", observed)
    widget._handle_detector_request(widget, request(widget, [5, 9]), [])
    assert [message["status"] for message in messages] == ["accepted", "complete"]
    assert len(queries) == 1 and widget.roi_center == [5, 9]
    assert (widget.roi_center_row, widget.roi_center_col) == (5, 9)
    rows_t, cols_t = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    mask_t = (rows_t - 5) ** 2 + (cols_t - 9) ** 2 <= widget.roi_radius ** 2
    expected_t = values_t[..., mask_t].sum(-1) / mask_t.sum()
    actual_t = torch.frombuffer(bytearray(widget.compare_virtual_image_bytes), dtype=torch.float32).reshape(2, 5, 7)
    torch.testing.assert_close(actual_t, expected_t, atol=0, rtol=0)


def test_center_request_recovers_same_position_after_compute_failure(viewer, monkeypatch):
    widget, _, messages = viewer
    compute = widget._compare_images

    def fail(indices, mask):
        raise MemoryError("injected bounded compute failure")

    monkeypatch.setattr(widget, "_compare_images", fail)
    content = request(widget, [5, 9])
    widget._handle_detector_request(widget, content, [])
    assert [message["status"] for message in messages] == ["accepted", "rejected"]
    assert not widget.compare_virtual_image_bytes
    monkeypatch.setattr(widget, "_compare_images", compute)
    widget._handle_detector_request(widget, dict(content, seq=2), [])
    assert messages[-1]["status"] == "complete" and widget.compare_virtual_image_bytes


def test_stale_view_settings_and_freed_source_reject_before_computing(viewer):
    widget, _, messages = viewer
    content = request(widget, [5, 9])
    widget.roi_radius = 4
    before = bytes(widget.compare_virtual_image_bytes)
    center = list(widget.roi_center)
    widget._handle_detector_request(widget, content, [])
    assert messages[-1]["status"] == "rejected"
    assert bytes(widget.compare_virtual_image_bytes) == before and widget.roi_center == center
    widget.free()
    widget._handle_detector_request(widget, request(widget, [5, 9], 2), [])
    assert [message["status"] for message in messages] == ["rejected", "rejected"]


def test_position_request_reads_once_and_recovers_after_failure(viewer, monkeypatch):
    widget, values_t, messages = viewer
    widget.compare_dp_mode = "selected"
    content = {"type": "show4dstem_position", "version": 1, "view": "first",
               "epoch": 0, "seq": 1, "center": [1, 3],
               "context": [widget.frame_idx, widget.vi_roi_mode, widget.compare_dp_mode]}
    compute = widget._update_frame
    calls = []

    def observed(change=None):
        assert messages[-1]["status"] == "accepted"
        calls.append((widget.pos_row, widget.pos_col))
        if len(calls) == 1:
            raise MemoryError("injected frame failure")
        compute(change)

    monkeypatch.setattr(widget, "_update_frame", observed)
    widget._handle_position_request(widget, content, [])
    widget._handle_position_request(widget, dict(content, seq=2), [])
    assert [message["status"] for message in messages] == ["accepted", "rejected", "accepted", "complete"]
    assert calls == [(1, 3), (1, 3)]
    actual = torch.frombuffer(bytearray(widget.frame_bytes), dtype=torch.float32).reshape(16, 16)
    torch.testing.assert_close(actual, values_t[0, 1, 3], atol=0, rtol=0)
    widget._handle_position_request(widget, dict(content, context=[1, "off", "selected"]), [])
    assert messages[-1]["status"] == "rejected" and len(calls) == 2


@pytest.mark.parametrize("mode, inner", [("circle", 0.0), ("annular", 2.25)])
def test_radius_request_acknowledges_before_one_exact_computation(viewer, monkeypatch, mode, inner):
    widget, values_t, messages = viewer
    widget.roi_mode = mode
    compute = widget._compare_images
    queries = []

    def observed(indices, mask):
        assert messages[-1]["status"] == "accepted"
        queries.append((widget.roi_radius, widget.roi_radius_inner))
        return compute(indices, mask)

    monkeypatch.setattr(widget, "_compare_images", observed)
    started = widget._last_interaction_at
    widget._msg_callbacks(widget, radius_request(widget, [5.75, inner]), [])
    assert [message["status"] for message in messages] == ["accepted", "complete"]
    assert all(message["type"] == "show4dstem_detector_radius_receipt" for message in messages)
    assert queries == [(5.75, inner)] and widget.roi_active
    assert widget._last_interaction_at >= started
    rows_t, cols_t = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    distance_t = torch.sqrt((rows_t - 7.5) ** 2 + (cols_t - 7.5) ** 2)
    mask_t = (distance_t >= inner) & (distance_t <= 5.75)
    expected_t = values_t[..., mask_t].sum(-1) / mask_t.sum()
    actual_t = torch.frombuffer(bytearray(widget.compare_virtual_image_bytes), dtype=torch.float32).reshape(2, 5, 7)
    torch.testing.assert_close(actual_t, expected_t, atol=0, rtol=0)


def test_radius_request_recovers_after_compute_failure(viewer, monkeypatch):
    widget, _, messages = viewer
    compute = widget._compare_images

    def fail(indices, mask):
        raise MemoryError("injected bounded compute failure")

    monkeypatch.setattr(widget, "_compare_images", fail)
    content = radius_request(widget, [5.75, 0])
    widget._handle_detector_radius_request(widget, content, [])
    assert [message["status"] for message in messages] == ["accepted", "rejected"]
    assert not widget.compare_virtual_image_bytes
    monkeypatch.setattr(widget, "_compare_images", compute)
    widget._handle_detector_radius_request(widget, dict(content, seq=2), [])
    assert messages[-1]["status"] == "complete" and widget.compare_virtual_image_bytes


def test_stale_or_invalid_radius_requests_leave_the_display_unchanged(viewer, monkeypatch):
    widget, _, messages = viewer
    stale = radius_request(widget, [5.75, 0])
    widget.roi_center = [5, 9]
    before = bytes(widget.compare_virtual_image_bytes)
    radii = widget.roi_radius, widget.roi_radius_inner
    queries = []
    monkeypatch.setattr(widget, "_compare_images", lambda *args: queries.append(args))
    invalid = ([0, 0], [float("inf"), 0], [5, float("nan")], [5, -1], [5, 5])
    requests = [stale, *[radius_request(widget, pair) for pair in invalid]]
    for content in requests:
        widget._handle_detector_radius_request(widget, content, [])
        assert messages[-1]["status"] == "rejected"
        assert bytes(widget.compare_virtual_image_bytes) == before
        assert (widget.roi_radius, widget.roi_radius_inner) == radii
    assert not queries
