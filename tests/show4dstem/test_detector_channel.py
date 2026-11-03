"""Custom center requests preserve canonical state and exact detector products."""

import pytest
import torch

from quantem.widget import Show4DSTEM

CONTEXT = "frame_idx vi_source view_mode compare_page_idx roi_mode roi_radius roi_radius_inner roi_width roi_height"


def request(viewer, center, seq=1):
    return {"type": "show4dstem_detector", "version": 1, "view": "first", "epoch": 0,
            "seq": seq, "center": center, "context": [getattr(viewer, key) for key in CONTEXT.split()]}


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
