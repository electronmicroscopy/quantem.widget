"""Scientific image buffers retain exact values through widget transport."""

import base64
import struct

import comm
from ipywidgets import Widget

from quantem.widget.show4dstem.widget import _ImageMemoryView


class ImageWidget(Widget):
    image = _ImageMemoryView().tag(sync=True)

    def __init__(self):
        self.messages = []
        super().__init__(comm=comm.DummyComm())

    def _send(self, message, buffers=None):
        self.messages.append((message, tuple(buffers or ())))


def test_owned_image_buffers_remain_exact_through_sync_and_embedding():
    # Include signed zero, non-finite values and a noncanonical NaN bit pattern.
    raw = bytes(range(256)) + struct.pack(
        '<8I', 0, 0x80000000, 0x3f800000, 0xbf800000,
        0x7f800000, 0xff800000, 0x7fc01234, 0xffffffff,
    )
    widget = ImageWidget()
    try:
        widget.image = memoryview(raw)
        first = widget.messages[-1][1][0]
        widget.messages.clear()
        with widget.hold_sync(), widget.hold_trait_notifications():
            widget.image = memoryview(raw[::-1])
            widget.image = memoryview(raw + b'x')
        assert len(widget.messages) == 1
        assert bytes(widget.messages[0][1][0]) == raw + b'x'
        assert bytes(first) == raw
        embedded = widget._get_embed_state(drop_defaults=False)['buffers']
        assert base64.b64decode(embedded[0]['data']) == raw + b'x'

        widget.messages.clear()
        widget.image = widget.image
        assert not widget.messages
        widget.set_state({'image': memoryview(raw)})
        assert all(message['method'] != 'update' for message, _ in widget.messages)
        assert bytes(widget.image) == raw
    finally:
        widget.close()
