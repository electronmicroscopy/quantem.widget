"""CPU-only: the resident viewer publishes exact batches over its loopback WebSocket."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
from tornado.ioloop import IOLoop
from tornado.websocket import websocket_connect

from quantem.gpu.io._resident import CudaResidentSource
from quantem.widget import Show4DSTEM
from quantem.widget.show4dstem_resident_stream import loopback_origin


def fixture_source():
    counts = np.arange(3*2*3*18*18, dtype=np.uint16).reshape(3, 2, 3, 18, 18)
    counts[..., -1, -1] = 65535
    valid = np.ones((18, 18), bool); valid[-1, -1] = False
    return CudaResidentSource(SimpleNamespace(), shape=counts.shape, device="cuda:0", generation="stream-fixture",
        storage_format="fixture-v1", source_codec="numpy-uint16-v1", sparse_codec="none-v1", resident_bytes=1,
        valid_pixels=valid, execute=lambda task: task(),
        integrate=lambda o, mask: counts[..., mask.astype(bool)].sum(axis=-1, dtype=np.uint32),
        patterns=lambda o, index: np.ascontiguousarray(counts.reshape(3, 6, 18, 18)[:, index]))


def test_loopback_origin_policy():
    assert loopback_origin("http://127.0.0.1:8888") and loopback_origin("http://localhost:1") and loopback_origin("https://[::1]:9")
    assert not loopback_origin("http://example.com") and not loopback_origin("http://100.86.162.126:8900") and not loopback_origin("ws://127.0.0.1:1")


def test_displayed_pattern_masks_invalid_pixels():
    widget = Show4DSTEM(fixture_source(), center=(8.5, 8.5), bf_radius=30, verbose=False, stream=False)
    frame = np.frombuffer(widget.frame_bytes, dtype=np.float32).reshape(18, 18)
    assert frame[-1, -1] == 0 and frame.max() < 65535
    assert widget._resident_source.frame_batch(0)[0, -1, -1] == 65535  # raw counts untouched
    assert widget._resident_stream is None and widget.resident_stream == {}


def test_stream_trait_and_exact_delivery():
    loop = IOLoop.current()
    widget = Show4DSTEM(fixture_source(), center=(8.5, 8.5), bf_radius=30, verbose=False)
    stream = widget.resident_stream
    assert stream["enabled"] and stream["url"].startswith("ws://127.0.0.1:") and stream["generation"] == "stream-fixture"
    assert widget.resident_batch_info["dtype"] == "<u4"  # trait path served the first batch before any consumer

    async def consume():
        conn = await websocket_connect(stream["url"])
        await conn.write_message(json.dumps({"type": "ready"}))
        header = json.loads(await conn.read_message())
        body = await conn.read_message()
        await conn.write_message(json.dumps({"type": "received", "request_id": header["request_id"]}))
        conn.close()
        return header, body

    header, body = loop.run_sync(consume, timeout=15)
    assert header["type"] == "resident_batch" and header["transport"] == "websocket" and header["dtype"] == "<u4"
    assert header["shape"] == [3, 2, 3] and header["bytes"] == len(body) == 3 * 2 * 3 * 4
    assert np.array_equal(np.frombuffer(body, dtype="<u4").reshape(3, 2, 3), widget._resident_images)
    assert widget._resident_stream.published[-1]["request_id"] == header["request_id"]
    widget._resident_stream.stop()
