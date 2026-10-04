"""Loopback WebSocket delivery of exact resident batches from the kernel to the browser.

Why: the Jupyter comm path serialises every complete virtual-image batch through
ZMQ, the server and the notebook websocket (about 50 ms for seven 512x512
uint32 images, several hundred ms for 66). The frontend already owns a worker
side consumer (``residentSocketWorker``) that adopts binary batches straight
from a WebSocket and paints them through the WebGPU batch canvas. This is the
kernel-side producer: one JSON header then one binary frame per batch, one
consumer, loopback origins only, a secret path token, acknowledgements kept for
timing. Scientific bytes are the same immutable exact uint32 counts the trait
path would carry.
"""
from __future__ import annotations

import json
import secrets
import time
from urllib.parse import urlsplit

_LOOPBACK = ("localhost", "127.0.0.1", "::1")


def loopback_origin(origin: str) -> bool:
    """Only a page served from this machine may consume the batch stream."""
    parsed = urlsplit(origin)
    return parsed.scheme in ("http", "https") and parsed.hostname in _LOOPBACK


class ResidentBatchStream:
    """Serve one browser consumer with exact resident batches over loopback."""

    def __init__(self, viewer) -> None:
        from tornado import httpserver, netutil, web, websocket

        self.viewer = viewer
        self.socket = None
        self.acks: list[dict] = []
        self.errors: list[str] = []
        self.published: list[dict] = []
        token = secrets.token_urlsafe(24)
        stream = self

        class BatchSocket(websocket.WebSocketHandler):
            def check_origin(self, origin):
                return loopback_origin(origin)

            def open(self):
                existing = stream.socket
                if existing is not None and existing.ws_connection is not None:
                    self.close(1008, "one consumer only")
                    return
                stream.socket = self
                self.set_nodelay(True)

            def on_message(self, message):
                if self is not stream.socket:
                    return
                try:
                    content = json.loads(message)
                except ValueError as error:
                    stream.errors.append(repr(error))
                    self.close(1002, "bad acknowledgement")
                    return
                content["host_epoch_ms"] = time.time() * 1000
                if content.get("type") == "ready":
                    # A (re)connected consumer paints the current batch immediately.
                    stream.viewer._resident_mask_key = None
                    stream.viewer._ensure_resident_images()
                else:
                    stream.acks.append(content)

            def on_close(self):
                if self is stream.socket:
                    stream.socket = None

        app = web.Application([(f"/batch/{token}", BatchSocket)], websocket_max_message_size=1 << 20, websocket_ping_interval=20)
        sockets = netutil.bind_sockets(0, address="127.0.0.1")
        self.port = sockets[0].getsockname()[1]
        self.server = httpserver.HTTPServer(app)
        self.server.add_sockets(sockets)
        self.url = f"ws://127.0.0.1:{self.port}/batch/{token}"

    @property
    def connected(self) -> bool:
        return self.socket is not None and self.socket.ws_connection is not None

    def publish(self, metadata: dict, body: bytes) -> None:
        """Send header then binary; a dropped socket is recorded, never raised into the kernel loop."""
        connection = self.socket
        before = time.perf_counter()
        metadata = dict(metadata, type="resident_batch", transport="websocket", bytes=len(body), published_epoch_ms=time.time() * 1000)
        connection.write_message(json.dumps(metadata))
        future = connection.write_message(body, binary=True)
        metadata["publish_ms"] = (time.perf_counter() - before) * 1000
        self.published.append(metadata)

        def done(fut):
            try:
                fut.result()
            except Exception as error:  # noqa: BLE001
                self.errors.append(repr(error))
        future.add_done_callback(done)

    def stop(self) -> None:
        if self.socket is not None:
            self.socket.close()
        self.server.stop()
