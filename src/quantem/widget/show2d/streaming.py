"""Show2D.stream: replace the displayed images repeatedly while the notebook stays interactive.

``update`` copies the images and returns. A worker thread publishes the
latest update when one of ``Show4DSTEM.stream``'s two acknowledgment credits
is free (``quantem.widget.live_display``); updates that arrive meanwhile
replace each other, so the browser always receives the newest images and
never a backlog. When the panel layout is unchanged only the panels whose
bytes changed are sent, as one custom message; the ``frame_bytes`` trait
keeps the full current images for saving and for views that mount later.
"""

import threading
from typing import Self

import numpy as np

from quantem.widget.live_display import DisplayCredits, LiveDisplayWorker
from quantem.widget.utils.array import to_numpy


class Show2DStream:
    """``Show2D.stream`` and ``update``."""

    _stream_worker: LiveDisplayWorker | None = None

    @classmethod
    def stream(cls, data=None, labels=None, **viewer_kwargs) -> Self:
        """Open a viewer whose images are replaced by repeated :meth:`update` calls.

        The viewer returns immediately; without ``data`` it waits for the
        first update. ``update`` may be called from any thread and never
        waits for the browser: at most two transfers await browser
        acknowledgment, later updates coalesce so the newest call wins, and
        panels whose pixels did not change are not resent. Pixels are sent
        at the resolution given (``display_bin=1`` unless you pass another).

        Examples
        --------
        >>> view = Show2D.stream(title="Live")  # doctest: +SKIP
        >>> view  # doctest: +SKIP
        >>> view.update([phase, dark_field], labels=["phase", "DF"])  # doctest: +SKIP
        """
        viewer_kwargs.setdefault("verbose", False)
        if data is None:
            if labels is not None:
                raise ValueError("labels= needs data=; pass labels with update(images, labels=...).")
            viewer = cls(np.zeros((1, 1), dtype=np.float32), labels=[""], **viewer_kwargs)
            viewer._set_waiting_empty_state()
            viewer.folder_status = "Waiting for the first image"
        else:
            viewer = cls(data, labels=labels, **viewer_kwargs)
        viewer._start_stream()
        return viewer

    def _start_stream(self) -> None:
        self._stream_lock = threading.RLock()        # held while the images and their traits change
        self._stream_queue_lock = threading.Lock()   # held only to hand an update to the worker
        self._stream_pending: tuple[np.ndarray, list | None] | None = None
        self._stream_in_flush = False
        self._stream_credits = DisplayCredits()
        self.observe(self._stream_on_full_frame_bytes, names=["frame_bytes"])
        self.on_msg(self._handle_stream_msg)
        self._stream_worker = LiveDisplayWorker("Show2D-stream", self._stream_flush)
        self._stream_worker.start()

    def update(self, images, labels=None) -> None:
        """Show new images: one 2D image, a list of 2D images, or an ``(N, H, W)`` stack.

        Returns after validating and copying the images; a later call made
        before this one reaches the browser replaces it. Raises if an
        earlier publication failed.
        """
        worker = self._stream_worker
        if worker is None or not worker.running:
            raise RuntimeError("update() needs a running viewer from Show2D.stream(...).")
        if worker.error is not None:
            raise RuntimeError(f"Show2D stream stopped: {worker.error}") from worker.error
        if isinstance(images, (list, tuple)):
            items = [to_numpy(item) for item in images]
            if not items or any(item.ndim != 2 for item in items):
                raise ValueError("update() takes 2D images; an RGB or stacked panel needs set_image().")
            if len({item.shape for item in items}) != 1:
                raise ValueError(f"update() images must share one shape, got {sorted({item.shape for item in items})}.")
            # The one copy: np.stack writes every image into a new float32 block.
            data = np.stack(items).astype(np.float32, copy=False)
        else:
            data = to_numpy(images)
            if data.ndim == 2:
                data = data[np.newaxis]
            if data.ndim != 3:
                raise ValueError(f"update() takes a 2D image or an (N, H, W) stack, got {data.ndim}D.")
            data = np.array(data, dtype=np.float32, copy=True)
        if 0 in data.shape:
            raise ValueError(f"Empty image stack: shape {data.shape}.")
        if np.iscomplexobj(data):
            raise TypeError("Show2D does not accept complex data. Convert first: np.abs(arr) or np.angle(arr).")
        if not np.isfinite(data).all():
            raise ValueError("Data contains NaN or inf. Clean first: np.nan_to_num(arr, nan=0, posinf=0, neginf=0).")
        if labels is not None:
            labels = ["" if label is None else str(label) for label in labels]
            if len(labels) != data.shape[0]:
                raise ValueError(f"labels length ({len(labels)}) must match n_images ({data.shape[0]}).")
        with self._stream_queue_lock:
            self._stream_pending = (data, labels)
        worker.wake()

    def _stream_has_pending(self) -> bool:
        with self._stream_queue_lock:
            return self._stream_pending is not None

    def _stream_wait_sent(self, timeout: float = 5.0) -> bool:
        """Wait until the latest update has left the worker (tests and scripts)."""
        return self._stream_worker.wait_idle(self._stream_has_pending, timeout)

    def _stream_layout(self) -> tuple:
        return (int(self.n_images), int(self.height), int(self.width), bool(self.offline),
                tuple(self.is_rgb), int(self.display_bin), tuple(self.image_rotations))

    def _stream_flush(self) -> None:
        """Publish the newest pending update when a credit is free."""
        with self._stream_lock:
            if not self._stream_credits.available():
                return
            with self._stream_queue_lock:
                pending, self._stream_pending = self._stream_pending, None
            if pending is not None:
                self._stream_publish(*pending)

    def _stream_publish(self, data: np.ndarray, labels: list | None) -> None:
        old_bytes, old_layout = self.frame_bytes, self._stream_layout()
        self._stream_in_flush = True
        try:
            with self.hold_sync():
                if self.folder_waiting:
                    self.folder_waiting = False
                    self.folder_status = ""
                self.set_image(data, labels)
                new_bytes = self.frame_bytes
                count = int(self.n_images)
                block = int(self.height) * int(self.width) * (1 if self.offline else 4)
                if (count and old_layout == self._stream_layout() and not any(self.is_rgb)
                        and not self.panel_stack_bytes and len(old_bytes) == len(new_bytes) >= count * block):
                    # Same layout: the trait keeps the full images for saving and
                    # new views, and only the changed panels travel.
                    self._states_to_send.discard("frame_bytes")
                    old_view, new_view = memoryview(old_bytes), memoryview(new_bytes)
                    slots = [slot for slot in range(count)
                             if old_view[slot * block:(slot + 1) * block] != new_view[slot * block:(slot + 1) * block]]
                    if slots:
                        self.send(
                            {"type": "show2d_stream_panels", "version": 1, "seq": self._stream_credits.take(),
                             "revision": int(self._stream_revision), "slots": slots, "block": block,
                             "bytes": len(new_bytes)},
                            buffers=[new_view[slot * block:(slot + 1) * block] for slot in slots],
                        )
                elif new_bytes is not old_bytes:
                    # A new layout replaces the whole image buffer in one acknowledged state update.
                    self._stream_seq = self._stream_credits.take()
                    self._stream_revision += 1
        finally:
            self._stream_in_flush = False

    def _stream_on_full_frame_bytes(self, _change) -> None:
        """Any other full image update starts a new base for the browser's panel patches."""
        if not self._stream_in_flush:
            self._stream_revision += 1

    def _stream_resend(self) -> None:
        """Send the full images once, for a view that mounted late or missed a transfer."""
        with self._stream_lock:
            self._stream_revision += 1
            self.send_state(["frame_bytes", "_stream_revision", "_stream_seq"])
            self._stream_credits.ready()

    def _handle_stream_msg(self, _widget, content, _buffers) -> None:
        if not isinstance(content, dict) or content.get("version") != 1:
            return
        kind = content.get("type")
        if kind == "show2d_frontend_ready":
            with self._stream_lock:
                self._stream_credits.ready()
                if content.get("revision") != int(self._stream_revision) or content.get("bytes") != len(self.frame_bytes):
                    self._stream_resend()
        elif kind == "show2d_stream_ack":
            if self._stream_credits.acknowledge(content.get("seq")) and self._stream_worker is not None:
                self._stream_worker.wake()
        elif kind == "show2d_stream_resync":
            self._stream_resend()

    def set_state(self, sync_data) -> None:
        """Browser edits wait for an in-progress update."""
        lock = getattr(self, "_stream_lock", None)
        if lock is None:
            return super().set_state(sync_data)
        with lock:
            super().set_state(sync_data)

    def close(self) -> None:
        """Stop the stream worker, then close the widget comm."""
        worker = self._stream_worker
        if worker is not None:
            worker.stop()
        super().close()

