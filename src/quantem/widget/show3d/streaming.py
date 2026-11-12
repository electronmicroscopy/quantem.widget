"""Show3D.stream: grow the stack one frame at a time while the notebook stays interactive.

Each ``append`` copies one frame and returns. A worker thread moves pending
frames into the stack and sends only their bytes to the browser, as one
custom message per transfer, with the bounded acknowledgment window of
``Show4DSTEM.stream`` (``quantem.widget.live_display``). The synced
``_offline_float_stack`` trait is left as it was; ``get_state`` substitutes
the complete stack, and a view that mounts later or loses a transfer asks
for one full resend.
"""

import threading
import time
from typing import Self

import numpy as np

from quantem.widget.live_display import DisplayCredits, LiveDisplayWorker
from quantem.widget.show3d.playback import stack_percentiles
from quantem.widget.utils.array import to_numpy

# Seconds after the user last moved the frame slider before the view follows the newest frame again.
STREAM_FOLLOW_PAUSE_S = 1.5


class Show3DStream:
    """``Show3D.stream`` and ``append``."""

    # Follow the newest frame while the user is on the last one. Set False to keep the current frame.
    stream_follow = True
    _stream_worker: LiveDisplayWorker | None = None

    @classmethod
    def stream(cls, data=None, *, labels: list[str] | None = None, **viewer_kwargs) -> Self:
        """Open a viewer that grows by one frame per :meth:`append`.

        The viewer returns immediately; without ``data`` it waits for the
        first frame. Every ``append`` transfers only the new frame's bytes,
        and the view follows the newest frame unless the user moved to an
        earlier one. At most two transfers await browser acknowledgment;
        frames that arrive meanwhile are sent together with the next
        transfer, never dropped. ``append`` may be called from any thread.

        Frames are shown at native resolution (``display_bin=1``). Auto
        contrast is computed per frame as it arrives, so a new frame never
        rescans the stack.

        Examples
        --------
        >>> view = Show3D.stream(title="Drift series")  # doctest: +SKIP
        >>> view  # doctest: +SKIP
        >>> view.append(image, label="scan 1")  # doctest: +SKIP
        """
        display_bin = viewer_kwargs.get("display_bin", 1)
        if display_bin != 1:
            raise ValueError("Show3D.stream shows native pixels; remove display_bin= or pass display_bin=1.")
        viewer_kwargs.setdefault("verbose", False)
        # a recording is watched with the frame thumbnail strip; mode= or thumbnails= say otherwise
        viewer_kwargs.setdefault("mode", "stream")
        if data is None:
            if labels:
                raise ValueError("labels= needs data=; label appended frames with append(image, label=...).")
            viewer = cls(np.zeros((1, 1, 1), dtype=np.float32), labels=[""], **viewer_kwargs)
            viewer._set_waiting_empty_state()
            viewer.folder_status = "Waiting for the first frame"
        else:
            viewer = cls(data, labels=labels, **viewer_kwargs)
            if int(viewer.n_panels) != 1 or viewer.is_rgb:
                viewer.close()
                raise ValueError("Show3D.stream shows one grayscale stack; pass a single (N, H, W) array.")
            viewer.slice_idx = int(viewer.n_slices) - 1
        viewer._start_stream()
        return viewer

    def _start_stream(self) -> None:
        self._stream_lock = threading.RLock()        # held while the stack and its traits change
        self._stream_queue_lock = threading.Lock()   # held only to hand frames to the worker
        self._stream_pending: list[tuple[np.ndarray, str | None]] = []
        self._stream_shape = tuple(self._data.shape[1:]) if int(self.n_slices) else None
        self._stream_buffer: np.ndarray | None = None
        self._stream_credits = DisplayCredits()
        self._stream_interaction_at = 0.0
        self.on_msg(self._handle_stream_msg)
        self._stream_worker = LiveDisplayWorker("Show3D-stream", self._stream_flush)
        self._stream_worker.start()

    def append(self, image, label: str | None = None) -> None:
        """Add one 2D frame to the end of a streamed stack.

        Returns after validating and copying the frame; the transfer happens
        on the viewer's worker thread. Raises if an earlier transfer failed.
        """
        worker = self._stream_worker
        if worker is None or not worker.running:
            raise RuntimeError("append() needs a running viewer from Show3D.stream(...).")
        if worker.error is not None:
            raise RuntimeError(f"Show3D stream stopped: {worker.error}") from worker.error
        frame = to_numpy(image)
        if frame.ndim != 2:
            raise ValueError(f"append() takes one 2D frame, got shape {frame.shape}.")
        if 0 in frame.shape:
            raise ValueError(f"Empty frame: shape {frame.shape}.")
        if np.iscomplexobj(frame):
            raise TypeError("Show3D does not accept complex data. Convert first: np.abs(arr) or np.angle(arr).")
        # The one copy: the caller may reuse its buffer as soon as this returns.
        frame = np.array(frame, dtype=np.float32, copy=True)
        if not np.isfinite(frame).all():
            raise ValueError("Frame contains NaN or inf. Clean first: np.nan_to_num(arr, nan=0, posinf=0, neginf=0).")
        with self._stream_queue_lock:
            if self._stream_shape is None:
                self._stream_shape = frame.shape
            elif frame.shape != self._stream_shape:
                raise ValueError(f"Frame shape {frame.shape} does not match the stream's {self._stream_shape}.")
            self._stream_pending.append((frame, None if label is None else str(label)))
        worker.wake()

    def _stream_has_pending(self) -> bool:
        with self._stream_queue_lock:
            return bool(self._stream_pending)

    def _stream_wait_sent(self, timeout: float = 5.0) -> bool:
        """Wait until every appended frame has left the worker (tests and scripts)."""
        return self._stream_worker.wait_idle(self._stream_has_pending, timeout)

    def _stream_flush(self) -> None:
        """Publish all pending frames as one transfer when a credit is free."""
        with self._stream_lock:
            if not self._stream_credits.available():
                return
            with self._stream_queue_lock:
                batch, self._stream_pending = self._stream_pending, []
            if batch:
                self._stream_publish(batch)

    def _stream_publish(self, batch: list[tuple[np.ndarray, str | None]]) -> None:
        if self.is_rgb or int(self.n_panels) != 1:
            raise ValueError("Show3D.stream appends to one grayscale stack; set_image() replaced it with another layout.")
        old_n = int(self.n_slices)
        rows, cols = batch[0][0].shape
        if old_n and tuple(self._data.shape[1:]) != (rows, cols):
            raise ValueError(f"Frame shape {(rows, cols)} does not match the stack's {tuple(self._data.shape[1:])}.")
        new_n = old_n + len(batch)
        current = self._data if old_n else np.empty((0, rows, cols), dtype=np.float32)
        buffer = self._stream_buffer
        # Frames land in a buffer that doubles when full, so an append copies one frame, not the stack.
        if buffer is None or buffer.shape[1:] != (rows, cols) or buffer.shape[0] < new_n or (old_n and current.base is not buffer):
            capacity = max(new_n, 2 * buffer.shape[0] if buffer is not None and buffer.shape[1:] == (rows, cols) else new_n)
            grown = np.empty((capacity, rows, cols), dtype=np.float32)
            grown[:old_n] = current
            buffer = grown
        for offset, (frame, _label) in enumerate(batch):
            buffer[old_n + offset] = frame
        self._stream_buffer = buffer
        seq = self._stream_credits.take()
        # The frames go first: the browser holds them before n_slices says they exist.
        self.send(
            {"type": "show3d_stream_frames", "version": 1, "seq": seq, "start": old_n,
             "count": len(batch), "rows": rows, "cols": cols, "revision": int(self.frame_seq)},
            buffers=[memoryview(buffer[index]) for index in range(old_n, new_n)],
        )
        new_frames = buffer[old_n:new_n]
        now = time.monotonic()
        follow = (self.stream_follow and not self.playing
                  and (old_n == 0 or int(self.slice_idx) >= old_n - 1)
                  and now - self._stream_interaction_at >= STREAM_FOLLOW_PAUSE_S)
        labels = list(self.labels) if len(self.labels) == old_n else [str(index) for index in range(old_n)]
        labels += [str(old_n + offset) if label is None else label for offset, (_frame, label) in enumerate(batch)]
        low, high = float(new_frames.min()), float(new_frames.max())
        with self.hold_sync():
            self._data = buffer[:new_n]
            self._display_data = self._data
            if not old_n:
                self._panel_width = cols
                self.height, self.width = rows, cols
                self.source_height, self.source_panel_width = rows, cols
                self.folder_waiting = False
                self.folder_status = ""
            else:
                low, high = min(self._data_min_off, low), max(self._data_max_off, high)
            self._data_min_off, self._data_max_off = low, high
            if self.diff_mode == "off":
                self.data_min, self.data_max = low, high
            self.n_slices = new_n
            self.labels = labels
            rotations = list(self.frame_rotations)
            self.frame_rotations = (rotations if len(rotations) == old_n else [0] * old_n) + [0] * len(batch)
            if self.auto_contrast and self.diff_mode == "off":
                ranges = [stack_percentiles([frame], self.percentile_low, self.percentile_high) for frame in new_frames]
                vmins, vmaxs = (list(self.auto_vmins), list(self.auto_vmaxs)) if old_n else ([], [])
                if len(vmins) == len(vmaxs) == old_n and all(span is not None for span in ranges):
                    self.auto_vmins = [*vmins, *(span[0] for span in ranges)]
                    self.auto_vmaxs = [*vmaxs, *(span[1] for span in ranges)]
                else:
                    self._refresh_auto_contrast_ranges()
            elif self.diff_mode != "off":
                self._on_diff_mode_change()  # zero-centred range and contrast over the grown stack
            if follow:
                self.slice_idx = new_n - 1
        if self.roi_active:
            self._on_roi_change()

    def _stream_resend(self) -> None:
        """Send the whole stack once, for a view that mounted late or missed a transfer."""
        with self._stream_lock:
            with self.hold_sync():
                if int(self.n_slices):
                    self._pack_exact_offline_stack()
                self.frame_seq += 1
            self._stream_credits.ready()

    def _handle_stream_msg(self, _widget, content, _buffers) -> None:
        if not isinstance(content, dict) or content.get("version") != 1:
            return
        kind = content.get("type")
        if kind == "show3d_frontend_ready":
            with self._stream_lock:
                self._stream_credits.ready()
                if content.get("frames") != int(self.n_slices) or content.get("revision") != int(self.frame_seq):
                    self._stream_resend()
        elif kind == "show3d_stream_ack":
            if self._stream_credits.acknowledge(content.get("seq")) and self._stream_worker is not None:
                self._stream_worker.wake()
        elif kind == "show3d_stream_resync":
            self._stream_resend()

    def _stream_trait_is_stale(self) -> bool:
        if self._stream_worker is None or self._data is None or self.is_rgb:
            return False
        return len(self._offline_float_stack) != int(self._data.nbytes)

    def get_state(self, key=None, drop_defaults=False):
        """A streamed stack is synced by appends; any full snapshot gets the complete stack."""
        lock = getattr(self, "_stream_lock", None)
        if lock is None:
            return super().get_state(key=key, drop_defaults=drop_defaults)
        with lock:
            state = super().get_state(key=key, drop_defaults=drop_defaults)
            if "_offline_float_stack" in state and self._stream_trait_is_stale():
                state["_offline_float_stack"] = memoryview(np.ascontiguousarray(self._data)).cast("B")
            return state

    def set_state(self, sync_data) -> None:
        """Browser edits wait for an in-progress append; a slider move pauses following."""
        lock = getattr(self, "_stream_lock", None)
        if lock is None:
            return super().set_state(sync_data)
        if "slice_idx" in sync_data or "playing" in sync_data:
            self._stream_interaction_at = time.monotonic()
        with lock:
            super().set_state(sync_data)

    def close(self) -> None:
        """Stop the stream worker, then close the widget comm."""
        worker = self._stream_worker
        if worker is not None:
            worker.stop()
        super().close()
