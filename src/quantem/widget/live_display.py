"""Bounded browser transfers for viewers that receive images while the notebook runs.

``Show4DSTEM.stream`` set the rule this module packages for ``Show2D.stream``
and ``Show3D.stream``: at most two display updates await browser
acknowledgment, and later updates coalesce until a credit returns. A kernel
busy with a user cell cannot dispatch acknowledgments, so after two seconds
without one a further transfer is allowed; a blocked or closed browser
therefore receives at most one update every two seconds instead of an
unbounded backlog.

Callers (``append``/``update``) only copy their image and wake a worker
thread; the worker owns every transfer, so the caller's thread never waits on
the browser.
"""

import threading
import time
from collections.abc import Callable

# Updates in flight before the next one waits for an acknowledgment (Show4DSTEM uses the same window).
DISPLAY_CREDITS = 2
# Seconds without an acknowledgment after which one more update is sent anyway.
STALLED_ACK_S = 2.0


class DisplayCredits:
    """Sequence numbers and acknowledgments for one widget's live transfers.

    Before any browser view announces itself, transfers are not gated: the
    first view may still be mounting inside the cell that started the stream.
    """

    def __init__(self, credits: int = DISPLAY_CREDITS, stalled_ack_s: float = STALLED_ACK_S):
        self.credits = int(credits)
        self.stalled_ack_s = float(stalled_ack_s)
        self.negotiated = False
        self.seq = 0
        self.acknowledged = 0
        self.sent_at = 0.0
        self._lock = threading.Lock()

    @property
    def outstanding(self) -> int:
        """Transfers sent but not yet acknowledged."""
        return self.seq - self.acknowledged

    def available(self, now: float | None = None) -> bool:
        """Whether one more transfer may leave now."""
        with self._lock:
            if not self.negotiated or self.outstanding < self.credits:
                return True
            now = time.monotonic() if now is None else now
            return now - self.sent_at >= self.stalled_ack_s

    def take(self) -> int:
        """Number the next transfer."""
        with self._lock:
            self.seq += 1
            self.sent_at = time.monotonic()
            return self.seq

    def ready(self) -> None:
        """A view mounted or resynchronized: nothing earlier is still awaited."""
        with self._lock:
            self.negotiated = True
            self.acknowledged = self.seq

    def acknowledge(self, seq) -> bool:
        """Apply a cumulative acknowledgment; stale or invalid numbers are ignored."""
        if type(seq) is not int:
            return False
        with self._lock:
            if not self.acknowledged < seq <= self.seq:
                return False
            self.acknowledged = seq
            return True


class LiveDisplayWorker:
    """Daemon thread that runs ``flush`` when woken, and periodically while transfers are held back."""

    def __init__(self, name: str, flush: Callable[[], None], poll_s: float = 0.05):
        self._flush = flush
        self._poll_s = float(poll_s)
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._idle = threading.Condition()
        self._busy = False
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def wake(self) -> None:
        self._wake.set()

    @property
    def running(self) -> bool:
        return self._thread.is_alive() and not self._stop.is_set()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout)

    def wait_idle(self, pending: Callable[[], bool], timeout: float = 5.0) -> bool:
        """Block until ``pending()`` is False and no flush is running; False on timeout."""
        deadline = time.monotonic() + timeout
        with self._idle:
            while self._busy or pending():
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self.running:
                    return not (self._busy or pending())
                self._wake.set()
                self._idle.wait(min(remaining, self._poll_s))
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(self._poll_s)
            self._wake.clear()
            if self._stop.is_set():
                break
            with self._idle:
                self._busy = True
            try:
                self._flush()
            except Exception as error:  # keep the viewer alive; the error is shown on the next call
                self.error = error
            finally:
                with self._idle:
                    self._busy = False
                    self._idle.notify_all()
