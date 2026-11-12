"""CPU scheduler checks; these do not open or drive a browser."""

import io
import json
import math
from contextlib import nullcontext

import native_pointer as pointer
import pytest

BOX = {"x": 10, "y": 20, "width": 100, "height": 200}


class Clock:
    now = 0.0

    def perf_counter(self):
        return self.now

    def time(self):
        return 1_000_000 + self.now


class Connection:
    def __init__(self, clock, latency):
        self.clock = clock
        self.latency = latency
        self.serial = 0
        self.pending = {}
        self.peak = 0
        self.controls = []
        self.closed = False

    def send(self, method, params):
        assert method == "Input.dispatchMouseEvent"
        assert params["type"] == "mouseMoved" and params["buttons"] == 1
        self.serial += 1
        self.pending[self.serial] = self.clock.now + self.latency(self.serial)
        self.peak = max(self.peak, len(self.pending))
        assert self.peak <= 4
        return self.serial

    def receive(self, timeout):
        if self.pending:
            earliest = min(self.pending, key=self.pending.get)
            when = self.pending[earliest]
            if when <= self.clock.now + timeout:
                self.clock.now = when
                del self.pending[earliest]
                return {"id": earliest, "result": {}}
        self.clock.now += timeout
        return None

    def call(self, method, params, timeout):
        if method == "Runtime.evaluate":
            return {"result": {"value": {"timeOrigin": 1_000_000}}}
        self.controls.append(params)
        return {}

    def close(self):
        self.closed = True


def setup(monkeypatch, latency):
    clock = Clock()
    connection = Connection(clock, latency)
    monkeypatch.setattr(pointer, "time", clock)
    monkeypatch.setattr(pointer, "_Connection", lambda url, timeout: connection)
    monkeypatch.setattr(pointer.urllib.request, "urlopen", lambda *args, **kwargs: nullcontext(
        io.StringIO(json.dumps([{"id": "owned", "type": "page", "webSocketDebuggerUrl": "ws://unused"}]))))
    return connection


@pytest.mark.parametrize("slow", [False, True])
def test_generated_positions_are_bounded_and_finish_at_latest(monkeypatch, slow):
    connection = setup(monkeypatch, lambda serial: (0.08 if serial % 3 else 0.02) if slow else 0.002)
    result = pointer.drag("owned", BOX, endpoint="http://unused", duration_s=0.15, rate_hz=120,
                          start=(0.3, 0.7))
    assert (connection.controls[1]["x"], connection.controls[1]["y"]) == (40, 160)
    assert result["status"] == "complete" and result["released"] and connection.closed
    assert result["requested"] == math.ceil(0.15 * 120)
    assert result["sent"] == result["acknowledged"]
    assert result["sent"] + result["coalesced"] == result["requested"]
    assert 1 <= result["max_outstanding"] <= 4
    assert result["events"][-1]["index"] == result["requested"] - 1
    assert connection.controls[-1]["type"] == "mouseReleased"
    assert connection.controls[-1]["buttons"] == 0
    assert result["final_position"] == [connection.controls[-1]["x"], connection.controls[-1]["y"]]
    assert result["host_wall_release_s"] >= result["host_wall_start_s"] + 0.15
    if slow:
        assert result["coalesced"] > 0 and result["drain_s"] > 0
    else:
        assert result["coalesced"] == 0


def test_missing_ack_reports_failure_and_releases_mouse(monkeypatch):
    connection = setup(monkeypatch, lambda serial: math.inf)
    result = pointer.drag("owned", BOX, endpoint="http://unused", duration_s=0.1, rate_hz=120, ack_timeout_s=0.2)
    assert result["status"] == "failed" and "TimeoutError" in result["error"]
    assert result["sent"] == 4 and result["acknowledged"] == 0
    assert result["released"] and connection.closed
    assert connection.controls[-1]["type"] == "mouseReleased"


def test_batched_browser_acknowledgements_do_not_serialize_sixty_hz_input(monkeypatch):
    """Observed 68/52/35/18 ms ACK batches still allow a bounded 60 Hz sender."""
    delays = (0.068, 0.052, 0.035, 0.018, 0.007)
    setup(monkeypatch, lambda serial: 0.008 if serial <= 3 else delays[(serial - 4) % 5])
    result = pointer.drag("owned", BOX, endpoint="http://unused", duration_s=5, rate_hz=60)
    assert result["status"] == "complete" and result["released"]
    assert result["requested"] == 300
    assert result["sent_during_interval"] >= 298
    assert result["sent"] == result["acknowledged"]
    assert result["max_outstanding"] == 4


def test_frame_count_exposes_stalls_and_complete_images_still_require_all_seven_panels():
    """A normal median interval does not imply full frame throughput."""
    from test_qem_drag import _summary

    panels = [f"show4dstem-compare-{index}" for index in range(7)]
    evidence = {
        "frames": [-1, 0, 16, 32, 48, 200, 201],
        "presentations": [
            {"queued": 20, "ready": 30, "revision": 1, "targets": panels},
            {"queued": 40, "ready": 60, "revision": 2, "targets": panels[:-1]},
            {"queued": 190, "ready": 210, "revision": 3, "targets": panels},
        ],
        "inputs": [{"at": 30, "trusted": True}, {"at": 40, "trusted": False}],
    }
    result = _summary(evidence, 7, 0, 200)
    assert result["raf_hz_during_drag"] == 25
    assert result["median_frame_interval_ms"] == 16
    assert result["p95_frame_interval_ms"] == pytest.approx(131.6)
    assert result["new_complete_image_revisions"] == 1
    assert result["new_image_hz"] == result["delivered_input_hz"] == 5


@pytest.mark.parametrize("frames", [[], [50], [50, 60]])
def test_frame_interval_summary_handles_sparse_observations(frames):
    from test_qem_drag import _summary

    result = _summary({"frames": frames, "presentations": [], "inputs": []}, 7, 0, 200)
    assert result["raf_hz_during_drag"] == len(frames) * 5
    assert result["median_frame_interval_ms"] == (10 if len(frames) == 2 else None)
    assert result["p95_frame_interval_ms"] == (10 if len(frames) == 2 else None)
