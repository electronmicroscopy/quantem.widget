"""Bounded native CDP mouse input for an already open, owned browser page.

This measures input delivery independently of serialized Playwright calls.
It does not measure scientific image throughput. No Network domain, browser
creation, tab selection by URL, or application state mutation is enabled.
"""

import json
import math
import statistics
import time
import urllib.request
from collections.abc import Callable, Mapping

import websocket


class _Connection:
    def __init__(self, url: str, timeout: float):
        self.socket = websocket.create_connection(
            url, suppress_origin=True, timeout=timeout
        )
        self.timeout = timeout
        self.serial = 0

    def send(self, method: str, params: dict) -> int:
        self.serial += 1
        self.socket.settimeout(self.timeout)
        self.socket.send(json.dumps({
            "id": self.serial, "method": method, "params": params,
        }))
        return self.serial

    def receive(self, timeout: float) -> dict | None:
        self.socket.settimeout(max(timeout, 0.0001))
        try:
            message = self.socket.recv()
        except websocket.WebSocketTimeoutException:
            return None
        if not message:
            raise ConnectionError("The page CDP connection closed during input.")
        return json.loads(message)

    def call(self, method: str, params: dict, timeout: float) -> dict:
        self.socket.settimeout(timeout)
        request = self.send(method, params)
        deadline = time.perf_counter() + timeout
        while remaining := deadline - time.perf_counter():
            if remaining <= 0:
                break
            reply = self.receive(remaining)
            if reply is not None and reply.get("id") == request:
                if "error" in reply:
                    raise RuntimeError(f"CDP {method} failed: {reply['error']}")
                return reply.get("result", {})
        raise TimeoutError(f"CDP {method} was not acknowledged within {timeout:g}s.")

    def close(self):
        self.socket.close()


def _mouse(kind: str, position: tuple[float, float], held: bool) -> dict:
    return {
        "type": kind, "x": position[0], "y": position[1],
        "button": "left" if held or kind == "mouseReleased" else "none",
        "buttons": 1 if held else 0, "clickCount": 1,
        "pointerType": "mouse",
    }


def _default_trajectory(elapsed: float) -> tuple[float, float]:
    return 0.5 + 0.14 * math.sin(elapsed * 8.04), 0.5 + 0.14 * math.cos(elapsed * 9.96)


def _run_moves(connection, box, duration, rate, limit, timeout, trajectory, report):
    """Keep only the newest unsent position while receiving independent ACKs."""
    began = time.perf_counter()
    report.update(host_wall_start_s=time.time(), host_monotonic_start_s=began)
    deadline = began + duration
    total = math.ceil(duration * rate)
    next_index = 0
    pending = None
    outstanding = {}
    last_position = (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    while time.perf_counter() < deadline or next_index < total or pending or outstanding:
        now = time.perf_counter()
        latest = min(total - 1, math.floor((now - began) * rate))
        if latest >= next_index:
            due = latest - next_index + 1
            report["requested"] += due
            report["coalesced"] += due - 1 + (pending is not None)
            horizontal, vertical = trajectory(latest / rate)
            if not all(math.isfinite(value) for value in (horizontal, vertical)):
                raise ValueError("The trajectory must return finite CSS coordinates.")
            pending = {
                "index": latest, "scheduled_s": latest / rate,
                "x": box["x"] + box["width"] * horizontal,
                "y": box["y"] + box["height"] * vertical,
            }
            next_index = latest + 1
        if pending is not None and len(outstanding) < limit:
            position = (pending["x"], pending["y"])
            pending["sent_s"] = time.perf_counter() - began
            request = connection.send("Input.dispatchMouseEvent", _mouse("mouseMoved", position, True))
            pending["id"] = request
            outstanding[request] = pending
            report["events"].append(pending)
            report["sent"] += 1
            report["max_outstanding"] = max(report["max_outstanding"], len(outstanding))
            last_position = position
            pending = None
        now = time.perf_counter()
        oldest_deadline = min((began + item["sent_s"] + timeout
                               for item in outstanding.values()), default=math.inf)
        if now >= oldest_deadline:
            raise TimeoutError(f"Mouse move acknowledgement exceeded {timeout:g}s.")
        if now >= deadline and next_index == total and pending is None and not outstanding:
            break
        next_tick = began + next_index / rate if next_index < total else deadline
        wait = min(0.05, oldest_deadline - now)
        if next_index < total or next_tick > now:
            wait = min(wait, max(0.000001, next_tick - now))
        reply = connection.receive(wait)
        if reply is not None and reply.get("id") in outstanding:
            event = outstanding.pop(reply["id"])
            event["ack_s"] = time.perf_counter() - began
            event["ack_latency_s"] = event["ack_s"] - event["sent_s"]
            if "error" in reply:
                event["error"] = reply["error"]
                raise RuntimeError(f"CDP mouse move failed: {reply['error']}")
            report["acknowledged"] += 1
    report["elapsed_with_drain_s"] = time.perf_counter() - began
    report["drain_s"] = max(0, report["elapsed_with_drain_s"] - duration)
    report["final_position"] = list(last_position)


def drag(
    target_id: str,
    box: Mapping[str, float],
    *,
    endpoint: str,
    duration_s: float = 5,
    rate_hz: float = 120,
    max_inflight: int = 4,
    ack_timeout_s: float = 10,
    trajectory: Callable[[float], tuple[float, float]] | None = None,
    start: tuple[float, float] = (0.5, 0.5),
    before_release: Callable[[], None] | None = None,
) -> dict:
    """Drive a real held mouse gesture with bounded concurrent CDP requests.

    Parameters
    ----------
    target_id : str
        Explicit ID of the caller's already open page. No other tab is selected.
    box : mapping
        CSS viewport ``x``, ``y``, ``width``, and ``height`` of the drag surface.
    endpoint : str
        Chrome's HTTP debugging endpoint.
    duration_s, rate_hz : float
        Requested input interval and position generation rate.
    max_inflight : int
        Maximum unacknowledged mouse moves; one latest unsent position is kept.
    ack_timeout_s : float
        Upper bound on waiting for one CDP command.
    trajectory : callable, optional
        Given elapsed seconds, return fractional horizontal/vertical position
        within ``box``.
    start : tuple
        Fractional initial press position within ``box``; default is its center.
    before_release : callable, optional
        Evidence callback after the timed input/drain and before release. Its
        screenshot or final-image checks do not enter the timed input interval.

    Returns
    -------
    dict
        Counts, per-event timings, release status and any protocol failure.
        ``status='complete'`` requires all sent moves and release acknowledged.
        The drain interval is reported separately from requested input duration.

    Examples
    --------
    >>> result = drag(target_id, dp.bounding_box(), rate_hz=120)
    >>> assert result["status"] == "complete", result["error"]
    """
    if not all(math.isfinite(value) and value > 0
               for value in (duration_s, rate_hz, ack_timeout_s)):
        raise ValueError("Duration, input rate, and ACK timeout must be positive and finite.")
    if not isinstance(max_inflight, int) or not 1 <= max_inflight <= 4:
        raise ValueError("Use max_inflight from 1 to 4; input backlog must remain bounded.")
    box = {key: float(box[key]) for key in ("x", "y", "width", "height")}
    if not all(math.isfinite(value) for value in box.values()) or min(box["width"], box["height"]) <= 0:
        raise ValueError("The drag box must have finite coordinates and positive dimensions.")
    if len(start) != 2 or not all(math.isfinite(value) and 0 <= value <= 1 for value in start):
        raise ValueError("The initial press must be inside the drag surface.")
    with urllib.request.urlopen(endpoint.rstrip("/") + "/json/list", timeout=ack_timeout_s) as reply:
        targets = json.load(reply)
    target = next((item for item in targets if item["id"] == target_id and item["type"] == "page"), None)
    if target is None:
        raise ValueError("The explicit page target was not found; refresh its target ID.")
    report = {
        "driver": "native CDP Input.dispatchMouseEvent; bounded ACKs",
        "duration_s": duration_s, "rate_hz": rate_hz,
        "inflight_limit": max_inflight, "requested": 0, "sent": 0,
        "acknowledged": 0, "coalesced": 0, "max_outstanding": 0,
        "events": [], "status": "failed", "error": None, "released": False,
    }
    connection = _Connection(target["webSocketDebuggerUrl"], ack_timeout_s)
    position = (box["x"] + box["width"] * start[0], box["y"] + box["height"] * start[1])
    pressed = False
    try:
        clock = connection.call("Runtime.evaluate", {
            "expression": "({timeOrigin:performance.timeOrigin,now:performance.now(),wall:Date.now()})",
            "returnByValue": True,
        }, ack_timeout_s)
        report["browser_clock"] = clock["result"]["value"]
        connection.call("Input.dispatchMouseEvent", _mouse("mouseMoved", position, False), ack_timeout_s)
        pressed = True  # Release even if the press is delivered but its ACK fails.
        connection.call("Input.dispatchMouseEvent", _mouse("mousePressed", position, True), ack_timeout_s)
        _run_moves(connection, box, duration_s, rate_hz, max_inflight,
                   ack_timeout_s, trajectory or _default_trajectory, report)
        if before_release is not None:
            before_release()
        report["status"] = "complete"
    except (OSError, websocket.WebSocketException, RuntimeError, ValueError) as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if pressed:
            if report["events"]:
                last = report["events"][-1]
                position = (last["x"], last["y"])
            try:
                connection.call("Input.dispatchMouseEvent", _mouse("mouseReleased", position, False), ack_timeout_s)
                report["released"] = True
            except (OSError, websocket.WebSocketException, RuntimeError, ValueError) as error:
                report["status"] = "failed"
                report["release_error"] = f"{type(error).__name__}: {error}"
        report["host_wall_release_s"] = time.time()
        connection.close()
    latencies = [event["ack_latency_s"] for event in report["events"] if "ack_latency_s" in event]
    report["ack_latency_median_s"] = statistics.median(latencies) if latencies else None
    report["ack_latency_max_s"] = max(latencies, default=None)
    report["sent_during_interval"] = sum(event["sent_s"] < duration_s for event in report["events"])
    report["acknowledged_during_interval"] = sum(
        event.get("ack_s", math.inf) < duration_s and "error" not in event
        for event in report["events"]
    )
    return report
