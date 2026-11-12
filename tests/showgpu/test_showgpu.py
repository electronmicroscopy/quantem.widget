"""A live monitor updates during work and releases its sampling thread."""

import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import psutil
import pytest
import traitlets

from quantem.widget import ShowGPU, showgpu


def test_live_sampling_pause_refresh_resume_and_close(monkeypatch):
    samples = []
    received = threading.Event()

    def sample(**kwargs):
        samples.append(time.monotonic())
        if len(samples) >= 3:
            received.set()
        return {"gpus": [], "timestamp": time.time(), "error": ""}

    monkeypatch.setattr(showgpu, "sample", sample)
    monitor = ShowGPU(interval=0.2)
    try:
        assert monitor.users is True and monitor.expand is True
        # The main execution thread is occupied; monitoring still advances.
        assert received.wait(timeout=2)
        monitor.running = False
        time.sleep(0.25)
        paused = monitor.snapshot["sequence"]
        time.sleep(0.25)
        assert monitor.snapshot["sequence"] == paused
        monitor.refresh()
        assert monitor.snapshot["sequence"] == paused + 1
        received.clear()
        monitor.running = True
        assert received.wait(timeout=2)
    finally:
        monitor.close()
    assert not monitor._worker.is_alive()
    after_close = len(samples)
    monitor.refresh()
    assert len(samples) == after_close


def test_failed_sample_is_visible_and_next_refresh_recovers(monkeypatch):
    def failed(**kwargs):
        raise OSError("Driver temporarily unavailable")

    monkeypatch.setattr(showgpu, "sample", failed)
    monitor = ShowGPU(running=False)
    try:
        assert "temporarily unavailable" in monitor.snapshot["error"]
        assert monitor.snapshot["gpus"] == []
        monkeypatch.setattr(
            showgpu,
            "sample",
            lambda **kwargs: {"gpus": [], "timestamp": time.time(), "error": ""},
        )
        monitor.refresh()
        assert monitor.snapshot["error"] == ""
    finally:
        monitor.close()


def test_refresh_requests_never_overlap_queries(monkeypatch):
    monitor = ShowGPU(running=False)
    entered = threading.Event()
    finish = threading.Event()
    calls = []

    def slow_sample(**kwargs):
        calls.append(1)
        entered.set()
        finish.wait(timeout=2)
        return {"gpus": [], "timestamp": time.time(), "error": ""}

    monkeypatch.setattr(showgpu, "sample", slow_sample)
    worker = threading.Thread(target=monitor.refresh)
    worker.start()
    try:
        assert entered.wait(timeout=1)
        monitor.refresh()
        assert len(calls) == 1
    finally:
        finish.set()
        worker.join(timeout=2)
        monitor.close()


def test_display_options_change_without_recreating_a_paused_monitor(monkeypatch):
    sampled_users = []

    def sample(*, users):
        sampled_users.append(users)
        return {"gpus": [], "timestamp": time.time(), "error": ""}

    monkeypatch.setattr(showgpu, "sample", sample)
    monitor = ShowGPU(users=False, expand=False, size="large", running=False)
    try:
        assert monitor.size == "large"
        assert monitor.expand is False
        assert sampled_users == [False]
        for size in ("small", "medium", "large", "xlarge"):
            monitor.size = size
            assert monitor.size == size
        # Resizing is a display change, not another driver query.
        assert sampled_users == [False]
        monitor.expand = True
        assert sampled_users == [False]
        monitor.users = True
        assert sampled_users == [False, True]
        with pytest.raises(traitlets.TraitError, match="small"):
            monitor.size = "huge"
    finally:
        monitor.close()


def test_compute_users_are_grouped_per_gpu_and_missing_values_stay_missing(monkeypatch):
    telemetry = pytest.importorskip("quantem.gpu.device.telemetry")
    queried_users = []

    def status(*, processes):
        queried_users.append(processes)
        result = {"gpus": [{"uuid": "GPU-a"}, {"uuid": "GPU-b"}], "error": ""}
        if processes:
            result["processes"] = [
                {"gpu_uuid": "GPU-a", "pid": 1, "used_bytes": 100},
                {"gpu_uuid": "GPU-a", "pid": 2, "used_bytes": 200},
                {"gpu_uuid": "GPU-b", "pid": 1, "used_bytes": None},
                {"gpu_uuid": "GPU-a", "pid": 3, "used_bytes": 300},
            ]
        return result

    def process(pid=None):
        if pid == 3:
            raise psutil.NoSuchProcess(pid)
        return SimpleNamespace(
            as_dict=lambda **kwargs: {"username": "example-user", "name": "python"},
            memory_info=lambda: SimpleNamespace(rss=50),
            parents=list,
            username=lambda: "example-user",
        )

    monkeypatch.setattr(telemetry, "status", status)
    monkeypatch.setattr(showgpu.psutil, "Process", process)
    snapshot = showgpu.sample(users=True)
    assert snapshot["gpus"][0]["users"] == [
        {
            "name": "example-user",
            "used_bytes": 300,
            "processes": 2,
            "process_details": [
                {"pid": 1, "name": "python", "used_bytes": 100, "created_at": None, "can_terminate": False},
                {"pid": 2, "name": "python", "used_bytes": 200, "created_at": None, "can_terminate": False},
            ],
        },
        {
            "name": None,
            "used_bytes": 300,
            "processes": 1,
            "process_details": [{"pid": 3, "name": None, "used_bytes": 300, "created_at": None, "can_terminate": False}],
        },
    ]
    assert snapshot["gpus"][1]["users"] == [
        {
            "name": "example-user",
            "used_bytes": None,
            "processes": 1,
            "process_details": [{"pid": 1, "name": "python", "used_bytes": None, "created_at": None, "can_terminate": False}],
        },
    ]
    assert "processes" not in snapshot
    snapshot = showgpu.sample(users=False)
    assert queried_users == [True, False]
    assert all("users" not in gpu for gpu in snapshot["gpus"])


def test_watch_unload_select_and_stop_only_the_current_acquisition(monkeypatch):
    monkeypatch.setattr(showgpu, "sample", lambda **kwargs: {"gpus": []})

    class Recording:
        def __init__(self):
            self.scans = [{"id": "scan-a", "state": "Ready"}]
            self.receiving = True

        def memory_snapshot(self):
            return {"scans": list(self.scans), "receiving": self.receiving}

        def unload(self, scan_id):
            self.scans = [scan for scan in self.scans if scan["id"] != scan_id]

        def close(self):
            pass  # The receiver finishes its current scan asynchronously.

    selected, stopped = [], threading.Event()
    viewer = SimpleNamespace(refresh_stream=lambda scan=None: selected.append(scan))
    # A real widget is weak-referenceable; use a small view owner here as well.
    class View:
        refresh_stream = staticmethod(viewer.refresh_stream)

    view, recording = View(), Recording()
    monitor = ShowGPU(running=False)
    try:
        monitor.watch(recording, viewer=view, stop=stopped.set)
        watch_id = monitor.snapshot["recording"]["watch_id"]
        monitor.handle_message(None, {"action": "select_scan", "watch_id": watch_id, "scan_id": "scan-a"}, [])
        assert selected == ["scan-a"]
        monitor.handle_message(None, {"action": "unload", "watch_id": watch_id, "scan_id": "scan-a"}, [])
        assert not monitor.snapshot["recording"]["scans"] and selected[-1] is None
        assert any("no longer retained" in event["message"] for event in monitor.snapshot["activity"])
        monitor.handle_message(None, {"action": "stop_acquisition", "watch_id": watch_id}, [])
        assert stopped.wait(2)
        assert monitor.action_status["busy"] and recording.receiving
        recording.receiving = False
        monitor._action_thread.join(2)
        assert stopped.is_set() and not recording.receiving
        assert "Acquisition stopped" in monitor.action_status["message"]
        replacement = Recording()
        monitor.watch(replacement)
        monitor.handle_message(None, {"action": "unload", "watch_id": watch_id, "scan_id": "scan-a"}, [])
        assert replacement.scans and monitor.action_status["error"]
    finally:
        recording.receiving = False
        monitor.close()


def test_terminate_owned_child_preserves_kernel_and_checks_process_identity(monkeypatch):
    monkeypatch.setattr(showgpu, "sample", lambda **kwargs: {"gpus": []})
    child = subprocess.Popen(
        [sys.executable, "-c", ("import signal, sys, time\n"
         "def stop(signum, frame):\n"
         " print('stopping', flush=True)\n"
         " sys.stdin.readline()\n"
         " sys.exit(0)\n"
         "signal.signal(signal.SIGTERM, stop)\n"
         "print('ready', flush=True)\n"
         "time.sleep(30)")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    monitor = ShowGPU(running=False)
    try:
        assert child.stdout.readline().strip() == "ready"
        created = psutil.Process(child.pid).create_time()
        monitor.snapshot = {"gpus": [{"users": [{"process_details": [
            {"pid": child.pid, "created_at": created, "can_terminate": True},
            {"pid": os.getpid(), "created_at": psutil.Process().create_time(), "can_terminate": True},
        ]}]}]}
        with pytest.raises(ValueError, match="current live sample"):
            monitor.terminate_process(child.pid, created - 1)
        assert child.poll() is None
        with pytest.raises(ValueError, match="identity or ownership"):
            monitor.terminate_process(os.getpid(), psutil.Process().create_time())
        monitor.handle_message(None, {"action": "terminate", "pid": child.pid, "created_at": created}, [])
        assert child.stdout.readline().strip() == "stopping"
        assert monitor.refresh()["terminating"] == [child.pid]
        assert "waiting for exit" in monitor.action_status["message"]
        child.stdin.write("finish\n")
        child.stdin.flush()
        child.wait(timeout=3)
        assert not monitor.refresh()["terminating"]
        assert monitor.action_status["message"] == f"PID {child.pid} exited."
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=3)
        monitor.close()
