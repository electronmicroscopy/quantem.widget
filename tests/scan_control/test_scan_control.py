"""ScanControl starts the runner's steps with the panel's settings and shows its status."""

import threading
import time
from collections import namedtuple

import pytest
import traitlets

from quantem.widget import ScanControl
from quantem.widget.scan_control import FOREVER

Status = namedtuple("Status", "state message dcu_ok revolon_ok high_voltage done total dropped files "
                              "free_bytes")


class FakeRunner:
    """The scan_runner.Runner interface, recording each step instead of moving a beam."""

    def __init__(self):
        self.calls = []
        self.state = Status("NOT_CONNECTED", "Not connected", False, False, None, 0, 0, None, [], None)
        self.closed = False

    def start(self, step, *args):
        self.calls.append((step, *args))
        if step == "connect":
            self.state = self.state._replace(state="READY", message="Ready - press Go", dcu_ok=True,
                                             revolon_ok=True, files=["scan_1_master.h5", "scan_1_data_000001.h5"],
                                             free_bytes=2 ** 40)
        return True

    def status(self):
        return self.state

    def stop(self):
        self.calls.append(("stop",))

    def file_info(self, names):
        return {name: (4 if name.endswith("master.h5") else 100, "2026-10-27T04:15:16+00:00") for name in names}

    def close(self):
        self.closed = True


def message(widget, **content):
    widget._handle_message(widget, content, [])


def test_actions_start_runner_steps_with_the_panel_settings():
    runner = FakeRunner()
    panel = ScanControl(runner, size=256, binned=False, count=3, folder="downloads")
    try:
        message(panel, action="connect")
        assert runner.calls[0] == ("connect", (256, False, "rdma", False))
        assert panel.status["state"] == "READY" and panel.status["dcu_ok"]
        assert panel.status["files"] == ["scan_1_master.h5", "scan_1_data_000001.h5"]
        assert [row["name"] for row in panel.status["scans"]] == ["scan_1"]
        assert panel.status["scans"][0]["bytes"] == 4 + 100
        message(panel, action="go")
        assert runner.calls[-1] == ("run", (256, False, "rdma", False), 3, None, True, True, True, True, False)
        panel.forever = True
        message(panel, action="go")
        assert runner.calls[-1] == ("run", (256, False, "rdma", False), FOREVER, None, True, True, True, True, False)
        message(panel, action="stop")
        assert runner.calls[-1] == ("stop",)
        message(panel, action="download", scans=["scan_1"], format="h5")
        assert runner.calls[-1] == ("download", ["scan_1_master.h5", "scan_1_data_000001.h5"], "downloads")
        message(panel, action="delete", scans=["scan_1"])
        assert runner.calls[-1] == ("delete", ["scan_1_master.h5", "scan_1_data_000001.h5"])
    finally:
        panel.close()
    assert runner.closed


def test_a_setting_changed_while_ready_sets_the_detector_up_again():
    runner = FakeRunner()
    panel = ScanControl(runner)
    try:
        panel.size = 1024                       # not connected: nothing to set up
        assert runner.calls == []
        message(panel, action="connect")
        panel.binned = False
        assert runner.calls[-1] == ("configure", (1024, False, "rdma", False))
    finally:
        panel.close()


def test_status_follows_the_runner_without_a_cell_running():
    runner = FakeRunner()
    panel = ScanControl(runner, interval=0.05)
    try:
        runner.state = runner.state._replace(state="SCANNING", message="Scan 2 of 5", done=1, total=5)
        deadline = time.monotonic() + 2
        while panel.status.get("message") != "Scan 2 of 5" and time.monotonic() < deadline:
            time.sleep(0.02)
        assert panel.status["state"] == "SCANNING" and panel.status["busy"]
        assert panel.status["done"] == 1 and panel.status["total"] == 5
    finally:
        panel.close()


def test_invalid_input_names_what_to_provide():
    with pytest.raises(TypeError, match="scan_runner.Runner"):
        ScanControl(object())
    runner = FakeRunner()
    panel = ScanControl(runner)
    try:
        with pytest.raises(traitlets.TraitError, match="size must be one of"):
            panel.size = 300
        with pytest.raises(traitlets.TraitError, match="count must be"):
            panel.count = 0
        panel.folder = ""
        message(panel, action="download", scans=["a"])
        assert "folder" in panel.status["notice"] and panel.status["notice_error"]
    finally:
        panel.close()


def test_a_busy_runner_is_reported():
    runner = FakeRunner()
    runner.start = lambda step, *args: False
    panel = ScanControl(runner)
    try:
        message(panel, action="connect")
        assert panel.status["notice"].startswith("Busy")
    finally:
        panel.close()
