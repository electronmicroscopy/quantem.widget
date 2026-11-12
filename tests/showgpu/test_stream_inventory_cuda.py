"""Unloading a streamed scan updates the viewer without stale scientific images."""

import numpy as np
import pytest

from quantem.widget import Show4DSTEM, ShowGPU


def test_monitor_selects_and_unloads_completed_scans_from_live_viewer(live_recording):
    cp = pytest.importorskip("cupy")
    recording = live_recording(receiving=False)
    raw = cp.arange(256 * 16 * 16, dtype=cp.uint16).reshape(256, 16, 16)
    for index in range(2):
        recording.add(raw + index, (16, 16), str(index))
    viewer = Show4DSTEM.stream(recording, view_mode="multiple")
    monitor = ShowGPU(running=False, users=False)
    try:
        if viewer._stream_thread is not None:
            viewer._stream_thread.join(5)
        monitor.watch(recording, viewer=viewer)
        report = monitor.snapshot["recording"]
        first, second = report["scans"]
        monitor.handle_message(None, {"action": "select_scan", "scan_id": second["id"],
                                      "watch_id": report["watch_id"]}, [])
        assert viewer.frame_idx == 1
        np.testing.assert_array_equal(viewer.pattern(2, 3), (raw[2 * 16 + 3] + 1).get())
        monitor.handle_message(None, {"action": "unload", "scan_id": first["id"],
                                      "watch_id": report["watch_id"]}, [])
        assert not monitor.action_status.get("error"), monitor.action_status
        assert viewer.n_frames == 1 and viewer.frame_idx == 0
        np.testing.assert_array_equal(viewer.pattern(2, 3), (raw[2 * 16 + 3] + 1).get())
        monitor.handle_message(None, {"action": "unload", "scan_id": second["id"],
                                      "watch_id": report["watch_id"]}, [])
        assert not monitor.action_status.get("error"), monitor.action_status
        assert viewer.n_frames == 0 and not monitor.snapshot["recording"]["scans"]
        assert np.isnan(np.frombuffer(viewer.frame_bytes, np.float32)).all()
        assert np.isnan(np.frombuffer(viewer.virtual_image_bytes, np.float32)).all()
        assert viewer.compare_panel_count == 0 and not viewer.compare_virtual_image_bytes
    finally:
        monitor.close()
        viewer.close()
        recording.release()
