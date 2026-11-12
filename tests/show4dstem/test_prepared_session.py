"""Prepared sessions keep widget interactions on the source's query interface."""

import threading
import time

import numpy as np
import pytest
import torch

from quantem.widget import Show4DSTEM
from quantem.widget.show4dstem.dense import DenseSession


def test_loaded_scan_prepares_its_session_without_closing_the_acquisition():
    values = torch.arange(5 * 7 * 16 * 16).reshape(5, 7, 16, 16)

    class LoadedScan:
        def detector_session(self):
            session = DenseSession(values)
            session.metadata = {"scan_sampling_A": [0.4, 0.5]}
            return session

    scan = LoadedScan()
    for _ in range(2):
        viewer = Show4DSTEM(scan, bright_field_center_px=(7.5, 7.5), bright_field_radius_px=3)
        try:
            assert viewer.scan_sampling_A == [0.4, 0.5]
            viewer.apply_preset("adf")
            mask = viewer._current_detector_mask()
            expected = values[..., torch.as_tensor(mask)].sum(-1).float().numpy()
            np.testing.assert_array_equal(viewer.virtual_image(), expected)
        finally:
            viewer.free()
            viewer.close()


def test_prepared_session_controls_and_lifetime():
    values = torch.arange(5 * 7 * 16 * 16).reshape(5, 7, 16, 16)
    session = DenseSession(values)
    session.metadata = {"scan_sampling_A": [0.4, 0.5]}
    viewer = Show4DSTEM(session, bright_field_center_px=(7.5, 7.5), bright_field_radius_px=3)
    try:
        assert viewer.scan_sampling_A == [0.4, 0.5]
        assert not viewer.direct_enabled
        for preset in ("bf", "adf", "haadf"):
            viewer.apply_preset(preset)
            mask = viewer._current_detector_mask()
            expected = values[..., torch.as_tensor(mask)].sum(-1).float().numpy()
            np.testing.assert_array_equal(viewer.virtual_image(), expected)
        viewer.pos_row, viewer.pos_col = 4, 6
        np.testing.assert_array_equal(viewer.pattern(4, 6), values[4, 6].float().numpy())
        viewer.vi_roi_mode = "rect"
        assert len(viewer.vi_roi_dp_bytes) == 16 * 16 * 4
        viewer.free()
        assert session._flat_t is None
        assert values.numel() == 5 * 7 * 16 * 16
    finally:
        viewer.close()


def test_virtual_image_transport_retains_each_query_buffer():
    values = torch.arange(5 * 7 * 16 * 16).reshape(5, 7, 16, 16)
    session = DenseSession(values)
    session.metadata = {}
    queried = []
    masked_sum = session.masked_sum

    def retain_query(mask):
        image = masked_sum(mask)
        queried.append(image)
        return image

    session.masked_sum = retain_query
    viewer = Show4DSTEM(session, bright_field_center_px=(7.5, 7.5), bright_field_radius_px=3)
    try:
        first = viewer.virtual_image_bytes
        frozen = bytes(first)
        assert np.shares_memory(np.frombuffer(first, np.float32), queried[-1])
        viewer.apply_preset("adf")
        assert bytes(first) == frozen
        assert bytes(viewer.virtual_image_bytes) != frozen
        assert np.shares_memory(np.frombuffer(viewer.virtual_image_bytes, np.float32), queried[-1])
        np.testing.assert_array_equal(viewer.virtual_image(), queried[-1])
    finally:
        viewer.free()
        viewer.close()


@pytest.mark.parametrize("kwargs", [{"offline": True}, {"offline": None}, {"scan_region": (0, 2, 0, 2)}])
@pytest.mark.parametrize("series", [False, True])
def test_prepared_session_rejects_unavailable_source_operations(kwargs, series):
    session = DenseSession(torch.zeros((5, 7, 16, 16)))
    session.metadata = {}
    with pytest.raises(ValueError, match="prepared detector session"):
        Show4DSTEM([session] if series else session, **kwargs)


def test_independent_panel_sizes_preserve_data_and_saved_view():
    values = torch.arange(5 * 7 * 16 * 16).reshape(5, 7, 16, 16)
    session = DenseSession(values)
    session.metadata = {}
    viewer = Show4DSTEM(session, dp_panel_width_px=300, vi_panel_width_px=720, sync_resize=False,
                       bright_field_center_px=(7.5, 7.5), bright_field_radius_px=3)
    try:
        before = viewer.virtual_image()
        state = viewer.state_dict()
        viewer.dp_panel_width_px, viewer.vi_panel_width_px, viewer.sync_resize = 450, 450, True
        viewer.load_state_dict(state)
        assert (viewer.dp_panel_width_px, viewer.vi_panel_width_px, viewer.sync_resize) == (300, 720, False)
        assert (viewer.shape_rows, viewer.shape_cols, viewer.det_rows, viewer.det_cols) == (5, 7, 16, 16)
        np.testing.assert_array_equal(viewer.virtual_image(), before)
        np.testing.assert_array_equal(viewer.pattern(4, 6), values[4, 6].float().numpy())
    finally:
        viewer.free()
        viewer.close()


def test_live_session_series_refresh_preserves_detector_and_selection():
    first = torch.full((8, 8, 16, 16), float('nan'))
    first[0] = 7
    second = torch.full_like(first, float('nan'))
    sessions = [DenseSession(first), DenseSession(second)]
    for session in sessions:
        session.metadata = {}
    viewer = Show4DSTEM(sessions[:1], bright_field_center_px=(7.5, 7.5), bright_field_radius_px=3)
    try:
        viewer.apply_preset('adf')
        viewer.pos_row, viewer.pos_col = 0, 3
        before = viewer._current_detector_mask().copy()
        first[1] = 11
        viewer.refresh()
        expected = first[..., torch.as_tensor(before)].sum(-1).float().numpy()
        np.testing.assert_array_equal(viewer.virtual_image(), expected)
        viewer.append(sessions[1:], labels=['second tilt'])
        hidden_image = viewer.virtual_image_bytes
        second[0] = 13
        viewer.refresh()
        assert viewer.virtual_image_bytes is hidden_image
        shown = np.frombuffer(viewer.compare_virtual_image_bytes, np.float32).reshape(2, 8, 8)
        expected = torch.stack((first, second))[..., torch.as_tensor(before)].mean(-1).numpy()
        np.testing.assert_array_equal(shown, expected)
        expected_sum = first[..., torch.as_tensor(before)].sum(-1).float().numpy()
        np.testing.assert_array_equal(viewer.virtual_image(), expected_sum)
        assert viewer.n_frames == 2
        assert (viewer.pos_row, viewer.pos_col) == (0, 3)
        np.testing.assert_array_equal(viewer._current_detector_mask(), before)
        np.testing.assert_array_equal(viewer._session(1).frame(3), second[0, 3].numpy())
    finally:
        viewer.close()


def test_live_average_omits_unacquired_patterns_and_updates_at_the_same_position():
    counts = [torch.full((3, 4, 8, 10), float("nan")) for _ in range(2)]
    counts[0][0, 0] = 6
    sessions = [DenseSession(values) for values in counts]
    for session in sessions:
        session.metadata = {}
    viewer = Show4DSTEM(sessions, view_mode="multiple", compare_dp_mode="average",
                       bright_field_center_px=(3, 4), bright_field_radius_px=2)
    try:
        viewer.pos_row = viewer.pos_col = 0
        np.testing.assert_array_equal(np.frombuffer(viewer.frame_bytes, np.float32), 6)
        counts[1][0, 0] = 10
        viewer.refresh()
        np.testing.assert_array_equal(np.frombuffer(viewer.frame_bytes, np.float32), 8)
        # Missing positions stay missing, while a measured zero participates.
        viewer.pos_col = 1
        assert np.isnan(np.frombuffer(viewer.frame_bytes, np.float32)).all()
        counts[0][0, 1] = 0
        counts[1][0, 1] = 10
        viewer.refresh()
        np.testing.assert_array_equal(np.frombuffer(viewer.frame_bytes, np.float32), 5)
    finally:
        viewer.close()


def test_browser_request_signals_priority_while_refresh_holds_the_lock():
    session = DenseSession(torch.ones((3, 4, 8, 10)))
    session.metadata = {}
    viewer = Show4DSTEM(session, bright_field_center_px=(3, 4), bright_field_radius_px=2)
    request = threading.Thread(target=viewer._handle_msg,
                               args=({"content": {"data": {"method": "request_state"}}},))
    try:
        viewer._last_interaction_at = 0
        with viewer._stream_lock:
            request.start()
            deadline = time.monotonic() + 2
            while not viewer._last_interaction_at and time.monotonic() < deadline:
                time.sleep(0.001)
            assert viewer._last_interaction_at > 0
            assert request.is_alive()  # It still waits for session/state safety.
    finally:
        request.join(timeout=2)
        viewer.close()
    assert not request.is_alive()


@pytest.mark.parametrize("options, widths, synced", [
    ({}, (0, 0, 0), True),
    ({"size": "large"}, (600, 0, 0), True),
    ({"size": "large", "dp_size": "medium", "image_size": "xlarge"}, (600, 450, 900), False),
    ({"dp_size": "small", "image_size": "large", "sync_resize": True}, (0, 300, 600), True),
])
def test_named_sizes_change_only_display(options, widths, synced):
    counts = torch.arange(3 * 4 * 8 * 10).reshape(3, 4, 8, 10)
    session = DenseSession(counts)
    session.metadata = {}
    viewer = Show4DSTEM(session, **options)
    try:
        assert (viewer.panel_width_px, viewer.dp_panel_width_px, viewer.vi_panel_width_px) == widths
        assert viewer.sync_resize is synced
        np.testing.assert_array_equal(viewer.pattern(1, 2), counts[1, 2].float().numpy())
    finally:
        viewer.close()


def test_named_size_rejects_conflicting_pixel_width():
    with pytest.raises(ValueError, match="Use dp_size"):
        Show4DSTEM(None, dp_size="medium", dp_panel_width_px=320)
