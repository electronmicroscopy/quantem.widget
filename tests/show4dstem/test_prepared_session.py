"""Prepared sessions keep widget interactions on the source's query interface."""

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
        assert not viewer.ssb_compute_enabled
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
def test_prepared_session_rejects_unavailable_source_operations(kwargs):
    session = DenseSession(torch.zeros((5, 7, 16, 16)))
    session.metadata = {}
    with pytest.raises(ValueError, match="prepared detector session"):
        Show4DSTEM(session, **kwargs)


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
