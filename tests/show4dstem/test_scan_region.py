"""Use the virtual-image selection as the same scan crop in analysis."""

import numpy as np
import pytest

from quantem.widget import Show4DSTEM


def test_selected_scan_rectangle_matches_viewer_pattern_and_saved_state():
    data = np.arange(8 * 9 * 4 * 4, dtype=np.float32).reshape(8, 9, 4, 4)
    viewer = Show4DSTEM(data, device="cpu", verbose=False)
    restored = Show4DSTEM(data, device="cpu", verbose=False)
    try:
        assert viewer.selected_scan_region() == (0, 8, 0, 9)
        viewer.vi_roi_mode = "rect"
        viewer.vi_roi_center = [2.5, 4.0]
        viewer.vi_roi_height = 3.0
        viewer.vi_roi_width = 4.0
        assert viewer.selected_scan_region() == (1, 5, 2, 7)
        expected = data[1:5, 2:7].mean(axis=(0, 1))
        np.testing.assert_allclose(viewer.vi_roi_pattern(), expected)
        restored.load_state_dict(viewer.state_dict())
        assert restored.selected_scan_region() == (1, 5, 2, 7)
        viewer.vi_roi_center = [0.2, 8.1]
        assert viewer.selected_scan_region() == (0, 2, 7, 9)
        viewer.vi_roi_mode = "circle"
        with pytest.raises(ValueError, match="Select Rect"):
            viewer.selected_scan_region()
        viewer.vi_roi_mode = "off"
        assert viewer.selected_scan_region() == (0, 8, 0, 9)
    finally:
        viewer.close()
        restored.close()


def test_signed_comparison_preserves_patterns_and_detector_means():
    data = np.linspace(-10, 10, 2 * 4 * 5 * 6 * 6, dtype=np.float32).reshape(2, 4, 5, 6, 6)
    viewer = Show4DSTEM(
        data, device="cpu", verbose=False, view_mode="multiple",
        compare_dp_mode="all", bright_field_center_px=(3, 3), bright_field_radius_px=2,
        dp_scale_mode="linear", vi_scale_mode="linear",
        dp_vmin=-10, dp_vmax=10, vi_vmin=-10, vi_vmax=10,
    )
    try:
        frames = np.frombuffer(viewer.compare_diffraction_bytes, dtype=np.float32).reshape(2, 6, 6)
        np.testing.assert_array_equal(frames, data[:, viewer.pos_row, viewer.pos_col])
        rows, cols = np.mgrid[:6, :6]
        mask = (rows - 3) ** 2 + (cols - 3) ** 2 <= 2 ** 2
        images = np.frombuffer(viewer.compare_virtual_image_bytes, dtype=np.float32).reshape(2, 4, 5)
        np.testing.assert_allclose(images, data[..., mask].mean(axis=-1), atol=1e-6)
        assert frames.min() < 0 < frames.max()
        assert images.min() < 0 < images.max()
        assert viewer.dp_vmin == viewer.vi_vmin == -10
    finally:
        viewer.close()
