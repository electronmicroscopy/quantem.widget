"""Native signed diffraction comparisons from one shared scan selection."""
import numpy as np
import pytest
import torch

from quantem.widget import Show4DSTEM


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native CUDA viewer check")
def test_all_patterns_follow_scan_and_visible_order_without_changing_data():
    data_t = torch.arange(3 * 4 * 4 * 16 * 16, device="cuda", dtype=torch.float32)
    data_t = data_t.reshape(3, 4, 4, 16, 16) - 5000
    reference_t = data_t.clone()
    viewer = Show4DSTEM(data_t, view_mode="multiple", compare_dp_mode="all")
    try:
        for row, col in [(0, 0), (1, 2), (3, 3)]:
            viewer.pos_row, viewer.pos_col = row, col
            actual = np.frombuffer(viewer.compare_diffraction_bytes, dtype=np.float32)
            np.testing.assert_array_equal(actual.reshape(3, 16, 16), data_t[:, row, col].cpu().numpy())
        viewer.compare_panel_indices = [2, 0]
        viewer._update_frame()
        assert viewer.compare_diffraction_indices == [2, 0]
        np.testing.assert_array_equal(
            np.frombuffer(viewer.compare_diffraction_bytes, dtype=np.float32).reshape(2, 16, 16),
            data_t[[2, 0], 3, 3].cpu().numpy(),
        )
        viewer.compare_dp_mode = "selected"
        assert viewer.compare_diffraction_bytes == b""
        viewer.roi_active = True
        viewer.roi_mode = "point"
        viewer.roi_center = [3.0, 4.0]
        viewer._refresh_compare_virtual_images()
        assert "unavailable" not in viewer.compare_status.lower()
        images = np.frombuffer(viewer.compare_virtual_image_bytes, dtype=np.float32)
        expected = data_t[viewer.compare_panel_indices, :, :, 3, 4].cpu().numpy()
        np.testing.assert_array_equal(images.reshape(expected.shape), expected)
        torch.testing.assert_close(data_t, reference_t, rtol=0, atol=0)
    finally:
        viewer.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native CUDA viewer check")
def test_shared_region_mean_keeps_methods_and_signed_data_separate():
    data_t = torch.arange(3 * 4 * 4 * 16 * 16, device="cuda", dtype=torch.float32)
    data_t = data_t.reshape(3, 4, 4, 16, 16) - 5000
    reference_t = data_t.clone()
    viewer = Show4DSTEM(data_t, view_mode="multiple", compare_dp_mode="all")
    try:
        viewer.vi_roi_mode = "circle"
        viewer.vi_roi_radius = 1
        for row, col in [(0, 0), (1, 2), (3, 3)]:
            viewer.vi_roi_center = [float(row), float(col)]
            assert viewer.vi_roi_receipt[:2] == [float(row), float(col)]
            rows_t, cols_t = torch.meshgrid(torch.arange(4, device="cuda"), torch.arange(4, device="cuda"), indexing="ij")
            mask_t = (rows_t - row) ** 2 + (cols_t - col) ** 2 <= 1
            actual = np.frombuffer(viewer.compare_diffraction_bytes, dtype=np.float32).reshape(3, 16, 16)
            np.testing.assert_allclose(actual, data_t[:, mask_t].mean(1).cpu().numpy(), rtol=0, atol=1e-6)
        viewer.vi_roi_mode = "off"
        actual = np.frombuffer(viewer.compare_diffraction_bytes, dtype=np.float32).reshape(3, 16, 16)
        np.testing.assert_array_equal(actual, data_t[:, viewer.pos_row, viewer.pos_col].cpu().numpy())
        torch.testing.assert_close(data_t, reference_t, rtol=0, atol=0)
    finally:
        viewer.close()
