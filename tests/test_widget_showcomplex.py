import json

import numpy as np
import pytest
import torch

from quantem.widget import ShowComplex2D

# =========================================================================
# Basic creation — dtypes and input types
# =========================================================================

def test_showcomplex_complex64():
    """Create widget from complex64 array."""
    data = (np.random.rand(32, 32) + 1j * np.random.rand(32, 32)).astype(np.complex64)
    widget = ShowComplex2D(data)
    assert widget.height == 32
    assert widget.width == 32
    assert len(widget.real_bytes) == 32 * 32 * 4
    assert len(widget.imag_bytes) == 32 * 32 * 4

def test_showcomplex_complex128():
    """Create widget from complex128 array."""
    data = np.random.rand(16, 24) + 1j * np.random.rand(16, 24)
    widget = ShowComplex2D(data)
    assert widget.height == 16
    assert widget.width == 24
    # complex128 → float32, so 4 bytes per element
    assert len(widget.real_bytes) == 16 * 24 * 4
    assert len(widget.imag_bytes) == 16 * 24 * 4

def test_showcomplex_torch_complex64():
    """Create widget from PyTorch complex64 tensor."""
    data = torch.randn(20, 30, dtype=torch.complex64)
    widget = ShowComplex2D(data)
    assert widget.height == 20
    assert widget.width == 30

def test_showcomplex_torch_complex128():
    """Create widget from PyTorch complex128 tensor."""
    data = torch.randn(12, 18, dtype=torch.complex128)
    widget = ShowComplex2D(data)
    assert widget.height == 12
    assert widget.width == 18

def test_showcomplex_tuple_input():
    """Create widget from (real, imag) tuple of numpy arrays."""
    real = np.random.rand(16, 16).astype(np.float32)
    imag = np.random.rand(16, 16).astype(np.float32)
    widget = ShowComplex2D((real, imag))
    assert widget.height == 16
    assert widget.width == 16

def test_showcomplex_tuple_float64():
    """Create widget from (real, imag) tuple of float64 arrays."""
    real = np.random.rand(10, 10)
    imag = np.random.rand(10, 10)
    widget = ShowComplex2D((real, imag))
    assert widget.height == 10
    assert widget.width == 10
    # float64 → float32
    assert len(widget.real_bytes) == 10 * 10 * 4

def test_showcomplex_tuple_torch():
    """Create widget from (real, imag) tuple of torch tensors."""
    real = torch.randn(10, 12)
    imag = torch.randn(10, 12)
    widget = ShowComplex2D((real, imag))
    assert widget.height == 10
    assert widget.width == 12

def test_showcomplex_nonsquare_tall():
    """Tall non-square array works."""
    data = (np.random.rand(64, 16) + 1j * np.random.rand(64, 16)).astype(np.complex64)
    widget = ShowComplex2D(data)
    assert widget.height == 64
    assert widget.width == 16

def test_showcomplex_nonsquare_wide():
    """Wide non-square array works."""
    data = (np.random.rand(8, 128) + 1j * np.random.rand(8, 128)).astype(np.complex64)
    widget = ShowComplex2D(data)
    assert widget.height == 8
    assert widget.width == 128

def test_showcomplex_small_2x2():
    """Minimal 2x2 array works."""
    data = np.array([[1 + 1j, 2 + 0j], [0 + 2j, 1 - 1j]], dtype=np.complex64)
    widget = ShowComplex2D(data)
    assert widget.height == 2
    assert widget.width == 2

def test_showcomplex_small_1x1():
    """1x1 array works."""
    data = np.array([[3 + 4j]], dtype=np.complex64)
    widget = ShowComplex2D(data)
    assert widget.height == 1
    assert widget.width == 1
    assert widget.stats_mean == pytest.approx(5.0)  # amplitude of 3+4j

# =========================================================================
# Input validation — rejection of invalid data
# =========================================================================

def test_showcomplex_rejects_real_float32():
    """Rejects non-complex float32 array."""
    data = np.random.rand(16, 16).astype(np.float32)
    with pytest.raises(ValueError, match="complex"):
        ShowComplex2D(data)

def test_showcomplex_rejects_real_float64():
    """Rejects non-complex float64 array."""
    data = np.random.rand(16, 16)
    with pytest.raises(ValueError, match="complex"):
        ShowComplex2D(data)

def test_showcomplex_rejects_real_int():
    """Rejects integer array."""
    data = np.ones((16, 16), dtype=np.int32)
    with pytest.raises(ValueError, match="complex"):
        ShowComplex2D(data)

def test_showcomplex_rejects_3d():
    """Rejects 3D complex array."""
    data = (np.random.rand(4, 16, 16) + 1j * np.random.rand(4, 16, 16)).astype(np.complex64)
    with pytest.raises(ValueError, match="2D"):
        ShowComplex2D(data)

def test_showcomplex_rejects_1d():
    """Rejects 1D complex array."""
    data = np.random.rand(64).astype(np.complex64)
    with pytest.raises(ValueError, match="2D"):
        ShowComplex2D(data)

def test_showcomplex_tuple_shape_mismatch():
    """Rejects (real, imag) with mismatched shapes."""
    real = np.random.rand(16, 16).astype(np.float32)
    imag = np.random.rand(16, 32).astype(np.float32)
    with pytest.raises(ValueError, match="same shape"):
        ShowComplex2D((real, imag))

def test_showcomplex_tuple_3d_rejects():
    """Rejects (real, imag) tuple of 3D arrays."""
    real = np.random.rand(4, 16, 16).astype(np.float32)
    imag = np.random.rand(4, 16, 16).astype(np.float32)
    with pytest.raises(ValueError, match="2D"):
        ShowComplex2D((real, imag))

# =========================================================================
# Display modes — initialization and switching
# =========================================================================

def test_showcomplex_display_mode_default():
    """Default display mode is amplitude."""
    data = (np.random.rand(8, 8) + 1j * np.random.rand(8, 8)).astype(np.complex64)
    widget = ShowComplex2D(data)
    assert widget.display_mode == "amplitude"

@pytest.mark.parametrize("mode", ["amplitude", "phase", "hsv", "real", "imag"])
def test_showcomplex_all_display_modes(mode):
    """Widget can be created with each display mode."""
    data = (np.random.rand(8, 8) + 1j * np.random.rand(8, 8)).astype(np.complex64)
    widget = ShowComplex2D(data, display_mode=mode)
    assert widget.display_mode == mode

def test_showcomplex_mode_cycle():
    """Switch through all display modes."""
    data = (np.random.rand(8, 8) + 1j * np.random.rand(8, 8)).astype(np.complex64)
    widget = ShowComplex2D(data, display_mode="amplitude")
    for mode in ["phase", "hsv", "real", "imag", "amplitude"]:
        widget.display_mode = mode
        assert widget.display_mode == mode
        # Stats should be updated (no crash)
        assert np.isfinite(widget.stats_mean)

# =========================================================================
# Statistics — correct values per display mode
# =========================================================================

def test_showcomplex_stats_amplitude():
    """Stats for amplitude mode: sqrt(3^2 + 4^2) = 5."""
    real = np.ones((8, 8), dtype=np.float32) * 3.0
    imag = np.ones((8, 8), dtype=np.float32) * 4.0
    widget = ShowComplex2D((real, imag), display_mode="amplitude")
    assert widget.stats_mean == pytest.approx(5.0)
    assert widget.stats_min == pytest.approx(5.0)
    assert widget.stats_max == pytest.approx(5.0)
    assert widget.stats_std == pytest.approx(0.0, abs=1e-5)

def test_showcomplex_stats_phase():
    """Stats for phase mode: atan2(0, 1) = 0."""
    real = np.ones((8, 8), dtype=np.float32)
    imag = np.zeros((8, 8), dtype=np.float32)
    widget = ShowComplex2D((real, imag), display_mode="phase")
    assert widget.stats_mean == pytest.approx(0.0)
    assert widget.stats_min == pytest.approx(0.0)
    assert widget.stats_max == pytest.approx(0.0)

def test_showcomplex_stats_phase_range():
    """Phase values are within [-pi, pi]."""
    data = (np.random.rand(32, 32) + 1j * np.random.rand(32, 32)).astype(np.complex64)
    widget = ShowComplex2D(data, display_mode="phase")
    assert widget.stats_min >= -np.pi - 1e-5
    assert widget.stats_max <= np.pi + 1e-5

def test_showcomplex_stats_amplitude_nonnegative():
    """Amplitude is always non-negative."""
    data = (np.random.rand(32, 32) + 1j * np.random.rand(32, 32)).astype(np.complex64)
    widget = ShowComplex2D(data, display_mode="amplitude")
    assert widget.stats_min >= 0

def test_showcomplex_stats_real():
    """Stats for real mode match input real part."""
    real = np.ones((8, 8), dtype=np.float32) * 7.0
    imag = np.ones((8, 8), dtype=np.float32) * 3.0
    widget = ShowComplex2D((real, imag), display_mode="real")
    assert widget.stats_mean == pytest.approx(7.0)
    assert widget.stats_min == pytest.approx(7.0)
    assert widget.stats_max == pytest.approx(7.0)

def test_showcomplex_stats_imag():
    """Stats for imaginary mode match input imag part."""
    real = np.ones((8, 8), dtype=np.float32) * 7.0
    imag = np.ones((8, 8), dtype=np.float32) * 3.0
    widget = ShowComplex2D((real, imag), display_mode="imag")
    assert widget.stats_mean == pytest.approx(3.0)
    assert widget.stats_min == pytest.approx(3.0)
    assert widget.stats_max == pytest.approx(3.0)

def test_showcomplex_stats_hsv():
    """HSV mode stats are computed on amplitude."""
    real = np.ones((8, 8), dtype=np.float32) * 3.0
    imag = np.ones((8, 8), dtype=np.float32) * 4.0
    widget = ShowComplex2D((real, imag), display_mode="hsv")
    assert widget.stats_mean == pytest.approx(5.0)  # amplitude

def test_showcomplex_stats_update_on_mode_change():
    """Changing display mode updates stats correctly."""
    real = np.ones((8, 8), dtype=np.float32) * 3.0
    imag = np.ones((8, 8), dtype=np.float32) * 4.0
    widget = ShowComplex2D((real, imag), display_mode="amplitude")
    assert widget.stats_mean == pytest.approx(5.0)

    widget.display_mode = "real"
    assert widget.stats_mean == pytest.approx(3.0)

    widget.display_mode = "imag"
    assert widget.stats_mean == pytest.approx(4.0)

    widget.display_mode = "phase"
    expected_phase = np.arctan2(4.0, 3.0)
    assert widget.stats_mean == pytest.approx(expected_phase)

    widget.display_mode = "hsv"
    assert widget.stats_mean == pytest.approx(5.0)  # back to amplitude

def test_showcomplex_stats_with_varied_data():
    """Stats computed correctly with non-uniform data."""
    real = np.array([[1, 2], [3, 4]], dtype=np.float32)
    imag = np.zeros((2, 2), dtype=np.float32)
    widget = ShowComplex2D((real, imag), display_mode="real")
    assert widget.stats_mean == pytest.approx(2.5)
    assert widget.stats_min == pytest.approx(1.0)
    assert widget.stats_max == pytest.approx(4.0)
    assert widget.stats_std == pytest.approx(np.std([1, 2, 3, 4]), abs=1e-5)

# =========================================================================
# Data integrity — bytes match input
# =========================================================================

def test_showcomplex_bytes_match_input():
    """real_bytes and imag_bytes faithfully represent input data."""
    real = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    imag = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    widget = ShowComplex2D((real, imag))
    recovered_real = np.frombuffer(widget.real_bytes, dtype=np.float32).reshape(2, 2)
    recovered_imag = np.frombuffer(widget.imag_bytes, dtype=np.float32).reshape(2, 2)
    np.testing.assert_array_equal(recovered_real, real)
    np.testing.assert_array_equal(recovered_imag, imag)

def test_showcomplex_complex_bytes_match():
    """Bytes from complex input correctly split into real/imag."""
    data = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex64)
    widget = ShowComplex2D(data)
    recovered_real = np.frombuffer(widget.real_bytes, dtype=np.float32).reshape(2, 2)
    recovered_imag = np.frombuffer(widget.imag_bytes, dtype=np.float32).reshape(2, 2)
    np.testing.assert_array_equal(recovered_real, np.array([[1, 3], [5, 7]], dtype=np.float32))
    np.testing.assert_array_equal(recovered_imag, np.array([[2, 4], [6, 8]], dtype=np.float32))

# =========================================================================
# Constructor options
# =========================================================================

def test_showcomplex_options():
    """Constructor options are applied."""
    data = (np.random.rand(8, 8) + 1j * np.random.rand(8, 8)).astype(np.complex64)
    widget = ShowComplex2D(
        data,
        title="Exit Wave",
        cmap="viridis",
        log_scale=True,
        auto_contrast=True,
        pixel_size=1.5,
    )
    assert widget.title == "Exit Wave"
    assert widget.cmap == "viridis"
    assert widget.log_scale is True
    assert widget.auto_contrast is True
    assert widget.pixel_size == pytest.approx(1.5)

def test_showcomplex_default_options():
    """Default options are sensible."""
    data = (np.random.rand(8, 8) + 1j * np.random.rand(8, 8)).astype(np.complex64)
    widget = ShowComplex2D(data)
    assert widget.title == ""
    assert widget.cmap == "inferno"
    assert widget.log_scale is False
    assert widget.auto_contrast is False
    assert widget.pixel_size == 0.0
    assert widget.show_stats is True
    assert widget.show_controls is True
    assert widget.show_fft is False
    assert widget.scale_bar_visible is True
    assert widget.image_width_px == 0
    assert widget.percentile_low == pytest.approx(1.0)
    assert widget.percentile_high == pytest.approx(99.0)

def test_showcomplex_show_controls():
    """show_controls can be toggled."""
    data = (np.random.rand(8, 8) + 1j * np.random.rand(8, 8)).astype(np.complex64)
    widget = ShowComplex2D(data)
    assert widget.show_controls is True
    w2 = ShowComplex2D(data, show_controls=False)
    assert w2.show_controls is False

def test_showcomplex_show_stats():
    """show_stats can be toggled."""
    data = (np.random.rand(8, 8) + 1j * np.random.rand(8, 8)).astype(np.complex64)
    w = ShowComplex2D(data, show_stats=False)
    assert w.show_stats is False

def test_showcomplex_show_fft():
    """show_fft can be toggled."""
    data = (np.random.rand(8, 8) + 1j * np.random.rand(8, 8)).astype(np.complex64)
    w = ShowComplex2D(data, show_fft=True)
    assert w.show_fft is True

def test_showcomplex_scale_bar():
    """scale_bar_visible option."""
    data = (np.random.rand(8, 8) + 1j * np.random.rand(8, 8)).astype(np.complex64)
    w = ShowComplex2D(data, scale_bar_visible=False)
    assert w.scale_bar_visible is False

def test_showcomplex_image_width_px():
    """image_width_px option."""
    data = (np.random.rand(8, 8) + 1j * np.random.rand(8, 8)).astype(np.complex64)
    w = ShowComplex2D(data, image_width_px=800)
    assert w.image_width_px == 800

def test_showcomplex_percentiles():
    """Custom percentile settings."""
    data = (np.random.rand(8, 8) + 1j * np.random.rand(8, 8)).astype(np.complex64)
    w = ShowComplex2D(data, percentile_low=5.0, percentile_high=95.0)
    assert w.percentile_low == pytest.approx(5.0)
    assert w.percentile_high == pytest.approx(95.0)

# =========================================================================
# Edge cases — special data
# =========================================================================

def test_showcomplex_constant_data():
    """Constant complex data doesn't crash."""
    data = np.ones((8, 8), dtype=np.complex64) * (2.0 + 3.0j)
    widget = ShowComplex2D(data)
    assert widget.stats_std == pytest.approx(0.0, abs=1e-5)

def test_showcomplex_pure_real():
    """Pure real (zero imaginary) data works."""
    data = np.ones((8, 8), dtype=np.complex64) * 5.0
    widget = ShowComplex2D(data, display_mode="amplitude")
    assert widget.stats_mean == pytest.approx(5.0)
    widget.display_mode = "imag"
    assert widget.stats_mean == pytest.approx(0.0)

def test_showcomplex_pure_imaginary():
    """Pure imaginary (zero real) data works."""
    data = np.ones((8, 8), dtype=np.complex64) * 5.0j
    widget = ShowComplex2D(data, display_mode="phase")
    assert widget.stats_mean == pytest.approx(np.pi / 2)
    widget.display_mode = "real"
    assert widget.stats_mean == pytest.approx(0.0)

def test_showcomplex_zero_data():
    """All-zero complex data doesn't crash."""
    data = np.zeros((8, 8), dtype=np.complex64)
    widget = ShowComplex2D(data)
    assert widget.stats_mean == pytest.approx(0.0)
    assert widget.stats_min == pytest.approx(0.0)
    assert widget.stats_max == pytest.approx(0.0)
    assert widget.stats_std == pytest.approx(0.0)

def test_showcomplex_negative_values():
    """Complex data with negative real/imag parts."""
    real = np.array([[-3, 2], [1, -4]], dtype=np.float32)
    imag = np.array([[4, -1], [-2, 3]], dtype=np.float32)
    widget = ShowComplex2D((real, imag), display_mode="real")
    assert widget.stats_min == pytest.approx(-4.0)
    assert widget.stats_max == pytest.approx(2.0)

def test_showcomplex_large_values():
    """Large values don't overflow."""
    data = np.ones((8, 8), dtype=np.complex64) * (1e6 + 1e6j)
    widget = ShowComplex2D(data, display_mode="amplitude")
    expected = np.sqrt(2) * 1e6
    assert widget.stats_mean == pytest.approx(expected, rel=1e-4)

def test_showcomplex_very_small_values():
    """Very small values are handled."""
    data = np.ones((8, 8), dtype=np.complex64) * (1e-7 + 1e-7j)
    widget = ShowComplex2D(data, display_mode="amplitude")
    assert widget.stats_mean > 0
    assert np.isfinite(widget.stats_mean)

# =========================================================================
# set_image
# =========================================================================

def test_showcomplex_set_image():
    """set_image replaces data, preserves settings."""
    data = np.random.rand(16, 16) + 1j * np.random.rand(16, 16)
    widget = ShowComplex2D(data, cmap="viridis", display_mode="phase")
    assert widget.height == 16

    new_data = np.random.rand(32, 24) + 1j * np.random.rand(32, 24)
    widget.set_image(new_data)
    assert widget.height == 32
    assert widget.width == 24
    assert widget.cmap == "viridis"
    assert widget.display_mode == "phase"
    assert len(widget.real_bytes) == 32 * 24 * 4

def test_showcomplex_set_image_tuple():
    """set_image with (real, imag) tuple."""
    data = np.random.rand(16, 16) + 1j * np.random.rand(16, 16)
    widget = ShowComplex2D(data)

    real = np.random.rand(20, 20).astype(np.float32)
    imag = np.random.rand(20, 20).astype(np.float32)
    widget.set_image((real, imag))
    assert widget.height == 20
    assert widget.width == 20

def test_showcomplex_set_image_torch():
    """set_image with PyTorch complex tensor."""
    data = np.random.rand(16, 16) + 1j * np.random.rand(16, 16)
    widget = ShowComplex2D(data)
    new_data = torch.randn(24, 32, dtype=torch.complex64)
    widget.set_image(new_data)
    assert widget.height == 24
    assert widget.width == 32

def test_showcomplex_set_image_rejects_real():
    """set_image rejects non-complex input."""
    data = np.random.rand(8, 8) + 1j * np.random.rand(8, 8)
    widget = ShowComplex2D(data)
    with pytest.raises(ValueError, match="complex"):
        widget.set_image(np.random.rand(8, 8).astype(np.float32))

def test_showcomplex_set_image_rejects_3d():
    """set_image rejects 3D input."""
    data = np.random.rand(8, 8) + 1j * np.random.rand(8, 8)
    widget = ShowComplex2D(data)
    with pytest.raises(ValueError, match="2D"):
        widget.set_image(np.random.rand(4, 8, 8) + 1j * np.random.rand(4, 8, 8))

def test_showcomplex_set_image_tuple_mismatch():
    """set_image rejects tuple with mismatched shapes."""
    data = np.random.rand(8, 8) + 1j * np.random.rand(8, 8)
    widget = ShowComplex2D(data)
    with pytest.raises(ValueError, match="same shape"):
        widget.set_image((np.random.rand(8, 8), np.random.rand(8, 16)))

def test_showcomplex_set_image_updates_stats():
    """set_image updates statistics."""
    real1 = np.ones((8, 8), dtype=np.float32) * 3.0
    imag1 = np.ones((8, 8), dtype=np.float32) * 4.0
    widget = ShowComplex2D((real1, imag1), display_mode="amplitude")
    assert widget.stats_mean == pytest.approx(5.0)

    real2 = np.ones((8, 8), dtype=np.float32) * 6.0
    imag2 = np.ones((8, 8), dtype=np.float32) * 8.0
    widget.set_image((real2, imag2))
    assert widget.stats_mean == pytest.approx(10.0)  # sqrt(36+64)

# =========================================================================
# State persistence — save / load / state_dict
# =========================================================================

def test_showcomplex_state_dict_roundtrip():
    """State dict can be saved and restored."""
    data = (np.random.rand(8, 8) + 1j * np.random.rand(8, 8)).astype(np.complex64)
    w1 = ShowComplex2D(data, display_mode="phase", cmap="viridis", log_scale=True, title="Test")
    state = w1.state_dict()
    w2 = ShowComplex2D(data, state=state)
    assert w2.display_mode == "phase"
    assert w2.cmap == "viridis"
    assert w2.log_scale is True
    assert w2.title == "Test"

def test_showcomplex_state_dict_completeness():
    """State dict contains all expected keys."""
    data = (np.random.rand(8, 8) + 1j * np.random.rand(8, 8)).astype(np.complex64)
    widget = ShowComplex2D(data)
    state = widget.state_dict()
    expected_keys = {
        "display_mode", "title", "cmap", "log_scale", "auto_contrast",
        "percentile_low", "percentile_high", "pixel_size",
        "scale_bar_visible", "show_fft", "show_stats", "show_controls",
        "image_width_px", "disabled_tools", "hidden_tools",
        "roi_mode", "roi_center_row", "roi_center_col",
        "roi_radius", "roi_width", "roi_height",
    }
    assert set(state.keys()) == expected_keys

def test_showcomplex_state_dict_all_options():
    """State dict roundtrip preserves all options."""
    data = (np.random.rand(8, 8) + 1j * np.random.rand(8, 8)).astype(np.complex64)
    w1 = ShowComplex2D(
        data,
        display_mode="hsv",
        title="Full Test",
        cmap="plasma",
        log_scale=True,
        auto_contrast=True,
        percentile_low=5.0,
        percentile_high=95.0,
        pixel_size=2.5,
        scale_bar_visible=False,
        show_fft=True,
        show_stats=False,
        show_controls=False,
        image_width_px=600,
    )
    state = w1.state_dict()
    w2 = ShowComplex2D(data, state=state)
    assert w2.display_mode == "hsv"
    assert w2.title == "Full Test"
    assert w2.cmap == "plasma"
    assert w2.log_scale is True
    assert w2.auto_contrast is True
    assert w2.percentile_low == pytest.approx(5.0)
    assert w2.percentile_high == pytest.approx(95.0)
    assert w2.pixel_size == pytest.approx(2.5)
    assert w2.scale_bar_visible is False
    assert w2.show_fft is True
    assert w2.show_stats is False
    assert w2.show_controls is False
    assert w2.image_width_px == 600

def test_showcomplex_save_load_file(tmp_path):
    """State can be saved to and loaded from a JSON file."""
    data = (np.random.rand(8, 8) + 1j * np.random.rand(8, 8)).astype(np.complex64)
    w1 = ShowComplex2D(data, display_mode="hsv", title="Ptycho")
    path = str(tmp_path / "state.json")
    w1.save(path)
    w2 = ShowComplex2D(data, state=path)
    assert w2.display_mode == "hsv"
    assert w2.title == "Ptycho"

def test_showcomplex_save_file_is_valid_json(tmp_path):
    """Saved state file is valid JSON."""
    data = (np.random.rand(8, 8) + 1j * np.random.rand(8, 8)).astype(np.complex64)
    w = ShowComplex2D(data, display_mode="phase", cmap="viridis")
    path = str(tmp_path / "state.json")
    w.save(path)
    with open(path) as f:
        state = json.load(f)
    assert state["metadata_version"] == "1.0"
    assert state["widget_name"] == "ShowComplex2D"
    assert isinstance(state["widget_version"], str)
    assert state["state"]["display_mode"] == "phase"
    assert state["state"]["cmap"] == "viridis"

def test_showcomplex_load_state_partial():
    """Loading partial state dict applies available keys."""
    data = (np.random.rand(8, 8) + 1j * np.random.rand(8, 8)).astype(np.complex64)
    widget = ShowComplex2D(data)
    widget.load_state_dict({"title": "Partial", "cmap": "plasma"})
    assert widget.title == "Partial"
    assert widget.cmap == "plasma"
    assert widget.display_mode == "amplitude"  # unchanged

def test_showcomplex_load_state_ignores_unknown():
    """Loading state dict ignores unknown keys."""
    data = (np.random.rand(8, 8) + 1j * np.random.rand(8, 8)).astype(np.complex64)
    widget = ShowComplex2D(data)
    widget.load_state_dict({"nonexistent_key": 42, "title": "OK"})
    assert widget.title == "OK"

# =========================================================================
# summary and repr
# =========================================================================

def test_showcomplex_summary(capsys):
    """Summary prints key information."""
    data = (np.random.rand(16, 16) + 1j * np.random.rand(16, 16)).astype(np.complex64)
    widget = ShowComplex2D(data, title="Wave")
    widget.summary()
    captured = capsys.readouterr()
    assert "Wave" in captured.out
    assert "16×16" in captured.out
    assert "complex" in captured.out
    assert "amplitude" in captured.out

def test_showcomplex_summary_with_pixel_size(capsys):
    """Summary shows pixel size."""
    data = (np.random.rand(16, 16) + 1j * np.random.rand(16, 16)).astype(np.complex64)
    widget = ShowComplex2D(data, pixel_size=1.5)
    widget.summary()
    captured = capsys.readouterr()
    assert "1.50" in captured.out

def test_showcomplex_summary_nm_conversion(capsys):
    """Summary converts large pixel sizes to nm."""
    data = (np.random.rand(16, 16) + 1j * np.random.rand(16, 16)).astype(np.complex64)
    widget = ShowComplex2D(data, pixel_size=15.0)
    widget.summary()
    captured = capsys.readouterr()
    assert "nm" in captured.out

def test_showcomplex_summary_phase_mode(capsys):
    """Summary shows phase mode."""
    data = (np.random.rand(16, 16) + 1j * np.random.rand(16, 16)).astype(np.complex64)
    widget = ShowComplex2D(data, display_mode="phase")
    widget.summary()
    captured = capsys.readouterr()
    assert "phase" in captured.out
    assert "hsv (cyclic)" in captured.out

def test_showcomplex_summary_log_fft(capsys):
    """Summary shows log scale and FFT."""
    data = (np.random.rand(16, 16) + 1j * np.random.rand(16, 16)).astype(np.complex64)
    widget = ShowComplex2D(data, log_scale=True, show_fft=True)
    widget.summary()
    captured = capsys.readouterr()
    assert "log" in captured.out
    assert "FFT" in captured.out

def test_showcomplex_repr():
    """Repr is compact and informative."""
    data = (np.random.rand(32, 64) + 1j * np.random.rand(32, 64)).astype(np.complex64)
    widget = ShowComplex2D(data, display_mode="hsv")
    r = repr(widget)
    assert "32×64" in r
    assert "hsv" in r

def test_showcomplex_repr_with_pixel_size():
    """Repr shows pixel size."""
    data = (np.random.rand(8, 8) + 1j * np.random.rand(8, 8)).astype(np.complex64)
    w = ShowComplex2D(data, pixel_size=2.0)
    r = repr(w)
    assert "2.00" in r

def test_showcomplex_repr_with_log():
    """Repr shows log scale."""
    data = (np.random.rand(8, 8) + 1j * np.random.rand(8, 8)).astype(np.complex64)
    w = ShowComplex2D(data, log_scale=True)
    r = repr(w)
    assert "log" in r

def test_showcomplex_repr_with_fft():
    """Repr shows FFT."""
    data = (np.random.rand(8, 8) + 1j * np.random.rand(8, 8)).astype(np.complex64)
    w = ShowComplex2D(data, show_fft=True)
    r = repr(w)
    assert "fft" in r

def test_showcomplex_repr_with_title():
    """Repr uses custom title."""
    data = (np.random.rand(8, 8) + 1j * np.random.rand(8, 8)).astype(np.complex64)
    w = ShowComplex2D(data, title="MyWave")
    r = repr(w)
    assert "MyWave" in r

# =========================================================================
# Trait modification after construction
# =========================================================================

def test_showcomplex_modify_cmap():
    """Colormap can be changed after construction."""
    data = (np.random.rand(8, 8) + 1j * np.random.rand(8, 8)).astype(np.complex64)
    w = ShowComplex2D(data, cmap="inferno")
    assert w.cmap == "inferno"
    w.cmap = "viridis"
    assert w.cmap == "viridis"

def test_showcomplex_modify_log_scale():
    """Log scale can be toggled after construction."""
    data = (np.random.rand(8, 8) + 1j * np.random.rand(8, 8)).astype(np.complex64)
    w = ShowComplex2D(data)
    assert w.log_scale is False
    w.log_scale = True
    assert w.log_scale is True

def test_showcomplex_modify_auto_contrast():
    """Auto contrast can be toggled after construction."""
    data = (np.random.rand(8, 8) + 1j * np.random.rand(8, 8)).astype(np.complex64)
    w = ShowComplex2D(data)
    assert w.auto_contrast is False
    w.auto_contrast = True
    assert w.auto_contrast is True

def test_showcomplex_modify_title():
    """Title can be changed after construction."""
    data = (np.random.rand(8, 8) + 1j * np.random.rand(8, 8)).astype(np.complex64)
    w = ShowComplex2D(data, title="Before")
    w.title = "After"
    assert w.title == "After"

def test_showcomplex_modify_show_fft():
    """show_fft can be toggled after construction."""
    data = (np.random.rand(8, 8) + 1j * np.random.rand(8, 8)).astype(np.complex64)
    w = ShowComplex2D(data)
    assert w.show_fft is False
    w.show_fft = True
    assert w.show_fft is True

def test_showcomplex_modify_pixel_size():
    """pixel_size can be changed after construction."""
    data = (np.random.rand(8, 8) + 1j * np.random.rand(8, 8)).astype(np.complex64)
    w = ShowComplex2D(data)
    assert w.pixel_size == 0.0
    w.pixel_size = 2.5
    assert w.pixel_size == pytest.approx(2.5)

# =========================================================================
# Dataset integration — duck typing
# =========================================================================

class MockDataset2d:
    """Duck-typed Dataset2d for ShowComplex2D tests."""
    def __init__(self, array, name="", sampling=(1.0, 1.0), units=("Å", "Å")):
        self.array = array
        self.name = name
        self.sampling = sampling
        self.units = units

def test_showcomplex_dataset_extracts_title():
    """Dataset title is auto-extracted."""
    arr = (np.random.rand(16, 16) + 1j * np.random.rand(16, 16)).astype(np.complex64)
    ds = MockDataset2d(arr, name="Ptycho Object")
    w = ShowComplex2D(ds)
    assert w.title == "Ptycho Object"

def test_showcomplex_dataset_extracts_pixel_size():
    """Dataset pixel size in Å is auto-extracted."""
    arr = (np.random.rand(16, 16) + 1j * np.random.rand(16, 16)).astype(np.complex64)
    ds = MockDataset2d(arr, name="Test", sampling=(1.5, 1.5), units=("Å", "Å"))
    w = ShowComplex2D(ds)
    assert w.pixel_size == pytest.approx(1.5)

def test_showcomplex_dataset_nm_conversion():
    """Dataset pixel size in nm is converted to Å."""
    arr = (np.random.rand(16, 16) + 1j * np.random.rand(16, 16)).astype(np.complex64)
    ds = MockDataset2d(arr, name="Test", sampling=(0.2, 0.2), units=("nm", "nm"))
    w = ShowComplex2D(ds)
    assert w.pixel_size == pytest.approx(2.0)  # 0.2 nm = 2.0 Å

def test_showcomplex_dataset_explicit_override():
    """Explicit parameters override dataset metadata."""
    arr = (np.random.rand(16, 16) + 1j * np.random.rand(16, 16)).astype(np.complex64)
    ds = MockDataset2d(arr, name="Auto Title", sampling=(2.0, 2.0), units=("Å", "Å"))
    w = ShowComplex2D(ds, title="My Title", pixel_size=3.0)
    assert w.title == "My Title"
    assert w.pixel_size == pytest.approx(3.0)

def test_showcomplex_set_image_dataset():
    """set_image accepts dataset object."""
    arr1 = (np.random.rand(16, 16) + 1j * np.random.rand(16, 16)).astype(np.complex64)
    w = ShowComplex2D(arr1)
    arr2 = (np.random.rand(32, 32) + 1j * np.random.rand(32, 32)).astype(np.complex64)
    ds = MockDataset2d(arr2, name="New")
    w.set_image(ds)
    assert w.height == 32
    assert w.width == 32

# =========================================================================
# Tool visibility / locking — disabled_tools and hidden_tools
# =========================================================================

@pytest.mark.parametrize("trait_name", ["disabled_tools", "hidden_tools"])
def test_showcomplex_tool_lists_default_empty(trait_name):
    data = np.random.rand(16, 16) + 1j * np.random.rand(16, 16)
    w = ShowComplex2D(data.astype(np.complex64))
    assert getattr(w, trait_name) == []

@pytest.mark.parametrize(
    ("trait_name", "ctor_kwargs", "expected"),
    [
        ("disabled_tools", {"disabled_tools": ["display", "Histogram"]}, ["display", "histogram"]),
        ("hidden_tools", {"hidden_tools": ["display", "Histogram"]}, ["display", "histogram"]),
        ("disabled_tools", {"disable_display": True, "disable_fft": True}, ["display", "fft"]),
        ("hidden_tools", {"hide_display": True, "hide_fft": True}, ["display", "fft"]),
        ("disabled_tools", {"disable_all": True}, ["all"]),
        ("hidden_tools", {"hide_all": True}, ["all"]),
    ],
)
def test_showcomplex_tool_lists_constructor_behavior(trait_name, ctor_kwargs, expected):
    data = np.random.rand(16, 16) + 1j * np.random.rand(16, 16)
    w = ShowComplex2D(data.astype(np.complex64), **ctor_kwargs)
    assert getattr(w, trait_name) == expected

@pytest.mark.parametrize("kwargs", [{"disabled_tools": ["not_real"]}, {"hidden_tools": ["not_real"]}])
def test_showcomplex_tool_lists_unknown_raises(kwargs):
    data = np.random.rand(16, 16) + 1j * np.random.rand(16, 16)
    with pytest.raises(ValueError, match="Unknown tool group"):
        ShowComplex2D(data.astype(np.complex64), **kwargs)

@pytest.mark.parametrize("trait_name", ["disabled_tools", "hidden_tools"])
def test_showcomplex_tool_lists_normalizes(trait_name):
    data = np.random.rand(16, 16) + 1j * np.random.rand(16, 16)
    w = ShowComplex2D(data.astype(np.complex64))
    setattr(w, trait_name, ["DISPLAY", "display", "fft"])
    assert getattr(w, trait_name) == ["display", "fft"]

# ── save_image ───────────────────────────────────────────────────────────

def test_showcomplex_save_image_amplitude(tmp_path):
    data = (np.random.rand(32, 32) + 1j * np.random.rand(32, 32)).astype(np.complex64)
    w = ShowComplex2D(data, display_mode="amplitude")
    out = w.save_image(tmp_path / "amp.png")
    assert out.exists()
    from PIL import Image
    img = Image.open(out)
    assert img.size == (32, 32)

def test_showcomplex_save_image_phase(tmp_path):
    data = (np.random.rand(32, 32) + 1j * np.random.rand(32, 32)).astype(np.complex64)
    w = ShowComplex2D(data)
    out = w.save_image(tmp_path / "phase.png", display_mode="phase")
    assert out.exists()

def test_showcomplex_save_image_hsv(tmp_path):
    data = (np.random.rand(32, 32) + 1j * np.random.rand(32, 32)).astype(np.complex64)
    w = ShowComplex2D(data)
    out = w.save_image(tmp_path / "hsv.png", display_mode="hsv")
    assert out.exists()

def test_showcomplex_save_image_pdf(tmp_path):
    data = (np.random.rand(16, 16) + 1j * np.random.rand(16, 16)).astype(np.complex64)
    w = ShowComplex2D(data)
    out = w.save_image(tmp_path / "out.pdf")
    assert out.exists()

def test_showcomplex_save_image_bad_format(tmp_path):
    data = (np.random.rand(16, 16) + 1j * np.random.rand(16, 16)).astype(np.complex64)
    w = ShowComplex2D(data)
    with pytest.raises(ValueError, match="Unsupported format"):
        w.save_image(tmp_path / "out.bmp")

def test_showcomplex_widget_version_is_set():
    data = (np.random.rand(8, 8) + 1j * np.random.rand(8, 8)).astype(np.complex64)
    w = ShowComplex2D(data)
    assert w.widget_version != "unknown"

# =========================================================================
# ROI — single-mode pattern
# =========================================================================

def test_showcomplex_roi_defaults():
    """ROI starts off with centered defaults."""
    data = (np.random.rand(64, 128) + 1j * np.random.rand(64, 128)).astype(np.complex64)
    w = ShowComplex2D(data)
    assert w.roi_mode == "off"
    assert w.roi_center_row == pytest.approx(32.0)
    assert w.roi_center_col == pytest.approx(64.0)
    assert w.roi_radius > 0

def test_showcomplex_roi_circle():
    """roi_circle sets mode and position."""
    data = (np.random.rand(32, 32) + 1j * np.random.rand(32, 32)).astype(np.complex64)
    w = ShowComplex2D(data)
    result = w.roi_circle(row=10, col=20, radius=5)
    assert result is w
    assert w.roi_mode == "circle"
    assert w.roi_center_row == pytest.approx(10.0)
    assert w.roi_center_col == pytest.approx(20.0)
    assert w.roi_radius == pytest.approx(5.0)

def test_showcomplex_roi_square():
    """roi_square sets mode and position."""
    data = (np.random.rand(32, 32) + 1j * np.random.rand(32, 32)).astype(np.complex64)
    w = ShowComplex2D(data)
    result = w.roi_square(row=8, col=12, radius=6)
    assert result is w
    assert w.roi_mode == "square"
    assert w.roi_center_row == pytest.approx(8.0)
    assert w.roi_radius == pytest.approx(6.0)

def test_showcomplex_roi_rect():
    """roi_rect sets mode and dimensions."""
    data = (np.random.rand(32, 32) + 1j * np.random.rand(32, 32)).astype(np.complex64)
    w = ShowComplex2D(data)
    result = w.roi_rect(row=16, col=16, width=20, height=10)
    assert result is w
    assert w.roi_mode == "rect"
    assert w.roi_width == pytest.approx(20.0)
    assert w.roi_height == pytest.approx(10.0)

def test_showcomplex_roi_center_compound():
    """roi_center compound trait syncs to row/col."""
    data = (np.random.rand(32, 32) + 1j * np.random.rand(32, 32)).astype(np.complex64)
    w = ShowComplex2D(data)
    w.roi_center = [5.0, 15.0]
    assert w.roi_center_row == pytest.approx(5.0)
    assert w.roi_center_col == pytest.approx(15.0)

def test_showcomplex_roi_state_roundtrip():
    """ROI state survives state_dict roundtrip."""
    data = (np.random.rand(32, 32) + 1j * np.random.rand(32, 32)).astype(np.complex64)
    w1 = ShowComplex2D(data)
    w1.roi_circle(row=10, col=20, radius=7)
    state = w1.state_dict()
    w2 = ShowComplex2D(data, state=state)
    assert w2.roi_mode == "circle"
    assert w2.roi_center_row == pytest.approx(10.0)
    assert w2.roi_center_col == pytest.approx(20.0)
    assert w2.roi_radius == pytest.approx(7.0)

def test_showcomplex_roi_tool_visibility():
    """ROI can be hidden/disabled via tool groups."""
    data = (np.random.rand(16, 16) + 1j * np.random.rand(16, 16)).astype(np.complex64)
    w = ShowComplex2D(data, hide_roi=True)
    assert "roi" in w.hidden_tools
    w2 = ShowComplex2D(data, disable_roi=True)
    assert "roi" in w2.disabled_tools
