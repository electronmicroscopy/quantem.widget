"""Tests for the shared Bragg-disk detection module (quantem.widget.bragg).

These exercise the pure functions in isolation:
  * detect_bragg_disks_single — finds known Gaussian spots within 1.5 px
  * vacuum_probe_kernel — Fourier phase-ramp shift moves a sub-pixel center

The pipeline matches py4DSTEM's `_find_Bragg_disks_single` (GPL, reference
only — nothing imported or copied here).
"""

import numpy as np
import pytest

from quantem.widget.bragg import (
    build_soft_disk_probe,
    detect_bragg_disks_single,
    vacuum_probe_kernel,
)


def _make_dp_with_peaks(det, peak_positions, sigma=2.0):
    yy, xx = np.meshgrid(np.arange(det), np.arange(det), indexing="ij")
    dp = np.zeros((det, det), dtype=np.float32)
    for r, c in peak_positions:
        dp += np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2.0 * sigma ** 2))
    return dp


def _origin_probe(det, sigma=2.0):
    """Origin-aligned Gaussian probe (peak at (0,0), wrapped corners)."""
    yy, xx = np.meshgrid(np.arange(det), np.arange(det), indexing="ij")
    return (
        np.exp(-((yy) ** 2 + (xx) ** 2) / (2.0 * sigma ** 2))
        + np.exp(-((yy - det) ** 2 + (xx) ** 2) / (2.0 * sigma ** 2))
        + np.exp(-((yy) ** 2 + (xx - det) ** 2) / (2.0 * sigma ** 2))
        + np.exp(-((yy - det) ** 2 + (xx - det) ** 2) / (2.0 * sigma ** 2))
    ).astype(np.float32)


def _match_peaks(found, expected, tol):
    for (er, ec) in expected:
        d = np.sqrt((found[:, 0] - er) ** 2 + (found[:, 1] - ec) ** 2)
        i = int(np.argmin(d))
        assert d[i] <= tol, (
            f"expected peak ({er}, {ec}) not within {tol}px; closest "
            f"({found[i, 0]:.2f}, {found[i, 1]:.2f}) d={d[i]:.2f}"
        )


def test_detect_three_known_gaussian_spots():
    det = 64
    peaks = [(20, 25), (40, 30), (28, 50)]
    dp = _make_dp_with_peaks(det, peaks)
    probe = _origin_probe(det)
    found = detect_bragg_disks_single(dp, probe, subpixel="multicorr", upsample_factor=4)
    assert found.shape[1] == 3
    assert found.shape[0] >= 3
    _match_peaks(found, peaks, tol=1.5)


@pytest.mark.parametrize("subpixel", ["pixel", "poly", "multicorr"])
def test_subpixel_modes_find_peaks(subpixel):
    det = 64
    peaks = [(20, 25), (40, 30), (28, 50)]
    dp = _make_dp_with_peaks(det, peaks)
    probe = _origin_probe(det)
    found = detect_bragg_disks_single(dp, probe, subpixel=subpixel, upsample_factor=4)
    assert found.shape[0] >= 3
    _match_peaks(found, peaks, tol=1.5)


def test_invalid_subpixel_raises():
    dp = np.zeros((16, 16), dtype=np.float32)
    probe = np.zeros((16, 16), dtype=np.float32)
    with pytest.raises(ValueError):
        detect_bragg_disks_single(dp, probe, subpixel="bogus")


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        detect_bragg_disks_single(
            np.zeros((16, 16), np.float32), np.zeros((8, 8), np.float32)
        )


def test_max_num_peaks_caps_output():
    det = 64
    peaks = [(8 + 10 * i, 8 + 10 * j) for i in range(5) for j in range(5)]
    dp = _make_dp_with_peaks(det, peaks)
    probe = _origin_probe(det)
    found = detect_bragg_disks_single(
        dp, probe, max_num_peaks=7, min_peak_spacing=5.0
    )
    assert found.shape[0] <= 7


def test_min_rel_threshold_removes_weak_peaks():
    det = 64
    yy, xx = np.meshgrid(np.arange(det), np.arange(det), indexing="ij")
    dp = (
        10.0 * np.exp(-((yy - 20) ** 2 + (xx - 25) ** 2) / 8.0)
        + 0.05 * np.exp(-((yy - 40) ** 2 + (xx - 30) ** 2) / 8.0)
        + 0.05 * np.exp(-((yy - 28) ** 2 + (xx - 50) ** 2) / 8.0)
    ).astype(np.float32)
    probe = _origin_probe(det)
    low = detect_bragg_disks_single(dp, probe, min_relative_intensity=0.0)
    high = detect_bragg_disks_single(dp, probe, min_relative_intensity=0.5)
    assert low.shape[0] >= 3
    assert high.shape[0] < low.shape[0]
    assert high.shape[0] >= 1


def test_empty_dp_returns_empty():
    det = 32
    found = detect_bragg_disks_single(
        np.zeros((det, det), np.float32), np.zeros((det, det), np.float32)
    )
    assert found.shape == (0, 3)


# ---------------------------------------------------------------------------
# vacuum_probe_kernel — Fourier phase-ramp shift
# ---------------------------------------------------------------------------

def test_kernel_integer_center_moves_peak_to_origin():
    det = 64
    cr, cc = 31, 28  # integer center
    probe = build_soft_disk_probe(det, det, cr, cc, radius=4.0)
    kernel = vacuum_probe_kernel(probe, cr, cc)
    # The disk center moves to the FFT origin. The flat-topped disk has a
    # plateau of equal maxima, so argmax ties break differently per platform;
    # assert the value at the origin instead of which tied pixel wins.
    kmax = float(kernel.max())
    assert kernel[0, 0] == pytest.approx(kmax, rel=1e-4)
    # A sign error would shift the disk to +center, not the origin: the old
    # center location must now be far outside the disk (~0).
    assert abs(float(kernel[cr, cc])) < 0.05 * kmax


def test_kernel_subpixel_center_no_quantize():
    """A non-integer center must NOT quantize to the nearest pixel.

    Detect spots placed at known sub-pixel offsets from a non-integer probe
    center; the phase-ramp shift should recover them precisely (< 0.4 px). A
    plain np.roll would bias every peak by up to half a pixel.
    """
    det = 64
    cr, cc = 31.7, 32.4
    yy, xx = np.mgrid[:det, :det].astype(np.float32)
    truth = [(cr + 12.3, cc - 8.6), (cr - 9.1, cc + 5.5), (cr + 4.4, cc + 14.2)]
    dp = np.zeros((det, det), dtype=np.float32)
    for ty, tx in truth:
        dp += np.exp(-((yy - ty) ** 2 + (xx - tx) ** 2) / (2.0 * 3.0 ** 2))

    probe = build_soft_disk_probe(det, det, cr, cc, radius=5.0)
    kernel = vacuum_probe_kernel(probe, cr, cc)
    found = detect_bragg_disks_single(
        dp, kernel, subpixel="multicorr", upsample_factor=16,
        min_peak_spacing=4.0, max_num_peaks=10,
    )
    assert found.shape[0] >= 3
    for ty, tx in truth:
        nearest = float(np.hypot(found[:, 0] - ty, found[:, 1] - tx).min())
        assert nearest < 0.4, (
            f"peak ({ty:.2f},{tx:.2f}) recovered to {nearest:.3f} px — "
            "sub-pixel center appears quantized"
        )


def test_kernel_subpixel_differs_from_roll():
    """The phase-ramp kernel for a non-integer center must differ from a
    nearest-pixel np.roll shift (proves it is not quantizing)."""
    det = 48
    cr, cc = 23.6, 24.3
    probe = build_soft_disk_probe(det, det, cr, cc, radius=4.0)
    kernel = vacuum_probe_kernel(probe, cr, cc)
    rolled = np.roll(np.roll(probe, -int(round(cr)), axis=0), -int(round(cc)), axis=1)
    assert not np.allclose(kernel, rolled, atol=1e-2)


def test_build_soft_disk_probe_shape_and_values():
    probe = build_soft_disk_probe(32, 40, 16.0, 20.0, radius=5.0)
    assert probe.shape == (32, 40)
    assert probe.dtype == np.float32
    assert probe[16, 20] == pytest.approx(1.0)
    assert probe[0, 0] == pytest.approx(0.0)
