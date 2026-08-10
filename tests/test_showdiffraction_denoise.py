"""Detection-level denoise: candidates from a denoised view, measurements from raw data."""

import numpy as np
import pytest
import traitlets

from quantem.widget import library_phase
from quantem.widget.showdiffraction import ShowDiffraction


def _spot_dp(size=128, center=(64, 64), spacing=24.0, amp=40.0, sigma=2.0):
    rows = np.arange(size, dtype=np.float64)[:, None]
    cols = np.arange(size, dtype=np.float64)[None, :]

    def blob(r, c, a, s):
        return a * np.exp(-((rows - r) ** 2 + (cols - c) ** 2) / (2 * s * s))

    dp = blob(center[0], center[1], 300.0, 4.0)
    truth = [
        (center[0], center[1] + spacing),
        (center[0], center[1] - spacing),
        (center[0] + spacing, center[1]),
        (center[0] - spacing, center[1]),
    ]
    for r, c in truth:
        dp = dp + blob(r, c, amp, sigma)
    return dp, truth


def _ring_dp(radii, size=256, amp=30.0, sigma=2.5):
    cen = (size // 2, size // 2)
    rows = np.arange(size, dtype=np.float64)[:, None]
    cols = np.arange(size, dtype=np.float64)[None, :]
    r = np.hypot(rows - cen[0], cols - cen[1])
    dp = 200.0 * np.exp(-(r**2) / (2 * 5.0**2))
    for rr in radii:
        dp = dp + amp * np.exp(-((r - rr) ** 2) / (2 * sigma**2))
    return dp, cen


def _true_spots_found(spots, truth, tol=2.0):
    return sum(
        any(abs(s["row"] - r) <= tol and abs(s["col"] - c) <= tol for r, c in truth)
        for s in spots
    )


def _nearest(widget, r, c):
    return min(np.hypot(s["row"] - r, s["col"] - c) for s in widget.spots)


def test_noisy_spot_detection_improves_with_denoise():
    dp, truth = _spot_dp()
    rng = np.random.default_rng(7)
    counts = rng.poisson(dp * 0.05).astype(np.float32)

    kwargs = dict(center=(64, 64), bf_radius=10, verbose=False)
    raw = ShowDiffraction(counts, detect_denoise="none", **kwargs).detect_spots(max_spots=8)
    auto = ShowDiffraction(counts, detect_denoise="auto", **kwargs).detect_spots(max_spots=8)

    assert len(auto.spots) == 4
    assert _true_spots_found(auto.spots, truth) == 4
    assert len(raw.spots) > len(auto.spots)


def test_denoise_does_not_shift_measured_positions():
    dp, truth = _spot_dp(amp=80.0)
    rng = np.random.default_rng(11)
    counts = rng.poisson(dp * 2.0).astype(np.float32)

    kwargs = dict(center=(64, 64), bf_radius=10, snap_radius=5, verbose=False)
    raw = ShowDiffraction(counts, detect_denoise="none", **kwargs).detect_spots(max_spots=8)
    auto = ShowDiffraction(counts, detect_denoise="anscombe", **kwargs).detect_spots(max_spots=8)

    for widget in (raw, auto):
        assert _true_spots_found(widget.spots, truth) == 4
    for r, c in truth:
        p_raw = _nearest(raw, r, c)
        p_auto = _nearest(auto, r, c)
        assert p_auto <= 1.0
        assert abs(p_auto - p_raw) <= 0.3


def test_noisy_ring_detection_and_raw_fit():
    radii = [50.0, 85.0]
    dp, cen = _ring_dp(radii, amp=12.0)
    rng = np.random.default_rng(3)
    counts = rng.poisson(dp * 0.2).astype(np.float32)

    w = ShowDiffraction(counts, center=cen, bf_radius=15, detect_denoise="auto", verbose=False)
    w.detect_rings(max_rings=4)
    found = sorted(ring["radius_px"] for ring in w.rings)
    for target in radii:
        assert any(abs(f - target) <= 3.0 for f in found)

    w.fit_ring_profile()
    for target in radii:
        assert any(
            r.get("fit_quality") is not None and abs(r["radius_px"] - target) <= 1.5
            for r in w.rings
        )


def test_auto_is_noop_on_clean_data():
    dp, truth = _spot_dp()
    kwargs = dict(center=(64, 64), bf_radius=10, verbose=False)
    raw = ShowDiffraction(dp.astype(np.float32), detect_denoise="none", **kwargs).detect_spots()
    auto = ShowDiffraction(dp.astype(np.float32), detect_denoise="auto", **kwargs).detect_spots()

    assert len(raw.spots) == len(auto.spots)
    for s_raw, s_auto in zip(raw.spots, auto.spots):
        assert s_raw["row"] == s_auto["row"] and s_raw["col"] == s_auto["col"]


def test_gain_scaled_counts_use_anscombe():
    dp, _ = _spot_dp()
    counts = np.random.default_rng(5).poisson(dp * 0.05).astype(np.float32)
    scaled = counts * 1.55e-5

    w = ShowDiffraction(scaled, verbose=False)
    assert w._resolve_detect_denoise(scaled) == "anscombe"


def test_show_detection_view_ships_denoised_frame():
    dp, _ = _spot_dp()
    counts = np.random.default_rng(5).poisson(dp * 0.05).astype(np.float32)

    w = ShowDiffraction(counts, detect_denoise="anscombe", verbose=False)
    raw_bytes = w.frame_bytes
    w.show_detection_view = True
    assert w.frame_bytes != raw_bytes

    assert np.array_equal(w._displayed_frame(), counts)
    w.show_detection_view = False
    assert w.frame_bytes == raw_bytes


def test_display_denoise_is_view_only():
    dp, _ = _spot_dp()
    counts = np.random.default_rng(5).poisson(dp * 0.05).astype(np.float32)

    w = ShowDiffraction(counts, verbose=False)
    raw_bytes = w.frame_bytes
    w.denoise = "nlm"
    assert w.frame_bytes != raw_bytes
    assert np.array_equal(w._displayed_frame(), counts)

    w.denoise = "none"
    assert w.frame_bytes == raw_bytes


def test_noise_sigma_floor_is_tunable():
    rows = np.arange(160, dtype=np.float64)[:, None]
    cols = np.arange(160, dtype=np.float64)[None, :]
    r = np.hypot(rows - 80, cols - 80)

    # Wide dim spots on a diffuse cloud
    dp = 80.0 * np.exp(-(r**2) / (2 * 20.0**2))
    truth = [(80, 125), (35, 80), (125, 80), (80, 35)]
    for rr, cc in truth:
        dp = dp + 6.0 * np.exp(-((rows - rr) ** 2 + (cols - cc) ** 2) / (2 * 4.0**2))
    counts = np.random.default_rng(9).poisson(dp).astype(np.float32)

    w = ShowDiffraction(counts, center=(80, 80), bf_radius=10, detect_denoise="anscombe", verbose=False)
    w.detect_spots(min_distance=10, min_relative=0.3)
    assert len(w.spots) == 0

    w.detect_spots(min_distance=10, min_relative=0.3, noise_sigma=3.0)
    assert _true_spots_found(w.spots, truth, tol=3.0) >= 2


def test_detected_au_ratios_index_as_fcc():
    au = library_phase("Au")
    k = 0.008
    hkls = ("111", "200", "220", "311")
    radii = [1.0 / (au.d_spacing(tuple(int(c) for c in h)) * k) for h in hkls]
    dp, cen = _ring_dp(radii, amp=25.0, sigma=2.0)
    counts = np.random.default_rng(6).poisson(dp * 0.5).astype(np.float32)

    w = ShowDiffraction(counts, center=cen, bf_radius=15, verbose=False)
    w.detect_rings(max_rings=4, exclude_radius=30)
    w.calibrate_from_phase(au)
    w.index_rings(au)

    ordered = sorted(w.rings, key=lambda r: r["radius_px"])
    assert [r["hkl"] for r in ordered] == list(hkls)
    assert all(r["d_error"] < 0.01 for r in ordered)
    assert abs(w.k_pixel_size - k) / k < 0.01


def test_recover_predicted_rings_rescues_missed_reflection():
    au = library_phase("Au")
    k = 0.008
    hkls = ("111", "200", "220", "311")
    radii = [1.0 / (au.d_spacing(tuple(int(c) for c in h)) * k) for h in hkls]

    size = 256
    cen = (size // 2, size // 2)
    rows = np.arange(size, dtype=np.float64)[:, None]
    cols = np.arange(size, dtype=np.float64)[None, :]
    r = np.hypot(rows - cen[0], cols - cen[1])
    dp = 200.0 * np.exp(-(r**2) / (2 * 5.0**2))
    for rr, amp in zip(radii, (25.0, 25.0, 4.0, 25.0)):
        dp = dp + amp * np.exp(-((r - rr) ** 2) / (2 * 2.0**2))
    counts = np.random.default_rng(6).poisson(dp * 0.5).astype(np.float32)

    w = ShowDiffraction(counts, center=cen, bf_radius=15, verbose=False)
    w.detect_rings(max_rings=3, exclude_radius=30)
    assert len(w.rings) == 3

    w.calibrate_from_phase(au)
    added = w.recover_predicted_rings(au)
    assert len(added) == 1 and abs(added[0] - radii[1]) <= 2.0

    w.index_rings(au)
    assert [x["hkl"] for x in sorted(w.rings, key=lambda y: y["radius_px"])] == list(hkls)


def test_detect_denoise_state_and_validation():
    dp, _ = _spot_dp()
    w = ShowDiffraction(dp.astype(np.float32), detect_denoise="gaussian", verbose=False)
    assert w.state_dict()["detect_denoise"] == "gaussian"

    restored = ShowDiffraction(dp.astype(np.float32), verbose=False, state=w.state_dict())
    assert restored.detect_denoise == "gaussian"

    with pytest.raises(traitlets.TraitError):
        ShowDiffraction(dp.astype(np.float32), detect_denoise="median", verbose=False)
