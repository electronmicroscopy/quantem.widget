import numpy as np
import pytest
from scipy import ndimage

from quantem.widget.showdiffraction import (
    align_frames,
    center_phase_correlation,
    center_symmetry,
    pick_center,
    ring_uniformity,
)


def _radius_grid(shape, center):
    rows, cols = np.indices(shape, dtype=np.float64)
    return np.hypot(rows - center[0], cols - center[1])


def _ring_pattern(shape, center, rings, width=2.0, background=True, noise=0.5, seed=0):
    r = _radius_grid(shape, center)
    img = np.zeros(shape, dtype=np.float64)
    for radius, amp in rings:
        img += amp * np.exp(-0.5 * ((r - radius) / width) ** 2)
    if background:
        img += 400.0 / (1.0 + (r / 8.0) ** 1.5)
    if noise:
        rng = np.random.default_rng(seed)
        img += rng.normal(0.0, noise, shape)
    return np.clip(img, 0.0, None)


def test_center_recovery():
    # center_symmetry refines a nearby guess
    truth = (61.3, 66.8)
    frame = _ring_pattern((128, 128), truth, [(20.0, 300.0), (38.0, 150.0)])
    row, col = center_symmetry(frame, guess=(59.0, 69.0))
    assert abs(row - truth[0]) < 0.3
    assert abs(col - truth[1]) < 0.3

    # center_phase_correlation on the same ring pattern
    row, col = center_phase_correlation(frame)
    assert abs(row - truth[0]) < 0.3
    assert abs(col - truth[1]) < 0.3

    # phase correlation far off-center
    truth = (24.7, 98.3)
    frame = _ring_pattern((128, 128), truth, [(15.0, 300.0), (28.0, 150.0)])
    row, col = center_phase_correlation(frame)
    assert abs(row - truth[0]) < 0.3
    assert abs(col - truth[1]) < 0.3

    # phase correlation with a masked beam stop
    truth = (64.2, 62.6)
    frame = _ring_pattern((128, 128), truth, [(22.0, 300.0), (40.0, 150.0)])
    mask = _radius_grid(frame.shape, truth) <= 12.0
    frame = np.where(mask, 0.0, frame)
    row, col = center_phase_correlation(frame, mask=mask)
    assert abs(row - truth[0]) < 0.3
    assert abs(col - truth[1]) < 0.3

    # pick_center auto-selects a method
    truth = (63.0, 66.0)
    frame = _ring_pattern((128, 128), truth, [(20.0, 300.0), (38.0, 150.0)])
    result = pick_center(frame)
    assert set(result) == {"row", "col", "method"}
    assert result["method"] in {"phase_corr", "symmetry"}
    assert abs(result["row"] - truth[0]) < 1.0
    assert abs(result["col"] - truth[1]) < 1.0

    # pick_center honors an explicit method and rejects an unknown one
    truth = (64.0, 64.0)
    frame = _ring_pattern((128, 128), truth, [(30.0, 300.0)])
    for method in ("symmetry", "phase_corr"):
        result = pick_center(frame, method=method)
        assert result["method"] == method
        assert abs(result["row"] - truth[0]) < 1.5
        assert abs(result["col"] - truth[1]) < 1.5
    with pytest.raises(ValueError):
        pick_center(frame, method="bogus")


def test_center_phase_correlation_integer_shift():
    # integer inversion shift: the wrap alternate lands on the frame border,
    # where the symmetry score degenerates, and must not win
    truth = (255.5, 255.5)
    rows, cols = np.indices((511, 511), dtype=np.float64)
    r = np.hypot(rows - truth[0], cols - truth[1])
    frame = 400.0 / (1.0 + (r / 8.0) ** 1.5)
    for radius, amp in [(80.0, 300.0), (150.0, 150.0), (250.0, 200.0)]:
        frame += amp * np.exp(-0.5 * ((r - radius) / 2.0) ** 2)
    # one unmirrored spot
    frame += 50.0 * np.exp(-0.5 * (((rows - 150.0) ** 2 + (cols - 350.0) ** 2) / 10.0**2))
    row, col = center_phase_correlation(frame)
    assert abs(row - truth[0]) < 0.5
    assert abs(col - truth[1]) < 0.5


def test_align_frames():
    # recovers known sub-pixel drifts
    base = _ring_pattern((96, 96), (47.5, 47.5), [(15.0, 300.0), (28.0, 150.0)])
    applied = [(0.0, 0.0), (2.5, -1.5), (-3.25, 2.0), (1.0, 4.5)]
    frames = np.stack([ndimage.shift(base, s, order=3) for s in applied])
    aligned, shifts, used = align_frames(frames)
    assert aligned.shape == frames.shape
    assert all(used)
    for measured, truth in zip(shifts, applied):
        assert abs(measured[0] + truth[0]) < 0.3
        assert abs(measured[1] + truth[1]) < 0.3

    # gates a garbage frame out and keeps the good ones
    rng = np.random.default_rng(1)
    garbage = ndimage.shift(base, (20.0, -14.0), order=1) + rng.normal(0.0, 50.0, base.shape)
    frames = np.stack([base, ndimage.shift(base, (1.5, -2.0), order=3), garbage])
    aligned, shifts, used = align_frames(frames, max_shift=8.0)
    assert used[0] and used[1]
    assert not used[2]
    assert np.allclose(aligned[2], frames[2])
    assert abs(shifts[1][0] + 1.5) < 0.3
    assert abs(shifts[1][1] - 2.0) < 0.3


def test_align_frames_external_reference():
    # Poisson-noisy frames register against a smooth noise-free model
    center = (47.5, 47.5)
    rows, cols = np.indices((96, 96), dtype=np.float64)
    r = np.hypot(rows - center[0], cols - center[1])
    model = 30.0 + 100.0 * np.exp(-0.5 * (r / 2.0) ** 2)
    spots = [(0, 20, 90.0), (0, -20, 90.0), (17, -6, 70.0), (-17, 6, 70.0), (12, 24, 50.0)]
    for d_row, d_col, amp in spots:
        off = (rows - center[0] - d_row) ** 2 + (cols - center[1] - d_col) ** 2
        model += amp * np.exp(-0.5 * off / 2.0**2)
    applied = (1.3, -2.4)
    shifted = np.clip(ndimage.shift(model, applied, order=3), 0.0, None)
    frames = np.stack(
        [np.random.default_rng(seed).poisson(shifted).astype(np.float64) for seed in (0, 1, 8)]
    )
    aligned, shifts, used = align_frames(frames, reference=model)
    assert all(used)
    for measured in shifts:
        assert abs(measured[0] + applied[0]) < 0.5
        assert abs(measured[1] + applied[1]) < 0.5


def test_ring_uniformity():
    # a full clean ring reads high coverage, low spread, high SNR
    center = (63.5, 63.5)
    frame = _ring_pattern((128, 128), center, [(30.0, 300.0)], background=False, noise=0.0)
    qc = ring_uniformity(frame, center, 30.0)
    assert qc["coverage"] > 0.95
    assert qc["cv"] < 0.2
    assert qc["snr"] > 5.0

    # a narrow arc reads low coverage and high spread
    rows, cols = np.indices(frame.shape, dtype=np.float64)
    theta = np.arctan2(rows - center[0], cols - center[1])
    frame = np.where(np.abs(theta) <= np.pi / 6.0, frame, 0.0)
    qc = ring_uniformity(frame, center, 30.0)
    assert qc["coverage"] < 0.4
    assert qc["cv"] > 1.0


def test_ring_uniformity_mask_and_empty_annulus():
    # a dead wedge scores against live sectors only once masked
    center = (63.5, 63.5)
    frame = _ring_pattern((128, 128), center, [(30.0, 300.0)], background=False, noise=0.0)
    rows, cols = np.indices(frame.shape, dtype=np.float64)
    theta = np.degrees(np.arctan2(rows - center[0], cols - center[1])) % 360.0
    wedge = (theta >= 0.0) & (theta <= 90.0)
    dead = np.where(wedge, 0.0, frame)
    unmasked = ring_uniformity(dead, center, 30.0)
    masked = ring_uniformity(dead, center, 30.0, mask=wedge)
    assert masked["coverage"] > unmasked["coverage"]
    assert masked["cv"] < unmasked["cv"]
    assert masked["coverage"] > 0.95

    # an annulus entirely off the detector carries no evidence
    empty = ring_uniformity(frame, center, 500.0)
    assert empty == {"cv": 0.0, "coverage": 0.0, "snr": 0.0}
