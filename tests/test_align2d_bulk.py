import numpy as np
import pytest

from quantem.widget import Align2DBulk


def _fourier_shift(img, dy, dx):
    """Sub-pixel shift via Fourier phase ramp (no scipy needed)."""
    h, w = img.shape
    fy = np.fft.fftfreq(h).reshape(-1, 1)
    fx = np.fft.fftfreq(w).reshape(1, -1)
    phase = np.exp(-2j * np.pi * (fy * dy + fx * dx))
    return np.real(np.fft.ifft2(np.fft.fft2(img) * phase)).astype(np.float32)


# === Basic construction ===

def test_bulk_numpy_3d():
    rng = np.random.default_rng(42)
    stack = rng.random((5, 64, 64)).astype(np.float32)
    w = Align2DBulk(stack, reference=0, max_shift=10)
    assert w.n_images == 5
    assert w.stack.shape[0] == 5
    assert len(w.offsets) == 5
    assert len(w.ncc) == 5

def test_bulk_list_of_2d():
    rng = np.random.default_rng(42)
    frames = [rng.random((32, 32)).astype(np.float32) for _ in range(3)]
    w = Align2DBulk(frames, reference=0)
    assert w.n_images == 3

def test_bulk_reference_offset_is_zero():
    rng = np.random.default_rng(42)
    stack = rng.random((4, 32, 32)).astype(np.float32)
    w = Align2DBulk(stack, reference=0, max_shift=5)
    assert w.offsets[0] == (0.0, 0.0)
    assert w.ncc[0] == pytest.approx(1.0)


# === Integer shift recovery ===

def test_bulk_recovers_integer_shift():
    rng = np.random.default_rng(42)
    ref = rng.random((64, 64)).astype(np.float32)
    shifted = np.roll(ref, (3, -5), axis=(0, 1))
    stack = np.stack([ref, shifted], axis=0)
    w = Align2DBulk(stack, reference=0, max_shift=10, crop=False)
    dx, dy = w.offsets[1]
    # roll(3, -5) -> undo is dx=+5, dy=-3
    assert dx == pytest.approx(5.0, abs=1.5)
    assert dy == pytest.approx(-3.0, abs=1.5)


# === Sub-pixel shift accuracy ===

def test_bulk_subpixel_shift():
    rng = np.random.default_rng(99)
    ref = rng.random((64, 64)).astype(np.float32)
    shifted = _fourier_shift(ref, 2.3, -1.7)
    stack = np.stack([ref, shifted], axis=0)
    w = Align2DBulk(stack, reference=0, max_shift=10, crop=False)
    dx, dy = w.offsets[1]
    assert abs(dx - 1.7) < 0.3, f"dx={dx}, expected ~1.7"
    assert abs(dy - (-2.3)) < 0.3, f"dy={dy}, expected ~-2.3"


# === Crop vs no-crop ===

def test_bulk_crop_shrinks_output():
    rng = np.random.default_rng(42)
    ref = rng.random((64, 64)).astype(np.float32)
    shifted = np.roll(ref, (3, -5), axis=(0, 1))
    stack = np.stack([ref, shifted], axis=0)
    w_crop = Align2DBulk(stack, reference=0, max_shift=10, crop=True)
    w_nocrop = Align2DBulk(stack, reference=0, max_shift=10, crop=False)
    assert w_crop.stack.shape[1] < w_nocrop.stack.shape[1] or w_crop.stack.shape[2] < w_nocrop.stack.shape[2]
    assert w_crop.crop_box is not None
    assert w_nocrop.crop_box is None


# === max_shift clamping ===

def test_bulk_max_shift_clamps():
    rng = np.random.default_rng(42)
    ref = rng.random((64, 64)).astype(np.float32)
    shifted = np.roll(ref, (10, -15), axis=(0, 1))
    stack = np.stack([ref, shifted], axis=0)
    w = Align2DBulk(stack, reference=0, max_shift=3, crop=False)
    dx, dy = w.offsets[1]
    assert abs(dx) <= 3.01
    assert abs(dy) <= 3.01


# === Bin factor ===

def test_bulk_bin_factor():
    rng = np.random.default_rng(42)
    stack = rng.random((3, 64, 64)).astype(np.float32)
    w = Align2DBulk(stack, reference=0, bin=2, crop=False)
    assert w.stack.shape == (3, 32, 32)

def test_bulk_bin_factor_4():
    rng = np.random.default_rng(42)
    stack = rng.random((3, 64, 64)).astype(np.float32)
    w = Align2DBulk(stack, reference=0, bin=4, crop=False)
    assert w.stack.shape == (3, 16, 16)


# === Non-zero reference ===

def test_bulk_nonzero_reference():
    rng = np.random.default_rng(42)
    ref = rng.random((64, 64)).astype(np.float32)
    shifted = np.roll(ref, (3, 5), axis=(0, 1))
    stack = np.stack([shifted, ref, shifted.copy()], axis=0)
    w = Align2DBulk(stack, reference=1, max_shift=10, crop=False)
    assert w.offsets[1] == (0.0, 0.0)
    assert w.ncc[1] == pytest.approx(1.0)
    assert w.reference_idx == 1


# === NCC values ===

def test_bulk_ncc_high_for_identical():
    rng = np.random.default_rng(42)
    ref = rng.random((32, 32)).astype(np.float32)
    stack = np.stack([ref, ref.copy(), ref.copy()], axis=0)
    w = Align2DBulk(stack, reference=0, crop=False)
    for ncc_val in w.ncc:
        assert ncc_val > 0.95


# === Properties ===

def test_bulk_repr():
    rng = np.random.default_rng(42)
    stack = rng.random((3, 32, 32)).astype(np.float32)
    w = Align2DBulk(stack, reference=0, crop=False)
    r = repr(w)
    assert "Align2DBulk" in r
    assert "3 frames" in r

def test_bulk_summary(capsys):
    rng = np.random.default_rng(42)
    stack = rng.random((3, 32, 32)).astype(np.float32)
    w = Align2DBulk(stack, reference=0, title="Test bulk", crop=False)
    w.summary()
    out = capsys.readouterr().out
    assert "Test bulk" in out
    assert "3" in out

def test_bulk_offsets_json_parseable():
    import json
    rng = np.random.default_rng(42)
    stack = rng.random((3, 32, 32)).astype(np.float32)
    w = Align2DBulk(stack, reference=0, crop=False)
    entries = json.loads(w.offsets_json)
    assert len(entries) == 3
    assert "dx" in entries[0]
    assert "dy" in entries[0]
    assert "ncc" in entries[0]

def test_bulk_bytes_not_empty():
    rng = np.random.default_rng(42)
    stack = rng.random((3, 32, 32)).astype(np.float32)
    w = Align2DBulk(stack, reference=0, crop=False)
    assert len(w.ref_bytes) > 0
    assert len(w.frame_bytes) > 0
