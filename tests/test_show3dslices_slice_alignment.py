import numpy as np
import pytest
import traitlets

from quantem.widget import Show3DSlices


def _texture(rows: int = 64, cols: int = 64) -> np.ndarray:
    row, col = np.mgrid[:rows, :cols].astype(np.float32)
    image = (
        np.sin(row / 3.1)
        + 0.7 * np.cos(col / 4.7)
        + 0.35 * np.sin((row + col) / 5.3)
    )
    image += 4.0 * np.exp(-((row - rows * 0.35) ** 2 + (col - cols * 0.42) ** 2) / 80.0)
    image += 2.5 * np.exp(-((row - rows * 0.68) ** 2 + (col - cols * 0.73) ** 2) / 55.0)
    return image.astype(np.float32)


def _shifted_stack(row_drift: int = 1, col_drift: int = -2, slices: int = 7) -> np.ndarray:
    base = _texture()
    frames = [
        np.roll(base, shift=(idx * row_drift, idx * col_drift), axis=(0, 1))
        for idx in range(slices)
    ]
    return np.stack(frames, axis=0).astype(np.float32)


def _fractionally_shifted_stack(
    row_drift: float = 0.75,
    col_drift: float = -0.4,
    slices: int = 7,
) -> np.ndarray:
    base = _texture()
    row_freq = np.fft.fftfreq(base.shape[0])[:, None]
    col_freq = np.fft.fftfreq(base.shape[1])[None, :]
    spectrum = np.fft.fft2(base)
    frames = []
    for idx in range(slices):
        phase = np.exp(
            -2j
            * np.pi
            * (row_freq * idx * row_drift + col_freq * idx * col_drift)
        )
        frames.append(np.fft.ifft2(spectrum * phase).real.astype(np.float32))
    return np.stack(frames, axis=0)


def test_show3dslices_slice_alignment_state_roundtrip():
    data = np.zeros((4, 8, 8), dtype=np.float32)
    widget = Show3DSlices(
        data,
        slice_alignment="manual",
        row_shift_px_per_slice=1.25,
        col_shift_px_per_slice=-0.5,
        show_controls=False,
    )

    state = widget.state_dict()
    assert state["slice_alignment"] == "manual"
    assert state["row_shift_px_per_slice"] == pytest.approx(1.25)
    assert state["col_shift_px_per_slice"] == pytest.approx(-0.5)
    assert state["slice_alignment_cached"] is True

    restored = Show3DSlices(data, state=state, show_controls=False)
    assert restored.slice_alignment == "manual"
    assert restored.row_shift_px_per_slice == pytest.approx(1.25)
    assert restored.col_shift_px_per_slice == pytest.approx(-0.5)
    assert restored.slice_alignment_cached is True


def test_show3dslices_slice_alignment_validates_mode_and_slopes():
    data = np.zeros((4, 8, 8), dtype=np.float32)

    with pytest.raises(traitlets.TraitError, match="slice_alignment"):
        Show3DSlices(data, slice_alignment="physical_tilt", show_controls=False)

    with pytest.raises(traitlets.TraitError, match="row_shift_px_per_slice"):
        Show3DSlices(data, row_shift_px_per_slice=float("nan"), show_controls=False)


def test_show3dslices_estimates_global_shift_to_apply():
    stack = _shifted_stack(row_drift=1, col_drift=-2, slices=7)
    widget = Show3DSlices(stack, show_controls=False)

    result = widget.estimate_slice_alignment(apply=True)

    assert widget.slice_alignment == "auto"
    assert widget.slice_alignment_cached is True
    # The stack content drifts row +1, col -2 per deeper slice. The display
    # correction is the opposite shift to apply to deeper slices.
    assert widget.row_shift_px_per_slice == pytest.approx(-1.0, abs=0.2)
    assert widget.col_shift_px_per_slice == pytest.approx(2.0, abs=0.4)
    assert result["fit_r2"]["row"] > 0.9
    assert result["fit_r2"]["col"] > 0.9
    assert len(result["adjacent_shift_apply_px"]) == stack.shape[0] - 1


def test_show3dslices_estimates_fractional_global_shift():
    stack = _fractionally_shifted_stack(row_drift=0.75, col_drift=-0.4, slices=7)
    widget = Show3DSlices(stack, show_controls=False)

    result = widget.estimate_slice_alignment(apply=True)

    assert result["row_shift_px_per_slice"] == pytest.approx(-0.75, abs=0.08)
    assert result["col_shift_px_per_slice"] == pytest.approx(0.4, abs=0.08)


def test_show3dslices_reset_slice_alignment():
    data = np.zeros((4, 8, 8), dtype=np.float32)
    widget = Show3DSlices(
        data,
        slice_alignment="manual",
        row_shift_px_per_slice=1.0,
        col_shift_px_per_slice=2.0,
        show_controls=False,
    )

    assert widget.reset_slice_alignment() is widget
    assert widget.slice_alignment == "off"
    assert widget.row_shift_px_per_slice == pytest.approx(0.0)
    assert widget.col_shift_px_per_slice == pytest.approx(0.0)
    assert widget.slice_alignment_cached is False


def test_show3dslices_reuses_cached_alignment_estimate(monkeypatch):
    stack = _shifted_stack(row_drift=1, col_drift=-1, slices=6)
    widget = Show3DSlices(stack, show_controls=False)
    first = widget.estimate_slice_alignment(apply=True)

    def fail_if_recomputed(*args, **kwargs):
        raise AssertionError("cached alignment should not run registration again")

    monkeypatch.setattr(
        "quantem.widget.show3dslices._estimate_adjacent_shift_apply",
        fail_if_recomputed,
    )
    widget.slice_alignment = "off"
    second = widget.estimate_slice_alignment(apply=True)

    assert second == first
    assert widget.slice_alignment == "auto"
    assert widget.slice_alignment_cached is True
    assert widget.slice_alignment_status.startswith("Cached row")


def test_show3dslices_alignment_is_display_only_copy():
    stack = _shifted_stack(row_drift=1, col_drift=0, slices=5)
    widget = Show3DSlices(
        stack,
        slice_alignment="manual",
        row_shift_px_per_slice=-1.0,
        show_controls=False,
    )

    aligned = widget._active_display_data()

    assert aligned is not None
    assert not np.shares_memory(aligned, widget._data[0])
    np.testing.assert_array_equal(widget._data[0], stack)
    assert aligned.shape == stack.shape


def test_show3dslices_frontend_alignment_request_estimates_and_clears():
    stack = _shifted_stack(row_drift=1, col_drift=-1, slices=6)
    widget = Show3DSlices(stack, show_controls=False)

    widget._slice_alignment_request = '{"mode": "estimate", "panel": 0, "id": "test"}'

    assert widget._slice_alignment_request == ""
    assert widget.slice_alignment == "auto"
    assert widget.row_shift_px_per_slice == pytest.approx(-1.0, abs=0.3)
    assert widget.col_shift_px_per_slice == pytest.approx(1.0, abs=0.3)
    assert widget.slice_alignment_cached is True
    assert "Aligned row" in widget.slice_alignment_status
