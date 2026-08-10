"""Show4DSTEM to ShowDiffraction handoff."""

import numpy as np
import pytest

from quantem.widget import Show4DSTEM
from quantem.widget.showdiffraction import ShowDiffraction


def _scan(seed=0):
    rng = np.random.default_rng(seed)
    data = rng.poisson(2.0, (3, 4, 32, 32)).astype(np.float32)
    data[:, :, 12:16, 20:24] += 40.0
    return data


def test_handoff_current_and_mean():
    data = _scan()
    w = Show4DSTEM(data, verbose=False)
    w.pos_row, w.pos_col = 1, 2

    current = w.to_showdiffraction(verbose=False)
    assert isinstance(current, ShowDiffraction)
    assert np.allclose(current._displayed_frame(), data[1, 2])

    mean = w.to_showdiffraction(source="mean", verbose=False)
    assert np.allclose(mean._displayed_frame(), data.reshape(-1, 32, 32).mean(axis=0), atol=1e-4)

    with pytest.raises(ValueError):
        w.to_showdiffraction(source="sum")


def test_handoff_transfers_calibration_and_kwargs():
    w = Show4DSTEM(_scan(), verbose=False)
    w.k_pixel_unit = "1/Å"
    w.k_pixel_size = 0.012

    sd = w.to_showdiffraction(dp_scale_mode="linear", verbose=False)
    assert sd.k_pixel_size == pytest.approx(0.012)
    assert sd.k_calibrated
    assert sd.dp_scale_mode == "linear"

    uncal = Show4DSTEM(_scan(), verbose=False).to_showdiffraction(verbose=False)
    assert not uncal.k_calibrated
