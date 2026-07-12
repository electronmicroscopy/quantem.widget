"""Frequency Filter family API tests for issue #154."""

import numpy as np
import pytest

from quantem.widget import Show2D, Show3D


def test_show2d_frequency_filter_is_view_only():
    # C1: active high-pass at construction, expect raw data/stats unchanged.
    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    widget = Show2D(
        data,
        frequency_filter="high-pass",
        frequency_filter_cutoff=0.2,
        verbose=False,
    )

    assert widget.frequency_filter == "highpass"
    assert widget.frequency_filter_enabled is True
    np.testing.assert_array_equal(widget._data[0], data)
    assert widget.stats_mean[0] == pytest.approx(float(data.mean()))


def test_show2d_frequency_filter_master_preserves_settings():
    # C2: master is off, expect configured band retained for exact re-enable.
    widget = Show2D(
        np.ones((8, 8), dtype=np.float32),
        frequency_filter="bandpass",
        frequency_filter_enabled=False,
        frequency_filter_center=0.4,
        frequency_filter_width=0.1,
        verbose=False,
    )

    assert widget.frequency_filter_enabled is False
    assert widget.frequency_filter == "bandpass"
    assert widget.frequency_filter_center == pytest.approx(0.4)
    assert widget.frequency_filter_width == pytest.approx(0.1)


def test_show2d_frequency_filter_accepts_per_panel_settings():
    # C3: A/B gallery, expect each panel to keep an independent scientific goal.
    data = [
        np.ones((8, 8), dtype=np.float32),
        np.eye(8, dtype=np.float32),
    ]
    widget = Show2D(
        data,
        frequency_filter=["highpass", "bandpass"],
        frequency_filter_cutoff=[0.08, 0.2],
        frequency_filter_center=[0.3, 0.45],
        frequency_filter_width=[0.1, 0.06],
        verbose=False,
    )

    assert widget.frequency_filter_scope == "panel"
    assert widget.frequency_filter_modes == ["highpass", "bandpass"]
    assert widget.frequency_filter_cutoffs == pytest.approx([0.08, 0.2])
    assert widget.frequency_filter_centers == pytest.approx([0.3, 0.45])
    assert widget.frequency_filter_widths == pytest.approx([0.1, 0.06])


def test_show3d_frequency_filter_state_round_trip():
    # C4: stack filter state, expect saved state keeps the exact view settings.
    data = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
    widget = Show3D(
        data,
        frequency_filter="low-pass",
        frequency_filter_cutoff=0.35,
        show_frequency_filter=True,
        offline=False,
        verbose=False,
    )

    state = widget.state_dict()
    assert state["frequency_filter"] == "lowpass"
    assert state["frequency_filter_enabled"] is True
    assert state["frequency_filter_cutoff"] == pytest.approx(0.35)
    assert state["show_frequency_filter"] is True
    np.testing.assert_array_equal(widget._data, data)


@pytest.mark.parametrize("viewer", [Show2D, Show3D])
def test_frequency_filter_rejects_invalid_normalized_cutoff(viewer):
    # C5: cutoff outside normalized Nyquist, expect a corrective error.
    data = np.ones((8, 8), dtype=np.float32) if viewer is Show2D else np.ones((3, 8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="between 0 and 1"):
        viewer(data, frequency_filter="highpass", frequency_filter_cutoff=1.2, verbose=False)
