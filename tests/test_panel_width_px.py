import numpy as np

from quantem.widget import Show2D, Show3D, Show3DSlices, Show4DSTEM


def test_show2d_panel_width_px_wins_over_size():
    data = np.zeros((13, 8, 8), dtype=np.float32)

    widget = Show2D(data, ncols=13, panel_width_px=70, size=999, verbose=False)

    assert widget.size == 70


def test_show3d_panel_width_px_sets_display_size_not_source_width():
    panels = [np.full((2, 8, 8), i, dtype=np.float32) for i in range(13)]

    widget = Show3D(
        *panels,
        max_cols=13,
        panel_gap=0,
        panel_width_px=70,
        size=999,
        show_controls=False,
    )

    assert widget.size == 70
    assert widget.max_cols == 13
    assert widget.n_panels == 13
    assert widget.panel_width_px == 8


def test_show3dslices_panel_width_px_syncs_to_frontend_state():
    data = np.zeros((4, 8, 8), dtype=np.float32)

    widget = Show3DSlices(data, panel_width_px=70, show_controls=False)

    assert widget.panel_width_px == 70
    assert widget.state_dict()["panel_width_px"] == 70


def test_show4dstem_panel_width_px_syncs_to_frontend_state():
    data = np.zeros((2, 2, 8, 8), dtype=np.float32)

    widget = Show4DSTEM(
        data,
        panel_width_px=70,
        show_controls=False,
        precompute_virtual_images=False,
        verbose=False,
    )

    assert widget.panel_width_px == 70
    assert widget.state_dict()["panel_width_px"] == 70
