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


def test_show3d_quantized_offline_uses_per_panel_ranges():
    panels = [
        np.linspace(10, 1000, 2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4),
        np.linspace(0, 100, 2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4),
        np.linspace(-0.3, 0.7, 2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4),
    ]

    widget = Show3D(*panels, offline=True, show_controls=False)
    packed = np.frombuffer(widget._offline_stack, dtype=np.uint8).reshape(2, 4, 12)

    assert widget._offline_mins == [10.0, 0.0, np.float32(-0.3)]
    assert widget._offline_maxs == [1000.0, 100.0, np.float32(0.7)]
    assert [
        (int(packed[:, :, i * 4 : (i + 1) * 4].min()), int(packed[:, :, i * 4 : (i + 1) * 4].max()))
        for i in range(3)
    ] == [(0, 255), (0, 255), (0, 255)]


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
