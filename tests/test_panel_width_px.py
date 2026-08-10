import gzip

import numpy as np
import pytest

from quantem.widget import Show2D, Show3D, Show3DSlices, Show4DSTEM


def test_show2d_panel_width_px_wins_over_size():
    data = np.zeros((13, 8, 8), dtype=np.float32)

    widget = Show2D(data, ncols=13, panel_width_px=70, size=999, verbose=False)

    assert widget.size == 70


def test_show2d_ncols_validates_and_roundtrips(tmp_path):
    data = np.zeros((4, 8, 8), dtype=np.float32)

    widget = Show2D(data, ncols=2, verbose=False)

    assert widget.ncols == 2
    assert widget.state_dict()["ncols"] == 2
    out = widget.export_html(tmp_path / "show2d_ncols.html", encoding="full")
    assert '"ncols": 2' in out.read_text()

    with pytest.raises(ValueError, match="ncols"):
        Show2D(data, ncols=0, verbose=False)


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
    assert widget.panel_width_px == 2
    assert widget.source_panel_width == 8


def test_show3d_quantized_offline_uses_per_panel_ranges():
    panels = [
        np.linspace(10, 1000, 2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4),
        np.linspace(0, 100, 2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4),
        np.linspace(-0.3, 0.7, 2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4),
    ]

    widget = Show3D(*panels, display_bin=1, show_controls=False)
    clone = widget._clone_for_html_export(quantized=True)
    assert clone._offline_float_stack == b""
    packed = np.frombuffer(clone._offline_stack, dtype=np.uint8).reshape(2, 4, 12)

    assert clone._offline_mins == [10.0, 0.0, np.float32(-0.3)]
    assert clone._offline_maxs == [1000.0, 100.0, np.float32(0.7)]
    assert [
        (int(packed[:, :, i * 4 : (i + 1) * 4].min()), int(packed[:, :, i * 4 : (i + 1) * 4].max()))
        for i in range(3)
    ] == [(0, 255), (0, 255), (0, 255)]
    clone.close()
    widget.close()


def test_show3dslices_panel_width_px_syncs_to_frontend_state():
    data = np.zeros((4, 8, 8), dtype=np.float32)

    widget = Show3DSlices(data, panel_width_px=70, show_controls=False)

    assert widget.panel_width_px == 70
    assert widget.state_dict()["panel_width_px"] == 70


def test_show3dslices_ui_mode_presets_and_scale_bar_alias():
    data = np.zeros((4, 8, 8), dtype=np.float32)

    presentation = Show3DSlices(data, ui_mode="presentation")
    assert presentation.show_title is True
    assert presentation.show_controls is True
    assert presentation.controls_collapsed is True
    assert presentation.show_stats is False
    assert presentation.show_crosshair is True
    assert presentation.scale_bar_visible is True

    report = Show3DSlices(data, ui_mode="report")
    assert report.show_title is True
    assert report.show_controls is False
    assert report.controls_collapsed is False
    assert report.show_stats is False
    assert report.show_crosshair is True
    assert report.scale_bar_visible is True

    minimal = Show3DSlices(data, ui_mode="minimal")
    assert minimal.show_title is False
    assert minimal.show_controls is False
    assert minimal.controls_collapsed is False
    assert minimal.show_stats is False
    assert minimal.show_crosshair is False
    assert minimal.scale_bar_visible is False

    override = Show3DSlices(
        data,
        ui_mode="minimal",
        show_title=True,
        show_controls=True,
        controls_collapsed=True,
        show_stats=True,
        show_crosshair=True,
        show_scale_bar=True,
    )
    assert override.show_title is True
    assert override.show_controls is True
    assert override.controls_collapsed is True
    assert override.show_stats is True
    assert override.show_crosshair is True
    assert override.scale_bar_visible is True
    assert override.expand_controls() is override
    assert override.controls_collapsed is False
    assert override.collapse_controls() is override
    assert override.controls_collapsed is True
    assert override.toggle_controls() is override
    assert override.controls_collapsed is False

    with pytest.raises(ValueError, match="show_scale_bar"):
        Show3DSlices(data, scale_bar_visible=True, show_scale_bar=False)


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


def test_show4dstem_offline_uint16_preserves_counts():
    data = np.array([[[[0, 255], [256, 4096]]]], dtype=np.uint16)

    widget = Show4DSTEM(
        data,
        offline=True,
        offline_dtype="uint16",
        precompute_virtual_images=False,
        verbose=False,
    )

    packed = np.frombuffer(gzip.decompress(widget._offline_stack), dtype=np.uint16)
    np.testing.assert_array_equal(packed, data.ravel())
