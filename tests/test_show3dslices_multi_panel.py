import numpy as np
import pytest

from quantem.widget import Show3DSlices


def test_show3dslices_accepts_4d_panel_stack():
    panels = np.stack(
        [
            np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4),
            np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) + 100,
            np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) - 50,
        ],
        axis=0,
    )

    widget = Show3DSlices(
        panels,
        panel_titles=["untilted", "tilted", "corrected"],
        show_stats=True,
        show_controls=False,
    )

    assert widget.panel_count == 3
    assert widget.panel_titles == ["untilted", "tilted", "corrected"]
    assert widget._data.shape == (3, 2, 3, 4)
    assert widget.nz == 2
    assert widget.ny == 3
    assert widget.nx == 4
    assert len(widget.volume_bytes) == panels.nbytes
    assert widget.state_dict()["active_panel"] == 0
    assert widget.state_dict()["panel_titles"] == ["untilted", "tilted", "corrected"]

    widget.active_panel = 1
    widget.slice_z = 0
    widget._compute_stats()
    assert widget.stats_min[0] == pytest.approx(100.0)


def test_show3dslices_accepts_data_b_shortcut():
    a = np.zeros((2, 3, 4), dtype=np.float32)
    b = np.ones((2, 3, 4), dtype=np.float32)

    widget = Show3DSlices(a, data_b=b, title="raw", title_b="corrected", show_controls=False)

    assert widget.panel_count == 2
    assert widget.panel_titles == ["raw", "corrected"]
    np.testing.assert_array_equal(widget._data[0], a)
    np.testing.assert_array_equal(widget._data[1], b)


def test_show3dslices_rejects_bad_panel_titles():
    data = np.zeros((2, 2, 3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="panel_titles"):
        Show3DSlices(data, panel_titles=["only one"], show_controls=False)


def test_show3dslices_multi_panel_export_roundtrips_state(tmp_path):
    data = np.arange(2 * 2 * 3 * 4, dtype=np.float32).reshape(2, 2, 3, 4)
    widget = Show3DSlices(
        data,
        title="multi panel slices",
        panel_titles=["tilted", "corrected"],
        show_controls=False,
    )
    widget.active_panel = 1

    out = widget.export_html(tmp_path / "show3dslices_multi.html", encoding="uint8")
    html = out.read_text()

    assert "multi panel slices" in html
    assert "panel_titles" in html
    assert "corrected" in html
    assert "active_panel" in html
