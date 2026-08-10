import json

import numpy as np
import pytest

from quantem.widget.show2d import Show2D
from quantem.widget.show3d import Show3D
from quantem.widget.show1d import Show1D


def test_show2d_to_show3d_uses_visible_panels_as_frames():
    data = np.stack(
        [
            np.arange(20, dtype=np.float32).reshape(4, 5),
            np.arange(20, dtype=np.float32).reshape(4, 5) + 100,
            np.arange(20, dtype=np.float32).reshape(4, 5) + 200,
        ]
    )
    widget = Show2D(
        data,
        labels=["raw", "filtered", "hidden"],
        title="viewer source",
        sampling=0.25,
        units="nm",
        cmap="magma",
        hidden_panels=["hidden"],
        show_stats=False,
        verbose=False,
    )

    out = widget.to_show3d()

    assert isinstance(out, Show3D)
    assert out.n_slices == 2
    assert out.n_panels == 1
    assert out.labels == ["raw", "filtered"]
    assert out.panel_titles == ["viewer source"]
    assert out.display_bin == 4
    assert out.pixel_size == pytest.approx(1.0)
    assert out.pixel_unit == "nm"
    np.testing.assert_allclose(out._data[0], data[0])
    np.testing.assert_allclose(out._data[1], data[1])


def test_show2d_prepare_request_creates_prepared_show3d_view():
    widget = Show2D(
        np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        labels=["a", "b"],
        verbose=False,
    )

    widget.handoff_request = json.dumps({"mode": "show3d", "id": "test", "panels": [1]})

    assert isinstance(widget.prepared_view, Show3D)
    assert widget.prepared_view.labels == ["b"]
    assert widget.prepared_view_widget is widget.prepared_view
    assert widget.handoff_status == "Showing 3D with 1 frame"

    widget.handoff_request = json.dumps({"mode": "clear", "id": "clear"})

    assert widget.prepared_view is None
    assert widget.prepared_view_widget is None
    assert widget.handoff_status == ""


def test_show3d_to_show2d_uses_current_frame_and_visible_panels():
    panel_a = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    panel_b = panel_a + 1000
    widget = Show3D(
        panel_a,
        panel_b,
        labels=["t0", "t1", "t2"],
        panel_titles=["signal", "reference"],
        title="movie",
        sampling=0.5,
        units="nm",
        cmap="inferno",
        hidden_panels=["reference"],
        show_stats=False,
        verbose=False,
    )
    widget.slice_idx = 2

    out = widget.to_show2d()

    assert isinstance(out, Show2D)
    assert out.n_images == 1
    assert out.labels == ["signal · t2 3/3"]
    # The Show2D handoff returns native source pixels and sampling even though
    # the live Show3D browser view uses its default 4× mean bin.
    assert out.pixel_size == pytest.approx(0.5)
    assert out.pixel_unit == "nm"
    assert out.cmap == "inferno"
    np.testing.assert_allclose(out._data[0], panel_a[2])


def test_show3d_prepare_request_creates_prepared_show2d_view():
    panel_a = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    panel_b = panel_a + 10
    widget = Show3D(
        panel_a,
        panel_b,
        panel_titles=["left", "right"],
        show_stats=False,
        verbose=False,
    )

    widget.handoff_request = json.dumps({"mode": "show2d", "id": "test", "frame": 1, "panel": [0, 1]})

    assert isinstance(widget.prepared_view, Show2D)
    assert widget.prepared_view.n_images == 2
    assert widget.prepared_view.labels == ["left 2/2", "right 2/2"]
    assert widget.prepared_view_widget is widget.prepared_view
    assert widget.handoff_status == "Showing 2D with 2 panels"

    widget.handoff_request = json.dumps({"mode": "clear", "id": "clear"})

    assert widget.prepared_view is None
    assert widget.prepared_view_widget is None
    assert widget.handoff_status == ""


def test_show1d_to_show2d_uses_selected_snapshot_group_and_review_state():
    widget = Show1D(
        {"frame-by-frame": [3.0, 2.0], "lambda 10": [4.0, 5.0]},
        image_cmap="magma",
        sampling=0.25,
        units="nm",
        show_snapshot_fft=True,
    )
    widget.snapshot(
        0,
        reference=np.zeros((4, 5), dtype=np.float32),
        **{
            "frame-by-frame": np.ones((4, 5), dtype=np.float32),
            "lambda_10": np.full((3, 4), 2.0, dtype=np.float32),
        },
    )
    widget.star_trial("frame-by-frame")
    widget.hide_trial("lambda_10")
    widget.goto_snapshot(0)

    out = widget.to_show2d()

    assert isinstance(out, Show2D)
    assert out.n_images == 2
    assert out.labels == ["reference", "frame-by-frame"]
    assert out.cmap == "magma"
    assert out.pixel_size == pytest.approx(0.25)
    assert out.pixel_unit == "nm"
    assert out.show_fft is True
    assert out.link_zoom is True
    assert out.link_pan is True
    assert out.link_contrast is True
    assert out.starred == [0, 1]
    np.testing.assert_allclose(out._data[1], np.ones((4, 5), dtype=np.float32))


def test_show1d_prepare_request_creates_prepared_show2d_view():
    widget = Show1D({"loss": [2.0, 1.0]})
    widget.snapshot(
        4,
        object=np.ones((3, 4), dtype=np.float32),
        probe=np.full((2, 2), 2.0, dtype=np.float32),
    )

    widget.handoff_request = json.dumps({
        "mode": "show2d",
        "id": "test",
        "group": 0,
        "images": ["probe"],
    })

    assert isinstance(widget.prepared_view, Show2D)
    assert widget.prepared_view.n_images == 1
    assert widget.prepared_view.labels == ["probe"]
    assert widget.prepared_view_widget is widget.prepared_view
    assert widget.handoff_status == "Showing 2D with 1 panel"

    widget.handoff_request = json.dumps({"mode": "clear", "id": "clear"})

    assert widget.prepared_view is None
    assert widget.prepared_view_widget is None
    assert widget.handoff_status == ""
