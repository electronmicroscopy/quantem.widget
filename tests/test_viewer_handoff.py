import json

import numpy as np
import pytest

from quantem.widget.show2d import Show2D
from quantem.widget.show3d import Show3D


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
    assert out.pixel_size == pytest.approx(0.25)
    assert out.pixel_unit == "nm"
    np.testing.assert_allclose(out._display_data[0], data[0])
    np.testing.assert_allclose(out._display_data[1], data[1])


def test_show2d_prepare_request_creates_prepared_show3d_view():
    widget = Show2D(
        np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        labels=["a", "b"],
        verbose=False,
    )

    widget.handoff_request = json.dumps({"mode": "show3d", "id": "test", "panels": [1]})

    assert isinstance(widget.prepared_view, Show3D)
    assert widget.prepared_view.labels == ["b"]
    assert widget.handoff_status == "Ready: 3D with 1 frame"
    assert widget.show_prepared_view() is widget.prepared_view


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
    assert widget.handoff_status == "Ready: 2D with 2 panels"
    assert widget.show_prepared_view() is widget.prepared_view
