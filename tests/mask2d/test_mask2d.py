"""Focused tests for the scientist-facing Mask2D selection contract."""

from types import SimpleNamespace

import numpy as np
import pytest

from quantem.widget import Mask2D
from quantem.widget.utils.roi_geometry import roi_masks


def test_mask2d_defaults_to_one_focused_selection_surface():
    widget = Mask2D(np.arange(80 * 96, dtype=np.float32).reshape(80, 96))

    assert widget.mask_shape == "rectangle"
    assert widget.roi_active is True
    assert widget.show_controls is False
    assert widget.show_stats is False
    assert widget.mask.shape == (80, 96)
    assert not widget.mask.any()
    assert widget.geometry is None


def test_mask2d_rectangle_is_available_as_python_boolean_mask():
    widget = Mask2D(np.zeros((40, 50), dtype=np.float32))
    widget.roi_list = [
        {
            "shape": "rectangle",
            "row": 20.0,
            "col": 25.0,
            "width": 10.0,
            "height": 8.0,
            "visible": True,
        }
    ]
    widget.roi_selected_idx = 0

    assert widget.mask.dtype == bool
    assert widget.mask[20, 25]
    assert widget.mask[16, 20]
    assert not widget.mask[15, 20]
    assert widget.geometry["center"] == {"row": 20.0, "col": 25.0}
    assert widget.geometry["bounds"] == {
        "row_min": 16.0,
        "row_max": 24.0,
        "col_min": 20.0,
        "col_max": 30.0,
    }


def test_mask2d_accepts_useful_image_display_configuration():
    widget = Mask2D(
        np.ones((32, 40), dtype=np.float32),
        shape="circle",
        title="Gold particle region",
        cmap="viridis",
        auto_contrast=True,
        sampling=0.2,
        units="nm",
    )

    assert widget.mask_shape == "circle"
    assert widget.title == "Gold particle region"
    assert widget.cmap == "viridis"
    assert widget.auto_contrast is True
    assert widget.pixel_size == pytest.approx(0.2)
    assert widget.pixel_unit == "nm"


def test_mask2d_preserves_dataset_calibration():
    dataset = SimpleNamespace(
        array=np.ones((32, 40), dtype=np.uint16),
        name="Gold HAADF",
        sampling=(0.2, 0.1),
        units=("nm", "nm"),
    )

    widget = Mask2D(dataset)

    assert widget.mask.shape == dataset.array.shape
    assert widget.pixel_size == pytest.approx(0.1)
    assert widget.pixel_unit == "nm"


def test_mask2d_preserves_native_4k_image_coordinates():
    image = np.zeros((4096, 4096), dtype=np.uint16)
    widget = Mask2D(image, shape="circle")
    widget.set_roi(2048, 2048, radius=512)

    mask = widget.mask

    assert mask.shape == image.shape
    assert mask.dtype == bool
    assert mask[2048, 2048]
    assert not mask[0, 0]


def test_mask2d_geometry_uses_the_selected_visible_region_index():
    widget = Mask2D(np.zeros((40, 50), dtype=np.float32))
    widget.roi_list = [
        {"shape": "circle", "row": 4, "col": 5, "radius": 2, "visible": False},
        {"shape": "circle", "row": 10, "col": 12, "radius": 3, "visible": True},
        {"shape": "circle", "row": 30, "col": 32, "radius": 4, "visible": True},
    ]
    widget.roi_selected_idx = 1

    assert widget.geometry["index"] == 1
    assert widget.geometry["center"] == {"row": 10.0, "col": 12.0}


def test_roi_masks_preserve_circle_and_square_geometry():
    masks = roi_masks(
        [
            {"shape": "circle", "row": 10, "col": 12, "radius": 3},
            {"shape": "square", "row": 20, "col": 22, "radius": 2},
        ],
        height=32,
        width=36,
    )

    assert masks.shape == (2, 32, 36)
    assert masks[0, 10, 15]
    assert not masks[0, 10, 16]
    assert masks[1, 18, 20]
    assert masks[1, 22, 24]
    assert not masks[1, 17, 20]


def test_mask2d_clear_keeps_selection_mode_ready():
    widget = Mask2D(np.ones((32, 32), dtype=np.float32), shape="circle")
    widget.set_roi(16, 16, radius=5)

    returned = widget.clear()

    assert returned is widget
    assert widget.roi_list == []
    assert widget.roi_active is True
    assert not widget.mask.any()


@pytest.mark.parametrize("shape", [(2, 20, 20), (20,), (20, 20, 3)])
def test_mask2d_rejects_non_2d_inputs(shape):
    with pytest.raises(ValueError, match="one 2-D image"):
        Mask2D(np.zeros(shape, dtype=np.float32))


def test_mask2d_accepts_singleton_image_stack_used_by_html_export():
    widget = Mask2D(np.zeros((1, 20, 20), dtype=np.float32))

    assert widget.mask.shape == (20, 20)


def test_mask2d_rejects_unknown_shape_with_corrective_choices():
    with pytest.raises(ValueError, match="circle, rectangle, square"):
        Mask2D(np.zeros((24, 24), dtype=np.float32), shape="polygon")


def test_mask2d_rejects_display_binning_that_would_change_mask_shape():
    with pytest.raises(ValueError, match="display_bin=1"):
        Mask2D(np.zeros((24, 24), dtype=np.float32), display_bin=2)


def test_mask2d_state_preserves_selection_shape():
    widget = Mask2D(np.zeros((24, 24), dtype=np.float32), shape="circle")

    state = widget.state_dict()

    assert state["mask_shape"] == "circle"


def test_mask2d_exports_selected_region_to_standalone_html(tmp_path):
    widget = Mask2D(np.zeros((24, 32), dtype=np.float32), shape="circle")
    widget.set_roi(12, 16, radius=5)

    path = widget.export_html(tmp_path / "mask2d.html")
    html = path.read_text(encoding="utf-8")

    assert path.is_file()
    assert '"mask_shape": "circle"' in html
    assert '"shape": "circle"' in html


def test_mask2d_rejects_spatially_downsampled_html_export(tmp_path):
    widget = Mask2D(np.zeros((24, 32), dtype=np.float32))

    with pytest.raises(ValueError, match="downsample=1"):
        widget.export_html(
            tmp_path / "mask2d.html",
            encoding="uint8",
            downsample=2,
        )
