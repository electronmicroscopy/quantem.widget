"""Plain layout keywords lay out the panels and leave the images unchanged."""

import numpy as np
import pytest

from quantem.widget import Show2D, Show3D


@pytest.mark.parametrize("viewer_class", [Show2D, Show3D])
def test_columns_is_the_layout_keyword_trait_and_state_key(viewer_class):
    images = np.random.default_rng(8).random((5, 3, 24, 32)).astype(np.float32)
    data = list(images[:, 0] if viewer_class is Show2D else images)
    viewer = viewer_class(data, columns=2, size="medium", verbose=False)
    assert viewer.columns == viewer.get_state()["columns"] == viewer.state_dict()["columns"] == 2
    assert viewer.panel_width_px == 450
    viewer.columns = 3
    assert viewer.get_state()["columns"] == 3
    np.testing.assert_array_equal(viewer._data, viewer_class(data, columns=2, verbose=False)._data)
    with pytest.raises(TypeError, match="ncols|max_cols"):
        viewer_class(data, **{"ncols" if viewer_class is Show2D else "max_cols": 2}, verbose=False)


@pytest.mark.parametrize("viewer_class", [Show2D, Show3D])
def test_rows_reflow_each_page_without_counting_other_pages(viewer_class):
    images = np.random.default_rng(9).random((2, 5, 3, 24, 32)).astype(np.float32)
    data = images[:, :, 0] if viewer_class is Show2D else images
    viewer = viewer_class(data, rows=2, size="small", verbose=False)
    assert viewer.columns == 3
    assert viewer.n_pages == 2
    viewer.page_idx = 1
    assert viewer.columns == 3
    assert viewer.panel_width_px == 300


@pytest.mark.parametrize("viewer_class", [Show2D, Show3D])
def test_conflicting_layout_requests_explain_how_to_keep_all_panels(viewer_class):
    images = np.ones((5, 3, 12, 16), dtype=np.float32)
    data = list(images[:, 0] if viewer_class is Show2D else images)
    with pytest.raises(ValueError, match="cannot hold 5 panels"):
        viewer_class(data, rows=2, columns=2, verbose=False)
    for value in (0, True, 2.5):
        with pytest.raises(ValueError, match="positive integer"):
            viewer_class(data, columns=value, verbose=False)
    viewer = viewer_class(data, verbose=False)
    with pytest.raises(ValueError, match="positive integer"):
        viewer.columns = 0
