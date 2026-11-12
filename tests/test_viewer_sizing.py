"""Panel presets preserve native images and explicit/saved display choices."""

import numpy as np
import pytest

from quantem.widget import Show2D, Show3D


@pytest.mark.parametrize("size, width", [("small", 300), ("medium", 450), ("large", 600), ("xlarge", 900)])
def test_named_sizes_keep_rectangular_images_and_stack_pixels(size, width):
    image = np.arange(24 * 40, dtype=np.float32).reshape(24, 40)
    stack = np.stack([image, image + 10])
    gallery = Show2D(image, size=size, verbose=False)
    movie = Show3D(stack, size=size, verbose=False)
    try:
        np.testing.assert_array_equal(gallery._data[0], image)
        np.testing.assert_array_equal(movie._get_source_panel_frame(0, 1), stack[1])
        assert gallery.panel_width_px == movie.panel_width_px == width
        assert gallery.display_bin == movie.display_bin == 1
    finally:
        gallery.close()
        movie.close()


def test_custom_and_saved_widths_override_the_named_size():
    image = np.zeros((24, 40), dtype=np.float32)
    original = Show2D(image, size="small", panel_width_px=380, verbose=False)
    restored = Show2D(image, size="large", state=original.state_dict(), verbose=False)
    try:
        assert original.panel_width_px == restored.panel_width_px == 380
    finally:
        original.close()
        restored.close()


def test_invalid_size_has_a_corrective_error():
    with pytest.raises(ValueError, match="choose 'small'.*'xlarge'"):
        Show2D(np.zeros((8, 8)), size="huge")
    with pytest.raises(TypeError, match="use panel_width_px"):
        Show3D(np.zeros((2, 8, 8)), size=400)
