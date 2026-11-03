"""ChooseLattice: three clicks on an image become an origin and two lattice vectors."""

import io

import numpy as np
import pytest
import traitlets
from PIL import Image

from quantem.widget import ChooseLattice
from quantem.widget.adapters.core import make_dataset


def lattice_image(size: int = 120, origin=(30.0, 27.0), u=(1.0, 15.0), v=(13.0, -5.0)) -> np.ndarray:
    """Gaussian peaks on an oblique lattice, like the tutorial image."""
    rows, cols = np.mgrid[0:size, 0:size]
    image = np.zeros((size, size), dtype=np.float32)
    for i in range(-1, 7):
        for j in range(-1, 7):
            peak_row = origin[0] + i * u[0] + j * v[0]
            peak_col = origin[1] + i * u[1] + j * v[1]
            image += np.exp(-((rows - peak_row) ** 2 + (cols - peak_col) ** 2) / (2 * 2.0**2)).astype(np.float32)
    return image


def test_image_is_sent_as_a_png_of_the_same_size():
    widget = ChooseLattice(lattice_image(), cmap="inferno", title="Synthetic atomic lattice")
    assert (widget.height, widget.width) == (120, 120)
    assert widget.title == "Synthetic atomic lattice"
    assert widget.points == []
    decoded = Image.open(io.BytesIO(bytes(widget.frame_bytes))).convert("RGB")
    assert decoded.size == (120, 120)
    hottest = np.asarray(decoded).sum(axis=2)
    # the brightest rendered pixel sits on a lattice peak, not on the dark background
    peak_row, peak_col = np.unravel_index(int(hottest.argmax()), hottest.shape)
    assert lattice_image()[peak_row, peak_col] > 0.5


def test_dataset2d_name_is_the_default_title():
    dataset = make_dataset(lattice_image(32), name="my lattice")
    widget = ChooseLattice(dataset)
    assert (widget.height, widget.width, widget.title) == (32, 32, "my lattice")
    with pytest.raises(ValueError, match="single 2D image"):
        ChooseLattice(np.zeros((2, 8, 8), dtype=np.float32))


def test_three_clicks_give_the_origin_and_both_lattice_vectors():
    widget = ChooseLattice(lattice_image())
    assert (widget.origin, widget.u, widget.v) == (None, None, None)
    widget.points = [[30.0, 27.0]]
    assert widget.origin == (30.0, 27.0) and widget.u is None
    widget.points = [[30.0, 27.0], [31.0, 42.0]]
    assert widget.u == (1.0, 15.0) and widget.v is None
    widget.points = [[30.0, 27.0], [31.0, 42.0], [43.0, 22.0]]
    assert widget.u == (1.0, 15.0)
    assert widget.v == (13.0, -5.0)
    widget.points = [[30.0, 27.0], [31.0, 42.0], [44.5, 21.0]]  # dragging the third point
    assert widget.v == (14.5, -6.0)
    widget.points = []
    assert (widget.origin, widget.u, widget.v) == (None, None, None)


def test_points_stay_inside_the_image_and_stop_at_three():
    widget = ChooseLattice(lattice_image(size=20)[:10])
    widget.points = [[-5.0, 100.0]]
    assert widget.points == [[0.0, 19.0]]
    with pytest.raises(traitlets.TraitError, match="at most 3"):
        widget.points = [[0, 0], [1, 1], [2, 2], [3, 3]]
    assert widget.points == [[0.0, 19.0]]
