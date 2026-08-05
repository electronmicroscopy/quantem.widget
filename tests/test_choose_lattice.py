from __future__ import annotations

import base64
import io

import numpy as np
import pytest
import traitlets
from PIL import Image

from quantem.widget.choose_lattice import ChooseLattice


def test_choose_lattice_construction_renders_frame() -> None:
    data = np.random.rand(48, 64).astype(np.float32)
    widget = ChooseLattice(data, title="lattice")

    assert widget.height == 48
    assert widget.width == 64
    assert widget.title == "lattice"
    assert widget.point_labels == ["Origin", "u", "v"]
    assert widget.points == []
    assert len(widget.frame_bytes) > 0

    decoded = Image.open(io.BytesIO(bytes(widget.frame_bytes))).convert("RGB")
    assert decoded.size == (64, 48)


def test_choose_lattice_rejects_non_2d_data() -> None:
    with pytest.raises(ValueError):
        ChooseLattice(np.random.rand(4, 8, 8).astype(np.float32))


def test_choose_lattice_set_points_and_properties() -> None:
    widget = ChooseLattice(np.random.rand(32, 32).astype(np.float32))

    assert widget.origin is None
    assert widget.a1 is None
    assert widget.a2 is None
    assert widget.u is None
    assert widget.v is None

    widget.set_points([[1.0, 2.0], [3.0, 4.0]])
    assert widget.origin == (1.0, 2.0)
    assert widget.a1 == (3.0, 4.0)
    assert widget.a2 is None
    assert widget.u == (2.0, 2.0)
    assert widget.v is None
    np.testing.assert_array_equal(widget.points_array, np.array([[1.0, 2.0], [3.0, 4.0]]))

    widget.set_points([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert widget.a2 == (5.0, 6.0)
    assert widget.u == (2.0, 2.0)
    assert widget.v == (4.0, 4.0)

    widget.clear_points()
    assert widget.points == []
    assert widget.origin is None
    assert widget.u is None
    assert widget.v is None


def test_choose_lattice_points_reject_more_than_three() -> None:
    widget = ChooseLattice(np.random.rand(16, 16).astype(np.float32))
    with pytest.raises(traitlets.TraitError):
        widget.set_points([[0, 0], [1, 1], [2, 2], [3, 3]])


def test_choose_lattice_points_clamp_to_image_bounds() -> None:
    widget = ChooseLattice(np.random.rand(10, 20).astype(np.float32))
    widget.set_points([[-5.0, 100.0]])
    assert widget.points == [[0.0, 19.0]]


def test_choose_lattice_static_png_b64_matches_frame_bytes() -> None:
    widget = ChooseLattice(np.random.rand(20, 30).astype(np.float32))
    png_b64 = widget._static_png_b64()
    assert png_b64 is not None
    decoded = Image.open(io.BytesIO(base64.b64decode(png_b64))).convert("RGB")
    assert decoded.size == (30, 20)


def test_choose_lattice_accepts_dataset2d() -> None:
    core_datastructures = pytest.importorskip("quantem.core.datastructures")
    dataset = core_datastructures.Dataset2d.from_array(
        np.random.rand(24, 24).astype(np.float32), name="my lattice"
    )
    widget = ChooseLattice(dataset)
    assert widget.height == 24
    assert widget.width == 24
    assert widget.title == "my lattice"


def test_choose_lattice_rejects_unknown_kwarg() -> None:
    with pytest.raises(TypeError):
        ChooseLattice(np.random.rand(8, 8).astype(np.float32), not_a_real_kwarg=True)


def test_choose_lattice_static_fallback_disabled_by_default() -> None:
    """Unlike Show2D/Show3D, ChooseLattice's live widget does not reliably
    hide the saved-notebook fallback sibling while interactive, so the
    fallback stays off by default to avoid a redundant visible image; users
    can opt in via notebook_preview_format="jpeg" explicitly."""
    widget = ChooseLattice(np.random.rand(8, 8).astype(np.float32))
    assert not widget._static_fallback_enabled()

    opted_in = ChooseLattice(
        np.random.rand(8, 8).astype(np.float32), notebook_preview_format="jpeg"
    )
    assert opted_in._static_fallback_enabled()
