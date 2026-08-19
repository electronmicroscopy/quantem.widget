from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from quantem.widget.showbragg import ShowBragg

pytest.importorskip("quantem.diffraction.bragg_vectors")

from quantem.core.datastructures.dataset4dstem import Dataset4dstem  # noqa: E402
from quantem.diffraction.bragg_vectors import BraggVectors  # noqa: E402
from quantem.diffraction.strain import StrainMap  # noqa: E402

G1 = np.array([0.0, 7.0])
G2 = np.array([7.0, 0.0])


def synthetic_dataset(scan=(6, 5), detector=(32, 32)) -> Dataset4dstem:
    """Square lattice of soft disks, identical at every scan position."""
    rows, cols = scan
    height, width = detector
    rr, cc = np.mgrid[0:height, 0:width]
    origin = np.array([height / 2, width / 2])

    pattern = np.zeros((height, width), dtype=np.float32)
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            spot = origin + a * G1 + b * G2
            pattern += np.exp(-((rr - spot[0]) ** 2 + (cc - spot[1]) ** 2) / 4.0)

    data = np.broadcast_to(pattern, (rows, cols, height, width)).copy()
    return Dataset4dstem.from_array(data, name="synthetic lattice")


@pytest.fixture
def widget() -> ShowBragg:
    return ShowBragg(synthetic_dataset())


@pytest.fixture
def fitted(widget: ShowBragg) -> ShowBragg:
    widget.detect()
    widget.fit()
    return widget


def test_construction_reports_shapes_and_renders_template(widget: ShowBragg) -> None:
    assert widget.scan_shape == [6, 5]
    assert widget.q_shape == [32, 32]
    assert widget.title == "synthetic lattice"
    assert widget.peaks is None

    decoded = Image.open(io.BytesIO(bytes(widget.template_png))).convert("RGB")
    assert decoded.size == (widget.template_shape[1], widget.template_shape[0])
    assert 0 < widget.template_shape[0] <= widget.q_shape[0]


def test_construction_from_bragg_vectors_keeps_detected_state() -> None:
    bragg = BraggVectors.from_dataset(synthetic_dataset())
    bragg.make_template_synthetic(radius=2.0)
    bragg.detect_disks(progressbar=False)

    widget = ShowBragg(bragg)

    assert widget.bragg is bragg
    assert widget.peaks is not None
    assert widget.bvm is not None
    assert len(widget.bvm_png) > 0


def test_rejects_bad_data_and_unknown_kwarg() -> None:
    with pytest.raises(TypeError):
        ShowBragg(np.random.rand(16, 16).astype(np.float32))
    with pytest.raises(TypeError):
        ShowBragg(synthetic_dataset(), not_a_real_kwarg=True)


def test_template_trait_change_rerenders_preview(widget: ShowBragg) -> None:
    before = bytes(widget.template_png)
    widget.template_radius = 5.0
    assert bytes(widget.template_png) != before


def test_probe_position_clamped_to_scan_shape(widget: ShowBragg) -> None:
    widget.probe_position = [99, -4]
    assert widget.probe_position == [5, 0]


def test_preview_detection_leaves_full_dataset_peaks_empty(widget: ShowBragg) -> None:
    preview = widget.detect(positions=[(0, 0), (2, 3)])

    assert preview.shape == (2,)
    assert widget.peaks is None
    assert widget.bvm is None
    assert widget.preview_peaks != ""


def test_basis_accepts_candidate_index_or_vector(widget: ShowBragg) -> None:
    widget.detect()

    widget.origin_index = 0
    widget.g1_index = 1
    widget.g2_index = 2
    by_index = tuple(np.array(v) for v in widget.basis)

    widget.origin_index = -1
    widget.g1_index = -1
    widget.g2_index = -1
    widget.origin_rc = list(by_index[0])
    widget.g1_rc = list(by_index[1])
    widget.g2_rc = list(by_index[2])
    by_vector = tuple(np.array(v) for v in widget.basis)

    for expected, actual in zip(by_index, by_vector):
        np.testing.assert_allclose(actual, expected)

    origin = np.asarray(widget.candidates[0][:2], dtype=float)
    np.testing.assert_allclose(by_index[0], origin)


def test_candidate_count_defaults_to_the_busiest_scan_position(widget: ShowBragg) -> None:
    assert widget.num_candidates == 0

    widget.detect()
    busiest = max(widget.peaks.row_counts())
    assert len(widget.candidates) <= busiest

    widget.num_candidates = 5
    assert len(widget.candidates) <= 5


def test_detect_and_fit_populate_results(fitted: ShowBragg) -> None:
    assert fitted.detection_state == "done"
    assert fitted.fit_state == "done"
    assert fitted.peaks.shape == (6, 5)
    assert fitted.bragg.u_array.shape == (6, 5, 2)
    assert len(fitted.mask_weight_png) > 0
    assert len(fitted.fit_error_png) > 0


def test_ui_mode_preset_and_control_toggles() -> None:
    widget = ShowBragg(synthetic_dataset(), ui_mode="report")
    assert widget.show_title is True
    assert widget.show_controls is False

    widget.expand_controls()
    assert widget.controls_collapsed is False
    widget.toggle_controls()
    assert widget.controls_collapsed is True

    explicit = ShowBragg(synthetic_dataset(), ui_mode="minimal", show_title=True)
    assert explicit.show_title is True
    assert explicit.show_controls is False


def test_strain_map_requires_a_fit(widget: ShowBragg) -> None:
    with pytest.raises(ValueError, match="fit_lattice"):
        widget.strain_map()

    widget.detect()
    widget.fit()
    assert isinstance(widget.strain_map(), StrainMap)
