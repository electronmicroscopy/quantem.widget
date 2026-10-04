"""CIF inspection preserves structure and expands the requested unit cells."""

import numpy as np
import pytest
Atoms = pytest.importorskip("ase").Atoms
from ase.io import write

from quantem.widget import ShowCIF


def cell():
    return Atoms(
        "BaTiO3",
        scaled_positions=[
            (0, 0, 0),
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0),
            (0.5, 0, 0.5),
            (0, 0.5, 0.5),
        ],
        cell=[3.991, 3.991, 4.0352],
        pbc=True,
    )


def test_cif_file_and_ase_match_and_filters_do_not_mutate(tmp_path):
    a = cell()
    path = tmp_path / "bto.cif"
    write(path, a)
    w = ShowCIF(path, repeats=(2, 3, 4))
    v = ShowCIF(a, repeats=(2, 3, 4))
    assert w.atom_count == 5 * 24
    assert np.frombuffer(w.atom_bytes, np.float32).reshape(-1, 4).shape == (120, 4)
    assert np.allclose(w.cell, v.cell)
    assert sorted(s["count"] for s in w.species) == [24, 24, 72]
    before = w.atom_bytes
    w.visible_species = [False] * 3
    assert w.atom_bytes == before
    assert np.allclose(w.atoms().positions, a.positions)
    copy = w.atoms()
    copy.positions[:] = 0
    assert np.allclose(w.atoms().positions, a.positions)
    assert w.export_html(tmp_path / "viewer.html").is_file()


def test_partial_sites_and_oversized_cells_are_explicit():
    a = cell()
    a.info["occupancy"] = {"0": {"Ba": 0.5}}
    with pytest.raises(ValueError, match="occupancies"):
        ShowCIF(a)
    with pytest.raises(ValueError, match="inspection region"):
        ShowCIF(cell(), repeats=(100, 100, 100))


@pytest.mark.parametrize(
    "kw",
    [
        {"repeats": (0, 2, 2)},
        {"repeats": (1.5, 2, 2)},
        {"zone_axis": (0, 0, 0)},
        {"zone_axis": (1.5, 1, 0)},
    ],
)
def test_invalid_inputs(kw):
    with pytest.raises(ValueError):
        ShowCIF(cell(), **kw)


def test_display_updates_validate_direction_and_species():
    import traitlets

    w = ShowCIF(cell())
    with pytest.raises(traitlets.TraitError, match="three integers"):
        w.zone_axis = [0, 0, 0]
    with pytest.raises(traitlets.TraitError, match="visibility flag"):
        w.visible_species = [True]
    w.zone_axis = [1, 1, 0]
    assert w.zone_axis == [1, 1, 0]


def test_degenerate_cell_fails_before_projection():
    a = cell()
    a.cell = [[1, 0, 0], [2, 0, 0], [0, 0, 1]]
    with pytest.raises(ValueError, match="finite 3D"):
        ShowCIF(a)


def test_repeat_updates_preserve_unit_cell_and_export_consistency(tmp_path):
    import traitlets

    original = cell()
    w = ShowCIF(original, repeats=(4, 4, 8))
    unit_bytes = w.unit_atom_bytes
    w.visible_species = [False, True, True]
    w.repeats = [2, 3, 4]
    expected = original.repeat((2, 3, 4))
    packed = np.frombuffer(w.atom_bytes, np.float32).reshape(-1, 4)
    assert w.atom_count == 120
    np.testing.assert_allclose(packed[:, :3], expected.positions, atol=2e-6)
    np.testing.assert_allclose(w.cell, expected.cell)
    assert sorted(s["count"] for s in w.species) == [24, 24, 72]
    assert w.unit_atom_bytes == unit_bytes
    np.testing.assert_array_equal(w.atoms().positions, original.positions)
    assert w.visible_species == [False, True, True]
    with pytest.raises(traitlets.TraitError):
        w.repeats = [100, 100, 100]
    with pytest.raises(traitlets.TraitError):
        w.repeats = [0, 3, 4]
    assert w.repeats == [2, 3, 4]
    assert w.export_html(tmp_path / "repeated.html").is_file()


def test_microscope_fov_and_calibration_preserve_structure(tmp_path):
    import traitlets

    w = ShowCIF(
        cell(),
        repeats=(4, 4, 8),
        view_mode="microscope",
        field_of_view_A=16,
        magnification_calibration=(1e6, 100),
    )
    before = (w.atom_bytes, w.unit_atom_bytes, w.cell, w.repeats)
    w.field_of_view_A = 64
    w.view_mode = "unit_cells"
    assert (w.atom_bytes, w.unit_atom_bytes, w.cell, w.repeats) == before
    w.view_mode = "microscope"
    html = w.export_html(tmp_path / "microscope.html").read_text()
    assert '"field_of_view_A": 64.0' in html
    assert '"view_mode": "microscope"' in html
    assert w.magnification_calibration == [1e6, 100]
    for bad in [0, -1, float("nan"), float("inf")]:
        with pytest.raises(traitlets.TraitError, match="finite and positive"):
            w.field_of_view_A = bad
    for bad in [[1e6], [1e6, 0], [1e6, float("nan")]]:
        with pytest.raises(traitlets.TraitError, match="reference magnification"):
            w.magnification_calibration = bad
    w.magnification_calibration = []
    with pytest.raises(traitlets.TraitError):
        w.view_mode = "unknown"


def test_orthogonal_view_export_preserves_atoms(tmp_path):
    w = ShowCIF(cell(), orthogonal_views=True)
    before = w.atom_bytes
    assert w.orthogonal_views
    w.orthogonal_views = False
    w.orthogonal_views = True
    assert w.atom_bytes == before
    assert (
        '"orthogonal_views": true' in w.export_html(tmp_path / "ortho.html").read_text()
    )


def test_phase_display_energy_and_colormap_are_saved_without_data_changes(tmp_path):
    import traitlets

    w = ShowCIF(
        cell(), energy_keV=300, potential_quantity="phase", potential_colormap="magma"
    )
    before = w.atom_bytes
    w.energy_keV = 60
    w.potential_colormap = "inferno"
    assert w.atom_bytes == before
    text = w.export_html(tmp_path / "phase.html").read_text()
    assert '"energy_keV": 60.0' in text
    assert '"potential_quantity": "phase"' in text
    assert '"potential_colormap": "inferno"' in text
    w.energy_keV = 0  # explicitly undefined phase, supported control endpoint
    for energy in [-1, 301, float("nan"), float("inf")]:
        with pytest.raises(traitlets.TraitError, match="between 0 and 300"):
            w.energy_keV = energy
    with pytest.raises(traitlets.TraitError):
        w.potential_colormap = "unknown"


def test_invalid_occupancy_and_large_repeat_product_fail_before_allocation():
    atoms = cell()
    atoms.info["occupancy"] = {"0": {"Ba": float("nan")}}
    with pytest.raises(ValueError, match="occupancies"):
        ShowCIF(atoms)
    with pytest.raises(ValueError, match="inspection region"):
        ShowCIF(cell(), repeats=(2**32, 2**32, 2))


def test_live_slice_and_direction_settings_reject_booleans():
    import traitlets

    viewer = ShowCIF(cell())
    for name, value in [("num_slices", True), ("zone_axis", [False, False, True])]:
        with pytest.raises(traitlets.TraitError):
            setattr(viewer, name, value)
    assert viewer.num_slices == 16
    assert viewer.zone_axis == [0, 0, 1]
