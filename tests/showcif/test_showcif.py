"""ShowCIF: the structure the scientist passed is what the browser gets, untouched by display."""

import numpy as np
import pytest
import traitlets

Atoms = pytest.importorskip("ase").Atoms
from ase.io import write

from quantem.widget import ShowCIF
from quantem.widget._lobato import QUANTEM_REQUIREMENT, scattering_factors_available
from quantem.widget.showcif import projected_atom_table


def bto() -> Atoms:
    """Cubic-like BaTiO3 perovskite cell (5 atoms), the ordered structure most tests inspect."""
    return Atoms(
        "BaTiO3",
        scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
        cell=[3.991, 3.991, 4.0352],
        pbc=True,
    )


def positions(widget: ShowCIF) -> np.ndarray:
    """The unit-cell rows (x, y, z, species index) the browser instances over ``repeats``."""
    return np.frombuffer(widget.unit_atom_bytes, np.float32).reshape(-1, 4)


# --- structure -------------------------------------------------------------------


def test_cif_file_and_ase_structure_expand_identically(tmp_path):
    path = tmp_path / "bto.cif"
    write(path, bto())
    from_file = ShowCIF(path, repeats=(2, 3, 4), zone_axis=(0, 0, 1))
    from_atoms = ShowCIF(bto(), repeats=(2, 3, 4))
    assert positions(from_file).shape == (5, 4)
    np.testing.assert_allclose(positions(from_file), positions(from_atoms), atol=1e-6)
    np.testing.assert_allclose(from_file.unit_cell, from_atoms.unit_cell)
    assert sorted(s["count"] for s in from_file.species) == [24, 24, 72]
    assert from_file.title == "BaTiO3 · Atomic structure"
    assert from_file.source_summary.startswith("bto.cif · 5 atoms/cell")
    before = from_file.unit_atom_bytes
    from_file.visible_species = [False] * 3
    from_file.zone_axis = [1, 1, 0]
    assert from_file.unit_atom_bytes == before


COD_STYLE_CIF = """data_bto
_chemical_formula_sum 'Ba O3 Ti'
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Ba1 Ba 0 0 0
Ti1 Ti 0.5 0.5 0.5
O1 O 0.5 0.5 0
O2 O 0.5 0 0.5
O3 O 0 0.5 0.5
"""


def test_title_names_the_formula_the_cif_writes_else_the_hill_formula(tmp_path):
    written = tmp_path / "written.cif"
    write(written, bto())
    cod = tmp_path / "cod.cif"
    cod.write_text(COD_STYLE_CIF)
    chloroform = Atoms("CHCl3", positions=[(0, 0, 0), (0, 0, 1.1), (1.7, 0, -0.5), (-0.9, 1.5, -0.5), (-0.9, -1.5, -0.5)], cell=[8, 8, 8], pbc=True)
    # ASE writes _chemical_formula_structural BaTiO3 and _chemical_formula_sum "Ba1 Ti1 O3".
    assert ShowCIF(written).title == "BaTiO3 · Atomic structure"
    assert ShowCIF(cod).title == "BaO3Ti · Atomic structure"
    looped = tmp_path / "looped.cif"
    looped.write_text(COD_STYLE_CIF.replace("_chemical_formula_sum 'Ba O3 Ti'", "loop_\n_chemical_formula_sum\n'Ba O3 Ti'"))
    assert ShowCIF(looped).title == "BaO3Ti · Atomic structure"
    # No CIF formula: Hill order puts C, then H, before the alphabetical rest (not CCl3H).
    assert ShowCIF(chloroform).title == "CHCl3 · Atomic structure"
    assert ShowCIF(written, title="Barium titanate").title == "Barium titanate"


def test_repeats_change_rebuilds_the_supercell_not_the_unit_cell():
    original = bto()
    widget = ShowCIF(original, repeats=(4, 4, 8))
    unit_bytes = widget.unit_atom_bytes
    widget.repeats = [2, 3, 4]
    expected = original.repeat((2, 3, 4))
    np.testing.assert_allclose(positions(widget)[:, :3], original.positions, atol=2e-6)
    assert sum(s["count"] for s in widget.species) == len(expected)
    assert widget.unit_atom_bytes == unit_bytes
    assert sorted(s["count"] for s in widget.species) == [24, 24, 72]
    with pytest.raises(traitlets.TraitError, match="inspection region"):
        widget.repeats = [100, 100, 100]
    assert widget.repeats == [2, 3, 4]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"repeats": (0, 2, 2)},
        {"repeats": (1.5, 2, 2)},
        {"repeats": (100, 100, 100)},
        {"zone_axis": (0, 0, 0)},
        {"zone_axis": (1.5, 1, 0)},
        {"num_slices": 0},
        {"num_slices": 65},
    ],
)
def test_invalid_inputs_fail_before_any_atoms_are_packed(kwargs):
    with pytest.raises(ValueError):
        ShowCIF(bto(), **kwargs)


def test_partial_occupancy_and_degenerate_cells_are_refused():
    partial = bto()
    partial.info["occupancy"] = {"0": {"Ba": 0.5}}
    with pytest.raises(ValueError, match="occupancies"):
        ShowCIF(partial)
    flat = bto()
    flat.cell = [[1, 0, 0], [2, 0, 0], [0, 0, 1]]
    with pytest.raises(ValueError, match="finite 3D"):
        ShowCIF(flat)


# --- orientation and tilt ------------------------------------------------------------


def test_specimen_tilt_is_display_state_only():
    atoms = bto()
    untilted = ShowCIF(atoms, repeats=(2, 2, 3))
    tilted = ShowCIF(atoms, repeats=(2, 2, 3))
    tilted.specimen_tilt_mrad = [12, -7]
    assert tilted.get_state()["specimen_tilt_mrad"] == [12.0, -7.0]
    assert tilted.unit_atom_bytes == untilted.unit_atom_bytes
    assert tilted.unit_cell == untilted.unit_cell
    for value in ([16, 0], [0], [0, np.nan], [0, np.inf]):
        with pytest.raises(traitlets.TraitError, match="row, column"):
            tilted.specimen_tilt_mrad = value
    tilted.view_mode = "microscope"
    tilted.field_of_view_A = 24
    assert tilted.unit_atom_bytes == untilted.unit_atom_bytes
    for bad in (0, -1, float("nan")):
        with pytest.raises(traitlets.TraitError, match="finite and positive"):
            tilted.field_of_view_A = bad
    with pytest.raises(traitlets.TraitError, match="three integers"):
        tilted.zone_axis = [0, 0, 0]
    with pytest.raises(traitlets.TraitError, match="visibility flag"):
        tilted.visible_species = [True]


# --- potential preview ----------------------------------------------------------------


needs_lobato = pytest.mark.skipif(not scattering_factors_available(), reason=f"needs {QUANTEM_REQUIREMENT}")

# Reference values from abTEM 1.0.10 (LobatoParametrization, which the preview used before): the projected
# scattering factor in V Å^3 at k = 0, and the radial table in V Å at r = 0, 0.1, 0.3, 1 and 2 Å.
ABTEM_F0 = {"O": 97.1497, "Ti": 420.352, "Ba": 874.706, "Au": 506.344}
ABTEM_TABLE = {
    ("O", 0.08): [292.311, 232.467, 83.7375, 2.97188, 0.0367903],
    ("Ti", 0.08): [722.101, 572.234, 219.441, 27.1521, 3.66128],
    ("Ba", 0.08): [1514.61, 1172.97, 411.173, 52.5606, 9.52267],
    ("Au", 0.2): [744.532, 693.886, 411.499, 22.7967, 1.06273],
}


@needs_lobato
@pytest.mark.parametrize("symbol, sigma", sorted(ABTEM_TABLE))
def test_projected_table_matches_abtem_and_keeps_the_projected_integral(symbol, sigma):
    """quantem's Lobato factors give abTEM's tables, and the blur conserves the radial integral (f(k=0))."""
    from scipy.integrate import simpson

    table = projected_atom_table(symbol, sigma)
    assert table.dtype == np.float32 and table.shape == (1601,) and not table.flags.writeable
    for r, expected in zip([0, 0.1, 0.3, 1, 2], ABTEM_TABLE[(symbol, sigma)]):
        assert table[round(r / 0.005)] == pytest.approx(expected, rel=1e-4, abs=1e-4)
    radius = np.arange(1601) * 0.005
    # abTEM evaluates with a float32 pi: agreement to about 1e-5
    np.testing.assert_allclose(simpson(2 * np.pi * radius * table, x=radius), ABTEM_F0[symbol], rtol=1e-4)


@needs_lobato
def test_potential_tables_follow_the_sigma_control_and_never_touch_atoms():
    atoms = Atoms("BaO", positions=[[0, 0, 0], [1, 1, 1]], cell=[4, 4, 4], pbc=True)
    plain = ShowCIF(atoms)
    preview = ShowCIF(atoms, potential=True, num_slices=16, energy_keV=300, potential_quantity="phase", potential_colormap="magma")
    assert plain.potential_bytes == b""
    tables = np.frombuffer(preview.potential_bytes, np.float32)
    assert tables.shape == (2 * 1601,) and np.isfinite(tables).all()
    assert preview.unit_atom_bytes == plain.unit_atom_bytes
    before = preview.potential_bytes
    preview.num_slices = 32
    preview.repeats = [1, 1, 2]
    assert preview.potential_bytes == before
    preview.potential_sigma_A = 0.12
    assert preview.potential_bytes != before
    assert preview.unit_atom_bytes == plain.unit_atom_bytes  # sigma and repeats never touch the unit cell
    with pytest.raises(traitlets.TraitError, match="0.04 and 0.5"):
        preview.potential_sigma_A = 0
    with pytest.raises(traitlets.TraitError, match="between 0 and 300"):
        preview.energy_keV = 301
    preview.energy_keV = 0  # phase explicitly disabled
