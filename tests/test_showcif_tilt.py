"""Specimen orientation persists without rewriting the scientist's structure."""

import numpy as np
import pytest
import traitlets
Atoms = pytest.importorskip("ase").Atoms
from quantem.widget import ShowCIF


def test_specimen_tilt_preserves_structure_and_serializes():
    atoms = Atoms(
        "BaTiO3",
        positions=[[0, 0, 0], [2, 2, 2], [2, 2, 0], [0, 2, 2], [2, 0, 2]],
        cell=[4, 4, 4],
        pbc=True,
    )
    original = atoms.positions.copy()
    zero = ShowCIF(atoms, repeats=(2, 2, 3))
    tilted = ShowCIF(atoms, repeats=(2, 2, 3), specimen_tilt_mrad=(12, -7))
    assert tilted.get_state()["specimen_tilt_mrad"] == [12.0, -7.0]
    assert tilted.unit_atom_bytes == zero.unit_atom_bytes
    assert tilted.atom_bytes == zero.atom_bytes
    assert tilted.cell == zero.cell
    np.testing.assert_array_equal(atoms.positions, original)
    tilted.specimen_tilt_mrad = [0, 0]
    assert tilted.get_state()["specimen_tilt_mrad"] == [0.0, 0.0]
    for value in [[16, 0], [0], [0, np.nan], [0, np.inf]]:
        with pytest.raises(traitlets.TraitError, match="row, column"):
            tilted.specimen_tilt_mrad = value
