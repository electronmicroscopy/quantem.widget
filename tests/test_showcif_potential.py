"""Independent physical units and numerical controls for the potential preview."""

import numpy as np
import pytest
Atoms = pytest.importorskip("ase").Atoms

from quantem.widget import ShowCIF
from quantem.widget._lobato import (
    PROJECTED_POTENTIAL_PER_SCATTERING_FACTOR,
    QUANTEM_REQUIREMENT,
    projected_scattering_factor,
    scattering_factors_available,
)
from quantem.widget._showcif_potential import projected_atom_table

needs_lobato = pytest.mark.skipif(not scattering_factors_available(), reason=f"needs {QUANTEM_REQUIREMENT}")

# Reference values computed with abTEM 1.0.10 (LobatoParametrization, which this module used before):
# the projected scattering factor in V Å^3 at k = 0, 0.5 and 2 1/Å ...
ABTEM_PROJECTED_SCATTERING_FACTOR = {
    "O": [97.1497, 58.9745, 7.59174],
    "Ti": [420.352, 136.719, 18.8571],
    "Sr": [627.671, 185.041, 30.9894],
    "Ba": [874.706, 265.199, 43.0961],
    "Au": [506.344, 289.191, 57.9116],
}
# ... and the radial table in V Å at r = 0, 0.1, 0.3, 1 and 2 Å.
ABTEM_TABLE = {
    ("O", 0.08): [292.311, 232.467, 83.7375, 2.97188, 0.0367903],
    ("Ti", 0.08): [722.101, 572.234, 219.441, 27.1521, 3.66128],
    ("Ba", 0.08): [1514.61, 1172.97, 411.173, 52.5606, 9.52267],
    ("Au", 0.2): [744.532, 693.886, 411.499, 22.7967, 1.06273],
}


def test_projected_potential_constant_is_h2_over_2pi_m0_e():
    # abTEM 1.0.10 1 / kappa, from older ASE CODATA constants
    assert PROJECTED_POTENTIAL_PER_SCATTERING_FACTOR == pytest.approx(47.87764685419816, rel=1e-7)


@needs_lobato
@pytest.mark.parametrize("symbol", sorted(ABTEM_PROJECTED_SCATTERING_FACTOR))
def test_scattering_factors_match_abtem(symbol):
    from ase.data import atomic_numbers

    f = projected_scattering_factor([atomic_numbers[symbol]], [0.0, 0.5, 2.0])[0]
    # abTEM evaluates with a float32 pi: agreement to ~1e-5
    np.testing.assert_allclose(f, ABTEM_PROJECTED_SCATTERING_FACTOR[symbol], rtol=2e-5)


@needs_lobato
@pytest.mark.parametrize("symbol, sigma", sorted(ABTEM_TABLE))
def test_radial_table_matches_abtem_and_keeps_the_projected_integral(symbol, sigma):
    from ase.data import atomic_numbers
    from scipy.integrate import simpson

    table = projected_atom_table(symbol, sigma)
    assert table.dtype == np.float32 and table.shape == (1601,) and not table.flags.writeable
    for r, expected in zip([0, 0.1, 0.3, 1, 2], ABTEM_TABLE[(symbol, sigma)]):
        assert table[round(r / 0.005)] == pytest.approx(expected, rel=1e-4, abs=1e-4)
    # Preserve the radial integral / zero-frequency potential, not arbitrary peak normalization.
    r = np.arange(1601) * 0.005
    f0 = projected_scattering_factor([atomic_numbers[symbol]], [0.0])[0, 0]
    np.testing.assert_allclose(simpson(2 * np.pi * r * table, x=r), f0, rtol=1e-4)


def test_missing_quantem_names_the_requirement(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_crystal(name, *args, **kwargs):
        if name == "quantem.diffraction.crystal":
            raise ImportError("No module named 'quantem.diffraction.crystal'")
        return real_import(name, *args, **kwargs)

    projected_atom_table.cache_clear()
    monkeypatch.setattr(builtins, "__import__", no_crystal)
    with pytest.raises(ImportError, match="PR #297") as info:
        projected_atom_table("O", 0.08)
    assert "potential=False" in str(info.value) and "abtem" not in str(info.value).lower()


@needs_lobato
def test_potential_opt_in_and_export_preserve_coordinates(tmp_path):
    a = Atoms("BaO", positions=[[0, 0, 0], [1, 1, 1]], cell=[4, 4, 4], pbc=True)
    plain = ShowCIF(a)
    v = ShowCIF(a, potential=True, num_slices=16)
    assert plain.potential_bytes == b""
    assert np.frombuffer(v.potential_bytes, np.float32).shape == (2 * 1601,)
    assert np.isfinite(np.frombuffer(v.potential_bytes, np.float32)).all()
    assert v.num_slices == 16
    assert v.atom_bytes == plain.atom_bytes
    before = v.potential_bytes
    v.num_slices = 32
    v.repeats = [1, 1, 2]
    assert v.potential_bytes == before
    v.potential_sigma_A = 0.12
    assert v.potential_bytes != before
    assert v.export_html(tmp_path / "potential.html").is_file()


@pytest.mark.parametrize(
    "kw",
    [
        {"num_slices": 0},
        {"num_slices": 65},
        {"num_slices": 1.2},
        {"potential_pixels": 100},
        {"potential": True, "potential_sigma_A": 0},
    ],
)
def test_invalid_preview_settings_fail_explicitly(kw):
    a = Atoms("O", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)
    with pytest.raises(ValueError):
        ShowCIF(a, **kw)
