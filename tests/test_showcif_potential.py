"""Independent physical units and numerical controls for the potential preview."""

import numpy as np
import pytest
Atoms = pytest.importorskip("ase").Atoms

from quantem.widget import ShowCIF
from quantem.widget._showcif_potential import projected_atom_table


@pytest.mark.parametrize("symbol", ["O", "Ti", "Ba"])
def test_radial_fourier_table_matches_independent_real_space_convolution(symbol):
    pytest.importorskip("abtem")
    from abtem.parametrizations import LobatoParametrization
    from scipy.integrate import quad
    from scipy.special import i0e

    sigma = 0.08
    potential = LobatoParametrization().projected_potential(symbol)
    table = projected_atom_table(symbol, sigma)
    for r in [0, 0.1, 0.3, 1, 2]:

        def integrand(t, r=r):
            v = float(potential(np.array([t]))[0])
            return (
                v
                * t
                / sigma**2
                * np.exp(-((r - t) ** 2) / (2 * sigma**2))
                * i0e(r * t / sigma**2)
            )

        expected = quad(
            integrand, 0, max(2, r + 1), epsabs=1e-7, points=[r] if r else None
        )[0]
        np.testing.assert_allclose(
            table[round(r / 0.005)], expected, rtol=2e-4, atol=2e-4
        )
    # Preserve the radial integral / zero-frequency potential, not arbitrary peak normalization.
    r = np.arange(1601) * 0.005
    from scipy.integrate import simpson

    expected = LobatoParametrization().projected_scattering_factor(symbol)(
        np.array([0.0])
    )[0]
    np.testing.assert_allclose(simpson(2 * np.pi * r * table, x=r), expected, rtol=5e-4)


def test_potential_opt_in_and_export_preserve_coordinates(tmp_path):
    pytest.importorskip("abtem")
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
