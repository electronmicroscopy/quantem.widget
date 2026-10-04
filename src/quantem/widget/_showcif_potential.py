"""Radial, Gaussian-regularized independent-atom projection tables for ShowCIF."""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=32)
def projected_atom_table(symbol: str, sigma: float = 0.08) -> np.ndarray:
    """Return a Lobato projected potential table in V Å.

    Parameters
    ----------
    symbol : str
        Neutral element symbol, using abTEM's Lobato parameters and units.
    sigma : float, default 0.08
        Explicit transverse Gaussian regularization width in Å. This is a
        preview filter, not a fitted displacement or a frozen-phonon model.

    Returns
    -------
    numpy.ndarray
        Float32 radial samples at 0.005 Å increments, from 0 to 8 Å inclusive.
        The inverse radial Fourier transform uses reciprocal distance in Å⁻¹.

    Notes
    -----
    These are infinite atomic projections. A site is assigned to exactly one
    depth slab. This is not finite-z integration, bonding charge density, or
    an electron-wave simulation. The finite inspection patch is not tiled
    beyond the explicitly requested repeats.
    """
    if not np.isfinite(sigma) or not 0.04 <= sigma <= 0.5:
        raise ValueError("potential_sigma_A must be between 0.04 and 0.5 Å.")
    try:
        from abtem.parametrizations import LobatoParametrization
        from scipy.integrate import simpson
        from scipy.special import j0
    except ImportError as exc:
        raise ImportError(
            "Potential previews require abTEM and SciPy. Install them in this "
            "environment, or use ShowCIF(..., potential=False) for atoms only."
        ) from exc

    # k_max gives exp(-32) Gaussian attenuation. dk resolves the full 8 Å table.
    k = np.linspace(0, 4 / (np.pi * sigma), 2049)
    radius = np.arange(1601, dtype=np.float64) * 0.005
    spectrum = LobatoParametrization().projected_scattering_factor(symbol)(k * k)
    spectrum = spectrum * np.exp(-2 * np.pi**2 * sigma**2 * k**2) * (2 * np.pi * k)
    table = simpson(j0(2 * np.pi * radius[:, None] * k) * spectrum, x=k, axis=1)
    if not np.all(np.isfinite(table)) or np.min(table) < -1e-3:
        raise ValueError(
            f"Invalid projected potential for {symbol}; check the abTEM parameters."
        )
    result = np.maximum(table, 0).astype(np.float32)
    result.flags.writeable = False
    return result
