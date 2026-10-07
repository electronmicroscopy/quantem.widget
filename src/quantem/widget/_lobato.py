"""Lobato electron scattering factors in projected-potential units.

The factors are Lobato & Van Dyck, Acta Cryst. A 70, 636 (2014),
https://doi.org/10.1107/S205327331401643X, evaluated by
``quantem.diffraction.crystal.electron_scattering_factor``. That function
arrives with electronmicroscopy/quantem PR #297, in the first release after
0.1.9, so it is imported on first use.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import constants

# h^2 / (2 pi m0 e) in V Å^2 (47.8776 V Å^2): turns a scattering factor f_e(k) in Å into the 2D Fourier transform of an
# atom's projected potential in V Å^3, V(k) = PROJECTED_POTENTIAL_PER_SCATTERING_FACTOR * f_e(k).
PROJECTED_POTENTIAL_PER_SCATTERING_FACTOR = constants.h**2 / (2 * math.pi * constants.m_e * constants.e) * 1e20

QUANTEM_REQUIREMENT = (
    "quantem with quantem.diffraction.Crystal and bloch "
    "(electronmicroscopy/quantem PR #297, first release after 0.1.9)"
)


def _scattering_factor_function(feature: str):
    try:
        from quantem.diffraction.crystal import electron_scattering_factor
    except ImportError as exc:
        raise ImportError(
            f"{feature} needs {QUANTEM_REQUIREMENT} for its Lobato electron "
            f"scattering factors. ({exc})"
        ) from exc
    return electron_scattering_factor


def require_scattering_factors(feature: str) -> None:
    """Check that quantem's Lobato scattering factors can be imported.

    Parameters
    ----------
    feature : str
        What needs them, used to start the error message.

    Raises
    ------
    ImportError
        If ``quantem.diffraction.crystal.electron_scattering_factor`` is
        missing, naming the quantem version that has it.
    """
    _scattering_factor_function(feature)


def scattering_factors_available() -> bool:
    """True when ``quantem.diffraction.crystal.electron_scattering_factor`` imports."""
    try:
        _scattering_factor_function("")
    except ImportError:
        return False
    return True


def projected_scattering_factor(numbers, k, feature: str = "This calculation") -> np.ndarray:
    """Fourier transform of each atom's infinite projected potential.

    Parameters
    ----------
    numbers : array_like of int
        Atomic numbers, shape (N,).
    k : array_like of float
        Spatial frequencies |k| in 1/Å (no factor 2 pi), any shape.
    feature : str, optional
        What needs the factors, for the ``ImportError`` message.

    Returns
    -------
    numpy.ndarray
        Float64 array of shape (N, *k.shape) in V Å^3: the Lobato scattering
        factor f_e(|k|) times h^2 / (2 pi m0 e). Its value at k = 0 is the
        integral of the projected potential over the plane.
    """
    import torch

    function = _scattering_factor_function(feature)
    k = np.asarray(k, dtype=np.float64)
    numbers = np.atleast_1d(np.asarray(numbers, dtype=np.int64))
    f = function(torch.as_tensor(numbers), torch.as_tensor(k.ravel(), dtype=torch.float64)).numpy()
    return f.reshape(numbers.shape + k.shape) * PROJECTED_POTENTIAL_PER_SCATTERING_FACTOR
