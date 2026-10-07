"""Crystal data for DiffractionSim: preset structures and the browser payload.

This module has no widget imports, so ``scripts/diffraction_sim_presets.py``
can generate the website presets from the same code the Jupyter widget uses.

The Bloch-wave tools come from ``quantem.diffraction`` (``Crystal`` and
``bloch``) and the preset structures from ASE, a dependency of that quantem
version. Both are imported on first use;
:func:`require_crystal_tools` raises an ``ImportError`` with the install step
when either is missing. ``Crystal`` and ``bloch`` are not in a quantem release
yet: they arrive with electronmicroscopy/quantem PR #297, in the first release
after 0.1.9.

Energies in this module are in eV (``energy_ev``), the unit of the quantem
Bloch-wave functions; the widget trait is ``energy_keV``.
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np
import torch

# ASE bulk structures offered in the crystal menu.
PRESETS: dict[str, dict[str, Any]] = {
    "Si (diamond cubic)": {"name": "Si", "crystalstructure": "diamond", "a": 5.431, "cubic": True},
    "Ge (diamond cubic)": {"name": "Ge", "crystalstructure": "diamond", "a": 5.658, "cubic": True},
    "Al (fcc)": {"name": "Al", "crystalstructure": "fcc", "a": 4.05, "cubic": True},
    "Cu (fcc)": {"name": "Cu", "crystalstructure": "fcc", "a": 3.615, "cubic": True},
    "Au (fcc)": {"name": "Au", "crystalstructure": "fcc", "a": 4.078, "cubic": True},
    "Fe (bcc)": {"name": "Fe", "crystalstructure": "bcc", "a": 2.866, "cubic": True},
    "W (bcc)": {"name": "W", "crystalstructure": "bcc", "a": 3.165, "cubic": True},
    "Ti (hcp)": {"name": "Ti", "crystalstructure": "hcp", "a": 2.9505, "c": 4.6855},
    "Mg (hcp)": {"name": "Mg", "crystalstructure": "hcp", "a": 3.209, "c": 5.211},
    "GaAs (zincblende)": {
        "name": "GaAs",
        "crystalstructure": "zincblende",
        "a": 5.653,
        "cubic": True,
    },
    "NaCl (rocksalt)": {"name": "NaCl", "crystalstructure": "rocksalt", "a": 5.64, "cubic": True},
    "SrTiO3 (perovskite)": {
        "symbols": ["Sr", "Ti", "O"],
        "basis": [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
        "spacegroup": 221,
        "cellpar": [3.905, 3.905, 3.905, 90, 90, 90],
    },
    "Al2O3 (corundum)": {
        "symbols": ["Al", "O"],
        "basis": [(0, 0, 0.3523), (0.3064, 0, 0.25)],
        "spacegroup": 167,
        "cellpar": [4.7602, 4.7602, 12.9933, 90, 90, 120],
    },
    "SiO2 (alpha quartz)": {
        "symbols": ["Si", "O"],
        "basis": [(0.4697, 0, 0), (0.4135, 0.2669, 0.1191)],
        "spacegroup": 154,
        "cellpar": [4.9134, 4.9134, 5.4052, 90, 90, 120],
    },
    "alpha-Mn (58 atoms)": {
        "symbols": ["Mn", "Mn", "Mn", "Mn"],
        "basis": [
            (0, 0, 0),
            (0.3175, 0.3175, 0.3175),
            (0.3570, 0.3570, 0.0348),
            (0.0896, 0.0896, 0.2820),
        ],
        "spacegroup": 217,
        "cellpar": [8.911, 8.911, 8.911, 90, 90, 90],
    },
}

_INSTALL_HINT = (
    "DiffractionSim needs quantem with quantem.diffraction.Crystal and bloch "
    "(electronmicroscopy/quantem PR #297, first release after 0.1.9), which "
    "brings ASE and spglib."
)


def require_crystal_tools() -> None:
    """Check that ASE and ``quantem.diffraction.Crystal`` can be imported.

    Raises
    ------
    ImportError
        If ASE or the quantem Bloch-wave tools are missing, with the install
        step in the message.
    """
    try:
        import ase  # noqa: F401
        from quantem.diffraction import bloch  # noqa: F401
        from quantem.diffraction.crystal import Crystal  # noqa: F401
    except ImportError as exc:
        raise ImportError(f"{_INSTALL_HINT} ({exc})") from exc


def preset_atoms(key: str):
    """Build the ASE ``Atoms`` of one preset structure.

    Parameters
    ----------
    key : str
        A key of :data:`PRESETS`, e.g. ``"Si (diamond cubic)"``.

    Returns
    -------
    ase.Atoms
        The conventional cell of the structure.
    """
    if key not in PRESETS:
        raise KeyError(f"unknown preset {key!r}; choose one of {list(PRESETS)}")
    require_crystal_tools()
    from ase.build import bulk

    spec = PRESETS[key]
    if "spacegroup" in spec:
        from ase.spacegroup import crystal as ase_crystal

        return ase_crystal(
            spec["symbols"],
            basis=spec["basis"],
            spacegroup=spec["spacegroup"],
            cellpar=spec["cellpar"],
        )
    return bulk(**spec)


def preset_crystal(key: str):
    """Build the quantem ``Crystal`` of one preset structure.

    Parameters
    ----------
    key : str
        A key of :data:`PRESETS`.

    Returns
    -------
    quantem.diffraction.crystal.Crystal
        The crystal, named by the part of ``key`` before the parenthesis
        (``"Si (diamond cubic)"`` gives ``"Si"``).
    """
    atoms = preset_atoms(key)
    from quantem.diffraction.crystal import Crystal

    return Crystal.from_ase(atoms, name=key.split(" (")[0], verbose=False)


def _f32_b64(a) -> str:
    raw = np.ascontiguousarray(np.asarray(a, dtype=np.float32)).tobytes()
    return base64.b64encode(raw).decode()


def prepare_crystal(crystal, energy_ev: float, k_max: float) -> dict[str, Any]:
    """Pack everything the browser needs to draw the cell and simulate patterns.

    The reflection list holds the points of the primitive reciprocal lattice
    within ``k_max``. Glide-forbidden reflections such as Si 200 and 222 are
    kept with zero kinematical intensity; they fill by multiple scattering in
    the Bloch calculation. Each reflection carries its kinematical
    ``|F_g|^2`` (Lobato) and its Bloch coupling ``U_g`` (absorptive
    Weickenmeier-Kohl factors at ``energy_ev`` when available, else
    ``gamma F_g / pi``). The browser builds the structure matrix from the same
    list, so couplings between beams further apart than ``k_max`` are taken as
    zero.

    Parameters
    ----------
    crystal : quantem.diffraction.crystal.Crystal
        The crystal. Its structure factors are computed here if missing or
        computed for a smaller ``k_max``.
    energy_ev : float
        Beam energy in eV.
    k_max : float
        Largest scattering vector of the reflection list, in 1/Å.

    Returns
    -------
    dict
        JSON-serializable crystal data: cell, reciprocal cell, fractional
        positions, atomic numbers, symbols, colors and radii, and the
        reflection list (``hkl_i16`` int16, ``F2``, ``U_re`` and ``U_im``
        float32, all base64), plus ``u0_imag``, ``absorptive``,
        ``n_reflections``, ``energy_ev``, ``wavelength`` (Å), ``k_max`` and
        ``hexagonal``.
    """
    require_crystal_tools()
    from ase.data import chemical_symbols, covalent_radii
    from ase.data.colors import jmol_colors
    from quantem.core.utils.utils import electron_wavelength_angstrom
    from quantem.diffraction import bloch

    if crystal.g_vec is None or crystal.k_max is None or crystal.k_max < k_max:
        crystal.calculate_structure_factors(k_max=k_max)
    have_dyn = (
        getattr(crystal, "U_dyn", None) is not None
        and abs(getattr(crystal, "dyn_energy_ev", -1) - energy_ev) < 1
        and getattr(crystal, "dyn_k_max", 0) >= k_max
    )
    if not have_dyn:
        # Without Weickenmeier-Kohl factors (e.g. an element missing from the
        # table) fall back to the elastic couplings gamma F_g / pi.
        try:
            crystal.calculate_dynamical_structure_factors(energy_ev=energy_ev, k_max=k_max)
            have_dyn = True
        except Exception:  # noqa: BLE001
            have_dyn = False
    hkl_u, g_u = bloch._beam_universe(crystal)
    keep = torch.linalg.norm(g_u, dim=1) <= k_max
    hkl = hkl_u[keep]
    gamma_rel = bloch.relativistic_gamma(energy_ev)
    U_g, u0_imag = _coupling_vector(crystal, hkl, gamma_rel)
    # kinematical |F|^2 (Lobato) on the same list
    lut = {tuple(h): i for i, h in enumerate(crystal.hkl.tolist())}
    F2 = torch.zeros(hkl.shape[0], dtype=torch.float64)
    for i, h in enumerate(hkl.tolist()):
        j = lut.get(tuple(h))
        if j is not None:
            F2[i] = crystal.struct_factors_int[j]
    numbers = crystal.numbers.numpy()
    hkl_i16 = np.ascontiguousarray(hkl.numpy().astype(np.int16))
    return {
        "name": crystal.name,
        "spacegroup": getattr(crystal, "spacegroup", ""),
        "pointgroup": getattr(crystal, "pointgroup", ""),
        "cell": crystal.lat_real.numpy().tolist(),
        "recip": crystal.lat_recip.numpy().tolist(),
        "positions_frac": crystal.positions_frac.numpy().tolist(),
        "numbers": numbers.tolist(),
        "symbols": [chemical_symbols[int(z)] for z in numbers],
        "colors": [jmol_colors[int(z)].tolist() for z in numbers],
        "radii": [float(covalent_radii[int(z)]) for z in numbers],
        "hkl_i16": base64.b64encode(hkl_i16.tobytes()).decode(),
        "F2": _f32_b64(F2.numpy()),
        "U_re": _f32_b64(U_g.real.numpy()),
        "U_im": _f32_b64(U_g.imag.numpy()),
        "u0_imag": float(u0_imag),
        "absorptive": bool(have_dyn),
        "n_reflections": int(hkl.shape[0]),
        "energy_ev": float(energy_ev),
        "wavelength": float(electron_wavelength_angstrom(energy_ev)),
        "k_max": float(k_max),
        "hexagonal": bool(getattr(crystal, "hexagonal_matching", False)),
    }


def _coupling_vector(crystal, hkl: torch.Tensor, gamma_rel: float) -> tuple[torch.Tensor, float]:
    """Return ``U_g`` for a list of hkl and the mean absorption ``U_000''``.

    ``U_g`` is zero where no factor is stored. The factor choice matches
    ``bloch._coupling_matrix``, with a vector lookup instead of the (N, N)
    difference matrix.
    """
    if getattr(crystal, "U_dyn", None) is not None:
        hkl_all, U_all = crystal.hkl_dyn, crystal.U_dyn
    else:
        hkl_all, U_all = crystal.hkl, crystal.struct_factors * (gamma_rel / np.pi)
    lut = {tuple(h): i for i, h in enumerate(hkl_all.tolist())}
    idx = torch.tensor([lut.get(tuple(h), -1) for h in hkl.tolist()], dtype=torch.long)
    U = torch.zeros(hkl.shape[0], dtype=torch.complex128)
    has = idx >= 0
    U[has] = U_all[idx[has]]
    i0 = lut.get((0, 0, 0), -1)
    u0_imag = (
        float(U_all[i0].imag) if (i0 >= 0 and getattr(crystal, "U_dyn", None) is not None) else 0.0
    )
    return U, u0_imag


def prepare_kossel_reference(
    crystal,
    energy_ev: float,
    thicknesses_A,
    angle_step_mrad: float = 3.0,
    k_max: float = 1.0,
) -> dict[str, Any]:
    """Compute the bright-field (Kossel) reference on the Lambert grid.

    The browser looks this pattern up for the pixel rendering of the Kikuchi
    mode.

    Parameters
    ----------
    crystal : quantem.diffraction.crystal.Crystal
        The crystal.
    energy_ev : float
        Beam energy in eV.
    thicknesses_A : sequence of float
        Specimen thicknesses in Å; the browser interpolates between them.
    angle_step_mrad : float, default=3.0
        Angular step of the grid in mrad.
    k_max : float, default=1.0
        Largest scattering vector of the Bloch calculation, in 1/Å.

    Returns
    -------
    dict
        ``shape``, ``step``, ``thicknesses`` and the float32 ``data`` (base64).
    """
    require_crystal_tools()
    from quantem.diffraction import bloch

    master = bloch.calculate_kossel_reference(
        crystal,
        list(thicknesses_A),
        energy_ev=energy_ev,
        angle_step_mrad=angle_step_mrad,
        sg_max=0.05,
        k_max=k_max,
        progress_bar=False,
    )
    lam = np.nan_to_num(master["lambert"], nan=float(np.nanmax(master["lambert"])))
    return {
        "shape": list(lam.shape),
        "step": float(master["step"]),
        "thicknesses": [float(t) for t in master["thicknesses"]],
        "data": _f32_b64(lam),
    }


__all__ = [
    "PRESETS",
    "prepare_crystal",
    "prepare_kossel_reference",
    "preset_atoms",
    "preset_crystal",
    "require_crystal_tools",
]
