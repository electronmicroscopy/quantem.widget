"""Crystal phase model for diffraction indexing.

Lattice parameters plus a systematic-absence rule give allowed reflections,
d-spacings, and inter-plane angles for indexing measured spots and rings. A
phase can also be built from a bare d-spacing table when only reference
spacings are available.
"""

import math
import pathlib
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pymatgen.core import Structure


def _allow_all(*_) -> bool:
    return True


def _allow_fcc(h: int, k: int, ell: int) -> bool:
    return (h % 2) == (k % 2) == (ell % 2)


def _allow_bcc(h: int, k: int, ell: int) -> bool:
    return (h + k + ell) % 2 == 0


def _allow_diamond(h: int, k: int, ell: int) -> bool:
    return (h % 2 == k % 2 == ell % 2 == 1) or (
        h % 2 == k % 2 == ell % 2 == 0 and (h + k + ell) % 4 == 0
    )


def _allow_hcp(h: int, k: int, ell: int) -> bool:
    return not ((h + 2 * k) % 3 == 0 and ell % 2 == 1)


def _allow_wurtzite(h: int, k: int, ell: int) -> bool:
    hh_family = h == k or k == -(h + k) or h == -(h + k)
    return not (hh_family and ell % 2 == 1)


def _allow_rhombohedral(h: int, k: int, ell: int) -> bool:
    return (-h + k + ell) % 3 == 0


_ABSENCE_RULES = {
    "none": _allow_all,
    "fcc": _allow_fcc,
    "bcc": _allow_bcc,
    "diamond": _allow_diamond,
    "hcp": _allow_hcp,
    "wurtzite": _allow_wurtzite,
    "rhombohedral": _allow_rhombohedral,
}

# Built-in standards
PHASE_LIBRARY = {
    # fcc metals
    "Au": {"a": 4.0782, "absences": "fcc"},
    "Ag": {"a": 4.0853, "absences": "fcc"},
    "Al": {"a": 4.0495, "absences": "fcc"},
    "Cu": {"a": 3.6149, "absences": "fcc"},
    "Ni": {"a": 3.5238, "absences": "fcc"},
    "Pt": {"a": 3.9242, "absences": "fcc"},
    "Pd": {"a": 3.8907, "absences": "fcc"},
    "Pb": {"a": 4.9508, "absences": "fcc"},
    "Ir": {"a": 3.8390, "absences": "fcc"},
    "Rh": {"a": 3.8034, "absences": "fcc"},
    # bcc metals
    "α-Fe": {"a": 2.8665, "absences": "bcc"},
    "W": {"a": 3.1652, "absences": "bcc"},
    "Cr": {"a": 2.8848, "absences": "bcc"},
    "Mo": {"a": 3.1472, "absences": "bcc"},
    "Nb": {"a": 3.3004, "absences": "bcc"},
    "Ta": {"a": 3.3058, "absences": "bcc"},
    "V": {"a": 3.0240, "absences": "bcc"},
    # diamond cubic
    "Si": {"a": 5.4310, "absences": "diamond"},
    "Ge": {"a": 5.6575, "absences": "diamond"},
    "C (diamond)": {"a": 3.5668, "absences": "diamond"},
    "α-Sn": {"a": 6.4892, "absences": "diamond"},
    # rocksalt
    "MgO": {"a": 4.2117, "absences": "fcc"},
    "NaCl": {"a": 5.6402, "absences": "fcc"},
    "LiF": {"a": 4.0270, "absences": "fcc"},
    "TiN": {"a": 4.2400, "absences": "fcc"},
    "TiC": {"a": 4.3280, "absences": "fcc"},
    "NiO": {"a": 4.1770, "absences": "fcc"},
    # fluorite
    "CaF2": {"a": 5.4626, "absences": "fcc"},
    "CeO2": {"a": 5.4113, "absences": "fcc"},
    "UO2": {"a": 5.4704, "absences": "fcc"},
    # zincblende
    "GaAs": {"a": 5.6533, "absences": "fcc"},
    "GaP": {"a": 5.4505, "absences": "fcc"},
    "InP": {"a": 5.8687, "absences": "fcc"},
    "InAs": {"a": 6.0583, "absences": "fcc"},
    "ZnS": {"a": 5.4109, "absences": "fcc"},
    "ZnSe": {"a": 5.6676, "absences": "fcc"},
    "CdTe": {"a": 6.4820, "absences": "fcc"},
    "3C-SiC": {"a": 4.3596, "absences": "fcc"},
    # spinel
    "Fe3O4": {"a": 8.3963, "absences": "fcc"},
    "γ-Fe2O3": {"a": 8.3515, "absences": "fcc"},
    "MgAl2O4": {"a": 8.0831, "absences": "fcc"},
    # primitive cubic
    "SrTiO3": {"a": 3.9050, "absences": "none"},
    "CsCl": {"a": 4.1230, "absences": "none"},
    # additional cubic phases
    "Th": {"a": 5.0842, "absences": "fcc"},
    "KCl": {"a": 6.2917, "absences": "fcc"},
    "KBr": {"a": 6.6000, "absences": "fcc"},
    "CoO": {"a": 4.2603, "absences": "fcc"},
    "MnO": {"a": 4.4448, "absences": "fcc"},
    "PbS": {"a": 5.9362, "absences": "fcc"},
    "PbSe": {"a": 6.1243, "absences": "fcc"},
    "PbTe": {"a": 6.4620, "absences": "fcc"},
    "AgCl": {"a": 5.5491, "absences": "fcc"},
    "AgBr": {"a": 5.7745, "absences": "fcc"},
    "ThO2": {"a": 5.5997, "absences": "fcc"},
    "BaF2": {"a": 6.2001, "absences": "fcc"},
    "SrF2": {"a": 5.7996, "absences": "fcc"},
    "AlAs": {"a": 5.6605, "absences": "fcc"},
    "GaSb": {"a": 6.0959, "absences": "fcc"},
    "InSb": {"a": 6.4794, "absences": "fcc"},
    "ZnTe": {"a": 6.1034, "absences": "fcc"},
    "c-BN": {"a": 3.6157, "absences": "fcc"},
    "γ-Al2O3": {"a": 7.9110, "absences": "fcc"},
    "Y2O3": {"a": 10.6040, "absences": "bcc"},
    "In2O3": {"a": 10.1170, "absences": "bcc"},
    "LaB6": {"a": 4.1569, "absences": "none"},
    "Cu2O": {"a": 4.2696, "absences": "none"},
    # tetragonal
    "TiO2 (rutile)": {"a": 4.5937, "c": 2.9587, "gamma": 90.0, "absences": "none"},
    "TiO2 (anatase)": {"a": 3.7852, "c": 9.5139, "gamma": 90.0, "absences": "bcc"},
    "SnO2": {"a": 4.7382, "c": 3.1871, "gamma": 90.0, "absences": "none"},
    "β-Sn": {"a": 5.8318, "c": 3.1819, "gamma": 90.0, "absences": "bcc"},
    # wurtzite
    "ZnO": {"a": 3.2495, "c": 5.2069, "gamma": 120.0, "absences": "wurtzite"},
    "GaN": {"a": 3.1890, "c": 5.1850, "gamma": 120.0, "absences": "wurtzite"},
    "AlN": {"a": 3.1110, "c": 4.9790, "gamma": 120.0, "absences": "wurtzite"},
    "InN": {"a": 3.5378, "c": 5.7033, "gamma": 120.0, "absences": "wurtzite"},
    "CdS (wurtzite)": {"a": 4.1365, "c": 6.7160, "gamma": 120.0, "absences": "wurtzite"},
    "CdSe (wurtzite)": {"a": 4.2985, "c": 7.0150, "gamma": 120.0, "absences": "wurtzite"},
    # rhombohedral
    "α-Al2O3": {"a": 4.7587, "c": 12.9929, "gamma": 120.0, "absences": "rhombohedral"},
    "α-Fe2O3 (hematite)": {"a": 5.0356, "c": 13.7489, "gamma": 120.0, "absences": "rhombohedral"},
    "Cr2O3": {"a": 4.9587, "c": 13.5942, "gamma": 120.0, "absences": "rhombohedral"},
    "CaCO3 (calcite)": {"a": 4.9890, "c": 17.0620, "gamma": 120.0, "absences": "rhombohedral"},
    "LiNbO3": {"a": 5.1483, "c": 13.8631, "gamma": 120.0, "absences": "rhombohedral"},
    # hcp metals + graphite
    "Ti": {"a": 2.9505, "c": 4.6826, "gamma": 120.0, "absences": "hcp"},
    "Zn": {"a": 2.6649, "c": 4.9468, "gamma": 120.0, "absences": "hcp"},
    "Mg": {"a": 3.2094, "c": 5.2107, "gamma": 120.0, "absences": "hcp"},
    "Co": {"a": 2.5071, "c": 4.0695, "gamma": 120.0, "absences": "hcp"},
    "Zr": {"a": 3.2320, "c": 5.1470, "gamma": 120.0, "absences": "hcp"},
    "Ru": {"a": 2.7059, "c": 4.2815, "gamma": 120.0, "absences": "hcp"},
    "Be": {"a": 2.2858, "c": 3.5843, "gamma": 120.0, "absences": "hcp"},
    "Cd": {"a": 2.9793, "c": 5.6196, "gamma": 120.0, "absences": "hcp"},
    "Re": {"a": 2.7610, "c": 4.4560, "gamma": 120.0, "absences": "hcp"},
    "Os": {"a": 2.7344, "c": 4.3174, "gamma": 120.0, "absences": "hcp"},
    "Hf": {"a": 3.1946, "c": 5.0511, "gamma": 120.0, "absences": "hcp"},
    "Y": {"a": 3.6474, "c": 5.7306, "gamma": 120.0, "absences": "hcp"},
    "C (graphite)": {"a": 2.4610, "c": 6.7080, "gamma": 120.0, "absences": "hcp"},
}


def library_phase(name: str) -> "Phase":
    """Build a :class:`Phase` from the built-in standards library."""
    if name not in PHASE_LIBRARY:
        raise ValueError(f"unknown library phase {name!r}; available: {sorted(PHASE_LIBRARY)}")
    entry = PHASE_LIBRARY[name]
    a, absences = entry["a"], entry["absences"]
    if "c" in entry:
        gamma = entry.get("gamma", 90.0)
        return Phase(name, a, a, entry["c"], 90.0, 90.0, gamma, absences=absences)
    return Phase.from_cubic(name, a, absences=absences)


def _format_hkl(hkl: Sequence[float]) -> str:
    indices = tuple(int(i) for i in hkl)
    if all(0 <= i < 10 for i in indices):
        return "".join(str(i) for i in indices)
    return "(" + ",".join(str(i) for i in indices) + ")"


def _parse_hkl_label(label: str) -> tuple[int, int, int] | None:
    body = label.strip().strip("()")
    try:
        if "," in body:
            parts = [int(p) for p in body.split(",")]
        else:
            parts = [int(c) for c in body]
    except ValueError:
        return None
    return tuple(parts) if len(parts) == 3 else None


def _canonical_hkl(hkl: Sequence[float]) -> tuple[int, int, int]:
    return tuple(sorted((abs(int(i)) for i in hkl), reverse=True))


class Phase:
    """A crystalline phase: lattice parameters (Å, degrees) + absence rule for
    geometry-aware indexing, or a reference d-spacing card for pure matching.
    """

    def __init__(
        self,
        name: str,
        a: float,
        b: float,
        c: float,
        alpha: float = 90.0,
        beta: float = 90.0,
        gamma: float = 90.0,
        absences: str = "none",
        _reference_lines: list[tuple[float, str, float | None]] | None = None,
    ) -> None:
        self.name = name
        self.absences = absences
        if absences not in _ABSENCE_RULES:
            raise ValueError(f"unknown absence rule {absences!r}; use {list(_ABSENCE_RULES)}")
        self._allowed_rule = _ABSENCE_RULES[absences]
        self._reference_lines = _reference_lines
        self._reflection_cache: list[dict] | None = None

        if _reference_lines is None:
            if min(a, b, c) <= 0:
                raise ValueError("lattice edge lengths must be positive")
            self.lattice = (float(a), float(b), float(c), float(alpha), float(beta), float(gamma))
            ca, cb, cg = (math.cos(math.radians(x)) for x in (alpha, beta, gamma))
            g = np.array(
                [
                    [a * a, a * b * cg, a * c * cb],
                    [a * b * cg, b * b, b * c * ca],
                    [a * c * cb, b * c * ca, c * c],
                ],
                dtype=np.float64,
            )
            self._g_star = np.linalg.inv(g)
        else:
            self.lattice = None
            self._g_star = None

    # --- Constructors ---
    @classmethod
    def from_cubic(cls, name: str, a: float, absences: str = "fcc") -> "Phase":
        """Cubic phase with edge ``a`` (Å) and a systematic-absence rule."""
        return cls(name, a, a, a, 90.0, 90.0, 90.0, absences=absences)

    @classmethod
    def from_dspacings(cls, name: str, entries: Iterable[Sequence]) -> "Phase":
        """Phase from reference entries: ``(d_Å, hkl_label[, intensity])``."""
        reference_lines = []
        for entry in entries:
            spacing, label = entry[0], entry[1]
            intensity = float(entry[2]) if len(entry) > 2 else None
            reference_lines.append((float(spacing), str(label), intensity))
        return cls(name, 1.0, 1.0, 1.0, _reference_lines=reference_lines)

    @classmethod
    def from_structure(
        cls, structure: "Structure", name: str | None = None, d_min: float = 0.7
    ) -> "Phase":
        """Phase from a pymatgen Structure using electron intensities."""
        from .phasedb import structure_reflections

        entries = [
            (r["d"], r["hkl_str"], r["intensity"])
            for r in structure_reflections(structure, d_min=d_min)
        ]
        if name is None:
            name = structure.composition.reduced_formula
        return cls.from_dspacings(name, entries)

    @classmethod
    def from_cif(
        cls, path: str | pathlib.Path, name: str | None = None, d_min: float = 0.7
    ) -> "Phase":
        """Phase from a CIF file (via pymatgen ``Structure.from_file``)."""
        try:
            from pymatgen.core import Structure
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Phase.from_cif needs pymatgen; install it with "
                "'pip install \"quantem.widget[phaseid]\"'"
            ) from exc
        return cls.from_structure(Structure.from_file(str(path)), name=name, d_min=d_min)

    # --- Geometry ---
    def d_spacing(self, hkl: Sequence[float]) -> float:
        """Interplanar spacing d_hkl in Å."""
        if self._g_star is None:
            raise ValueError("d_spacing requires a lattice-based Phase, not a d-spacing table")
        indices = np.asarray(hkl, dtype=np.float64)
        inverse_d_squared = float(indices @ self._g_star @ indices)
        if inverse_d_squared <= 0:
            raise ValueError("invalid reflection (000)")
        return 1.0 / math.sqrt(inverse_d_squared)

    def plane_angle(self, hkl1: Sequence[float], hkl2: Sequence[float]) -> float:
        """Angle in degrees between plane normals (hkl1) and (hkl2)."""
        if self._g_star is None:
            raise ValueError("plane_angle requires a lattice-based Phase, not a d-spacing table")
        indices1 = np.asarray(hkl1, dtype=np.float64)
        indices2 = np.asarray(hkl2, dtype=np.float64)
        numerator = float(indices1 @ self._g_star @ indices2)
        denominator = math.sqrt(
            float(indices1 @ self._g_star @ indices1) * float(indices2 @ self._g_star @ indices2)
        )
        if denominator == 0:
            return 0.0
        return math.degrees(math.acos(max(-1.0, min(1.0, numerator / denominator))))

    def is_allowed(self, hkl: Sequence[float]) -> bool:
        """Whether (hkl) is a non-origin reflection permitted by the absence rule."""
        h, k, ell = (int(i) for i in hkl)
        if h == 0 and k == 0 and ell == 0:
            return False
        return bool(self._allowed_rule(h, k, ell))

    # --- Reflections ---
    def reflections(self, d_min: float = 0.5, max_index: int | None = None) -> list[dict]:
        """Allowed reflection families, largest d first.

        By default ``max_index`` is sized so every family above ``d_min`` is
        enumerated.
        """
        if self._reference_lines is not None:
            reflections = [
                {
                    "hkl": _parse_hkl_label(label),
                    "hkl_str": label,
                    "d": spacing,
                    "multiplicity": None,
                    "intensity": intensity,
                }
                for spacing, label, intensity in self._reference_lines
                if spacing >= d_min
            ]
            return sorted(reflections, key=lambda reflection: -reflection["d"])

        if max_index is None:
            max_index = math.ceil(max(self.lattice[:3]) / d_min)
        families_by_d: dict[int, dict] = {}
        for h in range(-max_index, max_index + 1):
            for k in range(-max_index, max_index + 1):
                for ell in range(-max_index, max_index + 1):
                    hkl = (h, k, ell)
                    if not self.is_allowed(hkl):
                        continue
                    spacing = self.d_spacing(hkl)
                    if spacing < d_min:
                        continue
                    d_key = int(round(spacing * 1e4))
                    representative = _canonical_hkl(hkl)
                    family = families_by_d.get(d_key)
                    if family is None:
                        families_by_d[d_key] = {
                            "hkl": representative,
                            "hkl_str": _format_hkl(representative),
                            "d": spacing,
                            "multiplicity": 1,
                            "intensity": None,
                        }
                    else:
                        family["multiplicity"] += 1
                        if representative < family["hkl"]:
                            family["hkl"] = representative
                            family["hkl_str"] = _format_hkl(representative)
        return sorted(families_by_d.values(), key=lambda reflection: -reflection["d"])

    def _all_reflections(self) -> list[dict]:
        if self._reflection_cache is None:
            self._reflection_cache = self.reflections()
        return self._reflection_cache

    def match_d(self, d: float, tol: float = 0.03) -> list[dict]:
        """Reflections within fractional ``tol`` of ``d``, closest first."""
        if d <= 0:
            return []
        matches = []
        for reflection in self._all_reflections():
            error = abs(reflection["d"] - d) / d
            if error <= tol:
                matches.append({**reflection, "d_error": error})
        return sorted(matches, key=lambda reflection: reflection["d_error"])
