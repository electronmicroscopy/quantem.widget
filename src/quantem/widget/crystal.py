"""Crystal phase model for diffraction indexing.

Lattice parameters plus a systematic-absence rule give allowed reflections,
d-spacings, and inter-plane angles for indexing measured spots and rings. A
phase can also be built from a bare d-spacing table when only reference
spacings are available.
"""

import math
from collections.abc import Iterable, Sequence

import numpy as np


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

# Built-in standards: room-temperature lattice parameters in Å, each taken from
# the license-clean source cited on its line (non-cubic entries: line above) —
# NIST SRM certificates and NBS circulars/monographs (US public domain), the
# Crystallography Open Database (COD, CC0), or primary literature (numerical
# facts, cited for provenance). No values from proprietary compilations
# (ICDD PDF, Pearson's Handbook).
PHASE_LIBRARY = {
    # fcc metals
    "Au": {"a": 4.0782, "absences": "fcc"},  # COD 9008463 (Wyckoff 1963)
    "Ag": {"a": 4.0855, "absences": "fcc"},  # COD 1100136 (Spreadborough & Christian 1959)
    "Al": {"a": 4.0494, "absences": "fcc"},  # NBS Circ. 539 v1 (Swanson & Tatge 1953)
    "Cu": {"a": 3.6149, "absences": "fcc"},  # Lu & Chang 1941 (NBS Circ. 539 v1)
    "Ni": {"a": 3.5238, "absences": "fcc"},  # NBS Circ. 539 v1 (Swanson & Tatge 1953)
    "Pt": {"a": 3.9236, "absences": "fcc"},  # Arblaster 1997, Platin. Met. Rev. 41 12
    "Pd": {"a": 3.8902, "absences": "fcc"},  # Arblaster 2012, Platin. Met. Rev. 56 181
    "Pb": {"a": 4.9508, "absences": "fcc"},  # Klug 1946 (NBS Circ. 539 v1)
    "Ir": {"a": 3.8392, "absences": "fcc"},  # Arblaster 2010, Platin. Met. Rev. 54 93
    "Rh": {"a": 3.8034, "absences": "fcc"},  # Arblaster 1997, Platin. Met. Rev. 41 184
    # bcc metals
    "α-Fe": {"a": 2.8665, "absences": "bcc"},  # COD 9008536 (Wyckoff 1963)
    "W": {"a": 3.1652, "absences": "bcc"},  # NBS Mono. 25 Sec. 13 (internal standard)
    "Cr": {"a": 2.8839, "absences": "bcc"},  # NBS Circ. 539 v5 (1955), COD 5000220
    "Mo": {"a": 3.1472, "absences": "bcc"},  # NBS Circ. 539 v1 (Swanson & Tatge 1953)
    "Nb": {"a": 3.3004, "absences": "bcc"},  # COD 9008546 (Wyckoff 1963)
    "Ta": {"a": 3.3058, "absences": "bcc"},  # COD 9008552 (Wyckoff 1963)
    "V": {"a": 3.0241, "absences": "bcc"},  # COD 9012770 (James & Straumanis 1960)
    # diamond cubic
    "Si": {"a": 5.4311, "absences": "diamond"},  # NIST SRM 640f
    "Ge": {"a": 5.6578, "absences": "diamond"},  # COD 9011999 (Hom et al. 1975)
    "C (diamond)": {"a": 3.5668, "absences": "diamond"},  # COD 9008564 (Wyckoff 1963)
    "α-Sn": {"a": 6.4912, "absences": "diamond"},  # COD 9008568 (Wyckoff 1963)
    # rocksalt
    "MgO": {"a": 4.2117, "absences": "fcc"},  # NBS Circ. 539 v1 (Swanson & Tatge 1953)
    "NaCl": {"a": 5.6402, "absences": "fcc"},  # NBS Circ. 539 v2 (Swanson & Fuyat 1953)
    "LiF": {"a": 4.0270, "absences": "fcc"},  # NBS Circ. 539 v1 (Swanson & Tatge 1953)
    "TiN": {"a": 4.2390, "absences": "fcc"},  # COD 1100037 (Christensen 1978)
    "TiC": {"a": 4.3280, "absences": "fcc"},  # COD 9012564 (Christensen 1978)
    "NiO": {"a": 4.1771, "absences": "fcc"},  # COD 4329325 (Malingowski et al. 2012)
    "CaO": {"a": 4.8107, "absences": "fcc"},  # COD 7200686 (Verbraeken et al. 2009)
    "ZrN": {"a": 4.5780, "absences": "fcc"},  # COD 1538058 (Gatterer et al. 1975)
    "CrN": {"a": 4.1480, "absences": "fcc"},  # COD 1008956 (Nasr Eddine et al. 1977)
    "TaC": {"a": 4.4540, "absences": "fcc"},  # COD 9008731 (Wyckoff 1963)
    "NbC": {"a": 4.4691, "absences": "fcc"},  # COD 9008682 (Wyckoff 1963)
    "ZrC": {"a": 4.7004, "absences": "fcc"},  # COD 1562921 (Chinthaka Silva et al. 2012)
    # fluorite
    "CaF2": {"a": 5.4630, "absences": "fcc"},  # COD 9009005 (Wyckoff 1963)
    "CeO2": {"a": 5.4115, "absences": "fcc"},  # NIST SRM 674b
    "UO2": {"a": 5.4704, "absences": "fcc"},  # Grønvold 1955, J. Inorg. Nucl. Chem. 1 357
    # zincblende
    "GaAs": {"a": 5.6533, "absences": "fcc"},  # Straumanis & Kim 1965, J. Appl. Phys. 36 3822
    "GaP": {"a": 5.4505, "absences": "fcc"},  # COD 9008846 (Wyckoff 1963)
    "InP": {"a": 5.8687, "absences": "fcc"},  # COD 9008852 (Wyckoff 1963)
    "InAs": {"a": 6.0580, "absences": "fcc"},  # NBS Mono. 25 Sec. 3 (1964)
    "ZnS": {"a": 5.4093, "absences": "fcc"},  # COD 9000107 (Skinner 1961)
    "ZnSe": {"a": 5.6676, "absences": "fcc"},  # COD 9008857 (Wyckoff 1963)
    "CdTe": {"a": 6.4810, "absences": "fcc"},  # NBS Mono. 25 Sec. 3 (1964)
    "3C-SiC": {"a": 4.3596, "absences": "fcc"},  # Sultan et al. 2022, Materials 15 6229
    "CuI": {"a": 6.0630, "absences": "fcc"},  # COD 9004456 (Cooper & Hawthorne 1997)
    # spinel
    "Fe3O4": {"a": 8.3967, "absences": "fcc"},  # COD 9013529 (Bosi et al. 2009)
    "γ-Fe2O3": {"a": 8.3474, "absences": "fcc"},  # COD 9017489 (Shmakov et al. 1995)
    "MgAl2O4": {"a": 8.0836, "absences": "fcc"},  # COD 9002044 (Redfern et al. 1999)
    "Co3O4": {"a": 8.0821, "absences": "fcc"},  # COD 9005887 (Liu & Prewitt 1990)
    "CoFe2O4": {"a": 8.3806, "absences": "fcc"},  # COD 1533163 (Ferreira et al. 2003)
    "NiFe2O4": {"a": 8.3597, "absences": "fcc"},  # COD 2300289 (Kremenović et al. 2010)
    "ZnFe2O4": {"a": 8.4421, "absences": "fcc"},  # COD 9005102 (O'Neill 1992)
    # primitive cubic
    "SrTiO3": {"a": 3.9050, "absences": "none"},  # NBS Circ. 539 v3 (Swanson et al. 1954)
    "CsCl": {"a": 4.1230, "absences": "none"},  # NBS Circ. 539 v2 (Swanson & Fuyat 1953)
    # additional cubic phases
    "Th": {"a": 5.0843, "absences": "fcc"},  # COD 9008485 (Wyckoff 1963)
    "KCl": {"a": 6.2917, "absences": "fcc"},  # NBS Circ. 539 v1 (Swanson & Tatge 1953)
    "KBr": {"a": 6.6000, "absences": "fcc"},  # COD 9008650 (Wyckoff 1963)
    "CoO": {"a": 4.2630, "absences": "fcc"},  # COD 1533087 (Sasaki et al. 1979)
    "MnO": {"a": 4.4449, "absences": "fcc"},  # COD 9005946 (Pacalo & Graham 1991)
    "PbS": {"a": 5.9362, "absences": "fcc"},  # NBS Circ. 539 v2 (Swanson & Fuyat 1953)
    "PbSe": {"a": 6.1243, "absences": "fcc"},  # COD 9008695 (Wyckoff 1963)
    "PbTe": {"a": 6.4541, "absences": "fcc"},  # COD 9011358 (Noda et al. 1987)
    "AgCl": {"a": 5.5491, "absences": "fcc"},  # NBS Circ. 539 v4 (Swanson et al. 1955)
    "AgBr": {"a": 5.7745, "absences": "fcc"},  # NBS Circ. 539 v4 (Swanson et al. 1955)
    "ThO2": {"a": 5.5997, "absences": "fcc"},  # COD 9009046 (Wyckoff 1963)
    "BaF2": {"a": 6.2001, "absences": "fcc"},  # COD 9009004 (Wyckoff 1963)
    "SrF2": {"a": 5.7996, "absences": "fcc"},  # COD 9009043 (Wyckoff 1963)
    "AlAs": {"a": 5.6608, "absences": "fcc"},  # COD 1540257 (Leszczynski et al. 1992)
    "GaSb": {"a": 6.0959, "absences": "fcc"},  # Straumanis & Kim 1965, J. Appl. Phys. 36 3822
    "InSb": {"a": 6.4794, "absences": "fcc"},  # Straumanis & Kim 1965, J. Appl. Phys. 36 3822
    "ZnTe": {"a": 6.1026, "absences": "fcc"},  # COD 1540103 (Holland & Beck 1968)
    "c-BN": {"a": 3.6153, "absences": "fcc"},  # Kurdyumov et al. 1995, J. Appl. Cryst. 28 540
    "γ-Al2O3": {"a": 7.9140, "absences": "fcc"},  # COD 2107301 (Zhou & Snyder 1991)
    "Y2O3": {"a": 10.6040, "absences": "bcc"},  # COD 1513300 (Ferreira et al. 2005)
    "In2O3": {"a": 10.1170, "absences": "bcc"},  # COD 2310009 (Marezio 1966)
    "LaB6": {"a": 4.1568, "absences": "none"},  # NIST SRM 660c
    "Cu2O": {"a": 4.2696, "absences": "none"},  # NBS Circ. 539 v2 (Swanson & Fuyat 1953)
    # tetragonal
    # NIST SRM 674b
    "TiO2 (rutile)": {"a": 4.5940, "c": 2.9589, "gamma": 90.0, "absences": "none"},
    # NBS Mono. 25 Sec. 7 (1969)
    "TiO2 (anatase)": {"a": 3.7852, "c": 9.5139, "gamma": 90.0, "absences": "bcc"},
    # COD 2101853 (Bolzan et al. 1997)
    "SnO2": {"a": 4.7374, "c": 3.1864, "gamma": 90.0, "absences": "none"},
    # COD 1534488 (Lee & Raynor 1954)
    "β-Sn": {"a": 5.8317, "c": 3.1813, "gamma": 90.0, "absences": "bcc"},
    # COD 1513252 (Yasuda et al. 2009)
    "BaTiO3": {"a": 3.9905, "c": 4.0412, "gamma": 90.0, "absences": "none"},
    # primitive hexagonal
    # COD 1501516 (Litasov et al. 2010)
    "WC": {"a": 2.9059, "c": 2.8377, "gamma": 120.0, "absences": "none"},
    # COD 2002799 (Möhr et al. 1996)
    "TiB2": {"a": 3.0292, "c": 3.2284, "gamma": 120.0, "absences": "none"},
    # wurtzite
    # NIST SRM 674b
    "ZnO": {"a": 3.2499, "c": 5.2067, "gamma": 120.0, "absences": "wurtzite"},
    # Detchprohm et al. 1992, Jpn. J. Appl. Phys. 31 L1454
    "GaN": {"a": 3.1892, "c": 5.1850, "gamma": 120.0, "absences": "wurtzite"},
    # Schulz & Thiemann 1977, Solid State Commun. 23 815
    "AlN": {"a": 3.1100, "c": 4.9800, "gamma": 120.0, "absences": "wurtzite"},
    # Paszkowicz 1999, Powder Diffr. 14 258
    "InN": {"a": 3.5378, "c": 5.7033, "gamma": 120.0, "absences": "wurtzite"},
    # COD 9011663 (Xu & Ching 1993)
    "CdS (wurtzite)": {"a": 4.1370, "c": 6.7144, "gamma": 120.0, "absences": "wurtzite"},
    # COD 9011664 (Xu & Ching 1993)
    "CdSe (wurtzite)": {"a": 4.2985, "c": 7.0152, "gamma": 120.0, "absences": "wurtzite"},
    # COD 1100044 (Kisi & Elcombe 1989)
    "ZnS (wurtzite)": {"a": 3.8227, "c": 6.2607, "gamma": 120.0, "absences": "wurtzite"},
    # COD 1529745 (Cava et al. 1977)
    "β-AgI": {"a": 4.5980, "c": 7.5140, "gamma": 120.0, "absences": "wurtzite"},
    # rhombohedral
    # NIST SRM 676a
    "α-Al2O3": {"a": 4.7594, "c": 12.9923, "gamma": 120.0, "absences": "rhombohedral"},
    # NBS Mono. 25 Sec. 18 (1981)
    "α-Fe2O3 (hematite)": {"a": 5.0356, "c": 13.7489, "gamma": 120.0, "absences": "rhombohedral"},
    # NIST SRM 674b
    "Cr2O3": {"a": 4.9586, "c": 13.5965, "gamma": 120.0, "absences": "rhombohedral"},
    # NBS Circ. 539 v2 (Swanson & Fuyat 1953)
    "CaCO3 (calcite)": {"a": 4.9890, "c": 17.0620, "gamma": 120.0, "absences": "rhombohedral"},
    # COD 1541936 (Abrahams et al. 1966)
    "LiNbO3": {"a": 5.1483, "c": 13.8631, "gamma": 120.0, "absences": "rhombohedral"},
    # COD 2310889 (Cucka & Barrett 1962)
    "Bi": {"a": 4.5460, "c": 11.8620, "gamma": 120.0, "absences": "rhombohedral"},
    # COD 5000214 (Barrett et al. 1963)
    "Sb": {"a": 4.3084, "c": 11.2740, "gamma": 120.0, "absences": "rhombohedral"},
    # hcp metals + graphite
    # NBS Circ. 539 v3 (Swanson et al. 1954)
    "Ti": {"a": 2.9500, "c": 4.6860, "gamma": 120.0, "absences": "hcp"},
    # Jette & Foote 1935 (NBS Circ. 539 v1)
    "Zn": {"a": 2.6649, "c": 4.9468, "gamma": 120.0, "absences": "hcp"},
    # NBS Circ. 539 v1 (Swanson & Tatge 1953)
    "Mg": {"a": 3.2094, "c": 5.2103, "gamma": 120.0, "absences": "hcp"},
    # COD 9008492 (Wyckoff 1963)
    "Co": {"a": 2.5071, "c": 4.0686, "gamma": 120.0, "absences": "hcp"},
    # NBS Circ. 539 v2 (Swanson & Fuyat 1953)
    "Zr": {"a": 3.2320, "c": 5.1470, "gamma": 120.0, "absences": "hcp"},
    # NBS Circ. 539 v4 (Swanson et al. 1955)
    "Ru": {"a": 2.7058, "c": 4.2819, "gamma": 120.0, "absences": "hcp"},
    # Mackay & Hill 1963 (NBS Mono. 25 Sec. 9)
    "Be": {"a": 2.2858, "c": 3.5843, "gamma": 120.0, "absences": "hcp"},
    # NBS Circ. 539 v3 (Swanson et al. 1954)
    "Cd": {"a": 2.9793, "c": 5.6181, "gamma": 120.0, "absences": "hcp"},
    # COD 9008512 (Wyckoff 1963)
    "Re": {"a": 2.7608, "c": 4.4582, "gamma": 120.0, "absences": "hcp"},
    # NBS Circ. 539 v4 (Swanson et al. 1955)
    "Os": {"a": 2.7341, "c": 4.3197, "gamma": 120.0, "absences": "hcp"},
    # Russell 1953 (COD 1539076)
    "Hf": {"a": 3.1964, "c": 5.0511, "gamma": 120.0, "absences": "hcp"},
    # Spedding et al. 1956 (COD 9010984)
    "Y": {"a": 3.6474, "c": 5.7306, "gamma": 120.0, "absences": "hcp"},
    # Trucano & Chen 1975 (COD 9011577)
    "C (graphite)": {"a": 2.4640, "c": 6.7110, "gamma": 120.0, "absences": "hcp"},
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
