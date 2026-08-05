"""
showdiffraction: Interactive d-spacing analysis for 2D/3D diffraction patterns.
"""

import csv
import json
import math
import pathlib
import re
import tempfile
import time
import warnings
from collections.abc import Iterable, Sequence
from typing import Self

import anywidget
import numpy as np
import torch
import traitlets
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.signal.windows import tukey

from quantem.widget.export import ensure_mobile_viewport
from quantem.widget.utils.array import to_numpy
from quantem.widget.utils.state_io import resolve_widget_version, save_state_file, unwrap_state_payload
from quantem.widget.utils.ui import UiMode, resolve_ui_mode


# Crystal phases
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


def _allow_rhombohedral(h: int, k: int, ell: int) -> bool:
    return (-h + k + ell) % 3 == 0


def _allow_rhombohedral_c(h: int, k: int, ell: int) -> bool:
    if (h == 0 or k == 0 or h == -k) and ell % 2 == 1:
        return False
    return _allow_rhombohedral(h, k, ell)


def _allow_spinel(h: int, k: int, ell: int) -> bool:
    if not _allow_fcc(h, k, ell):
        return False
    low, mid, high = sorted((abs(h), abs(k), abs(ell)))
    if mid == 0:  # h00
        return high % 4 == 0
    if low == 0:  # hk0
        return (mid + high) % 4 == 0
    return True


def _allow_i41amd(h: int, k: int, ell: int) -> bool:
    if (h + k + ell) % 2 != 0:
        return False
    if ell % 2 == 1:
        return True
    if ell % 4 == 0:  # includes hk0
        return h % 2 == 0
    return h % 2 == 1  # l = 4n+2


def _allow_rutile(h: int, k: int, ell: int) -> bool:
    if h == 0 and (k + ell) % 2 == 1:
        return False
    if k == 0 and (h + ell) % 2 == 1:
        return False
    return True


def _allow_bixbyite(h: int, k: int, ell: int) -> bool:
    if (h + k + ell) % 2 != 0:
        return False
    if h == 0 and (k % 2 == 1 or ell % 2 == 1):
        return False
    if k == 0 and (h % 2 == 1 or ell % 2 == 1):
        return False
    if ell == 0 and (h % 2 == 1 or k % 2 == 1):
        return False
    return True


def _allow_cuprite(h: int, k: int, ell: int) -> bool:
    if (h + k + ell) % 2 == 0:
        return True
    return h % 2 == k % 2 == ell % 2


_ABSENCE_RULES = {
    "none": _allow_all,
    "fcc": _allow_fcc,
    "bcc": _allow_bcc,
    "diamond": _allow_diamond,
    "hcp": _allow_hcp,
    "wurtzite": _allow_hcp,  # 2b wurtzite sites zero the same reflections as hcp
    "rhombohedral": _allow_rhombohedral,
    "rhombohedral-c": _allow_rhombohedral_c,
    "spinel": _allow_spinel,  # Fd-3m d-glide: h00 needs h=4n, hk0 needs h+k=4n
    "i41amd": _allow_i41amd,  # I4_1/amd 4a/8e sites: 002, 110, 222 absent
    "rutile": _allow_rutile,  # P4_2/mnm n-glide: 0kl needs k+l even
    "bixbyite": _allow_bixbyite,  # Ia-3 a-glide: 0kl needs k, l even
    "cuprite": _allow_cuprite,  # Pn-3m 2a/4b sites: mixed parity with odd sum absent
}

# Built-in standards: room-temperature lattice parameters in Å, cited per entry
# from license-clean sources (NIST SRM, NBS circulars/monographs, COD, or
# primary literature); no values from proprietary compilations. Non-cubic
# entries carry their source on the line above.
PHASE_LIBRARY = {
    # fcc metals
    "Au": {"a": 4.0782, "absences": "fcc"},  # COD 9008463 (Wyckoff 1963)
    "Ag": {"a": 4.0855, "absences": "fcc"},  # COD 1100136 (Spreadborough & Christian 1959)
    "Al": {"a": 4.0494, "absences": "fcc"},  # NBS Circ. 539 v1 (Swanson & Tatge 1953)
    "Cu": {"a": 3.6149, "absences": "fcc"},  # Lu & Chang 1941 (NBS Circ. 539 v1)
    "Ni": {"a": 3.5238, "absences": "fcc"},  # NBS Circ. 539 v1 (Swanson & Tatge 1953)
    "Pt": {"a": 3.9236, "absences": "fcc"},  # Arblaster 1997, Platin. Met. Rev. 41 12, doi:10.1595/003214097X4111221
    "Pd": {"a": 3.8902, "absences": "fcc"},  # Arblaster 2012, Platin. Met. Rev. 56 181, doi:10.1595/147106712X646113
    "Pb": {"a": 4.9508, "absences": "fcc"},  # Klug 1946 (NBS Circ. 539 v1)
    "Ir": {"a": 3.8392, "absences": "fcc"},  # Arblaster 2010, Platin. Met. Rev. 54 93, doi:10.1595/147106710X493124
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
    "MgO": {"a": 4.2130, "absences": "fcc"},  # NBS Circ. 539 v1 (Swanson & Tatge 1953)
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
    "UO2": {"a": 5.4704, "absences": "fcc"},  # Grønvold 1955, J. Inorg. Nucl. Chem. 1 357, doi:10.1016/0022-1902(55)80046-2
    # zincblende
    "GaAs": {"a": 5.6533, "absences": "fcc"},  # Straumanis & Kim 1965, J. Appl. Phys. 36 3822
    "GaP": {"a": 5.4505, "absences": "fcc"},  # COD 9008846 (Wyckoff 1963)
    "InP": {"a": 5.8687, "absences": "fcc"},  # COD 9008852 (Wyckoff 1963)
    "InAs": {"a": 6.0580, "absences": "fcc"},  # NBS Mono. 25 Sec. 3 (1964)
    "ZnS": {"a": 5.4093, "absences": "fcc"},  # COD 9000107 (Skinner 1961)
    "ZnSe": {"a": 5.6676, "absences": "fcc"},  # COD 9008857 (Wyckoff 1963)
    "CdTe": {"a": 6.4810, "absences": "fcc"},  # NBS Mono. 25 Sec. 3 (1964)
    "3C-SiC": {"a": 4.3596, "absences": "fcc"},  # Sultan et al. 2022, Materials 15 6229, doi:10.3390/ma15186229
    "CuI": {"a": 6.0630, "absences": "fcc"},  # COD 9004456 (Cooper & Hawthorne 1997)
    # spinel
    "Fe3O4": {"a": 8.3967, "absences": "spinel"},  # COD 9013529 (Bosi et al. 2009)
    "γ-Fe2O3": {"a": 8.3474, "absences": "spinel"},  # COD 9017489 (Shmakov et al. 1995)
    "MgAl2O4": {"a": 8.0836, "absences": "spinel"},  # COD 9002044 (Redfern et al. 1999)
    "Co3O4": {"a": 8.0821, "absences": "spinel"},  # COD 9005887 (Liu & Prewitt 1990)
    "CoFe2O4": {"a": 8.3806, "absences": "spinel"},  # COD 1533163 (Ferreira et al. 2003)
    "NiFe2O4": {"a": 8.3390, "absences": "spinel"},  # Hill et al. 1979, Phys. Chem. Miner. 4 317, doi:10.1007/BF00307535
    "ZnFe2O4": {"a": 8.4421, "absences": "spinel"},  # COD 9005102 (O'Neill 1992)
    # primitive cubic
    "SrTiO3": {"a": 3.9050, "absences": "none"},  # NBS Circ. 539 v3 (Swanson et al. 1954)
    "CsCl": {"a": 4.1230, "absences": "none"},  # NBS Circ. 539 v2 (Swanson & Fuyat 1953)
    # additional cubic phases
    "Th": {"a": 5.0843, "absences": "fcc"},  # COD 9008485 (Wyckoff 1963)
    "KCl": {"a": 6.2931, "absences": "fcc"},  # NBS Circ. 539 v1 (Swanson & Tatge 1953)
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
    "c-BN": {"a": 3.6153, "absences": "fcc"},  # Kurdyumov et al. 1995, J. Appl. Cryst. 28 540, doi:10.1107/S002188989500197X
    "γ-Al2O3": {"a": 7.9140, "absences": "spinel"},  # COD 2107301 (Zhou & Snyder 1991)
    "Y2O3": {"a": 10.6040, "absences": "bixbyite"},  # COD 1513300 (Ferreira et al. 2005)
    "In2O3": {"a": 10.1170, "absences": "bixbyite"},  # COD 2310009 (Marezio 1966)
    "LaB6": {"a": 4.1568, "absences": "none"},  # NIST SRM 660c
    "Cu2O": {"a": 4.2696, "absences": "cuprite"},  # NBS Circ. 539 v2 (Swanson & Fuyat 1953)
    # tetragonal
    # NIST SRM 674b
    "TiO2 (rutile)": {"a": 4.5940, "c": 2.9589, "gamma": 90.0, "absences": "rutile"},
    # NBS Mono. 25 Sec. 7 (1969)
    "TiO2 (anatase)": {"a": 3.7852, "c": 9.5139, "gamma": 90.0, "absences": "i41amd"},
    # COD 2101853 (Bolzan et al. 1997)
    "SnO2": {"a": 4.7374, "c": 3.1864, "gamma": 90.0, "absences": "rutile"},
    # COD 1534488 (Lee & Raynor 1954)
    "β-Sn": {"a": 5.8317, "c": 3.1813, "gamma": 90.0, "absences": "i41amd"},
    # NBS Circ. 539 v3 (Swanson & Fuyat 1954)
    "BaTiO3": {"a": 3.9940, "c": 4.0380, "gamma": 90.0, "absences": "none"},
    # primitive hexagonal
    # COD 1501516 (Litasov et al. 2010)
    "WC": {"a": 2.9059, "c": 2.8377, "gamma": 120.0, "absences": "none"},
    # COD 2002799 (Möhr et al. 1996)
    "TiB2": {"a": 3.0292, "c": 3.2284, "gamma": 120.0, "absences": "none"},
    # wurtzite
    # NIST SRM 674b
    "ZnO": {"a": 3.2499, "c": 5.2067, "gamma": 120.0, "absences": "wurtzite"},
    # Detchprohm et al. 1992, Jpn. J. Appl. Phys. 31 L1454, doi:10.1143/JJAP.31.L1454
    "GaN": {"a": 3.1892, "c": 5.1850, "gamma": 120.0, "absences": "wurtzite"},
    # Schulz & Thiemann 1977, Solid State Commun. 23 815, doi:10.1016/0038-1098(77)90959-0
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
    # rhombohedral, R-3c/R3c (c glide)
    # NIST SRM 676a
    "α-Al2O3": {"a": 4.7594, "c": 12.9923, "gamma": 120.0, "absences": "rhombohedral-c"},
    # NBS Mono. 25 Sec. 18 (1981)
    "α-Fe2O3 (hematite)": {
        "a": 5.0356,
        "c": 13.7489,
        "gamma": 120.0,
        "absences": "rhombohedral-c",
    },
    # NIST SRM 674b
    "Cr2O3": {"a": 4.9586, "c": 13.5965, "gamma": 120.0, "absences": "rhombohedral-c"},
    # NBS Circ. 539 v2 (Swanson & Fuyat 1953)
    "CaCO3 (calcite)": {"a": 4.9890, "c": 17.0620, "gamma": 120.0, "absences": "rhombohedral-c"},
    # COD 1541936 (Abrahams et al. 1966)
    "LiNbO3": {"a": 5.1483, "c": 13.8631, "gamma": 120.0, "absences": "rhombohedral-c"},
    # rhombohedral, R-3m
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


def _is_cubic_lattice(lattice: tuple[float, float, float, float, float, float] | None) -> bool:
    if lattice is None:
        return False
    a, b, c, alpha, beta, gamma = lattice
    return (
        math.isclose(a, b)
        and math.isclose(b, c)
        and math.isclose(alpha, 90.0)
        and math.isclose(beta, 90.0)
        and math.isclose(gamma, 90.0)
    )


def _is_orthogonal_lattice(lattice: tuple[float, float, float, float, float, float]) -> bool:
    return (
        math.isclose(lattice[3], 90.0)
        and math.isclose(lattice[4], 90.0)
        and math.isclose(lattice[5], 90.0)
    )


def _canonical_hkl(
    hkl: Sequence[float],
    lattice: tuple[float, float, float, float, float, float] | None,
) -> tuple[int, int, int]:
    indices = tuple(int(i) for i in hkl)
    if _is_cubic_lattice(lattice):
        return tuple(sorted((abs(i) for i in indices), reverse=True))
    if lattice is not None and _is_orthogonal_lattice(lattice):
        h, k, ell = (abs(i) for i in indices)
        if math.isclose(lattice[0], lattice[1]):
            h, k = sorted((h, k), reverse=True)
        return (h, k, ell)
    for value in indices:
        if value < 0:
            return tuple(-i for i in indices)
        if value > 0:
            return indices
    return indices


def _label_preference(hkl: tuple[int, int, int]) -> tuple[int, tuple[int, int, int]]:
    """Sort key preferring the conventional family label: fewest negative
    indices, then lexicographically largest (h before k before l)."""
    return (sum(1 for i in hkl if i < 0), tuple(-i for i in hkl))


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
            if not all(0.0 < angle < 180.0 for angle in (alpha, beta, gamma)):
                raise ValueError("cell angles must be strictly between 0 and 180 degrees")
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
            if np.linalg.det(g) <= 0:
                raise ValueError("degenerate cell: angles do not form a valid lattice")
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
        enumerated; ``d_min`` is floored at 0.2 Å and ``max_index`` capped at
        25 to keep the enumeration bounded.
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

        d_min = max(float(d_min), 0.2)
        if max_index is None:
            max_index = math.ceil(max(self.lattice[:3]) / d_min)
        max_index = min(int(max_index), 25)
        distinct_labels = _is_orthogonal_lattice(self.lattice)
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
                    representative = _canonical_hkl(hkl, self.lattice)
                    family = families_by_d.setdefault(d_key, {"d": spacing, "labels": {}})
                    labels = family["labels"]
                    labels[representative] = labels.get(representative, 0) + 1
        reflections = []
        for family in families_by_d.values():
            labels = family["labels"]
            if distinct_labels:
                ordered = sorted(
                    labels.items(), key=lambda entry: (-entry[1], _label_preference(entry[0]))
                )
                representative = ordered[0][0]
                hkl_str = "/".join(_format_hkl(label) for label, _ in ordered)
            else:
                representative = min(labels, key=_label_preference)
                hkl_str = _format_hkl(representative)
            reflections.append(
                {
                    "hkl": representative,
                    "hkl_str": hkl_str,
                    "d": family["d"],
                    "multiplicity": sum(labels.values()),
                    "intensity": None,
                }
            )
        return sorted(reflections, key=lambda reflection: -reflection["d"])

    def _all_reflections(self) -> list[dict]:
        if self._reflection_cache is None:
            self._reflection_cache = self.reflections()
        return self._reflection_cache

    def match_d(self, d: float, tol: float = 0.03) -> list[dict]:
        """Reflections within fractional ``tol`` of ``d``, closest first.

        Errors are relative to the reference d, matching
        :func:`match_candidate`.
        """
        if d <= 0:
            return []
        matches = []
        for reflection in self._all_reflections():
            error = abs(reflection["d"] - d) / reflection["d"]
            if error <= tol:
                matches.append({**reflection, "d_error": error})
        return sorted(matches, key=lambda reflection: reflection["d_error"])


# Center estimation
# --- Internal helpers ---
def _bandpass(frame: np.ndarray, mask: np.ndarray | None = None, log: bool = True) -> np.ndarray:
    work = np.log1p(frame - frame.min()) if log else frame - frame.min()
    sigma = max(5.0, 0.02 * min(work.shape))
    work = work - ndimage.gaussian_filter(work, sigma=sigma)
    if mask is not None:
        work[mask] = 0.0
    return work - work.mean()


def _parabolic_offset(values: np.ndarray, p: int, n: int) -> float:
    lo, hi = values[(p - 1) % n], values[(p + 1) % n]
    denom = 2.0 * values[p] - lo - hi
    if denom <= 0:
        return 0.0
    delta = 0.5 * (hi - lo) / denom
    return delta if abs(delta) <= 1.0 else 0.0


def _peak_to_sidelobe(corr: np.ndarray, p_row: int, p_col: int, exclude: float = 5.0) -> float:
    n_rows, n_cols = corr.shape
    row_dist = np.abs(np.arange(n_rows) - p_row)
    col_dist = np.abs(np.arange(n_cols) - p_col)
    row_dist = np.minimum(row_dist, n_rows - row_dist)
    col_dist = np.minimum(col_dist, n_cols - col_dist)
    outside = (row_dist[:, None] > exclude) | (col_dist[None, :] > exclude)
    side = corr[outside]
    spread = side.std()
    if spread <= 0:
        return 0.0
    return float((corr[p_row, p_col] - side.mean()) / spread)


def _upsampled_peak(
    cross: np.ndarray, row0: float, col0: float, upsample: int
) -> tuple[float, float]:
    n_rows, n_cols = cross.shape
    f_row = np.fft.fftfreq(n_rows)
    f_col = np.fft.fftfreq(n_cols)
    offsets = np.arange(-int(np.ceil(1.5 * upsample)), int(np.ceil(1.5 * upsample)) + 1) / upsample
    rows = row0 + offsets
    cols = col0 + offsets
    e_row = np.exp(2j * np.pi * rows[:, None] * f_row[None, :])
    e_col = np.exp(2j * np.pi * f_col[:, None] * cols[None, :])
    local = (e_row @ cross @ e_col).real
    p_row, p_col = np.unravel_index(int(np.argmax(local)), local.shape)
    d_row = _parabolic_offset(local[:, p_col], p_row, local.shape[0]) / upsample
    d_col = _parabolic_offset(local[p_row, :], p_col, local.shape[1]) / upsample
    return float(rows[p_row] + d_row), float(cols[p_col] + d_col)


def _phase_shift(
    ref: np.ndarray, moving: np.ndarray, upsample: int = 1
) -> tuple[float, float, float]:
    window = tukey(ref.shape[0], 0.2)[:, None] * tukey(ref.shape[1], 0.2)[None, :]
    ref = ref * window
    moving = moving * window
    cross = np.fft.fft2(ref) * np.conj(np.fft.fft2(moving))
    cross = cross / np.maximum(np.abs(cross), 1e-12)
    f_row = np.fft.fftfreq(ref.shape[0])[:, None]
    f_col = np.fft.fftfreq(ref.shape[1])[None, :]
    cross = cross * np.exp(-(f_row**2 + f_col**2) / (2.0 * 0.15**2))
    corr = np.fft.ifft2(cross).real
    n_rows, n_cols = corr.shape
    p_row, p_col = np.unravel_index(int(np.argmax(corr)), corr.shape)
    psr = _peak_to_sidelobe(corr, p_row, p_col)
    if upsample > 1:
        s_row, s_col = _upsampled_peak(cross, float(p_row), float(p_col), upsample)
    else:
        s_row = p_row + _parabolic_offset(corr[:, p_col], p_row, n_rows)
        s_col = p_col + _parabolic_offset(corr[p_row, :], p_col, n_cols)
    return float(s_row), float(s_col), psr


def _wrap_signed(shift: float, n: int) -> float:
    shift = shift % n
    return shift - n if shift > n / 2 else shift


# --- Center estimation ---
def center_symmetry(
    frame: np.ndarray,
    guess: tuple[float, float] | None = None,
    search_radius: float = 8.0,
    mask: np.ndarray | None = None,
) -> tuple[float, float]:
    """Refine a center guess by local Friedel-symmetry autocorrelation."""
    frame = np.asarray(frame, dtype=np.float64)
    n_rows, n_cols = frame.shape
    if guess is None:
        guess = ((n_rows - 1) / 2.0, (n_cols - 1) / 2.0)

    work = _bandpass(frame, mask)
    spectrum = np.fft.fft2(work)
    corr = np.fft.ifft2(spectrum * spectrum).real
    target_row = (2.0 * guess[0]) % n_rows
    target_col = (2.0 * guess[1]) % n_cols
    row_idx = np.arange(n_rows, dtype=np.float64)
    col_idx = np.arange(n_cols, dtype=np.float64)
    row_dist = np.minimum(np.abs(row_idx - target_row), n_rows - np.abs(row_idx - target_row))
    col_dist = np.minimum(np.abs(col_idx - target_col), n_cols - np.abs(col_idx - target_col))
    near = (row_dist[:, None] <= 2.0 * search_radius) & (col_dist[None, :] <= 2.0 * search_radius)
    p_row, p_col = np.unravel_index(int(np.argmax(np.where(near, corr, -np.inf))), corr.shape)
    row2 = p_row + _parabolic_offset(corr[:, p_col], p_row, n_rows)
    col2 = p_col + _parabolic_offset(corr[p_row, :], p_col, n_cols)
    row = min(((row2 + offset) / 2.0 for offset in (0.0, n_rows)), key=lambda c: abs(c - guess[0]))
    col = min(((col2 + offset) / 2.0 for offset in (0.0, n_cols)), key=lambda c: abs(c - guess[1]))
    return float(row), float(col)


def center_phase_correlation(
    frame: np.ndarray, mask: np.ndarray | None = None, upsample: int = 20
) -> tuple[float, float]:
    """Estimate the inversion center by phase correlation."""
    frame = np.asarray(frame, dtype=np.float64)
    n_rows, n_cols = frame.shape
    work = _bandpass(frame, mask)
    rot = work[::-1, ::-1]
    d_row, d_col, _ = _phase_shift(work, rot, upsample=upsample)
    # Inversion center
    row_cands = [(n_rows - 1 + d) / 2.0 for d in (d_row % n_rows, d_row % n_rows - n_rows)]
    col_cands = [(n_cols - 1 + d) / 2.0 for d in (d_col % n_cols, d_col % n_cols - n_cols)]
    candidates = [
        (r, c)
        for r in row_cands
        for c in col_cands
        if 0.0 <= r <= n_rows - 1 and 0.0 <= c <= n_cols - 1
    ]
    # skip border candidates
    interior = [(r, c) for r, c in candidates if 1.0 <= r <= n_rows - 2 and 1.0 <= c <= n_cols - 2]
    if interior:
        candidates = interior
    if not candidates:
        candidates = [((n_rows - 1) / 2.0, (n_cols - 1) / 2.0)]
    scored = [(r, c, _symmetry_score(frame, (r, c), mask=mask)) for r, c in candidates]
    best = max(score for _, _, score in scored)
    mid = ((n_rows - 1) / 2.0, (n_cols - 1) / 2.0)
    row, col, _ = min(
        (candidate for candidate in scored if candidate[2] >= best - 1e-4),
        key=lambda candidate: math.hypot(candidate[0] - mid[0], candidate[1] - mid[1]),
    )
    return float(row), float(col)


# --- Quality metrics ---
def _symmetry_score(
    frame: np.ndarray, center: tuple[float, float], mask: np.ndarray | None = None
) -> float:
    # Friedel symmetry
    frame = np.asarray(frame, dtype=np.float64)
    n_rows, n_cols = frame.shape
    work = _bandpass(frame)
    rows, cols = np.indices((n_rows, n_cols), dtype=np.float64)
    rot_rows = 2.0 * center[0] - rows
    rot_cols = 2.0 * center[1] - cols
    valid = (
        (rot_rows >= 0.0) & (rot_rows <= n_rows - 1) & (rot_cols >= 0.0) & (rot_cols <= n_cols - 1)
    )
    rotated = ndimage.map_coordinates(work, [rot_rows, rot_cols], order=1, mode="nearest")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        rot_mask = ndimage.map_coordinates(
            mask.astype(np.float64), [rot_rows, rot_cols], order=1, mode="constant", cval=1.0
        )
        valid &= ~mask & (rot_mask < 0.5)
    a = work[valid]
    b = rotated[valid]
    if a.size < 16:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom <= 0:
        return 0.0
    return float(max(0.0, (a * b).sum() / denom))


def ring_uniformity(
    frame: np.ndarray,
    center: tuple[float, float],
    radius: float,
    half_width: float = 4.0,
    n_theta: int = 180,
    mask: np.ndarray | None = None,
) -> dict:
    """Azimuthal uniformity QC for one ring; ``mask`` pixels are excluded."""
    frame = np.asarray(frame, dtype=np.float64)
    rows, cols = np.indices(frame.shape, dtype=np.float64)
    d_row = rows - center[0]
    d_col = cols - center[1]
    annulus = np.abs(np.hypot(d_row, d_col) - radius) <= half_width
    if mask is not None:
        annulus &= ~np.asarray(mask, dtype=bool)
    theta = np.arctan2(d_row[annulus], d_col[annulus])
    sector = np.clip(((theta + np.pi) / (2.0 * np.pi) * n_theta).astype(int), 0, n_theta - 1)
    sums = np.bincount(sector, weights=frame[annulus], minlength=n_theta)
    counts = np.bincount(sector, minlength=n_theta)
    live = counts > 0
    if not live.any():
        return {"cv": 0.0, "coverage": 0.0, "snr": 0.0}
    profile = sums[live] / counts[live]
    mean = float(profile.mean())
    std = float(profile.std())
    cv = std / mean if mean > 0 else 0.0
    positive = profile[profile > 0]
    coverage = float(np.mean(profile > 0.5 * np.median(positive))) if positive.size else 0.0
    snr = min(mean / std, 999.0) if std > 0 else 999.0
    return {"cv": float(cv), "coverage": float(coverage), "snr": float(snr)}


# --- Dispatch ---
def pick_center(
    frame: np.ndarray,
    method: str = "auto",
    mask: np.ndarray | None = None,
    guess: tuple[float, float] | None = None,
    search_radius: float = 8.0,
) -> dict:
    """Estimate the pattern center with one method or an automatic pick."""
    frame = np.asarray(frame, dtype=np.float64)
    if method == "symmetry":
        row, col = center_symmetry(frame, guess=guess, search_radius=search_radius, mask=mask)
        name = "symmetry"
    elif method == "phase_corr":
        row, col = center_phase_correlation(frame, mask=mask)
        name = "phase_corr"
    elif method == "auto":
        p_row, p_col = center_phase_correlation(frame, mask=mask)
        s_row, s_col = center_symmetry(frame, guess=guess, search_radius=search_radius, mask=mask)
        candidates = [
            ("phase_corr", p_row, p_col, _symmetry_score(frame, (p_row, p_col), mask=mask)),
            ("symmetry", s_row, s_col, _symmetry_score(frame, (s_row, s_col), mask=mask)),
        ]
        name, row, col, _ = max(candidates, key=lambda c: c[3])
    else:
        raise ValueError(f"unknown method {method!r}; use auto, symmetry, or phase_corr")
    return {"row": float(row), "col": float(col), "method": name}


# --- Stack alignment ---
def align_frames(
    frames: np.ndarray,
    reference: np.ndarray | None = None,
    max_shift: float = 8.0,
) -> tuple[np.ndarray, list[tuple[float, float]], list[bool]]:
    """Align a stack of patterns by subpixel phase correlation."""
    frames = np.asarray(frames, dtype=np.float64)
    n_frames, n_rows, n_cols = frames.shape
    ref = frames[0] if reference is None else np.asarray(reference, dtype=np.float64)
    # linear bandpass for count statistics
    ref = _bandpass(ref, log=False)
    aligned = np.empty_like(frames)
    shifts: list[tuple[float, float]] = []
    used: list[bool] = []
    for i in range(n_frames):
        s_row, s_col, psr = _phase_shift(ref, _bandpass(frames[i], log=False))
        s_row = _wrap_signed(s_row, n_rows)
        s_col = _wrap_signed(s_col, n_cols)
        peak_quality = psr / (psr + 10.0) if psr > 0 else 0.0
        ok = np.hypot(s_row, s_col) <= max_shift and peak_quality >= 0.2
        shifts.append((float(s_row), float(s_col)))
        used.append(bool(ok))
        aligned[i] = ndimage.shift(frames[i], (s_row, s_col), order=1) if ok else frames[i]
    return aligned, shifts, used


# Reference-line matching
# out-of-tolerance pad cost
_NO_MATCH_COST = 1.0e6


def match_candidate(observed_d: Sequence[float], lines: Sequence[dict], tol: float = 0.03) -> dict:
    """Match measured d-spacings against one reference phase."""
    observed = [
        float(spacing)
        for spacing in observed_d
        if spacing and math.isfinite(float(spacing)) and float(spacing) > 0
    ]
    references = [
        (float(line["d"]), float(line.get("i_rel", line.get("intensity")) or 0.0))
        for line in lines
        if math.isfinite(float(line["d"])) and float(line["d"]) > 0
    ]
    has_intensity = any(rel_intensity > 0 for _, rel_intensity in references)
    n_observed = len(observed)
    if n_observed == 0 or not references:
        return {
            "matched": 0,
            "n_obs": n_observed,
            "mean_err": None,
            "n_missing_strong": 0 if has_intensity else None,
            "assignments": [],
        }
    observed_g = [1.0 / spacing for spacing in observed]
    reference_g = [1.0 / spacing for spacing, _ in references]

    # in-tolerance pair costs
    cost = np.full((n_observed, len(references)), _NO_MATCH_COST)
    d_errors = np.zeros_like(cost)
    for obs_index, observed_value in enumerate(observed_g):
        for ref_index, reference_value in enumerate(reference_g):
            error = abs(reference_value - observed_value) / observed_value
            if error <= tol:
                cost[obs_index, ref_index] = error
                d_errors[obs_index, ref_index] = (
                    abs(1.0 / observed_value - references[ref_index][0]) / references[ref_index][0]
                )

    # maximum matches first, then lowest total error
    assignments = [(obs_index, None) for obs_index in range(n_observed)]
    errors = []
    matched_refs = set()
    for obs_index, ref_index in zip(*linear_sum_assignment(cost)):
        if cost[obs_index, ref_index] >= _NO_MATCH_COST:
            continue
        assignments[obs_index] = (int(obs_index), int(ref_index))
        errors.append(float(d_errors[obs_index, ref_index]))
        matched_refs.add(int(ref_index))

    n_matched = len(errors)
    n_missing_strong = None
    if has_intensity:
        g_min, g_max = min(observed_g), max(observed_g)
        n_missing_strong = sum(
            1
            for ref_index, (_, rel_intensity) in enumerate(references)
            if rel_intensity >= 25.0
            and g_min <= reference_g[ref_index] <= g_max
            and ref_index not in matched_refs
        )

    return {
        "matched": n_matched,
        "n_obs": n_observed,
        "mean_err": (float(sum(errors) / n_matched) if n_matched else None),
        "n_missing_strong": n_missing_strong,
        "assignments": assignments,
    }


def match_sort_key(report: dict) -> tuple:
    """Sort phase reports from strongest to weakest match."""
    return (
        -report["matched"],
        report["n_missing_strong"] or 0,
        report["mean_err"] if report["mean_err"] is not None else 1.0,
    )


# Diffraction analysis
BF_RADIUS_FRACTION = 0.125
RING_FIT_MODELS = ("gaussian", "pseudo_voigt")

MEASUREMENT_COLUMNS = [
    "id",
    "kind",
    "raw_row",
    "raw_col",
    "row",
    "col",
    "row_err",
    "col_err",
    "r_pixels",
    "r_pixels_err",
    "g_inv_angstrom",
    "g_inv_angstrom_err",
    "d_angstrom",
    "d_angstrom_err",
    "angle_deg",
    "angle_deg_err",
    "intensity",
    "fit_quality",
    "fwhm_px",
    "fwhm_inv_angstrom",
    "intensity_integrated",
    "hkl",
    "hkl_candidates",
    "note",
]


# --- Records and formatting ---
def element_symbols(text: str) -> set[str]:
    """Element symbols found in a formula-like string."""
    return set(re.findall(r"[A-Z][a-z]?", text or ""))


def parse_elements(value) -> set[str] | None:
    """Element-symbol set from a string or iterable, or None if empty."""
    if not value:
        return None
    if isinstance(value, str):
        value = re.split(r"[,\s]+", value.strip())
    return {symbol.strip().capitalize() for symbol in value if symbol.strip()} or None


def index_assignment(candidate: dict | None) -> dict:
    """Indexing fields for a matched reflection candidate."""
    if candidate is None:
        return {"hkl": "", "d_ref": None, "d_error": None}
    return {
        "hkl": candidate["hkl_str"],
        "d_ref": candidate["d"],
        "d_error": candidate["d_error"],
    }


def empty_index_fields() -> dict:
    """Blank indexing fields for a spot or ring record."""
    return {
        "hkl": "",
        "hkl_candidates": [],
        "d_ref": None,
        "d_error": None,
        "note": "",
    }


def next_record_id(records) -> int:
    """Next one-based id for a list of record dicts."""
    return max((int(record["id"]) for record in records), default=0) + 1


def format_zone_axis(hkl1: tuple[int, int, int], hkl2: tuple[int, int, int]) -> str:
    """Zone-axis label ``[uvw]`` from two indexed reflections."""
    h1, k1, l1 = hkl1
    h2, k2, l2 = hkl2
    u = k1 * l2 - l1 * k2
    v = l1 * h2 - h1 * l2
    w = h1 * k2 - k1 * h2
    divisor = math.gcd(math.gcd(abs(u), abs(v)), abs(w))
    if divisor == 0:
        return ""
    u, v, w = u // divisor, v // divisor, w // divisor
    for axis in (u, v, w):
        if axis != 0:
            if axis < 0:
                u, v, w = -u, -v, -w
            break
    return "[" + "".join(str(axis) for axis in (u, v, w)) + "]"


# --- Input and masking ---
def normalize_data_input(
    data,
    *,
    title: str = "",
    pixel_size: float | None = None,
    k_pixel_size: float | None = None,
    replace_title: bool = False,
):
    """Unwrap Dataset-like input into array, title, and calibrations."""
    k_calibrated = False
    if hasattr(data, "_fields") and "data" in getattr(data, "_fields", ()):
        metadata = data.metadata or {}
        if pixel_size is None and metadata.get("pixel_size") is not None:
            pixel_size = float(metadata["pixel_size"])
        data = data.data

    if hasattr(data, "sampling") and hasattr(data, "array"):
        if (replace_title or not title) and getattr(data, "name", ""):
            title = str(data.name)
        units = list(getattr(data, "units", ["pixels"] * 4))
        if pixel_size is None and units and units[0] in ("Å", "angstrom", "A", "nm"):
            pixel_size = float(data.sampling[0])
            if units[0] == "nm":
                pixel_size *= 10
        if k_pixel_size is None and len(units) > 2 and units[2] in ("1/Å", "1/A"):
            k_pixel_size = float(data.sampling[2])
            k_calibrated = True
        data = data.array
    return data, title, pixel_size, k_pixel_size, k_calibrated


def pack_float32_halves(x: np.ndarray, y: np.ndarray) -> bytes:
    """Two arrays packed as concatenated float32 bytes."""
    return np.concatenate([x, y]).astype(np.float32).tobytes()


def build_analysis_mask(
    shape: tuple[int, int],
    regions: list[dict],
    center: tuple[float, float],
) -> np.ndarray | None:
    """Boolean exclusion mask from disk and wedge regions."""
    if not regions:
        return None
    n_rows, n_cols = shape
    rows = np.arange(n_rows, dtype=np.float64)[:, None]
    cols = np.arange(n_cols, dtype=np.float64)[None, :]
    center_row, center_col = center
    mask = np.zeros((n_rows, n_cols), dtype=bool)
    for region in regions:
        kind = region.get("kind")
        if kind == "disk":
            mask |= np.hypot(rows - region["row"], cols - region["col"]) <= region["radius"]
        elif kind == "wedge":
            # full-circle span: every angle masked
            if abs(float(region["end_deg"]) - float(region["start_deg"])) >= 360.0:
                mask[:] = True
                continue
            theta = np.degrees(np.arctan2(rows - center_row, cols - center_col)) % 360.0
            start = float(region["start_deg"]) % 360.0
            end = float(region["end_deg"]) % 360.0
            mask |= (
                (theta >= start) & (theta <= end)
                if start <= end
                else ((theta >= start) | (theta <= end))
            )
    return mask


# --- Radial and azimuthal profiles ---
def corrected_radius(
    d_row,
    d_col,
    *,
    ellipse_ratio: float = 1.0,
    ellipse_angle: float = 0.0,
    ellipse_corrected: bool = False,
):
    """Radius with optional elliptical-distortion correction.

    The correction is mean-preserving: an ellipse of semi-axes A, B maps to a
    circle of radius sqrt(A*B) (the mean radius), so a calibration set before
    the correction stays valid after it.
    """
    if not ellipse_corrected or ellipse_ratio == 1.0:
        return np.hypot(d_row, d_col)
    angle = math.radians(ellipse_angle)
    major = d_col * math.cos(angle) + d_row * math.sin(angle)
    minor = -d_col * math.sin(angle) + d_row * math.cos(angle)
    root_ratio = math.sqrt(ellipse_ratio)
    return np.hypot(major / root_ratio, minor * root_ratio)


def corrected_vector(
    d_row,
    d_col,
    *,
    ellipse_ratio: float = 1.0,
    ellipse_angle: float = 0.0,
    ellipse_corrected: bool = False,
):
    """Displacement vector under the same ellipse transform as corrected_radius."""
    if not ellipse_corrected or ellipse_ratio == 1.0:
        return d_row, d_col
    angle = math.radians(ellipse_angle)
    major = d_col * math.cos(angle) + d_row * math.sin(angle)
    minor = -d_col * math.sin(angle) + d_row * math.cos(angle)
    root_ratio = math.sqrt(ellipse_ratio)
    major, minor = major / root_ratio, minor * root_ratio
    return (
        major * math.sin(angle) + minor * math.cos(angle),
        major * math.cos(angle) - minor * math.sin(angle),
    )


def radial_profile_px(
    frame: np.ndarray,
    *,
    center: tuple[float, float],
    n_bins: int | None = None,
    max_radius: float | None = None,
    mask: np.ndarray | None = None,
    angular_range: tuple[float, float] | None = None,
    ellipse_ratio: float = 1.0,
    ellipse_angle: float = 0.0,
    ellipse_corrected: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Radial intensity profile in detector pixels."""
    n_rows, n_cols = frame.shape
    center_row, center_col = float(center[0]), float(center[1])
    if max_radius is None:
        max_radius = float(
            min(center_row, center_col, (n_rows - 1) - center_row, (n_cols - 1) - center_col)
        )
    max_radius = float(max(1.0, max_radius))
    n_bins = max(1, int(round(max_radius))) if n_bins is None else int(max(1, n_bins))

    rows = np.arange(n_rows, dtype=np.float64)[:, None]
    cols = np.arange(n_cols, dtype=np.float64)[None, :]
    d_row, d_col = rows - center_row, cols - center_col
    radii = corrected_radius(
        d_row,
        d_col,
        ellipse_ratio=ellipse_ratio,
        ellipse_angle=ellipse_angle,
        ellipse_corrected=ellipse_corrected,
    )
    flat_r = radii.ravel()
    flat_i = frame.astype(np.float64).ravel()
    keep = None if mask is None else ~mask.ravel()
    if angular_range is not None:
        start, end = float(angular_range[0]) % 360.0, float(angular_range[1]) % 360.0
        span = abs(float(angular_range[1]) - float(angular_range[0]))
        # full-circle span: no angular restriction
        if span < 360.0 and start != end:
            theta = np.degrees(np.arctan2(d_row, d_col)).ravel() % 360.0
            wedge = (
                (theta >= start) & (theta <= end)
                if start <= end
                else ((theta >= start) | (theta <= end))
            )
            keep = wedge if keep is None else keep & wedge
    if keep is not None:
        flat_r, flat_i = flat_r[keep], flat_i[keep]

    edges = np.linspace(0.0, max_radius, n_bins + 1)
    indices = np.digitize(flat_r, edges) - 1
    inside = (indices >= 0) & (indices < n_bins)
    indices = indices[inside]
    values = flat_i[inside]

    counts = np.bincount(indices, minlength=n_bins).astype(np.float64)
    sums = np.bincount(indices, weights=values, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        intensity = np.where(counts > 0, sums / counts, 0.0)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    return bin_centers.astype(np.float32), intensity.astype(np.float32)


def ring_sectors(
    frame: np.ndarray,
    *,
    center: tuple[float, float],
    radius_px: float,
    half_width: float,
    n_theta: int,
    mask: np.ndarray | None = None,
    use_corrected_radius: bool = True,
    ellipse_ratio: float = 1.0,
    ellipse_angle: float = 0.0,
    ellipse_corrected: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-sector counts and intensity sums around one ring."""
    frame = frame.astype(np.float64)
    center_row, center_col = center
    n_rows, n_cols = frame.shape
    rows = np.arange(n_rows, dtype=np.float64)[:, None]
    cols = np.arange(n_cols, dtype=np.float64)[None, :]
    d_row, d_col = rows - center_row, cols - center_col
    if use_corrected_radius:
        radii = corrected_radius(
            d_row,
            d_col,
            ellipse_ratio=ellipse_ratio,
            ellipse_angle=ellipse_angle,
            ellipse_corrected=ellipse_corrected,
        )
    else:
        radii = np.hypot(d_row, d_col)

    theta_centers = (np.arange(n_theta) + 0.5) * (360.0 / n_theta)
    selected = np.abs(radii - radius_px) <= half_width
    if mask is not None:
        selected &= ~mask
    if not selected.any():
        zero = np.zeros(n_theta)
        return theta_centers, zero.copy(), zero.copy(), zero.copy(), zero.copy()

    theta = np.degrees(np.arctan2(d_row, d_col)) % 360.0
    sector = np.minimum((theta[selected] / (360.0 / n_theta)).astype(int), n_theta - 1)
    intensity = frame[selected]
    # median pedestal, negatives clipped
    weight = np.clip(intensity - np.median(intensity), 0.0, None)
    counts = np.bincount(sector, minlength=n_theta).astype(np.float64)
    intensity_sum = np.bincount(sector, weights=intensity, minlength=n_theta)
    weight_sum = np.bincount(sector, weights=weight, minlength=n_theta)
    weighted_radius_sum = np.bincount(sector, weights=weight * radii[selected], minlength=n_theta)
    return theta_centers, counts, intensity_sum, weight_sum, weighted_radius_sum


def azimuthal_profile_from_frame(
    frame: np.ndarray,
    *,
    center: tuple[float, float],
    radius_px: float,
    half_width: float,
    n_theta: int,
    mask: np.ndarray | None = None,
    ellipse_ratio: float = 1.0,
    ellipse_angle: float = 0.0,
    ellipse_corrected: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthal intensity profile I(theta) around one ring."""
    theta, counts, intensity_sum, _, _ = ring_sectors(
        frame,
        center=center,
        radius_px=radius_px,
        half_width=half_width,
        n_theta=n_theta,
        mask=mask,
        use_corrected_radius=True,
        ellipse_ratio=ellipse_ratio,
        ellipse_angle=ellipse_angle,
        ellipse_corrected=ellipse_corrected,
    )
    intensity = np.where(counts > 0, intensity_sum / np.maximum(counts, 1.0), 0.0)
    return theta.astype(np.float32), intensity.astype(np.float32)


def texture_from_profile(
    theta_deg: np.ndarray, intensity: np.ndarray, *, return_profile: bool = False
) -> dict:
    """Texture strength and preferred angle from an azimuthal profile."""
    values = intensity.astype(np.float64)
    covered = values != 0.0
    if covered.sum() >= 8:
        live = np.sort(np.asarray(theta_deg, dtype=np.float64)[covered])
        gaps = np.diff(np.concatenate([live, live[:1] + 360.0]))
        span = 360.0 - float(gaps.max())
    else:
        span = 0.0
    if covered.sum() < 8 or span < 90.0:
        strength, angle = 0.0, 0.0
    else:
        theta = np.radians(theta_deg)[covered]
        design = np.column_stack([np.ones_like(theta), np.cos(2 * theta), np.sin(2 * theta)])
        (mean_level, cosine, sine), *_ = np.linalg.lstsq(design, values[covered], rcond=None)
        if mean_level <= 0:
            strength, angle = 0.0, 0.0
        else:
            strength = float(min(1.0, math.hypot(cosine, sine) / mean_level))
            angle = float(math.degrees(math.atan2(sine, cosine)) / 2.0 % 180.0)
    report = {"strength": strength, "angle_deg": angle, "coverage_deg": span}
    if return_profile:
        report["profile"] = (theta_deg, intensity)
    return report


# --- Ellipse and background fitting ---
def fit_ellipse_from_sectors(
    theta_centers: np.ndarray,
    counts: np.ndarray,
    weight_sum: np.ndarray,
    weighted_radius_sum: np.ndarray,
) -> dict:
    """Ellipse ratio and angle from ring-sector radii."""
    valid = (counts >= 10) & (weight_sum > 0)
    if valid.sum() < 8:
        raise ValueError(
            f"could not fit ellipse: ring found in {int(valid.sum())} sectors, need >= 8; "
            "check the ring radius and center"
        )
    # short arcs leave the order-1/order-2 harmonics collinear
    live = np.sort(np.asarray(theta_centers, dtype=np.float64)[valid])
    gaps = np.diff(np.concatenate([live, live[:1] + 360.0]))
    span = 360.0 - float(gaps.max())
    if span < 120.0:
        raise ValueError(
            f"could not fit ellipse: ring sectors span only {span:.0f} deg, need >= 120; "
            "reduce the excluded regions or pick a fuller ring"
        )
    radii_by_theta = weighted_radius_sum[valid] / weight_sum[valid]
    theta = np.radians(theta_centers)[valid]
    # order-1 terms absorb residual center error
    design = np.column_stack(
        [
            np.ones_like(theta),
            np.cos(theta),
            np.sin(theta),
            np.cos(2 * theta),
            np.sin(2 * theta),
        ]
    )
    solution, *_ = np.linalg.lstsq(design, radii_by_theta, rcond=None)
    mean_radius, cosine, sine = solution[0], solution[3], solution[4]
    epsilon = math.hypot(cosine, sine) / mean_radius
    ratio = (1.0 + epsilon) / (1.0 - epsilon) if epsilon < 1.0 else float("inf")
    angle = (0.5 * math.degrees(math.atan2(sine, cosine))) % 180.0
    residual = radii_by_theta - design @ solution
    return {
        "ratio": float(ratio),
        "angle_deg": float(angle),
        "r_mean": float(mean_radius),
        "residual_px": float(np.sqrt(np.mean(residual**2))),
        "n_sectors": int(valid.sum()),
    }


def fit_radial_background(
    radii_px: np.ndarray,
    intensity: np.ndarray,
    *,
    peak_windows: list[tuple[float, float]],
    exclude_radius: float,
    method: str = "power",
    poly_order: int = 3,
) -> np.ndarray:
    """Smooth background under a radial profile, excluding peak windows."""
    if method not in ("power", "poly"):
        raise ValueError(f"method must be 'power' or 'poly', got {method!r}")
    if poly_order < 0:
        raise ValueError(f"poly_order must be non-negative, got {poly_order}")

    radii = radii_px.astype(np.float64)
    values = intensity.astype(np.float64)
    keep = radii > float(exclude_radius)
    for lo, hi in peak_windows:
        keep &= ~((radii >= lo) & (radii <= hi))
    if method == "power":
        keep &= values > 0

    min_points = 2 if method == "power" else max(2, poly_order + 1)
    if keep.sum() < min_points:
        raise ValueError(
            "not enough background bins to fit; widen the profile or narrow peak_windows"
        )

    if method == "power":
        coefficients = np.polyfit(np.log(radii[keep]), np.log(values[keep]), 1)
        positive_radii = radii[radii > 0]
        eval_radii = np.maximum(radii, positive_radii.min())
        background = np.exp(np.polyval(coefficients, np.log(eval_radii)))
    else:
        coefficients = np.polyfit(radii[keep], values[keep], poly_order)
        background = np.polyval(coefficients, radii)
    return background.astype(np.float32)


# --- Peak fitting ---
def fit_gaussian_spot(
    frame: np.ndarray,
    row: float,
    col: float,
    *,
    half_window: int,
) -> dict | None:
    """Subpixel 2D Gaussian fit around a spot."""
    frame = np.asarray(frame, dtype=np.float32)
    half = max(4, int(half_window))
    row0, col0 = int(round(row)), int(round(col))
    row_lo, row_hi = max(0, row0 - half), min(frame.shape[0], row0 + half + 1)
    col_lo, col_hi = max(0, col0 - half), min(frame.shape[1], col0 + half + 1)
    patch = frame[row_lo:row_hi, col_lo:col_hi].astype(np.float64)
    if patch.shape[0] < 5 or patch.shape[1] < 5:
        return None
    try:
        from scipy.optimize import OptimizeWarning, curve_fit
    except Exception:
        return None

    n_rows, n_cols = patch.shape
    row_grid, col_grid = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")

    def gaussian_2d(coords, amplitude, row_center, col_center, sigma_row, sigma_col, offset):
        rows, cols = coords
        exponent = ((rows - row_center) / sigma_row) ** 2
        exponent += ((cols - col_center) / sigma_col) ** 2
        return (amplitude * np.exp(-0.5 * exponent) + offset).ravel()

    peak = np.unravel_index(int(np.argmax(patch)), patch.shape)
    initial = (
        float(patch.max() - patch.min()),
        float(peak[0]),
        float(peak[1]),
        2.0,
        2.0,
        float(patch.min()),
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            fit_params, covariance = curve_fit(
                gaussian_2d,
                (row_grid, col_grid),
                patch.ravel(),
                p0=initial,
                maxfev=5000,
            )
    except Exception:
        return None

    _, fit_row, fit_col, sigma_row, sigma_col, _ = fit_params
    if not (0 <= fit_row < n_rows and 0 <= fit_col < n_cols):
        return None

    parameter_errors = np.sqrt(np.abs(np.diag(covariance)))
    residual = patch.ravel() - gaussian_2d((row_grid, col_grid), *fit_params)
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((patch.ravel() - patch.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "row": float(row_lo + fit_row),
        "col": float(col_lo + fit_col),
        "row_err": float(parameter_errors[1]) if np.isfinite(parameter_errors[1]) else 0.0,
        "col_err": float(parameter_errors[2]) if np.isfinite(parameter_errors[2]) else 0.0,
        "sigma_row": float(abs(sigma_row)),
        "sigma_col": float(abs(sigma_col)),
        "fit_quality": float(r_squared),
    }


def _gaussian_peak(radius, amplitude, center, sigma, offset):
    return amplitude * np.exp(-0.5 * ((radius - center) / sigma) ** 2) + offset


def _pseudo_voigt_peak(radius, amplitude, center, sigma, offset, eta):
    gamma = sigma * 2.3548 / 2.0
    lorentzian = 1.0 / (1.0 + ((radius - center) / gamma) ** 2)
    gaussian_part = np.exp(-0.5 * ((radius - center) / sigma) ** 2)
    return amplitude * (eta * lorentzian + (1.0 - eta) * gaussian_part) + offset


def _ring_fit_window(radius_guess: float, centers: list[float], window: float | None) -> float:
    if window is not None:
        return float(window)
    gaps = [abs(radius_guess - center) for center in centers if center != radius_guess]
    return max(3.0, min(gaps) / 2.0) if gaps else max(6.0, 0.2 * radius_guess)


def _fit_ring_peak(
    radii_px: np.ndarray,
    intensity: np.ndarray,
    *,
    radius_guess: float,
    half_width: float,
    model: str,
) -> dict | None:
    from scipy.optimize import OptimizeWarning, curve_fit

    in_window = (radii_px >= radius_guess - half_width) & (radii_px <= radius_guess + half_width)
    radius_window = radii_px[in_window].astype(np.float64)
    intensity_window = intensity[in_window].astype(np.float64)
    if radius_window.size < 5:
        return None

    initial = [
        max(float(intensity_window.max() - intensity_window.min()), 1e-6),
        radius_guess,
        2.0,
        float(intensity_window.min()),
    ]
    bounds = (
        [0.0, radius_guess - half_width, 0.1, -np.inf],
        [np.inf, radius_guess + half_width, half_width, np.inf],
    )
    if model == "pseudo_voigt":
        initial = initial + [0.5]
        bounds = (bounds[0] + [0.0], bounds[1] + [1.0])
    peak_model = _gaussian_peak if model == "gaussian" else _pseudo_voigt_peak

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        fit_params, _ = curve_fit(
            peak_model,
            radius_window,
            intensity_window,
            p0=initial,
            bounds=bounds,
            maxfev=5000,
        )

    residual = intensity_window - peak_model(radius_window, *fit_params)
    ss_tot = float(np.sum((intensity_window - intensity_window.mean()) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / ss_tot if ss_tot > 0 else 0.0
    amplitude = float(fit_params[0])
    fit_radius = float(fit_params[1])
    sigma = float(abs(fit_params[2]))
    if model == "pseudo_voigt":
        eta = float(fit_params[4])
        gamma = sigma * 2.3548 / 2.0
        integrated = amplitude * (
            eta * math.pi * gamma + (1.0 - eta) * sigma * math.sqrt(2.0 * math.pi)
        )
    else:
        integrated = amplitude * sigma * math.sqrt(2.0 * math.pi)
    return {
        "raw_radius_px": float(radius_guess),
        "radius_px": fit_radius,
        "intensity": amplitude,
        "fwhm_px": 2.3548 * sigma,
        "intensity_integrated": integrated,
        "fit_quality": float(r_squared),
    }


def fit_ring_peaks(
    radii_px: np.ndarray,
    intensity: np.ndarray,
    rings,
    *,
    model: str = "gaussian",
    window: float | None = None,
) -> list[dict | None]:
    """Fit one radial peak for each ring record."""
    if model not in RING_FIT_MODELS:
        raise ValueError(f"model must be one of {RING_FIT_MODELS}, got {model!r}")
    if window is not None and window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    try:
        import scipy.optimize  # noqa: F401
    except ImportError as exc:
        raise ImportError("fit_ring_peaks needs scipy; install it to fit ring peaks") from exc

    centers = sorted(float(ring["radius_px"]) for ring in rings)
    updates = []
    for ring in rings:
        radius_guess = float(ring["radius_px"])
        half_width = _ring_fit_window(radius_guess, centers, window)
        try:
            updates.append(
                _fit_ring_peak(
                    radii_px,
                    intensity,
                    radius_guess=radius_guess,
                    half_width=half_width,
                    model=model,
                )
            )
        except Exception:
            updates.append(None)
    return updates


# --- Measurement export ---
def spot_measurement_record(spot: dict) -> dict:
    """Export row for one spot record."""
    return {
        "id": spot.get("id"),
        "kind": "spot",
        "raw_row": spot.get("raw_row"),
        "raw_col": spot.get("raw_col"),
        "row": spot.get("row"),
        "col": spot.get("col"),
        "row_err": spot.get("row_err"),
        "col_err": spot.get("col_err"),
        "r_pixels": spot.get("r_pixels"),
        "r_pixels_err": spot.get("r_pixels_err"),
        "g_inv_angstrom": spot.get("g_magnitude"),
        "g_inv_angstrom_err": spot.get("g_magnitude_err"),
        "d_angstrom": spot.get("d_spacing"),
        "d_angstrom_err": spot.get("d_spacing_err"),
        "angle_deg": spot.get("angle_deg"),
        "angle_deg_err": spot.get("angle_deg_err"),
        "intensity": spot.get("intensity"),
        "fit_quality": spot.get("fit_quality"),
        "fwhm_px": None,
        "fwhm_inv_angstrom": None,
        "intensity_integrated": None,
        "hkl": spot.get("hkl", ""),
        "hkl_candidates": "|".join(spot.get("hkl_candidates") or []),
        "note": spot.get("note", ""),
    }


def ring_measurement_record(ring: dict) -> dict:
    """Export row for one ring record."""
    return {
        "id": ring.get("id"),
        "kind": "ring",
        "raw_row": None,
        "raw_col": None,
        "row": None,
        "col": None,
        "row_err": None,
        "col_err": None,
        "r_pixels": ring.get("radius_px"),
        "r_pixels_err": None,
        "g_inv_angstrom": ring.get("g_magnitude"),
        "g_inv_angstrom_err": None,
        "d_angstrom": ring.get("d_spacing"),
        "d_angstrom_err": None,
        "angle_deg": None,
        "angle_deg_err": None,
        "intensity": ring.get("intensity"),
        "fit_quality": ring.get("fit_quality"),
        "fwhm_px": ring.get("fwhm_px"),
        "fwhm_inv_angstrom": ring.get("fwhm_inv_angstrom"),
        "intensity_integrated": ring.get("intensity_integrated"),
        "hkl": ring.get("hkl", ""),
        "hkl_candidates": "|".join(ring.get("hkl_candidates") or []),
        "note": ring.get("note", ""),
    }


def build_measurement_records(spots, rings) -> list[dict]:
    """Export rows for all spots and rings."""
    return [spot_measurement_record(spot) for spot in spots] + [
        ring_measurement_record(ring) for ring in rings
    ]


def measurement_metadata(state) -> dict:
    """Export metadata block from widget state values."""
    return {
        "widget_name": "ShowDiffraction",
        "center_row": state.get("center_row"),
        "center_col": state.get("center_col"),
        "center_method": state.get("center_method", ""),
        "k_pixel_size_inv_angstrom_per_px": state.get("k_pixel_size"),
        "calibrated": bool(state.get("k_calibrated")),
        "calibration_source": state.get("calibration_source", "none"),
        "calibration_ref_d_angstrom": state.get("calibration_ref_d", 0.0),
        "calibration_ref_radius_px": state.get("calibration_ref_radius", 0.0),
        "mask_regions": state.get("mask_regions", []),
        "background_subtracted": bool(state.get("profile_subtract_background")),
    }


def write_measurement_file(path, records, metadata) -> pathlib.Path:
    """Write measurement records to CSV or JSON."""
    path = pathlib.Path(path)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps({"metadata": metadata, "measurements": records}, indent=2))
    else:
        with open(path, "w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=MEASUREMENT_COLUMNS)
            writer.writeheader()
            writer.writerows(records)
    return path

class ShowDiffraction(anywidget.AnyWidget):
    """
    Interactive d-spacing analysis for 2D/3D diffraction patterns.

    Pick Bragg spots and rings on the diffraction pattern to measure d-spacings,
    g-vectors, and inter-spot angles, with optional sub-pixel Gaussian refinement.
    Works with a single 2D pattern (SAED) or a 3D stack of patterns, and accepts
    NumPy arrays, PyTorch tensors, or quantem datasets. 4D input is not supported.

    Parameters
    ----------
    data : np.ndarray or torch.Tensor
        2D ``(det_rows, det_cols)`` single pattern or 3D
        ``(n_frames, det_rows, det_cols)`` stack of patterns. A quantem dataset
        or io ``LoadResult`` is also accepted and unwrapped. 4D input raises.
    k_pixel_size : float, optional
        k-space sampling in 1/Å per pixel. Marks the pattern calibrated.
    pixel_size : float, optional
        Real-space pixel size in Å.
    center : tuple[float, float], optional
        (row, col) of the diffraction center in pixels. Defaults to the detector
        center, then auto-detected from the bright-field disk if also no radius.
    bf_radius : float, optional
        Bright-field disk radius in pixels. Defaults to 1/8 of the detector size.
    title : str, default ""
        Title displayed above the widget.
    snap_enabled : bool, default False
        Snap clicked spots to the local intensity maximum.
    snap_radius : int, default 5
        Search radius in pixels for snapping / Gaussian refinement.
    spot_refine : bool, default True
        Sub-pixel refine spots with a 2D Gaussian fit on add.
    dp_scale_mode : str, default "log"
        Diffraction display scaling ("linear", "log", "sqrt").
    ui_mode : {"interactive", "presentation", "report", "minimal"}, default "interactive"
        Shared viewer UI preset. Explicit ``show_*`` keyword arguments override
        preset values.
    show_title : bool, default True
        Show the top title row.
    show_stats : bool, default True
        Show statistics (mean, min, max, std).
    show_controls : bool, default True
        Show the control panel.
    controls_collapsed : bool, default False
        Start with controls hidden while keeping a recoverable ``Controls``
        button in the frontend.
    panel_width_px : int, optional
        Initial diffraction canvas width in CSS pixels. The frontend still lets
        users resize the panel interactively.
    verbose : bool, default True
        Print load timing on construction.
    state : str, pathlib.Path, or dict, optional
        Saved state to restore after construction.

    Examples
    --------
    >>> import numpy as np
    >>> from quantem.widget.showdiffraction import ShowDiffraction

    Single 2D diffraction pattern:

    >>> ShowDiffraction(np.random.rand(256, 256))

    Calibrated stack of diffraction patterns:

    >>> ShowDiffraction(np.random.rand(20, 128, 128), k_pixel_size=0.012)
    """

    _esm = pathlib.Path(__file__).parent / "static" / "showdiffraction.js"
    _CENTER_MODES = ("auto", "manual")
    _CENTER_METHODS = ("symmetry", "auto", "phase_corr")
    _SCALE_MODES = ("linear", "log", "sqrt")
    _LIST_STATE_FIELDS = {"spots", "rings", "custom_phases", "mask_regions"}
    _STATE_FIELDS = (
        "title",
        "n_source_frames",
        "frame_idx",
        "panel_width_px",
        "pixel_size",
        "k_pixel_size",
        "k_calibrated",
        "center_row",
        "center_col",
        "bf_radius",
        "spots",
        "rings",
        "zone_axis",
        "phase_match",
        "show_hkl",
        "snap_enabled",
        "snap_radius",
        "spot_refine",
        "center_mode",
        "calibration_source",
        "calibration_ref_d",
        "calibration_ref_radius",
        "calibration_rms_px",
        "ellipse_ratio",
        "ellipse_angle",
        "ellipse_corrected",
        "dp_colormap",
        "dp_scale_mode",
        "dp_invert",
        "dp_vmin_pct",
        "dp_vmax_pct",
        "show_title",
        "show_stats",
        "show_controls",
        "controls_collapsed",
        "show_profile",
        "profile_log",
        "profile_subtract_background",
        "phase_name",
        "custom_phases",
        "mask_regions",
        "show_mask",
        "profile_theta_min",
        "profile_theta_max",
        "show_azimuthal",
        "refine_method",
        "center_method",
        "identify_elements",
        "identify_custom_only",
    )
    _GEOMETRY_TRAITS = (
        "center_row",
        "center_col",
        "k_pixel_size",
        "k_calibrated",
        "ellipse_ratio",
        "ellipse_angle",
        "ellipse_corrected",
        "mask_regions",
    )
    _PROFILE_TRAITS = (
        "show_profile",
        "profile_subtract_background",
        "profile_theta_min",
        "profile_theta_max",
        "frame_idx",
    )

    # Core state
    widget_version = traitlets.Unicode("unknown").tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    n_frames = traitlets.Int(1).tag(sync=True)
    frame_idx = traitlets.Int(0).tag(sync=True)
    det_rows = traitlets.Int(1).tag(sync=True)
    det_cols = traitlets.Int(1).tag(sync=True)

    frame_bytes = traitlets.Bytes(b"").tag(sync=True)
    # Offline frame stack
    offline_frames = traitlets.Bytes(b"").tag(sync=True)

    # Offline render mode
    offline = traitlets.Bool(False).tag(sync=True)

    # HTML export bridge
    export_request = traitlets.Unicode("").tag(sync=True)
    export_status = traitlets.Unicode("").tag(sync=True)
    export_enabled = traitlets.Bool(True).tag(sync=True)
    export_payload = traitlets.Bytes(b"").tag(sync=True)
    export_payload_id = traitlets.Unicode("").tag(sync=True)
    export_filename = traitlets.Unicode("").tag(sync=True)

    # Detector calibration
    center_row = traitlets.Float(0.0).tag(sync=True)
    center_col = traitlets.Float(0.0).tag(sync=True)
    bf_radius = traitlets.Float(0.0).tag(sync=True)
    pixel_size = traitlets.Float(1.0).tag(sync=True)
    k_pixel_size = traitlets.Float(0.0).tag(sync=True)
    k_calibrated = traitlets.Bool(False).tag(sync=True)

    center_mode = traitlets.Unicode("auto").tag(sync=True)

    calibration_source = traitlets.Unicode("none").tag(sync=True)
    calibration_ref_d = traitlets.Float(0.0).tag(sync=True)
    calibration_ref_radius = traitlets.Float(0.0).tag(sync=True)
    calibration_rms_px = traitlets.Float(0.0).tag(sync=True)

    refine_method = traitlets.Unicode("auto").tag(sync=True)
    center_method = traitlets.Unicode("").tag(sync=True)

    # Ellipse correction
    ellipse_ratio = traitlets.Float(1.0).tag(sync=True)
    ellipse_angle = traitlets.Float(0.0).tag(sync=True)
    ellipse_corrected = traitlets.Bool(False).tag(sync=True)

    # Spots and rings
    spots = traitlets.List(traitlets.Dict()).tag(sync=True)
    snap_enabled = traitlets.Bool(False).tag(sync=True)
    snap_radius = traitlets.Int(5).tag(sync=True)

    rings = traitlets.List(traitlets.Dict()).tag(sync=True)

    spot_refine = traitlets.Bool(True).tag(sync=True)

    # Indexing
    zone_axis = traitlets.Unicode("").tag(sync=True)
    phase_match = traitlets.Unicode("").tag(sync=True)
    show_hkl = traitlets.Bool(True).tag(sync=True)

    # Frontend requests
    _spot_add_request = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    _spot_undo_request = traitlets.Bool(False).tag(sync=True)
    _spot_clear_request = traitlets.Bool(False).tag(sync=True)
    _ring_add_request = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    _ring_undo_request = traitlets.Bool(False).tag(sync=True)
    _ring_clear_request = traitlets.Bool(False).tag(sync=True)
    _calibrate_from_ring_request = traitlets.List(traitlets.Float(), default_value=[]).tag(
        sync=True
    )
    _calibrate_from_spot_request = traitlets.List(traitlets.Float(), default_value=[]).tag(
        sync=True
    )
    _detect_spots_request = traitlets.Int(0).tag(sync=True)  # max_spots, -1 = all
    _detect_rings_request = traitlets.Int(0).tag(sync=True)  # max_rings, -1 = all
    _spot_remove_request = traitlets.Int(0).tag(sync=True)  # spot id
    _spot_move_request = traitlets.List(traitlets.Float(), default_value=[]).tag(
        sync=True
    )  # id, row, col
    _ring_remove_request = traitlets.Int(0).tag(sync=True)  # ring id
    _refine_center_request = traitlets.Bool(False).tag(sync=True)
    _fit_rings_request = traitlets.Bool(False).tag(sync=True)
    _fit_ellipse_request = traitlets.Bool(False).tag(sync=True)
    _calibrate_phase_request = traitlets.Bool(False).tag(sync=True)
    _index_rings_request = traitlets.Bool(False).tag(sync=True)
    _index_spots_request = traitlets.Bool(False).tag(sync=True)
    _identify_request = traitlets.Bool(False).tag(sync=True)
    _auto_request = traitlets.Bool(False).tag(sync=True)
    _merge_request = traitlets.Bool(False).tag(sync=True)
    _quality_request = traitlets.Bool(False).tag(sync=True)
    analysis_status = traitlets.Unicode("").tag(sync=True)
    _quality = traitlets.Dict().tag(sync=True)
    selected_ring_id = traitlets.Int(0).tag(sync=True)

    # Analysis mask
    mask_regions = traitlets.List(traitlets.Dict()).tag(sync=True)
    show_mask = traitlets.Bool(True).tag(sync=True)

    # Phase workbench
    phase_name = traitlets.Unicode("").tag(sync=True)
    custom_phases = traitlets.List(traitlets.Dict()).tag(sync=True)
    _phase_library = traitlets.List(traitlets.Dict()).tag(sync=True)
    identify_elements = traitlets.Unicode("").tag(sync=True)
    identify_custom_only = traitlets.Bool(False).tag(sync=True)
    _identify_results = traitlets.List(traitlets.Dict()).tag(sync=True)

    # Display
    dp_colormap = traitlets.Unicode("inferno").tag(sync=True)
    dp_scale_mode = traitlets.Unicode("log").tag(sync=True)
    dp_invert = traitlets.Bool(False).tag(sync=True)
    dp_vmin_pct = traitlets.Float(0.0).tag(sync=True)
    dp_vmax_pct = traitlets.Float(100.0).tag(sync=True)

    # Profiles
    show_profile = traitlets.Bool(False).tag(sync=True)
    profile_log = traitlets.Bool(True).tag(sync=True)
    profile_subtract_background = traitlets.Bool(False).tag(sync=True)
    profile_theta_min = traitlets.Float(0.0).tag(sync=True)
    profile_theta_max = traitlets.Float(360.0).tag(sync=True)
    _profile_data = traitlets.Bytes(b"").tag(sync=True)  # float32 pairs
    show_azimuthal = traitlets.Bool(False).tag(sync=True)
    _azimuthal_data = traitlets.Bytes(b"").tag(sync=True)  # float32 pairs

    # Statistics
    dp_stats = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0, 0.0, 0.0]).tag(sync=True)

    # UI visibility
    show_title = traitlets.Bool(True).tag(sync=True)
    show_stats = traitlets.Bool(True).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)
    controls_collapsed = traitlets.Bool(False).tag(sync=True)
    panel_width_px = traitlets.Int(384).tag(sync=True)

    @traitlets.validate("center_mode")
    def _validate_center_mode(self, proposal):
        val = proposal["value"]
        if val not in self._CENTER_MODES:
            raise ValueError(f"center_mode must be one of {self._CENTER_MODES}, got {val!r}")
        return val

    @traitlets.validate("frame_idx")
    def _validate_frame_idx(self, proposal):
        # Saved-state bounds
        val = int(proposal["value"])
        n = max(1, int(self.n_frames))
        return max(0, min(val, n - 1))

    @traitlets.validate("dp_scale_mode")
    def _validate_dp_scale_mode(self, proposal):
        val = proposal["value"]
        if val not in self._SCALE_MODES:
            raise ValueError(f"dp_scale_mode must be one of {self._SCALE_MODES}, got {val!r}")
        return val

    def __init__(
        self,
        data: np.ndarray | torch.Tensor,
        k_pixel_size: float | None = None,
        pixel_size: float | None = None,
        center: tuple[float, float] | None = None,
        bf_radius: float | None = None,
        title: str = "",
        snap_enabled: bool = False,
        snap_radius: int = 5,
        spot_refine: bool = True,
        dp_scale_mode: str = "log",
        ui_mode: UiMode = "interactive",
        show_title: bool | None = None,
        show_stats: bool | None = None,
        show_controls: bool | None = None,
        controls_collapsed: bool | None = None,
        panel_width_px: int | None = None,
        offline: bool = False,
        verbose: bool = True,
        state=None,
        **kwargs,
    ):
        # apply after ingest, else clamped against n_frames=1
        initial_frame_idx = kwargs.pop("frame_idx", None)
        super().__init__(**kwargs)
        t_start = time.perf_counter()
        self.widget_version = resolve_widget_version()
        user_k_pixel_size = k_pixel_size is not None
        data, title, pixel_size, k_pixel_size, metadata_calibrated = normalize_data_input(
            data,
            title=title,
            pixel_size=pixel_size,
            k_pixel_size=k_pixel_size,
        )

        self._device = self._best_device()
        self._ingest_data(data)
        if initial_frame_idx is not None:
            self.frame_idx = int(initial_frame_idx)
        self._set_initial_calibration(
            pixel_size,
            k_pixel_size,
            metadata_calibrated=metadata_calibrated,
            user_k_pixel_size=user_k_pixel_size,
        )

        self.title = title
        self.dp_scale_mode = dp_scale_mode
        self.snap_enabled = snap_enabled
        self.snap_radius = snap_radius
        self.spot_refine = spot_refine
        ui = resolve_ui_mode(
            ui_mode,
            defaults={
                "show_title": True,
                "show_stats": True,
                "show_controls": True,
                "controls_collapsed": False,
            },
            overrides={
                "show_title": show_title,
                "show_stats": show_stats,
                "show_controls": show_controls,
                "controls_collapsed": controls_collapsed,
            },
        )
        self.show_title = bool(ui["show_title"])
        self.show_stats = bool(ui["show_stats"])
        self.show_controls = bool(ui["show_controls"])
        self.controls_collapsed = bool(ui["controls_collapsed"])
        if panel_width_px is not None:
            self.panel_width_px = int(panel_width_px)
        self.offline = offline

        self._set_initial_geometry(center, bf_radius)

        self._update_frame()
        self._bake_offline_frames()
        self._phase_library = [{"name": name, **entry} for name, entry in PHASE_LIBRARY.items()]
        self._observe_traits()

        if verbose:
            mem_mb = self._data.nelement() * 4 / 1e6
            print(f"  to {self._device}: {time.perf_counter() - t_start:.2f}s ({mem_mb:.1f} MB)")

        self._load_initial_state(state)

    @staticmethod
    def _best_device() -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _set_initial_calibration(
        self,
        pixel_size: float | None,
        k_pixel_size: float | None,
        *,
        metadata_calibrated: bool,
        user_k_pixel_size: bool,
    ) -> None:
        if pixel_size is not None:
            self.pixel_size = float(pixel_size)
        if k_pixel_size is not None and k_pixel_size > 0:
            self.k_pixel_size = float(k_pixel_size)
            self.k_calibrated = True
            self.calibration_source = "manual" if user_k_pixel_size else "metadata"
        elif metadata_calibrated:
            self.k_calibrated = True
            self.calibration_source = "metadata"

    def _set_initial_geometry(
        self,
        center: tuple[float, float] | None,
        bf_radius: float | None,
    ) -> None:
        if center is None:
            self.center_row = float(self.det_rows / 2)
            self.center_col = float(self.det_cols / 2)
        else:
            self.center_row = float(center[0])
            self.center_col = float(center[1])

        self.bf_radius = (
            float(bf_radius)
            if bf_radius is not None
            else min(self.det_rows, self.det_cols) * BF_RADIUS_FRACTION
        )
        if center is None and bf_radius is None:
            self.auto_detect_center()

    def _observe_traits(self) -> None:
        self.observe(self._update_frame, names=["frame_idx"])
        self.observe(self._bake_offline_frames, names=["offline"])
        self.observe(self._on_spot_add_request, names=["_spot_add_request"])
        self.observe(self._on_spot_undo_request, names=["_spot_undo_request"])
        self.observe(self._on_spot_clear_request, names=["_spot_clear_request"])
        self.observe(self._on_ring_add_request, names=["_ring_add_request"])
        self.observe(self._on_ring_undo_request, names=["_ring_undo_request"])
        self.observe(self._on_ring_clear_request, names=["_ring_clear_request"])
        self.observe(self._on_calibrate_from_ring_request, names=["_calibrate_from_ring_request"])
        self.observe(self._on_calibrate_from_spot_request, names=["_calibrate_from_spot_request"])
        self.observe(self._on_geometry_change, names=list(self._GEOMETRY_TRAITS))
        self.observe(self._on_detect_spots_request, names=["_detect_spots_request"])
        self.observe(self._on_detect_rings_request, names=["_detect_rings_request"])
        self.observe(self._on_spot_remove_request, names=["_spot_remove_request"])
        self.observe(self._on_spot_move_request, names=["_spot_move_request"])
        self.observe(self._on_ring_remove_request, names=["_ring_remove_request"])
        self.observe(self._on_status_request, names=list(self._STATUS_REQUESTS))
        self.observe(self._on_quality_request, names=["_quality_request"])
        self.observe(self._update_profile, names=list(self._PROFILE_TRAITS))
        self.observe(self._update_azimuthal, names=["show_azimuthal", "rings", "frame_idx"])
        self.observe(self._on_export_request_change, names=["export_request"])

    @staticmethod
    def _resolve_state(state) -> dict:
        if isinstance(state, (str, pathlib.Path)):
            return unwrap_state_payload(
                json.loads(pathlib.Path(state).read_text()),
                require_envelope=True,
                expected_widget="ShowDiffraction",
            )
        return unwrap_state_payload(state, expected_widget="ShowDiffraction")

    def _load_initial_state(self, state) -> None:
        if state is None:
            return
        self.load_state_dict(self._resolve_state(state))

    def _ingest_data(self, data):
        array = to_numpy(data)
        is_integer = np.issubdtype(array.dtype, np.integer)
        array = array.astype(np.float32)
        array = np.nan_to_num(array, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if array.size > 2**31 - 1 and self._device.type == "mps":
            self._device = torch.device("cpu")
        if is_integer:
            global_max = float(array.max())
            p999 = float(np.percentile(array, 99.9))
            # sparse counting frames (p999 = 0)
            if p999 > 0 and global_max > p999 * 5:
                array[array > p999 * 3] = 0
        ndim = array.ndim
        if ndim == 2:
            array = array[None, ...]
        elif ndim == 4:
            raise ValueError(
                "ShowDiffraction is for 2D/3D diffraction patterns; 4D input is not supported."
            )
        elif ndim != 3:
            raise ValueError(f"Expected a 2D or 3D array, got {ndim}D")
        self._det_shape = (array.shape[1], array.shape[2])
        self._data = torch.from_numpy(np.ascontiguousarray(array)).to(self._device)
        self.n_frames = int(array.shape[0])
        self.det_rows = self._det_shape[0]
        self.det_cols = self._det_shape[1]
        # new stack: reset merge provenance
        self._n_source_frames = None

    @property
    def detector_shape(self) -> tuple[int, int]:
        """Detector shape as ``(rows, cols)``."""
        return self._det_shape

    @property
    def n_source_frames(self) -> int | None:
        """Source-frame count before merge_frames appended a merged frame."""
        return self._n_source_frames

    @n_source_frames.setter
    def n_source_frames(self, value) -> None:
        self._n_source_frames = None if value is None else int(value)

    def auto_detect_center(self, *, refine: bool = False) -> Self:
        """Find the BF disk center/radius from the summed stack."""
        summed_dp = self._data.sum(dim=0)

        threshold = summed_dp.mean() + summed_dp.std()
        mask = summed_dp > threshold

        total = mask.sum()
        if total == 0:
            return self

        row_coords = torch.arange(self.det_rows, device=self._device, dtype=torch.float32)[:, None]
        col_coords = torch.arange(self.det_cols, device=self._device, dtype=torch.float32)[None, :]
        self.center_row = float((row_coords * mask).sum() / total)
        self.center_col = float((col_coords * mask).sum() / total)
        # Central component
        self.bf_radius = self._central_beam_radius(mask, self.center_row, self.center_col)
        self.center_mode = "auto"
        if refine:
            self.refine_center()
        return self

    def refine_center(self, *, method: str = "symmetry", search_radius: float = 8.0) -> Self:
        """Refine the center with symmetry, phase correlation, or auto."""
        if method not in self._CENTER_METHODS:
            raise ValueError(f"unknown refine method {method!r}")

        picked = pick_center(
            self._displayed_frame().astype(np.float64),
            method=method,
            mask=self._analysis_mask(),
            guess=(self.center_row, self.center_col),
            search_radius=search_radius,
        )
        self.center_row, self.center_col = float(picked["row"]), float(picked["col"])
        self.center_mode = "auto"
        self.center_method = picked["method"]
        return self

    def _central_beam_radius(self, mask, center_row: float, center_col: float) -> float:
        mask_np = mask.detach().cpu().numpy()
        try:
            from scipy.ndimage import label
        except Exception:
            return float(np.sqrt(float(mask_np.sum()) / np.pi))
        labels, n_labels = label(mask_np)
        if n_labels == 0:
            return 0.0
        row_idx = int(min(max(round(center_row), 0), mask_np.shape[0] - 1))
        col_idx = int(min(max(round(center_col), 0), mask_np.shape[1] - 1))
        central_label = int(labels[row_idx, col_idx])
        if central_label == 0:
            # Beam stop
            comp_rows, comp_cols = np.nonzero(labels)
            nearest = int(np.argmin((comp_rows - center_row) ** 2 + (comp_cols - center_col) ** 2))
            central_label = int(labels[comp_rows[nearest], comp_cols[nearest]])
        area = float((labels == central_label).sum())
        return float(np.sqrt(area / np.pi))

    def set_center(self, row: float, col: float) -> Self:
        """Set the diffraction center to (row, col) and mark the mode manual."""
        self.center_row = float(row)
        self.center_col = float(col)
        self.center_mode = "manual"
        return self

    def _get_frame(self, idx: int) -> np.ndarray:
        idx = max(0, min(int(idx), self.n_frames - 1))
        return self._data[idx].cpu().numpy().astype(np.float32)

    def _displayed_frame(self) -> np.ndarray:
        return self._get_frame(self.frame_idx)

    def _update_frame(self, change=None):
        frame = self._displayed_frame()
        self.dp_stats = [
            float(frame.mean()),
            float(frame.min()),
            float(frame.max()),
            float(frame.std()),
        ]
        self.frame_bytes = frame.tobytes()

    def _bake_offline_frames(self, change=None) -> None:
        # Offline stack
        if self.offline and self.n_frames > 1 and getattr(self, "_data", None) is not None:
            frames = self._data.cpu().numpy().astype(np.float32)
            self.offline_frames = np.ascontiguousarray(frames).tobytes()
        else:
            self.offline_frames = b""

    def _compute_spot_info(
        self,
        row: float,
        col: float,
        row_err: float = 0.0,
        col_err: float = 0.0,
        frame_idx: int | None = None,
    ) -> dict:
        d_row = row - self.center_row
        d_col = col - self.center_col
        r_pixels = float(
            corrected_radius(
                d_row,
                d_col,
                ellipse_ratio=self.ellipse_ratio,
                ellipse_angle=self.ellipse_angle,
                ellipse_corrected=self.ellipse_corrected,
            )
        )

        # Radial uncertainty
        if r_pixels > 0:
            r_err = math.hypot((d_row / r_pixels) * row_err, (d_col / r_pixels) * col_err)
        else:
            r_err = math.hypot(row_err, col_err)

        # sample the record's source frame
        frame = self._get_frame(self.frame_idx if frame_idx is None else frame_idx)
        r_int = max(0, min(self.det_rows - 1, int(round(row))))
        c_int = max(0, min(self.det_cols - 1, int(round(col))))
        intensity = float(frame[r_int, c_int])

        if self.k_calibrated and self.k_pixel_size > 0 and r_pixels > 0:
            g_magnitude = r_pixels * self.k_pixel_size
            d_spacing = 1.0 / g_magnitude
            # Propagated d error
            frac = r_err / r_pixels
            g_err = g_magnitude * frac
            d_err = d_spacing * frac
        else:
            g_magnitude = d_spacing = g_err = d_err = None

        return {
            "d_spacing": d_spacing,
            "d_spacing_err": d_err,
            "g_magnitude": g_magnitude,
            "g_magnitude_err": g_err,
            "r_pixels": r_pixels,
            "r_pixels_err": r_err,
            "intensity": intensity,
        }

    def _with_angles(self, spots) -> list:
        if not spots:
            return spots
        reference = spots[0]
        ref_row, ref_col = self._spot_vector(reference)
        ref_radius = math.hypot(ref_row, ref_col)
        ref_error = math.hypot(reference.get("row_err", 0.0), reference.get("col_err", 0.0))
        with_angles = []
        for spot in spots:
            delta_row, delta_col = self._spot_vector(spot)
            radius = math.hypot(delta_row, delta_col)
            if ref_radius > 0 and radius > 0:
                angle = self._measured_angle(reference, spot)
                spot_error = math.hypot(spot.get("row_err", 0.0), spot.get("col_err", 0.0))
                angle_err = math.degrees(math.hypot(spot_error / radius, ref_error / ref_radius))
            else:
                angle = None
                angle_err = None
            with_angles.append({**spot, "angle_deg": angle, "angle_deg_err": angle_err})
        return with_angles

    def detect_spots(
        self,
        max_spots: int | None = None,
        min_distance: int = 6,
        min_relative: float = 0.1,
        exclude_radius: float | None = None,
        replace: bool = True,
    ) -> Self:
        """Detect Bragg spots with contrast at least ``min_relative`` of the strongest peak."""
        frame = self._displayed_frame().astype(np.float64)
        n_rows, n_cols = frame.shape
        if exclude_radius is None:
            exclude_radius = max(self.bf_radius, 2.0 * float(min_distance))
        try:
            from scipy.ndimage import gaussian_filter, maximum_filter
        except Exception:
            return self
        work = np.log1p(np.clip(frame - frame.min(), 0.0, None))
        work = work - gaussian_filter(work, sigma=max(2.0, float(min_distance)))

        size = max(3, int(min_distance) | 1)  # odd window
        local_max = maximum_filter(work, size=size) == work
        rows = np.arange(n_rows)[:, None]
        cols = np.arange(n_cols)[None, :]
        radius = np.hypot(rows - self.center_row, cols - self.center_col)
        local_max &= radius > float(exclude_radius)
        exclusion = self._analysis_mask()
        if exclusion is not None:
            local_max &= ~exclusion
        local_max[0, :] = local_max[-1, :] = False
        local_max[:, 0] = local_max[:, -1] = False
        coords = np.argwhere(local_max)
        if replace:
            self.clear_spots()
        if coords.size == 0:
            return self
        prominence = work[coords[:, 0], coords[:, 1]]
        # contrast relative to the strongest peak, with a noise floor on noisy data
        contrast = np.expm1(prominence)
        sigma = 1.4826 * float(np.median(np.abs(work - np.median(work))))
        level = max(min_relative * float(contrast.max()), float(np.expm1(5.0 * sigma)))
        keep = (prominence > 0) & (contrast >= level)
        coords, prominence = coords[keep], prominence[keep]
        if coords.size:
            # min over opposite samples: crests high both sides, lone neighbors one
            angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
            ring_r = float(max(3, min_distance))
            rr = np.clip(coords[:, :1] + ring_r * np.sin(angles), 0, n_rows - 1).astype(int)
            cc = np.clip(coords[:, 1:] + ring_r * np.cos(angles), 0, n_cols - 1).astype(int)
            ring_vals = work[rr, cc]
            peak = work[coords[:, 0], coords[:, 1]]
            hi = np.minimum(ring_vals[:, :8], ring_vals[:, 8:]).max(axis=1)
            lo = np.percentile(ring_vals, 10, axis=1)
            isolated = (peak - hi) >= 0.5 * np.maximum(peak - lo, 1e-9)
            coords, prominence = coords[isolated], prominence[isolated]
        kept: list[int] = []
        for idx in np.argsort(-prominence):
            r0, c0 = coords[idx]
            if all(np.hypot(r0 - coords[j][0], c0 - coords[j][1]) > min_distance for j in kept):
                kept.append(int(idx))
        order = np.asarray(kept, dtype=int)
        if max_spots is not None:
            order = order[: int(max_spots)]
        for r0, c0 in coords[order]:
            self.add_spot(float(r0), float(c0))
        return self

    def detect_rings(
        self,
        max_rings: int | None = None,
        prominence_rel: float = 0.05,
        min_separation: int = 5,
        exclude_radius: float | None = None,
        replace: bool = True,
    ) -> Self:
        """Detect Debye-Scherrer rings from radial profile peaks (max_rings=None keeps all)."""
        try:
            radii_px, intensity = self._radial_profile()
        except Exception:
            return self
        y = np.asarray(intensity, dtype=np.float64)
        if y.size < 5:
            return self
        # import before clearing picks
        try:
            from scipy.ndimage import gaussian_filter1d
            from scipy.signal import find_peaks
        except Exception:
            return self
        if replace:
            self.clear_rings()
        if exclude_radius is None:
            exclude_radius = self.bf_radius
        y_log = np.log1p(np.clip(y - y.min(), 0.0, None))
        detrended = y_log - gaussian_filter1d(y_log, sigma=max(3.0, y_log.size / 20.0))
        span = float(detrended.max() - detrended.min())
        prominence = prominence_rel * span if span > 0 else None
        peaks, props = find_peaks(
            detrended, prominence=prominence, distance=max(1, int(min_separation))
        )
        if peaks.size == 0:
            return self
        outside_beam = radii_px[peaks] > float(exclude_radius)
        peaks = peaks[outside_beam]
        prominences = props["prominences"][outside_beam]
        if peaks.size == 0:
            return self
        strongest = np.argsort(prominences)[::-1]
        if max_rings is not None:
            strongest = strongest[: int(max_rings)]
        for p in sorted(peaks[strongest]):
            self.add_ring(float(radii_px[p]))
        return self

    def _on_detect_spots_request(self, change=None):
        n = self._detect_spots_request
        if n:
            self.detect_spots(max_spots=int(n) if n > 0 else None)
            self._detect_spots_request = 0

    def _on_detect_rings_request(self, change=None):
        n = self._detect_rings_request
        if n:
            self.detect_rings(max_rings=int(n) if n > 0 else None)
            self._detect_rings_request = 0

    def _snap_to_peak(self, row: float, col: float) -> tuple[float, float]:
        frame = self._displayed_frame()
        r, c = int(round(row)), int(round(col))
        radius = int(self.snap_radius)
        r0 = max(0, r - radius)
        r1 = min(self.det_rows, r + radius + 1)
        c0 = max(0, c - radius)
        c1 = min(self.det_cols, c + radius + 1)
        region = frame[r0:r1, c0:c1]
        if region.size == 0:
            return float(row), float(col)
        idx = np.unravel_index(region.argmax(), region.shape)
        return float(r0 + idx[0]), float(c0 + idx[1])

    def _pick_spot_fields(self, row: float, col: float) -> dict:
        # Position + measurement fields per the current pick mode (fit/snap/exact)
        raw_row, raw_col = float(row), float(col)
        row_err = col_err = 0.0
        fit_quality = None
        if self.spot_refine:
            fit = fit_gaussian_spot(
                self._displayed_frame(),
                raw_row,
                raw_col,
                half_window=self.snap_radius,
            )
            if fit is not None:
                row, col = fit["row"], fit["col"]
                row_err, col_err = fit["row_err"], fit["col_err"]
                fit_quality = fit["fit_quality"]
        elif self.snap_enabled:
            row, col = self._snap_to_peak(raw_row, raw_col)
        info = self._compute_spot_info(row, col, row_err=row_err, col_err=col_err)
        return {
            "row": float(row),
            "col": float(col),
            "raw_row": raw_row,
            "raw_col": raw_col,
            "row_err": float(row_err),
            "col_err": float(col_err),
            "fit_quality": fit_quality,
            # source-frame provenance
            "frame_idx": int(self.frame_idx),
            **empty_index_fields(),
            **info,
        }

    def add_spot(self, row: float, col: float) -> Self:
        """Add a spot, optionally refining or snapping it."""
        spot = {
            "id": next_record_id(self.spots),
            "angle_deg": None,
            "angle_deg_err": None,
            **self._pick_spot_fields(row, col),
        }
        self.spots = self._with_angles(list(self.spots) + [spot])
        return self

    def move_spot(self, spot_id: int, row: float, col: float) -> Self:
        """Move the spot with id ``spot_id``, re-picking it at the new position."""
        idx = next((i for i, s in enumerate(self.spots) if s["id"] == spot_id), None)
        if idx is None:
            return self
        spots = list(self.spots)
        spots[idx] = {**spots[idx], **self._pick_spot_fields(row, col)}
        self.spots = self._with_angles(spots)
        return self

    def clear_spots(self) -> Self:
        """Remove all spots."""
        self.spots = []
        return self

    def undo_spot(self) -> Self:
        """Remove the most recently added spot."""
        if self.spots:
            self.spots = list(self.spots[:-1])
        return self

    def remove_spot(self, spot_id: int) -> Self:
        """Remove the spot with id ``spot_id`` (no-op if not present)."""
        remaining = [s for s in self.spots if s["id"] != spot_id]
        if len(remaining) != len(self.spots):
            self.spots = self._with_angles(remaining)
        return self

    def _on_spot_add_request(self, change=None):
        val = self._spot_add_request
        if val and len(val) == 2:
            self.add_spot(val[0], val[1])
            self._spot_add_request = []

    def _on_spot_undo_request(self, change=None):
        if self._spot_undo_request:
            self.undo_spot()
            self._spot_undo_request = False

    def _on_spot_clear_request(self, change=None):
        if self._spot_clear_request:
            self.clear_spots()
            self._spot_clear_request = False

    def _on_spot_remove_request(self, change=None):
        if self._spot_remove_request:
            self.remove_spot(int(self._spot_remove_request))
            self._spot_remove_request = 0

    def _on_spot_move_request(self, change=None):
        val = self._spot_move_request
        if val and len(val) == 3:
            self.move_spot(int(val[0]), val[1], val[2])
            self._spot_move_request = []

    def _recompute_spots(self):
        if not self.spots:
            return
        spots = [
            {
                **s,
                **self._compute_spot_info(
                    s["row"],
                    s["col"],
                    s.get("row_err", 0.0),
                    s.get("col_err", 0.0),
                    frame_idx=s.get("frame_idx"),
                ),
            }
            for s in self.spots
        ]
        self.spots = self._with_angles(spots)

    def _on_geometry_change(self, change=None):
        # Derived geometry
        self._recompute_spots()
        self._recompute_rings()
        self._update_profile()
        self._update_azimuthal()

    def _compute_ring_info(self, radius_px: float, frame_idx: int | None = None) -> dict:
        if self.k_calibrated and self.k_pixel_size > 0:
            g_magnitude = float(radius_px) * self.k_pixel_size
            d_spacing = 1.0 / g_magnitude if g_magnitude > 0 else None
        else:
            g_magnitude = d_spacing = None
        # sample the record's source frame
        radii_px, intensity = self._radial_profile(frame_idx=frame_idx)
        ring_intensity = (
            float(intensity[int(np.argmin(np.abs(radii_px - radius_px)))])
            if radii_px.size
            else 0.0
        )
        return {
            "radius_px": float(radius_px),
            "g_magnitude": g_magnitude,
            "d_spacing": d_spacing,
            "intensity": ring_intensity,
        }

    def add_ring(self, radius_px: float) -> Self:
        """Add a ring at radius_px from the center (polycrystalline d-spacing pick)."""
        if radius_px <= 0:
            raise ValueError(f"radius_px must be positive, got {radius_px}")
        ring = {
            "id": next_record_id(self.rings),
            # source-frame provenance
            "frame_idx": int(self.frame_idx),
            **empty_index_fields(),
            **self._compute_ring_info(radius_px),
        }
        self.rings = list(self.rings) + [ring]
        return self

    def clear_rings(self) -> Self:
        """Remove all rings."""
        self.rings = []
        return self

    def undo_ring(self) -> Self:
        """Remove the most recently added ring."""
        if self.rings:
            self.rings = list(self.rings[:-1])
        return self

    def remove_ring(self, ring_id: int) -> Self:
        """Remove the ring with id ``ring_id`` (no-op if not present)."""
        remaining = [r for r in self.rings if r["id"] != ring_id]
        if len(remaining) != len(self.rings):
            self.rings = remaining
        return self

    def _recompute_rings(self):
        if not self.rings:
            return
        calibrated = self.k_calibrated and self.k_pixel_size > 0
        rings = []
        for r in self.rings:
            ring = {**r, **self._compute_ring_info(r["radius_px"], frame_idx=r.get("frame_idx"))}
            if ring.get("fwhm_px") is not None:
                ring["fwhm_inv_angstrom"] = (
                    ring["fwhm_px"] * self.k_pixel_size if calibrated else None
                )
            rings.append(ring)
        self.rings = rings

    def fit_ring_profile(
        self,
        *,
        window: float | None = None,
        model: str = "gaussian",
        subtract_background: bool = True,
    ) -> Self:
        """Fit each ring peak and store refined radius, width, area, and quality."""
        if not self.rings:
            raise ValueError("no rings to fit; call add_ring or detect_rings first")

        radii_px, intensity = self._radial_profile()
        profile = intensity.astype(np.float64)
        if subtract_background:
            try:
                _, background = self.radial_background()
                profile = profile - background.astype(np.float64)
            except ValueError:
                pass

        calibrated = self.k_calibrated and self.k_pixel_size > 0
        updates = fit_ring_peaks(radii_px, profile, self.rings, model=model, window=window)
        rings = []
        for ring, update in zip(self.rings, updates):
            ring = dict(ring)
            if update is None:
                ring["fit_quality"] = None
                rings.append(ring)
                continue
            raw_radius = update.pop("raw_radius_px")
            ring.setdefault("raw_radius_px", raw_radius)
            # fit re-measures on the displayed frame
            ring["frame_idx"] = int(self.frame_idx)
            ring.update(self._compute_ring_info(update["radius_px"]))
            ring.update(update)
            ring["fwhm_inv_angstrom"] = ring["fwhm_px"] * self.k_pixel_size if calibrated else None
            rings.append(ring)
        self.rings = rings
        return self

    def _on_ring_add_request(self, change=None):
        val = self._ring_add_request
        if val and len(val) == 1:
            try:
                self.add_ring(val[0])
            except ValueError:
                pass
            self._ring_add_request = []

    def _on_ring_undo_request(self, change=None):
        if self._ring_undo_request:
            self.undo_ring()
            self._ring_undo_request = False

    def _on_ring_clear_request(self, change=None):
        if self._ring_clear_request:
            self.clear_rings()
            self._ring_clear_request = False

    def _on_ring_remove_request(self, change=None):
        if self._ring_remove_request:
            self.remove_ring(int(self._ring_remove_request))
            self._ring_remove_request = 0

    # Request dispatch
    _STATUS_REQUESTS = {
        "_refine_center_request": ("Refine", "_do_refine_center"),
        "_fit_rings_request": ("Ring fit", "_do_fit_rings"),
        "_fit_ellipse_request": ("Ellipse", "_do_fit_ellipse"),
        "_calibrate_phase_request": ("Phase calibration", "_do_calibrate_phase"),
        "_index_rings_request": ("Ring indexing", "_do_index_rings"),
        "_index_spots_request": ("Spot indexing", "_do_index_spots"),
        "_identify_request": ("Identify", "_do_identify"),
        "_auto_request": ("Auto", "_do_auto"),
        "_merge_request": ("Merge", "_do_merge"),
    }

    def _on_status_request(self, change):
        if not change["new"]:
            return
        prefix, method = self._STATUS_REQUESTS[change["name"]]
        try:
            # KeyError/TypeError cover malformed custom_phases entries
            self.analysis_status = getattr(self, method)()
        except (ValueError, ImportError, KeyError, TypeError) as exc:
            self.analysis_status = f"{prefix} failed: {exc}"
        finally:
            setattr(self, change["name"], False)
        try:
            self.quality_report()
        except (ValueError, ImportError):
            pass

    def _on_quality_request(self, change=None):
        if not self._quality_request:
            return
        try:
            self.quality_report()
            self.analysis_status = "Quality updated"
        except (ValueError, ImportError) as exc:
            self.analysis_status = f"Quality failed: {exc}"
        self._quality_request = False

    def _do_refine_center(self) -> str:
        self.refine_center(method=self.refine_method)
        return f"Center ({self.center_row:.1f}, {self.center_col:.1f}) via {self.center_method}"

    def _do_fit_rings(self) -> str:
        self.fit_ring_profile()
        n_ok = sum(1 for r in self.rings if r.get("fit_quality") is not None)
        status = f"Fitted {n_ok}/{len(self.rings)} rings"
        try:
            tex = self.texture()
            status += f", texture {tex['strength']:.2f} at {tex['angle_deg']:.0f}°"
        except (ValueError, ImportError):
            pass
        return status

    def _do_fit_ellipse(self) -> str:
        report = self.fit_ellipse()
        return f"Ellipse ratio {report['ratio']:.3f} at {report['angle_deg']:.1f}°"

    def _do_calibrate_phase(self) -> str:
        phase = self._require_phase()
        self.calibrate_from_phase(phase)
        return (
            f"Calibrated from {phase.name}: k={self.k_pixel_size:.5f} 1/Å/px "
            f"(rms {self.calibration_rms_px:.2f} px)"
        )

    def _do_index_rings(self) -> str:
        phase = self._require_phase()
        self.index_rings(phase)
        n = sum(1 for r in self.rings if r.get("hkl"))
        return f"Indexed {n}/{len(self.rings)} rings against {phase.name}"

    def _do_index_spots(self) -> str:
        phase = self._require_phase()
        self.index_spots(phase)
        zone = f", zone {self.zone_axis}" if self.zone_axis else ""
        return f"Indexed spots against {phase.name}{zone}"

    def _do_identify(self) -> str:
        ranked = self.search_phases()
        return self._identify_summary(ranked)

    def _do_auto(self) -> str:
        self.run_auto()
        return self.analysis_status

    def _do_merge(self) -> str:
        report = self.merge_frames()
        status = f"Merged {report['n_used']}/{report['n_frames']} frames"
        if "after" in report:
            status += (
                f", ring coverage {report['before']['coverage']:.2f} to "
                f"{report['after']['coverage']:.2f}"
            )
        return status

    def _selected_phase(self) -> Phase | None:
        if not self.phase_name:
            return None
        if self.phase_name in PHASE_LIBRARY:
            return library_phase(self.phase_name)
        for entry in self.custom_phases:
            if entry.get("name") == self.phase_name:
                return self._custom_phase(entry)
        return None

    @staticmethod
    def _custom_phase(entry: dict) -> Phase:
        a = float(entry["a"])
        return Phase(
            entry["name"],
            a,
            float(entry.get("b", a)),
            float(entry.get("c", a)),
            float(entry.get("alpha", 90.0)),
            float(entry.get("beta", 90.0)),
            float(entry.get("gamma", 90.0)),
            absences=entry.get("absences", "none"),
        )

    def _require_phase(self) -> Phase:
        phase = self._selected_phase()
        if phase is None:
            raise ValueError("no phase selected; set phase_name or add a custom phase")
        return phase

    def _all_phases(self, custom_only: bool = False) -> list[Phase]:
        phases = [] if custom_only else [library_phase(name) for name in PHASE_LIBRARY]
        for entry in self.custom_phases:
            try:
                phases.append(self._custom_phase(entry))
            except (KeyError, ValueError, TypeError):
                continue
        return phases

    def run_auto(
        self,
        phase: Phase | None = None,
        *,
        max_rings: int = 8,
        exclude_radius: float | None = None,
    ) -> Self:
        """Run center finding, ring detection, fitting, calibration, and indexing.

        Silent on success; ``analysis_status`` only reports steps that failed.
        """
        phase = phase or self._selected_phase()
        problems = []
        self.auto_detect_center(refine=True)
        self.detect_rings(max_rings=max_rings, exclude_radius=exclude_radius)
        if not self.rings:
            problems.append("ring detection failed (no rings found)")
        if self.rings:
            try:
                self.fit_ring_profile()
                if all(r.get("fit_quality") is None for r in self.rings):
                    problems.append("ring fit failed")
            except (ValueError, ImportError):
                problems.append("ring fit failed")
        if phase is None and self.phase_name:
            problems.append(f'calibration failed (phase "{self.phase_name}" not found)')
        if phase is not None and self.rings:
            try:
                self.calibrate_from_phase(phase)
            except ValueError as exc:
                problems.append(f"calibration failed ({exc})")
                phase = None
        if phase is not None and self.k_calibrated:
            try:
                self.index_rings(phase)
                if not any(r.get("hkl") for r in self.rings):
                    problems.append("indexing failed (no rings matched)")
            except ValueError as exc:
                problems.append(f"indexing failed ({exc})")
            unindexed = sum(1 for r in self.rings if not r.get("hkl"))
            if unindexed and any(r.get("hkl") for r in self.rings):
                problems.append(
                    f"calibration left {unindexed} of {len(self.rings)} rings unindexed "
                    "(check excluded regions)"
                )
        self.analysis_status = "Auto: " + ", ".join(problems) if problems else ""
        return self

    def merge_frames(
        self, *, statistic: str = "mean", align: bool = True, max_shift: float = 8.0
    ) -> dict:
        """Align the stack and append the combined pattern as a new frame."""
        if self.n_frames < 2:
            raise ValueError("merge_frames needs a multi-frame stack")
        if statistic not in ("mean", "median", "max"):
            raise ValueError(f"statistic must be mean, median or max, got {statistic!r}")
        frames = self._data.cpu().numpy().astype(np.float64)
        # merge original frames only
        n_source = getattr(self, "_n_source_frames", None)
        if n_source is not None:
            frames = frames[:n_source]
        if align:
            aligned, shifts, used = align_frames(frames, max_shift=max_shift)
        else:
            aligned, shifts, used = frames, [(0.0, 0.0)] * len(frames), [True] * len(frames)
        if not any(used):
            raise ValueError("no frames survived alignment; raise max_shift or set align=False")
        stack = np.asarray([f for f, u in zip(aligned, used) if u])
        merged = getattr(np, statistic)(stack, axis=0)
        report = {
            "n_frames": int(len(frames)),
            "n_used": int(len(stack)),
            "shifts": [(float(s[0]), float(s[1])) for s in shifts],
            "used": [bool(u) for u in used],
        }
        if self.rings:
            r0 = max(r["radius_px"] for r in self.rings)
            center = (self.center_row, self.center_col)
            mask = self._analysis_mask()
            before_idx = min(self.frame_idx, len(frames) - 1)
            report["before"] = ring_uniformity(frames[before_idx], center, r0, mask=mask)
            report["after"] = ring_uniformity(merged, center, r0, mask=mask)
        self._ingest_data(np.concatenate([frames, merged[None]], axis=0).astype(np.float32))
        self._n_source_frames = int(len(frames))
        self.frame_idx = self.n_frames - 1
        self._update_frame()
        self._bake_offline_frames()
        # refresh panes explicitly
        self._update_profile()
        self._update_azimuthal()
        return report

    def quality_report(self) -> dict:
        """QC snapshot: center method, calibration, ellipse, ring fits,
        unexplained rings, mask coverage, and outermost-ring SNR.
        """
        mask = self._analysis_mask()
        indexed = [r for r in self.rings if r.get("hkl")]
        report = {
            # snapshot attribution
            "frame_idx": int(self.frame_idx),
            "center": {"method": self.center_method or self.center_mode},
            "calibration": {
                "source": self.calibration_source,
                "k_pixel_size": float(self.k_pixel_size),
                "rms_px": float(self.calibration_rms_px),
            },
            "ellipse": {
                "ratio": float(self.ellipse_ratio),
                "angle_deg": float(self.ellipse_angle),
                "corrected": bool(self.ellipse_corrected),
            },
            "rings": [{"id": r["id"], "fit_quality": r.get("fit_quality")} for r in self.rings],
            "n_unexplained_rings": (len(self.rings) - len(indexed)) if indexed else 0,
            "mask_coverage_pct": float(mask.mean() * 100.0) if mask is not None else 0.0,
        }
        if self.rings:
            outermost = max(self.rings, key=lambda r: r["radius_px"])
            # SNR on the frame the ring was measured on
            snr_idx = outermost.get("frame_idx")
            frame = (
                self._displayed_frame() if snr_idx is None else self._get_frame(snr_idx)
            ).astype(np.float64)
            report["ring_snr"] = ring_uniformity(
                frame, (self.center_row, self.center_col), float(outermost["radius_px"]), mask=mask
            )
        self._quality = report
        return report

    def _update_profile(self, change=None):
        if not self.show_profile:
            self._profile_data = b""
            return
        sector = (self.profile_theta_min, self.profile_theta_max)
        sector = None if sector == (0.0, 360.0) else sector
        try:
            radii, intensity = self.radial_profile(
                units="px",
                subtract_background=self.profile_subtract_background,
                angular_range=sector,
            )
        except ValueError as exc:
            self.analysis_status = f"Background subtract failed: {exc}"
            radii, intensity = self.radial_profile(units="px", angular_range=sector)
        self._profile_data = pack_float32_halves(radii, intensity)

    def _update_azimuthal(self, change=None):
        if not self.show_azimuthal:
            self._azimuthal_data = b""
            return
        try:
            self._azimuthal_data = pack_float32_halves(*self.azimuthal_profile())
        except ValueError as exc:
            self.analysis_status = f"Azimuthal profile failed: {exc}"
            self._azimuthal_data = b""

    def _radial_profile(
        self,
        *,
        n_bins: int | None = None,
        max_radius: float | None = None,
        center: tuple[float, float] | None = None,
        angular_range: tuple[float, float] | None = None,
        frame_idx: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        frame = self._displayed_frame() if frame_idx is None else self._get_frame(frame_idx)
        return radial_profile_px(
            frame,
            center=center or (self.center_row, self.center_col),
            n_bins=n_bins,
            max_radius=max_radius,
            mask=self._analysis_mask(),
            angular_range=angular_range,
            ellipse_ratio=self.ellipse_ratio,
            ellipse_angle=self.ellipse_angle,
            ellipse_corrected=self.ellipse_corrected,
        )

    def _analysis_mask(self) -> "np.ndarray | None":
        return build_analysis_mask(
            (self.det_rows, self.det_cols),
            self.mask_regions,
            (self.center_row, self.center_col),
        )

    def radial_profile(
        self,
        *,
        n_bins: int | None = None,
        max_radius: float | None = None,
        center: tuple[float, float] | None = None,
        units: str = "auto",
        angular_range: tuple[float, float] | None = None,
        subtract_background: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Azimuthally averaged profile in px, g, or d units.

        ``units="q"`` is kept as a legacy alias for ``"g"``.
        """
        if units not in ("auto", "px", "g", "q", "d"):
            raise ValueError(f"units must be 'auto', 'px', 'g', 'q' or 'd', got {units!r}")
        if n_bins is not None and n_bins <= 0:
            raise ValueError(f"n_bins must be positive, got {n_bins}")
        if max_radius is not None and max_radius <= 0:
            raise ValueError(f"max_radius must be positive, got {max_radius}")
        if angular_range is not None and len(angular_range) != 2:
            raise ValueError("angular_range must be a (start_deg, end_deg) pair")
        calibrated = self.k_calibrated and self.k_pixel_size > 0
        if units in ("g", "q", "d") and not calibrated:
            raise ValueError(
                f"radial_profile(units={units!r}) needs a calibrated pattern; call "
                "calibrate_from_ring / calibrate_from_spot / calibrate_from_phase first"
            )
        radii_px, intensity = self._radial_profile(
            n_bins=n_bins, max_radius=max_radius, center=center, angular_range=angular_range
        )
        if subtract_background:
            _, background = self.radial_background(
                n_bins=n_bins, max_radius=max_radius, center=center
            )
            intensity = intensity - background
        if units == "auto":
            units = "g" if calibrated else "px"
        if units == "px":
            return radii_px, intensity
        if units in ("g", "q"):
            return (radii_px * self.k_pixel_size).astype(np.float32), intensity
        keep = radii_px > 0
        d_axis = (1.0 / (radii_px[keep] * self.k_pixel_size)).astype(np.float32)
        return d_axis, intensity[keep]

    def _ring_radius_for(self, ring_id: int | None, radius_px: float | None) -> float:
        if radius_px is not None:
            if radius_px <= 0:
                raise ValueError(f"radius_px must be positive, got {radius_px}")
            return float(radius_px)
        if not self.rings:
            raise ValueError("no ring to analyze; call detect_rings / add_ring or pass radius_px")
        if ring_id is None:
            return float(max(self.rings, key=lambda r: r["radius_px"])["radius_px"])
        matches = [r for r in self.rings if r["id"] == ring_id]
        if not matches:
            raise ValueError(f"no ring with id {ring_id}; have {[r['id'] for r in self.rings]}")
        return float(matches[0]["radius_px"])

    def _ring_half_width(self, radius_px: float) -> float:
        """Default annulus half-width that keeps neighboring rings out."""
        gaps = [abs(radius_px - r["radius_px"]) for r in self.rings if r["radius_px"] != radius_px]
        return min(6.0, max(3.0, min(gaps) / 2.0)) if gaps else max(6.0, 0.25 * radius_px)

    def azimuthal_profile(
        self,
        *,
        ring_id: int | None = None,
        radius_px: float | None = None,
        width: float | None = None,
        n_theta: int = 180,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Intensity vs azimuth around a ring."""
        radius = self._ring_radius_for(ring_id, radius_px)
        half_width = float(width) if width is not None else self._ring_half_width(radius)
        return azimuthal_profile_from_frame(
            self._displayed_frame(),
            center=(self.center_row, self.center_col),
            radius_px=radius,
            half_width=half_width,
            n_theta=n_theta,
            mask=self._analysis_mask(),
            ellipse_ratio=self.ellipse_ratio,
            ellipse_angle=self.ellipse_angle,
            ellipse_corrected=self.ellipse_corrected,
        )

    def texture(
        self,
        *,
        ring_id: int | None = None,
        radius_px: float | None = None,
        width: float | None = None,
        n_theta: int = 180,
        return_profile: bool = False,
    ) -> dict:
        """Order-2 ring texture: strength in [0, 1] and 180-degree angle."""
        theta_deg, intensity = self.azimuthal_profile(
            ring_id=ring_id, radius_px=radius_px, width=width, n_theta=n_theta
        )
        return texture_from_profile(theta_deg, intensity, return_profile=return_profile)

    def fit_ellipse(self, ring_id: int | None = None, *, n_theta: int = 180) -> dict:
        """Fit ellipse distortion from ring radius vs azimuth."""
        radius = self._ring_radius_for(ring_id, None)
        half_width = self._ring_half_width(radius)
        theta_centers, counts, _, weight_sum, weighted_radius_sum = ring_sectors(
            self._displayed_frame(),
            center=(self.center_row, self.center_col),
            radius_px=radius,
            half_width=half_width,
            n_theta=n_theta,
            mask=self._analysis_mask(),
            use_corrected_radius=False,
        )
        report = fit_ellipse_from_sectors(theta_centers, counts, weight_sum, weighted_radius_sum)
        self.ellipse_ratio = report["ratio"]
        self.ellipse_angle = report["angle_deg"]
        return report

    def apply_ellipse_correction(self, *, enable: bool = True) -> Self:
        """Enable or disable radius circularization by the fitted ellipse."""
        self.ellipse_corrected = bool(enable)
        return self

    def radial_background(
        self,
        *,
        n_bins: int | None = None,
        max_radius: float | None = None,
        center: tuple[float, float] | None = None,
        method: str = "power",
        poly_order: int = 3,
        peak_windows: list[tuple[float, float]] | None = None,
        exclude_radius: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit a smooth radial background while excluding peaks."""
        radii_px, intensity = self._radial_profile(
            n_bins=n_bins, max_radius=max_radius, center=center
        )
        if exclude_radius is None:
            exclude_radius = self.bf_radius
        if peak_windows is None:
            peak_windows = []
            for ring in self.rings:
                half = ring.get("fwhm_px") or 6.0
                peak_windows.append((ring["radius_px"] - half, ring["radius_px"] + half))
        return radii_px, fit_radial_background(
            radii_px,
            intensity,
            peak_windows=peak_windows,
            exclude_radius=exclude_radius,
            method=method,
            poly_order=poly_order,
        )

    # Indexing and phase identification
    def _require_calibrated(self) -> None:
        if not (self.k_calibrated and self.k_pixel_size > 0):
            raise ValueError(
                "pattern is uncalibrated; call calibrate_from_ring / calibrate_from_spot first"
            )

    def _match_report(self, phase: Phase, d_values: Sequence[float | None], tol: float) -> dict:
        errors = []
        for d in d_values:
            if d and d > 0:
                cands = phase.match_d(d, tol)
                if cands:
                    errors.append(cands[0]["d_error"])
        n_total = sum(1 for d in d_values if d and d > 0)
        mean_err = float(np.mean(errors)) if errors else 0.0
        return {
            "name": phase.name,
            "n_matched": len(errors),
            "n_total": n_total,
            "mean_error": mean_err,
        }

    def _set_phase_match(self, report: dict, absences: str) -> None:
        pct = 100.0 * report["mean_error"]
        self.phase_match = (
            f"{report['name']} ({absences}): "
            f"{report['n_matched']}/{report['n_total']} matched, {pct:.1f}% mean error"
        )

    def index_rings(self, phase: Phase, tol: float = 0.03, replace: bool = True) -> Self:
        """Label rings by d-spacing match against a calibrated phase."""
        self._require_calibrated()
        rings = [dict(r) for r in self.rings]
        for r in rings:
            if not replace and r.get("hkl"):
                continue
            d = r.get("d_spacing")
            cands = phase.match_d(d, tol) if d else []
            r["hkl_candidates"] = [c["hkl_str"] for c in cands]
            r.update(index_assignment(cands[0] if cands else None))
        self.rings = rings
        self._set_phase_match(
            self._match_report(phase, [r.get("d_spacing") for r in rings], tol), phase.absences
        )
        return self

    def identify_phase(self, database, tol: float = 0.03) -> list[dict]:
        """Rank an explicit list of candidate phases against measured d-spacings.

        This is the primary verification workflow: build the candidates you
        expect (:func:`~quantem.widget.library_phase`,
        :meth:`Phase.from_cubic`, :meth:`Phase.from_dspacings`, ...) and rank
        only those. Use :meth:`search_phases` when you have no candidates in
        mind.
        """
        self._require_calibrated()
        phases = list(database)
        reports = self._rank_phases(self._observed_d(), phases, tol, max(len(phases), 1))
        self._set_identify_results(reports)
        return reports

    def search_phases(
        self,
        *,
        tol: float = 0.03,
        elements=None,
        exclude_elements=None,
        extra=None,
        custom_only: bool | None = None,
        top_n: int = 10,
    ) -> list[dict]:
        """Rank library, custom, and extra phases against measured d-spacings.

        With ``custom_only`` (default: the ``identify_custom_only`` trait) the
        library is skipped and only user candidates (custom phases plus
        ``extra``) are ranked.
        """
        self._require_calibrated()
        observed = self._observed_d()
        if custom_only is None:
            custom_only = self.identify_custom_only
        allowed = parse_elements(elements if elements is not None else self.identify_elements)
        excluded = parse_elements(exclude_elements)
        candidates = list(self._all_phases(custom_only=custom_only)) + list(extra or [])
        if custom_only and not candidates:
            raise ValueError("no candidate phases; add custom phases or pass extra")
        phases = []
        for phase in candidates:
            els = element_symbols(phase.name)
            if allowed is not None and els and not els <= allowed:
                continue
            if excluded and els & excluded:
                continue
            phases.append(phase)
        reports = self._rank_phases(observed, phases, tol, top_n)
        self._set_identify_results(reports)
        return reports

    def _observed_d(self) -> list[float]:
        source = self.rings if self.rings else self.spots
        observed = sorted(d for d in (x.get("d_spacing") for x in source) if d and d > 0)
        if not observed:
            raise ValueError("no measured d-spacings; add rings or spots first")
        return observed

    @staticmethod
    def _phase_lines(phase: Phase, d_min: float) -> list[dict]:
        return [
            {
                "d": reflection["d"],
                "hkl": reflection["hkl_str"],
                "i_rel": reflection["intensity"],
            }
            for reflection in phase.reflections(d_min=d_min)
        ]

    def _rank_phases(self, observed, phases, tol, top_n) -> list[dict]:
        reports = []
        d_min = min(observed) * 0.8
        for phase in phases:
            # skip degenerate candidates
            try:
                lines = self._phase_lines(phase, d_min=d_min)
            except ValueError:
                continue
            if len(lines) < 2:
                continue
            report = match_candidate(observed, lines, tol=tol)
            report.update(
                {
                    "phase_id": f"phase-{phase.name}",
                    "name": phase.name,
                    "lines": self._match_lines(observed, lines, report),
                }
            )
            report.pop("assignments", None)
            reports.append(report)
        if not reports:
            raise ValueError("no candidate phases pass the filters")
        reports.sort(key=match_sort_key)
        return reports[: int(top_n)]

    def _identify_summary(self, reports: list[dict]) -> str:
        top = reports[0]
        status = f"{top['name']}: {top['matched']}/{top['n_obs']} lines"
        if top["n_obs"] < 4:
            status += "; few measured lines"
        runners = ", ".join(
            report["name"]
            + (
                f" (also {report['matched']}/{report['n_obs']})"
                if report["matched"] == top["matched"]
                else ""
            )
            for report in reports[1:3]
        )
        return f"{status}; next: {runners}" if runners else status

    def _set_identify_results(self, reports: list[dict]) -> None:
        self._identify_results = reports[:10]
        self.phase_match = self._identify_summary(reports)

    @staticmethod
    def _match_lines(observed: list[float], lines: list[dict], report: dict) -> list[dict]:
        rows = []
        assignments = dict(report.get("assignments") or [])
        used_refs = set()
        for obs_index, measured_d in enumerate(observed):
            ref_index = assignments.get(obs_index)
            if ref_index is None:
                rows.append(
                    {
                        "obs_d": float(measured_d),
                        "ref_d": None,
                        "hkl": "",
                        "err": None,
                        "i_rel": None,
                    }
                )
            else:
                ref = lines[ref_index]
                used_refs.add(ref_index)
                rows.append(
                    {
                        "obs_d": float(measured_d),
                        "ref_d": float(ref["d"]),
                        "hkl": ref.get("hkl", ""),
                        "err": abs(ref["d"] - measured_d) / ref["d"],
                        "i_rel": ref.get("i_rel"),
                    }
                )
        lo, hi = min(observed), max(observed)
        missing = [
            (j, ref)
            for j, ref in enumerate(lines)
            if j not in used_refs and lo <= ref["d"] <= hi and (ref.get("i_rel") or 0) >= 25
        ]
        missing.sort(key=lambda x: -(x[1].get("i_rel") or 0))
        for _, ref in missing[:5]:
            rows.append(
                {
                    "obs_d": None,
                    "ref_d": float(ref["d"]),
                    "hkl": ref.get("hkl", ""),
                    "err": None,
                    "i_rel": ref.get("i_rel"),
                }
            )
        return rows

    def _spot_vector(self, spot: dict) -> tuple[float, float]:
        return corrected_vector(
            spot["row"] - self.center_row,
            spot["col"] - self.center_col,
            ellipse_ratio=self.ellipse_ratio,
            ellipse_angle=self.ellipse_angle,
            ellipse_corrected=self.ellipse_corrected,
        )

    def _measured_angle(self, s1: dict, s2: dict) -> float:
        dr1, dc1 = self._spot_vector(s1)
        dr2, dc2 = self._spot_vector(s2)
        r1, r2 = math.hypot(dr1, dc1), math.hypot(dr2, dc2)
        if r1 == 0 or r2 == 0:
            return 0.0
        cos_a = max(-1.0, min(1.0, (dr1 * dr2 + dc1 * dc2) / (r1 * r2)))
        return math.degrees(math.acos(cos_a))

    @staticmethod
    def _hkl_variants(phase: Phase, hkl: Sequence[int]) -> list[tuple[int, int, int]]:
        """All equal-d images of a reflection, canonical first.

        Signed permutations do not close hexagonal families (the 60-degree
        images of (110) are (-1,2,0)-type), so the whole index grid within a
        sum-of-|indices| bound is scanned for reflections at the same d.
        """
        d_ref = phase.d_spacing(hkl)
        bound = max(1, int(sum(abs(int(i)) for i in hkl)))
        axis = np.arange(-bound, bound + 1)
        grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
        inv_d_sq = np.einsum("ij,jk,ik->i", grid, phase._g_star, grid)
        ref = 1.0 / d_ref**2
        close = np.abs(inv_d_sq - ref) <= 2e-6 * ref  # rel_tol 1e-6 on d
        variants = {
            tuple(int(i) for i in row) for row in grid[close] if phase.is_allowed(row)
        }
        return sorted(variants, key=_label_preference)

    def _find_anchor_pair(
        self,
        phase: Phase,
        spots: list[dict],
        cand_lists: list[list[dict]],
        angle_tol: float,
    ) -> tuple[int, int, dict, dict] | None:
        """First non-collinear spot pair that matches phase angle geometry."""
        for i in range(len(spots)):
            for j in range(i + 1, len(spots)):
                if not cand_lists[i] or not cand_lists[j]:
                    continue
                measured = self._measured_angle(spots[i], spots[j])
                if measured < 1e-6:
                    continue
                best = None
                for ci in cand_lists[i]:
                    for cj in cand_lists[j]:
                        err = abs(phase.plane_angle(ci["hkl"], cj["hkl"]) - measured)
                        if err <= angle_tol and (best is None or err < best[0]):
                            best = (err, ci, cj)
                if best is not None:
                    return i, j, best[1], best[2]
        return None

    def index_spots(self, phase: Phase, tol: float = 0.03, angle_tol: float = 3.0) -> Self:
        """Index spots and solve the zone axis from an angle-consistent anchor pair."""
        self._require_calibrated()
        if phase.lattice is None:
            raise ValueError(
                "index_spots needs a lattice-based Phase (from_cubic / full constructor) "
                "for the inter-spot angle check; a d-spacing card has no angles"
            )
        spots = [dict(s) for s in self.spots]
        cand_lists = []
        for s in spots:
            d = s.get("d_spacing")
            cands = phase.match_d(d, tol) if d else []
            s["hkl_candidates"] = [c["hkl_str"] for c in cands]
            cand_lists.append(cands)

        anchors = self._find_anchor_pair(phase, spots, cand_lists, angle_tol)
        if anchors is None:
            for s, cands in zip(spots, cand_lists):
                s.update(index_assignment(cands[0] if cands else None))
            self.spots = self._with_angles(spots)
            self.zone_axis = ""
            return self

        i, j, ci, cj = anchors
        # zone axis from the second-reflection variant matching the measured angle
        measured_ij = self._measured_angle(spots[i], spots[j])
        cj_variant = min(
            self._hkl_variants(phase, cj["hkl"]),
            key=lambda v: round(abs(phase.plane_angle(ci["hkl"], v) - measured_ij), 6),
        )
        h1, k1, l1 = (int(x) for x in ci["hkl"])
        h2, k2, l2 = cj_variant
        axis = (k1 * l2 - l1 * k2, l1 * h2 - h1 * l2, h1 * k2 - k1 * h2)

        def in_zone(hkl):
            return hkl[0] * axis[0] + hkl[1] * axis[1] + hkl[2] * axis[2] == 0

        # remaining spots: family variant satisfying the zone law and angle
        anchor_choice = {i: (ci, tuple(int(x) for x in ci["hkl"])), j: (cj, cj_variant)}
        for idx, (s, cands) in enumerate(zip(spots, cand_lists)):
            if idx in anchor_choice:
                chosen, variant = anchor_choice[idx]
            else:
                measured = self._measured_angle(spots[i], s)
                chosen, variant, best_err = None, None, None
                for c in cands:
                    for v in self._hkl_variants(phase, c["hkl"]):
                        if not in_zone(v):
                            continue
                        err = round(abs(phase.plane_angle(ci["hkl"], v) - measured), 6)
                        if err <= angle_tol and (best_err is None or err < best_err):
                            chosen, variant, best_err = c, v, err
            if chosen is not None and not in_zone(tuple(int(x) for x in chosen["hkl"])):
                chosen = {**chosen, "hkl_str": _format_hkl(variant)}
            s.update(index_assignment(chosen))

        self.spots = self._with_angles(spots)
        self.zone_axis = format_zone_axis(ci["hkl"], cj_variant)
        self._set_phase_match(
            self._match_report(phase, [s.get("d_spacing") for s in spots], tol), phase.absences
        )
        return self

    def _apply_calibration(self, d_known: float, r_pixels: float, source: str) -> Self:
        self.k_pixel_size = 1.0 / (d_known * r_pixels)
        self.k_calibrated = True
        self.calibration_source = source
        self.calibration_ref_d = float(d_known)
        self.calibration_ref_radius = float(r_pixels)
        return self

    def calibrate_from_spot(self, row: float, col: float, d_known: float) -> Self:
        """Calibrate ``k_pixel_size`` from a spot of known d-spacing."""
        if d_known <= 0:
            raise ValueError(f"d_known must be positive, got {d_known}")
        r_pixels = float(
            corrected_radius(
                row - self.center_row,
                col - self.center_col,
                ellipse_ratio=self.ellipse_ratio,
                ellipse_angle=self.ellipse_angle,
                ellipse_corrected=self.ellipse_corrected,
            )
        )
        if r_pixels <= 0:
            raise ValueError("calibration point is at the center; no g-vector")
        return self._apply_calibration(d_known, r_pixels, "from_spot")

    def calibrate_from_ring(self, radius_px: float, d_known: float) -> Self:
        """Calibrate ``k_pixel_size`` from a ring of known d-spacing."""
        if d_known <= 0:
            raise ValueError(f"d_known must be positive, got {d_known}")
        if radius_px <= 0:
            raise ValueError(f"radius_px must be positive, got {radius_px}")
        return self._apply_calibration(d_known, radius_px, "from_ring")

    def calibrate_from_phase(self, phase: Phase, *, tol: float = 0.03, d_min: float = 0.5) -> Self:
        """Fit ``k_pixel_size`` by assigning ring-radius ratios to a known phase."""
        if len(self.rings) < 2:
            raise ValueError(
                "calibrate_from_phase needs >= 2 rings; use calibrate_from_ring for a single ring"
            )
        refl = phase.reflections(d_min=d_min)
        if not refl:
            raise ValueError(f"{phase.name} has no reflections above d_min={d_min}")
        radii = [float(r["radius_px"]) for r in self.rings]
        inv_d = [1.0 / rf["d"] for rf in refl]

        r_inner = min(radii)
        best = None
        for x0 in inv_d:
            scale = r_inner / x0
            assigned, errs = [], []
            for r in radii:
                x_pred = r / scale
                if x_pred <= 0:
                    assigned.append(None)
                    errs.append(None)
                    continue
                nearest = min(range(len(inv_d)), key=lambda i: abs(inv_d[i] - x_pred))
                err = abs(inv_d[nearest] - x_pred) / abs(x_pred)
                assigned.append(nearest if err <= tol else None)
                errs.append(err if err <= tol else None)
            used = [a for a in assigned if a is not None]
            n_ok = len(used)
            if n_ok < 2:
                continue
            in_tol = [e for e in errs if e is not None]
            mean_err = float(np.mean(in_tol))
            # only machine-precision assignments outrank the low-order preference
            worst_px = max(e * r for e, r in zip(errs, radii) if e is not None)
            exact = 1 if worst_px <= 1e-6 else 0
            candidate_key = (n_ok, exact, -sum(inv_d[a] for a in used), -mean_err)
            if best is None or candidate_key > best[0]:
                best = (candidate_key, assigned)
        if best is None:
            raise ValueError(
                f"could not assign >= 2 rings to {phase.name} reflections within tol={tol}; "
                "check the phase or calibrate_from_ring manually"
            )
        _, assigned = best

        pairs = [
            (radius_px, inv_d[reflection_index])
            for radius_px, reflection_index in zip(radii, assigned)
            if reflection_index is not None
        ]
        scale = sum(radius_px * q for radius_px, q in pairs) / sum(q * q for _, q in pairs)
        self.k_pixel_size = 1.0 / scale
        self.k_calibrated = True
        self.calibration_source = "from_phase"
        self.calibration_ref_d = 0.0
        self.calibration_ref_radius = 0.0

        resids = []
        rings = [dict(r) for r in self.rings]
        for ring, radius_px, reflection_index in zip(rings, radii, assigned):
            if reflection_index is None:
                ring["hkl_candidates"] = []
                ring.update(index_assignment(None))
                ring["radius_resid_px"] = None
                continue
            reflection = refl[reflection_index]
            measured_d = 1.0 / (radius_px * self.k_pixel_size)
            assignment = {
                "hkl_str": reflection["hkl_str"],
                "d": reflection["d"],
                "d_error": abs(measured_d - reflection["d"]) / reflection["d"],
            }
            ring["hkl_candidates"] = [reflection["hkl_str"]]
            ring.update(index_assignment(assignment))
            residual_px = radius_px - scale * inv_d[reflection_index]
            ring["radius_resid_px"] = residual_px
            resids.append(residual_px)
        self.rings = rings
        self.calibration_rms_px = float(np.sqrt(np.mean(np.square(resids))))
        return self

    def _on_calibrate_from_ring_request(self, change=None):
        val = self._calibrate_from_ring_request
        if val and len(val) == 2:
            try:
                self.calibrate_from_ring(val[0], val[1])
            except ValueError as exc:
                self.analysis_status = f"Calibrate failed: {exc}"
            self._calibrate_from_ring_request = []

    def _on_calibrate_from_spot_request(self, change=None):
        val = self._calibrate_from_spot_request
        if val and len(val) == 3:
            try:
                self.calibrate_from_spot(val[0], val[1], val[2])
            except ValueError as exc:
                self.analysis_status = f"Calibrate failed: {exc}"
            self._calibrate_from_spot_request = []

    def export_measurements(self, path: str) -> pathlib.Path:
        """Export spot and ring measurements as CSV or JSON."""
        return write_measurement_file(
            path,
            build_measurement_records(self.spots, self.rings),
            measurement_metadata(self.state_dict()),
        )

    @classmethod
    def measurements_from_state(cls, state, path=None):
        """Rebuild the measurement table from a saved state."""
        state = cls._resolve_state(state)
        records = build_measurement_records(state.get("spots", []), state.get("rings", []))
        if path is None:
            return records
        return write_measurement_file(path, records, measurement_metadata(state))

    def export_html(
        self,
        path: str | pathlib.Path | None = None,
        *,
        title: str | None = None,
        **options,
    ) -> pathlib.Path:
        """Write a standalone HTML viewer with exact float32 frames."""
        if not hasattr(self, "_data") or self._data is None:
            raise ValueError("Cannot export HTML after free(); rebuild the widget first.")
        export_path = pathlib.Path(path) if path is not None else self._default_html_export_path()
        self._write_html_export(export_path, title=title)
        ensure_mobile_viewport(export_path)
        size_mb = export_path.stat().st_size / (1024 * 1024)
        self.export_status = f"Exported {export_path.name} ({size_mb:.1f} MB, full float32)"
        return export_path

    def _on_export_request_change(self, change: dict) -> None:
        raw = str(change.get("new") or "")
        if not raw:
            return
        try:
            payload = json.loads(raw)
            mode = str(payload.get("mode", "single"))
            if mode == "clear":
                self.export_payload = b""
                self.export_payload_id = ""
                self.export_filename = ""
                return
            if payload.get("download"):
                filename = str(payload.get("filename") or self._default_html_export_path().name)
                request_id = str(payload.get("id") or "")
                self.export_status = f"Preparing {filename}..."
                html = self._html_export_bytes()
                self.export_filename = filename
                self.export_payload = html
                self.export_payload_id = request_id
                size_mb = len(html) / (1024 * 1024)
                self.export_status = f"Ready {filename} ({size_mb:.1f} MB, full float32)"
            else:
                self.export_status = "Exporting HTML..."
                self.export_html()
        except Exception as exc:
            self.export_status = f"Export failed: {exc}"

    def _default_html_export_path(self) -> pathlib.Path:
        label = self.title.strip() or "showdiffraction"
        slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        if not slug:
            slug = "showdiffraction"
        shape = f"{self.n_frames}x{self.det_rows}x{self.det_cols}"
        return pathlib.Path.cwd() / f"{slug}_{shape}.html"

    def _write_html_export(
        self,
        path: str | pathlib.Path,
        *,
        title: str | None = None,
    ) -> pathlib.Path:
        from ipywidgets.embed import dependency_state, embed_minimal_html

        export_path = pathlib.Path(path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        page_title = title or self.title or "ShowDiffraction"
        export_widget = self._clone_for_html_export()
        try:
            state = dependency_state([export_widget], drop_defaults=False)
            embed_minimal_html(
                str(export_path),
                views=[export_widget],
                title=page_title,
                drop_defaults=False,
                state=state,
            )
        finally:
            export_widget.close()
        return export_path

    def _html_export_bytes(self) -> bytes:
        with tempfile.TemporaryDirectory(prefix="showdiffraction-export-") as tmp:
            path = pathlib.Path(tmp) / self._default_html_export_path().name
            self._write_html_export(path)
            ensure_mobile_viewport(path)
            return path.read_bytes()

    def _clone_for_html_export(self) -> Self:
        if not hasattr(self, "_data") or self._data is None:
            raise ValueError("Cannot export HTML after free(); rebuild the widget first.")
        clone = type(self)(to_numpy(self._data), state=self.state_dict(), verbose=False)
        # derived panels are not state fields
        clone._identify_results = list(self._identify_results)
        clone._quality = dict(self._quality)
        clone.offline = True
        clone.export_enabled = False
        clone.export_status = ""
        clone.export_payload = b""
        clone.export_payload_id = ""
        clone.export_filename = ""
        clone._update_frame()
        return clone

    def set_image(self, data) -> Self:
        """Replace data. Preserves display settings, clears spots and rings."""
        data, title, pixel_size, k_pixel_size, metadata_calibrated = normalize_data_input(
            data,
            title=self.title,
            replace_title=True,
        )
        self.title = title
        if pixel_size is not None:
            self.pixel_size = float(pixel_size)
        if k_pixel_size is not None and k_pixel_size > 0:
            self.k_pixel_size = float(k_pixel_size)
            self.k_calibrated = True
            if metadata_calibrated:
                self.calibration_source = "metadata"
        self._ingest_data(data)
        self.frame_idx = min(self.frame_idx, self.n_frames - 1)
        self.spots = []
        self.rings = []
        self.auto_detect_center()
        self._update_frame()
        self._bake_offline_frames()
        # refresh panes explicitly
        self._update_profile()
        self._update_azimuthal()
        return self

    def state_dict(self):
        """Return the persistable widget state as a plain dict."""
        state = {}
        for field in self._STATE_FIELDS:
            value = getattr(self, field)
            state[field] = list(value) if field in self._LIST_STATE_FIELDS else value
        return state

    def save(self, path: str):
        """Write the widget state to a JSON file."""
        save_state_file(path, "ShowDiffraction", self.state_dict())

    def collapse_controls(self) -> Self:
        """Collapse controls behind the frontend ``Controls`` button."""
        self.controls_collapsed = True
        return self

    def expand_controls(self) -> Self:
        """Expand frontend controls when ``show_controls`` is enabled."""
        self.controls_collapsed = False
        return self

    def toggle_controls(self) -> Self:
        """Toggle whether frontend controls start collapsed."""
        self.controls_collapsed = not bool(self.controls_collapsed)
        return self

    def load_state_dict(self, state):
        """Restore widget state from a dict; unknown keys are ignored."""
        deferred = {}
        for key, val in state.items():
            if key not in self._STATE_FIELDS:
                continue
            # measurement records last, so geometry restores do not resample them
            if key in ("spots", "rings"):
                deferred[key] = val
                continue
            if key == "frame_idx":
                requested = int(val)
                self.frame_idx = requested
                if self.frame_idx != requested:
                    self.analysis_status = (
                        f"Saved frame_idx {requested} clamped to {self.frame_idx}: "
                        f"data has {self.n_frames} frames"
                    )
                continue
            setattr(self, key, val)
        for key, val in deferred.items():
            setattr(self, key, val)

    def summary(self):
        """Print a text summary of calibration, spots, rings, and indexing."""
        lines = [self.title or "ShowDiffraction"]
        lines.append(f"Frames:   {self.n_frames} (showing #{self.frame_idx})")
        k_info = f"{self.k_pixel_size:.4f} 1/Å/px" if self.k_calibrated else "uncalibrated"
        lines.append(f"Detector: {self.det_rows}x{self.det_cols} ({k_info})")
        if self.k_calibrated:
            source = {
                "from_phase": "phase",
                "from_ring": "ring",
                "from_spot": "spot",
            }.get(self.calibration_source, self.calibration_source)
            cal = f"Calibration: {source}"
            if self.calibration_ref_d > 0:
                cal += (
                    f" (d={self.calibration_ref_d:.3f} Å at r={self.calibration_ref_radius:.1f} px)"
                )
            elif self.calibration_source == "from_phase":
                cal += f" (rms {self.calibration_rms_px:.2f} px)"
            lines.append(cal)
        if self.ellipse_ratio != 1.0:
            state = "corrected" if self.ellipse_corrected else "not corrected"
            lines.append(
                f"Ellipse:  a/b={self.ellipse_ratio:.3f} at {self.ellipse_angle:.1f}° ({state})"
            )
        lines.append(
            f"Center:   ({self.center_row:.1f}, {self.center_col:.1f})  "
            f"BF r={self.bf_radius:.1f} px"
        )
        lines.append(f"Spots:    {len(self.spots)}")
        if self.spots:
            for s in self.spots[:5]:
                if s.get("d_spacing"):
                    derr = s.get("d_spacing_err")
                    d = f"{s['d_spacing']:.3f}±{derr:.3f} Å" if derr else f"{s['d_spacing']:.3f} Å"
                    label = f"d={d}"
                else:
                    label = f"r={s['r_pixels']:.1f} px"
                ang = f"  angle={s['angle_deg']:.1f}°" if s.get("angle_deg") is not None else ""
                hkl = f"  {s['hkl']}" if s.get("hkl") else ""
                lines.append(f"  #{s['id']} ({s['row']:.1f}, {s['col']:.1f}) {label}{ang}{hkl}")
            if len(self.spots) > 5:
                lines.append(f"  ... +{len(self.spots) - 5} more")
        lines.append(f"Rings:    {len(self.rings)}")
        if self.zone_axis:
            lines.append(f"Zone:     {self.zone_axis}")
        if self.phase_match:
            lines.append(f"Phase:    {self.phase_match}")
        lines.append(f"Display:  {self.dp_colormap} | {self.dp_scale_mode}")
        if self.snap_enabled:
            lines.append(f"Snap:     radius={self.snap_radius}")
        print("\n".join(lines))

    def __repr__(self) -> str:
        k_unit = "1/Å" if self.k_calibrated else "px"
        shape = f"({self.n_frames}, {self.det_rows}, {self.det_cols})"
        title_info = f", title='{self.title}'" if self.title else ""
        spots_info = f", spots={len(self.spots)}" if self.spots else ""
        return (
            f"ShowDiffraction(shape={shape}, "
            f"sampling=({self.pixel_size} Å, {self.k_pixel_size} {k_unit}), "
            f"frame={self.frame_idx}/{self.n_frames}{spots_info}{title_info})"
        )

    def free(self):
        """Free GPU memory held by this widget."""
        if hasattr(self, "_data"):
            del self._data
        import gc

        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
