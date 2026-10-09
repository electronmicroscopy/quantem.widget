"""Crystal phases for diffraction indexing: lattice geometry, systematic absences and the
built-in standards library.

A :class:`Phase` is either a lattice (cell edges in Å, angles in degrees, an absence rule)
from which allowed reflections and d-spacings are enumerated, or a reference line table
(``Phase.from_dspacings``) for pure d-spacing matching. :func:`library_phase` builds one
from :data:`PHASE_LIBRARY`.
"""

import math
from collections.abc import Iterable, Sequence
from typing import Self

import numpy as np


# --- Systematic absence rules: (h, k, l) -> allowed ---
# Each rule drops the reflections whose structure factor vanishes for that lattice and site
# set, so indexing never proposes a forbidden hkl and the d-spacing table lists only lines
# the pattern can show.
def allow_all(*_) -> bool:
    """Primitive cell without glide or screw extinctions: every reflection is allowed."""
    return True


def allow_fcc(h: int, k: int, ell: int) -> bool:
    """F-centred lattice: h, k, l all even or all odd."""
    return (h % 2) == (k % 2) == (ell % 2)


def allow_bcc(h: int, k: int, ell: int) -> bool:
    """I-centred lattice: h + k + l even."""
    return (h + k + ell) % 2 == 0


def allow_diamond(h: int, k: int, ell: int) -> bool:
    """Diamond (Fd-3m, atoms on 8a): h, k, l all odd, or all even with h + k + l = 4n."""
    return (h % 2 == k % 2 == ell % 2 == 1) or (
        h % 2 == k % 2 == ell % 2 == 0 and (h + k + ell) % 4 == 0
    )


def allow_hcp(h: int, k: int, ell: int) -> bool:
    """Hexagonal close packing (P6_3/mmc, atoms on 2c): absent when h + 2k = 3n with l odd."""
    return not ((h + 2 * k) % 3 == 0 and ell % 2 == 1)


def allow_rhombohedral(h: int, k: int, ell: int) -> bool:
    """R-centred lattice on hexagonal axes (obverse setting): -h + k + l = 3n."""
    return (-h + k + ell) % 3 == 0


def allow_rhombohedral_c(h: int, k: int, ell: int) -> bool:
    """R-3c / R3c: R-centring plus the c-glide, which removes h0l, 0kl and (h, -h, l)
    reflections with l odd."""
    if (h == 0 or k == 0 or h == -k) and ell % 2 == 1:
        return False
    return allow_rhombohedral(h, k, ell)


def allow_spinel(h: int, k: int, ell: int) -> bool:
    """Spinel (Fd-3m): F-centring plus the d-glide, h00 with h = 4n and 0kl with k + l = 4n
    (in any index order, on absolute values)."""
    if not allow_fcc(h, k, ell):
        return False
    low, mid, high = sorted((abs(h), abs(k), abs(ell)))
    if mid == 0:  # h00
        return high % 4 == 0
    if low == 0:  # hk0
        return (mid + high) % 4 == 0
    return True


def allow_i41amd(h: int, k: int, ell: int) -> bool:
    """I4_1/amd with atoms on 4a/8e (anatase, beta-Sn): h + k + l even; for even l,
    l = 4n needs h even and l = 4n + 2 needs h odd."""
    if (h + k + ell) % 2 != 0:
        return False
    if ell % 2 == 1:
        return True
    if ell % 4 == 0:  # includes hk0
        return h % 2 == 0
    return h % 2 == 1  # l = 4n+2


def allow_rutile(h: int, k: int, ell: int) -> bool:
    """Rutile (P4_2/mnm): the n-glide removes 0kl with k + l odd and, by the 4-fold axis,
    h0l with h + l odd."""
    if h == 0 and (k + ell) % 2 == 1:
        return False
    if k == 0 and (h + ell) % 2 == 1:
        return False
    return True


def allow_bixbyite(h: int, k: int, ell: int) -> bool:
    """Bixbyite (Ia-3): h + k + l even, and the cyclic a-glide: 0kl needs k and l even,
    h0l needs h and l even, hk0 needs h and k even."""
    if (h + k + ell) % 2 != 0:
        return False
    if h == 0 and (k % 2 == 1 or ell % 2 == 1):
        return False
    if k == 0 and (h % 2 == 1 or ell % 2 == 1):
        return False
    if ell == 0 and (h % 2 == 1 or k % 2 == 1):
        return False
    return True


def allow_cuprite(h: int, k: int, ell: int) -> bool:
    """Cuprite (Pn-3m, O on 2a, Cu on 4b): h + k + l even, or h, k, l all odd; mixed parity
    with an odd sum is absent."""
    if (h + k + ell) % 2 == 0:
        return True
    return h % 2 == k % 2 == ell % 2


ABSENCE_RULES = {
    "none": allow_all,
    "fcc": allow_fcc,
    "bcc": allow_bcc,
    "diamond": allow_diamond,
    "hcp": allow_hcp,
    "wurtzite": allow_hcp,  # 2b wurtzite sites zero the same reflections as hcp
    "rhombohedral": allow_rhombohedral,
    "rhombohedral-c": allow_rhombohedral_c,
    "spinel": allow_spinel,  # Fd-3m d-glide: h00 needs h=4n, hk0 needs h+k=4n
    "i41amd": allow_i41amd,  # I4_1/amd 4a/8e sites: 002, 110, 222 absent
    "rutile": allow_rutile,  # P4_2/mnm n-glide: 0kl needs k+l even
    "bixbyite": allow_bixbyite,  # Ia-3 a-glide: 0kl needs k, l even
    "cuprite": allow_cuprite,  # Pn-3m 2a/4b sites: mixed parity with odd sum absent
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


def format_hkl(hkl: Sequence[float]) -> str:
    """``311`` for single-digit non-negative indices, ``(h,k,l)`` otherwise."""
    indices = tuple(int(i) for i in hkl)
    if all(0 <= i < 10 for i in indices):
        return "".join(str(i) for i in indices)
    return "(" + ",".join(str(i) for i in indices) + ")"


def label_preference(hkl: tuple[int, int, int]) -> tuple[int, tuple[int, int, int]]:
    """Sort key preferring the conventional family label: fewest negative
    indices, then lexicographically largest (h before k before l)."""
    return (sum(1 for i in hkl if i < 0), tuple(-i for i in hkl))


def format_zone_axis(hkl1: tuple[int, int, int], hkl2: tuple[int, int, int]) -> str:
    """Zone-axis label ``[uvw]`` from two indexed reflections: the cross product of the
    plane normals, reduced by its gcd, with the first non-zero component positive."""
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


class Phase:
    """A crystalline phase: lattice parameters (Å, degrees) + absence rule for
    geometry-aware indexing, or a reference d-spacing card for pure matching.

    Parameters
    ----------
    name : str
        Label shown in phase tables and status lines.
    a, b, c : float
        Cell edges in Å.
    alpha, beta, gamma : float, default 90
        Cell angles in degrees.
    absences : str, default "none"
        Systematic-absence rule, a key of :data:`ABSENCE_RULES`.
    reference_lines : list of (d, hkl label, intensity or None), optional
        A d-spacing card instead of a lattice, as built by :meth:`from_dspacings`; the
        cell edges are then ignored.
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
        reference_lines: list[tuple[float, str, float | None]] | None = None,
    ) -> None:
        self.name = name
        self.absences = absences
        if absences not in ABSENCE_RULES:
            raise ValueError(f"unknown absence rule {absences!r}; use {list(ABSENCE_RULES)}")
        self.allowed_rule = ABSENCE_RULES[absences]
        self.reference_lines = reference_lines
        self.reflection_cache: list[dict] | None = None
        if reference_lines is not None:
            self.lattice = None
            self.g_star = None
            return
        if min(a, b, c) <= 0:
            raise ValueError("lattice edge lengths must be positive")
        if not all(0.0 < angle < 180.0 for angle in (alpha, beta, gamma)):
            raise ValueError("cell angles must be strictly between 0 and 180 degrees")
        self.lattice = (float(a), float(b), float(c), float(alpha), float(beta), float(gamma))
        ca, cb, cg = (math.cos(math.radians(angle)) for angle in (alpha, beta, gamma))
        # metric tensor; its inverse gives 1/d^2 = h^T G* h
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
        self.g_star = np.linalg.inv(g)

    @classmethod
    def from_cubic(cls, name: str, a: float, absences: str = "fcc") -> Self:
        """Cubic phase with edge ``a`` (Å) and a systematic-absence rule."""
        return cls(name, a, a, a, 90.0, 90.0, 90.0, absences=absences)

    @classmethod
    def from_dspacings(cls, name: str, entries: Iterable[Sequence]) -> Self:
        """Phase from reference entries: ``(d_Å, hkl_label[, intensity])``."""
        reference_lines = [
            (float(entry[0]), str(entry[1]), float(entry[2]) if len(entry) > 2 else None)
            for entry in entries
        ]
        return cls(name, 1.0, 1.0, 1.0, reference_lines=reference_lines)

    def d_spacing(self, hkl: Sequence[float]) -> float:
        """Interplanar spacing d_hkl in Å."""
        if self.g_star is None:
            raise ValueError("d_spacing requires a lattice-based Phase, not a d-spacing table")
        indices = np.asarray(hkl, dtype=np.float64)
        inverse_d_squared = float(indices @ self.g_star @ indices)
        if inverse_d_squared <= 0:
            raise ValueError("invalid reflection (000)")
        return 1.0 / math.sqrt(inverse_d_squared)

    def plane_angle(self, hkl1: Sequence[float], hkl2: Sequence[float]) -> float:
        """Angle in degrees between plane normals (hkl1) and (hkl2)."""
        if self.g_star is None:
            raise ValueError("plane_angle requires a lattice-based Phase, not a d-spacing table")
        indices1 = np.asarray(hkl1, dtype=np.float64)
        indices2 = np.asarray(hkl2, dtype=np.float64)
        numerator = float(indices1 @ self.g_star @ indices2)
        denominator = math.sqrt(
            float(indices1 @ self.g_star @ indices1) * float(indices2 @ self.g_star @ indices2)
        )
        if denominator == 0:
            return 0.0
        return math.degrees(math.acos(max(-1.0, min(1.0, numerator / denominator))))

    def is_allowed(self, hkl: Sequence[float]) -> bool:
        """Whether (hkl) is a non-origin reflection permitted by the absence rule."""
        h, k, ell = (int(i) for i in hkl)
        if h == 0 and k == 0 and ell == 0:
            return False
        return bool(self.allowed_rule(h, k, ell))

    def reflections(self, d_min: float = 0.5, max_index: int | None = None) -> list[dict]:
        """Allowed reflection families, largest d first.

        By default ``max_index`` is sized so every family above ``d_min`` is
        enumerated; ``d_min`` is floored at 0.2 Å and ``max_index`` capped at
        25 to keep the enumeration bounded.
        """
        if self.reference_lines is not None:
            reflections = []
            for spacing, label, intensity in self.reference_lines:
                if spacing < d_min:
                    continue
                body = label.strip().strip("()")
                # a label that is not three Miller indices still names the line, without an hkl
                try:
                    indices = [int(part) for part in body.split(",")] if "," in body else [int(digit) for digit in body]
                except ValueError:
                    indices = []
                reflections.append(
                    {
                        "hkl": tuple(indices) if len(indices) == 3 else None,
                        "hkl_str": label,
                        "d": spacing,
                        "multiplicity": None,
                        "intensity": intensity,
                    }
                )
            return sorted(reflections, key=lambda reflection: -reflection["d"])

        d_min = max(float(d_min), 0.2)
        if max_index is None:
            max_index = math.ceil(max(self.lattice[:3]) / d_min)
        max_index = min(int(max_index), 25)
        a, b, c = self.lattice[:3]
        orthogonal = all(math.isclose(angle, 90.0) for angle in self.lattice[3:])
        cubic = orthogonal and math.isclose(a, b) and math.isclose(b, c)
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
                    d_key = round(spacing * 1e4)
                    # one label per symmetry-equivalent family: sorted |hkl| for cubic,
                    # |h|,|k|,|l| (h,k ordered when a = b) for orthogonal cells, first
                    # non-zero index positive otherwise
                    if cubic:
                        representative = tuple(sorted((abs(i) for i in hkl), reverse=True))
                    elif orthogonal:
                        ah, ak, al = (abs(i) for i in hkl)
                        if math.isclose(a, b):
                            ah, ak = sorted((ah, ak), reverse=True)
                        representative = (ah, ak, al)
                    else:
                        representative = hkl
                        for value in hkl:
                            if value < 0:
                                representative = tuple(-i for i in hkl)
                                break
                            if value > 0:
                                break
                    family = families_by_d.setdefault(d_key, {"d": spacing, "labels": {}})
                    labels = family["labels"]
                    labels[representative] = labels.get(representative, 0) + 1
        reflections = []
        for family in families_by_d.values():
            labels = family["labels"]
            if orthogonal:
                ordered = sorted(
                    labels.items(), key=lambda entry: (-entry[1], label_preference(entry[0]))
                )
                representative = ordered[0][0]
                hkl_str = "/".join(format_hkl(label) for label, _ in ordered)
            else:
                representative = min(labels, key=label_preference)
                hkl_str = format_hkl(representative)
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

    def match_d(self, d: float, tol: float = 0.03) -> list[dict]:
        """Reflections within fractional ``tol`` of ``d``, closest first.

        Errors are relative to the reference d, matching
        :func:`~quantem.widget.showdiffraction.lattice.match_candidate`.
        """
        if d <= 0:
            return []
        if self.reflection_cache is None:
            self.reflection_cache = self.reflections()
        matches = []
        for reflection in self.reflection_cache:
            error = abs(reflection["d"] - d) / reflection["d"]
            if error <= tol:
                matches.append({**reflection, "d_error": error})
        return sorted(matches, key=lambda reflection: reflection["d_error"])


def library_phase(name: str) -> Phase:
    """Build a :class:`Phase` from the built-in standards library."""
    if name not in PHASE_LIBRARY:
        raise ValueError(f"unknown library phase {name!r}; available: {sorted(PHASE_LIBRARY)}")
    entry = PHASE_LIBRARY[name]
    a, absences = entry["a"], entry["absences"]
    if "c" in entry:
        return Phase(name, a, a, entry["c"], 90.0, 90.0, entry["gamma"], absences=absences)
    return Phase.from_cubic(name, a, absences=absences)


def phase_from_entry(entry: dict) -> Phase:
    """Build a :class:`Phase` from a custom-phase dict (``name``, ``a`` and optional
    ``b c alpha beta gamma absences``), the record the widget's Phase menu edits."""
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
