import math

import numpy as np
import pytest

from quantem.widget.crystal import Phase

# Gold: FCC, a = 4.078 Å. d_hkl and plane angles are analytic for a cubic cell.
AU_A = 4.078


def test_phase_cubic_dspacing():
    au = Phase.from_cubic("Au", AU_A, absences="fcc")
    assert au.d_spacing((1, 1, 1)) == pytest.approx(AU_A / np.sqrt(3), abs=1e-4)
    assert au.d_spacing((2, 0, 0)) == pytest.approx(AU_A / 2, abs=1e-4)
    assert au.d_spacing((2, 2, 0)) == pytest.approx(AU_A / np.sqrt(8), abs=1e-4)
    assert au.d_spacing((3, 1, 1)) == pytest.approx(AU_A / np.sqrt(11), abs=1e-4)


def test_phase_plane_angle_cubic():
    au = Phase.from_cubic("Au", AU_A, absences="fcc")
    # (111)^(200) = 54.7356°, (200)^(020) = 90°
    assert au.plane_angle((1, 1, 1), (2, 0, 0)) == pytest.approx(54.7356, abs=1e-2)
    assert au.plane_angle((2, 0, 0), (0, 2, 0)) == pytest.approx(90.0, abs=1e-4)


def test_fcc_systematic_absences():
    au = Phase.from_cubic("Au", AU_A, absences="fcc")
    # all-even or all-odd allowed; mixed parity forbidden
    assert au.is_allowed((1, 1, 1)) and au.is_allowed((2, 0, 0)) and au.is_allowed((2, 2, 0))
    assert not au.is_allowed((1, 0, 0)) and not au.is_allowed((1, 1, 0))
    assert not au.is_allowed((2, 1, 0))


def test_bcc_systematic_absences():
    fe = Phase.from_cubic("Fe", 2.866, absences="bcc")
    # h+k+l even allowed
    assert fe.is_allowed((1, 1, 0)) and fe.is_allowed((2, 0, 0)) and fe.is_allowed((2, 1, 1))
    assert not fe.is_allowed((1, 0, 0)) and not fe.is_allowed((1, 1, 1))


def test_reflections_sorted_and_filtered():
    au = Phase.from_cubic("Au", AU_A, absences="fcc")
    refls = au.reflections(d_min=1.0, max_index=4)
    ds = [r["d"] for r in refls]
    assert ds == sorted(ds, reverse=True)  # largest d first
    assert all(d >= 1.0 for d in ds)
    # innermost FCC family is {111}
    assert refls[0]["hkl_str"] == "111"
    assert refls[0]["d"] == pytest.approx(AU_A / np.sqrt(3), abs=1e-4)
    # no forbidden families present
    assert all(au.is_allowed(r["hkl"]) for r in refls)


def test_match_d_returns_ranked_candidates():
    au = Phase.from_cubic("Au", AU_A, absences="fcc")
    d111 = AU_A / np.sqrt(3)
    cands = au.match_d(d111 * 1.005, tol=0.02)  # measured 0.5% high
    assert cands, "expected at least one candidate within tolerance"
    assert cands[0]["hkl_str"] == "111"
    assert cands[0]["d_error"] == pytest.approx(0.005, abs=2e-3)  # fractional
    # a d-spacing far from any reflection returns nothing within a tight tol
    assert au.match_d(3.5, tol=0.01) == []


def test_reflections_default_covers_high_index_families():
    fe3o4 = Phase.from_cubic("Fe3O4", 8.3963, absences="fcc")
    d800 = 8.3963 / np.sqrt(64)
    by_hkl = {r["hkl_str"]: r for r in fe3o4.reflections(d_min=0.9)}
    assert "800" in by_hkl
    assert by_hkl["800"]["d"] == pytest.approx(d800, abs=1e-4)
    assert "840" in by_hkl
    cands = fe3o4.match_d(d800, tol=0.01)
    assert cands and cands[0]["hkl_str"] == "800"


def test_from_dspacings_reference_table():
    ref = Phase.from_dspacings("Sample", [(2.355, "111"), (2.039, "200"), (1.442, "220")])
    refls = ref.reflections()
    assert [r["hkl_str"] for r in refls] == ["111", "200", "220"]
    cands = ref.match_d(2.04, tol=0.02)
    assert cands[0]["hkl_str"] == "200"


def test_from_dspacings_carries_intensity():
    ref = Phase.from_dspacings("X", [(2.53, "311", 100), (1.48, "440", 40), (2.10, "400")])
    by_hkl = {r["hkl_str"]: r for r in ref.reflections()}
    assert by_hkl["311"]["intensity"] == 100
    assert by_hkl["400"]["intensity"] is None
    assert ref.match_d(2.53, tol=0.02)[0]["intensity"] == 100


def test_invalid_lattice_raises():
    with pytest.raises(ValueError):
        Phase.from_cubic("bad", 0.0)
    with pytest.raises(ValueError):
        Phase.from_cubic("bad", -1.0)


def test_diamond_absence_rule():
    """Diamond structure: all-odd allowed; all-even needs h+k+l divisible by 4."""
    si = Phase.from_cubic("Si", 5.4310, absences="diamond")
    assert si.is_allowed((1, 1, 1))
    assert si.is_allowed((2, 2, 0))
    assert si.is_allowed((3, 1, 1))
    assert si.is_allowed((4, 0, 0))
    assert not si.is_allowed((2, 0, 0))
    assert not si.is_allowed((2, 2, 2))
    assert not si.is_allowed((4, 2, 0))


def test_phase_library_names_and_values():
    from quantem.widget.crystal import PHASE_LIBRARY, library_phase

    assert {"Au", "Al", "Si", "α-Fe", "MgO", "Fe3O4"} <= set(PHASE_LIBRARY)
    au = library_phase("Au")
    assert au.name == "Au" and au.absences == "fcc"
    assert au.d_spacing((1, 1, 1)) == pytest.approx(4.0782 / math.sqrt(3), abs=1e-4)
    si = library_phase("Si")
    assert si.absences == "diamond"
    assert si.d_spacing((3, 1, 1)) == pytest.approx(5.4311 / math.sqrt(11), abs=1e-4)
    fe = library_phase("α-Fe")
    assert fe.absences == "bcc"


def test_library_phase_unknown_raises():
    from quantem.widget.crystal import library_phase

    with pytest.raises(ValueError, match="Kryptonite"):
        library_phase("Kryptonite")


def test_hcp_absence_rule():
    """P6_3/mmc: reflections with h+2k = 3n and l odd are absent."""
    ti = Phase("Ti", 2.9505, 2.9505, 4.6826, 90.0, 90.0, 120.0, absences="hcp")
    assert not ti.is_allowed((0, 0, 1))
    assert ti.is_allowed((0, 0, 2))
    assert ti.is_allowed((1, 0, 0))
    assert ti.is_allowed((1, 0, 1))
    assert not ti.is_allowed((1, 1, 1))  # h+2k = 3, l odd
    assert ti.is_allowed((1, 1, 2))


def test_hexagonal_library_dspacings():
    from quantem.widget.crystal import library_phase

    ti = library_phase("Ti")
    assert ti.d_spacing((1, 0, 0)) == pytest.approx(2.9500 * math.sqrt(3) / 2, abs=1e-3)
    graphite = library_phase("C (graphite)")
    assert graphite.d_spacing((0, 0, 2)) == pytest.approx(6.7110 / 2, abs=1e-3)
    assert not graphite.is_allowed((0, 0, 1))


def test_expanded_library_coverage():
    from quantem.widget.crystal import PHASE_LIBRARY, library_phase

    assert len(PHASE_LIBRARY) >= 40
    expected = {
        "Pd",
        "Pb",
        "Mo",
        "Ta",
        "NaCl",
        "TiN",
        "CaF2",
        "CeO2",
        "GaAs",
        "ZnS",
        "3C-SiC",
        "SrTiO3",
        "MgAl2O4",
        "Ti",
        "Mg",
        "Zn",
        "C (graphite)",
    }
    assert expected <= set(PHASE_LIBRARY)
    gaas = library_phase("GaAs")
    assert gaas.d_spacing((1, 1, 1)) == pytest.approx(5.6533 / math.sqrt(3), abs=1e-4)
    srtio3 = library_phase("SrTiO3")
    assert srtio3.is_allowed((1, 0, 0))  # simple cubic: no centering extinctions


def test_wurtzite_absence_rule():
    """Both species sit at 2b (1/3, 2/3, z), so wurtzite zeros the same
    reflections as hcp: h + 2k = 3n with odd l are structurally absent."""
    zno = Phase("ZnO", 3.2495, 3.2495, 5.2069, 90.0, 90.0, 120.0, absences="wurtzite")
    assert not zno.is_allowed((0, 0, 1))
    assert not zno.is_allowed((1, 1, 1))
    assert not zno.is_allowed((3, 0, 1))
    assert zno.is_allowed((0, 0, 2))
    assert zno.is_allowed((1, 0, 0))
    assert zno.is_allowed((1, 0, 1))
    assert zno.is_allowed((1, 1, 0))
    assert zno.is_allowed((1, 0, 3))


def test_non_cubic_reflection_labels_keep_axis_order():
    """Non-cubic phases keep h, k, l order; cubic phases use family labels."""
    graphite = Phase(
        "graphite",
        2.4640,
        2.4640,
        6.7110,
        90.0,
        90.0,
        120.0,
        absences="hcp",
    )
    by_d = graphite.reflections(d_min=2.0)
    assert by_d[0]["hkl_str"] == "002"
    assert by_d[0]["d"] == pytest.approx(6.7110 / 2, abs=1e-3)

    rutile = Phase("rutile", 4.5940, 4.5940, 2.9589)
    labels = {reflection["hkl_str"] for reflection in rutile.reflections(d_min=2.8)}
    assert "001" in labels
    assert "100" in labels


def test_rhombohedral_absence_rule():
    """R-3m (hexagonal setting): -h + k + l = 3n."""
    bi = Phase("Bi", 4.5460, 4.5460, 11.8620, 90.0, 90.0, 120.0, absences="rhombohedral")
    assert bi.is_allowed((0, 0, 3))
    assert bi.is_allowed((1, 0, 4))
    assert bi.is_allowed((1, 1, 0))
    assert bi.is_allowed((0, 1, 2))
    assert not bi.is_allowed((1, 0, 0))
    assert not bi.is_allowed((0, 1, 1))


def test_rhombohedral_c_glide_absence_rule():
    """R-3c/R3c adds a c glide: 00l, h0l and 0kl with odd l are absent."""
    hem = Phase("Fe2O3", 5.0356, 5.0356, 13.7489, 90.0, 90.0, 120.0, absences="rhombohedral-c")
    assert hem.is_allowed((0, 1, 2))
    assert hem.is_allowed((1, 0, 4))
    assert hem.is_allowed((1, 1, 0))
    assert hem.is_allowed((3, 0, 0))
    assert not hem.is_allowed((0, 0, 3))
    assert not hem.is_allowed((1, 0, 1))
    assert not hem.is_allowed((0, 1, 5))


def test_tetragonal_and_new_library_entries():
    from quantem.widget.crystal import PHASE_LIBRARY, library_phase

    assert len(PHASE_LIBRARY) >= 90
    anatase = library_phase("TiO2 (anatase)")
    assert anatase.d_spacing((1, 0, 1)) == pytest.approx(3.517, abs=5e-3)
    rutile = library_phase("TiO2 (rutile)")
    assert rutile.d_spacing((1, 1, 0)) == pytest.approx(4.5940 / math.sqrt(2), abs=1e-3)
    hematite = library_phase("α-Fe2O3 (hematite)")
    assert hematite.d_spacing((1, 0, 4)) == pytest.approx(2.700, abs=5e-3)
    zno = library_phase("ZnO")
    assert zno.d_spacing((1, 0, 1)) == pytest.approx(2.476, abs=5e-3)
    lab6 = library_phase("LaB6")
    assert lab6.is_allowed((1, 0, 0))
    assert lab6.d_spacing((1, 1, 0)) == pytest.approx(4.1568 / math.sqrt(2), abs=1e-3)


def test_library_additions():
    """d-spacing spot checks for the newer library entries against their cited cells."""
    from quantem.widget.crystal import library_phase

    checks = {
        "CaO": ((2, 0, 0), 2.4054),
        "ZrC": ((2, 0, 0), 2.3502),
        "CuI": ((1, 1, 1), 3.5005),
        "Co3O4": ((3, 1, 1), 2.4368),
        "NiFe2O4": ((3, 1, 1), 2.5205),
        "BaTiO3": ((1, 1, 0), 2.8217),
        "WC": ((0, 0, 1), 2.8377),
        "TiB2": ((1, 0, 1), 2.0359),
        "ZnS (wurtzite)": ((1, 0, 0), 3.3105),
        "Bi": ((0, 1, 2), 3.2802),
        "Sb": ((0, 1, 2), 3.1113),
    }
    for name, (hkl, d_ref) in checks.items():
        phase = library_phase(name)
        assert phase.is_allowed(hkl)
        assert abs(phase.d_spacing(hkl) - d_ref) < 0.002
