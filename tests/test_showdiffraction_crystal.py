import math

import numpy as np
import pytest

from quantem.widget import Phase

# Gold: FCC, a = 4.078 Å. d_hkl and plane angles are analytic for a cubic cell.
AU_A = 4.078


def test_cubic_dspacing_and_plane_angles():
    au = Phase.from_cubic("Au", AU_A, absences="fcc")
    # d_hkl = a / sqrt(h^2+k^2+l^2)
    assert au.d_spacing((1, 1, 1)) == pytest.approx(AU_A / np.sqrt(3), abs=1e-4)
    assert au.d_spacing((2, 0, 0)) == pytest.approx(AU_A / 2, abs=1e-4)
    assert au.d_spacing((2, 2, 0)) == pytest.approx(AU_A / np.sqrt(8), abs=1e-4)
    assert au.d_spacing((3, 1, 1)) == pytest.approx(AU_A / np.sqrt(11), abs=1e-4)
    # (111)^(200) = 54.7356°, (200)^(020) = 90°
    assert au.plane_angle((1, 1, 1), (2, 0, 0)) == pytest.approx(54.7356, abs=1e-2)
    assert au.plane_angle((2, 0, 0), (0, 2, 0)) == pytest.approx(90.0, abs=1e-4)


def test_reflections_and_match_d():
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

    # match_d returns ranked candidates
    d111 = AU_A / np.sqrt(3)
    cands = au.match_d(d111 * 1.005, tol=0.02)  # measured 0.5% high
    assert cands, "expected at least one candidate within tolerance"
    assert cands[0]["hkl_str"] == "111"
    assert cands[0]["d_error"] == pytest.approx(0.005, abs=2e-3)  # fractional
    # a d-spacing far from any reflection returns nothing within a tight tol
    assert au.match_d(3.5, tol=0.01) == []

    # default reflections reach high-index families
    fe3o4 = Phase.from_cubic("Fe3O4", 8.3963, absences="fcc")
    d800 = 8.3963 / np.sqrt(64)
    by_hkl = {r["hkl_str"]: r for r in fe3o4.reflections(d_min=0.9)}
    assert "800" in by_hkl
    assert by_hkl["800"]["d"] == pytest.approx(d800, abs=1e-4)
    assert "840" in by_hkl
    cands = fe3o4.match_d(d800, tol=0.01)
    assert cands and cands[0]["hkl_str"] == "800"

    # non-cubic phases keep h, k, l order; cubic phases use family labels
    graphite = Phase("graphite", 2.4640, 2.4640, 6.7110, 90.0, 90.0, 120.0, absences="hcp")
    by_d = graphite.reflections(d_min=2.0)
    assert by_d[0]["hkl_str"] == "002"
    assert by_d[0]["d"] == pytest.approx(6.7110 / 2, abs=1e-3)
    rutile = Phase("rutile", 4.5940, 4.5940, 2.9589)
    labels = {reflection["hkl_str"] for reflection in rutile.reflections(d_min=2.8)}
    assert "001" in labels
    assert "100" in labels


def test_match_d_reference_relative_error():
    # match_d and match_candidate agree at the tolerance boundary
    from quantem.widget.showdiffraction import match_candidate

    au = Phase.from_cubic("Au", AU_A, absences="fcc")
    d_ref = au.d_spacing((1, 1, 1))
    # 3.05% above the reference line: rejected in both paths
    d_obs = d_ref * 1.0305
    assert au.match_d(d_obs, tol=0.03) == []
    report = match_candidate([d_obs], [{"d": d_ref, "intensity": 100.0}], tol=0.03)
    assert report["matched"] == 0
    # 2.95% above: accepted in both paths with the same error
    d_obs = d_ref * 1.0295
    cands = au.match_d(d_obs, tol=0.03)
    assert cands and cands[0]["hkl_str"] == "111"
    assert cands[0]["d_error"] == pytest.approx(0.0295, abs=1e-4)
    report = match_candidate([d_obs], [{"d": d_ref, "intensity": 100.0}], tol=0.03)
    assert report["matched"] == 1
    assert report["mean_err"] == pytest.approx(0.0295, abs=1e-4)


def test_reflections_degenerate_families():
    # distinct cubic families at one d keep both labels
    si = Phase.from_cubic("Si", 5.4310, absences="diamond")
    d27 = 5.4310 / math.sqrt(27)
    refls = [r for r in si.reflections(d_min=1.0) if abs(r["d"] - d27) < 1e-3]
    assert len(refls) == 1
    assert refls[0]["hkl_str"] == "511/333"
    assert refls[0]["hkl"] == (5, 1, 1)
    assert refls[0]["multiplicity"] == 32
    cands = si.match_d(d27, tol=0.005)
    assert cands and cands[0]["hkl_str"] == "511/333"
    # hexagonal family members keep a single conventional label
    from quantem.widget import library_phase

    graphite = library_phase("C (graphite)")
    labels = {r["hkl_str"] for r in graphite.reflections(d_min=1.2)}
    assert "100" in labels
    assert not any("010" in label for label in labels)


def test_from_dspacings_and_lattice_validation():
    # reference table keeps its own hkl labels and matches back
    ref = Phase.from_dspacings("Sample", [(2.355, "111"), (2.039, "200"), (1.442, "220")])
    refls = ref.reflections()
    assert [r["hkl_str"] for r in refls] == ["111", "200", "220"]
    cands = ref.match_d(2.04, tol=0.02)
    assert cands[0]["hkl_str"] == "200"

    # optional intensity per line carries through reflections and matches
    ref = Phase.from_dspacings("X", [(2.53, "311", 100), (1.48, "440", 40), (2.10, "400")])
    by_hkl = {r["hkl_str"]: r for r in ref.reflections()}
    assert by_hkl["311"]["intensity"] == 100
    assert by_hkl["400"]["intensity"] is None
    assert ref.match_d(2.53, tol=0.02)[0]["intensity"] == 100

    # a non-positive lattice constant is rejected
    with pytest.raises(ValueError):
        Phase.from_cubic("bad", 0.0)
    with pytest.raises(ValueError):
        Phase.from_cubic("bad", -1.0)


def test_degenerate_cell_rejected():
    # angles outside (0, 180) are invalid
    for angles in ({"gamma": 180.0}, {"alpha": 0.0}, {"beta": -30.0}, {"gamma": 360.0}):
        with pytest.raises(ValueError):
            Phase("bad", 4.0, 4.0, 4.0, **angles)
    # in-range angles with a negative metric determinant are invalid too
    with pytest.raises(ValueError):
        Phase("bad", 4.0, 4.0, 4.0, 150.0, 150.0, 150.0)


def test_reflections_bounded():
    # absurd d_min: the 0.2 A floor and index cap keep enumeration finite
    import time

    au = Phase.from_cubic("Au", AU_A, absences="fcc")
    t0 = time.perf_counter()
    refls = au.reflections(d_min=1e-6)
    assert time.perf_counter() - t0 < 2.0
    assert refls
    assert min(r["d"] for r in refls) >= 0.2
    # an explicit max_index above the cap is clamped
    t0 = time.perf_counter()
    au.reflections(d_min=0.5, max_index=500)
    assert time.perf_counter() - t0 < 2.0


def test_structure_absence_rules():
    cases = [
        # fcc: all-even or all-odd allowed; mixed parity forbidden
        (
            Phase.from_cubic("Au", AU_A, absences="fcc"),
            [(1, 1, 1), (2, 0, 0), (2, 2, 0)],
            [(1, 0, 0), (1, 1, 0), (2, 1, 0)],
        ),
        # bcc: h+k+l even allowed
        (
            Phase.from_cubic("Fe", 2.866, absences="bcc"),
            [(1, 1, 0), (2, 0, 0), (2, 1, 1)],
            [(1, 0, 0), (1, 1, 1)],
        ),
        # diamond: all-odd allowed; all-even needs h+k+l divisible by 4
        (
            Phase.from_cubic("Si", 5.4310, absences="diamond"),
            [(1, 1, 1), (2, 2, 0), (3, 1, 1), (4, 0, 0)],
            [(2, 0, 0), (2, 2, 2), (4, 2, 0)],
        ),
        # hcp (P6_3/mmc): h+2k = 3n with l odd are absent
        (
            Phase("Ti", 2.9505, 2.9505, 4.6826, 90.0, 90.0, 120.0, absences="hcp"),
            [(0, 0, 2), (1, 0, 0), (1, 0, 1), (1, 1, 2)],
            [(0, 0, 1), (1, 1, 1)],
        ),
        # wurtzite: same extinctions as hcp (h+2k = 3n, l odd)
        (
            Phase("ZnO", 3.2495, 3.2495, 5.2069, 90.0, 90.0, 120.0, absences="wurtzite"),
            [(0, 0, 2), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 0, 3)],
            [(0, 0, 1), (1, 1, 1), (3, 0, 1)],
        ),
        # rhombohedral R-3m (hex setting): -h + k + l = 3n
        (
            Phase("Bi", 4.5460, 4.5460, 11.8620, 90.0, 90.0, 120.0, absences="rhombohedral"),
            [(0, 0, 3), (1, 0, 4), (1, 1, 0), (0, 1, 2)],
            [(1, 0, 0), (0, 1, 1)],
        ),
        # rhombohedral-c R-3c/R3c: c glide zeros 00l, h0l, 0kl with odd l
        (
            Phase("Fe2O3", 5.0356, 5.0356, 13.7489, 90.0, 90.0, 120.0, absences="rhombohedral-c"),
            [(0, 1, 2), (1, 0, 4), (1, 1, 0), (3, 0, 0)],
            [(0, 0, 3), (1, 0, 1), (0, 1, 5)],
        ),
        # spinel
        (
            Phase.from_cubic("Fe3O4", 8.3967, absences="spinel"),
            [(1, 1, 1), (2, 2, 0), (3, 1, 1), (2, 2, 2), (4, 0, 0), (4, 2, 2), (4, 4, 0), (6, 2, 0)],
            [(2, 0, 0), (4, 2, 0), (6, 0, 0), (6, 4, 0), (2, 1, 0)],
        ),
        # i41amd (anatase / β-Sn)
        (
            Phase("TiO2", 3.7852, 3.7852, 9.5139, absences="i41amd"),
            [(1, 0, 1), (0, 0, 4), (1, 0, 3), (1, 1, 2), (2, 0, 0), (2, 1, 1), (2, 2, 0), (2, 0, 4), (1, 1, 6)],
            [(0, 0, 2), (1, 1, 0), (2, 2, 2), (4, 0, 2), (1, 0, 0)],
        ),
        # rutile
        (
            Phase("TiO2", 4.5940, 4.5940, 2.9589, absences="rutile"),
            [(1, 1, 0), (1, 0, 1), (2, 0, 0), (1, 1, 1), (2, 1, 0), (0, 0, 2)],
            [(1, 0, 0), (0, 0, 1), (0, 1, 2), (1, 0, 2)],
        ),
        # bixbyite
        (
            Phase.from_cubic("Y2O3", 10.6040, absences="bixbyite"),
            [(2, 1, 1), (2, 2, 2), (4, 0, 0), (4, 1, 1), (3, 3, 2), (4, 4, 0)],
            [(1, 1, 0), (3, 1, 0), (1, 0, 1), (2, 1, 0)],
        ),
        # cuprite
        (
            Phase.from_cubic("Cu2O", 4.2696, absences="cuprite"),
            [(1, 1, 0), (1, 1, 1), (2, 0, 0), (2, 1, 1), (3, 1, 1), (2, 2, 2)],
            [(1, 0, 0), (2, 1, 0), (2, 2, 1), (3, 0, 0)],
        ),
    ]
    for phase, allowed, forbidden in cases:
        for hkl in allowed:
            assert phase.is_allowed(hkl), (phase.absences, hkl)
        for hkl in forbidden:
            assert not phase.is_allowed(hkl), (phase.absences, hkl)


def test_phase_library():
    from quantem.widget import library_phase
    from quantem.widget.showdiffraction import PHASE_LIBRARY

    # core names + per-phase metadata and d-spacings
    assert {"Au", "Al", "Si", "α-Fe", "MgO", "Fe3O4"} <= set(PHASE_LIBRARY)
    au = library_phase("Au")
    assert au.name == "Au" and au.absences == "fcc"
    assert au.d_spacing((1, 1, 1)) == pytest.approx(4.0782 / math.sqrt(3), abs=1e-4)
    si = library_phase("Si")
    assert si.absences == "diamond"
    assert si.d_spacing((3, 1, 1)) == pytest.approx(5.4311 / math.sqrt(11), abs=1e-4)
    fe = library_phase("α-Fe")
    assert fe.absences == "bcc"

    # unknown name raises
    with pytest.raises(ValueError, match="Kryptonite"):
        library_phase("Kryptonite")

    # hexagonal d-spacings and absence
    ti = library_phase("Ti")
    assert ti.d_spacing((1, 0, 0)) == pytest.approx(2.9500 * math.sqrt(3) / 2, abs=1e-3)
    graphite = library_phase("C (graphite)")
    assert graphite.d_spacing((0, 0, 2)) == pytest.approx(6.7110 / 2, abs=1e-3)
    assert not graphite.is_allowed((0, 0, 1))

    # expanded coverage (>= 40 entries)
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

    # tetragonal + newer entries (>= 90 entries)
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

    # d-spacing spot checks for later additions against their cited cells
    checks = {
        "CaO": ((2, 0, 0), 2.4054),
        "ZrC": ((2, 0, 0), 2.3502),
        "CuI": ((1, 1, 1), 3.5005),
        "Co3O4": ((3, 1, 1), 2.4368),
        "NiFe2O4": ((3, 1, 1), 2.5143),
        "BaTiO3": ((1, 1, 0), 2.8242),
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


def test_library_absence_reassignments():
    from quantem.widget import library_phase
    from quantem.widget.showdiffraction import PHASE_LIBRARY

    spinels = ["Fe3O4", "γ-Fe2O3", "MgAl2O4", "Co3O4", "CoFe2O4", "ZnFe2O4", "γ-Al2O3"]
    for name in spinels + ["NiFe2O4"]:
        assert PHASE_LIBRARY[name]["absences"] == "spinel"

    fe3o4 = library_phase("Fe3O4")
    by_hkl = {r["hkl_str"]: r["d"] for r in fe3o4.reflections(d_min=1.0)}
    assert "200" not in by_hkl and "420" not in by_hkl
    assert by_hkl["111"] == pytest.approx(4.848, abs=1e-3)
    assert by_hkl["220"] == pytest.approx(2.969, abs=1e-3)
    assert by_hkl["311"] == pytest.approx(2.532, abs=1e-3)
    assert by_hkl["222"] == pytest.approx(2.424, abs=1e-3)

    anatase = library_phase("TiO2 (anatase)")
    assert anatase.absences == "i41amd"
    by_hkl = {r["hkl_str"]: r["d"] for r in anatase.reflections(d_min=1.5)}
    assert "002" not in by_hkl and "110" not in by_hkl
    assert by_hkl["101"] == pytest.approx(3.517, abs=1e-3)

    rutile = library_phase("TiO2 (rutile)")
    assert rutile.absences == "rutile"
    by_hkl = {r["hkl_str"]: r["d"] for r in rutile.reflections(d_min=1.0)}
    assert "100" not in by_hkl and "001" not in by_hkl
    assert by_hkl["110"] == pytest.approx(3.248, abs=1e-3)

    sno2 = library_phase("SnO2")
    assert sno2.absences == "rutile"
    by_hkl = {r["hkl_str"]: r["d"] for r in sno2.reflections(d_min=1.0)}
    assert "100" not in by_hkl and "001" not in by_hkl
    assert by_hkl["110"] == pytest.approx(3.350, abs=1e-3)

    sn = library_phase("β-Sn")
    assert sn.absences == "i41amd"
    by_hkl = {r["hkl_str"]: r["d"] for r in sn.reflections(d_min=1.0)}
    assert "110" not in by_hkl
    assert by_hkl["200"] == pytest.approx(2.916, abs=1e-3)
    assert by_hkl["101"] == pytest.approx(2.793, abs=1e-3)

    for name, d211 in [("Y2O3", 4.330), ("In2O3", 4.130)]:
        oxide = library_phase(name)
        assert oxide.absences == "bixbyite"
        by_hkl = {r["hkl_str"]: r["d"] for r in oxide.reflections(d_min=1.5)}
        assert "110" not in by_hkl and "310" not in by_hkl
        assert by_hkl["211"] == pytest.approx(d211, abs=2e-3)

    cu2o = library_phase("Cu2O")
    assert cu2o.absences == "cuprite"
    by_hkl = {r["hkl_str"]: r["d"] for r in cu2o.reflections(d_min=1.0)}
    assert "100" not in by_hkl and "210" not in by_hkl
    assert by_hkl["110"] == pytest.approx(3.020, abs=2e-3)
