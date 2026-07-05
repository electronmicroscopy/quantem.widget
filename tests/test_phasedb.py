import pytest

from quantem.widget.phasedb import (
    match_candidate,
    match_sort_key,
    phases_from_cifs,
    structure_reflections,
)

# Diamond-cubic Si, P1 setting: robust to parse without symmetry machinery
SI_CIF = """data_Si
_cell_length_a 5.431
_cell_length_b 5.431
_cell_length_c 5.431
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Si1 Si 0.00 0.00 0.00 1
Si2 Si 0.50 0.50 0.00 1
Si3 Si 0.50 0.00 0.50 1
Si4 Si 0.00 0.50 0.50 1
Si5 Si 0.25 0.25 0.25 1
Si6 Si 0.75 0.75 0.25 1
Si7 Si 0.75 0.25 0.75 1
Si8 Si 0.25 0.75 0.75 1
"""


# --- Electron intensities ---


def test_structure_reflections_si_electron_intensities():
    pytest.importorskip("pymatgen")
    from pymatgen.core import Lattice, Structure

    si = Structure.from_spacegroup(227, Lattice.cubic(5.431), ["Si"], [[0, 0, 0]])
    refls = structure_reflections(si, d_min=0.8)
    ds = [r["d"] for r in refls]
    assert ds == sorted(ds, reverse=True)
    by_hkl = {r["hkl_str"]: r for r in refls}
    assert by_hkl["111"]["d"] == pytest.approx(3.135, abs=5e-3)
    assert "200" not in by_hkl  # diamond systematic absence
    assert not any(abs(d - 5.431 / 2) < 5e-3 for d in ds)
    # electron (Mott-Bethe) intensities, not X-ray + LP: 220 nearly as bright as 111
    assert by_hkl["220"]["intensity"] / by_hkl["111"]["intensity"] > 0.85
    assert by_hkl["111"]["multiplicity"] == 8


# --- Matcher (no pymatgen needed) ---


def _lines(entries):
    return [{"d": d, "i_rel": i} for d, i in entries]


def test_match_candidate_perfect_match():
    refs = _lines([(3.0, 100), (2.0, 80), (1.5, 60), (1.2, 40)])
    sc = match_candidate([3.0, 2.0, 1.5, 1.2], refs, tol=0.03)
    assert sc["matched"] == 4 and sc["n_obs"] == 4
    assert sc["mean_err"] == pytest.approx(0.0, abs=1e-9)
    assert sc["n_missing_strong"] == 0
    assert all(j is not None for _, j in sc["assignments"])


def test_match_candidate_uses_absolute_calibration():
    refs = _lines([(3.0, 100), (2.0, 80), (1.5, 60), (1.2, 40)])
    perfect = match_candidate([3.0, 2.0, 1.5, 1.2], refs, tol=0.03)
    off = match_candidate([3.0 * 1.05, 2.0 * 1.05, 1.5 * 1.05, 1.2 * 1.05], refs, tol=0.03)
    assert off["matched"] == 0
    assert match_sort_key(perfect) < match_sort_key(off)


def test_match_candidate_uses_each_reference_once():
    sc = match_candidate([2.0, 2.01], _lines([(2.0, 100)]), tol=0.03)
    assert sc["matched"] == 1
    assert [ref for _, ref in sc["assignments"]].count(0) == 1


def test_rank_unexplained_ring_below_full_explanation():
    # same observations, two candidate phases: one explains every ring,
    # the other leaves the 1.05 ring unexplained
    obs = [3.0, 2.0, 1.5, 1.05]
    explains_all = match_candidate(obs, _lines([(3.0, 100), (2.0, 80), (1.5, 60), (1.05, 40)]))
    leaves_one = match_candidate(obs, _lines([(3.0, 100), (2.0, 80), (1.5, 60), (1.2, 40)]))
    assert explains_all["matched"] == 4 and leaves_one["matched"] == 3
    assert match_sort_key(explains_all) < match_sort_key(leaves_one)


def test_rank_missing_strong_below_missing_weak():
    strong = _lines([(3.0, 100), (2.0, 90), (1.5, 80)])
    weak = _lines([(3.0, 100), (2.0, 10), (1.5, 80)])
    obs = [3.0, 1.5]  # the d=2.0 line is inside the observed range but unmatched
    sc_strong = match_candidate(obs, strong, tol=0.03)
    sc_weak = match_candidate(obs, weak, tol=0.03)
    assert sc_strong["n_missing_strong"] == 1 and sc_weak["n_missing_strong"] == 0
    assert match_sort_key(sc_weak) < match_sort_key(sc_strong)


def test_match_candidate_accepts_structure_reflections_key():
    # structure_reflections emits "intensity"; the matcher must honor it too
    refs = [
        {"d": 3.0, "intensity": 100.0},
        {"d": 2.0, "intensity": 90.0},
        {"d": 1.5, "intensity": 80.0},
    ]
    sc = match_candidate([3.0, 1.5], refs, tol=0.03)
    assert sc["matched"] == 2
    assert sc["n_missing_strong"] == 1  # the strong d=2.0 line is unexplained


def test_match_candidate_keeps_plain_match_report():
    dense = _lines([(1.0 + 0.01 * i, 20) for i in range(200)])  # 200 lines, d 1.0-2.99
    sc = match_candidate([2.5, 1.7], dense, tol=0.03)
    assert sc["matched"] == 2
    assert set(sc) == {
        "matched",
        "n_obs",
        "mean_err",
        "n_missing_strong",
        "assignments",
    }


def test_match_candidate_no_reference_lines():
    sc = match_candidate([2.5, 1.7], [], tol=0.03)
    assert sc["matched"] == 0
    assert sc["assignments"] == []


# --- Local CIF loading ---


def test_phases_from_cifs_loads_directory(tmp_path):
    pytest.importorskip("pymatgen")
    (tmp_path / "si.cif").write_text(SI_CIF)
    phases = phases_from_cifs(tmp_path)
    assert [p.name for p in phases] == ["Si"]
    r0 = phases[0].reflections()[0]
    assert r0["d"] == pytest.approx(3.135, abs=5e-3)
    assert r0["intensity"] is not None


def test_phases_from_cifs_accepts_file_list(tmp_path):
    pytest.importorskip("pymatgen")
    path = tmp_path / "si.cif"
    path.write_text(SI_CIF)
    phases = phases_from_cifs([path])
    assert [p.name for p in phases] == ["Si"]


def test_phases_from_cifs_skips_bad_files(tmp_path):
    pytest.importorskip("pymatgen")
    (tmp_path / "good.cif").write_text(SI_CIF)
    (tmp_path / "bad.cif").write_text("this is not a cif file\n")
    phases = phases_from_cifs(tmp_path)
    assert [p.name for p in phases] == ["Si"]


def test_phases_from_cifs_all_bad_raises(tmp_path):
    pytest.importorskip("pymatgen")
    (tmp_path / "bad.cif").write_text("this is not a cif file\n")
    with pytest.raises(ValueError, match="no CIF"):
        phases_from_cifs(tmp_path)
