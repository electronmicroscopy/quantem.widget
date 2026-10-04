"""PlanPtycho: the arithmetic matches the shared goldens, and the crystal side builds exact oriented cells."""

import json
import math
import pathlib

import numpy as np
import pytest

from quantem.widget.planptycho import (
    check_rows, check_statuses, detector_sampling_mrad, plan_geometry, recommended_settings, wavelength_A,
)

GOLDENS = json.loads((pathlib.Path(__file__).resolve().parents[2] / "js" / "planptycho" / "goldens.json").read_text())


def srtio3():
    from ase import Atoms
    return Atoms("SrTiO3", scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)], cell=[3.905] * 3, pbc=True)


def plan(**kwargs):
    pytest.importorskip("abtem")
    from quantem.widget import PlanPtycho
    return PlanPtycho(srtio3(), **kwargs)


# --- arithmetic shared with the browser

@pytest.mark.parametrize("name", sorted(GOLDENS["cases"]))
def test_geometry_grades_and_text_match_goldens(name):
    case = GOLDENS["cases"][name]; settings = case["settings"]
    geometry = plan_geometry(**settings)
    for key, expected in case["geometry"].items():
        assert geometry[key] == (None if expected is None else pytest.approx(expected, rel=1e-12)), key
    assert check_statuses(geometry) == case["statuses"]
    assert check_rows(geometry, detector_px=settings["detector_px"], scan_step_A=settings["scan_step_A"],
                      column_phase_rad_per_A=GOLDENS["column_phase_rad_per_A"]) == case["rows"]


def test_recommendations_and_calibration_match_goldens():
    for r in GOLDENS["recommended"]:
        assert recommended_settings(r["thickness_nm"], voltage_kV=r["voltage_kV"], semiangle_mrad=r["semiangle_mrad"], scan_step_A=r["scan_step_A"]) == r["expected"]
    for r in GOLDENS["detector_sampling"]:
        assert detector_sampling_mrad(r["detector"], r["camera_length_mm"]) == pytest.approx(r["expected"], rel=1e-12)


def test_srtio3_110nm_numbers():
    """110 nm SrTiO3 on a 184 px detector at 0.563 mrad: 35 A window, 0.19 A pixels, ~57 A beam at the exit, 4.4 nm depth of field."""
    g = plan_geometry(**GOLDENS["cases"]["srtio3_110nm_arina"]["settings"])
    assert wavelength_A(300) == pytest.approx(0.019687, abs=2e-6)
    assert g["window_A"] == pytest.approx(0.019687 / 0.563e-3, rel=1e-4)
    assert g["pixel_A"] == pytest.approx(0.190, abs=1e-3)
    assert g["exit_A"] == pytest.approx(2 * 0.030 * (1100 - 170) + 1.22 * g["wavelength_A"] / 0.030, rel=1e-12)
    assert g["depth_of_field_A"] == pytest.approx(43.75, abs=0.05)
    assert plan_geometry(**GOLDENS["cases"]["srtio3_110nm_tilted_4mrad"]["settings"])["lean_A"] == pytest.approx(1100 * math.tan(4e-3), rel=1e-9)


def test_recommended_focus_splits_the_specimen_and_every_check_passes_to_100_nm():
    """Arina 91 mm, 30 mrad: to 100 nm the recommended focus and scan pass every check. Beyond that no focus fits the beam
    in the 35.5 A window (150 nm: 46 A, 200 nm: 61 A); only the window cautions, and a longer camera length trades it for reach."""
    for thickness in (20, 30, 40, 50, 60, 70, 100, 150, 200):
        r = recommended_settings(thickness, voltage_kV=300, semiangle_mrad=30, scan_step_A=0.5)
        g = plan_geometry(voltage_kV=300, semiangle_mrad=30, focus_depth_nm=r["focus_depth_nm"], thickness_nm=thickness,
                          detector_px=192, detector_mrad_per_px=0.554, scan_step_A=0.5, scan_size_px=r["scan_size_px"])
        not_passing = {k for k, v in check_statuses(g).items() if v != "pass"}
        assert not_passing == (set() if thickness <= 100 else {"window"}), thickness


# --- crystal

def test_oriented_cells_are_exact_and_small():
    """Rotating an in-plane lattice vector onto x first makes the orthogonal cell exact: no strain, 30 atoms for [111]."""
    pytest.importorskip("abtem")
    from quantem.widget.planptycho import oriented_cell
    expected = {(0, 0, 1): ([3.905, 3.905, 3.905], 5), (0, 1, 1): ([3.905, 5.523, 5.523], 10), (1, 1, 1): ([5.523, 9.565, 6.764], 30), (1, 1, 2): ([5.523, 6.764, 9.565], 30)}
    for zone, (size, atoms) in expected.items():
        cell = oriented_cell(srtio3(), zone)
        assert np.diag(cell.cell.array) == pytest.approx(size, abs=1e-3) and len(cell) == atoms, zone
    with pytest.raises(ValueError, match="without straining"):
        oriented_cell(srtio3(), (3, 5, 7))


def test_holz_period_counts_centring():
    """fcc Au [011]: the orthogonal repeat is 5.77 A but the first non-empty reciprocal layer is the second (period 2.88 A)."""
    pytest.importorskip("abtem")
    from ase.build import bulk
    from quantem.widget.planptycho import holz_repeat_A, oriented_cell
    assert holz_repeat_A(oriented_cell(bulk("Au", cubic=True), (0, 1, 1))) == pytest.approx(4.078 / math.sqrt(2), abs=0.01)
    assert holz_repeat_A(oriented_cell(bulk("Fe", cubic=True), (1, 1, 1))) == pytest.approx(2.87 * math.sqrt(3) / 2, abs=0.01)
    assert holz_repeat_A(oriented_cell(srtio3(), (0, 0, 1))) == pytest.approx(3.905, abs=1e-6)


def test_srtio3_bragg_orders_and_column_phase():
    pytest.importorskip("abtem")
    from quantem.widget.planptycho import bragg_reflections, column_phase_rad_per_A, interaction_constant, oriented_cell, projected_potential
    cell = oriented_cell(srtio3(), (0, 0, 1)); phase = interaction_constant(300) * projected_potential(cell)
    assert phase.max() == pytest.approx(1.729, abs=0.01)                       # strongest (Sr) column per 3.905 A repeat, 0.05 A grid
    reflections = bragg_reflections(phase, (3.905, 3.905))
    order = lambda h, k: [r[2] for r in reflections if math.hypot(r[0], r[1]) == pytest.approx(math.hypot(h, k) / 3.905, rel=1e-9)]
    # [001]: Sr and TiO columns on one checkerboard, O on the other. {200} all in phase (strongest), {110} O against the rest,
    # {100} only Sr against TiO (Z 38 vs 30)
    assert max(order(2, 0)) == pytest.approx(1.0) and 0.5 < min(order(1, 1)) < 0.95 and 0.001 < max(order(1, 0)) < 0.05
    assert column_phase_rad_per_A(phase, [3.905] * 3, 0.19) < phase.max() / 3.905


# --- widget

def test_defaults_come_from_the_arina_preset_and_focus_splits_the_specimen():
    p = plan(thickness_nm=60)
    assert (p.detector, p.camera_length_mm, p.detector_px, p.detector_mrad_per_px, p.voltage_kV, p.semiangle_mrad) == ("Arina", 91.0, 192, pytest.approx(0.554), 300.0, 30.0)
    assert p.focus_depth_nm == 30.0
    assert set(p.report()["status"]) <= {"pass", "info"}


def test_zone_change_keeps_a_custom_title_and_builds_once(monkeypatch):
    pytest.importorskip("abtem")
    import quantem.widget.planptycho as module
    calls = []
    real = module.oriented_cell
    monkeypatch.setattr(module, "oriented_cell", lambda *a, **k: calls.append(a[1]) or real(*a, **k))
    p = module.PlanPtycho(srtio3(), zone_axis=(0, 1, 1), title="Mine")
    assert calls == [[0, 1, 1]] and p.title == "Mine"
    p.zone_axis = [1, 1, 1]
    assert p.title == "Mine" and calls[-1] == [1, 1, 1]
    q = module.PlanPtycho(srtio3())
    q.zone_axis = [0, 1, 1]
    assert q.title == "SrTiO3 [011]" and q.cell_size_A[1] == pytest.approx(3.905 * math.sqrt(2), rel=1e-6)


def test_camera_and_length_recalibrate_from_python():
    p = plan()
    p.camera_length_mm = 185
    assert p.detector_mrad_per_px == pytest.approx(0.269)
    p.detector = "EMPAD"
    assert (p.detector_px, p.detector_mrad_per_px) == (128, pytest.approx(0.269 * 1.5))
    binned = plan(detector_px=96)
    assert binned.detector_mrad_per_px * binned.detector_px == pytest.approx(0.554 * 192)
    binned.camera_length_mm = 115
    assert (binned.detector_px, binned.detector_mrad_per_px) == (96, pytest.approx(0.461 * 2))
    p.apply_preset("Arina · 300 kV · 21.4 mrad · 185 mm")
    assert (p.detector, p.detector_px, p.semiangle_mrad, p.detector_mrad_per_px) == ("Arina", 192, 21.4, pytest.approx(0.269))


def test_voltage_rescales_without_rebuilding(monkeypatch):
    pytest.importorskip("abtem")
    import quantem.widget.planptycho as module
    p = module.PlanPtycho(srtio3())
    potential = p._potential.copy()
    monkeypatch.setattr(module, "projected_potential", lambda *a, **k: pytest.fail("potential rebuilt for a voltage change"))
    p.voltage_kV = 200
    # the phase scales with the interaction constant; the object pixel (and so the blur) changes with the wavelength too
    expected = module.column_phase_rad_per_A(module.interaction_constant(200) * potential, p.cell_size_A, p.object_pixel_A)
    assert p.column_phase_rad_per_A == pytest.approx(expected, rel=1e-9)


def test_apply_thickness_and_view_depth_clamp():
    p = plan(thickness_nm=110)
    p.apply_thickness(40)
    assert (p.thickness_nm, p.focus_depth_nm, p.view_depth_nm, p.scan_size_px) == (40.0, 20.0, 20.0, 128)
    p.view_depth_nm = 35
    p.thickness_nm = 30
    assert p.view_depth_nm == 30


@pytest.mark.parametrize("kwargs, message", [({"zone_axis": (0, 0, 0)}, "zone_axis"), ({"zone_axis": (1.5, 0, 1)}, "zone_axis"), ({"thickness_nm": 0}, "thickness_nm"),
                                             ({"scan_step_A": -1}, "scan_step_A"), ({"detector_px": 0}, "detector_px"), ({"detector": "K3"}, "Unknown detector"),
                                             ({"preset": "nope"}, "Unknown preset"), ({"tilt_mrad": (1, 2, 3)}, "tilt_mrad")])
def test_invalid_settings_name_the_argument(kwargs, message):
    with pytest.raises(ValueError, match=message):
        plan(**kwargs)


def test_invalid_trait_values_are_refused():
    import traitlets
    p = plan()
    for name, value in (("detector_mrad_per_px", 0.0), ("detector", "K3"), ("zone_axis", [0, 0, 0])):
        with pytest.raises(traitlets.TraitError):
            setattr(p, name, value)


def test_missing_cif_and_materials_project_key_explain_the_fix(monkeypatch):
    pytest.importorskip("ase")
    from quantem.widget.planptycho import load_crystal
    with pytest.raises(ValueError, match="No such CIF file"):
        load_crystal("does_not_exist.cif")
    monkeypatch.delenv("MP_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MP_API_KEY"):
        load_crystal("mp-5229")


def test_materials_project_primitive_cells_become_conventional():
    pytest.importorskip("spglib")
    from ase.build import bulk
    from quantem.widget.planptycho import _conventional
    conventional = _conventional(bulk("Si"))                           # primitive fcc cell, 2 atoms
    assert len(conventional) == 8 and np.diag(conventional.cell.array) == pytest.approx([5.43] * 3, abs=0.01)


def test_collaborator_settings_as_reported():
    """A collaborator's acquisition: values passed as reported, no preset or camera length; C10 in the quantem sign."""
    p = plan(thickness_nm=40, voltage_kV=200, semiangle_mrad=24.5, detector_px=128, detector_mrad_per_px=0.9, scan_step_A=0.4, scan_size_px=256, c10_nm=-15)
    assert (p.detector, p.detector_px, p.detector_mrad_per_px, p.focus_depth_nm) == ("custom", 128, 0.9, 15.0)
    p.camera_length_mm = 185                                              # a custom camera keeps the reported sampling
    assert p.detector_mrad_per_px == 0.9
    with pytest.raises(ValueError, match="not both"):
        plan(c10_nm=-5, focus_depth_nm=5)
    with pytest.raises(ValueError, match="needs detector_px and detector_mrad_per_px"):
        plan(detector="custom", detector_px=128)
