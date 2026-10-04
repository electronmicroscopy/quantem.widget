"""Scientist workflow: full scan coverage, potential sampling, and honest convergence state."""

import numpy as np
import pytest
Atoms = pytest.importorskip("ase").Atoms
from quantem.widget import PlanPtycho


@pytest.fixture
def planner():
    pytest.importorskip("abtem")
    atoms = Atoms(
        "SrTiO3",
        scaled_positions=[
            (0, 0, 0),
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0),
            (0.5, 0, 0.5),
            (0, 0.5, 0.5),
        ],
        cell=[3.991, 3.991, 4.0352],
        pbc=True,
    )
    return PlanPtycho(
        atoms,
        thickness_nm=60,
        focus_depth_nm=20,
        scan_step_A=0.373,
        scan_size_px=128,
        detector_px=192,
        detector_mrad_per_px=0.5570968023269496,
    )


def test_simulation_plan_covers_full_scan_and_keeps_sampling_separate(planner):
    plan = planner.simulation_plan(repeats=(48, 48), pixels_per_cell=96)
    assert plan["scan_center_span_A"] == pytest.approx(47.371)
    assert plan["gpts"] == [4608, 4608]
    assert plan["extent_A"] == pytest.approx([191.568] * 2)
    assert plan["potential_sampling_A"] == pytest.approx([0.0415729166667] * 2)
    assert plan["z_repeats_to_cover"] == 149
    assert plan["geometric_fit"]
    assert not plan["boundary_convergence_verified"]
    assert not np.isclose(plan["potential_sampling_A"][0], planner.object_pixel_A)
    assert not planner.simulation_plan(repeats=(12, 12))["geometric_fit"]


def test_guard_and_tilt_change_coverage(planner):
    initial = planner.simulation_plan(repeats=(24, 24))
    planner.tilt_mrad = [20.0, 0.0]
    changed = planner.simulation_plan(repeats=(24, 24))
    assert changed["geometric_margin_A"][0] < initial["geometric_margin_A"][0]
    assert changed["geometric_margin_A"][1] == initial["geometric_margin_A"][1]
    assert not planner.simulation_plan(repeats=(24, 24), guard_A=100)["geometric_fit"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"repeats": (0, 2)},
        {"repeats": (2.5, 2)},
        {"pixels_per_cell": float("nan")},
        {"guard_A": -1},
    ],
)
def test_invalid_grid_fails_with_corrective_message(planner, kwargs):
    with pytest.raises(ValueError):
        planner.simulation_plan(**kwargs)


def test_live_simulation_settings_validate_and_drive_default_plan(planner):
    import traitlets

    planner.simulation_repeats = [48, 40]
    planner.simulation_pixels_per_cell = 64
    planner.simulation_guard_A = 8
    plan = planner.simulation_plan()
    assert plan["repeats"] == [48, 40]
    assert plan["gpts"] == [3072, 2560]
    assert plan["requested_guard_A"] == 8
    planner.simulation_plan(repeats=(12, 12), pixels_per_cell=32)
    assert planner.simulation_repeats == [48, 40]
    assert planner.simulation_pixels_per_cell == 64
    for name, value in [
        ("simulation_repeats", [0, 4]),
        ("simulation_repeats", [True, 4]),
        ("simulation_pixels_per_cell", 0),
        ("simulation_guard_A", float("nan")),
        ("wave_window_factor", True),
    ]:
        with pytest.raises(traitlets.TraitError):
            setattr(planner, name, value)
