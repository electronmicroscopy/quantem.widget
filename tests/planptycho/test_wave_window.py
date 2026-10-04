"""Changing wave support must preserve the measured acquisition."""

import pytest
from quantem.widget.planptycho import plan_geometry

SETTINGS = dict(
    voltage_kV=300,
    semiangle_mrad=30,
    focus_depth_nm=-10,
    thickness_nm=60,
    detector_px=192,
    detector_mrad_per_px=0.5570968023269496,
    scan_step_A=0.99775,
    scan_size_px=64,
)


def test_wide_wave_preserves_pixels_and_detector_angles():
    native = plan_geometry(**SETTINGS)
    wide = plan_geometry(**SETTINGS, wave_window_factor=2)
    assert wide["window_A"] == pytest.approx(2 * native["window_A"])
    for key in ["pixel_A", "theta_max_mrad", "reach", "widest_A", "scan_A"]:
        assert wide[key] == native[key]
    assert native["window_A"] < native["widest_A"] < wide["window_A"]


@pytest.mark.parametrize("factor", [0, 3, 1.5, True])
def test_invalid_support_factor(factor):
    with pytest.raises(ValueError, match="wave_window_factor"):
        plan_geometry(**SETTINGS, wave_window_factor=factor)


def test_widget_switch_keeps_cell_and_acquisition():
    pytest.importorskip("abtem")
    from ase import Atoms
    from quantem.widget import PlanPtycho

    atoms = Atoms(
        "BaTiO3",
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
    p = PlanPtycho(atoms, **SETTINGS)
    initial = (p.object_pixel_A, p.cell_bytes, p.detector_mrad_per_px, p.detector_px)
    p.wave_window_factor = 2
    assert p.wave_pixels == 384
    assert (
        p.object_pixel_A,
        p.cell_bytes,
        p.detector_mrad_per_px,
        p.detector_px,
    ) == initial
    assert p.geometry()["window_A"] == pytest.approx(p.window_A)
