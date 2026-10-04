"""Regenerate js/planptycho/goldens.json: PlanPtycho geometry, grades, check text, thickness recommendations and detector
calibration for fixed inputs.

Python (``plan_geometry``, ``check_statuses``, ``check_rows``, ``recommended_settings``, ``detector_sampling_mrad``) and
TypeScript (``planGeometry``, ``checkStatuses``, ``checkRows``, ``recommendedSettings``, ``detectorSamplingMrad``) are both
tested against this file, so a formula or a sentence changed on one side fails the other side's test. Run after an
intended change:

    python scripts/planptycho_goldens.py
"""

import json
import pathlib

from quantem.widget.planptycho import (
    ARINA_MRAD_PER_PX, DETECTORS, THICKNESS_PRESETS_NM, _ARINA_MRAD_MM, check_rows, check_statuses, detector_sampling_mrad,
    plan_geometry, recommended_settings,
)

ARINA_110 = dict(voltage_kV=300, semiangle_mrad=30, focus_depth_nm=17, thickness_nm=110, detector_px=184,
                 detector_mrad_per_px=0.563, scan_step_A=0.495, scan_size_px=48, tilt_mrad=[0.0, 0.0], holz_repeat_A=3.905)
CASES = {
    "srtio3_110nm_arina": ARINA_110,
    "srtio3_110nm_tilted_4mrad": {**ARINA_110, "tilt_mrad": [3.2, -2.4]},
    "thin_focused_on_surface": {**ARINA_110, "thickness_nm": 5, "focus_depth_nm": 0, "scan_step_A": 0.6, "scan_size_px": 256},
    "wide_scan_finer_detector": {**ARINA_110, "thickness_nm": 30, "detector_mrad_per_px": 0.25, "scan_size_px": 256},
    "edge_of_disk_small_step": {**ARINA_110, "thickness_nm": 5, "focus_depth_nm": 0, "scan_step_A": 0.4, "detector_mrad_per_px": 0.35, "scan_size_px": 256},
    "detector_cuts_disk": {**ARINA_110, "detector_px": 96, "detector_mrad_per_px": 0.55},
    "holz_in_the_corners": {**ARINA_110, "detector_px": 192, "detector_mrad_per_px": 0.9, "holz_repeat_A": 3.905},
    "collaborator_custom_camera": dict(voltage_kV=200, semiangle_mrad=24.5, focus_depth_nm=15, thickness_nm=40, detector_px=128,
                                       detector_mrad_per_px=0.9, scan_step_A=0.4, scan_size_px=256, tilt_mrad=[0.0, 0.0], holz_repeat_A=3.905),
    "cryo_200nm_small_semiangle": dict(voltage_kV=300, semiangle_mrad=4.0, focus_depth_nm=100, thickness_nm=200, detector_px=192,
                                       detector_mrad_per_px=0.0994, scan_step_A=2.0, scan_size_px=256, tilt_mrad=[0.0, 0.0], holz_repeat_A=None),
    "focus_below_exit": {**ARINA_110, "thickness_nm": 30, "focus_depth_nm": 40},
    "low_voltage_focus_above": dict(voltage_kV=80, semiangle_mrad=25, focus_depth_nm=-5, thickness_nm=8, detector_px=128,
                                    detector_mrad_per_px=0.8, scan_step_A=0.8, scan_size_px=64, tilt_mrad=[0.0, 1.0], holz_repeat_A=None),
}
COLUMN_PHASE = 0.1291


if __name__ == "__main__":
    cases = {}
    for name, settings in CASES.items():
        geometry = plan_geometry(**settings)
        cases[name] = {"settings": settings, "geometry": geometry, "statuses": check_statuses(geometry),
                       "rows": check_rows(geometry, detector_px=settings["detector_px"], scan_step_A=settings["scan_step_A"], column_phase_rad_per_A=COLUMN_PHASE)}
    out = {
        "cases": cases, "column_phase_rad_per_A": COLUMN_PHASE,
        "recommended": [{"thickness_nm": t, "semiangle_mrad": a, "voltage_kV": v, "scan_step_A": 0.5,
                         "expected": recommended_settings(t, voltage_kV=v, semiangle_mrad=a, scan_step_A=0.5)}
                        for t in (*THICKNESS_PRESETS_NM, 110, 250) for a, v in ((30, 300), (21.4, 300), (20, 80))],
        "detector_sampling": [{"detector": d, "camera_length_mm": cl, "expected": detector_sampling_mrad(d, cl)} for d in DETECTORS for cl in (91, 115, 150, 185)],
        "presets": {"detectors": DETECTORS, "arina_mrad_per_px": {f"{k:g}": v for k, v in ARINA_MRAD_PER_PX.items()}, "arina_mrad_mm": _ARINA_MRAD_MM},
    }
    path = pathlib.Path(__file__).resolve().parents[1] / "js" / "planptycho" / "goldens.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {len(cases)} cases, {len(out['recommended'])} recommendations, {len(out['detector_sampling'])} samplings to {path}")
