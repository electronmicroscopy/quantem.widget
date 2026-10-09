"""Regenerate js/show4dstem/maskGoldens.json: Show4DSTEM detector masks for fixed ROIs.

The Python masks (``show4dstem.detector.roi_mask``) are the reference for what a
detector ROI selects. tests/show4dstem checks that this file still equals them,
and js/show4dstem/detectorInteraction.test.ts checks that the overlay the
browser draws covers exactly these pixels, so a half-pixel shift on either side
fails a test. Run after an intended change:

    python scripts/show4dstem_mask_goldens.py
"""

import json
import pathlib

import numpy as np

from quantem.widget.show4dstem.detector import roi_mask

DETECTOR_SHAPE = (24, 24)  # a bin-8 Arina pattern, as in the gold_128 and gold_512 tutorial data
CASES = {
    "bin8 bright-field disk": dict(mode="circle", center=(11.36, 11.5), radius=6.077, radius_inner=0.0),
    "disk on a pixel centre": dict(mode="circle", center=(12.0, 12.0), radius=4.6, radius_inner=0.0),
    "disk at quarter pixels": dict(mode="circle", center=(5.25, 9.75), radius=3.3, radius_inner=0.0),
    "annular dark field": dict(mode="annular", center=(11.36, 11.5), radius=10.9, radius_inner=6.077),
    "square": dict(mode="square", center=(8.4, 14.6), radius=2.7, radius_inner=0.0),
}


def masks() -> dict:
    """``{name: {mode, center, radius, radius_inner, selected}}``, ``selected`` the row-major flat indices."""
    cases = {}
    for name, roi in CASES.items():
        mask = roi_mask(roi["mode"], DETECTOR_SHAPE, center_row=roi["center"][0], center_col=roi["center"][1],
                        radius=roi["radius"], radius_inner=roi["radius_inner"], width=0.0, height=0.0)
        cases[name] = {**roi, "center": list(roi["center"]), "selected": np.flatnonzero(mask).tolist()}
    return {"detector_shape": list(DETECTOR_SHAPE), "cases": cases}


if __name__ == "__main__":
    path = pathlib.Path(__file__).resolve().parents[1] / "js" / "show4dstem" / "maskGoldens.json"
    path.write_text(json.dumps(masks(), indent=1) + "\n")
    print(f"wrote {path}")
