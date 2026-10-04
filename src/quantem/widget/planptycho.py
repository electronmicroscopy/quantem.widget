"""planptycho: check multislice ptychography settings against a known crystal before the experiment.

A crystal (CIF path, ase ``Atoms`` or Materials Project id) is turned so a zone axis runs along the beam and its
projected potential is computed once (abTEM, Lobato parametrisation, static lattice). Everything that depends only on
the microscope and scan settings is geometry: beam width through the thickness, the model window the detector sampling
gives, probe overlap, detector reach, column lean from a sample tilt, where the focus sits. The
browser recomputes it on every slider move; :func:`plan_geometry`, :func:`check_statuses`, :func:`check_rows` and
:func:`recommended_settings` are the same arithmetic in Python, and both sides are pinned to ``js/planptycho/goldens.json``.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import pathlib
import urllib.request
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Sequence

import anywidget
import numpy as np
import traitlets

LOW_INDEX_ZONES = ((0, 0, 1), (0, 1, 1), (1, 1, 1), (0, 1, 2), (1, 1, 2))
_THERMAL_A = 0.08                      # rms thermal displacement: softens the static projection and damps the Bragg reflections
_CELL_SAMPLING_A = 0.05                # projected-potential grid; 0.1 A misses high-angle scattering in thick crystals
_MAX_G_INV_A = 4.0                     # Bragg reflections listed out to 4 1/A (79 mrad at 300 kV)
_DISPLAY_BLUR_A = 0.15                 # the static projection has 0.05 A peaks; drawn at ~0.3 A per screen pixel they alias

# Detector angular sampling at the native pixel count: measured Arina calibration at 300 kV (median over reconstructions,
# semiangle / fitted bright-field disk radius). The three camera lengths follow 1 / camera length within 6 %.
ARINA_MRAD_PER_PX = {91.0: 0.554, 115.0: 0.461, 185.0: 0.269}
_ARINA_MRAD_MM = 0.554 * 91.0
DETECTORS = {
    "Arina": {"pixels": 192, "pixel_pitch_um": 100.0, "note": "measured Arina calibration"},
    "EMPAD": {"pixels": 128, "pixel_pitch_um": 150.0, "note": "scaled from the Arina calibration by pixel pitch (150 / 100 um), not measured"},
}
# Microscope settings in routine use, named by the settings only.
MICROSCOPE_PRESETS = {
    "Arina · 300 kV · 30 mrad · 91 mm": {"voltage_kV": 300.0, "semiangle_mrad": 30.0, "detector": "Arina", "camera_length_mm": 91.0},
    "Arina · 300 kV · 25 mrad · 115 mm": {"voltage_kV": 300.0, "semiangle_mrad": 25.0, "detector": "Arina", "camera_length_mm": 115.0},
    "Arina · 300 kV · 21.4 mrad · 185 mm": {"voltage_kV": 300.0, "semiangle_mrad": 21.4, "detector": "Arina", "camera_length_mm": 185.0},
    "EMPAD · 300 kV · 30 mrad · 91 mm": {"voltage_kV": 300.0, "semiangle_mrad": 30.0, "detector": "EMPAD", "camera_length_mm": 91.0},
}
DEFAULT_PRESET = "Arina · 300 kV · 30 mrad · 91 mm"
THICKNESS_PRESETS_NM = (20, 30, 40, 50, 60, 70, 100, 150, 200)          # to 200 nm: cryo and biological sections


# --- arithmetic (mirrored in js/planptycho/geometry.ts) -------------------------------------------------------------

def wavelength_A(voltage_kV: float) -> float:
    """Relativistic electron wavelength in A: 12.26434 / sqrt(V (1 + 0.97848e-6 V)), V in volts."""
    volts = voltage_kV * 1e3
    return 12.26434 / math.sqrt(volts * (1.0 + 0.97848e-6 * volts))


def detector_sampling_mrad(detector: str, camera_length_mm: float) -> float:
    """Angular sampling (mrad per native pixel) of ``detector`` at ``camera_length_mm``: the measured Arina value at a
    calibrated camera length, else 50.4 mrad mm / camera length; other detectors scale by pixel pitch."""
    if detector not in DETECTORS:
        raise ValueError(f"Unknown detector {detector!r}; choose one of {sorted(DETECTORS)} or pass detector_px and detector_mrad_per_px.")
    arina = ARINA_MRAD_PER_PX.get(float(camera_length_mm), _ARINA_MRAD_MM / camera_length_mm)
    return arina * DETECTORS[detector]["pixel_pitch_um"] / DETECTORS["Arina"]["pixel_pitch_um"]


def plan_geometry(
    *,
    voltage_kV: float,
    semiangle_mrad: float,
    focus_depth_nm: float,
    thickness_nm: float,
    detector_px: int,
    detector_mrad_per_px: float,
    scan_step_A: float,
    scan_size_px: int,
    tilt_mrad: Sequence[float] = (0.0, 0.0),
    holz_repeat_A: float | None = None,
    wave_window_factor: int = 1,
) -> dict:
    """The numbers every check is built from (lengths in A, angles in mrad).

    - model window ``W = factor * lambda / dtheta``, object pixel ``lambda / (N dtheta)``, detector reach ``theta_max = N dtheta / 2`` (edge)
    - beam diameter at depth z: ``D(z) = 2 alpha |z - f| + 1.22 lambda / alpha`` (f = focus depth below the entrance)
    - depth of field ``2 lambda / alpha^2``; probe overlap ``1 - step / D(0)``
    - column lean across the thickness ``t tan|theta|``; first HOLZ ring ``sqrt(2 lambda / H)`` (H = lattice period along
      the beam)
    """
    if isinstance(wave_window_factor, bool) or wave_window_factor not in (1, 2):
        raise ValueError(
            "wave_window_factor must be 1 (native) or 2 (double width); measured detector sampling stays fixed."
        )
    lam = wavelength_A(voltage_kV)
    alpha = semiangle_mrad * 1e-3
    dtheta = detector_mrad_per_px * 1e-3
    thickness, focus = thickness_nm * 10.0, focus_depth_nm * 10.0
    airy = 1.22 * lam / alpha
    entrance, exit_ = (
        2 * alpha * abs(focus) + airy,
        2 * alpha * abs(thickness - focus) + airy,
    )
    native_window = lam / dtheta
    window = native_window * wave_window_factor
    tilt = math.hypot(*tilt_mrad)
    return {
        "wavelength_A": lam,
        "window_A": window,
        "pixel_A": native_window / detector_px,
        "theta_max_mrad": detector_px * detector_mrad_per_px / 2.0,
        "airy_A": airy,
        "entrance_A": entrance,
        "exit_A": exit_,
        "widest_A": max(entrance, exit_),
        "depth_of_field_A": 2.0 * lam / alpha**2,
        "scan_A": scan_size_px * scan_step_A,
        "overlap": 1.0 - scan_step_A / entrance,
        "reach": detector_px * detector_mrad_per_px / 2.0 / semiangle_mrad,
        "tilt_mrad": tilt,
        "lean_A": thickness * math.tan(tilt * 1e-3),
        "thickness_A": thickness,
        "focus_A": focus,
        "holz_mrad": (
            1e3 * math.sqrt(2.0 * lam / holz_repeat_A) if holz_repeat_A else None
        ),
    }


def check_statuses(geometry: dict) -> dict:
    """``pass`` / ``caution`` / ``fail`` for each graded check. Window, margin and lean only caution: simulated SrTiO3
    reconstructed past all three (130 nm: 68 A beam in a 35 A window; 24 A scans at 110 nm, weaker only at the edge;
    4 mrad tilt at 110 nm with the tilt measured and held). A focus outside the specimen only cautions: it widens the beam at
    one surface; the simulated SrTiO3 runs focused inside (17 nm into 90-130 nm)."""
    g = geometry
    return {"window": "pass" if g["widest_A"] <= g["window_A"] else "caution",
            "margin": "pass" if g["scan_A"] >= 2 * g["widest_A"] else "caution",
            "overlap": "pass" if g["overlap"] >= 0.6 else "caution" if g["overlap"] >= 0.3 else "fail",
            "reach": "pass" if g["reach"] >= 1.5 else "caution" if g["reach"] >= 1.0 else "fail",
            "lean": "pass" if g["lean_A"] <= g["pixel_A"] else "caution",
            "split": "pass" if 0.0 <= g["focus_A"] <= g["thickness_A"] else "caution"}


def _fixed(value: float, digits: int) -> str:
    """Fixed-point text that matches JavaScript ``toFixed`` for the values shown here."""
    return f"{value:.{digits}f}"


def _split_text(g: dict) -> str:
    """Where the specimen lies relative to the focus."""
    above, below = g["focus_A"] / 10.0, (g["thickness_A"] - g["focus_A"]) / 10.0
    if above < 0:
        return f"specimen entirely below the focus ({_fixed(-above, 1)} nm above the entrance)"
    if below < 0:
        return f"specimen entirely above the focus ({_fixed(-below, 1)} nm below the exit)"
    return f"{_fixed(above, 1)} nm above, {_fixed(below, 1)} nm below the focus"


def check_rows(geometry: dict, *, detector_px: int, scan_step_A: float, column_phase_rad_per_A: float = 0.0) -> list[dict]:
    """Every check as ``{id, label, status, value, rule, note}``, the text the widget shows (``checkRows`` in
    geometry.ts renders the same strings; both are pinned to the goldens). ``note`` is empty when a check passes."""
    g = geometry; status = check_statuses(g); radius = g["widest_A"] / 2.0; f = _fixed
    wave_pixels = round(g['window_A'] / g['pixel_A'])
    wide = wave_pixels != detector_px
    rows = [
        {"id": "window", "label": "Beam fits the virtual window", "value": f"{f(g['widest_A'], 0)} Å beam in a {f(g['window_A'], 0)} Å window",
         "rule": "widest beam (entrance or exit) ≤ window factor × wavelength / detector pixel angle" if wide else "widest beam (entrance or exit) ≤ wavelength / detector pixel angle",
         "note": "The beam wraps around the virtual window. Simulated SrTiO3 still reconstructed with it (130 nm: 68 Å beam in a 35 Å "
                 "window, picture 0.80). A longer camera length widens the window but lowers the detector reach; a 2x virtual window "
                 "in the reconstruction removes the wrap without changing the acquisition."},
        {"id": "margin", "label": "Scan margin for the spread beam", "value": f"{f(g['scan_A'], 0)} Å scan, beam radius {f(radius, 0)} Å",
         "rule": "scan side ≥ 4 × widest beam radius",
         "note": f"The outer ~{f(radius, 0)} Å of the scan is lit from one side only deep in the sample (110 nm SrTiO3, 24 Å scan: "
                 f"centre 0.79, edge 0.60). Scan {f(4 * radius, 0)} Å or more, or report only the interior."},
        {"id": "overlap", "label": "Probe overlap at the entrance", "value": f"{f(100 * g['overlap'], 0)} % (step {f(scan_step_A, 3)} Å, beam {f(g['entrance_A'], 1)} Å)",
         "rule": "1 − step / entrance beam diameter ≥ 60 %",
         "note": "Neighbouring probe positions barely share illuminated area; a smaller step or a focus deeper below the entrance increases the overlap."},
        {"id": "reach", "label": "Detector reach", "value": f"{f(g['theta_max_mrad'], 0)} mrad = {f(g['reach'], 1)} × semiangle",
         "rule": "detector edge ≥ 1.5 × semiangle",
         "note": "The detector cuts into the bright-field disk." if status["reach"] == "fail" else "Little dark field is recorded; scattering to high angles carries the fine detail."},
        {"id": "lean", "label": "Column lean across the thickness", "value": f"{f(g['lean_A'], 1)} Å at {f(g['tilt_mrad'], 1)} mrad tilt",
         "rule": "thickness × tan(tilt) ≤ object pixel",
         "note": "The columns lean across more than an object pixel: measure the tilt and hold it in the propagator. 110 nm SrTiO3 at "
                 "4 mrad reached 0.74 with the measured tilt held, 0.81 when the probe was also frozen for the first 10 iterations."},
        {"id": "split", "label": "Focus splits the specimen", "value": _split_text(g),
         "rule": "0 ≤ focus depth ≤ thickness",
         "note": "The whole thickness is on one side of the focus, so the beam is widest at one surface. Thick-sample runs focused "
                 "inside the specimen (110 nm SrTiO3: 17 nm below the entrance)."},
    ]
    for row in rows:
        row["status"] = status[row["id"]]
        if row["status"] == "pass":
            row["note"] = ""
    rows += [
        {"id": "pixel", "label": "Object pixel", "status": "info", "value": f"{f(g['pixel_A'], 3)} Å ({wave_pixels} px over {f(g['window_A'], 1)} Å)", "rule": "model window / wave pixels; measured detector unchanged" if wide else "window / detector pixels", "note": ""},
        {"id": "depth", "label": "Depth of field", "status": "info", "value": f"{f(g['depth_of_field_A'] / 10, 1)} nm",
         "rule": "2 × wavelength / semiangle²: the depth the probe resolves", "note": ""},
        {"id": "beam", "label": "Beam diameter", "status": "info", "value": f"entrance {f(g['entrance_A'], 1)} Å, focus {f(g['airy_A'], 1)} Å, exit {f(g['exit_A'], 1)} Å",
         "rule": "2 × semiangle × |depth − focus| + 1.22 × wavelength / semiangle", "note": ""},
    ]
    if column_phase_rad_per_A > 0:
        rows.append({"id": "phase", "label": "Column phase per nm", "status": "info", "value": f"{f(column_phase_rad_per_A * 10, 2)} rad (strongest column)",
                     "rule": "interaction constant × projected potential of the strongest column per nm of thickness", "note": ""})
    if g["holz_mrad"] is not None:
        edge = g["theta_max_mrad"]
        where = "on" if g["holz_mrad"] <= edge else "in the corners of" if g["holz_mrad"] <= math.sqrt(2) * edge else "beyond"
        rows.append({"id": "holz", "label": "First HOLZ ring", "status": "info", "value": f"{f(g['holz_mrad'], 0)} mrad, {where} the detector",
                     "rule": "√(2 × wavelength / lattice period along the beam)", "note": ""})
    return rows


def recommended_settings(thickness_nm: float, *, voltage_kV: float, semiangle_mrad: float, scan_step_A: float) -> dict:
    """Starting settings for a thickness, derived from the checks: focus at mid-thickness (the narrowest widest beam),
    and a scan wide enough that the margin check passes (at least 128 positions, multiples of 16). Reconstructions tested
    the checks only on 90-130 nm SrTiO3 (30 mrad, focus 17 nm)."""
    lam = wavelength_A(voltage_kV); alpha = semiangle_mrad * 1e-3
    focus_nm = thickness_nm / 2.0
    widest = 2 * alpha * thickness_nm * 10.0 / 2.0 + 1.22 * lam / alpha
    scan_px = max(128, 16 * math.ceil(2.0 * widest / scan_step_A / 16.0))
    return {"thickness_nm": float(thickness_nm), "focus_depth_nm": focus_nm, "scan_size_px": scan_px}


# --- crystal ---------------------------------------------------------------------------------------------------------

def _materials_project_key() -> str:
    key = os.environ.get("MP_API_KEY", "").strip()
    if not key:
        raise ValueError("A Materials Project id needs an API key in the MP_API_KEY environment variable "
                         "(https://next-gen.materialsproject.org/api). A CIF path or an ase Atoms works without one.")
    return key


def _conventional(atoms):
    """The conventional standard cell (spglib), so zone indices [uvw] mean what crystallographers mean by them;
    Materials Project serves primitive cells."""
    try:
        import spglib
    except ImportError as error:
        raise ValueError("Materials Project structures are primitive cells; install spglib (pip install spglib) to turn "
                         "them into the conventional cell, or pass a CIF of the conventional cell.") from error
    from ase import Atoms
    cell, scaled, numbers = spglib.standardize_cell((atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers), to_primitive=False, no_idealize=False)
    return Atoms(numbers=numbers, scaled_positions=scaled, cell=cell, pbc=True)


def load_crystal(structure) -> Any:
    """ase ``Atoms`` from a CIF path, an ``Atoms`` object, or a Materials Project id such as ``"mp-5229"`` (converted to
    the conventional cell)."""
    from ase import Atoms
    from ase.io import read
    if isinstance(structure, Atoms):
        return structure.copy()
    text = str(structure)
    if text.startswith("mp-"):
        request = urllib.request.Request(f"https://api.materialsproject.org/materials/core/?material_ids={text}&_fields=structure",
                                         headers={"X-API-KEY": _materials_project_key(), "User-Agent": "quantem-widget"})
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read())["data"]
        if not data:
            raise ValueError(f"Materials Project has no structure for {text!r}.")
        sites = data[0]["structure"]["sites"]
        if any(len(site["species"]) != 1 or abs(site["species"][0].get("occu", 1.0) - 1.0) > 1e-6 for site in sites):
            raise ValueError(f"{text} has partially occupied sites; pass a CIF with an ordered structure instead.")
        atoms = Atoms(symbols=[site["species"][0]["element"] for site in sites], scaled_positions=[site["abc"] for site in sites],
                      cell=np.asarray(data[0]["structure"]["lattice"]["matrix"], float), pbc=True)
        return _conventional(atoms)
    path = pathlib.Path(text)
    if not path.exists():
        raise ValueError(f"No such CIF file: {text!r}. Pass a CIF path, an ase Atoms, or a Materials Project id ('mp-...').")
    return read(path)


def oriented_cell(atoms, zone_axis: Sequence[int]):
    """The crystal turned so the zone axis [uvw] runs along the beam (z) and its shortest in-plane lattice vector along x,
    as the smallest orthogonal cell that repeats along z. With that in-plane alignment the orthogonal cell is exact; abTEM
    is told not to strain the lattice to fit one (without the alignment, [111] and [112] came out sheared by 2-4 %)."""
    from abtem import orthogonalize_cell
    turned = atoms.copy(); turned.rotate(np.asarray(zone_axis, float) @ turned.cell.array, "z", rotate_cell=True)
    cell = turned.cell.array; in_plane = None
    for n in itertools.product(range(-3, 4), repeat=3):
        vector = np.asarray(n, float) @ cell
        if any(n) and abs(vector[2]) < 1e-6 * np.linalg.norm(vector) and (in_plane is None or np.linalg.norm(vector) < np.linalg.norm(in_plane) - 1e-9):
            in_plane = vector
    turned.rotate(-math.degrees(math.atan2(in_plane[1], in_plane[0])), "z", rotate_cell=True)
    # abTEM's allow_transform=False does not raise (its check is commented out), so read the transform and refuse any strain
    oriented, transform = orthogonalize_cell(turned, max_repetitions=10, return_transform_matrix=True)
    if not np.allclose(transform, np.eye(3), atol=1e-3):
        raise ValueError(f"No orthogonal cell of this crystal along {list(zone_axis)} within 10 repetitions without straining the "
                         f"lattice; choose a lower-index zone axis.")
    return oriented


def holz_repeat_A(cell) -> float:
    """Lattice period along the beam that sets the first HOLZ ring: the orthogonal cell's z repeat divided by the index
    of the first reciprocal-lattice layer with a non-zero structure factor. Centring translations empty the first layers
    (fcc Au [011]: repeat 5.77 A, first ring from the second layer, period 2.88 A)."""
    scaled = cell.get_scaled_positions(); numbers = cell.numbers.astype(float); total = numbers.sum()
    hk = np.array([(h, k) for h in range(-4, 5) for k in range(-4, 5)], float)
    for layer in range(1, 9):
        phases = 2j * np.pi * (hk @ scaled[:, :2].T + layer * scaled[:, 2][None, :])
        if np.abs(np.exp(phases) @ numbers).max() > 1e-6 * total:
            return float(cell.cell[2, 2]) / layer
    return float(cell.cell[2, 2])


def projected_potential(cell, sampling_A: float = _CELL_SAMPLING_A) -> np.ndarray:
    """Projected potential of one repeat of the oriented cell (V A; abTEM, Lobato parametrisation, static lattice). It does
    not depend on the voltage; multiply by the interaction constant for the phase. Axis 0 runs along cell vector a."""
    from abtem import Potential
    potential = Potential(cell, sampling=sampling_A, parametrization="lobato", projection="infinite")
    return np.asarray(potential.build().project().array, dtype=np.float32)


def interaction_constant(voltage_kV: float) -> float:
    """Phase per unit projected potential (rad / V A) of a fast electron (abTEM ``energy2sigma``)."""
    from abtem.core.energy import energy2sigma
    return float(energy2sigma(voltage_kV * 1e3))


def bragg_reflections(phase: np.ndarray, cell_size_A: Sequence[float], max_g_inv_A: float = _MAX_G_INV_A) -> list[list[float]]:
    """Zero-order Laue zone reflections ``[g_row, g_col, relative amplitude]`` (1/A) of a periodic projected cell, from its
    Fourier coefficients damped by thermal motion (Debye-Waller ``exp(-2 pi^2 u^2 g^2)``, u = 0.08 A); the strongest
    non-zero reflection has amplitude 1."""
    coefficients = np.abs(np.fft.fft2(phase)) / phase.size
    rows, cols = phase.shape; length_row, length_col = cell_size_A
    h_max, k_max = min(int(max_g_inv_A * length_row), rows // 2 - 1), min(int(max_g_inv_A * length_col), cols // 2 - 1)
    out = []
    for h in range(-h_max, h_max + 1):
        for k in range(-k_max, k_max + 1):
            g_row, g_col = h / length_row, k / length_col; g = math.hypot(g_row, g_col)
            if (h or k) and g <= max_g_inv_A:
                out.append([g_row, g_col, float(coefficients[h % rows, k % cols]) * math.exp(-2 * math.pi**2 * _THERMAL_A**2 * g * g)])
    strongest = max((amplitude for *_, amplitude in out), default=1.0) or 1.0
    return [[g_row, g_col, amplitude / strongest] for g_row, g_col, amplitude in out if amplitude / strongest > 1e-3]


def column_phase_rad_per_A(phase: np.ndarray, cell_size_A: Sequence[float], object_pixel_A: float) -> float:
    """Phase of the strongest column per A of thickness as a reconstruction at ``object_pixel_A`` sees it: the static
    projection of one repeat blurred by thermal motion and half an object pixel, divided by the repeat."""
    from scipy.ndimage import gaussian_filter
    sampling = cell_size_A[0] / phase.shape[0]
    return float(gaussian_filter(phase, math.hypot(_THERMAL_A, object_pixel_A / 2.0) / sampling, mode="wrap").max() / cell_size_A[2])


# --- widget ----------------------------------------------------------------------------------------------------------

def _positive(name: str, value) -> None:
    if value is not None and not value > 0:
        raise ValueError(f"{name} must be positive, got {value!r}.")


class PlanPtycho(anywidget.AnyWidget):
    """Check multislice ptychography settings against a known crystal before the experiment.

    Parameters
    ----------
    structure : str or ase.Atoms
        CIF path, ``ase.Atoms``, or Materials Project id (``"mp-5229"``, needs ``MP_API_KEY`` and spglib).
    zone_axis : sequence of 3 int, default (0, 0, 1)
        Crystal direction [uvw] along the beam.
    thickness_nm : float, default 50
        Specimen thickness.
    preset : str, optional
        Microscope settings in routine use (``MICROSCOPE_PRESETS``); default ``"Arina · 300 kV · 30 mrad · 91 mm"``.
        ``voltage_kV``, ``semiangle_mrad``, ``detector`` and ``camera_length_mm`` override single values of it.
    voltage_kV, semiangle_mrad : float, optional
        Accelerating voltage and probe convergence semiangle.
    detector : {"Arina", "EMPAD", "custom"}, optional
        Camera; sets the pixel count and, with ``camera_length_mm``, the angular sampling. The Arina sampling is measured
        at 91, 115 and 185 mm (0.554, 0.461, 0.269 mrad per pixel) and scales as 1 / camera length otherwise; the EMPAD
        scales from it by pixel pitch. ``"custom"`` takes ``detector_px`` and ``detector_mrad_per_px`` as reported (the
        default when ``detector_mrad_per_px`` is given without a camera: settings from a collaborator).
    camera_length_mm : float, optional
        Nominal camera length.
    c10_nm : float, optional
        Nominal defocus C10 (quantem sign: negative focuses below the entrance surface, into the specimen), as a
        reconstruction or an acquisition log reports it. Same as ``focus_depth_nm = -c10_nm``; give one of the two.
    focus_depth_nm : float, optional
        Where the probe is focused, measured into the specimen from the entrance surface (negative: above it). Default:
        mid-thickness, which makes the widest beam as narrow as it can be.
    tilt_mrad : (float, float), default (0, 0)
        Specimen tilt off the zone axis, (row, col).
    detector_px : int, optional
        Detector pixels per side used in the reconstruction; overrides the camera's native count (96 after 2x binning of
        the Arina: same total angle, so the sampling per pixel doubles unless ``detector_mrad_per_px`` is given).
    detector_mrad_per_px : float, optional
        Angular sampling per pixel; overrides the calibration.
    wave_window_factor : {1, 2}, default 1
        Wave support relative to the measured detector grid. Factor 2 doubles
        the physical width and model pixels without changing object sampling
        or measured detector angles. Requires an intensity-integrating backend.
    scan_step_A : float, default 0.5
        Probe step.
    scan_size_px : int, default 128
        Scan positions per side.
    title : str, optional
        Heading; defaults to the formula and zone axis.

    Notes
    -----
    Needs ``ase`` and ``abtem`` (``pip install "quantem.widget[crystal]"``). :meth:`report` returns the checks as a
    DataFrame. The widget has no ``save_state`` or HTML export: the crystal is rebuilt from ``structure``.
    """

    _esm = pathlib.Path(__file__).parent / "static" / "planptycho.js"

    widget_version = traitlets.Unicode("unknown").tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    formula = traitlets.Unicode("").tag(sync=True)
    zone_axis = traitlets.List(traitlets.Int(), default_value=[0, 0, 1]).tag(sync=True)
    zone_axes = traitlets.List(traitlets.List(traitlets.Int()), default_value=[list(z) for z in LOW_INDEX_ZONES]).tag(sync=True)
    thickness_nm = traitlets.Float(50.0).tag(sync=True)
    voltage_kV = traitlets.Float(300.0).tag(sync=True)
    semiangle_mrad = traitlets.Float(30.0).tag(sync=True)
    focus_depth_nm = traitlets.Float(25.0).tag(sync=True)
    tilt_mrad = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0]).tag(sync=True)
    detector = traitlets.Unicode("Arina").tag(sync=True)
    camera_length_mm = traitlets.Float(91.0).tag(sync=True)
    detector_px = traitlets.Int(192).tag(sync=True)
    wave_window_factor = traitlets.Int(1).tag(sync=True)
    detector_mrad_per_px = traitlets.Float(0.554).tag(sync=True)
    scan_step_A = traitlets.Float(0.5).tag(sync=True)
    scan_size_px = traitlets.Int(128).tag(sync=True)
    view_depth_nm = traitlets.Float(25.0).tag(sync=True)
    # computed from the crystal (Python -> browser)
    cell_bytes = traitlets.Bytes(b"").tag(sync=True)
    cell_shape = traitlets.List(traitlets.Int(), default_value=[0, 0]).tag(sync=True)
    cell_size_A = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0, 0.0]).tag(sync=True)
    holz_repeat_A = traitlets.Float(0.0).tag(sync=True)
    bragg_inv_A = traitlets.List(traitlets.List(traitlets.Float()), default_value=[]).tag(sync=True)
    column_phase_rad_per_A = traitlets.Float(0.0).tag(sync=True)
    simulation_repeats = traitlets.List(traitlets.Int(), default_value=[24, 24]).tag(sync=True)
    simulation_pixels_per_cell = traitlets.Int(96).tag(sync=True)
    simulation_guard_A = traitlets.Float(5.0).tag(sync=True)

    # tables the browser offers as menus (Python owns them)
    detector_presets = traitlets.Dict({}).tag(sync=True)
    microscope_presets = traitlets.Dict({}).tag(sync=True)
    thickness_presets = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)

    def __init__(self, structure, *, zone_axis: Sequence[int] = (0, 0, 1), thickness_nm: float = 50.0, preset: str | None = None,
                 voltage_kV: float | None = None, semiangle_mrad: float | None = None, detector: str | None = None,
                 camera_length_mm: float | None = None, c10_nm: float | None = None, focus_depth_nm: float | None = None,
                 tilt_mrad: Sequence[float] = (0.0, 0.0), detector_px: int | None = None, detector_mrad_per_px: float | None = None,
                 scan_step_A: float = 0.5, scan_size_px: int = 128, wave_window_factor: int = 1, title: str = "") -> None:
        super().__init__()
        self._ready = False
        if isinstance(wave_window_factor, bool) or wave_window_factor not in (1, 2):
            raise ValueError("wave_window_factor must be 1 or 2; keep the measured detector unchanged.")
        self.wave_window_factor = wave_window_factor
        preset = preset or DEFAULT_PRESET
        if preset not in MICROSCOPE_PRESETS:
            raise ValueError(f"Unknown preset {preset!r}; choose one of {list(MICROSCOPE_PRESETS)}.")
        base = MICROSCOPE_PRESETS[preset]
        if c10_nm is not None and focus_depth_nm is not None:
            raise ValueError("Give c10_nm or focus_depth_nm, not both (focus_depth_nm = -c10_nm).")
        if c10_nm is not None:
            focus_depth_nm = -float(c10_nm)
        if detector is None and detector_mrad_per_px is not None and camera_length_mm is None:
            detector = "custom"                                                         # reported values, no calibration needed
        if detector == "custom" and (detector_px is None or detector_mrad_per_px is None):
            raise ValueError('detector="custom" needs detector_px and detector_mrad_per_px as reported for the acquisition.')
        voltage_kV = base["voltage_kV"] if voltage_kV is None else voltage_kV
        semiangle_mrad = base["semiangle_mrad"] if semiangle_mrad is None else semiangle_mrad
        detector = base["detector"] if detector is None else detector
        camera_length_mm = base["camera_length_mm"] if camera_length_mm is None else camera_length_mm
        zone = self._validate_zone(zone_axis)
        for name, value in (("thickness_nm", thickness_nm), ("voltage_kV", voltage_kV), ("semiangle_mrad", semiangle_mrad),
                            ("camera_length_mm", camera_length_mm), ("detector_px", detector_px),
                            ("detector_mrad_per_px", detector_mrad_per_px), ("scan_step_A", scan_step_A), ("scan_size_px", scan_size_px)):
            _positive(name, value)
        if len(tuple(tilt_mrad)) != 2:
            raise ValueError(f"tilt_mrad must be (row, col) in mrad, got {tilt_mrad!r}.")
        sampling = detector_mrad_per_px if detector == "custom" else detector_sampling_mrad(detector, camera_length_mm)   # raises for an unknown detector
        start = recommended_settings(thickness_nm, voltage_kV=voltage_kV, semiangle_mrad=semiangle_mrad, scan_step_A=scan_step_A)
        self._atoms = load_crystal(structure)
        self._custom_title = bool(title)
        native_px = detector_px if detector == "custom" else DETECTORS[detector]["pixels"]
        with self.hold_sync():
            self.formula = self._atoms.get_chemical_formula(mode="metal", empirical=True)
            self.detector_presets = {"detectors": DETECTORS, "arina_mrad_per_px": {f"{k:g}": v for k, v in ARINA_MRAD_PER_PX.items()}, "arina_mrad_mm": _ARINA_MRAD_MM}
            self.microscope_presets = MICROSCOPE_PRESETS
            self.thickness_presets = [float(t) for t in THICKNESS_PRESETS_NM]
            self.zone_axes = [list(z) for z in LOW_INDEX_ZONES] + ([zone] if tuple(zone) not in LOW_INDEX_ZONES else [])
            self.thickness_nm, self.voltage_kV, self.semiangle_mrad = float(thickness_nm), float(voltage_kV), float(semiangle_mrad)
            self.focus_depth_nm = start["focus_depth_nm"] if focus_depth_nm is None else float(focus_depth_nm)
            self.tilt_mrad = [float(v) for v in tilt_mrad]
            self.detector, self.camera_length_mm = detector, float(camera_length_mm)
            self.detector_px = int(detector_px or native_px)
            self.detector_mrad_per_px = float(detector_mrad_per_px) if detector_mrad_per_px else sampling * native_px / self.detector_px
            self.scan_step_A, self.scan_size_px = float(scan_step_A), int(scan_size_px)
            self.view_depth_nm = float(thickness_nm) / 2.0
            self.zone_axis = zone
            self.title = title or self._auto_title()
            self._build_crystal()
        try:
            self.widget_version = version("quantem-widget")
        except PackageNotFoundError:
            pass
        self._ready = True

    @traitlets.validate("wave_window_factor")
    def _validate_wave_window(self, proposal):
        if isinstance(proposal["value"], bool) or proposal["value"] not in (1, 2):
            raise traitlets.TraitError("wave_window_factor must be 1 or 2.")
        return proposal["value"]

    @traitlets.validate("simulation_repeats")
    def _validate_simulation_repeats(self, proposal):
        value = proposal["value"]
        if len(value) != 2 or any(isinstance(v, bool) or v < 1 for v in value):
            raise traitlets.TraitError(
                "simulation_repeats must contain two positive integers in (row, col) order."
            )
        return value

    @traitlets.validate("simulation_pixels_per_cell")
    def _validate_simulation_pixels(self, proposal):
        value = proposal["value"]
        if isinstance(value, bool) or value < 1:
            raise traitlets.TraitError(
                "simulation_pixels_per_cell must be a positive integer; try 96."
            )
        return value

    @traitlets.validate("simulation_guard_A")
    def _validate_simulation_guard(self, proposal):
        value = proposal["value"]
        if not np.isfinite(value) or value < 0:
            raise traitlets.TraitError(
                "simulation_guard_A must be finite and nonnegative; enlarge the cell to add margin."
            )
        return value

    # --- validation
    @staticmethod
    def _validate_zone(value) -> list[int]:
        zone = list(value)
        if len(zone) != 3 or not all(float(v).is_integer() for v in zone) or not any(zone):
            raise ValueError(f"zone_axis must be three integers [u, v, w], not all zero; got {value!r}.")
        return [int(v) for v in zone]

    @traitlets.validate("thickness_nm", "voltage_kV", "semiangle_mrad", "camera_length_mm", "detector_px",
                        "detector_mrad_per_px", "scan_step_A", "scan_size_px")
    def _validate_positive(self, proposal):
        try:
            _positive(proposal["trait"].name, proposal["value"])
        except ValueError as error:
            raise traitlets.TraitError(str(error)) from error
        return proposal["value"]

    @traitlets.validate("detector")
    def _validate_detector(self, proposal):
        if proposal["value"] not in DETECTORS and proposal["value"] != "custom":
            raise traitlets.TraitError(f"Unknown detector {proposal['value']!r}; choose one of {sorted(DETECTORS)}.")
        return proposal["value"]

    @traitlets.validate("zone_axis")
    def _validate_zone_axis(self, proposal):
        try:
            return self._validate_zone(proposal["value"])
        except ValueError as error:
            raise traitlets.TraitError(str(error)) from error

    # --- crystal
    def _auto_title(self) -> str:
        return f"{self.formula} [{''.join(str(v) for v in self.zone_axis)}]"

    def _build_crystal(self) -> None:
        cell = oriented_cell(self._atoms, self.zone_axis)
        self._potential = projected_potential(cell)
        self._cell_size = [float(cell.cell[0, 0]), float(cell.cell[1, 1]), float(cell.cell[2, 2])]
        self._holz_repeat = holz_repeat_A(cell)
        self._send_crystal()

    def _send_crystal(self) -> None:
        """Phase of one repeat at the current voltage, the Bragg list and the column phase, in one message."""
        from scipy.ndimage import gaussian_filter
        phase = interaction_constant(self.voltage_kV) * self._potential; self._phase = phase
        sampling = self._cell_size[0] / phase.shape[0]
        with self.hold_sync():
            self.cell_shape = [int(phase.shape[0]), int(phase.shape[1])]
            self.cell_size_A = self._cell_size
            self.holz_repeat_A = self._holz_repeat
            self.cell_bytes = gaussian_filter(phase, math.hypot(_THERMAL_A, _DISPLAY_BLUR_A) / sampling, mode="wrap").astype(np.float32).tobytes()   # display copy
            self.bragg_inv_A = bragg_reflections(phase, self._cell_size[:2])
            self._update_column_phase()

    def _update_column_phase(self) -> None:
        self.column_phase_rad_per_A = column_phase_rad_per_A(self._phase, self._cell_size, self.object_pixel_A)

    @traitlets.observe("zone_axis")
    def _on_zone(self, change) -> None:
        if self._ready and change["old"] != change["new"]:
            with self.hold_sync():
                if not self._custom_title:
                    self.title = self._auto_title()
                self._build_crystal()

    @traitlets.observe("voltage_kV")
    def _on_voltage(self, change) -> None:
        if self._ready and change["old"] != change["new"]:
            self._send_crystal()                        # the projected potential does not change with the voltage

    @traitlets.observe("detector", "camera_length_mm")
    def _on_camera(self, change) -> None:
        """A new camera takes its native pixel count; a new camera length keeps the pixel count (binning) and rescales."""
        if not self._ready or change["old"] == change["new"] or self.detector == "custom":
            return                                      # a custom camera keeps the reported pixels and sampling
        native_px = DETECTORS[self.detector]["pixels"]
        with self.hold_sync():
            if change["name"] == "detector":
                self.detector_px = native_px
            self.detector_mrad_per_px = detector_sampling_mrad(self.detector, self.camera_length_mm) * native_px / self.detector_px

    @traitlets.observe("detector_px", "detector_mrad_per_px")
    def _on_detector_sampling(self, change) -> None:
        if self._ready:
            self._update_column_phase()

    @traitlets.observe("thickness_nm")
    def _on_thickness(self, change) -> None:
        if self._ready and self.view_depth_nm > self.thickness_nm:
            self.view_depth_nm = self.thickness_nm

    # --- settings
    def apply_preset(self, name: str) -> None:
        """Set voltage, semiangle, camera and camera length from ``MICROSCOPE_PRESETS``."""
        if name not in MICROSCOPE_PRESETS:
            raise ValueError(f"Unknown preset {name!r}; choose one of {list(MICROSCOPE_PRESETS)}.")
        values = MICROSCOPE_PRESETS[name]
        with self.hold_sync():
            self.detector, self.camera_length_mm = values["detector"], values["camera_length_mm"]
            self.voltage_kV, self.semiangle_mrad = values["voltage_kV"], values["semiangle_mrad"]

    def apply_thickness(self, thickness_nm: float) -> None:
        """Set the thickness with the :func:`recommended_settings` for it (focus and scan size)."""
        values = recommended_settings(thickness_nm, voltage_kV=self.voltage_kV, semiangle_mrad=self.semiangle_mrad, scan_step_A=self.scan_step_A)
        with self.hold_sync():
            self.thickness_nm = values["thickness_nm"]
            self.focus_depth_nm, self.scan_size_px = values["focus_depth_nm"], values["scan_size_px"]
            self.view_depth_nm = values["focus_depth_nm"]

    # --- results
    def geometry(self) -> dict:
        """The numbers behind the checks for the current settings (:func:`plan_geometry`)."""
        return plan_geometry(voltage_kV=self.voltage_kV, semiangle_mrad=self.semiangle_mrad, focus_depth_nm=self.focus_depth_nm,
                             thickness_nm=self.thickness_nm, detector_px=self.detector_px,
                             detector_mrad_per_px=self.detector_mrad_per_px, scan_step_A=self.scan_step_A, scan_size_px=self.scan_size_px,
                             tilt_mrad=self.tilt_mrad, holz_repeat_A=self.holz_repeat_A or None,
                             wave_window_factor=self.wave_window_factor)

    def report(self):
        """The checks for the current settings as a DataFrame (one row per check: status, value, rule, note)."""
        import pandas as pd
        rows = check_rows(self.geometry(), detector_px=self.detector_px, scan_step_A=self.scan_step_A, column_phase_rad_per_A=self.column_phase_rad_per_A)
        return pd.DataFrame(rows).set_index("id")[["label", "status", "value", "rule", "note"]]

    def simulation_plan(
        self,
        *,
        repeats: Sequence[int] | None = None,
        pixels_per_cell: int | None = None,
        guard_A: float | None = None,
    ) -> dict[str, object]:
        """Size a periodic simulation cell around every planned probe position.

        Parameters
        ----------
        repeats : (int, int), optional
            Lateral unit-cell counts, in (row, col) order.
        pixels_per_cell : int, optional
            Potential pixels along each lateral unit-cell direction. This is
            distinct from the measured detector pixels.
        guard_A : float, optional
            Extra margin outside the geometric probe envelope on each side.

        Returns
        -------
        dict
            Extents, actual sampling, scan-center span and geometric margins.
            A passing margin is only a planning check: compare larger-cell
            diffraction and propagated boundary power before accepting a run.
            ``boundary_convergence_verified`` is always False.

        Notes
        -----
        Omitted arguments use the live ``simulation_repeats``,
        ``simulation_pixels_per_cell`` and ``simulation_guard_A`` traits,
        initially (24, 24), 96 and 5 Å. Explicit method arguments do not change
        those settings. This method plans geometry; it does not run abTEM.

        Examples
        --------
        >>> planner = PlanPtycho("crystal.cif", thickness_nm=60)
        >>> plan = planner.simulation_plan(repeats=(48, 48), guard_A=5)
        >>> plan["boundary_convergence_verified"]
        False
        """
        repeats = self.simulation_repeats if repeats is None else repeats
        pixels = (
            self.simulation_pixels_per_cell
            if pixels_per_cell is None
            else pixels_per_cell
        )
        guard = self.simulation_guard_A if guard_A is None else guard_A
        if len(repeats) != 2 or any(
            isinstance(x, bool) or not np.isfinite(x) or x < 1 or int(x) != x
            for x in repeats
        ):
            raise ValueError(
                "repeats must contain two positive integers in (row, col) order."
            )
        if (
            isinstance(pixels, bool)
            or not np.isfinite(pixels)
            or pixels < 1
            or int(pixels) != pixels
        ):
            raise ValueError(
                "pixels_per_cell must be a positive integer; use 96 as a starting point."
            )
        if not np.isfinite(guard) or guard < 0:
            raise ValueError(
                "guard_A must be finite and nonnegative; enlarge the cell to add margin."
            )
        geometry = self.geometry()
        span = (self.scan_size_px - 1) * self.scan_step_A
        extent = [self.cell_size_A[i] * repeats[i] for i in range(2)]
        lean = [
            self.thickness_nm * 10 * abs(math.tan(t * 1e-3)) for t in self.tilt_mrad
        ]
        margins = [
            (extent[i] - span - geometry["widest_A"]) / 2 - lean[i] for i in range(2)
        ]
        return dict(
            repeats=list(repeats),
            gpts=[int(n * pixels) for n in repeats],
            potential_sampling_A=[x / pixels for x in self.cell_size_A[:2]],
            extent_A=extent,
            scan_center_span_A=span,
            widest_probe_A=geometry["widest_A"],
            geometric_margin_A=margins,
            requested_guard_A=float(guard),
            geometric_fit=all(x >= guard for x in margins),
            z_repeats_to_cover=math.ceil(self.thickness_nm * 10 / self.cell_size_A[2]),
            requested_depth_A=self.thickness_nm * 10,
            boundary_convergence_verified=False,
        )

    @property
    def window_A(self) -> float:
        """Real-space wave window: factor times wavelength / measured detector angle."""
        return (
            self.wave_window_factor
            * wavelength_A(self.voltage_kV)
            / (self.detector_mrad_per_px * 1e-3)
        )

    @property
    def object_pixel_A(self) -> float:
        """Object pixel size: model window / wave pixels, independent of window factor."""
        return self.window_A / self.wave_pixels

    @property
    def wave_pixels(self) -> int:
        """Model wave pixels per side, distinct from measured detector pixels."""
        return self.detector_px * self.wave_window_factor
