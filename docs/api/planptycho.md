# PlanPtycho

Check multislice ptychography settings against a known crystal before the experiment. Give a crystal (CIF path,
`ase.Atoms`, or Materials Project id) and the microscope settings; the widget shows the beam through the specimen, the
reconstruction's model window, the probe on that window, the Bragg disks on the detector, and graded checks.

```python
from quantem.widget import PlanPtycho

plan = PlanPtycho("SrTiO3.cif", zone_axis=(0, 0, 1), thickness_nm=110, tilt_mrad=(3.2, -2.4), c10_nm=-17)
plan.report()               # the checks as a DataFrame
plan.apply_thickness(40)    # recommended focus and scan size for 40 nm
plan.apply_preset("Arina · 300 kV · 25 mrad · 115 mm")
plan.window_A, plan.object_pixel_A
```

Needs `ase`, `abtem` and, for Materials Project ids, `spglib` (`pip install "quantem.widget[crystal]"`).

## Settings from a collaborator

To check whether an acquisition someone else recorded can work for ptychography, pass the values as reported. Giving
`detector_mrad_per_px` without a camera name makes the camera `"custom"`; `c10_nm` is the defocus in the quantem sign
(negative focuses into the specimen; `focus_depth_nm = -c10_nm`):

```python
PlanPtycho("crystal.cif", thickness_nm=40, voltage_kV=200, semiangle_mrad=24.5,
           detector_px=128, detector_mrad_per_px=0.9, scan_step_A=0.4, scan_size_px=256, c10_nm=-15)
```

In the widget every number can be typed: click it, enter the value, press Enter.

## Presets

**Microscope** (`preset=`, or the menu in the title bar), named by their settings:

| Preset | kV | Semiangle | Camera | Camera length | mrad per pixel |
|---|---|---|---|---|---|
| Arina · 300 kV · 30 mrad · 91 mm (default) | 300 | 30 | Arina, 192 px | 91 mm | 0.554 |
| Arina · 300 kV · 25 mrad · 115 mm | 300 | 25 | Arina, 192 px | 115 mm | 0.461 |
| Arina · 300 kV · 21.4 mrad · 185 mm | 300 | 21.4 | Arina, 192 px | 185 mm | 0.269 |
| EMPAD · 300 kV · 30 mrad · 91 mm | 300 | 30 | EMPAD, 128 px | 91 mm | 0.831 (scaled) |

The Arina values are measured calibrations at 300 kV (semiangle over fitted bright-field disk radius); other camera
lengths scale as 50.4 mrad mm / camera length. The EMPAD is scaled from them by pixel pitch (150 / 100 um) and is not
measured. `detector_px` and `detector_mrad_per_px` override the calibration, e.g. `detector_px=96` for a 2x binned Arina.

**Thickness** (`apply_thickness`, or "Recommended ... nm" in the title bar), for 20 to 70, 100, 150 and 200 nm (cryo
and biological sections): focus at mid-thickness and a scan wide enough that the margin check passes (at least
128 x 128). These come from the checks below; reconstructions tested the checks on 90-130 nm SrTiO3 with the focus
17 nm below the entrance. With the Arina at 91 mm and 30 mrad every check passes to 100 nm; at 150 and 200 nm the beam
(46 and 61 A at best focus) outgrows the 35.5 A window. A 185 mm camera length widens the window to 73 A but reaches
only 26 mrad: the widget shows that trade-off rather than choosing for you.

## Checks

| Check | Formula | Grade |
|---|---|---|
| Beam fits the virtual window | widest beam `2 alpha abs(z - f) + 1.22 lambda / alpha` vs `lambda / dtheta` | caution when wider |
| Scan margin | scan side vs 4 x widest beam radius | caution when smaller |
| Probe overlap | `1 - step / entrance beam diameter` | pass >= 60 %, fail < 30 % |
| Detector reach | detector edge / semiangle | pass >= 1.5, fail < 1 |
| Column lean | `thickness x tan(tilt)` vs object pixel | caution when larger |
| Focus splits the specimen | `0 <= focus depth <= thickness` | caution when the whole slab is on one side |

Window, margin, lean and focus only caution: simulated 110-130 nm SrTiO3 reconstructed with the beam wider than the
window, a scan narrower than the beam (weaker only at the scan edge) and a 4 mrad tilt held at its measured value, all
with the focus inside the specimen. Information rows give the object pixel, the depth of field (`2 lambda / alpha^2`),
beam diameters, the phase of the strongest column per nm, and the first HOLZ ring (lattice period along the beam,
counting centring translations). Slice thickness is a reconstruction choice and is not part of the plan.

The Python functions (`plan_geometry`, `check_statuses`, `check_rows`, `recommended_settings`, `detector_sampling_mrad`)
and the browser arithmetic are pinned to the same `js/planptycho/goldens.json`, down to the text of each check.

## Reference

```{eval-rst}
.. autoclass:: quantem.widget.planptycho.PlanPtycho
   :members: report, geometry, apply_preset, apply_thickness, window_A, object_pixel_A

.. autofunction:: quantem.widget.planptycho.plan_geometry

.. autofunction:: quantem.widget.planptycho.check_rows

.. autofunction:: quantem.widget.planptycho.recommended_settings
```

## Interactive controls

| Control | Trait | Expected effect |
|---|---|---|
| Drag in **Top** | (view only) | moves the probe (starts at the scan centre); its beam at the view depth and the model window follow |
| Drag the dashed line in **Side** | `focus_depth_nm` | moves the focus through the stationary specimen; the labels show how much lies above and below it |
| Drag elsewhere in **Side** | `view_depth_nm` | sets the depth shown in Top and Probe |
| **Sample** switch on Probe | (view only) | draws the crystal at the view depth under the probe |
| Wheel / double-click on any panel | (view only) | zoom about the cursor / reset |
| Microscope and Recommended menus | several | apply a preset in one step |
| Zone | `zone_axis` | rebuilds the projection and Bragg disks in Python |
| Microscope: Voltage, Semiangle, C10 | `voltage_kV`, `semiangle_mrad`, `focus_depth_nm` (= -C10) | the knobs an operator turns; C10 is the defocus (negative focuses into the specimen) and the focus depth is read out; numbers can be typed |
| Sample: Zone, Thickness, Tilt row / col | `zone_axis`, `thickness_nm`, `tilt_mrad` | facts about the specimen |
| Camera, Length | `detector`, `camera_length_mm` | set `detector_px` and `detector_mrad_per_px` from the calibration |
| Camera custom: Pixels, Sampling | `detector`, `detector_px`, `detector_mrad_per_px` | a camera without a calibration: type the reported values |
| Binning | `detector_px`, `detector_mrad_per_px` | 1x / 2x / 4x: fewer pixels, the same total angle; the sampling per pixel is read out, not set (Python `detector_mrad_per_px=` overrides it for an uncalibrated camera) |
| Scan: Step, Size | `scan_step_A`, `scan_size_px` | probe step (magnification) and positions per side (64 to 512); the field of view is read out |

The widget has no `save_state` or HTML export; it is rebuilt from the crystal in one call.

## Simulation-cell planning from a CIF

A CIF path or ASE `Atoms` is accepted by the existing constructor. The
**Simulation Cell** section checks unit-cell repeats, potential pixels per cell,
and the clearance around the complete scan-center span `(N - 1) × step`.
Potential sampling is separate from the detector-derived reconstruction sampling.

```python
plan = PlanPtycho(
    "BaTiO3.cif", zone_axis=(0, 0, 1), thickness_nm=60,
    focus_depth_nm=20, scan_step_A=0.373, scan_size_px=128,
    detector_px=192, detector_mrad_per_px=0.5570968023269496,
)
plan.simulation_repeats = [48, 48]
plan.simulation_pixels_per_cell = 96
plan.simulation_plan()
```

`simulation_plan(repeats=(48, 48), pixels_per_cell=96, guard_A=5)` can also
calculate a candidate without changing the controls. The returned depth repeat
count covers the requested slab; the final repeat may need truncating to obtain
an exact thickness. The geometric margin includes the full directional tilt
excursion conservatively on each side.

This section is a planning check, not a simulation launcher. The Detector panel
is labelled **schematic**. Changing these controls does not update previously
computed multislice diffraction. Validate propagated boundary power, a larger
lateral cell, potential sampling and propagation slices with the simulation
backend before accepting a dataset. A geometric pass deliberately leaves
`boundary_convergence_verified=False`.

## Optional Virtual Window

`wave_window_factor=2` previews a 384 × 384 wave for a 192 × 192 measured detector. The **Virtual Window** controls at the bottom of the planner switch between native and doubled support. This is an optional adjustment; microscope, detector, scan, and specimen settings stay above it. This doubles the physical model width while retaining the real-space pixel size and measured angular reach; it does not change the acquired detector pixels.

```python
PlanPtycho("BaTiO3.cif", detector_px=192,
           detector_mrad_per_px=0.5570968023269496,
           wave_window_factor=2)
```

The reconstruction must support integrating its finer predicted diffraction intensities back onto the measured detector. The planning preview does not run or validate that reconstruction. Its free-space probe envelope is a geometric check, not a guarantee of multislice convergence. Use `ShowCIF` alongside the planner to inspect the same CIF and the atomic columns.
