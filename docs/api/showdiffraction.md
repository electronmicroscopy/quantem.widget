# ShowDiffraction

Interactive d-spacing analysis for a single 2D diffraction pattern or a 3D
stack (tilt/time series). Find the beam center, pick Bragg spots and rings,
read calibrated d-spacings, and calibrate k-space from a known reflection. See the
[ShowDiffraction tutorial](../tutorials/showdiffraction) for a worked example.

The primary phase workflow is candidate verification: you usually know which
phases to expect, so build them (`library_phase`, `Phase.from_cubic`,
`Phase.from_dspacings`, or custom phases in the Phase menu) and rank only
those with `identify_phase(database)` or the *candidates only* switch.
`search_phases()` against the built-in library is the fallback for when you
have no candidates in mind; narrow it with an element filter.

Reference data sources: every lattice parameter in the built-in phase library
(`PHASE_LIBRARY` in `showdiffraction/phases.py`) is a room-temperature value taken from a
license-clean source cited next to the entry: NIST SRM certificates and NBS
circulars/monographs (US public domain), the Crystallography Open Database
(CC0), or the primary literature. No values come from proprietary compilations
such as the ICDD PDF or Pearson's Handbook.

The lattice-based phase model enumerates d-spacings and geometric systematic
absences. It does not calculate structure-factor intensities, thermal effects,
or dynamical diffraction; use `Phase.from_dspacings` when matching against a
measured or literature line table with intensities.

Calibrated radial axes are reported as `g = 1/d` in `1/Å`, not as a `2πg`
scattering-vector axis.

## Viewer UI

`ShowDiffraction` has the shared `show_title`, `show_controls`,
`controls_collapsed`, and `show_stats` traits (set them after construction). See
[Viewer UI controls](viewer-ui).

## Reference

```{eval-rst}
.. autoclass:: quantem.widget.showdiffraction.ShowDiffraction
   :members:
   :show-inheritance:
```

## Interactive controls

Each control mutates the listed synced trait. Analysis buttons send one
`_request` message (`{"action": ..., "args": [...], "seq": n}`); Python runs the
action, reports in `analysis_status`, and clears `_request`. A UI-test agent acts on the
control, then asserts the trait changed and the canvas repainted (non-zero,
no console error, no NaN frame).

| Control | Trait | Expected effect |
|---|---|---|
| Colormap dropdown | `dp_colormap` | Pattern recolors to the chosen map |
| Scale mode dropdown | `dp_scale_mode` | Intensity mapped linear / log / sqrt |
| Invert toggle | `dp_invert` | Colormap reversed |
| Contrast histogram (dual-thumb slider) | `dp_vmin_pct`, `dp_vmax_pct` | Drag either thumb (mouse or touch) for a live preview; traits update once on release |
| Center mode dropdown | `center_mode` | `auto` re-detects the BF disk; `manual` enables click-to-set |
| Click to set center (manual) | `center_row`, `center_col` | Crosshair moves; spot d-spacings recompute |
| Detect spots | `_request`, `spots` | Auto-finds every isolated peak with contrast at least 10% of the strongest (`min_relative`); candidates come from the `detect_denoise` view, positions are refined on raw data; no count cap |
| Add / remove spot (click) | `_request`, `spots` | Marker placed/removed; d-spacing updates |
| Move spot (Move + drag) | `_request`, `spots` | Re-picks the spot at the drop position; stale hkl clears |
| Spot pick dropdown | `spot_refine`, `snap_enabled`, `snap_radius` | Clicked spots are Gaussian-fitted, snapped to the local maximum, or kept exactly as clicked |
| Detect rings | `_request`, `rings` | Auto-finds every Debye–Scherrer ring that stands at least 5 standard errors of the azimuthal mean above the profile's lower envelope; peaks are located on the `detect_denoise` view, significance and ring fits use the raw counts; no count cap |
| Add / remove ring | `_request`, `rings` | Ring overlay; ring d-spacing updates |
| Calibrate from spot / ring | `_request`, `k_pixel_size` | Sets k-space pixel size from a known d |
| Auto button | `_request`, `analysis_status` | Runs center, rings, calibration, fit, and indexing in one pass; the calibration is applied only when the ring-radius ratios verify it; status reports failed steps and the rings the phase does not explain |
| Phase menu | `phase_name`, `custom_phases` | Selects a library or custom phase for calibration and indexing; custom entries take a full lattice (a, b, c, α, β, γ) and absence rule |
| Identify candidates only | `identify_custom_only` | Identify ranks only custom phases, skipping the library |
| Calibrate from phase | `_request`, `calibration_rms_px` | Fits k-space sampling from the ring-to-reflection assignment whose radius ratios match the phase's d-spacing ratios within 1%; refuses (status says why) when two scales explain the rings about equally or most rings stay unexplained |
| Index rings / spots | `_request`, `zone_axis` | Fills hkl labels at the current calibration. A ring label needs the ring-radius ratios to agree with the phase and with the calibration (within 3 %); a ring whose d only falls within 3 % of a reflection shows that candidate with `?` and is named in the status. Spot indexing also solves the zone axis |
| Exclude menu | `mask_regions` | Edits wedge/disk regions excluded from analysis |
| Draw excluded disk / wedge (drag) | `mask_regions` | Drag on the pattern to add an excluded region with live preview |
| Mask view toggle | `show_mask` | Shows or hides excluded-region overlays |
| Fit rings | `_request` | Refines ring radius and width; fwhm column appears |
| Fit ellipse / use correction | `_request`, `ellipse_corrected` | Measures distortion; the switch circularizes radii |
| Profile panel | `show_profile`, `profile_log`, `profile_subtract_background`, `_profile_data` | Radial profile with ring markers; click adds a ring |
| Azimuthal panel | `show_azimuthal`, `_azimuthal_data` | Intensity vs azimuth around the outermost ring |
| hkl toggle | `show_hkl` | Shows or hides hkl labels on spots and rings |
| Stats toggle | `show_stats` | Shows or hides the pattern-statistics readout |
| Undo / clear spots and rings | `_request` | Removes the last or all markers (Ctrl+Z also undoes) |
| Center view | view transform | Recenters and zooms the view to the diffraction center |
| Spot / ring CSV and JSON | measurement tables | Downloads the visible measurement table rows |
| Refine method dropdown | `refine_method` | Picks the center-refinement algorithm (auto / symmetry / phase_corr) |
| Refine center | `_request`, `center_method` | Refines the center; records the method used |
| Merge frames (3D) | `_request` | Align + merge frames; appends the combined pattern |
| Element filter box | `identify_elements` | Restricts the phase search to these elements (e.g. `Fe,O`) |
| Identify phases | `_request`, `_identify_results` | Ranked candidate phases with per-line match tables |
| Quality panel | `_quality` | Analysis-quality snapshot: center, calibration, ellipse, ring fits, rings the selected phase does not explain, mask coverage |
| Ring click in profile | `selected_ring_id` | Highlights the picked ring marker (0 clears) |
| Frame slider (3D) | `frame_idx` | Scrubs to a different pattern in the stack |
| Pan (drag) / zoom (wheel) | view transform | Pattern translates / zooms about the cursor |
| Touch: two-finger pinch / drag, double-tap | view transform | Pinch zooms about the fingers, two-finger drag pans, double-tap resets the view |
| Export → PNG / HTML | `export_request`, `export_payload` | Saves the current view as a PNG image or a standalone HTML viewer |

```{seealso}
The shared HTML-export contract is documented in [html-export](html-export).
```
