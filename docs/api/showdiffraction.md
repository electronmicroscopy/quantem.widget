# ShowDiffraction

Interactive d-spacing analysis for a single 2D diffraction pattern or a 3D
stack (tilt/time series). Find the beam center, pick Bragg spots and rings,
read calibrated d-spacings, and calibrate k-space from a known reflection. For
full 4D-STEM datasets, use [Show4DSTEM](show4dstem). See the
[ShowDiffraction tutorial](../tutorials/showdiffraction) for a worked example.

## Reference

```{eval-rst}
.. autoclass:: quantem.widget.showdiffraction.ShowDiffraction
   :members:
   :show-inheritance:
```

## Interactive controls

Each control mutates the listed synced trait. A UI-test agent acts on the
control, then asserts the trait changed and the canvas repainted (non-zero,
no console error, no NaN frame).

| Control | Trait | Expected effect |
|---|---|---|
| Colormap dropdown | `dp_colormap` | Pattern recolors to the chosen map |
| Scale mode dropdown | `dp_scale_mode` | Intensity mapped linear / log / sqrt |
| Invert toggle | `dp_invert` | Colormap reversed |
| Contrast min / max sliders | `dp_vmin_pct`, `dp_vmax_pct` | Display clamp changes; histogram markers move |
| Center mode dropdown | `center_mode` | `auto` re-detects the BF disk; `manual` enables click-to-set |
| Click to set center (manual) | `center_row`, `center_col` | Crosshair moves; spot d-spacings recompute |
| Detect spots | `_detect_spots_request`, `spots` | Auto-finds Bragg peaks; d-spacing table fills |
| Add / remove spot (click) | `_spot_add_request`, `_spot_remove_request`, `spots` | Marker placed/removed; d-spacing updates |
| Snap toggle | `snap_enabled`, `snap_radius` | New spots snap to the nearest local maximum |
| Refine toggle | `spot_refine` | Sub-pixel Gaussian fit on/off |
| Detect rings | `_detect_rings_request`, `rings` | Auto-finds Debye–Scherrer rings |
| Add / remove ring | `_ring_add_request`, `_ring_remove_request`, `rings` | Ring overlay; ring d-spacing updates |
| Calibrate from spot / ring | `_calibrate_from_spot_request`, `_calibrate_from_ring_request`, `k_pixel_size` | Sets k-space pixel size from a known d |
| Frame slider (3D) | `frame_idx` | Scrubs to a different pattern in the stack |
| Pan (drag) / zoom (wheel) | view transform | Pattern translates / zooms about the cursor |
| Export → HTML | `export_request`, `export_payload` | Writes a standalone HTML viewer |

```{seealso}
The shared HTML-export contract is documented in [html-export](html-export).
```
