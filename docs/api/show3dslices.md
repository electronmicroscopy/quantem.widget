# Show3DSlices

Linked top and arbitrary-angle side cuts through a 3D volume. See the
[Show3DSlices tutorial](../tutorials/show3dslices).

## Viewer UI

`Show3DSlices` supports the shared `ui_mode`, `show_title`, `show_controls`,
`controls_collapsed`, `show_stats`, `show_crosshair`, and `show_scale_bar`
names. The saved-state scale-bar trait remains `scale_bar_visible` for
compatibility. See [Viewer UI controls](viewer-ui).

## Reference

```{eval-rst}
.. autoclass:: quantem.widget.show3dslices.Show3DSlices
   :members:
   :show-inheritance:
```

## Interactive controls

| Control | Trait | Expected effect |
|---|---|---|
| Page slider / play | `page_idx`, `n_pages`, `page_labels` | Switches or plays comparable volumes while preserving the same slice geometry and view state |
| Panel menu | `active_panel`, `panels_per_page`, `panel_titles` | Selects one panel slot within the active page |
| Crosshair drag | `slice_x`, `slice_y`, `slice_z` | Moves the cut position; all panels update together |
| Oblique line drag | `oblique_profile_line`, `oblique_angle`, `slice_x`, `slice_y` | Drag either endpoint to rotate/resize the side cut, or drag the line body to translate the whole cut freely in row/col while staying inside the top slice |
| Oblique angle | `oblique_angle` | Rotates the oblique cut plane |
| Colormap dropdown | `cmap` | Recolors all panels |
| Contrast min / max | `vmin`, `vmax` | Display clamp changes |
| Auto-contrast toggle | `auto_contrast` | Re-fits the percentile range |
| Log-scale toggle | `log_scale` | Log intensity mapping |
| Slice alignment | `slice_alignment`, `row_shift_px_per_slice`, `col_shift_px_per_slice` | Open **Advanced** and enable **Slice alignment**. The first enable estimates one global row/col shift per slice automatically; later raw/aligned toggles reuse the cached estimate and aligned display volume. The sliders refine the cached result without modifying the raw volume. |
| FFT toggle | `show_fft`, `fft_window` | Panels show power spectra |
| Z-stretch | `z_stretch` | Depth axis scaled for anisotropic voxels |
| Scale bar toggle | `scale_bar_visible` | Calibrated bar shows/hides |
| Export button | `export_request`, `export_status` | Writes a standalone HTML viewer (exact / quantized) |

Automatic slice alignment high-pass filters adjacent reconstructed slices with
a Gaussian sigma capped at 12 pixels, then refines each translation on a 20x
local matrix-DFT grid. The 12-pixel scale is an empirical preprocessing setting,
not a physical microscope parameter. The subpixel method follows
[Guizar-Sicairos, Thurman, and Fienup (2008), DOI 10.1364/OL.33.000156](https://doi.org/10.1364/OL.33.000156).
The resulting row/col trajectory is reduced to one global linear shift per
slice. This is reversible display post-processing; it does not add sample tilt
to the ptychographic forward model or modify the reconstructed volume.

## Compare volumes with pages

Pass a 4D stack plus `page_labels` when every page contains one volume. The
same top slice, side cut, oblique angle, zoom, contrast, and playback settings
remain active as the page changes:

```python
# Shape: page, depth, rows, cols
w = Show3DSlices(
    np.stack([raw_volume, aligned_volume]),
    page_labels=["raw", "aligned"],
    panel_titles=["object phase"],
)
```

Use a 5D array when each page has the same small panel set:

```python
# Shape: page, panel, depth, rows, cols
w = Show3DSlices(
    comparison_5d,
    page_labels=["single-slice", "multislice"],
    panel_titles=["object", "absolute error"],
)
```

Explicit page dictionaries are also accepted:

```python
w = Show3DSlices([
    {"title": "before", "volume": before},
    {"title": "after", "volume": after},
])
```

`set_page()`, `next_page()`, and `previous_page()` provide the same workflow
from Python. Existing 4D calls without `page_labels` keep their legacy
multi-panel meaning, so saved notebooks do not silently change interpretation.
