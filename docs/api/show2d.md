# Show2D

One or many 2D images with contrast control, FFT, ROIs, line profiles, and a
calibrated scale bar. See the [Show2D tutorial](../tutorials/show2d) for a
worked example.

## Reference

```{autodoc2-object} quantem.widget.show2d.Show2D
render_plugin = "myst"
```

## Interactive controls

Each control mutates the listed synced trait. A UI-test agent acts on the
control, then asserts the trait changed and the canvas repainted (non-zero,
no console error, no NaN frame).

| Control | Trait | Expected effect |
|---|---|---|
| Colormap dropdown | `cmap` | Canvas recolors to the chosen map |
| Contrast min / max sliders | `vmin`, `vmax` | Display clamp changes; histogram markers move |
| Auto-contrast toggle | `auto_contrast` | Re-fits `vmin`/`vmax` to the percentile range |
| Log-scale toggle | `log_scale` | Intensity mapped through log |
| FFT toggle | `show_fft` | Canvas shows the power spectrum; lattice spots appear |
| FFT window toggle | `fft_window` | Apodization on/off (ringing at edges differs) |
| Scale bar toggle | `scale_bar_visible` | Calibrated bar shows/hides (needs `pixel_size > 0`) |
| Pan (drag) | per-image pan | Image translates; with `link_pan` all panels move together |
| Zoom (wheel) | `initial_zoom`, `zoom_row`, `zoom_col` | Zooms about the cursor |
| Smooth toggle | `smooth` | Bilinear vs nearest sampling |
| ROI add / drag | `roi_active`, `roi_list`, `roi_selected_idx` | Region overlay; stats panel reports the ROI |
| Gallery select | `selected_idx` | Highlights the active panel |
| Diff mode | `diff_mode`, `diff_reference` | Panels render as difference vs the reference |

```{seealso}
The deeper behavioral spec (invariants, per-feature pass criteria, isolation
checks) lives alongside the integration test at `widget/docs/show2d-test-spec.md`.
```
