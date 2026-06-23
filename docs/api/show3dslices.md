# Show3DSlices

Orthogonal slices of a 3D volume side by side with a synced crosshair. See the
[Show3DSlices tutorial](../tutorials/show3dslices).

## Reference

```{autodoc2-object} quantem.widget.show3dslices.Show3DSlices
render_plugin = "myst"
```

## Interactive controls

| Control | Trait | Expected effect |
|---|---|---|
| Crosshair drag | `slice_x`, `slice_y`, `slice_z` | Moves the cut position; all panels update together |
| Oblique angle | `oblique_angle` | Rotates the oblique cut plane |
| Colormap dropdown | `cmap` | Recolors all panels |
| Contrast min / max | `vmin`, `vmax` | Display clamp changes |
| Auto-contrast toggle | `auto_contrast` | Re-fits the percentile range |
| Log-scale toggle | `log_scale` | Log intensity mapping |
| FFT toggle | `show_fft`, `fft_window` | Panels show power spectra |
| Z-stretch | `z_stretch` | Depth axis scaled for anisotropic voxels |
| Scale bar toggle | `scale_bar_visible` | Calibrated bar shows/hides |
| Export button | `export_request`, `export_status` | Writes a standalone HTML viewer (exact / quantized) |
