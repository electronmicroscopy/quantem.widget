# Show2D

One or many 2D images with contrast control, FFT, ROIs, line profiles, and a
calibrated scale bar. See the [Show2D tutorial](../tutorials/show2d) for a
worked example.

## Reference

```{eval-rst}
.. autoclass:: quantem.widget.show2d.Show2D
   :members:
   :show-inheritance:
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
| Viewer chrome preset | `ui_mode` plus explicit `show_*` kwargs | Applies shared display presets; see [Viewer UI controls](viewer-ui) |
| Control visibility | `show_controls`, `controls_collapsed`; `collapse_controls()`, `expand_controls()`, `toggle_controls()` | Permanently remove controls or temporarily collapse them behind the top GUI toggle |
| Title visibility | `show_title` | Top title row shows/hides |
| Stats visibility | `show_stats` | Mean/min/max/std readout shows/hides |
| Panel title visibility | `show_panel_titles`, `panel_title_font_size` | Per-panel labels show/hide and resize |
| Scale bar toggle | `show_scale_bar` (`scale_bar_visible` in saved state) | Calibrated bar shows/hides (needs `pixel_size > 0`) |
| Pan (drag) | per-image pan | Image translates; with `link_pan` all panels move together |
| Zoom (wheel) | `initial_zoom`, `zoom_row`, `zoom_col` | Zooms about the cursor |
| Smooth toggle | `smooth` | Bilinear vs nearest sampling |
| ROI add / drag | `roi_active`, `roi_list`, `roi_selected_idx` | Region overlay; stats panel reports the ROI |
| Gallery select | `selected_idx` | Highlights the active panel |
| Panel reorder | `panel_order`; `set_panel_order()`, `move_panel()`, `reset_panel_order()` | Reorders gallery display without changing source data, labels, stars, or hidden state |
| Diff mode | `diff_mode`, `diff_reference` | Panels render as difference vs the reference |

```python
w = Show2D(images, labels=["raw", "filtered", "residual"])
w.set_panel_order(["residual", "raw", "filtered"])
w.move_panel("raw", 0)
w.reset_panel_order()
```

```{seealso}
The deeper behavioral spec (invariants, per-feature pass criteria, isolation
checks) lives alongside the integration test at `widget/docs/show2d-test-spec.md`.
```

## Live image updates

Use `set_image()` to trigger a new browser render in an already displayed
`Show2D` widget. Keep a reference to the widget object, display it once, then
replace the image or gallery data through that method:

```python
import numpy as np
from quantem.widget import Show2D

w = Show2D(first_image, labels=["initial"], offline=False)
w

for step, next_image in enumerate(image_stream, start=1):
    w.set_image(next_image, labels=[f"step {step}"])
```

For a gallery, pass a 3D stack `(N, H, W)` and one label per panel:

```python
w.set_image(
    np.stack([raw, filtered, residual]),
    labels=["raw", "filtered", "residual"],
)
```

`set_image()` is the re-render trigger. Mutating the original NumPy array in
place does not notify the frontend. The method sends fresh synced `frame_bytes`
and resets state tied to the old image count or dimensions, including stale
hidden panels, stars, view box, ROIs, line profiles, detail tiles, and panel
order. Use `offline=False` for acquisition-style updates so the live Jupyter
Comm path carries each new frame instead of the saved/offline notebook path.
