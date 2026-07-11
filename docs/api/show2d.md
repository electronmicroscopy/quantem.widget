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
| FFT quality labels | `fft_metrics` | Compact in-panel label reports FFT sharpness, peak count, and peak SNR from the cached FFT magnitude |
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
| Local stack slider / play | `panel_frame_indices`, `panel_frame_counts`; `set_panel_frame()` | Scrubs only the selected 3D list item; static neighboring panels do not move |
| Gallery page controls | `page_idx`, `n_pages`, `panels_per_page`, `page_labels`, `page_starred` | Switch, star, or play through panel pages without changing the source stack |
| Panel reorder | `panel_order`; `set_panel_order()`, `move_panel()`, `reset_panel_order()` | Reorders gallery display without changing source data, labels, stars, or hidden state |
| Diff mode | `diff_mode`, `diff_reference` | Panels render as difference vs the reference |
| Link Denoise switch (gallery) | `denoise_scope` | Linked ("all"): denoise edits apply to every panel; unlinked ("panel"): edits apply to the selected panel only |
| View menu: Crop to view | `view_crop`; `crop_to_view()` | Commits the current viewport as the display extent (single panel, display-only, reversible) |
| View menu: Pad 5% / 10% / 20% | `pad_ratio` | Adds a border on each side, filled with the image minimum |
| View menu: Reset view | `reset_view_ops()` | Restores the uncropped, unpadded display bit-identically |

## Which denoise filter should I use?

The Denoise controls are hidden behind their own toggle by default; everything
here is display-only (the stored array, the stats row, and raw exports keep
the original counts, and an active denoise always announces itself with a
one-line banner). Three methods cover the space; binning is a separate knob,
so there is no "bin2_anscombe" menu entry: pick **Poisson (Anscombe)** and set
**Bin 2**.

| Your data | Use | Why |
|---|---|---|
| Sparse EDS / counting maps | **Poisson (Anscombe), Bin 2, sigma 6-10** | Respects Poisson count statistics; the standard choice for element maps |
| Very sparse maps (single counts) | Poisson (Anscombe), Bin 4, sigma 8-12 | More SNR from binning before smoothing |
| HAADF / decent-dose images | Gaussian, Bin 1, sigma 1-2 (or nothing) | The data is not count-starved; a light smooth is enough |
| Anything quantitative (FFT, intensities, stats) | None | Measure on raw counts; the stats row is always computed from raw data |

From Python, the same ladder is `denoise="anscombe", denoise_bin=2,
denoise_sigma=8` (per-panel lists supported for A/B galleries); legacy
spellings like `display_filter="bin2_anscombe"` keep working as aliases.

## Crop and pad the view (advanced)

Single-panel widgets can commit the current browser viewport as the display
extent and add a ratio-based border, either from the toolbar's **View** menu
(Crop to view, Pad 5% / 10% / 20%, Reset view) or from Python:

```python
w = Show2D(image, view_box=(64, 64, 96))  # zoom into a feature
w.crop_to_view()          # the 96x96 window becomes the displayed frame
w.pad_ratio = 0.1         # border on each side, 10% of max(rows, cols)
w.reset_view_ops()        # full frame again, bit-identical
```

Both ops honor the display-only contract:

- The stored array is never modified; `reset_view_ops()` returns the exact
  original frame bytes.
- The crop applies in the display pipeline **before** denoise, so an active
  denoise operates on the cropped region; the pad (filled with the image
  minimum, keeping the colormap floor) applies after it.
- The stats row keeps reporting the full raw data, and cursor coordinates
  remain full-image (row, col) while a crop or pad is active: the crop is a
  display window, not a new coordinate system.
- An active crop or pad is never silent: a one-line `view:` banner names the
  window and the ratio, e.g.
  `view: cropped to (64,64)-(160,160) · pad 10% (reset_view_ops() restores full frame)`.
- Both persist through `state_dict()` / `load_state_dict()`.

Galleries are not supported in this release; `crop_to_view()` raises
`NotImplementedError` so a multi-panel session never gets a silently wrong
window. Crop the arrays before display instead.

## FFT quality labels

Pass `show_fft=True` to show the FFT panel. By default, `fft_metrics=True`
adds a small white label inside each FFT panel with three quick checks:
sharpness, peak count, and peak SNR. These values are computed from the FFT
magnitude already used for rendering, so the label does not trigger a second
FFT. Set `fft_metrics=False` when a clean FFT image is more important than the
readout.

The first FFT for an image or ROI may take a moment on large data. After that,
Show2D reuses the cached FFT magnitude for redraws, zoom/pan, contrast changes,
and metric labels.

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

## Mixed static and local stack panels

A bare 3D array still means a static gallery of `N` images. To put an
independent stack inside one gallery panel, pass a list whose item is shaped
`(frames, rows, cols)`. Other list items can remain ordinary 2D images:

```python
from quantem.widget import Show2D

w = Show2D(
    [eds_sum_map, haadf_stack, ti_map, o_map],
    labels=["EDS sum", "HAADF", "Ti", "O"],
    panel_frame_indices=[0, -1, 0, 0],
    ncols=2,
    auto_contrast=True,
)
w
```

Only the HAADF panel gets an in-panel slider and play button. Its frame index
is independent of every other panel and remains keyed to that source panel
when panels are hidden or reordered. Click the stack panel and use the left or
right arrow key to scrub it. From Python, use
`w.set_panel_frame("HAADF", -1)`.

`state_dict()` records `panel_frame_indices`, and both exact-float32 and
quantized-uint8 HTML exports contain every local frame. For an interactively
restored notebook output, construct with `save_state=True`; the default
`save_state=False` deliberately stores only a compact static preview rather
than embedding the stack payload in the notebook.

`set_image()` accepts the same mixed list form:

```python
w.set_image(
    [next_eds_map, next_haadf_stack, next_ti_map, next_o_map],
    labels=["EDS sum", "HAADF", "Ti", "O"],
    panel_frame_indices=[0, 0, 0, 0],
)
```

## Watch a growing image folder

Use `Show2D.from_folder(...)` when a microscope or reconstruction job writes
new 2D images into one folder. Each newly readable file becomes another panel
in the existing gallery. The widget object is not rebuilt, so the current
selection and the state of panels already present remain stable.

```python
from quantem.widget import Show2D

w = Show2D.from_folder(
    "/data/session/haadf",
    pattern="*.tif",
    watch_interval=2.0,
    title="Live HAADF images",
)
w
```

`watch=True` is the default. Pass `watch=False` for a fixed folder or a
reproducible script that should update only when you call `poll_folder()`.

```python
new_panels = w.poll_folder()       # scan now; return newly appended indices
w.stop_folder_watch()             # pause background scans
w.watch_folder(interval=1.0)      # resume with a different interval
w.close()                         # stop watching and close the widget
```

Folder watching is append-only. A file already represented in the gallery is
not duplicated, an incomplete file is deferred until a later poll, and removing
or rewriting a source file does not silently remove or replace an existing
panel. Close long-running widgets when the notebook no longer needs them.

`Show2D.from_folder(...)` reads the scientific image data at its source
resolution. It is different from `ShowFolder`, which intentionally uses small
cached thumbnails for fast folder discovery and selection. Use `ShowFolder`
to decide what to open; use `Show2D.from_folder(...)` when the displayed pixel
data and live panel append behavior matter.

## Paged galleries

Use paged galleries when each view contains the same panel grid across several
analysis settings, iterations, or parameter values. A common example is a
4-by-4 reconstruction sweep where each page is one iteration or one denoising
parameter, and the panels within the page are the related output images.

Pass a 4D array with shape `(pages, panels_per_page, rows, cols)`:

```python
from quantem.widget import Show2D

w = Show2D(
    reconstruction_sweep,
    labels=[
        "lambda 0.01", "lambda 0.03", "lambda 0.10", "lambda 0.30",
        "lambda 1", "lambda 3", "lambda 10", "lambda 30",
        "raw", "filtered", "residual", "score",
        "phase", "amplitude", "mask", "diagnostic",
    ],
    page_labels=["iteration 10", "iteration 20", "iteration 30"],
    ncols=4,
)
w
```

You can also pass explicit page dictionaries when the page titles naturally
belong beside the image data:

```python
w = Show2D(
    [
        {"title": "iteration 10", "images": iter10_panels, "labels": panel_labels},
        {"title": "iteration 20", "images": iter20_panels, "labels": panel_labels},
    ],
    ncols=4,
)
```

Paged galleries keep the Python data model simple: all pages use the same panel
count and image shape, the browser renders only the active page, and page stars
are stored separately from panel stars. Use `star_page(page)` and
`unstar_page(page)` from Python, or the star button beside the page slider in
the widget. The page row also includes play/pause and a small FPS menu so
readers can step through iteration or parameter pages without touching the main
image controls. Manual slider scrubbing pauses page playback. For very large 4K
sweeps, start with a reduced or representative stack; a future lazy page
transport can avoid sending every page to the browser at once.

`set_image()` is the re-render trigger. Mutating the original NumPy array in
place does not notify the frontend. The method sends fresh synced `frame_bytes`
and resets state tied to the old image count or dimensions, including stale
hidden panels, stars, view box, ROIs, line profiles, detail tiles, and panel
order. Use `offline=False` for acquisition-style updates so the live Jupyter
Comm path carries each new frame instead of the saved/offline notebook path.
