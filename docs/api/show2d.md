# Show2D

One or many 2D images, including gallery panels with independent local slice
stacks, plus contrast control, FFT, ROIs, line profiles, and a calibrated scale
bar. See the [Show2D tutorial](../tutorials/show2d) for worked examples and
[Advanced Show2D](../tutorials/show2d_advanced) for report-ready galleries,
rich labels, local annotations, overlays, inset plots, and export patterns.

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
| More menu: Color shared | `panel_cmaps`, `panel_cmaps_memory` | Shared by default. Turn off to let the Color dropdown edit only the selected gallery panel; turn on for one shared colormap, then turn off again to restore the previous individual panel colormaps |
| Panel identity markers | `marker_colors`, `identity_colors`, `marker_style`, `row_markers`, `col_markers` | Optional panel colors plus row/column group frames make panels easy to reference in notebooks, reports, and agent instructions |
| Local panel annotations | `panel_annotations` | Optional multiple in-image labels per panel, placed by corner, normalized point, or normalized region box |
| Geometric panel overlays | `panel_overlays` (`overlays` shorthand) | Optional circle, rectangle, and square overlays in data or relative coordinates, with stroke/fill opacity and z-order; More -> Overlay Edit allows live select, move, resize, delete, and reset |
| Contrast min / max sliders | `vmin`, `vmax` | Display clamp changes; histogram markers move |
| Auto-contrast toggle | `auto_contrast` | Re-fits `vmin`/`vmax` to the percentile range |
| More menu: Contrast | `contrast_preset` | Applies scientist-friendly percentile ranges such as `1-99`, `2-98`, and `3-97`; the histogram stays visible below the image |
| Log-scale toggle | `log_scale` | Intensity mapped through log |
| FFT toggle | `show_fft` | Canvas shows the power spectrum; lattice spots appear |
| FFT window toggle | `fft_window` | Apodization on/off (ringing at edges differs) |
| FFT quality labels | `fft_metrics` | Compact in-panel label reports FFT sharpness, peak count, and peak SNR from the cached FFT magnitude |
| FFT zoom / pan | Browser-local per FFT panel; FFT `Link Zoom` and `Link Pan` controls | Wheel or pinch zoom updates an always-visible `N.N×` badge in that FFT panel; reset returns Show2D's `2.0×` FFT default |
| Viewer chrome preset | `ui_mode` plus explicit `show_*` kwargs | Applies shared display presets; see [Viewer UI controls](viewer-ui) |
| Control visibility | `show_controls`, `controls_collapsed`; `collapse_controls()`, `expand_controls()`, `toggle_controls()` | Permanently remove controls or temporarily collapse them behind the top GUI toggle |
| Title visibility | `show_title` | Top title row shows/hides |
| Stats visibility | `show_stats` | Mean/min/max/std readout shows/hides |
| Panel title visibility | `show_panel_titles`, `panel_title_font_size`, `panel_title_style` | Per-panel labels show/hide, resize, and optionally get title chrome such as background, border, and padding |
| Rich panel title spans | `panel_title_spans` | Optional structured `text` / `math` / `color` spans for symbols such as `λ` and `χ²` in panel titles, panel menus, stats, saved state, and exported HTML |
| Scale bar toggle | `show_scale_bar` (`scale_bar_visible` in saved state) | Calibrated bar shows/hides (needs `pixel_size > 0`) |
| Pan (drag) | per-image pan | Image translates; with `link_pan` all panels move together |
| Zoom (wheel) | `initial_zoom`, `zoom_row`, `zoom_col` | Zooms about the cursor |
| Smooth toggle | `smooth` | Bilinear vs nearest sampling |
| ROI add / drag | `roi_active`, `roi_list`, `roi_selected_idx`; `get_roi_geometries()` | Region overlay; stats panel reports the ROI; saved notebook previews include all ROI overlays and one comparable right-side zoom crop per visible ROI; Python can read circle centers/radii and rectangle/square corners in `(row, col)` coordinates |
| Gallery select | `selected_idx` | Highlights the active panel |
| Local stack slider / play | `panel_frame_indices`, `panel_frame_counts`, `panel_playback_fps`; `set_panel_frame()` | Every grayscale 3D list item gets independent slider/play controls; changing one panel does not move another, and constructor-configured playback speed adds no toolbar clutter |
| Gallery page controls | `page_idx`, `n_pages`, `panels_per_page`, `page_labels`, `page_starred` | Switch, star, or play through panel pages without changing the source stack |
| Panel reorder | `panel_order`; `set_panel_order()`, `move_panel()`, `reset_panel_order()` | Reorders gallery display without changing source data, labels, stars, or hidden state |
| Diff mode | `diff_mode`, `diff_reference` | Panels render as difference vs the reference |
| Link Denoise switch (gallery) | `denoise_scope` | Linked ("all"): denoise edits apply to every panel; unlinked ("panel"): edits apply to the selected panel only |
| Denoise master | `denoise_enabled` | Off shows raw pixels and hides the banner without discarding method, sigma, bin, or per-panel settings |
| Denoise Settings | `show_denoise` | Expands/collapses the Method, sigma, and bin editor without changing whether the effect is active |
| Filter master | `frequency_filter_enabled` | Turns frequency filtering on/off without discarding cutoff or band settings |
| Filter Settings | `show_frequency_filter` | Chooses None, Low-pass, High-pass, or Band-pass and exposes the FFT-ring parameters |
| More menu: Save State | `saved_view_states`, `saved_view_request`, `saved_view_status`; `save_view_state()`, `load_view_state()`, `delete_view_state()`, `clear_view_states()` | Save named lightweight inspection bookmarks, then load/update/delete them without storing another copy of the image data |
| View menu: Crop to view | `view_crop`; `crop_to_view()` | Commits the current viewport as the display extent (single panel, display-only, reversible) |
| View menu: Padding | `pad_ratio`, `pad_ratios`, `pad_fill_mode`, `pad_fill_modes`, `pad_scope`; `set_padding()` | Adds a ratio-based display border with min/median/mean fill; gallery edits can apply to all panels or the selected panel |
| View menu: Reset view | `reset_view_ops()` | Restores the uncropped, unpadded display bit-identically |
| More menu: Flip | `image_flips_horizontal`, `image_flips_vertical` | Display-only horizontal/vertical flips help compare orientations and point-defect neighborhoods without changing stored arrays |
| More menu: Rotate | `image_rotations`, `rotation_scope`; `rotation=`, `rotations=` | Display-only 0/90/180/270° rotation for every panel or the selected panel; raw data and `(row, col)` coordinates stay unchanged |
| Panel inset plots | `inset_plots` | Optional per-panel mini line plots for calibration curves, ACF/R sweeps, or other scientific context; hover reports the nearest plotted coordinate |
| Scale bar placement | `scale_bar_position`, `show_zoom_indicator` | Move the scale bar between bottom corners and hide the zoom badge when an inset plot needs that space |

## Rich panel labels and math

Panel labels can be plain strings, inline math strings, or structured rich
spans. Use this for parameter sweeps where the compact panel chrome should say
`λ`, `χ²`, `σ`, or similar symbols instead of spelling out the word.

```python
from quantem.widget import Show2D

Show2D(
    [raw, residual],
    labels=[
        r"$\lambda=0.03$ raw",
        r"$\chi^2$/pixel residual",
    ],
    show_stats=True,
)
```

The frontend renders common Greek symbols plus simple superscripts/subscripts
without requiring MathJax or KaTeX. It also normalizes doubled backslashes from
JSON/state files, so `\\lambda` is rendered as `λ` rather than as raw markup.
The same rich label path is used by panel titles, the `Panels` menu, the stats
row, and exported standalone HTML.

For mixed color or explicit math spans, use `panel_title_spans`. Each span can
contain `text`, `math`, and optional `color`:

```python
Show2D(
    [raw, filtered, residual],
    labels=["raw", "filtered", "residual"],
    panel_title_spans=[
        [{"math": r"\lambda=0.03"}, {"text": " raw"}],
        [{"text": "filtered", "color": "#34d399"}],
        [{"math": r"\chi^2"}, {"text": "/pixel residual"}],
    ],
)
```

Use `panel_annotations` when the label belongs to a local region inside a panel
instead of the whole panel. An annotation can also use `math` or `spans`, so a
panel can have both a rich title and multiple local callouts:

```python
Show2D(
    [raw, residual],
    panel_title_spans=[
        [{"math": r"\lambda=0.03"}, {"text": " raw"}],
        [{"math": r"\chi^2"}, {"text": "/pixel"}],
    ],
    panel_annotations={
        0: [
            {"math": r"\lambda", "position": "top-left", "variant": "pill"},
            {
                "spans": [{"text": "ROI "}, {"text": "A", "color": "#60a5fa"}],
                "box": [0.18, 0.25, 0.30, 0.16],
                "variant": "callout",
            },
        ],
        1: {"math": r"\chi^2", "position": "top-right", "variant": "outline"},
    },
)
```

## Circle and rectangle overlays

Use `panel_overlays` when the figure needs reproducible geometric callouts
in addition to analysis ROIs. Overlay coordinates use microscope image
coordinates by default: `(row, col)` for centers and `(row0, col0, row1, col1)`
for boxes. The same state is rendered in live Jupyter widgets, saved notebook
state, and standalone HTML exports.

```python
from quantem.widget import Show2D

Show2D(
    [raw, denoised, residual],
    labels=["raw", "denoised", "residual"],
    panel_overlays={
        "raw": [
            {
                "shape": "circle",
                "center": (96, 88),
                "radius": 14,
                "stroke": "#60a5fa",
                "stroke_width": 3,
            },
            {
                "shape": "rect",
                "box": (48, 58, 126, 146),
                "stroke": "#facc15",
                "fill": "#facc15",
                "fill_opacity": 0.12,
                "z_order": 1,
            },
        ],
        "denoised": {
            "shape": "square",
            "center": (96, 88),
            "size": 42,
            "stroke": "#34d399",
            "stroke_width": 2,
        },
    },
)
```

For a shared overlay on every panel, pass `overlays=[...]` instead of
`panel_overlays`. Use `coords="relative"` when the geometry should follow
normalized panel coordinates from 0 to 1:

```python
Show2D(
    [raw, denoised],
    labels=["raw", "denoised"],
    overlays=[
        {
            "shape": "circle",
            "center": (0.5, 0.5),
            "radius": 0.08,
            "coords": "relative",
            "stroke": "#f87171",
        }
    ],
)
```

The live widget and exported HTML also expose `More -> Overlay Edit` when
overlays are present. In edit mode, click a circle or rectangle to select it,
drag inside to move it, drag an edge to resize it, press Delete to remove the
selected overlay, and choose `Reset Overlays` to restore the constructor state.
Use ROIs when the shape should drive statistics, FFT crops, or Python
readback through `get_roi_geometries()`.

## Presentation and export chrome

`ui_mode="presentation"` starts with controls collapsed so the scientific image
gets the first view. It still leaves a compact `Controls` button for adjusting
the widget and an `Export` button for saving the current standalone HTML:

```python
w = Show2D(
    [raw, residual],
    labels=[r"$\lambda=0.03$ raw", r"$\chi^2$/pixel"],
    ui_mode="presentation",
    show_stats=True,
)
w.export_html("lambda_sweep.html", encoding="uint8")
```

In live Jupyter, the Export menu can request exact or uint8 standalone HTML
through the Python backend. In standalone HTML, the Export button downloads the
current embedded page using the representation already present in that file.

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

## Reuse ROI coordinates in notebooks and agents

ROIs are synced in `roi_list` and saved in `state_dict()`. For analysis code,
prefer `get_roi_geometries()` so the shape-specific coordinates are explicit:

```python
w = Show2D(img).set_roi(row=72, col=65, radius=12)
roi = w.get_roi_geometries()[0]
roi["center"]  # {"row": 72.0, "col": 65.0}
roi["radius"]  # 12.0
```

Rectangle and square ROIs include clockwise `corners`; annular ROIs include
`radius_inner` and `radius_outer`. Every ROI also includes `bounds` and
`bounds_clipped`, which lets an agent crop an array safely while preserving the
user's exact visible selection.

## Save microscope-stage view states

Show2D can store named lightweight inspection bookmarks. This is useful when a
scientist explores a notebook interactively — for example drawing an ROI around
a defect, turning on FFT, hiding panels, changing contrast, adding padding for a
drift check, and then wanting to come back to exactly that view later without
writing export code.

The GUI path is **More → Save State**. Existing states can be loaded, updated,
deleted one at a time, or deleted all at once. The Python API uses the same
state model:

```python
w = Show2D([raw, corrected, residual], labels=["raw", "corrected", "residual"])
w.add_roi(row=72, col=65).set_padding(0.2, fill="median")
w.show_fft = True

w.save_view_state("defect A drift check")
w.hidden_panels = [2]
w.save_view_state("raw vs corrected")

w.load_view_state("defect A drift check")
w.delete_view_state("raw vs corrected")
w.clear_view_states()
```

Saved view states include UI/view traits such as ROI list, selected ROI,
selected panel, hidden panels, panel order, local stack frame indices, contrast,
colormap, FFT, denoise/filter, crop, padding, zoom/view box, and scale-bar
settings. They do **not** store another copy of `frame_bytes`,
`panel_stack_bytes`, or raw source arrays. `state_dict()` includes the named
bookmarks so a saved notebook can reopen with the same list of microscope-stage
positions.

## Add scientific inset plots to panels

Advanced figure-making sometimes needs a compact plot inside each image panel:
for example an autocorrelation score versus an `R` ratio while reviewing
automatic denoising calibration. Use `inset_plots` for that per-panel context.
The top-level parameter stays singular and readable, while each dictionary keeps
the plot data, placement, labels, and style together.

```python
Show2D(
    denoised_panels,
    labels=["candidate A", "candidate B", "candidate C"],
    inset_plots=[
        {
            "x": r_values,
            "y": acf_values_for_panel,
            "point": (best_r, best_acf),
            "xlabel": "R",
            "ylabel": "ACF",
            "legend": "ACF/R",
            "annotation": f"R*={best_r:.2f}",
            "position": "bottom-right",
            "margin": 24,
            "size": 0.36,
            "height": 0.26,
            "show_ticks": True,
            "xticks": [0, 1],
            "yticks": [0, 1],
            "line_width": 2.6,
            "background_alpha": 0.58,
            "border_width": 0,
        }
        for acf_values_for_panel, best_r, best_acf in calibration_results
    ],
    scale_bar_position="bottom-left",
    show_zoom_indicator=False,
)
```

For normal notebooks, prefer `position` plus `margin`, `size`, and `height`.
Use `box=[left, top, width, height]` only when a publication figure needs exact
normalized panel coordinates. `border_width=0` gives a clean no-frame look;
increase `background_alpha` or use a subtle `border_color` when the underlying
image is noisy. On hover, the live widget reports the nearest plotted
coordinate using the axis labels, such as `R 0.465 · ACF 0.48`.

If `inset_plots` is present, the live widget shows an `Inset Chart` switch in
the `More` menu. A scientist can hide the chart during image inspection, turn
it back on for reporting, or drag the chart inside the panel. Dragging previews
freely and then snaps to the closest corner on release, updating `position` and
`margin` in the widget state.

## Remove a background or isolate a periodicity

Frequency Filter is deliberately separate from Denoise. Denoise promises a
cleaner view of the measurement; High-pass and Band-pass deliberately remove
real signal, so their displayed pixels must not be treated as measurable raw
counts. The stored array, statistics row, and raw exports remain unchanged.

```python
# Flatten a slow brightness gradient.
Show2D(eds_map, show_fft=True,
       frequency_filter="highpass", frequency_filter_cutoff=0.08)

# Isolate a lattice-frequency ring.
Show2D(lattice, show_fft=True,
       frequency_filter="bandpass",
       frequency_filter_center=0.32,
       frequency_filter_width=0.08)
```

Parameters are fractions of Nyquist from 0 to 1. Drag the cyan FFT ring to
select the cutoff or band center by eye. The FFT overlay dims rejected
frequencies: Low-pass keeps the clear area inside the ring, High-pass keeps the
clear area outside, and Band-pass keeps only the clear annulus. Filter and its
More-menu toggle are off by default; turning the master on with no remembered
mode starts with Low-pass. Denoise and Filter are chainable in
the fixed order Denoise then Filter; each active operation has its own banner.
Turn the Filter master off or choose None to restore the unfiltered view
without losing the settings.

For paged galleries, FFT work is demand-driven: only panels visible on the
current page are transformed. Visiting another page computes that page once
and retains the bounded cache for later return visits.

## Crop and pad the view (advanced)

Single-panel widgets can commit the current browser viewport as the display
extent. Single-panel and gallery widgets can add a ratio-based display border,
either from the toolbar's **View** menu or from Python:

```python
w = Show2D(image, view_box=(64, 64, 96))  # zoom into a feature
w.crop_to_view()          # the 96x96 window becomes the displayed frame
w.set_padding(0.1, fill="median")  # 10% border on each side
w.reset_view_ops()        # full frame again, bit-identical
```

For a drift-correction gallery, use per-panel padding to compare candidate
margins without rebuilding another notebook cell:

```python
w = Show2D([raw, corrected, residual], ncols=3)
w.set_padding(0.20, fill="mean", panels=[1])  # only corrected panel
```

Both ops honor the display-only contract:

- The stored array is never modified; `reset_view_ops()` returns the exact
  original frame bytes.
- The crop applies in the display pipeline **before** denoise, so an active
  denoise operates on the cropped region; the pad applies after it and uses the
  selected fill mode (`min`, `median`, or `mean`).
- The stats row keeps reporting the full raw data, and cursor coordinates
  remain full-image (row, col) while a crop or pad is active: the crop is a
  display window, not a new coordinate system.
- The histogram and canvas are repacked from the padded display frame, so the
  border contribution is visible while the raw data remain unchanged.
- An active crop or pad is never silent: a one-line `view:` banner names the
  window and the ratio, e.g.
  `view: cropped to (64,64)-(160,160) · pad 10% median (reset_view_ops() restores full frame)`.
- Both persist through `state_dict()` / `load_state_dict()`.

Crop-to-view remains single-panel in this release because it changes the
coordinate origin. Padding is gallery-safe: panels share the largest padded
canvas so comparisons stay aligned, while each panel can still have its own
ratio/fill mode.

## FFT quality labels

Pass `show_fft=True` to show the FFT panel. By default, `fft_metrics=True`
adds a small white label inside each FFT panel with three quick checks:
sharpness, peak count, and peak SNR. These values are computed from the FFT
magnitude already used for rendering, so the label does not trigger a second
FFT. Set `fft_metrics=False` when a clean FFT image is more important than the
readout.

The first FFT for an image, local panel frame, ROI, or window configuration may
take a moment on large data. After that, Show2D reuses that bounded cache entry
for redraws, zoom/pan, contrast changes, metric labels, and return visits to the
same slice. During a first-time slice computation, the previous valid FFT stays
visible instead of flashing to a dark loading panel.

Each interactive FFT panel also carries its own bottom-left zoom multiplier.
It starts at `2.0×`, follows wheel and touch-pinch zoom immediately, and returns
to `2.0×` on double-click, double-tap, or Reset. In a gallery, unlinked FFTs
report independent values; enabling FFT `Link Zoom` makes their values move
together without changing the real-space zoom.

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

## Independent local stack panels

A bare 3D array still means a static gallery of `N` images. To give gallery
panels their own frame or slice axes, pass a Python list whose 3D items are each
shaped `(frames, rows, cols)`. Every grayscale 3D list item becomes one panel
with its own slider, play/pause control, frame count, and current frame. Counts
may differ between panels. RGB(A) list items keep the color-image behavior
described in the constructor reference.

This is useful for tomography or multislice studies that produce several
reconstruction volumes. The scientist can keep four methods, regularization
settings, or slice thicknesses side by side, stop each one at the depth where a
feature or artifact is clearest, and retain linked spatial context without
forcing all panels onto one global slice index:

```python
from quantem.widget import Show2D

reconstructions = [
    recon_8_slices_20A,
    recon_16_slices_10A,
    recon_regularized,
    recon_alternative,
]

w = Show2D(
    reconstructions,
    labels=[
        "8 slices x 20 A",
        "16 slices x 10 A",
        "regularized",
        "alternative",
    ],
    panel_frame_indices=[3, 7, -1, 0],
    panel_playback_fps=4,
    ncols=2,
    link_zoom=True,
    link_pan=True,
)
w.set_panel_frame("16 slices x 10 A", 8)
w
```

Use a list even when all reconstructions happen to have the same slice count;
the outer list is what declares independent local stacks. Use Show3D instead
when all panels should advance together on one global frame or slice axis.

The list may also mix stack panels with ordinary 2D images:

```python
w = Show2D(
    [eds_sum_map, haadf_stack, ti_map, o_map],
    labels=["EDS sum", "HAADF", "Ti", "O"],
    panel_frame_indices=[0, -1, 0, 0],
    ncols=2,
    auto_contrast=True,
)
w
```

Only the HAADF panel gets an in-panel slider and play button. Each local frame
index remains keyed to its source panel when panels are hidden or reordered.
Click a stack panel and use the left or right arrow key to scrub it. From
Python, use `w.set_panel_frame("HAADF", -1)`.

Set `panel_playback_fps` when constructing the widget to choose the cadence for
all local-stack play buttons without adding another compact control. The
default is 10 fps; values above the 30 fps browser budget are capped. The
setting is saved in widget state and carried into standalone HTML exports,
while the per-panel playing/paused state remains browser-local so reopening a
report does not start playback unexpectedly.

The active panel's histogram, stats, and FFT follow its current local frame;
scrubbing it does not change neighboring panels. `state_dict()` records
`panel_frame_indices`, and single HTML exports using either `encoding="full"`
or `encoding="uint8"` contain every local frame. For an interactively restored
notebook output, construct with `save_state=True`; the default
`save_state=False` deliberately stores only a compact static preview rather
than embedding the stack payload in the notebook.

For single-image ROI review, that compact saved-notebook preview keeps the
scientific marks visible: the fallback PNG draws every visible ROI on the full
image and adds one right-side zoom crop per ROI. The zoom crops use a common
crop size where image boundaries allow it, so multiple ROIs remain visually
comparable. A reader who opens the notebook or report without rerunning the cell
can still see the marked features.

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
    page_size=20,
    title="Live HAADF images",
)
w
```

`watch=True` is the default. Pass `watch=False` for a fixed folder or a
reproducible script that should update only when you call `poll_folder()`.
Folder galleries default to `page_size=20`: page controls appear automatically
when the 21st ready image arrives. Use another positive integer to change the
visible limit, or `page_size=None` to keep one unpaged gallery. The final page
contains only real files, so 45 images form pages of 20, 20, and 5 panels.
Folder pages keep selection, stars, hidden state, and ordering keyed to each
source file rather than sharing a hidden slot across pages.

Change the grouping later without rebuilding the widget:

```python
w.set_folder_page_size(50)   # regroup the same source panels
w.set_folder_page_size(None) # show one unpaged gallery
```

Paging limits the panels, histograms, and FFT views rendered at one time. The
current folder implementation still retains the complete full-resolution
gallery in Python and transports its display previews to the browser; it is not
yet a lazy, bounded-memory source-page cache. Use `ShowFolder` for lightweight
thumbnail discovery when the full scientific arrays need not all be opened.

An empty watched folder remains mounted with a waiting view and becomes the
real gallery in the same widget model when its first stable file arrives.
The compact title-area badge reports `Watching`, `Updating`, `Waiting for file
completion`, `Watch error`, or `Stopped`; fixed `watch=False` snapshots do not
show the badge.

```python
new_panels = w.poll_folder()       # scan now; return newly appended indices
w.stop_folder_watch()             # pause background scans
w.watch_folder(interval=1.0)      # resume with a different interval
w.close()                         # stop watching and close the widget
```

Folder watching is append-only. A file already represented in the gallery is
not duplicated, an incomplete file is deferred until a later poll, and removing
or rewriting a source file does not silently remove or replace an existing
panel. An incompatible shape is reported without blocking a later compatible
file. Close long-running widgets when the notebook no longer needs them.

`Show2D.from_folder(...)` reads the scientific image data at its source
resolution. It is different from `ShowFolder`, which intentionally uses small
cached thumbnails for fast folder discovery and selection. Use `ShowFolder`
to decide what to open; use `Show2D.from_folder(...)` when the displayed pixel
data and live panel append behavior matter.

Maintainer real-time signoff follows
[S2D-18](../maintainer/storyboard-show2d.md#s2d-18-watch-a-live-emd-folder-in-place):
add genuine EMD files after the widget is visibly mounted and verify the same
browser canvas paints each full-resolution panel.

## Paged galleries

Direct `Show2D(...)` pages are comparison pages. Use them when each view
contains the same panel grid across several
analysis settings, iterations, or parameter values. A common example is a
4-by-4 reconstruction sweep where each page is one iteration or one denoising
parameter, and the panels within the page are the related output images.

These repeated-slot comparison pages are distinct from the sequential item
pages created by `Show2D.from_folder(..., page_size=20)`.

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
