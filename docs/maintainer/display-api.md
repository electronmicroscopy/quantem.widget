# Display API naming standard

Use the same words for the same display choices across QuantEM, quantem.gpu,
quantem.widget, and their tutorials. Scientists should choose image size and
layout without having to calculate figure dimensions or know the renderer.
This is the target contract for new APIs and reviewed migrations; it does not
mean that every existing function already accepts these keywords.

## Common options

| Option | Meaning | Convention |
| --- | --- | --- |
| `size` | Display size of each image panel | `"medium"` by default; `"small"`, `"medium"`, `"large"`, `"xlarge"` |
| `columns` | Images across a gallery | Usually specify this alone; derive rows automatically |
| `rows` | Images down a gallery | Optional alternative when row layout matters |
| `cmap` | Colormap | Use the existing scientific plotting name |
| `norm` | Display intensity transformation | Reuse the viewer's supported normalization choices |
| `vmin`, `vmax` | Display contrast limits | In the documented image-value units; do not modify the data |
| `title` | Image or panel label | Preserve meaningful labels in saved images |
| `save` | Path for a static figure export | Optional convenience on APIs that support static export |

These are shared names, not a requirement to add every option to every method.
Only expose a control when the operation supports it. Keep scientific data
saving distinct from exporting a display. Reuse existing viewers and return
types; a naming cleanup does not justify a new wrapper or result class.

## Size means one panel

For Show2D and Show3D, the image-panel width targets are:

| `size` | Panel width |
| --- | ---: |
| `"small"` | 300 CSS px |
| `"medium"` | 450 CSS px |
| `"large"` | 600 CSS px |
| `"xlarge"` | 900 CSS px |

The shared source is `src/quantem/widget/panel_sizes.json`, resolved by
`resolve_panel_width` in `src/quantem/widget/utils/ui.py`. Use this resolver
inside the widget package rather than adding a separate map to each viewer.
Other packages should use their existing visualization layer; do not introduce
a widget dependency into a computational backend just to import size constants.

Preserve aspect ratio. Two medium panels request two panel widths, plus their
controls and spacing; medium does not mean 450 px for the entire gallery.
Responsive layouts may shrink or wrap to fit the container. Keep font and
control dimensions on the shared UI tokens rather than inventing proportional
font sizes for each image preset.

Static views should use the same nominal per-panel targets through their
visualization layer. Figure inches and DPI are renderer details: at a 100-DPI
reference, 450 px corresponds to 4.5 inches. Titles, margins, export DPI and
notebook scaling mean that figure dimensions alone do not guarantee a 450-px
visible image. Check the rendered panel before claiming visual parity.

Beginner examples use `size="medium"`, never require `axsize=(6, 8)` or
`figsize=(6, 8)`, and do not calculate inches, pixels or DPI in notebook cells.
Existing precise publication-layout overrides can remain advanced options.
Reject conflicting explicit overrides during a migration rather than silently
discarding one; retain previously documented behavior until that migration.

Display size must not change the source data, reconstruction sampling,
precision, fitting, or exported scientific arrays. Do not trade scientific
quality for a smaller display. This standard does not change Show3DSlices,
Show4DSTEM or ShowPtycho's existing size behavior; those need separate review.

## Rows and columns

Use complete words, `rows` and `columns`, for new public layout controls.
Avoid introducing `nrows`, `ncols`, `num_cols` or `max_cols` as another naming
scheme for the same gallery operation. This does not rename scientific
coordinates: array coordinates remain `(row, col)`.

- With `columns=2`, place up to two images across and calculate the required
  rows. Five images produce three rows, with one image in the last row.
- With only `rows=2`, calculate the columns needed. Five images produce a
  two-row, three-column layout. Do not add `rows` to a simple helper unless
  users actually need this alternative.
- With both explicitly supplied, validate that `rows * columns` can hold all
  images. Raise a helpful error if it cannot; never drop images silently.
- Use row-major order. Do not create extra image panels just to fill unused
  slots. For a one-direction request, a smaller image count can use fewer
  rows or columns than requested.
- Accept positive integers; reject booleans, zero, negatives and fractions.
  Invalid values should say how to correct the call, for example
  `columns must be a positive integer; use columns=2`.

New comparison galleries default to two columns with automatic rows. When an
API supports both directions, distinguish omitted layout arguments from
explicit ones internally so a default column count cannot conflict with an
explicit `rows` request. A rename keeps the layout defaults unless a behavior
change is separately approved.

## Short calls first, detailed control when needed

The detector helper supports this short call:

```python
detector.show(data, size="medium", columns=2)
```

Its export options remain on the same method:

```python
detector.show(
    data,
    size="large",
    columns=2,
    cmap="inferno",
    save="detector-images.png",
)
```

Advanced users can retain arrays with `detector.bf`, `detector.df`,
`detector.adf` and `detector.mean`, then use the existing static or interactive
viewer. Per-image colormaps and histogram limits belong to that viewer's
existing panel controls. Document whether contrast is linked across panels;
do not add a parallel detector-specific histogram API.

## Renames and review

| API | Surface |
| --- | --- |
| `detector.show` | `size`, `rows`, `columns`, `cmap`, `save`; two columns by default |
| `Show2D` | `size`, `columns`, `rows`; three columns by default |
| `Show3D` | `size`, `columns`, `rows`; four columns by default |
| `Show4DSTEM` | `size`, `compare_columns` |

A rename has no transition: no alias, no deprecated keyword, no reader of the old
key. It reaches every caller in every repository at once; a saved record gets a
schema bump, one rewrite script that keeps `<name>.before`, and a CHANGELOG
"Renamed" table. Update docstrings, examples, errors and agent guidance in the
same change. Test defaults, invalid values, partial last rows and saved output. Verify representative static and interactive layouts, including
a narrow viewport, and confirm display options leave scientific arrays intact.
Keep design documentation distinct from evidence of a completed runtime change.

### Detector and direct ptychography display options

Use `detector_fit.show(size="medium", cmap="inferno", save="detector-fit.png")`
for the fitted detector disk. It has one panel, so no rows/columns controls.
Use the same keywords on `detector.show(data, columns=2)` for the detector gallery.

`ptycho.show("parameters", rows=2, cmap="viridis", save="fit.png")` lays out
the two static fit landscapes. Focus views accept `size`, `rows`, `columns`
and `cmap` in both Show2D/Show3D and static mode. Export with
`ptycho.show("focus", mode="static", save="focus.png")`; native image density
is retained. `save` exports the figure, while ShowPtycho
`save_dir` stores scientific calibration. Plain `ptycho.show()` retains
ShowPtycho's responsive layout and its existing Color control.

## Automatic output sampling

Auto chooses 1×, 2×, 3× or 4× from the acquisition calibration, with a
maximum of 4×. Select 8× explicitly with `ptycho.show(upsample=8)` or the
ShowPtycho Upsampling menu. For a standalone reconstruction, use
`ptycho.run(fit, upsample=8)`.

The nominal bandwidth requirement remains separate from the selected grid.
When Auto is capped, the fit report and viewer settings identify the larger
requirement; a 4× grid does not imply full nominal bandwidth. Fitting and
source data are unchanged. Saved Auto fits reopen under the current cap;
saved explicit 8× choices retain 8×.
