# ShowPDF

`ShowPDF` is an interactive front end to
`quantem.diffraction.PairDistributionFunction` for electron pair distribution
function (ePDF) analysis of 4D-STEM data.

The left panel shows a navigation image of the scan, on which you select scan
positions: a painted mask, a square probe, or a line band. The right panel
shows one curve of that selection: the azimuthal mean I(k) with its fitted
background B(k), the windowed reduced structure factor F(k), the reduced PDF
G(r), or the PDF g(r).

In a notebook, every curve comes from the library's `calculate_Gr` and
`calculate_gr`. When a setting changes in an exported HTML page, the page
recomputes the curves in the browser, repeating the library's steps in
JavaScript.

```python
from quantem.widget import ShowPDF

w = ShowPDF(
    dataset,                        # Dataset4dstem, array, or PairDistributionFunction
    k_min_fit=0.3,                  # 1/Å
    density_value=0.0528,           # atoms/Å³, for g(r) and the damping
    damp_origin_oscillations=True,
    plot_mode="gr",
)
w
```

## Requirements

ShowPDF needs quantem's `PairDistributionFunction` from
electronmicroscopy/quantem pull request 248 (`calculate_Gr` with `density` and
`max_iter`). Until the next quantem release, it is on quantem's
[dev branch](https://github.com/electronmicroscopy/quantem/tree/dev).
Importing ShowPDF works without it; constructing one raises an `ImportError`
that names the requirement.

quantem core does not install on an Intel Mac (it needs torch 2.7 and
NumPy 2) or on Windows on ARM (it needs hdf5plugin 6), so ShowPDF is not
available there. See [Installation](../install.md).

## Input

`data` is a `Dataset4dstem`, a 4D array
`(scan_row, scan_col, det_row, det_col)`, a single 2D pattern (a `Dataset2d`
or 2D array, analyzed as a 1 × 1 scan), or a `PairDistributionFunction`.

Calibrate the diffraction axes in 1/Å. k is 1/d (the 2π enters in the sine
transform); an uncalibrated dataset gives k in detector pixels.

Any other input goes through `PairDistributionFunction.from_data`, with the
constructor's `find_origin`, `origin_row`, `origin_col`, `num_annular_bins`,
`radial_min`, `radial_max`, `radial_step`, `two_fold_rotation_symmetry` and
`device` (the CPU unless given). For other `from_data` options, such as
`ellipse_params`, build the `PairDistributionFunction` yourself and pass it
in. ShowPDF computes with the object you pass and replaces its results.

The navigation image is a bright-field virtual image of the data unless
`nav_image` is given. Its scale bar uses the dataset's scan calibration, or
`pixel_size` (Å).

`set_data(data)` swaps in a new dataset. It keeps the parameters where they
still fit the new k range and resets the mask, probe, and line.

## Number density

ShowPDF never estimates the number density ρ₀; set `density_value` in
atoms/Å³. From a mass density ρ in g/cm³ and a mean atomic mass M in g/mol,
ρ₀ = 0.6022 ρ / M. A structure model gives it too.

G(r) does not need the density. g(r) = 1 + G(r) / (4π r ρ₀) and the low-r
correction (`damp_origin_oscillations`) do. While `density_value` is 0 or
less, g(r) stays empty, the damping is skipped, and the status line asks for
the density when either is requested.

## Parameters

The widget passes these to the library's `calculate_Gr` (k in 1/Å, r in Å):

| Trait | `calculate_Gr` argument | Default | Meaning |
|---|---|---|---|
| `k_min_fit`, `k_max_fit` | `k_min_fit`, `k_max_fit` | the data's k range | Range of the background fit B(k). |
| `k_min_window`, `k_max_window` | `k_min_window`, `k_max_window` | 0 | Range of the window on F(k). An end at or below 0 follows the fit range, as `None` does in the library. |
| `k_lowpass`, `k_highpass` | `k_lowpass`, `k_highpass` | 0 (off) | Widths of Gaussian filters on F(k). Both together make a band-pass, and a highpass above the lowpass is an error. |
| `r_max`, `r_step` | `r_max`, `r_step` | 20.0, 0.02 | The r grid: from 0 in steps of `r_step`, below `r_max`. |
| `damp_origin_oscillations` | `damp_origin_oscillations` | `False` | Low-r correction: corrects S(k) so that G(r) follows −4π ρ₀ r on [0, `r_cut`]. It runs only once `density_value` is set. |
| `r_cut` | `r_cut` | 0.8 | Upper end of the correction. Keep it below the nearest-neighbour distance. |
| `max_iter` | `max_iter` | 6 | Maximum number of iterations of the correction. |
| `density_value` | `density` | 0 (not set) | Number density ρ₀ in atoms/Å³, also passed to `calculate_gr`. |
| the selection | `mask_realspace` | the full scan | The scan positions averaged into I(k). |

The window is zero outside its range. It rises from 0 as sin² over the lowest
tenth of the range, and it is multiplied by the Lorch factor
sin(πk/k_max) / (πk/k_max), which falls to 0 at the window's upper end k_max.

The k window slider shows the window in effect. Moving one of its ends sets
that end, which then stops following the fit range; an end you leave in place
keeps its setting. Set `k_min_window` or `k_max_window` to 0 to make that end
follow the fit range again.

With damping on, the F(k) and G(r) plots show the corrected curves that the
transform used, and g(r) comes from the corrected G(r). g(r) is clamped at 0
(`set_pdf_positive=True`).

## Status line

The status line next to the title shows these hints:

| Status | When |
|---|---|
| "Set the density to apply damping and compute g(r)." | Damping is on and the density is not set. |
| "Set the density to compute g(r)." | g(r) is shown, damping is off, and the density is not set. |
| "The k window contains no k points; set its lower end below its upper end, more than one k step apart." | The window has no positive weight. |

The window hint takes precedence over the density hints, and an error message
(starting with "Error:", "g(r) error:" or "Line-scan error:") over both.

A window has no positive weight when its lower end is at or above its upper
end (after an end at or below 0 resolves to the fit range), or when no k
point lies strictly between its ends. F(k) and G(r) are then zero, to float32
rounding, and g(r) is 1.

When a computation fails, the curve plot keeps its last curves if only a
parameter changed. After a new selection or dataset, and in a line scan, it
is empty.

## Selecting scan positions

The mask, probe, and line buttons set `analysis_mode`. The count under the
image shows how many scan positions the curves average.

### Mask

Paint on the image with the ▭ (Rectangle), ○ (Circle), or ✎ (Freeform)
tool. In + mode, the default, painting excludes positions (shown red); in −
mode, painting includes them again.

Switching between + and − resets the whole mask: to all positions included
for +, and to all excluded for −. To analyze one region, switch to − and paint
it. Clear includes every position, and Invert inverts the mask and swaps + and
−.

From Python, `set_mask(mask)` takes a boolean `(scan_row, scan_col)` array in
which True includes a position, and switches to mask mode; `clear_mask()`
clears the mask. A mask that includes every position, or none, averages the
full scan.

### Probe

The probe is a square of 2 · `probe_size` − 1 positions per side, centred on
`(probe_row, probe_col)`, the scan centre by default. Click or drag on the
image to move it; the curves follow.

### Line

Drag on the image to draw a line from A to B, drag an end to move it, or drag
the line to shift it. From Python,
`set_line(row0, col0, row1, col1, width=None)` sets it in `(row, col)` scan
coordinates and switches to line mode; `clear_line()` removes it.

The band holds the positions within `line_width` / 2 of the line, with flat
ends. `line_width` is an odd number of scan pixels (1, 3, 5, ...): away from
the scan edge, a horizontal or vertical band of width w covers exactly w rows
or columns. The Width slider offers the odd widths up to 99 and extends to a
wider one set from Python.

The ends of the sampled line snap to the nearest scan positions (halves round
up), and a line that leaves the scan is clipped at its edge. With ⟂
(`line_perpendicular`), the widget samples the perpendicular bisector of the
drawn line instead (same midpoint and length, turned 90°), so a line traced
along a feature scans across it.

`line_mode="averaged"` (avg) gives one PDF from the whole band;
`line_mode="linescan"` (scan) gives one PDF per position along the line.

## Line scans

A line scan has a position at every scan pixel along the line's longer axis,
evenly spaced from A to B, up to `linescan_max_bins` positions (256 by
default).

Each scan position in the band is averaged into its nearest line-scan
position. The two end positions cover half a step, so they average fewer scan
positions. Each position's curve is what `calculate_Gr` gives for a mask of
its scan positions.

The stack holds the curve of the current plot mode, so changing the plot mode
recomputes it; a g(r) stack needs the density.

`linescan_view="heatmap"` shows every position, with A at the bottom and a
colorbar. `"waterfall"` stacks an evenly spaced subset of the positions that
ends at B, as close to `waterfall_count` curves (10 by default) as even
spacing allows. Each curve has the color of its position, with a matching dot
on the scan line.

## Saving results

| Method | Writes |
|---|---|
| `save_results(path, overwrite=False)` | A zip of the curves and settings (files below). `.zip` is added when missing, and an existing file raises `FileExistsError` unless `overwrite=True`. |
| `save_image(path, plot_mode=None)` | One curve as PNG, PDF, or TIFF, chosen by the file extension (or `format=`). It raises `ValueError` for a line scan, which has a curve per position, and for g(r) without a density. |
| `save(path)` | The settings, the selection, and the painted mask as JSON, which `ShowPDF(data, state=path)` restores. |
| `export_html(path=None, encoding="full")` | A standalone HTML viewer (HTML export, below). |

The results zip holds:

| File | Contents |
|---|---|
| `README.txt` | What each file holds and how to load it |
| `metadata.txt` | Mode, selection, and parameters, as plain text |
| `curves_k.csv` | Columns k, Ik, bg, Fk, where bg is B(k) |
| `curves_r.csv` | Columns r, Gr, gr; gr is empty without a density |
| `mask.npy` | The boolean scan mask, in mask mode only |
| `linescan_position.csv` | Line scans: each position's index and distance from A in scan pixels |
| `linescan_Ik.csv`, `linescan_Ik_bg.csv`, `linescan_Fk.csv`, `linescan_Gr.csv`, `linescan_gr.csv` | Line scans: an axis row (k or r), then one row per position; `linescan_gr.csv` only with a density |

F(k) and G(r) are the curves the transform used, corrected when the damping
ran. Missing values are empty CSV cells. In a line scan, `curves_k.csv` and
`curves_r.csv` hold the average of the whole band, and saving before a line
is drawn raises `RuntimeError`.

`state=` also takes the dict of `state_dict()`, and `load_state_dict` restores
one into a live widget. Keys it does not know are ignored, a mask saved for
another scan shape is dropped, and a rejected value leaves the widget
unchanged.

A saved NaN or infinite `density_value` loads as not set (0). A saved even
`line_width` loads as the next odd width, with a warning, and a width below 1
loads as 1.

## Saved notebooks

A saved notebook keeps the widget's settings and curves and a static picture of
it: the scan image beside the shown curve, or the line-scan stack. It leaves out
the scan image itself, the mask and the line-scan stack (`save_state=False`, the
default), so the file stays small at any scan size. Reopened without a kernel,
the widget shows the picture; rerunning the cell brings the interactive widget
back. With `save_state=True` the notebook keeps everything, and a reopened
notebook shows the full widget, although only a kernel, or an exported page,
recomputes its curves. `export_html` always embeds the scan image, mask and
stack.

## HTML export

`export_html(path=None, *, title=None, encoding="full")` writes one HTML file
that opens in a browser without a Python kernel; the Export HTML button builds
the same file and downloads it. Only the synced traits are embedded, never the
raw 4D-STEM data. The page loads the Jupyter widget manager from public CDNs,
so opening it needs a network connection.

| `encoding` | Export HTML menu item | In the exported page |
|---|---|---|
| `"full"` | HTML exact float32 curves, fixed selection | The k-space, damping, and density controls recompute the curves from the exported I(k). The mask, probe, and line are fixed. |
| `"uint8"` | HTML + quantized uint8 profiles, live selection | Also embeds every scan position's radial profile at 8 bits per k bin, so the mask, probe, and line stay interactive too. |

A `"full"` export of a line scan locks its parameter and plot-mode controls,
because a new stack needs the profiles. A `"full"` export made when no curves
were computed for the shown selection (its compute failed) keeps its parameter
controls locked too and shows the baked error.

A `"uint8"` export opens with the exact exported curves. Once the selection,
a parameter, or the plot mode changes, it recomputes them from the 8-bit
profiles.

The default file name is `<title>_<rows>x<cols>_exact.html` in the working
directory, or `_quantized.html` for `"uint8"`. `mode` accepts only `"single"`
and `downsample` only `None` or 1.

## Display options

| Trait | Default | Effect |
|---|---|---|
| `plot_mode` | `"Gr"` | The curve shown: `"Ik"`, `"Fk"`, `"Gr"`, or `"gr"` |
| `show_background` | `True` | Draws B(k) on the I(k) plot |
| `cmap` | `"gray"` | Colormap of the navigation image; the menu offers gray, inferno, viridis, and magma |
| `log_scale` | `False` | Logarithmic navigation image (the I(k) Log switch is separate and not saved) |
| `auto_contrast` | `True` | Navigation image contrast from the 2nd to the 98th percentile, else from the minimum to the maximum |
| `show_stats` | `True` | Shows the included-position count under the image and the value readout under the plot |
| `show_controls` | `True` | Shows the parameter controls and the brush, probe, line, and Cmap controls |

With `show_controls=False`, these stay: the mask, probe, and line buttons, the
mask tools (▭ ○ ✎, + / −, Clear, Invert), the plot-mode buttons, the plot
toolbar (Log, Background, Heatmap / Waterfall, N, Reset), and Export HTML. A
badge next to the title then lists the damping, k filters, or perpendicular
line sampling in effect, for example "Damped · Lowpass 0.020".

ShowPDF does not take the shared `ui_mode`, `show_title`, or
`controls_collapsed` names of the [Viewer UI controls](viewer-ui).

## Errors and limits

The float parameters (`k_min_fit`, `k_max_fit`, `k_min_window`,
`k_max_window`, `k_lowpass`, `k_highpass`, `r_max`, `r_step`, `r_cut`,
`density_value`) and the line ends (`line_row0`, `line_col0`, `line_row1`,
`line_col1`) must be finite numbers, and the line ends must lie within ±1e9
scan pixels. Any other value raises `TraitError` from the constructor, an
assignment, or `load_state_dict`, and `ValueError` from `set_line`. An
integer too large for a float (such as `10**400`) raises `OverflowError` from
the constructor or an assignment.

The integer options (`max_iter`, `probe_row`, `probe_col`, `probe_size`,
`line_width`, `linescan_max_bins`, `waterfall_count`) must be whole numbers;
2.0 is accepted, 2.5 is not truncated. `line_width` must also be odd,
`linescan_max_bins` and `waterfall_count` at least 1, and `linescan_view`
"heatmap" or "waterfall". Other values raise `TraitError` from the
constructor, an assignment, or `load_state_dict`. A keyword that is neither a
parameter nor a trait raises `TypeError`.

A NaN or infinite `pixel_size` raises `ValueError`, as does a dataset whose k
axis is not finite (or too large for float32); check the diffraction
calibration. `export_html` raises `ValueError` for a NaN or infinite float
trait, which is not valid JSON in the page, and writes no file.

## Reference

```{eval-rst}
.. autoclass:: quantem.widget.showpdf.ShowPDF
   :members:
   :show-inheritance:
```

## Interactive controls

Each control mutates the listed synced trait; a view-only control changes
only the display. A UI-test agent acts on the control, then asserts the trait
changed and the canvas repainted (non-zero, no console error, no NaN frame).

| Control | Trait | Expected effect |
|---|---|---|
| mask / probe / line buttons | `analysis_mode` | Switches how scan positions are selected; the curves recompute |
| ▭ ○ ✎ buttons (Rectangle, Circle, Freeform), then a drag on the image | `mask_tool`, `mask_b64`, `mask_version` | Paints the mask; the curves recompute on release |
| + / − button (key X) | `mask_b64`, `mask_version` | Switches between excluding and including paint, resetting the whole mask |
| Clear (key C), Invert (key I) | `mask_b64`, `mask_version` | Includes every position, or inverts the mask (the button also swaps + and −) |
| Brush slider | `mask_brush_size` | Freeform brush of 2·size − 1 positions per side, 1 to 20 |
| Probe click or drag | `probe_row`, `probe_col` | Moves the probe; the curves follow |
| Probe slider (on release) | `probe_size` | Probe square of 2·size − 1 positions per side, 1 to 100 |
| Line drag: draw, move an end, or move the line | `line_row0`…`line_col1`, `line_active`, `line_version` | Sets the line; an averaged line recomputes during the drag, a line scan on release |
| Width slider (on release) | `line_width` | Odd band widths from 1 to 99, or up to a wider one set from Python |
| ⟂ button | `line_perpendicular` | Samples along the perpendicular bisector of the drawn line |
| avg / scan buttons | `line_mode` | One averaged PDF, or a line scan |
| Clear button (line mode) | `line_active`, `line_row0`…`line_col1`, `line_version` | Removes the line |
| Cmap menu | `cmap` | Navigation image colormap |
| Image wheel, middle or Alt drag | view only | Zooms (0.5× to 32×) or pans the navigation image |
| I(k) / F(k) / G(r) / g(r) buttons (keys 1–3 for the first three) | `plot_mode` | Selects the curve; a line scan recomputes its stack |
| Log switch | view only | Logarithmic I(k) axis, on by default (I(k) mode) |
| Background switch | `show_background` | Shows or hides B(k) (I(k) mode) |
| Heatmap / Waterfall buttons, N field | `linescan_view`, `waterfall_count` | Line-scan rendering and number of stacked curves (line-scan mode) |
| Reset button (key R also resets the image zoom) | view only | Refits the plot axes |
| Curve plot drag, wheel, double-click | view only | Pans, zooms about the pointer, or refits the plot |
| Line-scan plot click, drag, double-click | view only | Adds a guide at constant r or k, moves it, or removes all guides; a click on a guide removes it |
| k fit slider | `k_min_fit`, `k_max_fit` | Refits the background; dashed lines mark the range on I(k) |
| k window slider | `k_min_window`, `k_max_window` | Moves the window on F(k); dashed lines mark it on F(k) |
| r max slider | `r_max` | End of the r grid, 1 to 50 Å |
| k lowpass, k highpass sliders | `k_lowpass`, `k_highpass` | F(k) filter widths, from 0 (off) to 0.1 1/Å |
| Damp switch | `damp_origin_oscillations` | Low-r correction (needs the density); shows the r_cut and iterations sliders |
| r_cut slider | `r_cut` | Upper end of the correction, on the r grid up to 5 Å or r max |
| iterations slider | `max_iter` | Maximum iterations of the correction, 1 to 30 |
| Density field (empty = not set) | `density_value` | g(r) and the damping; shown in g(r) mode or with damping on |
| Export HTML → HTML exact float32 curves, fixed selection / HTML + quantized uint8 profiles, live selection | `export_request`, `export_payload` | Downloads a standalone viewer (`encoding="full"` / `"uint8"`) |

```{seealso}
The shared HTML-export contract is documented in [html-export](html-export).
```
