# Show1D

Interactive 1D traces for live reconstruction metrics, line profiles, and
linked image snapshots. Use it for loss curves, optimizer diagnostics,
joint-time ptychography comparisons, and image-derived profiles that need a
visible 2D context.

## Resizing a scientific curve

Use `plot_width_px=700`, `plot_height_px=300`, and `max_width=900` to set a
bounded initial size. Width values of zero keep the responsive layout. The
bottom-right corner handle resizes a standalone curve horizontally and
vertically. With a snapshot/stats side panel, horizontal dragging redistributes
the plot and side-panel space. Preview stays in the browser during dragging;
width and height are saved on release and retained in widget state and HTML
exports. The Reset toolbar action restores the initial plot dimensions.

## Viewer UI

`Show1D` supports the shared `mode`, `show_title`, `show_controls`,
`controls_collapsed`, `show_stats`, `show_legend`, and `show_grid` names. See
[Viewer UI controls](viewer-ui). The toggles are synced traits and pass through
the constructor as keywords (`Show1D(data, show_legend=False)`).

## Live monitors

For a live notebook reconstruction, mutate one widget instead of recreating
cells. Use `append(...)` for one scalar sample per trace, `extend(...)` for a
block of samples, and `snapshot(...)` for grouped object/probe images:

```python
widget = Show1D.live(["lambda 1", "lambda 10"], title="overnight loss")
widget.extend(
    [0, 1, 2],
    **{"lambda 1": [3.0, 2.0, 1.0], "lambda 10": [4.0, 3.0, 2.5]},
)
widget.snapshot(2, label="iter 2", object=obj, probe=probe)
```

For joint-time ptychography, add snapshots at checkpoint iterations with
multiple named images such as `object_t0`, `object_t5`, `object_t11`, and
`probe`. Each call to `snapshot(...)` is one grouped checkpoint for playback
and thumbnail inspection. `star_snapshot_group(...)` marks a group in the
playback timeline.

Snapshot panels use the same scale-bar convention as `Show3D`: pass
`sampling=...` and `units=...` for calibrated physical units, or omit them for
a pixel scale bar. Set `show_scale_bar=False` for a clean export.

## Line profiles

`Show1D.from_image(image, line=((row0, col0), (row1, col1)))` samples a profile
in `(row, col)` image coordinates and keeps the image beside the trace.
`profile_width` averages that many parallel lines; `sampling` and `units`
calibrate the distance axis. `quantem.widget.show1d.sample_line_profile` is the
sampling function on its own.

## Built-in ducky example

The Show1D tutorial reads reconstruction snapshots for the synthetic Ducky
example from the public QuantEM data repository.

Ducky simulation tutorial: Georgios Varnavides, [*Trademark ducky dataset* (October 23, 2025)](https://github.com/electronmicroscopy/quantem-tutorials/blob/dbadc111734641226ab0454ed988feaf018e65c4/simulation_notebooks/ducky-dataset.ipynb). The viewer files contain subsequent reconstruction results for the synthetic example.

```python
import json

import numpy as np

from quantem.widget import Show1D
from quantem.widget.datasets import show1d_ducky

run = show1d_ducky()
events = [json.loads(line) for line in (run / "show1d_monitor.jsonl").read_text().splitlines()]
widget = Show1D(
    {name: [event["losses"][name] for event in events] for name in events[0]["losses"]},
    x=[event["iteration"] for event in events],
    title="Synthetic Ducky reconstructions (tutorial: Georgios Varnavides)",
    x_label="frame",
    y_label="final loss",
    x_integer=True,
    snapshot_columns=4,
    show_snapshot_fft=True,
    snapshot_contrast_preset="1-99",
)
widget.snapshot(11, label=events[11]["label"], **{name: np.load(run / path) for name, path in events[11]["snapshots"].items()})
widget
```

The default pins `widget/ducky/` in `bobleesj/quantem-data`; each line of
`show1d_monitor.jsonl` holds one frame's losses and the snapshot paths relative
to the folder. See [Tutorial datasets](datasets.md)
for the exact revision. This helper has one collection and no
`size` argument. Retain the visible source credit when exporting a notebook,
report or standalone widget. See [Tutorial datasets](./datasets.md) for the
exact revisions and inputs.

## HTML export

Show1D writes one page with exact float32 traces; its only packing choice is
the image image_bin:

```python
path = widget.export_html("defocus-review.html", image_bin=4)
```

`image_bin` may be `None`, `1`, `2`, `4`, or `8`. It
preserves every 1D trace sample and x coordinate while applying a NaN-aware
area mean only to linked 2D snapshots and profile images. Calibrated pixel size,
snapshot view centers, and profile coordinates are rescaled with the image, so
the downsampled review remains physically calibrated.

A full export can be large when hundreds of full-resolution snapshots are
linked, so choose `image_bin=2`, `4`, or `8` when one-file portability matters
more than preserving every image pixel.

The generated file is kernel-free and needs no network: RequireJS, AnyWidget
and the Jupyter HTML manager travel inside the page (about 1.2 MB).

`save_image("traces.png")` writes a matplotlib figure of the traces (`.pdf`
also works); `save("view.json")` writes the display state that
`Show1D(data, state="view.json")` restores.

## Reference

```{eval-rst}
.. autoclass:: quantem.widget.show1d.Show1D
   :members:
   :show-inheritance:

.. autofunction:: quantem.widget.show1d.sample_line_profile
```

## Interactive controls

Each control mutates the listed synced trait. A UI-test agent acts on the
control, then asserts the trait changed and the canvas repainted (non-zero,
no console error, no NaN frame).

| Control | Trait | Expected effect |
|---|---|---|
| Trace hover | read-only canvas overlay, local snapshot preview | Nearest trace point is highlighted and reported; when a snapshot group has the same x value, its images and group label preview in the side panel |
| Trace or legend click | `focused_trace`, `selected_snapshot_group_idx` | A trace-point click pins its matching snapshot group and emphasizes the trace; a legend click only emphasizes the trace, and double-clicking the plot restores all traces |
| Plot corner drag | `plot_height_px`, `side_panel_width_px` | Bottom-right loss-plot handle resizes plot height and reallocates width between the loss plot and snapshot panel |
| Reset view | `x_range`, `y_range`, `focused_trace` | Plot returns to full data extent |
| Grid toggle | `show_grid` | Grid lines show/hide |
| Log toggle | `log_scale` | Positive y values render on a logarithmic axis |
| Stats toggle | `show_stats` | Optional stats side table shows/hides; hidden by default |
| Legend toggle | `show_legend` | Trace legend shows/hides |
| Snapshot panel visibility API | `show_snapshots` | Reconstruction snapshot panel is shown by default; set this from Python for plot-only summaries |
| Plot thumbnail API | `show_snapshot_thumbnails`, `snapshot_thumbnail_size` | Plot thumbnails are shown by default; set size from Python when a notebook needs denser or larger checkpoint previews |
| Snapshot colormap menu | `image_cmap` | Profile/snapshot images use the selected scientific colormap |
| Snapshot contrast buttons | `snapshot_contrast_preset`, `snapshot_contrast_range` | Snapshot images use full, 0.5-99.5, 1-99, 2-98, or 5-95 percentile clipping; choosing a preset clears custom histogram clipping |
| Snapshot histogram drag | `snapshot_contrast_range`, `snapshot_panel_contrast_ranges` | Drag either endpoint knot to adjust min/max; drag the middle span to move the contrast window |
| Snapshot histogram visibility API | `show_snapshot_histogram` | Shows or hides the compact selected-snapshot histogram; it is shown by default |
| Snapshot histogram size API | `snapshot_histogram_width`, `snapshot_histogram_height` | Keeps the compact contrast histogram independent of the reconstruction grid size |
| Snapshot profile toggle | `show_snapshot_profile`, `snapshot_profile_line`, `snapshot_profile_height` | Draws a shared `(row, col)` line profile on reconstruction panels and compares visible panel intensities below the image grid |
| Snapshot columns menu | `snapshot_columns` | Snapshot object/probe image grid uses automatic overview columns or a fixed 1-8 columns |
| Snapshot FFT overlay position | `snapshot_overlay_position` | FFT inset overlays can sit in any corner; drag the inset to snap it to the nearest corner or set top-left, top-right, bottom-left, or bottom-right from Python |
| Snapshot panel corner drag | `snapshot_panel_width_px` | Every real snapshot tile has a Show2D-style corner grip; dragging any grip changes one shared tile size, keeps all panels equal, preserves the selected column count, and keeps controls aligned to the grid width |
| Snapshot playback star | `bookmarked_snapshot_groups`; `star_snapshot_group()` | Marks important reconstruction iterations in the playback timeline; starred positions render as gold timeline marks and persist in widget state/HTML export |
| Snapshot hide button | `hidden_snapshot_image_labels` | Hides an image label from the snapshot grid, and the trace of the same name from the loss plot, legend, and stats |
| Show all | `hidden_snapshot_image_labels` | Shown when every image of the selected group is hidden; restores all hidden labels |
| Snapshot scale bar API | `pixel_size`, `pixel_unit`, `show_scale_bar` | Snapshot panels show a Show3D-style scale bar and zoom readout |
| Snapshot real-space view API | `snapshot_real_space_zoom`, `snapshot_real_space_center` | Starts or restores real-space snapshot panels at a given zoom and `(row, col)` center |
| Snapshot FFT view API | `snapshot_fft_zoom`, `snapshot_fft_center` | Starts or restores FFT panels at a given zoom and `(row, col)` FFT center |
| Snapshot histogram | computed automatically | Selected snapshot histogram stays visible with draggable contrast knots and a numeric range readout; histogram and snapshot FFT use WebGPU when the browser has it, with a CPU fallback |
| Snapshot FFT toggle | `show_snapshot_fft`, `snapshot_fft_layout` | Log-magnitude FFTs show as compact inset overlays by default; set `snapshot_fft_layout="below"` for stacked panels |
| Snapshot FFT window toggle | `snapshot_fft_window` | Applies a Hann window before snapshot FFT computation |
| Snapshot FFT colormap menu | `snapshot_fft_cmap` | FFT panels use the selected scientific colormap |
| Snapshot play/pause | `snapshot_playing` | Snapshot groups advance through reconstruction checkpoints |
| Snapshot stop | `snapshot_playing`, `selected_snapshot_group_idx` | Playback stops and returns to the first snapshot group |
| Snapshot playback endpoint mode | `snapshot_loop`, `snapshot_bounce` | Playback either wraps, stops, or reverses direction at the first and final snapshot groups |
| Snapshot group slider | `selected_snapshot_group_idx`, `selected_snapshot_idx` | Object/probe/multi-object image group changes; plot marker moves to that iteration |
| Snapshot FPS slider | `snapshot_fps` | Playback speed changes in whole frames per second |
| Snapshot image wheel | local image view | Zooms snapshot/FFT panels under the cursor without scrolling the page |
| Snapshot image drag | local image view | Pans the shared snapshot/FFT zoom view |
| Zoom wheel | `x_range` | X-axis zooms about the cursor |
| Shift + zoom wheel | `y_range` | Y-axis zooms about the cursor |
| Double-click plot | `x_range`, `y_range`, `focused_trace` | View resets |
| Export -> HTML | `export_request`, `export_payload` | Writes a standalone interactive HTML viewer |
| Export -> CSV | browser download | Downloads current trace arrays as CSV |
| Export -> PNG | browser download | Downloads the current plot canvas |

`plot_height_px` and `side_panel_width_px` remain constructor/state parameters
for notebooks, reports, and saved views. They are intentionally not shown as
default toolbar sliders so overnight reconstruction review starts with fewer
visible controls.

```{seealso}
The shared HTML-export contract is documented in [html-export](html-export).
```
