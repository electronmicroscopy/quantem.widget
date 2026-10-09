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

`Show1D` supports the shared `ui_mode`, `show_title`, `show_controls`,
`controls_collapsed`, `show_stats`, `show_review`, `show_legend`, and
`show_grid` names. See [Viewer UI controls](viewer-ui). `show_review` is a
constructor parameter; the other toggles are synced traits and pass through
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

`Show1D.live` ranks the traces as losses (`review_mode="optimization"`), so the
Review panel shows the best trial, loss alerts, and the ranking table. A plain
`Show1D(data)` keeps `review_mode="trace"` and only sorts trials by label.

For joint-time ptychography, add snapshots at checkpoint iterations with
multiple named images such as `object_t0`, `object_t5`, `object_t11`, and
`probe`. Each call to `snapshot(...)` is one grouped checkpoint for playback
and thumbnail inspection. `goto_snapshot(index)` selects a group from Python
and `star_snapshot_group(...)` marks it in the playback timeline.

## File-backed monitors

For long reconstructions that should survive notebook disconnects, write a
JSONL monitor beside the run:

```python
Show1D.append_monitor_event(
    "run/show1d_monitor.jsonl",
    {
        "iteration": i,
        "losses": {"lambda 1": loss1, "lambda 10": loss10},
        "snapshots": {"lambda_1": "snapshots/lambda1_i040.npy"},
        "warnings": ["loss spike on lambda 10"],
    },
)
```

Each event may also carry `metrics` (per-trial values such as `rmse` or
`flicker` for the ranking table), `starred`, `hidden`, `notes`, and `tags`.
Reopen the file while the run is still writing:

```python
widget = Show1D.watch_run("run/show1d_monitor.jsonl", refresh_s=5)
```

`watch_run(...)` tails the monitor file: each complete appended line is applied
to the existing widget state, while a partially written trailing line is
ignored until the writer finishes it. Call `widget.stop_monitor()` when the
notebook no longer needs to poll. After a disconnect,
`Show1D.from_monitor_file(...)` rebuilds the same losses, snapshots, warnings,
stars, hidden trials, notes, and tags from disk. A directory path resolves to
`show1d_monitor.jsonl` inside it.

## Open the selected snapshot group in Show2D

To inspect the images behind a selected loss point with the full 2D analysis
toolkit, convert a snapshot group to `Show2D`:

```python
show1d.goto_snapshot(5)
show2d = show1d.to_show2d()
show2d
```

`to_show2d(group=None, images=None, title=None)` takes a group index or label
and group-local image indices or labels; hidden trials are left out by default.
The widget UI exposes the same path through **View -> View selected as 2D**,
which embeds the `Show2D` below the loss viewer. Image labels, colormap, scale
bar units, and stars carry over.

Snapshot panels use the same scale-bar convention as `Show3D`: pass
`sampling=...` and `units=...` for calibrated physical units, or omit them for
a pixel scale bar. Set `scale_bar_visible=False` for a clean export.

## Line profiles

`Show1D.from_image(image, line=((row0, col0), (row1, col1)))` samples a profile
in `(row, col)` image coordinates and keeps the image beside the trace.
`profile_width` averages that many parallel lines; `sampling` and `x_unit`
calibrate the distance axis. `quantem.widget.show1d.sample_line_profile` is the
sampling function on its own.

## Built-in ducky example

The Show1D tutorial reads the real ducky joint-time ptychography sweep from the
public QuantEM data repository:

```python
from quantem.widget import Show1D
from quantem.widget.datasets import show1d_ducky

run = show1d_ducky(size="small")
widget = Show1D.from_monitor_file(
    run / "show1d_monitor.jsonl",
    title="Real ducky joint iterative ptychography",
    x_label="frame",
    y_label="final loss",
    log_scale=False,
    snapshot_columns=4,
    show_snapshot_fft=True,
    snapshot_contrast_preset="1-99",
)
widget
```

The dataset files live under `widget-tutorials/show1d/ducky/small/...` in
`bobleesj/quantem-data`. See [Tutorial Datasets](./datasets.md) for the shared
`small`, `medium`, `large`, and `full` size convention.

## HTML export

Show1D follows the package export signature, with a deliberately narrower set
of supported values:

```python
path = widget.export_html(
    "defocus-review.html",
    mode="single",
    encoding="full",
    downsample=4,
)
```

`mode="single"` and `encoding="full"` are the only supported packaging and
encoding choices. `downsample` may be `None`, `1`, `2`, `4`, or `8`. It
preserves every 1D trace sample and x coordinate while applying a NaN-aware
area mean only to linked 2D snapshots and profile images. Calibrated pixel size,
snapshot view centers, and profile coordinates are rescaled with the image, so
the downsampled review remains physically calibrated.

Show1D does not provide `mode="folder"` or `encoding="uint8"`. Those values
raise `NotImplementedError` with guidance to use a full single export and an
image downsample factor. A full export can be large when hundreds of
full-resolution snapshots are linked, so choose `downsample=2`, `4`, or `8`
when one-file portability matters more than preserving every image pixel.

The generated file is kernel-free, but the embed loads RequireJS, AnyWidget,
and the Jupyter HTML manager from public CDNs. Treat it as a standalone review
file with a network dependency, not as proof of network-offline operation.

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
| Review toggle | `show_review` | Optional ranking, notes, tags, and alerts UI shows/hides; hidden by default |
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
| Snapshot star button | `starred_snapshot_image_labels` | In Review mode, marks candidate reconstructions to revisit while sweeping lambda or denoising settings |
| Snapshot hide button | `hidden_snapshot_image_labels` | In Review mode, hides bad trials from the snapshot grid, loss plot, legend, and stats |
| Show all hidden trials | `hidden_snapshot_image_labels` | In Review mode, restores hidden reconstruction trials |
| Starred-only toggle | `show_starred_only` | In Review mode, shows only starred candidates, while keeping reference panels visible |
| Ranking objective menu | `trial_sort_key` | In Review mode, sorts review rows and snapshot panels by final loss, RMSE, flicker, lambda, object/probe quality, alerts, or label |
| Ranking order toggle | `trial_sort_descending` | In Review mode, reverses candidate ranking order |
| Top-K menu | `top_trial_count` | In Review mode, restricts visible trials to the top ranked candidates |
| Trial filter field | `trial_filter_text` | In Review mode, filters trials by label, note, or tag |
| Trial notes editor toggle | `show_trial_notes` | In Review mode, shows or hides the note/tag editor while preserving stored notes and tags |
| Star best button | `starred_snapshot_image_labels`, `trial_rankings` | In Review mode, stars the current best ranked visible trial |
| Hide worst button | `hidden_snapshot_image_labels`, `trial_rankings` | In Review mode, hides the current worst ranked non-starred trial |
| Trial note field | `trial_notes` | In Review mode, stores per-trial review notes |
| Trial tag buttons | `trial_tags` | In Review mode, stores quick tags such as best, bad start, probe drift, and object issue |
| Review table | `trial_rankings`, `trial_alerts`, `best_trial_label` | In Review mode, shows candidate ranking, alerts, and the best trial |
| View -> View selected as 2D | `handoff_request`, `prepared_view_widget`, `handoff_status` | Builds an embedded Show2D gallery from the selected snapshot group for deeper image analysis |
| Snapshot scale bar API | `pixel_size`, `pixel_unit`, `scale_bar_visible` | Snapshot panels show a Show3D-style scale bar and zoom readout |
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
