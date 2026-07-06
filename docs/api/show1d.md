# Show1D

Interactive 1D traces for live reconstruction metrics, line profiles, and
linked image snapshots. Use it for loss curves, Adam/optimizer diagnostics,
joint-time ptychography comparisons, and image-derived profiles that need a
visible 2D context.

## Viewer UI

`Show1D` supports the shared `ui_mode`, `show_title`, `show_controls`,
`controls_collapsed`, `show_stats`, `show_legend`, and `show_grid` names. See
[Viewer UI controls](viewer-ui).

## Loss Comparisons

Use `Show1D.from_loss_runs` when a notebook needs to compare several optimizer
or reconstruction histories without hand-flattening labels:

```python
from quantem.widget import Show1D

widget = Show1D.from_loss_runs(
    {
        "lambda 1": {
            "data": lambda1_data_loss,
            "temporal": lambda1_temporal_loss,
        },
        "lambda 10": {
            "data": lambda10_data_loss,
            "temporal": lambda10_temporal_loss,
        },
    },
    x=iterations,
    losses=["data", "temporal"],
    label_template="{run} / {loss}",
    title="Joint iterative ptychography loss comparison",
    x_label="iteration",
    y_label="loss",
    log_scale=True,
)
```

For joint-time ptychography, add snapshots at checkpoint iterations with
multiple named images such as `object_t0`, `object_t5`, `object_t11`, and
`probe`. The frontend treats each call to `snapshot(...)` as one grouped
checkpoint for playback and thumbnail inspection.

For saved joint-time reports, `Show1D.from_joint_time_report(...,
frame_by_frame=True)` builds a frame-indexed loss view and groups the matching
`reconstructions.npz` images by frame:

```python
widget = Show1D.from_joint_time_report(
    "summary.json",
    frame_by_frame=True,
    snapshot_downsample=4,
    snapshot_columns=3,
    trial_sort_key="final_loss",
)
```

The review state is synced and exportable. Use `star_best_trial()`,
`hide_worst_trials()`, `set_trial_note(...)`, `tag_trial(...)`, and
`export_run_summary(...)` to preserve the morning review of an overnight sweep.

To inspect the reconstruction images behind a selected loss point with the full
2D analysis toolkit, convert the current snapshot group to `Show2D`:

```python
show1d = Show1D.from_joint_time_report("summary.json", frame_by_frame=True)
show1d.goto_snapshot(5)

show2d = show1d.to_show2d()
show2d
```

The widget UI exposes the same path through **View -> View selected as 2D**.
The embedded `Show2D` appears below the loss viewer and preserves image labels,
colormap, scale bar units, stars, hidden trials, and the active visible
comparison set. This is useful for opening the best/lambda-filtered
reconstructions directly into Show2D zoom, pan, histogram, FFT, profile, and
export tools without rebuilding arrays by hand.

Snapshot panels use the same scale-bar convention as `Show3D`: pass
`sampling=...` and `units=...` for calibrated physical units, or omit them for
a pixel scale bar. Use `show_scale_bar=False` for a clean export without scale
or zoom overlays.

## Overnight Monitors

For long reconstructions, write a JSONL monitor beside the run:

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

Reopen it later with:

```python
widget = Show1D.watch_run("run/show1d_monitor.jsonl", refresh_s=5)
```

If the kernel disconnects overnight, `Show1D.from_monitor_file(...)` rebuilds
the same losses, snapshots, warnings, stars, hidden trials, notes, and tags from
disk.

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
| Trace hover | read-only canvas overlay | Nearest trace point is highlighted and reported |
| Trace click | `focused_trace` | Selected trace remains emphasized; other traces fade |
| Trace dropdown | `focused_trace` | All traces or one trace is emphasized |
| Reset view | `x_range`, `y_range`, `focused_trace` | Plot returns to full data extent |
| Grid toggle | `show_grid` | Grid lines show/hide |
| Log toggle | `log_scale` | Positive y values render on a logarithmic axis |
| Stats toggle | `show_stats` | Stats side table shows/hides |
| Legend toggle | `show_legend` | Trace legend shows/hides |
| Plot height slider | `plot_height_px` | Loss/metric plot height resizes |
| Side panel width slider | `side_panel_width_px` | Snapshot/stats panel width resizes |
| Snapshots toggle | `show_snapshots` | Reconstruction snapshot panel shows/hides |
| Thumbnail toggle | `show_snapshot_thumbnails` | Grouped image thumbnails show/hide on snapshot points in the plot |
| Thumbnail size slider | `snapshot_thumbnail_size` | Plot thumbnails resize while keeping grouped object/probe montage layout |
| Snapshot colormap menu | `image_cmap` | Profile/snapshot images use the selected scientific colormap |
| Snapshot contrast buttons | `snapshot_contrast_preset` | Snapshot images use full, 0.5-99.5, 1-99, 2-98, or 5-95 percentile clipping |
| Snapshot columns menu | `snapshot_columns` | Snapshot object/probe image grid displays 1-4 columns |
| Snapshot star button | `starred_snapshot_image_labels` | Marks candidate reconstructions to revisit while sweeping lambda or denoising settings |
| Snapshot hide button | `hidden_snapshot_image_labels` | Hides bad trials from the snapshot grid, loss plot, legend, and stats |
| Show all hidden trials | `hidden_snapshot_image_labels` | Restores hidden reconstruction trials |
| Starred-only toggle | `show_starred_only` | Shows only starred candidates, while keeping reference panels visible |
| Ranking objective menu | `trial_sort_key` | Sorts review rows and snapshot panels by final loss, RMSE, flicker, lambda, object/probe quality, alerts, or label |
| Ranking order toggle | `trial_sort_descending` | Reverses candidate ranking order |
| Top-K menu | `top_trial_count` | Restricts visible trials to the top ranked candidates |
| Trial filter field | `trial_filter_text` | Filters trials by label, note, or tag |
| Star best button | `starred_snapshot_image_labels`, `trial_rankings` | Stars the current best ranked visible trial |
| Hide worst button | `hidden_snapshot_image_labels`, `trial_rankings` | Hides the current worst ranked non-starred trial |
| Trial note field | `trial_notes` | Stores per-trial review notes |
| Trial tag buttons | `trial_tags` | Stores quick tags such as best, bad start, probe drift, and object issue |
| Review table | `trial_rankings`, `trial_alerts`, `best_trial_label`, `run_summary` | Shows candidate ranking, alerts, and best-trial summary |
| View -> View selected as 2D | `handoff_request`, `prepared_view_widget`, `handoff_status` | Builds an embedded Show2D gallery from the selected snapshot group for deeper image analysis |
| Snapshot scale bar API | `pixel_size`, `pixel_unit`, `scale_bar_visible` | Snapshot panels show a Show3D-style scale bar and zoom readout |
| Snapshot histogram | computed automatically | Selected snapshot histogram stays visible so contrast presets can be adjusted directly |
| WebGPU preference API | `prefer_webgpu` | Hidden UI preference; histogram and snapshot FFT use WebGPU when available, with CPU fallback |
| Snapshot FFT toggle | `show_snapshot_fft` | Log-magnitude FFT panels show below snapshot images |
| Snapshot FFT window toggle | `snapshot_fft_window` | Applies a Hann window before snapshot FFT computation |
| Snapshot FFT colormap menu | `snapshot_fft_cmap` | FFT panels use the selected scientific colormap |
| Snapshot play/pause | `snapshot_playing` | Snapshot groups advance through reconstruction checkpoints |
| Snapshot stop | `snapshot_playing`, `selected_snapshot_group_idx` | Playback stops and returns to the first snapshot group |
| Snapshot group slider | `selected_snapshot_group_idx`, `selected_snapshot_idx` | Object/probe/multi-object image group changes; plot marker moves to that iteration |
| Snapshot FPS slider | `snapshot_fps` | Playback speed changes |
| Zoom wheel | `x_range` | X-axis zooms about the cursor |
| Shift + zoom wheel | `y_range` | Y-axis zooms about the cursor |
| Double-click plot | `x_range`, `y_range`, `focused_trace` | View resets |
| Export -> HTML | `export_request`, `export_payload` | Writes a standalone interactive HTML viewer |
| Export -> CSV | browser download | Downloads current trace arrays as CSV |
| Export -> PNG | browser download | Downloads the current plot canvas |

```{seealso}
The shared HTML-export contract is documented in [html-export](html-export).
```
