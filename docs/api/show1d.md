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

Snapshot panels use the same scale-bar convention as `Show3D`: pass
`sampling=...` and `units=...` for calibrated physical units, or omit them for
a pixel scale bar. Use `show_scale_bar=False` for a clean export without scale
or zoom overlays.

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
