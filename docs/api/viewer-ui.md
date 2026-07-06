# Viewer UI Controls

QuantEM viewers use the same names for common display chrome. Use explicit
keyword arguments for durable APIs, and use `ui_mode` as a convenience preset
for common sharing contexts.

The same tables are also available as the developer-facing [UI Guide](../developer/ui-guide).

## Presets

| Preset | Why it exists | Behavior |
|---|---|---|
| `ui_mode="interactive"` | Normal notebook analysis. | Full widget UI, controls open, default annotations visible. |
| `ui_mode="presentation"` | Shared HTML that should open cleanly but remain adjustable. | Controls start collapsed behind `Controls`; stats, resize handles, and zoom badges are hidden where supported. Titles, panel labels, and scale bars stay visible. |
| `ui_mode="report"` | Clean figure in a report or manuscript-style notebook. | Controls are removed with no GUI recovery button; stats, resize handles, and zoom badges are hidden. Titles, panel labels, and scale bars stay visible. |
| `ui_mode="minimal"` | Bare visual/GIF-like output. | Title, controls, stats, panel titles, scale bars, resize handles, zoom badges, and equivalent optional chrome are hidden where supported. |

Explicit keyword arguments override the preset:

```python
Show3D(
    data,
    ui_mode="presentation",
    show_panel_titles=True,
    show_scale_bar=True,
)
```

## Common Names

| Name | Widgets | Meaning |
|---|---|---|
| `show_title` | Show1D, Show2D, Show3D, Show3DSlices, Show4DSTEM, ShowEDS, ShowDiffraction | Show or hide the top title row. |
| `show_controls` | Show1D, Show2D, Show3D, Show3DSlices, Show4DSTEM, ShowEDS, ShowDiffraction | Expose controls at all. Set `False` for a permanently clean display with no GUI recovery button. |
| `controls_collapsed` | Show1D, Show2D, Show3D, Show3DSlices, Show4DSTEM, ShowEDS, ShowDiffraction | Start with controls hidden behind the top `Controls` button. |
| `collapse_controls()` | Show1D, Show2D, Show3D, Show3DSlices, Show4DSTEM, ShowEDS, ShowDiffraction | Programmatically collapse controls. |
| `expand_controls()` | Show1D, Show2D, Show3D, Show3DSlices, Show4DSTEM, ShowEDS, ShowDiffraction | Programmatically expand controls. |
| `toggle_controls()` | Show1D, Show2D, Show3D, Show3DSlices, Show4DSTEM, ShowEDS, ShowDiffraction | Programmatically toggle collapsed controls. |
| `show_stats` | Show1D, Show2D, Show3D, Show3DSlices, Show4DSTEM, ShowDiffraction | Show or hide mean/min/max/std readouts where the widget renders or computes them. |
| `show_panel_titles` | Show2D, Show3D | Show labels on individual image/volume panels. |
| `panel_title_font_size` | Show2D, Show3D | Font size for per-panel labels. |
| `show_scale_bar` | Show2D, Show3D, Show3DSlices, Show4DSTEM, ShowEDS | Show or hide scale bars. Some widgets keep the saved-state trait name `scale_bar_visible` for compatibility. |
| `show_resize_handles` | Show3D | Show or hide resize handles. Show2D hides resize handles automatically when controls are collapsed. |
| `show_zoom_indicator` | Show3D | Show or hide the zoom readout. |
| `show_legend` | Show1D | Show or hide trace legends. |
| `show_grid` | Show1D | Show or hide plot grid lines. |
| `show_crosshair` | Show3DSlices | Show or hide slice intersection guides. |

## Recommended Report Defaults

Use this for shared HTML reports where collaborators should see clean figures
first but can still reopen controls:

```python
widget = Show3D(data, ui_mode="presentation")
```

Use this for static-looking report embeds where controls should not be
recoverable but scientific context should remain:

```python
widget = Show3D(data, ui_mode="report")
```

Use this for the cleanest visual output:

```python
widget = Show3D(data, ui_mode="minimal")
```

Prefer explicit overrides instead of new one-off names. For example, use
`show_scale_bar=False` rather than widget-specific spellings like
`hide_scalebar=True`.
