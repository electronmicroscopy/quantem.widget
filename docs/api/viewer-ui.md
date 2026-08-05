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
| `controls_collapsed` | Show1D, Show2D, Show3D, Show3DSlices, Show4DSTEM, ShowEDS, ShowDiffraction | Start with controls hidden. Show4DSTEM exposes this as programmatic/state control only; other widgets may also show a top `Controls` button. |
| `collapse_controls()` | Show1D, Show2D, Show3D, Show3DSlices, Show4DSTEM, ShowEDS, ShowDiffraction | Programmatically collapse controls. |
| `expand_controls()` | Show1D, Show2D, Show3D, Show3DSlices, Show4DSTEM, ShowEDS, ShowDiffraction | Programmatically expand controls. |
| `toggle_controls()` | Show1D, Show2D, Show3D, Show3DSlices, Show4DSTEM, ShowEDS, ShowDiffraction | Programmatically toggle collapsed controls. |
| `show_stats` | Show1D, Show2D, Show3D, Show3DSlices, Show4DSTEM, ShowDiffraction | Show or hide mean/min/max/std readouts where the widget renders or computes them. |
| `show_panel_titles` | Show2D, Show3D | Show labels on individual image/volume panels. |
| `panel_title_font_size` | Show2D, Show3D | Font size for per-panel labels. |
| `panel_title_spans` | Show2D, Show3D | Optional structured colored text spans for panel-title status words; plain `labels` / `panel_titles` strings still work. |
| `panel_title_style` | Show2D, Show3D | Optional JSON-safe title chrome, for example background, text color, border color/width, padding, radius, and max width. Defaults preserve the normal transparent title look. |
| `panel_annotations` | Show2D, Show3D | Optional per-panel in-image labels. Use these for multiple local callouts, ROI labels, dose/status badges, or region notes that are not the panel title. |
| `marker_colors`, `marker_style` | Show2D, Show3D | Optional per-panel identity colors. `marker_style="around"` draws a color frame with a black inner edge. |
| `row_markers`, `col_markers` | Show2D, Show3D | Optional row/column group frames keyed by zero-based visible row or column slot. Each group draws one rectangle with no internal dividers. |
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

## Colored status words in panel titles

Use colored title spans when borders identify a group but the title itself
needs a status word, such as low / mid / high dose, under / cal / over
regularization, raw vs denoised, or a χ² value. Spans are structured
dictionaries, not HTML, so exported offline HTML remains theme-safe and
XSS-safe.

```python
from quantem.widget import Show2D, Show3D

status_titles = [
    [
        {"text": "BF denoise  "},
        {"text": "low", "color": "#60a5fa"},
        {"text": "  χ²="},
        {"text": "0.5", "color": "#f59e0b"},
    ],
    [
        {"text": "BF denoise  "},
        {"text": "mid", "color": "#34d399"},
        {"text": "  χ²="},
        {"text": "1.0", "color": "#34d399"},
    ],
    [
        {"text": "BF denoise  "},
        {"text": "high", "color": "#f87171"},
        {"text": "  χ²="},
        {"text": "1.8", "color": "#f87171"},
    ],
]

Show2D(images, labels=status_titles, marker_colors=["#60a5fa", "#34d399", "#f87171"])
Show3D(stack_a, stack_b, stack_c, panel_titles=status_titles)
```

The plain fallback text is still synced as the normal panel label/title, so
panel menus, title-based API lookups, notebook state, and `export_html(...)`
continue to behave like ordinary strings.

## Panel frames and title chrome

Use per-panel markers for individual identity and row/column markers when a
whole visual group should share one frame. Row and column keys are zero-based
visible grid slots, so hidden panels do not leave ghost group borders.

```python
Show2D(
    images,
    labels=status_titles,
    ncols=3,
    marker_colors=["#60a5fa", "#34d399", "#f87171"],
    marker_style="around",
    row_markers={0: "#60a5fa"},
    col_markers={2: "#f59e0b"},
    panel_title_style={
        "bg": "rgba(0,0,0,0.72)",
        "fg": "#ffffff",
        "border_color": "#60a5fa",
        "border_width": 1,
        "pad_x": 6,
        "pad_y": 2,
        "radius": 2,
        "max_width": "hug",
    },
)
```

For local region labels inside a panel, use `panel_annotations`. Longer
examples live in the [advanced tutorial](../tutorials/advanced.md).
