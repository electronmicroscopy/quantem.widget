# Advanced Show2D

`Show2D` can be a quick image viewer, but it is also the main review surface
for dense microscope comparisons: raw versus denoised panels, reconstruction
sweeps, residual maps, ACF summaries, ROI callouts, and export-ready figures.
This page collects the higher-level configuration patterns. For the exhaustive
trait list, see the [Show2D API reference](../api/show2d).

Use the basic [Show2D tutorial](show2d) when you only need to open images,
change contrast, zoom, or draw an ROI. Use this page when the widget is part of
a report or repeated scientific review.

## Choose The Right Show2D Pattern

| Review need | Use | Why |
| --- | --- | --- |
| One image or a short gallery | `Show2D(image_or_stack)` | Fast inspection with contrast, FFT, profile, ROI, and export controls |
| Raw / denoised / residual comparison | `Show2D([raw, denoised, residual], labels=...)` | Keeps panels in one linked inspection surface |
| Each panel has its own depth/time slider | `Show2D([stack_a, stack_b], labels=...)` | Every 3D list item gets independent frame controls |
| Many conditions, one page at a time | `pages=[...]` | Keeps dense sweeps readable without one huge grid |
| Reproducible visual callouts | `panel_annotations`, `panel_overlays`, `inset_plots` | Saved state and exported HTML reproduce the same figure intent |
| Live editable figure callouts | `panel_overlays` + More -> Overlay Edit | Reproducible circles/rectangles can be selected, moved, resized, deleted, or reset in live/exported HTML |
| Measurement geometry with Python readback | ROI tools | ROIs are the path for statistics, FFT crops, and `get_roi_geometries()` readback |

## Start With A Report-Ready Gallery

Give every panel a human-readable label, choose a fixed panel layout, and use
around-panel identity frames when the text refers to "blue", "green", or
"red" panels.

```python
from quantem.widget import Show2D

w = Show2D(
    [raw, denoised, residual],
    labels=["raw", "denoised", "residual"],
    ncols=3,
    marker_style="around",
    marker_colors=["#60a5fa", "#34d399", "#f87171"],
    link_pan=True,
    link_zoom=True,
    show_stats=True,
)
w
```

Use `ui_mode="presentation"` when the first view should be clean but still
recoverable. The exported HTML keeps the same state.

```python
w = Show2D(
    [raw, denoised, residual],
    labels=["raw", "denoised", "residual"],
    ncols=3,
    ui_mode="presentation",
    show_stats=True,
)
w.export_html("denoise-review.html", mode="single", encoding="uint8")
```

Presentation mode is not a screenshot mode. The user can still open controls,
zoom, pan, tune contrast, and export from the embedded page.

## Rich Math Titles

Panel titles can use plain Unicode, compact TeX-style math, or structured
spans. Use this for symbols such as `λ`, `χ²`, `σ`, `μ`, and short units.

```python
Show2D(
    [raw, residual],
    labels=[
        r"$\lambda=0.03$ raw",
        r"$\chi^2$/pixel residual",
    ],
    show_stats=True,
)
```

Use `panel_title_spans` when a title needs mixed color or a controlled math
span:

```python
Show2D(
    [raw, denoised, residual],
    labels=["raw", "denoised", "residual"],
    panel_title_spans=[
        [{"math": r"\lambda=0.03"}, {"text": " raw"}],
        [{"text": "denoised", "color": "#34d399"}],
        [{"math": r"\chi^2"}, {"text": "/pixel residual"}],
    ],
)
```

Keep panel titles short. Put full equations in Markdown near the widget and
use the title for the compact symbol or condition label.

## Local Labels Inside Panels

Use `panel_annotations` when the label belongs to a region inside an image,
not to the whole panel. Annotations can be keyed by panel label or panel index,
and each panel can have multiple labels.

```python
Show2D(
    [raw, denoised, residual],
    labels=["raw", "denoised", "residual"],
    panel_annotations={
        "raw": [
            {"text": "input", "position": "top-left", "variant": "pill"},
            {
                "spans": [
                    {"text": "ROI "},
                    {"text": "A", "color": "#60a5fa"},
                ],
                "box": [0.18, 0.25, 0.30, 0.16],
                "variant": "callout",
                "bg": "rgba(0,0,0,0.58)",
                "border_color": "#60a5fa",
            },
        ],
        "residual": {
            "math": r"\chi^2",
            "position": "top-right",
            "variant": "outline",
            "border_color": "#f87171",
        },
    },
)
```

Use corner placement for badges, `x`/`y` for point labels, and `box` for local
region labels:

```python
{"text": "corner", "position": "top-left"}
{"text": "point", "x": 0.62, "y": 0.35, "anchor": "center"}
{"text": "region", "box": [0.20, 0.25, 0.30, 0.18]}
```

## Geometric Overlays

Use `panel_overlays` when a circle, rectangle, or square is part of a
reproducible figure specification. Coordinates use data pixels by default with
QuantEM's user-facing `(row, col)` convention.

```python
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

Use `overlays=[...]` for one shared guide on every panel, or include `panel=`
inside a flat list when overlays are generated in a loop:

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

Show2D(
    [raw, denoised],
    labels=["raw", "denoised"],
    panel_overlays=[
        {"panel": "raw", "shape": "circle", "center": (96, 88), "radius": 14},
        {"panel": "denoised", "shape": "rect", "box": (48, 58, 126, 146)},
    ],
)
```

When overlays are present, open `More -> Overlay Edit` in the live widget or
exported HTML. Click an overlay to select it, drag inside to move it, drag an
edge to resize it, press Delete to remove the selected overlay, and choose
`Reset Overlays` to restore the constructor state. Use ROI tools when the
geometry should feed statistics, FFT crops, or Python readback.

## Inset Plots

Use `inset_plots` when each image needs its own small curve, for example an
ACF-vs-r trace, a residual sweep, a dose curve, or a convergence metric. The
plot lives inside the panel, so it stays with the image in saved state and
exported HTML.

```python
Show2D(
    [raw, denoised, residual],
    labels=["raw", "denoised", "residual"],
    inset_plots=[
        {
            "x": r_values,
            "y": raw_acf,
            "position": "bottom-right",
            "size": 0.32,
            "height": 0.22,
            "color": "#60a5fa",
            "xlabel": "R",
            "ylabel": "ACF",
            "legend": "ACF",
        },
        {
            "x": r_values,
            "y": denoised_acf,
            "position": "bottom-right",
            "size": 0.32,
            "height": 0.22,
            "color": "#34d399",
            "xlabel": "R",
            "ylabel": "ACF",
            "legend": "ACF",
        },
        None,
    ],
)
```

Keep inset plots small and use them to support the image, not to replace a
dedicated analysis figure. If the plot needs axes, legends, or detailed labels,
put the full plot below the widget and use the inset as the quick panel cue.

## Paging And Local Stacks

Use a list of 3D arrays when each panel has its own local frame axis. This is
different from `Show3D`, where panels usually share one global frame index.

```python
w = Show2D(
    [coarse_stack, medium_stack, fine_stack],
    labels=["coarse z", "medium z", "fine z"],
    panel_playback_fps=[6, 6, 6],
    ncols=3,
)
w.set_panel_frame("fine z", 8)
w
```

For many conditions, use pages so each view stays readable:

```python
w = Show2D(
    pages=[
        [raw_001, den_001, residual_001],
        [raw_002, den_002, residual_002],
    ],
    page_labels=["frame 001", "frame 002"],
    labels=["raw", "denoised", "residual"],
    ncols=3,
)
w
```

## Export Checklist

Before sharing a Show2D report, check these items:

- Use `labels` or `panel_title_spans` so every panel is identifiable.
- Use `marker_style="around"` or group markers when text refers to panel color.
- Keep local annotations and overlays reproducible through `panel_annotations`
  and `panel_overlays`.
- Use `ui_mode="presentation"` when the exported first view should be clean.
- State any `downsample` or `encoding="uint8"` choice in the surrounding report
  when scientific interpretation depends on exact pixels.
- For interactive proof, drive the exported HTML in a headed browser: zoom,
  pan, change contrast, switch pages or frames, and confirm overlays/labels
  remain visible.
