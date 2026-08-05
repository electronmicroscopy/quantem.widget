---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Advanced Show2D

`Show2D` can be a quick image viewer, but it is also the main review surface
for dense microscope comparisons: raw versus denoised panels, reconstruction
sweeps, residual maps, ACF summaries, ROI callouts, and export-ready figures.
This page collects the higher-level configuration patterns. For the exhaustive
trait list, see the [Show2D API reference](../api/show2d).

Use the basic [Show2D tutorial](show2d) when you only need to open images,
change contrast, zoom, or draw an ROI. Use this page when the widget is part of
a report or repeated scientific review.

```{code-cell} python
:tags: [remove-input]

import numpy as np

from quantem.widget import Show2D

rng = np.random.default_rng(42)
yy, xx = np.mgrid[-1:1:128j, -1:1:128j]


def lattice(theta=0.0, shear=0.0, noise=0.05):
    c, s = np.cos(theta), np.sin(theta)
    xr = c * xx - s * yy + shear * yy
    yr = s * xx + c * yy
    atoms = (
        np.cos(17 * np.pi * xr) * np.cos(13 * np.pi * yr)
        + 0.45 * np.cos(23 * np.pi * (xr + yr))
    )
    envelope = np.exp(-1.5 * (xx**2 + yy**2))
    image = envelope * atoms
    image += noise * rng.standard_normal(image.shape)
    return image.astype("float32")


raw = lattice(theta=0.00, shear=0.00, noise=0.10)
denoised = lattice(theta=0.02, shear=-0.04, noise=0.035)
residual = (raw - denoised).astype("float32")
scan90 = lattice(theta=np.pi / 2, shear=0.00, noise=0.10)
combined = 0.5 * (raw + scan90)
corrected0 = lattice(theta=0.04, shear=-0.10, noise=0.05)
corrected90 = lattice(theta=np.pi / 2 + 0.03, shear=0.08, noise=0.05)
corrected_combined = 0.5 * (corrected0 + corrected90)
panels = [raw, scan90, combined, corrected0, corrected90, corrected_combined]
labels = [
    "a 0° scan",
    "b 90° scan",
    "c combined",
    "d 0° scan corrected",
    "e 90° scan corrected",
    "f corrected combined",
]

r_values = np.linspace(0, 1, 48)
raw_acf = np.exp(-2.5 * r_values) * (1 + 0.12 * np.cos(20 * r_values))
denoised_acf = np.exp(-2.1 * r_values) * (1 + 0.08 * np.cos(18 * r_values))
coarse_stack = np.stack([lattice(theta=0.01 * i, shear=0.00, noise=0.08) for i in range(9)])
medium_stack = np.stack([lattice(theta=0.01 * i, shear=-0.03, noise=0.06) for i in range(9)])
fine_stack = np.stack([lattice(theta=0.01 * i, shear=-0.06, noise=0.04) for i in range(9)])
raw_001, den_001, residual_001 = raw, denoised, residual
raw_002 = lattice(theta=0.06, shear=0.02, noise=0.09)
den_002 = lattice(theta=0.08, shear=-0.03, noise=0.04)
residual_002 = (raw_002 - den_002).astype("float32")
```

## Choose the right Show2D pattern

| Review need | Use | Why |
| --- | --- | --- |
| One image or a short gallery | `Show2D(image_or_stack)` | Fast inspection with contrast, FFT, profile, ROI, and export controls |
| Raw / denoised / residual comparison | `Show2D([raw, denoised, residual], labels=...)` | Keeps panels in one linked inspection surface |
| Each panel has its own depth/time slider | `Show2D([stack_a, stack_b], labels=...)` | Every 3D list item gets independent frame controls |
| Many conditions, one page at a time | `Show2D(page_stack, page_labels=...)` | Keeps dense sweeps readable without one huge grid |
| Reproducible visual callouts | `panel_annotations`, `panel_overlays`, `inset_plots` | Saved state and exported HTML reproduce the same figure intent |
| Live editable figure callouts | `panel_overlays` + More -> Overlay Edit | Reproducible circles/rectangles can be selected, moved, resized, deleted, or reset in live/exported HTML |
| Measurement geometry with Python readback | ROI tools | ROIs are the path for statistics, FFT crops, and `get_roi_geometries()` readback |

## Start with a report-ready gallery

Give every panel a human-readable label, choose a fixed panel layout, and use
around-panel identity frames when the text refers to "blue", "green", or
"red" panels.

```{code-cell} python
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

```{code-cell} python
w = Show2D(
    [raw, denoised, residual],
    labels=["raw", "denoised", "residual"],
    ncols=3,
    ui_mode="presentation",
    show_stats=True,
)
w
```

Presentation mode is not a screenshot mode. The user can still open controls,
zoom, pan, tune contrast, and export from the embedded page.

## Rich math titles

Panel titles can use plain Unicode, compact TeX-style math, or structured
spans. Use this for symbols such as `λ`, `χ²`, `σ`, `μ`, and short units.

```{code-cell} python
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

```{code-cell} python
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

## Local labels inside panels

Use `panel_annotations` when the label belongs to a region inside an image,
not to the whole panel. Annotations can be keyed by panel label or panel index,
and each panel can have multiple labels. In the live widget, use
`More -> Overlay Edit` to drag these labels into their final figure position.

```{code-cell} python
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

## Geometric overlays

Use `panel_overlays` when a circle, rectangle, or square is part of a
reproducible figure specification. Coordinates use data pixels by default with
QuantEM's user-facing `(row, col)` convention.

```{code-cell} python
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
                "line_style": "dashed",
            },
            {
                "shape": "rect",
                "box": (48, 58, 126, 146),
                "stroke": "#facc15",
                "fill": "#facc15",
                "fill_opacity": 0.12,
                "dash": [6, 2, 1, 2],
            },
        ],
        "denoised": {
            "shape": "square",
            "center": (96, 88),
            "size": 42,
            "stroke": "#34d399",
            "stroke_width": 2,
            "line_style": "dotted",
        },
    },
)
```

Overlay strokes are solid by default. Use `line_style="dashed"`,
`line_style="dotted"`, or `line_style="dashdot"` for common figure styles, or
pass `dash=[on, off, ...]` for a custom pattern.

Use `overlays=[...]` for one shared guide on every panel, or include `panel=`
inside a flat list when overlays are generated in a loop:

```{code-cell} python
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

```{code-cell} python
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

## Inset plots

Use `inset_plots` when each image needs its own small curve, for example an
ACF-vs-r trace, a residual sweep, a dose curve, or a convergence metric. The
plot lives inside the panel, so it stays with the image in saved state and
exported HTML.

```{code-cell} python
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

## Publication SVG figures

Use these controls when `Show2D` is the source of a manuscript figure rather
than only an exploratory viewer. The live widget, exported HTML, and exported
SVG all use the same state: panel title placement, local labels, scale-bar
typography, inter-panel gutters, and vector overlays.

The examples in this section are executable documentation cells. They build a
small synthetic lattice so the rendered docs show the actual widget state
without depending on private publication data.

### Match the published PDF font

The drift-paper figures use a Helvetica-like sans-serif stack. On Linux,
`Nimbus Sans` or `Liberation Sans` usually substitutes for Helvetica/Arial.
On macOS, `Helvetica` or `Arial` is normally available. Put the preferred
family first and include fallbacks:

```{code-cell} python
PUBLICATION_FONT = (
    "Nimbus Sans, Helvetica, Arial, "
    "Liberation Sans, DejaVu Sans, sans-serif"
)
```

Use the same stack in every text layer that should match the paper:

```{code-cell} python
TITLE_STYLE = {
    "font_family": PUBLICATION_FONT,
    "font_weight": 700,
    "fg": "#ffffff",
    "outline_color": "#000000",
    "outline_width": 2.2,
    "align": "left",
    "x": 0.035,
    "y": 0.035,
    "anchor": "top-left",
}

SCALE_STYLE = {
    "font_family": PUBLICATION_FONT,
    "font_size": 16,
    "font_weight": 700,
    "color": "#ffffff",
    "outline_color": "#000000",
    "outline_width": 1.3,
    "bar_height": 5,
    "label_gap": 5,
    "offset": (0, -8),
}
```

`font_family` is a CSS/SVG font-family string. If the published PDF embeds
editable text, inspect the PDF fonts and put that family first. If the PDF
text was converted to paths, use the figure-generation notebook or the closest
installed Helvetica-like family.

To check a published PDF from a terminal:

```bash
pdffonts path/to/published_figure.pdf
```

If the PDF reports `Helvetica`, `Arial`, `NimbusSans`, or another embedded
font, put that font name first in `PUBLICATION_FONT`. If the PDF has no text
fonts, the labels were probably outlined or rasterized before publication; in
that case match the source figure script or use the closest installed
Helvetica-like family. The browser, exported SVG, and Illustrator can only use
fonts installed on the machine that opens the file, so keep common fallbacks in
the stack.

You can also ask the local font system which installed font will be used:

```bash
fc-match "Helvetica"
fc-match "Nimbus Sans"
```

### Change the font of panel labels

Whole-panel labels use `labels` for the text and `panel_title_style` for the
font and placement:

```{code-cell} python
w = Show2D(
    panels,
    labels=labels,
    ncols=3,
    show_panel_titles=True,
    panel_title_font_size=16,
    panel_title_style=TITLE_STYLE,
)
w
```

Use `x` and `y` as relative panel coordinates from 0 to 1. With
`anchor="top-left"`, `x=0.035`, `y=0.035` places the title near the upper-left
corner like a typical publication panel label. Use `offset=(dx, dy)` for final
pixel nudges after the relative placement is correct.

### Put multiple labels inside the same panel

Use `panel_annotations` for local labels such as `Ba+Ti` and `Sr`. These labels
are independent from the panel title, so they can sit below the title or over a
specific region of the image. They are exported as editable SVG text.

```{code-cell} python
annotation_labels = [
    "a 0° XEDS HAADF",
    "b 0° corrected XEDS HAADF",
    "c corrected 0°/90° ref",
    "d 0° Ba+Ti XEDS",
    "e 0° corrected Ba+Ti XEDS",
    "f composite on 0° HAADF",
]

w = Show2D(
    panels,
    labels=annotation_labels,
    ncols=3,
    panel_title_font_size=16,
    panel_title_style=TITLE_STYLE,
    panel_annotations={
        "f composite on 0° HAADF": [
            {
                "text": "Ba+Ti",
                "x": 0.20,
                "y": 0.15,
                "anchor": "center",
                "align": "center",
                "font_family": PUBLICATION_FONT,
                "font_size": 15,
                "font_weight": 800,
                "fg": "#ff4dff",
                "outline_color": "#000000",
                "outline_width": 1.8,
                "variant": "plain",
            },
            {
                "text": "Sr",
                "x": 0.80,
                "y": 0.15,
                "anchor": "center",
                "align": "center",
                "font_family": PUBLICATION_FONT,
                "font_size": 15,
                "font_weight": 800,
                "fg": "#58ff58",
                "outline_color": "#000000",
                "outline_width": 1.8,
                "variant": "plain",
            },
        ],
    },
)
w
```

For local annotations, `x=0.0, y=0.0` is the panel upper-left and
`x=1.0, y=1.0` is the lower-right. Start by placing labels where the chemistry
or feature appears, then use `anchor` and `align` to control whether the text
extends left, right, or centered from that point. `More -> Overlay Edit` can
fine-tune the final positions interactively without panning the image.

### Match black manuscript gutters and outer frame

For pixel-perfect manuscript grids, set the layer between panels, the outside
gallery frame, and the per-panel inner stroke independently:

```{code-cell} python
w = Show2D(
    panels,
    labels=labels,
    ncols=3,
    inter_panel_gap_px=2,
    inter_panel_gap_color="#000000",
    gallery_outer_border_px=2,
    gallery_outer_border_color="#000000",
    panel_inner_border_px=1,
    panel_inner_border_color="#000000",
)
w
```

With the settings above, `Show2D` places the first panel at `(2, 2)` in SVG
coordinates, inserts 2 pixels between panels, adds a 2-pixel frame around the
outside, and draws a 1-pixel inner stroke on each panel. To remove only the
between-panel layer while keeping individual frames, set `inter_panel_gap_px=0`
and keep `panel_inner_border_px > 0`.

### Put the scale bar on only one panel

`scale_bar_panels` accepts panel indices or labels. Use this when the paper
only shows a scale bar on one representative panel:

```{code-cell} python
w = Show2D(
    panels,
    labels=labels,
    sampling=0.0105,
    units="nm",
    scale_bar_panels=["f corrected combined"],
    scale_bar_length=2.0,
    scale_bar_label="2 nm",
    scale_bar_style=SCALE_STYLE,
    show_zoom_indicator=False,
)
w
```

Use `scale_bar_length` in the same physical units as `sampling`. Use
`scale_bar_label` when the text should be exactly what appears in the paper,
for example `"500 pm"` instead of an automatically formatted label.

### Export to Illustrator

```python
svg_path = w.export_svg("figure2_show2d.svg")
html_path = w.export_html("figure2_show2d.html", mode="single", encoding="uint8")
```

In the live widget, use `Export -> Preview SVG` before saving the SVG. This
replaces the canvas gallery with the generated SVG itself, so the preview is
the same payload that `Export -> SVG` downloads for Illustrator. Use this mode
for final panel-title placement, label nudges, gutter checks, scale-bar
positioning, and border inspection.

In the SVG, microscope image panels are embedded rasters. Labels, scale bars,
overlays, inset plots, group markers, colorbars, and panel frames are SVG
objects. In Illustrator, edit text and line work directly; keep the image
rasters as measured data unless you intentionally replace them.

Use this quick geometry check when exact borders matter:

```python
svg = svg_path.read_text()
assert '<image x="2" y="2"' in svg
assert 'stroke="#000000" stroke-width="1"' in svg
```

## Paging and local stacks

Use a list of 3D arrays when each panel has its own local frame axis. This is
different from `Show3D`, where panels usually share one global frame index.

```{code-cell} python
w = Show2D(
    [coarse_stack, medium_stack, fine_stack],
    labels=["coarse z", "medium z", "fine z"],
    panel_playback_fps=6,
    ncols=3,
)
w.set_panel_frame("fine z", 8)
w
```

For many conditions, use pages so each view stays readable:

```{code-cell} python
w = Show2D(
    np.stack([
        [raw_001, den_001, residual_001],
        [raw_002, den_002, residual_002],
    ]),
    page_labels=["frame 001", "frame 002"],
    labels=["raw", "denoised", "residual"],
    ncols=3,
)
w
```

## Export checklist

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
