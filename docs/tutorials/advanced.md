# Advanced

Deeper workflows for when the basic tutorials are not enough - each one built
around a real question from the microscope room:

- **ShowFolder session browser** - triage a finished session: thumbnail it,
  star the good fields of view, open them directly.
- **HTML and file export** - hand a collaborator one interactive HTML file, a
  figure, or a full report package.
- **Saving GIF and MP4 movies** - write raw arrays or widget-rendered views to
  GIF and MP4, with CUDA MP4 compression on NVIDIA workstations.
- **Memory management** - know what a load will cost in RAM/VRAM, pick the
  GPU, and give the memory back.
- **Live folder watching** - keep one widget open at the scope and let new
  acquisitions appear in it.
- **Comparing reconstructions** - panels with independent frame stacks for
  regularization sweeps, convergence checks, and slice-count comparisons.
- **Local panel annotations** - label multiple local regions inside Show2D or
  Show3D panels without turning those notes into whole-panel titles.
- **Show2D inset plots** - put a small calibration curve inside each image
  panel, so a denoise or reconstruction sweep carries the metric that explains
  why the scientist should trust that panel.

## Label local regions inside panels

Use whole-panel `labels` / `panel_titles` for panel identity, `marker_colors`
or group frames for visual identity, and `panel_annotations` for local notes
inside a panel. Annotations are useful for ROI names, dose/status badges,
χ² labels, residual warnings, and callouts that should stay attached to the
image when the widget is saved or exported.

Each annotation is JSON-safe and survives notebook state plus
`export_html(...)`. The default style is a readable badge over scientific
images. Built-in variants are `badge`, `pill`, `plain`, `outline`, and
`callout`.

### Show2D: multiple labels on one panel

`panel_annotations` can be a dictionary keyed by panel index or panel label.
Each value can be one annotation or a list of annotations.

```python
import numpy as np

from quantem.widget import Show2D

rng = np.random.default_rng(12)
yy, xx = np.mgrid[-1:1:192j, -1:1:192j]
base = np.exp(-((xx * 1.3) ** 2 + (yy * 0.9) ** 2) * 3)
images = np.stack(
    [
        base + 0.15 * np.sin(3 * np.pi * xx),
        base + 0.12 * np.cos(4 * np.pi * yy),
        base - 0.35,
    ]
).astype("float32")

Show2D(
    images,
    labels=["raw", "filtered", "residual"],
    ncols=3,
    marker_colors=["#60a5fa", "#34d399", "#f87171"],
    marker_style="around",
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
            {
                "text": "same region",
                "box": [0.18, 0.43, 0.30, 0.12],
                "variant": "outline",
                "border_color": "#facc15",
            },
        ],
        "filtered": {
            "text": "point label",
            "x": 0.68,
            "y": 0.32,
            "anchor": "center",
            "bg": "rgba(255,255,255,0.55)",
            "fg": "#111827",
        },
        "residual": {
            "text": "check residual",
            "position": "bottom-center",
            "variant": "plain",
            "font_size": 13,
        },
    },
)
```

### Show2D: single-panel shorthand

For a single image, pass a list directly. Every item labels the only panel, so
you do not need to write `panel=0`.

```python
Show2D(
    images[0],
    panel_annotations=[
        {"text": "single panel", "position": "top-left", "variant": "pill"},
        {
            "text": "no panel= needed",
            "position": "bottom-right",
            "variant": "outline",
            "border_color": "#facc15",
        },
    ],
)
```

### Show3D: target panels by title

For multi-panel Show3D, a flat list is often easiest. Include `panel=` on each
annotation and target either a panel index or a `panel_titles` value.

```python
from quantem.widget import Show3D

Show3D(
    raw_stack,
    denoised_stack,
    residual_stack,
    panel_titles=["raw stack", "denoised stack", "residual stack"],
    marker_style="around",
    panel_annotations=[
        {"panel": "raw stack", "text": "input", "position": "top-left"},
        {
            "panel": "raw stack",
            "text": "same panel",
            "position": "bottom-left",
            "variant": "outline",
            "border_color": "#facc15",
        },
        {
            "panel": "residual stack",
            "spans": [
                {"text": "χ² "},
                {"text": "high", "color": "#f87171"},
            ],
            "x": 0.5,
            "y": 0.18,
            "anchor": "top-center",
        },
        {
            "panel": "residual stack",
            "text": "region box",
            "box": [0.56, 0.48, 0.32, 0.16],
            "variant": "callout",
            "bg": "rgba(0,0,0,0.65)",
        },
    ],
)
```

### Placement and style reference

Use corner placement for badges, normalized points for callouts tied to a
feature, and normalized boxes when the label should occupy a local region.

```python
{"text": "corner", "position": "top-left"}
{"text": "point", "x": 0.62, "y": 0.35, "anchor": "center"}
{"text": "region", "box": [0.20, 0.25, 0.30, 0.18]}
```

Common style keys are `bg`, `fg`, `border_color`, `border_width`,
`font_size`, `font_weight`, `pad_x`, `pad_y`, `radius`, `opacity`, `align`,
`max_width`, and `class_name`. `class_name` is useful when exported HTML needs
project-specific CSS or when a browser test needs a stable selector.

### Math symbols in labels

Plain Unicode symbols work directly in panel titles and annotations:

```python
Show2D(images, labels=["λ=0.01", "χ² / pixel"])
```

For notebook code that should read like math, use inline TeX between `$...$`
or a structured `math` span. The widget renders common Greek symbols and
simple superscripts/subscripts without loading MathJax or KaTeX.

```python
Show2D(
    images,
    labels=[
        r"$\lambda=0.01$ raw",
        r"$\chi^2$/pixel residual",
    ],
    panel_annotations={
        0: {"math": r"\lambda", "position": "top-left", "variant": "pill"},
        1: {
            "spans": [
                {"math": r"\chi^2"},
                {"text": "/pixel"},
            ],
            "position": "top-right",
            "variant": "outline",
        },
    },
)

Show3D(
    raw_stack,
    residual_stack,
    panel_titles=[
        [{"math": r"\lambda=0.01"}, {"text": " object"}],
        r"$\chi^2$/pixel",
    ],
)
```

This math support is intentionally compact: use it for labels such as
`λ`, `χ²`, `σ`, `μ`, `Δ`, subscripts, superscripts, and short units. For a
full equation, keep the equation in Markdown/LaTeX near the widget and use a
short annotation inside the panel.

Prefer Python raw strings for TeX-style labels so the notebook source stays
readable. If a label comes from JSON, a widget state file, or another exporter
with doubled backslashes, the frontend normalizes those sequences before
rendering so `\\lambda` displays as `λ`, not `\λ`.

```python
math_labels = [
    r"$\lambda=0.03$ raw",
    r"$\chi^2$/pixel residual",
]

Show2D(images[:2], labels=math_labels)
Show3D(raw_stack, residual_stack, panel_titles=math_labels)
```

## Add inset plots to Show2D panels

A common review moment is not only "which image looks best?", but "which image
matches the calibration curve?" For example, during denoising calibration a
scientist may compare several Show2D panels while also tracking an ACF score,
residual ratio, dose curve, or other small diagnostic trace. `inset_plots` lets
each image panel carry that trace directly inside the panel, without creating a
separate Matplotlib figure cell that drifts away from the image.

The example below keeps the widget call readable. Each dictionary describes one
inset plot. When a list is supplied, the first plot is drawn on the first panel,
the second on the second panel, and so on. A single dictionary can also be
broadcast to every panel when the same curve should appear everywhere.

```python
import numpy as np

from quantem.widget import Show2D

rng = np.random.default_rng(4)
yy, xx = np.mgrid[:192, :192].astype("float32")
base = (
    1.0
    + 0.7 * np.cos(2 * np.pi * xx / 12)
    + 0.7 * np.cos(2 * np.pi * yy / 12)
)

images = []
insets = []
labels = []
for i, sigma in enumerate([0, 2, 4, 6]):
    noisy = base + rng.normal(0, 0.18, base.shape)
    # In a real notebook this would be the denoised reconstruction for sigma.
    preview = noisy if sigma == 0 else 0.75 * base + 0.25 * noisy
    images.append(preview.astype("float32"))
    labels.append(f"σ={sigma} preview")

    radius = np.linspace(0.1, 1.0, 16)
    acf = np.exp(-radius * (0.9 + 0.08 * sigma))
    insets.append(
        {
            "x": radius,
            "y": acf,
            "point": min(i * 4 + 2, len(radius) - 1),
            "xlabel": "R",
            "ylabel": "ACF",
            "legend": f"σ={sigma}",
            "position": "bottom-right",
            "margin": 0.04,
            "size": 0.34,
            "height": 0.24,
            "line_width": 2.2,
            "point_color": "#ffcc00",
            "background_alpha": 0.72,
            "show_ticks": True,
            "tick_font_size": 10,
            "label_font_size": 11,
            "legend_font_size": 11,
        }
    )

Show2D(
    images,
    labels=labels,
    ncols=2,
    cmap="inferno",
    inset_plots=insets,
    scale_bar_position="bottom-left",
    show_zoom_indicator=False,
)
```

When `inset_plots` is initialized, Show2D adds an `Inset Chart` switch under
`More`. Use it to hide or restore the charts without changing the data. The
charts can also be dragged directly inside the panel: while dragging, the chart
follows the pointer; when released, it snaps to the nearest corner and stores
that corner plus a margin in widget state.

### Position, sizing, and publication style

Use `position` when the plot belongs in a corner and `margin` when it needs a
little breathing room from the image edge. `size` is the inset width as a
fraction of the panel width, and `height` is the height as a fraction of the
panel height. These fractional controls make the same notebook behave naturally
when the panel is resized, exported to HTML, or saved as a notebook preview.

For figure-like layouts, use `box` for exact placement:

```python
inset_plots=[
    {
        "x": ratio,
        "y": acf,
        "xlabel": "R ratio",
        "ylabel": "ACF",
        "legend": "chosen σ",
        "box": [0.60, 0.58, 0.34, 0.26],  # x, y, width, height in panel units
        "background": "#000000",
        "background_alpha": 0.35,
        "border_width": 0,
        "text_color": "#ffffff",
        "tick_color": "#ffffff",
        "line_width": 3,
    }
]
```

If the inset competes with the scale bar, move the scale bar instead of hiding
the science:

```python
Show2D(
    image,
    inset_plots=calibration_plot,
    scale_bar_position="bottom-left",
    show_zoom_indicator=False,
)
```

### Hover readout and saved outputs

In the browser, hovering over the inset reports the nearest plotted coordinate
using the axis labels, for example `R 0.47 · ACF 0.48`. This is meant for the
human review loop: a scientist can point at the diagnostic curve and read the
exact value without leaving the image panel.

The inset specification is part of the Show2D widget state. It is included in
notebook state, static PNG fallback previews, and `export_html(...)`, so a saved
notebook or standalone HTML report keeps the same image-plus-diagnostic view
that the scientist inspected interactively.

```python
viewer = Show2D(
    images,
    labels=labels,
    inset_plots=insets,
    scale_bar_position="bottom-left",
    show_zoom_indicator=False,
)

viewer.save_image("denoise_calibration_panel.png")
viewer.export_html("denoise_calibration_panel.html")
```

For the complete parameter reference, see the [Show2D API page](../api/show2d.md).

```{tableofcontents}
```
