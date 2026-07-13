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
- **Show2D inset plots** - put a small calibration curve inside each image
  panel, so a denoise or reconstruction sweep carries the metric that explains
  why the scientist should trust that panel.

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
