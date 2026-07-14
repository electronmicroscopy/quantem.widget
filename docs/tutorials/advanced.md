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
- **Full-resolution Show3D folder viewers** - keep the microscope/reconstruction
  pixels at native shape while the browser loads a nearby data file instead of
  one giant HTML blob.
- **Rich math labels and annotations** - use λ, χ², colored spans, and local
  region labels in Show2D/Show3D panels without exposing raw TeX markup.
- **Show2D inset plots** - put a small calibration curve inside each image
  panel, so a denoise or reconstruction sweep carries the metric that explains
  why the scientist should trust that panel.

(full-resolution-show3d-folder-viewers)=

## Full-resolution Show3D folder viewers

Use a Show3D folder viewer when the review question is still a microscope
question: "Does this denoise frame preserve the atom columns?", "Does the
drift-corrected panel fail at the edge?", or "Can I scrub the full sweep without
waiting for Python?" A single HTML file is convenient for email, but it is the
wrong container for multi-GB, multi-panel, full-shape stacks.

The folder export keeps the viewer HTML small and stores the frame data beside
it:

```text
800C_1.3Mx_fullres/
├── index.html
├── manifest.json
└── offline_stack.u8
```

The current advanced Show3D API for this path is `export_sidecar(...)`. In user
docs, think of it as a folder export: the HTML is the viewer and the nearby data
file is the microscope stack.

Choose the export path by what the scientist needs to do:

| Review need | Recommended path | What it means |
| --- | --- | --- |
| Email or attach one browseable artifact | `export_html(..., mode="single", encoding="uint8", downsample=...)` | Compact review copy. Any downsample is explicit. |
| Scrub a multi-GB full-shape Show3D stack locally | `export_sidecar(out_dir)` | Small HTML plus nearby data file; serve over HTTP. |
| Recompute, stream fresh frames, or inspect exact backend arrays | live Jupyter widget | Python owns the data and the browser is the viewer. |

Create the review with the same panel structure a microscopist should inspect:

```python
from quantem.widget import Show3D

w = Show3D(
    raw_stack,
    tikhonov_stack,
    tv2_stack,
    panel_titles=["raw", "Tikhonov", "TV2"],
    title="800C 1.3Mx denoise full-resolution review",
    display_bin=1,
    link_contrast=False,
    debug=True,
)
w.export_sidecar("/data/reports/800C_1.3Mx_fullres")
```

`display_bin=1` is the important claim: the browser review is built from the
native panel shape. If you make a smaller browse copy, say so in the filename,
title, report, and export option.

Serve the folder with a Range-capable local HTTP server. Do not open the HTML
with `file://`; the browser must be allowed to fetch `offline_stack.u8`.

```bash
# Use your project or lab helper that supports HTTP Range requests.
python scripts/serve_sidecar_range.py \
  --dir /data/reports/800C_1.3Mx_fullres \
  --port 8803 --bind 127.0.0.1
```

Then open:

```text
http://127.0.0.1:8803/index.html
```

For a correct full-resolution folder review, the scientist-facing behavior is:

- The viewer shell appears immediately, before all frames are loaded.
- A load banner says whether the stack is still loading, cached, or ready.
- Real pixels appear as soon as the first visible frame is available.
- After the stack is ready, scrubbing and playback use browser-local cached
  frames instead of asking Python for each frame.
- Histogram and colormap changes repaint the current microscope view quickly,
  then rebuild any playback cache in the background.
- Zooming into a panel should not make the widget recompute hidden panels or
  unrelated frames before the visible view updates.

Before trusting a full-resolution export, run a small Range probe. For a
9-panel `2048 x 2048` uint8 stack, one frame is `9 * 2048 * 2048` bytes:

```bash
FRAME_BYTES=$((9 * 2048 * 2048))
curl -sD /tmp/headers.txt -o /tmp/one-frame.u8 \
  -H "Range: bytes=0-$((FRAME_BYTES - 1))" \
  http://127.0.0.1:8803/offline_stack.u8
head /tmp/headers.txt
ls -lh /tmp/one-frame.u8
```

The response should be `206 Partial Content`, and the downloaded file should be
one frame, not the whole multi-GB stack. This only proves delivery; it does not
replace browser testing.

Use this checklist for the browser signoff:

| User story | What to drive | What to report |
| --- | --- | --- |
| Open the review | load the URL in a fresh browser | `load_s`, first visible frame, ready banner, errors |
| Scrub like a microscopist | 20 ArrowRight or slider steps | advanced frame count, mean/p50/p95 ms |
| Play the time series | 5 s playback | transitions, effective FPS, dropped/stuck frames |
| Inspect detail | zoom into a panel, then scrub | visible panel updates, no hidden-panel stall |
| Tune display | drag histogram for 5 s | preview FPS, p50/p95 interval, release ms, cache rebuild s |
| Change contrast language | switch colormap and scale | menu/action latency, current image repaints |
| Look for leaks | repeat buttons, scrub, play, histogram, zoom | JS heap, DOM nodes/listeners, cache counts, RSS trend |

Example report line from a full-resolution `9 x 69 x 2048 x 2048` Show3D
folder viewer on Phil:

```text
load_s=3.8; histogram drag 5 s: 114 preview paints, 22.8 fps,
interval p50/p95=16.9/80.6 ms, render mean/p95=15.8/22.9 ms,
release=5.2 ms, cache clean=1.3 s; scrub after change 20/20,
mean/p95=2.1/4.5 ms; memory stress: JS heap +2 MB, listeners flat,
GPU/cache frame counts stable, RSS plateaued after initial allocation.
```

That example is intentionally written like a test result, not a feeling. If the
target is 30 FPS and the measured histogram drag is 22.8 FPS, report it as a
partial pass even when the widget is usable.

Common failure modes:

- **Blank or white viewer**: the browser did not fetch the stack, loaded stale
  HTML, or hit an allocation failure. Check the console and the load banner.
- **`Array buffer allocation failed`**: avoid one contiguous multi-GB fetch and
  restart the browser before final proof if repeated reloads fragmented memory.
- **Scrub is slow after "ready"**: profile draw/colormap/upload/cache counters;
  the data is probably in RAM, but the paint path is too expensive.
- **Histogram feels slow**: test drag preview separately from release/cache
  rebuild. The current view should update first; background cache can follow.
- **Fast only because it is smaller**: check `display_bin`, `downsample`, title,
  and report wording. Full-resolution claims must keep shape reduction explicit.

(label-local-regions-inside-panels)=

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

(rich-math-labels-and-presentation-exports)=

## Rich math labels and presentation exports

Scientific comparison panels often need symbols rather than prose: `λ` for a
regularization sweep, `χ²/pixel` for a residual score, or colored fragments in
one title. Plain Unicode symbols work directly in panel titles and annotations:

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

For richer title chrome, pass `panel_title_spans`. Each span can contain
`text`, `math`, and optional `color`. The same rendered title is used in the
panel image, the `Panels` menu, the stats row, saved notebook state, and
standalone HTML:

```python
rich_titles = [
    [{"math": r"\lambda=0.03"}, {"text": " raw"}],
    [{"text": "denoised", "color": "#34d399"}],
    [{"math": r"\chi^2"}, {"text": "/pixel residual"}],
]

Show2D(
    [raw, denoised, residual],
    panel_title_spans=rich_titles,
    ncols=3,
    show_stats=True,
)

Show3D(
    raw_stack,
    denoised_stack,
    residual_stack,
    panel_titles=rich_titles,
    show_stats=True,
)
```

Use `panel_annotations` when the label belongs to a region inside a panel rather
than to the whole panel. A panel can have several annotations, and each one can
also use `math` or `spans`:

```python
Show2D(
    [raw, residual],
    panel_title_spans=[
        [{"math": r"\lambda=0.03"}, {"text": " raw"}],
        [{"math": r"\chi^2"}, {"text": "/pixel residual"}],
    ],
    marker_style="around",
    marker_colors=["#60a5fa", "#f87171"],
    panel_annotations={
        0: [
            {"math": r"\lambda", "position": "top-left", "variant": "pill"},
            {
                "spans": [
                    {"text": "ROI "},
                    {"text": "A", "color": "#60a5fa"},
                ],
                "box": [0.18, 0.25, 0.30, 0.16],
                "variant": "callout",
            },
        ],
        1: {
            "spans": [{"math": r"\chi^2"}, {"text": " high", "color": "#f87171"}],
            "position": "top-right",
            "variant": "outline",
        },
    },
)
```

Presentation mode keeps this same rich label rendering but starts with the
controls collapsed, which is useful for shared notebooks and exported HTML:

```python
w = Show3D(
    raw_stack,
    residual_stack,
    panel_titles=[
        [{"math": r"\lambda=0.03"}, {"text": " raw"}],
        [{"math": r"\chi^2"}, {"text": "/pixel residual"}],
    ],
    ui_mode="presentation",
)
w.export_html("regularization_review.html", encoding="uint8")
```

The collapsed view still shows `Controls` and `Export` in the title chrome.
In live Jupyter, Show3D's Export menu can write HTML and request GIF/MP4
animation exports through Python. In standalone HTML, the page can download
itself as HTML; GIF/MP4 entries are visible but disabled because browser-side
movie encoding is not implemented yet.

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
