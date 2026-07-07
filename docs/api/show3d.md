# Show3D

A 3D volume scrubbed slice by slice, with playback and an interactive-HTML
export. See the [Show3D tutorial](../tutorials/show3d).

## Reference

```{eval-rst}
.. autoclass:: quantem.widget.show3d.Show3D
   :members:
   :show-inheritance:
```

## Interactive controls

| Control | Trait | Expected effect |
|---|---|---|
| Slice slider | `slice_idx` | Canvas shows that depth slice |
| Arrow keys | `slice_idx` | Step one slice per press |
| Play / pause | `playing` | Auto-advances slices at `fps` |
| Reverse | `reverse` | Playback direction flips |
| Boomerang | `boomerang` | Ping-pongs at the ends instead of looping |
| FPS field | `fps` | Playback rate changes |
| Loop range | `loop_start`, `loop_end` | Playback confined to the sub-range |
| Colormap dropdown | `cmap` | Canvas recolors |
| Export button | `export_request`, `export_status` | Writes a standalone HTML viewer |
| Page slider (paged galleries) | `page_idx`, `n_pages`, `panels_per_page` | Shows one page of panels at a time |
| Page star (paged galleries) | `page_starred`; `star_page()`, `unstar_page()` | Marks a whole page as worth revisiting |
| Panel layout (multi-panel) | `n_panels`, `link_panels`, `max_cols` | Panels arrange; linked scrub moves all |
| Panel visibility (multi-panel) | `hidden_panels` | Panels collapse from view without deleting data |
| Panel reorder (multi-panel) | `panel_order`; `set_panel_order()`, `move_panel()`, `reset_panel_order()` | Reorders panel display without changing source data, labels, stars, or hidden state |
| Viewer chrome preset | `ui_mode` plus explicit `show_*` kwargs | Applies shared display presets; see [Viewer UI controls](viewer-ui) |
| Control visibility | `show_controls`, `controls_collapsed`; `collapse_controls()`, `expand_controls()`, `toggle_controls()` | Permanently remove controls or temporarily collapse them behind the top GUI toggle |
| Title visibility | `show_title` | Top title row shows/hides |
| Statistics | `show_stats` | Optional mean/min/max/std readout |
| Panel title visibility | `show_panel_titles`, `panel_title_font_size` | Per-panel labels show/hide and resize |
| Scale bar visibility | `show_scale_bar` (`scale_bar_visible` in saved state) | Scale bar shows/hides |
| Resize / zoom chrome | `show_resize_handles`, `show_zoom_indicator` | Resize handles and zoom readout show/hide |

## Live stack updates

Use `set_image()` to replace the stack in an already displayed widget while a
notebook kernel is still running. Keep a reference to the widget, display it
once, then call `set_image(...)` whenever new stack data should be rendered.
For live acquisitions or reconstruction loops, construct the widget with
`offline=False` so frames travel over the live Jupyter Comm channel instead of
the saved/offline notebook-data path:

```python
import numpy as np
from quantem.widget import Show3D

frames = [first_frame]
w = Show3D(first_frame[None], labels=["frame 1"], offline=False)
w

for next_frame in acquisition:
    frames.append(next_frame)
    w.set_image(
        np.stack(frames),
        labels=[f"frame {i + 1}" for i in range(len(frames))],
    )
    w.slice_idx = len(frames) - 1
```

In a real JupyterLab browser session this updates the displayed frame as each
`set_image()` call is processed. A background thread is optional for UI
ergonomics, but is not required for the widget update itself.

`set_image()` is the re-render trigger. Mutating the original NumPy array in
place does not notify the frontend. The method writes a fresh current-frame
transfer, bumps `frame_seq`, invalidates playback buffers, clamps the current
slice and loop range, and resets stale panel-specific state when replacing a
multi-panel view with a single stack.

```{important}
Do not use the default tiny-stack constructor path for acquisition-style live
updates. Small stacks may auto-enable the offline notebook representation, which
is intended for saved notebooks and static exports. Pass `offline=False` when the
stack will grow over time.
```

## Panel visibility

Use panel visibility when a secondary panel is useful for validation but should
not take space in the first view. For example, an SSB reconstruction can keep
the mean diffraction pattern in the widget while hiding it from the canvas:

```python
w = Show3D(
    ssb_stack,
    mean_dp_stack,
    panel_titles=["SSB reconstruction", "Mean DP"],
    hidden_panels=["Mean DP"],
)
```

Panel references can be zero-based indices or exact panel titles:

```python
w.hide_panel("Mean DP")
w.hide_panel(1)
w.show_panel("Mean DP")
w.show_all_panels()
```

Hidden panels stay in the widget state and standalone HTML export. They are not
removed from the data, and readers can restore them from the `Panels` menu.

Use panel reordering when the comparison order should change without copying or
rebuilding the source stacks:

```python
w.set_panel_order(["Probe", "SSB reconstruction", "Mean DP"])
w.move_panel("SSB reconstruction", 0)
w.reset_panel_order()
```

Panel order is saved in widget state and standalone HTML. It is display-only:
hidden panels, stars, titles, and per-panel contrast remain keyed by the
original source panel index.

The statistics readout is off by default. Turn on `show_stats=True` in Python,
or use the `Stats` switch in the widget, when mean/min/max/std values are useful.

## Paged galleries

Use pages when each view is itself a small multi-panel movie. This is useful for
reconstruction sweeps, denoising parameters, or iteration checkpoints where each
page should show the same panel layout while the user scrubs pages:

```python
# Shape: pages, panels_per_page, frames, rows, cols
w = Show3D(
    stacks_5d,
    panel_titles=["raw", "filtered", "residual", "probe"],
    page_labels=["lambda 0.01", "lambda 0.03", "lambda 0.10"],
)
```

You can also pass explicit page dictionaries:

```python
w = Show3D([
    {"title": "iteration 10", "stacks": [raw_10, filtered_10, residual_10]},
    {"title": "iteration 20", "stacks": [raw_20, filtered_20, residual_20]},
])
```

Paged Show3D keeps the data in the normal multi-panel transport internally, so
HTML export, notebook state, panel hiding, panel stars, playback, FFT, and GIF/MP4
export use the same paths as ordinary Show3D. In page mode, `visible_panels`
returns only panels from the active page, and `to_show2d()` converts the current
visible page into a Show2D gallery.

```python
w.page_idx = 2
w.star_page(2)
show2d_page = w.to_show2d(frame=w.slice_idx)
```

## Animation exports

Use HTML when collaborators should keep scrubbing, zooming, and changing
contrast. Use GIF or MP4 when the result needs to drop into PowerPoint, email, or
a static report:

```python
w.save_gif("movie.gif", quality="medium", fps=6)
w.save_mp4("movie.mp4", quality="high", fps=12)
```

`quality="low"`, `"medium"`, and `"high"` control the exported spatial
resolution. GIF is always palette-limited, so medium is usually the practical
slide/email choice; high is sharper but larger. Pass `show_frame_labels=True`
when panel titles should include the same live-style frame label and count that
the widget canvas shows. GIF/MP4 exports keep the panel labels, scale bar, and
zoom readout styling consistent with the static/offline widget image output.
The widget **Export** menu keeps the common path simple: choose `GIF low`,
`GIF medium`, `GIF high`, or the matching MP4 option. The size shown in that
menu is estimated uncompressed RGB render work, so the final GIF/MP4 file is
usually smaller but can vary with image texture and palette compression.

```python
w.save_gif("movie.gif", quality="medium", fps=6, show_frame_labels=True)
```

The GIF/MP4 path exports the full panel frames. Browser-only zoom and pan
gestures are view state, so use HTML export when collaborators need to continue
zooming, panning, or changing contrast interactively.

Advanced animation choices stay in Python and the maintainer smoke report
rather than crowding the widget toolbar. Use the Python API for frame labels,
background color, bounce playback, and other presentation-specific choices:

```python
w.save_gif(
    "movie.gif",
    quality="medium",
    fps=6,
    playback="bounce",
    show_frame_labels=True,
    background="black",
)
```

```{note}
`export_html(quantized=True)` writes the smaller uint8 pack; the default writes
exact float32. See the [widget export tutorial](../tutorials/widget_export).
```
