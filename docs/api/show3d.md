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
| Panel layout (multi-panel) | `n_panels`, `link_panels`, `max_cols` | Panels arrange; linked scrub moves all |
| Panel visibility (multi-panel) | `hidden_panels` | Panels collapse from view without deleting data |
| Statistics | `show_stats` | Optional mean/min/max/std readout |

## Live stack updates

Use `set_image()` to replace the stack in an already displayed widget while a
notebook kernel is still running. For live acquisitions or reconstruction loops,
construct the widget with `offline=False` so frames travel over the live
Jupyter Comm channel instead of the saved/offline notebook-data path:

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

The statistics readout is off by default. Turn on `show_stats=True` in Python,
or use the `Stats` switch in the widget, when mean/min/max/std values are useful.

```{note}
`export_html(quantized=True)` writes the smaller uint8 pack; the default writes
exact float32. See the [widget export tutorial](../tutorials/widget_export).
```
