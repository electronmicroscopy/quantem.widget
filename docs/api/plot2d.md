# Plot2D

See the [interactive tutorial](../tutorials/plot2d.ipynb).

Use `quantem.widget.Plot2D` for a calibrated scalar map, such as G3 versus
distance and angle. Use Show2D for spatial images. Plot2D owns axes, colorbar,
zoom/pan and hover; it does not calculate correlations or train a model.

```python
import numpy as np
import quantem.widget as qw

qw.profile()
radius = (np.arange(100) + 0.5) * 0.1
angle = (np.arange(36) + 0.5) * 5
values = np.cos(np.deg2rad(angle[:, None])) ** 2 * radius[None, :]
plot = qw.Plot2D(
    values, x=radius, y=angle,
    x_label="Second-neighbor distance (Å)", y_label="Shared-root angle (°)",
    colorbar_label="Illustrative value", width=500, max_width=600,
)
plot
```

`x` contains column bin centers; `y` contains row bin centers. Both must be
increasing uniform grids. Row zero is at the bottom, matching Cartesian plots.
Source values and browser hover transport retain float64. The shared QuantEM
WebGPU colormap renderer uses float32 display buffers. Canvas fallback is
explicitly labeled when WebGPU is unavailable; it is not GPU acceleration.

Wheel over the map to zoom, then drag to pan. Zoom buttons and Reset View are
also available. The Color menu changes map and colorbar together; each plot is
independent. A browser-local animation-frame scheduler handles gestures without
Python round trips. Stable view bounds are saved after interaction.

Map pixels, hover values and color-scale metadata update together after
rendering completes, including during rapid replacements.

`plot.set_data(next_values)` preserves the original grid, color limits and
viewport. `plot.horizontal_line = 92.5` adds an angle-reading line without
resending the map. `plot.figure()` returns a closed Matplotlib figure with
the current axes, values, colormap and viewport, for example
`plot.figure().savefig("g3.svg")`. By default, saved snapshots omit the map array and retain a static PNG
preview. Use `save_state=True` to embed the complete float64 map for interactive
restoration in supporting frontends. Static previews record the view when
Python renders the preview or creates a full snapshot; they do not track
browser-only gestures continuously. Notebook-manager save/restore behavior
varies by frontend. Keep scientific data files separately.

## Current scope

This API targets small, finite scalar maps, not large spatial images. It does
not support nonuniform coordinates, logarithmic axes, or standalone
`export_html`. Full interactive embedding is opt-in with `save_state=True`;
keep large data outside notebooks. Static previews use stride sampling above
512 bins per axis, preserve calibrated bounds, and are for viewing only. Controls, axes and the canvas follow the
notebook or documentation light/dark theme.

## Reference

```{eval-rst}
.. autoclass:: quantem.widget.Plot2D
   :members: set_data, figure
```

## Interactive controls

| Control | Behavior |
|---|---|
| Color | Recolor the map and scale without modifying values. |
| Zoom In / Zoom Out | Zoom about the viewport center. |
| Wheel / drag | Zoom about the pointer; pan within the full grid. |
| Reset View / double-click | Restore full physical bounds. |
| Save PNG | Save the current canvas, including labels and color scale. |
| Hover | Inspect original `(row, col)`, calibrated coordinates and value. |
