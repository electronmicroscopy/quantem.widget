# All diffraction comparison (live)

Given a live Show4DSTEM with multiple datasets, when `compare_dp_mode="all"`
is selected, show each visible dataset's native diffraction pattern at the
same scan position. Keep signed values intact and use one absolute display
range across the grid. Scan selection updates every pattern. Clicking a grid
pattern moves the common detector center; the primary diffraction panel retains
the aperture geometry, radius, zoom and other tools.

The diffraction row uses the same comparison component as the virtual-image
row: wheel zoom, Shift-drag pan, Reset/double-click, scale bars, responsive
columns, panel selection, hide/star/reorder and resize. Diffraction scale bars
use detector pixels unless reciprocal-space calibration is provided. Color and
linear/signed-log settings follow the main diffraction controls. FFT/profile
tools remain in the primary diffraction panel, not each comparison tile.

Select Circle, Square or Rect in the scan ROI controls below the virtual
images, with Mean as the reduction. Each method is reduced independently over
identical scan positions. No averaging across methods occurs. Dragging updates
the shared outline on animation frames and requests live native reductions.
One kernel request is in flight; intermediate positions coalesce to the latest
pointer position. A `vi_roi_receipt` acknowledges each completed reduction,
including unchanged patterns, before the next pending position is sent. The
final pointer position is retained. Off restores the point patterns. This does
not establish 120 FPS derived-data updates: measure actual five-panel paints
separately from animation-frame cadence and backend execution. The receipt
contains row, col, backend wall milliseconds and revision; it is not a GPU-only
timer or proof that the browser painted the result. Native arrays remain signed and
unchanged. An empty selection must not be presented as a valid mean.

This additive layout reuses QuantEM.GPU's WebGPU range/colormap/compositing
engine. It does not apply denoising or change the original data. The existing
average/selected modes are unchanged. Python transfers only the requested
native patterns, not a second 4D cube. The all-pattern grid is live-kernel-only
at this stage; kernel-free exports, lazy-page stress and physical-phone behavior
require separate qualification. Do not claim these from a live dense-cube test.

Example:

```python
Show4DSTEM(stack, view_mode="multiple", compare_dp_mode="all",
           frame_labels=["Noisy", "SHINE", "Fourier", "Sadri", "Fusion"])
```
