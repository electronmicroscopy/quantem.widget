# ChooseLattice

Pick an ordered origin and two lattice-vector points on a 2D image. Displays
a single image, lets you wheel-zoom and drag-pan to inspect a region, and
lets you click 3 ordered points whose pixel coordinates (in the ORIGINAL,
un-zoomed image) are exposed for downstream lattice-vector calculations.

```python
import numpy as np
from quantem.widget import ChooseLattice

widget = ChooseLattice(image, cmap="gray")
```

After clicking the origin, then the two lattice points on the image:

```python
widget.origin   # (row, col) or None
widget.u        # second point - origin, or None until both are placed
widget.v        # third point - origin, or None until both are placed
widget.points   # the picked (row, col) pairs so far; assign to set them from Python
```

## Reference

```{eval-rst}
.. autoclass:: quantem.widget.choose_lattice.ChooseLattice
   :members:
   :show-inheritance:
```

## Interactive controls

| Control | Trait | Expected effect |
|---|---|---|
| Click on the image (fewer than 3 points placed) | `points` | Appends the next ordered point |
| Drag an existing point | `points` | Adjusts that point's pixel coordinates in place |
| Clear Points button | `points` | Resets to no points |
| Pan (drag) / zoom (wheel) | view transform | Image translates / zooms about the cursor |
| Double-click | view transform | Resets zoom/pan |
