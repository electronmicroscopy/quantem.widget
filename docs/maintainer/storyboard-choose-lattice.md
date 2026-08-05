# ChooseLattice Storyboard

Use with [Storyboard](storyboard).

## Stories

### CL-01: Open A Real Lattice Image Quickly

**User story**: As a microscopist about to measure a crystal's lattice
vectors, I want a useful preview of the atomic-resolution image in about a
second so I can start placing points without waiting.

**Primary widgets**: ChooseLattice.

**Data to use**: real atomic-resolution HAADF/ABF image with a visible
periodic lattice; a large real image (4k or larger) for the stress variant.

**Acceptance checks**:

- Load from a NumPy/PyTorch array, from a quantem `Dataset2d`, and from an
  in-memory real file load on the backend.
- Measure first visible paint and note image shape, dtype, and native bytes.
- Verify the title (or `Dataset2d.name` when not given explicitly) is
  readable and the image renders with no points placed (`Origin`, `u`, `v` all
  read "not placed").
- Verify the widget remains usable (zoom/pan responsive) while the
  backend/kernel is idle after first paint.
- Repeat with the large stress image and confirm pan/zoom stays responsive
  at native resolution.

### CL-02: Pick The Origin And Lattice Vectors

**User story**: As a user measuring a crystal lattice, I want to click an
ordered origin, `a1`, and `a2` on the image and immediately see the derived
lattice vectors so I can carry them into downstream analysis.

**Primary widgets**: ChooseLattice.

**Data to use**: real lattice image with clearly resolved atomic columns.

**Acceptance checks**:

- Click three points in order and verify each appears immediately with the
  correct label/color (`Origin` first, then the `u`/`v` markers) and a guide
  line from the origin to each.
- Verify the on-screen readout and `widget.origin`, `widget.a1`, `widget.a2`,
  `widget.u`, `widget.v`, and `widget.points_array` update together and that
  `u = a1 - origin`, `v = a2 - origin` hold exactly.
- Drag an existing point to a nearby atomic column and verify the point,
  guide lines, readout, and derived vectors update live with no lag.
- Click a 4th time after 3 points are placed and verify it does nothing (at
  most 3 points).
- Press **Clear Points** (and call `clear_points()` from Python) and verify
  all points, guide lines, and readouts reset together, and the button
  disables itself when there are no points to clear.
- Call `set_points(...)` from Python with fewer than 3 points and verify the
  widget reflects the partial state (missing points read "not placed",
  derived vectors stay `None` until both endpoints exist).

### CL-03: Zoom And Pan Without Losing Original Pixel Coordinates

**User story**: As a user placing points precisely, I want to zoom into a
specific atomic column and click without the reported coordinates shifting,
so the lattice vectors stay correct regardless of how I framed the click.

**Primary widgets**: ChooseLattice.

**Data to use**: real lattice image where individual atomic columns are only
distinguishable when zoomed in.

**Acceptance checks**:

- Wheel-zoom in on a specific atomic column, verify the zoom is anchored
  under the cursor, then place a point and confirm the reported `(row, col)`
  matches the column's position in the ORIGINAL, un-zoomed image.
- Drag-pan the view, place a second point, and verify its reported
  coordinates are also in original-image pixel space.
- Double-click to reset zoom/pan and verify already-placed points render at
  the correct screen position after the reset.
- Verify the live cursor readout next to the hint text tracks the hovered
  original-image pixel while zoomed and panned.

### CL-04: Save And Reopen Picked Points

**User story**: As a notebook user, I want my picked lattice points to
survive a saved-notebook reopen when I asked for that, and I want a lightweight
static preview otherwise, so I don't have to re-click every time I revisit the
notebook.

**Primary widgets**: ChooseLattice.

**Data to use**: real lattice image in a Jupyter notebook.

**Acceptance checks**:

- With `save_state=True`, place 3 points, press `Cmd+S`, close and reopen the
  notebook, and verify the image and all 3 points restore without rerunning
  the cell.
- With `save_state=False` and the default `notebook_preview_format=None`,
  verify NO static-fallback sibling (`img.quantem-static-fallback`) is added
  — this widget intentionally opts out of the shared fallback by default
  because its live view does not reliably hide that sibling while
  interactive, unlike Show2D/Show3D. A cold reopen with no kernel is expected
  to show "Error displaying widget: model not found" in this default
  configuration.
- Pass `notebook_preview_format="jpeg"` explicitly and verify the fallback
  sibling now appears in the saved notebook, and check whether it stays
  visibly hidden behind the live widget or shows as a redundant duplicate
  image — if the hide behavior is still broken for this widget, that is a
  known gap to fix before recommending opt-in fallback previews to users.
- Confirm the default (`save_state=False`, `notebook_preview_format=None`)
  saved notebook stays small — no full interactive point/zoom state and no
  fallback image baked in.
