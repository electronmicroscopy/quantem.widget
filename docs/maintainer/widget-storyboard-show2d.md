# Show2D Storyboard

Use with [Widget Storyboard](widget-storyboard).

## Stories

### S2D-01: Open A Large Real Image Quickly

**User story**: As a microscopist opening an image, I want a useful preview in
about a second for normal working sizes, and still within seconds for heavy
stress data, so I can decide whether the file is worth inspecting.

**Primary widgets**: Show2D.

**Data to use**: one real 4k or larger image; repeat with an 8+ panel real or
real-derived gallery for heavy signoff.

**Acceptance checks**:

- Load the image from Jupyter and from exported HTML when supported.
- Measure first visible paint and note display bin/native bytes.
- Verify the title/info badge communicates preview/detail state when binning is
  active.
- Verify the widget remains usable while the backend/kernel is idle after first
  paint.

### S2D-02: Arrange Panels For Comparison

**User story**: As a user comparing several real-space images, I want to choose
the number of columns so I can fit panels to my monitor, notebook, or paper
figure layout.

**Primary widgets**: Show2D.

**Data to use**: at least 8 real or real-derived panels.

**Acceptance checks**:

- Change columns through 1, 2, 3, 4, 6, 8, and 12.
- Verify the menu does not offer impractical counts above 12.
- Verify labels, scale bars, stats, histograms, and borders remain aligned.
- Verify the current zoom center and contrast do not jump during reflow.

### S2D-03: Hide Unimportant Panels

**User story**: As a user screening many images, I want to hide unimportant
panels while preserving the scientific state of the remaining panels.

**Primary widgets**: Show2D.

**Data to use**: a multi-panel gallery with labels and visible scale bars.

**Acceptance checks**:

- Hide one panel, multiple panels, and all-but-one panel.
- Verify layout, labels, stats, histograms, keyboard selection, export, and
  saved state ignore hidden panels.
- Restore panels and verify original order and panel state return.

### S2D-04: Inspect Native Pixels From A Fast Preview

**User story**: As a user inspecting atomic or lattice detail, I want a fast
binned preview to stream native-resolution tiles when I zoom in.

**Primary widgets**: Show2D.

**Data to use**: a 4k or larger real image with recognizable high-frequency
structure.

**Acceptance checks**:

- Zoom past preview resolution and verify a detail request is issued.
- Verify the returned tile uses native row/column coordinates and a tile
  ``bin`` smaller than ``_display_bin_factor`` when possible.
- Pan across detail boundaries and verify stale detail tiles are never drawn
  after the view changes.
- Verify cursor readout reports native ``(row, col)`` and labels value source
  as preview, detail, or native.

### S2D-05: Adjust Contrast On Noisy Data

**User story**: As a user adjusting contrast on noisy microscopy data, I want
histogram interactions to be smooth and visually correct.

**Primary widgets**: Show2D.

**Data to use**: noisy real data where stale tiles or contrast flashes are easy
to see.

**Acceptance checks**:

- Drag histogram min/max handles quickly.
- Drag histogram center/range repeatedly while hover readout is visible.
- Toggle auto contrast off/on and verify manual range is preserved until the
  user asks for auto again.
- Verify no stale square tile, ghost rectangle, white flash, or delayed color
  update remains.
- Record FPS for histogram drag and slider movement.

### S2D-06: Link And Unlink Comparison State

**User story**: As a user comparing related panels, I want linked zoom, pan, and
contrast to be optional and reversible.

**Primary widgets**: Show2D.

**Data to use**: multi-panel real data with shared features.

**Acceptance checks**:

- Enable linked zoom, pan, and contrast; verify panels move together.
- Disable each link mode; verify independent panel state works.
- Resize the grid and verify view anchors do not jump.

### S2D-07: Use FFT To Inspect Periodicity

**User story**: As a user looking for periodicity, I want FFT views for every
visible panel and I want them to remain fast during layout changes.

**Primary widgets**: Show2D.

**Data to use**: real data with lattice peaks; include one suspicious panel for
reference comparison.

**Acceptance checks**:

- Toggle FFT and verify every visible panel gets the expected FFT view.
- Change columns and resize panels; verify FFT alignment and spacing remain
  correct.
- Zoom and pan in FFT mode; verify events target FFT, not stale real-space
  layers.
- Compare one suspicious FFT against NumPy or a known reference.

### S2D-08: Measure Features With Overlays

**User story**: As a user measuring image features, I want profile and ROI
overlays to remain stable while I draw and drag them.

**Primary widgets**: Show2D.

**Data to use**: real image or gallery where line profiles and ROIs are useful.

**Acceptance checks**:

- Toggle Profile; draw, move, and delete a line profile.
- Toggle ROI tools; draw, drag, resize, save, restore, and delete ROIs.
- Verify high-frequency pointer labels do not pop or lag.
- Use keyboard navigation for previous/next panel, reset zoom, and delete ROI.

### S2D-09: Export And Share A Static Result

**User story**: As a user preparing a shareable result, I want export choices to
say exactly what they will save and produce files that reopen correctly.

**Primary widgets**: Show2D.

**Data to use**: single image and multi-panel gallery.

**Acceptance checks**:

- Open Export in live Jupyter and standalone HTML.
- Verify labels say ``HTML exact float32`` and/or ``HTML quantized uint8`` and
  show approximate sizes when known.
- Export exact and quantized HTML where supported.
- Open both files and drive columns, hide panels, FFT, histogram, zoom, and
  reset.
- Use Copy and verify output corresponds to the current visible state.

### S2D-10: Save And Reopen A Notebook

**User story**: As a notebook user, I want ``Cmd+S`` and reopen to preserve a
visible compact output without embedding huge pixel buffers.

**Primary widgets**: Show2D.

**Data to use**: a live Jupyter notebook with real or real-derived data.

**Acceptance checks**:

- Press ``Cmd+S`` in JupyterLab and reload/reopen the notebook.
- Verify the saved static output is visible and compact.
- Check ``metadata.widgets`` or ``get_state()`` for heavy-buffer leaks:
  ``frame_bytes``, ``_detail_bytes``, offline stacks, and export payloads must
  not be present when ``save_state=False``.

### S2D-11: Use The Widget On A Phone Or Narrow View

**User story**: As a user checking results on a phone or narrow screen, I want
controls to wrap and remain usable without covering the scientific image.

**Primary widgets**: Show2D.

**Data to use**: single image and multi-panel gallery.

**Acceptance checks**:

- Test a narrow mobile viewport.
- Verify controls wrap, labels do not overlap, panels remain usable, and any
  horizontal scrolling is intentional.
- Test touch-style zoom, pan, menu open, column selection, and panel visibility.
- For iPhone-specific claims, serve the page to a physical iPhone Safari test.
