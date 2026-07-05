# Show2D Storyboard

Use with [Storyboard](storyboard).

## Stories

### S2D-01: Open A Large Real Image Quickly

**User story**: As a microscopist opening an image, I want a useful preview in
about a second for normal working sizes, and still within seconds for heavy
stress data, so I can decide whether the file is worth inspecting.

**Primary widgets**: Show2D.

**Data to use**: one real 4k or larger image; repeat with an 8+ panel real or
real-derived gallery for heavy signoff.

**Acceptance checks**:

- Load the image from a real file path on the backend, from an in-memory array,
  and from exported HTML when supported.
- Measure first visible paint and note display bin/native bytes.
- Verify the title/info badge communicates preview/detail state when binning is
  active.
- Verify the widget remains usable while the backend/kernel is idle after first
  paint.
- Verify the notebook does not save the full 4k array unless the user explicitly
  chooses an export path that embeds data.

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

### S2D-12: Review A High-Throughput Denoising Batch

**User story**: As a user reviewing denoising or drift-correction results, I
want to open dozens of 4k images as a gallery, arrange them quickly, hide weak
outputs, and keep interaction fast enough to screen the batch without exporting
manual contact sheets.

**Primary widgets**: Show2D.

**Data to use**: real or real-derived 4k x 4k files from a denoising, drift, or
ptychography workflow. Test at least 30 panels for routine signoff; use 45 and
85 panels when backend storage and memory allow.

**Acceptance checks**:

- Load the gallery from file paths on the backend without copying files to the
  laptop.
- Record file count, native shape, dtype, total native bytes, first-paint time,
  display bin, and browser memory if available.
- Change columns through 2, 4, 6, 8, and 12; verify panel labels, scale bars,
  stats, histograms, and hover readouts remain aligned.
- Hide poor panels, restore them, and verify selection order and export state
  stay correct.
- Pan, zoom, histogram-drag, and resize repeatedly; record the interaction FPS
  method and result.
- Verify zooming into one panel streams or displays the highest-resolution
  available tile for that panel, while the rest of the gallery remains
  responsive.

### S2D-13: Keep Loading And Storage Lightweight

**User story**: As a notebook user working with large files, I want loading to
show a useful view quickly and saving to keep the notebook small, so I can come
back later without embedding gigabytes of image data.

**Primary widgets**: Show2D.

**Data to use**: one 4k or larger image and one 30+ panel 4k gallery from real
backend files.

**Acceptance checks**:

- Compare live Jupyter loading, saved-notebook reopen, and standalone HTML
  export paths.
- Verify live Jupyter uses backend file/array access for detail streaming rather
  than serializing every native pixel into widget state.
- Press ``Cmd+S``, reload the notebook, and verify the saved output is visible,
  compact, and labeled as preview/detail/offline as appropriate.
- Open Export and verify exact float32 and quantized uint8 HTML labels follow
  the Show3D wording and show approximate file sizes when known.
- Confirm saved notebook state and exported HTML payload sizes are recorded in
  the signoff report.

### S2D-14: Stress Interactive Controls On Many 4k Panels

**User story**: As a scientist screening high-throughput image results, I want
all high-frequency controls to remain smooth even when many large panels are on
screen, because slow hover, histogram, or zoom feedback makes the viewer
unusable for triage.

**Primary widgets**: Show2D.

**Data to use**: 30, 45, and 85 real or real-derived 4k x 4k panels when
available; otherwise record the largest real batch tested and why the larger
case was skipped.

**Acceptance checks**:

- Measure first paint, column reflow, histogram drag, mousewheel zoom, pan,
  hover readout, FFT toggle, and reset on the heavy gallery.
- Verify target interaction remains near 30 FPS for the controls under test, or
  record the limiting hardware/browser/data condition.
- Confirm stale preview/detail tiles are not drawn after rapid zoom, pan,
  resize, or contrast changes.
- Verify controls remain keyboard and pointer reachable when the gallery is
  taller than the viewport.
- Add failures or near misses to the performance log with the data path and
  exact shape so the case can be replayed.

### S2D-15: Inspect Images Full Screen On A Large Monitor

**User story**: As a microscopist using a workstation backend and a laptop or
desktop browser as the frontend, I want Show2D to use the available screen
cleanly so I can inspect the scientific image without fighting notebook chrome,
oversized controls, or wasted whitespace.

**Primary widgets**: Show2D.

**Data to use**: one real 4k or larger image, plus a 4+ panel real or
real-derived gallery.

**Acceptance checks**:

- Launch from a remote Jupyter backend path when possible: the HPC/workstation
  backend owns the data and Python kernel; the browser drives the widget from
  the local laptop.
- Open the notebook or exported HTML in a wide browser viewport and use browser
  full-screen mode.
- Verify the scientific image or gallery grows with the viewport while controls
  remain compact, content-sized, and aligned to the same design language as
  Show3D and Show4DSTEM.
- Verify top-right actions such as Export, Reset, Copy, and panel controls sit
  on the right edge of the widget header when there is available width.
- Drive zoom, pan, histogram center drag, FFT, profile, ROI, and panel reflow in
  the large view; record whether any interaction loses visible FPS compared with
  the notebook-sized view.
- Return to a normal notebook viewport and verify the layout contracts without
  clipped controls, wrapped labels, or stale full-screen sizing.
