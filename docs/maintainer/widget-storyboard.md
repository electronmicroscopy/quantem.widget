# Widget Storyboard

This document is the recurring AI/browser drive plan for QuantEM widgets. It is
written as scientific user stories, not as a fixed button script. The order of
stories can change for a release, bug, or dataset, but the agent report must say
which stories were driven, which were skipped, and why.

Use this for UI behavior, real-data workflows, browser testing, and release
signoff. Keep performance lessons, timing observations, and implementation
policy in [Performance](widget-performance).

## Story Format

Each story has four parts:

- **User story**: the scientific workflow and reason it matters.
- **Primary widgets**: widgets that must satisfy the story.
- **Data to use**: real or real-derived data preferred for signoff.
- **Acceptance checks**: concrete browser actions and expected outcomes.

Acceptance checks are executable, but they are subordinate to the story. Agents
should adapt the order and exact dataset to the change under test instead of
blindly clicking through a list.

## Agent Rules

- Drive the actual widget in the Codex in-app browser or Chrome; Python tests
  alone do not verify a story.
- Use an MJ-goat or buffle Jupyter backend when testing real data, large arrays,
  save/reopen, or backend streaming.
- Use real or real-derived microscopy data first. Synthetic data is a secondary
  control only.
- Test desktop and mobile-sized viewports. A narrow browser viewport is a
  pre-check; physical iPhone Safari is required for iPhone-specific claims.
- Record backend host, frontend browser, URL/notebook, widget source path,
  data path, shape, dtype, native bytes, panel count, frame count, display bin,
  first-paint time, and interaction FPS method.
- Start from a fresh render after code/build changes: rebuild, reload, rerun
  the notebook cell, or reopen exported HTML.
- Mark each story ``Pass``, ``Fail``, or ``Not verified``. Do not report
  "all good" from screenshots, DOM inspection, or unit tests alone.

## Show2D Stories

### S2D-01: Open A Large Real Image Quickly

**User story**: As a microscopist opening a large image, I want a useful preview
within seconds so I can decide whether the file is worth inspecting.

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

## Show3D Stories

### S3D-01: Open A Time Series Quickly

**User story**: As a user opening a time series or focal stack, I want first
paint in seconds so I can start scrubbing before the workflow feels blocked.

**Primary widgets**: Show3D.

**Data to use**: single-panel real 3D stack and a heavy real-derived multi-panel
movie such as 12 panels x 32 frames.

**Acceptance checks**:

- Measure first visible paint, payload size, display bin, and native shape.
- Verify first frame, current frame label, and histogram render correctly.
- Confirm display binning is explicit when native pixels are not available.

### S3D-02: Match Show2D Visual Language

**User story**: As a user comparing datasets side by side, I want multi-panel
Show3D to use the same visual language as Show2D: labels, scale bars, colormaps,
histogram controls, and compact status text.

**Primary widgets**: Show3D and Show2D reference gallery.

**Data to use**: current frame from the same real stack rendered through Show2D
for parity.

**Acceptance checks**:

- Compare labels, scale bars, color maps, panel borders, stats, and histogram
  controls against Show2D.
- Compare saved Show3D fallback pixels against a Show2D current-frame gallery
  for controlled parity tests.
- Verify one label per panel and no duplicated MP4/GIF labels.

### S3D-03: Arrange Movie Panels

**User story**: As a user arranging many movie panels, I want a column selector
so I can switch between one row, multiple rows, and dense galleries.

**Primary widgets**: Show3D.

**Data to use**: at least 8 movie panels; include a 12-panel heavy page for
release signoff.

**Acceptance checks**:

- Change columns through 1, 2, 3, 4, 6, 8, and 12.
- Verify the menu does not offer impractical counts above 12.
- Verify the current frame, zoom center, labels, scale bars, histogram, and
  contrast do not jump during reflow.

### S3D-04: Hide Movie Panels

**User story**: As a user screening many movie panels, I want to hide panels and
have playback, FFT, stats, export, and saved previews follow the visible set.

**Primary widgets**: Show3D.

**Data to use**: multi-panel real or real-derived movie.

**Acceptance checks**:

- Hide and restore panels while playback is stopped and while playback is
  active.
- Verify frame slider, FFT overlays, stats, exports, and saved previews use only
  visible panels.
- Verify hidden panels do not keep stale FFT/cache work alive.

### S3D-05: Play And Scrub Smoothly

**User story**: As a user playing or scrubbing a movie, I want the image, frame
label, histogram, and slider to stay synchronized at the selected FPS.

**Primary widgets**: Show3D.

**Data to use**: heavy real-derived multi-panel movie.

**Acceptance checks**:

- Press Play/Pause at 30 FPS and verify image and slider stay synchronized.
- Increase FPS and record whether the slider lags the image.
- Drag the frame slider slowly and quickly.
- Use keyboard frame stepping.
- Change averaging window during playback.
- Toggle Loop and Bounce and verify end-of-stack behavior.
- Verify no background flash, stale frame, or delayed label appears.

### S3D-06: Compare Dynamics Across Panels

**User story**: As a user comparing dynamics across panels, I want linked zoom
and linked contrast to be fast and reversible.

**Primary widgets**: Show3D.

**Data to use**: multi-panel stack with shared spatial features.

**Acceptance checks**:

- Toggle linked zoom and pan; verify panels move together.
- Toggle linked contrast; verify contrast changes apply consistently when
  linked and independently when unlinked.
- Change scale mode, colormap, Smooth, and histogram range while scrubbing.
- Resize the grid and verify current frame, labels, scale bars, and histogram UI
  do not jump unexpectedly.

### S3D-07: Inspect FFT In Flexible Layouts

**User story**: As a user inspecting reciprocal space, I want FFT layouts on
bottom, right, or overlay without changing real-space interaction semantics.

**Primary widgets**: Show3D.

**Data to use**: real stack with visible lattice peaks.

**Acceptance checks**:

- Toggle FFT bottom layout and verify spacing between Show3D and FFT, panel
  alignment, resize behavior, and histogram placement.
- Toggle FFT right layout and verify vertical height aligns with real-space
  panels and controls remain reachable.
- Toggle FFT overlay and verify every visible panel receives one overlay.

### S3D-08: Control FFT Overlays Independently

**User story**: As a user using FFT overlays, I want each overlay centered,
cached, independently zoomable, pannable, and draggable with corner snap.

**Primary widgets**: Show3D.

**Data to use**: heavy multi-panel movie with FFT overlay enabled.

**Acceptance checks**:

- Verify overlay starts centered on FFT center, not an edge or corner.
- Change overlay size and verify it resizes independently from the real-space
  panel grid.
- Drag overlay and release near each corner; verify snap-to-corner works.
- Wheel over overlay and verify FFT zooms, not the underlying real-space image.
- Use the documented pan gesture over the overlay and verify FFT panning works
  when zoomed.
- Verify cached/display-sized FFT does not recompute unnecessarily during
  playback, frame scrub, scroll, or resize.

### S3D-09: Trust FFT Peak Display

**User story**: As a user validating FFT peaks, I want suspicious FFT views
compared against a reference before trusting the display transform.

**Primary widgets**: Show3D.

**Data to use**: real panel/frame where peaks look broad, missing, or too dark.

**Acceptance checks**:

- Compare FFT magnitude and peak locations against NumPy or another trusted
  reference.
- Distinguish magnitude correctness from display transform/contrast problems.
- Verify FFT remains readable on black/dark backgrounds.

### S3D-10: Export Animations

**User story**: As a user making animations, I want GIF and MP4 exports to show
only image panels with clean borders, one label per panel, and predictable file
sizes.

**Primary widgets**: Show3D.

**Data to use**: single-panel, 3-panel, and 2x2 real time-series examples.

**Acceptance checks**:

- Export GIF and MP4 panel-only animations.
- Verify expected frame count, labels, border/background, playback speed, and
  file size.
- Verify quality/speed options are visible and have clear labels.

### S3D-11: Export Shareable HTML

**User story**: As a user sharing HTML, I want exact/quantized export labels and
sizes to match the Show2D export vocabulary.

**Primary widgets**: Show3D.

**Data to use**: single-panel and multi-panel real stacks.

**Acceptance checks**:

- Open Export and verify HTML exact/quantized/GIF/MP4 labels and approximate
  sizes.
- Verify cancellation/status text clears after the documented timeout.
- Export HTML exact and quantized where supported.
- Open exported files and drive playback, frame slider, columns, hide panels,
  FFT overlay, histogram, and reset.

### S3D-12: Save And Reopen A Notebook

**User story**: As a notebook user, I want ``Cmd+S`` and reopen to show a
compact Show3D fallback that is pixel-matched to a Show2D current-frame gallery.

**Primary widgets**: Show3D and Show2D reference gallery.

**Data to use**: Jupyter notebook with real or real-derived stack.

**Acceptance checks**:

- Press ``Cmd+S`` in JupyterLab and reload/reopen the notebook.
- Verify saved Show3D static output is visible.
- Compare fallback against Show2D current-frame gallery in controlled tests.
- Check ``metadata.widgets`` or ``get_state()`` for heavy-buffer leaks:
  ``frame_bytes``, ``_buffer_bytes``, offline stacks, and export payloads must
  not be present when ``save_state=False``.

### S3D-13: Use The Widget On A Phone Or Narrow View

**User story**: As a user checking Show3D on a phone or narrow screen, I want
playback controls, frame slider, FFT controls, and panel menus to remain
reachable.

**Primary widgets**: Show3D.

**Data to use**: single-panel and multi-panel stack.

**Acceptance checks**:

- Test a narrow mobile viewport.
- Verify controls wrap, playback controls stay reachable, labels fit, and the
  frame slider remains usable.
- Test touch-style drag and scroll gestures.
- For iPhone-specific claims, serve the page to a physical iPhone Safari test.

## Show3DSlices And Show4DSTEM Stories

### S3S-01: Inspect Orthogonal Slices

**User story**: As a user checking volume/slice data on a laptop or phone, I
want Show3DSlices to expose orthogonal views with responsive controls and clear
position labels.

**Primary widgets**: Show3DSlices.

**Data to use**: real or real-derived 3D volume.

**Acceptance checks**:

- Scrub each slice axis and verify labels, crosshair/position indicators, and
  image updates remain synchronized.
- Test narrow viewport and touch-style slider/drag behavior.
- Save/reopen notebook and verify visible compact output.

### S4D-01: Inspect 4D-STEM Virtual Image And Diffraction Together

**User story**: As a 4D-STEM user, I want a compact two-panel experience with
virtual image and diffraction view so I can relate scan position to diffraction
content.

**Primary widgets**: Show4DSTEM.

**Data to use**: real or tutorial 4D-STEM stack; include WebGPU-capable browser
when possible.

**Acceptance checks**:

- Move scan position and verify diffraction panel updates.
- Change virtual detector settings and verify virtual image updates.
- Verify WebGPU path when available and fallback path when not.
- Save/reopen notebook and verify the static two-panel fallback is visible.

## Release Report Template

Use this template in agent signoff reports:

```text
Verified:
- Stories driven:
- URL/notebook:
- Backend host/source path:
- Frontend browser:
- Data source and shape:
- First-paint time:
- Interaction FPS method/result:
- Save/reopen result:
- Exports opened:
- Tests run:

Not verified:
- Story IDs:
- Reason:

Remaining risk:
- Hardware/browser/data sizes not covered:
```

## Release-Gating Rule

- If any P0 story fails, do not tag an RC: first paint over roughly 10 s, blank
  saved output, heavy-buffer save leak, broken export menu, playback/slider
  desync, FFT correctness failure, or interaction far below the target FPS.
- If a P1 story is not verified, the RC report must say exactly why and who will
  verify it next. P1 examples: physical iPhone checks, maximum-size datasets,
  or hardware-specific WebGPU adapter coverage.
- The storyboard report must be linked from the release candidate signoff.
