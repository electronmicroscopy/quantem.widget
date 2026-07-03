# Show3D Storyboard

Use with [Widget Storyboard](widget-storyboard).

## Stories

### S3D-01: Open A Time Series Quickly

**User story**: As a user opening a time series or focal stack, I want first
paint in about a second for normal working sizes, and still within seconds for
heavy stress data, so I can start scrubbing before the workflow feels blocked.

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
