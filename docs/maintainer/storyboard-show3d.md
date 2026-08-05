# Show3D Storyboard

Use with [Storyboard](storyboard).

## Stories

### S3D-01: Open A Time Series Quickly

**User story**: As a user opening a time series or focal stack, I want first
paint in about a second for normal working sizes, and still within seconds for
heavy stress data, so I can start scrubbing before the workflow feels blocked.

**Primary widgets**: Show3D.

**Data to use**: single-panel real 3D stack and a heavy real-derived multi-panel
movie such as 12 panels x 32 frames.

**Acceptance checks**:

- Load the stack from real backend files or arrays without requiring the laptop
  to receive every native pixel before first paint.
- Measure first visible paint, payload size, display bin, and native shape.
- Verify first frame, current frame label, and histogram render correctly.
- Confirm display binning is explicit when native pixels are not available.
- Verify frame labels and per-panel metadata are available before playback, even
  when higher-resolution detail arrives later.

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
- Hover unselected movie panels and verify the cursor readout, frame-aware
  stats, and Color dropdown/reporting context follow the hovered panel without
  changing which panel the edit controls will modify.
- Pass ``cmap=["RdBu", "viridis", ...]`` for a multi-panel Show3D comparison
  and verify every panel keeps its own colormap in the Python API, live UI,
  saved state, static notebook fallback, and standalone HTML export. Hover or
  select each panel and confirm the Color dropdown reports that panel's map
  without changing neighboring panels.
- [x] **S3D-CMAP-1**: On a real MoS2 two-panel stack, verify the SSB phase
  panel renders with ``RdBu`` while the DPC phase panel renders with
  ``viridis``; scrub to another frame and confirm the frame label/canvas update
  while the panel-specific Color dropdown follows the hovered panel.
- Compare saved Show3D fallback pixels against a Show2D current-frame gallery
  for controlled parity tests.
- Verify one label per panel and no duplicated MP4/GIF labels.

### S3D-03: Arrange Movie Panels

**User story**: As a user arranging many movie panels, I want a column selector
and panel reorder control so I can switch between one row, multiple rows, dense
galleries, and the comparison order I want to share.

**Primary widgets**: Show3D.

**Data to use**: at least 8 movie panels; include a 12-panel heavy page for
release signoff.

**Acceptance checks**:

- Change columns through 1, 2, 3, 4, 6, 8, and 12.
- Verify the menu does not offer impractical counts above 12.
- Toggle `Reorder`, drag panels into a new order, reset the order, and confirm
  `panel_order`, `visible_panels`, Show2D handoff, saved state, and HTML export
  follow the same order.
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
- [x] **S3D-PLAY-4**: In an exported/offline multi-panel Show3D, zoom or pan a
  packed panel, press Play, then keep dragging while playback is active. Verify
  the current frame repaints in the zoomed viewport without touching the
  histogram or any other control.
- [x] **S3-DN-3**: Start with browser-side Gaussian denoise enabled, scrub from
  the initial frame to a later frame, and verify both canvases remain visibly
  smooth rather than briefly or permanently showing raw pixels.
- Use keyboard frame stepping.
- Change averaging window during playback.
- Toggle Loop and Bounce and verify end-of-stack behavior.
- Verify no background flash, stale frame, or delayed label appears.

### S3D-05A: Keep Multi-Panel Display State Independent

**User story**: As a scientist comparing several reconstructions over many
pages, I want each panel to retain its own contrast, colormap, zoom, pan, and
border state unless I explicitly link it, so I can inspect one result without
silently changing another or seeing an empty canvas between pages.

**Primary widgets**: Show3D.

**Data to use**: a real or real-derived multi-panel stack with at least three
pages and visibly different intensity ranges between panels.

**Acceptance checks**:

- With Link Contrast off, move one panel's histogram handle. Only that panel's
  clip range and pixels may change; neighboring panels keep their own histogram,
  color, and pixels.
- Change page by slider, keyboard, and Page Play while playback is running.
  The prior complete scientific pixels must remain until the next direct paint
  is ready; no black, white, or empty panel is acceptable during the change.
- Zoom, pan, resize, and hover any unselected panel. None of these actions may
  alter another panel's contrast, colormap, Smooth state, selected panel, or
  border.
- Verify title, scale bar, panel border, histogram, and frame label remain
  synchronized across the same page/frame transition.
- The canvas has one authoritative direct rendering path from the current
  frame and that panel's current state. Do not add a prebuilt
  `HTMLCanvasElement`/composite display cache or a separate display-cache state
  machine; numerical caches such as decoded source frames and FFT magnitudes are
  allowed only when they cannot own zoom, contrast, page, or panel state.
- Run the scenario in a live Jupyter widget and a freshly exported HTML file.
  Capture one transition-time screenshot or pixel sample, not only the final
  settled view.

### S3D-05B: Explore Time-Series Dynamics With Playback Presets

**User story**: As a scientist inspecting time-series Show3D data, I want
playback dynamics that act like temporal lenses, so I can slow down subtle
changes, bounce through reversible motion, focus on an event range, or hold key
frames without rewriting the notebook.

**Primary widgets**: Show3D.

**Data to use**: a real or real-derived time series with a visible event,
motion, relaxation, or reconstruction change. Include one exported HTML report
because collaborators often review time series outside Jupyter.

**Acceptance checks**:

- Keep the main toolbar simple. Put playback style in the playback row next to
  the frame star, not in the top toolbar More menu.
- Linear style: play through the current loop range at constant `fps`; verify
  frame label, slider, histogram, FFT, and scale bar stay synchronized.
- Power In, Power Out, and Ease In/Out styles: generate `playback_path` from
  the current loop range; verify users still control `fps`, `loop`,
  `boomerang`, and range from the existing playback row controls.
- Configure denoise plus a low-pass, high-pass, or band-pass FFT filter before
  pressing Play; verify transitions stay visibly filtered and smooth, the
  slider remains honest, and the viewer does not flash raw/noisy frames while
  the current filtered display settles.
- Run at least one organic notebook trial that starts from a plain
  `Show3D(stack)` or similarly minimal call. A reviewer should turn Denoise,
  Filter, FFT, Loop/Bounce, playback style, and speed/range controls on from
  the widget UI itself, not by preloading every behavior through kwargs.
- While that organic trial is playing, behave like a real reviewer: scrub the
  frame slider, pause/resume, drag denoise sigma, drag filter cutoff or band
  controls, accidentally or intentionally change a nearby display control such
  as colormap/Smooth, then keep playing. Verify the canvas remains filtered,
  the frame counter stays honest, and the image does not twitch from competing
  repaint paths.
- Use kwargs-heavy repro notebooks only as focused bug reproducers. They do not
  replace the organic UI-drive trial because they skip the user behavior that
  often exposes menu, focus, cache, and repaint races.
- Save state and export HTML; reopen and verify `fps`, `loop`, `boomerang`,
  range, and `playback_path` survive, but playback does not unexpectedly start
  unless explicitly requested.
- Drive the same presets with FFT visible, denoise/filter enabled, and hidden
  panels in a multi-panel movie; verify cache counters do not grow while idle
  and hidden panels do not keep doing work.

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
- While one panel is selected for controls, hover neighboring panels during
  scrub/playback and verify readout/stats follow the hovered panel but linked
  or independent edits still apply only according to the selected/link state.
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
- Press Play with FFT visible and verify the overlay follows the current frame
  at a bounded cadence rather than freezing on the pre-playback frame.
- In every layout, verify each FFT tile or inset shows the current shared
  `N.N×` multiplier, including uncalibrated data and narrow/mobile layouts;
  `show_zoom_indicator=False` must hide the FFT badges.
- Pause on a known frame, switch browser tabs, and return. Bottom, right, and
  overlay FFT layouts must restore their image pixels and multiplier without
  changing the frame or playback state and without another FFT computation.

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
- Verify the inset multiplier updates with FFT zoom and reset restores `1.0×`
  without changing the real-space multiplier.
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
- Verify expected frame count, multi-panel layout, live-style labels, scale bar,
  zoom readout, border/background, playback speed, and file size.
- Include at least one uint16 source-stack case. The decoded GIF/MP4 frames
  should match the widget-rendered frame size, panel count, black inter-panel
  gap, inner/outer borders, panel labels, scale bars, and playback fps.
- Verify quality/speed options are visible and have clear labels.
- For standalone HTML with encoded uint8 data, verify GIF/MP4 remain available
  but show a warning before export that the movie uses encoded display data and
  that exact/live export is required for publication-quality fidelity.
- Use labels and title spans containing symbols and inline math such as
  `\lambda=0.03 raw` and `$\\chi^2$/pixel`; verify the panel menu, stats rows,
  and animation labels render symbols/math and never show raw markup.
- For PowerPoint or Slack sharing, attach the exported GIF/MP4 and verify the
  receiving app recognizes the media. For PowerPoint, inspect the PPTX package
  for embedded `ppt/media/*` assets rather than linked local files, open the
  deck in native PowerPoint when Mac automation is available, and record whether
  playback was actually verified or blocked by OS permissions.
- Run `PYTHONPATH=src:. python scripts/widget_show3d_animation_smoke.py` when
  judging whether a GIF is good enough for PowerPoint/email sharing.

### S3D-11: Export Shareable HTML

**User story**: As a user sharing HTML, I want exact/quantized export labels and
sizes to match the Show2D export vocabulary.

**Primary widgets**: Show3D.

**Data to use**: single-panel and multi-panel real stacks.

**Acceptance checks**:

- Open Export and verify HTML exact/quantized/GIF/MP4 labels and approximate
  sizes.
- Collapse controls for presentation-style viewing and verify Export remains
  reachable from the title chrome in live and standalone HTML.
- In standalone HTML, verify GIF works in-browser. Verify MP4 is enabled when
  WebCodecs H.264 is supported, and otherwise remains visible but disabled with
  a browser-support or live-backend explanation.
- Verify cancellation/status text clears after the documented timeout.
- Export HTML exact and quantized where supported.
- [x] **S3-EX-2**: Open an exact standalone export with display-only denoise
  enabled. Before any interaction, verify the canvas is denoised; then scrub to
  another frame and verify the denoise remains applied.
- Open exported files and drive playback, frame slider, columns, hide panels,
  FFT overlay, histogram, and reset.
- In each standalone file, pause with FFT visible, switch away and back, and
  verify the main image plus bottom/right/overlay FFT canvases restore without
  pressing Play or moving the frame slider.

### S3D-12: Save And Reopen A Notebook

**User story**: As a notebook user, I want ``Cmd+S`` and reopen to show a
compact Show3D fallback that is pixel-matched to a Show2D current-frame gallery.

**Primary widgets**: Show3D and Show2D reference gallery.

**Data to use**: Jupyter notebook with real or real-derived stack.

**Acceptance checks**:

- Press ``Cmd+S`` in JupyterLab and reload/reopen the notebook.
- Verify saved Show3D static output is visible.
- Compare fallback against Show2D current-frame gallery in controlled tests.
- For a single-panel stack, set ``notebook_preview_frames=[0, mid, last]`` and
  ``notebook_preview_ncols=3``, save, and reopen without rerunning the cell.
  The saved PNG fallback must be a compact contact sheet of exactly those
  frame indices with the same frame labels, scale bar, colormap, and contrast
  language as the live current-frame preview.
- Repeat the save with no ``notebook_preview_frames`` and verify the default
  remains the current frame only.
- In multi-panel Show3D, set ``notebook_preview_frames`` and verify the saved
  preview intentionally stays one current frame per visible panel rather than
  multiplying panels by saved-frame count.
- Draw one circular ROI and then multiple visible ROIs on a single-panel stack,
  scrub to a representative frame, save, and reopen the notebook without
  rerunning the cell. The saved PNG fallback must show the current frame with
  all ROI overlays plus one ROI zoom/crop panel per visible ROI.
- After drawing circle, rectangle, square, and annular ROIs, call
  ``get_roi_geometries()`` and verify the returned ``(row, col)`` center,
  radius, corners, bounds, visibility, and colors match the visible overlays and
  survive ``state_dict()`` / ``load_state_dict()`` while the user scrubs frames.
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

### S3D-14: Review Many Denoising Or Time-Series Results

**User story**: As a user reviewing denoising, drift, focal-series, or
time-series outputs, I want many related stacks to open as movie panels quickly,
so I can compare temporal behavior without waiting for every native frame to
serialize into the browser.

**Primary widgets**: Show3D.

**Data to use**: real or real-derived stacks built from 4k x 4k source files.
Use at least 12 panels x 32 frames for routine signoff; add 30, 45, or 85
panel/file workflows when the scientific workflow produces that many outputs.

**Acceptance checks**:

- Load the stacks from backend file paths or prepared arrays and record backend
  host, file count, panel count, frame count, native shape, dtype, native bytes,
  display bin, first-paint time, and initial payload size.
- Verify a useful binned first frame appears before full-resolution detail or
  FFT work finishes.
- Scrub immediately after first paint and verify the image, slider, labels, and
  histogram stay synchronized.
- Change columns through 2, 3, 4, 6, 8, and 12 while preserving frame index,
  zoom anchor, labels, scale bars, and contrast state.
- Hide and restore panels during playback and verify hidden panels do not keep
  unnecessary frame or FFT work active.
- Sweep hover across visible panels while playback is paused and while it is
  playing. Verify the active readout/stats are per-hovered-panel and that hover
  alone does not change the selected panel, hidden-panel state, or playback
  target.
- Record whether playback, frame slider, histogram, and overlay interactions
  remain near the target FPS, or document the limiting case.

### S3D-15: Keep Movie Loading And Storage Lightweight

**User story**: As a notebook user working with large stacks, I want Show3D to
load a compact preview quickly, stream or reveal higher-resolution detail when I
ask for it, and save/reopen without embedding huge frame buffers.

**Primary widgets**: Show3D.

**Data to use**: single-panel 4k-derived stack and multi-panel real-derived
movie such as 12 panels x 32 frames or larger.

**Acceptance checks**:

- Compare live Jupyter loading, saved-notebook reopen, standalone HTML export,
  GIF export, and MP4 export for the same stack.
- Verify live Jupyter first paint is not blocked on all native frames, FFT
  overlays, or animation encoders.
- Press ``Cmd+S``, reload the notebook, and verify the saved fallback is visible
  and compact.
- Inspect widget state for heavy-buffer leaks when ``save_state=False``:
  full frame stacks, FFT caches, detail buffers, and export payloads should not
  be persisted.
- Open Export and verify exact float32, quantized uint8, GIF, and MP4 choices
  use the documented Show3D vocabulary and show approximate file sizes when
  known.

### S3D-16: Stress Playback, FFT, And Sliders On Heavy Movies

**User story**: As a scientist inspecting heavy movie data, I want playback,
scroll zoom, FFT overlay, histogram, and sliders to stay smooth because a slow
movie viewer hides dynamic behavior and wastes analysis time.

**Primary widgets**: Show3D.

**Data to use**: 12+ panel real-derived 2k or 4k-source movies, including at
least 32 frames. Use larger 30/45/85 file or panel workflows when available and
record skipped maximum cases explicitly.

**Acceptance checks**:

- Measure first paint, frame scrub, play at selected FPS, high-FPS playback,
  histogram drag, mousewheel zoom, pan, column reflow, FFT toggle, FFT overlay
  drag/snap, FFT overlay zoom/pan, and reset.
- Verify image, frame slider, frame labels, histogram, stats, and playback
  controls remain synchronized under stress.
- Verify FFT overlays use cached display-sized FFTs during playback and do not
  recompute full-resolution transforms unless the user explicitly requests a
  higher-fidelity FFT path.
- Verify overlay FFT starts centered, remains readable on dark backgrounds, and
  can be moved away from scale bars or important image features.
- Confirm no stale frame, stale FFT, white/yellow flash, blank overlay, or
  delayed slider state remains after rapid interaction.
- Repeat the paused tab-away/tab-return path after FFT zoom and pan. Require
  cached display repaint counters to increase while FFT miss, compute, and
  metric-compute counters remain unchanged.
- Add timing and failure notes to the performance log with exact data path,
  backend host, browser, adapter, shape, frame count, and panel count.

### S3D-17: Watch A Live EMD Frame Series In Place

**User story**: As a microscopist collecting an in-situ, focal, tilt, or
reconstruction series, I want one already displayed Show3D stack to append each
completed EMD as a frame, without rerunning the cell, creating another widget,
or interrupting inspection and playback.

**Primary widgets**: ``Show3D.from_folder(...)`` in live JupyterLab. Every
matching file is one frame in a single stack; it is not a new Show3D widget or
an additional gallery panel.

**Data to use**: Genuine same-shape Velox image EMD files from one real series,
added through atomic rename. Use at least three files for routine verification
and ten for release signoff. Include an incomplete EMD, an incompatible-shape
EMD, and a later compatible EMD. Record source path, shape, dtype, sampling,
native bytes, backend host, browser, widget commit, and watch interval. Keep
source data and generated reports outside git.

**Acceptance checks**:

- Keep folder arrivals on one frame axis regardless of file count. Crossing
  Show2D's default 20-panel threshold must leave ``n_panels == 1``,
  ``n_pages == 1``, and the page controls absent; the frame slider and playback
  are the navigation. Reject ``page_size=`` and ``page_labels=`` with a
  corrective suggestion to use ``Show2D.from_folder(...)`` when files should
  become independent gallery panels. Preserve explicit 5-D/list-of-page
  Show3D comparisons as a separate constructor workflow.
- In live JupyterLab, display exactly one ``Show3D.from_folder(folder,
  pattern="*.emd", watch=True, watch_interval=1.0, debug=True)`` object and
  capture its Python identity, widget model ID, and browser container before
  later files arrive.
- Start once with a valid EMD and separately before the first acquisition. The
  empty-folder case must remain visibly mounted and accept the first stable EMD
  without a cell rerun; mark it ``Fail`` until that behavior exists.
- Keep one compact accessible watch badge near the folder/title area in stable
  DOM. Require green-dot ``Watching`` only while the actual background worker
  is alive. Enter ``Updating`` while a discovery poll is active and keep it
  through real candidate validation, append, and authoritative frame/canvas
  paint. An idle poll may briefly show ``Updating`` but must return to
  ``Watching`` without decode, transfer, or repaint. Use amber ``Waiting for
  file completion`` for an incomplete EMD, red ``Watch error`` plus corrective
  detail for a bad shape or
  worker failure, and gray ``Stopped`` or ``Not watching`` after stop/close or
  when liveness cannot be established. A ``watch=False`` snapshot has no badge.
  A restored notebook model, static fallback, or standalone snapshot must never
  restore a green ``Watching`` state without a live worker. Capture browser
  assertions and screenshots for every state; color alone is insufficient.
- Complete a genuine EMD after the widget is visible. Verify the frame count,
  label, slider range, and canvas update automatically without calling
  ``poll_folder()``. Compare source shape, calibration, representative pixels,
  and the new frame canvas checksum against ``read_image(path).array``.
- Preserve the Python object, widget model, browser container, selected source
  path, bookmarks, starred frame, loop bounds, playback state, FFT, ROI,
  profile, zoom, pan, contrast, and calibration when natural ordering inserts a
  new filename before existing frames.
- Append while Play is active. Verify playback remains active and the image,
  slider, label, histogram, and frame cache remain synchronized without a blank
  or stale frame.
- Defer and retry an incomplete EMD without stopping the watcher or duplicating
  the frame. Report an incompatible shape without resizing it or blocking a
  subsequent compatible frame.
- Measure stable-file-to-Python-append and stable-file-to-first-canvas-paint
  separately for every arrival; report median and p95 with the poll interval.
  Backend state alone is not visible-user proof.
- Verify idle polls do no decode, transfer, repaint, or state-reset work.
  Exercise idempotent stop/resume, then call ``close()`` or ``free()`` and prove
  no watcher thread can mutate the stack afterward.

### S3D-18: Browse A True-Color Figure Gallery And Pick The Best

**User story**: As a user comparing candidate result figures (a lambda sweep,
competing reconstructions, before/after variants, RGB composites), I want to
load them as one labeled stack, scrub or keyboard through them in true color,
and identify the best one without exporting a manual contact sheet.

**Primary widgets**: Show3D.

**Data to use**: a real set of 3 or more same-shape figures, at least one in
true color `(H, W, 3)`. Build with `Show3D.from_figure_gallery({label: image})`,
`Show3D.from_rgb(stack)`, or a plain `(N, H, W, 3)` array.

**Acceptance checks**:

- Construct via `from_figure_gallery` with a `{label: image}` mapping and with a
  sequence plus `labels=`; verify each figure's name appears in the frame label
  and `n_slices` equals the figure count.
- Mix grayscale and color figures; verify the gallery unifies to true color
  (`is_rgb`) and every frame still shows the right content.
- Scrub and keyboard-arrow through all frames; verify each renders in true color
  with no scramble (decode uses the RGB stride, not a grayscale stride).
- Construct with `rgb=True` / `from_rgb` on a short color stack (<= 4 frames) the
  auto-detector cannot call, and `rgb=False` on an ambiguous `(3, H, W)` gray
  stack; verify the forced interpretation holds.
- Load a color figure with a recon `rotation_deg`; verify it rotates in true
  color (previously blocked) and stats read a re-derived luminance plane.

### S3D-19: Export A Shareable Color Movie Without A Dead File

**User story**: As a user sharing a color result movie, I want a single HTML
that actually opens in a browser, and a clear refusal (not a silently broken
700 MB file) when the movie is too big to embed.

**Primary widgets**: Show3D.

**Data to use**: a real RGB stack large enough that the float32 single-file
embed exceeds the safe limit, plus a small one that fits.

**Acceptance checks**:

- Export the small stack with default options; open the HTML over a local http
  server and verify true color renders and every frame decodes correctly.
- Export the large float32 stack; verify `export_html` refuses with a message
  naming the estimated size, the safe limit, and the smaller options
  (`encoding='uint8'`, `downsample=`, explicit `max_mb=`).
- Re-export the large stack with `encoding='uint8'`; verify it now fits and
  opens, and the color is visually faithful.
- Confirm `max_mb=<larger>` overrides the refusal for a deliberate big export.

### S3D-20: Scrub Full-Resolution Movies Over A Remote Jupyter Tunnel

**User story**: As a scientist running Jupyter on a GPU workstation but
viewing Show3D from my local laptop browser over an SSH tunnel, I want the
frame slider to feel real-time even for 3x3 grids of native 2048 x 2048 movie
panels, so I can judge dynamics without copying the dataset to the laptop or
waiting one to two seconds per slider tick.

**Primary widgets**: Show3D first; Show2D should follow the same remote
transport rule for comparable paged or stacked image review paths.

**Data to use**: A real or real-derived multi-panel movie on the backend. The
canonical stress case is a 3x3 grid of 2048 x 2048 float32 panels, where one
native concatenated frame is 150,994,944 bytes. Use the actual lab deployment
when possible: laptop browser on Apple Silicon, Jupyter kernel and data on a remote GPU workstation, and
an `ssh -L` tunnel. Synthetic data may be used only as a post-fix control when
it preserves the same native spatial shape and per-frame payload.

**Acceptance checks**:

- Measure before optimizing. Record Python prepare/wire/encode/trait-set time,
  browser receive/decode/paint-proxy time, and end-to-end UI latency as
  `now - sendTime`, over the real ssh tunnel rather than localhost.
- Verify the frame-server fast path is actually reachable from the local
  browser. A kernel-side `127.0.0.1` frame server is not enough when the browser
  runs on another machine; the browser's localhost is the laptop, not the GPU
  workstation.
- During active slider drag, avoid sending one full native frame through
  Jupyter Comm for every pointer tick. Prefer local cache, frame-server, or a
  bounded scrub-preview transport before falling back to native `slice_idx`
  commits.
- Preserve the full-resolution source arrays and committed frame path. Do not
  bin, crop, overwrite, or silently replace the scientific data to make the
  interaction faster.
- If drag-time display uses a reduced preview, print or log one explicit line
  naming the reduction factor and how to get native resolution. The expected
  language is equivalent to: `displaying {factor}x reduced frames during slider
  drag; release the slider or zoom/settle the view to request native full
  resolution`.
- Release the slider and verify native full-resolution pixels remain
  reachable. The committed frame path must still send or fetch the native frame
  and zoom/detail inspection must not be limited to the drag preview.
- Drive the live widget in JupyterLab from the laptop browser: drag slowly,
  drag quickly, step with keyboard, press Play/Pause, toggle loop/bounce, and
  verify the image, frame label, histogram, and slider stay synchronized.
- Record the story result as `Pass`, `Fail`, or `Not verified` with the
  backend host, browser host, tunnel port, data path, shape, dtype, native
  bytes, preview bytes/factor if any, debug counters, console errors, and
  report or notebook URL.
