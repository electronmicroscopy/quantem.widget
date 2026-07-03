# Performance

These notes capture interaction bugs that were easy to misread while building
the widgets. Keep this page short and practical: it should explain what went
wrong, how to recognize the pattern, and what to do instead.

## Mistake log: Show3D cursor readout pop

Date: 2026-06-30

Symptom: the Show3D cursor readout showed correct row, column, and value text,
but it felt "poppy" during fast mouse movement over multi-panel images. The
label appeared to flash or jump even though the image interaction itself was
working.

What was wrong:

- Mousemove events updated React state directly for every pointer event.
- The cursor readout mounted and unmounted as the pointer crossed valid and
  invalid image regions.
- The label width changed with each row, column, and formatted value, making
  the overlay feel unstable even when the coordinates were correct.
- The bug is a visual performance problem, not a numerical correctness problem,
  so unit tests alone would not catch it.

Fix:

- Route cursor readout updates through `requestAnimationFrame` so rapid pointer
  events collapse to at most one visual update per frame.
- Keep the overlay DOM stable while the cursor is active, and fade it with
  `opacity` / `transform` transitions instead of relying on mount/unmount.
- Use tabular numeric text and a minimum label width so value changes do not
  resize the label every frame.
- Keep pointer overlays `pointer-events: none` unless the user is deliberately
  interacting with the overlay control.
- Drive the widget in standalone HTML or Jupyter and move quickly across the
  canvas before calling the interaction smooth.

Rule for future cursor and hover UI:

- Treat cursor labels, value readouts, drag hints, hover controls, and ROI
  handles as high-frequency UI.
- Do not update React state on every raw `mousemove` or `touchmove` when the
  update is only for a visual overlay.
- Prefer refs plus `requestAnimationFrame` for the preview path, then commit
  stable widget state only when needed.
- Avoid popping overlays in and out of the DOM. Keep one stable element when
  possible and animate opacity or transform.
- If an overlay is hidden, make sure it does not steal pointer events from the
  scientific image underneath.

## Heavy Show2D / Show3D audit

Date: 2026-07-02

Goal: heavy scientific data should show something useful quickly, but users
must still be able to inspect the highest-resolution pixels the workflow can
support. These are related but not identical contracts for Show2D and Show3D.

Real-data stress input used for the audit:

- Source: ``/Users/macbook/repos/arina-ptycho/results/res36mrad_33.3us_512rpx_tR1_192qpx_bQ8.npz``.
- Show2D: eight real-derived 4096 x 4096 panels, tiled from ADF/SSB/WDD arrays
  to preserve real lattice/noise structure while stressing the browser path.
- Show3D: twelve real-derived panels, 32 frames each, 2048 x 2048 native source
  per panel.

Test topology:

- The data construction/export path ran from the quantem.widget Python backend
  on the workstation repo checkout used for the audit. In normal lab use this
  can be mjgoat or buffle: Python owns the large arrays, file I/O, export
  packing, and any live detail tile replies.
- The interaction path was tested in the Codex in-app browser on the Mac. That
  is the machine exercising browser rendering, canvas compositing, WebGPU
  colormapping when available, mouse/trackpad events, and exported-HTML mount
  behavior.
- Do not mix those measurements: Python export/build timings tell us backend
  packing cost; browser mount and interaction timings tell us whether the Mac
  frontend can actually use the result smoothly.

Hypotheses tested:

- Large widgets must first load a small enough preview to be useful quickly.
- Show2D can preserve exact native inspection by streaming visible full-res
  detail tiles after zoom, so auto-binning is an initial-view optimization.
- Show3D does not yet have the same LOD/detail-streaming contract. For Show3D,
  display binning is currently a real display tradeoff that must be documented.
- FFT overlays should use cached, display-sized inputs for heavy gallery/movie
  views. Overlay FFTs do not need full native-resolution input on every scroll
  or frame interaction.

Show2D result:

- ``display_bin="auto"`` produced a 4x binned 1024 x 1024 preview for each
  4096 x 4096 panel. Native stack size was about 512 MB; preview payload was
  about 32 MB, and the uint8 HTML export was about 11.3 MB.
- Backend benchmark: constructing the eight-panel real-derived 4K stack took
  about 0.25 s in the focused rerun; exporting the uint8 standalone HTML took
  about 0.13 s. A later focused rebuild/export after the Show2D export-menu fix
  completed in about 0.94 s total, including data construction and export.
- The Python detail path returned a native crop for zoomed inspection:
  requesting preview rows/cols 256:384 at display bin 4 returned a 512 x 512
  float32 tile with ``bin=1`` and native origin ``row0=1024, col0=1024``.
- Browser audit drove Profile, FFT, Stats, zoom/pan, histogram contrast,
  Smooth, and linked zoom/pan controls on the heavy page. The visible cursor
  readout reported native coordinates and native/detail value sources.
- Export audit: the exported standalone page intentionally has backend export
  disabled. The frontend must not show an empty backend-only export menu. The
  tested fix gives standalone pages a visible Show3D-style HTML export action,
  such as ``HTML quantized uint8`` for a quantized export, and reports the saved
  size, for example 11.9 MB for the audited page.

Show2D policy:

- Keep ``display_bin="auto"`` as the default for large galleries.
- The initial image may be a binned preview, but ``_data`` remains full
  resolution on the Python side.
- Once zoomed past preview resolution, the frontend requests only the visible
  full-resolution crop via ``_detail_request``. Small high-zoom windows can
  become native-pixel tiles; larger windows are lightly binned to keep replies
  responsive.
- The info popover and title badge must make this visible: preview first,
  streaming detail while the request is in flight, then detail/native when the
  crop is available. Cursor rows and columns are always native coordinates.

Show3D result:

- A 12-panel x 32-frame 2048 x 2048 source movie is about 6 GB of native
  float32 data before display packing.
- A ``display_bin=2`` standalone export still wrote about 512.7 MB because the
  binned panels were concatenated into 32 frames of 1024 x 12288 display data.
  In the in-app browser it stayed blank after about 104 seconds. Treat this as
  a failed load-fast configuration.
- The same source with ``display_bin=4`` wrote about 128.7 MB, mounted in about
  4.6 seconds in the in-app browser, and supported playback, frame scrubbing,
  linked zoom/pan, histogram contrast, Profile/Stats toggles, FFT overlay
  rendering, FFT overlay pan/zoom, and drag-to-snap FFT overlay placement.
- Backend benchmark for the practical ``display_bin=4`` Show3D export:
  building the real-derived data took about 6.6 s and writing the uint8 HTML
  took about 2.6 s. The slower/failing ``display_bin=2`` export took about
  9.7 s to write but was not usable as a load-fast browser artifact.
- FFT overlay interaction is a separate event path from real-space zoom. A bug
  in the first implementation relied on React ``onWheel`` propagation only; in
  exported HTML, some wheel events still reached the parent canvas listener and
  zoomed the real-space panels while the user was over an FFT inset. The fix is
  a capture-phase native wheel guard plus coordinate hit-testing against
  ``data-show3d-fft-inset`` rectangles. The audit rerun used the real-derived
  12 x 32 x 2048 page and verified repeated wheel in/out and drag over P06:
  real-space panels stayed at ``1.0x`` while the FFT inset zoomed/panned.
- FFT correctness was checked against NumPy on the P02 real-derived panel/frame
  that looked suspicious in the browser. The FFT magnitude and shifted peak
  locations matched NumPy; the visual problem was the display transform. The
  strongest non-DC peaks were close to the center, so the previous log +
  percentile display left a broad low-frequency pedestal that looked like a
  blob in the small overlay. Auto FFT display now subtracts a radial background
  per FFT tile before clipping, which makes the same peaks visible without
  changing the cached magnitude data used for measurements.
- Browser interaction observations are qualitative unless a timing is listed
  above. For heavy views, keep recording concrete mount time, visible frame
  response, scroll/zoom behavior, and whether the kernel becomes busy during
  frontend-only interactions.

Save/reopen audit:

- Date: 2026-07-02 / 2026-07-03 overnight pass.
- A dedicated notebook, ``tmp/codex_show_save_reopen_e2e.ipynb``, was executed
  through Jupyter with ``/Users/macbook/mjgoat/repos/quantem.widget/src`` placed
  first on ``sys.path`` so the MJ/Jupyter backend used the patched checkout.
- The notebook displayed one Show2D and one Show3D with ``save_state=False``.
  Both printed ``_static_fallback_jpeg=True`` and ``frame_bytes=False`` from
  their full ``get_state()`` snapshot.
- In JupyterLab on ``127.0.0.1:8811``, pressing ``Cmd+S`` and reloading the
  notebook still showed visible ``Show2D static render`` and ``Show3D static
  render`` image outputs with nonzero dimensions. This verifies the user-facing
  save/reopen path, not just the Python unit test path.
- The regression tests that should stay green are
  ``tests/test_save_state.py`` and
  ``tests/test_widget_performance_contract.py``. The performance contract is
  deliberately coarse: it verifies that lightweight save snapshots complete
  under a generous budget and do not contain heavy frame/detail/export buffers.
  Browser FPS needs browser-side instrumentation, not a normal pytest timing.

Show3D policy:

- Show3D currently does not have Show2D-style full-resolution tile streaming on
  zoom. ``display_bin=N`` is therefore a real display tradeoff, not just a
  transport optimization.
- For heavy multi-panel movies and exported HTML, prefer an explicit
  ``display_bin`` that keeps the exported display payload below roughly
  100-150 MB. In the audited 12 x 32 x 2K case, ``display_bin=4`` was practical
  and ``display_bin=2`` was not.
- If native pixels are required in Show3D, use ``display_bin=1`` in a live
  workflow that can tolerate the larger transfer and memory cost, or create a
  separate high-resolution focused view. Do not claim that a binned Show3D
  export can zoom back to exact native pixels.
- A future Show3D LOD design should mirror Show2D: first show a binned preview,
  then stream exact full-resolution tiles for the visible panel/frame window on
  zoom. Until that exists, docs and UI copy must be explicit about the tradeoff.

## Agent E2E Performance Checklist

Use this checklist whenever an agent changes Show2D or Show3D rendering,
playback, save-state, FFT, histogram, export, or large-data transfer code. A
change is not "verified" until the browser has been driven on real or
real-derived data and each item below is reported as Pass, Fail, or Not
Verified.

Test topology to report:

- Backend host: mjgoat, buffle, local Mac, or other. Include the repo checkout
  and whether the notebook inserted the intended ``src`` path ahead of any
  installed package.
- Frontend host/browser: Codex in-app browser on the Mac, Chrome, Safari, etc.
  This is the machine that proves canvas/WebGPU/pointer performance.
- Data: real file path or real-derived source, array shape, dtype, panel count,
  frame count, native bytes, display bin, exported HTML size, and whether the
  page is live Jupyter or standalone HTML.
- WebGPU status: available adapter when known, plus fallback path if WebGPU is
  unavailable. Do not claim WebGPU speed from a CPU/browser fallback.

Hard acceptance targets:

- First visible paint: usable preview appears in seconds. Target under 5 s for
  the standard heavy smoke pages. Anything approaching 10 s is a performance
  failure unless explicitly justified by data size and documented.
- Interaction rate: zoom, pan, histogram drag, frame scrub, playback, FFT
  overlay zoom/pan, and sliders should sustain about 30 FPS on the Mac frontend
  for the standard heavy smoke pages. A qualitative "seems fine" is not enough;
  use a browser-side frame counter or record the measured interaction FPS.
- Save/reopen: press ``Cmd+S`` in JupyterLab, reload or reopen the notebook,
  and verify Show2D and Show3D outputs remain visible. Unit tests are required
  but are not browser proof.
- Payload discipline: ``save_state=False`` must not embed heavy buffers such as
  ``frame_bytes``, ``_buffer_bytes``, ``_detail_bytes``, offline stacks, or
  export payloads in the full notebook-save snapshot.
- FFT: overlay and side/bottom FFT views must be cached and display-sized for
  heavy pages. They must not recompute native-resolution FFTs during every
  scroll, pan, playback tick, or slider movement.

Show2D checklist:

- Load a real/real-derived large gallery, for example 8 or more 4096 x 4096
  panels from the drift/ptycho data.
- Verify first paint shows a binned preview quickly and the title/info badge
  communicates the preview/detail state.
- Zoom past preview resolution and confirm a full-resolution detail request is
  issued. The returned tile must use native row/column coordinates and a tile
  ``bin`` smaller than ``_display_bin_factor`` when possible.
- Move the mouse over preview and detail regions. Cursor readout must stay
  stable and must label native/detail/preview value source correctly.
- Drive columns, panel visibility, Profile, FFT, histogram contrast handles,
  scale mode, colormap, Smooth, linked zoom/pan/contrast, Reset, Copy, and
  Export.
- Drag histogram handles and sliders repeatedly. The image must update at the
  interaction target rate without stale detail tiles, ghost rectangles, or
  delayed labels.
- Open the export menu in live Jupyter and standalone HTML. Labels must say
  exactly what will be exported, such as ``HTML exact float32`` or
  ``HTML quantized uint8``, and include approximate file size when known.
- Press ``Cmd+S``, reload the notebook, and verify visible Show2D output. Also
  run ``tests/test_save_state.py`` and
  ``tests/test_widget_performance_contract.py``.

Show3D checklist:

- Load a real/real-derived heavy movie, for example 12 panels x 32 frames from
  2048 x 2048 native source. Record ``display_bin`` and final display payload
  size.
- Verify first visible paint. ``display_bin=4`` was practical for the audited
  12 x 32 x 2K page; ``display_bin=2`` produced a too-large export and failed
  the load-fast requirement.
- Drive Play/Pause, frame slider, FPS slider, averaging slider, Loop/Bounce,
  keyboard frame stepping, and direct frame scrub. The visible frame and slider
  must track each other at the interaction target rate.
- Drive columns, panel visibility, Profile, Stats, linked zoom, linked
  contrast, colormap, scale mode, Smooth, histogram contrast handles, Reset,
  Copy, and Export.
- Test FFT in bottom, right, and overlay layouts. For overlay, verify every
  panel gets an overlay, overlay size control works, drag-to-snap moves all
  overlays consistently, wheel over FFT zooms/pans FFT instead of the
  real-space panels, and Shift-drag pans FFT detail.
- Verify FFT correctness against NumPy for at least one suspicious real panel
  or frame when peak visibility looks wrong. Distinguish magnitude correctness
  from display transform/contrast problems.
- Verify FFT performance: cached/display-sized FFT should not drop playback,
  frame scrub, or zoom below the interaction target. If FFT is enabled, repeat
  the playback and slider tests with FFT visible.
- Press ``Cmd+S``, reload the notebook, and verify visible Show3D output. Also
  run ``tests/test_save_state.py`` and
  ``tests/test_widget_performance_contract.py``.

Evidence format for agent reports:

- ``Verified``: URL/notebook, data source, backend host, browser host, widget
  versions, exact interactions driven, measured first-paint time, measured FPS
  or frame-count method, save/reopen result, and tests run.
- ``Not verified``: anything not actually driven in the browser, including FPS
  if no frame counter was used.
- ``Remaining risk``: unsupported browser/backend combinations, data sizes
  beyond the tested payload, or known paths such as Show3D native-pixel LOD that
  are not implemented yet.
- Never report "all good" from screenshots, DOM inspection, or Python tests
  alone.

## Show2D / Show3D Agent Storyboard

This storyboard is the recurring AI drive plan for Show2D and Show3D. Run it
when either widget changes, before broad release candidates, and whenever a
performance regression is suspected. Every item must be marked ``Pass``,
``Fail``, or ``Not verified``. Use real or real-derived data first; synthetic
data is only a control.

Shared setup:

- Open the target in the Codex in-app browser or Chrome on the frontend Mac.
- Use an MJ-goat or buffle Jupyter backend when the change involves real data,
  large arrays, or save/reopen behavior. Record the backend hostname and the
  widget source path inserted into ``sys.path``.
- Test at desktop and mobile-sized viewports. For mobile, use a narrow browser
  viewport as a pre-check; physical iPhone Safari still requires a separate
  iPhone run when the change is touch-specific.
- Capture the browser URL, data path, shape, dtype, native bytes, panel count,
  frame count, display bin, first-paint time, and interaction FPS method.
- Start from a fresh render after any code/build change: rebuild, reload, and
  rerun the notebook cell or reopen the exported HTML.

User-story map:

Use these stories to decide whether the lower-level checklist actually covers
the scientist's workflow. A release signoff can cite the story ID plus the
checklist item numbers that were driven in the browser.

Show2D user stories:

- S2D-01: As a microscopist opening a large single image, I want a useful
  preview within seconds so I can decide whether the file is worth inspecting.
- S2D-02: As a user comparing several real-space images, I want to choose the
  number of columns so I can arrange panels for my monitor and paper figures.
- S2D-03: As a user screening many panels, I want to hide unimportant panels
  without losing labels, stats, export order, or saved state.
- S2D-04: As a user inspecting atomic/lattice detail, I want the initial binned
  preview to stream native-resolution tiles when I zoom in.
- S2D-05: As a user reading coordinates, I want hover readouts to stay in
  native ``(row, col)`` coordinates even when the displayed preview is binned.
- S2D-06: As a user adjusting contrast on noisy data, I want histogram drags to
  update smoothly without stale square tiles or flash artifacts.
- S2D-07: As a user comparing panels, I want linked zoom, pan, and contrast to
  be optional and reversible.
- S2D-08: As a user looking for periodicity, I want FFT views to align with
  every visible panel and remain fast during resize and column changes.
- S2D-09: As a user measuring image features, I want profile and ROI overlays
  to be stable while I drag them quickly.
- S2D-10: As a user preparing a shareable result, I want export choices to
  clearly say exact float32 vs quantized uint8 HTML and show approximate size.
- S2D-11: As a notebook user, I want ``Cmd+S`` and reopen to preserve a visible
  compact output without embedding huge pixel buffers.
- S2D-12: As a mobile or narrow-window user, I want controls to wrap and remain
  usable without covering the scientific image.

Show3D user stories:

- S3D-01: As a user opening a time series or focal stack, I want first paint in
  seconds so I can start scrubbing before the workflow feels blocked.
- S3D-02: As a user comparing datasets side by side, I want multi-panel Show3D
  to use the same visual language as Show2D: labels, scale bars, colormaps,
  histogram controls, and compact status text.
- S3D-03: As a user arranging many movie panels, I want a column selector so I
  can switch between one row, multiple rows, and dense galleries.
- S3D-04: As a user screening many panels, I want to hide panels and have
  playback, FFT, stats, export, and saved previews follow the visible set.
- S3D-05: As a user playing a movie, I want the image, frame label, histogram,
  and slider to stay synchronized at the selected FPS.
- S3D-06: As a user scrubbing a large stack manually, I want slider movement to
  feel immediate and never show stale frames or background flashes.
- S3D-07: As a user comparing dynamics across panels, I want linked zoom and
  linked contrast to be fast and reversible.
- S3D-08: As a user inspecting reciprocal space, I want FFT layouts on bottom,
  right, or overlay without changing real-space interaction semantics.
- S3D-09: As a user using FFT overlays, I want one overlay per visible panel,
  centered on the FFT center, cached, independently zoomable, pannable, and
  draggable with corner snap.
- S3D-10: As a user validating FFT peaks, I want suspicious FFT views compared
  against NumPy or another reference before trusting the display transform.
- S3D-11: As a user making animations, I want GIF and MP4 exports to show only
  image panels with clean borders, one label per panel, and predictable sizes.
- S3D-12: As a user sharing HTML, I want exact/quantized export labels and
  sizes to match the Show2D export vocabulary.
- S3D-13: As a notebook user, I want ``Cmd+S`` and reopen to show a compact
  Show3D fallback that is pixel-matched to a Show2D current-frame gallery.
- S3D-14: As a user on constrained hardware, I want display binning to be
  explicit: fast preview is acceptable, but native-pixel availability must be
  clearly stated.
- S3D-15: As a mobile or narrow-window user, I want playback controls, frame
  slider, FFT controls, and panel menus to remain reachable.

Show2D storyboard:

1. Load a single real 4k or larger image. Verify first paint in seconds.
2. Load a 2-panel real gallery. Verify labels, panel borders, scale bars, and
   stats match the intended Show2D visual style.
3. Load 8 or more real-derived 4k panels. Verify first paint, memory, and
   scrolling remain usable.
4. Change the column selector through 1, 2, 3, 4, 6, 8, and 12. Verify the grid
   reflows without changing the current zoom center or contrast.
5. Confirm the column menu does not offer impractical counts above 12.
6. Hide one panel, multiple panels, and all-but-one panel. Verify layout,
   labels, stats, histograms, exports, and keyboard panel selection ignore the
   hidden panels.
7. Restore hidden panels. Verify original ordering and state return.
8. Toggle Profile. Draw, move, and delete a line profile. Verify high-frequency
   pointer labels do not pop or lag.
9. Toggle FFT. Verify each visible panel gets the expected FFT view and that
   changing columns does not misalign FFT panels.
10. In FFT mode, zoom and pan. Verify wheel and drag target the intended FFT
    view, not a stale real-space layer.
11. Compare one FFT against NumPy for a suspicious real panel when peak
    visibility looks wrong.
12. Toggle ROI tools. Draw, drag, resize, save, restore, and delete ROIs.
13. Drag histogram min/max handles quickly. Verify no stale square tile,
    ghost rectangle, white flash, or delayed color update remains.
14. Drag the histogram center/range repeatedly while the mouse readout is
    visible. Verify labels are stable and throttled.
15. Toggle Auto contrast off/on. Verify manual range is preserved until the
    user asks for auto again.
16. Change scale mode, colormap, and Smooth. Verify the image, histogram, and
    exported fallback all match the chosen display settings.
17. Enable linked zoom, pan, and contrast. Verify linked panels move together.
18. Disable linked zoom/pan/contrast. Verify independent panel state works.
19. Zoom past preview resolution on a binned big-image page. Verify native
    detail streaming starts and cursor readout reports native coordinates.
20. Pan across a native-detail boundary. Verify old detail tiles are never drawn
    after the view changes.
21. Resize the panel/grid. Verify the view remains anchored and does not jump.
22. Use keyboard navigation: previous/next panel, reset zoom, and delete ROI.
23. Open Copy. Verify copied output corresponds to the current visible state.
24. Open Export. Verify menu labels say ``HTML exact float32`` and/or
    ``HTML quantized uint8`` and show approximate sizes when known.
25. Export exact and quantized HTML where supported. Open both files and drive
    columns, hide panels, FFT, histogram, zoom, and reset.
26. Press ``Cmd+S`` in JupyterLab, reload the notebook, and verify the saved
    static output is visible and compact.
27. Check ``metadata.widgets`` or ``get_state()`` for heavy-buffer leaks:
    ``frame_bytes``, ``_detail_bytes``, offline stacks, and export payloads must
    not be present when ``save_state=False``.
28. Test a narrow mobile viewport. Verify controls wrap, panels remain usable,
    labels do not overlap, and horizontal scrolling is intentional if present.
29. On mobile/touch, test pinch or wheel-equivalent zoom, drag pan, menu open,
    column selection, and panel visibility.
30. Record FPS for zoom, pan, histogram drag, FFT interaction, and sliders. The
    target is about 30 FPS on the standard heavy pages.

Show3D storyboard:

1. Load a single-panel real 3D stack. Verify first paint and the current frame
   label.
2. Load a 2-panel real stack with different panel titles. Verify labels include
   panel title, dynamic frame label, and frame count.
3. Load a heavy real-derived multi-panel movie, e.g. 12 panels x 32 frames.
   Record native shape, display bin, payload size, and first-paint time.
4. Change the column selector through 1, 2, 3, 4, 6, 8, and 12. Verify the grid
   reflows without changing current frame, zoom center, or contrast.
5. Confirm the column menu does not offer impractical counts above 12.
6. Hide panels and restore them. Verify playback, frame slider, FFT overlays,
   stats, exports, and saved previews use only visible panels.
7. Press Play/Pause at 30 FPS. Verify the image and slider stay synchronized.
8. Increase FPS and repeat playback. Record whether the slider lags the image.
9. Drag the frame slider slowly and quickly. Verify no yellow/background flash,
   stale frame, or delayed label appears.
10. Use keyboard frame stepping. Verify frame labels and histogram update.
11. Change averaging window during playback. Verify it does not stall or desync.
12. Toggle Loop and Bounce. Verify end-of-stack behavior is correct.
13. Toggle Profile. Draw, move, and clear profile lines while scrubbing frames.
14. Toggle Stats. Verify multi-panel stats are readable and update with frame.
15. Toggle linked Zoom. Verify linked zoom/pan across all visible panels.
16. Toggle linked Contrast. Verify contrast changes apply consistently when
    linked, independently when unlinked.
17. Change scale mode, colormap, Smooth, and histogram range. Verify playback
    remains responsive.
18. Resize the grid/panels. Verify the current frame, labels, scale bars, and
    histogram UI do not jump or change layout unexpectedly.
19. Toggle FFT bottom layout. Verify spacing between Show3D and FFT, panel
    alignment, resize behavior, and histogram placement.
20. Toggle FFT right layout. Verify vertical height aligns with the real-space
    panels and controls remain reachable.
21. Toggle FFT overlay. Verify every visible panel receives one overlay.
22. Change FFT overlay size. Verify overlays resize independently from the
    real-space panel grid and avoid blocking scale bars when possible.
23. Drag FFT overlay and release near each corner. Verify snap-to-corner works.
24. Wheel over FFT overlay. Verify FFT zooms, not the underlying real-space
    image.
25. Shift-drag or the documented gesture over FFT overlay. Verify FFT panning
    works at zoomed-in scale.
26. Verify FFT overlay starts centered on the FFT center, not an edge or corner.
27. Compare one FFT against NumPy for a real panel/frame when peak visibility
    looks wrong.
28. Verify FFT cache behavior: revisiting a panel/frame should not recompute
    unnecessarily, and playback with FFT visible must stay responsive.
29. Open Export. Verify HTML exact/quantized/GIF/MP4 labels, approximate sizes,
    and cancellation/status cleanup behavior.
30. Export HTML exact and quantized where supported. Open both and drive
    playback, frame slider, columns, hide panels, FFT overlay, histogram, and
    reset.
31. Export GIF and MP4 for panel-only animations. Verify one label per panel,
    sane border/background, expected frame count, and file sizes.
32. Press ``Cmd+S`` in JupyterLab, reload the notebook, and verify the saved
    Show3D static output is visible.
33. Compare saved Show3D fallback pixels against a Show2D current-frame gallery
    for the same panels. The expected max channel difference is zero for the
    controlled pixel-parity test.
34. Check ``metadata.widgets`` or ``get_state()`` for heavy-buffer leaks:
    ``frame_bytes``, ``_buffer_bytes``, offline stacks, and export payloads must
    not be present when ``save_state=False``.
35. Test a narrow mobile viewport. Verify controls wrap, playback controls stay
    reachable, labels fit, and the frame slider remains usable.
36. Test touch-style drag and scroll gestures in the mobile viewport.
37. Record FPS for playback, frame slider drag, zoom, pan, histogram drag, FFT
    overlay zoom/pan, and export-page interaction. The target is about 30 FPS on
    the standard heavy pages.

Release-gating rule:

- If any P0 item fails, do not tag an RC: first paint over roughly 10 s, blank
  saved output, heavy-buffer save leak, broken export menu, playback/slider
  desync, FFT correctness failure, or interaction far below the target FPS.
- If a P1 item is not verified, the RC report must say exactly why and who will
  verify it next. P1 examples: mobile physical iPhone checks, maximum-size
  datasets, or hardware-specific WebGPU adapter coverage.
- The storyboard report must be linked from the release candidate signoff.

## Mistake log: ShowEDS band center drag

Date: 2026-06-27

Symptom: the ShowEDS real-data widget could compute maps quickly, but dragging
the center of the energy band still felt slightly delayed. The debug HUD showed
acceptable map, spectrum, and draw times, so the lag was initially missed.

What was wrong:

- The center-drag preview used the same React state path as normal committed
  widget state.
- Every mousemove could trigger widget rerender work and spectrum canvas work,
  even though the user only needed the visible band rectangle to translate.
- The performance HUD measured compute and draw durations, not the full
  pointer-to-preview latency that the user feels.
- The bottom MUI range slider is a poor target for very narrow energy windows
  because the two thumbs overlap. The spectrum band body is the reliable center
  drag target for narrow windows.

Fix:

- During center drag, move lightweight DOM preview overlays with imperative
  `transform` and `width` updates.
- Store the pending band in refs while dragging.
- Feed the pending band into the throttled map scheduler during drag, because
  the element-map overlay is part of the expected live feedback.
- Commit `band_start`, `band_end`, and notebook state once on mouseup.
- Keep endpoint drags on the normal precise state path.

Rule for future high-FPS widget selectors:

- Separate preview interaction from committed state.
- Use refs and CSS transforms for per-pointer-frame visual feedback.
- Do not call the interaction real-time based only on compute timings. Drive it
  in the in-app browser and judge pointer-to-preview response.
- Avoid Python/kernel round trips and notebook model saves during drag.
- Recompute expensive data on a throttle or on commit unless the computation is
  genuinely required for the next visual frame.
- If the user expects a derived overlay, map, or spectrum to move while dragging,
  that derived view is part of the preview and must be updated live through the
  fastest available scheduler.
- Keep all redundant views of the same selection synchronized during preview:
  the plot band, bottom slider handles, text readout, and derived overlay should
  move as one interaction.

This applies to ShowEDS energy bands and ROI drags, Show4DSTEM detector masks,
Show2D contrast controls, and any future draggable selector that needs to feel
attached to the pointer.

## Mistake log: EDS is a query source, not a spreadsheet

Date: 2026-06-28

Symptom: a real Velox EDS EMD file opened quickly in vendor tools, but the
prototype treated the spectrum image like a dense ``(row, col, energy)`` table
that had to be expanded before interaction. That was the wrong model. The user
usually asks for a current energy window, an ROI spectrum, or a visible preview,
not every empty channel in every pixel.

What was wrong:

- Native EDS files should be treated as query backends. Keep the file/chunks as
  the source and ask for only the data needed by the current view.
- A ShowEDS data folder is a prefix-cache export format. It is useful for small
  or deliberately spatial-binned portable demos, but it is not the default model
  for native no-bin analysis.
- Calling ``cube.compute()`` before a targeted query or explicit spatial binning
  defeats lazy I/O.
- Browser widget state is for small embedded demos, not native EMD storage.

Rule for future EDS work:

- Never expand a native EDS file just to prove a widget can open it.
- Default no-bin EMD loading to native/lazy queries.
- Build prefix-cache data folders only for existing caches, explicit sidecar
  requests, or intentional binned sharing/export workflows.
- Guard prefix-cache and widget-state sizes before reading data.
- Use lazy chunked sum-binning only for explicit portable demos and exports.
- Treat spatial binning as count-preserving; make energy binning explicit.
- The best long-term path is a sparse/tiled frontend backend: energy-window
  queries produce maps, spatial-window queries produce spectra, and WebGPU does
  the visible accumulation/drawing without Python round trips during drag.

Current ShowEDS policy:

- Small embedded cubes stay browser/WebGPU backed.
- ``ShowEDS.from_emd(..., backend="auto")`` uses an existing data folder when
  present; otherwise exact no-bin EMD uses the native lazy query path.
- Portable real-data demos can use an explicitly spatial-binned data folder.
- Exact one-file HTML export is not available for native lazy EMD because the
  exported page has no local query backend; use binned single-file export or a
  data-folder export when sharing outside Jupyter.

Update from the 0016 Velox stream test:

- Velox EDS ``SpectrumStream`` data is sparse event data. The logical dense
  shape can be tens of GB, but the actual useful stream can be a few hundred MB.
- Do not materialize zeros. Index the stream directly by channel and by pixel.
- A sparse stream data folder for the 2048 x 2048 x 4096 0016 file stores about
  26.9 million events in about 186 MB and keeps the full field of view exact.
- Full-field interaction should be validated with no crop and no binning before
  offering binned/export presets.
- If Jupyter ignores HTTP ``Range`` and returns ``200 OK`` with a whole file,
  slice the returned buffer when it contains the requested byte window instead
  of failing the sidecar worker.

## Mistake log: ShowEDS real-time interaction regression

Date: 2026-07-02

Symptom: the real DGGG 0039 EDS widget loaded and displayed the map/spectrum,
but changing the energy band or ROI felt slow on the full 1024 x 1024 x 4096
file. The standalone export could still render, so it was easy to mistake this
for a drawing problem instead of an interaction-backend problem.

What was wrong:

- ``backend="auto"`` could still fall back to the kernel-backed path for real
  sparse EDS data. That made pointer interaction depend on Python callbacks and
  notebook message traffic.
- The survey notebook default did not force the stream path, so old notebook
  outputs could make the widget look interactive while fresh runs were not.
- Kernel-backed interaction tried to keep recomputing map and spectrum data
  during drag. That is the wrong contract for pointer preview.
- The spectrum defaulted to a linear y scale. For sparse EDS this makes the
  high-count low-energy region dominate and hides useful peaks, so users have
  to turn on log scale manually before the spectrum looks reasonable.

Fix:

- Make ``ShowEDS.from_emd(..., backend="auto")`` prefer the sparse stream index
  whenever the native EMD stream can be indexed safely.
- Make survey EDS widgets default to ``backend="stream"`` so real screening
  notebooks do not silently take the slow kernel path.
- Keep kernel mode as an explicit fallback/debug path, and avoid expensive
  kernel recomputation during active pointer drags.
- Set the EDS spectrum default to log scale. Users can still turn it off, but
  the first view should expose peaks rather than only the total count wall.
- Verify with real data, not only synthetic cubes: export/open the real 0039
  stream widget, confirm ``1024x1024x4096 | Sparse stream | Cu K`` is visible,
  then drag both the energy band and ROI and watch the labels/counts update.

Low-pass and smoothing guidance:

- Low-pass filtering can be useful for making noisy EDS maps look more
  reasonable during exploration, especially when the selected band has low
  counts or the ROI/map is sparse.
- Treat it as a display/preview option, not a numerical correction. Saved
  counts, exported spectra, ROI sums, and element quantification should remain
  based on raw counts unless a workflow explicitly asks for filtered data.
- Default log scale is higher value than default low-pass filtering for EDS:
  log scale improves spectrum readability without changing the counts.
- If adding a low-pass UI, label it as smoothing, keep it off by default for
  quantitative readouts, and preserve an obvious raw view so users do not
  confuse denoised display with measured signal.

Rule for future real-time EDS work:

- Sparse event data should stay sparse from disk to browser. Do not materialize
  a dense cube or bounce through Python during drag.
- The default path for real screening must be interactive. Slow exact paths can
  exist, but they should be explicit.
- Validate the default path by rerunning a fresh notebook or standalone export
  from real EMD data, then drive the in-app browser. Do not accept stale notebook
  output as proof.
- Check the backend label, visible control state, and browser console before
  claiming the widget is fixed.
