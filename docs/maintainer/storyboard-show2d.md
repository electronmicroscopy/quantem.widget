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
- Enable Reorder, drag panels into a non-source order, and verify
  `panel_order`, keyboard navigation, hidden-panel export, and `to_show3d()`
  follow the displayed order while source-index labels/stars remain correct.

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
- Verify every interactive FFT panel always shows its own accurate multiplier,
  including the default `2.0×`; independent and linked wheel/pinch zoom must
  update the correct labels, and reset must restore `2.0×` without changing
  real-space zoom.
- While the viewer is paused, switch to another browser tab and return. Every
  real-space and FFT canvas must repaint automatically with the same zoom,
  pan, multiplier, and selected panel. FFT cache-hit state may change, but FFT
  miss and compute counters must not increase merely because the tab returned.
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
- Verify the single-HTML choices distinguish `encoding="full"` from
  `encoding="uint8"` and show approximate sizes when known.
- Export single HTML with both encodings where supported.
- [x] **EX-2**: Export a denoised view and a frequency-filtered view, then
  open each standalone page without touching a control. The first canvas must
  already show the filtered pixels; the histogram and canvas must not disagree.
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
- Open Export and verify the single-HTML `full` and `uint8` encoding choices
  follow the Show3D wording and show approximate file sizes when known.
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

### S2D-16: Compare Static EDS Maps With Local HAADF Frames

**User story**: As a microscopist reviewing an EDS acquisition, I want static
elemental maps beside the acquisition's HAADF frame stack so I can inspect
drift or damage without forcing every comparison panel onto one global frame.

**Primary widgets**: Show2D.

**Data to use**: one real Velox/EDS acquisition containing multiple HAADF
frames plus at least three static elemental maps. Keep private microscope data
outside git and record only shape, dtype, frame count, and timings.

**Acceptance checks**:

- Pass a list containing 2D maps and one or more `(frames, rows, cols)` items;
  verify only stack panels get an in-panel slider and play button.
- Scrub and play each stack independently. Verify pixels, histogram, stats,
  profile, diff, and FFT follow the current frame without changing static maps.
- Sustain a 30 Hz rAF-paced slider input stream on a normal-size EDS frame, or
  record the browser/data limit. Confirm the final canvas, not only the slider
  label, reaches the requested frame.
- Hide, restore, and reorder a stack panel; verify its frame index is unchanged.
- Select a stack panel and verify left/right arrows scrub it. Select a static
  panel and verify arrows retain ordinary gallery navigation.
- Save a notebook with `save_state=True`, close it, and reopen without running
  the cell. Verify the slider, current frame, and pixels restore.
- Export single HTML with `encoding="full"` and `encoding="uint8"`, open both
  without a kernel, and repeat scrub, play/pause, stats, FFT, hide/restore, and
  narrow-viewport checks with no browser errors.

### S2D-17: Scrub A Paged Comparison Gallery

**User story**: As a user comparing a parameter sweep organized as pages (for
example lambda values, each page holding the same panel set), I want the page
slider and page playback to swap the whole panel grid at once while every
per-panel control - histograms, labels, hide state - follows the page I am
looking at, not the union of all pages.

**Primary widgets**: Show2D.

**Data to use**: a real or real-derived `(pages, panels, rows, cols)` stack
with at least 3 pages x 4 panels, per-page distinct intensity ranges so
histogram mismatches are visible at a glance.

**Acceptance checks**:

- Construct with 4D pages plus `page_labels`; verify the page slider shows
  `1/N`, the label text, and play/pause advances pages at the configured fps.
- With `link_contrast=False`, verify exactly `panels_per_page` histograms are
  shown - never one per panel across ALL pages - and that their ranges change
  when the page changes. Repeat with the FFT histogram grid.
- Hide a panel on one page; verify the histogram grid drops that panel's
  histogram, and other pages' hide state is independent.
- Drag one page's histogram thumbs; verify only the current page's matching
  panel changes and the setting is still applied when scrubbing back to that
  page.
- Verify stats, hover readout, and selection follow the current page's panels
  after several slider scrubs and one full playback loop.
- Export offline HTML and repeat the histogram-count and page-scrub checks
  without a kernel.

### S2D-18: Watch A Live EMD Folder In Place

**User story**: As a microscopist collecting survey images, I want one already
displayed Show2D gallery to add each completed EMD as a full-resolution panel,
without rerunning the cell, creating another widget, or losing the curation and
measurement state of images already present.

**Primary widgets**: ``Show2D.from_folder(...)`` in live JupyterLab. ShowFolder
is a separate cached-preview and selection workflow; a ShowFolder refresh or a
standalone HTML snapshot does not prove this direct full-resolution watcher.

**Data to use**: Genuine compatible Velox image EMD files from one acquisition
session, copied through atomic rename into a temporary watched folder. Use at
least three files for routine verification and ten for release signoff. Include
an incomplete file, an incompatible-shape file, and a later compatible file.
Record source path, shape, dtype, sampling, native bytes, backend host, browser,
widget commit, and watch interval. Keep the source data and generated report
outside git.

**Acceptance checks**:

- In live JupyterLab, display exactly one ``Show2D.from_folder(folder,
  pattern="*.emd", watch=True, watch_interval=1.0, debug=True)`` object and
  capture its Python identity, widget model ID, and browser container before
  any later file arrives.
- Start once with a valid EMD and separately start before the first acquisition.
  The empty-folder case must show a visible waiting state and accept the first
  stable EMD without rerunning the cell; until that works, mark this check
  ``Fail`` rather than substituting a prepopulated folder.
- Keep one compact accessible watch badge near the folder/title area in stable
  DOM. Require green-dot ``Watching`` only while the actual background worker
  is alive. Enter ``Updating`` while a discovery poll is active and keep it
  through real candidate validation, append, and authoritative full-resolution
  paint. An idle poll may briefly show ``Updating`` but must return to
  ``Watching`` without decode, transfer, or repaint. Use amber ``Waiting for
  file completion`` for an incomplete EMD, red ``Watch error`` with corrective
  detail for a bad shape or worker failure, and gray ``Stopped`` or
  ``Not watching`` after stop/close
  or when liveness cannot be established. A ``watch=False`` snapshot has no
  badge. A restored notebook model, static fallback, or standalone snapshot
  must never restore a green ``Watching`` state without a live worker. Capture
  browser assertions and screenshots for every state; color alone is not the
  status signal.
- Atomically complete a genuine EMD after the widget is visible. Verify one new
  labeled panel paints automatically without calling ``poll_folder()``. Compare
  its shape, calibration, representative pixels, and canvas checksum against
  ``read_image(path).array`` at source resolution.
- Verify the Python object, widget model, and browser container remain the same.
  Preserve selected source path, stars, hidden panels, panel order, rotations,
  contrast, FFT, ROI/profile state, view box, and zoom when a naturally sorted
  filename is inserted before existing files.
- Leave an incomplete EMD in place and then complete it. Verify it remains
  retryable, appends exactly once when stable, and never stops the watcher.
- Add an incompatible-shape EMD followed by a compatible EMD. Report the
  mismatch without resizing scientific data or blocking the later valid append.
- Measure stable-file-to-Python-append and stable-file-to-first-canvas-paint
  separately for every arrival; report median and p95 with the poll interval.
  A trait/count change without visible paint is not a pass.
- Verify idle polls perform no reread, transfer, canvas repaint, or state reset.
  Exercise idempotent stop/resume, then call ``close()`` and prove no watcher
  thread can mutate the widget afterward.

### S2D-19: Compare Independent Slice Stacks Across Reconstructions

**User story**: As a tomography or multislice reconstruction scientist, I want
each Show2D gallery panel to hold a different reconstruction volume with its own
slice slider and play/pause control, so I can browse each result at its own
depth while keeping all reconstruction alternatives side by side in one
viewer.

**Why this matters scientifically**: One study may produce four reconstructions
with different slice counts, slice thicknesses, regularization settings,
alignment choices, or reconstruction methods. A feature or failure can be
clearest at a different depth in each result. Independent local slice controls
let the scientist stop each panel at the informative depth while linked zoom,
pan, and contrast preserve the spatial comparison. This avoids opening several
widgets or forcing every reconstruction onto one global slice index.

**Primary widgets**: Show2D. Use Show3D instead when every panel should advance
on one shared frame or slice axis.

**Data to use**: four real or real-derived tomography or multislice
reconstructions, each shaped `(slices, rows, cols)`. Include different slice
counts or slice thicknesses when scientifically meaningful, plus an optional
static 2-D reference. Record source, shape, dtype, slice count, slice thickness,
display binning, and timings. Put the reconstruction method and slice thickness
in each panel label because the compact slider readout reports only the current
slice and total count.

**Acceptance checks**:

- Pass a Python list of at least four 3-D reconstruction arrays; verify every
  stack panel gets its own slider, play/pause button, and `current/total`
  readout. Verify an optional 2-D panel gets no slice controls.
- Start panels at different `panel_frame_indices`, including a negative index,
  and verify each index resolves against that panel's own slice count.
- Scrub one panel and verify only that panel's index and canvas change. Its
  histogram, stats, FFT, profile, ROI, and diff inputs must follow the selected
  slice while neighboring panels remain unchanged.
- Play two stacks simultaneously, pause one, and verify the other continues;
  verify each stack wraps against its own slice count.
- Construct with `panel_playback_fps=4`, measure several slice transitions,
  and verify the shared local-stack cadence is approximately 4 fps without a
  new toolbar control. Verify the configured value survives state restore and
  standalone HTML export.
- With FFT visible, traverse every slice twice. After the initial FFT is
  visible, verify first-time slice computations retain the previous valid FFT
  instead of showing a full dark loading veil; on the second traversal, cache
  hits increase while misses and computes stay unchanged.
- Pause every local stack on a different slice, switch browser tabs, and return.
  Verify all real-space and FFT canvases restore without pressing Play, every
  local slice index is unchanged, and no cached FFT magnitude is recomputed.
- Verify linked zoom, pan, and contrast preserve spatial comparison without
  linking the local slice indices.
- Select each stack and scrub with left/right arrows. From Python, use
  `set_panel_frame(panel, slice_index)` with either a source index or unique
  panel label.
- Hide, restore, reorder, and responsively wrap panels; verify each local slice
  index remains keyed to its source reconstruction and controls do not overlap.
- Save and reopen with `save_state=True`. Export single HTML with
  `encoding="full"` and `encoding="uint8"`, then repeat independent scrub,
  play/pause, FFT, and narrow-layout checks without a kernel.
- On real reconstruction data, record first paint, memory and payload size,
  slider latency, and canvas repaint rate. Confirm the final canvas, not only
  the slider label, reaches every requested slice.

### S2D-20: Page A Growing Folder Gallery Automatically

**User story**: As a microscopist whose acquisition or reconstruction folder
may grow to hundreds of independent images, I want Show2D to show a bounded
number of panels at once and create additional pages automatically, so the
notebook remains readable while every full-resolution source image stays
available. I want a useful default and an explicit override rather than having
to reorganize files manually.

**Primary widgets**: ``Show2D.from_folder(...)``. This is sequential item
paging: each file is one independently curated panel. It is distinct from
S2D-17 comparison pages, where the same logical slots repeat across parameter
settings. ``Show3D.from_folder(...)`` deliberately does not use this behavior;
its files extend one frame axis.

**Data to use**: at least 45 naturally named, same-shape real or real-derived
images so the default layout contains 20, 20, and 5 panels. Also start with 20
files and append the 21st through an atomic rename while the widget is mounted.
Include a small deterministic NPY fixture for CI and genuine EMD images for
browser signoff.

**Acceptance checks**:

- ``Show2D.from_folder(folder)`` defaults to ``page_size=20``. Exactly 20 ready
  images remain one ordinary gallery; the 21st creates two pages without
  replacing the widget. ``page_size=N`` changes the limit and ``None`` disables
  automatic folder paging. ``set_folder_page_size(...)`` regroups the mounted
  widget in place. Reject booleans, zero, negatives, and non-integers with a
  corrective message.
- Show compact labels such as ``Images 1–20`` and ``Images 41–45`` plus a
  bounded, tabular ``current/total`` page readout. The final page contains only
  real source panels: never pad it with fake scientific images.
- Keep folder pages as independent items. Hiding, starring, selecting, or
  rotating one file must remain keyed to that file and must not affect the same
  numerical slot on another page. Preserve a valid global panel order when
  paging activates after a live arrival.
- Append across 20 -> 21 and later page boundaries while the user is browsing
  an older page. Preserve the Python object, widget model, browser container,
  selected source path, active page, ROI/profile, FFT, contrast, zoom, and
  curation state. Do not jump to the newest page unless the user follows it.
- Drive page 1, page 2, the last partial page, and page 1 again. Verify the DOM
  contains no more than ``page_size`` scientific canvases, current-page
  histograms and FFT panels match the visible files, number shortcuts select a
  slot on the active page, and rapid page changes cannot leave stale canvases.
- Verify page controls stay grouped on a narrow viewport, use the shared MUI
  control language, and remain keyboard accessible. Capture light/dark browser
  screenshots and console logs.
- Keep the folder watcher append-only: only the new source file is read from
  disk and existing scientific files are not reread. ``Updating`` must remain
  truthful until the newly visible panel paints, then return to ``Watching``.
- Record initial scan, append, page-switch, histogram/FFT, browser memory, comm
  payload, and paint timings for 20, 45, and 100+ images. The first
  implementation limits visible React panels but still retains and transports
  the full gallery; do not claim bounded Python/browser memory until an
  active-page transport, cache, and generation-safe prefetch path is separately
  implemented and verified.
