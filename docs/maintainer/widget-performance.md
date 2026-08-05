# Performance

These notes capture interaction bugs that were easy to misread while building
the widgets. Keep this page short and practical: it should explain what went
wrong, how to recognize the pattern, and what to do instead.

## Current summary

2026-07-05 Show4DSTEM loader work:

- Exact `uint16` no-bin loading is already on the same fast path as browse
  `uint8` for a single real 512 x 512 x 192 x 192 Arina master. On a private
  reference CUDA workstation, the benchmark script measured `uint16` at 0.634 s
  cold / 0.405 s hot and `uint8` at 0.591 s cold / 0.392 s hot on the freer GPU.
- Multi-GPU loading now uses disk-aware scheduling. `load(masters,
  devices=[0, 1])` interleaves files by physical disk before assigning work to
  GPUs, so folders split across independent NVMe disks can use disk bandwidth
  and GPU capacity together.
- Two local-only benchmark entrypoints own this proof:
  `scripts/widget_load_bench_matrix.py` for U16/U8 single and stacked load
  timing, and `scripts/widget_load_bench_sharded.py` for disk layout,
  two-GPU placement, cold/warm timing, and capacity boundaries.
- Keep real benchmark outputs under `/tmp/quantem-widget-load-bench/`. Do not
  commit private paths, raw data, generated benchmark payloads, screenshots, or
  large reports.
- The sampled masters resolve to one physical NVMe disk, so
  the smoke validates the sharded code path and report harness, not an actual
  two-disk bandwidth gain. Prove the disk speedup on a host where
  `group_by_disk(masters)` reports at least two real disks.
- 2026-07-06 U8 follow-up on real private reference masters:
  `load(master, dtype="u8", det_bin=1)` measured 0.850 s cold / 0.385 s hot
  for one 512 x 512 x 192 x 192 master with parity enabled. Sharded
  `dtype="u8"`, two no-bin masters across `devices=[0, 1]`, measured
  1.428 s cold / 0.641 s hot with one 9.0 GiB U8 file resident on each GPU.
  The single-master browse path meets the <0.5 s hot-load target; the two-master
  sharded reload is close but not below target yet. The full remote
  Show4DSTEM browser signoff was blocked in that workstation environment by a
  neighboring `quantem.core` circular import during `import quantem.widget`.
- 2026-07-25 Show4DSTEM WebGPU seven-tilt browser signoff:
  seven full audited U8-masked `512 x 512 x 192 x 192` HDF5 tilt families
  loaded through Chrome WebGPU on the Apple Metal adapter. The visible
  `Show4DSTEM.command` bundle used audited `h5_uint8_lossless=True`, low8
  bitshuffle/LZ4 decode, fetch/parse/decode queueing, and
  `compare_max_panels=7` with `compare_group_mode="all"`. A CDP-driven
  2-second detector drag measured live compare virtual-image GPU slot updates
  for all seven panels: `paintFps=21`, `lastPaintedPanels=7`,
  `lastAdoptedPanels=7`, and `lastComputeMs=0.8`. The first broken attempt
  showed `computeFps>0` but `paintFps=0`; the fix was to upload the colormap LUT
  before compare-grid GPU paints and render each live compare panel through a
  visible WebGPU canvas with `renderSlotDirectWithGpuRangeToCanvas`, matching
  the single-tilt VI path. The browser-local proof object is
  `window.__sh4dLiveViStats`; it intentionally does not sync high-frequency
  samples to Python traits.

## Timing protocol for every widget

Every widget report should separate loading speed from rendering speed. A fast
loader can still produce a slow browser widget, and a smooth browser widget can
still hide a slow Python data-prep step. When a widget, loader, tutorial, or
release signoff is changed, record these timings in the notebook output,
signoff report, or PR notes:

- **Input**: widget name, data shape, dtype, raw byte size, file count, file
  format, backend, and whether the data were cropped, binned, downsampled,
  quantized, sparsely streamed, or exact.
- **Load**: wall time for the public loader, for example
  ``load(...)``, ``load_eds(...)``, ``read_image(...)``,
  ``read_images(...)``, or the tutorial loader.
- **Pack/build**: wall time for array stacking, display-bin generation,
  side-index construction, export packing, or other Python work before widget
  construction.
- **First browser paint**: widget timing after the frontend has decoded and
  painted the first useful view. For widgets that expose timing traits, report
  ``render_total_ms``, ``render_python_build_ms``, and
  ``render_wire_js_ms``.
- **Interaction**: FPS or pointer-to-preview latency for the control under
  test. For drag/zoom/scrub workflows, say whether the kernel became busy and
  whether the visible preview moved during the drag or only after release.
- **Export/reopen**: standalone HTML size, export time, reopen time, and
  whether the exported page is exact or reduced.

Prefer ``verbose=True`` in public examples where it helps users debug why a
notebook is slow. Verbose output should be short and copyable. Good output
looks like this:

```text
read_images workers=8: 0.604s for 40 EMD files; dtype=uint16; shape=(4096, 4096)
np.stack: 0.588s; stack=(40, 4096, 4096) uint16; raw=1.25 GiB
Show3D live profile
-------------------
Backend       NVIDIA CUDA workstation
Data          40x4096x4096 | uint16 | 1.25 GiB
Load          604 ms
Pack/prep     588 ms
Widget build  5.47 s

Show3D timing
-------------
Data             40x4096x4096 | uint16 | 1.25 GiB
Render total     5.47 s
Python build     820 ms
Wire + JS paint  4.65 s
Frame server     on
Offline stack    off
```

Do not report only the fastest number. For large data, report the total path a
scientist actually waits for: file IO, Python packing, widget construction,
browser first paint, and the interaction that matters for the workflow. If a
number is measured in a script rather than in JupyterLab or exported HTML, label
it as a script measurement.

Timing traits and debug surfaces should stay stable across widgets:

- ``render_total_ms``: total Python construction through first browser paint,
  when the widget can observe the frontend paint.
- ``render_python_build_ms``: subset spent in Python widget setup.
- ``render_wire_js_ms``: transfer, browser decode, and first paint remainder.
- ``debug=True`` / debug HUD: opt-in browser interaction telemetry. Show2D,
  Show3D, and Show4DSTEM expose a browser-local `Debug UI FPS` badge for quick
  agent and human smoke checks in notebooks and exported HTML. That badge is
  only the browser paint loop; widget-specific debug metrics should separately
  report decode, draw, compute, cache, memory, and interaction-latency costs.

Do not collapse all debugging into one FPS number. A scalable debug surface has
one common metric plus widget-specific metrics:

- **Common**: UI FPS, dropped frames, pointer-to-preview latency.
- **Show2D**: image decode/draw, histogram draw, FFT prep/worker/GPU/post time,
  ROI/profile time, page scrub latency, and panel cache state.
- **Show3D**: frame fetch/decode, numerical source/FFT cache hit/miss, play
  FPS, frame scrub latency, remote Comm/tunnel receive latency,
  scrub-preview factor/bytes when active, FFT/FFT-metric time, and panel/page
  cache state.
- **Show4DSTEM**: diffraction-pattern fetch, virtual-detector compute, detector
  ROI drag latency, compare-grid cache, WebGPU/backend path, and GPU memory.
- **ShowEDS**: map compute/draw, spectrum compute/draw, backend path, and sparse
  stream lookup costs.

Keep detailed debug counters browser-local, for example under an existing
`window.__quantemShow*Perf` object. Do not sync high-frequency debug samples
back to Python traits.

## Widget Performance Stories

Use this table as the widget-level user story checklist. It is the maintainer
view of what "fast enough" means before a change is called done. The cloud/CI
path should cover the lightweight protocol and export checks. The local heavy
signoff should cover real private data, browser screenshots, and FPS/cache
evidence.

| Widget | Primary user story | Must stay real-time | Performance proof |
| --- | --- | --- | --- |
| Show1D | Inspect scalar traces, losses, spectra, or per-iteration diagnostics while deciding which image view to open. | Cursor/readout movement, snapshot selection, and handoff to Show2D. | Lightweight state tests plus browser story when handoff or snapshot rendering changes. |
| Show2D | Compare one image or many related images, often 4K microscopy outputs, with zoom, pan, histogram, FFT, profile, pages, and export. | Zoom/pan, histogram controls, page slider/play, hidden panels, FFT redraws after first compute, and HTML reopen. | `scripts/widget_browser_smoke.py` for exported HTML; `scripts/widget_heavy_perf_signoff.py` for 4K real-data panels. |
| Show3D | Scrub or play time series, focal stacks, iterative reconstructions, and multi-panel comparisons without rebuilding the widget. | Frame scrub/play, remote-tunnel drag preview with native restoration, page slider/play, hidden panels, independent panel contrast, FFT return-scrub cache, FFT metric labels, GIF/MP4/HTML export. | `scripts/widget_heavy_perf_signoff.py`; [S3D-20](storyboard-show3d.md#s3d-20-scrub-full-resolution-movies-over-a-remote-jupyter-tunnel) live Jupyter tunnel proof; exported HTML profile for existing reports; animation smoke for GIF/MP4. |
| Show3DSlices | Browse volume slices and orthogonal views with synchronized crosshair/plane controls. | Slice sliders, crosshair movement, oblique line endpoint/body drags, side-plane redraw, oblique FFT redraw during line drag, FFT return-scrub cache hits for slice/oblique sliders, histogram controls, FFT/log/smooth toggles, and export reopen. | Browser smoke plus focused visual story when slice/crosshair/oblique-line behavior changes. |
| Show4DSTEM | Inspect diffraction patterns and virtual images from real 4D-STEM datasets without loading unnecessary data. | Scan-position movement, detector drag, BF/ABF/ADF updates, compare pages, cache-backed folder reopen, lazy folder sessions, and export reopen. | `scripts/widget_show4dstem_heavy_signoff.py` covers direct backend/export interaction only. S4D-19 additionally requires a recorded fresh-process folder-paging/cache runner and browser report; do not infer that signoff from the heavy script. Lightweight CI checks only the protocol. |
| ShowEDS | Explore spectral maps, ROIs, energy bands, element lines, and sparse/folder-backed EDS cubes. | Band dragging, map/spectrum sync, ROI changes, periodic table selection, sparse lookup/cache, and export reopen. | Browser story plus EDS-specific real-data smoke when backend or map/spectrum logic changes. |
| ShowDiffraction | Inspect diffraction-like 2D patterns when a full 4D-STEM session is not needed. | Zoom/pan, histogram/contrast, peak/FFT-style overlays when present, and export reopen. | Lightweight export/browser smoke; use Show4DSTEM heavy signoff for full detector workflows. |
| ShowFolder | Browse a session folder before loading heavy data, cache thumbnails, and open selected data in the right viewer. | Folder scan, thumbnail cache, live `watch_once()`, selected Show2D/Show3D refresh, lazy Show4DSTEM handoff, and compact saved state. | `scripts/widget_showfolder_live_smoke.py`; keep generated thumbnail/report artifacts outside git unless intentionally committed. |

## Folder-watching performance contract

`Show2D.from_folder(...)`, `Show3D.from_folder(...)`, and
`Show4DSTEM.from_folder(...)` start watching by default. Treat their watcher as
an append path, not as a periodic full rebuild.

## Show3D remote tunnel scrub contract

Remote Jupyter is a first-class production path for Show3D: the browser may be
on a laptop while the kernel, data, CUDA device, and any Python frame server
live on a workstation reached through `ssh -L`.

## Show3D direct-display no-blank contract

The Show3D canvas is rendered directly from the selected frame and the selected
panel's state. Do not introduce a browser-side prebuilt canvas/composite display
cache or a second display-state machine: those paths have repeatedly let page,
frame, contrast, and zoom disagree. Decoded source frames and numerical FFT
results may be cached when their keys are scientific inputs, but a numerical
cache must never independently decide which pixels, contrast, or transform are
shown.

Page play, frame autoplay, keyboard steps, Page slider changes, manual
frame-slider drag, and release can each use different handlers. A fix that
covers autoplay does not prove manual scrubbing is safe. Keep these rules:

- One renderer owns every panel paint from the current frame plus that panel's
  current display state.
- Retain the last complete scientific pixels until the next direct paint is
  ready. Do not `clearRect`, hide the GPU canvas, or fill the viewport with an
  inter-panel color unless replacement scientific pixels are painted in the
  same task.
- Preserve zoom/pan across a page change, without recomputing or copying a
  different panel's contrast, colormap, Smooth state, or selection.
- With linking off, a histogram edit writes only the addressed panel state;
  with linking on, the propagation is explicit and testable.

Browser signoff must include transition-time screenshots or equivalent pixel
sampling during manual lower-slider drag/release, Page slider scrub, Page play,
and frame autoplay. Scan for both white and black spikes in the scientific
canvas area; a final nonblank screenshot is not enough.

### Mistake log: do not reintroduce the display cache casually

The retired display-cache path combined prebuilt frame canvases, asynchronous
page preparation, and independent contrast/zoom state. Under normal scientist
actions it produced stale frames, blinking contrast, cross-panel edits, and
blank canvases. Any proposal to reintroduce it requires a dedicated issue,
one rendering owner, live-Jupyter plus fresh-export visual proof, and the
independence/no-blank checks in S3D-05A.

A kernel-local frame server on `127.0.0.1` is only fast if the browser can reach
that exact endpoint. Across an SSH tunnel, the browser's localhost is the
laptop; use measurement to prove whether frame fetches are actually reaching
the backend endpoint before assuming the fast path works.

Full-resolution movie data must remain full resolution. Do not solve remote
scrub latency by silently binning, cropping, or replacing the source array.
During active frame-slider drag, Show3D may use a bounded display preview to
avoid shipping one native frame per pointer tick over Jupyter Comm. That
preview is a transport/display optimization only, not a data reduction.

Any preview factor greater than 1 must announce itself once in Python output or
browser logs and say how to get native pixels, for example by releasing the
slider or zooming/settling the view. Release, keyboard step, playback settle,
zoom/detail inspection, and explicit native fetch paths must keep native pixels
reachable. Reports should list native bytes, preview bytes, factor, Python
encode time, browser receive time, decode time, and paint/UI latency.

| Viewer | Required append behavior | Work that must not repeat |
|---|---|---|
| Show2D | Add each new full-resolution image as one panel; automatically render folder pages of at most `page_size` panels (default 20) | Reread existing source files; rebuild the widget; replace full-resolution data with ShowFolder thumbnails; render every folder panel at once after paging activates |
| Show3D | Add each new full-resolution image as a frame in one unpaged stack | Reread existing source files; rebuild the widget; infer Show2D-style pages from frame count |
| Show4DSTEM | Add each ready master as a cold lazy dataset | Load every new master into VRAM immediately; clear unrelated reduced-page caches |

For Show4DSTEM, raw 4D residency and reduced virtual-image caching are separate
budgets. A visible cold page may load and compute once. Returning to a warmed
page should use its reduced BF/ABF/ADF/HAADF result, while `page_budget` remains
free to evict raw masters that are no longer needed. A newly appended master
should invalidate or warm only the comparison pages whose membership changed.

The canonical folder-paging behaviors are
[S2D-18](storyboard-show2d.md#s2d-18-watch-a-live-emd-folder-in-place),
[S2D-20](storyboard-show2d.md#s2d-20-page-a-growing-folder-gallery-automatically),
[S3D-17](storyboard-show3d.md#s3d-17-watch-a-live-emd-frame-series-in-place),
[S4D-14](storyboard-show4dstem.md#s4d-14-watch-a-live-4d-stem-acquisition-folder-in-place),
[S4D-17](storyboard-show4dstem.md#s4d-17-page-a-folder-safely-on-one-cuda-gpu)
and
[S4D-18](storyboard-show4dstem.md#s4d-18-pool-multiple-gpus-and-stream-pages-progressively),
plus persistent stale-while-revalidate behavior in
[S4D-19](storyboard-show4dstem.md#s4d-19-reopen-a-folder-with-persistent-scientific-previews).
For these stories, measure page acknowledgement, first panel, half page, full
page, and warm return separately. A full-page number alone hides whether the UI
waited unnecessarily for its slowest master. Multi-GPU reports must also record
per-card budget/resident bytes and prove that obsolete page generations cannot
paint after a rapid page change.

Live progressive pages should keep a small sequence-tagged synced copy of the
latest ready panel in addition to any custom binary-message fast path. Some
notebook frontends deliver custom control messages but drop their binary buffer
when it is sent by a background worker; the synced panel prevents those viewers
from replacing a successfully computed page with `Unavailable` placeholders.

Folder-watching signoff must measure initial scan, idle poll, and
return-to-warm-page latency. Split append latency into two stages instead of
reporting one ambiguous number:

- stable/readiness-confirmed file to Python/model append and visible label,
  count, page control, or reserved-placeholder paint
- user selection or page request to the first scientific canvas paint; for
  Show4DSTEM this means both virtual-image and diffraction pixels

It must also verify:

- an idle poll performs no decode, transfer, render, or cache invalidation
- incomplete files are retried without killing the watcher
- each stable file appends exactly once and in deterministic order
- the Python widget identity and existing panel/frame state remain unchanged
- Show2D and Show3D pixels match the full-resolution source, not thumbnails
- Show4DSTEM raw residency stays within its GPU budget as masters accumulate
- `stop_folder_watch()` is idempotent and `close()` leaves no watcher or cache
  worker running
- the mounted Jupyter widget and browser container remain the same, and a real
  browser canvas repaint—not only a Python trait change—is captured

Keep test folders temporary and add files through an atomic rename when
possible. Report source shape, dtype, append count, cache hits/misses, resident
raw bytes, both append latency stages, and browser console errors. Use genuine
microscope data in live JupyterLab for scientific signoff; CI can use small
generated files only to prove lifecycle and cache invariants.

### Show4DSTEM persistent-preview cache contract

The folder preview cache is a scientific stale-while-revalidate path, not a raw
data cache. It may paint a validated reduced float32 BF/ABF/ADF/HAADF image from
disk immediately, but the UI must identify it as cached while the normal CUDA
page scheduler prepares authoritative raw interaction. Raw GPU residency,
current-widget host cache, and persistent disk cache have independent budgets.

Cache records are per master, with an ordered fingerprint of every linked
detector chunk, so reordering or repaging does not duplicate previews and a
single changed chunk invalidates only its source master. Lookup must be metadata
and reduced-array I/O only: reading the cache must not decode raw 4D detector
data or allocate CUDA memory. Writes are atomic, corrupt or incomplete entries
are misses, concurrent readers cannot observe partial payloads, and disk-limit
eviction removes complete least-recently-used entries.

For a matching page request, capture these browser-visible timestamps
separately:

1. click to the first cached panel paint;
2. click to all cached panels on the visible page;
3. click to the first fresh raw-backed panel replacement;
4. click to all fresh panels on the visible page;
5. click to the complete requested page, including misses; and
6. foreground completion to next/previous neighbor-prefetch completion.

Use ``window.__quantemShow4DSTEMPerf.comparePage`` to keep receipt evidence
(``firstCachedPanelReceiptAtMs`` and ``firstFreshPanelReceiptAtMs``) separate
from the double-animation-frame after-paint proxies
(``firstCachedPanelPaintAtMs``, ``firstFreshPanelPaintAtMs``,
``cachedVisiblePaintAtMs``, and ``freshVisiblePaintAtMs``). For a fresh widget,
also split API-call-to-model-ready from model-ready-to-cached paint because the
current implementation still loads one calibration/diffraction master before
mounting.

Also report cache lookup/read/write duration and bytes, entries and disk bytes,
hits, misses, invalidations, evictions, corruptions, raw decode and reduction
time, stale-generation drops, per-GPU resident bytes, Debug UI FPS, and browser
console errors. A cache speed claim requires a fresh-process second open; a
return to an in-memory page in the same widget is a different measurement.

Run cold, full-hit, partial-hit, source/chunk-change, forced-rebuild,
disabled-cache, and refresh-failure cases on one NVIDIA GPU first. Cached panels
must remain visible with `Cached preview · loading raw data` (or an explicit
refresh-failed state), preserve stable grid geometry, and never flash black or
be relabeled fresh before raw-backed replacement. Repeat the hit/miss mix on
multiple selected NVIDIA GPUs to prove cache I/O is not duplicated per device
and S4D-18's capacity, serialization, cancellation, and generation rules still
hold. On the reference host, use five fresh-widget opens and require median
cached-first paint <= 500 ms and <= 50% of matched cold, median cached-visible
page <= 2 s and <= 25% of matched cold completion, and no more than 10%
regression in fresh-visible or complete-page time versus disabled persistence.
Report p95 for each paint stage. CPU is a deterministic lifecycle control; MPS
needs its own signoff.

### Show2D local-panel stack signoff (2026-07-09)

Private real-data signoff used one collaborator Velox EDS acquisition with a
`131 x 234 x 237` uint16 HAADF stack and four `234 x 237` elemental maps. The
source stayed outside git. Standalone artifacts were served over local HTTP and
driven with headless Playwright Chromium because in-app browser control was not
available in that session.

| Export | File size | First visible slider | Result |
|---|---:|---:|---|
| HTML exact float32 | 40,928,929 bytes | 1.80 s | frame 130 restored; scrub/play/stats/FFT/hide-restore passed |
| HTML quantized uint8 | 10,760,236 bytes | 1.34 s | same interaction checks passed |

Only the HAADF panel exposed a frame control. Moving from frame 130 to frame 0
changed both the canvas checksum and the browser-local stats. Play advanced to
frame 4 and remained there after pause; hiding and restoring HAADF preserved
frame 4. FFT remained visible and updated after another frame change. Both
exports completed without page or console errors, and the quantized export was
also checked at 820 px viewport width.

A 30-step requestAnimationFrame-paced keyboard scrub completed at 54.8 Hz
without FFT and 65.9 Hz with FFT visible in that headless run, ending on frame
30. These are input-scheduling rates, not a claim of measured GPU canvas FPS;
the accompanying checksum/final-frame checks prove the input was applied. A
future browser debug harness should record per-frame canvas presentation time
directly.

For Show2D and Show3D FFT specifically, the first compute may be expensive.
Once a frame, panel set, ROI, and windowing state has been computed, returning
to it must be a cache hit. A return scrub should increase hit counters while
misses and compute counters stay unchanged. Do not put traitlet delivery
counters such as `frame_seq` into FFT cache keys; invalidate the cache when the
data source changes instead.

Browser foreground restore is also a display-only invalidation. A tab switch
may discard a visible 2D canvas or a presented WebGPU texture even though the
scientific data and FFT magnitude caches are still valid. Show2D and Show3D
therefore wait for foreground compositing to settle, then rebuild colormapped
offscreen layers from retained data and re-blit/re-present every visible image,
FFT, inset, and overlay canvas. Tab return must not advance playback, change a
slice/frame index, or rerun an FFT. Verification should require the foreground
repaint signal to advance while FFT miss and compute counters remain unchanged.

Use ``quantem.widget.profile_widget`` in profiling notebooks to time the Python
construction path in the same format:

```python
from quantem.widget import Show3D, profile_widget, widget_timing_report

widget, profile = profile_widget(
    "Show3D live",
    lambda: Show3D(stack, verbose=True, save_state=False),
    data=stack,
    load_ms=read_ms,
    pack_ms=stack_ms,
    backend="NVIDIA CUDA workstation",
)
widget
```

After the widget paints in JupyterLab, ``widget_timing_report(widget)`` returns
the first-paint timing table again. New widgets should provide equivalent
timing hooks rather than inventing a private debug vocabulary. Existing widgets
should be updated opportunistically when their load/render path changes.

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

## Mistake log: Show3D standalone HTML WebGPU playback blank

Date: 2026-07-09

Symptom: a standalone exported Show3D HTML report looked correct when the user
dragged the frame slider by hand, but the same packed multi-panel view could go
blank or show stale/empty panels while `Play` was active. Pausing or manually
scrubbing made the panels look correct again, which made ordinary after-pause
screenshots misleading.

What was wrong:

- The standalone offline playback path was not always the same as the manual
  scrub path.
- Packed multi-panel HTML playback could opt back into a direct WebGPU canvas
  route even though the stable scrub path rendered through the static 2D
  canvas.
- The direct WebGPU display canvas and the React/static canvas can race during
  opacity/display handoff in exported HTML, especially on browser GPU stacks
  where GPU work may present one frame later than the JavaScript state update.
- The earlier generic "offline GPU playback owns canvas" guard was not enough:
  packed multi-panel exports need their own rule because they have per-panel
  contrast, panel geometry, and page/frame playback controls.
- Testing only a paused frame, a slider drag, or a unit test misses the bug.

Fix:

- For offline packed multi-panel Show3D HTML, keep frame playback on the stable
  static 2D canvas path. The code guard is
  `offlinePackedPanelPlaybackUsesStaticCanvas`.
- Do not let packed-panel standalone playback use the direct WebGPU display
  canvas unless a browser test proves active playback stays nonblank on the
  exported HTML.
- Preserve per-panel contrast during playback. Do not fall back to a global
  packed-frame range for BF/DF/DPC/SSB/COM panels with different physical
  scales.
- Test both visible play controls: page playback (`Play pages`) and frame
  playback (`Play`, `Play forward`, or reverse controls).

Required verification for future changes:

- Build and reinstall the local widget checkout before exporting:
  `npm run build && python -m pip install -e .`.
- Regenerate the standalone HTML after the code change; the exported file embeds
  the widget bundle.
- Open the regenerated HTML in a real browser and capture screenshots while
  playback is actively running, not only after pause.
- Compare active playback against manual slider dragging on the same page and
  frame family.
- Check browser console errors and run the report smoke script when a report
  harness exists.

## Policy: Show4DSTEM backend and memory ownership

For Show4DSTEM performance reports, always separate three surfaces:

- **Python backend work**: file loading, detector binning, virtual detector
  computation, export packing, and any live kernel-backed recompute. Record
  whether this used CUDA/Torch, raw Metal/MPS, Torch-MPS, or CPU.
- **Browser interaction**: canvas rendering, pointer events, layout, and WebGPU
  work in live Jupyter or exported HTML. Record the browser and WebGPU adapter.
- **Saved/exported artifacts**: standalone HTML or HTML plus a data folder,
  which should not require Python, Torch, CUDA, or MPS after export.

On MacBook/Apple Silicon, the raw Metal/MPS path is preferred for large
first-pass 4D-STEM browsing because it gives tighter control over chunking,
detector binning, dtype, and transient memory than a generic Torch-MPS tensor
path. Torch-MPS can still be valid for specific tensor workflows, but reports
path. Scientific detector compute must fail clearly instead of falling back to
CPU.

Multi-master `load([masters])` differs by backend: CUDA eager-stacks masters
into one resident 5D array, while MPS decodes dataset 0
synchronously, shows the viewer immediately, and fills datasets 1..N-1 from a
single background GPU worker owned by `quantem.gpu.io`. Performance reports for
multi-master sessions must say which of the two paths ran.

GPU memory belongs to the backend data object and Python session, not the
visual widget alone. The viewer should avoid leaking buffers and should keep
saved state compact, but freeing GPU memory should be handled by backend/session
lifecycle: delete or replace the loaded data object, clear references, use a
backend-specific cache cleanup utility if one exists, or restart the kernel. Do
not add or test a misleading "free GPU" viewer button unless it explicitly
reports backend ownership and delegates to a documented backend cleanup path.

## Show4DSTEM heavy signoff

The repeatable heavy Show4DSTEM proof is
`scripts/widget_show4dstem_heavy_signoff.py --backend cuda`. It is local-only
and must stay out of normal CI because it depends on real lab ``*_master.h5``
files and can generate private screenshots and HTML exports.

The report must keep the same split as the policy above:

- backend: real master discovery, first NVIDIA/CUDA load, widget build time,
  backend shape/dtype/device, resident memory, append or stack-growth time for
  new masters, and Python/GPU memory before/after;
- export: explicit ``uint8``/``uint16`` choice, detector bin factor, file size,
  and packing time;
- browser: WebGPU adapter information, virtual-detector drag FPS, scan-position
  movement FPS, dataset/frame flip FPS for multi-master sessions, recompute
  latency, wheel-zoom FPS, console errors, and a screenshot.

For persistent-preview changes, add two fresh-process NVIDIA/CUDA runs over the
same source/configuration: one true cold population and one cache-backed reopen.
The second report must include click-to-cached-first, cached-visible-page,
fresh-first, fresh-visible-page, complete-page, and prefetch timing rather than
only the backend wall time. Preserve the cold raw decode timing as a background
I/O metric; a fast cached first paint does not make a 20-plus-second cold page
complete time disappear.

`--skip-browser` is allowed only to debug backend or export failures. It is not
a performance signoff because the user-facing requirement is smooth browser
interaction.

Use `--devices 0,1 --max-masters 20` or `--max-masters 30 --det-bin 1
--skip-browser` for the no-bin two-GPU capacity stress. That run answers a
different question from browser FPS: can the NVIDIA backend hold the requested
real stack, and if not, does it fail with an auditable report and release GPU
memory before the next run? A single 512 x 512 x 192 x 192 uint16 master is
about 18 GiB resident, so 20-30 masters no-bin is a capacity stress, not a
reasonable default expectation on every two-GPU machine.

2026-07-05 CUDA no-bin result on a private NVIDIA lab workstation: two RTX PRO
6000 GPUs (about 96 GiB each) loaded real experimental masters at `det_bin=1`. A
four-master stack (4 x 512 x 512 x 192 x 192 uint16, about 72 GiB resident)
passed the browser-enabled signoff: first master load was about 0.8 s, widget
build about 0.6 s, stack growth to four masters about 1.5 s, and browser
scan-position, detector, wheel-zoom, and dataset-flip checks all measured about
60 FPS. The compact standalone export for that four-master stack used
`uint8` with `export_det_bin=8`, produced an 85 MB HTML file, and took about
28-33 s on the workstation run. A 40-master no-bin capacity probe failed
cleanly while appending the fifth master with an allocation request of about
18 GiB; the script released GPU memory, reloaded the last successful
four-master stack, and wrote the failure report. This is expected for
eager-resident no-bin data: 30-40 files would be roughly 540-720 GiB of
detector data before viewer overhead. For 30-40 no-bin files as a normal
workflow, use the shipped paged path instead of eager residency:
`Dataset5dstem` lazy loaders plus `Show4DSTEM(..., page_budget=...)` (CLI
`--page-budget`, default `auto`) keep a bounded GPU-resident set and evict raw
masters on demand. The no-bin capacity stress above still measures the
eager-resident ceiling; a paged run answers the workflow question.

After the capacity stress, run a browser-enabled multi-master pass that fits in
memory. That pass must prove the user workflow, not only the load path: the
viewer opens quickly, additional masters are available through the Dataset
slider, and flipping between loaded datasets stays near the target FPS while
scan-position and detector interactions remain responsive.

Use `--backend mps` only for MacBook fallback checks. It should not replace the
NVIDIA/CUDA heavy signoff when that backend is available.

## Show4DSTEM loader benchmark matrix

Use these local-only scripts when the claim is about raw loader speed, not the
browser UI:

```bash
PYTHONPATH=src:. python scripts/widget_load_bench_matrix.py \
  --masters-glob "$QUANTEM_WIDGET_BENCH_MASTERS_GLOB"

PYTHONPATH=src:. python scripts/widget_load_bench_sharded.py \
  --masters-glob "$QUANTEM_WIDGET_BENCH_MASTERS_GLOB" \
  --devices 0,1
```

`QUANTEM_WIDGET_BENCH_MASTERS_GLOB` may be one glob or several globs separated
by `os.pathsep` (`:` on Linux/macOS). Use that form when a folder was split
across multiple NVMe mounts, for example:

```bash
export QUANTEM_WIDGET_BENCH_MASTERS_GLOB='/path/to/disk0/*_master.h5:/path/to/disk1/*_master.h5'
```

The scripts write Markdown tables under `/tmp/quantem-widget-load-bench/` by
default. Keep the generated reports local unless a release note intentionally
summarizes the numbers. Do not commit raw data, generated benchmark payloads, or
machine-specific private paths.

Loader policy:

- `dtype="u16"` is the exact-count browse/reconstruction path. Optimize it, but
  do not replace it with `uint8` silently.
- `dtype="u8"` is an explicit browse path that decodes directly into uint8 and
  clips values above 255. It is useful for fast folder screening and memory
  pressure, not for exact-count reconstruction.
- `load(masters, devices=[0, 1])` is disk-aware. The loader interleaves files by
  physical disk first, then assigns the interleaved stream to GPUs. If users
  split masters across independent disks, both disk bandwidth and GPU capacity
  can be used. If all masters are on one disk, sharding is still useful for
  capacity but the cold load remains disk-bound.
- Each benchmark case runs in a fresh subprocess. Do not time parity hashes
  inside the load timer; a full uint64 sum over tens of GiB is a correctness
  check, not load latency.

Current private reference-workstation measurement, single real 512 x 512 x 192 x 192 Arina
master, no detector binning, parity checked against the full tensor:

| path | first measured load | hot repeated load | resident size | note |
| --- | ---: | ---: | ---: | --- |
| `load(master, dtype="u16", det_bin=1)` | 0.953 s | 0.320-0.321 s | 19.33 GB | exact uint16 counts |
| `load(master, dtype="u8", det_bin=1)` | 1.26 s in a fresh process; 0.311 s after U16 warmup | 0.309-0.314 s | 9.66 GB | direct uint8 decode for browsing |

Post-migration smoke using the new scripts on the same private workstation
confirmed that the entrypoints run end to end. This smoke used
`--skip-parity`; rely on the full parity run above for numerical equivalence.

| script smoke | cold | warm | resident size | note |
| --- | ---: | ---: | ---: | --- |
| matrix, `dtype="u16"`, no-bin, single master on the freer GPU | 0.634 s | 0.405 s | 18.0 GiB | exact browse path, GPU0 was occupied |
| matrix, `dtype="u8"`, no-bin, single master on the freer GPU | 0.591 s | 0.392 s | 9.0 GiB | direct uint8 browse path |
| sharded, two masters, `devices=[0, 1]`, `det_bin=4` | 1.114 s | 0.684 s | 2.2 GiB | one real master per GPU |

2026-07-06 U8 no-bin verification, parity enabled:

| script | cold | warm | resident size | note |
| --- | ---: | ---: | ---: | --- |
| matrix, `dtype="u8"`, no-bin, single master | 0.850 s | 0.385 s | 9.0 GiB | meets the <0.5 s hot single-master target |
| sharded, two masters, `dtype="u8"`, no-bin, `devices=[0, 1]` | 1.428 s | 0.641 s | 18.0 GiB total | one 9.0 GiB U8 master per GPU; close, not yet <0.5 s |
| matrix, `dtype="u16"`, no-bin, single master | ERR | - | - | current GPU memory was not clean enough for the exact-count allocation |

The current sample resolved to one physical NVMe disk for the
available real masters, so the smoke validates sharded GPU placement and the
benchmark harness, not a real multi-disk bandwidth gain. To prove the disk
speedup, run `widget_load_bench_sharded.py` on a host where
`group_by_disk(masters)` reports two or more real disks.

Interpretation: U16 is already on the same hot path as U8. Further U16 wins are
more likely to come from first-use warmup, disk layout, or multi-file scheduling
than from changing the single-master decode kernel. For 20-40 no-bin masters,
use the sharded benchmark table to prove the real workflow: cold load, hot load,
per-GPU resident GiB, disk groups, and whether the run failed cleanly at the
expected capacity boundary.

IO review and next optimization targets:

- Keep `dtype="u8"` routed to `output_dtype=np.uint8` before any low-level load
  path. This must hold for single masters, stacked `load(masters, dtype="u8")`,
  `load(masters, devices=[...], dtype="u8")`, and legacy
  `load(masters, gpus=[...], stack=False, dtype="u8")`. If the route regresses,
  the public API can silently materialize uint16 first and lose the browse
  memory/speed benefit.
- The remaining <0.5 s gap for two no-bin U8 masters is not a single-file
  decoder problem. Focus on sharded reload overhead: per-worker first-use
  warmup, per-file group scheduling, thread startup, repeated HDF5 metadata
  opens, and whether each device can reuse pinned/compressed scratch buffers
  across files.
- Multi-disk proof is still missing. The scheduler is disk-aware, but the
  current real masters are all on `nvme2n1`. A valid multi-disk report must show
  `group_by_disk(masters)` with at least two disks and then compare one-disk and
  split-disk cold/warm timing with the same master count, dtype, and detector
  binning. Use `quantem data-transfer plan/copy/masters/show4dstem` to create
  and record that split layout; keep the manifest path and timing report local
  when it contains private workstation paths.
- Do not run two GPU-heavy loader benchmarks concurrently on the same GPUs.
  Parallel benchmark processes create artificial OOMs and hide the true loader
  behavior. Run U8/U16 and sharded/single cases serially unless the goal is an
  explicit contention test.
- Fix environment import blockers before claiming browser signoff. The direct
  HDF5 benchmarks can bypass `import quantem.widget`, but the real
  Show4DSTEM UI/export signoff cannot. A remote `quantem.core` circular import
  blocks the full widget path and must be reported as `Not verified`, not
  papered over with loader timings.
- For 30-40 no-bin masters, use the shipped lazy/paged resident set
  (`page_budget`, `Dataset5dstem` lazy loaders) instead of opening all masters
  hot. The benchmark can prove per-file load speed and GPU placement; the user
  workflow still needs fast dataset flipping with bounded resident memory and
  clear eviction/cache status in the viewer/report.

## Heavy Show2D / Show3D audit

Date: 2026-07-02

Goal: heavy scientific data should show something useful quickly, but users
must still be able to inspect the highest-resolution pixels the workflow can
support. These are related but not identical contracts for Show2D and Show3D.

Real-data stress input used for the audit:

- Source: local real ptychography reconstruction arrays from an ignored lab
  data directory. Keep the exact private path out of shared documentation.
- Show2D: eight real-derived 4096 x 4096 panels, tiled from ADF/SSB/WDD arrays
  to preserve real lattice/noise structure while stressing the browser path.
- Show3D: twelve real-derived panels, 32 frames each, 2048 x 2048 native source
  per panel.

Test topology:

- The data construction/export path ran from the quantem.widget Python backend
  on the workstation repo checkout used for the audit. In normal lab use this
  can be an HPC/workstation backend: Python owns the large arrays, file I/O, export
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
  through Jupyter with the patched checkout placed first on ``sys.path`` so the
  Jupyter backend used the local changes.
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

FFT metric label policy:

- FFT quality labels are useful for live microscopy because they make peak
  sharpness and reciprocal-space signal quality visible without opening another
  panel. They must stay cheap enough to leave playback and pointer interaction
  smooth.
- Compute FFT quality metrics from the cached FFT magnitude. Do not run a
  second FFT for a label.
- Cache the label by scientific inputs: FFT magnitude version, FFT dimensions,
  sampling, units, crop, and panel grid. Do not include Stats/Profile toggles,
  toolbar visibility, hover labels, or other display chrome in the metric cache
  key.
- Keep the label as a stable overlay inside the FFT image, using the same
  compact white-on-image style as panel labels. Do not add a separate stats row
  for the metric unless the user explicitly asks for a detailed readout.
- Verify correctness with a deterministic NumPy parity test before trusting the
  browser label. The current test is ``js/fftMetrics.numpy.test.ts``.
- Verify performance with browser counters. ``scripts/widget_heavy_perf_signoff.py``
  must report zero FFT compute growth and zero FFT metric compute growth while
  toggling the Stats UI.

## Agent Storyboards

Performance notes should capture what we learned: measured timings, failure
modes, payload tradeoffs, and implementation policy. UI behavior and recurring
AI/browser drive plans live in [Storyboard](storyboard). Repeatable heavy
browser performance gates live in
[Performance UI Testing](performance-ui-testing).

Use the storyboard for scientific user stories and release signoff. Link the
story IDs from performance investigations when a speed or correctness issue maps
to a recurring user workflow.

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
