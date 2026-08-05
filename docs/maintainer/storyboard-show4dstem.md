# Show4DSTEM Storyboard

Use with [Storyboard](storyboard).

MacBook support is a first-class Show4DSTEM target, not an afterthought. For
large real 4D-STEM data, agents should explicitly test Apple Silicon
raw-Metal/MPS loading and detector-binned U8 browse workflows, then separately
test browser/WebGPU exported HTML and any CUDA/Torch workstation path relevant
to the release.

## Stories

### S4D-01: Open 4D-STEM Data Quickly

**User story**: As a 4D-STEM user opening a scan, I want a useful virtual image
and diffraction preview in about a second for normal preview sizes so I can
start inspecting data immediately.

**Primary widgets**: Show4DSTEM.

**Data to use**: real or tutorial 4D-STEM stack; include full and binned preview
variants when available.

**Acceptance checks**:

- Load from Jupyter and exported HTML when supported.
- Measure first visible paint, scan shape, diffraction shape, dtype, native
  bytes, binning/downsampling, and WebGPU availability.
- Verify virtual image and diffraction panels render with labels, scale bars,
  and correct units.
- Verify frontend-only pan/zoom/detector interactions do not make the kernel
  busy unless backend recomputation is expected.

### S4D-02: Relate Scan Position To Diffraction Pattern

**User story**: As a scientist inspecting 4D-STEM, I want scan-position movement
in the virtual image to update the diffraction panel immediately.

**Primary widgets**: Show4DSTEM.

**Data to use**: real 4D-STEM scan with recognizable diffraction variation.

**Acceptance checks**:

- Click and drag scan position across the virtual image.
- Verify diffraction updates at the current scan position and labels/readouts
  stay synchronized.
- Hover scan positions without clicking and verify hover coordinates/readouts
  update for the hovered position without committing a new selected scan
  position or changing the diffraction panel until the user clicks or drags.
- Use keyboard or slider navigation if available.
- Record FPS or latency for scan-position movement.

### S4D-03: Tune Virtual Detectors

**User story**: As a 4D-STEM user, I want BF, ABF, ADF, HAADF, and custom
detector controls to update the virtual image interactively.

**Primary widgets**: Show4DSTEM.

**Data to use**: real 4D-STEM data with a visible central disk.

**Acceptance checks**:

- Switch detector presets and verify virtual image changes.
- Drag detector masks/rings and verify the virtual image updates without
  visible lag.
- Verify detector radius/angle labels are scientifically meaningful.
- Compare at least one detector result against a Python/reference computation
  for correctness-sensitive changes.

### S4D-04: Inspect Diffraction Details

**User story**: As a diffraction user, I want to pan, zoom, change contrast, and
inspect diffraction features without losing the linked scan context.

**Primary widgets**: Show4DSTEM.

**Data to use**: real diffraction stack with visible Bragg/disk features.

**Acceptance checks**:

- Pan and zoom the diffraction panel.
- Change diffraction contrast, colormap, log/linear scale, and smoothing.
- Hover diffraction features without selecting a new scan position or detector
  and verify diffraction readouts follow the hovered pixel while detector and
  virtual-image controls remain scoped to the explicitly selected state.
- Verify detector overlays remain aligned during pan/zoom/resize.
- Verify colorbar/histogram controls are readable on dark and light displays.

### S4D-05: Use WebGPU And Fallback Paths Correctly

**User story**: As a user on different hardware, I want the right compute path
for the surface I am using: MPS/raw Metal or CUDA/Torch for live Python-backed
work, WebGPU for browser/offline interaction, and a clear fallback when
acceleration is not available.

**Primary widgets**: Show4DSTEM.

**Data to use**: WebGPU-capable Mac/browser, MacBook MPS load path, CUDA/Torch
workstation path when available, plus a fallback browser or disabled WebGPU
environment when possible.

**Acceptance checks**:

- Record WebGPU adapter availability in the report.
- Record the GPU backend and loader path: CUDA, raw Metal/MPS, Torch-MPS, or
  browser WebGPU.
- Verify accelerated detector/virtual-image updates when WebGPU is available.
- Verify MacBook live-Jupyter browsing can use MPS/raw Metal loading and
  computation for first-pass review.
- Verify `backend="webgpu"` exported/offline pages use browser WebGPU and do not
  need Python, Torch, or MPS after export.
- Verify WebGPU unavailability produces a clear corrective error.
- Do not claim MPS/raw-Metal performance from a Torch-MPS path.

### S4D-06: Save, Export, And Reopen 4D-STEM Views

**User story**: As a notebook or sharing user, I want a compact two-panel saved
preview and shareable export that preserve the scientific context and make the
export precision obvious.

**Primary widgets**: Show4DSTEM.

**Data to use**: real or tutorial 4D-STEM notebook.

**Acceptance checks**:

- Press ``Cmd+S`` and reload/reopen the notebook.
- Verify the static two-panel fallback is visible.
- Open Export and verify the menu follows the same vocabulary as the other
  storyboards: ``HTML uint8`` for compact browse export, ``HTML full`` or
  ``HTML uint16`` for count-preserving export, with approximate size when known.
- Export compact uint8 HTML and reopen it.
- Export full/uint16 HTML or folder HTML when supported and reopen it.
- Drive scan position, diffraction pan/zoom, detector controls, and contrast in
  the exported page.
- Check lightweight save state for heavy-buffer leaks.
- Verify export status clears after completion/cancel in the same way as Show2D
  and Show3D.

### S4D-07: Use 4D-STEM On A Phone Or Narrow View

**User story**: As a user checking a 4D-STEM result on a phone or narrow screen,
I want virtual image, diffraction, and detector controls to remain reachable.

**Primary widgets**: Show4DSTEM.

**Data to use**: compact real or tutorial 4D-STEM export.

**Acceptance checks**:

- Test a narrow mobile viewport.
- Verify panels stack or resize intentionally and labels remain readable.
- Test touch-style scan-position movement, diffraction pan/zoom, detector
  control, and menu access.
- For iPhone-specific claims, serve the page to physical iPhone Safari.

### S4D-08: Export U8 And Full Data With Honest Reducers

**User story**: As a 4D-STEM user sharing data, I want compact U8 HTML for
quick browser inspection and full/count-preserving export when quantitative
detector counts matter, so collaborators know when a view is browse-quality and
when it can be used for quantitative checks.

**Primary widgets**: Show4DSTEM.

**Data to use**: real or real-derived 4D-STEM data with detector counts above
255 and a smaller count-limited control dataset where U8 should be nearly
lossless.

**Acceptance checks**:

- Export ``encoding="uint8"`` with ``downsample=1`` and with detector
  downsample values such as 2, 4, and 8; verify the UI and status text identify
  it as compact/browse U8 data.
- Verify U8 detector downsample uses the documented reducer, currently
  mean/average, so detector blocks do not immediately clip and wash out the
  bright-field disk.
- Export ``encoding="full"`` or ``uint16`` where supported and verify detector
  counts are preserved.
- When full/uint16 export is downsampled, verify the reducer is scientifically
  explicit. Prefer sum for count-preserving detector binning when the exported
  dtype can hold the result; use mean only when the goal is browse/display
  stability and label it that way.
- Compare at least one compact U8 export and one full/uint16 export against a
  Python reference for detector pixel values, virtual BF/ADF sums, and a custom
  mask.
- Confirm exported HTML opens without a Python kernel and that scan-position
  movement, detector masks, diffraction contrast, and virtual images remain
  interactive.

### S4D-09: Match The Shared Viewer GUI

**User story**: As a user moving between Show2D, Show3D, and Show4DSTEM, I want
the GUI layout and labels to feel consistent so I do not have to relearn export,
contrast, scale, reset, copy, and panel controls for each viewer.

**Primary widgets**: Show4DSTEM, with Show2D and Show3D as visual references.

**Data to use**: one Show4DSTEM export plus one Show2D and one Show3D reference
page using comparable colors, labels, scale bars, and top toolbar actions.

**Acceptance checks**:

- Compare top toolbar order, compact switch/menu styling, export button labels,
  reset/copy placement, histogram/color controls, scale bars, and status text
  against Show2D and Show3D.
- Verify the two-panel 4D-STEM layout keeps the virtual image and diffraction
  panel visually balanced on desktop, notebook, and narrow viewports.
- Verify labels and readouts use the same row/column convention and units as the
  other storyboards.
- Verify hover inspection follows the same rule as Show2D and Show3D: moving
  the pointer over an unselected virtual image, diffraction panel, detector
  mask, or ROI shows that target's readout without silently retargeting edit
  controls.
- Verify Export GUI choices match the Python API terms: ``mode``, ``encoding``,
  and ``downsample`` rather than older ambiguous names.
- Drive the same user path in live Jupyter and exported HTML; document any GUI
  difference that is intentional.

### S4D-10: Stress Export And WebGPU Reopen

**User story**: As a user sharing large 4D-STEM screening results, I want export
to finish in a practical time and the reopened artifact to stay responsive, so
large browser-shareable views do not become dead files.

**Primary widgets**: Show4DSTEM.

**Data to use**: the largest real or real-derived 4D-STEM dataset available on
the HPC/workstation backend for routine testing, plus a smaller deterministic
dataset for reference parity.

**Acceptance checks**:

- Measure export time, exported file/folder size, first paint after reopen, and
  WebGPU adapter availability.
- Reopen U8 HTML and full/folder HTML where supported; verify no Python kernel
  is required for the expected interactions.
- Drag scan position, detector ring, detector mask, diffraction pan/zoom, and
  contrast controls; record FPS or latency.
- Verify folder export clearly fails or explains what is missing if the
  companion data folder is moved.
- Add timings, reducer choice, dtype, downsample, browser, and backend host to
  the signoff report.

### S4D-11: Use MacBook MPS For Live Loading And U8 Export

**User story**: As a MacBook user opening large 4D-STEM data, I want first-pass
browsing to use the fast Apple Silicon path at native detector sampling when
memory allows, and to use detector-binned U8 only when I explicitly choose a
preview to avoid exhausting unified memory.

**Primary widgets**: Show4DSTEM.

**Data to use**: a real 4D-STEM master file on a MacBook or a MacBook-connected
Jupyter server, plus a smaller deterministic fixture for export parity.

**Acceptance checks**:

- Load first with ``load(path, backend="mps", det_bin=1)`` when the Mac memory
  budget allows, then construct ``Show4DSTEM`` from that result. Run
  ``det_bin=4`` or ``8`` only as an explicitly labeled preview/capacity check.
- Record load time, first paint, detector bin, dtype, resident memory, and
  whether the path is raw Metal/MPS, Torch-MPS, or CPU.
- Export full-detector WebGPU/HDF5 HTML when the gate is native detector
  behavior; export compact HTML with ``encoding="uint8"`` only as an explicitly
  labeled preview.
- Verify reopened HTML uses browser/WebGPU for interaction when available, not
  the Python MPS backend.
- Compare one virtual detector and one diffraction frame against a Python
  reference at the same binned/U8 precision, and separately document any
  expected clipping from U8 browse data.
- Repeat with ``encoding="full"`` or ``uint16`` when the data size allows, and
  verify count-preserving expectations separately from the U8 browse path.

### S4D-12: Explain Raw Metal MPS Versus Torch-MPS

**User story**: As a developer or power user debugging MacBook performance, I
want the report to say whether the viewer used raw Metal/MPS kernels or
Torch-MPS, because those paths have different memory behavior and performance
risks.

**Primary widgets**: Show4DSTEM.

**Data to use**: one MacBook MPS dataset large enough to expose memory pressure,
plus a tiny deterministic comparison dataset.

**Acceptance checks**:

- State why the selected path is raw Metal/MPS or Torch-MPS for the test.
- For the raw Metal/MPS path, verify loading and detector binning avoid
  materializing an unnecessary full CPU copy.
- For any Torch-MPS path, record tensor dtype, device, peak memory, and whether
  the operation falls back to CPU for unsupported kernels.
- Verify the same scientific operation is compared against a CPU/Python
  reference: detector bin, BF/ADF virtual image, diffraction frame, and ROI
  summed/mean diffraction when relevant.
- Document in the signoff whether the raw Metal path is used because it offers
  tighter control over chunking, dtype, and memory than generic Torch-MPS for
  this workflow.

### S4D-13: Keep GPU Memory Lifecycle Outside The Viewer UI

**User story**: As a user running many heavy 4D-STEM notebooks, I want GPU memory
to be released by backend/session lifecycle controls rather than by a
scientific viewer button, so the viewer stays focused on inspecting data and
does not hide ownership of GPU resources.

**Primary widgets**: Show4DSTEM, plus backend loader/session tooling.

**Data to use**: repeated open/close of a large MPS or CUDA-backed 4D-STEM
dataset.

**Acceptance checks**:

- Verify closing/deleting a widget view does not imply the backend data object
  or GPU allocation is freed unless the owning Python object/session is also
  released.
- Verify the documented cleanup path is backend/session level: delete or replace
  the loaded data object, clear references, stop/restart the kernel, or use the
  backend-specific cache cleanup utility when one exists.
- Verify notebook save/reopen does not persist GPU buffers or export payloads
  when ``save_state=False``.
- Verify exported HTML has no live Python GPU allocation and therefore does not
  need a "free GPU" control.
- If a future GUI exposes memory status, verify it reports backend ownership and
  links to cleanup instructions instead of pretending the viewer alone can free
  all GPU memory.

### S4D-14: Watch A Live 4D-STEM Acquisition Folder In Place

**User story**: As a microscopist collecting 4D-STEM, I want one already
mounted Jupyter Show4DSTEM to discover every newly completed acquisition,
expose it exactly once without rebuilding or silently changing precision, and
remain interactive while incomplete detector files finish writing.

**Primary widgets**: ``Show4DSTEM.from_folder(...)``. Test the CUDA
``Dataset5dstem`` path and the public ``backend="mps"`` path separately because
their paging and memory lifecycles differ. CPU is only a deterministic unit-test
reference. Standalone HTML is a snapshot and does not continue watching a
filesystem.

**Data to use**: A temporary watched folder and at least three genuine 4D-STEM
acquisition groups. Begin with one ready ``*_master.h5``. Introduce a second
master before one linked detector-data file exists, then complete it; add a
third compatible master atomically. Record master/chunk paths, scan and detector
shape, source dtype, requested ``det_bin`` and dtype, native bytes, backend
host, selected GPUs, and widget commit. Tiny generated HDF5 is a CI lifecycle
control only and does not establish real-workflow signoff.

**Acceptance checks**:

- Mount ``Show4DSTEM.from_folder(folder, watch=True, watch_interval=...,
  ...)`` before the later masters arrive. Capture the Python identity, widget
  model ID, browser container, Dataset/page, panel order, stars, hidden panels,
  detector ROI, scan cursor, zoom, and playback state.
- Keep one compact accessible watch badge near the folder/title area in stable
  DOM for both CUDA and MPS. Require green-dot ``Watching`` only while the
  actual watcher worker is alive. Enter ``Updating`` while discovery is active
  and keep it through real master/chunk validation and append. An idle poll may
  briefly show ``Updating`` but must return to ``Watching`` without decode,
  transfer, or repaint. If an arrival has a tile on the visible page, remain
  ``Updating`` until the new tile's current-generation, raw-backed virtual
  image is authoritatively painted; an older cached preview may remain visible
  while it refreshes but does not satisfy this transition. Use amber ``Waiting
  for file completion`` while a master/chunk is incomplete or not yet stable,
  red ``Watch error`` with
  corrective detail for a bad contract, paint failure, or worker failure, and
  gray ``Stopped`` or ``Not watching`` after stop/close or when liveness cannot
  be established. ``watch=False`` has no badge. A restored notebook model,
  static fallback, or standalone snapshot must never restore green ``Watching``
  without a live worker. Capture assertions and screenshots for every state;
  never infer state from color alone or leave a false green badge after exit.
- A missing, corrupt, unopenable, or still-changing linked data file must not
  append or load. Once the master and every required link are readable and
  stable, append it exactly once in deterministic acquisition order. A known
  master rewrite or removal must not duplicate or silently delete the active
  scientific view.
- Measure two visible stages separately: filesystem-ready to Dataset label,
  count, page control, or reserved placeholder paint; then selecting/requesting
  the new dataset to first virtual-image **and** diffraction paint. Do not call
  a Python trait update alone “append-to-paint.”
- On CUDA, append each master as a cold lazy ``Dataset5dstem`` slot. Do not
  eagerly load every arrival, clear unrelated reduced pages, exceed
  ``page_budget``, or silently change shape, dtype, detector bin, or scan bin.
  Recompute fit and placement safely after each append; cross-check the paging
  and generation rules in S4D-17 and S4D-18.
- On multi-GPU CUDA, record per-card budget and residency, prove every selected
  GPU receives work, serialize work within one device, and permit concurrent
  waves only across independent devices. Hidden panels remain released and are
  excluded from compare recompute.
- Repeat through ``Show4DSTEM.from_folder(..., backend="mps")`` on Apple
  Silicon. Identify the lazy MacBook/raw-Metal path explicitly, verify append
  ordering and memory, and report unsupported dtype or page-budget options as
  limitations rather than claiming CUDA ``Dataset5dstem`` behavior.
- Run a small synthetic CPU reference only in tests for lifecycle correctness;
  production must not route through it. On the same genuine source, verify a
  count-preserving full path such as ``det_bin=1, dtype="u16"`` and an explicit
  reduced preview path such as ``det_bin=4``; binned success is not proof of
  full-resolution support.
- After append, verify ``compare_dp_mode="selected"`` follows the clicked new
  dataset. If the average-DP mode is part of the change, also verify
  ``compare_dp_mode="average"`` matches a CPU reference over the current visible
  ready page, excluding hidden and incomplete datasets. Compare virtual images
  and diffraction patterns at two or more scan positions.
- Drive the live path in real JupyterLab through the in-app browser while files
  arrive. Capture before/after screenshots, console errors, Debug UI FPS and
  folder/page/cache/memory counters, detector drag, scan movement, diffraction
  pan/zoom, page flip/playback, and both latency stages.
- Verify ``stop_folder_watch()`` is idempotent and restartable. ``close()`` or
  ``free()`` must join watcher, page, preload, and cache workers; a file arriving
  after cleanup must not mutate the widget.

### S4D-15: Sign Off Real Heavy 4D-STEM Performance

**User story**: As a microscopist deciding whether Show4DSTEM is ready for a
real acquisition session, I want one report that proves the viewer loads a
large master quickly, chunks memory safely, appends new masters, and stays
interactive in the browser.

**Primary widgets**: Show4DSTEM with an NVIDIA/CUDA backend when available,
plus standalone exported HTML. Use the lazy MPS multi-dataset handle only for
MacBook fallback checks.

**Data to use**: local real ``*_master.h5`` files from a lab workstation or
HPC-backed acquisition folder. Do not commit these files or their generated
HTML reports to GitHub.

**Acceptance checks**:

- Run ``PYTHONPATH=src:. python scripts/widget_show4dstem_heavy_signoff.py
  --backend cuda`` with an explicit local real-data root or
  ``QUANTEM_WIDGET_4DSTEM_ROOTS``.
- Verify first-master load time, widget build time, backend shape, dtype, device,
  resident memory, and GPU memory before/after are in
  ``show4dstem-heavy-signoff-report.json``.
- Verify at least one additional ready master is measured through the current
  backend's append strategy: CUDA records eager stack-growth/reload timing;
  MPS records lazy live append timing.
- After multiple masters are loaded, drive the Dataset/frame slider end to end
  and record flip latency/FPS. A report that only measures first load does not
  prove the real browsing workflow.
- Verify standalone HTML export records explicit ``uint8``/``uint16`` and
  detector-bin settings, output size, and export time.
- Drive the exported HTML in Chromium and verify WebGPU/browser information,
  Dataset/frame flip FPS, virtual-detector drag FPS, scan-position drag FPS,
  recompute latency, and wheel-zoom FPS are recorded.
- Treat ``--skip-browser`` as backend/export debugging only, not performance
  signoff.
- For NVIDIA no-bin stress, run a separate capacity probe with 30-40 ready
  masters and ``--det-bin 1``. Passing means either the data fit and browser
  flip-around is measured, or the report fails clearly with the maximum loaded
  master count, allocation error, and GPU cleanup evidence. Do not call a
  30-40 file no-bin workflow supported just because a smaller stack is smooth.

### S4D-16: Screen Many 4D-STEM Datasets In Multiple Mode

**User story**: As a microscopist reviewing a session with many related
4D-STEM acquisitions, I want Show4DSTEM to show many virtual images at once
while sharing the diffraction ROI and scan cursor, so I can quickly decide
which datasets are useful, hide bad ones, star good ones, and preserve that
curation for later notebook cells or shared HTML.

**Primary widgets**: Show4DSTEM in ``view_mode="multiple"`` with 5D data or a
lazy multi-dataset handle.

**Data to use**: 8-14 binned real or real-derived 4D-STEM datasets for routine
browser smoke; 30-40 ready masters on CUDA or MPS for heavy signoff when the
backend and memory budget allow it.

**Acceptance checks**:

- Construct ``Show4DSTEM(..., view_mode="multiple", compare_cols=...)`` from
  multiple datasets and verify the multiple grid renders all ready panels without
  materializing an unsafe full stack on MPS.
- Verify desktop ``compare_cols`` is a maximum column count and the phone or
  narrow viewport caps the grid at two columns with readable tiles.
- Verify ``compare_panel_gap_px=0`` removes horizontal and vertical gutters
  between multiple panels, and nonzero values intentionally restore spacing.
- Scroll over multiple tiles and the single-panel diffraction/virtual-image
  canvases; verify wheel input zooms the image instead of scrolling the page,
  and zoom-out behaves symmetrically.
- Toggle ``compare_dp_mode`` between ``"average"`` and ``"selected"``; verify
  the diffraction panel either averages visible multiple panels or follows the
  clicked dataset.
- Star at least one useful dataset and hide at least one rejected dataset from
  the GUI; verify the visible panel count, labels, and selected dataset remain
  coherent.
- Reorder panels by drag or click-then-click reorder mode; verify dynamic order
  changes before/after release and the saved order is still visible after
  reset/show-all actions.
- Round-trip ``compare_panel_order``, ``compare_hidden_panels``,
  ``compare_starred_panels``, and ``compare_dp_mode`` through
  ``state_dict()`` / ``load_state_dict()`` and through exported HTML.
- Drive the same workflow in a physical phone browser when making iPhone/Safari
  claims; Chromium mobile emulation is only a pre-check.
- Record dataset count, ready count, backend, dtype, detector bin, grid column
  count on desktop/mobile, FPS, and whether the artifact is live Jupyter or
  standalone HTML.

### S4D-17: Page A Folder Safely On One CUDA GPU

**User story**: As a scientist with one CUDA GPU, I want to open a folder whose
complete 4D-STEM series exceeds usable VRAM, see the first useful page quickly,
and browse every dataset at the requested resolution without calculating a
manual memory limit or encountering a raw CUDA failure.

**Primary widgets**: ``Show4DSTEM.from_folder(...)`` in multiple mode.

**Data to use**: enough compatible real masters to exceed one selected GPU's
safe raw-residency budget after the requested detector bin and dtype. Include a
deterministic small fixture with a forced two-frame budget for CI.

**Acceptance checks**:

- Open with ``gpus=[0]`` and ``page_budget="auto"``. Verify discovery and the
  first useful page succeed whenever one processed master fits.
- Keep the visible page size independent from the raw residency window. A page
  may contain more panels than fit as raw tensors; the backend must process it
  in bounded waves without silently changing dtype, detector bin, or shape.
- Leave decoder/reduction headroom equal to at least one largest processed
  master plus bounded workspace before declaring the complete series resident.
- Verify foreground page loading cancels or safely waits for full-series preload
  and cache warming. Concurrent decoders must not touch the same CUDA device.
- Drive page 1, page 2, the last page, and page 1 again. Verify raw residency
  remains bounded, evicted folder masters reload correctly, and a warmed reduced
  page returns without reloading raw data.
- Record click-to-first-panel, click-to-complete-page, warm-return latency,
  resident bytes, evictions, reloads, cache hits, and GPU memory before/after.
- Treat CUDA illegal-address, host-register, OOM, stale-panel, or stuck worker
  errors as failures. Verify ``close()`` leaves no page/preload/cache worker.

### S4D-18: Pool Multiple GPUs And Stream Pages Progressively

**User story**: As a scientist with multiple CUDA GPUs, I want every selected
GPU to contribute its safe capacity and compute throughput while folders larger
than the combined working set remain paged. When I flip pages, stable panel
slots should appear immediately and fill progressively as datasets become
ready, rather than waiting for the slowest panel before showing anything.

**Primary widgets**: ``Show4DSTEM.from_folder(...)`` in multiple mode with
``gpus=[...]`` or ``gpus="all"``.

**Data to use**: real compatible masters on two or more CUDA GPUs, including
equal cards, intentionally unequal free-memory budgets, and—when available—a
folder distributed across independent physical disks. Use deterministic fake
budgets and delayed loaders for CI cancellation/progress tests.

**Acceptance checks**:

- Use only the explicitly selected process-visible GPUs. Compute a safe budget
  for each card and place cold masters according to available capacity; spare
  memory on a larger/freer card must not be stranded by fixed round-robin
  assignment. Every report must record ``CUDA_VISIBLE_DEVICES`` and map each
  logical index to the physical GPU UUID, PCI bus ID, model name, total memory,
  and free memory at the start of the run.
- Load one independent master allocation per participating GPU in each
  progressive wave. Different GPUs may load/compute concurrently, but work on
  one GPU remains serialized and independently reclaimable by LRU eviction.
- Keep the page grid stable from the click: reserve every expected slot, show a
  quiet loading state, and replace each slot as its float32 virtual image arrives
  without resizing or remounting the rest of the grid.
- Give every page request a generation. Rapid page 1 -> 2 -> 3 changes cancel
  obsolete work after its current safe wave, and late page-1/page-2 results must
  never overwrite page 3.
- Prioritize the visible page and selected diffraction source, then prefetch the
  next and previous pages for the current detector preset. Full-series preload
  and other detector-preset warming run only when foreground work is idle.
- Preserve reduced virtual-image pages in the bounded host cache independently
  of raw GPU residency. Verify a warm return does not require the old raw page
  to remain on a GPU.
- Record placeholder acknowledgement, first-panel, half-page, complete-page,
  and warm-return latency; per-GPU budget/resident bytes; cache hits/misses;
  evictions/reloads; stale-result drops; and browser paint/FPS evidence.
- Compare one-GPU and multi-GPU cold-page timing with the same data. Verify all
  selected GPUs receive work and that adding a useful GPU does not reduce safe
  capacity or make first-panel latency worse without an explained I/O limit.
- Repeat after a live folder append and after external memory pressure changes.
  Verify placement is recalculated safely and existing warmed pages remain valid.

### S4D-19: Reopen A Folder With Persistent Scientific Previews

**User story**: As a scientist returning to a large 4D-STEM folder, I want the
BF/ABF/ADF/HAADF images I already computed to appear immediately while the
authoritative raw data loads in the background. I need the viewer to say when I
am seeing a cached preview and when fresh raw-backed interaction is ready, so a
second open is useful instead of showing black panels for another cold decode.

**Primary widgets**: ``Show4DSTEM.from_folder(...)`` in multiple mode, first on
one NVIDIA CUDA GPU and then on multiple selected CUDA GPUs. The CPU path is a
lifecycle control; MPS support is a follow-up and must not be inferred from the
CUDA signoff.

**Data to use**: the same real 82-or-more-master folder used for cold progressive
paging, with linked detector chunks where present. Retain enough standard
virtual images to exceed one visible page, reopen from a new widget instance,
then change, replace, remove, and append individual masters/chunks. Use a tiny
multi-master fixture for deterministic CI hit, miss, corruption, eviction, and
cancellation cases.

**Acceptance checks**:

- Expose ``preview_cache="auto"``, ``preview_cache_dir=None``,
  ``preview_cache_max_bytes=4 << 30``, and
  ``rebuild_preview_cache=False`` on ``Show4DSTEM.from_folder(...)``. ``False``
  disables persistent reads and writes; the automatic mode uses the user cache,
  and an explicit directory supports a chosen local SSD. A shared or network
  filesystem is unverified until its atomic-rename and multi-process
  writer/rebuild/clear behavior pass explicitly; process-local locking alone is
  not a shared-cache guarantee. Keep this disk budget independent from
  ``compare_cache_max_bytes`` host memory and ``page_budget`` raw CUDA residency.
- Persist only reduced float32 BF, ABF, ADF, and HAADF virtual images. Never
  persist raw 4D detector tensors, diffraction patterns, arbitrary detector
  masks, CUDA allocations, credentials, or private source data outside the
  configured cache directory. ``warm_cache=True`` may fill the standard
  presets proactively; normal use writes a standard preset after computing it.
- Store previews per master, not per display page. Hiding, starring, reordering,
  changing page size, or moving a master to another page must reuse that
  master's valid preview without duplicating it.
- Validate each entry against a versioned processing key and the current source
  fingerprint: canonical master identity; master size, nanosecond mtime/ctime,
  device, and inode; the ordered identities, sizes, nanosecond mtimes/ctimes,
  devices, and inodes of every linked detector chunk; processed scan/detector
  shape; source/load dtype; detector bin; scan override; center and preset
  radii/mask geometry; and the cache schema/compute version. A change to one
  master or chunk invalidates only that master's presets. An unchanged append
  must not invalidate existing entries.
- A missing, unreadable, incomplete, or changing required chunk is not a valid
  cache hit. A corrupt, truncated, incompatible, or partially written cache
  artifact becomes a counted miss and is rebuilt safely; it must not break
  folder discovery or paint unverified pixels.
- Publish cache files atomically and make concurrent readers safe. Enforce
  ``preview_cache_max_bytes`` with deterministic whole-entry eviction while no
  writer can leave a manifest pointing at an incomplete payload. Cache lookup
  must not decode raw 4D data or allocate CUDA memory.
- On a matching second open, reserve the normal stable grid slots and paint each
  cached panel as soon as it is read. Show a quiet, explicit state such as
  ``Cached preview · loading raw data``; never label cached pixels ``Fresh`` or
  show an empty black panel where a valid preview is available.
- Keep startup accounting honest: this CUDA-first phase still validates and
  loads one raw master synchronously to establish detector shape, calibration,
  and the selected diffraction pattern. Report API-call-to-model-ready
  separately from model-ready-to-cached-paint, and do not claim a cache-only
  mount. A future metadata/calibration bootstrap may remove that final raw
  dependency without weakening provenance checks.
- Continue raw loading and reduction through the normal capacity-aware CUDA
  scheduler. Replace the cached pixels in the same panel slot when the current
  generation's fresh result arrives, without remounting the grid, changing
  contrast unexpectedly, or flashing black. Once raw data is ready, detector
  changes and diffraction inspection use the authoritative requested dtype and
  resolution.
- When a persistent preview is shown for a newly arrived master on the visible
  page, keep the folder-watch badge at ``Updating`` while the cached pixels stay
  useful. Return to ``Watching`` only after the corresponding current-generation
  raw-backed tile has reached authoritative browser paint; otherwise transition
  to the truthful amber waiting or red corrective-error state.
- Support partial hits. Paint cached panels first, show honest per-page progress,
  and schedule raw work only as needed for misses and authoritative refresh.
  Rapid page 1 -> 2 -> 3 changes must cancel obsolete refresh work after a safe
  wave; late cached or fresh results must never overwrite page 3.
- If refresh fails after a valid preview painted, keep the preview visible and
  mark it ``Cached preview · refresh failed`` with a corrective error. Do not
  silently relabel stale pixels as fresh, and do not discard a useful preview
  merely to return to a black placeholder.
- Expose a read-only ``preview_cache_info`` property with enabled state, path,
  byte limit/current bytes, entry count, hits, misses, invalidations, evictions,
  and errors. ``clear_preview_cache()`` deletes this widget's persistent preview
  namespace and resets its accounting without clearing ShowFolder thumbnails or
  pretending to free raw CUDA memory. ``rebuild_preview_cache=True`` ignores
  old entries for the new run and repopulates them safely.
- Measure browser paint, not only Python traits. Record click-to-cached-first
  panel, click-to-cached-visible-page, click-to-fresh-first panel,
  click-to-fresh-visible-page, click-to-complete-page, and neighbor-prefetch
  completion separately, plus cache lookup/read/write bytes and time, hit/miss
  counts, raw decode/reduction time, per-GPU residency, FPS, and console errors.
  The browser probe exposes the receipt and after-paint proxy fields under
  ``window.__quantemShow4DSTEMPerf.comparePage`` as
  ``firstCachedPanelReceiptAtMs``, ``firstFreshPanelReceiptAtMs``,
  ``firstCachedPanelPaintAtMs``, ``firstFreshPanelPaintAtMs``,
  ``cachedVisiblePaintAtMs``, and ``freshVisiblePaintAtMs``; keep receipt and
  double-animation-frame after-paint proxy evidence labeled separately.
- Compare cold first open, matching second open, partial-hit reopen, forced
  rebuild, and disabled-cache runs on one selected NVIDIA GPU. Cached-first
  paint must be materially faster than the approximately one-second progressive
  cold first panel. Keep the measured approximately 11.27-second visible-page
  completion separate from the approximately 22.91-second worker/neighbor-
  prefetch-idle time; cached previews must expose useful panels without waiting
  for either. On the reference host, over five fresh-widget page opens, require
  median
  click-to-cached-first <= 500 ms and <= 50% of the matched cold median, plus
  median cached-visible-page <= 2 s and <= 25% of matched cold visible-page
  time. Report p95 as evidence rather than hiding a slow outlier.
- Define storage conditions for every timing run. Distinguish a fresh
  Python/widget process from an OS/filesystem-page-cache-cold run, and record
  source/cache filesystem and locality, storage device class, HDF5 compression,
  linked-chunk count, bytes read, and achieved throughput. Never attribute an
  I/O-limited result to GPU scaling without that evidence.
- Persistent lookup must not serialize the raw refresh. On the same host and
  source, median click-to-fresh-visible-page and complete-page time may regress
  by at most 10% versus ``preview_cache=False``; otherwise record the cache I/O
  contention as a failed performance gate. Neighbor prefetch must start only
  after the visible foreground request reaches its ready state.
- Repeat on two or more selected NVIDIA GPUs. Persistent hits must remain
  backend-independent, while misses and refreshes use all eligible cards under
  S4D-18's per-device serialization and generation rules. Adding a GPU must not
  duplicate disk entries, corrupt the cache, exceed either memory budget, or
  make cached-first paint wait for the slowest raw wave.
- Verify ``close()`` joins cache readers/writers and CUDA page workers. Reopen in
  a fresh Python process to prove persistence, then clear the cache and prove the
  next open is a true miss. Leave cache artifacts and real-data reports outside
  git unless deliberately promoted into a maintainer fixture or runbook.

### S4D-20: Prove Folder Paging And Cache Behavior Overnight

**User story**: As a scientist leaving a large acquisition folder open
overnight, I want automatic paging, cached previews, live arrivals, and fresh
raw-backed replacement to remain truthful and responsive on either one or two
NVIDIA GPUs, so an ended runner or a green badge cannot hide a stalled,
memory-leaking, or stale scientific view.

**Primary widgets**: ``Show4DSTEM.from_folder(...)`` in a real JupyterLab
session and its browser frontend. This story is the endurance composition of
S4D-14, S4D-17, S4D-18, and S4D-19; those stories remain the canonical source
for watch-state, scheduler, generation, and cache correctness rather than being
repeated here.

**Data to use**: One compatible real acquisition series large enough to exceed
the safe raw-residency budget of one selected GPU, with linked detector chunks
when present. Use a staged watched-folder view of the real files for arrival
tests so the source acquisition is never rewritten. Add a separate real
full-detector control using ``det_bin=1`` and a count-preserving dtype, with at
least seven masters and enough masters to exceed the selected raw budget when
the source permits. Synthetic data is a CI control only.

**Acceptance checks**:

- Run an explicit, serial matrix so the topologies do not contend: one selected
  physical NVIDIA GPU, then the same workflow on two selected physical NVIDIA
  GPUs. Give each topology at least four clock hours and 100 completed canonical
  navigation cycles, for at least eight hours total. A canonical cycle requests
  page 1, page 2, the last page, and page 1 again; performs a rapid page 1 -> 2
  -> 3 cancellation check; and exercises selected/average diffraction, hide,
  star, scan movement, detector movement, diffraction zoom, and pan. A partial
  or restarted cycle does not count as completed.
- Hold source, ready-master set, page size, ``page_budget="auto"``, detector
  bin, dtype, detector preset, cache budget, and page sequence constant for the
  matched one-GPU/two-GPU comparison. Run cold cache-disabled, cache-populating,
  and matching cache-enabled phases in fresh Python processes. Reopen the
  matching cache at least five times per topology, and include the partial-hit,
  changed-master/chunk, corrupt-entry, forced-rebuild, disabled-cache, and clear
  cases from S4D-19 without corrupting source data or the canonical cache copy.
- During both topologies, introduce at least one real master with a required
  chunk withheld, then make the complete acquisition visible atomically. Prove
  it remains waiting while incomplete, appears exactly once when ready, keeps a
  valid cached preview visible when one exists, and reaches green ``Watching``
  only after a current-generation fresh tile receives the browser paint
  acknowledgement required by S4D-14 and S4D-19. A Python trait publication,
  cached paint, or backend worker completion is not fresh-paint proof.
- Exercise the viewer in actual JupyterLab through a controlled browser for each
  topology, not only through Python traits or an exported snapshot. Capture the
  stable loading slots, cached-first paint, fresh replacement in the same slot,
  every watch-badge state, page cancellation, and representative scientific
  interactions. Preserve timestamped screenshots, browser console output, model
  and kernel errors, and the receipt-versus-after-paint fields from
  ``window.__quantemShow4DSTEMPerf.comparePage``.
- Record a topology and provenance snapshot before every process: host, UTC
  start time, widget commit and dirty-diff identity, Python/Torch/CUDA/driver
  versions, ``CUDA_VISIBLE_DEVICES``, and each logical index's physical UUID,
  PCI bus ID, model, total memory, and free memory. Record source/cache canonical
  paths, filesystem and storage device, locality, compression, linked-chunk
  count, source fingerprint, ready-master count, shapes, dtype, detector bin,
  and cache schema/compute version. Repeat GPU memory and filesystem snapshots
  after each phase and at cleanup.
- Apply the cache latency and no-more-than-10-percent fresh-refresh regression
  gates in S4D-19 to each topology, including median and p95 over the required
  fresh-process reopens. For the matched GPU comparison, require two-GPU median
  fresh-first latency to be no more than 110% of one-GPU latency and median
  fresh-visible and complete-page latency to be no slower than one GPU. Prove
  both GPUs receive work. If storage/decode saturation prevents scaling, retain
  the measurements and mark the scaling gate limited or failed; do not convert
  an explanation into a pass.
- Divide each topology's completed endurance cycles into first and last
  quartiles. Require no more than 20% regression in p95 fresh-visible and
  complete-page latency, no monotonic resident host/GPU/cache growth after
  warm-up, zero stale-generation paints, zero unhandled browser/kernel errors,
  zero CUDA illegal-address/OOM errors, and zero stuck page, preload, watch, or
  cache workers. Report cache hits/misses/invalidations/evictions, raw
  evictions/reloads, stale drops, bytes read/written, throughput, per-GPU
  residency, and memory high-water marks even when the gate fails.
- Write an atomic run manifest after every phase and completed cycle. Include a
  monotonically advancing checkpoint, active topology/phase, cache namespace,
  owned process IDs, last successful page generation, and artifact paths. Emit
  a timestamped heartbeat at least every five minutes with progress, worker
  liveness, GPU memory/utilization, and the most recent error. A watchdog treats
  three missed heartbeats or 15 minutes without forward progress as a failure,
  captures stacks/logs/GPU state, stops only owned processes, and resumes from
  the last complete checkpoint in a fresh process.
- Prove resume behavior once with a controlled child-process interruption. The
  resumed run must preserve valid cache entries, avoid duplicate live arrivals
  and stale page paint, and distinguish pre-interruption, resumed, and fully
  continuous time in the report. Automatic restart attempts are bounded and
  visible; exhausting them fails the run instead of leaving it indefinitely
  ``running`` or declaring success because the launcher exited.
- Run the separate real no-bin leg on one GPU and then two GPUs with automatic
  paging and the same provenance/cleanup capture. At minimum, browse first,
  second, last, and warm-return pages and compare one virtual image and one
  diffraction result with a CPU reference. If one processed master cannot fit,
  or the available series cannot exceed the selected raw budget, record that
  exact capacity boundary as an unmet gate rather than inferring support from a
  detector-binned run.
- On normal completion, interruption, and failure, call the public cleanup path
  and prove the watcher, page, preload, cache, notebook, browser, tunnel, and
  watchdog processes owned by the run are gone. Record final GPU memory against
  baseline and preserve corrective evidence for any residual allocation. Never
  delete the real source; clear or corrupt only a run-owned cache/staging copy.
- Produce one durable top-level ``index.html`` plus machine-readable JSON
  summary, atomic manifest, phase/cycle timing table, GPU samples, source/cache
  provenance, browser screenshots and console log, kernel/worker logs, cache
  inventory, parity results, failure/restart timeline, and exact commands. Keep
  artifacts outside git, publish the exact report path and review URL, and mark
  every gate pass, fail, limited, skipped, or unavailable. A run is complete
  only when the report is readable and all required artifacts are present; an
  old report, an ended task, or a missing heartbeat is not current signoff.
