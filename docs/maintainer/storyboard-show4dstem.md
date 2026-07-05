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
- Record Python backend and data loader path: CUDA/Torch, raw Metal/MPS,
  Torch-MPS, CPU, or browser/WebGPU.
- Verify accelerated detector/virtual-image updates when WebGPU is available.
- Verify MacBook live-Jupyter browsing can use MPS/raw Metal loading and
  computation for first-pass review.
- Verify backend="web" exported/offline pages use browser WebGPU and do not
  need Python, Torch, or MPS after export.
- Verify fallback path is usable and clearly communicated when WebGPU is not
  available.
- Do not claim WebGPU performance from CPU fallback.
- Do not claim MPS/raw-Metal performance from a Torch-MPS or CPU path.

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
browsing to use the fast Apple Silicon path, usually detector-binned U8, so the
viewer opens quickly without exhausting unified memory.

**Primary widgets**: Show4DSTEM.

**Data to use**: a real 4D-STEM master file on a MacBook or a MacBook-connected
Jupyter server, plus a smaller deterministic fixture for export parity.

**Acceptance checks**:

- Load with ``load(path, backend="mps", det_bin=4 or 8, dtype="u8")`` and
  construct ``Show4DSTEM`` from that result.
- Record load time, first paint, detector bin, dtype, resident memory, and
  whether the path is raw Metal/MPS, Torch-MPS, or CPU.
- Export compact HTML with ``encoding="uint8"`` and reopen it in the browser.
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

### S4D-14: Append Live Scope Acquisitions

**User story**: As a microscope user collecting a session of 4D-STEM scans, I
want newly completed ``*_master.h5`` files to appear in the same Show4DSTEM
viewer so I can process data in real time without rebuilding the notebook.

**Primary widgets**: Show4DSTEM with the lazy MPS multi-dataset handle.

**Data to use**: a live or simulated acquisition folder where masters appear
over time, including at least one partial/incomplete master that should be
ignored until ready.

**Acceptance checks**:

- Start with ``load_macbook_datasets(folder, det_bin=4, scan_size=...)`` and
  mount ``Show4DSTEM(live)`` while only the first dataset is ready.
- Call ``live.watch_master_folder(folder, interval=...)`` and verify newly
  completed masters append into the existing Dataset slider.
- Verify partial masters are skipped until ``is_master_ready`` confirms linked
  data files exist.
- Verify repeated polls do not duplicate already loaded masters.
- Drive detector drag, scan-position movement, diffraction pan/zoom, FFT, and
  contrast after an append to confirm the active viewer remains real time.
- Record load/append timing, backend path, detector bin, dtype, scan size, and
  GPU memory behavior in the signoff report.
