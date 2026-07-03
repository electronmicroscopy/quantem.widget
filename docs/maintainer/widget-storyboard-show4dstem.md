# Show4DSTEM Storyboard

Use with [Widget Storyboard](widget-storyboard).

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

**User story**: As a user on different hardware, I want WebGPU acceleration
when available and a clear fallback when it is not.

**Primary widgets**: Show4DSTEM.

**Data to use**: WebGPU-capable Mac/browser plus a fallback browser or disabled
WebGPU environment when possible.

**Acceptance checks**:

- Record WebGPU adapter availability in the report.
- Verify accelerated detector/virtual-image updates when WebGPU is available.
- Verify fallback path is usable and clearly communicated when WebGPU is not
  available.
- Do not claim WebGPU performance from CPU fallback.

### S4D-06: Save, Export, And Reopen 4D-STEM Views

**User story**: As a notebook or sharing user, I want a compact two-panel saved
preview and shareable export that preserve the scientific context.

**Primary widgets**: Show4DSTEM.

**Data to use**: real or tutorial 4D-STEM notebook.

**Acceptance checks**:

- Press ``Cmd+S`` and reload/reopen the notebook.
- Verify the static two-panel fallback is visible.
- Export HTML where supported and reopen it.
- Drive scan position, diffraction pan/zoom, detector controls, and contrast in
  the exported page.
- Check lightweight save state for heavy-buffer leaks.

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
