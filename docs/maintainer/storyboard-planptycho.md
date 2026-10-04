# PlanPtycho Storyboard

Use with [Storyboard](storyboard).

## Stories

### PP-01: See Whether A Thick Crystal Fits The Settings

**User story**: As a microscopist planning multislice ptychography of a thick crystal, I want to give the crystal and my
usual settings and see at a glance what fits and what to change, before I spend microscope time.

**Primary widgets**: PlanPtycho.

**Data to use**: SrTiO3 built in the tutorial (cubic, a = 3.905 A), [001], 110 nm, focus 17 nm, 48 x 48 scan at 0.495 A.

**Acceptance checks**:

- First paint shows four panels (Top, Side, Probe, Detector), the controls and twelve check rows.
- Untilted, focus 17 nm: window and margin `caution`; overlap, reach, lean and focus split `pass`; each caution carries its evidence note.
- `plan.report()` in Python gives the same grades as the widget.

### PP-02: Drag The View Depth Through The Specimen

**User story**: As a microscopist, I want to drag down the side view and watch the beam widen, so I can see where it
outgrows the reconstruction window.

**Acceptance checks**:

- Dragging in Side moves the white depth line; the Top overlay depth and beam width update on every move.
- Near the exit surface the yellow beam circle in Top is larger than the dashed window square, and the Probe panel shows
  the wrapped beam with a red `wraps` note.
- Each update lands within one display frame (measure with two chained `requestAnimationFrame` calls: 33 ms at 60 Hz).

### PP-03: Move The Probe To The Scan Edge

**User story**: As a microscopist, I want to put the probe at the edge of the scan and see how much of its beam leaves
the scanned field deep in the specimen.

**Acceptance checks**:

- Dragging in Top moves the probe; the Side cone follows the same position.
- The probe starts at the scan centre; `Reset Probe` returns it there.

### PP-04: Change Camera, Camera Length And Binning

**User story**: As a microscopist, I want to switch between my cameras and camera lengths and see the model window and
checks change.

**Acceptance checks**:

- Camera `Arina` / `EMPAD` sets 192 / 128 px; Length 91 / 115 / 185 mm sets 0.554 / 0.461 / 0.269 mrad per Arina pixel.
- Binning 2x gives 96 px at twice the mrad per pixel (same edge angle, half the model window); Sampling is a read-out, labelled `custom calibration` only when Python overrides it.
- A 256 px scan grades margin `pass`; Camera `custom` shows editable Pixels and Sampling.

### PP-05: Change The Zone Axis

**User story**: As a microscopist, I want to switch the zone axis and see the projection and Bragg disks for it.

**Acceptance checks** (needs a live kernel):

- Zone `[011]` rebuilds the Top projection with a 3.905 x 5.523 A cell and new Bragg disks within about a second.

### PP-06: Tilt The Specimen

**User story**: As a microscopist whose crystal is a few mrad off the zone axis, I want to see how far the columns lean
through the thickness and whether that matters at my object pixel.

**Acceptance checks**:

- Tilt row 3.2, Tilt col -2.4 mrad at 110 nm: Top streaks along the tilt, Side columns lean, the Detector shows the zone
  axis (red cross) and HOLZ ring moved off the beam axis, and the lean check reads `caution`, 4.4 A at 4.0 mrad.

### PP-07: Presets And Recommended Settings

**User story**: As a microscopist, I want to pick my usual microscope setup and a thickness and get settings that pass.

**Acceptance checks**:

- The microscope menu sets voltage, semiangle, camera and camera length together (185 mm: 0.269 mrad, reach `caution`
  at 21.4 mrad).
- `Recommended 40 nm` sets focus 20 nm and a 128 px scan; with no tilt every graded check passes. `Recommended 200 nm`
  leaves only the window at `caution` on the Arina at 91 mm.

### PP-08: Move The Focus Through A Stationary Specimen, Zoom

**User story**: As a microscopist, I want to drag the focus through the specimen and look closely at any panel.

**Acceptance checks**:

- Dragging the dashed line in Side moves the focus (the slab does not move); the labels and the `Focus splits the
  specimen` check show the thickness above and below it, and a focus outside the slab gives `caution`.
- The wheel zooms each panel about the cursor without scrolling the page; the zoom factor shows by the scale bar;
  double-click resets.
- The `Sample` switch on Probe draws the crystal at the view depth under the probe.

### PP-09: Check A Collaborator's Acquisition

**User story**: As a microscopist sent someone else's acquisition settings, I want to type them in and see whether the
data can work for ptychography.

**Acceptance checks**:

- `PlanPtycho(..., voltage_kV=200, semiangle_mrad=24.5, detector_px=128, detector_mrad_per_px=0.9, scan_step_A=0.4,
  scan_size_px=256, c10_nm=-15)` shows Camera `custom` and every graded check `pass` at 40 nm.
- Clicking a number (thickness, C10, semiangle, pixels, sampling) and typing a value updates the panels and checks on Enter.
