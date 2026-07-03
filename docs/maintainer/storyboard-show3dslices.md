# Show3DSlices Storyboard

Use with [Storyboard](storyboard).

## Stories

### S3S-01: Open A Multislice Or Volume Dataset Quickly

**User story**: As a microscopist opening multislice, ptychography, or volume
data, I want a useful orthogonal-slice preview in about a second for normal
working sizes so I can orient myself before fine inspection.

**Primary widgets**: Show3DSlices.

**Data to use**: real or real-derived multislice microscopy volume and a larger
stress volume.

**Acceptance checks**:

- Load the volume from Jupyter and exported HTML when supported.
- Measure first visible paint, volume shape, dtype, native bytes, and display
  downsampling if any.
- Verify XY/XZ/YZ or equivalent slice panels render with labels and scale bars.
- Verify the widget remains usable after first paint without blocking the
  kernel for frontend-only interactions.

### S3S-02: Navigate Orthogonal Slices

**User story**: As a user exploring a 3D volume, I want slice sliders and
crosshair/position markers to stay synchronized across orthogonal views.

**Primary widgets**: Show3DSlices.

**Data to use**: volume with recognizable features across depth.

**Acceptance checks**:

- Scrub each slice axis slowly and quickly.
- Click in each slice panel and verify the other views jump to the correct
  corresponding position.
- Verify labels, crosshairs, and position readouts update together.
- Record FPS for slider scrub and click-to-position updates.

### S3S-03: Inspect Anisotropic Multislice Data

**User story**: As a multislice user, I want depth-axis stretch and scale bars
to make anisotropic voxels readable without changing the underlying data.

**Primary widgets**: Show3DSlices.

**Data to use**: multislice or volume data where Z sampling differs from XY.

**Acceptance checks**:

- Change depth stretch or equivalent display scaling.
- Verify scale bars and labels remain scientifically meaningful.
- Verify changing display stretch does not alter exported raw data semantics.
- Resize panels and verify aspect/position stays stable.

### S3S-04: Compare Contrast And FFT Across Slices

**User story**: As a user checking periodicity or artifacts through depth, I
want contrast, colormap, smoothing, and FFT controls to work consistently across
slices.

**Primary widgets**: Show3DSlices.

**Data to use**: volume with visible periodic structure or simulated lattice
features.

**Acceptance checks**:

- Change colormap, contrast, auto/manual range, smoothing, and log scale.
- Toggle FFT/log FFT when supported and verify FFT panels align with slices.
- Compare a suspicious FFT slice against NumPy or a known reference.
- Verify histogram and FFT interactions stay responsive.

### S3S-05: Save, Export, And Reopen Volume Views

**User story**: As a notebook or sharing user, I want saved notebooks and HTML
exports to reopen with visible slice views and without huge hidden buffers when
lightweight save is requested.

**Primary widgets**: Show3DSlices.

**Data to use**: real or real-derived volume in a Jupyter notebook.

**Acceptance checks**:

- Press ``Cmd+S`` in JupyterLab and reload/reopen the notebook.
- Verify the static fallback remains visible and compact.
- Export HTML where supported and reopen it.
- Verify slice sliders, contrast, FFT, and reset controls still work.
- Check saved widget state for heavy-buffer leaks when ``save_state=False``.

### S3S-06: Use Slices On A Phone Or Narrow View

**User story**: As a user checking a volume on a phone or narrow screen, I want
slice panels and controls to remain reachable and readable.

**Primary widgets**: Show3DSlices.

**Data to use**: representative volume with multiple slice panels visible.

**Acceptance checks**:

- Test a narrow mobile viewport.
- Verify panels stack or wrap intentionally and labels do not overlap.
- Test touch-style slider scrub, panel click, pan/zoom, and menu controls.
- For iPhone-specific claims, serve the page to physical iPhone Safari.
