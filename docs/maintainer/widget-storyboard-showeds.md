# ShowEDS Storyboard

Use with [Widget Storyboard](widget-storyboard).

## Stories

### EDS-01: Open A Real Spectrum Image Quickly

**User story**: As a microscopy user opening EDS/EELS spectrum-image data, I
want the spatial map and spectrum to become useful in about a second for normal
preview sizes so I can start selecting bands and ROIs.

**Primary widgets**: ShowEDS.

**Data to use**: real EDS/EELS spectrum-image data from MJ-goat or buffle; use
streaming sparse/indexed data for large cases.

**Acceptance checks**:

- Load through Jupyter with the intended backend path.
- Measure first visible paint, spatial shape, channel count, dtype, native
  bytes, sparse/streaming mode, and initial map/spectrum timing.
- Verify default log display is appropriate and readable.
- Verify frontend interactions do not block on full dense cube transfer when a
  streaming backend is available.

### EDS-02: Select Energy Bands Interactively

**User story**: As a user exploring spectra, I want to drag an energy band and
see the map and spectrum feedback update interactively.

**Primary widgets**: ShowEDS.

**Data to use**: real spectrum image with identifiable peaks.

**Acceptance checks**:

- Drag band center and endpoints slowly and quickly.
- Verify map overlay, shaded band, keV labels, and ROI band counts update
  together.
- Verify center drag uses a lightweight preview path and does not lag pointer
  movement.
- Record FPS or pointer-to-preview latency.

### EDS-03: Work With ROIs

**User story**: As a user measuring local chemistry, I want rectangular or
elliptical ROIs to update the spectrum and band counts quickly.

**Primary widgets**: ShowEDS.

**Data to use**: real spectrum image with spatially varying signal.

**Acceptance checks**:

- Draw, move, resize, save, restore, and delete ROIs.
- Verify ROI spectrum and band count update correctly.
- Verify ROI overlays remain readable over the spatial map.
- Verify ROI interaction remains responsive on large data.

### EDS-04: Choose Elements And Peaks

**User story**: As a user identifying chemistry, I want element/edge controls to
guide band placement without hiding the spectrum or map.

**Primary widgets**: ShowEDS.

**Data to use**: real EDS data with known elements when available.

**Acceptance checks**:

- Open element controls and choose several elements/edges.
- Verify peak markers and suggested bands align with the energy axis.
- Verify periodic-table or element UI remains usable in narrow viewports.
- Verify selected elements persist through save/reopen when supported.

### EDS-05: Inspect Map Display And Filtering

**User story**: As a user cleaning noisy EDS maps, I want log/linear scale,
contrast, smoothing, and optional filtering to make weak signals readable
without changing the underlying counts unexpectedly.

**Primary widgets**: ShowEDS.

**Data to use**: noisy real EDS map.

**Acceptance checks**:

- Toggle log/linear scale and verify the default is sensible for EDS.
- Adjust contrast, smoothing, and low-pass/filter controls if available.
- Verify counts/readouts still report the correct quantity and units.
- Verify filtering/display choices are reflected in export or clearly marked as
  display-only.

### EDS-06: Stream Large Spectrum Images

**User story**: As a user opening large EDS files, I want sparse/streaming
access so changing bands and ROIs stays interactive without loading the whole
cube into the browser.

**Primary widgets**: ShowEDS.

**Data to use**: real large EDS file with sparse index or backend streaming.

**Acceptance checks**:

- Confirm the widget uses the streaming backend rather than dense transfer.
- Change band and ROI repeatedly; verify updates stay interactive.
- Verify the kernel/backend does not recompute unnecessary full-cube work.
- Record map update time, spectrum update time, and browser FPS where possible.

### EDS-07: Export And Reopen EDS Results

**User story**: As a user sharing EDS results, I want exported HTML or folder
exports to reopen with map, spectrum, ROI, band, and element state intact.

**Primary widgets**: ShowEDS.

**Data to use**: small portable EDS data and large folder-backed EDS data.

**Acceptance checks**:

- Export standalone/folder HTML modes where supported.
- Reopen exports and drive band, ROI, spectrum scale, map contrast, and element
  controls.
- Verify exported labels explain whether data is exact, downsampled, sparse, or
  display-only.
- Verify file/folder sizes are reasonable and documented.

### EDS-08: Save And Reopen EDS Notebooks

**User story**: As a notebook user, I want ``Cmd+S`` and reopen to preserve a
useful EDS output without embedding huge dense spectra.

**Primary widgets**: ShowEDS.

**Data to use**: real EDS notebook with band, ROI, and selected elements.

**Acceptance checks**:

- Press ``Cmd+S`` and reload/reopen the notebook.
- Verify the visible output returns with useful map/spectrum context.
- Check saved widget state for large dense spectrum-image leaks when streaming
  or lightweight save is expected.

### EDS-09: Use EDS On A Phone Or Narrow View

**User story**: As a user checking an EDS result on a phone or narrow screen, I
want map, spectrum, band controls, ROI controls, and element controls to remain
reachable.

**Primary widgets**: ShowEDS.

**Data to use**: compact EDS export or notebook output.

**Acceptance checks**:

- Test a narrow mobile viewport.
- Verify map and spectrum stack or resize intentionally.
- Test touch-style band drag, ROI drag, spectrum pan/zoom, and element control.
- For iPhone-specific claims, serve the page to physical iPhone Safari.
