# changelog

## Unreleased

### New widgets
- **Merge4DSTEM** — stack multiple 4D-STEM datasets along a time axis with GPU-accelerated merge, detector binning, source preview, and Zarr export
- **Show1D** — interactive 1D viewer for spectra, profiles, and time series with multi-trace overlay, calibrated axes, log scale, and figure export

### Show2D, Show3D, Mark2D
- **breaking:** `image_width_px` renamed to `canvas_size` — pass `canvas_size=800` to set the canvas display width in pixels (0 = auto)

### Show1D
- auto-contrast with percentile clipping (default 2–98%)
- selectable peak markers with snap-to-local-max, Delete to remove
- grid density slider (5–50 lines)
- axis range lock: drag on X or Y axis to lock, double-click to unlock
- range-scoped statistics: mean/min/max/std and integral within locked range
- peak FWHM measurement with Gaussian fit overlay
- CSV export (range-only or full) from Export dropdown
- figure export (PDF + PNG) from Export dropdown
- legend click to focus/unfocus traces

### Show2D
- file loaders: `from_png`, `from_tiff`, `from_emd`, `from_path`, `from_folder(file_type=...)`
- stack reduction modes (`first`, `index`, `mean`, `max`, `sum`) for collapsing stacks to 2D
- real-time ROI FFT updates during drag (no longer deferred to mouseup)
- layout: stats bar and controls now match canvas width

### Show3D
- diff mode: `diff_mode="previous"` or `"first"` for frame-to-frame or cumulative change
- `profile_all_frames()`: extract the same line profile from every frame
- multi-ROI: place, resize, duplicate, and delete multiple ROIs with per-ROI color and stats
- one-click export bundle (`.zip` with PNG + ROI timeseries CSV + state JSON)
- file loaders: `from_emd`, `from_tiff`, `from_png`, `from_folder(file_type=...)`

### Edit2D
- **breaking:** mask mode now uses rectangle tool only (brush, ellipse, threshold removed for stability)
- undo/redo for mask operations (Ctrl+Z / Ctrl+Shift+Z) with 50-step history
- `brush_size` trait removed from state_dict

### Show2D, Show3D, Mark2D, Show4D, Show4DSTEM, ShowComplex2D, Show3DVolume
- ROI FFT: FFT shows cropped ROI region with real-time updates during drag
- d-spacing click: click FFT panel to measure d-spacing with sub-pixel Bragg spot snap

### ShowComplex2D
- single-mode ROI (circle, square, rectangle) with drag and resize

### Show3DVolume
- single-mode ROI on XY slice (circle, square, rectangle)

### Show2D, Show3D, Show4D, Show4DSTEM, ShowComplex2D, Show3DVolume
- Shift+drag rectangle ROI corner locks aspect ratio

### Edit2D
- per-image independent editing (`shared=False`) with Link toggle
- ArrowUp/ArrowDown nudge, Shift+drag aspect lock, histogram range labels

### Show3D, Show4D, Show4DSTEM
- quick-view presets: save/recall 3 display configurations (`1/2/3`, `Shift+1/2/3`)

## v0.0.8 (2026-02-20)

## v0.0.7 (2026-02-19)

### Bin
- `Bin` widget for calibration-aware 4D-STEM binning with BF/ADF QC previews
- export: `save_image()` (single panel or grid PNG/PDF), `save_zip()` (all panels + metadata), `save_gif()` (original vs binned comparison)

### Batch preprocessing
- preset-driven folder batch runner: `python -m quantem.widget.bin_batch`
- supports `.npy` and `.npz`, plus streamed 5D `.npy` processing for `time_axis=0`
- torch device can be selected via preset or `--device`

## v0.0.6 (2026-02-19)

### Show4DSTEM
- `save_image()` for programmatic export to PNG/PDF with optional metadata sidecar JSON
- export views: `diffraction`, `virtual`, `fft`, `all`
- `save_image()` supports temporary overrides (`position=(row, col)`, `frame_idx`) with automatic state restoration
- exported images now match interactive display settings (colormaps, scale modes, percentile clipping)
- `state_dict()` now includes scan position (`pos_row`, `pos_col`) for exact state restore
- `ArrowUp` / `ArrowDown` scan-row navigation (with `Shift` step), `Esc` to release focus

### Show4D
- `ArrowUp` / `ArrowDown` row navigation (with `Shift` step), `Esc` to release focus

### All profile widgets (Show2D, Show3D, Show4D, Show4DSTEM, Mark2D)
- line-profile endpoints are now draggable after placement
- dragging the line body translates both endpoints together (preserves line shape)
- hover cursor changes near profile endpoints/line

## v0.0.5 (2026-02-18)

### All profile widgets (Show2D, Show3D, Show4D, Show4DSTEM, Mark2D)
- `set_profile` now takes two `(row, col)` tuples: `set_profile((row0, col0), (row1, col1))`

### Show4DSTEM
- 5D time-series/tilt-series support: accepts `(n_frames, scan_rows, scan_cols, det_rows, det_cols)` arrays with frame slider, play/pause controls, and `frame_dim_label` (e.g. `"Tilt"`, `"Time"`, `"Focus"`)
- frame playback: fps slider, loop, bounce, reverse, transport buttons, `[` / `]` keyboard shortcuts
- grab-and-drag ROI: clicking inside the ROI drags with an offset instead of teleporting the center
- theme-aware ROI colors: darker green overlays in light theme for better visibility
- fixed resize handle hit area: was ~70px due to pixel mismatch, now correctly sized

## v0.0.4 (2026-02-16)

### Show2D
- ROI with inner/outer resize handles and cursor feedback
- line profile with hover value readout and resizable height
- live FFT filtering with mask painting
- colorbar overlay, export dropdown, clipboard copy
- auto-contrast with percentile clipping
- gallery mode with keyboard navigation

### Show3D
- ROI and lens controls moved to toggle panels
- line profile with hover value readout and resizable height
- live FFT filtering with mask painting
- FFT panel aligned with main canvas
- colorbar overlay, export dropdown (figure, PNG, GIF, ZIP), clipboard copy
- auto-contrast with percentile clipping

### Show3DVolume
- three orthogonal slice panels (XY, XZ, YZ) with synchronized cursors
- export figure with all three slices
- colorbar overlay, GIF/ZIP export

### Show4DSTEM
- virtual imaging with BF, ABF, ADF, custom ROI presets
- diffraction pattern viewer with annular ROI
- export dropdown and clipboard copy for both panels
- colorbar overlay

### Show4D
- dual navigation/signal panel layout
- ROI masking on navigation image
- path animation with GIF/ZIP export
- export dropdown and clipboard copy

### Align2D
- two-image overlay with opacity blending
- FFT-based auto-alignment via phase correlation
- difference view mode
- export figure for both panels

### Mark2D
- interactive point picker with click-to-place
- ROI support (rectangle, circle, annulus)
- snap-to-peak for precise atomic column positioning
- undo, colorbar overlay, export figure with markers

### Shared
- light/dark theme detection across all widgets
- colormap LUTs (inferno, viridis, plasma, magma, hot, gray)
- WebGPU FFT with CPU fallback
- HiDPI scale bar with unit conversion
- publication figure export via `exportFigure`
- state persistence (`state_dict`, `save`, `load_state_dict`, `state` param)
- `set_image` for replacing data without recreating widget
- NumPy, PyTorch, CuPy array support
- quantem Dataset metadata auto-extraction
- (row, col) coordinate convention

## v0.0.3 (2025-12-01)

- initial release with Show2D, Show3D, Show3DVolume, Show4DSTEM
- demo notebooks
- sphinx docs with pydata theme
