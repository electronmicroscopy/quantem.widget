# Changelog

One line per release candidate: the main user-facing thing that changed. Newest
first. Add an entry under **Unreleased** as you land a change; move it under the
new `rcN` heading when that rc is published to TestPyPI.

## Unreleased

- Refresh docs for accuracy: README lists all eight widgets (Show1D, ShowDiffraction added) and the tutorial dataset downloaders, the CLI reference drops the nonexistent `--widget` flag and documents `jupyter`/`qw`/`github`, the HTML export contract drops the unimplemented `float16` encoding, performance notes describe the shipped Show4DSTEM paging and MPS lazy multi-dataset path, and the orphaned load / save-state pages are back in the docs sidebar.
- Add kernel-side element detection to ShowEDS: `detect_elements()` finds significant peaks above a SNIP continuum background and ranks candidate elements with plain per-element reports (matched peaks, missing strong lines, energy error); a Detect button in the periodic-table menu fills the Auto-ID candidate chips, replacing the band-local single-channel heuristic.
- Add ShowFolder as the session browser for microscopy folders, with live refresh, thumbnail/QC previews, metadata tooltips, and lazy paged Show4DSTEM loading for folders of master files.
- Add Show4DSTEM dataset paging and live-folder append workflows so new 4D-STEM acquisitions can appear in the same viewer without rebuilding the notebook.
- Add Show4DSTEM multiple/compare views with panel curation, hide/star/reorder controls, selectable diffraction panels, cursor dragging across tiles, tighter mobile layouts, and safer GPU/memmap cleanup.
- Add paged Show2D / Show3D galleries for iteration or lambda sweeps, including page playback, manual scrub pause, panel reorder controls, stable Show3D panel layouts, and better live rerender docs.
- Add Show3D animation/export polish for GIF/MP4 sharing, frame labels, quality options, binned HTML export, and more reliable FFT overlays with zoom/pan behavior.
- Add Show1D live review workflows for loss curves and reconstruction snapshots, including snapshot thumbnails, hide/star review controls, resizable plots, compact histograms, and a tutorial/API update.
- Improve notebook/HTML sharing and maintainer automation: static widget fallbacks, WebP thumbnail guidance, browser smoke reports, timing/performance signoff, issue templates, and clearer agent/contributor commit guidance.
- Improve I/O and real-data performance paths with GPU image-loading docs, generic helpers from quantem.live, direct uint8 HDF5 browsing, disk-aware Show4DSTEM loader benchmarks, and DataTransfer handoff guidance.

## rc27 - 2026-06-30

- Add ShowDiffraction for calibrated diffraction images and stacks, including d-spacing/ring tools, k calibration, tutorial, API docs, and tests.
- Standardize the widget-level HTML export protocol across viewers, document the notebook/HTML/GitHub sharing paths, and add release/contributor/performance guidance for future widget work.
- Improve Show2D / Show3D / Show3DSlices / Show4DSTEM interaction polish: faster histogram center dragging, mobile/touch controls, tighter panel alignment, hosted example-data docs, and clearer docs navigation.
- Add the merged ShowEDS spectrum-image explorer baseline as experimental; the newer direct-EMD sparse-stream real-data path is still under active testing and is intentionally not part of this release-candidate signoff.

## rc26 - 2026-06-24

- Show2D hover readout (row, col, value, top-right of the image) is now clearly visible: larger, bold, opaque background instead of the old faint translucent text.
- The Hugging Face dataset hub (upload/download) moved to the shared `quantem.data` package; `quantem.widget.io.hub` re-exports it, so existing call sites keep working and data distribution is decoupled from the widgets.
- Show3D / Show3DSlices now take `sampling` + `units` like Show2D / Show4DSTEM (canonical; `pixel_size` kept as a legacy alias), the FFT backend (hardware WebGPU vs CPU) shows in the info tooltip, plus several offline-render and multi-panel FFT fixes.

## rc25 - 2026-06-24

- `quantem jupyter` now prints a highlighted banner with a copy-paste one-liner per OS (macOS / Linux / Windows) that opens the SSH tunnel and your laptop browser in one shot, so you paste a single line and the lab tab appears. JupyterLab's INFO log spam is silenced so the banner stands out, with fallback URL / tunnel-only lines and a link to SSH setup if you have no key yet.

## rc24 - 2026-06-24

- `quantem jupyter <notebook>` now runs on the GPU box itself and prints a paste-ready `http://localhost:<port>/...` URL for your laptop browser: kernel + GPU on the box, widgets in your browser, no laptop-side SSH or setup. First launch saves the SSH target so the printed tunnel line (`ssh -L ...`) is ready to copy. Bring your own tunnel (SSH `-L` or VS Code Remote-SSH), the same model as quantem.live.

## rc23 - 2026-06-24

- New `quantem` command line: `quantem show <path>` (auto-detect) plus `show2d` / `show3d` / `show4dstem`. Render an image, a folder of images, or 4D-STEM master(s) straight to a standalone HTML (images) or a live notebook (4D-STEM, or `--html`); saves to `~/Downloads` and opens automatically. Runs on CUDA / Apple Silicon (MPS) / CPU.
- `quantem show4dstem A B ...` (or a folder) stacks several masters into one 5D viewer with a Dataset slider to flip between scans; `--combined --html` writes that as one offline-WebGPU file.
- Widget HTML export is now a size-labeled dropdown showing the resulting detector resolution (e.g. "uint8 96x96"); PNG / PDF / ZIP figure export removed (HTML only, plus Copy).
- Show4DSTEM HTML export detector binning is now MEAN, not sum, so the bright field no longer clips at uint8 on real-count detectors; binning happens at load so the full stack never has to fit in memory.
- The export button is hidden in the already-exported HTML, and the exported 4D viewer paints its virtual image on mount (no longer blank until you nudge a control).
- Show3D temporal averaging (avg_window) moved to a GPU compute shader - avg=15 scrubs as fast as avg=1 (was a CPU per-pixel loop on the UI thread).
- `io.read_image` loads any common format (tif/png/jpg/bmp/npy/dm3/dm4 + non-Velox emd) into a `Dataset2d`.
- `io.read_images(folder)` loads a whole folder of mixed-format images into a `list[Dataset2d]`.

## rc22

- Scale bar auto-picks nm/Å/pm so labels are clean integers (no more 0.5 nm decimals).

## rc16

- Show4DSTEM browser folder-GUI workflow fix.
