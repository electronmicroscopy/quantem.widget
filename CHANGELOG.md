# Changelog

One line per release candidate: the main user-facing thing that changed. Newest
first. Add an entry under **Unreleased** as you land a change; move it under the
new `rcN` heading when that rc is published to TestPyPI.

## Unreleased

- Show3DSlices oblique panel geometry is no longer tied to the Align control.
  The GPU slice shader receives the cut's start/stop on every render, so the
  Angle and Position sliders move the vertical cut whether or not slice
  alignment is on. Previously, with Align off the shader received a degenerate
  zero-length segment and every output column sampled the same corner voxel, so
  the panel painted flat horizontal bands that ignored both sliders.
- Show3DSlices oblique panel now repaints per drag frame, matching the slice
  slider: the segment lives in comm-synced traits that React batches during a
  drag, so the panel is direct-painted from the resident GPU volume instead of
  waiting for the round-trip. Measured 8 of 8 mid-drag frames repainting where
  3 of 8 did before.
- Show3DSlices can estimate its global depth tilt in the browser. `Align` runs
  the same registration the kernel does - median centering, Gaussian high pass,
  Hann window, cross-correlation, upsampled-DFT subpixel refinement, linear fit
  - entirely in WebGPU, so exported standalone HTML aligns a stack with no
  Python attached and a live notebook skips a comm round-trip. The estimate is
  GPU-resident: each slice uploads once and the spectra stay in device buffers,
  so a 16 x 1688 x 1688 stack moves a few hundred KB back instead of about
  1.2 GB. Fitted slopes match the kernel estimator to under 2e-3 px/slice on
  real reconstructions, and the toolbar names the backend it used.
- Show3DSlices `Planes` toggles now show and hide the matching 2D slice panel,
  not just the plane inside the 3D volume view.
- Show3DSlices Align toggle now repaints both slice panels when switched on or
  off; the blit that copies each offscreen to its visible canvas had no
  dependency on the alignment state, so the panels kept the previous shifts.
- Show3DSlices reports the oblique plane center in fixed image pixels next to
  Position. Position is measured along the plane normal, an axis that turns with
  Angle, so its number moves under rotation even when the cut does not.

- ShowPtycho on MPS now uses the phase/loss-only `quantem.gpu` SSB path for
  interactive phase and loss updates instead of also accumulating the object
  wave. On a private full 512x512 real-data Apple GPU timing gate this lowered
  the prepared hot loop from about 229 ms to about 79 ms with only float32-level
  loss differences.
- Show4DSTEM WebGPU virtual-image/DPC mask construction now imports from the
  synced `quantem.gpu.webgpu` engine source; DPC row/col buttons can now use
  the browser WGSL backend even when no static DPC product maps were supplied.
  Browser signoff covers exported sidecar HTML and live Jupyter interaction with
  dataset flips, DPC row/col recompute, FFT toggles, PNG copy buttons, and
  BF/CoM/DPC WGSL parity.
- Show4DSTEM CUDA compare grids now reuse per-panel `quantem.gpu` compute
  backends, so repeated BF/ADF/DF updates keep detector-index and dense
  total-count caches instead of rebuilding them every refresh.
- ShowPtycho WebGPU folders now open with no server at all: double-click `index.html`, click "Open data folder", pick the folder (named in the banner, picker starts in Downloads). `quantem showptycho <folder>` still serves and opens it automatically - two equal paths, both in the folder README.
- Save inside the review persists to the folder: Save writes the phase JPEG plus the aberration state into `saves/` (and downloads the JPEG), so saved states reappear with Load / download / delete on any relaunch - double-click or CLI. The bundled range server accepts writes only under `saves/`.
- SSB reconstruction in the browser is 5.6x faster at full bright field: slider drags use a Fourier-domain BF sum with a single inverse FFT (the same `angle(mean(object))` estimator as the Python reference, corr 0.997), reaching ~50 FPS on a real 512x512x192x192 dataset at all 13137 BF pixels on an Apple-silicon laptop. Release commits keep the exact per-BF path for the loss readout.
- Resident G(q,k) stores the Hermitian half-plane by default (bit-exact, 2x less GPU memory, faster) with an opt-in snorm16 quantized mode (4x), a GPU-memory clamp on the BF count so big scans cannot crash small GPUs, and acceptance of rfft half-plane calibrations from the CUDA backend.
- Depend on `quantem.gpu[movie]>=0.0.1rc5` for the CUDA/MPS/CPU movie backend, keep migrated HDF5 and movie shims patch-compatible during the transition, and accept 3-axis `Show3DSlices(pixel_size=...)` tuples in release notebooks.
- Keep display filtering consistent throughout review: Show3D reapplies denoise when a user scrubs to another frame, standalone Show2D/Show3D HTML paints denoised and frequency-filtered pixels on first load, and unlinked Show2D galleries show and edit each selected panel's own denoise mode, sigma, and bin without changing neighboring panels.
- Add display-side denoise for sparse maps (EDS, low dose) to Show2D and Show3D: a Denoise menu (`none` / `gaussian` / `anscombe`, the count-respecting Anscombe smoother) with `denoise_sigma` and `denoise_bin` knobs, a `show_denoise` gate that keeps the controls row hidden until needed, and an always-on banner whenever a reduction is active. It is purely a view transform: the stored array, the stats row, and every export of raw data keep the original counts, and `none` is the lossless default. Show2D adds per-panel lists for raw-vs-denoised A/B galleries and runs the filter through a browser-side WebGPU pipeline, so exported HTML denoises without a kernel. Replaces the earlier `display_filter` / `display_sigma` / `spatial_bin` kwargs, which stay accepted as aliases for one release.
- Add a Show2D HAADF underlay: `underlay=True` on a `(haadf, map)` pair adds a third panel blending the map onto the HAADF lattice, with a `magenta` colormap so bright atomic columns render magenta instead of clipping to white. Tune with `underlay_alpha` and `underlay_haadf_gain`.
- Add reversible crop-to-view and pad view ops to single-panel Show2D: `crop_to_view()` commits the browser viewport as the display extent (crop applies before denoise), the `pad_ratio` kwarg/trait adds a minimum-valued border, the toolbar View menu gains Crop to view / Pad 5-20% / Reset view entries, an always-on `view:` banner announces any active reduction, and `reset_view_ops()` restores the full frame bit-identically.
- Move the gallery denoise scope toggle into the Link group (Link Zoom / Pan / Contrast / Denoise): checked applies denoise edits to every panel, unchecked scopes them to the selected panel; the denoise row keeps only the Filter / sigma / Bin knobs.

## rc30 - 2026-07-10

- Serve all tutorial data from the widget-organized `widget-tutorials/` tree on Hugging Face (`show2d_gold`, `show4dstem_gold`, ... — one call per dataset), retarget the Colab workshop notebooks to it, and document the upload protocol so contributors can share datasets the same way.
- Keep docs pages single-widget: docs/CI builds set `QUANTEM_WIDGET_STATIC_FALLBACK=0` so the saved-notebook static preview never duplicates the live widget, and Colab bootstrap cells are hidden from built pages via `remove-cell` tags.
- Adopt the scikit-package contribution standards (issue-first, one themed PR per issue, no force-push under review) in README/AGENTS/CONTRIBUTING, and reorganize the docs sidebar (Advanced section, API reference under Developers).
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

## rc29 - 2026-07-03

- Dead-code sweep: drop unused imports and add `__future__` annotations across the package (no behavior change; the feature bullets accumulated between rc27 and rc30 are listed under rc30).

## rc28 - 2026-06-30

- Add real-data widget tutorials and offline 4D-STEM support.

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
