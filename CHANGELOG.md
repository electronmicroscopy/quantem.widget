# Changelog

One line per release candidate: the main user-facing thing that changed. Newest
first. Add an entry under **Unreleased** as you land a change; move it under the
new `rcN` heading when that rc is published to TestPyPI.

## Unreleased

- Fixes from the final cross-platform round. `quantem show4dstem --html` packs
  counts above 255 as uint16 by default (`--dtype auto`) instead of clipping
  them to uint8, and says so; `--out` is the output folder and a file name is
  refused. The Show3D zoomed FFT opens with zero frequency at the centre. On
  Windows adapters below D3D feature level 12 (Chrome and Edge compile WGSL
  with FXC there), the WebGPU colormap shaders build again: display_normalize
  makes one exact division for every window (a window wider than float32 holds
  is halved first, exactly, on both paths), and an adapter that cannot build
  the shaders draws on the CPU with one console warning instead of painting
  black panels.

- Fixes from the hygiene pass. Show4DSTEM stops `pos_row`/`pos_col` at the
  scan edge (a row past it raised IndexError, a column past it showed the next
  row's pattern) and `pattern(row, col)` refuses a position outside the scan.
  `Show4DSTEM.from_folder` without quantem.gpu widens the series to hold a
  float master after integer counts exactly (fractions were truncated).
  Show3D blink compare alternates its two frames (two timers flipped the phase
  back). Show2D and Show3D name the FFT path in use (CPU or WebGPU); Show3D's
  info panel reported WebGPU on the CPU path. ShowDiffraction Identify with an
  empty element field searches the whole library (it found nothing). A folder
  watched from empty shows RGB files. `Show2D.save_image` labels the scale bar
  in the widget's unit (it always said Å), rotation pads mixed shapes with the
  median like construction, and `set_image` copies a float32 array. Show3DSlices
  switches carry their aria-labels again and browser Align fits the active
  panel. HTML exports close their export copy and its Layout model.
- Readers return the dataset type of the machine, so the same code runs on
  every machine. New `read_4dstem(path)` (also `io.read_4dstem`): a
  `quantem.gpu.io.Dataset4dstemGPU` when quantem.gpu is installed and a GPU is
  present, else a quantem core `Dataset4dstem` read densely into host memory;
  a list of paths returns one dataset per file. `Show4DSTEM(path)`,
  `Show4DSTEM.from_folder`, the `quantem show4dstem` notebook for several
  masters and `datasets.show4dstem_gold` (before: core `Dataset4dstem` on
  every machine) follow the same rule. `.npy` paths now load through
  quantem.gpu on a GPU; a file it cannot hold exactly (signed counts, inexact
  float64, a flat 3D array) is read densely with one printed line. Dense reads
  keep uint8 and uint16 counts and store wider counts as uint16 when they fit,
  as quantem.gpu does (before: the narrowest type, so uint16 files of low
  counts became uint8); a file is named without its extension.
  `datasets.showdiffraction_fe3o4` returns a `Dataset2d` (before: an array;
  use `.array`).
- quantem core (`quantem>=0.1.9`) is a dependency except on an Intel Mac and
  Windows on ARM, which cannot install it (torch 2.7 and NumPy 2; hdf5plugin
  6). There the readers return the widget's stand-in, whose `array`, `name`,
  `sampling` (float64 array), `units` (list), `metadata`, `shape`, `ndim` and
  `dtype` match quantem core's; it no longer converts with `np.asarray`, which
  quantem core datasets do not either. `read_image` of a Velox EMD fills
  `metadata`.
- Every widget shows the same numbers for NumPy arrays, torch tensors on any
  device, quantem core datasets (numpy- or tensor-backed), the stand-in and,
  for `Show4DSTEM`, `Dataset4dstemGPU`. `ShowPtycho` accepts a `Dataset2d` (its
  sampling draws the scale bar when its unit is a length), `Plot2D` tensors
  and datasets, and `Show3D` takes a `Dataset2d`'s scale bar. In
  `Show4DSTEM`, a `Dataset4dstemGPU` is titled with its file name as a path
  is, `offline=True` packs it frame by frame from scan windows (only `h5_urls=`
  still raises for it), `scan_region=` also cuts a dense scan before it moves
  to its device, a dataset's calibration applies only when neither `sampling`
  nor `units` is given, and a host scan that would not fit on its device (or a
  widened copy on torch 2.2) is refused with the memory needed instead of
  failing in torch. An export or offline pack holds one float32 dataset at a
  time besides the packed counts (before: every dataset as float32 plus a
  stacked copy).
- ShowDiffraction keeps the recorded counts by default. Zeroing hot pixels on
  integer data (above 3x the 99.9th percentile) is now opt-in with
  `remove_hot_pixels=True`; when such pixels are present a one-line notice
  says how many. Moving frames from MPS to the CPU (more than 2**31 - 1
  values) is now announced.
- `quantem show4dstem --max-gb` sets the largest dense array the command reads
  when quantem.gpu or a GPU is missing; the memory refusal names it.
- Show4DSTEM: resizing the scan-region (VI) ROI from its rim grows or shrinks
  it by the distance dragged, as the detector ROI does, instead of setting the
  size to the pointer's distance from the center; the scan ROI also takes the
  detector's grab rules (inner half moves, rim band resizes, rect edges resize
  along their axis). Diffraction and virtual-image profile points now center
  pixel i at i, the convention the line sampler reads: a profile clicked on
  pixel centers samples those pixels (before, the diffraction profile read half
  a pixel off and a virtual-image point snapped to the next pixel's corner).
  `profile_line` holds pixel indices for clicks on pixel centers; a saved
  `profile_line` from an earlier version draws half a pixel up and left. The
  saved-notebook preview draws ROIs, the scan cursor and the profile on pixel
  centers as the live canvases do.
- Show2D: in a gallery with unlinked contrast, dragging one FFT histogram
  redraws that FFT panel only. Every FFT canvas used to be recolored and
  redrawn on each drag tick (36 draws per other panel for two short drags);
  each canvas now redraws only when its own pixels, view or size change.
- The `quantem` command prints one line, `quantem: <message>`, and exits 1
  for a missing file, a bad argument, a missing package or a refused read
  (`OSError`, `ValueError`, `ImportError`, `MemoryError`); other errors, and an
  import that fails inside quantem.widget itself, keep their traceback.
- Dependencies match the imports: `scikit-image` (ShowDiffraction
  `denoise="nlm"`), `ipywidgets` and `ipython` are declared. PyYAML
  (`dataset.yaml`), ipyfilechooser (`FolderPicker`) and Playwright
  (`quantem github`) are imported where they are used, with the `pip install`
  command in the error. A test fails when a top-level import is not declared or
  a declared dependency has no importer.
- Integer counts reduce exactly on every device through one module,
  `quantem.widget.counts` (int32 blocks on Apple MPS, int64 elsewhere). The
  Show4DSTEM export and `quantem show4dstem --bin` now divide an exact integer
  block total once instead of averaging in float32, which rounded blocks above
  2**24 counts; ShowDiffraction counts its center mask without an int64
  reduction on MPS.
- Removed MP4 export: `Show3D.save_mp4`, the MP4 entries in the Show3D and
  ShowPtycho export menus (the Python ffmpeg path and the browser WebCodecs and
  MediaRecorder paths), and the `imageio-ffmpeg` dependency. GIF export is
  unchanged (byte-identical output); `quantem.gpu.movie` still writes MP4 from
  arrays.
- Removed PlanPtycho, its tutorial, API page, bundle and tests, and `spglib`.
  The `crystal` extra now holds only abTEM, which the ShowCIF potential preview
  uses.
- Sharp pixels by default: Show3D and Show3DSlices now default to
  `smooth=False` (nearest neighbour), and the Show2D FFT Smooth switch starts
  off. Bilinear interpolation is opt-in through the Smooth toggles. Canvases
  that enlarged with interpolation regardless of a toggle now follow it (Show3D
  FFT, FFT insets and kymograph; Show2D FFT) or draw sharp pixels (Show4DSTEM
  FFT of an ROI crop). The Show4DSTEM saved-notebook preview enlarges its
  panels with nearest neighbour as the live canvases do.
- Show3D's status line says when the display stack did not reach the browser
  (a stack above the 2048 MB browser message limit) and names the
  `display_bin` the notebook printed, instead of "WebGPU unavailable ...
  Embedded display stack unavailable · 0.00 MB". True-color stacks report
  canvas drawing without claiming WebGPU is unavailable.
- Show3D draws its CPU-canvas status line in the normal text color instead of
  error red: the CPU canvas is a supported display path.
- The Show4DSTEM dense-read refusal states the size the read needs: the size
  for each integer type the file's dtype and detector bit depth allow (for
  example 9.7 GB as uint8, 19.3 GB as uint16 or 77.3 GB as int64), instead of
  only the uint8 lower bound.
- Show4DSTEM checks its arguments and reads a file before it opens the widget
  model, so a refusal (a file above the memory ceiling, a bad argument) leaves
  no model in the frontend.
- Windows on ARM: `pip install "quantem.widget[cpu]"` no longer builds
  hdf5plugin from source. `hdf5plugin` and `imageio-ffmpeg` (no ARM64 wheels)
  are skipped there and `ncempy` stays below 1.15, which requires hdf5plugin.
  Show4DSTEM imports and reads `.npy` and gzip or uncompressed HDF5 without it;
  a compressed master raises an error naming hdf5plugin and the x64 Python
  route, and `save_mp4` names the missing ffmpeg. `read_image` registers
  hdf5plugin's filters itself for EMD files. The `[cuda]` hint appears only with
  an NVIDIA GPU, `[mps]` only on an Apple silicon Mac. WebGPU adapter requests
  leave out `powerPreference` on Windows, where Chromium ignores it and logs a
  warning. Install notes: `docs/install.md#windows`.
- Removed ShowEDS, its tutorial, API page and bundle, and the `numba` and
  `xraydb` dependencies only it used. View EDS maps with Show2D or Show3D.
- Removed, with their tests: Show3D's per-frame `frame_bytes` transport and
  `offline` trait (every Show3D already played from its embedded stack;
  `offline=` is still accepted and ignored), the browser benchmark hook
  `runTransformBurst`, `read_gif` (`read_image` still reads a GIF's first
  frame), the `quantem.widget.io.RgbImage` re-export and the folder widgets'
  `folder_errors` property.
- Show2D and Show3D show native pixels by default (`display_bin=1`). An active
  bin, explicit or Show2D's `display_bin="auto"` budget, prints one line with the
  factor and `pass display_bin=1 for native pixels`. Show3D's `"auto"` is gone.
  Show4DSTEM packing prints how many counts `uint8` clips. Record:
  `docs/2026-10-07-widget-cleanup.md`.
- Fixed: the FFT, kymograph and unlinked-gallery histograms of Show2D and Show3D
  draw their bars; denoise filters in the browser without WebGPU (the CPU port)
  instead of doing nothing.
- Removed, with their tests: `timing.py`, `io.schema`, `list_conditions`,
  `summary()` on Show1D/Show2D/Show3D/Show4DSTEM, the Show2D and Show4DSTEM
  `profile` properties, the Show2D `mode="folder"` export and its traits, the
  Show2D `gallery_gap_px` and `scale_bar_visible` parameter aliases, the
  compound denoise spellings (`bin2_anscombe`, ...) and the `tv`/`denova`
  display filters, the `show4dstem --dtype float32` choice, synced traits no
  code read (`widget_version`, `ssb_compute_manual_aberrations`,
  `scan_region_json`, `frequency_filter_banner`, `_webgpu_filter_ok`, ShowCIF
  `atom_bytes`/`atom_count`/`cell`, Plot2D `plot_width_px`) and test-only JS
  (`rotateStackInPlane`, `formatFrequencyFilterBanner`).

- Install with `quantem.widget[cuda]` (NVIDIA, through quantem.gpu[cuda]),
  `quantem.widget[mps]` (Apple silicon, through quantem.gpu[mps]) or
  `quantem.widget[cpu]` (no GPU, or an Intel Mac, which runs without
  quantem.gpu); quantem core and quantem.gpu are no longer base dependencies. The base install runs on torch 2.2
  and NumPy 1.26, the newest an Intel Mac can install.
- quantem core and quantem.gpu are reached only through
  `quantem.widget.adapters.core` and `quantem.widget.adapters.gpu`; without them
  readers return an `ArrayDataset` with the same `array`, `sampling`, `units`
  and `name`.
- One device rule: `quantem.widget.device.resolve_device(device="auto")` (CUDA,
  then MPS, then CPU, printed once; a tensor stays on its device). Show4DSTEM and
  ShowDiffraction take `device=`. A GPU without its quantem.gpu path prints one
  line with the `pip` command for that GPU (`[cuda]` or `[mps]`).
- `Show4DSTEM(path)` and `Show4DSTEM.from_folder` open `*_master.h5`, HDF5 and
  `.npy` files: through quantem.gpu when it and a GPU are present, else read
  densely by `show4dstem.reader` (hot pixels median-corrected as quantem.gpu
  does, lossless uint8/uint16 storage, one printed line, refusal above 80% of
  available memory, never binned). Dense tensors reduce in the widget's own torch
  code (`show4dstem.dense`), equal to quantem.gpu on the public gold master.
- Plot2D colormap tables and the Show4DSTEM detector geometry (`detector_mask`,
  `fit_probe`) live in the widget, byte-identical to quantem.gpu.

- Shared widget plumbing lives once: `export.py`, `state.py`, `fallback.py`,
  `pages.py`, `panels.py`, `colormap.py`, `render/figure.py` and `js/shared/`.
  Show2D, Show3D and Show4DSTEM are packages built on those mixins; scale-bar
  labels, saved-notebook previews and the export handshake behave the same in
  every widget. Record: `docs/2026-10-06-widget-refactor.md`.
- Show4DSTEM is one class in the `quantem.widget.show4dstem` package (widget,
  detector glue, export, preview). Removed with their tests and JS: the
  browser-side bslz4 companion, rANS and lazy-HDF5 sources (`offline_codec`,
  `data_url`, `rans_*`, `lazy_url(s)`, `h5_url`, `h5_uint8_lossless`, the CLI
  `--combined` flag), static `DPC_row`/`DPC_col`/`SSB`/`vi_source` maps and
  BF/ABF/ADF preset maps (`compute_ssb` still attaches its maps), 12 `ssb_*`
  solver parameters (pass them to `compute_ssb`), path animation, GIF export,
  `save_image`, ROI and compare-panel helper methods, the compare cache and
  warmers, the GPU memory badge, `debug`, `compare_layout`,
  `compare_panel_gap_px`, and the uint8/uint16 saturation filter that zeroed
  full-scale detector pixels on load (an exported uint8 page now keeps the
  clipped bright field instead of showing it black). `export_html` takes
  `dtype`, `det_bin`, `scan_bin`, `export_kind`, `dataset_scope`.
- Show3D keeps the constructor parameters and methods the tutorials, quantem.live
  and denoise use (34 parameters, 13 methods) and drops the other 93 parameters
  and 52 methods: alternative constructors, the Show2D hand-off, playback and
  panel helpers, the ROI and profile API, ptycho `config` transforms, the
  numpy/torch twin, the Python display-filter fallback, benchmark and debug
  traits. The module is now the package `quantem.widget.show3d` (widget,
  playback, export, fallback). A `Dataset3d` in nm now gets a scale bar in
  display pixels. Removed names: `docs/2026-10-06-widget-refactor.md`.

- Slim the package to the widgets and what the tutorials import: the `quantem`
  CLI keeps only `show4dstem` and `github`; the `web/` browse app, `Mask2D`,
  `Dataset5dstem` paging, `io.data_transfer`, `io.meta`, `io.memory`,
  `io.hub` beyond `download`, `movie`, `paths`, `render.thumbnail`, the
  `_timing` profiler, the export protocol classes, the alias shims and the
  signoff/smoke `scripts/` are removed, with their tests, maintainer docs and
  CI steps. Tutorial loaders live in `quantem.widget.datasets`; GIF/MP4 writers
  for arrays are `quantem.gpu.movie`. CI installs the TestPyPI quantem.gpu pin,
  builds the bundles, runs the size guards and `pytest -q`.

- Add `AGENTS.md` and `tests/test_public_repository.py`: the public-repository
  rules (no private names, paths, data files, widget state or image outputs in
  notebooks, no files over 5 MB except the gold demo gif) are enforced by a fast
  offline test that runs with `pytest`.

- Show4DSTEM opens `quantem.gpu.io.load` acquisitions in their encoded GPU storage
  (512 x 512 x 192 x 192 uint16 scans measured 0.1 to 2 GiB instead of 18 GiB):
  `Show4DSTEM(load(path))` is a live bounded view, `scan_region=` shows part of
  the scan, and `Show4DSTEM(load([a, b]))` a comparison grid labelled by file.
  `Show4DSTEM.from_folder` loads every ready master this way at full detector
  resolution, on CUDA or MPS, and keeps watching the folder; its binning, dtype,
  multi-GPU, paging and preview-cache options, the MPS-specific viewer,
  `quantem.widget.io.resident`, and the CLI `--gpus`/`--page-budget` options are
  removed. Compute SSB runs on CUDA or MPS through `quantem.gpu.SSB` (voltage
  only, every detected BF pixel), and the browser export reads integer `.qem`
  files. Viewer axes take the acquisition's recorded calibration. Requires
  quantem.gpu 0.0.1rc13, the restructured package.

## rc39 - 2026-10-03

- Show4DSTEM compares native diffraction patterns side by side in live
  Multiple view, with shared scan-region reductions and live detector dragging.
  Playback controls are reserved for Single view.

- Remove the ShowFolder session browser and `quantem showfolder` command. Open acquisitions directly with Show4DSTEM; image and acquisition `--watch` commands now use the viewers’ own `from_folder` methods.

- Show4DSTEM static render uses the Show2D overlay font and no forced gap between panels; depends on quantem.gpu 0.0.1rc11 (GPU-owned `Dataset4dstemGPU`, `io.load(files)` returns a list).
- Add `PlanPtycho`: give a crystal (CIF, `ase.Atoms` or Materials Project id) and the microscope settings, and see
  the beam through the specimen, the reconstruction's model window, the probe and the Bragg disks, with graded checks
  (window, scan margin, overlap, detector reach, column lean from tilt, focus inside the specimen). Microscope
  presets (Arina at 91, 115, 185 mm; EMPAD), a custom camera for a collaborator's reported values, and recommended
  settings for 20-200 nm.

- Add `Plot2D` for scalar maps with calibrated Cartesian axes, colormap
  selection, viewport controls, and editable Matplotlib figure export.

- Maintainer docs split pull requests into discuss-first (new widgets,
  cross-widget refactors) and incremental in-widget fixes, and add a
  `widget-tutorials/` upload page for the public
  [bobleesj/quantem-data](https://huggingface.co/datasets/bobleesj/quantem-data)
  dataset. A copyable agent prompt opens that Hugging Face Community
  pull request after `python -m check_meta` prints `ok`.
- Installation, Colab tutorial cells, and the verify snippet now install the
  TestPyPI wheel and import `load` from `quantem.gpu.io`. `quantem
  showdiffraction --demo` is documented as needing the public Fe3O4 hub folder,
  which is not in the current dataset snapshot.
- ShowDiffraction detection denoise: center refinement and spot/ring detection
  now run on a denoised view of the frame (`detect_denoise`, default `"auto"`:
  Anscombe for sparse counting data, light Gaussian for moderate-SNR data,
  identity when clean). All fits and measurements keep using the raw frame, so
  positions and radii are never biased by the smoothing.
- ShowDiffraction `detect_spots` exposes its shot-noise contrast floor as
  `noise_sigma`; lower it on frames whose diffuse scattering or detector
  shadows inflate the robust noise estimate past real peak contrast.
- ShowDiffraction display denoise: a view-only `denoise` trait (including the
  new Poisson non-local means `nlm` filter, which keeps spots sharp where the
  detection blur softens them) and a `show_detection_view` toggle that
  displays what detection saw; both leave stored data and measurements raw.
- Add `Mask2D`, a focused image selector that turns one full-resolution
  rectangle, square, or circle into a Boolean `(row, col)` mask for downstream
  analysis while preserving calibrated dataset display and standalone HTML.
- Add `ChooseLattice`, an interactive 2D selector for choosing an ordered
  origin, a1, and a2 and exposing their `(row, col)` coordinates and derived
  lattice vectors for downstream analysis.
- Exported standalone HTML no longer dies on a first-time (cold cache) load.
  requirejs's default 7 s per-module timeout could make the widget manager load
  a second copy of the anywidget runtime while the CDN fetch was still in
  flight; the model registered its binding in one copy, the view looked it up
  in the other, and the page showed only "Failed to create view ... 
  WidgetBinding not found". The export now pins one anywidget URL and disables
  the requirejs timeout, so every export-capable widget survives an empty
  browser cache. Verified on fresh Chrome profiles that previously reproduced
  the blank page deterministically.
- Show3D gained the same contrast percentile presets Show2D has: the More menu
  now offers Manual / 0.5-99.5 / 1-99 / 2-98 / 3-97 / 5-95 / 10-90, picking one
  pins the histogram window and drops Auto, and `contrast_preset="2-98"` works
  from the constructor and round-trips through `state_dict`.

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
- Fix ShowEDS spectrum line markers so major lines (Fe Ka, Cu Ka, Au La) keep markers across the full energy range.
- Harden ShowDiffraction analysis: calibration assignment and exactness, ring/ellipse/texture fit guards, zone-axis variants, stack frame provenance, per-axis canvas overlays, and hostile-input handling.
- Fix ShowDiffraction systematic absences for spinel, bixbyite, cuprite, rutile-type, and I41/amd phases: new structure-verified rules remove symmetry-forbidden lines (for example Fe3O4 200 and anatase 002) from predicted reflection lists.
- Improve ShowDiffraction phase matching: optimal line assignment replaces the greedy pass, missing-strong counts read n/a when a phase carries no intensities, full-circle radial wedges no longer collapse to zero width, and the ellipse fit's sector weighting is de-biased on noisy patterns.
- ShowDiffraction mobile/touch support: fluid layout at phone widths, pointer-event canvas interactions (tap, drag, two-finger pinch/pan, double-tap reset), and a dual-thumb contrast histogram slider that syncs traits once per gesture.
- ShowDiffraction phase library: pymatgen removed entirely (no more `phaseid` extra or CIF/Structure loaders); built-in lattice constants now carry per-entry source citations (NIST SRM / COD / primary literature).
- Depend on `quantem.gpu[movie]>=0.0.1rc5` for the CUDA/MPS/CPU movie backend, keep migrated HDF5 and movie shims patch-compatible during the transition, and accept 3-axis `Show3DSlices(pixel_size=...)` tuples in release notebooks.
- Add `quantem showdiffraction`: one command from a diffraction pattern (or the real Fe3O4 SAED via `--demo`) to an analyzed standalone HTML through the Auto pipeline, with `--phase` library calibration and indexing.
- Add the `showdiffraction_fe3o4` tutorial dataset loader (public dataset hub download with packaged fallback) and rework the ShowDiffraction tutorial around a quickstart and what the widget solves.
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
