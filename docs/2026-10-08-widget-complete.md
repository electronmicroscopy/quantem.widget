# quantem.widget: last removals and platform gaps (2026-10-08)

Branch `complete`, from `platform` (`12b3fb8`). Goal: remove PlanPtycho and the
MP4 export, and close the known gaps so `[cpu]`, `[mps]` and `[cuda]` installs
all work. One commit per item.

Consumer grep before each removal: quantem.live (src, notebooks), quantem.thick,
Live4DSTEM, Live4DSTEM-iOS, Live4DSTEM-linux, quantem-tutorials. denoise is out
of scope.

## 1. PlanPtycho

Removed `planptycho.py`, `js/planptycho/` (bundle, geometry, goldens),
`scripts/planptycho_goldens.py`, the tutorial, the API page, the toc entries,
`tests/planptycho/` (37 tests), the ShowCIF JS test that imported the
PlanPtycho geometry, the wheel-content checks for `planptycho.js`, and `spglib`
(only PlanPtycho used it). The demo route is now three notebooks.

Consumers: none in the roots above. `~/repos/quantem.gpu`
`simulation/workflow.py:236` names PlanPtycho in an error message (text only, no
import).

Kept: abTEM in the `crystal` extra. ShowCIF's `potential=True` preview (used by
the ShowCIF tutorial) builds its radial tables from abTEM's Lobato
parametrization, and its potential tests (three cases) still skip without
abTEM. Removing abTEM there means shipping the Lobato parameter table in this
MIT package; the table in abTEM is GPL-3.0, so that is a licensing decision,
not a cleanup.

## 2. MP4 export

Removed `Show3D.save_mp4`, `render.gif.write_mp4` and its even-size padding
helper, the MP4 branch of the Show3D toolbar request, the Show3D MP4 menu entry
with its in-browser WebCodecs H.264 encoder and MP4 muxer (about 470 lines), the
ShowPtycho MP4 entry with its MediaRecorder path and frame renderer (ShowPtycho
had no other animation export), the `imageio-ffmpeg` dependency and its
Windows on ARM marker, two tests and the MP4 mentions in the docs. With MP4
gone `save_gif` was the only caller of `_render_animation_frames`, so the
frame rendering now lives in `save_gif` (no 14-keyword forwarding).

Consumers: no quantem.live, quantem.thick, Live4DSTEM or quantem-tutorials
import of `save_mp4`, `write_mp4` or `imageio_ffmpeg`. quantem.live imports
`render.gif.finalize_frame`, which stays. denoise (out of scope) writes MP4
through `quantem.gpu.movie`; its `denova/export.py:291` still imports
`quantem.widget.movie`, a module removed before this branch.

GIF parity, `save_gif` before (platform `12b3fb8`) and after, same input
(two panels, 7 x 48 x 40, 0.05 nm sampling):

| Configuration | sha256 (first 16) | Bytes | Same |
|---|---|---:|---|
| quality low | dcd174ae3a58f85a | 8643 | yes |
| bounce, medium | ab44b8bcdcf26685 | 24071 | yes |
| slides preset | fa720894511614d0 | 31452 | yes |
| frame labels, high, 2x downsample | 723545061cbbea79 | 11425 | yes |
| frames 2-6, every 2, max 2, max edge 32 | 69d8f2ec7e60ecdd | 4699 | yes |
| no titles, scale bar or zoom, white | 33dd2ecb8a760ed2 | 33250 | yes |
| toolbar request (download, bounce, max 3) | c4786959388d2ff3 | 4955 | yes |

The Show3D test now also asserts that the toolbar GIF request returns the same
bytes as `save_gif` with the same options.

## 8. Movie export page

Deleted `docs/tutorials/movie_export.md`, its toc entry and the two links to it
(Advanced overview, API index). The page documented `quantem.gpu.movie`
(array-first GIF/MP4; it exists in quantem.gpu and is not touched here) and
`Show3D.save_mp4` (removed in item 2). The widget docs no longer document a
quantem.gpu module; `docs/api/show3d.md` still points array-first movies at
`quantem.gpu.movie`.

## 3. Exact integer reductions on every device

`count_bound`, `int32_block_sum` and `int32_block_max` moved from
`show4dstem/dense.py` into `src/quantem/widget/counts.py`, with two entry points
every widget calls: `exact_sum(values_t, dim, bound=None)` (an axis or a tuple
of axes) and `exact_max(values_t, bound=None)`. Without a bound off MPS they sum
in int64; on MPS, or with a bound, they run the int32 block path.
`detector_bin_mean` mean-bins detector blocks: counts through `exact_sum` and
one float64 division, float data in float32.

Torch reductions in the widgets, by call site:

| Call site | Before | Now |
|---|---|---|
| Show4DSTEM `DenseSession` sums and max | int32 blocks on MPS (local helpers) | `exact_sum` / `exact_max` (same math) |
| Show4DSTEM `_frame_array` (export `det_bin > 1`) | float32 cast, `mean(dim=(3, 5))` | `detector_bin_mean`: exact int64 total / bin area |
| `quantem show4dstem --bin`, dense path | `to(int64).sum` (int64 copy, CPU only) | `detector_bin_mean` |
| `quantem show4dstem --bin`, encoded path (CUDA or MPS) | float32 cast and sum | `detector_bin_mean` |
| ShowDiffraction `_auto_detect_center` mask count | `mask.sum()` (bool to int64 reduction) | `exact_sum(mask)` |
| ShowDiffraction frame sum for the center | float32 sum | unchanged: ShowDiffraction holds float32 frames by design |
| Show3D ROI plot mean, `aminmax` | CPU float32 tensors from NumPy | unchanged (no device, no integers) |

Measured: for 2 x 3 scan positions of 8 x 8 counts drawn from [2**21, 2**22),
4 x 4 bins, the old float32 mean differs from the exact mean in 8 of 24 blocks;
`detector_bin_mean` equals the int64 total / 16 in all 24. The int32 block path
equals the int64 path over tuple axes, with a bound near int32 (one-value blocks
and the 16-bit limb split), for maxima and for a bool count.

The Show4DSTEM export array of the gold tutorial input (gold_128_npy_bin8,
uint16, max 6057 counts) is byte-identical before and after for `det_bin` 1, 2,
4 and `scan_bin` 1, 2, from CPU and CUDA tensors: below 2**24 per block the
float32 mean was already exact.

Tests: `tests/infrastructure/test_platform.py` (block vs int64 parity, the
binned mean, and `test_widgets_reduce_counts_exactly_on_mps`, which runs
Show4DSTEM sums, maxima and virtual images past 2**31 per pixel and the
ShowDiffraction center on MPS against the CPU numbers; it skips without MPS and
was executed here with the device set to CPU). `test_shared_modules.py` now
fails if a widget redefines the three block helpers. Not run on an MPS GPU in
this branch: no Mac was used.

## 4. Declared dependencies

Static import scan of `src/quantem/widget` (every `import` / `from` outside the
standard library), checked against `[project] dependencies`:

| Module | Used by | Before | Now |
|---|---|---|---|
| `skimage` | ShowDiffraction `denoise="nlm"` (the ShowDiffraction tutorial sets it) | undeclared, lazy | declared `scikit-image>=0.22` |
| `ipywidgets` | Show1D (module top), `export_html` | undeclared (via anywidget) | declared `ipywidgets>=7.6` |
| `IPython` | saved-notebook preview display | undeclared (via ipywidgets) | declared `ipython>=6.1` |
| `yaml` | `io.metadata.dataset_yaml` (quantem.live only) | undeclared, module top | imported in `load_dataset_yaml`, error names `pip install pyyaml` |
| `ipyfilechooser` | `FolderPicker` (quantem.live only) | undeclared, lazy | error names `pip install ipyfilechooser` |
| `playwright` | `quantem github` screenshots | undeclared, lazy | error names `pip install playwright` |
| `abtem` | ShowCIF `potential=True` | `[crystal]` extra | error names `pip install "quantem.widget[crystal]"` |
| `rsciio` | none (ShowEDS was its only importer) | not declared | nothing to do |

Every declared dependency has an importer except `imagecodecs` (tifffile's
codec plugin) and `jupyterlab_widgets` (the JupyterLab widget manager), both
loaded by other packages. `test_optional_dependencies.py` now asserts all of
this from the source: a module imported outside a function is declared, an
optional one is imported only inside the function that needs it (quantem core
and quantem.gpu only at the top of `adapters/`), and the declared set minus the
imported set is exactly those two. The base-install subprocess test also runs
ShowDiffraction `denoise="nlm"`.

Import trace in a fresh CPU venv (uv, Python 3.12, torch 2.14.1+cpu, `-e
".[cpu,test]"`), every module imported and every widget constructed and
exported: no import failures; the distributions imported that are not declared
are transitive ones imported by matplotlib, anywidget and torch (`comm`,
`cycler`, `fonttools`, `kiwisolver`, `packaging`, `psygnal`, ...) and `tqdm`
(optional progress). torch 2.2.2 + numpy 1.26.4 venv:
`test_optional_dependencies.py` 5 passed, scikit-image 0.26.0 imports there.

## 7. `quantem` command errors

`cli.main` caught only `FileNotFoundError` and `ValueError`; a missing package
or a refused dense read printed a traceback. It now catches `OSError` (which
includes `FileNotFoundError`, `PermissionError` and h5py's open errors),
`ValueError`, `ImportError` and `MemoryError`, prints `quantem: <message>` on
one line (whitespace collapsed; the exception name when the message is empty)
and returns 1. An `ImportError` whose module is inside `quantem.widget` is
re-raised: that is a broken import in this package, not a missing install.
Every other exception keeps its traceback.

Test `tests/test_cli.py::test_user_errors_print_one_line_and_bugs_keep_their_traceback`:
a dense read refused by the memory ceiling (`--html` path: one
`quantem: skipped ...` line per master, then one summary line), a missing
master, an `ImportError` with an install command, an empty `MemoryError`, and a
`TypeError` and an internal `ImportError` that still raise.

## 5. Show2D unlinked gallery: FFT contrast drag

Reproduced in headed Chrome on display :0 (NVIDIA adapter, `blackwell`, and with
`navigator.gpu` removed), on the export of
`Show2D([gold, rot90(gold), flipud(gold)], show_fft=True, link_contrast=False)`
and in a live JupyterLab kernel. A probe counted `drawImage` / `putImageData` /
`clearRect` calls per scientific canvas while the mouse dragged the max handle
of the first FFT histogram:

| Build | Path | Draws on FFT 1 / FFT 2 / FFT 3, first drag (10 moves) | second drag (5 moves) |
|---|---|---|---|
| platform `12b3fb8` | WebGPU | 22 / 22 / 22 | 12 / 12 / 12 |
| platform `12b3fb8` | Canvas2D | 22 / 22 / 22 | 12 / 12 / 12 |
| this branch | WebGPU | 20 / 0 / 0 | 8 / 0 / 0 |
| this branch | Canvas2D | 20 / 0 / 0 | 10 / 0 / 0 |

Cause: the gallery FFT color effect recolored every visible slot whenever any
panel's contrast changed, and the draw effect then cleared and redrew every FFT
canvas. The other panels came out with the same pixels, so a pixel diff alone
shows nothing; the draws are the repaint. Fix (`js/show2d/index.tsx`): each
pipeline entry keeps the key of what its offscreen holds (source, size, scale,
auto, colormap, range, repaint signal) and is recolored only when that key
changes; each canvas keeps the key it was last drawn with (offscreen paint
stamp, zoom, pan, sizes, smoothing, repaint signal) and is redrawn only when it
changes. The dragged panel's final pixels equal the platform build's for the
same drag (0 differing pixels inside the FFT canvases).

Test: `tests/browser/test_webgpu_cpu_parity.py::test_unlinked_gallery_fft_contrast_drag_repaints_only_its_panel`
(WebGPU and Canvas2D) drags twice with the mouse and asserts the dragged FFT
panel repaints and changes while the other two have 0 draws and the same
checksum. On the platform bundles it fails with 36 draws on each other panel.
Screenshot read after the drags: the first FFT panel's center is brighter (max
handle at 1.6e+6), the other two FFT panels and their histograms (max 4.2e+6)
are unchanged.

## 6. Show4DSTEM scan ROI resize and profile pixel alignment

Reproduced in a live JupyterLab kernel, headed Chrome on display :0 (NVIDIA
`blackwell`), on a 16 x 16 x 16 x 16 synthetic (one bright detector pixel at
(5, 9); 30 screen px per scan pixel), with the platform build and this branch:

| Check | platform `12b3fb8` | this branch |
|---|---|---|
| Scan ROI circle r = 4 at (7, 7), pressed 3 px right of the center, dragged 3 px right: kernel `vi_roi_radius` during the drag | 4, 4, 5, 5, 6, 6 | 5, 5, 6, 6, 7, 7 |
| Radius after release (expected 4 + 3) | 6 (the pointer's distance) | 7 |
| Diffraction profile clicked on the centers of pixels (5, 2) and (5, 13): `profile_line` | (5.5, 2.5), (5.5, 13.5) | (5, 2), (5, 13) |
| Virtual-image profile clicked on the centers of (3, 3) and (12, 3): drawn column, scan px | 3.98 (pixel 3/4 boundary) | 3.48 (center of column 3) |

Causes. The scan ROI resize set the size to the pointer's distance from the
center on every move, so a press inside the rim band jumped; it now uses the
detector's `grabDetector` (press classification) and
`resizeDetectorFromPointer` (size changes by the distance moved since the
press), rounded to whole scan pixels as before. Profile points were stored in
image coordinates (pixel i spans [i, i + 1)) while `sampleLineProfile`
interpolates with pixel i at i: a diffraction point on a pixel center (5.5)
sampled halfway into the next row and column (a quarter of the bright pixel at
its center), and a virtual-image click was rounded from 3.5 to 4, so the point
was drawn on a pixel corner and sampled pixel 4. Both profiles now store, hit
test and drag in the mask convention (`canvasToMask`) and draw with
`maskToCanvas`, like the ROIs. The saved-notebook preview (`preview.py`) drew
ROI centers, the scan cursor and the profile at `i * scale`; it now draws at
`(i + 0.5) * scale` like the live canvases.

Screenshots read: before the drag both builds show the radius-4 circle on scan
pixel (7, 7); after it the platform circle is 12 px across and this branch's
14 px, top edge at row 0.5. The diffraction profile line runs through the
centers of row 5 in both builds (only the stored and sampled coordinates
differ). The virtual-image profile line runs along the boundary between columns
3 and 4 on the platform build and through the middle of column 3 here.

Tests: `js/show4dstem/detectorInteraction.test.ts` (a click on a pixel center
on a zoomed, panned canvas stores that index and the profile reads the pixel's
full value, 1000; the image-coordinate point reads 257.5);
`tests/show4dstem/test_show4dstem.py::test_saved_notebook_preview_draws_overlays_on_pixel_centers`
(fails on the platform build: profile row 50 instead of 55);
`tests/browser/test_show4dstem_live.py::test_scan_roi_resize_follows_the_grab_and_profiles_sit_on_pixel_centers`
(fails on the platform bundles with radius 6).

## Gates

| Gate | Result |
|---|---|
| fast pytest, live-env, GPU 0 | 376 passed, 21 skipped (the opt-in browser tier). One of six runs had 1 failure while other processes held GPU 0 at 99 %; 5 reruns passed and the failing test was not captured (the suite has wall-clock asserts in folder watching, ShowDiffraction and save-state tests) |
| CPU-only venv (uv, Python 3.12, torch 2.14.1+cpu, `-e ".[cpu,test]"`), plain pytest in the activated venv | 353 passed, 37 skipped (quantem.gpu parity and encoded paths, abTEM, MPS, browser) |
| torch 2.2.2 + numpy 1.26.4 venv (Python 3.11), `test_optional_dependencies.py` | 5 passed |
| `QUANTEM_GPU_SRC=../gpu-platform/src npm run build`, `npm run typecheck`, `npm test` | 10 bundles; 0 errors; 39 files, 197 tests |
| 14 tutorials through nbconvert (live-env) | 0 errors |
| browser tier (`QUANTEM_WIDGET_BROWSER=1`, display :0, NVIDIA `blackwell`, WebGPU on and removed) | 20 passed. An earlier full run had 19 passed and 1 failed: `test_show4dstem_live_kernel_same_pixels_and_numbers`, where the hover readout box was in the WebGPU virtual-image screenshot and not in the CPU one (the image pixels were equal); the test passed alone on this branch and on the platform build |

`uv pip compile` of `[cpu]`, Python 3.12, `--only-binary :all:`, PyPI plus the
PyTorch CPU index (`unsafe-best-match`):

| Platform | Result | torch | numpy | scikit-image | Packages | hdf5plugin |
|---|---|---|---|---|---:|---|
| linux x86_64 | resolves | 2.14.1+cpu | 2.5.3 | 0.26.0 | 65 | yes |
| macOS arm64 | resolves | 2.11.0 | 2.5.3 | 0.26.0 | 65 | yes |
| macOS x86_64 | resolves | 2.2.2 | 1.26.4 | 0.26.0 | 64 | yes |
| Windows x86_64 | resolves | 2.14.1+cpu | 2.5.3 | 0.26.0 | 64 | yes |
| Windows ARM64 | resolves | 2.14.1+cpu | 2.5.3 | 0.26.0 | 62 | skipped (marker) |

No resolution contains imageio-ffmpeg or spglib.
