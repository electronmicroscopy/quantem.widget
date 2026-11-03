# quantem.widget cleanup after the platform migration (2026-10-07)

Branch `cleanup`, from `platform` (`3db3d7b`). Question: which code is no longer
needed now that quantem core and quantem.gpu are optional, and do the display
defaults keep every pixel?

## Evidence

- Coverage (`coverage run`) of the fast pytest suite and of the 16 tutorials
  executed with nbconvert, separately, so a function reached only by tests shows
  up as such. 15,052 statements, 76 % covered by the two together.
- vulture over `src/quantem/widget`; knip over `js/` with the bundles as the only
  entries (production mode); `tsc --noUnusedLocals --noUnusedParameters` is
  already clean.
- Every synced trait checked for a JS reader (`useModelState`, `model.get`,
  `change:` listeners) and a Python reader.
- Every unexecuted public name grepped in quantem.live, denoise, quantem.thick,
  Live4DSTEM*, quantem-tutorials and quantem.gpu.

Classes used below: DEAD (no caller anywhere), TEST-ONLY (only tests call it),
CONSUMER-USED (another repository calls it), LIVE-UNEXERCISED (reachable from
the browser, the CLI or a user call, but no test runs it).

## Removed

| Name | Class | Note |
|---|---|---|
| `timing.py` | DEAD | nothing imports it |
| `io/schema` (`stamp_schema_version`, `atomic_write_json`, `versions.py`) | DEAD | quantem.live has its own `io.schema`; the one version check moved into `dataset_yaml.py` |
| `dataset_yaml.list_conditions`, `cli.MASTER_PATTERN` | DEAD | |
| `show2d.options._nonnegative_float`, `_normalize_rotation_list`, `_rotation_to_quarter_turns` | DEAD | |
| `summary()` on Show1D, Show2D, Show3D, Show4DSTEM; `profile` on Show2D, Show4DSTEM | DEAD | no tutorial or consumer; ShowDiffraction `summary()` stays (tutorial) |
| `quantem show4dstem --dtype float32` | DEAD | the export refused it; argparse now offers only `u8`/`u16` |
| display filter: `bin2`, `bin2_anscombe`, `bin4_anscombe`, `denova*`, `tv`, `filter=`, `resolve_denoise_mode`, `magenta_cmap`, `blend_map_on_haadf` | DEAD / TEST-ONLY | ShowDiffraction uses none/gaussian/anscombe/nlm; the browser menus offer none/gaussian/anscombe with a bin knob |
| `_webgpu_filter_ok` (Show2D, Show3D) | old design | the Python scipy filter it negotiated with is gone; it switched denoise off in live sessions without WebGPU |
| traits `frequency_filter_banner` (Show3D), `widget_version` (Show4DSTEM, ShowEDS), `ssb_compute_manual_aberrations`, `scan_region_json`, ShowCIF `atom_bytes`/`atom_count`/`cell`, Plot2D `plot_width_px` | DEAD / TEST-ONLY | no JS or Python reader; ShowCIF no longer ships the expanded supercell (up to 4 MB) the browser never read |
| Show2D `mode="folder"` export, `_write_html_folder_export_fast`, traits `frame_bytes_url`, `frame_bytes_urls`, `panel_stack_bytes_url` and their JS fetchers | DEAD | no tutorial, test or consumer |
| Show2D `gallery_gap_px` / `gallery_gap_color`, `scale_bar_visible=` parameter | legacy aliases | use `inter_panel_gap_*`, `show_scale_bar` |
| Show3D GIF `ppt_preset` / `preset=` request keys | DEAD | the toolbar sends `slides_preset` |
| attributes written and never read: `_vmin_user`, `_vmax_user`, `_folder_poll_waiting`, `_folder_poll_error`, `_notebook_preview_mime`, `_static_fallback_fill` | DEAD | |
| unused parameters: `cli._render_4dstem(label)`, `_normalize_panel_indices(name)`, `print_distribution_status(installed)` | DEAD | |
| torch duck typing in `profile` / `free_gpu` | old design | torch is a base dependency (>= 2.2 has `torch.backends.mps` and `torch.mps.*_allocated_memory`) |
| JS `rotateStackInPlane`, `formatFrequencyFilterBanner`, rotation goldens | TEST-ONLY | |

## Kept

CONSUMER-USED: `free_gpu`, `gpu.gpu_info`, `gpu.vram_status` (quantem.live
`gpu.py`, `widgets/live.py`); `folder_picker` (quantem.live `folder_picker.py`);
`io.hdf5_family.collect_hdf5_family` (quantem.live `server/routers/acquisitions.py`);
`io.metadata.dataset_yaml.load_dataset_yaml` / `resolve_condition` (quantem.live
screen pipeline, routers, ptycho config); `io.utils.atomic_write_json` /
`locked_json_rmw` (quantem.live orchestrator); `io.hub.download` (quantem.live
`cli/data/hub.py`); `render.gif.finalize_frame` (quantem.live `export.py`); Show3D
`offline=` (accepted and ignored; quantem.live `widgets/show3d_watch.py`,
`notebooks/screen.py`); Show2D `verbose=` (quantem.live `export.py`); the
`AcquisitionView` attributes `_bounded_detector_source`, `_detector_source`,
`_detector_region` (read by quantem.gpu `detector/session.py`, `bounded.py`);
`read_image(...).scan_rotation_deg` (quantem-tutorials drift notebooks).

LIVE-UNEXERCISED (no test yet): the `show4dstem --html` packed path, `--serve`
and the `github` browser capture; Show4DSTEM folder watching, the SSB compute
request, preset requests and the frontend-ready handshake; Show2D static
previews with ROIs and inset plots and the SVG inset/math paths; the Show3D ROI
plot; Show3DSlices plane-visibility observers; ShowDiffraction request handlers,
`pseudo_voigt_peak`, the extinction rules and `phase_from_entry`; the ShowEDS
EMD and Velox loaders (science stays); ShowPtycho sample-fit requests;
`device_info` (named in the refactor standards); `ArrayDataset.dtype`/`ndim`
(mirror quantem core).

Not removed, with evidence: the Show3D browser still carries a per-frame
`frame_bytes` transport and `!offline` branches (about 70 sites in
`js/show3d/index.tsx`), but Python always packs the embedded stack and sets
`offline=True`, so they never run. They are entangled with playback and scrub
paths; removing them needs its own change with the browser tier. The folder
mixin's `hasattr(widget, "pixel_sizes")` checks serve Show2D and Show3D.

## Lossless display defaults

Show3D defaulted to `display_bin=4` and Show2D to `display_bin="auto"`; both
reduced pixels without being asked (a 5 x 7 frame from `Show3D.from_folder`
became 1 x 1). Now both default to `display_bin=1`. An active bin prints one
line whatever `verbose` is, for example
`Show3D display bin 4x: 4096x4096 -> 1024x1024 (mean); pass display_bin=1 for native pixels`.
Show2D keeps `display_bin="auto"` as an opt-in budget (2.5 GB of browser GPU
buffers, 16 MiB of initial payload per float32 panel); full-resolution detail
still streams on zoom. Show3D's `"auto"` only meant 4, nothing used it, and it
is gone. Show4DSTEM `uint8` packing (the compact export default) now prints how
many counts it clips and points to `dtype="uint16"`.

Open time, live JupyterLab, headed Chrome on the NVIDIA adapter (WebGPU on),
GPU 0, from Shift+Enter on `Show3D(stack)` to the first painted frame:

| Stack (float32) | `display_bin` | Payload | Widget mounted | First frame painted | Result |
|---|---|---:|---:|---:|---|
| 50 x 4096 x 4096 | platform default (4) | 200 MB | 22.7 s | 22.9 s | WebGPU resident, 1024 x 1024 |
| 50 x 4096 x 4096 | 4 (this branch) | 200 MB | 19.1 s | 19.3 s | WebGPU resident, 1024 x 1024 |
| 50 x 4096 x 4096 | 2 | 800 MB | 23.9 s | 24.2 s | WebGPU resident, 2048 x 2048 |
| 50 x 4096 x 4096 | default (1) | 3200 MB | mounts | never | static preview only: the websocket drops the message, "Embedded display stack unavailable" |
| 34 x 4096 x 4096 | default (1) | 2176 MB | mounts | never | same failure |
| 30 x 4096 x 4096 | default (1) | 1920 MB | 17.5 s | 18.1 s | WebGPU resident, native |
| 50 x 1024 x 1024 | default (1) | 200 MB | 2.7 s | 2.9 s | WebGPU resident, native |

Python-side construction of the 50 x 4096 x 4096 stack takes 4.1 s at bin 4 and
5.2 s native. One kernel message above 2 GiB never reaches the browser, so a
native stack past that size is not usable interactively. The rule stays
lossless; instead Show2D and Show3D now print, for a payload over 2 GiB, the
smallest `display_bin` that fits (`Pass display_bin=2 (800 MB) for an
interactive view.`).

Measurement note: Playwright enables the CDP Network domain, which copies every
websocket frame to the driver as base64. With it, every 200 MB open above took
about 467 s, the same for every configuration. These numbers come from a raw
CDP client that uses only the Runtime and Input domains.

## Fixed

- The FFT, kymograph and unlinked-gallery histograms of Show2D and Show3D drew
  no bars on either path: the shared `Histogram` refactor kept call sites that
  passed neither `bins` nor `data`. They now pass the FFT magnitudes they already
  hold, binned over the slider range, so the bars sit under the clip handles.
  Show3DSlices was not affected. The bars follow a numpy histogram of the same
  data (Pearson r 0.9999). The new browser-tier test fails on the platform build
  (6 of 8 exports have a histogram with 0 bar columns) and passes here.
- Denoise did nothing in a live Show2D/Show3D without WebGPU: the browser waited
  for `_webgpu_filter_ok`, which the missing adapter set to False, and the Python
  filter it fell back to had been deleted. Measured in a live kernel with WebGPU
  removed, a gaussian-denoised panel painted exactly the raw pixels (mean
  difference 0.0 of 255) before and differs by 43.7 now; the CPU port runs.

## Gates

| Gate | Result |
|---|---|
| fast pytest, live-env, GPU 0 | 417 passed, 12 skipped (the opt-in browser tier) |
| CPU-only venv (uv, torch 2.14 CPU, `pip install -e ".[cpu,test]"`), plain pytest | 358 passed, 28 skipped (quantem.gpu, abTEM, browser) |
| torch 2.2.2 + numpy 1.26 venv, `tests/infrastructure/test_optional_dependencies.py` | passed |
| `QUANTEM_GPU_SRC=... npm run build`, `tsc --noEmit`, vitest | built 12 bundles; 0 errors; 39 files, 202 tests |
| 16 tutorials through nbconvert | 0 errors |
| browser tier (`QUANTEM_WIDGET_BROWSER=1`, headed Chrome on the NVIDIA adapter, WebGPU on and removed) | 12 passed |

The browser tier now checks that each export shows its data before comparing
pixels: every image and FFT panel on top and non-blank (more than 1 of spread,
8 or more colors; WebGPU canvases read from the screen), every histogram with
bars, the Show2D image histogram correlated with numpy (r > 0.9), and a panel
repainting after a histogram handle moves. The live Show4DSTEM test checks the
same for the diffraction, virtual-image and FFT panels and that they change
between scan positions, presets and colormaps; it now runs JupyterLab with an
empty config directory so user save hooks stay out.
