# quantem.widget dead code and hygiene pass (2026-10-08)

Branch `hygiene`, from `platform` (`afbe104`). Bob: "remove dead code and do code
hygiene a lot as you know my preferences". Order of work: correctness bugs, dead
code with evidence, hygiene to the lab's code style, then proof that numbers and
pixels did not change.

## 1. Correctness bugs

Each fix has a test that fails before it.

| Commit | Bug | Test |
|---|---|---|
| `fix: Show4DSTEM scan position stops at the scan edge` | `pos_row` past the scan raised IndexError in `DenseSession.frame`; `pos_col` past it showed the next row's pattern (flat index wrap); `-1` showed the last row | `test_scan_position_outside_the_scan_stops_at_its_edge` |
| `fix: Show4DSTEM.pattern refuses a position outside the scan` | `pattern(0, 6)` on a 6 x 6 scan returned the pattern at (1, 0) | same test |
| `fix: Show4DSTEM.from_folder widens the series for a float master` | the dense series widened only when a later master's itemsize grew, so int64 counts then float32, or uint16 then float16, truncated the fractions; now NumPy's promotion picks the narrowest dtype holding both | `test_from_folder_holds_a_float_master_after_integer_counts_exactly` (2 cases) |
| `fix: show3dslices alignment uses np.trunc, numpy 2.5 deprecates np.fix` | DeprecationWarning (an error under `-W error`) on NumPy 2.5; the alignment result is byte-identical | checked in the CPU venv (NumPy 2.5.3) |
| `fix: Show2D and Show3D state the FFT path they use, drop debug logs` | on the CPU path Show2D logged "WebGPU FFT initialized", "WebGPU colormap engine initialized" and "(WebGPU batch=4)" and showed "Computing FFT... (WebGPU)"; Show3D's info panel said "WebGPU: available" and "FFT compute: WebGPU". Timing logs removed; labels use the FFT object's `path` | browser tier, both paths |
| `fix: Show3D blink compare alternates, one timer instead of two` | two effects each toggled the blink phase at the same period, so the phase flipped twice per tick and the image never changed (0 switches in 4 s; 7 to 8 after) | `test_show3d_blink_compare_alternates_its_two_frames` (browser tier, WebGPU and CPU) |
| `fix: show3d and show3dslices exports close their clone model` | their export copy was freed but never closed: two models per export stayed open | `test_export_html_leaves_no_clone_model_open` (Show3D, Show3DSlices) |
| `fix: an HTML export closes its clone's Layout model too` | `Widget.close` leaves the child `Layout` open: one model per export for every widget | `test_widget_export_html_writes_standalone_state` now asserts no model is left |
| `fix: ShowDiffraction Identify with no element filter found no phase` | an empty element field split to `[""]`, became an empty set and filtered out every phase | `test_identify_with_an_empty_element_field_searches_the_whole_library` |
| `fix: folder watched from empty never showed RGB files` | the first RGB file set the expected shape to (H, W, 3), so its own confirming poll failed the (H, W) check | `tests/test_image_folder_watch.py` |
| `fix: Show2D.save_image scale bar labels the widget's pixel unit` | the label assumed Angstrom: the widget-export tutorial figure (0.149 nm/px) read "1 nm" under a 10 nm bar | `test_save_image_scale_bar_labels_the_pixel_unit` |
| `fix: Show2D rotation pads mixed shapes with the median, not zero` | single-frame rotation padded with 0 while construction pads with the median, which shifted stats (mean 0.50 to 0.37 in the test case) | `tests/show2d` |
| `fix: Show2D.set_image copies a float32 array instead of aliasing it` | a caller reusing its buffer changed the widget's pixels | `tests/show2d` |
| `fix: show3dslices switches keep their aria-label under MUI 7` | MUI 7 `Switch` overrides `inputProps` with `slotProps.input`, so none of the 12 switches had a label | `js/show3dslices/contrast.test.ts` |
| `fix: show3dslices browser Align fits the active panel, not empty data` | the active-panel slice was taken again from an array that already held only that panel: Align fitted an empty array for panel 1 and above | `js/show3dslices/contrast.test.ts` |

## 2. Dead code

Evidence: coverage of the fast pytest suite and of the 14 executed tutorials
(separately), a caller trace of every package function during pytest (which code
called it: package, test or library), vulture, knip in production mode with the
bundles as the only entries, a synced-trait reader check (JS reads, Python writes)
and a caller grep over quantem.live (main checkout and the widget-cuda-stack
branch), quantem.thick, Live4DSTEM, Live4DSTEM-iOS, Live4DSTEM-linux and
quantem-tutorials.

| Name | Class | Action |
|---|---|---|
| ShowPtycho traits `c12_min`, `c12_max`, `phi12_min`, `phi12_max`, `initial_panel_size` | never set by Python or JS | JS constants (`C12_RANGE`, `PHI12_RANGE`, `DEFAULT_PANEL` 800, which the seed always set) |
| ChooseLattice trait `point_labels` | never set | JS constant `POINT_LABELS` |
| `show3dslices.apply_global_slice_alignment` | TEST-ONLY | deleted with its test lines; the estimator stays |
| Show3D `averageSupported` gate, `supportsClientAverage`, panel title style hook, forwarder of `temporalAverageFrameIndices` | always true / always `{}` | deleted |
| Show4DSTEM JS: `compareViLiveMicrotaskRef`, `drawDpLiveRef`, `sourceLoadAbort`, `computes`, forwarding wrappers, duck-typed DetectorCompute fallbacks, the interactive branch of the size estimate | write-only or unreachable | deleted |
| Show2D JS: six write-only refs (one allocated a full-frame buffer on every rebuild), constant-zero diff shift, an unreachable pan tail | write-only or unreachable | deleted |
| ShowPtycho JS: `finalizeFFTMag`, unread FFT fields, `sweepSteps` state | unused | deleted / constant |
| Show1D JS `plotTitleVisible = false` branches, Show3DSlices aliases | constant | deleted |
| display filter stretch-gamma stage and its pipeline, per-slot `histReadBuffer` | never called / never read | deleted |
| `show3d/playback.py` `_RGB_LUMA` copy, Show2D `_write_html_export` override (identical to the mixin's) | duplicate | deleted |

Kept, CONSUMER-USED: `free_gpu`, `gpu.gpu_info`, `gpu.vram_status` (quantem.live
`gpu.py`, notebooks), `folder_picker` (quantem.live `folder_picker.py`, browse
notebooks), `io.hdf5_family.collect_hdf5_family` (quantem.live
`server/routers/acquisitions.py`), `io.metadata.dataset_yaml.load_dataset_yaml` /
`resolve_condition` (quantem.live screen pipeline, routers, ptycho config),
`io.utils.atomic_write_json` / `locked_json_rmw` (quantem.live orchestrator),
`render.gif.finalize_frame` and `save_image` (quantem.live `export.py`,
`notebook_publish.py`). Every other function the tutorials do not run has a
package caller (CLI paths, browser request handlers, MPS-only paths) or is public
API documented in the tutorials or API pages.

Not dead after all: the Show3D `!offline` / per-frame `frame_bytes` branches the
2026-10-07 record listed were already removed on `platform`. knip's ten "unused
exports" are all used inside their own module and exported for tests.

Debug instrumentation shipped to users, kept pending Bob's decision (the perf
measurements read it): `window.__quantemShow2DPerf` (Show2D, about 110 lines),
`window.__quantemShow3DPerf` (Show3D, about 370 lines, and 8 lines in
`display/colormaps.ts`), `window.__quantemShow3DSlicesPerf` (Show3DSlices, about
165 lines); Show4DSTEM also carries `window.__sh4d*`, `__loadprof` and bench hooks.

## 3. Hygiene

Ten parallel passes, one per file set, each proving its own behaviour before the
merge. Per area (insertions / deletions):

| Area | Files | + / - |
|---|---:|---|
| Python Show2D, Show1D | 8 | 1022 / 1146 |
| Python Show4DSTEM, Show3D, Show3DSlices | 13 | 596 / 474 |
| Python ShowDiffraction, ShowPtycho, ShowCIF, Plot2D, ChooseLattice | 8 | 638 / 430 |
| Python shared modules, CLI, io, scripts | 37 | 749 / 853 |
| JS Show2D | 4 | 1231 / 1495 |
| JS Show4DSTEM | 3 | 1011 / 1271 |
| JS Show3D | 4 | 819 / 847 |
| JS Show1D, Show3DSlices | 3 | 888 / 1083 |
| JS ShowPtycho, ShowDiffraction, ShowCIF, Plot2D, ChooseLattice | 10 | 1030 / 1122 |
| JS display, shared, root modules | 21 | 1195 / 1772 |
| tests | 16 | 201 / 65 |

What changed, by rule: Python functions without a docstring went from 308 of 924
to 13 of 948 (the rest are trivial nested helpers); `typing.Any`,
`from __future__`, aliased stdlib imports and em-dashes are gone from `src/`,
`tests/` and `scripts/`; try/finally bookkeeping and duck typing were removed
where the type is known (the folder watcher keeps `hasattr` for stand-in widgets
in its tests, documented); single-letter and cryptic names became row/col and
full words in every file (rename-only JS commits were checked to produce the same
bundle up to identifier names); about 60 repeated multi-line blocks became one
helper each, about 90 in all (profile sparklines, FFT padding, colormap passes, panel quantization,
calibration clears, JPEG encoding in the CLI, ...); comments that restated code or
described removed code were deleted or rewritten. No public name, trait, model
key or signature changed.

## 4. Proof that numbers and pixels did not change

| Gate | Before (`388b93e`, after the bug fixes and dead-code commits) | After |
|---|---|---|
| fast pytest, live-env, GPU 0 | 418 passed, 25 skipped | 425 passed, 27 skipped (7 new tests, 2 new browser tests skip here) |
| output freeze: every synced trait, `state_dict`, static PNG and exported HTML (without the embedded bundle) of all 10 widgets through scripted interactions | recorded | identical except Show4DSTEM `export_status`, whose size label reads 1.0 MB instead of 1.1 MB because the bundle is 9 kB smaller |
| CPU-only venv (Python 3.12, torch 2.14.1+cpu, NumPy 2.5.3) | 391 passed, 51 skipped | 398 passed, 53 skipped; no DeprecationWarning from quantem |
| torch 2.2.2 + NumPy 1.26.4 venv, `test_optional_dependencies.py` | 5 passed | 5 passed |
| `npm run build`, `tsc --noEmit`, vitest | 10 bundles, 0 errors, 46 files / 220 tests | 10 bundles, 0 errors, 46 files / 221 tests |
| 14 tutorials through nbconvert | 0 errors | 0 errors; text outputs equal except timings, GPU memory and the widget-export figure, whose scale bar now reads 10 nm (the fix above) |
| browser tier (headed Chrome, NVIDIA adapter, WebGPU on and removed) | 24 passed | 26 tests: 24 passed, 2 intermittent (the Show3D inset-wheel image check, the Show4DSTEM live panel sizes); both files rerun twice: 12 of 12, as on the base |
| all-widget driver, 10 widgets, WebGPU on and off | every widget passes on and off (Plot2D and Show3DSlices on a rerun: a hover lost to pointer activity on the display, a kernel-connection race) | every widget passes on and off (ShowDiffraction off on a rerun: the same kernel-connection race); every checked step and panel equal to the before run (32 x 32 luminance grids: 0 difference, at most 0.06 of 255 on ShowCIF and Show3DSlices) except one Show2D hover step on WebGPU, 242 pixels along the readout box edge, which the base commit reproduces exactly when run again now: the display, not the code |

## 5. For Bob: proposed public renames (not done)

- Show3DSlices traits `slice_x`, `slice_y`, `nx`, `ny`, `nz` (x/y for array axes);
  the JS oblique segment points `{x, y}` that feed them.
- Show4DSTEM `shape_rows`/`shape_cols` to `scan_rows`/`scan_cols`, beside
  `det_rows`/`det_cols`.
- Show3D trait `size` (holds the constructor's `panel_width_px`) and the
  `_offline_*` traits (the embedded stack) to `_embedded_*`.
- Show3D `offline=` (accepted and ignored; the widget-export tutorial passes it),
  `export_html(quantized=)` and `mode=` aliases: keep only `encoding=`.
- `*_idx` traits (`selected_idx`, `page_idx`, `roi_selected_idx`, ...) to `*_index`;
  `normalise` / `normalize` spelled one way.
- ShowDiffraction `lattice.radial_profile_px` to `radial_profile`;
  `Phase.reflection_cache` private; `measurement_record`, `index_fields`,
  ShowPtycho `write_json`, `finite_or_none` private; `utils.array._resize_image`
  (it pads) to `_pad_to_shape`.
- ShowDiffraction `profile_theta_min`/`profile_theta_max`, `detect_denoise` and
  `calibration_rms_px` are synced but no JS reads them: unsync, or delete the
  sector-profile knob nothing sets.

## 6. Found and not fixed

- Rotating an RGB panel in Show2D rotates only the luminance copy; the RGB block
  keeps its shape, so the browser offsets are wrong.
- Show3D: the ROI preview scale bar, the profile axis and the FFT d-spacing assume
  Angstrom whatever `pixel_unit` says.
- Show3D stats, ROI stats, profile and FFT read the denoised or filtered display
  frame, not the raw frame.
- `Show4DSTEM.free()` keeps the offline pack (up to the 2 GB budget) and the image
  byte traits; Show3D's `free` clears its pack.
- `Show4DSTEM.poll_folder` recognises an empty folder by matching quantem.gpu's
  error text.
- Show4DSTEM JS: `CompareVirtualGrid` has `residentSource = false`, so the shared
  canvas path, its effects and `sharedCanvasLayout.ts` never run.

Update (later the same day): the per-widget round fixed four of these: Show3D stats, ROI stats,
profile and FFT now read the unfiltered frame; `Show4DSTEM.free()` releases the offline pack;
`poll_folder` lists the folder; the dead `CompareVirtualGrid` shared canvas is deleted.
- `display/colormaps.ts` `applySlots`, `applySingleWithLut`, `renderSlots` copy the
  whole mapped buffer into a `count * 4` array: a RangeError for any slot uploaded
  with a capacity hint (no current caller does).
- `display/geometry.ts` CPU `sampleLineProfile` divides by a fractional width the
  WebGPU path rounds (traits are integers today).
- ShowDiffraction copy-to-clipboard catches only synchronous errors, so a refused
  permission never reaches the download fallback; ShowPtycho's higher-order badge
  says the loss is not computed, which `_reconstruct` does compute.
- quantem.gpu carries its own copies of the display colormap, FFT, filter and
  geometry code, about 1,000 lines apart from these.
- 151 single-use private helpers remain (most are observers and request handlers,
  which must be functions); inlining the rest is a separate change.
