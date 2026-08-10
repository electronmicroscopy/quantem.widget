# Performance UI Testing

Use this protocol when a change can affect load speed, FPS, large-data
interaction, notebook save/reopen, or export size. It is intentionally separate
from the storyboards: storyboards describe scientific behavior; this page
defines the real-data performance gate that agents must measure.

The default production topology is **an HPC/workstation with an NVIDIA GPU or
other lab compute backend** and the **local browser as the frontend**.
Python/Jupyter owns file I/O, large arrays, CUDA/MPS/CPU preprocessing, and
export packing. The browser owns canvas drawing, WebGPU, pointer events,
playback, menus, and exported HTML. Reports must not mix those timings.

## Maintainer Map

Use this table when you only need the high-level view of what each performance
or visual automation file proves. The point is to keep routine CI small, while
still having stronger browser, real-data, phone, and release gates when the
change needs them.

| File | What it proves | Main checks | When to run | Output |
| --- | --- | --- | --- | --- |
| `scripts/widget_local_signoff.sh` | The repository is generally ready. | Size guards, frontend build, Python tests, HTML export smoke, optional browser/mobile/performance gates; when `--browser --performance` are combined, it also drives the real-data performance exports in Chromium. | Default before saying a widget change is ready; `--quick --browser --mobile` for exported UI work; `--full --browser --performance` for release-facing real-data export UI proof. | Top-level `index.html` report under `/tmp/quantem-widget-local-signoff/...` or `--artifact-dir`. |
| `scripts/widget_signoff_dashboard.py` | The generated evidence can be reviewed from one page. | Pass/fail/skipped gate summary, failed gate list, next recommended runs, extracted FPS/load/export metrics, and links to raw reports. | Runs through local signoff/CI; rerun manually after dropping additional reports into the same artifact directory. | Top-level `index.html` everything dashboard plus `signoff-dashboard.json`. |
| `.github/workflows/widget-ci.yml` | CI can repeat the normal local signoff on clean Linux. | Same default local signoff path: build, tests, export smoke, docs build when not quick. | Automatically on PRs and pushes touching widget code/docs/tests. | GitHub Actions logs. |
| `scripts/widget_html_smoke.py` | Every export-capable widget can write standalone HTML with state. | `export_html()`, expected markers, file size, widget coverage matrix, browser-drive plan, small MoS2-like Show2D/Show3D lattice examples for meaningful visual review. | CI/default signoff; update when a widget gains or changes HTML export. | `index.html`, `report.json`, `browser-plan.json`, exported widget HTML files. |
| `scripts/widget_showfolder_live_smoke.py` | ShowFolder live-folder handoff stays centralized and functional. | Activates simultaneous all-image Show2D/Show3D through the public dual-view path, adds image files and `*_master.h5` markers, calls `watch_once()`, verifies active Show2D/Show3D/Show4DSTEM refresh, and writes reviewable exports. | CI/default signoff; update when ShowFolder watcher or selection handoff changes. | `showfolder-live/index.html`, `showfolder-live/report.json`, final ShowFolder/Show2D/Show3D/Show4DSTEM exports. |
| `scripts/widget_show3d_animation_smoke.py` | Show3D GIF exports are presentation-ready. | Dry-run size plan, multi-panel low/medium/high GIF previews, panel-gap control, live-style labels, scale bar, zoom readout, export seconds, file sizes, dimensions, frame count, frame-delta metric, optional local reference time-series source. | When GIF/MP4 animation export, PowerPoint sharing, or Show3D movie quality changes. | `index.html`, `report.json`, `show3d-timeseries-*.gif` unless `--dry-run`. |
| `scripts/widget_show3d_stress.py` | Show3D exact single-file HTML and folder sidecar paths stay responsive on real data. | Opens existing Show3D HTML, optionally generates a temporary folder sidecar from that same HTML, serves `offline_stack.u8` with HTTP Range, checks first nonblank paint, optional `--independent-contrast` mixed-product views, column reflow, FFT toggle, playback, zoom/pan stress, debug layout counters, transform render paths, FPS, console/network errors, and screenshots. | Local-only after Show3D render, layout, sidecar, WebGPU/canvas fallback, or large multi-panel interaction changes; never commit the generated report or generated sidecar. | `index.html`, `metrics.json`, screenshots, and optional generated sidecar under `/tmp` or `--artifact-dir`. |
| `scripts/widget_browser_smoke.py` | Exported HTML actually renders and responds in Chromium. | Nonblank canvases, wheel/drag interaction, switches, sliders, console/page/HTTP errors, `requestAnimationFrame` FPS, FFT state, storyboard IDs, optional mobile viewport. | `scripts/widget_local_signoff.sh --quick --browser`; add `--mobile` for narrow/touch layout changes. | `browser-smoke.html`, `browser-smoke-report.json`, screenshots. |
| `scripts/widget_performance_smoke.py` | Backend export packing and small real-data Show2D/Show3D payloads are measurable. | Real-data discovery, export time, output size, browser-drive plan. | `--performance` signoff or when checking export size/time trends. Add `--browser` to local signoff when those exports also need automated Chromium interaction/FPS proof. | `index.html`, `report.json`, `browser-plan.json`, exported real-data HTML; plus `browser-smoke.html` when driven through local signoff with `--browser`. |
| `scripts/widget_external_html_profile.py` | A standalone exported HTML report that already exists outside the repo remains interactive and fast. | Opens a provided URL, checks nonblank canvases, samples FPS, drives common Show3D page/play/hide controls when present, captures screenshots and console/page errors. | Local-only review of Tailscale-served or hosted real-data reports, especially when a user points to an existing exported HTML file. | `index.html`, `metrics.json`, screenshots under `/tmp` or `--artifact-dir`. |
| `scripts/widget_heavy_perf_signoff.py` | Heavy Show2D/Show3D real-data browser performance is acceptable on lab data. | Local real-data discovery, paged heavy exports, browser FPS, nonblank transition render, screenshots, page-scrub latency, hidden-panel persistence across page swaps, Show3D panel-independence checks, Show3D FFT overlay idle-cache guard, and FFT metric stats-toggle cache guard. | Local-only HPC/workstation performance claims; never normal CI. | `index.html`, `heavy-signoff-report.json`, `browser-smoke-report.json`, screenshots under `/tmp`. |
| `scripts/widget_show4dstem_heavy_signoff.py` | Heavy Show4DSTEM real-data loading, NVIDIA/CUDA backend memory, append/stack-growth, export, and browser interaction are acceptable on lab data. | Local 4D-STEM master discovery, CUDA first-load timing, backend memory report, append/stack-growth timing, dataset/frame flip FPS, virtual-detector drag FPS, scan-position FPS, browser WebGPU/backend split, GPU memory before/after. | Local-only Show4DSTEM performance claims; never normal CI. | `index.html`, `show4dstem-heavy-signoff-report.json`, exported Show4DSTEM HTML, browser screenshot under `/tmp`. |
| `scripts/widget_phone_handoff.py` | A human can verify physical phone Safari behavior with shared logs. | Serves report on `0.0.0.0`, prints Tailscale/HTTPS handoff command, records viewport/touch/pointer/WebGPU events. | Physical iPhone/iPad checks after browser smoke, especially WebGPU or touch changes. | Served report, `phone-probe.html`, `phone-events.ndjson`. |
| `scripts/widget_visual_signoff.sh` | Visual stories can be driven in Jupyter/browser before release. | Story-oriented widget drive packets, screenshots, selected release gates. | Broad UI or release-candidate work when a human/agent must drive real workflows. | Signoff packet/report under `/tmp` or configured artifact path. |
| `scripts/widget_agent_signoff.sh` | Agent-driven story redrive is structured and auditable. | Story IDs, issue observed, fix made, redriven evidence. | Before release candidates or after interaction-heavy fixes. | Agent signoff report. |
| `scripts/widget_release_check.sh` | The package can ship. | TypeScript typecheck/tests/build, standalone/offline browser build, Python compile, wheel build/content check. | Before tagging `widget-v*` release candidates. | Console logs and local `dist/widget-release-check` artifacts. |
| `tests/test_tutorial_data.py` | Tutorial data loaders work without heavy network downloads in CI. | Tiny monkeypatched Show2D/Show3D/Show4DSTEM datasets, ShowFolder fallback. | Normal pytest/CI. | Pytest result. |
| `tests/test_save_state.py` | Notebook state stays compact and visible instead of embedding huge arrays. | Saved widget MIME output, screenshot fallback, export state size behavior. | Normal pytest; important after save/export changes. | Pytest result. |
| `tests/test_widget_visual_jupyter.py` | Live Jupyter widgets can render nonblank canvases in browser-driven tests. | Jupyter display, nonblank canvas, browser interaction/FPS when enabled. | Opt-in browser/Jupyter visual verification. | Pytest/browser artifacts. |

Read the table left to right: unit tests and HTML export smoke protect protocol;
browser smoke protects rendered exported HTML; heavy signoff protects real-data
performance; phone handoff protects physical Safari/touch behavior; release
check protects packaging. Do not replace a stronger gate with a weaker one when
the user story depends on the stronger surface.

## The Five Gates

### 1. Real Dataset Matrix

Do not sign off performance with synthetic data alone. Use real or
real-derived data and record the exact path, shape, dtype, native bytes, and
backend host.

Preferred heavy datasets on the lab machines:

| Gate | Widget | Minimum real-data target | Preferred source |
|---|---|---|---|
| PUI-2D-4K | Show2D | 8 panels, 4096 x 4096 | Drift/denoise/ptycho real-space outputs on an HPC/workstation with NVIDIA GPU |
| PUI-2D-BATCH | Show2D | 30 panels, 4096 x 4096 | High-throughput denoise or drift batch |
| PUI-2D-STRESS | Show2D | 45 to 85 panels, 4096 x 4096 | Optional stress pass when backend memory allows |
| PUI-3D-SINGLE | Show3D | 1 panel, at least 512 x 512 x 100 frames | Real time series, focal stack, or SSB iteration stack |
| PUI-3D-MULTI | Show3D | 12 panels x 32 frames x 2048 x 2048 source | Real-derived drift/ptycho/gold stack |
| PUI-3D-EXPORT | Show3D | same as PUI-3D-MULTI | Full, uint8, and downsampled HTML export paths |
| PUI-EDS | ShowEDS | native sparse EDS stream, no hidden crop/bin | DGGG 0039 or equivalent Velox EDS stream |
| PUI-4DSTEM | Show4DSTEM | real scan with diffraction and virtual images | 4D-STEM tutorial or paper data on an HPC/workstation or hosted dataset |
| PUI-4DSTEM-NOBIN | Show4DSTEM | 30-40 ready real masters at `det_bin=1` as a capacity probe, plus a browser-enabled no-bin stack that fits | Private lab 4D-STEM masters on an NVIDIA workstation; never commit data or reports |
| PUI-FOLDER | ShowFolder | folder with many microscopy files | Real screening folder with cache reuse |

If a preferred source is unavailable, use the closest local real source and
mark the report `PASS WITH LIMITATION`. The report must say why the preferred
source was not used.

### 2. Backend And Frontend Topology

Every run must identify both sides:

- Backend: host, conda/uv environment, `quantem.widget` commit, Jupyter URL,
  data path, data shape, data construction time, export packing time.
- Frontend: Mac/browser, WebGPU adapter when relevant, page URL, viewport size,
  first visible paint, console errors, and whether the Jupyter kernel becomes
  busy during pointer-only interaction.

The normal heavy signoff is:

1. Build or install the widget package on the HPC/workstation backend.
2. Launch Jupyter on the HPC/workstation backend or the active local workstation.
3. Open the notebook or exported HTML from the Mac in the Codex in-app browser.
4. Drive the real UI from the Mac.
5. Store screenshots, timing JSON, and the final report outside the repo unless
   the user explicitly asks to commit them.

Real-data performance signoff is **local-only**. Do not upload lab workstation data,
absolute local paths, screenshots of private data, or heavy generated HTML to
GitHub. Normal CI must not depend on local lab data. CI owns lightweight protocol
and synthetic smoke coverage; the local heavy gate owns real data and browser
performance proof.

Standalone exported HTML is a separate surface. It must be tested in the
browser without assuming a Python kernel exists.

Rendered tutorial HTML is also a separate surface. A screenshot is not enough:
for interactive tutorials, execute the notebook, open the rendered HTML in a
browser, and drive the widget. For the Show4DSTEM tutorial, use:

```bash
PYTHONPATH=src:. python scripts/widget_tutorial_interactivity_smoke.py \
  docs/tutorials/show4dstem.ipynb \
  --artifact-dir /tmp/quantem-widget-tutorial-interactivity-smoke
```

The smoke fails if the rendered multi-panel widget does not update its
`DP at (...)` coordinate after a real drag. Keep these rendered HTML files and
screenshots under `/tmp` unless a sanitized artifact is explicitly requested.

When the real-data export already exists, profile the URL directly instead of
regenerating data:

```bash
PYTHONPATH=src:. python scripts/widget_external_html_profile.py \
  --url http://127.0.0.1:8779/path/to/exported-widget.html \
  --artifact-dir /tmp/quantem-widget-external-html-profile
```

Use this for Tailscale-served reports, overnight studies, and other external
HTML evidence. The report is local-only by default; do not commit private
screenshots, generated HTML, or metrics containing private paths unless the
user explicitly asks for a sanitized artifact.

### 3. Browser FPS And Latency Measurements

Measure the interactions users feel. Unit tests and Python timers are not
enough.

Show2D, Show3D, and Show4DSTEM support an internal debug overlay with
`debug=True`. It shows a compact `Debug UI FPS` badge in the widget title row
and is meant for live diagnosis in Jupyter and exported HTML. Agents should
turn it on when they need a quick, visible performance signal while driving a
widget:

```python
Show3D(stack, debug=True)
```

Keep it off by default, keep the sampler browser-local, and do not update
Python traits from the animation loop. The badge is one common metric: browser
paint-loop responsiveness. It does not measure Python load, GPU kernels, FFT
work, data decode, export packing, or cache behavior. A performance claim still
needs browser smoke, external HTML profile, or heavy signoff evidence with the
report path recorded in the final handoff.

Debug should scale as a telemetry protocol, not as one global number. New
widget-specific debug metrics should fit one of these buckets:

- UI: paint-loop FPS, dropped-frame count, pointer-to-preview latency.
- Decode: bytes read, transfer/decode time, frame fetch time, cache hit/miss.
- Draw: colormap, histogram, canvas/WebGL/WebGPU draw, overlay/layout time.
- Compute: FFT, virtual detector, EDS map/spectrum, Show1D profile, or other
  scientific computation time; separate CPU worker and GPU/WebGPU paths.
- Cache and memory: prewarm status, cache size, hit/miss counters, GPU memory
  label when available.
- Export/notebook: export packing time, HTML size, saved-output preview format,
  and notebook state size when relevant.

Expose these metrics in stable browser-local debug objects so agents can read
them without scraping text. Existing examples include
`window.__quantemShow3DPerf`, `window.__quantemShowEDSPerf`, and
`window.__quantemShow3DSlicesPerf`. Keep payloads small: latest value, rolling
average or last N samples, max, and counters are enough. The visible HUD should
show only the highest-signal summary; detailed values belong in the browser
object and the generated report.

FFT caching is part of the interaction contract for Show2D and Show3D. The
first FFT for a frame, panel set, ROI, and windowing state can be slow, but
returning to that same scientific input must reuse the cached magnitude so
slider scrub, page playback, zoom, pan, and metric labels stay real-time. Cache
keys should describe the FFT input (`frame`, data version, dimensions, visible
panels, layout, ROI, and windowing) and must not include delivery counters such
as traitlet frame-byte sequence numbers. Data-source changes should invalidate
the cache explicitly.

For Show2D, drive and record:

- first paint for PUI-2D-4K and PUI-2D-BATCH,
- wheel zoom and drag pan at native/detail zoom,
- histogram min/max drag and center drag,
- linked zoom/pan/contrast across panels,
- per-panel colormaps: pass a list of cmaps at construction, select/hover each
  panel, change one Color control, and verify saved state plus standalone HTML
  preserve independent panel maps,
- column reflow through 1, 2, 4, 6, 8, and 12 columns,
- paged panel sweeps: dragging or keyboard-stepping the Page slider should
  update the visible page from local browser state immediately, then commit the
  trait on the next animation frame,
- hidden panels in paged views should be page-slot based, so hiding panel slot 2
  keeps slot 2 hidden after moving to another page,
- FFT toggle, FFT pan/zoom, and FFT reflow,
- narrow mobile control wrapping: label/control pairs stay grouped for
  `Auto`, `Smooth`, `Zoom`, `Pan`, `Contrast`, and FFT controls, with no
  overlap or orphaned switch/menu controls,
- panel hide/restore,
- export menu open and exported HTML reopen,
- `Cmd+S` save/reopen with compact visible output.

For Show3D, drive and record:

- first paint for PUI-3D-SINGLE and PUI-3D-MULTI,
- play at 30 FPS and verify image, frame label, histogram, and slider stay
  synchronized,
- high-FPS playback stress and slider lag,
- frame slider scrub, keyboard frame step, loop, bounce, and averaging,
- remote Jupyter tunnel scrub for
  [S3D-20](storyboard-show3d.md#s3d-20-scrub-full-resolution-movies-over-a-remote-jupyter-tunnel):
  measure Python prepare/wire/encode/trait-set, browser receive/decode/paint,
  and UI latency over the real laptop-to-backend ssh tunnel; verify any
  drag-time preview announces its factor and that release restores native
  full-resolution pixels,
- paged panel sweeps: Page slider scrubs and Page play should not wait for a
  Python trait round trip before the rendered page changes,
- sidecar/manual scrub flash check: in exported Show3D HTML, zoom into a
  multi-panel paged sidecar view, scrub the lower frame slider by clicking,
  dragging the current-frame thumb, releasing, and then using Page play. The
  scientific canvas must keep the previous or next painted frame visible during
  every transition. Treat any full-panel black/white beat as a failure even if
  autoplay looks clean,
- hidden panels in paged views should be page-slot based and must remain hidden
  after Page slider scrubs, Page play, and page-label changes,
- wheel zoom and drag pan with linked zoom on/off,
- per-panel colormaps: pass a list of cmaps at construction, hover/select each
  panel, change one Color control, scrub frames, and verify saved state plus
  standalone HTML preserve independent panel maps without recoloring neighbors,
- column reflow through 1, 2, 4, 6, 8, and 12 columns,
- FFT bottom/right/overlay layouts,
- FFT overlay drag, corner snap, independent zoom, and pan,
- FFT cache behavior during scroll, resize, playback, and frame scrub; playback
  should recompute from the active displayed frame at a bounded cadence instead
  of freezing on the last paused FFT,
- FFT metric labels while toggling Stats, Profile, page playback, and panel
  visibility. These UI controls must not recompute FFTs or metric summaries
  unless the displayed scientific input changed,
- narrow mobile control wrapping: label/control pairs stay grouped for
  `Scale`, `Color`, `Auto`, `Smooth`, `Diff`, `fps`, `avg`, `Loop`, `Bounce`,
  FFT controls, and any page/column controls, with no overlap or orphaned
  switch/menu controls,
- export full, uint8, GIF, MP4, and downsampled uint8 HTML where supported,
- `Cmd+S` save/reopen with compact visible output.

Targets:

- First useful paint: about 1 s for normal data; within a few seconds for heavy
  data. Over 10 s is a release blocker unless explicitly accepted.
- Pointer interactions: target 30 FPS or better. If a path cannot hit 30 FPS,
  the report must say which interaction, dataset, browser, and likely cause.
- Playback/sliders: target 30 FPS for heavy practical views. Slider and image
  must stay synchronized at the selected FPS.
- Remote tunnel scrub: target immediate visual feedback during active drag.
  Native pixels must remain available after release or zoom/detail inspection.
  If a reduced preview is used during drag, the report must include the factor,
  preview bytes, native bytes, announcement text, and native-restoration check.
- Page sliders: target a visible page update within one animation frame. In
  heavy local signoff, page-scrub latency above 500 ms is treated as a failure
  unless the report documents a hardware or browser limitation.
- Mobile controls: labels and their switches, menus, sliders, or buttons must
  wrap as grouped pairs. A dense row may wrap onto another line, but it must
  not separate the label from the control it names.
- Export: menu labels must say the format and size class. Heavy standalone
  exports should offer compact choices rather than silently creating unusable
  hundreds-of-MB HTML.
- Save/reopen: notebook output must be visible after reload and must not embed
  heavy frame/detail/export buffers when compact state is requested.

### 4. Native Pixel And Preview Contract

Load fast first, then expose the highest resolution the widget contract
supports.

Show2D:

- `display_bin="auto"` may show a fast preview first.
- Native arrays stay on the backend.
- Zoomed inspection should stream a visible native/detail tile when available.
- Cursor readout reports native `(row, col)` and labels whether the value came
  from preview/detail/native data.
- Stale detail tiles must never draw after the view changes.

Show3D:

- Current heavy Show3D single-file export does not stream native detail tiles
  like Show2D.
- A Show3D folder export can preserve the native panel shape while keeping the
  large frame stack beside the HTML. Treat this as the full-shape browser review
  path, not as exact float32 source preservation unless the encoding says so.
- A binned/downsampled Show3D HTML export is a compact visual report, not a
  promise that zooming returns exact native pixels.
- If exact backend arrays, fresh computation, or non-exported precision are
  required, test a live workflow or focused view that can afford the native
  transfer.
- Any display bin, downsample, or export encoding must be explicit in the UI,
  report, and final handoff.

Show4DSTEM and ShowEDS:

- Native data should stay queryable or sparse where possible.
- Multi-master Show4DSTEM sessions must load quickly enough to become useful,
  then let the user flip through loaded datasets/frames from the browser at the
  target FPS. Timing first load without testing the dataset slider is not a
  complete signoff.
- A 30-40 master no-bin Show4DSTEM request is a memory-capacity gate first and
  a browser FPS gate only if the backend can actually keep that many masters
  available. Reports must state the per-master bytes, maximum loaded count,
  devices used, append failure if any, and cleanup result. Do not replace this
  with MPS when the requested backend is NVIDIA/CUDA.
- U8 no-bin browse timing and full Show4DSTEM UI signoff are different claims.
  It is valid to benchmark the direct HDF5 loader with
  `scripts/widget_load_bench_matrix.py` or
  `scripts/widget_load_bench_sharded.py`, but a release claim still needs the
  real Show4DSTEM widget/export/browser path. If `import quantem.widget` fails
  in the workstation environment, report the UI gate as blocked even when the
  direct HDF5 loader benchmarks pass.
- Multi-disk loading must be proven on a real multi-disk layout. A report where
  `group_by_disk(masters)` is `{'nvme2n1': N}` validates GPU sharding and
  capacity, but it does not validate aggregate disk bandwidth.
- Browser interaction should not require a Python round trip during drag unless
  the report calls out the limitation.
- WebGPU/MPS/CUDA usage must be recorded by surface: browser WebGPU is not the
  same as backend MPS/CUDA preprocessing.

### 5. Report And Release Decision

Every performance UI run writes a short report with this format:

```text
Performance UI report

Backend:
- Host:
- quantem.widget commit:
- Jupyter URL:
- Data path:
- Shape/dtype/native bytes:
- Backend build/export timings:

Frontend:
- Browser:
- URL/export file:
- Viewport:
- WebGPU adapter:
- Console errors:

Measured gates:
- First paint:
- Zoom/pan FPS:
- Histogram drag FPS:
- Slider/playback FPS:
- FFT FPS/cache notes:
- Export size/time/reopen:
- Save/reopen:

Result:
- PASS / PASS WITH LIMITATION / BLOCKED
- Story IDs covered:
- Screenshots/videos:
- Remaining risk:
```

For any automated run that writes multiple reports, open the top-level
everything dashboard first. It is the decision surface: failed gates, skipped
stronger gates, extracted metrics, and links to raw artifacts. The raw report
JSON remains the source of truth for exact numbers, but the dashboard is where a
human or agent should decide whether the claim is adequately proven.

Release rule:

- `PASS`: all affected gates were driven after the last code change.
- `PASS WITH LIMITATION`: the limitation is specific, documented, and accepted.
- `BLOCKED`: first paint is too slow, 30 FPS interaction fails on practical
  heavy data, export produces an unusable artifact without a compact option,
  saved notebook output is blank, FFT is incorrect, or the test used synthetic
  data for a real-data claim.

## Performance Evidence Registry

Do not commit raw heavy reports by default. Keep generated HTML, screenshots,
private file paths, timing JSON, and local browser artifacts in `/tmp` or CI
artifacts. Commit only distilled metadata when the result is useful as a future
baseline or protocol example.

Each evidence row should record:

- date and purpose,
- widget and scenario,
- dataset alias, not private path,
- shape, dtype, and native size,
- export mode, exported size, and export time,
- browser surface, FPS, and key interaction checks,
- cache counters or correctness checks that mattered,
- command family and local report path pattern,
- result and limitation.

Use three dataset tiers:

| Tier | Purpose | Storage policy |
| --- | --- | --- |
| CI tiny | Protocol coverage, export matrix, nonblank canvas, API behavior. | Commit only tiny generated fixtures or create them at test time. |
| Public real | Reproducible docs/tutorial/performance examples. | Prefer hosted datasets or small tracked examples when clone size stays reasonable. |
| Private heavy | Release-grade workstation/HPC proof. | Keep data, reports, screenshots, and private paths out of git; commit only alias-level summary. |

Current distilled evidence:

| Date | Evidence ID | Widget/scenario | Dataset alias | Shape/native size | Export | Browser/cache result | Command family | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-07-07 | `fft-metric-quick-cache-guard` | Show3D overlay FFT metric label, Stats toggle cache guard | `gold-haadf-local-quick` | 3 panels x 16 frames x 384 x 384, 27.0 MB native float32 | `uint8-4xbin`, 1.313 MB, 0.585 s | Browser smoke 121.3 FPS; FFT idle 120.7 FPS; idle FFT recompute growth 0; Stats-toggle FFT recompute growth 0; Stats-toggle FFT metric growth 0; label stayed `FFT sharp 13.2 · peaks 16 · SNR 14.9` | `scripts/widget_heavy_perf_signoff.py --quick --search-root <local-gold-dataset-root> --stats-toggles 4 --min-fps 1`; report pattern `/tmp/quantem-widget-heavy-signoff-fftmetrics-quick/index.html` | PASS as quick automation evidence; not a full release-grade heavy baseline because it used `--quick` and `--min-fps 1`. |
| 2026-07-07 | `fft-metric-full-cache-guard` | Show2D/Show3D heavy export, browser render, Show3D overlay FFT metric label, Stats toggle cache guard | `gold-haadf-local-full` | Show2D: 8 panels x 4096 x 4096, 512 MB native float32; Show3D: 12 panels x 32 frames x 2048 x 2048, 6144 MB native float32 | Show2D `uint8`, 11.326 MB, 0.141 s; Show2D `full`, 43.326 MB, 0.376 s; Show3D `uint8-4xbin`, 128.752 MB, 60.115 s | Browser smoke: Show2D `uint8` 120.3 FPS, Show2D `full` 120.2 FPS, Show3D 121.8 FPS with nonblank changing canvases; FFT idle 120.8 FPS; idle FFT recompute growth 0; Stats-toggle FFT recompute growth 0; Stats-toggle FFT metric growth 0 across 8 toggles; label stayed `FFT sharp 6.0 · peaks 516 · SNR 64.9` | `scripts/widget_heavy_perf_signoff.py --search-root <local-gold-dataset-root> --min-fps 30 --stats-toggles 8`; report pattern `/tmp/quantem-widget-heavy-signoff-fftmetrics-full/index.html` | PASS as local full heavy signoff evidence; raw report, screenshots, and private paths remain outside git. |

Keep quick rows only as automation debugging examples. A release or performance
claim should reference the strongest matching full local heavy signoff row.

## Minimal Agent Run

For the common Show2D + Show3D heavy gate, an agent should do the following:

1. Start from `main`, rebuild the frontend, and launch the HPC/workstation Jupyter
   backend with the patched checkout first on `PYTHONPATH`.
2. Run `PYTHONPATH=src:. python scripts/widget_performance_smoke.py` with
   real-data size options appropriate for the release gate. The script writes
   standalone exports, `index.html`, `report.json`, and `browser-plan.json`.
3. Serve the artifact directory and open the generated pages in the in-app
   browser.
4. Drive the in-app browser through all controls listed in Gate 3, including
   repeated fast gestures.
5. Save screenshots and browser timing JSON under `/tmp/quantem-widget-perf-ui`.
6. Run `PYTHONPATH=src pytest -q` and `npm run build`, then attach the browser
   report before claiming the release is ready.

The browser report is the proof. Passing tests without a browser report is
`Not verified` for UI performance.

## Local Heavy Signoff Command

Use this command on the lab workstation that can see the real data. It is the
preferred one-command signoff when a change touches Show2D/Show3D heavy
rendering, FFT overlays, exported HTML performance, or browser interaction
latency:

```bash
PYTHONPATH=src:. python scripts/widget_heavy_perf_signoff.py
```

By default it writes to:

```text
/tmp/quantem-widget-heavy-signoff/<timestamp>/
```

The heavy signoff:

- discovers local real microscopy images from common HPC/workstation data roots,
- builds paged Show2D real 4K exports and a paged Show3D real-derived heavy
  FFT overlay export,
- runs `scripts/widget_browser_smoke.py` against the generated standalone HTML,
- checks browser FPS against the configured threshold,
- checks that paged Show2D and Show3D keep hidden-panel state after page scrubs,
- checks that standalone Show3D page/frame transitions retain scientific pixels
  until the direct current-frame render is ready and that independent-panel
  contrast edits do not alter neighbors,
- checks that Show3D FFT cache counters do not grow while the page is idle,
- checks that a new Show3D FFT frame increments misses/computes once, then a
  return scrub to a previously computed frame increments hits while misses and
  computes stay unchanged,
- checks that Show3D FFT and FFT metric counters do not grow while toggling the
  Stats UI. Stats are display chrome; they must not invalidate cached FFT
  magnitudes or metric summaries,
- saves screenshots, command logs, `browser-smoke-report.json`, and
  `heavy-signoff-report.json`,
- writes `index.html` as the visual handoff report.

Use explicit roots when the default local data paths are not the active dataset:

```bash
PYTHONPATH=src:. python scripts/widget_heavy_perf_signoff.py \
  --search-root /path/to/local/real/microscopy/data
```

For repeated lab runs, set `QUANTEM_WIDGET_REAL_DATA_ROOTS` to one or more
local data roots separated by the platform path separator, then run the same
signoff command without hardcoding private paths in the repository.

Use `--quick` only while debugging the automation itself. A release or
performance claim needs the full local real-data run. Use `--skip-browser` only
to debug export generation; it intentionally reports that UI performance was
not fully verified.

This script is intentionally excluded from normal CI because it requires local
real data and can produce large private artifacts. Keep those artifacts under
`/tmp` or another ignored local directory.

## Heavy Compute UI Protocol

Use this protocol before adding any expensive browser-side computation such as
FFT metrics, peak finding, segmentation overlays, or live denoise previews.
The goal is to keep scientific feedback visible without dropping browser FPS.

Separate the work into four phases:

1. **Scientific input**: frame index, visible panel IDs, FFT window, ROI,
   sampling, units, display bin, and data version. These are the only values
   that should invalidate the expensive cache.
2. **Expensive compute**: FFT, peak finding, metric summaries, or GPU kernels.
   Cache these by a stable key derived from the scientific input. Do not put
   `show_stats`, labels, toolbar visibility, hover text, or layout chrome into
   that key unless they change the math.
3. **Display transform**: colormap, log/linear view, clipping, and annotations.
   This should reuse the cached numeric result whenever possible.
4. **Pointer UI**: hover labels, sliders, drag overlays, and toolbar state. Keep
   this path cheap: use refs, `requestAnimationFrame`, stable DOM nodes, and
   opacity/transform updates instead of recomputing arrays or forcing large
   React rerenders on every pointer event.

Paged Show2D/Show3D sliders are in the pointer/UI layer. Render the requested
page/frame through the one authoritative renderer, then batch only the synced
trait write with `requestAnimationFrame`. Hidden panels in paged viewers are
layout slots, not one absolute source index, so a hidden slot remains hidden as
the page changes. Keep the last complete scientific pixels visible until the
direct current page/frame paint is ready; do not maintain a separate prebuilt
canvas/composite display cache.

Show3D manual slider scrubs, slider release commits, keyboard steps, Page
slider changes, Page play, and autoplay must all obey the same no-blank and
panel-independence contract. Do not clear or hide a canvas, or fill a transformed
viewport with the inter-panel backing, unless replacement scientific pixels are
painted in the same task. With Link Contrast off, confirm an edit to one
histogram neither changes another panel's pixels nor rewrites its color range.

Every heavy compute feature needs both math proof and cache proof:

- Math proof: compare against NumPy, PyTorch, or another trusted reference on a
  deterministic microscopy-like fixture. For FFT metrics, the parity test is
  `js/fftMetrics.numpy.test.ts`.
- Cache proof: expose local debug counters for expensive work, for example
  `fftComputes`, `fftMetricComputes`, `lastFftMetricMs`, and the cache key. The
  browser signoff must assert those counters do not grow during cosmetic UI
  changes such as Stats toggles.
- Visual proof: open the rendered HTML or notebook in the browser, confirm the
  canvas is nonblank, and capture a report screenshot with the label/overlay
  visible.

Optimization order:

1. Cache first. Reuse display-sized arrays and cached FFT magnitudes before
   moving work to another thread or GPU path.
2. Downsample or summarize for labels. A quality label should not scan a full
   native 4K frame on every redraw when a sampled cached magnitude is enough.
3. Move real heavy kernels to WebGPU or a worker only when cached CPU work still
   shows up in the browser report.
4. Keep microbenchmarks local-only. Use them to compare algorithm cost, but do
   not make normal CI depend on workstation hardware, private data, or browser
   timing variance.

For the current FFT metric label, the accepted behavior is:

- Compute the label from the cached FFT magnitude, not from a second FFT.
- Recompute only when the FFT magnitude version, FFT crop dimensions, sampling,
  units, or visible FFT panel grid changes.
- Do not recompute when users toggle Stats, Profile, panel menu visibility,
  page playback controls, or other toolbar chrome.
- Verify the math with `npm test -- --run js/fftMetrics.numpy.test.ts
  js/fftMetrics.test.ts`.
- Verify the browser cache behavior with `scripts/widget_heavy_perf_signoff.py`;
  the report must include `show3d_fft_stats_toggle` with zero FFT and zero FFT
  metric growth.

## Local Show4DSTEM Heavy Signoff

Use this command when a change touches Show4DSTEM loading, chunking, lazy
multi-master append, detector interaction, scan-position browsing, WebGPU
browser drawing, or standalone HTML export:

```bash
PYTHONPATH=src:. python scripts/widget_show4dstem_heavy_signoff.py \
  --search-root /path/to/local/real/4dstem/data \
  --backend cuda \
  --max-masters 1 \
  --det-bin 1 \
  --export-det-bin 1 \
  --min-fps 30
```

By default it writes to:

```text
/tmp/quantem-widget-show4dstem-heavy-signoff/<timestamp>/
```

The Show4DSTEM signoff:

- discovers local ready ``*_master.h5`` files without committing those paths,
- measures CUDA first-master load time and widget build time on NVIDIA backends,
- records backend shape, dtype, device, resident memory, and memory before/after,
  and Python/GPU memory before and after each stage,
- measures additional masters through the active backend's append strategy:
  CUDA records eager stack-growth/reload timing, while MPS records live lazy
  append timing,
- exports standalone Show4DSTEM HTML with explicit ``uint8``/``uint16`` and
  detector binning labels,
- opens the export in Chromium, records browser WebGPU adapter information, and
  measures virtual-detector drag FPS, scan-position movement FPS, wheel-zoom FPS,
  and recompute latency,
- writes `show4dstem-heavy-signoff-report.json` and `index.html`.

Run the full-detector mode first when memory allows:

```bash
# Full detector path: this is the signoff gate for native detector behavior.
PYTHONPATH=src:. python scripts/widget_show4dstem_heavy_signoff.py \
  --search-root /path/to/local/real/4dstem/data \
  --backend cuda \
  --max-masters 1 \
  --det-bin 1 \
  --export-det-bin 1 \
  --min-fps 30

# Explicit preview path: only for a labeled capacity or automation diagnostic.
PYTHONPATH=src:. python scripts/widget_show4dstem_heavy_signoff.py \
  --search-root /path/to/local/real/4dstem/data \
  --backend cuda \
  --max-masters 2 \
  --det-bin 4 \
  --export-det-bin 4 \
  --min-fps 30
```

The no-bin pass is important because it exposes real resident memory pressure,
full-detector backend behavior, and virtual-detector latency. A detector-binned
pass is useful only as a labeled preview or capacity diagnostic; it does not
prove full-resolution Show4DSTEM behavior.

### Private Seven-Tilt Gate

Use this gate after Show4DSTEM backend, WebGPU, MPS, CUDA, export, or
high-frequency UI changes. The seven-tilt data are private: keep the real folder
path in a local environment variable, do not commit data or generated reports,
and do not paste raw JSON until checking that paths and screenshots are safe.

```bash
export MAC_SHOW4DSTEM_7TILT_DIR=/private/path/on/mac/to/512x512x192x192-seven-tilt-folder
export CUDA_SHOW4DSTEM_7TILT_DIR=/private/path/on/cuda-host/to/512x512x192x192-seven-tilt-folder
```

Do not copy the raw H5 data into the repo or reports. Use the seven entry
points below. They are full-detector gates: `--bin 1` means no detector
binning. Do not substitute a detector-binned run unless the report is explicitly
labeled as a quick diagnostic instead of signoff.

Before each machine's gate, build and install the checkout in the environment
that will launch the widget:

```bash
npm run build
python -m pip install -e .
python -c "import quantem.widget as w; print(w.__file__)"
```

```bash
# 1. macOS: Apple MPS backend, one master.
quantem show4dstem "$MAC_SHOW4DSTEM_7TILT_DIR" --backend mps --count 1 --bin 1 --dtype u8

# 2. macOS: Apple MPS backend, seven masters.
quantem show4dstem "$MAC_SHOW4DSTEM_7TILT_DIR" --backend mps --count 7 --bin 1 --dtype u8

# 3. macOS: browser/WebGPU HDF5-backed lane, one master.
quantem show4dstem "$MAC_SHOW4DSTEM_7TILT_DIR" --backend webgpu --html --count 1 --bin 1 --dtype u8

# 4. macOS: browser/WebGPU HDF5-backed lane, seven masters.
quantem show4dstem "$MAC_SHOW4DSTEM_7TILT_DIR" --backend webgpu --html --count 7 --bin 1 --dtype u8

# 5. CUDA host: CUDA backend, one master.
quantem show4dstem "$CUDA_SHOW4DSTEM_7TILT_DIR" --backend cuda --count 1 --devices 0 --bin 1 --dtype u8

# 6. CUDA host: CUDA backend, seven masters.
quantem show4dstem "$CUDA_SHOW4DSTEM_7TILT_DIR" --backend cuda --count 7 --devices 0 --bin 1 --dtype u8

# 7. CUDA host: CUDA backend, seven masters split across two GPUs.
quantem show4dstem "$CUDA_SHOW4DSTEM_7TILT_DIR" --backend cuda --count 7 --devices 0,1 --bin 1 --dtype u8
```

The MPS and CUDA entries launch live notebook-backed widgets. The WebGPU
entries create anonymous H5 symlinks inside the artifact folder, write an
interactive browser HTML, and leave a `Show4DSTEM.command` launcher in the
artifact folder. After launch, an agent should drive the UI in the desktop
browser or Jupyter: switch datasets, move scan positions, drag detector ROIs, adjust
histogram/contrast, check save/load where relevant, reload/reopen once, and
record screenshots, console errors, backend/device shape, and timing/FPS. The
gate is still local-only; normal CI must not depend on these data.

Use `--backend mps` outside this gate only for local MacBook fallback checks. It
is not the primary heavy signoff when an NVIDIA backend is available.

For the high-risk capacity test the user cares about, run the backend-only
stress first so a too-large request fails cleanly and releases memory:

```bash
PYTHONPATH=src:. python scripts/widget_show4dstem_heavy_signoff.py \
  --search-root /path/to/local/real/4dstem/data \
  --backend cuda \
  --devices 0,1 \
  --max-masters 30 \
  --det-bin 1 \
  --export-det-bin 8 \
  --skip-browser \
  --min-fps 30
```

This intentionally records CUDA memory before load, after load or OOM, and
after `free_gpu()` cleanup. A 20-30 file no-bin stack can exceed even a
two-GPU workstation because a single 512 x 512 x 192 x 192 uint16 master is
about 18 GiB resident before transient decompression overhead. If this stress
does not fit, the report should say where it failed and prove GPU memory was
returned before the next run. After the capacity pass, run a smaller no-bin
browser pass without `--skip-browser` to verify user interaction remains smooth.

Use `QUANTEM_WIDGET_4DSTEM_ROOTS` or `QUANTEM_WIDGET_REAL_DATA_ROOTS` to avoid
hardcoding private data roots in commands. Use `--quick` only while iterating on
the signoff script itself. Use `--skip-browser` only for backend/export
debugging; it intentionally reports that UI performance was not fully verified.
