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
| `scripts/widget_local_signoff.sh` | The repository is generally ready. | Size guards, frontend build, Python tests, HTML export smoke, optional browser/mobile/performance gates. | Default before saying a widget change is ready; `--quick --browser --mobile` for exported UI work. | Top-level `index.html` report under `/tmp/quantem-widget-local-signoff/...` or `--artifact-dir`. |
| `.github/workflows/widget-ci.yml` | CI can repeat the normal local signoff on clean Linux. | Same default local signoff path: build, tests, export smoke, docs build when not quick. | Automatically on PRs and pushes touching widget code/docs/tests. | GitHub Actions logs. |
| `scripts/widget_html_smoke.py` | Every export-capable widget can write standalone HTML with state. | `export_html()`, expected markers, file size, widget coverage matrix, browser-drive plan. | CI/default signoff; update when a widget gains or changes HTML export. | `index.html`, `report.json`, `browser-plan.json`, exported widget HTML files. |
| `scripts/widget_browser_smoke.py` | Exported HTML actually renders and responds in Chromium. | Nonblank canvases, wheel/drag interaction, switches, sliders, console/page/HTTP errors, `requestAnimationFrame` FPS, FFT state, storyboard IDs, optional mobile viewport. | `scripts/widget_local_signoff.sh --quick --browser`; add `--mobile` for narrow/touch layout changes. | `browser-smoke.html`, `browser-smoke-report.json`, screenshots. |
| `scripts/widget_performance_smoke.py` | Backend export packing and small real-data Show2D/Show3D payloads are measurable. | Real-data discovery, export time, output size, browser-drive plan. | `--performance` signoff or when checking export size/time trends. | `index.html`, `report.json`, `browser-plan.json`, exported real-data HTML. |
| `scripts/widget_heavy_perf_signoff.py` | Heavy Show2D/Show3D real-data browser performance is acceptable on lab data. | Local real-data discovery, heavy exports, browser FPS, nonblank render, screenshots, Show3D FFT overlay idle-cache guard. | Local-only HPC/workstation performance claims; never normal CI. | `index.html`, `heavy-signoff-report.json`, `browser-smoke-report.json`, screenshots under `/tmp`. |
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
| PUI-3D-EXPORT | Show3D | same as PUI-3D-MULTI | Exact/quantized/binned HTML export paths |
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

### 3. Browser FPS And Latency Measurements

Measure the interactions users feel. Unit tests and Python timers are not
enough.

For Show2D, drive and record:

- first paint for PUI-2D-4K and PUI-2D-BATCH,
- wheel zoom and drag pan at native/detail zoom,
- histogram min/max drag and center drag,
- linked zoom/pan/contrast across panels,
- column reflow through 1, 2, 4, 6, 8, and 12 columns,
- FFT toggle, FFT pan/zoom, and FFT reflow,
- panel hide/restore,
- export menu open and exported HTML reopen,
- `Cmd+S` save/reopen with compact visible output.

For Show3D, drive and record:

- first paint for PUI-3D-SINGLE and PUI-3D-MULTI,
- play at 30 FPS and verify image, frame label, histogram, and slider stay
  synchronized,
- high-FPS playback stress and slider lag,
- frame slider scrub, keyboard frame step, loop, bounce, and averaging,
- wheel zoom and drag pan with linked zoom on/off,
- column reflow through 1, 2, 4, 6, 8, and 12 columns,
- FFT bottom/right/overlay layouts,
- FFT overlay drag, corner snap, independent zoom, and pan,
- FFT cache behavior during scroll, resize, playback, and frame scrub,
- export exact, quantized, GIF, MP4, and binned quantized HTML where supported,
- `Cmd+S` save/reopen with compact visible output.

Targets:

- First useful paint: about 1 s for normal data; within a few seconds for heavy
  data. Over 10 s is a release blocker unless explicitly accepted.
- Pointer interactions: target 30 FPS or better. If a path cannot hit 30 FPS,
  the report must say which interaction, dataset, browser, and likely cause.
- Playback/sliders: target 30 FPS for heavy practical views. Slider and image
  must stay synchronized at the selected FPS.
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

- Current heavy Show3D export does not stream native detail tiles like Show2D.
- A binned Show3D HTML export is a compact visual report, not a promise that
  zooming returns exact native pixels.
- If native Show3D pixels are required, test a live workflow or focused view
  that can afford the native transfer.
- Any display bin or export bin must be explicit in the UI/report.

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

Release rule:

- `PASS`: all affected gates were driven after the last code change.
- `PASS WITH LIMITATION`: the limitation is specific, documented, and accepted.
- `BLOCKED`: first paint is too slow, 30 FPS interaction fails on practical
  heavy data, export produces an unusable artifact without a compact option,
  saved notebook output is blank, FFT is incorrect, or the test used synthetic
  data for a real-data claim.

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
- builds Show2D real 4K exports and a Show3D real-derived heavy FFT overlay
  export,
- runs `scripts/widget_browser_smoke.py` against the generated standalone HTML,
- checks browser FPS against the configured threshold,
- checks that Show3D FFT cache counters do not grow while the page is idle,
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

## Local Show4DSTEM Heavy Signoff

Use this command when a change touches Show4DSTEM loading, chunking, lazy
multi-master append, detector interaction, scan-position browsing, WebGPU
browser drawing, or standalone HTML export:

```bash
PYTHONPATH=src:. python scripts/widget_show4dstem_heavy_signoff.py \
  --search-root /path/to/local/real/4dstem/data \
  --backend cuda \
  --max-masters 2 \
  --det-bin 4 \
  --export-det-bin 4 \
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

Run two Show4DSTEM modes when memory allows:

```bash
# Practical browse path: detector-binned live data on NVIDIA/CUDA.
PYTHONPATH=src:. python scripts/widget_show4dstem_heavy_signoff.py \
  --search-root /path/to/local/real/4dstem/data \
  --backend cuda \
  --max-masters 2 \
  --det-bin 4 \
  --export-det-bin 4 \
  --min-fps 30

# No-bin backend path: full detector data in NVIDIA memory, compact export for sharing.
PYTHONPATH=src:. python scripts/widget_show4dstem_heavy_signoff.py \
  --search-root /path/to/local/real/4dstem/data \
  --backend cuda \
  --max-masters 1 \
  --det-bin 1 \
  --export-det-bin 8 \
  --min-fps 30
```

The no-bin pass is important because it exposes real resident memory pressure,
full-detector backend behavior, and virtual-detector latency.
Use a compact export bin for that pass unless the explicit goal is to measure a
large private standalone HTML payload.

Use `--backend mps` only for local MacBook fallback checks. It is not the
primary heavy signoff when an NVIDIA backend is available.

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
