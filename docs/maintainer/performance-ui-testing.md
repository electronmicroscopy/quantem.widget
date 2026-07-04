# Performance UI Testing

Use this protocol when a change can affect load speed, FPS, large-data
interaction, notebook save/reopen, or export size. It is intentionally separate
from the storyboards: storyboards describe scientific behavior; this page
defines the real-data performance gate that agents must measure.

The default production topology is **MJGOAT or another workstation as the
backend** and the **Mac browser as the frontend**. Python/Jupyter owns file I/O,
large arrays, CUDA/MPS/CPU preprocessing, and export packing. The browser owns
canvas drawing, WebGPU, pointer events, playback, menus, and exported HTML.
Reports must not mix those timings.

## The Five Gates

### 1. Real Dataset Matrix

Do not sign off performance with synthetic data alone. Use real or
real-derived data and record the exact path, shape, dtype, native bytes, and
backend host.

Preferred heavy datasets on the lab machines:

| Gate | Widget | Minimum real-data target | Preferred source |
|---|---|---|---|
| PUI-2D-4K | Show2D | 8 panels, 4096 x 4096 | Drift/denoise/ptycho real-space outputs on MJGOAT |
| PUI-2D-BATCH | Show2D | 30 panels, 4096 x 4096 | High-throughput denoise or drift batch |
| PUI-2D-STRESS | Show2D | 45 to 85 panels, 4096 x 4096 | Optional stress pass when backend memory allows |
| PUI-3D-SINGLE | Show3D | 1 panel, at least 512 x 512 x 100 frames | Real time series, focal stack, or SSB iteration stack |
| PUI-3D-MULTI | Show3D | 12 panels x 32 frames x 2048 x 2048 source | Real-derived drift/ptycho/gold stack |
| PUI-3D-EXPORT | Show3D | same as PUI-3D-MULTI | Exact/quantized/binned HTML export paths |
| PUI-EDS | ShowEDS | native sparse EDS stream, no hidden crop/bin | DGGG 0039 or equivalent Velox EDS stream |
| PUI-4DSTEM | Show4DSTEM | real scan with diffraction and virtual images | 4D-STEM tutorial or paper data on MJGOAT |
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

1. Build or install the widget package on MJGOAT.
2. Launch Jupyter on MJGOAT or the active workstation.
3. Open the notebook or exported HTML from the Mac in the Codex in-app browser.
4. Drive the real UI from the Mac.
5. Store screenshots, timing JSON, and the final report outside the repo unless
   the user explicitly asks to commit them.

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

1. Start from `main`, rebuild the frontend, and launch the MJGOAT Jupyter
   backend with the patched checkout first on `PYTHONPATH`.
2. Generate or open one Show2D notebook for PUI-2D-4K/PUI-2D-BATCH and one
   Show3D notebook for PUI-3D-MULTI/PUI-3D-EXPORT.
3. Drive the in-app browser through all controls listed in Gate 3, including
   repeated fast gestures.
4. Save screenshots and a timing JSON under `/tmp/quantem-widget-perf-ui`.
5. Run `PYTHONPATH=src pytest -q` and `npm run build`, then attach the browser
   report before claiming the release is ready.

The browser report is the proof. Passing tests without a browser report is
`Not verified` for UI performance.
