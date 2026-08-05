# Automation

Use this page to choose the right readiness gate for widget work. Automation
should stay small, repeatable, and inspectable. `scripts/widget_local_signoff.sh`
is the general entrypoint; other scripts exist for focused reports or stronger
real-data proof.

## Which Command Should I Run?

| Situation | Command | Evidence |
| --- | --- | --- |
| Small Python, docs, or protocol change | `scripts/widget_local_signoff.sh --quick` | Everything dashboard, size guards, frontend build, focused tests, HTML export smoke report. |
| Normal widget change before saying it is ready | `scripts/widget_local_signoff.sh` | Everything dashboard, full tests, HTML export smoke report, docs build. |
| ShowFolder or direct Show2D/Show3D/Show4DSTEM folder-watcher change | `PYTHONPATH=src:. python scripts/widget_showfolder_live_smoke.py --artifact-dir /tmp/quantem-widget-showfolder-live` | Lightweight lifecycle report proving probation, stable arrival, same-model updates, stop/static safety, and fresh Show4DSTEM visible-page paint; also retains ShowFolder handoff, thumbnail, and master-QC evidence. |
| Widget UI, interaction, or HTML export change | `scripts/widget_local_signoff.sh --quick --browser` | Everything dashboard, exported HTML plus automated Chromium report, screenshots, nonblank canvas checks, FPS. |
| Show3D GIF/PowerPoint animation review | `PYTHONPATH=src:. python scripts/widget_show3d_animation_smoke.py --artifact-dir /tmp/quantem-widget-show3d-gif` | Visual GIF report with low/medium/high quality previews, export seconds, file sizes, dimensions, frame count, and source data. |
| Mobile layout or touch-sensitive change | `scripts/widget_local_signoff.sh --quick --browser --mobile` | Desktop and 390x844 touch Chromium pre-check. |
| Existing standalone HTML report or Tailscale-served real-data export | `PYTHONPATH=src:. python scripts/widget_external_html_profile.py --url http://127.0.0.1:8779/path/to/export.html --artifact-dir /tmp/quantem-widget-external-html-profile` | Browser profile of the existing export: first ready time, nonblank canvases, screenshots, common Show3D play/page/hide interactions, console/page errors, and FPS. |
| Release-oriented validation | `scripts/widget_local_signoff.sh --full --performance` | Full local gates, frontend typecheck/tests, release check without wheel, real-data export timing report. |
| Real-data export plus browser validation | `scripts/widget_local_signoff.sh --full --browser --performance` | Full local gates plus Chromium drive of both the normal HTML smoke matrix and the real-data Show2D/Show3D performance exports. |
| Heavy real-data performance claim | `PYTHONPATH=src:. python scripts/widget_heavy_perf_signoff.py` | Local-only HPC/workstation real-data browser FPS, screenshots, paged Show2D/Show3D scrub and hidden-panel persistence, Show3D no-blank/panel-independence checks, FFT idle-cache and return-scrub cache reports, and FFT metric stats-toggle cache report. |
| Heavy Show4DSTEM real-data claim | `PYTHONPATH=src:. python scripts/widget_show4dstem_heavy_signoff.py --backend cuda` | Local-only real 4D-STEM NVIDIA/CUDA load, memory, append/stack-growth, export, browser WebGPU/FPS report. |
| Raw Show4DSTEM loader speed or U8/U16 claim | `PYTHONPATH=src:. python scripts/widget_load_bench_matrix.py` | Local-only private real-data cold/warm `load(...)` table with parity outside the timer. |
| Multi-disk / two-GPU Show4DSTEM load claim | `PYTHONPATH=src:. python scripts/widget_load_bench_sharded.py --devices 0,1` | Local-only private real-data disk layout, sharded GPU placement, cold/warm table, and capacity boundary. |
| Physical iPhone Safari behavior | `python scripts/widget_phone_handoff.py /tmp/quantem-widget-signoff` | Tailscale/HTTPS handoff URL plus phone viewport, touch, pointer, and WebGPU logs. |
| Local docs visual review | `scripts/docs_preview.sh` | Rendered documentation site in a browser. |

Use `--artifact-dir /tmp/quantem-widget-signoff` when coordinating with another
agent or serving the report over Tailscale.
When served locally, open the report URL in the browser, for example
`http://127.0.0.1:8779/index.html`.
Use `--search-root`, `QUANTEM_WIDGET_REAL_DATA_ROOTS`, or
`QUANTEM_WIDGET_4DSTEM_ROOTS` for private lab data locations instead of
hardcoding machine-specific paths in shared scripts.
Use `QUANTEM_WIDGET_BENCH_MASTERS_GLOB` for raw loader benchmarks. It can contain
multiple `:`-separated globs when a dataset was split across several disks.

## Verification Data

Use reconstructed experimental data for end-to-end widget, browser, visual,
and performance claims whenever an appropriate dataset is available. Record
the source description, shape, dtype, transforms, backend, and every crop,
binning, downsample, or encoding choice in the report. Keep private source
paths and raw data outside git.

Synthetic data is appropriate for focused unit tests, deterministic failure
cases, and small CI protocol checks. It does not establish real-workflow or
real-data performance signoff; label the limitation when synthetic data is the
only available evidence.

## Definition Of Done

A widget automation task is done only when:

- The command from the table above that matches the change has passed.
- The final answer names the exact command and report path.
- A human-readable report exists, usually `index.html`, `browser-smoke.html`, or
  a focused performance report.
- Browser-sensitive claims are backed by browser evidence, not Python tests
  alone.
- Browser artifacts were cleaned by `scripts/cleanup_browser_artifacts.py`.
- No large data files, generated HTML payloads, screenshots, scratch notebooks,
  or notebook output noise were added to git unless the user explicitly asked.
- Any limitation is stated plainly, including missing real data, skipped browser
  checks, non-physical mobile emulation, or unavailable hardware.

## Do Not Do This

- Do not claim browser behavior from Python unit tests alone.
- Do not use synthetic data for real-data performance claims.
- Do not run `pkill chrome` or kill normal interactive Chrome/Brave sessions.
- Do not commit `/tmp` reports, generated screenshots, generated HTML, private
  local paths, large arrays, or large notebook outputs.
- Do not run `quantem github` directly on source notebooks under
  `docs/tutorials/`.
- Do not add a new automation script unless the existing scripts cannot
  reasonably own the behavior.
- Do not replace a stronger gate with a weaker one when the user story depends
  on browser, mobile, physical device, or real-data behavior.

## Report Artifacts

Every automated run should produce an artifact that a human can open. Logs are
not enough for visual widgets. The top-level file is always:

```text
index.html
```

That file is generated by `scripts/widget_signoff_dashboard.py`. It is the
single-page everything dashboard for both humans and agents. Open it first. It
contains run metadata, failed gates, next recommended commands, evidence gates,
measured performance rows, and direct links to every generated report.

The paired machine-readable file is:

```text
signoff-dashboard.json
```

It contains the manifest, status counts, failed gate names, next recommended
actions, and normalized gate rows. Use this JSON when an agent needs to compare
runs or decide which stronger gate is still missing.

Default local signoff writes:

```text
/tmp/quantem-widget-local-signoff/<timestamp>/
```

The normal HTML smoke report contains:

- `index.html`: visual table with links to every standalone widget export.
- `report.json`: export sizes and backend timings.
- `browser-plan.json`: pages and interactions an agent should drive when manual
  browser signoff is needed.
- `showfolder-live/index.html` and `showfolder-live/report.json`: generated by
  local signoff as the ShowFolder live-folder smoke to prove both ShowFolder orchestration and direct public
  `Show2D.from_folder`, `Show3D.from_folder`, and `Show4DSTEM.from_folder`
  lifecycle behavior. The report records same-model state timelines, confirms
  stopped static exports do not retain false green `Watching`, and writes a
  browser-smoke-compatible `browser-plan.json`. It also displays WebP thumbnail
  previews and header-only 4D-STEM master QC rows.
- `browser-smoke.html` and `browser-smoke-report.json`, when `--browser` is
  used: automated Chromium results, screenshots, nonblank canvas checks, FPS,
  semantic control checks, and storyboard IDs.

Performance mode writes real-data Show2D and Show3D HTML exports plus its own
`index.html`, `report.json`, and `browser-plan.json`. When `--browser` is also
enabled, local signoff additionally writes `performance/browser-smoke.html` and
`performance/browser-smoke-report.json`, proving those real-data exports render,
respond, and meet the configured FPS threshold in Chromium. The top-level
dashboard reports this as `Real-data Show2D/Show3D browser smoke`. Heavy
performance signoff writes `heavy-signoff-report.json`, browser screenshots,
and an `index.html` summary. Keep real-data artifacts outside git.

External HTML profile writes:

- `index.html`: human-readable profile for an already exported widget report.
- `metrics.json`: URL, load timing, FPS samples, interaction results, console
  warnings/errors, and page errors.
- `screenshots/`: viewport and interaction screenshots for visual review.

Use this when the important evidence is an exported HTML file hosted elsewhere
rather than a new export generated by the current checkout.

Distilled long-lived baselines belong in the
`Performance Evidence Registry` in [Performance UI Testing](performance-ui-testing),
not as committed raw reports. The current full local heavy baseline is
`fft-metric-full-cache-guard` on the `gold-haadf-local-full` dataset alias; the
`fft-metric-quick-cache-guard` row on `gold-haadf-local-quick` exists only as
an automation debugging example.

Show3D GIF presentation smoke writes:

- `index.html`: PowerPoint/email-oriented GIF preview report.
- `report.json`: source, shape, native size, FPS, playback mode, quality tier,
  export time, GIF dimensions, frame count, file size, and frame-delta metric.
- `show3d-timeseries-*.gif`: low/medium/high GIF exports when the local
  reference time-series folder is present, otherwise the selected fallback source.

## Browser Cleanup

Browser tests must clean up after themselves:

```bash
python scripts/cleanup_browser_artifacts.py
```

The janitor removes stale QuantEM/Playwright/Chrome temp profiles and logs under
`/tmp` after they are older than six hours. It intentionally avoids global
Chrome killing and does not touch normal interactive browser sessions. Browser
automation should use isolated temporary profiles, close browser contexts in
`finally`, and leave only report artifacts needed for review.

If a browser process is launched manually, write a PID file under
`/tmp/quantem-widget-browser-pids` so the janitor can terminate only that known
browser process if a later run crashes.

`scripts/widget_local_signoff.sh` runs stale browser artifact cleanup at the
start of every signoff and again after `--browser` or `--performance` gates.
Keep this cleanup path in the protocol to prevent memory leaks, stale browser
caches, and zombie test profiles from changing future test results.

## CI Signoff

`.github/workflows/widget-ci.yml` runs local signoff on pull requests and pushes
that touch widget source, tests, scripts, docs, package files, or workflow
files. Keep CI aligned with `scripts/widget_local_signoff.sh`; if CI fails but
local signoff passes, treat it as a dependency or environment drift bug.

CI should upload the signoff artifact directory with `actions/upload-artifact`
so humans and agents can download `index.html`, `report.json`,
`browser-plan.json`, and any generated smoke reports from GitHub Actions. The
default CI gate should stay lightweight. Manual `workflow_dispatch` runs expose
inputs for quick/full mode, browser smoke, mobile Chromium pre-check,
performance smoke, and docs skipping. Combining browser and performance inputs
drives the generated real-data performance exports in Chromium. Browser-heavy,
mobile, physical phone,
and local real-data gates remain explicit opt-in gates unless the project
decides to pay that runtime cost.

## HTML Export Smoke

Generate standalone HTML for the export-capable widgets:

```bash
PYTHONPATH=src:. python scripts/widget_html_smoke.py
```

The smoke covers `Show2D`, `Show3D`, `Show3DSlices`, `Show4DSTEM`, `ShowEDS`,
`ShowDiffraction`, and `ShowFolder`. `Show2D` and `Show3D` use a small settings
matrix: single panel, multi-panel galleries, FFT modes, hidden/starred panels,
compact/no-title layouts, panel visibility, multi-panel Show3D, and
downsampled Show3D export. This checks export protocol coverage with tiny data,
not heavy interaction performance.

Use microscopy-like synthetic data for the tiny Show2D/Show3D cases. The
current smoke uses a MoS2-like HAADF lattice phantom, which keeps CI small while
making the visual report useful for reviewing atomic contrast, FFT peaks, scale
bars, and panel layout. Do not replace these with smooth blobs or plain random
noise unless the test is specifically about those inputs.

Update this smoke when a new public widget gains `export_html()`, or when an
existing widget's canonical small export options change.

## ShowFolder Live-Folder Smoke

Run the focused live-folder handoff proof:

```bash
PYTHONPATH=src:. python scripts/widget_showfolder_live_smoke.py \
  --artifact-dir /tmp/quantem-widget-showfolder-live
```

This script creates tiny generated images and HDF5 masters. It preserves the
ShowFolder all-image Show2D/Show3D and lazy Show4DSTEM handoff checks, then
mounts the three direct public `from_folder` viewers and records their lifecycle
timelines. The direct checks prove initial or arrival probation, one unchanged
follow-up poll, same Python/model identity, `Waiting`/`Updating`/`Watching`/
`Stopped`, stopped saved/static state without false green, and authoritative
Show4DSTEM active-page pixels before the final green state. Direct Show4DSTEM
uses the native GPU loader over tiny bitshuffle-LZ4 external-link masters; the older
ShowFolder-only 4D handoff explicitly labels its monkeypatched tiny loader.

The report writes top-level `exports` rows and `browser-plan.json` in the schema
accepted by `scripts/widget_browser_smoke.py`. Browser driving remains a
separate opt-in step; the default lifecycle smoke launches no browser.
Keep ShowFolder's real cache as numeric arrays for widget handoff. WebP belongs
in reports, dashboards, and other visual review surfaces, not in the selection
or data cache.

## Show3D GIF Presentation Smoke

Run the focused animation-export report when the question is whether a Show3D
movie is good enough for PowerPoint, email, or a quick supplement:

```bash
PYTHONPATH=src:. python scripts/widget_show3d_animation_smoke.py \
  --artifact-dir /tmp/quantem-widget-show3d-gif
```

When the data may be large, start with the dry run:

```bash
PYTHONPATH=src:. python scripts/widget_show3d_animation_smoke.py \
  --artifact-dir /tmp/quantem-widget-show3d-gif-dry-run \
  --dry-run
```

The dry run writes `index.html` and `report.json` but does not write GIF files.
It reports the source file size when known, the native input size, the derived
multi-panel array size, planned GIF dimensions, frame count, and projected
uncompressed RGB work for each quality tier. Read the `Decision` line first:
`Run the full GIF export` means the size plan is reasonable; `Reduce crop size,
frame count, or quality tiers` means adjust `--crop-size`, `--frames`, or
`--qualities` before spending time on the full export. Use `--max-native-mb` and
`--max-work-mb` to tighten or relax those dry-run warnings.

The default `auto` source uses local reference Show3D PNG/TIFF time-series frames
when they are available, then falls back to the public tutorial Show3D data, and
finally to a tiny synthetic CI fallback. Use `--source timeseries`, `--source
tutorial`, or `--source synthetic` when the report must be explicit.

The report is multi-panel by default (`Raw`, `Smoothed`, `Change`) so it checks
the case scientists usually paste into slides. It deliberately keeps live-style
panel labels, scale bars, and the `1.0x` zoom readout in the GIF. Use
`--panel-gap 0` when the review needs edge-to-edge panels with no whitespace
between images. Use `--no-panel-labels`, `--no-frame-labels`,
`--no-scale-bar`, and `--no-zoom-label` only when the test is deliberately
checking a cleaner output. The report prints those controls near the top so a
reader can tell whether labels were supposed to be visible.

The script exports GIF low, medium, and high. GIF quality means spatial
resolution: the palette remains GIF-limited, while `low`/`medium`/`high` trade
slide sharpness against file size. Use `--sampling-nm` to set the display
sampling used for the scale bar in this review report.

## Browser HTML Smoke

Drive generated standalone HTML in Chromium:

```bash
PYTHONPATH=src:. python scripts/widget_browser_smoke.py --artifact-dir /tmp/quantem-widget-signoff/html-smoke
```

This opens every page from `report.json`, verifies visible canvases are
nonblank, performs wheel/drag interactions, toggles visible switches, drags a
slider when one is present, records console/page/HTTP errors, samples
`requestAnimationFrame` FPS, records storyboard IDs, and saves screenshots.

Use mobile and FPS options when layout or performance changed:

```bash
PYTHONPATH=src:. python scripts/widget_browser_smoke.py \
  --artifact-dir /tmp/quantem-widget-signoff/html-smoke \
  --mobile \
  --min-fps 30
```

The 390x844 mobile mode is Chromium emulation. It is useful for automated
layout checks, but it is not proof of physical iPhone Safari behavior.

## Real Data And Physical Device Gates

Use `scripts/widget_heavy_perf_signoff.py` for a complete browser-driven
heavy-data proof. It wraps export performance smoke, runs browser drive, samples
FPS, records screenshots, asserts that paged Show2D/Show3D hidden-panel state
survives page scrubs, records Show3D standalone no-blank and panel-independence checks,
asserts that Show3D FFT overlay cache counters do not grow while idle, asserts
that a return scrub to a previously computed FFT frame hits cache without new
misses/computes, and asserts that toggling Stats does not recompute cached FFT
metric labels. Keep it out of normal CI because it depends on local real
datasets and can generate private heavy files.

Use `scripts/widget_show4dstem_heavy_signoff.py --backend cuda` for
Show4DSTEM. It discovers local real 4D-STEM masters, measures NVIDIA/CUDA
first-load time, backend shape/dtype/device and memory pressure, append or
stack-growth time for new masters, standalone HTML export time/size, browser
WebGPU adapter information, detector-drag FPS, scan-position movement FPS,
virtual-detector recompute latency, and GPU memory before/after. Keep it out of
normal CI and keep generated reports under `/tmp` unless a release explicitly
asks for them.

Use `scripts/widget_load_bench_matrix.py` and
`scripts/widget_load_bench_sharded.py` when the question is specifically loader
throughput, exact `uint16` versus browse `uint8`, disk layout, or two-GPU
capacity. These scripts write Markdown reports to
`/tmp/quantem-widget-load-bench/` by default and run real cases in subprocesses
so CUDA pools and page-cache state from one case do not leak into the next.
They are not browser signoff: after loader speed is acceptable, still run the
Show4DSTEM heavy signoff to prove WebGPU interaction, detector dragging,
scan-position movement, dataset flipping, and export reopen behavior.

Use `scripts/widget_phone_handoff.py` after a signoff run when the user asks for
real iPhone behavior. Serve the existing report directory, open the printed
Tailscale URL on the phone, and keep generated phone logs outside git unless a
release explicitly needs them.

## Docs Preview

Build and serve local documentation:

```bash
scripts/docs_preview.sh
```

To serve an existing build:

```bash
scripts/docs_preview.sh --no-build
```

Use this for rendered documentation feedback. Do not commit `docs/_build`.

## Size Guards

Keep the main branch lightweight:

```bash
python scripts/check_large_files.py
python scripts/check_notebook_sizes.py
```

These fail when tracked files, tutorial notebooks, or embedded notebook outputs
grow beyond the project thresholds. Large arrays and large HTML payloads should
be generated during docs builds or downloaded from public data hosting only when
the size justifies it.

## Script Ownership

| File | Why it exists | Runs by default | Update when |
| --- | --- | --- | --- |
| `scripts/widget_local_signoff.sh` | One local command for normal readiness gates. | Yes, in CI and local signoff. | The project-wide readiness definition changes. |
| `scripts/widget_signoff_dashboard.py` | Build the top-level `index.html` and `signoff-dashboard.json` that summarize all generated evidence in one place. | Yes, through local signoff and CI artifacts. | Report locations, gate status rules, or human evidence expectations change. |
| `scripts/cleanup_browser_artifacts.py` | Remove stale QuantEM/Playwright/Chrome temp profiles, logs, and known browser PID files without killing normal user browser sessions. | Yes, through local signoff. | Browser automation changes temp profile naming, PID tracking, or cleanup policy. |
| `scripts/check_large_files.py` | Prevent accidental large tracked data or rendered artifacts. | Yes. | File-size policy or allowlisted artifact types change. |
| `scripts/check_notebook_sizes.py` | Keep tutorial notebooks and embedded outputs clone-friendly. | Yes. | Notebook size policy changes. |
| `scripts/widget_html_smoke.py` | Verify every export-capable widget writes standalone HTML state, a visual report, and a browser-drive plan. | Yes. | A widget adds/removes/changes HTML export support. |
| `scripts/widget_showfolder_live_smoke.py` | Prove ShowFolder live folder watching refreshes simultaneous all-image Show2D/Show3D and active Show4DSTEM handoffs through public APIs, then writes a reviewable report. | Yes, through local signoff and CI. | ShowFolder watcher behavior, selection handoff, or Dataset4DSTEM master-folder refresh changes. |
| `scripts/widget_browser_smoke.py` | Open generated HTML exports in Chromium, check nonblank canvas rendering, semantic controls, FPS, storyboard coverage, desktop/mobile viewport behavior, and save screenshots. | No, only `--browser`. | Exported widget browser behavior, interaction contracts, mobile layout expectations, FPS thresholds, or report format changes. |
| `scripts/widget_phone_handoff.py` | Serve a signoff/report directory on `0.0.0.0`, print local/Tailscale URLs, and record physical phone viewport/touch probe logs. | No, manual physical-device handoff only. | Physical phone test workflow or Tailscale handoff expectations change. |
| `scripts/widget_performance_smoke.py` | Record real-data Show2D/Show3D export timing, payload sizes, report HTML, and browser-drive plan. | No, only `--performance`. | Real-data performance expectations change. |
| `scripts/widget_heavy_perf_signoff.py` | Local-only HPC/workstation heavy browser signoff for real Show2D/Show3D data, including browser FPS, screenshots, paged scrub/hidden-panel checks, Show3D no-blank/panel-independence checks, Show3D FFT idle-cache and return-scrub cache checks, and Show3D FFT metric stats-toggle cache checks. | No, never normal CI. | Heavy real-data datasets, FPS thresholds, page-scrub/cache expectations, FFT overlay/metric performance expectations, or report format change. |
| `scripts/widget_show4dstem_heavy_signoff.py` | Local-only Show4DSTEM heavy browser signoff for real 4D-STEM masters, including NVIDIA/CUDA load timing, append/stack-growth timing, dataset/frame flip FPS, memory pressure, virtual-detector FPS, scan-position FPS, and browser/WebGPU split. | No, never normal CI. | Show4DSTEM CUDA/MPS/cache/chunking behavior, exported HTML performance, or report format changes. |
| `scripts/docs_preview.sh` | Build and serve docs for local visual review. | No. | The docs build command or served path changes. |
| `.github/workflows/widget-ci.yml` | Run local signoff and upload signoff artifacts on PRs and main pushes. | Yes, on matching GitHub events. | Local signoff dependencies, trigger paths, or artifact policy changes. |

Avoid adding new automation files unless one of these scripts cannot reasonably
own the behavior. Prefer adding a focused test before adding another script.
