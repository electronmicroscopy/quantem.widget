# Widget Review Matrix

Use this page before reviewing, merging, or assigning agents to a
`quantem.widget` change. The goal is to turn "please review this widget" into a
concrete set of code, browser, performance, packaging, and science checks.

This matrix complements [Agent signoff](widget-agent-signoff),
[UI protocol](widget-ui-protocol), [Performance UI testing](performance-ui-testing),
and [Release](widget-release). Use those runbooks for the detailed browser and
release procedures.

## First Pass

Start every review with these questions:

1. Which widget or shared layer changed?
2. Is the change Python-only, frontend-only, export/state, browser/WebGPU,
   packaging, docs/tutorial, or scientific-data related?
3. Does a user feel the change with the mouse, keyboard, touch, playback, or
   saved/exported HTML?
4. Does the change add or alter hard-coded scientific data, bundled real data,
   file formats, dependencies, or public defaults?
5. Which tests, browser actions, and performance checks prove the affected
   surface still works?

If the answer to 3 is yes, source inspection and unit tests are not enough. Run
or request browser signoff.

## Review Surfaces

| Surface | Typical files | Review focus | Minimum checks |
|---|---|---|---|
| Python widget API | `src/quantem/widget/show*.py`, `state.py`, `export.py` | Shapes, dtypes, units, coordinate convention, trait defaults, state roundtrip, helpful errors | Focused `PYTHONPATH=src:. pytest -q tests/test_<widget>.py` plus changed shared tests |
| Frontend widget UI | `js/show*/index.tsx`, `js/figure.ts`, `js/stats.ts`, `js/theme.ts` | Layout, labels, theme, pointer events, canvas drawing, sliders, menus, accessibility names | `npm run build`, `npm run typecheck`, `npm test`, browser smoke |
| Browser WebGPU | `js/webgpu-volume.ts`, generated `js/.generated/engine/{device,io,detector,dpc,ssb}`, `web/src/utils/*`, Show4DSTEM/ShowEDS frontend paths | Adapter detection, explicit GPU-unavailable errors, WGSL equivalence, memory ownership, browser/export split | WebGPU-focused pytest/Vitest plus browser signoff on supported hardware |
| Backend GPU | `src/quantem/widget/gpu.py`, `show4dstem_mps.py`, and `quantem.gpu` domain APIs | CUDA/MPS/WebGPU routing, deterministic failure when GPU compute is unavailable, memory lifetime, dtype preservation | Focused backend tests and a real-data run on the intended machine |
| Export and save state | `export.py`, widget `export_html`, static fallback, `tests/test_save_state.py`, `tests/test_html_export_protocol.py` | Standalone HTML works without Python, state stays compact, exact/folder/downsample labels are honest | HTML protocol tests, export/reopen browser smoke, notebook save/reopen |
| Performance | Pointer UI, profile drawing, playback, histograms, data loaders | First paint, drag/zoom FPS, slider/playback sync, kernel busy state, export size/time | `scripts/widget_local_signoff.sh --quick --browser`, performance gate when large-data paths changed |
| Scientific logic | diffraction, DPC, FFT, virtual detector, EDS, calibration, units | Real workflow behavior, numerical parity, units, coordinate convention, tolerance defaults | Focused scientific tests using realistic arrays plus docs examples |
| Scientific data and constants | hard-coded tables, bundled `.npy/.npz/.h5`, tutorial data, reference values | DOI/source/database, license/redistribution, assumptions such as voltage or temperature | Provenance note in docs/code plus tests that do not depend on unlicensed data |
| Packaging | `pyproject.toml`, static assets, data files, wheel/sdist contents | Required vs optional dependencies, wheel contents, stale artifacts, fresh install | `scripts/widget_release_check.sh` or wheel-content check for release paths |
| Docs and notebooks | `docs/api`, `docs/tutorials`, `docs/maintainer` | Interactive docs remain interactive, outputs intentional, no giant notebook state, exactly one output per widget cell (no static-fallback duplicate under the live widget) | notebook size guard, docs build for broad docs changes, zero `img.quantem-static-fallback` on built pages |
| Repo hygiene | `._*`, `__pycache__`, generated screenshots, scratch files | No AppleDouble metadata, generated caches, huge artifacts, or unrelated experiments | `git status --short`, `scripts/check_large_files.py`, `scripts/check_notebook_sizes.py` |

## Widget-Specific Checklist

| Widget | Must-drive interactions |
|---|---|
| Show2D | histogram min/max and center drag, pan, wheel zoom, FFT toggle/pan/zoom, panel layout, scale bar, export/reopen, save/reopen |
| Show3D | frame scrub, play/pause, FPS control, loop/bounce, histogram drag, pan/zoom, export GIF/MP4/HTML where touched, save/reopen |
| Show3DSlices | slice sliders, crosshair, 3D view controls, panel size, histogram, FFT/log/smooth/export |
| Show4DSTEM | diffraction pan/zoom, scan-position update, BF/ABF/ADF detectors, detector drag, virtual image update, WebGPU path, export/reopen |
| ShowEDS | band center/edge drag, map/spectrum sync, ROI drag/resize, periodic table, element lines, WebGPU/sparse/folder backends, export/reopen |
| ShowFolder | real folder selection, thumbnails/cache, file/folder navigation, multiselect, selection save/load, compact state |
| ShowDiffraction | center pick/refine, spot/ring add/move/remove, d-spacing calibration, masks, profile/azimuthal panels, phase tools, export/reopen |

## Agent Assignments

Use multiple agents only when their scopes are separate. Give each agent the
same branch/commit and ask for evidence, not opinions.

| Agent | Owns | Output |
|---|---|---|
| Codebase mapper | Changed files, affected widgets, shared modules, likely blast radius | One-page surface map and suggested focused tests |
| Python/science reviewer | Algorithms, units, shape/dtype handling, scientific defaults, provenance | Findings with file/line refs and missing tests |
| Frontend reviewer | React/traitlets sync, canvas rendering, layout, labels, pointer events | Browser-observed issues and source refs |
| Performance reviewer | First paint, pointer latency, playback, save/export size, large-data behavior | Timing/FPS table and PASS/PASS WITH LIMITATION/BLOCKED |
| GPU/WebGPU reviewer | Browser WebGPU, backend CUDA/MPS/CPU, parity/fallback behavior | Adapter/backend report and parity checks |
| Packaging/docs reviewer | Wheel/sdist, optional deps, static assets, tutorials, notebook size | Fresh-install or wheel-content notes, docs risks |
| Final reviewer | Deduplicates findings, separates blockers from cleanup | Final review body ready for GitHub |

## Command Sets

Use the smallest set that covers the changed surface while iterating. Do not
claim broader coverage than you ran.

### Python-only widget change

```bash
PYTHONPATH=src:. pytest -q tests/test_widget.py tests/test_state_dict.py
PYTHONPATH=src:. pytest -q tests/test_<affected_widget>.py
```

### Frontend widget change

```bash
npm run build
npm run typecheck
npm test
PYTHONPATH=src:. pytest -q tests/test_html_export_protocol.py
```

Then drive the affected widget in JupyterLab or exported HTML.

### Export, save, or standalone HTML change

```bash
PYTHONPATH=src:. pytest -q tests/test_html_export_protocol.py tests/test_save_state.py
PYTHONPATH=src:. python scripts/widget_html_smoke.py --artifact-dir /tmp/quantem-widget-html-smoke
```

For browser behavior:

```bash
scripts/widget_local_signoff.sh --quick --browser
```

### WebGPU or GPU-sensitive change

```bash
PYTHONPATH=src:. pytest -q tests/test_show4dstem_webgpu_browser.py tests/test_wgsl_parity.py
PYTHONPATH=src:. pytest -q tests/test_quantem_gpu_ownership.py tests/test_core_dataset4dstem.py
```

Also record the browser WebGPU adapter and backend host. Any scientific CPU
fallback is a failure.

### Broad UI or release-candidate change

```bash
scripts/widget_local_signoff.sh --full --browser --performance
scripts/widget_visual_signoff.sh --full
scripts/widget_release_check.sh
```

## Blocking Rules

Block merge or release when any of these are true:

- a touched widget renders blank, throws browser console errors, or cannot
  complete its primary user story;
- a pointer interaction visibly lags or detaches from the cursor and no
  limitation is documented;
- notebook save/reopen embeds heavy buffers or loses visible output;
- exported HTML does not reopen without Python for a documented standalone mode;
- WebGPU/backend GPU behavior silently falls back without a user-visible reason;
- a required dependency is added without explaining why it cannot be optional;
- hard-coded scientific data, reference constants, or bundled real data lack
  source/DOI/database and redistribution/license clarity;
- large notebooks, rendered HTML, data files, AppleDouble `._*` files,
  `__pycache__`, or unrelated scratch artifacts are present in the intended
  diff.

## Review Comment Pattern

Good review comments should be short and line-level:

```text
Can we document the DOI/source and redistribution license for this hard-coded
reference data? If these values are approximate defaults rather than
authoritative reference data, please say that explicitly.
```

```text
This updates React state on every pointer move. Can we route the visual preview
through requestAnimationFrame and commit stable widget state only on release?
```

```text
Does this need to be a required dependency? If it only supports an optional
workflow, can we keep the base install lean and soft-import it with a helpful
install message?
```

## Final Review Template

Use this structure in PR summaries or agent handoffs:

```text
Review coverage
- Code surfaces:
- Tests run:
- Browser actions driven:
- Performance data:
- GPU/WebGPU data:
- Packaging/docs checks:
- Skipped checks and why:

Decision
- PASS / PASS WITH LIMITATION / BLOCKED
- Blockers:
- Follow-up cleanup:
```
