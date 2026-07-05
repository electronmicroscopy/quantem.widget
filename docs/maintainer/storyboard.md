# Storyboard

This is the maintainer index for recurring AI/browser drive plans. Storyboards
are split by viewer so agents can load only the relevant file and avoid editing
unrelated guidance.

Storyboards are written as scientific user stories, not fixed button scripts.
The order of stories can change for a release, bug, dataset, or widget, but the
agent report must say which story IDs were driven, which were skipped, and why.

Use these files for UI behavior, real-data workflows, browser testing, and
release signoff. Keep performance lessons, timing observations, and
implementation policy in [Performance](widget-performance). Use
[Performance UI Testing](performance-ui-testing) when the claim involves load
speed, FPS, heavy real data, HPC/backend topology, save/reopen timing, or
export size.

Every storyboard should cover four scientific workflow classes:

- **Loading and storage**: opening files quickly, avoiding huge saved-notebook
  state, and making export payload choices explicit.
- **Interactive inspection**: zoom, pan, histogram, FFT, profile, ROI, playback,
  and slider work at the target frame rate after the first view is visible.
- **High-throughput review**: many files or panels, often 30, 45, 85, or more
  4k images from denoising, drift, ptychography, or screening workflows.
- **Stress signoff**: intentionally heavy real or real-derived datasets that
  record first paint, steady interaction FPS, memory/payload size, and whether
  native pixels stream correctly after a fast preview.

## Operating Model

The primary production workflow is remote compute with local interaction:
scientists run Python, file I/O, CUDA/MPS/CPU backend work, and large-data
preparation on an HPC/workstation backend, then view and drive the widget from
their local laptop browser. This is the highest-priority path because large
electron microscopy datasets usually live next to the GPU, storage, and Jupyter
server, while the user wants a responsive viewer on the computer in front of
them.

The secondary workflow is local laptop use: a user installs ``quantem.widget``
and opens smaller files, tutorial data, exported HTML, or saved notebooks on
the same machine. This must remain supported, but it should not force the
remote-backend workflow to copy huge arrays, materialize sparse zeros, or wait
for Python round trips during pointer interaction.

For signoff, always state which mode was tested:

- **Remote backend / local frontend**: Jupyter kernel and data live on an
  HPC/workstation backend; Codex or the user drives the browser from a local
  laptop.
- **Local laptop**: Python kernel, files, and browser all live on the same
  laptop.
- **Standalone HTML**: no Python kernel; all required interactive data is in
  the exported page or a documented companion payload.
- **Saved notebook reopen**: no cell rerun; JupyterLab restores the saved
  compact widget view from notebook state.

## Why These Widgets Matter

Vendor viewers are useful for acquisition, but they are usually closed,
session-oriented, and difficult to connect to reproducible Python analysis.
``quantem.widget`` should be better for research workflows because it keeps the
viewer next to the code, data provenance, calibrated axes, saved notebook
state, and shareable HTML artifacts.

The expected advantage is not a prettier screenshot. The expected advantage is
that a microscopist can open real data, tune analysis parameters, inspect
pixels or spectra, save the exact view, reopen it later, export it for a
collaborator, and have an AI or human reviewer reproduce the same interaction
path. Storyboards should therefore test the full scientific loop, not just
whether a canvas appears.

## Storyboard Files

- [Show2D](storyboard-show2d)
- [Show3D](storyboard-show3d)
- [Show3DSlices](storyboard-show3dslices)
- [Show4DSTEM](storyboard-show4dstem)
- [ShowEDS](storyboard-showeds)
- [ShowFolder](storyboard-showfolder)

## Story Format

Each story has four parts:

- **User story**: the scientific workflow and reason it matters.
- **Primary widgets**: widgets that must satisfy the story.
- **Data to use**: real or real-derived data preferred for signoff.
- **Acceptance checks**: concrete browser actions and expected outcomes.

Acceptance checks are executable, but they are subordinate to the story. Agents
should adapt the order and exact dataset to the change under test instead of
blindly clicking through a list.

## Agent Rules

- Drive the actual widget in the Codex in-app browser or Chrome; Python tests
  alone do not verify a story.
- Use an HPC/workstation Jupyter backend when testing real data, large arrays,
  save/reopen, backend streaming, or any workflow intended to represent the
  normal lab deployment.
- Use real or real-derived microscopy data first. Synthetic data is a secondary
  control only.
- Include at least one high-throughput story for Show2D and Show3D when the
  change affects loading, layout, image drawing, FFT, playback, export, or
  saved notebook state. Prefer 4k x 4k real files and 30/45/85-file batches
  when those data are available on the backend.
- Test desktop and mobile-sized viewports. A narrow browser viewport is a
  pre-check; physical iPhone Safari is required for iPhone-specific claims.
- Record backend host, frontend browser, URL/notebook, widget source path,
  data path, shape, dtype, native bytes, panel count, frame count, display bin,
  first-paint time, and interaction FPS method.
- Start from a fresh render after code/build changes: rebuild, reload, rerun
  the notebook cell, or reopen exported HTML.
- Mark each story ``Pass``, ``Fail``, or ``Not verified``. Do not report
  "all good" from screenshots, DOM inspection, or unit tests alone.

## Release Report Template

Use this template in agent signoff reports:

```text
Verified:
- Stories driven:
- URL/notebook:
- Backend host/source path:
- Frontend browser:
- Data source and shape:
- First-paint time:
- Interaction FPS method/result:
- Save/reopen result:
- Exports opened:
- Tests run:

Not verified:
- Story IDs:
- Reason:

Remaining risk:
- Hardware/browser/data sizes not covered:
```

## Release-Gating Rule

- If any P0 story fails, do not tag an RC: first paint over roughly 10 s, blank
  saved output, heavy-buffer save leak, broken export menu, playback/slider
  desync, FFT correctness failure, or interaction far below the target FPS.
- If a P1 story is not verified, the RC report must say exactly why and who will
  verify it next. P1 examples: physical iPhone checks, maximum-size datasets,
  or hardware-specific WebGPU adapter coverage.
- The storyboard report must be linked from the release candidate signoff.
- Any Show2D or Show3D performance claim must also cite the measured gate from
  [Performance UI Testing](performance-ui-testing), including real dataset,
  backend host, browser surface, first paint, and FPS/latency observations.
