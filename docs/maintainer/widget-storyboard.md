# Widget Storyboard

This is the maintainer index for recurring AI/browser drive plans. Storyboards
are split by widget so agents can load only the relevant file and avoid editing
unrelated widget guidance.

Storyboards are written as scientific user stories, not fixed button scripts.
The order of stories can change for a release, bug, dataset, or widget, but the
agent report must say which story IDs were driven, which were skipped, and why.

Use these files for UI behavior, real-data workflows, browser testing, and
release signoff. Keep performance lessons, timing observations, and
implementation policy in [Performance](widget-performance).

## Widget Files

- [Show2D storyboard](widget-storyboard-show2d)
- [Show3D storyboard](widget-storyboard-show3d)
- [Show3DSlices storyboard](widget-storyboard-show3dslices)
- [Show4DSTEM storyboard](widget-storyboard-show4dstem)
- [ShowEDS storyboard](widget-storyboard-showeds)

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
- Use an MJ-goat or buffle Jupyter backend when testing real data, large arrays,
  save/reopen, or backend streaming.
- Use real or real-derived microscopy data first. Synthetic data is a secondary
  control only.
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
