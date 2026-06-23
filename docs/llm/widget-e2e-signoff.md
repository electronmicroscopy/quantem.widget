# Widget E2E Agent Signoff

Operator goal: let an autonomous agent replace repetitive human visual testing
for `quantem.widget` before release or after widget/rendering changes.

Canonical command:

```bash
cd /Users/macbook/repos/quantem/widget
scripts/widget_visual_signoff.sh
```

Discover options:

```bash
scripts/widget_visual_signoff.sh --help
```

This is not the same as normal CI. CI proves deterministic logic and API
contracts. This signoff proves the widgets still render, respond, and feel
correct in a real browser/Jupyter session.

## Scope

Widget families:

- `Show2D`
- `Show3D`
- `Show3DSlices`
- `Show4DSTEM`
- standalone browser GUI in `web/`

Specialized Show4DSTEM production paths:

- CUDA backend on host
- MPS/Metal backend on Phil/Mac
- live Jupyter WebGPU
- exported/offline HTML WebGPU
- standalone browser folder/5D stack WebGPU

## Layers

### 1. Deterministic Parity Gates

Run these on every release candidate and after math/rendering changes:

```bash
python -m pytest -q \
  tests/test_fft_parity.py \
  tests/test_wgsl_parity.py \
  tests/test_dpc_virtual_parity.py \
  tests/test_bslz4_offline.py \
  tests/test_load_cpu_uint8_clip.py \
  tests/test_state_dict.py
```

Expected signal: all pass. These tests belong in CI because they are bounded,
deterministic, and do not require real data folders.

### 2. Generic Widget Visual Gate

Run this when any widget frontend, state, trait, static asset, or Jupyter
rendering code changes:

```bash
QT_RUN_WIDGET_VISUAL_TESTS=1 python -m pytest -q tests/test_widget_visual_jupyter.py -s
```

Pass signals:

- JupyterLab starts on a temporary port.
- `Show2D`, `Show3D`, `Show3DSlices`, and `Show4DSTEM` outputs render as actual
  widget UI, not object repr text.
- Visible canvases are nonblank.
- Mouse drag changes screenshots.
- FPS is at least `QT_WIDGET_MIN_FPS` (default `30`).
- Temporary Jupyter, browser profile, and kernels are removed.

This is intentionally synthetic and bounded so it can run frequently.

### 3. Show4DSTEM Production Signoff

Use the specialized runbook for full backend/WebGPU migration checks:

```bash
docs/refactor/2026-06-06-show4dstem-agent-signoff-runbook.md
```

Run this when any of the following changes:

- `Show4DSTEM`
- `load(...)`
- CUDA/MPS data loading
- WebGPU compute
- bslz4/offline export
- standalone browser folder loading
- detector binning, dtype, ROI, FFT, DPC, BF/ADF/DF behavior

### 4. Exploratory Agent Sweep

Agents should not only replay fixed scripts. After the scripted gates pass, they
should explore nearby user stories and report surprises:

- Try a different colormap and contrast preset.
- Toggle FFT on/off and verify a new canvas appears/disappears.
- Drag image/scan/detector controls and compare screenshots.
- Move frame/Dataset sliders.
- Try compact/comfy and metadata/file-tree collapse controls.
- Use COPY/export buttons where available.
- Check browser console errors.
- Check GPU memory before/after.
- Clean up temp servers, Jupyter kernels, browser profiles, and exports.

Do not turn every exploratory click into a rigid test. When an exploratory
failure is reproducible and important, promote it into a deterministic opt-in
test or parity test.

## Hardware And Data Policy

- Use Phil/Mac for bounded MPS tests only. Do not run Sample-scale full-data
  browser/MPS stress tests on Phil unless explicitly requested.
- Use host/Linux for full Show4DSTEM CUDA and heavy WebGPU/offline tests.
- Prefer synthetic data for generic widget visual tests.
- Prefer bounded real crops for routine WebGPU live tests.
- Use full no-bin real data only for explicit release signoff.

## Required Final Report

Report these facts every time:

- branch, commit, and dirty status
- command(s) run
- parity test result
- generic widget visual result
- Show4DSTEM CUDA/MPS/WebGPU/offline results, if run
- browser URL/host and whether `navigator.gpu` was true
- FPS numbers and thresholds
- screenshots or artifact paths
- GPU memory before/after for heavy runs
- cleanup status
- caveats and unverified paths

## Promotion Rule

If an agent discovers a new failure mode, do one of these:

- add/update a deterministic parity test when the failure is math/API/state logic
- add/update an opt-in visual test when the failure requires browser/Jupyter UI
- add/update this runbook when the fix is operational knowledge

Future agents should not need to rediscover the same workflow by trial and error.
