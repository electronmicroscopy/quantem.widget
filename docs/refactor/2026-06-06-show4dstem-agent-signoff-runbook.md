# Show4DSTEM agent signoff runbook

Use this when an autonomous agent needs to revalidate the Show4DSTEM migration
across the four production paths:

1. CUDA Python backend
2. Phil MPS/Metal backend
3. WebGPU live/browser compute
4. WebGPU exported HTML with bslz4 companion chunks

The goal is to prove real browser interaction, not just imports. The pass
criteria are: widget renders nonblank, ROI/scan/Dataset controls change the
canvases, WebGPU tests report `navigator.gpu == true`, rAF is at least 30 FPS
for browser paths, and every temporary server/browser/export is cleaned up.

## Preflight

Run from the repo root on the target machine.

```bash
git branch --show-current
git rev-parse --short HEAD
git status --short
```

Expected signoff branch/tag after the migration:

```bash
git tag --points-at dab802c441db87437eaeb3c71086ef61c090632e
# show4dstem-migration-signoff-2026-06-05
```

Check for stale local test processes before starting:

```bash
pgrep -af 'jupyter-lab|ipykernel|http.server|show4dstem-|playwright|cdp-show4dstem|Chrome.*enable-unsafe-webgpu|chrome.*enable-unsafe-webgpu' || true
```

Only kill processes you started or processes using the explicit temp names/ports
in this runbook. Do not kill the user's normal browser, Codex extension host, or
Claude native host.

## Case 1 — CUDA backend on host

Purpose: prove Python `TorchBackend` load/compute/widget construction on a real
full no-bin CUDA stack.

Use a small foreground script, not an executed notebook artifact:

```bash
ssh host 'cd /home/user/repos/quantem/widget && /home/user/miniforge3/bin/mamba run -n cuda-env python - <<'"'"'PY'"'"'
from quantem.widget import load, Show4DSTEM
master = "/home/user/ssd/data/sample/series/Sample_3.6Mx_21.4mrad_185mmcl_20pA_29_master.h5"
res = load(master, det_bin=1, verbose=True)
w = Show4DSTEM(res, title="CUDA full no-bin signoff", precompute_virtual_images=False, verbose=True)
print("shape", w.shape_rows, w.shape_cols, w.det_rows, w.det_cols)
print("backend", type(w._compute).__name__, w._compute.device)
print("bytes", len(w.virtual_image_bytes), len(w.frame_bytes))
PY'
```

Pass signal: shape `512 512 192 192`, backend `TorchBackend` on CUDA, and both
virtual image/frame bytes populated.

## Case 2 — MPS/Metal backend on Phil

Purpose: prove Phil can open no-bin data in the raw-Metal path without freezing.
Keep this on gold or another bounded Phil-local dataset unless explicitly asked
to stress Sample-scale data.

```bash
cd /Users/macbook/repos/quantem/widget
/Users/macbook/miniforge3/bin/python - <<'PY'
import gc, time
from quantem.widget import load, Show4DSTEM
master = "/Users/macbook/data/george/gold-10/gold_10_master.h5"
res = load(master, backend="mps", det_bin=1, verbose=True)
w = Show4DSTEM(
    res,
    title="Phil MPS full no-bin signoff",
    fast_interaction=True,
    fast_interaction_async=True,
    precompute_virtual_images=False,
    verbose=True,
)
ok = w.wait_for_fast_interaction(timeout=60)
print("shape", w.shape_rows, w.shape_cols, w.det_rows, w.det_cols)
print("fast_sidecar_ready", ok, w.fast_interaction_ready, w.fast_interaction_building)
print("bytes", len(w.virtual_image_bytes), len(w.frame_bytes))
del w, res
gc.collect()
PY
```

Pass signal: raw-Metal viewer constructs, no-bin detector shape is preserved, the
fast sidecar becomes ready, and Phil remains responsive.

For real browser interaction, launch a temporary JupyterLab on a unique port,
open a temp notebook, run the same code with `display(w)`, drag a scan canvas and
detector ROI in headed Chrome, then stop the server. Do not leave Phil notebooks
or kernels running.

## Case 3 — WebGPU live/browser compute

Purpose: prove `backend="web"` routes reductions through browser WebGPU while a
live kernel exists.

Run the opt-in JupyterLab browser smoke first. By default it uses a small
synthetic two-frame stack so it exercises the live widget, Dataset/frame slider,
FFT toggle, ROI/scan drag, `navigator.gpu`, and rAF FPS without stressing Phil
or host.

```bash
QT_RUN_JUPYTER_WEBGPU_TESTS=1 python -m pytest -q tests/test_show4dstem_webgpu_live_jupyter.py
```

For a bounded real-data host run, point the test at Sample data and keep the
crop enabled:

```bash
QT_RUN_JUPYTER_WEBGPU_TESTS=1 \
QT_WEBGPU_LIVE_MASTER=/home/user/ssd/data/sample/series/Sample_3.6Mx_21.4mrad_185mmcl_20pA_29_master.h5 \
QT_WEBGPU_LIVE_DET_BIN=4 \
QT_WEBGPU_LIVE_DTYPE=u8 \
QT_WEBGPU_LIVE_CROP=96:160,96:160 \
python -m pytest -q tests/test_show4dstem_webgpu_live_jupyter.py
```

Only use full live data when explicitly requested and when GPU/browser memory is
available:

```bash
QT_RUN_JUPYTER_WEBGPU_TESTS=1 \
QT_WEBGPU_LIVE_MASTER=/home/user/ssd/data/sample/series/Sample_3.6Mx_21.4mrad_185mmcl_20pA_29_master.h5 \
QT_WEBGPU_LIVE_DET_BIN=1 \
QT_WEBGPU_LIVE_DTYPE=u8 \
QT_WEBGPU_LIVE_FULL=1 \
QT_WEBGPU_REQUIRE_FRAME_SLIDER=0 \
python -m pytest -q tests/test_show4dstem_webgpu_live_jupyter.py
```

Pass signals: `navigator.gpu == true`, at least four canvases, Dataset/frame
slider found and moved for bounded/default 5D cases, FFT toggle clicked, both
DP and virtual-image `COPY` buttons write `image/png` to the browser clipboard,
screenshot changes after drag, and rAF FPS is at least 30. The test prints a
JSON summary with canvas count, frame-slider status, copy-button count,
screenshot-change status, and measured FPS.

Manual bounded crop equivalent, if the automated test needs debugging:

```python
from quantem.widget import load, Show4DSTEM
master = "/home/user/ssd/data/sample/series/Sample_3.6Mx_21.4mrad_185mmcl_20pA_29_master.h5"
res = load(master, det_bin=4, dtype="u8", verbose=True)
crop = res.data[96:160, 96:160]
w = Show4DSTEM(crop, backend="web", title="WebGPU live crop", precompute_virtual_images=False)
display(w)
```

In headed Chrome/Playwright:

- Assert `await page.evaluate("!!navigator.gpu")` is `true`.
- Wait for at least four canvases.
- Drag the scan canvas and detector ROI.
- Compare before/after screenshots; they must differ.
- Measure rAF for 3 seconds; pass if FPS is at least 30.

## Case 4 — WebGPU exported HTML + bslz4 companion chunks

Purpose: prove large no-bin exported HTML works with companion chunks and lazy
multi-volume Dataset switching.

Small opt-in browser regression:

```bash
QT_RUN_BROWSER_TESTS=1 /Users/macbook/miniforge3/bin/python -m pytest -q tests/test_show4dstem_webgpu_browser.py
```

Full no-bin Linux smoke:

```python
from quantem.widget import load, Show4DSTEM
masters = [
    "/home/user/ssd/data/sample/series/Sample_3.6Mx_21.4mrad_185mmcl_20pA_29_master.h5",
    "/home/user/ssd/data/sample/series/Sample_3.6Mx_21.4mrad_185mmcl_20pA_30_master.h5",
]
res = load(masters, det_bin=1, dtype="u8", verbose=True)
w = Show4DSTEM(
    res,
    backend="web",
    offline_codec="bslz4",
    data_url="/home/user/ssd/tmp/show4dstem-agent-full-webgpu",
    title="Show4DSTEM full no-bin WebGPU export",
    precompute_virtual_images=False,
    verbose=True,
)
w.export_html("/home/user/ssd/tmp/show4dstem-agent-full-webgpu/index.html")
```

Serve and drive from headed Chrome on host:

```bash
cd /home/user/ssd/tmp/show4dstem-agent-full-webgpu
python3 -m http.server 8897 --bind 127.0.0.1
```

Pass signals:

- `navigator.gpu == true`
- HTML renders `512x512 | 192x192`
- Initial volume fetches `vol0/` chunk/meta files
- Real Dataset slider drag fetches `vol1/` chunk/meta files
- ROI/scan drag changes screenshots
- DP and virtual-image `COPY` buttons write `image/png` to the browser clipboard
- rAF is at least 30 FPS

## Cleanup

Always clean the exact temp resources created by the run. Prefer unique names
with `show4dstem-agent-` or `cdp-show4dstem-`.

```bash
# host
ssh host 'pkill -f "http.server 8897" || true; pkill -f "cdp-show4dstem" || true; rm -rf /home/user/ssd/tmp/show4dstem-agent-full-webgpu; pgrep -af "show4dstem-agent|cdp-show4dstem|http.server 8897|playwright" || true'

# Phil/local
pkill -f 'jupyter-lab.*show4dstem-agent' || true
pkill -f 'http.server 8898' || true
rm -rf /tmp/show4dstem-agent-* /tmp/cdp-show4dstem-*
pgrep -af 'show4dstem-agent|cdp-show4dstem|http.server 8898|ipykernel|jupyter-lab' || true
```

Final report checklist:

- branch, commit, and tag
- CUDA shape/backend result
- Phil MPS shape/sidecar/browser result
- WebGPU live `navigator.gpu`, canvas count, screenshot diff, FPS
- WebGPU exported/multi-volume `navigator.gpu`, `vol1/` fetch, screenshot diff,
  FPS
- cleanup confirmation: no temp exports, temp servers, temp Chrome/Playwright, or
  temp kernels left running
