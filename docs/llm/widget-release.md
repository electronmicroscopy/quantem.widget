# quantem.widget release runbook

Operator goal: release a `quantem.widget` RC or final package only after source,
wheel, standalone browser GUI, notebook, and real-browser user paths have been
checked.

Canonical local preflight:

```bash
cd /Users/macbook/repos/quantem/widget
scripts/widget_release_check.sh
```

Discover script options:

```bash
scripts/widget_release_check.sh --help
```

## Branch and Version

- Work from the release branch, currently `widget-show3d-show4dstem-kernels`.
- Release tags use `widget-vX.Y.ZrcN`, for example `widget-v0.0.1rc20`.
- The GitHub workflow derives the Python package version from the tag and
  publishes `quantem.widget` to TestPyPI.
- Do not tag if the worktree is dirty with uncommitted release changes.

## Required Gates Before Tagging

Run:

```bash
scripts/widget_release_check.sh
```

Expected success signals:

- TypeScript passes: `npm run typecheck`.
- Vitest passes: `npm test`.
- Widget static assets build: `npm run build`.
- Standalone browser GUI builds: `cd web && npm run build`.
- Offline/single-file browser GUI builds: `cd web && npm run build:offline`.
- Touched Python files compile.
- A local wheel is built and contains:
  - `quantem/widget/static/show4dstem.js`
  - `quantem/widget/static/browser/index.html`

## Real Browser Gate on host

Use this when Show4DSTEM browser, WebGPU, H5 folder loading, picker/watch, or
standalone HTML changed. Keep the test bounded and clean up after it.

Prerequisites:

- host has Chrome on `DISPLAY=:0`.
- Real Sample data is available, currently
  `/home/user/ssd/data/sample/series`.
- Use copied or same-filesystem hardlinked test folders under `/home/user`.

Required user-path checks:

- Launch the rebuilt standalone browser GUI from `web/dist`.
- Drive visible Chrome, not just import/load code.
- Click `Choose folder`.
- Select a real folder with the OS picker.
- Accept Chrome's folder permission prompt.
- Confirm `navigator.gpu === true`.
- Confirm the UI shows `watching`.
- Confirm the file tree lists full `512x512x192x192` masters.
- Confirm BF and CBED canvases are nonblank.
- Confirm single-master WebGPU auto-bin chooses a safe detector bin instead of
  leaving black canvases when no-bin exceeds browser VRAM.
- Switch BF, ADF, DF and confirm screenshots visibly differ.
- Toggle FFT and confirm the third panel appears.
- Measure `requestAnimationFrame` FPS; target is at least 30 fps, expect about
  60 fps on host for the bounded browser path.
- Drag the detector/aperture or scan crosshair with real mouse events and
  confirm the screenshot changes.
- Add another full master + sidecars to the watched folder while Chrome remains
  open and confirm the file tree/Stack viewer updates.

Report back with:

- Browser URL and host.
- Folder selected.
- Number of masters and shapes shown.
- Chosen `det_bin` and dtype.
- Decode/reduce timings from `window.__perf`.
- FPS result.
- Controls exercised.
- Screenshot paths if captured.
- Cleanup status and final `nvidia-smi` memory.

## Clean TestPyPI Install Gate

After pushing a `widget-v*` tag and the `Widget release` workflow succeeds,
install from TestPyPI in a clean environment. TestPyPI can take a minute to
propagate; retry once before treating "version not found" as failure.

```bash
python -m venv /tmp/qwidget-release-venv
/tmp/qwidget-release-venv/bin/python -m pip install --upgrade pip
/tmp/qwidget-release-venv/bin/python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
  'quantem.widget==X.Y.ZrcN'
/tmp/qwidget-release-venv/bin/python - <<'PY'
import quantem.widget as qw
print("version", qw.__version__)
print("has_public_api", all(hasattr(qw, name) for name in [
    "Show4DSTEM", "load", "Show2D", "Show3D",
]))
print("show4dstem_callable", callable(qw.Show4DSTEM))
print("all", qw.__all__)
PY
```

Expected success signal:

- Installed version matches the tag.
- Public API imports and `Show4DSTEM` is callable.

## Clean Jupyter User-Path Gate

Use this before promoting an RC to final PyPI or when notebook/widget code
changed.

Goal: mimic a new notebook user using the installed wheel, not the source tree.

Run from a clean env where `quantem.widget==X.Y.ZrcN` came from TestPyPI:

```python
from quantem.widget import load, Show4DSTEM

w = Show4DSTEM(load(master, det_bin=4, verbose=False), verbose=False)
w_web = Show4DSTEM(load(master, det_bin=4, verbose=False), backend="web")
w_web.export_html("/tmp/show4dstem.html")
```

Required visual checks in JupyterLab:

- Widget output renders, not only a Python object repr.
- BF/ADF/DF controls update the visible image.
- Frame/scan controls update the CBED.
- FFT toggle renders the FFT panel.
- `backend="web"` runs browser WebGPU when `navigator.gpu` is available.
- Exported HTML opens in a browser and renders nonblank.

Report back with:

- Installed package version and import path.
- Jupyter host/browser used.
- Data path and `det_bin`/dtype.
- Which controls were exercised.
- Whether exported HTML rendered.
- Any errors or warnings.

## Tag and Publish

Only after required gates pass:

```bash
git tag -a widget-vX.Y.ZrcN -m "quantem.widget X.Y.ZrcN"
git push origin widget-show3d-show4dstem-kernels
git push origin widget-vX.Y.ZrcN
gh run watch --repo bobleesj/quantem <Widget release run id> --exit-status
```

The release workflow publishes to TestPyPI. Do not claim the release is usable
until the workflow succeeds and the clean TestPyPI install smoke passes.

## Docs Workflow

The `widget docs` workflow is a separate public-docs path. Treat it as:

- Blocker for docs site updates.
- Not a package release blocker if the docs build job passes but GitHub Pages is
  skipped or rejected due to Pages/environment permissions.

Current repository rule: the `github-pages` environment rejects deployments from
`widget-show3d-show4dstem-kernels`. The workflow therefore builds docs on the
release branch, but only deploys Pages from `main`. To publish docs directly
from the release branch, first change the GitHub environment protection rule.

If it fails:

```bash
gh run view <run-id> --repo bobleesj/quantem --json jobs,conclusion,url
gh run view <run-id> --repo bobleesj/quantem --log-failed
```

Report whether the Jupyter Book build failed or only the Pages deploy failed.

## Final Report Format

Report:

- Commit and tag.
- Whether branch/tag were pushed.
- Local gates.
- Real browser gate.
- Clean TestPyPI install gate.
- Clean Jupyter gate, if run.
- Workflow status.
- Any caveats.
