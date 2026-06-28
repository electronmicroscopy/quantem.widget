# Release

Use this runbook for every `quantem.widget` release candidate or TestPyPI
release. Do not push a `widget-v*` tag until the local gates below pass.

## Release target

`quantem.widget` publishes release candidates to **TestPyPI** through the
`Widget release` GitHub Actions workflow. TestPyPI is the staging Python package
index. It is not PyPy, the Python runtime.

The release workflow is triggered by tags named:

```bash
widget-vX.Y.Z
widget-vX.Y.ZrcN
widget-vX.Y.Z.postN
```

For testing, prefer release-candidate tags such as:

```bash
widget-v0.0.8rc1
```

## Before tagging

Start from a clean, updated `main`:

```bash
git switch main
git pull origin main
git status --short
```

If the worktree contains scratch files, generated `._*` metadata, notebooks with
large outputs, or unrelated experiments, stop and clean or move them before
continuing.

## Local gates

Run the full Python and frontend checks:

```bash
PYTHONPATH=src pytest -q
npm run build
```

For widget/export changes, also run:

```bash
PYTHONPATH=src pytest -q tests/test_html_export_protocol.py tests/test_showeds.py
```

For release packaging, run:

```bash
scripts/widget_release_check.sh
```

This script runs the frontend typecheck/tests/build, standalone browser build,
offline browser build, Python compile smoke, local wheel build, and wheel-content
checks.

## Visual signoff

Widgets can import successfully while still rendering blank canvases. Before a
release candidate, run at least the quick visual smoke:

```bash
scripts/widget_visual_signoff.sh --quick
```

For Show4DSTEM/WebGPU-sensitive changes, run:

```bash
scripts/widget_visual_signoff.sh --show4dstem
```

For a release-candidate tag, prefer the full signoff:

```bash
scripts/widget_visual_signoff.sh --full
```

The full signoff includes generic widget visual tests, Show4DSTEM WebGPU browser
and Jupyter smokes, plus the local release build/check gate. Hardware-dependent
real-data CUDA/MPS checks may still need a separate machine-specific runbook.

## Tag and publish to TestPyPI

After the local gates pass, create and push the release-candidate tag:

```bash
git tag widget-v0.0.8rc1
git push origin widget-v0.0.8rc1
```

GitHub Actions will:

1. build frontend widget assets,
2. build the standalone browser GUI,
3. stamp the package version from the tag,
4. build the wheel,
5. run `twine check`,
6. verify required wheel contents,
7. publish to TestPyPI.

Watch the `Widget release` workflow until it completes.

## TestPyPI install check

After the workflow publishes, install the candidate in a fresh environment:

```bash
python -m venv /tmp/quantem-widget-rc
source /tmp/quantem-widget-rc/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  quantem-widget==0.0.8rc1
python - <<'PY'
import quantem.widget as qw
print(qw.__version__)
from quantem.widget import Show2D, Show3D, Show3DSlices, Show4DSTEM, ShowEDS
print("widgets import ok")
PY
```

If possible, also open a small notebook or exported HTML with each widget before
promoting the release.

## If something fails

Do not overwrite a published tag. Create a new release-candidate tag instead:

```bash
widget-v0.0.8rc2
```

Fix the issue on a normal branch, merge through a PR, pull updated `main`, rerun
the gates, and then push the next release-candidate tag.
