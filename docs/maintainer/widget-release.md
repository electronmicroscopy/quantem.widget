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
widget-v0.0.1rc1
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
PYTHONPATH=src:. pytest -q
npm run build
```

For widget/export changes, also run:

```bash
PYTHONPATH=src:. pytest -q tests/test_html_export_protocol.py tests/test_showeds.py
```

For release packaging, run:

```bash
scripts/widget_release_check.sh
```

This script runs the frontend typecheck/tests/build, standalone browser build,
offline browser build, Python compile smoke, local wheel build, and wheel-content
checks.

Before uploading to TestPyPI or PyPI, build fresh artifacts and inspect the
actual upload payload. Do not upload stale files from an old local `dist/`
directory.

```bash
rm -rf /tmp/quantem-widget-release-audit
mkdir -p /tmp/quantem-widget-release-audit
python -m build --outdir /tmp/quantem-widget-release-audit
tar -tzf /tmp/quantem-widget-release-audit/*.tar.gz | less
unzip -l /tmp/quantem-widget-release-audit/*.whl | less
```

Confirm the wheel and sdist contain no private microscope data, generated
reports, screenshots, built docs, caches, or large local artifacts. In
particular, no `.h5`, `.hdf5`, `.emd`, `.npy`, `.npz`, `.zarr`, `.tif`, `.dm3`,
`.dm4`, `.raw`, generated tutorial HTML, or private JSON/CSV report should be
present. The expected wheel contents are Python source, package metadata,
licenses, and built widget JavaScript/static assets.

## Visual signoff

Widgets can import successfully while still rendering blank canvases. Before a
release candidate, create an agent signoff packet and drive the affected widgets
in the browser:

```bash
scripts/widget_agent_signoff.sh --quick
```

The agent signoff is a fix-and-redrive loop: choose the relevant scientific
stories from [Storyboard](storyboard), open the docs page,
notebook, or exported HTML; click, drag, zoom, scrub, resize, and export; patch
anything that feels wrong; rebuild; refresh; and redrive the same story. Save
final screenshots or short videos in the packet before tagging. See
[Agent signoff](widget-agent-signoff).

Also run at least the quick visual smoke:

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

For broad UI releases, pair it with:

```bash
scripts/widget_agent_signoff.sh --full
```

The full signoff includes generic widget visual tests, Show4DSTEM WebGPU browser
and Jupyter smokes, plus the local release build/check gate. The human-readable
report should cite storyboard IDs that were verified or skipped. Hardware-
dependent real-data CUDA/MPS checks may still need a separate machine-specific
runbook.

## Tag and publish to TestPyPI

After the local gates pass, create and push the release-candidate tag:

```bash
git tag widget-v0.0.1rc1
git push origin widget-v0.0.1rc1
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
  quantem-widget==0.0.1rc1
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
widget-v0.0.1rc2
```

Fix the issue on a normal branch, merge through a PR, pull updated `main`, rerun
the gates, and then push the next release-candidate tag.
