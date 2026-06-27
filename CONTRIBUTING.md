# Contributing to quantem.widget

`quantem.widget` is an interactive widget package, so a passing import is not
enough. A good contribution should keep the Python API, frontend bundle, saved
widget state, standalone HTML export, and documentation in sync.

## Setup

Use Python 3.11 or newer and Node.js 22 when possible:

```bash
python -m pip install -e .
npm ci
```

For documentation work:

```bash
python -m pip install -r docs/requirements.txt
python -m pip install -e .
```

## Everyday Checks

Run the smallest checks that match the change:

```bash
PYTHONPATH=src pytest -q
npm run typecheck
npm test
npm run build
```

For HTML export or widget-state changes:

```bash
PYTHONPATH=src pytest -q tests/test_html_export_protocol.py
```

For ShowEDS changes:

```bash
PYTHONPATH=src pytest -q tests/test_showeds.py tests/test_html_export_protocol.py
```

For Show4DSTEM/WebGPU-sensitive changes:

```bash
QT_RUN_BROWSER_TESTS=1 PYTHONPATH=src pytest -q tests/test_show4dstem_webgpu_browser.py -s
```

For a quick visual smoke before a broad widget PR:

```bash
scripts/widget_visual_signoff.sh --quick
```

## HTML Export Expectations

All export-capable widgets should follow the public shape:

```python
widget.export_html(path=None, title=None, mode="single", encoding="full", downsample=None)
```

Use science-friendly public terms:

- `mode="single"` for a one-file HTML export.
- `mode="folder"` when large exact data lives next to the HTML file.
- `encoding` for stored data representation, for example `"full"` or `"uint8"`.
- `downsample` for shape reduction.

Do not introduce public labels such as "sidecar" or "linked folder" for new
user-facing APIs. The canonical reference is
`docs/api/html-export.md`.

## What To Commit

Commit durable source, tests, and public documentation:

- `src/quantem/widget/**`
- `js/**` when it is production frontend code or a real test
- `tests/**` when the files are tests, not local screenshots
- `docs/api/**`, `docs/tutorials/**`, and `docs/maintainer/**`
- `README.md`, `CONTRIBUTING.md`, and `AGENTS.md`

Do not commit local or generated files:

- AppleDouble metadata such as `._README.md`
- `docs/_build/`
- `node_modules/`, `build/`, and `dist/`
- local screenshots and browser debug images
- one-off files named like `_fpbench.ts`, `_f32test.ts`, or session notes unless
  they are promoted into a documented benchmark or test
- private notebooks, data, or machine-specific paths

Before opening a PR, run:

```bash
git status --short
git diff --stat
```

The diff should contain only files that belong to the change.

## Pull Requests

Keep PRs focused. Widget behavior, release process, and documentation cleanups
are easier to review as separate PRs.

Use the repository pull request template. Prefer a short single-line commit
message unless a detailed body is requested. Do not add `Co-authored-by` unless
the contributor explicitly asks for it.

## Release Candidates

Release candidates publish to TestPyPI from `widget-v*` tags. Follow
`docs/maintainer/widget-release.md` before creating or pushing a release tag.
