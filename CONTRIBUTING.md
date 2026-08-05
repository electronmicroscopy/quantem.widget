# Contributing to quantem.widget

`quantem.widget` is an interactive widget package, so a passing import is not
enough. A good contribution should keep the Python API, frontend bundle, saved
widget state, standalone HTML export, and documentation in sync.

## Branches and review

Work on a small themed branch per issue and open the PR from it. Once a PR is
under review, **do not force-push the branch**: rewriting history detaches the
reviewer's inline comments to "outdated", removes GitHub's "changes since your
last review" diff so the reviewer must re-read the whole PR, and breaks any
local checkout of the branch. Push ordinary follow-up commits instead and use
`git revert` to undo; the merge is squashed, so a messy branch history costs
nothing. New to the Git/GitHub workflow? Start with the tutorials in
[ophusgroup/dev](https://github.com/ophusgroup/dev).

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
PYTHONPATH=src:. pytest -q
npm run typecheck
npm test
npm run build
```

For HTML export or widget-state changes:

```bash
PYTHONPATH=src:. pytest -q tests/test_html_export_protocol.py
```

For exported-HTML UI changes, also run the browser-drive smoke:

```bash
scripts/widget_local_signoff.sh --quick --browser
```

For ShowEDS changes:

```bash
PYTHONPATH=src:. pytest -q tests/test_showeds.py tests/test_html_export_protocol.py
```

For Show4DSTEM/WebGPU-sensitive changes:

```bash
QT_RUN_BROWSER_TESTS=1 PYTHONPATH=src:. pytest -q tests/test_show4dstem_webgpu_browser.py -s
```

For a quick visual smoke before a broad widget PR:

```bash
scripts/widget_visual_signoff.sh --quick
```

For interaction-sensitive changes, run an agent signoff packet and drive the
widgets in the browser:

```bash
scripts/widget_agent_signoff.sh --quick
```

## Tutorial Notebooks

Canonical notebooks under `docs/tutorials/` remain interactive source
notebooks. Do not commit baked widget state (`metadata.widgets`) or re-execute
them merely to store outputs; run `scripts/check_notebook_sizes.py` before a
docs PR. Documentation CI executes the notebooks and sets
`QUANTEM_WIDGET_STATIC_FALLBACK=0`, so each widget cell renders one live widget
instead of a duplicate static preview. Do not change the docs build to notebook
cache mode, which drops saved widget state.

Keep Colab bootstrap plumbing in its own cell, tagged `remove-cell`, with the
installation command strictly inside the successful `google.colab` import
branch. Do not mix that bootstrap with tutorial code.

Use vectorized NumPy or Torch operations for tutorial data generation rather
than large Python loops. Collapse bulky synthetic-data construction when it is
not the teaching point, and keep the rendered scientific result as the focus.
If Torch is not otherwise required, import it locally and retain a NumPy path.

## Performance Expectations

For interactive widget changes, verify performance in the actual user path, not
only with unit tests. Open the affected notebook or exported HTML, drive the
changed controls, and check the widget debug HUD or another direct timing signal
when available.

Include at least one organic user path: begin with a minimal call such as
`Show2D(data)` or `Show3D(stack)` and enable the behavior through the UI. A
constructor configured with many test-only options is useful for a focused
regression, but does not replace the normal scientist workflow.

Report load, render, and interaction timings in the PR or handoff. At minimum,
name the data shape, dtype, raw size, backend, loader time, widget construction
time, first browser-paint time when available, and FPS/latency for the changed
interaction. Prefer `verbose=True` in debug notebooks so the output is
copyable. Use `quantem.widget.profile_widget` for notebook profiling when
possible so load, pack, and widget-build timings use the same table format
across viewers. The full timing checklist lives in
`docs/maintainer/widget-performance.md`.

Any loss of real-time interaction is a bug to fix or explicitly document. For
ShowEDS real-data workflows, band, ROI, zoom, contrast, and smooth/auto display
interactions should remain at 30 FPS or better.

Use `scripts/widget_agent_signoff.sh` for a fix-and-redrive session: the agent
opens the real notebook, docs page, or exported HTML, drives the controls like a
human, patches issues immediately, rebuilds, refreshes, and records screenshots
or short videos after the final code change. The full protocol is in
`docs/maintainer/widget-agent-signoff.md`.

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
- `README.md` and `CONTRIBUTING.md`

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

Use the repository pull request template.

## Commit Messages

Use a short Conventional Commit-style first line:

```text
type: short imperative summary
```

Use these common types:

- `feat:` for a user-facing feature.
- `fix:` for a bug fix.
- `docs:` for documentation-only changes.
- `test:` for test-only changes.
- `refactor:` for internal restructuring without behavior changes.
- `perf:` for performance improvements.
- `build:` for packaging, dependencies, or build tooling.
- `ci:` for GitHub Actions or CI workflow changes.
- `chore:` for maintenance that does not fit the above.

Examples:

```text
feat: add ShowFolder thumbnail cache
fix: stabilize Show2D resize handle
docs: update HTML export protocol
```

Prefer one-line commit messages unless a detailed body is requested. Do not add
`Co-authored-by` unless the contributor explicitly asks for it.

## Release Candidates

Release candidates publish to TestPyPI from `widget-v*` tags. Follow
`docs/maintainer/widget-release.md` before creating or pushing a release tag.
