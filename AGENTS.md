# quantem.widget

Interactive Jupyter widgets and standalone browser GUI for electron microscopy.

## Commit Messages

Use short Conventional Commit-style first lines for routine commits:

```text
type: short imperative summary
```

Use `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`, `build:`,
`ci:`, or `chore:` followed by a concise summary. Examples:
`fix: stabilize Show2D resize handle`, `docs: update release checklist`.
Keep commit messages single-line unless a detailed body is requested, preserve
the user's configured author identity, and do not add `Co-authored-by` trailers
unless explicitly requested.

## User-facing export language

For docs, menus, README text, and tutorials, use science-friendly export terms:

- Say **single** for a one-file HTML export. If precision matters, say
  **exact single**.
- Say **folder** for an HTML file that reads exact data from a nearby data
  folder or URL. Avoid leading with "sidecar" in user-facing text.
- Say **downsample** or **downsampled** for reduced-shape one-file exports.
  Binning may be mentioned as the widget-specific reducer, but the preferred
  public API option is `downsample`.
- Say **encoding** for stored data representation, such as `encoding="full"` or
  `encoding="uint8"`. Avoid using "quantized" as the preferred public API name.
- It is fine to mention old/internal parameter aliases in parentheses, for
  example "folder (`mode=\"folder\"`; old alias `mode=\"sidecar\"`)".

User-facing `mode` values should stay simple: `mode="single"` or
`mode="folder"`. Do not introduce public labels such as "sidecar",
"linked folder", or "linked data folder" unless documenting a legacy alias.

The package-wide HTML export API standard is:

```python
widget.export_html(path=None, title=None, mode="single", encoding="full", downsample=None)
```

Use compatibility names only when preserving existing APIs:
`quantized=True` -> `encoding="uint8"`, `dtype="uint8"` -> `encoding="uint8"`,
`det_bin=2` -> `downsample=2`, and `binning=4` -> `downsample=4`.

Keep implementation names such as `sidecar_url`, `from_sidecar`, and
`prepare_spectrum_image_sidecar` unchanged unless doing a deliberate API
migration.

## Release Workflow

For any `quantem.widget` release or release-candidate work, follow
`docs/maintainer/widget-release.md`.

For routine local readiness checks, run `scripts/widget_local_signoff.sh`.
Use `--quick` while iterating, `--quick --browser` for exported-HTML frontend
changes, and `--full --performance` before broad UI or release-candidate work.
See `docs/maintainer/automation.md`.
Each run writes a visual report directory with a top-level `index.html`; use
`--artifact-dir` when another agent or the user needs a stable path to inspect.

Do not create or push a `widget-v*` tag until the required local gates in that
runbook pass. For RCs intended for TestPyPI, also run the real browser/Jupyter
user-path checks called out in the runbook before tagging.

## Performance Verification

Interactive speed is a required part of widget correctness. For any frontend,
WebGPU, notebook-state, or export change that can affect a widget interaction,
drive the affected widget in JupyterLab or standalone HTML and verify that the
change did not regress responsiveness. Use the widget's debug HUD or another
direct timing signal when available, and report the measured FPS/latency in the
handoff or final summary. Do not rely only on unit tests for interaction-heavy
changes.

For UI performance patterns and known interaction mistakes, read
`docs/maintainer/widget-performance.md`. In particular, cursor labels, hover
readouts, drag hints, and other high-frequency pointer overlays should avoid raw
React state updates on every `mousemove`/touchmove; use animation-frame
scheduling, stable overlay DOM, and opacity/transform transitions where
possible.

For ShowEDS real-data work, keep band, ROI, zoom, contrast, and smooth/auto
display interactions at real-time speed. Treat loss of 30 FPS interaction as a
bug unless the limitation is explicitly documented and accepted.

## Runtime and Backend Choice

Do not run heavy widget workloads in the local Codex bundled runtime, system
Python, or a laptop CPU-only environment. Those are acceptable only for
lightweight inspections, small unit tests, and quick syntax/type checks.

For large tutorial data generation, exported-HTML browser reports, real-data
performance probes, long notebook runs, or GPU/WebGPU stress tests, use the
project development backend or an HPC/GPU-capable backend so the user's laptop
does not freeze. Keep generated reports and scratch artifacts outside the repo
unless they are intentionally promoted into a documented maintainer workflow.

## Widget UI Consistency

For widget visual style and control wording, read
`docs/maintainer/widget-ui-protocol.md` before editing frontend controls.

For in-widget toolbar dropdowns, use the shared MUI `Select`/`MenuItem` pattern
with the widget's `themedSelect` and `themedMenuProps` styling. Do not add
native HTML `<select>` controls to widget toolbars. Match the Show4DSTEM and
Show3D menu behavior so dropdown size, theme, z-order, and keyboard behavior
stay consistent across viewers.

Keep compact widget control labels free of decorative colons. Use labels such
as `Scale`, `Color`, `Auto`, `Smooth`, `Link`, `Zoom`, `Pan`, `Contrast`,
`FFT`, and `ROI` in toolbars and control rows. Colons are fine in explanatory
prose, tooltips, docs, and other full sentences where they improve readability,
but avoid them in dense controls where they waste horizontal space.

Keep each compact label grouped with the control it names. A label plus switch,
dropdown, slider, or compact button should wrap as one unit on mobile and
narrow layouts; the row may wrap, but the label must not separate from its
control. Verify this in the browser for Show2D/Show3D-style rows such as
`Auto`, `Smooth`, `Zoom`, `Pan`, `Contrast`, `fps`, and `avg`.

Use Title Case for command buttons and toolbar actions: `Copy`, `Export`,
`Reset`, `Add`, `Clear`, `Undo`, `Save Band`, and similar actions. Keep
scientific acronyms, detector labels, and file-format names uppercase when the
uppercase form is the term users recognize, such as `FFT`, `ROI`, `BF`, `ABF`,
`ADF`, `HTML`, `PNG`, `GIF`, and `MP4`.

## Agent Handoff Reporting

Whenever an agent finishes a concrete task, the final update must clearly state:

- **Done**: what changed or what was verified.
- **Next**: the immediate recommended next action for the user or agent.
- **Missing / not verified**: anything still untested, blocked, unstaged,
  uncommitted, unpushed, or needing user confirmation. If nothing is missing,
  say that explicitly.

Keep this short, but do it every time. The goal is that a user can leave and
come back knowing exactly where the work stands and what should happen next.

## Repository Hygiene

Keep public documentation in durable paths such as `README.md`,
`CONTRIBUTING.md`, `docs/api/`, `docs/tutorials/`, and `docs/maintainer/`.
Do not commit local session notes, one-off benchmark scripts, AppleDouble
`._*` files, generated docs builds, screenshots, or scratch files unless they
have been promoted into a documented test, example, or maintainer runbook.

Keep the main branch simple. Do not split documentation into a separate repo or
introduce Git LFS just to support routine widget docs. Prefer source notebooks,
small real rendered examples, and generated docs output. Use public data hosting
only when real tutorial data or standalone HTML would make normal clones
unnecessarily large.

## Tutorial Data Generation

When generating synthetic or real-derived arrays for `docs/tutorials/`, docs
builds, smoke reports, or browser probes, use vectorized NumPy or Torch
operations. Prefer Torch for larger lattice, page-stack, movie-stack, FFT, or
multi-panel examples because it keeps array construction fast on CPU/GPU
backends and avoids slow Python loops in notebooks.

Keep tutorials readable: hide or collapse bulky data-construction cells when
they are not the teaching point, and make the rendered widget or result the
focus. If Torch is not already a dependency for the path being edited, keep the
import local to the tutorial/helper and fall back to NumPy when Torch is
unavailable.
