# Automation

Use these commands to keep widget work repeatable across humans and agents.
Automation here should stay small, boring, and testable. Each script has one
job; `scripts/widget_local_signoff.sh` is the only general entrypoint.

## Local Signoff

Run the default signoff before saying a widget change is ready:

```bash
scripts/widget_local_signoff.sh
```

This runs repository size guards, frontend build, the full Python test suite,
HTML export smoke coverage for every export-capable widget, and the docs build.

For faster iteration while fixing one issue:

```bash
scripts/widget_local_signoff.sh --quick
```

For release-oriented validation:

```bash
scripts/widget_local_signoff.sh --full --performance
```

For visual/frontend-sensitive widget work, add the browser-drive gate:

```bash
scripts/widget_local_signoff.sh --quick --browser
```

Every signoff run writes a visual report directory. By default it uses:

```text
/tmp/quantem-widget-local-signoff/<timestamp>/
```

Use a fixed path when coordinating with another agent or serving the report over
Tailscale:

```bash
scripts/widget_local_signoff.sh --quick --artifact-dir /tmp/quantem-widget-signoff
```

Open `/tmp/quantem-widget-signoff/index.html` after the run. The top-level
report links to the widget export report and, when `--performance` is used, the
real-data performance report. The normal HTML smoke report contains:

- `index.html`: a visual table with links to every standalone widget export.
- `report.json`: sizes and backend export timings.
- `browser-plan.json`: the pages and interactions an agent should drive in the
  in-app browser for visual signoff.
- `browser-smoke.html` and `browser-smoke-report.json`, when `--browser` is
  used: automated Chromium results, screenshots, nonblank canvas checks, and
  basic interaction checks.

The performance mode writes real-data Show2D and Show3D HTML exports plus its
own `index.html`, `report.json`, and `browser-plan.json`. It is intentionally
separate from the default path because it can touch large real-data files.
Python timings cover backend loading and export packing; browser FPS still
requires opening the generated pages in the in-app browser and driving the
interactions from the browser plan.

Update this script when the project-wide definition of "ready" changes. Do not
add widget-specific debugging experiments here; put those in a focused test or
storyboard instead.

## CI Signoff

`.github/workflows/widget-ci.yml` runs the default local signoff on pull requests
and pushes that touch widget source, tests, scripts, or docs. Keep the workflow
aligned with `scripts/widget_local_signoff.sh` so local and CI behavior stay the
same.

If CI fails but local signoff passes, treat that as a dependency or environment
drift bug and fix the workflow rather than adding a second CI-only definition of
readiness.

## Docs Preview

Build and serve local documentation:

```bash
scripts/docs_preview.sh
```

Then open the printed local URL in a browser. To serve an existing build:

```bash
scripts/docs_preview.sh --no-build
```

Use this for rendered documentation feedback. Do not commit `docs/_build`.

This script is intentionally not part of CI. It is a local preview server for
visual review.

## HTML Export Smoke

Generate standalone HTML for the export-capable widgets:

```bash
PYTHONPATH=src:. python scripts/widget_html_smoke.py
```

The smoke covers `Show2D`, `Show3D`, `Show3DSlices`, `Show4DSTEM`, `ShowEDS`,
`ShowDiffraction`, and `ShowFolder`. `Show2D` and `Show3D` intentionally get a
small settings matrix rather than a single happy-path export: single panel,
multi-panel galleries, FFT modes, hidden/starred panels, compact/no-title
layouts, panel visibility, multi-panel Show3D, and downsampled Show3D export.
It verifies that each export writes widget state and expected content markers.
It also writes an `index.html` visual report and `browser-plan.json` in the
artifact directory, so another agent or reviewer can open the generated pages
and inspect the widgets directly.

Update this smoke when a new public widget gains `export_html()`, or when an
existing widget's canonical small export options change. Keep the datasets tiny;
this checks export protocol coverage, not heavy interaction performance.

## Browser HTML Smoke

Drive the generated standalone HTML in Chromium:

```bash
PYTHONPATH=src:. python scripts/widget_browser_smoke.py --artifact-dir /tmp/quantem-widget-signoff/html-smoke
```

This is the automated browser layer for exported HTML. It opens every page from
`report.json`, verifies visible canvases are nonblank, performs wheel/drag
interactions on the primary canvas, toggles visible switches, drags a slider
when one is present, records console/page/HTTP errors, and saves screenshots.

Run this after `scripts/widget_html_smoke.py`, or use
`scripts/widget_local_signoff.sh --quick --browser` to run both in order. Use
`--headed` on `scripts/widget_browser_smoke.py` when you want to watch the
browser drive around. Known harmless Chromium bootstrap warnings are recorded
separately from failures; rendering, HTTP, page, and unexpected console errors
still fail the gate.

## Size Guards

Keep the main branch lightweight:

```bash
python scripts/check_large_files.py
python scripts/check_notebook_sizes.py
```

These fail when tracked files, tutorial notebooks, or embedded notebook outputs
grow beyond the project thresholds. Small real rendered examples are acceptable;
large arrays and large HTML payloads should be generated during docs builds or
downloaded from public data hosting only when the size justifies it.

The guard scripts are tested by `tests/test_automation_scripts.py`. When changing
thresholds or failure wording, update those tests in the same commit.

## Script Ownership

| File | Why it exists | Runs by default | Update when |
| --- | --- | --- | --- |
| `scripts/widget_local_signoff.sh` | One local command for normal readiness gates. | Yes, in CI and local signoff. | The project-wide readiness definition changes. |
| `scripts/check_large_files.py` | Prevent accidental large tracked data or rendered artifacts. | Yes. | File-size policy or allowlisted artifact types change. |
| `scripts/check_notebook_sizes.py` | Keep tutorial notebooks and embedded outputs clone-friendly. | Yes. | Notebook size policy changes. |
| `scripts/widget_html_smoke.py` | Verify every export-capable widget writes standalone HTML state, a visual report, and a browser-drive plan. | Yes. | A widget adds/removes/changes HTML export support. |
| `scripts/widget_browser_smoke.py` | Open generated HTML exports in Chromium, check nonblank canvas rendering, drive basic controls, and save screenshots. | No, only `--browser`. | Exported widget browser behavior, interaction contracts, or report format changes. |
| `scripts/widget_performance_smoke.py` | Record real-data Show2D/Show3D export timing, payload sizes, report HTML, and browser-drive plan. | No, only `--performance`. | Real-data performance expectations change. |
| `scripts/docs_preview.sh` | Build and serve docs for local visual review. | No. | The docs build command or served path changes. |
| `.github/workflows/widget-ci.yml` | Run the same local signoff on PRs and main pushes. | Yes, on matching GitHub events. | Local signoff dependencies or trigger paths change. |

Avoid adding new automation files unless one of these scripts cannot reasonably
own the behavior. Prefer adding a focused test before adding another script.
