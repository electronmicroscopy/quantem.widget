# Creating a widget

This guide shows how to add a new interactive viewer or tool to `quantem.widget`. A new widget needs an issue or maintainer discussion first ([Pull requests](../maintainer/pull-requests.md)). Instructions for agentic-driven development are on the [Agent-assisted development](../agent-prompts) page, and the workflow for setup, test, and incremental PRs is in [CONTRIBUTING.md](https://github.com/electronmicroscopy/quantem.widget/blob/main/CONTRIBUTING.md).

Let's use `Ruler2D` as an example, a widget where the user places two endpoints on a 2D image and reads back a calibrated distance.

A widget has two parts:

- a Python `traitlets` class in [`src/quantem/widget/`](https://github.com/electronmicroscopy/quantem.widget/tree/main/src/quantem/widget), which owns the scientific data, validation, state, and scriptable API; and
- a React entry point in [`js/<bundle>/index.tsx`](https://github.com/electronmicroscopy/quantem.widget/tree/main/js), bundled by esbuild into `src/quantem/widget/static/<bundle>.js` and loaded through the Python class's `_esm` path.

Traits tagged with `sync=True` are the interface between Python and the browser.

Keep file I/O, GPU work, fitting, scientific transforms, and anything users may want to script on the Python side. Keep rendering and temporary interaction state in the browser.

Each widget should be self-contained, meaning that its Python module and `js/<bundle>/` folder holds everything specific to it. Adding or changing one widget should have as few side effects on the others as possible.

| Kind of code | Where it goes |
|---|---|
| Widget state, layout, and interactions | `src/quantem/widget/<widget>.py` and `js/<bundle>/` |
| Frontend helpers for several widgets | shared modules at the top of [`js/`](https://github.com/electronmicroscopy/quantem.widget/tree/main/js) |
| Display math - colormaps, ranges, histograms, display FFT, display geometry | [`js/display/`](https://github.com/electronmicroscopy/quantem.widget/tree/main/js/display) |
| WebGPU science kernels - detector reductions, HDF5/bslz4 browser IO, SSB | [`quantem.gpu`](https://github.com/bobleesj/quantem.gpu) |

Display math is widget source. Each operation has a JavaScript path that runs
without WebGPU and a WebGPU path that must give the same pixels; the widget
shows which one is active (`WebGPU` or `CPU` in the title row).

Science kernels belong to `quantem.gpu`.
[`scripts/sync-gpu-webgpu.mjs`](https://github.com/electronmicroscopy/quantem.widget/blob/main/scripts/sync-gpu-webgpu.mjs)
copies its `detector/`, `dpc/`, `formats/`, `io/` and `ssb/` sources into
`js/.generated/engine/` for the two bundles that use them, `show4dstem` and
`showptycho`. Building those bundles needs the quantem.gpu source (set
`QUANTEM_GPU_SRC=/path/to/quantem.gpu/src`, or install quantem.gpu in the Python
named by `PYTHON`); without it the build stops and names `QUANTEM_GPU_SRC`.
Every other bundle builds from widget source alone:

```bash
npm run build -- show2d show3d
```

Always edit science kernels in `quantem.gpu`, never the generated copy.

## Before you start

A new public widget is a discuss-first change. Open an issue or confirm with a
maintainer that the widget belongs in this package before choosing a base class
or writing React code. Incremental fixes and features that stay inside one
existing widget skip this step. See [Pull requests](../maintainer/pull-requests.md).

Before choosing a base class or writing React code, define what the widget is supposed to do.

For `Ruler2D`, the example would be that it accepts one 2D image, stores coordinates as `(row, col)` in the original image, allows two endpoints to be placed or dragged, and reports their calibrated distance.

This should be consistent across the Python API, frontend, tests, and documentation.

## Step 1: Choose an implementation

| Base | Use when | Example |
|---|---|---|
| Subclass an existing viewer (`Show2D`, `Show3D`, …) | The new widget is mostly an existing viewer with an additional interaction or analysis tool | [`Show3DSlices`](../api/show3dslices): [`show3dslices.py`](https://github.com/electronmicroscopy/quantem.widget/blob/main/src/quantem/widget/show3dslices/widget.py), [`js/show3dslices/index.tsx`](https://github.com/electronmicroscopy/quantem.widget/blob/main/js/show3dslices/index.tsx) |
| Standalone `anywidget.AnyWidget` | The widget has its own layout or only needs a small part of the existing viewer infrastructure | [`ChooseLattice`](../api/choose-lattice): [`choose_lattice.py`](https://github.com/electronmicroscopy/quantem.widget/blob/main/src/quantem/widget/choose_lattice.py), [`js/chooselattice/index.tsx`](https://github.com/electronmicroscopy/quantem.widget/blob/main/js/chooselattice/index.tsx) |

`Ruler2D` is better as a standalone widget. It only needs an image and two endpoints, so carrying the complete `Show2D` state would add unnecessary complexity. When unsure, start by finding the existing widget whose behavior is closest to what you are building.

## Step 2: Follow the naming conventions

| Piece | Convention | `Ruler2D` example |
|---|---|---|
| Public class | CapWords; `Show<Data>` for general viewers | `Ruler2D` |
| Python module | snake_case, normally one widget per module | `src/quantem/widget/ruler2d.py` |
| Frontend bundle | lowercase class name without separators | `js/ruler2d/index.tsx` → `static/ruler2d.js` |

The frontend bundle name is used in several build and release files, so keep its spelling identical everywhere.

## Step 3: Create the Python widget

Create the widget class in:

```text
src/quantem/widget/ruler2d.py
```

The Python class should define the public scientific interface. 

For instance, in `Ruler2D` Python would own the image, calibration, endpoint coordinates, and calculated distance.

The JavaScript bundle is loaded through `_esm`:

```python
_esm = pathlib.Path(__file__).parent / "static" / "ruler2d.js"
```

Use synchronized traits only for state that must cross between Python and the browser. Temporary interaction details (like hover or mouse position) should remain only on the front end.

When setting several synchronized traits during initialization, use `hold_sync()` so they are sent together.

It's helpful to validate inputs when they enter the widget. Errors should identify what was invalid and what the user should provide instead.

You can look at the closest existing widget for the preferred constructor, validation, state, and `save_state` patterns.

## Step 4: Create the frontend

Create the React entry point in:

```text
js/ruler2d/index.tsx
```

Ideally, the frontend parses synchronized traits, renders the widget/interactions, and writes user changes back to Python.

Use the existing frontend infrastructure:

| Module | Purpose |
|---|---|
| [`js/theme.ts`](https://github.com/electronmicroscopy/quantem.widget/blob/main/js/theme.ts) | widget themes and host environment |
| [`js/format.ts`](https://github.com/electronmicroscopy/quantem.widget/blob/main/js/format.ts) | trait decoding and formatting |
| [`js/display/colormaps.ts`](https://github.com/electronmicroscopy/quantem.widget/blob/main/js/display/colormaps.ts) | colormaps and image rendering |
| [`js/display/stats.ts`](https://github.com/electronmicroscopy/quantem.widget/blob/main/js/display/stats.ts) | display ranges and image statistics |
| [`js/figure.ts`](https://github.com/electronmicroscopy/quantem.widget/blob/main/js/figure.ts) | scale bars and figure formatting |
| [`js/staticFallback.ts`](https://github.com/electronmicroscopy/quantem.widget/blob/main/js/staticFallback.ts) | saved-notebook fallback behavior |

For `Ruler2D`, dragging an endpoint can update the line locally in the browser and synchronize the final coordinate when the user releases the pointer.

If the frontend develops numerical logic, move that logic into a separate TypeScript module, so that it can be tested independently of React. Keep that module inside the widget's own folder unless another widget needs it, as [`js/showdiffraction/measurements.ts`](https://github.com/electronmicroscopy/quantem.widget/blob/main/js/showdiffraction/measurements.ts) does.

## Step 5: Scientific vs display coords

Interactive image widgets often transform the image for zooming, panning, or resizing. Those display transformations should not change the coordinates exposed through the scientific API.

For `Ruler2D`, endpoint coordinates remain `(row, col)` positions in the original image regardless of how the image is displayed in the browser.

The same should apply to masks, diffraction spots, lattice points, line profiles, etc.

## Step 6: Register the widget

Once the Python class and frontend exist, register the widget with the package and build system, update:

| File | What to add |
|---|---|
| [`scripts/build.mjs`](https://github.com/electronmicroscopy/quantem.widget/blob/main/scripts/build.mjs) | frontend bundle |
| [`src/quantem/widget/__init__.py`](https://github.com/electronmicroscopy/quantem.widget/blob/main/src/quantem/widget/__init__.py) | lazy public export |
| [`scripts/widget_release_check.sh`](https://github.com/electronmicroscopy/quantem.widget/blob/main/scripts/widget_release_check.sh) | generated JS bundle |
| [`.github/workflows/widget-release.yml`](https://github.com/electronmicroscopy/quantem.widget/blob/main/.github/workflows/widget-release.yml) | generated JS bundle |

Then build the frontend and confirm that the public import works:

```bash
npm run build
PYTHONPATH=src:. python -c "from quantem.widget import Ruler2D" # replace Ruler2D with widget
```

If the widget supports additional protocols such as saved interactive state or HTML export, also register it in the corresponding protocol tests, [`tests/test_save_state.py`](https://github.com/electronmicroscopy/quantem.widget/blob/main/tests/test_save_state.py) and [`tests/test_html_export_protocol.py`](https://github.com/electronmicroscopy/quantem.widget/blob/main/tests/test_html_export_protocol.py).

## Step 7: Add tests

Recent widgets keep their tests in a folder of their own, which is the preferred layout for a new widget:

```text
tests/ruler2d/test_ruler2d.py
```

Test the behavior a user relies on.

For `Ruler2D`, the important cases are that valid 2D input works, invalid input is rejected, and known endpoints with known calibration produce the correct distance. [`tests/plot2d/`](https://github.com/electronmicroscopy/quantem.widget/tree/main/tests/plot2d) is a small example to read.

Frontend numerical helpers can be tested with Vitest.

Use browser tests - for behavior that requires users interaction, like clicking or zooming.

## Step 8: Notebooks

Large synchronized arrays can make saved notebooks unnecessarily large.

Widgets that contain substantial image or array state should follow the existing `save_state` protocol, described in [Save state & notebook size](save-state-and-notebook-size). By default, large data should not be embedded in the notebook.

## Step 9: Add HTML export

Standalone HTML export is optional.

Implement it when the widget produces something users are likely to save or share outside a live Python session. This can help potential users reproduce your widget without needing to launch a Jupyter session!

Add widgets supporting export to the corresponding HTML export tests.

## Step 10: Document the widget

Documentation is part of adding the widget.

Add:

```text
docs/api/ruler2d.md
```

and update:

```text
docs/api/index.md
docs/_toc.yml
README.md
```

The API page should explain what the widget does and its important properties.

Use an interactive-controls table when the widget has several controls. 

Add a tutorial notebook when the widget has a workflow that needs more explanation than can be done through solely the API page.

## Step 11: Checks

As good practice, manually check to see that the widget works in the browser and can be reproduced. 

During development, run the unit tests:

```bash
PYTHONPATH=src:. pytest -q tests/ruler2d
```

Then run the frontend checks:

```bash
npm run build
npm run typecheck
npm test
```

Before opening a pull request, run the test suite and the size guards:

```bash
PYTHONPATH=src pytest -q
python scripts/check_large_files.py
python scripts/check_notebook_sizes.py
```

The widget should now have a Python API, frontend bundle, tests, documentation and release registration consistent with the rest of `quantem.widget`.
