# quantem.widget

[![TestPyPI](https://img.shields.io/pypi/v/quantem-widget?pypiBaseUrl=https://test.pypi.org&label=TestPyPI)](https://test.pypi.org/project/quantem-widget/)

Interactive WebGPU visualization widgets for 4D-STEM and electron microscopy - in
Jupyter, or straight from the [command line](#command-line). Works with NumPy, PyTorch,
or CuPy arrays.

> Prototype on [TestPyPI](https://test.pypi.org/project/quantem-widget/). Built on
> [`quantem`](https://github.com/electronmicroscopy/quantem) core.

## Install

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ quantem.widget
```

Verify:

```bash
python -c "import quantem.widget; print(quantem.widget.__version__)"
```

## Widgets

| Widget | Input | Shows |
|---|---|---|
| `Show2D` | 2D image or stack | image + contrast, FFT, line profiles, scale bar |
| `Show3D` | 3D stack | scrub / play through frames |
| `Show3DSlices` | 3D volume | orthogonal-slice viewer |
| `Show4DSTEM` | 4D-STEM array | live virtual detectors (BF / ABF / ADF), CoM / iCoM / DPC |
| `ShowEDS` | EDS/EELS spectrum image | experimental linked element map, spectrum, energy band, and real-space ROI |
| `ShowFolder` | microscopy session folder | fast thumbnail browser, grouping, and file selection |

```python
import numpy as np
from quantem.widget import Show2D, Show3D, Show3DSlices, Show4DSTEM, ShowEDS, ShowFolder

Show2D(np.random.rand(512, 512))
Show4DSTEM(np.random.rand(64, 64, 128, 128))
ShowEDS(np.random.poisson(2, (64, 64, 256)).astype("uint16"))
ShowFolder("/data/session")
```

## Load data

```python
from quantem.widget import ShowEDS, Show4DSTEM, load, load_eds

data = load("scan_master.h5")   # Arina 4D-STEM .h5 -> GPU
Show4DSTEM(data)

eds = load_eds("spectrum_image.emd")  # Velox/RSCIIO EDS/EELS -> energy-last SpectrumImage
ShowEDS(eds, energy=8.04, width=0.24)
```

`quantem.widget.io` also provides `read_image`, `bin`, `download`, and more -
see the docs.

## Command line

Point `quantem` at a file or folder and it renders the right viewer - no notebook, no
Python. Installing the package adds the `quantem` command.

```bash
quantem show ./anything/                     # auto-detect content, pick the viewer
quantem show2d scan.png                      # an image            -> Show2D
quantem show3d ./frames/                     # a folder of frames  -> Show3D scrub
quantem show2d ./frames/ --watch             # live folder         -> append new images
quantem show4dstem ./masters/                # *_master.h5         -> live Show4DSTEM
quantem show4dstem a_master.h5 b_master.h5   # several masters     -> one 5D multi-tilt viewer
quantem show4dstem ./masters/ --html         # 4D-STEM             -> shareable offline HTML
quantem showfolder ./session/                # microscopy folder   -> ShowFolder notebook/HTML
quantem data-transfer plan ./raw/ /ssd0/run /ssd1/run --manifest run.json
quantem html tutorial.ipynb                  # a notebook          -> standalone offline HTML
```

| Command | Input | Output |
|---|---|---|
| `quantem show <path>` | anything | auto-detects and dispatches to one of the below |
| `quantem show2d <img / folder>` | one image, or a folder | Show2D HTML (a folder becomes a gallery); with `--watch`, a live ShowFolder notebook |
| `quantem show3d <folder>` | a folder of same-size frames | Show3D scrub HTML; with `--watch`, a live ShowFolder notebook |
| `quantem show4dstem <master(s) / folder>` | one or more `*_master.h5` | live Show4DSTEM notebook (or `--html`) |
| `quantem showfolder <folder>` | microscopy session folder | ShowFolder notebook (or `--html`) |
| `quantem data-transfer plan/inspect/copy` | `*_master.h5` folder plus target roots | manifest-backed transfer planning, state inspection, and explicit copy |
| `quantem html <notebook.ipynb>` | a notebook you wrote | runs it, bakes outputs into one offline HTML |

**Images** save a standalone HTML and open in your browser. **4D-STEM** opens a live,
kernel-backed notebook by default (full real-time interaction); `--html` instead bakes a
**self-contained offline viewer that runs entirely on WebGPU** - drag detectors, switch
BF/ABF/ADF, pan diffraction, all with no kernel. The detector is **mean-binned** (`--bin`,
default 8) so the packed stack stays small and fits a laptop's browser - you can browse
data that never fit full resolution. Several masters (a folder, or listed explicitly)
stack into one 5D viewer with a dataset slider (the multi-tilt case).

For live microscope sessions, keep the Show4DSTEM viewer mounted and append new
completed `*_master.h5` acquisitions into the same dataset slider. On Apple
Silicon, use `load_macbook_datasets(...)` and `live.watch_master_folder(...)`;
see the [Show4DSTEM API](docs/api/show4dstem.md#live-scope-folders) and
[Show4DSTEM storyboard](docs/maintainer/storyboard-show4dstem.md#s4d-14-append-live-scope-acquisitions).

**Notebooks**: `quantem html notebook.ipynb` is the share path for a tutorial or report
you wrote. It runs every cell, then bakes the outputs (Show2D/Show3D widgets included, as
static images) into one self-contained HTML that opens in any browser with no Python or
kernel. Use `--no-execute` to wrap the already-saved outputs as-is. The command prints the
file size so you know how heavy the share artifact is.

For GitHub notebook previews, make a copy and run
`quantem github notebook_github.ipynb --no-execute`. GitHub cannot run live widgets, so
this command keeps compressed pictures of each widget UI and removes heavy widget state.
See the HTML export docs for the widget capability table and folder-export guidance.

Everything lands in `~/Downloads` and opens automatically.

| Option | Effect |
|---|---|
| `--bin N` | detector mean-bin factor for 4D-STEM (default 8) |
| `--html` | 4D-STEM: write the offline-WebGPU HTML instead of a notebook |
| `--combined` | many masters -> one 5D HTML viewer (served locally) |
| `--widget {2d,3d,4dstem}` | force a widget instead of auto-detect |
| `--out PATH` | output file or directory (default `~/Downloads`) |
| `--no-open` | write the file(s) without launching a browser or Jupyter |
| `--title`, `-v/--verbose` | page title; verbose progress |

Runs on CUDA, Apple Silicon (MPS), or CPU - the loader picks the backend. On a MacBook,
`quantem show4dstem ./masters/ --html --bin 8` loads on Metal, bins, and writes a
double-clickable HTML in seconds.

## Docs

https://bobleesj.github.io/quantem.widget/

The source tutorial notebooks live in [`docs/tutorials`](docs/tutorials). They
can be opened directly in Colab. To make a GitHub-readable preview copy, see
[`docs/github-preview.md`](docs/github-preview.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, checks, widget export
expectations, agent signoff, and release-candidate guidance.

### Commit messages

Use a short Conventional Commit-style first line:

```text
type: short imperative summary
```

Common types are `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `build`,
`ci`, and `chore`. Examples:

```text
feat: add ShowFolder thumbnail cache
fix: stabilize Show2D resize handle
docs: update HTML export protocol
```

Keep commit messages single-line unless the change genuinely needs a body. Do
not add `Co-authored-by` trailers unless requested. The same standard is
mirrored in [CLAUDE.md](CLAUDE.md) for agent handoffs.

### Widget PR checklist

Use this before opening a widget PR. It is intentionally explicit so human
contributors and coding agents can both work through it line by line. GitHub
pre-fills new PR descriptions with this checklist from
[.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md); check
every box before requesting review, either verified yourself or checked off by
a coding agent, and mark items that do not apply as "n/a" with a reason. If you
edit the checklist here, update the template too.

- [ ] The widget has a small, stable Python API with NumPy-style docs, helpful
  errors, and `(row, col)` coordinate wording where positions are shown.
- [ ] The frontend follows the local viewer patterns instead of inventing a new
  design system; compare against [Show2D](docs/tutorials/show2d.ipynb),
  [Show3D](docs/tutorials/show3d.ipynb),
  [Show3DSlices](docs/tutorials/show3dslices.ipynb),
  [Show4DSTEM](docs/tutorials/show4dstem.ipynb), and
  [ShowEDS](docs/tutorials/showeds.ipynb). Follow the
  [widget UI protocol](docs/maintainer/widget-ui-protocol.md).
- [ ] Controls are compact and content-sized: use icon/text buttons for
  commands, switches for binary options, sliders for numeric values, menus for
  option sets, and avoid stretched empty control bars.
- [ ] Compact labels stay grouped with the control they name on mobile and
  narrow layouts. A row may wrap, but `Auto`, `Smooth`, `Zoom`, `Pan`,
  `Contrast`, `fps`, `avg`, and similar labels must not separate from their
  switch, menu, slider, or button.
- [ ] Compact widget control labels do not use decorative colons. Prefer
  `Scale`, `Color`, `Auto`, `Link`, `Zoom`, `Pan`, and `ROI` in dense toolbar
  rows; keep colons for explanatory prose and tooltips.
- [ ] Command buttons use Title Case, for example `Copy`, `Export`, `Reset`,
  `Add`, `Clear`, and `Undo`. Keep scientific acronyms and file formats
  uppercase, for example `FFT`, `ROI`, `BF`, `ADF`, `HTML`, `PNG`, and `MP4`.
- [ ] The widget supports both light and dark notebook/docs themes: all labels,
  borders, controls, plots, histograms, ROI handles, status text, and export UI
  remain readable.
- [ ] The widget has no hardcoded dark-only or light-only assumptions in plots,
  canvas backgrounds, tooltips, menus, or exported HTML.
- [ ] Histogram UI matches the existing Show2D-style interaction: compact panel,
  no extra whitespace, draggable min/max handles, fast center drag, and no
  visible lag.
- [ ] Any draggable selector has live preview separate from committed widget
  state; use refs/CSS transforms or an equivalent fast path during drag. See
  [performance notes](docs/maintainer/widget-performance.md).
- [ ] Use [Show4DSTEM](docs/tutorials/show4dstem.ipynb) detector dragging and
  [ShowEDS](docs/tutorials/showeds.ipynb) energy-band dragging as the real-time
  UX benchmark: aim for 60 FPS when feasible and keep live controls at 30 FPS
  or better.
- [ ] Real-time interactions are browser-driven and verified by actually
  dragging controls in JupyterLab or exported HTML, not only by reading code or
  unit tests.
- [ ] Show4DSTEM live-scope workflows append new ready `*_master.h5`
  acquisitions into the same live viewer without rebuilding the notebook. Verify
  new masters appear in the Dataset slider, partial files are skipped until
  ready, and detector/scan interactions remain real time after append. See
  [Show4DSTEM live scope folders](docs/api/show4dstem.md#live-scope-folders).
- [ ] Performance reports separate load time, widget build time, first browser
  paint, and interaction FPS/latency. Include data shape, dtype, raw size,
  backend, and any crop/bin/downsample/quantization. Prefer `verbose=True`
  output that users and agents can copy; use `quantem.widget.profile_widget`
  for profiling notebooks when possible. See
  [performance notes](docs/maintainer/widget-performance.md).
- [ ] For interaction-sensitive changes, run
  `scripts/widget_local_signoff.sh --quick --browser` for exported HTML/UI
  paths, fix issues immediately, rebuild, refresh, and redrive before claiming
  the widget is ready. See
  [Automation](docs/maintainer/automation.md) and
  [Agent signoff](docs/maintainer/widget-agent-signoff.md).
- [ ] Expensive work avoids Python/kernel round trips during pointer movement;
  use WebGPU, typed arrays, cached indexes, workers, or throttled schedulers
  where the widget interaction requires live feedback.
- [ ] Large scientific data stays honest about precision and size: do not
  silently crop, bin, downsample, quantize, or materialize sparse zeros.
- [ ] Keep `main` lightweight: small real rendered examples are fine, but large
  tutorial arrays or HTML payloads should be generated during docs builds or
  downloaded from public data hosting only when the size justifies it.
- [ ] Keep clone and install size small for microscope PCs. Real tutorial data
  belongs in public data hosting such as Hugging Face datasets, Zenodo, or
  release assets, then gets downloaded and cached by tutorial helpers at run
  time. Do not commit large real arrays, generated HTML, or rendered docs
  branches to this repository just to make examples work.
- [ ] CI should test data-loading protocol with tiny deterministic fixtures or
  monkeypatched downloads. Full real-data downloads are reserved for docs builds,
  release signoff, or local performance checks that explicitly opt in.
- [ ] Any binning/downsampling is explicit in the API and documentation, with
  the reducer named clearly, for example mean, sum, or display-scaled `uint8`.
- [ ] The widget exposes `export_html(path=None, title=None, mode="single",
  encoding="full", downsample=None)` when it can be exported. Follow the
  [HTML export protocol](docs/api/html-export.md).
- [ ] If the widget has an in-widget **Export** button, it uses the standard
  export traits and reports filename, mode, encoding/downsample choice, and
  output size.
- [ ] Saved Jupyter widget state works: after interacting, Cmd+S, close/reopen
  in JupyterLab, and confirm the view restores without rerunning cells when the
  environment supports saved widget state.
- [ ] Standalone HTML works without a live Python kernel, and the exported page
  preserves the intended theme, viewport, interaction state, and scale/contrast
  state.
- [ ] GitHub sharing is treated separately from live HTML: GitHub notebook
  previews should use static compressed widget pictures, never heavy live widget
  state. See [GitHub preview](docs/github-preview.md).
- [ ] Documentation includes a minimal tutorial notebook under
  [docs/tutorials](docs/tutorials) and an API page under [docs/api](docs/api)
  when a public widget or loader is added.
- [ ] Tutorial notebooks avoid unnecessary `display(...)` and extra display
  imports; let the returned widget render naturally.
- [ ] The change includes focused tests for Python state/export behavior and
  frontend build coverage where possible; start with `PYTHONPATH=src pytest -q`
  and `npm run build`, or run `scripts/widget_local_signoff.sh`.
- [ ] Before committing, inspect `git status --short` and `git diff --stat`;
  do not commit generated HTML, docs builds, screenshots, local notebooks,
  private data, or machine-specific notes.

## Issues

https://github.com/bobleesj/quantem.widget/issues
