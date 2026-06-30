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

```python
import numpy as np
from quantem.widget import Show2D, Show3D, Show3DSlices, Show4DSTEM, ShowEDS

Show2D(np.random.rand(512, 512))
Show4DSTEM(np.random.rand(64, 64, 128, 128))
ShowEDS(np.random.poisson(2, (64, 64, 256)).astype("uint16"))
```

## Load data

```python
from quantem.widget import ShowEDS, Show4DSTEM, load, load_eds

data = load("scan_master.h5")   # Arina 4D-STEM .h5 -> GPU
Show4DSTEM(data)

eds = load_eds("spectrum_image.emd")  # Velox/RSCIIO EDS/EELS -> energy-last SpectrumImage
ShowEDS(eds, energy=8.04, width=0.24)
```

`quantem.widget.io` also provides `survey`, `read_image`, `bin`, `download`, and
more - see the docs.

## Command line

Point `quantem` at a file or folder and it renders the right viewer - no notebook, no
Python. Installing the package adds the `quantem` command.

```bash
quantem show ./anything/                     # auto-detect content, pick the viewer
quantem show2d scan.png                      # an image            -> Show2D
quantem show3d ./frames/                     # a folder of frames  -> Show3D scrub
quantem show4dstem ./masters/                # *_master.h5         -> live Show4DSTEM
quantem show4dstem a_master.h5 b_master.h5   # several masters     -> one 5D multi-tilt viewer
quantem show4dstem ./masters/ --html         # 4D-STEM             -> shareable offline HTML
quantem html tutorial.ipynb                  # a notebook          -> standalone offline HTML
```

| Command | Input | Output |
|---|---|---|
| `quantem show <path>` | anything | auto-detects and dispatches to one of the below |
| `quantem show2d <img / folder>` | one image, or a folder | Show2D HTML (a folder becomes a gallery) |
| `quantem show3d <folder>` | a folder of same-size frames | Show3D scrub HTML |
| `quantem show4dstem <master(s) / folder>` | one or more `*_master.h5` | live Show4DSTEM notebook (or `--html`) |
| `quantem html <notebook.ipynb>` | a notebook you wrote | runs it, bakes outputs into one offline HTML |

**Images** save a standalone HTML and open in your browser. **4D-STEM** opens a live,
kernel-backed notebook by default (full real-time interaction); `--html` instead bakes a
**self-contained offline viewer that runs entirely on WebGPU** - drag detectors, switch
BF/ABF/ADF, pan diffraction, all with no kernel. The detector is **mean-binned** (`--bin`,
default 8) so the packed stack stays small and fits a laptop's browser - you can browse
data that never fit full resolution. Several masters (a folder, or listed explicitly)
stack into one 5D viewer with a dataset slider (the multi-tilt case).

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
expectations, and release-candidate guidance.

### Widget PR checklist

Use this before opening a widget PR. It is intentionally explicit so human
contributors and coding agents can both copy it into a PR description and work
through it line by line.

- [ ] The widget has a small, stable Python API with NumPy-style docs, helpful
  errors, and `(row, col)` coordinate wording where positions are shown.
- [ ] The frontend follows the local viewer patterns instead of inventing a new
  design system; compare against [Show2D](docs/tutorials/show2d.ipynb),
  [Show3D](docs/tutorials/show3d.ipynb),
  [Show3DSlices](docs/tutorials/show3dslices.ipynb),
  [Show4DSTEM](docs/tutorials/show4dstem.ipynb), and
  [ShowEDS](docs/tutorials/showeds.ipynb).
- [ ] Controls are compact and content-sized: use icon/text buttons for
  commands, switches for binary options, sliders for numeric values, menus for
  option sets, and avoid stretched empty control bars.
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
- [ ] Expensive work avoids Python/kernel round trips during pointer movement;
  use WebGPU, typed arrays, cached indexes, workers, or throttled schedulers
  where the widget interaction requires live feedback.
- [ ] Large scientific data stays honest about precision and size: do not
  silently crop, bin, downsample, quantize, or materialize sparse zeros.
- [ ] Real tutorial data is hosted through public `quantem.data` pages instead
  of committed to this repo; coordinate with Bob Lee if a new dataset needs
  online hosting.
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
  and `npm run build`.
- [ ] Before committing, inspect `git status --short` and `git diff --stat`;
  do not commit generated HTML, docs builds, screenshots, local notebooks,
  private data, or machine-specific notes.

## Issues

https://github.com/bobleesj/quantem.widget/issues
