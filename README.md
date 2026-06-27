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
| `ShowEDS` | EDS/EELS spectrum image | linked element map, spectrum, energy band, and real-space ROI |

```python
import numpy as np
from quantem.widget import Show2D, Show3D, Show3DSlices, Show4DSTEM, ShowEDS

Show2D(np.random.rand(512, 512))
Show4DSTEM(np.random.rand(64, 64, 128, 128))
ShowEDS(np.random.poisson(2, (64, 64, 256)).astype("uint16"))
```

## Load data

```python
from quantem.widget import load

data = load("scan_master.h5")   # Arina 4D-STEM .h5 -> GPU
Show4DSTEM(data)
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

## Issues

https://github.com/bobleesj/quantem.widget/issues
