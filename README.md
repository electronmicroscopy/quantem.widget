# quantem.widget

Interactive Jupyter widgets for electron microscopy visualization. Works with NumPy, CuPy, and PyTorch arrays.

## Installation

```bash
pip install quantem-widget
```

## Quick Start

### Show1D - 1D Data Viewer

```python
import numpy as np
from quantem.widget import Show1D

# Single spectrum
spectrum = np.random.rand(512)
Show1D(spectrum, x_label="Energy Loss", x_unit="eV", y_label="Counts")

# Multiple traces with calibrated X axis
energy = np.linspace(0, 800, 512).astype(np.float32)
Show1D(
    [spec1, spec2, spec3],
    x=energy,
    labels=["Region A", "Region B", "Region C"],
    title="EELS Comparison",
)
```

### Show2D - Static Image Viewer

```python
import numpy as np
from quantem.widget import Show2D

# Single image
image = np.random.rand(256, 256)
Show2D(image)

# Multiple images (gallery mode)
images = [img1, img2, img3]
Show2D(images, labels=["A", "B", "C"])
```

### Show3D - Stack Viewer with Playback

```python
import numpy as np
from quantem.widget import Show3D

# 3D stack (z-stack, time series, defocus series)
stack = np.random.rand(100, 256, 256)
Show3D(stack, title="My Stack", fps=5)
```

### Show3DVolume - Orthogonal Slice Viewer

```python
import numpy as np
from quantem.widget import Show3DVolume

volume = np.random.rand(64, 64, 64).astype(np.float32)
Show3DVolume(volume, title="My Volume", cmap="viridis")
```

### Show4DSTEM - 4D-STEM Viewer

```python
import numpy as np
from quantem.widget import Show4DSTEM

data = np.random.rand(64, 64, 128, 128)
w = Show4DSTEM(data, pixel_size=2.39, k_pixel_size=0.46)

# Programmatic export from current interactive state
w.save_image("dp.png", view="diffraction", position=(10, 12))
w.save_image("virtual.pdf", view="virtual", frame_idx=3)
w.save_image("all_panels.png", view="all")
```

### Bin - Interactive Binning + BF/ADF QC

```python
import numpy as np
from quantem.widget import Bin

data = np.random.rand(64, 64, 128, 128).astype(np.float32)
w = Bin(data, pixel_size=2.39, k_pixel_size=0.46, device="cpu")
w  # adjust scan/detector binning and inspect BF/ADF quality panels

# export current binning preset
w.save("bin_preset.json")

# get binned array for downstream workflows
binned = w.get_binned_data()
```

### Mark2D - Interactive Image Annotation

```python
import numpy as np
from quantem.widget import Mark2D

image = np.random.rand(256, 256)
w = Mark2D(image, max_points=5)
w

# After clicking, retrieve selected points
w.selected_points  # [{'row': 83, 'col': 120}, ...]
```

## Batch Binning Over Folders (Preset-Driven)

Use one-file-at-a-time batch processing for large time-series pipelines:

```bash
python -m quantem.widget.bin_batch \
  --input-dir /path/to/raw \
  --output-dir /path/to/binned \
  --preset bin_preset.json \
  --pattern '*.npy' \
  --recursive \
  --device cpu
```

Equivalent wrapper:

```bash
./scripts/bin_batch.py --input-dir /path/to/raw --output-dir /path/to/binned --preset bin_preset.json
```

Notes:
- Processing is sequential (one file at a time) to keep memory bounded.
- 5D `.npy` time series are streamed frame-by-frame when `time_axis=0` in the preset.
- Outputs include a JSONL manifest at `output_dir/bin_batch_manifest.jsonl`.
- Binning compute is torch-only; set `device` to `cpu`, `cuda`, or `mps`.

## Array Compatibility

All widgets accept NumPy arrays, PyTorch tensors, CuPy arrays, and quantem Dataset objects via duck typing. No manual conversion needed.

| Widget | NumPy | PyTorch | CuPy | quantem Dataset |
|--------|-------|---------|------|-----------------|
| Show1D | yes | yes | yes | duck typing |
| Show2D | yes | yes | yes | `Dataset2d` |
| Show3D | yes | yes | yes | `Dataset3d` |
| Show3DVolume | yes | yes | yes | `Dataset3d` |
| Show4D | yes | yes | yes | `Dataset4dstem` |
| Show4DSTEM | yes | yes | yes | `Dataset4dstem` |
| ShowComplex2D | yes | yes | yes | duck typing |
| Mark2D | yes | yes | yes | `Dataset2d` |
| Edit2D | yes | yes | yes | `Dataset2d` |
| Align2D | yes | yes | yes | `Dataset2d` |
| Bin | yes | yes | yes | `Dataset4dstem` |

When a quantem Dataset is passed, metadata (title, pixel size, units) is extracted automatically. Explicit parameters always override auto-extracted values.

```python
import torch
Show2D(torch.randn(256, 256))  # PyTorch tensor

from quantem.core.datastructures import Dataset3d
dataset = Dataset3d.from_array(stack, name="Focal Series", sampling=(1.0, 0.25, 0.25), units=("nm", "Å", "Å"))
Show3D(dataset)  # title and pixel_size extracted from dataset
```

## Documentation with Interactive Widgets

### Building docs locally

```bash
pip install -e ".[docs]"

# One-shot build
sphinx-build docs docs/_build/html
open docs/_build/html/index.html

# Live reload (rebuilds on file save, auto-refreshes browser)
sphinx-autobuild docs docs/_build/html --open-browser
```

The Sphinx documentation renders anywidget-based widgets interactively in the browser — users can zoom, pan, change colormaps, toggle FFT, etc. directly on the docs page without a running kernel.

### How it works

1. Notebooks are executed locally in JupyterLab, which saves **widget state** (JS bundle + binary image data) into the notebook metadata
2. nbsphinx renders the pre-saved widget state as interactive HTML using `@jupyter-widgets/html-manager`
3. GitHub Actions deploys to GitHub Pages on every push to `main`

### Adding or updating docs notebooks

```bash
# 1. Run the notebook in JupyterLab (widget state is saved on File > Save)
jupyter lab docs/examples/show2d/show2d_simple.ipynb

# 2. Verify widget state is embedded
python -c "import json; nb=json.load(open('docs/examples/show2d/show2d_simple.ipynb')); print('Widget state:', bool(nb.get('metadata',{}).get('widgets',{})))"

# 3. Commit the notebook (with widget state)
git add docs/examples/show2d/show2d_simple.ipynb
```

Docs example notebooks in `docs/examples/` can be either real files with saved widget state, or symlinks to `notebooks/` (which must also have widget state saved).

### Limitations of static docs

The docs pages embed widget state without a Python kernel, so only **JS-side features** work:
- **Works**: zoom, pan, colormap, log scale, FFT, auto-contrast, histogram, theme toggle
- **Doesn't work**: frame navigation (Show3D/Show4D), export (GIF/ZIP), `set_image()`, trait observers

For full interactivity (all features including Python backend), use **Google Colab**. Each notebook has a Colab badge at the top. You'll need to install from TestPyPI first:

```python
!pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ quantem-widget
```

## CI/CD

Two GitHub Actions workflows automate publishing and documentation:

### Docs deployment (`.github/workflows/docs.yml`)

Deploys Sphinx documentation to GitHub Pages on every push to `main`.

- **Trigger**: push to `main` branch (or manual `workflow_dispatch`)
- **What it does**: installs Node.js + Python deps, runs `npm install && pip install .` (hatch-jupyter-builder builds JS automatically), then `sphinx-build`
- **Prerequisite**: enable GitHub Pages in repo Settings → Pages → Source: **GitHub Actions**
- **URL**: https://bobleesj.github.io/quantem.widget/
- Notebooks are rendered with `nbsphinx_execute = "never"` — pre-saved outputs (including widget state) are used as-is, no execution on CI

### Versioning

This project follows [PEP 440](https://packaging.python.org/en/latest/discussions/versioning/) semantic versioning (`major.minor.patch`):

- **Major** (`1.0.0`) — breaking API changes
- **Minor** (`0.4.0`) — new features, widgets, or protocols
- **Patch** (`0.4.1`) — bug fixes, docs, notebook fixes

Pre-release suffixes: `0.5.0a1` (alpha) → `0.5.0b1` (beta) → `0.5.0rc1` (release candidate) → `0.5.0` (final).

Version is set in `pyproject.toml` — single source of truth.

### TestPyPI publishing (`.github/workflows/publish.yml`)

Publishes to TestPyPI when a version tag is pushed.

```bash
# 1. Bump version in pyproject.toml
# 2. Commit, tag, and push
git tag v0.4.0
git push origin main && git push origin v0.4.0
```

GitHub Actions compiles React/TypeScript, builds the Python wheel, and uploads to TestPyPI. Note: TestPyPI does not allow re-uploading the same version — always bump the version before tagging.

### Verify TestPyPI Release

After CI finishes, verify the published package in a clean environment:

```bash
./scripts/verify_testpypi.sh 0.4.0
```

This creates a fresh conda env, installs from TestPyPI, verifies all widget imports and JS bundles, then opens JupyterLab with a test notebook for visual inspection. When done, press Ctrl+C and clean up:

```bash
conda env remove -n test-widget-env -y
```

### TestPyPI Trusted Publisher Setup

1. Go to https://test.pypi.org/manage/account/publishing/
2. Add a new "pending publisher" with:
   - **PyPI project name**: `quantem-widget`
   - **Owner**: `bobleesj`
   - **Repository**: `quantem.widget`
   - **Workflow name**: `publish.yml`
   - **Environment**: leave blank

## Development

```bash
git clone https://github.com/bobleesj/quantem.widget.git
cd quantem.widget
npm install
npm run build
pip install -e .
```

## License

MIT
