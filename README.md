# quantem.widget

[![TestPyPI](https://img.shields.io/pypi/v/quantem-widget?pypiBaseUrl=https://test.pypi.org&label=TestPyPI)](https://test.pypi.org/project/quantem-widget/)

Interactive, GPU-accelerated visualization widgets for 4D-STEM and electron
microscopy. Use them in Jupyter notebooks, as local HTML files, or from the
command line. NumPy, PyTorch, CuPy, CUDA, Apple Silicon, and browser WebGPU
workflows are supported.

![Show4DSTEM WebGPU demo with a diffraction pattern and live virtual detector image](docs/_static/show4dstem-serin-gold.gif)

**Demo: Show4DSTEM HTML with WebGPU.** Explore live diffraction-pattern and
virtual-detector views locally in a browser on a personal laptop or supported
phone, without a Python kernel or remote compute server. Thanks to Serin Lee for
sharing this liquid-cell Au nanoparticle 4D-STEM dataset. Check Serin's 4D-STEM
and 5D-STEM segmentation and clustering work
([paper](https://academic.oup.com/mam/article-abstract/32/3/ozag044/8701498))
and the source data ([Zenodo](https://zenodo.org/records/18167694)).

> `quantem.widget` is currently a prototype on
> [TestPyPI](https://test.pypi.org/project/quantem-widget/) and is built on the
> [`quantem`](https://github.com/electronmicroscopy/quantem) core.

## Install

```bash
pip install -i https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ quantem.widget
```

See the [installation guide](https://electronmicroscopy.github.io/quantem.widget/install.html)
for backend setup, Colab instructions, and verification.

## Quick start

Open an image or microscopy dataset without writing a notebook:

```bash
quantem show image.tif
quantem show3d ./frames/
quantem show4dstem ./masters/
```

Or construct widgets directly in Python:

```python
import numpy as np
from quantem.widget import Show2D, Show4DSTEM

Show2D(np.random.random((512, 512)))
Show4DSTEM(np.random.random((64, 64, 128, 128)))
```

For real 4D-STEM data, load a master file onto the available GPU:

```python
from quantem.gpu.io import load
from quantem.widget import Show4DSTEM

Show4DSTEM(load("scan_master.h5"))
```

The [command-line guide](https://electronmicroscopy.github.io/quantem.widget/cli.html)
and [tutorials](https://electronmicroscopy.github.io/quantem.widget/tutorials/download_data.html)
cover data loading, HTML export, public example datasets, and complete workflows.

## Widgets

| Widget | Use it for | Learn more |
|---|---|---|
| `Show1D` | Scientific traces, reconstruction metrics, and live monitors | [API](https://electronmicroscopy.github.io/quantem.widget/api/show1d.html) |
| `Show2D` | Images, contrast, FFTs, ROIs, profiles, and scale bars | [tutorial](https://electronmicroscopy.github.io/quantem.widget/tutorials/show2d.html) · [API](https://electronmicroscopy.github.io/quantem.widget/api/show2d.html) |
| `Show3D` | Scrub and play through image or volume stacks | [tutorial](https://electronmicroscopy.github.io/quantem.widget/tutorials/show3d.html) · [API](https://electronmicroscopy.github.io/quantem.widget/api/show3d.html) |
| `Show3DSlices` | Inspect orthogonal slices through a 3D volume | [tutorial](https://electronmicroscopy.github.io/quantem.widget/tutorials/show3dslices.html) · [API](https://electronmicroscopy.github.io/quantem.widget/api/show3dslices.html) |
| `Show4DSTEM` | Live virtual detectors, multi-dataset review, and WebGPU HTML export | [tutorial](https://electronmicroscopy.github.io/quantem.widget/tutorials/show4dstem.html) · [export guide](https://electronmicroscopy.github.io/quantem.widget/tutorials/show4dstem_export.html) · [API](https://electronmicroscopy.github.io/quantem.widget/api/show4dstem.html) |
| `ShowPtycho` | Interactive SSB phase and aberration review | [API](https://electronmicroscopy.github.io/quantem.widget/api/showptycho.html) |
| `ShowDiffraction` | Measure diffraction spots, rings, spacing, and angles | [tutorial](https://electronmicroscopy.github.io/quantem.widget/tutorials/showdiffraction.html) · [API](https://electronmicroscopy.github.io/quantem.widget/api/showdiffraction.html) |
| `ChooseLattice` | Select an origin and lattice vectors | [API](https://electronmicroscopy.github.io/quantem.widget/api/choose-lattice.html) |
| `ShowEDS` | Explore linked EDS/EELS maps and spectra | — |
| `ShowFolder` | Browse, group, and select microscopy session files | [tutorial](https://electronmicroscopy.github.io/quantem.widget/tutorials/showfolder.html) · [API](https://electronmicroscopy.github.io/quantem.widget/api/showfolder.html) |

## Documentation

Visit the **[quantem.widget documentation](https://electronmicroscopy.github.io/quantem.widget/)**
for installation, tutorials, API references, command-line workflows, data I/O,
HTML sharing, and WebGPU export guidance.

`quantem.widget` bridges scientific data and computational algorithms. We
recommend agent-assisted development so researchers can spend less time writing
interface code and more time discovering scientific insight. Start with the copyable
**[agent-assisted development prompts](https://electronmicroscopy.github.io/quantem.widget/agent-prompts.html)**
for opening ARINA data, designing a widget, or preparing a pull request.

## Citing quantem.widget

If the quantEM interactive framework—including `quantem.widget`, GPU-accelerated
I/O, analysis, or reconstruction workflows on MPS or CUDA—contributed to your
research, please consider citing Lee et al., *Interactive Framework for
Real-Time 4DSTEM Analysis and Reconstruction*, *Microscopy and Microanalysis*
32 (Supplement 1), ozag053.941 (2026),
https://doi.org/10.1093/mam/ozag053.941.

## Contributing

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) for setup,
tests, documentation standards, and the pull-request workflow. That workflow
follows the reproducible scientific-software procedures described by
[scikit-package](https://scikit-package.github.io/scikit-package/).

Questions and bug reports belong in the
[issue tracker](https://github.com/electronmicroscopy/quantem.widget/issues).
