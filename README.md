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

**[Start with the documentation](https://electronmicroscopy.github.io/quantem.widget/)**
to load ARINA 4D-STEM data in Jupyter, open `Show4DSTEM`, and explore the
interactive widgets.

> `quantem.widget` is currently a prototype on
> [TestPyPI](https://test.pypi.org/project/quantem-widget/). It installs
> [`quantem`](https://github.com/electronmicroscopy/quantem) core and its readers
> return quantem datasets, except on an Intel Mac and Windows on ARM, which
> cannot install quantem core (see [Load data](#load-data)).

## Install

```bash
pip install "quantem.widget[cuda]"   # NVIDIA GPU (Linux tested; Windows: CPU tested, CUDA untested)
pip install "quantem.widget[mps]"    # Apple silicon Mac (Metal)
pip install "quantem.widget[cpu]"    # no GPU, or an Intel Mac (the same as plain pip install quantem.widget)
```

While the package is on TestPyPI, add `-i https://test.pypi.org/simple/
--extra-index-url https://pypi.org/simple/` to the line.

| Machine | Install | Widget compute (`device="auto"`) | `read_4dstem(path)`, `Show4DSTEM(path)` | Tested |
|---|---|---|---|---|
| Linux, NVIDIA GPU | `quantem.widget[cuda]` | CUDA | `Dataset4dstemGPU`, encoded on the GPU by quantem.gpu | yes |
| Windows, NVIDIA GPU | `quantem.widget[cuda]` | CUDA | `Dataset4dstemGPU`, encoded on the GPU by quantem.gpu | CPU tested, CUDA untested |
| Windows on ARM | torch from the PyTorch CPU index, then `quantem.widget[cpu]` (no quantem core) | CPU | the stand-in, dense read on the CPU; compressed masters need x64 Python | CPU tested |
| Mac, Apple silicon | `quantem.widget[mps]` | Apple GPU (MPS) | `Dataset4dstemGPU`, encoded on the GPU by quantem.gpu (Metal) | yes |
| Mac, Intel | `quantem.widget[cpu]` (torch 2.2, NumPy 1.26), no quantem.gpu, no quantem core | AMD GPU through torch MPS, else CPU | the stand-in, dense read on the CPU | yes |
| No GPU | `quantem.widget[cpu]` | CPU | quantem core `Dataset4dstem`, dense read on the CPU | yes |

On Windows, torch needs the
[Microsoft Visual C++ Redistributable](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist) for the
architecture of your Python (x64 or ARM64); installing it needs administrator
rights. PyPI has no torch for ARM64 Python on Windows on ARM: install it first
with `pip install torch --index-url https://download.pytorch.org/whl/cpu`
(Python 3.11 to 3.13). hdf5plugin has no ARM64 wheel, so `.npy` files and HDF5
files without its filters (uncompressed or gzip) open there, and compressed
Arina masters need x64 Python with `pip install hdf5plugin`. CUDA on Windows is
untested.

A wrong pick installs nothing harmful: `[cuda]` installs nothing extra on a Mac,
`[mps]` nothing outside Apple silicon. Every widget takes `device="auto"`
(CUDA, then Apple MPS, then CPU) and prints the device it chose once. When a GPU
is present but its quantem.gpu path is not installed, the widget prints one line
with the `pip` command for that GPU. See the
[installation guide](https://electronmicroscopy.github.io/quantem.widget/install.html)
for Colab instructions and verification.

## Load data

The readers return the dataset type of the machine, so the same code runs on
every machine:

```python
from quantem.widget import Show2D, Show4DSTEM, io

ds = io.read_image("overview.emd")          # quantem core Dataset2d
arr = ds.array                               # the pixels; ds.sampling and ds.units carry the calibration
Show2D(ds)

data = io.read_4dstem("scan_master.h5")      # Dataset4dstemGPU on a GPU, quantem core Dataset4dstem elsewhere
Show4DSTEM(data)
```

| Data | quantem.gpu and a GPU | No GPU route | Intel Mac, Windows on ARM |
|---|---|---|---|
| 4D-STEM file (`read_4dstem`, `datasets.show4dstem_gold`) | `quantem.gpu.io.Dataset4dstemGPU` | quantem core `Dataset4dstem` | stand-in |
| Image (`read_image`, `read_images`, `datasets.show2d_gold`) | quantem core `Dataset2d` | quantem core `Dataset2d` | stand-in |
| Image stack (`read_image_stack`, `datasets.show3d_gold`) | quantem core `Dataset3d` | quantem core `Dataset3d` | stand-in |

On a GPU, a 4D-STEM file is a `Dataset4dstemGPU`, not a torch tensor:

- counts are unsigned (uint16, uint32), and torch cannot add, multiply or
  take the max of those;
- the file stays compressed in GPU memory (the 512 x 512 x 192 x 192 gold
  scan: 5.6 GiB, against 18 GiB as dense uint16);
- quantem.gpu's CUDA and Metal kernels decode and sum it exactly in one pass.

Indexing and `read(scan_region=...)` still return torch tensors.

The stand-in, used where quantem core cannot install, has the same `array`,
`name`, `sampling`, `units`, `metadata`, `shape`, `ndim` and `dtype`. A color
PNG, JPEG or TIFF reads as an `RgbImage` with `array`, `name`, `sampling` and
`units`, and a folder of color frames as an `(N, H, W, 3)` array; quantem core
has no color-image dataset. Every widget also takes NumPy arrays and torch
tensors on any device.

## Try the interactive demos

Follow the [crystal-to-volume demo route](docs/tutorials/demo.md): inspect a
crystal with **ShowCIF**, then explore depth using **Show3D** and **Show3DSlices**.
The notebooks include runnable models, physical units, and controls to try.
The ShowCIF potential preview needs the `crystal` extra (abTEM); new APIs
require a source build containing them. The guide includes setup and browser requirements.

## Widgets

| Widget | Use it for | Learn more |
|---|---|---|
| `Plot2D` | Scalar maps with physical axes, color scales, and calibrated hover | [tutorial](docs/tutorials/plot2d.ipynb) · [API](docs/api/plot2d.md) |
| `Show1D` | Scientific traces, reconstruction metrics, and live monitors | [API](https://electronmicroscopy.github.io/quantem.widget/api/show1d.html) |
| `Show2D` | Images, contrast, FFTs, ROIs, profiles, and scale bars | [tutorial](https://electronmicroscopy.github.io/quantem.widget/tutorials/show2d.html) · [API](https://electronmicroscopy.github.io/quantem.widget/api/show2d.html) |
| `Show3D` | Scrub and play through image or volume stacks | [tutorial](https://electronmicroscopy.github.io/quantem.widget/tutorials/show3d.html) · [API](https://electronmicroscopy.github.io/quantem.widget/api/show3d.html) |
| `Show3DSlices` | Inspect linked top and oblique cuts through a 3D volume | [tutorial](https://electronmicroscopy.github.io/quantem.widget/tutorials/show3dslices.html) · [API](https://electronmicroscopy.github.io/quantem.widget/api/show3dslices.html) |
| `Show4DSTEM` | Live virtual detectors, multi-dataset review, and WebGPU HTML export | [tutorial](https://electronmicroscopy.github.io/quantem.widget/tutorials/show4dstem.html) · [export guide](https://electronmicroscopy.github.io/quantem.widget/tutorials/show4dstem_export.html) · [API](https://electronmicroscopy.github.io/quantem.widget/api/show4dstem.html) |
| `ShowPtycho` | Interactive SSB phase and aberration review | [API](https://electronmicroscopy.github.io/quantem.widget/api/showptycho.html) |
| `ShowDiffraction` | Measure diffraction spots, rings, spacing, and angles | [tutorial](https://electronmicroscopy.github.io/quantem.widget/tutorials/showdiffraction.html) · [API](https://electronmicroscopy.github.io/quantem.widget/api/showdiffraction.html) |
| `ChooseLattice` | Select an origin and lattice vectors | [API](https://electronmicroscopy.github.io/quantem.widget/api/choose-lattice.html) |
| `ShowCIF` | Crystal repeats, columns, tilt, and projected potential/phase previews | [tutorial](docs/tutorials/showcif.ipynb) · [API](docs/api/showcif.md) |

## Documentation

For a scalar map, import `Plot2D` from the same package:

```python
from quantem.widget import Plot2D

# values.shape == (len(angle), len(radius)); coordinates are bin centers.
plot = Plot2D(values, x=radius, y=angle,
              x_label="Distance (Å)", y_label="Angle (°)")
```

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

Would you like to create a widget? Linked [here](https://electronmicroscopy.github.io/quantem.widget/developer/widget-creation.html)
is a walkthrough that discusses what one should contain and the eleven steps needed to create a widget.

Each widget is meant to be self-contained in its own Python module and `js/`
folder. This way, every widget can reuse the same kernels and shared browser-GPU work like FFTs
can all belong in
[quantem.gpu](https://github.com/bobleesj/quantem.gpu).

Questions and bug reports belong in the
[issue tracker](https://github.com/electronmicroscopy/quantem.widget/issues).

[Experimental ANS sources](docs/developer/experimental-ans.md) describes the
encoded acquisitions that Show4DSTEM reads from `quantem.gpu.io.load` and the
QEM browser viewer, with documented format and performance limits.
