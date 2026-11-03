# Installation

`quantem.widget` is a pre-release on **TestPyPI**. The install command therefore
names two indexes: TestPyPI for quantem.widget and quantem.gpu, PyPI for
everything they depend on. Copy the block for your machine. Python 3.11 or newer.

**NVIDIA GPU**

```bash
pip install -i https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    "quantem.widget[cuda]"
```

**Apple silicon Mac**

```bash
pip install -i https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    "quantem.widget[mps]"
```

**No GPU, or an Intel Mac**

```bash
pip install -i https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    "quantem.widget[cpu]"
```

Then check what was installed and which device the widgets will use:

```python
import quantem.widget as qw

qw.profile()
```

- A wrong pick installs nothing harmful: `[cuda]` adds nothing extra on a Mac
  and `[mps]` nothing outside Apple silicon, and the widget's one-line notice
  names the right extra.
- On Windows, read [Windows](#windows) first.
- ShowPDF also needs quantem core with the `PairDistributionFunction` of
  quantem pull request 248, see [ShowPDF](api/showpdf).
- What each machine installs, which dataset type the readers return and what
  was tested is in [Platform support](developer/platforms.md).
- When quantem.widget reaches PyPI the command shortens to
  `pip install "quantem.widget[cuda]"` (or `[mps]`, `[cpu]`).

Tutorial data downloads (`quantem.widget.datasets`, `quantem.widget.io.download`)
need only `huggingface_hub`, which installs with the package. To upload new
shared datasets, add the `[hub]` extra, which pulls in `quantem.data`:

```bash
pip install -i https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    "quantem.widget[hub]"
```

Tutorial helpers reuse pinned shared acquisitions on
[bobleesj/quantem-data](https://huggingface.co/datasets/bobleesj/quantem-data).
[Tutorial datasets](api/datasets.md) records the canonical inputs and immutable
revision. Older folder layouts require their matching software version. The widget-only download path remains available;
delegation to `quantem.data` requires its matching revision-aware API.
No new shared-package or Colab release is implied by these source changes.

## Dependency-resolution errors

If installation reports that `anywidget>=0.11.0` was not found or that all
widget versions are unusable, check that the command includes **both indexes**.
`anywidget` is available on PyPI, not TestPyPI. Using only
`-i https://test.pypi.org/simple/` hides that dependency source.

For NVIDIA CUDA, run this complete line in your Python 3.11+ environment:

```bash
python -m pip install --upgrade -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "quantem.widget[cuda]"
```

For `uv`, use `uv pip install` with the same arguments. Keep PyPI as
`--extra-index-url`: uv gives that index priority for ordinary dependencies,
then finds the QuantEM pre-releases on TestPyPI. There is no need to disable
dependency checks with `--no-deps` or change uv's index security strategy.

If the complete command still fails, include the full resolver error, the exact
command, `python --version`, and your operating system in a bug report. The
final "requirements are unsatisfiable" line alone omits the underlying cause.

## Windows

- torch needs the [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist)
  for the architecture of the Python you run (x64 or ARM64). Installing it needs
  administrator rights. Without it, `import torch` fails with `OSError: [WinError 126]`
  while loading a DLL from `torch\lib`.
- The CPU path is tested on Windows 11 on ARM (x64 and ARM64 Python); CUDA on
  Windows is untested.

On Windows on ARM, PyPI has no torch for ARM64 Python. The PyTorch CPU index
has it for Python 3.11 to 3.13 (3.12 tested). Install it from there first, then
the widget:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -i https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    "quantem.widget[cpu]"
```

- hdf5plugin has no ARM64 wheel, so pip skips it on Windows on ARM. `.npy`
  files and HDF5 files without hdf5plugin's filters (uncompressed or gzip)
  open; a compressed (bitshuffle-LZ4) Arina master raises an error that names
  hdf5plugin. quantem core requires hdf5plugin, so pip skips it too and the
  readers return the widget's stand-in for its datasets.
- For compressed masters, use x64 Python, which Windows on ARM runs under
  emulation and which installs torch from PyPI, and add hdf5plugin with
  `pip install hdf5plugin`.
  x64 Python 3.12 and newer reports the ARM64 processor to pip, so pip skips
  it there too unless it is named.

## Crystal and volume demos

Follow the [interactive demo route](tutorials/demo.md) for ShowCIF, Show3D, and
Show3DSlices. The ShowCIF potential preview takes its Lobato scattering factors
from quantem, a base dependency:

```bash
python -m pip install -i https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ quantem.widget
```

Check that the installed build includes the API you want to demonstrate; the
new ShowCIF notebook requires a revision containing ShowCIF. For an unpublished
source revision, use the [developer setup](https://github.com/electronmicroscopy/quantem.widget/blob/main/CONTRIBUTING.md)
and install `'.'`, then build the JavaScript with `npm run build`.
Use the same Python environment for installation and the notebook kernel.

## Google Colab

Tutorials with a Colab badge can open directly in Colab. The new crystal demo
can also be run from a local source checkout as described above. Colab uses the same files that build these docs, so there is no
separate Colab copy to maintain.

Each Colab-ready tutorial has one collapsed **Install QuantEM** cell. Its two
plain steps download and run the shared `scripts/install_colab.py` installer, which
takes the newest `quantem.widget` wheel on TestPyPI and the newest `quantem.gpu`
wheel inside the range that widget release pins. Normal dependencies still come from PyPI, and Colab's loaded NumPy
and Numba versions are preserved. After installation, the cell calls
`quantem.widget.profile()` automatically so the notebook records the installed
QuantEM versions and active compute environment. Do not use TestPyPI as
Colab's package index or upgrade NumPy inside the running kernel: either can
leave the process with incompatible compiled extension modules.

Show4DSTEM also selects its kernel-backed compute path in Colab because Colab's
output iframe does not expose WebGPU. When that iframe mounts, the widget asks
the kernel for its first diffraction and virtual-image buffers again. The
tutorial can therefore use the normal, final `viewer` expression without
special display calls, sleeps, or state-resend code. Other notebook and
exported-HTML contexts keep the browser-compute path.

Common entry points:

| Tutorial | Colab | Source notebook |
|---|---|---|
| Example Data | [Open in Colab](https://colab.research.google.com/github/electronmicroscopy/quantem.widget/blob/main/docs/tutorials/download_data.ipynb) | [GitHub](https://github.com/electronmicroscopy/quantem.widget/blob/main/docs/tutorials/download_data.ipynb) |
| Show1D | [Open in Colab](https://colab.research.google.com/github/electronmicroscopy/quantem.widget/blob/main/docs/tutorials/show1d.ipynb) | [GitHub](https://github.com/electronmicroscopy/quantem.widget/blob/main/docs/tutorials/show1d.ipynb) |
| Show2D | [Open in Colab](https://colab.research.google.com/github/electronmicroscopy/quantem.widget/blob/main/docs/tutorials/show2d.ipynb) | [GitHub](https://github.com/electronmicroscopy/quantem.widget/blob/main/docs/tutorials/show2d.ipynb) |
| Show3D | [Open in Colab](https://colab.research.google.com/github/electronmicroscopy/quantem.widget/blob/main/docs/tutorials/show3d.ipynb) | [GitHub](https://github.com/electronmicroscopy/quantem.widget/blob/main/docs/tutorials/show3d.ipynb) |
| Show3DSlices | [Open in Colab](https://colab.research.google.com/github/electronmicroscopy/quantem.widget/blob/main/docs/tutorials/show3dslices.ipynb) | [GitHub](https://github.com/electronmicroscopy/quantem.widget/blob/main/docs/tutorials/show3dslices.ipynb) |
| Show4DSTEM | [Open in Colab](https://colab.research.google.com/github/electronmicroscopy/quantem.widget/blob/main/docs/tutorials/show4dstem.ipynb) | [GitHub](https://github.com/electronmicroscopy/quantem.widget/blob/main/docs/tutorials/show4dstem.ipynb) |
| ShowDiffraction | [Open in Colab](https://colab.research.google.com/github/electronmicroscopy/quantem.widget/blob/main/docs/tutorials/showdiffraction.ipynb) | [GitHub](https://github.com/electronmicroscopy/quantem.widget/blob/main/docs/tutorials/showdiffraction.ipynb) |
| Choose Lattice | [Open in Colab](https://colab.research.google.com/github/electronmicroscopy/quantem.widget/blob/main/docs/tutorials/choose_lattice.ipynb) | [GitHub](https://github.com/electronmicroscopy/quantem.widget/blob/main/docs/tutorials/choose_lattice.ipynb) |
| IO/GPU | [Open in Colab](https://colab.research.google.com/github/electronmicroscopy/quantem.widget/blob/main/docs/tutorials/io_gpu.ipynb) | [GitHub](https://github.com/electronmicroscopy/quantem.widget/blob/main/docs/tutorials/io_gpu.ipynb) |
| HTML and file export | [Open in Colab](https://colab.research.google.com/github/electronmicroscopy/quantem.widget/blob/main/docs/tutorials/widget_export.ipynb) | [GitHub](https://github.com/electronmicroscopy/quantem.widget/blob/main/docs/tutorials/widget_export.ipynb) |

## Devices

Every widget that computes takes `device=`: `"auto"` (default; CUDA, then
Apple MPS, then CPU, printed once), `"cuda"`, `"cuda:N"`, `"mps"` or `"cpu"`,
with the same meaning as in quantem.gpu. A torch tensor stays on its device
unless `device=` names another one, and that move is printed once.

- **NVIDIA CUDA** (`[cuda]`): the widgets compute in torch on the GPU, and
  `Show4DSTEM` keeps acquisitions encoded on the GPU through quantem.gpu, whose
  `[cuda]` extra brings the CuPy wheel.
- **Apple silicon** (`[mps]`): quantem.gpu keeps 4D-STEM acquisitions encoded on
  the Apple GPU and `Show4DSTEM` opens them at full detector resolution.
- **Intel Mac** (`[cpu]`): torch MPS on the AMD GPU for the widgets' own
  compute; files are read on the CPU. quantem.gpu is not available.
- **CPU**: everything runs, more slowly.
