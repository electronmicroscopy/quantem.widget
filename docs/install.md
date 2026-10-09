# Installation

Pick the line for your machine:

```bash
pip install "quantem.widget[cuda]"   # NVIDIA GPU (Linux tested; Windows: CPU tested, CUDA untested)
pip install "quantem.widget[mps]"    # Apple silicon Mac (Metal)
pip install "quantem.widget[cpu]"    # no GPU, or an Intel Mac (the same as plain pip install quantem.widget)
```

`quantem.widget` is currently published on **TestPyPI** (pre-release). Until it
reaches PyPI, add TestPyPI as the index and PyPI as the extra index so its
dependencies resolve normally:

```bash
pip install -i https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    "quantem.widget[cuda]"
```

Requires Python 3.11 or newer.

| Machine | Install | Widget compute (`device="auto"`) | `read_4dstem(path)`, `Show4DSTEM(path)` | Tested |
|---|---|---|---|---|
| Linux, NVIDIA GPU | `quantem.widget[cuda]` | CUDA | `Dataset4dstemGPU`, encoded on the GPU by quantem.gpu | yes |
| Windows, NVIDIA GPU | `quantem.widget[cuda]` | CUDA | `Dataset4dstemGPU`, encoded on the GPU by quantem.gpu | CPU tested, CUDA untested |
| Windows on ARM | torch from the PyTorch CPU index, then `quantem.widget[cpu]` (no quantem core) | CPU | the stand-in, dense read on the CPU; compressed masters need x64 Python | CPU tested |
| Mac, Apple silicon | `quantem.widget[mps]` | Apple GPU (MPS) | `Dataset4dstemGPU`, encoded on the GPU by quantem.gpu (Metal) | yes |
| Mac, Intel | `quantem.widget[cpu]` (torch 2.2, NumPy 1.26), no quantem.gpu, no quantem core | AMD GPU through torch MPS, else CPU | the stand-in, dense read on the CPU | yes |
| No GPU | `quantem.widget[cpu]` | CPU | quantem core `Dataset4dstem`, dense read on the CPU | yes |

- `[cpu]` (and a plain install) brings torch 2.2 or newer, NumPy, h5py,
  hdf5plugin and [quantem](https://github.com/electronmicroscopy/quantem) core
  (hdf5plugin and quantem core not on Windows on ARM, quantem core not on an
  Intel Mac, see below): every widget, and `Show4DSTEM` on in-memory arrays,
  tensors on any device and files read densely into memory.
- `[cuda]` adds [quantem.gpu](https://github.com/bobleesj/quantem.gpu) with
  CuPy; `[mps]` adds it with Metal and MLX. A 4D-STEM file then stays encoded on
  the GPU and loads in seconds. A wrong pick installs nothing harmful (`[cuda]`
  installs nothing extra on a Mac, `[mps]` nothing outside Apple silicon), and
  the widget's one-line notice names the right extra.
- Intel Macs are supported without quantem.gpu: PyTorch publishes no wheel newer
  than torch 2.2 for them and quantem.gpu needs torch 2.3. The widgets compute
  on the Mac's AMD GPU through torch MPS (else the CPU), and the first widget
  prints one line saying so. torch 2.2 imports only with NumPy 1.x on these
  Macs; the package pins `numpy<2` for them.
- The readers return the dataset type of the machine, so the same code runs
  everywhere: `read_4dstem` gives a `quantem.gpu.io.Dataset4dstemGPU` when
  quantem.gpu is installed and a GPU is present, else a quantem core
  `Dataset4dstem`; `read_image` and `read_images` give quantem core
  `Dataset2d`, `read_image_stack` a `Dataset3d`. quantem core needs torch 2.7
  and NumPy 2 (no Intel Mac wheels) and hdf5plugin 6 (no Windows on ARM
  wheel), so on those two machines the readers return the widget's stand-in,
  with the same `array`, `name`, `sampling`, `units`, `metadata`, `shape`,
  `ndim` and `dtype`. Every widget accepts all of these, NumPy arrays and
  torch tensors.

  ```python
  from quantem.widget import io

  ds = io.read_image(path)
  arr = ds.array
  ```

A file opened without the GPU path is read whole into memory. `Show4DSTEM`
prints one line with the shape, dtype, size and seconds taken, and refuses with
a clear message when the array would not fit in 80% of the available memory. It
never bins or crops to make the data fit.

Tutorial data downloads (`quantem.widget.datasets`, `quantem.widget.io.download`)
need only `huggingface_hub`, which installs with the package. To upload new
shared datasets, add the `[hub]` extra, which pulls in `quantem.data`:

```bash
pip install -i https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    "quantem.widget[hub]"
```

Widget tutorial fixtures live under `widget-tutorials/` on
[bobleesj/quantem-data](https://huggingface.co/datasets/bobleesj/quantem-data).
Upload and download commands are on that dataset card.

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
pip install "quantem.widget[cpu]"
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
Show3DSlices. The ShowCIF potential preview requires abTEM through the `crystal`
extra:

```bash
python -m pip install -i https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ 'quantem.widget[crystal]'
```

Check that the installed build includes the API you want to demonstrate; the
new ShowCIF notebook requires a revision containing ShowCIF. For an unpublished
source revision, use the [developer setup](https://github.com/electronmicroscopy/quantem.widget/blob/main/CONTRIBUTING.md)
and install `'.[crystal]'`, then build the JavaScript with `npm run build`.
Use the same Python environment for installation and the notebook kernel.

## Google Colab

Tutorials with a Colab badge can open directly in Colab. The new crystal demo
can also be run from a local source checkout as described above. Colab uses the same files that build these docs, so there is no
separate Colab copy to maintain.

Each Colab-ready tutorial has one collapsed **Install QuantEM** cell. Its two
plain steps download and run the shared `scripts/install_colab.py` installer, which
resolves only the newest `quantem.widget` and `quantem.gpu` wheel URLs from
TestPyPI. Normal dependencies still come from PyPI, and Colab's loaded NumPy
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

## Verify

```python
import quantem.widget as qw

qw.profile()
```
