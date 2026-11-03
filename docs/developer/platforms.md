# Platform support

What each install line of the [installation page](../install.md) brings on each
machine, which dataset type the readers return there, and what was tested.

| Machine | Install | Widget compute (`device="auto"`) | `read_4dstem(path)`, `Show4DSTEM(path)` | Tested |
|---|---|---|---|---|
| Linux, NVIDIA GPU | `quantem.widget[cuda]` | CUDA | kept encoded on the GPU by quantem.gpu | yes |
| Windows, NVIDIA GPU | `quantem.widget[cuda]` | CUDA | kept encoded on the GPU by quantem.gpu | CPU tested, CUDA untested |
| Windows on ARM | torch from the PyTorch CPU index, then `quantem.widget[cpu]` (no quantem core) | CPU | the stand-in, dense read on the CPU; compressed masters need x64 Python | CPU tested |
| Mac, Apple silicon | `quantem.widget[mps]` | Apple GPU (MPS) | kept encoded on the GPU by quantem.gpu (Metal) | yes |
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
  everywhere: `read_4dstem` gives quantem.gpu's dataset when
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
