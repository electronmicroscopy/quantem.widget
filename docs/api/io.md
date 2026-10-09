# Image and acquisition I/O

The `quantem.widget` readers return the dataset type of the machine, so the
same code runs on a GPU workstation, a laptop without a GPU and an Intel Mac.
The [QuantEM.GPU I/O guide](https://github.com/bobleesj/quantem.gpu/blob/main/docs/api/io.md) owns the detector
storage, supported formats, metadata, backend, and exactness contracts of the
GPU path. For a walkthrough of `uint8`/`uint16`, memory estimates and the image
readers, start with {doc}`IO/GPU <../tutorials/io_gpu>`; GPU selection and
cleanup are in {doc}`Memory management <../tutorials/memory_management>`.

## Read data on any machine

```python
from quantem.widget import Show2D, Show4DSTEM, io

ds = io.read_image(path)
arr = ds.array                      # the pixels; ds.sampling, ds.units and ds.name carry the rest
Show2D(ds)

data = io.read_4dstem("scan_master.h5")
Show4DSTEM(data)
```

| Reader | quantem.gpu and a GPU | No GPU route (`[cpu]`, or `device="cpu"`) | Intel Mac, Windows on ARM |
|---|---|---|---|
| `read_4dstem(path)` | `quantem.gpu.io.Dataset4dstemGPU`, encoded on the GPU | quantem core `Dataset4dstem`, read densely into host memory | stand-in, read densely |
| `read_image(path)`, `read_images(folder)` | quantem core `Dataset2d` | quantem core `Dataset2d` | stand-in |
| `read_image_stack(folder)` | quantem core `Dataset3d` | quantem core `Dataset3d` | stand-in |

`read_4dstem` loads through `quantem.gpu.io.load` when quantem.gpu is installed
and a CUDA or Metal GPU works; a list of paths returns one dataset per file. A
`.npy` file that quantem.gpu cannot hold exactly (signed counts, float64 that
float32 does not hold, a flat 3D array) is read densely instead, with one
printed line. A dense read keeps every count exactly, prints one line with the
time taken, and refuses rather than bins or crops above 80% of the available
memory. Counts keep uint8 or uint16, and wider counts become uint16 when every
count fits, on both routes. quantem core cannot install on an Intel Mac (torch
2.7, NumPy 2) or Windows on ARM (hdf5plugin 6); there the readers return the
widget's stand-in, with the same `array`, `name`, `sampling` (a float64 array),
`units` (a list), `metadata`, `shape`, `ndim` and `dtype` as a quantem core
dataset. A color PNG, JPEG or TIFF reads as an `RgbImage` with `array`,
`name`, `sampling` and `units`, and a folder of color frames as an
`(N, H, W, 3)` array: quantem core has no color-image dataset. Every widget
accepts each of these, NumPy arrays and torch tensors on any device, and shows
the same numbers for each.

On a GPU, a 4D-STEM file is a `Dataset4dstemGPU`, not a torch tensor:

- counts are unsigned (uint16, uint32), and torch cannot add, multiply or
  take the max of those;
- the file stays compressed in GPU memory (the 512 x 512 x 192 x 192 gold
  scan: 5.6 GiB, against 18 GiB as dense uint16);
- quantem.gpu's CUDA and Metal kernels decode and sum it exactly in one pass.

Indexing and `read(scan_region=...)` still return torch tensors.

Every 4D dataset has `shape`, `dtype`, `ndim`, `sampling`, `units` and
`metadata`, and `Show4DSTEM` takes any of them, with `scan_region=` for part of
the scan. A `Dataset4dstemGPU` reports an uncalibrated axis as `None` in
`sampling` and `units` where a dense dataset has `1.0` and `"pixels"`, and a
float64 file that float32 holds exactly as float32. A dense `Dataset4dstem`
also has `array` (a NumPy array) and `name`; a `Dataset4dstemGPU` keeps its
counts on the GPU, returns them as torch tensors from indexing
(`data[10, 12]`) and `read(scan_region=...)`, and has `close()`.

## Open an acquisition on the GPU and inspect a pattern

```python
from quantem.gpu import detector, io
from quantem.widget import Show2D, Show4DSTEM

data = io.load("scan_master.h5")   # what read_4dstem does on a GPU
Show4DSTEM(data)
```

`data` is a `quantem.gpu.io.Dataset4dstemGPU`, ANS encoded on the CUDA device or
the Apple GPU at native detector sampling and count dtype: a 512 x 512 x 192 x 192
uint16 scan, 18 GiB as a dense array, occupies about 0.1 to 2 GiB. Keep it open
while viewers or calculations use its encoded buffers. Array-style selection
decodes only the selected region into a Torch tensor on the acquisition's GPU;
it does not unpack the complete acquisition:

```python
Show2D(data[10, 12])
```

| Expression | Selection |
| --- | --- |
| `data[10, 12]` | One diffraction pattern |
| `data[8:12, 10:16]` | A rectangular scan patch |
| `data[10, 12, 64:128, 64:128]` | A detector crop at one position |
| `data[:, :, 95, 100]` | One detector pixel across the scan |
| `data.metadata` | Scientific calibration and source metadata |

```python
Show2D([detector.bf(data), detector.adf(data)], labels=["BF", "ADF"])
io.save("scan.qem", data)
```

Call `data.close()` after the last use. Scripted jobs can use
`with io.load(path) as data:` for automatic cleanup. A list of paths loads as a
list of encoded acquisitions, one per file; `Show4DSTEM(io.load(paths))` opens
them as a comparison grid labelled by file, and `Show4DSTEM(series[1])` opens
one of them. `io.load(path, backend="cuda", device=1)` selects a CUDA device; a
Mac loads onto MPS. See [load](load.md) for the function reference and the GPU
guide for multi-device storage policies.

## Read a scan region

Reconstruction, denoise and ROI workflows read the rectangular scan region they
need. `read` decodes that region into a Torch tensor on the acquisition's GPU:

```python
with io.load("scan_master.h5") as data:
    patch_t = data.read(scan_region=(160, 293, 234, 367))  # row_start, row_stop, col_start, col_stop
print(patch_t.shape)  # torch.Size([133, 133, 192, 192])
```

Stops are exclusive, and `detector_region=` bounds the detector pixels the same
way. For a drift-corrected time series, derive `scan_region` from the shared
specimen ROI, the frame shift and a small scan halo, then sample the final ROI
from the local patch. The detector counts remain raw; drift stays as
scan-position metadata.

## Discover acquisitions

```python
from quantem.gpu import io
from quantem.widget import Show4DSTEM

files = io.discover("/data/session")
Show4DSTEM(io.load(files[0]))
```

`io.discover(path, scan_shape=(512, 512))` keeps only matching acquisitions when
a folder mixes scan sizes. `io.inspect(path)` reads readiness and calibration
headers without decoding measurements. Use `Show4DSTEM.from_folder(path)` for a
live acquisition viewer.

For named sample data, see [Tutorial datasets](datasets.md). The [IO/GPU
notebook](../tutorials/io_gpu.ipynb) demonstrates a real Gold image session.

## Read images and image stacks

```python
from quantem.widget import Show2D, Show3D, read_image, read_images, read_image_stack

image = read_image("overview.tif")
Show2D(image)
```

```python
images = read_images("survey_images", workers=8)
Show2D(images)
```

```python
stack = read_image_stack("frames", file_type="tif", workers=8)
Show3D(stack)
```

These format readers run on the host. Calibrated image inputs retain their
sampling for viewer scale bars. This path is for individual images and image
sequences, not compressed 4D-STEM detector acquisitions.

## Share data and finished figures

Use [Contribute tutorial data](../tutorials/contribute_data.md) for shared
example datasets and [HTML and file export](../tutorials/widget_export.ipynb)
for interactive figures. Acquisition export belongs to `io.save`, described
in the GPU I/O guide; it is separate from exporting a viewer.

## Function reference

```{eval-rst}
.. autofunction:: quantem.widget.read_4dstem
.. autofunction:: quantem.widget.read_image
.. autofunction:: quantem.widget.read_images
.. autofunction:: quantem.widget.read_image_stack
```
