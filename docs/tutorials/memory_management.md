# Memory management

Use this page when you care about how much RAM/VRAM a dataset takes, which GPU
runs the work, and how to give the memory back when you are done. For opening
files and choosing a loader, start with {doc}`io_gpu`.

## Dtype in plain language

The `dtype` is how each number is stored.

| dtype | range | size | use it for |
|---|---:|---:|---|
| `uint8` / `u8` | 0 to 255 | 1 byte | fast preview copies |
| `uint16` / `u16` | 0 to 65535 | 2 bytes | raw detector counts |
| `float32` / `f4` | decimals | 4 bytes | processed maps |

For raw electron detector counts, start with `uint16`. It keeps the measured
counts exactly and is still much smaller than `float32`.

Use `uint8` only when you want a lightweight preview or tutorial copy. It is
fast and small, but it can saturate real counts above 255. `load` always keeps
the detector's own dtype; `uint8` appears only where you choose it, such as
`export_html(dtype="uint8")` or `quantem show4dstem ... --html --dtype uint8`.

## Size estimates

A `4096 x 4096` image is about:

| dtype | size |
|---|---:|
| `uint8` | 16 MB |
| `uint16` | 32 MB |
| `float32` | 64 MB |

A common `512 x 512 x 192 x 192` 4D-STEM scan is 18 GiB as a dense `uint16`
array. `load` never creates that array: it keeps the acquisition ANS encoded on
the GPU at full detector resolution.

| form | approximate size |
|---|---:|
| dense `uint16` array | 18 GiB |
| encoded acquisition from `load` | 0.1 to 2 GiB, depending on counts |
| bounded `read` of 64 scan rows, `uint16` | 2.25 GiB |

Leave a few GB free for the viewer, browser, and downstream processing.

## NVIDIA GPU workflow

Most lab workflows should run Python on the NVIDIA workstation and open
JupyterLab from a laptop. The workstation holds the data and runs the GPU work;
the laptop is the frontend.

```python
from quantem.gpu.io import load
from quantem.widget import Show4DSTEM

loaded = load("scan_master.h5")  # CUDA is selected automatically when available
Show4DSTEM(loaded)
```

The same call fits every common GPU size, because the encoded acquisition is a
small fraction of the dense array. Memory pressure comes from what you read or
reconstruct from it, such as large bounded reads or SSB workspaces.

Check the GPU before and after a large load with `quantem.widget.profile()`:

```python
import quantem.widget as qw
from quantem.gpu.io import load

qw.profile()  # check VRAM before loading
loaded = load("scan_001_master.h5", verbose=True)
print(loaded.shape, loaded.dtype,
      f"{loaded.resident_bytes / 2**30:.2f} GiB encoded, {loaded.logical_bytes / 2**30:.1f} GiB dense")
qw.profile()  # confirm VRAM after loading
```

For a real `256 x 256 x 192 x 192` Arina scan (uint32 counts on disk), the
print line reads:

```text
(256, 256, 192, 192) uint16 0.57 GiB encoded, 4.5 GiB dense
```

Read this as: every detector pixel and count is on the NVIDIA GPU in encoded
form, 0.57 GiB instead of 4.5 GiB; the counts fit in `uint16`, so they are
stored that way, and no copy has been binned or quantized. The `profile()` lines
show the whole GPU, including other processes.

## Can I choose the NVIDIA GPU inside the notebook?

Yes. Put this in the first notebook cell, before importing `torch`, `cupy`,
`quantem.widget`, or any other GPU package:

```python
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use physical NVIDIA GPU 0
```

Then import and load normally:

```python
import torch
from quantem.gpu.io import load
from quantem.widget import Show4DSTEM

print(torch.cuda.get_device_name(0))

loaded = load("scan_001_master.h5")
Show4DSTEM(loaded)
```

Example output on a Linux workstation with NVIDIA GPUs:

```text
cuda available: True
visible device count: 1
notebook device 0: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
free 80.3 GiB / total 94.9 GiB
```

Inside that notebook, the selected GPU is called `cuda:0`. CUDA renumbers the
visible device, so physical GPU 1 also appears as `cuda:0` if you selected it
with `CUDA_VISIBLE_DEVICES="1"`.

## How do I switch from GPU 0 to GPU 1?

Change the first cell, restart the kernel, then run from the top:

```python
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # switch to physical NVIDIA GPU 1
```

Example output after restarting the Python process with GPU 1 selected:

```text
cuda available: True
visible device count: 1
notebook device 0: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
free 94.4 GiB / total 95.0 GiB
```

Restarting matters. Once CUDA is initialized in a Python process, changing
`CUDA_VISIBLE_DEVICES` later in the notebook is not a reliable way to move the
work to another GPU.

## How do I use two NVIDIA GPUs at the same time?

Run one Jupyter process per GPU. Start each server with a different
`CUDA_VISIBLE_DEVICES` value:

```bash
# terminal 1: GPU 0
CUDA_VISIBLE_DEVICES=0 jupyter lab --no-browser --ip=0.0.0.0 --port=8888

# terminal 2: GPU 1
CUDA_VISIBLE_DEVICES=1 jupyter lab --no-browser --ip=0.0.0.0 --port=8889
```

Then open the printed URLs from your laptop. Each notebook sees its assigned
GPU as `cuda:0`.

Check what the notebook sees:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.mem_get_info())  # free bytes, total bytes
```

## Check and free GPU memory

Check the GPU before and after a large load:

```python
import torch

free, total = torch.cuda.mem_get_info()
print(f"free {free / 1e9:.1f} GB / total {total / 1e9:.1f} GB")
```

Keep handles to the acquisition and the viewer if you plan to release memory
later:

```python
from quantem.gpu.io import load
from quantem.widget import Show4DSTEM

loaded = load("scan_001_master.h5")
viewer = Show4DSTEM(loaded)
viewer
```

When you are done with that dataset, release the viewer first and then the
acquisition it borrows:

```python
viewer.free()    # releases widget tensor/backend caches
viewer.close()   # closes the ipywidget comm/model
loaded.close()   # returns the encoded storage to the GPU
```

A viewer from `Show4DSTEM.from_folder(...)` owns the acquisitions it loaded;
`viewer.free()` or `viewer.close()` closes them.

If memory is still occupied after this pattern, another variable, notebook, or
kernel still owns it. A small residual allocation can remain because CUDA keeps
a runtime context and small caches alive until the kernel exits. Shut down old
kernels from JupyterLab before assuming the GPU is stuck.

## Moving image data to Torch or CuPy

Most viewers accept NumPy arrays or quantem datasets directly, so you usually do
not need to move a PNG, TIFF, or EMD survey image to Torch just to view it. Move
data to the GPU when you are about to run your own GPU computation.

For Torch:

```python
import numpy as np
import torch
from quantem.widget import read_image

ds = read_image("haadf.emd")
image = np.ascontiguousarray(ds.array, dtype=np.float32)

device = "cuda" if torch.cuda.is_available() else "cpu"
image_t = torch.as_tensor(image, device=device)
```

For a large CPU array that you will reuse many times on an NVIDIA GPU:

```python
if torch.cuda.is_available():
    image_t = torch.from_numpy(image).pin_memory().to("cuda", non_blocking=True)
    torch.cuda.synchronize()
```

For CuPy:

```python
import cupy as cp

image_gpu = cp.asarray(image)
```

Keep raw detector counts as `uint16` until you need decimal math. Convert to
`float32` for filtering, fitting, normalization, neural networks, or display
processing. Avoid accidental `float64`; it doubles memory with no benefit for
normal interactive viewing.

For large `.npy` files, memory-map first so Python does not copy the whole file
before you decide what to view:

```python
import numpy as np

stack = np.load("stack.npy", mmap_mode="r")
preview = np.asarray(stack[::8], dtype=np.float32)  # explicit preview reduction
```

## Apple Silicon workflow

On a MacBook, the same API works:

```python
from quantem.gpu.io import load
from quantem.widget import Show4DSTEM

loaded = load("scan_master.h5")   # Apple GPU (MPS), encoded like on CUDA
Show4DSTEM(loaded)
```

Mac unified memory is shared by the operating system, browser, Python, and GPU.
The encoded acquisition uses a small part of it; large bounded reads and
reconstructions use the rest, so read the scan region you need rather than the
whole scan.

## Related pages

- {doc}`io_gpu`
- {doc}`../api/io`
- {doc}`show4dstem`
