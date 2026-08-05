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
fast and small, but it can saturate real counts above 255.

If you request `dtype="u8"` and any counts exceed 255, `load` warns you:

```text
Warning: dtype='u8' saturated 1,482,913 pixels (0.0124%) above 255 to 255.
Pass dtype='u16' or 'auto' to keep full counts.
```

That warning is intentional. It means the browsing copy is no longer exact, so
use `dtype="u16"` or the default load for quantitative work.

## Size estimates

A `4096 x 4096` image is about:

| dtype | size |
|---|---:|
| `uint8` | 16 MB |
| `uint16` | 32 MB |
| `float32` | 64 MB |

A common `512 x 512 x 192 x 192` 4D-STEM scan is much larger:

| load mode | approximate size |
|---|---:|
| full detector, `uint16` | 18-20 GB |
| `det_bin=2`, `uint16` | 4.5-5 GB |
| `det_bin=4`, `uint16` | 1.1-1.3 GB |
| `det_bin=4`, `uint8` | about 0.6 GB |

Leave a few GB free for the viewer, browser, and downstream processing.

## NVIDIA GPU workflow

Most lab workflows should run Python on the NVIDIA workstation and open
JupyterLab from a laptop. The workstation holds the data and runs the GPU work;
the laptop is the frontend.

```python
from quantem.gpu.io import load
from quantem.widget import Show4DSTEM

data = load("scan_master.h5")  # CUDA is selected automatically when available
Show4DSTEM(data)
```

Good first choices:

| GPU memory | first try |
|---:|---|
| 96 GB | `load(path)` |
| 48 GB | `load(path)` |
| 24 GB | `load(path)` for browsing, `load(path, det_bin=2)` if reconstruction also runs |
| 16 GB or less | `load(path, det_bin=4, dtype="u8")` for browsing |

Check the GPU before and after a large load with `quantem.widget.io.memory()`:

```python
from quantem.gpu.io import load
from quantem.widget import Show4DSTEM
from quantem.widget.io import memory

memory()  # check VRAM before loading
data = load("scan_001_master.h5", verbose=True)
print(data.data.shape, data.data.dtype, f"{data.data.nbytes / 1e9:.1f} GB")
memory()  # confirm VRAM after loading
```

Typical output:

```text
VRAM GPU0    12.6 /   95.0 GB used   (82.4 free)   [torch 0.0, cupy 0.0]   NVIDIA RTX PRO 6000
RAM          84.1 /  540.0 GB used   (447.2 free)
  Loaded 1,048,576 frames (19.3 GB) in 6.42s (3.0 GB/s)
(1024, 1024, 96, 96) uint16 19.3 GB
VRAM GPU0    32.1 /   95.0 GB used   (62.9 free)   [torch 0.0, cupy 19.3]  NVIDIA RTX PRO 6000
RAM          84.4 /  540.0 GB used   (446.8 free)
```

Read this as: the full detector-count stack is now on the NVIDIA GPU as
`uint16`; no browser copy has been quantized.

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

data = load("scan_001_master.h5")
Show4DSTEM(data)
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

Keep a handle to the viewer if you plan to release memory later:

```python
from quantem.gpu.io import load
from quantem.widget import Show4DSTEM

data = load("scan_001_master.h5")
viewer = Show4DSTEM(data)
viewer
```

When you are done with that dataset, free the viewer and delete the loaded data:

```python
viewer.free()   # releases widget tensor/backend caches
viewer.close()  # closes the ipywidget comm/model
del viewer
del data

import gc
import torch

gc.collect()
torch.cuda.empty_cache()

try:
    import cupy as cp
except Exception:
    pass
else:
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
```

If `viewer` or `data` still exists anywhere in the notebook, the memory is still
owned by the live Python process. That is correct behavior. A small residual
allocation can remain after cleanup because CUDA keeps a runtime context and
small caches alive until the kernel exits.

NVIDIA GPU cleanup check, using a real `Au_TiO2_030_master.h5` 4D-STEM scan
loaded as `det_bin=4, dtype="u8"`:

```text
GPU 0: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
before:              free 74.12 GiB / total 94.95 GiB
after load:          free 71.72 GiB   cupy used 2.25 GiB
after Show4DSTEM:    free 70.64 GiB   torch reserved 0.98 GiB
after cleanup:       free 73.74 GiB   cupy used 0.00 GiB
residual:            0.38 GiB CUDA runtime/cache overhead

GPU 1: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
before:              free 80.95 GiB / total 94.97 GiB
after load:          free 78.68 GiB   cupy used 2.25 GiB
after Show4DSTEM:    free 77.60 GiB   torch reserved 0.98 GiB
after cleanup:       free 80.70 GiB   cupy used 0.00 GiB
residual:            0.25 GiB CUDA runtime/cache overhead
```

If memory is still occupied after this pattern, another variable, notebook, or
kernel still owns it. Shut down old kernels from JupyterLab before assuming the
GPU is stuck.

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

data = load("scan_master.h5")
Show4DSTEM(data)
```

Mac unified memory is shared by the operating system, browser, Python, and GPU.
If the machine feels tight, start with:

```python
data = load("scan_master.h5", det_bin=2)
```

For a small preview or teaching copy:

```python
data = load("scan_master.h5", det_bin=4, dtype="u8")
```

## Related pages

- {doc}`io_gpu`
- {doc}`../api/io`
- {doc}`show4dstem`
