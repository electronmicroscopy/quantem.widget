# IO/GPU

Use this page when you have microscope data on disk and want to open it without
wasting memory. The basic workflow is:

```python
from quantem.widget import load, Show4DSTEM
from quantem.widget.io import survey

survey("/data/session")        # inspect first; no pixel data loaded
data = load("scan_master.h5")  # full precision when it fits
Show4DSTEM(data)
```

If the data is too large, reduce it deliberately:

```python
data = load("scan_master.h5", det_bin=2)              # smaller detector, still uint16
data = load("scan_master.h5", det_bin=4, dtype="u8")  # small browse copy
```

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

## Inspect before loading

Use `survey` to inspect a folder before allocating memory:

```python
from quantem.widget.io import survey

survey("/data/session")
```

It reports scan size, detector size, dtype, completeness, and memory estimates.
If the folder contains many scans, filter before loading:

```python
from quantem.widget.io import discover_masters

masters = discover_masters("/data/session", scan_shape=(512, 512))
```

## NVIDIA GPU workflow

Most lab workflows should run Python on the NVIDIA workstation and open
JupyterLab from a laptop. The workstation holds the data and runs the GPU work;
the laptop is the frontend.

```python
from quantem.widget import load, Show4DSTEM

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

Choose a specific NVIDIA GPU before launching Jupyter:

```bash
CUDA_VISIBLE_DEVICES=0 jupyter lab --no-browser --ip=0.0.0.0
CUDA_VISIBLE_DEVICES=1 jupyter lab --no-browser --ip=0.0.0.0
```

Check what the notebook sees:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.mem_get_info())  # free bytes, total bytes
```

Clean up the current Python kernel when you are done with a large dataset:

```python
del data

import gc
import torch

gc.collect()
torch.cuda.empty_cache()
```

This cleanup is for NVIDIA CUDA memory. If memory is still occupied, another
object or another Jupyter kernel still owns it.

## Apple Silicon workflow

On a MacBook, the same API works:

```python
from quantem.widget import load, Show4DSTEM

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

## What should I choose?

1. Need exact detector counts or reconstruction: keep `uint16`.
2. Browsing a session to find good fields of view: try `det_bin=4, dtype="u8"`.
3. Need live virtual detectors with useful angular detail: try `det_bin=2`.
4. Have 48 GB or more GPU memory: start full resolution.
5. Making a tutorial or HTML preview: use a visibly reduced copy and say how it
   was reduced.

Do not silently crop, bin, or change dtype in a tutorial. Put the reduction in
the code so another scientist can see exactly what happened.

## Related pages

- {doc}`../api/io`
- {doc}`show4dstem`
- {doc}`widget_export`
