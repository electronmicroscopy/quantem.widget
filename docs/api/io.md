# Load and I/O

The `quantem.widget.io` module loads, saves, discovers, and inspects 4D-STEM and
image data. For most workflows, start with `load` for 4D-STEM files or
`read_image` / `read_image_stack` for images.

## 4D-STEM loader

`load` reads compressed 4D-STEM data straight onto the GPU (CUDA / Apple Metal)
or CPU and returns a `LoadResult` you hand to [`Show4DSTEM`](show4dstem).

```python
from quantem.widget import load, Show4DSTEM

data = load("scan_master.h5")   # CUDA on a workstation, MPS on a MacBook
Show4DSTEM(data)
```

```{eval-rst}
.. autofunction:: quantem.widget.io.hdf5.load
```

```{tip}
`det_bin=2` or `det_bin=4` bins the detector on load to cut memory and speed
first paint. Pass a list of file paths to stack several datasets behind a single
"Dataset" slider.
```

### Backend selection

`load` detects the device automatically: an NVIDIA box loads onto **CUDA**, a Mac
loads onto **Apple Metal (MPS)**, and anything else falls back to **CPU**. No flag
is required. On a MacBook, the read uses a zero-copy Metal path, so a laptop can
browse 4D-STEM data that does not fit in RAM **if you bin at load**:

```python
data = load("scan_master.h5", det_bin=8)
Show4DSTEM(data)
```

The same workflow is available from the shell:

```bash
quantem show4dstem scan_master.h5 --bin 8
```

### Multi-file browsing

Use the same entry point for a time-series or tilt-series stack:

```python
masters = [
    "/data/session/file_001_master.h5",
    "/data/session/file_002_master.h5",
    "/data/session/file_003_master.h5",
]

data = load(masters, det_bin=4, dtype="u8")
Show4DSTEM(data)
```

The returned data has shape `(n_files, scan_y, scan_x, det_y, det_x)`.
`Show4DSTEM` labels the extra axis as `Dataset` and uses source filenames as
slider labels when they are available.

Memory rule of thumb for Sample-scale `512x512x192x192` data:

| mode | approximate resident size per file |
|---|---:|
| no bin, uint16 | 18-20 GiB |
| `det_bin=2`, uint16 | 4.5-5 GiB |
| `det_bin=4`, uint16 | 1.1-1.3 GiB |
| `det_bin=4`, `dtype="u8"` | about half of uint16 |

Use `det_bin=4, dtype="u8"` for first-pass browsing. Use uint16/no-bin only
when exact diffraction intensities matter and the GPU memory budget is clean.

## Other I/O helpers

```python
from quantem.widget.io import survey, read_image, get_metadata, bin, download
```

## Discover & inspect

```{eval-rst}
.. autofunction:: quantem.widget.io.survey.survey
```
```{eval-rst}
.. autofunction:: quantem.widget.io.hdf5.discover_masters
```
```{eval-rst}
.. autofunction:: quantem.widget.io.hdf5.get_metadata
```

## Images (2D / 3D)

```{eval-rst}
.. autofunction:: quantem.widget.io.image.read_image
```
```{eval-rst}
.. autofunction:: quantem.widget.io.image.read_image_stack
```

## Detector binning

```{eval-rst}
.. autofunction:: quantem.widget.io.hdf5.bin
```

## Hugging Face datasets

```{eval-rst}
.. autofunction:: quantem.widget.io.hub.list_datasets
```
```{eval-rst}
.. autofunction:: quantem.widget.io.hub.download
```

## Save

```{eval-rst}
.. autofunction:: quantem.widget.io.save.save
```
