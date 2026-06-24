# load

Reads compressed 4D-STEM data straight onto the GPU (CUDA / Apple Metal) or CPU
and returns a `LoadResult` you hand to [`Show4DSTEM`](show4dstem). Public import:

```python
from quantem.widget import load
```

## Reference

```{autodoc2-object} quantem.widget.io.hdf5.load
render_plugin = "myst"
```

```{tip}
`det_bin=2` (or `4`) bins the detector on load to cut memory and speed first
paint; pass a list of file paths to stack several datasets behind a single
"Dataset" slider.
```

## Backend (CUDA / Apple Silicon / CPU)

`load` detects the device automatically: an NVIDIA box loads onto **CUDA**, a Mac
loads onto **Apple Metal (MPS)**, and anything else falls back to **CPU**. No flag
is required - the same call works everywhere:

```python
from quantem.widget import load, Show4DSTEM

data = load("scan_master.h5")   # CUDA on a workstation, MPS on a MacBook
Show4DSTEM(data)
```

On a MacBook the read uses a zero-copy Metal path, so a laptop can browse 4D-STEM
data that does not fit in RAM **if you bin at load**: `det_bin` reduces the
detector on the way in (mean reduction), so the full multi-gigabyte stack never
has to materialize. A typical laptop session:

```python
data = load("scan_master.h5", det_bin=8)   # MPS, detector binned 8x -> small
Show4DSTEM(data)
```

The same is one shell command - see [the CLI](../cli): `quantem show4dstem
scan_master.h5 --bin 8`.

## Multi-File / 5D Browsing

Use one public entry point for a time-series or tilt-series stack:

```python
from quantem.widget import load, Show4DSTEM

masters = [
    "/data/session/file_001_master.h5",
    "/data/session/file_002_master.h5",
    "/data/session/file_003_master.h5",
]

r = load(masters, det_bin=4, dtype="u8")
w = Show4DSTEM(r)
```

The returned data has shape `(n_files, scan_y, scan_x, det_y, det_x)`.
`Show4DSTEM` labels the extra axis as `Dataset` and uses the source filenames
as slider labels when they are available.

For larger series or solver workflows that should stay sharded across GPUs,
use:

```python
parts = load(masters, det_bin=2, gpus=[0, 1], stack=False)
```

For an explicit logical series object:

```python
series = load(
    masters,
    det_bin=4,
    dtype="u8",
    series_type="time",
    series=[0.0, 1.0, 2.0],
    units=["s", "pixels", "pixels", "mrad", "mrad"],
)
```

Memory rule of thumb for Sample-scale `512x512x192x192` data:

| mode | approximate resident size per file |
|---|---:|
| no bin, uint16 | 18-20 GiB |
| `det_bin=2`, uint16 | 4.5-5 GiB |
| `det_bin=4`, uint16 | 1.1-1.3 GiB |
| `det_bin=4`, `dtype="u8"` | about half of uint16 |

Use `det_bin=4, dtype="u8"` for first-pass browsing. Use uint16/no-bin only
when exact diffraction intensities matter and the GPU memory budget is clean.
