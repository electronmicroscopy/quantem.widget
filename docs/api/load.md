# load

Reads compressed 4D-STEM data straight onto the GPU (CUDA / Apple Metal) or CPU
and returns a `LoadResult` you hand to [`Show4DSTEM`](show4dstem). Public import:

```python
from quantem.widget import load
```

## Reference

```{eval-rst}
.. autofunction:: quantem.widget.io.hdf5.load
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
detector on the way in with integer-sum binning, so the full multi-gigabyte
stack never has to materialize. A typical laptop session:

```python
data = load("scan_master.h5", det_bin=8)   # MPS, detector binned 8x -> small
Show4DSTEM(data)
```

MPS loads include a preflight memory guard. Before allocating Metal buffers, the
loader estimates the output footprint from HDF5 metadata and compares it to the
Mac's recommended Metal working set. No-bin loads that exceed that budget fail
early with a specific `det_bin` recommendation instead of risking a frozen
laptop:

```python
data = load("scan_master.h5", backend="mps", det_bin=4)
```

Bypass the check only when you have a clean memory budget and need the exact
no-bin stack:

```python
data = load("scan_master.h5", backend="mps", skip_mps_memory_check=True)
```

For fastest browse workflows, combine detector binning with compact dtype:

```python
data = load("scan_master.h5", backend="mps", det_bin=8, dtype="u8")
Show4DSTEM(data)
```

That path is intended for screening, layout, and export-to-browser review. Use
uint16/full precision when detector counts are part of the scientific claim and
the GPU memory budget is clean.

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

The returned data has shape `(n_files, scan_row, scan_col, det_row, det_col)`.
`Show4DSTEM` labels the extra axis as `Dataset` and uses the source filenames
as slider labels when they are available.

`dtype="u8"` is a browse contract, not a reconstruction contract. It routes to
the direct `uint8` output path before stacking or sharding, so the loader avoids
materializing a full `uint16` stack first. Counts above 255 clip to 255; use
`dtype="u16"` or omit `dtype` when detector counts are part of the scientific
claim.

For larger series or solver workflows that should stay sharded across GPUs,
use:

```python
parts = load(masters, det_bin=2, gpus=[0, 1], stack=False)
```

For browse-scale multi-GPU sessions, use the same public dtype vocabulary:

```python
parts = load(masters, det_bin=1, dtype="u8", devices=[0, 1])
```

The sharded result keeps one stack per GPU and records the file-to-device map in
the metadata. It is disk-aware: when masters are split across independent NVMe
mounts, the loader interleaves files by physical disk before assigning them to
GPUs. When all files live on one disk, sharding still increases GPU capacity,
but cold load remains disk-bound.

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
