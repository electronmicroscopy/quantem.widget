# load

Reads compressed 4D-STEM data straight onto CUDA or Apple Metal and returns a
typed `LoadResult` accepted directly by [`Show4DSTEM`](show4dstem). Public import:

```python
from quantem.gpu.io import load
```

## Reference

```{eval-rst}
.. autofunction:: quantem.gpu.io.load
```

```{tip}
The default load keeps native detector sampling when the memory budget allows.
Use `det_bin=2` or `4` only as an explicit preview or memory policy; pass a list
of file paths to stack several datasets behind a single "Dataset" slider.
```

## Backend (CUDA / Apple Silicon)

`load` detects the native GPU automatically: an NVIDIA box loads onto **CUDA**
and a Mac loads onto **Apple Metal (MPS)**. Scientific loading does not silently
fall back to CPU.

```python
from quantem.gpu.io import load
from quantem.widget import Show4DSTEM

result = load("scan_master.h5")   # CUDA on a workstation, MPS on a MacBook
Show4DSTEM(result)
```

On a MacBook the read uses a raw-Metal path, so a memory-rich laptop can browse
native 4D-STEM directly and smaller machines can use an explicit detector-bin
preview. A conservative preview session is:

```python
result = load("scan_master.h5", det_bin=8)   # MPS, detector binned 8x -> small preview
Show4DSTEM(result)
```

MPS loads include a preflight memory guard. Before allocating Metal buffers, the
loader estimates the output footprint from HDF5 metadata and compares it to the
Mac's recommended Metal working set. No-bin loads that exceed that budget fail
early with a specific `det_bin` recommendation instead of risking a frozen
laptop:

```python
data = load("scan_master.h5", backend="mps", det_bin=4)
```

For the smallest browse workflows, combine detector binning with compact dtype:

```python
result = load("scan_master.h5", backend="mps", det_bin=8, dtype="u8")
Show4DSTEM(result)
```

That path is intended for screening, layout, and export-to-browser review. Use
uint16/full precision when detector counts are part of the scientific claim and
the GPU memory budget is clean.

The same is one shell command - see [the CLI](../cli): `quantem show4dstem
scan_master.h5 --bin 8`.

## Multi-File / 5D Browsing

Use one public entry point for a time-series or tilt-series stack:

```python
from quantem.gpu.io import load
from quantem.widget import Show4DSTEM

masters = [
    "/data/session/file_001_master.h5",
    "/data/session/file_002_master.h5",
    "/data/session/file_003_master.h5",
]

r = load(masters, det_bin=1, dtype="u8")
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

For full-detector browse-scale multi-GPU sessions, use the same public dtype
vocabulary without detector binning when the memory budget allows:

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
    det_bin=1,
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

Use `det_bin=4, dtype="u8"` only for an explicit reduced preview on constrained
hardware. Use no-bin `uint16` for count-preserving science, or no-bin
`dtype="u8"` for compact full-detector browsing when clipping has been accepted
or audited.

## Scan-region loading for ROI workflows

Use `load(..., scan_region=...)` when a reconstruction or denoise workflow
needs only a rectangular scan patch, not the full scan plane. This is different
from loading the full frame and slicing afterward: the loader reads only the
selected HDF5 detector-frame chunks, decompresses them on CUDA, and returns a
local patch.

```python
from quantem.gpu.io import load

patch = load(
    "scan_master.h5",
    scan_region=(160, 293, 234, 367),  # row_start, row_stop, col_start, col_stop
).data

print(patch.shape)
# (133, 133, 192, 192)
```

The returned `LoadResult.data` shape is
`(region_rows, region_cols, detector_rows, detector_cols)`. Metadata records
both the original scan grid and the loaded patch:

```python
result = load("scan_master.h5", scan_region=(160, 293, 234, 367))
print(result.metadata["full_scan_shape"])  # e.g. (512, 512)
print(result.metadata["scan_region"])
```

For drift-corrected time-series work, compute the source scan box from the
shared specimen ROI plus a small halo, load that patch, then apply your existing
subpixel sampler in local patch coordinates. Do not save the sampled patch as a
new raw acquisition; drift is scan-position metadata, and detector counts stay
physically unchanged.

Measured on a native-detector 5D-STEM ROI loader timing check
(`10 x 128 x 128 x 192 x 192`, CUDA, no detector binning):

| Path | Loader wall time | Max loaded CuPy buffer |
|---|---:|---:|
| `load()` full frame, then crop | `9.66 s` | `18.0 GiB` |
| `load(..., scan_region=...)` patch, then crop | `2.44 s` | `1.215 GiB` |

The patch path is CUDA-only today and targets chunked 4D-STEM masters with one
detector frame per HDF5 chunk. Use `load()` for full-field browsing and for
Apple Metal/MPS until the region loader is ported there.
