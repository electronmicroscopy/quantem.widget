# Load and I/O — FAQ

Every question here comes from a real user session. Pick the one that matches
what you're trying to do; each answer is a copy-pasteable snippet.

For the full function reference, see [`load`](load) and the autodocs at the
bottom of this page.

---

## I'm on a Linux workstation with an NVIDIA RTX GPU. How do I load a scan?

```python
from quantem.widget import load, Show4DSTEM

data = load("scan_master.h5")
Show4DSTEM(data)
```

`load` auto-detects CUDA and decompresses straight onto the GPU (zero-copy
`cupy` → torch via dlpack). No flag needed. Works on every common workstation
GPU: RTX PRO 6000 Blackwell (96 GB), L40S / A100 (48 GB), RTX 4090 / A6000
(24 GB), and anything else with a working cupy install.

**Rough tier guidance for a 512×512×192×192 scan (~19 GB raw uint16):**

| GPU tier | Full-res u16 no-bin | Best default |
|---|---|---|
| **96 GB** (Blackwell) | fits everything, 3x scans in flight | `load(path)` |
| **48 GB** (L40S / A100) | fits with room for reconstruction | `load(path)` |
| **24 GB** (RTX 4090 / A6000) | fits browse (~21 GB peak) but tight for recon | `load(path)` (browse) or `load(path, det_bin=2)` (recon) |
| **16 GB** or less | bin at load | `load(path, det_bin=4, dtype="u8")` |

## I'm on a MacBook (Apple Silicon). How do I load a scan?

Same one-liner as CUDA:

```python
from quantem.widget import load, Show4DSTEM

data = load("scan_master.h5")
Show4DSTEM(data)
```

`load` auto-detects Apple Metal (MPS) and uses a zero-copy **raw-Metal**
chunked-frames path. Unified memory means "VRAM" = "RAM" — the same 24 GB
covers both. So a 24 GB M-series MacBook has to share load footprint with
macOS + browser + everything else running.

**Rough tier guidance for Mac unified memory:**

| MacBook Pro (unified) | Full-res u16 no-bin | Best default |
|---|---|---|
| **48-128 GB** (M2/M3/M4 Max, M3/M4 Ultra) | fits full-res comfortably | `load(path)` |
| **24-36 GB** (M-series Pro) | fits browse via raw-Metal chunked path | `load(path)` (browse) or `load(path, det_bin=2)` |
| **16-18 GB** (M-series base) | bin at load | `load(path, det_bin=4, dtype="u8")` |

The raw-Metal path streams frames from a chunked buffer rather than requiring
the whole 4D stack in one contiguous allocation, so a 24 GB Mac can browse
19 GB u16 no-bin without OOM even though the block wouldn't fit as a single
torch tensor on MPS.

For multi-scan on Mac, use `load([m1, m2, m3])` — dataset 0 shows in ~2 s,
and datasets 1..N-1 decode in a background worker behind the `Dataset` slider
(so a 5-file series streams in without freezing the UI).

## My GPU is 24 GB (RTX 4090 / A6000) and the scan is 512×512×192×192 (~19 GB uint16). Does it fit?

Yes, after the 2026-07-02 `mean_dp` fix. Full-res uint16 no-bin peak = ~21 GB
(data + widget). Fits 24 GB with ~2.5 GB headroom.

```python
data = load("scan_master.h5")   # dtype defaults to uint16, no bin
Show4DSTEM(data)                # ~21 GB VRAM peak
```

If you need more headroom for downstream compute (reconstruction, SSB), bin
the detector on the way in:

```python
data = load("scan_master.h5", det_bin=2)   # 512x512x96x96, ~5 GB
Show4DSTEM(data)
```

## My GPU is 48 GB (L40S / A100). Anything I need to know?

No. Load full-res u16 no-bin — plenty of headroom for browse + downstream
reconstruction in one process:

```python
data = load("scan_master.h5")   # ~21 GB peak, ~27 GB free after
Show4DSTEM(data)
```

You can also load 2-3 scans simultaneously for cross-scan comparison
without OOM. For time-series / tilt-series, `load([m1, m2, m3])` in one
call keeps them behind a single `Dataset` slider.

## My GPU is 96 GB (Blackwell). Anything I need to know?

No. Full-res u16 no-bin peaks at ~21 GB per scan — you can hold 3-4 scans
in VRAM at once, or one scan plus a full reconstruction workspace. Same
one-liner:

```python
data = load("scan_master.h5")
Show4DSTEM(data)
```

## I want to browse fast without caring about full detector detail.

Bin harder + drop to uint8. Great for scrolling through a session to find good
scans; not for reconstruction.

```python
data = load("scan_master.h5", det_bin=4, dtype="u8")
Show4DSTEM(data)
```

Resident size drops to roughly 5% of the no-bin uint16 baseline. Peak brightness
below 255 counts is fine (the loader warns if you'd saturate).

## I want to browse many scans as one dataset.

Pass a list. The result stacks them behind a `Dataset` slider inside
`Show4DSTEM`, so scrubbing = switching files:

```python
masters = [
    "/data/session/file_001_master.h5",
    "/data/session/file_002_master.h5",
    "/data/session/file_003_master.h5",
]
data = load(masters, det_bin=4, dtype="u8")
Show4DSTEM(data)
```

Result shape: `(n_files, scan_y, scan_x, det_y, det_x)`. Filenames become
slider labels.

## I want to load every master file in a folder.

```python
from quantem.widget import load, discover_masters, Show4DSTEM

masters = discover_masters("/data/session")   # sorted, filters to *_master.h5
data = load(masters, det_bin=4)
Show4DSTEM(data)
```

`discover_masters` also accepts a `scan_shape=(512, 512)` filter to keep only
matching acquisitions when a folder mixes scan sizes.

## Before loading anything, how do I check what's in a folder?

```python
from quantem.widget import survey

survey("/data/session")   # header-only walk: scan/det shapes + total size
```

Zero pixel reads. Reports each master's shape, dtype, chunks, and file size so
you can plan the budget before allocating a byte.

## How do I inspect a single master's calibration + metadata without loading it?

```python
from quantem.widget.io import get_metadata

meta = get_metadata("scan_master.h5")
print(meta)   # voltage_kV, semiangle_mrad, scan_sampling_A, det_shape, ...
```

## I have HAADF or a 2D image (Velox EMD, TIFF, PNG). How do I load that?

```python
from quantem.widget import read_image, Show2D

img = read_image("haadf.emd")   # Dataset2d with sampling + units
Show2D(img)
```

For a stack (multi-frame TIFF, sequence of PNGs):

```python
from quantem.widget.io import read_image_stack

stack = read_image_stack(["a.png", "b.png", "c.png"])
```

## I want the reference gold or MoS2 dataset from Hugging Face.

```python
from quantem.widget.io import list_datasets, download

list_datasets()                # what's shared
path = download("gold_drift_0deg")   # returns local path
data = load(path)
```

## I want to save a `LoadResult` back to disk (e.g. after binning).

```python
from quantem.widget.io import save

save(data, "binned_out.h5")   # compressed, matches original chunk shape
```

## What's the difference between `det_bin`, `dtype`, and `no bin`?

- `det_bin=1` (default): full-detector resolution. Every diffraction pixel
  preserved. CBED at full angular resolution.
- `det_bin=N > 1`: mean-reduces N×N detector blocks at load. `det_bin=2` on a
  192² detector → 96² output. Faster virtual-image compute; less angular detail.
- `dtype="u16"` (default): raw counts (0-65535). Exact for reconstruction.
- `dtype="u8"`: 0-255. Halves memory. Fine when max counts <255 (loader warns
  if you'd saturate).

## Memory rule of thumb for a 512×512×192×192 scan

| mode | resident VRAM per file |
|---|---:|
| no bin, uint16 | 18-20 GB |
| `det_bin=2`, uint16 | 4.5-5 GB |
| `det_bin=4`, uint16 | 1.1-1.3 GB |
| `det_bin=4`, `dtype="u8"` | ~0.6 GB |

`Show4DSTEM(data)` adds ~2-3 GB overhead (colormap, virtual-image cache, CBED
buffer) on top of the load footprint. Budget accordingly.

Which mode + your GPU / Mac tier at a glance:

| your box | full u16 no-bin | `det_bin=2` u16 | `det_bin=4` u8 |
|---|:---:|:---:|:---:|
| **NVIDIA 24 GB** (RTX 4090 / A6000) | browse ✓ · recon tight | recon ✓ | ✓ |
| **NVIDIA 48 GB** (L40S / A100) | browse + recon ✓ | ✓ | ✓ |
| **NVIDIA 96 GB** (Blackwell) | multi-scan + recon ✓ | ✓ | ✓ |
| **Mac 48+ GB** (M-series Max/Ultra) | browse + recon ✓ | ✓ | ✓ |
| **Mac 24-36 GB** (M-series Pro) | browse ✓ via raw-Metal chunked | ✓ | ✓ |
| **Mac 16-18 GB** (M-series base) | bin at load | ✓ | ✓ |

## Function reference

```{eval-rst}
.. autofunction:: quantem.widget.io.hdf5.load
```

### Discover + inspect

```{eval-rst}
.. autofunction:: quantem.widget.io.survey.survey
```
```{eval-rst}
.. autofunction:: quantem.widget.io.hdf5.discover_masters
```
```{eval-rst}
.. autofunction:: quantem.widget.io.hdf5.get_metadata
```

### Images (2D / 3D)

```{eval-rst}
.. autofunction:: quantem.widget.io.image.read_image
```
```{eval-rst}
.. autofunction:: quantem.widget.io.image.read_image_stack
```

### Detector binning

```{eval-rst}
.. autofunction:: quantem.widget.io.hdf5.bin
```

### Hugging Face datasets

```{eval-rst}
.. autofunction:: quantem.widget.io.hub.list_datasets
```
```{eval-rst}
.. autofunction:: quantem.widget.io.hub.download
```

### Save

```{eval-rst}
.. autofunction:: quantem.widget.io.save.save
```
