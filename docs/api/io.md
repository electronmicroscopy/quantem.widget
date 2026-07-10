# Load and I/O — FAQ

Every question here comes from a real user session. Pick the one that matches
what you're trying to do; each answer is a copy-pasteable snippet.

For a beginner-friendly walkthrough of `uint8`/`uint16`, memory estimates,
CUDA GPU selection, and cleanup, start with {doc}`IO/GPU <../tutorials/io_gpu>`.

For the full function reference, see `load` and the autodocs at the
bottom of this page.

---

## First-time walkthrough (no data of your own required)

If you don't have a 4D-STEM master.h5 on disk yet, use one of the reference
datasets on Hugging Face — the whole flow is four lines:

```python
from quantem.widget import ShowFolder, load, Show4DSTEM
from quantem.widget.io import list_datasets, download, discover_masters

# 1. See what's available (returns names like '4dstem/gold_512' with prefix)
list_datasets()

# 2. Download by SHORT name (drop the '4dstem/' prefix). Returns a Path
#    under ~/.cache/huggingface/... — cached after first call.
path = download("gold_512")

# 3. Look before you load — cached thumbnails, metadata, selection
ShowFolder(path)

# 4. Discover the master.h5 files + load the first + open the viewer
masters = discover_masters(path)     # sorted list of Path
data = load(masters[0], det_bin=4)   # fast browse preset (see next section)
Show4DSTEM(data)
```

**Gotcha**: `list_datasets()` returns `4dstem/gold_512` (with prefix) but
`download()` takes the SHORT name `gold_512` (no prefix). This is a quirk of
the underlying `quantem.data` API and will be aligned in a future release —
for now, drop the prefix.

The gold reference scans (`gold_512`, `gold_128_npy_bin8`, etc.) are 1-5 GB
compressed and load in seconds on any modern GPU or Mac. Once this works
end-to-end, swap `download(...)` for `Path("/data/session")` and everything
downstream is identical.

## `ShowFolder(path)` — what's in this folder before I load anything?

`ShowFolder` is the folder-level entry point for real microscope sessions. It
builds cached thumbnails for image files, shows acquisition metadata, lets you
star files for downstream analysis, saves that selection, and can open starred
images immediately as Show2D or Show3D from the embedded selection panel.

```python
from quantem.widget import ShowFolder

folder = ShowFolder("/data/session")
folder.paths("image")  # selected files after you star panels
```

For 4D-STEM master files, use `discover_masters` to return sorted master paths
for a scripted load. Then inspect one file with `get_metadata` or load it with
the memory-reduction options shown below.

Prefer `discover_masters` when you just want the sorted paths back for a
scripted load:

```python
from quantem.widget.io import discover_masters

masters = discover_masters("/data/session")               # all
masters = discover_masters("/data/session", scan_shape=(512, 512))  # filter by scan size
```

Prefer `get_metadata` when you want raw HDF5 attributes of ONE file without
loading it — returns a dict of HDF5 tree paths plus the widget-friendly keys
`scan_shape`, `detector_shape`, `n_frames`, `dwell_time_us`, `saturation`,
`detector_name`:

```python
from quantem.widget.io import get_metadata

meta = get_metadata("/data/session/scan_00_master.h5")
print(meta["scan_shape"], meta["detector_shape"])
# e.g. (512, 512) (192, 192)
```

## Lightweight visual thumbnails

Use `quantem.widget.render.thumbnail` when you need compact visual previews for
folder reports, static dashboards, or quick review pages. WebP is the default
preview format because it gives small files for noisy microscopy images while
still showing the structure a human needs for browsing: particles, scan
artifacts, contrast, FFT-like texture, and bad frames. This matters when a
folder contains hundreds of images, when an HTML report is opened on a laptop or
phone, or when a CI/dashboard page needs many thumbnails without becoming a
large artifact.

Use WebP thumbnails for:

- folder-browser previews and cached visual review pages
- maintainer smoke dashboards and HTML reports
- static report previews where file size matters more than exact pixel values
- quick human decisions such as "which file should I open next?"

Do not use WebP thumbnails for scientific data storage, measurements,
publication figures, exact widget state, or HTML exports that need interaction.
WebP is a visual preview and may be lossy. Keep scientific arrays in array/HDF5
formats when values need to be reused.

`q85` means "quality 85" for a lossy image encoder. Higher values keep more
visual detail and make larger files; lower values make smaller files and can
show compression artifacts. It is only a preview setting, not a scientific data
type or measurement setting.

Use this policy when choosing a widget output image format:

| Surface | Preferred format | Why |
|---|---|---|
| Saved-notebook fallback for `Show2D` / `Show3D` | JPEG preview generated from the widget render | Very portable across JupyterLab, VS Code, Colab, GitHub previews, and nbconvert; much smaller than PNG for noisy microscopy images |
| Folder browser, ShowFolder reports, smoke dashboards | WebP thumbnail, usually quality 85 | Smallest practical preview for pages with many images |
| Publication-style static output from `save_image(...)` | PNG, PDF, or TIFF | Stable, lossless or publication-friendly output |
| Interactive sharing | HTML export | Keeps controls live without a Python kernel |
| Exact analysis data or reproducible widget state | Array/HDF5 data plus JSON view state, or `save_state=True` only for small widgets | Preserves values and interactivity instead of storing a lossy preview |

The default saved-notebook path should stay conservative: keep notebooks small
by omitting heavy widget buffers, but use a broadly supported JPEG preview so
the output remains visible when someone opens the notebook without rerunning
the kernel. If a local workflow values smaller files more than maximum
notebook-tool compatibility, choose WebP explicitly:

```python
Show2D(image, notebook_preview_format="webp", notebook_preview_quality=85)
Show3D(stack, notebook_preview_format="webp", notebook_preview_quality=85)
```

```python
from quantem.widget.render import save_thumbnail, thumbnail_webp

webp_bytes = thumbnail_webp(image, size=256, cmap="inferno")
save_thumbnail(image, "preview.webp", size=256, cmap="inferno")
```

Use `save_image(...)` on a widget when you want a publication-style figure,
`export_html(...)` when you want the interactive widget, and `io.save(...)` or a
domain file format when you want data that another analysis step will consume.

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

Result shape: `(n_files, scan_row, scan_col, det_row, det_col)`. Filenames become
slider labels.

On a CUDA workstation with multiple GPUs, keep large browse series sharded
instead of forcing every master into one allocation:

```python
data = load(masters, det_bin=1, dtype="u8", devices=[0, 1])
Show4DSTEM(data)
```

`dtype="u8"` is the fast browse contract. It decodes directly into uint8 before
stacking or sharding, so the loader does not build a full uint16 stack first.
Values above 255 clip, so use uint16/no-bin when detector counts are the
scientific result.

The sharded path is disk-aware. If masters live on independent NVMe devices,
`load(..., devices=[0, 1])` interleaves files by physical disk and GPU. If every
file is on one disk, sharding still increases GPU capacity and keeps flipping
bounded, but cold loading stays limited by that disk.

## I want a viewer to follow a growing folder.

Show2D, Show3D, and Show4DSTEM share one folder-watching lifecycle. Watching is
enabled by default, and every viewer can be paused, polled, resumed, and closed
without constructing a replacement widget.

| Viewer | What a new file becomes | Data and memory behavior |
|---|---|---|
| `Show2D.from_folder(...)` | One new gallery panel | Reads only the new full-resolution source file; preserves the existing widget and panel state |
| `Show3D.from_folder(...)` | One new frame in a single displayed stack | Reads only the new full-resolution source file; preserves the existing widget and frame state |
| `Show4DSTEM.from_folder(...)` | One cold lazy 4D-STEM dataset | Loads raw data only when visible; a bounded GPU cache evicts older raw pages as needed |

```python
from quantem.widget import Show2D, Show3D, Show4DSTEM

images = Show2D.from_folder("/data/session/images", pattern="*.tif")
movie = Show3D.from_folder("/data/session/frames", pattern="frame_*.tif")
scans = Show4DSTEM.from_folder(
    "/data/session/4dstem",
    pattern="*_master.h5",
    det_bin=1,
    page_size=5,
)
```

The common lifecycle is:

```python
added = images.poll_folder()          # one immediate scan
images.stop_folder_watch()            # idempotent pause
images.watch_folder(interval=1.0)     # resume
images.close()                        # stop background work and close the comm
```

The same methods apply to all three viewers. `poll_folder()` returns the
zero-based indices appended by that scan. Pass `watch=False` to any
`from_folder(...)` call for deterministic manual polling. Watching is
append-only: known files are not duplicated, transiently incomplete files wait
for a later poll, and deletions do not remove already displayed scientific data.

These APIs load source data for scientific display. `ShowFolder` serves a
different purpose: it uses cached WebP thumbnails and metadata so a large
session can be browsed and selected quickly. Thumbnail pixels must never be
substituted for the full-resolution arrays opened by Show2D or Show3D, or for
the lazy raw masters opened by Show4DSTEM.

## I want to load every master file in a folder.

Use `Show4DSTEM.from_folder(...)` when the folder can grow or when you want a
GPU-resident cache instead of loading every master immediately:

```python
from quantem.widget import Show4DSTEM

w = Show4DSTEM.from_folder(
    "/data/session",
    backend="cuda",
    gpus=[0, 1],
    det_bin=1,
    dtype="auto",       # keep real counts; use "u8" only for clipped fast preview
    page_budget="auto",
    view_mode="multiple",
    compare_cols=3,
)
```

CUDA multi-GPU multiple views preload only the initial visible page with the
optimized multi-file loader. Other masters stay as lazy slots, and new ready
masters append through `poll_folder()` / `watch_folder()` without rebuilding the
widget. Watching starts by default. Raw masters enter as cold lazy datasets;
they use GPU memory only when selected or included in a visible page, and older
raw pages are evicted when the resident budget requires it.

Use explicit discovery plus `load(...)` when the file list is fixed and you want
to control exactly what enters the stack:

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
from quantem.widget import ShowFolder

folder = ShowFolder("/data/session")  # thumbnails, metadata, selection, cache
```

Use the embedded selection panel to open starred images as Show2D or Show3D.
For 4D-STEM master files, pair this with `discover_masters` and `get_metadata`
before calling `load`.

## How do I inspect a single master's calibration + metadata without loading it?

```python
from quantem.widget.io import get_metadata

meta = get_metadata("scan_master.h5")
print(meta)   # voltage_kV, semiangle_mrad, scan_sampling_A, det_shape, ...
```

## I have HAADF or a 2D image (Velox EMD, TIFF, PNG). How do I load that?

```python
from quantem.widget import Show2D, read_image

img = read_image("haadf.emd")   # Dataset2d with sampling + units
Show2D(img)
```

For a stack (multi-frame TIFF, sequence of PNGs):

```python
from quantem.widget import Show3D, read_image_stack

stack = read_image_stack("frames", pattern="frame_*.png")
Show3D(stack)
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

Detector files are often integers, not floating-point images. If you are new to
dtype choices: `uint16` (`u16`) stores exact raw detector counts from 0 to
65535 in 2 bytes per pixel. `uint8` (`u8`) stores 0 to 255 in 1 byte per pixel,
so it is smaller and faster for display, but it can saturate real count data.
Use `uint16` for scientific loading and reconstruction; use `uint8` only for an
explicit preview or browsing copy.

Which mode + your GPU / Mac tier at a glance:

| your box | full u16 no-bin | `det_bin=2` u16 | `det_bin=4` u8 |
|---|:---:|:---:|:---:|
| **NVIDIA 24 GB** (RTX 4090 / A6000) | browse ✓ · recon tight | recon ✓ | ✓ |
| **NVIDIA 48 GB** (L40S / A100) | browse + recon ✓ | ✓ | ✓ |
| **NVIDIA 96 GB** (Blackwell) | multi-scan + recon ✓ | ✓ | ✓ |
| **Mac 48+ GB** (M-series Max/Ultra) | browse + recon ✓ | ✓ | ✓ |
| **Mac 24-36 GB** (M-series Pro) | browse ✓ via raw-Metal chunked | ✓ | ✓ |
| **Mac 16-18 GB** (M-series base) | bin at load | ✓ | ✓ |

## How do I choose a specific NVIDIA GPU?

Set `CUDA_VISIBLE_DEVICES` before launching Jupyter. This controls which
NVIDIA GPU the Python kernel can see:

```bash
CUDA_VISIBLE_DEVICES=0 jupyter lab --no-browser --ip=0.0.0.0
```

Use `1`, `2`, etc. for another physical GPU. This is a CUDA/NVIDIA control; it
is not used for Apple Silicon or CPU-only machines.

```bash
CUDA_VISIBLE_DEVICES=1 jupyter lab --no-browser --ip=0.0.0.0
```

Inside the notebook:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.mem_get_info())
```

To release memory from the current Python kernel:

```python
del data

import gc
import torch

gc.collect()
torch.cuda.empty_cache()
```

If memory is still occupied, another object or another Jupyter kernel still
owns it. Shut down that kernel from JupyterLab or stop the Python process.

## Function reference

```{eval-rst}
.. autofunction:: quantem.widget.io.hdf5.load
```

### Discover + inspect

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
