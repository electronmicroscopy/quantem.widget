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
from quantem.gpu.io import load
from quantem.widget import ShowFolder, Show4DSTEM
from quantem.gpu.io import discover
from quantem.widget.io import list_datasets, download

# 1. See what's available (returns names like '4dstem/gold_512' with prefix)
list_datasets()

# 2. Download by SHORT name (drop the '4dstem/' prefix). Returns a Path
#    under ~/.cache/huggingface/... — cached after first call.
path = download("gold_512")

# 3. Look before you load — cached thumbnails, metadata, selection
ShowFolder(path)

# 4. Discover the master.h5 files + load the first + open the viewer
masters = discover(path)             # sorted list of Path
result = load(masters[0])            # native detector sampling when it fits
Show4DSTEM(result)
```

**Gotcha**: `list_datasets()` returns `4dstem/gold_512` (with prefix) but
`download()` takes the SHORT name `gold_512` (no prefix). This is a quirk of
the underlying `quantem.data` API and will be aligned in a future release —
for now, drop the prefix.

The gold reference scans (`gold_512` and friends) are 1-5 GB
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

For 4D-STEM master files, use `quantem.gpu.io.discover` to return sorted master paths
for a scripted load. Then inspect one file with `quantem.gpu.io.inspect` or load it with
the memory-reduction options shown below.

Prefer `discover` when you just want the sorted paths back for a
scripted load:

```python
from quantem.gpu.io import discover

masters = discover("/data/session")               # all
masters = discover("/data/session", scan_shape=(512, 512))  # filter by scan size
```

Use `inspect` when you want readiness, shape, dtype, and calibration metadata
without loading detector frames:

```python
from quantem.gpu.io import inspect

report = inspect("/data/session/scan_00_master.h5")
print(report.scan_shape, report.detector_shape, report.dtype)
# e.g. (512, 512) (192, 192)
```

## How do I load only a scan ROI without loading the full frame first?

Use `load(..., scan_region=...)` for reconstruction or denoise workflows that
need a scan patch plus halo instead of the full scan plane. It reads only the
selected HDF5 detector-frame chunks and returns a local CuPy patch:

```python
from quantem.gpu.io import load

result = load(
    "/data/session/scan_00_master.h5",
    scan_region=(160, 293, 234, 367),
)
patch = result.data
print(patch.shape)  # (133, 133, 192, 192)
```

This is for analysis pipelines, not first-pass full-field browsing. For a
drift-corrected time series, derive `scan_region` from the shared specimen ROI,
the frame shift, and a small scan halo, then sample the final ROI from the
local patch. The detector counts remain raw; drift stays as scan-position
metadata.

On a native-detector ROI loader timing check, loading ten full frames
before cropping took `9.66 s` and used an `18.0 GiB` temporary per frame.
Loading the needed `133 x 133` patch took `2.44 s` and used a `1.215 GiB`
temporary per frame. The current region loader is CUDA-only; use
`load()` for Apple Metal/MPS until the region path is ported there.

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
from quantem.gpu.io import load
from quantem.widget import Show4DSTEM

result = load("scan_master.h5")
Show4DSTEM(result)
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
from quantem.gpu.io import load
from quantem.widget import Show4DSTEM

result = load("scan_master.h5")
Show4DSTEM(result)
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
result = load("scan_master.h5")   # dtype defaults to uint16, no bin
Show4DSTEM(result)                # ~21 GB VRAM peak
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

## I want a smaller preview and do not need full detector detail.

Bin deliberately and drop to uint8. This is useful for scrolling through a
session on constrained hardware; it is not a reconstruction or count-preserving
path.

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
data = load(masters, det_bin=1, dtype="u8")
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
scientific result. Add `det_bin=2` or `4` only when you intentionally want a
detector-reduced preview.

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
| `Show2D.from_folder(...)` | One new gallery panel; visible pages default to 20 panels | Reads only the new full-resolution source file; preserves the existing widget and per-file panel state |
| `Show3D.from_folder(...)` | One new frame in a single unpaged stack | Reads only the new full-resolution source file; preserves the existing widget and frame state |
| `Show4DSTEM.from_folder(...)` | One cold lazy 4D-STEM dataset | Loads raw data only when visible; a bounded GPU cache evicts older raw pages as needed |

```python
from quantem.widget import Show2D, Show3D, Show4DSTEM

images = Show2D.from_folder(
    "/data/session/images",
    pattern="*.tif",
    page_size=20,  # another positive integer, or None for one gallery
)
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

Show2D folder pages are sequential independent files, not the repeated-slot
comparison pages accepted by direct `Show2D(...)`. Show3D folder files never
cross a page threshold: they always extend one frame axis, even when the folder
contains hundreds of frames.

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
from quantem.gpu.io import discover, load
from quantem.widget import Show4DSTEM

masters = discover("/data/session")   # sorted, filters to *_master.h5
data = load(masters, det_bin=1)
Show4DSTEM(data)
```

`discover` also accepts a `scan_shape=(512, 512)` filter to keep only
matching acquisitions when a folder mixes scan sizes. Add `det_bin=2` or `4`
only when you intentionally want a detector-reduced preview.

## Before loading anything, how do I check what's in a folder?

```python
from quantem.widget import ShowFolder

folder = ShowFolder("/data/session")  # thumbnails, metadata, selection, cache
```

Use the embedded selection panel to open starred images as Show2D or Show3D.
For 4D-STEM master files, pair this with `discover` and `inspect`
before calling `load`.

## How do I inspect a single master's calibration + metadata without loading it?

```python
from quantem.gpu.io import inspect

report = inspect("scan_master.h5")
print(report.metadata)  # voltage, semiangle, sampling, and source metadata
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
from quantem.gpu.io import save

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
is not used for Apple Silicon machines.

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

## I have data others should use. How do I upload it?

The shared Hugging Face dataset repo
([bobleesj/quantem-data](https://huggingface.co/datasets/bobleesj/quantem-data),
MIT license) is the one place tutorial and reference data lives. Upload and
download commands, including Hugging Face pull requests, are on that dataset
card. This section is the `quantem.widget.io` helper reference for
maintainers who already have write access. The upload steps are:

1. **Install the hub extra and log in once.** Uploading needs a Hugging Face
   account and a write token from
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens):

   ```bash
   pip install "quantem.widget[hub]"
   hf auth login   # paste a Write token; answer n to git credential
   ```

   Token steps for a Community pull request (no write access) are on the
   dataset card. A Read token cannot open a PR. Ignore any hint to run
   `git config --global credential.helper store`.

2. **Upload with the bucket + sidecar convention.** The repo has two trees:

   - `widget-tutorials/<widget>/<dataset>/<size>/` — the baseline tutorial
     bundles behind `quantem.widget.datasets` (sizes `small`/`medium`/
     `large`/`full`). Contributors add these with a Hugging Face pull
     request, not `quantem.widget.io.upload`. See
     [Contribute tutorial data](../tutorials/contribute_data.md).
     Datasets shared by several widgets (the gold HAADF feeds both Show2D
     and Show3D) live under `widget-tutorials/shared/`.
   - `4dstem/` and `haadf/` — full-size originals for power users. A folder
     of Arina `*_master.h5` files goes under `4dstem/`, a single image file
     under `haadf/` (those are also the defaults for a directory vs a file).

   For `4dstem/` and `haadf/`, pass `meta=` so the helper writes a
   `quantem_meta.json` sidecar. Tutorial loaders read `meta.json` from the
   staged `widget-tutorials/` folder instead:

   ```python
   from quantem.widget.io import upload

   upload(
       "/data/session/gold_512/",          # folder -> 4dstem/gold_512/*
       name="gold_512",
       folder="4dstem",
       meta={"sampling": [0.5, 0.5], "units": ["A", "A"],
             "voltage_kV": 300, "probe_mrad": 30, "camera_length_mm": 91},
   )
   ```

   Write access to `bobleesj/quantem-data` is for maintainers. Contributors
   open a Hugging Face pull request on the dataset card, then a GitHub pull
   request for the named loader
   ([Contribute tutorial data](../tutorials/contribute_data.md)).

3. **Verify like a user would.** List, download to a fresh path, and open it
   in the widget before announcing the dataset:

   ```python
   from quantem.widget.io import list_datasets, download, status

   list_datasets()            # '4dstem/gold_512' should appear
   folder = download("gold_512")
   status()                   # repo-wide file/size snapshot
   ```

Remove a mistake with `delete("name")` — it deletes every file under the
dataset's folder, so double-check `list_datasets()` first. Uploads to the
shared repo are published under its MIT license; only upload data you have
the right to share.

## Function reference

```{eval-rst}
.. autofunction:: quantem.gpu.io.load
```

### Discover + inspect

```{eval-rst}
.. autofunction:: quantem.gpu.io.discover
```
```{eval-rst}
.. autofunction:: quantem.gpu.io.inspect
```

### Images (2D / 3D)

```{eval-rst}
.. autofunction:: quantem.widget.io.image.read_image
```
```{eval-rst}
.. autofunction:: quantem.widget.io.image.read_image_stack
```

### Hugging Face datasets

```{eval-rst}
.. autofunction:: quantem.widget.io.hub.list_datasets
```
```{eval-rst}
.. autofunction:: quantem.widget.io.hub.download
```
```{eval-rst}
.. autofunction:: quantem.widget.io.hub.upload
```
```{eval-rst}
.. autofunction:: quantem.widget.io.hub.status
```
```{eval-rst}
.. autofunction:: quantem.widget.io.hub.delete
```

### Save

```{eval-rst}
.. autofunction:: quantem.gpu.io.save
```
