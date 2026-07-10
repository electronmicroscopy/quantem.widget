# IO/GPU

Use this page when you have microscope data on disk and want to open it. The
basic workflow is:

```python
from quantem.widget import ShowFolder, load, Show4DSTEM
from quantem.widget.io import discover_masters

folder = ShowFolder("/data/session")  # browse thumbnails, cache, star files
masters = discover_masters("/data/session")
data = load(masters[0])               # full precision when it fits
Show4DSTEM(data)
```

If the data is too large, reduce it deliberately:

```python
data = load("scan_master.h5", det_bin=2)              # smaller detector, still uint16
data = load("scan_master.h5", det_bin=4, dtype="u8")  # small browse copy
```

The loader prints useful progress by default. Keep those lines visible while you
are learning a new dataset; they tell you how much memory was allocated, whether
the data was narrowed safely, and whether a preview dtype clipped counts. For
what the dtypes mean, how much memory each load mode takes, GPU selection, and
freeing memory afterward, see {doc}`memory_management`.

## Inspect before loading

Use `ShowFolder` to inspect microscopy folders before allocating memory. It
builds cached thumbnails, shows image metadata, lets you star files for
downstream analysis, and can open the starred images immediately as Show2D or
Show3D from the embedded selection panel.

```python
from quantem.widget import ShowFolder

folder = ShowFolder("/data/session")
folder.paths("image")  # selected files after you star panels
```

For 4D-STEM master files, use `discover_masters` to collect the candidate HDF5
masters before loading:

```python
from quantem.widget.io import discover_masters

masters = discover_masters("/data/session", scan_shape=(512, 512))
```

## Two different IO paths

Use the 4D-STEM loader for detector stacks, and use the image readers for
ordinary survey images. They are intentionally different:

| data on disk | reader | what is accelerated |
|---|---|---|
| 4D-STEM HDF5 master files | `load(...)` | optimized detector decompression, optional detector binning, GPU-resident arrays when CUDA or Metal is available |
| HAADF/ADF/BF survey images, PNG, TIFF, JPEG, GIF, DM, NPY | `read_image(...)`, `read_images(...)` | CPU file decoding through the format library, optional threaded batch reads, then explicit GPU transfer if you need Torch/CuPy |
| folders of same-size image frames | `read_image_stack(...)` | threaded CPU decoding, preallocated stack, then optional GPU transfer |

PNG, TIFF, and EMD survey-image decoding is not a WebGPU operation and is not
decoded by the browser. It happens in Python. That is still the right path for
single images and 50-60 survey files because the files are independent and can
be read in parallel. Move the resulting arrays to the GPU only when you are
about to compute on them (see {doc}`memory_management`).

## Survey images and frame folders

### Open a HAADF, EMD survey image, PNG, or TIFF

Not every dataset is a 4D-STEM master file. For a single 2D microscope image,
use `read_image`. It returns a calibrated `Dataset2d` when the file carries
metadata, so `Show2D` can draw the scale bar automatically.

```python
from quantem.widget import Show2D, read_image

haadf = read_image("haadf.emd")       # Velox EMD HAADF / survey image
Show2D(haadf)
```

The same reader works for common image files:

```python
from quantem.widget import Show2D, read_image

image = read_image("overview.tif")    # also .png, .jpg, .bmp, .gif, .dm3, .dm4, .npy
Show2D(image)
```

Use this path for HAADF, ADF, BF, overview images, diffraction snapshots saved
as images, and quick previews. It avoids the 4D-STEM decompression path
entirely, so loading is immediate for normal image sizes.

### Open many EMD, PNG, TIFF, or DM survey images

For a folder of independent microscope images, use `read_images`. This keeps the
simple one-image API but reads many files concurrently when you ask for workers.

```python
from quantem.widget import Show2D, read_images

images = read_images("survey_images", workers=8)
Show2D(images)
```

This is the best fit for 50-60 HAADF/ADF/BF survey images that may be different
sizes or formats. If every frame is the same size and you want a time/depth
slider, use `read_image_stack` instead.

### Open a folder of PNG or TIFF frames

For an in-situ sequence, tilt series, denoising sweep, or any folder of
same-size frames, use `read_image_stack`. It decodes PNG/TIFF frames in parallel
and returns a `Dataset3d` for `Show3D`.

```python
from quantem.widget import Show3D, read_image_stack

stack = read_image_stack("frames", file_type="tif", workers=8)
Show3D(stack)
```

You can also use a glob pattern:

```python
stack = read_image_stack("frames", pattern="frame_*.png", workers=8)
```

Keep the files in their original integer dtype on disk. The reader converts the
viewer stack to `float32`, which is the right display/processing dtype for most
image stacks and avoids accidental `float64` memory growth.

### Profile your own image folder

Before changing file formats, measure the folder you actually have:

```python
import time
from pathlib import Path
from quantem.widget import read_images

folder = Path("survey_images")

t0 = time.perf_counter()
images = read_images(folder, workers=8)
dt = time.perf_counter() - t0

pixels = sum(ds.array.size for ds in images)
print(f"{len(images)} images")
print(f"{pixels / 1e6:.0f} megapixels")
print(f"{dt:.2f} s")
print(f"{pixels / dt / 1e6:.0f} megapixels/s")
```

Use this to compare `workers=1` and `workers=8`. On many image folders the
parallel path is faster; on slow network storage, the disk can become the limit.

Example local profile with 24 synthetic `4096 x 4096` `uint16` survey frames
(`0.81 GB` raw pixels):

| folder | `workers=1` | `workers=8` | note |
|---|---:|---:|---|
| uncompressed TIFF | 0.40 s | 0.16 s | fast image frames |
| PNG | 3.42 s | 0.74 s | slower because PNG decompression is real work |
| Velox-style EMD survey images | 0.22 s | 0.12 s | fast when image data is stored directly |

For 50-60 full 4k survey images, prefer EMD or TIFF when you control the export
format. PNG is fine for screenshots and compact sharing, but it is not the
fastest format for high-throughput analysis.

### Open a diffraction image or diffraction stack

Use `ShowDiffraction` for a single diffraction pattern or a small stack of
patterns:

```python
from quantem.widget import ShowDiffraction, read_image

dp = read_image("diffraction.tif")
ShowDiffraction(dp.array)
```

For a folder of diffraction frames:

```python
from quantem.widget import ShowDiffraction, read_image_stack

dp_stack = read_image_stack("diffraction_frames", file_type="tif")
ShowDiffraction(dp_stack.array)
```

This is separate from `Show4DSTEM`: use `ShowDiffraction` when the data is just
one detector image or a detector-image stack, and use `Show4DSTEM(load(...))`
when you have a scan grid with a diffraction pattern at every probe position.

## 4D-STEM detector stacks

### Open one scan at full precision

Use this when the scan fits your GPU and you want the most faithful interactive
view:

```python
from quantem.widget import load, Show4DSTEM

data = load("scan_001_master.h5", verbose=True)
print(data.data.shape, data.data.dtype, f"{data.data.nbytes / 1e9:.1f} GB")
Show4DSTEM(data)
```

To check VRAM before and after the load, and to free it when you are done, see
{doc}`memory_management`.

### Browse a large folder quickly

Use this when you have a whole microscope session and want to find the good
fields of view first:

```python
from quantem.widget import load, Show4DSTEM
from quantem.widget.io import discover_masters

masters = discover_masters("/data/session", scan_shape=(512, 512))
data = load(masters, det_bin=4, dtype="u8", verbose=True)
Show4DSTEM(data)
```

This keeps all scans behind one dataset slider. It is a browsing copy, not the
final reconstruction dataset.

Typical output:

```text
  Loaded 262,144 frames (0.6 GB) in 4.81s (4.0 GB/s)
  Loaded in uint8 for browsing - using 0.6 GB, half of uint16
  (decoded straight to uint8, so peak memory stayed low). Reconstruction uses raw uint16.
```

If this path prints a saturation warning, the preview still works for finding
fields of view, but it is not count-exact.

### Load several scans without throwing away detector counts

Use detector binning but keep `uint16` when counts still matter:

```python
data = load(masters, det_bin=2, dtype="u16")
Show4DSTEM(data)
```

This is a good middle ground for a 24 GB GPU: much smaller than full detector,
but still count-preserving.

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

- {doc}`memory_management`
- {doc}`../api/io`
- {doc}`show4dstem`
- {doc}`widget_export`
