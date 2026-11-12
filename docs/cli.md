# Command line

Installing `quantem.widget` adds the `quantem-widget` command. `show2d` and `show3d`
turn image files into one shareable HTML page, `show4dstem` opens 4D-STEM
acquisitions or creates a viewer notebook, and `github` makes a widget notebook
displayable in GitHub's static preview.

```bash
quantem-widget show2d a.dm4 b.tif c.png                   # images            -> one Show2D gallery page
quantem-widget show2d ./survey/ --columns 4               # folder of images  -> one Show2D gallery page
quantem-widget show3d stack.tif                           # (frames, rows, cols) file -> one Show3D page
quantem-widget show3d ./frames/                           # folder of frames  -> one Show3D page
quantem-widget show4dstem ./masters/                      # *_master.h5        -> live Show4DSTEM notebook
quantem-widget show4dstem a_master.h5 b_master.h5         # several masters    -> one comparison viewer
quantem-widget show4dstem ./masters/ --html               # 4D-STEM            -> shareable offline HTML
quantem-widget show4dstem a.qem b.qem --html              # native QEM files   -> browser viewer folder
quantem-widget github tutorial_github.ipynb --no-execute  # optional static copy for GitHub preview
```

Other widgets (ShowPtycho) and whole notebooks export from
Python: `widget.export_html(...)`, or `jupyter nbconvert --to html notebook.ipynb`.

## Subcommands

| Command | Input | Output |
|---|---|---|
| `quantem-widget show2d <files / folder>` | 2D images (`.dm3/.dm4`, `.emd`, `.tif`, `.png`, `.jpg`, `.npy`) | one Show2D HTML page, one panel per image |
| `quantem-widget show3d <stack / files / folder>` | a `.npy` or `.tif` stack, several stacks, 2D frame files, or a folder of frames | one Show3D HTML page |
| `quantem-widget show4dstem <master(s) / folder>` | one or more `*_master.h5` | a live Show4DSTEM notebook (or `--html`) |
| `quantem-widget show4dstem <file.qem ...> --html` | one or more native `.qem` files | a WebGPU folder that reads the encoded acquisitions |
| `quantem-widget github <notebook.ipynb>` | an optional static copy of a notebook | strips widget state and embeds compressed pictures for GitHub's notebook preview |

**HDF5 acquisitions** open a live, kernel-backed notebook by default, at full
detector sampling. Supported acquisitions stay encoded on the GPU when the
accelerated loader is available. `--html --backend webgpu` instead writes a
**browser folder** that reads compressed HDF5 files beside the viewer: drag
detectors, switch BF/ABF/ADF, and pan diffraction without a Python kernel.
Open `index.html` and grant the data folder when prompted, or double-click
`Show4DSTEM.command` to serve that same folder locally. With `--backend auto`,
an HDF5 `--html` export packs the counts in Python into HTML.

Several masters (a folder, or listed explicitly) open as **one comparison
viewer**: a live notebook shows one virtual image per master with a shared
detector ROI, and a WebGPU folder export adds a Dataset slider to flip between scans.
WebGPU HDF5 folders use anonymous local links such as `dataset_00_master.h5`
and `dataset_00_data_*.h5`; rerunning the CLI replaces the generated viewer
folder so stale HTML and metadata do not survive.

Outputs land in `~/Downloads` by default. Use `--save-dir PATH` to choose another
writable folder.

## Show4DSTEM HTML export

Use the CLI when you want a quick browser artifact from raw masters:

```bash
quantem-widget show4dstem scan_001_master.h5 --backend webgpu --html --detector-bin 1
quantem-widget show4dstem ./session_masters --backend webgpu --html --count 7 --detector-bin 1 --save-dir ~/Downloads
quantem-widget show4dstem scan_001_master.h5 scan_002_master.h5 --backend webgpu --html --detector-bin 1
```

`--detector-bin N` replaces each N x N detector block by its mean, rounded to the
nearest count, in an `--html` export packed in Python (`--backend auto`). The
default is `--detector-bin 1`, meaning full detector sampling.
Use a larger value only for an explicit preview, and label that reduction in
the report. The `--backend webgpu` folder reads the source HDF5 chunks in the
browser and requires `--detector-bin 1`; live notebooks always keep full detector
sampling.

Use `--backend webgpu --html --detector-bin 1` when the user wants the full native
detector sampling path without opening Jupyter:

```bash
quantem-widget show4dstem /data/session --backend webgpu --html --count 7 --detector-bin 1 --dtype uint8 --save-dir ~/Downloads
```

That command writes a browser folder with anonymous H5 symlinks plus
`Show4DSTEM.command`, so it does not copy raw data into a giant HTML file.
Double-click `index.html` and grant the export folder when Chrome asks, or use
`Show4DSTEM.command` when you want the local server path. Multi-master WebGPU
exports open as one dataset-slider viewer. For a compact collaborator review,
use the Python `export_kind="report"` path:

```python
from pathlib import Path
from quantem.widget import Show4DSTEM, read_4dstem

masters = sorted(Path("/data/session").glob("*_master.h5"))
viewer = Show4DSTEM(read_4dstem(masters))
viewer.export_html("show4dstem_report.html", export_kind="report", dtype="uint8")
```

See [Show4DSTEM export recipes](tutorials/show4dstem_export) for the decision
table.

### Native QEM acquisitions

Pass existing `.qem` files directly when exporting encoded acquisitions:

```bash
quantem-widget show4dstem a.qem b.qem --html --save-dir ./viewer
```

This route writes a WebGPU viewer folder and links the unchanged inputs under
`qem/`, retaining their input order. It does not unpack and repack the counts in
Python. Serve the exported folder over HTTP to stream the files, or open its
`index.html` and select the linked local files when prompted. Keep the viewer
and its input files together when sharing; local symlinks alone do not copy
their targets to another machine. This route requires `--html`; HDF5 packing
options such as detector binning do not transform native QEM inputs.

## Show2D and Show3D pages

`show2d` and `show3d` read the files with the same readers as
`quantem.widget.io` and write the page the widget's own
`export_html(...)` writes, the same file the toolbar Export button saves. The
page opens with no Python kernel and no network, on a laptop, on a phone, or
inside another HTML page. It renders with WebGPU when the browser has it.

```bash
quantem-widget show2d before.dm4 after.dm4 --labels before after --title "Anneal 300C"
quantem-widget show2d ./survey/ --pattern "*.emd" --columns 4 --size small
quantem-widget show3d drift_series.tif --title "Drift series" --fps 10
quantem-widget show3d ./tilt_series/ --encoding uint8 --cmap gray
quantem-widget show3d run_a.npy run_b.npy --title "Run A vs B"    # two stacks, side by side
```

What each input becomes:

| Input | `show2d` | `show3d` |
|---|---|---|
| several image files | one panel per file, labelled by file name | one frame per file, in the order given (same shape) |
| one folder | one panel per image matching `--pattern`, in natural order (`image_2` before `image_10`) | one frame per image matching `--pattern`, in natural order |
| one `.npy` / `.tif` stack | frame 0, with a printed line saying so; use `show3d` for every frame | every frame |
| several stacks | (use `show3d`) | one panel per stack, scrubbed together |

Files may mix formats (a `.tif` next to a `.png`). Calibrated files (DM,
Velox EMD) carry their pixel size into the scale bar; TIFF, PNG, JPEG and NPY
carry none, so pass `--sampling 0.05 --units nm` for a physical scale bar (it
also overrides a DM or EMD calibration). Without either, the bar is in pixels.

Show2D links contrast across panels by default, which suits images of one
acquisition. For images on different value scales (an 8-bit PNG next to
float data), pass `--no-link-contrast` so each panel gets its own range;
linked, the smaller-valued panels show nearly black.
Color (RGB) images keep their color in both `show2d` and `show3d` pages with
the default full encoding; `--encoding uint8` holds grayscale panels only and
refuses a color image rather than drop its channels.

### Lossless by default

The default `--encoding full` embeds the exact float32 pixels, at full
resolution. Nothing is binned, cropped or filtered unless you ask:

- `--encoding uint8` packs each panel over its own range: about a quarter of
  the size and the same picture after the colormap, but no longer the exact
  values.
- `--image-bin 2` (or 4, 8) mean-bins the embedded pixels, with
  `--encoding uint8`; the scale bar follows. The written line names the packing.
- A page estimated above `--max-mb` (default 80 MB) is refused before anything
  is written, naming the uint8 size; large single files often fail to open from
  disk. `--max-mb none` writes any size.

The file is named from the title (lowercased, every run of other characters
made one underscore), the shape and the packing, for example
`anneal_300c_2x2048x2048_exact.html` or `drift_series_200x512x512_quantized.html`,
in `--save-dir` (default `~/Downloads`).

### Share or embed the page

- **Send the file.** It is one self-contained `.html`; double-click opens it.
- **Embed it in a report or website** with an `<iframe>`. Put the page in the
  same folder as the host page (or give its relative path) and serve the folder
  over http(s), for example `python -m http.server`. Give the frame the height
  the page needs; it scrolls inside a shorter frame. As a guide, a one-panel
  Show3D page with its controls is about 850 px tall and a Show2D gallery about
  550 px per row of medium panels; `document.documentElement.scrollHeight`
  inside the page gives the exact figure.

  ```html
  <iframe src="anneal_300c_2x2048x2048_exact.html"
          style="width:100%; height:720px; border:0" loading="lazy"></iframe>
  ```

- **Serve it for a remote viewer**: `--serve` opens it from a local HTTP
  server and prints the URL, which an SSH tunnel can forward.

WebGPU needs a secure context, such as `localhost` or `https`. The badge
beside the title says which renderer drew the page: `WebGPU` or `CPU`.

### For agents and scripts

Write without opening a browser, into a known folder, and read the one line it
prints:

```bash
quantem-widget show3d ./frames/ --title "Drift series" --no-open --save-dir ./report
# wrote ./report/drift_series_40x512x512_exact.html (55.4 MB, full float32)
```

- Success exits 0 and prints `wrote <path> (<size> MB, <packing>)` on stdout.
- A problem you can fix (a missing file, a label count, a page above
  `--max-mb`) exits 1 with one `quantem-widget: ...` line on stderr naming the flag
  that fixes it. Anything else is a bug and keeps its traceback.
- `--title` sets the file name, so a script can predict it.
- Before reporting what a page shows, open it in a browser and check
  `navigator.gpu.requestAdapter()` gives real hardware, not `swiftshader`.

See [Agent-assisted development](agent-prompts.md) for a copyable prompt.

## GitHub preview copy

GitHub's notebook renderer does not run widget JavaScript. `quantem-widget github`
edits a notebook copy in place: it strips the offline widget state, keeps each
widget's Python-rendered scientific preview (or captures the full UI in a
browser when no preview exists), re-encodes the pictures as JPEG and leaves the
print outputs. Never run it on the canonical tutorial notebooks; make a copy
first:

```bash
cp analysis.ipynb analysis_github.ipynb
quantem-widget github analysis_github.ipynb --no-execute
```

## Options

| Option | Effect |
|---|---|
| `--detector-bin N` | `--html` packed in Python: detector mean-bin factor (default 1). `--backend webgpu` and live notebooks require 1 |
| `--backend auto/webgpu` | `auto` (default) loads onto CUDA, Apple MPS or the CPU, whichever the machine has; `webgpu` with `--html` writes a browser-owned full-detector HDF5-backed viewer |
| `--count N` | require and load exactly this many compatible masters from the input |
| `--dtype auto/uint8/uint16` | HDF5 HTML dtype; Python packing defaults to `auto`, choosing uint8 when counts fit and uint16 otherwise. An explicit uint8 choice is for compact browsing. Native QEM keeps its stored dtype |
| `--serve` | open via a local HTTP server even for self-contained files (tunnelable URL) |
| `--html` | write the offline-WebGPU HTML instead of a notebook |
| `--save-dir PATH` | output folder (default `~/Downloads`) |
| `--no-open` | write the file(s) without launching a browser or Jupyter |
| `--title`, `-v/--verbose` | page title; verbose progress |
| `--no-execute`, `--quality Q`, `--max-width PX`, `--timeout-s S` | `github`: reuse saved outputs, JPEG quality, embedded picture width, per-cell execution timeout |
| `--labels A B ...` | `show2d`/`show3d`: one label per image (Show2D) or per frame (Show3D); default the file names |
| `--cmap`, `--vmin`, `--vmax` | `show2d`/`show3d`: colormap and display contrast limits, in image units; the data is unchanged |
| `--sampling S`, `--units U` | `show2d`/`show3d`: pixel size for the scale bar; overrides a DM or EMD calibration |
| `--no-link-contrast` | `show2d`/`show3d`: give each panel its own contrast range (Show2D links them by default) |
| `--columns N`, `--size small/medium/large/xlarge` | `show2d`/`show3d`: panels across; width of each panel (300, 450, 600, 900 CSS px) |
| `--encoding full/uint8`, `--image-bin N`, `--max-mb MB/none` | `show2d`/`show3d`: exact float32 or uint8 pixels; mean-bin factor (uint8 only); size refusal |
| `--pattern GLOB` | `show2d`/`show3d` folder input: which files to read |
| `--log-scale` (`show2d`), `--fps F` (`show3d`) | log display; playback rate |

## Backends

The loader picks the accelerated backend automatically: **CUDA** on an NVIDIA
box and **Apple Metal (MPS)** on a Mac. `--backend webgpu` hands the compute to
the browser instead of the Python process. On a MacBook:

```bash
quantem-widget show4dstem ./masters/ --backend webgpu --html --count 1 --detector-bin 1
```

uses browser WebGPU and writes a double-clickable HDF5-backed folder without
copying raw data. Without `--backend webgpu`, an HDF5 `--html` export packs the
array in Python. `--detector-bin N` with `N > 1` mean-bins the detector as an explicit
sampling reduction; the resulting values can still exceed 255. Keep the default
`--dtype auto` for range-based Python packing, or choose `uint16` when the wider
detector-count range is needed. Binning alone does not guarantee uint8 safety.
See [Load and I/O](api/io) for the backend details.
