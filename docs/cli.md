# Command line

Installing `quantem.widget` adds the `quantem` command.
Point it at a file or a folder and it renders the right viewer - no notebook,
no Python.

```bash
quantem show ./anything/                     # auto-detect content, pick the viewer
quantem show2d scan.png                       # an image            -> Show2D
quantem show3d ./frames/                       # a folder of frames -> Show3D scrub
quantem show2d ./frames/ --watch               # live folder        -> append new images
quantem show4dstem ./masters/                  # *_master.h5        -> live Show4DSTEM
quantem show4dstem a_master.h5 b_master.h5     # several masters    -> one 5D multi-tilt viewer
quantem show4dstem ./masters/ --html           # 4D-STEM            -> shareable offline HTML
quantem showptycho scan_master.h5               # raw 4D-STEM master -> full-BF SSB review project
quantem showptycho ./masters/                    # master folder     -> ShowPtycho project catalog
quantem showfolder ./session/                  # microscopy folder  -> ShowFolder notebook/HTML
quantem html tutorial.ipynb                    # a notebook         -> standalone interactive HTML
quantem github tutorial_github.ipynb --no-execute # optional static copy for GitHub preview
```

## Subcommands

| Command | Input | Output |
|---|---|---|
| `quantem show <path>` | anything | auto-detects and dispatches to one of the below |
| `quantem show2d <image / folder>` | one image, or a folder | a Show2D HTML (a folder becomes a gallery); with `--watch`, a live ShowFolder notebook |
| `quantem show3d <folder>` | a folder of same-size frames | a Show3D scrub HTML; with `--watch`, a live ShowFolder notebook |
| `quantem show4dstem <master(s) / folder>` | one or more `*_master.h5` | a live Show4DSTEM notebook (or `--html`) |
| `quantem showptycho <master.h5 / folder>` | raw `*_master.h5` files, a folder of masters, or an existing ShowPtycho project | runs full-BF SSB and builds one index with direct ShowPtycho and Show4DSTEM browser viewers |
| `quantem showfolder <folder>` | microscopy session folder | a ShowFolder notebook (or `--html`) |
| `quantem html <notebook.ipynb>` | a notebook you wrote | runs it, or with `--no-execute` exports saved outputs/state, into one standalone interactive HTML |
| `quantem github <notebook.ipynb>` | an optional static copy of a notebook | strips widget state and embeds compressed pictures for GitHub's notebook preview |

**Images** save a standalone HTML and open in your browser. **4D-STEM** opens a
live, kernel-backed notebook by default (full real-time interaction); `--html`
instead writes an **offline WebGPU browser folder** - drag detectors, switch
BF/ABF/ADF, pan diffraction, all with no kernel. Full-detector WebGPU exports
keep compressed HDF5 files beside the viewer. Open `index.html` and grant the
data folder when prompted, or double-click `Show4DSTEM.command` to serve that
same folder locally without a grant click.

Several masters (a folder, or listed explicitly) stack into **one 5D viewer with a
Dataset slider** to flip between scans. WebGPU HDF5 folders use anonymous local
links such as `tilt_00_master.h5` and `tilt_00_data_*.h5`; rerunning the CLI
replaces the generated viewer folder so stale HTML and metadata do not survive.

Image and Show4DSTEM outputs land in `~/Downloads` by default. ShowPtycho
projects use the user-owned `~/QuantEM/showptycho/<acquisition>` root. Use
`--out PATH` to choose another writable project folder or `--in-place` to opt
into `SOURCE/quantem/showptycho`.

## Show4DSTEM HTML export

Use the CLI when you want a quick browser artifact from raw masters:

```bash
quantem show4dstem scan_001_master.h5 --backend webgpu --html --bin 1
quantem show4dstem ./session_masters --backend webgpu --html --count 7 --bin 1 --out ~/Downloads
quantem show4dstem scan_001_master.h5 scan_002_master.h5 --backend webgpu --html --bin 1
```

`--bin` is detector mean binning for the exported browser payload. The default
is `--bin 1`, meaning full detector sampling. Use a larger value only for an
explicit preview, and label that reduction in the report.

Use `--backend webgpu --html --bin 1` when the user wants the full native
detector sampling path without opening Jupyter:

```bash
quantem show4dstem /data/session --backend webgpu --html --count 7 --bin 1 --dtype uint8 --out ~/Downloads
```

That command writes a browser folder with anonymous H5 symlinks plus
`Show4DSTEM.command`, so it does not copy raw data into a giant HTML file. It is
the right no-notebook choice when native detector detail matters. Double-click
`index.html` and grant the export folder when Chrome asks, or use
`Show4DSTEM.command` when you want the local server path. Multi-master WebGPU
exports open as one dataset-slider viewer; generated review/demo exports should
use `view_mode="multiple"` and `compare_dp_mode="selected"` when the point is to
compare tilts or scans side by side. For a compact collaborator review, use the
Python `export_kind="report"` path below.

For large live folders, curated review grids, or collaborator inspection,
open a live viewer and export a compact report from Python instead:

```python
from quantem.widget import Show4DSTEM

viewer = Show4DSTEM.from_folder(
    "/data/session",
    gpus=[0, 1],
    det_bin=1,
    dtype="u8",
    view_mode="multiple",
    page_size=12,
)

viewer.export_html(
    "show4dstem_report.html",
    export_kind="report",
    dataset_scope="unhidden",
    scan_bin=2,
    det_bin=8,
    dtype="uint8",
)
```

Use `export_kind="interactive"` from Python when you want the same offline
browser interaction as the CLI but need finer control over real-space binning,
detector binning, or dtype:

```python
viewer.export_html(
    "show4dstem_interactive.html",
    export_kind="interactive",
    dtype="uint8",
    scan_bin=2,
    det_bin=4,
)
```

See [Show4DSTEM export recipes](tutorials/show4dstem_export) for the decision
table and LLM-friendly checklist.

## ShowPtycho folder review

ShowPtycho WebGPU review can start directly from one `*_master.h5`:

```bash
quantem showptycho reference_512_master.h5
```

The command looks for a matching ShowPtycho calibration next to the master, for
example `quantem/showptycho/<dataset>/calibration.json`. When that file is not present, it
uses quick-start defaults and prints them before loading:

```text
semiangle=30 mrad, scan_sampling=0.5 A, voltage=300 kV
```

Those defaults are enough for fast local review and collaborator handoff. For
measurement, publication, or calibration signoff, provide the microscope
geometry explicitly:

```bash
quantem showptycho reference_512_master.h5 \
  --semiangle 30 --scan-sampling 0.264 --voltage-kv 300
```

ShowPtycho always uses native detector sampling because ptychography review must
not silently downsample the bright-field disk. The generated project contains
an `index.html` catalog, one exact ShowPtycho folder per dataset, and one direct
Show4DSTEM WebGPU viewer per dataset. Show4DSTEM reads the original compressed
HDF5 family through hard links or symbolic links; it does not duplicate the raw detector data. ShowPtycho stores only
the source links and exact BF evidence required by its browser workflow. It does not save persistent
float32 reference images or a complex64 BF reducer by default. The browser
decodes HDF5 chunks on WebGPU and builds the BF-indexed reducers transiently.
The default interactive BF policy is full selected BF (`--drag-bf 1.0`) so the
first view uses all known BF evidence without loading non-BF detector pixels.
Use `--drag-bf 0.3` or another smaller fraction only when you intentionally want
a faster exploratory preview.

The default ShowPtycho browser source is the exact BF-column payload. The
exported folder also carries the original compressed HDF5 family under
`source/` for provenance and the Show4DSTEM companion. This keeps ShowPtycho
startup focused on the detector columns required by SSB without detector
binning or loading non-BF pixels.

Existing ShowPtycho WebGPU exports are also folders because the microscopy
payload can be several gigabytes. Open them with the CLI:

```bash
quantem showptycho ./logic013_512_bfr24/
```

or let auto-detection choose the same path:

```bash
quantem show ./logic013_512_bfr24/
```

The command validates `manifest.json`, prints the compressed HDF5 source
summary, starts the required local HTTP server with byte-range support, opens
`index.html`, and stays alive until Ctrl-C. Use `--port 8900` for a fixed port
or `--bind 0.0.0.0` only when you intentionally want another device on the
network to reach the viewer. Share the whole folder with a colleague; sending
only `index.html` omits the HDF5 source files needed for WebGPU reconstruction.

## Options

| Option | Effect |
|---|---|
| `--bin N` | detector mean-bin factor; Show4DSTEM defaults to 1 and ShowPtycho always uses native detector sampling |
| `--backend auto/cuda/mps/webgpu` | Show4DSTEM backend; use `webgpu` with `--html` for a browser-owned full-detector HDF5-backed viewer. ShowPtycho accepts `auto/cuda/mps` |
| `--count N` | Show4DSTEM: require and load exactly this many compatible masters from the input |
| `--devices 0,1` | Show4DSTEM CUDA placement; alias of `--gpus` |
| `--dtype uint8/uint16/float32` | browse/storage dtype; `uint8` is compact browse, `uint16` keeps the wider detector-count range |
| `--serve` | open via a local HTTP server even for self-contained files (tunnelable URL) |
| `--port N`, `--bind ADDR` | folder exports: local HTTP server port (default auto) and bind address (default 127.0.0.1) |
| `--quantized` | image widgets: uint8 pack for a smaller file |
| `--html` | 4D-STEM: write the offline-WebGPU HTML instead of a notebook |
| `--watch` | folder: write a live ShowFolder-watched notebook; Show2D/Show3D append new image files, Show4DSTEM opens lazy masters |
| `--gpus 0,1`, `--page-budget auto` | watched Show4DSTEM: pick CUDA cards and GPU-resident dataset cache policy |
| `--combined` | many masters -> one 5D HTML viewer (served locally) |
| `--out PATH` | output file or directory (default `~/Downloads`) |
| `--no-open` | write the file(s) without launching a browser or Jupyter |
| `--title`, `-v/--verbose` | page title; verbose progress |
| `--calibration`, `--semiangle`, `--scan-sampling`, `--det-sampling`, `--voltage-kv` | ShowPtycho master generation geometry and calibration controls |
| `--trials N`, `--refinement nelder-mead/none` | ShowPtycho: exact GPU Optuna trial budget (default 200; `0` reuses a resolved calibration) and post-trial refinement |
| `--in-place`, `--anonymize` | ShowPtycho: write under `SOURCE/quantem/showptycho`; redact the local acquisition name/path from saved provenance |
| `--drag-bf X` | ShowPtycho BF fraction or count; default `1.0` is full BF, `0.3` is 30 percent, values greater than 1 are explicit BF-pixel counts |
| `--size PX`, `--fft`, `--force` | ShowPtycho: initial panel size, open with the FFT panel visible, rebuild an existing output folder |

## Backends

The loader picks the accelerated backend automatically - **CUDA** on an NVIDIA
box and **Apple Metal (MPS)** on a Mac. `--backend webgpu` hands the compute to
the browser instead of the Python process. On a MacBook:

```bash
quantem show4dstem ./masters/ --backend webgpu --html --count 1 --bin 1
```

uses browser WebGPU and writes a double-clickable HDF5-backed folder without
copying raw data. If you pass `--bin N` with `N > 1`, the detector is
**mean-binned** (not summed) so the bright field never clips at uint8. See
[Load and I/O](api/io) for the backend + binning details.
