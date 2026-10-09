# Command line

Installing `quantem.widget` adds the `quantem` command with two subcommands:
`show4dstem` opens 4D-STEM masters without a notebook, and `github` makes a
widget notebook displayable in GitHub's static preview.

```bash
quantem show4dstem ./masters/                  # *_master.h5        -> live Show4DSTEM notebook
quantem show4dstem a_master.h5 b_master.h5     # several masters    -> one comparison viewer
quantem show4dstem ./masters/ --html           # 4D-STEM            -> shareable offline HTML
quantem show4dstem ./masters/ --watch          # live folder        -> notebook that appends new masters
quantem github tutorial_github.ipynb --no-execute # optional static copy for GitHub preview
```

Everything else (Show2D, Show3D, ShowDiffraction, ShowPtycho exports, notebook
HTML) is a Python call on the widget: `widget.export_html(...)`, or
`jupyter nbconvert --to html notebook.ipynb` for a whole notebook.

## Subcommands

| Command | Input | Output |
|---|---|---|
| `quantem show4dstem <master(s) / folder>` | one or more `*_master.h5` | a live Show4DSTEM notebook (or `--html`) |
| `quantem github <notebook.ipynb>` | an optional static copy of a notebook | strips widget state and embeds compressed pictures for GitHub's notebook preview |

**4D-STEM** opens a live, kernel-backed notebook by default (full real-time
interaction; each master stays encoded on the GPU at full detector sampling);
`--html` instead writes an **offline WebGPU browser folder**: drag detectors,
switch BF/ABF/ADF, pan diffraction, all with no kernel. Full-detector WebGPU
exports keep compressed HDF5 files beside the viewer. Open `index.html` and
grant the data folder when prompted, or double-click `Show4DSTEM.command` to
serve that same folder locally without a grant click.

Several masters (a folder, or listed explicitly) open as **one comparison
viewer**: a live notebook shows one virtual image per master with a shared
detector ROI, and an HTML export adds a Dataset slider to flip between scans.
WebGPU HDF5 folders use anonymous local links such as `dataset_00_master.h5`
and `dataset_00_data_*.h5`; rerunning the CLI replaces the generated viewer
folder so stale HTML and metadata do not survive.

Outputs land in `~/Downloads` by default. Use `--out PATH` to choose another
writable folder.

## Show4DSTEM HTML export

Use the CLI when you want a quick browser artifact from raw masters:

```bash
quantem show4dstem scan_001_master.h5 --backend webgpu --html --bin 1
quantem show4dstem ./session_masters --backend webgpu --html --count 7 --bin 1 --out ~/Downloads
quantem show4dstem scan_001_master.h5 scan_002_master.h5 --backend webgpu --html --bin 1
```

`--bin N` replaces each N x N detector block by its mean, rounded to the
nearest count, in an `--html` export packed in Python (`--backend auto`,
`cuda`, or `mps`). The default is `--bin 1`, meaning full detector sampling.
Use a larger value only for an explicit preview, and label that reduction in
the report. The `--backend webgpu` folder reads the source HDF5 chunks in the
browser and requires `--bin 1`; live notebooks always keep full detector
sampling.

Use `--backend webgpu --html --bin 1` when the user wants the full native
detector sampling path without opening Jupyter:

```bash
quantem show4dstem /data/session --backend webgpu --html --count 7 --bin 1 --dtype uint8 --out ~/Downloads
```

That command writes a browser folder with anonymous H5 symlinks plus
`Show4DSTEM.command`, so it does not copy raw data into a giant HTML file.
Double-click `index.html` and grant the export folder when Chrome asks, or use
`Show4DSTEM.command` when you want the local server path. Multi-master WebGPU
exports open as one dataset-slider viewer. For a compact collaborator review,
use the Python `export_kind="report"` path:

```python
from quantem.widget import Show4DSTEM

viewer = Show4DSTEM.from_folder("/data/session")
viewer.wait_for_folder()
viewer.export_html("show4dstem_report.html", export_kind="report", dtype="uint8")
```

See [Show4DSTEM export recipes](tutorials/show4dstem_export) for the decision
table.

## GitHub preview copy

GitHub's notebook renderer does not run widget JavaScript. `quantem github`
edits a notebook copy in place: it strips the offline widget state, keeps each
widget's Python-rendered scientific preview (or captures the full UI in a
browser when no preview exists), re-encodes the pictures as JPEG and leaves the
print outputs. Never run it on the canonical tutorial notebooks; make a copy
first:

```bash
cp analysis.ipynb analysis_github.ipynb
quantem github analysis_github.ipynb --no-execute
```

## Options

| Option | Effect |
|---|---|
| `--bin N` | `--html` packed in Python: detector mean-bin factor (default 1). `--backend webgpu` and live notebooks require 1 |
| `--backend auto/cuda/mps/webgpu` | Show4DSTEM backend; use `webgpu` with `--html` for a browser-owned full-detector HDF5-backed viewer |
| `--count N` | require and load exactly this many compatible masters from the input |
| `--dtype uint8/uint16` | `--html` packed or browser-decoded dtype; `uint8` is compact browse, `uint16` keeps the wider detector-count range |
| `--serve` | open via a local HTTP server even for self-contained files (tunnelable URL) |
| `--html` | write the offline-WebGPU HTML instead of a notebook |
| `--watch`, `--watch-interval S` | folder: write a live viewer notebook that opens the ready masters and appends new ones |
| `--scan-size N` | watched folder: keep only masters with this square scan size |
| `--out PATH` | output file or directory (default `~/Downloads`) |
| `--no-open` | write the file(s) without launching a browser or Jupyter |
| `--title`, `-v/--verbose` | page title; verbose progress |
| `--no-execute`, `--quality Q`, `--max-width PX`, `--timeout S` | `github`: reuse saved outputs, JPEG quality, embedded picture width, per-cell execution timeout |

## Backends

The loader picks the accelerated backend automatically: **CUDA** on an NVIDIA
box and **Apple Metal (MPS)** on a Mac. `--backend webgpu` hands the compute to
the browser instead of the Python process. On a MacBook:

```bash
quantem show4dstem ./masters/ --backend webgpu --html --count 1 --bin 1
```

uses browser WebGPU and writes a double-clickable HDF5-backed folder without
copying raw data. Without `--backend webgpu`, `--html` packs the array in
Python; `--bin N` with `N > 1` then **mean-bins** (not sums) the detector so the
bright field never clips at uint8. See [Load and I/O](api/io) for the backend
details.
