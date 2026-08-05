# Show4DSTEM Export Recipes

This page is the copy-paste reference for exporting `Show4DSTEM` results. It is
written for both humans and LLM agents: choose the goal, copy the matching
recipe, and keep the reduction choices explicit.

## Decision Table

| Goal | Recommended path | Output | Opens without Python? | Contains raw 4D data? |
|---|---|---|---:|---:|
| Continue analysis | `Show4DSTEM(load(...))` or `quantem show4dstem ...` | Live notebook | no | yes, in the Python session |
| Share a compact screening result | `export_html(export_kind="report")` | One HTML report | yes | no |
| Share an offline detector-ROI browser | `export_html(export_kind="interactive")` | WebGPU HTML/folder | yes | yes, binned/encoded |
| Export quickly from a terminal | `quantem show4dstem ... --backend webgpu --html --count N` | WebGPU HDF5 folder | yes | yes, in source HDF5 files |
| Export full native detector sampling from a terminal | `quantem show4dstem ... --backend webgpu --html --bin 1 --dtype uint8` | WebGPU HDF5 folder | yes | yes, native detector sampling |
| Preserve compressed HDF5 beside the viewer | WebGPU HDF5 folder | `index.html` + `Show4DSTEM.command` + `.viewer/` + `tilt_NN_master.h5` + `tilt_NN_data_*.h5` | yes, via folder grant or local server | yes, in source HDF5 files |

Default recommendation: use `export_kind="report"` for large folders, many
datasets, or collaborator screening. Use `export_kind="interactive"` only when
the recipient must drag detector ROIs in the exported browser.

## Minimal Report Export

Report export writes a compact, self-contained HTML file. It contains rendered
virtual-image PNG pages and a representative diffraction pattern, not raw 4D
detector data. This is the safest default for large folder viewers.

```python
from quantem.widget import Show4DSTEM

viewer = Show4DSTEM.from_folder(
    "/data/session",
    gpus=[0, 1],
    det_bin=1,
    dtype="u8",
    view_mode="multiple",
    page_size=12,
    compare_group_mode="paged",
    compare_dp_mode="selected",
)

path = viewer.export_html(
    "show4dstem_report.html",
    export_kind="report",
    dataset_scope="unhidden",  # "current_page", "starred", "unhidden", or "all"
    scan_bin=2,                # mean-bin real-space PNG pages
    det_bin=8,                 # mean-bin representative diffraction thumbnail
    dtype="uint8",
)
print(path)
```

Use `dataset_scope` deliberately:

| `dataset_scope` | Meaning |
|---|---|
| `"current_page"` | Export only the visible review page. |
| `"starred"` | Export curated starred panels. |
| `"unhidden"` | Export every panel that has not been hidden. |
| `"all"` | Export every dataset, including hidden panels. |

## Minimal Interactive Raw 4D Export

Interactive export embeds a raw 4D payload, after the explicit `scan_bin`,
`det_bin`, and `dtype` choices. The exported page can run virtual-detector
interaction in the browser without a Python kernel, but the file can be much
larger than a report.

```python
path = viewer.export_html(
    "show4dstem_interactive.html",
    export_kind="interactive",
    dtype="uint8",  # "uint16" keeps the wider integer range and makes a larger file
    scan_bin=2,     # mean-bin scan rows/cols before embedding raw 4D
    det_bin=4,      # mean-bin detector rows/cols before embedding raw 4D
)
print(path)
```

Use this when the recipient needs to:

- drag BF/ABF/ADF/HAADF or custom detector ROIs offline,
- inspect diffraction and virtual-image panels without Jupyter,
- review a small number of binned datasets in a browser.

Do not use this as the first choice for a large multi-master screening report.
Report export is smaller and faster because it does not embed raw 4D data.

## Terminal Export

The command-line path is the fastest way to hand a master or folder to a
non-notebook user. It writes a folder-backed browser viewer rather than copying
raw HDF5 into one huge HTML file:

```bash
quantem show4dstem /data/session --backend webgpu --html --count 1 --out ~/Downloads
```

Useful variants:

```bash
# One master, full detector sampling, browser WebGPU HDF5 folder.
quantem show4dstem scan_001_master.h5 --backend webgpu --html --bin 1

# Seven compatible masters as one 5D viewer with a Dataset slider.
quantem show4dstem /data/session --backend webgpu --html --count 7 --bin 1

# Write without opening a browser.
quantem show4dstem /data/session --backend webgpu --html --count 7 --no-open
```

CLI `--bin` is detector mean binning for the export. The default is `--bin 1`,
meaning full detector sampling. Use a larger value only when making an explicit
preview, and state that reduction in the report.

## Full Native Export Without A Notebook

Some users do not want Jupyter at all, and sometimes they want native detector
sampling. Use the CLI with `--html --bin 1`:

```bash
# Native detector sampling, browser WebGPU HDF5 folder.
quantem show4dstem scan_001_master.h5 --backend webgpu --html --bin 1 --dtype uint8 --out ~/Downloads

# Native detector sampling for every master in a folder.
quantem show4dstem /data/session --backend webgpu --html --count 7 --bin 1 --dtype uint8 --out ~/Downloads

# Several native-sampling masters in one Dataset-slider viewer.
quantem show4dstem scan_001_master.h5 scan_002_master.h5 \
  --backend webgpu --html --bin 1 --dtype uint8 --out ~/Downloads
```

This is the no-notebook path for a full interactive browser artifact. It keeps
native detector sampling and reads source HDF5 frames through a direct browser
folder grant or local range server. Use it when native detector detail matters.
Use an explicit detector bin only for a preview, and use
`export_kind="report"` when the recipient only needs a curated review page.

Equivalent Python:

```python
viewer.export_html(
    "show4dstem_full_interactive.html",
    export_kind="interactive",
    dtype="uint16",
    scan_bin=1,
    det_bin=1,
)
```

## Size And Fidelity Rules

Use these rules when choosing export settings:

| Setting | Effect |
|---|---|
| `export_kind="report"` | Smallest practical review artifact; no raw 4D payload. |
| `export_kind="interactive"` | Browser-owned raw 4D interaction; larger file. |
| `dtype="uint8"` | Compact browse payload; use when count clipping/narrowing is acceptable or already audited. |
| `dtype="uint16"` | Larger payload; keeps the wider integer range for interactive export. |
| `scan_bin=2/4/8` | Mean-bin real-space scan pixels before export. |
| `det_bin=2/4/8` | Mean-bin detector pixels before export. |

Both `scan_bin` and `det_bin` use mean binning in Show4DSTEM export. They reduce
display/review payload size; they are not a hidden scientific reconstruction
step. Put the exact settings in notebooks, captions, and handoff notes.

## Choosing Uint8 Or Uint16

The dtype choice is separate from the binning choice:

| Choose | When | Tradeoff |
|---|---|---|
| `dtype="uint8"` / `--dtype uint8` | First-pass browsing, laptop-sized HTML, tutorials, quick collaborator review where detector-count saturation is acceptable or audited. | 1 byte per detector pixel after any binning; values above 255 clip/narrow, so do not call it exact unless the count range fits. |
| `dtype="uint16"` / `--dtype uint16` | Native-count review, full/no-bin export, detector detail, or any claim where high-count detector values matter. | 2 bytes per detector pixel after any binning; larger files and more browser/GPU memory, but keeps the 0-65535 integer range. |

Use `uint8` for the common full-detector WebGPU browse path:

```bash
quantem show4dstem /data/session --backend webgpu --html --count 7 --bin 1 --dtype uint8
```

Use `uint16` when the browser artifact must preserve the wider integer range:

```bash
quantem show4dstem /data/session --backend webgpu --html --count 7 --bin 1 --dtype uint16
```

Use `uint16` from Python when the exported browser must preserve the wider
integer range:

```python
viewer.export_html(
    "show4dstem_uint16_interactive.html",
    export_kind="interactive",
    dtype="uint16",
    scan_bin=1,
    det_bin=1,
)
```

`uint16` does not make a report export raw. A report still contains rendered
PNG review pages. `uint16` matters for `export_kind="interactive"`, where the
browser receives a raw 4D payload.

## Opening Exported Files

Report export is one self-contained HTML file. It can be double-clicked or
served from any static host.

Interactive raw 4D and HDF5-folder exports write a `Show4DSTEM.command`
launcher next to the HTML when the browser needs byte-range access to nearby
data files. Three ways to open the exported folder:

- **Double-click `Show4DSTEM.command`** (macOS). A Terminal window starts the
  bundled range server for this exact folder and Chrome opens the viewer —
  nothing to install, no clicks in the browser. Closing the Terminal window
  stops the server.
- **Double-click `index.html`** and grant the export folder when Chrome shows
  **Open data folder** (File System Access; browsers without the folder picker
  fall back to a plain file chooser).
- **`quantem show out/`** from a terminal serves the folder and opens the
  viewer without the grant click — handy over remote connections.

Keep the HDF5 files next to the HTML; sending only `index.html` is not a
complete interactive export.

For phones and tablets, WebGPU requires a secure context. Use HTTPS or localhost
and see [Viewing exported HTML on mobile](../maintainer/viewing-html-on-mobile.md).

## LLM Checklist

When generating or reviewing Show4DSTEM export code, verify these points:

- State the goal: live notebook, report, interactive raw 4D, CLI export, or HDF5 bundle.
- Use `export_kind="report"` for large folders unless raw 4D browser interaction is required.
- Use `export_kind="interactive"` only with explicit `dtype`, `scan_bin`, and `det_bin`.
- Do not call a report export "raw" or "exact"; it contains rendered PNG virtual images.
- Do not call `dtype="uint8"` exact unless detector counts were audited to fit.
- Mention that `scan_bin` and `det_bin` are mean-binning choices.
- For multi-master or tilt-series review demos, use `view_mode="multiple"` and
  `compare_dp_mode="selected"` unless an average diffraction pattern is the
  actual measurement being shown.
- Keep generated HTML, screenshots, and private data outside the repository unless they are intentional docs artifacts.

## Related Pages

- [Show4DSTEM tutorial](show4dstem)
- [Show4DSTEM API](../api/show4dstem)
- [HTML export API](../api/html-export)
- [Command line](../cli)
