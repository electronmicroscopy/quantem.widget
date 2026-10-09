# HTML export

`quantem.widget` has three sharing paths:

1. **Widget-level HTML**: `widget.export_html(...)` writes one standalone widget
   viewer. Use this when a single Show1D / Show2D / Show3D / Show3DSlices /
   Show4DSTEM view is the artifact.
2. **Notebook-level HTML**: `jupyter nbconvert --to html notebook.ipynb`
   exports a whole notebook with its saved widget state (add `--execute` to
   rerun it first). Use this for reports and
   tutorials that combine text, figures, and multiple widgets.
3. **GitHub preview notebook**: `quantem github notebook.ipynb --no-execute`
   keeps a notebook readable on GitHub by replacing live widgets with compressed
   images of the widget UI.

This page defines the widget-level convention. It is intentionally a structural
protocol, not a base class. Each widget keeps its own packing and performance
logic, but the public shape is shared and easy for users, tests, and LLM agents
to find.

> **Opening an exported widget on another machine (desktop, phone or tablet):** interactive widgets
> recompute with WebGPU, and browsers expose WebGPU **only in a secure context: HTTPS, `localhost`, or a
> local file (`file://`)**. An exported HTML served over plain `http://<ip>:<port>` falls back to CPU canvas
> (Show3D reports "WebGPU unavailable: not a secure context"), and on a phone it renders the first frame but
> ignores taps, because `navigator.gpu` is withheld over insecure origins. The simplest fix is to copy the file
> to the viewing machine and open it from disk (for example `rsync report.html mac:Downloads/` then open it in
> Chrome); a 20-frame 364 x 364 Show3D stack then reports "WebGPU resident" on an Apple M-series Mac. To serve
> it instead, use **HTTPS** (for example `tailscale serve --bg --https=443 http://127.0.0.1:<port>`). Full
> explanation, browser flags, and a debugging page: see
> [Viewing exported HTML on mobile](../maintainer/viewing-html-on-mobile.md).

## Python contract

Every export-capable widget exposes `export_html(path=None, *, title=None, ...)`.
It returns the written `pathlib.Path`, creates parent directories, and updates
`widget.export_status` with the filename, size, and selected encoding. The
exported page hydrates with the ipywidgets HTML manager and runs without a live
Python kernel. Browser-side changes in the HTML are local; they do not write
back to the `.ipynb` or `.html` file.

| Widget | Python API |
|---|---|
| Show1D | `export_html(path=None, *, title=None, mode="single", encoding="full", downsample=None)` |
| Show2D | `export_html(path=None, *, title=None, mode="single", encoding="full", downsample=None, quantized=None, max_mb=...)` |
| Show3D | `export_html(path=None, *, title=None, mode="single", encoding="full", downsample=None, quantized=None, max_mb=...)` |
| Show3DSlices | `export_html(path=None, *, title=None, encoding="full")` |
| Show4DSTEM | `export_html(path=None, *, title=None, dtype="uint8", det_bin=1, scan_bin=1, export_kind="interactive", dataset_scope="unhidden")` |
| ShowDiffraction | `export_html(path=None, *, title=None)` |
| ShowPtycho | `export(path)` writes a folder: the widget page over exact bright-field counts; the browser builds transient BF-indexed reducers in WebGPU |

`encoding="uint8"` (or `quantized=True`) stores display-scaled data; `downsample`
reduces the shape before export. Each reduction is explicit: nothing is binned
or quantized unless the call asks for it.

## HTML export button

Widgets that expose an in-widget **Export** button share these synced traits
(`HtmlExportMixin` in `quantem/widget/export.py`):

| Trait | Direction | Purpose |
|---|---|---|
| `export_enabled` | Python -> JS | Show or disable the export UI |
| `export_request` | JS -> Python | JSON request: mode, request id, filename, download flag |
| `export_status` | Python -> JS | Human-readable progress, size, or error |
| `export_payload` | Python -> JS | HTML bytes when the browser initiated a download |
| `export_payload_id` | Python -> JS | Echoes the request id so JS downloads the right payload once |
| `export_filename` | Python -> JS | Suggested download filename |

The Show2D and Show3D menus send `{"mode": "exact" | "quantized", "id": ..., "filename": ..., "download": true}`;
`download=true` means Python builds HTML bytes into `export_payload`, otherwise
it writes the file by calling `export_html(...)`.

## Widget capability table

| Widget | Encoding | Downsample | Reducer / notes |
|---|---|---|---|
| Show1D | `full` | `1`, `2`, `4`, `8` | preserves every trace/x sample; linked 2D snapshot and profile panels use a NaN-aware area mean, with pixel size and panel/profile coordinates rescaled |
| Show2D | `full`, `uint8` | `1`, `2`, `4`, `8` (uint8 only) | `uint8` stores display-scaled image data; downsample is a mean bin |
| Show3D | `full`, `uint8` | `1`, `2`, `4`, `8` (uint8 only) | `uint8` stores display-scaled volume data; downsample is a mean bin |
| Show3DSlices | `full`, `uint8` | none | `uint8` packs against the global range |
| Show4DSTEM | `dtype="uint8"`, `"uint16"` | `det_bin`, `scan_bin`: `1`, `2`, `4`, `8` | mean bin; `uint8` clips counts above 255 and says how many; `export_kind="report"` writes static PNG virtual-image pages with no raw 4D; interactive exports write a local launcher beside the page |

## Single-file exports

`export_html` writes one HTML file: easiest to email, upload, and move around.
When the exact data is too large for one file, Show2D and Show3D refuse above
`max_mb` and name the smaller options (`encoding="uint8"`, `downsample=2` or
`4`); pass `max_mb=None` to force the size. Show4DSTEM interactive exports and
the CLI WebGPU route keep their data beside the page and serve it with
`Show4DSTEM.command`: `index.html`, `Show4DSTEM.command`, `.viewer/`, and
anonymous `dataset_NN_master.h5` / `dataset_NN_data_*.h5` links. Double-click
`index.html` and grant the folder in Chromium, or run the command file to serve
the same folder locally.

Reducer choice is part of the scientific contract. Compact `uint8` 4D-STEM
exports should avoid immediate clipping, so detector downsample may use a
mean/average reducer and should be labeled as browse-quality. Count-preserving
exports such as `full`/uint16 should preserve detector counts; if they
downsample, sum is usually the scientifically expected reducer when the stored
dtype can hold the result. If a widget chooses mean for a full export, the UI
and docs must say so.

## Show4DSTEM report versus interactive export

Show4DSTEM intentionally exposes an additional `export_kind` option because
4D-STEM data can be too large for a single raw interactive artifact.

Use `export_kind="report"` for collaborator screening, multi-master folder
reviews, and static scientific handoff. It writes a self-contained HTML report
with virtual-image PNG pages and representative diffraction images. It does not
embed raw 4D detector data, so the reader cannot drag a new detector ROI in the
exported page.

Use `export_kind="interactive"` when the reader must keep changing detector ROIs
offline in the browser. It embeds or serves a binned raw-4D payload and runs the
virtual-detector math in WebGPU. This can be much larger than a report. The
payload is the viewer's 4D array. A live viewer over `quantem.gpu.io.load`
acquisitions, including `Show4DSTEM.from_folder(...)`, reads that array in small
scan windows, so the export copies every acquisition to host memory at the
chosen dtype and binning while the GPU keeps only the encoded storage.

For raw HDF5 masters, prefer the CLI WebGPU folder route when the user wants
native detector sampling without a notebook:

```bash
quantem show4dstem /data/session --backend webgpu --html --count 7 --bin 1 --dtype uint8
```

Choose `dtype="uint8"` for compact browse payloads and `dtype="uint16"` when
the exported interactive raw-4D payload must preserve the wider detector-count
range. `uint8` uses one byte per detector pixel and may clip values above 255;
`uint16` uses two bytes per detector pixel and can produce much larger browser
artifacts. The full no-bin interactive path is:

```python
viewer.export_html(
    "show4dstem_full_interactive.html",
    export_kind="interactive",
    dtype="uint16",
    scan_bin=1,
    det_bin=1,
)
```

Copyable examples:

```python
# Compact report: safe default for large folders.
viewer.export_html(
    "show4dstem_report.html",
    export_kind="report",
    dataset_scope="unhidden",
    scan_bin=2,
    det_bin=8,
    dtype="uint8",
)

# Offline raw 4D browser: use only when the exported page needs live ROI changes.
viewer.export_html(
    "show4dstem_interactive.html",
    export_kind="interactive",
    dtype="uint8",
    scan_bin=2,
    det_bin=4,
)
```

For more recipes, see [Show4DSTEM export recipes](../tutorials/show4dstem_export.md).

## Notebook sharing

HTML export and GitHub preview solve different problems:

| Command | Output | Interactive | Use it for |
|---|---|---:|---|
| normal `.ipynb` saved from Jupyter | notebook with widget state | yes, in Jupyter | continuing work |
| `jupyter nbconvert --to html notebook.ipynb` | standalone HTML page | yes, in a browser | sharing an interactive report |
| `quantem github notebook.ipynb --no-execute` | optional notebook copy with compressed widget pictures | no | GitHub notebook preview |

GitHub does not run widget JavaScript. Use the hosted documentation, Colab,
Jupyter, or the nbconvert HTML for real interaction. The `quantem github` command is
only for a separate non-interactive notebook copy for GitHub's native renderer;
never run it in place on the canonical tutorial notebooks.

## Implementation checklist

For a new widget, copy the pattern from the closest existing widget:

- Public `export_html(...) -> pathlib.Path`.
- `HtmlExportMixin` for the toolbar handshake; override
  `_html_export_options(...)`, `_default_html_export_path(...)` and
  `_export_mode_label(...)` for the widget's options.
- `_clone_for_html_export(...)` when the standalone artifact needs a
  packed/export-only widget state; `write_widget_html` writes the page.
- Export status strings that include both file size and encoding.

The data packing stays widget-specific so full, uint8 and downsampled exports
are honest about precision, size, and performance.
