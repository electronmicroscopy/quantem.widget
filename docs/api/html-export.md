# HTML export

`quantem.widget` has three sharing paths:

1. **Widget-level HTML**: `widget.export_html(...)` writes one standalone widget
   viewer. Use this when a single Show1D / Show2D / Show3D / Show3DSlices /
   Show4DSTEM / ShowEDS view is the artifact.
2. **Notebook-level HTML**: `quantem html notebook.ipynb --no-execute` exports a
   whole notebook with its saved widget state. Use this for reports and
   tutorials that combine text, figures, and multiple widgets.
3. **GitHub preview notebook**: `quantem github notebook.ipynb --no-execute`
   keeps a notebook readable on GitHub by replacing live widgets with compressed
   images of the widget UI.

This page defines the widget-level convention. It is intentionally a structural
protocol, not a base class. Each widget keeps its own packing and performance
logic, but the public shape is shared and easy for users, tests, and LLM agents
to find.

> **Opening an exported widget on a phone or tablet:** interactive widgets
> recompute with WebGPU, and browsers expose WebGPU **only in a secure context
> (HTTPS or localhost)**. An exported HTML served over plain `http://<ip>:<port>`
> to a phone will render the first frame but ignore taps, because `navigator.gpu`
> is withheld over insecure origins. Serve it over **HTTPS** (for example
> `tailscale serve --bg --https=443 http://127.0.0.1:<port>`). Full explanation,
> browser flags, and a debugging page: see
> [Viewing exported HTML on mobile](../maintainer/viewing-html-on-mobile.md).

## Python contract

Every export-capable widget should expose:

```python
from pathlib import Path

path: Path = widget.export_html(
    path=None,          # str | pathlib.Path | None
    title=None,         # optional browser page title when supported
    mode="single",      # "single" or "folder"
    encoding="full",    # "full", "uint8", or widget-specific
    downsample=None,    # None, 2, 4, or widget-specific
)
```

The method returns the written `pathlib.Path`, creates parent directories, and
updates `widget.export_status` with the filename, size, and selected export mode.
The exported page must hydrate with the ipywidgets HTML manager and run without a
live Python kernel. Browser-side changes in the HTML are local; they do not write
back to the `.ipynb` or `.html` file.

For type hints or feature detection, use the structural protocol:

```python
from quantem.widget import SupportsHtmlExport, supports_html_export

def maybe_export(widget: object) -> None:
    if supports_html_export(widget):
        widget.export_html("viewer.html")
```

```{eval-rst}
.. autoclass:: quantem.widget.export.SupportsHtmlExport
   :members:

.. autoclass:: quantem.widget.export.SupportsFrontendHtmlExport
   :members:
```

## HTML export button

Widgets that expose an in-widget **Export** button should use the same synced
traits:

| Trait | Direction | Purpose |
|---|---|---|
| `export_enabled` | Python -> JS | Show or disable the export UI |
| `export_request` | JS -> Python | JSON request: mode, request id, filename, download flag |
| `export_status` | Python -> JS | Human-readable progress, size, or error |
| `export_payload` | Python -> JS | HTML bytes when the browser initiated a download |
| `export_payload_id` | Python -> JS | Echoes the request id so JS downloads the right payload once |
| `export_filename` | Python -> JS | Suggested download filename |

The standard request shape is:

```json
{
  "mode": "single",
  "encoding": "full",
  "downsample": null,
  "id": "unique-request-id",
  "filename": "viewer.html",
  "download": true
}
```

For older widgets, legacy request modes such as `exact`, `quantized`, or
`uint8-bin2` may still be accepted. New code should use `mode`, `encoding`, and
`downsample`. `download=true` means Python should build HTML bytes into
`export_payload`; otherwise Python may write directly to disk by calling
`export_html(...)`.

## Standard options

Use the same three option names across widgets:

| Option | Meaning | Preferred values |
|---|---|---|
| `mode` | where the data lives | `single`, `folder` |
| `encoding` | how the data is stored | `full`, `uint8`, widget-specific |
| `downsample` | whether dimensions are reduced before export | `None`, `2`, `4`, widget-specific |

`mode` is packaging. It should not imply lower precision or a smaller shape.
`encoding` is storage representation. It replaces vague API names like
`quantized`. `downsample` is shape reduction. Each widget must document whether
the reducer is mean, sum, min/max, or another operation, and the export UI must
label the choice clearly enough that users know whether the file is
browse-quality or count-preserving.

## Widget capability table

| Widget | HTML export | Mode | Encoding | Downsample | Folder export | Reducer / notes |
|---|---|---|---|---|---|---|
| Show1D | yes | `single` | `full` | `1`, `2`, `4`, `8` | no | preserves every trace/x sample; linked 2D snapshot and profile panels use a NaN-aware area mean, with pixel size and panel/profile coordinates rescaled |
| Show2D | yes | `single` | `full`, `uint8` | planned | no | `uint8` stores display-scaled image data |
| Show3D | yes | `single` | `full`, `uint8` | planned | no | `uint8` stores display-scaled volume data |
| Show3DSlices | yes | `single` | `full`, `uint8` | planned | no | `uint8` stores display-scaled volume data |
| Show4DSTEM | yes | `single`, `folder` when data already uses a companion folder | `uint8`, `full` | `1`, `2`, `4`, `8` | sometimes | `uint8` detector downsample uses mean for compact browse export; full/uint16 export should document whether downsample is count-preserving sum or display-stable mean |
| ShowEDS | yes | `single`, `folder` | `full` | `2`, `4` | yes | count-preserving sum downsample across spatial and energy axes |

The public Python calls are:

| Widget | Python API |
|---|---|
| Show1D | `export_html(path=None, title=None, mode="single", encoding="full", downsample=None)` |
| Show2D | `export_html(path=None, title=None, mode="single", encoding="full", downsample=None)` |
| Show3D | `export_html(path=None, title=None, mode="single", encoding="full", downsample=None)` |
| Show3DSlices | `export_html(path=None, title=None, mode="single", encoding="full", downsample=None)` |
| Show4DSTEM | `export_html(path=None, title=None, mode="single", encoding="uint8", downsample=1)` |
| ShowEDS | `export_html(path=None, title=None, mode="single", encoding="full", downsample=None)` |
| ShowFolder | `export_html(path=None, title=None)` |

Existing compatibility aliases remain supported:

| Old option | Preferred option |
|---|---|
| `quantized=True` | `encoding="uint8"` |
| `dtype="uint8"` | `encoding="uint8"` |
| `dtype="uint16"` | `encoding="full"` |
| `det_bin=2` | `downsample=2` |
| `binning=4` | `downsample=4` |
| `mode="sidecar"` | `mode="folder"` |

## Single and folder exports

Use `mode="single"` when you want one HTML file. This is the default because it
is easiest to email, upload, and move around.

Use `mode="folder"` when the HTML file should read exact data from a nearby
data folder or URL. The HTML contains the viewer and startup state; the large
dataset stays outside the HTML file.

Use `mode="folder"` when:

- the exact dataset is too large to put inside one HTML file,
- the audience is internal and can keep a folder next to the HTML,
- preserving full data matters more than having one portable file.

Use `mode="single"` when:

- you are sending one file by email or Slack,
- the file will be posted as a simple web download,
- the reader may move the HTML without its data folder.

Use `downsample=2` or `downsample=4` when you still want `mode="single"` but
the exact dataset is too large. Downsampling makes a smaller one-file HTML
export by combining nearby pixels, voxels, detector pixels, or energy channels.
For lab sharing, `mode="folder"` is useful because it keeps exact data without
making the HTML enormous.

Reducer choice is part of the scientific contract. Compact `uint8` 4D-STEM
exports should avoid immediate clipping, so detector downsample may use a
mean/average reducer and should be labeled as browse-quality. Count-preserving
exports such as `full`/uint16 should preserve detector counts; if they
downsample, sum is usually the scientifically expected reducer when the stored
dtype can hold the result. If a widget chooses mean for a full export, the UI
and docs must say so.

## Notebook sharing

HTML export and GitHub preview solve different problems:

| Command | Output | Interactive | Use it for |
|---|---|---:|---|
| normal `.ipynb` saved from Jupyter | notebook with widget state | yes, in Jupyter | continuing work |
| `quantem html notebook.ipynb --no-execute` | standalone HTML page | yes, in a browser | sharing an interactive report |
| `quantem github notebook.ipynb --no-execute` | optional notebook copy with compressed widget pictures | no | GitHub notebook preview |

GitHub does not run widget JavaScript. Use the hosted documentation, Colab,
Jupyter, or `quantem html` for real interaction. The `quantem github` command is
only for a separate non-interactive notebook copy for GitHub's native renderer;
never run it in place on the canonical tutorial notebooks.

## Implementation checklist

For a new widget, copy the pattern from the closest existing widget:

- Public `export_html(...) -> pathlib.Path`.
- Private `_default_html_export_path(...)` with a readable slug, shape, and mode.
- Private `_write_html_export(...)` using `ipywidgets.embed.embed_minimal_html`
  with `dependency_state(..., drop_defaults=False)`.
- Optional `_html_export_bytes(...)` for the browser download path.
- Optional `_clone_for_html_export(...)` when the standalone artifact needs a
  packed/export-only widget state.
- Standard HTML export button traits listed above.
- Export status strings that include both file size and mode.

Do not force all widgets through one base class. The shared part is the public
shape; the data packing must remain widget-specific so full, uint8,
downsampled, and folder exports can be honest about precision, size, and
performance.
