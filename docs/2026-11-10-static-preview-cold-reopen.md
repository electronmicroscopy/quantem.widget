# Static preview on a cold reopen (2026-11-10)

## Question

The widgets publish one render three ways: an image MIME inside the widget-view bundle, a separate `display_data`
sibling, and an in-model `_static_fallback_image` trait. `fallback.py` said JupyterLab never rehydrates a widget on a
cold reopen, while its own `_store_static_fallback_preview` docstring and `pyproject.toml` said it does. Can the three
collapse to one?

## Setup

JupyterLab 4.6.3, jupyterlab_widgets 3.0.16, anywidget 0.11.0, ipywidgets 8.1.8, headed Chrome. One notebook: a
Show2D with `save_state=False` and one with `save_state=True`. Run all cells in Lab, save, stop the server, start it
again, open the notebook without running a cell, read the DOM. Done once with Lab's "Save Widget State
Automatically" (`@jupyter-widgets/jupyterlab-manager` `saveState`) at its default (off) and once on.

## Result

| Lab `saveState` | Cell | Saved file | Cold reopen shows |
|---|---|---|---|
| off (default) | `save_state=False` | widget-view output + sibling `image/jpeg`; no `metadata.widgets` | "model not found", the sibling image |
| off (default) | `save_state=True` | widget-view output only; no `metadata.widgets` | "model not found", nothing else |
| on | `save_state=False` | `metadata.widgets` with the 60 kB in-model preview | the rehydrated model showing the in-model preview |
| on | `save_state=True` | `metadata.widgets` with the pixel buffers | the interactive widget |

The image MIME inside the widget-view bundle rendered in no case.

## Conclusion

Within JupyterLab the sibling is the only image under the default setting, and the in-model preview is the only one
when widget state is saved, so neither can go. The in-bundle image is not used by Lab; whether GitHub, nbviewer or
VS Code use it was not checked here, so it stays. No path was removed. `save_state=True` reopens empty under Lab's
default setting: it needs "Save Widget State Automatically" on.
