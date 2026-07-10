# `save_state`: keep widget notebooks small, still show the render

## The problem

Every quantem widget (Show2D / Show3D / Show4DSTEM) syncs its pixel buffers to the
frontend as `traitlets.Bytes(...).tag(sync=True)` traits. When a notebook is saved,
ipywidgets serializes those buffers into `metadata.widgets` as base64. A 5-panel 4k
Show2D gallery baked **~1 GB into a single `.ipynb`** - the pictures in the cell output
were only ~11 MB; the 1 GB was the embedded widget *state*.

That state buys almost nothing: **JupyterLab does not rehydrate a widget from saved
state on a cold reopen** - it re-renders from the kernel on Run. So the 1 GB was dead
weight except for the rare "reopen kernel-less and keep interacting" case.

## The contract

Every widget takes `save_state: bool = False`.

- **`save_state=False` (default)** - do not persist the bulk pixel buffers. Attach a
  **static `image/png`** of the current view to the cell output instead, so a cold
  reopen (GitHub, nbviewer, kernel-less Lab) still shows *how it looked*. Live Jupyter
  still renders the full interactive widget (richest mime wins).
- **`save_state=True`** - embed the full interactive state so a reopened notebook
  restores the live widget without a kernel. No static PNG (the widget restores itself).

## How it is implemented (mirror this for any new widget)

Six pieces, all in the widget class:

1. `save_state: bool = False` kwarg in `__init__` (before `**kwargs`).
2. `self._save_state = bool(save_state)` set **before** `super().__init__(**kwargs)`
   (so any `get_state` during comm-open already sees it).
3. `_UNSAVED_HEAVY_KEYS = (...)` - the bulk `sync=True` Bytes traits (frame buffers,
   prefetch buffers, offline stacks, export blobs). **Not** the tiny control-sized ones;
   keep a small current-frame/virtual-image buffer so a cold reopen gets a first paint.
4. `get_state(self, key=None, drop_defaults=False)` override:
   ```python
   state = super().get_state(key=key, drop_defaults=drop_defaults)
   if key is None and not getattr(self, "_save_state", False):
       for k in self._UNSAVED_HEAVY_KEYS:
           state.pop(k, None)
   return state
   ```
   **The `key is None` guard is load-bearing.** ipywidgets calls `get_state(None)` only
   for the full save/embed snapshot. The live-render path (`send_state`, `hold_sync`)
   calls it with a specific key or a set - those must pass through untouched, or the
   frontend never receives the pixels and the widget renders **blank**.
5. `_static_png_b64(...)` - a downsampled matplotlib render of the current view, base64.
   Stride-downsample; rendering full 4k here would dominate display time.
6. `_repr_mimebundle_` override - when `not self._save_state`, add
   `data["image/png"] = self._static_png_b64()` (handle the bundle being a
   `(data, metadata)` tuple or a dict).

### Per-widget heavy keys (as of this writing - verify against the source)

| widget | `_UNSAVED_HEAVY_KEYS` | static PNG source |
|---|---|---|
| Show2D | `frame_bytes`, `export_payload` | the N panels in `_data` |
| Show3D | `frame_bytes`, `_buffer_bytes`, `_offline_stack`, `_offline_float_stack`, `export_payload` | evenly-spaced slices of the z-stack |
| Show4DSTEM | `_offline_stack`, `export_payload`, `_gif_data` | the virtual (BF/ADF) image |

## Executing notebooks small (the other half)

The `get_state` trim covers the Python embed path, but two more things bite when you run
notebooks non-interactively:

- **`metadata.widgets` is sticky.** `nbconvert` does *not* clear an existing
  `metadata.widgets`, so a stale 1 GB blob rides along every rerun even with
  `store_widget_state=False`. **Clear it first** if a prior run bloated it:
  ```python
  nb["metadata"].pop("widgets", None)
  ```
- The bulk data also rides the **comm buffers** the widget sends to render (nbconvert
  records them). Run without persisting those:
  ```bash
  jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.store_widget_state=False <nb>
  ```
  With `save_state=False`, the static PNG in the cell output preserves the visual.

## Verifying a change to this code (the drive-test protocol)

A screenshot-only pass can miss a blank render, and a size-only pass can miss a broken
save. Verify **both halves**:

1. **Render path intact (browser-free, deterministic):** for `save_state=False`, assert
   `heavy_key in get_state(heavy_key)` (targeted survives) while
   `heavy_key not in get_state(None)` (full snapshot trimmed). This is the exact
   mechanism that delivers pixels to the frontend; if the targeted lookup keeps the key,
   live render works. (Show3D's `frame_bytes` is empty at construction because it streams
   on scrub - the key must still be *present*, just empty.)
2. **Visual oracle (browser-free):** execute on real data, extract the cell's
   `image/png`, and *look at it*. Confirm the panels are the real render, not blank.
3. **Live browser drive (when the environment allows headed Chrome):** open the notebook
   on `:1`, Run the cell, screenshot the mounted canvas, vision-confirm. This is the only
   check that exercises the real JS render path; do it when a real GPU + headed Chrome is
   available (see the `widget-drive` skill).

The regression shield for steps 1-2 lives in `tests/test_save_state.py` (5 invariants ×
3 widgets). Extend it whenever a new widget adopts `save_state`.
