# Saved widget state and notebook size

## Why saved state matters

Image widgets synchronize pixel buffers to the frontend as traits. Saving their
full state in `metadata.widgets` can make a notebook much larger than its cell
outputs: a five-panel 4k Show2D gallery previously embedded about 1 GB of state.
Some notebook frontends cannot restore that state on a cold reopen, while an
HTML export with a widget manager can use it. Choose the saved artifact for its
reader rather than embedding every buffer by default.

## The current protocols

Widgets that use `SavedStateMixin` and `StaticFallbackMixin`, including Show1D,
Show2D, Show3D, Show4DSTEM and ShowPDF, support these choices:

- **`save_state=False` (default):** trim bulk pixel buffers from the saved full
  state and provide a static preview where configured. The live widget still
  receives its pixels. The configurable preview defaults to JPEG; WebP and PNG
  are available through `notebook_preview_format`.
- **`save_state=True`:** retain the synchronized state and omit the static
  preview. A compatible widget manager can restore interactions supported by
  the embedded data. This does not restart a Python folder watcher, preserve a
  GPU acquisition owner, or provide computations that require a live kernel.

These are opt-in protocols, not parameters guaranteed on every widget.
ChooseLattice has no `save_state` argument. Show3DSlices and ShowDiffraction use
state-file and HTML-export protocols without the static-preview mixins. Plot2D
uses the shared mixins with a fixed PNG preview. Check the widget's current
constructor and export API before choosing a persistence mode.

The preview is a display artifact, not calibrated numerical data. Use the
widget's explicit export method or the scientific data owner to save values.

## Reuse the shared implementation

The implementation is in
[`state.py`](https://github.com/electronmicroscopy/quantem.widget/blob/main/src/quantem/widget/state.py)
and [`fallback.py`](https://github.com/electronmicroscopy/quantem.widget/blob/main/src/quantem/widget/fallback.py).
For a new widget adopting these protocols:

1. Follow a current participating widget's initialization and mixin order.
   Set `_save_state` before opening the comm and configure the preview format,
   quality and size through the shared fallback implementation.
2. Declare `_UNSAVED_HEAVY_KEYS` for that widget's large synchronized buffers.
   Keep small control state and use the shared initial-mount handling so the
   first live view receives its data.
   ShowPDF, for example, drops `nav_image_bytes`, `mask_b64`, `mask_bytes`,
   `linescan_bytes` and `export_payload`, and its preview is the scan image
   beside the shown curve or line-scan stack.
3. Supply the widget-specific `_static_png_b64` renderer. The shared fallback
   converts that render to the selected preview format and manages its display.
   Bound the preview dimensions so it remains inexpensive.

Do not copy `get_state` or MIME/display plumbing into each widget. The shared
implementation distinguishes a full saved snapshot from the initial live mount
and targeted updates. In particular, `get_state(key=<name or set>)` must retain
requested buffers: `send_state` and `hold_sync` use those updates to draw the
live view. Saved folder-watcher state is marked stopped because a notebook
snapshot cannot retain a running Python thread.

## Interactive documentation and static copies

Tracked tutorial notebooks contain source without saved `metadata.widgets` or
image outputs. The documentation build executes them afresh and captures the
interactive state in the generated site. Keep that workflow intact; disabling
state capture during the docs build can leave blank or static-only tutorials.

For a separate static GitHub preview, make a copy outside `docs/tutorials/` and
follow [GitHub preview copy](../cli.md#github-preview-copy). Never run
`quantem github` in place on canonical documentation notebooks.

For a local execution that deliberately needs only static previews, copy the
notebook first. Clear any existing `metadata.widgets` from that copy before
running it: `store_widget_state=False` does not remove old embedded state.
Then execute the copy without capturing new widget state:

```bash
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.store_widget_state=False analysis_static.ipynb
```

Inspect the result: static previews exist only for widgets supporting that
protocol, and frontends differ in their handling of image and widget outputs.
This command does not certify an interactive documentation build.

## Verify both saving and live rendering

1. **State contract:** with `save_state=False`, verify full saved state omits
   declared heavy buffers while targeted key and key-set updates retain them.
   Verify the initial mount separately, plus full state when `save_state=True`.
2. **Static preview:** decode the configured image MIME output and inspect it.
   Check JPEG, WebP or PNG as applicable; do not assume every widget emits PNG.
3. **Live browser:** execute the notebook, inspect the mounted widget, change a
   control and confirm pixels update. Source-only assertions cannot prove the
   JavaScript render path. Also check a cold reopen and the intended HTML export.

[`tests/test_save_state.py`](https://github.com/electronmicroscopy/quantem.widget/blob/main/tests/test_save_state.py)
checks participating widgets, preview formats and saved-state behavior. Extend
protocol tests when adding a participant, and run the repository's notebook
size guards before publishing. A small notebook and a rendered screenshot alone
do not establish that its live interaction works.
