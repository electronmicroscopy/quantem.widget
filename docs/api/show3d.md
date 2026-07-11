# Show3D

A 3D volume scrubbed slice by slice, with playback and an interactive-HTML
export. See the [Show3D tutorial](../tutorials/show3d).

## Reference

```{eval-rst}
.. autoclass:: quantem.widget.show3d.Show3D
   :members:
   :show-inheritance:
```

## Interactive controls

| Control | Trait | Expected effect |
|---|---|---|
| Slice slider | `slice_idx` | Canvas shows that depth slice |
| Arrow keys | `slice_idx` | Step one slice per press |
| Play / pause | `playing` | Auto-advances slices at `fps` |
| Reverse | `reverse` | Playback direction flips |
| Boomerang | `boomerang` | Ping-pongs at the ends instead of looping |
| FPS field | `fps` | Playback rate changes |
| Loop range | `loop_start`, `loop_end` | Playback confined to the sub-range |
| Colormap dropdown | `cmap` | Canvas recolors |
| Export button | `export_request`, `export_status` | Writes a standalone HTML viewer |
| Page controls (paged galleries) | `page_idx`, `n_pages`, `panels_per_page`, `page_starred`; `star_page()`, `unstar_page()` | Shows, stars, or plays through one page of panels at a time |
| Panel layout (multi-panel) | `n_panels`, `link_panels`, `max_cols` | Panels arrange; linked scrub moves all |
| Panel visibility (multi-panel) | `hidden_panels` | Panels collapse from view without deleting data |
| Panel reorder (multi-panel) | `panel_order`; `set_panel_order()`, `move_panel()`, `reset_panel_order()` | Reorders panel display without changing source data, labels, stars, or hidden state |
| Viewer chrome preset | `ui_mode` plus explicit `show_*` kwargs | Applies shared display presets; see [Viewer UI controls](viewer-ui) |
| Control visibility | `show_controls`, `controls_collapsed`; `collapse_controls()`, `expand_controls()`, `toggle_controls()` | Permanently remove controls or temporarily collapse them behind the top GUI toggle |
| Title visibility | `show_title` | Top title row shows/hides |
| Statistics | `show_stats` | Optional mean/min/max/std readout |
| Panel title visibility | `show_panel_titles`, `panel_title_font_size` | Per-panel labels show/hide and resize |
| Scale bar visibility | `show_scale_bar` (`scale_bar_visible` in saved state) | Scale bar shows/hides |
| FFT toggle | `show_fft` | Shows the FFT view for the current frame or visible panel grid |
| FFT quality labels | `fft_metrics` | Compact in-panel label reports FFT sharpness, peak count, and peak SNR from the cached FFT magnitude |
| FFT window toggle | `fft_window` | Apodization on/off before FFT rendering |
| Resize / zoom chrome | `show_resize_handles`, `show_zoom_indicator` | Resize handles and zoom readouts show/hide; the zoom setting covers every real-space panel and FFT tile/inset |
| FFT layout and initial view | `fft_layout`, `fft_overlay_position`, `fft_overlay_size`, `fft_overlay_zoom` | Places FFTs below, right, or inside every panel and initializes their shared zoom |

## FFT quality labels

Pass `show_fft=True` to show the FFT view. By default, `fft_metrics=True`
adds a small white label inside the FFT panel with sharpness, peak count, and
peak SNR. In multi-panel FFT views, Show3D summarizes the visible FFT tiles.
The metrics reuse the cached FFT magnitude used for rendering, so frame
playback, zoom, and pan do not trigger an extra FFT for the label. Set
`fft_metrics=False` for a clean FFT image.

The first FFT for a frame or ROI may take a moment on large data. After that,
Show3D reuses the cached FFT magnitude when you return to the same frame and
when you redraw, zoom, pan, scrub, or show metric labels.

Every visible FFT tile or overlay inset shows the shared live magnification as
an `N.N×` badge, even for uncalibrated arrays. Wheel or pinch zoom updates it;
double-click, double-tap, or Reset returns to `1.0×`. Pass
`fft_overlay_zoom=2.0` to initialize any FFT layout at `2.0×`, and set
`show_zoom_indicator=False` to hide both real-space and FFT zoom badges.

## Live stack updates

Use `set_image()` to replace the stack in an already displayed widget while a
notebook kernel is still running. Keep a reference to the widget, display it
once, then call `set_image(...)` whenever new stack data should be rendered.
For live acquisitions or reconstruction loops, construct the widget with
`offline=False` so frames travel over the live Jupyter Comm channel instead of
the saved/offline notebook-data path:

```python
import numpy as np
from quantem.widget import Show3D

frames = [first_frame]
w = Show3D(first_frame[None], labels=["frame 1"], offline=False)
w

for next_frame in acquisition:
    frames.append(next_frame)
    w.set_image(
        np.stack(frames),
        labels=[f"frame {i + 1}" for i in range(len(frames))],
    )
    w.slice_idx = len(frames) - 1
```

In a real JupyterLab browser session this updates the displayed frame as each
`set_image()` call is processed. A background thread is optional for UI
ergonomics, but is not required for the widget update itself.

`set_image()` is the re-render trigger. Mutating the original NumPy array in
place does not notify the frontend. The method writes a fresh current-frame
transfer, bumps `frame_seq`, invalidates playback buffers, clamps the current
slice and loop range, and resets stale panel-specific state when replacing a
multi-panel view with a single stack.

```{important}
Do not use the default tiny-stack constructor path for acquisition-style live
updates. Small stacks may auto-enable the offline notebook representation, which
is intended for saved notebooks and static exports. Pass `offline=False` when the
stack will grow over time.
```

## Watch a growing frame folder

Use `Show3D.from_folder(...)` when each matching image file is the next frame
in one time series, focal series, or reconstruction history. New files append
to the single displayed stack; they do not create additional gallery panels or
replace frames that are already loaded.

```python
from quantem.widget import Show3D

w = Show3D.from_folder(
    "/data/session/reconstruction",
    pattern="frame_*.tif",
    watch_interval=2.0,
    title="Live reconstruction",
)
w
```

`watch=True` is the default. The first folder scan establishes deterministic
frame order, then the watcher appends newly readable files without rebuilding
the widget or rereading unchanged source files. Use Show2D instead when each
file should be a separate comparison panel. An empty watched folder stays
mounted and changes into the real stack in the same widget model after the
first stable frame. The compact title-area badge reports `Watching`,
`Updating`, `Waiting for file completion`, `Watch error`, or `Stopped`; fixed
`watch=False` snapshots do not show it.

```python
new_frames = w.poll_folder()       # scan now; return newly appended indices
w.stop_folder_watch()             # pause background scans
w.watch_folder(interval=1.0)      # resume with a different interval
w.close()                         # stop watching and close the widget
```

Folder watching is append-only. Files already represented in the stack are not
duplicated, incomplete files wait for a later poll, and source removals or
rewrites do not alter existing frames silently. An incompatible shape is
reported without blocking a later compatible frame. Pass `watch=False` when a
fixed folder must remain fixed.

`Show3D.from_folder(...)` reads full-resolution source frames. `ShowFolder`
uses cached thumbnails to browse and select a session quickly; those thumbnails
are not the data used by the folder-backed Show3D stack.

Maintainer real-time signoff follows
[S3D-17](../maintainer/storyboard-show3d.md#s3d-17-watch-a-live-emd-frame-series-in-place):
append genuine EMD frames after the stack is visibly mounted and verify the
same browser canvas, playback state, and frame controls update.

## Panel visibility

Use panel visibility when a secondary panel is useful for validation but should
not take space in the first view. For example, an SSB reconstruction can keep
the mean diffraction pattern in the widget while hiding it from the canvas:

```python
w = Show3D(
    ssb_stack,
    mean_dp_stack,
    panel_titles=["SSB reconstruction", "Mean DP"],
    hidden_panels=["Mean DP"],
)
```

Panel references can be zero-based indices or exact panel titles:

```python
w.hide_panel("Mean DP")
w.hide_panel(1)
w.show_panel("Mean DP")
w.show_all_panels()
```

Hidden panels stay in the widget state and standalone HTML export. They are not
removed from the data, and readers can restore them from the `Panels` menu.

Use panel reordering when the comparison order should change without copying or
rebuilding the source stacks:

```python
w.set_panel_order(["Probe", "SSB reconstruction", "Mean DP"])
w.move_panel("SSB reconstruction", 0)
w.reset_panel_order()
```

Panel order is saved in widget state and standalone HTML. It is display-only:
hidden panels, stars, titles, and per-panel contrast remain keyed by the
original source panel index.

The statistics readout is off by default. Turn on `show_stats=True` in Python,
or use the `Stats` switch in the widget, when mean/min/max/std values are useful.

## Paged galleries

Use pages when each view is itself a small multi-panel movie. This is useful for
reconstruction sweeps, denoising parameters, or iteration checkpoints where each
page should show the same panel layout while the user scrubs pages:

```python
# Shape: pages, panels_per_page, frames, rows, cols
w = Show3D(
    stacks_5d,
    panel_titles=["raw", "filtered", "residual", "probe"],
    page_labels=["lambda 0.01", "lambda 0.03", "lambda 0.10"],
)
```

You can also pass explicit page dictionaries:

```python
w = Show3D([
    {"title": "iteration 10", "stacks": [raw_10, filtered_10, residual_10]},
    {"title": "iteration 20", "stacks": [raw_20, filtered_20, residual_20]},
])
```

Paged Show3D keeps the data in the normal multi-panel transport internally, so
HTML export, notebook state, panel hiding, panel stars, playback, FFT, and GIF/MP4
export use the same paths as ordinary Show3D. In page mode, `visible_panels`
returns only panels from the active page, and `to_show2d()` converts the current
visible page into a Show2D gallery. The page row has its own play/pause button
and FPS menu; the lower frame playback controls still scrub time or depth
inside the active page. Manual page scrubbing pauses page playback, and Show3D
keeps rendering work scoped to the active visible page.

Paged views use independent automatic percentile clipping by default because
separate reconstructions often have different numerical ranges. This preserves
structure on every page instead of letting one high-amplitude reconstruction
set the contrast for all others. Pass `link_contrast=True` when identical color
limits are scientifically required for direct amplitude comparison. Ordinary
non-paged multi-panel views keep linked contrast by default.

```python
w.page_idx = 2
w.star_page(2)
show2d_page = w.to_show2d(frame=w.slice_idx)
```

### Compare depth profiles across pages

When every page contains one panel, the line-profile kymograph follows the
active page. Draw the profile once, enable **Kymograph**, and scrub the page row
to compare the same spatial line through raw/corrected volumes, reconstruction
methods, or experimental conditions. The kymograph title includes the active
page label; its horizontal line coordinate and depth/time axis stay matched.

```python
# Shape: pages, 1 panel per page, depth, rows, cols
comparison = np.stack([single_slice, multislice], axis=0)[:, None]
w = Show3D(
    comparison,
    page_labels=["single-slice", "multislice"],
    panel_titles=["object phase"],
    dim_label="Depth",
)
w.set_profile((row0, col0), (row1, col1))

# Numerical form: page, depth, distance along the line
matched_profiles = w.profile_all_pages()
```

For pages containing several panels, pass `panel_slot=` to
`profile_all_pages()` or `panel=` to `profile_all_frames()` when extracting a
specific numerical profile. The interactive kymograph intentionally remains a
single-panel-page tool so its line and depth axes are unambiguous.

## Animation exports

Use HTML when collaborators should keep scrubbing, zooming, and changing
contrast. Use GIF or MP4 when the result needs to drop into PowerPoint, email, or
a static report:

```python
w.save_gif("movie.gif", quality="medium", fps=6)
w.save_mp4("movie.mp4", quality="high", fps=12)
```

`quality="low"`, `"medium"`, and `"high"` control the exported spatial
resolution. GIF is always palette-limited, so medium is usually the practical
slide/email choice; high is sharper but larger. Pass `show_frame_labels=True`
when panel titles should include the same live-style frame label and count that
the widget canvas shows. GIF/MP4 exports keep the panel labels, scale bar, and
zoom readout styling consistent with the static/offline widget image output.
The widget **Export** menu keeps the common path simple: choose `GIF low`,
`GIF medium`, `GIF high`, or the matching MP4 option. The size shown in that
menu is estimated uncompressed RGB render work, so the final GIF/MP4 file is
usually smaller but can vary with image texture and palette compression.

```python
w.save_gif("movie.gif", quality="medium", fps=6, show_frame_labels=True)
```

The GIF/MP4 path exports the full panel frames. Browser-only zoom and pan
gestures are view state, so use HTML export when collaborators need to continue
zooming, panning, or changing contrast interactively.

Advanced animation choices stay in Python and the maintainer smoke report
rather than crowding the widget toolbar. Use the Python API for frame labels,
background color, bounce playback, and other presentation-specific choices:

```python
w.save_gif(
    "movie.gif",
    quality="medium",
    fps=6,
    playback="bounce",
    show_frame_labels=True,
    background="black",
)
```

```{note}
`export_html(quantized=True)` writes the smaller uint8 pack; the default writes
exact float32. See the [widget export tutorial](../tutorials/widget_export).
```
