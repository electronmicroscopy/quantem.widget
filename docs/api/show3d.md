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
| FPS field | `fps` | Playback rate changes, capped at 60 fps |
| Loop range | `loop_start`, `loop_end` | Playback confined to the sub-range |
| Playback dynamics | `fps`, `loop`, `boomerang`, `playback_path`; future `playback_preset` | More-menu presets can play a time series linearly, bounce at ends, slow down around key frames, or follow a custom frame path without changing the underlying stack |
| Colormap dropdown | `cmap`, `panel_cmaps` | Shared by default. More → Color shared can unlock per-panel colormaps for advanced comparisons |
| Panel identity markers | `marker_colors`, `identity_colors`, `marker_style`, `row_markers`, `col_markers`, `panel_groups` | Optional panel colors plus row, column, or rectangular group frames make panels easy to reference in notebooks, reports, and agent instructions |
| Local panel annotations | `panel_annotations` | Optional multiple in-image labels per panel, placed by corner, normalized point, or normalized region box |
| Geometric panel overlays | `panel_overlays` (`overlays` shorthand) | Optional circle, rectangle, and square overlays in data or relative coordinates, with stroke/fill opacity and z-order, rendered while scrubbing and in exported HTML; More -> Overlay Edit allows live select, move, resize, delete, and reset |
| Contrast preset dropdown | `contrast_preset` | Applies percentile ranges such as `1-99`, `2-98`, and `3-97` while keeping the main histogram UI compact |
| Advanced histogram toggle | `show_histogram_advanced`, `histogram_advanced` | Exposes detailed histogram controls only when a user asks for them |
| Export button | `export_request`, `export_status` | Writes standalone HTML from live widgets; live Show3D can also request GIF/MP4 animation exports |
| Page controls (paged galleries) | `page_idx`, `n_pages`, `panels_per_page`, `page_starred`; `star_page()`, `unstar_page()` | Shows, stars, or plays through one page of panels at a time |
| Panel layout (multi-panel) | `n_panels`, `link_panels`, `max_cols` | Panels arrange; linked scrub moves all |
| Panel visibility (multi-panel) | `hidden_panels` | Panels collapse from view without deleting data |
| Panel reorder (multi-panel) | `panel_order`; `set_panel_order()`, `move_panel()`, `reset_panel_order()` | Reorders panel display without changing source data, labels, stars, or hidden state |
| Viewer chrome preset | `ui_mode` plus explicit `show_*` kwargs | Applies shared display presets; see [Viewer UI controls](viewer-ui) |
| Control visibility | `show_controls`, `controls_collapsed`; `collapse_controls()`, `expand_controls()`, `toggle_controls()` | Permanently remove controls or temporarily collapse them behind the top GUI toggle |
| Title visibility | `show_title` | Top title row shows/hides |
| Statistics | `show_stats` | Optional mean/min/max/std readout |
| Panel title visibility | `show_panel_titles`, `panel_title_font_size`, `panel_title_style` | Per-panel labels show/hide, resize, and optionally get title chrome such as background, border, and padding |
| Rich panel title spans | `panel_title_spans` | Optional structured `text` / `math` / `color` spans for symbols such as `λ` and `χ²` in panel titles, panel menus, stats, animation labels, saved state, and exported HTML |
| Scale bar visibility | `show_scale_bar` (`scale_bar_visible` in saved state) | Scale bar shows/hides |
| Saved notebook preview frames | `notebook_preview_frames`, `notebook_preview_ncols`; `set_notebook_preview_frames()` | Single-panel saved notebooks can reopen as a compact contact sheet of selected frame indices instead of only the current frame |
| ROI add / drag | `roi_active`, `roi_list`, `roi_selected_idx`; `get_roi_geometries()` | Single-panel stack ROI overlays stay visible while scrubbing; saved notebook previews include all visible ROI overlays and right-side zoom crops; Python can read circle centers/radii and rectangle/square corners in `(row, col)` coordinates |
| FFT toggle | `show_fft` | Shows the FFT view for the current frame or visible panel grid |
| FFT quality labels | `fft_metrics` | Compact in-panel label reports FFT sharpness, peak count, and peak SNR from the cached FFT magnitude |
| FFT window toggle | `fft_window` | Apodization on/off before FFT rendering |
| Resize / zoom chrome | `show_resize_handles`, `show_zoom_indicator` | Resize handles and zoom readouts show/hide; the zoom setting covers every real-space panel and FFT tile/inset |
| FFT layout and initial view | `fft_layout`, `fft_overlay_position`, `fft_overlay_size`, `fft_overlay_zoom` | Places FFTs below, right, or inside every panel and initializes their shared zoom |
| Denoise | `denoise_enabled`, `denoise`, `denoise_sigma`, `denoise_bin`, `show_denoise` | The master swaps raw/denoised frames without losing settings; Settings expands the Method/σ/bin editor; an active filter also reshapes FFT |
| Filter | `frequency_filter_enabled`, `frequency_filter`, `frequency_filter_cutoff`, `frequency_filter_center`, `frequency_filter_width`, `show_frequency_filter` | View-only low/high/band-pass filtering with a draggable FFT ring; stored frames, statistics, and raw exports remain unchanged |
| Sub-pixel alignment | `subpixel_align_enabled`, `subpixel_align_reference` | More-menu display alignment for a drifting single-panel stack; the browser estimates row/column shifts against the reference frame and keeps raw data unchanged |
| More menu: Flip | `flip_horizontal`, `flip_vertical`, `flip_rows`, `flip_cols` | Display-only orientation checks for row/column or horizontal/vertical review |
| More menu: Rotate | `image_rotation`, `rotation_scope`, `frame_rotations`; `rotation=`, `rotations=` | Display-only 0/90/180/270° rotation for the whole stack or the selected frame |
| More menu: Compare | `compare_mode`, `compare_pair`, `blink_fps`, `diff_cmap`, `compare_background` | Blink, difference, or overlay two frames for point-defect and time-series change detection |

## Rich panel labels and math

Use rich labels when panel titles are scientific variables rather than plain
names. Show3D accepts inline math strings or structured spans in
`panel_titles` / `panel_title_spans`:

```python
from quantem.widget import Show3D

Show3D(
    raw_stack,
    residual_stack,
    panel_titles=[
        [{"math": r"\lambda=0.03"}, {"text": " raw"}],
        [{"math": r"\chi^2"}, {"text": "/pixel residual"}],
    ],
    show_stats=True,
)
```

Plain strings with `$...$` also work:

```python
Show3D(
    raw_stack,
    residual_stack,
    panel_titles=[r"$\lambda=0.03$ raw", r"$\chi^2$/pixel residual"],
)
```

The frontend renders common Greek symbols plus simple superscripts/subscripts
without MathJax or KaTeX. It also normalizes doubled backslashes from JSON/state
files, so `\\lambda` displays as `λ`. Panel titles, the `Panels` menu, stats
rows, animation labels, saved notebook state, and exported HTML all use the
same rich rendering path.

Use `panel_annotations` when a label belongs to a local feature or region
inside a panel. Each annotation can use `text`, `math`, or `spans`:

```python
Show3D(
    raw_stack,
    residual_stack,
    panel_titles=[
        [{"math": r"\lambda=0.03"}, {"text": " raw"}],
        [{"math": r"\chi^2"}, {"text": "/pixel"}],
    ],
    panel_annotations=[
        {"panel": 0, "math": r"\lambda", "position": "top-left", "variant": "pill"},
        {
            "panel": 1,
            "spans": [{"math": r"\chi^2"}, {"text": " high", "color": "#f87171"}],
            "box": [0.56, 0.48, 0.32, 0.16],
            "variant": "callout",
        },
    ],
)
```

## Circle and rectangle overlays

Show3D accepts the same `panel_overlays` API as Show2D. Use it for
reproducible geometry that should stay fixed while a stack scrubs or plays.
Coordinates are data pixels by default and follow `(row, col)` ordering.

```python
from quantem.widget import Show3D

Show3D(
    raw_stack,
    denoised_stack,
    residual_stack,
    panel_titles=["raw", "denoised", "residual"],
    panel_overlays={
        "raw": {
            "shape": "circle",
            "center": (96, 88),
            "radius": 14,
            "stroke": "#60a5fa",
            "stroke_width": 3,
        },
        "denoised": [
            {
                "shape": "rect",
                "box": (48, 58, 126, 146),
                "stroke": "#facc15",
                "fill": "#facc15",
                "fill_opacity": 0.12,
            },
            {
                "shape": "square",
                "center": (96, 88),
                "size": 42,
                "stroke": "#34d399",
                "z_order": 1,
            },
        ],
    },
)
```

Pass `overlays=[...]` to broadcast the same geometry to every panel, or add
`panel="raw"` / `panel=0` to each flat-list entry when building the overlay
list programmatically. Set `coords="relative"` for normalized 0-1 geometry
that follows panels with different pixel shapes.

When overlays are present, the live widget and exported HTML show
`More -> Overlay Edit`. Turn it on to select an overlay, drag inside the shape
to move it, drag an edge to resize it, press Delete to remove the selected
overlay, or choose `Reset Overlays` to restore the constructor state. Use ROIs
instead when the geometry should drive statistics, FFT crops, or Python
readback through `get_roi_geometries()`.

## Presentation and export chrome

`ui_mode="presentation"` starts Show3D with controls collapsed while preserving
the top `Controls` button. The `Export` button also remains visible in that
collapsed title chrome:

```python
w = Show3D(
    raw_stack,
    residual_stack,
    panel_titles=[r"$\lambda=0.03$ raw", r"$\chi^2$/pixel residual"],
    ui_mode="presentation",
)
w.export_html("lambda_movie.html", encoding="uint8")
```

In a live Python-backed widget, the Export menu can write HTML and request
GIF/MP4 animation exports through the Python backend. In standalone HTML, the
page can download itself as HTML; GIF/MP4 entries remain visible but disabled
with a backend-required explanation until browser-side animation encoding is
implemented.

The denoise family matches Show2D. See
[Which denoise filter should I use?](show2d.md#which-denoise-filter-should-i-use)
for the recommendation ladder (sparse EDS, very sparse, decent-dose HAADF,
quantitative). An active denoise reshapes the FFT magnitude too, so set
`denoise="none"` for quantitative FFT work.

Frequency Filter follows the same scientist-facing contract as Show2D while
remaining a separate control from Denoise. For example, a reconstruction stack
with a slow background can start as `Show3D(stack, show_fft=True,
frequency_filter="highpass", frequency_filter_cutoff=0.08)`. A lattice stack
can use `"bandpass"` with `frequency_filter_center` and
`frequency_filter_width`. Values are fractions of Nyquist from 0 to 1, and the
cyan FFT ring can be dragged during scrubbing or playback. The browser applies
Denoise first and Filter second; the raw stack and quantitative exports are
never replaced by the view. Filter lives under More and is off by default. Its
FFT overlay dims rejected frequencies and labels the clear region as Inside
kept, Outside kept, or Band kept.

## Sub-pixel alignment for drifting stacks

For a single-panel stack that is already available in the browser, use
`subpixel_align=True` or open **More → Sub-pixel align**. The first version
aligns the displayed frames to `subpixel_align_reference` (default frame 0) with
phase-correlation registration and bilinear display shifts. It is intentionally
view-only: the original stack, raw exports, and quantitative data remain
unchanged.

The More menu reports the reference frame, the approximate row/column padding
implied by the shifts, and whether WebGPU or CPU FFT was used. Multi-panel, RGB,
or streamed-only stacks currently show an explanatory status instead of
silently aligning the wrong data.

## Playback dynamics for time-series review

Show3D playback is not only a convenience for movies. For experimental
time-series data, a scientist often wants to probe temporal behavior: a defect
appears slowly, a reconstruction accelerates, a relaxation comes back, or only
a short sub-range matters. The lightweight API already exposes the core pieces:
`fps`, `loop`, `boomerang`, `loop_start`, `loop_end`, and `playback_path`.

The UI path is a compact **More → Playback Dynamics** section, not another
crowded toolbar row. The presets write those existing traits, so the same
behavior remains scriptable and serializable:

| Preset | Intended user behavior |
|---|---|
| Linear | Step forward through the loop range at constant `fps` |
| Slow | Lower `fps` for careful inspection of subtle frame-to-frame changes |
| Bounce | Ping-pong with `boomerang=True` so reversible dynamics are easy to see |
| Focus range | Set `loop_start` / `loop_end` around an event and play only that interval |
| Hold key frames | Populate `playback_path` with repeated important frames so the eye can settle |
| Custom path | Let a notebook or agent provide exact frame indices for non-uniform timing |

The stored frame stack is never resampled or duplicated. These controls only
change the order and cadence of frame display. Exported HTML and saved widget
state should preserve the selected playback dynamics, while a reopened notebook
should not unexpectedly start playing unless the user explicitly requests that
behavior.

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

## Reuse ROI coordinates across a stack

Single-panel Show3D uses the same ROI state contract as Show2D. ROIs are synced
in `roi_list`, saved by `state_dict()`, and exposed through
`get_roi_geometries()` for analysis code or agents:

```python
w = Show3D(stack).set_roi(row=72, col=65, radius=12)
roi = w.get_roi_geometries()[0]
roi["center"]  # {"row": 72.0, "col": 65.0}
roi["radius"]  # 12.0
```

Rectangle and square ROIs include clockwise `corners`; annular ROIs include
`radius_inner` and `radius_outer`. Saved notebook previews of a single-panel
Show3D frame show all visible ROI overlays and one zoom crop per visible ROI,
so a report reader can compare the marked sites without rerunning the notebook.

## Save a contact sheet of selected frames

By default, a lightweight saved notebook preview shows the current Show3D
frame. For single-panel stacks, pass explicit zero-based frame indices when a
cold-reopen report should show several representative frames:

```python
w = Show3D(
    stack,
    notebook_preview_frames=[0, 12, 25, 80],
    notebook_preview_ncols=3,
)
```

A notebook can also decide this interactively before saving:

```python
w.set_notebook_preview_frames([0, w.slice_idx, stack.shape[0] - 1], ncols=3)
```

Multi-panel Show3D intentionally keeps the saved preview to the current frame
of each visible panel; combining many movie panels with many saved frames is
better handled as an explicit report or animation export.

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
file should be a separate comparison panel.

Folder size never creates Show3D pages. Whether the folder has 2, 20, or 200
files, each file extends the frame axis of one stack and the frame
slider/playback controls remain its navigation. `page_size=` and `page_labels=`
are therefore rejected by `Show3D.from_folder(...)`; use
`Show2D.from_folder(..., page_size=20)` for independent paged images. Explicit
5-D or list-of-page Show3D comparison data remain a separate constructor mode.

An empty watched folder stays
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

Use `panel_groups` when several panels should be read as one comparison block.
The group box follows the displayed panel positions, so it still works after
panel hiding or reordering:

```python
w = Show3D(
    bf_raw,
    df_raw,
    bf_denoised,
    df_denoised,
    panel_titles=["BF raw", "DF raw", "BF denoised", "DF denoised"],
    max_cols=2,
    panel_groups=[
        {"panels": [0, 1], "color": "#2563eb", "label": "raw"},
        {"start": 2, "end": 3, "color": "#16a34a", "label": "denoised"},
    ],
)
```

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

## Full-resolution folder export (advanced)

Use the folder export when a microscopist needs to scrub a large stack at the
native detector/reconstruction size instead of opening a reduced one-file HTML.
This writes a small `index.html` beside an `offline_stack.u8` data file and a
`manifest.json`:

```python
from quantem.widget import Show3D

w = Show3D(stack, title="800C 1.3Mx full-resolution review", debug=True)
w.export_sidecar("/data/reports/800C_1.3Mx_fullres")
```

Serve the folder over Range-capable local HTTP; do not open the HTML with
`file://` because the browser must fetch the data file:

```bash
# Use your project or lab helper that supports HTTP Range requests.
python scripts/serve_sidecar_range.py \
  --dir /data/reports/800C_1.3Mx_fullres \
  --port 8803 --bind 127.0.0.1
```

Then open `http://127.0.0.1:8803/index.html`. The viewer shell should appear
immediately. The browser then loads the full stack into memory, shows the load
status banner, and swaps to the cached playback path when the stack is ready.
Changing Color or the histogram range repaints the current microscope view
immediately, marks the playback cache as updating, and rebuilds the remaining
frames in the background. Scrubbing during that rebuild should still advance
the current frame; once the banner clears, playback uses the cached full-stack
path again.

This workflow preserves the source spatial shape used to construct the widget.
If you intentionally want a smaller browse artifact, make that explicit with a
separate downsampled/single-file export rather than treating it as the
full-resolution review copy. For the end-to-end browser checklist and example
timing report, see the [advanced tutorial](../tutorials/advanced.md).

## Animation exports

Use HTML when collaborators should keep scrubbing, zooming, and changing
contrast. Use GIF or MP4 when the result needs to drop into PowerPoint, email, or
a static report:

```python
w.save_gif("movie.gif", quality="medium", fps=6)
w.save_mp4("movie.mp4", quality="high", fps=12)
```

For array-first workflows that do not need to construct a widget, use
`quantem.widget.movie.save_gif(...)` or `quantem.widget.movie.save_mp4(...)`.

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
`export_html(quantized=True)` writes the smaller single-file uint8 pack; the
default writes exact float32 into one HTML file. For multi-GB Show3D reviews,
use the folder export above instead of forcing one huge HTML file. See the
[widget export tutorial](../tutorials/widget_export).
```
