# Show3D

A stack of 2D images scrubbed frame by frame, with playback, side-by-side
panels, paged galleries and standalone HTML and GIF exports. See the
[Show3D tutorial](../tutorials/show3d).

## Reference

```{eval-rst}
.. autoclass:: quantem.widget.show3d.Show3D
   :members:
   :show-inheritance:
```

## Interactive controls

| Control | Trait | Expected effect |
| --- | --- | --- |
| Slice slider, arrow keys | `slice_idx` | Canvas shows that frame |
| Play / pause, reverse, boomerang | `playing`, `reverse`, `boomerang` | Auto-advances at `fps`, flips direction, ping-pongs at the ends |
| FPS field | `fps` | Playback rate, capped at 60 fps |
| Loop range, bookmarks, custom path | `loop`, `loop_start`, `loop_end`, `bookmarked_frames`, `playback_path` | Playback confined to a range or an explicit frame order |
| Moving average | `avg_window` | Mean of 1 to 15 neighbouring frames; source data unchanged. Auto keeps the stack-wide contrast window, so averaging only reduces noise |
| Colormap dropdown | `cmap`, `panel_cmaps` | Shared by default; More > Color shared unlocks per-panel colormaps |
| Smooth toggle | `smooth` | Off by default: each data pixel is drawn as a sharp block (nearest neighbour). On: bilinear interpolation wherever the canvas enlarges a frame, its FFT or the kymograph. Display only; the data are unchanged |
| Contrast | `auto_contrast`, `percentile_low`, `percentile_high`, `vmin`, `vmax`, `vmin_per_panel`, `vmax_per_panel`, `contrast_preset`, `log_scale` | Percentile or fixed ranges, per stack or per panel |
| Panel layout (multi-panel) | `n_panels`, `link_panels`, `max_cols`, `panel_gap` | Panels arrange in rows; linked zoom and pan move all |
| Panel visibility and order | `hidden_panels`, `selected_panels`, `panel_order` | Hover actions collapse or reorder panels without touching data |
| Page controls (paged galleries) | `page_idx`, `n_pages`, `panels_per_page`, `page_starred`; `star_page()` | Shows and stars one page of panels at a time |
| Local panel annotations | `panel_annotations` | In-image labels per panel, placed by corner, point or box |
| Geometric panel overlays | `panel_overlays` | Circle, rectangle and square overlays in data or relative coordinates; More > Overlay Edit moves them live |
| Rich panel titles | `panel_titles`, `panel_title_spans` | Titles may carry `text` / `math` / `color` spans such as `λ` and `χ²` |
| Viewer chrome preset | `ui_mode` plus `show_title`, `show_stats`, `show_scale_bar` | Shared display presets; see [Viewer UI controls](viewer-ui) |
| Scale bar | `pixel_size`, `pixel_unit`, `scale_bar_visible` | Physical scale bar from `sampling` / `units` or a `Dataset3d`; the ROI preview scale bar, line-profile axis and FFT d-spacing use the same unit |
| Statistics | `show_stats` | Optional mean/min/max/std readout. Like the ROI statistics, line profile, FFT and cursor value it describes the frame as analysed (moving average, alignment, compare and difference applied), never the denoised or frequency-filtered image |
| FFT | `show_fft`, `fft_layout`, `fft_overlay_position`, `fft_overlay_size`, `fft_overlay_zoom`, `fft_window` | FFT of the current frame below, beside or inside every panel |
| Denoise | `denoise_enabled`, `denoise`, `denoise_sigma`, `denoise_bin`, `show_denoise` | Browser-side display filter; an active reduction prints a one-line banner. Changes the image only: the measurements read the unfiltered frame |
| Frequency filter | `frequency_filter_enabled`, `frequency_filter`, `frequency_filter_cutoff`, `frequency_filter_center`, `frequency_filter_width` | View-only low, high or band pass with a draggable FFT ring; the FFT shows the unfiltered spectrum, so the ring sits on the real peaks |
| More menu: Flip, Rotate | `flip_horizontal`, `flip_vertical`, `image_rotation`, `rotation_scope`, `frame_rotations` | Display-only orientation changes |
| More menu: Compare | `compare_mode`, `compare_pair`, `blink_fps`, `diff_cmap`, `compare_background`, `diff_mode` | Blink, difference or overlay two frames; diff against the first or previous frame |
| More menu: Sub-pixel align, Kymograph | `subpixel_align_enabled`, `subpixel_align_reference`, `show_kymograph` | Browser-side drift alignment and a space-time view of the drawn profile |
| ROI and line profile | `roi_active`, `roi_list`, `roi_selected_idx`, `profile_line`, `profile_width` | Single-panel ROI with a per-frame mean plot; line profile across the frame |
| Export button | `export_request`, `export_status` | Writes standalone HTML or GIF from the live widget |

## Large stacks

The browser plays every frame from one display stack sent in a single kernel
message, and one message holds at most 2048 MB. For a larger stack, Show3D
prints the smallest `display_bin` that fits, for example `Pass display_bin=2
(800 MB) for an interactive view.`, and the widget shows its static preview with
a status line saying that the display stack did not reach the browser and naming
the same `display_bin`. Nothing is binned unless you pass it.

## Live stack updates

Use `set_image()` to replace the stack in an already displayed widget. Keep a
reference to the widget, display it once, then call `set_image(...)` whenever
new frames should be rendered. Mutating the original array in place does not
notify the browser.

```python
import numpy as np
from quantem.widget import Show3D

frames = [first_frame]
w = Show3D(first_frame[None], labels=["frame 1"])
w

for next_frame in acquisition:
    frames.append(next_frame)
    w.set_image(np.stack(frames), labels=[f"frame {i + 1}" for i in range(len(frames))])
    w.slice_idx = len(frames) - 1
```

## Watch a growing frame folder

`Show3D.from_folder(...)` plays every matching image file as the next frame of
one stack and appends new files in place. `file_types` accepts one extension
or a list; `pattern` filters names; both apply when given. A missing folder is
created when `watch=True`. Supported files include EMD, TIFF, PNG, NumPy,
DM3/DM4, JPEG, BMP and GIF.

```python
w = Show3D.from_folder("/data/session/reconstruction", file_types="tif", pattern="frame_*", watch_interval=2.0)
new_frames = w.poll_folder()   # scan now; return newly appended indices
w.stop_folder_watch()          # pause background scans
w.watch_folder(interval=1.0)   # resume
w.close()                      # stop watching and close the widget
```

Folder watching is append-only: existing frames are never duplicated or
rewritten, incomplete files wait for a later poll, and an incompatible shape is
reported without blocking later frames. Use `Show2D.from_folder` when each file
should be its own gallery panel.

## Paged galleries

Pass a 5D array shaped `(pages, panels_per_page, frames, rows, cols)` when each
view is itself a small multi-panel movie, such as a regularization sweep:

```python
w = Show3D(
    stacks_5d,
    panel_titles=["raw", "filtered", "residual", "probe"],
    page_labels=["lambda 0.01", "lambda 0.03", "lambda 0.10"],
)
w.page_idx = 2
w.star_page(2)
```

Pages use independent percentile contrast by default because separate
reconstructions often have different ranges; pass `link_contrast=True` when
identical limits are required.

## Portable HTML export

`export_html` writes one standalone file. The default embeds the exact float32
display stack; `encoding="uint8"` writes a compact pack and accepts
`downsample=2`, `4` or `8`. Exports above 80 MB are refused with the smaller
options named; pass `max_mb=None` to force one.

```python
w.export_html("review.html", encoding="uint8", downsample=2)
```

## Animation exports

`save_gif` renders the visible panels with the current colormap,
contrast, titles and scale bars. `quality` picks the resolution tier
(`"high"`, `"medium"`, `"low"`); `frame_start`, `frame_stop`, `every_n` and
`max_frames` select frames; `playback="bounce"` plays forward then back;
`slides_preset=True` caps the export at 40 frames and 512 px per panel.

```python
w.save_gif("movie.gif", quality="medium", fps=8, slides_preset=True)
```

The widget **Export** menu offers the same GIF and HTML choices. For
array-first movies without a widget use `quantem.gpu.movie`.
