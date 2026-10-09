# quantem.widget slim refactor (2026-10-06)

Branch `widget-slim`, built from `54334d7` (GitHub main) in five steps: the package
slim (step 1, `6c72430`), the shared modules (`slim-shared`), and one package per
widget (`slim-show2d`, `slim-show4dstem`, `slim-show3d`), merged in that order.

Rule applied per widget: a constructor parameter or public method stays only if a
tutorial (`docs/tutorials/*.ipynb` code cells or `docs/tutorials/*.md` code blocks),
another widget, or `quantem.live` / `denoise` / `quantem.thick` source uses it.
Debug, benchmark and timing hooks go regardless. Shared plumbing lives once, in
`export.py`, `state.py`, `fallback.py`, `pages.py`, `panels.py`, `colormap.py` and
`render/figure.py` (Python) and `js/shared/` (JS).

## Merge

Each widget branch moved its widget into a package and deleted plumbing while
`slim-shared` rewrote the same single file to call the mixins. The packages are the
structure; the shared intent was re-applied inside them:

| Class | Bases (besides the folder mixin and `anywidget.AnyWidget`) |
|---|---|
| `Show2D` | `Show2DExport, Show2DState, Show2DStaticPng, SavedStateMixin, HtmlExportMixin, PagesMixin, PanelsMixin, StaticFallbackMixin` |
| `Show3D` | `Show3DFallback, Show3DExport, Show3DPlayback, SavedStateMixin, HtmlExportMixin, PagesMixin, PanelsMixin, StaticFallbackMixin` |
| `Show4DSTEM` | `SavedStateMixin, HtmlExportMixin, StaticFallbackMixin` |

Deleted from the packages in favour of the shared module: `get_state`,
`_with_initial_live_mount_state`, `_repr_mimebundle_`, `_ipython_display_`, `save`,
the six `export_*` traits, `_on_export_request_change`, `_export_mode_label`,
`_html_export_bytes`, the slug builder, the `embed_minimal_html` calls (now
`write_widget_html`), the page and panel validators, `star_page`, the `Colormap`
enums, the scale-bar label port (`_round_to_nice`, `_js_round`, `_unit_symbol`,
`_format_scale_label` in Show2D; `_round_to_nice_value`, `_format_scale_label` in
Show4DSTEM), the colormap LUT (`colorize`) and `_store_static_fallback_preview`.
Kept as widget-specific overrides: Show2D `export_svg`, folder-mode HTML and
`_html_export_options`; Show3D GIF/MP4 (`_export_request` handles `gif`/`mp4` and
defers HTML to the mixin), `_release_export_clone` (`free()`); Show4DSTEM report
export (`_write_html_export` dispatches interactive/report, `dataset_scope` travels in
the options).

The shared modules changed where every user of a helper had deleted it:

- `ControlsMixin` is not mixed into Show2D, Show3D or Show4DSTEM (all three branches
  removed `collapse_controls` / `expand_controls` / `toggle_controls`; set
  `controls_collapsed`). Show1D, ShowEDS, ShowDiffraction and Show3DSlices kept it
  until wave 2 (below).
- `roi.py` (`RoiMixin`, `sample_line`, `sample_profile_strip`) and
  `utils/roi_geometry.py` are deleted: Show2D and Show3D removed the ROI and profile
  API and nothing samples a line in Python any more. Show2D keeps its `profile`
  property.
- `PanelsMixin` keeps the `selected_panels` / `panel_order` validators and the panel
  reference resolution; `ordered_panels`, `show_panel`, `show_all_panels`,
  `set_panel_order`, `reset_panel_order`, `move_panel` and the `ordered_panel_indices`
  / `moved_panel_order` helpers are gone (removed by both galleries; Show4DSTEM removed
  its `compare_*` twins).
- `PagesMixin` keeps the page validators and `star_page`; `starred_pages` and
  `unstar_page` are gone.
- `_store_static_fallback_preview` has one default on `StaticFallbackMixin` (Plot2D
  overrides it to keep a lossless PNG); the two `getattr` call sites call it directly.
- `static_overlay_font` moved from Show2D to `render/figure.py` (Show4DSTEM's preview
  imports it from there).

Gates on the merged branch also required: no `from __future__` (19 files), no
`TYPE_CHECKING` guards (`io/image.py`, `plot2d.py`, `showcif.py` import `Dataset2d`,
`Figure`, `Atoms` for real; `RgbImage` is defined before its first use) and no
underscore module names (`folder_watch_status.py`, `showcif_potential.py`). Every
annotation in `quantem.widget` evaluates on Python 3.11 to 3.14.

Shared tests: `tests/test_frequency_filter.py` is deleted (both parametrised viewers
removed the Python frequency-filter path); `tests/test_display_filter.py` keeps the
`utils.display_filter` function tests; `tests/test_viewer_handoff.py` keeps the
Show1D -> Show2D hand-off; `tests/test_read_image.py` tests `read_gif` without the
removed `from_gif` constructors; `tests/infrastructure/test_shared_modules.py` walks
packages as well as modules.

### Wave 2

Five more branches from `165e3c0`, merged with `--no-ff` in widget order:
`slim-show1d`, `slim-show3dslices` (Show3DSlices, ShowCIF, Plot2D, ChooseLattice),
`slim-showeds`, `slim-showdiffraction`, `slim-showptycho` (ShowPtycho, PlanPtycho).
Conflicts were textual only: this record (every section kept, then ordered by
widget), the controls rows in `docs/api/viewer-ui.md` and
`docs/developer/ui-guide.md` (every branch's removals applied), the `__init__.py`
exports (the ShowEDS and ShowPtycho removals both applied, `ShowCIF` added), the
ShowEDS / ShowDiffraction cases in `tests/test_html_export_protocol.py` (each
branch's own row) and the overrides set in `tests/infrastructure/test_shared_modules.py`
(both branches removed their entry, so the set and its check are gone). No two
branches edited the same shared module.

After the merge:

- `ControlsMixin` is deleted from `state.py`: ShowEDS was its last user and no
  tutorial or downstream source calls `collapse_controls` / `expand_controls` /
  `toggle_controls`. The three method rows left `viewer-ui.md` and `ui-guide.md`,
  and the Show2D / Show4DSTEM API tables stopped listing them.
- The `except ImportError` scipy fallback in `showeds/loader.py` is deleted (numba is
  a hard dependency); the scipy counting sort is now the parity reference inside
  `tests/showeds/test_showeds.py`.
- The 13 remaining `except Exception` clauses (`image_folder.py`,
  `command_launcher.py` launcher scripts, `fallback.py`, `folder_watch_status.py`,
  `io/schema`, `io/image.py`, `utils/array.py`) catch named exception types.
- `docs/api/showeds.md` describes the package API (no `load_eds`,
  `SpectrumImage`, folder-mode export).

### Behavior changes a user can notice

| Widget | Change | Source |
|---|---|---|
| Show3D | A `Dataset3d` in nm gets a scale bar in display pixels (`pixel_size` includes the display bin, as `sampling=` already did); the old bar was 4x too short at `display_bin=4` | slim-show3d |
| Show3D | `save_image` writes RGB PNG instead of RGBA (pixels identical, alpha was constant 255) | slim-shared |
| Show3D | uint8 export label reads "uint8" instead of "encoded uint8"; "Exporting HTML..." names the mode | slim-shared |
| Show3D | Saved-state folder-watch text says "No completed image was captured" (shared wording) | merge |
| Show4DSTEM | The uint8/uint16 saturation filter on load is gone: an exported uint8 page shows the clipped bright field flat instead of black; `dp_global_max` 254 -> 255 | slim-show4dstem |
| Show4DSTEM | `bf_radius` differs in the eighth digit (float64 `detector.fit_probe`), same disk pixels | slim-show4dstem |
| Show4DSTEM | Static PNG scale-bar label follows js/figure.ts: "50 px" -> "50 pixels" for uncalibrated data, sub-1 values re-ladder (0.5 nm -> "5 Å") | slim-shared |
| Show4DSTEM | `export_html` takes `dtype`, `det_bin`, `scan_bin`, `export_kind`, `dataset_scope`; the aliases `mode`, `encoding`, `downsample`, `real_space_bin` are gone | slim-show4dstem |
| Show2D | `export_html` differs from before only in the removed state keys, the `_anywidget_id` module path (`quantem.widget.show2d.widget.Show2D`) and the rebuilt JS bundle | slim-show2d |
| Show2D, Show3D | `Colormap` is the shared 14-name enum (Show2D previously listed 6) | slim-shared |
| GIF export | Scale-bar labels round half away from zero and re-ladder 0.5 nm to "5 Å" like the live canvas | slim-shared |
| All | Removed constructor parameters, methods and traits listed per widget below | per branch |

## Show1D

Branch `slim-show1d`. `src/quantem/widget/show1d.py` (3,883 lines) became the package
`src/quantem/widget/show1d/` (1,622 lines): `widget.py` (class, traits, validators, constructors,
live append / snapshot, monitor file, hand-off), `review.py` (trial ranking rows, alerts, best
trial behind the Review panel), `export.py` (`export_html`, `save_image`, the export clone and the
static PNG preview), `state.py` (`state_dict` / `load_state_dict`; `save` and the `get_state` trim
come from the shared `SavedStateMixin`). `from quantem.widget import Show1D` and
`from quantem.widget.show1d import Show1D` keep working; `_anywidget_id` is now
`quantem.widget.show1d.widget.Show1D`. `js/show1d/index.tsx` 5,988 -> 5,785 lines.

Users outside this repository: denova (`fourdstem/neural.py`, `shine/workflow.py`) calls
`Show1D.live(traces, title, x_label, x_integer, y_label, plot_height_px, side_panel_width_px,
image_cmap)`, `append` and `snapshot`; all kept. quantem.live and quantem.thick src do not use Show1D.

Kept constructor parameters (24): `data x labels title x_label y_label x_integer log_scale ui_mode
show_review plot_height_px side_panel_width_px image_cmap snapshot_columns snapshot_panel_width_px
snapshot_histogram_width snapshot_histogram_height profile_image profile_line profile_width sampling
units state save_state`. Every synced trait still passes through `**kwargs` (for example
`show_legend=False`, `snapshot_contrast_preset="1-99"`, `review_mode="optimization"`); an unknown
keyword raises `TypeError`.

Kept public methods (18): `live from_image from_monitor_file watch_run append_monitor_event append
extend snapshot set_data goto_snapshot star_snapshot_group to_show2d stop_monitor save_image
export_html summary state_dict load_state_dict`, plus `save` and `get_state` from the shared mixin
and the module function `sample_line_profile`.

Removed constructor parameters (7):

| Parameters | Why |
| --- | --- |
| `show_scale_bar` | alias of the `scale_bar_visible` trait, which still passes as a keyword |
| `prefer_webgpu` | alternative-backend hook; trait removed |
| `monitor_path monitor_refresh_s` | `from_monitor_file` / `watch_run(refresh_s=)` cover it; traits removed |
| `notebook_preview_format notebook_preview_quality notebook_preview_max_px` | the mixin defaults (jpeg, quality 88, 512 px) apply |

The other 43 former explicit parameters (`colors x_unit y_unit show_title show_stats show_legend
show_grid show_controls controls_collapsed line_width plot_width_px max_width`, the `snapshot_*`
view settings, `pixel_size pixel_unit scale_bar_visible`, the `trial_*` review settings,
`starred_snapshot_image_labels hidden_snapshot_image_labels bookmarked_snapshot_groups`) are no
longer named in the signature but are accepted unchanged as trait keywords.

Removed public methods (36):

| Group | Methods | Why |
| --- | --- | --- |
| Alternative constructors | `from_loss_runs from_joint_time_report from_example` | no tutorial or external caller; the tutorial uses `show1d_ducky` + `from_monitor_file` |
| Aliases and internals made public | `append_scalar apply_monitor_events refresh_monitor start_monitor set_profile_image` | `append`, `from_monitor_file`, `watch_run`, `from_image` cover them |
| Markers and jump detection | `add_marker clear_markers detect_jumps clear_detected_jumps` | tests only; `markers` trait and the browser marker drawing removed |
| Playback | `play pause stop` | tests only; the browser play button drives `snapshot_playing` |
| Trial review sugar | `star_trial unstar_trial clear_starred_trials hide_trial show_trial show_all_trials set_trial_note clear_trial_note tag_trial untag_trial clear_trial_tags set_starred_only set_trial_sort rank_trials star_best_trial hide_worst_trials` | tests only; the Review panel writes the same traits (`starred_snapshot_image_labels`, `hidden_snapshot_image_labels`, `trial_notes`, `trial_tags`, `trial_sort_key`, ...) and monitor events carry `starred` / `hidden` / `notes` / `tags` |
| Snapshot group stars | `unstar_snapshot_group toggle_snapshot_group_star clear_snapshot_group_stars` | tests only; `star_snapshot_group` (tutorial) stays and the browser toggles the trait |
| Side exports | `export_csv export_run_summary` | tests only |

Removed traits (8): `markers method_labels run_summary report_metadata widget_version prefer_webgpu
monitor_path monitor_refresh_s`. Browser side: the marker drawing, method-label chips and the
`window.__quantemShow1DPerf` pointer / FFT perf counters are gone from `js/show1d/index.tsx`.

Tests: `tests/test_widget_show1d.py` (39 tests, 1,196 lines) and `tests/test_viewer_handoff.py`
(2 Show1D hand-off tests) became `tests/show1d/test_show1d.py` (15 tests, 322 lines, about 2.4 s):
series and stats vs NumPy, live append / extend NaN back-fill, UI presets and validators, line
profile exact on a bilinear plane, snapshot groups, monitor-file append / rebuild / tail, `to_show2d`
and the hand-off request, state round trip, `export_html` structure and downsample, static preview
and `save_image`.

Byte identity for the tutorial inputs (ducky monitor, line profile, reopened synthetic monitor,
protocol traces): the export HTML page shell is identical apart from the random model id; the
embedded widget state is identical apart from the 8 removed traits, `_anywidget_id` and the bundle.
The static PNG previews and `save_image` PNG are byte-identical. `state_dict()` no longer carries the
derived `best_trial_label`, `trial_rankings`, `trial_alerts` (recomputed from the traces on load).

## Show2D

`src/quantem/widget/show2d.py` (9,316 lines) became the package `src/quantem/widget/show2d/`:
`widget.py` (class, traits, constructor, observers, folder watching), `options.py` (constructor
option normalisers that Show3D also imports), `export.py` (`export_svg` as named functions plus the
standalone HTML export), `fallback.py` (static PNG preview; the scale-bar label port lives in `render/figure.py`),
`state.py` (`state_dict` / `load_state_dict`; `save` and the `get_state` trim come from the shared `SavedStateMixin`). `from quantem.widget import
Show2D` and `from quantem.widget.show2d import Show2D` keep working; `_anywidget_id` is now
`quantem.widget.show2d.widget.Show2D`.

Kept constructor parameters (58): `data labels page_labels title ui_mode cmap sampling units
show_scale_bar scale_bar_visible scale_bar_position scale_bar_panels scale_bar_length
scale_bar_label scale_bar_style show_zoom_indicator show_fft show_stats show_controls
controls_collapsed show_panel_titles panel_title_spans panel_title_font_size panel_title_style
log_scale auto_contrast vmin vmax link_zoom link_pan link_contrast ncols size panel_width_px smooth
zoom display_bin offline panel_frame_indices panel_playback_fps starred marker_colors marker_style
row_markers inset_plots panel_annotations overlays panel_overlays inter_panel_gap_px
inter_panel_gap_color gallery_outer_border_px gallery_outer_border_color panel_inner_border_px
panel_inner_border_color gallery_gap_px state save_state verbose`. Everything after `labels` is
keyword-only. Trait names (for example `pixel_size=`) still pass through `**kwargs`.

Kept public methods and properties (12): `from_folder set_image set_panel_frame star_page save_image
export_svg export_html save state_dict load_state_dict summary profile` plus `poll_folder`,
`watch_folder`, `stop_folder_watch`, `folder_paths`, `close` from the folder mixin and the
ipywidgets `get_state` override.

Removed constructor parameters (54):

| Group | Parameters | Why |
| --- | --- | --- |
| Debug | `debug` | debug hook (trait and browser FPS badge removed too) |
| Display-filter Python fallback | `denoise denoise_sigma denoise_bin denoise_scope show_denoise frequency_filter frequency_filter_enabled frequency_filter_cutoff frequency_filter_center frequency_filter_width show_frequency_filter display_filter display_sigma spatial_bin filter_per_panel` | no tutorial; the browser owns display denoise and frequency filtering (`js/displayFilter.ts`, `js/frequencyFilter.ts`). The traits stay for the browser UI; Python no longer filters pixels, mirrors knobs or writes the banner |
| Underlay / dual-gain chemistry blend | `underlay underlay_alpha underlay_haadf_gain underlay_mode stretch_percentiles display_gamma dual_gain` | no tutorial; traits and the browser slider row removed |
| View ops (crop / pad) | `view_box pad_ratio pad_fill_mode pad_scope center zoom_row zoom_col` | tests only; `view_crop`, `pad_*`, `view_banner`, `_view_crop_offset` traits and the browser View menu removed. `view_box`, `zoom_row`, `zoom_col` stay as browser-synced traits |
| Rotation and flips | `rotation rotations rotation_scope flip_rows flip_cols image_flips_horizontal image_flips_vertical` | tests only or nowhere; the browser rotate and flip buttons still drive `image_rotations` / `image_flips_*` |
| Overlay and diff sugar | `overlay diff_mode` | tests only; `diff_mode` stays a browser toggle |
| Panel management | `hidden_panels panel_order` | tests only; the traits stay for the browser UI |
| Trait mirrors of browser toggles | `show_title fft_window fft_metrics contrast_preset histogram_advanced show_histogram_advanced identity_colors col_markers show_inset_plots gallery_gap_color` | tests only or nowhere; `histogram_advanced`, `show_histogram_advanced`, `identity_colors`, `gallery_gap_color` had no browser reader and are gone as traits too |
| Notebook preview knobs | `notebook_preview_format notebook_preview_quality notebook_preview_max_px` | tests only; the mixin defaults (jpeg, quality 88, 512 px) apply |
| Private test hooks | `_skip_initial_frame_pack _skip_initial_stats _preserve_input_dtype_for_export` | test fixtures |

Removed public methods and properties (49):

| Group | Names |
| --- | --- |
| Saved-view bookmarks | `save_view_state load_view_state delete_view_state clear_view_states` (traits `saved_view_states saved_view_request saved_view_status` and the browser Save State menu removed) |
| View ops | `current_view crop_to_view reset_view_ops set_padding view_corner view_center` |
| Display filter | `set_denoise` |
| Hand-off to Show3D | `to_show3d` (traits `handoff_request handoff_status handoff_enabled prepared_view_widget` and the browser View as 3D entry removed) |
| Panel management | `ordered_panels visible_panels starred_panels starred_pages set_hidden_panels hide_panel show_panel show_all_panels set_panel_order reset_panel_order move_panel set_starred_panels star_panel unstar_panel unstar_page` |
| Controls | `collapse_controls expand_controls toggle_controls` (set `controls_collapsed` directly) |
| Rotation | `rotate` (set `image_rotations`) |
| ROI and profile | `set_profile clear_profile add_roi clear_rois delete_selected_roi get_roi_geometries roi_geometries set_roi roi_circle roi_square roi_rectangle roi_annular profile_values profile_distance` (`roi_list`, `profile_line` and the `profile` property stay) |
| Folder | `from_gif folder_page_size set_folder_page_size` (`from_folder(page_size=...)` stays) |

Removed traits with no remaining reader: `widget_version frequency_filter_banner _gpu_max_buffer_mb
flip_rows flip_cols histogram_advanced show_histogram_advanced identity_colors`, the view-op,
underlay, saved-view and hand-off traits listed above, and `debug`. Removed timing hooks: the
first-paint timing print, `render_total_ms` / `render_python_build_ms` / `render_wire_js_ms` and the
`Rendered:` line of `summary()`; `_on_first_render` only drops the pixel buffers from saved state.

Browser side (`js/show2d/index.tsx`): removed the debug FPS badge, the `_gpu_max_buffer_mb` report,
the crop/pad View menu and its banner, the underlay slider row, the saved-state menu, the View as 3D
hand-off and the embedded prepared view. The display filter, frequency filter, rotation, flip, panel
hide/order and diff UI are unchanged.

Export identity for the tutorial inputs (`show2d_gold` and the `show3d_gold` first frame): `save_image`
PNG, `export_svg` and the static PNG preview are byte-identical before and after. `export_html`
differs only in the removed state keys, the `_anywidget_id` module path and the rebuilt JS bundle;
the rest of the HTML is identical.

Tests: `tests/show2d/test_show2d.py` replaces `tests/test_show2d_{view,pages,rgb,mixed_stacks,
panel_visibility,saved_view_states}.py` and the Show2D parts of `test_save_state.py`,
`test_state_dict.py`, `test_viewer_handoff.py`, `test_display_filter.py`, `test_frequency_filter.py`,
`test_image_folder_watch.py` and `test_read_image.py`.

## Show3D

`src/quantem/widget/show3d.py` (7,625 lines) became the package
`src/quantem/widget/show3d/`: `widget.py` (traits, validators, constructor,
`set_image`, `from_folder`, state, `summary`, `free`), `playback.py` (the
embedded display stack, frame access, auto-contrast ranges, ROI plot),
`export.py` (HTML, PNG, GIF, MP4) and `fallback.py` (the static preview; the
saved-notebook trim comes from the shared `SavedStateMixin`). `from quantem.widget import Show3D` and
`quantem.widget.show3d.Show3D` are unchanged.

What uses Show3D outside the tutorials: quantem.live (`export.py`,
`screen_report.py`, `show3d_watch.py`, `ptycho_search_export.py`) passes
`labels panel_titles title show_title display_bin sampling units cmap vmin vmax
auto_contrast smooth show_scale_bar link_contrast max_cols verbose offline
panel_gap panel_width_px dim_label show_fft notebook_preview_format`, calls
`export_html(encoding, downsample, max_mb, title)`, `save_gif(fps, quality,
downsample, show_frame_labels, show_scale_bar)`, `set_image(labels=)`,
`slice_idx`, `free()`, `close()`. denoise (`data.py`, `results.py`,
`shine/results.py`, `imaging/alignment.py`, `calibration_plotting.py`,
`fourdstem/series.py`) passes `labels panel_titles cmap display_bin offline
title show_fft max_cols ui_mode vmin vmax auto_contrast link_contrast smooth
panel_width_px show_stats sampling units show_scale_bar` and calls
`export_html(title, encoding)`. quantem.thick does not use Show3D.

Kept constructor parameters (34 plus `*data_args`): `*data_args labels panel_titles page_labels
title ui_mode show_title show_stats show_scale_bar cmap vmin vmax auto_contrast
link_contrast sampling units smooth panel_annotations panel_overlays fps
avg_window show_fft fft_layout fft_overlay_zoom panel_width_px max_cols
panel_gap dim_label display_bin offline save_state notebook_preview_format
notebook_preview_quality notebook_preview_max_px verbose` (the last five are
the saved-notebook contract shared by every widget and tested in
`tests/test_save_state.py`).

Kept public methods (13): `from_folder set_image state_dict load_state_dict
save export_html save_image save_gif save_mp4 summary star_page free
get_state` plus the folder mixin (`poll_folder watch_folder stop_folder_watch
folder_paths folder_errors close`).

Removed constructor parameters (93): `panel_title_spans panel_frame_labels
frame_metadata panel_frame_metadata frame_label_format panel_real_frames
hidden_panels panel_order show_controls controls_collapsed image_rotation
rotation rotations rotation_scope log_scale image_vmin_pct image_vmax_pct
percentile_low percentile_high contrast_preset histogram_advanced
show_histogram_advanced marker_colors marker_style row_markers col_markers
panel_groups overlays flip_horizontal flip_vertical compare_mode compare_pair
blink_fps diff_cmap compare_background timestamps timestamp_unit
fft_overlay_position fft_overlay_size fft_window fft_metrics debug size crop
padding pad_mode config rotation_deg post_crop apply_config_transforms rgb
diff_mode buffer_size use_torch device hideable state notebook_preview_frames
notebook_preview_ncols denoise
denoise_sigma denoise_bin denoise_scope display_filter display_sigma
spatial_bin show_denoise frequency_filter frequency_filter_enabled
frequency_filter_cutoff frequency_filter_center frequency_filter_width
frequency_filter_scope show_frequency_filter subpixel_align
subpixel_align_reference inter_panel_gap_px inter_panel_gap_color
gallery_outer_border_px gallery_outer_border_color panel_inner_border_px
panel_inner_border_color panel_title_font_size panel_title_style
show_panel_titles show_resize_handles show_zoom_indicator
dedupe_identical_panels scale_bar_visible link_zoom link_pan link_panels
show_playback` (the last five were `**kwargs` aliases).

Removed public methods and properties (52): `from_rgb from_figure_gallery
from_gif from_panel_folders to_show2d play pause stop goto star_panel
unstar_panel starred_frames starred_pages ordered_panels visible_panels
unstar_page set_hidden_panels hide_panel show_panel show_all_panels
set_panel_order reset_panel_order move_panel collapse_controls expand_controls
toggle_controls visible_indices hide show set_hidden show_all roi add_roi
clear_rois delete_selected_roi get_roi_geometries roi_geometries
set_notebook_preview_frames clear_notebook_preview_frames set_roi roi_circle
roi_square roi_rectangle roi_annular profile profile_values profile_distance
set_profile clear_profile profile_all_frames profile_all_pages
save_animation_preview` and the class `AnimationExportPreview`.

Removed synced traits (37, with their JS counterparts): `benchmark_request
benchmark_result _js_rendered debug handoff_request handoff_status
handoff_enabled prepared_view_widget hideable hidden_indices timestamps
timestamp_unit frame_metadata panel_frame_metadata frame_label_format
panel_frame_labels identity_colors marker_colors marker_style row_markers
col_markers panel_groups panel_title_style panel_title_font_size
inter_panel_gap_color gallery_outer_border_px gallery_outer_border_color
panel_inner_border_px panel_inner_border_color notebook_preview_frames
notebook_preview_ncols source_bytes histogram_advanced show_histogram_advanced
flip_rows flip_cols fft_metrics`; Python-only `render_total_ms
render_python_build_ms render_wire_js_ms widget_version roi_stats`.

Removed code paths: the numpy/torch twin behind `use_torch` (stats now go
through torch on the host array; no implicit device copy), the ptycho
`config` crop/pad/rotate transforms (the `_crop_stack _pad_stack
_normalize_crop _normalize_padding` helpers moved to
`utils/recon_config.py` for Show3DSlices), the Python display-filter fallback
(`_filter_frame`, the filter cache and the `_webgpu_filter_ok` observer; the
browser owns denoise and the Python side only mirrors the traits and prints
the reduction banner), dict-form paged input, the first-render timing print,
the cupy/torch cache flush in `free()`, and the Show2D hand-off.

JS (`js/show3d/index.tsx`, 17,068 to 16,245 lines): the three benchmark
effects, the debug FPS badge, the `_js_rendered` signal, the View > "View frame
as 2D" hand-off and embedded Show2D view, hidden-frame navigation, panel
identity markers and row/column/panel group frames, `panel_frame_labels`, and
the two inline scale-bar drawings (browser GIF export and the per-panel UI
overlay) which now share one `drawPanelScaleBar` helper built on
`roundToNiceValue` / `formatScaleLabel` from `js/figure.ts`.

Behavior change: a `Dataset3d` input in nm now gets a scale bar in display
pixels (`pixel_size` includes the display bin, as it already did for
`sampling=`); the old value was 4x too short at the default `display_bin=4`.
Export HTML for the tutorial gold stack is otherwise identical trait for trait
and byte for byte in the pixel payload; the static notebook PNG is identical
once the same pixel size is applied to the old code.

Tests: `tests/show3d/test_show3d.py` (17 tests, under 2 s) replaces the six
files under `tests/show3d/` and the Show3D-only tests in `test_save_state.py`,
`test_display_filter.py`, `test_frequency_filter.py`, `test_viewer_handoff.py`
and `test_read_image.py`. `Show2D.to_show3d` passes parameters Show3D no longer
has; its two tests in `test_viewer_handoff.py` were removed with the Show3D
hand-off tests (the method is tests-only and is deleted on the Show2D branch).

## Show3DSlices

`src/quantem/widget/show3dslices.py` (2,390 lines) plus `utils/recon_config.py`
(206 lines, used by nothing else) became the package
`src/quantem/widget/show3dslices/`: `widget.py` (717 lines) and `alignment.py`
(204 lines). `from quantem.widget import Show3DSlices` and
`quantem.widget.show3dslices.Show3DSlices` are unchanged.

`alignment.py` holds the slice-alignment science with the math untouched:
`estimate_global_slice_alignment(stack)` (adjacent-slice registration after
median centering, Gaussian high-pass and Hann window, 20x matrix-DFT subpixel
refinement after Guizar-Sicairos 2008, cumulative trajectory, linear fit),
`apply_global_slice_alignment(stack, row, col)` (nearest-edge bilinear display
shift) and the primitives `registration_image`, `estimate_adjacent_shift`,
`upsampled_dft`. The registration image and the adjacent shifts were checked
bit-identical against the old module on three synthetic drifts before the old
code was deleted; `js/sliceAlignment.ts` still mirrors the same constants.

What uses Show3DSlices outside the tutorials: quantem.live (`export.py`,
`control/pipelines/ptycho_search_export.py`, `server/rendering/ptycho_html.py`,
`notebooks/compare.py`) passes `title show_title sampling units pixel_size
dim_labels cmap smooth slice_alignment z_stretch panel_width_px show_scale_bar
show_fft fft_window image_vmin_pct image_vmax_pct vmin vmax auto_contrast` and
calls `export_html(path, encoding=, title=)`, `free()` and `close()`. denoise
does not use Show3DSlices (it imports Show2D, Show3D and Show4DSTEM only);
quantem.thick does not import quantem.widget. The tutorial passes `sampling
units title`.

Kept constructor parameters (23 plus `data`): `title panel_titles page_labels
show_title cmap sampling units pixel_size show_scale_bar z_stretch
panel_width_px show_fft fft_window smooth auto_contrast vmin vmax
image_vmin_pct image_vmax_pct slice_alignment dim_labels offline`.

Kept public methods (7): `estimate_slice_alignment reset_slice_alignment
export_html state_dict load_state_dict save free`.

Removed constructor parameters (30): `data_b title_b ui_mode config
apply_config_transforms crop padding pad_mode rotation_deg post_crop
scale_bar_visible show_controls controls_collapsed show_stats show_crosshair
orthographic flip show_diff log_scale row_shift_px_per_slice
col_shift_px_per_slice fps loop reverse boomerang linked_contrast play_axis
state compact` and the `**kwargs` typo check. The display traits behind them
(`show_controls controls_collapsed show_crosshair orthographic flip log_scale
row_shift_px_per_slice col_shift_px_per_slice fps loop reverse boomerang
play_axis`) stay synced because the browser controls write them.

Removed public methods (11): `set_page next_page previous_page play pause stop
save_image summary collapse_controls expand_controls toggle_controls`.

Removed traits: Python-only `widget_version viewer_kind show_stats stats_mean
stats_min stats_max stats_std` (and `_compute_stats`). No synced trait was
removed, so `js/show3dslices/index.tsx` is unchanged.

Removed code paths: the ptycho `config.json` crop/pad/rotate/post-crop
transforms (`utils/recon_config.py` deleted, the quantem.gpu
`rotate_stack_inplane` import with it), the `data_b` two-panel shortcut, dict
page descriptors (`{"title":..., "volume":...}`), the Python stats
reductions, PNG/PDF/TIFF single-slice export, the `mode=`/`downsample=`/
`quantized=` export aliases (the toolbar's `exact`/`quantized` modes map onto
`encoding`), the saved-state `dual_mode`/`show_diff` legacy rejection and the
`play_axis` 1/2 legacy mapping. `load_state_dict` now drops unknown keys
through `_STATE_KEYS` as Show3D does.

Byte identity: for the tutorial volume the export HTML (`full` and `uint8`)
differs from the old build only in the random widget model ids and in
`_anywidget_id` (`show3dslices.Show3DSlices` to
`show3dslices.widget.Show3DSlices`); the embedded `_esm` bundle hash and the
volume payload are identical. `state_dict` lost the two removed keys
`show_stats` and `viewer_kind`. Show3DSlices has no static notebook PNG.

Tests: `tests/show3dslices/test_show3dslices.py` (9 tests, 1.6 s) replaces
`test_show3dslices_multi_panel.py`, `test_show3dslices_pages.py` and
`test_show3dslices_slice_alignment.py`. It checks the browser volume against
numpy indexing, the uint8 pack, Dataset3d sampling, alignment recovery of a
known Fourier-shifted drift to 0.1 px with the aligned stack within 15% of the
drift error, caching and the browser request path, pages, state round trip and
the export HTML structure plus toolbar download.

## Show4DSTEM

Branch `slim-show4dstem`. One package `src/quantem/widget/show4dstem/`
(`widget.py` class and folder watcher, `detector.py` masks over
`quantem.gpu.detector`, `export.py` interactive page, report and WebGPU bundle,
`preview.py` static PNG) replaces `show4dstem.py`, `show4dstem_factory.py`,
`show4dstem_bounded.py` and `show4dstem_webgpu_export.py` (8318 lines to 2983).
JS `js/show4dstem/index.tsx` 11247 to 10699 lines; `lazy.ts`,
`roiRadiusDrag.ts` and three tests of the removed sources deleted.

Kept because a tutorial, the CLI, quantem.live or denoise uses it: constructor
`data scan_region scan_shape detector_shape sampling units center bf_radius
precompute_virtual_images ssb_voltage_kV ssb_semiangle_mrad ssb_scan_sampling_A
ssb_det_sampling_mrad frame_dim_label frame_labels view_mode compare_cols
compare_grid_width_px compare_max_panels compare_group_mode compare_dp_mode
title ui_mode show_title show_controls controls_collapsed show_stats
show_scale_bar offline offline_dtype h5_urls backend show_fft panel_width_px
verbose state save_state notebook_preview_*` plus trait kwargs such as
`dp_scale_mode`; methods `from_folder export_html compute_ssb apply_preset
summary profile state_dict save load_state_dict free close poll_folder
wait_for_folder watch_folder stop_folder_watch pattern virtual_image`.

Removed constructor parameters: `DPC_row DPC_col SSB vi_source`,
`ssb_bf_radius ssb_bf_intensity_threshold ssb_aberrations
ssb_rotation_angle_deg ssb_n_trials ssb_refine ssb_seed
ssb_manual_aberrations ssb_c10_nm ssb_c12_nm ssb_phi12_deg
ssb_compute_enabled` (all accepted by `compute_ssb(**kwargs)`), `compare_layout
compare_panel_gap_px compare_cache_pages compare_cache_max_bytes`, `data_url
offline_codec h5_url rans_url rans_count rans_format rans_files rans_dtype
lazy_url lazy_urls h5_uint8_lossless`, `fft_window debug dp_vmin dp_vmax
vi_vmin vi_vmax` (still traits).

Removed public methods and properties: `set_vi_product_map set_vi_preset_map
set_path play pause stop goto raster roi_circle roi_point roi_square
roi_annular roi_rect auto_detect_center save_image collapse_controls
expand_controls toggle_controls position scan_shape detector_shape set_profile
clear_profile profile_values profile_distance compare_ordered_panels
compare_visible_panels set_compare_hidden_panels hide_compare_panel
show_compare_panel show_all_compare_panels set_compare_panel_order
reset_compare_panel_order move_compare_panel set_compare_page
next_compare_page previous_compare_page show_compare_paged_groups
show_compare_all_groups set_compare_starred_panels star_compare_panel
unstar_compare_panel warm_compare_cache stop_compare_cache_warm
preload_all_datasets stop_dataset_preload wait_for_dataset_preload`;
`export_html` aliases `mode encoding downsample real_space_bin`.

Removed traits: `_offline_url _offline_chunks _offline_gzip _offline_bslz4
_h5_url _rans_url _rans_format _rans_files _rans_dtype _h5_uint8_lossless
_lazy_url _lazy_urls vi_preset_labels vi_preset_map_frames
vi_preset_maps_bytes path_playing path_index path_length path_interval_ms
path_loop debug compare_layout compare_panel_gap_px gpu_memory_label
memory_warning _gif_export_requested _gif_data _gif_metadata_json`.

Removed codecs and paths: gzip companion folder (chunked `data_url`), bslz4
companion volumes, rANS browser-resident set (and the QEM local-file picker),
lazy radial-profile source, the single `h5_url` branch (one URL now travels in
`_h5_urls`), the `--combined` 5D CLI export, the compare virtual-image and
diffraction caches with `warm_compare_cache`, the per-panel sparse masked-sum
kernels (every reduction is a `quantem.gpu.detector` session), the GPU memory
badge, the uint16/uint8 saturation filter on load, 12 `perf_counter` timing
prints and the verbose backend table. Kept: inline gzip `_offline_stack`
(what `export_html` and `quantem show4dstem --html` write) and the linked-HDF5
browser source (`quantem show4dstem --backend webgpu --html`).

Export of the tutorial input: state identical except the removed traits,
`bf_radius` in the eighth digit (float64 `detector.fit_probe` instead of a
float32 sqrt; same disk pixels), and `dp_global_max` 254 to 255 (the old
clone zeroed clipped pixels). Static PNG byte-identical. The gold uint8 export
shows a flat bright-field image because every disk count clips at 255; the
old page showed it black. `--dtype uint16` shows the structure.

quantem.live uses no Show4DSTEM Python API: `notebook_publish.py` and
`notebook_save_hook.py` match the text `Show4DSTEM(` in notebooks, and
`server/routers/browse.py` carries its own port of the old
`auto_detect_center` in comments. denoise `fourdstem/scan.py` calls
`Show4DSTEM(list_or_stack, view_mode, compare_cols, compare_dp_mode,
panel_width_px, compare_grid_width_px, frame_labels, title)`, all kept.

## ShowEDS

Branch `slim-showeds`. One package `src/quantem/widget/showeds/` (`widget.py`
class, static preview and HTML export; `loader.py` EMD readers, Velox stream
index, data folder, band and ROI reductions; `elements.py` X-ray line table and
element detection) replaces `showeds.py` (3688 lines to 1503). JS
`js/showeds/index.tsx` 3996 to 3815 lines. Tests: `tests/test_showeds.py`
replaced by `tests/showeds/test_showeds.py` (13 tests, about 2 s), every
reduction checked against a numpy sum over the same dense cube.

Kept because the tutorial (code cells and the fenced `from_emd` example) uses
it: constructor `cube energy_keV title ui_mode show_title show_controls
controls_collapsed show_scale_bar base_image energy width element_label
candidate_elements selected_elements panel_width_px spectrum_width_px
spectrum_height_px log_spectrum sampling units state save_state`;
`from_emd(path, backend, sidecar_dir, sidecar_url, spatial_bin, energy_bin,
title)`, `export_html(path, title, downsample)`, `detect_elements`,
`state_dict`, `load_state_dict`, `save`.
numba stays: indexing the Velox event stream of a tutorial-sized cube
(96x96x320, 13M events, 4 detectors) takes 130 ms with numba and 1582 ms with
the scipy counting sorts (identical arrays). numba is a hard dependency, so the wave-2 merge
deleted the `except ImportError` scipy fallback from `loader.py`; the scipy
counting sort lives on in `tests/showeds/test_showeds.py` as the parity reference.

Removed:

- Constructor parameters `band` and `roi` (use `energy`/`width` and
  `load_state_dict`), `roi_shape`, `auto_identify`, `line_hints`,
  `map_vmin_pct`, `map_vmax_pct`, `overlay_opacity`, `pixel_size`,
  `pixel_unit`, `scale_bar_visible`, `show_line_hints`, `smooth`,
  `saved_bands`, `saved_rois` (all still traits and saved state; set them or
  pass `state=`), `initial_map`, `initial_spectrum`, `lazy_path`,
  `sidecar_url`, `max_state_bytes`, `export_presets`, `show_debug`,
  `debug_control_visible`.
- Methods `from_sidecar` (pass `load_sidecar(folder)` to the constructor, or
  `from_emd`, which reuses an existing folder), `get_state`, and
  `collapse_controls` / `expand_controls` / `toggle_controls` (no tutorial or
  downstream caller; set `controls_collapsed`). Removed at the wave-2 merge
  with the shared `ControlsMixin`, which no widget mixed in any more.
- Traits `show_debug`, `debug_control_visible`, `export_presets`,
  `export_sidecar_bytes` (browser data-folder zip export) and the JS debug
  panel and export presets.
- Public names `SpectrumImage` (with `.array`, `.shape`, `.show()`),
  `load_eds`, `load_emd_spectrum_image`, `prepare_spectrum_image_sidecar`,
  `prepare_spectrum_stream_sidecar`, `load_spectrum_image_sidecar`,
  `load_spectrum_stream_sidecar`, and the top-level re-exports
  `quantem.widget.SpectrumImage`, `bin_spectrum_image`, `load_eds`,
  `load_emd_spectrum_image`. The stream-index data folder format is gone; the
  stream index is embedded in the widget, the prefix-array data folder stays
  (`loader.prepare_sidecar` / `loader.load_sidecar`).

Export for the tutorial input: static PNG byte-identical; every binary buffer
(cube, base image, first map, first spectrum, stream arrays) identical. The
exported state drops the four removed traits, adds the shared
`_static_fallback_jpeg`/`_static_fallback_mime` keys, and `_anywidget_id`
moves to `quantem.widget.showeds.widget.ShowEDS`.

## ShowDiffraction

Branch `slim-showdiffraction`. One package `src/quantem/widget/showdiffraction/`
(`widget.py` class and state, `lattice.py` center, ring, spot, ellipse, merge and
indexing math, `phases.py` `Phase`, `PHASE_LIBRARY`, `library_phase`) replaces
`showdiffraction.py` (3998 lines to 3117). The 23 per-action `_*_request` traits
are one `_request` channel (`{"action", "args", "seq"}`); Python runs the action,
reports in `analysis_status` and clears it. JS `js/showdiffraction/index.tsx`
2114 to 2098 lines. Five test files (2815 lines, 99 tests) are one
`tests/showdiffraction/test_showdiffraction.py` (278 lines, 13 tests, about 3 s)
that checks the science against references: Fe3O4 d-spacings, a known center,
a [001] cubic lattice, a known ellipse and known frame shifts.

Kept because the tutorial uses it: constructor `data k_pixel_size center
bf_radius title panel_width_px offline verbose` plus trait kwargs; methods
`run_auto detect_spots index_spots identify_phase search_phases summary
export_html save state_dict load_state_dict free measurements_from_state`;
traits such as `detect_denoise denoise phase_name custom_phases dp_colormap
dp_scale_mode show_title show_controls controls_collapsed show_stats
snap_enabled snap_radius spot_refine` (set after construction). quantem.live,
denoise and quantem.thick do not import ShowDiffraction.

Removed constructor parameters: `controls_collapsed detect_denoise
dp_scale_mode show_controls show_stats show_title snap_enabled snap_radius
spot_refine` (still traits), `pixel_size state ui_mode`.

Removed public methods and properties: `add_ring add_spot move_spot
remove_ring remove_spot undo_ring undo_spot clear_rings clear_spots
calibrate_from_ring calibrate_from_spot calibrate_from_phase refine_center
auto_detect_center set_center detect_rings fit_ellipse fit_ring_profile
apply_ellipse_correction index_rings merge_frames quality_report
radial_profile radial_background azimuthal_profile texture
recover_predicted_rings export_measurements set_image collapse_controls
expand_controls toggle_controls detector_shape n_source_frames`. The browser
buttons still run the spot, ring, calibration, fit, index and merge actions
through `_request`; the math lives in `lattice.py`.

Removed traits: `pixel_size show_detection_view widget_version` and the 23
`_*_request` traits (`_auto_request _calibrate_from_ring_request
_calibrate_from_spot_request _calibrate_phase_request _detect_rings_request
_detect_spots_request _fit_ellipse_request _fit_rings_request
_identify_request _index_rings_request _index_spots_request _merge_request
_quality_request _refine_center_request _ring_add_request _ring_clear_request
_ring_remove_request _ring_undo_request _spot_add_request _spot_clear_request
_spot_move_request _spot_remove_request _spot_undo_request`). The module no
longer re-exports its helper functions (`pick_center`, `align_frames`, ...):
import them from `quantem.widget.showdiffraction.lattice`.

Science identical: k, rms, ring radii and hkl, center, spots, zone and phase
match on the tutorial inputs and the test patterns, bit for bit. Export of the
tutorial inputs: state identical except the removed traits, `_request` added
and `_anywidget_id` now `quantem.widget.showdiffraction.widget.ShowDiffraction`.
ShowDiffraction has no static PNG fallback before or after. The tutorial prints
the same text before and after.

## ShowCIF

`showcif.py` (456 lines) absorbed `showcif_potential.py` (58 lines) as
`projected_atom_table` under a `# --- projected potential tables` section;
322 lines total. ASE is imported at the top (it is a hard dependency of the
module already), `_ready`/`hasattr(self, "_atoms")` duck typing became class
attributes, and the position packing that appeared twice is one
`_packed_positions`. Nothing outside quantem.widget uses ShowCIF.

Tutorial use: `ShowCIF(structure, repeats, zone_axis)`, `ShowCIF(structure,
repeats, potential, num_slices, energy_keV, potential_quantity,
potential_colormap)`, then the traits `view_mode`, `field_of_view_A` and
`specimen_tilt_mrad` set after construction.

Kept constructor parameters (9 plus `structure`): `repeats zone_axis title
energy_keV potential_colormap potential_quantity num_slices potential`.

Removed constructor parameters (7, all still synced traits the browser
writes): `specimen_tilt_mrad view_mode field_of_view_A orthogonal_views
magnification_calibration potential_sigma_A potential_pixels`.

Removed public methods (2): `atoms` and `export_html` (tests-only; the
`docs/api/showcif.md` HTML export section went with it). `ShowCIF` is now
listed in `quantem.widget.__all__`.

Byte identity: the atom, unit-cell and potential byte payloads and every
synced trait for the three tutorial constructions are identical before and
after (only the random layout model id differs).

Tests: `tests/showcif/test_showcif.py` (14 tests, 1.5 s) replaces
`test_showcif.py`, `test_showcif_potential.py` and `test_showcif_tilt.py`:
CIF file and ASE structure expand identically, repeats rebuild the supercell
and not the unit cell, invalid inputs, tilt and field of view as display-only
state, and the projected table against abTEM's Lobato projected potential
convolved with the Gaussian in real space (skipped with a reason when abTEM
is missing).

## ShowPtycho

Branch `slim-showptycho`. One package `src/quantem/widget/showptycho/`
(`widget.py` the `ShowPtycho` class over a `quantem.gpu.SSB` session, `export.py`
the browser folder) replaces `showptycho.py` and `showptycho_webgpu_export.py`
(2585 lines to 789). JS `js/showptycho/index.tsx` 4922 to 4765 lines (stage
timers and test-only phase publishing removed). `tests/test_showptycho.py`
(1446 lines, 36 tests) became `tests/showptycho/test_showptycho.py` (195 lines,
8 tests over a real SSB session).

No quantem.live, denoise or quantem.thick source imports ShowPtycho,
PtychoCalibration or load_ptycho_calibration (grep, 2026-10-06); the tutorials
call `ShowPtycho(ssb)`, `ShowPtycho(ssb, fft_on=True)`,
`ShowPtycho(ssb, source_file=..., save_dir=...)` and `w.export(path, title=...)`.
No `ShowPtycho.from_master` was added: no tutorial or caller builds a widget
from a master file without first building the `SSB` session.

Kept: constructor `ssb source_file save_dir fft_on`; method `export(path,
title=)`; every synced trait (the browser drives them all, including `upsample`
and `drag_bf`).

Removed constructor parameters: raw-data path `data_or_ssb` (now only an `SSB`)
with `backend semiangle_mrad scan_sampling_A det_sampling voltage_kV scan_shape
bf_intensity_threshold bf_radius aberrations rotation_angle_deg` (build the
`SSB` instead); `c10_range c12_range phi12_range rotation_range` (ranges follow
the fit); `drag_bf upsample size` (traits, or the toolbar); `calibration`
(reopening a saved calibration).

Removed public names: `PtychoCalibration load_ptycho_calibration
save_ptycho_calibration export_showptycho_webgpu_folder` and the package
exports of the first two; methods `free_drag_state pinned starred`; `export`
keywords `backend data overwrite decode_dtype webgpu_source gpu_memory_gb` (the
folder always uses the exact BF columns, 4.5 GiB browser budget, overwrite on).

Behavior changes: the folder no longer links the raw HDF5 files into `source/`
and has no HDF5 browser fallback; `result_json` no longer carries
`time_ms d2h_ms bytes_ms trait_ms py_total_ms`. For the tutorial-shaped input
the phase bytes, calibration, stars, `snapshots/*`, `source/bf_columns.qem` and
the export `index.html` (outside those timing fields and generated ids) are
byte-identical before and after.

## PlanPtycho

`src/quantem/widget/planptycho.py` stays one file (766 to 679 lines). The three
test files (361 lines) became `tests/planptycho/test_planptycho.py` (226 lines);
the arithmetic is still pinned to `js/planptycho/goldens.json`
(`scripts/planptycho_goldens.py`).

Kept: constructor `structure thickness_nm tilt_mrad c10_nm scan_step_A
scan_size_px` plus any synced trait by keyword (`zone_axis voltage_kV
semiangle_mrad detector camera_length_mm detector_px detector_mrad_per_px
focus_depth_nm wave_window_factor title`); methods `report simulation_plan`;
properties `window_A object_pixel_A wave_pixels`; module functions
`plan_geometry check_rows recommended_settings load_crystal` and the crystal
helpers.

Removed: constructor `preset` (set the traits, or the title-bar menu); methods
`apply_preset apply_thickness` (the browser menus do the same through traits)
and `geometry` (now private); trait `widget_version`. The tutorial markdown now
shows trait assignment instead. `report()`, `simulation_plan()` and every synced
trait are identical before and after for the tutorial input.

## Plot2D

`plot2d.py` 258 to 210 lines. It now mixes in `SavedStateMixin`, so the
copied `get_state` trim and the PNG `_store_static_fallback_preview` override
are gone (the mixin encodes a PNG preview as PNG already); `set_data` and the
view traits invalidate the cached preview so a notebook save renders the
current map. `_save_state` is a plain attribute, not a synced trait, as in
every other widget. Matplotlib names are imported at the top. Nothing outside
quantem.widget uses Plot2D.

Kept constructor parameters (12 plus `data`): `x y x_label y_label
colorbar_label title vmin vmax width max_width save_state`.

Removed constructor parameters (2): `cmap` (the `cmap` trait stays; the
browser's Color menu and Python assignment set it) and `height`
(`plot_height_px` stays at its 350 px default). Public methods `set_data` and
`figure` are unchanged.

Byte identity: the static PNG preview, the Matplotlib figure and every synced
trait for the tutorial map are identical; only the removed `_save_state`
trait left the state. `tests/plot2d/test_plot2d.py` (4 tests) is unchanged.

## ChooseLattice

`choose_lattice.py` 272 to 103 lines. `Dataset2d` is imported from
quantem.core instead of the lazy duck-typed probe; the `_reject_unknown_kwargs`
copy, the PNG helpers and the `StaticFallbackMixin` (whose preview was off by
default and tests-only) are gone. Nothing outside quantem.widget uses
ChooseLattice; the tutorial passes `cmap title` and reads `origin u v`.

Kept constructor parameters (2 plus `data`): `cmap title`.

Removed constructor parameters (10): `vmin vmax log_scale point_labels points
save_state notebook_preview_format notebook_preview_quality
notebook_preview_max_px` and `**kwargs`. Contrast is the 1st to 99th
percentile the default already used.

Removed public methods and properties (6): `set_points clear_points
points_array a1 a2` (assign the `points` trait instead) and the Python-only
`widget_version` trait.

Byte identity: the PNG frame and every synced trait for the tutorial image are
identical; only `widget_version` left the state.

Tests: `tests/choose_lattice/test_choose_lattice.py` (4 tests, under 1 s)
replaces `tests/test_choose_lattice.py`: the frame PNG, the Dataset2d title,
three clicks giving the origin and both vectors, clamping and the three-point
limit.

## Shared modules

Branch `slim-shared`. Gate per commit: full pytest, `npm run build`, `npx tsc --noEmit`,
`npx vitest run`, and a byte comparison of `export_html`, the static PNG preview,
`save_image`, `save_gif` and `export_svg` for tutorial-shaped inputs of all 12 widgets.
The module and line counts below are as measured on that branch, before the widget
packages landed and the trims in the Merge section above.

### Python: one home per helper

| Module | Owns | Was copied in |
|---|---|---|
| `render/figure.py` | `round_to_nice`, `unit_symbol`, `format_scale_label` (port of js/figure.ts) | show2d, show4dstem, render/gif |
| `colormap.py` | `Colormap` (14 names, the superset), `VALID_CMAPS`, `cmap_to_name`, `is_cmap_sequence`, `colorize` LUT | show2d (6 names), show3d (14), show3dslices; LUT at 9 sites |
| `state.py` | `ControlsMixin` (deleted at the wave-2 merge), `LiveMountMixin`, `StateFileMixin.save`, `SavedStateMixin.get_state` trim | controls x7, live mount x3, save x7, get_state x4 |
| `pages.py` | `PagesMixin` (page validators, hidden slots, star_page), `resolve_page_labels` | show2d, show3d (+ labels in show3dslices) |
| `panels.py` | `resolve_panel_ref`, `normalize_panel_refs`, `ordered_panel_indices`, `validate_panel_order`, `moved_panel_order`, `PanelsMixin` | show2d, show3d, show4dstem `compare_*` |
| `roi.py` | `sample_line`, `sample_profile_strip`, `RoiMixin` (15 methods) | show2d, show3d (+ sample_line in show4dstem) |
| `export.py` | `HtmlExportMixin` (6 traits, request observer, default path, bytes, write), `export_slug`, `write_widget_html` | 7 widgets |
| `fallback.py` | `StaticFallbackMixin` (moved from utils/static_fallback.py) | 7 widgets import it |

Widget files: show2d 9316 -> 8598, show3d 7625 -> 6930, show4dstem 7646 -> 7404,
show1d 4017 -> 3897, showeds 3739 -> 3689, showdiffraction 4087 -> 3998,
show3dslices 2533 -> 2400 lines. Shared modules total 1333 lines.

### JS: js/shared/

Histogram (x5 unified), InfoTooltip (x6), KeyboardShortcuts (x5),
useMobileViewport (x3), the LaTeX title renderer family (x2), the panel overlay
family (x2), pointToSegmentDistance (x5), export size / filename / shortcut
helpers, resolveDisplayBounds; useDebugFps and DebugPerfBadge deleted; show2d and
show3d draw their scale bars through figure.ts (`drawScaleBarInRegion` with
options reproducing each inline copy). Net -2324 source lines; bundles grow 0 to
0.5 % (each carries the superset component).

### Byte identity for the tutorial inputs

| Output | Result |
|---|---|
| export_html, every widget, exact and uint8 (bundle masked) | identical |
| export_html raw (bundle included) | differs after the JS commits only, by the bundle |
| static PNG preview, every widget | identical except Show4DSTEM (below) |
| Show2D save_image, export_svg; Show4DSTEM save_image; Show3D save_gif | identical |
| Show3D save_image | RGBA PNG -> RGB PNG (pixels identical; alpha was constant 255) |
| Show4DSTEM static PNG | scale bar label "50 px" -> "50 pixels" |

Intentional changes, all from using the js/figure.ts label rule everywhere:
the old Show4DSTEM formatter mapped every non-Å/mrad unit to "px" and printed
sub-1 px as `.1f`; the old GIF formatter used banker's rounding and never
re-laddered 0.5 nm to "5 Å". The live canvas already showed the new labels.
Other status-text changes (no export bytes): Show3D's uint8 export label reads
"uint8" instead of "encoded uint8"; "Exporting HTML..." now names the mode;
Show3D `add_roi` appends a new ROI like Show2D instead of editing the selected
one; Show3D panel error messages say "label" like Show2D.

### Not unified, and why

- The three paged-input normalisers (`_normalise_show2d_pages`, `_normalise_show3d_pages`,
  `_normalise_show3dslices_pages`): different input schemas (images / stacks /
  volumes, different dict keys, 4D vs 5D arrays). Only the page-label resolution is shared.
- `set_hidden_panels`, `hide_panel`, `visible_panels`, `_validate_hidden_panels`:
  Show2D's folder item pages hide absolute panels, Show3D hides page slots; the
  bodies differ in substance. Show3D stars frames per panel, Show2D stars panels.
- The static PNG renderers: Show2D's matplotlib gallery (reused by Show3D),
  Show4DSTEM's two-panel PIL composite, Show1D, ShowEDS, Plot2D and ChooseLattice
  draw different layouts. One renderer would change their PNG bytes.
- Per-widget `_clone_for_html_export`, `_normalise_html_export_options`,
  `_default_html_export_path` shapes, Show4DSTEM report export, ShowEDS
  sum-binned export (`_clone_for_html_export` bins the cube or data folder).
- `_export_light` stays in the five widgets that have it: adding it to Show4DSTEM
  and ShowDiffraction would add a synced key to their exported state.
- Hand-off traits (`handoff_*`, Show2D / Show3D / Show1D) are a tests-only feature
  left for the per-widget decision.
- JS: `formatStat` (three different policies), ShowPtycho's Histogram and
  `formatSavedBytes`, the GB-tier size formatters in show4dstem and showeds, and
  the per-widget filename composers.
