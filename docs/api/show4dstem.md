# Show4DSTEM

Public import:

```python
from quantem.widget import load, Show4DSTEM
```

`Show4DSTEM` is a dispatcher/factory with one operator-facing API. It picks the
viewer from what `load(...)` returns and from the requested widget backend:
CUDA/Torch on Linux, raw Metal on Apple Silicon MPS loads, CPU fallback, or
browser WebGPU.

The MPS code is intentionally a backend implementation, not a separate public
viewer to choose in notebooks. Direct names such as `Show4DSTEMMPS`,
`Show4DSTEM_MACBOOK`, and `show_4dstem_mps(...)` stay importable for old
notebooks and backend tests, but new code should use the single factory:
`Show4DSTEM(load(path, backend="mps", det_bin=...))`.

Canonical forms:

```python
# Auto-pick CUDA / MPS / CPU from the loaded data.
w = Show4DSTEM(load(path))

# Apple Silicon raw-Metal path, with sampling read from metadata when present.
w = Show4DSTEM(load(path, backend="mps", det_bin=4))

# Multi-dataset stack: one viewer, one Dataset slider.
w = Show4DSTEM(load([path1, path2, path3], det_bin=4))

# Multi-dataset comparison: one shared diffraction ROI, many virtual images.
w = Show4DSTEM(
    load([path1, path2, path3], det_bin=4),
    view_mode="multiple",
    compare_cols=3,
)

# Dynamic folder browse: first page paints now; the rest preload if they fit.
w = Show4DSTEM.from_folder(
    "/data/session",
    gpus=[0, 1],
    det_bin=1,
    columns=5,
    page_size=5,
    compare_dp_mode="selected",
    warm_cache=True,
    watch=True,
)

# Apple Silicon live acquisition folder: dataset 0 appears first, then newly
# completed *_master.h5 files append into the same Dataset slider.
from quantem.widget.multidataset_mps import load_macbook_datasets

live = load_macbook_datasets("/data/live-scope-session", det_bin=4, scan_size=512)
w = Show4DSTEM(live)
live.watch_master_folder("/data/live-scope-session", interval=2.0, scan_size=512)

# Live-kernel WebGPU: the browser owns virtual-detector compute.
w = Show4DSTEM(load(path), backend="web")

# Standalone backendless export for large data: HTML + companion data folder.
w = Show4DSTEM(load(path), backend="web", offline_codec="bslz4",
               data_url="show4dstem-data")
w.export_html("show4dstem.html")
```

`backend="browser"`, `backend="webgpu"`, and `offline=True` are compatibility
aliases for `backend="web"`.

## Backend ownership

Show4DSTEM has two different acceleration surfaces:

- **Live Python-backed viewers** use the data object returned by ``load(...)``.
  Depending on hardware this may be CUDA/Torch, raw Metal/MPS on Apple Silicon,
  Torch-MPS for specific paths, or CPU fallback.
- **Exported/offline browser viewers** use the packed HTML/folder payload and
  browser WebGPU when available. After export, interaction should not depend on
  Python, Torch, CUDA, or MPS.

On Apple Silicon, prefer the raw Metal/MPS loading path for large first-pass
browsing because it can control chunking, detector binning, and dtype more
tightly than a generic Torch-MPS tensor path. Torch-MPS remains useful for
some tensor workflows, but reports should say which path was used.

MPS loading also has an automatic preflight memory guard. If a no-bin or large
Metal allocation would exceed the Mac's conservative working-set budget,
`load(..., backend="mps")` fails before allocating and recommends a safer
`det_bin` value. This is intentional: it protects laptop sessions from
unresponsive unified-memory pressure while keeping the MPS backend automatic.

Routing lives in `quantem.widget.show4dstem_factory`: chunked MPS payloads and
lazy MacBook multi-dataset handles go to the raw-Metal backend, while CUDA/CPU
arrays and CUDA 5D dataset wrappers stay on the universal base viewer. This
keeps the user-facing API stable while backend-specific code stays isolated.

## Live scope folders

For real-time processing on a microscope or acquisition workstation, prefer the
direct folder-backed API when you want ready masters to become available without
materializing a full 5D stack:

```python
from quantem.widget import Show4DSTEM

widget = Show4DSTEM.from_folder(
    "/data/live-scope-session",
    gpus=[0, 1],             # selects CUDA and distributes lazy frames
    det_bin=1,
    columns=5,
    page_size=5,
    compare_dp_mode="selected",
    warm_cache=True,
    watch_interval=2.0,
)
widget
```

`from_folder(...)` keeps the folder as lazy slots instead of materializing a full
5D stack before first paint. The initial page loads first, then the default
`preload_all_if_fits=True` policy calculates the complete raw footprint from the
known frame shape and dtype. If that footprint fits the selected GPUs, every
unhidden dataset loads in the background across those GPUs. If it does not fit,
the viewer keeps full-resolution lazy paging; it does not silently detector-bin,
real-space-bin, or narrow the dtype. Set `preload_all_if_fits=False` to keep the
page-on-demand policy even when the series would fit.

The title row reports both GPU allocation and raw residency, for example
`raw 20/20 resident` or `raw 4/20 resident`. New ready masters can be appended
manually with `widget.poll_folder()` or by the default folder watcher. Each
append re-evaluates whether the complete unhidden series still fits. Use
`watch=False` for a fixed folder or a script that calls `poll_folder()`
explicitly. Hidden multiple-grid panels are released from the raw resident cache
and skipped by compare computes until unhidden.

Discovery and metadata do not copy raw 4D arrays to a GPU. A newly appended
master starts lazy, then joins the background full-series preload only when the
new total still fits. Otherwise, selecting it or including it in a visible page
loads it on demand. `page_budget` bounds raw GPU residency and evicts older raw
pages as needed; reduced virtual-image cache entries use separate host-memory
limits. Appending a master invalidates or warms only affected comparison pages,
so unrelated cached pages remain fast.

The folder lifecycle matches Show2D and Show3D:

```python
new_datasets = widget.poll_folder()       # append newly ready masters now
widget.stop_folder_watch()                # pause background discovery
widget.watch_folder(interval=1.0)         # resume discovery
widget.close()                            # stop watchers/workers and close
```

Folder watching is append-only. Known masters are not duplicated, incomplete
or externally linked masters wait until they are readable, and removing a file
does not silently delete a dataset from an active scientific view.

`warm_cache=True` preserves the original detector data. It loads raw masters in
memory-aware batches, computes the standard BF/ABF/ADF/HAADF virtual images,
keeps only those small 2D results in host memory, and releases raw pages as the
worker advances when full residency is unavailable. Cold pages still pay real
disk/decompression cost; warmed page and preset changes reuse cached results.
`compare_dp_mode="selected"` keeps scan-position movement responsive without
loading every master just to average the diffraction panel.

This path uses the original master data at the requested `det_bin` and `dtype`.
It does not use ShowFolder's cached thumbnails. Set `det_bin=1` and keep the
count-preserving dtype when full detector resolution is required.

For folders with tens or hundreds of masters, `page_size` is the number of
datasets shown and preloaded together. `columns` controls the grid width:

```python
widget = Show4DSTEM.from_folder(
    "/data/session",
    gpus=[0, 1],
    det_bin=1,
    columns=3,
    page_size=12,
    page_budget=4,          # resident lazy/GPU cache
    compare_group_mode="paged",
    compare_cache_pages=16, # reduced VI page cache, not raw 4D VRAM
)

widget.set_compare_page(1)       # second zero-based page
widget.next_compare_page()
widget.previous_compare_page()
widget.show_compare_all_groups() # collapse pages into one dense grid
widget.show_compare_paged_groups()
widget.preload_all_datasets()    # re-run the fit check in the background
widget.wait_for_dataset_preload(timeout=120)  # deterministic scripts/tests
```

The page control appears in the multiple-grid header whenever the visible
dataset count exceeds `page_size`. `compare_group_mode="paged"` shows
one group at a time with precise group buttons plus a compact play/pause
control. `compare_group_mode="all"` collapses all visible groups into one dense
grid for screening tens or hundreds of reduced virtual images. All mode computes
lazy folder data in `page_size` batches when the complete raw series does not
fit. When it does fit, display pages reuse the already resident raw frames.
Hidden panels remain hidden and are not recomputed. Use `page_budget` for the
raw resident cache policy and `page_size` for the compute/display batch size.
Existing code may continue to use `compare_cols` and `compare_max_panels`; new
folder-browse code should use the shorter names.

Automatic residency uses 98% as an upper data fraction, then keeps a bounded
adaptive workspace margin for reductions and allocator metadata. Unlike the old
fixed 4 GiB reserve, the margin scales from the known frame size and GPU size.
Pass `page_reserve_vram_bytes=` or `page_max_vram_bytes=` for an explicit policy.
Pass `gpus=[0, 1]` to spread frames round-robin over specific cards, or
`gpus="all"` to use every CUDA device visible to the process.

The multiple-grid BF/ABF/ADF/HAADF previews are cached as reduced float32
virtual-image pages. This lets page 1 -> page 2 -> page 1 return the already
computed thumbnails without keeping page 1's raw 4D tensors in VRAM. Tune
`compare_cache_pages` for how many reduced pages to keep and
`compare_cache_max_bytes` for the host-memory cap. This cache is separate from
`page_budget`: `page_budget` controls raw 4D GPU residency, while
`compare_cache_pages` controls small rendered page previews.

For folder-backed multi-master browsing, `dtype="auto"` is resolved to a stable
`u16` load dtype. A lazy series needs every page to share shape and dtype; using
per-master auto-narrowing could otherwise make one page `uint16` and a later
page `uint8`. Pass `dtype="u8"` explicitly only when lossy clipping is acceptable
for browsing.

If a folder contains mixed scan shapes, `from_folder(...)` chooses the largest
metadata-compatible group and warns with the skipped count. This prevents a
mixed folder from failing halfway through page loading. Use `scan_size=` or a
narrower `pattern=` when you intentionally want a smaller group.

On the Apple Silicon raw-Metal path, `load_macbook_datasets(...)` remains the
backend-specific live handle:

```python
from quantem.widget import Show4DSTEM
from quantem.widget.multidataset_mps import load_macbook_datasets

live = load_macbook_datasets("/data/live-scope-session", det_bin=4, scan_size=512)
widget = Show4DSTEM(live, title="Live 4D-STEM")
live.watch_master_folder("/data/live-scope-session", interval=2.0, scan_size=512)
widget
```

`watch_master_folder(...)` polls for `*_master.h5` files, ignores masters whose
linked data files are not present yet, and appends only new acquisitions. The
notebook cell and viewer stay stable; the dataset slider grows as files become
ready. Use `live.stop_watch()` before switching to a different folder.

GPU memory is owned by the loaded data object and the Python session, not by the
visual widget alone. The live widget shows a compact GPU memory label in its
title row when CUDA or MPS memory is visible. To release all memory, remove or
replace the backend data object, clear references, use backend-specific cleanup
utilities when provided, or restart the kernel/session. Exported HTML has no
live Python GPU allocation, so it should not expose a "free GPU memory" control.

## Multiple grid

Use `view_mode="multiple"` when the extra frame axis represents multiple
acquisitions that should be inspected side by side. The viewer keeps the
standard diffraction-panel workflow: one shared detector ROI, one shared scan
cursor, and one Dataset slider. The virtual-image side becomes a grid of ready
frames or datasets. Older notebooks that pass `view_mode="compare"` still load
as the same multiple-grid mode. Older `view_mode="temporal"` inputs are treated
as `view_mode="single"` because the one-at-a-time dataset browser is the single
view.

```python
from quantem.widget import load, Show4DSTEM

widget = Show4DSTEM(
    load([path1, path2, path3, path4], det_bin=4),
    view_mode="multiple",
    compare_cols=2,
    compare_panel_gap_px=0,
    compare_max_panels=4,
    compare_dp_mode="average",
)
widget
```

`compare_cols=0` lets the frontend pick a responsive layout. `compare_layout`
accepts `"side"` and `"top"` for placing the shared diffraction panel next to
or above the multiple grid. Positive `compare_cols` values are treated as the
maximum grid columns on desktop; narrow/mobile viewports cap the grid at two
columns so the tiles remain touch-friendly. On lazy MPS multi-dataset loads, the
grid starts with the first decoded dataset and appends tiles as the background
loader marks additional datasets ready; it does not materialize a full 5D stack
just to build the comparison.

When the visible set is larger than `compare_max_panels`, the multiple grid is
paged like Show2D/Show3D galleries. `compare_page_idx` is zero-based and
`compare_page_count` is synced widget state, so notebooks can drive pages
programmatically or save/restore the current page with the rest of the widget
state.

Set `compare_group_mode="all"` or call `widget.show_compare_all_groups()` to
collapse all visible pages into one dense comparison grid. Call
`widget.show_compare_paged_groups()` to restore page-by-page browsing. The
shared diffraction panel keeps using the active page for `compare_dp_mode="average"`
so scan-position drags stay responsive even when the virtual-image grid is
showing every reduced panel.

`compare_panel_gap_px=0` renders the virtual-image grid edge-to-edge for dense
screening. Increase it when a report or presentation needs visible gutters
between panels. Mouse-wheel or trackpad scroll over a multiple tile zooms the
shared virtual-image grid instead of scrolling the page; double-click a tile to
reset the compare zoom. The single-panel diffraction and virtual-image canvases
use the same scroll-to-zoom behavior.

The shared diffraction panel defaults to `compare_dp_mode="average"`, which
shows the mean diffraction pattern at the current scan position across visible
ready multiple panels. Use `compare_dp_mode="selected"` when the diffraction
panel should follow the clicked/active dataset instead.

Multiple panel curation is stored on the widget, so a notebook can reuse the
same state in a later cell or saved HTML export:

```python
widget.set_compare_panel_order(["scan-3", "scan-0", "scan-1", "scan-2"])
widget.hide_compare_panel("scan-4")
widget.star_compare_panel("scan-3")

state = widget.state_dict()
another_widget.load_state_dict(state)
```

The GUI exposes the same state: the star and hide icons live on each multiple
tile, the reorder button enables drag-and-drop or click-then-click ordering, and
the multiple toolbar can restore hidden panels or reset the saved panel state.

## Exporting reports and raw 4D viewers

Show4DSTEM has two HTML export modes with different goals:

| Export kind | Use when | Data included | Memory behavior |
|---|---|---|---|
| `export_kind="report"` | Sharing a curated folder/multiple-grid result or saving a compact screening report | Static PNG virtual-image pages plus a representative diffraction pattern | Page-aware; lazy folder data is rendered page by page and raw 4D tensors are not embedded |
| `export_kind="interactive"` | The recipient must drive the actual 4D dataset offline in the browser | Raw 4D payload, optionally quantized/binned | Can be large; use binning deliberately before sending |

Report exports are the safe default for large lazy folders:

```python
widget.export_html(
    "show4dstem_report.html",
    export_kind="report",
    dataset_scope="unhidden",  # "current_page", "starred", or "all" also work
    scan_bin=2,                # mean-bin real space for smaller PNG pages
    det_bin=8,                 # mean-bin the representative DP thumbnail
    dtype="uint8",
)
```

Interactive raw exports remain available when the exported HTML needs the
backendless Show4DSTEM widget, not just a report:

```python
widget.export_html(
    "show4dstem_interactive.html",
    export_kind="interactive",
    dtype="uint8",       # "uint16" keeps the exact integer range but is larger
    scan_bin=2,          # real-space mean bin before embedding raw 4D
    det_bin=4,           # detector mean bin before embedding raw 4D
)
```

Both `scan_bin` and `det_bin` use mean binning, not summing. This keeps display
exports from saturating `uint8` and makes the file-size estimate in the GUI
match the binned payload shape. The GUI export menu labels the same distinction
as **HTML report: static PNG, no raw 4D** and **HTML interactive raw 4D**.
The interactive section offers a size-sorted ladder of `uint8`/`uint16`,
real-space-bin, and detector-bin presets so users can choose between a quick
preview, a practical offline browser file, and exact raw 4D HTML deliberately.

## Reference

```{eval-rst}
.. autoclass:: quantem.widget.show4dstem.Show4DSTEM
   :members:
   :show-inheritance:
```

```{note}
The generated reference above is the universal base viewer. The public
`quantem.widget.Show4DSTEM` factory accepts the same viewer options plus dispatch
options such as `backend="web"`, `offline_codec`, `data_url`, and
`export_html(...)`.
```

## Interactive controls

With a running kernel these recompute on the GPU backend (CUDA / MPS / CPU). In
`backend="web"` mode, the same controls run in the browser via WebGPU with no
Python round trip - see [Performance](../maintainer/widget-performance).

| Control | Trait | Expected effect |
|---|---|---|
| Detector position (drag on diffraction) | `pos_row`, `pos_col` | Virtual image recomputes for that probe position |
| BF aperture radius | `bf_radius` | Bright-field disk grows/shrinks; virtual image updates |
| Aperture center | `center_row`, `center_col` | Recenters the detector on the unscattered beam |
| Detector ROI mode | `roi_mode`, `roi_active` | Switch BF / annular / rectangular detector |
| Annular inner / outer | `roi_radius_inner`, `roi_radius` | ADF annulus geometry |
| Virtual-image ROI | `vi_roi_mode`, `vi_roi_center_row`, `vi_roi_center_col` | Pick a real-space region to average its diffraction |
| FFT toggle | `show_fft`, `fft_window` | Power spectrum of the virtual image |
| Multiple grid | `view_mode="multiple"`, `compare_cols`, `compare_panel_gap_px`, `compare_max_panels`, `compare_group_mode`, `compare_layout` | Shows ready frames/datasets as synchronized virtual images sharing the detector ROI and scan cursor; `compare_group_mode="all"` collapses pages into one dense grid |
| Multiple DP source | `compare_dp_mode` | Shows either the average DP across visible multiple panels or the selected panel's DP |
| Multiple panel state | `compare_panel_order`, `compare_hidden_panels`, `compare_starred_panels`; `set_compare_panel_order()`, `hide_compare_panel()`, `show_all_compare_panels()`, `star_compare_panel()` | Saves/reuses panel order, hidden panels, and starred picks across cells, state files, and HTML export |
| Viewer chrome preset | `ui_mode` plus explicit `show_*` kwargs | Applies shared display presets; see [Viewer UI controls](viewer-ui) |
| Control visibility | `show_controls`, `controls_collapsed`; `collapse_controls()`, `expand_controls()`, `toggle_controls()` | Permanently remove controls or programmatically collapse/expand them for clean exports |
| Title visibility | `show_title` | Top title row shows/hides |
| Stats visibility | `show_stats` | DP, virtual-image, and FFT stats bars show/hide |
| Scale bar visibility | `show_scale_bar` | DP and virtual-image scale bars show/hide |
| Scan-path playback | `path_playing`, `path_index`, `path_interval_ms` | Sweeps the probe across the scan |
| k-space calibration | `k_pixel_size`, `k_calibrated` | Diffraction axes read in mrad when calibrated |
