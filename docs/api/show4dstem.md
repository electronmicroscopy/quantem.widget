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

# Dynamic folder browse: first ready master paints now; others stay lazy.
w = Show4DSTEM.from_folder(
    "/data/session",
    backend="cuda",
    gpus=[0, 1],
    det_bin=1,
    dtype="u8",
    page_budget="auto",
    view_mode="multiple",
    compare_cols=3,
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
    backend="cuda",          # optional; omit for the default loader route
    gpus=[0, 1],             # round-robin lazy frame placement
    det_bin=1,
    dtype="u8",
    page_budget="auto",     # GPU-resident cache, not total dataset count
    view_mode="multiple",
    compare_cols=3,
    watch=True,
    watch_interval=2.0,
)
widget
```

`from_folder(...)` loads only the first ready master to infer shape and paint the
viewer. The remaining discovered masters are lazy slots: they allocate GPU
memory only when selected or included in the visible multiple grid. New ready
masters can be appended manually with `widget.poll_folder()` or automatically
with `watch=True` / `widget.watch_folder(interval=...)`. Hidden multiple-grid
panels are released from the lazy resident cache and are skipped by compare
computes until unhidden.

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
| Multiple grid | `view_mode="multiple"`, `compare_cols`, `compare_panel_gap_px`, `compare_max_panels`, `compare_layout` | Shows ready frames/datasets as synchronized virtual images sharing the detector ROI and scan cursor |
| Multiple DP source | `compare_dp_mode` | Shows either the average DP across visible multiple panels or the selected panel's DP |
| Multiple panel state | `compare_panel_order`, `compare_hidden_panels`, `compare_starred_panels`; `set_compare_panel_order()`, `hide_compare_panel()`, `show_all_compare_panels()`, `star_compare_panel()` | Saves/reuses panel order, hidden panels, and starred picks across cells, state files, and HTML export |
| Viewer chrome preset | `ui_mode` plus explicit `show_*` kwargs | Applies shared display presets; see [Viewer UI controls](viewer-ui) |
| Control visibility | `show_controls`, `controls_collapsed`; `collapse_controls()`, `expand_controls()`, `toggle_controls()` | Permanently remove controls or programmatically collapse/expand them for clean exports |
| Title visibility | `show_title` | Top title row shows/hides |
| Stats visibility | `show_stats` | DP, virtual-image, and FFT stats bars show/hide |
| Scale bar visibility | `show_scale_bar` | DP and virtual-image scale bars show/hide |
| Scan-path playback | `path_playing`, `path_index`, `path_interval_ms` | Sweeps the probe across the scan |
| k-space calibration | `k_pixel_size`, `k_calibrated` | Diffraction axes read in mrad when calibrated |
