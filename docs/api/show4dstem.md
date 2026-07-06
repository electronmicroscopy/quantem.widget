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

For real-time processing on a microscope or acquisition workstation, keep one
Show4DSTEM viewer mounted and append new completed acquisitions into it. On the
Apple Silicon raw-Metal path, `load_macbook_datasets(...)` returns a lazy handle
that owns the live multi-dataset container:

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
visual widget alone. To release GPU memory, remove or replace the backend data
object, clear references, use backend-specific cleanup utilities when provided,
or restart the kernel/session. Exported HTML has no live Python GPU allocation,
so it should not expose a "free GPU memory" control.

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
| Viewer chrome preset | `ui_mode` plus explicit `show_*` kwargs | Applies shared display presets; see [Viewer UI controls](viewer-ui) |
| Control visibility | `show_controls`, `controls_collapsed`; `collapse_controls()`, `expand_controls()`, `toggle_controls()` | Permanently remove controls or temporarily collapse them behind the top GUI toggle |
| Title visibility | `show_title` | Top title row shows/hides |
| Stats visibility | `show_stats` | DP, virtual-image, and FFT stats bars show/hide |
| Scale bar visibility | `show_scale_bar` | DP and virtual-image scale bars show/hide |
| Scan-path playback | `path_playing`, `path_index`, `path_interval_ms` | Sweeps the probe across the scan |
| k-space calibration | `k_pixel_size`, `k_calibrated` | Diffraction axes read in mrad when calibrated |
