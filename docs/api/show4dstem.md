# Show4DSTEM

Public import:

```python
from quantem.widget import load, Show4DSTEM
```

`Show4DSTEM` is a dispatcher/factory with one operator-facing API. It picks the
viewer from what `load(...)` returns and from the requested widget backend:
CUDA/Torch on Linux, raw Metal on Apple Silicon MPS loads, CPU fallback, or
browser WebGPU.

Canonical forms:

```python
# Auto-pick CUDA / MPS / CPU from the loaded data.
w = Show4DSTEM(load(path))

# Apple Silicon raw-Metal path, with sampling read from metadata when present.
w = Show4DSTEM(load(path, backend="mps", det_bin=4))

# Multi-dataset stack: one viewer, one Dataset slider.
w = Show4DSTEM(load([path1, path2, path3], det_bin=4))

# Live-kernel WebGPU: the browser owns virtual-detector compute.
w = Show4DSTEM(load(path), backend="web")

# Standalone backendless export for large data: HTML + companion data folder.
w = Show4DSTEM(load(path), backend="web", offline_codec="bslz4",
               data_url="show4dstem-data")
w.export_html("show4dstem.html")
```

`backend="browser"`, `backend="webgpu"`, and `offline=True` are compatibility
aliases for `backend="web"`.

## Reference

```{autodoc2-object} quantem.widget.show4dstem.Show4DSTEM
render_plugin = "myst"
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
Python round trip - see [Performance](../perf/index).

| Control | Trait | Expected effect |
|---|---|---|
| Detector position (drag on diffraction) | `pos_row`, `pos_col` | Virtual image recomputes for that probe position |
| BF aperture radius | `bf_radius` | Bright-field disk grows/shrinks; virtual image updates |
| Aperture center | `center_row`, `center_col` | Recenters the detector on the unscattered beam |
| Detector ROI mode | `roi_mode`, `roi_active` | Switch BF / annular / rectangular detector |
| Annular inner / outer | `roi_radius_inner`, `roi_radius` | ADF annulus geometry |
| Virtual-image ROI | `vi_roi_mode`, `vi_roi_center_row`, `vi_roi_center_col` | Pick a real-space region to average its diffraction |
| FFT toggle | `show_fft`, `fft_window` | Power spectrum of the virtual image |
| Scan-path playback | `path_playing`, `path_index`, `path_interval_ms` | Sweeps the probe across the scan |
| k-space calibration | `k_pixel_size`, `k_calibrated` | Diffraction axes read in mrad when calibrated |
