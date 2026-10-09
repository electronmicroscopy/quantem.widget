# Show4DSTEM

## Compare diffraction patterns side by side

For a live notebook with several datasets, use
`Show4DSTEM(stack, view_mode="multiple", compare_dp_mode="all")`.
Each visible dataset shows its native diffraction pattern at the same scan
position. Circle, Square or Rect scan selection with Mean compares each
dataset over identical scan positions, never an average across datasets.
Dragging the detector in any diffraction tile updates the shared virtual
detector and all virtual images before release.

The diffraction grid shares the virtual-image grid's zoom, pan, reset,
scale bars and column layout. Playback is available in Single view only.
The `all` mode requires a live kernel; standalone export is not qualified for
this layout. Existing `selected` and `average` modes remain available.
Packed sources require their own reduction support and are not established
by this dense-array feature.

Public import:

```python
from quantem.widget import Show4DSTEM, read_4dstem
```

`Show4DSTEM` is one operator-facing factory. It chooses the viewer from its
input:

- a file path (`*_master.h5`, HDF5 or `.npy`) opens as `read_4dstem(path)`
  does: a `quantem.gpu.io.Dataset4dstemGPU` when `quantem.widget[cuda]` or
  `[mps]` is installed and a GPU is present, else a quantem core
  `Dataset4dstem` read whole into host memory (one printed line with the time
  taken; a file larger than 80% of the available memory is refused, never
  binned or cropped) and reduced on `device=`;
- a `Dataset4dstemGPU` (from `read_4dstem` on a GPU or `quantem.gpu.io.load`)
  opens as a live view over its GPU storage;
- a quantem core `Dataset4dstem` (numpy- or tensor-backed) or the widget's
  stand-in opens as its array or tensor, with its name as the title and its
  calibration on the scale bars;
- a list of datasets with one scan and detector shape opens as a comparison
  grid labelled by dataset name (the file name for `read_4dstem`);
- a NumPy array or Torch tensor opens in the base viewer, which also supports
  browser WebGPU compute and offline export.

The viewer shows the same virtual images, patterns and fitted disk for each
of these inputs. Counts are stored in
  the narrowest integer type that holds the largest count, which is known only
  after reading, so the refusal states the size for each type the file's dtype
  and detector bit depth allow, for example `needs between 9.7 GB and 77.3 GB`
  with `19.3 GB as uint16` for a 512 x 512 scan of 192 x 192 uint32 frames.

`device=` places a NumPy array or a dense file read: `"auto"` (CUDA, then
Apple MPS, then CPU, printed once), `"cuda"`, `"cuda:N"`, `"mps"` or `"cpu"`.
A tensor stays on its device unless `device=` names another. Virtual images,
the mean pattern and scan-ROI patterns of a dense tensor are integer sums in
torch on that device, divided once in float64 for a mean; quantem.gpu's
encoded path gives the same numbers.

Canonical forms:

```python
# One acquisition: on the GPU with quantem.gpu (CUDA or MPS), else read densely.
viewer = Show4DSTEM(read_4dstem(path))

# Part of the scan: (row_start, row_stop, col_start, col_stop), exclusive stops.
viewer = Show4DSTEM(read_4dstem(path), scan_region=(128, 384, 128, 384))

# Several acquisitions: one shared diffraction ROI, one virtual image per file.
viewer = Show4DSTEM(read_4dstem([path1, path2, path3]), compare_cols=3)

# A master by path: the same as Show4DSTEM(read_4dstem(path)).
viewer = Show4DSTEM("/data/session/scan_master.h5")

# Every ready master in a folder; masters completed later are appended.
viewer = Show4DSTEM.from_folder("/data/session")

# An array in the Python session: browser-owned compute and standalone export.
viewer = Show4DSTEM(array, backend="webgpu")
viewer.export_html("show4dstem.html")
```

Use the CLI `quantem show4dstem ... --backend webgpu --html` folder export for
a standalone browser viewer over large HDF5 acquisitions.

## Encoded acquisitions

On a GPU, `read_4dstem(path)` (which calls `quantem.gpu.io.load(path)`) keeps
the acquisition ANS encoded on the GPU at native detector
sampling and count dtype. A 512 x 512 x 192 x 192 uint16 Arina scan, 18 GiB as
a dense array, occupies about 0.1 to 2 GiB depending on its counts. The viewer
never expands it: virtual images are summed on the encoded storage, and each
diffraction pattern comes from a bounded read. CUDA and MPS acquisitions use
the same viewer.

`offline=True` packs a GPU acquisition for the browser as it packs an array:
the acquisition is read in small scan windows into the packed host array, so
the GPU never holds the dense cube, and a pack above the 2 GB budget is skipped
with one printed line. `h5_urls=` (browser reads of the source files) raises
for an acquisition. `export_html` works from a live view: an interactive
export reads the acquisition in small scan windows into the embedded array (a
host copy of the full data at the chosen `dtype` and binning), and a report
export (`export_kind="report"`, static PNG pages) embeds no raw 4D data. For a
standalone browser viewer over the source files, use the CLI
`--backend webgpu --html` folder export.

The viewer borrows the acquisition. Keep a handle when you plan to release GPU
memory, and close it after the viewer (this and the next sections use
quantem.gpu directly and need a GPU):

```python
from quantem.gpu.io import load

loaded = load(path)
viewer = Show4DSTEM(loaded)
viewer

# later, when you are done with this dataset
viewer.close()
loaded.close()
```

## Backend ownership

Fit the diffraction disk once and supply the same geometry to the viewer and
virtual detectors:

```python
from quantem.gpu import detector
from quantem.gpu.io import load

loaded = load(path)
center, radius = detector.fit_probe(detector.mean(loaded))
Show4DSTEM(loaded, center=center, bf_radius=radius)
```

The center is `(row, col)` and the radius is in detector pixels. Supplying
both skips the viewer's automatic disk estimation. `fit_probe` estimates disk
geometry, not the complex probe or its aberrations. `detector.mean` and the
other `quantem.gpu.detector` products (`bf`, `adf`, `masked_sum`, or a
`detector.prepare(loaded)` session for repeated queries) read the encoded
storage directly.

For a bounded Torch tensor on the same GPU, read a scan region:

```python
patterns_t = loaded.read(scan_region=(100, 164, 100, 164))  # (64, 64, det_row, det_col)
Show4DSTEM(patterns_t, center=center, bf_radius=radius)
```

Show4DSTEM has two acceleration surfaces:

- **Live Python-backed viewers** compute in the kernel on CUDA or MPS, over an
  encoded acquisition, a tensor, or an array.
- **Exported browser viewers** use the packed HTML or folder payload and
  browser WebGPU. After export, interaction does not depend on Python, Torch,
  CUDA, or MPS.

`Show4DSTEM` is one class in `quantem.widget.show4dstem`: an acquisition from
`io.load` becomes a bounded view inside the constructor, and every other input
is a dense tensor on its device.

## Live scope folders

For real-time processing on a microscope or acquisition workstation, open the
acquisition folder directly:

```python
from quantem.widget import Show4DSTEM

widget = Show4DSTEM.from_folder(
    "/data/live-scope-session",
    scan_size=512,          # keep only 512 x 512 scans in a mixed folder
    columns=5,              # grid width
    page_size=10,           # datasets per page
    watch_interval=2.0,     # seconds between folder polls
)
widget
```

`from_folder(...)` loads every ready `*_master.h5` (`pattern=`, `recursive=`)
with `quantem.gpu.io.load` into encoded storage on one CUDA device or the Apple
GPU (`backend=`, `device=`), at full detector resolution. At about 0.1 to 2 GiB
per 512 x 512 x 192 x 192 scan, the whole folder stays resident without
detector binning or paging. The viewer opens once the first master is loaded;
the others join the comparison grid in the background, and
`widget.wait_for_folder()` blocks until they have.

Masters that are not completely written are skipped (`ready_only=True`). When
the folder mixes geometries, the largest group sharing one scan and detector
shape is shown; `verbose=True` reports the skipped count. Use `scan_size=` or a
narrower `pattern=` to choose another group, and `max_masters=` or
`min_masters=` to bound the count. Other keyword arguments, such as
`compare_dp_mode="selected"` or `title=`, go to the viewer.

With `watch=True` (default), a master that completes while the viewer is open
is appended once its header signature is unchanged on two consecutive polls, so
a file still being written is never read. A compact title-area badge
distinguishes a live `Watching` worker, `Updating`, `Waiting for file
completion`, a corrective `Watch error`, and `Stopped`; `watch=False` opens a
fixed snapshot with no badge.

The folder lifecycle matches Show2D and Show3D:

```python
widget.wait_for_folder()                  # block until the opening masters are loaded
new_datasets = widget.poll_folder()       # append newly completed masters now
widget.stop_folder_watch()                # pause background discovery
widget.watch_folder(interval=1.0)         # resume discovery
widget.free()                             # close the acquisitions from_folder loaded
widget.close()                            # stop folder work, release them, close the widget
```

Folder watching is append-only. Known masters are not duplicated, incomplete
or externally linked masters wait until they are readable, and removing a file
does not delete a dataset from an active scientific view.

To verify the live path, introduce genuine master/chunk files while one
Jupyter widget is mounted and measure both discovery/control paint and the
requested virtual-image/diffraction paint.

This path reads the original master data, not cached thumbnails.

On Apple Silicon the same call loads onto the Apple GPU; pass `backend="mps"`
to require it:

```python
widget = Show4DSTEM.from_folder(
    "/data/live-scope-session",
    backend="mps",
    scan_size=512,
    title="Live 4D-STEM",
)
widget
```

GPU memory belongs to the loaded acquisitions and the Python session, not to
the visual widget alone. `free()` and `close()` release the acquisitions that
`from_folder` loaded; acquisitions you pass to `Show4DSTEM(...)` stay open until
you call their `close()`. The live widget shows a compact GPU memory label in
its title row when CUDA or MPS memory is visible. Exported HTML has no live
Python GPU allocation, so it does not expose a "free GPU memory" control.

## Compute SSB

With a live kernel on CUDA or MPS, the **Compute SSB** control, or
`viewer.compute_ssb()`, runs `quantem.gpu.SSB(...).find_aberrations(...)` on the
current 4D frame. It attaches the SSB phase and the aligned DPC row/col maps as
virtual-image sources and switches the virtual image to the phase. Supply the
microscope calibration when you open the viewer:

```python
from quantem.gpu.io import load

loaded = load(path)
viewer = Show4DSTEM(
    loaded,
    ssb_voltage_kV=300,
    ssb_semiangle_mrad=30,
    ssb_scan_sampling_A=0.5,
)
phase = viewer.compute_ssb()   # 200 trials plus Nelder-Mead refinement by default
```

The beam energy comes from `ssb_voltage_kV`, and the fit uses every detected
bright-field pixel (`ssb_bf_intensity_threshold`, `ssb_bf_radius`).
`ssb_n_trials`, `ssb_refine`, and `ssb_seed` control the search. Compute SSB
needs one 4D frame on a square 128, 256, or 512 scan grid. An encoded
acquisition goes to SSB without a dense copy; SSB decodes only the
bright-field disk.

## Multiple grid

Use `view_mode="multiple"` when the extra frame axis represents multiple
acquisitions that should be inspected side by side. The viewer keeps the
standard diffraction-panel workflow: one shared detector ROI, one shared scan
cursor, and one Dataset slider. The virtual-image side becomes a grid of ready
frames or datasets. Use `view_mode="single"` for one-at-a-time browsing. A list
of datasets from `read_4dstem` and `Show4DSTEM.from_folder(...)` open in
multiple mode by default.

```python
from quantem.widget import Show4DSTEM, read_4dstem

widget = Show4DSTEM(
    read_4dstem([path1, path2, path3, path4]),
    view_mode="multiple",
    compare_cols=2,
    compare_panel_gap_px=0,
    compare_max_panels=4,
    compare_dp_mode="selected",
)
widget
```

`compare_cols=0` lets the frontend pick a responsive layout. `compare_layout`
accepts `"side"` and `"top"` for placing the shared diffraction panel next to
or above the multiple grid. Positive `compare_cols` values are treated as the
maximum grid columns on desktop; narrow/mobile viewports cap the grid at two
columns so the tiles remain touch-friendly. A comparison of encoded
acquisitions never stacks them into one 5D array; each tile reads its own
acquisition.

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

The constructor default is `compare_dp_mode="average"`, which shows the mean
diffraction pattern at the current scan position across visible ready multiple
panels. For tilt-series and dataset-review demos, prefer
`compare_dp_mode="selected"` so the diffraction panel follows the clicked or
active dataset. Use `"average"` only when the mean diffraction pattern across
the current visible page is the intended measurement.

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
| `export_kind="report"` | Sharing a curated folder/multiple-grid result or saving a compact screening report | Static PNG virtual-image pages plus a representative diffraction pattern | Page-aware; folder data is rendered page by page and raw 4D tensors are not embedded |
| `export_kind="interactive"` | The recipient must drive the actual 4D dataset offline in the browser | Raw 4D payload, explicitly encoded as `uint8` or `uint16` and optionally binned | Can be large; use dtype and binning deliberately before sending. Needs the 4D array in the Python session, so it applies to viewers opened from an array or tensor |

Quick decision rule:

- Use `report` first for large folders, many masters, starred/hidden panel
  curation, and collaborator screening.
- Use `interactive` only when the exported browser page must still recompute new
  detector ROIs from raw 4D data.
- Use the CLI `quantem show4dstem ... --backend webgpu --html --bin 1 --dtype uint8`
  for a terminal-made full-detector browser artifact that reads source H5 files.
- Use `--dtype uint16` when a no-notebook user needs the wider detector-count
  range and accepts the larger browser/GPU memory footprint.
- Keep the original notebook or Python script when the recipient needs to keep
  doing analysis, not just view an export.

Report exports are the safe default for large folders:

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
backendless Show4DSTEM widget, not just a report. They embed the viewer's 4D
array; a viewer over `io.load` acquisitions reads it in small scan windows, so
the export copies the whole acquisition to host memory. Open the viewer on a
bounded `loaded.read(scan_region=...)` to export part of a scan:

```python
widget = Show4DSTEM(array)
widget.export_html(
    "show4dstem_interactive.html",
    export_kind="interactive",
    dtype="uint8",       # "uint16" keeps the exact integer range but is larger
    scan_bin=2,          # real-space mean bin before embedding raw 4D
    det_bin=4,           # detector mean bin before embedding raw 4D
)
```

Full native interactive export, without detector or scan binning:

```python
widget.export_html(
    "show4dstem_full_interactive.html",
    export_kind="interactive",
    dtype="uint16",
    scan_bin=1,
    det_bin=1,
)
```

Equivalent CLI for users who do not want a notebook:

```bash
quantem show4dstem /data/session --backend webgpu --html --count 7 --bin 1 --dtype uint8 --out ~/Downloads
```

Both `scan_bin` and `det_bin` use mean binning, not summing. This keeps display
exports from saturating `uint8` and makes the file-size estimate in the GUI
match the binned payload shape. The GUI export menu labels the same distinction
as **HTML report: static PNG, no raw 4D** and **HTML interactive raw 4D**.
The interactive section offers a size-sorted ladder of `uint8`/`uint16`,
real-space-bin, and detector-bin presets so users can choose between a quick
preview, a practical offline browser file, and exact raw 4D HTML deliberately.

Choose export dtype deliberately:

| Dtype | Use for | Do not use for |
|---|---|---|
| `uint8` | Compact browser payloads, first-pass screening, tutorials, and audited low-count data. | Claims that need high detector counts unless values above 255 are known not to matter. |
| `uint16` | Full/native interactive exports, detector-detail review, and count-range-preserving browser payloads. | Small public demos or reports where the recipient only needs rendered virtual-image pages. |

`uint8` uses one byte per exported detector pixel and may clip/narrow detector
values above 255. `uint16` uses two bytes per exported detector pixel and can
produce much larger artifacts, but preserves the wider 0-65535 integer range.
The dtype choice matters for `export_kind="interactive"` because that path sends
raw 4D data to the browser. A report export remains a rendered PNG review
artifact even if `dtype="uint16"` is passed.

For LLM agents and scripted docs, prefer these explicit parameter names:

```python
widget.export_html(
    path="show4dstem_report.html",
    export_kind="report",
    dataset_scope="unhidden",
    dtype="uint8",
    scan_bin=2,
    det_bin=8,
)
```

Do not describe a report export as "raw" or "exact"; it is a rendered review
artifact. Do not describe a `uint8` interactive export as exact unless the
detector count range was audited. Always mention the `scan_bin` and `det_bin`
values in a figure caption, notebook markdown cell, or handoff note.

See the copyable [Show4DSTEM export recipes](../tutorials/show4dstem_export.md)
for terminal commands, report settings, and interactive raw-4D settings.

## WebGPU HDF5 Folder

For large HDF5 acquisitions, prefer the CLI folder export. It keeps the
compressed `*_master.h5` family on disk and lets the browser read local HDF5
chunks through a folder grant or local range server. Startup should not wait on
a CLI-time full-stack conversion. Do not make normal CLI launches depend on
precomputed `profile.bin`/`com.bin` sidecars; those are generated products, not
the fast click-to-open path.

```bash
quantem show4dstem /path/to/h5_family --backend webgpu --html --bin 1 --count 1
quantem show4dstem /path/to/h5_family --backend webgpu --html --bin 1 --count 7
```

The folder writes `Show4DSTEM.command`, a hidden `.viewer/` server/viewer
folder, and linked `tilt_NN_master.h5`/`tilt_NN_data_*.h5` files. On macOS,
double-clicking the command starts a local range-capable server and opens the
vendored viewer page in Chrome. The exported page ships the required browser
widget-manager assets, so the handoff does not depend on public CDNs. Rerunning
the same CLI into the same `--out` replaces the generated
`*_show4dstem_webgpu` viewer folder, which prevents stale HTML or stale links
from surviving a regeneration.

Double-clicking `index.html` directly is also supported in Chromium browsers
with the File System Access API: click **Open data folder** and grant the
export folder that contains `index.html`, `.viewer/`, and the anonymous H5
links. Use `Show4DSTEM.command` when you want the no-prompt local-server path.

`--dtype uint8` decodes the low byte of each count in the browser (compact
browse); the default `uint16` keeps native counts, and the bundle injects
`__QT_H5_DECODE_DTYPE="u2"` so the high bitplanes are kept.

The browser loader also honors these optional globals when injected before the
widget bundle:

| Global | Purpose |
|---|---|
| `__QT_H5_DECODE_DTYPE` | `"uint8"`, `"u2"`/`"uint16"`, `"native"`, or `"float32"` decode request |
| `__QT_H5_DET_BIN` | Detector mean-binning factor for local HDF5 loads |
| `__QT_H5_SCAN_REGION` | `(row_start, row_stop, col_start, col_stop)` scan crop |
| `__QT_H5_SOURCE_SCAN_ROWS`, `__QT_H5_SOURCE_SCAN_COLS` | Source scan shape when loading a cropped region |
| `__QT_H5_FETCH_WINDOW`, `__QT_H5_DECODE_QUEUE` | HTTP fetch/decode queue depth |
| `__QT_H5_LOCAL_GROUP`, `__QT_H5_LOCAL_WORKERS` | Browser local-file read/decode grouping |

The maintainer performance notes record the current seven-panel WebGPU
compare-grid signoff.

## Reference

```{eval-rst}
.. autoclass:: quantem.widget.show4dstem.Show4DSTEM
   :members: from_folder, export_html, compute_ssb, apply_preset, state_dict, save, load_state_dict, free, close, poll_folder, wait_for_folder, watch_folder, stop_folder_watch, pattern, virtual_image
```

```{note}
`scan_region=` applies to an acquisition from `io.load`; `backend="webgpu"`,
`offline` and `offline_dtype` apply to array and tensor input.
```

## Interactive controls

With a running kernel these recompute on the GPU backend (CUDA or MPS). In
`backend="webgpu"` mode, the same controls run in the browser with no
Python round trip - see [Performance](../maintainer/widget-performance).

| Control | Trait | Expected effect |
|---|---|---|
| Detector position (drag on diffraction) | `pos_row`, `pos_col` | Virtual image recomputes for that probe position |
| BF aperture radius | `bf_radius` | Bright-field disk grows/shrinks; virtual image updates |
| Aperture center | `center_row`, `center_col` | Recenters the detector on the unscattered beam |
| Detector ROI mode | `roi_mode`, `roi_active` | Switch BF / annular / rectangular detector |
| Annular inner / outer | `roi_radius_inner`, `roi_radius` | ADF annulus geometry |
| Virtual-image ROI | `vi_roi_mode`, `vi_roi_center_row`, `vi_roi_center_col` | Pick a real-space region to average its diffraction |
| FFT toggle | `show_fft`, `fft_window` | Power spectrum of the virtual image, drawn with sharp pixels |
| VI Smooth switch | `vi_smooth` | Off by default: sharp scan pixels; on: bilinear interpolation of the virtual image. The diffraction pattern is always drawn with sharp pixels |
| Multiple grid | `view_mode="multiple"`, `compare_cols`, `compare_panel_gap_px`, `compare_max_panels`, `compare_group_mode`, `compare_layout` | Shows ready frames/datasets as synchronized virtual images sharing the detector ROI and scan cursor; `compare_group_mode="all"` collapses pages into one dense grid |
| Multiple DP source | `compare_dp_mode` | Shows either the average DP across visible multiple panels or the selected panel's DP |
| Multiple panel state | `compare_panel_order`, `compare_hidden_panels`, `compare_starred_panels`; `set_compare_panel_order()`, `hide_compare_panel()`, `show_all_compare_panels()`, `star_compare_panel()` | Saves/reuses panel order, hidden panels, and starred picks across cells, state files, and HTML export |
| Viewer chrome preset | `ui_mode` plus explicit `show_*` kwargs | Applies shared display presets; see [Viewer UI controls](viewer-ui) |
| Control visibility | `show_controls`, `controls_collapsed` | Permanently remove controls or programmatically collapse/expand them for clean exports |
| Title visibility | `show_title` | Top title row shows/hides |
| Stats visibility | `show_stats` | DP, virtual-image, and FFT stats bars show/hide |
| Scale bar visibility | `show_scale_bar` | DP and virtual-image scale bars show/hide |
| Scan-path playback | `path_playing`, `path_index`, `path_interval_ms` | Sweeps the probe across the scan |
| k-space calibration | `k_pixel_size`, `k_calibrated` | Diffraction axes read in mrad when calibrated |
