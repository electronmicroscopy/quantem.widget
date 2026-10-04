# ShowPtycho

Interactive ptychography aberration review for a prepared SSB reconstruction or
raw 4D-STEM data. `ShowPtycho` lets a microscopist tune defocus, astigmatism,
scan-detector rotation, phase flip, display contrast, and the FFT view while
watching the reconstructed phase update.

## Workflow and ownership

`quantem.widget` owns presentation, project layout, and browser export. It does
not implement CUDA, MPS, or WebGPU SSB kernels. Exact fitting and final
reconstruction enter through the shared `quantem.gpu.SSB` API and return one
`SSBResult`, regardless of compute backend. `ShowPtycho` consumes the prepared
`SSB` session so it can reuse resident buffers for notebook interaction; the
returned result remains the compact analysis output.

The canonical folder workflow is:

1. `quantem showptycho SOURCE` discovers native HDF5 masters and resolves the
   microscope geometry.
2. `quantem.gpu.SSB` performs the requested exact trials, Nelder-Mead
   refinement, and final reconstruction on the selected compute backend.
3. The widget writes one project containing the result, calibration,
   provenance, exact bright-field evidence, and browser launchers.
4. Reopening that project serves the existing files; it does not silently fit
   again or choose a different backend.

Backend-specific choices therefore stop at the `quantem.gpu` boundary. The
widget records both the software revision that produced a fit and the revision
that exported the project, so re-exporting never rewrites historical scientific
provenance.

## What The Browser Computes

In a live notebook, Python owns the reconstruction state and uses the available
backend, usually CUDA on Linux or Metal/MPS on Apple silicon. The browser is the
viewer.

In a WebGPU folder export, the browser owns the interactive review. The default
folder contains a small `index.html` viewer, calibration metadata under
`snapshots/`, and exact microscopy payloads under `source/`. The default browser
payload is a BF-column file (`source/bf_columns.u8` or `.u16`), so the browser
range-reads only the bright-field evidence it needs on open. Moving
C10, C12, phi12, or scan rotation makes the browser build BF-indexed `G(k)`
reducers transiently in GPU memory and run the SSB phase reconstruction from
those transient buffers. The export does not persist expanded float32 images or
complex64 BF reducers by default.

FFT is a display analysis of the current reconstructed phase. When the FFT panel
is visible, the widget computes the FFT from the latest phase image and redraws
that FFT panel. It is not re-running the raw detector preprocessing.

## Real-Space Crop And SSB Refit

For a region-specific probe/aberration fit, open the prepared CUDA SSB with the
raw master path. ShowPtycho then exposes `Crop` in the top-right action group
beside `Export` and `Reset`. Enable `Crop`, drag a rectangle on the phase panel,
inspect the live row/column region readout, then use `Refit SSB`.
The widget reloads only that scan region from the original HDF5 source, runs 200
SSB optimization trials followed by refinement, and replaces the phase/FFT and
their calibration with the result.

```python
from quantem.gpu import SSB
from quantem.widget import ShowPtycho

ssb = SSB.open(
    "reference_master.h5",
    backend="auto",
    semiangle_mrad=20.0,
    scan_sampling_A=0.276,
    voltage_kV=300.0,
)
result = ssb.find_aberrations(trials=200, refinement="nelder-mead")
w = ShowPtycho(
    ssb,
    source_file="reference_master.h5",
    fft_on=True,
)
```

The selection may be rectangular; each dimension must span at least `32` scan
positions. `Crop Reset` clears the selection without changing the current
reconstruction. This is a reconstruction operation, not a display crop.

The refit control is absent from standalone HTML/WebGPU exports and MPS-only
sessions. Those modes can inspect an existing result interactively, but only a
source-backed CUDA session has the raw detector data and SSB optimizer needed
to make a new scientific fit.

## Sample Tilt (Thick Crystals)

Standard SSB treats the sample as one thin plane. In a thick, tilted crystal
each depth sees a different defocus and sits shifted sideways by depth x tilt,
so the lattice washes out along the tilt: often only one set of fringes survives.
The **Sample tilt (thick SSB)** panel, above the aberration sliders, adds the
sample to the model:

- **tilt row / tilt col** (mrad, scan frame) and **thickness** (nm). Thickness
  0 is standard SSB, exactly. Thickness is the depth spread the model averages
  over, not a measured sample thickness.
- **Fit tilt** fits defocus, astigmatism, tilt and thickness together and moves
  every slider to the result. The fitted C10 is the defocus at mid-depth.
- The status line gives the tilt in the scan frame and in the ptychography
  object frame (the scan frame rotated by the scan-detector rotation). The
  object-frame value seeds a quantem.thick reconstruction directly.

Open with the fit already run: the widget shows the session's latest fit.

```python
from quantem.gpu import SSB
from quantem.widget import ShowPtycho

ssb = SSB.open(path, voltage_kV=300.0, semiangle_mrad=30.0, scan_sampling_A=0.495,
               det_sampling=0.5554, rotation_angle_deg=-8.6)
standard = ssb.find_aberrations()                 # defocus + astigmatism
ShowPtycho(ssb)                      # standard SSB
tilted = ssb.find_aberrations(tilt=True)          # + sample tilt and depth spread, jointly
ShowPtycho(ssb)                      # opens on the fitted tilt
pd.concat([standard.report(), tilted.report()])
```

Saved calibrations store the tilt (`tilt_mrad` scan frame, `tilt_object_mrad`,
`depth_spread_nm`) and reopening one restores the panel.

Where it runs: live notebooks with a CUDA or MPS SSB session (sliders and Fit
tilt), and exported WebGPU folders (sliders; fitting needs Python). It does not
yet combine with the higher-order aberration panel.

Validation: on a simulated 15 nm BaTiO3 crystal tilted (3, -4) mrad the fit
returns (3.0, -4.1) and ~0 for the untilted control; on a logic-device dataset
the fitted tilt matches the one ptychography learns to 2.4 deg. Speed (logic
crop, 128 x 128 scan): Fit tilt about 5 s on CUDA and 6 s on MPS; full
512 x 512 field about 45 s on CUDA.

## Units

C10, C12 and every higher-order magnitude are nm. quantem.gpu's SSB engines
compute in Angstrom and, before 2026-09-24, reported that Angstrom number under
the nm label (a -10 nm defocus showed as -100 "nm"). Calibrations written since
carry `"aberration_unit": "nm"`; older files are read /10 automatically. Slider
ranges are C10 +-40 nm and C12 0-10 nm, the same physical span as before.

## Bright-Field Count

The `BF` count is the number of bright-field detector pixels used by the SSB
sum. It is a reconstruction quality/speed control, not a real-space crop and
not a display downsample. A `512 x 512` phase image stays `512 x 512` whether
the browser uses 30 percent of the BF disk or the full BF disk.

Use the full BF count for final microscopy claims:

- It averages information from the whole selected BF disk.
- It gives the lowest-noise, most stable phase and loss readout.
- It is the closest browser result to the backend reference reconstruction.
- It costs more first-use GPU time because every selected BF pixel contributes
  a complex `G(k)` image. Those reducers are transient in the default folder
  workflow, not saved as a persistent cache.

Use fewer BF pixels explicitly for exploration:

- Slider drag, playback, and sweep review can stay responsive.
- The large aberration trend is usually visible quickly.
- The phase can be noisier or slightly biased compared with full BF, especially
  for weak features, small BF fractions, or precise signoff.
- The UI status reports the active count as `used/total BF`, so a scientist can
  tell whether they are looking at a drag preview or a full-BF result.

For example, `drag_bf=0.3` is useful for rapidly finding the right defocus or
astigmatism neighborhood on memory-constrained browsers. The default is
`drag_bf=1.0`, so the first view and saved/signoff states use the full selected
BF disk.

## WebGPU Signoff Checklist

Use this checklist before claiming that a WebGPU folder export is ready for
interactive microscopy review:

- [ ] Record the WebGPU adapter. SwiftShader, llvmpipe, or any software adapter
  is not valid performance evidence.
- [ ] Record the native scan size. The browser kernels support square
  `128 x 128`, `256 x 256`, `512 x 512`, and `1024 x 1024` phase grids.
- [ ] Record the payload path: BF-column default or explicit compressed-HDF5 fallback.
  Do not treat a saved `g_bf.c64`/float32 cache as the default sharing path.
- [ ] Record the BF policy. Include both selected BF pixels and active aperture
  BF pixels, for example `542/1805 selected, 379 active`.
- [ ] Record first-use timing: bytes fetched, fetch time, unpack/decode time,
  FFT/reducer setup time, and total time.
- [ ] Drive C10, C12, phi12, scan rotation, BF, phase histogram, phase colormap,
  FFT toggle, FFT histogram, and flip controls in the browser.
- [ ] Report UI/GPU mean, p50, p95, and FPS-equivalent timing for repeated
  interactions. A visible image alone is not a performance signoff.
- [ ] Capture screenshots that show the phase image, FFT when enabled, BF
  status text, and performance readout.

Current implementation coverage:

| Native phase grid | Browser support | Current signoff status |
|---|---|---|
| `128 x 128` | Implemented by the shared WGSL path | Source/unit guard covered; use real headed data before paper claims |
| `256 x 256` | Implemented by the shared WGSL path | Source/unit guard covered; use real headed data before paper claims |
| `512 x 512` | Implemented by the shared WGSL path | Real experimental full-BF browser drive has reached about 24 FPS for C10 changes |
| `1024 x 1024` | Implemented by the shared WGSL path | Real-data BF-column browser drive works, but full active-BF controls remain about 6 FPS on Apple Silicon Metal and fail the 30 FPS target |

## Folder Export

Use a WebGPU folder export when a colleague needs to open the same ptychography
review without the notebook kernel:

```python
w = ShowPtycho(ssb, fft_on=True)  # starts at full selected BF
w.export("logic013_512_review")
```

To start the viewer directly in the authoritative full-BF mode, pass
`drag_bf=1.0`:

```python
w = ShowPtycho(ssb, drag_bf=1.0, fft_on=True)
```

The default folder contains:

```text
logic013_512_review/
├── index.html
├── ShowPtycho.command
├── snapshots/
│   ├── manifest.json
│   ├── cal.json
│   ├── snapshots.json
│   └── README.md
└── source/
    ├── bf_columns.u8
    ├── scan_master.h5
    ├── scan_data_000001.h5
    └── ...
```

The BF-column file is exact detector evidence, not detector binning. The
compressed HDF5 files remain in `source/` as provenance and fallback data, but
the browser opens from BF columns by default.

The `snapshots/` folder is the persistent review state. `snapshots/cal.json`
stores the active calibration, `snapshots/snapshots.json` stores saved
aberration states, and reopening the folder loads those states automatically
after the browser has a folder grant or the command/local server is serving the
folder. Pressing **Save** updates the snapshot JSON in place; it should not
prompt for a separate download in the normal local folder workflow.

Open the folder with the `quantem` CLI:

```bash
quantem showptycho /path/to/logic013_512_review
```

The command validates the folder, prints the compressed HDF5 source summary, starts the required local
HTTP server, reports the BF-column browser source when present, and opens
`index.html` in the browser. It stays alive until Ctrl-C.
Use `--port 8900` only when you need a stable URL, and use `--bind 0.0.0.0`
only when the viewer should be reachable from another device. Double-clicking
`index.html` is supported in Chromium browsers that expose the File System
Access API: click **Open data folder** and grant the export folder. Use the CLI
or `ShowPtycho.command` when you want the no-prompt local-server path.

The HTML file should stay small because it is only the viewer. The microscopy
payload is under `source/`; the export avoids writing `g_bf.c64`, `.f32`
reference images, or detector-binned copies by default.

## Reference

```{eval-rst}
.. autofunction:: quantem.widget.showptycho.ShowPtycho

.. autoclass:: quantem.widget.showptycho.PtychoCalibration
   :members:
```

## Interactive Controls

Each control should repaint the current phase image without requiring a
notebook round trip in WebGPU folder mode.

| Control | User effect | Expected behavior |
|---|---|---|
| C10 slider | Tune defocus | Phase updates; status reports GPU/UI time |
| C12 slider | Tune 2-fold astigmatism magnitude | Phase updates; loss shown for full-BF results |
| phi12 slider | Tune astigmatism angle | Phase updates; FFT follows when visible |
| Rotation slider | Tune scan-detector rotation | Reconstruction updates in the live backend path |
| FFT toggle | Show/hide reciprocal-space phase FFT | FFT computes only when visible |
| BF slider | Choose how much of the BF disk contributes to each interactive reconstruction | Starts at full BF; drag left for a smaller exploratory subset |
| Phase colormap / histogram | Change display mapping | Current image repaints; reconstruction data is unchanged |
| FFT colormap / histogram | Change FFT display mapping | FFT panel repaints; phase reconstruction is unchanged |
| Flip phase | Invert phase sign | Current phase and FFT update without recomputing BF data |
| Save/star | Mark a useful aberration state | Save only after recomputing with the intended BF count |

```{seealso}
The shared folder/single-file export language is documented in
[HTML export](html-export).
```
