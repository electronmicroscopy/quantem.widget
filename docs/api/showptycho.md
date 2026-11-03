# ShowPtycho

Interactive ptychography aberration review for a prepared `quantem.gpu.SSB`
session. `ShowPtycho` lets a microscopist tune defocus, astigmatism,
scan-detector rotation, phase flip, display contrast, and the FFT view while
watching the reconstructed phase update.

Given an array instead, `ShowPtycho` is a reconstruction viewer that runs on
any computer, with no quantem.gpu:

```python
ShowPtycho(phase)                                  # real array: the phase (rad)
ShowPtycho(object_wave, sampling=0.2, units="Å")   # complex array: phase and amplitude
ShowPtycho(torch.from_numpy(object_wave).to("cuda"))   # torch tensors on any device
```

| Input | Shown | Hidden |
|---|---|---|
| 2D real array (NumPy or torch) | phase, FFT, histograms, scale bar (`sampling` in `units`: Å, nm or pm) | aberration sliders, Sampling, Save, pins, sweeps (one line names the SSB session they need) |
| 2D complex array | the above plus amplitude (Amp) and the complex HSV view | as above |
| `quantem.gpu.SSB` session | phase, FFT, histograms, scale bar and everything below | Amp and Complex: SSB recovers the phase only, so there is no amplitude to show |

The browser draws float32. A float32 or complex64 array is shown exactly; a
float64 or complex128 array is rounded once and a printed line gives the
largest change. `angle` and `abs` run on the tensor's own device.

## Workflow and ownership

`quantem.widget` owns presentation, project layout, and browser export. It does
not implement CUDA, MPS, or WebGPU SSB kernels. Exact fitting and final
reconstruction enter through the shared `quantem.gpu.SSB` API and return one
`SSBResult`, regardless of compute backend. `ShowPtycho` consumes the prepared
`SSB` session so it can reuse resident buffers for notebook interaction; the
returned result remains the compact analysis output.

### Load data, then open the widget

Keep the notebook to two steps. First load and inspect the acquisition:

```python
from quantem.gpu import SSB, io
from quantem.widget import ShowPtycho

data = io.load("scan.qem")
data
```

Then reconstruct and open ShowPtycho. Voltage, convergence and scan step come
from metadata; set missing ones on `data.calibration` first
(`data.calibration.semiangle_mrad = 30`). The scan-detector rotation is the
recorded one, else estimated from the centre-of-mass curl:

```python
ssb = SSB(data)
ssb.reconstruct()
viewer = ssb.show(local_tilt=True, show_fft=True, save_dir="ssb-results")
viewer
```

`ssb.show(...)` is `ShowPtycho(ssb, ...)`; both forms take the same options.

Detector sampling reuses the saved calibration. When missing, SSB fits the
full-field probe disk automatically and retains the estimate for **Save**.
The known convergence semiangle is required to establish the angular scale.
An omitted rotation is the recorded one, else estimated from the curl of the
centre-of-mass field, known modulo 180 degrees; `reconstruct()` settles the
180-degree choice from the sign of the atom columns. The launch table names the
source of each value.

One compact launch table appears automatically: **Value used**, **Unit**,
and **Source**. A **File value** column appears only when a recorded value
or unit differs; unchanged rows stay blank in that column. Equal row and column
sampling is shown once, while unequal sampling keeps the `(row, col)` pair.
Sources distinguish **Recorded**, **Overridden**, **Explicit override**,
**Estimated**, and **Default**. Explicit keywords remain overrides even when
they equal stored values. These labels show provenance, not confidence.
The launch table is a snapshot; `ssb.calibration_table()` provides the complete
file/current comparison later. Neither table changes the source QEM.

Perform full-field joint fitting and region selection inside the widget.
Turn on **Local Tilt**, click **Draw**, and drag in any direction. Select a
rectangle or its result to edit it; **Delete** removes that region and its view
without refitting the others. Each result also has a top-right delete button
that appears on hover or keyboard focus (always visible on touch screens).
The Delete key works while the overview region editor has focus.
**Max Cols** limits the image columns, including Overview; narrower
notebook windows use fewer columns. This display setting does not resample the
reconstruction and is retained by Save.

**Save** (or `viewer.save_calibration()`, which returns the path) writes
`calibration.json` in `save_dir`. Restart the kernel and run
the same cells: ShowPtycho restores the saved optics, tilt, depth and display
settings before reconstructing, also over a session that `ssb.reconstruct()`
has just fitted, and prints one line naming the restored values and the
session values they replaced. The toolbar says **Restored saved full-field
fit**.

The acquisition identity, scan region and launch calibration must match; a
launch rotation 180 degrees from the saved one still matches, since the fit's
rotation check chooses that branch. A changed file, detector calibration or
crop leaves the supplied session intact and explains why the saved fit was
not loaded. Older saves without an acquisition identity need to be saved once
from their fitted session. Use `restore=False` to keep the supplied session.
In-memory arrays are not matched by filename.
With `local_tilt=True`, saved rectangles and their available joint-fit views
are restored too. Each view is reconstructed from the full scan using its own
saved correction; no regional optimization runs on restart. A region without
a saved fit still needs **Fit Regions**.

`viewer.export(...)` writes the browser project and exact bright-field evidence;
the notebook's Save button writes numerical calibration only.

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
payload reuses an unchanged whole-scan QEM acquisition
(`source/acquisition.qem`) on CUDA and MPS, without re-encoding. The browser
checks the acquisition's header identity and file checksums, decodes the integer counts on its GPU, gathers
every selected BF pixel and builds the native scan spectra once. A scan crop,
changed counts or another source format instead exports an exact BF companion:
`source/bf_columns.qem` on CUDA or `bf_columns.u8` / `.u16` on MPS. Moving
C10, C12, phi12, or scan rotation reuses those resident spectra. It does not
fetch the counts or repeat the scan FFT. The export does not persist expanded float32 images or
complex64 BF reducers by default.

FFT is a display analysis of the current reconstructed phase. When the FFT panel
is visible, the widget computes the FFT from the latest phase image and redraws
that FFT panel. It is not re-running the raw detector preprocessing.

## SSB fit methods

quantem fits SSB aberrations with a physical model of the measurement, not an image-quality score. For each trial
set of aberrations, and the sample's tilt and thickness, it predicts the double-overlap data the detector should have
recorded, then keeps the parameters whose prediction agrees best with the measurement (least squares). This
model-based, tilt-corrected fit is the default everywhere, and it is where SSB in quantem is heading.

| Call | Method | Use it for |
|---|---|---|
| `find_aberrations()` | model-based, tilt-corrected | the default: every fit |
| `find_aberrations(tilt=False)` | model-based, tilt held at 0 | thin, untilted samples, or to see what tilt correction changes |
| `find_aberrations(tilt=False, objective="phase_variance")` | phase variance, the earlier SSB search | a backup: reproducing older results, cross-checking a fit |

Phase variance chooses the aberrations that make the corrected phases of all bright-field pixels agree best: the
least variance of the corrected phase across them. It stays as a backup, not an equal alternative: it cannot see
sample tilt or thickness, it scores only how consistent the bright-field images are rather than how well a model
explains the measurement, and it is about 10x slower. On simulated crystals with known aberrations, the model-based
fit lands about 3x closer in C10.

## Tilt correction

The fit is tilt-corrected by default: `ssb.find_aberrations()` fits the SSB
model with the sample tilt and depth spread, and the viewer opens on that fit:

```python
ssb.find_aberrations(check_rotation=False)
viewer = ShowPtycho(ssb, show_fft=True)
viewer
```

The other two methods are explicit, with quantem.gpu's names:
`ssb.find_aberrations(tilt=False)` fits the same model at tilt 0, and
`ssb.find_aberrations(tilt=False, objective="phase_variance")` is the
phase-variance backup. Joint fit, Refit SSB and Fit Regions all run the
model-based, tilt-corrected fit.

**Tilt correction** starts on. Switch it off to compare the phase and FFT with
zero sample tilt, keeping the same fitted aberrations and depth spread. This
is a preview setting (`viewer.tilt_correction = False`), not an edit to the
fitted tilt. **Save** retains the fitted tilt in the calibration and session
for subsequent reconstruction. The switch is disabled when no sample
correction has been fitted. This does not rotate the scan or align depth slices.

## Reusing the reviewed phase

For a numerical handoff to another reconstruction, choose the same estimator
at both stages:

```python
viewer = ShowPtycho(ssb, phase_estimator="phase_of_mean")
```

Save records this estimator along with the resolved sampling factor and optics.
Omitting the option preserves QuantEM.GPU's phase-of-the-mean default on
CUDA and MPS, including higher-order aberrations. These estimators
can produce different numerical phase values. Explicit-estimator reviews stay
in the notebook; live browser-folder export rejects them until that handoff
preserves the estimator too.

## Output sampling

```python
from quantem.widget import ShowPtycho

viewer = ShowPtycho(ssb, show_fft=True)
viewer
```

Output sampling is automatic. ShowPtycho uses QuantEM.GPU's
`ssb_upsampling_factor` rule: the smallest supported grid covering the nominal
circular-aperture SSB frequency range, `2 * semiangle / wavelength`, along both
scan axes. This depends on calibrated voltage, convergence angle and scan step.
Detector truncation and noise can reduce usable information; automatic sampling
is not a measured resolution claim. The field of view, measured scan data and
fitted coefficients stay unchanged. The scale bar and crop coordinates follow
the output sampling; fitting and its loss remain on the native grid.

Defocus, astigmatism, higher-order aberrations and optional tilt correction use
the same automatic sampling and coherent phase estimator on CUDA and MPS. There
is no sampling or averaging choice in the toolbar. Save records the resolved
factor, requested mode and phase estimator for downstream use.

Explicit factors remain available through the Python `upsampling_factor`
attribute for controlled comparisons. CUDA and MPS support 1, 2, 3, 4 and 8;
WebGPU uses 1, 2, 4 and 8, rounding Auto up when 3× is needed. Every output uses
all bright-field pixels. A finer grid increases reconstruction work; it does not
change the native-grid fitting objective.

The export budgets at least 4.5 GiB for resident scan spectra, increasing that
budget to hold every BF pixel's complex64 Hermitian half-plane when needed
(`gpu_memory_gb` in `snapshots/cal.json`). Encoded counts, decoder tables and
output buffers require additional memory. A GPU with insufficient capacity
rejects the run; detector pixels are never removed to make it fit.

## Real-Space Crop And SSB Refit

For a region-specific probe/aberration fit, open the prepared SSB session with
the raw master path. ShowPtycho then exposes `Crop` in the top-right action group
beside `Export` and `Reset`. Enable `Crop`, drag a rectangle on the phase panel,
inspect the live row/column region readout, then use `Refit SSB`.
The widget loads the original HDF5 source into encoded GPU storage, decodes
only that scan region, runs the tilt-corrected fit (200 trials followed by
refinement, `SSB(...).find_aberrations(...)`), and replaces the phase/FFT and their
calibration with the result.

```python
from quantem.gpu import SSB, io
from quantem.widget import ShowPtycho

data = io.load("acquisition_master.h5")
ssb = SSB(
    data,
    semiangle_mrad=20.0,
    scan_sampling_A=(0.276, 0.276),
    voltage_kV=300.0,
)
result = ssb.find_aberrations()     # tilt-corrected, 200 trials
w = ShowPtycho(ssb, show_fft=True)   # the session names its master (ssb.source_path)
```

The selection may be rectangular; each dimension must span at least `32` scan
positions. `Crop Reset` clears the selection without changing the current
reconstruction. This is a reconstruction operation, not a display crop.

The refit control is absent from standalone HTML/WebGPU exports and from
sessions without a readable source master. Those modes can inspect an existing
result interactively, but only a source-backed live session has the raw
detector data and SSB optimizer needed to make a new scientific fit.

## Sample Tilt (Thick Crystals)

Standard SSB treats the sample as one thin plane. In a thick, tilted crystal
each depth sees a different defocus and sits shifted sideways by depth x tilt,
so the lattice washes out along the tilt: often only one set of fringes survives.
quantem.gpu's default fit adds the sample to the model, and the
**Sample tilt (thick SSB)** panel, above the aberration sliders, shows it:

- **tilt row / tilt col** (mrad, scan frame) and **depth spread** (nm). Depth
  spread 0 is standard SSB, exactly. It is the depth the model averages over,
  not a measured sample thickness.
- **Joint fit** fits defocus, astigmatism, tilt and depth spread together and
  moves every slider to the result. The fitted C10 is the defocus at mid-depth.
- The status line gives the tilt in the scan frame; Save also records it along
  the detector axes (`propagator_tilt_mrad`, the scan frame turned by the
  scan-detector rotation and swapped when the rotation is flipped), which
  seeds a quantem.thick reconstruction directly.

Open with the fit already run: the widget shows the session's latest fit.

```python
from quantem.gpu import SSB
from quantem.widget import ShowPtycho

ssb = SSB.open(path, voltage_kV=300.0, semiangle_mrad=30.0, scan_sampling_A=(0.495, 0.495),
               detector_sampling_mrad=(0.5554, 0.5554), scan_detector_rotation_deg=-8.6)
standard = ssb.find_aberrations(tilt=False)     # defocus + astigmatism at tilt 0
ShowPtycho(ssb)                                 # the tilt-0 control
tilted = ssb.find_aberrations()                 # the default: + sample tilt and depth spread, jointly
ShowPtycho(ssb)                                 # opens on the fitted tilt
pd.concat([standard.report(), tilted.report()])
```

**Save** writes the tilt into `calibration.json` (`sample_tilt_mrad` in the
scan frame, `propagator_tilt_mrad` along the detector axes, the ptychography
frame, from `quantem.gpu.geometry.sample_to_propagator_tilt` with the session's
rotation and flip, and
`depth_spread_nm`, the model's depth spread, which is not a sample thickness).

Where it runs: live notebooks with a CUDA or MPS SSB session (sliders and Joint
fit), and exported WebGPU folders (sliders; fitting needs Python). It does not
yet combine with the higher-order aberration panel.

Validation: on a simulated 15 nm BaTiO3 crystal tilted (3, -4) mrad the fit
returns (3.0, -4.1) and ~0 for the untilted control. Speed (a 128 x 128 scan
crop): Joint fit about 5 s on CUDA and 6 s on MPS; full 512 x 512 field about
45 s on CUDA.

## Units

Every key carries its unit. Aberrations are `C10_nm`, `C12_nm` and `phi12_deg`,
with higher orders as `C21_nm` and `phi21_deg` (and so on) in the same
`aberrations` object; the rotation is `scan_detector_rotation_deg`, the physical
angle in [0, 360), beside `scan_detector_rotation_flipped`; `scan_sampling_A` and
`detector_sampling_mrad` are `[row, col]`. `calibration.json`,
`showptycho_stars.json`, the exported `snapshots/cal.json` and the widget's
messages use these names, and each file names its layout in a `schema` string
(`quantem.ssb-calibration/6`, `quantem.ssb-stars/3`,
`quantem.ssb-browser-calibration/3`, `quantem.ssb-browser-folder/3`,
`quantem.ssb-snapshots/2`). No file holds an absolute path: `calibration.json`
and each star name the acquisition as `source`, its file name and, when the
data lives under the save folder, its path relative to it; the session
identity a restore compares is quantem.gpu's portable `source_identity` (file
names, sizes, SHA-256 of each file's first MiB, scan and detector shapes and
the flagged-pixel fill). A folder moved or copied whole, data included,
restores its fit; another acquisition under the same file name is refused.
The display-only phase sign is
`display_negate_phase`; regional fits hold `sample_tilt_mrad`,
`sample_agreement` and `num_completed_trials`. A file of another layout is not
converted; `scripts/rewrite_ssb_calibration.py` in the repository carries
earlier calibration, star and snapshot files over once (a `/5` calibration
with a session identity needs its data beside it or `--data PATH`). A
Show4DSTEM Compute SSB download (`quantem.show4dstem-ssb-calibration/3`) is
another record and is not restored as `calibration.json`.
quantem.gpu's SSB engines compute in Angstrom and radians; the widget converts
once, in `quantem.widget.adapters.gpu`. Slider ranges are C10 +-40 nm and C12
0-10 nm.

## Bright-Field Pixels

ShowPtycho uses all available bright-field detector pixels for slider updates,
sweeps and saved reconstructions. The count follows the calibrated aperture;
there is no BF-count control. Older saved widgets with a reduced drag count
return to the full set when opened. This does not change the scan extent or
output sampling.

## WebGPU Signoff Checklist

Use this checklist before claiming that a WebGPU folder export is ready for
interactive microscopy review:

- [ ] Record the WebGPU adapter. SwiftShader, llvmpipe, or any software adapter
  is not valid performance evidence.
- [ ] Record the native scan size. The browser kernels support square
  `128 x 128`, `256 x 256`, `512 x 512`, and `1024 x 1024` phase grids.
- [ ] Record the payload: the reused QEM acquisition or exact BF-column file under `source/`.
  Do not treat a saved `g_bf.c64`/float32 cache as the default sharing path.
- [ ] Record the full logical BF count and the active aperture BF count.
- [ ] Record first-use timing: bytes fetched, fetch time, unpack/decode time,
  FFT/reducer setup time, and total time.
- [ ] Drive C10, C12, phi12, scan rotation, phase histogram, phase colormap,
  FFT toggle, FFT histogram, and flip controls in the browser.
- [ ] Report UI/GPU mean, p50, p95, and FPS-equivalent timing for repeated
  interactions. A visible image alone is not a performance signoff.
- [ ] Capture screenshots that show the phase image, FFT when enabled,
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
w = ShowPtycho(ssb, show_fft=True)  # starts at full selected BF
w.export("scan_review")
```

The folder contains:

```text
scan_review/
├── index.html
├── ShowPtycho.command
├── snapshots/
│   ├── manifest.json
│   ├── cal.json
│   ├── snapshots.json
│   └── README.md
└── source/
    └── acquisition.qem
```

The existing QEM acquisition is reused when it still matches the whole-scan
session. Detector crop offsets and transposition are recorded for the browser's
BF gather. Other sessions export exact `bf_columns.qem` counts on CUDA or a
`bf_columns.u8` / `.u16` companion on MPS. The raw HDF5 acquisition is not copied.
Sources are hardlinked when possible, otherwise symlinked or copied. The
manifest records which: a symlinked review still needs its external source
when moved or shared.

The `snapshots/` folder is the persistent review state. `snapshots/cal.json`
stores the active calibration, `snapshots/snapshots.json` stores saved
aberration states, and reopening the folder loads those states automatically
after the browser has a folder grant or the command/local server is serving the
folder. Pressing **Save** updates the snapshot JSON in place; it should not
prompt for a separate download in the normal local folder workflow.

Open the folder by double-clicking `ShowPtycho.command`, which starts a
range-capable local HTTP server and opens `index.html`. Double-clicking
`index.html` directly is supported in Chromium browsers that expose the File
System Access API: click **Open data folder** and grant the export folder. Any
Range-capable static server (for example `python -m http.server` run inside the
folder) also works, including from another device on the network.

The HTML file should stay small because it is only the viewer. The microscopy
payload is under `source/`; the export avoids writing `g_bf.c64`, `.f32`
reference images, or detector-binned copies by default.

## Reference

```{eval-rst}
.. autoclass:: quantem.widget.showptycho.ShowPtycho
   :members: export
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
| Phase colormap / histogram | Change display mapping | Current image repaints; reconstruction data is unchanged |
| FFT colormap / histogram | Change FFT display mapping | FFT panel repaints; phase reconstruction is unchanged |
| Negate phase (display) | Negate the displayed phase sign (`display_negate_phase`) | Current phase and FFT update without recomputing BF data; no fitted number changes |
| Save/star | Mark a useful aberration state | Full bright-field reconstruction is retained |

```{seealso}
The shared folder/single-file export language is documented in
[HTML export](html-export).
```

## Local tilt workspace

Keep a live SSB session as the input to retain the original C10, C12, phi12,
rotation, Sample, FFT and snapshot controls. The optional **Local Tilt** switch
adds region editing to the same image. It starts off; there are no workspace
tabs. An array input remains a static image and cannot expose live optics.

```python
viewer = ShowPtycho(ssb, local_tilt=True, show_fft=True)
viewer
```

Draw, move or resize rectangular regions on the overview. **Auto** refits after
release, and **Fit Region** or **Fit Regions** reruns fitting explicitly. Each
region retains independent controls, fit parameters and results; selecting
**Overview** restores the full-field controls. Moving one region leaves the
others intact.

The built-in workflow keeps detector geometry and scan-detector rotation common.
For each changed region it jointly fits C10, C12, astigmatism angle, both sample
tilt components and depth spread, with five repeats of 200 trials plus Newton
refinement on the selected GPU backend (`viewer.fit_regions(num_trials=...,
refinement=...)` changes both, also for the button). Progress shows the active repeat and
elapsed time. It reconstructs the full scan with each regional fit, then crops
the corrected regional view. The overview fit remains intact.

The settings accept up to 64 overlapping windows, each at least 32 native scan
samples per axis. A freehand phase-scoring mask is not an arbitrary-support SSB
fit; region selection currently uses rectangles. Contrast alone does not
identify physical grain boundaries. Changing the shared calibration marks local
results stale; obsolete in-flight replies cannot replace a newer selection.

Named SSB results appear beside the overview. The existing Color, Smooth and
phase-percentile controls apply to all SSB panels; percentiles are evaluated
per image. The top-right **View settings** gear groups **Show FFT**,
**Inset / Side by Side**, FFT color and contrast. Expand **Local Tilt** there
for exact bounds, overlapping windows and linked or independent regional zoom;
**Sweep Ranges** and **SSB Crop** hold the other occasional controls. Local
Tilt opens FFT as an inset; Side by Side uses an equally sized image panel.
The defocus sweep, Pin
and Save controls remain available; Randomize has been removed. The display transport is
float32; Python preserves the caller-supplied image arrays and dtype. A live SSB session supplies fitting automatically; no notebook callback is needed.
Custom callbacks remain available for application integrations.

### Continue in another notebook cell

```python
regions = viewer.regions
parameters = viewer.ssb_parameters
fits = viewer.region_fits
images = viewer.region_images
```

These are independent snapshots. Rerun the cell after interacting with the
widget. Fits and images raise a corrective error if their selection or shared
optics no longer match. To use the same workflow without pointer interaction:

```python
viewer.regions = regions
fits = viewer.fit_regions()
images = viewer.region_images
```

## Reuse the reviewed calibration

`viewer.calibration(region="Upper")` returns an independent dictionary with
the keys and units of the `calibration.json` Save writes (its `id` and
`generated_at_utc` stay None until saved): C10/C12 in nm, phi12 in degrees,
sample tilt `(row, col)` in mrad in the scan frame, depth spread in nm, and
half-open native scan bounds.
A stable region ID is also accepted. `viewer.calibration()` explicitly selects
the overview, irrespective of the visible controls. Busy or stale regional fits
raise with corrective guidance. Numerical state is independent of display settings.

Add `include_phase=True` to include the full `ssb_phase` array in radians. This
reuses the viewer's prepared GPU session at the selected model; it does not
prepare another acquisition or refit. The array is an independent copy.

See the [H5/QEM to myptycho notebook](../tutorials/showptycho_to_myptycho.ipynb)
for region selection, joint fitting, independent iterative runs and saved results.
