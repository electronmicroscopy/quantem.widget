# ShowPtycho in Jupyter

`ShowPtycho` is an interactive **SSB** (single-sideband) aberration explorer for
4D-STEM data. You tune defocus (C10), astigmatism (C12 / phi12), and scan-detector
rotation and watch the reconstructed phase and its FFT update live in the notebook.

SSB is a *direct* (non-iterative) phase retrieval: fast and interactive, but
lower quality than iterative multislice ptychography. Use ShowPtycho for quick
aberration tuning and review, not as a substitute for a full iterative
reconstruction.

To export a standalone HTML viewer you can open without a kernel — the folder
ships a double-click `ShowPtycho.command` launcher, or open `index.html` in
Chrome and grant it the data folder — see
[Export and run ShowPtycho](showptycho_export.md).

## The one rule: always fit before you view

```python
from quantem.gpu import SSB
from quantem.widget import ShowPtycho

# 1. Open the native source with your microscope calibration.
ssb = SSB.open(
    "scan_master.h5",
    backend="auto",
    semiangle_mrad=30.0,        # convergence semiangle, mrad
    scan_sampling_A=0.264,      # real-space scan step, Angstrom
    voltage_kV=300.0,
    rotation_angle_deg=158.9,   # scan-detector rotation (run find_rotation if unknown)
)

# 2. Fit and refine the aberrations. THIS STEP IS REQUIRED.
result = ssb.fit(trials=200, refinement="nelder-mead")

# 3. Open the interactive widget — it reuses the prepared GPU session.
ShowPtycho(ssb)
```

### Do NOT skip step 2

```python
# WRONG — this NEVER fits. It uses whatever aberrations you pass verbatim,
# so the phase and FFT are junk unless your numbers were already perfect.
ShowPtycho(data, semiangle_mrad=30.0, scan_sampling_A=0.264,
           voltage_kV=300.0,
           aberrations={"C10": 78.0, "C12": 17.0, "phi12": 0.5})
```

`ShowPtycho(data, aberrations=...)` is a convenience constructor that trusts the
aberrations you hand it. It does not fit them. If you want the solver to find
the aberrations, build an `SSB`, call `fit(trials=200,
refinement="nelder-mead")`, and pass that same prepared `ssb` object to
`ShowPtycho(ssb)`. The returned `SSBResult` is also available as `result` for
non-interactive analysis through `result.phase`, `result.amplitude`, and
`result.object_wave`.

You can confirm the solve ran: the stats bar shows a non-null `loss`, and the
`Optuna trials + Nelder-Mead` panel at the bottom is populated.

## No detector binning

Build the reconstruction at the **native detector size** (`det_bin=1`, the
default). Native (e.g. 192x192) is what resolves light columns such as oxygen in
a perovskite; binning throws that away. Binning also breaks the HTML export (the
browser cannot bin), so keep the whole workflow un-binned.

## Region-specific refit (crop)

A smaller crop often converges more physically than the full field of view: a
single global aberration and rotation hold better over a small region, so a crop
can resolve oxygen the full FOV cannot.

Two ways to crop:

- **Interactively.** Construct the widget with the raw master path so the `Crop`
  action appears next to `Export`/`Reset`. Enable `Crop`, drag a rectangle on the
  phase, then `Refit SSB` — the widget reloads only that scan region from the
  HDF5 source, runs 200 optimization trials plus refinement, and replaces the
  phase/FFT and calibration.

- **In code.** Load only the region, then fit as usual:

  ```python
  from quantem.gpu.io import load

  data = load("scan_master.h5", dtype=None,
              scan_region=(128, 384, 128, 384)).data   # 256x256 center crop
  ssb = SSB.from_array(
      data,
      semiangle_mrad=30.0,
      scan_sampling_A=0.264,
      voltage_kV=300.0,
      rotation_angle_deg=158.9,
  )
  result = ssb.fit(trials=200, refinement="nelder-mead")
  ShowPtycho(ssb)
  ```

  256x256 is a good crop size: small enough for region-specific aberrations, big
  enough that the phase is not blocky. 128x128 works but displays coarse.

## Checklist

1. Leave `SSB.open(..., dtype=None)` at its default for native detector precision.
2. Native detector, `det_bin=1` — do not bin.
3. `ssb.fit(trials=200, refinement="nelder-mead")` — the fit is not optional.
4. Pass the `ssb` object to `ShowPtycho`, not `data` + hand-typed aberrations.
5. Confirm: stats bar `loss` is non-null and the trials panel is populated.
