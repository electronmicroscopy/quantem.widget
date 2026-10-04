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
result = ssb.find_aberrations(trials=200, refinement="nelder-mead")

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
  ssb = SSB(
      data,
      semiangle_mrad=30.0,
      scan_sampling_A=0.264,
      voltage_kV=300.0,
      rotation_angle_deg=158.9,
  )
  result = ssb.find_aberrations(trials=200, refinement="nelder-mead")
  ShowPtycho(ssb)
  ```

  256x256 is a good crop size: small enough for region-specific aberrations, big
  enough that the phase is not blocky. 128x128 works but displays coarse.

## Is your crystal tilted?

**The question.** SSB treats the sample as one thin sheet. A real crystal is a
few nanometres thick and rarely sits exactly on the zone axis. If it leans,
does SSB still see a sharp lattice, and can it tell you how much it leans?

**Predict first.** A column tilted by 5 mrad through 10 nm of crystal: how far
does its bottom sit from its top, compared with a 4 A lattice spacing? Does
every lattice direction blur the same way?

```python
for tilt_mrad in (1, 3, 5, 10):
    walk_A = 100.0 * tilt_mrad * 1e-3          # 10 nm = 100 A of depth
    print(f"{tilt_mrad:2d} mrad -> {walk_A:.1f} A walk, {walk_A / 4.0:.0%} of a 4 A spacing")
```

At 5 mrad the column walks half an angstrom, and only *along* the tilt: the
lattice blurs in one direction and stays sharp in the other. Standard SSB has
no depth, so it cannot express this.

**The experiment.** Simulate a crystal whose tilt you know: BaTiO3 [001],
15 nm thick, leaning by (3, -4) mrad, then reconstruct it both ways. The
simulation uses [abTEM](https://abtem.readthedocs.io) on the GPU and takes
about a minute; no data file is needed.

```python
import abtem
import numpy as np
from ase import Atoms

def tilted_crystal(tilt_mrad=(3.0, -4.0), thickness_A=152.0):
    """abTEM 4D-STEM of BaTiO3 [001] with every atom at depth z shifted by z x tilt (row, col)."""
    abtem.config.set({"device": "gpu"})
    a, cells = 4.0, 9
    layers, box = int(round(thickness_A / a)), cells * a
    basis = [("Ba", (0, 0, 0)), ("Ti", (0.5, 0.5, 0.5)), ("O", (0.5, 0.5, 0)), ("O", (0.5, 0, 0.5)), ("O", (0, 0.5, 0.5))]
    symbols, positions = [], []
    for i in range(cells):
        for j in range(cells):
            for k in range(layers):
                for symbol, (fr, fc, fz) in basis:
                    z = (k + fz) * a
                    symbols.append(symbol)
                    positions.append((((i + fr) * a + z * tilt_mrad[0] * 1e-3) % box,
                                      ((j + fc) * a + z * tilt_mrad[1] * 1e-3) % box, z + 0.5))
    atoms = Atoms(symbols, positions=positions, cell=[box, box, layers * a + 1.0], pbc=True)
    potential = abtem.Potential(atoms, sampling=0.08, slice_thickness=a / 2, projection="infinite", parametrization="lobato")
    probe = abtem.Probe(energy=300e3, semiangle_cutoff=30, defocus=layers * a / 2)   # focused at mid-depth
    scan = abtem.GridScan(start=(2 * a, 2 * a), end=(6 * a, 6 * a), gpts=(64, 64), endpoint=False)
    measurement = probe.scan(potential, scan=scan, detectors=abtem.PixelatedDetector(max_angle=45)).compute()
    return np.asarray(measurement.array, dtype=np.float32), float(measurement.angular_sampling[0])

data, det_mrad = tilted_crystal()
ssb = SSB(data, backend="auto", voltage_kV=300.0, semiangle_mrad=30.0,
                     scan_sampling_A=0.25, det_sampling=det_mrad, rotation_angle_deg=0.0)
```

First the thin-sheet model. Look at the FFT: are the lattice spots equally
sharp in every direction?

```python
standard = ssb.find_aberrations(verbose=False)
ShowPtycho(ssb, fft_on=True)
```

Now let the sample lean. `fit(tilt=True)` fits the same aberrations together
with a tilt and a depth spread. Compare: which spots sharpened, and did the
defocus move?

```python
tilted = ssb.find_aberrations(tilt=True, verbose=False)
ShowPtycho(ssb, fft_on=True)          # opens on the fitted tilt; drag the Sample tilt sliders
```

**What the fit found.**

```python
import pandas as pd
pd.concat([standard.report(), tilted.report()])
```

The tilt comes back as about (3.0, -4.1) mrad, the value built into the
simulation, and `tilted.tilt_fit_gain` is above 1.2: the leaning model
explains the data better than the thin sheet. An untilted control crystal
gives (-0.3, -0.1) mrad. Reading the table:

- `C10` is now the defocus at the middle of the crystal, so it can differ from
  the standard fit's value. This is why tilt and defocus are fitted together:
  fitting the tilt after the standard fit leaves C10 behind and stalls.
- `depth spread (nm)` is how deep the model's column walk extends, not a
  measured thickness (the 15.2 nm crystal comes back as 10-13 nm).
- The tilt is in the scan frame. The Sample tilt panel also shows it in the
  ptychography object frame, ready to seed a multislice reconstruction.

**The model in one line.** A slice at depth `z` sees defocus `C10 + z` and is
shifted by `z * tilt`. Averaging over depth multiplies each SSB overlap term
by a real `sinc` weight, which falls fastest for spatial frequencies along the
tilt: the one-directional blur you predicted. With zero depth every weight is 1
and the model is standard SSB.

**When not to trust it.**

- The loss in the stats bar does not reward tilt. Judge by `tilt_fit_gain` and
  the FFT, not by the loss.
- A tilt at the search limit (25 mrad) or a gain close to 1 is not a
  measurement.
- One tilt direction can be loosely determined on real films (about 0.6 mrad).
- The tilt slider is interactive on crops; at a full 512 x 512 scan each
  update takes a few hundred ms on CUDA.

See the [SSB API](https://bobleesj.github.io/quantem.gpu/api/ssb.html) for
`fit(tilt=True)`, `preview(tilt_mrad=..., depth_spread_nm=...)` and the
evidence behind the defaults.

## Checklist

1. Leave `SSB.open(..., dtype=None)` at its default for native detector precision.
2. Native detector, `det_bin=1` — do not bin.
3. `ssb.find_aberrations(trials=200, refinement="nelder-mead")` — the fit is not optional.
4. Pass the `ssb` object to `ShowPtycho`, not `data` + hand-typed aberrations.
5. Confirm: stats bar `loss` is non-null and the trials panel is populated.
6. Thick or possibly mistilted crystal: also run `ssb.find_aberrations(tilt=True)` and compare `report()` rows.
