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

## Load the gold scan

The example is the public gold nanoparticle acquisition: a 512 x 512 scan with
the native 192 x 192 detector. The first run downloads about 4.6 GB from the
[quantem-data](https://huggingface.co/datasets/bobleesj/quantem-data) dataset
at a pinned revision; later runs reuse the local cache. `io.load` keeps every
count and every detector pixel, encoded on the GPU.

```python
from pathlib import Path

from huggingface_hub import snapshot_download
from quantem.gpu import SSB, io
import quantem.widget as qw

qw.profile()
folder = snapshot_download(
    "bobleesj/quantem-data", repo_type="dataset",
    revision="5925e4b49b2ca28ff82d6d5a6b9a8bfea9edfb20",
    allow_patterns="4dstem/gold_512/*",
)
data = io.load(str(Path(folder) / "4dstem/gold_512/gold_004_master.h5"))
data
```

For your own acquisition, pass its `.qem` file or H5 master instead.

## State the beam

SSB needs the beam energy, the convergence semi-angle and the scan step. The
file records the energy; set the others on `data.calibration`. The scan step
cannot be recovered from the patterns. A single number applies to both
(row, col) axes.

```python
data.calibration.semiangle_mrad = 30
data.calibration.scan_sampling_A = 0.5
```

## Calibrate the detector

The bright-field disk is the image of the probe aperture, so its edge sits at
the convergence semi-angle: the semi-angle divided by the disk's edge radius in
pixels is the detector step in mrad per pixel. `fit_detector_sampling()` fits
the disk on the mean pattern of the encoded scan and records the step, the disk
centre and its edge radius on `data.calibration`; SSB and Show4DSTEM read them.

```python
fit = data.calibration.fit_detector_sampling()
fit.show()
```

```python
data.calibration
```

On the gold scan the disk edge is 49.7 px from its centre, so the detector step
is about 0.6035 mrad per pixel. The table shows each value's source: recorded in
the file, overridden by you, or fitted from the bright-field disk. A calibrated
camera length of your own wins: set `data.calibration.detector_sampling_mrad`
before or after the fit.

## SSB in quantem: model-based by default

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

## Reconstruct

```python
ssb = SSB(data)
fit = ssb.reconstruct()
```

`SSB(data)` estimates the angle between the scan and the detector. The
centre-of-mass shift of each pattern is a gradient field, and the rotation that
removes its curl is the scan-detector rotation: about 178 degrees here, not
flipped. The curl fixes the angle only modulo 180 degrees: at 178 + 180 = 358
degrees every centre-of-mass vector is reversed and the curl is the same. The
two branches give conjugate phases, so one shows the gold particles bright and
the other dark.

`reconstruct()` fits the aberrations on both branches, independently, and keeps
the one whose atomic columns carry positive phase. On this scan the columns do
not decide (z +0.0), so the phase tail does: SSB keeps 358 degrees, where the
phase has the longer positive tail (skewness +0.22 against -0.22), and one line
says so. The fit gives C10 near -18.4 nm, C12 2.6 nm and phi12 -37.6 degrees.

Look at both branches yourself. The fit stays on the native scan grid; each
branch's own fit is shown reconstructed at `upsampling_factor="auto"`, the
smallest factor whose output grid holds the whole SSB bandwidth (4× for the
0.5 Å gold step):

```python
branches = fit.rotation_branches
panels = branches.upsampled_phases()  # each branch's fit at upsampling_factor="auto"
factor = branches.upsampling_factor
qw.Show2D(
    list(panels),
    labels=[f"{branch.scan_detector_rotation_deg:.1f}°, {factor}x auto" + (" kept" if index == branches.kept else "")
            for index, branch in enumerate(branches.branches)],
    cmap="inferno", sampling=fit.object_sampling_A[1] / factor, units="Å",
)
```

The other branch reconstructs nearly the same phase, negated: the gold
particles are bright at the kept angle and dark on the other branch; zoom in
and compare the two histograms. `branches.show()` draws the same panels above
the evidence the choice used, from the native phases. Polarity is a heuristic,
not a measurement: on crystals the preferred tail can change with the region, so
check that a crop of the field agrees before you trust the angle. If you know
the angle from your microscope, pass it: `SSB(data, scan_detector_rotation_deg=358)`.

The fit is the default of the table above, model-based and tilt-corrected. To
constrain it, or to run one of the other two methods, call `find_aberrations`
before `reconstruct()`; the reconstruction then reuses that fit:

```python
# ssb.find_aberrations(search_ranges={"C10_nm": (-50, 50)})  # tilt-corrected, defocus searched in -50..50 nm
# ssb.find_aberrations(tilt=False)                           # the same model at tilt 0
# ssb.find_aberrations(tilt=False, objective="phase_variance")  # the phase-variance backup
```

## Review in ShowPtycho

```python
viewer = ssb.show(local_tilt=True, show_fft=True, save_dir="ssb-results")
viewer
```

`ssb.show()` opens ShowPtycho on the fitted session, at 358 degrees and C10
near -18.4 nm, and shows the phase and its FFT. Output sampling is chosen from
the calibration: 4x here, 0.125 A per pixel. Drag C10, C12 and phi12 to see the
phase update live. The launch table identifies saved values, overrides and
estimates. **Save** restores a matching full-field joint fit and regional state
on restart.

The [H5/QEM handoff notebook](showptycho_to_myptycho.ipynb) initializes independent
myptycho runs from these regional models. `viewer.calibration(region="Upper")`
returns a snapshot of current numerical settings without saving or refitting.

## View a reconstruction on any computer

To look at a reconstruction you already have, from SSB or any other method,
pass the array. It needs no GPU and no quantem.gpu: a real array is the phase,
a complex array is the object wave, shown as its phase and amplitude. NumPy
arrays and torch tensors on any device work.

```python
import numpy as np
from quantem.widget import ShowPtycho

rows, cols = np.mgrid[0:256, 0:256]
phase = 0.3 * np.cos(2 * np.pi * rows / 15.6) * np.cos(2 * np.pi * cols / 15.6)   # 3.9 A lattice
amplitude = 1.0 - 0.05 * np.cos(2 * np.pi * rows / 15.6) ** 2
object_wave = amplitude * np.exp(1j * phase)

ShowPtycho(object_wave, sampling=0.25, units="Å", show_fft=True)
```

The phase, amplitude (Amp), FFT, histograms and the scale bar show the array's
own numbers; a float64 or complex128 array is drawn in float32 and the widget
prints the largest change that causes. The aberration sliders, Save, pins and
sweeps need an SSB session, and the widget says so in one line.

## No detector binning

Build the reconstruction at the **native detector size**. `SSB.open` and
`quantem.gpu.io.load` keep every detector pixel; do not bin an array before
passing it to `SSB(...)`. Native (e.g. 192x192) is what resolves light columns
such as oxygen in a perovskite; binning throws that away. Binning also breaks
the HTML export (the browser cannot bin), so keep the whole workflow un-binned.

## Regional joint models

Enable **Local Tilt** to draw independent rectangular regions on the overview.
Moving or resizing a region invalidates its previous fit; automatic fitting
reruns all five repeats of 200 trials plus Newton. Aberrations, tilt and depth
are fitted together. Select a region to inspect and adjust its own controls.
The overview and other regional models are retained.

A region estimates a local model, but its displayed phase is a crop of the
**full-scan reconstruction** using that model. Smaller regions may have weaker
parameter identifiability; compare repeat variability and the images before
interpreting tilt as physical bending or a boundary.

Hover over a region's upper-right corner to delete it, or use its durable
control-menu action. Display settings include the maximum number of columns.
The [iterative handoff notebook](showptycho_to_myptycho.ipynb) uses each current
regional model to initialize a separate run.

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
simulation uses [abTEM](https://abtem.readthedocs.io), a separate multislice
package that quantem.widget does not install or depend on: install it yourself
(`pip install abtem`) to run this cell. It runs on the GPU and takes about a
minute; no data file is needed.

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
          scan_sampling_A=(0.25, 0.25), detector_sampling_mrad=(det_mrad, det_mrad), scan_detector_rotation_deg=0.0)
```

First the thin-sheet model, `tilt=False`. Look at the FFT: are the lattice
spots equally sharp in every direction?

```python
standard = ssb.find_aberrations(tilt=False, verbose=False)
ssb.show(show_fft=True)
```

Now let the sample lean. `find_aberrations()`, the default, fits the same aberrations together
with a tilt and a depth spread. Compare: which spots sharpened, and did the
defocus move?

```python
tilted = ssb.find_aberrations(verbose=False)
ssb.show(show_fft=True)                 # opens on the fitted tilt; drag the Sample tilt sliders
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
`find_aberrations()` (tilt-corrected by default), `preview(sample_tilt_mrad=..., depth_spread_nm=...)` and the
evidence behind the defaults.

## Workflow checklist

1. Load the native H5 or QEM acquisition with `io.load` and set missing beam values on `data.calibration`.
2. Open `SSB(data)`, which estimates the rotation, then `reconstruct()`, which fits the aberrations and settles the 180-degree branch.
3. Open ShowPtycho with `ssb.show()` and review the launch calibration.
4. Jointly fit aberrations, tilt and depth in the widget, or restore a matching saved joint fit.
5. Inspect the phase, FFT and repeat variability; lower fit loss alone does not establish a physical tilt.
6. Save the review for restart, then use the regional handoff for iterative reconstruction.
