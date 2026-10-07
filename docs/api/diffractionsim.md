# DiffractionSim

Rotate a unit cell and watch its electron diffraction pattern follow. The left
panel draws the cell, the right panel the nanobeam pattern, CBED disks, or the
Kikuchi pattern of the same orientation. Kinematical and Bloch wave
intensities are computed in the browser, so dragging the crystal needs no
Python round trip.

```python
from quantem.widget import DiffractionSim

sim = DiffractionSim("Si (diamond cubic)", zone_axis=[1, 1, 0])
sim
```

The crystal can be a preset name (`DiffractionSim.presets_available()`), a CIF
path, an ASE `Atoms` object, or a quantem `Crystal`.

## Requirements

DiffractionSim needs quantem with `quantem.diffraction.Crystal` and
`quantem.diffraction.bloch`. They come with electronmicroscopy/quantem
PR #297 and are in no quantem release yet; the first release after 0.1.9 will
include them. That quantem version also brings ASE and spglib, so no extra is
needed. With an older quantem, constructing the widget raises an `ImportError`
that names this requirement, and the DiffractionSim tests are skipped.

This widget is unrelated to [ShowDiffraction](showdiffraction), which measures
experimental patterns.

## Reference

```{eval-rst}
.. autoclass:: quantem.widget.diffraction_sim.DiffractionSim
   :members: presets_available, crystal, set_crystal, set_zone_axis, compute_kossel_reference, state_dict, load_state_dict, save, export_html
```

## Physics

`energy_keV` and `k_max_inv_A` set the reflection list, computed in Python:
the reciprocal lattice points within `k_max_inv_A`, each with its kinematical
`|F_g|^2` (Lobato and Van Dyck factors) and its Bloch coupling `U_g`
(absorptive Weickenmeier and Kohl factors at `energy_keV`). Changing either
trait recomputes the list.

With `dynamical=True`, reflections within `sg_max_inv_A` of the Ewald sphere
enter a Bloch wave calculation; the rest take thin-slab intensities. With
`dynamical=False`, every reflection takes `|F_g|^2` with a Gaussian excitation
envelope of width `sigma_excitation_inv_A`. `precession_deg` averages the
pattern over `n_precession` incident tilts on the precession cone.

`mode` is `"nanobeam"`, `"cbed"` or `"kikuchi"`. `"kossel"` is accepted as the
earlier name of `"kikuchi"` and stored as `"kikuchi"`. The pixel rendering of
the Kikuchi mode needs a reference pattern from `compute_kossel_reference()`
(about a minute for silicon at 3 mrad).

Units are in the trait names: `_keV`, `_A` (Å), `_inv_A` (1/Å), `_mrad`,
`_deg`, `_px` (CSS pixels), `_pct` (percent).

### Data sources

| Data | Source | Used for |
|---|---|---|
| Kinematical scattering factors | Lobato & Van Dyck, Acta Cryst. A 70, 636 (2014), [doi:10.1107/S205327331401643X](https://doi.org/10.1107/S205327331401643X), from quantem | `\|F_g\|^2` in Python; `js/diffractionsim-web/lobato.ts` holds the same parameters, copied from quantem's `data/lobato.json` (MIT license), for CIFs loaded in the web build |
| Absorptive Bloch couplings | Weickenmeier & Kohl, Acta Cryst. A 47, 590 (1991), [doi:10.1107/S0108767391004774](https://doi.org/10.1107/S0108767391004774), from quantem | `U_g` of every crystal set up in Python, including the web build presets |
| Atom colors and radii | ASE (`jmol_colors`, `covalent_radii`) | Cell drawing |

CIFs loaded in the browser (web build only) do not get calculated absorption:
their absorptive coupling is an approximate fixed fraction of the elastic one
(`ABSORPTION_FRACTION = 0.08` in `js/diffractionsim-web/cif.ts`), which damps
the thickness fringes at a plausible rate. The presets carry the
Weickenmeier-Kohl factors.

## Interactive controls

| Control | Trait | Expected effect |
|---|---|---|
| Crystal menu | `preset` | Loads the preset and recomputes the reflection list |
| Energy menu | `energy_keV` | Recomputes the reflection list at the new energy |
| Zone axis field, Go | `orientation` | Puts [uvw] (or [uvtw] for hexagonal cells) along the beam |
| Copy | none | Copies both panels as one PNG |
| Export | none | PNG of both panels, or a standalone HTML page |
| Reset | `orientation` | Returns to the identity orientation |
| Drag the cell | `orientation` | Tilts the crystal; the near face follows the pointer |
| Drag the pattern | `orientation` | Tilts the crystal so the pattern follows the pointer |
| Shift-drag or two-finger twist | `orientation` | Rotates about the beam |
| Double-click a disk | `orientation` | Tilts to the two-beam condition of that reflection |
| Double-click empty space | `orientation` | Moves the Laue circle centre to the click |
| x, y, z buttons and Step | `rotation_step_deg` | Rotates about the screen axes by the step |
| Spin ↔ ↕ and speed | `rotation_speed_deg_per_s` | Rotates continuously about a screen axis |
| Cell Axes, Axis Labels, Polyhedra, Ewald Sphere | `show_cell_axes`, `show_labels`, `polyhedra`, `show_ewald` | Cell panel overlays |
| Cells | `n_cells` | Number of cells drawn along a, b, c (1 to 6) |
| Nanobeam / CBED / Kikuchi | `mode` | Pattern type |
| Markers / Disks / Pixels (Lines / Pixels) | `render` | Pattern rendering |
| Dynamical | `dynamical` | Bloch wave or kinematical intensities |
| hkl Labels | `show_hkl` | Labels the strongest reflections or lines |
| Kikuchi Lines | `show_kikuchi` | Overlays Kikuchi line pairs on nanobeam and CBED |
| Quality | `quality` | Bloch beam cap and CBED tilt sampling |
| Compute Reference | `kossel_json` | Computes the Kikuchi reference pattern in Python |
| Thickness | `thickness_A` | Specimen thickness |
| Semiangle | `semiangle_mrad` | CBED and nanobeam disk radius |
| Precession | `precession_deg` | Precession half angle |
| Pattern Range | `pattern_range_inv_A` | Scattering vector at the panel edge |
| Field Half-Angle | `field_mrad` | Kikuchi pattern half angle |
| Excitation Error σ | `sigma_excitation_inv_A` | Kinematical envelope width (Dynamical off) |
| Marker sliders | `marker_power`, `marker_size_px` | Marker area exponent and largest radius |
| Scale, Exponent, Color, histogram | `scaling`, `power`, `cmap`, `vmin_pct`, `vmax_pct` | Pixel rendering display |

`cmap` takes the shared widget colormaps, their `_r` reversals, and
`turbo_black` and `turbo_black_r`, whose lookup tables come from
`quantem.core.visualization` (trait `cmap_luts`).

## State

```python
sim.save("si_110.json")
sim2 = DiffractionSim(state="si_110.json")
```

`state_dict()` holds the settings that restore the view: preset, energy,
orientation, mode and every display setting. It does not hold the reflection
list or the Kikuchi reference pattern, which are recomputed. A crystal from a
CIF or ASE is passed again: `DiffractionSim("my.cif", state="my.json")`.

With `save_state=False` (the default), a saved notebook keeps the settings and
a PNG of both panels, which the frontend captures after the view settles, and
leaves out the Kikuchi reference pattern. `save_state=True` embeds everything.

## HTML export

```python
sim.export_html("si_110.html", presets=["Al (fcc)", "Ti (hcp)"])
```

The page follows the [HTML export](html-export) protocol: it hydrates with the
ipywidgets HTML manager, which loads from a CDN, so opening it needs network
access. All simulations then run in the browser: rotation, thickness, mode and
display changes need no Python. The beam energy and `k_max_inv_A` are fixed to
the exported values; `presets` embeds additional structures for the crystal
menu. The Export menu in the widget writes the same page through the export
bridge. Only `mode="single"`, `encoding="full"` and `downsample=None` apply.

## Web page build

`npm run build:web` (not `npm run build`) writes
`dist/web/diffraction-sim.js`, a single ESM module with
`export default { render }` for the MyST `anywidget` directive, built from
`js/diffractionsim-web/` without React. The website uses it by hand-copying
the file:

```bash
npm run build:web
cp dist/web/diffraction-sim.js ../site/landing/widgets/diffraction-sim.js
```

Its structure menu is `js/diffractionsim-web/presets.ts` (about 1 MB),
generated by `scripts/diffraction_sim_presets.py` at 200 keV and
`k_max` 3.0 1/Å. The file is committed because the web build needs it and
regenerating it needs the unreleased quantem above; run the script again after
changing the presets or the crystal code (`--check` compares with the
committed file). Directive options are listed at the top of
`js/diffractionsim-web/index.ts` and use the trait names; the web build
defaults to CBED with Kikuchi lines on.

```text
:::{anywidget} ../widgets/diffraction-sim.js
{"preset": "Si (diamond cubic)", "zone_axis": [1, 1, 0], "thickness_A": 400}
:::
```
