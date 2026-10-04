# ShowCIF

Inspect a CIF or ASE structure with linked WebGPU atom, column, and potential
views. Start with the crystal and its orientation:

```python
from quantem.widget import ShowCIF

viewer = ShowCIF("crystal.cif", repeats=(4, 4, 8), zone_axis=(0, 0, 1))
viewer
```

Install `quantem.widget[crystal]` for ASE and the optional abTEM potential tables.
Rendering requires WebGPU in a secure browser context (HTTPS or localhost).

## Reference

```{eval-rst}
.. autoclass:: quantem.widget.ShowCIF
   :members: atoms, export_html
```

## Structure and orientation

`repeats` expands the original unit cell along lattice vectors a, b, c. ASE
expands CIF symmetry; partial or mixed occupancies require an explicitly ordered
model. The maximum inspection size is 250,000 atoms.

`zone_axis` is the direct-lattice direction `u*a + v*b + w*c`, including for
oblique cells. It is not a reciprocal-plane normal. The 3D camera rotates
independently of the projection direction. `atoms()` returns an independent
copy of the original cell, before repeats, display filters, or specimen tilt.

```python
viewer.zone_axis = [1, 1, 0]
viewer.specimen_tilt_mrad = [10, -5]  # (row, col)
original = viewer.atoms()
```

Specimen tilt ranges from −15 to +15 mrad per component. Positive row tilt leans
a nominal beam column downward with depth; positive column tilt leans it right.
The components define a rigid rotation vector about the nominal beam frame's
right and up axes, centered on the inspection cell. At nonzero tilt, `[uvw]`
continues to label the nominal starting direction. Atom projections, depth
bounds, slice membership, and potential maps use this same geometry.

## Microscope field of view

```python
viewer = ShowCIF(
    "crystal.cif",
    view_mode="microscope",
    field_of_view_A=40,
    magnification_calibration=(1_000_000, 100),
)
viewer
```

Microscope mode sets a physical square field of view; it does not add atoms.
The optional calibration is `(reference magnification, reference FOV in Å)`
for one fixed camera/acquisition geometry. Magnification scales that FOV
inversely. No universal magnification calibration is assumed.

Orthogonal views show column–depth and row–depth in the selected beam frame,
including oblique cells. They are not necessarily crystallographic a/b/c planes.
In microscope mode they fit the specimen depth while the main projection retains
its entered FOV. Every atom and potential panel has a physical scale bar.

## Potential and expected phase

```python
viewer = ShowCIF(
    "crystal.cif",
    potential=True,
    num_slices=16,
    energy_keV=300,
    potential_quantity="phase",
    potential_colormap="magma",
)
viewer
```

Enable **Potential / phase maps** in the settings gear. This optional preview
supports up to 8,192 atoms. Larger structures remain available in the atom
viewer; reduce repeats to enable their potential preview. No atoms are silently
removed to meet the limit.

Maps use neutral abTEM Lobato **infinite atomic projections**, assigned to depth
slabs by atomic-site position. They are not finite-z potential integrals, bonded
charge density, or a multislice propagation. A finite inspection patch is used;
there are no periodic images beyond the explicit repeats.

| Quantity | Meaning |
|---|---|
| Integrated · V Å | Sum of projected atomic potentials in the slab |
| Thickness Average · V | Integrated potential divided by that slab's thickness |
| Expected Phase · rad | Relativistic interaction constant σ(E) × integrated V Å |

Phase uses the positive `exp(i*phi)` convention and is unwrapped. At 300 keV,
σ is approximately 0.000652616 rad/(V Å). The energy control spans 0–300 keV;
zero explicitly disables phase, which is undefined there. Individual slab
phases sum to the full projected phase. This is **not thick-specimen exit-wave
phase**: propagation, channeling, aberrations, and partial coherence are absent.

`potential_sigma_A` is the explicit transverse Gaussian regularization of the
atomic tables (default 0.08 Å, range 0.04–0.5 Å). It is not thermal displacement
or a fitted experimental resolution. **Blur σ** applies an additional WebGPU
display filter, with zero extension outside the finite patch and a three-sigma
kernel. It preserves the original potential planes. The preview grid is explicit
(128, 256, or 512 pixels); FOV changes alter its Å/px readout and show an
undersampling warning where appropriate. **Max** changes color limits only.

## Interactive controls

ShowCIF shares Show3D/Show3DSlices sliders, colormap selection, play/pause icons,
and light/dark colors. Controls have square corners. The settings gear hides
less-used panels until needed.

| Control | State or behavior |
|---|---|
| Unit Cells | `repeats`; updates atom instances and the physical cell |
| Species | `visible_species`; hides atoms and excludes them from the potential preview |
| Beam | `zone_axis`; changes projections, not the 3D camera |
| Row / Col tilt | `specimen_tilt_mrad`; animation-frame-coalesced preview, saved on release |
| View / FOV | `view_mode`, `field_of_view_A`; fits the cell or an explicit physical field |
| Settings gear | Toggles column projection, orthogonal views, slice controls, and potential maps |
| Atom slice / Avg | One zero-based slice center and a moving slab width; atoms are not averaged |
| Potential Depth / Avg / Play | Independent potential slice center, arithmetic mean width, and playback |
| Color / Quantity / Energy | `potential_colormap`, `potential_quantity`, `energy_keV` |
| Columns / Blur / Max | Preview layout and display-only image controls |
| Atom radius / Zoom | Schematic markers and camera scale; coordinates remain unchanged |

The atom and potential slicers both start with `num_slices`. Potential slicing
and playback settings can subsequently use their own 1–64 plane count.
Averaging uses Show3D's centered window, shifted inward at the ends: with 16
planes and Avg 3, centers 0 and 1 use planes 0–2, center 2 uses 1–3, and center
15 uses 13–15. It neither wraps nor shortens the edge mean.

The potential moving average is an arithmetic **mean per plane**, not a sum.
For thickness-average V it divides by one plane's thickness; for phase it
multiplies the mean integrated potential by σ(E). Full Projection continues to
show the full-depth sum (or full-depth thickness average for V). Atom panels
show the selected slab union, without averaging atomic coordinates.

The optional gallery shows individual planes. Hover or keyboard focus previews a
plane; leaving restores the committed selection, and clicking commits it.
Playback loops at 5 fps by default and pauses during scrubbing, gallery preview,
when hidden, or at invalid phase energy. GPU-resident averaging and filtering
avoid image readback; geometry changes recompute the potential planes.

## HTML export

```python
viewer.export_html("structure.html")
```

The export embeds atomic coordinates, potential tables, and synced settings
(including repeats, tilt, FOV, energy, colormap, and orthogonal visibility).
Transient camera, panel-visibility, playback, blur, and slice selections reset
when reopened. The widget manager loads from a CDN: the export requires network
access and WebGPU, and is not an offline simulation package.
