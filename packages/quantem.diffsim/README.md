# quantem.diffsim

An interactive electron diffraction simulator for web pages and Dash apps. It shows a crystal and its diffraction pattern side by side for any orientation: nanobeam, convergent beam (CBED) and Kikuchi patterns, with kinematical or Bloch wave intensities, thickness, convergence, precession and beam energy controls. Every calculation runs in the browser, so the server only sends the structure.

The structure factors use the Lobato and Van Dyck electron scattering factors ([Acta Cryst. A 70, 636 (2014)](https://doi.org/10.1107/S205327331401643X)), and the Bloch wave couplings use the absorptive Weickenmeier and Kohl scattering factors with a Debye-Waller factor ([Acta Cryst. A 47, 590 (1991)](https://doi.org/10.1107/S0108767391004774)), as computed in [quantEM](https://electronmicroscopy.github.io/quantem-docs/). The same simulator runs in Jupyter as `quantem.widget.DiffractionSim`.

The package has no dependencies. It ships the browser bundle (one ES module, about 1.2 MB) and a few Python helpers.

## Standalone page

```python
from pymatgen.core import Lattice, Structure
import quantem.diffsim as diffsim

si = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.431), ["Si"], [[0, 0, 0]])
diffsim.save_html("si.html", si, zone_axis=[1, 1, 0], mode="nanobeam", title="Si")
```

`structure_to_cif` accepts a pymatgen `Structure`, an ASE `Atoms` object, CIF text or a CIF path. A pymatgen structure is converted to its conventional standard cell first (`conventional=False` keeps the given cell), so zone axes refer to the conventional lattice.

## Dash

```python
from dash import Dash, Input, Output, dcc, html
from quantem.diffsim import dash as diffsim_dash
import quantem.diffsim as diffsim

app = Dash(__name__)
diffsim_dash.register(app)
app.layout = html.Div([dcc.Store(id="structure"), diffsim_dash.layout("tem")])

@app.callback(Output(diffsim_dash.settings_id("tem"), "data"), Input("structure", "data"))
def show(structure):
    return diffsim.settings(Structure.from_dict(structure), zone_axis=[0, 0, 1], mode="nanobeam")
```

`register(app)` serves the bundle from the app's Flask server and adds one clientside callback that draws every simulator in the layout. Each new settings dictionary redraws that simulator.

## Settings

`settings()` checks the names below. Lengths are in Å, angles in mrad unless named `_deg`, and scattering vectors in 1/Å.

| Name | Default | Meaning |
|---|---|---|
| `zone_axis` | [110], or [0001] for hexagonal cells | Initial beam direction [uvw] or [uvtw] |
| `mode` | `"cbed"` | `"nanobeam"`, `"cbed"` or `"kikuchi"` |
| `dynamical` | `True` | Bloch wave intensities; `False` gives kinematical intensities |
| `energy_keV` | 200 | Beam energy |
| `thickness_A` | 400 | Specimen thickness |
| `semiangle_mrad` | 3 | Convergence semiangle |
| `precession_deg` | 0 | Precession angle |
| `pattern_range_inv_A` | 3 | Largest scattering vector shown |
| `field_mrad` | 50 | Field of view of the Kikuchi pattern |
| `thermal_sigma_A` | 0.05 | RMS thermal displacement |
| `power`, `vmax` | 0.5, 0.5 | Display brightness exponent and upper contrast limit |
| `show_kikuchi`, `show_hkl`, `show_labels`, `show_ewald`, `polyhedra` | | Overlays |
| `n_cells` | [1, 1, 1] | Unit cells drawn |
| `panel_width_px` | 400 | Panel size |
| `title`, `structure_name` | | Heading and structure name |
| `show_logo` | `True` | The "Powered by quantEM" link to the quantEM documentation |

## License

MIT. Source and issues: [electronmicroscopy/quantem.widget](https://github.com/electronmicroscopy/quantem.widget).
