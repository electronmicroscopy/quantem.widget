# Compare several 4D-STEM datasets or tilts

Use this workflow for a tilt series, repeated acquisition, dose series, or
other set of compatible `*_master.h5` files. One shared detector controls every
virtual-image panel, while the selected dataset supplies the diffraction
pattern.

The masters must have matching scan shape, detector shape, and frame count.
List them explicitly when order matters, such as a known tilt-angle sequence.

## Jupyter notebook

```python
from quantem.gpu.io import load
from quantem.widget import Show4DSTEM

masters = [
    "/data/tilts/sample_m6deg_master.h5",
    "/data/tilts/sample_m4deg_master.h5",
    "/data/tilts/sample_m2deg_master.h5",
    "/data/tilts/sample_0deg_master.h5",
    "/data/tilts/sample_p2deg_master.h5",
    "/data/tilts/sample_p4deg_master.h5",
    "/data/tilts/sample_p6deg_master.h5",
]

data = load(masters)
viewer = Show4DSTEM(data)
viewer
```

That is the complete beginner call. Native detector sampling and the source
count dtype are preserved. `Show4DSTEM` detects the extra dataset axis, opens
the Multiple view, uses the filenames as labels, and shows the selected
dataset's diffraction pattern. Put the sample name and tilt angle in each
filename so the viewer labels remain meaningful.

The first usable panel should appear before the complete series finishes
loading. Each later panel fills its reserved position as that master becomes
resident. You can begin dragging the detector on the loaded panels while the
remaining masters continue loading.

Use **Selected** for ordinary tilt review: clicking a virtual-image tile makes
its dataset the source of the diffraction pattern. Use **Average** only when
the mean diffraction pattern across the loaded visible datasets is the
scientific quantity you intend to inspect.

## Local WebGPU viewer

List the masters in the desired order:

```bash
quantem show4dstem \
  /data/tilts/sample_m6deg_master.h5 \
  /data/tilts/sample_m4deg_master.h5 \
  /data/tilts/sample_m2deg_master.h5 \
  /data/tilts/sample_0deg_master.h5 \
  /data/tilts/sample_p2deg_master.h5 \
  /data/tilts/sample_p4deg_master.h5 \
  /data/tilts/sample_p6deg_master.h5 \
  --backend webgpu --html
```

If a folder contains only the compatible masters you want to compare, the
short form is:

```bash
quantem show4dstem /data/tilts --backend webgpu --html
```

Both commands keep native detector sampling and use the compact browser browse
dtype. Add `--dtype uint16` only when the browser view must preserve counts
above 255.

The generated viewer opens in Multiple mode and loads datasets progressively.
On macOS, double-click `Show4DSTEM.command` and keep its Terminal window open.
The source HDF5 files remain local; WebGPU interaction runs in the browser.

## What to verify

1. Panel labels match the intended sample and tilt order.
2. Each virtual image appears as its dataset loads; the grid does not wait for
   the final master before becoming useful.
3. Clicking a tile changes the selected diffraction pattern.
4. Dragging the detector updates every loaded virtual-image panel immediately.
5. Average produces a diffraction pattern from the loaded visible datasets and
   does not remain in a requested/loading state.

## Next steps

- [Open one 4D-STEM dataset](show4dstem_single)
- [Show4DSTEM export recipes](show4dstem_export)
- [Show4DSTEM API reference](../api/show4dstem)
