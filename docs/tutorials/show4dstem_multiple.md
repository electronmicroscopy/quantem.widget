# Compare several 4D-STEM datasets or tilts

Use this workflow for a tilt series, repeated acquisition, dose series, or
other set of compatible `*_master.h5` files. One shared detector controls every
virtual-image panel, while the selected dataset supplies the diffraction
pattern.

The masters must have matching scan shape, detector shape, and frame count.
List them explicitly when order matters, such as a known tilt-angle sequence.

## Jupyter notebook

```python
from quantem.widget import Show4DSTEM, read_4dstem

masters = [
    "/data/tilts/sample_m6deg_master.h5",
    "/data/tilts/sample_m4deg_master.h5",
    "/data/tilts/sample_m2deg_master.h5",
    "/data/tilts/sample_0deg_master.h5",
    "/data/tilts/sample_p2deg_master.h5",
    "/data/tilts/sample_p4deg_master.h5",
    "/data/tilts/sample_p6deg_master.h5",
]

acquisitions = read_4dstem(masters)   # one dataset per master, encoded on the GPU when there is one
viewer = Show4DSTEM(acquisitions)
viewer
```

That is the complete beginner call. Native detector sampling and the source
count dtype are preserved, and each acquisition stays encoded on the GPU (about
0.1 to 2 GiB for a 512 x 512 x 192 x 192 scan). `Show4DSTEM` opens the list in
the Multiple view, uses the filenames as labels, and shows the selected
dataset's diffraction pattern. Put the sample name and tilt angle in each
filename so the viewer labels remain meaningful.

`read_4dstem(masters)` returns after every master has loaded. To see the first panel
while the rest are still loading, open the folder instead:

```python
viewer = Show4DSTEM.from_folder("/data/tilts")
```

The viewer opens after the first master, and each later panel joins the grid
as its master loads. You can begin dragging the detector on the loaded panels
while the remaining masters continue loading. `from_folder` orders masters by
file name, so list them explicitly with `read_4dstem` when the tilt order differs.

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
2. With `from_folder` or the WebGPU folder, each virtual image appears as its
   dataset loads; the grid does not wait for the final master before becoming
   useful.
3. Clicking a tile changes the selected diffraction pattern.
4. Dragging the detector updates every loaded virtual-image panel immediately.
5. Average produces a diffraction pattern from the loaded visible datasets and
   does not remain in a requested/loading state.

## Next steps

- [Open one 4D-STEM dataset](show4dstem_single)
- [Show4DSTEM export recipes](show4dstem_export)
- [Show4DSTEM API reference](../api/show4dstem)
