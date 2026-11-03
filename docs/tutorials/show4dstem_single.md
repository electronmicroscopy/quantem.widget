# Open one 4D-STEM dataset

Use this workflow when you want to inspect one completed `*_master.h5` file:
move through scan positions, drag a virtual detector, and compare BF, ABF, and
ADF images without reducing the detector grid.

## Jupyter notebook

Read the master, then let Jupyter render the viewer as the last expression:

```python
from quantem.widget import Show4DSTEM, read_4dstem

loaded = read_4dstem("/data/session/scan_001_master.h5")
viewer = Show4DSTEM(loaded)
viewer
```

The default keeps native detector sampling and the source count dtype. With
quantem.gpu installed, the reader selects CUDA on an NVIDIA workstation or
Metal/MPS on Apple Silicon and keeps the acquisition encoded on the GPU as a
`Dataset4dstemGPU`; without a GPU it reads the counts into a quantem core
`Dataset4dstem` in host memory, and the same cell runs. On the GPU: a 512 x 512 x 192 x 192 uint16 scan,
18 GiB as a dense array, occupies about 0.1 to 2 GiB. Virtual images are summed
on the encoded storage, and each diffraction pattern is decoded as you move the
scan position.

After the widget appears:

1. Drag over the real-space image to choose a scan position and inspect its
   diffraction pattern.
2. Drag or resize the detector on the diffraction pattern. The virtual image
   updates while you drag.
3. Try BF, ABF, and ADF presets, then adjust diffraction and virtual-image
   contrast independently.

## Local WebGPU viewer

Use the CLI when you want the same dataset in a local browser without keeping
a notebook kernel alive:

```bash
quantem show4dstem /data/session/scan_001_master.h5 --backend webgpu --html
```

This keeps native detector sampling and uses the compact browser browse dtype.
Add `--dtype uint16` only when the browser view must preserve counts above 255.

The command creates a folder containing `index.html`, `Show4DSTEM.command`, a
nested `.viewer/`, and a nested `data/` directory linked to the source HDF5
family. On macOS, double-click `Show4DSTEM.command`. Keep its Terminal window
open while using the viewer; closing it stops the local server.

The browser fetches diffraction frames as needed and computes the virtual image
with WebGPU. A loading message means data are moving from the local HDF5 files
into the browser; it is not an upload to a remote service.

## Next steps

- [Compare several datasets or tilts](show4dstem_multiple)
- [Show4DSTEM export recipes](show4dstem_export)
- [Show4DSTEM API reference](../api/show4dstem)
