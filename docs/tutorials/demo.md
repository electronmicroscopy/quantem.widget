# Interactive demo route

Inspect a volume in depth. These two notebooks are the short route through the
volume widgets. Run each notebook from the top in JupyterLab; the final expression in a
widget cell displays the interactive view.

| Step | Notebook | Try during the demo |
|---|---|---|
| 1 · Depth playback | [Show3D](show3d.ipynb) | Scrub, play, compare Avg 1 with Avg 3, and inspect the FFT |
| 2 · Depth comparison | [Show3DSlices](show3dslices.ipynb) | Move the crosshair, rotate an oblique cut, and inspect the linked 3D view |

Show3D downloads a public HAADF image and builds a moving-crop
stack. Show3DSlices uses a synthetic scalar volume. None is presented as an
experimental reconstruction or a quantitative multislice simulation.

## Prepare once

Follow [Installation](../install.md) in the **same environment as the Jupyter
kernel**. Each notebook calls `qw.profile()` so a presenter can record the actual environment.

For a clone of this revision:

```bash
python -m pip install -e .
npm ci
npm run build
python -m jupyter lab
```

Select that environment's Python kernel and restart it after changing the
installation. Use a WebGPU-capable browser on localhost or HTTPS. If WebGPU is
unavailable, check the browser's graphics support and secure origin before the
demo. An ordinary HTTP LAN/IP URL is not equivalent to localhost.

Run the notebooks once before presenting. Show3D's public example needs network
access on the first run to download its data. Standalone widget HTML carries
its widget-manager runtime and opens with the network disconnected.

## What the audience should take away

- A moving average is a display mean across planes. It is distinct from a full
  depth projection and leaves the reconstruction unchanged.

## Other tools

| Task | Starting notebook or guide |
|---|---|
| Image, FFT, contrast, ROI | [Show2D](show2d.ipynb) |
| Maps with physical coordinates | [Plot2D](plot2d.ipynb) |
| Time traces and metrics | [Show1D](show1d.ipynb) |
| Scan positions and virtual detectors | [Show4DSTEM](show4dstem.ipynb) |
| Direct ptychography phase and aberrations | [ShowPtycho workflow guide](showptycho.md) |

These are separate workflows, not additional prerequisites for the two-step
demo. ShowPtycho's linked page is a workflow guide, not a newly validated
standalone notebook. See the [API index](../api/index.md) for complete parameters
and [HTML export](widget_export.ipynb) for sharing an interactive result.
