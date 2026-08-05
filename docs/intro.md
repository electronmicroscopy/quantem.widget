# quantem.widget

Interactive, GPU-aware Python widgets for electron microscopy, built on
[anywidget](https://anywidget.dev). Each widget renders a real-time canvas in
JupyterLab, VS Code, or Colab.

```python
from quantem.gpu.io import load
from quantem.widget import Show1D, Show2D, Show3D, Show3DSlices, Show4DSTEM, ShowPtycho, ShowEDS, ShowDiffraction, ShowFolder
```

## Quickest start: no notebook needed

After [installing](install), point the `quantem` command at a file or folder and
it renders the right viewer in your browser:

```bash
quantem show2d image.tif         # an image            -> Show2D
quantem show3d ./frames/         # a folder of frames  -> Show3D scrub
quantem show4dstem ./masters/    # 4D-STEM master(s)   -> live viewer (or --html)
```

It saves to `~/Downloads`, opens automatically, and picks the GPU (CUDA / Apple
Metal) for you. Full details on [the command line](cli) page.

## Built for two platforms

We serve two audiences first:

- **macOS on Apple M-chips** - the Metal (MPS) GPU.
- **Linux with NVIDIA CUDA** - workstations and HPC.

**CUDA and MPS are the primary backends.** Work stays on the GPU as PyTorch
tensors; we avoid NumPy on the hot path. Automatic scientific loading and
compute never silently fall back to CPU: an unsupported machine fails with a
corrective error. The explicit CPU reference exists for parity tests, while the
viewers can still display ordinary NumPy arrays supplied by a user. For large
datasets, bin the detector at load (`det_bin`) to cut memory and speed first
paint - see [Load and I/O](api/io).

## Widgets

| Widget | Use it for | Tutorial · API |
|---|---|---|
| `Show1D` | Interactive traces, live reconstruction metrics, line profiles, and linked image snapshots | [API](api/show1d) |
| `Show2D` | One or many 2D images: contrast, FFT, ROIs, line profiles, scale bars | [tutorial](tutorials/show2d) · [API](api/show2d) |
| `Show3D` | A 3D volume scrubbed slice-by-slice (e.g. a ptychographic object) | [tutorial](tutorials/show3d) · [API](api/show3d) |
| `Show3DSlices` | Side-by-side slices of a 3D volume across an axis | [tutorial](tutorials/show3dslices) · [API](api/show3dslices) |
| `Show4DSTEM` | 4D-STEM: live virtual detectors, multi-master review, and WebGPU HTML export | [tutorial](tutorials/show4dstem) · [export](tutorials/show4dstem_export) · [API](api/show4dstem) |
| `ShowPtycho` | Ptychography aberration review: phase, FFT, BF-count tradeoffs, and WebGPU folder export | [API](api/showptycho) |
| `ShowDiffraction` | 2D/3D diffraction d-spacing: Bragg spots, rings, center finding, k calibration | [tutorial](tutorials/showdiffraction) · [API](api/showdiffraction) |
| `ShowFolder` | Folder-level microscopy browser: navigate a session, review thumbnails, select files/folders, and save curation state | [tutorial](tutorials/showfolder) · [API](api/showfolder) |

The [Tutorials](tutorials/download_data) walk through each widget on real public
data where practical, with compact synthetic data only where it keeps an example
portable. Real tutorial datasets are downloaded from public data hosting such as
Hugging Face and cached locally; they are not committed to this repository or
bundled into the Python wheel. That keeps clone size and microscope-PC installs
small while still letting the rendered docs use realistic microscopy examples.
The [Show4DSTEM export recipes](tutorials/show4dstem_export) show how to choose
between compact report HTML, interactive raw-4D WebGPU HTML, and terminal
exports. The [ShowFolder tutorial](tutorials/showfolder) covers folder browsing
workflows and how to [save and share widget exports](tutorials/widget_export). The
[API reference](api/index) documents every parameter, method, and interactive
control (and doubles as a UI-test spec for automated agents). All example data
here is synthetic or pulled from a public Hugging Face dataset - no private data
ships in the docs.

Every widget accepts a NumPy array, a PyTorch tensor (CPU or GPU), or a quantem
`Dataset` (`Dataset2d` / `Dataset3d` / `Dataset4dstem`), pulling calibration and
units automatically from the dataset when present.

## Offline by default in these docs

The Show2D, Show3D, and Show3DSlices examples on this site were exported with
`encoding="uint8"`, which bakes the display data into the widget as a **uint8**
stack (4x smaller than float32, and the colormap clamps to 256 levels anyway so
it looks identical). The canvas below each example stays fully
interactive in this static page with no running kernel: scrub, zoom, change
contrast, toggle the FFT - all in the browser. Show4DSTEM goes further: for a
small dataset its virtual-detector math runs in **WebGPU**, so dragging the
aperture recomputes the virtual image in the browser. Show4DSTEM exports make
the dtype explicit: `uint8` is a compact browse payload, while `uint16` keeps the
wider detector-count range at a larger size. See
[Show4DSTEM export recipes](tutorials/show4dstem_export) for when to choose each.

ShowEDS uses the same saved-widget model for synthetic and small cubes in single
mode with exact data. For large native EDS/EELS files, the notebook keeps the
interactive state while the exact count data stays in a data folder. Portable
HTML demos can be exported with count-preserving sum downsampling when
full-resolution data would be too large for public sharing.

See [Installation](install) to get started.

## Getting help

- **Questions or bugs:** open an issue at
  [github.com/electronmicroscopy/quantem.widget/issues](https://github.com/electronmicroscopy/quantem.widget/issues).
- **Maintained by** the Ophus group. Contributions and feedback are welcome via
  pull request or issue.
