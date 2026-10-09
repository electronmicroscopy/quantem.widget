# Experimental ANS sources in Show4DSTEM

ANS (asymmetric numeral systems) is the lossless count representation that
`quantem.gpu.io.load` uses for 4D-STEM acquisitions on CUDA and MPS. The codec,
container validation, CUDA/Metal/WebGPU decoding, and scientific GPU math
belong to **quantem.gpu**. Show4DSTEM provides the viewer and interaction
policy.

Use matching source revisions of both packages. This is not a PyPI release and
does not establish a stable container or private API compatibility promise.
The backend's `docs/api/qem-python.md` and `docs/api/qem-codecs.md` describe the
QEM file and its codecs; read them in the matching `quantem.gpu` checkout.

## Live viewers

An acquisition from `io.load` is already ANS encoded on the GPU, and
`Show4DSTEM` reads it directly:

```python
from quantem.gpu.io import load
from quantem.widget import Show4DSTEM

loaded = load("acquisition_master.h5")
Show4DSTEM(loaded)
```

See [Show4DSTEM](../api/show4dstem.md#encoded-acquisitions) for the live
behavior. These views need a kernel; `quantem show4dstem --backend webgpu
--html` is the offline path, reading the HDF5 family in the browser.

## Building the shared backend

Install the matching `quantem.gpu` and widget checkouts in your development
environment, then build the widget against that exact backend:

```bash
python -m pip install -e /path/to/quantem.gpu -e /path/to/quantem.widget
cd /path/to/quantem.widget
npm ci
QUANTEM_GPU_SRC=/path/to/quantem.gpu/src PYTHON=python npm run build
```

`scripts/sync-gpu-webgpu.mjs` recreates the ignored generated engine tree
`js/.generated/engine/` from the science entries of `quantem.gpu`'s
`webgpu/sources.json` manifest (`detector/`, `dpc/`, `formats/`, `io/` and
`ssb/`); only the `show4dstem` and `showptycho` bundles use it. Display kernels
are widget source under `js/display/`. Edit the package sources, not that
generated tree. A widget built against an
unrelated backend checkout is not the validated experimental pair. Regenerate
exported HTML after rebuilding so an old viewer cannot retain old shader code.
