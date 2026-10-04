# Experimental ANS sources in Show4DSTEM

ANS (asymmetric numeral systems) is an opt-in lossless count representation.
The codec, container validation, CUDA/WebGPU decoding, and scientific GPU math
belong to **quantem.gpu**. Show4DSTEM provides the viewer and interaction policy.
Ordinary HDF5 and packed/dense workflows do not select ANS implicitly.

Use matching source revisions of both packages. The source integration
uses backend commit `dd88bc1e`, with CPU codec/export, frontend and ownership
checks. The earlier hardware evidence remains documented separately below.
This is not a PyPI release and does not establish a stable container or private
API compatibility promise. The backend's
`docs/developer/experimental-resident-ans.md` is the canonical feature
description. Read it in the matching `quantem.gpu` checkout.

Encoding creates a new container atomically and refuses to overwrite an existing
file. Original acquisitions remain intact. Source ownership and cancellation
have regression coverage; hardware and performance qualification remains limited
as described below. These safeguards do not make the experimental paths a new
default or a blanket guarantee across all hardware.

## Canonical count-ANS files

```python
from quantem.gpu import io
from quantem.widget.show4dstem_webgpu_export import export_show4dstem_rans_viewer

saved = io.save("acquisition.ans", counts, format="quantem", compression="ans", backend="cpu")
html = export_show4dstem_rans_viewer([saved.path], "ans-viewer")
```

`counts` must have native uint8/uint16 values and dimensions
`(scan_row, scan_col, detector_row, detector_col)`. The encoder is a bounded CPU
reference implementation; it is not qualified for real-time full-acquisition
encoding. Export links the containers and preserves every source count. Open
the generated viewer and grant its linked `.ans` files with **Open count-ANS
files**. For a series, pass an ordered list of paths with matching shape and
dtype. A detector-validity mask can be supplied through `valid_pixels`; it does
not overwrite raw sentinels in the source container.

The same exporter accepts a retained detector-rANS build manifest through an
explicit compatibility adapter. The source112 tANS series used for large
resident experiments is another explicit package adapter. These formats share
implementation ownership, not an interchangeable bitstream. Source112's
internal Huffman checkpoints are not a required encoding choice for other data.

## Interaction and scope

The source112 adapter loads complete acquisitions progressively. Its selected
and averaged point DPs remain in GPU buffers during scan dragging. Statistics,
histogram, and hover values refresh after the cursor pauses; stale values are
withheld while a new pattern is being displayed. Copy reads the visible GPU
canvas. Detector dragging remains live while the pointer is held.

Other backends keep their existing execution paths. CUDA `Show4DSTEM(source)`
requires the existing resident-owner protocol; the canonical-file WebGPU
export above does not imply that any CUDA object returned by `io.load` can be
passed directly to that factory.

This feature does **not** guarantee 120 displayed frames/s, all66 loading under
20 seconds, general WebGPU bitpacked-input support, or automatic multi-GPU
placement. The latest small source112 validation used three acquisitions on an
identified NVIDIA Blackwell adapter and exact native references. See the
backend guide for the measured limits; rAF and render-submission counters are
not screen-presentation measurements.

## Building the shared backend

Install the matching `quantem.gpu` and widget checkouts in your development
environment, then build the widget against that exact backend:

```bash
python -m pip install -e /path/to/quantem.gpu -e /path/to/quantem.widget
cd /path/to/quantem.widget
npm ci
QUANTEM_GPU_SRC=/path/to/quantem.gpu/src PYTHON=python npm run build
```

`scripts/sync-gpu-webgpu.mjs` recreates the ignored generated engine tree from
`quantem.gpu`. Edit the package sources, not that generated tree. A widget built
against an unrelated backend checkout is not the validated experimental pair.
Regenerate exported HTML after rebuilding so an old viewer cannot retain old
shader code.
