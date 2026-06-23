# Performance

How to stay fast on large datasets, written for the two platforms we serve:
**macOS on Apple M-chips (MPS)** and **Linux with NVIDIA CUDA (HPC)**.

## Where the time goes

- **Show2D, Show3D, Show3DSlices** render in your browser. The backend (CUDA /
  MPS / CPU) barely matters - the data is packed once and the colormap runs on
  the browser GPU. The lever here is **transport**, not the compute backend.
- **Show4DSTEM** computes a fresh virtual image every time you move the detector.
  This is the one widget where the GPU backend matters - with one exception
  below: a small dataset can compute entirely in the browser via WebGPU.

## 4D-STEM in the browser (`backend="web"` / `offline=True`)

For a small dataset, pass `backend="web"` (or the compatibility alias
`offline=True`). The stack ships to the browser once and the virtual-detector
reductions run in **WebGPU** - no Python kernel in the loop. Browser transport is
uint8-clipped for the detector stack, so it is exact for detector counts
`<=255`; hot/dead pixels are carried separately. Use the CUDA/MPS kernel path
when full uint16 count fidelity is required.

This is what makes the Show4DSTEM example in these docs interactive, and it powers
kernel-less shared HTML and Colab demos. Measured ~0.7 ms per virtual-image
recompute at 100x100 scan x 96x96 detector - faster than a kernel round-trip,
because there is no comm latency. Large datasets keep the kernel path.

## Large 4D-STEM datasets (Show4DSTEM)

- **Bin the detector on load: `load(path, det_bin=2)`** (or `4`). This is the
  single biggest win - it cuts memory and speeds first paint, with little visual
  cost for picking detectors.
- **Keep data on the GPU as a PyTorch tensor.** Work stays in the native dtype
  (integer detector counts sum exactly); we avoid casting to float32 or going
  through NumPy on the hot path, which would double memory.
- **Release memory when done: `widget.free()`** - frees the GPU buffers held by
  the widget (a plain `del` does not, because traitlets keeps a reference).

## Backends

| Platform | Backend | Show4DSTEM path |
|---|---|---|
| Linux + NVIDIA | **CUDA** | PyTorch viewer on GPU; integer detector reductions via CuPy |
| macOS Apple M-chip | **MPS** | a dedicated **raw-Metal** kernel - `torch.mps` is too slow for the masked-sum, so the masked detector sum is a hand-written Metal shader (reads the BF disk in place, ~8 fps on a full dataset) |
| anything else | **CPU** | the same PyTorch viewer, just slower - a fallback, not a target |

The other three widgets render in the browser regardless of backend, so they
feel the same on Mac and Linux.

## Transport: browser vs VS Code

For the browser-rendered widgets, interactivity is limited by how fast each
update reaches the page. A real **browser tab** (`http://localhost:<port>/lab`)
is the fastest. **VS Code's Jupyter** adds an extension-host IPC hop (~hundreds
of ms per MB on a big frame), so a slider can feel choppy there even when the
Python side is fast. If a widget feels slow in VS Code, open it in a browser tab.
