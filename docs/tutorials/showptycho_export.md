# Export and run ShowPtycho

This tutorial exports a ShowPtycho reconstruction as a **browser folder** and
opens it with no Python kernel and no Jupyter. You can run it by granting the
folder to Chrome, or by using the generated local launcher. The exported viewer
opens from exact bright-field detector columns by default, then runs SSB live as
you tune aberrations.

If you just want the interactive widget inside a notebook, see
[ShowPtycho in Jupyter](showptycho.md) instead. This page assumes you already
have a **fitted** `ssb` (solved with `ssb.find_aberrations(trials=200,
refinement="nelder-mead")`; an unfitted export uses only the supplied starting
aberrations).

## Export

```python
from quantem.widget import ShowPtycho

# ssb is already fitted: result = ssb.find_aberrations(trials=200, refinement="nelder-mead")
w = ShowPtycho(ssb, source_file="scan_master.h5", save_dir="out/")
w.export("out/", title="my sample SSB")
```

This writes a folder:

- `index.html` — the viewer
- `ShowPtycho.command` — double-click launcher for Chrome on macOS
- `source/`: the exact bright-field detector counts: `bf_columns.qem`
  (lossless ANS, written by a CUDA session) or the `bf_columns.u8` /
  `bf_columns.u16` companion an MPS `SSB.open` session already holds
- `snapshots/` — calibration, manifest, viewer snapshots, and review metadata

The export persists no expanded float32 images, no complex64 BF reducers and no
copy of the raw HDF5 acquisition. The browser range-reads the counts under
`source/` and never decodes the compressed HDF5 stack.
Saved aberration states live in `snapshots/snapshots.json`; reopening the
folder through a folder grant or a local HTTP server reads them back into
the snapshot strip automatically.

### Export at native detector size

The WebGPU browser export **cannot bin the detector**. If the `ssb` was built
from a detector-binned array (a 96x96 calibration) while the stored counts are
native 192x192, the browser mismatches the calibration and shows

```
detector shape mismatch; HDF5 has 192x192, calibration has 96x96
```

with blank panels. Always build and export at native detector size, as
`SSB.open` and `quantem.gpu.io.load` do.

## Run it

The exported folder needs `source/` and `snapshots/` present next to
`index.html`. There are three ways to open it.

### A. Double-click `ShowPtycho.command` (macOS, zero setup)

1. Double-click `ShowPtycho.command` at the folder root.
2. A Terminal window starts the bundled range server (stdlib-only, uses the
   Mac's built-in Python) and Chrome opens the viewer already wired to it.
3. Close the Terminal window when done; that stops the server.

The launcher serves only this folder, from wherever it sits — copy the folder
to another Mac and the same double-click works, nothing to install.

### B. Double-click `index.html` (File System Access)

1. Double-click `index.html`.
2. Click **Open data folder** and grant the folder the HTML lives in (browsers
   without the folder picker fall back to a plain file chooser).
3. It renders, starting at the embedded calibration snapshot.

One grant per session. This works fully offline.

### C. Any static server (no grant click)

```bash
cd out/ && python -m http.server 8900
```

Any Range-capable static server works: open `http://localhost:8900/` and the
viewer loads without the manual folder grant. Use this when double-click + grant
is inconvenient (for example over a remote connection, with the port tunnelled).

## What you can do in the viewer

- Drag **C10 / C12 / phi12 / rotation** — the browser rebuilds the BF-indexed
  `G(k)` reducers and re-runs SSB live; the phase and FFT update in tens of
  milliseconds on a real GPU.
- Toggle the **FFT** panel to watch Bragg spots sharpen as aberrations improve.
- Change colormap and contrast.
- **Save** writes the current aberrations and preview JPEG into `snapshots/`
  without prompting for a separate download in the normal local folder workflow.

## Verify WebGPU is on real hardware

Interactive speed requires a real GPU. If a browser falls back to a software
renderer (SwiftShader), the reconstruction still runs but slowly, and any timing
you read is meaningless. On a real adapter the stats bar names the hardware
(for example `nvidia ...` or `apple ...`); a software fallback will not. New GPUs
can be missing from a browser's allow-list — if WebGPU is unexpectedly absent,
launch the browser with GPU blocklisting ignored.

## Checklist

1. The `ssb` was fitted with `find_aberrations(trials=200, refinement="nelder-mead")` before export.
2. Native detector size: the browser cannot bin.
3. Export writes a clean root: `index.html`, `ShowPtycho.command`, `source/`,
   and `snapshots/`.
4. Open by double-click + **Open data folder**, or serve `out/` over HTTP.
5. On open, the stats bar shows a non-null `loss` and the phase renders.

## Privacy

The folder records the source file as `redacted_local_source`; no acquisition
name or local path is written under `snapshots/` or into the viewer state. The
detector evidence itself is still experimental data and must be yours to
share.
