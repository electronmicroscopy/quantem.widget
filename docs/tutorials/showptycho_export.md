# Export and run ShowPtycho

This tutorial exports a ShowPtycho reconstruction as a **browser folder** and
opens it with no Python kernel and no Jupyter. You can run it by granting the
folder to Chrome, or by using the generated local launcher. The exported viewer
reuses the original QEM acquisition when it exactly matches the session, then
runs SSB live as you tune aberrations. Other sessions export exact bright-field
detector columns automatically.

If you just want the interactive widget inside a notebook, see
[ShowPtycho in Jupyter](showptycho.md) instead. This page assumes you already
have a **fitted** `ssb` (solved with `ssb.find_aberrations()`, the
model-based, tilt-corrected default with 200 trials; an unfitted export uses
only the supplied starting aberrations). The
[SSB fit methods](../api/showptycho.md#ssb-fit-methods) table lists the other two.

## Export

```python
from quantem.widget import ShowPtycho

# ssb is already fitted: result = ssb.find_aberrations()  # tilt-corrected, 200 trials
w = ShowPtycho(ssb, save_dir="out/")   # opened with SSB.open("scan_master.h5", ...), the session names its file
w.export("out/", title="my sample SSB")
```

This writes a folder:

- `index.html` — the viewer
- `ShowPtycho.command` — double-click launcher for Chrome on macOS
- `source/acquisition.qem`: the unchanged whole-scan QEM acquisition, reused
  without re-encoding on CUDA or MPS. If the session cannot reuse its source,
  `source/` instead holds exact BF columns: `bf_columns.qem` on CUDA or
  `bf_columns.u8` / `bf_columns.u16` on MPS.
- `snapshots/` — calibration, manifest, viewer snapshots, and review metadata

The export persists no expanded float32 images, no complex64 BF reducers and no
copy of the raw HDF5 acquisition. The browser range-reads the counts under
`source/` and never decodes the compressed HDF5 stack.
It gathers every selected BF pixel on the GPU, including the original detector
crop offset and axis order. A scan crop or changed counts use the exact BF
companion instead of reusing an incompatible acquisition.
The source is hardlinked when possible, otherwise symlinked or copied. Check
`snapshots/manifest.json` before moving a review: a symlink still depends on the
external source file, while a hardlink or copy is independent.
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

The launcher serves only this folder. Include the actual source file when
copying a symlinked review to another Mac; the same double-click then works.

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

- Drag **C10 / C12 / phi12 / rotation** — the browser reuses the resident
  scan spectra and reconstructs the phase and FFT on its GPU. Interaction
  speed depends on the scan, detector selection, sampling and GPU.
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

1. The `ssb` was fitted with `find_aberrations()` (tilt-corrected, 200 trials) before export.
2. Native detector size: the browser cannot bin.
3. Export writes a clean root: `index.html`, `ShowPtycho.command`, `source/`,
   and `snapshots/`.
4. Open by double-click + **Open data folder**, or serve `out/` over HTTP.
5. On open, the stats bar shows a non-null `loss` and the phase renders.

## Privacy

The folder records the source file as `redacted_local_source`; no acquisition
name or local path is written under `snapshots/` or into the viewer state. The
reused QEM file retains its original metadata and detector measurements. Review
that source before sharing; it must be yours to share.
