# Export and run ShowPtycho

This tutorial exports a ShowPtycho reconstruction as a **browser folder** and
opens it with no Python kernel and no Jupyter. You can run it by granting the
folder to Chrome, or by using the generated local launcher. The exported viewer
opens from exact bright-field detector columns by default, then runs SSB live as
you tune aberrations.

If you just want the interactive widget inside a notebook, see
[ShowPtycho in Jupyter](showptycho.md) instead. This page assumes you already
have a **fitted** `ssb` (solved with `ssb.fit(trials=200,
refinement="nelder-mead")`; an unfitted export uses only the supplied starting
aberrations).

## Export

```python
from quantem.widget import ShowPtycho

# ssb is already fitted: result = ssb.fit(trials=200, refinement="nelder-mead")
w = ShowPtycho(ssb, source_file="scan_master.h5", save_dir="out/")
w.export("out/", title="my sample SSB")
```

This writes a folder:

- `index.html` — the viewer
- `ShowPtycho.command` — double-click launcher for Chrome on macOS
- `source/` — the exact BF-column browser source plus linked HDF5 evidence
- `snapshots/` — calibration, manifest, viewer snapshots, and review metadata

The export persists no expanded float32 images and no complex64 BF reducers.
By default, the browser range-reads `source/bf_columns.u8` or
`source/bf_columns.u16` and does not decode the compressed HDF5 stack on open.
Saved aberration states live in `snapshots/snapshots.json`; reopening the
folder through a folder grant or `quantem showptycho out/` reads them back into
the snapshot strip automatically.

### Export at native detector size

The WebGPU browser export **cannot bin the detector**. If the `ssb` was built
with `det_bin=2` (a 96x96 calibration) but the embedded HDF5 is native 192x192,
the browser decodes 192x192, mismatches the calibration, and shows

```
detector shape mismatch; HDF5 has 192x192, calibration has 96x96
```

with blank panels. Always build and export at native detector size
(`det_bin=1`, the default).

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

### C. CLI (serves and opens, no grant click)

```bash
quantem showptycho out/
```

The command serves the folder over range-capable HTTP and opens it, so the viewer
loads without the manual folder-grant. Use this when double-click + grant is
inconvenient (for example over a remote connection).

## What you can do in the viewer

- Drag **C10 / C12 / phi12 / rotation** — the browser rebuilds the BF-indexed
  `G(k)` reducers and re-runs SSB live; the phase and FFT update in tens of
  milliseconds on a real GPU.
- Toggle the **FFT** panel to watch Bragg spots sharpen as aberrations improve.
- Change colormap, contrast, and the amplitude/complex view.
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

1. The `ssb` was fitted with `fit(trials=200, refinement="nelder-mead")` before export.
2. Native detector (`det_bin=1`) — the browser cannot bin.
3. Export writes a clean root: `index.html`, `ShowPtycho.command`, `source/`,
   and `snapshots/`.
4. Open by double-click + **Open data folder**, or `quantem showptycho out/`.
5. On open, the stats bar shows a non-null `loss` and the phase renders.

## Privacy

Exports embed source HDF5 file basenames in the metadata under `snapshots/` and
in the viewer state. The direct `w.export(...)` example above is therefore for
local or otherwise trusted use.

For a community-facing folder, build it through the canonical CLI and request
redaction explicitly:

```bash
quantem showptycho scan_master.h5 \
  --out shared-review \
  --anonymize \
  --trials 200 \
  --refinement nelder-mead
```

`--anonymize` replaces the local acquisition name and source paths in saved
calibration and optimization provenance while retaining the scientific fit and
software-version record. Inspect the resulting folder before publishing it;
the detector evidence itself is still experimental data and must be yours to
share.
