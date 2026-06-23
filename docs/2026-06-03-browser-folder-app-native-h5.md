# Native Arina .h5 decoded in the browser, no Python, no server (2026-06-03)

## Question

Can a single HTML file open a folder of native Arina `.h5` 4D-STEM datasets and decode
every dataset entirely on the GPU (WGSL), bit-exact vs CUDA, fast enough to be
interactive - so a colleague opens the HTML, points at a folder, and browses?

## Setup

- Data: `/home/user/ssd/_hf_stage_gold30/gold_30mrad1.3mx04/` - one `*_master.h5` + 27
  `*_data_NNNNNN.h5` slabs, 512x512x192x192 uint16, 4 dead px (pixel_mask).
- Reader: `js/engine/h5reader.ts` - jsfive (pure-JS HDF5, no wasm) pulls raw bitshuffle+LZ4
  chunks straight out of the file into a `Bslz4Spec`; the engine WGSL decoder eats them.
- Compute: `js/engine/{device,bslz4,compute}.ts` - the SAME single engine source the
  Show4DSTEM widget uses.
- App: `js/app/{index,folder}.ts` + `app.html` - folder picker, dataset gallery, viewer
  with a draggable detector on the mean diffraction driving a live virtual image.
- GPU: NVIDIA RTX PRO 6000 Blackwell, Chrome 147 forced onto NVIDIA Vulkan (NOT
  SwiftShader - confirmed `arch=blackwell`, WebGL renderer string, header shows
  "GPU: nvidia blackwell").

## Results

| Check | Result |
|---|---|
| jsfive raw chunk vs h5py `read_direct_chunk((0,0,0))` | bit-exact, 12586 B |
| `h5reader` spec offset tiling, all frames | 10000/10000 tile exactly, 0 mismatch |
| Browser GPU mean DP vs h5py reference (uint8, bad-px zeroed) | maxAbs = 0.000e+0 |
| Single slab (10000 frames, 123 MB) | load 263ms / parse 158ms / decode 232ms / reduce 16ms |
| Full 512x512 (27 files, 9.66 GB u8 on GPU) | loaded in 10.9 s, real gold-NP virtual image |
| Detector drag BF -> ADF | virtual image updates live (hash changed) |

Parity is by composition: jsfive chunk == h5py chunk (proven), and engine WGSL decode ==
CUDA decode (prior parity tests, uint8+uint16). The spec offset-tiling check confirms
`blockMeta` points at exactly the LZ4 streams h5py's filter reads.

## Conclusion

The vision works. A native Arina `.h5` folder decodes fully client-side on the GPU,
bit-exact, fast enough to browse (sub-second per 10k-frame slab, ~11 s for a full 9.66 GB
512x512). Single compute source. No Python, no server. The mean diffraction shows the
bright-field disk; dragging the detector gives live virtual-image contrast.

## What this does NOT cover yet (next)

- Laptop budget: 9.66 GB needs a workstation GPU. Auto-bin (detector and/or scan) on load
  to fit a laptop's VRAM is not implemented - currently full-res only.
- Single shareable HTML: now built by the Vite `web/` app with
  `npm run build:offline`.
- Full Show4DSTEM viewer (BF/ABF/ADF presets, ROI modes, FFT, per-probe DP scrub): the
  app's viewer is currently minimal (mean DP + draggable detector + virtual image).
- Real File System Access picker drive on a Mac or Linux desktop requires a user
  gesture. It has now been driven with native mouse events against visible
  Chrome/GNOME on host.

## Update: GUI is now the quantem.live Browse page (not a hand-rolled viewer)

The hand-rolled `js/app/` viewer was replaced by a duplicate of the quantem.live web
dashboard's Browse page, per the user: "reuse the browser GUI from quantem.live, make a
duplicate just for HTML WebGPU based. Don't create your own GUI."

New standalone Vite React app at `widget/web/`:
- Copied the Browse-page transitive closure (17 files) verbatim from
  `quantem.live/web/src`: `pages/browse/{Browse,FileTree,MetaRail,Viewer}.tsx` + theme,
  ShortcutRegistry, hooks, and the viewer utils (gpu-colormap, webgpu-fft, colormaps,
  stats, scalebar, lineProfile, fft). No source changes to the GUI components.
- The ONLY rewrite is the data layer. `web/src/local/store.ts` scans a locally-picked
  folder (File System Access / webkitdirectory) into the same `Session[]`/`MasterFile`
  tree the GUI expects, and reimplements every `fetch*` in `pages/browse/types.ts`
  against the shared `js/engine` WGSL engine (symlinked to `web/src/engine`): `maskedSum`
  for virtual images (BF=disk, ADF/DF=annulus, free-form shapes for the detector drag),
  `frameAt` for per-probe CBED, `reduceFrames` for the summed-ROI DP, plus an auto-fit BF
  disk from the mean DP. Server EventSources are neutralized by a no-op `EventSource` stub
  in `main.tsx`; GPU-cache/preload calls become local no-ops (the engine has its own LRU).
- App shell `web/src/App.tsx`: reuses the quantem.live MUI theme verbatim, gates on a
  folder picker, then renders `<Browse/>`.

Verified on host (NVIDIA Blackwell, Chrome 147 + NVIDIA Vulkan): full gold04
512x512x192x192 (9.66 GB) loads in ~10 s, the Browse GUI renders the gold-NP BF image +
the CBED disk, and switching BF -> ADF drives the GPU annulus live (image inverts to
bright NPs). Zero console errors. CDP-driven via a `window.__loadServed(base, names)` hook
(the OS folder picker can't be driven headlessly). Build: `npm run build` in `web/`,
692 KB / 220 KB gzip.

## Update: CoM/iCoM (DPC) + single-file folder deliverable

- Added a `maskedCoM` WGSL kernel to `js/engine/compute.ts` (one thread per scan position,
  intensity-weighted detector centroid over the aperture mask, badPx excluded). The store
  wires CoMx/CoMy/CoMmag directly, and iCoM via a CPU complex-FFT Poisson solve
  (`phi_hat = (-i kx Gx - i ky Gy)/(kx^2+ky^2)`, DC=0) in `web/src/local/store.ts` (radix-2
  FFT, falls back to CoM magnitude for non-power-of-two scans). SSB stays unsupported
  (full ptychography). Verified on host: CoM mag/row/col/iCoM each render distinct fields
  on gold04 512^2; iCoM shows the expected smooth DPC phase.
- Scan-crosshair -> CBED confirmed (clicking the real-space canvas updates the DP via
  `frameAt`).
- Single-file deliverable: `cd web && npm run build:offline` writes a runnable
  `dist/index.html`. The build may also emit a worker asset because the normal
  served app uses worker reads, but offline mode compiles the app to use
  main-thread file reads, so `index.html` is the artifact to share. Drop it in a
  folder, double-click (file://), click "Choose folder", pick the Arina folder,
  and Browse loads. NOTE: file:// CANNOT fetch sibling files (CORS), so the
  folder-pick click is mandatory - it is the browser security boundary that
  grants access to local bytes.

## Folder selection and watch semantics

- A standalone HTML file cannot read sibling `.h5` files just because it lives
  in the same folder. Browsers deliberately block that: local file access
  requires a user-granted File System Access handle or a file-input selection.
  The supported UX is therefore: open the HTML, click **Choose folder**, select
  the folder containing `*_master.h5` and `*_data_*.h5`, then browse. No Python
  or Jupyter kernel is used after that grant.
- Native File System Access (`showDirectoryPicker`) recursively scans the chosen
  parent folder and polls it every 1 s while folder watch is enabled. New
  `_master.h5` + sidecar `_data_*.h5` files appear without reselecting the
  folder. Refreshes are serialized so a slow scan cannot overlap the next tick.
- `webkitdirectory` fallback is snapshot-based. It cannot live-watch a folder,
  but reselecting the same folder replaces that root instead of creating
  `folder-2`, so newly copied files appear after reselect.
- A `_master.h5` is required. Orphan `_data_*.h5` files are ignored until the
  master appears. A master with missing sidecars is listed as not loadable, and
  corrupt/incomplete masters are skipped until a later scan sees a valid file.
- The scan phase reads only masters for metadata/readiness. Sidecar data bytes
  are read only when the user opens a dataset.

## VRAM and detector binning

- Single-master WebGPU Browse now uses the same VRAM-aware auto-bin planner as
  5D/multi-dataset Browse. A full 512x512x192x192 master is not assumed to fit
  no-bin in browser VRAM; the app picks the smallest detector bin that fits the
  conservative browser-GPU budget for the current `uint8`/`uint16` setting.
- On the host visual smoke, the full Sample 512x512x192x192 master chose
  `det_bin=2`, decoded in 4.37 s, reduced BF in 18 ms, rendered nonblank BF/DP,
  and measured ~61 fps via `requestAnimationFrame`.
- Native picker release-gate retest on host (visible Chrome/GNOME, real mouse
  events) selected `/home/user/AAAA_QWIDGET_NATIVE`, accepted Chrome's folder
  permission prompt, scanned two full Sample masters, showed `watching`,
  `2 files · 1 sessions`, Stack viewer `1/2`, and rendered nonblank BF/CBED.
  The warmed `det_bin=2,uint8` decode path completed in 1.77 s with BF reduce
  in 20 ms.

## Rejected / notes

- jsfive cannot enumerate the master's `entry/data` external-link group (btree-v2 link
  decode hits an assert). Not needed: the app globs `*_data_*.h5` by filename and reads
  scan size (`ntrigger`) + hot pixels (`pixel_mask`) from the master directly. pixel_mask
  nonzero count (4) matched the per-file saturation heuristic exactly.
- Current code imports the public `jsfive` module directly; the stale deep-import
  Vite aliases were removed.
