# Bounded Show4DSTEM virtual images: native ANS path was never reached

Date: 2026-09-18. Data: 512 x 512 x 192 x 192 encoded QEM (ANS resident,
`runtime-column-rans-spatial-v2`), region (0, 64, 320, 384), BF disk radius 12
at (95.5, 95.5). cuda-env, torch 2.10, RTX PRO 6000, GPU0 shared.

## Question

`Show4DSTEM(scan, scan_region=...)` on an encoded resident felt slow when the
virtual aperture moved, although the resident detector reductions in
quantem.gpu take milliseconds over the whole scan.

## Setup

`prepare(view).masked_sum(mask, output='native')` timed directly, where `view`
is `quantem.widget.show4dstem_bounded._View(scan, region)`, the wrapper the
widget builds for a `.read` source. Compared against
`prepare(scan).masked_sum(numpy_mask)` on the raw resident.

## Numbers

| Path | Per query |
|---|---|
| raw resident, native kernels, full 512 x 512 | 2 to 5 ms |
| `_View` from the stale branch copy (widget path: `prepare` + `masked_sum`) | 150 to 650 ms |
| decoding the 64 x 64 region once (`view.read`) | 180 ms |
| `_View` on `main`, cached session | 0.8 to 1.4 ms |
| `_View` on `main`, widget path (`prepare` each call) | 3 to 6 ms |

Output on `main` is bit-identical to decode-then-reduce (max abs diff 0).

## Cause

`quantem.gpu.detector.backends.bounded.BoundedDetectorCompute` routes to the
native kernels only when the wrapper exposes `_detector_source` (an encoded
`Dataset4dstemGPU`) and `_detector_region`. The stale wrapper exposed `source`
and `region` only, so `_native` stayed `None` and every query decoded the region in
32-column blocks (128 reads) and reduced in torch.

## Resolution

`main` (commit `47591394`) already sets `_detector_source` and
`_detector_region` in `_View.__init__`. The slow numbers came from a kernel
whose `PYTHONPATH` pointed at a stale branch checkout carrying an older copy
of `show4dstem_bounded.py` without those two attributes. Against `main`:
3 ms per query, bit-identical. Contract test added:
`tests/show4dstem/test_bounded_comparison.py::test_bounded_view_exposes_detector_source_for_native_reductions`.

## Not done

- Caching the `DetectorSession` on the wrapper would save another 2 to 3 ms per
  query and keep the incremental-mask state. 3 to 5 ms is below the comm
  round trip.

## Follow-up the same evening: "virtual image stays black"

Two unrelated causes, neither in the widget code:

1. **Orphaned kernel.** The JupyterLab server behind the notebook tab was
   restarted; the page kept showing widgets whose kernel no longer existed
   (status bar: `Disconnected`). The diffraction panel still responded to
   scan-point moves from the browser-side neighbor cache, while every virtual
   image update needs Python, so only the VI looked frozen. Reloading the tab
   and re-running the cells fixes it. A visible "kernel gone" cue on the widget
   would make this obvious.
2. **Headless Linux Chrome with `--use-angle=vulkan --disable-gpu-sandbox` on
   Xvfb cannot display GPU-accelerated 2D canvases at all** (a plain
   `fillRect` on a fresh canvas never shows up; readback returns zeros until
   Chrome demotes the canvas to CPU after repeated `getImageData`). The DP
   panel survives because it has a WebGPU display path; the VI panel is
   Canvas2D. Add `--disable-accelerated-2d-canvas` when using that browser for
   widget screenshots. WebGPU compute is unaffected.

Verified on the Mac (Metal adapter) that the VI canvases are painted (4096 of
4096 opaque pixels, ~250 colours in the compare panels).

The widget's JS bundles under `src/quantem/widget/static/` are gitignored
build artifacts; after switching checkouts run `npm run build` (needs the
Linux `esbuild` binary: `npm install`) or the Python and JS drift apart.
