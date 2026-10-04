# quantem.gpu migration status

Reusable numerical math and GPU kernels are being moved from `quantem.widget`
to `quantem.gpu`. This is an incremental ownership migration, not a rewrite of
the widgets and not a blanket claim that every widget path is complete.

## Ownership boundary

`quantem.gpu` owns mathematics or kernels only when at least two real widget
call sites use the same scientific semantics. `quantem.widget` continues to own
React components, viewer state, interaction policy, caches, workflow
orchestration, exports, and widget-specific scientific controls and defaults.

The browser sources copied into `js/.generated/engine/` are generated build
inputs. Edit their canonical source in `quantem.gpu`, then run:

```bash
QUANTEM_GPU_SRC=/path/to/quantem.gpu/src npm run sync:webgpu
```

Thin modules in `quantem.widget/js/` may re-export canonical functions so
existing imports remain stable during the migration. They must not grow a
second implementation.

## Browser execution rule

For migrated scientific operations, the production browser path requires a
hardware WebGPU adapter. Missing WebGPU, a software adapter, shader failure, or
device loss is reported as unavailable; it must not silently run the same
numerical operation in JavaScript. Scalar TypeScript, NumPy, and SciPy versions
remain only as independent parity references and test oracles.

This rule applies to the migrated FFT, histogram/colorization, display filter,
frequency filter, ROI crop/mask, line profile, FFT-peak, detector reduction,
CoM-magnitude, and iCoM paths. It does not claim that every legacy or
widget-specific browser calculation has already migrated.

## Current coverage

| Area | Shared consumers | Status |
| --- | --- | --- |
| finite extrema, histograms, normalization, signed-log transforms, and LUT colorization | Show2D, Show3D, Show3DSlices, Show4DSTEM, ShowEDS, ShowPtycho | migrated candidate |
| FFT, shift, magnitude, peak localization, quality metrics, and reciprocal-coordinate math | Show2D, Show3D, Show3DSlices, Show4DSTEM, ShowPtycho, as applicable | migrated candidate; widget-specific FFT display policies remain local |
| Gaussian/Anscombe display filtering and bin/resample stages | Show2D, Show3D | migrated candidate; active production path requires WebGPU |
| Smooth radial frequency filtering | Show2D, Show3D | migrated candidate; padded odd/nonsquare FFT domain is preserved through the inverse |
| ROI masked crops and bilinear line profiles | Show2D, Show3D, Show4DSTEM, Browse | migrated candidate; active production path requires WebGPU |
| FFT local-peak refinement | Show2D, Show3D, Show3DSlices, Show4DSTEM, ShowPtycho | migrated candidate; active production path requires WebGPU |
| disk and annulus detector masks | Show4DSTEM detector compute and local browser compute | migrated candidate |
| masked DPC magnitude and iCoM integration | Show4DSTEM and local Browse | migrated candidate; GPU reduction and post-processing |
| block binning, affine transforms, correlation, and subpixel alignment | varies | under audit; not migrated where semantics differ |
| iterative ptychography reconstruction | none in this migration | explicitly out of scope |

“Migrated candidate” means canonical ownership and call sites have moved. It is
not a release claim until the acceptance gates below pass on the final commits.

## Acceptance gates

A primitive is complete only when all applicable gates pass:

1. Freeze a NumPy or other independent reference result before changing the
   implementation, including zeros, constants, negative values, extreme finite
   values, NaN/Inf, odd and nonsquare shapes, and dtype boundaries.
2. Compare every backend that actually implements the primitive: CPU, CUDA,
   Metal, and WebGPU. An unavailable backend is recorded as unavailable, not as
   passing.
3. Prove at least two real widget consumers and retain focused call-site tests.
4. Build all affected widget bundles and verify that generated engine sources
   and package artifacts contain the canonical implementation.
5. Drive affected standalone HTML or notebook widgets after the final code
   change: validate every marked scientific output for nonzero pixels, dynamic
   range, and multiple colors or tones; exercise the relevant controls, inspect
   browser errors, and save screenshots. Hardware signoff must pass
   `--require-hardware-webgpu`. Use
   `scripts/widget_local_signoff.sh --quick --browser` for the repeatable HTML
   matrix and [Agent signoff](widget-agent-signoff) for human-driven review.
6. Record limitations and semantic rejections. Reduced precision, resolution,
   or evidence cannot be used to claim parity or performance.

## Known limitations

- Auto-contrast percentages, sampling, linked-panel behavior, and display
  defaults intentionally remain widget policy.
- ShowPtycho's padded/cropped FFT coordinates are not yet represented by the
  shared reciprocal-coordinate helper.
- ShowPtycho's optional complex-HSV display does not yet have a WebGPU shader;
  it now reports the unavailable view instead of silently CPU-rendering it.
- ShowEDS sparse stream map, preview, and spectrum paths require WebGPU. Its
  dense prefix-plane representation remains separate because it is an I/O
  representation with different semantics, not the same reusable sparse
  kernel.
- Show3D registration, quantized kymograph sampling, export encoders, and
  several widget-specific display policies remain local. They are not evidence
  that the shared migrated operations have a CPU fallback.
- Some similarly named operations have different scientific definitions—for
  example positive-only versus signed logarithms, detector annuli versus FFT
  crop circles, and different alignment algorithms—and are intentionally not
  unified.
- The ordinary headless standalone matrix does not expose hardware WebGPU and
  therefore cannot sign off these migrated production paths. Hardware WebGPU
  numerical parity and the full-widget macOS interaction matrix must remain
  separate release gates.
- Browser WebGPU parity is verified over the supported finite test range. The
  full positive/negative `float32` extrema remain a documented browser-driver
  limitation until hardware WebGPU produces stable histogram evidence there.

Update this page when ownership, consumers, or evidence changes. Do not replace
the per-run test and signoff reports with this status summary.

## Candidate evidence (2026-08-17)

The tested local branches are named `webgpu-required-widget-compute` in both
repositories. No branch was pushed and no PR or release was created.

| Repository | Tested code head | Parent | Starting main |
| --- | --- | --- | --- |
| quantem.gpu | `5cd285250911974c738e9c911bd00a170873bf45` | `94c59e22754e1a49779e9270b36308d8051652c2` | `bfdfa7e0ebefd2b8a86655dbd162368dbc2fa6c3` |
| quantem.widget | `de3dc3989561c767c36547298df2b6cc8ee9cd22` | `a830ea6d5bc22be3ad588227d936afcd5cf7505a` | `9211cac4dcd6fc7b87157246bbffda4bf725f173` |

The final candidate was installed from tracked Git archives into a fresh,
isolated environment on Phil. This caught missing candidate-version, pandas,
ipykernel, macOS Chrome-launch, and test-fixture dependencies before signoff.

### Numerical and build gates

- Linux quantem.gpu: `293 passed, 55 skipped`; real CUDA display parity:
  `4 passed`.
- Phil/macOS quantem.gpu: `262 passed, 77 skipped` with the MPS extra.
- Phil/macOS quantem.widget: `1134 passed, 33 skipped, 47 warnings`; no test
  failures. Warnings are recorded deprecations/runtime guidance, not numerical
  mismatches.
- TypeScript/Vitest: `26` files and `154 passed`; TypeScript typecheck passed.
- Headed Chrome hardware-WebGPU parity: `7 passed` on adapter
  `apple metal-3`, with `softwareAdapter=false`. This covers NumPy-referenced
  histogram/LUT/FFT evidence plus detector masked sum, center of mass, DPC,
  odd/nonsquare filters, ROI crop/mask, line profile, and FFT peak paths.
- Swift/Metal: `27 passed` (`16` Metal4DSTEM and `11` MetalDisplay tests) on
  arm64e macOS.
- `npm run build` built all widget bundles; `web/npm run build` built the Browse
  application. A fresh `npm ci` and `npm audit --json` reported zero
  vulnerabilities.

### End-to-end browser gates

The fresh standalone matrix generated 19 exports and passed `38/38` across
desktop and 390x844 touch layouts in headed Phil Chrome. This includes Show1D,
five Show2D layouts, six Show3D layouts, Show3DSlices, Show4DSTEM single and
compare, a ShowPtycho WebGPU folder, ShowEDS and ShowDiffraction.
Every page acquired `apple / metal-3`; all 38 adapter records reported
`is_fallback_adapter=false` and `software=false`.

The browser captured and tested every visible element marked
`data-quantem-scientific-output`, rather than accepting the largest canvas as a
proxy for the page. All `126/126` output regions passed: at least eight colors
or tones, channel span at least 8, nonblack-pixel fraction at least 0.5%, and
mean luminance greater than 1. The observed worst cases were 303 colors or
tones, channel span 64, nonblack fraction 90.1%, and mean luminance 4.162.
Grayscale scientific data are valid when they have tonal range; chromatic
saturation is not required. Every page was interactive, above the 20
browser-rAF FPS gate, and free of page, console-error, and HTTP failures. The
lowest measured browser-rAF rate was 119.89. Browser-rAF is an event-loop health
signal, not interaction latency.

- Show2D native stress: 40 exact 4096x4096 uint8 panels; linked contrast
  changed six visible panels, linked zoom/pan changed six, keyboard
  selection/hide/restore passed, and every sampled canvas was nonblank. The
  final linked pan-to-paint measurement was 8.7 ms; a separate single wheel
  batch was 149.9 ms and is not hidden by the rAF number.
- Show3D real-data stress: 24 real 2048x2048 drift frames passed desktop, wide,
  and narrow viewports with `apple metal-3`, nonblank composited-pixel checks,
  histogram Auto-to-manual contrast, slider and keyboard scrub, playback, FFT
  toggle, and zoom/pan. Final interaction latency was 1.3-1.8 ms in the widget
  debug evidence.
- Show3DSlices: a real-derived odd/nonsquare 17x511x769 volume; slice playback,
  oblique-angle, FFT, and contrast interactions changed visible pixels.
- Show4DSTEM: actual single-tilt and seven-tilt command launchers; all 28
  seven-tilt HDF5 family requests completed, and detector-radius drag changed
  the virtual-image canvas while the pointer remained held.
- ShowEDS: a 96x128x512 spectrum image with deterministic Au-like peaks; band,
  ROI, log-scale, Save Band, and Save ROI paths passed, and both map and
  spectrum reported `webgpu` as their producing backend.
- ShowPtycho: non-iterative SSB review only, using a real 128x128 crop from a
  512x512 HDF5 source prepared on MPS and reopened through the actual WebGPU
  folder launcher. Exported BF columns matched probed raw source coordinates;
  BF count and C10 changed nonblank phase/FFT evidence, amplitude was
  GPU-native, and complex HSV reported unavailable instead of using a CPU
  fallback.

The stress pass corrected three false-negative test-driver assumptions: MUI
range inputs use an implicit slider role, visible-panel hashes must follow the
viewport, and WebGPU canvas pixels must be checked from the browser's
composited output. Show3D's existing WebGPU-to-retained-2D zoom fallback is
accepted only when the destination remains readable and nonblank; the dangerous
2D-to-WebGPU swap remains a failing continuity condition. This presentation
fallback is not a CPU implementation of the migrated numerical kernels.

The hardware pass found and fixed two product blockers before this evidence was
recorded: native uint8 Show2D galleries could paint black at 4096x4096, and a
cropped ShowPtycho export wrote the full source scan plane while declaring the
crop shape. The final matrix had no application console errors; one existing
ShowDiffraction passive-listener warning remains recorded.

These are candidate-branch results, not a public release signoff. CUDA and
Metal are recorded only for primitives with real implementations; a WebGPU-only
display primitive is not presented as CUDA/Metal parity. Iterative ptychography
remains explicitly outside this migration.
