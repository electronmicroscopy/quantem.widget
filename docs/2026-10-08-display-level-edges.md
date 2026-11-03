# WebGPU and CPU colormap levels on level edges (2026-10-08)

## Question

Plot2D on Apple Metal could paint a value exactly on a colormap level edge in
the neighbouring colour. The WGSL `display_normalize` divided `(value - low) /
span` with the GPU's `/`, which WGSL allows to be 2.5 ULP off, while the CPU
path (`displayNormalize`, `Math.fround` at every step) is correctly rounded.
One ULP below an edge is the previous level: `fround(2 / 6) * 255` is level 85,
one ULP lower is level 84. Does this happen on the NVIDIA adapter too, and does
an exact division fix it?

## Setup

`tests/browser/test_display_level_edges.py`: the shader source of
`js/display/colormaps.ts` compiled on the real adapter (a Linux workstation, Chrome 147,
NVIDIA RTX PRO 6000 Blackwell through Vulkan, adapter `nvidia / blackwell`),
`display_normalize` evaluated for every level edge of 8 windows ([0, 6], [0, 1],
[-1, 1], [0, 255], [3, 7], [1e-3, 2e-3], [-5000, 65535], [0.1, 0.7]) and one
float32 step either side: 6144 values. Reference: NumPy float32
`(value - low) / (high - low)` clipped to [0, 1], the CPU path's arithmetic.

## Results

| shader | float32 results differing from the CPU path | LUT levels differing | levels differing at the exact edge values (2048) |
|---|---|---|---|
| `672dc0b` (plain `/`) | 1434 | 394 | 374 |
| `display_divide` (integer long division) | 0 (one subnormal, flushed to 0 by the GPU, level 0 on both) | 0 | 0 |

So the bug was not Metal-only: on NVIDIA the WebGPU path painted 18 % of exact
edge values one level away from the Canvas2D path. The colormap, histogram and
volume shaders all share `display_normalize`, so the fix applies to every widget.

`display_divide` splits both operands into their 24-bit significands and an
exponent, divides the significands with `rounded_quotient` (binary long
division with round-to-nearest-even, the routine the temporal-average shader
already used for integer counts) and applies the exponent difference with
`ldexp`. It costs a 23 to 24 step integer loop per pixel and no measured frame
rate: driver perf mode, base `672dc0b` against this branch, back to back on GPU 0
(WebGPU on, `nvidia / blackwell`):

| widget, size | interaction | page fps p50, base / after | frame ms p95, base / after | input-to-pixels ms p95, base / after |
|---|---|---|---|---|
| Plot2D 36 x 24 | wheel zoom | 60 / 60 | 16.7 / 16.8 | 48 / 49 |
| Plot2D 2048 x 2048 | wheel zoom | 60 / 60 | 16.8 / 16.7 | 48 / 49 |
| Plot2D 2048 x 2048 | drag pan | 60 / 60 | 16.7 / 16.8 | 33 / 31 |
| ShowPtycho gold_512 SSB | wheel zoom | 60 / 60 | 16.8 / 16.8 | 47 / 47 |
| Show3D 100 x 1024 x 1024 | playback (60 fps set) | 60 / 60 | 16.8 / 16.7 | - |
| Show3D 100 x 1024 x 1024 | slider scrub | 60 / 60 | 16.8 / 16.8 | 28 / 25 |

## Cost of the exact division per pixel

Question: does the exact display division slow WebGPU drawing?

Setup: one compute pass of display_normalize over a 4096 x 4096 float32 frame
(16.8 M pixels, values 0..65535, window 100.5..60000.25 from a uniform), the
three shader versions interleaved, 12 rounds of 20 dispatches each, wall time
per dispatch (median). Headless Chrome on the two Macs; headed Chrome on the
NVIDIA Vulkan driver on the Linux workstation, whose GPUs were at 97-100 %
with other jobs during the run, so the NVIDIA times are upper bounds.

| GPU (adapter) | plain `/` (before this fix) | bit-loop exact (1a18a8c) | loop-free exact, one call (42ff5e8) |
|---|---|---|---|
| NVIDIA RTX PRO 6000 (nvidia / blackwell) | 0.18 ms | 0.48 ms | 0.30 ms |
| Apple M5 Max (apple / metal-3) | 0.28 ms | 0.97 ms | 0.44 ms |
| Radeon Pro 560, Intel Mac (amd / gcn-4) | 2.0 ms | 8.9 ms | 4.1 ms |

Conclusion: the exactness costs 0.1-0.2 ms per 4096 x 4096 frame on the
NVIDIA and Apple GPUs, about 1 % of a 16.7 ms frame. On the 2017 Intel Mac
GPU, the bit-loop version took half the frame budget at this size. The
loop-free version halves that cost on all three GPUs. Frames of display size
(up to a few megapixels) cost proportionally less.

## Windows FXC: the wide-window branch did not build

Question: why did Plot2D, Show2D and Show3DSlices paint black with WebGPU on in
the Windows 11 ARM VM (Parallels, Edge) after this fix, when they passed there
before it?

Setup: compile-only probe pages (one `createComputePipelineAsync` per shader
variant, nothing dispatched) in Edge in the VM. The Parallels adapter is D3D
feature level 11.1, so Chrome's D3D12 backend compiles WGSL through FXC, not
DXC; Linux, Apple and the Intel Mac never take that path. Every variant builds
on SwiftShader, so the WGSL is valid.

| variant | FXC result |
|---|---|
| COLORMAP_SHADER before this fix (plain `/`) | builds |
| COLORMAP_SHADER with display_divide at its 7 call sites (bit-loop quotient) | `E_FAIL` |
| the same, quotient without a loop (three u32 divisions) | `E_FAIL` |
| the same, display_divide reduced to `ldexp(a / b, e)` (no quotient at all) | `E_FAIL` |
| the same, wide-window branch removed (one display_divide call left) | builds |
| rounded_quotient alone, display_divide alone, display_normalize with constant bounds | build |

Conclusion: the integer quotient was never the trigger. FXC fails on the
wide-window branch of display_normalize (a window whose `high - low` overflows
float32), which inlined six more display_divide calls on runtime bounds. That
branch now halves such a window first and shares the single division: halving
is exact for normal numbers, and a window that wide has low < 0 < high both far
above the smallest normal, so the quotient is the correctly rounded
`(value - low) / (high - low)` of the halved values on both paths. Low and high
are halved in their exponent bits: with `* 0.5` the NVIDIA compiler turned
`high * 0.5 - low * 0.5` into `(high - low) * 0.5`, which overflows again
(2290 of 2304 wide-window edge values wrong in the browser test). Three wide
windows are now part of the level-edge tests, CPU replay and NVIDIA. The
colormap engine also builds COLORMAP_SHADER once with
`createComputePipelineAsync` before use and returns no engine (CPU drawing, one
console warning) if that rejects, since a pipeline that fails to build paints
black without an exception; Show2D's engine factory goes through the same check.

Rejected: removing the quotient loop (built on the rest of the probe, still
`E_FAIL`); a constant loop bound (still `E_FAIL`); `exp2` for `ldexp` (not the
cause); `high * 0.5 - low * 0.5` (factored back into an overflow on NVIDIA).

## Rejected

- A host-computed reciprocal `255 / span` in the uniform: exact for ranges the
  host knows, but several shaders take their range from a GPU reduction
  (`range_in`, `ranges[panel]`), where the reciprocal would again be a GPU
  division; it would also change the CPU path's rounding.
- Snapping with an epsilon: moves the disagreement to values one epsilon from
  an edge instead of removing it.
- WGSL `fma` to correct a hardware quotient: the WGSL spec lets `fma` be
  computed unfused, so the residual is not exact.

## Volume slice zoom

The Show3DSlices volume shaders area-average every source pixel an output pixel
covers, `fullW / canvasW / zoom` per axis, with no bound as the zoom approached
zero. The shader now clamps the zoom at `VOLUME_SLICE_MIN_ZOOM = 0.5`, the
widget's own minimum, so the average stays exact and bounded (a strided 16 x 16
sample cap was tried first and dropped: it reduced the data silently).
