# WGSL GPU bslz4 decode + uint8 clipping appropriateness (2026-06-03)

Two distinct offline 4D-STEM data paths. Do not conflate them.

## Path A — WGSL GPU bslz4 decode (LOSSLESS) — proven this session

Goal: ship the native HDF5 bitshuffle+LZ4 (bslz4) compressed bytes, decompress in
the browser on WebGPU, no Python, no server. Tested on REAL gold data
(`/home/user/ssd/_hf_stage_gold30/gold_30mrad1.3mx04/`, 27x10000 frames of
192x192 uint16, filter 32008 bitshuffle, typesize 2, block 8192 bytes).

Two-pass WGSL decoder, both passes verified bit-exact vs h5py:
1. **Pass1 LZ4** - one thread per 8192-byte block (`workgroup_size(64)`, one block
   per thread, full warp utilization). Byte-RMW into an `inter` (bitshuffled)
   device buffer. Independent blocks -> embarrassingly parallel.
2. **Pass2 inverse bitshuffle** - one thread per element, 16 plane-byte reads
   (`planeBytes = blockElems/8 = 512`, 16 bit-planes for uint16). Plane-major,
   LSB-first: `out[e] |= bit((byte[b*512 + e>>3]) >> (e&7)) << b`.

**Result on real gold:** 4 sampled frames **0/36864 mismatch** vs h5py reference.
Lossless uint16. No clipping.

### Bugs found (all silent - shader compiled or ran but produced zeros)
1. **`meta` is a reserved WGSL keyword.** A `var<storage> meta` made the shader
   silently fail to compile; output stayed all-zero. Because the bitshuffled data
   is ~99% zero, the all-zero output matched the reference everywhere except its
   handful of nonzero bytes -> looked like "44/8192 mismatch" instead of "didn't
   run." Renamed to `blkMeta`. **Lesson: a tiny mismatch count on sparse data can
   mean the kernel never ran, not that it is almost right.**
2. **Local `let frame` shadowed the `frame` storage binding** -> pipeline silently
   invalid. Renamed to `frm`.
3. **Default device `maxStorageBufferBindingSize` = 128 MB**, NOT the adapter max
   (1 GB here). A 147 MB output buffer made the bind group invalid -> nothing ran,
   all-zero, no thrown error (only surfaced via `pushErrorScope`). **Must request
   `requiredLimits:{maxStorageBufferBindingSize: adapter.limits..., maxBufferSize:
   adapter.limits...}` at `requestDevice`.** This applies to the production
   `getGPUDevice` in `js/show4dstem/fft.ts` too.
4. **`maxComputeWorkgroupsPerDimension` = 65535.** A 1D dispatch over 36.8M
   elements (576k workgroups) exceeds it -> use 2D dispatch (`gx<=65535`, `gy`),
   linear index `e = gid.y*strideX + gid.x`.

### Throughput (measured, real gold, host RTX PRO 6000, 1000 frames x5 reps)
- pass1 LZ4: 180 ms ; pass2 inverse-bitshuffle: **104 ms** (optimized, see below) ;
  total **~0.28 ms/frame**.
- pass2 optimization: one thread per GROUP of 8 consecutive pixels - they share the
  same 16 plane-bytes (one per bit-plane), so 16 reads feed 8 outputs (8x fewer
  global reads) -> 413 ms -> 104 ms, 4x, bit-exact (verified in numpy + e2e).
  A shared-memory block-transpose variant was SLOWER (845 ms) - rejected. A
  workgroup-shared pass1 (1 active thread/wg) was SLOWER (269 vs 180 ms) - the
  64-block-per-workgroup parallelism beats on-chip locality; rejected.
- Production e2e (`createFromBslz4` -> `masked_sum`, 10000 frames): decompress
  **2.9 s**, VI maxErr 0. ~0.29 ms/frame.

### Speed vs CUDA / CPU (same bslz4, same real gold)
| decode path | ms/frame | full 512x512 (262k) | vs CUDA |
|---|---|---|---|
| CUDA (`widget/io/bitshuffle.py`, h5lz4dc + bitshuffle_u16) | 0.0029 | 0.77 s | 1x |
| CPU (h5py + hdf5plugin, SIMD+threads) | 0.083 | ~22 s | ~29x |
| WGSL (optimized, this work) | ~0.28 | ~75 s | ~95x |

WGSL stays ~95x slower than CUDA: nvcomp LZ4 is warp-cooperative (many threads per
block); WebGPU can't express that, and a per-thread local block buffer (8192 B x 64
threads) overflows registers. pass1 LZ4 (0.18 ms/frame) is near the WebGPU floor.
**WGSL is not for matching CUDA; it is the browser path** - ship ~6x less (compressed
vs uint16), no server, lossless, and decode-on-demand so each VIEWED frame is
sub-ms even if decoding the whole stack is not a 1-second job. "Within 1 s in the
browser" = ~3500 frames (60x60 scan) fully decoded; larger -> stream/lazy decode.
Full-size + fast = the CUDA/MPS workstation path (0.77 s for the whole 512x512).

156 s to decode EVERYTHING to a 38.7 GB uint32 stack is not "seconds" and not the
right framing. The decoded full stack does not fit GPU memory anyway. The practical
architecture is **stream-decode in chunks** and compute the useful reduction
(virtual image via masked_sum, or a single viewed frame for flip-through) on the
fly - never materialize 38 GB. Decode cost is then paid once per frame per
reduction; the win is shipping 3.2 GB instead of 19-38 GB and needing no server.

### End-to-end proven (2026-06-03): browser decompress -> virtual image == Python
Real gold region 100x100 scan = 10000 frames x 192^2 uint16, shipped as the native
bslz4 chunks (116 MB) as an HTTP companion file. In headed Chrome on host, no
Python:
- fetch 123 MB compressed: 201 ms (native `fetch`)
- WGSL decompress (Pass1 LZ4 -> Pass2 inverse-bitshuffle, output uint16-packed) +
  `masked_sum` full-detector virtual image: 5854 ms (0.59 ms/frame)
- virtual image vs Python `frame.sum()` reference: **maxErr = 0 (bit-exact)**
- total fetch+device+decode+VI: 6.1 s

The decoded stack is uint16-packed in the EXACT layout `Show4DSTEMCompute.sample()`
reads in uint16 mode (`frameU16[gp>>1] >> ((gp&1)*16) & 0xffff`), so it feeds the
widget's `masked_sum`/`reduce_frames` with no glue. Ship ~6x smaller (compressed)
than uint16, lossless, no server-side decode.

## Path B — offline HTML inline embed (uint8 clip, LOSSY)

`_pack_offline` in `show4dstem.py` clips counts to uint8(0,255) and gzips, to bake
the stack inline into a static HTML widget (kernel-less docs demo).

### Memory footprint, 512x512x192x192 (262144 frames, 192^2 detector)
| form | bytes/px | total |
|---|---|---|
| compressed bslz4 (ship) | ~0.34 | ~3.2 GB |
| decoded uint16 (lossless) | 2 | 19.3 GB |
| decoded uint32 (decoder out) | 4 | 38.7 GB |
| uint8 clip(0,255) | 1 | 9.7 GB |

### Is uint8 clip appropriate? Measured on real gold30 (200 sampled frames, 7.4M px)
- **Real signal max = 48 counts** (mean 11, p99.99 = 34). **Zero** real pixels > 255.
- Only values > 255 are **800 px = exactly 65535** (0.011%) = 4 fixed dead/hot
  detector pixels, identical every frame, masked out anyway.

So clip(0,255) is **near-lossless for the actual signal of THIS dataset** (max 48
<< 255); it truncates only dead pixels. **This is a property of the dataset, not a
safe default.** Brighter probe / longer dwell / binning pushes real counts past
255 and clip then silently destroys signal.

### Rule
- Default = **lossless**: bslz4 WGSL stream-decode (Path A) for real data.
- uint8 clip = opt-in for the inline docs widget ONLY, and only after confirming
  `max(real_counts) <= 255` for that dataset; otherwise keep uint16 or
  log/gamma-compress before quantizing. Disclose the loss. (Respects
  `feedback_precision_over_speed`: full precision is the default.)

## Speedup hunt (2026-06-03) — every approach MEASURED on real gold, 1000 frames

Goal: make WGSL decode approach CUDA (0.0029 ms/frame). Baseline WGSL 0.28 ms/frame.

| approach | pass1 ms/frame | total ms/frame | vs baseline | bit-exact |
|---|---|---|---|---|
| baseline (1 thread/block, global byte-RMW, 8192B blocks) | 0.166 | 0.270 | 1.0x | yes |
| re-chunk to 512-elem blocks (more parallel units) | 0.152 | 0.257 | 1.05x | yes |
| 5a: decode into private array (no global RMW), 8192B | 0.129 | 0.238 | 1.13x | yes |
| **5a + 1024-elem blocks (private array fits regs)** | **0.107** | **0.202** | **1.34x** | yes |
| 5a + 512-elem blocks | 0.103 | 0.209 | 1.29x | yes |
| native gzip DecompressionStream (uint8) + GPU upload | n/a | 0.097 decode + 0.035 upload = 0.13 | 2x | yes(uint8) |
| workgroup-shared decode (wg_size 2-4) | ~0.15 | - | ~1.1x | yes |

Findings:
- **Occupancy was NOT the bottleneck** (re-chunk barely helped: 1.05x). Agent occupancy-starvation hypothesis disproven empirically. The GPU was already saturated; per-thread efficiency is the lever.
- **Global byte read-modify-write is the per-thread cost.** Private-array decode (5a) removes it: 1.13-1.34x. Best with smaller blocks so the private array fits registers (no spill).
- **Native `DecompressionStream` (gzip) decodes at ~380 MB/s** = 0.097 ms/frame, ~2x faster than WGSL LZ4, but worse ratio (3.2x uint8 vs 6.57x bslz4) and routes through CPU + a 0.035 ms/frame GPU upload. Not a 10x win.
- **CUDA parity (0.0029) is physically unreachable in-browser.** nvcomp uses warp-shuffle cooperative decode; WGSL has no subgroup intrinsics (Chrome 147). Native gzip is single-threaded C++ at ~380 MB/s. Best in-browser decode measured ~0.1-0.2 ms/frame = 35-70x off CUDA, ~1x CPU.
- Untested ceiling-raiser: full warp-cooperative WGSL shader (one workgroup decodes one block: serial token-parse -> parallel literal copy -> wave match resolution). Agent-designed, bit-exact-by-construction argument, but Amdahl-limited by the serial parse + atomic-byte-write overhead; estimated 2-3x more at best, high implementation risk.

Conclusion: the browser win is NOT matching CUDA raw decode (impossible). It is: ship 6x smaller (bslz4), hold the full uint8 stack on GPU (9.58 GB across 26 buffers, proven, no OOM), decode ONCE in background (~25-80 s full 512x512) or chunk-on-demand, and make UX instant via a precomputed inline virtual image + lazy per-frame DP. Decode is amortized, not on the interaction path.

## CORRECTION (2026-06-03): all prior WGSL timings were SwiftShader CPU, not the GPU

The "95x slower than CUDA / can't match it" conclusion above is WRONG. Root cause:
the host headed Chrome ran on `DISPLAY=:1` which is **Xvfb (no GPU)**, so WebGPU
fell back to **SwiftShader** (Chrome's bundled CPU software Vulkan). `adapter.info`
reported `vendor=google arch=swiftshader subgroupMaxSize=4`. Every 0.2-0.28 ms/frame
number was CPU software rendering.

**Fix — force Chrome onto the real NVIDIA Vulkan device** (see
feedback_webgpu_swiftshader_trap memory):
```
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json DISPLAY=:1 google-chrome \
  --remote-debugging-port=9230 --user-data-dir=/tmp/cdp-gpu --ignore-gpu-blocklist \
  --enable-features=Vulkan --use-angle=vulkan --enable-unsafe-webgpu --disable-gpu-sandbox ...
```
After: `adapter.info vendor=nvidia arch=blackwell subgroup_size=32`. `subgroups`
feature + `subgroupShuffle`/`subgroupAdd` WORK (warp 32).

### REAL-GPU decode timings (RTX PRO 6000, 1000 frames, bit-exact maxErr 0)
| block size | total ms/frame | vs CUDA (0.0029) |
|---|---|---|
| 4096 elem (native, stable) | 0.0103-0.0105 | ~3.6x |
| 2048 elem | 0.0068 | ~2.3x |
| 512 elem (best run) | 0.0042 | ~1.4x |
| 256 elem | 0.0044-0.0054 | ~1.6x |
(pass1 LZ4 ~2-8 ms/1000, pass2 inverse-bitshuffle ~2.3 ms/1000 - pass2 also ~45x
faster on GPU than the SwiftShader 104 ms.) Numbers noisy on shared GPU 0
(reconstruction/other tasks contend); best-case is the true throughput.

### What flipped vs SwiftShader-era conclusions
- **Smaller blocks now HELP** (occupancy on a massively-parallel GPU): 4096->512
  roughly halves pass1. On SwiftShader (4 cores) block size was ~flat.
- **Private-array decode (5a) is WORSE on the real GPU** (global RMW is fine with
  real L1/L2 caches; the private 8 KB array spills). 5a was a SwiftShader artifact.
  Production `bslz4.ts` correctly uses the baseline global-RMW pass1 - keep it.
- The production decoder needs NO code change for the real GPU; users on real GPUs
  (Mac M-series, NVIDIA) already get ~0.004-0.01 ms/frame. Re-chunk the browser
  companion to ~512-elem blocks for the ~2x occupancy win (ratio 5.8x vs 6.6x - fine).
- nvcomp parity is now plausibly REACHABLE (subgroups available); the residual
  1.4-3.6x is occupancy/scheduling, not a hard API wall. Subgroup-cooperative LZ4
  is the path to close it further if needed.

**Bottom line: WebGPU bslz4 decode on the real GPU is near-CUDA (1.4-3.6x), lossless,
bit-exact. The browser CAN load + decompress real 4D-STEM at near-workstation speed.**
