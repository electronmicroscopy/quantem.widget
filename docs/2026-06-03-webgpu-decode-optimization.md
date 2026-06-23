# WebGPU 4D-STEM load optimization: 16.8s -> 1.3-4.4s, and the uint32 parity bug (2026-06-03)

## Question

The standalone WebGPU browser app (`widget/web/`) loaded a full Arina dataset far slower
than Python's `quantem.widget.load` (~2s). How fast can a native `.h5` 4D-STEM dataset go
from a picked folder to a rendered virtual image, all in-browser, and where are the floors?

## Setup

- App: `widget/web/` (quantem.live Browse GUI + the `js/engine/` WGSL engine).
- Real data, multiple users + dtypes: gold04 (3 GB, **uint16**), gold06 (6.6 GB, **uint32**),
  karen SiN (3.9 GB, uint32), wmill dggg (5.5 GB), steph lamella (801 MB), george gold_10
  (743 MB). Driven over CDP via `DOM.setFileInputFiles` (the picker's exact `File.arrayBuffer`
  disk path) on headed Chrome + NVIDIA Vulkan (Blackwell, NOT SwiftShader - asserted).
- Parity reference: h5py mean DP (clip uint8, integer sum, pixel_mask bad px zeroed).

## The critical bug: uint32 datasets decoded 16x wrong

`h5reader` detected only uint8/uint16; uint32 detector data (`<u4`, common for high dynamic
range) fell through to uint16. The decode then read 16 bit-planes from a 32-plane bitshuffle
-> every value ~16x off. The IMAGE looked right (relative contrast preserved) so it passed
vision checks; only a numerical mean-DP-vs-h5py check caught it. **Lesson: vision is not
parity. Always check a numerical reference on real data of every dtype.**

Fix: detect `<u1`/`<u2`/`<u4` -> 8/16/32 planes (`srcBytes` 1/2/4 -> `blockElems`,
`nBlocksPerFrame`, `blockMeta` all correct per dtype), threaded `srcDtype` store -> kernel
(templated `__NBITS__` in the fused kernel). Verified bit-exact after:
- gold04 uint16: browser 147010.54 == h5py 147010.55
- gold06 uint32: browser 2529764.38 == h5py 2529764.50 (diff = float32 summation rounding)

## Results (per-stage, measured)

| Stage | Before | After | How |
|---|---|---|---|
| File read | 3.0s (2.3 GB/s, main thread) | **0.7s (10 GB/s)** | **8 Web Workers** each call File.arrayBuffer |
| jsfive parse | 2.8s | **0.25s** | custom DataView HDF5 v1 B-tree reader (no per-node alloc) |
| GPU upload | mappedAtCreation 4s | staging pool + Float64 wide copy | reuse MAP_WRITE buffers (kills 1.7s alloc/zero); f64 copy = 20 GB/s |
| GPU kernel | "1.9s" (mismeasured) | **459ms** (timestamp) | fused shared-mem decode (no interBuf); the 1.9s was upload-DMA |
| bitshuffle | 16/32-plane transpose | OR-fold | uint8 output: low-8 transpose + OR high planes (2x uint16 / 4x uint32 less) |

End-to-end (picker disk path, parity-verified):

| Dataset | Size | Load |
|---|---|---|
| george gold_10 | 743 MB | 1.3s |
| steph lamella | 801 MB | 1.5s |
| karen SiN (uint32) | 3.9 GB | 3.3s |
| wmill dggg | 5.5 GB | 3.7s |
| gold06 (uint32) | 6.6 GB | 4.3s |

**Sub-GB hits the 1-2s Python sweet spot.** Big datasets ~1s/GB.

## The breakthrough: File.arrayBuffer is main-thread-bound, not bandwidth-bound

Main-thread `File.arrayBuffer` (sequential or `Promise.all`) caps at ~2.3 GB/s. Measured
alternatives:
- `Blob.slice()` 8x parallel ranges: **2.0 GB/s** (slower - more overhead)
- `File.stream()`: 2.1 GB/s (slower)
- OPFS `createSyncAccessHandle` (worker): 3.5 GB/s (and needs an OPFS copy first)
- **8 Web Workers each `File.arrayBuffer` + transfer buffer back: 10.1 GB/s** <- 4.4x

So the 2.3 GB/s was the single main thread (file delivery + ArrayBuffer allocation), not the
disk or an IPC pipe. A worker pool parallelizes it to near disk speed. `readWorker.ts` reads
+ runs the fast btree parse, transfers `{buffer, blockMeta}` back zero-copy.

## Floors (what's left)

- File read: ~10 GB/s with workers (near Python's C disk read).
- GPU kernel: 459ms (uint16), decode-bound on the **serial LZ4** (the OR-fold proved the
  bitshuffle is not the bottleneck). Memory-traffic floor is ~11ms (16 GB / 1.47 TB/s), so
  ~40x compute headroom remains in the serial decode.
- WebGPU upload: spec mandates a staging copy on discrete GPUs (gpuweb#2388); cudaMemcpy's
  64 GB/s is unreachable, realistic ~5-10 GB/s.

For big datasets the wall is now the **decode** (serial LZ4 + upload of the GBs). The path to
a ~100ms kernel is the silx warp-cooperative decode (parallel literal/match copies), but
WGSL's u32-only shared memory forces `atomic<u32>` for the byte-strided parallel writes - a
complex, parity-sensitive rewrite, not yet done.

## meanDP reduce: 460 ms -> 38 ms (one thread/pixel -> frame-parallel + batched submit)

After the full gold04 (uint16) load decoded in ~2.3 s, the post-decode mean-DP reduce was
taking 190-460 ms - it should be <100 ms (it sums 9.66 GB of uint8 into 36864 floats; the
1.47 TB/s memory floor is ~7 ms). Three compounding bugs, all measured on real gold04
(262144 frames, 192x192), bit-exact vs h5py throughout (meanDP sum 147010.5400 == 147010.5401):

| fix | reduce ms | what |
|---|---|---|
| baseline | 190-460 | one thread per detector pixel (~37K threads), each looping over ALL 262144 frames |
| frame-parallel | 175 | 2D grid: gid.y splits frames into 64 strided slices, each thread does one atomicAdd into the integer dp. ~64x more threads -> saturates the GPU (was ~12% occupancy, 88% idle) |
| batched submit | 100-122 | the decode makes ONE GPU buffer per data file (27 files -> 27 chunks); the reduce was 27 separate submits. Record all 27 passes into ONE encoder + ONE submit -> GPU 153 ms -> 55 ms |
| folded readback | **38-66** | the 147 KB dp readback was a SECOND submit+fence (~30-50 ms of pure mapAsync round-trip latency). Fold the dp->readback copy into the same encoder: one submit, one sync |

Lesson: a memory-bound GPU reduction that runs 50x over its bandwidth floor is almost always
**under-parallelized** (too few threads to hide latency) and/or **over-submitted** (one
dispatch+sync per data chunk). Parallelize across the reduced-over axis, batch all chunks into
one submit, and never pay a second fence for a tiny readback. Integer atomicAdd keeps it
bit-exact (addition is associative), so parity is free.

## Strategy D (round-based parallel LZ4): parity-exact but a perf regression (rejected)

Built + parity-verified the round-based dataflow decode (`FUSED_D_WGSL`, `verifyFusedD`,
toggle `globalThis.__BSLZ4_PARALLEL`). Idea: all 64 lanes parse the block's token stream
redundantly, then in repeated rounds each lane writes only the output bytes it owns whose
LZ4 source is already final (DONE bitmap), self-terminating when a round resolves nothing.
The LZ4 RLE identity `out[di+k] = out[di-off + (k mod off)]` flattens the period-`off` run so
each match byte depends only on the `off` bytes BEFORE the match (not on a depth-`ml` chain),
capping dataflow depth at ~60 on real Arina.

**Parity: bit-exact.** `verifyFusedD` on gold04 (uint16) and gold06 (uint32), full 10k-frame
files: `nDiff=0, maxDiff=0` vs the serial fused kernel, meanDP identical.

**Perf: 3-5x SLOWER, so kept default-OFF.** Per 10k-frame file (timestamp/wall, Blackwell):

| dtype | serial fused | Strategy D |
|---|---|---|
| uint16 (gold04) | **118 ms** | 348-356 ms |
| uint32 (gold06) | **131-138 ms** | 681-697 ms |

**Why it can't win (the real lesson):** round-based dataflow does **O(maxDepth x blockBytes)**
work - every round re-scans every match byte - while the serial thread-0 decode is
**O(blockBytes)**. With maxDepth ~60 that is ~60x more total work; spread over 64 lanes it is
roughly break-even on raw ops, and the per-round atomics + ~60 `workgroupBarrier`s push it
3-5x past serial. The serial fused kernel (~118 ms/file) is already at the ~100 ms goal and
is the right default. The ONLY structurally-faster LZ4 is **warp-cooperative copy inside a
SINGLE forward pass** (lanes split each match's byte copy as the serial cursor advances),
which keeps O(blockBytes) total work - a different algorithm, not a tuning of Strategy D.

**WGSL bring-up bugs caught (all silent: a compile error drops the dispatch, leaving Dawn's
zero-initialized output buffer - looks like "kernel produced nothing", not an error):**
- `(a >> b & c)` - mixing `>>` and `&` needs parens: `((a >> b) & c)`.
- `workgroupBarrier()` whose reachability depends on an `atomicLoad` (early `break` on a
  convergence flag) is rejected ("must be uniform control flow"). Fix: FIXED-count loop, gate
  only the WORK (a non-uniform branch with NO barrier inside is legal), never the barrier.
- `active` is a WGSL reserved keyword (use `working`).
- **A backtick inside a `//` comment in a backtick-template-literal WGSL string silently closes
  the string** - and vite's `node_modules/.vite` transform cache served the STALE module across
  rebuilds, so source edits appeared to have no effect. `rm -rf node_modules/.vite` before
  rebuilding when a symlinked source (`web/src/engine -> ../../js/engine`) edit is ignored.

## Rejected / dead ends

- **Worker PARSE with buffer transferred IN**: regressed to 7.6s - transferring 7.5 GB to
  workers + reading all upfront cost more than the parallel parse saved. The win is workers
  reading the File THEMSELVES (only the result transfers back).
- **All-28-reads upfront** (no grouping): 7.4s - memory pressure, lost the decode pipeline.
- **mappedAtCreation per file**: the alloc+zero of host-visible memory is ~3.9 GB/s (1.7s for
  6.6 GB) and scales with bytes; a reused staging pool amortizes it.
- **OR-fold high-plane STORAGE skip** (4096-byte shared): risks parity (LZ4 matches may cross
  the low/high plane boundary); only the high-plane bitshuffle is OR-folded, never the decode.

## Files

`js/engine/h5reader.ts` (fast btree + uint32 detect + zero-copy), `js/engine/bslz4.ts` (fused
kernel + `__NBITS__` + staging pool + wide copy + OR-fold), `js/engine/compute.ts`
(`fromGpuChunks`), `web/src/local/{store,readWorker}.ts` (worker read pool + srcDtype + LRU),
`web/src/App.tsx` (File/handle source for workers).
