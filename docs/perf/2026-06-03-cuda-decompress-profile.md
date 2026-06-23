# CUDA bitshuffle+LZ4 decode profile (gold 30mrad, 512×512×192×192 uint16)

Date: 2026-06-03
Branch: `widget-show3d-show4dstem-kernels`
GPU: RTX PRO 6000 Blackwell Max-Q (GPU 1, `CUDA_VISIBLE_DEVICES=1`), CC 12.0, 188 SMs
Data: `/home/user/ssd/_hf_stage_gold30/gold_30mrad1.3mx04/gold_30mrad1.3mx04_master.h5`
(262144 frames, 192×192 uint16, frame_bytes = 73728, 9 full 8 KB blocks/frame, no tail; 3.15 GB compressed → 19.3 GB output).
Profiling only — **no kernel code was modified.**

Kernels under test (from `io/bitshuffle.py`, launched in `io/hdf5.py::_decompress_prepared`):
- `h5lz4dc_batched` — LZ4 decompress. Grid `((max_blocks+1)//2, 1, n_frames)`, block `(32, 2, 1)`. One warp per 8 KB block, 2 blocks/CTA.
- `shuf_8192_16_batched` — inverse bitshuffle for uint16. Grid `(n_8kb, 16, n_frames)`, block `(256,1,1)`.

---

## 1. Baseline

### Public `load()` wall time (median of 5, warm page cache)
| metric | value |
|---|---|
| `load(MASTER, verbose=False)` median | **15204 ms** |
| effective (19.3 GB / time) | **1.3 GB/s** |

**This 15.2 s is NOT the decode kernels.** It is dominated by a post-decode advisory.
See section 5 — the decode pipeline itself is ~0.36 s.

### Decode pipeline only (`_load_master_optimized`, returns GPU array)
| metric | value |
|---|---|
| `_load_master_optimized` (warm) | **351–369 ms** |
| effective (19.3 GB / time) | **~53.6 GB/s** |
| `get_metadata` | 5–7 ms |

This 360 ms includes H2D upload of the 3.15 GB compressed buffer (double-buffered
async), scratch allocs, pixel-mask scatter, and the result copy — not just the two kernels.

### Per-kernel timing (CUDA events, isolated launches, output GB/s)
Measured by replaying each kernel on a held GPU input (harness `/tmp/kern_harness.py`):

| frames | LZ4 ms | LZ4 GB/s | bitshuffle ms | bitshuffle GB/s | LZ4 / shuf split |
|---|---|---|---|---|---|
| 1000 | 0.174 | 424 | 0.171 | 432 | 50 / 50 |
| 4000 | 0.627 | 471 | 0.672 | 439 | 48 / 52 |
| 8000 | 1.208 | 488 | 1.340 | 440 | 47 / 53 |
| 20000 | 2.959 | 498 | 3.405 | 433 | 46 / 54 |

**Kernel-only floor:** the two kernels together do the full 19.3 GB output in **~83 ms (≈231 GB/s output)**.
The 360 ms full-pipeline number is ~4× that, so the remaining ~280 ms is H2D + alloc + mask + result write,
not the decode math. The two kernels are roughly **balanced** (LZ4 ~46–50 %, bitshuffle ~50–54 %),
so a 2–3× win requires speeding up **both**, not just one.

---

## 2. ncu metrics (per kernel, at 3000 frames — enough to fill 188 SMs)

`ncu --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis
--section WarpStateStats --section LaunchStats --section ComputeWorkloadAnalysis`,
`--launch-skip 1 --launch-count 1` (steady-state launch, warmup skipped).

### `h5lz4dc_batched` (LZ4 decompress)
| metric | value | note |
|---|---|---|
| Duration | 957 µs | |
| Memory throughput (Max BW) | **52.3 %** | |
| Compute (SM) throughput | **65.3 %** | highest of the two pipelines |
| memory throughput | 25.6 % | not memory-bound |
| L2 cache throughput | 52.3 % | |
| L1/TEX hit rate | 80.1 % | |
| Achieved occupancy | 72.9 % (theoretical 100 %) | |
| Registers / thread | 40 | block limit = 24 blocks (warp-limited, not reg-limited) |
| Warp cycles / issued instr | 12.5 | |
| Avg active threads / warp | **27.8 / 32** | ~13 % lanes idle → branch/loop divergence |
| Executed IPC | 2.79 | |
| Dominant pipeline | **ALU 45.2 %** | integer/logic — LZ4 token decode + byte copies |
| Tail effect | **25 %** est. speedup | grid 15000 blocks = 3 full waves + 1465-block partial wave |

LZ4 is **compute/ALU-bound with a meaningful tail-effect** and modest lane divergence.
Neither memory nor occupancy is the wall. The work is the inherently serial LZ4 token
walk done by 1 warp per 8 KB block (`coopCopy*` spreads only the copies across 32 lanes;
the token/LSIC parsing in `decompressStream` is effectively scalar with `__syncwarp` between steps).

### `shuf_8192_16_batched` (inverse bitshuffle, uint16)
| metric | value | note |
|---|---|---|
| Duration | 997 µs | |
| Memory throughput (Max BW) | **83.2 %** | near roofline |
| Compute (SM) throughput | 83.2 % | |
| L1/TEX cache throughput | **84.1 %** | **the actual ceiling** |
| memory throughput | 41.8 % | |
| L2 cache throughput | 20.4 % | |
| L1/TEX hit rate | 25.0 % | |
| Achieved occupancy | 87.7 % (theoretical 100 %) | |
| Registers / thread | 40 | block limit = **6 blocks** (register- AND warp-limited at 256-thread block) |
| Warp cycles / issued instr | 25.1 | |
| Avg active threads / warp | 32 / 32 | no divergence |
| Dominant warp stall | **CTA barrier — 12.4 of 25.1 cycles (49 %)** | `__syncthreads()` after the 32-byte smem load |
| Dominant pipeline | ALU 21.7 % | the 16-iteration bit-test loop |

Bitshuffle is **L1/shared-memory-throughput bound (84 %)**, not memory-bound (42 %).
Half of every warp's issue gap is spent **waiting at the single `__syncthreads()` barrier**:
only the first 32 of 256 threads do the coalesced smem load, so 224 threads/CTA idle at the
barrier while 32 load — a classic load-imbalance-into-barrier stall. The per-element work
(read 16 smem bytes, test 16 bits, OR into a uint16) is light; the kernel is gated by the
load phase + barrier, not the bit math.

---

## 3. Bottleneck diagnosis

Using the project rule of thumb (mem BW <50 % → coalescing/stride; compute <50 % → occupancy/regs; both low → stalls):

- **`shuf_8192_16_batched` is the harder floor.** Memory throughput is **83 %** and L1/TEX
  is **84 %** — it is essentially at the L1/shared roofline. The remaining headroom is the
  **barrier stall (49 % of issue cycles)** caused by only 32/256 threads doing the smem load
  while the rest wait. memory is only 42 %, so the kernel is **not** moving bytes efficiently
  relative to what the detector frame actually needs; it re-reads the same 8 KB block through
  L1 16× (once per output group) — L1/TEX hit rate is only 25 %, meaning the 8 KB block is not
  being reused across the 16 groups, it is re-fetched. The structural problem: **each 8 KB
  block is loaded 16 separate times (once per `blockIdx.y` group), and inside each CTA only
  32 of 256 threads load.**

- **`h5lz4dc_batched` is ALU/compute-bound with a tail.** 65 % compute, 52 % memory,
  73 % occupancy, ALU the top pipeline at 45 %, plus a **25 % tail-effect** from the partial
  wave and ~13 % lane divergence. The serial LZ4 token walk (one warp per block) limits it;
  the cooperative 32-lane copies are fine but the parse is scalar.

**Single biggest bottleneck:** the two kernels are co-equal in wall time (~50/50), so neither
alone is "the" floor — but **bitshuffle is closest to its hardware roofline (84 % L1)** and is
losing ~half its cycles to a barrier driven by a 32-of-256-thread load imbalance, while LZ4
has the most *recoverable* slack (tail effect 25 % + ALU divergence + scalar parse). Of the
two, **LZ4 has the larger easy headroom; bitshuffle needs a structural rewrite of its smem
load to gain.**

---

## 4. Recommended optimization approaches (ranked, independently testable)

### A. Bitshuffle: load each 8 KB block ONCE per CTA, all 256 threads cooperating (highest value)
- **Hypothesis:** the 84 % L1/TEX + 49 % barrier stall + 25 % L1 hit rate all stem from the
  same root: grid is `(n_8kb, 16, n_frames)` so each 8 KB block is launched as **16 separate
  CTAs** that each re-read overlapping bytes, and inside a CTA only `tid < 32` loads the 512-byte
  smem tile while 224 threads stall at `__syncthreads()`.
- **Kernel:** `shuf_8192_16_batched`.
- **Mechanism:** collapse `blockIdx.y` (the 16 groups) into one CTA that loads the whole 8 KB
  block into smem **once** with all 256 threads (coalesced, 32 B/thread), one `__syncthreads()`,
  then each thread reconstructs its element(s) from the resident block. Eliminates 15/16 of the
  redundant global loads, raises L1 hit rate, and removes the load-imbalance barrier stall (all
  256 threads participate in the load → no idle-at-barrier). Expected to move memory toward the
  copy-bound roofline (we measured 734 GB/s plain copy bandwidth vs current 42 % memory).
- **Plausible gain:** 1.5–2× on the bitshuffle half (it is the more roofline-limited kernel, so
  the gain caps near the L1→memory rebalance).
- **Risk:** medium. Index math must stay bit-exact; needs the frozen-baseline parity check
  against current output. Watch smem footprint (8 KB block + uint16 out staging vs 6-block
  occupancy limiter).

### B. LZ4: eliminate the tail-effect + reduce ALU divergence (highest easy headroom)
- **Hypothesis:** ncu reports a **25 % tail-effect** (3 full waves + a 1465-block partial wave)
  and avg active threads/warp 27.8/32 (~13 % lane idle). Both are recoverable without changing
  the algorithm.
- **Kernel:** `h5lz4dc_batched`.
- **Mechanism:** (1) **grid sizing** — the tail comes from `((max_blocks+1)//2, 1, n_frames)`
  producing a grid whose block count isn't a clean multiple of the wave size; persistent-block
  / grid-stride over (frame × block) so every SM stays busy to the end removes up to 25 %.
  (2) **Pack 4 warps/CTA instead of 2** (block `(32,4,1)`) to amortize the `__shared__ buffer`
  and raise eligible warps per scheduler, hiding the scalar-parse latency. (3) the cooperative
  copies already vectorize across 32 lanes; the divergence is in the LSIC/token branches — minor.
- **Plausible gain:** 1.2–1.4× on the LZ4 half from tail removal + better warp packing.
- **Risk:** low-medium. Pure launch-config + CTA-packing change; the decompress logic is
  untouched, so parity risk is low. Must re-verify smem-per-CTA stays under the occupancy limit.

### C. Fuse LZ4 + bitshuffle into one kernel (eliminate the lz4_scratch round-trip)
- **Hypothesis:** today LZ4 writes the full 19.3 GB unshuffled stream to `lz4_scratch` in memory,
  then bitshuffle reads all 19.3 GB back and writes 19.3 GB to `shuf_scratch` — i.e. one extra
  full **write + read of 19.3 GB through memory** (~38 GB of traffic) purely to hand off between
  the two kernels. At 734 GB/s plain copy that handoff is ~50 ms of pure memory traffic, a large
  share of the ~83 ms kernel floor.
- **Kernel:** new fused `h5lz4dc + shuf_8192_16`.
- **Mechanism:** have the LZ4 kernel decompress an 8 KB block into **smem** (it already streams
  through a smem buffer), then run the bit-transpose on that resident block and write only the
  final reconstructed uint16 to memory — never materializing the intermediate unshuffled bytes in
  global memory. Removes one full memory write + read of the dataset.
- **Plausible gain:** the most likely path to the full **2–3×** target, because it removes traffic
  neither kernel can avoid while they stay separate. Stacks on top of A and B.
- **Risk:** high. Requires reconciling LZ4's 1-warp-per-8 KB-block layout with bitshuffle's
  256-thread-per-block layout in a single launch, sizing smem for both, and keeping output
  bit-exact. Biggest engineering effort; do A + B first to de-risk, then fuse.

**Suggested order:** B (cheap, low risk, ~1.3×) → A (structural but contained, ~1.5–2× on its
half) → C (fusion, the lever that reaches 2–3× overall). A+B alone plausibly reach ~1.5–1.8×
combined; C is what closes to 2–3×.

---

## 5. Side finding (NOT a kernel issue, but it dominates `load()` wall time)

The 15.2 s public-`load()` time is **entirely** the post-decode browse-dtype advisory in
`io/hdf5.py::_browse_dtype_advise_and_cast` (line ~1915–1916):

```python
mx = int(data.max())                       # 7.0 s   over 19.3 GB
pct255 = float((data > 255).mean()) * 100  # 7.8 s   over 19.3 GB
```

Measured directly on the decoded GPU array:
- `data.max()` → **6983 ms** (≈ 2.8 GB/s)
- `(data > 255).mean()` → **7852 ms** (≈ 2.5 GB/s)

Root cause is **not** the decode kernels and **not** array layout (the array is C-contiguous).
A plain `cp.copy` of 10 GB runs at **734 GB/s**, but `cp.sum`/`cp.max` of the same buffer run at
**~3 GB/s** on this Blackwell card (CC 12.0) — CuPy's reduction launch is ~200–250× below roofline
here, independent of the decode path (reproduced on a fresh `cp.zeros`). GPU clocks were healthy
(SM 2340 MHz, P1, 46 °C), so this is a CuPy/sm_120 reduction-grid issue, not throttling.

Implication for the operator: every default `load()` pays ~15 s for an advisory message it could
compute on a downsampled view or skip. This is far larger than any decode-kernel win and should be
filed separately (it is general CuPy reduction behaviour, not the bitshuffle/LZ4 kernels in scope here).

---

## Reproduce
- Per-kernel timing: `CUDA_VISIBLE_DEVICES=1 NFRAMES=4000 MODE=time python /tmp/kern_harness.py`
- ncu (LZ4): `CUDA_VISIBLE_DEVICES=1 NFRAMES=3000 MODE=ncu ncu --kernel-name regex:h5lz4dc_batched --launch-skip 1 --launch-count 1 --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis --section WarpStateStats --section LaunchStats --section ComputeWorkloadAnalysis python /tmp/kern_harness.py`
- ncu (bitshuffle): same with `--kernel-name regex:shuf_8192_16_batched`
- Harness: `/tmp/kern_harness.py` (prepares the master once via `_prepare_master`, replays the two kernels on held GPU inputs; no kernel source modified).
