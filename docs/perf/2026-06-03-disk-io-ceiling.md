# Disk-IO ceiling for cold 4D-STEM loads (2026-06-03)

Motivation: with 40-50 datasets and limited VRAM/RAM, **cold disk read is the load cost
that matters** (warm page cache is the lucky case). Goal: drive cold read toward zero and
toward using disk as an out-of-core memory tier for reconstruction. Measured on host,
gold master (27 files, 3.15 GB compressed) on the WD SN850X. No loader changes — pure
measurement. Cold = `posix_fadvise(DONTNEED)` self-evict (no root needed).

## Hardware

| NVMe | model | mount | negotiated link | ~seq BW |
|---|---|---|---|---|
| nvme0 | Sample 9100 PRO 1TB | **unmounted** (only /boot/efi) | **32 GT/s Gen5 x4** | ~14 GB/s |
| nvme1 | Sample 9100 PRO 1TB | `/`, `/home/user`, `/tmp` | 16 GT/s Gen4 x4 | ~7 GB/s |
| nvme2 | WD_BLACK SN850X 8TB | `/home/user/ssd` (DATA) | 16 GT/s Gen4 x4 | ~7 GB/s |

GPU0 PCIe Gen5 x8 (~32 GB/s H2D) — the bus is NOT the floor; the disk is.

## Cold vs warm read scaling (WD nvme2, gold 3.15 GB)

| threads | cold GB/s | warm GB/s |
|---|---|---|
| 1 | 2.36 | 9.46 |
| 8 | **6.02** | 30.6 |
| 12 (loader default) | 5.83 | 31.8 |
| 16 | 6.08 | 31.3 |
| 32-48 | ~5.5 | ~29 |

- **Cold ceiling ≈ 6.0 GB/s** = 82% of the WD's 7.3 GB/s rating. The loader's 12-thread pool
  is already in the optimal 8-16 band. The READ path is near-optimal for this drive.
- 3.15 GB cold ≈ **525 ms**; warm ≈ 100 ms. The 376 ms full-load number reported earlier was warm.

## Things that did NOT help (measured, rejected)

- **kvikio 26.02 (GPUDirect Storage lib) in compat mode**: 0.22 cold / 2.77 warm GB/s on a
  single file — *slower* than POSIX. Without the `nvidia-fs` kernel driver kvikio can't do
  real NVMe→VRAM DMA; compat falls back to synchronous unaligned host reads. Useless until
  the GDS driver is installed.
- **Striping across nvme1 + nvme2 concurrently**: 4.58 GB/s aggregate — LESS than nvme2
  alone (6.02). nvme1 is the busy root drive (3.48 GB/s under OS/dashboard contention) and
  drags the aggregate. A real striping win needs dedicated, idle drives, not the root disk.
- More threads (>16): no gain, slight loss (host queue depth).

## The real levers to push cold toward zero (all need operator/root action)

1. **Mount + use the idle Gen5 nvme0 (~14 GB/s) for hot datasets.** It negotiates 32 GT/s
   (true Gen5) but holds no data today. Moving working datasets there is ~2.3× the WD's cold
   BB for the cost of an `fstab` mount. Biggest near-free win. NEEDS ROOT (mount).
2. **GPUDirect Storage** — install `nvidia-fs` + `modprobe`, then kvikio (already installed)
   does NVMe→VRAM DMA, bypassing the CPU and the host-pinned bounce. This is the enabler for
   disk-as-memory out-of-core recon (read batches straight to GPU, CPU free). NEEDS ROOT.
3. **Stream-overlap read with GPU compute (software, parity-gated).** Today prepare (disk)
   and decompress (GPU) are serial. Pipelining them hides the GPU phase under the read so
   *felt* latency approaches the raw read time; for recon, stream batches NVMe→GPU on demand
   and never hold the full dataset — disk BW only matters if it can't keep the GPU fed
   (6 GB/s/drive vs the kernel consume rate). This is the out-of-core recon architecture.
4. **Dedicated RAID0 across multiple idle NVMe** (nvme0 + a second dedicated drive) →
   additive cold BW (~14+14 ≈ 28 GB/s). NEEDS ROOT + dedicating drives.

## Bottom line

The loader's read is already at 82% of this Gen4 drive's floor at the optimal thread count —
software thread-tuning is done. Cold read speed now is a **hardware + architecture** problem:
faster/striped drives (Gen5 nvme0 is idle), GPUDirect Storage for CPU-free NVMe→VRAM, and
stream-overlap so the read hides behind GPU compute. The last one is the path to "disk as a
memory tier for reconstruction."

## Exploration: is the read strategy optimal? (2026-06-03, exhaustive sweep)

Swept every software knob on the WD Gen4 to confirm the loader's read can't be tuned faster.
Single 7 GB master + 5-dataset (30 GB) workload, cold (DONTNEED evict), raw-byte reads.

Single master (7 GB):
| axis | result |
|---|---|
| block size 256 KB → 64 MB | flat 5.6-5.8 GB/s — irrelevant (64 MB slightly worse) |
| threads 4 / 8 / 12 / **16** / 24 / 32 | 5.52 / 5.79 / 5.81 / **5.96** / 5.71 / 5.62 — 16 is the sweet spot |
| readahead fadvise SEQ+WILLNEED | +6% (5.81 vs 5.49 off) — keep it |
| O_DIRECT | 6.05 vs 5.82 buffered (~4% cold) BUT kills warm cache |
| **warm (page cache)** | **36 GB/s — 6× cold** |

5 datasets (30 GB, 135 files), cold:
| strategy | GB/s |
|---|---|
| A sequential per-dataset, 16 thr | 5.97 |
| **B unified pool, 16 thr, buffered** | **6.00** (best) |
| B 24 / 32 thr | 5.83 / 5.41 (thread overhead) |
| C O_DIRECT 16 / 24 thr | 5.67 / 5.43 (worse at 30 GB scale) |

**Verdict: cold single-Gen4-drive read is saturated at ~6 GB/s. No software knob moves it** —
block size, thread count beyond 16, O_DIRECT, read pattern all flat or worse. The loader's
12-16-thread buffered+fadvise approach is already optimal. Read-loop tuning is DONE.

Best config for reading N Arina datasets on one drive: **unified thread pool over ALL files,
16 threads, buffered, posix_fadvise(SEQUENTIAL|WILLNEED), 4 MB blocks.**

The only ways past ~6 GB/s (all already filed):
- **warm/RAM cache = 36 GB/s (6×)** for RE-reads → pin hot datasets in page cache (#760 family);
  iterative time-series on the same datasets is warm after first touch — essentially free 6×.
- **Gen5 drive #761** (~14, 2×) · **multi-disc→multi-GPU #762** (N×) · **overlap read w/ compute #760**.
- **io_uring** (no python binding in env): same queue depth with 1-2 threads, lower CPU — a
  CPU-efficiency win, NOT a raw-BW win (drive is the wall). Future, needs a C binding.
