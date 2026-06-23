# MPS no-bin load footprint: 24 GB Mac freeze → 20.2 GB floor (2026-06-02)

## Question

On a 24 GB MacBook (phil, M5), `load(master)` of a no-bin Arina 4D-STEM stack
(`device`, 512×512×192×192 uint16 = 19.33 GB) FROZE the machine — twice —
when followed by `dpc()` / `virtual()`. The data is 19.33 GB; the box has 24 GB.
Why does it not fit, and what is the real floor?

## Setup

- Host: phil, Apple M5, 24 GB unified memory, branch `widget-show3d-show4dstem-kernels`.
- Data: `/Users/macbook/data/sample/device_master.h5` (262144 frames, 192²).
- Measured `currentAllocatedSize()` (Metal) AND `phys_footprint` (`vmmap -summary`,
  the number macOS uses for memory pressure / jetsam — Activity Monitor "Memory").
- Watchdog thread aborts the process at RSS > 22.5 GB or swap growth > 3 GB so a
  bad run can never freeze the Mac.

## Findings

| stage | metric | value |
|---|---|---|
| before fix (eager bin2 sidecar) | `currentAllocatedSize` | **26.85 GB** → freeze |
| after fix, no-bin load | `currentAllocatedSize` | 22.01 GB |
| after fix, no-bin load | **phys_footprint** | **20.20 GB** |
| sum of returned chunk buffers | — | 19.33 GB (exact data) |
| full e2e peak (load+virtual+dpc) | **phys_footprint** | **20.20 GB** (3.80 GB headroom under 24) |

Two things were wrong, one big and one a red herring:

1. **The +4.8 GB bin2 viewer sidecar (the freeze).** `load_mps_4dstem`
   defaulted to `fast_det_bin=2`, fusing a detector-bin2 copy (4.83 GB) into the
   SAME decode pass as the 19.33 GB no-bin chunks. That sidecar is a *viewer*
   fast-scrub aid; `dpc`/`virtual`/`Dataset4dstemGPU` never touch it. So every
   no-bin `load()` cost 19.33 + 4.83 + ~2 GB scratch ≈ 26.85 GB → over 24 → swap
   thrash → freeze. Fix: default `fast_det_bin=None`. The viewer still gets its
   sidecar — `Show4DSTEMMPS` builds it lazily in a background thread on open
   (`ChunkedFrames.ensure_fast_interaction`), so the viewer self-heals and the
   compute path never pays.

2. **`currentAllocatedSize` over-reports by ~1.8 GB (the red herring).** After
   the fix it reads 22.01 GB and does NOT drop on `del result` — even though the
   chunk buffers sum to exactly 19.33 GB and there are *zero* live
   `MPSDecompressor` instances. `currentAllocatedSize` is a Metal *reserved-pool*
   number that retains freed-but-cached shared buffers; it is NOT live resident
   memory. The metric that actually drives the freeze is **`phys_footprint` =
   20.20 GB** — only **+0.87 GB over the 19.33 GB data floor**. That 0.87 GB is
   the decode pipeline's working set (lz4 + compressed staging, double-buffered
   for disk‖GPU overlap). Chasing it to exactly 19.33 means serializing the
   pipeline (`QT_MPS_GPU_DEPTH=1`) and losing decode speed — not worth it.

## Conclusion

Real no-bin footprint on a 24 GB Mac is **20.2 GB** (19.33 data + 0.87 working
set), 3.8 GB headroom — at the practical floor. The freeze was the eager bin2
sidecar, now viewer-lazy. Speed held: load 2.5 s (8.1 GB/s), dpc 2.5 s, virtual
2.4 s; full e2e returns rotation 170.9° matching the quantem.live reference.

Rejected: (a) serializing the decode to reclaim the 0.87 GB working set — costs
speed for memory the box has; (b) trusting `currentAllocatedSize` as the budget
— it over-reports Metal's reserved cache by ~1.8 GB and would have sent us
chasing phantom memory.

## Fix

`src/quantem/widget/kernels/io/mps.py` — `load_mps_4dstem`: `fast_det_bin`
default `2 → None`; always `clear_mps_cache()` after a load so a later
`del result` releases the data instead of the cached decompressor pinning it.

## Follow-up: the viewer scrub sidecar — bin4 fits 24 GB AND is 16× faster

The viewer recomputes a BF/ADF virtual image (a detector masked-sum) on every
ROI-drag frame; that per-frame latency IS the scrub FPS. The fast path uses a
detector-binned sidecar so each sum touches fewer pixels. The old default was
bin2 (96², 4.8 GB) — but 19.33 + 4.8 = 24.1 GB does NOT fit a 24 GB Mac (the
viewer would freeze the same way the eager-sidecar load did). Switched the
default to a memory-aware **bin4** sidecar (`default_fast_bin()`: bin4 on
≤32 GB unified memory, bin2 above), built IN PLACE from the resident no-bin
chunks (`MetalVirtualImage.bin_chunks`, a general f×f Metal kernel) — no disk
re-decode, no decompress scratch, so the only new memory is the 1.2 GB sidecar.

Measured on phil (M5, 24 GB), no-bin device, BF mask:

| sidecar | det | ms/frame | FPS | extra mem | total phys |
|---|---|---|---|---|---|
| full-res (no sidecar) | 192² | 78.5 | 12.7 | 0 | 19.3 GB |
| bin2 | 96² | (4.8 GB) | — | 4.8 GB | **24.1 GB — does not fit** |
| **bin4 (new default)** | 48² | **4.8** | **207.6** | 1.21 GB | **20.5 GB (peak 21.2)** |

bin4 is strictly better on a 24 GB Mac: 16× faster scrub than full-res (207 vs
12.7 FPS, well past the 60 FPS bar) AND it fits with 2.7 GB headroom. The
coarser 48² detector mask is plenty for a BF/ADF scrub preview; precise virtual
detectors still use the full-res `vi`. Files: `reductions.msl`
(`bin_detector_u16` general kernel), `kernels/compute/mps.py`
(`bin_chunks`/`_bin_mask`/`_upsample_bin_dp`/`default_fast_bin`,
`ensure_fast_interaction` now in-place), `show4dstem_mps.py` (call sites use
`data.fast_bin`).
