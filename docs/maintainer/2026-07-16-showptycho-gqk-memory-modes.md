# 2026-07-16 — ShowPtycho WebGPU resident-memory experiment: Exact and compact preview storage

## Question

The ShowPtycho WebGPU folder viewer keeps the per-BF-pixel `G(q,k)` reducers
resident on the GPU for the whole review session (that residency is what makes
aberration-slider drags ~15 ms instead of a full HDF5 re-decode). Resident cost
is `active_BF_pixels x scan_pixels x 8 bytes` (complex64). On a 512x512 scan at
full BF that projects to tens of GB — workstation-only. Question: **can the
resident footprint shrink without losing precision or drag speed**, so
collaborators can open these folders on ordinary laptops?

## Setup

- Dataset: a real experimental 4D-STEM acquisition
  (512x512 scan, 192x192 Arina, 19.3 GB raw, max pixel 17 counts).
  Calibration: fresh SSB fit, 200 Optuna trials + Nelder-Mead refine
  (`Show4DSTEM.compute_ssb`, rotation seeded from a prior batch fit).
- Current export command: `quantem showptycho <master> --calibration <fit.json>`
  with the historical BF-column companion enabled
  (WebGPU folder, compressed HDF5 + `bf_columns.u8` companion, 5.9 GB on disk).
  Current exports use the compressed-HDF5 WebGPU source directly; the
  BF-column companion transport was later removed.
- Viewer host: reference Apple Silicon laptop (M5 MacBook, 24 GB unified,
  Chrome, adapter `apple/metal-3`
  — real hardware confirmed via `adapter.info`, not SwiftShader).
- Harness: folder served by `scripts/serve_sidecar_range.py` (plain
  `python -m http.server` FAILS — no HTTP Range support, viewer dies with
  "Failed to fetch"). Page driven over CDP; public viewer choices are Exact
  (default, or `?gqk=exact`) and compact preview (`?gqk=preview` or
  `?gqk=herm16`). Earlier parity runs also measured the old n x n storage
  baseline before that runtime path was removed. The engine stashes
  `globalThis.__quantemSsbLast = {gqkMode, residentGqkBytes, gpuMs, phase, ...}`
  after every reconstruct, and the harness pulls the raw `Float32Array` phase
  out over CDP for offline numpy comparison. Same aberrations, same BF set
  (preview 0.30 => 3941 requested, 408 active aperture pixels) in all runs.

## Raw numbers

| G(q,k) mode | resident VRAM | reconstruct (gpuMs) | max abs dphase vs old n x n baseline | rms dphase | phase corr |
|---|---|---|---|---|---|
| old n x n baseline (removed runtime path) | 0.856 GB | 18.2 ms | — | — | — |
| Exact / `herm` (complex64 Hermitian half-plane) | 0.429 GB (2.0x) | 16.4 ms | **0.0 (bit-exact)** | 0.0 | 1.0 |
| Compact preview / `herm16` (half-plane + snorm16, one f32 scale per BF px) | 0.215 GB (4.0x) | 12.6 ms | 1.204e-4 rad | 2.60e-5 rad | 0.9999954 |

Phase image span was 0.0748 rad, so compact preview worst-case error is 0.16 % of
span (rms 0.03 %) — far below shot noise on 17-count data.

Projection for this dataset at **full BF (13137 px, ~1360 active est.)**:
old n x n baseline ~2.9 GB, Exact ~1.4 GB, compact preview ~0.7 GB.
Upper-bound projection if every selected BF pixel were aperture-active:
27.6 / 13.9 / 6.9 GB. Per-BF-pixel cost by scan size: 2 MB at 512^2,
0.5 MB at 256^2, 0.125 MB at 128^2 for the old n x n baseline;
divide by 2 for Exact and by 4 for compact preview.

## Memory planning table

These are resident `G(q,k)` reducer sizes only. First load may need additional
temporary chunk, phase/loss, canvas, and browser memory. `Active BF` means
nonzero-aperture BF pixels after the BF policy is applied; it is often smaller
than the raw BF label shown in the UI.

| Scan | Active BF | Typical use | Old n x n baseline (not runtime) | Exact default | Compact preview |
|---|---:|---|---:|---:|---:|
| 512x512 | 12 | Small smoke test | 25 MB | 13 MB | 6.3 MB |
| 512x512 | 408 | 0.30 BF preview in the reference report | 856 MB | 429 MB | 215 MB |
| 512x512 | 1360 | Full-BF estimate for the sparse experimental 512 dataset | 2.85 GB | 1.43 GB | 0.72 GB |
| 512x512 | 9070 | Dense experimental full active BF | 19.0 GB | 9.55 GB | 4.77 GB |
| 1024x1024 | 12 | Small smoke test | 101 MB | 50 MB | 25 MB |
| 1024x1024 | 1382 | Reference full active BF | 11.6 GB | 5.81 GB | 2.90 GB |
| 1024x1024 | 9070 | Workstation stress projection | 76.1 GB | 38.1 GB | 19.1 GB |

Formula:

- Old n x n baseline: `active_BF * N * N * 8` bytes.
- Exact default: `active_BF * N * (N/2 + 1) * 8` bytes.
- Compact preview: `active_BF * N * (N/2 + 1) * 4 + active_BF * 4` bytes.

## Why herm is exactly lossless — and faster

The stored `G(q,k)` is the scan-space FFT of each BF pixel's intensity trace,
and intensities are real, so `G(-q,k) = conj(G(q,k))`. Storing the
`n x (n/2+1)` half-plane and mirror-conjugating on fetch is algebra, not
approximation. Measured **bit-exact** (max diff literally 0.0): the radix-2
FFT's rounding errors are themselves conjugate-symmetric for real input, so
the discarded half was a bitwise mirror all along. It is *faster* because the
per-drag reduce is bandwidth-bound (re-reads all of G every slider move) and
now reads half the bytes. Strictly better on every axis => **`herm` is the new
default Exact path**; the old n x n storage branch has been removed from the
runtime and now exists only as historical baseline data in this report.

## Implementation (`quantem.gpu.ssb.compute.webgpu/backend.ts`)

- `GqkMode` = `herm | herm16`, resolved from `?gqk=` URL param or
  `globalThis.__QUANTEM_SHOWPTYCHO_GQK_MODE__`; default `herm`. Public aliases:
  `exact` -> `herm`, `preview` -> `herm16`.
- `makeSsbShader(n, mode)` templates a `fetch_g(local_bf, bf_global, row, x)`
  WGSL helper: direct read for `x <= n/2`, mirror `(r,c) = ((n-row) % n, n-x)`
  + conjugate otherwise; `herm16` additionally
  `unpack2x16snorm(word) * gqkScale[bf_global]`.
- Build path unchanged (temporary n x n chunks, gather + in-place FFT), then a new
  GPU post-pass `transformGqkChunks` runs `scaleMax` (per-BF-pixel max |G| over
  the half-plane, workgroup tree reduce) and `compact` (copy or
  `pack2x16snorm(clamp(v/scale))`) per chunk, destroying each temporary chunk
  immediately — build peak only briefly exceeds the old peak by one compacted
  chunk; the *resident* session footprint is what shrinks.
- `__quantemSsbLast` debug hook on every reconstruct for harnesses.

## Rejected / deferred ideas

- **float16 storage**: 2x, but real mantissa loss on FFT accumulations
  (10-bit mantissa vs values spanning ~4.5e6 dynamic range). Rejected —
  strictly worse than compact preview, which spends its 16 bits after per-pixel
  scaling.
- **Band-limit crop in q**: exact only when the scan oversamples the 2-alpha
  double-overlap disk. This dataset (10.4 A scan sampling, 30 mrad, 300 kV;
  q_Nyquist 0.048 1/A << 2-alpha/lambda 3.05 1/A) is in-band across the whole
  q-plane — zero win here, dataset-dependent in general. Not implemented.
- **Store raw counts, rebuild G per drag**: unbounded memory win but turns
  every slider move into a full FFT rebuild — kills the 15 ms interactivity.
  Only sensible as a future explicit "final full-BF render" button.
- **Aberration-basis factorization of the k-sum**: the gamma weight is
  nonlinear in the coefficients (`e^{i chi}`), no exact low-rank split. Rejected.
- **Streaming the initial build** (bounded peak, not just bounded resident):
  requires either re-decoding all HDF5 chunks per BF chunk (N_chunks x slower
  load) or a compact real-u16 time-domain gather buffer. Deferred; noted as
  the remaining lever for laptop-friendly *first load*.

## Gotchas recorded

- `python -m http.server` cannot serve ShowPtycho folders (no Range support);
  use `scripts/serve_sidecar_range.py --dir <folder> --port <p>`.
- `collectActiveBfIndices` drops zero-aperture-weight BF pixels: the
  "3941/13137 BF" UI label overstates the resident set (408 active here).
  Memory projections must use *active* BF counts.
- `FULL_STACK_GPU_BUDGET_BYTES` (4.5 GB) in the synced ShowPtycho WebGPU SSB
  source is still dead code
  — the VRAM clamp for the BF slider remains unimplemented. With `herm` default
  the pressure is halved but a 512^2 full-BF drag can still device-lost a small
  GPU. Follow-up: cap effective BF by `budget / (storedPlane x bytesPer)`.

## Backend coverage checklist (for follow-up agents)

Status as of 2026-07-16. The math is backend-independent: every SSB backend
builds `G(q,k)` as the scan-space FFT of real intensity traces, so the
Hermitian identity `G(-q,k) = conj(G(q,k))` holds everywhere, and the
per-BF-pixel snorm16/int16 block quantization transfers directly.

| Optimization | WebGPU (`quantem.gpu/ssb/compute/webgpu/backend.ts`, generated under the matching widget engine tree) | CUDA (`quantem.gpu/ssb/compute/cuda`) | MPS (`quantem.gpu/ssb/compute/mps`) |
|---|---|---|---|
| Exact Hermitian half-plane G(q,k) (2x, bit-exact, faster) | **DONE — default** | TODO — `self.G_qk` is n x n complex64; also the streaming `result_buffer`/staging buffers (batch x bf x scan^2 x c64) would halve | TODO — `mx.complex64` n x n storage; gamma kernels at mps.py:430-440 already compute conj explicitly, mirror fetch slots in there |
| Compact preview snorm16/int16 block storage (4x, ~1e-4 rad error) | **DONE — opt-in** `?gqk=preview` (`?gqk=herm16` alias) | TODO — cupy int16 pairs + per-BF f32 scale; dequant inside the variance/correction kernels | TODO — mx int16 + scale; check MLX gather perf before committing |
| VRAM budget clamp on BF count | TODO — `FULL_STACK_GPU_BUDGET_BYTES` still dead code | n/a (96 GB workstation assumption baked in; revisit for L40S) | TODO — unified memory, clamp matters most on 8-16 GB Macs |
| Streamed initial build (bounded peak, not just resident) | TODO — needs real-u16 time-domain gather or per-chunk re-decode | n/a today | TODO |

WebGPU scan-size coverage:

| Scan size | Exact complex64 Hermitian `G(q,k)` | Compact preview `herm16` | Verification status |
|---|---|---|---|
| 128x128 | Done | Done | Synthetic parity sweep passed. |
| 256x256 | Done | Done | Synthetic parity sweep passed. |
| 512x512 | Done | Done | Real experimental parity and headed-browser timing measured. |
| 1024x1024 | Done | Done | Synthetic parity sweep passed; compact preview is preview-quality because error was larger than at 512. |

Verification recipe for a port (what was used here): compute the same
reconstruction with the optimization off and on (same aberrations, same BF
set), assert `max|dphase|` is 0.0 for Hermitian and < ~1e-3 of the phase span
for int16; then compare per-drag wall time — Hermitian must not be slower
(it reads half the bytes; if it is slower, the mirror fetch broke coalescing).
Raw parity harness for the WebGPU case: CDP + `globalThis.__quantemSsbLast`
(this doc's Setup section).

## Scan-size sweep (added same day)

Reference kernel sweep across all supported scan sizes, three modes each, on the reference laptop
(`apple/metal-3`). 128/256/1024 are synthetic Arina-style masters written with
`quantem.gpu.io.save` (uint16, 48x48 detector, disk + gradient-shift phase
object, semiangle 8 mrad, det sampling 1 mrad/px); 512 is the real
experimental row from the table above. Preview BF (0.30); activeBf ~60 for
the synthetic sets, 408 for real 512. gpuMs is the first reconstruct (launch-
dominated at small BF counts; the 512-real row is the bandwidth-relevant one).

| scan | mode | resident | gpuMs | max abs dphase vs old n x n baseline | rms | phase span |
|---|---|---|---|---|---|---|
| 128 | old n x n baseline | 8.0 MB | 4.7 | — | — | 1.002 rad |
| 128 | Exact / herm | 4.1 MB | 4.5 | **0.0** | 0.0 | |
| 128 | Compact preview / herm16 | 2.0 MB | 4.8 | 7.0e-4 (0.07 % span) | 1.5e-4 | |
| 256 | old n x n baseline | 31.5 MB | 8.5 | — | — | 0.813 rad |
| 256 | Exact / herm | 15.9 MB | 5.8 | **0.0** | 0.0 | |
| 256 | Compact preview / herm16 | 7.9 MB | 5.3 | 1.3e-3 (0.16 % span) | 3.1e-4 | |
| 512 (real) | old n x n baseline | 856 MB | 18.2 | — | — | 0.075 rad |
| 512 (real) | Exact / herm | 429 MB | 16.4 | **0.0** | 0.0 | |
| 512 (real) | Compact preview / herm16 | 215 MB | 12.6 | 1.2e-4 (0.16 % span) | 2.6e-5 | |
| 1024 | old n x n baseline | 495 MB | 13.1 | — | — | 0.459 rad |
| 1024 | Exact / herm | 248 MB | 12.5 | **0.0** | 0.0 | |
| 1024 | Compact preview / herm16 | 124 MB | 11.6 | 6.2e-3 (1.35 % span) | 1.3e-3 | |

Conclusions:

- **Exact / herm is bit-exact at every supported size** (128/256/512/1024,
  synthetic and real data) and never slower. Safe as the unconditional default.
- **Compact preview / herm16 error grows with scan size**: 0.07 % of span at 128 up to 1.35 %
  at 1024. Cause: one snorm16 scale per BF pixel spans the whole q-plane, and
  the dynamic range inside G(q,k) (DC-dominated peak vs weak high-q tail)
  widens with n, so a single per-pixel scale under-resolves the tail.
  Recommendation: compact preview is comfortably below shot noise up to 512; at
  1024 treat it as preview-only, or implement **per-q-row block scales**
  (n scales per BF pixel instead of 1, +0.4 % memory) to pull the error back
  down — noted as the follow-up for whoever extends the quantization.
- Repro: masters under a local private SSB sweep directory, harness
  `sweep_run.py` in the session scratchpad, per-mode viewer selection via
  `?gqk=`.

## Lessons learned (process, not just numbers)

1. **Ask the symmetry question before the hardware question.** The 2x win here
   did not come from a faster kernel — it came from noticing the input is real,
   so half the stored spectrum was a mathematical mirror. Physics/math
   equivalences (Hermitian symmetry, band limits, separability, known output
   realness) reduce the PROBLEM; occupancy and coalescing only speed up
   whatever problem is left. Always ask first: what symmetry, invariance, or
   physical constraint makes part of this data or compute redundant?
2. **Bandwidth-bound loops convert memory wins into speed wins for free.** The
   per-drag reduce re-reads all of G(q,k) every slider move, so halving the
   bytes halved the traffic — herm was faster, not merely smaller. When a loop
   is bandwidth-bound, compression IS optimization.
3. **Quantize after per-block scaling, in the domain with bounded dynamic
   range.** snorm16 works because each BF pixel gets its own scale. The same
   16 bits as raw float16 would have failed (mantissa loss across ~1e6 dynamic
   range). And the residual error law is set by the dynamic range INSIDE each
   block - which grows with scan size - hence the 1024^2 degradation and the
   per-q-row-scale follow-up.
4. **Bit-exactness is testable and worth demanding.** Expected ~1e-7 rounding
   differences from the mirror fetch; measured literally 0.0 because radix-2
   FFT rounding is itself conjugate-symmetric for real input. A tolerance-free
   `array_equal` assertion is a far stronger regression net than atol=1e-6.
5. **Measure the active set, not the labeled set.** UI said 3941 BF pixels;
   only 408 carried nonzero aperture weight. Memory projections from labels
   were 10x off. Instrument the engine (resident-bytes counter, `__quantemSsbLast`
   hook) instead of computing footprints from UI numbers.
6. **A parity harness is one page of code.** URL-param mode switch + a
   globalThis hook holding the raw Float32Array + CDP pull + numpy compare.
   Built once, it validated the default flip, the sweep, and will validate the
   CUDA/MPS ports.
7. **Verify the adapter before believing any GPU number** (`adapter.info` must
   not be SwiftShader), and serve folder exports with a Range-capable server -
   two silent failure modes that produce plausible-looking nonsense.

## Follow-ups landed 2026-07-17

- **VRAM budget clamp wired** (was dead code): the active BF set is capped at
  `budget / (storedPlane x bytesPerValue)` with uniform stride, default 4.5 GB
  (`__QUANTEM_SHOWPTYCHO_GQK_BUDGET_GB__` override), status line announces the
  clamp. Mode-aware: herm16 admits 4x more BF pixels under the same budget.
- **rfft half-plane calibrations accepted**: the CUDA backend now stores
  Hermitian-half `G_qk` (n x n/2+1) and exports that `g_shape`; the viewer
  derives n from either layout (it rebuilds its own G from the folder source).
- **Flattened scan input**: verified `(N, det, det)` works end to end -
  square scan inferred (or `scan_shape=` explicit) before `g_shape` is written.
- **Corrected-calibration visual A/B** (scan sampling fixed to 10.4 A,
  refit C10 383 nm / C12 79 nm / phi12 80 deg): old n x n baseline vs compact
  preview phase images indistinguishable; difference map is structureless
  noise, max 1.24e-4 rad (0.16 % of span), rms 2.7e-5 rad (0.035 %).
  Old n x n baseline 856 MB / 13.8 ms vs compact preview 215 MB / 14.0 ms at
  408 BF. Caution from the fit: C10 and C12 landed near the +-400 / 100 nm
  search bounds - widen the search when refitting.

## Object-redraw fast path ported from CUDA (2026-07-17 late)

The CUDA team's overnight identity - `mean_bf(ifft2(corrected_bf)) ==
ifft2(mean_bf(corrected_bf))` - ported to the WebGPU drag path: sum the
gamma-corrected G(q,k) over BF pixels in Fourier space (one bandwidth-bound
pass per chunk), then ONE 2D inverse FFT + one atan2 pass, instead of two FFT
passes + atan2 per BF pixel.

Measured (512x512 experimental folder, M-series, herm storage, 408 active BF):

| path | per-reconstruct | estimator |
|---|---|---|
| exact per-BF (before / loss commits) | 18 ms | mean(angle(object)) |
| object fast path (drags, default now) | **7-8 ms (2.5x)** | angle(mean(object)) - same as Python `SSB.result()` |

Correctness: fast vs exact at identical state corr 0.9965; fast vs the Python
CUDA `result()` object phase corr 0.997 (better than the exact mean-phase
path's 0.9916, because the estimator now MATCHES the backend reference).
Loss commits (slider release at full BF) always use the exact per-BF path -
`mean(angle)` variance needs per-BF phases.

Debug war story worth keeping: an initial "the fast path is broken"
conclusion (corr 0.3) was a BASELINE ERROR - comparing a 408-BF preview
subset against the full-13137-BF reference; the exact path scored the same
0.379 against that wrong baseline. Rule: when validating an estimator change,
hold BF subset, aberrations, and estimator definition fixed and compare
apples-to-apples first. Opt-out: `globalThis.__QUANTEM_SSB_OBJ_FAST__ = false`.
Also gained: `__QUANTEM_SSB_OBJ_ONLY_CHUNK__` chunk-isolation debug toggle.

### Object fast path across all sizes (same session)

Initial reconstruct, herm storage, preview BF, M-series (synthetic folders for
128/256/1024 carry only ~60 active BF; the real 512 row carries 408):

| scan | exact path | object fast path | note |
|---|---|---|---|
| 128 (61 BF) | 4.5 ms | 7.1 ms | tiny-BF regime: fixed FFT/dispatch overhead dominates, no win |
| 256 (60 BF) | 5.8 ms | 6.6 ms | parity |
| 512 real (408 BF) | 18 ms | **7.8 ms (2.3x)** | the win scales with BF count |
| 1024 (59 BF) | 12.5 ms | 12.4 ms | parity at synthetic BF; expect large win at real BF counts |

The speedup comes from removing two FFT passes + atan2 PER BF PIXEL, so it
grows linearly with active BF count. At <100 BF both paths are launch-bound
and equal; potential refinement: auto-select exact path below ~100 active BF
(both are instant there, so not urgent).

### Real-data full-BF result (the number that matters)

Re-measured with REAL pointer drags on the 512x512x192x192 experimental
dataset (19.3 GB), drag-BF at 1.0 = all 13137 BF (3418 aperture-active),
recorder capturing every reconstruct (24 in one drag):

| path | per-reconstruct | FPS |
|---|---|---|
| exact per-BF + loss (commits) | 112-127 ms | ~8 |
| **object fast path (drags)** | **18-23 ms (median ~20)** | **~50** |

**5.6x at real full BF.** 20 ms for 3.6 GB of herm G reads = ~180 GB/s -
right at M-class unified-memory bandwidth, i.e. the fast path is now
bandwidth-floor-limited, which is as fast as this storage layout can go.
Earlier suspect 4-8 ms "full-BF" readings were stale-stash sampling
artifacts; only synthetic input events miss the drag handler - real pointer
drags hit the fast path correctly, so there is NO routing gap for users.
Rule reinforced: never quote FPS from small/synthetic BF counts - the
per-size table above with ~60-BF synthetic folders shows path parity only
because both are launch-bound there; real BF counts are where the identity
pays.
