# Offline 4D-STEM browser compression - findings (2026-06-03)

Question: compress 4D-STEM so the browser reads + decompresses it at high ratio,
**lossless, no Python, no server** - just native browser APIs. Tested on real
data (Berk FAU STO/TiO2 120x120x96x96; gold_512 bin8 512x512x24x24).

## Verified result: WGSL GPU LZ4 decode works
Ported the LZ4 block decode (from the repo's CUDA/Metal `bslz4` kernel) to a WGSL
compute shader (single-thread-per-block, byte-packed in u32). Ran in headed
Chrome on a real bslz4 block: **0/8192 byte mismatches - bit-exact.** GPU LZ4 in
the browser is feasible (blocks are independent -> parallel across blocks).

## But: byte-shuffle + native gzip beats bslz4, and is simpler

Measured compression ratio on real data (lossless):

| data | gzip | **byte-shuffle + gzip** | bslz4 |
|---|---|---|---|
| Berk uint16 (sparse raw detector) | 10.2x | **11.8x** | 8.8x |
| Berk uint8 (quantized) | 5.9x | 5.9x | 2.6x |
| gold bin8 uint16 (dense/binned) | 1.6x | 1.9x | 1.5x |
| gold bin8 uint8 | 2.3x | 2.3x | 1.6x |

Conclusions:
1. **byte-shuffle + gzip is the best codec here**, not bitshuffle+LZ4. The
   detector's high byte is mostly zero (low counts), so byte-shuffle groups all
   the zero high-bytes together -> gzip crushes them.
2. **uint16 beats uint8.** Quantizing to uint8 *destroys* the high/low-byte
   structure shuffle exploits, so uint8 compresses *worse* (5.9x) AND loses
   precision. Shipping **uint16 + shuffle + gzip is 11.8x AND lossless**.
3. **10-20x needs sparse raw data.** Berk raw detector -> ~12x. Pre-binned gold
   (dense, high counts) -> ~2x. Ratio tracks sparsity, not the codec choice.

## Recommended pipeline (native, lossless, no WGSL/WASM needed)
```
uint16 stack -> byte-shuffle (transpose byte planes) -> gzip
  ship the .gz (companion file on Pages, or HF)
browser: fetch -> DecompressionStream('gzip')  [native]  -> byte-unshuffle [cheap transpose] -> uint16
```
This is fully native (`DecompressionStream` is built-in), lossless, ~10-12x on
sparse data, and needs **no custom GPU decoder**. The WGSL LZ4 decoder works and
is a good tool for the streaming/real-time future, but for the offline widget
shuffle+gzip is the simpler, higher-ratio, lossless win.

## Hardware ceilings (tested)
- **WebGPU single storage buffer = 1 GB** on the test NVIDIA adapter
  (`maxStorageBufferBindingSize`). So the decompressed stack that backs live
  detector interaction must be <= 1 GB per buffer, or split across N buffers.
- **512x512x48x48 (604 MB)** fits one buffer -> full interaction works (3 s inline).
- **512x512x96x96 (2.4 GB)** exceeds one buffer -> needs multi-buffer chunking;
  inline HTML opens (first paint) but full-detector compute needs chunked buffers.
- **512x512x192x192 (9.6 GB uint8)**: full interactive needs ~10x 1 GB buffers =
  ~10 GB GPU memory -> infeasible on typical hardware. The **flip-through** dream
  (scrub scan positions -> per-position diffraction) IS feasible at any size via
  **lazy per-frame decode** (decompress only the viewed frame's block on demand,
  tiny memory) + a precomputed inline virtual image. That is the path to "load +
  flip through a huge dataset in seconds, native, no Python."

## Speed (tested, phil M-series Mac)
- companion-fetch (data out of the HTML) -> **first paint 0.5-1 s** regardless of
  stack size (the page is tiny; data streams in background).
- inline 512x512x48x48 (gzip) 3 s; 512x512x96x96 23 s (parse-bound).
- The codec affects *download bytes*; first-paint speed comes from getting the
  bulk out of the HTML parse path (companion-fetch) + an inline precomputed VI.

## UPDATE: 512x512x192x192 in the browser IS feasible (all pieces tested)

Earlier "can't hold 9.7 GB" was WRONG - contaminated by 52 leaked Chrome tabs. Clean tests:

- **WebGPU total GPU memory:** allocated **40 x 512 MB = 20 GB, no OOM** (loop limit, not a wall) on the 96 GB-VRAM box. The 1 GB cap is PER-BUFFER, not total.
- **JS single ArrayBuffer:** ~2 GB cap (2^31). Circumvented by streaming - never materialize the whole stack in one JS buffer.
- **uint8 clip(0,255):** on real gold 192x192, **99.989% of pixels EXACT** (only 0.011% saturated/dead pixels clip). Near-lossless because real counts are 0-~204. (Global-linear uint8 was the trap: one dead 2^32-1 pixel set the scale and crushed all signal to 0.)
- **Chunked-streaming masked_sum (KEYSTONE, tested bit-exact):** stream the stack chunk-by-chunk (scan-row ranges) through ONE reused ~1 GB GPU buffer, dispatch masked_sum per chunk writing the VI slice at its global scan offset. Verified **maxErr = 0 vs the full numpy reference** (4 chunks, 1 buffer). Scales to any size: 9.7 GB streams through one 1 GB scratch, GPU peak = 1 GB + 1 MB VI.
- **WGSL LZ4 decode:** bit-exact (0/8192) - GPU decompression works if a wasm/native codec isn't preferred.

**Architecture for 512x512x192x192, native browser, no Python:**
```
save: uint8 clip(0,255) -> byte-shuffle -> gzip, in chunks (scan-row ranges) -> ship to HF/Pages
load: for each chunk: fetch -> DecompressionStream('gzip') -> unshuffle (<2GB ArrayBuffer)
      -> writeBuffer into ONE ~1GB GPU scratch -> masked_sum dispatch -> accumulate VI slice
recompute on aperture change: re-stream (or hold N chunk-buffers if VRAM allows for instant interaction)
flip-through: decode only the viewed frame's chunk on demand
```
GPU peak = 1 GB (streaming) regardless of dataset size. Feasible on big-VRAM (this box, phil 24 GB unified likely); not 8 GB. Every component above is measured, not claimed.

## Next steps
1. Switch the offline pack to **uint16 + byte-shuffle + gzip** (lossless, higher
   ratio) instead of uint8 + gzip - except where the GPU buffer / memory budget
   forces binning.
2. **Companion-fetch + inline precomputed VI** for instant first paint.
3. **Lazy per-frame decode** to flip through arbitrarily large datasets.
4. Multi-buffer chunking only if live full-detector interaction on >1 GB stacks
   is required.
