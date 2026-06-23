# How offline data is embedded

The interactive widgets in these docs run with **no Python kernel** - the data
lives in the browser and the compute runs in WebGPU. The hard part is getting a
multi-megabyte array into the page *fast* without losing precision. This page
documents how Show4DSTEM (the heaviest case) does it.

## The problem

A 4D-STEM stack is large (a full 512x512 scan x 24x24 detector is 288 MB as
uint16). The naive path - base64 the array into the widget-state JSON inside the
HTML - is slow and deploy-hostile:

- The browser must parse the whole multi-hundred-MB HTML text, then `JSON.parse`
  the embedded string, on the main thread, before the widget even mounts.
- A 200 MB+ HTML file cannot be committed to GitHub (>100 MB is rejected, >50 MB
  warns) and bloats the repo forever.

## The pipeline

```
detector counts (uint16)
  → uint8 clip to [0, 255]                # exact for common low-count pixels
  → gzip (lossless)                        # ~2-3x smaller again
  → [inline base64]  or  [companion .gz]   # two delivery modes (below)
  → DecompressionStream('gzip')            # native, off the parse path, lossless
  → WebGPU storage buffer                  # masked_sum / reduce_frames in WGSL
```

**Why uint8 clipping is acceptable for the browser viewer.** Most browse-mode
detector counts in these datasets are well below 255, so the stored uint8 value
is the raw detector count, not a rescaled approximation. Saturated/hot pixels are
tracked separately and masked in the browser path so they do not dominate the
virtual image. The raw reconstruction path remains uint16; uint8 clipping is only
the browser display/interaction pack.

```{warning}
This is a display/interactivity path, not the reconstruction data path. Counts
above 255 clip, so use the live CUDA/MPS uint16 path for quantitative work where
high-count saturation matters. The browser path is designed for responsive
inspection: virtual detector drag, probe-frame scrub, FFT, and exported review.
```

**Why gzip.** `DecompressionStream('gzip')` is native (Baseline since May 2023:
Safari 16.4+, Chrome 80+, Firefox 113+), runs in C++ off the HTML-parse path, and
is bit-exact (lossless). Detector data (lots of low counts) compresses ~2-3x.

## Measured (full 512x512 gold, real data)

| Pack | HTML size | Cold open (Linux / 8 GB Mac) |
|---|---|---|
| uint16 base64 inline | 404 MB | 32 s / ~100 s (or freeze) |
| uint8 base64 inline | 203 MB | 9 s / 51 s |
| **uint8 + gzip inline** | **92 MB** | **3 s / ~15 s** |

All three render a virtual image **bit-identical to each other** (same uint8
source) - the speedups are pure size/parse wins. gzip is lossless; the only
quantization is the shared uint16→uint8 step (see the bright-field warning above).

## Two delivery modes

| | Docs (GitHub Pages) | Share with a colleague |
|---|---|---|
| Opened over | HTTP | double-click (`file://`) |
| Layout | tiny HTML **+ companion `.gz`** fetched at runtime | **one self-contained `.html`** |
| Why | a 90 MB inline file bloats the repo + trips GitHub's 50 MiB warning; Pages serves a sibling and same-origin `fetch()` works over HTTP | people won't manage a folder, and `fetch()` of a sibling is CORS-blocked under `file://` |
| Trait | `data_url=` (relative) | `offline=True` (inline gzipped) |

Both share the same gzip + `DecompressionStream` + WebGPU code; only the *source
of bytes* differs (a `fetch()` vs an inline base64 trait).

## Full no-bin and multi-dataset exports

Large no-bin 4D-STEM stacks use the bslz4 companion path instead of one huge
self-contained HTML file:

```python
w = Show4DSTEM(data, backend="web", offline_codec="bslz4", data_url="show4dstem-data")
w.export_html("show4dstem.html")
```

For a full `512 x 512 x 192 x 192` stack, the HTML remains small and the data
lands in a sibling directory of bslz4 chunks. Serve the HTML and companion
directory over HTTP from the same parent directory; a browser cannot fetch
sibling files from `file://`.

5D stacks are exported as lazy browser volumes:

```python
w = Show4DSTEM(stack5d, backend="web", offline_codec="bslz4", data_url="stack-data")
```

The exported metadata is `{volumes:[...]}`. The Dataset/frame slider decodes the
selected volume on demand and keeps a small browser-side LRU, so multiple full
no-bin datasets do not have to be resident in WebGPU memory at once.

Signoff point: `show4dstem-migration-signoff-2026-06-05` verified CUDA, Phil
MPS, WebGPU live/browser compute, exported WebGPU, and lazy multi-volume WebGPU.

## Going faster still (roadmap)

The goal: least internet (bytes) **and** least compute (decode). Ideas, lossless:

- **Instant first paint (recommended next step):** ship a tiny precomputed virtual
  image inline (KB) so the page renders immediately, and load the full stack in the
  background - detector interaction lights up a couple seconds later. This is the
  only lever that feels instant on **both** surfaces: on `file://` the cold-open
  floor is Blink's single-threaded HTML-text parse of the inline bytes (which no
  worker or GPU trick can beat), so the only way to feel instant there is to not
  block on the big payload at all. Adds ~zero bytes, zero precision loss.
- **OPFS / IndexedDB cache:** after the first fetch, cache the decoded bytes;
  re-opening the page is instant with zero network and zero decompress.
- **Better codec (measure first):** brotli (`DecompressionStream('brotli')`) is
  ~15-20% smaller than gzip, but Streams-API brotli is newer + unevenly shipped
  (Safari 18.4+; not broadly in Chromium yet) - check support before relying on it.
  zstd and bitshuffle+LZ4 (the upstream HDF5 detector codec) compress integer data
  even better but need a **wasm decoder** that itself costs bytes + compute to
  load - often erasing the ratio gain. For a widget whose problem is bytes+compute,
  native gzip is usually the sweet spot; reserve wasm codecs for a measured spike.
- **Streaming decode:** overlap fetch + decompress + GPU upload so first frame
  appears before the whole stack lands.
- **Off-main-thread:** decompress in a Web Worker (transferable buffer) so the UI
  never blocks during load.
