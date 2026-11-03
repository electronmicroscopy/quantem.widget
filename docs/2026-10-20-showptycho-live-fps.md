# ShowPtycho live SSB frame rate, 2026-10-20

## Question

How many new phase images per second does a microscopist see while dragging
the C10 slider of a live MPS SSB session, and what limits it?

## Setup

One full 512 × 512 × 192 × 192 acquisition (8,939 bright-field pixels) on an
Apple M5 Max, opened with `ssb.show(show_fft=True)` in JupyterLab and Chrome
(WebGPU adapter: Apple Metal, not a software fallback). A scripted pointer
dragged the C10 thumb back and forth for six seconds; frames are paints of the
phase panel and of the FFT panel inside the drag. The kernel side logged every
`preview` call. All GPU work ran under a shared lock with other jobs.

## Changes

- The slider keeps two overview requests in flight instead of one. The kernel
  answers requests in order, so it starts the next reconstruction while the
  previous phase is still on its way to the browser. Newer slider values
  still wait as one latest-wins request; the release commit is never dropped;
  regional views keep one request.
- One display FFT runs at a time and only the newest waiting phase is
  transformed next. Starting one per phase and discarding all but the newest
  starved the FFT panel once phases arrived faster than a 1024² display FFT.
- Mean subtraction, the Hann window, padding, interleaving and the magnitude
  split run in the display worker with the same arithmetic (unit-tested bit
  for bit), leaving the input thread free to dispatch the next request.
- Rotation slider drags (trait updates from the browser) skip the variance
  loss like the other drags; the release commit computes it. A rotation step
  is now the 0.8 ms geometry update plus the phase, not phase and loss.

quantem.gpu's separable C10/C12 coherent kernel (same date) cut the native
preview from about 57 to 24 ms and the 2× preview from 38 to 28 ms.

## Measurements (phase panel fps, two drags each)

| Configuration | Native 512² | 2× 1024² (phase / FFT) |
| --- | ---: | ---: |
| Before (old kernel, one request) | 16.5 / 15.6 | 19.9 / 19.0 (FFT same) |
| New kernel, one request | 25.5 / 27.2 | 20.6 / 22.5 (FFT 11 / 10) |
| + two requests in flight | 34.5 / 33.7 | 24.6 / 24.6 (FFT 5.6 / 5.1) |
| + one display FFT at a time | not measured | 22.0 / 22.6 (FFT 22.2 / 22.6) |
| + FFT preparation in the worker | not measured | 23.7 / 23.6 (FFT 22.2 / 21.9) |
| All changes, rebased, final run | 37.4 / 35.5 | 23.8 / 24.4 (FFT 21.8 / 22.4) |

At native sampling the kernel now idles about 1 ms between requests and the
frame rate follows the reconstruction time (23.5 ms median in the final run). At 2× the kernel could deliver
about 32 phases per second; the browser paints about 24 because every frame
also recomputes two full-resolution histograms and percentile windows on the
input thread and colormaps both 1024² panels.

## Next

Move the per-frame histogram and percentile passes of the phase and FFT
panels onto the GPU colormap engine (or one shared pass) so 2× follows the
reconstruction rate.
