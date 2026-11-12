# ShowPtycho update path outside the kernel, 2026-10-24

## Question

A live frame-151 session at 3x Auto (1536 x 1536 phase, FFT panel on) updated at
3 to 12 frames per second while a slider was dragged, with only 2 to 60 ms of
that in the CUDA reconstruction. Where did the rest of each update go, and how
much of it can the widget remove without reducing the data?

## Setup

Kernel on GPU 1 (RTX PRO 6000 Max-Q), headed Chrome 147 on the GPU workstation with the
NVIDIA Vulkan adapter (`adapter.info`: nvidia / blackwell, subgroup 32),
JupyterLab on localhost. Real pointer drags through CDP `Input` events (hold the
middle thumb, 20 steps 150 ms apart for latency, then 120 moves at 60 Hz for
the frame rate); a frame is a new paint stamp of the phase (FFT) panel. The
driver uses raw CDP only: Playwright enables the DevTools Network domain, which
alone slowed a 9.4 MB WebSocket message from 11 ms to 70 ms and made the
transport look 2 to 3 times worse than it is.

## Where the time went (C10, phase method, per update)

| Stage | Before | After |
| --- | ---: | ---: |
| browser send to Python handler | 2 ms | 2 ms |
| reconstruction (`preview`) | 25-45 ms | 25-45 ms (unchanged, other work) |
| `tobytes` + trait send | 1.5 ms | 1.5 ms |
| Comm transfer, kernel to browser model (9.4 MB) | 26-28 ms | 26-34 ms |
| phase statistics (worker copy + CPU histogram) | 41 ms | GPU, 5 ms incl. upload |
| colormap, offscreen canvas, ImageBitmap, drawImage | 11 ms | one render pass, < 1 ms |
| FFT panel after the phase (CPU window/pad, FFT up/readback, CPU magnitude) | +165 ms | +1 to 2 ms |
| widget re-renders per drag frame | 3 to 4 (about 11 ms each) | 0 to 1 |

## Changes

- Display statistics and the FFT panel run on the colormap engine's device
  (`js/showptycho/gpuFrame.ts`): one upload of the phase, then min/max/sum and
  the 1024-bin percentile histogram, mean removal, Hann window and zero pad,
  the 2048 x 2048 FFT, shifted log1p magnitude and its histogram. Only two
  4 KB records come back. The phase panel paints as soon as its record lands.
- Panels paint straight from the float32 slot into their own WebGPU canvas
  (colormap, contrast window, zoom and pan in one pass; nearest sampling, or an
  area average when smoothing a shrunk image). No ImageBitmap or offscreen
  canvas per frame; contrast, colormap, zoom and pan never reach Python.
- One message per drag frame: an identical request already on its way is not
  sent again, and Python does not echo regularization strengths the browser
  sent. The release request always runs (it is the exact image).
- A drag re-renders only the dragged slider (aberration, regularization and
  higher-order sliders hold their own value until release); the busy mark
  clears after 200 ms quiet instead of toggling twice per frame; the FFT host
  copy alternates two buffers instead of allocating 16 MB per frame.

## Fidelity

Every update is the full 3x Auto float32 phase; nothing is binned, cropped or
sent in reduced precision. On a real 1536 x 1536 frame the GPU statistics give
the same 1 to 99 percentile window as the CPU reference (88 of 2.36 M pixels
fall in a neighbouring 1/1024 bin from float32 bin-edge rounding); the GPU FFT
magnitude differs from the previous display path by at most 3.3e-4 on values
up to 8.5. A unit test checks that the 256 display bins rebuilt from the 1024
percentile bins equal `computeHistogramFromBytes`.

## Per control (real drags, GPU 1)

| Control | kernel ms | latency before | latency after | fps before (phase / FFT) | fps after |
| --- | ---: | ---: | ---: | ---: | ---: |
| C10 phase | 45 | 160 | 87 | 9.3 / 4.3 | 17.2 / 17.2 |
| C12 phase | 45 | 156 | 89 | 11.3 / 5.0 | 16.7 / 16.7 |
| phi12 phase | 45 | 141 | 87 | 11.6 / 4.8 | 16.9 / 16.9 |
| C21 phase | 45 | 143 | 86 | 6.4 / 4.5 | 15.2 / 15.2 |
| C30 phase | 45 | 165 | 82 | 5.7 / 4.5 | 16.8 / 16.8 |
| phase regularization | 2 | 93 | 43 | 6.1 / 4.7 | 27.1 / 27.1 |
| C10 complex | 61 | 151 | 105 | 9.3 / 4.4 | 12.6 / 12.6 |
| phase regularization, complex | 5 | 99 | 51 | 7.8 / 4.8 | 25.8 / 25.9 |
| amplitude regularization, complex | 5 | 98 | 49 | 7.2 / 4.7 | 25.0 / 25.0 |
| C10 SSB | drag path | 128 | 82 | 8.9 / 4.8 | 9.5 / 9.5 |
| C12 SSB | drag path | 121 | 84 | 9.0 / 5.1 | 9.5 / 9.5 |
| C21 SSB | 43 | 158 | 84 | 6.9 / 4.8 | 16.6 / 16.6 |

## What bounds the rest

With a 2 ms reconstruction an update now takes about 45 ms and a drag reaches
26 to 27 fps. The Jupyter Comm path is the floor: a bare 9.4 MB `widget.send`
takes 25 ms one way and 33 ms per frame back to back (about 30 fps), against
11 ms for the same bytes over a plain Tornado WebSocket in the same Chrome.
Chrome receives, JupyterLab copies and deserializes (4 to 6 ms of main thread
per frame) and ipywidgets sets the model. Other rows follow the reconstruction
time. Over the Thunderbolt link to the viewing laptop, plain HTTP moved 3.0 GB/s (3 ms per
frame) and an SSH stream about 480 MB/s (20 ms per frame), so a tunnelled
session should land near the same 30 fps Comm ceiling.

## Not done

- Lossless compression of the float32 phase: not tried. The browser has no
  fast native decoder for a float-aware codec; it would have to beat 25 ms.
- Splitting a frame into many messages: 64 KB chunks halved a raw WebSocket's
  time per frame in a DevTools-attached page (71 to 36 ms); not measured
  through Jupyter.
- Trait vs `widget.send(buffers=...)`: same 25 ms for 9.4 MB, so the trait stays.
