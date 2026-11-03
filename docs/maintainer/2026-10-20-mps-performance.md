# MPS interactivity: what got faster, why the rest is slow, how to profile (2026-10-20)

One night of work on how ShowPtycho and Show4DSTEM feel on a MacBook GPU
(Apple M5 Max, macOS 26.4, Chrome with the Apple Metal WebGPU adapter): what
changed, where the time goes now, and the browser harness that catches a
regression. quantem.gpu's side (the SSB preview kernel, the aberration search,
the detector sums) is in its `docs/maintainer/2026-10-20-mps-performance.md`.

"A 512² × 192² scan" below is one full 512 × 512 × 192 × 192 uint16
acquisition that is not public. Numbers labelled "gold" are on the public
gold scan (`bobleesj/quantem-data`, `4dstem/gold_512`) and anyone can
reproduce them with `scripts/bench_interactivity.py`.

## Summary

| Interaction (512² × 192² scan) | Before | After | Limit now |
| --- | ---: | ---: | --- |
| ShowPtycho C10 drag, native 512² | 16.5 fps | 37.4 fps | the 22 ms reconstruction |
| ShowPtycho C10 drag, 2× 1024² | 19.9 fps | 23.8 fps, then up to 29.6 fps (display pass) | the kernel's 1024² phase |
| Show4DSTEM probe drag | 60 /s, 9.6 / 14.0 ms | 60 /s, 6.1 / 7.2 ms | 60 Hz display |
| Show4DSTEM BF disk drag | 59.6 /s, 13.8 / 28.2 ms | 59.9 /s, 10.6 / 14.0 ms | 60 Hz display |
| Show4DSTEM ADF outer resize | 44.3 /s, 32.0 / 36.9 ms | 59.6 /s, 11.9 / 19.7 ms | 60 Hz display |
| Show4DSTEM preset click to image | 37 ms | 22 ms | kernel sum and transport |
| Show4DSTEM scan-ROI mean drag | 17.2 /s, 65 ms | 18.0 /s, 61 ms | the 47 ms mean-pattern kernel |

Rates are kernel updates per second; latencies are p50 / p95 from the pointer
event to the paint of the image it caused. Drags were scripted at 120 pointer
events per second on a 60 Hz display, with every GPU job under one lock.

A later display pass (the frame statistics, renders and presents below) was
measured as before / after pairs, both builds in one GPU lock hold, on the
60 Hz display while the screen was shared (WindowServer 21 to 50 % of the
GPU), so its absolute rates sit below the table's:

| Interaction (512² × 192² scan) | Before | After |
| --- | ---: | ---: |
| ShowPtycho C10 drag, native, same hold | 19.9 fps | 33.9 fps |
| ShowPtycho C10 drag, 2×, same hold | 14.2 fps | 22.7 fps |
| ShowPtycho C10 drag, 2×, median of 3 runs (range) | 21.0 fps (14.2 to 26.8) | 26.7 fps (22.7 to 29.6) |
| ShowPtycho widget renders per 2× phase | 5.1 | 3.4 |
| Show4DSTEM BF disk drag, latency p50 / p95 | 12.6 / 16.6 ms | 9.5 / 16.0 ms |
| Show4DSTEM ADF outer resize, latency p50 / p95 | 11.2 / 29.2 ms | 12.3 / 26.4 ms |
| Show4DSTEM widget renders per probe / BF / ADF update | 4.0 / 6.5 / 5.9 | 2.0 / 2.0 / 2.0 |
| Show4DSTEM presents per BF / ADF image | 2 / 2 | 1 / 1 |

## What we optimized and why

### ShowPtycho

- **Two slider requests in flight.** With one request the kernel idled for
  the whole browser round trip (2.7 ms at native, 8.5 ms at 2× for a 4 MB
  float32 frame) before it heard the next value. It now gets the next
  request while the previous phase is on its way; newer values still wait
  as one latest-wins request and the release commit is never dropped. With
  quantem.gpu's faster kernel, native went from 25.5 to 34.5 fps.
- **One display FFT at a time.** Starting a display FFT per phase and
  discarding all but the newest starved the FFT panel once phases arrived
  faster than a 1024² FFT (5 fps at 2×). One FFT runs at a time and the
  newest waiting phase goes next: FFT panel 5 → 22 fps at 2×.
- **FFT preparation in the worker.** Mean subtraction, the Hann window,
  padding, interleaving and the magnitude split moved into the display
  worker with the same arithmetic (unit-tested bit for bit), freeing the
  input thread to dispatch the next request: 2× phase 22.0 → 23.7 fps.
- **Rotation drags skip the loss.** A rotation slider step computed the
  phase and the variance loss; drags now compute only the phase (the 0.8 ms
  geometry update plus the preview) and the release computes the loss.

- **Frame statistics off the input thread.** Each phase was binned three
  times on the input thread: the 1-99 percentile window and the histogram
  bins of the phase (1024 and 256 bins, about 6 ms at 1024²) and the
  percentile window of the FFT magnitude (about 3.5 ms). The phase is now
  binned in a display-statistics worker and the FFT magnitude in the FFT
  worker next to its magnitude, both from the input-thread functions' own
  source, so every window and bin is bit-identical; a contrast change walks
  the cached 1024 bins.
- **No widget render per frame.** A painted frame set the panel versions and
  data ranges in widget state, which re-rendered the whole widget (two or
  three times per phase with the FFT). The frame now goes to a per-panel
  store that only the histograms and the cursor readouts read; the widget
  renders when a canvas changes size. Setting `busy`, the preview error or
  the viewed pin to the value they already held also ran the widget, on
  every request and reply; those setters now skip unchanged values.
- **Slider state once per animation frame.** The dragged aberration's value
  and label update once per animation frame and at release; every pointer
  event still sends its request with its own value.
- Together, on the 512² × 192² scan at 60 Hz, under the same GPU lock hold
  as the build before them: native 19.9 → 33.9 fps, 2× 14.2 → 22.7 fps
  (WindowServer at about 22 % of the GPU); over all runs the 2× median is
  21.0 → 26.7 fps, best 29.6, and the request p95 fell from up to 500 ms to
  70 to 175 ms. On gold both are kernel-bound and unchanged. The painted
  phase and FFT canvases are bit-identical to the previous build on three
  fixed gold frames (2×, 2× at another rotation, native).

### Show4DSTEM

- **Memoized toolbars and control rows.** Every frame of a drag re-rendered
  the whole widget, and most of that time was MUI/emotion restyling blocks
  whose inputs had not changed. React time per render 2.3 → 1.1 ms; ADF
  resize 49.5 → 59 updates/s.
- **One render per diffraction pattern.** The pattern's statistics are
  computed in the render that brings it and painted in a layout effect:
  renders per pattern 4.9 → 4.0, receive-to-paint 3.1 → 2.2 ms.
- **The probe pattern once per scan-position message.** Moving the probe
  set row and column as two trait changes and read the pattern twice:
  kernel time per move 1.06 → 0.55 ms.
- **A preset click is one message.** The detector traits and the image of a
  preset now travel together (8 → 2 messages including the echo), and a
  preset image computed on demand is cached like a precomputed one: repeat
  click to image 29.4 → 22.1 ms, click to kernel reply 21.6 → 13.2 ms.
- **One present and fewer renders per virtual image.** A new image was
  presented twice: once from the previous GPU upload by the render that
  brought it, then again after the upload had waited a microtask for the
  device and bumped a version to re-render. A mirroring effect copied the
  trait into state, and the stats bar's and histogram's values (including
  the GPU histogram readback) were widget state. Now the upload runs before
  the histogram and present effects of the render that brought the image,
  the browser-computed image is derived from the trait it was published
  over, and the stats bar and histogram read their own store: on the
  512² × 192² scan, widget renders per bright-field update 6.5 → 3.9,
  annular 5.9 → 3.0, presents 2 → 1, bright-field latency p50 12.6 → 9.5 ms.
- **One render per animation frame while dragging.** A bright-field drag set
  the overlay center on every pointer event and again in the animation frame
  that writes `roi_center`; the frame now does both. Hover states set on
  every pointer move, and the FFT crop and line profiles cleared on every
  image while unused, skip unchanged values (`useChangedState`). Widget
  renders per update: probe 4.0 → 2.0, bright-field 3.9 → 2.0, annular
  3.0 → 2.0.
- quantem.gpu's incremental exact masked sum (a dragged mask updates the
  previous image by the few hundred pixels that changed) cut the kernel time
  per detector update from about 6 to 2 ms.

## Where the time goes now

- **ShowPtycho at 2× is now kernel-bound.** With two requests in flight the
  request latency (63 to 65 ms p50) is twice the kernel's time per 1024²
  phase, and the panels paint every phase it delivers; the input thread is
  idle about half of a drag. What remains on it per frame is copying: the
  4 MB phase into the GPU and the workers, the FFT spectrum out of the GPU
  and back. A slider drag still renders the widget once per animation frame
  for the thumb and its label.
- **Show4DSTEM drags.** One present per image and two widget renders per
  update on probe, bright-field and annular drags: one in the animation frame
  that writes the pointer's trait (and moves the overlay), one for the image
  or pattern the kernel sends back.
- **Scan-ROI mean pattern.** quantem.gpu averages the patterns in the scan
  circle in one pass per (detector pixel, scan block): 47 ms for a 76.8 px
  radius, so the drag paints about 18 times per second.
- **The gold scan as `io.load` keeps it.** Its masters store uint32. Until
  quantem.gpu stored uint32 counts that fit 16 bits as uint16 (the same
  day), they sat in the generic bit-lane resident, where every detector sum
  decodes the whole scan: a Show4DSTEM bright-field drag on
  `io.load(gold, hot_pixel_correction="median")` updated 0.3 times per
  second (2.4 s per update). They now load as the uint16 count resident and
  the same drag runs at the 60 Hz display rate (59 to 93 updates per second
  over two runs, 13 to 16 ms latency). The harness drives Show4DSTEM on the
  loaded data directly.
- **ShowPtycho's automatic sampling on gold.** `upsampling_factor="auto"`
  chooses 2× for gold's 0.5 Å step. quantem.gpu's 2× preview cost 72 ms, not
  because it lacked the half spectrum but because at this step the
  bright-field support covers the whole 2× grid and the two fine rows of
  each source row read the same spectra twice. Computing both rows in one
  thread brought it to about 54 ms, and the default drag from about 10
  to 11 or 12 frames per second. Native sampling is 27 ms per preview and
  paints about 20 to 23 frames per second, below the 2 / (27 ms) a
  kernel-bound queue of two would allow; the request latency (about 60 ms)
  was measured while WindowServer used 20 to 48 % of the GPU for screen
  sharing, which slows both the kernel and the browser.

## Things that look like a regression but are not

- **SwiftShader.** Chrome's CPU renderer runs WebGPU without a GPU; every
  timing on it is meaningless. The harness logs `adapter.info` and stops
  when the adapter is `swiftshader`, Google's, or a fallback.
- **Another GPU client.** WindowServer uses 8 to 35 % of the GPU while the
  screen is shared or streamed, `mediaanalysisd` analyses new photos and
  videos on the GPU, and another notebook or agent may be running Metal
  work. The report lists the busiest other GPU clients before and after the
  run; `compare` warns about any at 10 % or more.
- **Timings taken without the lock.** On a shared machine run the harness
  under the same GPU lock as every other job.
- **A hidden or occluded window.** Chrome throttles timers and animation
  frames in background windows. The harness starts Chrome with throttling
  disabled; keep its window on screen anyway.
- **The 60 Hz cap.** Rates near 60 per second are display-bound; a change
  from 59.7 to 59.2 is noise, which is why `compare` also requires an
  absolute change of 2 per second (3 ms for latencies).
- **Playwright's Network domain.** It copies every websocket frame to the
  driver; large messages then take minutes (see the performance notes).
  The harness uses raw CDP with the Page, Runtime and Input domains only.

## How to profile

### The interactivity harness

```bash
npm ci && npm run build
python scripts/bench_interactivity.py run --out interactivity.json   # about 2 minutes
python scripts/bench_interactivity.py compare interactivity.json
```

`run` starts its own JupyterLab and headed Chrome (free ports, a temporary
profile and settings), opens one notebook per widget on the cached gold scan
(`--download` fetches it), drives the widgets with CDP pointer events at
120 Hz, writes the report and stops and deletes everything it started.
Chrome opens on the main display and the report records its measured refresh
rate (`display_hz`): with a 60 Hz and a 120 Hz display attached, the same
build measured different rates, latencies and render counts depending on
where the window opened, and `compare` warns when the baseline's rate differs.
`run --save-baseline` re-records `scripts/baselines/interactivity-<chip>.json`;
on a shared machine `merge a.json b.json c.json --save-baseline` records the
median of several runs, metric by metric.
All timing is in the page (`performance.now`), from instrumentation injected
before JupyterLab loads: widget messages on the kernel websocket (with
ShowPtycho's request ids), canvas presents and paints, and trusted pointer
events.

| Metric | Meaning |
| --- | --- |
| `showptycho.<native,2x>.phase_fps`, `fft_fps` | panel paints per second during a 6 s C10 drag |
| `showptycho.*.request_latency_p50_ms`, `_p95_ms` | `ssb-preview` request to the phase with the same id |
| `showptycho.*.requests_per_s` | requests the slider sent |
| `show4dstem.<probe,bf_drag,adf_resize>.updates_per_s` | kernel image updates per second during a 3 s drag |
| `show4dstem.*.paints_per_s` | paints of the panel that shows the update |
| `show4dstem.*.latency_p50_ms`, `_p95_ms` | pointer event to the paint of the update it caused |

`compare` exits 1 when a rate drops, or a latency rises, by more than 20 %
(50 % for p95 latencies, which rest on a handful of samples; override with
`--tolerance show4dstem.bf_drag.latency_p50_ms=0.3`) and by more than 2 per
second or 3 ms.

### Gold baseline on the M5 Max

| Interaction (gold) | Rate | Latency p50 / p95 |
| --- | ---: | ---: |
| ShowPtycho C10 drag, native | 23.0 phase / 22.9 FFT fps | 57 / 63 ms (request) |
| ShowPtycho C10 drag, 2x | 12.5 phase / 12.5 FFT fps | 110 / 117 ms (request) |
| Show4DSTEM probe drag | 59.9 /s | 6.5 / 7.6 ms |
| Show4DSTEM bright-field disk drag | 59.5 /s, one present per image | 15.0 / 16.8 ms |
| Show4DSTEM annular outer resize | 58.6 /s, one present per image | 10.1 / 15.0 ms |

Recorded with `merge` on 2026-10-20 from the two runs, of four on the browser display-pass build, whose
other GPU clients stayed low (WindowServer 4 to 6 %, a dictation server 6 %): WebGPU adapter apple / metal-3
(subgroup 32), the window on the 60 Hz main display (`display_hz` 59.1), 1512 × 982 viewport at device scale
2, each run under the shared GPU lock, Show4DSTEM on the gold scan as `io.load` stores it. With two runs
`merge` keeps the better value of each metric, so the p95 latencies are optimistic until a third quiet run
is merged. The previous baseline counted two presents per virtual image (about 118 paints per second at 59
updates); one of them showed the previous image.

Record or compare a baseline only from runs whose reported other GPU clients are low. Screen sharing
(WindowServer) and a dictation server run outside the GPU lock and took 40 to 70 % of the GPU during parts
of the night; runs taken then are slower in both the kernel and the browser, and `compare` prints a warning
for any client at 10 % or more. The probe drag's p95 is one or two frame intervals (about 8 or 15 ms) from
run to run, so a 15 ms run can fail its 50 % tolerance against an 8 ms baseline: rerun, or pass
`--tolerance` for that metric.

### Reading a failure

1. Read the warnings: another GPU client on either side means rerun first.
2. Rerun the harness; a real regression repeats.
3. Compare with quantem.gpu's `tests/benchmarks/mps_ssb.py` on the same
   checkout. If the kernel path (`preview_*`, `detector_*`) moved too, the
   cause is there; if only the browser numbers moved, look at the widget's
   JS (renders per update, main-thread work) or the comm messages.
4. ShowPtycho: a lower `phase_fps` with an unchanged request latency points
   at the browser; a higher latency with unchanged kernel times at the comm
   path (message size, extra round trips).
5. A deliberate trade re-records the baseline in the same commit, with the
   reason in the message.

### Deeper profiles

- Count React renders and their cost with `React.Profiler` around the
  blocks of a widget; a drag should cost one render per update. Without a
  rebuild, a page script that defines `window.__REACT_DEVTOOLS_GLOBAL_HOOK__`
  (`inject` returning an id, `onCommitFiberRoot` recording a timestamp) before
  JupyterLab loads counts every commit per React copy; the widget bundles
  carry React 19 and JupyterLab React 18. A commit is not a widget render: a
  store-backed histogram re-rendering alone also commits.
- `Profiler.start` / `Profiler.stop` over CDP during a scripted drag gives a
  sampled JS profile; restyling (emotion `sx`) at the top means a block that
  should be memoized is not.
- Log `ssb.preview` calls in the kernel (wrap the session's method in the
  notebook) to separate kernel time from transport and paint.
