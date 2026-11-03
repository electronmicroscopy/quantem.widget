# quantem.widget rc42 integration: measurements (2026-10-11)

Branch `integrate-rc42`: the TypeScript cleanup and the Python cleanup merged on
the released rc41 commit (`6bb0281`), the requests each cleanup left for the
other side, and a merge of the signed-integer scan-ROI fix. This page records
what was measured during the integration, the numbers, and what was tried and
dropped. Every timing was taken on one NVIDIA workstation GPU that other jobs
kept at 98 to 100% utilization, with a host load average of 14 to 18, so each
comparison alternates the two trees back to back and states its spread.

## 1. One detector message, one reduction

Question: how many times does the kernel reduce the virtual image for one
detector drag message, and what does a drag step cost?

The frontend saves `roi_center_row`, `roi_center_col` and the pair `roi_center`
in one message. traitlets applies all three and then calls the observer once
per changed trait, so `_on_roi_change` ran three times with the same detector.

Setup: the public gold acquisition, 256 x 256 scan, 192 x 192 detector, loaded
encoded with `quantem.gpu.io.load`; `Show4DSTEM(acquisition)`; 40 drag steps of
the circular bright-field detector applied with `widget.set_state({...})`, the
call the comm handler makes; median time per message, three rounds after a warm
round; reductions counted by wrapping `_compute_virtual_image_from_roi`.

| Tree | Reductions per message | ms per message (three rounds) |
|---|---|---|
| rc41 with quantem.gpu rc15 (released wheels) | 3.00 | 50.2, 50.2, 49.1 |
| merged tree before the change, pass 1 | 3.00 | 50.9, 50.8, 50.7 |
| after the change, pass 1 | 1.00 | 18.1, 17.9, 18.0 |
| merged tree before the change, pass 2 | 3.00 | 50.8, 49.3, 50.7 |
| after the change, pass 2 | 1.00 | 17.1, 13.8, 13.8 |

In a quieter minute the same script read 22.7 to 26.3 ms before the change.
The image after the last drag step has the same SHA-256 before and after
(`cf3fdb6c4eecb847...`): the same detector, reduced once.

The diffraction panel's Reset wrote the column and the row in two saves: two
messages, two reductions, 45 to 47 ms. It now writes one message: one
reduction, 18.5 to 23.9 ms.

In a live JupyterLab (headed Chrome, NVIDIA adapter) one click that moves the
detector recorded `[[12, 3], [12, 3], [12, 3]]` reductions in the kernel before
the change and `[[12, 3]]` after
(`test_detector_click_and_reset_each_reach_the_kernel_as_one_reduction`).

Change: `_on_roi_change` remembers the detector (active, mode, center, radii,
width, height) of the last notification and returns when it has not changed, so
any message that changes several ROI traits reduces once (a drag, an annulus
edit of both radii, a rectangle resize). `_on_roi_center_change` only splits the
pair into the row and column, inside `hold_trait_notifications`, so the detector
never moves row first and column second.

Rejected:

- Sending only `roi_center` from the frontend. The kernel would then write the
  row and column back to the browser after every step, and a late echo could
  move the model's row and column behind a drag that has gone on.
- A second trait holding the detector tuple, observed for the reduction.
  traitlets would drop the duplicate notifications by itself, but it adds a
  trait and two observers where a four-line comparison does the same.
- Overriding `set_state` to reduce once after a message. It needs a flag set
  and cleared around the call, which an exception would leave set.

## 2. Static scale bar labels against the browser

`render/figure.py` is the Python port of `js/figure.ts`. The TypeScript cleanup
taught the browser table `pixels` (shown as px) and the meter, millimeter and
micrometer spellings; the Python port did not have them, so a static render
could print "50 pixels" or "0.00 m" under a bar the browser labels "50 px" or
"1 nm". The port now holds the same table in Angstrom.

A sweep of 4.4 million labels (22 unit spellings, values from 1e-9 to 1e9) of
the old table against the new one differs in one case: a bar of 1e-7 mm. The
table in nm computed 1e-7 x 1e6 = 0.0999... nm and printed "100 pm"; in
Angstrom 1e-7 x 1e7 is exactly 1 and the label is "1 Å", which is what the
browser prints. Both test suites hold that case.

The browser had the mirror case. `roundToNiceValue` built its power of ten
with `Math.pow(10, n)`, and V8 returns 0.00009999999999999999 for
`Math.pow(10, -4)`, so a 1e-4 um bar came to 0.9999999999999999 Å and was
labelled "100 pm" where Python, whose `10 ** -4` is exact, prints "1 Å". The
browser now parses the power from its decimal text (`Number("1e-4")` is
exact). Both test suites hold `(1e-4, "um") -> "1 Å"`.

## 3. Spot timings against rc41

Released wheels (quantem.widget rc41, quantem.gpu rc15) against the integration
tree with quantem.gpu `d1a01af`. Minimum of five runs after a warm run, two
alternating passes, the smaller of the two; "focused" rows repeat a path with
12 to 40 runs in three alternating passes and give the range of the minima.

| Path | rc41 | rc42 tree |
|---|---|---|
| Show2D construct 2048 x 2048 | 19.6 ms | 20.4 ms |
| Show2D construct gallery 8 x 1024 x 1024 (focused) | 66.7 to 73.5 ms | 66.1 to 76.2 ms |
| Show2D `save_image` PNG 2048 x 2048 | 776 ms | 450 ms |
| Show2D `export_html` float32 2048 x 2048 | 201 ms | 130 ms |
| Show2D `export_html` uint8 2048 x 2048 | 74.4 ms | 48.0 ms |
| Show3D construct 50 x 1024 x 1024 (focused) | 320.0 to 325.6 ms | 325.3 to 327.1 ms |
| Show3D `save_image` PNG | 112 ms | 103 ms |
| Show3D `export_html` uint8 50 x 1024 x 1024 (focused) | 1.88 to 2.56 s | 1.74 to 3.00 s |
| Show3DSlices construct 128^3 (focused) | 4.11 to 4.58 ms | 4.32 to 4.59 ms |
| Show1D construct 8 x 4096 (focused) | 1.44 to 1.47 ms | 1.45 to 1.47 ms |
| Show4DSTEM construct, dense 64 x 64 x 96 x 96 | 99.2 ms | 92.5 ms |
| Show4DSTEM `export_html` uint8, dense | 3.03 s | 3.04 s |
| Show4DSTEM construct, encoded gold 256 x 256 | 304 ms | 220 ms |
| Show4DSTEM detector drag message, encoded | 90.2 ms | 21.6 ms |
| Show4DSTEM scan position move, encoded (median) | 21.1 to 25.0 ms | 22.1 to 23.5 ms |
| Show4DSTEM `export_html` encoded gold, detector bin 2 (180 MB, four runs) | 49.0 to 87.2 s | 51.7 to 69.7 s |

The Show3D uint8 page is 72.0 MB against 70.8 MB: the page now carries its
runtime so it opens with no network.

CPU display paths of the frontend, the same benchmark file run under vitest in
both trees (minimum of nine runs, four alternating passes, range of the minima;
the output digest is identical in every row):

| Path | 6bb0281 | rc42 tree |
|---|---|---|
| `fft2d` forward 1024 x 1024 | 43.3 to 47.3 ms | 42.9 to 48.4 ms |
| FFT magnitude and shift 1024 x 1024 | 49.3 to 51.9 ms | 49.2 to 51.5 ms |
| display filter gaussian sigma 4, 1024 x 1024 | 54.7 to 55.2 ms | 54.3 to 68.9 ms |
| display filter anscombe sigma 4, 1024 x 1024 | 60.1 to 75.8 ms | 61.7 to 76.9 ms |
| display filter gaussian bin 2, 1024 x 1024 | 16.5 to 16.9 ms | 16.6 to 16.8 ms |
| frequency filter lowpass 1024 x 1024 | 143.1 to 151.5 ms | 153.0 to 162.9 ms |
| `computeStats` 2048 x 2048 | 26.7 to 28.7 ms | 26.8 to 40.9 ms |
| `findDataRange` 2048 x 2048 | 9.0 to 9.2 ms | 9.0 to 14.6 ms |
| `percentileClip` 2 to 98%, 2048 x 2048 | 12.8 to 12.9 ms | 12.7 to 16.7 ms |

The frequency filter row is the one whose ranges do not overlap. The filter was
then timed alone (three rounds of 25 runs, two alternating passes): 156.5 to
163.0 ms on `6bb0281` and 153.9 to 161.3 ms on the rc42 tree, and the zero-pad
and crop it gained for images that are not a power of two cost 1.1 ms of that.
The gap in the mixed run comes from the benchmarks that ran before it in the
same process, not from the filter.

Conclusion: no path is slower beyond the spread of its own repeats; the detector
drag, `save_image`, the Show2D exports and the encoded construction are faster.

Rejected: a power-of-two branch in `applyFrequencyFilterCPU` that skips the pad
and the crop. It would save about 1 ms of 155.

`Show2D.save_image` writing the raw PNG of a 2048 x 2048 image, peak Python
memory by `tracemalloc`: 208 MiB on rc41, 112 MiB on the rc42 tree.

## 4. A frame-rate test that follows the host load

`test_show3d_stats_follow_the_moving_average_while_playing` asks for more than
60 animation frames in 2.5 s. On a virtual display under load the browser
delivers fewer. Run alternately on the released wheels and on the rc42 tree
under the same load, it failed with 58 and 55 frames on rc41 and with 34 on
rc42 in the first two passes, then passed on both; the standard deviations it
reads were right in every run (0.78, never the single-frame 1.37). The frame
count measures the host, not the widget.
