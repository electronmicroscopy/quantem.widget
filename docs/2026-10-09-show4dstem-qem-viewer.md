# Show4DSTEM over .qem acquisitions in the browser (2026-10-09)

## Question

Can the Show4DSTEM WebGPU folder open a series of `.qem` acquisitions (integer
counts, ANS-coded, authenticated per 64 MiB chunk) with no kernel, show which
acquisition is loading and paint each panel as soon as it is resident, while
every count stays exact?

## Change

- `show4dstem.export.write_qem_bundle(paths, out_dir)` (and `quantem show4dstem
  a.qem b.qem --html`) links the files into `out_dir/qem` by relative path and
  writes the viewer, `index.html` and `Show4DSTEM.command`. Only the headers are
  read, for the shared scan and detector shape.
- Served, the page streams each file by HTTP range (`qemHttpFiles`); opened
  from disk, it asks for the files (picker or a dropped folder).
  `RansResidentSeries` authenticates and decodes one acquisition at a time,
  three files ahead, and publishes them in file order.
- Each published acquisition adds its panel; the others keep a "Loading" tile
  in their grid position (four fixed columns), and the status line names the
  acquisition, its file and its chunk progress.
- Hovering a resident panel reads the exact uint32 count under the pointer
  from the GPU (four bytes per read, latest position only).
- Detector interaction: the center of a small rect or ring moves it (the edge
  hit margins used to cover it). The range server answers HEAD without a body,
  keeps connections alive and sends file bytes with `sendfile`.

## Setup

- Data: a seven-tilt acquisition, 512 x 512 scan, 192 x 192 detector, uint16
  counts, one `.qem` file per tilt (1.14 to 1.30 GB, 8.3 GB in all), written
  with the declared hot-pixel correction applied.
- Hosts: a Linux workstation (NVIDIA RTX PRO 6000 Blackwell, Chrome 147
  headed, Vulkan, adapter `nvidia / blackwell`) and an Apple silicon Mac
  (Chrome 154 headless, adapter `apple / metal-3`) reading the same files from
  the workstation's disk over NFS. The adapter was asserted before every run.
- Served: the folder's own range server (`Show4DSTEM.command` server; Python
  3.13 on Linux, `/usr/bin/python3` 3.9 on the Mac). Local: the page opened from
  disk and given the seven files through its file input.
- Timing: seconds from page navigation (or from handing over the files) to the
  first and to all seven painted panels, read per animation frame in the page
  (a panel counts once its loading tile is gone). GPU memory: the Chrome GPU
  process (nvidia-smi) on Linux, the whole machine's GPU memory in use (ioreg)
  on the Mac.
- quantem.gpu at the `port-qem-streaming` commit 80e64eb. Before: `dev` has no
  `.qem` viewer, so the reference is the private demo viewer exported on
  2026-10-08 before this work (it loads every file before painting and only
  from local files).
- Exactness: BF (circle, 113 px), the BF disk dragged off axis (111 px), ADF
  (annulus 20 to 80 px, 18 836 px) and a 4 x 4 px rect, each read back from the
  viewer for all seven tilts (1 835 008 values per detector), against an
  independent CUDA decode of the same files summed in int64; hover readouts on
  every panel against the same sums.

## Results

Three runs each, seconds:

| Host | Viewer, source | First panel | All seven panels | GPU memory peak |
|---|---|---|---|---|
| Linux NVIDIA | this change, served | 4.38, 4.38, 4.91 | 19.66, 19.21, 19.40 | 11.1 to 11.2 GB (idle 0.4) |
| Linux NVIDIA | this change, local files | 3.69, 3.37, 3.44 | 18.95, 17.94, 18.09 | 11.1 to 12.0 GB |
| Linux NVIDIA | demo of 2026-10-08, local files | 146.6, >150, >150 | 146.6, >150, >150 | 16.1 to 18.3 GB |
| Mac Metal | this change, served | 2.37, 2.09, 2.08 | 11.53, 11.02, 11.37 | +5.6 to +6.5 GB |
| Mac Metal | this change, local files | 2.20, 1.89, 1.79 | 11.91, 10.06, 9.56 | +5.5 to +6.4 GB |
| Mac Metal | demo of 2026-10-08, local files | 129.7 (one run) | 129.7 | +10.8 GB |

The first acquisition (1.19 GB), in seconds: headers 0.5 (Linux) and 0.3
(Mac); reading, authenticating and staging 2.2 to 2.3 and 0.8 to 1.1; the
decoder checkpoints 0.75 and 0.36; first virtual image painted 0.5 and 0.6
later. The other six follow at about 2.5 s (Linux) and 1.5 s (Mac) each.

Every one of the 4 x 1 835 008 values per host equals the CUDA reference, and
so do all hover readouts (34 on Linux, 30 on the Mac; a pixel under a panel's
resize grip shows none). The browser masks the dragged disk with float64
distances and the widget's Python masks with float32; at the dragged center
(59.999998, 130.000002) they differ by one boundary pixel.

A 4 x 4 px rect pressed at its center and dragged by (10, 15) px: `dev` resized
it (to 34 x 47 and 34 x 24 px) and left the center; this change moved the
center to (102, 104) and kept the size, on both hosts. The seven files dropped
on a viewer opened from disk (a browser-level drag): this change loaded them
(18.9 s Linux, 9.9 s Mac); `dev`'s HDF5 viewer ignored the drop.

## Conclusion

The first panel paints after one acquisition, 2 to 5 s, instead of after the
whole series, and the series is resident in 10 to 20 s instead of 130 s or
more, with about 60 % of the GPU memory of the earlier demo. Reading, authenticating
and staging the files dominates on both hosts; the decoder checkpoints are
second. Every count shown or hovered is exact.

## Not taken

- Benchmark hooks of the private demo (`__QT_RANS_PROGRESS`,
  `__QT_RANS_PROFILE`, `__QT_COMPARE_SUBMISSION`, `__QT_COMPARE_MAX_INFLIGHT`,
  `__sh4d.compareDisplaySamples`): measurement globals; the submission global
  was shared by every widget on a page, so one viewer's paint could be credited
  to another's request.
- Re-using finished panels while a series loads: the quantem.gpu series keeps
  each acquisition's mask, so a refresh after a new acquisition decodes only
  the new one; the other panels cost one image copy each.
- Display normalization to means: this viewer shows sums, as the HDF5 viewer
  does, and the hover reads the same sums exactly.
