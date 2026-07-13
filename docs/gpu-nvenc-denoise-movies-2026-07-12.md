# GPU NVENC Denoise Movie Pipeline - 2026-07-12

## Summary

Added `scripts/make_movies_gpu.py` for the denoise-paper full-resolution
comparison movies on mjgoat.  The script uses NVIDIA PyNvVideoCodec to feed
NV12 frames backed by CuPy GPU arrays directly to NVENC on GPU 0.  FFmpeg from
`imageio-ffmpeg` is used only for stream-copy muxing of the NVENC elementary
stream into MP4; it does not encode video.

The frame math stays on the GPU after the `.npz` cache is loaded: raw-stack
percentiles, grayscale contrast scaling, `uint8` conversion, neutral NV12
chroma creation, and label compositing all use CuPy.  Text is rasterized once
per label into a small PIL mask, uploaded to GPU, and composited into a small
frame region so per-frame PIL drawing is not in the encode loop.

Update: the script now defaults to `--load-mode npz-memmap`, which parses the
ZIP local header for the uncompressed `panels.npy` member and memory-maps the
array bytes directly inside the `.npz`.  This avoids `np.load` materializing the
entire 9-panel array in CPU RAM before GPU transfer.

## Command

```bash
CUDA_VISIBLE_DEVICES=0 /home/owner/miniforge3/envs/cuda-env/bin/python \
  scripts/make_movies_gpu.py 800C_1.3Mx
```

Useful options:

```bash
--codec h264        # default; also supports hevc and av1
--qp 18             # default NVENC constqp quality
--no-labels         # skip label compositing
--tiled             # also emit a 3x3 tiled movie
--parallel-panels 7 # run all seven per-panel encoders at once
--cpu-baseline-fps  # override the default measured CPU baseline
```

## Measured Check

Environment:

- Host: mjgoat
- Python: `/home/owner/miniforge3/envs/cuda-env/bin/python`
- GPU 0: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- Encoder: PyNvVideoCodec / NVENC, verified by `nvidia-smi`
  `encoder.stats.sessionCount`

Completed production run:

```bash
CUDA_VISIBLE_DEVICES=0 /home/owner/miniforge3/envs/cuda-env/bin/python -u \
  scripts/make_movies_gpu.py 800C_1.3Mx
CUDA_VISIBLE_DEVICES=0 /home/owner/miniforge3/envs/cuda-env/bin/python -u \
  scripts/make_movies_gpu.py 800C_3.6Mx
CUDA_VISIBLE_DEVICES=0 /home/owner/miniforge3/envs/cuda-env/bin/python -u \
  scripts/make_movies_gpu.py 400C_5.1Mx
```

Results:

- All 21 per-panel MP4s were written under
  `/home/owner/publications/denoise-paper/figures/movies/<slug>/`.
- PyAV validation confirmed `h264`, `2048x2048`, and the expected frame count
  for every output: `130` for `400C_5.1Mx`, `69` for `800C_1.3Mx`, and `91`
  for `800C_3.6Mx`.
- `nvidia-smi` reported `encoder.stats.sessionCount=1` during every encode.
- GPU encode throughput, including GPU contrast scaling and GPU label
  compositing:
  - `400C_5.1Mx`: `113.92 fps` raw; `166.54-258.75 fps` denoised.
  - `800C_1.3Mx`: `111.78 fps` raw; `250.09-378.84 fps` denoised.
  - `800C_3.6Mx`: `108.74 fps` raw; `250.41-336.86 fps` denoised.
- A focused full raw-panel CPU reference using the original PIL + libx264 path
  encoded `69` native `2048x2048` frames in `3.786 s`, or `18.22 fps`.
- Against that CPU reference, the raw-panel GPU path measured about `6.0x` to
  `6.3x` faster.  Denoised panels measured about `9.1x` to `20.8x` faster on
  the same baseline.

The `.npz` cache load/decompression is separate from encode timing and remains
CPU-bound.  The encode throughput printed by the script starts when NVENC input
generation begins and excludes cache decompression.

## Faster Loader Check

The cache files are ZIP-wrapped but `ZIP_STORED`, not compressed.  Directly
mapping `panels.npy` inside each `.npz` opened all three caches in about
`0.001-0.010 s`.

A timed end-to-end run for `800C_1.3Mx` with the direct memmap loader and seven
parallel per-panel encoders completed in `7.38 s`:

```bash
CUDA_VISIBLE_DEVICES=0 /home/owner/miniforge3/envs/cuda-env/bin/python -u \
  scripts/make_movies_gpu.py 800C_1.3Mx --parallel-panels 7
```

That timing includes memmap open, raw percentile, copying the seven selected
panels to GPU, fused CUDA frame preparation, seven concurrent NVENC sessions,
MP4 muxing, and PyAV validation.

## Two-GPU Batch Scheduler

Added `scripts/make_movies_gpu_batch.py` to keep one dataset process active per
physical GPU.  The batch runner sets `CUDA_VISIBLE_DEVICES=<gpu>` for each
child process and calls `make_movies_gpu.py` with `--gpu-id 0`, so each child
sees its assigned physical GPU as local device 0.  It writes per-job logs plus
`batch_summary.json` and `batch_summary.csv`.

Production profile command:

```bash
/home/owner/miniforge3/envs/cuda-env/bin/python -u \
  scripts/make_movies_gpu_batch.py \
  --summary-dir /home/owner/publications/denoise-paper/figures/movies/_batch_profiles/20260712-codex-two-gpu \
  --gpus 0,1 \
  --parallel-panels 7 \
  --no-validate
```

Result for the three available caches:

- Total batch wall: `22.69 s` for `21` full-resolution MP4s.
- Summary CSV:
  `/home/owner/publications/denoise-paper/figures/movies/_batch_profiles/20260712-codex-two-gpu/batch_summary.csv`
- `400C_5.1Mx`: `16.70 s` batch wall, `733.6 MB` output,
  `2.19 s` raw percentile, `6.46 s` GPU preload, `6.40 s` parallel encode wall,
  `3.22 s` summed mux.
- `800C_1.3Mx`: `10.82 s` batch wall, `187.4 MB` output,
  `1.69 s` raw percentile, `3.73 s` GPU preload, `4.21 s` parallel encode wall,
  `1.46 s` summed mux.
- `800C_3.6Mx`: `11.86 s` batch wall, `343.9 MB` output,
  `1.22 s` raw percentile, `4.38 s` GPU preload, `5.06 s` parallel encode wall,
  `3.35 s` summed mux.

The bottleneck is split across host-to-GPU panel transfer, NVENC/container wall
time, raw percentile, and MP4 muxing.  The fused CUDA frame-prep kernel is not
the dominant cost at this point.

## Follow-up Bottleneck Profile

Scratch outputs were written under:

```text
/home/owner/publications/denoise-paper/figures/movies/_profile_scratch/
```

For `800C_1.3Mx` on GPU 0, H.264, full resolution, seven per-panel MP4s:

| Variant | Total | Percentile | Transfer/preload | Encode wall | Mux sum | Output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| warm H.264 baseline | `5.69 s` | `0.71 s` | `1.17 s` | `3.65 s` | `1.55 s` | `190.6 MB` |
| cached contrast H.264 | `4.90 s` | `0.00 s` | `1.49 s` | `3.23 s` | `1.44 s` | `187.6 MB` |
| worker-side transfer | `7.66 s` | `1.30 s` | `13.60 s` summed | `6.16 s` | `1.42 s` | `187.6 MB` |
| ffmpeg pipe mux | `6.41 s` | `0.85 s` | `1.55 s` | `3.83 s` | `0.00 s` | `187.7 MB` |
| cached HEVC | `6.14 s` | `0.00 s` | `2.41 s` | `3.56 s` | `0.79 s` | `161.3 MB` |
| cached AV1, 4 panels | `6.02 s` | `0.00 s` | `1.68 s` | `4.16 s` | `0.42 s` | `119.9 MB` |
| cached H.264, no labels | `6.59 s` | `0.00 s` | `3.66 s` | `2.75 s` | `1.43 s` | `187.7 MB` |

Findings:

- `--contrast-cache read-write` is a low-risk repeat-run win.  It stores the
  raw-panel 1/99 percentile window next to the output and removes roughly
  `0.7-2.2 s` of percentile work per dataset when the cache hits.
- Moving panel transfer into encoder workers is not a win.  It overlaps some
  work, but summed transfer contention grows and wall time did not improve.
- Pipe muxing is valid but slower for this workload.  The large raw panel
  back-pressures ffmpeg stdin, so muxing moves into the encode critical path.
  Keep `--mux-mode temp` as the production default.
- HEVC and AV1 reduce bytes, especially AV1, but they did not improve wall
  time for the current seven-MP4 full-resolution workflow.  AV1 also failed
  once with seven concurrent sessions, then succeeded with four, matching the
  physical encoder-engine count.
- Skipping labels is not a reliable wall-time win in this profile.  The main
  cost is still encoder/container and data movement.

The hard ceiling is physical: one GPU has four NVENC engines.  Seven concurrent
sessions keep the queue full, but they time-slice.  Two to three orders of
magnitude faster is not realistic for unchanged output semantics: seven
full-resolution MP4 containers per dataset, all frames, no downsample/crop, and
lossy video encoding.  To pursue another order of magnitude, the output contract
has to change: fewer containers, a tiled/contact-sheet video, fewer frames,
precomputed/contiguous GPU-ready outputs, or a non-MP4 artifact.

## VRAM and Slot Scheduling Check

The selected seven output panels can fit in VRAM comfortably:

| Slug | Full cache | Selected seven panels |
| --- | ---: | ---: |
| `400C_5.1Mx` | `19.63 GB` | `15.27 GB` |
| `800C_1.3Mx` | `10.42 GB` | `8.10 GB` |
| `800C_3.6Mx` | `13.74 GB` | `10.69 GB` |
| total selected |  | `34.06 GB` |

A direct probe loaded all three selected-panel stacks onto GPU 0 at once.  It
used about `36.26 GB` of VRAM and took `19.94 s` to stage the data.  That proves
VRAM capacity is not the limiter for the current three datasets.  It also shows
that naive "load everything first" is not faster by itself; the transfer cost
still has to be overlapped with encoding or reused across multiple export
variants.

The better near-term optimization is slot scheduling.  Added `--gpu-slots` to
`scripts/make_movies_gpu_batch.py`, where repeated GPU IDs intentionally allow
more than one dataset process on the same GPU.  Per-slot panel parallelism is
encoded as `gpu:parallel_panels`.

Useful profile commands:

```bash
# Current best under the observed machine load: put the largest dataset on the
# cleaner GPU 1, and run both smaller datasets concurrently on GPU 0.
/home/owner/miniforge3/envs/cuda-env/bin/python -u \
  scripts/make_movies_gpu_batch.py \
  --out-root /home/owner/publications/denoise-paper/figures/movies/_profile_scratch/concurrent_gpu1_p4 \
  --summary-dir /home/owner/publications/denoise-paper/figures/movies/_profile_scratch/concurrent_gpu1_p4/_batch_profiles/inverted_slots \
  --gpu-slots 1:7,0:4,0:4 \
  --schedule largest-first \
  --no-validate \
  --panel-transfer preload \
  --contrast-cache read-write \
  --mux-mode temp \
  --codec h264
```

Measured results:

- Manual concurrent run, largest on GPU 0 and both smaller datasets on GPU 1:
  `18.46 s` for 21 MP4s.
- Slot scheduler with the same nominal assignment later measured `21.76 s`
  because GPU 0 had an unrelated `chi2_structure_4dstem_denoise.py` process
  holding about `46 GB` VRAM and competing for the device.
- Inverted slot scheduler, largest on cleaner GPU 1 and both smaller datasets
  on busy GPU 0: `14.74 s` for 21 MP4s.

This means the scheduler should choose slots based on current GPU load, not only
GPU index.  For larger batches, inspect `nvidia-smi` first, then put the largest
datasets on the least busy GPU slots and use repeated slots for smaller datasets
when VRAM permits.
