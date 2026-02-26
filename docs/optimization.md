# Optimization

Performance documentation for the quantem.widget data pipeline, focused on 4D-STEM and 5D-STEM workflows on Apple Silicon.

## Data Pipeline Architecture

### Metal → NumPy → PyTorch MPS (the unavoidable copy)

The end-to-end data path on Apple Silicon:

```
Disk (bitshuffle+LZ4)
  → Metal compute shader decompresses into MTLBuffer (StorageModeShared)
    → np.frombuffer() wraps the Metal buffer as a numpy view (zero-copy, same pointer)
      → torch.from_numpy().to("mps") copies into PyTorch's own MTLBuffer (memcpy!)
```

The last step is a **memcpy within unified DRAM** — both the Metal decompressor's output buffer and PyTorch's MPS buffer sit in the same physical memory, but PyTorch MPS cannot adopt external Metal buffers. There is no `torch.from_metal_buffer()` API. This copy is the primary init bottleneck:

| Data size | memcpy time | Notes |
|-----------|-------------|-------|
| 0.6 GB (det_bin=8) | ~30 ms | Negligible |
| 2.3 GB (det_bin=4) | ~110 ms | Acceptable |
| 9.0 GB (det_bin=2) | ~1.2 s | Main bottleneck |
| 5.6 GB (5D, det_bin=4) | ~7.7 s | Significant but one-time |

This copy is necessary because MPS virtual imaging (`tensordot` at 22ms) is significantly faster than CPU alternatives (34ms numpy, 139ms CPU torch at full 192×192 detector). At the unbinned 256×256×192×192 size, CPU torch tensordot is 139ms vs MPS at 22ms — MPS is required for interactive drag.

**Why not skip PyTorch and stay on numpy?** For small detectors (det_bin=4+), numpy tensordot at 34ms is usable. But at full detector resolution (192×192), numpy tensordot takes 1.9 seconds — completely unusable for interactive work. MPS is essential.

**Why not write directly to a PyTorch MPS buffer?** PyTorch MPS tensors are allocated through PyTorch's internal Metal allocator. Our Metal compute shaders use their own `MTLBuffer` allocations. PyTorch has no API to wrap an external Metal buffer as an MPS tensor (`torch.from_dlpack` does not support MPS). Until PyTorch adds this, the memcpy is unavoidable.

### Raw Float32 Pipeline

All widgets send **raw float32** data to JavaScript. Normalization, log scale, auto-contrast percentile clipping, histogram computation, and colormap LUT application all happen in JS for instant interactivity. Python never pre-renders colormapped images — it only sends the raw numerical data.

Show4DSTEM uses PyTorch MPS for virtual imaging computation (`tensordot`/sparse indexing for BF/ADF/custom mask integration), keeping the heavy math on GPU while JS handles display.

## 5D-STEM Eager Loading

Show4DSTEM loads the full 5D tensor to GPU at init time. This trades slower initialization for instant frame switching during interactive work.

### How It Works

- The full `(n_frames, scan_r, scan_c, det_r, det_c)` tensor is copied to MPS at widget creation.
- `_data[frame_idx]` returns a **GPU tensor view** (0ms) — not a copy. Frame switching is instantaneous.
- Virtual imaging (`tensordot` with the ROI mask) runs entirely on GPU, so BF/ADF/custom ROI updates during drag are real-time.

### MPS INT_MAX Fallback

PyTorch MPS has a hard limit of `INT_MAX` (2^31 - 1 = 2,147,483,647) elements per tensor. When the total element count exceeds this, Show4DSTEM automatically falls back to CPU torch tensors.

**Real-world 5D dataset sizes:**

| Config | Shape | Elements | Memory | Backend |
|--------|-------|----------|--------|---------|
| det_bin=8, 10 files | 10 x 256 x 256 x 24 x 24 | 377M | ~1.5 GB | MPS |
| det_bin=4, 10 files | 10 x 256 x 256 x 48 x 48 | 1.5B | ~6 GB | MPS |
| det_bin=2, 10 files | 10 x 256 x 256 x 96 x 96 | 6.0B | ~24 GB | CPU (exceeds INT_MAX) |

### Init and Frame Switching Benchmarks

Measured with synthetic 5D data on Apple M5 (24 GB):

| Config | Init (numpy→MPS) | Global min/max | Frame switch | `auto_detect_center` sum |
|--------|-------------------|----------------|--------------|--------------------------|
| det_bin=8 (1.4 GB) | 253 ms | 137 ms | **7 µs** | 31 ms |
| det_bin=4 (5.6 GB) | 7.7 s | 177 ms | **8 µs** | 75 ms |

Frame switching is a tensor view (7–8 µs) — effectively instant. The init cost scales with tensor size, but this is a one-time cost at widget creation.

**Comparison with previous lazy loading approach:**

| Strategy | Latency per frame switch | Notes |
|----------|--------------------------|-------|
| Eager GPU (current) | **7 µs** | Tensor view, no copy |
| Lazy NumPy→MPS copy | 28 ms | `torch.from_numpy().to("mps")` per frame |
| Lazy with `.copy()` | 96 ms | Unnecessary contiguous copy overhead |

Eager loading eliminates per-frame latency entirely, making 5D time/tilt series exploration feel instantaneous.

## Virtual Imaging Performance

Show4DSTEM computes virtual images by integrating diffraction patterns over a mask (BF disk, ADF annulus, or custom ROI). The implementation uses `tensordot` with sparse indexing for small masks.

### 256×256×96×96 (det_bin=2)

| Method | MPS | CPU torch | NumPy | Notes |
|--------|-----|-----------|-------|-------|
| tensordot (BF, 317 px) | **22 ms** | 34 ms | 34 ms | Default path |
| sparse sum (BF, 317 px) | **5 ms** | 21 ms | 64 ms | Used for small masks |
| tensordot (ADF, 952 px) | **23 ms** | 34 ms | 34 ms | |
| elementwise | 127 ms | — | 222 ms | Avoided |

### 256×256×192×192 (no binning, 9.2 GB)

| Method | CPU torch | NumPy | Notes |
|--------|-----------|-------|-------|
| tensordot (BF) | 139 ms | **1,918 ms** | MPS unavailable (>INT_MAX) |
| sparse sum (BF, 1257 px) | **85 ms** | 374 ms | |

At det_bin=2, MPS sparse sum at 5ms gives ~200fps during ROI drag. Even at full detector resolution where MPS is unavailable, CPU torch sparse at 85ms is usable. **No debouncing is needed** — the user sees real-time virtual image updates as they move the detector ROI.

## IO.arina_file GPU Pipeline

`IO.arina_file` reads bitshuffle+LZ4 compressed 4D-STEM data using Metal GPU decompression on Apple Silicon.

### Double-Buffered Architecture

The pipeline uses double buffering to overlap IO and decompression:

1. **CPU** reads compressed chunk N+1 from disk
2. **GPU** decompresses chunk N via Metal compute shaders
3. These run concurrently — disk IO is fully hidden behind GPU work

The bottleneck is GPU decompression, not disk IO: decompressing 262k frames of bitshuffle+LZ4 takes ~1.5s on M5, while the 1.7 GB disk read at 8.2 GB/s SSD throughput completes in ~0.2s.

### Buffer Sizing

Compressed buffer allocation uses a conservative formula:

```
buffer_size = max(256 MB, max_frames * frame_bytes // 4)
```

The worst observed compression ratio is ~7:1. Using `// 4` (4:1 ratio) provides headroom for poorly-compressing datasets.

### Early Validation

The pipeline checks file existence before starting the GPU pipeline, failing fast for incomplete datasets rather than discovering missing chunks mid-decompression.

## Benchmarks (Apple M5, 24 GB)

### IO.arina_file Single File

SnMoS2 dataset: 262,144 frames, 192 x 192 detector pixels.

| Config | Output Shape | Memory | Load Time |
|--------|-------------|--------|-----------|
| det_bin=2 | 512 x 512 x 96 x 96 | 9.0 GB | 1.8 s |
| det_bin=4 | 512 x 512 x 48 x 48 | 2.3 GB | 1.7 s |
| det_bin=8 | 512 x 512 x 24 x 24 | 0.6 GB | 1.8 s |

Load time is dominated by GPU decompression and is nearly constant across bin factors — binning happens after decompression.

### IO.arina_folder (Multi-File 5D)

Korean sample: 12 files, ~65k frames each.

| Config | Output Shape | Memory | Load | +Show4DSTEM |
|--------|-------------|--------|------|-------------|
| det_bin=8 (10 files) | 10 x 256 x 256 x 24 x 24 | 1.5 GB | 9.5 s | 11.0 s |
| det_bin=4 (10 files) | 10 x 256 x 256 x 48 x 48 | 6.0 GB | 10.8 s | 16.3 s |

The "+Show4DSTEM" column includes widget initialization (MPS tensor copy + initial virtual image computation).

## Memory Guidelines

### Estimating Memory Usage

Memory for a 4D-STEM dataset in float32:

```
memory_bytes = scan_r * scan_c * (det_r / bin)^2 * 4
```

For 5D datasets (time/tilt series), multiply by `n_frames`.

**Examples (512 x 512 scan):**

| det_bin | Detector | Per-Frame 4D | 10-Frame 5D |
|---------|----------|-------------|-------------|
| 8 | 24 x 24 | 0.6 GB | 6.0 GB |
| 4 | 48 x 48 | 2.3 GB | 23 GB |
| 2 | 96 x 96 | 9.0 GB | 90 GB |
| 1 | 192 x 192 | 36 GB | 360 GB |

### Automatic Bin Selection

`det_bin="auto"` picks the smallest bin factor that fits in available RAM, balancing resolution against memory constraints.

### Scan Binning

`scan_bin=2` halves scan resolution in each dimension, reducing the 4D data size by 4x. This is useful for quick survey analysis before committing to full-resolution processing.

## Tips for Users

1. **Quick survey**: use `det_bin=8` for fast loading and exploration. Switch to `det_bin=2` for publication-quality analysis once you have identified regions of interest.

2. **Scan binning**: `scan_bin=2` quarters the 4D data size at the cost of halved spatial resolution. Useful for large-area surveys.

3. **Large datasets (>16 GB)**: use `det_bin=4` or higher to stay within MPS memory limits. The `det_bin="auto"` option handles this automatically.

4. **5D frame switching is instant**: the full tensor is on GPU, so scrubbing through time/tilt frames has zero latency. There is no need to precompute virtual images for each frame.

5. **Auto-center detection**: sums all diffraction frames across all scan positions (and all time/tilt frames for 5D). This may take a few seconds for large datasets but only runs once at initialization.

6. **Hot pixel filtering**: adds negligible overhead (~1% of load time). Leave it enabled unless you have a specific reason to skip it.

## Memory Management

4D-STEM datasets are large (1–10+ GB). When switching between datasets in a notebook session, you must explicitly free both the widget's GPU tensor and the source numpy array. Python's garbage collector alone is not enough — the MPS allocator caches freed buffers and does not return them to the system until `torch.mps.empty_cache()` is called.

### Using `free()`

Show4DSTEM provides a `free()` convenience method that handles the full cleanup:

```python
w.free()          # deletes MPS tensor, runs gc, flushes MPS cache
del result        # free the source numpy IOResult array
```

### Manual cleanup

If you need fine-grained control:

```python
import gc, torch

del w             # delete widget (releases reference to MPS tensor)
del result        # delete IOResult (releases numpy array)
gc.collect()      # trigger Python garbage collection
torch.mps.empty_cache()  # flush MPS allocator cache back to system
```

### Why `del` alone isn't enough

The MPS allocator maintains an internal free-list of GPU buffers. When a PyTorch tensor is deleted, the underlying Metal buffer is returned to this free-list — not to the operating system. Subsequent `torch` allocations reuse these cached buffers (which is fast), but if you're loading a new dataset with a different size, the cached buffers are useless and just waste memory. `torch.mps.empty_cache()` drains this free-list, making the memory available for new allocations.

### Monitoring memory

```python
torch.mps.current_allocated_memory() / 1e9   # GB currently in use
torch.mps.driver_allocated_memory() / 1e9    # GB allocated by Metal driver
```
