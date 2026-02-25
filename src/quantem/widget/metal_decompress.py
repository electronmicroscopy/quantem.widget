"""MPS-accelerated bitshuffle+LZ4 decompression for arina 4D-STEM data.

Uses Metal compute shaders on Apple Silicon with unified memory — GPU
decompresses directly into numpy-accessible memory, no copies needed.

Prefer the public API via ``IO.arina_file()``::

    from quantem.widget import IO
    data = IO.arina_file("master.h5", det_bin=2)

This module is the MPS backend; ``IO.arina_file()`` auto-detects the GPU.
(``IO.arina()`` still works as an alias.)
"""

from __future__ import annotations

import os
import time
import Metal
import h5py
import hdf5plugin  # noqa: F401 - registers bitshuffle filter
import numpy as np
from numba import njit, prange

__all__ = ["load_arina"]

# ---------------------------------------------------------------------------
# Metal Shading Language kernels
# ---------------------------------------------------------------------------
_METAL_SOURCE = r'''
#include <metal_stdlib>
using namespace metal;

// ---- LZ4 decompression constants ----
constant uint DECOMP_THREADS_PER_CHUNK = 32;
constant uint DECOMP_INPUT_BUFFER_SIZE = 256;  // 32 * sizeof(ulong)
constant uint DECOMP_BUFFER_PREFETCH_DIST = 128;

inline void syncCTA() {
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

inline ushort readWordDevice(const device uchar* address) {
    return ushort(address[0]) | (ushort(address[1]) << 8);
}
inline ushort readWordTG(threadgroup const uchar* address) {
    return ushort(address[0]) | (ushort(address[1]) << 8);
}

struct token_type { uint num_literals; uint num_matches; };

inline token_type decodePair(uchar num) {
    return token_type{uint((num & 0xf0) >> 4), uint(num & 0x0f)};
}

// ---- Cooperative copy functions ----
inline void coopCopyNoOverlap(
    device uchar* dest, const device uchar* source,
    uint length, uint tid, uint stride
) {
    for (uint i = tid; i < length; i += stride)
        dest[i] = source[i];
}
inline void coopCopyNoOverlapFromTG(
    device uchar* dest, threadgroup const uchar* source,
    uint length, uint tid, uint stride
) {
    for (uint i = tid; i < length; i += stride)
        dest[i] = source[i];
}
inline void coopCopyRepeat(
    device uchar* dest, const device uchar* source,
    uint dist, uint length, uint tid, uint stride
) {
    for (uint i = tid; i < length; i += stride)
        dest[i] = source[i % dist];
}
inline void coopCopyRepeatFromTG(
    device uchar* dest, threadgroup const uchar* source,
    uint dist, uint length, uint tid, uint stride
) {
    for (uint i = tid; i < length; i += stride)
        dest[i] = source[i % dist];
}
inline void coopCopyOverlap(
    device uchar* dest, const device uchar* source,
    uint dist, uint length, uint tid, uint stride
) {
    if (dist < length) coopCopyRepeat(dest, source, dist, length, tid, stride);
    else               coopCopyNoOverlap(dest, source, length, tid, stride);
}
inline void coopCopyOverlapFromTG(
    device uchar* dest, threadgroup const uchar* source,
    uint dist, uint length, uint tid, uint stride
) {
    if (dist < length) coopCopyRepeatFromTG(dest, source, dist, length, tid, stride);
    else               coopCopyNoOverlapFromTG(dest, source, length, tid, stride);
}

// ---- Buffer load: fill threadgroup cache from device memory ----
inline void bufferLoadAt(
    threadgroup uchar* buffer,
    const device uchar* compData,
    uint comp_length,
    thread long& buf_offset,
    uint tid
) {
    ulong addr = ulong(compData) + ulong(buf_offset);
    ulong aligned = (addr / 8) * 8;
    buf_offset = long(aligned - ulong(compData));
    if (uint(buf_offset) + DECOMP_INPUT_BUFFER_SIZE <= comp_length) {
        const device ulong* src = (const device ulong*)(compData + buf_offset);
        threadgroup ulong* dst = (threadgroup ulong*)buffer;
        dst[tid] = src[tid];
    } else {
        for (uint i = tid; i < DECOMP_INPUT_BUFFER_SIZE; i += DECOMP_THREADS_PER_CHUNK) {
            if (uint(buf_offset) + i < comp_length)
                buffer[i] = compData[buf_offset + i];
        }
    }
    syncCTA();
}

inline uint readLSIC(
    threadgroup const uchar* buffer,
    const device uchar* compData,
    long buf_offset, uint buf_end,
    thread uint& idx
) {
    uint num = 0;
    uchar next = 0xff;
    while (next == 0xff && idx < buf_end) {
        next = buffer[idx - uint(buf_offset)];
        idx++;
        num += next;
    }
    while (next == 0xff) {
        next = compData[idx];
        idx++;
        num += next;
    }
    return num;
}

inline void decompressStream(
    threadgroup uchar* buffer,
    device uchar* decompData,
    const device uchar* compData,
    uint comp_end,
    uint tid
) {
    long buf_offset = 0;
    bufferLoadAt(buffer, compData, comp_end, buf_offset, tid);
    uint decomp_idx = 0;
    uint comp_idx = 0;
    while (comp_idx < comp_end) {
        uint buf_end = uint(buf_offset) + DECOMP_INPUT_BUFFER_SIZE;
        if (comp_idx + DECOMP_BUFFER_PREFETCH_DIST > buf_end) {
            buf_offset = long(comp_idx);
            bufferLoadAt(buffer, compData, comp_end, buf_offset, tid);
        }
        buf_end = uint(buf_offset) + DECOMP_INPUT_BUFFER_SIZE;
        token_type tok = decodePair(buffer[comp_idx - uint(buf_offset)]);
        comp_idx++;
        uint num_literals = tok.num_literals;
        if (tok.num_literals == 15)
            num_literals += readLSIC(buffer, compData, buf_offset, buf_end, comp_idx);
        uint literalStart = comp_idx;
        if (num_literals + comp_idx > buf_end)
            coopCopyNoOverlap(decompData + decomp_idx, compData + comp_idx,
                              num_literals, tid, DECOMP_THREADS_PER_CHUNK);
        else
            coopCopyNoOverlapFromTG(decompData + decomp_idx,
                                    buffer + (comp_idx - uint(buf_offset)),
                                    num_literals, tid, DECOMP_THREADS_PER_CHUNK);
        comp_idx += num_literals;
        decomp_idx += num_literals;
        if (comp_idx < comp_end) {
            ushort match_offset;
            if (comp_idx + 2 > buf_end)
                match_offset = readWordDevice(compData + comp_idx);
            else
                match_offset = readWordTG(buffer + (comp_idx - uint(buf_offset)));
            comp_idx += 2;
            uint match_len = 4 + tok.num_matches;
            if (tok.num_matches == 15)
                match_len += readLSIC(buffer, compData, buf_offset, buf_end, comp_idx);
            if (match_offset <= num_literals
                && (long(literalStart) >= buf_offset
                    && literalStart + num_literals <= buf_end)) {
                coopCopyOverlapFromTG(
                    decompData + decomp_idx,
                    buffer + (literalStart + (num_literals - match_offset) - uint(buf_offset)),
                    match_offset, match_len, tid, DECOMP_THREADS_PER_CHUNK);
                syncCTA();
            } else {
                syncCTA();
                coopCopyOverlap(
                    decompData + decomp_idx,
                    decompData + decomp_idx - match_offset,
                    match_offset, match_len, tid, DECOMP_THREADS_PER_CHUNK);
            }
            decomp_idx += match_len;
        }
    }
}

inline uint read32be(const device uchar* p) {
    return (uint(p[0]) << 24) | (uint(p[1]) << 16) | (uint(p[2]) << 8) | uint(p[3]);
}

// =====================================================================
// Kernel 1: Batched LZ4 decompression
// Grid:  ((max_blocks+1)/2, 1, n_frames)
// Block: (32, 2, 1)
// =====================================================================
kernel void h5lz4dc_batched(
    const device uchar* compressed          [[buffer(0)]],
    const device uint*  chunk_offsets        [[buffer(1)]],
    const device uint*  block_starts         [[buffer(2)]],
    const device uint*  block_counts         [[buffer(3)]],
    const device uint*  block_offsets        [[buffer(4)]],
    constant uint&      blocksize            [[buffer(5)]],
    constant uint&      frame_bytes          [[buffer(6)]],
    device uchar*       decompressed         [[buffer(7)]],
    uint3 tpig  [[threadgroup_position_in_grid]],
    uint3 tpit  [[thread_position_in_threadgroup]],
    uint3 tptg  [[threads_per_threadgroup]]
) {
    uint frame_id       = tpig.z;
    uint block_in_frame = tpig.x * tptg.y + tpit.y;
    uint chunk_offset   = chunk_offsets[frame_id];
    uint block_offset   = block_offsets[frame_id];
    uint num_blocks     = block_counts[frame_id];
    threadgroup uchar tg_buffer[256 * 2];
    if (block_in_frame < num_blocks) {
        uint blk_start = block_starts[block_offset + block_in_frame];
        const device uchar* input = compressed + chunk_offset + blk_start + 4;
        uint comp_size = read32be(compressed + chunk_offset + blk_start);
        device uchar* output = decompressed + frame_id * frame_bytes
                               + block_in_frame * blocksize;
        decompressStream(tg_buffer + tpit.y * 256, output, input, comp_size, tpit.x);
    }
}

// =====================================================================
// Kernel 2a: Batched bitshuffle unshuffle (8192-byte blocks, 32-bit elems)
//
// Optimized: 32 SIMD groups per threadgroup (1024 threads), each group
// unshuffles 32 elements using shared memory to cache bit planes.
//
// Grid:  (ceil(total_groups / 32), 1, n_frames)
// Block: (32, 32, 1)  →  32 SIMD groups of 32 threads
// =====================================================================
kernel void shuf_8192_32_batched(
    const device uint*  in   [[buffer(0)]],
    device uint*        out  [[buffer(1)]],
    constant uint&      frame_u32s        [[buffer(2)]],
    constant uint&      groups_per_block  [[buffer(3)]],
    constant uint&      total_groups      [[buffer(4)]],
    uint3 tpig       [[threadgroup_position_in_grid]],
    uint  lane       [[thread_index_in_simdgroup]],
    uint  sg_id      [[simdgroup_index_in_threadgroup]]
) {
    uint frame_id  = tpig.z;
    uint group_id  = tpig.x * 32 + sg_id;
    if (group_id >= total_groups) return;
    uint block_id       = group_id / groups_per_block;
    uint group_in_block = group_id % groups_per_block;
    const device uint* block_in = in + frame_id * frame_u32s
                                     + block_id * 2048;
    // Cache 32 bit planes in shared memory (one set per SIMD group)
    threadgroup uint planes[32][32];
    planes[sg_id][lane] = block_in[lane * groups_per_block + group_in_block];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    uint result = 0;
    for (int j = 0; j < 32; j++)
        if (planes[sg_id][j] & (1U << lane))
            result |= (1U << j);
    out[frame_id * frame_u32s + group_id * 32 + lane] = result;
}

// =====================================================================
// Kernel 2b: Batched bitshuffle unshuffle (8192-byte blocks, 16-bit elems)
//
// Optimized: 32 SIMD groups per threadgroup (1024 threads), each group
// unshuffles 32 elements using shared memory to cache bit planes.
// 32x fewer threadgroup launches vs one-group-per-threadgroup version.
//
// Grid:  (ceil(groups_per_frame / 32), 1, n_frames)
// Block: (32, 32, 1)  →  32 SIMD groups of 32 threads
// =====================================================================
kernel void shuf_8192_16_batched(
    const device uint*  in   [[buffer(0)]],
    device ushort*      out  [[buffer(1)]],
    constant uint&      frame_u16s        [[buffer(2)]],
    constant uint&      groups_per_block  [[buffer(3)]],
    constant uint&      total_groups      [[buffer(4)]],
    uint3 tpig       [[threadgroup_position_in_grid]],
    uint  lane       [[thread_index_in_simdgroup]],
    uint  sg_id      [[simdgroup_index_in_threadgroup]]
) {
    uint frame_id  = tpig.z;
    uint group_id  = tpig.x * 32 + sg_id;

    if (group_id >= total_groups) return;

    uint block_id       = group_id / groups_per_block;
    uint group_in_block = group_id % groups_per_block;

    const device uint* block_in = in + frame_id * (frame_u16s / 2)
                                     + block_id * 2048;

    // Cache 16 bit planes in shared memory (one set per SIMD group)
    threadgroup uint planes[32][16];
    if (lane < 16)
        planes[sg_id][lane] = block_in[lane * groups_per_block + group_in_block];
    simdgroup_barrier(mem_flags::mem_threadgroup);

    ushort result = 0;
    for (int j = 0; j < 16; j++)
        if (planes[sg_id][j] & (1U << lane))
            result |= ushort(1U << j);

    out[frame_id * frame_u16s + group_id * 32 + lane] = result;
}

// =====================================================================
// Kernel 3: Detector binning (mean, any bin factor)
// Reads from bitshuffle output, writes float32 binned result.
// Grid:  (ceil(out_cols/16), ceil(out_rows/16), n_frames)
// Block: (16, 16, 1)
// =====================================================================
kernel void bin_mean_u16(
    const device ushort* in  [[buffer(0)]],
    device float*        out [[buffer(1)]],
    constant uint& in_row_stride  [[buffer(2)]],
    constant uint& in_frame_elems [[buffer(3)]],
    constant uint& out_cols       [[buffer(4)]],
    constant uint& out_frame_elems [[buffer(5)]],
    constant uint& bin_factor     [[buffer(6)]],
    uint3 pos [[thread_position_in_grid]]
) {
    uint out_c = pos.x;
    uint out_r = pos.y;
    if (out_c >= out_cols || out_r >= out_cols) return;
    uint frame = pos.z;
    uint in_r = out_r * bin_factor;
    uint in_c = out_c * bin_factor;
    uint in_base = frame * in_frame_elems;
    float sum = 0.0f;
    for (uint br = 0; br < bin_factor; br++)
        for (uint bc = 0; bc < bin_factor; bc++)
            sum += float(in[in_base + (in_r + br) * in_row_stride + in_c + bc]);
    out[frame * out_frame_elems + out_r * out_cols + out_c] =
        sum / float(bin_factor * bin_factor);
}

kernel void bin_mean_u32(
    const device uint* in    [[buffer(0)]],
    device float*      out   [[buffer(1)]],
    constant uint& in_row_stride  [[buffer(2)]],
    constant uint& in_frame_elems [[buffer(3)]],
    constant uint& out_cols       [[buffer(4)]],
    constant uint& out_frame_elems [[buffer(5)]],
    constant uint& bin_factor     [[buffer(6)]],
    uint3 pos [[thread_position_in_grid]]
) {
    uint out_c = pos.x;
    uint out_r = pos.y;
    if (out_c >= out_cols || out_r >= out_cols) return;
    uint frame = pos.z;
    uint in_r = out_r * bin_factor;
    uint in_c = out_c * bin_factor;
    uint in_base = frame * in_frame_elems;
    float sum = 0.0f;
    for (uint br = 0; br < bin_factor; br++)
        for (uint bc = 0; bc < bin_factor; bc++)
            sum += float(in[in_base + (in_r + br) * in_row_stride + in_c + bc]);
    out[frame * out_frame_elems + out_r * out_cols + out_c] =
        sum / float(bin_factor * bin_factor);
}
'''

# ---------------------------------------------------------------------------
# Compile Metal kernels at import time
# ---------------------------------------------------------------------------
_device = Metal.MTLCreateSystemDefaultDevice()
_options = Metal.MTLCompileOptions.alloc().init()
_library, _compile_error = _device.newLibraryWithSource_options_error_(
    _METAL_SOURCE, _options, None
)
if _compile_error:
    raise RuntimeError(f"Metal shader compile error: {_compile_error}")
_h5lz4dc_fn = _library.newFunctionWithName_("h5lz4dc_batched")
_shuf32_fn = _library.newFunctionWithName_("shuf_8192_32_batched")
_shuf16_fn = _library.newFunctionWithName_("shuf_8192_16_batched")
_bin_u16_fn = _library.newFunctionWithName_("bin_mean_u16")
_bin_u32_fn = _library.newFunctionWithName_("bin_mean_u32")
_h5lz4dc_pipeline, _ = _device.newComputePipelineStateWithFunction_error_(_h5lz4dc_fn, None)
_shuf32_pipeline, _ = _device.newComputePipelineStateWithFunction_error_(_shuf32_fn, None)
_shuf16_pipeline, _ = _device.newComputePipelineStateWithFunction_error_(_shuf16_fn, None)
_bin_u16_pipeline, _ = _device.newComputePipelineStateWithFunction_error_(_bin_u16_fn, None)
_bin_u32_pipeline, _ = _device.newComputePipelineStateWithFunction_error_(_bin_u32_fn, None)
_queue = _device.newCommandQueue()


# ---------------------------------------------------------------------------
# Header parser (numba, runs on CPU in parallel)
# ---------------------------------------------------------------------------
@njit(parallel=True)
def _bin_mean(src, dst, brow, bcol):
    """Bin a 3D array (n, rows, cols) by averaging brow×bcol blocks."""
    nf, dr, dc = dst.shape
    scale = np.float32(1.0 / (brow * bcol))
    for i in prange(nf):
        for r in range(dr):
            rb = r * brow
            for c in range(dc):
                cb = c * bcol
                s = np.float32(0.0)
                for br in range(brow):
                    for bc in range(bcol):
                        s += np.float32(src[i, rb + br, cb + bc])
                dst[i, r, c] = s * scale


@njit(cache=True, parallel=True)
def _parse_headers(
    buffer, chunk_sizes, chunk_offsets,
    block_starts_out, block_counts_out,
    n_frames, n_blocks_per_frame,
):
    """Parse bitshuffle+LZ4 chunk headers in parallel."""
    for i in prange(n_frames):
        offset = chunk_offsets[i]
        chunk = buffer[offset : offset + chunk_sizes[i]]
        uncomp_size = (
            int(chunk[0]) << 56 | int(chunk[1]) << 48
            | int(chunk[2]) << 40 | int(chunk[3]) << 32
            | int(chunk[4]) << 24 | int(chunk[5]) << 16
            | int(chunk[6]) << 8  | int(chunk[7])
        )
        block_size = (
            int(chunk[8]) << 24 | int(chunk[9]) << 16
            | int(chunk[10]) << 8 | int(chunk[11])
        )
        n_blocks = (uncomp_size + block_size - 1) // block_size
        block_counts_out[i] = n_blocks
        pos = 12
        base_idx = i * n_blocks_per_frame
        for b in range(n_blocks):
            block_starts_out[base_idx + b] = pos
            comp_size = (
                int(chunk[pos]) << 24 | int(chunk[pos + 1]) << 16
                | int(chunk[pos + 2]) << 8 | int(chunk[pos + 3])
            )
            pos += 4 + comp_size


def _metal_buffer_alloc(nbytes):
    """Allocate an MTLBuffer of given size (shared memory)."""
    return _device.newBufferWithLength_options_(
        nbytes, Metal.MTLResourceStorageModeShared
    )


def _numpy_view(mtl_buf, dtype, count):
    """Get a writable numpy view of a Metal buffer (zero-copy, unified memory)."""
    mv = mtl_buf.contents().as_buffer(mtl_buf.length())
    return np.frombuffer(mv, dtype=dtype, count=count)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class MPSDecompressor:
    """MPS-accelerated decompressor for bitshuffle+LZ4 HDF5 datasets.

    Uses Metal compute shaders on Apple Silicon. All buffers are pre-allocated
    in unified memory and reused across calls — no per-call allocation or
    CPU-GPU transfers.

    Parameters
    ----------
    max_compressed_bytes : int, optional
        Maximum total compressed data per load call, by default 150 MB.
    max_frames : int, optional
        Maximum number of frames per load call, by default 11000.
    frame_bytes : int, optional
        Decompressed bytes per frame, by default 192*192*2 (uint16).
    n_blocks_per_frame : int, optional
        LZ4 blocks per frame, by default 9 for 192x192 uint16.
    """

    def __init__(
        self,
        max_compressed_bytes: int = 150 * 1024 * 1024,
        max_frames: int = 11_000,
        frame_bytes: int = 192 * 192 * 2,
        n_blocks_per_frame: int = 9,
    ):
        self.max_frames = max_frames
        self.frame_bytes = frame_bytes
        self.n_blocks_per_frame = n_blocks_per_frame
        # Pre-allocate Metal buffers with numpy views (unified memory)
        self._comp_mtl = _metal_buffer_alloc(max_compressed_bytes)
        self._comp_np = _numpy_view(self._comp_mtl, np.uint8, max_compressed_bytes)
        total_output = max_frames * frame_bytes
        self._lz4_mtl = _metal_buffer_alloc(total_output)
        self._shuf_mtl = _metal_buffer_alloc(total_output)
        self._result_np = _numpy_view(self._shuf_mtl, np.uint8, total_output)
        # Pre-allocate metadata Metal buffers with numpy views
        self._co_mtl = _metal_buffer_alloc(max_frames * 4)
        self._co_np = _numpy_view(self._co_mtl, np.uint32, max_frames)
        max_blocks = max_frames * n_blocks_per_frame
        self._bs_mtl = _metal_buffer_alloc(max_blocks * 4)
        self._bs_np = _numpy_view(self._bs_mtl, np.uint32, max_blocks)
        self._bc_mtl = _metal_buffer_alloc(max_frames * 4)
        self._bc_np = _numpy_view(self._bc_mtl, np.uint32, max_frames)
        self._bo_mtl = _metal_buffer_alloc((max_frames + 1) * 4)
        self._bo_np = _numpy_view(self._bo_mtl, np.uint32, max_frames + 1)
        # CPU-side arrays for chunk reading
        self._chunk_sizes = np.zeros(max_frames, dtype=np.uint32)
        # Buffer B (for double-buffering in load_master)
        self._comp_mtl_b = _metal_buffer_alloc(max_compressed_bytes)
        self._comp_np_b = _numpy_view(self._comp_mtl_b, np.uint8, max_compressed_bytes)
        self._co_mtl_b = _metal_buffer_alloc(max_frames * 4)
        self._co_np_b = _numpy_view(self._co_mtl_b, np.uint32, max_frames)
        self._bs_mtl_b = _metal_buffer_alloc(max_blocks * 4)
        self._bs_np_b = _numpy_view(self._bs_mtl_b, np.uint32, max_blocks)
        self._bc_mtl_b = _metal_buffer_alloc(max_frames * 4)
        self._bc_np_b = _numpy_view(self._bc_mtl_b, np.uint32, max_frames)
        self._bo_mtl_b = _metal_buffer_alloc((max_frames + 1) * 4)
        self._bo_np_b = _numpy_view(self._bo_mtl_b, np.uint32, max_frames + 1)
        self._chunk_sizes_b = np.zeros(max_frames, dtype=np.uint32)
        # Large output buffer for load_master() — allocated on first use
        self._out_mtl = None
        self._out_np = None
        self._out_nbytes = 0

    def _ensure_output_buffer(self, nbytes):
        """Allocate (or reuse) a large Metal output buffer for all frames."""
        if nbytes <= self._out_nbytes:
            return
        self._out_mtl = _metal_buffer_alloc(nbytes)
        self._out_np = _numpy_view(self._out_mtl, np.uint8, nbytes)
        self._out_nbytes = nbytes

    def _read_chunk(self, filepath, comp_np, co_np, chunk_sizes):
        """Read raw compressed HDF5 chunks into pre-allocated buffers."""
        with h5py.File(filepath, "r") as f:
            ds = f["entry/data/data"]
            n_frames = ds.shape[0]
            frame_shape = ds.shape[1:]
            dtype = ds.dtype
            offset = 0
            for i in range(n_frames):
                _, raw = ds.id.read_direct_chunk((i, 0, 0))
                chunk_len = len(raw)
                co_np[i] = offset
                chunk_sizes[i] = chunk_len
                comp_np[offset : offset + chunk_len] = np.frombuffer(
                    raw, dtype=np.uint8
                )
                offset += chunk_len
        return n_frames, frame_shape, dtype

    def _submit_gpu(self, n_frames, frame_bytes, elem_size, out_byte_offset,
                    comp_mtl, co_mtl, bs_mtl, bc_mtl, bo_mtl, max_blocks):
        """Submit LZ4 + bitshuffle GPU work, return uncommitted command buffer."""
        cmd = _queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        # LZ4
        enc.setComputePipelineState_(_h5lz4dc_pipeline)
        enc.setBuffer_offset_atIndex_(comp_mtl, 0, 0)
        enc.setBuffer_offset_atIndex_(co_mtl, 0, 1)
        enc.setBuffer_offset_atIndex_(bs_mtl, 0, 2)
        enc.setBuffer_offset_atIndex_(bc_mtl, 0, 3)
        enc.setBuffer_offset_atIndex_(bo_mtl, 0, 4)
        enc.setBytes_length_atIndex_(
            np.array([8192], dtype=np.uint32).tobytes(), 4, 5
        )
        enc.setBytes_length_atIndex_(
            np.array([frame_bytes], dtype=np.uint32).tobytes(), 4, 6
        )
        enc.setBuffer_offset_atIndex_(self._lz4_mtl, 0, 7)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake((max_blocks + 1) // 2, 1, n_frames),
            Metal.MTLSizeMake(32, 2, 1),
        )
        enc.memoryBarrierWithScope_(Metal.MTLBarrierScopeBuffers)
        # Bitshuffle
        n_8kb = frame_bytes // 8192
        if elem_size == 2:
            groups_per_block = 8192 // (elem_size * 32)
            groups_per_frame = n_8kb * groups_per_block
            frame_elems = frame_bytes // 2
            enc.setComputePipelineState_(_shuf16_pipeline)
            enc.setBuffer_offset_atIndex_(self._lz4_mtl, 0, 0)
            enc.setBuffer_offset_atIndex_(self._out_mtl, out_byte_offset, 1)
            enc.setBytes_length_atIndex_(
                np.array([frame_elems], dtype=np.uint32).tobytes(), 4, 2
            )
            enc.setBytes_length_atIndex_(
                np.array([groups_per_block], dtype=np.uint32).tobytes(), 4, 3
            )
            enc.setBytes_length_atIndex_(
                np.array([groups_per_frame], dtype=np.uint32).tobytes(), 4, 4
            )
            tg_count = (groups_per_frame + 31) // 32
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSizeMake(tg_count, 1, n_frames),
                Metal.MTLSizeMake(32, 32, 1),
            )
        else:
            groups_per_block = 2048 // 32
            groups_per_frame = n_8kb * groups_per_block
            frame_elems = frame_bytes // 4
            enc.setComputePipelineState_(_shuf32_pipeline)
            enc.setBuffer_offset_atIndex_(self._lz4_mtl, 0, 0)
            enc.setBuffer_offset_atIndex_(self._out_mtl, out_byte_offset, 1)
            enc.setBytes_length_atIndex_(
                np.array([frame_elems], dtype=np.uint32).tobytes(), 4, 2
            )
            enc.setBytes_length_atIndex_(
                np.array([groups_per_block], dtype=np.uint32).tobytes(), 4, 3
            )
            enc.setBytes_length_atIndex_(
                np.array([groups_per_frame], dtype=np.uint32).tobytes(), 4, 4
            )
            tg_count = (groups_per_frame + 31) // 32
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSizeMake(tg_count, 1, n_frames),
                Metal.MTLSizeMake(32, 32, 1),
            )
        enc.endEncoding()
        cmd.commit()
        return cmd

    def _submit_gpu_binned(self, n_frames, frame_bytes, elem_size,
                           out_byte_offset, det_row, det_col, bin_factor,
                           comp_mtl, co_mtl, bs_mtl, bc_mtl, bo_mtl,
                           max_blocks):
        """Submit LZ4 + bitshuffle + bin GPU work. Returns command buffer."""
        cmd = _queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        # LZ4
        enc.setComputePipelineState_(_h5lz4dc_pipeline)
        enc.setBuffer_offset_atIndex_(comp_mtl, 0, 0)
        enc.setBuffer_offset_atIndex_(co_mtl, 0, 1)
        enc.setBuffer_offset_atIndex_(bs_mtl, 0, 2)
        enc.setBuffer_offset_atIndex_(bc_mtl, 0, 3)
        enc.setBuffer_offset_atIndex_(bo_mtl, 0, 4)
        enc.setBytes_length_atIndex_(
            np.array([8192], dtype=np.uint32).tobytes(), 4, 5
        )
        enc.setBytes_length_atIndex_(
            np.array([frame_bytes], dtype=np.uint32).tobytes(), 4, 6
        )
        enc.setBuffer_offset_atIndex_(self._lz4_mtl, 0, 7)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake((max_blocks + 1) // 2, 1, n_frames),
            Metal.MTLSizeMake(32, 2, 1),
        )
        enc.memoryBarrierWithScope_(Metal.MTLBarrierScopeBuffers)
        # Bitshuffle → _shuf_mtl (temporary)
        n_8kb = frame_bytes // 8192
        if elem_size == 2:
            groups_per_block = 8192 // (elem_size * 32)
            groups_per_frame = n_8kb * groups_per_block
            frame_elems = frame_bytes // 2
            enc.setComputePipelineState_(_shuf16_pipeline)
            enc.setBuffer_offset_atIndex_(self._lz4_mtl, 0, 0)
            enc.setBuffer_offset_atIndex_(self._shuf_mtl, 0, 1)
            enc.setBytes_length_atIndex_(
                np.array([frame_elems], dtype=np.uint32).tobytes(), 4, 2
            )
            enc.setBytes_length_atIndex_(
                np.array([groups_per_block], dtype=np.uint32).tobytes(), 4, 3
            )
            enc.setBytes_length_atIndex_(
                np.array([groups_per_frame], dtype=np.uint32).tobytes(), 4, 4
            )
            tg_count = (groups_per_frame + 31) // 32
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSizeMake(tg_count, 1, n_frames),
                Metal.MTLSizeMake(32, 32, 1),
            )
        else:
            groups_per_block = 2048 // 32
            groups_per_frame = n_8kb * groups_per_block
            frame_elems = frame_bytes // 4
            enc.setComputePipelineState_(_shuf32_pipeline)
            enc.setBuffer_offset_atIndex_(self._lz4_mtl, 0, 0)
            enc.setBuffer_offset_atIndex_(self._shuf_mtl, 0, 1)
            enc.setBytes_length_atIndex_(
                np.array([frame_elems], dtype=np.uint32).tobytes(), 4, 2
            )
            enc.setBytes_length_atIndex_(
                np.array([groups_per_block], dtype=np.uint32).tobytes(), 4, 3
            )
            enc.setBytes_length_atIndex_(
                np.array([groups_per_frame], dtype=np.uint32).tobytes(), 4, 4
            )
            tg_count = (groups_per_frame + 31) // 32
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSizeMake(tg_count, 1, n_frames),
                Metal.MTLSizeMake(32, 32, 1),
            )
        enc.memoryBarrierWithScope_(Metal.MTLBarrierScopeBuffers)
        # Bin: _shuf_mtl → _out_mtl at offset
        out_det_row = det_row // bin_factor
        out_det_col = det_col // bin_factor
        out_frame_elems = out_det_row * out_det_col
        in_frame_elems = det_row * det_col
        bin_pipeline = _bin_u16_pipeline if elem_size == 2 else _bin_u32_pipeline
        enc.setComputePipelineState_(bin_pipeline)
        enc.setBuffer_offset_atIndex_(self._shuf_mtl, 0, 0)
        enc.setBuffer_offset_atIndex_(self._out_mtl, out_byte_offset, 1)
        enc.setBytes_length_atIndex_(
            np.array([det_col], dtype=np.uint32).tobytes(), 4, 2
        )
        enc.setBytes_length_atIndex_(
            np.array([in_frame_elems], dtype=np.uint32).tobytes(), 4, 3
        )
        enc.setBytes_length_atIndex_(
            np.array([out_det_col], dtype=np.uint32).tobytes(), 4, 4
        )
        enc.setBytes_length_atIndex_(
            np.array([out_frame_elems], dtype=np.uint32).tobytes(), 4, 5
        )
        enc.setBytes_length_atIndex_(
            np.array([bin_factor], dtype=np.uint32).tobytes(), 4, 6
        )
        grid_x = (out_det_col + 15) // 16
        grid_y = (out_det_row + 15) // 16
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake(grid_x, grid_y, n_frames),
            Metal.MTLSizeMake(16, 16, 1),
        )
        enc.endEncoding()
        cmd.commit()
        return cmd

    def load_master(self, master_path: str) -> np.ndarray:
        """Load all chunks from an arina master file via MPS.

        Uses double-buffering: reads chunk N+1 while GPU processes chunk N.
        Writes each chunk's result directly at the correct offset in a single
        output buffer — no intermediate copies.

        Parameters
        ----------
        master_path : str
            Path to the arina master HDF5 file.

        Returns
        -------
        np.ndarray
            Numpy array with shape (total_frames, det_rows, det_cols).
        """
        t0 = time.perf_counter()
        master_dir = os.path.dirname(master_path)
        # Discover chunks and metadata from master
        with h5py.File(master_path, "r") as f:
            chunk_keys = sorted(f["entry/data"].keys())
            ds0 = f[f"entry/data/{chunk_keys[0]}"]
            frame_shape = ds0.shape[1:]
            dtype = ds0.dtype
            frame_bytes = int(np.prod(frame_shape) * np.dtype(dtype).itemsize)
            elem_size = np.dtype(dtype).itemsize
        # Resolve chunk file paths
        prefix = os.path.basename(master_path).replace("_master.h5", "")
        chunk_files = []
        for k in chunk_keys:
            suffix = k.split("_")[-1]
            chunk_files.append(
                os.path.join(master_dir, f"{prefix}_data_{suffix}.h5")
            )
        # Count total frames
        chunk_n_frames = []
        for cf in chunk_files:
            with h5py.File(cf, "r") as f:
                chunk_n_frames.append(f["entry/data/data"].shape[0])
        total_frames = sum(chunk_n_frames)
        total_bytes = total_frames * frame_bytes
        self._ensure_output_buffer(total_bytes)
        n_blocks_per_frame = (frame_bytes + 8191) // 8192
        # Double-buffer sets: A (primary) and B
        bufs = [
            (self._comp_np, self._co_np, self._bs_np, self._bc_np,
             self._bo_np, self._chunk_sizes,
             self._comp_mtl, self._co_mtl, self._bs_mtl, self._bc_mtl,
             self._bo_mtl),
            (self._comp_np_b, self._co_np_b, self._bs_np_b, self._bc_np_b,
             self._bo_np_b, self._chunk_sizes_b,
             self._comp_mtl_b, self._co_mtl_b, self._bs_mtl_b, self._bc_mtl_b,
             self._bo_mtl_b),
        ]
        t_setup = time.perf_counter()
        # Read first chunk into buffer A
        comp_np, co_np, bs_np, bc_np, bo_np, csizes, \
            comp_mtl, co_mtl, bs_mtl, bc_mtl, bo_mtl = bufs[0]
        self._read_chunk(chunk_files[0], comp_np, co_np, csizes)
        _parse_headers(comp_np, csizes, co_np, bs_np, bc_np,
                       chunk_n_frames[0], n_blocks_per_frame)
        bo_np[0] = 0
        bo_np[1 : chunk_n_frames[0] + 1] = np.cumsum(bc_np[:chunk_n_frames[0]])
        max_blk = int(bc_np[:chunk_n_frames[0]].max())
        frame_offset = 0
        n_chunks = len(chunk_files)
        for ci in range(n_chunks):
            cur = bufs[ci % 2]
            comp_np, co_np, bs_np, bc_np, bo_np, csizes, \
                comp_mtl, co_mtl, bs_mtl, bc_mtl, bo_mtl = cur
            n_frames = chunk_n_frames[ci]
            out_byte_offset = frame_offset * frame_bytes
            # Submit GPU (async — returns immediately)
            cmd = self._submit_gpu(
                n_frames, frame_bytes, elem_size, out_byte_offset,
                comp_mtl, co_mtl, bs_mtl, bc_mtl, bo_mtl, max_blk,
            )
            # While GPU runs, read + parse next chunk into the other buffer
            if ci + 1 < n_chunks:
                nxt = bufs[(ci + 1) % 2]
                comp_np_n, co_np_n, bs_np_n, bc_np_n, bo_np_n, csizes_n, \
                    *_ = nxt
                self._read_chunk(chunk_files[ci + 1], comp_np_n, co_np_n,
                                 csizes_n)
                nf_next = chunk_n_frames[ci + 1]
                _parse_headers(comp_np_n, csizes_n, co_np_n, bs_np_n,
                               bc_np_n, nf_next, n_blocks_per_frame)
                bo_np_n[0] = 0
                bo_np_n[1 : nf_next + 1] = np.cumsum(bc_np_n[:nf_next])
                max_blk = int(bc_np_n[:nf_next].max())
            # Wait for current GPU to finish
            cmd.waitUntilCompleted()
            frame_offset += n_frames
        t_total = time.perf_counter()
        result = self._out_np[:total_bytes].view(dtype).reshape(
            (total_frames,) + frame_shape
        )
        print(
            f"MPSDecompressor.load_master: {total_frames} frames, "
            f"{total_bytes / 1e9:.2f} GB, "
            f"{t_total - t0:.3f}s"
        )
        return result

    def load(
        self,
        filepath: str,
        dataset_path: str = "entry/data/data",
        n_frames: int | None = None,
        verbose: bool = True,
    ) -> np.ndarray:
        """Load and decompress a bitshuffle+LZ4 HDF5 dataset via MPS.

        Returns a zero-copy view into the pre-allocated unified memory buffer.
        The view is overwritten on the next load() call.

        Parameters
        ----------
        filepath : str
            Path to the HDF5 file.
        dataset_path : str, optional
            Path to the dataset within the HDF5 file.
        n_frames : int, optional
            Number of frames to load. If None, loads all frames.

        Returns
        -------
        np.ndarray
            Numpy array with shape (n_frames, height, width).
        """
        t0 = time.perf_counter()

        # ---- Read raw chunks directly into pre-allocated Metal buffer ----
        with h5py.File(filepath, "r") as f:
            ds = f[dataset_path]
            total_in_file = ds.shape[0]
            n_frames = min(n_frames, total_in_file) if n_frames else total_in_file
            frame_shape = ds.shape[1:]
            dtype = ds.dtype
            frame_bytes = int(np.prod(frame_shape) * np.dtype(dtype).itemsize)
            offset = 0
            for i in range(n_frames):
                _, raw = ds.id.read_direct_chunk((i, 0, 0))
                chunk_len = len(raw)
                self._co_np[i] = offset
                self._chunk_sizes[i] = chunk_len
                self._comp_np[offset : offset + chunk_len] = np.frombuffer(
                    raw, dtype=np.uint8
                )
                offset += chunk_len
            total_compressed = offset
        t_read = time.perf_counter()

        # ---- Parse headers directly into pre-allocated Metal buffers ----
        # n_blocks_per_frame matches actual block count, so block_starts
        # layout matches block_offsets indexing — no repack needed
        n_blocks_per_frame = (frame_bytes + 8191) // 8192
        _parse_headers(
            self._comp_np, self._chunk_sizes, self._co_np,
            self._bs_np, self._bc_np, n_frames, n_blocks_per_frame,
        )
        self._bo_np[0] = 0
        self._bo_np[1 : n_frames + 1] = np.cumsum(self._bc_np[:n_frames])
        max_blocks_per_frame = int(self._bc_np[:n_frames].max())
        t_parse = time.perf_counter()

        # ---- Single command buffer: LZ4 + barrier + bitshuffle ----
        cmd = _queue.commandBuffer()
        enc = cmd.computeCommandEncoder()

        # LZ4 decompression
        enc.setComputePipelineState_(_h5lz4dc_pipeline)
        enc.setBuffer_offset_atIndex_(self._comp_mtl, 0, 0)
        enc.setBuffer_offset_atIndex_(self._co_mtl, 0, 1)
        enc.setBuffer_offset_atIndex_(self._bs_mtl, 0, 2)
        enc.setBuffer_offset_atIndex_(self._bc_mtl, 0, 3)
        enc.setBuffer_offset_atIndex_(self._bo_mtl, 0, 4)
        enc.setBytes_length_atIndex_(
            np.array([8192], dtype=np.uint32).tobytes(), 4, 5
        )
        enc.setBytes_length_atIndex_(
            np.array([frame_bytes], dtype=np.uint32).tobytes(), 4, 6
        )
        enc.setBuffer_offset_atIndex_(self._lz4_mtl, 0, 7)
        enc.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSizeMake((max_blocks_per_frame + 1) // 2, 1, n_frames),
            Metal.MTLSizeMake(32, 2, 1),
        )

        # Memory barrier between LZ4 output and bitshuffle input
        enc.memoryBarrierWithScope_(Metal.MTLBarrierScopeBuffers)

        # Bitshuffle unshuffle
        elem_size = np.dtype(dtype).itemsize
        n_8kb = frame_bytes // 8192
        if elem_size == 2:
            groups_per_block = 8192 // (elem_size * 32)  # 128
            groups_per_frame = n_8kb * groups_per_block   # 1152
            frame_u16s = frame_bytes // 2
            enc.setComputePipelineState_(_shuf16_pipeline)
            enc.setBuffer_offset_atIndex_(self._lz4_mtl, 0, 0)
            enc.setBuffer_offset_atIndex_(self._shuf_mtl, 0, 1)
            enc.setBytes_length_atIndex_(
                np.array([frame_u16s], dtype=np.uint32).tobytes(), 4, 2
            )
            enc.setBytes_length_atIndex_(
                np.array([groups_per_block], dtype=np.uint32).tobytes(), 4, 3
            )
            enc.setBytes_length_atIndex_(
                np.array([groups_per_frame], dtype=np.uint32).tobytes(), 4, 4
            )
            # 32 SIMD groups per threadgroup → 32x fewer launches
            tg_count = (groups_per_frame + 31) // 32
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSizeMake(tg_count, 1, n_frames),
                Metal.MTLSizeMake(32, 32, 1),
            )
        else:
            groups_per_block = 2048 // 32  # 64
            groups_per_frame = n_8kb * groups_per_block
            frame_u32s = frame_bytes // 4
            enc.setComputePipelineState_(_shuf32_pipeline)
            enc.setBuffer_offset_atIndex_(self._lz4_mtl, 0, 0)
            enc.setBuffer_offset_atIndex_(self._shuf_mtl, 0, 1)
            enc.setBytes_length_atIndex_(
                np.array([frame_u32s], dtype=np.uint32).tobytes(), 4, 2
            )
            enc.setBytes_length_atIndex_(
                np.array([groups_per_block], dtype=np.uint32).tobytes(), 4, 3
            )
            enc.setBytes_length_atIndex_(
                np.array([groups_per_frame], dtype=np.uint32).tobytes(), 4, 4
            )
            tg_count = (groups_per_frame + 31) // 32
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSizeMake(tg_count, 1, n_frames),
                Metal.MTLSizeMake(32, 32, 1),
            )

        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        t_gpu = time.perf_counter()

        # ---- Zero-copy result via unified memory ----
        total_bytes = n_frames * frame_bytes
        result = self._result_np[:total_bytes].view(dtype).reshape(
            (n_frames,) + frame_shape
        )
        t_total = time.perf_counter()

        if verbose:
            print(
                f"MPSDecompressor.load: {n_frames} frames, "
                f"{total_compressed / 1e6:.0f} MB compressed → "
                f"{total_bytes / 1e6:.0f} MB decompressed"
            )
            print(
                f"  read: {t_read - t0:.3f}s | "
                f"parse: {t_parse - t_read:.3f}s | "
                f"gpu: {t_gpu - t_parse:.3f}s | "
                f"total: {t_total - t0:.3f}s"
            )
        return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
_decompressor_cache: dict[int, MPSDecompressor] = {}


def _get_decompressor(frame_bytes, max_frames=11_000):
    """Get or create a decompressor sized for the given frame byte size."""
    cache_key = (frame_bytes, max_frames)
    if cache_key not in _decompressor_cache:
        n_blocks = (frame_bytes + 8191) // 8192
        # Scale compressed buffer: worst observed bitshuffle+LZ4 ratio ~7:1,
        # use //4 for headroom (386 MB for uint32, 256 MB for uint16)
        max_comp = max(256 * 1024 * 1024, max_frames * frame_bytes // 4)
        _decompressor_cache[cache_key] = MPSDecompressor(
            max_compressed_bytes=max_comp,
            max_frames=max_frames,
            frame_bytes=frame_bytes,
            n_blocks_per_frame=n_blocks,
        )
    return _decompressor_cache[cache_key]


def _parse_master(master_path):
    """Read metadata and chunk file list from an arina master file."""
    master_dir = os.path.dirname(master_path)
    prefix = os.path.basename(master_path).replace("_master.h5", "")
    with h5py.File(master_path, "r") as f:
        chunk_keys = sorted(f["entry/data"].keys())
        ds0 = f[f"entry/data/{chunk_keys[0]}"]
        det_shape = ds0.shape[1:]  # (192, 192)
        dtype = ds0.dtype
        spec = f["entry/instrument/detector/detectorSpecific"]
        ntrigger = int(spec["ntrigger"][()])
    chunk_files = []
    chunk_n_frames = []
    missing = []
    for k in chunk_keys:
        suffix = k.split("_")[-1]
        cf = os.path.join(master_dir, f"{prefix}_data_{suffix}.h5")
        if not os.path.exists(cf):
            missing.append(os.path.basename(cf))
            continue
        chunk_files.append(cf)
        with h5py.File(cf, "r") as f:
            chunk_n_frames.append(f["entry/data/data"].shape[0])
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)}/{len(chunk_keys)} chunk files for "
            f"{os.path.basename(master_path)}: {missing}"
        )
    return det_shape, dtype, ntrigger, chunk_files, chunk_n_frames


def load_arina(
    master_path: str,
    det_bin: int = 1,
    scan_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Load an arina 4D-STEM dataset with Metal GPU decompression.

    Decompresses bitshuffle+LZ4 data on the GPU and optionally bins the
    detector axes on the fly. With binning, only the smaller binned result
    is kept in memory, so datasets larger than RAM can be loaded.

    Parameters
    ----------
    master_path : str
        Path to the arina master HDF5 file.
    det_bin : int, optional
        Detector binning factor (applied to both axes), by default 1.
    scan_shape : tuple of (int, int), optional
        Reshape into (scan_rows, scan_cols, det_rows, det_cols). If None
        and det_bin > 1, inferred as (sqrt(n), sqrt(n)) from ntrigger.

    Returns
    -------
    np.ndarray
        If scan_shape: (scan_rows, scan_cols, det_rows, det_cols) float32.
        Otherwise: (n_frames, det_rows, det_cols) in original dtype.

    Examples
    --------
    >>> data = load_arina("SnMoS2s_001_master.h5")
    >>> data.shape
    (262144, 192, 192)

    >>> data = load_arina("SnMoS2s_001_master.h5", det_bin=2)
    >>> data.shape
    (512, 512, 96, 96)
    """
    t0 = time.perf_counter()
    det_shape, dtype, ntrigger, chunk_files, chunk_n_frames = _parse_master(
        master_path
    )
    total_frames = sum(chunk_n_frames)
    det_row, det_col = det_shape
    elem_size = np.dtype(dtype).itemsize
    frame_bytes = int(np.prod(det_shape) * elem_size)
    n_blocks_per_frame = (frame_bytes + 8191) // 8192
    max_chunk_frames = max(chunk_n_frames)
    dec = _get_decompressor(frame_bytes, max_frames=max_chunk_frames)
    # Compute output shape
    if det_bin > 1:
        out_det_row = det_row // det_bin
        out_det_col = det_col // det_bin
        out_dtype = np.float32
    else:
        out_det_row, out_det_col = det_row, det_col
        out_dtype = dtype
    # Warmup numba JIT (skip if already warmed)
    if not getattr(dec, '_jit_warm', False):
        dec.load(chunk_files[0], n_frames=1, verbose=False)
        dec._jit_warm = True
    if det_bin > 1:
        # GPU pipeline: LZ4 → bitshuffle → bin, with double-buffered reads
        out_frame_bytes = out_det_row * out_det_col * 4  # float32
        total_out_bytes = total_frames * out_frame_bytes
        dec._ensure_output_buffer(total_out_bytes)
        # Warmup binned GPU pipeline (triggers Metal lazy buffer mapping)
        if not getattr(dec, '_bin_warm', False):
            bufs_0 = (dec._comp_np, dec._co_np, dec._bs_np, dec._bc_np,
                       dec._bo_np, dec._chunk_sizes,
                       dec._comp_mtl, dec._co_mtl, dec._bs_mtl, dec._bc_mtl,
                       dec._bo_mtl)
            comp_np_w, co_np_w, bs_np_w, bc_np_w, bo_np_w, csizes_w, \
                comp_mtl_w, co_mtl_w, bs_mtl_w, bc_mtl_w, bo_mtl_w = bufs_0
            dec._read_chunk(chunk_files[0], comp_np_w, co_np_w, csizes_w)
            _parse_headers(comp_np_w, csizes_w, co_np_w, bs_np_w, bc_np_w,
                           1, n_blocks_per_frame)
            bo_np_w[0] = 0
            bo_np_w[1] = int(bc_np_w[0])
            cmd_w = dec._submit_gpu_binned(
                1, frame_bytes, elem_size, 0,
                det_row, det_col, det_bin,
                comp_mtl_w, co_mtl_w, bs_mtl_w, bc_mtl_w, bo_mtl_w,
                int(bc_np_w[0]),
            )
            cmd_w.waitUntilCompleted()
            dec._bin_warm = True
        bufs = [
            (dec._comp_np, dec._co_np, dec._bs_np, dec._bc_np,
             dec._bo_np, dec._chunk_sizes,
             dec._comp_mtl, dec._co_mtl, dec._bs_mtl, dec._bc_mtl,
             dec._bo_mtl),
            (dec._comp_np_b, dec._co_np_b, dec._bs_np_b, dec._bc_np_b,
             dec._bo_np_b, dec._chunk_sizes_b,
             dec._comp_mtl_b, dec._co_mtl_b, dec._bs_mtl_b, dec._bc_mtl_b,
             dec._bo_mtl_b),
        ]
        # Read first chunk
        comp_np, co_np, bs_np, bc_np, bo_np, csizes, \
            comp_mtl, co_mtl, bs_mtl, bc_mtl, bo_mtl = bufs[0]
        dec._read_chunk(chunk_files[0], comp_np, co_np, csizes)
        _parse_headers(comp_np, csizes, co_np, bs_np, bc_np,
                       chunk_n_frames[0], n_blocks_per_frame)
        bo_np[0] = 0
        bo_np[1 : chunk_n_frames[0] + 1] = np.cumsum(bc_np[:chunk_n_frames[0]])
        max_blk = int(bc_np[:chunk_n_frames[0]].max())
        frame_offset = 0
        n_chunks = len(chunk_files)
        for ci in range(n_chunks):
            cur = bufs[ci % 2]
            comp_np, co_np, bs_np, bc_np, bo_np, csizes, \
                comp_mtl, co_mtl, bs_mtl, bc_mtl, bo_mtl = cur
            nf = chunk_n_frames[ci]
            out_byte_offset = frame_offset * out_frame_bytes
            cmd = dec._submit_gpu_binned(
                nf, frame_bytes, elem_size, out_byte_offset,
                det_row, det_col, det_bin,
                comp_mtl, co_mtl, bs_mtl, bc_mtl, bo_mtl, max_blk,
            )
            # Read + parse next chunk while GPU runs
            if ci + 1 < n_chunks:
                nxt = bufs[(ci + 1) % 2]
                comp_np_n, co_np_n, bs_np_n, bc_np_n, bo_np_n, csizes_n, \
                    *_ = nxt
                dec._read_chunk(chunk_files[ci + 1], comp_np_n, co_np_n,
                                csizes_n)
                nf_next = chunk_n_frames[ci + 1]
                _parse_headers(comp_np_n, csizes_n, co_np_n, bs_np_n,
                               bc_np_n, nf_next, n_blocks_per_frame)
                bo_np_n = nxt[4]
                bo_np_n[0] = 0
                bo_np_n[1 : nf_next + 1] = np.cumsum(bc_np_n[:nf_next])
                max_blk = int(bc_np_n[:nf_next].max())
            cmd.waitUntilCompleted()
            frame_offset += nf
        output = dec._out_np[:total_out_bytes].view(np.float32).reshape(
            total_frames, out_det_row, out_det_col
        )
    else:
        # No binning — use load_master for double-buffered raw decompression
        output = dec.load_master(chunk_files[0].replace(
            "_data_000001.h5", "_master.h5"
        ))
    t_total = time.perf_counter()
    # Infer scan shape
    if scan_shape is None and det_bin > 1:
        side = int(total_frames ** 0.5)
        if side * side == total_frames:
            scan_shape = (side, side)
    if scan_shape is not None:
        output = output.reshape(*scan_shape, out_det_row, out_det_col)
    print(
        f"load_arina: {total_frames} frames, "
        f"det ({det_row},{det_col}) → ({out_det_row},{out_det_col}), "
        f"{t_total - t0:.2f}s"
    )
    return output
