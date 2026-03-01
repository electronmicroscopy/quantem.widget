"""CUDA-accelerated bitshuffle+LZ4 decompression for arina 4D-STEM data.

Uses CuPy + CUDA compute kernels on NVIDIA GPUs.

Follows the same approach as ``quantem.cuda.io``: read all chunks through
the master file in a single pass, parse headers with numba, then run
LZ4 + bitshuffle (+ optional binning) kernels on the GPU.

Prefer the public API via ``IO.arina_file()``::

    from quantem.widget import IO
    data = IO.arina_file("master.h5", det_bin=2)

This module is the CUDA backend; ``IO.arina_file()`` auto-detects the GPU.
"""

from __future__ import annotations

import os
import re
import time

import cupy as cp
import h5py
import hdf5plugin  # noqa: F401 - registers bitshuffle filter
import numpy as np
from numba import njit, prange
from tqdm.auto import tqdm

__all__ = ["load_arina", "free_gpu"]

BLOCK_SIZE = 8192

# ---------------------------------------------------------------------------
# CUDA kernel source — decompression + bitshuffle + binning
#
# The LZ4 decompression kernel (h5lz4dc_batched) is derived from NVIDIA
# nvcomp, licensed under the BSD 3-Clause License:
#
#   Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
#  Please see the NOTICE file.

_CUDA_SOURCE = r'''
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef long long int64_t;

using offset_type = uint16_t;
using word_type = uint32_t;
using position_type = uint32_t;
using double_word_type = uint64_t;
using item_type = uint32_t;

constexpr const int DECOMP_THREADS_PER_CHUNK = 32;
constexpr const int DECOMP_CHUNKS_PER_BLOCK = 2;
constexpr const position_type DECOMP_INPUT_BUFFER_SIZE
    = DECOMP_THREADS_PER_CHUNK * sizeof(double_word_type);
constexpr const position_type DECOMP_BUFFER_PREFETCH_DIST
    = DECOMP_INPUT_BUFFER_SIZE / 2;

inline __device__ void syncCTA() {
    if (DECOMP_THREADS_PER_CHUNK > 32) __syncthreads();
    else __syncwarp();
}

inline __device__ int warpBallot(int vote) {
    return __ballot_sync(0xffffffff, vote);
}

inline __device__ offset_type readWord(const uint8_t* const address) {
    offset_type word = 0;
    for (size_t i = 0; i < sizeof(offset_type); ++i)
        word |= address[i] << (8 * i);
    return word;
}

struct token_type {
    position_type num_literals;
    position_type num_matches;

    __device__ bool hasNumLiteralsOverflow() const { return num_literals >= 15; }
    __device__ bool hasNumMatchesOverflow() const { return num_matches >= 19; }

    __device__ position_type numLiteralsOverflow() const {
        return hasNumLiteralsOverflow() ? num_literals - 15 : 0;
    }
    __device__ uint8_t numLiteralsForHeader() const {
        return hasNumLiteralsOverflow() ? 15 : num_literals;
    }
    __device__ position_type numMatchesOverflow() const {
        return hasNumMatchesOverflow() ? num_matches - 19 : 0;
    }
    __device__ uint8_t numMatchesForHeader() const {
        return hasNumMatchesOverflow() ? 15 : num_matches - 4;
    }
    __device__ position_type lengthOfLiteralEncoding() const {
        if (hasNumLiteralsOverflow()) {
            const position_type num = numLiteralsOverflow();
            return (num / 0xff) + 1;
        }
        return 0;
    }
    __device__ position_type lengthOfMatchEncoding() const {
        if (hasNumMatchesOverflow()) {
            const position_type num = numMatchesOverflow();
            return (num / 0xff) + 1;
        }
        return 0;
    }
};

class BufferControl {
public:
    __device__ BufferControl(uint8_t* const buffer, const uint8_t* const compData,
                             const position_type length)
        : m_offset(0), m_length(length), m_buffer(buffer), m_compData(compData) {}

    inline __device__ position_type readLSIC(position_type& idx) const {
        position_type num = 0;
        uint8_t next = 0xff;
        while (next == 0xff && idx < end()) {
            next = rawAt(idx)[0];
            ++idx;
            num += next;
        }
        while (next == 0xff) {
            next = m_compData[idx];
            ++idx;
            num += next;
        }
        return num;
    }

    inline __device__ const uint8_t* raw() const { return m_buffer; }
    inline __device__ const uint8_t* rawAt(const position_type i) const {
        return raw() + (i - begin());
    }

    inline __device__ uint8_t operator[](const position_type i) const {
        if (i >= m_offset && i - m_offset < DECOMP_INPUT_BUFFER_SIZE)
            return m_buffer[i - m_offset];
        return m_compData[i];
    }

    inline __device__ void setAndAlignOffset(const position_type offset) {
        const uint8_t* const alignedPtr = reinterpret_cast<const uint8_t*>(
            (reinterpret_cast<size_t>(m_compData + offset) / sizeof(double_word_type))
            * sizeof(double_word_type));
        m_offset = alignedPtr - m_compData;
    }

    inline __device__ void loadAt(const position_type offset) {
        setAndAlignOffset(offset);
        if (m_offset + DECOMP_INPUT_BUFFER_SIZE <= m_length) {
            const double_word_type* const word_data
                = reinterpret_cast<const double_word_type*>(m_compData + m_offset);
            double_word_type* const word_buffer
                = reinterpret_cast<double_word_type*>(m_buffer);
            word_buffer[threadIdx.x] = word_data[threadIdx.x];
        } else {
            #pragma unroll
            for (int i = threadIdx.x; i < DECOMP_INPUT_BUFFER_SIZE;
                 i += DECOMP_THREADS_PER_CHUNK) {
                if (m_offset + i < m_length)
                    m_buffer[i] = m_compData[m_offset + i];
            }
        }
        syncCTA();
    }

    inline __device__ position_type begin() const { return m_offset; }
    inline __device__ position_type end() const { return m_offset + DECOMP_INPUT_BUFFER_SIZE; }

private:
    int64_t m_offset;
    const position_type m_length;
    uint8_t* const m_buffer;
    const uint8_t* const m_compData;
};

inline __device__ void coopCopyNoOverlap(uint8_t* const dest, const uint8_t* const source,
                                         const position_type length) {
    for (position_type i = threadIdx.x; i < length; i += blockDim.x)
        dest[i] = source[i];
}

inline __device__ void coopCopyRepeat(uint8_t* const dest, const uint8_t* const source,
                                      const position_type dist, const position_type length) {
    for (position_type i = threadIdx.x; i < length; i += blockDim.x)
        dest[i] = source[i % dist];
}

inline __device__ void coopCopyOverlap(uint8_t* const dest, const uint8_t* const source,
                                       const position_type dist, const position_type length) {
    if (dist < length) coopCopyRepeat(dest, source, dist, length);
    else coopCopyNoOverlap(dest, source, length);
}

inline __device__ token_type decodePair(const uint8_t num) {
    return token_type{static_cast<uint8_t>((num & 0xf0) >> 4),
                      static_cast<uint8_t>(num & 0x0f)};
}

inline __device__ void decompressStream(uint8_t* buffer, uint8_t* decompData,
                                        const uint8_t* compData, const position_type comp_end) {
    BufferControl ctrl(buffer, compData, comp_end);
    ctrl.loadAt(0);
    position_type decomp_idx = 0;
    position_type comp_idx = 0;

    while (comp_idx < comp_end) {
        if (comp_idx + DECOMP_BUFFER_PREFETCH_DIST > ctrl.end())
            ctrl.loadAt(comp_idx);

        token_type tok = decodePair(*ctrl.rawAt(comp_idx));
        ++comp_idx;

        position_type num_literals = tok.num_literals;
        if (tok.num_literals == 15)
            num_literals += ctrl.readLSIC(comp_idx);
        const position_type literalStart = comp_idx;

        if (num_literals + comp_idx > ctrl.end())
            coopCopyNoOverlap(decompData + decomp_idx, compData + comp_idx, num_literals);
        else
            coopCopyNoOverlap(decompData + decomp_idx, ctrl.rawAt(comp_idx), num_literals);

        comp_idx += num_literals;
        decomp_idx += num_literals;

        if (comp_idx < comp_end) {
            offset_type offset;
            if (comp_idx + sizeof(offset_type) > ctrl.end())
                offset = readWord(compData + comp_idx);
            else
                offset = readWord(ctrl.rawAt(comp_idx));

            comp_idx += sizeof(offset_type);

            position_type match = 4 + tok.num_matches;
            if (tok.num_matches == 15)
                match += ctrl.readLSIC(comp_idx);

            if (offset <= num_literals
                && (ctrl.begin() <= literalStart && ctrl.end() >= literalStart + num_literals)) {
                coopCopyOverlap(decompData + decomp_idx,
                                ctrl.rawAt(literalStart + (num_literals - offset)), offset, match);
                syncCTA();
            } else {
                syncCTA();
                coopCopyOverlap(decompData + decomp_idx,
                                decompData + decomp_idx - offset, offset, match);
            }
            decomp_idx += match;
        }
    }
}

inline __device__ uint32_t read32be_batch(const uint8_t* address) {
    return ((uint32_t)(255 & address[0]) << 24 | (uint32_t)(255 & address[1]) << 16 |
            (uint32_t)(255 & address[2]) << 8  | (uint32_t)(255 & address[3]));
}

// =====================================================================
// Kernel 1: Batched LZ4 decompression
// Grid:  ((max_blocks+1)/2, 1, n_frames)
// Block: (32, 2, 1)
// =====================================================================
extern "C" __global__ void h5lz4dc_batched(
    const uint8_t* const compressed, const uint32_t* const chunk_offsets,
    const uint32_t* const block_starts, const uint32_t* const block_counts,
    const uint32_t* const block_offsets, const uint32_t blocksize,
    const uint32_t frame_bytes, uint8_t* const decompressed
) {
    const int frame_id = blockIdx.z;
    const int block_in_frame = blockIdx.x * blockDim.y + threadIdx.y;
    const uint32_t chunk_offset = chunk_offsets[frame_id];
    const uint32_t block_offset = block_offsets[frame_id];
    const uint32_t num_blocks = block_counts[frame_id];
    __shared__ uint8_t buffer[DECOMP_INPUT_BUFFER_SIZE * DECOMP_CHUNKS_PER_BLOCK];

    if (block_in_frame < num_blocks) {
        const uint32_t block_start = block_starts[block_offset + block_in_frame];
        const uint8_t* input = compressed + chunk_offset + block_start + 4;
        const uint32_t comp_size = read32be_batch(compressed + chunk_offset + block_start);
        uint8_t* output = decompressed + frame_id * frame_bytes + block_in_frame * blocksize;
        decompressStream(buffer + threadIdx.y * DECOMP_INPUT_BUFFER_SIZE, output, input, comp_size);
    }
}

// =====================================================================
// Kernel 2a: Batched bitshuffle unshuffle (8192-byte blocks, 32-bit)
// Grid:  (n_8kb_blocks, 1, n_frames)
// Block: (32, 32, 1)
// =====================================================================
extern "C" __global__ void shuf_8192_32_batched(
    const uint32_t* __restrict__ in, uint32_t* __restrict__ out, const uint32_t frame_u32s
) {
    const int frame_id = blockIdx.z;
    const uint32_t* frame_in = in + frame_id * frame_u32s;
    uint32_t* frame_out = out + frame_id * frame_u32s;
    __shared__ uint32_t smem[32][33];

    smem[threadIdx.y][threadIdx.x] = frame_in[threadIdx.x + threadIdx.y * 64 +
                                               blockIdx.x * 2048 + blockIdx.y * 32];
    __syncthreads();

    uint32_t v = smem[threadIdx.x][threadIdx.y];
    #pragma unroll 32
    for (int i = 0; i < 32; i++)
        smem[i][threadIdx.y] = __ballot_sync(0xFFFFFFFFU, v & (1U << i));
    __syncthreads();

    frame_out[threadIdx.x + threadIdx.y * 32 + blockIdx.y * 1024 + blockIdx.x * 2048] =
        smem[threadIdx.x][threadIdx.y];
}

// =====================================================================
// Kernel 2b: Batched bitshuffle unshuffle (8192-byte blocks, 16-bit)
// Grid:  (n_8kb_blocks, 16, n_frames)
// Block: (256, 1, 1)
// =====================================================================
extern "C" __global__ void shuf_8192_16_batched(
    const uint8_t* __restrict__ in, uint16_t* __restrict__ out, const uint32_t frame_bytes
) {
    const int frame_id = blockIdx.z;
    const uint8_t* frame_in = in + frame_id * frame_bytes;
    uint16_t* frame_out = out + (frame_id * frame_bytes) / 2;

    const int block_8kb = blockIdx.x;
    const int group = blockIdx.y;
    const int tid = threadIdx.x;
    const int elem_in_block = group * 256 + tid;

    __shared__ uint8_t smem[16][32];

    if (tid < 32) {
        const uint8_t* block_base = frame_in + block_8kb * 8192;
        #pragma unroll 16
        for (int b = 0; b < 16; b++) {
            smem[b][tid] = block_base[b * 512 + group * 32 + tid];
        }
    }
    __syncthreads();

    const int byte_in_group = tid / 8;
    const int bit_in_byte = tid % 8;

    uint16_t result = 0;
    #pragma unroll 16
    for (int b = 0; b < 16; b++) {
        if (smem[b][byte_in_group] & (1U << bit_in_byte)) {
            result |= (1U << b);
        }
    }

    const int out_idx = block_8kb * 4096 + elem_in_block;
    frame_out[out_idx] = result;
}

// =====================================================================
// Kernel 3a: Detector binning (uint16 → float32)
// Grid:  (ceil(out_cols/16), ceil(out_rows/16), n_frames)
// Block: (16, 16, 1)
// =====================================================================
extern "C" __global__ void bin_mean_u16(
    const uint16_t* __restrict__ in, float* __restrict__ out,
    const uint32_t in_row_stride, const uint32_t in_frame_elems,
    const uint32_t out_cols, const uint32_t out_frame_elems,
    const uint32_t bin_factor
) {
    uint32_t out_c = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t out_r = blockIdx.y * blockDim.y + threadIdx.y;
    if (out_c >= out_cols || out_r >= out_cols) return;
    uint32_t frame = blockIdx.z;
    uint32_t in_r = out_r * bin_factor;
    uint32_t in_c = out_c * bin_factor;
    uint32_t in_base = frame * in_frame_elems;
    float sum = 0.0f;
    for (uint32_t br = 0; br < bin_factor; br++)
        for (uint32_t bc = 0; bc < bin_factor; bc++)
            sum += float(in[in_base + (in_r + br) * in_row_stride + in_c + bc]);
    out[frame * out_frame_elems + out_r * out_cols + out_c] =
        sum / float(bin_factor * bin_factor);
}

// =====================================================================
// Kernel 3b: Detector binning (uint32 → float32)
// Grid:  (ceil(out_cols/16), ceil(out_rows/16), n_frames)
// Block: (16, 16, 1)
// =====================================================================
extern "C" __global__ void bin_mean_u32(
    const uint32_t* __restrict__ in, float* __restrict__ out,
    const uint32_t in_row_stride, const uint32_t in_frame_elems,
    const uint32_t out_cols, const uint32_t out_frame_elems,
    const uint32_t bin_factor
) {
    uint32_t out_c = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t out_r = blockIdx.y * blockDim.y + threadIdx.y;
    if (out_c >= out_cols || out_r >= out_cols) return;
    uint32_t frame = blockIdx.z;
    uint32_t in_r = out_r * bin_factor;
    uint32_t in_c = out_c * bin_factor;
    uint32_t in_base = frame * in_frame_elems;
    float sum = 0.0f;
    for (uint32_t br = 0; br < bin_factor; br++)
        for (uint32_t bc = 0; bc < bin_factor; bc++)
            sum += float(in[in_base + (in_r + br) * in_row_stride + in_c + bc]);
    out[frame * out_frame_elems + out_r * out_cols + out_c] =
        sum / float(bin_factor * bin_factor);
}
'''

# ---------------------------------------------------------------------------
# Compile CUDA kernels at import time
# ---------------------------------------------------------------------------
_cuda_module = cp.RawModule(code=_CUDA_SOURCE, options=("-std=c++11", "-w"))
_h5lz4dc_kernel = _cuda_module.get_function("h5lz4dc_batched")
_shuf32_kernel = _cuda_module.get_function("shuf_8192_32_batched")
_shuf16_kernel = _cuda_module.get_function("shuf_8192_16_batched")
_bin_u16_kernel = _cuda_module.get_function("bin_mean_u16")
_bin_u32_kernel = _cuda_module.get_function("bin_mean_u32")


# ---------------------------------------------------------------------------
# Header parser (numba, runs on CPU in parallel)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# GPUDecompressor — single buffer set, no double-buffering
# ---------------------------------------------------------------------------
class GPUDecompressor:
    """GPU-accelerated decompressor for bitshuffle+LZ4 HDF5 datasets.

    Pre-allocates pinned host + GPU buffers sized for the dataset.
    Reused across calls via module-level ``_default_decompressor``.
    """

    def __init__(
        self,
        max_compressed_bytes: int = 1024 * 1024 * 1024,
        max_frames: int = 100_000,
        max_frame_bytes: int = 192 * 192 * 4,
        n_blocks_per_frame: int = 18,
    ):
        self.max_compressed_bytes = max_compressed_bytes
        self.max_frames = max_frames
        self.max_frame_bytes = max_frame_bytes
        self.n_blocks_per_frame = n_blocks_per_frame

        # Pinned host memory
        self._pinned_mem = cp.cuda.alloc_pinned_memory(max_compressed_bytes)
        self._pinned_buffer = np.frombuffer(
            self._pinned_mem, dtype=np.uint8, count=max_compressed_bytes
        )
        self._chunk_sizes = np.zeros(max_frames, dtype=np.uint32)
        self._chunk_offsets = np.zeros(max_frames, dtype=np.uint32)
        self._block_counts = np.zeros(max_frames, dtype=np.uint32)
        self._block_starts_flat = np.zeros(
            max_frames * n_blocks_per_frame, dtype=np.uint32
        )
        self._block_offsets = np.zeros(max_frames + 1, dtype=np.uint32)

        # GPU buffers
        self._concat_gpu = cp.empty(max_compressed_bytes, dtype=cp.uint8)
        total_output_bytes = max_frames * max_frame_bytes
        self._lz4_output = cp.empty(total_output_bytes, dtype=cp.uint8)
        self._shuffled_output = cp.empty(total_output_bytes, dtype=cp.uint8)


_default_decompressor: GPUDecompressor | None = None


def _get_or_create_decompressor(
    total_frames: int,
    frame_bytes: int,
    n_blocks_per_frame: int,
    total_compressed_estimate: int,
) -> GPUDecompressor:
    """Return (and cache) a GPUDecompressor large enough for the dataset."""
    global _default_decompressor
    max_compressed = max(total_compressed_estimate, 1024 * 1024 * 1024)

    needs_new = (
        _default_decompressor is None
        or total_frames > _default_decompressor.max_frames
        or frame_bytes > _default_decompressor.max_frame_bytes
        or max_compressed > _default_decompressor.max_compressed_bytes
        or n_blocks_per_frame > _default_decompressor.n_blocks_per_frame
    )
    if needs_new:
        _default_decompressor = GPUDecompressor(
            max_compressed_bytes=max_compressed,
            max_frames=total_frames + 1000,
            max_frame_bytes=frame_bytes,
            n_blocks_per_frame=n_blocks_per_frame,
        )
    return _default_decompressor


# ---------------------------------------------------------------------------
# Core loader — reads all chunks through master, single GPU pass
# ---------------------------------------------------------------------------
def _load_master_gpu(
    master_path: str,
    det_bin: int = 1,
) -> tuple[np.ndarray, tuple[int, int], np.dtype, int]:
    """Load all chunks from an arina master file via GPU.

    Reads all frames into pinned memory in a single pass through the
    master file, then runs LZ4 + bitshuffle (+ optional binning) on GPU.

    Returns (output_array, det_shape, dtype, total_frames).
    """
    t0 = time.perf_counter()

    with h5py.File(master_path, "r") as f:
        data_group = f["entry/data"]
        chunk_names = sorted(
            name for name in data_group.keys()
            if re.match(r"data_\d{6}", name)
        )

        # Get metadata from first chunk
        first_ds = data_group[chunk_names[0]]
        frame_shape = first_ds.shape[1:]
        dtype = first_ds.dtype
        elem_size = np.dtype(dtype).itemsize
        frame_bytes = int(np.prod(frame_shape)) * elem_size
        n_blocks_per_frame = frame_bytes // BLOCK_SIZE
        det_row, det_col = frame_shape

        # Count total frames
        chunk_frame_counts = [data_group[name].shape[0] for name in chunk_names]
        total_frames = sum(chunk_frame_counts)

        # Estimate compressed size
        estimated_compressed = int(total_frames * frame_bytes * 0.8)
        max_compressed = max(estimated_compressed, 1024 * 1024 * 1024)

        # Get or create decompressor
        dec = _get_or_create_decompressor(
            total_frames, frame_bytes, n_blocks_per_frame, max_compressed
        )

        # Read ALL chunks into pinned memory in one pass
        offset = 0
        frame_idx = 0
        chunk_iter = chunk_names
        if len(chunk_names) > 1:
            chunk_iter = tqdm(chunk_names, desc="reading chunks", leave=False)
        for chunk_name in chunk_iter:
            ds = data_group[chunk_name]
            n_chunk_frames = ds.shape[0]
            for i in range(n_chunk_frames):
                _, raw = ds.id.read_direct_chunk((i, 0, 0))
                chunk_len = len(raw)
                dec._chunk_offsets[frame_idx] = offset
                dec._chunk_sizes[frame_idx] = chunk_len
                dec._pinned_buffer[offset : offset + chunk_len] = (
                    np.frombuffer(raw, dtype=np.uint8)
                )
                offset += chunk_len
                frame_idx += 1
        total_compressed = offset

    t_read = time.perf_counter()

    # Parse headers (numba parallel)
    _parse_headers(
        dec._pinned_buffer,
        dec._chunk_sizes,
        dec._chunk_offsets,
        dec._block_starts_flat,
        dec._block_counts,
        total_frames,
        n_blocks_per_frame,
    )
    dec._block_offsets[1 : total_frames + 1] = np.cumsum(
        dec._block_counts[:total_frames]
    )
    total_blocks = int(dec._block_offsets[total_frames])
    t_parse = time.perf_counter()

    # Transfer compressed data + metadata to GPU
    dec._concat_gpu[:total_compressed].set(dec._pinned_buffer[:total_compressed])
    chunk_offsets_gpu = cp.asarray(dec._chunk_offsets[:total_frames])
    block_starts_gpu = cp.asarray(dec._block_starts_flat[:total_blocks])
    block_counts_gpu = cp.asarray(dec._block_counts[:total_frames])
    block_offsets_gpu = cp.asarray(dec._block_offsets[: total_frames + 1])

    # LZ4 decompress in batches of 10000
    max_blocks = int(dec._block_counts[:total_frames].max())
    max_batch = 10000
    for start in range(0, total_frames, max_batch):
        end = min(start + max_batch, total_frames)
        batch_n = end - start
        byte_offset = start * frame_bytes
        _h5lz4dc_kernel(
            ((max_blocks + 1) // 2, 1, batch_n),
            (32, 2, 1),
            (
                dec._concat_gpu,
                chunk_offsets_gpu[start:],
                block_starts_gpu,
                block_counts_gpu[start:],
                block_offsets_gpu[start:],
                np.uint32(BLOCK_SIZE),
                np.uint32(frame_bytes),
                dec._lz4_output[byte_offset:],
            ),
        )

    # Bitshuffle
    n_8kb = frame_bytes // BLOCK_SIZE
    if elem_size == 2:
        for start in range(0, total_frames, max_batch):
            end = min(start + max_batch, total_frames)
            batch_n = end - start
            byte_offset = start * frame_bytes
            _shuf16_kernel(
                (n_8kb, 16, batch_n),
                (256, 1, 1),
                (
                    dec._lz4_output[byte_offset:],
                    dec._shuffled_output[byte_offset:].view(cp.uint16),
                    np.uint32(frame_bytes),
                ),
            )
    else:
        frame_u32s = frame_bytes // 4
        for start in range(0, total_frames, max_batch):
            end = min(start + max_batch, total_frames)
            batch_n = end - start
            byte_offset = start * frame_bytes
            _shuf32_kernel(
                (n_8kb, 2, batch_n),
                (32, 32, 1),
                (
                    dec._lz4_output[byte_offset:].view(cp.uint32),
                    dec._shuffled_output[byte_offset:].view(cp.uint32),
                    np.uint32(frame_u32s),
                ),
            )

    cp.cuda.Device().synchronize()
    t_gpu = time.perf_counter()

    # Get decompressed result on GPU
    total_output_bytes = total_frames * frame_bytes
    decompressed_gpu = dec._shuffled_output[:total_output_bytes].view(dtype).reshape(
        (total_frames,) + frame_shape
    )

    # Binning (on GPU) or copy to CPU
    if det_bin > 1:
        out_det_row = det_row // det_bin
        out_det_col = det_col // det_bin
        in_frame_elems = det_row * det_col
        out_frame_elems = out_det_row * out_det_col

        out_gpu = cp.empty(total_frames * out_frame_elems, dtype=cp.float32)

        bin_kernel = _bin_u16_kernel if elem_size == 2 else _bin_u32_kernel
        grid_x = (out_det_col + 15) // 16
        grid_y = (out_det_row + 15) // 16

        # Bin in batches to keep grid z-dim within limits
        for start in range(0, total_frames, max_batch):
            end = min(start + max_batch, total_frames)
            batch_n = end - start
            in_view = decompressed_gpu[start:end].ravel()
            out_view = out_gpu[start * out_frame_elems : end * out_frame_elems]
            bin_kernel(
                (grid_x, grid_y, batch_n),
                (16, 16, 1),
                (
                    in_view,
                    out_view,
                    np.uint32(det_col),
                    np.uint32(in_frame_elems),
                    np.uint32(out_det_col),
                    np.uint32(out_frame_elems),
                    np.uint32(det_bin),
                ),
            )
        cp.cuda.Device().synchronize()

        output = out_gpu.reshape(total_frames, out_det_row, out_det_col)
    else:
        out_det_row, out_det_col = det_row, det_col
        output = decompressed_gpu

    t_total = time.perf_counter()
    print(
        f"_load_master_gpu: {total_frames} frames, "
        f"read {t_read - t0:.3f}s, parse {t_parse - t_read:.3f}s, "
        f"gpu {t_gpu - t_parse:.3f}s, "
        f"total {t_total - t0:.3f}s"
    )

    return output, (det_row, det_col), dtype, total_frames


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_arina(
    master_path: str,
    det_bin: int = 1,
    scan_shape: tuple[int, int] | None = None,
):
    """Load an arina 4D-STEM dataset with CUDA GPU decompression.

    Reads all chunks through the master file in a single pass, then runs
    LZ4 + bitshuffle (+ optional binning) kernels on the GPU.
    Returns a CuPy array (data stays on GPU).

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
    cupy.ndarray
        If scan_shape: (scan_rows, scan_cols, det_rows, det_cols) float32.
        Otherwise: (n_frames, det_rows, det_cols) in original dtype.
    """
    t0 = time.perf_counter()

    output, (det_row, det_col), dtype, total_frames = _load_master_gpu(
        master_path, det_bin=det_bin
    )

    if det_bin > 1:
        out_det_row = det_row // det_bin
        out_det_col = det_col // det_bin
    else:
        out_det_row, out_det_col = det_row, det_col

    # Infer scan shape
    if scan_shape is None and det_bin > 1:
        side = int(total_frames ** 0.5)
        if side * side == total_frames:
            scan_shape = (side, side)
    if scan_shape is not None:
        output = output.reshape(*scan_shape, out_det_row, out_det_col)

    t_total = time.perf_counter()
    print(
        f"load_arina: {total_frames} frames, "
        f"det ({det_row},{det_col}) → ({out_det_row},{out_det_col}), "
        f"{t_total - t0:.2f}s"
    )
    return output


def free_gpu():
    """Release all CUDA GPU memory held by the decompressor and CuPy pool."""
    import gc

    global _default_decompressor
    _default_decompressor = None
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    free = cp.cuda.Device().mem_info[0]
    print(f"GPU free: {free / 1024**3:.1f} GB")
