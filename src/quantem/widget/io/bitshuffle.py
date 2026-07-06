"""CUDA kernels for bitshuffle+LZ4 compression and decompression.

This module contains the raw CUDA kernel source code and compiled kernels.
Kernels are compiled at module import time for fast first load().

Kernels
-------
h5lz4dc_kernel : LZ4 decompression kernel (batched)
bitshuffle_kernel : Inverse bitshuffle kernel (decompression)
bitshuffle_fwd_kernel : Forward bitshuffle kernel (compression)
lz4_compress_kernel : LZ4 compression kernel
compact_kernel : Compact scattered compressed blocks
"""
from __future__ import annotations

# cupy is the CUDA toolkit; it is absent on a Mac / plain laptop. Guard the
# import so this module loads anywhere — the kernels are compiled lazily on
# first access (see the module __getattr__ at the bottom), so a non-CUDA box
# can import quantem.widget.io without ever touching cupy.
try:
    import cupy as cp
except ImportError:  # pragma: no cover - exercised only on non-CUDA hosts
    cp = None

# =============================================================================
# CUDA KERNEL SOURCE CODE
# =============================================================================

_CUDA_LZ4_SOURCE = r'''
/*
 * LZ4 decompression kernel extracted from NVIDIA nvcomp
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 * BSD-3-Clause License
 */
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef long long int64_t;
typedef unsigned long size_t;

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

extern "C" __global__ void h5lz4dc_batched(
    const uint8_t* const compressed, const uint64_t* const chunk_offsets,
    const uint32_t* const block_starts, const uint32_t* const block_counts,
    const uint32_t* const block_offsets, const uint32_t blocksize,
    const uint32_t frame_bytes, uint8_t* const decompressed
) {
    const int frame_id = blockIdx.z;
    const int block_in_frame = blockIdx.x * blockDim.y + threadIdx.y;
    // chunk_offset is a 64-bit absolute byte offset into the read buffer:
    // when total compressed data exceeds 4 GB (e.g. a dense Arina gold
    // 4D-STEM acquisition with low LZ4 compression ratio) a uint32 here
    // would silently wrap and produce garbage reads.
    const uint64_t chunk_offset = chunk_offsets[frame_id];
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

// Optimized unshuffle for uint16 data using shared memory.
// Grid: (n_8kb_blocks, 1, n_frames), Block: (256, 1, 1)
// Each 8KB block = 4096 uint16 elements = 16 bitplanes x 512 bytes.
//
// Why this layout: the previous version launched the SAME 8 KB block as 16
// separate CTAs (one per blockIdx.y group) and inside each CTA only 32 of 256
// threads did the 512-byte smem load while 224 idled at __syncthreads(). That
// re-fetched every block 16x through L1 (25% hit rate) and bled ~49% of issue
// cycles to the load-imbalance barrier (ncu, 2026-06-03 profile).
//
// Here one CTA owns the whole 8 KB block: all 256 threads cooperatively load
// it into smem ONCE (coalesced uint32, 8 words/thread), one barrier, then each
// thread reconstructs 16 output elements (two byte-positions x 8 bits) from the
// resident block. This collapses the 16x redundant global loads, removes the
// idle-at-barrier stall (every thread participates in the load), and moves the
// kernel to the copy-bound roofline. Output layout is unchanged.
extern "C" __global__ void shuf_8192_16_batched(
    const uint8_t* __restrict__ in, uint16_t* __restrict__ out, const uint32_t frame_bytes
) {
    const int frame_id = blockIdx.z;
    const int block_8kb = blockIdx.x;
    const int tid = threadIdx.x;        // 0-255

    const uint8_t* block_base = in + (size_t)frame_id * frame_bytes + (size_t)block_8kb * 8192;
    uint16_t* frame_out = out + ((size_t)frame_id * frame_bytes) / 2 + (size_t)block_8kb * 4096;

    __shared__ uint8_t smem[8192];

    // Load the whole 8 KB block coalesced as 2048 uint32 words, 8 per thread.
    const uint32_t* src32 = reinterpret_cast<const uint32_t*>(block_base);
    uint32_t* sm32 = reinterpret_cast<uint32_t*>(smem);
    #pragma unroll 8
    for (int i = tid; i < 2048; i += 256) {
        sm32[i] = src32[i];
    }
    __syncthreads();

    // Each thread owns two byte-positions p in {tid, tid+256} (0-511), each of
    // which transposes the 16 plane-bits at that byte into 8 consecutive output
    // elements p*8 .. p*8+7. The 8 stores are contiguous and adjacent threads
    // write the next run of 8, so the global writes coalesce.
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        const int p = tid + half * 256;
        uint16_t r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0;
        #pragma unroll 16
        for (int b = 0; b < 16; b++) {
            const uint8_t byte = smem[b * 512 + p];
            const uint16_t bb = (uint16_t)1 << b;
            if (byte & 0x01) r0 |= bb;
            if (byte & 0x02) r1 |= bb;
            if (byte & 0x04) r2 |= bb;
            if (byte & 0x08) r3 |= bb;
            if (byte & 0x10) r4 |= bb;
            if (byte & 0x20) r5 |= bb;
            if (byte & 0x40) r6 |= bb;
            if (byte & 0x80) r7 |= bb;
        }
        uint16_t* o = frame_out + p * 8;
        o[0] = r0; o[1] = r1; o[2] = r2; o[3] = r3;
        o[4] = r4; o[5] = r5; o[6] = r6; o[7] = r7;
    }
}

// Simple fallback for uint16 (for debugging/verification)
extern "C" __global__ void shuf_8192_16_simple(
    const uint8_t* __restrict__ in, uint16_t* __restrict__ out, const uint32_t frame_bytes
) {
    const int frame_id = blockIdx.z;
    const uint8_t* frame_in = in + frame_id * frame_bytes;
    uint16_t* frame_out = out + (frame_id * frame_bytes) / 2;

    const int elem_idx = blockIdx.x * 256 + threadIdx.x;
    const int n_elems = frame_bytes / 2;

    if (elem_idx >= n_elems) return;

    const int block_8kb = elem_idx / 4096;
    const int elem_in_block = elem_idx % 4096;
    const int byte_in_plane = elem_in_block / 8;
    const int bit_in_byte = elem_in_block % 8;

    uint16_t result = 0;
    const uint8_t* block_base = frame_in + block_8kb * 8192;

    #pragma unroll 16
    for (int b = 0; b < 16; b++) {
        uint8_t byte_val = block_base[b * 512 + byte_in_plane];
        if (byte_val & (1U << bit_in_byte)) {
            result |= (1U << b);
        }
    }

    frame_out[elem_idx] = result;
}

extern "C" __global__ void shuf_tail_16_batched(
    const uint8_t* __restrict__ in, uint16_t* __restrict__ out, const uint32_t frame_bytes
) {
    const uint32_t tail_bytes = frame_bytes % 8192;
    if (tail_bytes == 0) return;

    const int frame_id = blockIdx.z;
    const uint8_t* frame_in = in + (uint64_t)frame_id * frame_bytes;
    uint16_t* frame_out = out + ((uint64_t)frame_id * frame_bytes) / 2;

    const uint32_t full_blocks = frame_bytes / 8192;
    const uint32_t tail_elems = tail_bytes / 2;
    const uint32_t bitplane_bytes = tail_elems / 8;
    const uint32_t elem = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem >= tail_elems) return;

    const uint8_t* block_base = frame_in + (uint64_t)full_blocks * 8192;
    const uint32_t byte_in_plane = elem / 8;
    const uint32_t bit_in_byte = elem % 8;

    uint16_t result = 0;
    #pragma unroll 16
    for (int b = 0; b < 16; b++) {
        const uint8_t byte_val = block_base[(uint64_t)b * bitplane_bytes + byte_in_plane];
        if (byte_val & (1U << bit_in_byte)) {
            result |= (1U << b);
        }
    }

    frame_out[full_blocks * 4096 + elem] = result;
}

extern "C" __global__ void shuf_tail_32_batched(
    const uint8_t* __restrict__ in, uint32_t* __restrict__ out, const uint32_t frame_bytes
) {
    const uint32_t tail_bytes = frame_bytes % 8192;
    if (tail_bytes == 0) return;

    const int frame_id = blockIdx.z;
    const uint8_t* frame_in = in + (uint64_t)frame_id * frame_bytes;
    uint32_t* frame_out = out + ((uint64_t)frame_id * frame_bytes) / 4;

    const uint32_t full_blocks = frame_bytes / 8192;
    const uint32_t tail_elems = tail_bytes / 4;
    const uint32_t bitplane_bytes = tail_elems / 8;
    const uint32_t elem = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem >= tail_elems) return;

    const uint8_t* block_base = frame_in + (uint64_t)full_blocks * 8192;
    const uint32_t byte_in_plane = elem / 8;
    const uint32_t bit_in_byte = elem % 8;

    uint32_t result = 0;
    #pragma unroll 32
    for (int b = 0; b < 32; b++) {
        const uint8_t byte_val = block_base[(uint64_t)b * bitplane_bytes + byte_in_plane];
        if (byte_val & (1U << bit_in_byte)) {
            result |= (1U << b);
        }
    }

    frame_out[full_blocks * 2048 + elem] = result;
}

// Forward bitshuffle for compression
// Takes normal data and outputs bitshuffled data
// Inverse of shuf_8192_32_batched (unshuffle)
extern "C" __global__ void bitshuffle_fwd_8192_32(
    const uint32_t* __restrict__ in, uint32_t* __restrict__ out, const uint32_t frame_u32s
) {
    const int frame_id = blockIdx.z;
    const uint32_t* frame_in = in + frame_id * frame_u32s;
    uint32_t* frame_out = out + frame_id * frame_u32s;
    __shared__ uint32_t smem[32][33];

    // Load from normal layout (same positions as unshuffle output)
    // Note: NO transpose here - we load smem[y][x] directly
    smem[threadIdx.y][threadIdx.x] = frame_in[threadIdx.x + threadIdx.y * 32 +
                                               blockIdx.y * 1024 + blockIdx.x * 2048];
    __syncthreads();

    // Bit transpose using ballot_sync (NO smem transpose before ballot)
    uint32_t v = smem[threadIdx.y][threadIdx.x];  // Read without transpose
    #pragma unroll 32
    for (int i = 0; i < 32; i++)
        smem[i][threadIdx.y] = __ballot_sync(0xFFFFFFFFU, v & (1U << i));
    __syncthreads();

    // Write to bitshuffled layout with transpose (same positions as unshuffle input)
    frame_out[threadIdx.x + threadIdx.y * 64 + blockIdx.x * 2048 + blockIdx.y * 32] =
        smem[threadIdx.y][threadIdx.x];  // Transpose on write
}


// Forward bitshuffle for uint16/float16 compression
// Inverse of shuf_8192_16_batched (unshuffle)
// Grid: (n_8kb_blocks, 16, n_frames), Block: (256, 1, 1)
// Each 8KB block = 4096 16-bit elements, split into 16 groups of 256
extern "C" __global__ void bitshuffle_fwd_8192_16(
    const uint16_t* __restrict__ in, uint8_t* __restrict__ out, const uint32_t frame_bytes
) {
    const int frame_id = blockIdx.z;
    const uint16_t* frame_in = in + ((uint64_t)frame_id * frame_bytes) / 2;
    uint8_t* frame_out = out + (uint64_t)frame_id * frame_bytes;

    const int block_8kb = blockIdx.x;
    const int group = blockIdx.y;
    const int tid = threadIdx.x;

    const int elem_in_block = group * 256 + tid;
    const int in_idx = block_8kb * 4096 + elem_in_block;
    const uint16_t val = frame_in[in_idx];

    const int byte_in_group = tid / 8;
    const int bit_in_byte = tid % 8;

    __shared__ uint8_t smem[16][32];

    if (tid < 32) {
        #pragma unroll 16
        for (int b = 0; b < 16; b++) {
            smem[b][tid] = 0;
        }
    }
    __syncthreads();

    #pragma unroll 16
    for (int b = 0; b < 16; b++) {
        if (val & (1U << b)) {
            atomicOr((unsigned int*)&smem[b][byte_in_group & ~3],
                     (1U << bit_in_byte) << (8 * (byte_in_group & 3)));
        }
    }
    __syncthreads();

    if (tid < 32) {
        uint8_t* block_base = frame_out + block_8kb * 8192;
        #pragma unroll 16
        for (int b = 0; b < 16; b++) {
            block_base[b * 512 + group * 32 + tid] = smem[b][tid];
        }
    }
}

extern "C" __global__ void bitshuffle_fwd_tail_16(
    const uint16_t* __restrict__ in, uint8_t* __restrict__ out, const uint32_t frame_bytes
) {
    const uint32_t tail_bytes = frame_bytes % 8192;
    if (tail_bytes == 0) return;

    const int frame_id = blockIdx.z;
    const uint16_t* frame_in = in + ((uint64_t)frame_id * frame_bytes) / 2;
    uint8_t* frame_out = out + (uint64_t)frame_id * frame_bytes;

    const uint32_t full_blocks = frame_bytes / 8192;
    const uint32_t tail_elems = tail_bytes / 2;
    const uint32_t bitplane_bytes = tail_elems / 8;
    const uint32_t byte_in_plane = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t bit = blockIdx.y;
    if (byte_in_plane >= bitplane_bytes) return;

    uint8_t packed = 0;
    #pragma unroll 8
    for (int k = 0; k < 8; k++) {
        const uint32_t elem = byte_in_plane * 8 + k;
        if (elem < tail_elems) {
            const uint16_t val = frame_in[full_blocks * 4096 + elem];
            if (val & (1U << bit)) {
                packed |= (1U << k);
            }
        }
    }

    uint8_t* block_base = frame_out + (uint64_t)full_blocks * 8192;
    block_base[(uint64_t)bit * bitplane_bytes + byte_in_plane] = packed;
}

extern "C" __global__ void bitshuffle_fwd_tail_32(
    const uint32_t* __restrict__ in, uint8_t* __restrict__ out, const uint32_t frame_bytes
) {
    const uint32_t tail_bytes = frame_bytes % 8192;
    if (tail_bytes == 0) return;

    const int frame_id = blockIdx.z;
    const uint32_t* frame_in = in + ((uint64_t)frame_id * frame_bytes) / 4;
    uint8_t* frame_out = out + (uint64_t)frame_id * frame_bytes;

    const uint32_t full_blocks = frame_bytes / 8192;
    const uint32_t tail_elems = tail_bytes / 4;
    const uint32_t bitplane_bytes = tail_elems / 8;
    const uint32_t byte_in_plane = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t bit = blockIdx.y;
    if (byte_in_plane >= bitplane_bytes) return;

    uint8_t packed = 0;
    #pragma unroll 8
    for (int k = 0; k < 8; k++) {
        const uint32_t elem = byte_in_plane * 8 + k;
        if (elem < tail_elems) {
            const uint32_t val = frame_in[full_blocks * 2048 + elem];
            if (val & (1U << bit)) {
                packed |= (1U << k);
            }
        }
    }

    uint8_t* block_base = frame_out + (uint64_t)full_blocks * 8192;
    block_base[(uint64_t)bit * bitplane_bytes + byte_in_plane] = packed;
}

// =============================================================================
// LZ4 Compression Kernel (Optimized)
// =============================================================================
// Uses shared memory hash table and parallel initialization.
// Thread 0 performs greedy encoding, all threads help with init.
//
// Optimizations:
// - Shared memory hash table (faster than global memory)
// - Parallel hash table initialization (32 threads)
// - Coalesced input reads where possible
// =============================================================================

#define LZ4_HASH_BITS 9
#define LZ4_HASH_SIZE (1 << LZ4_HASH_BITS)
#define LZ4_HASH_MASK (LZ4_HASH_SIZE - 1)
#define LZ4_MAX_MATCH_LEN 128
#define LZ4_WARP_SIZE 32

// Compute hash of 4-byte sequence (FNV-style multiplicative hash)
__device__ __forceinline__ uint32_t lz4_hash4(const uint8_t* ptr) {
    uint32_t seq = (uint32_t)ptr[0] |
                   ((uint32_t)ptr[1] << 8) |
                   ((uint32_t)ptr[2] << 16) |
                   ((uint32_t)ptr[3] << 24);
    return ((seq * 2654435761U) >> (32 - LZ4_HASH_BITS)) & LZ4_HASH_MASK;
}

// Load 4 bytes as uint32 safely (handles unaligned addresses)
__device__ __forceinline__ uint32_t lz4_load_u32(const uint8_t* ptr) {
    return (uint32_t)ptr[0] |
           ((uint32_t)ptr[1] << 8) |
           ((uint32_t)ptr[2] << 16) |
           ((uint32_t)ptr[3] << 24);
}

// Find match length between two positions
// LZ4 requires last 5 bytes to be literals, so limit match length accordingly
__device__ __forceinline__ uint32_t lz4_match_length(
    const uint8_t* in,
    uint32_t pos,
    uint32_t match_pos,
    uint32_t chunk_size
) {
    uint32_t len = 0;
    // Leave at least 5 bytes for final literals (LZ4 format requirement)
    uint32_t end_limit = chunk_size > 5 ? chunk_size - 5 : 0;
    uint32_t max_from_pos = pos < end_limit ? end_limit - pos : 0;
    uint32_t max_len = min(max_from_pos, (uint32_t)LZ4_MAX_MATCH_LEN);

    // Compare byte by byte (safe for any alignment)
    while (len < max_len && in[pos + len] == in[match_pos + len]) {
        len++;
    }

    return len;
}

// Output buffer stride per chunk - must handle LZ4 worst case expansion
// LZ4 worst case: every token has 1-byte literal + token + offset = ~1.06x
// Use 2x input size to be safe (never exceeds this)
#define LZ4_OUTPUT_STRIDE(chunk_size) ((chunk_size) * 2)

extern "C" __global__ void lz4_compress_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    uint32_t* __restrict__ output_sizes,
    const uint32_t chunk_size,
    const uint32_t n_chunks
) {
    const int chunk_id = blockIdx.x;
    if (chunk_id >= n_chunks) return;

    const int tid = threadIdx.x;
    // Use 64-bit arithmetic to avoid overflow with large datasets (>4GB)
    const uint8_t* in = input + (uint64_t)chunk_id * chunk_size;
    const uint32_t output_stride = LZ4_OUTPUT_STRIDE(chunk_size);
    uint8_t* out = output + (uint64_t)chunk_id * output_stride;

    // Shared memory hash table (8KB) - much faster than global memory
    // Sentinel 0xFFFF means "no entry" (valid positions are 0..chunk_size-1)
    __shared__ uint16_t hash_table[LZ4_HASH_SIZE];

    // Parallel hash table initialization (all 32 threads)
    for (int i = tid; i < LZ4_HASH_SIZE; i += LZ4_WARP_SIZE) {
        hash_table[i] = 0xFFFF;
    }
    __syncwarp();

    // Only thread 0 does encoding (LZ4 is inherently sequential)
    if (tid != 0) return;

    uint32_t in_pos = 0;
    uint32_t out_pos = 0;
    uint32_t anchor = 0;

    while (in_pos < chunk_size - 12) {
        uint32_t hash = lz4_hash4(in + in_pos);
        uint32_t match_pos = hash_table[hash];
        hash_table[hash] = (uint16_t)in_pos;

        uint32_t mlen = 0;
        if (match_pos != 0xFFFF && in_pos > match_pos && in_pos - match_pos < 65535) {
            if (lz4_load_u32(in + in_pos) == lz4_load_u32(in + match_pos)) {
                mlen = lz4_match_length(in, in_pos, match_pos, chunk_size);
            }
        }

        if (mlen >= 4) {
            uint32_t literal_len = in_pos - anchor;
            uint32_t ml = mlen - 4;

            uint8_t token = ((literal_len >= 15 ? 15 : literal_len) << 4) |
                           (ml >= 15 ? 15 : ml);
            out[out_pos++] = token;

            if (literal_len >= 15) {
                uint32_t rem = literal_len - 15;
                while (rem >= 255) {
                    out[out_pos++] = 255;
                    rem -= 255;
                }
                out[out_pos++] = (uint8_t)rem;
            }

            // Copy literals
            for (uint32_t i = 0; i < literal_len; i++) {
                out[out_pos++] = in[anchor + i];
            }

            // Match offset (little endian, 2 bytes)
            uint16_t moff = in_pos - match_pos;
            out[out_pos++] = moff & 0xFF;
            out[out_pos++] = (moff >> 8) & 0xFF;

            // Extended match length (if >= 15)
            if (ml >= 15) {
                uint32_t rem = ml - 15;
                while (rem >= 255) {
                    out[out_pos++] = 255;
                    rem -= 255;
                }
                out[out_pos++] = (uint8_t)rem;
            }

            in_pos += mlen;
            anchor = in_pos;
        } else {
            in_pos++;
        }
    }

    // Emit remaining literals (LZ4 requires last 5 bytes to be literals)
    uint32_t literal_len = chunk_size - anchor;
    if (literal_len > 0) {
        uint8_t token = (literal_len >= 15 ? 15 : literal_len) << 4;
        out[out_pos++] = token;

        if (literal_len >= 15) {
            uint32_t rem = literal_len - 15;
            while (rem >= 255) {
                out[out_pos++] = 255;
                rem -= 255;
            }
            out[out_pos++] = (uint8_t)rem;
        }

        for (uint32_t i = 0; i < literal_len; i++) {
            out[out_pos++] = in[anchor + i];
        }
    }

    output_sizes[chunk_id] = out_pos;
}

extern "C" __global__ void lz4_compress_var_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    uint32_t* __restrict__ output_sizes,
    const uint32_t frame_bytes,
    const uint32_t block_size,
    const uint32_t max_stride,
    const uint32_t blocks_per_frame,
    const uint32_t n_chunks
) {
    const int chunk_id = blockIdx.x;
    if (chunk_id >= n_chunks) return;

    const int tid = threadIdx.x;
    const uint32_t frame_id = chunk_id / blocks_per_frame;
    const uint32_t block_in_frame = chunk_id - frame_id * blocks_per_frame;
    const uint32_t block_offset = block_in_frame * block_size;
    if (block_offset >= frame_bytes) {
        if (tid == 0) output_sizes[chunk_id] = 0;
        return;
    }
    const uint32_t chunk_size = min(block_size, frame_bytes - block_offset);
    const uint8_t* in = input + (uint64_t)frame_id * frame_bytes + block_offset;
    uint8_t* out = output + (uint64_t)chunk_id * max_stride;

    __shared__ uint16_t hash_table[LZ4_HASH_SIZE];
    for (int i = tid; i < LZ4_HASH_SIZE; i += LZ4_WARP_SIZE) {
        hash_table[i] = 0xFFFF;
    }
    __syncwarp();

    if (tid != 0) return;

    uint32_t in_pos = 0;
    uint32_t out_pos = 0;
    uint32_t anchor = 0;

    while (chunk_size > 12 && in_pos < chunk_size - 12) {
        uint32_t hash = lz4_hash4(in + in_pos);
        uint32_t match_pos = hash_table[hash];
        hash_table[hash] = (uint16_t)in_pos;

        uint32_t mlen = 0;
        if (match_pos != 0xFFFF && in_pos > match_pos && in_pos - match_pos < 65535) {
            if (lz4_load_u32(in + in_pos) == lz4_load_u32(in + match_pos)) {
                mlen = lz4_match_length(in, in_pos, match_pos, chunk_size);
            }
        }

        if (mlen >= 4) {
            uint32_t literal_len = in_pos - anchor;
            uint32_t ml = mlen - 4;

            uint8_t token = ((literal_len >= 15 ? 15 : literal_len) << 4) |
                           (ml >= 15 ? 15 : ml);
            out[out_pos++] = token;

            if (literal_len >= 15) {
                uint32_t rem = literal_len - 15;
                while (rem >= 255) {
                    out[out_pos++] = 255;
                    rem -= 255;
                }
                out[out_pos++] = (uint8_t)rem;
            }

            for (uint32_t i = 0; i < literal_len; i++) {
                out[out_pos++] = in[anchor + i];
            }

            uint16_t moff = in_pos - match_pos;
            out[out_pos++] = moff & 0xFF;
            out[out_pos++] = (moff >> 8) & 0xFF;

            if (ml >= 15) {
                uint32_t rem = ml - 15;
                while (rem >= 255) {
                    out[out_pos++] = 255;
                    rem -= 255;
                }
                out[out_pos++] = (uint8_t)rem;
            }

            in_pos += mlen;
            anchor = in_pos;
        } else {
            in_pos++;
        }
    }

    uint32_t literal_len = chunk_size - anchor;
    if (literal_len > 0) {
        uint8_t token = (literal_len >= 15 ? 15 : literal_len) << 4;
        out[out_pos++] = token;

        if (literal_len >= 15) {
            uint32_t rem = literal_len - 15;
            while (rem >= 255) {
                out[out_pos++] = 255;
                rem -= 255;
            }
            out[out_pos++] = (uint8_t)rem;
        }

        for (uint32_t i = 0; i < literal_len; i++) {
            out[out_pos++] = in[anchor + i];
        }
    }

    output_sizes[chunk_id] = out_pos;
}

// Compact scattered compressed blocks into contiguous buffer
// Each thread handles one block
extern "C" __global__ void compact_compressed(
    const uint8_t* __restrict__ scattered,      // Scattered buffer with max_stride per block
    uint8_t* __restrict__ compact,               // Output compact buffer
    const uint32_t* __restrict__ sizes,          // Size of each compressed block
    const uint64_t* __restrict__ offsets,        // Destination offset for each block (prefix sum)
    const uint32_t max_stride,                   // Max bytes per block in scattered buffer
    const uint32_t n_blocks
) {
    const uint64_t block_id = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (block_id >= n_blocks) return;

    const uint32_t size = sizes[block_id];
    const uint64_t src_offset = block_id * max_stride;
    const uint64_t dst_offset = offsets[block_id];

    // Copy compressed bytes
    for (uint32_t i = 0; i < size; i++) {
        compact[dst_offset + i] = scattered[src_offset + i];
    }
}

// Pack scattered LZ4 blocks directly into HDF5 bitshuffle chunk byte buffers.
// One CUDA block handles one detector frame / HDF5 chunk. This replaces the
// previous CPU-side Numba packing pass during save().
extern "C" __global__ void pack_h5_chunks_kernel(
    const uint8_t* __restrict__ scattered,       // LZ4 blocks, max_stride bytes each
    const uint32_t* __restrict__ sizes,          // LZ4 size for each 8KB block
    const uint64_t* __restrict__ chunk_starts,   // Destination start per frame
    uint8_t* __restrict__ packed,                // Packed HDF5 chunk byte stream
    const uint32_t n_frames,
    const uint32_t n_8kb,
    const uint32_t max_stride,
    const uint64_t frame_bytes,
    const uint32_t block_size
) {
    const uint32_t frame_id = blockIdx.x;
    if (frame_id >= n_frames) return;

    const uint32_t tid = threadIdx.x;
    const uint64_t dst0 = chunk_starts[frame_id];

    // HDF5 bitshuffle header: big-endian uint64 uncompressed bytes,
    // followed by big-endian uint32 block size.
    if (tid < 12) {
        uint8_t value = 0;
        if (tid < 8) {
            value = (frame_bytes >> ((7 - tid) * 8)) & 0xFF;
        } else {
            value = (block_size >> ((11 - tid) * 8)) & 0xFF;
        }
        packed[dst0 + tid] = value;
    }

    uint64_t dst = dst0 + 12;
    const uint64_t base_block = (uint64_t)frame_id * n_8kb;
    for (uint32_t b = 0; b < n_8kb; b++) {
        const uint64_t block_id = base_block + b;
        const uint32_t sz = sizes[block_id];

        if (tid < 4) {
            packed[dst + tid] = (sz >> ((3 - tid) * 8)) & 0xFF;
        }

        const uint8_t* src = scattered + block_id * max_stride;
        uint8_t* out = packed + dst + 4;
        for (uint32_t j = tid; j < sz; j += blockDim.x) {
            out[j] = src[j];
        }
        dst += 4 + sz;
    }
}

extern "C" __global__ void clip_u16_to_u8_count_kernel(
    const uint16_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    const uint64_t n,
    uint64_t* __restrict__ block_counts
) {
    extern __shared__ uint64_t counts[];
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    uint64_t local = 0;

    for (; i < n; i += stride) {
        const uint16_t v = src[i];
        const bool clipped = v > 255;
        local += clipped ? 1 : 0;
        dst[i] = (uint8_t)(clipped ? 255 : v);
    }

    counts[threadIdx.x] = local;
    __syncthreads();

    for (uint32_t offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            counts[threadIdx.x] += counts[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_counts[blockIdx.x] = counts[0];
    }
}

extern "C" __global__ void clip_u16_to_u8_kernel(
    const uint16_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    const uint64_t n
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = (uint64_t)blockDim.x * gridDim.x;

    for (; i < n; i += stride) {
        const uint16_t v = src[i];
        dst[i] = (uint8_t)(v > 255 ? 255 : v);
    }
}

extern "C" __global__ void clip_u32_to_u8_count_kernel(
    const uint32_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    const uint64_t n,
    uint64_t* __restrict__ block_counts
) {
    extern __shared__ uint64_t counts[];
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    uint64_t local = 0;

    for (; i < n; i += stride) {
        const uint32_t v = src[i];
        const bool clipped = v > 255;
        local += clipped ? 1 : 0;
        dst[i] = (uint8_t)(clipped ? 255 : v);
    }

    counts[threadIdx.x] = local;
    __syncthreads();

    for (uint32_t offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            counts[threadIdx.x] += counts[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_counts[blockIdx.x] = counts[0];
    }
}

extern "C" __global__ void clip_u32_to_u8_kernel(
    const uint32_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    const uint64_t n
) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = (uint64_t)blockDim.x * gridDim.x;

    for (; i < n; i += stride) {
        const uint32_t v = src[i];
        dst[i] = (uint8_t)(v > 255 ? 255 : v);
    }
}
'''

# =============================================================================
# COMPILED KERNELS
# =============================================================================

# Kernel names are exposed as module attributes (e.g. ``bitshuffle.h5lz4dc_kernel``)
# but the CUDA module is compiled LAZILY on first access, not at import. This
# keeps `import quantem.widget.io.bitshuffle` working on a non-CUDA box (the
# compile needs cupy + a GPU); only a caller that actually decompresses on the
# cuda backend triggers the compile. The compiled functions are cached after
# the first access, so there is no per-launch overhead — the lookup cost is
# paid once.
_KERNEL_FUNCS = {
    "h5lz4dc_kernel": "h5lz4dc_batched",
    "bitshuffle_kernel": "shuf_8192_32_batched",
    "bitshuffle_kernel_u16": "shuf_8192_16_batched",
    "bitshuffle_kernel_u16_simple": "shuf_8192_16_simple",
    "bitshuffle_tail_kernel_u16": "shuf_tail_16_batched",
    "bitshuffle_tail_kernel_u32": "shuf_tail_32_batched",
    "bitshuffle_fwd_kernel": "bitshuffle_fwd_8192_32",
    "bitshuffle_fwd_kernel_u16": "bitshuffle_fwd_8192_16",
    "bitshuffle_fwd_tail_kernel_u16": "bitshuffle_fwd_tail_16",
    "bitshuffle_fwd_tail_kernel_u32": "bitshuffle_fwd_tail_32",
    "lz4_compress_kernel": "lz4_compress_kernel",
    "lz4_compress_var_kernel": "lz4_compress_var_kernel",
    "compact_kernel": "compact_compressed",
    "pack_h5_chunks_kernel": "pack_h5_chunks_kernel",
    "clip_u16_to_u8_kernel": "clip_u16_to_u8_kernel",
    "clip_u32_to_u8_kernel": "clip_u32_to_u8_kernel",
    "clip_u16_to_u8_count_kernel": "clip_u16_to_u8_count_kernel",
    "clip_u32_to_u8_count_kernel": "clip_u32_to_u8_count_kernel",
}
_cuda_module = None


def _compile_module():
    """Compile the CUDA kernel source once and cache it. Needs cupy + a GPU."""
    global _cuda_module
    if _cuda_module is None:
        if cp is None:
            raise ImportError(
                "cupy is required to compile the bitshuffle+LZ4 CUDA kernels "
                "(the cuda decompress backend). Install cupy on an NVIDIA box, "
                "or use load(backend='cpu' / 'mps')."
            )
        _cuda_module = cp.RawModule(
            code=_CUDA_LZ4_SOURCE, options=("-std=c++11", "-w")
        )
    return _cuda_module


def __getattr__(name: str):
    """PEP 562 lazy attribute: compile + return a kernel the first time its
    module-level name is read, then cache it in globals so later reads are a
    plain dict hit."""
    fn = _KERNEL_FUNCS.get(name)
    if fn is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    kernel = _compile_module().get_function(fn)
    globals()[name] = kernel
    return kernel
