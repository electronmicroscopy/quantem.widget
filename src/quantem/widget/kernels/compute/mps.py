"""MPS compute kernels: BF/DF/ADF reductions over chunked unified-memory buffers.

Extracted from widgets/show4dstem_mps.py (the macbook viewer) into kernels/compute
per docs/dev-notes/2026-06-01-kernels-backend-architecture.md. Holds the raw-Metal
masked-sum / detector-sum / prefix reductions + the ChunkedFrames container. The
widget imports MetalVirtualImage + ChunkedFrames from here; this module has NO
widget/UI dependency.
"""
from __future__ import annotations

import gc
import os
import time
import threading
import numpy as np
from numba import njit, prange


# Raw Metal masked-sum kernel: one thread per scan position, reads uint16 detector
# in place, accumulates int, skips non-masked pixels (so a BF disk reads ~8% of
# the detector and is faster than a full read). No dtype cast.
import pathlib as _pathlib
_MASKED_SUM_MSL = (_pathlib.Path(__file__).parent / 'metal' / 'reductions.msl').read_text()

# Chunk buffers bound per command buffer. Binding all 27 (19.3 GB) in one command
# buffer errors with status=5 (working-set limit). Measured sweet spot is 13:
# CPB=8 -> 3 fps, CPB=10/13 -> 8 fps, CPB=27 -> status=5 error. Fewer, fuller
# command buffers mean fewer waitUntilCompleted stalls (wait-last, not wait-each,
# cut BF from 475 ms to 194 ms). 13 keeps each command buffer's working set under
# the limit while halving the wait count vs 8.
_CHUNKS_PER_CMDBUF = 13
_COMMAND_BUFFER_BYTES = 9_500_000_000
_DEFAULT_COMPACT_TARGET_BYTES = 3_600_000_000
_PREFIX_NUMBA_MAX_SPANS = 128
_RADIAL_INTERACTION_IDLE_DELAY = 0.75


@njit(parallel=True, nogil=True, cache=True)
def _row_prefix_sum_chunk_numba(
    chunk: np.ndarray,
    left_rows: np.ndarray,
    left_cols: np.ndarray,
    right_rows: np.ndarray,
    right_cols: np.ndarray,
    out: np.ndarray,
):
    """Exact row-prefix span sum over one chunk.

    The input is a numpy view of a shared Metal buffer. On a memory-pressured
    MacBook, this CPU-side parallel path is steadier than the Metal gather for
    compact BF/DF masks while still reading the same zero-copy no-bin data.
    """
    nframes = chunk.shape[0]
    nspans = right_rows.shape[0]
    for frame in prange(nframes):
        total = 0
        for sp in range(nspans):
            rr = right_rows[sp]
            rc = right_cols[sp]
            val = int(chunk[frame, rr, rc])
            lr = left_rows[sp]
            if lr >= 0:
                lc = left_cols[sp]
                val -= int(chunk[frame, lr, lc])
            total += val
        out[frame] = total


def _chunk_nbytes(chunk) -> int:
    return int(getattr(chunk, "nbytes", np.asarray(chunk).nbytes))


def _chunk_groups(chunks: list):
    """Yield chunk index groups that stay below the Metal working-set limit."""
    start = 0
    total = 0
    count = 0
    for ci, chunk in enumerate(chunks):
        nbytes = _chunk_nbytes(chunk)
        if count and (
            count >= _CHUNKS_PER_CMDBUF
            or total + nbytes > _COMMAND_BUFFER_BYTES
        ):
            yield range(start, ci)
            start = ci
            total = 0
            count = 0
        total += nbytes
        count += 1
    if count:
        yield range(start, len(chunks))


def _format_seconds(seconds: float) -> str:
    if seconds < 10:
        return f"{seconds:.2f}s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem = int(round(seconds - minutes * 60))
    return f"{minutes}m {rem:02d}s"


def _bin_mask(mask: np.ndarray, binf: int = 2) -> np.ndarray:
    """Downsample a full detector mask to the binf sidecar grid.

    A binned detector pixel is "in the mask" if ANY of its binf*binf raw pixels
    are, so a virtual-detector edge never disappears at coarser bin factors.
    """
    mask = np.asarray(mask, dtype=bool)
    binf = int(binf)
    rows = mask.shape[-2] // binf
    cols = mask.shape[-1] // binf
    return (mask[: rows * binf, : cols * binf]
            .reshape(rows, binf, cols, binf).any(axis=(1, 3)))


def _upsample_bin_dp(dp: np.ndarray, out_shape: tuple[int, int],
                     binf: int = 2) -> np.ndarray:
    arr = np.asarray(dp, dtype=np.float32)
    binf = int(binf)
    return np.repeat(np.repeat(arr, binf, axis=0), binf, axis=1)[
        : out_shape[0], : out_shape[1]
    ]


def _bin2_mask(mask: np.ndarray) -> np.ndarray:
    """Back-compat: bin2 mask downsample (see :func:`_bin_mask`)."""
    return _bin_mask(mask, 2)


def _upsample_bin2_dp(dp: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """Back-compat: bin2 DP upsample (see :func:`_upsample_bin_dp`)."""
    return _upsample_bin_dp(dp, out_shape, 2)


def default_fast_bin() -> int:
    """Scrub-sidecar bin factor that fits the host's unified/GPU memory.

    The bin2 sidecar of a no-bin 512x512x192x192 stack is 4.8 GB on top of the
    19.3 GB data = 24.1 GB, which does not fit a 24 GB Mac (it swaps and the
    machine freezes). bin4 is 1.2 GB -> ~20.5 GB total, fits. So: bin4 on Macs
    with <= ~32 GB unified memory, bin2 on larger boxes where the sharper
    detector grid is free. Falls back to bin4 (the safe choice) if the memory
    size can't be read.
    """
    try:
        import subprocess
        total = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                   capture_output=True, text=True,
                                   timeout=3).stdout.strip())
        return 2 if total > 32 * 1024**3 else 4
    except Exception:
        return 4


class MetalVirtualImage:
    """Raw-Metal BF/DF/ADF over a list of unified-memory Metal-buffer chunks.

    Each chunk is a ``_MtlArray`` (numpy view over a Metal buffer) from
    ``MPSDecompressor.load_master_chunked``. The masked-sum kernel reads the
    underlying ``_mtl`` buffer directly — no torch, no copy, no cast.
    """

    def __init__(self, chunks: list, *, row_prefix: bool = False):
        import Metal
        from quantem.widget.kernels.io import mps as _mps

        self._mps = _mps
        self._Metal = Metal
        self.chunks = chunks
        self.det = tuple(int(x) for x in chunks[0].shape[1:])
        self.ndet = self.det[0] * self.det[1]
        self.n = int(sum(int(c.shape[0]) for c in chunks))
        self._offsets = [0]
        for c in chunks:
            self._offsets.append(self._offsets[-1] + int(c.shape[0]))
        dev = _mps._device
        lib, err = dev.newLibraryWithSource_options_error_(
            _MASKED_SUM_MSL, _mps._options, None)
        if err:
            raise RuntimeError(f"masked_sum kernel compile failed: {err}")
        self._pipe, _ = dev.newComputePipelineStateWithFunction_error_(
            lib.newFunctionWithName_("masked_sum_u16"), None)
        self._detsum_pipe, _ = dev.newComputePipelineStateWithFunction_error_(
            lib.newFunctionWithName_("detector_sum_u16"), None)
        self._detsum_prefix_pipe, _ = dev.newComputePipelineStateWithFunction_error_(
            lib.newFunctionWithName_("detector_sum_prefix_u16"), None)
        self._row_overflow_pipe, _ = dev.newComputePipelineStateWithFunction_error_(
            lib.newFunctionWithName_("row_sum_overflow_u16"), None)
        self._row_prefix_pipe, _ = dev.newComputePipelineStateWithFunction_error_(
            lib.newFunctionWithName_("row_prefix_u16_inplace"), None)
        self._bin_pipe, _ = dev.newComputePipelineStateWithFunction_error_(
            lib.newFunctionWithName_("bin_detector_u16"), None)
        self._mean_dp_pipe, _ = dev.newComputePipelineStateWithFunction_error_(
            lib.newFunctionWithName_("mean_dp_sum_u16"), None)
        self._mean_dp_prefix_pipe, _ = dev.newComputePipelineStateWithFunction_error_(
            lib.newFunctionWithName_("mean_dp_sum_prefix_u16"), None)
        self._span_raw_pipe, _ = dev.newComputePipelineStateWithFunction_error_(
            lib.newFunctionWithName_("rowspan_sum_u16"), None)
        self._span_pipe, _ = dev.newComputePipelineStateWithFunction_error_(
            lib.newFunctionWithName_("rowspan_sum_prefix_u16"), None)
        self._span_tg_pipe, _ = dev.newComputePipelineStateWithFunction_error_(
            lib.newFunctionWithName_("rowspan_sum_prefix_tg_u16"), None)
        self._span_tile_pipe, _ = dev.newComputePipelineStateWithFunction_error_(
            lib.newFunctionWithName_("rowspan_sum_prefix_tile_u16"), None)
        self._span_simd8_pipe, _ = dev.newComputePipelineStateWithFunction_error_(
            lib.newFunctionWithName_("rowspan_sum_prefix_simd8_u16"), None)
        self._radial_pipe, _ = dev.newComputePipelineStateWithFunction_error_(
            lib.newFunctionWithName_("radial_cumsum_u16"), None)
        self._radial_prefix_pipe, _ = dev.newComputePipelineStateWithFunction_error_(
            lib.newFunctionWithName_("radial_cumsum_prefix_u16"), None)
        self._radial_dual_prefix_pipe, _ = dev.newComputePipelineStateWithFunction_error_(
            lib.newFunctionWithName_("radial_cumsum_dual_prefix_u16"), None)
        self._radial_dual_prefix_tg_pipe, _ = dev.newComputePipelineStateWithFunction_error_(
            lib.newFunctionWithName_("radial_cumsum_dual_prefix_tg_u16"), None)
        self._com_pipe, _ = dev.newComputePipelineStateWithFunction_error_(
            lib.newFunctionWithName_("com_u16"), None)
        # one int32 output buffer per chunk (reused every recompute)
        self._out_mtls = [_mps._metal_buffer_alloc(int(c.shape[0]) * 4)
                          for c in chunks]
        self._out_nps = [_mps._numpy_view(self._out_mtls[i], np.int32,
                                          int(c.shape[0]))
                         for i, c in enumerate(chunks)]
        # CoM output: float2 (CoMx, CoMy) per frame per chunk (8 bytes/frame)
        self._com_out_mtls = [_mps._metal_buffer_alloc(int(c.shape[0]) * 8)
                              for c in chunks]
        self._com_out_nps = [_mps._numpy_view(self._com_out_mtls[i], np.float32,
                                              int(c.shape[0]) * 2)
                             for i, c in enumerate(chunks)]
        # constant buffers: ndet (shared) + per-chunk nframes
        self._ndet_mtl = _mps._metal_buffer_alloc(4)
        _mps._numpy_view(self._ndet_mtl, np.uint32, 1)[0] = self.ndet
        self._detrows_mtl = _mps._metal_buffer_alloc(4)
        _mps._numpy_view(self._detrows_mtl, np.uint32, 1)[0] = self.det[0]
        self._detcols_mtl = _mps._metal_buffer_alloc(4)
        _mps._numpy_view(self._detcols_mtl, np.uint32, 1)[0] = self.det[1]
        self._nf_mtls = []
        for c in chunks:
            b = _mps._metal_buffer_alloc(4)
            _mps._numpy_view(b, np.uint32, 1)[0] = int(c.shape[0])
            self._nf_mtls.append(b)
        # mask buffer (one detector frame, written per recompute)
        self._mask_mtl = _mps._metal_buffer_alloc(self.ndet)
        self._mask_np = _mps._numpy_view(self._mask_mtl, np.uint8, self.ndet)
        self._full = np.empty(self.n, dtype=np.int32)
        self._row_prefix = bool(row_prefix)
        self._row_prefix_warmed = False
        self._row_prefix_numba_warmed = False
        self._overflow_mtl = _mps._metal_buffer_alloc(4)
        self._overflow_np = _mps._numpy_view(self._overflow_mtl, np.uint32, 1)
        self._nspans_mtl = _mps._metal_buffer_alloc(4)
        self._nspans_np = _mps._numpy_view(self._nspans_mtl, np.uint32, 1)
        self._span_capacity = 0
        self._span_starts_mtl = None
        self._span_ends_mtl = None
        self._span_starts_np = None
        self._span_ends_np = None
        self._roi_idx_capacity = [0 for _ in chunks]
        self._roi_idx_mtls = [None for _ in chunks]
        self._roi_idx_nps = [None for _ in chunks]
        self._roi_nidx_mtls = []
        self._roi_nidx_nps = []
        self._roi_sum_mtls = []
        self._roi_sum_nps = []
        for _ in chunks:
            nidx_mtl = _mps._metal_buffer_alloc(4)
            nidx_np = _mps._numpy_view(nidx_mtl, np.uint32, 1)
            self._roi_nidx_mtls.append(nidx_mtl)
            self._roi_nidx_nps.append(nidx_np)
            sum_mtl = _mps._metal_buffer_alloc(self.ndet * 4)
            self._roi_sum_mtls.append(sum_mtl)
            self._roi_sum_nps.append(
                _mps._numpy_view(sum_mtl, np.uint32, self.ndet)
            )
        self._roi_accum = np.empty(self.ndet, dtype=np.uint64)
        self._roi_mean = np.empty(self.ndet, dtype=np.float32)
        self._radial_center = None
        self._radial_nbins = 0
        self._radial_radbin_mtl = None
        self._radial_radbin_np = None
        self._radial_floor_radbin_mtl = None
        self._radial_floor_radbin_np = None
        self._radial_out_mtl = None
        self._radial_out_np = None
        self._radial_floor_out_mtl = None
        self._radial_floor_out_np = None

    @property
    def row_prefix_enabled(self) -> bool:
        return self._row_prefix

    def bin2_chunks(self, *, verbose: bool = True) -> list:
        """Back-compat: detector-bin2 sidecar (see :meth:`bin_chunks`)."""
        return self.bin_chunks(2, verbose=verbose)

    def bin_chunks(self, binf: int = 2, *, verbose: bool = True) -> list:
        """Build a detector-binf uint16 sidecar for fast live interaction.

        binf*binf raw pixels sum into one sidecar pixel, in place over the
        resident no-bin chunks (no disk re-decode, no decompress scratch). bin4
        keeps the sidecar at 1.2 GB so the whole no-bin viewer fits a 24 GB Mac
        (~20.5 GB); bin2 (4.8 GB) is for boxes with memory to spare.
        """
        binf = int(binf)
        if self.det[0] % binf or self.det[1] % binf:
            raise ValueError(
                f"bin{binf} fast interaction requires detector dims divisible by {binf}")
        if verbose:
            print(f"Building detector-bin{binf} virtual-image cache")
        t0 = time.perf_counter()
        out_shape = (self.det[0] // binf, self.det[1] // binf)
        outndet = out_shape[0] * out_shape[1]
        out_chunks = []
        from quantem.widget.kernels.io import mps as _mps

        for chunk in self.chunks:
            nf = int(chunk.shape[0])
            out_mtl = _mps._metal_buffer_alloc(nf * outndet * 2)
            out_np = _mps._numpy_view(out_mtl, np.uint16, nf * outndet)
            arr = out_np.reshape((nf, *out_shape)).view(_mps._MtlArray)
            arr._mtl = out_mtl
            out_chunks.append(arr)

        outcols_mtl = _mps._metal_buffer_alloc(4)
        _mps._numpy_view(outcols_mtl, np.uint32, 1)[0] = out_shape[1]
        outndet_mtl = _mps._metal_buffer_alloc(4)
        _mps._numpy_view(outndet_mtl, np.uint32, 1)[0] = outndet
        binf_mtl = _mps._metal_buffer_alloc(4)
        _mps._numpy_view(binf_mtl, np.uint32, 1)[0] = binf

        Metal = self._Metal
        cmds = []
        for group in _chunk_groups(self.chunks):
            cmd = self._mps._queue.commandBuffer()
            enc = cmd.computeCommandEncoder()
            enc.setComputePipelineState_(self._bin_pipe)
            for ci in group:
                nf = int(self.chunks[ci].shape[0])
                total = nf * outndet
                enc.setBuffer_offset_atIndex_(self.chunks[ci]._mtl, 0, 0)
                enc.setBuffer_offset_atIndex_(out_chunks[ci]._mtl, 0, 1)
                enc.setBuffer_offset_atIndex_(self._detcols_mtl, 0, 2)
                enc.setBuffer_offset_atIndex_(outcols_mtl, 0, 3)
                enc.setBuffer_offset_atIndex_(outndet_mtl, 0, 4)
                enc.setBuffer_offset_atIndex_(self._nf_mtls[ci], 0, 5)
                enc.setBuffer_offset_atIndex_(binf_mtl, 0, 6)
                enc.dispatchThreadgroups_threadsPerThreadgroup_(
                    Metal.MTLSizeMake((total + 255) // 256, 1, 1),
                    Metal.MTLSizeMake(256, 1, 1))
            enc.endEncoding()
            cmd.commit()
            cmds.append(cmd)
        cmds[-1].waitUntilCompleted()
        if verbose:
            print(
                f"Detector-bin{binf} virtual-image cache ready in "
                f"{_format_seconds(time.perf_counter() - t0)}"
            )
        return out_chunks

    def masked_sum(self, mask2d: np.ndarray) -> np.ndarray:
        """Per-scan-position detector-masked sum -> (N,) int32, via raw Metal.

        Dispatches all chunks across a few command buffers (<= 8 chunks each, to
        stay under the working-set limit), commits them, waits on the last, then
        gathers the per-chunk outputs. uint16 in place, no cast.
        """
        if self._row_prefix:
            return self._masked_sum_prefix(mask2d)
        return self._masked_sum_raw(mask2d)

    def _masked_sum_raw(self, mask2d: np.ndarray) -> np.ndarray:
        starts, ends = self._mask_spans(mask2d)
        if 0 < int(starts.size) <= 512:
            return self._masked_sum_raw_spans(starts, ends)
        if int(starts.size) == 0:
            self._full.fill(0)
            return self._full
        Metal = self._Metal
        self._mask_np[:] = np.asarray(mask2d, dtype=bool).reshape(-1).astype(np.uint8)
        cmds = []
        chunks, pipe = self.chunks, self._pipe
        for group in _chunk_groups(chunks):
            cmd = self._mps._queue.commandBuffer()
            enc = cmd.computeCommandEncoder()
            enc.setComputePipelineState_(pipe)
            for ci in group:
                nf = int(chunks[ci].shape[0])
                enc.setBuffer_offset_atIndex_(chunks[ci]._mtl, 0, 0)
                enc.setBuffer_offset_atIndex_(self._mask_mtl, 0, 1)
                enc.setBuffer_offset_atIndex_(self._out_mtls[ci], 0, 2)
                enc.setBuffer_offset_atIndex_(self._ndet_mtl, 0, 3)
                enc.setBuffer_offset_atIndex_(self._nf_mtls[ci], 0, 4)
                enc.dispatchThreadgroups_threadsPerThreadgroup_(
                    Metal.MTLSizeMake((nf + 255) // 256, 1, 1),
                    Metal.MTLSizeMake(256, 1, 1))
            enc.endEncoding()
            cmd.commit()
            cmds.append(cmd)
        cmds[-1].waitUntilCompleted()
        for ci in range(len(chunks)):
            self._full[self._offsets[ci]:self._offsets[ci + 1]] = self._out_nps[ci]
        return self._full

    def center_of_mass(self, mask2d: np.ndarray | None = None):
        """Per-scan-position center of mass over the masked detector, raw Metal.

        Returns ``(com_col, com_row)`` each ``(N,)`` float32 in ABSOLUTE detector
        coordinates (col = Sum col*I / Sum I, row = Sum row*I / Sum I) - the DPC
        vector field before mean-subtraction / rotation. ``mask2d`` None means the
        full detector. Reads uint16 chunks in place with int64 accumulators in the
        kernel - no float32 copy of the data, so it fits no-bin in 24 GB. One
        streaming pass; each frame's CoM is fully within its chunk (no cross-chunk
        accumulation). Matches engine.dpc.compute_center_of_mass.
        """
        Metal = self._Metal
        if mask2d is None:
            self._mask_np[:] = 1
        else:
            self._mask_np[:] = np.asarray(mask2d, dtype=bool).reshape(-1).astype(np.uint8)
        chunks, pipe = self.chunks, self._com_pipe
        cmds = []
        for group in _chunk_groups(chunks):
            cmd = self._mps._queue.commandBuffer()
            enc = cmd.computeCommandEncoder()
            enc.setComputePipelineState_(pipe)
            for ci in group:
                nf = int(chunks[ci].shape[0])
                enc.setBuffer_offset_atIndex_(chunks[ci]._mtl, 0, 0)
                enc.setBuffer_offset_atIndex_(self._mask_mtl, 0, 1)
                enc.setBuffer_offset_atIndex_(self._com_out_mtls[ci], 0, 2)
                enc.setBuffer_offset_atIndex_(self._ndet_mtl, 0, 3)
                enc.setBuffer_offset_atIndex_(self._detcols_mtl, 0, 4)
                enc.setBuffer_offset_atIndex_(self._nf_mtls[ci], 0, 5)
                enc.dispatchThreadgroups_threadsPerThreadgroup_(
                    Metal.MTLSizeMake((nf + 255) // 256, 1, 1),
                    Metal.MTLSizeMake(256, 1, 1))
            enc.endEncoding()
            cmd.commit()
            cmds.append(cmd)
        cmds[-1].waitUntilCompleted()
        com = np.empty((self.n, 2), dtype=np.float32)
        for ci in range(len(chunks)):
            com[self._offsets[ci]:self._offsets[ci + 1]] = self._com_out_nps[ci].reshape(-1, 2)
        return com[:, 0].copy(), com[:, 1].copy()  # com_col, com_row

    def _masked_sum_raw_spans(
        self,
        starts: np.ndarray,
        ends: np.ndarray,
    ) -> np.ndarray:
        Metal = self._Metal
        nspans = int(starts.size)
        self._ensure_span_buffers(nspans)
        self._span_starts_np[:nspans] = starts
        self._span_ends_np[:nspans] = ends
        self._nspans_np[0] = nspans
        cmds = []
        chunks = self.chunks
        for group in _chunk_groups(chunks):
            cmd = self._mps._queue.commandBuffer()
            enc = cmd.computeCommandEncoder()
            enc.setComputePipelineState_(self._span_raw_pipe)
            for ci in group:
                nf = int(chunks[ci].shape[0])
                enc.setBuffer_offset_atIndex_(chunks[ci]._mtl, 0, 0)
                enc.setBuffer_offset_atIndex_(self._span_starts_mtl, 0, 1)
                enc.setBuffer_offset_atIndex_(self._span_ends_mtl, 0, 2)
                enc.setBuffer_offset_atIndex_(self._out_mtls[ci], 0, 3)
                enc.setBuffer_offset_atIndex_(self._nspans_mtl, 0, 4)
                enc.setBuffer_offset_atIndex_(self._ndet_mtl, 0, 5)
                enc.setBuffer_offset_atIndex_(self._nf_mtls[ci], 0, 6)
                enc.dispatchThreadgroups_threadsPerThreadgroup_(
                    Metal.MTLSizeMake((nf + 255) // 256, 1, 1),
                    Metal.MTLSizeMake(256, 1, 1))
            enc.endEncoding()
            cmd.commit()
            cmds.append(cmd)
        cmds[-1].waitUntilCompleted()
        for ci in range(len(chunks)):
            self._full[self._offsets[ci]:self._offsets[ci + 1]] = self._out_nps[ci]
        return self._full

    def enable_row_prefix(self, *, verbose: bool = True) -> bool:
        """Transform raw chunks in place to exact uint16 row-prefix chunks.

        This keeps no-bin 192x192 information and memory footprint, but virtual
        detector sums over compact masks become span sums: two reads per
        contiguous detector-row span instead of thousands of scattered reads.
        """
        if self._row_prefix:
            return True
        Metal = self._Metal
        if verbose:
            print("Preparing exact row-prefix interaction layout")
        t0 = time.perf_counter()
        self._overflow_np[0] = 0
        cmds = []
        for group in _chunk_groups(self.chunks):
            cmd = self._mps._queue.commandBuffer()
            enc = cmd.computeCommandEncoder()
            enc.setComputePipelineState_(self._row_overflow_pipe)
            for ci in group:
                nf = int(self.chunks[ci].shape[0])
                total_rows = nf * self.det[0]
                enc.setBuffer_offset_atIndex_(self.chunks[ci]._mtl, 0, 0)
                enc.setBuffer_offset_atIndex_(self._overflow_mtl, 0, 1)
                enc.setBuffer_offset_atIndex_(self._detcols_mtl, 0, 2)
                enc.setBuffer_offset_atIndex_(self._detrows_mtl, 0, 3)
                enc.setBuffer_offset_atIndex_(self._nf_mtls[ci], 0, 4)
                enc.dispatchThreadgroups_threadsPerThreadgroup_(
                    Metal.MTLSizeMake((total_rows + 255) // 256, 1, 1),
                    Metal.MTLSizeMake(256, 1, 1))
            enc.endEncoding()
            cmd.commit()
            cmds.append(cmd)
        cmds[-1].waitUntilCompleted()
        if int(self._overflow_np[0]) != 0:
            raise RuntimeError(
                "Cannot enable no-bin fast interaction: at least one detector "
                "row sum exceeds uint16 prefix capacity. Raw exact mode is unchanged."
            )

        cmds = []
        for group in _chunk_groups(self.chunks):
            cmd = self._mps._queue.commandBuffer()
            enc = cmd.computeCommandEncoder()
            enc.setComputePipelineState_(self._row_prefix_pipe)
            for ci in group:
                nf = int(self.chunks[ci].shape[0])
                total_rows = nf * self.det[0]
                enc.setBuffer_offset_atIndex_(self.chunks[ci]._mtl, 0, 0)
                enc.setBuffer_offset_atIndex_(self._detcols_mtl, 0, 1)
                enc.setBuffer_offset_atIndex_(self._detrows_mtl, 0, 2)
                enc.setBuffer_offset_atIndex_(self._nf_mtls[ci], 0, 3)
                enc.dispatchThreadgroups_threadsPerThreadgroup_(
                    Metal.MTLSizeMake((total_rows + 255) // 256, 1, 1),
                    Metal.MTLSizeMake(256, 1, 1))
            enc.endEncoding()
            cmd.commit()
            cmds.append(cmd)
        cmds[-1].waitUntilCompleted()
        self._row_prefix = True
        self._row_prefix_warmed = False
        self._warm_row_prefix_numba()
        if verbose:
            print(
                "Prepared exact row-prefix interaction layout in "
                f"{_format_seconds(time.perf_counter() - t0)}"
            )
        return True

    def _ensure_span_buffers(self, nspans: int):
        if nspans <= self._span_capacity:
            return
        self._span_capacity = max(64, int(nspans))
        nbytes = self._span_capacity * 4
        self._span_starts_mtl = self._mps._metal_buffer_alloc(nbytes)
        self._span_ends_mtl = self._mps._metal_buffer_alloc(nbytes)
        self._span_starts_np = self._mps._numpy_view(
            self._span_starts_mtl, np.uint32, self._span_capacity)
        self._span_ends_np = self._mps._numpy_view(
            self._span_ends_mtl, np.uint32, self._span_capacity)

    def _ensure_roi_index_buffer(self, ci: int, nindices: int):
        if nindices <= self._roi_idx_capacity[ci]:
            return
        capacity = max(256, int(nindices), self._roi_idx_capacity[ci] * 2)
        self._roi_idx_capacity[ci] = capacity
        self._roi_idx_mtls[ci] = self._mps._metal_buffer_alloc(capacity * 4)
        self._roi_idx_nps[ci] = self._mps._numpy_view(
            self._roi_idx_mtls[ci], np.uint32, capacity)

    def mean_frames(self, frame_indices: np.ndarray) -> np.ndarray:
        """Average selected scan positions into one diffraction pattern."""
        indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            self._roi_mean.fill(0)
            return self._roi_mean.reshape(self.det)
        if int(indices.min()) < 0 or int(indices.max()) >= self.n:
            raise IndexError("frame index out of bounds")
        indices = np.sort(indices.astype(np.uint32, copy=False))

        active = []
        for ci in range(len(self.chunks)):
            start = self._offsets[ci]
            stop = self._offsets[ci + 1]
            lo = int(np.searchsorted(indices, start, side="left"))
            hi = int(np.searchsorted(indices, stop, side="left"))
            nlocal = hi - lo
            if nlocal == 0:
                continue
            self._ensure_roi_index_buffer(ci, nlocal)
            self._roi_idx_nps[ci][:nlocal] = indices[lo:hi] - start
            self._roi_nidx_nps[ci][0] = nlocal
            active.append(ci)

        if not active:
            self._roi_mean.fill(0)
            return self._roi_mean.reshape(self.det)

        Metal = self._Metal
        active_set = set(active)
        pipe = self._mean_dp_prefix_pipe if self._row_prefix else self._mean_dp_pipe
        cmds = []
        for group in _chunk_groups(self.chunks):
            selected = [ci for ci in group if ci in active_set]
            if not selected:
                continue
            cmd = self._mps._queue.commandBuffer()
            enc = cmd.computeCommandEncoder()
            enc.setComputePipelineState_(pipe)
            for ci in selected:
                enc.setBuffer_offset_atIndex_(self.chunks[ci]._mtl, 0, 0)
                enc.setBuffer_offset_atIndex_(self._roi_idx_mtls[ci], 0, 1)
                enc.setBuffer_offset_atIndex_(self._roi_sum_mtls[ci], 0, 2)
                enc.setBuffer_offset_atIndex_(self._ndet_mtl, 0, 3)
                enc.setBuffer_offset_atIndex_(self._roi_nidx_mtls[ci], 0, 4)
                if self._row_prefix:
                    enc.setBuffer_offset_atIndex_(self._detcols_mtl, 0, 5)
                enc.dispatchThreadgroups_threadsPerThreadgroup_(
                    Metal.MTLSizeMake((self.ndet + 255) // 256, 1, 1),
                    Metal.MTLSizeMake(256, 1, 1))
            enc.endEncoding()
            cmd.commit()
            cmds.append(cmd)
        cmds[-1].waitUntilCompleted()

        self._roi_accum.fill(0)
        for ci in active:
            self._roi_accum += self._roi_sum_nps[ci]
        np.divide(
            self._roi_accum,
            float(indices.size),
            out=self._roi_mean,
            casting="unsafe",
        )
        return self._roi_mean.reshape(self.det)

    def _mask_spans(self, mask2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = np.asarray(mask2d, dtype=bool).reshape(self.det)
        starts = []
        ends = []
        ncols = self.det[1]
        for row in range(mask.shape[0]):
            cols = np.flatnonzero(mask[row])
            if cols.size == 0:
                continue
            breaks = np.flatnonzero(cols[1:] != cols[:-1] + 1) + 1
            for seg in np.split(cols, breaks):
                starts.append(row * ncols + int(seg[0]))
                ends.append(row * ncols + int(seg[-1]))
        return np.asarray(starts, dtype=np.uint32), np.asarray(ends, dtype=np.uint32)

    def _warm_row_prefix_numba(self):
        if self._row_prefix_numba_warmed:
            return
        scratch = np.empty(1, dtype=np.int32)
        _row_prefix_sum_chunk_numba(
            np.asarray(self.chunks[0][:1]),
            np.asarray([-1], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            scratch,
        )
        self._row_prefix_numba_warmed = True

    def _masked_sum_prefix_numba(
        self,
        starts: np.ndarray,
        ends: np.ndarray,
    ) -> np.ndarray:
        detcols = int(self.det[1])
        right_flat = ends.astype(np.int64, copy=False)
        right_rows = right_flat // detcols
        right_cols = right_flat - right_rows * detcols
        left_flat = starts.astype(np.int64, copy=True)
        row_start = (starts % self.det[1]) == 0
        left_flat[~row_start] -= 1
        left_rows = np.empty_like(left_flat)
        left_cols = np.zeros_like(left_flat)
        left_rows[row_start] = -1
        left_rows[~row_start] = left_flat[~row_start] // detcols
        left_cols[~row_start] = (
            left_flat[~row_start] - left_rows[~row_start] * detcols
        )
        self._warm_row_prefix_numba()
        for ci, chunk in enumerate(self.chunks):
            _row_prefix_sum_chunk_numba(
                np.asarray(chunk),
                left_rows,
                left_cols,
                right_rows,
                right_cols,
                self._out_nps[ci],
            )
        for ci in range(len(self.chunks)):
            self._full[self._offsets[ci]:self._offsets[ci + 1]] = self._out_nps[ci]
        return self._full

    def _masked_sum_prefix(self, mask2d: np.ndarray) -> np.ndarray:
        Metal = self._Metal
        starts, ends = self._mask_spans(mask2d)
        nspans = int(starts.size)
        if nspans == 0:
            self._full.fill(0)
            return self._full
        if nspans <= _PREFIX_NUMBA_MAX_SPANS:
            return self._masked_sum_prefix_numba(starts, ends)
        self._ensure_span_buffers(nspans)
        lefts = starts.astype(np.uint32, copy=True)
        row_start = (starts % self.det[1]) == 0
        lefts[~row_start] -= np.uint32(1)
        lefts[row_start] = np.uint32(0xFFFFFFFF)
        self._span_starts_np[:nspans] = lefts
        self._span_ends_np[:nspans] = ends
        self._nspans_np[0] = nspans
        cmds = []
        chunks = self.chunks
        mode = os.environ.get("QT_MPS_PREFIX_MODE", "auto").lower()
        use_simd8 = 16 <= nspans <= 512
        use_tile = False
        use_tg = nspans > 128
        if mode == "simple":
            use_simd8 = use_tile = use_tg = False
        elif mode == "tg":
            use_simd8 = use_tile = False
            use_tg = True
        elif mode == "tile":
            use_simd8 = use_tg = False
            use_tile = True
        elif mode == "simd8":
            use_simd8 = True
            use_tile = use_tg = False
        pipe = (
            self._span_simd8_pipe if use_simd8
            else self._span_tile_pipe if use_tile
            else self._span_tg_pipe if use_tg
            else self._span_pipe
        )
        for group in _chunk_groups(chunks):
            cmd = self._mps._queue.commandBuffer()
            enc = cmd.computeCommandEncoder()
            enc.setComputePipelineState_(pipe)
            for ci in group:
                nf = int(chunks[ci].shape[0])
                enc.setBuffer_offset_atIndex_(chunks[ci]._mtl, 0, 0)
                enc.setBuffer_offset_atIndex_(self._span_starts_mtl, 0, 1)
                enc.setBuffer_offset_atIndex_(self._span_ends_mtl, 0, 2)
                enc.setBuffer_offset_atIndex_(self._out_mtls[ci], 0, 3)
                enc.setBuffer_offset_atIndex_(self._nspans_mtl, 0, 4)
                enc.setBuffer_offset_atIndex_(self._ndet_mtl, 0, 5)
                enc.setBuffer_offset_atIndex_(self._detcols_mtl, 0, 6)
                enc.setBuffer_offset_atIndex_(self._nf_mtls[ci], 0, 7)
                if use_simd8:
                    enc.dispatchThreadgroups_threadsPerThreadgroup_(
                        Metal.MTLSizeMake((nf + 7) // 8, 1, 1),
                        Metal.MTLSizeMake(256, 1, 1))
                elif use_tile:
                    enc.dispatchThreadgroups_threadsPerThreadgroup_(
                        Metal.MTLSizeMake((nf + 7) // 8, 1, 1),
                        Metal.MTLSizeMake(256, 1, 1))
                elif use_tg:
                    enc.dispatchThreadgroups_threadsPerThreadgroup_(
                        Metal.MTLSizeMake(nf, 1, 1),
                        Metal.MTLSizeMake(128, 1, 1))
                else:
                    enc.dispatchThreadgroups_threadsPerThreadgroup_(
                        Metal.MTLSizeMake((nf + 255) // 256, 1, 1),
                        Metal.MTLSizeMake(256, 1, 1))
            enc.endEncoding()
            cmd.commit()
            cmds.append(cmd)
        cmds[-1].waitUntilCompleted()
        for ci in range(len(chunks)):
            self._full[self._offsets[ci]:self._offsets[ci + 1]] = self._out_nps[ci]
        if not self._row_prefix_warmed and nspans <= 512:
            self._row_prefix_warmed = True
            for _ in range(3):
                self._masked_sum_prefix(mask2d)
        return self._full

    def _ensure_radial_cache(self, center_row: float, center_col: float):
        center = (round(float(center_row), 3), round(float(center_col), 3))
        if self._radial_center == center and self._radial_out_np is not None:
            return
        rows = np.arange(self.det[0], dtype=np.float32)[:, None]
        cols = np.arange(self.det[1], dtype=np.float32)[None, :]
        dist = np.sqrt((rows - float(center_row)) ** 2 + (cols - float(center_col)) ** 2)
        ceilbin = np.ceil(dist).astype(np.int32).reshape(-1)
        floorbin = np.floor(dist).astype(np.int32).reshape(-1)
        nbins = max(int(ceilbin.max()), int(floorbin.max())) + 1

        nbytes_radbin = self.ndet * 4
        if self._radial_radbin_mtl is None:
            self._radial_radbin_mtl = self._mps._metal_buffer_alloc(nbytes_radbin)
            self._radial_radbin_np = self._mps._numpy_view(
                self._radial_radbin_mtl, np.int32, self.ndet
            )
        self._radial_radbin_np[:] = ceilbin
        if self._row_prefix:
            if self._radial_floor_radbin_mtl is None:
                self._radial_floor_radbin_mtl = self._mps._metal_buffer_alloc(
                    nbytes_radbin
                )
                self._radial_floor_radbin_np = self._mps._numpy_view(
                    self._radial_floor_radbin_mtl, np.int32, self.ndet
                )
            self._radial_floor_radbin_np[:] = floorbin

        out_count = self.n * nbins
        out_nbytes = out_count * 4
        if self._radial_out_mtl is None or self._radial_out_mtl.length() < out_nbytes:
            self._radial_out_mtl = self._mps._metal_buffer_alloc(out_nbytes)
            self._radial_out_np = self._mps._numpy_view(
                self._radial_out_mtl, np.int32, out_count
            )
        else:
            self._radial_out_np = self._mps._numpy_view(
                self._radial_out_mtl, np.int32, out_count
            )
        if self._row_prefix:
            if (
                self._radial_floor_out_mtl is None
                or self._radial_floor_out_mtl.length() < out_nbytes
            ):
                self._radial_floor_out_mtl = self._mps._metal_buffer_alloc(out_nbytes)
                self._radial_floor_out_np = self._mps._numpy_view(
                    self._radial_floor_out_mtl, np.int32, out_count
                )
            else:
                self._radial_floor_out_np = self._mps._numpy_view(
                    self._radial_floor_out_mtl, np.int32, out_count
                )

        use_radial_tg = bool(self._row_prefix and nbins <= 384)
        pipe = (
            self._radial_dual_prefix_tg_pipe if use_radial_tg
            else self._radial_dual_prefix_pipe if self._row_prefix
            else self._radial_pipe
        )
        Metal = self._Metal
        cmds = []
        for group in _chunk_groups(self.chunks):
            cmd = self._mps._queue.commandBuffer()
            enc = cmd.computeCommandEncoder()
            enc.setComputePipelineState_(pipe)
            for ci in group:
                nf = int(self.chunks[ci].shape[0])
                out_offset = self._offsets[ci] * nbins * 4
                enc.setBuffer_offset_atIndex_(self.chunks[ci]._mtl, 0, 0)
                enc.setBuffer_offset_atIndex_(self._radial_radbin_mtl, 0, 1)
                if self._row_prefix:
                    enc.setBuffer_offset_atIndex_(self._radial_floor_radbin_mtl, 0, 2)
                    enc.setBuffer_offset_atIndex_(self._radial_out_mtl, out_offset, 3)
                    enc.setBuffer_offset_atIndex_(
                        self._radial_floor_out_mtl, out_offset, 4
                    )
                    enc.setBuffer_offset_atIndex_(self._ndet_mtl, 0, 5)
                    enc.setBuffer_offset_atIndex_(self._detcols_mtl, 0, 6)
                    enc.setBytes_length_atIndex_(
                        np.array([nbins], dtype=np.uint32).tobytes(), 4, 7
                    )
                    enc.setBuffer_offset_atIndex_(self._nf_mtls[ci], 0, 8)
                else:
                    enc.setBuffer_offset_atIndex_(self._radial_out_mtl, out_offset, 2)
                    enc.setBuffer_offset_atIndex_(self._ndet_mtl, 0, 3)
                    enc.setBytes_length_atIndex_(
                        np.array([nbins], dtype=np.uint32).tobytes(), 4, 4
                    )
                    enc.setBuffer_offset_atIndex_(self._nf_mtls[ci], 0, 5)
                if use_radial_tg:
                    enc.dispatchThreadgroups_threadsPerThreadgroup_(
                        Metal.MTLSizeMake(nf, 1, 1),
                        Metal.MTLSizeMake(256, 1, 1))
                else:
                    enc.dispatchThreadgroups_threadsPerThreadgroup_(
                        Metal.MTLSizeMake((nf + 255) // 256, 1, 1),
                        Metal.MTLSizeMake(256, 1, 1))
            enc.endEncoding()
            cmd.commit()
            cmds.append(cmd)
        cmds[-1].waitUntilCompleted()
        self._radial_center = center
        self._radial_nbins = nbins

    def radial_cache_ready(self, center_row: float, center_col: float) -> bool:
        center = (round(float(center_row), 3), round(float(center_col), 3))
        if self._radial_center != center or self._radial_out_np is None:
            return False
        return (not self._row_prefix) or self._radial_floor_out_np is not None

    def radial_masked_sum(
        self,
        *,
        center_row: float,
        center_col: float,
        outer_radius: float,
        inner_radius: float = 0.0,
        build: bool = True,
    ) -> np.ndarray | None:
        outer = float(outer_radius)
        inner = float(inner_radius)
        outer_i = int(round(outer))
        inner_i = int(round(inner))
        if abs(outer - outer_i) > 1e-6 or abs(inner - inner_i) > 1e-6:
            return None
        if outer_i < 0 or inner_i < 0 or outer_i < inner_i:
            return None
        if inner_i > 0 and not self._row_prefix:
            return None
        if not self.radial_cache_ready(center_row, center_col):
            if not build:
                return None
            self._ensure_radial_cache(center_row, center_col)
        if inner_i >= self._radial_nbins:
            self._full.fill(0)
            return self._full
        outer_i = min(outer_i, self._radial_nbins - 1)
        radial_ceil = self._radial_out_np[: self.n * self._radial_nbins].reshape(
            self.n, self._radial_nbins
        )
        if inner_i <= 0:
            self._full[:] = radial_ceil[:, outer_i]
        else:
            radial_floor = self._radial_floor_out_np[
                : self.n * self._radial_nbins
            ].reshape(self.n, self._radial_nbins)
            self._full[:] = radial_ceil[:, outer_i] - radial_floor[:, inner_i - 1]
        return self._full

    def detector_sum(self) -> np.ndarray:
        """Per-pixel sum over all scan positions -> mean DP (det_h, det_w) float32.

        Raw Metal: one thread per detector pixel sums over a chunk's frames and
        atomic-adds into the shared output. ~1.7s for the full 19.3 GB (vs ~13 s
        for a host numpy reduce). Used once, at auto-center.
        """
        Metal = self._Metal
        mps = self._mps
        ds_out = mps._metal_buffer_alloc(self.ndet * 4)
        ds_np = mps._numpy_view(ds_out, np.int32, self.ndet)
        ds_np[:] = 0
        # detector_sum atomic-adds into ONE shared buffer; concurrent command
        # buffers racing the same atomics is risky, so wait each group here (it
        # runs once, at auto-center — the per-drag BF path is the one that uses
        # the faster wait-last, where each chunk writes its own output).
        for group in _chunk_groups(self.chunks):
            cmd = mps._queue.commandBuffer()
            enc = cmd.computeCommandEncoder()
            enc.setComputePipelineState_(
                self._detsum_prefix_pipe if self._row_prefix else self._detsum_pipe)
            for ci in group:
                enc.setBuffer_offset_atIndex_(self.chunks[ci]._mtl, 0, 0)
                enc.setBuffer_offset_atIndex_(ds_out, 0, 1)
                enc.setBuffer_offset_atIndex_(self._ndet_mtl, 0, 2)
                if self._row_prefix:
                    enc.setBuffer_offset_atIndex_(self._detcols_mtl, 0, 3)
                    enc.setBuffer_offset_atIndex_(self._nf_mtls[ci], 0, 4)
                else:
                    enc.setBuffer_offset_atIndex_(self._nf_mtls[ci], 0, 3)
                enc.dispatchThreadgroups_threadsPerThreadgroup_(
                    Metal.MTLSizeMake((self.ndet + 255) // 256, 1, 1),
                    Metal.MTLSizeMake(256, 1, 1))
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        return ds_np.reshape(self.det).astype(np.float32)


class ChunkedFrames:
    """A 3D ``(N, det_h, det_w)`` uint16 view over the Metal-buffer chunks.

    Forwards ``shape``/``dtype``/``ndim`` so ``Show4DSTEM.__init__`` treats it as
    a flat-scan 3D stack, and serves single-frame cursor reads from the numpy
    views. BF/DF go through the raw-Metal ``MetalVirtualImage`` (``self.vi``), not
    through any torch op.
    """

    _is_gpu_frames = True

    def __init__(self, chunks: list, *, row_prefix: bool = False):
        metadata = {}
        det_bin = 1
        fast_chunks = None
        # Scrub sidecar bin factor: bin4 on a 24 GB Mac (fits + ~4x fewer
        # detector pixels per virtual-image sum = higher scrub FPS), bin2 on
        # bigger boxes. An explicit fast_det_bin in the load metadata wins.
        fast_det_bin = default_fast_bin()
        if hasattr(chunks, "chunks"):
            metadata = dict(getattr(chunks, "metadata", {}) or {})
            det_bin = int(
                getattr(chunks, "det_bin", metadata.get("det_bin", 1)) or 1
            )
            fast_chunks = getattr(chunks, "fast_chunks", None)
            fast_det_bin = int(
                getattr(chunks, "fast_det_bin",
                        metadata.get("fast_det_bin", fast_det_bin)) or fast_det_bin
            )
            row_prefix = bool(
                row_prefix
                or getattr(chunks, "row_prefix", False)
                or metadata.get("row_prefix", False)
            )
            chunks = chunks.chunks
        if chunks:
            row_prefix = bool(row_prefix or getattr(chunks[0], "_row_prefix", False))
        if not chunks:
            raise ValueError("ChunkedFrames requires at least one chunk")
        import torch
        self._torch = torch
        self.chunks = chunks
        self.metadata = metadata
        self.det_bin = det_bin
        self._det = tuple(int(x) for x in chunks[0].shape[1:])
        self._frame_elems = self._det[0] * self._det[1]
        self._n = int(sum(int(c.shape[0]) for c in chunks))
        self.shape = (self._n, *self._det)
        self.ndim = 3
        self.dtype = torch.uint16
        # Heavy compute stays in raw Metal buffers; torch is only used by the
        # base widget for small masks/traits. Keeping these helper tensors on
        # CPU avoids MPS allocator startup cost and high-watermark pressure.
        self.device = torch.device("cpu")
        self._offsets = [0]
        for c in chunks:
            self._offsets.append(self._offsets[-1] + int(c.shape[0]))
        self.vi = MetalVirtualImage(chunks, row_prefix=row_prefix)
        self.fast_chunks = fast_chunks
        self.fast_vi = MetalVirtualImage(fast_chunks) if fast_chunks else None
        self.fast_bin = fast_det_bin

    def element_size(self) -> int:
        return 2

    def numel(self) -> int:
        return self._n * self._frame_elems

    @property
    def nbytes(self) -> int:
        return self.numel() * 2

    def __len__(self) -> int:
        return self._n

    def _locate(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += self._n
        for ci in range(len(self.chunks)):
            if idx < self._offsets[ci + 1]:
                return ci, idx - self._offsets[ci]
        raise IndexError(idx)

    def frame(self, idx: int) -> np.ndarray:
        """One diffraction pattern (det_h, det_w) as numpy uint16 — cursor read."""
        ci, local = self._locate(int(idx))
        frame = np.asarray(self.chunks[ci][local])
        if not self.vi.row_prefix_enabled:
            return frame
        out = np.empty_like(frame)
        out[:, 0] = frame[:, 0]
        out[:, 1:] = (
            frame[:, 1:].astype(np.uint32) - frame[:, :-1].astype(np.uint32)
        ).astype(frame.dtype)
        return out

    def column(self, row: int, col: int) -> np.ndarray:
        """One detector pixel over all scan positions as numpy uint16."""
        row = int(row)
        col = int(col)
        out = np.empty(self._n, dtype=np.uint16)
        for ci, chunk in enumerate(self.chunks):
            start = self._offsets[ci]
            stop = self._offsets[ci + 1]
            if self.vi.row_prefix_enabled and col > 0:
                out[start:stop] = (
                    np.asarray(chunk[:, row, col], dtype=np.uint32)
                    - np.asarray(chunk[:, row, col - 1], dtype=np.uint32)
                ).astype(np.uint16)
            else:
                out[start:stop] = np.asarray(chunk[:, row, col], dtype=np.uint16)
        return out

    def ensure_fast_interaction(self, *, verbose: bool = True) -> MetalVirtualImage:
        """Prepare the detector-bin``fast_bin`` sidecar for fast virtual images.

        Built IN PLACE by binning the resident no-bin chunks (``bin_chunks``) -
        no disk re-decode, no decompress scratch. So the only new memory is the
        sidecar output itself (1.2 GB at bin4), and the build is a single Metal
        pass over data already on the GPU. This is what lets the no-bin viewer
        open + scrub on a 24 GB Mac without a second 19 GB decode spike.
        """
        if self.det_bin > 1:
            return self.vi
        if self.fast_vi is not None:
            return self.fast_vi
        if self.vi.row_prefix_enabled:
            raise RuntimeError(
                "fast_interaction requires raw MPS chunks; load without "
                "row_prefix=True."
            )
        from quantem.widget.show4dstem_mps import _drop_cached_decompressor  # lazy: avoids circular import
        _drop_cached_decompressor()
        gc.collect()
        self.fast_chunks = self.vi.bin_chunks(self.fast_bin, verbose=verbose)
        gc.collect()
        self.fast_vi = MetalVirtualImage(self.fast_chunks)
        return self.fast_vi

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self._torch.from_numpy(self.frame(int(key)))
        if (isinstance(key, tuple) and len(key) == 3
                and isinstance(key[0], slice) and key[0] == slice(None)):
            r, c = int(key[1]), int(key[2])
            if self.vi.row_prefix_enabled and c > 0:
                cols = [
                    (
                        np.asarray(ch[:, r, c], dtype=np.uint32)
                        - np.asarray(ch[:, r, c - 1], dtype=np.uint32)
                    ).astype(np.uint16)
                    for ch in self.chunks
                ]
            else:
                cols = [np.asarray(ch[:, r, c]) for ch in self.chunks]
            return self._torch.from_numpy(np.concatenate(cols))
        raise TypeError("ChunkedFrames: integer-frame or [:, r, c] indexing only")




class MultiChunkedFrames(ChunkedFrames):
    """N datasets stacked as a 5D ``(N, scan, scan, det, det)`` view.

    Subclasses ChunkedFrames so every ``isinstance(data, ChunkedFrames)`` check in
    the viewer passes. Holds a list of per-dataset ChunkedFrames; ``active_idx``
    (driven by the viewer's frame slider) selects which one ``vi`` / ``fast_vi`` /
    ``chunks`` / ``frame()`` proxy to. So ONE Show4DSTEM_MACBOOK slides across the
    datasets via the native 5D frame axis, each slide instant once that dataset is
    decoded. Datasets may be filled lazily (a background loader appends to the
    list + sets the slot ready); ``__getitem__`` is the per-frame dataset accessor
    the base 5D path expects.
    """

    _is_gpu_frames = True

    def __init__(self, datasets: "list[ChunkedFrames]", n_total: int | None = None,
                 names: "list[str] | None" = None):
        """``datasets`` must hold at least dataset 0 (decoded). ``n_total`` sizes
        the 5D frame axis upfront (default len(datasets)) so the slider spans all
        slots immediately; later slots may be None until a background loader fills
        them via ``set_dataset(i, frames)``. The viewer's frame slider thus shows
        all N positions from the start; sliding to an unfilled slot falls back to
        the last ready dataset (the viewer also shows a loading badge)."""
        if not datasets or datasets[0] is None:
            raise ValueError("MultiChunkedFrames needs dataset 0 decoded")
        n_total = int(n_total) if n_total else len(datasets)
        self.datasets: list = list(datasets) + [None] * (n_total - len(datasets))
        self.names: list = list(names) if names else [f"dataset {i}" for i in range(n_total)]
        # Set by the viewer so each background-decoded dataset can grow the frame
        # slider + refresh the loading banner. Signature: on_ready(idx: int).
        self.on_ready = None
        self.active_idx = 0
        d0 = self.datasets[0]
        self._det = d0._det
        self._frame_elems = d0._frame_elems
        self.det_bin = d0.det_bin
        self._torch = d0._torch
        scan = int(round(d0._n ** 0.5))
        self._scan = (scan, scan)
        self.shape = (n_total, scan, scan, *self._det)
        self.ndim = 5
        self.dtype = d0.dtype
        self.device = d0.device

    def set_dataset(self, idx: int, frames: "ChunkedFrames") -> None:
        """Background loader fills slot ``idx`` once decoded, then pings the viewer
        (if wired) so the frame slider grows + the loading banner updates."""
        self.datasets[idx] = frames
        if self.on_ready is not None:
            self.on_ready(idx)

    @property
    def n_ready(self) -> int:
        """How many dataset slots are decoded so far (>=1)."""
        return sum(1 for d in self.datasets if d is not None)

    def is_ready(self, idx: int) -> bool:
        return 0 <= idx < len(self.datasets) and self.datasets[idx] is not None

    def set_active(self, idx: int) -> None:
        idx = max(0, min(int(idx), len(self.datasets) - 1))
        # if the requested slot isn't decoded yet, hold the last ready dataset so
        # the viewer paints SOMETHING (the loading badge tells the user to wait)
        if self.datasets[idx] is None:
            ready = [i for i in range(len(self.datasets)) if self.datasets[i] is not None]
            below = [i for i in ready if i <= idx]
            idx = (below[-1] if below else (ready[0] if ready else 0))
        self.active_idx = idx

    @property
    def _active(self) -> "ChunkedFrames":
        d = self.datasets[self.active_idx]
        if d is None:  # active should always be ready via set_active guard
            d = next(x for x in self.datasets if x is not None)
        return d

    # --- proxy the attrs the viewer touches to the active dataset ---
    @property
    def vi(self):
        return self._active.vi

    @property
    def fast_vi(self):
        return self._active.fast_vi

    @property
    def chunks(self):
        return self._active.chunks

    @property
    def _n(self):
        return self._active._n

    @property
    def metadata(self):
        return self._active.metadata

    def frame(self, idx: int) -> np.ndarray:
        return self._active.frame(idx)

    def ensure_fast_interaction(self, *args, **kwargs):
        return self._active.ensure_fast_interaction(*args, **kwargs)

    def __len__(self) -> int:
        return len(self.datasets)

    def __getitem__(self, key):
        # base 5D path does self._data[frame_idx] -> the i-th dataset (a 3D stack)
        if isinstance(key, (int, np.integer)):
            d = self.datasets[int(key)]
            return d if d is not None else self._active
        # scan-position lookup data[pos_row, pos_col] -> the active dataset's
        # frame at that scan position. The viewer's _update_frame uses this for
        # the CBED at the cursor (hit when n_frames == 1, e.g. a single-dataset
        # folder); flatten (row, col) to the active dataset's integer frame index
        # so it routes through ChunkedFrames.frame().
        if (isinstance(key, tuple) and len(key) == 2
                and all(isinstance(k, (int, np.integer)) for k in key)):
            r, c = int(key[0]), int(key[1])
            return self._active[r * self._scan[1] + c]
        return self._active[key]
