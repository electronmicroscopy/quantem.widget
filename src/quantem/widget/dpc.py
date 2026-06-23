"""Differential Phase Contrast (CoM / DPC / iDPC) for 4D-STEM — backend-agnostic.

CoM, DPC, and iDPC are DERIVED scalar fields, not raw 4D data, so they live here
(viewed with ``Show2D``), separate from the raw ``Show4DSTEM`` viewer.

The only expensive step is the per-scan-position center of mass over the full
detector - one pass over the (no-bin) 4D block:
  - MPS (MacBook): the raw-Metal ``com_u16`` kernel over chunked uint16 buffers,
    int64 accumulate, no float cast - fits the full 19 GB no-bin stack in 24 GB.
  - CUDA / CPU: a chunked numpy/torch CoM (same formula).
Everything after CoM (rotation alignment, Fourier integration) is small-field
math on the ``(scan_row, scan_col)`` CoM, ported 1:1 from quantem.live's
``engine.dpc`` so results match the dashboard.

Usage::

    from quantem.widget import load, dpc, Show2D
    result = dpc(load("scan_master.h5"))     # CoM + auto-rotation + iDPC
    Show2D(result.phase)                      # the iDPC phase image
    Show2D(result.com_col)                    # raw DPC field (col)
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from quantem.widget.utils.array import unwrap_core_4dstem


@dataclass
class DPCResult:
    """Outputs of :func:`dpc`, all ``(scan_row, scan_col)`` float32 for ``Show2D``."""
    phase: np.ndarray            # iDPC phase (Fourier-integrated CoM)
    com_row: np.ndarray          # raw CoM, detector-row component (mean-subtracted)
    com_col: np.ndarray          # raw CoM, detector-col component (mean-subtracted)
    com_row_aligned: np.ndarray  # rotation-aligned CoM row
    com_col_aligned: np.ndarray  # rotation-aligned CoM col
    rotation_deg: float          # scan<->detector rotation (auto or forced)
    use_transpose: bool
    elapsed: float


# --- small-field math (ported 1:1 from quantem.live.engine.dpc, cp -> np) ---


def _freq_grid_2d(shape):
    f_row = np.fft.fftfreq(shape[0]).astype(np.float32)
    f_col = np.fft.fftfreq(shape[1]).astype(np.float32)
    return np.meshgrid(f_row, f_col, indexing="ij")


def _rotate_vector_batch(v_row, v_col, angles_rad):
    c = np.cos(angles_rad)[:, None, None]
    s = np.sin(angles_rad)[:, None, None]
    return c * v_row - s * v_col, s * v_row + c * v_col


def _curl_batch(v_row, v_col):
    # curl = d(v_col)/d_row - d(v_row)/d_col, central differences, mean-squared
    dv_row_dcol = 0.5 * (v_row[:, 1:-1, 2:] - v_row[:, 1:-1, :-2])
    dv_col_drow = 0.5 * (v_col[:, 2:, 1:-1] - v_col[:, :-2, 1:-1])
    curl = dv_col_drow - dv_row_dcol
    return (curl ** 2).mean(axis=(1, 2))


def find_optimal_rotation(com_row, com_col, rotation_steps=180):
    """Rotation (deg) that minimizes the curl of the CoM field; tests transpose too."""
    angles = np.linspace(0, np.pi, rotation_steps, dtype=np.float32)
    r, c = _rotate_vector_batch(com_row, com_col, angles)
    rt, ct = _rotate_vector_batch(com_col, com_row, angles)
    curls = _curl_batch(r, c)
    curls_t = _curl_batch(rt, ct)
    stacked = np.concatenate([curls, curls_t])
    idx = int(stacked.argmin())
    use_transpose = idx >= rotation_steps
    ai = idx % rotation_steps
    angle_deg = float(angles[ai]) * 180.0 / np.pi
    if use_transpose:
        return rt[ai], ct[ai], angle_deg, True
    return r[ai], c[ai], angle_deg, False


def reconstruct_phase_from_gradient(grad_row, grad_col):
    """Fourier-integrate the gradient field (Poisson solver) -> phase image."""
    grad_row = grad_row.astype(np.float32)
    grad_col = grad_col.astype(np.float32)
    g0 = np.fft.fft2(grad_row)
    g1 = np.fft.fft2(grad_col)
    k0, k1 = _freq_grid_2d(grad_row.shape)
    k2 = k0 ** 2 + k1 ** 2
    k2[0, 0] = 1.0
    phase_fft = (-1j * 0.25) * (k0 * g0 + k1 * g1) / k2
    phase_fft[0, 0] = 0
    return np.real(np.fft.ifft2(phase_fft)).astype(np.float32)


# --- CoM dispatch (the only step that touches the 4D block) ---


def _com_numpy(data, scan_shape, chunk=4096):
    """Chunked numpy/cupy CoM over a 3D/4D array - never casts the whole block."""
    xp = np
    try:
        import cupy as _cp
        if type(data).__module__.split(".")[0] == "cupy":
            xp = _cp
    except ImportError:
        pass
    if data.ndim == 4:
        data = data.reshape(-1, *data.shape[-2:])
    n, kr, kc = data.shape
    row_idx = xp.arange(kr, dtype=xp.float64)
    col_idx = xp.arange(kc, dtype=xp.float64)
    com_row = xp.empty(n, dtype=xp.float32)
    com_col = xp.empty(n, dtype=xp.float32)
    for i in range(0, n, chunk):
        block = data[i:i + chunk]
        sum_kc = block.sum(axis=2, dtype=xp.float64)   # (b, kr)
        sum_kr = block.sum(axis=1, dtype=xp.float64)   # (b, kc)
        isum = xp.maximum(sum_kc.sum(axis=1), 1e-10)
        com_row[i:i + chunk] = ((sum_kc * row_idx[None, :]).sum(axis=1) / isum).astype(xp.float32)
        com_col[i:i + chunk] = ((sum_kr * col_idx[None, :]).sum(axis=1) / isum).astype(xp.float32)
    if xp is not np:
        com_row, com_col = xp.asnumpy(com_row), xp.asnumpy(com_col)
    return com_row, com_col


def center_of_mass(data, scan_shape=None, mask=None):
    """Per-scan-position CoM ``(com_row, com_col)``, mean-subtracted, ``(scan,scan)``.

    MPS chunked input -> raw-Metal ``com_u16`` (no-bin, no float cast). Array input
    (numpy/cupy/torch) -> chunked numpy/cupy CoM. Same formula either way.
    """
    data = unwrap_core_4dstem(data)
    # Unwrap a LoadResult
    if hasattr(data, "_fields") and "data" in getattr(data, "_fields", ()):
        data = data.data
    # MPS raw chunks (MPSChunked4DSTEM or ChunkedFrames) -> Metal CoM kernel
    vi = getattr(data, "vi", None)
    if vi is None and hasattr(data, "chunks"):
        from quantem.widget.kernels.compute.mps import ChunkedFrames
        data = ChunkedFrames(data)
        vi = data.vi
    if vi is not None:
        n = vi.n
        sr = int(round(n ** 0.5)); sc = n // sr
        det = vi.det
        m = None if mask is None else np.asarray(mask).reshape(det)
        com_col, com_row = vi.center_of_mass(m)   # absolute coords
    else:
        import torch
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        elif type(data).__module__.split(".")[0] != "cupy":
            # numpy / list -> ndarray. cupy passes through: _com_numpy reduces it
            # on-device (xp=cp) and returns numpy, so CUDA input works too.
            data = np.asarray(data)
        if scan_shape is None:
            n = data.shape[0] if data.ndim == 3 else data.shape[0] * data.shape[1]
            sr = int(round(n ** 0.5)); sc = n // sr
        else:
            sr, sc = scan_shape
        com_row, com_col = _com_numpy(data, (sr, sc))
    com_row = np.asarray(com_row, dtype=np.float32) - float(np.mean(com_row))
    com_col = np.asarray(com_col, dtype=np.float32) - float(np.mean(com_col))
    return com_row.reshape(sr, sc), com_col.reshape(sr, sc)


def dpc(data, scan_shape=None, *, rotation_angle_deg=None, rotation_steps=180,
        mask=None, verbose=False) -> DPCResult:
    """Center-of-mass -> optimal scan/detector rotation -> iDPC phase.

    ``data`` is ``load(...)`` output (MPS chunks, cupy, or numpy). The CoM is the
    one pass over the 4D block; rotation + integration are small-field. View the
    result with ``Show2D`` (``result.phase`` for iDPC, ``result.com_col`` for the
    raw DPC field).
    """
    t0 = time.perf_counter()
    com_row, com_col = center_of_mass(data, scan_shape=scan_shape, mask=mask)
    if rotation_angle_deg is None:
        cr, cc, angle, transp = find_optimal_rotation(com_row, com_col, rotation_steps)
    else:
        a = np.radians(rotation_angle_deg)
        cr = np.cos(a) * com_row - np.sin(a) * com_col
        cc = np.sin(a) * com_row + np.cos(a) * com_col
        angle, transp = float(rotation_angle_deg), False
    # Match engine.dpc.DPC.reconstruct: when transpose was selected the alignment
    # swapped (k_col, k_row), so swap back before integrating; zero-mean the phase;
    # and negate it (STEM convention - atoms appear dark).
    grad_row, grad_col = (cc, cr) if transp else (cr, cc)
    phase = reconstruct_phase_from_gradient(grad_row, grad_col)
    phase = phase - phase.mean()
    phase = (-phase).astype(np.float32)
    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"DPC: rotation {angle:.1f} deg (transpose={transp}), "
              f"{com_row.shape[0]}x{com_row.shape[1]} in {elapsed:.2f}s")
    return DPCResult(phase=phase, com_row=com_row, com_col=com_col,
                     com_row_aligned=cr.astype(np.float32), com_col_aligned=cc.astype(np.float32),
                     rotation_deg=angle, use_transpose=transp, elapsed=elapsed)


# --- friendly views: a CoM vector (.x / .y) and the iDPC phase image ---


class CoM:
    """Center-of-mass vector field: ``.row`` (vertical) and ``.col`` (horizontal)
    deflection maps, each a 2D ``(scan_row, scan_col)`` image for ``Show2D``.

    ``row``/``col`` match the rest of the API (``ds.center`` is ``(row, col)``),
    so the detector-axis convention is the same everywhere - no ``x``/``y``."""

    def __init__(self, com_row: np.ndarray, com_col: np.ndarray):
        self._row, self._col = com_row, com_col

    @property
    def row(self) -> np.ndarray:
        """Vertical CoM (detector-row deflection)."""
        return self._row

    @property
    def col(self) -> np.ndarray:
        """Horizontal CoM (detector-column deflection)."""
        return self._col


def com(data, **kwargs) -> CoM:
    """Center-of-mass vector field of ``data``: use ``.row`` / ``.col``."""
    com_row, com_col = center_of_mass(data, **kwargs)
    return CoM(com_row, com_col)


def idpc(data, **kwargs) -> np.ndarray:
    """Integrated-DPC phase image (CoM -> auto rotation -> Fourier integrate)."""
    return dpc(data, **kwargs).phase
