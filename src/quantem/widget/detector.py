"""Virtual detectors (bright / annular-dark / dark field) for 4D-STEM.

Primary API - place a virtual detector on 4D-STEM data and get its image, with
collection angles in **mrad**::

    from quantem.widget import load, Show2D
    data = load("master.h5")
    Show2D(bf(data))                       # bright field (the bright disk)
    Show2D(adf(data))                      # annular dark field (auto band)
    Show2D(adf(data, inner=50, outer=180)) # collection angles in mrad
    Show2D(df(data))                       # outside the bright disk

``bf`` / ``adf`` / ``df`` are thin geometry over the shared compute backend: they
build a boolean detector mask and call :func:`masked_sum` - the same fast
reduction that Show4DSTEM and live Browse use. The probe (disk center + size)
auto-fits from the mean diffraction pattern. MacBook (MPS) runs the raw-Metal
masked-sum over chunked uint16 buffers; CUDA / CPU runs torch. **No binning** on
either path.

The lower-level :func:`virtual` function (below) is mode-based
(DP/BF/ABF/ADF/HAADF/DF, bands measured in the auto-detected disk radius) and is
mainly the reference path the parity tests pin; ``ds.bf()`` etc. are the API.
"""

from __future__ import annotations

import numpy as np

from quantem.widget.utils.array import unwrap_core_4dstem


def _resolve_backend(data):
    """Return a compute backend (MetalCompute on MPS chunks, TorchCompute on array)."""
    data = unwrap_core_4dstem(data)
    if hasattr(data, "_fields") and "data" in getattr(data, "_fields", ()):
        data = data.data
    # raw MPS chunks -> wrap so compute_backend sees a _is_gpu_frames source
    if hasattr(data, "chunks") and not getattr(data, "_is_gpu_frames", False):
        from quantem.widget.kernels.compute.mps import ChunkedFrames
        data = ChunkedFrames(data)
    from quantem.widget.kernels.compute.backends import compute_backend
    return compute_backend(data)


def _scan_shape(data, backend) -> tuple[int, int]:
    if hasattr(data, "_fields") and "data" in getattr(data, "_fields", ()):
        data = data.data
    data = unwrap_core_4dstem(data)
    if hasattr(backend, "scan_shape"):
        return tuple(int(x) for x in backend.scan_shape)
    shape = getattr(data, "shape", None)
    if shape is not None and len(shape) >= 4:
        return int(shape[0]), int(shape[1])
    n = int(getattr(backend, "n_frames"))
    sr = int(round(n ** 0.5))
    return sr, n // sr


def _semiangle_mrad(data):
    if hasattr(data, "_fields") and "metadata" in getattr(data, "_fields", ()):
        meta = data.metadata or {}
        return meta.get("semiangle_mrad") or meta.get("semi_angle_mrad")
    meta = getattr(data, "metadata", None)
    if isinstance(meta, dict):
        return meta.get("semiangle_mrad") or meta.get("semi_angle_mrad")
    return getattr(data, "semiangle_mrad", None)


def mean_dp(data) -> np.ndarray:
    """Mean diffraction pattern for array/load/core-dataset/MPS inputs."""
    return np.asarray(_resolve_backend(data).mean_dp(), dtype=np.float32)


def masked_sum(data, det_mask) -> np.ndarray:
    """Masked detector sum over scan positions.

    This is the small public helper for code that needs the shared widget/live
    masked-sum compute path without constructing a widget-local dataset object.
    """
    backend = _resolve_backend(data)
    return np.asarray(backend.masked_sum(det_mask), dtype=np.float32).reshape(_scan_shape(data, backend))


def auto_probe(mean_dp):
    """Detect the probe (BF disk) from the mean diffraction pattern.

    Threshold at ``mean + std``, take the centroid of the bright disk for the
    center, and ``radius = sqrt(area / pi)``. Matches Show4DSTEM.auto_detect_center.
    Returns ``((center_row, center_col), bf_radius)``.
    """
    dp = np.asarray(mean_dp, dtype=np.float32)
    thr = float(dp.mean()) + float(dp.std())
    mask = dp > thr
    total = int(mask.sum())
    if total == 0:
        h, w = dp.shape
        return (h / 2.0, w / 2.0), min(h, w) * 0.25
    rows = np.arange(dp.shape[0], dtype=np.float32)[:, None]
    cols = np.arange(dp.shape[1], dtype=np.float32)[None, :]
    cy = float((rows * mask).sum() / total)
    cx = float((cols * mask).sum() / total)
    radius = float(np.sqrt(total / np.pi))
    return (cy, cx), radius


def detector_mask(center, lo_px, hi_px, det_shape) -> np.ndarray:
    """THE virtual-detector geometry primitive: boolean ``(det_row, det_col)`` mask
    of pixels whose distance from ``center`` (row, col) is in ``[lo_px, hi_px]``
    detector pixels. Every detector everywhere - ``ds.bf/adf/df``, the standalone
    ``virtual``, and the Show4DSTEM viewer's circle/annular ROIs - builds its mask
    here, so a viewer ROI and ``ds.adf()`` are pixel-identical by construction."""
    cy, cx = center
    rows = np.arange(det_shape[0], dtype=np.float32)[:, None]
    cols = np.arange(det_shape[1], dtype=np.float32)[None, :]
    dist = np.sqrt((rows - cy) ** 2 + (cols - cx) ** 2)
    return (dist >= lo_px) & (dist <= hi_px)


def _detector_mask(mode, center, bf_radius, det_shape, inner, outer):
    """Mode-based mask (BF/ABF/ADF/HAADF/DF, bands in disk-radius units) for the
    standalone :func:`virtual`. Resolves the band to pixel radii, then defers to
    :func:`detector_mask` - the one geometry primitive."""
    r = float(max(1.0, bf_radius))
    bands = {
        "BF": (0.0, r),
        "ABF": (0.5 * r, r),
        "ADF": (r, 2.0 * r),
        "HAADF": (2.0 * r, 4.0 * r),
        "DF": (r, np.inf),
    }
    if mode == "ANNULAR":
        lo, hi = (inner if inner is not None else 0.0) * r, (outer if outer is not None else np.inf) * r
    else:
        lo, hi = bands[mode]
    return detector_mask(center, lo, hi, det_shape)


# --- virtual detectors: thin geometry over the shared compute backend ---
# bf/adf/df build a boolean detector mask, then call the dataset's masked-sum
# (the single fast reduction in kernels/compute - the same one Show4DSTEM and any
# GUI use). Stateless: the probe auto-fits per call (override via center/radius),
# nothing is cached. Re-execute to rerun; cache at the edges (viewer/browser/caller).


def _mrad_to_px(data, mrad: float, radius: float) -> float:
    """Collection angle in mrad -> detector pixel radius. The bright disk radius
    spans ``semiangle_mrad``, so a mrad angle maps to
    ``mrad / semiangle_mrad * radius``."""
    semiangle_mrad = _semiangle_mrad(data)
    if not semiangle_mrad:
        raise ValueError(
            "inner / outer are collection angles in mrad, but the convergence "
            "semi-angle is unknown for this data. Store semiangle_mrad in metadata "
            "or pass detector pixels instead: adf(data, inner=..., outer=..., unit='px').")
    return float(mrad) / float(semiangle_mrad) * radius


def _to_px(data, value: float, unit: str, radius: float) -> float:
    """A collection-angle radius -> detector pixels. ``unit='mrad'`` (default)
    converts via the convergence semi-angle; ``unit='px'`` is already pixels
    (calibration-free, exact)."""
    unit = str(unit).lower()
    if unit in ("px", "pixel", "pixels"):
        return float(value)
    if unit == "mrad":
        return _mrad_to_px(data, value, radius)
    raise ValueError(f"unit must be 'mrad' or 'px', got {unit!r}")


def _probe(data, center=None, radius=None):
    if center is not None and radius is not None:
        return (float(center[0]), float(center[1])), float(radius)
    auto_center, auto_radius = auto_probe(mean_dp(data))
    center = (float(center[0]), float(center[1])) if center is not None else auto_center
    radius = float(radius) if radius is not None else auto_radius
    return center, radius


def _detector_image(data, center, lo_px: float, hi_px: float) -> np.ndarray:
    """Masked-sum image over the annulus ``lo_px .. hi_px`` detector pixels.
    Stateless - builds the mask via :func:`detector_mask` and runs the
    shared-backend masked-sum each call."""
    mask = detector_mask(center, lo_px, hi_px, mean_dp(data).shape)
    return masked_sum(data, mask)


def bf(data, center=None, radius=None) -> np.ndarray:
    """Bright-field image of ``data``: the bright disk (the unscattered probe).
    Probe auto-fits unless ``center``/``radius`` (detector pixels) are given."""
    center, radius = _probe(data, center, radius)
    return _detector_image(data, center, 0.0, radius)


def adf(data, inner: float | None = None, outer: float | None = None,
        unit: str = "mrad", center=None, radius=None) -> np.ndarray:
    """Annular-dark-field image of ``data``, collected between ``inner`` and
    ``outer``. ``unit='mrad'`` (default, needs ``ds.semiangle_mrad``) or
    ``unit='px'`` (raw detector pixels). Omit either for the automatic band:
    ``inner`` = the bright-disk edge, ``outer`` = twice that. Probe auto-fits
    unless ``center``/``radius`` (detector pixels) are given."""
    center, radius = _probe(data, center, radius)
    lo_px = radius if inner is None else _to_px(data, inner, unit, radius)
    hi_px = 2.0 * radius if outer is None else _to_px(data, outer, unit, radius)
    return _detector_image(data, center, lo_px, hi_px)


def df(data, inner: float | None = None, unit: str = "mrad",
       center=None, radius=None) -> np.ndarray:
    """Dark-field image of ``data``: everything collected beyond ``inner``.
    ``unit='mrad'`` (default, needs ``ds.semiangle_mrad``) or ``unit='px'``.
    Omit ``inner`` for everything outside the bright disk. Probe auto-fits
    unless ``center``/``radius`` (detector pixels) are given."""
    center, radius = _probe(data, center, radius)
    lo_px = radius if inner is None else _to_px(data, inner, unit, radius)
    return _detector_image(data, center, lo_px, np.inf)


def virtual(data, mode="BF", *, center=None, bf_radius=None, inner=None, outer=None):
    """Virtual image for ``mode`` with automatic probe fitting. See module docstring.

    ``mode`` is case-insensitive (DP/BF/ABF/ADF/HAADF/DF/annular). ``center`` and
    ``bf_radius`` override the auto-detected probe; ``inner``/``outer`` (BF-radius
    units) define a custom band when ``mode="annular"``. Returns a 2D float array
    (detector-space for DP, scan-space otherwise) for ``Show2D``.
    """
    dp = mean_dp(data)
    mode = str(mode).strip().upper()
    if mode == "DP":
        return dp
    if center is None or bf_radius is None:
        c_auto, r_auto = auto_probe(dp)
        center = center if center is not None else c_auto
        bf_radius = bf_radius if bf_radius is not None else r_auto
    mask = _detector_mask(mode, center, bf_radius, dp.shape, inner, outer)
    return masked_sum(data, mask)


# --- Migrated from quantem.live.engine.preprocess.brightfield ---
def detect_bf_radius(
    mean_dp: cp.ndarray,
    threshold_ratio: float = 0.1
) -> tuple[tuple[int, int], int]:
    """
    Detect BF disk center and radius from mean diffraction pattern.

    Runs entirely on GPU. Uses intensity thresholding for center-of-mass
    and radial profile analysis for the half-max radius.

    Parameters
    ----------
    mean_dp : cp.ndarray
        Mean diffraction pattern with shape (k_row, k_col).
    threshold_ratio : float
        Fraction of max intensity for thresholding (default: 0.1).

    Returns
    -------
    tuple[tuple[int, int], int]
        ((row_center, col_center), radius) - center coordinates and
        radius in pixels.

    Raises
    ------
    ValueError
        If the diffraction pattern is empty, all-zero, or contains
        only NaN/Inf values.
    """
    import cupy as cp
    if mean_dp.ndim != 2:
        raise ValueError(
            f"Expected 2D diffraction pattern, got {mean_dp.ndim}D "
            f"with shape {mean_dp.shape}"
        )
    n_k_row, n_k_col = mean_dp.shape
    if n_k_row == 0 or n_k_col == 0:
        raise ValueError(
            f"Diffraction pattern has zero-size dimension: shape {mean_dp.shape}"
        )
    dp = mean_dp.astype(cp.float32)
    dp_max = float(cp.nanmax(dp))
    if not np.isfinite(dp_max) or dp_max <= 0:
        raise ValueError(
            "Diffraction pattern has no positive finite values - "
            "cannot detect BF disk. Check that your data is loaded correctly."
        )
    # Threshold to find BF disk
    threshold = threshold_ratio * dp_max
    mask = dp > threshold
    if not bool(cp.any(mask)):
        raise ValueError(
            f"No pixels above threshold ({threshold_ratio:.0%} of max intensity). "
            f"The diffraction pattern may be too noisy or empty."
        )
    # Center of mass on GPU
    mask_f = mask.astype(cp.float32)
    total = float(mask_f.sum())
    row_coords = cp.arange(n_k_row, dtype=cp.float32).reshape(-1, 1)
    col_coords = cp.arange(n_k_col, dtype=cp.float32).reshape(1, -1)
    row_center_f = float((row_coords * mask_f).sum() / total)
    col_center_f = float((col_coords * mask_f).sum() / total)
    if not (np.isfinite(row_center_f) and np.isfinite(col_center_f)):
        raise ValueError(
            "Center-of-mass calculation returned NaN - "
            "diffraction pattern may be degenerate."
        )
    row_center = max(0, min(int(round(row_center_f)), n_k_row - 1))
    col_center = max(0, min(int(round(col_center_f)), n_k_col - 1))
    # Radial profile on GPU
    dr = cp.arange(n_k_row, dtype=cp.float32) - row_center
    dc = cp.arange(n_k_col, dtype=cp.float32) - col_center
    DR, DC = cp.meshgrid(dr, dc, indexing='ij')
    R = cp.sqrt(DR**2 + DC**2)
    # Integer-binned radial profile (vectorized, no Python loop)
    max_r = min(row_center, col_center, n_k_row - row_center, n_k_col - col_center)
    if max_r < 2:
        return (row_center, col_center), max(1, min(n_k_row, n_k_col) // 4)
    R_int = cp.rint(R).astype(cp.int32).ravel()
    dp_flat = dp.ravel()
    profile = cp.zeros(max_r, dtype=cp.float32)
    counts = cp.zeros(max_r, dtype=cp.float32)
    valid = R_int < max_r
    cp.add.at(profile, R_int[valid], dp_flat[valid])
    cp.add.at(counts, R_int[valid], cp.ones_like(dp_flat[valid]))
    nonzero = counts > 0
    profile[nonzero] /= counts[nonzero]
    # Gaussian smooth the profile on GPU (1D convolution)
    if int(profile.size) > 5:
        sigma = 2.0
        ksize = int(6 * sigma + 1) | 1  # ensure odd
        x = cp.arange(ksize, dtype=cp.float32) - ksize // 2
        kernel = cp.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()
        # Pad and convolve
        padded = cp.pad(profile, ksize // 2, mode='edge')
        profile_smooth = cp.convolve(padded, kernel, mode='valid')[:profile.size]
        center_intensity = float(profile_smooth[:5].mean())
        half_max = center_intensity * 0.5
        below_half = cp.where(profile_smooth < half_max)[0]
        if below_half.size > 0:
            radius = int(below_half[0])
        else:
            radius = int(profile.size) // 2
    else:
        radius = min(n_k_row, n_k_col) // 4
    radius = max(1, radius)
    return (row_center, col_center), radius


# --- Migrated from quantem.live.engine.preprocess (dp_mean + virtual_image) ---
def dp_mean(data: cp.ndarray) -> cp.ndarray:
    """
    Compute mean diffraction pattern on GPU.

    Uses integer reduction (``uint64`` accumulator) so there is no
    intermediate float32 copy of the full 4D array. For 512x512 x 192x192
    this saves ~38 GB of transient VRAM compared with
    ``data.astype(float32).mean(axis=0)``.

    Parameters
    ----------
    data : cp.ndarray
        3D ``(N, det_row, det_col)`` or 4D ``(scan_row, scan_col, det_row, det_col)``.

    Returns
    -------
    cp.ndarray
        2D array (det_row, det_col), float32.
    """
    import cupy as cp
    if data.ndim == 3:
        n = data.shape[0]
        return data.sum(axis=0, dtype=cp.uint64).astype(cp.float32) / n
    scan_row, scan_col = data.shape[0], data.shape[1]
    n = scan_row * scan_col
    return data.reshape(n, *data.shape[2:]).sum(axis=0, dtype=cp.uint64).astype(cp.float32) / n




def virtual_image(
    data: cp.ndarray,
    center_row: float,
    center_col: float,
    radius: float | None = None,
    inner_radius: float | None = None,
    outer_radius: float | None = None,
    chunk_size: int | None = None,
) -> cp.ndarray:
    """
    Compute a virtual image by summing masked detector pixels.

    Uses fancy indexing + integer reduction so we only allocate a copy of
    the masked pixels, not the entire detector. For a 512x512 scan with a
    BF mask of ~9000 pixels, peak VRAM is ~5 GB instead of ~40 GB.

    For large scans / wide annuli (e.g. DF outer=6.0 on 512², where the
    masked-pixels copy can still be GB-scale), the scan dimension is
    chunked. The chunk size is auto-tuned: an initial conservative
    estimate runs chunk 0, the actual peak allocation is measured, then
    remaining chunks are retuned. Uses the CLAUDE.md "Adaptive GPU
    chunking" pattern (estimate → measure → retune → safety factor 0.5).

    Parameters
    ----------
    data : cp.ndarray
        3D ``(N, det_row, det_col)`` or 4D ``(scan_row, scan_col, det_row, det_col)``.
        For 3D input, returns 2D ``(n, n)`` if the scan is square, else 1D.
    center_row, center_col : float
        Center of the detector mask in pixels.
    radius : float, optional
        Radius for circular (BF) mask.
    inner_radius, outer_radius : float, optional
        Radii for annular (DF) mask.
    chunk_size : int, optional
        Override the auto-tuned chunk size. ``None`` → auto. Pass a large
        value (or ``data.shape[0]``) to disable chunking.

    Returns
    -------
    cp.ndarray
        2D ``(scan_row, scan_col)`` for 4D input, or 2D ``(n, n)`` for 3D
        input (auto-detected square scan).
    """
    import cupy as cp
    import math
    import os

    det_row, det_col = data.shape[-2], data.shape[-1]

    # Single-source compute (DEFAULT): use the quantem.widget detector geometry +
    # masked-sum that the Jupyter widget and any GUI share, instead of the local
    # cupy path below. Proven bit-identical to the cupy path on real 512x512x192x192
    # (docs/2026-06-03-quantem-live-widget-single-source-migration.md). The cupy path
    # below is the fallback: any failure falls straight through to it, and
    # QT_LIVE_WIDGET_COMPUTE=0 forces it (memory-tuned chunked reduction, useful on
    # constrained GPUs). So the dashboard can never break - worst case it's cupy.
    if os.environ.get("QT_LIVE_WIDGET_COMPUTE", "1") != "0":
        try:
            from quantem.widget import Dataset4dstemGPU
            from quantem.widget.detector import detector_mask
            lo = 0.0 if radius is not None else float(inner_radius)
            hi = float(radius) if radius is not None else float(outer_radius)
            wmask = detector_mask((float(center_row), float(center_col)), lo, hi,
                                  (int(det_row), int(det_col)))
            wvi = Dataset4dstemGPU(data).masked_sum(wmask)  # widget single source
            wvi = cp.asarray(np.asarray(wvi), dtype=cp.float32)
            if data.ndim == 4:
                return wvi.reshape(data.shape[0], data.shape[1])
            side = int(math.isqrt(int(wvi.size)))
            return wvi.reshape(side, side) if side * side == wvi.size else wvi.ravel()
        except Exception:  # noqa: BLE001 — fall through to the cupy path
            pass

    k_row = cp.arange(det_row, dtype=cp.float32)
    k_col = cp.arange(det_col, dtype=cp.float32)
    k_row, k_col = cp.meshgrid(k_row, k_col, indexing='ij')
    dist = cp.sqrt((k_row - center_row) ** 2 + (k_col - center_col) ** 2)

    if radius is not None:
        mask = dist <= radius
    elif inner_radius is not None and outer_radius is not None:
        mask = (dist >= inner_radius) & (dist <= outer_radius)
    else:
        raise ValueError("Provide either radius (BF) or inner_radius + outer_radius (DF)")

    # Grab only the masked pixels, sum them with an integer accumulator.
    indices = cp.where(mask.ravel())[0]
    data_2d = data.reshape(-1, det_row * det_col)  # view, no copy
    n_total = int(data_2d.shape[0])
    n_masked = int(indices.size)

    # Initial chunk-size estimate. The transient on each chunk is the
    # fancy-index copy: chunk × n_masked × dtype_bytes. Aim to use ~25%
    # of free memory per chunk so a concurrent allocation (e.g. another
    # tab loading data) doesn't push us over.
    itemsize = int(data.dtype.itemsize)
    if chunk_size is None:
        try:
            free_bytes, _ = cp.cuda.runtime.memGetInfo()
            budget = max(64 * 1024 * 1024, int(free_bytes * 0.25))
            est_per_pos = max(1, n_masked * itemsize)
            chunk_size = max(1024, min(n_total, budget // est_per_pos))
        except Exception:  # noqa: BLE001 — fall back to one shot
            chunk_size = n_total
    chunk_size = max(1, min(chunk_size, n_total))

    # One-shot path when the whole scan fits in a single chunk.
    if chunk_size >= n_total:
        vi_1d = data_2d[:, indices].sum(axis=-1, dtype=cp.uint64).astype(cp.float32)
    else:
        # Pre-allocate output (small, just N_scan floats).
        vi_1d = cp.empty(n_total, dtype=cp.float32)
        i = 0
        while i < n_total:
            end = min(i + chunk_size, n_total)
            if i == 0:
                cp.cuda.runtime.deviceSynchronize()
                cp.get_default_memory_pool().free_all_blocks()
            chunk_acc = data_2d[i:end, indices].sum(axis=-1, dtype=cp.uint64)
            vi_1d[i:end] = chunk_acc.astype(cp.float32)
            del chunk_acc
            # After the first chunk, measure actual per-position cost and
            # retune subsequent chunks. The peak alloc divided by the chunk
            # we just ran gives the realized cost; pick 0.5× free / cost
            # for the next chunks (safety factor for fragmentation).
            if i == 0:
                try:
                    pool = cp.get_default_memory_pool()
                    free_bytes, _ = cp.cuda.runtime.memGetInfo()
                    used_now = int(pool.used_bytes())
                    realized_per_pos = max(1, used_now // max(1, end - i))
                    new_chunk = max(1024, int(free_bytes * 0.5 / realized_per_pos))
                    chunk_size = min(n_total, new_chunk)
                except Exception:  # noqa: BLE001
                    pass
            i = end

    if data.ndim == 4:
        return vi_1d.reshape(data.shape[0], data.shape[1])
    n = data.shape[0]
    side = int(math.isqrt(n))
    if side * side == n:
        return vi_1d.reshape(side, side)
    return vi_1d

