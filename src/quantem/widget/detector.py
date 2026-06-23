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
