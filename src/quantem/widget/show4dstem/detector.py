"""Detector and scan masks for the viewer, and the bright-field disk fit.

The widget owns the geometry of every virtual detector (``detector_mask``,
``fit_probe``); ``quantem.gpu.detector`` uses the same definitions, which the
parity tests check, so a viewer ROI and ``detector.bf`` / ``detector.adf`` are
pixel-identical. This module turns the widget's ROI traits into masks and hands
them to a detector session.
"""

import numpy as np

PRESET_RADII = {"bf": (0.0, 1.0), "abf": (0.5, 1.0), "adf": (1.0, 2.0), "haadf": (2.0, 4.0)}


def roi_mask(
    mode: str,
    det_shape: tuple[int, int],
    *,
    center_row: float,
    center_col: float,
    radius: float,
    radius_inner: float,
    width: float,
    height: float,
) -> np.ndarray:
    """Boolean detector mask for one viewer ROI.

    ``circle`` and ``annular`` go through ``detector_mask`` so they match the
    library's virtual detectors pixel for pixel; ``square``,
    ``rect`` and the single-pixel ``point`` are axis-aligned selections. A
    degenerate size (zero radius or side) falls back to the point so the virtual
    image never becomes an all-zero sum.
    """
    center = (center_row, center_col)
    if mode == "circle" and radius > 0:
        return detector_mask(center, 0.0, radius, det_shape)
    if mode == "annular" and radius > 0:
        return detector_mask(center, radius_inner, radius, det_shape)
    rows = np.arange(det_shape[0], dtype=np.float32)[:, None]
    cols = np.arange(det_shape[1], dtype=np.float32)[None, :]
    if mode == "square" and radius > 0:
        return (np.abs(cols - center_col) <= radius) & (np.abs(rows - center_row) <= radius)
    if mode == "rect" and width > 0 and height > 0:
        return (np.abs(cols - center_col) <= width / 2) & (np.abs(rows - center_row) <= height / 2)
    point = np.zeros(det_shape, dtype=bool)
    row = int(max(0, min(round(center_row), det_shape[0] - 1)))
    col = int(max(0, min(round(center_col), det_shape[1] - 1)))
    point[row, col] = True
    return point


def preset_mask(name: str, det_shape: tuple[int, int], center: tuple[float, float], bf_radius: float) -> np.ndarray:
    """Detector mask of a named preset (BF, ABF, ADF, HAADF) around the fitted disk.

    The annuli are multiples of the bright-field radius: ABF covers the outer
    half of the disk, ADF one to two radii, HAADF two to four radii. These are
    the ranges ``Show4DSTEM.apply_preset`` writes into the ROI traits, so a
    preset click and its precomputed image use the same pixels.
    """
    inner, outer = PRESET_RADII[name]
    radius = max(1.0, float(bf_radius))
    return detector_mask(center, inner * radius, outer * radius, det_shape)


def scan_indices(
    mode: str,
    scan_shape: tuple[int, int],
    *,
    center_row: float,
    center_col: float,
    radius: float,
    width: float,
    height: float,
) -> np.ndarray:
    """Flat row-major scan indices inside the virtual-image ROI.

    The diffraction panel reduces (mean, sum or max) the patterns at these
    positions; an empty selection means the ROI lies outside the scan.
    """
    rows = np.arange(scan_shape[0], dtype=np.float32)[:, None]
    cols = np.arange(scan_shape[1], dtype=np.float32)[None, :]
    if mode == "circle":
        inside = (rows - center_row) ** 2 + (cols - center_col) ** 2 <= radius**2
    elif mode == "square":
        inside = (np.abs(rows - center_row) <= radius) & (np.abs(cols - center_col) <= radius)
    else:
        inside = (np.abs(rows - center_row) <= height / 2) & (np.abs(cols - center_col) <= width / 2)
    return np.flatnonzero(inside)


def fit_disk(mean_dp: np.ndarray, det_shape: tuple[int, int], default_ratio: float) -> tuple[float, float, float]:
    """Bright-field disk ``(center_row, center_col, radius)`` from the mean pattern.

    ``fit_probe`` thresholds at mean plus one standard
    deviation and takes the equivalent-area radius. When no pixel passes the
    threshold (a blank or uniform pattern) the viewer keeps the detector
    midpoint and ``default_ratio`` of the smaller detector side, which is what
    the untouched traits already hold.
    """
    (center_row, center_col), radius = fit_probe(mean_dp)
    pattern = np.asarray(mean_dp, dtype=np.float32)
    if not np.any(pattern > float(pattern.mean()) + float(pattern.std())):
        return det_shape[0] / 2, det_shape[1] / 2, min(det_shape) * default_ratio
    return center_row, center_col, radius


def detector_mask(center: tuple[float, float], inner_px: float, outer_px: float, det_shape: tuple[int, int]) -> np.ndarray:
    """Boolean ``(det_row, det_col)`` mask of the pixels between two radii of ``center``.

    A pixel is selected when its float32 distance ``sqrt((row - r0)^2 + (col - c0)^2)``
    from ``center`` lies in ``[inner_px, outer_px]``. Computing the distance in
    float32, as ``quantem.gpu.detector.detector_mask`` does, keeps the boundary
    pixels of a BF disk identical between the widget and the library.
    """
    center_row, center_col = center
    rows = np.arange(det_shape[0], dtype=np.float32)[:, None]
    cols = np.arange(det_shape[1], dtype=np.float32)[None, :]
    distance = np.sqrt((rows - center_row) ** 2 + (cols - center_col) ** 2)
    return (distance >= inner_px) & (distance <= outer_px)


def fit_probe(mean_dp: np.ndarray) -> tuple[tuple[float, float], float]:
    """Bright-field disk ``((center_row, center_col), radius)`` of a mean pattern.

    Pixels above ``mean + std`` (population standard deviation) form the disk;
    the centre is their unweighted centroid and the radius ``sqrt(area / pi)``.
    With no pixel above the threshold the detector midpoint and a quarter of the
    smaller side are returned.
    """
    pattern = np.asarray(mean_dp, dtype=np.float32)
    disk = pattern > float(pattern.mean()) + float(pattern.std())
    area = int(disk.sum())
    if area == 0:
        detector_rows, detector_cols = pattern.shape
        return (detector_rows / 2.0, detector_cols / 2.0), min(detector_rows, detector_cols) * 0.25
    rows = np.arange(pattern.shape[0], dtype=np.float32)[:, None]
    cols = np.arange(pattern.shape[1], dtype=np.float32)[None, :]
    center_row = float((rows * disk).sum() / area)
    center_col = float((cols * disk).sum() / area)
    return (center_row, center_col), float(np.sqrt(area / np.pi))
