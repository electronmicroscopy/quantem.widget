"""Detector and scan masks for the viewer, and the bright-field disk fit.

The widget owns the geometry of every virtual detector (``detector_mask``);
``quantem.gpu.detector`` uses the same definitions, which the parity tests
check, so a viewer ROI and ``detector.bf`` / ``detector.adf`` are
pixel-identical. The bright-field disk is quantem.gpu's
``detector.fit_bright_field_disk``, the disk SSB reads. This module turns the
widget's ROI traits into masks and hands them to a detector session.
"""

import math

import numpy as np

from quantem.widget.adapters import gpu as gpu_adapter

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


def detector_reach_px(det_shape: tuple[int, int], center: tuple[float, float]) -> float:
    """Radius of the largest whole circle around ``center`` the detector holds: the distance to its nearest edge."""
    rows, cols = det_shape
    center_row, center_col = (float(value) for value in center)
    return max(0.0, min(center_row + 0.5, rows - 0.5 - center_row, center_col + 0.5, cols - 0.5 - center_col))


def preset_radii(name: str, det_shape: tuple[int, int], center: tuple[float, float],
                 bright_field_radius_px: float) -> tuple[float, float, float] | None:
    """``(inner, outer, nominal_outer)`` detector radii of a named preset, or None when the detector holds no ring.

    The annuli are multiples of the bright-field radius: ABF covers the outer
    half of the disk, ADF one to two radii, HAADF two to four radii. ADF and
    HAADF rings end at ``detector_reach_px``: beyond it a ring is cut at the
    detector edges and keeps only its corners. A ring narrower than one pixel
    there (HAADF of a disk wider than a quarter of the detector) is None.
    """
    inner, outer = (factor * max(1.0, float(bright_field_radius_px)) for factor in PRESET_RADII[name])
    if name in ("bf", "abf"):
        return inner, outer, outer
    reach = detector_reach_px(det_shape, center)
    if reach - inner < 1.0:
        return None
    return inner, min(outer, reach), outer


def preset_mask(name: str, det_shape: tuple[int, int], center: tuple[float, float],
                bright_field_radius_px: float) -> np.ndarray | None:
    """Detector mask of a named preset (BF, ABF, ADF, HAADF) around the fitted disk, or None (see ``preset_radii``).

    These are the radii ``Show4DSTEM.apply_preset`` writes into the ROI traits,
    so a preset click and its precomputed image use the same pixels.
    """
    radii = preset_radii(name, det_shape, center, bright_field_radius_px)
    return None if radii is None else detector_mask(center, radii[0], radii[1], det_shape)


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
    """Bright-field disk ``(center_row, center_col, edge_radius)`` from the mean pattern (``fit_bright_field_disk``).

    When no pixel stands above mean plus one standard deviation (a blank or
    uniform pattern) the viewer keeps the detector midpoint and
    ``default_ratio`` of the smaller detector side, which is what the untouched
    traits already hold.
    """
    pattern = np.asarray(mean_dp, dtype=np.float64)
    if not np.any(pattern > pattern.mean() + pattern.std()):
        return det_shape[0] / 2, det_shape[1] / 2, min(det_shape) * default_ratio
    (center_row, center_col), radius = fit_bright_field_disk(pattern)
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


def fit_bright_field_disk(mean_dp: np.ndarray) -> tuple[tuple[float, float], float]:
    """Bright-field disk ``((center_row, center_col), edge_radius)`` of a mean pattern, in detector pixels.

    quantem.gpu's ``detector.fit_bright_field_disk``: the centre is the
    centroid of the pixels above ``mean + std`` (population standard
    deviation) and the radius the one at which the disk falls to half its
    plateau, where the convergence semiangle sits. An install without
    quantem.gpu (no supported GPU torch) computes the same rule here in
    float64; ``tests/infrastructure/test_platform.py`` checks the two agree.
    """
    if gpu_adapter.available():
        return gpu_adapter.fit_bright_field_disk(mean_dp)
    pattern = np.asarray(mean_dp, dtype=np.float64)
    disk = pattern > pattern.mean() + pattern.std()
    if not disk.any():
        raise ValueError("The mean diffraction pattern has no bright-field disk: no pixel exceeds its mean plus one "
                         "standard deviation. Pass bright_field_center_px= and bright_field_radius_px=.")
    rows, cols = np.nonzero(disk)
    half_plateau = 0.5 * float(np.median(pattern[disk]))
    radius = math.sqrt(np.count_nonzero(pattern > half_plateau) / math.pi)
    return (float(rows.mean()), float(cols.mean())), radius
