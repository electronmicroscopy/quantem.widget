"""Diffraction-pattern analysis: center finding, radial and azimuthal profiles, peak and
ring fitting, spot and ring detection, k calibration, phase matching and zone-axis
indexing.

Every function takes plain numpy arrays and :class:`~quantem.widget.showdiffraction.phases.Phase`
objects and returns numbers, arrays or dicts; the widget supplies its frame, center and
ellipse state and stores the results in traits. :func:`radial_profile_px` also takes a
tensor and reduces it where it lives, because the widget re-measures it on every center
change.
"""

import math
import warnings
from collections.abc import Sequence

import numpy as np
import torch
from scipy import ndimage
from scipy.optimize import OptimizeWarning, curve_fit, linear_sum_assignment
from scipy.signal import find_peaks
from scipy.signal.windows import tukey

from quantem.widget.showdiffraction.phases import (
    Phase,
    format_hkl,
    format_zone_axis,
    label_preference,
)

RING_FIT_MODELS = ("gaussian", "pseudo_voigt")
# a ring peak must stand this many standard errors of the azimuthal mean above the profile envelope
RING_MIN_SNR = 5.0
# relative tolerance of the scale-free ratio test that verifies a phase calibration
RATIO_TOL = 0.01
# a second scale within this many nats (ln 100: 100x as likely by chance) makes a calibration ambiguous
AMBIGUITY_MARGIN = math.log(100.0)
# out-of-tolerance pad cost for the assignment problem
NO_MATCH_COST = 1.0e6
# Gaussian FWHM over sigma, 2 sqrt(2 ln 2)
FWHM_PER_SIGMA = 2.3548
# median absolute deviation to Gaussian sigma
MAD_TO_SIGMA = 1.4826


# --- Center estimation ---
def bandpass(frame: np.ndarray, mask: np.ndarray | None = None, log: bool = True) -> np.ndarray:
    """Zero-mean, high-passed (optionally log-compressed) frame for correlation.

    The log compresses the central beam so the weak outer reflections carry weight in
    the symmetry correlations; the Gaussian high-pass removes the diffuse background.
    """
    work = np.log1p(frame - frame.min()) if log else frame - frame.min()
    sigma = max(5.0, 0.02 * min(work.shape))
    work = work - ndimage.gaussian_filter(work, sigma=sigma)
    if mask is not None:
        work[mask] = 0.0
    return work - work.mean()


def parabolic_offset(values: np.ndarray, p: int, n: int) -> float:
    """Sub-sample peak offset from a parabola through ``values[p-1:p+2]`` (cyclic).

    The vertex of the parabola through ``(-1, lo)``, ``(0, values[p])``, ``(1, hi)`` lies at
    ``0.5 (hi - lo) / (2 values[p] - lo - hi)``. A stencil that is not a maximum, or a vertex
    more than one sample away, gives 0. Without it every correlation peak, and so every
    center and shift, would be quantized to whole pixels.
    """
    lo, hi = values[(p - 1) % n], values[(p + 1) % n]
    denom = 2.0 * values[p] - lo - hi
    if denom <= 0:
        return 0.0
    delta = 0.5 * (hi - lo) / denom
    return delta if abs(delta) <= 1.0 else 0.0


def phase_shift(ref: np.ndarray, moving: np.ndarray, upsample: int = 1) -> tuple[float, float, float]:
    """Shift of ``moving`` relative to ``ref`` by phase correlation.

    Returns ``(row_shift, col_shift, peak_to_sidelobe)``; the shifts are in the
    ``[0, n)`` convention of the correlation peak index. With ``upsample > 1`` the
    peak is refined on a local DFT grid (Guizar-Sicairos), otherwise by a parabola.
    """
    # tapered edges: the wrap-around discontinuity of the FFT would otherwise add a peak at zero shift
    window = tukey(ref.shape[0], 0.2)[:, None] * tukey(ref.shape[1], 0.2)[None, :]
    ref = ref * window
    moving = moving * window
    cross = np.fft.fft2(ref) * np.conj(np.fft.fft2(moving))
    cross = cross / np.maximum(np.abs(cross), 1e-12)
    f_row = np.fft.fftfreq(ref.shape[0])[:, None]
    f_col = np.fft.fftfreq(ref.shape[1])[None, :]
    # whitening gives noise-dominated high frequencies full weight; a Gaussian low-pass
    # (sigma 0.15 cycles/px) keeps them from breaking the peak into single-pixel spikes
    cross = cross * np.exp(-(f_row**2 + f_col**2) / (2.0 * 0.15**2))
    correlation = np.fft.ifft2(cross).real
    n_rows, n_cols = correlation.shape
    peak_row, peak_col = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    # peak-to-sidelobe ratio: peak height over the statistics outside a 5 px exclusion
    row_dist = np.abs(np.arange(n_rows) - peak_row)
    col_dist = np.abs(np.arange(n_cols) - peak_col)
    row_dist = np.minimum(row_dist, n_rows - row_dist)
    col_dist = np.minimum(col_dist, n_cols - col_dist)
    sidelobe = correlation[(row_dist[:, None] > 5.0) | (col_dist[None, :] > 5.0)]
    spread = sidelobe.std()
    psr = float((correlation[peak_row, peak_col] - sidelobe.mean()) / spread) if spread > 0 else 0.0
    if upsample <= 1:
        shift_row = peak_row + parabolic_offset(correlation[:, peak_col], peak_row, n_rows)
        shift_col = peak_col + parabolic_offset(correlation[peak_row, :], peak_col, n_cols)
        return float(shift_row), float(shift_col), psr
    offsets = np.arange(-int(np.ceil(1.5 * upsample)), int(np.ceil(1.5 * upsample)) + 1) / upsample
    rows = peak_row + offsets
    cols = peak_col + offsets
    e_row = np.exp(2j * np.pi * rows[:, None] * f_row.ravel()[None, :])
    e_col = np.exp(2j * np.pi * f_col.ravel()[:, None] * cols[None, :])
    local = (e_row @ cross @ e_col).real
    local_row, local_col = np.unravel_index(int(np.argmax(local)), local.shape)
    sub_row = parabolic_offset(local[:, local_col], local_row, local.shape[0]) / upsample
    sub_col = parabolic_offset(local[local_row, :], local_col, local.shape[1]) / upsample
    return float(rows[local_row] + sub_row), float(cols[local_col] + sub_col), psr


def center_symmetry(
    frame: np.ndarray,
    guess: tuple[float, float] | None = None,
    search_radius: float = 8.0,
    mask: np.ndarray | None = None,
) -> tuple[float, float]:
    """Refine a center guess by local Friedel-symmetry autocorrelation.

    The autocorrelation of a centrosymmetric pattern peaks at twice the center, so the
    peak nearest ``2 * guess`` (within ``2 * search_radius``) halves to the refined center.
    """
    frame = np.asarray(frame, dtype=np.float64)
    n_rows, n_cols = frame.shape
    if guess is None:
        guess = ((n_rows - 1) / 2.0, (n_cols - 1) / 2.0)
    work = bandpass(frame, mask)
    spectrum = np.fft.fft2(work)
    correlation = np.fft.ifft2(spectrum * spectrum).real
    target_row = (2.0 * guess[0]) % n_rows
    target_col = (2.0 * guess[1]) % n_cols
    row_idx = np.arange(n_rows, dtype=np.float64)
    col_idx = np.arange(n_cols, dtype=np.float64)
    row_dist = np.minimum(np.abs(row_idx - target_row), n_rows - np.abs(row_idx - target_row))
    col_dist = np.minimum(np.abs(col_idx - target_col), n_cols - np.abs(col_idx - target_col))
    near = (row_dist[:, None] <= 2.0 * search_radius) & (col_dist[None, :] <= 2.0 * search_radius)
    peak_row, peak_col = np.unravel_index(int(np.argmax(np.where(near, correlation, -np.inf))), correlation.shape)
    twice_row = peak_row + parabolic_offset(correlation[:, peak_col], peak_row, n_rows)
    twice_col = peak_col + parabolic_offset(correlation[peak_row, :], peak_col, n_cols)
    # the peak index is twice the center modulo n: of the two halves keep the one nearer the guess
    row = min(((twice_row + offset) / 2.0 for offset in (0.0, n_rows)), key=lambda center: abs(center - guess[0]))
    col = min(((twice_col + offset) / 2.0 for offset in (0.0, n_cols)), key=lambda center: abs(center - guess[1]))
    return float(row), float(col)


def symmetry_score(
    frame: np.ndarray, center: tuple[float, float], mask: np.ndarray | None = None
) -> float:
    """Normalized correlation between the band-passed frame and its inversion about
    ``center`` (Friedel symmetry), clipped to ``[0, 1]``."""
    frame = np.asarray(frame, dtype=np.float64)
    n_rows, n_cols = frame.shape
    work = bandpass(frame)
    rows, cols = np.indices((n_rows, n_cols), dtype=np.float64)
    rotated_rows = 2.0 * center[0] - rows
    rotated_cols = 2.0 * center[1] - cols
    valid = (
        (rotated_rows >= 0.0) & (rotated_rows <= n_rows - 1) & (rotated_cols >= 0.0) & (rotated_cols <= n_cols - 1)
    )
    rotated = ndimage.map_coordinates(work, [rotated_rows, rotated_cols], order=1, mode="nearest")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        # interpolated as a fraction: a pixel whose inversion lands mostly on masked pixels is dropped
        rotated_mask = ndimage.map_coordinates(
            mask.astype(np.float64), [rotated_rows, rotated_cols], order=1, mode="constant", cval=1.0
        )
        valid &= ~mask & (rotated_mask < 0.5)
    values = work[valid]
    rotated_values = rotated[valid]
    if values.size < 16:
        return 0.0
    values = values - values.mean()
    rotated_values = rotated_values - rotated_values.mean()
    norm = float(np.sqrt((values * values).sum() * (rotated_values * rotated_values).sum()))
    if norm <= 0:
        return 0.0
    return float(max(0.0, (values * rotated_values).sum() / norm))


def center_phase_correlation(
    frame: np.ndarray, mask: np.ndarray | None = None, upsample: int = 20
) -> tuple[float, float]:
    """Estimate the inversion center by phase correlation of the frame with its
    180-degree rotation; the wrap-around ambiguity is resolved by symmetry score.

    Rotating by 180 degrees maps a center ``c`` to ``n - 1 - c``, so the measured shift is
    ``2 c - (n - 1)`` and ``c = (n - 1 + shift) / 2`` per axis. The shift is known only
    modulo ``n``, which leaves two centers per axis; the pair whose inversion best matches
    the frame wins, ties going to the one nearest the detector middle.
    """
    frame = np.asarray(frame, dtype=np.float64)
    n_rows, n_cols = frame.shape
    work = bandpass(frame, mask)
    shift_row, shift_col, _ = phase_shift(work, work[::-1, ::-1], upsample=upsample)
    row_candidates = [(n_rows - 1 + shift) / 2.0 for shift in (shift_row % n_rows, shift_row % n_rows - n_rows)]
    col_candidates = [(n_cols - 1 + shift) / 2.0 for shift in (shift_col % n_cols, shift_col % n_cols - n_cols)]
    candidates = [
        (row, col)
        for row in row_candidates
        for col in col_candidates
        if 0.0 <= row <= n_rows - 1 and 0.0 <= col <= n_cols - 1
    ]
    # border candidates are wrap-around artefacts unless nothing else is left
    interior = [(row, col) for row, col in candidates if 1.0 <= row <= n_rows - 2 and 1.0 <= col <= n_cols - 2]
    if interior:
        candidates = interior
    if not candidates:
        candidates = [((n_rows - 1) / 2.0, (n_cols - 1) / 2.0)]
    scored = [(row, col, symmetry_score(frame, (row, col), mask=mask)) for row, col in candidates]
    best = max(score for _, _, score in scored)
    middle = ((n_rows - 1) / 2.0, (n_cols - 1) / 2.0)
    row, col, _ = min(
        (candidate for candidate in scored if candidate[2] >= best - 1e-4),
        key=lambda candidate: math.hypot(candidate[0] - middle[0], candidate[1] - middle[1]),
    )
    return float(row), float(col)


def pick_center(
    frame: np.ndarray,
    method: str = "auto",
    mask: np.ndarray | None = None,
    guess: tuple[float, float] | None = None,
    search_radius: float = 8.0,
) -> dict:
    """Estimate the pattern center with one method or an automatic pick.

    ``"auto"`` runs both estimators and keeps the one with the higher symmetry score.
    ``"symmetry"`` refines ``guess`` within ``search_radius``; ``"phase_corr"`` needs no
    guess. Returns ``{"row", "col", "method"}`` with the method that produced the center.
    """
    frame = np.asarray(frame, dtype=np.float64)
    if method == "symmetry":
        row, col = center_symmetry(frame, guess=guess, search_radius=search_radius, mask=mask)
        name = "symmetry"
    elif method == "phase_corr":
        row, col = center_phase_correlation(frame, mask=mask)
        name = "phase_corr"
    elif method == "auto":
        phase_row, phase_col = center_phase_correlation(frame, mask=mask)
        symmetry_row, symmetry_col = center_symmetry(frame, guess=guess, search_radius=search_radius, mask=mask)
        candidates = [
            ("phase_corr", phase_row, phase_col, symmetry_score(frame, (phase_row, phase_col), mask=mask)),
            ("symmetry", symmetry_row, symmetry_col, symmetry_score(frame, (symmetry_row, symmetry_col), mask=mask)),
        ]
        name, row, col, _ = max(candidates, key=lambda candidate: candidate[3])
    else:
        raise ValueError(f"unknown method {method!r}; use auto, symmetry, or phase_corr")
    return {"row": float(row), "col": float(col), "method": name}


def central_beam_radius(mask: np.ndarray, center_row: float, center_col: float) -> float:
    """Equivalent-disk radius of the connected bright component under the center.

    A beam stop can leave the center pixel dark, in which case the nearest component is
    taken. Returns 0 when the mask is empty. The radius is ``sqrt(area / pi)`` of that
    component.
    """
    labels, n_labels = ndimage.label(mask)
    if n_labels == 0:
        return 0.0
    row_idx = int(min(max(round(center_row), 0), mask.shape[0] - 1))
    col_idx = int(min(max(round(center_col), 0), mask.shape[1] - 1))
    central_label = int(labels[row_idx, col_idx])
    if central_label == 0:
        component_rows, component_cols = np.nonzero(labels)
        nearest = int(np.argmin((component_rows - center_row) ** 2 + (component_cols - center_col) ** 2))
        central_label = int(labels[component_rows[nearest], component_cols[nearest]])
    area = float((labels == central_label).sum())
    return float(np.sqrt(area / np.pi))


def align_frames(
    frames: np.ndarray,
    reference: np.ndarray | None = None,
    max_shift: float = 8.0,
) -> tuple[np.ndarray, list[tuple[float, float]], list[bool]]:
    """Align a stack of patterns to the first (or ``reference``) by subpixel phase
    correlation. Frames whose shift exceeds ``max_shift`` or whose correlation peak is
    weak are passed through unshifted and flagged ``False`` in the third return."""
    frames = np.asarray(frames, dtype=np.float64)
    n_frames, n_rows, n_cols = frames.shape
    reference = frames[0] if reference is None else np.asarray(reference, dtype=np.float64)
    # linear bandpass keeps count statistics for the correlation
    reference_work = bandpass(reference, log=False)
    aligned = np.empty_like(frames)
    shifts: list[tuple[float, float]] = []
    used: list[bool] = []
    for i in range(n_frames):
        shift_row, shift_col, psr = phase_shift(reference_work, bandpass(frames[i], log=False))
        # peak index in [0, n) to a signed shift
        shift_row = shift_row % n_rows
        shift_row = shift_row - n_rows if shift_row > n_rows / 2 else shift_row
        shift_col = shift_col % n_cols
        shift_col = shift_col - n_cols if shift_col > n_cols / 2 else shift_col
        # quality 0.2 is a peak-to-sidelobe ratio of 2.5: below it the peak may be noise
        peak_quality = psr / (psr + 10.0) if psr > 0 else 0.0
        usable = np.hypot(shift_row, shift_col) <= max_shift and peak_quality >= 0.2
        shifts.append((float(shift_row), float(shift_col)))
        used.append(bool(usable))
        aligned[i] = ndimage.shift(frames[i], (shift_row, shift_col), order=1) if usable else frames[i]
    return aligned, shifts, used


def ring_uniformity(
    frame: np.ndarray,
    center: tuple[float, float],
    radius: float,
    half_width: float = 4.0,
    n_theta: int = 180,
    mask: np.ndarray | None = None,
) -> dict:
    """Azimuthal uniformity QC for one ring: coefficient of variation, fraction of
    sectors above half the median and mean/std SNR; ``mask`` pixels are excluded."""
    frame = np.asarray(frame, dtype=np.float64)
    rows, cols = np.indices(frame.shape, dtype=np.float64)
    d_row = rows - center[0]
    d_col = cols - center[1]
    annulus = np.abs(np.hypot(d_row, d_col) - radius) <= half_width
    if mask is not None:
        annulus &= ~np.asarray(mask, dtype=bool)
    theta = np.arctan2(d_row[annulus], d_col[annulus])
    sector = np.clip(((theta + np.pi) / (2.0 * np.pi) * n_theta).astype(int), 0, n_theta - 1)
    sums = np.bincount(sector, weights=frame[annulus], minlength=n_theta)
    counts = np.bincount(sector, minlength=n_theta)
    live = counts > 0
    if not live.any():
        return {"cv": 0.0, "coverage": 0.0, "snr": 0.0}
    profile = sums[live] / counts[live]
    mean = float(profile.mean())
    std = float(profile.std())
    cv = std / mean if mean > 0 else 0.0
    positive = profile[profile > 0]
    coverage = float(np.mean(profile > 0.5 * np.median(positive))) if positive.size else 0.0
    snr = min(mean / std, 999.0) if std > 0 else 999.0
    return {"cv": float(cv), "coverage": float(coverage), "snr": float(snr)}


# --- Geometry, masks and profiles ---
def build_analysis_mask(
    shape: tuple[int, int],
    regions: list[dict],
    center: tuple[float, float],
) -> np.ndarray | None:
    """Boolean exclusion mask from disk (``row col radius``) and wedge
    (``start_deg end_deg`` about ``center``) regions; None when there are none."""
    if not regions:
        return None
    n_rows, n_cols = shape
    rows = np.arange(n_rows, dtype=np.float64)[:, None]
    cols = np.arange(n_cols, dtype=np.float64)[None, :]
    center_row, center_col = center
    mask = np.zeros((n_rows, n_cols), dtype=bool)
    for region in regions:
        kind = region.get("kind")
        if kind == "disk":
            mask |= np.hypot(rows - region["row"], cols - region["col"]) <= region["radius"]
        elif kind == "wedge":
            # full-circle span: every angle masked
            if abs(float(region["end_deg"]) - float(region["start_deg"])) >= 360.0:
                mask[:] = True
                continue
            theta = np.degrees(np.arctan2(rows - center_row, cols - center_col)) % 360.0
            mask |= in_wedge(theta, region["start_deg"], region["end_deg"])
    return mask


def in_wedge(theta_deg: np.ndarray, start_deg: float, end_deg: float) -> np.ndarray:
    """Pixels whose azimuth ``theta_deg`` (in [0, 360)) lies between ``start_deg`` and ``end_deg``.

    Both ends are taken modulo 360 and the wedge runs through increasing angle from start to
    end, so a wedge across 0 degrees (start 350, end 10) wraps instead of selecting its
    complement. The analysis mask and the sector profile share it so a wedge drawn in the
    browser excludes exactly the pixels the profile leaves out.
    """
    start, end = float(start_deg) % 360.0, float(end_deg) % 360.0
    if start <= end:
        return (theta_deg >= start) & (theta_deg <= end)
    return (theta_deg >= start) | (theta_deg <= end)


def corrected_radius(
    d_row,
    d_col,
    *,
    ellipse_ratio: float = 1.0,
    ellipse_angle: float = 0.0,
    ellipse_corrected: bool = False,
):
    """Radius with optional elliptical-distortion correction.

    The correction is mean-preserving: an ellipse of semi-axes A, B maps to a
    circle of radius sqrt(A*B) (the mean radius), so a calibration set before
    the correction stays valid after it.
    """
    if not ellipse_corrected or ellipse_ratio == 1.0:
        return np.hypot(d_row, d_col)
    return np.hypot(*ellipse_to_circle(d_row, d_col, ellipse_ratio, ellipse_angle))


def corrected_vector(
    d_row,
    d_col,
    *,
    ellipse_ratio: float = 1.0,
    ellipse_angle: float = 0.0,
    ellipse_corrected: bool = False,
):
    """Displacement vector under the same ellipse transform as :func:`corrected_radius`."""
    if not ellipse_corrected or ellipse_ratio == 1.0:
        return d_row, d_col
    major, minor = ellipse_to_circle(d_row, d_col, ellipse_ratio, ellipse_angle)
    angle = math.radians(ellipse_angle)
    return (
        major * math.sin(angle) + minor * math.cos(angle),
        major * math.cos(angle) - minor * math.sin(angle),
    )


def ellipse_to_circle(d_row, d_col, ellipse_ratio: float, ellipse_angle: float):
    """Components of a displacement along the ellipse's major and minor axes, scaled onto its
    mean-radius circle.

    The major axis lies ``ellipse_angle`` degrees from the column axis. Dividing the major
    component by sqrt(ratio) and multiplying the minor one by it maps semi-axes A and B both
    to sqrt(A*B), so :func:`corrected_radius` and :func:`corrected_vector` undo the
    distortion without changing the mean radius a calibration was set on.
    """
    angle = math.radians(ellipse_angle)
    major = d_col * math.cos(angle) + d_row * math.sin(angle)
    minor = -d_col * math.sin(angle) + d_row * math.cos(angle)
    root_ratio = math.sqrt(ellipse_ratio)
    return major / root_ratio, minor * root_ratio


def radial_profile_px(
    frame,
    *,
    center: tuple[float, float],
    n_bins: int | None = None,
    max_radius: float | None = None,
    mask: np.ndarray | None = None,
    angular_range: tuple[float, float] | None = None,
    ellipse_ratio: float = 1.0,
    ellipse_angle: float = 0.0,
    ellipse_corrected: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Azimuthally averaged intensity vs radius in detector pixels, with its standard error.

    Returns ``(radii, intensity, standard_error)``. Bins are one pixel wide out to the
    nearest detector edge unless ``n_bins`` / ``max_radius`` say otherwise;
    ``angular_range`` restricts the average to a wedge. The standard error of a bin is
    ``sqrt(s^2 / n)`` with ``s^2`` the sample variance of its ``n`` pixels (infinite
    below two pixels): the uncertainty of the azimuthal mean, which ring detection
    needs to tell a ring from a fluctuation. Azimuthal texture and the radial slope
    inside a bin only enlarge it, so significance against it is conservative.

    ``frame`` is a NumPy array or a tensor (``(rows, cols)``). A tensor is reduced on its
    device, so a 2048 x 2048 frame on CUDA is profiled in milliseconds without a copy
    back; Apple MPS has no float64, so an MPS frame is reduced on the CPU. Only the
    per-bin sums come back to finish the mean and the standard error.
    """
    frame = torch.as_tensor(frame)
    if frame.device.type == "mps":
        # the sums of squares need float64, which Apple MPS lacks
        frame = frame.cpu()
    device = frame.device
    n_rows, n_cols = frame.shape
    center_row, center_col = float(center[0]), float(center[1])
    if max_radius is None:
        max_radius = float(
            min(center_row, center_col, (n_rows - 1) - center_row, (n_cols - 1) - center_col)
        )
    max_radius = float(max(1.0, max_radius))
    n_bins = max(1, round(max_radius)) if n_bins is None else int(max(1, n_bins))
    d_row = torch.arange(n_rows, dtype=torch.float64, device=device)[:, None] - center_row
    d_col = torch.arange(n_cols, dtype=torch.float64, device=device)[None, :] - center_col
    # the mean-preserving ellipse correction of corrected_radius; the wedge stays in detector angles
    radial_row, radial_col = (
        ellipse_to_circle(d_row, d_col, ellipse_ratio, ellipse_angle)
        if ellipse_corrected and ellipse_ratio != 1.0
        else (d_row, d_col)
    )
    # sqrt of the squared distance, not hypot: CUDA's hypot misses exact radii such as
    # hypot(54, 240) = 246 by one ulp, which moves pixels on an integer bin edge into the bin below
    radii = (radial_row.square() + radial_col.square()).sqrt()
    edges = np.linspace(0.0, max_radius, n_bins + 1)
    # np.digitize(radii, edges) - 1: bin n_bins collects everything at or past the last edge
    bins = torch.bucketize(radii, torch.from_numpy(edges).to(device), right=True) - 1
    keep = None if mask is None else ~torch.as_tensor(mask, device=device)
    if angular_range is not None:
        start_deg, end_deg = (float(angle) for angle in angular_range)
        # a full-circle span, or one whose ends coincide modulo 360, is no angular restriction
        if abs(end_deg - start_deg) < 360.0 and start_deg % 360.0 != end_deg % 360.0:
            wedge = in_wedge(torch.rad2deg(torch.atan2(d_row, d_col)) % 360.0, start_deg, end_deg)
            keep = wedge if keep is None else keep & wedge
    if keep is not None:
        # excluded pixels join the overflow bin instead of being compacted out: same sums, no gather
        bins = torch.where(keep, bins, n_bins)
    bins = bins.flatten()
    # float64 sums of squares: in float32 the variance of a bright bin cancels to noise
    values = frame.to(torch.float64).flatten()
    counts, sums, sums_sq = (
        torch.bincount(bins, weights=weights, minlength=n_bins + 1)[:n_bins].cpu().numpy()
        for weights in (None, values, values * values)
    )
    counts = counts.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        intensity = np.where(counts > 0, sums / counts, 0.0)
        variance = np.maximum(sums_sq - counts * intensity**2, 0.0) / (counts - 1.0)
        standard_error = np.where(counts > 1, np.sqrt(variance / counts), np.inf)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    return bin_centers.astype(np.float32), intensity.astype(np.float32), standard_error


def ring_sectors(
    frame: np.ndarray,
    *,
    center: tuple[float, float],
    radius_px: float,
    half_width: float,
    n_theta: int,
    mask: np.ndarray | None = None,
    use_corrected_radius: bool = True,
    ellipse_ratio: float = 1.0,
    ellipse_angle: float = 0.0,
    ellipse_corrected: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-sector pixel counts, intensity sums, pedestal-subtracted weights and
    weighted radii around one ring (``theta, counts, intensity_sum, weight_sum,
    weighted_radius_sum``); the inputs of the azimuthal profile and the ellipse fit.

    Only the annulus' bounding box is read: a corrected radius ``r`` lies at most
    ``r * max(sqrt(ratio), 1 / sqrt(ratio))`` from the center on the detector, so the box
    holds every selected pixel in the same row-major order and the sums are unchanged,
    while a ring well inside a 2048 x 2048 frame reads a fraction of it.
    """
    center_row, center_col = center
    n_rows, n_cols = frame.shape
    stretch = math.sqrt(max(ellipse_ratio, 1.0 / ellipse_ratio)) if use_corrected_radius and ellipse_corrected else 1.0
    reach = (radius_px + half_width) * stretch + 1.0
    row_start, row_stop = max(0, math.floor(center_row - reach)), min(n_rows, math.ceil(center_row + reach) + 1)
    col_start, col_stop = max(0, math.floor(center_col - reach)), min(n_cols, math.ceil(center_col + reach) + 1)
    frame = frame[row_start:row_stop, col_start:col_stop].astype(np.float64)
    if mask is not None:
        mask = mask[row_start:row_stop, col_start:col_stop]
    rows = np.arange(row_start, row_stop, dtype=np.float64)[:, None]
    cols = np.arange(col_start, col_stop, dtype=np.float64)[None, :]
    d_row, d_col = rows - center_row, cols - center_col
    if use_corrected_radius:
        radii = corrected_radius(
            d_row,
            d_col,
            ellipse_ratio=ellipse_ratio,
            ellipse_angle=ellipse_angle,
            ellipse_corrected=ellipse_corrected,
        )
    else:
        radii = np.hypot(d_row, d_col)
    theta_centers = (np.arange(n_theta) + 0.5) * (360.0 / n_theta)
    selected = np.abs(radii - radius_px) <= half_width
    if mask is not None:
        selected &= ~mask
    if not selected.any():
        zero = np.zeros(n_theta)
        return theta_centers, zero.copy(), zero.copy(), zero.copy(), zero.copy()
    theta = np.degrees(np.arctan2(d_row, d_col)) % 360.0
    sector = np.minimum((theta[selected] / (360.0 / n_theta)).astype(int), n_theta - 1)
    intensity = frame[selected]
    # median pedestal, negatives clipped: the ring, not the background, weights the radius
    weight = np.clip(intensity - np.median(intensity), 0.0, None)
    counts = np.bincount(sector, minlength=n_theta).astype(np.float64)
    intensity_sum = np.bincount(sector, weights=intensity, minlength=n_theta)
    weight_sum = np.bincount(sector, weights=weight, minlength=n_theta)
    weighted_radius_sum = np.bincount(sector, weights=weight * radii[selected], minlength=n_theta)
    return theta_centers, counts, intensity_sum, weight_sum, weighted_radius_sum


def azimuthal_profile_from_frame(
    frame: np.ndarray,
    *,
    center: tuple[float, float],
    radius_px: float,
    half_width: float,
    n_theta: int,
    mask: np.ndarray | None = None,
    ellipse_ratio: float = 1.0,
    ellipse_angle: float = 0.0,
    ellipse_corrected: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthal intensity profile I(theta) around one ring."""
    theta, counts, intensity_sum, _, _ = ring_sectors(
        frame,
        center=center,
        radius_px=radius_px,
        half_width=half_width,
        n_theta=n_theta,
        mask=mask,
        use_corrected_radius=True,
        ellipse_ratio=ellipse_ratio,
        ellipse_angle=ellipse_angle,
        ellipse_corrected=ellipse_corrected,
    )
    intensity = np.where(counts > 0, intensity_sum / np.maximum(counts, 1.0), 0.0)
    return theta.astype(np.float32), intensity.astype(np.float32)


def texture_from_profile(theta_deg: np.ndarray, intensity: np.ndarray) -> dict:
    """Order-2 texture of an azimuthal profile: ``strength`` in [0, 1] (second-harmonic
    amplitude over the mean) and the preferred ``angle_deg`` in [0, 180); zero unless at
    least 8 sectors spanning 90 degrees are covered."""
    values = intensity.astype(np.float64)
    covered = values != 0.0
    span = covered_arc_deg(theta_deg, covered) if covered.sum() >= 8 else 0.0
    if span < 90.0:
        strength, angle = 0.0, 0.0
    else:
        theta = np.radians(theta_deg)[covered]
        design = np.column_stack([np.ones_like(theta), np.cos(2 * theta), np.sin(2 * theta)])
        (mean_level, cosine, sine), *_ = np.linalg.lstsq(design, values[covered], rcond=None)
        if mean_level <= 0:
            strength, angle = 0.0, 0.0
        else:
            strength = float(min(1.0, math.hypot(cosine, sine) / mean_level))
            angle = float(math.degrees(math.atan2(sine, cosine)) / 2.0 % 180.0)
    return {"strength": strength, "angle_deg": angle, "coverage_deg": span}


def covered_arc_deg(theta_deg: np.ndarray, covered: np.ndarray) -> float:
    """Angular extent in degrees of the ``covered`` sectors: 360 minus the widest gap between
    neighbouring covered sector angles (cyclic).

    A harmonic fit of I(theta) or r(theta) over a short arc is ill-conditioned, so the texture
    and ellipse fits refuse arcs below a minimum span instead of reporting a spurious result.
    """
    live = np.sort(np.asarray(theta_deg, dtype=np.float64)[covered])
    gaps = np.diff(np.concatenate([live, live[:1] + 360.0]))
    return 360.0 - float(gaps.max())


def fit_ellipse_from_sectors(
    theta_centers: np.ndarray,
    counts: np.ndarray,
    weight_sum: np.ndarray,
    weighted_radius_sum: np.ndarray,
) -> dict:
    """Ellipse ratio and angle from ring-sector radii.

    r(theta) is fitted with order-1 terms (which absorb residual center error) and
    order-2 terms whose amplitude over the mean radius gives the eccentricity.
    """
    valid = (counts >= 10) & (weight_sum > 0)
    if valid.sum() < 8:
        raise ValueError(
            f"could not fit ellipse: ring found in {int(valid.sum())} sectors, need >= 8; "
            "check the ring radius and center"
        )
    # short arcs leave the order-1/order-2 harmonics collinear
    span = covered_arc_deg(theta_centers, valid)
    if span < 120.0:
        raise ValueError(
            f"could not fit ellipse: ring sectors span only {span:.0f} deg, need >= 120; "
            "reduce the excluded regions or pick a fuller ring"
        )
    radii_by_theta = weighted_radius_sum[valid] / weight_sum[valid]
    theta = np.radians(theta_centers)[valid]
    design = np.column_stack(
        [np.ones_like(theta), np.cos(theta), np.sin(theta), np.cos(2 * theta), np.sin(2 * theta)]
    )
    solution, *_ = np.linalg.lstsq(design, radii_by_theta, rcond=None)
    mean_radius, cosine, sine = solution[0], solution[3], solution[4]
    epsilon = math.hypot(cosine, sine) / mean_radius
    ratio = (1.0 + epsilon) / (1.0 - epsilon) if epsilon < 1.0 else float("inf")
    angle = (0.5 * math.degrees(math.atan2(sine, cosine))) % 180.0
    residual = radii_by_theta - design @ solution
    return {
        "ratio": float(ratio),
        "angle_deg": float(angle),
        "r_mean": float(mean_radius),
        "residual_px": float(np.sqrt(np.mean(residual**2))),
        "n_sectors": int(valid.sum()),
    }


def fit_radial_background(
    radii_px: np.ndarray,
    intensity: np.ndarray,
    *,
    peak_windows: list[tuple[float, float]],
    exclude_radius: float,
    method: str = "power",
    poly_order: int = 3,
) -> np.ndarray:
    """Smooth background under a radial profile (power law or polynomial) fitted to the
    bins outside the central beam and the ``peak_windows``."""
    if method not in ("power", "poly"):
        raise ValueError(f"method must be 'power' or 'poly', got {method!r}")
    if poly_order < 0:
        raise ValueError(f"poly_order must be non-negative, got {poly_order}")
    radii = radii_px.astype(np.float64)
    values = intensity.astype(np.float64)
    keep = radii > float(exclude_radius)
    for lo, hi in peak_windows:
        keep &= ~((radii >= lo) & (radii <= hi))
    if method == "power":
        keep &= values > 0
    min_points = 2 if method == "power" else max(2, poly_order + 1)
    if keep.sum() < min_points:
        raise ValueError(
            "not enough background bins to fit; widen the profile or narrow peak_windows"
        )
    if method == "power":
        coefficients = np.polyfit(np.log(radii[keep]), np.log(values[keep]), 1)
        positive_radii = radii[radii > 0]
        eval_radii = np.maximum(radii, positive_radii.min())
        background = np.exp(np.polyval(coefficients, np.log(eval_radii)))
    else:
        coefficients = np.polyfit(radii[keep], values[keep], poly_order)
        background = np.polyval(coefficients, radii)
    return background.astype(np.float32)


# --- Peak fitting ---
def fit_gaussian_spot(frame: np.ndarray, row: float, col: float, *, half_window: int) -> dict | None:
    """Subpixel 2D Gaussian fit around a spot; None when the patch is too small, the
    fit fails or the fitted center leaves the patch."""
    frame = np.asarray(frame, dtype=np.float32)
    half_size = max(4, int(half_window))
    pick_row, pick_col = round(row), round(col)
    row_lo, row_hi = max(0, pick_row - half_size), min(frame.shape[0], pick_row + half_size + 1)
    col_lo, col_hi = max(0, pick_col - half_size), min(frame.shape[1], pick_col + half_size + 1)
    # the fit and its R^2 sums run in float64
    patch = frame[row_lo:row_hi, col_lo:col_hi].astype(np.float64)
    if patch.shape[0] < 5 or patch.shape[1] < 5:
        return None
    n_rows, n_cols = patch.shape
    row_grid, col_grid = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")

    def gaussian_2d(coords, amplitude, row_center, col_center, sigma_row, sigma_col, offset):
        """Axis-aligned 2D Gaussian plus offset, flattened for ``curve_fit``:
        ``A exp(-((r - r0)^2 / s_r^2 + (c - c0)^2 / s_c^2) / 2) + offset``."""
        rows, cols = coords
        exponent = ((rows - row_center) / sigma_row) ** 2
        exponent += ((cols - col_center) / sigma_col) ** 2
        return (amplitude * np.exp(-0.5 * exponent) + offset).ravel()

    peak = np.unravel_index(int(np.argmax(patch)), patch.shape)
    initial = (
        float(patch.max() - patch.min()),
        float(peak[0]),
        float(peak[1]),
        2.0,
        2.0,
        float(patch.min()),
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            fit_params, covariance = curve_fit(
                gaussian_2d, (row_grid, col_grid), patch.ravel(), p0=initial, maxfev=5000
            )
    except (RuntimeError, ValueError):
        return None
    _, fit_row, fit_col, sigma_row, sigma_col, _ = fit_params
    if not (0 <= fit_row < n_rows and 0 <= fit_col < n_cols):
        return None
    parameter_errors = np.sqrt(np.abs(np.diag(covariance)))
    residual = patch.ravel() - gaussian_2d((row_grid, col_grid), *fit_params)
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((patch.ravel() - patch.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "row": float(row_lo + fit_row),
        "col": float(col_lo + fit_col),
        "row_err": float(parameter_errors[1]) if np.isfinite(parameter_errors[1]) else 0.0,
        "col_err": float(parameter_errors[2]) if np.isfinite(parameter_errors[2]) else 0.0,
        "sigma_row": float(abs(sigma_row)),
        "sigma_col": float(abs(sigma_col)),
        "fit_quality": float(r_squared),
    }


def gaussian_peak(radius, amplitude, center, sigma, offset):
    """Radial ring peak ``A exp(-(r - r0)^2 / (2 sigma^2)) + offset``, the default
    :func:`fit_ring_peaks` model."""
    return amplitude * np.exp(-0.5 * ((radius - center) / sigma) ** 2) + offset


def pseudo_voigt_peak(radius, amplitude, center, sigma, offset, eta):
    """Pseudo-Voigt ring peak ``A (eta L + (1 - eta) G) + offset`` with a Lorentzian ``L`` and
    a Gaussian ``G`` of the same FWHM (``gamma`` is the Lorentzian half width).

    Size and strain broadening give ring profiles longer tails than a Gaussian; the mixing
    fraction ``eta`` lets :func:`fit_ring_peaks` follow them.
    """
    gamma = sigma * FWHM_PER_SIGMA / 2.0
    lorentzian = 1.0 / (1.0 + ((radius - center) / gamma) ** 2)
    gaussian_part = np.exp(-0.5 * ((radius - center) / sigma) ** 2)
    return amplitude * (eta * lorentzian + (1.0 - eta) * gaussian_part) + offset


def fit_ring_peaks(
    radii_px: np.ndarray,
    intensity: np.ndarray,
    rings,
    *,
    model: str = "gaussian",
    window: float | None = None,
) -> list[dict | None]:
    """Fit one radial peak per ring record (``radius_px``) and return the refined radius,
    amplitude, FWHM, integrated intensity and R^2, or None where the fit fails.

    The fit window defaults to half the gap to the nearest other ring so neighbouring
    rings do not pull the peak.
    """
    if model not in RING_FIT_MODELS:
        raise ValueError(f"model must be one of {RING_FIT_MODELS}, got {model!r}")
    if window is not None and window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    peak_model = gaussian_peak if model == "gaussian" else pseudo_voigt_peak
    centers = sorted(float(ring["radius_px"]) for ring in rings)
    updates = []
    for ring in rings:
        radius_guess = float(ring["radius_px"])
        if window is not None:
            half_width = float(window)
        else:
            gaps = [abs(radius_guess - center) for center in centers if center != radius_guess]
            half_width = max(3.0, min(gaps) / 2.0) if gaps else max(6.0, 0.2 * radius_guess)
        in_window = (radii_px >= radius_guess - half_width) & (radii_px <= radius_guess + half_width)
        radius_window = radii_px[in_window].astype(np.float64)
        intensity_window = intensity[in_window].astype(np.float64)
        if radius_window.size < 5:
            updates.append(None)
            continue
        initial = [
            max(float(intensity_window.max() - intensity_window.min()), 1e-6),
            radius_guess,
            2.0,
            float(intensity_window.min()),
        ]
        bounds = (
            [0.0, radius_guess - half_width, 0.1, -np.inf],
            [np.inf, radius_guess + half_width, half_width, np.inf],
        )
        if model == "pseudo_voigt":
            initial = initial + [0.5]
            bounds = (bounds[0] + [0.0], bounds[1] + [1.0])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                fit_params, _ = curve_fit(
                    peak_model, radius_window, intensity_window, p0=initial, bounds=bounds, maxfev=5000
                )
        except (RuntimeError, ValueError):
            updates.append(None)
            continue
        residual = intensity_window - peak_model(radius_window, *fit_params)
        ss_tot = float(np.sum((intensity_window - intensity_window.mean()) ** 2))
        r_squared = 1.0 - float(np.sum(residual**2)) / ss_tot if ss_tot > 0 else 0.0
        amplitude = float(fit_params[0])
        sigma = float(abs(fit_params[2]))
        if model == "pseudo_voigt":
            eta = float(fit_params[4])
            gamma = sigma * FWHM_PER_SIGMA / 2.0
            integrated = amplitude * (
                eta * math.pi * gamma + (1.0 - eta) * sigma * math.sqrt(2.0 * math.pi)
            )
        else:
            integrated = amplitude * sigma * math.sqrt(2.0 * math.pi)
        updates.append(
            {
                "raw_radius_px": float(radius_guess),
                "radius_px": float(fit_params[1]),
                "intensity": amplitude,
                "fwhm_px": FWHM_PER_SIGMA * sigma,
                "intensity_integrated": integrated,
                "fit_quality": float(r_squared),
            }
        )
    return updates


# --- Detection ---
def pick_detect_denoise(frame: np.ndarray) -> str:
    """Choose the detection denoise for a frame: ``"anscombe"`` for sparse counting data
    (integer-valued counts with a low median), ``"gaussian"`` for continuous data whose
    peaks are weak against the robust noise estimate, ``"none"`` otherwise."""
    frame = np.asarray(frame, dtype=np.float64)
    positive = frame[frame > 0]
    if positive.size == 0:
        return "none"
    counts = positive / float(positive.min())
    if np.allclose(counts, np.round(counts), atol=1e-3):
        return "anscombe" if float(np.median(counts)) <= 30.0 else "none"
    residual = frame - ndimage.median_filter(frame, size=3)
    noise = MAD_TO_SIGMA * float(np.median(np.abs(residual)))
    if noise <= 0.0:
        return "none"
    signal = float(np.percentile(frame, 99.5) - np.median(frame))
    return "gaussian" if signal < 50.0 * noise else "none"


def detect_spot_coords(
    frame: np.ndarray,
    *,
    center: tuple[float, float],
    exclude_radius: float,
    min_distance: int,
    min_relative: float,
    noise_sigma: float,
    mask: np.ndarray | None = None,
    max_spots: int | None = None,
) -> list[tuple[int, int]]:
    """Integer (row, col) of isolated Bragg peaks, strongest first.

    Local maxima of the log, high-passed frame outside ``exclude_radius`` are kept when
    their contrast reaches ``min_relative`` of the strongest peak and ``noise_sigma``
    robust sigmas, when they stand clear of a ring crest (the minimum over opposite
    samples on a circle of ``min_distance`` must drop by half) and when no stronger kept
    peak lies within ``min_distance``.
    """
    frame = np.asarray(frame, dtype=np.float64)
    n_rows, n_cols = frame.shape
    work = np.log1p(np.clip(frame - frame.min(), 0.0, None))
    work = work - ndimage.gaussian_filter(work, sigma=max(2.0, float(min_distance)))
    window = max(3, int(min_distance) | 1)  # odd window
    local_max = ndimage.maximum_filter(work, size=window) == work
    rows = np.arange(n_rows)[:, None]
    cols = np.arange(n_cols)[None, :]
    radius = np.hypot(rows - center[0], cols - center[1])
    local_max &= radius > float(exclude_radius)
    if mask is not None:
        local_max &= ~mask
    local_max[0, :] = local_max[-1, :] = False
    local_max[:, 0] = local_max[:, -1] = False
    coords = np.argwhere(local_max)
    if coords.size == 0:
        return []
    prominence = work[coords[:, 0], coords[:, 1]]
    contrast = np.expm1(prominence)
    noise = MAD_TO_SIGMA * float(np.median(np.abs(work - np.median(work))))
    level = max(min_relative * float(contrast.max()), float(np.expm1(noise_sigma * noise)))
    keep = (prominence > 0) & (contrast >= level)
    coords, prominence = coords[keep], prominence[keep]
    if coords.size:
        angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
        ring_radius = float(max(3, min_distance))
        # truncated to pixel indices of 16 samples on a circle around each peak
        ring_rows = np.clip(coords[:, :1] + ring_radius * np.sin(angles), 0, n_rows - 1).astype(int)
        ring_cols = np.clip(coords[:, 1:] + ring_radius * np.cos(angles), 0, n_cols - 1).astype(int)
        ring_values = work[ring_rows, ring_cols]
        # on a ring crest one pair of opposite samples stays high; around an isolated peak all drop
        crest = np.minimum(ring_values[:, :8], ring_values[:, 8:]).max(axis=1)
        floor = np.percentile(ring_values, 10, axis=1)
        isolated = (prominence - crest) >= 0.5 * np.maximum(prominence - floor, 1e-9)
        coords, prominence = coords[isolated], prominence[isolated]
    kept: list[int] = []
    for idx in np.argsort(-prominence):
        peak_row, peak_col = coords[idx]
        if all(np.hypot(peak_row - coords[j][0], peak_col - coords[j][1]) > min_distance for j in kept):
            kept.append(int(idx))
    if max_spots is not None:
        kept = kept[: int(max_spots)]
    return [(int(peak_row), int(peak_col)) for peak_row, peak_col in coords[kept]]


def detect_ring_radii(
    radii_px: np.ndarray,
    intensity: np.ndarray,
    standard_error: np.ndarray,
    *,
    exclude_radius: float,
    min_snr: float = RING_MIN_SNR,
    min_separation: int = 5,
) -> list[float]:
    """Radii of every significant ring peak outside ``exclude_radius``, most significant first.

    The profile is smoothed by one bin (a Gaussian, so bin-to-bin noise does not split
    or fake a peak), and a ring is a local maximum above the profile's lower envelope
    (a grey opening over ``radial_envelope_width`` bins: the largest curve under the
    profile that no feature narrower than the window lifts) whose prominence is at
    least ``min_snr`` times ``standard_error`` at the peak. Pass the standard error of
    the raw frame's azimuthal mean: a denoised ``intensity`` then only locates peaks
    and significance stays tied to the measured counts. Callers keep the first N for a
    "max rings" budget.

    Why an envelope and not a smoothed trend: subtracting a Gaussian-smoothed profile
    (the earlier detector) leaves negative lobes on both sides of every strong feature,
    so the flat background between the central beam and the first ring became a
    "peak" with a large prominence and took a ring slot. A monotone tail (the central
    beam) or a flat background is unchanged by the opening, so it leaves no residual.
    Noise peaks measured this way stay below 3 standard errors on synthetic Poisson
    patterns from 0.5 to 10^4 counts per pixel (docs/2026-10-08-showdiffraction-auto-index.md).
    """
    smoothed = ndimage.gaussian_filter1d(np.asarray(intensity, dtype=np.float64), 1.0)
    if smoothed.size < 5:
        return []
    residual = smoothed - ndimage.grey_opening(smoothed, size=radial_envelope_width(smoothed.size), mode="nearest")
    peaks, properties = find_peaks(residual, prominence=0.0, distance=max(1, int(min_separation)))
    keep = np.asarray(radii_px)[peaks] > float(exclude_radius)
    peaks, prominences = peaks[keep], properties["prominences"][keep]
    with np.errstate(divide="ignore"):
        snr = prominences / np.asarray(standard_error, dtype=np.float64)[peaks]
    significant = snr >= float(min_snr)
    peaks, snr = peaks[significant], snr[significant]
    return [float(radii_px[peak]) for peak in peaks[np.argsort(-snr, kind="stable")]]


def radial_envelope_width(n_bins: int) -> int:
    """Odd window (bins) of the profile envelope: 1/8 of the profile, at least 9 bins.

    It must exceed the base of a ring (about 2.5 FWHM) so the envelope passes under
    every ring; neighbouring rings closer than the window keep their own maxima in the
    residual, only their shared base is lifted.
    """
    return max(9, int(n_bins) // 8 | 1)


# --- Calibration ---
def calibration_scale_from_phase(
    radii: Sequence[float],
    inv_d: Sequence[float],
    *,
    ratio_tol: float = RATIO_TOL,
    precision: float = 1.0e-3,
    margin: float = AMBIGUITY_MARGIN,
) -> dict:
    """Pixels per 1/Å that explain the ring radii of a known phase, verified scale-free.

    For a single phase every ring radius is ``r_i = scale * g_j`` (``g = 1/d``), so the
    ratios of ring radii must equal ratios of reflection ``g``, whatever the camera
    length. Every (ring, reflection) pair proposes a scale; each proposal assigns every
    ring to its nearest reflection (one ring per reflection), keeps the rings within
    ``ratio_tol`` (relative, so this is a ratio test), re-estimates the scale as the
    median of ``r / g`` over them (one ring biased by an unresolved neighbour does not
    drag the others) and iterates.

    A proposal is scored as a log likelihood ratio against chance, in nats:

    - each explained ring adds ``-ln p`` with ``p = min(1, 2 max(e, precision) / s)``,
      the probability that a ring lands within its relative error ``e`` of a reflection
      whose nearest neighbouring reflection is a relative gap ``s`` away;
    - each reflection the proposal predicts between its innermost and outermost
      explained ring costs ``ln 2`` (seen or not, as if half of the allowed reflections
      were too weak to show);
    - each ring the proposal leaves unexplained costs ``ln 10`` (as if one detected ring
      in ten came from another phase or an artefact).

    Dense high-order reflections (a spinel above a few 1/Å) make chance matches cheap,
    and an exact alias (every ``g`` doubled for a cubic lattice) predicts many rings that
    are not there, so neither can outscore a precise match of the sparse low-order
    reflections. That is how the earlier "most rings within 3 %" rule accepted a
    consistent but wrong scale.

    Returns ``scale`` (px per 1/Å), ``assigned`` (index into ``inv_d`` per ring, or
    None), ``errors`` (relative, or None), ``n_explained``, ``n_predicted``, ``rms``
    (relative, explained rings), ``score`` and ``alternative`` (the best proposal at a
    scale more than ``ratio_tol`` away, or None). Raises ``ValueError`` when no scale
    explains two rings, when more rings are unexplained than explained, or when the
    alternative scores within ``margin`` of the best: then the rings do not decide the
    scale and no calibration is better than a guessed one.
    """
    radii = np.asarray(radii, dtype=np.float64)
    g = np.asarray(inv_d, dtype=np.float64)
    order = np.argsort(g)
    g_sorted = g[order]
    gaps = np.diff(g_sorted) / g_sorted[:-1] if g_sorted.size > 1 else np.zeros(0)
    # relative gap to the nearest neighbouring reflection; a lone reflection is never ambiguous
    spacing = np.minimum(np.minimum(np.append(gaps, np.inf), np.insert(gaps, 0, np.inf)), 1.0)

    def solve(scale: float) -> dict | None:
        """Refine one proposed ``scale`` (assign, keep rings within ``ratio_tol``, median
        refit, at most four rounds) and score it; None when fewer than two rings fit."""
        for _ in range(4):
            predicted = radii / scale
            index = np.searchsorted(g_sorted, predicted)
            lower = np.clip(index - 1, 0, g_sorted.size - 1)
            upper = np.clip(index, 0, g_sorted.size - 1)
            nearest = np.where(np.abs(g_sorted[lower] - predicted) <= np.abs(g_sorted[upper] - predicted), lower, upper)
            errors = np.abs(predicted - g_sorted[nearest]) / g_sorted[nearest]
            explained = errors <= ratio_tol
            # one ring per reflection: the closer ring keeps it
            for reflection in np.unique(nearest[explained]):
                rings = np.flatnonzero(explained & (nearest == reflection))
                explained[rings[rings != rings[np.argmin(errors[rings])]]] = False
            if explained.sum() < 2:
                return None
            refit = float(np.median(radii[explained] / g_sorted[nearest[explained]]))
            converged = abs(refit / scale - 1.0) < 1e-12
            scale = refit
            if converged:
                break
        chance = np.minimum(1.0, 2.0 * np.maximum(errors[explained], precision) / spacing[nearest[explained]])
        g_lo, g_hi = radii[explained].min() / scale, radii[explained].max() / scale
        n_predicted = int(np.count_nonzero((g_sorted >= g_lo * (1 - ratio_tol)) & (g_sorted <= g_hi * (1 + ratio_tol))))
        return {
            "scale": scale,
            "assigned": [
                int(order[reflection]) if is_explained else None for reflection, is_explained in zip(nearest, explained)
            ],
            "errors": [float(error) if is_explained else None for error, is_explained in zip(errors, explained)],
            "n_explained": int(explained.sum()),
            "n_predicted": n_predicted,
            "rms": float(np.sqrt(np.mean(errors[explained] ** 2))),
            "score": float(-np.log(chance).sum() - n_predicted * math.log(2.0) - (~explained).sum() * math.log(10.0)),
        }

    proposals = {}
    for scale in np.unique(radii[:, None] / g_sorted[None, :]):
        solution = solve(float(scale))
        if solution is not None:
            key = tuple(solution["assigned"])
            if key not in proposals or solution["score"] > proposals[key]["score"]:
                proposals[key] = solution
    if not proposals:
        raise ValueError(
            f"no scale explains two or more rings within {100 * ratio_tol:.1f}% (ratio test); "
            "check the phase, or calibrate from one ring of known d"
        )
    ranked = sorted(proposals.values(), key=lambda solution: -solution["score"])
    best = ranked[0]
    best["alternative"] = next((other for other in ranked[1:] if abs(other["scale"] / best["scale"] - 1.0) > ratio_tol), None)
    n_rings = len(radii)
    if n_rings - best["n_explained"] > best["n_explained"]:
        raise ValueError(
            f"only {best['n_explained']} of {n_rings} rings fit one scale within {100 * ratio_tol:.1f}%; "
            "the phase does not explain this pattern"
        )
    alternative = best["alternative"]
    if alternative is not None and alternative["score"] >= best["score"] - margin:
        raise ValueError(
            f"ambiguous: {best['n_explained']} rings fit {best['scale']:.4g} px per 1/Å and "
            f"{alternative['n_explained']} fit {alternative['scale']:.4g}; calibrate from one ring of known d"
        )
    return best


# --- Phase matching ---
def match_candidate(observed_d: Sequence[float], lines: Sequence[dict], tol: float = 0.03) -> dict:
    """Match measured d-spacings against one reference phase's lines.

    Observed and reference lines are paired in reciprocal space by a one-to-one
    assignment that maximises the matches within ``tol`` and then minimises the total
    error. ``n_missing_strong`` counts reference lines with relative intensity >= 25 inside
    the observed range that found no partner (None when the lines carry no intensities).
    """
    observed = [
        float(spacing)
        for spacing in observed_d
        if spacing and math.isfinite(float(spacing)) and float(spacing) > 0
    ]
    references = [
        (float(line["d"]), float(line.get("i_rel", line.get("intensity")) or 0.0))
        for line in lines
        if math.isfinite(float(line["d"])) and float(line["d"]) > 0
    ]
    has_intensity = any(rel_intensity > 0 for _, rel_intensity in references)
    n_observed = len(observed)
    if n_observed == 0 or not references:
        return {
            "matched": 0,
            "n_obs": n_observed,
            "mean_err": None,
            "n_missing_strong": 0 if has_intensity else None,
            "assignments": [],
        }
    observed_g = [1.0 / spacing for spacing in observed]
    reference_g = [1.0 / spacing for spacing, _ in references]
    cost = np.full((n_observed, len(references)), NO_MATCH_COST)
    d_errors = np.zeros_like(cost)
    for obs_index, observed_value in enumerate(observed_g):
        for ref_index, reference_value in enumerate(reference_g):
            error = abs(reference_value - observed_value) / observed_value
            if error <= tol:
                cost[obs_index, ref_index] = error
                d_errors[obs_index, ref_index] = (
                    abs(1.0 / observed_value - references[ref_index][0]) / references[ref_index][0]
                )
    assignments = [(obs_index, None) for obs_index in range(n_observed)]
    errors = []
    matched_refs = set()
    for obs_index, ref_index in zip(*linear_sum_assignment(cost)):
        if cost[obs_index, ref_index] >= NO_MATCH_COST:
            continue
        assignments[obs_index] = (int(obs_index), int(ref_index))
        errors.append(float(d_errors[obs_index, ref_index]))
        matched_refs.add(int(ref_index))
    n_matched = len(errors)
    n_missing_strong = None
    if has_intensity:
        g_min, g_max = min(observed_g), max(observed_g)
        n_missing_strong = sum(
            1
            for ref_index, (_, rel_intensity) in enumerate(references)
            if rel_intensity >= 25.0
            and g_min <= reference_g[ref_index] <= g_max
            and ref_index not in matched_refs
        )
    return {
        "matched": n_matched,
        "n_obs": n_observed,
        "mean_err": (float(sum(errors) / n_matched) if n_matched else None),
        "n_missing_strong": n_missing_strong,
        "assignments": assignments,
    }


def match_sort_key(report: dict) -> tuple:
    """Sort phase reports from strongest to weakest match: most lines matched, fewest
    missing strong lines, lowest mean error."""
    return (
        -report["matched"],
        report["n_missing_strong"] or 0,
        report["mean_err"] if report["mean_err"] is not None else 1.0,
    )


def rank_phases(observed: list[float], phases: Sequence[Phase], tol: float, top_n: int) -> list[dict]:
    """Rank candidate phases against sorted observed d-spacings.

    Each report carries ``name phase_id matched n_obs mean_err n_missing_strong`` and
    ``lines``: one row per observed d (with its assigned reference line, if any) plus up
    to five strong reference lines inside the observed range that were not matched.
    """
    reports = []
    d_min = min(observed) * 0.8
    d_low, d_high = min(observed), max(observed)
    for phase in phases:
        # a degenerate candidate (d-spacing card below d_min) has no lines to match
        try:
            lines = [
                {"d": reflection["d"], "hkl": reflection["hkl_str"], "i_rel": reflection["intensity"]}
                for reflection in phase.reflections(d_min=d_min)
            ]
        except ValueError:
            continue
        if len(lines) < 2:
            continue
        report = match_candidate(observed, lines, tol=tol)
        assignments = dict(report.pop("assignments"))
        rows = []
        used_refs = set()
        for obs_index, measured_d in enumerate(observed):
            ref_index = assignments.get(obs_index)
            if ref_index is None:
                rows.append({"obs_d": float(measured_d), "ref_d": None, "hkl": "", "err": None, "i_rel": None})
                continue
            ref = lines[ref_index]
            used_refs.add(ref_index)
            rows.append(
                {
                    "obs_d": float(measured_d),
                    "ref_d": float(ref["d"]),
                    "hkl": ref["hkl"],
                    "err": abs(ref["d"] - measured_d) / ref["d"],
                    "i_rel": ref["i_rel"],
                }
            )
        missing = [
            ref
            for j, ref in enumerate(lines)
            if j not in used_refs and d_low <= ref["d"] <= d_high and (ref["i_rel"] or 0) >= 25
        ]
        missing.sort(key=lambda ref: -(ref["i_rel"] or 0))
        rows += [
            {"obs_d": None, "ref_d": float(ref["d"]), "hkl": ref["hkl"], "err": None, "i_rel": ref["i_rel"]}
            for ref in missing[:5]
        ]
        reports.append({**report, "phase_id": f"phase-{phase.name}", "name": phase.name, "lines": rows})
    if not reports:
        raise ValueError("no candidate phases pass the filters")
    reports.sort(key=match_sort_key)
    return reports[: int(top_n)]


def identify_summary(reports: list[dict]) -> str:
    """One-line status for ranked phase reports: the winner, a warning when fewer than
    four lines were measured, and the two runners-up (flagging ties)."""
    top = reports[0]
    status = f"{top['name']}: {top['matched']}/{top['n_obs']} lines"
    if top["n_obs"] < 4:
        status += "; few measured lines"
    runners = ", ".join(
        report["name"]
        + (f" (also {report['matched']}/{report['n_obs']})" if report["matched"] == top["matched"] else "")
        for report in reports[1:3]
    )
    return f"{status}; next: {runners}" if runners else status


def match_report(phase: Phase, d_values: Sequence[float | None], tol: float) -> str:
    """Phase-match status line: ``name (absences): matched/total matched, x% mean error``."""
    errors = []
    for d in d_values:
        if d and d > 0:
            candidates = phase.match_d(d, tol)
            if candidates:
                errors.append(candidates[0]["d_error"])
    n_total = sum(1 for d in d_values if d and d > 0)
    mean_err = float(np.mean(errors)) if errors else 0.0
    return f"{phase.name} ({phase.absences}): {len(errors)}/{n_total} matched, {100.0 * mean_err:.1f}% mean error"


# --- Zone-axis indexing ---
def vector_angle(vector1: tuple[float, float], vector2: tuple[float, float]) -> float:
    """Angle in degrees between two g-vectors (0 when either is null)."""
    norm1, norm2 = math.hypot(*vector1), math.hypot(*vector2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    # clipped: rounding can push the cosine of parallel vectors just past 1
    cosine = max(-1.0, min(1.0, (vector1[0] * vector2[0] + vector1[1] * vector2[1]) / (norm1 * norm2)))
    return math.degrees(math.acos(cosine))


def hkl_variants(phase: Phase, hkl: Sequence[int]) -> list[tuple[int, int, int]]:
    """All equal-d images of a reflection, canonical first.

    Signed permutations do not close hexagonal families (the 60-degree
    images of (110) are (-1,2,0)-type), so the whole index grid within a
    sum-of-|indices| bound is scanned for reflections at the same d.
    """
    d_ref = phase.d_spacing(hkl)
    bound = max(1, int(sum(abs(int(i)) for i in hkl)))
    axis = np.arange(-bound, bound + 1)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    inv_d_sq = np.einsum("ij,jk,ik->i", grid, phase.g_star, grid)
    target = 1.0 / d_ref**2
    close = np.abs(inv_d_sq - target) <= 2e-6 * target  # rel_tol 1e-6 on d
    variants = {tuple(int(i) for i in row) for row in grid[close] if phase.is_allowed(row)}
    return sorted(variants, key=label_preference)


def index_spot_vectors(
    phase: Phase,
    vectors: Sequence[tuple[float, float]],
    d_values: Sequence[float | None],
    *,
    tol: float,
    angle_tol: float,
) -> tuple[list[dict | None], list[list[dict]], str]:
    """Assign hkl to spots from their g-vectors and d-spacings, solving the zone axis.

    Each spot gets d-spacing candidates from the phase. The first non-collinear spot
    pair whose candidates reproduce the measured inter-spot angle anchors the zone axis
    (cross product of the two plane normals, choosing the equal-d variant of the second
    reflection that matches the angle). Every other spot takes the candidate variant that
    satisfies the zone law and the measured angle to the first anchor. Without an anchor
    pair each spot takes its closest d candidate and the zone axis is empty.

    Returns ``(assignments, candidate_lists, zone_axis)`` where each assignment is a
    ``match_d`` record (its ``hkl_str`` relabelled to the in-zone variant) or None.
    """
    candidate_lists = [phase.match_d(d, tol) if d else [] for d in d_values]
    n_spots = len(vectors)
    anchors = None
    for i in range(n_spots):
        for j in range(i + 1, n_spots):
            if anchors is not None or not candidate_lists[i] or not candidate_lists[j]:
                continue
            measured = vector_angle(vectors[i], vectors[j])
            if measured < 1e-6:
                continue
            best = None
            for candidate_i in candidate_lists[i]:
                for candidate_j in candidate_lists[j]:
                    error = abs(phase.plane_angle(candidate_i["hkl"], candidate_j["hkl"]) - measured)
                    if error <= angle_tol and (best is None or error < best[0]):
                        best = (error, candidate_i, candidate_j)
            if best is not None:
                anchors = (i, j, best[1], best[2])
    if anchors is None:
        return [candidates[0] if candidates else None for candidates in candidate_lists], candidate_lists, ""
    i, j, candidate_i, candidate_j = anchors
    measured_ij = vector_angle(vectors[i], vectors[j])
    variant_j = min(
        hkl_variants(phase, candidate_j["hkl"]),
        key=lambda variant: round(abs(phase.plane_angle(candidate_i["hkl"], variant) - measured_ij), 6),
    )
    h1, k1, l1 = (int(index) for index in candidate_i["hkl"])
    h2, k2, l2 = variant_j
    zone = (k1 * l2 - l1 * k2, l1 * h2 - h1 * l2, h1 * k2 - k1 * h2)

    def in_zone(hkl: Sequence[int]) -> bool:
        """Weiss zone law ``h u + k v + l w = 0``: the reflection lies in the anchored zone."""
        return hkl[0] * zone[0] + hkl[1] * zone[1] + hkl[2] * zone[2] == 0

    anchor_choice = {i: (candidate_i, tuple(int(index) for index in candidate_i["hkl"])), j: (candidate_j, variant_j)}
    assignments: list[dict | None] = []
    for spot_index, candidates in enumerate(candidate_lists):
        if spot_index in anchor_choice:
            chosen, variant = anchor_choice[spot_index]
        else:
            measured = vector_angle(vectors[i], vectors[spot_index])
            chosen, variant, best_error = None, None, None
            for candidate in candidates:
                for candidate_variant in hkl_variants(phase, candidate["hkl"]):
                    if not in_zone(candidate_variant):
                        continue
                    error = round(abs(phase.plane_angle(candidate_i["hkl"], candidate_variant) - measured), 6)
                    if error <= angle_tol and (best_error is None or error < best_error):
                        chosen, variant, best_error = candidate, candidate_variant, error
        if chosen is not None and not in_zone(tuple(int(index) for index in chosen["hkl"])):
            chosen = {**chosen, "hkl_str": format_hkl(variant)}
        assignments.append(chosen)
    return assignments, candidate_lists, format_zone_axis(candidate_i["hkl"], variant_j)
