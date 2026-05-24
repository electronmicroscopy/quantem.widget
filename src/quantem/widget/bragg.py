"""Generalized Bragg-disk detection for diffraction patterns.

Shared by diffraction/stack widgets (Show3D; Show4DSTEM should
import this in the future to drop its duplicated copy)
"""

import numpy as np
import torch

from quantem.core.utils.imaging_utils import upsampled_correlation_torch


def detect_bragg_disks_single(
    dp: np.ndarray,
    probe_kernel: np.ndarray,
    *,
    corr_power: float = 1.0,
    sigma: float = 2.0,
    edge_boundary: int = 1,
    min_relative_intensity: float = 0.005,
    min_absolute_intensity: float = 0.0,
    min_peak_spacing: float = 4.0,
    max_num_peaks: int = 70,
    subpixel: str = "multicorr",
    upsample_factor: int = 4,
) -> np.ndarray:
    """Detect Bragg disks in a single diffraction pattern.

    Cross-correlates ``dp`` with ``probe_kernel`` (origin-aligned vacuum probe),
    finds local maxima, filters them, and refines to sub-pixel precision.

    Parameters
    ----------
    dp : (n_rows, n_cols) array
        Diffraction pattern (float32, non-negative).
    probe_kernel : (n_rows, n_cols) array
        Origin-aligned vacuum-probe template (peak at (0, 0)); see
        :func:`vacuum_probe_kernel`. Must match ``dp`` shape.
    corr_power : float
        Correlation power: 1 = cross-correlation, 0 = phase, between = hybrid.
    sigma : float
        Gaussian smoothing of the correlogram before maxima (0 disables).
    edge_boundary : int
        Reject peaks within this many pixels of the edge.
    min_relative_intensity, min_absolute_intensity : float
        Drop peaks below this fraction of the brightest / this absolute value.
    min_peak_spacing : float
        Minimum peak separation in pixels (greedy NMS).
    max_num_peaks : int
        Keep at most this many peaks (intensity-sorted, descending).
    subpixel : str
        ``"pixel"``, ``"poly"`` (3x3 parabolic), or ``"multicorr"`` (DFT upsample).
    upsample_factor : int
        Multicorr upsampling factor (>= 2).

    Returns
    -------
    peaks : (N, 3) float32 array
        ``[row, col, intensity]`` rows, sorted by intensity descending.
    """
    from scipy.ndimage import gaussian_filter, maximum_filter

    if subpixel not in {"pixel", "poly", "multicorr"}:
        raise ValueError(
            f"subpixel must be 'pixel', 'poly', or 'multicorr', got {subpixel!r}"
        )

    dp = np.asarray(dp, dtype=np.float32)
    template = np.asarray(probe_kernel, dtype=np.float32)
    if dp.shape != template.shape:
        raise ValueError(
            f"dp shape {dp.shape} must match probe_kernel shape {template.shape}"
        )

    # (1) FFTs
    dp_ft = np.fft.fft2(dp)
    template_ft_conj = np.conj(np.fft.fft2(template))

    # (2) cross-power spectrum (hybrid correlation when corr_power != 1)
    m = dp_ft * template_ft_conj
    if corr_power != 1.0:
        # |m|^p * exp(i*angle(m)) — preserves phase, modulates magnitude
        cc_ft = (np.abs(m) ** float(corr_power)) * np.exp(1j * np.angle(m))
    else:
        cc_ft = m
    cc = np.maximum(np.real(np.fft.ifft2(cc_ft)), 0.0).astype(np.float32)

    # (3) Gaussian blur of correlogram
    if sigma > 0:
        cc_smooth = gaussian_filter(cc, float(sigma)).astype(np.float32)
    else:
        cc_smooth = cc

    # (4) local maxima detection
    footprint_size = max(3, int(round(float(min_peak_spacing))))
    if footprint_size % 2 == 0:
        footprint_size += 1
    filtered = maximum_filter(cc_smooth, size=footprint_size, mode="constant", cval=0.0)
    maxima = (cc_smooth == filtered) & (cc_smooth > 0)

    # edge boundary
    eb = max(1, int(edge_boundary))
    if eb < maxima.shape[0] and eb < maxima.shape[1]:
        maxima[:eb, :] = False
        maxima[-eb:, :] = False
        maxima[:, :eb] = False
        maxima[:, -eb:] = False

    rows, cols = np.nonzero(maxima)
    if rows.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    intensities = cc_smooth[rows, cols]

    # sort by intensity (descending)
    order = np.argsort(-intensities)
    rows = rows[order]
    cols = cols[order]
    intensities = intensities[order]

    # (5) filtering
    if min_absolute_intensity > 0:
        keep = intensities >= float(min_absolute_intensity)
        rows, cols, intensities = rows[keep], cols[keep], intensities[keep]
    if intensities.size and min_relative_intensity > 0:
        peak_max = float(intensities[0])
        if peak_max > 0:
            keep = intensities >= float(min_relative_intensity) * peak_max
            rows, cols, intensities = rows[keep], cols[keep], intensities[keep]

    # greedy NMS for minimum spacing
    if min_peak_spacing > 0 and rows.size > 1:
        spacing_sq = float(min_peak_spacing) ** 2
        keep_mask = np.ones(rows.size, dtype=bool)
        for i in range(rows.size):
            if not keep_mask[i]:
                continue
            dy = rows[i + 1 :] - rows[i]
            dx = cols[i + 1 :] - cols[i]
            too_close = (dy * dy + dx * dx) < spacing_sq
            keep_mask[i + 1 :] &= ~too_close
        rows = rows[keep_mask]
        cols = cols[keep_mask]
        intensities = intensities[keep_mask]

    # cap at max_num_peaks
    if int(max_num_peaks) > 0 and rows.size > int(max_num_peaks):
        rows = rows[: int(max_num_peaks)]
        cols = cols[: int(max_num_peaks)]
        intensities = intensities[: int(max_num_peaks)]

    if rows.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    qy = rows.astype(np.float64)
    qx = cols.astype(np.float64)
    inten = intensities.astype(np.float64)

    # (6) subpixel refinement
    if subpixel == "pixel":
        pass
    elif subpixel == "poly":
        n_rows, n_cols = cc_smooth.shape
        for i in range(qy.size):
            r = int(rows[i])
            c = int(cols[i])
            if r <= 0 or r >= n_rows - 1 or c <= 0 or c >= n_cols - 1:
                continue
            center = float(cc_smooth[r, c])
            row_minus = float(cc_smooth[r - 1, c])
            row_plus = float(cc_smooth[r + 1, c])
            col_minus = float(cc_smooth[r, c - 1])
            col_plus = float(cc_smooth[r, c + 1])
            denom_y = 4.0 * center - 2.0 * row_plus - 2.0 * row_minus
            denom_x = 4.0 * center - 2.0 * col_plus - 2.0 * col_minus
            dy = (row_plus - row_minus) / denom_y if denom_y != 0 else 0.0
            dx = (col_plus - col_minus) / denom_x if denom_x != 0 else 0.0
            # clamp to [-0.5, 0.5] to avoid runaway
            dy = max(-0.5, min(0.5, dy))
            dx = max(-0.5, min(0.5, dx))
            qy[i] += dy
            qx[i] += dx
    else:  # multicorr — DFT upsampling via quantem primitive
        # upsampled_correlation_torch asserts factor > 2; clamp to 3.
        up = max(3, int(upsample_factor))
        cc_ft_full = np.fft.fft2(cc)  # use the un-smoothed cc for upsampling
        cc_ft_torch = torch.from_numpy(cc_ft_full)
        for i in range(qy.size):
            xy_shift = torch.tensor([float(qy[i]), float(qx[i])], dtype=torch.float64)
            try:
                refined = upsampled_correlation_torch(cc_ft_torch, up, xy_shift)
                qy[i] = float(refined[0].item())
                qx[i] = float(refined[1].item())
            except Exception:
                # fall back to pixel coords if DFT upsample fails (e.g. near edge)
                pass

    out = np.column_stack([qy, qx, inten]).astype(np.float32)
    return out


def vacuum_probe_kernel(
    probe_centered: np.ndarray,
    center_row: float,
    center_col: float,
) -> np.ndarray:
    """Shift a centered vacuum probe to the FFT origin via a Fourier phase ramp.

    Moves the probe peak from ``(center_row, center_col)`` to ``(0, 0)`` (FFT
    convention, like py4DSTEM's ``probe.kernel``) so correlation peaks land at
    the Bragg-disk positions. The phase-ramp shift is sub-pixel-correct, unlike
    ``np.roll`` which would quantize non-integer centers to half a pixel.

    Parameters
    ----------
    probe_centered : (n_rows, n_cols) array
        Vacuum probe with its peak at ``(center_row, center_col)``.
    center_row, center_col : float
        Probe center (may be non-integer).

    Returns
    -------
    kernel : (n_rows, n_cols) float32 array
        Origin-aligned probe kernel for matched filtering.
    """
    probe_centered = np.asarray(probe_centered, dtype=np.float32)
    n_rows, n_cols = probe_centered.shape
    center_row = float(center_row)
    center_col = float(center_col)
    ky = np.fft.fftfreq(n_rows).astype(np.float32)[:, None]
    kx = np.fft.fftfreq(n_cols).astype(np.float32)[None, :]
    # FFT shift theorem g(y)=f(y+a) <-> G(k)=F(k)*exp(+2j*pi*k*a): peak -> origin.
    phase_ramp = np.exp(2j * np.pi * (ky * center_row + kx * center_col))
    ft = np.fft.fft2(probe_centered) * phase_ramp
    return np.real(np.fft.ifft2(ft)).astype(np.float32)


def build_soft_disk_probe(
    n_rows: int,
    n_cols: int,
    center_row: float,
    center_col: float,
    radius: float,
    soft_edge: float = 2.0,
) -> np.ndarray:
    """Build a soft-edge disk vacuum probe (1 inside, linear roll-off at the rim).

    The "user-visible" centered probe; detection shifts it to the origin with
    :func:`vacuum_probe_kernel`.

    Parameters
    ----------
    n_rows, n_cols : int
        Probe shape (match the diffraction pattern).
    center_row, center_col : float
        Disk center (may be non-integer).
    radius : float
        Disk radius in pixels.
    soft_edge : float
        Half-width of the roll-off region in pixels.
    """
    rows = np.arange(int(n_rows), dtype=np.float32)[:, None]
    cols = np.arange(int(n_cols), dtype=np.float32)[None, :]
    center_row = float(center_row)
    center_col = float(center_col)
    radius = max(float(radius), 1.0)
    soft_edge = max(float(soft_edge), 1e-3)
    dist = np.sqrt((rows - center_row) ** 2 + (cols - center_col) ** 2)
    probe = np.clip((radius + soft_edge - dist) / (2 * soft_edge), 0.0, 1.0)
    return probe.astype(np.float32)
