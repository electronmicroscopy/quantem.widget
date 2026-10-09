"""Global slice alignment for reconstructed multislice stacks.

A multislice ptychography object whose slices drift laterally with depth
shows one tilted column where the sample has a straight one. The estimator
registers each pair of adjacent slices, accumulates the shifts into a
trajectory and fits one straight line through it: the slope is the display
shift to apply per deeper slice. The volume is never modified; the browser
shifts slices on display only.

Registration follows Guizar-Sicairos, Thurman and Fienup, Opt. Lett. 33, 156
(2008), https://doi.org/10.1364/OL.33.000156: a whole-pixel cross-correlation
peak refined on a 20x local matrix-DFT grid. The Gaussian high-pass before
registration suppresses the slowly varying background that would otherwise
dominate the correlation of low-contrast phase slices; 12 px is an empirical
preprocessing scale, not a microscope parameter. ``js/sliceAlignment.ts``
mirrors every constant and step so an exported HTML page estimates the same
shifts without a kernel.
"""

import math
from collections.abc import Sequence

import numpy as np

HIGHPASS_SIGMA_PX = 12.0
UPSAMPLE_FACTOR = 20
DFT_REGION_FACTOR = 1.5


def estimate_global_slice_alignment(stack: np.ndarray) -> dict:
    """Fit one row/col display shift per slice through a ``(nz, ny, nx)`` stack.

    Adjacent slices are registered after median subtraction, Gaussian
    high-pass filtering and Hann windowing; the adjacent shifts are summed into
    a cumulative trajectory and a straight line is fit against slice index.
    The returned slopes are the shifts to APPLY to deeper slices (the negative
    of the measured drift), centred on the middle slice when rendered.

    Returns a JSON-serializable dict with ``row_shift_px_per_slice``,
    ``col_shift_px_per_slice``, the adjacent and cumulative shifts, the fit
    intercepts, per-axis R² and the normalized correlation peak per pair.
    """
    if stack.ndim != 3:
        raise ValueError(f"slice alignment expects a 3D stack, got shape {stack.shape}")
    if stack.shape[0] < 2:
        raise ValueError("slice alignment requires at least 2 slices")
    adjacent: list[list[float]] = []
    quality: list[float] = []
    ref = registration_image(stack[0])
    ref_spectrum = np.fft.fft2(ref)
    for idx in range(stack.shape[0] - 1):
        mov = registration_image(stack[idx + 1])
        mov_spectrum = np.fft.fft2(mov)
        row_shift, col_shift, score = estimate_adjacent_shift(ref, mov, ref_spectrum, mov_spectrum)
        adjacent.append([row_shift, col_shift])
        quality.append(score)
        ref, ref_spectrum = mov, mov_spectrum
    adjacent_arr = np.asarray(adjacent, dtype=np.float64)
    cumulative = np.zeros((stack.shape[0], 2), dtype=np.float64)
    cumulative[1:] = np.cumsum(adjacent_arr, axis=0)
    slice_index = np.arange(stack.shape[0], dtype=np.float64)
    slopes: list[float] = []
    intercepts: list[float] = []
    r2: list[float] = []
    for dim in range(2):
        slope, intercept = np.polyfit(slice_index, cumulative[:, dim], 1)
        residual = cumulative[:, dim] - (slope * slice_index + intercept)
        ss_res = float(np.sum(residual * residual))
        ss_tot = float(np.sum((cumulative[:, dim] - np.mean(cumulative[:, dim])) ** 2))
        if ss_tot <= 1e-12:
            fit_r2 = 1.0 if ss_res <= 1e-12 else 0.0
        else:
            fit_r2 = max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
        slopes.append(float(slope))
        intercepts.append(float(intercept))
        r2.append(fit_r2)
    return {
        "row_shift_px_per_slice": slopes[0],
        "col_shift_px_per_slice": slopes[1],
        "adjacent_shift_apply_px": adjacent_arr.tolist(),
        "cumulative_shift_apply_px": cumulative.tolist(),
        "fit_intercept_px": intercepts,
        "fit_r2": {"row": r2[0], "col": r2[1]},
        "quality": quality,
    }


# --- registration primitives ---------------------------------------------------


def registration_image(image: np.ndarray) -> np.ndarray:
    """Median-centre, Gaussian high-pass and Hann-window one slice before registration.

    The high-pass (FFT of a reflect-padded copy, sigma capped at 12 px and at
    a sixth of the short side) removes the smooth background that would pull
    the correlation peak toward zero shift; the Hann window removes the
    wrap-around edge the FFT cross-correlation would otherwise see.
    """
    out = np.asarray(image, dtype=np.float32)
    if not np.isfinite(out).all():
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = out - np.median(out)
    sigma = min(HIGHPASS_SIGMA_PX, max(1.0, min(out.shape) / 6.0))
    rows, cols = out.shape
    if sigma > 0 and rows >= 3 and cols >= 3:
        pad = max(1, int(math.ceil(4.0 * sigma)))
        padded = np.pad(out, ((pad, pad), (pad, pad)), mode="reflect")
        row_freq = np.fft.fftfreq(padded.shape[0])[:, None]
        col_freq = np.fft.fftfreq(padded.shape[1])[None, :]
        gaussian = np.exp(-2.0 * (math.pi**2) * (sigma**2) * (row_freq * row_freq + col_freq * col_freq))
        blur_padded = np.fft.ifft2(np.fft.fft2(padded) * gaussian).real
        blur = blur_padded[pad : pad + rows, pad : pad + cols]
        out = out - blur.astype(np.float32, copy=False)
    window = np.outer(np.hanning(rows), np.hanning(cols)).astype(np.float32)
    return np.ascontiguousarray(out * window, dtype=np.float32)


def estimate_adjacent_shift(
    ref: np.ndarray,
    mov: np.ndarray,
    ref_spectrum: np.ndarray,
    mov_spectrum: np.ndarray,
) -> tuple[float, float, float]:
    """Subpixel ``(row, col, quality)`` shift that aligns ``mov`` onto ``ref``.

    The whole-pixel peak of the inverse-FFT cross-correlation is refined on a
    ``UPSAMPLE_FACTOR`` grid over a 1.5-pixel region with the matrix DFT of
    ``upsampled_dft``. ``quality`` is the correlation peak over the product of
    the two image norms (1 for identical images, 0 for a flat slice).
    """
    if ref.shape != mov.shape:
        raise ValueError(f"registration shape mismatch: {ref.shape} != {mov.shape}")
    if float(np.std(ref)) == 0.0 or float(np.std(mov)) == 0.0:
        return 0.0, 0.0, 0.0
    product = ref_spectrum * np.conj(mov_spectrum)
    correlation = np.fft.ifft2(product)
    coarse_peak = np.unravel_index(int(np.argmax(np.abs(correlation))), correlation.shape)
    shape = np.asarray(correlation.shape, dtype=np.float64)
    shifts = np.asarray(coarse_peak, dtype=np.float64)
    midpoint = np.trunc(shape / 2.0)
    wrap = shifts > midpoint
    shifts[wrap] -= shape[wrap]
    upsample = UPSAMPLE_FACTOR
    shifts = np.round(shifts * upsample) / upsample
    region_size = int(math.ceil(upsample * DFT_REGION_FACTOR))
    dft_shift = float(np.trunc(region_size / 2.0))
    offsets = dft_shift - shifts * upsample
    refined_correlation = np.conj(upsampled_dft(np.conj(product), (region_size, region_size), upsample, offsets))
    refined_peak = np.asarray(np.unravel_index(int(np.argmax(np.abs(refined_correlation))), refined_correlation.shape),
                              dtype=np.float64)
    shifts += (refined_peak - dft_shift) / upsample
    peak_value = float(np.abs(correlation[coarse_peak]))
    norm = float(np.sqrt(np.sum(ref * ref) * np.sum(mov * mov)))
    quality = peak_value / norm if norm > 0 else 0.0
    return float(shifts[0]), float(shifts[1]), quality


def upsampled_dft(spectrum: np.ndarray, region_shape: tuple[int, int], upsample_factor: int, offsets: Sequence[float]) -> np.ndarray:
    """Inverse DFT of ``spectrum`` on a small upsampled region, as two matrix products.

    Evaluating only ``region_shape`` output samples around ``offsets`` costs
    O(region x N) instead of the O((uN)²) of zero-padding the whole spectrum
    by the upsample factor (Guizar-Sicairos 2008, Eq. 3).
    """
    rows, cols = spectrum.shape
    region_rows, region_cols = region_shape
    row_coords = np.arange(region_rows, dtype=np.float64)[:, None] - float(offsets[0])
    col_coords = np.arange(region_cols, dtype=np.float64)[:, None] - float(offsets[1])
    row_freq = np.fft.fftfreq(rows, d=float(upsample_factor))[None, :]
    col_freq = np.fft.fftfreq(cols, d=float(upsample_factor))[None, :]
    row_kernel = np.exp((-2j * math.pi) * row_coords * row_freq)
    col_kernel = np.exp((-2j * math.pi) * col_coords * col_freq)
    return row_kernel @ spectrum @ col_kernel.T
