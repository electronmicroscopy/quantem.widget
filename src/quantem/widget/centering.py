"""Center estimation and stack alignment for diffraction patterns."""

import numpy as np
from scipy import ndimage
from scipy.signal.windows import tukey


# --- Internal helpers ---
def _bandpass(frame: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    work = np.log1p(frame - frame.min())
    sigma = max(5.0, 0.02 * min(work.shape))
    work = work - ndimage.gaussian_filter(work, sigma=sigma)
    if mask is not None:
        work[mask] = 0.0
    return work - work.mean()


def _parabolic_offset(values: np.ndarray, p: int, n: int) -> float:
    lo, hi = values[(p - 1) % n], values[(p + 1) % n]
    denom = 2.0 * values[p] - lo - hi
    if denom <= 0:
        return 0.0
    delta = 0.5 * (hi - lo) / denom
    return delta if abs(delta) <= 1.0 else 0.0


def _peak_to_sidelobe(corr: np.ndarray, p_row: int, p_col: int, exclude: float = 5.0) -> float:
    n_rows, n_cols = corr.shape
    row_dist = np.abs(np.arange(n_rows) - p_row)
    col_dist = np.abs(np.arange(n_cols) - p_col)
    row_dist = np.minimum(row_dist, n_rows - row_dist)
    col_dist = np.minimum(col_dist, n_cols - col_dist)
    outside = (row_dist[:, None] > exclude) | (col_dist[None, :] > exclude)
    side = corr[outside]
    spread = side.std()
    if spread <= 0:
        return 0.0
    return float((corr[p_row, p_col] - side.mean()) / spread)


def _upsampled_peak(
    cross: np.ndarray, row0: float, col0: float, upsample: int
) -> tuple[float, float]:
    n_rows, n_cols = cross.shape
    f_row = np.fft.fftfreq(n_rows)
    f_col = np.fft.fftfreq(n_cols)
    offsets = np.arange(-int(np.ceil(1.5 * upsample)), int(np.ceil(1.5 * upsample)) + 1) / upsample
    rows = row0 + offsets
    cols = col0 + offsets
    e_row = np.exp(2j * np.pi * rows[:, None] * f_row[None, :])
    e_col = np.exp(2j * np.pi * f_col[:, None] * cols[None, :])
    local = (e_row @ cross @ e_col).real
    p_row, p_col = np.unravel_index(int(np.argmax(local)), local.shape)
    d_row = _parabolic_offset(local[:, p_col], p_row, local.shape[0]) / upsample
    d_col = _parabolic_offset(local[p_row, :], p_col, local.shape[1]) / upsample
    return float(rows[p_row] + d_row), float(cols[p_col] + d_col)


def _phase_shift(
    ref: np.ndarray, moving: np.ndarray, upsample: int = 1
) -> tuple[float, float, float]:
    window = tukey(ref.shape[0], 0.2)[:, None] * tukey(ref.shape[1], 0.2)[None, :]
    ref = ref * window
    moving = moving * window
    cross = np.fft.fft2(ref) * np.conj(np.fft.fft2(moving))
    cross = cross / np.maximum(np.abs(cross), 1e-12)
    f_row = np.fft.fftfreq(ref.shape[0])[:, None]
    f_col = np.fft.fftfreq(ref.shape[1])[None, :]
    cross = cross * np.exp(-(f_row**2 + f_col**2) / (2.0 * 0.15**2))
    corr = np.fft.ifft2(cross).real
    n_rows, n_cols = corr.shape
    p_row, p_col = np.unravel_index(int(np.argmax(corr)), corr.shape)
    psr = _peak_to_sidelobe(corr, p_row, p_col)
    if upsample > 1:
        s_row, s_col = _upsampled_peak(cross, float(p_row), float(p_col), upsample)
    else:
        s_row = p_row + _parabolic_offset(corr[:, p_col], p_row, n_rows)
        s_col = p_col + _parabolic_offset(corr[p_row, :], p_col, n_cols)
    return float(s_row), float(s_col), psr


def _wrap_signed(shift: float, n: int) -> float:
    shift = shift % n
    return shift - n if shift > n / 2 else shift


# --- Center estimation ---
def center_symmetry(
    frame: np.ndarray,
    guess: tuple[float, float] | None = None,
    search_radius: float = 8.0,
    mask: np.ndarray | None = None,
) -> tuple[float, float]:
    """Refine a center guess by local Friedel-symmetry autocorrelation."""
    frame = np.asarray(frame, dtype=np.float64)
    n_rows, n_cols = frame.shape
    if guess is None:
        guess = ((n_rows - 1) / 2.0, (n_cols - 1) / 2.0)

    work = _bandpass(frame, mask)
    spectrum = np.fft.fft2(work)
    corr = np.fft.ifft2(spectrum * spectrum).real
    target_row = (2.0 * guess[0]) % n_rows
    target_col = (2.0 * guess[1]) % n_cols
    row_idx = np.arange(n_rows, dtype=np.float64)
    col_idx = np.arange(n_cols, dtype=np.float64)
    row_dist = np.minimum(np.abs(row_idx - target_row), n_rows - np.abs(row_idx - target_row))
    col_dist = np.minimum(np.abs(col_idx - target_col), n_cols - np.abs(col_idx - target_col))
    near = (row_dist[:, None] <= 2.0 * search_radius) & (col_dist[None, :] <= 2.0 * search_radius)
    p_row, p_col = np.unravel_index(int(np.argmax(np.where(near, corr, -np.inf))), corr.shape)
    row2 = p_row + _parabolic_offset(corr[:, p_col], p_row, n_rows)
    col2 = p_col + _parabolic_offset(corr[p_row, :], p_col, n_cols)
    row = min(((row2 + offset) / 2.0 for offset in (0.0, n_rows)), key=lambda c: abs(c - guess[0]))
    col = min(((col2 + offset) / 2.0 for offset in (0.0, n_cols)), key=lambda c: abs(c - guess[1]))
    return float(row), float(col)


def center_phase_correlation(
    frame: np.ndarray, mask: np.ndarray | None = None, upsample: int = 20
) -> tuple[float, float]:
    """Estimate the inversion center by phase correlation."""
    frame = np.asarray(frame, dtype=np.float64)
    n_rows, n_cols = frame.shape
    work = _bandpass(frame, mask)
    rot = work[::-1, ::-1]
    d_row, d_col, _ = _phase_shift(work, rot, upsample=upsample)
    # Inversion center
    row_cands = [(n_rows - 1 + d) / 2.0 for d in (d_row % n_rows, d_row % n_rows - n_rows)]
    col_cands = [(n_cols - 1 + d) / 2.0 for d in (d_col % n_cols, d_col % n_cols - n_cols)]
    candidates = [
        (r, c)
        for r in row_cands
        for c in col_cands
        if 0.0 <= r <= n_rows - 1 and 0.0 <= c <= n_cols - 1
    ]
    if not candidates:
        candidates = [((n_rows - 1) / 2.0, (n_cols - 1) / 2.0)]
    row, col = max(candidates, key=lambda rc: _symmetry(frame, rc, mask=mask))
    return float(row), float(col)


# --- Quality metrics ---
def _symmetry(
    frame: np.ndarray, center: tuple[float, float], mask: np.ndarray | None = None
) -> float:
    # Friedel symmetry
    frame = np.asarray(frame, dtype=np.float64)
    n_rows, n_cols = frame.shape
    work = _bandpass(frame)
    rows, cols = np.indices((n_rows, n_cols), dtype=np.float64)
    rot_rows = 2.0 * center[0] - rows
    rot_cols = 2.0 * center[1] - cols
    valid = (
        (rot_rows >= 0.0) & (rot_rows <= n_rows - 1) & (rot_cols >= 0.0) & (rot_cols <= n_cols - 1)
    )
    rotated = ndimage.map_coordinates(work, [rot_rows, rot_cols], order=1, mode="nearest")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        rot_mask = ndimage.map_coordinates(
            mask.astype(np.float64), [rot_rows, rot_cols], order=1, mode="constant", cval=1.0
        )
        valid &= ~mask & (rot_mask < 0.5)
    a = work[valid]
    b = rotated[valid]
    if a.size < 16:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom <= 0:
        return 0.0
    return float(max(0.0, (a * b).sum() / denom))


def ring_uniformity(
    frame: np.ndarray,
    center: tuple[float, float],
    radius: float,
    half_width: float = 4.0,
    n_theta: int = 180,
) -> dict:
    """Azimuthal uniformity QC for one ring."""
    frame = np.asarray(frame, dtype=np.float64)
    rows, cols = np.indices(frame.shape, dtype=np.float64)
    d_row = rows - center[0]
    d_col = cols - center[1]
    annulus = np.abs(np.hypot(d_row, d_col) - radius) <= half_width
    theta = np.arctan2(d_row[annulus], d_col[annulus])
    sector = np.clip(((theta + np.pi) / (2.0 * np.pi) * n_theta).astype(int), 0, n_theta - 1)
    sums = np.bincount(sector, weights=frame[annulus], minlength=n_theta)
    counts = np.bincount(sector, minlength=n_theta)
    profile = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    mean = float(profile.mean())
    std = float(profile.std())
    cv = std / mean if mean > 0 else 0.0
    positive = profile[profile > 0]
    coverage = float(np.mean(profile > 0.5 * np.median(positive))) if positive.size else 0.0
    snr = min(mean / std, 999.0) if std > 0 else 999.0
    return {"cv": float(cv), "coverage": float(coverage), "snr": float(snr)}


# --- Dispatch ---
def pick_center(
    frame: np.ndarray,
    method: str = "auto",
    mask: np.ndarray | None = None,
    guess: tuple[float, float] | None = None,
    search_radius: float = 8.0,
) -> dict:
    """Estimate the pattern center with one method or an automatic pick."""
    frame = np.asarray(frame, dtype=np.float64)
    if method == "symmetry":
        row, col = center_symmetry(frame, guess=guess, search_radius=search_radius, mask=mask)
        name = "symmetry"
    elif method == "phase_corr":
        row, col = center_phase_correlation(frame, mask=mask)
        name = "phase_corr"
    elif method == "auto":
        p_row, p_col = center_phase_correlation(frame, mask=mask)
        s_row, s_col = center_symmetry(frame, guess=guess, search_radius=search_radius, mask=mask)
        candidates = [
            ("phase_corr", p_row, p_col, _symmetry(frame, (p_row, p_col), mask=mask)),
            ("symmetry", s_row, s_col, _symmetry(frame, (s_row, s_col), mask=mask)),
        ]
        name, row, col, _ = max(candidates, key=lambda c: c[3])
    else:
        raise ValueError(f"unknown method {method!r}; use auto, symmetry, or phase_corr")
    return {"row": float(row), "col": float(col), "method": name}


# --- Stack alignment ---
def align_frames(
    frames: np.ndarray,
    reference: np.ndarray | None = None,
    max_shift: float = 8.0,
) -> tuple[np.ndarray, list[tuple[float, float]], list[bool]]:
    """Align a stack of patterns by subpixel phase correlation."""
    frames = np.asarray(frames, dtype=np.float64)
    n_frames, n_rows, n_cols = frames.shape
    ref = frames[0] if reference is None else np.asarray(reference, dtype=np.float64)
    ref = _bandpass(ref)
    aligned = np.empty_like(frames)
    shifts: list[tuple[float, float]] = []
    used: list[bool] = []
    for i in range(n_frames):
        s_row, s_col, psr = _phase_shift(ref, _bandpass(frames[i]))
        s_row = _wrap_signed(s_row, n_rows)
        s_col = _wrap_signed(s_col, n_cols)
        peak_quality = psr / (psr + 10.0) if psr > 0 else 0.0
        ok = np.hypot(s_row, s_col) <= max_shift and peak_quality >= 0.2
        shifts.append((float(s_row), float(s_col)))
        used.append(bool(ok))
        aligned[i] = ndimage.shift(frames[i], (s_row, s_col), order=1) if ok else frames[i]
    return aligned, shifts, used
