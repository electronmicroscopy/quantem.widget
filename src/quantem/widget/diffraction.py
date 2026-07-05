"""Small reusable helpers for diffraction widgets."""

import csv
import json
import math
import pathlib
import re
import warnings

import numpy as np

BF_RADIUS_FRACTION = 0.125
RING_FIT_MODELS = ("gaussian", "pseudo_voigt")

MEASUREMENT_COLUMNS = [
    "id",
    "kind",
    "raw_row",
    "raw_col",
    "row",
    "col",
    "row_err",
    "col_err",
    "r_pixels",
    "r_pixels_err",
    "g_inv_angstrom",
    "g_inv_angstrom_err",
    "d_angstrom",
    "d_angstrom_err",
    "angle_deg",
    "angle_deg_err",
    "intensity",
    "fit_quality",
    "fwhm_px",
    "fwhm_inv_angstrom",
    "intensity_integrated",
    "hkl",
    "note",
]


# --- Records and formatting ---
def element_symbols(text: str) -> set[str]:
    """Element symbols found in a formula-like string."""
    return set(re.findall(r"[A-Z][a-z]?", text or ""))


def parse_elements(value) -> set[str] | None:
    """Element-symbol set from a string or iterable, or None if empty."""
    if not value:
        return None
    if isinstance(value, str):
        value = re.split(r"[,\s]+", value.strip())
    return {symbol.strip().capitalize() for symbol in value if symbol.strip()} or None


def index_assignment(candidate: dict | None) -> dict:
    """Indexing fields for a matched reflection candidate."""
    if candidate is None:
        return {"hkl": "", "d_ref": None, "d_error": None}
    return {
        "hkl": candidate["hkl_str"],
        "d_ref": candidate["d"],
        "d_error": candidate["d_error"],
    }


def empty_index_fields() -> dict:
    """Blank indexing fields for a spot or ring record."""
    return {
        "hkl": "",
        "hkl_candidates": [],
        "d_ref": None,
        "d_error": None,
        "note": "",
    }


def next_record_id(records) -> int:
    """Next one-based id for a list of record dicts."""
    return max((int(record["id"]) for record in records), default=0) + 1


def format_zone_axis(hkl1: tuple[int, int, int], hkl2: tuple[int, int, int]) -> str:
    """Zone-axis label ``[uvw]`` from two indexed reflections."""
    h1, k1, l1 = hkl1
    h2, k2, l2 = hkl2
    u = k1 * l2 - l1 * k2
    v = l1 * h2 - h1 * l2
    w = h1 * k2 - k1 * h2
    divisor = math.gcd(math.gcd(abs(u), abs(v)), abs(w))
    if divisor == 0:
        return ""
    u, v, w = u // divisor, v // divisor, w // divisor
    for axis in (u, v, w):
        if axis != 0:
            if axis < 0:
                u, v, w = -u, -v, -w
            break
    return "[" + "".join(str(axis) for axis in (u, v, w)) + "]"


# --- Input and masking ---
def normalize_data_input(
    data,
    *,
    title: str = "",
    pixel_size: float | None = None,
    k_pixel_size: float | None = None,
    replace_title: bool = False,
):
    """Unwrap Dataset-like input into array, title, and calibrations."""
    k_calibrated = False
    if hasattr(data, "_fields") and "data" in getattr(data, "_fields", ()):
        metadata = data.metadata or {}
        if pixel_size is None and metadata.get("pixel_size") is not None:
            pixel_size = float(metadata["pixel_size"])
        data = data.data

    if hasattr(data, "sampling") and hasattr(data, "array"):
        if (replace_title or not title) and getattr(data, "name", ""):
            title = str(data.name)
        units = list(getattr(data, "units", ["pixels"] * 4))
        if pixel_size is None and units and units[0] in ("Å", "angstrom", "A", "nm"):
            pixel_size = float(data.sampling[0])
            if units[0] == "nm":
                pixel_size *= 10
        if k_pixel_size is None and len(units) > 2 and units[2] in ("1/Å", "1/A"):
            k_pixel_size = float(data.sampling[2])
            k_calibrated = True
        data = data.array
    return data, title, pixel_size, k_pixel_size, k_calibrated


def pack_float32_halves(x: np.ndarray, y: np.ndarray) -> bytes:
    """Two arrays packed as concatenated float32 bytes."""
    return np.concatenate([x, y]).astype(np.float32).tobytes()


def build_analysis_mask(
    shape: tuple[int, int],
    regions: list[dict],
    center: tuple[float, float],
) -> np.ndarray | None:
    """Boolean exclusion mask from disk and wedge regions."""
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
            theta = np.degrees(np.arctan2(rows - center_row, cols - center_col)) % 360.0
            start = float(region["start_deg"]) % 360.0
            end = float(region["end_deg"]) % 360.0
            mask |= (
                (theta >= start) & (theta <= end)
                if start <= end
                else ((theta >= start) | (theta <= end))
            )
    return mask


# --- Radial and azimuthal profiles ---
def corrected_radius(
    d_row,
    d_col,
    *,
    ellipse_ratio: float = 1.0,
    ellipse_angle: float = 0.0,
    ellipse_corrected: bool = False,
):
    """Radius with optional elliptical-distortion correction."""
    if not ellipse_corrected or ellipse_ratio == 1.0:
        return np.hypot(d_row, d_col)
    angle = math.radians(ellipse_angle)
    major = d_col * math.cos(angle) + d_row * math.sin(angle)
    minor = -d_col * math.sin(angle) + d_row * math.cos(angle)
    return np.hypot(major / ellipse_ratio, minor)


def radial_profile_px(
    frame: np.ndarray,
    *,
    center: tuple[float, float],
    n_bins: int | None = None,
    max_radius: float | None = None,
    mask: np.ndarray | None = None,
    angular_range: tuple[float, float] | None = None,
    ellipse_ratio: float = 1.0,
    ellipse_angle: float = 0.0,
    ellipse_corrected: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Radial intensity profile in detector pixels."""
    n_rows, n_cols = frame.shape
    center_row, center_col = float(center[0]), float(center[1])
    if max_radius is None:
        max_radius = float(
            min(center_row, center_col, (n_rows - 1) - center_row, (n_cols - 1) - center_col)
        )
    max_radius = float(max(1.0, max_radius))
    n_bins = max(1, int(round(max_radius))) if n_bins is None else int(max(1, n_bins))

    rows = np.arange(n_rows, dtype=np.float64)[:, None]
    cols = np.arange(n_cols, dtype=np.float64)[None, :]
    d_row, d_col = rows - center_row, cols - center_col
    radii = corrected_radius(
        d_row,
        d_col,
        ellipse_ratio=ellipse_ratio,
        ellipse_angle=ellipse_angle,
        ellipse_corrected=ellipse_corrected,
    )
    flat_r = radii.ravel()
    flat_i = frame.astype(np.float64).ravel()
    keep = None if mask is None else ~mask.ravel()
    if angular_range is not None:
        theta = np.degrees(np.arctan2(d_row, d_col)).ravel() % 360.0
        start, end = float(angular_range[0]) % 360.0, float(angular_range[1]) % 360.0
        wedge = (
            (theta >= start) & (theta <= end)
            if start <= end
            else ((theta >= start) | (theta <= end))
        )
        keep = wedge if keep is None else keep & wedge
    if keep is not None:
        flat_r, flat_i = flat_r[keep], flat_i[keep]

    edges = np.linspace(0.0, max_radius, n_bins + 1)
    indices = np.digitize(flat_r, edges) - 1
    inside = (indices >= 0) & (indices < n_bins)
    indices = indices[inside]
    values = flat_i[inside]

    counts = np.bincount(indices, minlength=n_bins).astype(np.float64)
    sums = np.bincount(indices, weights=values, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        intensity = np.where(counts > 0, sums / counts, 0.0)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    return bin_centers.astype(np.float32), intensity.astype(np.float32)


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
    """Per-sector counts and intensity sums around one ring."""
    frame = frame.astype(np.float64)
    center_row, center_col = center
    n_rows, n_cols = frame.shape
    rows = np.arange(n_rows, dtype=np.float64)[:, None]
    cols = np.arange(n_cols, dtype=np.float64)[None, :]
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
    weight = intensity - intensity.min()
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


def texture_from_profile(
    theta_deg: np.ndarray, intensity: np.ndarray, *, return_profile: bool = False
) -> dict:
    """Texture strength and preferred angle from an azimuthal profile."""
    pedestal_free = intensity.astype(np.float64) - float(intensity.min())
    total = pedestal_free.sum()
    if total <= 0:
        strength, angle = 0.0, 0.0
    else:
        component = np.sum(pedestal_free * np.exp(2j * np.radians(theta_deg)))
        strength = float(abs(component) / total)
        angle = float(np.degrees(np.angle(component)) / 2.0 % 180.0)
    report = {"strength": strength, "angle_deg": angle}
    if return_profile:
        report["profile"] = (theta_deg, intensity)
    return report


# --- Ellipse and background fitting ---
def fit_ellipse_from_sectors(
    theta_centers: np.ndarray,
    counts: np.ndarray,
    weight_sum: np.ndarray,
    weighted_radius_sum: np.ndarray,
) -> dict:
    """Ellipse ratio and angle from ring-sector radii."""
    valid = (counts >= 10) & (weight_sum > 0)
    if valid.sum() < 8:
        raise ValueError(
            f"could not fit ellipse: ring found in {int(valid.sum())} sectors, need >= 8; "
            "check the ring radius and center"
        )
    radii_by_theta = weighted_radius_sum[valid] / weight_sum[valid]
    theta = np.radians(theta_centers)[valid]
    design = np.column_stack([np.ones_like(theta), np.cos(2 * theta), np.sin(2 * theta)])
    (mean_radius, cosine, sine), *_ = np.linalg.lstsq(design, radii_by_theta, rcond=None)
    epsilon = math.hypot(cosine, sine) / mean_radius
    ratio = (1.0 + epsilon) / (1.0 - epsilon) if epsilon < 1.0 else float("inf")
    angle = (0.5 * math.degrees(math.atan2(sine, cosine))) % 180.0
    residual = radii_by_theta - design @ np.array([mean_radius, cosine, sine])
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
    """Smooth background under a radial profile, excluding peak windows."""
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
def fit_gaussian_spot(
    frame: np.ndarray,
    row: float,
    col: float,
    *,
    half_window: int,
) -> dict | None:
    """Subpixel 2D Gaussian fit around a spot."""
    frame = np.asarray(frame, dtype=np.float32)
    half = max(4, int(half_window))
    row0, col0 = int(round(row)), int(round(col))
    row_lo, row_hi = max(0, row0 - half), min(frame.shape[0], row0 + half + 1)
    col_lo, col_hi = max(0, col0 - half), min(frame.shape[1], col0 + half + 1)
    patch = frame[row_lo:row_hi, col_lo:col_hi].astype(np.float64)
    if patch.shape[0] < 5 or patch.shape[1] < 5:
        return None
    try:
        from scipy.optimize import OptimizeWarning, curve_fit
    except Exception:
        return None

    n_rows, n_cols = patch.shape
    row_grid, col_grid = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")

    def gaussian_2d(coords, amplitude, row_center, col_center, sigma_row, sigma_col, offset):
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
                gaussian_2d,
                (row_grid, col_grid),
                patch.ravel(),
                p0=initial,
                maxfev=5000,
            )
    except Exception:
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


def _gaussian_peak(radius, amplitude, center, sigma, offset):
    return amplitude * np.exp(-0.5 * ((radius - center) / sigma) ** 2) + offset


def _pseudo_voigt_peak(radius, amplitude, center, sigma, offset, eta):
    gamma = sigma * 2.3548 / 2.0
    lorentzian = 1.0 / (1.0 + ((radius - center) / gamma) ** 2)
    gaussian_part = np.exp(-0.5 * ((radius - center) / sigma) ** 2)
    return amplitude * (eta * lorentzian + (1.0 - eta) * gaussian_part) + offset


def _ring_fit_window(radius_guess: float, centers: list[float], window: float | None) -> float:
    if window is not None:
        return float(window)
    gaps = [abs(radius_guess - center) for center in centers if center != radius_guess]
    return max(6.0, min(gaps) / 2.0) if gaps else max(6.0, 0.2 * radius_guess)


def _fit_ring_peak(
    radii_px: np.ndarray,
    intensity: np.ndarray,
    *,
    radius_guess: float,
    half_width: float,
    model: str,
) -> dict | None:
    from scipy.optimize import OptimizeWarning, curve_fit

    in_window = (radii_px >= radius_guess - half_width) & (radii_px <= radius_guess + half_width)
    radius_window = radii_px[in_window].astype(np.float64)
    intensity_window = intensity[in_window].astype(np.float64)
    if radius_window.size < 5:
        return None

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
    peak_model = _gaussian_peak if model == "gaussian" else _pseudo_voigt_peak

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        fit_params, _ = curve_fit(
            peak_model,
            radius_window,
            intensity_window,
            p0=initial,
            bounds=bounds,
            maxfev=5000,
        )

    residual = intensity_window - peak_model(radius_window, *fit_params)
    ss_tot = float(np.sum((intensity_window - intensity_window.mean()) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / ss_tot if ss_tot > 0 else 0.0
    amplitude = float(fit_params[0])
    fit_radius = float(fit_params[1])
    sigma = float(abs(fit_params[2]))
    return {
        "raw_radius_px": float(radius_guess),
        "radius_px": fit_radius,
        "intensity": amplitude,
        "fwhm_px": 2.3548 * sigma,
        "intensity_integrated": amplitude * sigma * math.sqrt(2.0 * math.pi),
        "fit_quality": float(r_squared),
    }


def fit_ring_peaks(
    radii_px: np.ndarray,
    intensity: np.ndarray,
    rings,
    *,
    model: str = "gaussian",
    window: float | None = None,
) -> list[dict | None]:
    """Fit one radial peak for each ring record."""
    if model not in RING_FIT_MODELS:
        raise ValueError(f"model must be one of {RING_FIT_MODELS}, got {model!r}")
    if window is not None and window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    try:
        import scipy.optimize  # noqa: F401
    except ImportError as exc:
        raise ImportError("fit_ring_peaks needs scipy; install it to fit ring peaks") from exc

    centers = sorted(float(ring["radius_px"]) for ring in rings)
    updates = []
    for ring in rings:
        radius_guess = float(ring["radius_px"])
        half_width = _ring_fit_window(radius_guess, centers, window)
        try:
            updates.append(
                _fit_ring_peak(
                    radii_px,
                    intensity,
                    radius_guess=radius_guess,
                    half_width=half_width,
                    model=model,
                )
            )
        except Exception:
            updates.append(None)
    return updates


# --- Measurement export ---
def spot_measurement_record(spot: dict) -> dict:
    """Export row for one spot record."""
    return {
        "id": spot.get("id"),
        "kind": "spot",
        "raw_row": spot.get("raw_row"),
        "raw_col": spot.get("raw_col"),
        "row": spot.get("row"),
        "col": spot.get("col"),
        "row_err": spot.get("row_err"),
        "col_err": spot.get("col_err"),
        "r_pixels": spot.get("r_pixels"),
        "r_pixels_err": spot.get("r_pixels_err"),
        "g_inv_angstrom": spot.get("g_magnitude"),
        "g_inv_angstrom_err": spot.get("g_magnitude_err"),
        "d_angstrom": spot.get("d_spacing"),
        "d_angstrom_err": spot.get("d_spacing_err"),
        "angle_deg": spot.get("angle_deg"),
        "angle_deg_err": spot.get("angle_deg_err"),
        "intensity": spot.get("intensity"),
        "fit_quality": spot.get("fit_quality"),
        "fwhm_px": None,
        "fwhm_inv_angstrom": None,
        "intensity_integrated": None,
        "hkl": spot.get("hkl", ""),
        "note": spot.get("note", ""),
    }


def ring_measurement_record(ring: dict) -> dict:
    """Export row for one ring record."""
    return {
        "id": ring.get("id"),
        "kind": "ring",
        "raw_row": None,
        "raw_col": None,
        "row": None,
        "col": None,
        "row_err": None,
        "col_err": None,
        "r_pixels": ring.get("radius_px"),
        "r_pixels_err": None,
        "g_inv_angstrom": ring.get("g_magnitude"),
        "g_inv_angstrom_err": None,
        "d_angstrom": ring.get("d_spacing"),
        "d_angstrom_err": None,
        "angle_deg": None,
        "angle_deg_err": None,
        "intensity": ring.get("intensity"),
        "fit_quality": ring.get("fit_quality"),
        "fwhm_px": ring.get("fwhm_px"),
        "fwhm_inv_angstrom": ring.get("fwhm_inv_angstrom"),
        "intensity_integrated": ring.get("intensity_integrated"),
        "hkl": ring.get("hkl", ""),
        "note": ring.get("note", ""),
    }


def build_measurement_records(spots, rings) -> list[dict]:
    """Export rows for all spots and rings."""
    return [spot_measurement_record(spot) for spot in spots] + [
        ring_measurement_record(ring) for ring in rings
    ]


def measurement_metadata(state) -> dict:
    """Export metadata block from widget state values."""
    return {
        "widget_name": "ShowDiffraction",
        "center_row": state.get("center_row"),
        "center_col": state.get("center_col"),
        "center_method": state.get("center_method", ""),
        "k_pixel_size_inv_angstrom_per_px": state.get("k_pixel_size"),
        "calibrated": bool(state.get("k_calibrated")),
        "calibration_source": state.get("calibration_source", "none"),
        "calibration_ref_d_angstrom": state.get("calibration_ref_d", 0.0),
        "calibration_ref_radius_px": state.get("calibration_ref_radius", 0.0),
        "mask_regions": state.get("mask_regions", []),
        "background_subtracted": bool(state.get("profile_subtract_background")),
    }


def write_measurement_file(path, records, metadata) -> pathlib.Path:
    """Write measurement records to CSV or JSON."""
    path = pathlib.Path(path)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps({"metadata": metadata, "measurements": records}, indent=2))
    else:
        with open(path, "w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=MEASUREMENT_COLUMNS)
            writer.writeheader()
            writer.writerows(records)
    return path
