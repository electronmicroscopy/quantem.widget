"""Display-only denoise and binning for sparse scientific maps.

Sparse EDS and low-dose diffraction are hard to read raw: single-count speckle
hides the signal that is plainly there after a modest bin and a Poisson-aware
smooth. These helpers make that readable VIEW without ever touching the stored
data: every function takes an array in and returns a new float32 array out, so
the widget buffer keeps its raw counts and ``mode="none"`` returns to them.

ShowDiffraction calls :func:`apply_display_filter` for its detection and
display views. The same function is the reference that the browser port in
``js/display/filter.ts`` (Show2D / Show3D denoise menu) is tested against.

House rule: defaults are lossless (``mode="none"``, ``spatial_bin=1``) and any
active reduction is announced once through :func:`format_display_filter_banner`.
"""

import numpy as np

DISPLAY_FILTER_MODES = ("none", "gaussian", "anscombe", "nlm")
_IDENTITY_MODES = {"none", "off", "raw", ""}


def _normalize_mode(mode: str) -> str:
    """Canonical lowercase mode name; identity spellings collapse to none."""
    mode = str(mode).strip().lower()
    return "none" if mode in _IDENTITY_MODES else mode


def _anscombe_gauss(image: np.ndarray, sigma: float) -> np.ndarray:
    """Anscombe variance stabilize, Gaussian smooth, inverse (Poisson-like)."""
    from scipy import ndimage

    image = np.asarray(image, dtype=np.float32)
    scale = float(np.percentile(image, 99.5) + 1e-9)
    counts = np.clip(image / scale * 30.0, 0, None)  # pseudo-counts
    stabilized = 2.0 * np.sqrt(counts + 3.0 / 8.0)
    stabilized = ndimage.gaussian_filter(stabilized, max(1.0, float(sigma) * 0.85))
    inverse = np.clip((stabilized * 0.5) ** 2 - 3.0 / 8.0, 0, None)
    return (inverse * scale / 30.0).astype(np.float32)


def _bin2(image: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """2x spatial bin (~sqrt(4) SNR), optional light smooth, zoom back to shape."""
    from scipy import ndimage

    image = np.asarray(image, dtype=np.float32)
    n_rows, n_cols = image.shape[-2:]
    binned = ndimage.zoom(image, 0.5, order=1)
    if sigma is not None:
        binned = ndimage.gaussian_filter(binned, max(1.0, float(sigma) / 2.5))
    return ndimage.zoom(
        binned, (n_rows / binned.shape[0], n_cols / binned.shape[1]), order=1
    ).astype(np.float32)


def _bin_passes(image: np.ndarray, spatial_bin: int) -> np.ndarray:
    """One ``_bin2`` pass for ``spatial_bin=2``, two for 4: the binning before a method smooths."""
    if spatial_bin >= 2:
        image = _bin2(image, None)
    if spatial_bin == 4:
        image = _bin2(image, None)
    return image


def _nlm(image: np.ndarray) -> np.ndarray:
    """Poisson non-local means: variance-stabilize, patch-average, invert."""
    from skimage.restoration import denoise_nl_means

    image = np.asarray(image, dtype=np.float32)
    positive = image[image > 0]
    if positive.size == 0:
        return image.copy()

    # Integer counts times a gain make the smallest positive value one count;
    # other data gets the 99.5th-percentile pseudo-count scale of _anscombe_gauss.
    gain = float(positive.min())
    ratio = positive / gain
    if not np.allclose(ratio, np.round(ratio), atol=1e-3):
        gain = float(np.percentile(positive, 99.5)) / 30.0

    counts = np.clip(image, 0.0, None) / gain
    stabilized = 2.0 * np.sqrt(counts + 3.0 / 8.0)
    smoothed = denoise_nl_means(
        stabilized, patch_size=5, patch_distance=6, h=0.8, sigma=1.0, fast_mode=True
    )
    inverse = np.clip((smoothed * 0.5) ** 2 - 3.0 / 8.0, 0.0, None)
    return (inverse * gain).astype(np.float32)


def apply_display_filter(
    image: np.ndarray,
    *,
    mode: str = "none",
    sigma: float = 4.0,
    spatial_bin: int = 1,
) -> np.ndarray:
    """Display-only denoise/smooth for a 2D map. Does not invent counts.

    The input array is never modified: the return value is a new float32
    array intended for the display path only (before contrast/colormap).
    Reconstruction and analysis must keep using the raw stored data.

    Parameters
    ----------
    image
        2D map, shape ``(n_rows, n_cols)``. Any numeric dtype.
    mode
        - ``"none"`` (also ``"off"``/``"raw"``): identity, the default.
        - ``"gaussian"``: Gaussian smooth of width ``sigma``. With
          ``spatial_bin`` >= 2 it smooths on the binned grid with the lighter
          ``max(1, sigma/2.5)`` kernel.
        - ``"anscombe"``: Anscombe transform, Gaussian, inverse; respects
          Poisson statistics of count data. With ``spatial_bin`` >= 2 the
          smoothing width becomes ``max(2, sigma*0.75)`` on the binned map.
        - ``"nlm"``: Poisson non-local means, variance-stabilized patch
          averaging that keeps spots sharp (requires scikit-image).
    sigma
        Smoothing scale in pixels for the Gaussian/Anscombe modes.
    spatial_bin
        2x bin passes for SNR: 1 (off), 2, or 4. The result is zoomed back to
        the input shape.

    Returns
    -------
    np.ndarray
        Filtered float32 view array with the input's (n_rows, n_cols) shape.
    """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(
            "apply_display_filter expects a 2D (n_rows, n_cols) map; "
            f"got shape {image.shape}. Filter stacks/frames one 2D slice at a time."
        )
    if spatial_bin not in (1, 2, 4):
        raise ValueError(f"spatial_bin must be 1, 2, or 4; got {spatial_bin!r}")
    requested = mode
    mode = _normalize_mode(mode)
    filtered = image.astype(np.float32, copy=True)
    sigma = float(sigma)
    if mode == "gaussian":
        # Binned gaussian smooths on the binned grid with the lighter
        # max(1, sigma/2.5) kernel, then zooms back.
        if spatial_bin == 4:
            return _bin2(_bin2(filtered, None), sigma)
        if spatial_bin == 2:
            return _bin2(filtered, sigma)
        from scipy import ndimage

        return ndimage.gaussian_filter(filtered, sigma).astype(np.float32)
    if mode == "anscombe":
        # Best practical stack for sparse counts: bin for SNR, then Poisson VST
        binned_sigma = max(2.0, sigma * 0.75) if spatial_bin >= 2 else sigma
        return _anscombe_gauss(_bin_passes(filtered, spatial_bin), binned_sigma)
    if mode not in ("none", "nlm"):
        raise ValueError(f"mode must be one of {'|'.join(DISPLAY_FILTER_MODES)} (or 'off'/'raw'); got {requested!r}")
    # none and nlm apply the bin knob as plain pre-passes before the method.
    filtered = _bin_passes(filtered, spatial_bin)
    return _nlm(filtered) if mode == "nlm" else filtered


def format_display_filter_banner(
    mode: str,
    sigma: float,
    spatial_bin: int = 1,
) -> str:
    """One-line notice for an ACTIVE display reduction.

    Returns an empty string when the filter is identity, so callers can
    ``print`` unconditionally. Announcing reductions is a house rule: a user
    must always know their view is filtered and how to get raw counts back.

    Examples
    --------
    >>> from quantem.widget.utils.display_filter import format_display_filter_banner
    >>> format_display_filter_banner("anscombe", 8, spatial_bin=2)
    "denoise: anscombe σ=8 bin2 (set denoise='none' for raw counts)"
    >>> format_display_filter_banner("none", 4)
    ''
    """
    mode = _normalize_mode(mode)
    if mode == "none" and int(spatial_bin) == 1:
        return ""
    parts = ["raw"] if mode == "none" else [mode, f"σ={sigma:g}"]
    if int(spatial_bin) > 1:
        parts.append(f"bin{int(spatial_bin)}")
    return "denoise: " + " ".join(parts) + " (set denoise='none' for raw counts)"
