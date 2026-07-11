"""Display-only denoise and binning for sparse scientific maps.

Sparse EDS and low-dose STEM maps are hard to read raw: single-count speckle
hides the lattice-periodic signal that is plainly there after a modest bin and
a Poisson-aware smooth. These helpers make that readable VIEW without ever
touching the stored data: every function takes an array in and returns a new
float32 array out, so the Dataset / widget buffer keeps its raw counts and a
user can always return to them with ``filter="none"``.

The math is copied from the drift-paper figure pipeline
(``quantem.imaging.drift.figure.correct_3d``: ``_bin2``, ``_anscombe_gauss``,
``denoise_map``, ``_resolve_cmap("magenta")``, ``blend_map_on_haadf``) so that
what a user sees live in a widget matches the published figures.

House rule: defaults are lossless (``filter="none"``, ``spatial_bin=1``) and
any active reduction must be announced once through
:func:`format_display_filter_banner`. No silent data reduction, ever.
"""

from __future__ import annotations

import numpy as np

DISPLAY_FILTER_MODES = (
    "none",
    "gaussian",
    "bin2",
    "anscombe",
    "bin2_anscombe",
    "bin4_anscombe",
    "tv",
    "denova",
    "denova_tv",
    "denova_tv12",
)
_IDENTITY_MODES = {"none", "off", "raw", ""}


def _normalize_mode(filter: str) -> str:
    """Canonical lowercase mode name; identity spellings collapse to none."""
    mode = str(filter).strip().lower().replace("-", "_")
    if mode in _IDENTITY_MODES:
        return "none"
    aliases = {
        "bin2anscombe": "bin2_anscombe",
        "bin_anscombe": "bin2_anscombe",
        "bin4anscombe": "bin4_anscombe",
        "poisson": "anscombe",
        "anscombe_gaussian": "anscombe",
        "denova_tv1_2": "denova_tv12",
    }
    return aliases.get(mode, mode)


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


def _denova(image: np.ndarray, method: str = "tv") -> np.ndarray:
    """Lab denova denoiser (auto-lambda). Requires the denova package."""
    try:
        from denova import denoise as denova_denoise
    except ImportError as exc:
        raise ImportError(
            "display filter 'denova*' requires the denova package; install it "
            "or pick one of: " + ", ".join(m for m in DISPLAY_FILTER_MODES if "denova" not in m)
        ) from exc
    result = denova_denoise(np.asarray(image, dtype=np.float32), method=method)
    return np.asarray(result.output, dtype=np.float32)


def apply_display_filter(
    image: np.ndarray,
    *,
    filter: str = "none",
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
    filter
        Display filter mode:

        - ``"none"`` (also ``"off"``/``"raw"``): identity, the default.
        - ``"gaussian"``: plain Gaussian smooth of width ``sigma``.
        - ``"bin2"``: 2x spatial bin for SNR, light smooth, zoom back.
        - ``"anscombe"``: Anscombe transform, Gaussian, inverse; respects
          Poisson statistics of count data.
        - ``"bin2_anscombe"``: bin2 then Anscombe; the recommended stack for
          sparse EDS maps.
        - ``"bin4_anscombe"``: two bin2 passes then Anscombe.
        - ``"tv"``: total-variation denoise (requires scikit-image).
        - ``"denova"`` / ``"denova_tv"`` / ``"denova_tv12"``: lab denova
          denoiser when the optional package is installed.
    sigma
        Smoothing scale in pixels for the Gaussian/Anscombe modes.
    spatial_bin
        Extra 2x bin passes applied BEFORE the filter mode: 1 (off), 2, or 4.
        Independent of the binning already inside ``bin2*`` modes.

    Returns
    -------
    np.ndarray
        Filtered float32 view array with the input's (n_rows, n_cols) shape.

    Examples
    --------
    >>> import numpy as np
    >>> from quantem.widget.utils.display_filter import apply_display_filter
    >>> counts = np.random.poisson(0.3, (256, 256)).astype(np.float32)
    >>> view = apply_display_filter(counts, filter="bin2_anscombe", sigma=8)
    >>> view.shape == counts.shape
    True
    """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(
            "apply_display_filter expects a 2D (n_rows, n_cols) map; "
            f"got shape {image.shape}. Filter stacks/frames one 2D slice at a time."
        )
    if spatial_bin not in (1, 2, 4):
        raise ValueError(f"spatial_bin must be 1, 2, or 4; got {spatial_bin!r}")
    mode = _normalize_mode(filter)
    out = image.astype(np.float32, copy=True)
    if spatial_bin >= 2:
        out = _bin2(out, None)
    if spatial_bin == 4:
        out = _bin2(out, None)
    sigma = float(sigma)
    if mode == "none":
        return out
    if mode == "gaussian":
        from scipy import ndimage

        return ndimage.gaussian_filter(out, sigma).astype(np.float32)
    if mode == "bin2":
        return _bin2(out, sigma)
    if mode == "anscombe":
        return _anscombe_gauss(out, sigma)
    if mode == "bin2_anscombe":
        # Best practical stack for sparse EDS: bin for SNR, then Poisson VST
        return _anscombe_gauss(_bin2(out, None), max(2.0, sigma * 0.75))
    if mode == "bin4_anscombe":
        return _anscombe_gauss(_bin2(_bin2(out, None), None), max(2.0, sigma * 0.75))
    if mode == "tv":
        try:
            from skimage.restoration import denoise_tv_chambolle
        except ImportError as exc:
            raise ImportError("display filter 'tv' requires scikit-image") from exc
        weight = max(0.02, min(0.25, sigma / 50.0))
        return denoise_tv_chambolle(out.astype(np.float64), weight=weight).astype(np.float32)
    if mode == "denova":
        return _denova(out, method="tv")
    if mode == "denova_tv":
        return _denova(out, method="tv")
    if mode == "denova_tv12":
        return _denova(out, method="tv12")
    raise ValueError(
        "filter must be one of "
        + "|".join(DISPLAY_FILTER_MODES)
        + f" (or 'off'/'raw'); got {filter!r}"
    )


def format_display_filter_banner(
    filter: str,
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
    >>> format_display_filter_banner("bin2_anscombe", 8)
    "display: bin2_anscombe σ=8 (set display_filter='none' for raw counts)"
    >>> format_display_filter_banner("none", 4)
    ''
    """
    mode = _normalize_mode(filter)
    if mode == "none" and int(spatial_bin) == 1:
        return ""
    parts = [mode if mode != "none" else "raw"]
    if mode not in ("none",):
        sigma_text = f"{sigma:g}"
        parts.append(f"σ={sigma_text}")
    if int(spatial_bin) > 1:
        parts.append(f"bin{int(spatial_bin)}")
    return "display: " + " ".join(parts) + " (set display_filter='none' for raw counts)"


def magenta_cmap():
    """EDS magenta colormap whose high end stays pink-magenta, never white.

    White high-stops make bright chemistry columns look like HAADF signal
    instead of chemistry; this ramp keeps the hue readable at full intensity.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap

    Examples
    --------
    >>> from quantem.widget.utils.display_filter import magenta_cmap
    >>> cmap = magenta_cmap()
    >>> tuple(round(c, 2) for c in cmap(1.0)[:3]) != (1.0, 1.0, 1.0)
    True
    """
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "eds_magenta",
        [
            "#050008",  # near-black
            "#4a0038",
            "#9a0078",
            "#d020a8",
            "#ff40d0",  # bright magenta (not white)
            "#ff90e8",  # light pink-magenta peak
        ],
    )


def blend_map_on_haadf(
    map_01: np.ndarray,
    haadf_01: np.ndarray,
    *,
    alpha: float = 0.95,
    haadf_gain: float = 0.35,
    cmap=None,
) -> np.ndarray:
    """HAADF-modulated chemistry: bright lattice sites go colored, not white.

    A naive lerp keeps full-bright gray HAADF wherever the sparse map is weak,
    so Z-contrast columns stay white and read as structure, not chemistry.
    Instead: hue comes from the element map through ``cmap`` (which must not
    end in white), luma is map presence times HAADF structure so high-Z
    columns that also have counts light up in the map color, and map-empty
    regions keep only a dim gray lattice for context.

    Parameters
    ----------
    map_01
        Element map normalized to [0, 1], shape ``(n_rows, n_cols)``.
    haadf_01
        HAADF image normalized to [0, 1], same shape.
    alpha
        Overall chemistry opacity in [0, 1].
    haadf_gain
        Strength of the dim structural ghost in map-empty regions, in [0, 1].
    cmap
        Matplotlib colormap for the map hue. Default: :func:`magenta_cmap`.

    Returns
    -------
    np.ndarray
        float32 RGB image, shape ``(n_rows, n_cols, 3)``, values in [0, 1].

    Examples
    --------
    >>> import numpy as np
    >>> from quantem.widget.utils.display_filter import blend_map_on_haadf
    >>> rgb = blend_map_on_haadf(np.random.rand(64, 64), np.random.rand(64, 64))
    >>> rgb.shape
    (64, 64, 3)
    """
    map_01 = np.clip(np.asarray(map_01, dtype=np.float32), 0.0, 1.0)
    haadf_01 = np.clip(np.asarray(haadf_01, dtype=np.float32), 0.0, 1.0)
    gain = float(np.clip(haadf_gain, 0.0, 1.0))
    opacity = float(np.clip(alpha, 0.0, 1.0))
    if cmap is None:
        cmap = magenta_cmap()
    color = np.asarray(cmap(map_01)[..., :3], dtype=np.float32)
    # Map presence (slight gamma so mid counts still colorize columns)
    presence = np.power(map_01, 0.75) * opacity
    # Structure: never a pure white base; HAADF only brightens the color
    structure = 0.30 + 0.70 * haadf_01
    chemistry = color * (presence * structure)[..., None]
    # Dim lattice only where chemistry is absent (no white cores under map)
    ghost_weight = (1.0 - np.clip(presence * 1.35, 0.0, 1.0)) * (0.12 + 0.22 * gain)
    ghost = (haadf_01 * ghost_weight)[..., None]
    ghost_rgb = np.concatenate([ghost, ghost, ghost], axis=-1)
    return np.clip(chemistry + ghost_rgb, 0.0, 1.0)
