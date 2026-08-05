"""
Show3DSlices: ptycho-oriented oblique slice viewer.

Displays a top slice plus one arbitrary-angle vertical plane with interactive
sliders and a contextual 3D orientation view. All slicing happens in
JavaScript for instant response. A single widget may hold pages of comparable
volumes and keep slice geometry synchronized while the user switches pages.
"""
import base64
import copy
import gc
import io
import json
import math
import os
import pathlib
import tempfile
import warnings
from numbers import Real
from collections.abc import Mapping, Sequence
from typing import Self

import anywidget
import numpy as np
import traitlets

from quantem.widget.utils.array import to_numpy
from quantem.widget.show3d import (
    _crop_stack,
    _normalize_crop,
    _normalize_padding,
    _pad_stack,
)
# Config helpers live in a widget-free module so Show3D can share them without a
# circular import. Re-exported here for back-compat (notebooks import them from
# this module).
from quantem.widget.utils.recon_config import (
    _centered_crop_for_shape,
    _config_float,
    _config_get,
    _is_default_pixel_size,
    _load_quantem_config,
    _normalize_rotation_deg,
    _pixel_size_from_quantem_config,
    _post_crop_from_quantem_config,
    _rotate_stack_inplane,
)
from quantem.widget.show2d import _reject_unknown_kwargs
from quantem.widget.utils.state_io import (
    resolve_widget_version,
    save_state_file,
    unwrap_state_payload,
)
from quantem.widget.utils.ui import UiMode, resolve_ui_mode


# Names that JS bundle's GPUColormapEngine knows. Keep in sync with
# js/colormaps.ts COLORMAPS table.
_VALID_CMAPS = frozenset({
    "inferno", "viridis", "plasma", "magma", "hot", "gray", "hsv", "turbo",
    "cividis", "RdBu", "RdBu_r", "seismic", "twilight", "twilight_shifted",
})
_VALID_SLICE_ALIGNMENT = frozenset({"off", "auto", "manual"})
_MAX_PLAYBACK_FPS = 30.0
# Registration settings used by the established full-resolution experimental
# reconstructed-stack analysis. The 20x local matrix-DFT refinement follows
# Guizar-Sicairos et al., Opt. Lett. 33, 156-158 (2008),
# https://doi.org/10.1364/OL.33.000156.
_SLICE_ALIGNMENT_HIGHPASS_SIGMA_PX = 12.0
_SLICE_ALIGNMENT_UPSAMPLE_FACTOR = 20
_SLICE_ALIGNMENT_DFT_REGION_FACTOR = 1.5


def _is_show3dslices_page_dict(value: object) -> bool:
    """Return True for an explicit Show3DSlices page descriptor."""
    if not isinstance(value, Mapping):
        return False
    return any(
        key in value
        for key in ("volumes", "volume", "panels", "data", "array")
    )


def _as_show3dslices_panel_array(value: object, *, context: str) -> np.ndarray:
    """Normalize one page payload to ``(panel, z, y, x)``."""
    raw = value.array if hasattr(value, "array") else value
    if isinstance(raw, (list, tuple)) and raw and all(
        hasattr(item, "array") or hasattr(item, "shape") for item in raw
    ):
        arrays = [
            to_numpy(item.array if hasattr(item, "array") else item)
            for item in raw
        ]
        if not all(arr.ndim == 3 for arr in arrays):
            shapes = [arr.shape for arr in arrays]
            raise ValueError(f"{context} volumes must all be 3D, got {shapes}")
        arr = np.stack(arrays, axis=0)
    else:
        arr = to_numpy(raw)
    if arr.ndim == 3:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 4:
        raise ValueError(
            f"{context} must contain a 3D volume or 4D panel stack, got shape {arr.shape}"
        )
    return arr


def _normalise_show3dslices_pages(
    data: object,
    *,
    panel_titles: Sequence[str] | None,
    page_labels: Sequence[str | None] | None,
) -> tuple[object, list[str] | None, int, int, list[str]]:
    """Flatten page-aware inputs while preserving the legacy 4D panel path."""
    explicit_pages = (
        isinstance(data, (list, tuple))
        and len(data) > 0
        and all(_is_show3dslices_page_dict(item) for item in data)
    )
    if explicit_pages:
        page_arrays: list[np.ndarray] = []
        inferred_labels: list[str] = []
        flattened_titles: list[str] = []
        panels_per_page: int | None = None
        panel_shape: tuple[int, int, int] | None = None
        for page_idx, page in enumerate(data):
            raw = page.get(
                "volumes",
                page.get(
                    "volume",
                    page.get("panels", page.get("data", page.get("array"))),
                ),
            )
            if raw is None:
                raise ValueError(
                    f"Show3DSlices page {page_idx} is missing volumes/volume/panels/data/array"
                )
            arr = _as_show3dslices_panel_array(
                raw,
                context=f"Show3DSlices page {page_idx}",
            )
            if panels_per_page is None:
                panels_per_page = int(arr.shape[0])
                panel_shape = tuple(int(v) for v in arr.shape[1:])
            elif int(arr.shape[0]) != panels_per_page:
                raise ValueError(
                    "Every Show3DSlices page must contain the same panel count; "
                    f"page 0 has {panels_per_page}, page {page_idx} has {arr.shape[0]}"
                )
            if tuple(int(v) for v in arr.shape[1:]) != panel_shape:
                raise ValueError(
                    "Every Show3DSlices page panel must share one (z, y, x) shape; "
                    f"expected {panel_shape}, got {arr.shape[1:]}"
                )
            page_arrays.append(arr)
            inferred_labels.append(
                str(page.get("title") or page.get("label") or f"Page {page_idx + 1}")
            )
            local_titles = page.get("panel_titles", page.get("labels"))
            if local_titles is not None:
                if len(local_titles) != int(arr.shape[0]):
                    raise ValueError(
                        f"Show3DSlices page {page_idx} panel_titles length "
                        f"({len(local_titles)}) must match {arr.shape[0]} panels"
                    )
                flattened_titles.extend(str(value) for value in local_titles)
            elif panel_titles is not None:
                if len(panel_titles) == int(arr.shape[0]):
                    flattened_titles.extend(str(value) for value in panel_titles)
                elif len(panel_titles) == len(data) * int(arr.shape[0]):
                    start = page_idx * int(arr.shape[0])
                    flattened_titles.extend(
                        str(value)
                        for value in panel_titles[start : start + int(arr.shape[0])]
                    )
                else:
                    raise ValueError(
                        "panel_titles for paged Show3DSlices must have length "
                        f"panels_per_page ({arr.shape[0]}) or total panels "
                        f"({len(data) * int(arr.shape[0])}), got {len(panel_titles)}"
                    )
            else:
                flattened_titles.extend(
                    "Volume" if int(arr.shape[0]) == 1 else f"Panel {idx + 1}"
                    for idx in range(int(arr.shape[0]))
                )
        n_pages = len(page_arrays)
        labels = [
            "" if value is None else str(value)
            for value in (page_labels if page_labels is not None else inferred_labels)
        ]
        if len(labels) != n_pages:
            raise ValueError(
                f"page_labels length ({len(labels)}) must match n_pages ({n_pages})"
            )
        flattened = np.concatenate(page_arrays, axis=0)
        return flattened, flattened_titles, n_pages, int(panels_per_page or 1), labels

    raw = data.array if hasattr(data, "array") else data
    if (
        page_labels is not None
        and isinstance(raw, (list, tuple))
        and raw
        and all(hasattr(item, "array") or hasattr(item, "shape") for item in raw)
    ):
        arrays = [
            to_numpy(item.array if hasattr(item, "array") else item)
            for item in raw
        ]
        if not all(arr.ndim == 3 for arr in arrays):
            raise ValueError("Show3DSlices page lists must contain 3D volumes")
        raw = np.stack(arrays, axis=0)
    try:
        arr = to_numpy(raw)
    except (TypeError, ValueError):
        return data, list(panel_titles) if panel_titles is not None else None, 1, 0, []

    if arr.ndim == 5:
        n_pages, panels_per_page = int(arr.shape[0]), int(arr.shape[1])
        labels = [
            "" if value is None else str(value)
            for value in (
                page_labels
                if page_labels is not None
                else [f"Page {idx + 1}" for idx in range(n_pages)]
            )
        ]
        if len(labels) != n_pages:
            raise ValueError(
                f"page_labels length ({len(labels)}) must match n_pages ({n_pages})"
            )
        if panel_titles is None:
            slot_titles = [
                "Volume" if panels_per_page == 1 else f"Panel {idx + 1}"
                for idx in range(panels_per_page)
            ]
            titles = slot_titles * n_pages
        elif len(panel_titles) == panels_per_page:
            titles = [str(value) for _ in range(n_pages) for value in panel_titles]
        elif len(panel_titles) == n_pages * panels_per_page:
            titles = [str(value) for value in panel_titles]
        else:
            raise ValueError(
                "panel_titles for paged Show3DSlices must have length "
                f"{panels_per_page} or {n_pages * panels_per_page}, got {len(panel_titles)}"
            )
        return (
            arr.reshape(n_pages * panels_per_page, *arr.shape[2:]),
            titles,
            n_pages,
            panels_per_page,
            labels,
        )

    if arr.ndim == 4 and page_labels is not None:
        n_pages = int(arr.shape[0])
        labels = ["" if value is None else str(value) for value in page_labels]
        if len(labels) != n_pages:
            raise ValueError(
                f"page_labels length ({len(labels)}) must match n_pages ({n_pages})"
            )
        if panel_titles is None:
            titles = ["Volume"] * n_pages
        elif len(panel_titles) == 1:
            titles = [str(panel_titles[0])] * n_pages
        elif len(panel_titles) == n_pages:
            titles = [str(value) for value in panel_titles]
        else:
            raise ValueError(
                "panel_titles for single-volume Show3DSlices pages must have "
                f"length 1 or {n_pages}, got {len(panel_titles)}"
            )
        return arr, titles, n_pages, 1, labels

    return data, list(panel_titles) if panel_titles is not None else None, 1, 0, []


def _finite_centered_2d(image: np.ndarray) -> np.ndarray:
    """Return a finite float32 image centered by its median."""
    out = np.asarray(image, dtype=np.float32)
    if not np.isfinite(out).all():
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = out - np.median(out)
    return out


def _fft_gaussian_highpass(image: np.ndarray, sigma: float) -> np.ndarray:
    """Subtract a reflect-boundary Gaussian blur using NumPy FFT primitives."""
    if sigma <= 0:
        return image
    rows, cols = image.shape
    if rows < 3 or cols < 3:
        return image
    pad = max(1, int(math.ceil(4.0 * sigma)))
    padded = np.pad(image, ((pad, pad), (pad, pad)), mode="reflect")
    fy = np.fft.fftfreq(padded.shape[0])[:, None]
    fx = np.fft.fftfreq(padded.shape[1])[None, :]
    gaussian = np.exp(-2.0 * (math.pi**2) * (sigma**2) * (fy * fy + fx * fx))
    blur_padded = np.fft.ifft2(np.fft.fft2(padded) * gaussian).real
    blur = blur_padded[pad : pad + rows, pad : pad + cols]
    return image - blur.astype(np.float32, copy=False)


def _slice_registration_image(image: np.ndarray) -> np.ndarray:
    """Preprocess one slice before adjacent-slice registration."""
    out = _finite_centered_2d(image)
    sigma = min(
        _SLICE_ALIGNMENT_HIGHPASS_SIGMA_PX,
        max(1.0, min(out.shape) / 6.0),
    )
    out = _fft_gaussian_highpass(out, sigma=sigma)
    rows, cols = out.shape
    window = np.outer(np.hanning(rows), np.hanning(cols)).astype(np.float32)
    return np.ascontiguousarray(out * window, dtype=np.float32)


def _upsampled_dft_2d(
    spectrum: np.ndarray,
    region_shape: tuple[int, int],
    upsample_factor: int,
    offsets: Sequence[float],
) -> np.ndarray:
    """Evaluate a small upsampled inverse-DFT region by matrix products."""
    rows, cols = spectrum.shape
    region_rows, region_cols = region_shape
    row_coords = np.arange(region_rows, dtype=np.float64)[:, None] - float(offsets[0])
    col_coords = np.arange(region_cols, dtype=np.float64)[:, None] - float(offsets[1])
    row_freq = np.fft.fftfreq(rows, d=float(upsample_factor))[None, :]
    col_freq = np.fft.fftfreq(cols, d=float(upsample_factor))[None, :]
    row_kernel = np.exp((-2j * math.pi) * row_coords * row_freq)
    col_kernel = np.exp((-2j * math.pi) * col_coords * col_freq)
    return row_kernel @ spectrum @ col_kernel.T


def _estimate_adjacent_shift_apply(
    ref: np.ndarray,
    mov: np.ndarray,
    *,
    ref_spectrum: np.ndarray | None = None,
    mov_spectrum: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Estimate the subpixel row/col shift that aligns ``mov`` to ``ref``."""
    if ref.shape != mov.shape:
        raise ValueError(f"registration shape mismatch: {ref.shape} != {mov.shape}")
    if ref.ndim != 2:
        raise ValueError(f"registration expects 2D slices, got {ref.ndim}D")
    if float(np.std(ref)) == 0.0 or float(np.std(mov)) == 0.0:
        return 0.0, 0.0, 0.0

    if ref_spectrum is None:
        ref_spectrum = np.fft.fft2(ref)
    if mov_spectrum is None:
        mov_spectrum = np.fft.fft2(mov)
    product = ref_spectrum * np.conj(mov_spectrum)
    corr = np.fft.ifft2(product)
    coarse_peak = np.unravel_index(int(np.argmax(np.abs(corr))), corr.shape)
    shape = np.asarray(corr.shape, dtype=np.float64)
    shifts = np.asarray(coarse_peak, dtype=np.float64)
    midpoint = np.fix(shape / 2.0)
    wrap = shifts > midpoint
    shifts[wrap] -= shape[wrap]

    upsample = _SLICE_ALIGNMENT_UPSAMPLE_FACTOR
    shifts = np.round(shifts * upsample) / upsample
    region_size = int(math.ceil(upsample * _SLICE_ALIGNMENT_DFT_REGION_FACTOR))
    dft_shift = float(np.fix(region_size / 2.0))
    offsets = dft_shift - shifts * upsample
    refined_corr = np.conj(
        _upsampled_dft_2d(
            np.conj(product),
            (region_size, region_size),
            upsample,
            offsets,
        )
    )
    refined_peak = np.asarray(
        np.unravel_index(int(np.argmax(np.abs(refined_corr))), refined_corr.shape),
        dtype=np.float64,
    )
    shifts += (refined_peak - dft_shift) / upsample

    peak_value = float(np.abs(corr[coarse_peak]))
    norm = float(np.sqrt(np.sum(ref * ref) * np.sum(mov * mov)))
    quality = peak_value / norm if norm > 0 else 0.0
    return float(shifts[0]), float(shifts[1]), quality


def _linear_fit_r2(z: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> float:
    fit = slope * z + intercept
    resid = y - fit
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 1e-12:
        return 1.0 if ss_res <= 1e-12 else 0.0
    return max(0.0, min(1.0, 1.0 - ss_res / ss_tot))


def _apply_global_slice_alignment_stack(
    stack: np.ndarray,
    row_shift_px_per_slice: float,
    col_shift_px_per_slice: float,
) -> np.ndarray:
    """Return a display-aligned copy using nearest-edge bilinear sampling."""
    row_slope = float(row_shift_px_per_slice)
    col_slope = float(col_shift_px_per_slice)
    if abs(row_slope) < 1e-12 and abs(col_slope) < 1e-12:
        return stack
    nz, ny, nx = stack.shape
    row_grid = np.arange(ny, dtype=np.float32)[:, None]
    col_grid = np.arange(nx, dtype=np.float32)[None, :]
    center = (nz - 1) / 2.0
    out = np.empty_like(stack, dtype=np.float32)
    for idx in range(nz):
        row_src = np.clip(row_grid - (idx - center) * row_slope, 0, ny - 1)
        col_src = np.clip(col_grid - (idx - center) * col_slope, 0, nx - 1)
        r0 = np.floor(row_src).astype(np.intp)
        c0 = np.floor(col_src).astype(np.intp)
        r1 = np.minimum(r0 + 1, ny - 1)
        c1 = np.minimum(c0 + 1, nx - 1)
        tr = row_src - r0
        tc = col_src - c0
        slc = stack[idx]
        out[idx] = (
            (slc[r0, c0] * (1.0 - tc) + slc[r0, c1] * tc) * (1.0 - tr)
            + (slc[r1, c0] * (1.0 - tc) + slc[r1, c1] * tc) * tr
        )
    return out


class Show3DSlices(anywidget.AnyWidget):
    """
    Linked top-slice and oblique vertical slice viewer for 3D volumes.

    Renders an XY slice plus one arbitrary-angle vertical plane through a
    ``(nz, ny, nx)`` volume in linked canvases with synchronized context,
    optional FFT, and playback over depth or plane position. Pass a 4D array of
    shape ``(panel, nz, ny, nx)`` to compare several volumes with one panel
    selector while preserving the same slice, side-plane, contrast, and camera
    controls. Designed for multislice ptychography reconstructions and other
    small-to-medium 3D volumes where the operator wants to scrub through depth
    while marching an angled plane through lateral regions. The raw volume data
    is sent once over the Jupyter Comm channel and re-sliced in JS for each
    scrubber update.

    Features
    --------
    - Linked XY plus arbitrary-angle vertical slice canvases
    - Comparable volume pages with shared slice geometry and page playback
    - Playback over Z slice, oblique plane position, or both
    - Adjustable oblique endpoints, angle, and perpendicular position bounds
    - Log scale, percentile auto-contrast, manual ``vmin`` / ``vmax``
    - Per-plane statistics (opt-in via ``show_stats``)
    - Anisotropic scale bars via ``pixel_size_axes`` (e.g. nz << nxy multislice)
    - Depth-axis ``z_stretch`` for non-cubic volumes (CSS scaling, zero memory)
    - ``panel_width_px`` constructor sizing for compact or wide panel layouts
    - Display-only global slice post-alignment with Auto/manual row/col slopes
    - FFT panel with optional Hann window for the active plane
    - PNG / PDF / TIFF single-slice export with plane and index selection
    - JSON state save/load via ``state_dict`` / ``load_state_dict`` / ``save``
    - Explicit ``free`` to release RAM held by traitlets observers

    Config Contract
    ---------------
    The ``config`` argument is an explicit QuantEM reconstruction convenience,
    not a general-purpose transform registry. It accepts either a parsed
    ``config.json`` mapping or a path to that JSON file. Show3DSlices reads only
    the following keys:

    - ``reconstruction.slice_thickness_A`` and
      ``reconstruction.obj_sampling_A_per_px`` for ``pixel_size`` when
      ``pixel_size`` is not provided.
    - ``data.rotation_deg`` for ``rotation_deg`` when ``rotation_deg`` is not
      provided and ``apply_config_transforms`` is True.
    - ``object.cropped_shape`` for a centered post-rotation crop when
      ``post_crop`` is not provided and ``apply_config_transforms`` is True.
      If that key is absent, it falls back to
      ``reconstruction.obj_padding_px`` and then ``input.padding``.

    Explicit constructor arguments always win over config-derived values. To
    use only config sampling metadata, pass ``apply_config_transforms=False``.
    To force no config rotation or post-crop while still leaving transforms
    enabled, pass ``rotation_deg=0`` and/or ``post_crop=0`` explicitly.

    The spatial transform order is fixed and intentionally visible:
    ``crop`` -> ``padding`` -> ``rotation_deg`` -> ``post_crop``. The resulting
    float32 volume is then sent to the browser. No hidden JS-side rotation,
    crop, or precision-changing transform is applied from ``config``.

    Parameters
    ----------
    data : array_like
        3D array of shape ``(nz, ny, nx)`` or 4D array of shape
        ``(panel, nz, ny, nx)``. Also accepts a quantem ``Dataset3d`` (uses
        ``.array`` and ``.sampling`` automatically). A 5D array uses shape
        ``(page, panel, nz, ny, nx)``. A 4D stack becomes one-volume pages when
        ``page_labels`` is provided; without ``page_labels`` it retains the
        legacy multi-panel meaning.
    data_b : array_like, optional
        Back-compatible two-panel shortcut. When provided, ``data`` and
        ``data_b`` must be matching 3D volumes and the frontend exposes a
        panel selector.
    title : str, optional
        Title displayed above the viewer.
    panel_titles : sequence of str, optional
        Per-panel labels for 4D input or ``data_b`` comparison. Length must
        match the number of panels. For 5D pages, pass one title per panel slot
        or one title per flattened panel.
    page_labels : sequence of str, optional
        Labels for comparable volumes/pages. Page switching keeps the same
        top/side slice geometry, zoom, and display controls for direct depth
        comparison.
    ui_mode : {"interactive", "presentation", "report", "minimal"}, default "interactive"
        Shared viewer UI preset. Explicit ``show_*`` keyword arguments override
        preset values.
    show_title : bool, default True
        Show the top title row.
    cmap : str, default "plasma"
        Colormap name. One of the project's valid colormaps (``"inferno"``,
        ``"viridis"``, ``"magma"``, ``"gray"``, ``"plasma"``, ...).
    sampling : float or sequence of 3 floats, optional
        Voxel sampling for the scale bars. Pass a scalar for isotropic data, or a
        3-tuple `(pz, py, px)` for anisotropic data (e.g. multislice ptycho with
        z-thickness >> xy-sampling). Pairs with ``units``; ``nm`` is converted to
        Å. Matches the quantem ``Dataset`` / ``Show2D`` / ``Show4DSTEM`` convention.
    units : str or sequence of str, optional
        Unit(s) for ``sampling`` (e.g. ``"A"``, ``"nm"``). One value applies to
        all axes; a sequence sets each axis.
    pixel_size : float or sequence of 3 floats, optional
        Back-compatible scale-bar sampling in Å/pixel. Prefer ``sampling`` and
        ``units`` for new code. A 3-tuple is interpreted as ``(pz, py, px)``.
    config : mapping or path-like, optional
        Parsed QuantEM reconstruction ``config.json`` data, or a path to it.
        This is only used for the keys listed in "Config Contract" above.
        It can infer sampling metadata, scan-alignment rotation, and
        post-rotation FOV crop, but explicit constructor arguments override
        every inferred value.
    apply_config_transforms : bool, default True
        If True, use config-derived ``rotation_deg`` and ``post_crop`` only
        when those arguments are omitted. If False, ``config`` may still fill
        ``pixel_size`` but will not rotate or crop the volume.
    crop : int or tuple[int, int] or tuple[int, int, int, int], default 0
        Spatial crop applied before padding and rotation. Use an int to crop
        all sides, ``(rows, cols)`` for symmetric row/column cropping, or
        ``(top, bottom, left, right)`` for side-specific cropping.
    padding : int or tuple[int, int], default 0
        Spatial padding applied after crop and before rotation. This mirrors
        Show3D and is useful when rotating a scan-aligned view without clipping
        real signal at the canvas edge.
    pad_mode : {"median", "constant"}, default "median"
        Padding fill mode. ``"median"`` uses a single stack median so borders
        are contrast-neutral; ``"constant"`` fills with 0.
    rotation_deg : float, optional
        In-plane row/column rotation in degrees, applied after crop/padding.
        If omitted and ``config`` is provided, uses ``data.rotation_deg`` from
        the config; otherwise defaults to 0. Uses built-in float32 bilinear
        interpolation with nearest-edge fill, no SciPy dependency, and keeps
        the output shape fixed.
    post_crop : int or tuple[int, int] or tuple[int, int, int, int], optional
        Spatial crop applied after rotation. If omitted and ``config`` is
        provided, uses ``object.cropped_shape`` or reconstruction padding from
        the config; otherwise defaults to 0.
    show_stats : bool, default False
        Compute per-slice statistics traits on each slice change (`widget.stats_mean`,
        `stats_min`, `stats_max`, `stats_std`, each a legacy list of 3 floats).
        Python-side only; the JS widget does not render a stats bar. Set False to
        skip 12 reductions per slice scrub on multi-MB volumes when you don't
        need the values.
    show_controls : bool, default True
        Show secondary controls for color, colorbar, smoothing, crosshair,
        z-stretch, contrast, and playback. The slice toolbar keeps only FFT
        and Reset Zoom visible.
    controls_collapsed : bool, default False
        Start with controls hidden while keeping a recoverable ``Controls``
        button in the frontend.
    show_crosshair : bool, default True
        Show slice intersection guides across orthogonal panels.
    show_fft : bool, default False
        Toggle FFT panel for the active plane.
    fft_window : bool, default False
        Apply a 2D Hann window to each displayed slice before FFT. Useful for
        suppressing edge leakage/streaking in reciprocal-space panels.
    log_scale : bool, default False
        Use signed log1p for intensity mapping.
    auto_contrast : bool, default True
        Use percentile-based contrast (2nd-98th). On for ptycho phase data
        (long-tailed histogram - manual contrast usually crushes the signal).
    vmin, vmax : float, optional
        Manual contrast limits.
    slice_alignment : {"off", "auto", "manual"}, default "off"
        Display-only global post-alignment mode. ``"auto"`` estimates one
        row/col shift-per-slice slope from adjacent slice registration when no
        slopes are provided; ``"manual"`` applies the provided slopes directly.
        The raw volume is not modified.
    row_shift_px_per_slice, col_shift_px_per_slice : float, default 0.0
        Global display shift applied per slice relative to the middle slice,
        using the project-wide row/col convention. Positive row shifts move
        deeper slices down; positive col shifts move them right.
    fps : float, default 30.0
        Playback speed when scrubbing one axis, capped at 30.
    play_axis : int, default 0
        Which control to animate (0=slice/Z, 1=oblique plane position,
        3=both). Legacy saved states with 2 are mapped to 1.
    dim_labels : list of str, optional
        Labels for data axes 0, 1, 2 in that order. Default ["slice", "row", "col"]
        matches the project-wide detector-plane convention (axis 0 = multislice
        depth, axis 1 = row, axis 2 = col). Pass any 3-string list to override.
    offline : bool, default False
        Pack ``volume_bytes`` as uint8 plus global min/max metadata for static
        nbconvert HTML reports. Leave False for live notebooks that need exact
        float32 values in the browser.

    Example
    -------
    Instantiate Show3DSlices on a multislice ptycho object, scrub through
    depth, and export a slice to PNG:

    >>> import numpy as np
    >>> from quantem.widget import Show3DSlices
    >>> volume = np.random.rand(14, 256, 256).astype(np.float32)
    >>> w = Show3DSlices(volume, title="multislice object", cmap="viridis",
    ...                  sampling=(2.0, 0.2, 0.2), units="A", z_stretch=8)
    >>> w.play()  # doctest: +SKIP
    >>> w.save_image("xy_slice7.png", plane="xy", slice_idx=7)  # doctest: +SKIP

    Notes
    -----
    - The raw volume data is sent once over the Comm channel; subsequent slice
      moves, page changes, and panel switches only ship trait updates, not the data.
    - One volume is active at a time so every page reuses the same top/side cut,
      camera, and display state. For simultaneous side-by-side 3D volume
      rendering, use ``Show3DVolume``.
    - Saved states that include ``dual_mode`` / ``show_diff`` keys are still
      rejected because those imply a different rendering contract.
    - Call ``free()`` before discarding the widget to release RAM pinned by
      traitlets observers.
    """

    _esm = pathlib.Path(__file__).parent / "static" / "show3dslices.js"
    _widget_name = "Show3DSlices"
    _viewer_kind = "slices"

    widget_version = traitlets.Unicode("unknown")  # Python-only: telemetry readout
    viewer_kind = traitlets.Unicode("slices")       # Python-only: introspection / state save

    # Volume dimensions
    nx = traitlets.Int(1).tag(sync=True)
    ny = traitlets.Int(1).tag(sync=True)
    nz = traitlets.Int(1).tag(sync=True)
    panel_count = traitlets.CInt(1).tag(sync=True)
    active_panel = traitlets.CInt(0).tag(sync=True)
    panel_titles = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    n_pages = traitlets.CInt(1).tag(sync=True)
    page_idx = traitlets.CInt(0).tag(sync=True)
    panels_per_page = traitlets.CInt(0).tag(sync=True)
    page_labels = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    # Slice positions
    slice_x = traitlets.CInt(0).tag(sync=True)
    slice_y = traitlets.CInt(0).tag(sync=True)
    slice_z = traitlets.CInt(0).tag(sync=True)
    oblique_angle = traitlets.Float(0.0).tag(sync=True)
    oblique_profile_line = traitlets.List(traitlets.Dict(), default_value=[]).tag(sync=True)
    # Raw volume data (sent once)
    volume_bytes = traitlets.Bytes(b"").tag(sync=True)
    # Offline HTML report mode: volume_bytes is uint8-quantized against this
    # global range. Default live path keeps exact float32 bytes.
    offline = traitlets.Bool(False).tag(sync=True)
    # True only on a clone written by export_html: forces the standalone HTML to
    # render on a light/white background regardless of the viewer's OS theme.
    _export_light = traitlets.Bool(False).tag(sync=True)
    _offline_min = traitlets.Float(0.0).tag(sync=True)
    _offline_max = traitlets.Float(1.0).tag(sync=True)
    # Frontend-triggered standalone HTML export. The request is JSON so repeated
    # exports of the same mode can include a unique id and still sync.
    export_request = traitlets.Unicode("").tag(sync=True)
    export_status = traitlets.Unicode("").tag(sync=True)
    export_enabled = traitlets.Bool(True).tag(sync=True)
    export_payload = traitlets.Bytes(b"").tag(sync=True)
    export_payload_id = traitlets.Unicode("").tag(sync=True)
    export_filename = traitlets.Unicode("").tag(sync=True)
    # Display
    title = traitlets.Unicode("").tag(sync=True)
    show_title = traitlets.Bool(True).tag(sync=True)
    cmap = traitlets.Unicode("plasma").tag(sync=True)
    log_scale = traitlets.Bool(False).tag(sync=True)
    auto_contrast = traitlets.Bool(True).tag(sync=True)
    vmin = traitlets.Float(None, allow_none=True).tag(sync=True)
    vmax = traitlets.Float(None, allow_none=True).tag(sync=True)
    # Scale bar. `pixel_size` is a scalar (lateral sampling, used by XY/oblique width-axis
    # scale bars). `pixel_size_axes` is the full per-axis triple [z, y, x] in the same
    # units - populated from tuple/list input; defaults to [pixel_size]*3 for scalar input.
    # Both sync to JS; JS uses pixel_size_axes when present for depth-axis scale bars
    # and falls back to pixel_size for the lateral axes.
    pixel_size = traitlets.Float(0.0).tag(sync=True)
    pixel_size_axes = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0, 0.0]).tag(sync=True)
    scale_bar_visible = traitlets.Bool(True).tag(sync=True)
    # Depth-axis display stretch for non-cubic volumes (CSS-only, zero memory).
    # Scales the oblique panel display height. Useful when nz << nxy (e.g. multislice
    # ptycho with nz=14, nxy=730 -> set z_stretch high to make depth panels readable).
    z_stretch = traitlets.Float(30.0).tag(sync=True)
    panel_width_px = traitlets.Int(0).tag(sync=True)
    view_state = traitlets.Dict(default_value={}).tag(sync=True)
    # UI
    show_controls = traitlets.Bool(True).tag(sync=True)
    controls_collapsed = traitlets.Bool(False).tag(sync=True)
    show_stats = traitlets.Bool(False)  # Python-only: gates _compute_stats reductions, no JS bar
    show_crosshair = traitlets.Bool(True).tag(sync=True)
    show_fft = traitlets.Bool(False).tag(sync=True)
    fft_window = traitlets.Bool(False).tag(sync=True)
    fft_colormap = traitlets.Unicode("inferno").tag(sync=True)
    fft_log_scale = traitlets.Bool(False).tag(sync=True)
    fft_auto = traitlets.Bool(True).tag(sync=True)
    orthographic = traitlets.Bool(False).tag(sync=True)
    smooth = traitlets.Bool(True).tag(sync=True)
    flip = traitlets.Bool(False).tag(sync=True)
    show_colorbar = traitlets.Bool(False).tag(sync=True)
    image_vmin_pct = traitlets.Float(0.0).tag(sync=True)
    image_vmax_pct = traitlets.Float(100.0).tag(sync=True)
    show_slice_planes = traitlets.Bool(True).tag(sync=True)
    plane_visibility = traitlets.List(
        traitlets.Bool(),
        default_value=[True, True],
    ).tag(sync=True)
    # Display-only post-processing alignment. The raw volume stays unchanged;
    # JS applies one global row/col shift per slice when enabled.
    slice_alignment = traitlets.Unicode("off").tag(sync=True)
    row_shift_px_per_slice = traitlets.Float(0.0).tag(sync=True)
    col_shift_px_per_slice = traitlets.Float(0.0).tag(sync=True)
    slice_alignment_cached = traitlets.Bool(False).tag(sync=True)
    slice_alignment_status = traitlets.Unicode("").tag(sync=True)
    _slice_alignment_request = traitlets.Unicode("").tag(sync=True)
    volume_opacity = traitlets.Float(0.5).tag(sync=True)
    slice_plane_opacity = traitlets.Float(0.35).tag(sync=True)
    # Axis labels (dim 0, 1, 2). Use detector-plane convention: axis 0 = slice
    # (multislice depth), axis 1 = row, axis 2 = col. The public plane UI is
    # now Top + Side; these labels remain for data coordinates and exports.
    dim_labels = traitlets.List(traitlets.Unicode(), default_value=["slice", "row", "col"]).tag(sync=True)
    # Stats (3 values: xy, xz, yz)
    # stats_*: programmatic Python access only (no JS consumer). Don't sync.
    stats_mean = traitlets.List(traitlets.Float())
    stats_min = traitlets.List(traitlets.Float())
    stats_max = traitlets.List(traitlets.Float())
    stats_std = traitlets.List(traitlets.Float())
    # Playback
    playing = traitlets.Bool(False).tag(sync=True)
    reverse = traitlets.Bool(False).tag(sync=True)
    boomerang = traitlets.Bool(True).tag(sync=True)
    fps = traitlets.Float(30.0).tag(sync=True)
    loop = traitlets.Bool(True).tag(sync=True)
    play_axis = traitlets.Int(0).tag(sync=True)  # 0=slice, 1=oblique plane, 3=All
    # Validators (consistent with Show3D)

    @traitlets.validate("cmap")
    def _validate_cmap(self, proposal: dict) -> str:
        """Reject unknown colormap names. Keeps JS LUT lookup safe."""
        val = str(proposal["value"])
        if val not in _VALID_CMAPS:
            raise traitlets.TraitError(
                f"Unknown cmap {val!r}. Valid: {sorted(_VALID_CMAPS)}"
            )
        return val

    @traitlets.validate("fft_colormap")
    def _validate_fft_colormap(self, proposal: dict) -> str:
        """Reject unknown FFT colormap names. Keeps JS LUT lookup safe."""
        val = str(proposal["value"])
        if val not in _VALID_CMAPS:
            raise traitlets.TraitError(
                f"Unknown fft_colormap {val!r}. Valid: {sorted(_VALID_CMAPS)}"
            )
        return val

    @traitlets.validate("image_vmin_pct", "image_vmax_pct")
    def _validate_image_percent(self, proposal: dict) -> float:
        """Clamp histogram contrast handles to [0, 100]."""
        val = float(proposal["value"])
        if not math.isfinite(val):
            raise traitlets.TraitError(f"{proposal['trait'].name} must be finite, got {val}")
        return max(0.0, min(val, 100.0))

    @traitlets.validate("oblique_angle")
    def _validate_oblique_angle(self, proposal: dict) -> float:
        """Normalize the arbitrary vertical plane angle to [0, 180)."""
        val = float(proposal["value"])
        if not math.isfinite(val):
            raise traitlets.TraitError(f"oblique_angle must be finite, got {val}")
        return val % 180.0

    @traitlets.validate("oblique_profile_line")
    def _validate_oblique_profile_line(self, proposal: dict) -> list[dict[str, float]]:
        """Accept [] or two Show3D-style endpoint dicts with row/col keys."""
        val = list(proposal["value"])
        if len(val) == 0:
            return []
        if len(val) != 2:
            raise traitlets.TraitError(
                f"oblique_profile_line must be [] or two endpoints, got {len(val)}"
            )
        out: list[dict[str, float]] = []
        for point in val:
            row = float(point["row"])
            col = float(point["col"])
            if not math.isfinite(row) or not math.isfinite(col):
                raise traitlets.TraitError("oblique_profile_line endpoints must be finite")
            out.append({"row": row, "col": col})
        return out

    @traitlets.validate("volume_opacity", "slice_plane_opacity")
    def _validate_opacity(self, proposal: dict) -> float:
        """Clamp volume display opacity sliders to [0, 1]."""
        val = float(proposal["value"])
        if not math.isfinite(val):
            raise traitlets.TraitError(f"{proposal['trait'].name} must be finite, got {val}")
        return max(0.0, min(val, 1.0))

    @traitlets.validate("plane_visibility")
    def _validate_plane_visibility(self, proposal: dict) -> list[bool]:
        """Accept new [XY, oblique] flags plus legacy [XY, XZ, YZ] state."""
        val = [bool(v) for v in proposal["value"]]
        if len(val) not in (2, 3):
            raise traitlets.TraitError(
                f"plane_visibility must have length 2 [XY, oblique], got {len(val)}"
            )
        return val[:2]

    @traitlets.validate("slice_alignment")
    def _validate_slice_alignment(self, proposal: dict) -> str:
        """Restrict post-alignment to the supported global modes."""
        val = str(proposal["value"] or "off").strip().lower().replace("-", "_")
        if val not in _VALID_SLICE_ALIGNMENT:
            raise traitlets.TraitError(
                f"slice_alignment must be one of {sorted(_VALID_SLICE_ALIGNMENT)}, got {proposal['value']!r}"
            )
        return val

    @traitlets.validate("row_shift_px_per_slice", "col_shift_px_per_slice")
    def _validate_slice_alignment_shift(self, proposal: dict) -> float:
        """Require finite global alignment slopes."""
        val = float(proposal["value"])
        if not math.isfinite(val):
            raise traitlets.TraitError(f"{proposal['trait'].name} must be finite, got {val}")
        return val

    @traitlets.validate("fps")
    def _validate_fps(self, proposal: dict) -> float:
        """Reject invalid fps and cap playback at the browser budget."""
        val = float(proposal["value"])
        if not math.isfinite(val):
            raise traitlets.TraitError(f"fps must be finite, got {val}")
        if val <= 0:
            raise traitlets.TraitError(f"fps must be > 0, got {val}")
        return min(val, _MAX_PLAYBACK_FPS)

    @traitlets.validate("pixel_size")
    def _validate_pixel_size(self, proposal: dict) -> float:
        """Reject NaN/inf/negative pixel size to keep scale bar math sane."""
        val = float(proposal["value"])
        if math.isnan(val) or math.isinf(val):
            raise traitlets.TraitError(f"pixel_size must be finite, got {val}")
        if val < 0:
            raise traitlets.TraitError(f"pixel_size must be >= 0, got {val}")
        return val

    @traitlets.validate("pixel_size_axes")
    def _validate_pixel_size_axes(self, proposal: dict) -> list[float]:
        """Enforce 3-element finite non-negative [pz, py, px] tuple."""
        val = [float(v) for v in proposal["value"]]
        if len(val) != 3:
            raise traitlets.TraitError(
                f"pixel_size_axes must have length 3, got {len(val)}"
            )
        for v in val:
            if not math.isfinite(v):
                raise traitlets.TraitError(f"pixel_size_axes values must be finite, got {val}")
            if v < 0:
                raise traitlets.TraitError(f"pixel_size_axes values must be >= 0, got {val}")
        return val

    @traitlets.validate("play_axis")
    def _validate_play_axis(self, proposal: dict) -> int:
        """Restrict playback to slice, oblique plane, or both.

        Legacy Show3DSlices states used 1=Y and 2=X. Both now map to the
        single arbitrary-angle plane so old notebooks do not resurrect row/col
        playback labels.
        """
        val = int(proposal["value"])
        if val == 0 or val == 3:
            return val
        if val in (1, 2):
            return 1
        raise traitlets.TraitError(f"play_axis must be 0/1/3, got {val}")

    @traitlets.validate("z_stretch")
    def _validate_z_stretch(self, proposal: dict) -> float:
        """Clamp z stretch to [1, 50] so the 3D view stays usable."""
        val = float(proposal["value"])
        if math.isnan(val) or math.isinf(val):
            raise traitlets.TraitError(f"z_stretch must be finite, got {val}")
        return max(1.0, min(val, 50.0))

    @traitlets.validate("panel_width_px")
    def _validate_panel_width_px(self, proposal: dict) -> int:
        val = int(proposal["value"])
        if val < 0:
            raise traitlets.TraitError(f"panel_width_px must be >= 0, got {val}")
        return val

    @traitlets.validate("panel_count")
    def _validate_panel_count(self, proposal: dict) -> int:
        val = int(proposal["value"])
        if val < 1:
            raise traitlets.TraitError(f"panel_count must be >= 1, got {val}")
        return val

    @traitlets.validate("active_panel")
    def _validate_active_panel(self, proposal: dict) -> int:
        return max(0, min(int(proposal["value"]), max(0, int(self.panel_count) - 1)))

    @traitlets.validate("panel_titles")
    def _validate_panel_titles(self, proposal: dict) -> list[str]:
        val = [str(v) for v in proposal["value"]]
        if len(val) not in (0, int(self.panel_count)):
            raise traitlets.TraitError(
                f"panel_titles must be empty or length panel_count={self.panel_count}, "
                f"got {len(val)}"
            )
        return val

    @traitlets.validate("n_pages")
    def _validate_n_pages(self, proposal: dict) -> int:
        val = int(proposal["value"])
        if val < 1:
            raise traitlets.TraitError(f"n_pages must be >= 1, got {val}")
        return val

    @traitlets.validate("panels_per_page")
    def _validate_panels_per_page(self, proposal: dict) -> int:
        val = int(proposal["value"])
        if val < 0:
            raise traitlets.TraitError(f"panels_per_page must be >= 0, got {val}")
        return val

    @traitlets.validate("page_idx")
    def _validate_page_idx(self, proposal: dict) -> int:
        return max(0, min(int(proposal["value"]), max(0, int(self.n_pages) - 1)))

    @traitlets.validate("page_labels")
    def _validate_page_labels(self, proposal: dict) -> list[str]:
        val = [str(value) for value in proposal["value"]]
        if len(val) not in (0, int(self.n_pages)):
            raise traitlets.TraitError(
                f"page_labels must be empty or length n_pages={self.n_pages}, got {len(val)}"
            )
        return val

    @traitlets.validate("dim_labels")
    def _validate_dim_labels(self, proposal: dict) -> list[str]:
        """Enforce exactly 3 axis labels for Z/Y/X. Reject bare strings up
        front since `list("zyx")` would silently produce single-character
        labels in saved state and panel headers (typo footgun)."""
        raw = proposal["value"]
        if isinstance(raw, (str, bytes)):
            raise traitlets.TraitError(
                f"dim_labels must be a list of 3 strings, got bare {type(raw).__name__}: {raw!r}"
            )
        val = list(raw)
        if len(val) != 3:
            raise traitlets.TraitError(
                f"dim_labels must have length 3, got {len(val)}"
            )
        for i, label in enumerate(val):
            if not isinstance(label, str):
                raise traitlets.TraitError(
                    f"dim_labels[{i}] must be a string, got {type(label).__name__}: {label!r}"
                )
        return val

    @traitlets.validate("slice_z")
    def _validate_slice_z(self, proposal: dict) -> int:
        """Clamp slice_z to [0, nz-1]."""
        return max(0, min(int(proposal["value"]), max(0, int(self.nz) - 1)))

    @traitlets.validate("slice_y")
    def _validate_slice_y(self, proposal: dict) -> int:
        """Clamp slice_y to [0, ny-1]."""
        return max(0, min(int(proposal["value"]), max(0, int(self.ny) - 1)))

    @traitlets.validate("slice_x")
    def _validate_slice_x(self, proposal: dict) -> int:
        """Clamp slice_x to [0, nx-1]."""
        return max(0, min(int(proposal["value"]), max(0, int(self.nx) - 1)))

    @traitlets.validate("vmax")
    def _validate_vmax_ge_vmin(self, proposal: dict) -> float | None:
        """Reject vmax < vmin and non-finite values."""
        new_vmax = proposal["value"]
        if new_vmax is not None:
            if not math.isfinite(new_vmax):
                raise traitlets.TraitError(f"vmax must be finite, got {new_vmax}")
            if self.vmin is not None and new_vmax < self.vmin:
                raise traitlets.TraitError(
                    f"vmax ({new_vmax}) must be >= vmin ({self.vmin})"
                )
        return new_vmax

    @traitlets.validate("vmin")
    def _validate_vmin_le_vmax(self, proposal: dict) -> float | None:
        """Reject vmin > vmax and non-finite values."""
        new_vmin = proposal["value"]
        if new_vmin is not None:
            if not math.isfinite(new_vmin):
                raise traitlets.TraitError(f"vmin must be finite, got {new_vmin}")
            if self.vmax is not None and new_vmin > self.vmax:
                raise traitlets.TraitError(
                    f"vmin ({new_vmin}) must be <= vmax ({self.vmax})"
                )
        return new_vmin

    # =========================================================================
    # === Construction ===
    # =========================================================================

    def __init__(
        self,
        data,
        data_b=None,
        *,
        title: str = "",
        title_b: str = "",
        panel_titles: Sequence[str] | None = None,
        page_labels: Sequence[str | None] | None = None,
        ui_mode: UiMode = "interactive",
        show_title: bool | None = None,
        cmap: str = "plasma",
        sampling: float | Sequence[float] | None = None,
        units: str | Sequence[str] | None = None,
        pixel_size: float | Sequence[float] | None = None,
        config: Mapping | str | pathlib.Path | None = None,
        apply_config_transforms: bool = True,
        crop: int | tuple[int, int] | tuple[int, int, int, int] = 0,
        padding: int | tuple[int, int] = 0,
        pad_mode: str = "median",
        rotation_deg: float | None = None,
        post_crop: int | tuple[int, int] | tuple[int, int, int, int] | None = None,
        scale_bar_visible: bool | None = None,
        show_scale_bar: bool | None = None,
        z_stretch: float | None = None,
        panel_width_px: int = 0,
        show_controls: bool | None = None,
        controls_collapsed: bool | None = None,
        show_stats: bool | None = None,
        show_crosshair: bool | None = None,
        show_fft: bool = False,
        fft_window: bool = False,
        orthographic: bool = False,
        smooth: bool = True,
        flip: bool = False,
        show_diff: bool = False,
        log_scale: bool = False,
        auto_contrast: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        image_vmin_pct: float = 0.0,
        image_vmax_pct: float = 100.0,
        slice_alignment: str = "off",
        row_shift_px_per_slice: float = 0.0,
        col_shift_px_per_slice: float = 0.0,
        fps: float = 30.0,
        loop: bool = True,
        reverse: bool = False,
        boomerang: bool = True,
        linked_contrast: bool = True,
        play_axis: int = 0,
        dim_labels: list[str] | None = None,
        offline: bool = False,
        state=None,
        **kwargs,
    ):
        kwargs.pop("compact", None)
        _reject_unknown_kwargs(type(self), kwargs)
        if show_diff:
            raise ValueError(
                "Show3DSlices does not support difference mode. Pass explicit "
                "comparison panels and inspect one active volume at a time."
            )
        if linked_contrast is not True:
            raise ValueError(
                "Show3DSlices uses one shared contrast range across panels; "
                "linked_contrast must remain True."
            )
        super().__init__(**kwargs)
        self.widget_version = resolve_widget_version()
        self.viewer_kind = self._viewer_kind
        # Pre-seed so free() / __repr__ / summary() are safe even if a validator
        # raises before _data is assigned below (e.g. wrong ndim or complex data).
        self._data: np.ndarray | None = None
        self._slice_alignment_estimates: dict[int, dict] = {}
        self._syncing_plane_visibility = False
        self._syncing_pages = False
        config_data = _load_quantem_config(config)
        rotation_deg_was_set = rotation_deg is not None
        post_crop_was_set = post_crop is not None

        # `sampling` + `units` is the canonical quantem convention (Dataset.sampling
        # /.units, Show2D, Show4DSTEM). Convert a user-provided sampling to the
        # internal Å `pixel_size` trait the scale bar reads (nm -> 10x).
        if pixel_size is None:
            pixel_size = 0.0
        if sampling is not None:
            _unit_scale = {"": 1.0, "nm": 10.0, "a": 1.0, "å": 1.0, "angstrom": 1.0, "angstroms": 1.0}
            def _scale(unit):
                key = str(unit).strip().lower() if unit else ""
                if key not in _unit_scale:
                    raise ValueError(f"unsupported unit: {unit!r}")
                return _unit_scale[key]
            if isinstance(sampling, (tuple, list)):
                unit_list = list(units) if isinstance(units, (tuple, list)) else [units] * len(sampling)
                pixel_size = [float(s) * _scale(unit_list[min(i, len(unit_list) - 1)]) for i, s in enumerate(sampling)]
            else:
                unit0 = units[0] if isinstance(units, (tuple, list)) else units
                pixel_size = float(sampling) * _scale(unit0)

        # Duck-typed Dataset3d extraction (matches Show2D / Show3D pattern).
        # `array` is the required payload; title/sampling/units are optional
        # metadata so lightweight Dataset3d-like wrappers work naturally.
        if hasattr(data, "array"):
            name = getattr(data, "name", "")
            if not title and name:
                title = name
            pixel_size_is_default = _is_default_pixel_size(pixel_size)
            if pixel_size_is_default and hasattr(data, "sampling"):
                try:
                    units = list(getattr(data, "units", []) or [])
                    samp = list(data.sampling)
                    # Unit conversion to Å per axis. If units are absent, assume
                    # sampling is already in Å so Dataset3d-like objects remain
                    # easy to use. A single unit applies to every sampling axis.
                    def unit_scale(unit):
                        if not unit:
                            return 1.0
                        unit_str = str(unit).strip().lower()
                        if unit_str == "nm":
                            return 10.0
                        if unit_str in ("a", "å", "angstrom", "angstroms"):
                            return 1.0
                        raise ValueError(f"unsupported Dataset3d unit: {unit!r}")

                    def axis_unit(axis_index: int):
                        if not units:
                            return ""
                        if len(units) == 1:
                            return units[0]
                        return units[axis_index]

                    if len(samp) >= 3:
                        pixel_size = [
                            float(samp[i]) * unit_scale(axis_unit(i))
                            for i in (-3, -2, -1)
                        ]
                    elif len(samp) >= 1:
                        pixel_size = float(samp[-1]) * unit_scale(axis_unit(-1))
                except (IndexError, TypeError, ValueError):
                    pass
            data = data.array

        secondary_title = ""
        if hasattr(data_b, "array"):
            secondary_title = str(getattr(data_b, "name", "") or "")
            data_b = data_b.array

        if config_data is not None and _is_default_pixel_size(pixel_size):
            config_pixel_size = _pixel_size_from_quantem_config(config_data)
            if config_pixel_size is not None:
                pixel_size = config_pixel_size

        if data_b is not None and page_labels is not None:
            raise ValueError(
                "Use either data_b for a legacy two-panel comparison or "
                "page_labels for pages, not both"
            )
        if data_b is None:
            data, panel_titles, n_pages, panels_per_page, resolved_page_labels = (
                _normalise_show3dslices_pages(
                    data,
                    panel_titles=panel_titles,
                    page_labels=page_labels,
                )
            )
        else:
            n_pages, panels_per_page, resolved_page_labels = 1, 0, []

        data = to_numpy(data)
        if data_b is not None:
            data_b = to_numpy(data_b)
            if data.ndim != 3 or data_b.ndim != 3:
                raise ValueError(
                    "Show3DSlices data_b comparison requires two 3D volumes; "
                    f"got data={data.ndim}D and data_b={data_b.ndim}D"
                )
            if data.shape != data_b.shape:
                raise ValueError(
                    f"Show3DSlices data_b shape mismatch: {data.shape} != {data_b.shape}"
                )
            data = np.stack([data, data_b], axis=0)
        elif data.ndim == 3:
            data = data[np.newaxis, ...]
        elif data.ndim != 4:
            raise ValueError(
                f"Show3DSlices requires 3D data or 4D panel data, got {data.ndim}D"
            )
        if 0 in data.shape:
            raise ValueError(f"Empty volume: shape {data.shape}. All dims must be >= 1.")
        if np.iscomplexobj(data):
            raise TypeError(
                "Show3DSlices does not accept complex data. Convert first: "
                "np.abs(arr) for magnitude or np.angle(arr) for phase."
            )
        if not np.isfinite(data).all():
            raise ValueError(
                "Data contains NaN or inf. Clean first: "
                "np.nan_to_num(arr, nan=0, posinf=0, neginf=0)."
            )
        with np.errstate(over="ignore", invalid="ignore"):
            self._data = data.astype(np.float32, copy=False)
        if not np.isfinite(self._data).all():
            raise ValueError(
                "Data exceeds float32 range (|value| > 3.4e38) after cast; "
                "rescale first before passing to Show3DSlices."
            )
        if config_data is not None and apply_config_transforms:
            if not rotation_deg_was_set:
                rotation_deg = _config_float(config_data, "data", "rotation_deg") or 0.0
            if not post_crop_was_set:
                post_crop = _post_crop_from_quantem_config(self._data.shape[1:], config_data)
        if rotation_deg is None:
            rotation_deg = 0.0
        if post_crop is None:
            post_crop = 0
        self._crop = _normalize_crop(crop)
        self._padding = _normalize_padding(padding)
        self._pad_mode = str(pad_mode)
        self._rotation_deg = _normalize_rotation_deg(rotation_deg)
        self._post_crop = _normalize_crop(post_crop)
        transformed_panels = []
        for panel in self._data:
            panel = _crop_stack(panel, self._crop)
            panel = _pad_stack(panel, self._padding, self._pad_mode)
            panel = _rotate_stack_inplane(panel, self._rotation_deg)
            panel = _crop_stack(panel, self._post_crop)
            transformed_panels.append(panel)
        self._data = np.ascontiguousarray(np.stack(transformed_panels, axis=0), dtype=np.float32)
        self.panel_count, self.nz, self.ny, self.nx = self._data.shape
        self.n_pages = int(max(1, n_pages))
        self.panels_per_page = int(max(0, panels_per_page))
        self.page_idx = 0
        self.page_labels = list(resolved_page_labels)
        if self.n_pages > 1:
            if self.panels_per_page <= 0:
                raise ValueError("panels_per_page must be positive for Show3DSlices pages")
            if self.panel_count != self.n_pages * self.panels_per_page:
                raise ValueError(
                    "paged Show3DSlices expects panel_count == n_pages * panels_per_page, "
                    f"got {self.panel_count} != {self.n_pages} * {self.panels_per_page}"
                )
        else:
            self.panels_per_page = 0
            self.page_labels = []
        if panel_titles is None:
            if self.panel_count == 1:
                panel_titles = []
            elif data_b is not None:
                panel_titles = [
                    title or "Panel 1",
                    title_b or secondary_title or "Panel 2",
                ]
            else:
                panel_titles = [f"Panel {i + 1}" for i in range(self.panel_count)]
        panel_titles = [str(v) for v in panel_titles]
        if panel_titles and len(panel_titles) != self.panel_count:
            raise ValueError(
                f"panel_titles length {len(panel_titles)} must match panel_count {self.panel_count}"
            )
        self.panel_titles = panel_titles
        self.active_panel = 0

        # Default to middle slices
        self.slice_z = self.nz // 2
        self.slice_y = self.ny // 2
        self.slice_x = self.nx // 2

        if (
            scale_bar_visible is not None
            and show_scale_bar is not None
            and bool(scale_bar_visible) != bool(show_scale_bar)
        ):
            raise ValueError("Use either show_scale_bar or scale_bar_visible, not conflicting values")
        ui = resolve_ui_mode(
            ui_mode,
            defaults={
                "show_title": True,
                "show_controls": True,
                "controls_collapsed": False,
                "show_stats": False,
                "show_crosshair": True,
                "show_scale_bar": True,
            },
            overrides={
                "show_title": show_title,
                "show_controls": show_controls,
                "controls_collapsed": controls_collapsed,
                "show_stats": show_stats,
                "show_crosshair": show_crosshair,
                "show_scale_bar": scale_bar_visible if scale_bar_visible is not None else show_scale_bar,
            },
        )
        self.title = title
        self.show_title = bool(ui["show_title"])
        self.cmap = cmap
        # pixel_size accepts: None → 0 (no scale bar), scalar (isotropic), or
        # 3-tuple/list/ndarray (anisotropic: [pz, py, px] in the same units).
        # For 3-tuple input the scalar trait is set to the lateral mean (py+px)/2
        # so existing scale-bar code keeps working; the full triple is published
        # via pixel_size_axes for per-axis scale bars.
        if pixel_size is None:
            pixel_size = 0.0
        if isinstance(pixel_size, Real) or np.isscalar(pixel_size):
            ps_scalar = float(pixel_size)
            ps_axes = [ps_scalar, ps_scalar, ps_scalar]
        else:
            try:
                ps_axes = [float(v) for v in pixel_size]
            except TypeError:
                raise TypeError(
                    f"pixel_size must be a scalar or 3-element sequence (Å/pixel), "
                    f"got {type(pixel_size).__name__}."
                )
            if len(ps_axes) != 3:
                raise ValueError(
                    f"pixel_size as a sequence must have exactly 3 elements [pz, py, px], "
                    f"got {len(ps_axes)}."
                )
            for v in ps_axes:
                if not math.isfinite(v) or v < 0:
                    raise ValueError(f"pixel_size_axes must be finite and >= 0, got {ps_axes}.")
            ps_scalar = (ps_axes[1] + ps_axes[2]) / 2.0  # lateral mean
        self.pixel_size = ps_scalar
        self.pixel_size_axes = ps_axes
        self.scale_bar_visible = bool(ui["show_scale_bar"])
        if z_stretch is None:
            z_stretch = 30.0
        self.z_stretch = float(z_stretch)
        self.panel_width_px = int(panel_width_px)
        self.show_controls = bool(ui["show_controls"])
        self.controls_collapsed = bool(ui["controls_collapsed"])
        self.show_stats = bool(ui["show_stats"])
        self.show_crosshair = bool(ui["show_crosshair"])
        self.show_fft = show_fft
        self.fft_window = fft_window
        self.orthographic = orthographic
        self.smooth = smooth
        self.flip = flip
        self.log_scale = log_scale
        self.auto_contrast = auto_contrast
        self.vmin = vmin
        self.vmax = vmax
        self.image_vmin_pct = image_vmin_pct
        self.image_vmax_pct = image_vmax_pct
        self.row_shift_px_per_slice = row_shift_px_per_slice
        self.col_shift_px_per_slice = col_shift_px_per_slice
        self.slice_alignment = slice_alignment
        self.slice_alignment_cached = (
            self.slice_alignment != "off"
            or abs(self.row_shift_px_per_slice) >= 1e-12
            or abs(self.col_shift_px_per_slice) >= 1e-12
        )
        self.slice_alignment_status = ""
        if (
            self.slice_alignment == "auto"
            and abs(self.row_shift_px_per_slice) < 1e-12
            and abs(self.col_shift_px_per_slice) < 1e-12
        ):
            self.estimate_slice_alignment(apply=True)
        self.fps = fps
        self.loop = loop
        self.reverse = reverse
        self.boomerang = boomerang
        self.play_axis = play_axis
        self.offline = bool(offline)
        if dim_labels is not None:
            self.dim_labels = dim_labels

        self._compute_stats()
        self._sync_volume_bytes()
        self.observe(self._on_slice_change, names=["slice_x", "slice_y", "slice_z", "active_panel"])
        self.observe(self._on_page_change, names=["page_idx"])
        self.observe(self._on_active_panel_page_change, names=["active_panel"])
        self.observe(self._on_playing_change, names=["playing"])
        self.observe(self._on_show_stats_change, names=["show_stats"])
        self.observe(self._on_export_request_change, names=["export_request"])
        self.observe(self._on_show_slice_planes_change, names=["show_slice_planes"])
        self.observe(self._on_plane_visibility_change, names=["plane_visibility"])
        self.observe(self._on_slice_alignment_request_change, names=["_slice_alignment_request"])

        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(
                    json.loads(pathlib.Path(state).read_text()),
                    require_envelope=True,
                    expected_widget=self._widget_name,
                )
            else:
                state = unwrap_state_payload(state, expected_widget=self._widget_name)
            self.load_state_dict(state)

    # =========================================================================
    # === Public API ===
    # =========================================================================

    def _sync_volume_bytes(self) -> None:
        """Sync exact live float32 bytes or packed offline report bytes."""
        if self._data is None:
            self.volume_bytes = b""
            return

        arr = np.ascontiguousarray(self._data, dtype=np.float32)
        if not self.offline:
            self._offline_min = 0.0
            self._offline_max = 1.0
            self.volume_bytes = arr.tobytes()
            return

        lo = float(arr.min())
        hi = float(arr.max())
        rng = hi - lo
        if rng <= 0:
            quantized = np.zeros(arr.shape, dtype=np.uint8)
        else:
            quantized = np.clip(np.rint((arr - lo) * (255.0 / rng)), 0, 255).astype(np.uint8)
        self._offline_min = lo
        self._offline_max = hi
        self.volume_bytes = quantized.tobytes()

    def _on_plane_visibility_change(self, change: dict) -> None:
        if self._syncing_plane_visibility:
            return
        visible = any(bool(v) for v in change["new"])
        if self.show_slice_planes == visible:
            return
        self._syncing_plane_visibility = True
        try:
            self.show_slice_planes = visible
        finally:
            self._syncing_plane_visibility = False

    def _on_show_slice_planes_change(self, change: dict) -> None:
        if self._syncing_plane_visibility:
            return
        visible = bool(change["new"])
        if visible:
            if any(bool(v) for v in self.plane_visibility):
                return
            next_visibility = [True, True]
        else:
            next_visibility = [False, False]
        if list(self.plane_visibility) == next_visibility:
            return
        self._syncing_plane_visibility = True
        try:
            self.plane_visibility = next_visibility
        finally:
            self._syncing_plane_visibility = False

    def __repr__(self) -> str:
        page_label = (
            f"{self.n_pages} pages × " if int(self.n_pages) > 1 else ""
        )
        panel_label = (
            f"{self.panels_per_page} panels/page × "
            if int(self.n_pages) > 1 and int(self.panels_per_page) > 1
            else f"{self.panel_count} panels × "
            if int(self.n_pages) <= 1 and int(self.panel_count) > 1
            else ""
        )
        return (
            f"{self._widget_name}({page_label}{panel_label}{self.nz}×{self.ny}×{self.nx}, "
            f"page={self.page_idx}, "
            f"active_panel={self.active_panel}, "
            f"slices=({self.slice_z},{self.slice_y},{self.slice_x}), cmap={self.cmap})"
        )

    def state_dict(self) -> dict:
        """Return a JSON-serializable snapshot of every user-tunable trait.

        Captures display config (cmap, log scale, contrast, FFT, orthographic,
        smooth, flip, crosshair), scale-bar settings (``pixel_size``,
        ``pixel_size_axes``, ``z_stretch``), playback config (fps, loop,
        reverse, boomerang, play axis), the three slice positions, and the
        dimension labels. The raw volume data is NOT included; pair the
        snapshot with the original ``data`` argument on restore.

        Returns
        -------
        dict
            Mapping of trait name -> serializable value. Suitable for
            ``json.dump`` or use with ``save`` / ``load_state_dict``.

        Example
        -------
        >>> from quantem.widget import Show3DSlices
        >>> w = Show3DSlices(volume, cmap="viridis")  # doctest: +SKIP
        >>> state = w.state_dict()  # doctest: +SKIP
        >>> w2 = Show3DSlices(volume)  # doctest: +SKIP
        >>> w2.load_state_dict(state)  # doctest: +SKIP

        Notes
        -----
        - Schema versioning is handled inside ``save``; the dict returned
          here is the unversioned inner payload.
        """
        return {
            # Widget discriminator: lets load_state_dict reject cross-widget
            # loads (e.g. Show3D.state_dict() into Show3DSlices) cleanly.
            "_widget": "Show3DSlices",
            "title": self.title,
            "show_title": self.show_title,
            "viewer_kind": self.viewer_kind,
            "n_pages": self.n_pages,
            "page_idx": self.page_idx,
            "panels_per_page": self.panels_per_page,
            "page_labels": list(self.page_labels),
            "active_panel": self.active_panel,
            "panel_titles": list(self.panel_titles),
            "cmap": self.cmap,
            "log_scale": self.log_scale,
            "auto_contrast": self.auto_contrast,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "show_stats": self.show_stats,
            "show_controls": self.show_controls,
            "controls_collapsed": self.controls_collapsed,
            "show_crosshair": self.show_crosshair,
            "show_fft": self.show_fft,
            "fft_window": self.fft_window,
            "fft_colormap": self.fft_colormap,
            "fft_log_scale": self.fft_log_scale,
            "fft_auto": self.fft_auto,
            "orthographic": self.orthographic,
            "smooth": self.smooth,
            "flip": self.flip,
            "show_colorbar": self.show_colorbar,
            "image_vmin_pct": self.image_vmin_pct,
            "image_vmax_pct": self.image_vmax_pct,
            "show_slice_planes": self.show_slice_planes,
            "plane_visibility": list(self.plane_visibility),
            "slice_alignment": self.slice_alignment,
            "row_shift_px_per_slice": self.row_shift_px_per_slice,
            "col_shift_px_per_slice": self.col_shift_px_per_slice,
            "slice_alignment_cached": self.slice_alignment_cached,
            "volume_opacity": self.volume_opacity,
            "slice_plane_opacity": self.slice_plane_opacity,
            "pixel_size": self.pixel_size,
            "pixel_size_axes": list(self.pixel_size_axes),
            "scale_bar_visible": self.scale_bar_visible,
            "z_stretch": self.z_stretch,
            "panel_width_px": self.panel_width_px,
            "view_state": dict(self.view_state),
            "slice_x": self.slice_x,
            "slice_y": self.slice_y,
            "slice_z": self.slice_z,
            "oblique_angle": self.oblique_angle,
            "oblique_profile_line": list(self.oblique_profile_line),
            "fps": self.fps,
            "loop": self.loop,
            "reverse": self.reverse,
            "boomerang": self.boomerang,
            "play_axis": self.play_axis,
            "dim_labels": list(self.dim_labels),
        }

    def collapse_controls(self) -> Self:
        """Collapse controls behind the frontend ``Controls`` button."""
        self.controls_collapsed = True
        return self

    def expand_controls(self) -> Self:
        """Expand frontend controls when ``show_controls`` is enabled."""
        self.controls_collapsed = False
        return self

    def toggle_controls(self) -> Self:
        """Toggle whether frontend controls start collapsed."""
        self.controls_collapsed = not bool(self.controls_collapsed)
        return self

    def set_page(self, page: int) -> Self:
        """Show a zero-based page while preserving the active panel slot."""
        self.page_idx = int(page)
        return self

    def next_page(self) -> Self:
        """Advance to the next page, clamped at the final page."""
        return self.set_page(int(self.page_idx) + 1)

    def previous_page(self) -> Self:
        """Move to the previous page, clamped at page zero."""
        return self.set_page(int(self.page_idx) - 1)

    def estimate_slice_alignment(
        self,
        *,
        panel: int | None = None,
        apply: bool = True,
        force: bool = False,
    ) -> dict:
        """Estimate one global row/col post-alignment slope through the stack.

        Adjacent slices are registered after median subtraction, Gaussian
        high-pass filtering, and Hann windowing. The adjacent shifts are
        accumulated, then a straight line is fit versus slice index. The fitted
        slope is the display shift to apply per deeper slice, centered around
        the middle slice when rendered.

        Parameters
        ----------
        panel : int or None, optional
            Panel index to estimate for 4D panel stacks. Defaults to the active
            panel.
        apply : bool, default True
            If True, store the fitted slopes in ``row_shift_px_per_slice`` and
            ``col_shift_px_per_slice`` and switch ``slice_alignment`` to
            ``"auto"``.
        force : bool, default False
            Recompute even when this panel already has a cached estimate.

        Returns
        -------
        dict
            JSON-serializable diagnostics with fitted shifts, cumulative
            trajectory, adjacent shifts, fit R² values, and correlation quality.
        """
        if self._data is None:
            raise ValueError("Cannot estimate slice alignment after free(); rebuild the widget first.")
        panel_idx = int(self.active_panel if panel is None else panel)
        panel_idx = max(0, min(panel_idx, int(self.panel_count) - 1))
        cached = self._slice_alignment_estimates.get(panel_idx)
        if cached is not None and not force:
            result = copy.deepcopy(cached)
            if apply:
                with self.hold_sync():
                    self.row_shift_px_per_slice = result["row_shift_px_per_slice"]
                    self.col_shift_px_per_slice = result["col_shift_px_per_slice"]
                    self.slice_alignment = "auto"
                    self.slice_alignment_cached = True
                    self.slice_alignment_status = (
                        f"Cached row {result['row_shift_px_per_slice']:+.3f} px/slice, "
                        f"col {result['col_shift_px_per_slice']:+.3f} px/slice"
                    )
            return result
        stack = np.asarray(self._data[panel_idx], dtype=np.float32)
        if stack.ndim != 3:
            raise ValueError(f"slice alignment expects a 3D stack, got shape {stack.shape}")
        if stack.shape[0] < 2:
            raise ValueError("slice alignment requires at least 2 slices")

        adjacent: list[list[float]] = []
        quality: list[float] = []
        ref = _slice_registration_image(stack[0])
        ref_spectrum = np.fft.fft2(ref)
        for idx in range(stack.shape[0] - 1):
            mov = _slice_registration_image(stack[idx + 1])
            mov_spectrum = np.fft.fft2(mov)
            row_shift, col_shift, score = _estimate_adjacent_shift_apply(
                ref,
                mov,
                ref_spectrum=ref_spectrum,
                mov_spectrum=mov_spectrum,
            )
            adjacent.append([row_shift, col_shift])
            quality.append(score)
            ref = mov
            ref_spectrum = mov_spectrum

        adjacent_arr = np.asarray(adjacent, dtype=np.float64)
        cumulative = np.zeros((stack.shape[0], 2), dtype=np.float64)
        cumulative[1:] = np.cumsum(adjacent_arr, axis=0)
        z = np.arange(stack.shape[0], dtype=np.float64)
        slopes: list[float] = []
        intercepts: list[float] = []
        r2: list[float] = []
        for dim in range(2):
            slope, intercept = np.polyfit(z, cumulative[:, dim], 1)
            slopes.append(float(slope))
            intercepts.append(float(intercept))
            r2.append(_linear_fit_r2(z, cumulative[:, dim], float(slope), float(intercept)))

        result = {
            "panel": panel_idx,
            "row_shift_px_per_slice": slopes[0],
            "col_shift_px_per_slice": slopes[1],
            "adjacent_shift_apply_px": adjacent_arr.tolist(),
            "cumulative_shift_apply_px": cumulative.tolist(),
            "fit_intercept_px": intercepts,
            "fit_r2": {"row": r2[0], "col": r2[1]},
            "quality": quality,
        }
        self._slice_alignment_estimates[panel_idx] = copy.deepcopy(result)
        if apply:
            with self.hold_sync():
                self.row_shift_px_per_slice = slopes[0]
                self.col_shift_px_per_slice = slopes[1]
                self.slice_alignment = "auto"
                self.slice_alignment_cached = True
                self.slice_alignment_status = (
                    f"Aligned row {slopes[0]:+.3f} px/slice, "
                    f"col {slopes[1]:+.3f} px/slice"
                )
        return result

    def reset_slice_alignment(self) -> Self:
        """Turn off post-alignment and discard cached alignment results."""
        self._slice_alignment_estimates.clear()
        with self.hold_sync():
            self.slice_alignment = "off"
            self.row_shift_px_per_slice = 0.0
            self.col_shift_px_per_slice = 0.0
            self.slice_alignment_cached = False
            self.slice_alignment_status = ""
        return self

    def save(self, path: str) -> None:
        """Write the current widget state to a versioned JSON file.

        Wraps ``state_dict`` in a small envelope that records the widget type
        (``"Show3DSlices"``) and a schema version so ``load_state_dict`` can
        refuse states that belong to a different widget. The raw volume data
        is NOT written; only the display / playback / slice configuration.

        Parameters
        ----------
        path : str
            Destination JSON file path. Parent directories must already exist.

        Returns
        -------
        None

        Example
        -------
        >>> from quantem.widget import Show3DSlices
        >>> w = Show3DSlices(volume, cmap="viridis", sampling=(2.0, 0.2, 0.2), units="A")  # doctest: +SKIP
        >>> w.save("slices_state.json")  # doctest: +SKIP

        Notes
        -----
        - To restore, instantiate ``Show3DSlices`` with the (possibly different)
          volume and call ``w.load_state_dict(json.loads(open(path).read()))``.
        """
        save_state_file(path, self._widget_name, self.state_dict())

    def export_html(
        self,
        path: str | pathlib.Path | None = None,
        *,
        title: str | None = None,
        mode: str = "single",
        encoding: str = "full",
        downsample: int | None = None,
        quantized: bool | None = None,
    ) -> pathlib.Path:
        """Write a standalone HTML viewer for sharing.

        The exact export embeds the current float32 volume bytes and preserves
        numerical precision. The quantized export writes the existing offline
        uint8 representation plus global min/max metadata, which makes a much
        smaller single-file report for visual sharing. Preferred export options
        are ``mode="single"``, ``encoding="full"`` or ``encoding="uint8"``,
        and ``downsample=None``. ``quantized`` is kept as a compatibility alias
        for ``encoding="uint8"``.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            Destination HTML path. Defaults to the current working directory
            with the widget title, volume shape, and export mode in the name.
        quantized : bool, default False
            If True, write the uint8 offline pack. If False, write exact
            float32 bytes.
        title : str, optional
            Browser page title. Defaults to the widget title or class name.

        Returns
        -------
        pathlib.Path
            The written HTML file.
        """
        if self._data is None:
            raise ValueError("Cannot export HTML after free(); rebuild the widget first.")

        quantized = self._normalise_html_export_options(
            mode=mode,
            encoding=encoding,
            downsample=downsample,
            quantized=quantized,
        )
        export_path = pathlib.Path(path) if path is not None else self._default_html_export_path(quantized)
        self._write_html_export(export_path, quantized=quantized, title=title)
        size_mb = export_path.stat().st_size / (1024 * 1024)
        label = self._export_mode_label(quantized)
        self.export_status = f"Exported {export_path.name} ({size_mb:.1f} MB, {label})"
        return export_path

    def _normalise_html_export_options(
        self,
        *,
        mode: str = "single",
        encoding: str = "full",
        downsample: int | None = None,
        quantized: bool | None = None,
    ) -> bool:
        raw_mode = str(mode or "single").strip().lower().replace("_", "-")
        if raw_mode in {"exact", "full"}:
            raw_mode = "single"
            encoding = "full"
        elif raw_mode in {"quantized", "uint8", "u8"}:
            raw_mode = "single"
            encoding = "uint8"
        if raw_mode != "single":
            raise ValueError("Show3DSlices HTML export supports mode='single'")
        if downsample not in (None, 1, "1", "", 0, "0"):
            raise NotImplementedError("Show3DSlices HTML export does not support downsample yet")
        raw_encoding = str(encoding or "full").strip().lower().replace("_", "-")
        if quantized is True:
            raw_encoding = "uint8"
        elif quantized is False and raw_encoding in {"quantized", "uint8", "u8"}:
            raw_encoding = "uint8"
        if raw_encoding in {"full", "exact", "float32", "f32"}:
            return False
        if raw_encoding in {"uint8", "u8", "quantized"}:
            return True
        raise ValueError(f"unknown Show3DSlices export encoding {encoding!r}; expected 'full' or 'uint8'")

    def load_state_dict(self, state: dict) -> None:
        """Apply a saved ``state_dict`` snapshot to this widget.

        Restores display, playback, slice, and labeling configuration from a
        dict produced by ``state_dict``. States that include the deprecated
        ``dual_mode`` / ``show_diff`` keys are rejected with a hint to use
        ``Show3DVolume`` for two-volume comparisons. Unknown keys (typically
        from a newer widget version or a typo) emit a ``UserWarning`` and are
        dropped. Deprecated keys (``title_b``, ``linked_contrast``, ``compact``)
        are silently ignored.

        ``vmin`` / ``vmax`` are cleared first so either bound can be set
        regardless of the current contrast limits. Forward-compat: a saved
        state with only the scalar ``pixel_size`` (no ``pixel_size_axes``)
        mirrors the scalar across all three axes so depth scale bars stay
        in sync with the lateral one.

        Parameters
        ----------
        state : dict
            Mapping previously returned by ``state_dict`` (or its on-disk
            equivalent).

        Returns
        -------
        None
            Mutates the widget in place.

        Example
        -------
        >>> import json
        >>> from quantem.widget import Show3DSlices
        >>> w = Show3DSlices(volume)  # doctest: +SKIP
        >>> w.load_state_dict(json.load(open("slices_state.json")))  # doctest: +SKIP

        Notes
        -----
        - Saved files from a different widget type are caught earlier in the
          ``save`` envelope check, not here.
        """
        # Reject cross-widget loads up front. Without this check, loading a
        # Show3D state_dict() into Show3DSlices would partially apply the
        # overlapping keys (cmap, log_scale, etc.) and leave the widget in a
        # plausible-but-wrong state.
        state = dict(state)
        marker = state.pop("_widget", None)
        if marker is not None and marker != "Show3DSlices":
            raise ValueError(
                f"load_state_dict: state was saved from {marker!r}, not Show3DSlices. "
                f"Use the matching widget class to load it."
            )
        # Surface validator errors. Warn on unknown keys (typo / wrong widget version).
        if state.get("dual_mode") or state.get("show_diff"):
            raise ValueError(
                "Show3DSlices pages compare explicit input volumes but do not "
                "support legacy dual/diff state. Use Show3DVolume for a derived "
                "two-volume difference view."
            )
        # Derive allowed keys from state_dict() so the two stay in lockstep -
        # adding a trait to state_dict() automatically lets load_state_dict()
        # accept it. Deprecated/back-compat keys are tolerated, then filtered.
        allowed = set(self.state_dict().keys())
        deprecated = {"dual_mode", "show_diff", "title_b", "linked_contrast", "compact"}
        unknown = [k for k in state if k not in allowed and k not in deprecated]
        if unknown:
            warnings.warn(
                f"load_state_dict ignored unknown keys: {unknown}. "
                "Likely typo or saved by a different widget version.",
                stacklevel=2,
            )
        state = {k: v for k, v in state.items() if k in allowed}
        # Page topology comes from the supplied data shape. Restore the active
        # page and labels, but never let a state file reinterpret the payload.
        state.pop("n_pages", None)
        state.pop("panels_per_page", None)
        if "page_idx" in state:
            state["page_idx"] = max(
                0,
                min(int(state["page_idx"]), max(0, int(self.n_pages) - 1)),
            )
        if "page_labels" in state and len(state["page_labels"]) != int(self.n_pages):
            state.pop("page_labels")
        if "panel_titles" in state and len(state["panel_titles"]) not in (
            0,
            int(self.panel_count),
        ):
            state.pop("panel_titles")
        if "plane_visibility" in state:
            state["show_slice_planes"] = any(bool(v) for v in state["plane_visibility"])
        elif "show_slice_planes" in state:
            state["plane_visibility"] = [bool(state["show_slice_planes"])] * 2
        state.pop("viewer_kind", None)
        vmin_marker = object()
        vmax_marker = object()
        vmin = state.pop("vmin", vmin_marker)
        vmax = state.pop("vmax", vmax_marker)
        if vmin is not vmin_marker or vmax is not vmax_marker:
            new_vmin = self.vmin if vmin is vmin_marker else vmin
            new_vmax = self.vmax if vmax is vmax_marker else vmax
            if new_vmin is not None and new_vmax is not None and float(new_vmin) > float(new_vmax):
                raise traitlets.TraitError(f"vmin ({new_vmin}) must be <= vmax ({new_vmax})")
            # Clear first so either half of a valid saved pair can be loaded
            # regardless of the widget's current contrast limits.
            self.vmin = None
            self.vmax = None
            if new_vmin is not None:
                self.vmin = float(new_vmin)
            if new_vmax is not None:
                self.vmax = float(new_vmax)
        for key, val in state.items():
            if self.has_trait(key):
                setattr(self, key, val)
        # Forward-compat: state saved before pixel_size_axes existed only has the
        # scalar pixel_size. Mirror it across all three axes so depth scale bars
        # don't desync from the lateral one after a load.
        if "pixel_size" in state and "pixel_size_axes" not in state:
            ps = float(state["pixel_size"])
            self.pixel_size_axes = [ps, ps, ps]

    def free(self) -> None:
        """Release RAM held by this widget.

        Drops the numpy volume and clears the ``volume_bytes`` sync trait, then
        triggers a ``gc.collect()``. ``del widget`` alone does NOT free memory:
        traitlets installs strong observer references that pin the widget's
        refcount until ``free`` is called.

        Returns
        -------
        None
            Mutates the widget in place. After this call, the volume is gone
            and rendering will be blank; rebuild a new widget for further use.

        Example
        -------
        >>> from quantem.widget import Show3DSlices
        >>> w = Show3DSlices(volume)  # doctest: +SKIP
        >>> w.free()  # doctest: +SKIP

        Notes
        -----
        - Idempotent: calling ``free`` twice is a no-op.
        """
        if self._data is None:
            return
        # Stop frontend playback FIRST so JS rAF loop tears down before we null
        # out the byte buffers it's reading from.
        self.playing = False
        self._data = None
        self._slice_alignment_estimates.clear()
        self.slice_alignment_cached = False
        self.volume_bytes = b""
        gc.collect()

    def summary(self) -> None:
        """Print a one-screen status report for the current widget.

        Sections include: title and volume shape (with pixel-size readout
        when set), current slice positions per axis with the user's
        ``dim_labels``, raw data min/max/mean, and display config (cmap,
        contrast, log/linear, FFT, Hann window). Useful for notebook
        reproducibility and bug reports.

        Returns
        -------
        None
            Prints to stdout.

        Example
        -------
        >>> from quantem.widget import Show3DSlices
        >>> w = Show3DSlices(volume, title="multislice")  # doctest: +SKIP
        >>> w.summary()  # doctest: +SKIP
        """
        lines = [self.title or self._widget_name, "═" * 32]
        panel_prefix = f"{self.panel_count} panels × " if self.panel_count > 1 else ""
        lines.append(f"Volume:   {panel_prefix}{self.nz}×{self.ny}×{self.nx}")
        if self.pixel_size > 0:
            ps = self.pixel_size
            unit = f"{ps / 10:.2f} nm/px" if ps >= 10 else f"{ps:.2f} Å/px"
            lines[-1] += f" ({unit})"
        labels = list(self.dim_labels)
        if self.panel_count > 1:
            title_for_panel = (
                self.panel_titles[self.active_panel]
                if self.active_panel < len(self.panel_titles)
                else f"Panel {self.active_panel + 1}"
            )
            lines.append(f"Panel:    {self.active_panel + 1}/{self.panel_count} {title_for_panel}")
        lines.append(
            f"Slices:   {labels[0]}={self.slice_z}  {labels[1]}={self.slice_y}  {labels[2]}={self.slice_x}"
        )
        if hasattr(self, "_data") and self._data is not None:
            arr = self._active_data()
            lines.append(
                f"Data:     min={float(arr.min()):.4g}  max={float(arr.max()):.4g}  mean={float(arr.mean()):.4g}"
            )
        scale = "log" if self.log_scale else "linear"
        if self.vmin is not None and self.vmax is not None:
            contrast = f"vmin={self.vmin:.4g}, vmax={self.vmax:.4g}"
        elif self.auto_contrast:
            contrast = "auto contrast"
        else:
            contrast = "manual contrast"
        display = f"{self.cmap} | {contrast} | {scale}"
        if self.show_fft:
            display += " | FFT"
            if self.fft_window:
                display += " Hann"
        lines.append(f"Display:  {display}")
        if self.slice_alignment != "off":
            lines.append(
                "Align:    "
                f"{self.slice_alignment} "
                f"row={self.row_shift_px_per_slice:+.4g} px/slice "
                f"col={self.col_shift_px_per_slice:+.4g} px/slice"
            )
        print("\n".join(lines))

    def play(self) -> Self:
        """Start slice playback along the current ``play_axis``.

        Sets the ``playing`` trait to ``True``. The JS animation loop scrubs
        the control selected by ``play_axis`` (``0=slice/Z``, ``1=oblique
        plane position``, ``3=both``) at ``fps`` frames per second.

        Returns
        -------
        Self
            The widget, for chaining (``w.play()``).

        Example
        -------
        >>> from quantem.widget import Show3DSlices
        >>> w = Show3DSlices(volume, fps=10, play_axis=0)  # doctest: +SKIP
        >>> w.play()  # doctest: +SKIP
        """
        self.playing = True
        return self

    def pause(self) -> Self:
        """Pause playback at the current slice indices.

        Sets ``playing`` to ``False`` without resetting ``slice_z`` /
        ``slice_y`` / ``slice_x``. Per-slice statistics (if ``show_stats``
        is on) are refreshed for the current slice on pause.

        Returns
        -------
        Self
            The widget, for chaining.

        Example
        -------
        >>> from quantem.widget import Show3DSlices
        >>> w = Show3DSlices(volume)  # doctest: +SKIP
        >>> w.play().pause()  # doctest: +SKIP
        """
        self.playing = False
        return self

    def stop(self) -> Self:
        """Stop playback and recenter all three slice indices.

        Sets ``playing`` to ``False`` and resets ``slice_z`` / ``slice_y`` /
        ``slice_x`` to the geometric center of the volume (``nz // 2`` etc.).
        Use ``pause`` to keep the current slice indices.

        Returns
        -------
        Self
            The widget, for chaining.

        Example
        -------
        >>> from quantem.widget import Show3DSlices
        >>> w = Show3DSlices(volume)  # doctest: +SKIP
        >>> w.play().stop()  # doctest: +SKIP
        """
        self.playing = False
        self.slice_z = self.nz // 2
        self.slice_y = self.ny // 2
        self.slice_x = self.nx // 2
        return self

    def save_image(
        self,
        path: str | pathlib.Path,
        *,
        plane: str | None = None,
        slice_idx: int | None = None,
        format: str | None = None,
        dpi: int = 150,
    ) -> pathlib.Path:
        """Save a single 2D slice of the volume as PNG, PDF, or TIFF.

        Extracts the requested axis-aligned slice, colorizes it with the current
        ``cmap`` and contrast (``vmin`` / ``vmax`` or 2-98 percentile
        auto-contrast), and writes it to ``path``. ``log_scale`` and ``flip``
        are honored so the saved file matches what the browser shows for that
        plane.

        Plane / index mapping for this legacy export helper:

        - ``"xy"``: slice along Z, ``slice_idx`` indexes into ``nz``; defaults
          to current ``slice_z``.
        - ``"xz"``: slice along Y, ``slice_idx`` indexes into ``ny``; defaults
          to current ``slice_y``.
        - ``"yz"``: slice along X, ``slice_idx`` indexes into ``nx``; defaults
          to current ``slice_x``.

        Parameters
        ----------
        path : str or pathlib.Path
            Output file path. Parent directories are created if needed.
        plane : str | None, optional
            One of ``"xy"``, ``"xz"``, ``"yz"``. Defaults to ``"xy"``. The
            interactive browser UI now exposes Top plus an arbitrary Side
            plane; this helper keeps the older axis-aligned export names for
            backwards compatibility.
        slice_idx : int | None, optional
            Slice index along the chosen axis. Defaults to the current
            position for that plane.
        format : str | None, optional
            One of ``"png"``, ``"pdf"``, ``"tiff"``. If omitted, inferred
            from the file extension; defaults to ``"png"`` if no extension.
            Mismatched extension and explicit ``format`` is allowed: the
            container written follows ``format``.
        dpi : int, default 150
            DPI metadata written into the file.

        Returns
        -------
        pathlib.Path
            The written file path.

        Example
        -------
        >>> from quantem.widget import Show3DSlices
        >>> w = Show3DSlices(volume, cmap="viridis")  # doctest: +SKIP
        >>> w.save_image("xy_top.png", plane="xy", slice_idx=0)  # doctest: +SKIP
        >>> w.save_image("xz_mid.pdf", plane="xz")  # doctest: +SKIP

        Notes
        -----
        - PDF output is converted to RGB internally (no alpha channel).
        - Slice indices outside the valid range for the chosen plane raise
          ``IndexError``; unsupported planes or extensions raise ``ValueError``.
        """
        from matplotlib import colormaps
        from PIL import Image

        path = pathlib.Path(path)
        fmt = (format or path.suffix.lstrip(".").lower() or "png").lower()
        if fmt not in ("png", "pdf", "tiff", "tif"):
            raise ValueError(f"Unsupported format: {fmt!r}. Use 'png', 'pdf', or 'tiff'.")

        plane = (plane or "xy").lower()
        if plane == "xy":
            idx = slice_idx if slice_idx is not None else self.slice_z
            max_idx = self.nz
        elif plane == "xz":
            idx = slice_idx if slice_idx is not None else self.slice_y
            max_idx = self.ny
        elif plane == "yz":
            idx = slice_idx if slice_idx is not None else self.slice_x
            max_idx = self.nx
        else:
            raise ValueError(f"Unknown plane: {plane!r}. Use 'xy', 'xz', or 'yz'.")

        if idx < 0 or idx >= max_idx:
            raise IndexError(f"Slice index {idx} out of range [0, {max_idx}) for plane '{plane}'")

        data = self._active_display_data()
        if data is None:
            raise ValueError("Cannot save image after free(); rebuild the widget first.")
        if plane == "xy":
            slc = data[idx]
        elif plane == "xz":
            slc = data[:, idx, :]
        else:
            slc = data[:, :, idx]

        normalized = self._normalize_slice(slc)
        cmap_fn = colormaps.get_cmap(self.cmap)
        rgba = (cmap_fn(normalized / 255.0) * 255).astype(np.uint8)

        img = Image.fromarray(rgba)
        # PDF requires RGB (no alpha) and the PDF plugin registered.
        if fmt == "pdf":
            Image.init()
            img = img.convert("RGB")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Pass format explicitly so a mismatched extension (e.g. format="tiff"
        # with path="out.bin") still writes the requested container.
        pil_format = {"png": "PNG", "pdf": "PDF", "tiff": "TIFF", "tif": "TIFF"}[fmt]
        img.save(str(path), format=pil_format, dpi=(dpi, dpi))
        return path

    # =========================================================================
    # === Observers ===
    # =========================================================================

    def _on_page_change(self, change: dict) -> None:
        """Move the active flattened panel to the same slot on the new page."""
        if self._syncing_pages or int(self.n_pages) <= 1 or int(self.panels_per_page) <= 0:
            return
        old_page = max(0, min(int(change.get("old", 0)), int(self.n_pages) - 1))
        new_page = max(0, min(int(change.get("new", 0)), int(self.n_pages) - 1))
        slot = int(self.active_panel) - old_page * int(self.panels_per_page)
        slot = max(0, min(slot, int(self.panels_per_page) - 1))
        target = new_page * int(self.panels_per_page) + slot
        self._syncing_pages = True
        try:
            self.active_panel = target
        finally:
            self._syncing_pages = False

    def _on_active_panel_page_change(self, change: dict) -> None:
        """Keep page_idx synchronized when callers select an absolute panel."""
        if self._syncing_pages or int(self.n_pages) <= 1 or int(self.panels_per_page) <= 0:
            return
        target_page = int(change.get("new", 0)) // int(self.panels_per_page)
        target_page = max(0, min(target_page, int(self.n_pages) - 1))
        if target_page == int(self.page_idx):
            return
        self._syncing_pages = True
        try:
            self.page_idx = target_page
        finally:
            self._syncing_pages = False

    def _on_slice_change(self, change: dict) -> None:
        """Recompute slice stats when the user scrubs sliders; skip during playback
        to avoid 12 reductions per frame."""
        if self.playing:
            return
        self._compute_stats()

    def _on_playing_change(self, change: dict) -> None:
        """Refresh stats once playback stops so the displayed values match the
        final slice indices."""
        if not self.playing:
            self._compute_stats()

    def _on_show_stats_change(self, change: dict) -> None:
        """Lazily populate stats traits the first time the caller opts in."""
        if change.get("new"):
            self._compute_stats()

    def _on_export_request_change(self, change: dict) -> None:
        """Handle toolbar export requests from the live notebook frontend."""
        raw = str(change.get("new") or "")
        if not raw:
            return
        try:
            payload = json.loads(raw)
            mode = str(payload.get("mode", "exact"))
            if mode == "clear":
                self.export_payload = b""
                self.export_payload_id = ""
                self.export_filename = ""
                return
            quantized = self._normalise_html_export_options(
                mode=mode,
                encoding=str(payload.get("encoding", "full")),
                downsample=payload.get("downsample"),
                quantized=None,
            )
            if payload.get("download"):
                filename = str(payload.get("filename") or self._default_html_export_path(quantized).name)
                request_id = str(payload.get("id") or "")
                self.export_status = f"Preparing {filename}..."
                html = self._html_export_bytes(quantized=quantized)
                self.export_filename = filename
                self.export_payload = html
                self.export_payload_id = request_id
                size_mb = len(html) / (1024 * 1024)
                label = self._export_mode_label(quantized)
                self.export_status = f"Ready {filename} ({size_mb:.1f} MB, {label})"
            else:
                self.export_status = f"Exporting {mode} HTML..."
                self.export_html(quantized=quantized)
        except Exception as exc:
            self.export_status = f"Export failed: {exc}"

    def _on_slice_alignment_request_change(self, change: dict) -> None:
        """Handle browser requests to estimate global slice alignment."""
        raw = str(change.get("new") or "")
        if not raw:
            return
        try:
            payload = json.loads(raw)
            mode = str(payload.get("mode", "estimate")).lower()
            if mode == "clear":
                return
            if mode == "reset":
                self.reset_slice_alignment()
                return
            if mode != "estimate":
                raise ValueError(f"unknown slice-alignment request mode {mode!r}")
            panel = payload.get("panel", None)
            self.slice_alignment_status = "Estimating slice alignment..."
            result = self.estimate_slice_alignment(
                panel=panel,
                apply=True,
                force=bool(payload.get("force", False)),
            )
            self.slice_alignment_status = (
                f"Aligned row {result['row_shift_px_per_slice']:+.3f}, "
                f"col {result['col_shift_px_per_slice']:+.3f} px/slice"
            )
        except Exception as exc:
            self.slice_alignment_status = f"Slice alignment failed: {exc}"
        finally:
            self._slice_alignment_request = ""

    # =========================================================================
    # === Internal primitives ===
    # =========================================================================

    def _compute_stats(self) -> None:
        """Compute legacy statistics for the 3 axis-aligned slice indices.

        Skipped when show_stats is False to avoid 12 reductions
        per slice movement on multi-MB volumes (JS does not render a stats bar; this
        is for programmatic access only when the caller has opted in).
        """
        data = self._active_display_data()
        if not self.show_stats or data is None:
            return
        slices = [
            data[self.slice_z, :, :],
            data[:, self.slice_y, :],
            data[:, :, self.slice_x],
        ]
        with self.hold_sync():
            self.stats_mean = [float(np.mean(s, dtype=np.float64)) for s in slices]
            self.stats_min = [float(np.min(s)) for s in slices]
            self.stats_max = [float(np.max(s)) for s in slices]
            self.stats_std = [float(np.std(s, dtype=np.float64)) for s in slices]

    def _default_html_export_path(self, quantized: bool) -> pathlib.Path:
        """Build a stable, human-readable export filename in the kernel cwd."""
        label = self.title.strip() or self._widget_name
        slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        if not slug:
            slug = "show3dslices"
        mode = "quantized" if quantized else "exact"
        return pathlib.Path.cwd() / f"{slug}_{self.nz}x{self.ny}x{self.nx}_{mode}.html"

    def _export_mode_label(self, quantized: bool) -> str:
        return "uint8" if quantized else "full float32"

    def _write_html_export(
        self,
        path: str | pathlib.Path,
        *,
        quantized: bool,
        title: str | None = None,
    ) -> pathlib.Path:
        """Write a standalone HTML export without updating toolbar status."""
        from ipywidgets.embed import dependency_state, embed_minimal_html

        from .export import ensure_mobile_viewport

        export_path = pathlib.Path(path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        page_title = title or self.title or self._widget_name
        export_widget = self._clone_for_html_export(quantized=quantized)
        try:
            state = dependency_state([export_widget], drop_defaults=False)
            embed_minimal_html(
                str(export_path),
                views=[export_widget],
                title=page_title,
                drop_defaults=False,
                state=state,
            )
        finally:
            export_widget.free()
        ensure_mobile_viewport(export_path)
        return export_path

    def _html_export_bytes(self, *, quantized: bool) -> bytes:
        """Build a standalone HTML export in a temp directory and return bytes."""
        with tempfile.TemporaryDirectory(prefix="show3dslices-export-") as tmp:
            path = pathlib.Path(tmp) / self._default_html_export_path(quantized).name
            self._write_html_export(path, quantized=quantized)
            return path.read_bytes()

    def _clone_for_html_export(self, *, quantized: bool) -> Self:
        """Create an export-only widget with current state and requested packing."""
        clone_data = self._data
        if int(self.n_pages) > 1 and int(self.panels_per_page) > 0:
            clone_data = self._data.reshape(
                int(self.n_pages),
                int(self.panels_per_page),
                int(self.nz),
                int(self.ny),
                int(self.nx),
            )
        clone = type(self)(
            clone_data,
            title=self.title,
            show_title=self.show_title,
            cmap=self.cmap,
            sampling=list(self.pixel_size_axes),
            show_scale_bar=self.scale_bar_visible,
            z_stretch=self.z_stretch,
            show_controls=self.show_controls,
            controls_collapsed=self.controls_collapsed,
            show_stats=self.show_stats,
            show_crosshair=self.show_crosshair,
            show_fft=self.show_fft,
            fft_window=self.fft_window,
            orthographic=self.orthographic,
            smooth=self.smooth,
            flip=self.flip,
            log_scale=self.log_scale,
            auto_contrast=self.auto_contrast,
            vmin=self.vmin,
            vmax=self.vmax,
            image_vmin_pct=self.image_vmin_pct,
            image_vmax_pct=self.image_vmax_pct,
            slice_alignment=self.slice_alignment,
            row_shift_px_per_slice=self.row_shift_px_per_slice,
            col_shift_px_per_slice=self.col_shift_px_per_slice,
            fps=self.fps,
            loop=self.loop,
            reverse=self.reverse,
            boomerang=self.boomerang,
            play_axis=self.play_axis,
            dim_labels=list(self.dim_labels),
            panel_titles=list(self.panel_titles),
            page_labels=list(self.page_labels) if int(self.n_pages) > 1 else None,
            offline=quantized,
        )
        clone.load_state_dict(self.state_dict())
        clone.export_enabled = False
        clone._export_light = True
        return clone

    def _active_data(self) -> np.ndarray | None:
        """Return the selected 3D panel from the stored panel stack."""
        if self._data is None:
            return None
        panel = max(0, min(int(self.active_panel), int(self.panel_count) - 1))
        return self._data[panel]

    def _active_display_data(self) -> np.ndarray | None:
        """Return the selected panel with display-only slice alignment applied."""
        data = self._active_data()
        if data is None or self.slice_alignment == "off":
            return data
        return _apply_global_slice_alignment_stack(
            data,
            self.row_shift_px_per_slice,
            self.col_shift_px_per_slice,
        )

    def _normalize_slice(self, slc: np.ndarray) -> np.ndarray:
        """Map a 2D slice into a uint8 buffer matching what JS renders. Mirrors
        log_scale, flip, vmin/vmax, and auto_contrast so saved images stay
        pixel-faithful to the on-screen view."""
        if self.log_scale:
            slc = np.sign(slc) * np.log1p(np.abs(slc))
        # Mirror JS path: when flip=True the on-screen renderer negates the data
        # and flips the contrast range (min<->max with sign). Python-side saved
        # images should match what the user sees on screen.
        if self.flip:
            slc = -slc
        if self.vmin is not None and self.vmax is not None:
            vmin = float(self.vmin)
            vmax = float(self.vmax)
            if self.log_scale:
                vmin = float(np.sign(vmin) * np.log1p(abs(vmin)))
                vmax = float(np.sign(vmax) * np.log1p(abs(vmax)))
            if self.flip:
                vmin, vmax = -vmax, -vmin
        elif self.auto_contrast:
            vmin = float(np.percentile(slc, 2))
            vmax = float(np.percentile(slc, 98))
        else:
            vmin = float(slc.min())
            vmax = float(slc.max())
        if vmax > vmin:
            return np.clip((slc - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
        return np.zeros(slc.shape, dtype=np.uint8)
