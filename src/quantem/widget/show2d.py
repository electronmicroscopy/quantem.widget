"""
show2d: 2D comparison viewer with optional FFT and histogram analysis.

For displaying a single image or a gallery of multiple images. Individual list
items may be local frame stacks; unlike Show3D, Show2D does not impose one
shared frame axis across the whole gallery.
"""

import json
import math
import pathlib
import tempfile
import warnings
from enum import StrEnum
from typing import Any, Iterable, Self, Sequence

import anywidget
import ipywidgets
import matplotlib
import matplotlib.patches
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import traitlets
from quantem.widget._image_folder import (
    ImageFolderRecord,
    WatchedImageFolder,
    WatchedImageFolderMixin,
)
from quantem.widget._folder_watch_status import FOLDER_WATCH_STATE_VALUES
from quantem.widget.utils.array import _b64_safe, _resize_image, to_numpy
from quantem.widget.utils.state_io import resolve_widget_version, save_state_file, unwrap_state_payload
from quantem.widget.utils.static_fallback import StaticFallbackMixin
from quantem.widget.utils.ui import UiMode, resolve_ui_mode

_CORE_IMAGE_DATASET_IMPORT_ATTEMPTED = False
_CORE_IMAGE_DATASET_TYPES: tuple[type[Any], ...] = ()


def _core_image_dataset_types() -> tuple[type[Any], ...]:
    """Return core image dataset classes when quantem.core is importable."""
    global _CORE_IMAGE_DATASET_IMPORT_ATTEMPTED, _CORE_IMAGE_DATASET_TYPES
    if not _CORE_IMAGE_DATASET_IMPORT_ATTEMPTED:
        _CORE_IMAGE_DATASET_IMPORT_ATTEMPTED = True
        try:
            from quantem.core.datastructures import Dataset2d, Dataset3d
        except Exception:
            _CORE_IMAGE_DATASET_TYPES = ()
        else:
            _CORE_IMAGE_DATASET_TYPES = (Dataset2d, Dataset3d)
    return _CORE_IMAGE_DATASET_TYPES


# Sentinel for "caller did not pass this kwarg", so a deprecated alias can
# fill an unset denoise kwarg while an explicitly passed new kwarg wins even
# when its value equals the trait default (e.g. denoise="none").
_UNSET = object()

# Saved-state keys from the display_filter-era API mapped onto the denoise
# family, so old .qwstate files and notebooks keep loading.
_DENOISE_STATE_ALIASES = {
    "display_filter": "denoise",
    "display_sigma": "denoise_sigma",
    "spatial_bin": "denoise_bin",
    "display_filters": "denoise_modes",
    "display_sigmas": "denoise_sigmas",
    "spatial_bins": "denoise_bins",
    "display_filter_banner": "denoise_banner",
}

_DEFAULT_FOLDER_PAGE_SIZE = 20


def _reject_unknown_kwargs(cls, kwargs: dict) -> None:
    """Raise TypeError if kwargs contains any key that isn't a declared trait.

    anywidget/traitlets silently accept unknown keys, which let stale notebooks
    pass obsolete params like ``pixel_size_angstrom=0.5`` with no warning.  This
    helper catches typos and renamed-trait references at construction time.
    """
    traits = set(cls.class_trait_names())
    unknown = [k for k in kwargs if k not in traits]
    if unknown:
        key = sorted(unknown)[0]
        raise TypeError(
            f"{cls.__name__}() got unexpected keyword argument {key!r}. "
            f"Check for typos or a renamed parameter (e.g. canvas_size → size, "
            f"image_width_px → size, pixel_size_angstrom → pixel_size)."
        )


def _validate_folder_page_size(page_size: int | None) -> int | None:
    """Return a normalized Show2D folder page size."""
    if page_size is None:
        return None
    if isinstance(page_size, (bool, np.bool_)) or not isinstance(
        page_size,
        (int, np.integer),
    ):
        raise TypeError(
            "page_size must be a positive integer or None; "
            "use None to display the folder as one unpaged gallery"
        )
    value = int(page_size)
    if value < 1:
        raise ValueError(
            f"page_size must be >= 1 or None, got {page_size!r}; "
            "use None to disable folder paging"
        )
    return value


def _is_show2d_page_dict(value: object) -> bool:
    """Return True for a dict-like Show2D page specification."""
    return isinstance(value, dict) and any(key in value for key in ("images", "data", "array"))


def _normalise_show2d_pages(
    data,
    *,
    labels: list[str | None] | None,
    page_labels: Sequence[str | None] | None,
) -> tuple[object, list[str | None] | None, int, int, list[str], list[int]]:
    """Flatten Show2D paged input into the established gallery stack.

    Public page data is ``pages x panels x rows x cols``. Internally this stays
    compatible with Show2D's existing gallery transport by flattening pages into
    one stack and syncing page metadata separately. The browser then renders
    only the active page.
    """
    explicit_page_dicts = (
        isinstance(data, (list, tuple))
        and len(data) > 0
        and all(_is_show2d_page_dict(item) for item in data)
    )
    if explicit_page_dicts:
        page_arrays: list[np.ndarray] = []
        flattened_labels: list[str | None] = []
        inferred_page_labels: list[str] = []
        panels_per_page: int | None = None
        for page_idx, page in enumerate(data):
            raw_images = page.get("images", page.get("data", page.get("array")))
            if raw_images is None:
                raise ValueError(
                    f"Show2D page {page_idx} is missing images/data/array. "
                    "Use {'title': '...', 'images': [...]}"
                )
            arr = to_numpy(raw_images)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            if arr.ndim != 3:
                raise ValueError(
                    f"Show2D page {page_idx} must contain a 2D image or 3D panel stack, "
                    f"got shape {arr.shape}"
                )
            if panels_per_page is None:
                panels_per_page = int(arr.shape[0])
            elif int(arr.shape[0]) != panels_per_page:
                raise ValueError(
                    "Every Show2D page must contain the same number of panels; "
                    f"page 0 has {panels_per_page}, page {page_idx} has {arr.shape[0]}"
                )
            page_arrays.append(arr)
            inferred_page_labels.append(str(page.get("title") or page.get("label") or f"Page {page_idx + 1}"))
            page_panel_labels = page.get("labels")
            if page_panel_labels is not None:
                if len(page_panel_labels) != int(arr.shape[0]):
                    raise ValueError(
                        f"Show2D page {page_idx} labels length ({len(page_panel_labels)}) "
                        f"must match its panel count ({arr.shape[0]})"
                    )
                flattened_labels.extend([None if label is None else str(label) for label in page_panel_labels])
            elif labels is not None:
                if len(labels) == int(arr.shape[0]):
                    flattened_labels.extend([None if label is None else str(label) for label in labels])
                elif panels_per_page is not None and len(labels) == len(data) * panels_per_page:
                    start = page_idx * panels_per_page
                    stop = start + panels_per_page
                    flattened_labels.extend([None if label is None else str(label) for label in labels[start:stop]])
                else:
                    raise ValueError(
                        "labels for paged Show2D must have length panels_per_page "
                        f"({arr.shape[0]}) or n_pages * panels_per_page "
                        f"({len(data) * arr.shape[0]}), got {len(labels)}"
                    )
            else:
                flattened_labels.extend([f"Panel {i + 1}" for i in range(int(arr.shape[0]))])
        stack = np.stack(page_arrays, axis=0)
        n_pages = int(stack.shape[0])
        panels = int(stack.shape[1])
        resolved_page_labels = [
            "" if label is None else str(label)
            for label in (page_labels if page_labels is not None else inferred_page_labels)
        ]
        if len(resolved_page_labels) != n_pages:
            raise ValueError(f"page_labels length ({len(resolved_page_labels)}) must match n_pages ({n_pages})")
        return (
            stack.reshape(n_pages * panels, *stack.shape[-2:]),
            flattened_labels,
            n_pages,
            panels,
            resolved_page_labels,
            [0] * n_pages,
        )

    arr = None
    if not isinstance(data, list):
        try:
            arr = to_numpy(data)
        except Exception:
            arr = None
    if arr is not None and arr.ndim == 4:
        n_pages, panels = int(arr.shape[0]), int(arr.shape[1])
        resolved_page_labels = [
            "" if label is None else str(label)
            for label in (page_labels if page_labels is not None else [f"Page {i + 1}" for i in range(n_pages)])
        ]
        if len(resolved_page_labels) != n_pages:
            raise ValueError(f"page_labels length ({len(resolved_page_labels)}) must match n_pages ({n_pages})")
        if labels is not None:
            if len(labels) == panels:
                resolved_labels = [
                    "" if label is None else str(label)
                    for _ in range(n_pages)
                    for label in labels
                ]
            elif len(labels) == n_pages * panels:
                resolved_labels = [None if label is None else str(label) for label in labels]
            else:
                raise ValueError(
                    "labels for paged Show2D must have length panels_per_page "
                    f"({panels}) or n_pages * panels_per_page ({n_pages * panels}), got {len(labels)}"
                )
        else:
            resolved_labels = [f"Panel {i + 1}" for _ in range(n_pages) for i in range(panels)]
        return (
            arr.reshape(n_pages * panels, *arr.shape[-2:]),
            resolved_labels,
            n_pages,
            panels,
            resolved_page_labels,
            [0] * n_pages,
        )

    return data, labels, 1, 0, [], []


def _round_to_nice(value: float) -> float:
    """Round a physical length to a 'nice' value (1, 2, 5, 10, 20, 50, ...)."""
    if value <= 0:
        return 1.0
    exp = math.floor(math.log10(value))
    base = 10 ** exp
    mantissa = value / base
    if mantissa < 1.5:
        return base
    elif mantissa < 3.5:
        return 2 * base
    elif mantissa < 7.5:
        return 5 * base
    else:
        return 10 * base


def _js_round(value: float) -> int:
    """JS ``Math.round`` (half away from zero for positives); Python's
    banker's rounding would format 2.5 as "2" where the widget shows "3"."""
    return math.floor(value + 0.5)


# Ports of js/figure.ts unit tables so the static PNG's scale bar label is
# character-identical to the live widget's canvas label.
# Length-unit ladder, each as its size in nm: a sub-1 value in one unit
# (e.g. 0.5 nm) displays as a clean integer in a smaller unit (5 Å), because
# microscopists read "5 Å", not "0.50 nm".
_LENGTH_UNITS_NM: tuple[tuple[str, float], ...] = (
    ("mm", 1e6), ("µm", 1e3), ("nm", 1.0), ("Å", 0.1), ("pm", 1e-3),
)
# Base unit (the trait's unit) -> nm. Only length units rescale; anything else
# (mrad, ps, px, ...) keeps its own unit and the decimal fallback.
_BASE_UNIT_NM: dict[str, float] = {
    "mm": 1e6, "µm": 1e3, "μm": 1e3, "micron": 1e3, "microns": 1e3, "um": 1e3,
    "nm": 1.0, "nanometer": 1.0, "nanometers": 1.0,
    "å": 0.1, "angstrom": 0.1, "angstroms": 0.1, "ang": 0.1, "a": 0.1,
    "pm": 1e-3, "picometer": 1e-3, "picometers": 1e-3,
}


def _unit_symbol(unit: str) -> str:
    """Display symbol for a unit string (port of js/figure.ts unitSymbol).

    Users pass units like "micron"/"A" on a Dataset; the widget renders the
    conventional glyph (µm, Å) so labels read like a journal figure. Unknown
    strings pass through unchanged."""
    u = (unit or "").strip()
    lc = u.lower()
    if lc in ("micron", "microns", "um") or u in ("μm", "µm"):
        return "µm"
    if lc in ("angstrom", "angstroms", "ang", "a") or u == "Å":
        return "Å"
    if lc in ("nanometer", "nanometers", "nm"):
        return "nm"
    if lc in ("picometer", "picometers", "pm"):
        return "pm"
    if lc in ("millimeter", "millimeters", "mm"):
        return "mm"
    if lc in ("picosecond", "picoseconds", "ps"):
        return "ps"
    if lc in ("femtosecond", "femtoseconds", "fs"):
        return "fs"
    if lc in ("nanosecond", "nanoseconds", "ns"):
        return "ns"
    return u


def _format_scale_label(value: float, unit: str) -> str:
    """Scale bar label (port of js/figure.ts formatScaleLabel).

    Length values auto-pick the unit that reads as a clean integer:
    0.5 nm -> "5 Å", 0.005 nm -> "5 pm". Non-length units (mrad, ps, px)
    keep their unit. ``_round_to_nice`` gives n*10^k and every ladder step
    is a power of 10, so the rescaled number is always exact."""
    nice = _round_to_nice(value)
    base_nm = _BASE_UNIT_NM.get((unit or "").strip().lower())
    if base_nm is None:
        sym = _unit_symbol(unit)
        return f"{_js_round(nice)} {sym}" if nice >= 1 else f"{nice:.2f} {sym}"
    value_nm = nice * base_nm
    # largest ladder unit where the value is >= 1 -> the fewest-digit integer
    for sym, unit_nm in _LENGTH_UNITS_NM:
        if value_nm / unit_nm >= 1:
            return f"{_js_round(value_nm / unit_nm)} {sym}"
    sym, unit_nm = _LENGTH_UNITS_NM[-1]
    return f"{_js_round(value_nm / unit_nm)} {sym}"


# Rec. 709 luma weights: the standard perceptual grayscale reduction of RGB.
_RGB_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _is_rgb_item(item: np.ndarray) -> bool:
    """True when a gallery item is an ``(H, W, 3)`` / ``(H, W, 4)`` color image.

    Detection rule (documented in the Show2D docstring): inside a LIST input,
    any item with ``ndim == 3`` and a trailing dim of 3 or 4 is RGB(A). A bare
    3-D ARRAY input keeps ``(N, H, W)`` stack semantics unless its trailing dim
    is 3/4 AND its leading dim is > 4 (so a 3-frame stack of tiny images never
    silently flips to RGB)."""
    return item.ndim == 3 and item.shape[-1] in (3, 4)


def _normalize_rgb(item: np.ndarray) -> np.ndarray:
    """Display-ready ``(H, W, 3)`` float32 in [0, 1] from an RGB(A) item.

    uint8 input scales by 1/255; float input is clipped to [0, 1] (overlays
    like drift's green-magenta ``overlay_pair`` are already in that range).
    An alpha channel is dropped: RGB panels bypass the contrast pipeline and
    are composited on an opaque canvas, so alpha has nothing to blend with."""
    rgb = np.asarray(item)[..., :3]
    if rgb.dtype == np.uint8:
        return rgb.astype(np.float32) / 255.0
    return np.clip(rgb.astype(np.float32), 0.0, 1.0)


def _normalize_grayscale_panel_items(
    images: Sequence[np.ndarray],
    panel_frame_indices: Sequence[int] | None = None,
) -> tuple[list[np.ndarray], list[int], np.ndarray]:
    """Normalize a mixed static/stack gallery to per-panel ``(F, H, W)`` arrays.

    A 2-D item is a static one-frame panel. A 3-D item is a local frame stack
    for that panel. Spatial shapes are center-padded to one common gallery size,
    while frame counts remain independent.
    """
    if not images:
        raise ValueError("Show2D requires at least one image panel")

    stacks: list[np.ndarray] = []
    for panel, image in enumerate(images):
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        elif arr.ndim != 3:
            raise ValueError(
                "Show2D list items must be 2D images or 3D (frames, rows, cols) "
                f"stacks; panel {panel} has shape {arr.shape}"
            )
        if 0 in arr.shape:
            raise ValueError(
                f"Show2D panel {panel} is empty (shape {arr.shape}); all dimensions must be >= 1"
            )
        if np.iscomplexobj(arr):
            raise TypeError(
                f"Show2D panel {panel} contains complex data. Convert first with "
                "np.abs(arr) for magnitude or np.angle(arr) for phase."
            )
        stack = np.array(arr, dtype=np.float32, copy=True)
        if not np.isfinite(stack).all():
            raise ValueError(
                f"Show2D panel {panel} contains NaN or inf. Clean first with "
                "np.nan_to_num(arr, nan=0, posinf=0, neginf=0)."
            )
        stacks.append(stack)

    target_h = max(int(stack.shape[-2]) for stack in stacks)
    target_w = max(int(stack.shape[-1]) for stack in stacks)
    normalized: list[np.ndarray] = []
    for stack in stacks:
        if stack.shape[-2:] == (target_h, target_w):
            normalized.append(stack)
        else:
            normalized.append(
                np.stack(
                    [_resize_image(frame, target_h, target_w) for frame in stack],
                    axis=0,
                ).astype(np.float32, copy=False)
            )

    if panel_frame_indices is None:
        indices = [0] * len(normalized)
    else:
        if len(panel_frame_indices) != len(normalized):
            raise ValueError(
                "panel_frame_indices length "
                f"({len(panel_frame_indices)}) must match panel count ({len(normalized)})"
            )
        indices = []
        for panel, (raw_index, stack) in enumerate(zip(panel_frame_indices, normalized)):
            index = int(raw_index)
            if index < 0:
                index += int(stack.shape[0])
            if index < 0 or index >= int(stack.shape[0]):
                raise ValueError(
                    f"panel_frame_indices[{panel}]={raw_index} is outside the valid "
                    f"range [0, {stack.shape[0]})"
                )
            indices.append(index)

    current = np.stack(
        [stack[index] for stack, index in zip(normalized, indices)],
        axis=0,
    ).astype(np.float32, copy=False)
    return normalized, indices, current


def _compose_overlay_pair(reference: np.ndarray, moving: np.ndarray, mode: str) -> np.ndarray:
    """Color overlay of two grayscale images for checking alignment.

    Math parity with ``quantem.imaging.drift.plot.overlay_pair`` (replicated
    locally: the widget must not depend on quantem's drift module). Both images
    share ONE 1/99-percentile scale so the two colors match in brightness;
    per-image scaling would unbalance them. ``green-magenta`` is the
    colorblind-safe default: reference -> magenta, moving -> green, aligned ->
    white. ``rgb`` gives classic red/green (aligned -> yellow) on request only.
    """
    ref = np.asarray(reference, dtype=np.float32)
    mov = np.asarray(moving, dtype=np.float32)
    lo = min(float(np.percentile(ref, 1)), float(np.percentile(mov, 1)))
    hi = max(float(np.percentile(ref, 99)), float(np.percentile(mov, 99)))
    def _norm(arr):
        return np.clip((arr - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    ref, mov = _norm(ref), _norm(mov)
    if mode == "rgb":
        return np.stack([ref, mov, np.zeros_like(ref)], -1)  # reference red, moving green, aligned yellow
    return np.stack([ref, mov, ref], -1)                     # reference magenta, moving green, aligned white


def _compose_dual(
    map_a_01: np.ndarray, map_b_01: np.ndarray, gain: Sequence[float]
) -> np.ndarray:
    """Two-map magenta+green composite for underlay ``mode='dual'``.

    Same colorblind-safe convention as :func:`_compose_overlay_pair`
    ``green-magenta``: map A -> magenta (red+blue), map B -> green, co-located
    signal -> white. The inputs are already display-normalized to [0, 1]; each
    channel is then scaled by its own ``gain`` so a weak element can be lifted
    against a strong one. The stored arrays are never touched.
    """
    a = np.clip(np.asarray(map_a_01, dtype=np.float32) * float(gain[0]), 0.0, 1.0)
    b = np.clip(np.asarray(map_b_01, dtype=np.float32) * float(gain[1]), 0.0, 1.0)
    return np.stack([a, b, a], axis=-1).astype(np.float32)  # A magenta, B green, both -> white


_OVERLAY_FONT: list[str] | None = None


def _static_overlay_font() -> list[str]:
    """Closest installed match to the widget's ``-apple-system, BlinkMacSystemFont,
    'Segoe UI', sans-serif`` stack, resolved once. Filtering to installed fonts
    avoids matplotlib findfont warnings on machines without Helvetica/Arial."""
    global _OVERLAY_FONT
    if _OVERLAY_FONT is None:
        from matplotlib import font_manager
        installed = {f.name for f in font_manager.fontManager.ttflist}
        preferred = ("Helvetica Neue", "Segoe UI", "Arial", "Liberation Sans")
        _OVERLAY_FONT = [f for f in preferred if f in installed] + ["DejaVu Sans"]
    return _OVERLAY_FONT


class Colormap(StrEnum):
    INFERNO = "inferno"
    VIRIDIS = "viridis"
    MAGMA = "magma"
    PLASMA = "plasma"
    GRAY = "gray"


_MAX_PANEL_PLAYBACK_FPS = 30.0


class Show2D(WatchedImageFolderMixin, StaticFallbackMixin, anywidget.AnyWidget):
    """
    2D image comparison viewer with optional local panel stacks and analysis.

    Display a single image or multiple images in a gallery layout. A 3-D item
    inside a list is an independent local stack for that panel, with its own
    in-panel slider and playback. A bare 3-D array remains a static gallery;
    use Show3D when every panel should share one global frame axis.

    Parameters
    ----------
    data : array_like
        2D array (height, width) for single image, or
        3D array (N, height, width) for multiple images displayed as gallery.
        A Python list may mix 2D images with 3D ``(frames, height, width)``
        arrays. Each grayscale 3D list item is one gallery panel with an
        independent slider, playback state, frame count, and current frame;
        local frame counts do not need to match between panels.
        RGB images are first-class: inside a LIST input, any item shaped
        ``(H, W, 3)`` / ``(H, W, 4)`` is treated as an RGB(A) color image
        (uint8 or float in [0, 1]; alpha is dropped). A bare 3-D ARRAY keeps
        the historical ``(N, H, W)`` stack semantics unless its trailing dim
        is 3/4 AND its leading dim is > 4, in which case it is a single RGB
        image. Mixed galleries (grayscale + RGB side by side) are supported:
        RGB panels are display-ready and bypass the colormap, contrast,
        auto/log pipeline; their stats row, histogram, FFT, and line profile
        read the Rec. 709 luminance; the hover readout shows the (r, g, b)
        triplet. ``offline=True``, ``export_html`` and ``rotate`` are not
        supported when RGB panels are present.
    labels : list of str, optional
        Labels for each image in gallery mode.
    title : str, optional
        Title to display above the image(s).
    cmap : str, default "inferno"
        Colormap name ("magma", "viridis", "gray", "inferno", "plasma").
    sampling : float or tuple of float, optional
        Pixel size per axis ``(row, col)``. Scalar broadcasts to both axes.
        Used for scale bar display. Defaults to ``(1, 1)``.
    units : str or list of str, optional
        Unit string per axis. Scalar broadcasts to both. Common: ``"A"``,
        ``"nm"``, ``"pixels"``. Defaults to ``["pixels", "pixels"]``.
    show_fft : bool, default False
        Show FFT and histogram panels. Every interactive FFT panel includes its
        current browser-side zoom multiplier (for example ``2.0×`` or
        ``5.0×``), including independently zoomed gallery FFTs.
    fft_metrics : bool, default True
        Show compact FFT quality labels inside FFT panels when FFT is visible.
        The frontend computes these labels from the cached FFT magnitude and
        does not trigger another FFT.
    show_controls : bool, default True
        Show the live control UI. Set ``False`` for a permanently clean display.
    controls_collapsed : bool, default False
        Start with the live control UI collapsed behind a small GUI toggle.
        Unlike ``show_controls=False``, users can expand the controls in the
        frontend and Python can call ``expand_controls()`` later.
    show_stats : bool, default True
        Show statistics (mean, min, max, std).
    debug : bool, default False
        Show a compact frontend FPS/debug badge in the widget title row.
    log_scale : bool, default False
        Use log scale for intensity mapping.
    auto_contrast : bool, default False
        Use percentile-based contrast.
    vmin : float, optional
        Absolute minimum intensity for color mapping. When both vmin and vmax
        are set, all gallery images share the same intensity scale: essential
        for A/B visual comparison.
    vmax : float, optional
        Absolute maximum intensity for color mapping.
    diff_mode : bool, default False
        Append a live signed-difference panel to a grayscale gallery (mirrors
        ``overlay``): the gallery becomes ``[a, b, a - b]`` for a pair, using a
        symmetric diverging colormap centered on zero so positive and negative
        residuals read with equal weight. The reference frame defaults to panel
        0 (change it with the ``diff_reference`` trait); the diff recomputes
        whenever either frame or the reference changes, so it tracks scrubbing
        stacks live. With more than two frames one diff panel is appended per
        non-reference grayscale frame (``#ref - #other``). RGB panels never get
        a diff panel. Combine with ``overlay=True`` for ``[a, b, overlay]`` plus
        the diff panel.
    overlay : bool or str, default False
        Compose an alignment overlay panel from a 2-image grayscale gallery
        (mirrors ``diff_mode``): the gallery becomes ``[a, b, overlay]`` with
        image 0 -> magenta, image 1 -> green, aligned -> white (the
        colorblind-safe default; same math as quantem drift's
        ``overlay_pair``: one shared 1/99-percentile scale across both
        images). ``overlay="rgb"`` gives classic red/green (aligned ->
        yellow). Requires exactly 2 grayscale images. Combine with
        ``diff_mode=True`` for ``[a, b, overlay]`` plus the dynamic signed
        diff panel of the pair.
    underlay : bool or str, default False
        Compose chemistry-on-structure from exactly two grayscale inputs
        ``[haadf, map]``: the gallery becomes ``[haadf, map, map on HAADF]``,
        blending the element map (magenta) onto the HAADF (gray structure) as
        a third RGB panel so sparse EDS signal reads against the lattice
        without washing bright columns to white. Pass ``True`` or ``"haadf"``.
        Requires exactly two single-frame grayscale images (no per-panel frame
        stacks) and is mutually exclusive with ``overlay``. The two sources are
        never modified; only the composed panel is added. See ``underlay_alpha``
        and ``underlay_haadf_gain`` to tune the blend.
    underlay_alpha : float, default 0.95
        Opacity of the element map over the HAADF in the ``underlay`` blend, in
        ``[0, 1]``. Higher means more chemistry, less structure; the slider
        re-blends live without touching the sources.
    underlay_haadf_gain : float, default 0.35
        Brightness of the HAADF structure showing through the ``underlay``
        blend, in ``[0, 1]``. Lower keeps the map colors saturated over a dim
        lattice; higher lets more of the gray structure through.
    underlay_mode : {"haadf", "dual"}, default "haadf"
        Composite recipe for the third ``underlay`` panel. ``"haadf"`` blends a
        single element map (magenta) onto the HAADF lattice. ``"dual"`` takes
        two grayscale maps ``[map A, map B]`` (no HAADF) and composes a
        colorblind-safe magenta+green panel: map A -> magenta, map B -> green,
        co-located signal -> white. Both modes need exactly two single-frame
        grayscale inputs.
    stretch_percentiles : sequence of float, default (4.0, 99.0)
        Low/high display-stretch percentiles applied to the element map(s)
        before colorizing, matching the drift-paper Fig4 sweep. The slider
        re-stretches live without touching the stored counts; must satisfy
        ``0 <= low < high <= 100``.
    display_gamma : float, default 0.75
        Presence gamma inside the ``"haadf"`` blend, > 0. Below 1 lifts
        mid-count columns into color; above 1 keeps only the brightest lit.
        Ignored in ``"dual"`` mode (the magenta+green composite has no ghost).
    dual_gain : sequence of float, default (1.0, 1.0)
        Per-channel brightness ``[gain A, gain B]`` for the ``"dual"`` composite,
        each >= 0. Raise one channel to balance a weak element against a strong
        one; ignored in ``"haadf"`` mode.
    ncols : int, default 3
        Number of columns in gallery mode.
    panel_frame_indices : sequence of int, optional
        Initial frame for every panel when ``data`` is a list containing local
        3-D stacks. Each value is resolved against that panel's own frame count.
        Static 2-D panels accept only ``0``. Negative indices follow Python
        indexing, so ``-1`` starts a stack on its final frame.
    panel_playback_fps : float, default 10
        Playback speed shared by the independent local-stack play buttons.
        Configure it at construction without adding another toolbar control.
        Values above 30 are capped at the browser playback budget.
    size : int, default 0
        Canvas rendering size in CSS pixels (the on-screen width of each image).
        ``0`` uses the frontend default: 500 px for a single image, 300 px per
        image in gallery mode.  Pass e.g. ``size=800`` to enlarge for a
        presentation, or ``size=200`` to compress alongside a control panel.
        This controls **display only**: the underlying image resolution is
        never resampled; zooming into a 4K image preserves every pixel.
    save_state : bool, default False
        When False, saved notebooks omit heavy image buffers and keep a compact
        static preview for cold reopen. Set True only for small widgets that
        must reopen interactively without rerunning the kernel.
    notebook_preview_format : {"jpeg", "webp", "png"} or None, default "jpeg"
        Static preview format used when ``save_state=False``. ``"jpeg"`` is the
        most portable notebook default, ``"webp"`` is smaller for local/report
        workflows, ``"png"`` is lossless but larger, and ``None`` disables the
        preview.
    notebook_preview_quality : int, default 88
        Lossy preview quality for JPEG/WebP, from 1 to 100. Ignored for PNG.
    notebook_preview_max_px : int, default 512
        Longest panel side for the saved-notebook preview. Lower values make
        notebooks smaller; higher values make the static fallback sharper.
    denoise : str or sequence of str, default "none"
        Display-only denoise method for sparse maps (EDS, low dose). Three
        orthogonal choices: ``"none"``, ``"gaussian"``, or ``"anscombe"``
        (Poisson count-respecting smoothing); binning is the separate
        ``denoise_bin`` knob. Recommendation ladder: sparse EDS ->
        ``"anscombe"`` with ``denoise_bin=2`` and sigma 6-10; very sparse ->
        ``"anscombe"`` with ``denoise_bin=4`` and sigma 8-12; decent-dose
        HAADF -> ``"gaussian"`` sigma 1-2 or ``"none"``; anything
        quantitative -> ``"none"``. The compound spellings ``"bin2"``,
        ``"bin2_anscombe"`` and ``"bin4_anscombe"`` stay accepted as aliases
        that fold into (mode, bin); ``"tv"``/``"denova*"`` remain available
        from Python (not in the UI menu). A scalar applies to every panel; a
        sequence (one entry per panel) gives each panel its own method, e.g.
        ``["none", "anscombe"]`` for a raw vs denoised A/B gallery. Pure view
        transform: the stored array, the stats row, and every export of raw
        data keep the original counts, and the lossless default is
        ``"none"``. When active, a one-line banner announces the reduction
        and how to get raw counts back. RGB panels are never filtered.
        Independent of ``display_bin`` (the GPU display budget knob). The
        per-panel ``denoise_modes`` / ``denoise_sigmas`` / ``denoise_bins``
        lists are the source of truth; the scalar ``denoise`` /
        ``denoise_sigma`` / ``denoise_bin`` traits are the UI editor and,
        while ``denoise_scope == "panel"``, mirror only the selected panel.
        Set per-panel values imperatively with :meth:`set_denoise`.
    denoise_sigma : float or sequence of float, default 4.0
        Smoothing scale in pixels for the Gaussian/Anscombe display filters.
        A sequence sets one sigma per panel.
    denoise_bin : {1, 2, 4} or sequence, default 1
        Display-side 2x bin passes for SNR, combined with ``denoise``.
        ``1`` (the default) is lossless. A sequence sets one bin factor per
        panel. This is the SNR knob for sparse maps (EDS, low dose): it trades
        resolution for counts. It is orthogonal to ``display_bin``, which is a
        performance-only downsample to fit the GPU display budget and does not
        change reported intensities. Reach for ``denoise_bin`` to see faint
        chemistry, ``display_bin`` only to render a huge gallery faster.
    show_denoise : bool, default False
        Shows the denoise controls row; does not itself denoise - use
        ``denoise=`` for that. Hidden by default to keep the widget clean;
        auto-enabled when any panel starts with an active denoise. An active
        reduction always shows its banner, even with the row hidden.
    denoise_scope : {"all", "panel"}, default "all"
        UI knob scope: "all" applies Denoise/σ/Bin edits to every panel,
        "panel" edits only the selected panel. Passing any per-panel sequence
        switches to the "panel" scope automatically. In gallery mode the
        toggle lives in the Link group (Link Zoom / Pan / Contrast / Denoise):
        checked means linked ("all"), unchecked means per panel.

        .. deprecated::
            The ``display_filter``-era kwargs ``display_filter``,
            ``display_sigma``, ``spatial_bin`` and ``filter_per_panel`` are
            still accepted for one release and map onto ``denoise``,
            ``denoise_sigma``, ``denoise_bin`` and ``denoise_scope``
            respectively. Passing any of them emits a ``DeprecationWarning``;
            if both a new kwarg and its deprecated alias are given, the new
            kwarg wins. In particular ``spatial_bin`` maps onto ``denoise_bin``
            (the SNR knob), not ``display_bin`` (the performance downsample).
    pad_ratio : float, default 0.0
        Ratio-based border added on each side of the displayed frame, as a
        fraction of the image's max(rows, cols). Valid range 0 to 1. The
        border value is the frame minimum, which keeps the colormap floor.
        Display-only and reversible (single-panel widgets only): combine
        with :meth:`crop_to_view` and undo with :meth:`reset_view_ops`.
        When active, a one-line ``view:`` banner announces the reduction.
    Attributes
    ----------
    render_total_ms : int or None
        End-to-end wall clock from constructor start to first browser paint,
        populated by a JS→Python round-trip after the first canvas render.
        ``None`` until the browser has actually painted; also printed to stdout
        when it fires.  Use to triage "is it Python, wire, or the browser?"
        during live acquisitions.
    render_python_build_ms : int or None
        Subset of ``render_total_ms`` covering Python ``__init__`` only.
    render_wire_js_ms : int or None
        Subset covering everything after Python returns: Comm transfer, JS
        decode, colormap, and canvas paint.

    Examples
    --------
    >>> import numpy as np
    >>> from quantem.widget import Show2D

    Single 2D NumPy array:

    >>> Show2D(np.random.rand(512, 512))

    PyTorch tensor (CPU or GPU, any dtype):

    >>> import torch
    >>> Show2D(torch.rand(512, 512))

    3D NumPy stack ``(N, H, W)`` rendered as a gallery:

    >>> Show2D(np.random.rand(6, 256, 256), ncols=3)

    List of arrays with different shapes (center-padded to a common canvas):

    >>> Show2D([np.random.rand(256, 256), np.random.rand(300, 400)])

    One static map beside an independently scrubbed HAADF stack:

    >>> haadf = np.random.rand(26, 256, 256)
    >>> Show2D([np.random.rand(256, 256), haadf],
    ...        labels=["EDS map", "HAADF"], panel_frame_indices=[0, -1])

    Four tomography reconstructions with independent slice counts and sliders:

    >>> reconstructions = [
    ...     np.random.rand(n_slices, 64, 64)
    ...     for n_slices in (8, 12, 16, 20)
    ... ]
    >>> labels = ["baseline", "regularized", "fine z", "alternative"]
    >>> w = Show2D(reconstructions, labels=labels,
    ...            panel_frame_indices=[3, 5, 7, -1],
    ...            panel_playback_fps=4, ncols=2, show_fft=True)
    >>> _ = w.set_panel_frame("fine z", 8)

    quantem ``Dataset2d``: title, sampling, units auto-extracted:

    >>> from quantem.core.datastructures import Dataset2d
    >>> ds = Dataset2d.from_array(np.random.rand(512, 512))
    >>> Show2D(ds)

    quantem ``Dataset3d``: gallery view of N frames with calibration:

    >>> from quantem.core.datastructures import Dataset3d
    >>> ds = Dataset3d.from_array(np.random.rand(6, 256, 256))
    >>> Show2D(ds, ncols=3)

    A/B comparison with shared contrast and linked zoom/pan:

    >>> a, b = np.random.rand(512, 512), np.random.rand(512, 512)
    >>> Show2D([a, b], vmin=0, vmax=1, link_zoom=True, link_pan=True)

    Per-image absolute contrast (one ``vmin``/``vmax`` per image):

    >>> Show2D([a, b], vmin=[0.0, 0.2], vmax=[1.0, 0.8])

    Drift comparison: diff mode adds a ``A - B`` panel alongside the originals
    (gallery becomes ``[A, B, A - B]``):

    >>> Show2D([a, b], diff_mode=True, link_zoom=True, link_pan=True)

    Alignment overlay: append a composed green-magenta RGB panel
    (gallery becomes ``[a, b, overlay]``; aligned regions read white):

    >>> Show2D([a, b], overlay=True)

    RGB images are first-class gallery items (e.g. drift's green-magenta
    ``overlay_pair`` composite next to the grayscale merge):

    >>> rgb_overlay = np.stack([a, b, a], axis=-1)  # (H, W, 3) in [0, 1]
    >>> Show2D([a, rgb_overlay], labels=["merged", "overlay (RGB)"])

    Large image: display-only canvas size (full resolution preserved):

    >>> Show2D(np.random.rand(4096, 4096), size=800)

    Per-panel display width for galleries. Use ``ncols`` to choose an
    intentional gallery shape, for example ``ncols=2`` for a 2×2 comparison
    of four images or ``ncols=4`` for a single row:

    >>> Show2D(np.random.rand(13, 128, 128), ncols=13, panel_width_px=70)

    Static export to PDF or PNG (vector PDF for publication figures):

    >>> w = Show2D(np.random.rand(512, 512), sampling=0.5, units="nm")
    >>> w.save_image("figure.pdf", dpi=150)

    Denoise a sparse EDS map for display (raw counts stay untouched):

    >>> eds_map = np.random.poisson(0.3, (256, 256)).astype(np.float32)
    >>> Show2D(eds_map, denoise="anscombe", denoise_bin=2, denoise_sigma=8)

    Raw vs denoised A/B gallery from one call (per-panel ``denoise`` list):

    >>> Show2D([eds_map, eds_map], denoise=["none", "anscombe"], denoise_sigma=8)

    Chemistry on structure: blend an element map onto HAADF (magenta on gray):

    >>> haadf = np.random.rand(256, 256).astype(np.float32)
    >>> Show2D([haadf, eds_map], underlay=True)

    Zoom into a feature, commit that window as the display, then undo it:

    >>> w = Show2D(eds_map, view_box=(64, 64, 96))  # zoom to a 96x96 region
    >>> w.crop_to_view()      # the window becomes the whole displayed frame
    >>> w.reset_view_ops()    # back to the full frame, bit-identical
    """

    _esm = pathlib.Path(__file__).parent / "static" / "show2d.js"

    # =========================================================================
    # Core State
    # GPU memory budget for display buffers (MB). Each 4K image needs ~192 MB.
    # 12×4K = 2304 MB fits. 16+ triggers auto-bin.
    _GPU_DISPLAY_BUDGET_MB = 2500
    # Wire budget for the initial frame payload, per panel. A 6144x6144 float32
    # panel is 151 MB and Jupyter's kernel->browser channel moves ~24 MB/s, so
    # shipping it wholesale stalls first paint for ~6 s while the ~1000 px
    # canvas can only show ~1 MP of it. Panels above this budget send a binned
    # preview instead; the browser streams full-res detail for the visible
    # window on zoom (maps-style). 16 MiB keeps a plain 2048x2048 float32
    # image (exactly 16 MiB) on the classic full-payload path.
    _WIRE_BUDGET_BYTES_PER_PANEL = 16 * 1024 * 1024
    # Cap on one detail reply so a zoom refetch stays sub-second on the slow
    # Tornado channel. The JS coarsens the tile bin until the crop fits.
    _DETAIL_BUDGET_BYTES = 8 * 1024 * 1024

    # =========================================================================
    widget_version = traitlets.Unicode("unknown").tag(sync=True)
    n_images = traitlets.Int(1).tag(sync=True)
    folder_waiting = traitlets.Bool(False).tag(sync=True)
    folder_status = traitlets.Unicode("").tag(sync=True)
    folder_watch_state = traitlets.Enum(
        values=FOLDER_WATCH_STATE_VALUES,
        default_value="hidden",
    ).tag(sync=True)
    folder_watch_detail = traitlets.Unicode("").tag(sync=True)
    n_pages = traitlets.Int(1).tag(sync=True)
    page_idx = traitlets.Int(0).tag(sync=True)
    panels_per_page = traitlets.Int(0).tag(sync=True)
    page_kind = traitlets.Enum(
        ["comparison", "items"],
        default_value="comparison",
    ).tag(sync=True)
    page_labels = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    page_starred = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    height = traitlets.Int(1).tag(sync=True)
    width = traitlets.Int(1).tag(sync=True)
    _display_bin_factor = traitlets.Int(1).tag(sync=True)  # 1 = full-res, 2/4/8 = binned
    # Display-only denoise/bin for sparse maps (EDS, low dose). View transform
    # applied while packing frame_bytes; the stored data is never modified and
    # the lossless default is "none". Independent of _display_bin_factor (GPU
    # budget); denoise_bin here is the EDS bin-for-SNR knob.
    denoise = traitlets.Unicode("none").tag(sync=True)
    denoise_sigma = traitlets.Float(4.0).tag(sync=True)
    denoise_bin = traitlets.Int(1).tag(sync=True)
    # Per-panel resolved knobs: one entry per panel, the packing source of
    # truth. Constructed from scalar (broadcast) or sequence kwargs; the
    # scalar traits above are the UI-facing editor whose scope is controlled
    # by denoise_scope ("all" broadcasts, "panel" edits the selected panel).
    denoise_modes = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    denoise_sigmas = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    denoise_bins = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    denoise_scope = traitlets.Unicode("all").tag(sync=True)
    denoise_banner = traitlets.Unicode("").tag(sync=True)
    # The denoise controls row is hidden by default; this toggle shows it.
    # Auto-enabled at construction when any panel starts with an active
    # denoise so the knobs that explain the view are immediately visible.
    show_denoise = traitlets.Bool(False).tag(sync=True)
    # Master ON/OFF of the denoise EFFECT (the "Denoise" toggle). Off shows the
    # raw view; the per-panel config (modes/sigmas/bins) is PRESERVED, just
    # gated, so toggling back on restores it. Distinct from show_denoise, which
    # is only the editor-row visibility.
    denoise_enabled = traitlets.Bool(True).tag(sync=True)
    # Browser-side filter negotiation: JS sets this True when a real (non
    # software) WebGPU adapter is available. Python then ships RAW frames for
    # panels whose mode the browser can evaluate (gaussian/bin2/anscombe
    # stacks; see BROWSER_DISPLAY_FILTER_MODES) and the WGSL compute port in
    # js/displayFilter.ts filters client-side, so the sigma slider scrubs live
    # with no kernel round trip and kernel-less HTML exports keep working
    # knobs. tv/denova* panels always stay on this Python path.
    _webgpu_filter_ok = traitlets.Bool(False).tag(sync=True)
    # Chemistry-on-structure view: HAADF-modulated blend of an element map on
    # the HAADF lattice as a third RGB panel (haadf | map | blend). Enabled at
    # construction with underlay=True on exactly two grayscale inputs.
    underlay = traitlets.Bool(False).tag(sync=True)
    underlay_alpha = traitlets.Float(0.95).tag(sync=True)
    underlay_haadf_gain = traitlets.Float(0.35).tag(sync=True)
    # Fig4 parity knobs, all live-scrubbable: the map's display stretch
    # (low/high percentile), the presence gamma inside the HAADF blend, the
    # composite mode ('haadf' = map-on-HAADF, 'dual' = two-map magenta+green),
    # and per-channel gains for the dual composite.
    stretch_percentiles = traitlets.List(
        traitlets.Float(), default_value=[4.0, 99.0]
    ).tag(sync=True)
    display_gamma = traitlets.Float(0.75).tag(sync=True)
    underlay_mode = traitlets.Unicode("haadf").tag(sync=True)
    dual_gain = traitlets.List(traitlets.Float(), default_value=[1.0, 1.0]).tag(sync=True)
    _gpu_max_buffer_mb = traitlets.Int(0).tag(sync=True)  # GPU reports maxBufferSize (JS→Python)
    # Flipped True by JS after the first colormap pass has painted to canvas.
    # Used by the Python-side truthful timing print (end-to-end wall clock, not just __init__).
    _js_rendered = traitlets.Bool(False).tag(sync=True)
    frame_bytes = traitlets.Bytes(b"").tag(sync=True)
    # Optional per-panel frame stacks. Static panels keep count=1 and are not
    # duplicated in panel_stack_bytes; offsets are -1 for those panels.
    panel_frame_counts = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    panel_frame_indices = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    panel_playback_fps = traitlets.Float(10.0).tag(sync=True)
    panel_stack_offsets = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    panel_stack_bytes = traitlets.Bytes(b"").tag(sync=True)
    _panel_stack_mins = traitlets.List(trait=traitlets.Float(), default_value=[]).tag(sync=True)
    _panel_stack_maxs = traitlets.List(trait=traitlets.Float(), default_value=[]).tag(sync=True)
    # Offline mode: stack quantized to uint8 against global (min, max). 4x
    # smaller than float32 — drops standalone HTML from ~200 MB to ~110 MB.
    # JS dequantizes on read. Eye can't tell uint8 from float32 after colormap
    # reduces to 256 levels anyway.
    offline = traitlets.Bool(False).tag(sync=True)
    # True only on a clone written by export_html: forces the standalone HTML to
    # render on a light/white background regardless of the viewer's OS theme.
    _export_light = traitlets.Bool(False).tag(sync=True)
    _static_fallback_jpeg = traitlets.Unicode("").tag(sync=True)
    _static_fallback_mime = traitlets.Unicode("image/jpeg").tag(sync=True)
    _offline_min = traitlets.Float(0.0).tag(sync=True)
    _offline_max = traitlets.Float(1.0).tag(sync=True)
    # Per-image quantization ranges. A gallery's panels can span very different
    # intensities; one GLOBAL (min, max) then wastes most of the 256 uint8 codes
    # on whichever panel is widest, so a narrow panel keeps only a handful of
    # levels and its histogram shows a coarse comb. Quantizing each panel against
    # its OWN (min, max) gives every panel the full 256 codes -> clean histogram.
    _offline_mins = traitlets.List(trait=traitlets.Float(), default_value=[]).tag(sync=True)
    _offline_maxs = traitlets.List(trait=traitlets.Float(), default_value=[]).tag(sync=True)
    # Maps-style detail streaming (active whenever the preview is binned, i.e.
    # _display_bin_factor > 1): JS writes a JSON request describing the visible
    # window per panel; Python replies with cropped + binned float32 tiles.
    # Request/response over traits, same pattern as export_request below.
    _detail_request = traitlets.Unicode("").tag(sync=True)
    _detail_meta = traitlets.Unicode("").tag(sync=True)
    _detail_bytes = traitlets.Bytes(b"").tag(sync=True)
    export_request = traitlets.Unicode("").tag(sync=True)
    export_status = traitlets.Unicode("").tag(sync=True)
    export_enabled = traitlets.Bool(True).tag(sync=True)
    export_payload = traitlets.Bytes(b"").tag(sync=True)
    export_payload_id = traitlets.Unicode("").tag(sync=True)
    export_filename = traitlets.Unicode("").tag(sync=True)
    handoff_request = traitlets.Unicode("").tag(sync=True)
    handoff_status = traitlets.Unicode("").tag(sync=True)
    handoff_enabled = traitlets.Bool(True).tag(sync=True)
    prepared_view_widget = traitlets.Instance(ipywidgets.Widget, allow_none=True).tag(
        sync=True,
        **ipywidgets.widget_serialization,
    )
    labels = traitlets.List(traitlets.Unicode()).tag(sync=True)
    # Per-panel RGB flag. True panels carry display-ready (H, W, 3) pixels that
    # bypass the colormap/contrast pipeline in JS; False panels are grayscale.
    is_rgb = traitlets.List(traitlets.Bool(), default_value=[]).tag(sync=True)
    starred = traitlets.List(traitlets.Int()).tag(sync=True)
    hidden_panels = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    hidden_page_slots = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    panel_order = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    show_panel_titles = traitlets.Bool(True).tag(sync=True)
    panel_title_font_size = traitlets.Int(11).tag(sync=True)
    gallery_gap_px = traitlets.Int(0).tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    show_title = traitlets.Bool(True).tag(sync=True)
    cmap = traitlets.Unicode("inferno").tag(sync=True)
    ncols = traitlets.Int(3).tag(sync=True)

    # =========================================================================
    # Display Options
    # =========================================================================
    log_scale = traitlets.Bool(False).tag(sync=True)
    auto_contrast = traitlets.Bool(False).tag(sync=True)
    vmin = traitlets.Float(None, allow_none=True).tag(sync=True)
    vmax = traitlets.Float(None, allow_none=True).tag(sync=True)
    vmins = traitlets.List(trait=traitlets.Float(allow_none=True), allow_none=True, default_value=None).tag(sync=True)
    vmaxs = traitlets.List(trait=traitlets.Float(allow_none=True), allow_none=True, default_value=None).tag(sync=True)

    # =========================================================================
    # Scale Bar
    # =========================================================================
    pixel_size = traitlets.Float(0.0).tag(sync=True)
    pixel_sizes = traitlets.List(trait=traitlets.Float(), default_value=[]).tag(sync=True)
    pixel_unit = traitlets.Unicode("pixels").tag(sync=True)
    scale_bar_visible = traitlets.Bool(True).tag(sync=True)
    size = traitlets.Int(0).tag(sync=True)  # Canvas rendering size in CSS pixels; 0 = frontend default
    smooth = traitlets.Bool(False).tag(sync=True)
    initial_zoom = traitlets.Float(1.0).tag(sync=True)
    zoom_row = traitlets.Float(None, allow_none=True).tag(sync=True)
    zoom_col = traitlets.Float(None, allow_none=True).tag(sync=True)
    # Live viewport (row0, row1, col0, col1) in image pixel coordinates.
    # Python -> JS at construction (the view_box= sugar below derives zoom +
    # zoom_row/zoom_col from it); JS -> Python on every pan/zoom (debounced
    # ~100 ms), so current_view always reflects what is on screen. The box is
    # axis-aligned: two corners (row0, col0)/(row1, col1) define all four.
    view_box = traitlets.List(trait=traitlets.Float(), default_value=[]).tag(sync=True)
    # Reversible display-window ops (single panel only). view_crop holds the
    # committed crop (row0, row1, col0, col1) in FULL-RESOLUTION image pixels
    # (empty = no crop); pad_ratio adds a constant border (value = frame
    # minimum) around the displayed frame. Both are display-only view
    # transforms applied while packing frame_bytes: the stored data is never
    # modified and reset_view_ops() restores the full frame bit-identically.
    view_crop = traitlets.List(trait=traitlets.Int(), default_value=[]).tag(sync=True)
    pad_ratio = traitlets.Float(0.0).tag(sync=True)
    # One-line announcement of an active crop/pad. Same house rule as
    # denoise_banner: an active view reduction is never silent.
    view_banner = traitlets.Unicode("").tag(sync=True)
    # (row, col) native-pixel offset of the packed frame's origin relative to
    # the full image. JS adds it to cursor readouts so displayed coordinates
    # stay full-image while a crop or pad is active.
    _view_crop_offset = traitlets.List(trait=traitlets.Int(), default_value=[0, 0]).tag(sync=True)
    link_zoom = traitlets.Bool(False).tag(sync=True)
    link_pan = traitlets.Bool(False).tag(sync=True)
    link_contrast = traitlets.Bool(True).tag(sync=True)
    diff_mode = traitlets.Bool(False).tag(sync=True)
    diff_reference = traitlets.Int(0).tag(sync=True)

    # =========================================================================
    # UI Visibility
    # =========================================================================
    show_controls = traitlets.Bool(True).tag(sync=True)
    controls_collapsed = traitlets.Bool(False).tag(sync=True)
    show_stats = traitlets.Bool(True).tag(sync=True)
    debug = traitlets.Bool(False).tag(sync=True)
    stats_mean = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_min = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_max = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_std = traitlets.List(traitlets.Float()).tag(sync=True)

    # =========================================================================
    # Analysis Panels (FFT + Histogram shown together)
    # =========================================================================
    show_fft = traitlets.Bool(False).tag(sync=True)
    fft_window = traitlets.Bool(True).tag(sync=True)
    fft_metrics = traitlets.Bool(True).tag(sync=True)

    # =========================================================================
    # Selected Image (for single-image analysis display)
    # =========================================================================
    selected_idx = traitlets.Int(0).tag(sync=True)

    # =========================================================================
    # ROI Selection
    # =========================================================================
    roi_active = traitlets.Bool(False).tag(sync=True)
    roi_list = traitlets.List([]).tag(sync=True)
    roi_selected_idx = traitlets.Int(-1).tag(sync=True)

    # =========================================================================
    # Line Profile
    # =========================================================================
    profile_line = traitlets.List(traitlets.Dict()).tag(sync=True)

    # =========================================================================
    # Per-Image Rotation
    # =========================================================================
    image_rotations = traitlets.List(traitlets.Int(), []).tag(sync=True)

    @classmethod
    def from_gif(
        cls,
        path: str | pathlib.Path,
        *,
        ncols: int | None = None,
        labels: list[str | None] | bool | None = None,
        title: str | None = None,
        **kwargs,
    ) -> Self:
        """Open a GIF as a static frame gallery.

        Animated GIFs become an ``(N, H, W)`` stack displayed as a contact sheet.
        Use ``ncols`` to choose an intentional layout, for example ``ncols=2``
        for a 2x2 denoising comparison or ``ncols=1`` for a vertical strip.
        """
        from quantem.widget.io import read_gif  # noqa: PLC0415

        ds = read_gif(path)
        frame_count = int(ds.array.shape[0])
        if ncols is None:
            ncols = max(1, math.ceil(math.sqrt(frame_count)))
        if labels is True:
            resolved_labels = [f"Frame {i + 1}" for i in range(frame_count)]
        elif labels:
            resolved_labels = list(labels)
        else:
            resolved_labels = None
        return cls(
            ds,
            labels=resolved_labels,
            title=ds.name if title is None else title,
            ncols=ncols,
            **kwargs,
        )

    @classmethod
    def from_folder(
        cls,
        path: str | pathlib.Path,
        *,
        pattern: str = "*",
        recursive: bool = False,
        watch: bool = True,
        watch_interval: float = 1.0,
        page_size: int | None = _DEFAULT_FOLDER_PAGE_SIZE,
        **kwargs,
    ) -> Self:
        """Display readable folder images as a paged full-resolution gallery.

        Files are ordered naturally (``image_2`` before ``image_10``) and read
        through :func:`quantem.widget.io.read_image`, including EMD, TIFF, PNG,
        NPY, and DM calibration paths. The same widget is updated when stable
        files are added. Unreadable files remain retryable. At most
        ``page_size`` panels are visible at once; pass ``None`` to keep one
        unpaged gallery.

        Parameters
        ----------
        path : str or pathlib.Path
            Folder containing independent 2D image files.
        pattern : str, default "*"
            Glob selecting files within ``path``.
        recursive : bool, default False
            Search matching files below subdirectories as well.
        watch : bool, default True
            Start background polling immediately.
        watch_interval : float, default 1.0
            Seconds between background polls.
        page_size : int or None, default 20
            Maximum visible panels per page. Paging appears only after the
            ready image count exceeds this value. Pass ``None`` to disable
            automatic folder paging.
        **kwargs
            Normal :class:`Show2D` options. File-derived labels and data are
            managed by the folder source.
        """
        if "labels" in kwargs:
            raise TypeError(
                "Show2D.from_folder() derives labels from file paths so new files "
                "remain identifiable; remove labels= or construct Show2D directly."
            )
        if "page_labels" in kwargs or "page_kind" in kwargs:
            raise TypeError(
                "Show2D.from_folder() manages page labels and folder-page "
                "semantics as files arrive; remove page_labels=/page_kind= and "
                "use page_size= to configure the gallery"
            )
        resolved_page_size = _validate_folder_page_size(page_size)
        source = WatchedImageFolder(
            path,
            pattern=pattern,
            recursive=recursive,
            interval=watch_interval,
            mode="panels",
        )
        arrays, records = source.read_initial(
            allow_empty=watch,
            require_unchanged_followup=watch,
        )
        explicit_calibration = any(
            key in kwargs
            for key in ("sampling", "units", "pixel_size", "pixel_sizes", "pixel_unit")
        )
        kwargs.setdefault("title", source.folder.name)
        kwargs.setdefault("verbose", False)
        if arrays:
            initial_data = arrays
            initial_labels = [source.label(record.path) for record in records]
        else:
            display_bin = kwargs.get("display_bin", "auto")
            placeholder_side = (
                max(1, int(display_bin))
                if isinstance(display_bin, int) and not isinstance(display_bin, bool)
                else 1
            )
            initial_data = [np.zeros((placeholder_side, placeholder_side), dtype=np.float32)]
            initial_labels = [""]
        widget = cls(initial_data, labels=initial_labels, **kwargs)
        widget._folder_display_bin_request = kwargs.get("display_bin", "auto")
        widget._folder_page_size = resolved_page_size
        with widget.hold_sync():
            widget.page_kind = "items"
            if not arrays:
                widget._set_folder_waiting_empty_state()
            widget._sync_folder_pages(anchor_panel=0)
        source.attach(widget, explicit_calibration=explicit_calibration)
        if watch:
            widget.watch_folder(interval=watch_interval)
        return widget

    @property
    def folder_page_size(self) -> int | None:
        """Maximum visible panels per page for this folder-backed gallery."""
        self._require_folder_source()
        value = getattr(self, "_folder_page_size", None)
        return None if value is None else int(value)

    def set_folder_page_size(self, page_size: int | None) -> Self:
        """Change automatic folder pagination without rebuilding the widget.

        Parameters
        ----------
        page_size : int or None
            Maximum visible panels per page, or ``None`` for one unpaged
            gallery.

        Returns
        -------
        Show2D
            This widget, for method chaining.
        """
        self._require_folder_source()
        self._folder_page_size = _validate_folder_page_size(page_size)
        self._sync_folder_pages(anchor_panel=int(self.selected_idx))
        return self

    def __init__(
        self,
        data: np.ndarray | list[np.ndarray],
        labels: list[str | None] = None,
        page_labels: Sequence[str | None] | None = None,
        title: str = "",
        ui_mode: UiMode = "interactive",
        show_title: bool | None = None,
        cmap: str | Colormap = Colormap.INFERNO,
        sampling: float | tuple[float, float] | list[float] | None = None,
        units: str | list[str] | None = None,
        scale_bar_visible: bool | None = None,
        show_scale_bar: bool | None = None,
        show_fft: bool = False,
        fft_window: bool = True,
        fft_metrics: bool = True,
        show_controls: bool | None = None,
        controls_collapsed: bool | None = None,
        show_stats: bool | None = None,
        debug: bool = False,
        verbose: bool = True,
        log_scale: bool = False,
        auto_contrast: bool = False,
        offline: bool = False,
        vmin: float | list | None = None,
        vmax: float | list | None = None,
        ncols: int = 3,
        panel_frame_indices: Sequence[int] | None = None,
        panel_playback_fps: float = 10.0,
        size: int = 0,
        panel_width_px: int = 0,
        smooth: bool = False,
        zoom: float = 1.0,
        zoom_row: float | None = None,
        zoom_col: float | None = None,
        center: tuple | list | None = None,
        link_zoom: bool | None = None,
        link_pan: bool | None = None,
        link_contrast: bool = True,
        diff_mode: bool = False,
        overlay: bool | str = False,
        view_box: tuple | list | None = None,
        pad_ratio: float = 0.0,
        display_bin: int | str = "auto",
        hidden_panels: Sequence[int | str] | int | str | None = None,
        starred: Sequence[int | str] | int | str | None = None,
        panel_order: Sequence[int | str] | None = None,
        show_panel_titles: bool | None = None,
        panel_title_font_size: int = 11,
        gallery_gap_px: int = 0,
        state=None,
        save_state: bool = False,
        notebook_preview_format: str | None = "jpeg",
        notebook_preview_quality: int = 88,
        notebook_preview_max_px: int = 512,
        denoise: str | Sequence[str] = _UNSET,
        denoise_sigma: float | Sequence[float] = _UNSET,
        denoise_bin: int | Sequence[int] = _UNSET,
        denoise_scope: str = _UNSET,
        show_denoise: bool = False,
        display_filter: str | Sequence[str] | None = None,
        display_sigma: float | Sequence[float] | None = None,
        spatial_bin: int | Sequence[int] | None = None,
        filter_per_panel: bool | None = None,
        underlay: bool | str = False,
        underlay_alpha: float = 0.95,
        underlay_haadf_gain: float = 0.35,
        underlay_mode: str = "haadf",
        stretch_percentiles: Sequence[float] = (4.0, 99.0),
        display_gamma: float = 0.75,
        dual_gain: Sequence[float] = (1.0, 1.0),
        **kwargs,
    ):
        import time as _time
        _t0 = _time.perf_counter()
        # Reject typos and stale kwargs (e.g. image_width_px, pixel_size_angstrom).
        # anywidget/traitlets silently ignores unknown keys, which hid the
        # pixel_size_angstrom bug in show2d_all_features.ipynb for months.
        _reject_unknown_kwargs(type(self), kwargs)
        # Deprecated aliases from the display_filter-era API (one rc of
        # compatibility). Key off "was this kwarg supplied" via the _UNSET
        # sentinel so an explicit new kwarg wins even when its value equals the
        # trait default; a deprecated alias only fills an unset new kwarg. Each
        # supplied alias warns regardless of who wins the value.
        denoise_supplied = denoise is not _UNSET
        denoise_sigma_supplied = denoise_sigma is not _UNSET
        denoise_bin_supplied = denoise_bin is not _UNSET
        denoise_scope_supplied = denoise_scope is not _UNSET
        if not denoise_supplied:
            denoise = "none"
        if not denoise_sigma_supplied:
            denoise_sigma = 4.0
        if not denoise_bin_supplied:
            denoise_bin = 1
        def _is_panel_seq(v):
            return isinstance(v, (list, tuple)) and not isinstance(v, str)
        _per_panel_denoise = (
            (denoise_supplied and _is_panel_seq(denoise))
            or (denoise_sigma_supplied and _is_panel_seq(denoise_sigma))
            or (denoise_bin_supplied and _is_panel_seq(denoise_bin))
        )
        if not denoise_scope_supplied:
            # Per-panel denoise specified at construction implies per-panel
            # editing: the single denoise control edits the SELECTED panel
            # rather than broadcasting to all (which would clobber the author's
            # per-panel setup on the first interactive edit). Explicit
            # denoise_scope / filter_per_panel below still win.
            denoise_scope = "panel" if _per_panel_denoise else "all"
        if display_filter is not None:
            warnings.warn(
                "display_filter is deprecated; use denoise= instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if not denoise_supplied:
                denoise = display_filter
        if display_sigma is not None:
            warnings.warn(
                "display_sigma is deprecated; use denoise_sigma= instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if not denoise_sigma_supplied:
                denoise_sigma = display_sigma
        if spatial_bin is not None:
            warnings.warn(
                "spatial_bin is deprecated; use denoise_bin= instead (the SNR "
                "knob for sparse maps, not the display_bin performance "
                "downsample).",
                DeprecationWarning,
                stacklevel=2,
            )
            if not denoise_bin_supplied:
                denoise_bin = spatial_bin
        if filter_per_panel is not None:
            warnings.warn(
                "filter_per_panel is deprecated; use denoise_scope='all' or "
                "denoise_scope='panel' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if not denoise_scope_supplied:
                denoise_scope = "all" if filter_per_panel else "panel"
        data, labels, n_pages, panels_per_page, resolved_page_labels, resolved_page_starred = _normalise_show2d_pages(
            data,
            labels=labels,
            page_labels=page_labels,
        )
        panel_width_px = int(panel_width_px)
        if panel_width_px < 0:
            raise ValueError(f"panel_width_px must be >= 0, got {panel_width_px}")
        if panel_width_px > 0:
            size = panel_width_px
        ncols = int(ncols)
        if ncols < 1:
            raise ValueError(f"ncols must be >= 1, got {ncols}")
        gallery_gap_px = int(gallery_gap_px)
        if gallery_gap_px < 0:
            raise ValueError(f"gallery_gap_px must be >= 0, got {gallery_gap_px}")
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
                "show_stats": True,
                "show_panel_titles": True,
                "show_scale_bar": True,
            },
            overrides={
                "show_title": show_title,
                "show_controls": show_controls,
                "controls_collapsed": controls_collapsed,
                "show_stats": show_stats,
                "show_panel_titles": show_panel_titles,
                "show_scale_bar": scale_bar_visible if scale_bar_visible is not None else show_scale_bar,
            },
        )
        show_title = bool(ui["show_title"])
        show_controls = bool(ui["show_controls"])
        controls_collapsed = bool(ui["controls_collapsed"])
        show_stats = bool(ui["show_stats"])
        show_panel_titles = bool(ui["show_panel_titles"])
        scale_bar_visible = bool(ui["show_scale_bar"])
        # save_state controls whether the heavy pixel buffers are persisted into
        # the notebook's metadata.widgets on save. Default False: a plain display
        # embeds only light traits + a static image preview, so a 5-panel 4k
        # gallery does not bake ~1 GB into the .ipynb. Set True to persist full
        # interactive state so a reopened notebook restores the widget without
        # a kernel.
        self._save_state = bool(save_state)
        self._configure_static_fallback(
            notebook_preview_format=notebook_preview_format,
            notebook_preview_quality=notebook_preview_quality,
            notebook_preview_max_px=notebook_preview_max_px,
        )
        super().__init__(**kwargs)
        self._static_fallback_mime = self._static_fallback_mime_type()
        # hold_sync() batches ALL traitlet assignments into a single comm message
        # sent when the context manager exits.  Without this, each self.x = y
        # fires a separate round-trip over the ZMQ/websocket channel, which
        # can add 20+ seconds for a 30-image gallery in VS Code Jupyter.
        if center is not None:
            # center=(row, col) sugar: the friendliest way to say where to look
            zoom_row, zoom_col = float(center[0]), float(center[1])
        with self.hold_sync():
            self._init_sync(
                data=data, labels=labels, title=title, cmap=cmap,
                n_pages=n_pages, panels_per_page=panels_per_page,
                page_labels=resolved_page_labels, page_starred=resolved_page_starred,
                show_title=show_title,
                sampling=sampling, units=units, scale_bar_visible=scale_bar_visible,
                show_fft=show_fft, fft_window=fft_window, fft_metrics=fft_metrics,
                show_controls=show_controls, controls_collapsed=controls_collapsed,
                show_stats=show_stats, debug=debug,
                log_scale=log_scale, auto_contrast=auto_contrast, offline=offline,
                vmin=vmin, vmax=vmax,
                ncols=ncols, panel_frame_indices=panel_frame_indices,
                panel_playback_fps=panel_playback_fps,
                size=size, smooth=smooth, zoom=zoom,
                zoom_row=zoom_row, zoom_col=zoom_col,
                link_zoom=link_zoom, link_pan=link_pan, link_contrast=link_contrast,
                diff_mode=diff_mode, overlay=overlay, view_box=view_box,
                pad_ratio=pad_ratio,
                display_bin=display_bin, hidden_panels=hidden_panels, starred=starred,
                panel_order=panel_order,
                show_panel_titles=show_panel_titles, panel_title_font_size=panel_title_font_size,
                gallery_gap_px=gallery_gap_px,
                verbose=verbose, state=state, _t0=_t0,
                denoise=denoise, denoise_sigma=denoise_sigma,
                denoise_bin=denoise_bin, denoise_scope=denoise_scope,
                denoise_scope_explicit=denoise_scope_supplied,
                show_denoise=show_denoise,
                underlay=underlay, underlay_alpha=underlay_alpha,
                underlay_haadf_gain=underlay_haadf_gain,
                underlay_mode=underlay_mode, stretch_percentiles=stretch_percentiles,
                display_gamma=display_gamma, dual_gain=dual_gain)

    def _init_sync(self, *, data, labels, title, cmap, n_pages, panels_per_page,
                   page_labels, page_starred, show_title, sampling, units,
                   scale_bar_visible, show_fft, fft_window, fft_metrics,
                   show_controls, controls_collapsed, show_stats, debug, log_scale, auto_contrast, offline,
                   vmin, vmax,
                   ncols, panel_frame_indices, panel_playback_fps, size, smooth, zoom, zoom_row, zoom_col,
                   link_zoom, link_pan, link_contrast, diff_mode, overlay, view_box,
                   pad_ratio, display_bin, hidden_panels, starred, panel_order, show_panel_titles,
                   panel_title_font_size, gallery_gap_px, verbose, state, _t0,
                   denoise="none", denoise_sigma=4.0, denoise_bin=1,
                   denoise_scope="all", denoise_scope_explicit=False, show_denoise=False,
                   underlay=False, underlay_alpha=0.95,
                   underlay_haadf_gain=0.35, underlay_mode="haadf",
                   stretch_percentiles=(4.0, 99.0), display_gamma=0.75,
                   dual_gain=(1.0, 1.0)):
        import time as _time
        self._verbose = verbose
        self.widget_version = resolve_widget_version()
        self._display_data = None  # initialized after data setup
        self._display_bin = 1
        self.prepared_view = None
        self.prepared_view_widget = None
        self.n_pages = int(max(1, n_pages))
        self.panels_per_page = int(max(0, panels_per_page))
        self.page_idx = 0
        self.page_labels = list(page_labels or [])
        self.page_starred = list(page_starred or [])

        # First-class support for quantem Dataset2d / Dataset3d:
        # auto-extract array + sampling + units from the dataset object.
        core_image_dataset_types = _core_image_dataset_types()
        is_core_image_dataset = (
            bool(core_image_dataset_types)
            and isinstance(data, core_image_dataset_types)
        )
        has_image_dataset_shape = (
            hasattr(data, "array") and hasattr(data, "name") and hasattr(data, "sampling")
        )
        if is_core_image_dataset or has_image_dataset_shape:
            if not title and data.name:
                title = data.name
            if sampling is None:
                sampling = tuple(float(s) for s in data.sampling[-2:])
            if units is None and hasattr(data, "units"):
                units = list(data.units[-2:])
            data = data.array
        # Same auto-extract for list/tuple of Dataset2d (gallery from per-file load).
        elif isinstance(data, (list, tuple)) and len(data) > 0:
            first = data[0]
            first_is_core_image_dataset = (
                bool(core_image_dataset_types)
                and isinstance(first, core_image_dataset_types)
            )
            first_has_image_dataset_shape = (
                hasattr(first, "array") and hasattr(first, "sampling")
            )
            if first_is_core_image_dataset or first_has_image_dataset_shape:
                if sampling is None:
                    sampling = tuple(float(s) for s in first.sampling[-2:])
                if units is None and hasattr(first, "units"):
                    units = list(first.units[-2:])
                data = [d.array for d in data]

        # Convert NumPy / PyTorch / list inputs to a NumPy array.
        # RGB detection rule: per-ITEM in a list input, an item with ndim == 3
        # and shape[-1] in (3, 4) is an RGB(A) image. A single non-list ndim==3
        # input is RGB only when shape[-1] in (3, 4) AND shape[0] > 4;
        # otherwise it keeps the historical (N, H, W) stack semantics.
        rgb_flags: list[bool] = []
        rgb_frames: list[np.ndarray | None] = []
        panel_stacks: list[np.ndarray] | None = None
        resolved_panel_frame_indices: list[int] | None = None
        if isinstance(data, (list, tuple)):
            images = [to_numpy(d) for d in data]
            rgb_flags = [_is_rgb_item(img) for img in images]
        else:
            arr = to_numpy(data)
            if arr.ndim == 3 and arr.shape[-1] in (3, 4) and arr.shape[0] > 4:
                images, rgb_flags = [arr], [True]
            else:
                images, rgb_flags = None, []
                data = arr
        if images is not None and any(rgb_flags):
            stack_panels = [
                idx for idx, (img, is_rgb) in enumerate(zip(images, rgb_flags))
                if not is_rgb and img.ndim == 3
            ]
            if stack_panels:
                raise NotImplementedError(
                    "Show2D does not yet mix RGB panels with local grayscale frame "
                    f"stacks (stack panel indices: {stack_panels}). Convert the RGB "
                    "panel to grayscale or use a separate Show2D."
                )
            # Mixed gallery: normalize RGB items to display-ready [0, 1] and
            # reduce them to Rec. 709 luminance for the grayscale machinery
            # (stats, histogram, FFT, profile all read the luminance plane).
            normalized = [_normalize_rgb(img) if flag else np.asarray(img, dtype=np.float32)
                          for img, flag in zip(images, rgb_flags)]
            shapes = [img.shape[:2] for img in normalized]
            if len(set(shapes)) > 1:
                max_h = max(s[0] for s in shapes)
                max_w = max(s[1] for s in shapes)
                normalized = [
                    np.stack([_resize_image(ch, max_h, max_w) for ch in img.transpose(2, 0, 1)], axis=-1)
                    if flag else _resize_image(img, max_h, max_w)
                    for img, flag in zip(normalized, rgb_flags)
                ]
            rgb_frames = [img if flag else None for img, flag in zip(normalized, rgb_flags)]
            data = np.stack([img @ _RGB_LUMA if flag else img
                             for img, flag in zip(normalized, rgb_flags)])
        elif images is not None:
            panel_stacks, resolved_panel_frame_indices, data = _normalize_grayscale_panel_items(
                images,
                panel_frame_indices,
            )

        # Ensure 3D shape (N, H, W)
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        # Avoid redundant copy: np.asarray is a no-op when already float32 + contiguous
        if data.dtype == np.float32:
            self._data = np.array(data, dtype=np.float32, copy=True)
        else:
            self._data = np.asarray(data, dtype=np.float32)
        self._rgb_frames = rgb_frames or [None] * int(data.shape[0])
        self.is_rgb = [f is not None for f in self._rgb_frames]
        if panel_stacks is None:
            panel_stacks = [self._data[i:i + 1] for i in range(int(self._data.shape[0]))]
            if panel_frame_indices is None:
                resolved_panel_frame_indices = [0] * len(panel_stacks)
            else:
                if len(panel_frame_indices) != len(panel_stacks):
                    raise ValueError(
                        "panel_frame_indices length "
                        f"({len(panel_frame_indices)}) must match panel count ({len(panel_stacks)})"
                    )
                resolved_panel_frame_indices = []
                for panel, raw_index in enumerate(panel_frame_indices):
                    index = int(raw_index)
                    if index == -1:
                        index = 0
                    if index != 0:
                        raise ValueError(
                            f"panel_frame_indices[{panel}]={raw_index} is outside the valid range [0, 1)"
                        )
                    resolved_panel_frame_indices.append(0)
        # overlay sugar (mirrors diff_mode): compose an alignment overlay panel
        # from the two grayscale inputs so users never hand-build (H, W, 3).
        overlay_label = None
        if overlay:
            if any(stack.shape[0] > 1 for stack in panel_stacks):
                raise NotImplementedError(
                    "overlay= is not supported with local frame-stack panels because "
                    "the composed overlay would not follow frame changes"
                )
            if overlay is not True and str(overlay) not in ("green-magenta", "rgb"):
                raise ValueError(f"overlay must be True, 'green-magenta', or 'rgb', got {overlay!r}")
            if self._data.shape[0] != 2 or any(self.is_rgb):
                raise ValueError(
                    f"overlay requires exactly 2 grayscale images, got "
                    f"{self._data.shape[0]} image(s)"
                    + (" including RGB panels" if any(self.is_rgb) else "")
                )
            mode = "rgb" if str(overlay) == "rgb" else "green-magenta"
            composed = _compose_overlay_pair(self._data[0], self._data[1], mode)
            # gallery becomes [a, b, overlay]; the overlay's luminance plane
            # joins _data so stats/histogram/FFT machinery sees three panels
            self._data = np.concatenate([self._data, (composed @ _RGB_LUMA)[None]], axis=0)
            self._rgb_frames = [None, None, composed]
            self.is_rgb = [False, False, True]
            panel_stacks = [self._data[i:i + 1] for i in range(int(self._data.shape[0]))]
            resolved_panel_frame_indices = [0] * len(panel_stacks)
            overlay_label = f"overlay ({mode})"
            data = self._data  # n_images/height/width below read `data`
        # underlay sugar: chemistry-on-structure. Blend the element map onto
        # the HAADF lattice as a third RGB panel so bright columns render in
        # the map color, never as white HAADF cores.
        if underlay:
            if overlay:
                raise ValueError("Use either overlay or underlay, not both")
            if str(underlay).lower() not in ("true", "1", "haadf"):
                raise ValueError(f"underlay must be True or 'haadf', got {underlay!r}")
            mode = str(underlay_mode).strip().lower()
            if mode not in ("haadf", "dual"):
                raise ValueError(
                    f"underlay_mode must be 'haadf' or 'dual', got {underlay_mode!r}"
                )
            # Both modes need exactly two single-frame grayscale inputs: haadf
            # mode reads them as (haadf, map), dual mode as (map A, map B).
            inputs_desc = "(map A, map B)" if mode == "dual" else "(haadf, map)"
            if self._data.shape[0] != 2 or any(self.is_rgb):
                raise ValueError(
                    f"underlay requires exactly 2 grayscale images {inputs_desc}; got "
                    f"{self._data.shape[0]} image(s)"
                    + (" including RGB panels" if any(self.is_rgb) else "")
                )
            if any(stack.shape[0] > 1 for stack in panel_stacks):
                raise NotImplementedError(
                    "underlay does not support per-panel frame stacks; pass single frames"
                )
            self.underlay = True
            self.underlay_mode = mode
            self.underlay_alpha = float(underlay_alpha)
            self.underlay_haadf_gain = float(underlay_haadf_gain)
            self.stretch_percentiles = [float(stretch_percentiles[0]), float(stretch_percentiles[1])]
            self.display_gamma = float(display_gamma)
            self.dual_gain = [float(dual_gain[0]), float(dual_gain[1])]
            self.cmap = str(cmap)
            self._underlay_haadf_idx = 0
            self._underlay_map_idx = 1
            # Raw blend now; the display-filtered blend is recomputed in
            # _update_all_frames once the filter knobs are set below.
            composed = self._compute_underlay_blend()
            self._data = np.concatenate([self._data, (composed @ _RGB_LUMA)[None]], axis=0)
            self._rgb_frames = [None, None, composed]
            self.is_rgb = [False, False, True]
            panel_stacks = [self._data[i:i + 1] for i in range(int(self._data.shape[0]))]
            resolved_panel_frame_indices = [0] * len(panel_stacks)
            overlay_label = "dual composite" if mode == "dual" else "map on HAADF"
            data = self._data
        if offline and any(self.is_rgb):
            raise NotImplementedError(
                "offline=True is not supported for RGB panels; RGB frames are "
                "sent as full float32 and bypass the uint8 quantization path."
            )
        # Store originals for rotation reset: views into _data (no copy).
        # Only materialized as independent copies when a rotation is applied.
        self._data_original = [self._data[i] for i in range(self._data.shape[0])]
        self._originals_are_views = True
        self.n_images = int(data.shape[0])
        self._panel_stacks = panel_stacks
        self._panel_stacks_original = [stack for stack in panel_stacks]
        self._panel_stack_originals_are_views = True
        self.panel_frame_counts = [int(stack.shape[0]) for stack in panel_stacks]
        self.panel_frame_indices = list(resolved_panel_frame_indices or [0] * self.n_images)
        self.panel_stack_offsets = [-1] * self.n_images
        self.height = int(data.shape[1])
        self.width = int(data.shape[2])
        self.image_rotations = [0] * self.n_images
        if self.n_pages > 1:
            if self.panels_per_page <= 0:
                raise ValueError("panels_per_page must be > 0 when n_pages > 1")
            if self.n_images != self.n_pages * self.panels_per_page:
                raise ValueError(
                    f"paged Show2D expects n_images == n_pages * panels_per_page, "
                    f"got {self.n_images} != {self.n_pages} * {self.panels_per_page}"
                )
            if not self.page_labels:
                self.page_labels = [f"Page {i + 1}" for i in range(self.n_pages)]
            if len(self.page_labels) != self.n_pages:
                raise ValueError(
                    f"page_labels length ({len(self.page_labels)}) must equal n_pages ({self.n_pages})"
                )
            if not self.page_starred:
                self.page_starred = [0] * self.n_pages

        # Labels
        if labels is None:
            self.labels = [f"Image {i+1}" for i in range(self.n_images)]
            if overlay_label:
                self.labels = self.labels[:-1] + [overlay_label]
        else:
            resolved_labels = list(labels)
            # user labeled the two inputs; the composed overlay names itself
            if overlay_label and len(resolved_labels) == self.n_images - 1:
                resolved_labels.append(overlay_label)
            self.labels = resolved_labels
        self.starred = [0] * self.n_images
        self.hidden_panels = []
        self.hidden_page_slots = []
        self.show_panel_titles = bool(show_panel_titles)
        self.panel_title_font_size = int(panel_title_font_size)
        self.gallery_gap_px = int(gallery_gap_px)
        if starred is not None:
            self.set_starred_panels(starred)
        if hidden_panels is not None:
            self.set_hidden_panels(hidden_panels)
        if panel_order is not None:
            self.set_panel_order(panel_order)

        # Options
        self.title = title
        self.show_title = bool(show_title)
        self.cmap = cmap
        # Resolve sampling + units to scalar pixel_size + pixel_unit (column axis).
        # Scalar shorthand: sampling=0.5 → (0.5, 0.5). units="nm" → ["nm", "nm"].
        if sampling is None:
            pass  # keep the trait: pixel_size= may already be set directly via kwargs
        elif isinstance(sampling, (int, float)):
            self.pixel_size = float(sampling)
        else:
            self.pixel_size = float(sampling[-1])
        if units is None:
            pass  # keep the trait: pixel_unit= may already be set directly via kwargs
        elif isinstance(units, str):
            self.pixel_unit = units
        else:
            self.pixel_unit = str(units[-1])
        self.scale_bar_visible = scale_bar_visible
        self.pixel_sizes = []
        self.size = size
        self.smooth = smooth
        # view_box sugar: sets zoom + zoom_row/col to center on box.
        # Two forms: (r0, r1, c0, c1) explicit bounds, or the friendlier
        # (row0, col0, size) = top-left corner + square size.
        if view_box is not None:
            vb = [float(v) for v in view_box]
            if len(vb) == 3:
                vb = [vb[0], vb[0] + vb[2], vb[1], vb[1] + vb[2]]
            r0, r1, c0, c1 = vb
            box_h = max(1.0, r1 - r0)
            box_w = max(1.0, c1 - c0)
            zoom = float(min(self.height / box_h, self.width / box_w))
            zoom_row = (r0 + r1) / 2
            zoom_col = (c0 + c1) / 2
            self.view_box = [r0, r1, c0, c1]
        self.initial_zoom = zoom
        self.zoom_row = zoom_row
        self.zoom_col = zoom_col
        # Auto-link zoom + pan in gallery (n_images >= 2) so dragging one panel
        # follows the other — typical compare/diff workflow. Single image: no-op.
        self.link_zoom = (self.n_images >= 2) if link_zoom is None else link_zoom
        self.link_pan = (self.n_images >= 2) if link_pan is None else link_pan
        self.link_contrast = link_contrast
        self.diff_mode = diff_mode if self.n_images >= 2 else False
        if show_fft and self.height * self.width > 2048 * 2048:
            warnings.warn(
                f"FFT on {self.height}×{self.width} image ({self.height * self.width / 1e6:.1f}M pixels) "
                f"may be slow. Consider using ROI FFT for a sub-region.",
                stacklevel=2,
            )
        self.show_fft = show_fft
        self.fft_window = fft_window
        self.fft_metrics = bool(fft_metrics)
        self.show_controls = show_controls
        self.controls_collapsed = bool(controls_collapsed)
        self.show_stats = show_stats
        self.debug = bool(debug)
        self.log_scale = log_scale
        self.auto_contrast = auto_contrast
        self.offline = offline
        # Standalone HTML export packs panels through the grayscale stack
        # machinery, which cannot carry (H, W, 3) pixels: hide the export menu
        # and reject programmatic export instead of corrupting RGB panels.
        if any(self.is_rgb):
            self.export_enabled = False
        # Accept scalar OR list for vmin/vmax. List → per-image (vmins/vmaxs).
        if isinstance(vmin, (list, tuple)) or isinstance(vmax, (list, tuple)):
            n = self.n_images
            def _expand(v):
                if v is None:
                    return [None] * n
                if isinstance(v, (list, tuple)):
                    if len(v) != n:
                        raise ValueError(f"vmin/vmax list has length {len(v)} but n_images is {n}. Pass a list of length {n} or a scalar to apply uniformly.")
                    return [None if x is None else float(x) for x in v]
                return [float(v)] * n
            self.vmins = _expand(vmin)
            self.vmaxs = _expand(vmax)
            self.vmin = None
            self.vmax = None
        else:
            self.vmin = vmin
            self.vmax = vmax
        self.ncols = ncols
        self.panel_playback_fps = panel_playback_fps

        # Auto-bin for display: keep full-res in _data, send binned to JS.
        # GPU memory budget: ~2 GB for display buffers (128 MB per image at 4K).
        # At 4K: max ~16 full-res. Beyond that, auto-downsample.
        if display_bin == "auto":
            # Each 4K image needs ~192 MB GPU buffers (float32 + RGBA + read)
            # Tested: 12×4K (2.3 GB) works, 24×4K (4.6 GB) OOMs
            # Budget: 2.5 GB allows 12×4K full-res, bins above that
            gpu_budget_mb = self._GPU_DISPLAY_BUDGET_MB
            per_image_mb = (self.height * self.width * 4 * 3) / (1024 * 1024)  # 3 buffers
            total_mb = self.n_images * per_image_mb
            if total_mb > gpu_budget_mb:
                # Find minimum bin factor to fit
                for bf in [2, 4, 8]:
                    binned_mb = self.n_images * per_image_mb / (bf * bf)
                    if binned_mb <= gpu_budget_mb:
                        self._display_bin = bf
                        break
                else:
                    self._display_bin = 8
                # no surprise binning: announce the reduction and the way out
                print(f"Show2D display auto-binned {self._display_bin}x to fit the GPU display budget "
                      f"({total_mb:.0f} MB > {gpu_budget_mb} MB); pass display_bin=1 for native pixels")
            # Wire budget (maps-style first paint): a panel whose float32 bytes
            # exceed the per-panel budget would stall first paint for seconds on
            # a remote Jupyter channel, while the screen canvas can only display
            # a tiny fraction of those pixels at initial zoom. Send a preview
            # binned so the longest side is ~2x the canvas CSS size; zooming
            # streams full-res crops of the visible window via _detail_request,
            # so full fidelity is preserved without ever shipping the full frame.
            # Large galleries need a lower preview floor: 36x1024x1024 float32
            # is still 144 MB before comm/base64/browser work, while the
            # default 200 px panel can only display a fraction of those pixels.
            panel_bytes = self.height * self.width * 4
            if any(self.is_rgb):
                panel_bytes *= 3  # RGB panels carry 3 interleaved float planes
            if panel_bytes > self._WIRE_BUDGET_BYTES_PER_PANEL:
                preview_floor_px = 512.0 if self.n_images >= 16 else 1024.0
                preview_px = max(preview_floor_px, 2.0 * self._static_canvas_css_px())
                wire_bin = max(2, math.ceil(max(self.height, self.width) / preview_px))
                while panel_bytes / (wire_bin * wire_bin) > self._WIRE_BUDGET_BYTES_PER_PANEL:
                    wire_bin += 1
                self._display_bin = max(self._display_bin, wire_bin)
        elif isinstance(display_bin, int) and display_bin > 1:
            self._display_bin = display_bin

        if self._display_bin > 1:
            from quantem.widget.utils.array import bin2d
            orig_h, orig_w = self._data.shape[1], self._data.shape[2]
            self._display_data = bin2d(self._data, factor=self._display_bin, mode="mean")
            self.height = int(self._display_data.shape[1])
            self.width = int(self._display_data.shape[2])
            if self.pixel_size > 0:
                self.pixel_size = self.pixel_size * self._display_bin
            # User-facing view coordinates (center=, view_box=) are full-res
            # pixels, but JS interprets the traits in preview pixels once the
            # display is binned. Rescale so a zoom target lands on the same
            # physical feature at any bin factor.
            if self.zoom_row is not None:
                self.zoom_row = self.zoom_row / self._display_bin
            if self.zoom_col is not None:
                self.zoom_col = self.zoom_col / self._display_bin
            if self.view_box:
                self.view_box = [v / self._display_bin for v in self.view_box]
            self._display_bin_factor = self._display_bin
            # RGB panels bin per channel so the display copy matches the
            # luminance plane's resolution (JS derives offsets from one H×W).
            self._display_rgb = [
                None if f is None else
                bin2d(f.transpose(2, 0, 1), factor=self._display_bin, mode="mean").transpose(1, 2, 0)
                for f in self._rgb_frames
            ]
            self._display_panel_stacks = [
                bin2d(stack, factor=self._display_bin, mode="mean")
                for stack in self._panel_stacks
            ]
            if verbose:
                print(f"  Display bin {self._display_bin}×: {orig_h}×{orig_w} → {self.height}×{self.width} ({self._display_data.nbytes // 1024 // 1024} MB preview; full-res detail streams on zoom)")
        else:
            self._display_data = self._data
            self._display_bin_factor = 1
            self._display_rgb = self._rgb_frames
            self._display_panel_stacks = self._panel_stacks

        # Display-only filter knobs (view transforms; the stored raw data and
        # the stats row stay untouched). Scalars broadcast to every panel; a
        # sequence gives each panel its own filter/sigma/bin (e.g. a raw vs
        # filtered A/B gallery). Set before the first frame pack so a
        # constructor-selected filter shows from the first paint.
        self._display_filter_ready = False
        self._filter_knob_sync = False
        n_panels = int(self._data.shape[0])

        def per_panel(value, kind, cast):
            if isinstance(value, (list, tuple, np.ndarray)):
                values = [cast(v) for v in value]
                if len(values) != n_panels:
                    raise ValueError(
                        f"{kind} sequence length ({len(values)}) must equal the "
                        f"panel count ({n_panels})"
                    )
                return values, False
            return [cast(value)] * n_panels, True

        filters, filters_scalar = per_panel(denoise, "denoise", str)
        sigmas, sigmas_scalar = per_panel(denoise_sigma, "denoise_sigma", float)
        bins, bins_scalar = per_panel(denoise_bin, "denoise_bin", int)
        # Compound spellings (bin2, bin2_anscombe, bin4_anscombe) are aliases
        # for (mode, bin); the traits always hold the canonical trio.
        from quantem.widget.utils.display_filter import resolve_denoise_mode

        resolved = [resolve_denoise_mode(m, b) for m, b in zip(filters, bins)]
        filters = [m for m, _ in resolved]
        bins = [b for _, b in resolved]
        self.denoise_modes = filters
        self.denoise_sigmas = sigmas
        self.denoise_bins = bins
        # Scalar traits are the UI editor, mirroring the selected panel.
        self._filter_knob_sync = True
        self.denoise = self.denoise_modes[0]
        self.denoise_sigma = float(sigmas[0])
        self.denoise_bin = int(bins[0])
        self._filter_knob_sync = False
        # A sequence means independent per-panel knobs, so UI edits scope to
        # the selected panel; scalars keep the broadcast "all" scope. Passing a
        # per-panel sequence AND an explicit denoise_scope="all" is a
        # contradiction ("all" broadcasts one value, the sequence gives each
        # panel its own), so reject it instead of silently forcing "panel".
        scalar_knobs = filters_scalar and sigmas_scalar and bins_scalar
        if denoise_scope_explicit and denoise_scope == "all" and not scalar_knobs:
            raise ValueError(
                "denoise_scope='all' broadcasts one setting to every panel, but "
                "a per-panel denoise/denoise_sigma/denoise_bin sequence was also "
                "given. Pass scalar denoise knobs with denoise_scope='all', or "
                "drop denoise_scope and let the sequence select per-panel scope."
            )
        broadcast = denoise_scope != "panel" and scalar_knobs
        self.denoise_scope = "all" if broadcast else "panel"
        self._display_filter_ready = True
        self._refresh_display_filter_banner(announce=True)
        # The master denoise switch starts ON iff the widget was built with an
        # active denoise config; a clean widget starts OFF (the toggle enables it).
        self.denoise_enabled = self._has_denoise_config()
        # Denoise controls stay hidden on a clean widget; an active denoise
        # (or an explicit request) reveals them from the first paint.
        self.show_denoise = bool(show_denoise) or self._display_filter_active()

        # Reversible view ops: a crop is committed later via crop_to_view(),
        # but the pad kwarg can start active. Geometry (frame extent, cursor
        # offset, banner) is synced before the first frame pack; the observer
        # stays inert until _view_ops_ready so the constructor zoom survives.
        self._view_ops_ready = False
        self.pad_ratio = float(pad_ratio)
        self._refresh_view_ops(announce=True)
        self._view_ops_ready = True

        # Compute initial stats (from full-res data)
        self._compute_all_stats()

        # Send display data to JS (possibly binned)
        self._update_all_frames()

        self.selected_idx = 0

        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(
                    json.loads(pathlib.Path(state).read_text()),
                    require_envelope=True,
                )
            else:
                state = unwrap_state_payload(state)
            self.load_state_dict(state)

        # Stash wall-clock start on the instance; the observer below prints the
        # TRUE end-to-end time after JS signals first paint.  The Python-only
        # __init__ number is misleading for widget UX: a widget is not "done"
        # until the browser has painted its first frame.
        self._init_t0 = _t0
        self._init_py_elapsed_ms = (_time.perf_counter() - _t0) * 1000
        self.observe(self._on_first_render, names=["_js_rendered"])
        self.observe(self._on_export_request_change, names=["export_request"])
        self.observe(self._on_handoff_request_change, names=["handoff_request"])
        self.observe(self._on_detail_request_change, names=["_detail_request"])

    @traitlets.validate("starred")
    def _validate_starred(self, proposal: dict) -> list[int]:
        """Normalize per-image star flags."""
        val = list(proposal["value"])
        n_img = int(self.n_images)
        if not val:
            return [0] * n_img
        if len(val) != n_img:
            raise traitlets.TraitError(
                f"starred length ({len(val)}) must equal n_images ({n_img})"
            )
        return [1 if int(v) else 0 for v in val]

    @traitlets.validate("panel_frame_indices")
    def _validate_panel_frame_indices(self, proposal: dict) -> list[int]:
        """Validate one independent frame index for every source panel."""
        values = list(proposal["value"])
        counts = list(getattr(self, "panel_frame_counts", []))
        n_images = int(getattr(self, "n_images", 0))
        if not values and n_images > 0:
            return [0] * n_images
        if len(values) != n_images:
            raise traitlets.TraitError(
                f"panel_frame_indices length ({len(values)}) must equal n_images ({n_images})"
            )
        if len(counts) != n_images:
            counts = [1] * n_images
        normalized: list[int] = []
        for panel, (raw_index, count) in enumerate(zip(values, counts)):
            index = int(raw_index)
            count = max(1, int(count))
            if index < 0 or index >= count:
                raise traitlets.TraitError(
                    f"panel_frame_indices[{panel}]={index} is outside the valid range [0, {count})"
                )
            normalized.append(index)
        return normalized

    @traitlets.validate("panel_playback_fps")
    def _validate_panel_playback_fps(self, proposal: dict) -> float:
        """Reject invalid local-stack playback rates and cap browser work."""
        value = float(proposal["value"])
        if not math.isfinite(value):
            raise traitlets.TraitError(
                f"panel_playback_fps must be finite, got {value}"
            )
        if value <= 0:
            raise traitlets.TraitError(
                f"panel_playback_fps must be > 0, got {value}"
            )
        return min(value, _MAX_PANEL_PLAYBACK_FPS)

    @traitlets.observe("panel_frame_indices")
    def _on_panel_frame_indices_changed(self, change) -> None:
        """Keep Python analysis/detail state aligned with browser-local scrubbing."""
        if getattr(self, "_updating_panel_frames", False):
            return
        stacks = getattr(self, "_panel_stacks", None)
        if not stacks or len(stacks) != int(getattr(self, "n_images", 0)):
            return
        indices = list(change.get("new") or [])
        if len(indices) != len(stacks):
            return
        self._data = np.stack(
            [stack[index] for stack, index in zip(stacks, indices)],
            axis=0,
        ).astype(np.float32, copy=False)
        display_stacks = getattr(self, "_display_panel_stacks", None)
        if display_stacks and len(display_stacks) == len(stacks):
            self._display_data = np.stack(
                [stack[index] for stack, index in zip(display_stacks, indices)],
                axis=0,
            ).astype(np.float32, copy=False)
        elif int(getattr(self, "_display_bin", 1)) <= 1:
            self._display_data = self._data
        if hasattr(self, "stats_mean"):
            self._compute_all_stats()
        if hasattr(self, "_detail_meta"):
            self._detail_request = ""
            self._detail_meta = ""
            self._detail_bytes = b""

    @traitlets.validate("page_idx")
    def _validate_page_idx(self, proposal: dict) -> int:
        """Clamp the active page index to the available page range."""
        n_pages = max(1, int(getattr(self, "n_pages", 1)))
        return int(max(0, min(int(proposal["value"]), n_pages - 1)))

    @traitlets.validate("page_starred")
    def _validate_page_starred(self, proposal: dict) -> list[int]:
        """Normalize per-page star flags."""
        val = list(proposal["value"])
        n_pages = max(1, int(getattr(self, "n_pages", 1)))
        if not val:
            return [0] * n_pages
        if len(val) != n_pages:
            raise traitlets.TraitError(
                f"page_starred length ({len(val)}) must equal n_pages ({n_pages})"
            )
        return [1 if int(v) else 0 for v in val]

    @traitlets.validate("hidden_panels")
    def _validate_hidden_panels(self, proposal: dict) -> list[int]:
        """Normalize hidden image indices and keep at least one image visible."""
        n_img = int(self.n_images)
        if n_img <= 0:
            return []
        clean: list[int] = []
        seen: set[int] = set()
        for value in proposal["value"]:
            idx = int(value)
            if 0 <= idx < n_img and idx not in seen:
                clean.append(idx)
                seen.add(idx)
        clean.sort()
        if len(clean) >= n_img:
            raise traitlets.TraitError(
                "hidden_panels cannot hide every panel; at least one panel must remain visible"
            )
        return self._normalize_item_page_hidden(clean)

    def _normalize_hidden_page_slots(
        self,
        values: Sequence[object],
        *,
        drop_if_full: bool = False,
    ) -> list[int]:
        """Normalize reusable hidden panel slots for paged galleries."""
        if (
            int(self.n_pages) <= 1
            or int(self.panels_per_page) <= 0
            or str(self.page_kind) == "items"
        ):
            return []
        n_slots = int(self.panels_per_page)
        clean_set: set[int] = set()
        for value in values or []:
            if isinstance(value, bool):
                continue
            try:
                slot = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= slot < n_slots:
                clean_set.add(slot)
        clean = sorted(clean_set)
        if len(clean) >= n_slots:
            if drop_if_full:
                clean = clean[:-1]
            else:
                raise traitlets.TraitError(
                    "hidden_page_slots cannot hide every page slot; at least one panel must remain visible"
                )
        return clean

    def _hidden_page_slots_from_panels(
        self,
        panels: Sequence[int],
        *,
        drop_if_full: bool = False,
    ) -> list[int]:
        """Map absolute hidden image indices to reusable page slots."""
        if (
            int(self.n_pages) <= 1
            or int(self.panels_per_page) <= 0
            or str(self.page_kind) == "items"
        ):
            return []
        n_img = int(self.n_images)
        per_page = int(self.panels_per_page)
        slots = [
            int(panel) % per_page
            for panel in panels
            if 0 <= int(panel) < n_img
        ]
        return self._normalize_hidden_page_slots(slots, drop_if_full=drop_if_full)

    @traitlets.validate("hidden_page_slots")
    def _validate_hidden_page_slots(self, proposal: dict) -> list[int]:
        """Normalize hidden page slots for paged galleries."""
        return self._normalize_hidden_page_slots(proposal["value"])

    @traitlets.validate("panel_order")
    def _validate_panel_order(self, proposal: dict) -> list[int]:
        """Normalize optional display order over source image indices."""
        n_img = int(self.n_images)
        values = list(proposal["value"] or [])
        if not values:
            return []
        clean: list[int] = []
        try:
            for value in values:
                if isinstance(value, bool):
                    raise TypeError
                clean.append(int(value))
        except (TypeError, ValueError) as exc:
            raise traitlets.TraitError("panel_order must contain integer panel indices") from exc
        expected = list(range(n_img))
        if len(clean) != n_img or sorted(clean) != expected:
            raise traitlets.TraitError(
                "panel_order must include every panel index exactly once "
                f"(expected a permutation of {expected!r})"
            )
        return clean

    def _panel_title_for_index(self, panel: int) -> str:
        """Return the user-facing label for a Show2D image panel."""
        if 0 <= panel < len(self.labels) and self.labels[panel]:
            return str(self.labels[panel])
        return f"Image {panel + 1}"

    def _resolve_panel_ref(self, panel: int | str) -> int:
        """Resolve a panel index or exact label into a zero-based image index."""
        if isinstance(panel, bool):
            raise TypeError("panel must be an integer index or exact label, not bool")
        if isinstance(panel, int):
            idx = int(panel)
            if 0 <= idx < int(self.n_images):
                return idx
            raise ValueError(f"panel index {idx} out of range [0, {self.n_images})")
        if isinstance(panel, str):
            titles = [self._panel_title_for_index(i) for i in range(int(self.n_images))]
            matches = [i for i, title in enumerate(titles) if title == panel]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError(
                    f"panel label {panel!r} is not unique; use a zero-based panel index instead"
                )
            available = ", ".join(repr(title) for title in titles)
            raise ValueError(f"unknown panel label {panel!r}; available labels: {available}")
        raise TypeError(
            f"panel must be an integer index or exact label, got {type(panel).__name__}"
        )

    def _normalize_panel_refs(
        self,
        panels: Sequence[int | str] | int | str,
        *,
        allow_empty: bool = False,
    ) -> list[int]:
        """Resolve and de-duplicate image panel references."""
        if isinstance(panels, (str, int)) and not isinstance(panels, bool):
            values: Sequence[int | str] = [panels]
        else:
            values = panels  # type: ignore[assignment]
        out: list[int] = []
        seen: set[int] = set()
        for panel in values:
            idx = self._resolve_panel_ref(panel)
            if idx not in seen:
                out.append(idx)
                seen.add(idx)
        if not out and not allow_empty:
            raise ValueError("at least one panel index or label is required")
        return out

    def _on_first_render(self, change):
        import time as _time
        from quantem.widget._timing import format_widget_render_timing
        if not change.get("new"):
            return
        total_ms = (_time.perf_counter() - self._init_t0) * 1000
        py_ms = self._init_py_elapsed_ms
        shape = (
            (self.n_images, self.height, self.width)
            if self.n_images > 1
            else (self.height, self.width)
        )
        mem = self._data.nbytes
        # Expose as attributes so tests and notebooks can assert on them.
        # These are the ground truth for "did JS actually paint": if they're
        # None, the JS side never signaled first render.
        self.render_total_ms = int(total_ms)
        self.render_python_build_ms = int(py_ms)
        self.render_wire_js_ms = int(total_ms - py_ms)
        if not getattr(self, "_save_state", False) and (self.frame_bytes or self.panel_stack_bytes):
            # The frontend has decoded and painted the pixels by the time it
            # flips ``_js_rendered``. Clear the synced model buffer so a later
            # notebook save stores the static PNG fallback, not the full 4k
            # array, while targeted initial sync remains intact.
            self.frame_bytes = b""
            self.panel_stack_bytes = b""
        if not getattr(self, "_verbose", True):
            pass
        else:
            print(
                format_widget_render_timing(
                    "Show2D",
                    shape=shape,
                    dtype=str(self._data.dtype),
                    raw_bytes=int(mem),
                    total_ms=total_ms,
                    python_ms=py_ms,
                    wire_js_ms=total_ms - py_ms,
                ),
                flush=True,
            )
        # Detach observer: one-shot, we only care about the first paint.
        try:
            self.unobserve(self._on_first_render, names=["_js_rendered"])
        except (ValueError, KeyError):
            pass

    def _folder_ordered_panel_indices(self) -> list[int]:
        """Return the complete path-stable order for a folder item gallery."""
        n_images = int(getattr(self, "n_images", 0))
        order = list(getattr(self, "panel_order", []) or [])
        if (
            len(order) == n_images
            and sorted(order) == list(range(n_images))
        ):
            return order
        return list(range(n_images))

    def _normalize_item_page_hidden(
        self,
        values: Sequence[int],
        *,
        drop_if_full: bool = False,
    ) -> list[int]:
        """Keep at least one concrete item visible on every folder page."""
        hidden = {int(value) for value in values}
        if (
            int(getattr(self, "n_pages", 1)) <= 1
            or int(getattr(self, "panels_per_page", 0)) <= 0
            or str(getattr(self, "page_kind", "comparison")) != "items"
        ):
            return sorted(hidden)
        order = self._folder_ordered_panel_indices()
        page_size = int(self.panels_per_page)
        for page_index, start in enumerate(range(0, len(order), page_size)):
            page = order[start : start + page_size]
            if page and all(panel in hidden for panel in page):
                if not drop_if_full:
                    raise traitlets.TraitError(
                        "hidden_panels cannot hide every panel on folder page "
                        f"{page_index + 1}; "
                        "leave at least one source image visible"
                    )
                hidden.discard(page[-1])
        return sorted(hidden)

    def _validate_folder_item_pages_visible(
        self,
        hidden: Iterable[int],
        operation: str,
    ) -> None:
        """Reject item-page state that would make any folder page empty."""
        if str(self.page_kind) != "items" or int(self.panels_per_page) <= 0:
            return
        hidden_set = {int(panel) for panel in hidden}
        page_size = int(self.panels_per_page)
        order = self._folder_ordered_panel_indices()
        for page_idx, start in enumerate(range(0, len(order), page_size)):
            panels = order[start : start + page_size]
            if panels and all(panel in hidden_set for panel in panels):
                raise ValueError(
                    f"{operation} would hide every panel on folder page "
                    f"{page_idx + 1}; leave at least one visible per page"
                )

    def _sync_folder_pages(
        self,
        *,
        anchor_panel: int | None = None,
        preferred_page: int | None = None,
    ) -> None:
        """Recompute sequential folder pages without rebuilding the widget."""
        page_size = getattr(self, "_folder_page_size", None)
        n_images = int(getattr(self, "n_images", 0))
        old_page = (
            int(getattr(self, "page_idx", 0))
            if preferred_page is None
            else int(preferred_page)
        )
        old_starred = list(getattr(self, "page_starred", []) or [])

        if page_size is None or n_images <= int(page_size):
            n_pages = 1
            panels_per_page = 0
            page_labels: list[str] = []
            next_page = 0
        else:
            panels_per_page = int(page_size)
            n_pages = int(math.ceil(n_images / panels_per_page))
            page_labels = [
                f"Images {start + 1}\u2013{min(n_images, start + panels_per_page)}"
                for start in range(0, n_images, panels_per_page)
            ]
            next_page = max(0, min(old_page, n_pages - 1))
            if anchor_panel is not None:
                order = self._folder_ordered_panel_indices()
                try:
                    position = order.index(int(anchor_panel))
                except (TypeError, ValueError):
                    pass
                else:
                    next_page = min(n_pages - 1, position // panels_per_page)

        next_starred = [0] * n_pages
        for index in range(min(len(old_starred), n_pages)):
            next_starred[index] = 1 if int(old_starred[index]) else 0

        with self.hold_sync():
            self.page_kind = "items"
            self.n_pages = n_pages
            self.panels_per_page = panels_per_page
            self.page_labels = page_labels
            self.page_starred = next_starred
            self.page_idx = next_page
            # Folder pages hide concrete files, not a repeated comparison slot.
            self.hidden_page_slots = []

    def _set_folder_waiting_empty_state(self) -> None:
        """Represent an acquisition folder with no readable image yet."""
        empty = np.empty((0, 0, 0), dtype=np.float32)
        with self.hold_sync():
            self._data = empty
            self._display_data = empty
            self._data_original = []
            self._panel_stacks = []
            self._panel_stacks_original = []
            self._display_panel_stacks = []
            self._rgb_frames = []
            self._display_rgb = []
            self.n_images = 0
            self.height = 0
            self.width = 0
            self.labels = []
            self.is_rgb = []
            self.starred = []
            self.hidden_panels = []
            self.hidden_page_slots = []
            self.panel_order = []
            self.image_rotations = []
            self.panel_frame_counts = []
            self.panel_frame_indices = []
            self.panel_stack_offsets = []
            self.panel_stack_bytes = b""
            self._panel_stack_mins = []
            self._panel_stack_maxs = []
            self.frame_bytes = b""
            self.stats_mean = []
            self.stats_min = []
            self.stats_max = []
            self.stats_std = []
            self._offline_mins = []
            self._offline_maxs = []
            self._static_fallback_jpeg = ""
            self.selected_idx = 0
            self.n_pages = 1
            self.page_idx = 0
            self.panels_per_page = 0
            self.page_labels = []
            self.page_starred = [0]
            self.folder_waiting = True

    def set_image(
        self,
        data,
        labels: list[str | None] | None = None,
        *,
        panel_frame_indices: Sequence[int] | None = None,
    ) -> None:
        """Replace the displayed image stack without rebuilding the widget.

        This is the light-weight live-update path used by ShowFolder watched
        selections. It preserves display controls such as colormap, contrast,
        FFT/profile toggles, and gallery layout, while resetting per-panel state
        tied to the previous image count or dimensions.
        """
        if isinstance(data, (list, tuple)):
            images = [
                to_numpy(item.array if hasattr(item, "array") else item)
                for item in data
            ]
            rgb_panels = [idx for idx, image in enumerate(images) if _is_rgb_item(image)]
            if rgb_panels:
                raise NotImplementedError(
                    "Show2D.set_image does not yet replace RGB panels; "
                    f"RGB panel indices: {rgb_panels}"
                )
            panel_stacks, resolved_indices, data = _normalize_grayscale_panel_items(
                images,
                panel_frame_indices,
            )
        else:
            if hasattr(data, "array") and hasattr(data, "name") and hasattr(data, "sampling"):
                data = data.array
            data = to_numpy(data)
            if data.ndim == 2:
                data = data[np.newaxis, ...]
            if data.ndim != 3:
                raise ValueError(f"Show2D.set_image expects a 2D image or 3D stack, got {data.ndim}D")
            if 0 in data.shape:
                raise ValueError(f"Empty image stack: shape {data.shape}. All dims must be >= 1.")
            if np.iscomplexobj(data):
                raise TypeError(
                    "Show2D does not accept complex data. Convert first: "
                    "np.abs(arr) for magnitude or np.angle(arr) for phase."
                )
            data = np.asarray(data, dtype=np.float32)
            if not np.isfinite(data).all():
                raise ValueError(
                    "Data contains NaN or inf. Clean first: "
                    "np.nan_to_num(arr, nan=0, posinf=0, neginf=0)."
                )
            panel_stacks = [data[i:i + 1] for i in range(int(data.shape[0]))]
            if panel_frame_indices is None:
                resolved_indices = [0] * len(panel_stacks)
            else:
                _, resolved_indices, data = _normalize_grayscale_panel_items(
                    [data[i] for i in range(int(data.shape[0]))],
                    panel_frame_indices,
                )

        previous_shape = (int(self.n_images), int(self.height), int(self.width))
        self._updating_panel_frames = True
        try:
            with self.hold_sync():
                self._data = data
                self._panel_stacks = panel_stacks
                self._panel_stacks_original = [stack for stack in panel_stacks]
                self._panel_stack_originals_are_views = True
                self.panel_frame_counts = [int(stack.shape[0]) for stack in panel_stacks]
                self.n_images = int(data.shape[0])
                self.panel_frame_indices = list(resolved_indices)
                self.panel_stack_offsets = [-1] * self.n_images
                self._data_original = [self._data[i] for i in range(self._data.shape[0])]
                self._originals_are_views = True
                self._rgb_frames = [None] * int(data.shape[0])
                self._display_rgb = self._rgb_frames
                self.is_rgb = [False] * int(data.shape[0])
                self.n_pages = 1
                self.page_idx = 0
                self.panels_per_page = 0
                self.page_labels = []
                self.page_starred = [0]
                self.image_rotations = [0] * self.n_images
                self.starred = [0] * self.n_images
                self.hidden_panels = []
                self.panel_order = []
                if labels is None:
                    self.labels = [f"Image {i + 1}" for i in range(self.n_images)]
                else:
                    if len(labels) != self.n_images:
                        raise ValueError(
                            f"labels length ({len(labels)}) must match n_images ({self.n_images})"
                        )
                    self.labels = ["" if label is None else str(label) for label in labels]
                self.selected_idx = min(int(self.selected_idx), self.n_images - 1)
                self.roi_list = []
                self.roi_selected_idx = -1
                self.profile_line = []
                self._detail_request = ""
                self._detail_meta = ""
                self._detail_bytes = b""
                self.vmins = None
                self.vmaxs = None

                self._display_bin = max(1, int(getattr(self, "_display_bin", 1)))
                if self._display_bin > 1:
                    from quantem.widget.utils.array import bin2d

                    self._display_panel_stacks = [
                        bin2d(stack, factor=self._display_bin, mode="mean")
                        for stack in self._panel_stacks
                    ]
                else:
                    self._display_panel_stacks = self._panel_stacks
                self._display_data = np.stack(
                    [stack[index] for stack, index in zip(self._display_panel_stacks, resolved_indices)],
                    axis=0,
                ).astype(np.float32, copy=False)
                display = self._display_data
                self.height = int(display.shape[1])
                self.width = int(display.shape[2])
                self._display_bin_factor = self._display_bin
                if (self.n_images, self.height, self.width) != previous_shape:
                    self.view_box = []
                    self.zoom_row = None
                    self.zoom_col = None
                if list(self.view_crop) or float(self.pad_ratio) > 0.0:
                    # Replacing the pixels invalidates a committed crop/pad:
                    # the window described a view of the OLD data.
                    self._view_ops_ready = False
                    self.view_crop = []
                    self.pad_ratio = 0.0
                    self._view_ops_ready = True
                    self._refresh_view_ops(announce=False)
                self._compute_all_stats()
                self._update_all_frames()
        finally:
            self._updating_panel_frames = False

    def _apply_folder_image_records(
        self,
        old_records: list[ImageFolderRecord],
        new_records: list[ImageFolderRecord],
        changed_arrays: dict[pathlib.Path, np.ndarray],
    ) -> None:
        """Replace folder panels while remapping panel state through file paths."""
        old_paths = [record.path for record in old_records]
        new_paths = [record.path for record in new_records]
        old_page_idx = int(getattr(self, "page_idx", 0))
        old_index = {path: idx for idx, path in enumerate(old_paths)}
        new_index = {path: idx for idx, path in enumerate(new_paths)}
        old_display_bin = max(1, int(getattr(self, "_display_bin", 1)))
        if getattr(self, "_folder_display_bin_request", None) == "auto":
            sample = changed_arrays.get(new_paths[0])
            if sample is None and old_paths:
                sample = np.asarray(self._panel_stacks_original[old_index[new_paths[0]]][0])
            if sample is not None:
                self._display_bin = self._folder_auto_display_bin(
                    len(new_paths),
                    int(sample.shape[-2]),
                    int(sample.shape[-1]),
                )
        new_display_bin = max(1, int(getattr(self, "_display_bin", 1)))
        coordinate_scale = old_display_bin / new_display_bin

        if not old_paths:
            self.set_image(
                [changed_arrays[path] for path in new_paths],
                labels=[self._folder_source.label(path) for path in new_paths],
            )
            self._sync_folder_pages(anchor_panel=0)
            return

        originals = list(getattr(self, "_panel_stacks_original", []))
        arrays: list[np.ndarray] = []
        for path in new_paths:
            if path in changed_arrays:
                arrays.append(changed_arrays[path])
                continue
            idx = old_index[path]
            stack = originals[idx]
            arrays.append(np.asarray(stack[0]))

        selected_path = (
            old_paths[int(self.selected_idx)]
            if old_paths and 0 <= int(self.selected_idx) < len(old_paths)
            else old_paths[0]
        )
        starred_by_path = {
            path: bool(self.starred[idx])
            for idx, path in enumerate(old_paths)
            if idx < len(self.starred)
        }
        hidden_paths = {
            old_paths[idx]
            for idx in self.hidden_panels
            if 0 <= int(idx) < len(old_paths)
        }
        rotations_by_path = {
            path: int(self.image_rotations[idx])
            for idx, path in enumerate(old_paths)
            if idx < len(self.image_rotations)
        }

        ordered_paths: list[pathlib.Path] = []
        if self.panel_order:
            ordered_paths.extend(
                old_paths[idx]
                for idx in self.panel_order
                if 0 <= int(idx) < len(old_paths)
            )
            ordered_paths.extend(path for path in new_paths if path not in ordered_paths)

        roi_active = bool(self.roi_active)
        roi_list = list(self.roi_list)
        roi_selected_idx = int(self.roi_selected_idx)
        profile_line = list(self.profile_line)
        view_box = [float(value) * coordinate_scale for value in self.view_box]
        zoom_row = (
            None if self.zoom_row is None else float(self.zoom_row) * coordinate_scale
        )
        zoom_col = (
            None if self.zoom_col is None else float(self.zoom_col) * coordinate_scale
        )

        def remap_panel_values(values):
            if values is None or len(values) != len(old_paths):
                return None
            by_path = {path: values[idx] for idx, path in enumerate(old_paths)}
            return [by_path.get(path) for path in new_paths]

        vmins = remap_panel_values(self.vmins)
        vmaxs = remap_panel_values(self.vmaxs)

        with self.hold_sync():
            self.set_image(
                arrays,
                labels=[self._folder_source.label(path) for path in new_paths],
            )
            self.image_rotations = [rotations_by_path.get(path, 0) for path in new_paths]
            self.starred = [int(starred_by_path.get(path, False)) for path in new_paths]
            self.panel_order = [new_index[path] for path in ordered_paths if path in new_index]
            self.selected_idx = new_index.get(selected_path, 0)
            # ``set_image`` resets paging traits. Keep the page the scientist
            # was reviewing even when selection lives on a different page.
            self._sync_folder_pages(preferred_page=old_page_idx)
            self.hidden_panels = self._normalize_item_page_hidden(
                [
                    new_index[path]
                    for path in hidden_paths
                    if path in new_index
                ],
                drop_if_full=True,
            )
            self.roi_active = roi_active
            self.roi_list = roi_list
            self.roi_selected_idx = roi_selected_idx
            self.profile_line = profile_line
            self.view_box = view_box
            self.zoom_row = zoom_row
            self.zoom_col = zoom_col
            if vmins is not None:
                self.vmins = vmins
            if vmaxs is not None:
                self.vmaxs = vmaxs

    def _folder_auto_display_bin(
        self,
        n_images: int,
        height: int,
        width: int,
    ) -> int:
        """Resolve ``display_bin='auto'`` for the current watched inventory."""
        factor = 1
        per_image_mb = (height * width * 4 * 3) / (1024 * 1024)
        total_mb = int(n_images) * per_image_mb
        if total_mb > self._GPU_DISPLAY_BUDGET_MB:
            for candidate in (2, 4, 8):
                if total_mb / (candidate * candidate) <= self._GPU_DISPLAY_BUDGET_MB:
                    factor = candidate
                    break
            else:
                factor = 8
        panel_bytes = height * width * 4
        if panel_bytes > self._WIRE_BUDGET_BYTES_PER_PANEL:
            preview_floor_px = 512.0 if int(n_images) >= 16 else 1024.0
            canvas_css_px = (
                float(self.size)
                if int(self.size) > 0
                else 300.0
                if int(n_images) > 1
                else 500.0
            )
            preview_px = max(preview_floor_px, 2.0 * canvas_css_px)
            wire_bin = max(2, math.ceil(max(height, width) / preview_px))
            while (
                panel_bytes / (wire_bin * wire_bin)
                > self._WIRE_BUDGET_BYTES_PER_PANEL
            ):
                wire_bin += 1
            factor = max(factor, wire_bin)
        return factor

    def __repr__(self) -> str:
        if self.n_images > 1:
            shape = f"{self.n_images}×{self.height}×{self.width}"
            return f"Show2D({shape}, idx={self.selected_idx}, cmap={self.cmap})"
        return f"Show2D({self.height}×{self.width}, cmap={self.cmap})"

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.log_scale:
            frame = np.log1p(np.maximum(frame, 0))
        if self.vmin is not None and self.vmax is not None:
            vmin = float(self.vmin)
            vmax = float(self.vmax)
            if self.log_scale:
                vmin = float(np.log1p(max(vmin, 0)))
                vmax = float(np.log1p(max(vmax, 0)))
        elif self.auto_contrast:
            vmin = float(np.percentile(frame, 2))
            vmax = float(np.percentile(frame, 98))
        else:
            vmin = float(frame.min())
            vmax = float(frame.max())
        if vmax > vmin:
            normalized = np.clip((frame - vmin) / (vmax - vmin) * 255, 0, 255)
            return normalized.astype(np.uint8)
        return np.zeros(frame.shape, dtype=np.uint8)

    @staticmethod
    def _downsample_static_frame(frame: np.ndarray, max_px: int) -> np.ndarray:
        """Area-downsample a frame for notebook PNG fallback rendering."""
        if max_px <= 0:
            return frame
        h, w = frame.shape[-2:]
        factor = max(1, int(math.ceil(max(h, w) / max_px)))
        if factor == 1:
            return frame

        trimmed_h = max(1, h // factor) * factor
        trimmed_w = max(1, w // factor) * factor
        trimmed = frame[:trimmed_h, :trimmed_w]
        return trimmed.reshape(
            trimmed_h // factor,
            factor,
            trimmed_w // factor,
            factor,
        ).mean(axis=(1, 3))

    def save_image(
        self,
        path: str | pathlib.Path,
        *,
        idx: int | None = None,
        format: str | None = None,
        dpi: int = 150,
        title: bool | str = False,
        colorbar: bool = False,
        scalebar: bool = False,
    ) -> pathlib.Path:
        """Save current image as PNG, PDF, or TIFF.

        When ``title``, ``colorbar``, or ``scalebar`` are enabled, the output
        is a publication-quality figure rendered via matplotlib. Otherwise a
        raw colormapped image is saved directly (faster, exact pixel output).

        Parameters
        ----------
        path : str or pathlib.Path
            Output file path.
        idx : int, optional
            Image index in gallery mode. Defaults to current selected_idx.
        format : str, optional
            'png', 'pdf', or 'tiff'. If omitted, inferred from file extension.
        dpi : int, default 150
            Output DPI.
        title : bool or str, default False
            ``True`` uses the widget title, a string sets a custom title.
        colorbar : bool, default False
            Include a colorbar showing the intensity mapping.
        scalebar : bool, default False
            Include a scale bar (requires ``pixel_size > 0``).

        Returns
        -------
        pathlib.Path
            The written file path.
        """
        from matplotlib import colormaps
        from PIL import Image

        path = pathlib.Path(path)
        fmt = (format or path.suffix.lstrip(".").lower() or "png").lower()
        if fmt not in ("png", "pdf", "tiff", "tif"):
            raise ValueError(f"Unsupported format: {fmt!r}. Use 'png', 'pdf', or 'tiff'.")

        i = idx if idx is not None else self.selected_idx
        if i < 0 or i >= self.n_images:
            raise IndexError(f"Image index {i} out of range [0, {self.n_images})")

        panel_rgb = self._rgb_frames[i] if getattr(self, "_rgb_frames", None) else None
        frame = self._data[i]
        cmap_fn = colormaps.get_cmap(self.cmap)
        path.parent.mkdir(parents=True, exist_ok=True)
        if panel_rgb is not None:
            # RGB panels are display-ready: save the color pixels directly,
            # bypassing colormap and contrast. Colorbar is meaningless for an
            # RGB composite, so it is skipped.
            display_pixels = (np.clip(panel_rgb, 0.0, 1.0) * 255).astype(np.uint8)
            colorbar = False
        else:
            normalized = self._normalize_frame(frame)
            display_pixels = None

        use_figure = title or colorbar or scalebar
        if not use_figure:
            if panel_rgb is not None:
                img = Image.fromarray(display_pixels, mode="RGB")
            else:
                rgba = (cmap_fn(normalized / 255.0) * 255).astype(np.uint8)
                img = Image.fromarray(rgba)
                if fmt == "pdf":
                    Image.init()
                    img = img.convert("RGB")
            img.save(str(path), dpi=(dpi, dpi))
            return path

        # Publication-quality figure via matplotlib
        h, w = frame.shape
        aspect = h / w
        fig_w = 6
        fig, ax = plt.subplots(figsize=(fig_w, fig_w * aspect))
        if panel_rgb is not None:
            im = ax.imshow(display_pixels, origin="upper")
        else:
            im = ax.imshow(normalized, cmap=cmap_fn, vmin=0, vmax=255, origin="upper")
        ax.axis("off")

        if title:
            label = title if isinstance(title, str) else self.title
            if label:
                ax.set_title(label, fontsize=14, fontweight="bold", pad=8)

        if colorbar:
            # Map 0–255 back to data-space values for tick labels
            if self.log_scale:
                frame_proc = np.log1p(np.maximum(frame, 0))
            else:
                frame_proc = frame
            if self.vmin is not None and self.vmax is not None:
                dmin = float(self.vmin)
                dmax = float(self.vmax)
                if self.log_scale:
                    dmin = float(np.log1p(max(dmin, 0)))
                    dmax = float(np.log1p(max(dmax, 0)))
            elif self.auto_contrast:
                dmin = float(np.percentile(frame_proc, 2))
                dmax = float(np.percentile(frame_proc, 98))
            else:
                dmin = float(frame_proc.min())
                dmax = float(frame_proc.max())
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            n_ticks = 5
            tick_positions = np.linspace(0, 255, n_ticks)
            tick_labels = [f"{dmin + (dmax - dmin) * t / 255:.4g}" for t in tick_positions]
            cb.set_ticks(tick_positions)
            cb.set_ticklabels(tick_labels)

        if scalebar and self.pixel_size > 0:
            # Compute a nice scale bar length
            target_frac = 0.2  # ~20% of image width
            raw_length_px = target_frac * w
            raw_length_phys = raw_length_px * self.pixel_size  # in Å
            nice = _round_to_nice(raw_length_phys)
            bar_px = nice / self.pixel_size
            if nice >= 10:
                label_text = f"{nice / 10:.4g} nm"
            else:
                label_text = f"{nice:.4g} Å"
            margin = 0.03
            bar_y = h * (1 - margin) - 2
            bar_x = w * (1 - margin) - bar_px
            ax.plot([bar_x, bar_x + bar_px], [bar_y, bar_y],
                    color="white", linewidth=3, solid_capstyle="butt")
            ax.plot([bar_x, bar_x + bar_px], [bar_y, bar_y],
                    color="black", linewidth=1, solid_capstyle="butt")
            ax.text(bar_x + bar_px / 2, bar_y - h * 0.02, label_text,
                    color="white", fontsize=10, fontweight="bold",
                    ha="center", va="bottom",
                    path_effects=[
                        matplotlib.patheffects.withStroke(linewidth=2, foreground="black")
                    ])

        fig.savefig(str(path), dpi=dpi, bbox_inches="tight",
                    facecolor="white", pad_inches=0.1)
        plt.close(fig)
        return path

    @property
    def view_corner(self):
        """Current view as ``(row0, col0, size)`` - paste straight into ``view_box=``.

        ``view_box`` live-syncs from JS on every pan/zoom, so after interacting this
        reads the region on screen. Restore next session with
        ``Show2D(..., view_box=w.view_corner)``.
        """
        if not self.view_box:
            return (0.0, 0.0, float(max(self.height, self.width)))
        r0, r1, c0, c1 = self.view_box
        return (round(r0, 1), round(c0, 1), round(max(r1 - r0, c1 - c0), 1))

    @property
    def view_center(self):
        """Current view as ``dict(zoom=..., zoom_row=..., zoom_col=...)`` - paste as kwargs.

        The center/zoom flavor of :attr:`view_corner`: restore with
        ``Show2D(..., **w.view_center)``.
        """
        if not self.view_box:
            return dict(zoom=1.0, center=None)
        r0, r1, c0, c1 = self.view_box
        zoom = float(min(self.height / max(1.0, r1 - r0), self.width / max(1.0, c1 - c0)))
        return dict(zoom=round(zoom, 2), center=(round((r0 + r1) / 2, 1), round((c0 + c1) / 2, 1)))

    def _on_detail_request_change(self, change: dict) -> None:
        """Serve a maps-style detail request: crop the visible window from the
        FULL-resolution data, mean-bin it to near canvas resolution, and reply
        with a small float32 buffer. This is how a binned preview still shows
        true full-res pixels under zoom: the browser swaps the tile in over
        the preview, and the 100+ MB full frame never crosses the wire.
        Coordinates arrive in preview pixels (the JS-side image space) and are
        scaled back to full resolution here; the reply reports full-res
        coordinates plus the tile bin so JS can place it exactly."""
        raw = str(change.get("new") or "")
        if not raw or getattr(self, "_data", None) is None:
            return
        from quantem.widget.utils.array import bin2d
        try:
            request = json.loads(raw)
        except json.JSONDecodeError:
            return
        factor = max(1, int(self._display_bin_factor))
        full_h, full_w = int(self._data.shape[1]), int(self._data.shape[2])
        # With a crop/pad active, JS coordinates are frame-local: shift into
        # full-image pixels here, clamp to the committed crop window so a
        # tile never un-crops the view, and reply in frame-local coordinates.
        offset_row, offset_col = (int(v) for v in self._view_crop_offset)
        if len(self.view_crop) == 4:
            win_row0, win_row1, win_col0, win_col1 = (int(v) for v in self.view_crop)
        else:
            win_row0, win_row1, win_col0, win_col1 = 0, full_h, 0, full_w
        tiles: list[dict] = []
        blocks: list[bytes] = []
        offset = 0
        for spec in request.get("tiles", []):
            panel = int(spec.get("panel", -1))
            # RGB panels keep preview-only rendering: a color tile would need a
            # second interleaved payload path for a rare panel type.
            if not (0 <= panel < self.n_images) or (panel < len(self.is_rgb) and self.is_rgb[panel]):
                continue
            bin_factor = max(1, int(spec.get("bin", 1)))
            # Snap the window outward to bin multiples anchored at the image
            # origin so mean-binning blocks tile the crop with no partial edges.
            row0 = max(win_row0, math.floor((float(spec["row0"]) * factor + offset_row) / bin_factor) * bin_factor)
            col0 = max(win_col0, math.floor((float(spec["col0"]) * factor + offset_col) / bin_factor) * bin_factor)
            row1 = min(win_row1, math.ceil((float(spec["row1"]) * factor + offset_row) / bin_factor) * bin_factor)
            col1 = min(win_col1, math.ceil((float(spec["col1"]) * factor + offset_col) / bin_factor) * bin_factor)
            if row1 <= row0 or col1 <= col0:
                continue
            crop = self._data[panel, row0:row1, col0:col1]  # view: never copies the full array
            tile = bin2d(crop, factor=bin_factor, mode="mean") if bin_factor > 1 else crop
            tile = np.ascontiguousarray(tile, dtype=np.float32)
            if self._panel_filter_active(panel):
                # A raw full-res tile over a filtered preview would silently
                # un-filter the zoomed view. Filtering the crop is edge-
                # approximate near the tile border, but honest about the knobs.
                # Detail tiles stay on this Python path even when the browser
                # filters the preview (_webgpu_filter_ok): tiles only exist in
                # kernel sessions and already round-trip, and the committed
                # knobs here match the browser's committed knobs.
                tile = np.ascontiguousarray(self._filter_display_frame(tile, panel=panel))
            blocks.append(tile.tobytes())
            tiles.append({"panel": panel, "row0": row0 - offset_row, "col0": col0 - offset_col,
                          "rows": int(tile.shape[0]), "cols": int(tile.shape[1]),
                          "bin": bin_factor, "offset": offset})
            offset += tile.nbytes
        # One comm message for bytes + meta so JS never pairs a fresh meta with
        # a stale buffer (or vice versa) mid-update.
        with self.hold_sync():
            self._detail_bytes = _b64_safe(b"".join(blocks))
            self._detail_meta = json.dumps({"id": str(request.get("id", "")), "tiles": tiles})

    # Traits that carry the bulk pixel payload. Dropped from the saved-notebook
    # snapshot when save_state is False so a plain display stays a few MB, not GB.
    _UNSAVED_HEAVY_KEYS = (
        "frame_bytes",
        "panel_stack_bytes",
        "export_payload",
        "_detail_bytes",
    )

    def get_state(self, key=None, drop_defaults=False):
        """Trait state for comm sync and notebook embedding.

        ipywidgets calls this with ``key=None`` to snapshot the FULL state that
        gets written into the saved notebook's ``metadata.widgets``. When
        ``save_state`` is False we drop the heavy image buffers from that
        snapshot, so a plain Show2D does not bake ~1 GB of pixels into the
        .ipynb. Targeted syncs (``key`` is a name or set, used by hold_sync /
        send_state during live rendering) are untouched, so the frontend still
        receives ``frame_bytes`` normally. ``save_state=True`` embeds everything
        so a reopened notebook restores the interactive widget without a kernel.
        """
        state = super().get_state(key=key, drop_defaults=drop_defaults)
        if key is None and not getattr(self, "_save_state", False):
            if not self._static_fallback_enabled():
                state.pop("_static_fallback_jpeg", None)
                state.pop("_static_fallback_mime", None)
            elif not self._static_fallback_jpeg:
                png = self._static_fallback_png_b64()
                if png:
                    self._store_static_fallback_preview(png)
                    state["_static_fallback_jpeg"] = self._static_fallback_jpeg
                    state["_static_fallback_mime"] = self._static_fallback_mime
            for heavy_key in self._UNSAVED_HEAVY_KEYS:
                state.pop(heavy_key, None)
        if (
            key is None
            and not getattr(self, "_initial_live_mount_state", False)
            and state.get("folder_watch_state") in {
            "watching",
            "updating",
            "waiting",
            "error",
            }
        ):
            # Saved widget state retains pixels, not the Python watcher thread.
            # Keep snapshots truthful even when exported while acquisition is
            # still running in the current kernel.
            state["folder_watch_state"] = "stopped"
            state["folder_watch_detail"] = (
                "Folder watcher is not running in saved widget state. "
                "Re-run the cell to resume live folder updates."
            )
            if state.get("folder_waiting"):
                state["folder_status"] = (
                    "No completed image was captured in this saved widget. "
                    "Re-run the cell to resume folder watching."
                )
        return state

    def _with_initial_live_mount_state(self, fn, *args, **kwargs):
        """Keep live watcher state truthful during the first display handshake."""
        self._initial_live_mount_state = True
        try:
            return fn(*args, **kwargs)
        finally:
            self._initial_live_mount_state = False

    def _repr_mimebundle_(self, **kwargs):
        return self._with_initial_live_mount_state(
            super()._repr_mimebundle_,
            **kwargs,
        )

    def _ipython_display_(self):
        return self._with_initial_live_mount_state(super()._ipython_display_)

    # Colormaps the frontend treats as sequential; a signed diff panel switches
    # to a diverging map (RdBu) because zero must sit at the visual midpoint.
    _SEQUENTIAL_CMAPS = frozenset(
        {"inferno", "viridis", "plasma", "magma", "hot", "gray", "turbo"}
    )

    @staticmethod
    def _signed_log1p(values: np.ndarray | float) -> np.ndarray | float:
        """Signed log1p, matching the frontend's ``applyLogScale``.

        The widget maps negative intensities to ``-log1p(-v)`` so diff-like
        data keeps its sign under log scale; a plain ``log1p(clip(v, 0))``
        would collapse everything below zero and diverge from the live render.
        """
        return np.sign(values) * np.log1p(np.abs(values))

    @staticmethod
    def _format_stat(value: float) -> str:
        """Format a statistic like the widget's stats row (JS ``formatNumber``).

        ``0`` stays ``"0"``; magnitudes >= 1000 or < 0.01 use two-decimal
        scientific notation with an unpadded exponent (``5.85e+3``, matching
        JS ``toExponential(2)``); everything else uses two fixed decimals.
        """
        if value == 0:
            return "0"
        if abs(value) >= 1000 or abs(value) < 0.01:
            mantissa, exponent = f"{value:.2e}".split("e")
            return f"{mantissa}e{int(exponent):+d}"
        return f"{value:.2f}"

    def _resolve_panel_display_ranges(self, frames: list[np.ndarray]) -> list[tuple[float, float]]:
        """Per-panel ``(vmin, vmax)`` in display space, mirroring the frontend.

        Precedence (same as the JS colormap effect in ``js/show2d/index.tsx``):
        per-image ``vmins[i]/vmaxs[i]`` beat the scalar ``vmin/vmax``, which
        beats ``auto_contrast`` (2/98 percentiles), which beats the frame's
        min/max. With ``link_contrast`` on a gallery and no explicit ranges,
        panels share one merged range so cross-panel intensities compare
        directly. Under ``log_scale`` the limits live in signed-log1p space,
        exactly like the shader. Ranges are computed on the FULL-resolution
        display frames so the PNG's contrast matches the widget even though
        the PNG pixels are later area-binned.
        """
        num = len(frames)
        # RGB panels bypass the contrast pipeline entirely: their range is the
        # fixed display-ready [0, 1], and they are excluded from linked-contrast
        # merging so a [0, 1] overlay never drags a counts-scaled panel's window.
        rgb = list(self.is_rgb) if any(self.is_rgb) else [False] * num
        gray = [i for i in range(num) if not (i < len(rgb) and rgb[i])]
        def to_display(value: float) -> float:
            return float(self._signed_log1p(float(value))) if self.log_scale else float(value)
        has_absolute = self.vmin is not None and self.vmax is not None
        per_mins = list(self.vmins) if self.vmins else [None] * num
        per_maxs = list(self.vmaxs) if self.vmaxs else [None] * num
        has_per_image = [per_mins[i] is not None and per_maxs[i] is not None for i in range(num)]
        base_ranges: list[tuple[float, float]] = []
        for i, frame in enumerate(frames):
            if i < len(rgb) and rgb[i]:
                base_ranges.append((0.0, 1.0))
            elif has_per_image[i]:
                base_ranges.append((to_display(per_mins[i]), to_display(per_maxs[i])))
            elif has_absolute:
                base_ranges.append((to_display(self.vmin), to_display(self.vmax)))
            else:
                base_ranges.append((to_display(frame.min()), to_display(frame.max())))
        linked_shared = (
            self.link_contrast and len(gray) >= 2 and not has_absolute and not any(has_per_image)
        )
        shared_base = None
        if linked_shared:
            shared_base = (min(base_ranges[i][0] for i in gray), max(base_ranges[i][1] for i in gray))
        auto_ranges: list[tuple[float, float]] = []
        use_auto = self.auto_contrast and not has_absolute
        if use_auto:
            for i, frame in enumerate(frames):
                if has_per_image[i] or (i < len(rgb) and rgb[i]):
                    auto_ranges.append(base_ranges[i])
                    continue
                processed = self._signed_log1p(frame) if self.log_scale else frame
                lo, hi = (float(v) for v in np.percentile(processed, (2, 98)))
                # Sparse/clustered data can collapse the 2-98% window to a
                # point; fall back to full extrema like computeAutoRange does.
                full_lo, full_hi = float(processed.min()), float(processed.max())
                if hi - lo <= max(1e-12, abs(full_hi - full_lo) * 1e-6):
                    lo, hi = full_lo, full_hi
                auto_ranges.append((lo, hi))
            if linked_shared:
                shared_auto = (min(auto_ranges[i][0] for i in gray), max(auto_ranges[i][1] for i in gray))
        ranges: list[tuple[float, float]] = []
        for i in range(num):
            if i < len(rgb) and rgb[i]:
                ranges.append((0.0, 1.0))
            elif use_auto and not has_per_image[i]:
                ranges.append(shared_auto if linked_shared else auto_ranges[i])
            else:
                ranges.append(shared_base if linked_shared else base_ranges[i])
        return ranges

    def _static_panel_rgb(
        self,
        frame: np.ndarray,
        vmin: float,
        vmax: float,
        cmap_name: str,
        *,
        apply_log: bool | None = None,
    ) -> np.ndarray:
        """Colormap one panel exactly as the live widget maps pixels to colors.

        Pipeline: optional signed-log1p on the data, clip-normalize into the
        display-space ``[vmin, vmax]`` window, then look up the matplotlib
        colormap (the same LUT the JS mirrors). Returning explicit RGB uint8
        keeps matplotlib's own norm machinery out of the loop, so the PNG's
        pixel mapping is byte-identical to what tests can compute independently.
        """
        from matplotlib import colormaps
        apply_log = self.log_scale if apply_log is None else apply_log
        processed = self._signed_log1p(frame) if apply_log else frame
        if vmax > vmin:
            normalized = np.clip((processed - vmin) / (vmax - vmin), 0.0, 1.0)
        else:
            normalized = np.zeros(processed.shape, dtype=np.float64)
        rgba = colormaps.get_cmap(cmap_name)(normalized)
        return (rgba[..., :3] * 255).astype(np.uint8)

    def _static_panel_specs(self) -> list[dict]:
        """Panel plan for the static PNG, mirroring the live widget's layout.

        One entry per visible image panel, plus one signed diff panel per
        non-reference image when ``diff_mode`` is on (the widget renders
        ``ref - other`` with a symmetric range around zero and a diverging
        colormap). Each spec carries the full-resolution frame plus the
        resolved display-space contrast window and a stats line, so the PNG
        renderer only has to bin and colormap.
        """
        frames_source = getattr(self, "_display_data", None)
        if frames_source is None:
            frames_source = getattr(self, "_data", None)
        if frames_source is None or len(frames_source) == 0:
            return []
        frames = [frames_source[i] for i in range(len(frames_source))]
        ranges = self._resolve_panel_display_ranges(frames)
        def stats_line(mean: float, lo: float, hi: float, std: float) -> str:
            fmt = self._format_stat
            return f"Mean {fmt(mean)}   Min {fmt(lo)}   Max {fmt(hi)}   Std {fmt(std)}"

        def panel_stats_line(index: int, frame: np.ndarray) -> str:
            stat_arrays = (self.stats_mean, self.stats_min, self.stats_max, self.stats_std)
            if all(index < len(values) for values in stat_arrays):
                return stats_line(
                    self.stats_mean[index],
                    self.stats_min[index],
                    self.stats_max[index],
                    self.stats_std[index],
                )
            data = np.asarray(frame, dtype=np.float64)
            return stats_line(
                float(np.nanmean(data)),
                float(np.nanmin(data)),
                float(np.nanmax(data)),
                float(np.nanstd(data)),
            )

        specs: list[dict] = []
        display_rgb = getattr(self, "_display_rgb", None) or [None] * len(frames)
        for i in self.visible_panels:
            if i >= len(frames):
                continue
            label = self._panel_title_for_index(i) if self.show_panel_titles else ""
            panel_rgb = display_rgb[i] if i < len(display_rgb) else None
            frame = panel_rgb if panel_rgb is not None else frames[i]
            specs.append({
                # RGB panels pass their display-ready pixels straight through:
                # no colormap, no log, no contrast window (stats stay luminance).
                "frame": frame,
                "rgb": panel_rgb is not None,
                "vmin": ranges[i][0],
                "vmax": ranges[i][1],
                "cmap": self.cmap,
                "apply_log": self.log_scale and panel_rgb is None,
                "label": label,
                "stats": panel_stats_line(i, frame),
            })
        if self.diff_mode and len(frames) >= 2:
            ref = int(self.diff_reference)
            diff_cmap = "RdBu" if self.cmap in self._SEQUENTIAL_CMAPS else self.cmap
            for other in range(len(frames)):
                # RGB panels never get a diff panel: a signed residual against
                # a display-ready color composite is meaningless.
                if other == ref or (other < len(self.is_rgb) and self.is_rgb[other]):
                    continue
                diff = frames[ref] - frames[other]
                # Symmetric window centers zero on the diverging map's midpoint,
                # so positive and negative residuals read with equal weight.
                sym = float(max(abs(float(diff.min())), abs(float(diff.max())))) or 1.0
                label = ("Diff (A − B)" if len(frames) == 2
                         else f"Diff (#{ref + 1} − #{other + 1})")
                specs.append({
                    "frame": diff,
                    "vmin": -sym,
                    "vmax": sym,
                    "cmap": diff_cmap,
                    "apply_log": False,  # widget diffs raw data, never log-scaled
                    "label": label if self.show_panel_titles else "",
                    "stats": stats_line(float(diff.mean()), float(diff.min()),
                                        float(diff.max()), float(diff.std())),
                })
        return specs

    @staticmethod
    def _center_crop_slices(height: int, width: int, zoom: float) -> tuple[slice, slice]:
        """Central 1/zoom crop, matching the live widget's zoomed viewport.

        The widget at zoom z (pan 0) scales the full image about the canvas
        center, so the visible region is the central ``height/z x width/z``
        window. The static PNG must show the same pixels or the fallback
        looks nothing like the screenshot the user saved."""
        if zoom <= 1:
            return slice(0, height), slice(0, width)
        crop_h = max(1, _js_round(height / zoom))
        crop_w = max(1, _js_round(width / zoom))
        top = (height - crop_h) // 2
        left = (width - crop_w) // 2
        return slice(top, top + crop_h), slice(left, left + crop_w)

    def _static_canvas_css_px(self) -> float:
        """CSS width of the live panel canvas, the length every JS overlay
        constant (16px font, 12px margin, 60px bar target) is relative to.

        Port of js/show2d/index.tsx: the ``size`` trait when set, else
        SINGLE_IMAGE_TARGET (500) for one image / GALLERY_IMAGE_TARGET (300)
        for a gallery. Without this the static overlays would be drawn for a
        fictitious canvas size and read visibly smaller than the widget's."""
        if self.size > 0:
            return float(self.size)
        return 300.0 if self.n_images > 1 else 500.0

    def _static_overlay_texts(self, specs: list[dict] | None = None,
                              *, css_px: float | None = None) -> list[tuple[str, str, str, float]]:
        """Per-panel overlay strings for the static PNG, one tuple
        ``(label, zoom_text, bar_text, bar_px)`` per panel.

        Pure port of js/figure.ts drawScaleBarHiDPI's math, evaluated on a
        ``css_px``-wide canvas (default: the live widget's own canvas CSS
        width from ``_static_canvas_css_px``): effectiveZoom =
        zoom * cssWidth / imageWidth, target bar 60 CSS px rounded to a nice
        physical length, label via formatScaleLabel. Uncalibrated data gets
        pixelSize 1 and unit "px" exactly like the widget (show2d/index.tsx
        overlay effect). ``bar_px`` is the bar length in panel CSS px;
        ``bar_text``/``bar_px`` are ""/0.0 when the scale bar is hidden.
        Exposed separately from the renderer so tests can assert the strings
        without OCR-ing the PNG."""
        if specs is None:
            specs = self._static_panel_specs()
        if css_px is None:
            css_px = self._static_canvas_css_px()
        # widget clamps initial_zoom to [MIN_ZOOM, MAX_ZOOM] (index.tsx)
        zoom = min(max(float(self.initial_zoom) or 1.0, 0.5), 20.0)
        zoom_text = f"{zoom:.1f}×"  # JS: `${zoom.toFixed(1)}×`
        calibrated = self.pixel_size > 0
        pixel_size = self.pixel_size if calibrated else 1.0
        unit = self.pixel_unit if calibrated else "px"
        texts: list[tuple[str, str, str, float]] = []
        for spec in specs:
            if not self.scale_bar_visible:
                texts.append((spec["label"], zoom_text, "", 0.0))
                continue
            full_w = spec["frame"].shape[1]
            effective_zoom = zoom * css_px / full_w
            # 60 CSS px target bar, rounded to a nice physical length
            nice = _round_to_nice(60.0 / effective_zoom * pixel_size)
            bar_px = nice / pixel_size * effective_zoom
            texts.append((spec["label"], zoom_text,
                          _format_scale_label(nice, unit), bar_px))
        return texts

    def _static_png_b64(self, *, max_px: int = 512, dpi: int = 160) -> str | None:
        """Base64 PNG of all panels, attached to the cell output.

        With ``save_state`` False the interactive widget state is not embedded,
        so a reopened notebook (GitHub, nbviewer, cold Lab) would show nothing.
        This render mirrors the live widget panel-for-panel: same colormap,
        same per-panel or linked contrast window (resolved on the
        full-resolution frame so percentile cuts match the widget, then
        applied to the binned pixels), the same central 1/zoom viewport,
        diff panel(s) when ``diff_mode`` is on, and the widget's own in-panel
        overlays - label top-center, zoom badge bottom-left, scale bar with
        its label bottom-right - at the exact CSS-pixel geometry the JS draws
        on a panel of this size. Panels are area-mean binned to ~``max_px``
        so atomic-lattice detail averages instead of aliasing, and the render
        stays cheap on every display.
        """
        import base64
        import io as _io
        specs = self._static_panel_specs()
        if not specs:
            return None
        num = len(specs)
        # Total-pixel budget: a large survey gallery (e.g. 38 panels) at a fixed
        # 512 px/panel produced a ~27 MB PNG and a ~73 MB notebook (noisy STEM
        # content compresses poorly, ~2.7 bytes/px). The fallback is a reopen
        # preview, not the data, so shrink per-panel resolution as the panel
        # count grows (~2 MP total -> a few MB, floor 160 px) instead of
        # scaling the file linearly with the gallery. Small galleries keep the
        # full 512 px/panel.
        budget_px = int((2_000_000 / num) ** 0.5)
        max_px = max(160, min(max_px, budget_px))
        css_w = self._static_canvas_css_px()
        overlays = self._static_overlay_texts(specs, css_px=css_w)
        zoom = min(max(float(self.initial_zoom) or 1.0, 0.5), 20.0)
        ncols = max(1, min(self.ncols, num))
        nrows = (num + ncols - 1) // ncols
        # cells sized to the panels' cropped aspect so every image fills its
        # cell exactly: a taller cell would pad panels with white and make the
        # horizontal gutters read wider than the vertical ones
        h0, w0 = specs[0]["frame"].shape[:2]
        rows0, cols0 = self._center_crop_slices(h0, w0, zoom)
        aspect = (rows0.stop - rows0.start) / (cols0.stop - cols0.start)
        # inter-panel gutter is the widget's own gallery gap (CSS px of the
        # live canvas), identical horizontally and vertically
        gap_frac = max(0, int(self.gallery_gap_px)) / css_w
        cell_w_in = max_px / dpi  # cell width in inches so 1 panel = max_px device px
        cell_h_in = cell_w_in * aspect
        gap_in = gap_frac * cell_w_in
        # Build an unmanaged Figure rather than registering one through
        # pyplot. In a live Jupyter kernel this renderer runs from a
        # ``post_execute`` callback, alongside matplotlib-inline's own figure
        # flushing callback. A pyplot-managed multi-panel figure can be
        # cleared by that callback before ``savefig`` draws it, leaving a
        # correctly sized but completely white saved-notebook preview. A
        # standalone Figure has the same Agg rendering path without entering
        # Jupyter's global figure-manager lifecycle.
        fig = matplotlib.figure.Figure(
            figsize=(
                ncols * cell_w_in + (ncols - 1) * gap_in,
                nrows * cell_h_in + (nrows - 1) * gap_in,
            )
        )
        # wspace/hspace are fractions of cell width/height; both resolve to
        # the same gap_in inches so the white gutters match to the pixel
        grid = fig.add_gridspec(nrows, ncols, wspace=gap_frac,
                                hspace=gap_frac / aspect,
                                left=0, right=1, bottom=0, top=1)
        font_family = _static_overlay_font()
        # the live canvas is css_w CSS px wide but the PNG panel is max_px
        # device px wide, so every CSS-px size renders scaled by max_px/css_w
        point = (max_px / css_w) * 72.0 / dpi  # matplotlib points per CSS px
        for idx, (spec, (label, zoom_text, bar_text, bar_css)) in enumerate(zip(specs, overlays)):
            ax = fig.add_subplot(grid[idx // ncols, idx % ncols])
            ax.axis("off")
            frame = spec["frame"]
            rows, cols = self._center_crop_slices(frame.shape[0], frame.shape[1], zoom)
            if spec.get("rgb"):
                # RGB panels bypass the colormap: bin each channel and pass the
                # display-ready pixels straight through, matching the live view.
                cropped = frame[rows, cols]
                binned_channels = [self._downsample_static_frame(cropped[..., ch], max_px=max_px)
                                   for ch in range(3)]
                rgb = (np.clip(np.stack(binned_channels, axis=-1), 0.0, 1.0) * 255).astype(np.uint8)
            else:
                binned = self._downsample_static_frame(frame[rows, cols], max_px=max_px)
                rgb = self._static_panel_rgb(binned, spec["vmin"], spec["vmax"],
                                             spec["cmap"], apply_log=spec["apply_log"])
            ax.imshow(rgb, interpolation="nearest")
            bin_h, bin_w = rgb.shape[:2]
            ax.set(xlim=(-0.5, bin_w - 0.5), ylim=(bin_h - 0.5, -0.5))
            # CSS px -> data px: the panel canvas is css_w CSS px wide showing
            # bin_w image pixels, so overlay geometry scales by bin_w / css_w
            css_h = css_w * bin_h / bin_w
            k = bin_w / css_w

            def css_xy(x_css: float, y_css: float) -> tuple[float, float]:
                return x_css * k - 0.5, y_css * k - 0.5
            # widget textShadow "1px 1px 0 rgba(0,0,0,0.85), 0 0 3px ..." reads
            # as a soft dark outline; a thin translucent black stroke is its
            # closest matplotlib match (a full-opacity stroke looks stenciled)
            stroke = [matplotlib.patheffects.withStroke(
                linewidth=1.5 * point, foreground=(0, 0, 0, 0.8))]
            if label:
                # panel title Box (show2d/index.tsx): top 6px inside the image,
                # centered, bold max(8, panel_title_font_size) px white @ 95%
                title_px = max(8, int(self.panel_title_font_size or 11))
                ax.text(*css_xy(css_w / 2, 6), label, color=(1, 1, 1, 0.95),
                        fontsize=title_px * point, fontweight="bold",
                        fontfamily=font_family, ha="center", va="top",
                        path_effects=stroke)
            if self.scale_bar_visible:
                # drawScaleBarHiDPI (js/figure.ts): margin 12, bar 5 px thick
                # bottom-right, 16px label centered 4px above the bar, zoom
                # badge left-aligned at x=12 sharing the bar's bottom edge,
                # all under a soft (1,1)-offset half-black shadow
                bar_x = css_w - bar_css - 12
                bar_y = css_h - 12
                ax.add_patch(matplotlib.patches.Rectangle(
                    css_xy(bar_x + 1, bar_y + 1), bar_css * k, 5 * k,
                    facecolor=(0, 0, 0, 0.5), edgecolor="none"))
                ax.add_patch(matplotlib.patches.Rectangle(
                    css_xy(bar_x, bar_y), bar_css * k, 5 * k,
                    facecolor="white", edgecolor="none"))
                ax.text(*css_xy(bar_x + bar_css / 2, bar_y - 4), bar_text,
                        color="white", fontsize=16 * point,
                        fontfamily=font_family, ha="center", va="bottom",
                        path_effects=stroke)
                ax.text(*css_xy(12, css_h - 12 + 5), zoom_text,
                        color="white", fontsize=16 * point,
                        fontfamily=font_family, ha="left", va="bottom",
                        path_effects=stroke)
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor="white",
                    bbox_inches="tight", pad_inches=0.05)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    # _repr_mimebundle_ / _ipython_display_ / static-fallback sibling plumbing
    # comes from StaticFallbackMixin (utils/static_fallback.py); this class only
    # supplies _static_png_b64 above.

    def _store_static_fallback_preview(self, png_b64: str) -> None:
        """Store a compact saved-notebook preview inside lightweight state.

        JupyterLab can rehydrate a saved anywidget model even when
        ``save_state=False`` stripped the heavy ``frame_bytes``. In the common
        last-expression output path there is no separate static sibling output,
        so the frontend needs a small in-model preview to avoid mounting a blank
        widget after Cmd+S and reopen.
        """
        if getattr(self, "_save_state", False):
            return
        encoded = self._encode_static_fallback_b64(png_b64)
        if not encoded:
            self._static_fallback_jpeg = ""
            self._static_fallback_mime = ""
            return
        mime, image_b64 = encoded
        self._static_fallback_jpeg = image_b64
        self._static_fallback_mime = mime

    def state_dict(self):
        return {
            "title": self.title,
            "show_title": self.show_title,
            "cmap": self.cmap,
            "log_scale": self.log_scale,
            "auto_contrast": self.auto_contrast,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "labels": list(self.labels),
            "starred": list(self.starred),
            "n_pages": int(self.n_pages),
            "page_idx": int(self.page_idx),
            "panels_per_page": int(self.panels_per_page),
            "page_kind": str(self.page_kind),
            "folder_page_size": getattr(self, "_folder_page_size", None),
            "page_labels": list(self.page_labels),
            "page_starred": list(self.page_starred),
            "hidden_panels": list(self.hidden_panels),
            "hidden_page_slots": list(self.hidden_page_slots),
            "panel_order": list(self.panel_order),
            "panel_frame_indices": list(self.panel_frame_indices),
            "panel_playback_fps": float(self.panel_playback_fps),
            "show_panel_titles": self.show_panel_titles,
            "panel_title_font_size": self.panel_title_font_size,
            "show_stats": self.show_stats,
            "debug": self.debug,
            "show_fft": self.show_fft,
            "fft_window": self.fft_window,
            "fft_metrics": self.fft_metrics,
            "show_controls": self.show_controls,
            "controls_collapsed": self.controls_collapsed,
            "pixel_size": self.pixel_size,
            "pixel_sizes": list(self.pixel_sizes),
            "pixel_unit": self.pixel_unit,
            "scale_bar_visible": self.scale_bar_visible,
            "size": self.size,
            "smooth": self.smooth,
            "initial_zoom": self.initial_zoom,
            "vmins": self.vmins,
            "vmaxs": self.vmaxs,
            "link_zoom": self.link_zoom,
            "link_pan": self.link_pan,
            "link_contrast": self.link_contrast,
            "zoom_row": self.zoom_row,
            "zoom_col": self.zoom_col,
            "view_box": list(self.view_box),
            "view_crop": list(self.view_crop),
            "pad_ratio": float(self.pad_ratio),
            "diff_mode": self.diff_mode,
            # Which panel the signed-diff panel subtracts from; without it a
            # saved non-default diff reference silently reverts to panel 0.
            "diff_reference": int(self.diff_reference),
            "ncols": self.ncols,
            "selected_idx": self.selected_idx,
            "roi_active": self.roi_active,
            "roi_list": self.roi_list,
            "roi_selected_idx": self.roi_selected_idx,
            "profile_line": self.profile_line,
            "image_rotations": list(self.image_rotations),
            "display_bin": self._display_bin,
            "denoise": self.denoise,
            "denoise_sigma": self.denoise_sigma,
            "denoise_bin": self.denoise_bin,
            "denoise_scope": self.denoise_scope,
            "show_denoise": self.show_denoise,
            "denoise_modes": list(self.denoise_modes),
            "denoise_sigmas": list(self.denoise_sigmas),
            "denoise_bins": list(self.denoise_bins),
            # underlay= is construction-time sugar: the composed "map on HAADF"
            # RGB panel is rebuilt from the kernel, not stored, so it is NOT
            # restored on a kernel-less reopen. These blend knobs are kept so a
            # live (kernel-backed) widget re-blends to the saved look; on a cold
            # reopen they are inert (there is no underlay panel to tune).
            "underlay_alpha": self.underlay_alpha,
            "underlay_haadf_gain": self.underlay_haadf_gain,
            "underlay_mode": self.underlay_mode,
            "stretch_percentiles": list(self.stretch_percentiles),
            "display_gamma": self.display_gamma,
            "dual_gain": list(self.dual_gain),
        }

    def save(self, path: str):
        save_state_file(path, "Show2D", self.state_dict())

    def export_html(self, path: str | pathlib.Path | None = None,
                    *,
                    title: str | None = None,
                    mode: str = "single",
                    encoding: str = "full",
                    downsample: int | None = None,
                    quantized: bool | None = None) -> pathlib.Path:
        """Write a standalone HTML viewer for this widget.

        The exported file mounts the live anywidget JS bundle with the current
        widget state (data, labels, cmap, vmin/vmax, log_scale, sampling, ...).
        Opens in any browser without a Jupyter kernel.
        Preferred export options are ``mode="single"``, ``encoding="full"`` or
        ``encoding="uint8"``, and ``downsample=None``. ``quantized`` is kept as
        a compatibility alias for ``encoding="uint8"``.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            Destination HTML path.
        quantized : bool, default False
            Store the displayed image stack as uint8 with min/max metadata.
            This is smaller and visually equivalent after colormapping. The
            default stores exact float32 display values.
        title : str, optional
            Browser page title. Defaults to widget ``title`` or "Show2D".
        """
        if self._data is None:
            raise ValueError("Cannot export HTML after free(); rebuild the widget first.")
        if any(self.is_rgb):
            raise NotImplementedError(
                "export_html is not supported when the gallery contains RGB panels; "
                "the export clone rebuilds from the grayscale stack and would drop the color channels."
            )

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
            raise ValueError("Show2D HTML export supports mode='single'")
        if downsample not in (None, 1, "1", "", 0, "0"):
            raise NotImplementedError("Show2D HTML export does not support downsample yet")
        raw_encoding = str(encoding or "full").strip().lower().replace("_", "-")
        if quantized is True:
            raw_encoding = "uint8"
        elif quantized is False and raw_encoding in {"quantized", "uint8", "u8"}:
            raw_encoding = "uint8"
        if raw_encoding in {"full", "exact", "float32", "f32"}:
            return False
        if raw_encoding in {"uint8", "u8", "quantized"}:
            return True
        raise ValueError(f"unknown Show2D export encoding {encoding!r}; expected 'full' or 'uint8'")

    def _on_export_request_change(self, change: dict) -> None:
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

    def _default_html_export_path(self, quantized: bool) -> pathlib.Path:
        label = self.title.strip() or "show2d"
        slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        if not slug:
            slug = "show2d"
        mode = "quantized" if quantized else "exact"
        shape = f"{self.n_images}x{self.height}x{self.width}" if self.n_images > 1 else f"{self.height}x{self.width}"
        return pathlib.Path.cwd() / f"{slug}_{shape}_{mode}.html"

    def _export_mode_label(self, quantized: bool) -> str:
        return "uint8" if quantized else "full float32"

    def _write_html_export(
        self,
        path: str | pathlib.Path,
        *,
        quantized: bool,
        title: str | None = None,
    ) -> pathlib.Path:
        from ipywidgets.embed import dependency_state, embed_minimal_html

        from .export import ensure_mobile_viewport

        export_path = pathlib.Path(path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        page_title = title or self.title or "Show2D"
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
            export_widget.close()
        ensure_mobile_viewport(export_path)
        return export_path

    def _html_export_bytes(self, *, quantized: bool) -> bytes:
        with tempfile.TemporaryDirectory(prefix="show2d-export-") as tmp:
            path = pathlib.Path(tmp) / self._default_html_export_path(quantized).name
            self._write_html_export(path, quantized=quantized)
            return path.read_bytes()

    def _clone_for_html_export(self, *, quantized: bool) -> Self:
        if any(self.is_rgb):
            raise NotImplementedError(
                "HTML export is not supported when the gallery contains RGB panels; "
                "the export clone rebuilds from the grayscale stack and would drop the color channels."
            )
        data = self._display_data if self._display_data is not None else self._data
        if data is None:
            raise ValueError("Cannot export HTML after free(); rebuild the widget first.")
        has_local_stacks = any(count > 1 for count in self.panel_frame_counts)
        if has_local_stacks:
            display_stacks = getattr(self, "_display_panel_stacks", None)
            if not display_stacks:
                raise ValueError("Cannot export local panel stacks after their data has been freed")
            export_data = [
                np.ascontiguousarray(stack if stack.shape[0] > 1 else stack[0], dtype=np.float32)
                for stack in display_stacks
            ]
        else:
            export_data = np.ascontiguousarray(data, dtype=np.float32)
        clone = type(self)(
            export_data,
            labels=list(self.labels),
            page_labels=list(self.page_labels) if self.n_pages > 1 else None,
            title=self.title,
            show_title=self.show_title,
            cmap=self.cmap,
            sampling=self.pixel_size if self.pixel_size > 0 else None,
            units=self.pixel_unit,
            scale_bar_visible=self.scale_bar_visible,
            show_fft=self.show_fft,
            fft_window=self.fft_window,
            fft_metrics=self.fft_metrics,
            show_controls=self.show_controls,
            controls_collapsed=self.controls_collapsed,
            show_stats=self.show_stats,
            debug=self.debug,
            verbose=False,
            log_scale=self.log_scale,
            auto_contrast=self.auto_contrast,
            offline=quantized,
            vmin=self.vmin if self.vmin is not None else self.vmins,
            vmax=self.vmax if self.vmax is not None else self.vmaxs,
            ncols=self.ncols,
            panel_frame_indices=list(self.panel_frame_indices),
            panel_playback_fps=self.panel_playback_fps,
            size=self.size,
            smooth=self.smooth,
            zoom=self.initial_zoom,
            zoom_row=self.zoom_row,
            zoom_col=self.zoom_col,
            link_zoom=self.link_zoom,
            link_pan=self.link_pan,
            link_contrast=self.link_contrast,
            diff_mode=self.diff_mode,
            hidden_panels=list(self.hidden_panels),
            starred=[i for i, value in enumerate(self.starred) if value],
            panel_order=list(self.panel_order),
            show_panel_titles=self.show_panel_titles,
            panel_title_font_size=self.panel_title_font_size,
            display_bin=1,
        )
        clone.pixel_sizes = list(self.pixel_sizes)
        clone.page_kind = str(self.page_kind)
        clone.n_pages = int(self.n_pages)
        clone.panels_per_page = int(self.panels_per_page)
        clone.page_labels = list(self.page_labels)
        clone.page_idx = int(self.page_idx)
        clone.page_starred = list(self.page_starred)
        clone.load_state_dict(self.state_dict())
        clone.offline = quantized
        clone._export_light = True
        clone._save_state = True
        clone.export_enabled = False
        clone.export_status = ""
        clone.export_payload = b""
        clone.export_payload_id = ""
        clone.export_filename = ""
        clone.handoff_enabled = False
        clone.handoff_status = ""
        clone.handoff_request = ""
        clone.prepared_view = None
        clone.prepared_view_widget = None
        # Exported pages ship RAW frames plus the filter knobs: the browser
        # port (WGSL compute, CPU fallback without WebGPU) filters client-side
        # so sigma scrubs live with no kernel. Panels on non portable modes
        # (tv, denova*) still bake their Python-filtered pixels because
        # _panel_browser_filtered is per panel.
        clone._webgpu_filter_ok = True
        clone._update_all_frames()
        return clone

    def load_state_dict(self, state):
        state = dict(state)
        page_kind = str(state.get("page_kind", self.page_kind))
        if page_kind not in {"comparison", "items"}:
            page_kind = str(self.page_kind)
            state.pop("page_kind", None)
        try:
            saved_n_pages = max(1, int(state.get("n_pages", self.n_pages)))
            saved_panels_per_page = max(
                0,
                int(state.get("panels_per_page", self.panels_per_page)),
            )
        except (TypeError, ValueError):
            saved_n_pages = int(self.n_pages)
            saved_panels_per_page = int(self.panels_per_page)
            state.pop("n_pages", None)
            state.pop("panels_per_page", None)
        n_images = int(self.n_images)
        valid_pages = saved_n_pages == 1 and saved_panels_per_page == 0
        if saved_n_pages > 1 and saved_panels_per_page > 0:
            if page_kind == "items":
                valid_pages = (
                    saved_n_pages == math.ceil(n_images / saved_panels_per_page)
                )
            else:
                valid_pages = n_images == saved_n_pages * saved_panels_per_page
        if valid_pages:
            # Page index, labels, and hidden-state normalization below all need
            # the saved layout installed before they are validated.
            self.page_kind = page_kind
            self.n_pages = saved_n_pages
            self.panels_per_page = saved_panels_per_page
        else:
            state.pop("n_pages", None)
            state.pop("panels_per_page", None)
            state.pop("page_kind", None)
        folder_page_size = state.pop("folder_page_size", None)
        if page_kind == "items" and hasattr(self, "_folder_source"):
            try:
                self._folder_page_size = _validate_folder_page_size(
                    folder_page_size,
                )
            except (TypeError, ValueError):
                pass
        # A crop saved from a single-panel session cannot apply to a gallery.
        if int(self.n_images) != 1:
            state.pop("view_crop", None)
        if "page_idx" in state:
            try:
                state["page_idx"] = int(max(0, min(int(state["page_idx"]), int(self.n_pages) - 1)))
            except (TypeError, ValueError):
                state.pop("page_idx")
        if "page_starred" in state and isinstance(state["page_starred"], list):
            if len(state["page_starred"]) != int(self.n_pages):
                state.pop("page_starred")
        if "page_labels" in state and isinstance(state["page_labels"], list):
            if len(state["page_labels"]) != int(self.n_pages):
                state.pop("page_labels")
        if "starred" in state and isinstance(state["starred"], list) and len(state["starred"]) != int(self.n_images):
            state.pop("starred")
        if "hidden_panels" in state and isinstance(state["hidden_panels"], list):
            n_img = int(self.n_images)
            clean_set: set[int] = set()
            for value in state["hidden_panels"]:
                if isinstance(value, bool):
                    continue
                try:
                    idx = int(value)
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < n_img:
                    clean_set.add(idx)
            clean = sorted(clean_set)
            if len(clean) >= n_img:
                clean = clean[:-1]
            state["hidden_panels"] = self._normalize_item_page_hidden(
                clean,
                drop_if_full=True,
            )
        if "hidden_page_slots" in state and isinstance(state["hidden_page_slots"], list):
            state["hidden_page_slots"] = self._normalize_hidden_page_slots(
                state["hidden_page_slots"],
                drop_if_full=True,
            )
        elif int(self.n_pages) > 1 and "hidden_panels" in state and isinstance(state["hidden_panels"], list):
            state["hidden_page_slots"] = self._hidden_page_slots_from_panels(
                state["hidden_panels"],
                drop_if_full=True,
            )
        if "panel_order" in state and isinstance(state["panel_order"], list):
            try:
                order = [int(value) for value in state["panel_order"] if not isinstance(value, bool)]
            except (TypeError, ValueError):
                order = []
            if len(order) != int(self.n_images) or sorted(order) != list(range(int(self.n_images))):
                state.pop("panel_order")
            else:
                state["panel_order"] = order
        if "panel_frame_indices" in state and isinstance(state["panel_frame_indices"], list):
            counts = list(self.panel_frame_counts)
            raw_indices = state["panel_frame_indices"]
            try:
                indices = [
                    int(value) if not isinstance(value, bool) else -1
                    for value in raw_indices
                ]
            except (TypeError, ValueError):
                indices = []
            if len(indices) != int(self.n_images) or any(
                value < 0 or value >= max(1, int(counts[panel]))
                for panel, value in enumerate(indices)
            ):
                state.pop("panel_frame_indices")
            else:
                state["panel_frame_indices"] = indices
        for key, val in state.items():
            # Silent migrations for renamed keys in older saved state files.
            if key == "pixel_size_angstrom":
                key = "pixel_size"
            elif key == "canvas_size":
                key = "size"
            elif key in _DENOISE_STATE_ALIASES:
                key = _DENOISE_STATE_ALIASES[key]
            elif key == "filter_per_panel":
                key, val = "denoise_scope", ("all" if val else "panel")
            if key == "display_bin":
                self._display_bin = val
                continue
            if hasattr(self, key):
                setattr(self, key, val)

    @property
    def current_view(self) -> dict:
        """The field of view currently on screen, in image and calibrated coordinates.

        Every pan/zoom in the browser syncs the visible region back to Python
        (debounced ~100 ms), so after zooming into a feature you can capture
        exactly where you are for figure-making: ``view = w.current_view``.
        Feed ``view["box"]`` back as ``Show2D(data, view_box=view["box"])`` to
        reproduce the same crop later. The box is axis-aligned, so the two
        corners (row0, col0) and (row1, col1) define all four.

        Returns a dict with ``row0/col0/row1/col1`` (image pixel coordinates),
        ``height/width`` (visible extent in pixels), ``zoom``, ``box`` (the
        (row0, row1, col0, col1) tuple accepted by ``view_box=``), and when
        ``pixel_size > 0`` the calibrated ``*_cal`` extents plus ``unit``.
        """
        if len(self.view_box) == 4:
            row0, row1, col0, col1 = (float(v) for v in self.view_box)
        else:
            # No pan/zoom synced from the browser yet: derive the viewport
            # from the construction-time zoom center so the property is
            # usable immediately after Show2D(data, zoom=..., zoom_row=...).
            zoom = float(self.initial_zoom) if self.initial_zoom else 1.0
            center_row = float(self.zoom_row) if self.zoom_row is not None else self.height / 2
            center_col = float(self.zoom_col) if self.zoom_col is not None else self.width / 2
            half_h, half_w = self.height / (2 * zoom), self.width / (2 * zoom)
            row0, row1 = max(0.0, center_row - half_h), min(float(self.height), center_row + half_h)
            col0, col1 = max(0.0, center_col - half_w), min(float(self.width), center_col + half_w)
        view = {"row0": row0, "col0": col0, "row1": row1, "col1": col1,
                "height": row1 - row0, "width": col1 - col0,
                "zoom": float(self.initial_zoom),
                "box": (row0, row1, col0, col1)}
        if self.pixel_size > 0:
            scale = self.pixel_size
            view.update({"row0_cal": row0 * scale, "col0_cal": col0 * scale,
                         "row1_cal": row1 * scale, "col1_cal": col1 * scale,
                         "height_cal": (row1 - row0) * scale, "width_cal": (col1 - col0) * scale,
                         "unit": self.pixel_unit})
        return view

    def crop_to_view(self) -> Self:
        """Commit the current browser viewport as the display extent.

        Zoom into a feature (mouse wheel, or ``view_box=`` at construction),
        then call this to make that window the whole displayed frame: the
        widget repacks only the committed region, so an active denoise
        operates on the cropped view. Display-only and reversible: the
        stored array is never modified, the stats row keeps reporting the
        full raw data, and cursor coordinates stay full-image (row, col).
        :meth:`reset_view_ops` restores the full frame bit-identically.

        Returns
        -------
        Self
            The widget, for chaining.

        Raises
        ------
        NotImplementedError
            For galleries (crop-to-view is single panel only in this
            release) and for RGB panels.

        Examples
        --------
        >>> w = Show2D(image, view_box=(64, 64, 96))  # zoom into a feature
        >>> w.crop_to_view()      # the 96x96 window becomes the display extent
        >>> w.reset_view_ops()    # back to the full frame, bit-identical
        """
        if int(self.n_images) != 1:
            raise NotImplementedError(
                f"crop_to_view() supports a single panel; this widget shows "
                f"{int(self.n_images)} panels. Crop the arrays before display for galleries."
            )
        if any(self.is_rgb):
            raise NotImplementedError(
                "crop_to_view() supports grayscale panels; this panel is RGB."
            )
        view = self.current_view
        factor = max(1, int(self._display_bin_factor))
        offset_row, offset_col = (int(v) for v in self._view_crop_offset)
        self.view_crop = [
            int(math.floor(view["row0"])) * factor + offset_row,
            int(math.ceil(view["row1"])) * factor + offset_row,
            int(math.floor(view["col0"])) * factor + offset_col,
            int(math.ceil(view["col1"])) * factor + offset_col,
        ]
        return self

    def reset_view_ops(self) -> Self:
        """Restore the uncropped, unpadded display.

        Undoes :meth:`crop_to_view` and ``pad_ratio`` in one call. Both ops
        are display-only view transforms, so resetting returns the exact
        original frame bytes; the stored data was never touched.

        Returns
        -------
        Self
            The widget, for chaining.

        Examples
        --------
        >>> w = Show2D(image, pad_ratio=0.1)
        >>> w.reset_view_ops()  # full frame again, no border
        """
        with self.hold_sync():
            self.view_crop = []
            self.pad_ratio = 0.0
        return self

    def set_denoise(
        self,
        mode: str,
        *,
        sigma: float | None = None,
        bin: int | None = None,
        panels: Sequence[int] | int | None = None,
    ) -> Self:
        """Set the display denoise for every panel, or a chosen panel subset.

        The imperative twin of the ``denoise=`` constructor kwarg, chainable
        like :meth:`crop_to_view` and :meth:`set_roi`. Denoise is a pure view
        transform: the stored array, the stats row, and every raw-data export
        keep the original counts, and an active reduction still announces
        itself with the one-line banner. The per-panel ``denoise_modes`` /
        ``denoise_sigmas`` / ``denoise_bins`` lists are the source of truth;
        the scalar ``denoise`` / ``denoise_sigma`` / ``denoise_bin`` traits
        mirror only the selected panel while ``denoise_scope == "panel"``.

        Parameters
        ----------
        mode : str
            Denoise method for the targeted panels: ``"none"``, ``"gaussian"``,
            or ``"anscombe"``. The compound spellings ``"bin2"``,
            ``"bin2_anscombe"`` and ``"bin4_anscombe"`` fold into a (mode, bin)
            pair; ``"tv"``/``"denova*"`` stay available from Python.
        sigma : float or None, optional
            Smoothing scale in pixels. ``None`` leaves each targeted panel's
            current sigma unchanged.
        bin : int or None, optional
            Display-side SNR bin factor (1, 2, or 4). ``None`` leaves each
            targeted panel's current bin unchanged, unless a compound ``mode``
            (e.g. ``"bin2_anscombe"``) sets it.
        panels : int, sequence of int, or None, optional
            Panel indices to change. ``None`` (the default) targets every
            panel and applies one uniform setting; passing a subset edits only
            those panels and switches ``denoise_scope`` to ``"panel"`` so later
            UI edits stay per panel.

        Returns
        -------
        Self
            The widget, for chaining.

        Raises
        ------
        ValueError
            When a ``panels`` index is out of range for the widget.

        Examples
        --------
        >>> import numpy as np
        >>> from quantem.widget import Show2D
        >>> a = np.random.poisson(0.3, (256, 256)).astype("float32")
        >>> b = np.random.poisson(0.3, (256, 256)).astype("float32")
        >>> w = Show2D([a, b])                             # a raw A/B gallery
        >>> _ = w.set_denoise("anscombe", sigma=8, bin=2)  # denoise both panels
        >>> _ = w.set_denoise("none", panels=[0])          # keep panel 0 raw
        """
        from quantem.widget.utils.display_filter import resolve_denoise_mode

        n_panels = int(self.n_images)
        if panels is None:
            targets = list(range(n_panels))
            subset = False
        else:
            if isinstance(panels, (int, np.integer)):
                panels = [int(panels)]
            targets = [int(p) for p in panels]
            for p in targets:
                if not (0 <= p < n_panels):
                    raise ValueError(
                        f"set_denoise panels index {p} is out of range for a "
                        f"{n_panels}-panel widget (valid 0..{n_panels - 1})."
                    )
            subset = True
        resolved_mode, resolved_bin = resolve_denoise_mode(
            str(mode), 1 if bin is None else int(bin)
        )
        modes = list(self.denoise_modes) or [str(self.denoise)] * n_panels
        sigmas = list(self.denoise_sigmas) or [float(self.denoise_sigma)] * n_panels
        bins = list(self.denoise_bins) or [int(self.denoise_bin)] * n_panels
        for p in targets:
            modes[p] = resolved_mode
            if sigma is not None:
                sigmas[p] = float(sigma)
            if bin is not None or resolved_bin > 1:
                bins[p] = int(resolved_bin)
        # Write the per-panel lists in one batch: suppress the per-list repack
        # observer and the scalar mirror during the write, then repack once
        # (mirrors the constructor). The scalar knobs re-sync to the selected
        # panel so the editor keeps showing what is on screen.
        with self.hold_sync():
            prev_ready = getattr(self, "_display_filter_ready", False)
            self._display_filter_ready = False
            self._filter_knob_sync = True
            try:
                if subset:
                    self.denoise_scope = "panel"
                self.denoise_modes = modes
                self.denoise_sigmas = sigmas
                self.denoise_bins = bins
                sel = min(int(self.selected_idx), max(0, n_panels - 1))
                self.denoise = modes[sel]
                self.denoise_sigma = float(sigmas[sel])
                self.denoise_bin = int(bins[sel])
            finally:
                self._display_filter_ready = prev_ready
                self._filter_knob_sync = False
            if prev_ready:
                if self._display_filter_active():
                    self.show_denoise = True
                self._refresh_display_filter_banner(announce=True)
                self._update_all_frames()
        return self

    def to_show3d(
        self,
        panels: Sequence[int | str] | int | str | None = None,
        *,
        title: str | None = None,
        copy: bool = True,
        include_hidden: bool = False,
    ):
        """Create a ``Show3D`` stack from this Show2D image gallery.

        The default uses currently visible panels as frames, preserving labels,
        colormap, sampling, scale-bar units, and contrast settings. Use
        ``include_hidden=True`` when hidden panels should also become frames.
        """
        from quantem.widget.show3d import Show3D

        if any(self.is_rgb):
            raise ValueError("Show2D.to_show3d() supports grayscale panels; RGB panels cannot become Show3D frames yet")
        if panels is None:
            panel_indices = list(range(int(self.n_images))) if include_hidden else self.visible_panels
        else:
            panel_indices = self._normalize_panel_refs(panels)
            if not include_hidden:
                hidden = set(int(i) for i in self.hidden_panels)
                panel_indices = [idx for idx in panel_indices if idx not in hidden]
        if not panel_indices:
            raise ValueError("Show2D.to_show3d() needs at least one visible panel")

        frames = [np.asarray(self._data[idx], dtype=np.float32) for idx in panel_indices]
        shapes = {frame.shape for frame in frames}
        if len(shapes) != 1:
            raise ValueError(
                "Show2D.to_show3d() requires selected panels to have the same shape; "
                f"got {sorted(shapes)!r}"
            )
        stack = np.stack(frames, axis=0)
        if copy:
            stack = np.array(stack, copy=True)

        vmin = self.vmin
        vmax = self.vmax
        if self.link_contrast and self.vmins and len(self.vmins) == int(self.n_images):
            chosen_vmins = [self.vmins[idx] for idx in panel_indices]
            chosen_vmaxs = [self.vmaxs[idx] for idx in panel_indices]
            if len({float(v) for v in chosen_vmins}) == 1:
                vmin = float(chosen_vmins[0])
            if len({float(v) for v in chosen_vmaxs}) == 1:
                vmax = float(chosen_vmaxs[0])

        return Show3D(
            stack,
            labels=[self._panel_title_for_index(idx) for idx in panel_indices],
            panel_titles=[title or self.title or "Show2D"],
            title=title if title is not None else self.title,
            show_title=self.show_title,
            cmap=self.cmap,
            vmin=vmin,
            vmax=vmax,
            sampling=self.pixel_size if self.pixel_size > 0 else None,
            units=self.pixel_unit,
            smooth=self.smooth,
            log_scale=self.log_scale,
            auto_contrast=self.auto_contrast,
            show_fft=False,
            show_stats=self.show_stats,
            show_controls=self.show_controls,
            controls_collapsed=self.controls_collapsed,
            size=int(self.size or 0),
            panel_title_font_size=int(self.panel_title_font_size or 11),
            show_panel_titles=self.show_panel_titles,
            show_scale_bar=self.scale_bar_visible,
            save_state=False,
            verbose=False,
        )

    def _on_handoff_request_change(self, change: dict) -> None:
        """Build a Python-side prepared view from a frontend toolbar request."""
        raw = change.get("new") or ""
        if not raw:
            return
        try:
            request = json.loads(raw)
            mode = str(request.get("mode", "show3d")).lower()
            if mode == "clear":
                self.prepared_view = None
                self.prepared_view_widget = None
                self.handoff_status = ""
                return
            if mode != "show3d":
                raise ValueError(f"unsupported handoff mode {mode!r}")
            self.prepared_view = self.to_show3d(panels=request.get("panels", None))
            self.prepared_view_widget = self.prepared_view
            n_frames = int(getattr(self.prepared_view, "n_slices", 0))
            self.handoff_status = f"Showing 3D with {n_frames} frame{'s' if n_frames != 1 else ''}"
        except Exception as exc:  # pragma: no cover - defensive comm boundary
            self.prepared_view = None
            self.prepared_view_widget = None
            self.handoff_status = f"View failed: {exc}"

    @property
    def ordered_panels(self) -> list[int]:
        """Zero-based image panel indices in the current display order."""
        if int(self.n_pages) > 1 and int(self.panels_per_page) > 0:
            start = int(self.page_idx) * int(self.panels_per_page)
            stop = min(start + int(self.panels_per_page), int(self.n_images))
            if str(self.page_kind) == "items":
                return self._folder_ordered_panel_indices()[start:stop]
            return list(range(start, stop))
        order = list(self.panel_order or [])
        if len(order) == int(self.n_images) and sorted(order) == list(range(int(self.n_images))):
            return order
        return list(range(int(self.n_images)))

    @property
    def visible_panels(self) -> list[int]:
        """Zero-based image panel indices currently visible in the gallery."""
        if int(self.n_pages) > 1 and int(self.panels_per_page) > 0:
            if str(self.page_kind) == "items":
                hidden = set(self.hidden_panels)
                return [panel for panel in self.ordered_panels if panel not in hidden]
            hidden_slots = set(
                self.hidden_page_slots
                or self._hidden_page_slots_from_panels(self.hidden_panels, drop_if_full=True)
            )
            per_page = int(self.panels_per_page)
            return [i for i in self.ordered_panels if (i % per_page) not in hidden_slots]
        hidden = set(self.hidden_panels)
        return [i for i in self.ordered_panels if i not in hidden]

    @property
    def starred_panels(self) -> list[int]:
        """Zero-based image panel indices marked with a star."""
        return [i for i, value in enumerate(self.starred) if value]

    @property
    def starred_pages(self) -> list[int]:
        """Zero-based page indices marked with a star."""
        return [i for i, value in enumerate(self.page_starred) if value]

    def set_hidden_panels(self, panels: Sequence[int | str] | int | str) -> Self:
        """Replace the hidden panel set by index or exact label.

        Hidden panels remain in the widget state and standalone HTML export, but
        they are collapsed from the gallery until restored. At least one panel
        must stay visible.
        """
        hidden = self._normalize_panel_refs(panels, allow_empty=True)
        if int(self.n_pages) > 1 and str(self.page_kind) == "items":
            self._validate_folder_item_pages_visible(hidden, "set_hidden_panels")
        if int(self.n_pages) > 1 and str(self.page_kind) != "items":
            try:
                hidden_page_slots = self._hidden_page_slots_from_panels(hidden)
            except traitlets.TraitError as exc:
                raise ValueError(
                    "set_hidden_panels would hide every page slot; leave at least one visible"
                ) from exc
        else:
            hidden_page_slots = []
            if int(self.n_pages) > 1 and str(self.page_kind) == "items":
                try:
                    hidden = self._normalize_item_page_hidden(hidden)
                except traitlets.TraitError as exc:
                    raise ValueError(str(exc)) from exc
        if len(hidden) >= int(self.n_images):
            raise ValueError("set_hidden_panels would hide every panel; leave at least one visible")
        self.hidden_panels = sorted(hidden)
        self.hidden_page_slots = hidden_page_slots
        return self

    def hide_panel(self, *panels: int | str) -> Self:
        """Hide one or more image panels by zero-based index or exact label."""
        to_hide = set(self.hidden_panels)
        to_hide.update(self._normalize_panel_refs(list(panels)))
        if int(self.n_pages) > 1 and str(self.page_kind) == "items":
            self._validate_folder_item_pages_visible(to_hide, "hide_panel")
        if int(self.n_pages) > 1 and str(self.page_kind) != "items":
            try:
                hidden_page_slots = self._hidden_page_slots_from_panels(sorted(to_hide))
            except traitlets.TraitError as exc:
                raise ValueError("hide_panel would hide every page slot; leave at least one visible") from exc
        else:
            hidden_page_slots = []
            if int(self.n_pages) > 1 and str(self.page_kind) == "items":
                try:
                    to_hide = set(self._normalize_item_page_hidden(sorted(to_hide)))
                except traitlets.TraitError as exc:
                    raise ValueError(str(exc)) from exc
        if len(to_hide) >= int(self.n_images):
            raise ValueError("hide_panel would hide every panel; leave at least one visible")
        self.hidden_panels = sorted(to_hide)
        self.hidden_page_slots = hidden_page_slots
        return self

    def show_panel(self, *panels: int | str) -> Self:
        """Restore one or more hidden image panels by zero-based index or exact label."""
        to_show = set(self._normalize_panel_refs(list(panels)))
        hidden = sorted(set(self.hidden_panels) - to_show)
        self.hidden_panels = hidden
        self.hidden_page_slots = self._hidden_page_slots_from_panels(hidden)
        return self

    def show_all_panels(self) -> Self:
        """Restore every image panel in the gallery."""
        self.hidden_panels = []
        self.hidden_page_slots = []
        return self

    def set_panel_order(self, panels: Sequence[int | str]) -> Self:
        """Set the gallery display order by panel index or exact label.

        The order is display-only: source data, labels, hidden state, stars, and
        pixel sizes stay keyed by their original panel indices.
        """
        order = self._normalize_panel_refs(panels, allow_empty=True)
        if not order:
            self.panel_order = []
            return self
        expected = set(range(int(self.n_images)))
        if len(order) != int(self.n_images) or set(order) != expected:
            raise ValueError("set_panel_order requires every panel exactly once")
        self.panel_order = order
        return self

    def reset_panel_order(self) -> Self:
        """Restore the natural source-image order."""
        self.panel_order = []
        return self

    def move_panel(self, panel: int | str, position: int) -> Self:
        """Move one panel to a zero-based display position."""
        idx = self._resolve_panel_ref(panel)
        order = (
            self._folder_ordered_panel_indices()
            if int(self.n_pages) > 1 and str(self.page_kind) == "items"
            else self.ordered_panels
        )
        order.remove(idx)
        pos = int(position)
        if pos < 0:
            pos = 0
        if pos > len(order):
            pos = len(order)
        order.insert(pos, idx)
        self.panel_order = order
        return self

    def set_panel_frame(self, panel: int | str, frame: int) -> Self:
        """Set the displayed frame for one local stack panel.

        Static panels have one frame and therefore only accept frame ``0``.
        Negative indices follow normal Python indexing, so ``-1`` selects the
        final frame (useful for Velox/EDS acquisitions whose exported HAADF is
        the last survey frame).
        """
        panel_idx = self._resolve_panel_ref(panel)
        count = int(self.panel_frame_counts[panel_idx])
        frame_idx = int(frame)
        if frame_idx < 0:
            frame_idx += count
        if frame_idx < 0 or frame_idx >= count:
            if count == 1:
                raise IndexError(
                    f"panel {panel_idx} is static and has one frame; only frame 0 is valid"
                )
            raise IndexError(
                f"frame index {frame} out of range for panel {panel_idx} with {count} frame(s)"
            )
        indices = list(self.panel_frame_indices)
        indices[panel_idx] = frame_idx
        self.panel_frame_indices = indices
        return self

    def collapse_controls(self) -> Self:
        """Collapse the live control UI while leaving the GUI toggle available."""
        self.controls_collapsed = True
        return self

    def expand_controls(self) -> Self:
        """Expand the live control UI."""
        self.controls_collapsed = False
        return self

    def toggle_controls(self) -> Self:
        """Toggle the collapsed state of the live control UI."""
        self.controls_collapsed = not bool(self.controls_collapsed)
        return self

    def set_starred_panels(self, panels: Sequence[int | str] | int | str) -> Self:
        """Replace the set of starred image panels by index or exact label."""
        starred = [0] * int(self.n_images)
        for idx in self._normalize_panel_refs(panels, allow_empty=True):
            starred[idx] = 1
        self.starred = starred
        return self

    def star_panel(self, panel: int | str) -> Self:
        """Mark an image panel with a star."""
        idx = self._resolve_panel_ref(panel)
        starred = list(self.starred)
        if len(starred) != int(self.n_images):
            starred = [0] * int(self.n_images)
        starred[idx] = 1
        self.starred = starred
        return self

    def unstar_panel(self, panel: int | str) -> Self:
        """Clear the star on an image panel."""
        idx = self._resolve_panel_ref(panel)
        starred = list(self.starred)
        if len(starred) != int(self.n_images):
            starred = [0] * int(self.n_images)
        starred[idx] = 0
        self.starred = starred
        return self

    def star_page(self, page: int) -> Self:
        """Mark a page with a star."""
        idx = int(page)
        if idx < 0 or idx >= int(self.n_pages):
            raise ValueError(f"page index {idx} out of range [0, {self.n_pages})")
        starred = list(self.page_starred)
        if len(starred) != int(self.n_pages):
            starred = [0] * int(self.n_pages)
        starred[idx] = 1
        self.page_starred = starred
        return self

    def unstar_page(self, page: int) -> Self:
        """Clear the star on a page."""
        idx = int(page)
        if idx < 0 or idx >= int(self.n_pages):
            raise ValueError(f"page index {idx} out of range [0, {self.n_pages})")
        starred = list(self.page_starred)
        if len(starred) != int(self.n_pages):
            starred = [0] * int(self.n_pages)
        starred[idx] = 0
        self.page_starred = starred
        return self

    def summary(self):
        """Print a human-readable snapshot of the widget's current state.

        Reports image dimensions and pixel size, data min/max/mean, display
        settings (colormap, contrast, scale, FFT), active ROIs and profile
        line, per-image rotations, and the most recent render timings.
        """
        lines = [self.title or "Show2D", "═" * 32]
        if self.n_images > 1:
            lines.append(f"Image:    {self.n_images}×{self.height}×{self.width} ({self.ncols} cols)")
        else:
            lines.append(f"Image:    {self.height}×{self.width}")
        if self.pixel_size > 0:
            ps = self.pixel_size
            if ps >= 10:
                lines[-1] += f" ({ps / 10:.2f} nm/px)"
            else:
                lines[-1] += f" ({ps:.2f} Å/px)"
        if hasattr(self, "_data") and self._data is not None:
            arr = self._data
            lines.append(f"Data:     min={float(arr.min()):.4g}  max={float(arr.max()):.4g}  mean={float(arr.mean()):.4g}")
        cmap = self.cmap
        scale = "log" if self.log_scale else "linear"
        if self.vmin is not None and self.vmax is not None:
            contrast = f"vmin={self.vmin:.4g}, vmax={self.vmax:.4g}"
        elif self.auto_contrast:
            contrast = "auto contrast"
        else:
            contrast = "manual contrast"
        display = f"{cmap} | {contrast} | {scale}"
        if self.show_fft:
            display += " | FFT"
            if not self.fft_window:
                display += " (no window)"
        lines.append(f"Display:  {display}")
        if self.roi_active and self.roi_list:
            lines.append(f"ROI:      {len(self.roi_list)} region(s)")
        if self.profile_line:
            p0, p1 = self.profile_line[0], self.profile_line[1]
            lines.append(f"Profile:  ({p0['row']:.0f}, {p0['col']:.0f}) → ({p1['row']:.0f}, {p1['col']:.0f})")
        non_zero = [(i, r * 90) for i, r in enumerate(self.image_rotations) if r % 4 != 0]
        if non_zero:
            parts = [f"#{i}={deg}°" for i, deg in non_zero]
            lines.append(f"Rotated:  {', '.join(parts)}")
        rt = getattr(self, "render_total_ms", None)
        if rt is not None:
            pb = getattr(self, "render_python_build_ms", 0)
            wj = getattr(self, "render_wire_js_ms", 0)
            lines.append(f"Rendered: {rt} ms total (Python build {pb} ms, wire+JS {wj} ms)")
        else:
            lines.append("Rendered: (pending first browser paint)")
        print("\n".join(lines))

    def _compute_all_stats(self):
        """Compute statistics for all images (vectorized over all frames)."""
        # Vectorized reduction over (H, W) is faster than per-image loops
        # for large galleries (e.g. 12×4096×4096: 164ms vs 191ms).
        axes = (1, 2) if self._data.ndim == 3 else None
        self.stats_mean = np.mean(self._data, axis=axes).ravel().tolist()
        self.stats_min = np.min(self._data, axis=axes).ravel().tolist()
        self.stats_max = np.max(self._data, axis=axes).ravel().tolist()
        self.stats_std = np.std(self._data, axis=axes).ravel().tolist()

    @traitlets.validate("denoise")
    def _validate_display_filter(self, proposal: dict) -> str:
        """Normalize and reject unknown display filter modes early."""
        from quantem.widget.utils.display_filter import DISPLAY_FILTER_MODES, _normalize_mode

        mode = _normalize_mode(proposal["value"])
        if mode != "none" and mode not in DISPLAY_FILTER_MODES:
            raise traitlets.TraitError(
                "denoise must be one of "
                + "|".join(DISPLAY_FILTER_MODES)
                + f" (or 'off'/'raw'); got {proposal['value']!r}"
            )
        return mode

    @traitlets.validate("denoise_scope")
    def _validate_denoise_scope(self, proposal: dict) -> str:
        value = str(proposal["value"]).strip().lower()
        if value not in ("all", "panel"):
            raise traitlets.TraitError(
                f"denoise_scope must be 'all' or 'panel'; got {proposal['value']!r}"
            )
        return value

    @traitlets.validate("denoise_bin")
    def _validate_spatial_bin(self, proposal: dict) -> int:
        value = int(proposal["value"])
        if value not in (1, 2, 4):
            raise traitlets.TraitError(f"denoise_bin must be 1, 2, or 4; got {value}")
        return value

    @traitlets.validate("underlay_mode")
    def _validate_underlay_mode(self, proposal: dict) -> str:
        value = str(proposal["value"]).strip().lower()
        if value not in ("haadf", "dual"):
            raise traitlets.TraitError(
                f"underlay_mode must be 'haadf' or 'dual'; got {proposal['value']!r}"
            )
        return value

    @traitlets.validate("stretch_percentiles")
    def _validate_stretch_percentiles(self, proposal: dict) -> list[float]:
        value = [float(v) for v in proposal["value"]]
        if len(value) != 2:
            raise traitlets.TraitError(
                "stretch_percentiles must be [low, high]; got "
                f"{proposal['value']!r}"
            )
        lo, hi = value
        if not (0.0 <= lo < hi <= 100.0):
            raise traitlets.TraitError(
                f"stretch_percentiles must satisfy 0 <= low < high <= 100; got [{lo}, {hi}]"
            )
        return value

    @traitlets.validate("display_gamma")
    def _validate_display_gamma(self, proposal: dict) -> float:
        value = float(proposal["value"])
        if not math.isfinite(value) or value <= 0:
            raise traitlets.TraitError(f"display_gamma must be > 0; got {value}")
        return value

    @traitlets.validate("dual_gain")
    def _validate_dual_gain(self, proposal: dict) -> list[float]:
        value = [float(v) for v in proposal["value"]]
        if len(value) != 2 or any((not math.isfinite(v)) or v < 0 for v in value):
            raise traitlets.TraitError(
                "dual_gain must be two non-negative finite values [gain_a, gain_b]; "
                f"got {proposal['value']!r}"
            )
        return value

    @traitlets.validate("denoise_modes")
    def _validate_display_filters(self, proposal: dict) -> list[str]:
        from quantem.widget.utils.display_filter import DISPLAY_FILTER_MODES, _normalize_mode

        modes = [_normalize_mode(value) for value in proposal["value"]]
        for mode in modes:
            if mode != "none" and mode not in DISPLAY_FILTER_MODES:
                raise traitlets.TraitError(
                    "denoise_modes entries must be one of "
                    + "|".join(DISPLAY_FILTER_MODES)
                    + f" (or 'off'/'raw'); got {mode!r}"
                )
        return modes

    @traitlets.validate("denoise_bins")
    def _validate_spatial_bins(self, proposal: dict) -> list[int]:
        values = [int(value) for value in proposal["value"]]
        for value in values:
            if value not in (1, 2, 4):
                raise traitlets.TraitError(
                    f"denoise_bins entries must be 1, 2, or 4; got {value}"
                )
        return values

    @traitlets.validate("pad_ratio")
    def _validate_pad_ratio(self, proposal: dict) -> float:
        value = float(proposal["value"])
        if not 0.0 <= value <= 1.0:
            raise traitlets.TraitError(
                f"pad_ratio must be between 0 and 1 (border as a fraction of "
                f"max(rows, cols)); got {value}"
            )
        # pad is a single-panel display window, like crop_to_view(): a gallery
        # or an RGB panel cannot pad, so reject it loudly instead of silently
        # ignoring the border (and announcing one that never happened). 0.0
        # always passes, so reset_view_ops() and the default stay valid.
        if value > 0 and getattr(self, "_data", None) is not None:
            if int(self.n_images) != 1:
                raise NotImplementedError(
                    f"pad_ratio supports a single panel; this widget shows "
                    f"{int(self.n_images)} panels. Pad the array before display "
                    f"for galleries."
                )
            if any(self.is_rgb):
                raise NotImplementedError(
                    "pad_ratio supports grayscale panels; this panel is RGB."
                )
        return value

    @traitlets.validate("view_crop")
    def _validate_view_crop(self, proposal: dict) -> list[int]:
        values = [int(v) for v in proposal["value"]]
        if not values:
            return []
        if len(values) != 4:
            raise traitlets.TraitError(
                f"view_crop must be empty or (row0, row1, col0, col1) in "
                f"full-resolution image pixels; got {len(values)} values"
            )
        if getattr(self, "_data", None) is None:
            return values
        if int(self.n_images) != 1:
            raise traitlets.TraitError(
                f"view_crop supports a single panel; this widget shows "
                f"{int(self.n_images)} panels"
            )
        full_h, full_w = int(self._data.shape[1]), int(self._data.shape[2])
        row0, row1, col0, col1 = values
        row0, col0 = max(0, row0), max(0, col0)
        row1, col1 = min(full_h, row1), min(full_w, col1)
        if row1 - row0 < 2 or col1 - col0 < 2:
            raise traitlets.TraitError(
                f"view_crop window must span at least 2x2 pixels inside the "
                f"{full_h}x{full_w} image; got {values}"
            )
        return [row0, row1, col0, col1]

    # =========================================================================
    # Reversible view ops: crop-to-view + pad (display window, single panel)
    # =========================================================================
    def _view_ops_geometry(self) -> tuple[tuple[int, int, int, int], int]:
        """Active crop window and pad width in DISPLAY pixels.

        The view_crop trait holds full-resolution coordinates while the
        packed frames live in display pixels (after any _display_bin), so
        the window is rescaled here. No crop returns the full display frame.
        RGB panels bypass view ops (the mixed packing path ships display
        RGB pixels that these helpers never see).
        """
        data = self._display_data if self._display_data is not None else self._data
        full_h, full_w = int(data.shape[1]), int(data.shape[2])
        single_scalar = int(self.n_images) == 1 and not any(self.is_rgb)
        factor = max(1, int(self._display_bin_factor))
        if len(self.view_crop) == 4 and single_scalar:
            row0, row1, col0, col1 = (int(v) for v in self.view_crop)
            crop = (
                max(0, row0 // factor),
                min(full_h, -(-row1 // factor)),
                max(0, col0 // factor),
                min(full_w, -(-col1 // factor)),
            )
        else:
            crop = (0, full_h, 0, full_w)
        pad = 0
        if float(self.pad_ratio) > 0 and single_scalar:
            # Border width is a fraction of the image's max(rows, cols) so
            # the same ratio reads the same at any aspect; the border value
            # is the frame minimum, which keeps the colormap floor.
            pad = max(1, round(float(self.pad_ratio) * max(full_h, full_w)))
        return crop, pad

    def _view_ops_active(self) -> bool:
        crop, pad = self._view_ops_geometry()
        data = self._display_data if self._display_data is not None else self._data
        return pad > 0 or crop != (0, int(data.shape[1]), 0, int(data.shape[2]))

    def _crop_view_stack(self, data: np.ndarray) -> np.ndarray:
        """Crop VIEW of the display stack (before denoise); never copies."""
        (row0, row1, col0, col1), _pad = self._view_ops_geometry()
        return data[:, row0:row1, col0:col1]

    def _pad_view_stack(self, data: np.ndarray) -> np.ndarray:
        """Constant border around each frame (after denoise); value = frame min."""
        _crop, pad = self._view_ops_geometry()
        if pad <= 0:
            return data
        frames = []
        for frame in np.asarray(data, dtype=np.float32):
            finite = frame[np.isfinite(frame)]
            fill = float(finite.min()) if finite.size else 0.0
            frames.append(np.pad(frame, pad, constant_values=fill))
        return np.stack(frames, axis=0)

    def _refresh_view_ops(self, *, announce: bool) -> None:
        """Sync the frame extent, cursor offset, and crop/pad notice.

        An active view reduction is never silent (house rule): the banner
        names the crop window in full-image coordinates and the pad ratio,
        and says that reset_view_ops() restores the full frame.
        """
        (row0, row1, col0, col1), pad = self._view_ops_geometry()
        factor = max(1, int(self._display_bin_factor))
        self.height = (row1 - row0) + 2 * pad
        self.width = (col1 - col0) + 2 * pad
        self._view_crop_offset = [(row0 - pad) * factor, (col0 - pad) * factor]
        parts = []
        if len(self.view_crop) == 4:
            r0, r1, c0, c1 = (int(v) for v in self.view_crop)
            parts.append(f"cropped to ({r0},{c0})-({r1},{c1})")
        # Announce the pad only when it was actually applied. The geometry
        # gates the pad on single_scalar, so pad > 0 (not pad_ratio > 0) is the
        # honest test: a gallery/RGB widget never pads, so it must never claim
        # a border in the banner. Constructing such a widget with pad_ratio > 0
        # raises in _validate_pad_ratio, but a later trait assignment could
        # still reach here.
        if pad > 0:
            parts.append(f"pad {float(self.pad_ratio):.0%}")
        banner = (
            f"view: {' · '.join(parts)} (reset_view_ops() restores full frame)"
            if parts
            else ""
        )
        changed = banner != self.view_banner
        self.view_banner = banner
        if announce and banner and changed:
            print(banner)

    @traitlets.observe("view_crop", "pad_ratio")
    def _on_view_ops_change(self, change: dict) -> None:
        """Repack the display frames when the crop window or pad changes."""
        if not getattr(self, "_view_ops_ready", False):
            return
        with self.hold_sync():
            self._refresh_view_ops(announce=True)
            # The committed window now fills the frame: clear the stale
            # zoom/viewport so the browser paints the new extent at 1x.
            self.view_box = []
            self.initial_zoom = 1.0
            self.zoom_row = None
            self.zoom_col = None
            self._update_all_frames()

    @traitlets.observe("denoise", "denoise_sigma", "denoise_bin")
    def _on_display_filter_scalar_change(self, change: dict) -> None:
        """UI editor knobs write through to the per-panel lists.

        Scope follows ``denoise_scope``: "all" broadcasts the edit to every
        panel, "panel" edits only the selected panel.
        """
        if not getattr(self, "_display_filter_ready", False) or self._filter_knob_sync:
            return
        idx = min(int(self.selected_idx), max(0, int(self.n_images) - 1))
        n_panels = int(self.n_images)

        def updated(current, value):
            values = list(current)
            if len(values) != n_panels:
                values = [value] * n_panels
            if self.denoise_scope != "panel":
                return [value] * n_panels
            values[idx] = value
            return values

        name = change["name"]
        if name == "denoise":
            from quantem.widget.utils.display_filter import resolve_denoise_mode

            mode, extra_bin = resolve_denoise_mode(str(change["new"]))
            if mode != str(change["new"]):
                # Compound alias: rewrite the scalar knobs to the canonical
                # (mode, bin) pair; the recursive observer runs write the
                # per-panel lists for each.
                self.denoise = mode
                if extra_bin > 1:
                    self.denoise_bin = max(int(self.denoise_bin), extra_bin)
                return
            self.denoise_modes = updated(self.denoise_modes, mode)
        elif name == "denoise_sigma":
            self.denoise_sigmas = updated(self.denoise_sigmas, float(change["new"]))
        else:
            self.denoise_bins = updated(self.denoise_bins, int(change["new"]))

    @traitlets.observe("denoise_modes", "denoise_sigmas", "denoise_bins")
    def _on_display_filter_change(self, change: dict) -> None:
        """Repack the display frames when per-panel knobs change (no disk I/O)."""
        if not getattr(self, "_display_filter_ready", False):
            return
        self._refresh_display_filter_banner(announce=True)
        self._update_all_frames()

    @traitlets.observe("selected_idx")
    def _on_selected_panel_filter_sync(self, change: dict) -> None:
        """Panel scope: the scalar editor knobs mirror the selected panel."""
        if not getattr(self, "_display_filter_ready", False) or self.denoise_scope != "panel":
            return
        idx = int(change["new"])
        if not (0 <= idx < len(self.denoise_modes)):
            return
        self._filter_knob_sync = True
        try:
            self.denoise = self.denoise_modes[idx]
            self.denoise_sigma = float(self.denoise_sigmas[idx])
            self.denoise_bin = int(self.denoise_bins[idx])
        finally:
            self._filter_knob_sync = False

    def _panel_filter_knobs(self, panel: int) -> tuple[str, float, int]:
        filters = self.denoise_modes
        if not filters or panel >= len(filters):
            return str(self.denoise), float(self.denoise_sigma), int(self.denoise_bin)
        return (
            str(filters[panel]),
            float(self.denoise_sigmas[panel]),
            int(self.denoise_bins[panel]),
        )

    def _panel_filter_active(self, panel: int) -> bool:
        from quantem.widget.utils.display_filter import _normalize_mode

        mode, _sigma, denoise_bin = self._panel_filter_knobs(panel)
        return _normalize_mode(mode) != "none" or denoise_bin > 1

    def _has_denoise_config(self) -> bool:
        """Any panel carries an active denoise config, regardless of the master
        on/off. This is the CONFIG check (used to seed denoise_enabled)."""
        return any(self._panel_filter_active(i) for i in range(int(self.n_images)))

    def _display_filter_active(self) -> bool:
        """Denoise is actually applied: has a config AND the master switch is on.
        Turning denoise_enabled off shows raw while preserving the config."""
        return self._has_denoise_config() and bool(self.denoise_enabled)

    def _panel_browser_filtered(self, panel: int) -> bool:
        """True when this panel's display filter runs in the browser.

        The WGSL port (js/displayFilter.ts) covers the gaussian/bin2/anscombe
        modes; when the frontend negotiated ``_webgpu_filter_ok`` those panels
        ship raw pixels and the browser filters. Non portable modes (tv,
        denova*) keep the Python path even in a WebGPU session.
        """
        from quantem.widget.utils.display_filter import (
            BROWSER_DISPLAY_FILTER_MODES,
            _normalize_mode,
        )

        if not bool(self._webgpu_filter_ok):
            return False
        mode, _sigma, _spatial_bin = self._panel_filter_knobs(panel)
        return _normalize_mode(mode) in BROWSER_DISPLAY_FILTER_MODES

    @traitlets.observe("_webgpu_filter_ok")
    def _on_webgpu_filter_ok_change(self, change: dict) -> None:
        """Repack frames when the browser filter negotiation flips.

        True: portable panels repack raw (the browser filters them). False
        (e.g. reopening in a browser without WebGPU): Python filtering packs
        the view again, so the fallback shows identical pixels.
        """
        if not getattr(self, "_display_filter_ready", False):
            return
        if self._display_filter_active():
            self._update_all_frames()

    def _refresh_display_filter_banner(self, *, announce: bool) -> None:
        """Sync the one-line reduction notice; print it when it changes.

        Announcing an active reduction is a house rule: the user must always
        know the view is filtered and that ``denoise='none'`` restores
        raw counts. Mixed per-panel setups summarize which panels are filtered.
        """
        from quantem.widget.utils.display_filter import format_display_filter_banner

        knobs = [self._panel_filter_knobs(i) for i in range(int(self.n_images))]
        active = [i for i in range(int(self.n_images)) if self._panel_filter_active(i)]
        if not active:
            banner = ""
        elif len(set(knobs)) == 1:
            mode, sigma, denoise_bin = knobs[0]
            banner = format_display_filter_banner(mode, sigma, denoise_bin)
        else:
            per_panel = ", ".join(
                f"p{i}:{knobs[i][0]} σ={knobs[i][1]:g}"
                + (f" bin{knobs[i][2]}" if knobs[i][2] > 1 else "")
                for i in active
            )
            banner = f"denoise: {per_panel} (set denoise='none' for raw counts)"
        changed = banner != self.denoise_banner
        self.denoise_banner = banner
        if announce and banner and changed:
            print(banner)

    def _filter_display_frame(self, frame: np.ndarray, panel: int | None = None) -> np.ndarray:
        from quantem.widget.utils.display_filter import apply_display_filter

        if panel is None:
            mode, sigma, denoise_bin = (
                str(self.denoise),
                float(self.denoise_sigma),
                int(self.denoise_bin),
            )
        else:
            mode, sigma, denoise_bin = self._panel_filter_knobs(panel)
        return apply_display_filter(
            np.asarray(frame), mode=mode, sigma=sigma, spatial_bin=denoise_bin
        )

    def _filtered_frames(self, data: np.ndarray) -> np.ndarray:
        """Filtered VIEW of the display stack; the input arrays are never touched.

        Each panel uses its own knobs from the per-panel lists. RGB panels are
        skipped: per-channel filtering of composed color panels changes hue
        balance, so only 2D scalar panels are filtered.
        """
        if not self._display_filter_active():
            return data
        frames = []
        for i in range(int(data.shape[0])):
            rgb = bool(self.is_rgb[i]) if i < len(self.is_rgb) else False
            # Browser-filtered panels ship raw: the WGSL port applies the same
            # math client-side (live sigma scrub, kernel-less HTML exports).
            skip = rgb or not self._panel_filter_active(i) or self._panel_browser_filtered(i)
            frames.append(
                np.asarray(data[i], dtype=np.float32)
                if skip
                else self._filter_display_frame(data[i], panel=i)
            )
        return np.stack(frames, axis=0)

    def _filtered_underlay_input(self, panel: int) -> np.ndarray:
        """A source panel with its active display filter applied, if any.

        The underlay/dual composite always matches the filtered map panels next
        to it; the stored arrays keep raw counts.
        """
        raw = np.asarray(self._data[panel], dtype=np.float32)
        if self._panel_filter_active(panel):
            return self._filter_display_frame(raw, panel=panel)
        return raw

    def _compute_underlay_blend(self) -> np.ndarray:
        """Chemistry composite from the two source panels, dispatched by mode.

        ``underlay_mode='haadf'`` blends one element map (magenta) onto the
        HAADF lattice; ``'dual'`` composes two maps into a magenta+green panel.
        Both stretch each map with ``stretch_percentiles`` for display; the
        stored arrays keep raw counts.
        """
        from quantem.widget.utils.display_filter import blend_map_on_haadf, magenta_cmap

        lo_pct = float(self.stretch_percentiles[0])
        hi_pct = float(self.stretch_percentiles[1])

        def norm01(image, low, high):
            lo, hi = np.percentile(image, [low, high])
            span = hi - lo if hi > lo else 1.0
            return np.clip((image - lo) / span, 0.0, 1.0)

        if str(self.underlay_mode).lower() == "dual":
            map_a = self._filtered_underlay_input(0)
            map_b = self._filtered_underlay_input(1)
            return _compose_dual(
                norm01(map_a, lo_pct, hi_pct),
                norm01(map_b, lo_pct, hi_pct),
                gain=(float(self.dual_gain[0]), float(self.dual_gain[1])),
            )

        haadf = np.asarray(self._data[self._underlay_haadf_idx], dtype=np.float32)
        map_view = self._filtered_underlay_input(self._underlay_map_idx)

        cmap_name = str(self.cmap).lower()
        if cmap_name in ("magenta", "eds_magenta", "mag"):
            cmap = magenta_cmap()
        else:
            import matplotlib.pyplot as plt

            try:
                cmap = plt.colormaps[str(self.cmap)]
            except KeyError:
                cmap = magenta_cmap()
        return blend_map_on_haadf(
            # stretch_percentiles defaults to (4, 99), the dominant stretch of
            # the drift-paper Fig4 sweep; sparse maps at (2, 99.5) render dark.
            norm01(map_view, lo_pct, hi_pct),
            norm01(haadf, 1.0, 99.9),
            alpha=float(self.underlay_alpha),
            haadf_gain=float(self.underlay_haadf_gain),
            gamma=float(self.display_gamma),
            cmap=cmap,
        ).astype(np.float32)

    @traitlets.observe(
        "underlay_alpha", "underlay_haadf_gain", "underlay_mode",
        "stretch_percentiles", "display_gamma", "dual_gain",
    )
    def _on_underlay_change(self, change: dict) -> None:
        """Re-blend the chemistry panel live when a Fig4 knob moves."""
        if getattr(self, "_underlay_map_idx", None) is None or not getattr(
            self, "_display_filter_ready", False
        ):
            return
        self._update_all_frames()

    def _update_all_frames(self):
        """Send display data to JS (possibly binned for large galleries)."""
        if getattr(self, "_underlay_map_idx", None) is not None and getattr(
            self, "_display_filter_ready", False
        ):
            # The blend depends on filter knobs + underlay sliders; recompute
            # the RGB panel here so every knob path repacks a fresh blend.
            composed = self._compute_underlay_blend()
            self._rgb_frames[-1] = composed
            display_rgb = list(getattr(self, "_display_rgb", self._rgb_frames))
            if self._display_bin > 1:
                from quantem.widget.utils.array import bin2d

                display_rgb[-1] = bin2d(
                    composed.transpose(2, 0, 1), factor=self._display_bin, mode="mean"
                ).transpose(1, 2, 0)
            else:
                display_rgb[-1] = composed
            self._display_rgb = display_rgb
        data = self._display_data if self._display_data is not None else self._data
        # Reversible view ops: crop the display window BEFORE denoise (so the
        # filter operates on the cropped region) and pad AFTER it (so the
        # border stays a flat minimum instead of smearing into the filter).
        data = self._crop_view_stack(data)
        data = self._filtered_frames(data)
        data = self._pad_view_stack(data)
        if any(self.is_rgb):
            # Mixed packing: each panel is one contiguous float32 block, W*H
            # floats for grayscale, 3*W*H interleaved floats for RGB. JS derives
            # per-panel offsets from the synced is_rgb flags.
            display_rgb = getattr(self, "_display_rgb", self._rgb_frames)
            blocks = [
                np.ascontiguousarray(display_rgb[i], dtype=np.float32).tobytes()
                if self.is_rgb[i] else
                np.ascontiguousarray(data[i], dtype=np.float32).tobytes()
                for i in range(int(data.shape[0]))
            ]
            self.frame_bytes = _b64_safe(b"".join(blocks))
        elif self.offline:
            # Quantize to uint8 PER IMAGE (4x smaller than float32). Each panel uses
            # its own (min, max) so every panel gets the full 256 codes - a global
            # range would starve narrow-range panels and comb their histograms.
            # Display-only: the colormap reduces to 256 levels anyway.
            arr = np.ascontiguousarray(data, dtype=np.float32)  # (n, H, W)
            flat = arr.reshape(arr.shape[0], -1)
            out = np.empty(flat.shape, dtype=np.uint8)
            mins, maxs = [], []
            for i in range(flat.shape[0]):
                finite = flat[i][np.isfinite(flat[i])]
                lo = float(finite.min()) if finite.size else 0.0
                hi = float(finite.max()) if finite.size else 1.0
                rng = hi - lo if hi > lo else 1.0
                out[i] = np.clip((flat[i] - lo) * (255.0 / rng), 0, 255).astype(np.uint8)
                mins.append(lo)
                maxs.append(hi)
            self._offline_mins = mins
            self._offline_maxs = maxs
            self._offline_min = mins[0]  # back-compat scalars (single-image readers)
            self._offline_max = maxs[0]
            self.frame_bytes = _b64_safe(out.tobytes())
        else:
            self.frame_bytes = _b64_safe(data.tobytes())
        self._update_panel_stack_bytes()

    def _update_panel_stack_bytes(self) -> None:
        """Pack only multi-frame panel data for browser-local frame changes."""
        stacks = getattr(self, "_display_panel_stacks", None)
        if not stacks or not any(int(stack.shape[0]) > 1 for stack in stacks):
            self.panel_stack_offsets = [-1] * int(getattr(self, "n_images", 0))
            self.panel_stack_bytes = b""
            self._panel_stack_mins = []
            self._panel_stack_maxs = []
            return

        offsets: list[int] = []
        blocks: list[bytes] = []
        mins = [0.0] * len(stacks)
        maxs = [1.0] * len(stacks)
        float_offset = 0
        filter_active = self._display_filter_active()
        for panel, stack in enumerate(stacks):
            if int(stack.shape[0]) <= 1:
                offsets.append(-1)
                continue
            rgb = bool(self.is_rgb[panel]) if panel < len(self.is_rgb) else False
            if not rgb:
                # View ops track the main frame so browser-local frame
                # scrubbing shows the same committed crop window.
                stack = self._crop_view_stack(stack)
            # Browser-filtered panels ship raw stacks; JS filters each frame
            # on scrub with the same per-panel knobs.
            skip = rgb or not self._panel_filter_active(panel) or self._panel_browser_filtered(panel)
            if filter_active and not skip:
                # Per-frame filtered VIEW so browser-local frame scrubbing shows
                # the same filter as the main frame; the stored stack is untouched.
                stack = np.stack(
                    [
                        self._filter_display_frame(stack[k], panel=panel)
                        for k in range(int(stack.shape[0]))
                    ],
                    axis=0,
                )
            if not rgb:
                stack = self._pad_view_stack(stack)
            arr = np.ascontiguousarray(stack, dtype=np.float32)
            offsets.append(float_offset)
            float_offset += int(arr.size)
            if self.offline:
                finite = arr[np.isfinite(arr)]
                lo = float(finite.min()) if finite.size else 0.0
                hi = float(finite.max()) if finite.size else 1.0
                rng = hi - lo if hi > lo else 1.0
                blocks.append(
                    np.clip((arr - lo) * (255.0 / rng), 0, 255)
                    .astype(np.uint8)
                    .tobytes()
                )
                mins[panel] = lo
                maxs[panel] = hi
            else:
                blocks.append(arr.tobytes())
        self.panel_stack_offsets = offsets
        self._panel_stack_mins = mins if self.offline else []
        self._panel_stack_maxs = maxs if self.offline else []
        self.panel_stack_bytes = _b64_safe(b"".join(blocks))

    def _apply_rotations(self):
        """Re-rotate each displayed image from its original by ``image_rotations[i] * 90°``.

        This is purely a display-time reorientation of each 2D image via
        ``np.rot90`` — it is NOT scan rotation (which would rotate the
        scan grid in a 4D-STEM dataset). Originals are kept in
        ``_data_original`` so successive rotations compose from the
        unrotated source rather than accumulating interpolation error.
        Mixed shapes after rotation are center-padded to a common size.
        """
        has_local_stacks = any(
            count > 1 for count in getattr(self, "panel_frame_counts", [])
        )
        if has_local_stacks:
            has_rotation = any(
                (self.image_rotations[i] if i < len(self.image_rotations) else 0) % 4 != 0
                for i in range(len(self._panel_stacks_original))
            )
            if not has_rotation and self._panel_stack_originals_are_views:
                return
            if self._panel_stack_originals_are_views and has_rotation:
                self._panel_stacks_original = [
                    stack.copy() for stack in self._panel_stacks_original
                ]
                self._panel_stack_originals_are_views = False
            rotated_stacks = []
            for panel, original in enumerate(self._panel_stacks_original):
                k = (
                    self.image_rotations[panel]
                    if panel < len(self.image_rotations)
                    else 0
                ) % 4
                rotated_stacks.append(
                    original if k == 0 else np.rot90(original, k=k, axes=(-2, -1))
                )
            target_h = max(int(stack.shape[-2]) for stack in rotated_stacks)
            target_w = max(int(stack.shape[-1]) for stack in rotated_stacks)
            normalized_stacks = []
            for stack in rotated_stacks:
                if stack.shape[-2:] == (target_h, target_w):
                    normalized_stacks.append(np.asarray(stack, dtype=np.float32))
                else:
                    normalized_stacks.append(
                        np.stack(
                            [_resize_image(frame, target_h, target_w) for frame in stack],
                            axis=0,
                        ).astype(np.float32, copy=False)
                    )
            self._panel_stacks = normalized_stacks
            indices = list(self.panel_frame_indices)
            self._data = np.stack(
                [stack[index] for stack, index in zip(normalized_stacks, indices)],
                axis=0,
            ).astype(np.float32, copy=False)
            if self._display_bin > 1:
                from quantem.widget.utils.array import bin2d

                self._display_panel_stacks = [
                    bin2d(stack, factor=self._display_bin, mode="mean")
                    for stack in normalized_stacks
                ]
            else:
                self._display_panel_stacks = normalized_stacks
            self._display_data = np.stack(
                [stack[index] for stack, index in zip(self._display_panel_stacks, indices)],
                axis=0,
            ).astype(np.float32, copy=False)
            self._data_original = [self._data[i] for i in range(self._data.shape[0])]
            self._originals_are_views = True
            self.height = int(self._display_data.shape[1])
            self.width = int(self._display_data.shape[2])
            self._compute_all_stats()
            self._update_all_frames()
            return

        # Materialize originals as independent copies only when a non-zero
        # rotation exists (they start as views into _data to avoid 800MB copy at init)
        has_rotation = any(
            (self.image_rotations[i] if i < len(self.image_rotations) else 0) % 4 != 0
            for i in range(len(self._data_original))
        )
        # No-rotation fast path: skip 30+ MB of redundant tobytes + stats recomputation
        # on every widget init.  The observer fires once when image_rotations = [0]*n
        # is assigned in __init__; without this guard that triggered a full frame
        # rebuild + stats recompute for a no-op.
        if not has_rotation and self._originals_are_views:
            return
        if self._originals_are_views and has_rotation:
            self._data_original = [img.copy() for img in self._data_original]
            self._originals_are_views = False
        rotated = []
        for i, orig in enumerate(self._data_original):
            k = self.image_rotations[i] if i < len(self.image_rotations) else 0
            k = k % 4
            if k == 0:
                rotated.append(orig)
            else:
                rotated.append(np.rot90(orig, k=k))
        # If shapes differ after rotation, center-pad all to max dims
        shapes = [img.shape for img in rotated]
        if len(set(shapes)) > 1:
            max_h = max(s[0] for s in shapes)
            max_w = max(s[1] for s in shapes)
            padded = []
            for img in rotated:
                h, w = img.shape
                pad_top = (max_h - h) // 2
                pad_bot = max_h - h - pad_top
                pad_left = (max_w - w) // 2
                pad_right = max_w - w - pad_left
                padded.append(np.pad(img, ((pad_top, pad_bot), (pad_left, pad_right)), mode="constant", constant_values=0))
            rotated = padded
        self._data = np.stack(rotated).astype(np.float32)
        # Recompute display data if binning is active
        if self._display_bin > 1:
            from quantem.widget.utils.array import bin2d
            self._display_data = bin2d(self._data, factor=self._display_bin, mode="mean")
        else:
            self._display_data = self._data
        display = self._display_data if self._display_data is not None else self._data
        self.height = int(display.shape[1])
        self.width = int(display.shape[2])
        self._compute_all_stats()
        self._update_all_frames()

    @traitlets.observe("image_rotations")
    def _on_image_rotations_changed(self, change):
        if hasattr(self, "_data_original"):
            self._apply_rotations()

    def rotate(self, idx: int, angle: int) -> Self:
        """Rotate image ``idx`` by ``angle`` degrees (CCW-positive, matches np.rot90).

        Rotation convention follows ``np.rot90``::

            angle | image_rotations | np.rot90 k | direction
            ------+-----------------+------------+----------
              90  |        1        |     1      | 90° CCW
             180  |        2        |     2      | 180°
             -90  |        3        |     3      | 90° CW
             360  |        0        |     0      | identity

        Parameters
        ----------
        idx : int
            Image index in the gallery (0-based).
        angle : int
            Rotation angle in degrees (must be a multiple of 90).
            Positive = counter-clockwise, negative = clockwise.

        Returns
        -------
        Self
        """
        if any(self.is_rgb):
            raise NotImplementedError(
                "rotate() is not supported when the gallery contains RGB panels; "
                "rotation rebuilds panels from the grayscale stack and would desync the color pixels."
            )
        if angle % 90 != 0:
            raise ValueError(f"Rotation angle must be a multiple of 90 (got {angle}). Use 0, 90, 180, 270, or -90, -180, -270.")
        if idx < 0 or idx >= self.n_images:
            raise IndexError(f"Image index {idx} out of range [0, {self.n_images})")
        k = (angle // 90) % 4
        rots = list(self.image_rotations)
        while len(rots) < self.n_images:
            rots.append(0)
        rots[idx] = (rots[idx] + k) % 4
        self.image_rotations = rots
        return self

    def _sample_profile(self, row0, col0, row1, col1):
        img = self._data[self.selected_idx]
        h, w = img.shape
        dc, dr = col1 - col0, row1 - row0
        length = (dc**2 + dr**2) ** 0.5
        n = max(2, int(np.ceil(length)))
        t = np.linspace(0, 1, n)
        cs = col0 + t * dc
        rs = row0 + t * dr
        ci = np.floor(cs).astype(int)
        ri = np.floor(rs).astype(int)
        cf = cs - ci
        rf = rs - ri
        c0c = np.clip(ci, 0, w - 1)
        c1c = np.clip(ci + 1, 0, w - 1)
        r0c = np.clip(ri, 0, h - 1)
        r1c = np.clip(ri + 1, 0, h - 1)
        return (img[r0c, c0c] * (1 - cf) * (1 - rf) +
                img[r0c, c1c] * cf * (1 - rf) +
                img[r1c, c0c] * (1 - cf) * rf +
                img[r1c, c1c] * cf * rf).astype(np.float32)

    def set_profile(self, start: tuple, end: tuple):
        """Set a line profile between two points (image pixel coordinates).

        Parameters
        ----------
        start : tuple of (row, col)
            Start point in pixel coordinates.
        end : tuple of (row, col)
            End point in pixel coordinates.
        """
        row0, col0 = start
        row1, col1 = end
        self.profile_line = [
            {"row": float(row0), "col": float(col0)},
            {"row": float(row1), "col": float(col1)},
        ]

    def clear_profile(self):
        """Clear the current line profile."""
        self.profile_line = []

    def _upsert_selected_roi(self, updates: dict):
        rois = list(self.roi_list)
        color_cycle = ["#4fc3f7", "#81c784", "#ffb74d", "#ce93d8", "#ef5350", "#ffd54f", "#90a4ae", "#a1887f"]
        defaults = {
            "shape": "square",
            "row": int(self.height // 2),
            "col": int(self.width // 2),
            "radius": 10,
            "radius_inner": 5,
            "width": 20,
            "height": 20,
            "line_width": 2,
            "highlight": False,
            "visible": True,
            "locked": False,
        }
        if self.roi_selected_idx >= 0 and self.roi_selected_idx < len(rois):
            current = {**defaults, **rois[self.roi_selected_idx]}
            if not current.get("color"):
                current["color"] = color_cycle[self.roi_selected_idx % len(color_cycle)]
            rois[self.roi_selected_idx] = {**current, **updates}
        else:
            rois.append({**defaults, "color": color_cycle[len(rois) % len(color_cycle)], **updates})
            self.roi_selected_idx = len(rois) - 1
        self.roi_list = rois
        self.roi_active = True

    def add_roi(self, row: int | None = None, col: int | None = None, shape: str = "square") -> Self:
        with self.hold_sync():
            self.roi_selected_idx = -1
            self._upsert_selected_roi({
                "shape": shape,
                "row": int(self.height // 2 if row is None else row),
                "col": int(self.width // 2 if col is None else col),
            })
        return self

    def clear_rois(self) -> Self:
        with self.hold_sync():
            self.roi_list = []
            self.roi_selected_idx = -1
            self.roi_active = False
        return self

    def delete_selected_roi(self) -> Self:
        idx = int(self.roi_selected_idx)
        if idx < 0 or idx >= len(self.roi_list):
            return self
        with self.hold_sync():
            rois = [roi for i, roi in enumerate(self.roi_list) if i != idx]
            self.roi_list = rois
            self.roi_selected_idx = min(idx, len(rois) - 1) if rois else -1
            if not rois:
                self.roi_active = False
        return self

    def set_roi(self, row: int, col: int, radius: int = 10) -> Self:
        with self.hold_sync():
            self._upsert_selected_roi({"shape": "circle", "row": int(row), "col": int(col), "radius": int(radius)})
        return self

    def roi_circle(self, radius: int = 10) -> Self:
        with self.hold_sync():
            self._upsert_selected_roi({"shape": "circle", "radius": int(radius)})
        return self

    def roi_square(self, half_size: int = 10) -> Self:
        with self.hold_sync():
            self._upsert_selected_roi({"shape": "square", "radius": int(half_size)})
        return self

    def roi_rectangle(self, width: int = 20, height: int = 10) -> Self:
        with self.hold_sync():
            self._upsert_selected_roi({"shape": "rectangle", "width": int(width), "height": int(height)})
        return self

    def roi_annular(self, inner: int = 5, outer: int = 10) -> Self:
        with self.hold_sync():
            self._upsert_selected_roi({"shape": "annular", "radius_inner": int(inner), "radius": int(outer)})
        return self

    @property
    def profile(self):
        """Get profile line endpoints as [(row0, col0), (row1, col1)] or [].

        Returns
        -------
        list of tuple
            Line endpoints in pixel coordinates, or empty list if no profile.
        """
        return [(p["row"], p["col"]) for p in self.profile_line]

    @property
    def profile_values(self):
        """Get intensity values along the profile line as a numpy array.

        Returns
        -------
        np.ndarray or None
            Float32 array of sampled intensities, or None if no profile.
        """
        if len(self.profile_line) < 2:
            return None
        p0, p1 = self.profile_line
        return self._sample_profile(p0["row"], p0["col"], p1["row"], p1["col"])

    @property
    def profile_distance(self):
        """Get total distance of the profile line in calibrated units.

        Returns
        -------
        float or None
            Distance in angstroms (if pixel_size > 0) or pixels.
            None if no profile line is set.
        """
        if len(self.profile_line) < 2:
            return None
        p0, p1 = self.profile_line
        dc = p1["col"] - p0["col"]
        dr = p1["row"] - p0["row"]
        dist_px = (dc**2 + dr**2) ** 0.5
        if self.pixel_size > 0:
            return dist_px * self.pixel_size
        return dist_px
