"""Show2D: a 2D image viewer and comparison gallery.

One image or a gallery of images (each panel may be its own frame stack),
with FFT and histogram analysis, ROIs, line profiles, linked zoom/pan and
contrast, paged galleries, and watched-folder updates. Rendering runs in the
browser from the synced traits below; this module owns the Python side:
input normalisation, the frame byte packing, the folder watcher hooks and the
observers that answer the browser.
"""

import json
import math
import pathlib
import warnings
from collections.abc import Mapping, Sequence
from typing import Self

import anywidget
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import torch
import traitlets
from matplotlib import colormaps
from PIL import Image

from quantem.widget.adapters import core as core_adapter
from quantem.widget.folder_watch_status import FOLDER_WATCH_STATE_VALUES
from quantem.widget.colormap import Colormap, cmap_to_name, is_cmap_sequence
from quantem.widget.export import HtmlExportMixin
from quantem.widget.fallback import StaticFallbackMixin
from quantem.widget.pages import PagesMixin
from quantem.widget.panels import PanelsMixin
from quantem.widget.render.figure import format_scale_label, round_to_nice
from quantem.widget.state import SavedStateMixin, announce_browser_limit
from quantem.widget.image_folder import ImageFolderRecord, WatchedImageFolder, WatchedImageFolderMixin
from quantem.widget.io.image import RgbImage
from quantem.widget.show2d.options import (
    _expand_title_spans_for_flattened_labels,
    _normalise_title_span_sequence,
    _normalize_inset_plot_specs,
    _normalize_marker_mapping,
    _normalize_panel_annotations,
    _normalize_panel_indices,
    _normalize_panel_overlays,
    _normalize_panel_title_style,
    _normalize_scale_bar_style,
    _title_span_sequence_length,
)
from quantem.widget.show2d.export import Show2DExport
from quantem.widget.show2d.fallback import Show2DStaticPng
from quantem.widget.show2d.state import Show2DState
from quantem.widget.utils.array import _b64_safe, _resize_image, bin2d, to_numpy
from quantem.widget.utils.state_io import unwrap_state_payload
from quantem.widget.utils.ui import UiMode, resolve_ui_mode

_DEFAULT_FOLDER_PAGE_SIZE = 20
_MAX_PANEL_PLAYBACK_FPS = 30.0
# Rec. 709 luma weights: the standard perceptual grayscale reduction of RGB.
_RGB_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _nonnegative_int(value: object, *, name: str) -> int:
    """Return a nonnegative integer for pixel-width options.

    Gap and border widths arrive as ints, floats or strings; a negative width
    has no meaning, so it fails here under the option's name.
    """
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer, got {value!r}") from exc
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer, got {value!r}")
    return result


def _reject_unknown_kwargs(cls, kwargs: dict) -> None:
    """Raise TypeError if kwargs contains any key that isn't a declared trait.

    anywidget/traitlets silently accept unknown keys, which let stale notebooks
    pass obsolete params like ``pixel_size_angstrom=0.5`` with no warning.  This
    helper catches typos and renamed-trait references at construction time.
    """
    traits = set(cls.class_trait_names())
    unknown = [key for key in kwargs if key not in traits]
    if unknown:
        first = sorted(unknown)[0]
        raise TypeError(
            f"{cls.__name__}() got unexpected keyword argument {first!r}. "
            f"Check for typos or a renamed parameter (e.g. canvas_size → size, "
            f"image_width_px → size, pixel_size_angstrom → pixel_size)."
        )


_BROWSER_DENOISE_MODES = ("none", "gaussian", "anscombe")


def _denoise_mode(mode: object) -> str:
    """A denoise mode the browser filter runs; off/raw/"" mean none.

    Any other name would pass the frame through unfiltered while the controls
    name a filter, so it raises.
    """
    name = str(mode).strip().lower()
    name = "none" if name in ("", "off", "raw") else name
    if name not in _BROWSER_DENOISE_MODES:
        raise ValueError(f"denoise must be one of {_BROWSER_DENOISE_MODES}, got {mode!r}")
    return name


def _per_panel(value, name: str, cast, n_panels: int) -> list:
    """One value per panel from a scalar (broadcast) or a per-panel sequence."""
    if isinstance(value, (list, tuple, np.ndarray)):
        values = [cast(item) for item in value]
        if len(values) != n_panels:
            raise ValueError(f"{name} sequence length ({len(values)}) must equal the panel count ({n_panels})")
        return values
    return [cast(value)] * n_panels


def _resolve_page_labels(
    page_labels: Sequence[str | None] | None,
    default_labels: list[str],
    n_pages: int,
) -> list[str]:
    """``page_labels`` (or the defaults) as text, one per page.

    A wrong count would leave a page untitled or shift every title by one
    page, so it raises instead.
    """
    resolved = ["" if label is None else str(label) for label in (page_labels if page_labels is not None else default_labels)]
    if len(resolved) != n_pages:
        raise ValueError(f"page_labels length ({len(resolved)}) must match n_pages ({n_pages})")
    return resolved


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
        and all(isinstance(item, dict) and any(key in item for key in ("images", "data", "array")) for item in data)
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
            page_panels = to_numpy(raw_images)
            if page_panels.ndim == 2:
                page_panels = page_panels[np.newaxis, ...]
            if page_panels.ndim != 3:
                raise ValueError(
                    f"Show2D page {page_idx} must contain a 2D image or 3D panel stack, "
                    f"got shape {page_panels.shape}"
                )
            if panels_per_page is None:
                panels_per_page = int(page_panels.shape[0])
            elif int(page_panels.shape[0]) != panels_per_page:
                raise ValueError(
                    "Every Show2D page must contain the same number of panels; "
                    f"page 0 has {panels_per_page}, page {page_idx} has {page_panels.shape[0]}"
                )
            page_arrays.append(page_panels)
            inferred_page_labels.append(str(page.get("title") or page.get("label") or f"Page {page_idx + 1}"))
            page_panel_labels = page.get("labels")
            if page_panel_labels is not None:
                if len(page_panel_labels) != int(page_panels.shape[0]):
                    raise ValueError(
                        f"Show2D page {page_idx} labels length ({len(page_panel_labels)}) "
                        f"must match its panel count ({page_panels.shape[0]})"
                    )
                flattened_labels.extend([None if label is None else str(label) for label in page_panel_labels])
            elif labels is not None:
                if len(labels) == int(page_panels.shape[0]):
                    flattened_labels.extend([None if label is None else str(label) for label in labels])
                elif len(labels) == len(data) * panels_per_page:
                    start = page_idx * panels_per_page
                    stop = start + panels_per_page
                    flattened_labels.extend([None if label is None else str(label) for label in labels[start:stop]])
                else:
                    raise ValueError(
                        "labels for paged Show2D must have length panels_per_page "
                        f"({page_panels.shape[0]}) or n_pages * panels_per_page "
                        f"({len(data) * page_panels.shape[0]}), got {len(labels)}"
                    )
            else:
                flattened_labels.extend([f"Panel {i + 1}" for i in range(int(page_panels.shape[0]))])
        stack = np.stack(page_arrays, axis=0)
        n_pages = int(stack.shape[0])
        panels = int(stack.shape[1])
        return (
            stack.reshape(n_pages * panels, *stack.shape[-2:]),
            flattened_labels,
            n_pages,
            panels,
            _resolve_page_labels(page_labels, inferred_page_labels, n_pages),
            [0] * n_pages,
        )

    array = to_numpy(data) if isinstance(data, (np.ndarray, torch.Tensor)) else None
    if array is not None and array.ndim == 4:
        n_pages, panels = int(array.shape[0]), int(array.shape[1])
        resolved_page_labels = _resolve_page_labels(page_labels, [f"Page {i + 1}" for i in range(n_pages)], n_pages)
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
            array.reshape(n_pages * panels, *array.shape[-2:]),
            resolved_labels,
            n_pages,
            panels,
            resolved_page_labels,
            [0] * n_pages,
        )

    return data, labels, 1, 0, [], []


def _is_image_dataset(item) -> bool:
    """Whether ``item`` carries ``.array`` plus ``.sampling`` / ``.units``: a quantem core dataset,
    the widget's ``ArrayDataset`` stand-in, or an ``RgbImage`` from ``read_image``."""
    return isinstance(item, RgbImage) or core_adapter.is_dataset(item)


def _is_rgb_item(item: np.ndarray) -> bool:
    """True when a gallery item is an ``(H, W, 3)`` / ``(H, W, 4)`` color image.

    Detection rule (documented in the Show2D docstring): inside a LIST input,
    any item with ``ndim == 3`` and a trailing dim of 3 or 4 is RGB(A). A bare
    3-D ARRAY input keeps ``(N, H, W)`` stack semantics unless its trailing dim
    is 3/4 AND its leading dim is > 4 (so a 3-frame stack of tiny images never
    silently flips to RGB)."""
    return item.ndim == 3 and item.shape[-1] in (3, 4)


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
        pixels = np.asarray(image)
        if pixels.ndim == 2:
            pixels = pixels[np.newaxis, ...]
        elif pixels.ndim != 3:
            raise ValueError(
                "Show2D list items must be 2D images or 3D (frames, rows, cols) "
                f"stacks; panel {panel} has shape {pixels.shape}"
            )
        if 0 in pixels.shape:
            raise ValueError(
                f"Show2D panel {panel} is empty (shape {pixels.shape}); all dimensions must be >= 1"
            )
        if np.iscomplexobj(pixels):
            raise TypeError(
                f"Show2D panel {panel} contains complex data. Convert first with "
                "np.abs(arr) for magnitude or np.angle(arr) for phase."
            )
        # A float32 copy: the browser transport is float32, and the widget
        # must own its pixels so later edits of the caller's array cannot
        # desync the stats from the display.
        stack = np.array(pixels, dtype=np.float32, copy=True)
        if not np.isfinite(stack).all():
            raise ValueError(
                f"Show2D panel {panel} contains NaN or inf. Clean first with "
                "np.nan_to_num(arr, nan=0, posinf=0, neginf=0)."
            )
        stacks.append(stack)
    normalized = _pad_to_common_shape(stacks)

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
    return normalized, indices, _current_frames(normalized, indices)


def _pad_to_common_shape(stacks: list[np.ndarray]) -> list[np.ndarray]:
    """Center-pad every ``(F, H, W)`` stack, frame by frame, to the largest H and W among them.

    The browser receives one H x W for the whole gallery, so smaller panels are
    padded (with their median, see ``_resize_image``) instead of resampled.
    """
    target_h = max(int(stack.shape[-2]) for stack in stacks)
    target_w = max(int(stack.shape[-1]) for stack in stacks)
    return [
        stack if stack.shape[-2:] == (target_h, target_w)
        else np.stack([_resize_image(frame, target_h, target_w) for frame in stack])
        for stack in stacks
    ]


def _pad_rgb(frame: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Center-pad an ``(H, W, 3)`` panel channel by channel, as ``_resize_image`` pads a grayscale one."""
    return np.stack([_resize_image(channel, rows, cols) for channel in frame.transpose(2, 0, 1)], axis=-1)


def _binned_rgb(frames: list, factor: int) -> list:
    """Mean-bin each RGB panel per channel, so the display copy matches the binned luminance plane."""
    return [
        None if frame is None else bin2d(frame.transpose(2, 0, 1), factor=factor, mode="mean").transpose(1, 2, 0)
        for frame in frames
    ]


def _current_frames(stacks: Sequence[np.ndarray], indices: Sequence[int]) -> np.ndarray:
    """The ``(N, H, W)`` stack of each panel's selected frame: what stats, FFT and detail tiles read."""
    return np.stack([stack[index] for stack, index in zip(stacks, indices)])


def _quantize_uint8(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """``values`` as uint8 codes over their own finite ``(low, high)``, plus that range for the browser to dequantize.

    Each panel gets its own range so every panel keeps all 256 codes; a shared
    range would starve narrow-range panels and comb their histograms. A panel
    with no finite value maps over (0, 1); a constant one over a unit span.
    """
    finite = values[np.isfinite(values)]
    low = float(finite.min()) if finite.size else 0.0
    high = float(finite.max()) if finite.size else 1.0
    span = high - low if high > low else 1.0
    return np.clip((values - low) * (255.0 / span), 0, 255).astype(np.uint8), low, high


class Show2D(
    WatchedImageFolderMixin,
    Show2DExport,
    Show2DState,
    Show2DStaticPng,
    SavedStateMixin,
    HtmlExportMixin,
    PagesMixin,
    PanelsMixin,
    StaticFallbackMixin,
    anywidget.AnyWidget,
):
    """Interactive viewer for one 2D image or a gallery of images.

    Parameters
    ----------
    data : array_like, Dataset2d, Dataset3d, or list of these
        ``(H, W)`` for one image, ``(N, H, W)`` for a gallery, ``(P, N, H, W)``
        for ``P`` pages of ``N`` panels. A list may mix 2D images with 3D
        ``(frames, H, W)`` stacks, giving that panel its own frame scrubber.
        Items with a trailing dimension of 3 or 4 are RGB(A) panels. A quantem
        ``Dataset2d`` / ``Dataset3d`` supplies ``sampling``, ``units`` and the
        title. NumPy, PyTorch and CuPy arrays are accepted.
    labels : list of str, optional
        One label per panel. Entries may also be span lists
        (``[{"text": "a", "color": "#f00"}, {"math": r"\\lambda_3"}]``).
    page_labels : list of str, optional
        One label per page for ``(P, N, H, W)`` input.
    title : str, optional
        Gallery title above the panels.
    ui_mode : {"interactive", "presentation", "report"}, default "interactive"
        Preset for which controls are shown; the ``show_*`` flags override it.
    cmap : str or list of str, default "inferno"
        Colormap, or one per panel.
    sampling : float or (row, col), optional
        Pixel size in ``units``; drives the scale bar and calibrated readouts.
    units : str, optional
        Length unit of ``sampling`` (``"A"``, ``"nm"``, ...).
    show_scale_bar : bool, optional
        Draw the scale bar (kept in the ``scale_bar_visible`` trait).
    scale_bar_position : {"bottom-right", "bottom-left"}
    scale_bar_panels : list of int or str, optional
        Panels (index or label) that draw a scale bar; default all.
    scale_bar_length : float, optional
        Fixed bar length in ``units`` instead of the automatic nice length.
    scale_bar_label : str, optional
        Fixed bar text instead of the formatted length.
    scale_bar_style : dict, optional
        Bar and label styling (``font_size``, ``color``, ``bar_height``, ...).
    show_zoom_indicator : bool, default False
        Draw the zoom factor beside the scale bar.
    show_fft, show_stats, show_controls, controls_collapsed, show_panel_titles : bool, optional
        Visibility of the FFT/histogram panel, the stats row, the control bar
        and the panel titles.
    panel_title_spans : list, optional
        Rich title spans per panel (same schema as span labels).
    panel_title_font_size : int, default 11
    panel_title_style : dict, optional
        Title badge styling (``fg``, ``bg``, ``align``, ``outline_width``, ...).
    log_scale, auto_contrast : bool, default False
        Signed-log1p display and 2/98 percentile contrast.
    vmin, vmax : float or list of float, optional
        Fixed contrast window, or one per panel.
    link_zoom, link_pan : bool, optional
        Galleries link zoom and pan by default; a single image never does.
    link_contrast : bool, default True
        One shared contrast window across panels.
    ncols : int, default 3
        Gallery columns.
    size, panel_width_px : int, default 0
        Panel canvas width in CSS px (0 = frontend default). ``panel_width_px``
        wins when both are given.
    smooth : bool, default False
        Bilinear instead of nearest-neighbour magnification.
    zoom : float, default 1.0
        Initial zoom factor.
    center : (row, col), optional
        Image point the initial view is centred on, in full-resolution pixels.
    display_bin : int or "auto", default 1
        Mean-bin factor of the preview sent to the browser; full-res detail
        streams on zoom either way. The default 1 sends native pixels. An
        integer above 1 bins by that factor. ``"auto"`` picks the smallest
        factor that keeps the gallery under 2.5 GB of browser GPU buffers
        (float32 + RGBA + readback per panel) and each float32 panel under
        16 MiB of initial payload. Any active bin prints one line.
    offline : bool, default False
        Quantize the stack to uint8 per panel for standalone HTML export.
    panel_frame_indices : list of int, optional
        Initial frame per stack panel (negative indices count from the end).
    panel_playback_fps : float, default 10
        Playback rate for stack panels.
    starred : list of int or str, optional
        Panels (index or label) to star.
    marker_colors : list of str, optional
        One identity color per panel, drawn as a bar (``marker_style="left"``)
        or a frame (``"around"``).
    row_markers : dict, optional
        ``{row_index: color}`` frames around gallery rows.
    inset_plots, panel_annotations, overlays, panel_overlays : optional
        Per-panel inset curves, text annotations and shape overlays; see
        ``show2d.annotations`` for the dict schema.
    inter_panel_gap_px, inter_panel_gap_color, gallery_outer_border_px,
    gallery_outer_border_color, panel_inner_border_px, panel_inner_border_color : optional
        Gallery chrome.
    gallery_gap_px : int, optional
        Older name of ``inter_panel_gap_px``; pass one of the two.
    denoise : {"none", "gaussian", "anscombe"} or list of str, default "none"
        Display-only denoise run in the browser (the data and stats stay raw);
        a list gives each panel its own mode. Turns the denoise switch on and
        shows its controls when any panel is filtered.
    denoise_sigma : float or list of float, default 4.0
        Smoothing width in pixels, per panel when a list.
    denoise_bin : int or list of int, default 1
        Spatial bin (1, 2 or 4) before the denoise, per panel when a list.
    state : dict, str or pathlib.Path, optional
        A ``state_dict`` or a file written by ``save`` to restore the view.
    save_state : bool, default False
        Embed the full interactive state in the saved notebook (large). The
        default stores a static preview plus light traits.
    verbose : bool, default True
        Add the preview size to the display-bin notice (the notice itself
        always prints when a bin is active).

    Examples
    --------
    >>> from quantem.widget import Show2D
    >>> Show2D([raw, denoised], labels=["raw", "denoised"], sampling=0.2, units="nm", cmap="gray")
    """

    _esm = pathlib.Path(__file__).parent.parent / "static" / "show2d.js"

    # Browser GPU memory budget for display buffers (MB). Each 4K image needs
    # ~192 MB, so 12x4K = 2304 MB fits and 16+ triggers auto-bin.
    _DISPLAY_BUFFER_BUDGET_MB = 2500
    # Wire budget for the initial frame payload, per panel. A 6144x6144 float32
    # panel is 151 MB and Jupyter's kernel->browser channel moves ~24 MB/s, so
    # shipping it wholesale stalls first paint for ~6 s while the ~1000 px
    # canvas can only show ~1 MP of it. Panels above this budget send a binned
    # preview instead; the browser streams full-res detail for the visible
    # window on zoom (maps-style). 16 MiB keeps a plain 2048x2048 float32
    # image (exactly 16 MiB) on the classic full-payload path.
    _WIRE_BUDGET_BYTES_PER_PANEL = 16 * 1024 * 1024

    # Python-side state with safe defaults so traitlets validators can run
    # before __init__ has populated the instance.
    _data: np.ndarray | None = None
    _display_data: np.ndarray | None = None
    _display_bin = 1
    _display_rgb: list = []
    _rgb_frames: list = []
    _rgb_frames_original: list = []
    _panel_stacks: list = []
    _display_panel_stacks: list = []
    _updating_panel_frames = False
    _folder_source: WatchedImageFolder | None = None
    _folder_page_size: int | None = None
    _folder_display_bin_request: int | str = 1

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
    # Browser-side display denoise and frequency filter (js/display/filter.ts,
    # js/display/frequencyFilter.ts). The browser owns the view transform: it filters
    # raw frames client-side, mirrors the scalar editor traits to the per-panel
    # lists by scope, and writes the banner. Python only ships raw pixels and
    # keeps the traits so saved state and exports carry the knobs.
    denoise = traitlets.Unicode("none").tag(sync=True)
    denoise_sigma = traitlets.Float(4.0).tag(sync=True)
    denoise_bin = traitlets.Int(1).tag(sync=True)
    denoise_modes = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    denoise_sigmas = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    denoise_bins = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    denoise_scope = traitlets.Enum(["all", "panel"], default_value="all").tag(sync=True)
    denoise_banner = traitlets.Unicode("").tag(sync=True)
    show_denoise = traitlets.Bool(False).tag(sync=True)
    denoise_enabled = traitlets.Bool(True).tag(sync=True)
    frequency_filter = traitlets.Enum(
        ["none", "lowpass", "highpass", "bandpass"], default_value="none"
    ).tag(sync=True)
    frequency_filter_enabled = traitlets.Bool(False).tag(sync=True)
    frequency_filter_cutoff = traitlets.Float(0.15).tag(sync=True)
    frequency_filter_center = traitlets.Float(0.30).tag(sync=True)
    frequency_filter_width = traitlets.Float(0.12).tag(sync=True)
    frequency_filter_modes = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    frequency_filter_cutoffs = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    frequency_filter_centers = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    frequency_filter_widths = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    frequency_filter_scope = traitlets.Enum(["all", "panel"], default_value="all").tag(sync=True)
    show_frequency_filter = traitlets.Bool(False).tag(sync=True)
    # Flipped True by JS after the first colormap pass has painted to canvas;
    # Python then drops the synced pixel buffers from the saved-notebook state.
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
    # Offline mode: stack quantized to uint8 per panel against its own (min, max)
    # so every panel keeps the full 256 codes; JS dequantizes on read.
    offline = traitlets.Bool(False).tag(sync=True)
    # True only on a clone written by export_html: forces the standalone HTML to
    # render on a light/white background regardless of the viewer's OS theme.
    _export_light = traitlets.Bool(False).tag(sync=True)
    _static_fallback_jpeg = traitlets.Unicode("").tag(sync=True)
    _static_fallback_mime = traitlets.Unicode("image/jpeg").tag(sync=True)
    _offline_min = traitlets.Float(0.0).tag(sync=True)
    _offline_max = traitlets.Float(1.0).tag(sync=True)
    _offline_mins = traitlets.List(trait=traitlets.Float(), default_value=[]).tag(sync=True)
    _offline_maxs = traitlets.List(trait=traitlets.Float(), default_value=[]).tag(sync=True)
    # Maps-style detail streaming (active whenever the preview is binned, i.e.
    # _display_bin_factor > 1): JS writes a JSON request describing the visible
    # window per panel; Python replies with cropped + binned float32 tiles.
    _detail_request = traitlets.Unicode("").tag(sync=True)
    _detail_meta = traitlets.Unicode("").tag(sync=True)
    _detail_bytes = traitlets.Bytes(b"").tag(sync=True)
    labels = traitlets.List(traitlets.Unicode()).tag(sync=True)
    panel_title_spans = traitlets.List(default_value=[]).tag(sync=True)
    # Per-panel RGB flag. True panels carry display-ready (H, W, 3) pixels that
    # bypass the colormap/contrast pipeline in JS; False panels are grayscale.
    is_rgb = traitlets.List(traitlets.Bool(), default_value=[]).tag(sync=True)
    starred = traitlets.List(traitlets.Int()).tag(sync=True)
    hidden_panels = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    hidden_page_slots = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    panel_order = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    show_panel_titles = traitlets.Bool(True).tag(sync=True)
    panel_title_font_size = traitlets.Int(11).tag(sync=True)
    panel_title_style = traitlets.Dict(default_value={}).tag(sync=True)
    inter_panel_gap_px = traitlets.Int(0).tag(sync=True)
    inter_panel_gap_color = traitlets.Unicode("").tag(sync=True)
    gallery_outer_border_px = traitlets.Int(0).tag(sync=True)
    gallery_outer_border_color = traitlets.Unicode("").tag(sync=True)
    panel_inner_border_px = traitlets.Float(1.0).tag(sync=True)
    panel_inner_border_color = traitlets.Unicode("#d0d0d0").tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    show_title = traitlets.Bool(True).tag(sync=True)
    cmap = traitlets.Unicode("inferno").tag(sync=True)
    panel_cmaps = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    panel_cmaps_memory = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    ncols = traitlets.Int(3).tag(sync=True)
    log_scale = traitlets.Bool(False).tag(sync=True)
    auto_contrast = traitlets.Bool(False).tag(sync=True)
    contrast_preset = traitlets.Unicode("custom").tag(sync=True)
    vmin = traitlets.Float(None, allow_none=True).tag(sync=True)
    vmax = traitlets.Float(None, allow_none=True).tag(sync=True)
    vmins = traitlets.List(trait=traitlets.Float(allow_none=True), allow_none=True, default_value=None).tag(sync=True)
    vmaxs = traitlets.List(trait=traitlets.Float(allow_none=True), allow_none=True, default_value=None).tag(sync=True)
    marker_colors = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    marker_style = traitlets.Enum(["left", "around"], default_value="left").tag(sync=True)
    row_markers = traitlets.Dict(default_value={}).tag(sync=True)
    col_markers = traitlets.Dict(default_value={}).tag(sync=True)
    selected_panels = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    inset_plots = traitlets.List(traitlets.Dict(), default_value=[]).tag(sync=True)
    show_inset_plots = traitlets.Bool(True).tag(sync=True)
    panel_annotations = traitlets.List(traitlets.List(traitlets.Dict()), default_value=[]).tag(sync=True)
    panel_overlays = traitlets.List(traitlets.List(traitlets.Dict()), default_value=[]).tag(sync=True)
    pixel_size = traitlets.Float(0.0).tag(sync=True)
    pixel_sizes = traitlets.List(trait=traitlets.Float(), default_value=[]).tag(sync=True)
    pixel_unit = traitlets.Unicode("pixels").tag(sync=True)
    scale_bar_visible = traitlets.Bool(True).tag(sync=True)
    scale_bar_position = traitlets.Unicode("bottom-right").tag(sync=True)
    scale_bar_panels = traitlets.List(trait=traitlets.Int(), default_value=[]).tag(sync=True)
    scale_bar_length = traitlets.Float(None, allow_none=True).tag(sync=True)
    scale_bar_label = traitlets.Unicode("").tag(sync=True)
    scale_bar_style = traitlets.Dict(default_value={}).tag(sync=True)
    show_zoom_indicator = traitlets.Bool(False).tag(sync=True)
    size = traitlets.Int(0).tag(sync=True)  # Canvas rendering size in CSS pixels; 0 = frontend default
    smooth = traitlets.Bool(False).tag(sync=True)
    initial_zoom = traitlets.Float(1.0).tag(sync=True)
    zoom_row = traitlets.Float(None, allow_none=True).tag(sync=True)
    zoom_col = traitlets.Float(None, allow_none=True).tag(sync=True)
    # Live viewport (row0, row1, col0, col1) in image pixel coordinates,
    # synced from the browser on every pan/zoom (debounced ~100 ms).
    view_box = traitlets.List(trait=traitlets.Float(), default_value=[]).tag(sync=True)
    link_zoom = traitlets.Bool(False).tag(sync=True)
    link_pan = traitlets.Bool(False).tag(sync=True)
    link_contrast = traitlets.Bool(True).tag(sync=True)
    diff_mode = traitlets.Bool(False).tag(sync=True)
    diff_reference = traitlets.Int(0).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)
    controls_collapsed = traitlets.Bool(False).tag(sync=True)
    show_stats = traitlets.Bool(True).tag(sync=True)
    stats_mean = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_min = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_max = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_std = traitlets.List(traitlets.Float()).tag(sync=True)
    show_fft = traitlets.Bool(False).tag(sync=True)
    fft_window = traitlets.Bool(True).tag(sync=True)
    fft_metrics = traitlets.Bool(True).tag(sync=True)
    selected_idx = traitlets.Int(0).tag(sync=True)
    roi_active = traitlets.Bool(False).tag(sync=True)
    roi_list = traitlets.List([]).tag(sync=True)
    roi_selected_idx = traitlets.Int(-1).tag(sync=True)
    profile_line = traitlets.List(traitlets.Dict()).tag(sync=True)
    image_rotations = traitlets.List(traitlets.Int(), []).tag(sync=True)
    rotation_scope = traitlets.Enum(["all", "panel"], default_value="all").tag(sync=True)
    image_flips_horizontal = traitlets.List(traitlets.Bool(), default_value=[]).tag(sync=True)
    image_flips_vertical = traitlets.List(traitlets.Bool(), default_value=[]).tag(sync=True)

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
        if page_size is not None and (isinstance(page_size, bool) or not isinstance(page_size, (int, np.integer)) or int(page_size) < 1):
            raise ValueError(
                f"page_size must be a positive integer or None, got {page_size!r}; "
                "use None to display the folder as one unpaged gallery"
            )
        resolved_page_size = None if page_size is None else int(page_size)
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
            display_bin = kwargs.get("display_bin", 1)
            placeholder_side = (
                max(1, int(display_bin))
                if isinstance(display_bin, int) and not isinstance(display_bin, bool)
                else 1
            )
            initial_data = [np.zeros((placeholder_side, placeholder_side), dtype=np.float32)]
            initial_labels = [""]
        widget = cls(initial_data, labels=initial_labels, **kwargs)
        widget._folder_display_bin_request = kwargs.get("display_bin", 1)
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

    def __init__(
        self,
        data,
        labels: Sequence[str | Sequence[Mapping[str, object]] | None] | None = None,
        *,
        page_labels: Sequence[str | None] | None = None,
        title: str = "",
        ui_mode: UiMode = "interactive",
        cmap: str | Colormap | Sequence[str | Colormap] = Colormap.INFERNO,
        sampling: float | tuple[float, float] | list[float] | None = None,
        units: str | list[str] | None = None,
        show_scale_bar: bool | None = None,
        scale_bar_position: str = "bottom-right",
        scale_bar_panels: Sequence[int | str] | int | str | None = None,
        scale_bar_length: float | None = None,
        scale_bar_label: str | None = None,
        scale_bar_style: Mapping[str, object] | None = None,
        show_zoom_indicator: bool = False,
        show_fft: bool = False,
        show_stats: bool | None = None,
        show_controls: bool | None = None,
        controls_collapsed: bool | None = None,
        show_panel_titles: bool | None = None,
        panel_title_spans: Sequence[object] | None = None,
        panel_title_font_size: int = 11,
        panel_title_style: Mapping[str, object] | None = None,
        log_scale: bool = False,
        auto_contrast: bool = False,
        vmin: float | list | None = None,
        vmax: float | list | None = None,
        link_zoom: bool | None = None,
        link_pan: bool | None = None,
        link_contrast: bool = True,
        ncols: int = 3,
        size: int = 0,
        panel_width_px: int = 0,
        smooth: bool = False,
        zoom: float = 1.0,
        center: Sequence[float] | None = None,
        display_bin: int | str = 1,
        offline: bool = False,
        panel_frame_indices: Sequence[int] | None = None,
        panel_playback_fps: float = 10.0,
        starred: Sequence[int | str] | int | str | None = None,
        marker_colors: Sequence[str] | None = None,
        marker_style: str = "left",
        row_markers: Mapping[object, object] | None = None,
        inset_plots: Sequence[dict[str, object] | None] | dict[str, object] | None = None,
        panel_annotations: object = None,
        overlays: object = None,
        panel_overlays: object = None,
        inter_panel_gap_px: int | None = None,
        gallery_gap_px: int | None = None,
        inter_panel_gap_color: str | None = None,
        gallery_outer_border_px: int | None = None,
        gallery_outer_border_color: str | None = None,
        panel_inner_border_px: float | int | None = None,
        panel_inner_border_color: str | None = None,
        denoise: str | Sequence[str] = "none",
        denoise_sigma: float | Sequence[float] = 4.0,
        denoise_bin: int | Sequence[int] = 1,
        state: dict | str | pathlib.Path | None = None,
        save_state: bool = False,
        verbose: bool = True,
        **kwargs,
    ):
        # Reject typos and stale kwargs (e.g. image_width_px, pixel_size_angstrom).
        # anywidget/traitlets silently ignores unknown keys, which hid the
        # pixel_size_angstrom bug in show2d_all_features.ipynb for months.
        _reject_unknown_kwargs(type(self), kwargs)
        requested_panel_cmaps = (
            [cmap_to_name(item) for item in cmap]
            if is_cmap_sequence(cmap)
            else []
        )
        base_cmap = requested_panel_cmaps[0] if requested_panel_cmaps else cmap_to_name(cmap)
        plain_labels, title_spans_from_labels = _normalise_title_span_sequence(labels)
        explicit_plain_labels, explicit_title_spans = _normalise_title_span_sequence(panel_title_spans)
        if plain_labels is None and explicit_plain_labels is not None:
            plain_labels = explicit_plain_labels
        raw_title_span_len = (
            _title_span_sequence_length(panel_title_spans)
            if panel_title_spans is not None
            else (len(plain_labels or []) if title_spans_from_labels else 0)
        )
        data, labels, n_pages, panels_per_page, resolved_page_labels, resolved_page_starred = _normalise_show2d_pages(
            data,
            labels=plain_labels,
            page_labels=page_labels,
        )
        resolved_panel_title_spans = _expand_title_spans_for_flattened_labels(
            explicit_title_spans or title_spans_from_labels,
            original_len=raw_title_span_len,
            n_panels=len(labels or []),
            n_pages=n_pages,
            panels_per_page=panels_per_page,
        )
        panel_width_px = int(panel_width_px)
        if panel_width_px < 0:
            raise ValueError(f"panel_width_px must be >= 0, got {panel_width_px}")
        if panel_width_px > 0:
            size = panel_width_px
        ncols = int(ncols)
        if ncols < 1:
            raise ValueError(f"ncols must be >= 1, got {ncols}")
        if gallery_gap_px is not None:
            if inter_panel_gap_px is not None:
                raise ValueError("Use either inter_panel_gap_px= or gallery_gap_px= (its older name), not both")
            inter_panel_gap_px = gallery_gap_px
        inter_panel_gap_px = 0 if inter_panel_gap_px is None else _nonnegative_int(inter_panel_gap_px, name="inter_panel_gap_px")
        inter_panel_gap_color = "" if inter_panel_gap_color is None else str(inter_panel_gap_color)
        gallery_outer_border_px = (
            0 if gallery_outer_border_px is None
            else _nonnegative_int(gallery_outer_border_px, name="gallery_outer_border_px")
        )
        gallery_outer_border_color = (
            inter_panel_gap_color if gallery_outer_border_color is None else str(gallery_outer_border_color)
        )
        panel_inner_border_px = 1.0 if panel_inner_border_px is None else float(panel_inner_border_px)
        if not math.isfinite(panel_inner_border_px) or panel_inner_border_px < 0:
            raise ValueError(f"panel_inner_border_px must be a nonnegative number, got {panel_inner_border_px!r}")
        panel_inner_border_color = "#d0d0d0" if panel_inner_border_color is None else str(panel_inner_border_color)
        if scale_bar_position not in {"bottom-right", "bottom-left"}:
            raise ValueError(
                "scale_bar_position must be 'bottom-right' or 'bottom-left'; "
                f"got {scale_bar_position!r}"
            )
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
                "show_title": None,
                "show_controls": show_controls,
                "controls_collapsed": controls_collapsed,
                "show_stats": show_stats,
                "show_panel_titles": show_panel_titles,
                "show_scale_bar": show_scale_bar,
            },
        )
        # save_state controls whether the heavy pixel buffers are persisted into
        # the notebook's metadata.widgets on save. Default False: a plain display
        # embeds only light traits + a static image preview, so a 5-panel 4k
        # gallery does not bake ~1 GB into the .ipynb.
        self._save_state = bool(save_state)
        self._configure_static_fallback()
        super().__init__(**kwargs)
        self._static_fallback_mime = self._static_fallback_mime_type()
        row_markers = _normalize_marker_mapping(row_markers, name="row_markers")
        panel_title_style = _normalize_panel_title_style(panel_title_style)
        scale_bar_style = _normalize_scale_bar_style(scale_bar_style)
        if scale_bar_length is not None and (not np.isfinite(float(scale_bar_length)) or float(scale_bar_length) <= 0):
            raise ValueError(f"scale_bar_length must be a positive finite value, got {scale_bar_length!r}")
        n_display_panels = len(labels or []) or (int(data.shape[0]) if isinstance(data, np.ndarray) and data.ndim >= 3 else 1)
        scale_bar_panels = _normalize_panel_indices(
            scale_bar_panels,
            n_items=n_display_panels,
            labels=labels,
        )
        panel_annotations = _normalize_panel_annotations(
            panel_annotations,
            n_items=n_display_panels,
            labels=labels,
        )
        if overlays is not None and panel_overlays is not None:
            raise ValueError("Use either overlays= or panel_overlays=, not both")
        panel_overlays = _normalize_panel_overlays(
            panel_overlays if panel_overlays is not None else overlays,
            n_items=n_display_panels,
            labels=labels,
        )
        # hold_sync() batches every trait assignment into a single comm message
        # sent when the context manager exits; without it each self.x = y is a
        # separate round trip, which adds 20+ s for a 30-image gallery in VS Code.
        with self.hold_sync():
            self._verbose = verbose
            self.n_pages = int(max(1, n_pages))
            self.panels_per_page = int(max(0, panels_per_page))
            self.page_idx = 0
            self.page_labels = list(resolved_page_labels or [])
            self.page_starred = list(resolved_page_starred or [])
            # A quantem dataset (or a list of them) supplies sampling, units and
            # the title, RgbImage from read_image too; to_numpy reads the values.
            if _is_image_dataset(data):
                if not title and data.name:
                    title = data.name
                if sampling is None:
                    sampling = tuple(float(value) for value in data.sampling[-2:])
                if units is None:
                    units = list(data.units[-2:])
            elif isinstance(data, (list, tuple)) and len(data) > 0 and _is_image_dataset(data[0]):
                first = data[0]
                if sampling is None:
                    sampling = tuple(float(value) for value in first.sampling[-2:])
                if units is None:
                    units = list(first.units[-2:])
            # RGB detection rule: per-ITEM in a list input, an item with ndim == 3
            # and shape[-1] in (3, 4) is an RGB(A) image. A single non-list ndim==3
            # input is RGB only when shape[-1] in (3, 4) AND shape[0] > 4;
            # otherwise it keeps the historical (N, H, W) stack semantics.
            rgb_flags: list[bool] = []
            rgb_frames: list[np.ndarray | None] = []
            panel_stacks: list[np.ndarray] | None = None
            resolved_panel_frame_indices: list[int] | None = None
            if isinstance(data, (list, tuple)):
                images = [to_numpy(item) for item in data]
                rgb_flags = [_is_rgb_item(image) for image in images]
            else:
                array = to_numpy(data)
                if array.ndim == 3 and array.shape[-1] in (3, 4) and array.shape[0] > 4:
                    images, rgb_flags = [array], [True]
                else:
                    images, rgb_flags = None, []
                    data = array
            if images is not None and any(rgb_flags):
                stack_panels = [
                    index for index, (image, is_rgb) in enumerate(zip(images, rgb_flags))
                    if not is_rgb and image.ndim == 3
                ]
                if stack_panels:
                    raise NotImplementedError(
                        "Show2D does not yet mix RGB panels with local grayscale frame "
                        f"stacks (stack panel indices: {stack_panels}). Convert the RGB "
                        "panel to grayscale or use a separate Show2D."
                    )
                # Mixed gallery: RGB(A) items become display-ready (H, W, 3) in
                # [0, 1] (uint8 scales by 1/255, float is clipped; alpha is dropped
                # since panels composite on an opaque canvas and RGB bypasses the
                # contrast pipeline). Their Rec. 709 luminance feeds the grayscale
                # machinery: stats, histogram, FFT and profile all read that plane.
                normalized = [
                    (image[..., :3].astype(np.float32) / 255.0 if image.dtype == np.uint8 else np.clip(image[..., :3].astype(np.float32), 0.0, 1.0))
                    if flag else np.asarray(image, dtype=np.float32)
                    for image, flag in zip(images, rgb_flags)
                ]
                shapes = [image.shape[:2] for image in normalized]
                if len(set(shapes)) > 1:
                    max_h = max(shape[0] for shape in shapes)
                    max_w = max(shape[1] for shape in shapes)
                    normalized = [
                        _pad_rgb(image, max_h, max_w) if flag else _resize_image(image, max_h, max_w)
                        for image, flag in zip(normalized, rgb_flags)
                    ]
                rgb_frames = [image if flag else None for image, flag in zip(normalized, rgb_flags)]
                data = np.stack([image @ _RGB_LUMA if flag else image
                                 for image, flag in zip(normalized, rgb_flags)])
            elif images is not None:
                panel_stacks, resolved_panel_frame_indices, data = _normalize_grayscale_panel_items(
                    images,
                    panel_frame_indices,
                )
            if data.ndim == 2:
                data = data[np.newaxis, ...]
            # The widget owns its pixels: a copy, so later in-place edits of the
            # caller's array never desync the stats row from the display.
            self._data = np.array(data, dtype=np.float32, copy=True)
            self._rgb_frames = rgb_frames or [None] * int(data.shape[0])
            self._rgb_frames_original = list(self._rgb_frames)
            self.is_rgb = [frame is not None for frame in self._rgb_frames]
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
                    for panel, raw_index in enumerate(panel_frame_indices):
                        if int(raw_index) not in (0, -1):
                            raise ValueError(
                                f"panel_frame_indices[{panel}]={raw_index} is outside the valid range [0, 1)"
                            )
                    resolved_panel_frame_indices = [0] * len(panel_stacks)
            if offline and any(self.is_rgb):
                raise NotImplementedError(
                    "offline=True is not supported for RGB panels; RGB frames are "
                    "sent as full float32 and bypass the uint8 quantization path."
                )
            # Originals for rotation reset: views into _data, materialized as
            # copies only when a rotation is applied.
            self._data_original = list(self._data)
            self._originals_are_views = True
            self.n_images = int(data.shape[0])
            self._panel_stacks = panel_stacks
            self._panel_stacks_original = list(panel_stacks)
            self._panel_stack_originals_are_views = True
            # The frame observer would rebuild _data before the display copy exists.
            self._updating_panel_frames = True
            self.panel_frame_counts = [int(stack.shape[0]) for stack in panel_stacks]
            self.panel_frame_indices = list(resolved_panel_frame_indices or [0] * self.n_images)
            self._updating_panel_frames = False
            self.panel_stack_offsets = [-1] * self.n_images
            self.inset_plots = _normalize_inset_plot_specs(inset_plots, n_items=self.n_images)
            self.panel_annotations = list(panel_annotations or [])
            self.panel_overlays = list(panel_overlays or [])
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
            self.labels = [f"Image {i + 1}" for i in range(self.n_images)] if labels is None else list(labels)
            self.panel_title_spans = resolved_panel_title_spans if len(resolved_panel_title_spans) == self.n_images else []
            self.starred = [0] * self.n_images
            self.hidden_panels = []
            self.hidden_page_slots = []
            self.show_panel_titles = bool(ui["show_panel_titles"])
            self.panel_title_font_size = int(panel_title_font_size)
            self.panel_title_style = dict(panel_title_style)
            self.inter_panel_gap_px = inter_panel_gap_px
            self.inter_panel_gap_color = inter_panel_gap_color
            self.gallery_outer_border_px = gallery_outer_border_px
            self.gallery_outer_border_color = gallery_outer_border_color
            self.panel_inner_border_px = panel_inner_border_px
            self.panel_inner_border_color = panel_inner_border_color
            if starred is not None:
                flags = [0] * self.n_images
                for panel in self._normalize_panel_refs(starred, allow_empty=True):
                    flags[panel] = 1
                self.starred = flags
            self.title = title
            self.show_title = bool(ui["show_title"])
            self.cmap = base_cmap
            if requested_panel_cmaps:
                cmaps = list(requested_panel_cmaps)
                if len(cmaps) == 1:
                    cmaps = cmaps * self.n_images
                elif len(cmaps) != self.n_images:
                    raise ValueError(
                        f"cmap sequence length ({len(cmaps)}) must be 1 or match "
                        f"the number of Show2D panels ({self.n_images})"
                    )
                self.panel_cmaps = cmaps
                self.panel_cmaps_memory = list(cmaps)
            else:
                self.panel_cmaps = []
                self.panel_cmaps_memory = []
            # sampling / units resolve to the column-axis pixel_size + pixel_unit.
            # Scalar shorthand: sampling=0.5 -> (0.5, 0.5); units="nm" -> ["nm", "nm"].
            # None keeps the trait: pixel_size= may have been set directly via kwargs.
            if isinstance(sampling, (int, float)):
                self.pixel_size = float(sampling)
            elif sampling is not None:
                self.pixel_size = float(sampling[-1])
            if isinstance(units, str):
                self.pixel_unit = units
            elif units is not None:
                self.pixel_unit = str(units[-1])
            self.scale_bar_visible = bool(ui["show_scale_bar"])
            self.scale_bar_position = scale_bar_position
            self.scale_bar_panels = list(scale_bar_panels)
            self.scale_bar_length = None if scale_bar_length is None else float(scale_bar_length)
            self.scale_bar_label = "" if scale_bar_label is None else str(scale_bar_label)
            self.scale_bar_style = dict(scale_bar_style)
            self.show_zoom_indicator = bool(show_zoom_indicator)
            self.pixel_sizes = []
            self.size = size
            self.smooth = smooth
            colors = [] if marker_colors is None else [str(value) for value in marker_colors]
            if colors and len(colors) != self.n_images:
                raise ValueError(
                    f"marker_colors length ({len(colors)}) must match "
                    f"the number of Show2D panels ({self.n_images})"
                )
            self.marker_colors = colors
            self.marker_style = str(marker_style).lower()
            self.row_markers = dict(row_markers or {})
            self.selected_panels = []
            self.initial_zoom = zoom
            if center is not None:
                # center=(row, col): where the zoomed view looks, in full-resolution pixels.
                self.zoom_row, self.zoom_col = float(center[0]), float(center[1])
            self.image_flips_horizontal = [False] * self.n_images
            self.image_flips_vertical = [False] * self.n_images
            # Auto-link zoom + pan in a gallery so dragging one panel follows the
            # others, the typical compare workflow. Single image: no-op.
            self.link_zoom = (self.n_images >= 2) if link_zoom is None else link_zoom
            self.link_pan = (self.n_images >= 2) if link_pan is None else link_pan
            self.link_contrast = link_contrast
            if show_fft and self.height * self.width > 2048 * 2048:
                warnings.warn(
                    f"FFT on {self.height}×{self.width} image ({self.height * self.width / 1e6:.1f}M pixels) "
                    f"may be slow. Consider using ROI FFT for a sub-region.",
                    stacklevel=2,
                )
            self.show_fft = show_fft
            self.show_controls = bool(ui["show_controls"])
            self.controls_collapsed = bool(ui["controls_collapsed"])
            self.show_stats = bool(ui["show_stats"])
            self.log_scale = log_scale
            self.auto_contrast = auto_contrast
            self.offline = offline
            # Standalone HTML export packs panels through the grayscale stack
            # machinery, which cannot carry (H, W, 3) pixels: hide the export menu
            # and reject programmatic export instead of corrupting RGB panels.
            if any(self.is_rgb):
                self.export_enabled = False
            # Scalar OR list for vmin/vmax. List -> per-image (vmins/vmaxs).
            if isinstance(vmin, (list, tuple)) or isinstance(vmax, (list, tuple)):
                n_images = self.n_images

                def expand(value):
                    """One contrast limit per panel from a scalar, a per-panel list, or None (unset)."""
                    if value is None:
                        return [None] * n_images
                    if isinstance(value, (list, tuple)):
                        if len(value) != n_images:
                            raise ValueError(
                                f"vmin/vmax list has length {len(value)} but n_images is {n_images}. "
                                f"Pass a list of length {n_images} or a scalar to apply uniformly."
                            )
                        return [None if limit is None else float(limit) for limit in value]
                    return [float(value)] * n_images

                self.vmins = expand(vmin)
                self.vmaxs = expand(vmax)
                self.vmin = None
                self.vmax = None
            else:
                self.vmin = vmin
                self.vmax = vmax
            self.ncols = ncols
            self.panel_playback_fps = panel_playback_fps
            # Browser display denoise: a scalar knob applies to every panel, a
            # sequence gives each panel its own (a raw vs filtered A/B gallery).
            self.denoise_modes = _per_panel(denoise, "denoise", _denoise_mode, self.n_images)
            self.denoise_sigmas = _per_panel(denoise_sigma, "denoise_sigma", float, self.n_images)
            self.denoise_bins = _per_panel(denoise_bin, "denoise_bin", int, self.n_images)
            self.denoise = self.denoise_modes[0]
            self.denoise_sigma = self.denoise_sigmas[0]
            self.denoise_bin = self.denoise_bins[0]
            if any(isinstance(knob, (list, tuple, np.ndarray)) for knob in (denoise, denoise_sigma, denoise_bin)):
                if kwargs.get("denoise_scope") == "all":
                    raise ValueError(
                        "denoise_scope='all' applies one setting to every panel, but a per-panel "
                        "denoise/denoise_sigma/denoise_bin sequence was given"
                    )
                self.denoise_scope = "panel"
            # The denoise switch starts on, and its controls show, exactly when a filter was asked for.
            self.denoise_enabled = any(mode != "none" or bin_ > 1 for mode, bin_ in zip(self.denoise_modes, self.denoise_bins))
            self.show_denoise = self.show_denoise or self.denoise_enabled
            self.frequency_filter_modes = ["none"] * self.n_images
            self.frequency_filter_cutoffs = [0.15] * self.n_images
            self.frequency_filter_centers = [0.30] * self.n_images
            self.frequency_filter_widths = [0.12] * self.n_images
            # A scientist comparing panels adjusts one result at a time, so
            # per-panel filter edits are the gallery default.
            self.frequency_filter_scope = "panel" if self.n_images > 1 else "all"

            # Keep full-res in _data; send a binned preview only when asked
            # (an integer, or "auto" over the GPU / wire budget).
            if display_bin == "auto":
                self._display_bin = self._auto_display_bin(self.n_images, self.height, self.width)
            elif isinstance(display_bin, int) and display_bin > 1:
                self._display_bin = display_bin
            if self._display_bin > 1:
                full_h, full_w = self._data.shape[1], self._data.shape[2]
                self._display_data = bin2d(self._data, factor=self._display_bin, mode="mean")
                self.height = int(self._display_data.shape[1])
                self.width = int(self._display_data.shape[2])
                if self.pixel_size > 0:
                    self.pixel_size = self.pixel_size * self._display_bin
                # center= / zoom_row / zoom_col are full-resolution pixels; the
                # browser reads them in preview pixels once the display is binned.
                if self.zoom_row is not None:
                    self.zoom_row = self.zoom_row / self._display_bin
                if self.zoom_col is not None:
                    self.zoom_col = self.zoom_col / self._display_bin
                self._display_bin_factor = self._display_bin
                # JS derives every panel's offset from one H x W.
                self._display_rgb = _binned_rgb(self._rgb_frames, self._display_bin)
                self._display_panel_stacks = self._binned_panel_stacks(self._panel_stacks)
                # no surprise binning: announce the reduction and the way out
                reason = "auto, over the display budget" if display_bin == "auto" else "display_bin"
                preview_size = f", {self._display_data.nbytes // 1024 // 1024} MB preview" if verbose else ""
                print(f"Show2D display bin {self._display_bin}x ({reason}): {full_h}x{full_w} -> {self.height}x{self.width}"
                      f"{preview_size}; full-res detail streams on zoom; pass display_bin=1 for native pixels")
            else:
                self._display_data = self._data
                self._display_bin_factor = 1
                self._display_rgb = self._rgb_frames
                self._display_panel_stacks = self._panel_stacks
            self._compute_all_stats()
            self._update_all_frames()
            announce_browser_limit("Show2D", len(self.frame_bytes) + len(self.panel_stack_bytes), self._display_bin_factor)
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
        self.observe(self._on_first_render, names=["_js_rendered"])
        self.observe(self._on_export_request_change, names=["export_request"])
        self.observe(self._on_detail_request_change, names=["_detail_request"])

    @traitlets.validate("starred")
    def _validate_starred(self, proposal: dict) -> list[int]:
        """Normalize per-image star flags."""
        flags = list(proposal["value"])
        n_images = int(self.n_images)
        if not flags:
            return [0] * n_images
        if len(flags) != n_images:
            raise traitlets.TraitError(
                f"starred length ({len(flags)}) must equal n_images ({n_images})"
            )
        return [1 if int(flag) else 0 for flag in flags]

    @traitlets.validate("panel_frame_indices")
    def _validate_panel_frame_indices(self, proposal: dict) -> list[int]:
        """Validate one independent frame index for every source panel."""
        values = list(proposal["value"])
        counts = list(self.panel_frame_counts)
        n_images = int(self.n_images)
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
        if self._updating_panel_frames:
            return
        stacks = self._panel_stacks
        if not stacks or len(stacks) != int(self.n_images):
            return
        indices = list(change["new"])
        if len(indices) != len(stacks):
            return
        self._data = _current_frames(stacks, indices)
        display_stacks = self._display_panel_stacks
        if display_stacks and len(display_stacks) == len(stacks):
            self._display_data = _current_frames(display_stacks, indices)
        elif int(self._display_bin) <= 1:
            self._display_data = self._data
        self._compute_all_stats()
        self._clear_detail_tiles()

    def _clear_detail_tiles(self) -> None:
        """Drop the streamed full-resolution tiles: they show the previous frame or image and would paint stale pixels."""
        self._detail_request = ""
        self._detail_meta = ""
        self._detail_bytes = b""

    def _binned_panel_stacks(self, stacks: list[np.ndarray]) -> list[np.ndarray]:
        """Panel stacks mean-binned by the display bin for the browser preview; the same list when unbinned."""
        if self._display_bin > 1:
            return [bin2d(stack, factor=self._display_bin, mode="mean") for stack in stacks]
        return stacks

    @traitlets.validate("hidden_panels")
    def _validate_hidden_panels(self, proposal: dict) -> list[int]:
        """Normalize hidden image indices and keep at least one image visible."""
        n_images = int(self.n_images)
        if n_images <= 0:
            return []
        hidden = sorted({int(value) for value in proposal["value"] if 0 <= int(value) < n_images})
        if len(hidden) >= n_images:
            raise traitlets.TraitError(
                "hidden_panels cannot hide every panel; at least one panel must remain visible"
            )
        return self._normalize_item_page_hidden(hidden)

    def _panel_count(self) -> int:
        """Number of panels the shared panel-reference helpers resolve against."""
        return int(self.n_images)

    def _uses_item_pages(self) -> bool:
        """Whether pages hold distinct files (a folder gallery) rather than one comparison layout repeated per page."""
        return str(self.page_kind) == "items"

    def _panel_title_for_index(self, panel: int) -> str:
        """Return the user-facing label for a Show2D image panel."""
        if 0 <= panel < len(self.labels) and self.labels[panel]:
            return str(self.labels[panel])
        return f"Image {panel + 1}"

    def _on_first_render(self, change: dict) -> None:
        """After the browser painted, drop the synced pixel buffers from the
        model so a later notebook save stores the static preview, not the
        full array; the targeted initial sync already delivered them."""
        if not change.get("new"):
            return
        if not self._save_state and (self.frame_bytes or self.panel_stack_bytes):
            self.frame_bytes = b""
            self.panel_stack_bytes = b""
        self.unobserve(self._on_first_render, names=["_js_rendered"])

    def _panel_display_order(self) -> list[int]:
        """``panel_order`` when it is a complete permutation of the panels, else the natural order.

        A partial or stale order (files just arrived) must never drop or
        duplicate a panel, so anything but a full permutation is ignored.
        """
        n_images = int(self.n_images)
        order = list(self.panel_order or [])
        if len(order) == n_images and sorted(order) == list(range(n_images)):
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
        if int(self.n_pages) <= 1 or int(self.panels_per_page) <= 0 or not self._uses_item_pages():
            return sorted(hidden)
        order = self._panel_display_order()
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

    def _sync_folder_pages(
        self,
        *,
        anchor_panel: int | None = None,
        preferred_page: int | None = None,
    ) -> None:
        """Recompute sequential folder pages without rebuilding the widget."""
        page_size = self._folder_page_size
        n_images = int(self.n_images)
        old_page = int(self.page_idx) if preferred_page is None else int(preferred_page)
        old_starred = list(self.page_starred or [])

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
            order = self._panel_display_order()
            if anchor_panel is not None and int(anchor_panel) in order:
                next_page = min(n_pages - 1, order.index(int(anchor_panel)) // panels_per_page)

        next_starred = [1 if index < len(old_starred) and int(old_starred[index]) else 0 for index in range(n_pages)]

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
            self._rgb_frames_original = []
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

        This updates a live image selection. It preserves display controls such as colormap, contrast,
        FFT/profile toggles, and gallery layout, while resetting per-panel state
        tied to the previous image count or dimensions.
        """
        if isinstance(data, (list, tuple)):
            images = [to_numpy(item) for item in data]
            rgb_panels = [index for index, image in enumerate(images) if _is_rgb_item(image)]
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
            # A float32 copy, as in __init__: the widget owns its pixels, so a
            # caller reusing its buffer cannot desync the stats from the display.
            data = np.array(data, dtype=np.float32, copy=True)
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
                    list(data),
                    panel_frame_indices,
                )

        previous_shape = (int(self.n_images), int(self.height), int(self.width))
        self._updating_panel_frames = True
        try:
            with self.hold_sync():
                self._data = data
                self._panel_stacks = panel_stacks
                self._panel_stacks_original = list(panel_stacks)
                self._panel_stack_originals_are_views = True
                self.panel_frame_counts = [int(stack.shape[0]) for stack in panel_stacks]
                self.n_images = int(data.shape[0])
                self.panel_frame_indices = list(resolved_indices)
                self.panel_stack_offsets = [-1] * self.n_images
                self._data_original = list(self._data)
                self._originals_are_views = True
                self._rgb_frames = [None] * int(data.shape[0])
                self._rgb_frames_original = list(self._rgb_frames)
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
                self._clear_detail_tiles()
                self.vmins = None
                self.vmaxs = None

                self._display_bin = max(1, int(self._display_bin))
                self._display_panel_stacks = self._binned_panel_stacks(self._panel_stacks)
                self._display_data = _current_frames(self._display_panel_stacks, resolved_indices)
                self.height = int(self._display_data.shape[1])
                self.width = int(self._display_data.shape[2])
                self._display_bin_factor = self._display_bin
                if (self.n_images, self.height, self.width) != previous_shape:
                    self.view_box = []
                    self.zoom_row = None
                    self.zoom_col = None
                self._compute_all_stats()
                self._update_all_frames()
                announce_browser_limit("Show2D", len(self.frame_bytes) + len(self.panel_stack_bytes), self._display_bin_factor)
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
        old_page_idx = int(self.page_idx)
        old_index = {path: index for index, path in enumerate(old_paths)}
        new_index = {path: index for index, path in enumerate(new_paths)}
        old_display_bin = max(1, int(self._display_bin))
        if self._folder_display_bin_request == "auto":
            sample_image = changed_arrays.get(new_paths[0])
            if sample_image is None and old_paths:
                sample_image = np.asarray(self._panel_stacks_original[old_index[new_paths[0]]][0])
            if sample_image is not None:
                self._display_bin = self._auto_display_bin(
                    len(new_paths),
                    int(sample_image.shape[-2]),
                    int(sample_image.shape[-1]),
                )
        new_display_bin = max(1, int(self._display_bin))
        if new_display_bin != old_display_bin and new_display_bin > 1:
            print(f"Show2D display bin {new_display_bin}x (auto, over the display budget); pass display_bin=1 for native pixels")
        coordinate_scale = old_display_bin / new_display_bin

        if not old_paths:
            self.set_image(
                [changed_arrays[path] for path in new_paths],
                labels=[self._folder_source.label(path) for path in new_paths],
            )
            self._sync_folder_pages(anchor_panel=0)
            return

        # Unchanged files keep the pixels already loaded (frame 0 of their original stack).
        arrays = [
            changed_arrays[path] if path in changed_arrays else np.asarray(self._panel_stacks_original[old_index[path]][0])
            for path in new_paths
        ]
        selected_path = (
            old_paths[int(self.selected_idx)]
            if 0 <= int(self.selected_idx) < len(old_paths)
            else old_paths[0]
        )
        starred_by_path = {
            path: bool(self.starred[index])
            for index, path in enumerate(old_paths)
            if index < len(self.starred)
        }
        hidden_paths = {
            old_paths[index]
            for index in self.hidden_panels
            if 0 <= int(index) < len(old_paths)
        }
        rotations_by_path = {
            path: int(self.image_rotations[index])
            for index, path in enumerate(old_paths)
            if index < len(self.image_rotations)
        }

        ordered_paths: list[pathlib.Path] = []
        if self.panel_order:
            ordered_paths.extend(
                old_paths[index]
                for index in self.panel_order
                if 0 <= int(index) < len(old_paths)
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
            """Per-panel values (e.g. ``vmins``) re-keyed from the old file order to the new; None when they do not fit."""
            if values is None or len(values) != len(old_paths):
                return None
            by_path = {path: values[index] for index, path in enumerate(old_paths)}
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

    def _auto_display_bin(self, n_images: int, height: int, width: int) -> int:
        """Preview bin factor for ``display_bin="auto"``: the smallest of 1/2/4/8
        that keeps the gallery inside the GPU display budget, raised further when
        one float32 panel exceeds the wire budget so the first paint ships a
        ~1024 px preview and full-res detail streams on zoom."""
        factor = 1
        per_image_mb = (height * width * 4 * 3) / (1024 * 1024)  # float32 + RGBA + readback
        total_mb = int(n_images) * per_image_mb
        if total_mb > self._DISPLAY_BUFFER_BUDGET_MB:
            for candidate in (2, 4, 8):
                if total_mb / (candidate * candidate) <= self._DISPLAY_BUFFER_BUDGET_MB:
                    factor = candidate
                    break
            else:
                factor = 8
        panel_bytes = height * width * 4
        if panel_bytes > self._WIRE_BUDGET_BYTES_PER_PANEL:
            preview_floor_px = 512.0 if int(n_images) >= 16 else 1024.0
            canvas_css_px = float(self.size) if int(self.size) > 0 else 300.0 if int(n_images) > 1 else 500.0
            preview_px = max(preview_floor_px, 2.0 * canvas_css_px)
            wire_bin = max(2, math.ceil(max(height, width) / preview_px))
            while panel_bytes / (wire_bin * wire_bin) > self._WIRE_BUDGET_BYTES_PER_PANEL:
                wire_bin += 1
            factor = max(factor, wire_bin)
        return factor

    def __repr__(self) -> str:
        """Shape and colormap at a glance, e.g. ``Show2D(3×512×512, idx=0, cmap=gray)``."""
        if self.n_images > 1:
            shape = f"{self.n_images}×{self.height}×{self.width}"
            return f"Show2D({shape}, idx={self.selected_idx}, cmap={self.cmap})"
        return f"Show2D({self.height}×{self.width}, cmap={self.cmap})"

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
        path = pathlib.Path(path)
        image_format = (format or path.suffix.lstrip(".").lower() or "png").lower()
        if image_format not in ("png", "pdf", "tiff", "tif"):
            raise ValueError(f"Unsupported format: {image_format!r}. Use 'png', 'pdf', or 'tiff'.")

        panel_index = idx if idx is not None else self.selected_idx
        if panel_index < 0 or panel_index >= self.n_images:
            raise IndexError(f"Image index {panel_index} out of range [0, {self.n_images})")

        panel_rgb = self._rgb_frames[panel_index]
        frame = self._data[panel_index]
        colormap = colormaps.get_cmap(self.cmap)
        path.parent.mkdir(parents=True, exist_ok=True)
        if panel_rgb is not None:
            # RGB panels are display-ready: save the color pixels directly,
            # bypassing colormap and contrast. Colorbar is meaningless for an
            # RGB composite, so it is skipped.
            display_pixels = (np.clip(panel_rgb, 0.0, 1.0) * 255).astype(np.uint8)
            colorbar = False
        else:
            # Same contrast window as the browser: explicit vmin/vmax, else the
            # 2/98 percentiles under auto_contrast, else the frame extrema.
            display_frame = np.log1p(np.maximum(frame, 0)) if self.log_scale else frame
            if self.vmin is not None and self.vmax is not None:
                display_min, display_max = float(self.vmin), float(self.vmax)
                if self.log_scale:
                    display_min, display_max = float(np.log1p(max(display_min, 0))), float(np.log1p(max(display_max, 0)))
            elif self.auto_contrast:
                display_min, display_max = (float(value) for value in np.percentile(display_frame, (2, 98)))
            else:
                display_min, display_max = float(display_frame.min()), float(display_frame.max())
            # 8-bit levels: the colormap lookup and the PNG are both 256-level.
            if display_max > display_min:
                levels = np.clip((display_frame - display_min) / (display_max - display_min) * 255, 0, 255).astype(np.uint8)
            else:
                levels = np.zeros(frame.shape, dtype=np.uint8)
            display_pixels = None

        use_figure = title or colorbar or scalebar
        if not use_figure:
            if panel_rgb is not None:
                image = Image.fromarray(display_pixels, mode="RGB")
            else:
                rgba = (colormap(levels / 255.0) * 255).astype(np.uint8)
                image = Image.fromarray(rgba)
                if image_format == "pdf":
                    # PDF has no alpha channel.
                    image = image.convert("RGB")
            image.save(str(path), dpi=(dpi, dpi))
            return path

        height, width = frame.shape
        figure_width_in = 6
        fig, ax = plt.subplots(figsize=(figure_width_in, figure_width_in * (height / width)))
        if panel_rgb is not None:
            image_artist = ax.imshow(display_pixels, origin="upper")
        else:
            image_artist = ax.imshow(levels, cmap=colormap, vmin=0, vmax=255, origin="upper")
        ax.axis("off")

        if title:
            label = title if isinstance(title, str) else self.title
            if label:
                ax.set_title(label, fontsize=14, fontweight="bold", pad=8)

        if colorbar:
            # The image holds 0-255 levels; ticks read back in data values.
            colorbar_artist = fig.colorbar(image_artist, ax=ax, fraction=0.046, pad=0.04)
            tick_positions = np.linspace(0, 255, 5)
            tick_labels = [f"{display_min + (display_max - display_min) * tick / 255:.4g}" for tick in tick_positions]
            colorbar_artist.set_ticks(tick_positions)
            colorbar_artist.set_ticklabels(tick_labels)

        if scalebar and self.pixel_size > 0:
            # A nice physical length near 20% of the image width.
            nice = round_to_nice(0.2 * width * self.pixel_size)
            bar_px = nice / self.pixel_size
            label_text = format_scale_label(nice, self.pixel_unit)
            margin = 0.03
            bar_y = height * (1 - margin) - 2
            bar_x = width * (1 - margin) - bar_px
            ax.plot([bar_x, bar_x + bar_px], [bar_y, bar_y],
                    color="white", linewidth=3, solid_capstyle="butt")
            ax.plot([bar_x, bar_x + bar_px], [bar_y, bar_y],
                    color="black", linewidth=1, solid_capstyle="butt")
            ax.text(bar_x + bar_px / 2, bar_y - height * 0.02, label_text,
                    color="white", fontsize=10, fontweight="bold",
                    ha="center", va="bottom",
                    path_effects=[
                        matplotlib.patheffects.withStroke(linewidth=2, foreground="black")
                    ])

        fig.savefig(str(path), dpi=dpi, bbox_inches="tight",
                    facecolor="white", pad_inches=0.1)
        plt.close(fig)
        return path

    def _on_detail_request_change(self, change: dict) -> None:
        """Serve a maps-style detail request: crop the visible window from the
        FULL-resolution data, mean-bin it to near canvas resolution, and reply
        with a small float32 buffer. This is how a binned preview still shows
        true full-res pixels under zoom: the browser swaps the tile in over
        the preview, and the 100+ MB full frame never crosses the wire.
        Coordinates arrive in preview pixels (the JS-side image space) and are
        scaled back to full resolution here; the reply reports full-res
        coordinates plus the tile bin so JS can place it exactly."""
        request_text = str(change.get("new") or "")
        if not request_text:
            return
        try:
            request = json.loads(request_text)
        except json.JSONDecodeError:
            return
        factor = max(1, int(self._display_bin_factor))
        full_h, full_w = int(self._data.shape[1]), int(self._data.shape[2])
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
            row0 = max(0, math.floor(float(spec["row0"]) * factor / bin_factor) * bin_factor)
            col0 = max(0, math.floor(float(spec["col0"]) * factor / bin_factor) * bin_factor)
            row1 = min(full_h, math.ceil(float(spec["row1"]) * factor / bin_factor) * bin_factor)
            col1 = min(full_w, math.ceil(float(spec["col1"]) * factor / bin_factor) * bin_factor)
            if row1 <= row0 or col1 <= col0:
                continue
            crop = self._data[panel, row0:row1, col0:col1]  # view: never copies the full array
            tile = bin2d(crop, factor=bin_factor, mode="mean") if bin_factor > 1 else crop
            tile = np.ascontiguousarray(tile, dtype=np.float32)
            blocks.append(tile.tobytes())
            tiles.append({"panel": panel, "row0": row0, "col0": col0,
                          "rows": int(tile.shape[0]), "cols": int(tile.shape[1]),
                          "bin": bin_factor, "offset": offset})
            offset += tile.nbytes
        # One comm message for bytes + meta so JS never pairs a fresh meta with
        # a stale buffer (or vice versa) mid-update.
        with self.hold_sync():
            self._detail_bytes = _b64_safe(b"".join(blocks))
            self._detail_meta = json.dumps({"id": str(request.get("id", "")), "tiles": tiles})

    def _ordered_panels(self) -> list[int]:
        """Zero-based image panel indices in the current display order."""
        if int(self.n_pages) > 1 and int(self.panels_per_page) > 0:
            start = int(self.page_idx) * int(self.panels_per_page)
            stop = min(start + int(self.panels_per_page), int(self.n_images))
            if self._uses_item_pages():
                return self._panel_display_order()[start:stop]
            return list(range(start, stop))
        return self._panel_display_order()

    def _visible_panels(self) -> list[int]:
        """Zero-based image panel indices currently visible in the gallery."""
        if int(self.n_pages) > 1 and int(self.panels_per_page) > 0 and not self._uses_item_pages():
            # Comparison pages hide a slot (the same position on every page), not one image.
            hidden_slots = set(
                self.hidden_page_slots
                or self._hidden_page_slots_from_panels(self.hidden_panels, drop_if_full=True)
            )
            per_page = int(self.panels_per_page)
            return [panel for panel in self._ordered_panels() if (panel % per_page) not in hidden_slots]
        hidden = set(self.hidden_panels)
        return [panel for panel in self._ordered_panels() if panel not in hidden]

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

    def _compute_all_stats(self):
        """Compute statistics for all images (vectorized over all frames)."""
        # Vectorized reduction over (H, W) is faster than per-image loops
        # for large galleries (e.g. 12×4096×4096: 164ms vs 191ms).
        self.stats_mean = np.mean(self._data, axis=(1, 2)).tolist()
        self.stats_min = np.min(self._data, axis=(1, 2)).tolist()
        self.stats_max = np.max(self._data, axis=(1, 2)).tolist()
        self.stats_std = np.std(self._data, axis=(1, 2)).tolist()

    def _update_all_frames(self):
        """Send display data to JS (possibly binned for large galleries)."""
        data = self._display_data
        if any(self.is_rgb):
            # Mixed packing: each panel is one contiguous float32 block, W*H
            # floats for grayscale, 3*W*H interleaved floats for RGB. JS derives
            # per-panel offsets from the synced is_rgb flags.
            blocks = [
                np.ascontiguousarray(self._display_rgb[panel] if self.is_rgb[panel] else data[panel], dtype=np.float32).tobytes()
                for panel in range(int(data.shape[0]))
            ]
            self.frame_bytes = _b64_safe(b"".join(blocks))
        elif self.offline:
            # uint8 per image (4x smaller than float32), display-only: the
            # colormap reduces to 256 levels anyway.
            values = np.ascontiguousarray(data, dtype=np.float32).reshape(data.shape[0], -1)
            codes = np.empty(values.shape, dtype=np.uint8)
            mins, maxs = [], []
            for panel, panel_values in enumerate(values):
                codes[panel], low, high = _quantize_uint8(panel_values)
                mins.append(low)
                maxs.append(high)
            self._offline_mins = mins
            self._offline_maxs = maxs
            self._offline_min = mins[0]  # back-compat scalars (single-image readers)
            self._offline_max = maxs[0]
            self.frame_bytes = _b64_safe(codes.tobytes())
        else:
            self.frame_bytes = _b64_safe(data.tobytes())
        # Multi-frame panels also ship their whole stack so the browser scrubs
        # frames locally; static panels keep offset -1 and are not duplicated.
        stacks = self._display_panel_stacks
        if not stacks or not any(int(stack.shape[0]) > 1 for stack in stacks):
            self.panel_stack_offsets = [-1] * int(self.n_images)
            self.panel_stack_bytes = b""
            self._panel_stack_mins = []
            self._panel_stack_maxs = []
            return
        offsets: list[int] = []
        blocks: list[bytes] = []
        mins = [0.0] * len(stacks)
        maxs = [1.0] * len(stacks)
        float_offset = 0
        for panel, stack in enumerate(stacks):
            if int(stack.shape[0]) <= 1:
                offsets.append(-1)
                continue
            stack_values = np.ascontiguousarray(stack, dtype=np.float32)
            offsets.append(float_offset)
            float_offset += int(stack_values.size)
            if self.offline:
                codes, mins[panel], maxs[panel] = _quantize_uint8(stack_values)
                blocks.append(codes.tobytes())
            else:
                blocks.append(stack_values.tobytes())
        self.panel_stack_offsets = offsets
        self._panel_stack_mins = mins if self.offline else []
        self._panel_stack_maxs = maxs if self.offline else []
        self.panel_stack_bytes = _b64_safe(b"".join(blocks))

    def _apply_rotations(self):
        """Re-rotate each displayed image from its original by ``image_rotations[i] * 90°``.

        This is purely a display-time reorientation of each 2D image via
        ``np.rot90``: it is NOT scan rotation (which would rotate the
        scan grid in a 4D-STEM dataset). Originals are kept in
        ``_data_original`` so successive rotations compose from the
        unrotated source rather than accumulating interpolation error.
        Mixed shapes after rotation are center-padded to a common size.
        """
        if any(count > 1 for count in self.panel_frame_counts):
            quarter_turns = self._quarter_turns(len(self._panel_stacks_original))
            if not any(quarter_turns) and self._panel_stack_originals_are_views:
                return
            if self._panel_stack_originals_are_views:
                self._panel_stacks_original = [stack.copy() for stack in self._panel_stacks_original]
                self._panel_stack_originals_are_views = False
            normalized_stacks = _pad_to_common_shape([
                original if turns == 0 else np.rot90(original, k=turns, axes=(-2, -1))
                for original, turns in zip(self._panel_stacks_original, quarter_turns)
            ])
            self._panel_stacks = normalized_stacks
            indices = list(self.panel_frame_indices)
            self._data = _current_frames(normalized_stacks, indices)
            self._display_panel_stacks = self._binned_panel_stacks(normalized_stacks)
            self._display_data = _current_frames(self._display_panel_stacks, indices)
            self._data_original = list(self._data)
            self._originals_are_views = True
            self.height = int(self._display_data.shape[1])
            self.width = int(self._display_data.shape[2])
            self._compute_all_stats()
            self._update_all_frames()
            return

        quarter_turns = self._quarter_turns(len(self._data_original))
        # No-rotation fast path: skip 30+ MB of redundant tobytes + stats recomputation
        # on every widget init.  The observer fires once when image_rotations = [0]*n
        # is assigned in __init__; without this guard that triggered a full frame
        # rebuild + stats recompute for a no-op.
        if not any(quarter_turns) and self._originals_are_views:
            return
        # Originals start as views into _data (no 800 MB copy at init) and
        # become independent copies only once a rotation exists.
        if self._originals_are_views:
            self._data_original = [image.copy() for image in self._data_original]
            self._originals_are_views = False
        rotated = [
            original if turns == 0 else np.rot90(original, k=turns)
            for original, turns in zip(self._data_original, quarter_turns)
        ]
        # An RGB panel's color block turns with its luminance plane: the
        # browser lays every panel out from one H x W.
        rotated_rgb = [
            None if frame is None else frame if turns == 0 else np.rot90(frame, k=turns)
            for frame, turns in zip(self._rgb_frames_original, quarter_turns)
        ]
        shapes = [image.shape for image in rotated]
        if len(set(shapes)) > 1:
            max_h = max(shape[0] for shape in shapes)
            max_w = max(shape[1] for shape in shapes)
            rotated = [_resize_image(image, max_h, max_w) for image in rotated]
            rotated_rgb = [None if frame is None else _pad_rgb(frame, max_h, max_w) for frame in rotated_rgb]
        self._data = np.stack(rotated)
        self._rgb_frames = rotated_rgb
        if self._display_bin > 1:
            self._display_data = bin2d(self._data, factor=self._display_bin, mode="mean")
            self._display_rgb = _binned_rgb(self._rgb_frames, self._display_bin)
        else:
            self._display_data = self._data
            self._display_rgb = self._rgb_frames
        self.height = int(self._display_data.shape[1])
        self.width = int(self._display_data.shape[2])
        self._compute_all_stats()
        self._update_all_frames()

    def _quarter_turns(self, n_panels: int) -> list[int]:
        """Each panel's rotation as quarter turns in 0..3; a panel without an entry is unrotated."""
        return [
            (self.image_rotations[panel] if panel < len(self.image_rotations) else 0) % 4
            for panel in range(n_panels)
        ]

    @traitlets.observe("image_rotations")
    def _on_image_rotations_changed(self, change):
        """Re-rotate the panels when ``image_rotations`` changes; before ``__init__`` has built the pixels there is nothing to rotate."""
        if self._data is not None:
            self._apply_rotations()


