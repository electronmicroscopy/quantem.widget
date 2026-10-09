"""Show3D: scrub, play and compare stacks of 2D images.

One (N, H, W) stack, or several stacks side by side, becomes one float32
display stack that the browser plays back without another kernel round trip
(on WebGPU, or the Canvas2D path without it). The display stack is the native
pixels unless ``display_bin`` asks for a mean bin, which prints one line. The
native arrays stay in Python for GIF rendering and the saved-notebook
preview.
"""

import math
import pathlib
import warnings
from collections.abc import Mapping, Sequence
from typing import Self

import anywidget
import numpy as np
import torch
import traitlets

from quantem.widget.adapters import core as core_adapter
from quantem.widget.folder_watch_status import FOLDER_WATCH_STATE_VALUES
from quantem.widget.image_folder import (
    ImageFolderRecord,
    WatchedImageFolder,
    WatchedImageFolderMixin,
)
from quantem.widget.colormap import VALID_CMAPS, Colormap, cmap_to_name
from quantem.widget.export import HtmlExportMixin
from quantem.widget.fallback import StaticFallbackMixin
from quantem.widget.pages import PagesMixin
from quantem.widget.panels import PanelsMixin
from quantem.widget.show2d.options import (
    _expand_title_spans_for_flattened_labels,
    _normalise_title_span_sequence,
    _normalize_panel_annotations,
    _normalize_panel_overlays,
)
from quantem.widget.show2d.widget import _nonnegative_int
from quantem.widget.show3d.export import Show3DExport
from quantem.widget.show3d.fallback import Show3DFallback
from quantem.widget.show3d.playback import Show3DPlayback
from quantem.widget.utils.array import bin2d, to_numpy
from quantem.widget.utils.display_filter import format_display_filter_banner
from quantem.widget.state import SavedStateMixin, announce_browser_limit
from quantem.widget.utils.ui import UiMode, resolve_ui_mode


_MAX_PLAYBACK_FPS = 60.0


def _all_finite(values: np.ndarray, *, chunk_size: int = 1_000_000) -> bool:
    """Chunked finite scan so a multi-GB stack never allocates one giant mask."""
    flat = np.ravel(values)
    for start in range(0, flat.size, chunk_size):
        if not np.isfinite(flat[start : start + chunk_size]).all():
            return False
    return True


def _resolve_rgb_stack(data: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return ``(stack, is_rgb)`` with color as (N, H, W, 3) and gray as (N, H, W).

    A trailing axis of 3 or 4 means color: any (N, H, W, 3) stack, or a single
    (H, W, 3) frame told apart from a 3-frame gray stack by ``H > 4``. The
    heuristic is what lets ``read_images`` PNG stacks display in true color
    without a flag.
    """
    trailing_color = data.shape[-1] in (3, 4)
    if data.ndim == 3 and trailing_color and int(data.shape[0]) > 4:
        return data[None, ..., :3], True
    if data.ndim == 4 and trailing_color:
        return data[..., :3], True
    if data.ndim != 3:
        raise ValueError(
            f"Expected (N, H, W) gray stack or (N, H, W, 3) RGB stack, got {data.ndim}D shape {data.shape}"
        )
    return data, False


def _display_ready_color(data: np.ndarray) -> np.ndarray:
    """Normalize a color stack to float32 in [0, 1] (uint8 input divides by 255)."""
    data = data.astype(np.float32, copy=False)
    if data.size and float(np.nanmax(data)) > 1.5:
        data = data / 255.0
    return np.clip(data, 0.0, 1.0)


def _validated_float32(stack: np.ndarray, panel_name: str) -> np.ndarray:
    """Cast to float32 and reject NaN/inf before and after the cast.

    float64 values beyond 3.4e38 silently overflow to inf on cast and would
    poison every min/max and percentile downstream.
    """
    if np.iscomplexobj(stack):
        raise TypeError(
            f"{panel_name}: complex data not accepted. Convert first: "
            "np.abs(arr) for magnitude or np.angle(arr) for phase."
        )
    if not _all_finite(stack):
        raise ValueError(
            f"{panel_name} contains NaN or inf. Clean first: np.nan_to_num(arr, nan=0, posinf=0, neginf=0)."
        )
    with np.errstate(over="ignore", invalid="ignore"):
        stack_f32 = stack.astype(np.float32, copy=False)
    if not _all_finite(stack_f32):
        raise ValueError(
            f"{panel_name} exceeds float32 range (|value| > 3.4e38) after cast; "
            "rescale first: arr = arr / np.max(np.abs(arr))."
        )
    return stack_f32


class Show3D(
    WatchedImageFolderMixin,
    Show3DFallback,
    Show3DExport,
    Show3DPlayback,
    SavedStateMixin,
    HtmlExportMixin,
    PagesMixin,
    PanelsMixin,
    StaticFallbackMixin,
    anywidget.AnyWidget,
):
    """Interactive viewer for sequential 2D images.

    Parameters
    ----------
    *data_args : array_like
        One ``(N, H, W)`` stack, or several stacks shown side by side. A 2D
        image is a one-frame stack; a ``(N, H, W, 3)`` array is true color; a
        5D ``(pages, panels, N, H, W)`` array is a paged gallery. Torch tensors
        and quantem ``Dataset3d`` objects are accepted; a ``Dataset3d`` also
        supplies the title, sampling and units.
    labels : list of str, optional
        One label per frame. Defaults to the frame index.
    panel_titles : sequence, optional
        One title per panel. Entries may be strings or span lists with
        ``{"math": ...}`` / ``{"text": ...}`` items.
    page_labels : sequence of str, optional
        One label per page for 5D input.
    title : str
        Title above the canvas; a ``Dataset3d`` name is used when empty.
    ui_mode : {"interactive", "presentation", "report", "minimal"}
        Chrome preset. ``show_title``, ``show_stats`` and ``show_scale_bar``
        override the preset when given.
    cmap : str or Colormap
        Colormap for every panel.
    vmin, vmax : float, optional
        Fixed display range. ``None`` uses the stack range or auto contrast.
    auto_contrast : bool
        Percentile contrast per panel over the whole stack.
    link_contrast : bool, optional
        Share contrast handle movement across panels. Multi-panel input
        defaults to ``False`` because panels may be different physical
        quantities.
    sampling : float or sequence of float, optional
        Pixel size for the scale bar. A sequence uses its first entry.
    units : str or sequence of str, optional
        Unit of ``sampling`` (``"A"``, ``"nm"``, ...).
    smooth : bool
        Bilinear interpolation when the canvas enlarges a frame. Default False
        paints every data pixel as a sharp block (nearest neighbour).
    panel_annotations, panel_overlays : mapping or sequence, optional
        Text annotations and geometric overlays per panel, keyed by index or
        panel title.
    fps : float
        Playback rate, capped at 60.
    avg_window : int
        Moving-window frame average shown during scrubbing; 1 shows raw frames.
        Auto contrast keeps the stack-wide window, so averaging only reduces noise.
    show_fft : bool
        Show the FFT of the current frame.
    fft_layout : {"bottom", "right", "overlay"}
        Where the FFT is drawn.
    fft_overlay_zoom : float
        Initial FFT magnification, 1 to 32.
    panel_width_px : int
        Display width per panel in CSS pixels; 0 uses the frontend default.
    max_cols : int, optional
        Panels per row; 0 is one row.
    panel_gap : int
        Gap between panels in CSS pixels.
    dim_label : str
        Name of the frame axis in the controls ("Frame", "Slice", ...).
    display_bin : int
        Spatial mean bin of the browser display stack. The default 1 shows
        native pixels; a larger factor shrinks the payload of a large stack
        and prints one line naming the reduction. GIF exports always
        render the native arrays.
    offline : bool, optional
        Accepted and ignored, for callers that distinguish live from exported
        widgets: the browser always plays from the one embedded display stack.
    save_state : bool
        Persist the pixel traits into the saved notebook so it reopens
        interactively without a kernel. Default False keeps a static preview.
    notebook_preview_format : {"jpeg", "webp", "png"} or None
        Format of the static preview written into a saved notebook.
    notebook_preview_quality : int
        JPEG/WebP quality of that preview, 1 to 100.
    notebook_preview_max_px : int
        Longest panel side of that preview in pixels.
    verbose : bool
        Add the display-stack size to the display-bin notice (the notice
        itself always prints when a bin is active).
    """

    _esm = pathlib.Path(__file__).parent.parent / "static" / "show3d.js"

    slice_idx = traitlets.CInt(0).tag(sync=True)
    n_slices = traitlets.Int(1).tag(sync=True)
    folder_waiting = traitlets.Bool(False).tag(sync=True)
    folder_status = traitlets.Unicode("").tag(sync=True)
    folder_watch_state = traitlets.Enum(values=FOLDER_WATCH_STATE_VALUES, default_value="hidden").tag(sync=True)
    folder_watch_detail = traitlets.Unicode("").tag(sync=True)
    height = traitlets.Int(1).tag(sync=True)
    width = traitlets.Int(1).tag(sync=True)
    # Browser-owned display filters. Python only mirrors the state and
    # announces an active reduction; the stored stack is never filtered.
    denoise = traitlets.Unicode("none").tag(sync=True)
    denoise_sigma = traitlets.Float(4.0).tag(sync=True)
    denoise_bin = traitlets.Int(1).tag(sync=True)
    denoise_modes = traitlets.List(traitlets.Unicode()).tag(sync=True)
    denoise_sigmas = traitlets.List(traitlets.Float()).tag(sync=True)
    denoise_bins = traitlets.List(traitlets.Int()).tag(sync=True)
    denoise_scope = traitlets.Enum(["all", "panel"], default_value="all").tag(sync=True)
    denoise_banner = traitlets.Unicode("").tag(sync=True)
    show_denoise = traitlets.Bool(False).tag(sync=True)
    denoise_enabled = traitlets.Bool(True).tag(sync=True)
    frequency_filter = traitlets.Enum(["none", "lowpass", "highpass", "bandpass"], default_value="none").tag(sync=True)
    frequency_filter_enabled = traitlets.Bool(False).tag(sync=True)
    frequency_filter_cutoff = traitlets.Float(0.15).tag(sync=True)
    frequency_filter_center = traitlets.Float(0.30).tag(sync=True)
    frequency_filter_width = traitlets.Float(0.12).tag(sync=True)
    frequency_filter_modes = traitlets.List(traitlets.Unicode()).tag(sync=True)
    frequency_filter_cutoffs = traitlets.List(traitlets.Float()).tag(sync=True)
    frequency_filter_centers = traitlets.List(traitlets.Float()).tag(sync=True)
    frequency_filter_widths = traitlets.List(traitlets.Float()).tag(sync=True)
    frequency_filter_scope = traitlets.Enum(["all", "panel"], default_value="all").tag(sync=True)
    show_frequency_filter = traitlets.Bool(False).tag(sync=True)
    subpixel_align_enabled = traitlets.Bool(False).tag(sync=True)
    subpixel_align_reference = traitlets.Int(0).tag(sync=True)
    is_rgb = traitlets.Bool(False).tag(sync=True)
    # Bumped on every stack replacement so the frontend re-fires its render
    # effects even when a Bytes trait compares identical.
    frame_seq = traitlets.Int(0).tag(sync=True)
    _export_light = traitlets.Bool(False).tag(sync=True)
    _static_fallback_jpeg = traitlets.Unicode("").tag(sync=True)
    _static_fallback_mime = traitlets.Unicode("image/jpeg").tag(sync=True)
    # Transport: the browser slices every frame from this one embedded display
    # stack, exact float32 (live widget, full export) or uint8 (quantized export).
    _offline_stack = traitlets.Bytes(b"").tag(sync=True)
    _offline_float_stack = traitlets.Bytes(b"").tag(sync=True)
    _offline_min = traitlets.Float(0.0).tag(sync=True)
    _offline_max = traitlets.Float(1.0).tag(sync=True)
    _offline_mins = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    _offline_maxs = traitlets.List(traitlets.Float(), default_value=[]).tag(sync=True)
    display_bin = traitlets.Int(1).tag(sync=True)
    source_height = traitlets.Int(0).tag(sync=True)
    source_panel_width = traitlets.Int(0).tag(sync=True)
    labels = traitlets.List(traitlets.Unicode()).tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    show_title = traitlets.Bool(True).tag(sync=True)
    cmap = traitlets.Unicode("plasma").tag(sync=True)
    panel_cmaps = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    dim_label = traitlets.Unicode("Frame").tag(sync=True)
    dim_sampling = traitlets.Float(1.0).tag(sync=True)
    dim_unit = traitlets.Unicode("").tag(sync=True)

    n_panels = traitlets.Int(1).tag(sync=True)
    panel_titles = traitlets.List(traitlets.Unicode()).tag(sync=True)
    panel_title_spans = traitlets.List(default_value=[]).tag(sync=True)
    panel_width_px = traitlets.Int(0).tag(sync=True)
    shared_panel_source = traitlets.Bool(False).tag(sync=True)
    # One starred frame per panel; -1 = unset.
    starred = traitlets.List(traitlets.Int()).tag(sync=True)
    hidden_panels = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    selected_panels = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    panel_order = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    # Real frame count per panel when stacks of different length were padded
    # to the longest; the frontend marks frames past the real end.
    panel_real_frames = traitlets.List(traitlets.Int()).tag(sync=True)
    n_pages = traitlets.Int(1).tag(sync=True)
    page_idx = traitlets.Int(0).tag(sync=True)
    panels_per_page = traitlets.Int(0).tag(sync=True)
    page_labels = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    page_starred = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    hidden_page_slots = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    link_panels = traitlets.Bool(True).tag(sync=True)
    link_contrast = traitlets.Bool(True).tag(sync=True)
    view_state = traitlets.Dict(default_value={}).tag(sync=True)
    max_cols = traitlets.Int(4).tag(sync=True)
    show_resize_handles = traitlets.Bool(True).tag(sync=True)
    show_zoom_indicator = traitlets.Bool(False).tag(sync=True)
    show_panel_titles = traitlets.Bool(True).tag(sync=True)
    inter_panel_gap_px = traitlets.Int(0).tag(sync=True)
    panel_gap = traitlets.Int(0).tag(sync=True)

    playing = traitlets.Bool(False).tag(sync=True)
    reverse = traitlets.Bool(False).tag(sync=True)
    boomerang = traitlets.Bool(True).tag(sync=True)
    fps = traitlets.Float(30.0).tag(sync=True)
    # The displayed frame is the mean of avg_window consecutive frames; the
    # window slides inward at the stack ends so its width never shrinks.
    avg_window = traitlets.Int(1).tag(sync=True)
    loop = traitlets.Bool(True).tag(sync=True)
    loop_start = traitlets.Int(0).tag(sync=True)
    loop_end = traitlets.Int(-1).tag(sync=True)
    bookmarked_frames = traitlets.List(traitlets.Int()).tag(sync=True)
    playback_path = traitlets.List(traitlets.Int()).tag(sync=True)

    show_controls = traitlets.Bool(True).tag(sync=True)
    controls_collapsed = traitlets.Bool(False).tag(sync=True)
    show_stats = traitlets.Bool(False).tag(sync=True)
    stats_mean = traitlets.Float(0.0).tag(sync=True)
    stats_min = traitlets.Float(0.0).tag(sync=True)
    stats_max = traitlets.Float(0.0).tag(sync=True)
    stats_std = traitlets.Float(0.0).tag(sync=True)
    log_scale = traitlets.Bool(False).tag(sync=True)
    auto_contrast = traitlets.Bool(True).tag(sync=True)
    contrast_preset = traitlets.Unicode("custom").tag(sync=True)
    percentile_low = traitlets.Float(0.5).tag(sync=True)
    percentile_high = traitlets.Float(99.5).tag(sync=True)
    image_vmin_pct = traitlets.Float(0.0).tag(sync=True)
    image_vmax_pct = traitlets.Float(100.0).tag(sync=True)
    vmin = traitlets.Float(None, allow_none=True).tag(sync=True)
    vmax = traitlets.Float(None, allow_none=True).tag(sync=True)
    vmin_per_panel = traitlets.List(traitlets.Float(None, allow_none=True), default_value=[]).tag(sync=True)
    vmax_per_panel = traitlets.List(traitlets.Float(None, allow_none=True), default_value=[]).tag(sync=True)
    data_min = traitlets.Float(0.0).tag(sync=True)
    data_max = traitlets.Float(0.0).tag(sync=True)
    auto_vmins = traitlets.List(traitlets.Float()).tag(sync=True)
    auto_vmaxs = traitlets.List(traitlets.Float()).tag(sync=True)
    auto_vmins_per_panel = traitlets.List(traitlets.Float()).tag(sync=True)
    auto_vmaxs_per_panel = traitlets.List(traitlets.Float()).tag(sync=True)

    pixel_size = traitlets.Float(0.0).tag(sync=True)
    pixel_unit = traitlets.Unicode("A").tag(sync=True)
    scale_bar_visible = traitlets.Bool(True).tag(sync=True)
    smooth = traitlets.Bool(False).tag(sync=True)
    image_rotation = traitlets.Int(0).tag(sync=True)
    rotation_scope = traitlets.Enum(["all", "frame"], default_value="all").tag(sync=True)
    frame_rotations = traitlets.List(traitlets.Int(), default_value=[]).tag(sync=True)
    flip_horizontal = traitlets.Bool(False).tag(sync=True)
    flip_vertical = traitlets.Bool(False).tag(sync=True)

    roi_active = traitlets.Bool(False).tag(sync=True)
    roi_list = traitlets.List([]).tag(sync=True)
    roi_selected_idx = traitlets.Int(-1).tag(sync=True)
    roi_plot_data = traitlets.Bytes(b"").tag(sync=True)
    size = traitlets.Int(0).tag(sync=True)

    diff_mode = traitlets.Enum(["off", "previous", "first"], default_value="off").tag(sync=True)
    compare_mode = traitlets.Enum(["off", "blink", "difference", "overlay"], default_value="off").tag(sync=True)
    compare_pair = traitlets.List(traitlets.Int(), default_value=[0, 1]).tag(sync=True)
    blink_fps = traitlets.Float(2.0).tag(sync=True)
    diff_cmap = traitlets.Unicode("magenta-green").tag(sync=True)
    compare_background = traitlets.Enum(["light", "dark"], default_value="dark").tag(sync=True)
    panel_annotations = traitlets.List(traitlets.List(traitlets.Dict()), default_value=[]).tag(sync=True)
    panel_overlays = traitlets.List(traitlets.List(traitlets.Dict()), default_value=[]).tag(sync=True)

    show_fft = traitlets.Bool(False).tag(sync=True)
    fft_layout = traitlets.Enum(["bottom", "right", "overlay"], default_value="bottom").tag(sync=True)
    fft_overlay_position = traitlets.Enum(
        ["top-left", "top-right", "bottom-left", "bottom-right"], default_value="top-left"
    ).tag(sync=True)
    fft_overlay_size = traitlets.Float(0.35).tag(sync=True)
    fft_overlay_zoom = traitlets.Float(1.0).tag(sync=True)
    fft_window = traitlets.Bool(True).tag(sync=True)
    profile_line = traitlets.List(traitlets.Dict()).tag(sync=True)
    profile_width = traitlets.Int(1).tag(sync=True)
    show_kymograph = traitlets.Bool(False).tag(sync=True)

    @traitlets.validate("compare_pair")
    def _validate_compare_pair(self, proposal: dict) -> list[int]:
        """Clamp the blink and difference pair into the stack; anything but two indices is refused."""
        values = list(proposal["value"])
        if len(values) != 2:
            raise traitlets.TraitError("compare_pair must contain exactly two indices")
        n_slices = max(1, int(self.n_slices))
        return [max(0, min(n_slices - 1, int(value))) for value in values]

    @traitlets.validate("blink_fps")
    def _validate_blink_fps(self, proposal: dict) -> float:
        """Clamp the blink rate to 0.25 to 8 fps; a non-positive or non-finite rate is refused."""
        value = float(proposal["value"])
        if not math.isfinite(value) or value <= 0:
            raise traitlets.TraitError(f"blink_fps must be positive and finite, got {value}")
        return max(0.25, min(8.0, value))

    @traitlets.validate("avg_window")
    def _validate_avg_window(self, proposal: dict) -> int:
        """Refuse a frame-average window outside 1 to 15 frames rather than silently clamp it."""
        value = int(proposal["value"])
        if not 1 <= value <= 15:
            raise traitlets.TraitError(f"avg_window must be in [1, 15], got {value}")
        return value

    @traitlets.validate("image_vmin_pct", "image_vmax_pct")
    def _validate_image_clip_pct(self, proposal: dict) -> float:
        """Contrast clip percentiles must be finite and in [0, 100]."""
        value = float(proposal["value"])
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise traitlets.TraitError(f"{proposal['trait'].name} must be in [0, 100], got {value}")
        return value

    @traitlets.validate("playback_path")
    def _validate_playback_path(self, proposal: dict) -> list:
        """Wrap custom playback indices into the stack, so a path written for a longer stack still plays."""
        n_slices = max(1, int(self.n_slices))
        return [int(frame) % n_slices for frame in proposal["value"]]

    @traitlets.validate("subpixel_align_reference")
    def _validate_subpixel_align_reference(self, proposal: dict) -> int:
        """Clamp the alignment reference frame into the stack."""
        n_slices = max(1, int(self.n_slices))
        return max(0, min(n_slices - 1, int(proposal["value"])))

    @traitlets.validate("vmax")
    def _validate_vmax(self, proposal: dict) -> float | None:
        """A manual upper limit must be finite and not below ``vmin``."""
        new_vmax = proposal["value"]
        if new_vmax is not None:
            if not math.isfinite(new_vmax):
                raise traitlets.TraitError(f"vmax must be finite, got {new_vmax}")
            if self.vmin is not None and new_vmax < self.vmin:
                raise traitlets.TraitError(f"vmax ({new_vmax}) must be >= vmin ({self.vmin})")
        return new_vmax

    @traitlets.validate("vmin")
    def _validate_vmin(self, proposal: dict) -> float | None:
        """A manual lower limit must be finite and not above ``vmax``."""
        new_vmin = proposal["value"]
        if new_vmin is not None:
            if not math.isfinite(new_vmin):
                raise traitlets.TraitError(f"vmin must be finite, got {new_vmin}")
            if self.vmax is not None and new_vmin > self.vmax:
                raise traitlets.TraitError(f"vmin ({new_vmin}) must be <= vmax ({self.vmax})")
        return new_vmin

    @traitlets.validate("cmap")
    def _validate_cmap(self, proposal: dict) -> str:
        """Refuse unknown colormap names so the browser's lookup table never misses."""
        value = str(proposal["value"])
        if value not in VALID_CMAPS:
            raise traitlets.TraitError(f"Unknown cmap {value!r}. Valid: {sorted(VALID_CMAPS)}")
        return value

    @traitlets.validate("panel_cmaps")
    def _validate_panel_cmaps(self, proposal: dict) -> list[str]:
        """Per-panel colormaps: known names, one per panel, or empty for the shared ``cmap``."""
        values = [str(value) for value in proposal["value"]]
        for value in values:
            if value not in VALID_CMAPS:
                raise traitlets.TraitError(f"Unknown panel cmap {value!r}. Valid: {sorted(VALID_CMAPS)}")
        if values and len(values) != int(self.n_panels):
            raise traitlets.TraitError(f"panel_cmaps length must match n_panels ({int(self.n_panels)}), got {len(values)}")
        return values

    @traitlets.validate("bookmarked_frames")
    def _validate_bookmarks(self, proposal: dict) -> list:
        """Drop bookmarks outside the stack, as after ``set_image`` shrinks it."""
        n_slices = max(1, int(self.n_slices))
        return [int(frame) for frame in proposal["value"] if 0 <= int(frame) < n_slices]

    @traitlets.validate("loop_end")
    def _validate_loop_end(self, proposal: dict) -> int:
        """Clamp the loop end into the stack (-1 is the last frame); an end before ``loop_start`` is refused."""
        value = int(proposal["value"])
        if value < 0:
            return value
        value = min(value, max(1, int(self.n_slices)) - 1)
        if value < int(self.loop_start):
            raise traitlets.TraitError(f"loop_end ({value}) must be >= loop_start ({self.loop_start})")
        return value

    @traitlets.validate("loop_start")
    def _validate_loop_start(self, proposal: dict) -> int:
        """Clamp the loop start into the stack; a start after a set ``loop_end`` is refused."""
        value = max(0, min(int(proposal["value"]), max(1, int(self.n_slices)) - 1))
        end = int(self.loop_end)
        if end >= 0 and value > end:
            raise traitlets.TraitError(f"loop_start ({value}) must be <= loop_end ({end})")
        return value

    @traitlets.validate("pixel_size")
    def _validate_pixel_size(self, proposal: dict) -> float:
        """The scale-bar pixel size must be finite and non-negative (0 means uncalibrated)."""
        value = float(proposal["value"])
        if not math.isfinite(value) or value < 0:
            raise traitlets.TraitError(f"pixel_size must be finite and >= 0, got {value}")
        return value

    @traitlets.validate("labels")
    def _validate_labels(self, proposal: dict) -> list:
        """One label per frame, or none."""
        value = list(proposal["value"])
        if value and len(value) != int(self.n_slices):
            raise traitlets.TraitError(f"labels length ({len(value)}) must equal n_slices ({self.n_slices}) or be empty")
        return value

    @traitlets.validate("panel_titles")
    def _validate_panel_titles(self, proposal: dict) -> list:
        """One title per panel, or none."""
        value = list(proposal["value"])
        if value and len(value) != int(self.n_panels):
            raise traitlets.TraitError(f"panel_titles length ({len(value)}) must equal n_panels ({self.n_panels}) or be empty")
        return value

    @traitlets.validate("starred")
    def _validate_starred(self, proposal: dict) -> list:
        """One starred frame per panel (-1 for none); an empty list unstars every panel."""
        value = list(proposal["value"])
        n_panels = int(self.n_panels)
        if not value:
            return [-1] * n_panels
        if len(value) != n_panels:
            raise traitlets.TraitError(f"starred length ({len(value)}) must equal n_panels ({n_panels})")
        for panel, frame in enumerate(value):
            if frame != -1 and not 0 <= frame < int(self.n_slices):
                raise traitlets.TraitError(f"starred[{panel}] = {frame} out of range [-1, {self.n_slices})")
        return value

    @traitlets.validate("hidden_panels")
    def _validate_hidden_panels(self, proposal: dict) -> list:
        """Hidden panels deduplicated and sorted; hiding every panel is refused so the view is never blank."""
        n_panels = int(self.n_panels)
        clean = sorted({int(value) for value in proposal["value"] if 0 <= int(value) < n_panels})
        if len(clean) >= n_panels:
            raise traitlets.TraitError("hidden_panels cannot hide every panel; at least one panel must remain visible")
        return clean

    @traitlets.validate("vmin_per_panel", "vmax_per_panel")
    def _validate_panel_bounds(self, proposal: dict) -> list:
        """Per-panel limits: one finite value or None per panel, each panel's ``vmin`` not above its ``vmax``."""
        name = proposal["trait"].name
        value = list(proposal["value"])
        n_panels = int(self.n_panels)
        if len(value) != n_panels:
            raise traitlets.TraitError(f"{name} length ({len(value)}) must equal n_panels ({n_panels})")
        for i, bound in enumerate(value):
            if bound is not None and not math.isfinite(float(bound)):
                raise traitlets.TraitError(f"{name}[{i}] must be finite or None, got {bound}")
        other = list(self.vmax_per_panel if name == "vmin_per_panel" else self.vmin_per_panel)
        if len(other) == len(value):
            for i, (bound, other_bound) in enumerate(zip(value, other)):
                if bound is None or other_bound is None:
                    continue
                low, high = (bound, other_bound) if name == "vmin_per_panel" else (other_bound, bound)
                if float(low) > float(high):
                    raise traitlets.TraitError(f"vmin_per_panel[{i}] ({low}) must be <= vmax_per_panel[{i}] ({high})")
        return value

    @traitlets.validate("fps")
    def _validate_fps(self, proposal: dict) -> float:
        """Playback rate must be positive and finite; capped at ``_MAX_PLAYBACK_FPS``."""
        value = float(proposal["value"])
        if not math.isfinite(value) or value <= 0:
            raise traitlets.TraitError(f"fps must be finite and > 0, got {value}")
        return min(value, _MAX_PLAYBACK_FPS)

    @traitlets.validate("slice_idx")
    def _validate_slice_idx(self, proposal: dict) -> int:
        """Clamp the current frame into the stack."""
        return max(0, min(int(proposal["value"]), max(1, int(self.n_slices)) - 1))

    @traitlets.validate("profile_width")
    def _validate_profile_width(self, proposal: dict) -> int:
        """A line profile averages at least one pixel across."""
        value = int(proposal["value"])
        if value < 1:
            raise traitlets.TraitError(f"profile_width must be >= 1, got {value}")
        return value

    @traitlets.validate("percentile_low")
    def _validate_percentile_low(self, proposal: dict) -> float:
        """The low auto-contrast percentile lies in [0, 100] and below the high one."""
        value = float(proposal["value"])
        if not 0 <= value <= 100 or value >= float(self.percentile_high):
            raise traitlets.TraitError(f"percentile_low ({value}) must be in [0, 100] and < percentile_high ({self.percentile_high})")
        return value

    @traitlets.validate("percentile_high")
    def _validate_percentile_high(self, proposal: dict) -> float:
        """The high auto-contrast percentile lies in [0, 100] and above the low one."""
        value = float(proposal["value"])
        if not 0 <= value <= 100 or value <= float(self.percentile_low):
            raise traitlets.TraitError(f"percentile_high ({value}) must be in [0, 100] and > percentile_low ({self.percentile_low})")
        return value

    @traitlets.validate("roi_selected_idx")
    def _validate_roi_selected_idx(self, proposal: dict) -> int:
        """Clamp the selected ROI into the list; any negative value means none (-1)."""
        value = int(proposal["value"])
        if value < 0:
            return -1
        return min(value, max(0, len(self.roi_list) - 1))

    @traitlets.validate("roi_list")
    def _validate_roi_list(self, proposal: dict) -> list:
        """Every ROI has a known shape and non-negative sizes, so the mask and the outline agree."""
        value = list(proposal["value"])
        for i, roi in enumerate(value):
            shape = roi.get("shape", "circle")
            if shape not in {"circle", "square", "rectangle", "annular"}:
                raise traitlets.TraitError(f"ROI {i}: unknown shape {shape!r}")
            for key in ("radius", "radius_inner", "width", "height"):
                if key in roi and roi[key] is not None and float(roi[key]) < 0:
                    raise traitlets.TraitError(f"ROI {i}: {key} must be >= 0, got {roi[key]}")
        return value

    @traitlets.validate("denoise_bin")
    def _validate_denoise_bin(self, proposal: dict) -> int:
        """The display denoise bin is 1, 2 or 4."""
        value = int(proposal["value"])
        if value not in (1, 2, 4):
            raise traitlets.TraitError(f"denoise_bin must be 1, 2, or 4; got {value}")
        return value

    @traitlets.observe("denoise", "denoise_sigma", "denoise_bin", "denoise_enabled")
    def _on_denoise_change(self, change: dict) -> None:
        """Announce an active display reduction; the browser applies the filter."""
        active = bool(self.denoise_enabled) and not self.is_rgb and (self.denoise != "none" or int(self.denoise_bin) > 1)
        banner = format_display_filter_banner(self.denoise, float(self.denoise_sigma), int(self.denoise_bin)) if active else ""
        if banner and banner != self.denoise_banner:
            print(banner)
        self.denoise_banner = banner

    # --- construction ----------------------------------------------------------

    @classmethod
    def from_folder(
        cls,
        path: str | pathlib.Path,
        *,
        pattern: str = "*",
        file_types: str | Sequence[str] | None = None,
        recursive: bool = False,
        watch: bool = True,
        watch_interval: float = 1.0,
        **kwargs,
    ) -> Self:
        """Play naturally ordered folder images as one stack that grows in place.

        Every file is decoded through :func:`quantem.widget.io.read_image`;
        partially written files stay pending until a later poll. A missing
        folder is created when ``watch=True``. Frame labels come from the file
        names so arrivals stay identifiable.
        """
        if "labels" in kwargs:
            raise TypeError("Show3D.from_folder() derives frame labels from file paths; remove labels=.")
        source = WatchedImageFolder(
            path,
            pattern=pattern,
            file_types=file_types,
            recursive=recursive,
            interval=watch_interval,
            mode="frames",
            create=watch,
        )
        arrays, records = source.read_initial(allow_empty=watch, require_unchanged_followup=watch)
        explicit_calibration = "sampling" in kwargs or "units" in kwargs
        kwargs.setdefault("title", source.folder.name)
        kwargs.setdefault("verbose", False)
        if arrays:
            widget = cls(np.stack(arrays, axis=0), labels=[source.label(record.path) for record in records], **kwargs)
        else:
            display_bin = kwargs.get("display_bin", 1)
            side = max(1, int(display_bin)) if isinstance(display_bin, int) else 1
            widget = cls(np.zeros((1, side, side), dtype=np.float32), labels=[""], **kwargs)
            widget._set_folder_waiting_empty_state()
        source.attach(widget, explicit_calibration=explicit_calibration)
        if watch:
            widget.watch_folder(interval=watch_interval)
        return widget

    def __init__(
        self,
        *data_args,
        labels: list[str] | None = None,
        panel_titles: Sequence[str | Sequence[Mapping[str, object]] | Mapping[str, object] | None] | None = None,
        page_labels: Sequence[str | None] | None = None,
        title: str = "",
        ui_mode: UiMode = "interactive",
        show_title: bool | None = None,
        show_stats: bool | None = None,
        show_scale_bar: bool | None = None,
        cmap: str | Colormap = Colormap.PLASMA,
        vmin: float | None = None,
        vmax: float | None = None,
        auto_contrast: bool = True,
        link_contrast: bool | None = None,
        sampling: float | Sequence[float] | None = None,
        units: str | Sequence[str] | None = None,
        smooth: bool = False,
        panel_annotations: Sequence[object] | Mapping[object, object] | object | None = None,
        panel_overlays: Sequence[object] | Mapping[object, object] | object | None = None,
        fps: float = 30.0,
        avg_window: int = 1,
        show_fft: bool = False,
        fft_layout: str = "bottom",
        fft_overlay_zoom: float = 1.0,
        panel_width_px: int = 0,
        max_cols: int | None = None,
        panel_gap: int = 0,
        dim_label: str = "Frame",
        display_bin: int = 1,
        offline: bool | None = None,
        save_state: bool = False,
        notebook_preview_format: str | None = "jpeg",
        notebook_preview_quality: int = 88,
        notebook_preview_max_px: int = 512,
        verbose: bool = True,
    ):
        """Validate the display options, open the widget model, then load the stacks in one batched sync."""
        ui = resolve_ui_mode(
            ui_mode,
            defaults={
                "show_title": True,
                "show_controls": True,
                "controls_collapsed": False,
                "show_stats": False,
                "show_panel_titles": True,
                "show_resize_handles": True,
                "show_zoom_indicator": False,
                "show_scale_bar": True,
            },
            overrides={"show_title": show_title, "show_stats": show_stats, "show_scale_bar": show_scale_bar},
        )
        fft_layout = str(fft_layout).lower()
        if fft_layout not in {"bottom", "right", "overlay"}:
            raise ValueError(f"fft_layout must be one of 'bottom', 'right', or 'overlay', got {fft_layout!r}")
        fft_overlay_zoom = float(fft_overlay_zoom)
        if not 1.0 <= fft_overlay_zoom <= 32.0:
            raise ValueError(f"fft_overlay_zoom must be in [1.0, 32.0], got {fft_overlay_zoom}")
        panel_width_px = int(panel_width_px)
        if panel_width_px < 0:
            raise ValueError(f"panel_width_px must be >= 0, got {panel_width_px}")
        if isinstance(display_bin, bool) or not isinstance(display_bin, int) or display_bin < 1:
            raise ValueError(f"display_bin must be an integer >= 1; got {display_bin!r}")
        self._verbose = bool(verbose)
        # False keeps the saved notebook light: pixels are replaced by the
        # static preview on save. True persists the interactive state.
        self._save_state = bool(save_state)
        self._roi_plot_timer = None
        # The comm opens inside super().__init__ and snapshots get_state()
        # before any stack is loaded; the preview renderer checks for None.
        self._data = None
        self._display_data = None
        self._rgb_data = None
        self._source_panels = None
        self._configure_static_fallback(
            notebook_preview_format=notebook_preview_format,
            notebook_preview_quality=notebook_preview_quality,
            notebook_preview_max_px=notebook_preview_max_px,
        )
        gap = _nonnegative_int(panel_gap, name="panel_gap")
        super().__init__(
            show_title=bool(ui["show_title"]),
            show_controls=bool(ui["show_controls"]),
            controls_collapsed=bool(ui["controls_collapsed"]),
            show_stats=bool(ui["show_stats"]),
            show_panel_titles=bool(ui["show_panel_titles"]),
            show_resize_handles=bool(ui["show_resize_handles"]),
            show_zoom_indicator=bool(ui["show_zoom_indicator"]),
            scale_bar_visible=bool(ui["show_scale_bar"]),
            inter_panel_gap_px=gap,
            panel_gap=gap,
        )
        self._static_fallback_mime = self._static_fallback_mime_type()
        # hold_sync batches every trait assignment into one comm message; a
        # large stack otherwise costs one round trip per trait.
        with self.hold_sync():
            self._load_stack(
                data_args,
                labels=labels,
                panel_titles=panel_titles,
                page_labels=page_labels,
                title=title,
                sampling=sampling,
                units=units,
                display_bin=display_bin,
            )
            self.cmap = cmap_to_name(cmap)
            self.panel_cmaps = []
            if max_cols is not None:
                self.max_cols = int(max_cols)
            self.link_contrast = bool(link_contrast) if link_contrast is not None else int(self.n_panels) <= 1
            self.vmin = vmin
            self.vmax = vmax
            self.smooth = bool(smooth)
            self.auto_contrast = bool(auto_contrast)
            self.fps = float(fps)
            self.avg_window = int(avg_window)
            self.show_fft = bool(show_fft)
            self.fft_layout = fft_layout
            self.fft_overlay_zoom = fft_overlay_zoom
            self.size = panel_width_px
            self.dim_label = str(dim_label)
            n_panels = int(self.n_panels)
            self.denoise_scope = "all" if n_panels <= 1 else "panel"
            self.frequency_filter_scope = "all" if n_panels <= 1 else "panel"
            self.denoise_modes = ["none"] * n_panels
            self.denoise_sigmas = [4.0] * n_panels
            self.denoise_bins = [1] * n_panels
            self.denoise_enabled = False
            self.frequency_filter_modes = ["none"] * n_panels
            self.frequency_filter_cutoffs = [0.15] * n_panels
            self.frequency_filter_centers = [0.30] * n_panels
            self.frequency_filter_widths = [0.12] * n_panels
            self.panel_annotations = _normalize_panel_annotations(panel_annotations, n_items=n_panels, labels=list(self.panel_titles))
            self.panel_overlays = _normalize_panel_overlays(panel_overlays, n_items=n_panels, labels=list(self.panel_titles))
            self._refresh_auto_contrast_ranges()
            self._pack_exact_offline_stack()
        announce_browser_limit("Show3D", len(self._offline_float_stack), int(self.display_bin))
        self.observe(self._on_roi_change, names=["roi_active", "roi_list", "roi_selected_idx"])
        self.observe(self._on_diff_mode_change, names=["diff_mode"])
        self.observe(self._on_export_request_change, names=["export_request"])

    def _load_stack(
        self,
        data_args: tuple,
        *,
        labels: list[str] | None,
        panel_titles,
        page_labels: Sequence[str | None] | None,
        title: str,
        sampling,
        units,
        display_bin: int,
    ) -> None:
        """Validate the input stacks and set every data-derived trait.

        Panels are concatenated along the column axis into one wide frame so the
        browser receives a single display stack; ``panel_width_px`` tells it
        where each panel starts. Stacks of unequal length are padded with their
        last frame and ``panel_real_frames`` records the true counts.
        """
        if len(data_args) == 0:
            raise TypeError("Show3D requires at least one data argument")
        if len(data_args) == 1 and isinstance(data_args[0], (list, tuple)) and len(data_args[0]) > 0:
            data_args = tuple(data_args[0])
        plain_titles, spans_from_titles = _normalise_title_span_sequence(panel_titles)
        span_count = len(plain_titles or []) if spans_from_titles else 0
        # A 5D (pages, panels, N, H, W) array is a paged gallery: every page
        # shows ``panels`` stacks and the browser scrubs pages with one control,
        # so it flattens into the ordinary multi-panel path.
        n_pages, panels_per_page, resolved_page_labels = 1, 0, []
        if len(data_args) == 1 and isinstance(data_args[0], (np.ndarray, torch.Tensor)) and to_numpy(data_args[0]).ndim == 5:
            paged = to_numpy(data_args[0])
            n_pages, panels_per_page = int(paged.shape[0]), int(paged.shape[1])
            resolved_page_labels = [
                "" if label is None else str(label)
                for label in (page_labels if page_labels is not None else [f"Page {i + 1}" for i in range(n_pages)])
            ]
            if len(resolved_page_labels) != n_pages:
                raise ValueError(f"page_labels length ({len(resolved_page_labels)}) must match n_pages ({n_pages})")
            if plain_titles is None:
                plain_titles = [f"Panel {i + 1}" for _ in range(n_pages) for i in range(panels_per_page)]
            elif len(plain_titles) == panels_per_page:
                plain_titles = [str(label) for _ in range(n_pages) for label in plain_titles]
            elif len(plain_titles) != n_pages * panels_per_page:
                raise ValueError(
                    "panel_titles for paged Show3D must have length panels_per_page "
                    f"({panels_per_page}) or n_pages * panels_per_page ({n_pages * panels_per_page}), got {len(plain_titles)}"
                )
            data_args = tuple(paged.reshape(n_pages * panels_per_page, *paged.shape[2:]))
        first = data_args[0]
        dataset_sampling = None
        dataset_unit = None
        dim_sampling = None
        dim_unit = None
        if core_adapter.is_dataset(first, ndim=2) or core_adapter.is_dataset(first, ndim=3):
            title = title or (first.name or "")
            # the image axes are the last two; a stack's first axis is its frames
            dataset_sampling = float(first.sampling[-2])
            dataset_unit = str(first.units[-2])
            if len(first.sampling) >= 3:
                dim_sampling = float(first.sampling[0])
                dim_unit = str(first.units[0])
        panels = []
        for index, item in enumerate(data_args):
            stack = to_numpy(item)
            if stack.ndim == 2:
                stack = stack[None, ...]
            stack, is_rgb = _resolve_rgb_stack(stack)
            if 0 in stack.shape:
                raise ValueError(f"Panel {index}: empty stack shape {stack.shape}. All dims must be >= 1.")
            if index > 0 and bool(is_rgb) != bool(panels[0][1]):
                raise ValueError("Show3D panels must be all grayscale or all RGB")
            stack = _display_ready_color(stack) if is_rgb else _validated_float32(stack, f"Panel {index}")
            if index > 0 and stack.shape[1:3] != panels[0][0].shape[1:3]:
                raise ValueError(
                    f"Panel {index} image shape {stack.shape[1:3]} must match panel 0 {panels[0][0].shape[1:3]}; "
                    "crop or pad the panels to one (H, W) first."
                )
            panels.append((stack, is_rgb))
        is_rgb = panels[0][1]
        arrays = [stack for stack, _ in panels]
        self.n_panels = len(arrays)
        self.panel_titles = list(plain_titles) if plain_titles else [f"Panel {i + 1}" for i in range(len(arrays))]
        spans = _expand_title_spans_for_flattened_labels(
            spans_from_titles,
            original_len=span_count,
            n_panels=len(arrays),
            n_pages=n_pages,
            panels_per_page=panels_per_page,
        )
        self.panel_title_spans = list(spans) if spans and len(spans) == len(arrays) else []
        self.starred = [-1] * len(arrays)
        self.panel_order = []
        self.hidden_panels = []
        self.selected_panels = []
        self.vmin_per_panel = [None] * len(arrays)
        self.vmax_per_panel = [None] * len(arrays)
        self.n_pages = int(n_pages)
        self.panels_per_page = int(panels_per_page)
        self.page_idx = 0
        self.page_labels = list(resolved_page_labels)
        self.page_starred = [0] * int(n_pages) if n_pages > 1 else []
        self.hidden_page_slots = []
        real_frames = [stack.shape[0] for stack in arrays]
        max_frames = max(real_frames)
        padded = any(frames != max_frames for frames in real_frames)
        if padded:
            arrays = [
                stack if frames == max_frames
                else np.concatenate([stack, np.broadcast_to(stack[-1:], (max_frames - frames, *stack.shape[1:]))], axis=0)
                for stack, frames in zip(arrays, real_frames)
            ]
        self.panel_real_frames = real_frames if padded else []
        shared = len(arrays) > 1 and all(other is arrays[0] for other in arrays[1:])
        native_height, native_width = int(arrays[0].shape[1]), int(arrays[0].shape[2])
        # RGB keeps native pixels: its browser paint path has no mean bin.
        effective_bin = 1 if (is_rgb or native_height < display_bin or native_width < display_bin) else int(display_bin)
        self._source_panels = None
        if shared:
            # The same stack repeated across panels is sent once; the browser
            # draws that frame into every slot.
            self.shared_panel_source = True
            self._multi_panel_bin = 0
            data = arrays[0]
        elif len(arrays) > 1 and not is_rgb and effective_bin > 1:
            # Bin each panel before the concat so a 4k x (4k * panels) slab is
            # never materialized; the native panels stay for GIF rendering.
            self.shared_panel_source = False
            self._source_panels = arrays
            arrays = [np.asarray(bin2d(stack, factor=effective_bin, mode="mean"), dtype=np.float32) for stack in arrays]
            self._multi_panel_bin = effective_bin
            effective_bin = 1
            size = f", {sum(stack.nbytes for stack in arrays) / 2**20:.0f} MB display stack" if self._verbose else ""
            print(
                f"Show3D display bin {self._multi_panel_bin}x: {native_height}x{native_width} -> "
                f"{arrays[0].shape[1]}x{arrays[0].shape[2]} per panel (mean{size}); pass display_bin=1 for native pixels"
            )
            data = np.concatenate(arrays, axis=2)
        else:
            self.shared_panel_source = False
            self._multi_panel_bin = 0
            data = np.concatenate(arrays, axis=2) if len(arrays) > 1 else arrays[0]
        self._panel_width = int(arrays[0].shape[2])
        self.panel_width_px = self._panel_width if len(arrays) > 1 else 0
        self.is_rgb = bool(is_rgb)
        self._data = self._establish_rgb_luminance(data, is_rgb=is_rgb)
        self.n_slices = int(self._data.shape[0])
        self._display_bin = effective_bin
        self.display_bin = max(effective_bin, int(self._multi_panel_bin))
        if effective_bin > 1:
            self._display_data = bin2d(self._data, factor=effective_bin, mode="mean")
            size = f", {self._display_data.nbytes / 2**20:.0f} MB display stack" if self._verbose else ""
            print(
                f"Show3D display bin {effective_bin}x: {self._data.shape[1]}x{self._data.shape[2]} -> "
                f"{self._display_data.shape[1]}x{self._display_data.shape[2]} (mean{size}); pass display_bin=1 for native pixels"
            )
        else:
            self._display_data = self._data
        self.height = int(self._display_data.shape[1])
        self.width = int(self._display_data.shape[2])
        if self._source_panels:
            self.source_height = int(self._source_panels[0].shape[1])
            self.source_panel_width = int(self._source_panels[0].shape[2])
        else:
            self.source_height = int(self._data.shape[1])
            self.source_panel_width = native_width if shared else max(1, int(self._data.shape[2]) // len(arrays))
        if shared:
            self.panel_width_px = self.width
        # Explicit sampling/units win; a Dataset3d in nm is shown in Angstrom
        # so the scale bar matches the other quantem viewers.
        pixel_size = 0.0 if sampling is None else float(sampling[0] if isinstance(sampling, (tuple, list)) else sampling)
        pixel_unit = None if units is None else str(units[0] if isinstance(units, (tuple, list)) else units)
        dataset_in_nm = dataset_unit in ("nm", "nanometer")
        if pixel_size == 0.0 and dataset_sampling is not None:
            pixel_size = dataset_sampling * (10 if dataset_in_nm and pixel_unit in (None, "A", "Å", "angstrom", "Angstrom") else 1)
        if pixel_unit is None:
            pixel_unit = "A" if dataset_in_nm else (dataset_unit or "A")
        self.pixel_size = pixel_size * max(1, int(self.display_bin))
        self.pixel_unit = pixel_unit or "A"
        self.dim_sampling = dim_sampling or 1.0
        self.dim_unit = dim_unit or ""
        self.title = title
        self.labels = list(labels) if labels is not None else [str(i) for i in range(self.n_slices)]
        self.frame_rotations = [0] * self.n_slices
        self.slice_idx = self.n_slices // 2
        self._set_data_range()

    def _set_data_range(self) -> None:
        """Set ``data_min``/``data_max`` from the native stack and remember them.

        The diff modes overwrite the range with a zero-centred one; the copy in
        ``_data_min_off``/``_data_max_off`` is what turning a diff mode off restores.
        """
        stack_min, stack_max = torch.aminmax(torch.from_numpy(self._data))
        self.data_min = float(stack_min)
        self.data_max = float(stack_max)
        self._data_min_off = self.data_min
        self._data_max_off = self.data_max

    def set_image(self, data, labels: list[str] | None = None, *, display_bin: int | None = None) -> None:
        """Replace the stack in place, keeping colormap, contrast and playback state.

        The widget becomes single-panel. ROIs and the line profile are cleared
        when the frame shape changes; bookmarks and the loop range are clamped
        to the new frame count.
        """
        data = to_numpy(data)
        if data.ndim == 2:
            data = data[None, ...]
        data, is_rgb = _resolve_rgb_stack(data)
        if 0 in data.shape:
            raise ValueError(f"Empty stack: shape {data.shape}. All dims must be >= 1.")
        data = _display_ready_color(data) if is_rgb else _validated_float32(data, "Stack")
        if self._roi_plot_timer is not None:
            self._roi_plot_timer.cancel()
            self._roi_plot_timer = None
        self.playing = False
        prev_shape = (int(self.height), int(self.width))
        self.is_rgb = bool(is_rgb)
        self._data = self._establish_rgb_luminance(data, is_rgb=is_rgb)
        self.n_panels = 1
        self.panel_titles = []
        self.panel_title_spans = []
        self.starred = [-1]
        self.hidden_panels = []
        self.selected_panels = []
        self.panel_order = []
        self.panel_annotations = []
        self.panel_overlays = []
        self.n_pages = 1
        self.page_idx = 0
        self.panels_per_page = 0
        self.page_labels = []
        self.page_starred = []
        self.hidden_page_slots = []
        self.vmin_per_panel = [None]
        self.vmax_per_panel = [None]
        self._multi_panel_bin = 0
        self._source_panels = None
        self.shared_panel_source = False
        self.panel_real_frames = []
        self._panel_width = int(data.shape[2])
        self.n_slices = int(data.shape[0])
        if display_bin is None:
            replacement_bin = max(1, int(self.display_bin))
        elif isinstance(display_bin, bool) or not isinstance(display_bin, int) or display_bin < 1:
            raise ValueError(f"display_bin must be an integer >= 1, got {display_bin!r}")
        else:
            replacement_bin = int(display_bin)
        self._display_data = bin2d(self._data, factor=replacement_bin, mode="mean") if replacement_bin > 1 else self._data
        if replacement_bin > 1 and replacement_bin != int(self.display_bin):
            print(
                f"Show3D display bin {replacement_bin}x: {self._data.shape[1]}x{self._data.shape[2]} -> "
                f"{self._display_data.shape[1]}x{self._display_data.shape[2]} (mean); pass display_bin=1 for native pixels"
            )
        self._display_bin = replacement_bin
        self.display_bin = replacement_bin
        self.height = int(self._display_data.shape[1])
        self.width = int(self._display_data.shape[2])
        self.source_height = int(self._data.shape[1])
        self.source_panel_width = int(self._data.shape[2])
        self._set_data_range()
        self.labels = list(labels) if labels is not None else [str(i) for i in range(self.n_slices)]
        self.frame_rotations = [0] * self.n_slices
        self.slice_idx = min(self.slice_idx, self.n_slices - 1)
        if (self.height, self.width) != prev_shape:
            self.roi_list = []
            self.roi_selected_idx = -1
            self.profile_line = []
        self.bookmarked_frames = list(self.bookmarked_frames)
        if self.loop_start >= self.n_slices:
            self.loop_start = 0
        if self.loop_end >= self.n_slices:
            self.loop_end = -1
        self._refresh_auto_contrast_ranges()
        self._pack_exact_offline_stack()
        announce_browser_limit("Show3D", len(self._offline_float_stack), int(self.display_bin))
        self.frame_seq += 1

    def _set_folder_waiting_empty_state(self) -> None:
        """Represent a watched folder that has no readable frame yet."""
        empty = np.empty((0, 0, 0), dtype=np.float32)
        with self.hold_sync():
            self._data = empty
            self._display_data = empty
            self.n_slices = 0
            self.height = 0
            self.width = 0
            self.labels = []
            self.frame_seq += 1
            self._offline_stack = b""
            self._offline_float_stack = b""
            self._offline_mins = []
            self._offline_maxs = []
            self._static_fallback_jpeg = ""
            self.slice_idx = 0
            self.bookmarked_frames = []
            self.loop_start = 0
            self.loop_end = -1
            self.starred = [-1]
            self.panel_real_frames = []
            self.data_min = 0.0
            self.data_max = 0.0
            self._data_min_off = 0.0
            self._data_max_off = 0.0
            self.folder_waiting = True

    def _apply_folder_image_records(
        self,
        old_records: list[ImageFolderRecord],
        new_records: list[ImageFolderRecord],
        changed_arrays: dict[pathlib.Path, np.ndarray],
    ) -> None:
        """Replace folder frames while keeping path-addressed viewer state.

        Called by the folder watcher. Frame-indexed state (current frame,
        bookmarks, star, loop range) is re-keyed by file path so a file that
        arrives earlier in the natural order does not shift the user's view.
        """
        old_paths = [record.path for record in old_records]
        new_paths = [record.path for record in new_records]
        old_index = {path: idx for idx, path in enumerate(old_paths)}
        new_index = {path: idx for idx, path in enumerate(new_paths)}
        display_bin = max(1, int(self._display_bin))
        new_labels = [self._folder_source.label(path) for path in new_paths]
        if not old_paths:
            self.set_image(np.stack([changed_arrays[path] for path in new_paths], axis=0), labels=new_labels, display_bin=display_bin)
            return
        existing = self._rgb_data if (self.is_rgb and self._rgb_data is not None) else self._data
        stack = np.stack(
            [changed_arrays[path] if path in changed_arrays else np.asarray(existing[old_index[path]]) for path in new_paths],
            axis=0,
        )
        selected_path = old_paths[int(self.slice_idx)] if 0 <= int(self.slice_idx) < len(old_paths) else old_paths[0]
        bookmark_paths = {old_paths[idx] for idx in self.bookmarked_frames if 0 <= int(idx) < len(old_paths)}
        starred_path = old_paths[int(self.starred[0])] if self.starred and 0 <= int(self.starred[0]) < len(old_paths) else None
        loop_start_path = old_paths[min(max(int(self.loop_start), 0), len(old_paths) - 1)]
        loop_end_path = None if int(self.loop_end) < 0 else old_paths[int(self.loop_end)]
        was_playing = bool(self.playing)
        roi_active, roi_list, roi_selected_idx = bool(self.roi_active), list(self.roi_list), int(self.roi_selected_idx)
        profile_line = list(self.profile_line)
        with self.hold_sync():
            self.set_image(stack, labels=new_labels, display_bin=display_bin)
            self.slice_idx = new_index.get(selected_path, 0)
            self.bookmarked_frames = sorted(new_index[path] for path in bookmark_paths if path in new_index)
            self.starred = [new_index[starred_path] if starred_path in new_index else -1]
            self.loop_start = 0
            self.loop_end = -1 if loop_end_path is None else new_index.get(loop_end_path, -1)
            self.loop_start = new_index.get(loop_start_path, 0)
            self.roi_active = roi_active
            self.roi_list = roi_list
            self.roi_selected_idx = roi_selected_idx
            self.profile_line = profile_line
            self.playing = was_playing

    def __repr__(self) -> str:
        """Stack shape, current frame and colormap on one line."""
        return f"Show3D({self.n_slices}x{self.height}x{self.width}, frame={self.slice_idx}, cmap={self.cmap})"

    # --- state -----------------------------------------------------------------

    _STATE_KEYS = (
        "title", "cmap", "panel_cmaps", "log_scale", "auto_contrast",
        # percentile_high before percentile_low and loop_end before loop_start so
        # the cross-validators accept a round trip in dict order.
        "percentile_high", "percentile_low", "image_vmin_pct", "image_vmax_pct", "contrast_preset",
        "vmin", "vmax", "link_contrast", "link_panels", "view_state", "vmin_per_panel", "vmax_per_panel",
        "show_title", "show_stats", "show_controls", "controls_collapsed", "max_cols", "panel_gap",
        "inter_panel_gap_px", "show_panel_titles", "show_zoom_indicator", "show_fft", "fft_layout",
        "fft_overlay_position", "fft_overlay_size", "fft_overlay_zoom", "show_kymograph", "fft_window",
        "pixel_size", "pixel_unit", "smooth", "image_rotation", "rotation_scope", "frame_rotations",
        "flip_horizontal", "flip_vertical", "scale_bar_visible", "size", "fps", "avg_window", "loop",
        "reverse", "boomerang", "loop_end", "loop_start", "bookmarked_frames", "starred", "hidden_panels",
        "selected_panels", "panel_order", "n_pages", "page_idx", "panels_per_page", "page_labels",
        "page_starred", "hidden_page_slots", "playback_path", "slice_idx", "roi_active", "roi_list",
        "roi_selected_idx", "profile_line", "profile_width", "diff_mode", "compare_mode", "compare_pair",
        "blink_fps", "diff_cmap", "compare_background", "panel_annotations", "panel_overlays",
        "denoise", "denoise_modes", "show_denoise", "denoise_enabled", "denoise_sigma", "denoise_sigmas",
        "denoise_bin", "denoise_bins", "denoise_scope", "frequency_filter", "frequency_filter_modes",
        "frequency_filter_enabled", "frequency_filter_cutoff", "frequency_filter_cutoffs",
        "frequency_filter_center", "frequency_filter_centers", "frequency_filter_width",
        "frequency_filter_widths", "frequency_filter_scope", "show_frequency_filter",
        "subpixel_align_enabled", "subpixel_align_reference", "dim_label", "dim_sampling", "dim_unit",
        "labels", "panel_titles", "panel_title_spans",
    )
    # Keys whose length is tied to the stack; dropped on load when the sizes differ.
    _FRAME_LENGTH_KEYS = ("labels", "frame_rotations")
    _PANEL_LENGTH_KEYS = ("starred", "panel_titles", "panel_title_spans", "panel_cmaps", "vmin_per_panel", "vmax_per_panel", "panel_order")

    def state_dict(self) -> dict:
        """JSON-serializable snapshot of every user-tunable trait (no pixels)."""
        state = {"_widget": "Show3D"}
        for key in self._STATE_KEYS:
            value = getattr(self, key)
            state[key] = dict(value) if isinstance(value, dict) else list(value) if isinstance(value, list) else value
        return state

    def load_state_dict(self, state: dict) -> None:
        """Apply a ``state_dict`` snapshot, dropping keys that no longer fit.

        Length-coupled keys whose size disagrees with the current stack or
        panel count are skipped so a state saved from one trial loads onto
        another. Cross-validated pairs (percentiles, vmin/vmax, loop range) are
        applied in an order the validators accept. Unknown keys warn.
        """
        state = dict(state)
        marker = state.pop("_widget", None)
        if marker is not None and marker != "Show3D":
            raise ValueError(f"load_state_dict: state was saved from {marker!r}, not Show3D.")
        n_slices, n_panels = int(self.n_slices), int(self.n_panels)
        for key in self._FRAME_LENGTH_KEYS:
            if key in state and 0 < len(state[key]) != n_slices:
                state.pop(key)
        for key in self._PANEL_LENGTH_KEYS:
            if key in state and 0 < len(state[key]) != n_panels:
                state.pop(key)
        if "hidden_panels" in state:
            state["hidden_panels"] = sorted({int(panel) for panel in state["hidden_panels"] if 0 <= int(panel) < n_panels})[: n_panels - 1]
        if int(self.n_pages) > 1:
            state["n_pages"] = int(self.n_pages)
            state["panels_per_page"] = int(self.panels_per_page)
            for key in ("page_labels", "page_starred"):
                if key in state and len(state[key]) != int(self.n_pages):
                    state.pop(key)
        else:
            for key in ("n_pages", "page_idx", "panels_per_page", "page_labels", "page_starred", "hidden_page_slots"):
                state.pop(key, None)
        if n_panels > 1:
            for key in ("roi_active", "roi_list", "roi_selected_idx"):
                state.pop(key, None)
        unknown = [key for key in state if key not in self._STATE_KEYS]
        for key in unknown:
            state.pop(key)
        if "vmin_per_panel" in state or "vmax_per_panel" in state:
            self.vmin_per_panel = [None] * n_panels
            self.vmax_per_panel = [None] * n_panels
            self.vmax_per_panel = list(state.pop("vmax_per_panel", self.vmax_per_panel))
            self.vmin_per_panel = list(state.pop("vmin_per_panel", self.vmin_per_panel))
        if "percentile_low" in state or "percentile_high" in state:
            low = float(state.pop("percentile_low", self.percentile_low))
            high = float(state.pop("percentile_high", self.percentile_high))
            if high <= float(self.percentile_low):
                self.percentile_low = low
                self.percentile_high = high
            else:
                self.percentile_high = high
                self.percentile_low = low
        if "vmin" in state or "vmax" in state:
            new_vmin = state.pop("vmin", self.vmin)
            new_vmax = state.pop("vmax", self.vmax)
            self.vmin = None
            self.vmax = None
            self.vmin = None if new_vmin is None else float(new_vmin)
            self.vmax = None if new_vmax is None else float(new_vmax)
        if "loop_start" in state or "loop_end" in state:
            start = max(0, min(int(state.pop("loop_start", self.loop_start)), n_slices - 1))
            end = int(state.pop("loop_end", self.loop_end))
            end = -1 if end < 0 else min(end, n_slices - 1)
            self.loop_start = 0
            self.loop_end = end
            self.loop_start = start if end < 0 else min(start, end)
        for key, value in state.items():
            setattr(self, key, value)
        if unknown:
            warnings.warn(f"load_state_dict ignored unknown keys: {unknown}.", stacklevel=2)

    def _panel_count(self) -> int:
        """Number of panels, for the shared panel and page helpers (``PanelsMixin`` hook)."""
        return int(self.n_panels)

    def _panel_title_for_index(self, panel: int) -> str:
        """Title of one panel, ``"Panel <n>"`` when unset; panel references by title resolve against it."""
        if 0 <= panel < len(self.panel_titles) and self.panel_titles[panel]:
            return str(self.panel_titles[panel])
        return f"Panel {panel + 1}"

    @property
    def _visible_panels(self) -> list[int]:
        """Panel indices in display order on the current page, minus hidden ones."""
        n_panels = int(self.n_panels)
        if int(self.n_pages) > 1 and int(self.panels_per_page) > 0:
            start = int(self.page_idx) * int(self.panels_per_page)
            ordered = list(range(start, min(start + int(self.panels_per_page), n_panels)))
            hidden_slots = set(self.hidden_page_slots)
            return [panel for panel in ordered if (panel - start) not in hidden_slots] or ordered[:1]
        order = list(self.panel_order) if len(self.panel_order) == n_panels else list(range(n_panels))
        hidden = set(self.hidden_panels)
        return [panel for panel in order if panel not in hidden]

    def free(self) -> None:
        """Release the stack and every synced pixel buffer.

        ``del widget`` alone does not free memory because traitlets observers
        pin the widget; call this before discarding it. Idempotent.
        """
        self.stop_folder_watch()
        if self._data is None:
            return
        # Stop frontend playback before the buffers it reads are dropped.
        self.playing = False
        if self._roi_plot_timer is not None:
            self._roi_plot_timer.cancel()
            self._roi_plot_timer = None
        self._data = None
        self._display_data = None
        self._rgb_data = None
        self._source_panels = None
        for trait in ("roi_plot_data", "_offline_stack", "_offline_float_stack", "export_payload"):
            setattr(self, trait, b"")
