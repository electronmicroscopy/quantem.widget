"""
mark2d: Interactive 2D image annotation widget.

Mark points (atom positions, features), draw ROIs, measure distances,
snap to intensity peaks. Supports gallery mode with multiple images.
"""

import json
import pathlib
from typing import List, Optional

import anywidget
import numpy as np
import traitlets

from quantem.widget.array_utils import to_numpy, _resize_image
from quantem.widget.json_state import resolve_widget_version, save_state_file, unwrap_state_payload
from quantem.widget.tool_parity import (
    bind_tool_runtime_api,
    build_tool_groups,
    normalize_tool_groups,
)


_MARKER_SHAPES = ["circle", "triangle", "square", "diamond", "star"]
_MARKER_COLORS = [
    "#f44336", "#4caf50", "#2196f3", "#ff9800", "#9c27b0",
    "#00bcd4", "#ffeb3b", "#e91e63", "#8bc34a", "#ff5722",
]
_COLOR_NAMES = {
    "#f44336": "red", "#4caf50": "green", "#2196f3": "blue",
    "#ff9800": "orange", "#9c27b0": "purple", "#00bcd4": "cyan",
    "#ffeb3b": "yellow", "#e91e63": "pink", "#8bc34a": "lime",
    "#ff5722": "deep orange",
    # ROI colors
    "#0f0": "green", "#ff0": "yellow", "#0af": "cyan",
    "#f0f": "magenta", "#f80": "orange", "#f44": "red",
    # Common CSS names
    "white": "white", "black": "black", "red": "red",
    "#ff0": "yellow", "#0ff": "cyan", "#f00": "red",
}

def _color_name(hex_code: str) -> str:
    return _COLOR_NAMES.get(hex_code.lower(), hex_code)


class Mark2D(anywidget.AnyWidget):
    """
    Interactive point picker for 2D images.

    Click on an image to select points (atom positions, features, lattice
    vectors). Supports gallery mode for comparing multiple images, pre-loaded
    points from detection algorithms, multiple ROI overlays with pixel
    statistics, snap-to-peak for precise atom column picking, and calibrated
    distance measurements between points.

    Parameters
    ----------
    data : array_like
        Image data. Accepts:

        - 2D array ``(H, W)`` — single image
        - 3D array ``(N, H, W)`` — gallery of N images
        - List of 2D arrays — gallery (resized to common dimensions)
        - ``Dataset2d`` object — array and sampling auto-extracted

    scale : float, default 1.0
        Display scale factor. Values > 1 enlarge the canvas.
    dot_size : int, default 12
        Diameter of point markers in CSS pixels.
    max_points : int, default 10
        Maximum number of points per image. Oldest points are removed
        when the limit is exceeded.

    ncols : int, default 3
        Number of columns in the gallery grid (ignored for single images).
    labels : list of str, optional
        Per-image labels shown below each gallery tile and in the header.
        Defaults to ``"Image 1"``, ``"Image 2"``, etc.

    marker_border : int, default 2
        Border width of point markers in pixels (0–6). The border grows
        inward from the marker edge, so the overall marker size stays
        constant. Set to 0 for borderless markers.
    marker_opacity : float, default 1.0
        Opacity of point markers (0.1–1.0).
    label_size : int, default 0
        Font size in pixels for the numbered label above each marker.
        ``0`` means auto-scale relative to ``dot_size``.
    label_color : str, default ""
        CSS color for numbered labels (e.g. ``"white"``, ``"#ff0"``).
        Empty string uses the automatic theme color.

    pixel_size : float, default 0.0
        Pixel size in angstroms. When set, the widget displays a calibrated
        scale bar and shows point-to-point distances in physical units
        (angstroms or nanometers). ``0`` means uncalibrated.

    points : list or ndarray, optional
        Pre-populate the widget with initial points. Useful for reviewing
        or refining positions from an atom-finding algorithm. Accepts:

        - List of ``(row, col)`` tuples: ``[(10, 20), (30, 40)]``
        - List of dicts with optional shape/color:
          ``[{"row": 10, "col": 20, "shape": "star", "color": "#f00"}]``
        - NumPy array of shape ``(N, 2)`` with columns ``[row, col]``
        - For gallery: list of the above, one per image

        When ``shape`` or ``color`` are omitted, they cycle through the
        built-in palettes (5 shapes, 10 colors).

    marker_shape : str, default "circle"
        Active marker shape for new points. One of ``"circle"``,
        ``"triangle"``, ``"square"``, ``"diamond"``, ``"star"``.
        Synced bidirectionally — changes in the UI are reflected in Python.
    marker_color : str, default "#f44336"
        Active marker color for new points (CSS color string).
        Synced bidirectionally — changes in the UI are reflected in Python.
    snap_enabled : bool, default False
        Whether snap-to-peak is active. When ``True``, clicked positions
        are snapped to the local intensity maximum within ``snap_radius``.
    snap_radius : int, default 5
        Search radius in pixels for snap-to-peak.
    title : str, default ""
        Title displayed in the widget header. Empty string shows ``"Mark2D"``.
    show_stats : bool, default True
        Show statistics bar (mean, min, max, std) below the canvas.
    cmap : str, default "gray"
        Colormap for image rendering. Options: ``"gray"``, ``"inferno"``,
        ``"viridis"``, ``"plasma"``, ``"magma"``, ``"hot"``.
    auto_contrast : bool, default True
        Enable automatic contrast via 2–98% percentile clipping.
        When ``False``, contrast is controlled by the histogram slider.
    log_scale : bool, default False
        Apply log(1+x) transform before rendering. Useful for images
        with large dynamic range (e.g. diffraction patterns).
    show_fft : bool, default False
        Show FFT power spectrum alongside the image.
    disabled_tools : list of str, optional
        Tool groups to disable in the frontend UI/interaction layer. This is
        useful for shared notebooks where viewers should not be able to modify
        selected controls or annotations. Supported values:
        ``"points"``, ``"roi"``, ``"profile"``, ``"display"``,
        ``"marker_style"``, ``"snap"``, ``"navigation"``, ``"view"``,
        ``"export"``, ``"all"``.
    disable_points : bool, default False
        Convenience flag equivalent to including ``"points"`` in
        ``disabled_tools``.
    disable_roi : bool, default False
        Convenience flag equivalent to including ``"roi"`` in
        ``disabled_tools``.
    disable_profile : bool, default False
        Convenience flag equivalent to including ``"profile"`` in
        ``disabled_tools``.
    disable_display : bool, default False
        Convenience flag equivalent to including ``"display"`` in
        ``disabled_tools``.
    disable_marker_style : bool, default False
        Convenience flag equivalent to including ``"marker_style"`` in
        ``disabled_tools``.
    disable_snap : bool, default False
        Convenience flag equivalent to including ``"snap"`` in
        ``disabled_tools``.
    disable_navigation : bool, default False
        Convenience flag equivalent to including ``"navigation"`` in
        ``disabled_tools``.
    disable_view : bool, default False
        Convenience flag equivalent to including ``"view"`` in
        ``disabled_tools``.
    disable_export : bool, default False
        Convenience flag equivalent to including ``"export"`` in
        ``disabled_tools``.
    disable_all : bool, default False
        Convenience flag equivalent to ``disabled_tools=["all"]``.
    hidden_tools : list of str, optional
        Tool groups to hide from the frontend UI. Hidden tools are also
        interaction-locked (equivalent to disabled for behavior), but their
        controls are not rendered. Supported values:
        ``"points"``, ``"roi"``, ``"profile"``, ``"display"``,
        ``"marker_style"``, ``"snap"``, ``"navigation"``, ``"view"``,
        ``"export"``, ``"all"``.
    hide_points : bool, default False
        Convenience flag equivalent to including ``"points"`` in
        ``hidden_tools``.
    hide_roi : bool, default False
        Convenience flag equivalent to including ``"roi"`` in
        ``hidden_tools``.
    hide_profile : bool, default False
        Convenience flag equivalent to including ``"profile"`` in
        ``hidden_tools``.
    hide_display : bool, default False
        Convenience flag equivalent to including ``"display"`` in
        ``hidden_tools``.
    hide_marker_style : bool, default False
        Convenience flag equivalent to including ``"marker_style"`` in
        ``hidden_tools``.
    hide_snap : bool, default False
        Convenience flag equivalent to including ``"snap"`` in
        ``hidden_tools``.
    hide_navigation : bool, default False
        Convenience flag equivalent to including ``"navigation"`` in
        ``hidden_tools``.
    hide_view : bool, default False
        Convenience flag equivalent to including ``"view"`` in
        ``hidden_tools``.
    hide_export : bool, default False
        Convenience flag equivalent to including ``"export"`` in
        ``hidden_tools``.
    hide_all : bool, default False
        Convenience flag equivalent to ``hidden_tools=["all"]``.

    Attributes
    ----------
    selected_points : list
        Currently placed points, synced bidirectionally with the widget.

        - **Single image**: flat list of point dicts
          ``[{"row": 10, "col": 20, "shape": "circle", "color": "#f44336"}, ...]``
        - **Gallery mode**: list of lists, one per image
          ``[[point, ...], [point, ...], ...]``

        Each point dict has keys: ``row`` (int), ``col`` (int), ``shape`` (str),
        ``color`` (str). You can set this attribute from Python to update
        the widget in real time.

    roi_list : list
        Currently defined ROI overlays, synced with the widget. Each ROI
        is a dict with keys:

        - ``id`` (int) — unique identifier
        - ``mode`` (str) — ``"circle"``, ``"square"``, or ``"rectangle"``
        - ``row``, ``col`` (int) — center position in image pixels
        - ``radius`` (int) — radius for circle/square modes
        - ``rectW``, ``rectH`` (int) — width/height for rectangle mode
        - ``color`` (str) — CSS stroke color
        - ``opacity`` (float) — opacity (0.1–1.0)

        Set from Python to programmatically define ROIs, or read after
        interactive use to retrieve user-defined regions.

    Notes
    -----
    **Marker shapes**: circle, triangle, square, diamond, star (5 shapes
    that cycle automatically).

    **Marker colors**: 10 colors that cycle: red, green, blue, orange,
    purple, cyan, yellow, pink, lime, deep orange.

    **Snap-to-peak**: When enabled in the UI, clicking snaps the point to
    the local intensity maximum within a configurable search radius. Useful
    for precise atom column picking on HAADF-STEM images.

    **Distance measurements**: Distances between consecutive points are
    displayed in the point list. With ``pixel_size`` set, distances
    are shown in angstroms (< 10 Å) or nanometers (>= 10 Å).

    **ROI statistics**: When an ROI is active, the widget computes and
    displays mean, standard deviation, min, max, and pixel count for the
    region. When multiple ROIs exist, a summary table shows all ROI stats.
    Active ROIs also show dotted horizontal/vertical center guide lines.

    **Pairwise distances**: When 2+ points are placed, a table below the
    point list shows distances between all pairs of points.

    **Line profile**: Toggle "Profile" mode in the controls, then click
    two points to sample intensity along a line. A sparkline graph with
    calibrated x-axis appears below the canvas. Also available
    programmatically via ``set_profile()``, ``profile_values``, and
    ``profile_distance``.

    **Keyboard shortcuts** (widget must be focused):

    - ``Delete`` / ``Backspace`` — remove last point (undo)
    - ``Ctrl+Z`` / ``Cmd+Z`` — undo
    - ``Ctrl+Shift+Z`` / ``Cmd+Shift+Z`` — redo
    - ``1``–``6`` — select ROI #1–6
    - Arrow keys — nudge active ROI by 1 pixel
    - Arrow keys (no ROI, gallery) — navigate between images
    - ``Escape`` — deselect ROI

    Examples
    --------
    Basic point picking:

    >>> import numpy as np
    >>> from quantem.widget import Mark2D
    >>> img = np.random.rand(256, 256).astype(np.float32)
    >>> w = Mark2D(img, scale=1.5, max_points=5)
    >>> w  # display in notebook; click to place points
    >>> w.selected_points  # read back placed points

    Pre-loaded points from a detection algorithm:

    >>> peaks = find_atom_columns(img)  # returns (N, 2) array
    >>> w = Mark2D(img, points=peaks, pixel_size=0.82)
    >>> # Points appear immediately; user can add/remove/adjust

    Pre-loaded points with custom appearance:

    >>> pts = [
    ...     {"row": 200, "col": 100, "shape": "star", "color": "#ff0"},
    ...     {"row": 250, "col": 150, "shape": "diamond", "color": "#0ff"},
    ... ]
    >>> w = Mark2D(img, points=pts, marker_border=0, marker_opacity=0.8)

    Gallery mode for comparing multiple images:

    >>> imgs = [original, filtered, denoised]
    >>> w = Mark2D(imgs, ncols=3, labels=["Raw", "Filtered", "Denoised"])
    >>> # Points are tracked independently per image

    Gallery with per-image pre-loaded points:

    >>> per_image_pts = [[(10, 20)], [(30, 40), (50, 60)], []]
    >>> w = Mark2D(imgs, points=per_image_pts)

    Programmatic ROI management:

    >>> w = Mark2D(img)
    >>> w.add_roi(row=128, col=128, mode="circle", radius=30, color="#0f0")
    >>> w.add_roi(row=200, col=200, mode="rectangle", rect_w=80, rect_h=40)
    >>> w.roi_list  # inspect ROI parameters
    >>> w.roi_center()  # center of most recently added ROI -> (200, 200)
    >>> w.roi_radius()  # radius for circle/square, None for rectangle
    >>> w.roi_size()  # shape-aware size dict (e.g. width/height for rectangle)
    >>> w.roi_list = []  # clear all ROIs

    Snap-to-peak for precise atom picking:

    >>> w = Mark2D(haadf_image, snap_enabled=True, snap_radius=8,
    ...             pixel_size=0.82)
    >>> # Clicks auto-snap to the nearest intensity maximum

    Custom marker defaults:

    >>> w = Mark2D(img, marker_shape="star", marker_color="#ff9800")
    >>> # All new points will be orange stars until changed in the UI

    Human-friendly tool locking:

    >>> w = Mark2D(img, disable_points=True, disable_roi=True, disable_display=True)
    >>> w_read_only = Mark2D(img, disable_all=True)
    >>> w_clean = Mark2D(img, hide_display=True, hide_export=True)

    Save and restore full widget state (state portability):

    >>> # User A: create widget, place points and ROIs interactively
    >>> w = Mark2D(img, pixel_size=1.5)
    >>> # ... user clicks to place points, adds ROIs, enables snap ...
    >>> state = {
    ...     "points": w.selected_points,
    ...     "rois": w.roi_list,
    ...     "marker_shape": w.marker_shape,
    ...     "marker_color": w.marker_color,
    ...     "snap_enabled": w.snap_enabled,
    ...     "snap_radius": w.snap_radius,
    ... }
    >>> # User B: restore exact same state on another machine
    >>> w2 = Mark2D(img, pixel_size=1.5,
    ...              points=state["points"],
    ...              marker_shape=state["marker_shape"],
    ...              marker_color=state["marker_color"],
    ...              snap_enabled=state["snap_enabled"],
    ...              snap_radius=state["snap_radius"])
    >>> w2.roi_list = state["rois"]

    Line profile (programmatic):

    >>> w = Mark2D(img, pixel_size=0.82)
    >>> w.set_profile((10, 20), (100, 200))
    >>> w.profile_values  # sampled intensities along the line
    >>> w.profile_distance  # total distance in angstroms

    Export points as NumPy array:

    >>> w = Mark2D(img, points=[(10, 20), (30, 40)])
    >>> w.points_as_array()  # shape (2, 2), columns [row, col]
    >>> w.points_as_dict()  # [{"row": 10, "col": 20}, ...]
    """

    _esm = pathlib.Path(__file__).parent / "static" / "mark2d.js"

    # Image data (gallery-capable, matching Show2D pattern)
    widget_version = traitlets.Unicode("unknown").tag(sync=True)
    n_images = traitlets.Int(1).tag(sync=True)
    width = traitlets.Int(0).tag(sync=True)
    height = traitlets.Int(0).tag(sync=True)
    frame_bytes = traitlets.Bytes(b"").tag(sync=True)
    img_min = traitlets.List(traitlets.Float()).tag(sync=True)
    img_max = traitlets.List(traitlets.Float()).tag(sync=True)

    # Gallery controls
    selected_idx = traitlets.Int(0).tag(sync=True)
    ncols = traitlets.Int(3).tag(sync=True)
    labels = traitlets.List(traitlets.Unicode()).tag(sync=True)

    # UI controls
    scale = traitlets.Float(1.0).tag(sync=True)
    selected_points = traitlets.List().tag(sync=True)
    dot_size = traitlets.Int(12).tag(sync=True)
    max_points = traitlets.Int(10).tag(sync=True)

    # Marker styling (advanced)
    marker_border = traitlets.Int(2).tag(sync=True)
    marker_opacity = traitlets.Float(1.0).tag(sync=True)
    label_size = traitlets.Int(0).tag(sync=True)
    label_color = traitlets.Unicode("").tag(sync=True)

    # Scale bar
    pixel_size = traitlets.Float(0.0).tag(sync=True)

    # Active marker selection (synced for state portability)
    marker_shape = traitlets.Unicode("circle").tag(sync=True)
    marker_color = traitlets.Unicode("#f44336").tag(sync=True)

    # Snap-to-peak (synced for state portability)
    snap_enabled = traitlets.Bool(False).tag(sync=True)
    snap_radius = traitlets.Int(5).tag(sync=True)

    # ROI overlays (synced to JS)
    roi_list = traitlets.List().tag(sync=True)

    # Line profile
    profile_line = traitlets.List(traitlets.Dict()).tag(sync=True)

    # Display options
    title = traitlets.Unicode("").tag(sync=True)
    show_stats = traitlets.Bool(True).tag(sync=True)

    # Colormap and contrast (synced for state portability)
    cmap = traitlets.Unicode("gray").tag(sync=True)
    auto_contrast = traitlets.Bool(True).tag(sync=True)
    log_scale = traitlets.Bool(False).tag(sync=True)
    show_fft = traitlets.Bool(False).tag(sync=True)
    fft_window = traitlets.Bool(True).tag(sync=True)

    # Canvas sizing
    canvas_size = traitlets.Int(0).tag(sync=True)

    # Control visibility
    show_controls = traitlets.Bool(True).tag(sync=True)

    # Optional UI/tool lockout for shared notebooks
    disabled_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)
    hidden_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)

    @classmethod
    def _normalize_tool_groups(cls, tool_groups) -> List[str]:
        """Validate and normalize tool group values with stable ordering."""
        return normalize_tool_groups("Mark2D", tool_groups)

    @classmethod
    def _build_disabled_tools(
        cls,
        disabled_tools=None,
        disable_points: bool = False,
        disable_roi: bool = False,
        disable_profile: bool = False,
        disable_display: bool = False,
        disable_marker_style: bool = False,
        disable_snap: bool = False,
        disable_navigation: bool = False,
        disable_view: bool = False,
        disable_export: bool = False,
        disable_all: bool = False,
    ) -> List[str]:
        """Build disabled_tools from explicit list and ergonomic boolean flags."""
        return build_tool_groups(
            "Mark2D",
            tool_groups=disabled_tools,
            all_flag=disable_all,
            flag_map={
                "points": disable_points,
                "roi": disable_roi,
                "profile": disable_profile,
                "display": disable_display,
                "marker_style": disable_marker_style,
                "snap": disable_snap,
                "navigation": disable_navigation,
                "view": disable_view,
                "export": disable_export,
            },
        )

    @classmethod
    def _build_hidden_tools(
        cls,
        hidden_tools=None,
        hide_points: bool = False,
        hide_roi: bool = False,
        hide_profile: bool = False,
        hide_display: bool = False,
        hide_marker_style: bool = False,
        hide_snap: bool = False,
        hide_navigation: bool = False,
        hide_view: bool = False,
        hide_export: bool = False,
        hide_all: bool = False,
    ) -> List[str]:
        """Build hidden_tools from explicit list and ergonomic boolean flags."""
        return build_tool_groups(
            "Mark2D",
            tool_groups=hidden_tools,
            all_flag=hide_all,
            flag_map={
                "points": hide_points,
                "roi": hide_roi,
                "profile": hide_profile,
                "display": hide_display,
                "marker_style": hide_marker_style,
                "snap": hide_snap,
                "navigation": hide_navigation,
                "view": hide_view,
                "export": hide_export,
            },
        )

    @traitlets.validate("disabled_tools")
    def _validate_disabled_tools(self, proposal):
        return self._normalize_tool_groups(proposal["value"])

    @traitlets.validate("hidden_tools")
    def _validate_hidden_tools(self, proposal):
        return self._normalize_tool_groups(proposal["value"])

    # Percentile clipping
    percentile_low = traitlets.Float(2.0).tag(sync=True)
    percentile_high = traitlets.Float(98.0).tag(sync=True)

    # Per-image statistics
    stats_mean = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_min = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_max = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_std = traitlets.List(traitlets.Float()).tag(sync=True)

    def __init__(
        self,
        data,
        scale: float = 1.0,
        dot_size: int = 12,
        max_points: int = 10,
        ncols: int = 3,
        labels: Optional[List[str]] = None,
        marker_border: int = 2,
        marker_opacity: float = 1.0,
        label_size: int = 0,
        label_color: str = "",
        pixel_size: float = 0.0,
        points=None,
        marker_shape: str = "circle",
        marker_color: str = "#f44336",
        snap_enabled: bool = False,
        snap_radius: int = 5,
        title: str = "",
        show_stats: bool = True,
        cmap: str = "gray",
        auto_contrast: bool = True,
        log_scale: bool = False,
        show_fft: bool = False,
        fft_window: bool = True,
        canvas_size: int = 0,
        show_controls: bool = True,
        disabled_tools: Optional[List[str]] = None,
        disable_points: bool = False,
        disable_roi: bool = False,
        disable_profile: bool = False,
        disable_display: bool = False,
        disable_marker_style: bool = False,
        disable_snap: bool = False,
        disable_navigation: bool = False,
        disable_view: bool = False,
        disable_export: bool = False,
        disable_all: bool = False,
        hidden_tools: Optional[List[str]] = None,
        hide_points: bool = False,
        hide_roi: bool = False,
        hide_profile: bool = False,
        hide_display: bool = False,
        hide_marker_style: bool = False,
        hide_snap: bool = False,
        hide_navigation: bool = False,
        hide_view: bool = False,
        hide_export: bool = False,
        hide_all: bool = False,
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
        state=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.widget_version = resolve_widget_version()
        self.show_stats = show_stats
        self.scale = scale
        self.dot_size = dot_size
        self.max_points = max_points
        self.ncols = ncols
        self.marker_border = marker_border
        self.marker_opacity = marker_opacity
        self.label_size = label_size
        self.label_color = label_color
        self.marker_shape = marker_shape
        self.marker_color = marker_color
        self.snap_enabled = snap_enabled
        self.snap_radius = snap_radius
        self.cmap = cmap
        self.auto_contrast = auto_contrast
        self.log_scale = log_scale
        self.show_fft = show_fft
        self.fft_window = fft_window
        self.canvas_size = canvas_size
        self.show_controls = show_controls
        self.disabled_tools = self._build_disabled_tools(
            disabled_tools=disabled_tools,
            disable_points=disable_points,
            disable_roi=disable_roi,
            disable_profile=disable_profile,
            disable_display=disable_display,
            disable_marker_style=disable_marker_style,
            disable_snap=disable_snap,
            disable_navigation=disable_navigation,
            disable_view=disable_view,
            disable_export=disable_export,
            disable_all=disable_all,
        )
        self.hidden_tools = self._build_hidden_tools(
            hidden_tools=hidden_tools,
            hide_points=hide_points,
            hide_roi=hide_roi,
            hide_profile=hide_profile,
            hide_display=hide_display,
            hide_marker_style=hide_marker_style,
            hide_snap=hide_snap,
            hide_navigation=hide_navigation,
            hide_view=hide_view,
            hide_export=hide_export,
            hide_all=hide_all,
        )
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self._set_data(data, labels)
        # Explicit overrides take priority over Dataset metadata
        if title:
            self.title = title
        if pixel_size != 0.0:
            self.pixel_size = pixel_size
        if points is not None:
            self.selected_points = self._normalize_points(points)
        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(
                    json.loads(pathlib.Path(state).read_text()),
                    require_envelope=True,
                )
            else:
                state = unwrap_state_payload(state)
            self.load_state_dict(state)

    def _set_data(self, data, labels=None):
        # Check if data is a Dataset2d and extract metadata
        if hasattr(data, "array") and hasattr(data, "name") and hasattr(data, "sampling"):
            if data.name:
                self.title = data.name
            if hasattr(data, "units"):
                units = list(data.units)
                sampling_val = float(data.sampling[-1])
                if units[-1] in ("nm",):
                    self.pixel_size = sampling_val * 10  # nm → Å
                elif units[-1] in ("Å", "angstrom", "A"):
                    self.pixel_size = sampling_val
            data = data.array

        if isinstance(data, list):
            images = [to_numpy(d) for d in data]
            for img in images:
                if img.ndim != 2:
                    raise ValueError("Each image in the list must be 2D (H, W).")
            shapes = [img.shape for img in images]
            if len(set(shapes)) > 1:
                max_h = max(s[0] for s in shapes)
                max_w = max(s[1] for s in shapes)
                images = [_resize_image(img, max_h, max_w) for img in images]
            arr = np.stack(images).astype(np.float32)
        else:
            arr = to_numpy(data).astype(np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            elif arr.ndim == 3:
                pass  # (N, H, W) gallery
            else:
                raise ValueError("Expected 2D (H,W) or 3D (N,H,W) array, or list of 2D arrays.")

        self._data = arr
        n, h, w = arr.shape
        self.n_images = n
        self.height = h
        self.width = w

        # Per-image min/max and statistics
        mins, maxs, means, stds = [], [], [], []
        for i in range(n):
            frame = arr[i]
            mins.append(float(frame.min()))
            maxs.append(float(frame.max()))
            means.append(float(frame.mean()))
            stds.append(float(frame.std()))
        self.img_min = mins
        self.img_max = maxs
        self.stats_mean = means
        self.stats_min = mins
        self.stats_max = maxs
        self.stats_std = stds

        # Labels
        if labels is not None:
            self.labels = list(labels)
        else:
            self.labels = [f"Image {i + 1}" for i in range(n)]

        # Frame bytes (raw float32, all images concatenated)
        self.frame_bytes = arr.tobytes()

        # Reset points
        if n == 1:
            self.selected_points = []
        else:
            self.selected_points = [[] for _ in range(n)]

        self.selected_idx = 0

    def _normalize_point(self, p, idx):
        if isinstance(p, dict):
            return {
                "row": int(p["row"]),
                "col": int(p["col"]),
                "shape": p.get("shape", _MARKER_SHAPES[idx % len(_MARKER_SHAPES)]),
                "color": p.get("color", _MARKER_COLORS[idx % len(_MARKER_COLORS)]),
            }
        if isinstance(p, (list, tuple)) and len(p) == 2:
            return {
                "row": int(p[0]),
                "col": int(p[1]),
                "shape": _MARKER_SHAPES[idx % len(_MARKER_SHAPES)],
                "color": _MARKER_COLORS[idx % len(_MARKER_COLORS)],
            }
        raise ValueError(f"Invalid point format: {p}")

    def _normalize_point_list(self, pts):
        if isinstance(pts, np.ndarray):
            if pts.ndim == 2 and pts.shape[1] == 2:
                return [self._normalize_point((int(pts[i, 0]), int(pts[i, 1])), i)
                        for i in range(pts.shape[0])]
            raise ValueError(f"Expected (N, 2) array, got shape {pts.shape}")
        return [self._normalize_point(p, i) for i, p in enumerate(pts)]

    def _normalize_points(self, raw_points):
        if self.n_images == 1:
            return self._normalize_point_list(raw_points)
        # Gallery: expect list of point lists, one per image
        if not isinstance(raw_points, (list, tuple)):
            raise ValueError("Gallery mode requires list of point lists")
        if len(raw_points) != self.n_images:
            raise ValueError(
                f"Expected {self.n_images} point lists, got {len(raw_points)}"
            )
        return [self._normalize_point_list(pts) for pts in raw_points]

    def set_image(self, data, labels=None):
        """
        Replace the displayed image(s) and reset all points.

        Can switch between single-image and gallery modes. All existing
        points are cleared; ROIs are preserved.

        Parameters
        ----------
        data : array_like
            2D array ``(H, W)``, 3D array ``(N, H, W)``, or list of 2D
            arrays. Same formats as the constructor.
        labels : list of str, optional
            Per-image labels for gallery mode. If ``None``, defaults to
            ``"Image 1"``, ``"Image 2"``, etc.

        Examples
        --------
        >>> w = Mark2D(img1)
        >>> w.set_image(img2)  # switch to a different image
        >>> w.set_image([img1, img2, img3], labels=["A", "B", "C"])
        """
        self._set_data(data, labels)

    def add_roi(self, row, col, shape="square", radius=30, width=60, height=40,
                color="#0f0", opacity=0.8):
        """
        Add an ROI overlay to the widget.

        Multiple ROIs can be added. Each gets a unique ID. In the widget,
        the user can click ROI centers to select them, drag to reposition,
        and adjust size/color/opacity interactively. The widget also displays
        pixel statistics (mean, std, min, max) for the active ROI.

        Parameters
        ----------
        row, col : int
            Center position in image pixel coordinates (row, col).
        shape : str, default "circle"
            ROI shape. One of ``"circle"``, ``"square"``, or ``"rectangle"``.
        radius : int, default 30
            Radius in pixels for circle and square modes.
        width : int, default 60
            Width in pixels for rectangle mode.
        height : int, default 40
            Height in pixels for rectangle mode.
        color : str, default "#0f0"
            Stroke color as a CSS color string (e.g. ``"#ff0"``, ``"red"``).
        opacity : float, default 0.8
            Stroke opacity (0.1–1.0).

        Examples
        --------
        >>> w = Mark2D(img)
        >>> w.add_roi(128, 128)  # green circle at center
        >>> w.add_roi(50, 50, shape="square", radius=20, color="#ff0")
        >>> w.add_roi(200, 100, shape="rectangle", width=80, height=30)
        >>> len(w.roi_list)  # 3
        """
        roi_id = max((r["id"] for r in self.roi_list), default=-1) + 1
        roi = {
            "id": roi_id,
            "shape": shape,
            "row": int(row), "col": int(col),
            "radius": int(radius),
            "width": int(width), "height": int(height),
            "color": color, "opacity": float(opacity),
        }
        self.roi_list = [*self.roi_list, roi]

    def clear_rois(self):
        """
        Remove all ROI overlays.

        Examples
        --------
        >>> w.add_roi(100, 100)
        >>> w.add_roi(200, 200)
        >>> w.clear_rois()
        >>> w.roi_list  # []
        """
        self.roi_list = []

    def _resolve_roi(self, index: Optional[int] = None, roi_id: Optional[int] = None):
        """Resolve one ROI by index or id (defaults to the most recently added ROI)."""
        if index is not None and roi_id is not None:
            raise ValueError("Pass either index or roi_id, not both.")
        if not self.roi_list:
            raise ValueError("No ROIs are defined.")

        if roi_id is not None:
            target_id = int(roi_id)
            for roi in self.roi_list:
                if int(roi.get("id", -1)) == target_id:
                    return roi
            raise ValueError(f"ROI id {roi_id} not found.")

        idx = -1 if index is None else int(index)
        try:
            return self.roi_list[idx]
        except IndexError as exc:
            raise IndexError(
                f"ROI index {idx} out of range for {len(self.roi_list)} ROIs."
            ) from exc

    def roi_center(self, index: Optional[int] = None, roi_id: Optional[int] = None):
        """
        Return ROI center coordinates as ``(row, col)``.

        By default, returns the center of the most recently added ROI.

        Parameters
        ----------
        index : int, optional
            ROI list index to query. Supports negative indexing.
        roi_id : int, optional
            ROI ``id`` value to query. Mutually exclusive with ``index``.

        Returns
        -------
        tuple of int
            Center point ``(row, col)``.
        """
        roi = self._resolve_roi(index=index, roi_id=roi_id)
        return int(roi["row"]), int(roi["col"])

    def roi_radius(self, index: Optional[int] = None, roi_id: Optional[int] = None):
        """
        Return ROI radius for ``circle``/``square`` ROIs.

        For ``rectangle`` ROIs, returns ``None`` (use ``roi_size()`` for
        rectangle width/height).
        By default, queries the most recently added ROI.

        Parameters
        ----------
        index : int, optional
            ROI list index to query. Supports negative indexing.
        roi_id : int, optional
            ROI ``id`` value to query. Mutually exclusive with ``index``.

        Returns
        -------
        int or None
            Radius in pixels for circle/square ROIs, otherwise ``None``.
        """
        roi = self._resolve_roi(index=index, roi_id=roi_id)
        if roi.get("shape") == "rectangle":
            return None
        return int(roi["radius"])

    def roi_size(self, index: Optional[int] = None, roi_id: Optional[int] = None):
        """
        Return shape-aware ROI size information.

        - ``circle`` / ``square`` -> ``{"shape", "radius", "diameter"}``
        - ``rectangle`` -> ``{"shape", "width", "height"}``

        Parameters
        ----------
        index : int, optional
            ROI list index to query. Supports negative indexing.
        roi_id : int, optional
            ROI ``id`` value to query. Mutually exclusive with ``index``.

        Returns
        -------
        dict
            Shape-aware size dictionary for the selected ROI.
        """
        roi = self._resolve_roi(index=index, roi_id=roi_id)
        shape = str(roi.get("shape", "circle"))
        if shape == "rectangle":
            return {
                "shape": shape,
                "width": int(roi["width"]),
                "height": int(roi["height"]),
            }
        radius = int(roi["radius"])
        return {
            "shape": shape,
            "radius": radius,
            "diameter": 2 * radius,
        }

    def _sample_profile(self, row0, col0, row1, col1):
        """Sample intensity values along a line using bilinear interpolation."""
        idx = self.selected_idx if self.n_images > 1 else 0
        img = self._data[idx]
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
        vals = (img[r0c, c0c] * (1 - cf) * (1 - rf) +
                img[r0c, c1c] * cf * (1 - rf) +
                img[r1c, c0c] * (1 - cf) * rf +
                img[r1c, c1c] * cf * rf)
        return vals.astype(np.float32)

    def set_profile(self, start: tuple, end: tuple):
        """
        Set a line profile between two points.

        The profile is drawn on the canvas and intensity values are sampled
        along the line with bilinear interpolation. A sparkline graph appears
        below the canvas.

        Parameters
        ----------
        start : tuple of (row, col)
            Start point in image pixel coordinates.
        end : tuple of (row, col)
            End point in image pixel coordinates.

        Examples
        --------
        >>> w = Mark2D(img, pixel_size=0.82)
        >>> w.set_profile((10, 20), (100, 200))
        >>> w.profile_values  # sampled intensities along the line
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

    @property
    def profile(self):
        """
        Get profile line endpoints as ``[(row0, col0), (row1, col1)]`` or ``[]``.

        Returns
        -------
        list of tuple
            Line endpoints in pixel coordinates, or empty list if no profile.
        """
        return [(p["row"], p["col"]) for p in self.profile_line]

    @property
    def profile_values(self):
        """
        Get intensity values along the profile line as a numpy array.

        Returns
        -------
        np.ndarray or None
            Float32 array of sampled intensities, or ``None`` if no profile.
        """
        if len(self.profile_line) < 2:
            return None
        p0, p1 = self.profile_line
        return self._sample_profile(p0["row"], p0["col"], p1["row"], p1["col"])

    @property
    def profile_distance(self):
        """
        Get total distance of the profile line in calibrated units.

        Returns
        -------
        float or None
            Distance in angstroms (if ``pixel_size > 0``) or pixels.
            ``None`` if no profile line is set.
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

    def __repr__(self) -> str:
        is_gallery = self.n_images > 1
        # Points count
        if is_gallery:
            per_img = [len(pts) if isinstance(pts, list) else 0
                       for pts in self.selected_points]
            pts_str = "+".join(str(n) for n in per_img)
            total = sum(per_img)
        else:
            total = len(self.selected_points)
            pts_str = str(total)

        # Shape string
        if is_gallery:
            shape = f"{self.n_images}×{self.height}×{self.width}"
        else:
            shape = f"{self.height}×{self.width}"

        name = self.title if self.title else "Mark2D"
        parts = [f"{name}({shape}"]

        if is_gallery:
            parts.append(f"idx={self.selected_idx}")

        # Pixel size
        if self.pixel_size > 0:
            ps = self.pixel_size
            if ps >= 10:
                parts.append(f"px={ps / 10:.2f} nm")
            else:
                parts.append(f"px={ps:.2f} Å")

        # Points
        if total > 0:
            parts.append(f"pts={pts_str}")
        else:
            parts.append("pts=0")

        # ROIs
        n_rois = len(self.roi_list)
        if n_rois > 0:
            parts.append(f"rois={n_rois}")

        # Non-default imaging settings
        if self.cmap != "gray":
            parts.append(f"cmap={self.cmap}")
        if self.log_scale:
            parts.append("log")
        if not self.auto_contrast:
            parts.append("manual contrast")
        if self.show_fft:
            parts.append("fft")
        if self.snap_enabled:
            parts.append(f"snap r={self.snap_radius}")

        return ", ".join(parts) + ")"

    def summary(self):
        """
        Print a detailed summary of the widget state.

        Shows image info, display settings, points with coordinates,
        ROI details, and marker configuration.

        Examples
        --------
        >>> w = Mark2D(img, points=[(10, 20), (30, 40)],
        ...             pixel_size=0.82, cmap='viridis')
        >>> w.summary()
        Mark2D
        ═══════════════════════════════
        Image:    128×128 (0.82 Å/px)
        Display:  viridis | auto contrast | linear
        ...
        """
        is_gallery = self.n_images > 1
        name = self.title if self.title else "Mark2D"
        lines = [name, "═" * 32]

        # Image info
        if is_gallery:
            shape = f"{self.n_images}×{self.height}×{self.width}"
            lines.append(f"Image:    {shape} ({self.ncols} cols)")
        else:
            shape = f"{self.height}×{self.width}"
            lines.append(f"Image:    {shape}")
        if self.pixel_size > 0:
            ps = self.pixel_size
            if ps >= 10:
                lines[-1] += f" ({ps / 10:.2f} nm/px)"
            else:
                lines[-1] += f" ({ps:.2f} Å/px)"
        if self.scale != 1.0:
            lines[-1] += f"  scale={self.scale}x"

        # Data range
        if hasattr(self, "_data") and self._data is not None:
            arr = self._data
            lines.append(f"Data:     min={float(arr.min()):.4g}  max={float(arr.max()):.4g}  mean={float(arr.mean()):.4g}  dtype={arr.dtype}")

        # Display settings
        cmap = self.cmap
        scale = "log" if self.log_scale else "linear"
        contrast = "auto contrast" if self.auto_contrast else "manual contrast"
        display = f"{cmap} | {contrast} | {scale}"
        if self.show_fft:
            display += " | FFT"
            if not self.fft_window:
                display += " (no window)"
        lines.append(f"Display:  {display}")

        # Point formatting helper
        def _fmt_point(j, p, prev=None):
            color = _color_name(p.get("color", ""))
            coord = f"  {j + 1}. ({p['row']}, {p['col']})  {p.get('shape', 'circle')} {color}"
            if prev is not None:
                dr, dc = p["row"] - prev["row"], p["col"] - prev["col"]
                dist = (dr * dr + dc * dc) ** 0.5
                if self.pixel_size > 0:
                    phys = dist * self.pixel_size
                    if phys >= 10:
                        coord += f"  ↔ {phys / 10:.2f} nm"
                    else:
                        coord += f"  ↔ {phys:.2f} Å"
                else:
                    coord += f"  ↔ {dist:.1f} px"
            return coord

        # Points
        if is_gallery:
            for i in range(self.n_images):
                pts = self.selected_points[i] if i < len(self.selected_points) else []
                label = self.labels[i] if i < len(self.labels) else f"Image {i + 1}"
                lines.append(f"Points [{label}]: {len(pts)}/{self.max_points}")
                for j, p in enumerate(pts):
                    lines.append(_fmt_point(j, p, pts[j - 1] if j > 0 else None))
        else:
            pts = self.selected_points
            lines.append(f"Points:   {len(pts)}/{self.max_points}")
            for j, p in enumerate(pts):
                lines.append(_fmt_point(j, p, pts[j - 1] if j > 0 else None))

        # ROIs
        if self.roi_list:
            lines.append(f"ROIs:     {len(self.roi_list)}")
            for roi in self.roi_list:
                mode = roi["shape"]
                pos = f"({roi['row']}, {roi['col']})"
                if mode == "rectangle":
                    size = f"{roi['width']}×{roi['height']}"
                    area_px = roi["width"] * roi["height"]
                elif mode == "circle":
                    size = f"r={roi['radius']}"
                    area_px = 3.14159265 * roi["radius"] ** 2
                else:  # square
                    size = f"r={roi['radius']}"
                    area_px = (2 * roi["radius"]) ** 2
                color = _color_name(roi["color"])
                if self.pixel_size > 0:
                    ps = self.pixel_size
                    area_phys = area_px * ps * ps
                    if area_phys >= 100:
                        area_str = f"  area={area_phys / 100:.1f} nm²"
                    else:
                        area_str = f"  area={area_phys:.1f} Å²"
                else:
                    area_str = f"  area={area_px:.0f} px²"
                lines.append(f"  {roi['id']+1}. {mode} at {pos}  {size}  {color}{area_str}")

        # Marker settings
        color = _color_name(self.marker_color)
        marker = f"{self.marker_shape} {color}  size={self.dot_size}px"
        if self.marker_border != 2:
            marker += f"  border={self.marker_border}"
        if self.marker_opacity != 1.0:
            marker += f"  opacity={self.marker_opacity:.0%}"
        lines.append(f"Marker:   {marker}")

        # Snap
        if self.snap_enabled:
            lines.append(f"Snap:     ON (radius={self.snap_radius} px)")
        if self.disabled_tools:
            lines.append(f"Locked:   {', '.join(self.disabled_tools)}")
        if self.hidden_tools:
            lines.append(f"Hidden:   {', '.join(self.hidden_tools)}")

        print("\n".join(lines))

    def points_as_array(self):
        """
        Return placed points as a NumPy array of shape ``(N, 2)`` with columns ``[row, col]``.

        In gallery mode, returns a list of arrays (one per image).

        Examples
        --------
        >>> w = Mark2D(img, points=[(10, 20), (30, 40)])
        >>> w.points_as_array()
        array([[10, 20],
               [30, 40]])
        """
        if self.n_images > 1:
            result = []
            for pts in self.selected_points:
                if pts:
                    result.append(np.array([[p["row"], p["col"]] for p in pts], dtype=np.float64))
                else:
                    result.append(np.empty((0, 2), dtype=np.float64))
            return result
        pts = self.selected_points
        if not pts:
            return np.empty((0, 2), dtype=np.float64)
        return np.array([[p["row"], p["col"]] for p in pts], dtype=np.float64)

    def points_as_dict(self):
        """
        Return placed points as a list of ``{"row": int, "col": int}`` dicts.

        In gallery mode, returns a list of lists (one per image).

        Examples
        --------
        >>> w = Mark2D(img, points=[(10, 20), (30, 40)])
        >>> w.points_as_dict()
        [{'row': 10, 'col': 20}, {'row': 30, 'col': 40}]
        """
        if self.n_images > 1:
            return [
                [{"row": p["row"], "col": p["col"]} for p in pts]
                for pts in self.selected_points
            ]
        return [{"row": p["row"], "col": p["col"]} for p in self.selected_points]

    def clear_points(self):
        """
        Remove all placed points from all images.

        Examples
        --------
        >>> w.clear_points()
        >>> w.selected_points  # [] or [[], [], ...]
        """
        if self.n_images == 1:
            self.selected_points = []
        else:
            self.selected_points = [[] for _ in range(self.n_images)]

    @property
    def points_enabled(self) -> bool:
        """
        Whether adding/editing points is enabled via ``disabled_tools``.

        This convenience toggle controls the ``"points"`` lock in
        ``disabled_tools``. It does not modify ``hidden_tools``.
        """
        disabled = {str(t).strip().lower() for t in self.disabled_tools}
        return "all" not in disabled and "points" not in disabled

    @points_enabled.setter
    def points_enabled(self, enabled: bool):
        enabled = bool(enabled)
        disabled = [str(t).strip().lower() for t in self.disabled_tools]

        if enabled:
            if "all" in disabled:
                raise ValueError(
                    "Cannot enable points while disabled_tools contains 'all'. "
                    "Remove 'all' first."
                )
            if "points" in disabled:
                self.disabled_tools = [t for t in disabled if t != "points"]
            return

        if "all" in disabled or "points" in disabled:
            return
        self.disabled_tools = [*disabled, "points"]

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.log_scale:
            frame = np.log1p(np.maximum(frame, 0))
        if self.auto_contrast:
            vmin = float(np.percentile(frame, self.percentile_low))
            vmax = float(np.percentile(frame, self.percentile_high))
        else:
            vmin = float(frame.min())
            vmax = float(frame.max())
        if vmax > vmin:
            normalized = np.clip((frame - vmin) / (vmax - vmin) * 255, 0, 255)
            return normalized.astype(np.uint8)
        return np.zeros(frame.shape, dtype=np.uint8)

    def save_image(
        self,
        path: str | pathlib.Path,
        *,
        idx: int | None = None,
        include_markers: bool = True,
        format: str | None = None,
        dpi: int = 150,
    ) -> pathlib.Path:
        """Save current image as PNG or PDF, optionally with marker overlays.

        Parameters
        ----------
        path : str or pathlib.Path
            Output file path.
        idx : int, optional
            Image index in gallery mode. Defaults to current selected_idx.
        include_markers : bool, default True
            If True, render marker points on the exported image.
        format : str, optional
            'png' or 'pdf'. If omitted, inferred from file extension.
        dpi : int, default 150
            Output DPI metadata.

        Returns
        -------
        pathlib.Path
            The written file path.
        """
        from matplotlib import colormaps
        from PIL import Image, ImageDraw

        path = pathlib.Path(path)
        fmt = (format or path.suffix.lstrip(".").lower() or "png").lower()
        if fmt not in ("png", "pdf", "tiff", "tif"):
            raise ValueError(f"Unsupported format: {fmt!r}. Use 'png', 'pdf', or 'tiff'.")

        i = idx if idx is not None else self.selected_idx
        if i < 0 or i >= self.n_images:
            raise IndexError(f"Image index {i} out of range [0, {self.n_images})")

        frame = self._data[i]
        normalized = self._normalize_frame(frame)
        cmap_fn = colormaps.get_cmap(self.cmap)
        rgba = (cmap_fn(normalized / 255.0) * 255).astype(np.uint8)

        img = Image.fromarray(rgba)

        if include_markers:
            # In gallery mode, points are nested per image; single-image is flat
            if self.n_images > 1:
                pts = self.selected_points[i] if i < len(self.selected_points) else []
            else:
                pts = self.selected_points

            if pts:
                draw = ImageDraw.Draw(img)
                r = max(2, self.dot_size // 2)
                for pt in pts:
                    row, col = pt.get("row", 0), pt.get("col", 0)
                    color = pt.get("color", "#f44336")
                    draw.ellipse(
                        [col - r, row - r, col + r, row + r],
                        fill=color,
                        outline="white",
                    )

        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path), dpi=(dpi, dpi))
        return path

    def state_dict(self):
        """
        Return a dict of all restorable widget state.

        Use this to persist the widget state across kernel restarts.
        Pass the returned dict as the ``state`` parameter to a new
        ``Mark2D`` to restore everything.

        Examples
        --------
        >>> w = Mark2D(img, pixel_size=1.5)
        >>> # ... user places points, adds ROIs, changes settings ...
        >>> state = w.state_dict()
        >>> # Later (or after kernel restart):
        >>> w2 = Mark2D(img, state=state)
        """
        return {
            "selected_points": self.selected_points,
            "roi_list": self.roi_list,
            "profile_line": self.profile_line,
            "selected_idx": self.selected_idx,
            "marker_shape": self.marker_shape,
            "marker_color": self.marker_color,
            "dot_size": self.dot_size,
            "max_points": self.max_points,
            "marker_border": self.marker_border,
            "marker_opacity": self.marker_opacity,
            "label_size": self.label_size,
            "label_color": self.label_color,
            "snap_enabled": self.snap_enabled,
            "snap_radius": self.snap_radius,
            "cmap": self.cmap,
            "auto_contrast": self.auto_contrast,
            "log_scale": self.log_scale,
            "show_fft": self.show_fft,
            "fft_window": self.fft_window,
            "show_stats": self.show_stats,
            "show_controls": self.show_controls,
            "disabled_tools": self.disabled_tools,
            "hidden_tools": self.hidden_tools,
            "percentile_low": self.percentile_low,
            "percentile_high": self.percentile_high,
            "title": self.title,
            "pixel_size": self.pixel_size,
            "scale": self.scale,
            "canvas_size": self.canvas_size,
        }

    def save(self, path: str):
        """
        Save widget state to a JSON file.

        Parameters
        ----------
        path : str
            File path to write (e.g. ``"analysis.json"``).

        Examples
        --------
        >>> w = Mark2D(img)
        >>> # ... place points, add ROIs ...
        >>> w.save("my_analysis.json")
        >>> # After kernel restart:
        >>> w2 = Mark2D(img, state="my_analysis.json")
        """
        save_state_file(path, "Mark2D", self.state_dict())

    def load_state_dict(self, state):
        """
        Restore widget state from a dict returned by ``state_dict()``.

        Parameters
        ----------
        state : dict
            State dict from a previous ``state_dict()`` call.
            Missing keys are silently skipped.

        Examples
        --------
        >>> state = old_widget.state_dict()
        >>> new_widget = Mark2D(img)
        >>> new_widget.load_state_dict(state)
        """
        COMPAT_MAP = {"colormap": "cmap", "pixel_size_angstrom": "pixel_size"}
        ROI_KEY_MAP = {"mode": "shape", "rectW": "width", "rectH": "height"}
        for key, val in state.items():
            key = COMPAT_MAP.get(key, key)
            if key == "roi_list" and isinstance(val, list):
                val = [
                    {ROI_KEY_MAP.get(k, k): v for k, v in roi.items()}
                    for roi in val
                ]
            if hasattr(self, key):
                setattr(self, key, val)


bind_tool_runtime_api(Mark2D, "Mark2D")
