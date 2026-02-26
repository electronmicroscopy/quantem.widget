"""
show2d: Static 2D image viewer with optional FFT and histogram analysis.

For displaying a single image or a static gallery of multiple images.
Unlike Show3D (interactive), Show2D focuses on static visualization.
"""

import json
import pathlib
import io
import base64
import math
from enum import StrEnum
from typing import Optional, Union, List, Self

import anywidget
import matplotlib.pyplot as plt
import numpy as np
import traitlets

from quantem.widget.array_utils import to_numpy, _resize_image
from quantem.widget.io import IO, IOResult
from quantem.widget.json_state import resolve_widget_version, save_state_file, unwrap_state_payload
from quantem.widget.tool_parity import (
    bind_tool_runtime_api,
    build_tool_groups,
    normalize_tool_groups,
)



class Colormap(StrEnum):
    INFERNO = "inferno"
    VIRIDIS = "viridis"
    MAGMA = "magma"
    PLASMA = "plasma"
    GRAY = "gray"


class Show2D(anywidget.AnyWidget):
    """
    Static 2D image viewer with optional FFT and histogram analysis.

    Display a single image or multiple images in a gallery layout.
    For interactive stack viewing with playback, use Show3D instead.

    Parameters
    ----------
    data : array_like
        2D array (height, width) for single image, or
        3D array (N, height, width) for multiple images displayed as gallery.
    labels : list of str, optional
        Labels for each image in gallery mode.
    title : str, optional
        Title to display above the image(s).
    cmap : str, default "inferno"
        Colormap name ("magma", "viridis", "gray", "inferno", "plasma").
    pixel_size : float, optional
        Pixel size in angstroms for scale bar display.
    show_fft : bool, default False
        Show FFT and histogram panels.
    show_stats : bool, default True
        Show statistics (mean, min, max, std).
    log_scale : bool, default False
        Use log scale for intensity mapping.
    auto_contrast : bool, default False
        Use percentile-based contrast.
    ncols : int, default 3
        Number of columns in gallery mode.
    disabled_tools : list of str, optional
        Tool groups to lock while still showing controls. Supported:
        ``"display"``, ``"histogram"``, ``"stats"``, ``"navigation"``,
        ``"view"``, ``"export"``, ``"roi"``, ``"profile"``, ``"all"``.
    disable_* : bool, optional
        Convenience flags (``disable_display``, ``disable_histogram``,
        ``disable_stats``, ``disable_navigation``, ``disable_view``,
        ``disable_export``, ``disable_roi``, ``disable_profile``,
        ``disable_all``) equivalent to adding those keys to
        ``disabled_tools``.
    hidden_tools : list of str, optional
        Tool groups to hide from the UI. Uses the same keys as
        ``disabled_tools``.
    hide_* : bool, optional
        Convenience flags mirroring ``disable_*`` for ``hidden_tools``.

    Examples
    --------
    >>> import numpy as np
    >>> from quantem.widget import Show2D
    >>>
    >>> # Single image with FFT
    >>> Show2D(image, title="HRTEM Image", show_fft=True, pixel_size=1.0)
    >>>
    >>> # Gallery of multiple images
    >>> labels = ["Raw", "Filtered", "FFT"]
    >>> Show2D([img1, img2, img3], labels=labels, ncols=3)
    """

    _esm = pathlib.Path(__file__).parent / "static" / "show2d.js"
    _css = pathlib.Path(__file__).parent / "static" / "show2d.css"

    # =========================================================================
    # Core State
    # =========================================================================
    widget_version = traitlets.Unicode("unknown").tag(sync=True)
    n_images = traitlets.Int(1).tag(sync=True)
    height = traitlets.Int(1).tag(sync=True)
    width = traitlets.Int(1).tag(sync=True)
    frame_bytes = traitlets.Bytes(b"").tag(sync=True)
    labels = traitlets.List(traitlets.Unicode()).tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    cmap = traitlets.Unicode("inferno").tag(sync=True)
    ncols = traitlets.Int(3).tag(sync=True)

    # =========================================================================
    # Display Options
    # =========================================================================
    log_scale = traitlets.Bool(False).tag(sync=True)
    auto_contrast = traitlets.Bool(False).tag(sync=True)

    # =========================================================================
    # Scale Bar
    # =========================================================================
    pixel_size = traitlets.Float(0.0).tag(sync=True)
    scale_bar_visible = traitlets.Bool(True).tag(sync=True)
    canvas_size = traitlets.Int(0).tag(sync=True)

    # =========================================================================
    # UI Visibility
    # =========================================================================
    show_controls = traitlets.Bool(True).tag(sync=True)
    show_stats = traitlets.Bool(True).tag(sync=True)
    disabled_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)
    hidden_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)
    stats_mean = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_min = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_max = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_std = traitlets.List(traitlets.Float()).tag(sync=True)

    # =========================================================================
    # Analysis Panels (FFT + Histogram shown together)
    # =========================================================================
    show_fft = traitlets.Bool(False).tag(sync=True)
    fft_window = traitlets.Bool(True).tag(sync=True)

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

    @classmethod
    def _normalize_tool_groups(cls, tool_groups) -> List[str]:
        return normalize_tool_groups("Show2D", tool_groups)

    @classmethod
    def _build_disabled_tools(
        cls,
        disabled_tools=None,
        disable_display: bool = False,
        disable_histogram: bool = False,
        disable_stats: bool = False,
        disable_navigation: bool = False,
        disable_view: bool = False,
        disable_export: bool = False,
        disable_roi: bool = False,
        disable_profile: bool = False,
        disable_all: bool = False,
    ) -> List[str]:
        return build_tool_groups(
            "Show2D",
            tool_groups=disabled_tools,
            all_flag=disable_all,
            flag_map={
                "display": disable_display,
                "histogram": disable_histogram,
                "stats": disable_stats,
                "navigation": disable_navigation,
                "view": disable_view,
                "export": disable_export,
                "roi": disable_roi,
                "profile": disable_profile,
            },
        )

    @classmethod
    def _build_hidden_tools(
        cls,
        hidden_tools=None,
        hide_display: bool = False,
        hide_histogram: bool = False,
        hide_stats: bool = False,
        hide_navigation: bool = False,
        hide_view: bool = False,
        hide_export: bool = False,
        hide_roi: bool = False,
        hide_profile: bool = False,
        hide_all: bool = False,
    ) -> List[str]:
        return build_tool_groups(
            "Show2D",
            tool_groups=hidden_tools,
            all_flag=hide_all,
            flag_map={
                "display": hide_display,
                "histogram": hide_histogram,
                "stats": hide_stats,
                "navigation": hide_navigation,
                "view": hide_view,
                "export": hide_export,
                "roi": hide_roi,
                "profile": hide_profile,
            },
        )

    @traitlets.validate("disabled_tools")
    def _validate_disabled_tools(self, proposal):
        return self._normalize_tool_groups(proposal["value"])

    @traitlets.validate("hidden_tools")
    def _validate_hidden_tools(self, proposal):
        return self._normalize_tool_groups(proposal["value"])

    def __init__(
        self,
        data: Union[np.ndarray, List[np.ndarray]],
        labels: Optional[List[str]] = None,
        title: str = "",
        cmap: Union[str, Colormap] = Colormap.INFERNO,
        pixel_size: float = 0.0,
        scale_bar_visible: bool = True,
        show_fft: bool = False,
        fft_window: bool = True,
        show_controls: bool = True,
        show_stats: bool = True,
        log_scale: bool = False,
        auto_contrast: bool = False,
        disabled_tools: Optional[List[str]] = None,
        disable_display: bool = False,
        disable_histogram: bool = False,
        disable_stats: bool = False,
        disable_navigation: bool = False,
        disable_view: bool = False,
        disable_export: bool = False,
        disable_roi: bool = False,
        disable_profile: bool = False,
        disable_all: bool = False,
        hidden_tools: Optional[List[str]] = None,
        hide_display: bool = False,
        hide_histogram: bool = False,
        hide_stats: bool = False,
        hide_navigation: bool = False,
        hide_view: bool = False,
        hide_export: bool = False,
        hide_roi: bool = False,
        hide_profile: bool = False,
        hide_all: bool = False,
        ncols: int = 3,
        canvas_size: int = 0,
        state=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.widget_version = resolve_widget_version()

        # Check if data is an IOResult and extract metadata
        if isinstance(data, IOResult):
            if not title and data.title:
                title = data.title
            if pixel_size == 0.0 and data.pixel_size is not None:
                pixel_size = data.pixel_size
            if labels is None and data.labels:
                labels = data.labels
            data = data.data

        # Check if data is a Dataset2d and extract metadata
        if hasattr(data, "array") and hasattr(data, "name") and hasattr(data, "sampling"):
            if not title and data.name:
                title = data.name
            if pixel_size == 0.0 and hasattr(data, "units"):
                units = list(data.units)
                sampling_val = float(data.sampling[-1])
                if units[-1] in ("nm",):
                    pixel_size = sampling_val * 10  # nm → Å
                elif units[-1] in ("Å", "angstrom", "A"):
                    pixel_size = sampling_val
            data = data.array

        # Convert input to NumPy (handles NumPy, CuPy, PyTorch)
        if isinstance(data, list):
            images = [to_numpy(d) for d in data]

            # Check if all images have the same shape
            shapes = [img.shape for img in images]
            if len(set(shapes)) > 1:
                # Different sizes - resize all to the largest
                max_h = max(s[0] for s in shapes)
                max_w = max(s[1] for s in shapes)
                images = [_resize_image(img, max_h, max_w) for img in images]

            data = np.stack(images)
        else:
            data = to_numpy(data)

        # Ensure 3D shape (N, H, W)
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        self._data = data.astype(np.float32)
        self.n_images = int(data.shape[0])
        self.height = int(data.shape[1])
        self.width = int(data.shape[2])

        # Labels
        if labels is None:
            self.labels = [f"Image {i+1}" for i in range(self.n_images)]
        else:
            self.labels = list(labels)

        # Options
        self.title = title
        self.cmap = cmap
        self.pixel_size = pixel_size
        self.scale_bar_visible = scale_bar_visible
        self.canvas_size = canvas_size
        self.show_fft = show_fft
        self.fft_window = fft_window
        self.show_controls = show_controls
        self.show_stats = show_stats
        self.log_scale = log_scale
        self.auto_contrast = auto_contrast
        self.disabled_tools = self._build_disabled_tools(
            disabled_tools=disabled_tools,
            disable_display=disable_display,
            disable_histogram=disable_histogram,
            disable_stats=disable_stats,
            disable_navigation=disable_navigation,
            disable_view=disable_view,
            disable_export=disable_export,
            disable_roi=disable_roi,
            disable_profile=disable_profile,
            disable_all=disable_all,
        )
        self.hidden_tools = self._build_hidden_tools(
            hidden_tools=hidden_tools,
            hide_display=hide_display,
            hide_histogram=hide_histogram,
            hide_stats=hide_stats,
            hide_navigation=hide_navigation,
            hide_view=hide_view,
            hide_export=hide_export,
            hide_roi=hide_roi,
            hide_profile=hide_profile,
            hide_all=hide_all,
        )
        self.ncols = ncols

        # Compute initial stats
        self._compute_all_stats()

        # Send raw float32 data to JS (normalization happens in JS for speed)
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

    def set_image(self, data, labels=None):
        """Replace the displayed image(s). Preserves all display settings."""
        if hasattr(data, "array") and hasattr(data, "name") and hasattr(data, "sampling"):
            data = data.array
        if isinstance(data, list):
            images = [to_numpy(d) for d in data]
            shapes = [img.shape for img in images]
            if len(set(shapes)) > 1:
                max_h = max(s[0] for s in shapes)
                max_w = max(s[1] for s in shapes)
                images = [_resize_image(img, max_h, max_w) for img in images]
            data = np.stack(images)
        else:
            data = to_numpy(data)
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        self._data = data.astype(np.float32)
        self.n_images = int(data.shape[0])
        self.height = int(data.shape[1])
        self.width = int(data.shape[2])
        if labels is not None:
            self.labels = list(labels)
        else:
            self.labels = [f"Image {i+1}" for i in range(self.n_images)]
        self.selected_idx = 0
        self._compute_all_stats()
        self._update_all_frames()

    def __repr__(self) -> str:
        if self.n_images > 1:
            shape = f"{self.n_images}×{self.height}×{self.width}"
            return f"Show2D({shape}, idx={self.selected_idx}, cmap={self.cmap})"
        return f"Show2D({self.height}×{self.width}, cmap={self.cmap})"

    def _repr_mimebundle_(self, **kwargs):
        """Return widget view + static PNG fallback.

        Live Jupyter renders the interactive widget. Static contexts
        (nbsphinx, GitHub, nbviewer) fall back to the embedded PNG.
        """
        bundle = super()._repr_mimebundle_(**kwargs)
        data_dict = bundle[0] if isinstance(bundle, tuple) else bundle
        n = self.n_images
        ncols = min(self.ncols, n)
        nrows = math.ceil(n / ncols)
        cell = 4
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(cell * ncols, cell * nrows),
            squeeze=False,
        )
        for i in range(nrows * ncols):
            r, c = divmod(i, ncols)
            ax = axes[r][c]
            if i < n:
                ax.imshow(self._data[i], cmap=self.cmap, origin="upper")
                ax.set_title(self.labels[i], fontsize=10)
            ax.axis("off")
        if self.title:
            fig.suptitle(self.title, fontsize=12)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        data_dict["image/png"] = base64.b64encode(buf.getvalue()).decode("ascii")
        if isinstance(bundle, tuple):
            return (data_dict, bundle[1])
        return data_dict

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.log_scale:
            frame = np.log1p(np.maximum(frame, 0))
        if self.auto_contrast:
            vmin = float(np.percentile(frame, 2))
            vmax = float(np.percentile(frame, 98))
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
        format: str | None = None,
        dpi: int = 150,
    ) -> pathlib.Path:
        """Save current image as PNG or PDF.

        Parameters
        ----------
        path : str or pathlib.Path
            Output file path.
        idx : int, optional
            Image index in gallery mode. Defaults to current selected_idx.
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
        from PIL import Image

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
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path), dpi=(dpi, dpi))
        return path

    def state_dict(self):
        return {
            "title": self.title,
            "cmap": self.cmap,
            "log_scale": self.log_scale,
            "auto_contrast": self.auto_contrast,
            "show_stats": self.show_stats,
            "show_fft": self.show_fft,
            "fft_window": self.fft_window,
            "show_controls": self.show_controls,
            "disabled_tools": self.disabled_tools,
            "hidden_tools": self.hidden_tools,
            "pixel_size": self.pixel_size,
            "scale_bar_visible": self.scale_bar_visible,
            "canvas_size": self.canvas_size,
            "ncols": self.ncols,
            "selected_idx": self.selected_idx,
            "roi_active": self.roi_active,
            "roi_list": self.roi_list,
            "roi_selected_idx": self.roi_selected_idx,
            "profile_line": self.profile_line,
        }

    def save(self, path: str):
        save_state_file(path, "Show2D", self.state_dict())

    def load_state_dict(self, state):
        for key, val in state.items():
            if key == "pixel_size_angstrom":
                key = "pixel_size"
            if hasattr(self, key):
                setattr(self, key, val)

    def summary(self):
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
        contrast = "auto contrast" if self.auto_contrast else "manual contrast"
        display = f"{cmap} | {contrast} | {scale}"
        if self.show_fft:
            display += " | FFT"
            if not self.fft_window:
                display += " (no window)"
        lines.append(f"Display:  {display}")
        if self.disabled_tools:
            lines.append(f"Locked:   {', '.join(self.disabled_tools)}")
        if self.hidden_tools:
            lines.append(f"Hidden:   {', '.join(self.hidden_tools)}")
        if self.roi_active and self.roi_list:
            lines.append(f"ROI:      {len(self.roi_list)} region(s)")
        if self.profile_line:
            p0, p1 = self.profile_line[0], self.profile_line[1]
            lines.append(f"Profile:  ({p0['row']:.0f}, {p0['col']:.0f}) → ({p1['row']:.0f}, {p1['col']:.0f})")
        print("\n".join(lines))

    def _compute_all_stats(self):
        """Compute statistics for all images."""
        means, mins, maxs, stds = [], [], [], []
        for i in range(self.n_images):
            img = self._data[i]
            means.append(float(np.mean(img)))
            mins.append(float(np.min(img)))
            maxs.append(float(np.max(img)))
            stds.append(float(np.std(img)))
        self.stats_mean = means
        self.stats_min = mins
        self.stats_max = maxs
        self.stats_std = stds

    def _update_all_frames(self):
        """Send raw float32 data to JS (normalization happens in JS for speed)."""
        self.frame_bytes = self._data.tobytes()

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


bind_tool_runtime_api(Show2D, "Show2D")
