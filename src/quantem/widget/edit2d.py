"""
edit2d: Interactive crop/pad/mask tool for 2D images.

Visually define a rectangular output region on a 2D image.
Region inside image bounds crops; region outside pads.
Mask mode allows painting a binary mask on the image.
"""

import json
import pathlib
from typing import Optional, Union, List, Tuple

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


class Edit2D(anywidget.AnyWidget):
    """
    Interactive visual crop/pad tool for 2D images.

    Display a 2D image with a draggable crop rectangle. The rectangle
    can be positioned anywhere -- inside the image for cropping, extending
    beyond image bounds for padding, or fully enclosing the image for
    pure padding.

    Parameters
    ----------
    data : array_like
        2D array (height, width) for a single image, or
        3D array (N, height, width) or list of 2D arrays for multi-image mode.
        All images are cropped with the same region.
    bounds : tuple of int, optional
        Initial crop bounds as (top, left, bottom, right) in image pixel
        coordinates. Negative values and values exceeding image dimensions
        are allowed for padding. If None, defaults to the full image extent.
    fill_value : float, default 0.0
        Fill value for padded regions outside the original image bounds.
    title : str, default ""
        Title displayed in the widget header.
    cmap : str, default "gray"
        Colormap name.
    pixel_size : float, default 0.0
        Pixel size in angstroms for scale bar display.
    show_stats : bool, default True
        Show statistics bar.
    show_controls : bool, default True
        Show control row.
    show_display_controls : bool, default True
        Show display control group.
    show_edit_controls : bool, default True
        Show edit control group.
    show_histogram : bool, default True
        Show histogram control group.
    log_scale : bool, default False
        Log intensity mapping.
    auto_contrast : bool, default True
        Percentile-based contrast.
    disabled_tools : list of str, optional
        Tool groups to disable in the frontend UI/interaction layer.
        Supported values: ``"mode"``, ``"edit"``, ``"display"``,
        ``"histogram"``, ``"stats"``, ``"navigation"``, ``"export"``,
        ``"view"``, ``"all"``.
    disable_* : bool, optional
        Convenience flags (``disable_mode``, ``disable_edit``,
        ``disable_display``, ``disable_histogram``, ``disable_stats``,
        ``disable_navigation``, ``disable_export``, ``disable_view``,
        ``disable_all``) equivalent to including those tool names in
        ``disabled_tools``.
    hidden_tools : list of str, optional
        Tool groups to hide from the frontend UI. Hidden tools are also
        interaction-locked (equivalent to disabled for behavior).
    hide_* : bool, optional
        Convenience flags (``hide_mode``, ``hide_edit``,
        ``hide_display``, ``hide_histogram``, ``hide_stats``,
        ``hide_navigation``, ``hide_export``, ``hide_view``,
        ``hide_all``) equivalent to including those tool names in
        ``hidden_tools``.

    Examples
    --------
    >>> import numpy as np
    >>> from quantem.widget import Edit2D
    >>> img = np.random.rand(256, 256).astype(np.float32)
    >>> crop = Edit2D(img)
    >>> crop  # display, draw crop region interactively
    >>> crop.result  # returns cropped NumPy array
    >>> crop.crop_bounds  # (top, left, bottom, right) tuple
    """

    _esm = pathlib.Path(__file__).parent / "static" / "edit2d.js"
    _css = pathlib.Path(__file__).parent / "static" / "edit2d.css"

    # =========================================================================
    # Core State
    # =========================================================================
    n_images = traitlets.Int(1).tag(sync=True)
    height = traitlets.Int(1).tag(sync=True)
    width = traitlets.Int(1).tag(sync=True)
    frame_bytes = traitlets.Bytes(b"").tag(sync=True)
    labels = traitlets.List(traitlets.Unicode()).tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    cmap = traitlets.Unicode("gray").tag(sync=True)

    # =========================================================================
    # Crop Region (synced bidirectionally with JS)
    # =========================================================================
    crop_top = traitlets.Int(0).tag(sync=True)
    crop_left = traitlets.Int(0).tag(sync=True)
    crop_bottom = traitlets.Int(0).tag(sync=True)
    crop_right = traitlets.Int(0).tag(sync=True)
    fill_value = traitlets.Float(0.0).tag(sync=True)

    # =========================================================================
    # Display Options
    # =========================================================================
    log_scale = traitlets.Bool(False).tag(sync=True)
    auto_contrast = traitlets.Bool(True).tag(sync=True)

    # =========================================================================
    # Scale Bar
    # =========================================================================
    pixel_size = traitlets.Float(0.0).tag(sync=True)

    # =========================================================================
    # UI Visibility
    # =========================================================================
    show_controls = traitlets.Bool(True).tag(sync=True)
    show_stats = traitlets.Bool(True).tag(sync=True)
    show_display_controls = traitlets.Bool(True).tag(sync=True)
    show_edit_controls = traitlets.Bool(True).tag(sync=True)
    show_histogram = traitlets.Bool(True).tag(sync=True)
    disabled_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)
    hidden_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)
    stats_mean = traitlets.Float(0.0).tag(sync=True)
    stats_min = traitlets.Float(0.0).tag(sync=True)
    stats_max = traitlets.Float(0.0).tag(sync=True)
    stats_std = traitlets.Float(0.0).tag(sync=True)

    # =========================================================================
    # Mode: "crop" or "mask"
    # =========================================================================
    mode = traitlets.Unicode("crop").tag(sync=True)

    # =========================================================================
    # Mask State
    # =========================================================================
    mask_bytes = traitlets.Bytes(b"").tag(sync=True)
    mask_tool = traitlets.Unicode("rectangle").tag(sync=True)
    mask_action = traitlets.Unicode("add").tag(sync=True)

    # =========================================================================
    # Gallery (multi-image)
    # =========================================================================
    selected_idx = traitlets.Int(0).tag(sync=True)

    # =========================================================================
    # Shared / Independent editing
    # =========================================================================
    shared = traitlets.Bool(True).tag(sync=True)
    per_image_crops_json = traitlets.Unicode("[]").tag(sync=True)
    per_image_masks_bytes = traitlets.Bytes(b"").tag(sync=True)

    @classmethod
    def _normalize_tool_groups(cls, tool_groups) -> List[str]:
        """Validate and normalize tool group values with stable ordering."""
        return normalize_tool_groups("Edit2D", tool_groups)

    @classmethod
    def _build_disabled_tools(
        cls,
        disabled_tools=None,
        disable_mode: bool = False,
        disable_edit: bool = False,
        disable_display: bool = False,
        disable_histogram: bool = False,
        disable_stats: bool = False,
        disable_navigation: bool = False,
        disable_export: bool = False,
        disable_view: bool = False,
        disable_all: bool = False,
    ) -> List[str]:
        """Build disabled_tools from explicit list and ergonomic boolean flags."""
        return build_tool_groups(
            "Edit2D",
            tool_groups=disabled_tools,
            all_flag=disable_all,
            flag_map={
                "mode": disable_mode,
                "edit": disable_edit,
                "display": disable_display,
                "histogram": disable_histogram,
                "stats": disable_stats,
                "navigation": disable_navigation,
                "export": disable_export,
                "view": disable_view,
            },
        )

    @classmethod
    def _build_hidden_tools(
        cls,
        hidden_tools=None,
        hide_mode: bool = False,
        hide_edit: bool = False,
        hide_display: bool = False,
        hide_histogram: bool = False,
        hide_stats: bool = False,
        hide_navigation: bool = False,
        hide_export: bool = False,
        hide_view: bool = False,
        hide_all: bool = False,
    ) -> List[str]:
        """Build hidden_tools from explicit list and ergonomic boolean flags."""
        return build_tool_groups(
            "Edit2D",
            tool_groups=hidden_tools,
            all_flag=hide_all,
            flag_map={
                "mode": hide_mode,
                "edit": hide_edit,
                "display": hide_display,
                "histogram": hide_histogram,
                "stats": hide_stats,
                "navigation": hide_navigation,
                "export": hide_export,
                "view": hide_view,
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
        bounds: Optional[Tuple[int, int, int, int]] = None,
        fill_value: float = 0.0,
        mode: str = "crop",
        shared: bool = True,
        labels: Optional[List[str]] = None,
        title: str = "",
        cmap: str = "gray",
        pixel_size: float = 0.0,
        show_controls: bool = True,
        show_stats: bool = True,
        show_display_controls: bool = True,
        show_edit_controls: bool = True,
        show_histogram: bool = True,
        log_scale: bool = False,
        auto_contrast: bool = True,
        disabled_tools: Optional[List[str]] = None,
        disable_mode: bool = False,
        disable_edit: bool = False,
        disable_display: bool = False,
        disable_histogram: bool = False,
        disable_stats: bool = False,
        disable_navigation: bool = False,
        disable_export: bool = False,
        disable_view: bool = False,
        disable_all: bool = False,
        hidden_tools: Optional[List[str]] = None,
        hide_mode: bool = False,
        hide_edit: bool = False,
        hide_display: bool = False,
        hide_histogram: bool = False,
        hide_stats: bool = False,
        hide_navigation: bool = False,
        hide_export: bool = False,
        hide_view: bool = False,
        hide_all: bool = False,
        state=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.widget_version = resolve_widget_version()
        self.mode = mode

        # Check if data is a Dataset2d and extract metadata
        if hasattr(data, "array") and hasattr(data, "name") and hasattr(data, "sampling"):
            if not title and data.name:
                title = data.name
            if pixel_size == 0.0 and hasattr(data, "units"):
                units = list(data.units)
                sampling_val = float(data.sampling[-1])
                if units[-1] in ("nm",):
                    pixel_size = sampling_val * 10  # nm -> angstrom
                elif units[-1] in ("\u00c5", "angstrom", "A"):
                    pixel_size = sampling_val
            data = data.array

        # Convert input to NumPy (handles NumPy, CuPy, PyTorch)
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

        # Labels
        if labels is None:
            if self.n_images == 1:
                self.labels = ["Image"]
            else:
                self.labels = [f"Image {i+1}" for i in range(self.n_images)]
        else:
            self.labels = list(labels)

        # Options
        self.title = title
        self.cmap = cmap
        self.pixel_size = pixel_size
        self.show_controls = show_controls
        self.show_stats = show_stats
        self.show_display_controls = show_display_controls
        self.show_edit_controls = show_edit_controls
        self.show_histogram = show_histogram
        self.log_scale = log_scale
        self.auto_contrast = auto_contrast
        self.disabled_tools = self._build_disabled_tools(
            disabled_tools=disabled_tools,
            disable_mode=disable_mode,
            disable_edit=disable_edit,
            disable_display=disable_display,
            disable_histogram=disable_histogram,
            disable_stats=disable_stats,
            disable_navigation=disable_navigation,
            disable_export=disable_export,
            disable_view=disable_view,
            disable_all=disable_all,
        )
        self.hidden_tools = self._build_hidden_tools(
            hidden_tools=hidden_tools,
            hide_mode=hide_mode,
            hide_edit=hide_edit,
            hide_display=hide_display,
            hide_histogram=hide_histogram,
            hide_stats=hide_stats,
            hide_navigation=hide_navigation,
            hide_export=hide_export,
            hide_view=hide_view,
            hide_all=hide_all,
        )
        self.fill_value = fill_value

        # Crop bounds
        if bounds is not None:
            self.crop_top, self.crop_left, self.crop_bottom, self.crop_right = bounds
        else:
            self.crop_top = 0
            self.crop_left = 0
            self.crop_bottom = self.height
            self.crop_right = self.width

        self.shared = shared
        if not self.shared and self.n_images > 1:
            crop = {"top": self.crop_top, "left": self.crop_left,
                    "bottom": self.crop_bottom, "right": self.crop_right}
            self.per_image_crops_json = json.dumps([crop] * self.n_images)

        # Compute stats for current image
        self._compute_stats()

        # Send raw float32 data to JS
        self.frame_bytes = self._data.tobytes()

        self.selected_idx = 0

        # State restoration (must be last)
        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(
                    json.loads(pathlib.Path(state).read_text()),
                    require_envelope=True,
                )
            else:
                state = unwrap_state_payload(state)
            self.load_state_dict(state)

    def _compute_stats(self):
        img = self._data[self.selected_idx]
        self.stats_mean = float(np.mean(img))
        self.stats_min = float(np.min(img))
        self.stats_max = float(np.max(img))
        self.stats_std = float(np.std(img))

    def _crop_single_with_bounds(self, img, top, left, bottom, right):
        h, w = img.shape
        out_h = bottom - top
        out_w = right - left
        if out_h <= 0 or out_w <= 0:
            return np.empty((0, 0), dtype=img.dtype)
        result = np.full((out_h, out_w), self.fill_value, dtype=img.dtype)
        src_top = max(0, top)
        src_left = max(0, left)
        src_bottom = min(h, bottom)
        src_right = min(w, right)
        if src_top >= src_bottom or src_left >= src_right:
            return result
        dst_top = src_top - top
        dst_left = src_left - left
        result[dst_top:dst_top + (src_bottom - src_top),
               dst_left:dst_left + (src_right - src_left)] = \
            img[src_top:src_bottom, src_left:src_right]
        return result

    def _crop_single(self, img: np.ndarray) -> np.ndarray:
        return self._crop_single_with_bounds(
            img, self.crop_top, self.crop_left, self.crop_bottom, self.crop_right
        )

    def _apply_mask(self, img: np.ndarray, m: np.ndarray) -> np.ndarray:
        out = img.copy()
        out[m] = self.fill_value
        return out

    def _get_per_image_crops(self):
        default = {"top": self.crop_top, "left": self.crop_left,
                    "bottom": self.crop_bottom, "right": self.crop_right}
        if not self.per_image_crops_json or self.per_image_crops_json == "[]":
            return [{**default} for _ in range(self.n_images)]
        crops = json.loads(self.per_image_crops_json)
        while len(crops) < self.n_images:
            crops.append({**default})
        return crops[:self.n_images]

    def _get_per_image_masks(self):
        size = self.height * self.width
        total = self.n_images * size
        if not self.per_image_masks_bytes or len(self.per_image_masks_bytes) != total:
            return [np.zeros((self.height, self.width), dtype=bool) for _ in range(self.n_images)]
        all_masks = np.frombuffer(self.per_image_masks_bytes, dtype=np.uint8).reshape(
            self.n_images, self.height, self.width
        )
        return [all_masks[i] > 0 for i in range(self.n_images)]

    @property
    def mask(self) -> np.ndarray:
        """Current mask as a boolean array (H, W). True = masked."""
        if not self.shared and self.n_images > 1:
            masks = self._get_per_image_masks()
            idx = min(self.selected_idx, self.n_images - 1)
            return masks[idx]
        if not self.mask_bytes:
            return np.zeros((self.height, self.width), dtype=bool)
        arr = np.frombuffer(self.mask_bytes, dtype=np.uint8).reshape(
            self.height, self.width
        )
        return arr > 0

    @property
    def result(self) -> Union[np.ndarray, List[np.ndarray]]:
        """Return result based on current mode.

        Crop mode: cropped/padded image(s).
        Mask mode: image(s) with masked pixels set to fill_value.
        In independent mode (shared=False), each image gets its own crop/mask.
        """
        if self.shared or self.n_images == 1:
            if self.mode == "mask":
                m = self.mask
                if self.n_images == 1:
                    return self._apply_mask(self._data[0], m)
                return [self._apply_mask(self._data[i], m) for i in range(self.n_images)]
            if self.n_images == 1:
                return self._crop_single(self._data[0])
            return [self._crop_single(self._data[i]) for i in range(self.n_images)]

        # Independent mode
        if self.mode == "mask":
            masks = self._get_per_image_masks()
            return [self._apply_mask(self._data[i], masks[i]) for i in range(self.n_images)]
        crops = self._get_per_image_crops()
        return [
            self._crop_single_with_bounds(
                self._data[i], c["top"], c["left"], c["bottom"], c["right"]
            )
            for i, c in enumerate(crops)
        ]

    @property
    def crop_bounds(self) -> Tuple[int, int, int, int]:
        """Current crop bounds as (top, left, bottom, right).

        In independent mode, returns the current image's bounds.
        """
        if not self.shared and self.n_images > 1:
            crops = self._get_per_image_crops()
            idx = min(self.selected_idx, self.n_images - 1)
            c = crops[idx]
            return (c["top"], c["left"], c["bottom"], c["right"])
        return (self.crop_top, self.crop_left, self.crop_bottom, self.crop_right)

    @crop_bounds.setter
    def crop_bounds(self, bounds: Tuple[int, int, int, int]):
        top, left, bottom, right = bounds
        if not self.shared and self.n_images > 1:
            crops = self._get_per_image_crops()
            idx = min(self.selected_idx, self.n_images - 1)
            crops[idx] = {"top": top, "left": left, "bottom": bottom, "right": right}
            self.per_image_crops_json = json.dumps(crops)
        else:
            self.crop_top, self.crop_left, self.crop_bottom, self.crop_right = bounds

    @property
    def crop_size(self) -> Tuple[int, int]:
        """Output size as (height, width)."""
        top, left, bottom, right = self.crop_bounds
        return (bottom - top, right - left)

    def set_image(self, data, **kwargs):
        """Replace the image data."""
        if hasattr(data, "array") and hasattr(data, "name") and hasattr(data, "sampling"):
            if "title" not in kwargs and data.name:
                self.title = data.name
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
        self.crop_top = 0
        self.crop_left = 0
        self.crop_bottom = self.height
        self.crop_right = self.width
        self.mask_bytes = b""
        self.per_image_crops_json = "[]"
        self.per_image_masks_bytes = b""
        self._compute_stats()
        self.frame_bytes = self._data.tobytes()

    # =========================================================================
    # Export
    # =========================================================================

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
        format: str | None = None,
        dpi: int = 150,
    ) -> pathlib.Path:
        """Save current image as PNG, PDF, or TIFF.

        In crop mode, saves the cropped/padded result.
        In mask mode, saves the masked result.

        Parameters
        ----------
        path : str or pathlib.Path
            Output file path.
        format : str, optional
            'png', 'pdf', or 'tiff'. If omitted, inferred from file extension.
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

        result = self.result
        if isinstance(result, list):
            result = result[self.selected_idx]

        normalized = self._normalize_frame(result)
        cmap_fn = colormaps.get_cmap(self.cmap)
        rgba = (cmap_fn(normalized / 255.0) * 255).astype(np.uint8)

        img = Image.fromarray(rgba)
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path), dpi=(dpi, dpi))
        return path

    # =========================================================================
    # State Protocol
    # =========================================================================

    def state_dict(self):
        sd = {
            "title": self.title,
            "cmap": self.cmap,
            "mode": self.mode,
            "log_scale": self.log_scale,
            "auto_contrast": self.auto_contrast,
            "show_controls": self.show_controls,
            "show_stats": self.show_stats,
            "show_display_controls": self.show_display_controls,
            "show_edit_controls": self.show_edit_controls,
            "show_histogram": self.show_histogram,
            "disabled_tools": self.disabled_tools,
            "hidden_tools": self.hidden_tools,
            "pixel_size": self.pixel_size,
            "fill_value": self.fill_value,
            "crop_top": self.crop_top,
            "crop_left": self.crop_left,
            "crop_bottom": self.crop_bottom,
            "crop_right": self.crop_right,
            "shared": self.shared,
        }
        if not self.shared and self.n_images > 1:
            sd["per_image_crops"] = self._get_per_image_crops()
        return sd

    def save(self, path: str):
        save_state_file(path, "Edit2D", self.state_dict())

    def load_state_dict(self, state):
        for key, val in state.items():
            if key == "pixel_size_angstrom":
                key = "pixel_size"
            if key == "per_image_crops":
                self.per_image_crops_json = json.dumps(val)
                continue
            if hasattr(self, key):
                setattr(self, key, val)
        # Clear stale per-image state when restoring to shared mode
        if state.get("shared", True) and "per_image_crops" not in state:
            self.per_image_crops_json = "[]"
            self.per_image_masks_bytes = b""

    def summary(self):
        name = self.title if self.title else "Edit2D"
        lines = [name, "═" * 32]
        lines.append(f"Image:    {self.height}×{self.width}")
        if self.n_images > 1:
            link = "shared" if self.shared else "independent"
            lines[-1] += f" ({self.n_images} images, {link})"
        if self.pixel_size > 0:
            ps = self.pixel_size
            if ps >= 10:
                lines[-1] += f" ({ps / 10:.2f} nm/px)"
            else:
                lines[-1] += f" ({ps:.2f} Å/px)"
        lines.append(f"Mode:     {self.mode}")
        if self.mode == "crop":
            crop_h, crop_w = self.crop_size
            top, left, bottom, right = self.crop_bounds
            lines.append(
                f"Crop:     ({top}, {left}) → "
                f"({bottom}, {right})  "
                f"= {crop_h}×{crop_w}"
            )
            lines.append(f"Fill:     {self.fill_value}")
        else:
            mask_px = int(np.sum(self.mask)) if (self.mask_bytes or self.per_image_masks_bytes) else 0
            total = self.height * self.width
            pct = 100 * mask_px / total if total > 0 else 0
            lines.append(f"Mask:     {mask_px} px ({pct:.1f}%)")
        scale = "log" if self.log_scale else "linear"
        contrast = "auto" if self.auto_contrast else "manual"
        lines.append(f"Display:  {self.cmap} | {contrast} | {scale}")
        if self.disabled_tools:
            lines.append(f"Locked:   {', '.join(self.disabled_tools)}")
        if self.hidden_tools:
            lines.append(f"Hidden:   {', '.join(self.hidden_tools)}")
        print("\n".join(lines))

    def __repr__(self):
        independent = not self.shared and self.n_images > 1
        suffix = ", independent" if independent else ""
        imgs = f", {self.n_images} images" if self.n_images > 1 else ""
        if self.mode == "mask":
            mask_px = int(np.sum(self.mask)) if (self.mask_bytes or self.per_image_masks_bytes) else 0
            total = self.height * self.width
            pct = 100 * mask_px / total if total > 0 else 0
            return f"Edit2D({self.height}x{self.width}{imgs}, mask={mask_px}px ({pct:.1f}%){suffix})"
        crop_h, crop_w = self.crop_size
        top, left, _, _ = self.crop_bounds
        return (
            f"Edit2D({self.height}x{self.width}{imgs}, "
            f"crop={crop_h}x{crop_w} at ({top},{left}), "
            f"fill={self.fill_value}{suffix})"
        )


bind_tool_runtime_api(Edit2D, "Edit2D")
