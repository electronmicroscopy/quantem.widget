"""
showcomplex: Interactive complex-valued image viewer.

For displaying complex data from ptychography, holography, and exit wave
reconstruction. Supports amplitude, phase, HSV, real, and imaginary display modes.
"""

import json
import pathlib
from typing import List, Optional

import anywidget
import numpy as np
import traitlets

from quantem.widget.array_utils import to_numpy
from quantem.widget.json_state import resolve_widget_version, save_state_file, unwrap_state_payload
from quantem.widget.tool_parity import (
    bind_tool_runtime_api,
    build_tool_groups,
    normalize_tool_groups,
)


class ShowComplex2D(anywidget.AnyWidget):
    """
    Interactive viewer for complex-valued 2D data.

    Display complex images from ptychography, holography, or exit wave
    reconstruction with five visualization modes: amplitude, phase, HSV
    (hue=phase, brightness=amplitude), real part, and imaginary part.

    Parameters
    ----------
    data : array_like (complex) or tuple of (real, imag)
        Complex 2D array of shape (height, width) with dtype complex64 or
        complex128. Also accepts a tuple ``(real, imag)`` of two real arrays.
    display_mode : str, default "amplitude"
        Initial display mode: ``"amplitude"``, ``"phase"``, ``"hsv"``,
        ``"real"``, or ``"imag"``.
    title : str, optional
        Title displayed in the widget header.
    cmap : str, default "inferno"
        Colormap for amplitude/real/imag modes. Phase and HSV modes use
        a fixed cyclic colormap.
    pixel_size : float, default 0.0
        Pixel size in angstroms for scale bar display.
    log_scale : bool, default False
        Apply log(1+x) to amplitude before display.
    auto_contrast : bool, default False
        Use percentile-based contrast.
    show_fft : bool, default False
        Show FFT panel.
    show_stats : bool, default True
        Show statistics bar.
    show_controls : bool, default True
        Show control panel.

    Examples
    --------
    >>> import numpy as np
    >>> from quantem.widget import ShowComplex2D
    >>>
    >>> # Complex exit wave
    >>> data = np.exp(1j * phase) * amplitude
    >>> ShowComplex2D(data, title="Exit Wave", display_mode="hsv")
    >>>
    >>> # From real and imaginary parts
    >>> ShowComplex2D((real_part, imag_part), display_mode="phase")
    """

    _esm = pathlib.Path(__file__).parent / "static" / "showcomplex.js"
    _css = pathlib.Path(__file__).parent / "static" / "showcomplex.css"

    # Core state
    height = traitlets.Int(1).tag(sync=True)
    width = traitlets.Int(1).tag(sync=True)
    real_bytes = traitlets.Bytes(b"").tag(sync=True)
    imag_bytes = traitlets.Bytes(b"").tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)

    # Display mode
    display_mode = traitlets.Unicode("amplitude").tag(sync=True)
    cmap = traitlets.Unicode("inferno").tag(sync=True)

    # Display options
    log_scale = traitlets.Bool(False).tag(sync=True)
    auto_contrast = traitlets.Bool(False).tag(sync=True)
    percentile_low = traitlets.Float(1.0).tag(sync=True)
    percentile_high = traitlets.Float(99.0).tag(sync=True)

    # Scale bar
    pixel_size = traitlets.Float(0.0).tag(sync=True)
    scale_bar_visible = traitlets.Bool(True).tag(sync=True)

    # UI
    show_stats = traitlets.Bool(True).tag(sync=True)
    show_fft = traitlets.Bool(False).tag(sync=True)
    show_controls = traitlets.Bool(True).tag(sync=True)
    image_width_px = traitlets.Int(0).tag(sync=True)
    disabled_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)
    hidden_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)

    # ROI (single-mode, same pattern as Show4D)
    roi_mode = traitlets.Unicode("off").tag(sync=True)  # "off", "circle", "square", "rect"
    roi_center_row = traitlets.Float(0.0).tag(sync=True)
    roi_center_col = traitlets.Float(0.0).tag(sync=True)
    roi_center = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0]).tag(sync=True)
    roi_radius = traitlets.Float(5.0).tag(sync=True)
    roi_width = traitlets.Float(10.0).tag(sync=True)
    roi_height = traitlets.Float(10.0).tag(sync=True)

    # Statistics (recomputed per display_mode)
    stats_mean = traitlets.Float(0.0).tag(sync=True)
    stats_min = traitlets.Float(0.0).tag(sync=True)
    stats_max = traitlets.Float(0.0).tag(sync=True)
    stats_std = traitlets.Float(0.0).tag(sync=True)

    @classmethod
    def _normalize_tool_groups(cls, tool_groups) -> List[str]:
        return normalize_tool_groups("ShowComplex2D", tool_groups)

    @classmethod
    def _build_disabled_tools(
        cls,
        disabled_tools=None,
        disable_display: bool = False,
        disable_histogram: bool = False,
        disable_fft: bool = False,
        disable_roi: bool = False,
        disable_stats: bool = False,
        disable_export: bool = False,
        disable_view: bool = False,
        disable_all: bool = False,
    ) -> List[str]:
        return build_tool_groups(
            "ShowComplex2D",
            tool_groups=disabled_tools,
            all_flag=disable_all,
            flag_map={
                "display": disable_display,
                "histogram": disable_histogram,
                "fft": disable_fft,
                "roi": disable_roi,
                "stats": disable_stats,
                "export": disable_export,
                "view": disable_view,
            },
        )

    @classmethod
    def _build_hidden_tools(
        cls,
        hidden_tools=None,
        hide_display: bool = False,
        hide_histogram: bool = False,
        hide_fft: bool = False,
        hide_roi: bool = False,
        hide_stats: bool = False,
        hide_export: bool = False,
        hide_view: bool = False,
        hide_all: bool = False,
    ) -> List[str]:
        return build_tool_groups(
            "ShowComplex2D",
            tool_groups=hidden_tools,
            all_flag=hide_all,
            flag_map={
                "display": hide_display,
                "histogram": hide_histogram,
                "fft": hide_fft,
                "roi": hide_roi,
                "stats": hide_stats,
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
        data,
        display_mode: str = "amplitude",
        title: str = "",
        cmap: str = "inferno",
        pixel_size: float = 0.0,
        log_scale: bool = False,
        auto_contrast: bool = False,
        percentile_low: float = 1.0,
        percentile_high: float = 99.0,
        show_fft: bool = False,
        show_stats: bool = True,
        show_controls: bool = True,
        scale_bar_visible: bool = True,
        image_width_px: int = 0,
        disabled_tools: Optional[List[str]] = None,
        disable_display: bool = False,
        disable_histogram: bool = False,
        disable_fft: bool = False,
        disable_roi: bool = False,
        disable_stats: bool = False,
        disable_export: bool = False,
        disable_view: bool = False,
        disable_all: bool = False,
        hidden_tools: Optional[List[str]] = None,
        hide_display: bool = False,
        hide_histogram: bool = False,
        hide_fft: bool = False,
        hide_roi: bool = False,
        hide_stats: bool = False,
        hide_export: bool = False,
        hide_view: bool = False,
        hide_all: bool = False,
        state=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.widget_version = resolve_widget_version()

        # Dataset duck typing
        _extracted_title = None
        _extracted_pixel_size = None
        if hasattr(data, "array") and hasattr(data, "name") and hasattr(data, "sampling"):
            _extracted_title = data.name if data.name else None
            if hasattr(data, "units"):
                units = list(data.units)
                sampling_val = float(data.sampling[-1])
                if units[-1] in ("nm",):
                    _extracted_pixel_size = sampling_val * 10  # nm → Å
                elif units[-1] in ("Å", "angstrom", "A"):
                    _extracted_pixel_size = sampling_val
            data = data.array

        # Handle (real, imag) tuple input
        if isinstance(data, tuple) and len(data) == 2:
            real_arr = to_numpy(data[0]).astype(np.float32)
            imag_arr = to_numpy(data[1]).astype(np.float32)
            if real_arr.shape != imag_arr.shape:
                raise ValueError(
                    f"Real and imaginary parts must have same shape, "
                    f"got {real_arr.shape} and {imag_arr.shape}"
                )
            if real_arr.ndim != 2:
                raise ValueError(f"Expected 2D arrays, got {real_arr.ndim}D")
            self._real = real_arr
            self._imag = imag_arr
        else:
            arr = to_numpy(data)
            if not np.issubdtype(arr.dtype, np.complexfloating):
                raise ValueError(
                    f"Expected complex array (complex64/complex128), got {arr.dtype}. "
                    f"Use ShowComplex2D((real, imag)) for real-valued input."
                )
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D array, got {arr.ndim}D")
            self._real = arr.real.astype(np.float32)
            self._imag = arr.imag.astype(np.float32)

        self.height = int(self._real.shape[0])
        self.width = int(self._real.shape[1])

        # Options
        self.display_mode = display_mode
        self.title = title if title else (_extracted_title or "")
        self.cmap = cmap
        if pixel_size == 0.0 and _extracted_pixel_size is not None:
            pixel_size = _extracted_pixel_size
        self.pixel_size = pixel_size
        self.log_scale = log_scale
        self.auto_contrast = auto_contrast
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.show_fft = show_fft
        self.show_stats = show_stats
        self.show_controls = show_controls
        self.scale_bar_visible = scale_bar_visible
        self.image_width_px = image_width_px
        self.disabled_tools = self._build_disabled_tools(
            disabled_tools=disabled_tools,
            disable_display=disable_display,
            disable_histogram=disable_histogram,
            disable_fft=disable_fft,
            disable_roi=disable_roi,
            disable_stats=disable_stats,
            disable_export=disable_export,
            disable_view=disable_view,
            disable_all=disable_all,
        )
        self.hidden_tools = self._build_hidden_tools(
            hidden_tools=hidden_tools,
            hide_display=hide_display,
            hide_histogram=hide_histogram,
            hide_fft=hide_fft,
            hide_roi=hide_roi,
            hide_stats=hide_stats,
            hide_export=hide_export,
            hide_view=hide_view,
            hide_all=hide_all,
        )

        # ROI defaults (centered, radius proportional to image size)
        default_roi_size = max(3, min(self.height, self.width) // 6)
        self.roi_center_row = float(self.height / 2)
        self.roi_center_col = float(self.width / 2)
        self.roi_center = [float(self.height / 2), float(self.width / 2)]
        self.roi_radius = float(default_roi_size)
        self.roi_width = float(default_roi_size * 2)
        self.roi_height = float(default_roi_size)

        # Compute stats for initial display mode
        self._update_stats()

        # Send data to JS
        self.real_bytes = self._real.tobytes()
        self.imag_bytes = self._imag.tobytes()

        # Observers
        self.observe(self._on_display_mode_change, names=["display_mode"])
        self.observe(self._on_roi_center_change, names=["roi_center"])

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

    def _get_display_data(self, mode: str | None = None) -> np.ndarray:
        mode = mode or self.display_mode
        if mode == "amplitude":
            return np.sqrt(self._real ** 2 + self._imag ** 2)
        elif mode == "phase":
            return np.arctan2(self._imag, self._real)
        elif mode == "real":
            return self._real
        elif mode == "imag":
            return self._imag
        elif mode == "hsv":
            return np.sqrt(self._real ** 2 + self._imag ** 2)
        else:
            raise ValueError(f"Unknown display mode: {mode!r}")

    def _update_stats(self):
        data = self._get_display_data()
        self.stats_mean = float(data.mean())
        self.stats_min = float(data.min())
        self.stats_max = float(data.max())
        self.stats_std = float(data.std())

    def _on_display_mode_change(self, change=None):
        self._update_stats()

    def _on_roi_center_change(self, change=None):
        val = self.roi_center
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            self.roi_center_row = float(val[0])
            self.roi_center_col = float(val[1])

    def roi_circle(self, row=None, col=None, radius=None) -> "ShowComplex2D":
        if row is not None:
            self.roi_center_row = float(row)
        if col is not None:
            self.roi_center_col = float(col)
        if radius is not None:
            self.roi_radius = float(radius)
        self.roi_mode = "circle"
        return self

    def roi_square(self, row=None, col=None, radius=None) -> "ShowComplex2D":
        if row is not None:
            self.roi_center_row = float(row)
        if col is not None:
            self.roi_center_col = float(col)
        if radius is not None:
            self.roi_radius = float(radius)
        self.roi_mode = "square"
        return self

    def roi_rect(self, row=None, col=None, width=None, height=None) -> "ShowComplex2D":
        if row is not None:
            self.roi_center_row = float(row)
        if col is not None:
            self.roi_center_col = float(col)
        if width is not None:
            self.roi_width = float(width)
        if height is not None:
            self.roi_height = float(height)
        self.roi_mode = "rect"
        return self

    def set_image(self, data):
        """Replace the complex data. Preserves all display settings."""
        if hasattr(data, "array") and hasattr(data, "name") and hasattr(data, "sampling"):
            data = data.array
        if isinstance(data, tuple) and len(data) == 2:
            real_arr = to_numpy(data[0]).astype(np.float32)
            imag_arr = to_numpy(data[1]).astype(np.float32)
            if real_arr.shape != imag_arr.shape:
                raise ValueError(
                    f"Real and imaginary parts must have same shape, "
                    f"got {real_arr.shape} and {imag_arr.shape}"
                )
            if real_arr.ndim != 2:
                raise ValueError(f"Expected 2D arrays, got {real_arr.ndim}D")
            self._real = real_arr
            self._imag = imag_arr
        else:
            arr = to_numpy(data)
            if not np.issubdtype(arr.dtype, np.complexfloating):
                raise ValueError(
                    f"Expected complex array (complex64/complex128), got {arr.dtype}."
                )
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D array, got {arr.ndim}D")
            self._real = arr.real.astype(np.float32)
            self._imag = arr.imag.astype(np.float32)
        self.height = int(self._real.shape[0])
        self.width = int(self._real.shape[1])
        self._update_stats()
        self.real_bytes = self._real.tobytes()
        self.imag_bytes = self._imag.tobytes()

    # =========================================================================
    # Export
    # =========================================================================

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
            return np.clip((frame - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
        return np.zeros(frame.shape, dtype=np.uint8)

    def save_image(
        self,
        path: str | pathlib.Path,
        *,
        display_mode: str | None = None,
        format: str | None = None,
        dpi: int = 150,
    ) -> pathlib.Path:
        """Save current view as PNG, PDF, or TIFF.

        Parameters
        ----------
        path : str or pathlib.Path
            Output file path.
        display_mode : str, optional
            Override display mode. One of 'amplitude', 'phase', 'hsv',
            'real', 'imag'. Defaults to current display_mode.
        format : str, optional
            'png', 'pdf', or 'tiff'. If omitted, inferred from file extension.
        dpi : int, default 150
            Output DPI metadata.

        Returns
        -------
        pathlib.Path
            The written file path.
        """
        import matplotlib.colors as mcolors

        from matplotlib import colormaps
        from PIL import Image

        path = pathlib.Path(path)
        fmt = (format or path.suffix.lstrip(".").lower() or "png").lower()
        if fmt not in ("png", "pdf", "tiff", "tif"):
            raise ValueError(f"Unsupported format: {fmt!r}. Use 'png', 'pdf', or 'tiff'.")

        mode = display_mode or self.display_mode
        valid_modes = ("amplitude", "phase", "hsv", "real", "imag")
        if mode not in valid_modes:
            raise ValueError(f"Unknown display_mode: {mode!r}. Use one of {valid_modes}.")

        if mode == "hsv":
            amp = np.sqrt(self._real ** 2 + self._imag ** 2)
            phase = np.arctan2(self._imag, self._real)

            amp_min, amp_max = float(amp.min()), float(amp.max())
            if amp_max > amp_min:
                amp_norm = (amp - amp_min) / (amp_max - amp_min)
            else:
                amp_norm = np.zeros_like(amp)

            hue = (phase + np.pi) / (2 * np.pi)
            hsv_array = np.stack([hue, np.ones_like(hue), amp_norm], axis=-1)
            rgb = mcolors.hsv_to_rgb(hsv_array)
            rgba = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)
            rgba[:, :, :3] = (rgb * 255).astype(np.uint8)
            rgba[:, :, 3] = 255
            img = Image.fromarray(rgba)
        else:
            data = self._get_display_data(mode)

            if self.log_scale and mode in ("amplitude", "real", "imag"):
                data = np.log1p(np.maximum(data, 0))

            if self.auto_contrast:
                vmin = float(np.percentile(data, self.percentile_low))
                vmax = float(np.percentile(data, self.percentile_high))
            else:
                vmin = float(data.min())
                vmax = float(data.max())

            if vmax > vmin:
                normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
            else:
                normalized = np.zeros_like(data)

            cmap_name = "hsv" if mode == "phase" else self.cmap
            cmap_fn = colormaps.get_cmap(cmap_name)
            rgba = (cmap_fn(normalized) * 255).astype(np.uint8)
            img = Image.fromarray(rgba)

        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path), dpi=(dpi, dpi))
        return path

    # =========================================================================
    # State Protocol
    # =========================================================================

    def state_dict(self):
        return {
            "display_mode": self.display_mode,
            "title": self.title,
            "cmap": self.cmap,
            "log_scale": self.log_scale,
            "auto_contrast": self.auto_contrast,
            "percentile_low": self.percentile_low,
            "percentile_high": self.percentile_high,
            "pixel_size": self.pixel_size,
            "scale_bar_visible": self.scale_bar_visible,
            "show_fft": self.show_fft,
            "show_stats": self.show_stats,
            "show_controls": self.show_controls,
            "image_width_px": self.image_width_px,
            "roi_mode": self.roi_mode,
            "roi_center_row": self.roi_center_row,
            "roi_center_col": self.roi_center_col,
            "roi_radius": self.roi_radius,
            "roi_width": self.roi_width,
            "roi_height": self.roi_height,
            "disabled_tools": self.disabled_tools,
            "hidden_tools": self.hidden_tools,
        }

    def save(self, path: str):
        """Save widget state to a JSON file."""
        save_state_file(path, "ShowComplex2D", self.state_dict())

    def load_state_dict(self, state):
        """Restore widget state from a dict."""
        if "pixel_size_angstrom" in state and "pixel_size" not in state:
            state = dict(state, pixel_size=state.pop("pixel_size_angstrom"))
        for key, val in state.items():
            if hasattr(self, key):
                setattr(self, key, val)

    def summary(self):
        """Print a human-readable summary of the widget state."""
        name = self.title if self.title else "ShowComplex2D"
        lines = [name, "═" * 32]
        lines.append(f"Image:    {self.height}×{self.width} (complex)")
        if self.pixel_size > 0:
            ps = self.pixel_size
            if ps >= 10:
                lines[-1] += f" ({ps / 10:.2f} nm/px)"
            else:
                lines[-1] += f" ({ps:.2f} Å/px)"
        amp = np.sqrt(self._real ** 2 + self._imag ** 2)
        lines.append(
            f"Amp:      min={float(amp.min()):.4g}  max={float(amp.max()):.4g}  "
            f"mean={float(amp.mean()):.4g}"
        )
        phase = np.arctan2(self._imag, self._real)
        lines.append(
            f"Phase:    min={float(phase.min()):.4g}  max={float(phase.max()):.4g}  "
            f"mean={float(phase.mean()):.4g}"
        )
        mode = self.display_mode
        cmap = self.cmap if mode in ("amplitude", "real", "imag") else "hsv (cyclic)"
        scale = "log" if self.log_scale else "linear"
        contrast = "auto" if self.auto_contrast else "manual"
        lines.append(f"Display:  {mode} | {cmap} | {contrast} | {scale}")
        if self.show_fft:
            lines[-1] += " | FFT"
        print("\n".join(lines))

    def __repr__(self) -> str:
        name = self.title if self.title else "ShowComplex2D"
        parts = [f"{name}({self.height}×{self.width}"]
        parts.append(f"mode={self.display_mode}")
        if self.pixel_size > 0:
            ps = self.pixel_size
            if ps >= 10:
                parts.append(f"px={ps / 10:.2f} nm")
            else:
                parts.append(f"px={ps:.2f} Å")
        if self.log_scale:
            parts.append("log")
        if self.show_fft:
            parts.append("fft")
        return ", ".join(parts) + ")"


bind_tool_runtime_api(ShowComplex2D, "ShowComplex2D")
