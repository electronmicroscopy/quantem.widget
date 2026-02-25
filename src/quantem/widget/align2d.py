"""
Align2D: Interactive image alignment widget.
Overlay two 2D images with alpha blending and drag/pad to align.
Auto-alignment via FFT cross-correlation with live NCC display.
"""
import json
import pathlib
from typing import Union

import anywidget
import numpy as np
import traitlets

from quantem.widget.array_utils import to_numpy, _resize_image
from quantem.widget.io import IOResult
from quantem.widget.json_state import resolve_widget_version, save_state_file, unwrap_state_payload
from quantem.widget.tool_parity import (
    bind_tool_runtime_api,
    build_tool_groups,
    normalize_tool_groups,
)


def _tukey_2d(h: int, w: int, alpha: float = 0.2) -> np.ndarray:
    """2D Tukey window — flat center, cosine-tapered edges."""
    def _t1d(n: int) -> np.ndarray:
        if n <= 1:
            return np.ones(n)
        x = np.linspace(0, 1, n)
        win = np.ones(n)
        left = x < alpha / 2
        right = x > 1 - alpha / 2
        win[left] = 0.5 * (1 + np.cos(2 * np.pi / alpha * (x[left] - alpha / 2)))
        win[right] = 0.5 * (1 + np.cos(2 * np.pi / alpha * (x[right] - 1 + alpha / 2)))
        return win
    return np.outer(_t1d(h), _t1d(w))


def _dft_upsample(
    fa_conj_fb: np.ndarray,
    peak_y: int,
    peak_x: int,
    upsample_factor: int = 100,
    region: float = 1.5,
) -> tuple[float, float]:
    """Matrix DFT sub-pixel refinement (Guizar-Sicairos et al. 2008).

    Evaluates the inverse DFT at upsampled coordinates in a small region
    around the integer peak. 100x upsampling -> 1/100 pixel accuracy.
    """
    h, w = fa_conj_fb.shape
    size = int(np.ceil(region * upsample_factor))
    ups_y = peak_y + (np.arange(size) - size // 2) / upsample_factor
    ups_x = peak_x + (np.arange(size) - size // 2) / upsample_factor
    # Use proper frequency indices (negative freqs for k >= N/2)
    freq_y = np.fft.fftfreq(h) * h
    freq_x = np.fft.fftfreq(w) * w
    row_kernel = np.exp(2j * np.pi * ups_y[:, None] * freq_y[None, :] / h)
    col_kernel = np.exp(2j * np.pi * ups_x[:, None] * freq_x[None, :] / w)
    upsampled = np.real(row_kernel @ fa_conj_fb @ col_kernel.T)
    up_y, up_x = np.unravel_index(np.argmax(upsampled), upsampled.shape)
    return float(ups_y[up_y]), float(ups_x[up_x])


def _cross_correlate_fft(
    a: np.ndarray,
    b: np.ndarray,
    max_shift_x: int = 0,
    max_shift_y: int = 0,
) -> tuple[float, float]:
    """Phase correlation with Tukey windowing and constrained peak search."""
    h, w = a.shape
    # Tukey window to suppress edge artifacts
    win = _tukey_2d(h, w)
    a_win = (a - a.mean()) * win
    b_win = (b - b.mean()) * win
    fa = np.fft.fft2(a_win)
    fb = np.fft.fft2(b_win)
    # Phase correlation: normalize by magnitude
    cross_power = fa * np.conj(fb)
    cross_power /= np.abs(cross_power) + 1e-10
    xcorr = np.real(np.fft.ifft2(cross_power))
    # Constrain search to valid shift range
    if max_shift_x > 0 or max_shift_y > 0:
        msy = min(max_shift_y, h // 2) if max_shift_y > 0 else h // 2
        msx = min(max_shift_x, w // 2) if max_shift_x > 0 else w // 2
        valid_y = np.zeros(h, dtype=bool)
        valid_y[:msy + 1] = True
        valid_y[max(h - msy, 0):] = True
        valid_x = np.zeros(w, dtype=bool)
        valid_x[:msx + 1] = True
        valid_x[max(w - msx, 0):] = True
        xcorr = np.where(np.outer(valid_y, valid_x), xcorr, -np.inf)
    peak_y, peak_x = np.unravel_index(np.argmax(xcorr), xcorr.shape)
    sub_y, sub_x = _dft_upsample(cross_power, int(peak_y), int(peak_x))
    dy = float(sub_y if sub_y <= h / 2 else sub_y - h)
    dx = float(sub_x if sub_x <= w / 2 else sub_x - w)
    return dx, dy


def _compute_ncc(a: np.ndarray, b: np.ndarray, dx: float, dy: float) -> float:
    """Compute normalized cross-correlation at a specific sub-pixel offset."""
    h, w = a.shape
    idx, idy = int(np.floor(dx)), int(np.floor(dy))
    fx, fy = dx - idx, dy - idy
    # Overlap region (shrunk by 1 for bilinear +1 neighbor)
    y_start = max(0, idy)
    y_end = min(h, h + idy - 1)
    x_start = max(0, idx)
    x_end = min(w, w + idx - 1)
    if y_end <= y_start or x_end <= x_start:
        return 0.0
    a_crop = a[y_start:y_end, x_start:x_end]
    # Bilinear interpolation of b
    by_s, by_e = y_start - idy, y_end - idy
    bx_s, bx_e = x_start - idx, x_end - idx
    b_interp = (b[by_s:by_e, bx_s:bx_e] * (1 - fx) * (1 - fy)
                + b[by_s:by_e, bx_s + 1:bx_e + 1] * fx * (1 - fy)
                + b[by_s + 1:by_e + 1, bx_s:bx_e] * (1 - fx) * fy
                + b[by_s + 1:by_e + 1, bx_s + 1:bx_e + 1] * fx * fy)
    a_c = a_crop - a_crop.mean()
    b_c = b_interp - b_interp.mean()
    denom = np.sqrt(np.sum(a_c ** 2) * np.sum(b_c ** 2))
    if denom == 0:
        return 0.0
    return float(np.sum(a_c * b_c) / denom)


class Align2D(anywidget.AnyWidget):
    """
    Interactive alignment of two 2D images.

    Parameters
    ----------
    image_a : array_like
        First 2D image (reference, stays fixed).
    image_b : array_like
        Second 2D image (draggable).
    title : str, optional
        Title displayed above the viewer.
    label_a : str, default "Image A"
        Label for the first image.
    label_b : str, default "Image B"
        Label for the second image.
    cmap : str, default "gray"
        Colormap name.
    opacity : float, default 0.5
        Blend ratio (0 = only A, 1 = only B).
    padding : float, default 0.2
        Fractional padding on each side. Adjustable from the frontend.
    pixel_size : float, default 0.0
        Pixel size in Å for scale bar (0 = uncalibrated).
    canvas_size : int, default 300
        Initial canvas size in CSS pixels for each column.
    auto_align : bool, default True
        Automatically compute initial alignment via cross-correlation.
    max_shift : float, default 0.0
        Maximum allowed shift in pixels (0 = unlimited, constrained by padding).
    rotation : float, default 0.0
        Initial rotation angle of image B in degrees.
    hist_source : str, default "a"
        Which image to show in the histogram ("a" or "b").

    Examples
    --------
    >>> import numpy as np
    >>> from quantem.widget import Align2D
    >>> a = np.random.rand(64, 64).astype(np.float32)
    >>> b = np.random.rand(64, 64).astype(np.float32)
    >>> Align2D(a, b, title="Alignment")
    """

    _esm = pathlib.Path(__file__).parent / "static" / "align2d.js"
    _css = pathlib.Path(__file__).parent / "static" / "align2d.css"

    # Image dimensions (unpadded)
    height = traitlets.Int(1).tag(sync=True)
    width = traitlets.Int(1).tag(sync=True)

    # Image data (unpadded, float32 bytes)
    image_a_bytes = traitlets.Bytes(b"").tag(sync=True)
    image_b_bytes = traitlets.Bytes(b"").tag(sync=True)

    # Padding (fractional, adjustable from frontend)
    padding = traitlets.Float(0.2).tag(sync=True)

    # Median values for padding fill
    median_a = traitlets.Float(0.0).tag(sync=True)
    median_b = traitlets.Float(0.0).tag(sync=True)

    # Alignment offset (image B relative to A, in pixels)
    dx = traitlets.Float(0.0).tag(sync=True)
    dy = traitlets.Float(0.0).tag(sync=True)

    # Rotation angle (degrees, of image B around its center)
    rotation = traitlets.Float(0.0).tag(sync=True)

    # Auto-aligned values (stored so user can restore)
    auto_dx = traitlets.Float(0.0).tag(sync=True)
    auto_dy = traitlets.Float(0.0).tag(sync=True)

    # Cross-correlation: NCC at (0,0) offset (baseline before alignment)
    xcorr_zero = traitlets.Float(0.0).tag(sync=True)

    # NCC at auto-aligned position (accurate, computed by Python)
    ncc_aligned = traitlets.Float(0.0).tag(sync=True)

    # Display
    title = traitlets.Unicode("").tag(sync=True)
    cmap = traitlets.Unicode("gray").tag(sync=True)
    opacity = traitlets.Float(0.5).tag(sync=True)
    label_a = traitlets.Unicode("Image A").tag(sync=True)
    label_b = traitlets.Unicode("Image B").tag(sync=True)

    # Scale bar
    pixel_size = traitlets.Float(0.0).tag(sync=True)

    # Bounds
    max_shift = traitlets.Float(0.0).tag(sync=True)

    # UI
    canvas_size = traitlets.Int(300).tag(sync=True)
    hist_source = traitlets.Unicode("a").tag(sync=True)

    # Tool visibility / locking
    disabled_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)
    hidden_tools = traitlets.List(traitlets.Unicode()).tag(sync=True)

    @classmethod
    def _normalize_tool_groups(cls, tool_groups) -> list[str]:
        return normalize_tool_groups("Align2D", tool_groups)

    @classmethod
    def _build_disabled_tools(
        cls,
        disabled_tools=None,
        disable_alignment: bool = False,
        disable_overlay: bool = False,
        disable_display: bool = False,
        disable_histogram: bool = False,
        disable_stats: bool = False,
        disable_export: bool = False,
        disable_view: bool = False,
        disable_all: bool = False,
    ) -> list[str]:
        return build_tool_groups(
            "Align2D",
            tool_groups=disabled_tools,
            all_flag=disable_all,
            flag_map={
                "alignment": disable_alignment,
                "overlay": disable_overlay,
                "display": disable_display,
                "histogram": disable_histogram,
                "stats": disable_stats,
                "export": disable_export,
                "view": disable_view,
            },
        )

    @classmethod
    def _build_hidden_tools(
        cls,
        hidden_tools=None,
        hide_alignment: bool = False,
        hide_overlay: bool = False,
        hide_display: bool = False,
        hide_histogram: bool = False,
        hide_stats: bool = False,
        hide_export: bool = False,
        hide_view: bool = False,
        hide_all: bool = False,
    ) -> list[str]:
        return build_tool_groups(
            "Align2D",
            tool_groups=hidden_tools,
            all_flag=hide_all,
            flag_map={
                "alignment": hide_alignment,
                "overlay": hide_overlay,
                "display": hide_display,
                "histogram": hide_histogram,
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
        image_a: Union[np.ndarray, "torch.Tensor"],
        image_b: Union[np.ndarray, "torch.Tensor"],
        title: str = "",
        label_a: str = "Image A",
        label_b: str = "Image B",
        cmap: str = "gray",
        opacity: float = 0.5,
        padding: float = 0.2,
        pixel_size: float = 0.0,
        canvas_size: int = 300,
        auto_align: bool = True,
        max_shift: float = 0.0,
        rotation: float = 0.0,
        hist_source: str = "a",
        disabled_tools=None,
        disable_alignment: bool = False,
        disable_overlay: bool = False,
        disable_display: bool = False,
        disable_histogram: bool = False,
        disable_stats: bool = False,
        disable_export: bool = False,
        disable_view: bool = False,
        disable_all: bool = False,
        hidden_tools=None,
        hide_alignment: bool = False,
        hide_overlay: bool = False,
        hide_display: bool = False,
        hide_histogram: bool = False,
        hide_stats: bool = False,
        hide_export: bool = False,
        hide_view: bool = False,
        hide_all: bool = False,
        state=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.widget_version = resolve_widget_version()

        # Check if inputs are IOResult and extract metadata
        for img_ref in ("image_a", "image_b"):
            img_data = image_a if img_ref == "image_a" else image_b
            if isinstance(img_data, IOResult):
                if not title and img_data.title:
                    title = img_data.title
                if pixel_size == 0.0 and img_data.pixel_size is not None:
                    pixel_size = img_data.pixel_size
                if img_ref == "image_a":
                    image_a = img_data.data
                else:
                    image_b = img_data.data

        # Check if inputs are Dataset2d and extract metadata
        for img_data in (image_a, image_b):
            if hasattr(img_data, "array") and hasattr(img_data, "name") and hasattr(img_data, "sampling"):
                if not title and img_data.name:
                    title = img_data.name
                if pixel_size == 0.0 and hasattr(img_data, "units"):
                    units = list(img_data.units)
                    sampling_val = float(img_data.sampling[-1])
                    # pixel_size is in Å — convert if units are nm
                    if units[-1] in ("nm", "nanometer"):
                        sampling_val = sampling_val * 10  # nm → Å
                    pixel_size = sampling_val

        # Extract arrays from Dataset objects
        if hasattr(image_a, "array"):
            image_a = image_a.array
        if hasattr(image_b, "array"):
            image_b = image_b.array

        a = to_numpy(image_a).astype(np.float32)
        b = to_numpy(image_b).astype(np.float32)
        if a.ndim != 2:
            raise ValueError(f"Align2D requires 2D images, image_a is {a.ndim}D")
        if b.ndim != 2:
            raise ValueError(f"Align2D requires 2D images, image_b is {b.ndim}D")

        # Resize smaller to match larger
        target_h = max(a.shape[0], b.shape[0])
        target_w = max(a.shape[1], b.shape[1])
        if a.shape != (target_h, target_w):
            a = _resize_image(a, target_h, target_w)
        if b.shape != (target_h, target_w):
            b = _resize_image(b, target_h, target_w)

        self.height = target_h
        self.width = target_w
        self.padding = padding

        # Medians for padding fill
        self.median_a = float(np.median(a))
        self.median_b = float(np.median(b))

        # Display options
        self.title = title
        self.label_a = label_a
        self.label_b = label_b
        self.cmap = cmap
        self.opacity = opacity
        self.pixel_size = pixel_size
        self.canvas_size = canvas_size
        self.max_shift = max_shift
        self.rotation = rotation
        self.hist_source = hist_source
        self.disabled_tools = self._build_disabled_tools(
            disabled_tools=disabled_tools,
            disable_alignment=disable_alignment,
            disable_overlay=disable_overlay,
            disable_display=disable_display,
            disable_histogram=disable_histogram,
            disable_stats=disable_stats,
            disable_export=disable_export,
            disable_view=disable_view,
            disable_all=disable_all,
        )
        self.hidden_tools = self._build_hidden_tools(
            hidden_tools=hidden_tools,
            hide_alignment=hide_alignment,
            hide_overlay=hide_overlay,
            hide_display=hide_display,
            hide_histogram=hide_histogram,
            hide_stats=hide_stats,
            hide_export=hide_export,
            hide_view=hide_view,
            hide_all=hide_all,
        )

        # Cross-correlation at (0,0) — baseline
        self.xcorr_zero = _compute_ncc(a, b, 0.0, 0.0)

        # Translation alignment via phase correlation
        if auto_align:
            try:
                limit_x = int(max_shift if max_shift > 0 else target_w * padding)
                limit_y = int(max_shift if max_shift > 0 else target_h * padding)
                best_dx, best_dy = _cross_correlate_fft(a, b, limit_x, limit_y)
                self.dx = max(-limit_x, min(limit_x, best_dx))
                self.dy = max(-limit_y, min(limit_y, best_dy))
                # Store auto values so user can restore
                self.auto_dx = self.dx
                self.auto_dy = self.dy
                # Compute NCC at aligned position
                self.ncc_aligned = _compute_ncc(a, b, self.dx, self.dy)
            except Exception as e:
                import warnings
                warnings.warn(f"Auto-alignment failed: {e}", stacklevel=2)

        # Send unpadded bytes
        self.image_a_bytes = a.tobytes()
        self.image_b_bytes = b.tobytes()

        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(
                    json.loads(pathlib.Path(state).read_text()),
                    require_envelope=True,
                )
            else:
                state = unwrap_state_payload(state)
            self.load_state_dict(state)

    def set_images(self, image_a, image_b, auto_align=True):
        """Replace both images. Preserves display settings, recomputes alignment."""
        if hasattr(image_a, "array"):
            image_a = image_a.array
        if hasattr(image_b, "array"):
            image_b = image_b.array
        a = to_numpy(image_a).astype(np.float32)
        b = to_numpy(image_b).astype(np.float32)
        if a.ndim != 2:
            raise ValueError(f"Align2D requires 2D images, image_a is {a.ndim}D")
        if b.ndim != 2:
            raise ValueError(f"Align2D requires 2D images, image_b is {b.ndim}D")
        target_h = max(a.shape[0], b.shape[0])
        target_w = max(a.shape[1], b.shape[1])
        if a.shape != (target_h, target_w):
            a = _resize_image(a, target_h, target_w)
        if b.shape != (target_h, target_w):
            b = _resize_image(b, target_h, target_w)
        self.height = target_h
        self.width = target_w
        self.median_a = float(np.median(a))
        self.median_b = float(np.median(b))
        self.xcorr_zero = _compute_ncc(a, b, 0.0, 0.0)
        self.dx = 0.0
        self.dy = 0.0
        self.rotation = 0.0
        if auto_align:
            try:
                limit_x = int(self.max_shift if self.max_shift > 0 else target_w * self.padding)
                limit_y = int(self.max_shift if self.max_shift > 0 else target_h * self.padding)
                best_dx, best_dy = _cross_correlate_fft(a, b, limit_x, limit_y)
                self.dx = max(-limit_x, min(limit_x, best_dx))
                self.dy = max(-limit_y, min(limit_y, best_dy))
                self.auto_dx = self.dx
                self.auto_dy = self.dy
                self.ncc_aligned = _compute_ncc(a, b, self.dx, self.dy)
            except Exception:
                pass
        self.image_a_bytes = a.tobytes()
        self.image_b_bytes = b.tobytes()

    def __repr__(self) -> str:
        return f"Align2D({self.height}×{self.width}, dx={self.dx:.1f}, dy={self.dy:.1f}, rot={self.rotation:.1f}°)"

    def state_dict(self):
        return {
            "title": self.title,
            "label_a": self.label_a,
            "label_b": self.label_b,
            "cmap": self.cmap,
            "opacity": self.opacity,
            "padding": self.padding,
            "dx": self.dx,
            "dy": self.dy,
            "rotation": self.rotation,
            "pixel_size": self.pixel_size,
            "max_shift": self.max_shift,
            "canvas_size": self.canvas_size,
            "hist_source": self.hist_source,
            "disabled_tools": self.disabled_tools,
            "hidden_tools": self.hidden_tools,
        }

    def save(self, path: str):
        save_state_file(path, "Align2D", self.state_dict())

    def load_state_dict(self, state):
        for key, val in state.items():
            if hasattr(self, key):
                setattr(self, key, val)

    def summary(self):
        lines = [self.title or "Align2D", "═" * 32]
        lines.append(f"Image:    {self.height}×{self.width}")
        if self.pixel_size > 0:
            ps = self.pixel_size
            if ps >= 10:
                lines[-1] += f" ({ps / 10:.2f} nm/px)"
            else:
                lines[-1] += f" ({ps:.2f} Å/px)"
        lines.append(f"Labels:   A={self.label_a!r}  B={self.label_b!r}")
        lines.append(f"Offset:   dx={self.dx:.2f}  dy={self.dy:.2f}  rotation={self.rotation:.2f}°")
        lines.append(f"Display:  {self.cmap} | opacity={self.opacity:.0%} | padding={self.padding:.0%}")
        if self.ncc_aligned != 0:
            lines.append(f"NCC:      aligned={self.ncc_aligned:.4f}  zero={self.xcorr_zero:.4f}")
        print("\n".join(lines))

    def reset_alignment(self):
        self.dx = 0.0
        self.dy = 0.0
        self.rotation = 0.0

    @property
    def offset(self) -> tuple[float, float]:
        """Return (dx, dy) alignment offset."""
        return (self.dx, self.dy)


bind_tool_runtime_api(Align2D, "Align2D")
