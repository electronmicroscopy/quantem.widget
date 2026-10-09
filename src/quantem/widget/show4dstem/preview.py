"""Host-side renders of the viewer's two panels.

The browser draws the live widget; Python only renders when a picture has to
exist without a kernel: the static preview a saved notebook falls back to, and
the virtual-image pages of a report export. Both map pixels to colors exactly
like the frontend (same scale mode, same percentile window, same colormap).
"""

import base64
import io

import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from quantem.widget.colormap import colorize
from quantem.widget.render.figure import format_scale_label, round_to_nice, static_overlay_font


def scale_values(values: np.ndarray, mode: str) -> np.ndarray:
    """Apply the panel's display scale: ``log`` is log1p of the clipped counts."""
    values = np.asarray(values, dtype=np.float32)
    if mode == "log":
        return np.log1p(np.maximum(values, 0.0))
    return values


def display_range(values: np.ndarray, vmin_pct: float, vmax_pct: float) -> tuple[float, float]:
    """Linear percentile window of the scaled values, as the contrast sliders define it."""
    low = float(values.min()) if values.size else 0.0
    high = float(values.max()) if values.size else 0.0
    lo_pct, hi_pct = sorted((max(0.0, min(100.0, vmin_pct)), max(0.0, min(100.0, vmax_pct))))
    return low + lo_pct / 100 * (high - low), low + hi_pct / 100 * (high - low)


def colormap_rgb(values: np.ndarray, cmap_name: str, vmin: float, vmax: float) -> np.ndarray:
    """uint8 RGB image of ``values`` through a matplotlib colormap between ``vmin`` and ``vmax``."""
    values = np.asarray(values, dtype=np.float32)
    normalized = np.zeros_like(values) if vmax <= vmin else np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
    return colorize(normalized, cmap_name)


def png_data_uri(rgb: np.ndarray) -> str:
    """Inline PNG data URI for a report page."""
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def diffraction_rgb(widget) -> np.ndarray:
    """The diffraction panel as RGB: the reduced ROI pattern when a scan ROI is active, else the cursor pattern."""
    raw = widget.vi_roi_pattern() if widget.vi_roi_mode != "off" else widget.pattern(widget.pos_row, widget.pos_col)
    return _panel_rgb(raw, widget.dp_scale_mode, widget.dp_colormap, (widget.dp_vmin, widget.dp_vmax),
                      (widget.dp_vmin_pct, widget.dp_vmax_pct))


def virtual_rgb(widget) -> np.ndarray:
    """The virtual-image panel as RGB with the widget's colormap and contrast window."""
    return _panel_rgb(widget.virtual_image(), widget.vi_scale_mode, widget.vi_colormap, (widget.vi_vmin, widget.vi_vmax),
                      (widget.vi_vmin_pct, widget.vi_vmax_pct))


def _panel_rgb(values: np.ndarray, scale_mode: str, cmap_name: str, window, window_pct) -> np.ndarray:
    """``values`` through one panel's display scale, contrast window and colormap, as the frontend paints them.

    An explicit ``(vmin, vmax)`` window is clipped at zero and scaled like the
    data; without one the percentile sliders ``window_pct`` set the window.
    """
    scaled = scale_values(values, scale_mode)
    vmin, vmax = window
    if vmin is not None and vmax is not None:
        bounds = scale_values(np.array([max(vmin, 0), max(vmax, 0)]), scale_mode)
        vmin, vmax = float(bounds[0]), float(bounds[1])
    else:
        vmin, vmax = display_range(scaled, *window_pct)
    return colormap_rgb(scaled, cmap_name, vmin, vmax)


def static_png_b64(widget, *, max_px: int = 384, dpi: int = 160) -> str | None:
    """Base64 PNG of the live two-panel layout: virtual image beside the diffraction pattern.

    A saved notebook without a kernel still shows where the scan cursor was and
    which detector ROI produced the virtual image, while the heavy 4D stack
    stays out of the notebook metadata.
    """
    if widget._data is None:
        return None
    panel_px = max(64, min(int(widget.panel_width_px or max_px), int(max_px)))
    title = str(widget.title or "").strip()
    title_h = 18 if title else 0
    virtual = _panel(widget, "virtual", panel_px)
    diffraction = _panel(widget, "diffraction", panel_px)
    composite = Image.new("RGB", (panel_px * 2, panel_px + title_h), color=(255, 255, 255))
    if title:
        ImageDraw.Draw(composite).text((2, 2), title, fill=(0, 0, 0), font=ImageFont.load_default())
    composite.paste(virtual, (0, title_h))
    composite.paste(diffraction, (panel_px, title_h))
    buffer = io.BytesIO()
    composite.save(buffer, format="PNG", dpi=(dpi, dpi))
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _panel(widget, panel_key: str, panel_px: int) -> Image.Image:
    """One preview panel resized to the live panel size, with its overlays and scale bar."""
    rgb = virtual_rgb(widget) if panel_key == "virtual" else diffraction_rgb(widget)
    panel = Image.fromarray(rgb, mode="RGB")
    if panel.size != (panel_px, panel_px):
        # Enlarged panels keep sharp data pixels as the live canvases draw them (the pattern
        # always, the virtual image unless vi_smooth); a shrink needs Lanczos to avoid aliasing.
        if panel_px < max(panel.size):
            resample = Image.Resampling.LANCZOS
        elif panel_key == "virtual" and widget.vi_smooth:
            resample = Image.Resampling.BILINEAR
        else:
            resample = Image.Resampling.NEAREST
        panel = panel.resize((panel_px, panel_px), resample=resample)
    if panel_key == "virtual":
        _draw_virtual_overlays(widget, panel)
        _draw_scalebar(panel, float(widget.pixel_size), widget.pixel_unit or "px")
    else:
        _draw_diffraction_overlays(widget, panel)
        _draw_scalebar(panel, float(widget.k_pixel_size), "mrad" if widget.k_calibrated else "px")
    return panel


def _draw_crosshair(draw, x: float, y: float, size: float, color, width: int) -> None:
    """Plus marker of half-length ``size`` at screen point ``(x, y)``, the live canvases' position marker."""
    draw.line([(x - size, y), (x + size, y)], fill=color, width=width)
    draw.line([(x, y - size), (x, y + size)], fill=color, width=width)


def _draw_scalebar(image: Image.Image, pixel_size: float, unit: str) -> None:
    """Show2D's static scale-bar geometry (16 px label, 5 px bar, 12 px margin) so 4D and 2D previews match."""
    if pixel_size <= 0:
        return
    draw = ImageDraw.Draw(image, mode="RGBA")
    font = ImageFont.truetype(
        font_manager.findfont(font_manager.FontProperties(family=static_overlay_font())), 16
    )
    width, height = image.size
    margin, thickness = 12, 5
    nice_physical = round_to_nice(max(36, int(width * 0.15)) * pixel_size)
    bar_px = min(max(12, int(round(nice_physical / pixel_size))), max(12, int(width * 0.8)))
    x1, y1 = width - margin, height - margin
    x0, y0 = x1 - bar_px, y1 - thickness
    draw.rectangle([(x0 + 1, y0 + 1), (x1 + 1, y1 + 1)], fill=(0, 0, 0, 180))
    draw.rectangle([(x0, y0), (x1, y1)], fill=(255, 255, 255, 255))
    label = format_scale_label(nice_physical, unit)
    bbox = draw.textbbox((0, 0), label, font=font)
    tx = x0 + (bar_px - (bbox[2] - bbox[0])) / 2
    ty = y0 - (bbox[3] - bbox[1]) - 4
    draw.text((tx + 1, ty + 1), label, fill=(0, 0, 0, 220), font=font)
    draw.text((tx, ty), label, fill=(255, 255, 255, 255), font=font)
    zoom_bbox = draw.textbbox((0, 0), "1.0x", font=font)
    zy = height - margin - (zoom_bbox[3] - zoom_bbox[1])
    draw.text((margin + 1, zy + 1), "1.0x", fill=(0, 0, 0, 220), font=font)
    draw.text((margin, zy), "1.0x", fill=(255, 255, 255, 255), font=font)


def _draw_diffraction_overlays(widget, image: Image.Image) -> None:
    """Detector ROI outline and center marker in the frontend's green, plus the line profile.

    ROI centers and profile points center pixel i at i (the mask convention), and the image
    draws pixel i over [i, i + 1), so a coordinate lands at ``(i + 0.5) * scale`` as in the browser.
    """
    draw = ImageDraw.Draw(image, mode="RGBA")
    width, height = image.size
    scale_x = width / max(1, widget.det_cols)
    scale_y = height / max(1, widget.det_rows)
    cx = (widget.roi_center_col + 0.5) * scale_x
    cy = (widget.roi_center_row + 0.5) * scale_y
    if widget.roi_active and widget.roi_mode != "point":
        stroke, fill = (0, 220, 0, 240), (0, 220, 0, 45)
        _draw_roi(draw, widget.roi_mode, (cx, cy), widget.roi_radius, widget.roi_width, widget.roi_height,
                  (scale_x, scale_y), stroke, fill)
        if widget.roi_mode == "annular":
            _draw_roi(draw, "circle", (cx, cy), widget.roi_radius_inner, 0.0, 0.0, (scale_x, scale_y), stroke, (0, 0, 0, 0))
    marker = (0, 220, 0, 255) if widget.roi_active else (255, 100, 100, 255)
    _draw_crosshair(draw, cx, cy, size=max(6, int(min(width, height) * 0.03)), color=marker, width=2)
    if len(widget.profile_line) == 2:
        start, end = widget.profile_line
        x0, y0 = (start["col"] + 0.5) * scale_x, (start["row"] + 0.5) * scale_y
        x1, y1 = (end["col"] + 0.5) * scale_x, (end["row"] + 0.5) * scale_y
        draw.line([(x0, y0), (x1, y1)], fill=(0, 200, 255, 240), width=max(1, int(widget.profile_width)))
        for x, y in ((x0, y0), (x1, y1)):
            draw.ellipse([(x - 3, y - 3), (x + 3, y + 3)], fill=(0, 200, 255, 255))


def _draw_virtual_overlays(widget, image: Image.Image) -> None:
    """Scan cursor crosshair and, when active, the scan ROI that feeds the diffraction panel (pixel centers, as above)."""
    draw = ImageDraw.Draw(image, mode="RGBA")
    width, height = image.size
    scale_x = width / max(1, widget.shape_cols)
    scale_y = height / max(1, widget.shape_rows)
    size = max(6, int(min(width, height) * 0.03))
    _draw_crosshair(draw, (widget.pos_col + 0.5) * scale_x, (widget.pos_row + 0.5) * scale_y, size=size,
                    color=(255, 100, 100, 240), width=2)
    if widget.vi_roi_mode == "off":
        return
    cx = (widget.vi_roi_center_col + 0.5) * scale_x
    cy = (widget.vi_roi_center_row + 0.5) * scale_y
    stroke, fill = (180, 80, 255, 240), (180, 80, 255, 45)
    _draw_roi(draw, widget.vi_roi_mode, (cx, cy), widget.vi_roi_radius, widget.vi_roi_width, widget.vi_roi_height,
              (scale_x, scale_y), stroke, fill)
    _draw_crosshair(draw, cx, cy, size=size, color=stroke, width=2)


def _draw_roi(draw, mode: str, center: tuple[float, float], radius: float, width: float, height: float,
              scale: tuple[float, float], stroke, fill) -> None:
    """One ROI outline at screen point ``center``: an ellipse for ``circle`` and ``annular``, else a rectangle.

    A ``rect`` spans ``width`` by ``height`` detector or scan pixels and a
    ``square`` has half side ``radius``, as the live canvases draw them;
    ``scale`` converts pixels to preview pixels per axis.
    """
    cx, cy = center
    scale_x, scale_y = scale
    if mode in ("circle", "annular"):
        rx, ry = radius * scale_x, radius * scale_y
        draw.ellipse([(cx - rx, cy - ry), (cx + rx, cy + ry)], outline=stroke, fill=fill, width=2)
    else:
        rx = (width / 2 if mode == "rect" else radius) * scale_x
        ry = (height / 2 if mode == "rect" else radius) * scale_y
        draw.rectangle([(cx - rx, cy - ry), (cx + rx, cy + ry)], outline=stroke, fill=fill, width=2)
