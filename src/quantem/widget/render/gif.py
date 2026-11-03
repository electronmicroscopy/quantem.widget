"""
Shared animation export helpers for Show3D.

A widget produces per-frame contrast-normalized uint8 arrays (so the GIF matches
the histogram range / log scale the operator set); these helpers colorize with
the chosen matplotlib colormap, burn in a scale bar whose format matches the
widget's figure-export scale bar exactly (js/figure.ts), downscale by a quality
factor, and assemble the animation at the widget's fps. No widget dependency:
only numpy / matplotlib / PIL.
"""
import math
import pathlib

import numpy as np

from quantem.widget import colormap
from quantem.widget.render.figure import format_scale_label, round_to_nice

# Resolution multiplier per quality tier. GIF is a 256-colour palette format
# regardless, so quality here means spatial resolution (and therefore file size).
QUALITY_SCALE = {"high": 1.0, "medium": 0.6, "low": 0.35}
MIN_TITLE_FONT_SIZE = 12
MIN_SCALE_BAR_THICKNESS = 5
MIN_OVERLAY_MARGIN = 12
BACKGROUND_COLORS = {
    "dark": (12, 12, 12),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
}


def _draw_text_shadow(draw, xy: tuple[float, float], text: str, font, *, anchor: str | None = None) -> None:
    """Draw white overlay text with the 1 px shadow used by widget canvases."""
    kwargs = {"fill": (0, 0, 0), "font": font}
    if anchor is not None:
        kwargs["anchor"] = anchor
    draw.text((xy[0] + 1, xy[1] + 1), text, **kwargs)
    kwargs["fill"] = (255, 255, 255)
    draw.text(xy, text, **kwargs)


def _draw_scalebar(
    image,
    pixel_size: float,
    unit: str,
    *,
    show_zoom_indicator: bool = False,
    zoom: float = 1.0,
):
    """Burn the Show3D canvas scale bar and optional zoom readout into a frame.

    The geometry mirrors the widget canvas overlay, with publication-readable
    minimums: a 60 px target bar capped at 25% of the panel width, at least a
    5 px bar thickness, at least 12 px text, and at least 12 px margin.  The
    zoom readout is bottom-left when enabled; the scale bar is bottom-right.
    """
    from PIL import ImageDraw
    if pixel_size <= 0:
        return image
    width, height = image.size
    target_bar_px = min(60.0, width * 0.25)
    bar_thickness = MIN_SCALE_BAR_THICKNESS
    font_size = 16  # the canvas label size, already above the 12 px publication minimum
    margin = MIN_OVERLAY_MARGIN
    effective_zoom = max(1e-6, float(zoom))
    nice_phys = round_to_nice((target_bar_px / effective_zoom) * pixel_size)
    bar_px = (nice_phys / pixel_size) * effective_zoom
    bar_y = height - margin
    bar_x = width - bar_px - margin
    draw = ImageDraw.Draw(image)
    draw.rectangle([bar_x, bar_y, bar_x + bar_px, bar_y + bar_thickness], fill=(255, 255, 255))
    label = format_scale_label(nice_phys, unit)
    font = _font(font_size, bold=True)
    _draw_text_shadow(draw, (bar_x + bar_px / 2, bar_y - 4), label, font, anchor="mb")
    if show_zoom_indicator:
        _draw_text_shadow(draw, (margin, height - margin + bar_thickness), f"{effective_zoom:.1f}x", font, anchor="lb")
    return image


def colorize(normalized_uint8: np.ndarray, cmap_name: str):
    """Map a 0-255 normalized frame through a matplotlib colormap to an RGB PIL image."""
    from PIL import Image
    return Image.fromarray(colormap.colorize(normalized_uint8 / 255.0, cmap_name), mode="RGB")


def animation_output_scale(
    width: int,
    height: int,
    quality: str,
    *,
    image_bin: int = 1,
    max_edge_px: int | None = None,
) -> float:
    """Return the explicit display scale for an exported animation frame."""
    width = max(1, int(width))
    height = max(1, int(height))
    image_bin = max(1, int(image_bin))
    scale = min(1.0, float(QUALITY_SCALE.get(quality, 1.0)) / float(image_bin))
    if max_edge_px is not None:
        max_edge_px = max(1, int(max_edge_px))
        scale = min(scale, max_edge_px / float(max(width, height)))
    return max(scale, 1.0 / float(max(width, height)))


def finalize_frame(
    img,
    quality: str,
    pixel_size: float,
    unit: str,
    *,
    show_zoom_indicator: bool = False,
    zoom: float = 1.0,
    image_bin: int = 1,
    max_edge_px: int | None = None,
):
    """Downscale explicitly, then draw the scale bar at output resolution."""
    scale = animation_output_scale(
        img.size[0],
        img.size[1],
        quality,
        image_bin=image_bin,
        max_edge_px=max_edge_px,
    )
    if scale < 1.0:
        from PIL import Image
        width, height = img.size
        img = img.resize(
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            Image.LANCZOS,
        )
    # Each output pixel spans pixel_size / scale of sample after the downscale.
    return _draw_scalebar(
        img,
        pixel_size / scale,
        unit,
        show_zoom_indicator=show_zoom_indicator,
        zoom=zoom,
    )


def normalize_background(background: str | tuple[int, int, int]) -> tuple[int, int, int]:
    """Normalize a named or RGB animation-grid background color."""
    if isinstance(background, str):
        key = background.strip().lower()
        if key in BACKGROUND_COLORS:
            return BACKGROUND_COLORS[key]
        if key.startswith("#") and len(key) == 7:
            try:
                return tuple(int(key[i : i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]
            except ValueError:
                pass
        raise ValueError(
            "background must be 'dark', 'black', 'white', or a '#RRGGBB' color"
        )
    if len(background) != 3:
        raise ValueError("background RGB tuple must contain exactly three values")
    return tuple(max(0, min(255, int(value))) for value in background)


def _font(size: int, *, bold: bool = False):
    """DejaVu Sans at ``size`` px (matplotlib ships it), else PIL's bitmap font, so frames render on any machine."""
    from PIL import ImageFont
    names = ["DejaVuSans-Bold.ttf"] if bold else ["DejaVuSans.ttf", "DejaVuSans-Bold.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_panel_title(image, text: str, font_size: int) -> None:
    """Draw a compact centered panel title over the image."""
    if not text:
        return
    from PIL import ImageDraw
    font = _font(max(MIN_TITLE_FONT_SIZE, int(font_size)), bold=True)
    draw = ImageDraw.Draw(image)
    left, _, right, _ = draw.textbbox((0, 0), text, font=font)
    _draw_text_shadow(draw, (max(2, (image.size[0] - (right - left)) / 2), 3), text, font)


def compose_panel_grid(
    images: list,
    *,
    panel_titles: list[str] | None = None,
    show_panel_titles: bool = True,
    title_font_size: int = 11,
    max_cols: int = 4,
    panel_gap: int = 10,
    background: str | tuple[int, int, int] = "dark",
):
    """Compose per-panel PIL images into the widget-style grid: ``max_cols`` columns,
    ``panel_gap`` px of ``background`` between cells, each panel centered in its cell
    under its title."""
    from PIL import Image
    n_panels = len(images)
    cols = n_panels if max_cols <= 0 else min(max(1, int(max_cols)), n_panels)
    rows = int(math.ceil(n_panels / cols))
    cell_width = max(image.size[0] for image in images)
    cell_height = max(image.size[1] for image in images)
    gap = max(0, int(panel_gap))
    canvas = Image.new(
        "RGB",
        (cols * cell_width + gap * (cols - 1), rows * cell_height + gap * (rows - 1)),
        normalize_background(background),
    )
    for index, source in enumerate(images):
        panel = source.convert("RGB").copy()
        if show_panel_titles and panel_titles and index < len(panel_titles):
            _draw_panel_title(panel, str(panel_titles[index]), title_font_size)
        row, col = divmod(index, cols)
        x = col * (cell_width + gap) + (cell_width - panel.size[0]) // 2
        y = row * (cell_height + gap) + (cell_height - panel.size[1]) // 2
        canvas.paste(panel, (x, y))
    return canvas


def write_gif(frames: list, path: str | pathlib.Path, fps: float) -> pathlib.Path:
    """Assemble RGB PIL frames into a looping GIF at the given fps."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(1, int(round(1000.0 / max(0.1, fps))))
    frames[0].save(
        str(path), save_all=True, append_images=frames[1:],
        duration=duration, loop=0, optimize=True, disposal=2,
    )
    return path
