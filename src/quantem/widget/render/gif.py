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
import subprocess
import tempfile

import numpy as np

# Resolution multiplier per quality tier. GIF is a 256-colour palette format
# regardless, so quality here means spatial resolution (and therefore file size).
QUALITY_SCALE = {"high": 1.0, "medium": 0.6, "low": 0.35}
MIN_TITLE_FONT_SIZE = 12
MIN_SCALE_FONT_SIZE = 12
MIN_SCALE_BAR_THICKNESS = 5
MIN_OVERLAY_MARGIN = 12
BACKGROUND_COLORS = {
    "dark": (12, 12, 12),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "transparent": (0, 0, 0),
}


def _round_to_nice_value(value: float) -> float:
    """Port of js/figure.ts roundToNiceValue: snap to 1 / 2 / 5 / 10 x 10^n."""
    if value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    normalized = value / magnitude
    if normalized < 1.5:
        return magnitude
    if normalized < 3.5:
        return 2 * magnitude
    if normalized < 7.5:
        return 5 * magnitude
    return 10 * magnitude


def _unit_symbol(unit: str) -> str:
    """Port of js/figure.ts unitSymbol: render the conventional glyph."""
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
    return u


def _format_scale_label(value: float, unit: str) -> str:
    """Port of js/figure.ts formatScaleLabel."""
    nice = _round_to_nice_value(value)
    sym = _unit_symbol(unit)
    return f"{round(nice)} {sym}" if nice >= 1 else f"{nice:.2f} {sym}"


def _draw_text_shadow(draw, xy: tuple[float, float], text: str, font, *, anchor: str | None = None) -> None:
    """Draw white overlay text with the 1 px shadow used by widget canvases."""
    kwargs = {"fill": (0, 0, 0), "font": font}
    if anchor is not None:
        kwargs["anchor"] = anchor
    draw.text((xy[0] + 1, xy[1] + 1), text, **kwargs)
    kwargs["fill"] = (255, 255, 255)
    draw.text(xy, text, **kwargs)


def _draw_scalebar(
    img,
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
    from PIL import ImageDraw, ImageFont
    if pixel_size <= 0:
        return img
    width, height = img.size
    target_bar_px = min(60.0, width * 0.25)
    bar_thickness = MIN_SCALE_BAR_THICKNESS
    font_size = max(MIN_SCALE_FONT_SIZE, 16)
    margin = MIN_OVERLAY_MARGIN
    effective_zoom = max(1e-6, float(zoom))
    nice_phys = _round_to_nice_value((target_bar_px / effective_zoom) * pixel_size)
    bar_px = (nice_phys / pixel_size) * effective_zoom
    bar_y = height - margin
    bar_x = width - bar_px - margin
    draw = ImageDraw.Draw(img)
    draw.rectangle([bar_x, bar_y, bar_x + bar_px, bar_y + bar_thickness], fill=(255, 255, 255))
    label = _format_scale_label(nice_phys, unit)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    _draw_text_shadow(draw, (bar_x + bar_px / 2, bar_y - 4), label, font, anchor="mb")
    if show_zoom_indicator:
        _draw_text_shadow(draw, (margin, height - margin + bar_thickness), f"{effective_zoom:.1f}x", font, anchor="lb")
    return img


def colorize(normalized_uint8: np.ndarray, cmap_name: str):
    """Map a 0-255 normalized frame through a matplotlib colormap to an RGB PIL image."""
    from matplotlib import colormaps
    from PIL import Image
    cmap_fn = colormaps.get_cmap(cmap_name)
    rgb = (cmap_fn(normalized_uint8 / 255.0)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def animation_output_scale(
    width: int,
    height: int,
    quality: str,
    *,
    downsample: int = 1,
    max_edge_px: int | None = None,
) -> float:
    """Return the explicit display scale for an exported animation frame."""
    width = max(1, int(width))
    height = max(1, int(height))
    downsample = max(1, int(downsample))
    scale = min(1.0, float(QUALITY_SCALE.get(quality, 1.0)) / float(downsample))
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
    downsample: int = 1,
    max_edge_px: int | None = None,
):
    """Downscale explicitly, then draw the scale bar at output resolution."""
    scale = animation_output_scale(
        img.size[0],
        img.size[1],
        quality,
        downsample=downsample,
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
        pixel_size / scale if scale > 0 else pixel_size,
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
    return tuple(max(0, min(255, int(v))) for v in background)


def _font(size: int, *, bold: bool = False):
    from PIL import ImageFont
    names = ["DejaVuSans-Bold.ttf"] if bold else ["DejaVuSans.ttf", "DejaVuSans-Bold.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_panel_title(img, text: str, font_size: int) -> None:
    """Draw a compact centered panel title over the image."""
    if not text:
        return
    from PIL import ImageDraw
    width, _height = img.size
    font = _font(max(MIN_TITLE_FONT_SIZE, int(font_size)), bold=True)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    x = max(2, (width - tw) / 2)
    y = 3
    draw.text((x + 1, y + 1), text, fill=(0, 0, 0), font=font)
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def compose_panel_grid(
    images: list,
    *,
    panel_titles: list[str] | None = None,
    frame_labels: list[str] | None = None,
    show_panel_titles: bool = True,
    title_font_size: int = 11,
    max_cols: int = 4,
    panel_gap: int = 10,
    background: str | tuple[int, int, int] = "dark",
    outer_border: int = 0,
    outer_border_color: str | tuple[int, int, int] | None = None,
    panel_inner_border: int = 0,
    panel_inner_border_color: str | tuple[int, int, int] = "black",
):
    """Compose per-panel PIL images into the widget-style panel grid."""
    if not images:
        raise ValueError("compose_panel_grid requires at least one image")
    from PIL import Image, ImageDraw
    n_panels = len(images)
    cols = n_panels if max_cols <= 0 else min(max(1, int(max_cols)), n_panels)
    rows = int(math.ceil(n_panels / cols))
    widths = [img.size[0] for img in images]
    heights = [img.size[1] for img in images]
    cell_w = max(widths)
    cell_h = max(heights)
    gap = max(0, int(panel_gap))
    outer = max(0, int(outer_border))
    inner = max(0, int(panel_inner_border))
    bg_rgb = normalize_background(background)
    outer_rgb = normalize_background(outer_border_color) if outer_border_color is not None else bg_rgb
    canvas = Image.new(
        "RGB",
        (
            cols * cell_w + gap * (cols - 1) + 2 * outer,
            rows * cell_h + gap * (rows - 1) + 2 * outer,
        ),
        outer_rgb,
    )
    if gap > 0 and bg_rgb != outer_rgb:
        draw = ImageDraw.Draw(canvas)
        draw.rectangle(
            [
                outer,
                outer,
                canvas.size[0] - outer,
                canvas.size[1] - outer,
            ],
            fill=bg_rgb,
            outline=None,
        )
    for i, src in enumerate(images):
        panel = src.convert("RGB").copy()
        if show_panel_titles:
            title_parts: list[str] = []
            if panel_titles and i < len(panel_titles):
                title_parts.append(str(panel_titles[i]))
            if frame_labels and i < len(frame_labels) and frame_labels[i]:
                title_parts.append(str(frame_labels[i]))
            _draw_panel_title(panel, " · ".join(title_parts), title_font_size)
        row, col = divmod(i, cols)
        x = outer + col * (cell_w + gap) + (cell_w - panel.size[0]) // 2
        y = outer + row * (cell_h + gap) + (cell_h - panel.size[1]) // 2
        canvas.paste(panel, (x, y))
        if inner > 0:
            draw = ImageDraw.Draw(canvas)
            color = normalize_background(panel_inner_border_color)
            for offset in range(inner):
                draw.rectangle(
                    [
                        x + offset,
                        y + offset,
                        x + panel.size[0] - 1 - offset,
                        y + panel.size[1] - 1 - offset,
                    ],
                    outline=color,
                )
    return canvas


def _even_rgb_array(frame) -> np.ndarray:
    """Return an RGB uint8 array padded to even dimensions for H.264."""
    arr = np.asarray(frame.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        padded = np.zeros((h + pad_h, w + pad_w, 3), dtype=np.uint8)
        padded[:h, :w] = arr
        if pad_h:
            padded[h:, :w] = arr[h - 1 : h]
        if pad_w:
            padded[:h, w:] = arr[:, w - 1 : w]
        if pad_h and pad_w:
            padded[h:, w:] = arr[h - 1, w - 1]
        arr = padded
    return arr


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


def write_mp4(frames: list, path: str | pathlib.Path, fps: float, *, crf: int = 18) -> pathlib.Path:
    """Assemble RGB PIL frames into an H.264 MP4 using the local ffmpeg binary."""
    if not frames:
        raise ValueError("write_mp4 requires at least one frame")
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio_ffmpeg
    except ImportError:
        ffmpeg = "ffmpeg"
    else:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    arrays = [_even_rgb_array(frame) for frame in frames]
    height, width = arrays[0].shape[:2]
    for i, arr in enumerate(arrays[1:], start=1):
        if arr.shape[:2] != (height, width):
            raise ValueError(
                f"all MP4 frames must have the same size; frame 0 is {(height, width)}, "
                f"frame {i} is {arr.shape[:2]}"
            )
    with tempfile.TemporaryDirectory(prefix="quantem-show3d-mp4-") as tmp:
        tmp_path = pathlib.Path(tmp)
        for i, arr in enumerate(arrays):
            from PIL import Image
            Image.fromarray(arr, mode="RGB").save(tmp_path / f"frame_{i:06d}.png")
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            f"{max(0.1, float(fps))}",
            "-i",
            str(tmp_path / "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(int(crf)),
            str(path),
        ]
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "save_mp4 requires ffmpeg on PATH. Install ffmpeg or use save_gif instead."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"ffmpeg failed while writing MP4: {exc}") from exc
    return path
