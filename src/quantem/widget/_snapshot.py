"""Shared widget-faithful PNG snapshot helper.

Mirrors the WebGPU pipeline (log_scale -> normalize -> cmap LUT) so the static
PNG matches the canvas. Adds title, per-panel labels, and scale bar overlays
via PIL Draw, matching what the operator sees in the widget. Used by Show2D /
Show3D / Show3DSlices / Show4DSTEM ``_repr_mimebundle_`` static fallback.
"""
from __future__ import annotations

import io
from typing import Sequence

import numpy as np


_VIRIDIS = "viridis"
_LABEL_PAD = 22  # px reserved above each panel for label
_TITLE_PAD = 26  # px reserved above the canvas for title
_SCALEBAR_H = 4
_SCALEBAR_NICE_NM = (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)


def _normalize(arr: np.ndarray, vmin: float | None, vmax: float | None,
               log: bool) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if log:
        arr = np.sign(arr) * np.log1p(np.abs(arr))
    if vmin is None or vmax is None:
        finite = arr[np.isfinite(arr)]
        if finite.size:
            if vmin is None:
                vmin = float(np.percentile(finite, 2))
            if vmax is None:
                vmax = float(np.percentile(finite, 98))
        else:
            vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-9
    return np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)


def _apply_cmap(norm: np.ndarray, cmap: str) -> np.ndarray:
    import matplotlib.cm as cm
    try:
        rgba = cm.get_cmap(cmap)(norm)
    except Exception:
        rgba = cm.get_cmap(_VIRIDIS)(norm)
    return (rgba[..., :3] * 255).astype(np.uint8)


def _downsample_to(arr: np.ndarray, max_px: int) -> tuple[np.ndarray, int]:
    """Return downsampled array + stride (1 if untouched)."""
    h, w = arr.shape[:2]
    if max(h, w) <= max_px:
        return arr, 1
    step = max(h // max_px, w // max_px, 1)
    return arr[::step, ::step], step


def _get_font(size: int):
    from PIL import ImageFont
    for name in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _pick_scalebar_nm(image_width_px: int, sampling_A_per_px: float) -> float | None:
    """Pick a scale bar length (in nm) that occupies ~15% of the image width."""
    if not sampling_A_per_px or sampling_A_per_px <= 0:
        return None
    img_width_nm = image_width_px * sampling_A_per_px / 10.0  # A -> nm
    target_nm = img_width_nm * 0.15
    best = min(_SCALEBAR_NICE_NM, key=lambda v: abs(v - target_nm))
    # Reject if bar would be >50% or <5% of image width
    frac = best / img_width_nm
    if 0.05 <= frac <= 0.5:
        return best
    return None


def _draw_scale_bar(draw, panel_x: int, panel_y: int, panel_w: int, panel_h: int,
                    sampling_A_per_px: float, native_w: int) -> None:
    """Draw a white scale bar with label in bottom-right of panel."""
    if not sampling_A_per_px or sampling_A_per_px <= 0:
        return
    # Sampling refers to native pixels; after downsample stride k, sampling_per_displayed_px = sampling*k
    displayed_sampling = sampling_A_per_px * native_w / panel_w
    bar_nm = _pick_scalebar_nm(panel_w, displayed_sampling)
    if bar_nm is None:
        return
    bar_px = int(round(bar_nm * 10.0 / displayed_sampling))
    bar_x1 = panel_x + panel_w - 8
    bar_x0 = bar_x1 - bar_px
    bar_y1 = panel_y + panel_h - 8
    bar_y0 = bar_y1 - _SCALEBAR_H
    if bar_x0 < panel_x + 4:
        return
    # White rectangle with thin black outline
    draw.rectangle([bar_x0 - 1, bar_y0 - 1, bar_x1 + 1, bar_y1 + 1], fill=(0, 0, 0))
    draw.rectangle([bar_x0, bar_y0, bar_x1, bar_y1], fill=(255, 255, 255))
    # Label
    text = f"{int(bar_nm) if bar_nm >= 1 else bar_nm} nm"
    font = _get_font(11)
    tw = draw.textlength(text, font=font) if hasattr(draw, "textlength") else len(text) * 6
    tx = bar_x0 + (bar_px - tw) // 2
    ty = bar_y0 - 16
    # Black shadow + white text
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        draw.text((tx + dx, ty + dy), text, font=font, fill=(0, 0, 0))
    draw.text((tx, ty), text, font=font, fill=(255, 255, 255))


def render_panels_png(panels: Sequence[np.ndarray], *,
                      cmaps: Sequence[str] | str = "viridis",
                      ncols: int = 3,
                      max_px_per_panel: int = 256,
                      vmin: float | None = None,
                      vmax: float | None = None,
                      log: bool = False,
                      labels: Sequence[str] | None = None,
                      title: str | None = None,
                      sampling_A_per_px: float | None = None,
                      pad_px: int = 8,
                      pad_value: int = 255,
                      upscale_to_max: bool = True) -> bytes:
    """Tile N panels into a single PNG with title + per-panel labels + scale bar.

    Mirrors the widget canvas: cmap LUT, log/contrast, scale bar (when sampling
    provided), per-panel label, optional title above grid. Panels rescaled to
    a common pixel size so disparate shapes don't swim in whitespace.
    """
    if not panels:
        return b""
    if isinstance(cmaps, str):
        cmaps = [cmaps] * len(panels)
    if labels is None:
        labels = [""] * len(panels)
    elif len(labels) < len(panels):
        labels = list(labels) + [""] * (len(panels) - len(labels))
    # Pass 1: normalize + cmap (record native widths for scale-bar math)
    rgbs: list[np.ndarray] = []
    native_widths: list[int] = []
    max_h = max_w = 0
    for img, cm_name in zip(panels, cmaps):
        native_w = img.shape[1]
        norm = _normalize(img, vmin, vmax, log)
        norm, _ = _downsample_to(norm, max_px_per_panel)
        rgb = _apply_cmap(norm, cm_name)
        rgbs.append(rgb)
        native_widths.append(native_w)
        max_h = max(max_h, rgb.shape[0])
        max_w = max(max_w, rgb.shape[1])
    # Pass 2: upscale tiny panels to match the largest pair of dims (preserve aspect)
    if upscale_to_max:
        from PIL import Image as _PILImage
        target_max = max(max_h, max_w)
        scaled: list[np.ndarray] = []
        for rgb in rgbs:
            h, w = rgb.shape[:2]
            scale = target_max / max(h, w)
            new_h = max(1, int(round(h * scale)))
            new_w = max(1, int(round(w * scale)))
            if new_h == h and new_w == w:
                scaled.append(rgb)
            else:
                im = _PILImage.fromarray(rgb).resize((new_w, new_h),
                                                     _PILImage.NEAREST)
                scaled.append(np.asarray(im))
        rgbs = scaled
        max_h = max(r.shape[0] for r in rgbs)
        max_w = max(r.shape[1] for r in rgbs)
    # Compose canvas with reserved bands for title + per-panel labels
    n = len(rgbs)
    nrows = (n + ncols - 1) // ncols
    title_band = _TITLE_PAD if title else 0
    has_labels = any(bool(s) for s in labels)
    label_band = _LABEL_PAD if has_labels else 0
    cell_h = max_h + label_band
    canvas_h = title_band + nrows * cell_h + (nrows + 1) * pad_px
    canvas_w = ncols * max_w + (ncols + 1) * pad_px
    canvas = np.full((canvas_h, canvas_w, 3), pad_value, dtype=np.uint8)
    for i, rgb in enumerate(rgbs):
        r, c = divmod(i, ncols)
        y0 = title_band + pad_px + r * (cell_h + pad_px) + label_band + (max_h - rgb.shape[0]) // 2
        x0 = pad_px + c * (max_w + pad_px) + (max_w - rgb.shape[1]) // 2
        h, w = rgb.shape[:2]
        canvas[y0:y0 + h, x0:x0 + w] = rgb
    # Draw overlays
    from PIL import Image, ImageDraw
    pim = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pim)
    if title:
        font = _get_font(14)
        tw = draw.textlength(title, font=font) if hasattr(draw, "textlength") else len(title) * 7
        draw.text(((canvas_w - tw) // 2, 4), title, font=font, fill=(20, 20, 20))
    if has_labels:
        font = _get_font(11)
        for i, lbl in enumerate(labels):
            if not lbl:
                continue
            r, c = divmod(i, ncols)
            y_label = title_band + pad_px + r * (cell_h + pad_px) + 4
            x_cell = pad_px + c * (max_w + pad_px)
            tw = draw.textlength(lbl, font=font) if hasattr(draw, "textlength") else len(lbl) * 6
            draw.text((x_cell + (max_w - tw) // 2, y_label), lbl, font=font, fill=(20, 20, 20))
    if sampling_A_per_px:
        # Scale bar on each panel (panel native_widths drive sampling per displayed px)
        for i, (rgb, native_w) in enumerate(zip(rgbs, native_widths)):
            r, c = divmod(i, ncols)
            y0 = title_band + pad_px + r * (cell_h + pad_px) + label_band + (max_h - rgb.shape[0]) // 2
            x0 = pad_px + c * (max_w + pad_px) + (max_w - rgb.shape[1]) // 2
            _draw_scale_bar(draw, x0, y0, rgb.shape[1], rgb.shape[0],
                            sampling_A_per_px, native_w)
    buf = io.BytesIO()
    pim.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def render_image_png(img2d: np.ndarray, *,
                     cmap: str = "viridis",
                     vmin: float | None = None,
                     vmax: float | None = None,
                     log: bool = False,
                     max_px: int = 512,
                     label: str | None = None,
                     title: str | None = None,
                     sampling_A_per_px: float | None = None) -> bytes:
    """Single-panel convenience wrapper around render_panels_png."""
    return render_panels_png(
        [img2d], cmaps=cmap, ncols=1, max_px_per_panel=max_px,
        vmin=vmin, vmax=vmax, log=log,
        labels=[label] if label else None,
        title=title, sampling_A_per_px=sampling_A_per_px,
    )
