"""Show2D exports: a hybrid SVG figure and the standalone HTML viewer.

``export_svg`` keeps figure chrome editable (frames, titles, markers, scale
bar, annotations, overlays, insets) and embeds the measured pixels as PNG
panels. The HTML export clones the widget into an offline model and embeds it
with the live JS bundle so the file opens without a kernel.
"""

import base64
import html
import io
import math
import pathlib
import re
import textwrap
from collections.abc import Callable, Mapping, Sequence
from typing import Self

import matplotlib
import numpy as np
from PIL import Image

from quantem.widget.export import export_slug
from quantem.widget.utils.array import bin2d

_SVG_FONT = "-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
_LATEX_SYMBOLS = {
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\lambda": "λ", r"\mu": "μ", r"\sigma": "σ", r"\chi": "χ",
    r"\omega": "ω", r"\Delta": "Δ", r"\Theta": "Θ", r"\pm": "±",
    r"\times": "×", r"\cdot": "·", r"\degree": "°", r"\angstrom": "Å",
    r"\le": "≤", r"\ge": "≥", r"\neq": "≠", r"\approx": "≈",
    r"\infty": "∞",
}
_SUPERSCRIPT = str.maketrans({
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
    "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾",
    "n": "ⁿ", "i": "ⁱ",
})
_SUBSCRIPT = str.maketrans({
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
    "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
    "+": "₊", "-": "₋", "=": "₌", "(": "₍", ")": "₎",
    "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ",
    "k": "ₖ", "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ",
    "p": "ₚ", "r": "ᵣ", "s": "ₛ", "t": "ₜ", "u": "ᵤ",
    "v": "ᵥ", "x": "ₓ",
})


def _esc_text(value: object) -> str:
    """``value`` as SVG element text: ``<`` and ``&`` escaped so a label cannot open a tag."""
    return html.escape(str(value), quote=False)


def _esc_attr(value: object) -> str:
    """``value`` for a double-quoted SVG attribute: quotes escaped too, so a label cannot end the attribute."""
    return html.escape(str(value), quote=True)


def _svg_color(value: object, fallback: str = "") -> str:
    """CSS color for SVG: ``rgba(...)`` collapses to ``rgb(...)`` because SVG 1.1
    renderers (Illustrator) drop the whole attribute on an alpha component."""
    text = str(value if value not in (None, "") else fallback).strip()
    match = re.fullmatch(
        r"rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(?:0|1|0?\.\d+)\s*\)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return f"rgb({int(match.group(1))}, {int(match.group(2))}, {int(match.group(3))})"
    return text


def _math_text(value: object) -> str:
    """Plain Unicode for a LaTeX-ish label (``$\\lambda_{3}^2$`` -> ``λ₃²``):
    SVG text has no math mode, so symbols and sub/superscripts are translated."""
    text = str(value or "").strip().strip("$")
    while "\\\\" in text:
        text = text.replace("\\\\", "\\")
    for key, symbol in _LATEX_SYMBOLS.items():
        text = text.replace(key, symbol)

    def read_group(start: int) -> tuple[str, int]:
        """The ``{...}`` group (or single character) at ``start`` and the index after it; nested braces count."""
        if start >= len(text) or text[start] != "{":
            return (text[start] if start < len(text) else ""), start + 1
        depth = 0
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[start + 1:index], index + 1
        return text[start + 1:], len(text)

    out: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char in {"^", "_"} and index + 1 < len(text):
            group, next_index = read_group(index + 1)
            out.append(group.translate(_SUPERSCRIPT if char == "^" else _SUBSCRIPT))
            index = next_index
            continue
        if char in {"{", "}"}:
            index += 1
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _span_text(span: Mapping[str, object]) -> str:
    """Plain text of one rich span; a ``math`` span is translated to Unicode since SVG text has no math mode."""
    if span.get("math") not in (None, ""):
        return _math_text(span.get("math", ""))
    return str(span.get("text", ""))


def _overlay_svg(
    spec: Mapping[str, object],
    x: float,
    y: float,
    panel_w: float,
    panel_h: float,
    view: tuple[float, float, float, float],
) -> str:
    """One shape overlay (circle or rect) placed in panel coordinates.

    ``coords="data"`` specs are in image pixels and must be mapped through the
    visible window ``view = (row0, row1, col0, col1)``; ``relative`` specs are
    fractions of the panel.
    """
    row0, row1, col0, col1 = view
    shape = str(spec.get("shape", "circle")).lower()
    if shape == "rectangle":
        shape = "rect"
    coords = str(spec.get("coords", "data")).lower()
    view_h = max(1.0, row1 - row0)
    view_w = max(1.0, col1 - col0)

    def to_x(col: float) -> float:
        """SVG x of an image column (or a panel-width fraction for relative specs)."""
        return x + col * panel_w if coords == "relative" else x + (col - col0) / view_w * panel_w

    def to_y(row: float) -> float:
        """SVG y of an image row (or a panel-height fraction for relative specs)."""
        return y + row * panel_h if coords == "relative" else y + (row - row0) / view_h * panel_h

    def to_radius(radius: float) -> float:
        """SVG radius of an image-pixel radius (the longer visible axis sets the scale), or of a fraction of the shorter panel side."""
        if coords == "relative":
            return radius * min(panel_w, panel_h)
        return radius / max(view_w, view_h) * max(panel_w, panel_h)

    stroke = _svg_color(spec.get("stroke"), "#00e5ff")
    stroke_width = max(0.0, float(spec.get("stroke_width", 2.0)))
    # Dash pattern: an explicit "dash" list wins, else a named line_style.
    raw_dash = spec.get("dash")
    dash_values = (
        [float(step) for step in raw_dash if float(step) >= 0]
        if isinstance(raw_dash, Sequence) and not isinstance(raw_dash, (str, bytes, bytearray))
        else []
    )
    if not any(dash_values):
        unit = max(1.0, stroke_width)
        dash_values = {
            "dashed": [4 * unit, 2 * unit], "dash": [4 * unit, 2 * unit],
            "dotted": [unit, 1.8 * unit], "dot": [unit, 1.8 * unit],
            "dashdot": [4 * unit, 2 * unit, unit, 2 * unit], "dash-dot": [4 * unit, 2 * unit, unit, 2 * unit],
        }.get(str(spec.get("line_style", "solid")).lower().replace("_", "-"), [])
    dash_attr = (
        f' stroke-dasharray="{_esc_attr(" ".join(f"{step:g}" for step in dash_values))}" stroke-linecap="round"'
        if dash_values else ""
    )
    opacity = max(0.0, min(1.0, float(spec.get("opacity", 1.0))))
    stroke_opacity = opacity * max(0.0, min(1.0, float(spec.get("stroke_opacity", 1.0))))
    fill_value = spec.get("fill", "none")
    fill = "none" if fill_value in (None, "", "none", "None") else _svg_color(fill_value)
    fill_opacity = opacity * max(0.0, min(1.0, float(spec.get("fill_opacity", 1.0 if fill != "none" else 0.0))))
    common = (
        f' fill="{_esc_attr(fill)}" fill-opacity="{fill_opacity:g}"'
        f' stroke="{_esc_attr(stroke)}" stroke-width="{stroke_width:g}"'
        f' stroke-opacity="{stroke_opacity:g}"{dash_attr}'
    )
    if shape == "circle":
        center_x = to_x(float(spec.get("col", 0.0)))
        center_y = to_y(float(spec.get("row", 0.0)))
        radius = to_radius(max(0.0, float(spec.get("radius", 0.0))))
        return f'<circle cx="{center_x:g}" cy="{center_y:g}" r="{radius:g}"{common}/>'
    spec_row0 = float(spec.get("row0", 0.0))
    spec_col0 = float(spec.get("col0", 0.0))
    spec_row1 = float(spec.get("row1", spec_row0))
    spec_col1 = float(spec.get("col1", spec_col0))
    left = to_x(min(spec_col0, spec_col1))
    right = to_x(max(spec_col0, spec_col1))
    top = to_y(min(spec_row0, spec_row1))
    bottom = to_y(max(spec_row0, spec_row1))
    return f'<rect x="{left:g}" y="{top:g}" width="{max(0, right - left):g}" height="{max(0, bottom - top):g}"{common}/>'


def _annotation_anchor(position: str) -> tuple[float, float, str, str]:
    """Relative (x, y) plus SVG text-anchor / baseline for a corner keyword."""
    if "left" in position:
        frac_x, anchor = 0.0, "start"
    elif "right" in position:
        frac_x, anchor = 1.0, "end"
    else:
        frac_x, anchor = 0.5, "middle"
    if "top" in position:
        frac_y, baseline = 0.0, "hanging"
    elif "bottom" in position:
        frac_y, baseline = 1.0, "baseline"
    else:
        frac_y, baseline = 0.5, "middle"
    return frac_x, frac_y, anchor, baseline


def _annotation_svg(spec: Mapping[str, object], x: float, y: float, panel_w: float, panel_h: float) -> str:
    """One text annotation (badge, outline, callout or plain) inside a panel."""
    position = str(spec.get("position", "top-left"))
    font_size = max(6.0, float(spec.get("font_size", 10.0)))
    font_family = str(spec.get("font_family", _SVG_FONT))
    variant = str(spec.get("variant", "badge"))
    pad_x = max(0.0, float(spec.get("pad_x", 6.0 if variant != "plain" else 0.0)))
    pad_y = max(0.0, float(spec.get("pad_y", 2.0 if variant != "plain" else 0.0)))
    opacity = max(0.0, min(1.0, float(spec.get("opacity", 1.0))))
    offset = spec.get("offset", (0.0, 0.0))
    offset_x, offset_y = (float(offset[0]), float(offset[1])) if isinstance(offset, Sequence) and len(offset) >= 2 else (0.0, 0.0)
    align_anchor = {
        "left": "start", "start": "start", "center": "middle",
        "middle": "middle", "right": "end", "end": "end",
    }.get(str(spec.get("align", "")).lower())
    if "box" in spec:
        left, top, width, height = (float(value) for value in spec["box"])
        if align_anchor == "start":
            anchor_x = x + left * panel_w + pad_x + offset_x
        elif align_anchor == "end":
            anchor_x = x + (left + width) * panel_w - pad_x + offset_x
        else:
            anchor_x = x + (left + width / 2.0) * panel_w + offset_x
        anchor_y = y + (top + height / 2.0) * panel_h + offset_y
        anchor = align_anchor or "middle"
        baseline = "middle"
    elif "x" in spec and "y" in spec:
        anchor_x = x + float(spec.get("x", 0.0)) * panel_w + offset_x
        anchor_y = y + float(spec.get("y", 0.0)) * panel_h + offset_y
        _, _, anchor, baseline = _annotation_anchor(str(spec.get("anchor", "center")))
        anchor = align_anchor or anchor
    else:
        frac_x, frac_y, anchor, baseline = _annotation_anchor(position)
        anchor = align_anchor or anchor
        margin = 10.0
        anchor_x = x + margin + frac_x * (panel_w - 2 * margin) + offset_x
        anchor_y = y + margin + frac_y * (panel_h - 2 * margin) + offset_y
    spans = spec.get("spans")
    text = _math_text(spec.get("math")) if spec.get("math") not in (None, "") else str(spec.get("text", ""))
    text_len = max(len(text), sum(len(_span_text(span)) for span in spans) if isinstance(spans, Sequence) else 0)
    box_w = text_len * font_size * 0.62 + 2 * pad_x
    box_h = font_size * 1.25 + 2 * pad_y
    fg = _svg_color(spec.get("fg", spec.get("color", "#fff")))
    outline_width = max(0.0, float(spec.get("outline_width", 0.0)))
    outline_color = _svg_color(spec.get("outline_color"), "rgba(0,0,0,0.85)")
    parts = [f'<g opacity="{opacity:g}">']
    if variant != "plain":
        bg = _svg_color(spec.get("bg"), "rgba(0,0,0,0.72)")
        border_color = _svg_color(spec.get("border_color"), "rgba(255,255,255,0.5)")
        border_width = max(0.0, float(spec.get("border_width", 1.0 if variant in {"outline", "callout"} else 0.0)))
        rect_x = anchor_x - box_w / 2 if anchor == "middle" else anchor_x - box_w if anchor == "end" else anchor_x
        rect_y = anchor_y - box_h / 2 if baseline == "middle" else anchor_y - font_size - pad_y if baseline == "baseline" else anchor_y
        parts.append(
            f'<rect x="{rect_x:g}" y="{rect_y:g}" width="{box_w:g}" height="{box_h:g}" '
            f'rx="{float(spec.get("radius", 3.0)):g}" fill="{_esc_attr(bg)}" '
            f'stroke="{_esc_attr(border_color)}" stroke-width="{border_width:g}"/>'
        )
    text_y = anchor_y + (font_size * 0.4 if baseline == "middle" else 0.0)
    text_attrs = (
        f'x="{anchor_x:g}" y="{text_y:g}" text-anchor="{anchor}" '
        f'font-family="{_esc_attr(font_family)}" '
        f'font-size="{font_size:g}" font-weight="{_esc_attr(spec.get("font_weight", 700))}"'
    )
    fill_parts = []
    if isinstance(spans, Sequence) and not isinstance(spans, (str, bytes, bytearray)):
        for span in spans:
            if not isinstance(span, Mapping):
                continue
            color_attr = f' fill="{_esc_attr(_svg_color(span["color"]))}"' if span.get("color") else ""
            fill_parts.append(f'<tspan{color_attr}>{_esc_text(_span_text(span))}</tspan>')
    else:
        fill_parts.append(_esc_text(text))
    if outline_width > 0:
        plain = _esc_text(text if text else "".join(_span_text(span) for span in spans or [] if isinstance(span, Mapping)))
        parts.append(
            f'<text {text_attrs} fill="none" stroke="{_esc_attr(outline_color)}" '
            f'stroke-width="{outline_width:g}" stroke-linejoin="round">{plain}</text>'
        )
    parts.append(f'<text {text_attrs} fill="{_esc_attr(fg)}">{"".join(fill_parts)}</text></g>')
    return "".join(parts)


def _inset_svg(
    spec: Mapping[str, object],
    x: float,
    y: float,
    panel_w: float,
    panel_h: float,
    fallback_color: str,
    *,
    scale_bar_visible: bool,
) -> str:
    """One inset curve (e.g. a calibration plot) drawn as a vector polyline."""
    if not spec:
        return ""
    x_values = np.asarray(spec.get("x", []), dtype=float).ravel()
    y_values = np.asarray(spec.get("y", []), dtype=float).ravel()
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    if x_values.size != y_values.size or finite.sum() < 2:
        return ""
    x_values = x_values[finite]
    y_values = y_values[finite]
    xlim = tuple(float(value) for value in spec.get("xlim", (float(x_values.min()), float(x_values.max()))))
    ylim = tuple(float(value) for value in spec.get("ylim", (float(y_values.min()), float(y_values.max()))))
    if xlim[1] <= xlim[0]:
        xlim = (xlim[0] - 0.5, xlim[0] + 0.5)
    if ylim[1] <= ylim[0]:
        ylim = (ylim[0] - 0.5, ylim[0] + 0.5)
    size = max(0.18, min(0.62, float(spec.get("size", 0.31))))
    box_w = max(78.0, min(panel_w * 0.62, panel_w * size))
    box_h = max(50.0, min(panel_h * 0.55, panel_w * float(spec.get("height", size * 0.68))))
    if "box" in spec:
        left, top, width, height = (float(value) for value in spec["box"])
        box_w = max(48.0, min(panel_w, panel_w * width))
        box_h = max(34.0, min(panel_h, panel_h * height))
        box_x = x + max(0.0, min(panel_w - box_w, panel_w * left))
        box_y = y + max(0.0, min(panel_h - box_h, panel_h * top))
    else:
        raw_margin = spec.get("margin", (12.0, 12.0))
        if isinstance(raw_margin, (int, float)):
            margin_x = margin_y = float(raw_margin)
        else:
            margin_x, margin_y = (float(value) for value in list(raw_margin)[:2])
        position = str(spec.get("position", "bottom-right"))
        box_x = x + (panel_w - box_w - margin_x if "right" in position else panel_w / 2 - box_w / 2 if "center" in position else margin_x)
        # A bottom-right inset lifts above the scale bar so the two never overlap.
        box_y = y + (panel_h - box_h - margin_y - (34.0 if scale_bar_visible and position == "bottom-right" else 0.0) if "bottom" in position else panel_h / 2 - box_h / 2 if "center" in position else margin_y + 18.0)
    show_ticks = bool(spec.get("show_ticks", False))
    tick_font = max(5.0, min(14.0, float(spec.get("tick_font_size", 7.0))))
    label_font = max(6.0, min(16.0, float(spec.get("label_font_size", 8.0))))
    legend_font = max(6.0, min(18.0, float(spec.get("legend_font_size", 9.0))))
    pad_left = max(22.0, tick_font * 3.2) if show_ticks or spec.get("ylabel") else 10.0
    pad_right = 7.0
    pad_top = max(13.0, legend_font + 6.0) if spec.get("title") or spec.get("legend") else 7.0
    pad_bottom = max(16.0, tick_font + label_font + 4.0) if show_ticks or spec.get("xlabel") else 8.0
    plot_x0 = box_x + pad_left
    plot_y0 = box_y + pad_top
    plot_w = box_w - pad_left - pad_right
    plot_h = box_h - pad_top - pad_bottom
    if plot_w <= 8 or plot_h <= 8:
        return ""

    def to_svg_x(value: float) -> float:
        """SVG x of a data x value inside the inset's plot area."""
        return plot_x0 + (float(value) - xlim[0]) / (xlim[1] - xlim[0]) * plot_w

    def to_svg_y(value: float) -> float:
        """SVG y of a data y value; SVG y grows downward, so larger values sit higher."""
        return plot_y0 + plot_h - (float(value) - ylim[0]) / (ylim[1] - ylim[0]) * plot_h

    points = " ".join(f"{to_svg_x(x_value):g},{to_svg_y(y_value):g}" for x_value, y_value in zip(x_values, y_values))
    line_color = _svg_color(spec.get("color"), fallback_color)
    text_color = _svg_color(spec.get("text_color"), "rgba(255,255,255,0.92)")
    tick_color = _svg_color(spec.get("tick_color"), "rgba(255,255,255,0.72)")
    parts = [
        "<g>",
        f'<rect x="{box_x:g}" y="{box_y:g}" width="{box_w:g}" height="{box_h:g}" '
        f'fill="{_esc_attr(_svg_color(spec.get("background"), "#0a0c10"))}" fill-opacity="{max(0.0, min(1.0, float(spec.get("background_alpha", 0.68)))):g}" '
        f'stroke="{_esc_attr(_svg_color(spec.get("border_color"), "rgba(255,255,255,0.34)"))}" stroke-width="{float(spec.get("border_width", 1.0)):g}"/>',
        f'<path d="M {plot_x0:g} {plot_y0:g} V {plot_y0 + plot_h:g} H {plot_x0 + plot_w:g}" fill="none" stroke="{_esc_attr(tick_color)}" stroke-opacity="0.45" stroke-width="1"/>',
        f'<polyline points="{points}" fill="none" stroke="{_esc_attr(line_color)}" stroke-width="{max(1.4, float(spec.get("line_width", 2.0))):g}" stroke-linejoin="round" stroke-linecap="round"/>',
    ]
    if "point" in spec:
        point = np.asarray(spec["point"], dtype=float).ravel()
        if point.size == 2 and np.isfinite(point).all():
            parts.append(f'<circle cx="{to_svg_x(point[0]):g}" cy="{to_svg_y(point[1]):g}" r="3.4" fill="{_esc_attr(_svg_color(spec.get("point_color"), "#fff"))}" stroke="#000" stroke-width="1.5"/>')
    if spec.get("title"):
        parts.append(f'<text x="{box_x + 6:g}" y="{box_y + 12:g}" font-family="{_SVG_FONT}" font-size="{legend_font:g}" font-weight="700" fill="{_esc_attr(text_color)}">{_esc_text(spec["title"])}</text>')
    if spec.get("legend"):
        parts.append(f'<text x="{box_x + 6:g}" y="{box_y + box_h - 6:g}" font-family="{_SVG_FONT}" font-size="{legend_font:g}" font-weight="700" fill="{_esc_attr(line_color)}">{_esc_text(spec["legend"])}</text>')
    if spec.get("xlabel"):
        parts.append(f'<text x="{box_x + box_w - 7:g}" y="{box_y + box_h - 3:g}" text-anchor="end" font-family="{_SVG_FONT}" font-size="{label_font:g}" fill="{_esc_attr(tick_color)}">{_esc_text(spec["xlabel"])}</text>')
    if spec.get("ylabel"):
        parts.append(f'<text x="{box_x + 5:g}" y="{plot_y0 + 2:g}" transform="rotate(-90 {box_x + 5:g} {plot_y0 + 2:g})" text-anchor="end" font-family="{_SVG_FONT}" font-size="{label_font:g}" fill="{_esc_attr(tick_color)}">{_esc_text(spec["ylabel"])}</text>')
    parts.append("</g>")
    return "".join(parts)


def _colorbar_svg(
    panel: Mapping[str, object],
    x: float,
    y: float,
    panel_w: float,
    panel_h: float,
    panel_index: int,
    format_stat: Callable[[float], str],
) -> tuple[str, str]:
    """Vertical gradient colorbar for one panel: ``(gradient def, body)``.

    Nine sampled stops keep the gradient an editable vector element; the end
    labels use the stats-row number format so the figure reads like the widget.
    """
    gradient_id = f"show2d-svg-colorbar-{panel_index}"
    colormap = matplotlib.colormaps.get_cmap(str(panel["cmap"]))
    stops = []
    for step in range(9):
        fraction = step / 8.0
        red, green, blue, _alpha = colormap(fraction)
        stops.append(
            f'<stop offset="{fraction * 100:g}%" stop-color="rgb({int(red * 255)}, {int(green * 255)}, {int(blue * 255)})"/>'
        )
    bar_h = min(160.0, panel_h * 0.62)
    bar_w = 10.0
    bar_x = x + panel_w - 22.0
    bar_y = y + 18.0
    body = [
        "<g>",
        f'<rect x="{bar_x - 1:g}" y="{bar_y - 1:g}" width="{bar_w + 2:g}" height="{bar_h + 2:g}" fill="#000" fill-opacity="0.45"/>',
        f'<rect x="{bar_x:g}" y="{bar_y:g}" width="{bar_w:g}" height="{bar_h:g}" fill="url(#{gradient_id})" stroke="#fff" stroke-opacity="0.75" stroke-width="0.75"/>',
        f'<text x="{bar_x - 4:g}" y="{bar_y + 4:g}" text-anchor="end" font-family="{_SVG_FONT}" font-size="9" fill="#fff">{_esc_text(format_stat(float(panel["vmax"])))}</text>',
        f'<text x="{bar_x - 4:g}" y="{bar_y + bar_h:g}" text-anchor="end" font-family="{_SVG_FONT}" font-size="9" fill="#fff">{_esc_text(format_stat(float(panel["vmin"])))}</text>',
        "</g>",
    ]
    return (
        f'<linearGradient id="{gradient_id}" x1="0" x2="0" y1="1" y2="0">{"".join(stops)}</linearGradient>',
        "".join(body),
    )


def _title_position(
    title_style: Mapping[str, object],
    x: float,
    y: float,
    panel_w: float,
    panel_h: float,
    font_size: float,
) -> tuple[float, float, str]:
    """Baseline (x, y) and text-anchor of a panel title from ``panel_title_style``."""
    raw_offset = title_style.get("offset", (0.0, 0.0))
    if isinstance(raw_offset, Sequence) and not isinstance(raw_offset, (str, bytes, bytearray)):
        offset_values = list(raw_offset)
        offset_x = float(offset_values[0]) if offset_values else 0.0
        offset_y = float(offset_values[1]) if len(offset_values) > 1 else 0.0
    else:
        offset_x = offset_y = 0.0
    if "x" in title_style or "y" in title_style:
        anchor_value = str(title_style.get("anchor", "top-center")).lower()
        base_y = y + float(title_style.get("y", 0.0)) * panel_h + offset_y
        if "bottom" in anchor_value:
            title_y = base_y
        elif "center" in anchor_value:
            title_y = base_y + font_size * 0.35
        else:
            title_y = base_y + font_size
        anchor = "end" if "right" in anchor_value else "middle" if "center" in anchor_value else "start"
        return x + float(title_style.get("x", 0.5)) * panel_w + offset_x, title_y, anchor
    align = str(title_style.get("align", "center")).lower()
    if align in {"left", "start"}:
        return x + 28.0, y + 6.0 + font_size, "start"
    if align in {"right", "end"}:
        return x + panel_w - 28.0, y + 6.0 + font_size, "end"
    return x + panel_w / 2.0, y + 6.0 + font_size, "middle"


def _panel_title_svg(
    widget,
    panel_index: int,
    label: str,
    x: float,
    y: float,
    panel_w: float,
    panel_h: float,
) -> list[str]:
    """Panel title as editable text: a shadow or outline copy under the fill
    so the label stays readable over bright and dark pixels alike."""
    title_style = dict(widget.panel_title_style or {})
    font_family = str(title_style.get("font_family", _SVG_FONT))
    fg = _svg_color(title_style.get("fg"), "#fff")
    opacity = max(0.0, min(1.0, float(title_style.get("opacity", 0.95))))
    font_weight = title_style.get("font_weight", 700)
    outline_width = max(0.0, float(title_style.get("outline_width", 0.0)))
    outline_color = _svg_color(title_style.get("outline_color"), "rgba(0,0,0,0.85)")
    font_size = max(8, int(widget.panel_title_font_size or 11))
    spans = widget.panel_title_spans[panel_index] if panel_index < len(widget.panel_title_spans) else []
    title_x, title_y, anchor = _title_position(title_style, x, y, panel_w, panel_h, font_size)
    common = f'text-anchor="{anchor}" font-family="{_esc_attr(font_family)}" font-size="{font_size}" font-weight="{_esc_attr(font_weight)}"'

    def underlay(line_y: float, text: str) -> str:
        """The dark copy drawn under one title line: a 1 px drop shadow, or a stroke when ``outline_width`` is set."""
        if outline_width <= 0:
            return f'<text x="{title_x + 1:g}" y="{line_y + 1:g}" {common} fill="#000" fill-opacity="0.85">{_esc_text(text)}</text>'
        return (
            f'<text x="{title_x:g}" y="{line_y:g}" {common} fill="none" '
            f'stroke="{_esc_attr(outline_color)}" stroke-width="{outline_width:g}" '
            f'stroke-linejoin="round">{_esc_text(text)}</text>'
        )

    fill_attrs = f'{common} fill="{_esc_attr(fg)}" fill-opacity="{opacity:g}"'
    elements: list[str] = []
    if spans:
        plain = "".join(_span_text(span) for span in spans if isinstance(span, Mapping))
        elements.append(underlay(title_y, plain))
        elements.append(f'<text x="{title_x:g}" y="{title_y:g}" {fill_attrs}>')
        for span in spans:
            if not isinstance(span, Mapping):
                continue
            color_attr = f' fill="{_esc_attr(_svg_color(span["color"]))}"' if span.get("color") else ""
            elements.append(f'<tspan{color_attr}>{_esc_text(_span_text(span))}</tspan>')
        elements.append("</text>")
        return elements
    wrap_width = max(1, int(max(24, panel_w - 56) / max(1, font_size * 0.55)))
    for line_idx, line in enumerate(textwrap.wrap(label.strip(), width=wrap_width, break_long_words=True, max_lines=3)):
        line_y = title_y + line_idx * font_size * 1.2
        elements.append(underlay(line_y, line))
        elements.append(f'<text x="{title_x:g}" y="{line_y:g}" {fill_attrs}>{_esc_text(line)}</text>')
    return elements


def _scale_bar_svg(
    widget,
    bar_text: str,
    bar_px: float,
    x: float,
    y: float,
    panel_w: float,
    panel_h: float,
) -> list[str]:
    """Scale bar rectangle plus label (and the zoom badge) for one panel,
    styled from ``scale_bar_style`` exactly like the browser overlay."""
    scale_style = dict(widget.scale_bar_style or {})
    offset_x, offset_y = (float(value) for value in scale_style.get("offset", (0.0, 0.0)))
    bar_height = float(scale_style.get("bar_height", 5.0))
    label_gap = float(scale_style.get("label_gap", 4.0))
    font_size = float(scale_style.get("font_size", 16.0))
    font_family = str(scale_style.get("font_family", _SVG_FONT))
    font_weight = scale_style.get("font_weight", "")
    color = _svg_color(scale_style.get("color"), "#fff")
    outline_color = _svg_color(scale_style.get("outline_color"), "#000")
    outline_width = float(scale_style.get("outline_width", 0.0))
    shadow_value = scale_style.get("shadow_color")
    shadow_color = _svg_color(shadow_value, "#000")
    scale_left = widget.scale_bar_position == "bottom-left"
    bar_x = x + (12 if scale_left else panel_w - bar_px - 12) + offset_x
    bar_y = y + panel_h - 12 + offset_y
    weight_attr = f' font-weight="{_esc_attr(font_weight)}"' if font_weight != "" else ""
    font_attrs = f'font-family="{_esc_attr(font_family)}" font-size="{font_size:g}"{weight_attr}'
    elements: list[str] = []
    if shadow_value is not None:
        elements.append(
            f'<rect x="{bar_x + 1:g}" y="{bar_y + 1:g}" width="{bar_px:g}" '
            f'height="{bar_height:g}" fill="{_esc_attr(shadow_color)}" fill-opacity="0.5"/>'
        )
    elements.append(
        f'<rect x="{bar_x:g}" y="{bar_y:g}" width="{bar_px:g}" height="{bar_height:g}" fill="{_esc_attr(color)}"/>'
    )
    text_x = bar_x + bar_px / 2
    text_y = bar_y - label_gap
    if outline_width > 0:
        elements.append(
            f'<text x="{text_x:g}" y="{text_y:g}" text-anchor="middle" {font_attrs} '
            f'fill="none" stroke="{_esc_attr(outline_color)}" '
            f'stroke-width="{outline_width:g}" stroke-linejoin="round">{_esc_text(bar_text)}</text>'
        )
    else:
        elements.append(
            f'<text x="{text_x + 1:g}" y="{text_y + 1:g}" text-anchor="middle" {font_attrs} '
            f'fill="{_esc_attr(shadow_color)}" fill-opacity="0.85">{_esc_text(bar_text)}</text>'
        )
    elements.append(
        f'<text x="{text_x:g}" y="{text_y:g}" text-anchor="middle" {font_attrs} '
        f'fill="{_esc_attr(color)}">{_esc_text(bar_text)}</text>'
    )
    if widget.show_zoom_indicator:
        zoom_text = f"{min(max(float(widget.initial_zoom) or 1.0, 0.5), 20.0):.1f}×"
        zoom_x = x + (panel_w - 12 if scale_left else 12)
        anchor = "end" if scale_left else "start"
        elements.extend([
            f'<text x="{zoom_x + 1:g}" y="{y + panel_h - 6:g}" text-anchor="{anchor}" {font_attrs} '
            f'fill="{_esc_attr(shadow_color)}" fill-opacity="0.85">{_esc_text(zoom_text)}</text>',
            f'<text x="{zoom_x:g}" y="{y + panel_h - 7:g}" text-anchor="{anchor}" {font_attrs} '
            f'fill="{_esc_attr(color)}">{_esc_text(zoom_text)}</text>',
        ])
    return elements


def _marker_svg(
    widget,
    n_panels: int,
    ncols: int,
    panel_w: float,
    gap: int,
    border: int,
    marker_top: float,
    row_heights: list[int],
) -> list[str]:
    """Row and column marker frames (``row_markers`` / ``col_markers``).

    ``border`` is the gallery's outer border width and ``marker_top`` the y of
    the first panel row; each marker is a colored 3 px frame with a thin black
    inner line so it reads on both bright and dark panels.
    """
    elements: list[str] = []
    for raw_row, color in dict(widget.row_markers or {}).items():
        color = _svg_color(color)
        row_idx = int(raw_row)
        if row_idx < 0 or row_idx >= len(row_heights):
            continue
        row_y = marker_top + sum(row_heights[:row_idx]) + row_idx * gap
        row_count = min(ncols, max(0, n_panels - row_idx * ncols))
        row_w = row_count * panel_w + max(0, row_count - 1) * gap
        elements.extend([
            f'<rect x="{border:g}" y="{row_y:g}" width="{row_w:g}" height="{row_heights[row_idx]:g}" fill="none" stroke="{_esc_attr(color)}" stroke-width="3"/>',
            f'<rect x="{border + 3:g}" y="{row_y + 3:g}" width="{max(0, row_w - 6):g}" height="{max(0, row_heights[row_idx] - 6):g}" fill="none" stroke="#000" stroke-opacity="0.9" stroke-width="2"/>',
        ])
    for raw_col, color in dict(widget.col_markers or {}).items():
        color = _svg_color(color)
        col_idx = int(raw_col)
        if col_idx < 0 or col_idx >= ncols:
            continue
        slots = [slot for slot in range(n_panels) if slot % ncols == col_idx]
        if not slots:
            continue
        row_min = min(slot // ncols for slot in slots)
        row_max = max(slot // ncols for slot in slots)
        col_x = border + col_idx * (panel_w + gap)
        col_y = marker_top + sum(row_heights[:row_min]) + row_min * gap
        col_h = sum(row_heights[row_min:row_max + 1]) + max(0, row_max - row_min) * gap
        elements.extend([
            f'<rect x="{col_x:g}" y="{col_y:g}" width="{panel_w:g}" height="{col_h:g}" fill="none" stroke="{_esc_attr(color)}" stroke-width="3"/>',
            f'<rect x="{col_x + 3:g}" y="{col_y + 3:g}" width="{max(0, panel_w - 6):g}" height="{max(0, col_h - 6):g}" fill="none" stroke="#000" stroke-opacity="0.9" stroke-width="2"/>',
        ])
    return elements


class Show2DExport:
    """Mixin: SVG figure export and standalone HTML export."""

    _HTML_EXPORT_SAFE_MB = 80.0

    def export_svg(
        self,
        path: str | pathlib.Path | None = None,
        *,
        scale: float = 3,
        include_scale_bar: bool = True,
        include_colorbar: bool = False,
        title: str | None = None,
    ) -> pathlib.Path:
        """Export the current gallery as a hybrid SVG figure.

        Panel frames, marker bars, panel labels, title, annotations, overlays
        and the scale bar stay editable vector elements; the image panels are
        embedded as PNG at ``scale`` times the widget display size, so
        Illustrator or Inkscape gets sharp pixels to place in a manuscript.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            Output SVG path. Defaults to ``<title>_<shape>.svg`` in the
            working directory.
        scale : float, default 3
            Embedded image scale relative to the display panel size (1 to 8).
        include_scale_bar : bool, default True
            Draw the scale bar when scale bars are visible on the widget.
        include_colorbar : bool, default False
            Add an editable colorbar to every panel.
        title : str, optional
            Figure title override. Defaults to the widget title.
        """
        chrome = self._gallery_export_chrome()
        specs = self._static_panel_specs()
        if not specs:
            raise ValueError("Show2D has no visible panels to export")
        export_path = pathlib.Path.cwd() / f"{self._export_stem()}.svg" if path is None else pathlib.Path(path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_scale = max(1.0, min(8.0, float(scale)))
        panel_w = int(round(self._static_canvas_css_px()))
        gap = chrome["inter_panel_gap_px"]
        gap_color = chrome["inter_panel_gap_color"]
        border = chrome["gallery_outer_border_px"]
        border_color = chrome["gallery_outer_border_color"]
        panel_border_px = chrome["panel_inner_border_px"]
        panel_border_color = chrome["panel_inner_border_color"]
        ncols = max(1, min(int(self.ncols), len(specs)))
        title_text = self.title if title is None else str(title)
        title_h = 30 if title_text and self.show_title else 0
        draw_scale = bool(include_scale_bar and self.scale_bar_visible)
        # The visible window: what the browser last synced, else the
        # construction-time zoom about (zoom_row, zoom_col).
        if len(self.view_box) == 4:
            row0, row1, col0, col1 = (float(value) for value in self.view_box)
        else:
            zoom = float(self.initial_zoom) or 1.0
            center_row = float(self.zoom_row) if self.zoom_row is not None else self.height / 2
            center_col = float(self.zoom_col) if self.zoom_col is not None else self.width / 2
            half_h, half_w = self.height / (2 * zoom), self.width / (2 * zoom)
            row0, row1 = max(0.0, center_row - half_h), min(float(self.height), center_row + half_h)
            col0, col1 = max(0.0, center_col - half_w), min(float(self.width), center_col + half_w)
        view = (row0, row1, col0, col1)
        resample = Image.Resampling.BILINEAR if self.smooth else Image.Resampling.NEAREST
        panels: list[dict[str, object]] = []
        overlays = self._static_overlay_texts(specs, css_px=panel_w)
        for spec, (_label, _zoom_text, bar_text, bar_px) in zip(specs, overlays):
            frame_pixels = np.asarray(spec["frame"])
            height, width = frame_pixels.shape[:2]
            crop_row0 = max(0, min(height - 1, int(math.floor(row0))))
            crop_row1 = max(crop_row0 + 1, min(height, int(math.ceil(row1))))
            crop_col0 = max(0, min(width - 1, int(math.floor(col0))))
            crop_col1 = max(crop_col0 + 1, min(width, int(math.ceil(col1))))
            cropped = frame_pixels[crop_row0:crop_row1, crop_col0:crop_col1]
            if spec.get("rgb"):
                # PNG bytes are 8-bit: display-ready [0, 1] color maps straight to 0..255.
                rgb = (np.clip(cropped[..., :3], 0.0, 1.0) * 255).astype(np.uint8)
            else:
                rgb = self._static_panel_rgb(
                    cropped,
                    float(spec["vmin"]),
                    float(spec["vmax"]),
                    str(spec["cmap"]),
                    apply_log=bool(spec.get("apply_log")),
                )
            panel_h = max(1, int(round(panel_w * rgb.shape[0] / max(1, rgb.shape[1]))))
            image = Image.fromarray(rgb, mode="RGB")
            embed_w = max(1, int(round(panel_w * export_scale)))
            embed_h = max(1, int(round(panel_h * export_scale)))
            if image.size != (embed_w, embed_h):
                image = image.resize((embed_w, embed_h), resample=resample)
            png_buffer = io.BytesIO()
            image.save(png_buffer, format="PNG")
            panels.append({
                "panel_index": int(spec.get("panel_index", len(panels))),
                "label": str(spec.get("label", "")),
                "height": panel_h,
                "png": base64.b64encode(png_buffer.getvalue()).decode("ascii"),
                "bar_text": bar_text if draw_scale else "",
                "bar_px": float(bar_px) if draw_scale else 0.0,
                "vmin": float(spec.get("vmin", 0.0)),
                "vmax": float(spec.get("vmax", 1.0)),
                "cmap": str(spec.get("cmap", self.cmap)),
            })

        row_heights = [
            max(int(panel["height"]) for panel in panels[start:start + ncols])
            for start in range(0, len(panels), ncols)
        ]
        svg_w = 2 * border + ncols * panel_w + (ncols - 1) * gap
        svg_h = title_h + 2 * border + sum(row_heights) + max(0, len(row_heights) - 1) * gap
        elements: list[str] = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{svg_w}" '
                f'height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}" '
                f'role="img" aria-label="{_esc_attr(title_text or "Show2D SVG export")}">'
            ),
        ]
        defs: list[str] = []
        if title_h:
            elements.append(
                f'<text x="{svg_w / 2:g}" y="19" text-anchor="middle" '
                f'font-family="{_SVG_FONT}" '
                f'font-size="14" font-weight="700" fill="#111">{_esc_text(title_text)}</text>'
            )
        if border > 0 and border_color:
            elements.append(
                f'<rect x="0" y="{title_h:g}" width="{svg_w:g}" height="{max(0, svg_h - title_h):g}" '
                f'fill="{_esc_attr(_svg_color(border_color))}"/>'
            )
        if gap > 0 and gap_color:
            elements.append(
                f'<rect x="{border:g}" y="{title_h + border:g}" '
                f'width="{max(0, svg_w - 2 * border):g}" height="{max(0, svg_h - title_h - 2 * border):g}" '
                f'fill="{_esc_attr(_svg_color(gap_color))}"/>'
            )
        y = title_h + border
        for row_idx, row_h in enumerate(row_heights):
            for col_idx, panel in enumerate(panels[row_idx * ncols:(row_idx + 1) * ncols]):
                x = border + col_idx * (panel_w + gap)
                panel_h = int(panel["height"])
                panel_index = int(panel["panel_index"])
                clip_id = f"show2d-svg-panel-clip-{panel_index}-{row_idx}-{col_idx}"
                defs.append(
                    f'<clipPath id="{clip_id}"><rect x="{x:g}" y="{y:g}" width="{panel_w:g}" height="{panel_h:g}"/></clipPath>'
                )
                marker_color = (
                    _svg_color(self.marker_colors[panel_index])
                    if panel_index < len(self.marker_colors) and self.marker_colors[panel_index]
                    else ""
                )
                elements.extend([
                    f'<g id="show2d-panel-{panel_index}">',
                    f'<rect x="{x}" y="{y}" width="{panel_w}" height="{panel_h}" fill="#000"/>',
                    (
                        f'<image x="{x}" y="{y}" width="{panel_w}" height="{panel_h}" '
                        f'xlink:href="data:image/png;base64,{panel["png"]}" preserveAspectRatio="none"/>'
                    ),
                ])
                if panel_border_px > 0:
                    elements.append(
                        f'<rect x="{x}" y="{y}" width="{panel_w}" height="{panel_h}" fill="none" '
                        f'stroke="{_esc_attr(_svg_color(panel_border_color, "#d0d0d0"))}" stroke-width="{panel_border_px:g}"/>'
                    )
                if marker_color and str(self.marker_style or "left") == "around":
                    elements.append(
                        f'<rect x="{x + 1.5:g}" y="{y + 1.5:g}" width="{max(0, panel_w - 3):g}" '
                        f'height="{max(0, panel_h - 3):g}" fill="none" '
                        f'stroke="{_esc_attr(marker_color)}" stroke-width="3"/>'
                    )
                elif marker_color:
                    elements.append(
                        f'<rect x="{x}" y="{y}" width="5" height="{panel_h}" fill="{_esc_attr(marker_color)}"/>'
                    )
                if panel["label"]:
                    elements.extend(_panel_title_svg(self, panel_index, str(panel["label"]), x, y, panel_w, panel_h))
                elements.append(f'<g clip-path="url(#{clip_id})">')
                if bool(self.show_inset_plots) and panel_index < len(self.inset_plots):
                    elements.append(_inset_svg(
                        self.inset_plots[panel_index], x, y, panel_w, panel_h, marker_color,
                        scale_bar_visible=bool(self.scale_bar_visible),
                    ))
                if panel_index < len(self.panel_overlays):
                    for overlay_spec in self.panel_overlays[panel_index]:
                        elements.append(_overlay_svg(overlay_spec, x, y, panel_w, panel_h, view))
                if include_colorbar:
                    gradient, colorbar = _colorbar_svg(panel, x, y, panel_w, panel_h, panel_index, self._format_stat)
                    defs.append(gradient)
                    elements.append(colorbar)
                if panel_index < len(self.panel_annotations):
                    for annotation_spec in self.panel_annotations[panel_index]:
                        elements.append(_annotation_svg(annotation_spec, x, y, panel_w, panel_h))
                elements.append("</g>")
                if panel["bar_text"] and float(panel["bar_px"]) > 0:
                    elements.extend(_scale_bar_svg(
                        self, str(panel["bar_text"]), float(panel["bar_px"]), x, y, panel_w, panel_h,
                    ))
                elements.append("</g>")
            y += row_h + (gap if row_idx < len(row_heights) - 1 else 0)
        elements.extend(_marker_svg(self, len(panels), ncols, panel_w, gap, border, title_h + border, row_heights))
        if defs:
            elements.insert(2, f'<defs>{"".join(defs)}</defs>')
        elements.append("</svg>")
        export_path.write_text("\n".join(elements), encoding="utf-8")
        return export_path

    def _estimate_html_export_mb(self, *, quantized: bool, downsample: int) -> float:
        """Rough MB an embedded single-file Show2D HTML would occupy.

        ``export_html`` checks this before writing so an oversized page fails
        fast with the uint8 and downsample alternatives, instead of producing
        a file Chrome cannot open from disk. The pixels are base64 (4/3 of the
        raw bytes) plus about 2 MB of bundle and page.
        """
        has_local_stacks = any(count > 1 for count in self.panel_frame_counts)
        if has_local_stacks:
            n_values = sum(stack.size for stack in self._display_panel_stacks)
        else:
            n_values = self._display_data.size
        n_values //= downsample**2
        bytes_per_value = 1 if quantized else 4
        payload_mb = n_values * bytes_per_value * (4.0 / 3.0) / (1024 * 1024)
        return payload_mb + 2.0

    def export_html(self, path: str | pathlib.Path | None = None,
                    *,
                    title: str | None = None,
                    mode: str = "single",
                    encoding: str = "full",
                    downsample: int | None = None,
                    quantized: bool | None = None,
                    max_mb: float | None = _HTML_EXPORT_SAFE_MB) -> pathlib.Path:
        """Write a standalone HTML viewer for this widget.

        The exported file mounts the live anywidget JS bundle with the current
        widget state (data, labels, cmap, vmin/vmax, log_scale, sampling, ...).
        Opens in any browser without a Jupyter kernel.
        Preferred export options are ``mode="single"``, ``encoding="full"`` or
        ``encoding="uint8"``, and ``downsample=None``. Use ``downsample=2`` /
        ``4`` / ``8`` with ``encoding="uint8"`` for compact visual reports.
        ``quantized`` is kept as a compatibility alias for ``encoding="uint8"``.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            Destination HTML path.
        quantized : bool, optional
            Store the displayed image stack as uint8 with min/max metadata.
            This is smaller and visually equivalent after colormapping. The
            default stores exact float32 display values.
        title : str, optional
            Browser page title. Defaults to widget ``title`` or "Show2D".
        mode : {"single"}, default "single"
            One self-contained file.
        encoding : {"full", "uint8"}, default "full"
            Exact float32 display values, or uint8 per panel.
        downsample : {None, 1, 2, 4, 8}, optional
            Mean-bin factor for the embedded pixels; above 1 needs
            ``encoding="uint8"``. The scale bar calibration follows.
        max_mb : float or None, default 80
            Refuse an estimated file size above this; ``None`` disables the check.

        Returns
        -------
        pathlib.Path
            The written file path.
        """
        self._require_grayscale_gallery("export_html")
        quantized, downsample_factor = self._normalise_html_export_options(
            mode=mode,
            encoding=encoding,
            downsample=downsample,
            quantized=quantized,
        )
        if max_mb is not None:
            estimate_mb = self._estimate_html_export_mb(quantized=quantized, downsample=downsample_factor)
            if estimate_mb > float(max_mb):
                uint8_mb = self._estimate_html_export_mb(quantized=True, downsample=downsample_factor)
                raise ValueError(
                    f"This export would embed about {estimate_mb:.0f} MB into one HTML file, "
                    f"above the {float(max_mb):.0f} MB safe limit (large single-file exports often "
                    f"fail to open under Chrome file://). Options: encoding='uint8' "
                    f"(about {uint8_mb:.0f} MB), downsample=2 or 4 to shrink spatially, or pass "
                    f"max_mb={estimate_mb:.0f} to force this size."
                )
        export_path = pathlib.Path(path) if path is not None else self._default_html_export_path(quantized, downsample=downsample_factor)
        self._write_html_export(export_path, quantized=quantized, title=title, downsample=downsample_factor)
        size_mb = export_path.stat().st_size / (1024 * 1024)
        label = self._export_mode_label(quantized, downsample=downsample_factor)
        self.export_status = f"Exported {export_path.name} ({size_mb:.1f} MB, {label})"
        return export_path

    def _require_grayscale_gallery(self, action: str) -> None:
        """Refuse an HTML export of a gallery with RGB panels.

        The export clone rebuilds from the grayscale luminance stack, so an
        RGB panel would silently come out gray. ``action`` names the entry
        point in the error.
        """
        if any(self.is_rgb):
            raise NotImplementedError(
                f"{action} is not supported when the gallery contains RGB panels; "
                "the export clone rebuilds from the grayscale stack and would drop the color channels."
            )

    def _normalise_html_export_options(
        self,
        *,
        mode: str = "single",
        encoding: str = "full",
        downsample: int | None = None,
        quantized: bool | None = None,
    ) -> tuple[bool, int]:
        """Resolve the export keywords to ``(quantized, downsample)``, rejecting combinations that cannot be honoured.

        ``mode`` also accepts the encoding names (``"exact"``, ``"uint8"``, ...)
        and ``quantized=True`` forces uint8. A downsample above 1 requires
        uint8 because the exact float32 export promises every native pixel.
        """
        mode_name = str(mode or "single").strip().lower().replace("_", "-")
        if mode_name in {"exact", "full"}:
            mode_name = "single"
            encoding = "full"
        elif mode_name in {"quantized", "uint8", "u8"}:
            mode_name = "single"
            encoding = "uint8"
        if mode_name != "single":
            raise ValueError("Show2D HTML export supports mode='single'")
        if downsample in (None, "", 0, "0"):
            downsample_factor = 1
        else:
            if isinstance(downsample, bool):
                raise ValueError("Show2D HTML export downsample must be an integer factor, not bool")
            downsample_factor = int(downsample)
        if downsample_factor < 1:
            raise ValueError(f"Show2D HTML export downsample must be >= 1, got {downsample!r}")
        if downsample_factor not in {1, 2, 4, 8}:
            raise ValueError("Show2D HTML export downsample must be one of 1, 2, 4, or 8")
        encoding_name = str(encoding or "full").strip().lower().replace("_", "-")
        if quantized is True:
            encoding_name = "uint8"
        if encoding_name in {"full", "exact", "float32", "f32"}:
            if downsample_factor != 1:
                raise ValueError("Show2D exact float32 HTML export does not support downsample; use encoding='uint8'")
            return False, downsample_factor
        if encoding_name in {"uint8", "u8", "quantized"}:
            return True, downsample_factor
        raise ValueError(f"unknown Show2D export encoding {encoding!r}; expected 'full' or 'uint8'")

    def _html_export_options(self, payload: dict, mode: str) -> dict:
        """``export_html`` keywords from a toolbar request, which carries ``encoding`` and ``downsample``."""
        quantized, downsample = self._normalise_html_export_options(
            mode=mode,
            encoding=str(payload.get("encoding", "full")),
            downsample=payload.get("downsample"),
        )
        return {"quantized": quantized, "downsample": downsample}

    def _export_stem(self) -> str:
        """``<title>_<shape>`` file-name stem shared by the SVG and HTML exports, so one figure's files sort together."""
        shape = f"{self.n_images}x{self.height}x{self.width}" if self.n_images > 1 else f"{self.height}x{self.width}"
        return f"{export_slug(self.title, 'show2d')}_{shape}"

    def _default_html_export_path(self, quantized: bool, *, downsample: int = 1) -> pathlib.Path:
        """``<stem>_<exact|quantized>[_Nxdownsample].html`` in the working directory, naming how the pixels were packed."""
        encoding_label = "quantized" if quantized else "exact"
        suffix = f"_{downsample}xdownsample" if quantized and downsample > 1 else ""
        return pathlib.Path.cwd() / f"{self._export_stem()}_{encoding_label}{suffix}.html"

    def _clone_for_html_export(self, *, quantized: bool, downsample: int = 1) -> Self:
        """An export-only Show2D: the displayed pixels mean-binned by ``downsample``, every view setting, no export menu.

        The standalone page has no kernel behind it, so the clone carries its
        own copy of what is on screen, paints on a light background, and with
        ``quantized`` packs the pixels as uint8 per panel.
        """
        self._require_grayscale_gallery("HTML export")

        def binned_frames(stack: np.ndarray) -> np.ndarray:
            """A ``(F, H, W)`` stack mean-binned frame by frame by ``downsample``; the pixel size below scales to match."""
            if downsample <= 1:
                return stack
            return np.stack([bin2d(frame, factor=downsample, mode="mean") for frame in stack])

        has_local_stacks = any(count > 1 for count in self.panel_frame_counts)
        if has_local_stacks:
            # A one-frame panel goes in as a 2D image so the clone keeps it static.
            export_data = [
                binned_frames(stack) if stack.shape[0] > 1 else binned_frames(stack)[0]
                for stack in self._display_panel_stacks
            ]
        else:
            export_data = binned_frames(self._display_data)
        export_pixel_size = self.pixel_size * downsample if self.pixel_size > 0 else self.pixel_size
        clone = type(self)(
            export_data,
            labels=list(self.labels),
            title=self.title,
            cmap=list(self.panel_cmaps) if self.panel_cmaps else self.cmap,
            sampling=export_pixel_size if export_pixel_size > 0 else None,
            units=self.pixel_unit,
            show_scale_bar=self.scale_bar_visible,
            scale_bar_position=self.scale_bar_position,
            scale_bar_panels=list(self.scale_bar_panels),
            scale_bar_length=self.scale_bar_length,
            scale_bar_label=self.scale_bar_label,
            scale_bar_style=dict(self.scale_bar_style),
            show_zoom_indicator=self.show_zoom_indicator,
            show_fft=self.show_fft,
            show_controls=self.show_controls,
            controls_collapsed=self.controls_collapsed,
            show_stats=self.show_stats,
            verbose=False,
            log_scale=self.log_scale,
            auto_contrast=self.auto_contrast,
            offline=quantized,
            vmin=self.vmin if self.vmin is not None else self.vmins,
            vmax=self.vmax if self.vmax is not None else self.vmaxs,
            ncols=self.ncols,
            panel_frame_indices=list(self.panel_frame_indices),
            panel_playback_fps=self.panel_playback_fps,
            size=self.size,
            smooth=self.smooth,
            zoom=self.initial_zoom,
            link_zoom=self.link_zoom,
            link_pan=self.link_pan,
            link_contrast=self.link_contrast,
            starred=[i for i, value in enumerate(self.starred) if value],
            show_panel_titles=self.show_panel_titles,
            panel_title_font_size=self.panel_title_font_size,
            panel_title_style=dict(self.panel_title_style),
            inter_panel_gap_px=int(self.inter_panel_gap_px),
            inter_panel_gap_color=str(self.inter_panel_gap_color),
            gallery_outer_border_px=int(self.gallery_outer_border_px),
            gallery_outer_border_color=str(self.gallery_outer_border_color),
            panel_inner_border_px=float(self.panel_inner_border_px),
            panel_inner_border_color=str(self.panel_inner_border_color),
            row_markers=dict(self.row_markers),
            panel_annotations=list(self.panel_annotations),
            panel_overlays=list(self.panel_overlays),
            display_bin=1,
        )
        clone.pixel_sizes = list(self.pixel_sizes)
        clone.page_kind = str(self.page_kind)
        clone.n_pages = int(self.n_pages)
        clone.panels_per_page = int(self.panels_per_page)
        clone.page_labels = list(self.page_labels)
        clone.page_idx = int(self.page_idx)
        clone.page_starred = list(self.page_starred)
        clone.load_state_dict(self.state_dict())
        clone._export_light = True
        clone._save_state = True
        clone.export_enabled = False
        clone.export_status = ""
        clone.export_payload = b""
        clone.export_payload_id = ""
        clone.export_filename = ""
        clone._update_all_frames()
        return clone

