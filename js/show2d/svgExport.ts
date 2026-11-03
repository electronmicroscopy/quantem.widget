// Vector pieces of Show2D's SVG export: escaped text, wrapped titles, panel
// overlays and annotations, inset plots and colorbars as SVG elements. The
// export composes them around the PNG panels so the chrome stays vector.

import { formatNumber } from "../format";
import { LATEX_SYMBOLS, readLatexGroup, type RichTitleSpan } from "../shared/latexTitle";
import {
  overlayDashPattern,
  overlayGeometry,
  styleNumber,
  styleString,
  type PanelAnnotationSpec,
  type PanelOverlaySpec,
} from "../shared/panelOverlays";
import { formatInsetTick, insetPlotGeometry, SYSTEM_FONT, type InsetPlotSpec } from "./insetPlots";

/**
 * SVG paint attributes for a CSS color, as `fill="rgb(r, g, b)" fill-opacity="a"`.
 * SVG 1.1 readers such as Illustrator do not parse rgba(), so the color's
 * alpha moves into the -opacity attribute, multiplied by `opacity`; the
 * attribute is left out when the product is 1.
 */
export function svgPaint(property: "fill" | "stroke", value: unknown, fallback = "", opacity = 1): string {
  const color = styleString(value, fallback);
  const rgba = /^\s*rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([\d.]+)\s*\)\s*$/i.exec(color);
  const paint = rgba ? `rgb(${Number(rgba[1])}, ${Number(rgba[2])}, ${Number(rgba[3])})` : color;
  const alpha = (rgba ? Number(rgba[4]) : 1) * opacity;
  return `${property}="${escapeXmlAttr(paint)}"${alpha < 1 ? ` ${property}-opacity="${alpha}"` : ""}`;
}

/**
 * The alphabetic baseline of text the canvas draws with textBaseline "top",
 * "middle" or "bottom" at `y`, from typical ascent and descent fractions of
 * the font size: SVG 1.1 readers ignore dominant-baseline.
 */
function alphabeticY(y: number, fontSize: number, baseline: "top" | "middle" | "bottom"): number {
  if (baseline === "top") return y + 0.8 * fontSize;
  if (baseline === "middle") return y + 0.35 * fontSize;
  return y - 0.2 * fontSize;
}

export function escapeXmlText(value: unknown): string {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

export function escapeXmlAttr(value: unknown): string {
  return escapeXmlText(value).replace(/"/g, "&quot;");
}

function measureSvgTextWidth(text: string, fontSize: number): number {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) return text.length * fontSize * 0.55;
  ctx.font = `700 ${fontSize}px ${SYSTEM_FONT}`;
  return ctx.measureText(text).width;
}

export function wrapSvgTextLines(text: string, fontSize: number, maxWidth: number, maxLines: number = 3): string[] {
  const words = String(text || "").trim().split(/\s+/).filter(Boolean);
  if (words.length === 0) return [];
  const lines: string[] = [];
  let current = "";

  const pushCurrent = () => {
    if (current) {
      lines.push(current);
      current = "";
    }
  };

  const appendLongWord = (word: string) => {
    let fragment = "";
    for (const char of word) {
      const candidate = fragment + char;
      if (fragment && measureSvgTextWidth(candidate, fontSize) > maxWidth) {
        lines.push(fragment);
        fragment = char;
        if (lines.length >= maxLines) return;
      } else {
        fragment = candidate;
      }
    }
    current = fragment;
  };

  for (const word of words) {
    if (lines.length >= maxLines) break;
    const candidate = current ? `${current} ${word}` : word;
    if (measureSvgTextWidth(candidate, fontSize) <= maxWidth) {
      current = candidate;
      continue;
    }
    pushCurrent();
    if (lines.length >= maxLines) break;
    if (measureSvgTextWidth(word, fontSize) <= maxWidth) {
      current = word;
    } else {
      appendLongWord(word);
    }
  }
  pushCurrent();
  return lines.slice(0, maxLines);
}

function svgDashAttributes(overlay: PanelOverlaySpec, lineWidth: number): string {
  const pattern = overlayDashPattern(overlay, lineWidth);
  if (!pattern.length) return "";
  return ` stroke-dasharray="${escapeXmlAttr(pattern.map((dash) => `${dash}`).join(" "))}" stroke-linecap="round"`;
}

function renderLatexMathToText(expr: string): string {
  const superscript: Record<string, string> = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
    "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾",
    n: "ⁿ", i: "ⁱ",
  };
  const subscript: Record<string, string> = {
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
    "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
    "+": "₊", "-": "₋", "=": "₌", "(": "₍", ")": "₎",
    a: "ₐ", e: "ₑ", h: "ₕ", i: "ᵢ", j: "ⱼ", k: "ₖ",
    l: "ₗ", m: "ₘ", n: "ₙ", o: "ₒ", p: "ₚ", r: "ᵣ",
    s: "ₛ", t: "ₜ", u: "ᵤ", v: "ᵥ", x: "ₓ",
  };
  const convertScript = (text: string, table: Record<string, string>, marker: string): string =>
    text.split("").map((ch) => table[ch] || `${marker}${ch}`).join("");
  const normalized = String(expr || "")
    .trim()
    .replace(/^\$|\$$/g, "")
    .replace(/\\+(?=[A-Za-z])/g, "\\")
    .replace(/\\([A-Za-z]+)/g, (_match, command: string) => LATEX_SYMBOLS[command] || command);
  let out = "";
  for (let i = 0; i < normalized.length; i += 1) {
    const ch = normalized[i];
    if ((ch === "^" || ch === "_") && i + 1 < normalized.length) {
      const table = ch === "^" ? superscript : subscript;
      if (normalized[i + 1] === "{") {
        const group = readLatexGroup(normalized, i + 1);
        out += convertScript(group.text, table, ch);
        i = group.next - 1;
      } else {
        out += convertScript(normalized[i + 1], table, ch);
        i += 1;
      }
      continue;
    }
    if (ch === "{" || ch === "}") continue;
    out += ch;
  }
  return out;
}

export function svgPanelOverlayElement(
  overlay: PanelOverlaySpec,
  toScreenX: (col: number) => number,
  toScreenY: (row: number) => number,
  imageW: number,
  imageH: number,
): string {
  const geometry = overlayGeometry(overlay, imageW, imageH);
  const opacity = styleNumber(overlay.opacity, 1);
  const strokeOpacity = opacity * styleNumber(overlay.stroke_opacity, 1);
  const fillOpacity = opacity * styleNumber(overlay.fill_opacity, overlay.fill_color ? 1 : 0);
  const fill = overlay.fill_color ? svgPaint("fill", overlay.fill_color, "none", fillOpacity) : `fill="none"`;
  const lineWidth = Math.max(0, styleNumber(overlay.line_width, 2));
  const common = `${fill} ${svgPaint("stroke", overlay.color, "#00e5ff", strokeOpacity)} stroke-width="${lineWidth}"${svgDashAttributes(overlay, lineWidth)}`;
  if (geometry.shape === "circle") {
    const cx = toScreenX(geometry.col);
    const cy = toScreenY(geometry.row);
    const radius = Math.max(0, (Math.abs(toScreenX(geometry.col + geometry.radius) - cx) + Math.abs(toScreenY(geometry.row + geometry.radius) - cy)) / 2);
    return `<circle cx="${cx}" cy="${cy}" r="${radius}" ${common}/>`;
  }
  const x0 = toScreenX(geometry.col0);
  const y0 = toScreenY(geometry.row0);
  const x1 = toScreenX(geometry.col1);
  const y1 = toScreenY(geometry.row1);
  return `<rect x="${Math.min(x0, x1)}" y="${Math.min(y0, y1)}" width="${Math.abs(x1 - x0)}" height="${Math.abs(y1 - y0)}" ${common}/>`;
}

export function svgTextFromRichSpans(spans: RichTitleSpan[] | undefined, fallback: string): { text: string; spans: Array<{ text: string; color?: string }> } {
  if (!spans?.length) return { text: fallback, spans: [{ text: fallback }] };
  const parts = spans.map((span) => ({
    text: span.math ? renderLatexMathToText(String(span.math)) : String(span.text ?? ""),
    color: styleString(span.color) || undefined,
  }));
  return { text: parts.map((part) => part.text).join(""), spans: parts };
}

export function svgPanelAnnotationElement(spec: PanelAnnotationSpec, x: number, y: number, panelW: number, panelH: number): string {
  const position = spec.position || "top-left";
  const offset = Array.isArray(spec.offset) ? spec.offset.map(Number) : [0, 0];
  const margin = 10;
  let anchorX = x + margin;
  let anchorY = y + margin;
  let anchor = "start";
  let baseline = "hanging";
  const align = styleString(spec.align, "").toLowerCase();
  const alignAnchor = align === "left" || align === "start" ? "start"
    : align === "right" || align === "end" ? "end"
    : align === "center" || align === "middle" ? "middle"
    : "";
  if (Array.isArray(spec.box) && spec.box.length >= 4) {
    if (alignAnchor === "start") anchorX = x + Number(spec.box[0]) * panelW + Math.max(0, styleNumber(spec.pad_x, 0));
    else if (alignAnchor === "end") anchorX = x + (Number(spec.box[0]) + Number(spec.box[2])) * panelW - Math.max(0, styleNumber(spec.pad_x, 0));
    else anchorX = x + (Number(spec.box[0]) + Number(spec.box[2]) / 2) * panelW;
    anchorY = y + (Number(spec.box[1]) + Number(spec.box[3]) / 2) * panelH;
    anchor = alignAnchor || "middle";
    baseline = "middle";
  } else if (Number.isFinite(spec.x) && Number.isFinite(spec.y)) {
    anchorX = x + Number(spec.x) * panelW;
    anchorY = y + Number(spec.y) * panelH;
    const anchorValue = spec.anchor || "center";
    anchor = String(anchorValue).includes("right") ? "end" : String(anchorValue).includes("center") ? "middle" : "start";
    baseline = String(anchorValue).includes("bottom") ? "baseline" : String(anchorValue).includes("center") ? "middle" : "hanging";
    if (alignAnchor) anchor = alignAnchor;
  } else {
    if (position.includes("right")) { anchorX = x + panelW - margin; anchor = "end"; }
    else if (position.includes("center")) { anchorX = x + panelW / 2; anchor = "middle"; }
    if (position.includes("bottom")) { anchorY = y + panelH - margin; baseline = "baseline"; }
    else if (position.includes("center")) { anchorY = y + panelH / 2; baseline = "middle"; }
    if (alignAnchor) anchor = alignAnchor;
  }
  anchorX += Number(offset[0] || 0);
  anchorY += Number(offset[1] || 0);
  const fontSize = Math.max(6, styleNumber(spec.font_size, 10));
  const rich = svgTextFromRichSpans(spec.math ? [{ math: spec.math }] : spec.spans, spec.text || "");
  const variant = spec.variant || "badge";
  const opacity = Math.max(0, Math.min(1, styleNumber(spec.opacity, 1)));
  const fontFamily = styleString(spec.font_family, SYSTEM_FONT);
  const outlineWidth = Math.max(0, styleNumber(spec.outline_width, 0));
  const chunks: string[] = [`<g opacity="${opacity}">`];
  // The screen badge pads its text 6 px left and right and centers it
  // vertically in a 1.25 em + 4 px box; a plain label has no box or padding.
  const padX = variant === "plain" ? 0 : 6;
  const boxW = Math.max(12, rich.text.length * fontSize * 0.62 + 2 * padX);
  const boxH = fontSize * 1.25 + 4;
  const boxX = anchor === "middle" ? anchorX - boxW / 2 : anchor === "end" ? anchorX - boxW : anchorX;
  const boxY = baseline === "middle" ? anchorY - boxH / 2 : baseline === "baseline" ? anchorY - boxH : anchorY;
  if (variant !== "plain") {
    chunks.push(`<rect x="${boxX}" y="${boxY}" width="${boxW}" height="${boxH}" rx="${styleNumber(spec.radius, 3)}" ${svgPaint("fill", spec.bg, "rgba(0,0,0,0.72)")} ${svgPaint("stroke", spec.border_color, "rgba(255,255,255,0.5)")} stroke-width="${Math.max(0, styleNumber(spec.border_width, variant === "outline" || variant === "callout" ? 1 : 0))}"/>`);
  }
  const textX = anchor === "start" ? anchorX + padX : anchor === "end" ? anchorX - padX : anchorX;
  const textY = alphabeticY(boxY + boxH / 2, fontSize, "middle");
  const textAttrs = `x="${textX}" y="${textY}" text-anchor="${anchor}" font-family="${escapeXmlAttr(fontFamily)}" font-size="${fontSize}" font-weight="${escapeXmlAttr(spec.font_weight ?? 700)}"`;
  if (outlineWidth > 0) {
    chunks.push(`<text ${textAttrs} fill="none" ${svgPaint("stroke", spec.outline_color, "rgba(0,0,0,0.85)")} stroke-width="${outlineWidth}" stroke-linejoin="round">${escapeXmlText(rich.text)}</text>`);
  }
  chunks.push(`<text ${textAttrs} ${svgPaint("fill", spec.color, "#fff")}>`);
  rich.spans.forEach((span) => chunks.push(`<tspan${span.color ? ` ${svgPaint("fill", span.color)}` : ""}>${escapeXmlText(span.text)}</tspan>`));
  chunks.push("</text></g>");
  return chunks.join("");
}

/**
 * An inset plot as SVG, element for element what drawInsetPlot paints on the
 * canvas: box, axes, ticks and their labels, the clipped line, the marker,
 * title, legend and annotation corners, axis labels and the panel index.
 */
export function svgInsetPlotElement(
  spec: InsetPlotSpec | null | undefined,
  x: number,
  y: number,
  panelW: number,
  panelH: number,
  fallbackColor: string,
  showScaleBar: boolean,
  panel: number,
  clipId: string,
): string {
  const inset = insetPlotGeometry(spec, panelW, panelH, showScaleBar);
  if (!inset || !spec) return "";
  const { finite, xlim, ylim, x0, y0, boxW, boxH, plotX0, plotY0, plotW, plotH, showTicks, tickFont, labelFont, legendFont } = inset;
  const left = x + x0;
  const top = y + y0;
  const plotLeft = x + plotX0;
  const plotTop = y + plotY0;
  const plotBottom = plotTop + plotH;
  const toPlotX = (value: number) => plotLeft + (value - xlim[0]) / (xlim[1] - xlim[0]) * plotW;
  const toPlotY = (value: number) => plotBottom - (value - ylim[0]) / (ylim[1] - ylim[0]) * plotH;
  const lineColor = spec.color || fallbackColor;
  const textColor = spec.text_color || "rgba(255,255,255,0.92)";
  const tickColor = spec.tick_color || "rgba(255,255,255,0.72)";
  const backgroundAlpha = Math.max(0, Math.min(1, Number(spec.background_alpha ?? 0.68)));
  const borderWidth = Math.max(0, Math.min(6, Number(spec.border_width ?? 1)));
  const font = (size: number, weight = "") => `font-family="${escapeXmlAttr(SYSTEM_FONT)}" font-size="${size}"${weight ? ` font-weight="${weight}"` : ""}`;
  const text = (textX: number, textY: number, value: string, attributes: string) =>
    `<text x="${textX}" y="${textY}" ${attributes}>${escapeXmlText(value)}</text>`;
  const chunks = [
    `<g>`,
    `<defs><clipPath id="${escapeXmlAttr(clipId)}"><rect x="${plotLeft}" y="${plotTop}" width="${plotW}" height="${plotH}"/></clipPath></defs>`,
    // A background color is drawn as given; the default box takes background_alpha.
    `<rect x="${left}" y="${top}" width="${boxW}" height="${boxH}" ${spec.background ? svgPaint("fill", spec.background) : svgPaint("fill", "rgb(10, 12, 16)", "", backgroundAlpha)}${borderWidth > 0 ? ` ${svgPaint("stroke", spec.border_color, "rgba(255,255,255,0.34)")} stroke-width="${borderWidth}"` : ""}/>`,
    `<path d="M ${plotLeft} ${plotTop} V ${plotBottom} H ${plotLeft + plotW}" fill="none" ${svgPaint("stroke", spec.tick_color, "rgba(255,255,255,0.28)")} stroke-width="1"/>`,
  ];
  if (showTicks) {
    const xticks = Array.isArray(spec.xticks) && spec.xticks.length > 0 ? spec.xticks.map(Number) : [xlim[0], xlim[1]];
    const yticks = Array.isArray(spec.yticks) && spec.yticks.length > 0 ? spec.yticks.map(Number) : [ylim[0], ylim[1]];
    const tickStroke = `${svgPaint("stroke", spec.tick_color, "rgba(255,255,255,0.34)")} stroke-width="1"`;
    for (const value of xticks) {
      const tickX = toPlotX(value);
      if (!Number.isFinite(value) || tickX < plotLeft - 0.5 || tickX > plotLeft + plotW + 0.5) continue;
      chunks.push(
        `<path d="M ${tickX} ${plotBottom} V ${plotBottom + 3}" ${tickStroke}/>`,
        text(tickX, alphabeticY(plotBottom + 4, tickFont, "top"), formatInsetTick(value), `text-anchor="middle" ${font(tickFont)} ${svgPaint("fill", tickColor)}`),
      );
    }
    for (const value of yticks) {
      const tickY = toPlotY(value);
      if (!Number.isFinite(value) || tickY < plotTop - 0.5 || tickY > plotBottom + 0.5) continue;
      chunks.push(
        `<path d="M ${plotLeft - 3} ${tickY} H ${plotLeft}" ${tickStroke}/>`,
        text(plotLeft - 5, alphabeticY(tickY, tickFont, "middle"), formatInsetTick(value), `text-anchor="end" ${font(tickFont)} ${svgPaint("fill", tickColor)}`),
      );
    }
  }
  const points = finite.map(([dataX, dataY]) => `${toPlotX(dataX)},${toPlotY(dataY)}`).join(" ");
  chunks.push(`<polyline points="${points}" clip-path="url(#${escapeXmlAttr(clipId)})" fill="none" ${svgPaint("stroke", lineColor)} stroke-width="${Math.max(1.4, Number(spec.line_width ?? 2))}" stroke-linejoin="round" stroke-linecap="round"/>`);
  if (Array.isArray(spec.point) && spec.point.length >= 2) {
    const markerX = toPlotX(Number(spec.point[0]));
    const markerY = toPlotY(Number(spec.point[1]));
    if (markerX >= plotLeft - 1 && markerX <= plotLeft + plotW + 1 && markerY >= plotTop - 1 && markerY <= plotBottom + 1) {
      chunks.push(`<circle cx="${markerX}" cy="${markerY}" r="3.4" ${svgPaint("fill", spec.point_color, "#fff")} ${svgPaint("stroke", "rgba(0,0,0,0.75)")} stroke-width="1.5"/>`);
    }
  }
  if (spec.title) chunks.push(text(left + 6, alphabeticY(top + 4, legendFont, "top"), spec.title, `${font(legendFont, "700")} ${svgPaint("fill", textColor)}`));
  // Corner texts carry the canvas's 0.8 px drop shadow.
  const cornerText = (value: string | undefined, position: string | undefined, color: string) => {
    if (!value) return;
    const corner = position || "top-left";
    const right = corner.includes("right");
    const bottom = corner.includes("bottom");
    const textX = right ? left + boxW - 6 : left + 6;
    const textY = alphabeticY(bottom ? top + boxH - 4 : top + 4, legendFont, bottom ? "bottom" : "top");
    const attributes = `text-anchor="${right ? "end" : "start"}" ${font(legendFont, "700")}`;
    chunks.push(
      text(textX + 0.8, textY + 0.8, value, `${attributes} ${svgPaint("fill", "rgba(0,0,0,0.45)")}`),
      text(textX, textY, value, `${attributes} ${svgPaint("fill", color)}`),
    );
  };
  cornerText(spec.legend, spec.legend_position, spec.text_color || lineColor);
  cornerText(spec.annotation, spec.annotation_position || "top-right", textColor);
  if (spec.xlabel) chunks.push(text(left + boxW - 7, alphabeticY(top + boxH - 3, labelFont, "bottom"), spec.xlabel, `text-anchor="end" ${font(labelFont)} ${svgPaint("fill", tickColor)}`));
  if (spec.ylabel) {
    chunks.push(`<g transform="translate(${left + 5} ${plotTop + 2}) rotate(-90)">${text(0, alphabeticY(0, labelFont, "top"), spec.ylabel, `text-anchor="end" ${font(labelFont)} ${svgPaint("fill", tickColor)}`)}</g>`);
  }
  if (spec.show_panel_index) chunks.push(text(left + boxW - 5, alphabeticY(top + boxH - 4, 7, "bottom"), `${panel + 1}`, `text-anchor="end" ${font(7)} ${svgPaint("fill", "rgba(255,255,255,0.42)")}`));
  chunks.push("</g>");
  return chunks.join("");
}

export function svgColorbarElements(lut: Uint8Array, x: number, y: number, panelW: number, panelH: number, vmin: number, vmax: number, id: string): { def: string; body: string } {
  const stops: string[] = [];
  for (let step = 0; step <= 8; step += 1) {
    const frac = step / 8;
    const lutIndex = Math.max(0, Math.min(255, Math.round(frac * 255))) * 3;
    stops.push(`<stop offset="${frac * 100}%" stop-color="rgb(${lut[lutIndex]}, ${lut[lutIndex + 1]}, ${lut[lutIndex + 2]})"/>`);
  }
  const barH = Math.min(160, panelH * 0.62);
  const barX = x + panelW - 22;
  const barY = y + 18;
  return {
    def: `<linearGradient id="${id}" x1="0" x2="0" y1="1" y2="0">${stops.join("")}</linearGradient>`,
    body: `<g><rect x="${barX - 1}" y="${barY - 1}" width="12" height="${barH + 2}" fill="#000" fill-opacity="0.45"/><rect x="${barX}" y="${barY}" width="10" height="${barH}" fill="url(#${id})" stroke="#fff" stroke-opacity="0.75" stroke-width="0.75"/><text x="${barX - 4}" y="${barY + 4}" text-anchor="end" font-family="${SYSTEM_FONT}" font-size="9" fill="#fff">${escapeXmlText(formatNumber(vmax))}</text><text x="${barX - 4}" y="${barY + barH}" text-anchor="end" font-family="${SYSTEM_FONT}" font-size="9" fill="#fff">${escapeXmlText(formatNumber(vmin))}</text></g>`,
  };
}
