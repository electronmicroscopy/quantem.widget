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
import { insetPlotGeometry, SYSTEM_FONT, type InsetPlotSpec } from "./insetPlots";

export function svgColor(value: unknown, fallback = ""): string {
  return styleString(value, fallback).replace(
    /rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(?:0|1|0?\.\d+)\s*\)/gi,
    (_match, r, g, b) => `rgb(${Number(r)}, ${Number(g)}, ${Number(b)})`,
  );
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
  const fillOpacity = opacity * styleNumber(overlay.fill_opacity, overlay.fill ? 1 : 0);
  const stroke = svgColor(overlay.stroke, "#00e5ff");
  const fill = overlay.fill ? svgColor(overlay.fill, "none") : "none";
  const lineWidth = Math.max(0, styleNumber(overlay.stroke_width, 2));
  const common = `fill="${escapeXmlAttr(fill)}" fill-opacity="${fillOpacity}" stroke="${escapeXmlAttr(stroke)}" stroke-width="${lineWidth}" stroke-opacity="${strokeOpacity}"${svgDashAttributes(overlay, lineWidth)}`;
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
  const fg = svgColor(spec.fg ?? spec.color, "#fff");
  const opacity = Math.max(0, Math.min(1, styleNumber(spec.opacity, 1)));
  const fontFamily = styleString(spec.font_family, SYSTEM_FONT);
  const outlineWidth = Math.max(0, styleNumber(spec.outline_width, 0));
  const outlineColor = svgColor(spec.outline_color, "rgba(0,0,0,0.85)");
  const chunks: string[] = [`<g opacity="${opacity}">`];
  if (variant !== "plain") {
    const boxW = Math.max(12, rich.text.length * fontSize * 0.62 + 12);
    const boxH = fontSize * 1.25 + 4;
    const boxX = anchor === "middle" ? anchorX - boxW / 2 : anchor === "end" ? anchorX - boxW : anchorX;
    const boxY = baseline === "middle" ? anchorY - boxH / 2 : baseline === "baseline" ? anchorY - boxH : anchorY;
    chunks.push(`<rect x="${boxX}" y="${boxY}" width="${boxW}" height="${boxH}" rx="${styleNumber(spec.radius, 3)}" fill="${escapeXmlAttr(svgColor(spec.bg, "rgba(0,0,0,0.72)"))}" stroke="${escapeXmlAttr(svgColor(spec.border_color, "rgba(255,255,255,0.5)"))}" stroke-width="${Math.max(0, styleNumber(spec.border_width, variant === "outline" || variant === "callout" ? 1 : 0))}"/>`);
  }
  const textY = baseline === "middle" ? anchorY + fontSize * 0.35 : anchorY;
  const textAttrs = `x="${anchorX}" y="${textY}" text-anchor="${anchor}" font-family="${escapeXmlAttr(fontFamily)}" font-size="${fontSize}" font-weight="${escapeXmlAttr(spec.font_weight ?? 700)}"`;
  if (outlineWidth > 0) {
    chunks.push(`<text ${textAttrs} fill="none" stroke="${escapeXmlAttr(outlineColor)}" stroke-width="${outlineWidth}" stroke-linejoin="round">${escapeXmlText(rich.text)}</text>`);
  }
  chunks.push(`<text ${textAttrs} fill="${escapeXmlAttr(fg)}">`);
  rich.spans.forEach((span) => chunks.push(`<tspan${span.color ? ` fill="${escapeXmlAttr(svgColor(span.color))}"` : ""}>${escapeXmlText(span.text)}</tspan>`));
  chunks.push("</text></g>");
  return chunks.join("");
}

export function svgInsetPlotElement(spec: InsetPlotSpec | null | undefined, x: number, y: number, panelW: number, panelH: number, fallbackColor: string, scaleBarVisible: boolean): string {
  const inset = insetPlotGeometry(spec, panelW, panelH, scaleBarVisible);
  if (!inset || !spec) return "";
  const { finite, xlim, ylim, x0, y0, boxW, boxH, plotX0, plotY0, plotW, plotH, legendFont } = inset;
  const toPlotX = (value: number) => x + plotX0 + (value - xlim[0]) / (xlim[1] - xlim[0]) * plotW;
  const toPlotY = (value: number) => y + plotY0 + plotH - (value - ylim[0]) / (ylim[1] - ylim[0]) * plotH;
  const points = finite.map(([dataX, dataY]) => `${toPlotX(dataX)},${toPlotY(dataY)}`).join(" ");
  const lineColor = svgColor(spec.color, fallbackColor);
  const textColor = svgColor(spec.text_color, "rgba(255,255,255,0.92)");
  const tickColor = svgColor(spec.tick_color, "rgba(255,255,255,0.72)");
  const chunks = [
    `<g>`,
    `<rect x="${x + x0}" y="${y + y0}" width="${boxW}" height="${boxH}" fill="${escapeXmlAttr(svgColor(spec.background, "#0a0c10"))}" fill-opacity="${Math.max(0, Math.min(1, Number(spec.background_alpha ?? 0.68)))}" stroke="${escapeXmlAttr(svgColor(spec.border_color, "rgba(255,255,255,0.34)"))}" stroke-width="${Number(spec.border_width ?? 1)}"/>`,
    `<path d="M ${x + plotX0} ${y + plotY0} V ${y + plotY0 + plotH} H ${x + plotX0 + plotW}" fill="none" stroke="${escapeXmlAttr(tickColor)}" stroke-opacity="0.45" stroke-width="1"/>`,
    `<polyline points="${points}" fill="none" stroke="${escapeXmlAttr(lineColor)}" stroke-width="${Math.max(1.4, Number(spec.line_width ?? 2))}" stroke-linejoin="round" stroke-linecap="round"/>`,
  ];
  if (Array.isArray(spec.point) && spec.point.length >= 2) {
    chunks.push(`<circle cx="${toPlotX(Number(spec.point[0]))}" cy="${toPlotY(Number(spec.point[1]))}" r="3.4" fill="${escapeXmlAttr(svgColor(spec.point_color, "#fff"))}" stroke="#000" stroke-width="1.5"/>`);
  }
  if (spec.title) chunks.push(`<text x="${x + x0 + 6}" y="${y + y0 + 12}" font-family="${SYSTEM_FONT}" font-size="${legendFont}" font-weight="700" fill="${escapeXmlAttr(textColor)}">${escapeXmlText(spec.title)}</text>`);
  if (spec.legend) chunks.push(`<text x="${x + x0 + 6}" y="${y + y0 + boxH - 6}" font-family="${SYSTEM_FONT}" font-size="${legendFont}" font-weight="700" fill="${escapeXmlAttr(lineColor)}">${escapeXmlText(spec.legend)}</text>`);
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
