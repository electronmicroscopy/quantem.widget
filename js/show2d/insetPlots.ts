// Inset line plots drawn over a Show2D panel (the inset_plots trait): the
// box geometry inside the panel, the canvas drawing, and the hover readout.

/** The widget's UI font stack (canvas text, SVG export, panel titles). */
export const SYSTEM_FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";

export type InsetPlotSpec = {
  x?: number[];
  y?: number[];
  points?: [number, number][];
  point?: [number, number];
  xlim?: [number, number];
  ylim?: [number, number];
  box?: [number, number, number, number];
  xticks?: number[];
  yticks?: number[];
  show_ticks?: boolean;
  show_panel_index?: boolean;
  title?: string;
  legend?: string;
  legend_position?: "top-left" | "top-right" | "bottom-left" | "bottom-right";
  annotation?: string;
  annotation_position?: "top-left" | "top-right" | "bottom-left" | "bottom-right";
  xlabel?: string;
  ylabel?: string;
  color?: string;
  point_color?: string;
  border_color?: string;
  text_color?: string;
  tick_color?: string;
  position?: "bottom-right" | "bottom-left" | "bottom-center" | "top-right" | "top-left" | "top-center" | "center" | "center-left" | "center-right";
  margin?: number | [number, number];
  size?: number;
  height?: number;
  line_width?: number;
  border_width?: number;
  tick_font_size?: number;
  label_font_size?: number;
  legend_font_size?: number;
  background?: string;
  background_alpha?: number;
};

export type InsetHoverInfo = {
  idx: number;
  leftPct: number;
  topPct: number;
  text: string;
};

export type InsetDragState = {
  idx: number;
  offsetX: number;
  offsetY: number;
  boxW: number;
  boxH: number;
};

function finiteMinMax(values: number[]): [number, number] | null {
  let lo = Infinity;
  let hi = -Infinity;
  for (const value of values) {
    if (!Number.isFinite(value)) continue;
    if (value < lo) lo = value;
    if (value > hi) hi = value;
  }
  return lo <= hi ? [lo, hi] : null;
}

function expandFlatRange([lo, hi]: [number, number]): [number, number] {
  if (hi > lo) return [lo, hi];
  const pad = Math.max(1, Math.abs(lo) * 0.05);
  return [lo - pad, hi + pad];
}

function formatInsetTick(value: number): string {
  const abs = Math.abs(value);
  if (abs > 0 && (abs < 0.01 || abs >= 1000)) return value.toExponential(1);
  if (abs >= 100) return value.toFixed(0);
  if (abs >= 10) return value.toFixed(1).replace(/\.0$/, "");
  return value.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
}

function formatInsetValue(value: number): string {
  const abs = Math.abs(value);
  if (abs > 0 && (abs < 0.001 || abs >= 10000)) return value.toExponential(2);
  if (abs >= 100) return value.toFixed(1);
  if (abs >= 10) return value.toFixed(2);
  return value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
}

function drawInsetCornerText(
  ctx: CanvasRenderingContext2D,
  text: string | undefined,
  position: string | undefined,
  x0: number,
  y0: number,
  boxW: number,
  boxH: number,
  fontPx: number,
  color: string,
): void {
  if (!text) return;
  const corner = position || "top-left";
  const right = corner.includes("right");
  const bottom = corner.includes("bottom");
  ctx.font = `700 ${fontPx}px ${SYSTEM_FONT}`;
  ctx.textAlign = right ? "right" : "left";
  ctx.textBaseline = bottom ? "bottom" : "top";
  ctx.fillStyle = "rgba(0,0,0,0.45)";
  const x = right ? x0 + boxW - 6 : x0 + 6;
  const y = bottom ? y0 + boxH - 4 : y0 + 4;
  ctx.fillText(text, x + 0.8, y + 0.8);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
}

export function insetPlotGeometry(
  spec: InsetPlotSpec | null | undefined,
  cssW: number,
  cssH: number,
  scaleBarVisible: boolean,
): {
  finite: [number, number][];
  xlim: [number, number];
  ylim: [number, number];
  x0: number;
  y0: number;
  boxW: number;
  boxH: number;
  plotX0: number;
  plotY0: number;
  plotW: number;
  plotH: number;
  showTicks: boolean;
  tickFont: number;
  labelFont: number;
  legendFont: number;
} | null {
  if (!spec) return null;
  const x = Array.isArray(spec.x) ? spec.x.map(Number) : null;
  const y = Array.isArray(spec.y) ? spec.y.map(Number) : null;
  if (!y || y.length < 2) return null;
  const xSeries = x && x.length === y.length ? x : y.map((_, idx) => idx);
  const finite = xSeries.map((xValue, idx) => [xValue, y[idx]] as [number, number])
    .filter(([xValue, yValue]) => Number.isFinite(xValue) && Number.isFinite(yValue));
  if (finite.length < 2) return null;
  const xValues = finite.map(([value]) => value);
  const yValues = finite.map(([, value]) => value);
  const xlim = expandFlatRange((Array.isArray(spec.xlim) && spec.xlim.length >= 2
    ? [Number(spec.xlim[0]), Number(spec.xlim[1])]
    : finiteMinMax(xValues)) as [number, number]);
  const ylim = expandFlatRange((Array.isArray(spec.ylim) && spec.ylim.length >= 2
    ? [Number(spec.ylim[0]), Number(spec.ylim[1])]
    : finiteMinMax(yValues)) as [number, number]);
  if (!Number.isFinite(xlim[0] + xlim[1] + ylim[0] + ylim[1])) return null;

  const sizeFrac = Math.max(0.18, Math.min(0.62, Number(spec.size ?? 0.31)));
  let boxW = Math.max(78, Math.min(cssW * 0.62, cssW * sizeFrac));
  let boxH = Math.max(50, Math.min(cssH * 0.55, cssW * Number(spec.height ?? sizeFrac * 0.68)));
  const rawMargin = Array.isArray(spec.margin)
    ? spec.margin.map(Number)
    : [Number(spec.margin ?? 12), Number(spec.margin ?? 12)];
  const marginX = Math.max(0, Number.isFinite(rawMargin[0]) ? rawMargin[0] : 12);
  const marginY = Math.max(0, Number.isFinite(rawMargin[1]) ? rawMargin[1] : marginX);
  const position = spec.position || "bottom-right";
  let x0: number;
  let y0: number;
  if (Array.isArray(spec.box) && spec.box.length >= 4) {
    const [left, top, widthFrac, heightFrac] = spec.box.map(Number);
    boxW = Math.max(48, Math.min(cssW, cssW * Math.max(0.05, Math.min(1, widthFrac))));
    boxH = Math.max(34, Math.min(cssH, cssH * Math.max(0.05, Math.min(1, heightFrac))));
    x0 = Math.max(0, Math.min(cssW - boxW, cssW * Math.max(0, Math.min(1, left))));
    y0 = Math.max(0, Math.min(cssH - boxH, cssH * Math.max(0, Math.min(1, top))));
  } else {
    if (position.includes("right")) x0 = cssW - boxW - marginX;
    else if (position.includes("center")) x0 = cssW / 2 - boxW / 2;
    else x0 = marginX;
    const scaleBarOffset = scaleBarVisible && position === "bottom-right" ? 34 : 0;
    if (position.includes("bottom")) y0 = cssH - boxH - marginY - scaleBarOffset;
    else if (position.includes("center")) y0 = cssH / 2 - boxH / 2;
    else y0 = marginY + 18;
  }
  const showTicks = Boolean(spec.show_ticks);
  const tickFont = Math.max(5, Math.min(14, Number(spec.tick_font_size ?? 7)));
  const labelFont = Math.max(6, Math.min(16, Number(spec.label_font_size ?? 8)));
  const legendFont = Math.max(6, Math.min(18, Number(spec.legend_font_size ?? 9)));
  const padL = showTicks || spec.ylabel ? Math.max(22, tickFont * 3.2) : 10;
  const padR = 7;
  const padT = spec.title || spec.legend ? Math.max(13, legendFont + 6) : 7;
  const padB = showTicks || spec.xlabel ? Math.max(16, tickFont + labelFont + 4) : 8;
  const plotX0 = x0 + padL;
  const plotY0 = y0 + padT;
  const plotW = boxW - padL - padR;
  const plotH = boxH - padT - padB;
  if (plotW <= 8 || plotH <= 8) return null;
  return { finite, xlim, ylim, x0, y0, boxW, boxH, plotX0, plotY0, plotW, plotH, showTicks, tickFont, labelFont, legendFont };
}

export function insetHoverAt(
  spec: InsetPlotSpec | null | undefined,
  panel: number,
  cssW: number,
  cssH: number,
  cssX: number,
  cssY: number,
  scaleBarVisible: boolean,
): InsetHoverInfo | null {
  const inset = insetPlotGeometry(spec, cssW, cssH, scaleBarVisible);
  if (!inset) return null;
  const { finite, xlim, ylim, x0, y0, boxW, boxH, plotX0, plotY0, plotW, plotH } = inset;
  if (cssX < x0 || cssX > x0 + boxW || cssY < y0 || cssY > y0 + boxH) return null;
  const toPlotX = (value: number) => plotX0 + (value - xlim[0]) / (xlim[1] - xlim[0]) * plotW;
  const toPlotY = (value: number) => plotY0 + plotH - (value - ylim[0]) / (ylim[1] - ylim[0]) * plotH;
  let best = finite[0];
  let bestDist = Infinity;
  for (const point of finite) {
    const dx = toPlotX(point[0]) - cssX;
    const dy = toPlotY(point[1]) - cssY;
    const dist = dx * dx + dy * dy;
    if (dist < bestDist) {
      bestDist = dist;
      best = point;
    }
  }
  const xName = spec?.xlabel || "x";
  const yName = spec?.ylabel || "y";
  return {
    idx: panel,
    leftPct: Math.max(3, Math.min(58, (cssX / cssW) * 100 + 2)),
    topPct: Math.max(5, Math.min(90, (cssY / cssH) * 100 - 6)),
    text: `${xName} ${formatInsetValue(best[0])} · ${yName} ${formatInsetValue(best[1])}`,
  };
}

export function drawInsetPlot(
  ctx: CanvasRenderingContext2D,
  spec: InsetPlotSpec | null | undefined,
  panel: number,
  cssW: number,
  cssH: number,
  fallbackColor: string,
  scaleBarVisible: boolean,
): void {
  const inset = insetPlotGeometry(spec, cssW, cssH, scaleBarVisible);
  if (!inset || !spec) return;
  const { finite, xlim, ylim, x0, y0, boxW, boxH, plotX0, plotY0, plotW, plotH, showTicks, tickFont, labelFont, legendFont } = inset;
  const toPlotX = (value: number) => plotX0 + (value - xlim[0]) / (xlim[1] - xlim[0]) * plotW;
  const toPlotY = (value: number) => plotY0 + plotH - (value - ylim[0]) / (ylim[1] - ylim[0]) * plotH;
  const lineColor = spec.color || fallbackColor;
  const pointColor = spec.point_color || "#fff";
  const textColor = spec.text_color || "rgba(255,255,255,0.92)";
  const tickColor = spec.tick_color || "rgba(255,255,255,0.72)";
  const backgroundAlpha = Math.max(0, Math.min(1, Number(spec.background_alpha ?? 0.68)));

  ctx.save();
  ctx.fillStyle = spec.background || `rgba(10, 12, 16, ${backgroundAlpha})`;
  ctx.strokeStyle = spec.border_color || "rgba(255,255,255,0.34)";
  ctx.lineWidth = Math.max(0, Math.min(6, Number(spec.border_width ?? 1)));
  ctx.fillRect(x0, y0, boxW, boxH);
  if (ctx.lineWidth > 0) ctx.strokeRect(x0, y0, boxW, boxH);

  ctx.strokeStyle = spec.tick_color || "rgba(255,255,255,0.28)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(plotX0, plotY0);
  ctx.lineTo(plotX0, plotY0 + plotH);
  ctx.lineTo(plotX0 + plotW, plotY0 + plotH);
  ctx.stroke();

  if (showTicks) {
    const xticks = Array.isArray(spec.xticks) && spec.xticks.length > 0 ? spec.xticks.map(Number) : [xlim[0], xlim[1]];
    const yticks = Array.isArray(spec.yticks) && spec.yticks.length > 0 ? spec.yticks.map(Number) : [ylim[0], ylim[1]];
    ctx.font = `${tickFont}px ${SYSTEM_FONT}`;
    ctx.fillStyle = tickColor;
    ctx.strokeStyle = spec.tick_color || "rgba(255,255,255,0.34)";
    ctx.lineWidth = 1;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const value of xticks) {
      if (!Number.isFinite(value)) continue;
      const tickX = toPlotX(value);
      if (tickX < plotX0 - 0.5 || tickX > plotX0 + plotW + 0.5) continue;
      ctx.beginPath();
      ctx.moveTo(tickX, plotY0 + plotH);
      ctx.lineTo(tickX, plotY0 + plotH + 3);
      ctx.stroke();
      ctx.fillText(formatInsetTick(value), tickX, plotY0 + plotH + 4);
    }
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const value of yticks) {
      if (!Number.isFinite(value)) continue;
      const tickY = toPlotY(value);
      if (tickY < plotY0 - 0.5 || tickY > plotY0 + plotH + 0.5) continue;
      ctx.beginPath();
      ctx.moveTo(plotX0 - 3, tickY);
      ctx.lineTo(plotX0, tickY);
      ctx.stroke();
      ctx.fillText(formatInsetTick(value), plotX0 - 5, tickY);
    }
  }

  ctx.save();
  ctx.beginPath();
  ctx.rect(plotX0, plotY0, plotW, plotH);
  ctx.clip();
  ctx.strokeStyle = lineColor;
  ctx.lineWidth = Math.max(1.4, Number(spec.line_width ?? 2));
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.shadowColor = "rgba(0,0,0,0.55)";
  ctx.shadowBlur = 2;
  ctx.beginPath();
  finite.forEach(([dataX, dataY], idx) => {
    const lineX = toPlotX(dataX);
    const lineY = toPlotY(dataY);
    if (idx === 0) ctx.moveTo(lineX, lineY);
    else ctx.lineTo(lineX, lineY);
  });
  ctx.stroke();
  ctx.restore();

  if (Array.isArray(spec.point) && spec.point.length >= 2) {
    const pointX = Number(spec.point[0]);
    const pointY = Number(spec.point[1]);
    if (Number.isFinite(pointX) && Number.isFinite(pointY)) {
      const markerX = toPlotX(pointX);
      const markerY = toPlotY(pointY);
      if (markerX >= plotX0 - 1 && markerX <= plotX0 + plotW + 1 && markerY >= plotY0 - 1 && markerY <= plotY0 + plotH + 1) {
        ctx.fillStyle = pointColor;
        ctx.strokeStyle = "rgba(0,0,0,0.75)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(markerX, markerY, 3.4, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    }
  }

  ctx.shadowBlur = 0;
  ctx.fillStyle = textColor;
  ctx.font = `700 ${legendFont}px ${SYSTEM_FONT}`;
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  if (spec.title) ctx.fillText(spec.title, x0 + 6, y0 + 4);
  drawInsetCornerText(ctx, spec.legend, spec.legend_position, x0, y0, boxW, boxH, legendFont, spec.text_color || lineColor);
  drawInsetCornerText(ctx, spec.annotation, spec.annotation_position || "top-right", x0, y0, boxW, boxH, legendFont, textColor);
  ctx.font = `${labelFont}px ${SYSTEM_FONT}`;
  ctx.fillStyle = tickColor;
  if (spec.xlabel) {
    ctx.textAlign = "right";
    ctx.textBaseline = "bottom";
    ctx.fillText(spec.xlabel, x0 + boxW - 7, y0 + boxH - 3);
  }
  if (spec.ylabel) {
    ctx.save();
    ctx.translate(x0 + 5, plotY0 + 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "right";
    ctx.textBaseline = "top";
    ctx.fillText(spec.ylabel, 0, 0);
    ctx.restore();
  }
  if (spec.show_panel_index) {
    ctx.fillStyle = "rgba(255,255,255,0.42)";
    ctx.font = `7px ${SYSTEM_FONT}`;
    ctx.textAlign = "right";
    ctx.textBaseline = "bottom";
    ctx.fillText(`${panel + 1}`, x0 + boxW - 5, y0 + boxH - 4);
  }
  ctx.restore();
}
