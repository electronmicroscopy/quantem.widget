import { type RichTitleSpan } from "./latexTitle";

/** Panel overlays (circles and rectangles in data or relative coordinates), their
 *  hit testing and drag editing, plus the annotation badge styling that show2d
 *  and show3d render on top of each image panel. */
export type PanelAnnotationSpec = {
  text?: string;
  math?: string;
  spans?: RichTitleSpan[];
  position?: string;
  anchor?: string;
  x?: number;
  y?: number;
  box?: [number, number, number, number];
  variant?: string;
  class_name?: string;
  bg?: string;
  fg?: string;
  color?: string;
  border_color?: string;
  border_width?: number;
  font_size?: number;
  font_weight?: string | number;
  font_family?: string;
  pad_x?: number;
  pad_y?: number;
  radius?: number;
  opacity?: number;
  align?: string;
  max_width?: string;
  offset?: [number, number];
  outline_color?: string;
  outline_width?: number;
};
export type PanelOverlaySpec = {
  shape?: "circle" | "rect" | "rectangle" | "square";
  coords?: "data" | "relative";
  row?: number;
  col?: number;
  radius?: number;
  row0?: number;
  col0?: number;
  row1?: number;
  col1?: number;
  stroke?: string;
  stroke_width?: number;
  line_style?: string;
  dash?: number[];
  fill?: string;
  opacity?: number;
  fill_opacity?: number;
  stroke_opacity?: number;
  z_order?: number;
};
export type OverlaySelection = { panel: number; overlay: number };
export type OverlayDragState = {
  mode: "move" | "resize";
  panel: number;
  overlay: number;
  handle?: string;
  startRow: number;
  startCol: number;
  original: PanelOverlaySpec;
};

/** Finite number from a loosely typed spec field, else `fallback`. */
export function styleNumber(value: unknown, fallback: number): number {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

/** Non-blank string from a loosely typed spec field, else `fallback`. */
export function styleString(value: unknown, fallback = ""): string {
  return typeof value === "string" && value.trim() ? value : fallback;
}

/** `color` with `alpha` applied when it is #rgb or #rrggbb; other CSS colors pass through, "none" is undefined. */
function withAlpha(color: string | undefined, alpha: number): string | undefined {
  if (!color || color === "none") return undefined;
  const a = Math.max(0, Math.min(1, alpha));
  if (color.startsWith("#")) {
    const hex = color.slice(1);
    if (hex.length === 3 || hex.length === 6) {
      const full = hex.length === 3 ? hex.split("").map((ch) => ch + ch).join("") : hex;
      const r = parseInt(full.slice(0, 2), 16);
      const g = parseInt(full.slice(2, 4), 16);
      const b = parseInt(full.slice(4, 6), 16);
      if ([r, g, b].every(Number.isFinite)) return `rgba(${r}, ${g}, ${b}, ${a})`;
    }
  }
  return color;
}

/** Canvas dash pattern: an explicit `dash`, else the named line style scaled by the line width. */
export function overlayDashPattern(overlay: PanelOverlaySpec, lineWidth: number): number[] {
  const custom = Array.isArray(overlay.dash)
    ? overlay.dash.map((value) => Number(value)).filter((value) => Number.isFinite(value) && value >= 0)
    : [];
  if (custom.some((value) => value > 0)) return custom;
  const unit = Math.max(1, lineWidth);
  const lineStyle = styleString(overlay.line_style, "solid").toLowerCase().replace("_", "-");
  if (lineStyle === "dashed" || lineStyle === "dash") return [4 * unit, 2 * unit];
  if (lineStyle === "dotted" || lineStyle === "dot") return [unit, 1.8 * unit];
  if (lineStyle === "dashdot" || lineStyle === "dash-dot") return [4 * unit, 2 * unit, unit, 2 * unit];
  return [];
}

export function annotationAnchorTransform(anchor: string | undefined): string {
  const value = anchor || "top-left";
  const x = value.endsWith("center") || value === "center" ? "-50%" : value.endsWith("right") ? "-100%" : "0";
  const y = value.startsWith("center") || value === "center" ? "-50%" : value.startsWith("bottom") ? "-100%" : "0";
  return `translate(${x}, ${y})`;
}

function annotationPositionSx(spec: PanelAnnotationSpec): Record<string, unknown> {
  const margin = 8;
  const position = spec.position || "top-left";
  const offset = Array.isArray(spec.offset) ? spec.offset : [0, 0];
  if (Array.isArray(spec.box) && spec.box.length === 4) {
    const [left, top, width, height] = spec.box;
    return {
      left: `calc(${left * 100}% + ${offset[0] || 0}px)`,
      top: `calc(${top * 100}% + ${offset[1] || 0}px)`,
      width: `${width * 100}%`,
      minHeight: `${height * 100}%`,
    };
  }
  if (Number.isFinite(spec.x) && Number.isFinite(spec.y)) {
    return {
      left: `calc(${Number(spec.x) * 100}% + ${offset[0] || 0}px)`,
      top: `calc(${Number(spec.y) * 100}% + ${offset[1] || 0}px)`,
      transform: annotationAnchorTransform(spec.anchor || "center"),
    };
  }
  const sx: Record<string, unknown> = {};
  if (position.includes("top")) sx.top = margin + (offset[1] || 0);
  if (position.includes("bottom")) sx.bottom = margin - (offset[1] || 0);
  if (position.includes("left")) sx.left = margin + (offset[0] || 0);
  if (position.includes("right")) sx.right = margin - (offset[0] || 0);
  if (position === "top-center" || position === "center" || position === "bottom-center") {
    sx.left = `calc(50% + ${offset[0] || 0}px)`;
  }
  if (position === "center-left" || position === "center" || position === "center-right") {
    sx.top = `calc(50% + ${offset[1] || 0}px)`;
  }
  sx.transform = annotationAnchorTransform(spec.anchor || position);
  return sx;
}

export function panelAnnotationSx(spec: PanelAnnotationSpec, zIndex = 7): Record<string, unknown> {
  const variant = spec.variant || "badge";
  const plain = variant === "plain";
  const outline = variant === "outline";
  const callout = variant === "callout";
  const pill = variant === "pill";
  const fg = styleString(spec.fg ?? spec.color, plain ? "rgba(255,255,255,0.92)" : "#fff");
  const bg = styleString(spec.bg, plain ? "transparent" : "rgba(0,0,0,0.72)");
  const borderWidth = Math.max(0, styleNumber(spec.border_width, outline || callout ? 1 : 0));
  return {
    position: "absolute",
    ...annotationPositionSx(spec),
    display: "block",
    boxSizing: "border-box",
    pointerEvents: "none",
    zIndex,
    px: spec.pad_x != null ? `${Math.max(0, styleNumber(spec.pad_x, 0))}px` : (plain ? 0 : "6px"),
    py: spec.pad_y != null ? `${Math.max(0, styleNumber(spec.pad_y, 0))}px` : (plain ? 0 : "2px"),
    borderRadius: spec.radius != null ? `${Math.max(0, styleNumber(spec.radius, 0))}px` : (pill ? "999px" : "3px"),
    background: bg,
    color: fg,
    border: borderWidth > 0 ? `${borderWidth}px solid ${styleString(spec.border_color, "rgba(255,255,255,0.5)")}` : "none",
    opacity: spec.opacity != null ? Math.max(0, Math.min(1, styleNumber(spec.opacity, 1))) : 1,
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    ...(spec.font_family ? { fontFamily: spec.font_family } : {}),
    fontSize: `${Math.max(6, styleNumber(spec.font_size, 10))}px`,
    fontWeight: spec.font_weight != null ? spec.font_weight : 700,
    lineHeight: 1.2,
    textAlign: styleString(spec.align, "center"),
    whiteSpace: Array.isArray(spec.box) ? "normal" : "nowrap",
    overflow: "hidden",
    textOverflow: "ellipsis",
    maxWidth: styleString(spec.max_width, Array.isArray(spec.box) ? "100%" : "calc(100% - 16px)"),
    textShadow: plain && styleNumber(spec.outline_width, 0) <= 0 ? "0 1px 2px rgba(0,0,0,0.85)" : "none",
    WebkitTextStroke: styleNumber(spec.outline_width, 0) > 0 ? `${styleNumber(spec.outline_width, 0)}px ${styleString(spec.outline_color, "rgba(0,0,0,0.85)")}` : undefined,
    paintOrder: styleNumber(spec.outline_width, 0) > 0 ? "stroke fill" : undefined,
    boxShadow: callout ? "0 1px 4px rgba(0,0,0,0.45)" : "none",
  };
}

export function drawROI(
  ctx: CanvasRenderingContext2D,
  x: number, y: number,
  shape: "circle" | "square" | "rectangle" | "annular",
  radius: number, w: number, h: number,
  activeColor: string, inactiveColor: string,
  active: boolean = false, innerRadius: number = 0
): void {
  // Caller sets ctx.lineWidth from roi.line_width; don't clobber.
  const strokeColor = active ? activeColor : inactiveColor;
  ctx.strokeStyle = strokeColor;
  if (shape === "circle") {
    ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI * 2); ctx.stroke();
  } else if (shape === "square") {
    ctx.strokeRect(x - radius, y - radius, radius * 2, radius * 2);
  } else if (shape === "rectangle") {
    ctx.strokeRect(x - w / 2, y - h / 2, w, h);
  } else if (shape === "annular") {
    ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI * 2); ctx.stroke();
    ctx.strokeStyle = active ? "#0ff" : inactiveColor;
    ctx.beginPath(); ctx.arc(x, y, innerRadius, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = (active ? activeColor : inactiveColor) + "15";
    ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI * 2); ctx.arc(x, y, innerRadius, 0, Math.PI * 2, true); ctx.fill();
    ctx.strokeStyle = strokeColor;
  }
  if (active) {
    ctx.beginPath();
    ctx.moveTo(x - 5, y); ctx.lineTo(x + 5, y);
    ctx.moveTo(x, y - 5); ctx.lineTo(x, y + 5);
    ctx.stroke();
  }
}

/**
 * Multipliers from spec coordinates to image pixels: 1 for "data" coordinates,
 * the image height, width and shorter side for "relative" ones.
 */
function overlayScales(overlay: PanelOverlaySpec, imageW: number, imageH: number): { scaleRow: number; scaleCol: number; scaleRadius: number } {
  const relative = (overlay.coords || "data") === "relative";
  return {
    scaleRow: relative ? imageH : 1,
    scaleCol: relative ? imageW : 1,
    scaleRadius: relative ? Math.min(imageW, imageH) : 1,
  };
}

/**
 * Screen center and radius of a circle overlay. The radius is the mean of the
 * row and column screen radii, so a non-square pixel aspect still draws a circle.
 */
function screenCircle(
  circle: { row: number; col: number; radius: number },
  toScreenX: (col: number) => number,
  toScreenY: (row: number) => number,
): { x: number; y: number; radius: number } {
  const x = toScreenX(circle.col);
  const y = toScreenY(circle.row);
  const radius = Math.max(0, (Math.abs(toScreenX(circle.col + circle.radius) - x) + Math.abs(toScreenY(circle.row + circle.radius) - y)) / 2);
  return { x, y, radius };
}

/** Stroke and fill every overlay in z_order, mapping image pixels to the screen through the callbacks. */
export function drawPanelOverlays(
  ctx: CanvasRenderingContext2D,
  overlays: PanelOverlaySpec[] | undefined,
  toScreenX: (col: number) => number,
  toScreenY: (row: number) => number,
  imageW: number,
  imageH: number,
): void {
  if (!overlays?.length) return;
  const ordered = [...overlays].sort((a, b) => styleNumber(a.z_order, 0) - styleNumber(b.z_order, 0));
  for (const overlay of ordered) {
    const opacity = styleNumber(overlay.opacity, 1);
    const strokeOpacity = opacity * styleNumber(overlay.stroke_opacity, 1);
    const fillOpacity = opacity * styleNumber(overlay.fill_opacity, overlay.fill ? 1 : 0);
    const stroke = withAlpha(styleString(overlay.stroke, "#00e5ff"), strokeOpacity);
    const fill = withAlpha(overlay.fill, fillOpacity);
    ctx.save();
    ctx.lineWidth = Math.max(0, styleNumber(overlay.stroke_width, 2));
    ctx.setLineDash(overlayDashPattern(overlay, ctx.lineWidth));
    ctx.lineCap = ctx.getLineDash().length ? "round" : "butt";
    if (fill) ctx.fillStyle = fill;
    if (stroke) ctx.strokeStyle = stroke;
    const geom = overlayGeometry(overlay, imageW, imageH);
    if (geom.shape === "circle") {
      const circle = screenCircle(geom, toScreenX, toScreenY);
      ctx.beginPath();
      ctx.arc(circle.x, circle.y, circle.radius, 0, Math.PI * 2);
      if (fill) ctx.fill();
      if (stroke && ctx.lineWidth > 0) ctx.stroke();
    } else {
      const x0 = toScreenX(geom.col0);
      const y0 = toScreenY(geom.row0);
      const x1 = toScreenX(geom.col1);
      const y1 = toScreenY(geom.row1);
      const x = Math.min(x0, x1);
      const y = Math.min(y0, y1);
      const width = Math.abs(x1 - x0);
      const height = Math.abs(y1 - y0);
      if (fill) ctx.fillRect(x, y, width, height);
      if (stroke && ctx.lineWidth > 0) ctx.strokeRect(x, y, width, height);
    }
    ctx.restore();
  }
}

/** Overlay in image pixels: a circle, or a rectangle with sorted corners (anything not a circle). */
export function overlayGeometry(overlay: PanelOverlaySpec, imageW: number, imageH: number) {
  const { scaleRow, scaleCol, scaleRadius } = overlayScales(overlay, imageW, imageH);
  const shape = overlay.shape === "rectangle" ? "rect" : (overlay.shape || "circle");
  if (shape === "circle") {
    return {
      shape,
      row: styleNumber(overlay.row, 0) * scaleRow,
      col: styleNumber(overlay.col, 0) * scaleCol,
      radius: Math.max(0, styleNumber(overlay.radius, 0) * scaleRadius),
    };
  }
  const row0 = styleNumber(overlay.row0, 0) * scaleRow;
  const col0 = styleNumber(overlay.col0, 0) * scaleCol;
  const row1 = styleNumber(overlay.row1, row0) * scaleRow;
  const col1 = styleNumber(overlay.col1, col0) * scaleCol;
  return {
    shape,
    row0: Math.min(row0, row1),
    col0: Math.min(col0, col1),
    row1: Math.max(row0, row1),
    col1: Math.max(col0, col1),
  };
}

/**
 * Topmost overlay under (row, col): resize when within `hitRadius` of a circle
 * rim or rectangle edge (with the touched edges as the handle), else move when inside.
 */
export function panelOverlayHit(
  overlays: PanelOverlaySpec[] | undefined,
  row: number,
  col: number,
  imageW: number,
  imageH: number,
  hitRadius: number,
): { overlay: number; mode: "move" | "resize"; handle?: string } | null {
  if (!overlays?.length) return null;
  const ordered = overlays.map((overlay, index) => ({ overlay, index })).sort((a, b) => styleNumber(a.overlay.z_order, 0) - styleNumber(b.overlay.z_order, 0));
  for (let orderIdx = ordered.length - 1; orderIdx >= 0; orderIdx -= 1) {
    const { overlay, index } = ordered[orderIdx];
    const geom = overlayGeometry(overlay, imageW, imageH);
    if (geom.shape === "circle") {
      const dist = Math.hypot(col - geom.col, row - geom.row);
      if (Math.abs(dist - geom.radius) <= hitRadius) return { overlay: index, mode: "resize" };
      if (dist <= geom.radius) return { overlay: index, mode: "move" };
      continue;
    }
    const inside = col >= geom.col0 - hitRadius && col <= geom.col1 + hitRadius && row >= geom.row0 - hitRadius && row <= geom.row1 + hitRadius;
    if (!inside) continue;
    const nearLeft = Math.abs(col - geom.col0) <= hitRadius;
    const nearRight = Math.abs(col - geom.col1) <= hitRadius;
    const nearTop = Math.abs(row - geom.row0) <= hitRadius;
    const nearBottom = Math.abs(row - geom.row1) <= hitRadius;
    if (nearLeft || nearRight || nearTop || nearBottom) {
      return {
        overlay: index,
        mode: "resize",
        handle: `${nearTop ? "t" : ""}${nearBottom ? "b" : ""}${nearLeft ? "l" : ""}${nearRight ? "r" : ""}` || "br",
      };
    }
    return { overlay: index, mode: "move" };
  }
  return null;
}

/** Overlay spec after a move or resize drag, clamped to the image and kept in the spec's coordinates. */
export function updateOverlayFromDrag(
  original: PanelOverlaySpec,
  mode: "move" | "resize",
  startRow: number,
  startCol: number,
  row: number,
  col: number,
  imageW: number,
  imageH: number,
  handle = "br",
): PanelOverlaySpec {
  const { scaleRow, scaleCol, scaleRadius } = overlayScales(original, imageW, imageH);
  const toSpecRow = (value: number) => value / scaleRow;
  const toSpecCol = (value: number) => value / scaleCol;
  const toSpecRadius = (value: number) => value / scaleRadius;
  const geom = overlayGeometry(original, imageW, imageH);
  const next = { ...original };
  if (geom.shape === "circle") {
    if (mode === "move") {
      next.row = toSpecRow(Math.max(0, Math.min(imageH, geom.row + row - startRow)));
      next.col = toSpecCol(Math.max(0, Math.min(imageW, geom.col + col - startCol)));
    } else {
      next.radius = toSpecRadius(Math.max(1, Math.hypot(col - geom.col, row - geom.row)));
    }
    return next;
  }
  let row0 = geom.row0;
  let row1 = geom.row1;
  let col0 = geom.col0;
  let col1 = geom.col1;
  if (mode === "move") {
    const rowDelta = row - startRow;
    const colDelta = col - startCol;
    const rectHeight = row1 - row0;
    const rectWidth = col1 - col0;
    row0 = Math.max(0, Math.min(imageH - rectHeight, row0 + rowDelta));
    row1 = row0 + rectHeight;
    col0 = Math.max(0, Math.min(imageW - rectWidth, col0 + colDelta));
    col1 = col0 + rectWidth;
  } else {
    if (handle.includes("t")) row0 = row;
    if (handle.includes("b") || (!handle.includes("t") && !handle.includes("l") && !handle.includes("r"))) row1 = row;
    if (handle.includes("l")) col0 = col;
    if (handle.includes("r") || (!handle.includes("t") && !handle.includes("b") && !handle.includes("l"))) col1 = col;
    if (Math.abs(row1 - row0) < 1) row1 = row0 + (row1 >= row0 ? 1 : -1);
    if (Math.abs(col1 - col0) < 1) col1 = col0 + (col1 >= col0 ? 1 : -1);
  }
  next.row0 = toSpecRow(Math.max(0, Math.min(imageH, Math.min(row0, row1))));
  next.row1 = toSpecRow(Math.max(0, Math.min(imageH, Math.max(row0, row1))));
  next.col0 = toSpecCol(Math.max(0, Math.min(imageW, Math.min(col0, col1))));
  next.col1 = toSpecCol(Math.max(0, Math.min(imageW, Math.max(col0, col1))));
  return next;
}

/** Dashed white outline 3 px outside the selected overlay. */
export function drawPanelOverlaySelection(
  ctx: CanvasRenderingContext2D,
  overlay: PanelOverlaySpec | undefined,
  toScreenX: (col: number) => number,
  toScreenY: (row: number) => number,
  imageW: number,
  imageH: number,
): void {
  if (!overlay) return;
  const geom = overlayGeometry(overlay, imageW, imageH);
  ctx.save();
  ctx.setLineDash([5, 3]);
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 1.5;
  if (geom.shape === "circle") {
    const circle = screenCircle(geom, toScreenX, toScreenY);
    ctx.beginPath();
    ctx.arc(circle.x, circle.y, circle.radius + 3, 0, Math.PI * 2);
    ctx.stroke();
  } else {
    const x0 = toScreenX(geom.col0);
    const y0 = toScreenY(geom.row0);
    const x1 = toScreenX(geom.col1);
    const y1 = toScreenY(geom.row1);
    ctx.strokeRect(Math.min(x0, x1) - 3, Math.min(y0, y1) - 3, Math.abs(x1 - x0) + 6, Math.abs(y1 - y0) + 6);
  }
  ctx.setLineDash([]);
  ctx.restore();
}

/** Valid, sorted, de-duplicated hidden page slots; at least one slot always stays visible. */
export function normalizeHiddenPageSlots(values: unknown, maxSlots: number): number[] {
  const nSlots = Math.max(0, Math.trunc(Number(maxSlots) || 0));
  if (!Array.isArray(values) || nSlots <= 1) return [];
  const clean = new Set<number>();
  for (const value of values) {
    const slot = Math.trunc(Number(value));
    if (Number.isFinite(slot) && slot >= 0 && slot < nSlots) clean.add(slot);
  }
  const sorted = Array.from(clean).sort((a, b) => a - b);
  if (sorted.length >= nSlots) sorted.pop();
  return sorted;
}
