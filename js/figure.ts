/**
 * Shared scale bar, colorbar, and overlay utilities for all canvas-based widgets.
 * Provides HiDPI-aware rendering with automatic unit conversion.
 */

import { formatNumber } from "./format";

/** Round a physical value to a "nice" number (1, 2, 5, 10, 20, 50, ...) */
export function roundToNiceValue(value: number): number {
  if (value <= 0) return 1;
  const magnitude = Math.pow(10, Math.floor(Math.log10(value)));
  const normalized = value / magnitude;
  if (normalized < 1.5) return magnitude;
  if (normalized < 3.5) return 2 * magnitude;
  if (normalized < 7.5) return 5 * magnitude;
  return 10 * magnitude;
}

/**
 * Normalize a unit string to its scientific symbol for DISPLAY only. Users pass
 * units like "micron"/"um" on a Dataset; we keep the trait verbatim but render
 * the conventional glyph (µm, Å) so labels read like a journal figure. Unknown
 * strings pass through unchanged. Case-insensitive on the spelled-out forms.
 */
export function unitSymbol(unit: string): string {
  const trimmed = (unit || "").trim();
  const lower = trimmed.toLowerCase();
  if (lower === "micron" || lower === "microns" || lower === "um" || trimmed === "μm" || trimmed === "µm") return "µm";
  if (lower === "angstrom" || lower === "angstroms" || lower === "ang" || trimmed === "Å" || lower === "a") return "Å";
  if (lower === "nanometer" || lower === "nanometers" || lower === "nm") return "nm";
  if (lower === "picometer" || lower === "picometers" || lower === "pm") return "pm";
  if (lower === "millimeter" || lower === "millimeters" || lower === "mm") return "mm";
  if (lower === "picosecond" || lower === "picoseconds" || lower === "ps") return "ps";
  if (lower === "femtosecond" || lower === "femtoseconds" || lower === "fs") return "fs";
  if (lower === "nanosecond" || lower === "nanoseconds" || lower === "ns") return "ns";
  return trimmed;
}

// Length-unit ladder for the scale bar, each as its size in nm. Lets a sub-1 value
// in one unit (e.g. 0.5 nm) display as a clean integer in a smaller unit (500 pm /
// 5 A) instead of a decimal - microscopists read "5 A", not "0.50 nm".
const LENGTH_UNITS_NM: { sym: string; nm: number }[] = [
  { sym: "mm", nm: 1e6 }, { sym: "µm", nm: 1e3 }, { sym: "nm", nm: 1 }, { sym: "Å", nm: 0.1 }, { sym: "pm", nm: 1e-3 },
];
// Base unit (the trait's unit) -> nm. Only length units rescale; anything else
// (mrad, ps, px, ...) keeps its own unit and the old decimal fallback.
const BASE_UNIT_NM: Record<string, number> = {
  mm: 1e6, "µm": 1e3, "μm": 1e3, micron: 1e3, microns: 1e3, um: 1e3,
  nm: 1, nanometer: 1, nanometers: 1, "å": 0.1, angstrom: 0.1, angstroms: 0.1, ang: 0.1, a: 0.1, pm: 1e-3, picometer: 1e-3, picometers: 1e-3,
};

/** Format scale bar label. Length values auto-pick the unit that reads as a clean
 *  integer (no decimals) - 0.5 nm -> "5 Å", 0.005 nm -> "5 pm". Non-length units
 *  (mrad, ps, px) keep their unit. roundToNiceValue gives n×10^k, and every ladder
 *  step is a power of 10, so the rescaled number is always exact. */
export function formatScaleLabel(value: number, unit: string): string {
  const nice = roundToNiceValue(value);
  const baseNm = BASE_UNIT_NM[(unit || "").trim().toLowerCase()];
  if (baseNm === undefined) {
    // not a length unit - keep the unit, fall back to integer-or-decimal
    const sym = unitSymbol(unit);
    return nice >= 1 ? `${Math.round(nice)} ${sym}` : `${nice.toFixed(2)} ${sym}`;
  }
  const valueNm = nice * baseNm;
  // largest ladder unit where the value is >= 1 -> the cleanest (fewest-digit) integer
  const pick = LENGTH_UNITS_NM.find((ladderUnit) => valueNm / ladderUnit.nm >= 1) ?? LENGTH_UNITS_NM[LENGTH_UNITS_NM.length - 1];
  return `${Math.round(valueNm / pick.nm)} ${pick.sym}`;
}

const FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";

/** Format a zoom multiplier consistently across image and FFT overlays. */
export function formatZoomLabel(zoom: number): string {
  return `${zoom.toFixed(1)}×`;
}

type ScaleBarRegion = { x: number; y: number; width: number; height: number };

/** How the label (and zoom indicator) is lifted off the image: a soft canvas
 *  shadow, or a hard copy drawn 1 px down-right in the shadow color. */
type ScaleBarTextShadow = { kind: "blur" | "offset"; color: string };

export interface ScaleBarOptions {
  position?: "bottom-right" | "bottom-left";
  showZoomIndicator?: boolean;
  /** Bar length the eye reads easily, before rounding to a nice physical value. */
  targetBarPx?: number;
  /** Caps the target at this fraction of the region width so small panels keep a bar that fits. */
  maxBarFraction?: number;
  /** Explicit bar length in `unit`; skips the nice rounding. */
  physicalLength?: number | null;
  /** Explicit label text in place of the formatted physical length. */
  label?: string | null;
  barThickness?: number;
  margin?: number;
  /** Gap between the bar top and the label baseline. */
  labelGap?: number;
  /** Shift applied to the bar and its label, not to the zoom indicator. */
  offset?: [number, number];
  font?: string;
  color?: string;
  textShadow?: ScaleBarTextShadow;
  /** Half-transparent copy of the bar drawn 1 px down-right; off by default. */
  barShadowColor?: string | null;
  /** Stroke drawn behind the label instead of the text shadow. */
  outline?: { color: string; width: number } | null;
  /** Distance from the region bottom to the zoom indicator baseline; defaults to margin - barThickness. */
  zoomIndicatorGap?: number;
}

const DEFAULT_TEXT_SHADOW: ScaleBarTextShadow = { kind: "blur", color: "rgba(0, 0, 0, 0.5)" };

/**
 * Bar placement in CSS pixels inside `region`. `effectiveZoom` is screen pixels
 * per image pixel. Returns null when the inputs cannot produce a finite bar.
 */
export function scaleBarGeometry(
  region: ScaleBarRegion,
  effectiveZoom: number,
  pixelSize: number,
  unit: string,
  options: ScaleBarOptions = {},
): { barX: number; barY: number; barPx: number; barHeight: number; label: string; scaleLeft: boolean } | null {
  if (region.width <= 0 || region.height <= 0 || pixelSize <= 0 || !(effectiveZoom > 0) || !Number.isFinite(effectiveZoom)) return null;
  const margin = options.margin ?? 12;
  const targetBarPx = options.maxBarFraction != null
    ? Math.min(options.targetBarPx ?? 60, region.width * options.maxBarFraction)
    : (options.targetBarPx ?? 60);
  const explicitPhysical = Number(options.physicalLength);
  const nicePhysical = Number.isFinite(explicitPhysical) && explicitPhysical > 0
    ? explicitPhysical
    : roundToNiceValue((targetBarPx / effectiveZoom) * pixelSize);
  const barPx = (nicePhysical / pixelSize) * effectiveZoom;
  const scaleLeft = (options.position || "bottom-right") === "bottom-left";
  const [offsetX, offsetY] = options.offset ?? [0, 0];
  return {
    barX: region.x + (scaleLeft ? margin : region.width - barPx - margin) + offsetX,
    barY: region.y + region.height - margin + offsetY,
    barPx,
    barHeight: options.barThickness ?? 5,
    label: options.label && options.label.trim() ? options.label : formatScaleLabel(nicePhysical, unit),
    scaleLeft,
  };
}

function applyTextShadow(ctx: CanvasRenderingContext2D, shadow: ScaleBarTextShadow): void {
  if (shadow.kind !== "blur") return;
  ctx.shadowColor = shadow.color;
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;
}

function clearShadow(ctx: CanvasRenderingContext2D): void {
  ctx.shadowColor = "transparent";
  ctx.shadowBlur = 0;
  ctx.shadowOffsetX = 0;
  ctx.shadowOffsetY = 0;
}

/** Zoom multiplier in the corner opposite the scale bar, on the bar's baseline. */
export function drawZoomIndicatorInRegion(
  ctx: CanvasRenderingContext2D,
  region: ScaleBarRegion,
  zoom: number,
  options: ScaleBarOptions = {},
): void {
  const margin = options.margin ?? 12;
  const barThickness = options.barThickness ?? 5;
  const scaleLeft = (options.position || "bottom-right") === "bottom-left";
  const textShadow = options.textShadow ?? DEFAULT_TEXT_SHADOW;
  const color = options.color ?? "white";
  const zoomX = region.x + (scaleLeft ? region.width - margin : margin);
  const zoomY = region.y + region.height - (options.zoomIndicatorGap ?? (margin - barThickness));
  ctx.save();
  ctx.font = options.font ?? `16px ${FONT}`;
  ctx.textAlign = scaleLeft ? "right" : "left";
  ctx.textBaseline = "bottom";
  applyTextShadow(ctx, textShadow);
  const text = formatZoomLabel(zoom);
  if (textShadow.kind === "offset") {
    ctx.fillStyle = textShadow.color;
    ctx.fillText(text, zoomX + 1, zoomY + 1);
  }
  ctx.fillStyle = color;
  ctx.fillText(text, zoomX, zoomY);
  ctx.restore();
}

/**
 * Draw the scale bar (and optional zoom indicator) inside a CSS-pixel region of
 * an already DPR-scaled context. Every widget panel, export frame and overlay
 * canvas goes through here so the bar looks the same everywhere.
 */
export function drawScaleBarInRegion(
  ctx: CanvasRenderingContext2D,
  region: ScaleBarRegion,
  zoom: number,
  effectiveZoom: number,
  pixelSize: number,
  unit: string,
  options: ScaleBarOptions = {},
): void {
  const geom = scaleBarGeometry(region, effectiveZoom, pixelSize, unit, options);
  if (geom) {
    const color = options.color ?? "white";
    const textShadow = options.textShadow ?? DEFAULT_TEXT_SHADOW;
    const labelGap = options.labelGap ?? 4;
    ctx.save();
    clearShadow(ctx);
    if (options.barShadowColor) {
      ctx.fillStyle = options.barShadowColor;
      ctx.globalAlpha = 0.5;
      ctx.fillRect(geom.barX + 1, geom.barY + 1, geom.barPx, geom.barHeight);
      ctx.globalAlpha = 1;
    }
    ctx.fillStyle = color;
    ctx.fillRect(geom.barX, geom.barY, geom.barPx, geom.barHeight);
    applyTextShadow(ctx, textShadow);
    ctx.font = options.font ?? `16px ${FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    const labelX = geom.barX + geom.barPx / 2;
    const labelY = geom.barY - labelGap;
    if (options.outline && options.outline.width > 0) {
      ctx.lineJoin = "round";
      ctx.strokeStyle = options.outline.color;
      ctx.lineWidth = options.outline.width;
      ctx.strokeText(geom.label, labelX, labelY);
    } else if (textShadow.kind === "offset") {
      ctx.fillStyle = textShadow.color;
      ctx.fillText(geom.label, labelX + 1, labelY + 1);
    }
    ctx.fillStyle = color;
    ctx.fillText(geom.label, labelX, labelY);
    ctx.restore();
  }
  if (options.showZoomIndicator === true) drawZoomIndicatorInRegion(ctx, region, zoom, options);
}

/**
 * Draw the scale bar across a whole DPR-scaled canvas, in CSS pixels. `zoom`
 * multiplies the fit-to-width scale cssWidth / imageWidth to give screen pixels
 * per image pixel.
 */
function drawScaleBarOnCanvas(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  dpr: number,
  zoom: number,
  pixelSize: number,
  unit: string,
  imageWidth: number,
  options: ScaleBarOptions,
): void {
  ctx.save();
  ctx.scale(dpr, dpr);
  const cssWidth = canvas.width / dpr;
  const cssHeight = canvas.height / dpr;
  const effectiveZoom = zoom * (cssWidth / imageWidth);
  drawScaleBarInRegion(ctx, { x: 0, y: 0, width: cssWidth, height: cssHeight }, zoom, effectiveZoom, pixelSize, unit, options);
  ctx.restore();
}

/**
 * Draw scale bar and zoom indicator on a high-DPI UI canvas.
 * Renders crisp text/lines independent of the image resolution.
 */
export function drawScaleBarHiDPI(
  canvas: HTMLCanvasElement,
  dpr: number,
  zoom: number,
  pixelSize: number,
  unit: string,
  imageWidth: number,
  options: {
    position?: "bottom-right" | "bottom-left";
    showZoomIndicator?: boolean;
  } = {},
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawScaleBarOnCanvas(ctx, canvas, dpr, zoom, pixelSize, unit, imageWidth, options);
}

/**
 * Draw reciprocal-space scale bar on an FFT overlay canvas.
 * Only draws when fftPixelSize > 0 (i.e. real-space calibration is available).
 */
export function drawFFTScaleBarHiDPI(
  canvas: HTMLCanvasElement,
  dpr: number,
  fftZoom: number,
  fftPixelSize: number,
  imageWidth: number,
  unit: string = "1/px",
  showZoomIndicator: boolean = false,
) {
  const ctx = canvas.getContext("2d");
  if (!ctx || fftPixelSize <= 0) return;
  drawScaleBarOnCanvas(ctx, canvas, dpr, fftZoom, fftPixelSize, unit, imageWidth, { showZoomIndicator: showZoomIndicator === true });
}

/**
 * Draw a vertical colorbar on a canvas context (already DPR-scaled by caller).
 * Gradient strip on right edge with vmin/vmax labels and optional log indicator.
 */
export function drawColorbar(
  ctx: CanvasRenderingContext2D,
  cssW: number,
  cssH: number,
  lut: Uint8Array,
  vmin: number,
  vmax: number,
  logScale: boolean,
) {
  const barW = 12;
  const barH = Math.round(cssH * 0.6);
  const barX = cssW - barW - 12;
  const barY = Math.round((cssH - barH) / 2);

  // Gradient strip (bottom=vmin, top=vmax)
  for (let row = 0; row < barH; row++) {
    const fraction = 1 - row / (barH - 1);
    const entry = Math.round(fraction * 255) * 3;
    ctx.fillStyle = `rgb(${lut[entry]},${lut[entry + 1]},${lut[entry + 2]})`;
    ctx.fillRect(barX, barY + row, barW, 1);
  }
  ctx.strokeStyle = "rgba(255,255,255,0.5)";
  ctx.lineWidth = 1;
  ctx.strokeRect(barX, barY, barW, barH);
  // The drop shadow keeps the labels readable over any image.
  ctx.shadowColor = "rgba(0, 0, 0, 0.7)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;
  ctx.font = `11px ${FONT}`;
  ctx.fillStyle = "white";
  ctx.textAlign = "right";
  ctx.textBaseline = "bottom";
  ctx.fillText(formatNumber(vmax), barX - 4, barY + 6);
  ctx.textBaseline = "top";
  ctx.fillText(formatNumber(vmin), barX - 4, barY + barH - 4);
  if (logScale) {
    ctx.textBaseline = "middle";
    ctx.fillText("log", barX - 4, barY + barH / 2);
  }
}
