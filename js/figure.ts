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
  const u = (unit || "").trim();
  const lc = u.toLowerCase();
  if (lc === "micron" || lc === "microns" || lc === "um" || u === "μm" || u === "µm") return "µm";
  if (lc === "angstrom" || lc === "angstroms" || lc === "ang" || u === "Å" || lc === "a") return "Å";
  if (lc === "nanometer" || lc === "nanometers" || lc === "nm") return "nm";
  if (lc === "picometer" || lc === "picometers" || lc === "pm") return "pm";
  if (lc === "millimeter" || lc === "millimeters" || lc === "mm") return "mm";
  if (lc === "picosecond" || lc === "picoseconds" || lc === "ps") return "ps";
  if (lc === "femtosecond" || lc === "femtoseconds" || lc === "fs") return "fs";
  if (lc === "nanosecond" || lc === "nanoseconds" || lc === "ns") return "ns";
  return u;
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
  const pick = LENGTH_UNITS_NM.find((u) => valueNm / u.nm >= 1) ?? LENGTH_UNITS_NM[LENGTH_UNITS_NM.length - 1];
  return `${Math.round(valueNm / pick.nm)} ${pick.sym}`;
}

const FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";

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
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();
  ctx.scale(dpr, dpr);

  const cssWidth = canvas.width / dpr;
  const cssHeight = canvas.height / dpr;
  const scaleX = cssWidth / imageWidth;
  const effectiveZoom = zoom * scaleX;

  const targetBarPx = 60;
  const barThickness = 5;
  const fontSize = 16;
  const margin = 12;

  const targetPhysical = (targetBarPx / effectiveZoom) * pixelSize;
  const nicePhysical = roundToNiceValue(targetPhysical);
  const barPx = (nicePhysical / pixelSize) * effectiveZoom;

  const barY = cssHeight - margin;
  const barX = cssWidth - barPx - margin;

  ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;

  ctx.fillStyle = "white";
  ctx.fillRect(barX, barY, barPx, barThickness);

  const label = formatScaleLabel(nicePhysical, unit);
  ctx.font = `${fontSize}px ${FONT}`;
  ctx.fillStyle = "white";
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText(label, barX + barPx / 2, barY - 4);

  ctx.textAlign = "left";
  ctx.textBaseline = "bottom";
  ctx.fillText(`${zoom.toFixed(1)}×`, margin, cssHeight - margin + barThickness);

  ctx.restore();
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
) {
  const ctx = canvas.getContext("2d");
  if (!ctx || fftPixelSize <= 0) return;

  ctx.save();
  ctx.scale(dpr, dpr);

  const cssWidth = canvas.width / dpr;
  const cssHeight = canvas.height / dpr;
  const scaleX = cssWidth / imageWidth;
  const effectiveZoom = fftZoom * scaleX;

  const targetBarPx = 60;
  const barThickness = 5;
  const fontSize = 16;
  const margin = 12;

  const targetPhysical = (targetBarPx / effectiveZoom) * fftPixelSize;
  const nicePhysical = roundToNiceValue(targetPhysical);
  const barPx = (nicePhysical / fftPixelSize) * effectiveZoom;

  const barY = cssHeight - margin;
  const barX = cssWidth - barPx - margin;

  ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;

  ctx.fillStyle = "white";
  ctx.fillRect(barX, barY, barPx, barThickness);

  const label = formatScaleLabel(nicePhysical, unit);
  ctx.font = `${fontSize}px ${FONT}`;
  ctx.fillStyle = "white";
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText(label, barX + barPx / 2, barY - 4);

  ctx.textAlign = "left";
  ctx.textBaseline = "bottom";
  ctx.fillText(`${fftZoom.toFixed(1)}×`, margin, cssHeight - margin + barThickness);

  ctx.restore();
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
    const t = 1 - row / (barH - 1);
    const lutIdx = Math.round(t * 255);
    const r = lut[lutIdx * 3];
    const g = lut[lutIdx * 3 + 1];
    const b = lut[lutIdx * 3 + 2];
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(barX, barY + row, barW, 1);
  }

  // Border
  ctx.strokeStyle = "rgba(255,255,255,0.5)";
  ctx.lineWidth = 1;
  ctx.strokeRect(barX, barY, barW, barH);

  // Labels with drop shadow
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

