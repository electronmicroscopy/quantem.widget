import * as React from "react";
import { createRender, useModelState, useModel } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Button from "@mui/material/Button";
import Switch from "@mui/material/Switch";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Menu from "@mui/material/Menu";
import Slider, { type SliderProps } from "@mui/material/Slider";
import CircularProgress from "@mui/material/CircularProgress";
import { alpha } from "@mui/material/styles";
import { useTheme } from "../theme";
import { useHideStaticFallback } from "../staticFallback";
import { COLORMAPS, renderToOffscreen, renderToOffscreenReuse } from "../display/colormaps";
import { findDataRange, percentileClip, applyLogScale } from "../display/stats";
import {
  downloadBlob,
  extractBytes,
  extractFloat32,
  formatNumber,
  preserveRestoredWidgetModelsOnSave,
} from "../format";
import { exportTitleSlug, formatEstimatedHtmlSize, formatSavedBytes } from "../shared/exportFormat";
import { computeGr, dampOriginOscillations, computeGofR, kWindow, resolveWindowBounds } from "./pdf_compute";
import {
  type LineEnds,
  effectiveLineEnds,
  lineBandColumns,
  lineBandIncludes,
  linescanSampleCount,
  linescanSampleIndex,
  sampleFraction,
  sampledBinIndices,
} from "./linescan";
import { committedFitEnd, committedWindowEnd } from "./kwindow";
import { roundToNiceValue } from "../figure";
import { drawNavScaleBar } from "./scalebar";
import { rCutSlider } from "./rcut";

// ============================================================================
// Constants
// ============================================================================
const DPR = Math.max(1, window.devicePixelRatio || 1);
const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 };
const FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
const MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, monospace";
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 32;
const CMAP_OPTIONS = ["gray", "inferno", "viridis", "magma"] as const;

// Canvas sizes (CSS px); the nav image is NAV_SIZE wide.
const PLOT_W = 520;
const PLOT_H = 360;
const NAV_SIZE = PLOT_H;

// Plot margins and ticks (CSS px).
const MARGIN_TOP = 12;
const MARGIN_RIGHT = 16;
const MARGIN_BOTTOM = 56;
const MARGIN_LEFT_MIN = 72;
const AXIS_TICK_PX = 4;
const TICK_LABEL_W = 50;
const PLOT_AREA_W = PLOT_W - MARGIN_LEFT_MIN - MARGIN_RIGHT;
const PLOT_AREA_H = PLOT_H - MARGIN_TOP - MARGIN_BOTTOM;

const TRACE_COLORS = {
  ik: "#4fc3f7",
  bg: "#ef5350",
  fk: "#66bb6a",
  gr: "#4fc3f7",
  pdf: "#ba68c8",
};

// Status hints, the same text as _DENSITY_HINT_DAMPING, _DENSITY_HINT_GR and _WINDOW_HINT in showpdf.py.
const DENSITY_HINT_DAMPING = "Set the density to apply damping and compute g(r).";
const DENSITY_HINT_GR = "Set the density to compute g(r).";
// An empty k window makes F(k) and G(r) zero up to a float32 residue at k max (g(r) about 1).
const WINDOW_HINT = "The k window contains no k points; set its lower end below its upper end, more than one k step apart.";
// Status lines of a failed compute.
const ERROR_STATUS = /^(Error|Line-scan error|g\(r\) error)/;

// Options of the button groups: value, then label (and the mask tools' tooltip).
const ANALYSIS_MODES = ["mask", "probe", "line"] as const;
const MASK_TOOLS = [
  ["rectangle", "▭", "Rectangle"],
  ["circle", "○", "Circle"],
  ["freeform", "✎", "Freeform"],
] as const;
const LINE_MODES = [["averaged", "avg"], ["linescan", "scan"]] as const;
const PLOT_MODES = [["Ik", "I(k)"], ["Fk", "F(k)"], ["Gr", "G(r)"], ["gr", "g(r)"]] as const;
const LINESCAN_VIEWS = [["heatmap", "Heatmap"], ["waterfall", "Waterfall"]] as const;

// ============================================================================
// Style tokens
// ============================================================================
const compactBtn = { fontSize: 12, minWidth: 36, px: 1, py: 0.25, textTransform: "none" as const };
const switchSmall = {
  "& .MuiSwitch-thumb": { width: 12, height: 12 },
  "& .MuiSwitch-switchBase": { padding: "4px" },
};
const sliderSmall = { "& .MuiSlider-thumb": { width: 10, height: 10 } };
const parameterSliderSx = { flex: 1, "& .MuiSlider-thumb": { width: 10, height: 10 } };
const numericSx = { fontVariantNumeric: "tabular-nums", whiteSpace: "nowrap" } as const;
// Keeps a label with its control when a row wraps.
const controlPairSx = {
  display: "inline-flex", alignItems: "center", gap: `${SPACING.SM}px`, flexShrink: 0,
} as const;

// ============================================================================
// Types
// ============================================================================
// Layout of the drawn line-scan plot, to map the pointer to r/k, position and value.
type HeatmapGeometry = {
  rLo: number;   // r/k at the left and right edges
  rHi: number;
  pwH: number;   // plot width without the colorbar's room
  lenUnits: number;   // line length, in Å when calibrated
  N: number;   // samples along the line
  M: number;   // points per curve
  data: Float32Array;   // the N x M stack
  waterfall: boolean;
  wfPad: number;   // waterfall padding at top and bottom
  yLo: number;   // heatmap position range
  yHi: number;
  sel: number[];   // samples drawn in the waterfall
};

// Image of a line's band for the band described by `key`; image null when the band is empty.
type BandOverlay = { key: string; image: HTMLCanvasElement | null; cLo: number; rLo: number };

// Pointer position and axis ranges when a plot pan started.
type PlotPanStart = { mx: number; my: number; xMin: number; xMax: number; yMin: number; yMax: number };

// Whether the scan position (r, c) is selected.
type ScanPredicate = (r: number, c: number) => boolean;

// Values a control just committed, which this render's trait values do not include yet.
type OfflineOver = Partial<{
  kMinFit: number; kMaxFit: number; kMinWindow: number; kMaxWindow: number;
  kLowpass: number; kHighpass: number; rMax: number; rCut: number; maxIter: number;
  damp: boolean; densityValue: number; lineWidth: number;
}>;

// ============================================================================
// Helpers
// ============================================================================
// --- Trait bytes ---
// Float64 trait bytes (the exported exact k axis); null when empty.
function extractFloat64(dataView: DataView): Float64Array | null {
  const bytes = extractBytes(dataView);
  if (bytes.length === 0 || bytes.length % 8 !== 0) return null;
  return new Float64Array(bytes.slice().buffer);
}

// Float32 trait bytes of an array.
function packF32(arr: ArrayLike<number>): DataView {
  const f = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) f[i] = arr[i];
  return new DataView(f.buffer);
}

// Byte length of a trait value that may be unset.
function byteLength(v?: DataView | null): number {
  return v?.byteLength ?? 0;
}

// --- Mask ---
function fillRectMask(
  mask: Uint8Array, w: number, h: number,
  r0: number, c0: number, r1: number, c1: number, value: number,
) {
  const minR = Math.max(0, Math.min(r0, r1));
  const maxR = Math.min(h - 1, Math.max(r0, r1));
  const minC = Math.max(0, Math.min(c0, c1));
  const maxC = Math.min(w - 1, Math.max(c0, c1));
  for (let r = minR; r <= maxR; r++)
    for (let c = minC; c <= maxC; c++)
      mask[r * w + c] = value;
}

function fillCircleMask(
  mask: Uint8Array, w: number, h: number,
  cr: number, cc: number, radius: number, value: number,
) {
  const r2 = radius * radius;
  const rMin = Math.max(0, Math.floor(cr - radius));
  const rMax = Math.min(h - 1, Math.ceil(cr + radius));
  const cMin = Math.max(0, Math.floor(cc - radius));
  const cMax = Math.min(w - 1, Math.ceil(cc + radius));
  for (let r = rMin; r <= rMax; r++)
    for (let c = cMin; c <= cMax; c++)
      if ((r - cr) ** 2 + (c - cc) ** 2 <= r2)
        mask[r * w + c] = value;
}

// Mask decoded from mask_b64 (one byte per scan position); null when it does not fit the scan.
function decodeMask(maskB64: string, n: number): Uint8Array | null {
  if (!maskB64) return null;
  const bin = atob(maskB64);
  if (bin.length !== n) return null;
  const mask = new Uint8Array(n);
  for (let i = 0; i < n; i++) mask[i] = bin.charCodeAt(i);
  return mask;
}

// Mask encoded for mask_b64, one byte per scan position (Bytes traits do not sync JS to Python
// in anywidget).
function encodeMask(mask: Uint8Array): string {
  let binary = "";
  for (let i = 0; i < mask.length; i++) binary += String.fromCharCode(mask[i]);
  return btoa(binary);
}

// --- Scan and line geometry ---
// Scan pixel size in Å; a denormal or absurd one (some datasets carry ~1e-321) counts as uncalibrated.
function isCalibrated(unit: string, pixelSize: number): boolean {
  return unit === "Å" && Number.isFinite(pixelSize) && pixelSize > 1e-6;
}

// Length of a line in scan px.
function lineLength(line: LineEnds): number {
  return Math.hypot(line.r1 - line.r0, line.c1 - line.c0);
}

// Undirected orientation of a line in degrees: 0 horizontal, 90 vertical.
function lineOrientationDeg(line: LineEnds): number {
  const angle = Math.atan2(-(line.r1 - line.r0), line.c1 - line.c0) * 180 / Math.PI;
  return (angle % 180 + 180) % 180;
}

// Pointer position relative to the element's top-left corner (CSS px).
function pointerOffset(el: HTMLElement, e: React.MouseEvent): { x: number; y: number } {
  const rect = el.getBoundingClientRect();
  return { x: e.clientX - rect.left, y: e.clientY - rect.top };
}

// Distance from the point (px, py) to the segment from (ax, ay) to (bx, by).
function distanceToSegment(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax, dy = by - ay;
  const lenSq = dx * dx + dy * dy;
  let t = lenSq > 0 ? ((px - ax) * dx + (py - ay) * dy) / lenSq : 0;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

// A shift of the segment with ends a and b (in [0, n - 1]), limited so both ends stay in range.
function clampShift(shift: number, a: number, b: number, n: number): number {
  return Math.max(-Math.min(a, b), Math.min(n - 1 - Math.max(a, b), shift));
}

// --- Curves, controls and status ---
// The k window in effect: a window end <= 0 follows the fit range, as in calculate_Gr.
function windowInEffect(
  kMinWindow: number, kMaxWindow: number, kMinFit: number, kMaxFit: number,
): [number, number] {
  return [kMinWindow > 0 ? kMinWindow : kMinFit, kMaxWindow > 0 ? kMaxWindow : kMaxFit];
}

// Linear interpolation of the curve (xa ascending, ya) at x, held at its end values outside it.
function interpolate(xa: Float32Array, ya: Float32Array, x: number): number {
  const n = Math.min(xa.length, ya.length);
  if (x <= xa[0]) return ya[0];
  if (x >= xa[n - 1]) return ya[n - 1];
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const m = (lo + hi) >> 1;
    if (xa[m] <= x) lo = m;
    else hi = m;
  }
  const span = xa[hi] - xa[lo];
  const t = span !== 0 ? (x - xa[lo]) / span : 0;
  return ya[lo] + t * (ya[hi] - ya[lo]);
}

// The plot's x axis is r for G(r) and g(r), k for I(k) and F(k).
function isRAxis(plotMode: string): boolean {
  return plotMode === "Gr" || plotMode === "gr";
}

// The plotted curve's name (axis title) and symbol (readout).
function curveLabels(plotMode: string): { name: string; symbol: string } {
  if (plotMode === "Ik") return { name: "I(k)", symbol: "I" };
  if (plotMode === "Fk") return { name: "F(k)", symbol: "F" };
  if (plotMode === "gr") return { name: "g(r)", symbol: "g" };
  return { name: "G(r)", symbol: "G" };
}

// Density input text: empty while unset (0), so the "required" placeholder shows.
function fmtDensity(v: number): string {
  return v > 0 ? String(Number(v.toPrecision(4))) : "";
}

// Status hint while the density is unset and an output needs it.
function densityHint(density: number, damp: boolean, plotMode: string): string {
  if (density > 0) return "";
  if (damp) return DENSITY_HINT_DAMPING;
  if (plotMode === "gr") return DENSITY_HINT_GR;
  return "";
}

// Enter leaves a number field, which commits it.
function blurOnEnter(e: React.KeyboardEvent<HTMLInputElement>) {
  if (e.key === "Enter") e.currentTarget.blur();
}

// --- Offline recompute ---
// Profile row (or column) of scan row (or column) i, with `ratio` scan positions per profile
// position (1 unless the export binned the scan).
function profileIndex(i: number, ratio: number, count: number): number {
  return ratio === 1 ? i : Math.min(count - 1, Math.floor(i / ratio));
}

// Add the quantized profile at `base`, dequantized per k bin as lo + q / 255 * (hi - lo), to acc.
function addProfile(acc: Float64Array, q: Uint8Array, base: number, lo: Float32Array, hi: Float32Array) {
  for (let k = 0; k < acc.length; k++) acc[k] += lo[k] + (q[base + k] / 255) * (hi[k] - lo[k]);
}

// --- Canvas drawing ---
function snap(v: number): number {
  return Math.round(v) + 0.5;
}

function computeTicks(min: number, max: number, maxTicks = 8): number[] {
  const range = max - min;
  if (range <= 0 || !isFinite(range)) return [min, max];
  const step = roundToNiceValue(range / maxTicks);
  if (step <= 0) return [min, max];
  const start = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let v = start; v <= max + step * 0.001; v += step) {
    if (v >= min - step * 0.001) ticks.push(v);
  }
  return ticks.length > 0 ? ticks : [min, max];
}

// Stroke one straight segment with the current style.
function strokeSegment(ctx: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number) {
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
}

// CSS color of the LUT entry at `offset` (its red byte; green and blue follow).
function lutRgb(lut: Uint8Array, offset: number): string {
  return `rgb(${lut[offset]},${lut[offset + 1]},${lut[offset + 2]})`;
}

// LUT offset of the waterfall color of sample `bin` of N: turbo by position along the line, its
// dark ends trimmed on a dark theme so the A and B curves stay visible.
function waterfallColorOffset(bin: number, N: number, isDark: boolean): number {
  const t0 = isDark ? 0.06 : 0.0;
  const t1 = isDark ? 0.94 : 1.0;
  const t = N === 1 ? 0.5 : t0 + (bin / (N - 1)) * (t1 - t0);
  return Math.round(t * 255) * 3;
}

// ============================================================================
// Nav overlay drawing (canvas px)
// ============================================================================
// Zoom and pan about the canvas center, as the nav image is drawn.
function applyNavTransform(
  ctx: CanvasRenderingContext2D, w: number, h: number, panX: number, panY: number, zoom: number,
) {
  ctx.translate(w / 2 + panX * DPR, h / 2 + panY * DPR);
  ctx.scale(zoom, zoom);
  ctx.translate(-w / 2, -h / 2);
}

// Probe footprint: the square of scan positions used, filled, and its inscribed circle.
function drawProbe(
  ctx: CanvasRenderingContext2D, probe: { row: number; col: number }, size: number,
  scX: number, scY: number, zoom: number,
) {
  const half = Math.max(0, size - 1);
  const side = 2 * half + 1;
  const sx = (probe.col - half) * scX;
  const sy = (probe.row - half) * scY;
  const sw = side * scX;
  const sh = side * scY;
  ctx.fillStyle = "rgba(100,200,255,0.18)";
  ctx.fillRect(sx, sy, sw, sh);
  ctx.strokeStyle = "rgba(100,200,255,1.0)";
  ctx.lineWidth = 2 / zoom;
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.ellipse(sx + sw / 2, sy + sh / 2, sw / 2, sh / 2, 0, 0, Math.PI * 2);
  ctx.stroke();
}

// Image of the scan positions a line's band selects, one pixel per position, over the band's
// bounding box; each row tests only the columns that can be in the band.
function renderBandImage(
  line: LineEnds, width: number, scanRows: number, scanCols: number,
): Omit<BandOverlay, "key"> {
  const inBand = lineBandIncludes(line, width);
  if (!inBand) return { image: null, cLo: 0, rLo: 0 };
  const pad = width / 2 + 1;   // past the band's half width
  const rLo = Math.max(0, Math.floor(Math.min(line.r0, line.r1) - pad));
  const rHi = Math.min(scanRows - 1, Math.ceil(Math.max(line.r0, line.r1) + pad));
  const cLo = Math.max(0, Math.floor(Math.min(line.c0, line.c1) - pad));
  const cHi = Math.min(scanCols - 1, Math.ceil(Math.max(line.c0, line.c1) + pad));
  const bw = cHi - cLo + 1;
  const bh = rHi - rLo + 1;
  const image = document.createElement("canvas");
  image.width = bw;
  image.height = bh;
  const bctx = image.getContext("2d")!;
  const pixels = bctx.createImageData(bw, bh);
  for (let r = rLo; r <= rHi; r++) {
    const [lo, hi] = lineBandColumns(line, width, r, scanCols);
    for (let c = Math.max(lo, cLo); c <= Math.min(hi, cHi); c++) {
      if (!inBand(r, c)) continue;
      const i = ((r - rLo) * bw + (c - cLo)) * 4;
      // rgba(100,200,255,0.20)
      pixels.data[i] = 100;
      pixels.data[i + 1] = 200;
      pixels.data[i + 2] = 255;
      pixels.data[i + 3] = 51;
    }
  }
  bctx.putImageData(pixels, 0, 0);
  return { image, cLo, rLo };
}

// The drawn line when ⟂ is on (dimmed, dashed), with square drag handles at its ends.
function drawReferenceLine(
  ctx: CanvasRenderingContext2D, ax: number, ay: number, bx: number, by: number, zoom: number,
) {
  ctx.strokeStyle = "rgba(180,180,180,0.75)";
  ctx.lineWidth = 1.5 / zoom;
  ctx.setLineDash([4 / zoom, 3 / zoom]);
  strokeSegment(ctx, ax, ay, bx, by);
  ctx.setLineDash([]);
  const half = 3.5 / zoom;
  ctx.fillStyle = "rgba(200,200,200,0.95)";
  for (const [x, y] of [[ax, ay], [bx, by]]) {
    ctx.beginPath();
    ctx.rect(x - half, y - half, half * 2, half * 2);
    ctx.fill();
  }
}

// The scan line from A to B, with an arrowhead at B so its direction is unambiguous.
function drawScanLine(
  ctx: CanvasRenderingContext2D, ax: number, ay: number, bx: number, by: number, zoom: number,
) {
  ctx.strokeStyle = "rgba(100,200,255,1.0)";
  ctx.lineWidth = 2 / zoom;
  ctx.lineCap = "butt";
  ctx.setLineDash([]);
  strokeSegment(ctx, ax, ay, bx, by);
  const segLen = Math.hypot(bx - ax, by - ay);
  if (segLen > 0) {
    const ux = (bx - ax) / segLen, uy = (by - ay) / segLen;
    const ah = 9 / zoom;
    const px = -uy, py = ux;   // perpendicular
    ctx.fillStyle = "rgba(255,110,110,1)";
    ctx.beginPath();
    ctx.moveTo(bx, by);
    ctx.lineTo(bx - ah * ux + ah * 0.5 * px, by - ah * uy + ah * 0.5 * py);
    ctx.lineTo(bx - ah * ux - ah * 0.5 * px, by - ah * uy - ah * 0.5 * py);
    ctx.closePath();
    ctx.fill();
  }
}

// Labels A (green) and B (red) at the scan line's start and end, with dots unless the
// waterfall's sampling markers take their place.
function drawLineEnds(
  ctx: CanvasRenderingContext2D, ax: number, ay: number, bx: number, by: number, zoom: number,
  withDots: boolean,
) {
  const hr = 4 / zoom;
  const ends = [
    [ax, ay, "rgba(100,225,140,1)", "A"],
    [bx, by, "rgba(255,110,110,1)", "B"],
  ] as const;
  for (const [ex, ey, color, label] of ends) {
    ctx.fillStyle = color;
    if (withDots) {
      ctx.beginPath();
      ctx.ellipse(ex, ey, hr, hr, 0, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.font = `${12 / zoom}px ${FONT}`;
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";
    ctx.fillText(label, ex + hr + 1 / zoom, ey - hr);
  }
}

// Image of the mask, one pixel per scan position: excluded positions translucent red.
function renderMaskImage(mask: Uint8Array, cols: number, rows: number): HTMLCanvasElement {
  const image = document.createElement("canvas");
  image.width = cols;
  image.height = rows;
  const mctx = image.getContext("2d")!;
  const pixels = mctx.createImageData(cols, rows);
  for (let i = 0; i < mask.length; i++) {
    if (mask[i] === 0) {
      pixels.data[i * 4] = 255;
      pixels.data[i * 4 + 1] = 0;
      pixels.data[i * 4 + 2] = 0;
      pixels.data[i * 4 + 3] = 110;
    }
  }
  mctx.putImageData(pixels, 0, 0);
  return image;
}

// ============================================================================
// Plot drawing (CSS px)
// ============================================================================
// The finite values among the first n of data.
function finiteValues(data: Float32Array, n: number): Float32Array {
  let count = 0;
  for (let i = 0; i < n; i++) if (isFinite(data[i])) count++;
  const finite = new Float32Array(count);
  let j = 0;
  for (let i = 0; i < n; i++) {
    const v = data[i];
    if (isFinite(v)) finite[j++] = v;
  }
  return finite;
}

// Message shown in the plot while there is no line-scan stack to draw.
function linescanPlaceholder(computing: boolean, hasLine: boolean): string {
  if (computing) return "Computing line-scan…";
  if (hasLine) return "No line-scan curves to show (see the status line)";
  return "Draw a line to compute the line-scan";
}

// A gray message centered at (x, y), for a plot with nothing to draw.
function drawPlotMessage(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, isDark: boolean) {
  ctx.fillStyle = isDark ? "#888" : "#777";
  ctx.font = `14px ${FONT}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, x, y);
}

// The left and bottom axis lines of a plot area, on pixel centers.
function strokeAxes(ctx: CanvasRenderingContext2D, left: number, top: number, width: number, height: number) {
  ctx.beginPath();
  ctx.moveTo(snap(left), top);
  ctx.lineTo(snap(left), snap(top + height));
  ctx.lineTo(left + width, snap(top + height));
  ctx.stroke();
}

// Vertical colorbar of `lut` from vmin (bottom) to vmax (top), with the end and middle values.
function drawColorbar(
  ctx: CanvasRenderingContext2D, lut: Uint8Array, vmin: number, vmax: number,
  barX: number, barY: number, barH: number, isDark: boolean,
) {
  const barW = 12;
  for (let row = 0; row < barH; row++) {
    const t = 1 - row / Math.max(1, barH - 1);
    ctx.fillStyle = lutRgb(lut, Math.round(t * 255) * 3);
    ctx.fillRect(barX, barY + row, barW, 1);
  }
  ctx.strokeStyle = isDark ? "#666" : "#999";
  ctx.lineWidth = 1;
  ctx.strokeRect(snap(barX), snap(barY), barW, barH);
  ctx.fillStyle = isDark ? "#aaa" : "#555";
  ctx.font = `10px ${MONO}`;
  ctx.textAlign = "left";
  const labelX = barX + barW + 3;
  ctx.textBaseline = "top";
  ctx.fillText(formatNumber(vmax), labelX, barY);
  ctx.textBaseline = "middle";
  ctx.fillText(formatNumber((vmin + vmax) / 2), labelX, barY + barH / 2);
  ctx.textBaseline = "bottom";
  ctx.fillText(formatNumber(vmin), labelX, barY + barH);
}

// A value label boxed beside the point (cx, cy), moved to its left at the plot's right edge.
function drawValueLabel(
  ctx: CanvasRenderingContext2D, label: string, cx: number, cy: number, right: number,
  opacity: number, isDark: boolean,
) {
  const top = MARGIN_TOP;
  const bottom = MARGIN_TOP + PLOT_AREA_H;
  ctx.font = `12px ${MONO}`;
  const tw = ctx.measureText(label).width + 8;
  let bx = cx + 10;
  if (bx + tw > right) bx = cx - tw - 10;
  const by = Math.max(top, Math.min(bottom - 18, cy - 22));
  ctx.fillStyle = isDark ? `rgba(30,30,30,${opacity})` : `rgba(255,255,255,${opacity})`;
  ctx.fillRect(bx, by, tw, 18);
  ctx.strokeStyle = isDark ? "#555" : "#ccc";
  ctx.lineWidth = 1;
  ctx.strokeRect(bx, by, tw, 18);
  ctx.fillStyle = isDark ? "#eee" : "#333";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText(label, bx + 4, by + 9);
}

// The value in `values` nearest to `target` (the first of equally near ones).
function nearestValue(values: number[], target: number): number {
  let best = values[0];
  for (const v of values) {
    if (Math.abs(v - target) < Math.abs(best - target)) best = v;
  }
  return best;
}

// r/k at canvas x on the line-scan plot.
function heatmapRAt(g: HeatmapGeometry, sx: number): number {
  return g.rLo + ((sx - MARGIN_LEFT_MIN) / g.pwH) * (g.rHi - g.rLo);
}

// Position along the line at canvas y on the line-scan plot; A (0) is at the bottom, and the
// waterfall spreads the line over the band inside its padding.
function heatmapPositionAt(g: HeatmapGeometry, sy: number): number {
  const mT = MARGIN_TOP;
  const ph = PLOT_AREA_H;
  if (g.waterfall) return (((mT + ph - g.wfPad) - sy) / Math.max(1, ph - 2 * g.wfPad)) * g.lenUnits;
  return g.yLo + (1 - (sy - mT) / ph) * (g.yHi - g.yLo);
}

// Hover readout of the line-scan plot at canvas (sx, sy): r/k and the value of the nearest sample
// (in the waterfall, the nearest drawn curve), reported at that sample's own position.
function heatmapValueAt(g: HeatmapGeometry, sx: number, sy: number) {
  const r = heatmapRAt(g, sx);
  const pos = heatmapPositionAt(g, sy);
  const rFrac = (r - g.rLo) / ((g.rHi - g.rLo) || 1);
  const j = Math.max(0, Math.min(g.M - 1, Math.round(rFrac * (g.M - 1))));
  let bin = linescanSampleIndex(pos / (g.lenUnits || 1), g.N);
  if (g.waterfall && g.sel.length) bin = nearestValue(g.sel, bin);
  const val = g.data[bin * g.M + j];
  return { pos: sampleFraction(bin, g.N) * g.lenUnits, r, val };
}

// ============================================================================
// Slider with guarded commits
// ============================================================================
// CommitSlider passes MUI's commit on only after an onChange since the last commit or cancelled
// press, cancels the compatibility mouse events of every touch press, and ignores a change that
// only moves a single value set outside [min, max] to MUI's clamp of it. MUI commits its last
// onChange value at the end of every interaction, also one that changed nothing (a click that
// moves no thumb, a key at the end of the range), and that value can predate a change made
// underneath the slider (by Python, set_data, or the fit range for the k window).
function CommitSlider({ onChange, onChangeCommitted, ...props }: SliderProps) {
  const changedRef = React.useRef(false);   // onChange since the last commit or cancelled press
  const pressRef = React.useRef(props.value);   // the value shown when the press started
  const sentRef = React.useRef(props.value);   // the last value passed to onChange
  const [generation, setGeneration] = React.useState(0);   // remounts the slider after a cancel
  const { value, min = 0, max = 100 } = props;

  // After a tap the browser repeats the press as mouse events, which MUI would take as a second
  // press committing a second value. A pen keeps its mouse events, the only ones MUI gets from
  // a pen in Chromium.
  const handlePointerDown = (e: React.PointerEvent) => {
    pressRef.current = value;
    if (e.pointerType === "touch") e.preventDefault();
  };

  // A cancelled touch press never commits (MUI ignores touchcancel): show the value it started
  // from again, unless the value changed underneath, and remount the slider, since MUI keeps
  // listening for the cancelled touch id.
  const handlePointerCancel = (e: React.PointerEvent) => {
    if (changedRef.current && value === sentRef.current) {
      onChange?.(e.nativeEvent, pressRef.current as number | number[], 0);
    }
    changedRef.current = false;
    setGeneration((g) => g + 1);
  };

  const handleChange = (e: Event, v: number | number[], thumb: number) => {
    // A value Python set beyond the bound (r max 60) shows at the bound; a key towards the bound
    // must not set it there, and ignoring the change keeps the readout showing the value.
    const outOfRange = typeof value === "number" && (value < min || value > max);
    if (outOfRange && v === Math.max(min, Math.min(value, max))) return;
    changedRef.current = true;
    sentRef.current = v;
    onChange?.(e, v, thumb);
  };

  const handleChangeCommitted = (e: React.SyntheticEvent | Event, v: number | number[]) => {
    if (!changedRef.current) return;
    changedRef.current = false;
    onChangeCommitted?.(e, v);
  };

  return (
    <Slider
      key={generation}
      {...props}
      onPointerDown={handlePointerDown}
      onPointerCancel={handlePointerCancel}
      onChange={handleChange}
      onChangeCommitted={handleChangeCommitted}
    />
  );
}

// ============================================================================
// Main component
// ============================================================================
function ShowPDFWidget() {
  const model = useModel();
  React.useEffect(() => preserveRestoredWidgetModelsOnSave(model), [model]);
  // Standalone export (no kernel): the controls recompute the curves in JS. Exports render light,
  // as the other widgets' exports do.
  const [offline] = useModelState<boolean>("offline");
  const { themeInfo, colors } = useTheme(!!offline);
  const isDark = themeInfo.theme === "dark";

  // --- Model state ---
  const [title] = useModelState<string>("title");
  const [scanRows] = useModelState<number>("scan_rows");
  const [scanCols] = useModelState<number>("scan_cols");
  // Standalone HTML export: the button sends a request, Python builds the file, the browser downloads it.
  const [, setExportRequest] = useModelState<string>("export_request");
  const [exportStatus] = useModelState<string>("export_status");
  const [exportEnabled] = useModelState<boolean>("export_enabled");
  const [exportPayload] = useModelState<DataView>("export_payload");
  const [exportPayloadId] = useModelState<string>("export_payload_id");
  const [exportPayloadFilename] = useModelState<string>("export_filename");
  const [navImageBytes] = useModelState<DataView>("nav_image_bytes");
  const [navPixelSize] = useModelState<number>("nav_pixel_size");
  const [navUnit] = useModelState<string>("nav_unit");
  const [maskPixelCount, setMaskPixelCount] = useModelState<number>("mask_pixel_count");
  const [maskFraction, setMaskFraction] = useModelState<number>("mask_fraction");
  const [maskB64] = useModelState<string>("mask_b64");
  const [maskVersion, setMaskVersion] = useModelState<number>("mask_version");
  const [maskTool, setMaskTool] = useModelState<string>("mask_tool");
  const [maskBrushSize, setMaskBrushSize] = useModelState<number>("mask_brush_size");
  const [ikXBytes, setIkXBytes] = useModelState<DataView>("ik_x_bytes");
  const [ikYBytes, setIkYBytes] = useModelState<DataView>("ik_y_bytes");
  const [ikBgYBytes, setIkBgYBytes] = useModelState<DataView>("ik_bg_y_bytes");
  const [fkXBytes, setFkXBytes] = useModelState<DataView>("fk_x_bytes");
  const [fkYBytes, setFkYBytes] = useModelState<DataView>("fk_y_bytes");
  const [grXBytes, setGrXBytes] = useModelState<DataView>("gr_x_bytes");
  const [grYBytes, setGrYBytes] = useModelState<DataView>("gr_y_bytes");
  const [pdfXBytes, setPdfXBytes] = useModelState<DataView>("pdf_x_bytes");
  const [pdfYBytes, setPdfYBytes] = useModelState<DataView>("pdf_y_bytes");
  const [kMinFit, setKMinFit] = useModelState<number>("k_min_fit");
  const [kMaxFit, setKMaxFit] = useModelState<number>("k_max_fit");
  const [kMinWindow, setKMinWindow] = useModelState<number>("k_min_window");
  const [kMaxWindow, setKMaxWindow] = useModelState<number>("k_max_window");
  const [rMax, setRMax] = useModelState<number>("r_max");
  const [rStep] = useModelState<number>("r_step");
  const [kLowpass, setKLowpass] = useModelState<number>("k_lowpass");
  const [kHighpass, setKHighpass] = useModelState<number>("k_highpass");
  const [dampOrigin, setDampOrigin] = useModelState<boolean>("damp_origin_oscillations");
  // Quantized radial profile of every scan position (exports only), to recompute mask, probe and line.
  const [offlineRecompute] = useModelState<boolean>("offline_recompute");
  const [offlineNk] = useModelState<number>("offline_nk");
  const [offlineRows] = useModelState<number>("offline_rows");
  const [offlineCols] = useModelState<number>("offline_cols");
  const [offlineProfiles] = useModelState<DataView>("offline_profiles");
  const [offlineProfilesLo] = useModelState<DataView>("offline_profiles_lo");
  const [offlineProfilesHi] = useModelState<DataView>("offline_profiles_hi");
  const [offlineQq] = useModelState<DataView>("offline_qq");
  const [rCut, setRCut] = useModelState<number>("r_cut");
  const [maxIter, setMaxIter] = useModelState<number>("max_iter");
  // Number density (atoms/Å³) for g(r) and damping; 0 = not set (never estimated).
  const [densityValue, setDensityValue] = useModelState<number>("density_value");
  const [kMinAvail] = useModelState<number>("k_min_available");
  const [kMaxAvail] = useModelState<number>("k_max_available");
  const [plotMode, setPlotMode] = useModelState<string>("plot_mode");
  const [showBackground, setShowBackground] = useModelState<boolean>("show_background");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [logScale] = useModelState<boolean>("log_scale");
  const [autoContrast] = useModelState<boolean>("auto_contrast");
  const [showStats] = useModelState<boolean>("show_stats");
  const [showControls] = useModelState<boolean>("show_controls");
  const [computing] = useModelState<boolean>("computing");
  const [statusMessage] = useModelState<string>("status_message");
  const [analysisMode, setAnalysisMode] = useModelState<string>("analysis_mode");
  // An export without the per-position profiles cannot recompute a new mask, probe or line
  // (the 4D data is not in the file), so those controls are locked.
  const lockMask = !!offline && !offlineRecompute;
  const [probeRow, setProbeRow] = useModelState<number>("probe_row");
  const [probeCol, setProbeCol] = useModelState<number>("probe_col");
  const [probeSize, setProbeSize] = useModelState<number>("probe_size");
  const [lineRow0] = useModelState<number>("line_row0");
  const [lineCol0] = useModelState<number>("line_col0");
  const [lineRow1] = useModelState<number>("line_row1");
  const [lineCol1] = useModelState<number>("line_col1");
  const [lineWidth, setLineWidth] = useModelState<number>("line_width");
  const [lineActive] = useModelState<boolean>("line_active");
  const [lineMode, setLineMode] = useModelState<string>("line_mode");
  // The plot shows the line-scan stack (heatmap or waterfall) instead of one curve.
  const isLinescan = analysisMode === "line" && lineMode === "linescan";
  // A curves-only export cannot rebuild a line-scan stack, and without an I(k) (its compute had
  // failed) it cannot recompute anything, so its parameter controls are locked too.
  const lockParameters = lockMask && (isLinescan || extractBytes(ikYBytes).length === 0);
  const [linePerp, setLinePerp] = useModelState<boolean>("line_perpendicular");
  const [lineScanView, setLineScanView] = useModelState<string>("linescan_view");
  const [waterfallCount, setWaterfallCount] = useModelState<number>("waterfall_count");
  const [linescanBytes, setLinescanBytes] = useModelState<DataView>("linescan_bytes");
  const [nLinescan, setNLinescan] = useModelState<number>("n_linescan");
  const [linescanNPoints, setLinescanNPoints] = useModelState<number>("linescan_n_points");
  const [linescanAxisBytes, setLinescanAxisBytes] = useModelState<DataView>("linescan_axis_bytes");
  const [linescanMaxBins] = useModelState<number>("linescan_max_bins");

  // --- Local state ---
  const navH = Math.round(NAV_SIZE * (scanRows / Math.max(scanCols, 1)));
  const [navZoom, setNavZoom] = React.useState(1);
  const [navPanX, setNavPanX] = React.useState(0);
  const [navPanY, setNavPanY] = React.useState(0);
  const [maskAction, setMaskAction] = React.useState<"add" | "subtract">("add");
  const [maskRenderVersion, setMaskRenderVersion] = React.useState(0);
  // Corners of the mask shape being dragged (scan coords).
  const [shapePreview, setShapePreview] = React.useState<LineEnds | null>(null);
  const [plotXMin, setPlotXMin] = React.useState(0);
  const [plotXMax, setPlotXMax] = React.useState(10);
  const [plotYMin, setPlotYMin] = React.useState(-1);
  const [plotYMax, setPlotYMax] = React.useState(1);
  const [ikLogScale, setIkLogScale] = React.useState(true);
  const [cursorData, setCursorData] = React.useState<{ x: number; y: number } | null>(null);
  // Heatmap hover readout: position along the line, r/k, and the stack value there.
  const [hmHover, setHmHover] = React.useState<{ pos: number; r: number; val: number } | null>(null);
  const [localProbe, setLocalProbe] = React.useState({ row: probeRow, col: probeCol });
  const probeGenRef = React.useRef(0);
  const isDraggingProbeRef = React.useRef(false);
  // Line ends (scan coords): a local copy of the model's, for smooth dragging.
  const [localLine, setLocalLine] = React.useState<LineEnds>({
    r0: lineRow0, c0: lineCol0, r1: lineRow1, c1: lineCol1,
  });
  const localLineRef = React.useRef(localLine);
  const lineGenRef = React.useRef(0);
  const lineVersionRef = React.useRef(0);
  const [localLineWidth, setLocalLineWidth] = React.useState(lineWidth);
  // Largest line width seen in this view: the Width slider's range keeps it, so narrowing a width
  // above 99 (set from Python) does not shrink the range.
  const [maxLineWidthSeen, setMaxLineWidthSeen] = React.useState(lineWidth);
  if (lineWidth > maxLineWidthSeen) setMaxLineWidthSeen(lineWidth);
  const isDrawingLineRef = React.useRef(false);
  const isDraggingLineBodyRef = React.useRef(false);
  const draggingEndpointRef = React.useRef<0 | 1 | null>(null);
  const lineDrawStartRef = React.useRef<{ row: number; col: number } | null>(null);
  const lineBodyStartRef = React.useRef<{ imgX: number; imgY: number; line: LineEnds } | null>(null);
  // Heatmap guides (vertical lines at constant r/k), and the plot layout to map a click to r/k.
  const [hmGuides, setHmGuides] = React.useState<number[]>([]);
  const heatmapGeomRef = React.useRef<HeatmapGeometry | null>(null);
  const draggingGuideRef = React.useRef<{ index: number; moved: boolean; added: boolean } | null>(null);
  // Control values while they are edited; the effects below copy the model's values in.
  const [localKFit, setLocalKFit] = React.useState<[number, number]>([kMinFit, kMaxFit]);
  const [localKWin, setLocalKWin] = React.useState(windowInEffect(kMinWindow, kMaxWindow, kMinFit, kMaxFit));
  const [localRMax, setLocalRMax] = React.useState(rMax);
  const [localKLowpass, setLocalKLowpass] = React.useState(kLowpass);
  const [localKHighpass, setLocalKHighpass] = React.useState(kHighpass);
  const [localRCut, setLocalRCut] = React.useState(rCut);
  const [localMaxIter, setLocalMaxIter] = React.useState(maxIter);
  const [localDensity, setLocalDensity] = React.useState(fmtDensity(densityValue));
  // Typed into since the density field got focus or the model's density replaced its text.
  const densityEditedRef = React.useRef(false);
  const [localProbeSize, setLocalProbeSize] = React.useState(probeSize);
  const [localWfCount, setLocalWfCount] = React.useState(String(waterfallCount));
  const [dataVersion, setDataVersion] = React.useState(0);
  // Bumped when the offscreen nav image is redrawn; a counter of its own, since that redraw
  // depends on dataVersion.
  const [navRenderVersion, setNavRenderVersion] = React.useState(0);

  React.useEffect(() => { setLocalKFit([kMinFit, kMaxFit]); }, [kMinFit, kMaxFit]);
  React.useEffect(() => {
    setLocalKWin(windowInEffect(kMinWindow, kMaxWindow, kMinFit, kMaxFit));
  }, [kMinWindow, kMaxWindow, kMinFit, kMaxFit]);
  React.useEffect(() => { setLocalRMax(rMax); }, [rMax]);
  React.useEffect(() => { setLocalKLowpass(kLowpass); }, [kLowpass]);
  React.useEffect(() => { setLocalKHighpass(kHighpass); }, [kHighpass]);
  React.useEffect(() => { setLocalRCut(rCut); }, [rCut]);
  React.useEffect(() => { setLocalMaxIter(maxIter); }, [maxIter]);
  // The model's density replaces the field's text, and with it any edit.
  React.useEffect(() => {
    setLocalDensity(fmtDensity(densityValue));
    densityEditedRef.current = false;
  }, [densityValue]);
  React.useEffect(() => { setLocalProbeSize(probeSize); }, [probeSize]);
  React.useEffect(() => { setLocalWfCount(String(waterfallCount)); }, [waterfallCount]);
  React.useEffect(() => { setLocalProbe({ row: probeRow, col: probeCol }); }, [probeRow, probeCol]);
  React.useEffect(() => {
    const nl = { r0: lineRow0, c0: lineCol0, r1: lineRow1, c1: lineCol1 };
    localLineRef.current = nl;
    setLocalLine(nl);
  }, [lineRow0, lineCol0, lineRow1, lineCol1]);
  React.useEffect(() => { setLocalLineWidth(lineWidth); }, [lineWidth]);
  // The line actually sampled: the drawn line or, when line_perpendicular is on, its perpendicular
  // bisector, with both ends snapped to scan positions (as ShowPDF._effective_line).
  function effectiveLine(l: LineEnds) {
    return effectiveLineEnds(l, linePerp, scanRows, scanCols);
  }
  // Length (scan px) and orientation readout of the line actually sampled, as drawn on the scan.
  const sampledLine = effectiveLine(localLine);
  const lineLenPx = lineLength(sampledLine);
  const lineOrientDeg = lineOrientationDeg(sampledLine);

  // --- Refs ---
  // The user zoomed or panned the plot, so new curves keep the view instead of refitting it.
  const userZoomedRef = React.useRef(false);
  const navCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const navUiRef = React.useRef<HTMLCanvasElement>(null);  // HiDPI overlay for the scale bar
  const navOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const navOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const bandOverlayRef = React.useRef<BandOverlay | null>(null);
  const navImgDataRef = React.useRef<ImageData | null>(null);
  const rawNavRef = React.useRef<Float32Array | null>(null);
  const maskRef = React.useRef<Uint8Array | null>(null);
  const plotCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const ikXRef = React.useRef<Float32Array | null>(null);
  const ikYRef = React.useRef<Float32Array | null>(null);
  const ikBgRef = React.useRef<Float32Array | null>(null);
  const fkXRef = React.useRef<Float32Array | null>(null);
  const fkYRef = React.useRef<Float32Array | null>(null);
  const grXRef = React.useRef<Float32Array | null>(null);
  const grYRef = React.useRef<Float32Array | null>(null);
  const pdfXRef = React.useRef<Float32Array | null>(null);
  const pdfYRef = React.useRef<Float32Array | null>(null);
  const isPanningRef = React.useRef(false);
  const panStartRef = React.useRef<{ x: number; y: number; px: number; py: number } | null>(null);
  const isPaintingRef = React.useRef(false);
  const shapeStartRef = React.useRef<{ row: number; col: number } | null>(null);
  const isDraggingShapeRef = React.useRef(false);
  const isPlotPanRef = React.useRef(false);
  const plotPanStartRef = React.useRef<PlotPanStart | null>(null);

  // =========================================================================
  // Nav image and its overlay
  // =========================================================================
  // Parse the nav image; size the offscreen canvas and the mask to the scan.
  React.useEffect(() => {
    if (!navImageBytes || navImageBytes.byteLength < 4) return;
    const raw = extractFloat32(navImageBytes);
    if (!raw) return;
    rawNavRef.current = raw;
    const offscreen = navOffscreenRef.current;
    if (!offscreen || offscreen.width !== scanCols || offscreen.height !== scanRows) {
      const oc = document.createElement("canvas");
      oc.width = scanCols;
      oc.height = scanRows;
      navOffscreenRef.current = oc;
      navImgDataRef.current = new ImageData(scanCols, scanRows);
    }
    if (!maskRef.current || maskRef.current.length !== scanRows * scanCols) {
      maskRef.current = new Uint8Array(scanRows * scanCols).fill(1);
    }
    setDataVersion((v) => v + 1);
  }, [navImageBytes, scanRows, scanCols]);

  // The mask from the model: mask_b64 is the mask Python analyzes, so the overlay shows it after
  // this view paints and after Python changes it (set_mask, clear_mask, set_data, load_state_dict).
  React.useEffect(() => {
    const n = scanRows * scanCols;
    const decoded = decodeMask(maskB64, n);
    if (decoded) maskRef.current = decoded;
    else if (n > 0) maskRef.current = new Uint8Array(n).fill(1);
    setMaskRenderVersion((v) => v + 1);
  }, [maskB64, scanRows, scanCols]);

  // Render the nav image to the offscreen canvas (expensive).
  React.useEffect(() => {
    const raw = rawNavRef.current;
    const oc = navOffscreenRef.current;
    const imgData = navImgDataRef.current;
    if (!raw || !oc || !imgData) return;
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
    const processed = logScale ? applyLogScale(raw) : raw;
    let vmin: number, vmax: number;
    if (autoContrast) {
      ({ vmin, vmax } = percentileClip(processed, 2, 98));
    } else {
      const range = findDataRange(processed);
      vmin = range.min;
      vmax = range.max;
    }
    renderToOffscreenReuse(processed, lut, vmin, vmax, oc, imgData);
    setNavRenderVersion((v) => v + 1);
  }, [cmap, logScale, autoContrast, dataVersion]);

  // Draw the offscreen image, zoomed and panned (cheap).
  React.useLayoutEffect(() => {
    const cvs = navCanvasRef.current;
    const oc = navOffscreenRef.current;
    if (!cvs || !oc || oc.width === 0) return;
    const w = NAV_SIZE * DPR;
    const h = navH * DPR;
    cvs.width = w;
    cvs.height = h;
    const ctx = cvs.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, w, h);
    ctx.save();
    applyNavTransform(ctx, w, h, navPanX, navPanY, navZoom);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(oc, 0, 0, scanCols, scanRows, 0, 0, w, h);
    ctx.restore();
  }, [navRenderVersion, navZoom, navPanX, navPanY, navH, scanCols, scanRows]);

  // Scale bar, on the HiDPI overlay.
  React.useEffect(() => {
    const cvs = navUiRef.current;
    if (!cvs) return;
    cvs.width = NAV_SIZE * DPR;
    cvs.height = navH * DPR;
    // In nm (1 nm = 10 Å) when the scan is calibrated, otherwise in pixels.
    const calibrated = isCalibrated(navUnit, navPixelSize);
    const pxSize = calibrated ? navPixelSize / 10 : (navPixelSize || 1);
    const unit = calibrated ? "nm" : "px";
    drawNavScaleBar(cvs, DPR, navZoom, pxSize, unit, scanCols);
  }, [navZoom, navPanX, navPanY, navPixelSize, navUnit, scanCols, scanRows, navH]);

  // Overlay: the probe, the line, or the mask with the shape being dragged.
  React.useLayoutEffect(() => {
    const cvs = navOverlayRef.current;
    if (!cvs) return;
    const w = NAV_SIZE * DPR;
    const h = navH * DPR;
    cvs.width = w;
    cvs.height = h;
    const ctx = cvs.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, w, h);
    const scX = (NAV_SIZE * DPR) / Math.max(scanCols, 1);   // canvas px per scan px
    const scY = (navH * DPR) / Math.max(scanRows, 1);
    // The band image (up to scan size) is kept only while a line band is drawn.
    if (analysisMode !== "line") bandOverlayRef.current = null;

    if (analysisMode === "probe") {
      ctx.save();
      applyNavTransform(ctx, w, h, navPanX, navPanY, navZoom);
      drawProbe(ctx, localProbe, localProbeSize, scX, scY, navZoom);
      ctx.restore();
      return;
    }

    if (analysisMode === "line") {
      // The drawn line (its ends are the drag handles) and the scan line (its bisector when ⟂).
      const ll = localLine;
      const noLine = ll.r0 === ll.r1 && ll.c0 === ll.c1;
      if (!lineActive && !isDrawingLineRef.current && noLine) {
        bandOverlayRef.current = null;
        return;
      }
      const sl = effectiveLine(ll);
      const refAx = (ll.c0 + 0.5) * scX, refAy = (ll.r0 + 0.5) * scY;
      const refBx = (ll.c1 + 0.5) * scX, refBy = (ll.r1 + 0.5) * scY;
      const ax = (sl.c0 + 0.5) * scX, ay = (sl.r0 + 0.5) * scY;   // scan start (A)
      const bx = (sl.c1 + 0.5) * scX, by = (sl.r1 + 0.5) * scY;   // scan end (B)
      ctx.save();
      applyNavTransform(ctx, w, h, navPanX, navPanY, navZoom);
      // Without ⟂ the drawn line is the scan line.
      if (linePerp) drawReferenceLine(ctx, refAx, refAy, refBx, refBy, navZoom);
      // The band image is cached per band, so pan and zoom redraws reuse it.
      const bandKey = `${sl.r0},${sl.c0},${sl.r1},${sl.c1},${localLineWidth},${scanRows},${scanCols}`;
      if (bandOverlayRef.current?.key !== bandKey) {
        bandOverlayRef.current = { key: bandKey, ...renderBandImage(sl, localLineWidth, scanRows, scanCols) };
      }
      const band = bandOverlayRef.current;
      if (band.image) {
        // Unsmoothed, so each selected scan position is a whole cell, as in the mask overlay.
        const { image, cLo, rLo } = band;
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(
          image, 0, 0, image.width, image.height,
          cLo * scX, rLo * scY, image.width * scX, image.height * scY,
        );
      }
      drawScanLine(ctx, ax, ay, bx, by, navZoom);
      // In the waterfall the only dots are its sampling markers, so they match the curve colors.
      const wfMarkers = lineMode === "linescan" && lineScanView === "waterfall" && nLinescan > 0;
      drawLineEnds(ctx, ax, ay, bx, by, navZoom, !wfMarkers);
      if (wfMarkers) {
        // A dot where each stacked curve is taken, in its curve's color; a thin white ring keeps
        // it legible on the image.
        const lut = COLORMAPS.turbo || COLORMAPS.viridis;
        const mr = 4 / navZoom;
        for (const bin of sampledBinIndices(nLinescan, Math.max(1, waterfallCount))) {
          const f = sampleFraction(bin, nLinescan);
          const mx = ax + f * (bx - ax);
          const my = ay + f * (by - ay);
          ctx.beginPath();
          ctx.ellipse(mx, my, mr, mr, 0, 0, Math.PI * 2);
          ctx.fillStyle = lutRgb(lut, waterfallColorOffset(bin, nLinescan, isDark));
          ctx.fill();
          ctx.lineWidth = 1.2 / navZoom;
          ctx.strokeStyle = "rgba(255,255,255,0.9)";
          ctx.stroke();
        }
      }
      ctx.restore();
      return;
    }

    // Mask: excluded positions translucent red, and the outline of a shape being dragged.
    const mask = maskRef.current;
    if (!mask || mask.length === 0) return;
    const maskImage = renderMaskImage(mask, scanCols, scanRows);
    ctx.save();
    applyNavTransform(ctx, w, h, navPanX, navPanY, navZoom);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(maskImage, 0, 0, scanCols, scanRows, 0, 0, w, h);
    ctx.restore();
    if (shapePreview && isDraggingShapeRef.current) {
      const { r0, c0, r1, c1 } = shapePreview;
      ctx.save();
      applyNavTransform(ctx, w, h, navPanX, navPanY, navZoom);
      ctx.strokeStyle = maskAction === "add" ? "rgba(100,200,255,0.9)" : "rgba(255,100,100,0.9)";
      ctx.lineWidth = 2 / navZoom;
      ctx.setLineDash([4 / navZoom, 4 / navZoom]);
      const sx = Math.min(c0, c1) * scX;
      const sy = Math.min(r0, r1) * scY;
      const sw = Math.abs(c1 - c0) * scX;
      const sh = Math.abs(r1 - r0) * scY;
      if (maskTool === "circle") {
        ctx.beginPath();
        ctx.ellipse((c0 + c1) / 2 * scX, (r0 + r1) / 2 * scY, sw / 2, sh / 2, 0, 0, Math.PI * 2);
        ctx.stroke();
      } else {
        ctx.strokeRect(sx, sy, sw, sh);
      }
      ctx.restore();
    }
  }, [
    maskRenderVersion, navZoom, navPanX, navPanY, navH, scanRows, scanCols, shapePreview, maskAction,
    maskTool, analysisMode, localProbe, localProbeSize, localLine, localLineWidth, lineActive, linePerp,
    lineMode, lineScanView, waterfallCount, nLinescan, isDark,
  ]);

  // =========================================================================
  // Plot
  // =========================================================================
  // Parse the curves and fit the axes. A layout effect declared before the plot's draw effect,
  // so the plot draws the curves just received.
  const grEmptyRef = React.useRef(true);
  React.useLayoutEffect(() => {
    ikXRef.current = ikXBytes ? extractFloat32(ikXBytes) : null;
    ikYRef.current = ikYBytes ? extractFloat32(ikYBytes) : null;
    ikBgRef.current = ikBgYBytes ? extractFloat32(ikBgYBytes) : null;
    fkXRef.current = fkXBytes ? extractFloat32(fkXBytes) : null;
    fkYRef.current = fkYBytes ? extractFloat32(fkYBytes) : null;
    grXRef.current = grXBytes ? extractFloat32(grXBytes) : null;
    grYRef.current = grYBytes ? extractFloat32(grYBytes) : null;
    pdfXRef.current = pdfXBytes ? extractFloat32(pdfXBytes) : null;
    pdfYRef.current = pdfYBytes ? extractFloat32(pdfYBytes) : null;
    // g(r) arriving in the empty g(r) plot (the density was just set) is fitted, even if that
    // plot was zoomed.
    const grEmpty = !pdfXRef.current || pdfXRef.current.length === 0;
    if (plotMode === "gr" && grEmptyRef.current && !grEmpty) userZoomedRef.current = false;
    grEmptyRef.current = grEmpty;
    if (!userZoomedRef.current) autoFitPlot();
  }, [ikXBytes, ikYBytes, ikBgYBytes, fkXBytes, fkYBytes, grXBytes, grYBytes, pdfXBytes, pdfYBytes]);

  React.useEffect(() => { refitPlot(); }, [plotMode]);
  // r/k units change with plot mode, so existing guide positions become invalid.
  React.useEffect(() => { setHmGuides([]); }, [plotMode]);

  // The x and y arrays of the curve the plot mode shows.
  function shownCurve(): [Float32Array | null, Float32Array | null] {
    if (plotMode === "Ik") return [ikXRef.current, ikYRef.current];
    if (plotMode === "Fk") return [fkXRef.current, fkYRef.current];
    if (plotMode === "gr") return [pdfXRef.current, pdfYRef.current];
    return [grXRef.current, grYRef.current];
  }

  // Fit the axes to the shown curve. logOverride lets the Log switch refit with its new value,
  // which ikLogScale does not hold yet in the switch's handler.
  function autoFitPlot(logOverride?: boolean) {
    const [xArr, yArr] = shownCurve();
    if (plotMode === "gr" && (!xArr || xArr.length === 0)) {
      // No g(r) (the density is unset): the r grid of G(r) and a neutral g(r) range.
      const r = grXRef.current;
      if (r && r.length > 0) {
        const rR = findDataRange(r);
        setPlotXMin(rR.min);
        setPlotXMax(rR.max);
      }
      setPlotYMin(0);
      setPlotYMax(2);
      return;
    }
    if (!xArr || !yArr || xArr.length === 0) return;
    const xR = findDataRange(xArr);
    const useLog = plotMode === "Ik" && (logOverride ?? ikLogScale);
    const yR = findDataRange(yArr);
    setPlotXMin(xR.min);
    setPlotXMax(xR.max);
    if (useLog) {
      const logMax = yR.max > 0 ? Math.log10(yR.max) : 0;
      // About 6 decades below the max when the data reach 0 or below (no log), never inverted.
      let logMin = yR.min > 0 ? Math.log10(yR.min) : logMax - 6;
      if (!(logMin < logMax)) logMin = logMax - 1;
      const logPad = (logMax - logMin) * 0.05 || 0.1;
      setPlotYMin(logMin - logPad);
      setPlotYMax(logMax + logPad);
    } else {
      const yPad = (yR.max - yR.min) * 0.05 || 0.1;
      setPlotYMin(yR.min - yPad);
      setPlotYMax(yR.max + yPad);
    }
  }

  // Forget the user's zoom and fit the axes to the shown curve.
  function refitPlot(logOverride?: boolean) {
    userZoomedRef.current = false;
    autoFitPlot(logOverride);
  }

  // Value of the shown curve at x (linear interpolation).
  function curveValueAt(x: number): number | null {
    const [xa, ya] = shownCurve();
    if (!xa || !ya || xa.length === 0) return null;
    return interpolate(xa, ya, x);
  }

  // Line-scan plot: a heatmap of r/k (x) against position along the line (y), colored by the
  // curve value, or the stacked curves of the waterfall.
  function drawLinescanPlot(ctx: CanvasRenderingContext2D) {
    const mL = MARGIN_LEFT_MIN;
    const mT = MARGIN_TOP;
    const ph = PLOT_AREA_H;
    const waterfall = lineScanView === "waterfall";
    // The heatmap keeps room on the right for its value colorbar. The waterfall has none: its
    // colors encode position, which the markers on the scan line show.
    const colorbarRoom = waterfall ? 0 : 52;
    const pwH = PLOT_AREA_W - colorbarRoom;
    ctx.fillStyle = isDark ? "#1a1a1a" : "#f8f8f8";
    ctx.fillRect(0, 0, PLOT_W, PLOT_H);
    const N = nLinescan;
    const M = linescanNPoints;
    const ready = N > 0 && M > 0 && !!linescanBytes && linescanBytes.byteLength >= N * M * 4;
    if (!ready) {
      heatmapGeomRef.current = null;
      const hasLine = lineActive && (localLine.r0 !== localLine.r1 || localLine.c0 !== localLine.c1);
      drawPlotMessage(ctx, linescanPlaceholder(computing, hasLine), mL + pwH / 2, mT + ph / 2, isDark);
      return;
    }
    const data = extractFloat32(linescanBytes);
    if (!data) return;
    const hasAxis = !!linescanAxisBytes && linescanAxisBytes.byteLength >= M * 4;
    const axis = hasAxis ? extractFloat32(linescanAxisBytes) : null;
    const { vmin, vmax } = percentileClip(finiteValues(data, N * M), 1, 99);
    const lut = COLORMAPS.viridis || COLORMAPS.inferno;
    // Length of the line actually sampled (its ends snapped to scan positions).
    const lenScan = lineLength(effectiveLine(localLine));
    const calibrated = isCalibrated(navUnit, navPixelSize);
    const lenUnits = calibrated ? lenScan * navPixelSize : lenScan;
    const rLo = axis ? axis[0] : 0;
    const rHi = axis ? axis[M - 1] : 1;
    const hx = (rv: number) => mL + ((rv - rLo) / ((rHi - rLo) || 1)) * pwH;   // r/k → x
    // The waterfall pads top and bottom, so the outermost curves and their deflection do not clip.
    const selW = waterfall ? sampledBinIndices(N, Math.max(1, waterfallCount)) : [];
    const Kw = Math.max(1, selW.length);
    const wfPad = waterfall ? Math.min(0.24 * ph, Math.max(8, (2.2 * ph) / (Kw + 4.4))) : 0;
    const yBot = mT + ph - wfPad;
    const yInner = ph - 2 * wfPad;
    // Heatmap rows are centered on the samples (evenly spaced from A at 0 to the tip B, see
    // linescan.ts), so the position axis reaches half a step beyond both ends.
    const posOf = (bin: number) => sampleFraction(bin, N) * lenUnits;
    const step = N > 1 ? posOf(1) - posOf(0) : lenUnits;
    const yLo = N > 1 ? posOf(0) - step / 2 : 0;
    const yHi = N > 1 ? posOf(N - 1) + step / 2 : lenUnits;
    // position → y
    const hy = (pos: number) => {
      if (waterfall) return yBot - (pos / (lenUnits || 1)) * yInner;
      return mT + ph - ((pos - yLo) / ((yHi - yLo) || 1)) * ph;
    };
    heatmapGeomRef.current = { rLo, rHi, pwH, lenUnits, N, M, data, waterfall, wfPad, yLo, yHi, sel: selW };

    if (waterfall) {
      // One curve per drawn sample, in its position's color. The amplitude scale is the 2-98
      // percentile range of the drawn curves, not their min/max, so a big first peak does not
      // wash out the smaller oscillations.
      const svals: number[] = [];
      for (const bin of selW) {
        for (let j = 0; j < M; j++) {
          const v = data[bin * M + j];
          if (isFinite(v)) svals.push(v);
        }
      }
      const pr = svals.length ? percentileClip(Float32Array.from(svals), 2, 98) : { vmin: 0, vmax: 1 };
      const pRange = (pr.vmax - pr.vmin) || 1;
      const vmid = (pr.vmin + pr.vmax) / 2;
      const lane = yInner / Kw;
      const amp = (2.2 * lane) / pRange;   // stretch small features
      const maxDefl = 2.0 * lane;   // soft-saturation scale for tall peaks
      // turbo separates adjacent curves by hue.
      const wlut = COLORMAPS.turbo || lut;
      // r/k of point j: from the axis, or spread evenly over [rLo, rHi] without one.
      const rAt = (j: number) => {
        if (axis) return axis[j];
        return M > 1 ? rLo + (j / (M - 1)) * (rHi - rLo) : rLo;
      };
      ctx.save();
      ctx.beginPath();
      ctx.rect(mL, mT, pwH, ph);
      ctx.clip();
      // Back (top) to front (bottom).
      for (let s = Kw - 1; s >= 0; s--) {
        const bin = selW[s];
        const baseY = hy(posOf(bin));
        ctx.beginPath();
        for (let j = 0; j < M; j++) {
          const xv = hx(rAt(j));
          const v = data[bin * M + j];
          const dyRaw = (isFinite(v) ? v - vmid : 0) * amp;
          // Soft saturation: linear for small features, tall peaks rounded off within the lane.
          const dy = maxDefl * Math.tanh(dyRaw / maxDefl);
          const yv = baseY - dy;
          if (j === 0) ctx.moveTo(xv, yv);
          else ctx.lineTo(xv, yv);
        }
        ctx.strokeStyle = lutRgb(wlut, waterfallColorOffset(bin, N, isDark));
        ctx.lineWidth = 1.4;
        ctx.stroke();
      }
      ctx.restore();
    } else {
      // Heatmap image, M (r/k, left to right) x N; row 0 is the end B, so position increases upward.
      const img = new Float32Array(M * N);
      for (let bin = 0; bin < N; bin++) {
        for (let j = 0; j < M; j++) {
          const v = data[bin * M + j];
          img[(N - 1 - bin) * M + j] = isFinite(v) ? v : vmin;
        }
      }
      const off = renderToOffscreen(img, M, N, lut, vmin, vmax);
      if (off) {
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(off, 0, 0, M, N, mL, mT, pwH, ph);
      }
    }

    // Guides: vertical lines at constant r/k, to check how straight features are.
    for (const gv of hmGuides) {
      const cx = snap(hx(gv));
      if (cx < mL || cx > mL + pwH) continue;
      ctx.strokeStyle = isDark ? "rgba(255,255,255,0.85)" : "rgba(0,0,0,0.8)";
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 4]);
      strokeSegment(ctx, cx, mT, cx, mT + ph);
      ctx.setLineDash([]);
      ctx.fillStyle = isDark ? "#eee" : "#222";
      ctx.font = `10px ${MONO}`;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(formatNumber(gv), cx, mT + 2);
    }

    // Axes, ticks (x: r/k, y: position) and titles.
    ctx.strokeStyle = isDark ? "#666" : "#999";
    ctx.lineWidth = 1;
    ctx.setLineDash([]);
    strokeAxes(ctx, mL, mT, pwH, ph);
    ctx.fillStyle = isDark ? "#aaa" : "#555";
    ctx.font = `13px ${FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const tv of computeTicks(rLo, rHi, Math.max(3, Math.floor(pwH / TICK_LABEL_W)))) {
      const cx = snap(hx(tv));
      if (cx < mL - 1 || cx > mL + pwH + 1) continue;
      strokeSegment(ctx, cx, mT + ph, cx, mT + ph + AXIS_TICK_PX);
      ctx.fillText(formatNumber(tv), cx, mT + ph + AXIS_TICK_PX + 2);
    }
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const tv of computeTicks(0, lenUnits, Math.max(3, Math.floor(ph / 40)))) {
      const cy = snap(hy(tv));
      if (cy < mT - 1 || cy > mT + ph + 1) continue;
      strokeSegment(ctx, mL, cy, mL - AXIS_TICK_PX, cy);
      ctx.fillText(formatNumber(tv), mL - AXIS_TICK_PX - 2, cy);
    }
    ctx.font = `14px ${FONT}`;
    ctx.fillStyle = isDark ? "#ccc" : "#333";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(isRAxis(plotMode) ? "r (Å)" : "k (Å⁻¹)", mL + pwH / 2, mT + ph + AXIS_TICK_PX + 26);
    ctx.save();
    ctx.translate(14, mT + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(calibrated ? "position (Å)" : "position (px)", 0, 0);
    ctx.restore();

    // B and A, tying the plot to the ends of the scan line.
    ctx.font = `11px ${MONO}`;
    ctx.textAlign = "left";
    ctx.fillStyle = "rgba(255,110,110,1)";
    ctx.textBaseline = "top";
    ctx.fillText("B", mL + 4, mT + 3);
    ctx.fillStyle = "rgba(100,225,140,1)";
    ctx.textBaseline = "bottom";
    ctx.fillText("A", mL + 4, mT + ph - 3);
    if (!waterfall) drawColorbar(ctx, lut, vmin, vmax, mL + pwH + 14, mT, ph, isDark);

    // Hover crosshair and readout: r/k, position → value.
    if (hmHover) {
      const cx = snap(hx(hmHover.r));
      const cy = snap(hy(hmHover.pos));
      if (cx >= mL && cx <= mL + pwH && cy >= mT && cy <= mT + ph) {
        ctx.strokeStyle = isDark ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.5)";
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        strokeSegment(ctx, cx, mT, cx, mT + ph);
        strokeSegment(ctx, mL, cy, mL + pwH, cy);
        ctx.setLineDash([]);
        const r = formatNumber(hmHover.r);
        const pos = formatNumber(hmHover.pos);
        const val = formatNumber(hmHover.val);
        drawValueLabel(ctx, `${r}, ${pos} → ${val}`, cx, cy, mL + pwH, 0.92, isDark);
      }
    }
    if (computing) {
      ctx.fillStyle = isDark ? "rgba(0,0,0,0.3)" : "rgba(255,255,255,0.3)";
      ctx.fillRect(mL, mT, pwH, ph);
    }
  }

  // The plot mode's curve, with its grid, axes, fit or window range, crosshair and readout.
  function drawCurvePlot(ctx: CanvasRenderingContext2D) {
    const mL = MARGIN_LEFT_MIN;
    const mT = MARGIN_TOP;
    const pw = PLOT_AREA_W;
    const ph = PLOT_AREA_H;
    const xMin = plotXMin, xMax = plotXMax, yMin = plotYMin, yMax = plotYMax;
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;
    const d2cx = (dx: number) => mL + ((dx - xMin) / xRange) * pw;
    const useLogY = plotMode === "Ik" && ikLogScale;
    // Axis value → canvas y (yMin and yMax are already log10 values on a log axis).
    const d2cy = (dy: number) => mT + ph - ((dy - yMin) / yRange) * ph;
    // Data value → canvas y, through log10 on a log axis.
    const d2cyData = (dy: number) => {
      let v = dy;
      if (useLogY) v = dy > 0 ? Math.log10(dy) : yMin;
      return mT + ph - ((v - yMin) / yRange) * ph;
    };

    ctx.fillStyle = isDark ? "#1a1a1a" : "#f8f8f8";
    ctx.fillRect(0, 0, PLOT_W, PLOT_H);
    // Grid
    ctx.strokeStyle = isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)";
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 3]);
    const xTicks = computeTicks(xMin, xMax, Math.max(3, Math.floor(pw / TICK_LABEL_W)));
    for (const tv of xTicks) {
      const cx = snap(d2cx(tv));
      strokeSegment(ctx, cx, mT, cx, mT + ph);
    }
    const yTicks = computeTicks(yMin, yMax, Math.max(3, Math.floor(ph / 40)));
    for (const tv of yTicks) {
      const cy = snap(d2cy(tv));
      strokeSegment(ctx, mL, cy, mL + pw, cy);
    }
    ctx.setLineDash([]);
    // Axes and ticks
    ctx.strokeStyle = isDark ? "#666" : "#999";
    ctx.lineWidth = 1;
    strokeAxes(ctx, mL, mT, pw, ph);
    ctx.fillStyle = isDark ? "#aaa" : "#555";
    ctx.font = `13px ${FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const tv of xTicks) {
      const cx = snap(d2cx(tv));
      strokeSegment(ctx, cx, mT + ph, cx, mT + ph + AXIS_TICK_PX);
      ctx.fillText(formatNumber(tv), cx, mT + ph + AXIS_TICK_PX + 2);
    }
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    let yTickW = 0;   // widest y tick label
    for (const tv of yTicks) {
      const cy = snap(d2cy(tv));
      strokeSegment(ctx, mL, cy, mL - AXIS_TICK_PX, cy);
      const text = useLogY ? formatNumber(Math.pow(10, tv)) : formatNumber(tv);
      ctx.fillText(text, mL - AXIS_TICK_PX - 2, cy);
      yTickW = Math.max(yTickW, ctx.measureText(text).width);
    }
    // Axis titles; the y title moves left of wide tick labels (log I(k): "100.00").
    ctx.font = `14px ${FONT}`;
    ctx.fillStyle = isDark ? "#ccc" : "#333";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(isRAxis(plotMode) ? "r (Å)" : "k (Å⁻¹)", mL + pw / 2, mT + ph + AXIS_TICK_PX + 26);
    const yTitleX = Math.max(2, Math.min(14, mL - AXIS_TICK_PX - 2 - yTickW - 18));
    ctx.save();
    ctx.translate(yTitleX, mT + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(curveLabels(plotMode).name, 0, 0);
    ctx.restore();

    // Traces, clipped to the plot area.
    ctx.save();
    ctx.beginPath();
    ctx.rect(mL, mT, pw, ph);
    ctx.clip();
    const drawTrace = (xD: Float32Array | null, yD: Float32Array | null, color: string, dashed = false) => {
      if (!xD || !yD) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      if (dashed) ctx.setLineDash([4, 3]);
      else ctx.setLineDash([]);
      ctx.beginPath();
      let started = false;
      const len = Math.min(xD.length, yD.length);
      for (let i = 0; i < len; i++) {
        if (!isFinite(yD[i])) continue;
        const cx = d2cx(xD[i]);
        const cy = d2cyData(yD[i]);
        if (!started) {
          ctx.moveTo(cx, cy);
          started = true;
        } else {
          ctx.lineTo(cx, cy);
        }
      }
      ctx.stroke();
      ctx.setLineDash([]);
    };
    // Dashed vertical lines at the ends of the fit or window range, labelled.
    const drawKRange = (k0: number, k1: number, color: string, label0: string, label1: string) => {
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.setLineDash([14, 6]);
      ctx.fillStyle = color;
      ctx.font = `11px ${MONO}`;
      ctx.textBaseline = "top";
      for (const [kv, lbl, align] of [[k0, label0, "left"], [k1, label1, "right"]] as const) {
        if (!isFinite(kv)) continue;
        const cx = snap(d2cx(kv));
        if (cx < mL || cx > mL + pw) continue;
        strokeSegment(ctx, cx, mT, cx, mT + ph);
        ctx.textAlign = align;
        const tx = align === "left" ? cx + 3 : cx - 3;
        ctx.fillText(lbl, tx, mT + 2);
      }
      ctx.restore();
    };
    // Dashed horizontal line at axis value y (g(r) = 1, G(r) = 0).
    const drawLevel = (y: number) => {
      ctx.strokeStyle = isDark ? "rgba(255,255,255,0.2)" : "rgba(0,0,0,0.2)";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      const cy = d2cy(y);
      strokeSegment(ctx, mL, cy, mL + pw, cy);
      ctx.setLineDash([]);
    };
    if (plotMode === "Ik") {
      drawTrace(ikXRef.current, ikYRef.current, TRACE_COLORS.ik);
      if (showBackground) drawTrace(ikXRef.current, ikBgRef.current, TRACE_COLORS.bg, true);
      drawKRange(kMinFit, kMaxFit, "#ffb300", "k fit", "");
    } else if (plotMode === "Fk") {
      drawTrace(fkXRef.current, fkYRef.current, TRACE_COLORS.fk);
      const [winLo, winHi] = windowInEffect(kMinWindow, kMaxWindow, kMinFit, kMaxFit);
      drawKRange(winLo, winHi, "#ffb300", "k window", "");
    } else if (plotMode === "gr") {
      if (pdfXRef.current && pdfXRef.current.length > 0) {
        drawLevel(1);
        drawTrace(pdfXRef.current, pdfYRef.current, TRACE_COLORS.pdf);
      } else if (!(densityValue > 0)) {
        // No g(r) without a density: say so in the plot, not only in the status line.
        drawPlotMessage(ctx, DENSITY_HINT_GR, mL + pw / 2, mT + ph / 2, isDark);
      }
    } else {
      drawLevel(0);
      drawTrace(grXRef.current, grYRef.current, TRACE_COLORS.gr);
    }
    ctx.restore();

    // Crosshair; the readout snaps to the shown curve's value at the cursor's x.
    if (cursorData) {
      const cval = curveValueAt(cursorData.x);
      const cx = d2cx(cursorData.x);
      if (cval !== null && isFinite(cval) && cx >= mL && cx <= mL + pw) {
        const cy = d2cyData(cval);
        ctx.strokeStyle = isDark ? "rgba(255,255,255,0.25)" : "rgba(0,0,0,0.25)";
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        strokeSegment(ctx, snap(cx), mT, snap(cx), mT + ph);
        ctx.setLineDash([]);
        if (cy >= mT && cy <= mT + ph) {
          ctx.fillStyle = isDark ? "#eee" : "#333";
          ctx.beginPath();
          ctx.ellipse(cx, cy, 3, 3, 0, 0, Math.PI * 2);
          ctx.fill();
        }
        const label = `${formatNumber(cursorData.x)}, ${formatNumber(cval)}`;
        drawValueLabel(ctx, label, cx, cy, mL + pw, 0.9, isDark);
      }
    }
    if (computing) {
      ctx.fillStyle = isDark ? "rgba(0,0,0,0.3)" : "rgba(255,255,255,0.3)";
      ctx.fillRect(mL, mT, pw, ph);
    }
  }

  // Draw the plot: the line-scan stack, or the plot mode's curve. The dependencies are the
  // values the two draw functions read.
  React.useLayoutEffect(() => {
    const cvs = plotCanvasRef.current;
    if (!cvs) return;
    cvs.width = PLOT_W * DPR;
    cvs.height = PLOT_H * DPR;
    const ctx = cvs.getContext("2d");
    if (!ctx) return;
    ctx.scale(DPR, DPR);
    if (analysisMode === "line" && lineMode === "linescan") drawLinescanPlot(ctx);
    else drawCurvePlot(ctx);
  }, [
    plotXMin, plotXMax, plotYMin, plotYMax, plotMode, showBackground, ikLogScale, cursorData, computing,
    isDark, ikXBytes, ikYBytes, ikBgYBytes, fkXBytes, fkYBytes, grXBytes, grYBytes, pdfXBytes, pdfYBytes,
    kMinFit, kMaxFit, kMinWindow, kMaxWindow, densityValue,
    analysisMode, lineMode, lineScanView, waterfallCount, nLinescan, linescanNPoints, linescanBytes,
    linescanAxisBytes, localLine, linePerp, navPixelSize, navUnit, lineActive, hmGuides, hmHover,
  ]);

  // =========================================================================
  // Nav: coordinates, mask, probe and line edits
  // =========================================================================
  // Pointer position in un-zoomed canvas (image) coordinates, where the overlay draws.
  function screenToImgXY(e: React.MouseEvent) {
    const cvs = navCanvasRef.current;
    if (!cvs) return { imgX: 0, imgY: 0 };
    const { x: sx, y: sy } = pointerOffset(cvs, e);
    const cx = NAV_SIZE / 2, cy = navH / 2;
    return { imgX: (sx - cx - navPanX) / navZoom + cx, imgY: (sy - cy - navPanY) / navZoom + cy };
  }

  // The scan pixel under the pointer (pixel r covers [r, r + 1); positions are drawn at r + 0.5).
  function screenToImage(e: React.MouseEvent) {
    if (!navCanvasRef.current) return { row: 0, col: 0 };
    const { imgX, imgY } = screenToImgXY(e);
    return {
      row: Math.max(0, Math.min(scanRows - 1, Math.floor((imgY / navH) * scanRows))),
      col: Math.max(0, Math.min(scanCols - 1, Math.floor((imgX / NAV_SIZE) * scanCols))),
    };
  }

  // Image coordinates of a scan position's pixel center, as the overlay draws it.
  function scanToImgXY(row: number, col: number) {
    return {
      imgX: (col + 0.5) / Math.max(scanCols, 1) * NAV_SIZE,
      imgY: (row + 0.5) / Math.max(scanRows, 1) * navH,
    };
  }

  const maskVersionRef = React.useRef(maskVersion ?? 0);
  // Send the mask to Python, which recomputes; an export with the profiles recomputes in JS.
  function syncMaskToPython() {
    if (!maskRef.current) return;
    if (offlineReady) {
      recomputeOfflineAll({});
      return;
    }
    model.set("mask_b64", encodeMask(maskRef.current));
    model.save_changes();
    // Bump mask_version to trigger the Python observer, from the model's value: another view of
    // the same widget may have bumped it already.
    maskVersionRef.current = ((model.get("mask_version") as number) ?? maskVersionRef.current) + 1;
    setMaskVersion(maskVersionRef.current);
  }

  // Mask edits of the keys and buttons; each redraws the overlay and syncs the mask.
  function clearMask() {
    if (!maskRef.current) return;
    maskRef.current.fill(1);
    setMaskRenderVersion((v) => v + 1);
    syncMaskToPython();
  }

  function invertMask() {
    const mask = maskRef.current;
    if (!mask) return;
    for (let i = 0; i < mask.length; i++) mask[i] = mask[i] ? 0 : 1;
    setMaskRenderVersion((v) => v + 1);
    syncMaskToPython();
  }

  // The Invert button also swaps add and subtract (the I key inverts only).
  const handleInvertClick = () => {
    if (!maskRef.current) return;
    invertMask();
    setMaskAction((a) => (a === "add" ? "subtract" : "add"));
  };

  // Switching between add and subtract resets the whole mask: all positions included before
  // adding (painting excludes), all excluded before subtracting (painting re-includes).
  function switchMaskAction() {
    setMaskAction((a) => {
      if (maskRef.current) {
        maskRef.current.fill(a === "add" ? 0 : 1);
        setMaskRenderVersion((v) => v + 1);
        syncMaskToPython();
      }
      return a === "add" ? "subtract" : "add";
    });
  }

  // "+" adds to the mask (excludes positions from the analysis: 0); "−" re-includes them (1).
  const maskPaintValue = maskAction === "add" ? 0 : 1;

  // Paint the freeform brush (a square of 2·size−1 positions) at a scan position.
  function paintBrush(mask: Uint8Array, row: number, col: number) {
    const half = maskBrushSize - 1;
    fillRectMask(mask, scanCols, scanRows, row - half, col - half, row + half, col + half, maskPaintValue);
    setMaskRenderVersion((v) => v + 1);
  }

  // Live probe move, coalesced to at most one model update per display frame.
  function syncProbeRAF(row: number, col: number) {
    const r = Math.max(0, Math.min(scanRows - 1, row));
    const c = Math.max(0, Math.min(scanCols - 1, col));
    setLocalProbe({ row: r, col: c });
    const gen = ++probeGenRef.current;
    requestAnimationFrame(() => {
      if (gen !== probeGenRef.current) return;
      if (r !== probeRow) setProbeRow(r);
      if (c !== probeCol) setProbeCol(c);
    });
  }

  // Set the local line, clamped to the scan, and return it.
  function applyLineLocal(line: LineEnds) {
    const clampR = (v: number) => Math.max(0, Math.min(scanRows - 1, v));
    const clampC = (v: number) => Math.max(0, Math.min(scanCols - 1, v));
    const nl = { r0: clampR(line.r0), c0: clampC(line.c0), r1: clampR(line.r1), c1: clampC(line.c1) };
    localLineRef.current = nl;
    setLocalLine(nl);
    return nl;
  }

  // Send the line's ends with a single line_version bump, so Python recomputes once.
  function commitLine(nl: LineEnds) {
    model.set("line_row0", nl.r0);
    model.set("line_col0", nl.c0);
    model.set("line_row1", nl.r1);
    model.set("line_col1", nl.c1);
    if (!lineActive) model.set("line_active", true);
    lineVersionRef.current = ((model.get("line_version") as number) ?? lineVersionRef.current) + 1;
    model.set("line_version", lineVersionRef.current);
    model.save_changes();
    // No kernel: recompute the line or line-scan in JS.
    if (offlineReady) recomputeOfflineAll({});
  }

  // During a drag the overlay follows every move. The averaged line also commits live, at most
  // once per frame; the line-scan's heavy recompute waits for mouse-up.
  function dragLine(line: LineEnds) {
    const nl = applyLineLocal(line);
    if (lineMode !== "linescan") {
      const gen = ++lineGenRef.current;
      requestAnimationFrame(() => {
        if (gen === lineGenRef.current) commitLine(nl);
      });
    }
  }

  function clearLine() {
    isDrawingLineRef.current = false;
    isDraggingLineBodyRef.current = false;
    draggingEndpointRef.current = null;
    const nl = { r0: 0, c0: 0, r1: 0, c1: 0 };
    localLineRef.current = nl;
    setLocalLine(nl);
    model.set("line_row0", 0);
    model.set("line_col0", 0);
    model.set("line_row1", 0);
    model.set("line_col1", 0);
    model.set("line_active", false);
    lineVersionRef.current = ((model.get("line_version") as number) ?? 0) + 1;
    model.set("line_version", lineVersionRef.current);
    model.save_changes();
  }

  // A press in line mode grabs an end or the body of the line, or starts drawing a new line.
  function startLineGesture(e: React.MouseEvent, row: number, col: number) {
    const hit = 8 / navZoom;   // hit radius in image px
    const { imgX, imgY } = screenToImgXY(e);
    if (lineActive) {
      const e0 = scanToImgXY(localLineRef.current.r0, localLineRef.current.c0);
      const e1 = scanToImgXY(localLineRef.current.r1, localLineRef.current.c1);
      const d0 = Math.hypot(imgX - e0.imgX, imgY - e0.imgY);
      const d1 = Math.hypot(imgX - e1.imgX, imgY - e1.imgY);
      if (d0 < hit || d1 < hit) {
        draggingEndpointRef.current = d0 <= d1 ? 0 : 1;
        return;
      }
      if (distanceToSegment(imgX, imgY, e0.imgX, e0.imgY, e1.imgX, e1.imgY) < hit) {
        isDraggingLineBodyRef.current = true;
        lineBodyStartRef.current = { imgX, imgY, line: { ...localLineRef.current } };
        return;
      }
    }
    isDrawingLineRef.current = true;
    lineDrawStartRef.current = { row, col };
    dragLine({ r0: row, c0: col, r1: row, c1: col });
  }

  // =========================================================================
  // Nav mouse handlers
  // =========================================================================
  const handleNavMouseDown = (e: React.MouseEvent) => {
    if (!maskRef.current) return;
    e.preventDefault();
    // Middle button, or Alt with the left button, pans.
    if (e.button === 1 || (e.button === 0 && e.altKey)) {
      isPanningRef.current = true;
      panStartRef.current = { x: e.clientX, y: e.clientY, px: navPanX, py: navPanY };
      return;
    }
    if (lockMask) return;   // a locked selection blocks painting and the probe and line drags
    const { row, col } = screenToImage(e);
    if (analysisMode === "line") {
      startLineGesture(e, row, col);
      return;
    }
    if (analysisMode === "probe") {
      isDraggingProbeRef.current = true;
      syncProbeRAF(row, col);
      return;
    }
    if (maskTool === "freeform") {
      isPaintingRef.current = true;
      paintBrush(maskRef.current, row, col);
    } else {
      isDraggingShapeRef.current = true;
      shapeStartRef.current = { row, col };
      setShapePreview({ r0: row, c0: col, r1: row, c1: col });
    }
  };

  const handleNavMouseMove = (e: React.MouseEvent) => {
    if (isPanningRef.current && panStartRef.current) {
      setNavPanX(panStartRef.current.px + e.clientX - panStartRef.current.x);
      setNavPanY(panStartRef.current.py + e.clientY - panStartRef.current.y);
      return;
    }
    const { row, col } = screenToImage(e);
    if (isDraggingProbeRef.current) {
      syncProbeRAF(row, col);
      return;
    }
    if (isDrawingLineRef.current && lineDrawStartRef.current) {
      const s = lineDrawStartRef.current;
      dragLine({ r0: s.row, c0: s.col, r1: row, c1: col });
      return;
    }
    if (draggingEndpointRef.current !== null) {
      const moved = draggingEndpointRef.current === 0 ? { r0: row, c0: col } : { r1: row, c1: col };
      dragLine({ ...localLineRef.current, ...moved });
      return;
    }
    if (isDraggingLineBodyRef.current && lineBodyStartRef.current) {
      const { imgX, imgY } = screenToImgXY(e);
      const start = lineBodyStartRef.current;
      // Move in whole scan positions, so the drawn ends stay on the scan grid, and stop at the
      // scan edge; the shift is limited, not each end, so the line keeps its length and direction.
      const L0 = start.line;
      const dRow = clampShift(Math.round((imgY - start.imgY) / navH * scanRows), L0.r0, L0.r1, scanRows);
      const dCol = clampShift(Math.round((imgX - start.imgX) / NAV_SIZE * scanCols), L0.c0, L0.c1, scanCols);
      dragLine({ r0: L0.r0 + dRow, c0: L0.c0 + dCol, r1: L0.r1 + dRow, c1: L0.c1 + dCol });
      return;
    }
    if (isPaintingRef.current && maskRef.current) paintBrush(maskRef.current, row, col);
    if (isDraggingShapeRef.current && shapeStartRef.current) {
      setShapePreview({ r0: shapeStartRef.current.row, c0: shapeStartRef.current.col, r1: row, c1: col });
    }
  };

  const handleNavMouseUp = () => {
    if (isPanningRef.current) {
      isPanningRef.current = false;
      panStartRef.current = null;
      return;
    }
    if (isDraggingProbeRef.current) {
      isDraggingProbeRef.current = false;
      return;
    }
    if (isDrawingLineRef.current) {
      isDrawingLineRef.current = false;
      lineDrawStartRef.current = null;
      commitLine(localLineRef.current);
      return;
    }
    if (draggingEndpointRef.current !== null) {
      draggingEndpointRef.current = null;
      commitLine(localLineRef.current);
      return;
    }
    if (isDraggingLineBodyRef.current) {
      isDraggingLineBodyRef.current = false;
      lineBodyStartRef.current = null;
      commitLine(localLineRef.current);
      return;
    }
    if (isPaintingRef.current) {
      isPaintingRef.current = false;
      syncMaskToPython();
      return;
    }
    if (isDraggingShapeRef.current && shapeStartRef.current && maskRef.current && shapePreview) {
      const { r0, c0, r1, c1 } = shapePreview;
      if (maskTool === "circle") {
        const centerRow = (r0 + r1) / 2;
        const centerCol = (c0 + c1) / 2;
        const radius = Math.max(Math.abs(c1 - c0), Math.abs(r1 - r0)) / 2;
        fillCircleMask(maskRef.current, scanCols, scanRows, centerRow, centerCol, radius, maskPaintValue);
      } else {
        fillRectMask(maskRef.current, scanCols, scanRows, r0, c0, r1, c1, maskPaintValue);
      }
      setMaskRenderVersion((v) => v + 1);
      syncMaskToPython();
    }
    isDraggingShapeRef.current = false;
    shapeStartRef.current = null;
    setShapePreview(null);
  };

  // Leaving the image ends painting, keeping the strokes so far, and the shape and probe drags.
  const handleNavMouseLeave = () => {
    if (isPaintingRef.current) syncMaskToPython();
    isPaintingRef.current = false;
    isDraggingShapeRef.current = false;
    isDraggingProbeRef.current = false;
    setShapePreview(null);
  };

  const handleNavWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setNavZoom((z) => Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z * (e.deltaY > 0 ? 0.9 : 1.1))));
  };

  // =========================================================================
  // Plot mouse handlers
  // =========================================================================
  // Pointer position in the data coordinates of the 1D plot.
  function plotScreenToData(e: React.MouseEvent) {
    const cvs = plotCanvasRef.current;
    if (!cvs) return null;
    const { x: sx, y: sy } = pointerOffset(cvs, e);
    return {
      x: plotXMin + ((sx - MARGIN_LEFT_MIN) / PLOT_AREA_W) * (plotXMax - plotXMin),
      y: plotYMin + ((MARGIN_TOP + PLOT_AREA_H - sy) / PLOT_AREA_H) * (plotYMax - plotYMin),
    };
  }

  // Pointer position on the line-scan plot with that plot's layout; null before it is drawn.
  function heatmapPointer(e: React.MouseEvent) {
    const g = heatmapGeomRef.current;
    const cvs = plotCanvasRef.current;
    if (!g || !cvs) return null;
    const { x: sx, y: sy } = pointerOffset(cvs, e);
    const insideX = sx >= MARGIN_LEFT_MIN && sx <= MARGIN_LEFT_MIN + g.pwH;
    const insideY = sy >= MARGIN_TOP && sy <= MARGIN_TOP + PLOT_AREA_H;
    return { g, sx, sy, inside: insideX && insideY };
  }

  const handlePlotMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    // Line-scan plot: a click on empty space adds a guide, a click on a guide removes it (at
    // mouse-up), and a drag moves it.
    if (isLinescan) {
      const p = heatmapPointer(e);
      if (p && p.inside) {
        const r = heatmapRAt(p.g, p.sx);
        const tolR = ((p.g.rHi - p.g.rLo) / p.g.pwH) * 6;
        const near = hmGuides.findIndex((v) => Math.abs(v - r) <= tolR);
        if (near >= 0) {
          draggingGuideRef.current = { index: near, moved: false, added: false };
        } else {
          draggingGuideRef.current = { index: hmGuides.length, moved: false, added: true };
          setHmGuides((prev) => [...prev, r]);
        }
      }
      return;
    }
    isPlotPanRef.current = true;
    plotPanStartRef.current = {
      mx: e.clientX, my: e.clientY, xMin: plotXMin, xMax: plotXMax, yMin: plotYMin, yMax: plotYMax,
    };
  };

  const handlePlotMouseMove = (e: React.MouseEvent) => {
    if (draggingGuideRef.current) {
      const p = heatmapPointer(e);
      if (p) {
        const r = Math.max(p.g.rLo, Math.min(p.g.rHi, heatmapRAt(p.g, p.sx)));
        const idx = draggingGuideRef.current.index;
        draggingGuideRef.current.moved = true;
        setHmGuides((prev) => {
          const guides = [...prev];
          if (idx < guides.length) guides[idx] = r;
          return guides;
        });
      }
      return;
    }
    // Line-scan plot: r/k, position and the stack value under the pointer.
    if (isLinescan) {
      const p = heatmapPointer(e);
      if (p) setHmHover(p.inside ? heatmapValueAt(p.g, p.sx, p.sy) : null);
      return;
    }
    const start = plotPanStartRef.current;
    if (isPlotPanRef.current && start) {
      const dx = e.clientX - start.mx;
      const dy = e.clientY - start.my;
      const dxD = -(dx / PLOT_AREA_W) * (start.xMax - start.xMin);
      const dyD = (dy / PLOT_AREA_H) * (start.yMax - start.yMin);
      setPlotXMin(start.xMin + dxD);
      setPlotXMax(start.xMax + dxD);
      setPlotYMin(start.yMin + dyD);
      setPlotYMax(start.yMax + dyD);
      userZoomedRef.current = true;
      return;
    }
    const d = plotScreenToData(e);
    if (d) setCursorData(d);
  };

  const handlePlotMouseUp = () => {
    const d = draggingGuideRef.current;
    if (d) {
      draggingGuideRef.current = null;
      // A click (no drag) on an existing guide removes it; a click on empty space added one.
      if (!d.moved && !d.added) setHmGuides((prev) => prev.filter((_, i) => i !== d.index));
      return;
    }
    isPlotPanRef.current = false;
    plotPanStartRef.current = null;
  };

  const handlePlotMouseLeave = () => {
    isPlotPanRef.current = false;
    draggingGuideRef.current = null;
    setCursorData(null);
    setHmHover(null);
  };

  // A double click removes the line-scan plot's guides, or refits the 1D plot.
  const handlePlotDoubleClick = () => {
    if (isLinescan) {
      setHmGuides([]);
      return;
    }
    refitPlot();
  };

  // Zoom about the pointer.
  const handlePlotWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const d = plotScreenToData(e);
    if (!d) return;
    const f = e.deltaY > 0 ? 1.1 : 1 / 1.1;
    setPlotXMin((p) => d.x - (d.x - p) * f);
    setPlotXMax((p) => d.x + (p - d.x) * f);
    setPlotYMin((p) => d.y - (d.y - p) * f);
    setPlotYMax((p) => d.y + (p - d.y) * f);
    userZoomedRef.current = true;
  };

  // =========================================================================
  // Keyboard
  // =========================================================================
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
    switch (e.key.toLowerCase()) {
      case "r":
        // Reset the scan zoom and refit the plot.
        setNavZoom(1);
        setNavPanX(0);
        setNavPanY(0);
        refitPlot();
        break;
      case "x":
        if (!lockMask) switchMaskAction();
        break;
      case "c":
        if (!lockMask) clearMask();
        break;
      case "i":
        if (!lockMask) invertMask();
        break;
      case "1":
        if (!lockParameters) setPlotMode("Ik");
        break;
      case "2":
        if (!lockParameters) setPlotMode("Fk");
        break;
      case "3":
        if (!lockParameters) setPlotMode("Gr");
        break;
    }
  };

  // The wheel zooms the image and the plot, so it must not scroll the page; that needs
  // non-passive listeners, which React's onWheel is not.
  React.useEffect(() => {
    const preventScroll = (e: WheelEvent) => e.preventDefault();
    const opts: AddEventListenerOptions = { passive: false };
    const navBox = navCanvasRef.current?.parentElement;
    const plotBox = plotCanvasRef.current?.parentElement;
    navBox?.addEventListener("wheel", preventScroll, opts);
    plotBox?.addEventListener("wheel", preventScroll, opts);
    return () => {
      navBox?.removeEventListener("wheel", preventScroll);
      plotBox?.removeEventListener("wheel", preventScroll);
    };
  }, []);

  // =========================================================================
  // Standalone HTML export
  // =========================================================================
  const pendingExportRef = React.useRef<{ id: string; filename: string; profiles: boolean } | null>(null);
  const [exportBusy, setExportBusy] = React.useState(false);
  const [exportAnchor, setExportAnchor] = React.useState<HTMLElement | null>(null);
  const [localExportStatus, setLocalExportStatus] = React.useState("");

  // Estimated HTML size of each export option, from the bytes it embeds.
  const nkExport = byteLength(ikXBytes) / 4;   // ik_x_bytes is the float32 k axis
  const baseExportBytes = byteLength(navImageBytes)
    + byteLength(ikXBytes) + byteLength(ikYBytes) + byteLength(ikBgYBytes)
    + byteLength(fkXBytes) + byteLength(fkYBytes)
    + byteLength(grXBytes) + byteLength(grYBytes)
    + byteLength(pdfXBytes) + byteLength(pdfYBytes)
    + byteLength(linescanBytes) + byteLength(linescanAxisBytes)
    + nkExport * 8;   // offline_qq, float64
  const profileExportBytes = scanRows * scanCols * nkExport + nkExport * 8;   // uint8 profiles, float32 lo/hi
  const curvesHtmlSize = formatEstimatedHtmlSize(baseExportBytes);
  const profilesHtmlSize = formatEstimatedHtmlSize(baseExportBytes + profileExportBytes);

  // With profiles (encoding "uint8") the file also embeds every scan position's 8-bit profile, so
  // mask, probe and line stay interactive; without (encoding "full") it holds the curves only.
  const handleExportHtml = (profiles: boolean) => {
    const base = exportTitleSlug(title, "showpdf");
    const filename = `${base}_${scanRows}x${scanCols}_${profiles ? "quantized" : "exact"}.html`;
    const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    pendingExportRef.current = { id, filename, profiles };
    setLocalExportStatus(`Preparing ${filename}...`);
    setExportBusy(true);
    const encoding = profiles ? "uint8" : "full";
    setExportRequest(JSON.stringify({ mode: "single", download: true, id, filename, encoding }));
  };

  const handleExportChoice = (profiles: boolean) => {
    setExportAnchor(null);
    handleExportHtml(profiles);
  };

  // Download the file Python built for this view's request.
  React.useEffect(() => {
    const pending = pendingExportRef.current;
    if (!pending || exportPayloadId !== pending.id) return;
    const bytes = extractBytes(exportPayload);
    if (bytes.length === 0) return;
    // Copy out of the shared buffer if this is a subarray view.
    const wholeBuffer = bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength;
    const payload = wholeBuffer ? bytes : bytes.slice();
    const blob = new Blob([payload as BlobPart], { type: "text/html;charset=utf-8" });
    const filename = exportPayloadFilename || pending.filename;
    downloadBlob(blob, filename);
    pendingExportRef.current = null;
    setExportBusy(false);
    const contents = pending.profiles
      ? "exact float32 curves + quantized uint8 profiles"
      : "exact float32 curves";
    setLocalExportStatus(`Saved ${filename} (${formatSavedBytes(payload.byteLength)}, ${contents})`);
    // Release the embedded bytes from both sides so they are not re-synced.
    setExportRequest(JSON.stringify({ mode: "clear", id: `${pending.id}-clear` }));
  }, [exportPayload, exportPayloadId, exportPayloadFilename, setExportRequest]);

  // A failed export delivers no payload: release the button for a retry (the status shows the
  // error). An idle view drops its stale "Saved …" for a newer Python failure or export_html()
  // result (from another view, or code).
  React.useEffect(() => {
    const failed = exportStatus?.startsWith("Export failed");
    if (pendingExportRef.current && failed) {
      pendingExportRef.current = null;
      setExportBusy(false);
      setLocalExportStatus("");
    } else if (!pendingExportRef.current && (failed || exportStatus?.startsWith("Exported "))) {
      setLocalExportStatus("");
    }
  }, [exportStatus]);

  // =========================================================================
  // Offline recompute (standalone export, no kernel)
  // =========================================================================
  // The controls rerun the k-space pipeline in JS and write the results into the curve traits,
  // so the plot redraws as for Python's results. An export that carries the quantized profile of
  // every scan position (offlineReady) also recomputes I(k) for a new mask, probe or line.
  const offlineReady = !!(offline && offlineRecompute && offlineProfiles && offlineProfilesLo
    && offlineProfilesHi && offlineNk > 0 && offlineRows > 0 && offlineCols > 0);
  const offProf = React.useMemo(() => {
    return offlineProfiles ? extractBytes(offlineProfiles) : null;
  }, [offlineProfiles]);
  const offLo = React.useMemo(() => {
    return offlineProfilesLo ? extractFloat32(offlineProfilesLo) : null;
  }, [offlineProfilesLo]);
  const offHi = React.useMemo(() => {
    return offlineProfilesHi ? extractFloat32(offlineProfilesHi) : null;
  }, [offlineProfilesHi]);
  // Every export bakes the exact float64 k axis: the background fit must select the same k bins as
  // the library, which the float32 curve axis cannot guarantee at the range ends.
  const offQq = React.useMemo(() => {
    return offlineQq ? extractFloat64(offlineQq) : null;
  }, [offlineQq]);
  const offlineK = (): Float64Array | null => {
    if (offQq) return offQq;
    const k32 = ikXBytes ? extractFloat32(ikXBytes) : null;
    return k32 ? Float64Array.from(k32) : null;
  };

  // Mean profile I(k) over the scan positions `include` selects, and their number; null when it
  // selects none.
  const offlineMeanIk = (include: ScanPredicate) => {
    const q = offProf, lo = offLo, hi = offHi;
    if (!offlineReady || !q || !lo || !hi) return null;
    const sr = scanRows / offlineRows, sc = scanCols / offlineCols;   // 1 unless scan-binned
    const acc = new Float64Array(offlineNk);
    let n = 0;
    for (let r = 0; r < scanRows; r++) {
      const pr = profileIndex(r, sr, offlineRows);
      for (let c = 0; c < scanCols; c++) {
        if (!include(r, c)) continue;
        const base = (pr * offlineCols + profileIndex(c, sc, offlineCols)) * offlineNk;
        addProfile(acc, q, base, lo, hi);
        n++;
      }
    }
    if (n === 0) return null;
    for (let k = 0; k < offlineNk; k++) acc[k] /= n;
    return { ik: acc, n };
  };

  // The scan positions of the current single selection: the mask, the probe or the averaged line.
  const offlineSelectionInclude = (lineWidthPx: number): ScanPredicate | null => {
    if (analysisMode === "probe") {
      const half = Math.max(0, localProbeSize - 1);
      const r0 = localProbe.row - half, r1 = localProbe.row + half;
      const c0 = localProbe.col - half, c1 = localProbe.col + half;
      return (r, c) => r >= r0 && r <= r1 && c >= c0 && c <= c1;
    }
    if (analysisMode === "line") {
      // From the line's geometry, not the lineActive state, which lags behind a fresh draw.
      return lineBandIncludes(effectiveLine(localLineRef.current), lineWidthPx);
    }
    // An all-ones mask (no mask) selects every position.
    const m = maskRef.current;
    if (!m || m.every((v) => v !== 0)) return () => true;
    return (r, c) => m[r * scanCols + c] === 1;
  };

  // Parameters of an offline recompute, as ShowPDF._curve_params; `over` carries the value a
  // control just committed. Damping and g(r) need the density and are skipped while it is unset.
  const offlineParams = (over: OfflineOver) => {
    const dVal = over.densityValue ?? densityValue;
    return {
      gr: {
        kMinFit: over.kMinFit ?? kMinFit, kMaxFit: over.kMaxFit ?? kMaxFit,
        kMinWindow: over.kMinWindow ?? kMinWindow, kMaxWindow: over.kMaxWindow ?? kMaxWindow,
        kLowpass: over.kLowpass ?? kLowpass, kHighpass: over.kHighpass ?? kHighpass,
        rMax: over.rMax ?? rMax, rStep,
      },
      damp: over.damp ?? dampOrigin,
      density: dVal > 0 ? dVal : null,
      rCut: over.rCut ?? rCut,
      maxIter: over.maxIter ?? maxIter,
    };
  };

  // An export's status line (no kernel sets status_message): a refused recompute, or the same
  // window and density hints as ShowPDF.
  const [offlineError, setOfflineError] = React.useState("");
  // Until the browser first recomputes (or refuses to), an export shows the error Python baked
  // into it, if any; after that, its own status.
  const [offlineRecomputed, setOfflineRecomputed] = React.useState(false);
  const bakedError = ERROR_STATUS.test(statusMessage || "") ? statusMessage : "";
  // Whether the current k window has no positive weight, judged as the library does: with the
  // bounds computeGr resolves (0 window ends follow the fit range, 0 fit ends are the k axis ends)
  // and its float32 window.
  const offlineWindowEmpty = React.useMemo(() => {
    const qq = offline ? offlineK() : null;
    if (!qq || qq.length < 2) return false;
    const b = resolveWindowBounds(qq, { kMinFit, kMaxFit, kMinWindow, kMaxWindow });
    return !kWindow(qq, b.kMinWindow, b.kMaxWindow).some((w) => w > 0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [offline, offQq, ikXBytes, kMinFit, kMaxFit, kMinWindow, kMaxWindow]);
  const offlineStatus = offlineError
    || (offlineRecomputed ? "" : bakedError)
    || (offlineWindowEmpty ? WINDOW_HINT : "")
    || densityHint(densityValue, dampOrigin, plotMode);
  const shownStatus = offline ? offlineStatus : statusMessage;

  const setOfflineSelectionStats = (n: number) => {
    const total = scanRows * scanCols;
    setMaskPixelCount(n);
    setMaskFraction(total > 0 ? n / total : 0);
  };

  // The curves of the single selection. I(k) comes from the mask, probe or line when the export
  // carries the profiles, otherwise from the baked curve; an empty selection averages the full
  // scan, as calculate_Gr does without a mask.
  const recomputeOffline = (over: OfflineOver) => {
    const qf = offlineK();
    if (!qf) return;
    let ik: Float32Array | Float64Array | null = null;
    let fromSel = false;
    if (offlineReady) {
      const inc = offlineSelectionInclude(over.lineWidth ?? localLineWidth);
      const s = (inc ? offlineMeanIk(inc) : null) ?? offlineMeanIk(() => true);
      if (s) {
        ik = s.ik;
        fromSel = true;
        setOfflineSelectionStats(s.n);
      }
    }
    if (!ik) ik = ikYBytes ? extractFloat32(ikYBytes) : null;
    if (!ik) return;
    const p = offlineParams(over);
    let res, damped;
    try {
      res = computeGr(qf, Float64Array.from(ik), p.gr);
      damped = p.damp && p.density !== null
        ? dampOriginOscillations(qf, res, { density: p.density, rCut: p.rCut, maxIter: p.maxIter })
        : null;
    } catch (e) {
      // As a failed Python recompute: keep the previous curves, report why.
      setOfflineError(`Error: ${(e as Error).message}`);
      setOfflineRecomputed(true);
      return;
    }
    setOfflineError("");
    setOfflineRecomputed(true);
    // The I(k) plot shows the selection's profile.
    if (fromSel) {
      setIkYBytes(packF32(ik));
      setIkXBytes(packF32(qf));
    }
    setIkBgYBytes(packF32(res.bg));
    // F(k) and G(r) as the transform produced them: the damped ones when damping ran.
    setFkXBytes(packF32(qf));
    setFkYBytes(packF32(damped ? damped.fkWin : res.fkWin));
    const grShown = damped ? damped.gr : res.gr;
    setGrXBytes(packF32(res.r));
    setGrYBytes(packF32(grShown));
    // g(r) = 1 + G(r)/(4*pi*r*rho0) needs the density; empty while it is unset.
    if (p.density !== null) {
      setPdfXBytes(packF32(res.r));
      setPdfYBytes(packF32(computeGofR(res.r, grShown, p.density)));
    } else {
      setPdfXBytes(packF32(new Float64Array(0)));
      setPdfYBytes(packF32(new Float64Array(0)));
    }
  };

  // The line-scan stack, as ShowPDF._compute_linescan.
  const recomputeOfflineLinescan = (over: OfflineOver) => {
    const qf = offlineK();
    if (!offlineReady || !qf || !offProf || !offLo || !offHi) return;
    const clear = () => {
      setNLinescan(0);
      setLinescanNPoints(0);
      setLinescanBytes(packF32(new Float64Array(0)));
      setLinescanAxisBytes(packF32(new Float64Array(0)));
    };
    // From the line's geometry, not the lineActive state, which lags behind model.set during a
    // fresh draw: a band means a real line.
    const line = effectiveLine(localLineRef.current);
    const dr = line.r1 - line.r0, dc = line.c1 - line.c0, segSq = dr * dr + dc * dc;
    const inBand = lineBandIncludes(line, over.lineWidth ?? localLineWidth);
    if (!inBand) {
      clear();
      setOfflineSelectionStats(0);
      setOfflineError("");
      setOfflineRecomputed(true);
      return;
    }
    // Samples evenly spaced from A to the tip B, one per scan step along the longer axis; each
    // position in the band is averaged into its nearest sample (ShowPDF._linescan_samples).
    const nBins = linescanSampleCount(dr, dc, linescanMaxBins ?? 256);
    const Nk = offlineNk;
    const sr = scanRows / offlineRows, sc = scanCols / offlineCols;   // 1 unless scan-binned
    const sums: Float64Array[] = [];
    const counts: number[] = new Array(nBins).fill(0);
    for (let b = 0; b < nBins; b++) sums.push(new Float64Array(Nk));
    for (let r = 0; r < scanRows; r++) {
      const pr = profileIndex(r, sr, offlineRows);
      for (let c = 0; c < scanCols; c++) {
        if (!inBand(r, c)) continue;
        const bin = linescanSampleIndex(((r - line.r0) * dr + (c - line.c0) * dc) / segSq, nBins);
        const base = (pr * offlineCols + profileIndex(c, sc, offlineCols)) * Nk;
        addProfile(sums[bin], offProf, base, offLo, offHi);
        counts[bin]++;
      }
    }
    setOfflineSelectionStats(counts.reduce((a, b) => a + b, 0));
    const p = offlineParams(over);
    let axis: Float64Array | null = null;
    let M = 0;
    let stack: Float32Array | null = null;
    try {
      for (let b = 0; b < nBins; b++) {
        if (counts[b] === 0) continue;
        const binIk = sums[b];
        for (let k = 0; k < Nk; k++) binIk[k] /= counts[b];
        const res = computeGr(qf, binIk, p.gr);
        const damped = p.damp && p.density !== null
          ? dampOriginOscillations(qf, res, { density: p.density, rCut: p.rCut, maxIter: p.maxIter })
          : null;
        // The plot mode's curve; g(r) needs the density.
        let x: ArrayLike<number>, y: ArrayLike<number>;
        if (plotMode === "Ik") {
          x = qf;
          y = binIk;
        } else if (plotMode === "Fk") {
          x = qf;
          y = damped ? damped.fkWin : res.fkWin;
        } else if (plotMode === "gr") {
          if (p.density === null) continue;
          x = res.r;
          y = computeGofR(res.r, damped ? damped.gr : res.gr, p.density);
        } else {
          x = res.r;
          y = damped ? damped.gr : res.gr;
        }
        if (!stack) {
          axis = Float64Array.from(x);
          M = y.length;
          stack = new Float32Array(nBins * M).fill(NaN);
        }
        const nn = Math.min(M, y.length);
        for (let j = 0; j < nn; j++) stack[b * M + j] = y[j];
      }
    } catch (e) {
      // As a failed Python line-scan recompute: clear the stack, report why.
      clear();
      setOfflineError(`Line-scan error: ${(e as Error).message}`);
      setOfflineRecomputed(true);
      return;
    }
    setOfflineError("");
    setOfflineRecomputed(true);
    if (!stack || !axis) {
      clear();
      return;
    }
    setNLinescan(nBins);
    setLinescanNPoints(M);
    setLinescanBytes(new DataView(stack.buffer));
    setLinescanAxisBytes(packF32(axis));
  };

  // The line-scan rebuilds its stack; every other selection its curves.
  const recomputeOfflineAll = (over: OfflineOver) => {
    if (offlineReady && isLinescan) recomputeOfflineLinescan(over);
    else recomputeOffline(over);
  };

  // Recompute when the selection mode, probe, plot mode or line settings change; mask paints and
  // line edits recompute through syncMaskToPython and commitLine, at mouse-up rather than mid-drag.
  // Not on load: the exported curves are exact, while a recompute starts from the 8-bit profiles.
  const offlineLoadedRef = React.useRef(false);
  React.useEffect(() => {
    if (!offlineReady) return;
    if (!offlineLoadedRef.current) {
      offlineLoadedRef.current = true;
      return;
    }
    recomputeOfflineAll({});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    offlineReady, analysisMode, lineMode, plotMode, localProbe.row, localProbe.col, probeSize, linePerp,
    lineActive,
  ]);

  // =========================================================================
  // Control commits
  // =========================================================================
  // A slider commit that leaves its trait unchanged must not recompute: an export would replace
  // its exact curves with recomputed ones.
  const commitLineWidth = (v: number) => {
    if (v === lineWidth) return;
    setLineWidth(v);
    if (offline) recomputeOfflineAll({ lineWidth: v });
  };

  const commitKFit = (v: [number, number]) => {
    // An end left where it was keeps its value (MUI reports it clamped to the slider).
    const lo = committedFitEnd(v[0], kMinFit, kMinAvail, kMaxAvail);
    const hi = committedFitEnd(v[1], kMaxFit, kMinAvail, kMaxAvail);
    // The readout shows the kept ends, not the slider's bound, also when they are unchanged.
    setLocalKFit([lo, hi]);
    if (lo === kMinFit && hi === kMaxFit) return;
    setKMinFit(lo);
    setKMaxFit(hi);
    if (offline) recomputeOfflineAll({ kMinFit: lo, kMaxFit: hi });
  };

  const commitKWindow = (v: [number, number]) => {
    // An end left where it was keeps its setting (<= 0: follow the fit range).
    const lo = committedWindowEnd(v[0], kMinWindow, kMinFit, kMinAvail, kMaxAvail);
    const hi = committedWindowEnd(v[1], kMaxWindow, kMaxFit, kMinAvail, kMaxAvail);
    // The readout shows the window in effect, also when the setting is unchanged.
    setLocalKWin(windowInEffect(lo, hi, kMinFit, kMaxFit));
    if (lo === kMinWindow && hi === kMaxWindow) return;
    setKMinWindow(lo);
    setKMaxWindow(hi);
    if (offline) recomputeOfflineAll({ kMinWindow: lo, kMaxWindow: hi });
  };

  const commitRMax = (v: number) => {
    if (v === rMax) return;
    setRMax(v);
    if (offline) recomputeOfflineAll({ rMax: v });
  };

  const commitKLowpass = (v: number) => {
    if (v === kLowpass) return;
    setKLowpass(v);
    if (offline) recomputeOfflineAll({ kLowpass: v });
  };

  const commitKHighpass = (v: number) => {
    if (v === kHighpass) return;
    setKHighpass(v);
    if (offline) recomputeOfflineAll({ kHighpass: v });
  };

  const commitMaxIter = (v: number) => {
    if (v === maxIter) return;
    setMaxIter(v);
    if (offline) recomputeOfflineAll({ maxIter: v });
  };

  // r cut moves on the r grid of G(r) and its readouts show r_step's decimals. Values are rounded
  // to the grid, since MUI's keyboard path does not round.
  const rCutRange = rCutSlider(rStep, rMax);
  const roundRCut = (v: number) => Number(v.toFixed(rCutRange.decimals));
  const commitRCut = (v: number) => {
    const rc = roundRCut(v);
    if (rc === rCut) return;
    setRCut(rc);
    if (offline) recomputeOfflineAll({ rCut: rc });
  };

  const toggleDamp = (on: boolean) => {
    setDampOrigin(on);
    if (offline) recomputeOfflineAll({ damp: on });
  };

  // The Log switch refits at once with its new value.
  const handleLogScaleChange = (log: boolean) => {
    setIkLogScale(log);
    refitPlot(log);
  };

  // A count >= 1 other than the current one is set; otherwise the field shows the current count.
  const commitWaterfallCount = () => {
    const v = parseInt(localWfCount, 10);
    if (!isNaN(v) && v >= 1 && v !== waterfallCount) setWaterfallCount(v);
    else setLocalWfCount(String(waterfallCount));
  };

  // Density field: densityEditedRef records typing since the field got focus. Leaving the field
  // commits an edit: a number >= 0 sets the density, an empty field unsets it (0).
  const handleDensityFocus = () => {
    densityEditedRef.current = false;
  };

  const handleDensityChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    densityEditedRef.current = true;
    setLocalDensity(e.target.value);
  };

  const handleDensityBlur = (e: React.FocusEvent<HTMLInputElement>) => {
    const edited = densityEditedRef.current;
    densityEditedRef.current = false;
    // A malformed number reads as "" (validity.badInput): keep the density and clear the text
    // (restoring "" while unset would not re-render).
    if (e.currentTarget.validity.badInput) {
      e.currentTarget.value = "";
      setLocalDensity(fmtDensity(densityValue));
      return;
    }
    // Leaving the field unedited keeps the density (the field shows it rounded).
    if (!edited) return;
    const text = localDensity.trim();
    const v = text === "" ? 0 : parseFloat(text);
    if (!isNaN(v) && v >= 0 && v !== densityValue) {
      setDensityValue(v);
      if (offline) recomputeOfflineAll({ densityValue: v });
    } else {
      setLocalDensity(fmtDensity(densityValue));
    }
  };

  // =========================================================================
  // View
  // =========================================================================
  // Active processing is never invisible: without the controls (show_controls=False hides the
  // damping and filter sliders and the ⟂ button), a badge in the title row lists it.
  const hiddenProcessing: { short: string; full: string }[] = [];
  if (!showControls) {
    if (dampOrigin && densityValue > 0) {
      const full = `Low-r damping on (r cut ${rCut.toFixed(rCutRange.decimals)} Å, ${maxIter} iterations)`;
      hiddenProcessing.push({ short: "Damped", full });
    }
    if (kLowpass > 0) {
      const k = kLowpass.toFixed(3);
      hiddenProcessing.push({ short: `Lowpass ${k}`, full: `k lowpass ${k}` });
    }
    if (kHighpass > 0) {
      const k = kHighpass.toFixed(3);
      hiddenProcessing.push({ short: `Highpass ${k}`, full: `k highpass ${k}` });
    }
    if (analysisMode === "line" && linePerp) {
      hiddenProcessing.push({ short: "⟂ line", full: "Sampling perpendicular to the drawn line" });
    }
  }

  const isStatusError = ERROR_STATUS.test(shownStatus || "");
  let statusColor = colors.textMuted;
  if (isStatusError) statusColor = "#d32f2f";
  else if (computing) statusColor = colors.accent;

  const maskActionMode = maskAction === "add"
    ? "Add mode: painting excludes positions (red)"
    : "Subtract mode: painting re-includes positions";
  const maskActionTitle = `${maskActionMode}. Click to switch; switching resets the whole mask (X).`;
  const navDraws = analysisMode === "probe" || analysisMode === "line" || maskTool === "freeform";
  const navCursor = navDraws ? "crosshair" : "default";
  const lineText = lineActive
    ? `len ${lineLenPx.toFixed(1)} px · θ ${lineOrientDeg.toFixed(1)}°`
    : "draw a line…";
  const plotCursor = isLinescan ? "col-resize" : "default";

  // The hovered value, shown under the plot. With nothing hovered there is no readout line, so the
  // controls sit right under the plot.
  const xSymbol = isRAxis(plotMode) ? "r" : "k";
  const ySymbol = curveLabels(plotMode).symbol;
  let plotReadout = "";
  if (isLinescan) {
    if (hmHover) {
      const r = formatNumber(hmHover.r, 4);
      const pos = formatNumber(hmHover.pos, 2);
      const val = formatNumber(hmHover.val, 4);
      plotReadout = `${xSymbol} = ${r}, position = ${pos}, ${ySymbol} = ${val}`;
    }
  } else if (cursorData) {
    const v = curveValueAt(cursorData.x);
    if (v !== null) {
      plotReadout = `${xSymbol} = ${formatNumber(cursorData.x, 4)}, ${ySymbol} = ${formatNumber(v, 4)}`;
    }
  }

  // This view's own export status while it has one, otherwise Python's.
  const exportShownStatus = localExportStatus || exportStatus;
  const exportButtonTitle = exportShownStatus
    || "Export a standalone HTML viewer (saved to your Downloads folder)";
  const exportSpinner = <CircularProgress size={12} sx={{ color: colors.accent }} />;
  const curvesExportTitle = isLinescan
    ? "Curves stored as float32. The line-scan stack is fixed: selection, k-space, damping, density and plot-mode controls are locked (use live selection to keep them)."
    : "Curves stored as float32. Mask, probe and line are fixed; the k-space, damping and density controls still recompute.";

  // Styles that follow the theme.
  const typo = {
    title: { fontSize: 15, fontWeight: 600, color: colors.accent, fontFamily: FONT },
    label: { fontSize: 13, color: colors.text, fontFamily: FONT },
    labelSmall: { fontSize: 12, fontWeight: 600, color: colors.textMuted, fontFamily: FONT },
    value: { fontSize: 12, fontFamily: MONO, color: colors.text },
  };
  const parameterLabelSx = { ...typo.labelSmall, minWidth: 55, flexShrink: 0 };
  const readoutSx = { ...typo.value, ...numericSx, color: colors.textMuted };
  // content-box: the 1px border sits outside the canvas size, also where box-sizing is
  // border-box (the standalone export).
  const imageBox = {
    position: "relative" as const,
    boxSizing: "content-box" as const,
    border: `1px solid ${colors.border}`,
    overflow: "hidden",
    bgcolor: isDark ? "#111" : "#eee",
  };
  const overlayCanvasStyle = {
    position: "absolute", top: 0, left: 0, width: NAV_SIZE, height: navH, pointerEvents: "none",
  } as const;
  const themedSelect = {
    fontSize: 12,
    height: 28,
    bgcolor: colors.controlBg,
    color: colors.text,
    "& .MuiSvgIcon-root": { color: colors.text },
    "& .MuiOutlinedInput-notchedOutline": { borderColor: colors.border },
    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: colors.accent },
  };
  const themedMenuProps = {
    PaperProps: {
      sx: {
        bgcolor: colors.controlBg,
        color: colors.text,
        border: `1px solid ${colors.border}`,
        "& .MuiMenuItem-root:hover, & .MuiMenuItem-root.Mui-selected, & .MuiMenuItem-root.Mui-focusVisible": {
          bgcolor: colors.bg,
        },
      },
    },
    sx: { zIndex: 9999 },
  };

  return (
    <Box
      className="showpdf-root"
      tabIndex={0}
      onKeyDown={handleKeyDown}
      sx={{
        fontFamily: FONT, outline: "none", border: "none", p: 1, overflowX: "auto", bgcolor: colors.bg,
        color: colors.text, "&:focus": { outline: "none" }, "&:focus-visible": { outline: "none" },
      }}
    >
      {/* Title row: title, hidden-processing badge, status */}
      <Stack
        direction="row"
        alignItems="center"
        justifyContent="space-between"
        sx={{ mb: `${SPACING.SM}px`, gap: 1, minWidth: 0, flexWrap: "wrap", rowGap: "4px" }}
      >
        <Stack direction="row" alignItems="center" sx={{ minWidth: 0 }}>
          <Typography
            sx={{
              ...typo.title, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
            }}
            title={title || "PDF"}
          >
            {title || "PDF"}
          </Typography>
          {hiddenProcessing.length > 0 && (
            <Box
              component="span"
              title={hiddenProcessing.map((p) => p.full).join("; ")}
              sx={{
                ml: 0.5, px: 0.5, py: 0, fontSize: 11, fontWeight: 600, borderRadius: "3px",
                backgroundColor: alpha(colors.accent, 0.13), color: colors.accent,
                border: `1px solid ${alpha(colors.accent, 0.27)}`, whiteSpace: "nowrap", flexShrink: 0,
              }}
            >
              {hiddenProcessing.map((p) => p.short).join(" · ")}
            </Box>
          )}
        </Stack>
        <Stack direction="row" alignItems="center" gap={1} sx={{ minWidth: 0 }}>
          {computing && <CircularProgress size={14} sx={{ color: colors.accent }} />}
          {shownStatus && (
            <Typography
              title={shownStatus}
              aria-label={shownStatus}
              sx={{
                ...typo.labelSmall, maxWidth: 360, overflow: "hidden", textOverflow: "ellipsis",
                whiteSpace: "nowrap", color: statusColor,
              }}
            >
              {shownStatus}
            </Typography>
          )}
        </Stack>
      </Stack>

      <Stack direction="row" useFlexGap sx={{ flexWrap: "nowrap", gap: `${SPACING.LG}px` }}>
        {/* LEFT: Nav + Mask */}
        <Box sx={{ width: NAV_SIZE, flexShrink: 0 }}>
          <Stack
            direction="row"
            justifyContent="space-between"
            alignItems="center"
            sx={{ mb: `${SPACING.XS}px`, minHeight: 32, flexWrap: "wrap", rowGap: `${SPACING.XS}px` }}
          >
            <span style={{ fontSize: 12, fontFamily: FONT, color: colors.textMuted }}>
              Scan ({scanRows}×{scanCols})
            </span>
            <Stack direction="row" alignItems="center" spacing={`2px`}>
              {ANALYSIS_MODES.map((m) => (
                <Button
                  key={m}
                  size="small"
                  variant={analysisMode === m ? "contained" : "outlined"}
                  disabled={lockMask}
                  sx={{ ...compactBtn, minWidth: 38 }}
                  onClick={() => setAnalysisMode(m)}
                  aria-pressed={analysisMode === m}
                >
                  {m}
                </Button>
              ))}
            </Stack>
          </Stack>
          {analysisMode === "mask" && (
            <Stack
              direction="row"
              spacing={`2px`}
              alignItems="center"
              sx={{ mb: `${SPACING.XS}px`, flexWrap: "wrap", rowGap: `${SPACING.XS}px` }}
            >
              {MASK_TOOLS.map(([tool, glyph, name]) => (
                <Button
                  key={tool}
                  size="small"
                  variant={maskTool === tool ? "contained" : "outlined"}
                  disabled={lockMask}
                  sx={{ ...compactBtn, minWidth: 24 }}
                  onClick={() => setMaskTool(tool)}
                  title={name}
                  aria-label={name}
                >
                  {glyph}
                </Button>
              ))}
              <Button
                size="small"
                variant={maskAction === "add" ? "contained" : "outlined"}
                disabled={lockMask}
                sx={{ ...compactBtn, minWidth: 24 }}
                color={maskAction === "add" ? "primary" : "error"}
                title={maskActionTitle}
                aria-label={maskActionTitle}
                onClick={switchMaskAction}
              >
                {maskAction === "add" ? "+" : "−"}
              </Button>
              <Button
                size="small"
                sx={compactBtn}
                disabled={lockMask}
                title="Clear the mask (C)"
                onClick={clearMask}
              >
                Clear
              </Button>
              <Button
                size="small"
                sx={compactBtn}
                disabled={lockMask}
                title="Invert the mask and swap add/subtract mode (I inverts only)"
                onClick={handleInvertClick}
              >
                Invert
              </Button>
            </Stack>
          )}
          <Box
            sx={imageBox}
            style={{ width: NAV_SIZE, height: navH }}
            data-quantem-scientific-output="showpdf-navigation"
          >
            <canvas ref={navCanvasRef} style={{ width: NAV_SIZE, height: navH, display: "block" }} />
            <canvas ref={navOverlayRef} style={overlayCanvasStyle} />
            <canvas ref={navUiRef} style={overlayCanvasStyle} />
            <div
              style={{ position: "absolute", inset: 0, cursor: navCursor }}
              onMouseDown={handleNavMouseDown}
              onMouseMove={handleNavMouseMove}
              onMouseUp={handleNavMouseUp}
              onMouseLeave={handleNavMouseLeave}
              onWheel={handleNavWheel}
            />
          </Box>
          {showStats && (
            <div
              style={{
                fontSize: 12, fontFamily: MONO, color: colors.textMuted, marginTop: SPACING.XS,
                fontVariantNumeric: "tabular-nums",
              }}
            >
              {maskPixelCount} / {scanRows * scanCols} included ({(maskFraction * 100).toFixed(1)}%)
            </div>
          )}
          {showControls && (
            <Box sx={{ mt: `${SPACING.SM}px` }}>
              {analysisMode === "mask" && maskTool === "freeform" && (
                <Stack direction="row" alignItems="center" gap={1} sx={{ mb: `${SPACING.XS}px` }}>
                  <Typography sx={typo.labelSmall}>Brush</Typography>
                  <Slider
                    value={maskBrushSize}
                    onChange={(_, v) => setMaskBrushSize(v as number)}
                    disabled={lockMask}
                    min={1} max={20} size="small"
                    sx={{ ...sliderSmall, width: 80 }}
                    aria-label="Brush size"
                  />
                  <Typography sx={{ ...typo.value, ...numericSx }}>{maskBrushSize}</Typography>
                </Stack>
              )}
              {analysisMode === "probe" && (
                <Stack direction="row" alignItems="center" gap={1} sx={{ mb: `${SPACING.XS}px` }}>
                  <Typography
                    sx={typo.labelSmall}
                    title="Probe size: the square spans 2·size−1 scan positions per side"
                  >
                    Probe
                  </Typography>
                  <CommitSlider
                    value={localProbeSize}
                    onChange={(_, v) => setLocalProbeSize(v as number)}
                    onChangeCommitted={(_, v) => setProbeSize(v as number)}
                    disabled={lockMask}
                    min={1} max={100} size="small"
                    sx={{ ...sliderSmall, width: 80 }}
                    aria-label="Probe size"
                  />
                  <Typography sx={{ ...typo.value, ...numericSx }}>{localProbeSize}</Typography>
                  <Typography
                    sx={{
                      ...typo.value, ...numericSx, color: colors.textMuted, ml: `${SPACING.SM}px`,
                      minWidth: `${String(scanRows).length + String(scanCols).length + 5}ch`,
                    }}
                  >
                    @({localProbe.row}, {localProbe.col})
                  </Typography>
                </Stack>
              )}
              {analysisMode === "line" && (
                <Stack
                  direction="row"
                  alignItems="center"
                  gap={1}
                  sx={{ mb: `${SPACING.XS}px`, flexWrap: "wrap", rowGap: `${SPACING.XS}px` }}
                >
                  <Box sx={controlPairSx}>
                    <Typography sx={typo.labelSmall}>Width</Typography>
                    <CommitSlider
                      value={localLineWidth}
                      onChange={(_, v) => setLocalLineWidth(v as number)}
                      onChangeCommitted={(_, v) => commitLineWidth(v as number)}
                      disabled={lockMask}
                      min={1} max={Math.max(99, maxLineWidthSeen)} step={2} size="small"
                      sx={{ ...sliderSmall, width: 80 }}
                      aria-label="Line width"
                    />
                    <Typography sx={{ ...typo.value, ...numericSx }}>{localLineWidth}</Typography>
                  </Box>
                  <Button
                    size="small"
                    variant={linePerp ? "contained" : "outlined"}
                    disabled={lockMask}
                    sx={{ ...compactBtn, minWidth: 24 }}
                    onClick={() => setLinePerp(!linePerp)}
                    aria-pressed={linePerp}
                    title="Sample perpendicular to the drawn line"
                    aria-label="Sample perpendicular to the drawn line"
                  >
                    ⟂
                  </Button>
                  <Box sx={controlPairSx}>
                    {LINE_MODES.map(([lm, label]) => (
                      <Button
                        key={lm}
                        size="small"
                        variant={lineMode === lm ? "contained" : "outlined"}
                        disabled={lockMask}
                        sx={{ ...compactBtn, minWidth: 30 }}
                        onClick={() => setLineMode(lm)}
                        aria-pressed={lineMode === lm}
                      >
                        {label}
                      </Button>
                    ))}
                  </Box>
                  <Typography sx={{ ...typo.value, ...numericSx, color: colors.textMuted }} title={lineText}>
                    {lineText}
                  </Typography>
                  <Button size="small" sx={compactBtn} disabled={lockMask || !lineActive} onClick={clearLine}>
                    Clear
                  </Button>
                </Stack>
              )}
              <Stack direction="row" alignItems="center" gap={1}>
                <Typography sx={typo.labelSmall}>Cmap</Typography>
                <Select
                  size="small"
                  value={cmap}
                  onChange={(e) => setCmap(e.target.value)}
                  sx={{ ...themedSelect, minWidth: 80 }}
                  MenuProps={themedMenuProps}
                  inputProps={{ "aria-label": "Color map of the navigation image" }}
                >
                  {CMAP_OPTIONS.map((name) => (
                    <MenuItem key={name} value={name} sx={{ fontSize: 12 }}>{name}</MenuItem>
                  ))}
                </Select>
              </Stack>
            </Box>
          )}
          {exportEnabled && (
            <Box
              sx={{ mt: `${SPACING.SM}px`, pt: `${SPACING.SM}px`, borderTop: `1px solid ${colors.border}` }}
            >
              <Button
                size="small"
                variant="outlined"
                fullWidth
                disabled={exportBusy}
                startIcon={exportBusy ? exportSpinner : undefined}
                sx={{ ...compactBtn, "&.Mui-disabled": { color: colors.text, borderColor: colors.border } }}
                onClick={(e) => setExportAnchor(e.currentTarget)}
                aria-haspopup="menu"
                aria-controls={exportAnchor ? "showpdf-export-menu" : undefined}
                aria-expanded={exportAnchor ? "true" : undefined}
                title={exportButtonTitle}
              >
                {exportBusy ? "Exporting…" : "Export HTML"}
              </Button>
              <Menu
                id="showpdf-export-menu"
                anchorEl={exportAnchor}
                open={Boolean(exportAnchor)}
                onClose={() => setExportAnchor(null)}
                anchorOrigin={{ vertical: "bottom", horizontal: "left" }}
                transformOrigin={{ vertical: "top", horizontal: "left" }}
                MenuListProps={{ "aria-label": "Export options" }}
                {...themedMenuProps}
              >
                <MenuItem
                  sx={{ fontSize: 12 }}
                  onClick={() => handleExportChoice(false)}
                  title={curvesExportTitle}
                >
                  HTML exact float32 curves, fixed selection ({curvesHtmlSize})
                </MenuItem>
                <MenuItem
                  sx={{ fontSize: 12 }}
                  onClick={() => handleExportChoice(true)}
                  title="Also embeds every scan position's radial profile at 8 bits per k bin, so mask, probe and line stay interactive."
                >
                  HTML + quantized uint8 profiles, live selection ({profilesHtmlSize})
                </MenuItem>
              </Menu>
              {exportShownStatus && (
                <Typography
                  sx={{
                    ...typo.value, fontFamily: FONT, mt: `${SPACING.XS}px`, overflow: "hidden",
                    textOverflow: "ellipsis", whiteSpace: "nowrap",
                    color: exportShownStatus.startsWith("Export failed") ? "#d32f2f" : colors.textMuted,
                  }}
                  title={exportShownStatus}
                  aria-label={exportShownStatus}
                >
                  {exportShownStatus}
                </Typography>
              )}
            </Box>
          )}
        </Box>

        {/* RIGHT: Curves */}
        <Box sx={{ width: PLOT_W, flexShrink: 0 }}>
          {/* Spacer as tall as the mask tools row on the left, so both canvases line up in mask mode. */}
          {analysisMode === "mask" && (
            <Stack
              direction="row"
              spacing={`2px`}
              alignItems="center"
              aria-hidden
              sx={{
                mb: `${SPACING.XS}px`, flexWrap: "wrap", rowGap: `${SPACING.XS}px`, visibility: "hidden",
              }}
            >
              <Button size="small" sx={{ ...compactBtn, minWidth: 24 }}>▭</Button>
            </Stack>
          )}
          <Stack
            direction="row"
            alignItems="center"
            sx={{ mb: `${SPACING.XS}px`, minHeight: 32, flexWrap: "wrap", rowGap: `${SPACING.XS}px` }}
          >
            <Stack direction="row" spacing={`${SPACING.XS}px`}>
              {PLOT_MODES.map(([mode, label]) => (
                <Button
                  key={mode}
                  size="small"
                  variant={plotMode === mode ? "contained" : "outlined"}
                  sx={compactBtn}
                  aria-pressed={plotMode === mode}
                  disabled={lockParameters}
                  onClick={() => setPlotMode(mode)}
                >
                  {label}
                </Button>
              ))}
            </Stack>
            <Box sx={{ flex: 1 }} />
            <Stack direction="row" alignItems="center" gap={1}>
              {plotMode === "Ik" && (
                <>
                  <Typography sx={typo.labelSmall}>Log</Typography>
                  <Switch
                    checked={ikLogScale}
                    onChange={(e) => handleLogScaleChange(e.target.checked)}
                    size="small"
                    sx={switchSmall}
                    slotProps={{ input: { role: "switch", "aria-label": "Log scale for the I(k) axis" } }}
                  />
                  <Typography sx={typo.labelSmall} title="Show the fitted background B(k)">
                    Background
                  </Typography>
                  <Switch
                    checked={showBackground}
                    onChange={(e) => setShowBackground(e.target.checked)}
                    size="small"
                    sx={switchSmall}
                    slotProps={{ input: { role: "switch", "aria-label": "Show the fitted background B(k)" } }}
                  />
                </>
              )}
              {isLinescan && LINESCAN_VIEWS.map(([v, label]) => (
                <Button
                  key={v}
                  size="small"
                  variant={lineScanView === v ? "contained" : "outlined"}
                  sx={compactBtn}
                  aria-pressed={lineScanView === v}
                  onClick={() => setLineScanView(v)}
                >
                  {label}
                </Button>
              ))}
              {isLinescan && lineScanView === "waterfall" && (
                <>
                  <Typography
                    sx={typo.labelSmall}
                    title="Number of curves to stack (evenly sampled along the line)"
                  >
                    N
                  </Typography>
                  <input
                    type="number" min={1} step={1} value={localWfCount}
                    title="Number of curves to stack (evenly sampled along the line)"
                    aria-label="N, number of stacked curves"
                    onChange={(e) => setLocalWfCount(e.target.value)}
                    onBlur={commitWaterfallCount}
                    onKeyDown={blurOnEnter}
                    style={{
                      width: 48, boxSizing: "content-box", fontSize: 12, padding: "2px 4px", fontFamily: MONO,
                      background: colors.controlBg, color: colors.text, border: `1px solid ${colors.border}`,
                    }}
                  />
                </>
              )}
              <Button
                size="small"
                sx={compactBtn}
                onClick={() => refitPlot()}
                aria-label="Reset plot zoom"
                title="Refit the plot axes (R also resets the scan zoom)"
              >
                Reset
              </Button>
            </Stack>
          </Stack>
          <Box
            sx={imageBox}
            style={{ width: PLOT_W, height: PLOT_H }}
            data-quantem-scientific-output="showpdf-curves"
          >
            <canvas
              ref={plotCanvasRef}
              style={{ width: PLOT_W, height: PLOT_H, display: "block", cursor: plotCursor }}
              onMouseDown={handlePlotMouseDown}
              onMouseMove={handlePlotMouseMove}
              onMouseUp={handlePlotMouseUp}
              onMouseLeave={handlePlotMouseLeave}
              onDoubleClick={handlePlotDoubleClick}
              onWheel={handlePlotWheel}
            />
          </Box>
          {showStats && plotReadout !== "" && (
            <Typography
              sx={{
                ...typo.value, ...numericSx, mt: `${SPACING.XS}px`,
                overflow: "hidden", textOverflow: "ellipsis",
              }}
            >
              {plotReadout}
            </Typography>
          )}
          {showControls && (
            <Box
              sx={{ mt: `${SPACING.SM}px`, display: "flex", flexDirection: "column", gap: `${SPACING.XS}px` }}
            >
              <Stack direction="row" alignItems="center" gap={1} sx={{ flexWrap: "nowrap" }}>
                <Typography sx={parameterLabelSx}>k fit</Typography>
                <CommitSlider
                  value={localKFit}
                  onChange={(_, v) => setLocalKFit(v as [number, number])}
                  onChangeCommitted={(_, v) => commitKFit(v as [number, number])}
                  disabled={lockParameters}
                  min={kMinAvail} max={kMaxAvail} step={0.01} size="small"
                  sx={parameterSliderSx}
                  aria-label="k fit range (background fit)"
                />
                <Typography sx={{ ...readoutSx, minWidth: 90 }}>
                  [{localKFit[0].toFixed(2)}, {localKFit[1].toFixed(2)}]
                </Typography>
              </Stack>
              <Stack direction="row" alignItems="center" gap={1} sx={{ flexWrap: "nowrap" }}>
                <Typography sx={parameterLabelSx}>k window</Typography>
                <CommitSlider
                  value={localKWin}
                  onChange={(_, v) => setLocalKWin(v as [number, number])}
                  onChangeCommitted={(_, v) => commitKWindow(v as [number, number])}
                  disabled={lockParameters}
                  min={kMinAvail} max={kMaxAvail} step={0.01} size="small"
                  sx={parameterSliderSx}
                  aria-label="k window range"
                />
                <Typography sx={{ ...readoutSx, minWidth: 90 }}>
                  [{localKWin[0].toFixed(2)}, {localKWin[1].toFixed(2)}]
                </Typography>
              </Stack>
              <Stack direction="row" alignItems="center" gap={1} sx={{ flexWrap: "nowrap" }}>
                <Typography sx={parameterLabelSx}>r max</Typography>
                <CommitSlider
                  value={localRMax}
                  onChange={(_, v) => setLocalRMax(v as number)}
                  onChangeCommitted={(_, v) => commitRMax(v as number)}
                  disabled={lockParameters}
                  min={1} max={50} step={0.5} size="small"
                  sx={parameterSliderSx}
                  aria-label="r max"
                />
                <Typography sx={{ ...readoutSx, minWidth: 50 }}>{localRMax.toFixed(1)} Å</Typography>
              </Stack>
              <Stack direction="row" alignItems="center" gap={1} sx={{ flexWrap: "nowrap" }}>
                <Typography sx={parameterLabelSx}>k lowpass</Typography>
                <CommitSlider
                  value={localKLowpass}
                  onChange={(_, v) => setLocalKLowpass(v as number)}
                  onChangeCommitted={(_, v) => commitKLowpass(v as number)}
                  disabled={lockParameters}
                  min={0} max={0.1} step={0.001} size="small"
                  sx={parameterSliderSx}
                  aria-label="k lowpass filter"
                />
                <Typography sx={{ ...readoutSx, minWidth: 50 }}>
                  {localKLowpass > 0 ? localKLowpass.toFixed(3) : "off"}
                </Typography>
              </Stack>
              <Stack direction="row" alignItems="center" gap={1} sx={{ flexWrap: "nowrap" }}>
                <Typography sx={parameterLabelSx}>k highpass</Typography>
                <CommitSlider
                  value={localKHighpass}
                  onChange={(_, v) => setLocalKHighpass(v as number)}
                  onChangeCommitted={(_, v) => commitKHighpass(v as number)}
                  disabled={lockParameters}
                  min={0} max={0.1} step={0.001} size="small"
                  sx={parameterSliderSx}
                  aria-label="k highpass filter"
                />
                <Typography sx={{ ...readoutSx, minWidth: 50 }}>
                  {localKHighpass > 0 ? localKHighpass.toFixed(3) : "off"}
                </Typography>
              </Stack>
              <Stack direction="row" alignItems="center" gap={1} sx={{ flexWrap: "nowrap" }}>
                <Typography
                  sx={{ ...typo.labelSmall, flexShrink: 0 }}
                  title="Damp low-r oscillations (needs the density)"
                >
                  Damp
                </Typography>
                <Switch
                  checked={dampOrigin}
                  disabled={lockParameters}
                  onChange={(e) => toggleDamp(e.target.checked)}
                  size="small"
                  sx={switchSmall}
                  slotProps={{ input: { role: "switch", "aria-label": "Damp low-r oscillations" } }}
                />
                {dampOrigin && (
                  <>
                    <Typography
                      sx={{ ...typo.labelSmall, ml: 1, flexShrink: 0 }}
                      title="Upper end r (Å) of the low-r correction: G(r) is forced onto -4π ρ₀ r on [0, r cut]. Set it below the nearest-neighbour distance"
                    >
                      r_cut
                    </Typography>
                    <CommitSlider
                      value={localRCut}
                      onChange={(_, v) => setLocalRCut(roundRCut(v as number))}
                      onChangeCommitted={(_, v) => commitRCut(v as number)}
                      disabled={lockParameters}
                      min={rCutRange.min} max={rCutRange.max} step={rCutRange.step} size="small"
                      sx={{ ...sliderSmall, flex: 1 }}
                      aria-label="r cut (upper end of the low-r correction)"
                    />
                    <Typography sx={{ ...readoutSx, minWidth: `${rCutRange.decimals + 4}ch` }}>
                      {localRCut.toFixed(rCutRange.decimals)} Å
                    </Typography>
                  </>
                )}
              </Stack>
              {dampOrigin && (
                <Stack direction="row" alignItems="center" gap={1} sx={{ flexWrap: "nowrap" }}>
                  <Typography
                    sx={parameterLabelSx}
                    title="Maximum number of iterations of the low-r correction"
                  >
                    iterations
                  </Typography>
                  <CommitSlider
                    value={localMaxIter}
                    onChange={(_, v) => setLocalMaxIter(v as number)}
                    onChangeCommitted={(_, v) => commitMaxIter(v as number)}
                    disabled={lockParameters}
                    min={1} max={30} step={1} size="small"
                    sx={parameterSliderSx}
                    aria-label="iterations (maximum of the low-r correction)"
                  />
                  <Typography sx={{ ...readoutSx, minWidth: 50 }}>{localMaxIter}</Typography>
                </Stack>
              )}
              {(plotMode === "gr" || dampOrigin) && (
                <Stack direction="row" alignItems="center" gap={1}>
                  <Typography sx={parameterLabelSx}>Density</Typography>
                  <input
                    type="number"
                    step="0.001"
                    min={0}
                    value={localDensity}
                    placeholder="required"
                    title="Number density (atoms/Å³) for g(r) and the damping. It is never estimated; clear the field to unset it."
                    aria-label="Number density in atoms per cubic ångström"
                    disabled={lockParameters}
                    onFocus={handleDensityFocus}
                    onChange={handleDensityChange}
                    onBlur={handleDensityBlur}
                    onKeyDown={blurOnEnter}
                    style={{
                      width: 90, boxSizing: "content-box", fontSize: 12, padding: "3px 6px", fontFamily: MONO,
                      border: `1px solid ${colors.border}`,
                      background: isDark ? "#1a1a1a" : "#fff",
                      color: colors.text,
                      outline: "none",
                    }}
                  />
                  <Typography sx={typo.labelSmall}>Å⁻³</Typography>
                </Stack>
              )}
            </Box>
          )}
        </Box>
      </Stack>
    </Box>
  );
}

// A notebook saved with save_state=False keeps a static picture instead of the scan image, mask and
// line-scan stack. Reopened without a kernel, the widget shows that picture; live, it hides the
// picture's separate output.
function ShowPDFRoot() {
  const model = useModel();
  const rootRef = React.useRef<HTMLDivElement>(null);
  const [scanRows] = useModelState<number>("scan_rows");
  const [navImageBytes] = useModelState<DataView>("nav_image_bytes");
  const [preview] = useModelState<string>("_static_fallback_jpeg");
  const [previewMime] = useModelState<string>("_static_fallback_mime");
  const hasPixels = byteLength(navImageBytes) > 0;
  useHideStaticFallback(model, rootRef, hasPixels);
  if (!hasPixels && scanRows > 0 && preview) {
    return (
      <div ref={rootRef} style={{ fontFamily: FONT, padding: 8 }}>
        <img
          src={`data:${previewMime || "image/png"};base64,${preview}`}
          alt="Saved ShowPDF view"
          style={{ maxWidth: "100%", display: "block" }}
        />
        <div style={{ fontSize: 12, opacity: 0.7 }}>Saved preview · rerun the cell for interaction</div>
      </div>
    );
  }
  return (
    <div ref={rootRef}>
      <ShowPDFWidget />
    </div>
  );
}

export default { render: createRender(ShowPDFRoot) };
