import * as React from "react";
import { createRender, useModel, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Divider from "@mui/material/Divider";
import IconButton from "@mui/material/IconButton";
import InputAdornment from "@mui/material/InputAdornment";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import Select from "@mui/material/Select";
import Slider from "@mui/material/Slider";
import Stack from "@mui/material/Stack";
import Switch from "@mui/material/Switch";
import TextField from "@mui/material/TextField";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import ClearIcon from "@mui/icons-material/Clear";
import DownloadIcon from "@mui/icons-material/Download";
import FastForwardIcon from "@mui/icons-material/FastForward";
import FastRewindIcon from "@mui/icons-material/FastRewind";
import GridOffIcon from "@mui/icons-material/GridOff";
import GridOnIcon from "@mui/icons-material/GridOn";
import ImageIcon from "@mui/icons-material/Image";
import PauseIcon from "@mui/icons-material/Pause";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import RestartAltIcon from "@mui/icons-material/RestartAlt";
import ShowChartIcon from "@mui/icons-material/ShowChart";
import StarIcon from "@mui/icons-material/Star";
import StarBorderIcon from "@mui/icons-material/StarBorder";
import StopIcon from "@mui/icons-material/Stop";
import TableChartIcon from "@mui/icons-material/TableChart";
import VisibilityIcon from "@mui/icons-material/Visibility";
import VisibilityOffIcon from "@mui/icons-material/VisibilityOff";
import { downloadBlob, extractBytes, extractFloat32, formatNumber, preserveRestoredWidgetModelsOnSave } from "../format";
import { COLORMAPS, COLORMAP_NAMES, getGPUColormapEngine } from "../colormaps";
import { computeHistogramFromBytes, findDataRange, percentileClip } from "../stats";
import { applyHannWindow2D, autoEnhanceFFT, computeMagnitude, fft2dAsync, fftshift, getWebGPUFFT, nextPow2, WebGPUFFT } from "../fft";
import { formatScaleLabel, roundToNiceValue } from "../figure";
import { useTheme } from "../theme";
import { EmbeddedWidgetView } from "../embeddedWidget";

const SHOW1D_TO_SHOW2D_LINKED_TRAITS = [
  { source: "image_cmap", target: "cmap" },
  { source: "show_snapshot_fft", target: "show_fft" },
  { source: "show_stats" },
  { source: "show_controls" },
  { source: "controls_collapsed" },
  { source: "scale_bar_visible" },
];

type Marker = {
  x?: number;
  label?: string;
  kind?: string;
};

type ProfilePoint = {
  row?: number;
  col?: number;
};

type HoverPoint = {
  trace: number;
  point: number;
  x: number;
  y: number;
  px: number;
  py: number;
};

type PlotGeometry = {
  left: number;
  right: number;
  top: number;
  bottom: number;
  width: number;
  height: number;
  plotW: number;
  plotH: number;
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  logScale: boolean;
};

type ImageViewState = {
  zoom: number;
  panX: number;
  panY: number;
};

type SnapshotFftCacheEntry = {
  data: Float32Array;
  width: number;
  height: number;
  backend: string;
};

type TrialRanking = {
  label?: string;
  trace_index?: number;
  rank?: number;
  score?: number;
  lambda?: number;
  final_loss?: number;
  min_loss?: number;
  rmse?: number;
  flicker?: number;
  object_quality?: number;
  probe_quality?: number;
  alert_count?: number;
  starred?: boolean;
  hidden?: boolean;
  note?: string;
  tags?: string[];
};

type TrialAlert = {
  label?: string;
  kind?: string;
  severity?: string;
  message?: string;
};

type Show1DWritableFile = {
  write: (data: BlobPart) => Promise<void>;
  close: () => Promise<void>;
};

type Show1DFileHandle = {
  createWritable: () => Promise<Show1DWritableFile>;
};

type Show1DSavePickerOptions = {
  suggestedName?: string;
  types?: { description: string; accept: Record<string, string[]> }[];
};

type Show1DWindow = Window & typeof globalThis & {
  showSaveFilePicker?: (options?: Show1DSavePickerOptions) => Promise<Show1DFileHandle>;
};

const EMPTY_BYTES = new Uint8Array(0);
const DEFAULT_SIZE = { width: 820, height: 380 };
const controlRow = {
  display: "flex",
  alignItems: "center",
  flexWrap: "wrap",
  gap: "6px",
  px: 0.75,
  py: 0.5,
  width: "fit-content",
  maxWidth: "100%",
  boxSizing: "border-box" as const,
};
const controlPanel = {
  select: { minWidth: 90, fontSize: 11, "& .MuiSelect-select": { py: 0.5 } },
};
const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};
const sliderStyles = {
  small: {
    py: 0,
    "& .MuiSlider-thumb": { width: 10, height: 10 },
    "& .MuiSlider-rail": { height: 2 },
    "& .MuiSlider-track": { height: 2 },
    "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" },
  },
};
const switchStyles = {
  small: {
    "& .MuiSwitch-thumb": { width: 12, height: 12 },
    "& .MuiSwitch-switchBase": { padding: "4px" },
  },
};
const snapshotContrastPresets = [
  { value: "full", label: "full", low: 0, high: 100 },
  { value: "0.5-99.5", label: "0.5-99.5", low: 0.5, high: 99.5 },
  { value: "1-99", label: "1-99", low: 1, high: 99 },
  { value: "2-98", label: "2-98", low: 2, high: 98 },
  { value: "5-95", label: "5-95", low: 5, high: 95 },
] as const;
const FFT_DISPLAY_RANGE: [number, number] = [0, 1];
const typography = {
  label: { fontSize: 11 },
  value: { fontSize: 10, fontVariantNumeric: "tabular-nums" as const },
};

function useElementSize(ref: React.RefObject<HTMLElement | null>, fallback: { width: number; height: number }) {
  const [size, setSize] = React.useState(fallback);
  React.useLayoutEffect(() => {
    const node = ref.current;
    if (!node || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(([entry]) => {
      const rect = entry.contentRect;
      setSize({
        width: Math.max(280, Math.round(rect.width)),
        height: Math.max(180, Math.round(rect.height)),
      });
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, [ref]);
  return size;
}

function safeFloat32(data: DataView | ArrayBuffer | Uint8Array | null | undefined, expectedFloats: number): Float32Array {
  return extractFloat32(data ?? EMPTY_BYTES, expectedFloats) ?? new Float32Array(0);
}

function finiteExtent(values: Float32Array | number[], fallback: [number, number] = [0, 1]): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const value of values) {
    if (!Number.isFinite(value)) continue;
    if (value < lo) lo = value;
    if (value > hi) hi = value;
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return fallback;
  if (lo === hi) {
    const pad = Math.max(1, Math.abs(lo) * 0.05);
    return [lo - pad, hi + pad];
  }
  const pad = (hi - lo) * 0.04;
  return [lo - pad, hi + pad];
}

function xExtent(xData: Float32Array, nPoints: number): [number, number] {
  if (xData.length >= nPoints && nPoints > 0) return finiteExtent(xData.subarray(0, nPoints), [0, Math.max(1, nPoints - 1)]);
  return [0, Math.max(1, nPoints - 1)];
}

function yExtent(
  yData: Float32Array,
  nTraces: number,
  nPoints: number,
  xData: Float32Array,
  xRange: [number, number],
  logScale: boolean,
  hiddenTraces?: Set<number>,
): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (let trace = 0; trace < nTraces; trace += 1) {
    if (hiddenTraces?.has(trace)) continue;
    const offset = trace * nPoints;
    for (let point = 0; point < nPoints; point += 1) {
      const x = xData.length > point ? xData[point] : point;
      if (x < xRange[0] || x > xRange[1]) continue;
      const y = yData[offset + point];
      if (!Number.isFinite(y) || (logScale && y <= 0)) continue;
      if (y < lo) lo = y;
      if (y > hi) hi = y;
    }
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return logScale ? [1e-6, 1] : [0, 1];
  if (lo === hi) {
    const pad = Math.max(Math.abs(lo) * 0.05, logScale ? Math.max(lo * 0.5, 1e-9) : 1);
    return [Math.max(logScale ? Number.MIN_VALUE : -Infinity, lo - pad), hi + pad];
  }
  if (logScale) return [Math.max(lo / 1.25, Number.MIN_VALUE), hi * 1.25];
  const pad = (hi - lo) * 0.08;
  return [lo - pad, hi + pad];
}

function niceTicks(lo: number, hi: number, target = 5): number[] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo === hi) return [lo];
  const span = Math.abs(hi - lo);
  const raw = span / Math.max(1, target);
  const power = Math.pow(10, Math.floor(Math.log10(raw)));
  const unit = raw / power;
  const step = (unit >= 5 ? 5 : unit >= 2 ? 2 : 1) * power;
  const start = Math.ceil(Math.min(lo, hi) / step) * step;
  const end = Math.max(lo, hi);
  const ticks: number[] = [];
  for (let value = start; value <= end + step * 0.5; value += step) {
    ticks.push(value);
    if (ticks.length > 12) break;
  }
  return ticks.length ? ticks : [lo, hi];
}

function log10(value: number): number {
  return Math.log(value) / Math.LN10;
}

function transformY(value: number, logScale: boolean): number {
  return logScale ? log10(Math.max(value, Number.MIN_VALUE)) : value;
}

function dataToX(x: number, geom: PlotGeometry): number {
  return geom.left + ((x - geom.xMin) / Math.max(geom.xMax - geom.xMin, 1e-12)) * geom.plotW;
}

function dataToY(y: number, geom: PlotGeometry): number {
  const lo = transformY(geom.yMin, geom.logScale);
  const hi = transformY(geom.yMax, geom.logScale);
  return geom.top + (1 - (transformY(y, geom.logScale) - lo) / Math.max(hi - lo, 1e-12)) * geom.plotH;
}

function pixelToX(px: number, geom: PlotGeometry): number {
  const t = (px - geom.left) / Math.max(geom.plotW, 1);
  return geom.xMin + t * (geom.xMax - geom.xMin);
}

function pixelToY(py: number, geom: PlotGeometry): number {
  const lo = transformY(geom.yMin, geom.logScale);
  const hi = transformY(geom.yMax, geom.logScale);
  const value = hi - ((py - geom.top) / Math.max(geom.plotH, 1)) * (hi - lo);
  return geom.logScale ? Math.pow(10, value) : value;
}

function clampRange(range: [number, number], full: [number, number]): [number, number] {
  const fullSpan = Math.max(full[1] - full[0], 1e-12);
  let lo = Math.max(full[0], Math.min(full[1], range[0]));
  let hi = Math.max(full[0], Math.min(full[1], range[1]));
  if (hi - lo < fullSpan * 1e-6) {
    const center = (lo + hi) / 2;
    lo = center - fullSpan * 5e-7;
    hi = center + fullSpan * 5e-7;
  }
  return [lo, hi];
}

function cssColor(colors: string[], idx: number): string {
  return colors[idx] || ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"][idx % 6];
}

function shortMethodLabel(label: string): string {
  if (label === "frame_by_frame") return "frame";
  if (label.startsWith("joint_lambda_")) return `lambda ${label.slice("joint_lambda_".length)}`;
  return label.replace(/_/g, " ");
}

function trialKey(label: string): string {
  return String(label || "").toLowerCase().replace(/[^a-z0-9]+/g, "");
}

function uniqueStrings(values: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const value of values) {
    const text = String(value || "").trim();
    if (!text || seen.has(text)) continue;
    seen.add(text);
    out.push(text);
  }
  return out;
}

function lookupByTrialKey<T>(mapping: Record<string, T> | undefined, label: string): T | undefined {
  if (!mapping) return undefined;
  const key = trialKey(label);
  for (const [rawLabel, value] of Object.entries(mapping)) {
    if (trialKey(rawLabel) === key) return value;
  }
  return undefined;
}

function parseLambdaLabel(label: string): number {
  const match = String(label || "").replace(/_/g, " ").match(/lambda\s+([-+]?\d*\.?\d+(?:e[-+]?\d+)?)/i);
  if (!match) return Number.NaN;
  return Number(match[1]);
}

function isReferenceLabel(label: string): boolean {
  return trialKey(label).includes("reference");
}

function rankingNumber(row: TrialRanking, key: string): number {
  const clean = key === "default" ? "final_loss" : key;
  if (clean === "object_quality" || clean === "probe_quality") {
    const value = Number(row[clean]);
    return Number.isFinite(value) ? -value : Number.NaN;
  }
  const value = Number((row as Record<string, unknown>)[clean]);
  return Number.isFinite(value) ? value : Number.NaN;
}

function rankingDisplayValue(row: TrialRanking, key: string): string {
  const clean = key === "default" ? "final_loss" : key;
  const raw = Number((row as Record<string, unknown>)[clean]);
  return Number.isFinite(raw) ? formatNumber(raw, clean === "lambda" ? 3 : 4) : "";
}

function trimTrailingZeros(value: string): string {
  return value.replace(/(\.\d*?[1-9])0+$/, "$1").replace(/\.0+$/, "");
}

function isIntegerLike(value: number): boolean {
  return Number.isFinite(value) && Math.abs(value - Math.round(value)) < 1e-6;
}

function formatCompactValue(value: number, decimals = 3): string {
  if (!Number.isFinite(value)) return "";
  if (isIntegerLike(value)) return String(Math.round(value));
  const abs = Math.abs(value);
  if (abs !== 0 && (abs >= 10000 || abs < 0.001)) return value.toExponential(Math.min(decimals, 3));
  if (abs >= 1000) return trimTrailingZeros(value.toFixed(1));
  if (abs >= 100) return trimTrailingZeros(value.toFixed(2));
  return trimTrailingZeros(value.toFixed(decimals));
}

function formatAxisValue(value: number): string {
  return formatCompactValue(value, 3);
}

function formatRangeValue(value: number): string {
  return formatCompactValue(value, 2);
}

function backendBadge(backend: string): string {
  return backend === "webgpu" ? "" : backend;
}

function axisPositionText(value: number, label: string, unit: string): string {
  const formatted = formatAxisValue(value);
  if (!formatted) return "";
  const axis = label.trim() || "x";
  return `${axis} ${formatted}${unit.trim() ? ` ${unit.trim()}` : ""}`;
}

function labelAlreadyContainsValue(label: string, value: number): boolean {
  const formatted = formatAxisValue(value);
  if (!formatted) return false;
  const escaped = formatted.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`(^|[^0-9.\\-])${escaped}([^0-9.]|$)`).test(label);
}

function makeExportFilename(title: string, nTraces: number, nPoints: number, mode: "html" | "csv" | "png"): string {
  let slug = (title || "show1d").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
  while (slug.includes("__")) slug = slug.replace(/__/g, "_");
  if (!slug) slug = "show1d";
  return `${slug}_${nTraces}x${nPoints}.${mode}`;
}

function formatSavedBytes(bytes: number): string {
  const mb = Math.max(0, bytes) / (1024 * 1024);
  if (mb >= 100) return `${Math.round(mb)} MB`;
  if (mb >= 10) return `${mb.toFixed(1)} MB`;
  return `${mb.toFixed(2)} MB`;
}

function formatEstimatedHtmlSize(payloadBytes: number): string {
  const htmlBytes = Math.max(0, payloadBytes) * 4 / 3 + 600_000;
  const mb = htmlBytes / (1024 * 1024);
  if (mb >= 100) return `~${Math.round(mb)} MB`;
  if (mb >= 10) return `~${mb.toFixed(1)} MB`;
  return `~${mb.toFixed(2)} MB`;
}

function isAbortLikeError(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
}

function csvForTraces(xData: Float32Array, yData: Float32Array, labels: string[], nTraces: number, nPoints: number): string {
  const header = ["x", ...Array.from({ length: nTraces }, (_, idx) => labels[idx] || `trace ${idx + 1}`)];
  const rows = [header.join(",")];
  for (let point = 0; point < nPoints; point += 1) {
    const row = [xData.length > point ? xData[point] : point];
    for (let trace = 0; trace < nTraces; trace += 1) row.push(yData[trace * nPoints + point]);
    rows.push(row.map((value) => Number.isFinite(value) ? String(value) : "").join(","));
  }
  return `${rows.join("\n")}\n`;
}

function clampValue(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function clampSnapshotFps(value: number): number {
  return clampValue(Number.isFinite(value) ? value : 2, 0.25, 24);
}

function clampThumbnailSize(value: number): number {
  return Math.round(clampValue(Number.isFinite(value) ? value : 48, 24, 112));
}

function clampImageZoom(value: number): number {
  return clampValue(Number.isFinite(value) ? value : 1, 1, 32);
}

function clampImagePan(
  panX: number,
  panY: number,
  zoom: number,
  canvasW: number,
  canvasH: number,
  imageW: number,
  imageH: number,
): ImageViewState {
  const fit = Math.min(canvasW / Math.max(1, imageW), canvasH / Math.max(1, imageH));
  const drawW = Math.max(1, imageW * fit * zoom);
  const drawH = Math.max(1, imageH * fit * zoom);
  const maxX = Math.max(0, (drawW - canvasW) / 2) + 24;
  const maxY = Math.max(0, (drawH - canvasH) / 2) + 24;
  const cleanZoom = clampImageZoom(zoom);
  return {
    zoom: cleanZoom,
    panX: cleanZoom <= 1.0001 ? 0 : clampValue(panX, -maxX, maxX),
    panY: cleanZoom <= 1.0001 ? 0 : clampValue(panY, -maxY, maxY),
  };
}

function normaliseSnapshotContrastPreset(value: string): string {
  return snapshotContrastPresets.some((preset) => preset.value === value) ? value : "full";
}

function finitePercentileClip(data: Float32Array, low: number, high: number): { vmin: number; vmax: number } | null {
  let finiteCount = 0;
  for (let idx = 0; idx < data.length; idx += 1) {
    if (Number.isFinite(data[idx])) finiteCount += 1;
  }
  if (!finiteCount) return null;
  let source: Float32Array = data;
  if (finiteCount !== data.length) {
    source = new Float32Array(finiteCount);
    let outIdx = 0;
    for (let idx = 0; idx < data.length; idx += 1) {
      if (Number.isFinite(data[idx])) {
        source[outIdx] = data[idx];
        outIdx += 1;
      }
    }
  }
  const clipped = percentileClip(source, low, high);
  if (!Number.isFinite(clipped.vmin) || !Number.isFinite(clipped.vmax) || clipped.vmax <= clipped.vmin) return null;
  return { vmin: clipped.vmin, vmax: clipped.vmax };
}

function resolveSnapshotDisplayRange(data: Float32Array, presetValue: string): [number, number] {
  const preset = snapshotContrastPresets.find((item) => item.value === normaliseSnapshotContrastPreset(presetValue)) ?? snapshotContrastPresets[0];
  if (preset.value === "full") return finiteExtent(data, [0, 1]);
  const clipped = finitePercentileClip(data, preset.low, preset.high);
  if (!clipped) return finiteExtent(data, [0, 1]);
  return [clipped.vmin, clipped.vmax];
}

function blankSnapshotFft(): SnapshotFftCacheEntry {
  return { data: new Float32Array(0), width: 0, height: 0, backend: "empty" };
}

function normaliseFftDisplay(data: Float32Array, width: number, height: number): Float32Array {
  if (!data.length || width <= 0 || height <= 0) return new Float32Array(0);
  const logMag = new Float32Array(data.length);
  for (let idx = 0; idx < data.length; idx += 1) {
    const value = data[idx];
    logMag[idx] = Number.isFinite(value) && value > 0 ? Math.log1p(value) : 0;
  }
  const { min, max } = autoEnhanceFFT(logMag, width, height);
  const scale = 1 / Math.max(max - min, 1e-12);
  const out = new Float32Array(logMag.length);
  for (let idx = 0; idx < logMag.length; idx += 1) {
    out[idx] = clampValue((logMag[idx] - min) * scale, 0, 1);
  }
  return out;
}

async function computeSnapshotFft(
  image: Float32Array,
  width: number,
  height: number,
  useWindow: boolean,
  preferWebgpu: boolean,
  gpuFftRef: React.MutableRefObject<WebGPUFFT | null>,
): Promise<SnapshotFftCacheEntry> {
  if (!image.length || width <= 0 || height <= 0) return blankSnapshotFft();
  const fftW = nextPow2(width);
  const fftH = nextPow2(height);
  const real = new Float32Array(fftW * fftH);
  const source = image.slice();
  if (useWindow) applyHannWindow2D(source, width, height);
  for (let row = 0; row < height; row += 1) {
    real.set(source.subarray(row * width, row * width + width), row * fftW);
  }
  const imag = new Float32Array(real.length);
  if (preferWebgpu) {
    try {
      const fft = gpuFftRef.current ?? await getWebGPUFFT();
      gpuFftRef.current = fft;
      if (fft) {
        const result = await fft.fft2D(real, imag, fftW, fftH, false);
        fftshift(result.real, fftW, fftH);
        fftshift(result.imag, fftW, fftH);
        return {
          data: normaliseFftDisplay(computeMagnitude(result.real, result.imag), fftW, fftH),
          width: fftW,
          height: fftH,
          backend: "webgpu",
        };
      }
    } catch {
      gpuFftRef.current = null;
    }
  }
  const result = await fft2dAsync(real, imag, fftW, fftH, false);
  return {
    data: normaliseFftDisplay(result.magnitude, fftW, fftH),
    width: fftW,
    height: fftH,
    backend: "cpu",
  };
}

function snapshotGroupForImage(index: number, groupIndices: number[] | undefined): number {
  const raw = groupIndices?.[index];
  return Number.isFinite(raw) ? Math.max(0, Math.round(Number(raw))) : index;
}

function extractPackedImage(
  data: Float32Array,
  imageIndex: number,
  packedHeight: number,
  packedWidth: number,
  imageHeight: number,
  imageWidth: number,
): Float32Array {
  if (imageIndex < 0 || packedHeight <= 0 || packedWidth <= 0 || imageHeight <= 0 || imageWidth <= 0) {
    return new Float32Array(0);
  }
  const plane = packedHeight * packedWidth;
  const start = imageIndex * plane;
  if (start < 0 || start + plane > data.length) return new Float32Array(0);
  if (imageHeight === packedHeight && imageWidth === packedWidth) {
    return data.subarray(start, start + plane);
  }
  const out = new Float32Array(imageHeight * imageWidth);
  for (let row = 0; row < imageHeight; row += 1) {
    const src = start + row * packedWidth;
    out.set(data.subarray(src, src + imageWidth), row * imageWidth);
  }
  return out;
}

function drawFloatImage(
  canvas: HTMLCanvasElement,
  data: Float32Array,
  height: number,
  width: number,
  line: ProfilePoint[],
  lut: Uint8Array,
  colors: { bg: string; border: string; accent: string; text: string },
  view: ImageViewState = { zoom: 1, panX: 0, panY: 0 },
  drawSizeLabel = true,
  displayRange?: [number, number],
): void {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const cssW = Math.max(1, Math.round(rect.width));
  const cssH = Math.max(1, Math.round(rect.height));
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = colors.bg;
  ctx.fillRect(0, 0, cssW, cssH);
  if (!data.length || height <= 0 || width <= 0) return;

  const [vmin, vmax] = displayRange && Number.isFinite(displayRange[0]) && Number.isFinite(displayRange[1]) && displayRange[0] < displayRange[1]
    ? displayRange
    : finiteExtent(data, [0, 1]);
  const img = ctx.createImageData(width, height);
  const scale = 255 / Math.max(vmax - vmin, 1e-12);
  for (let i = 0; i < width * height; i += 1) {
    const raw = data[i];
    const finite = Number.isFinite(raw);
    const v = finite ? Math.max(0, Math.min(255, Math.round((raw - vmin) * scale))) : 0;
    const lutIdx = v * 3;
    img.data[i * 4] = lut[lutIdx];
    img.data[i * 4 + 1] = lut[lutIdx + 1];
    img.data[i * 4 + 2] = lut[lutIdx + 2];
    img.data[i * 4 + 3] = finite ? 255 : 0;
  }

  const bitmap = document.createElement("canvas");
  bitmap.width = width;
  bitmap.height = height;
  bitmap.getContext("2d")?.putImageData(img, 0, 0);
  const fit = Math.min(cssW / width, cssH / height);
  const zoom = clampImageZoom(view.zoom);
  const drawW = Math.max(1, width * fit * zoom);
  const drawH = Math.max(1, height * fit * zoom);
  const x0 = cssW / 2 - drawW / 2 + view.panX;
  const y0 = cssH / 2 - drawH / 2 + view.panY;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(bitmap, x0, y0, drawW, drawH);
  ctx.strokeStyle = colors.border;
  ctx.lineWidth = 1;
  ctx.strokeRect(x0, y0, drawW, drawH);
  if (line.length >= 2) {
    const a = line[0];
    const b = line[1];
    if (Number.isFinite(a.row) && Number.isFinite(a.col) && Number.isFinite(b.row) && Number.isFinite(b.col)) {
      ctx.strokeStyle = colors.accent;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x0 + Number(a.col) * fit * zoom, y0 + Number(a.row) * fit * zoom);
      ctx.lineTo(x0 + Number(b.col) * fit * zoom, y0 + Number(b.row) * fit * zoom);
      ctx.stroke();
      ctx.fillStyle = colors.accent;
      ctx.beginPath();
      ctx.arc(x0 + Number(a.col) * fit * zoom, y0 + Number(a.row) * fit * zoom, 3, 0, Math.PI * 2);
      ctx.arc(x0 + Number(b.col) * fit * zoom, y0 + Number(b.row) * fit * zoom, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  if (drawSizeLabel) {
    ctx.fillStyle = colors.text;
    ctx.font = "10px system-ui, sans-serif";
    ctx.fillText(`${width} x ${height}`, Math.max(4, x0 + 8), Math.min(cssH - 4, y0 + drawH - 8));
  }
}

function drawFloatThumbnail(
  ctx: CanvasRenderingContext2D,
  data: Float32Array,
  height: number,
  width: number,
  x: number,
  y: number,
  targetW: number,
  targetH: number,
  selected: boolean,
  lut: Uint8Array,
  colors: { bgAlt: string; border: string; accent: string },
  displayRange?: [number, number],
): void {
  if (!data.length || height <= 0 || width <= 0 || targetW <= 0 || targetH <= 0) return;
  ctx.save();
  ctx.fillStyle = colors.bgAlt;
  ctx.globalAlpha = 0.95;
  ctx.fillRect(x, y, targetW, targetH);
  ctx.globalAlpha = 1;

  const [vmin, vmax] = displayRange && Number.isFinite(displayRange[0]) && Number.isFinite(displayRange[1]) && displayRange[0] < displayRange[1]
    ? displayRange
    : finiteExtent(data, [0, 1]);
  const img = ctx.createImageData(width, height);
  const scale = 255 / Math.max(vmax - vmin, 1e-12);
  for (let i = 0; i < width * height; i += 1) {
    const raw = data[i];
    const finite = Number.isFinite(raw);
    const v = finite ? Math.max(0, Math.min(255, Math.round((raw - vmin) * scale))) : 0;
    const lutIdx = v * 3;
    img.data[i * 4] = lut[lutIdx];
    img.data[i * 4 + 1] = lut[lutIdx + 1];
    img.data[i * 4 + 2] = lut[lutIdx + 2];
    img.data[i * 4 + 3] = finite ? 255 : 0;
  }

  const bitmap = document.createElement("canvas");
  bitmap.width = width;
  bitmap.height = height;
  bitmap.getContext("2d")?.putImageData(img, 0, 0);
  const fit = Math.max(targetW / width, targetH / height);
  const drawW = Math.max(1, width * fit);
  const drawH = Math.max(1, height * fit);
  const drawX = x + (targetW - drawW) / 2;
  const drawY = y + (targetH - drawH) / 2;
  ctx.imageSmoothingEnabled = false;
  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, targetW, targetH);
  ctx.clip();
  ctx.drawImage(bitmap, drawX, drawY, drawW, drawH);
  ctx.restore();
  if (selected) {
    ctx.strokeStyle = colors.accent;
    ctx.lineWidth = 2;
    ctx.strokeRect(x + 0.5, y + 0.5, targetW - 1, targetH - 1);
  }
  ctx.restore();
}

function snapshotGroupThumbnailMetrics(count: number, maxWidth: number) {
  const n = Math.max(0, Math.min(9, count));
  if (n === 0 || maxWidth <= 0) return { cols: 0, rows: 0, cell: 0, width: 0, height: 0 };
  const cols = n === 1 ? 1 : n <= 3 ? n : n <= 4 ? 2 : 3;
  const rows = Math.ceil(n / cols);
  const cell = Math.max(5, Math.floor(maxWidth / cols));
  return { cols, rows, cell, width: cell * cols, height: cell * rows };
}

function drawSnapshotGroupThumbnail(
  ctx: CanvasRenderingContext2D,
  images: { data: Float32Array; height: number; width: number }[],
  x: number,
  y: number,
  size: number,
  selected: boolean,
  lut: Uint8Array,
  colors: { bgAlt: string; border: string; accent: string; text: string },
  contrastPreset: string,
): void {
  const valid = images.filter((image) => image.data.length && image.height > 0 && image.width > 0).slice(0, 9);
  if (!valid.length || size <= 0) return;
  const { cols, cell, width, height } = snapshotGroupThumbnailMetrics(valid.length, size);

  ctx.save();
  ctx.fillStyle = colors.bgAlt;
  ctx.globalAlpha = 0.96;
  ctx.fillRect(x, y, width, height);
  ctx.globalAlpha = 1;
  for (let idx = 0; idx < valid.length; idx += 1) {
    const col = idx % cols;
    const row = Math.floor(idx / cols);
    drawFloatThumbnail(
      ctx,
      valid[idx].data,
      valid[idx].height,
      valid[idx].width,
      x + col * cell,
      y + row * cell,
      cell,
      cell,
      false,
      lut,
      colors,
      resolveSnapshotDisplayRange(valid[idx].data, contrastPreset),
    );
  }
  ctx.strokeStyle = selected ? colors.accent : colors.border;
  ctx.lineWidth = selected ? 2 : 1;
  ctx.strokeRect(x + 0.5, y + 0.5, width - 1, height - 1);
  if (images.length > valid.length) {
    ctx.fillStyle = colors.text;
    ctx.font = "9px system-ui, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(`+${images.length - valid.length}`, x + width - 4, y + height - 4);
  }
  ctx.restore();
}

function drawPanelScaleBarHiDPI(
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

  const pxSize = pixelSize > 0 ? pixelSize : 1;
  const cssWidth = canvas.width / dpr;
  const cssHeight = canvas.height / dpr;
  if (cssWidth <= 0 || cssHeight <= 0 || imageWidth <= 0) return;

  const scaleX = cssWidth / imageWidth;
  const effectiveZoom = Math.max(zoom * scaleX, 1e-12);
  const targetBarPx = Math.min(60, cssWidth * 0.25);
  const nicePhysical = roundToNiceValue((targetBarPx / effectiveZoom) * pxSize);
  const barPx = Math.max(1, (nicePhysical / pxSize) * effectiveZoom);
  const margin = 12;
  const barThickness = 5;
  const fontSize = 16;
  const barY = cssHeight - margin;
  const barX = Math.max(margin, cssWidth - barPx - margin);

  ctx.save();
  ctx.scale(dpr, dpr);
  ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;
  ctx.fillStyle = "white";
  ctx.fillRect(barX, barY, Math.min(barPx, cssWidth - margin - barX), barThickness);

  ctx.font = `${fontSize}px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`;
  ctx.fillStyle = "white";
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText(formatScaleLabel(nicePhysical, unit || "px"), barX + Math.min(barPx, cssWidth - margin - barX) / 2, barY - 4);

  ctx.textAlign = "left";
  ctx.textBaseline = "bottom";
  ctx.fillText(`${zoom.toFixed(1)}×`, margin, cssHeight - margin + barThickness);
  ctx.restore();
}

function InteractiveFloatCanvas({
  data,
  height,
  width,
  lut,
  colors,
  label,
  selected = false,
  loading = false,
  displayRange,
  scaleBarVisible,
  pixelSize,
  pixelUnit,
  ariaLabel,
  view,
  onViewChange,
  onSelect,
}: {
  data: Float32Array;
  height: number;
  width: number;
  lut: Uint8Array;
  colors: { bgAlt: string; border: string; accent: string; text: string };
  label: string;
  selected?: boolean;
  loading?: boolean;
  displayRange?: [number, number];
  scaleBarVisible: boolean;
  pixelSize: number;
  pixelUnit: string;
  ariaLabel: string;
  view: ImageViewState;
  onViewChange: (view: ImageViewState) => void;
  onSelect?: () => void;
}) {
  const hostRef = React.useRef<HTMLDivElement>(null);
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const overlayRef = React.useRef<HTMLCanvasElement>(null);
  const viewRef = React.useRef<ImageViewState>(view);
  const pendingViewRef = React.useRef<ImageViewState | null>(null);
  const viewRafRef = React.useRef<number | null>(null);
  const dragRef = React.useRef<{ pointerId: number; x: number; y: number; panX: number; panY: number; zoom: number } | null>(null);
  const [drawTick, setDrawTick] = React.useState(0);

  const cleanView = React.useCallback((next: ImageViewState) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    return clampImagePan(
      next.panX,
      next.panY,
      next.zoom,
      rect?.width ?? Math.max(1, width),
      rect?.height ?? Math.max(1, height),
      width,
      height,
    );
  }, [height, width]);

  const scheduleView = React.useCallback((next: ImageViewState) => {
    const clean = cleanView(next);
    pendingViewRef.current = clean;
    viewRef.current = clean;
    if (viewRafRef.current !== null) return;
    viewRafRef.current = window.requestAnimationFrame(() => {
      viewRafRef.current = null;
      if (pendingViewRef.current) onViewChange(pendingViewRef.current);
    });
  }, [cleanView, onViewChange]);

  React.useLayoutEffect(() => {
    const node = hostRef.current;
    if (!node || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => setDrawTick((current) => current + 1));
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  React.useEffect(() => () => {
    if (viewRafRef.current !== null) window.cancelAnimationFrame(viewRafRef.current);
  }, []);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const localView = cleanView(view);
    viewRef.current = localView;
    drawFloatImage(canvas, data, height, width, [], lut, {
      bg: colors.bgAlt,
      border: selected ? colors.accent : colors.border,
      accent: colors.accent,
      text: colors.text,
    }, localView, false, displayRange);
  }, [cleanView, colors, data, displayRange, drawTick, height, lut, selected, view, width]);

  React.useEffect(() => {
    const overlay = overlayRef.current;
    const host = hostRef.current;
    if (!overlay || !host) return;
    const rect = host.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const cssW = Math.max(1, rect.width);
    const cssH = Math.max(1, rect.height);
    overlay.width = Math.round(cssW * dpr);
    overlay.height = Math.round(cssH * dpr);
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (!scaleBarVisible) return;
    const localView = cleanView(view);
    const pxSize = pixelSize > 0 ? pixelSize : 1;
    const unit = pixelSize > 0 ? pixelUnit : "px";
    drawPanelScaleBarHiDPI(overlay, dpr, localView.zoom, pxSize, unit, width);
  }, [cleanView, drawTick, pixelSize, pixelUnit, scaleBarVisible, view, width]);

  const handleWheel = React.useCallback((event: React.WheelEvent<HTMLCanvasElement>) => {
    event.preventDefault();
    onSelect?.();
    const rect = event.currentTarget.getBoundingClientRect();
    const current = viewRef.current;
    const nextZoom = clampImageZoom(current.zoom * Math.exp(-event.deltaY * 0.0014));
    const factor = nextZoom / Math.max(current.zoom, 1e-12);
    const pointerX = event.clientX - rect.left - rect.width / 2;
    const pointerY = event.clientY - rect.top - rect.height / 2;
    scheduleView({
      zoom: nextZoom,
      panX: pointerX - (pointerX - current.panX) * factor,
      panY: pointerY - (pointerY - current.panY) * factor,
    });
  }, [onSelect, scheduleView]);

  const handlePointerDown = React.useCallback((event: React.PointerEvent<HTMLCanvasElement>) => {
    onSelect?.();
    const current = viewRef.current;
    dragRef.current = {
      pointerId: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      panX: current.panX,
      panY: current.panY,
      zoom: current.zoom,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  }, [onSelect]);

  const handlePointerMove = React.useCallback((event: React.PointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    event.preventDefault();
    scheduleView({
      zoom: drag.zoom,
      panX: drag.panX + event.clientX - drag.x,
      panY: drag.panY + event.clientY - drag.y,
    });
  }, [scheduleView]);

  const stopDrag = React.useCallback((event: React.PointerEvent<HTMLCanvasElement>) => {
    if (dragRef.current?.pointerId === event.pointerId) dragRef.current = null;
  }, []);

  const resetView = React.useCallback((event: React.MouseEvent<HTMLCanvasElement>) => {
    event.stopPropagation();
    onSelect?.();
    scheduleView({ zoom: 1, panX: 0, panY: 0 });
  }, [onSelect, scheduleView]);

  return (
    <Box ref={hostRef} sx={{ position: "relative", width: "100%", height: "100%", overflow: "hidden", bgcolor: colors.bgAlt }}>
      <canvas
        ref={canvasRef}
        onWheel={handleWheel}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={stopDrag}
        onPointerCancel={stopDrag}
        onDoubleClick={resetView}
        aria-label={ariaLabel}
        style={{
          width: "100%",
          height: "100%",
          display: "block",
          cursor: dragRef.current ? "grabbing" : view.zoom > 1.01 ? "grab" : "zoom-in",
          touchAction: "none",
        }}
      />
      <canvas
        ref={overlayRef}
        aria-hidden="true"
        style={{
          position: "absolute",
          inset: 0,
          width: "100%",
          height: "100%",
          pointerEvents: "none",
          display: "block",
        }}
      />
      <Box
        sx={{
          position: "absolute",
          zIndex: 2,
          top: 6,
          left: 0,
          width: "100%",
          px: 0.75,
          boxSizing: "border-box",
          color: "rgba(255,255,255,0.95)",
          pointerEvents: "none",
          textAlign: "center",
          textShadow: "1px 1px 0 rgba(0,0,0,0.85), 0 0 3px rgba(0,0,0,0.75)",
          userSelect: "none",
        }}
      >
        <Typography sx={{ fontSize: 11, fontWeight: 700, lineHeight: 1.2, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {label}
        </Typography>
      </Box>
      {loading && (
        <Box
          sx={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            bgcolor: "rgba(255,255,255,0.38)",
            color: "#111",
            pointerEvents: "none",
            zIndex: 3,
          }}
        >
          <Typography sx={{ fontSize: 10, fontWeight: 600 }}>FFT...</Typography>
        </Box>
      )}
    </Box>
  );
}

function SnapshotImageCanvas({
  data,
  imageIndex,
  packedHeight,
  packedWidth,
  imageHeight,
  imageWidth,
  lut,
  fftLut,
  colors,
  showFft,
  fftWindow,
  preferWebgpu,
  contrastPreset,
  selected,
  label,
  scaleBarVisible,
  pixelSize,
  pixelUnit,
  imageView,
  fftView,
  fftCacheRef,
  fftGpuRef,
  onImageViewChange,
  onFftViewChange,
  onSelect,
}: {
  data: Float32Array;
  imageIndex: number;
  packedHeight: number;
  packedWidth: number;
  imageHeight: number;
  imageWidth: number;
  lut: Uint8Array;
  fftLut: Uint8Array;
  colors: { bgAlt: string; border: string; accent: string; text: string };
  showFft: boolean;
  fftWindow: boolean;
  preferWebgpu: boolean;
  contrastPreset: string;
  selected: boolean;
  label: string;
  scaleBarVisible: boolean;
  pixelSize: number;
  pixelUnit: string;
  imageView: ImageViewState;
  fftView: ImageViewState;
  fftCacheRef: React.MutableRefObject<Map<string, SnapshotFftCacheEntry>>;
  fftGpuRef: React.MutableRefObject<WebGPUFFT | null>;
  onImageViewChange: (view: ImageViewState) => void;
  onFftViewChange: (view: ImageViewState) => void;
  onSelect: () => void;
}) {
  const image = React.useMemo(
    () => extractPackedImage(data, imageIndex, packedHeight, packedWidth, imageHeight, imageWidth),
    [data, imageHeight, imageIndex, imageWidth, packedHeight, packedWidth],
  );
  const imageDisplayRange = React.useMemo(
    () => resolveSnapshotDisplayRange(image, contrastPreset),
    [contrastPreset, image],
  );
  const [fftEntry, setFftEntry] = React.useState<SnapshotFftCacheEntry | null>(null);
  const [fftLoading, setFftLoading] = React.useState(false);

  React.useEffect(() => {
    if (!showFft) {
      setFftEntry(null);
      setFftLoading(false);
      return;
    }
    const cacheKey = `${imageIndex}:${imageWidth}x${imageHeight}:${fftWindow ? "hann" : "raw"}:${preferWebgpu ? "gpu" : "cpu"}`;
    const cached = fftCacheRef.current.get(cacheKey);
    if (cached) {
      setFftEntry(cached);
      setFftLoading(false);
      return;
    }
    let canceled = false;
    setFftLoading(true);
    setFftEntry(null);
    void computeSnapshotFft(image, imageWidth, imageHeight, fftWindow, preferWebgpu, fftGpuRef)
      .then((entry) => {
        if (canceled) return;
        fftCacheRef.current.set(cacheKey, entry);
        setFftEntry(entry);
      })
      .catch(() => {
        if (!canceled) setFftEntry({ data: new Float32Array(0), width: imageWidth, height: imageHeight, backend: "error" });
      })
      .finally(() => {
        if (!canceled) setFftLoading(false);
      });
    return () => { canceled = true; };
  }, [fftCacheRef, fftGpuRef, fftWindow, image, imageHeight, imageIndex, imageWidth, preferWebgpu, showFft]);

  const displayFft = fftEntry ?? {
    data: new Float32Array(0),
    width: nextPow2(Math.max(1, imageWidth)),
    height: nextPow2(Math.max(1, imageHeight)),
    backend: fftLoading ? "..." : "pending",
  };
  const fftBackend = backendBadge(displayFft.backend);
  const fftLabel = fftBackend ? `FFT ${fftBackend}` : "FFT";

  return (
    <Box sx={{ display: "flex", flexDirection: "column", width: "100%", minWidth: 0, bgcolor: colors.bgAlt }}>
      <Box sx={{ position: "relative", aspectRatio: `${Math.max(1, imageWidth)} / ${Math.max(1, imageHeight)}`, minHeight: 0 }}>
        <InteractiveFloatCanvas
          data={image}
          height={imageHeight}
          width={imageWidth}
          lut={lut}
          colors={colors}
          label={label}
          selected={selected}
          displayRange={imageDisplayRange}
          scaleBarVisible={scaleBarVisible}
          pixelSize={pixelSize}
          pixelUnit={pixelUnit}
          ariaLabel={`${label} snapshot image`}
          view={imageView}
          onViewChange={onImageViewChange}
          onSelect={onSelect}
        />
      </Box>
      {showFft && (
        <Box
          sx={{
            position: "relative",
            aspectRatio: `${Math.max(1, displayFft.width)} / ${Math.max(1, displayFft.height)}`,
            minHeight: 0,
            borderTop: `1px solid ${colors.border}`,
          }}
        >
          <InteractiveFloatCanvas
            data={displayFft.data}
            height={displayFft.height}
            width={displayFft.width}
            lut={fftLut}
            colors={colors}
            label={fftLabel}
            selected={false}
            displayRange={FFT_DISPLAY_RANGE}
            scaleBarVisible={scaleBarVisible}
            pixelSize={1}
            pixelUnit="px"
            loading={fftLoading}
            ariaLabel={`${label} ${fftLabel}`}
            view={fftView}
            onViewChange={onFftViewChange}
            onSelect={onSelect}
          />
        </Box>
      )}
    </Box>
  );
}

function MiniHistogram({
  bins,
  dataMin,
  dataMax,
  clipMin,
  clipMax,
  backend,
  colors,
}: {
  bins: number[];
  dataMin: number;
  dataMax: number;
  clipMin: number;
  clipMax: number;
  backend: string;
  colors: { bgAlt: string; border: string; textMuted: string; accent: string };
}) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const backendText = backendBadge(backend);
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.round(rect.width));
    const height = Math.max(1, Math.round(rect.height));
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = colors.bgAlt;
    ctx.fillRect(0, 0, width, height);
    const displayBins = 64;
    const step = Math.max(1, Math.floor(bins.length / displayBins));
    const reduced = Array.from({ length: displayBins }, (_, idx) => {
      let sum = 0;
      for (let j = 0; j < step; j += 1) sum += bins[idx * step + j] || 0;
      return sum / step;
    });
    const maxBin = Math.max(...reduced, 0.001);
    const barW = width / displayBins;
    const span = Math.max(dataMax - dataMin, 1e-12);
    for (let idx = 0; idx < displayBins; idx += 1) {
      const barH = (reduced[idx] / maxBin) * Math.max(1, height - 3);
      const value = dataMin + ((idx + 0.5) / displayBins) * span;
      const insideClip = value >= clipMin && value <= clipMax;
      ctx.fillStyle = insideClip ? colors.accent : colors.textMuted;
      ctx.globalAlpha = insideClip ? 0.32 + 0.55 * (reduced[idx] / maxBin) : 0.16;
      ctx.fillRect(idx * barW, height - barH, Math.max(1, barW - 0.5), barH);
    }
    ctx.globalAlpha = 1;
    if (clipMax > clipMin && dataMax > dataMin) {
      const x0 = clampValue(((clipMin - dataMin) / span) * width, 0, width);
      const x1 = clampValue(((clipMax - dataMin) / span) * width, 0, width);
      ctx.strokeStyle = colors.accent;
      ctx.globalAlpha = 0.9;
      ctx.beginPath();
      ctx.moveTo(x0, 0);
      ctx.lineTo(x0, height);
      ctx.moveTo(x1, 0);
      ctx.lineTo(x1, height);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    ctx.strokeStyle = colors.border;
    ctx.strokeRect(0.5, 0.5, width - 1, height - 1);
  }, [bins, clipMax, clipMin, colors, dataMax, dataMin]);

  return (
    <Box>
      <Stack direction="row" alignItems="center" spacing={0.75} sx={{ mb: 0.35 }}>
        <Typography sx={{ fontSize: 11, fontWeight: 600 }}>Histogram</Typography>
        {backendText && <Typography sx={{ fontSize: 10, color: colors.textMuted }}>{backendText}</Typography>}
        <Box sx={{ flex: 1 }} />
        <Typography sx={{ fontSize: 10, color: colors.textMuted, fontVariantNumeric: "tabular-nums" }}>
          {formatRangeValue(clipMin)} - {formatRangeValue(clipMax)}
        </Typography>
      </Stack>
      <Box sx={{ height: 48, bgcolor: colors.bgAlt, border: `1px solid ${colors.border}`, borderRadius: 1, overflow: "hidden" }}>
        <canvas ref={canvasRef} style={{ width: "100%", height: "100%", display: "block" }} />
      </Box>
    </Box>
  );
}

function Show1DWidget() {
  const model = useModel();
  React.useEffect(() => preserveRestoredWidgetModelsOnSave(model), [model]);

  const [offlineForTheme] = useModelState<boolean>("_export_light");
  const [title] = useModelState<string>("title");
  const [showTitle] = useModelState<boolean>("show_title");
  const [yBytes] = useModelState<DataView>("y_bytes");
  const [xBytes] = useModelState<DataView>("x_bytes");
  const [nTraces] = useModelState<number>("n_traces");
  const [nPoints] = useModelState<number>("n_points");
  const [labels] = useModelState<string[]>("labels");
  const [colors] = useModelState<string[]>("colors");
  const [methodLabels] = useModelState<string[]>("method_labels");
  const [xLabel] = useModelState<string>("x_label");
  const [yLabel] = useModelState<string>("y_label");
  const [xUnit] = useModelState<string>("x_unit");
  const [yUnit] = useModelState<string>("y_unit");
  const [logScale, setLogScale] = useModelState<boolean>("log_scale");
  const [showStats, setShowStats] = useModelState<boolean>("show_stats");
  const [showLegend, setShowLegend] = useModelState<boolean>("show_legend");
  const [showGrid, setShowGrid] = useModelState<boolean>("show_grid");
  const [showControls] = useModelState<boolean>("show_controls");
  const controlsVisible = Boolean(showControls);
  const [lineWidth] = useModelState<number>("line_width");
  const [plotHeightPx, setPlotHeightPx] = useModelState<number>("plot_height_px");
  const [sidePanelWidthPx, setSidePanelWidthPx] = useModelState<number>("side_panel_width_px");
  const [focusedTrace, setFocusedTrace] = useModelState<number>("focused_trace");
  const [xRange, setXRange] = useModelState<number[]>("x_range");
  const [yRange, setYRange] = useModelState<number[]>("y_range");
  const [markers] = useModelState<Marker[]>("markers");
  const [statsMean] = useModelState<number[]>("stats_mean");
  const [statsMin] = useModelState<number[]>("stats_min");
  const [statsMax] = useModelState<number[]>("stats_max");
  const [statsStd] = useModelState<number[]>("stats_std");
  const [snapshotBytes] = useModelState<DataView>("snapshot_bytes");
  const [nSnapshots] = useModelState<number>("n_snapshots");
  const [snapshotHeight] = useModelState<number>("snapshot_height");
  const [snapshotWidth] = useModelState<number>("snapshot_width");
  const [snapshotIterations] = useModelState<number[]>("snapshot_iterations");
  const [snapshotLabels] = useModelState<string[]>("snapshot_labels");
  const [snapshotHeights] = useModelState<number[]>("snapshot_heights");
  const [snapshotWidths] = useModelState<number[]>("snapshot_widths");
  const [snapshotImageLabels] = useModelState<string[]>("snapshot_image_labels");
  const [starredSnapshotImageLabels, setStarredSnapshotImageLabels] = useModelState<string[]>("starred_snapshot_image_labels");
  const [hiddenSnapshotImageLabels, setHiddenSnapshotImageLabels] = useModelState<string[]>("hidden_snapshot_image_labels");
  const [trialNotes, setTrialNotes] = useModelState<Record<string, string>>("trial_notes");
  const [trialTags, setTrialTags] = useModelState<Record<string, string[]>>("trial_tags");
  const [showStarredOnly, setShowStarredOnly] = useModelState<boolean>("show_starred_only");
  const [trialSortKey, setTrialSortKey] = useModelState<string>("trial_sort_key");
  const [trialSortDescending, setTrialSortDescending] = useModelState<boolean>("trial_sort_descending");
  const [trialFilterText, setTrialFilterText] = useModelState<string>("trial_filter_text");
  const [topTrialCount, setTopTrialCount] = useModelState<number>("top_trial_count");
  const [trialRankings] = useModelState<TrialRanking[]>("trial_rankings");
  const [trialAlerts] = useModelState<TrialAlert[]>("trial_alerts");
  const [bestTrialLabel] = useModelState<string>("best_trial_label");
  const [runSummary] = useModelState<Record<string, unknown>>("run_summary");
  const [snapshotGroupIndices] = useModelState<number[]>("snapshot_group_indices");
  const [snapshotGroupIterations] = useModelState<number[]>("snapshot_group_iterations");
  const [snapshotGroupLabels] = useModelState<string[]>("snapshot_group_labels");
  const [nSnapshotGroups] = useModelState<number>("n_snapshot_groups");
  const [selectedSnapshotIdx, setSelectedSnapshotIdx] = useModelState<number>("selected_snapshot_idx");
  const [selectedSnapshotGroupIdx, setSelectedSnapshotGroupIdx] = useModelState<number>("selected_snapshot_group_idx");
  const [showSnapshots, setShowSnapshots] = useModelState<boolean>("show_snapshots");
  const [showSnapshotThumbnails, setShowSnapshotThumbnails] = useModelState<boolean>("show_snapshot_thumbnails");
  const [showSnapshotFft, setShowSnapshotFft] = useModelState<boolean>("show_snapshot_fft");
  const [snapshotFftWindow, setSnapshotFftWindow] = useModelState<boolean>("snapshot_fft_window");
  const [snapshotFftCmap, setSnapshotFftCmap] = useModelState<string>("snapshot_fft_cmap");
  const [snapshotContrastPreset, setSnapshotContrastPreset] = useModelState<string>("snapshot_contrast_preset");
  const [snapshotThumbnailSize, setSnapshotThumbnailSize] = useModelState<number>("snapshot_thumbnail_size");
  const [snapshotColumns, setSnapshotColumns] = useModelState<number>("snapshot_columns");
  const [imageCmap, setImageCmap] = useModelState<string>("image_cmap");
  const [scaleBarVisible] = useModelState<boolean>("scale_bar_visible");
  const [pixelSize] = useModelState<number>("pixel_size");
  const [pixelUnit] = useModelState<string>("pixel_unit");
  const [preferWebgpu] = useModelState<boolean>("prefer_webgpu");
  const [snapshotPlaying, setSnapshotPlaying] = useModelState<boolean>("snapshot_playing");
  const [snapshotFps, setSnapshotFps] = useModelState<number>("snapshot_fps");
  const [profileImageBytes] = useModelState<DataView>("profile_image_bytes");
  const [profileImageHeight] = useModelState<number>("profile_image_height");
  const [profileImageWidth] = useModelState<number>("profile_image_width");
  const [profileLine] = useModelState<ProfilePoint[]>("profile_line");
  const [, setExportRequest] = useModelState<string>("export_request");
  const [exportStatus] = useModelState<string>("export_status");
  const [exportEnabled] = useModelState<boolean>("export_enabled");
  const [exportPayload] = useModelState<DataView>("export_payload");
  const [exportPayloadId] = useModelState<string>("export_payload_id");
  const [exportPayloadFilename] = useModelState<string>("export_filename");
  const [, setHandoffRequest] = useModelState<string>("handoff_request");
  const [handoffStatus] = useModelState<string>("handoff_status");
  const [handoffEnabled] = useModelState<boolean>("handoff_enabled");
  const [preparedViewWidget] = useModelState<unknown>("prepared_view_widget");

  const { colors: themeColors } = useTheme(Boolean(offlineForTheme));
  const yData = React.useMemo(() => safeFloat32(yBytes, Math.max(0, nTraces * nPoints)), [yBytes, nTraces, nPoints]);
  const xData = React.useMemo(() => safeFloat32(xBytes, Math.max(0, nPoints)), [xBytes, nPoints]);
  const snapshotData = React.useMemo(
    () => safeFloat32(snapshotBytes, Math.max(0, nSnapshots * snapshotHeight * snapshotWidth)),
    [snapshotBytes, nSnapshots, snapshotHeight, snapshotWidth],
  );
  const profileImageData = React.useMemo(
    () => safeFloat32(profileImageBytes, Math.max(0, profileImageHeight * profileImageWidth)),
    [profileImageBytes, profileImageHeight, profileImageWidth],
  );
  const hiddenTrialLabels = React.useMemo(() => uniqueStrings(hiddenSnapshotImageLabels ?? []), [hiddenSnapshotImageLabels]);
  const starredTrialLabels = React.useMemo(() => uniqueStrings(starredSnapshotImageLabels ?? []), [starredSnapshotImageLabels]);
  const hiddenTrialKeys = React.useMemo(() => new Set(hiddenTrialLabels.map(trialKey)), [hiddenTrialLabels]);
  const starredTrialKeys = React.useMemo(() => new Set(starredTrialLabels.map(trialKey)), [starredTrialLabels]);
  const normalisedSortKey = React.useMemo(() => String(trialSortKey || "final_loss"), [trialSortKey]);
  const baseTrialRows = React.useMemo<TrialRanking[]>(() => {
    const rowsByKey = new Map<string, TrialRanking>();
    for (const row of trialRankings ?? []) {
      const label = String(row.label || "");
      if (label) rowsByKey.set(trialKey(label), { ...row, label });
    }
    for (let idx = 0; idx < nTraces; idx += 1) {
      const label = labels?.[idx] || `Trace ${idx + 1}`;
      const key = trialKey(label);
      const existing = rowsByKey.get(key) ?? {};
      const offset = idx * nPoints;
      let finalLoss = Number(existing.final_loss);
      let minLoss = Number(existing.min_loss);
      if (!Number.isFinite(finalLoss) || !Number.isFinite(minLoss)) {
        finalLoss = Number.NaN;
        minLoss = Number.POSITIVE_INFINITY;
        for (let point = 0; point < nPoints; point += 1) {
          const value = yData[offset + point];
          if (!Number.isFinite(value)) continue;
          finalLoss = value;
          if (value < minLoss) minLoss = value;
        }
        if (!Number.isFinite(minLoss)) minLoss = Number.NaN;
      }
      rowsByKey.set(key, {
        ...existing,
        label,
        trace_index: idx,
        lambda: Number.isFinite(Number(existing.lambda)) ? Number(existing.lambda) : parseLambdaLabel(label),
        final_loss: finalLoss,
        min_loss: minLoss,
        rmse: Number(existing.rmse),
        flicker: Number(existing.flicker),
        object_quality: Number(existing.object_quality),
        probe_quality: Number(existing.probe_quality),
        alert_count: Number(existing.alert_count) || 0,
        starred: starredTrialKeys.has(key),
        hidden: hiddenTrialKeys.has(key),
        note: lookupByTrialKey(trialNotes, label) ?? String(existing.note || ""),
        tags: lookupByTrialKey(trialTags, label) ?? existing.tags ?? [],
      });
    }
    return Array.from(rowsByKey.values()).filter((row) => Number.isFinite(Number(row.trace_index)));
  }, [hiddenTrialKeys, labels, nPoints, nTraces, starredTrialKeys, trialNotes, trialRankings, trialTags, yData]);
  const sortedTrialRows = React.useMemo(() => {
    const rows = [...baseTrialRows];
    if (normalisedSortKey === "label") {
      rows.sort((a, b) => String(a.label || "").localeCompare(String(b.label || "")));
    } else {
      rows.sort((a, b) => {
        const av = rankingNumber(a, normalisedSortKey);
        const bv = rankingNumber(b, normalisedSortKey);
        if (Number.isFinite(av) && Number.isFinite(bv) && av !== bv) return av - bv;
        if (Number.isFinite(av) !== Number.isFinite(bv)) return Number.isFinite(av) ? -1 : 1;
        return String(a.label || "").localeCompare(String(b.label || ""));
      });
    }
    if (trialSortDescending) rows.reverse();
    return rows.map((row, idx) => ({ ...row, rank: idx + 1, score: rankingNumber(row, normalisedSortKey) }));
  }, [baseTrialRows, normalisedSortKey, trialSortDescending]);
  const trialRowByKey = React.useMemo(() => {
    const out = new Map<string, TrialRanking>();
    for (const row of sortedTrialRows) {
      if (row.label) out.set(trialKey(row.label), row);
    }
    return out;
  }, [sortedTrialRows]);
  const filterText = String(trialFilterText || "").trim().toLowerCase();
  const topTrialLimit = Math.max(0, Number.isFinite(topTrialCount) ? Math.round(topTrialCount) : 0);
  const topTrialKeys = React.useMemo(() => {
    if (topTrialLimit <= 0) return new Set<string>();
    return new Set(sortedTrialRows.slice(0, topTrialLimit).map((row) => trialKey(row.label || "")));
  }, [sortedTrialRows, topTrialLimit]);
  const passesReviewFilter = React.useCallback((label: string) => {
    if (isReferenceLabel(label)) return true;
    const key = trialKey(label);
    const row = trialRowByKey.get(key);
    if (showStarredOnly && !starredTrialKeys.has(key)) return false;
    if (topTrialLimit > 0 && !topTrialKeys.has(key)) return false;
    if (filterText) {
      const haystack = [
        label,
        row?.note || "",
        ...(row?.tags ?? []),
      ].join(" ").toLowerCase();
      if (!haystack.includes(filterText)) return false;
    }
    return true;
  }, [filterText, showStarredOnly, starredTrialKeys, topTrialKeys, topTrialLimit, trialRowByKey]);
  const imageLabelForIndex = React.useCallback(
    (imageIdx: number) => snapshotImageLabels?.[imageIdx] || snapshotLabels?.[imageIdx] || `image ${imageIdx + 1}`,
    [snapshotImageLabels, snapshotLabels],
  );
  const isSnapshotImageHidden = React.useCallback(
    (imageIdx: number) => {
      const label = imageLabelForIndex(imageIdx);
      return hiddenTrialKeys.has(trialKey(label)) || !passesReviewFilter(label);
    },
    [hiddenTrialKeys, imageLabelForIndex, passesReviewFilter],
  );
  const hiddenTraceIndices = React.useMemo(
    () => Array.from({ length: nTraces }, (_, idx) => idx).filter((idx) => {
      const label = labels?.[idx] || `Trace ${idx + 1}`;
      return hiddenTrialKeys.has(trialKey(label)) || !passesReviewFilter(label);
    }),
    [hiddenTrialKeys, labels, nTraces, passesReviewFilter],
  );
  const hiddenTraceSet = React.useMemo(() => new Set(hiddenTraceIndices), [hiddenTraceIndices]);
  const visibleTraceIndices = React.useMemo(
    () => sortedTrialRows
      .map((row) => Number(row.trace_index))
      .filter((idx) => Number.isInteger(idx) && idx >= 0 && idx < nTraces && !hiddenTraceSet.has(idx)),
    [hiddenTraceSet, nTraces, sortedTrialRows],
  );
  const hiddenTrialCount = hiddenTrialLabels.length;
  const activeReviewCount = React.useMemo(
    () => sortedTrialRows.filter((row) => row.label && !hiddenTraceSet.has(Number(row.trace_index))).length,
    [hiddenTraceSet, sortedTrialRows],
  );

  const rootRef = React.useRef<HTMLDivElement>(null);
  const plotHostRef = React.useRef<HTMLDivElement>(null);
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const profileCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const plotSize = useElementSize(plotHostRef, DEFAULT_SIZE);
  const geomRef = React.useRef<PlotGeometry | null>(null);
  const [hover, setHover] = React.useState<HoverPoint | null>(null);
  const histogramSlotRef = React.useRef(9000 + Math.floor(Math.random() * 100000));
  const [snapshotHistogramBins, setSnapshotHistogramBins] = React.useState<number[]>(new Array(256).fill(0));
  const [snapshotHistogramRange, setSnapshotHistogramRange] = React.useState<[number, number]>([0, 1]);
  const [snapshotHistogramClipRange, setSnapshotHistogramClipRange] = React.useState<[number, number]>([0, 1]);
  const [snapshotHistogramBackend, setSnapshotHistogramBackend] = React.useState("cpu");
  const snapshotFftCacheRef = React.useRef<Map<string, SnapshotFftCacheEntry>>(new Map());
  const snapshotFftGpuRef = React.useRef<WebGPUFFT | null>(null);
  const [snapshotImageView, setSnapshotImageView] = React.useState<ImageViewState>({ zoom: 1, panX: 0, panY: 0 });
  const [snapshotFftView, setSnapshotFftView] = React.useState<ImageViewState>({ zoom: 1, panX: 0, panY: 0 });
  const hoverRafRef = React.useRef<number | null>(null);
  const pendingHoverRef = React.useRef<HoverPoint | null>(null);
  const [exportAnchor, setExportAnchor] = React.useState<HTMLElement | null>(null);
  const [viewMenuAnchor, setViewMenuAnchor] = React.useState<HTMLElement | null>(null);
  const [exportBusy, setExportBusy] = React.useState(false);
  const [localExportStatus, setLocalExportStatus] = React.useState("");
  const pendingExportRef = React.useRef<{ id: string; filename: string; handle: Show1DFileHandle | null } | null>(null);

  const fullXRange = React.useMemo(() => xExtent(xData, nPoints), [xData, nPoints]);
  const effectiveXRange: [number, number] = React.useMemo(() => {
    if (xRange?.length === 2 && Number.isFinite(xRange[0]) && Number.isFinite(xRange[1]) && xRange[0] < xRange[1]) {
      return [xRange[0], xRange[1]];
    }
    return fullXRange;
  }, [xRange, fullXRange]);
  const fullYRange = React.useMemo(
    () => yExtent(yData, nTraces, nPoints, xData, effectiveXRange, logScale, hiddenTraceSet),
    [yData, nTraces, nPoints, xData, effectiveXRange, logScale, hiddenTraceSet],
  );
  const effectiveYRange: [number, number] = React.useMemo(() => {
    if (yRange?.length === 2 && Number.isFinite(yRange[0]) && Number.isFinite(yRange[1]) && yRange[0] < yRange[1]) {
      if (!logScale || yRange[0] > 0) return [yRange[0], yRange[1]];
    }
    return fullYRange;
  }, [yRange, fullYRange, logScale]);
  const hasProfileImage = profileImageData.length >= profileImageHeight * profileImageWidth && profileImageHeight > 0 && profileImageWidth > 0;
  const hasSnapshots = snapshotData.length >= nSnapshots * snapshotHeight * snapshotWidth && nSnapshots > 0 && snapshotHeight > 0 && snapshotWidth > 0;
  const snapshotImageGroups = React.useMemo(
    () => Array.from({ length: Math.max(0, nSnapshots) }, (_, idx) => snapshotGroupForImage(idx, snapshotGroupIndices)),
    [nSnapshots, snapshotGroupIndices],
  );
  const groupCount = React.useMemo(() => {
    let highestGroup = -1;
    for (const groupIdx of snapshotImageGroups) highestGroup = Math.max(highestGroup, groupIdx);
    const metadataGroupCount = Math.max(
      0,
      Number.isFinite(nSnapshotGroups) ? Math.round(nSnapshotGroups) : 0,
      snapshotGroupIterations?.length ?? 0,
      highestGroup + 1,
    );
    return metadataGroupCount > 0 ? metadataGroupCount : (hasSnapshots ? nSnapshots : 0);
  }, [hasSnapshots, nSnapshotGroups, nSnapshots, snapshotGroupIterations, snapshotImageGroups]);
  const snapshotGroups = React.useMemo(() => {
    const groups = Array.from({ length: groupCount }, () => [] as number[]);
    for (let idx = 0; idx < nSnapshots; idx += 1) {
      const groupIdx = snapshotImageGroups[idx] ?? idx;
      if (groupIdx >= 0 && groupIdx < groupCount) groups[groupIdx].push(idx);
    }
    return groups;
  }, [groupCount, nSnapshots, snapshotImageGroups]);
  const legacySelectedSnapshot = nSnapshots > 0
    ? clampValue(Math.round(selectedSnapshotIdx < 0 ? nSnapshots - 1 : selectedSnapshotIdx), 0, nSnapshots - 1)
    : -1;
  const selectedGroup = groupCount > 0
    ? clampValue(
      Math.round(selectedSnapshotGroupIdx >= 0 ? selectedSnapshotGroupIdx : (snapshotImageGroups[legacySelectedSnapshot] ?? legacySelectedSnapshot)),
      0,
      groupCount - 1,
    )
    : -1;
  const selectedGroupAllImageIndices = selectedGroup >= 0 ? (snapshotGroups[selectedGroup] ?? []) : [];
  const selectedGroupImageIndices = selectedGroupAllImageIndices
    .filter((imageIdx) => !isSnapshotImageHidden(imageIdx))
    .sort((a, b) => {
      const aLabel = imageLabelForIndex(a);
      const bLabel = imageLabelForIndex(b);
      if (isReferenceLabel(aLabel) !== isReferenceLabel(bLabel)) return isReferenceLabel(aLabel) ? -1 : 1;
      const ar = Number(trialRowByKey.get(trialKey(aLabel))?.rank ?? Number.MAX_SAFE_INTEGER);
      const br = Number(trialRowByKey.get(trialKey(bLabel))?.rank ?? Number.MAX_SAFE_INTEGER);
      return ar - br || aLabel.localeCompare(bLabel);
    });
  const selectedSnapshot = selectedGroupImageIndices.includes(legacySelectedSnapshot)
    ? legacySelectedSnapshot
    : selectedGroupImageIndices[0] ?? -1;
  const selectedSnapshotIteration = selectedGroup >= 0 && Number.isFinite(snapshotGroupIterations?.[selectedGroup])
    ? Number(snapshotGroupIterations[selectedGroup])
    : selectedSnapshot >= 0 && snapshotIterations?.length > selectedSnapshot
      ? snapshotIterations[selectedSnapshot]
      : null;
  const sidePanelVisible = (showSnapshots && hasSnapshots) || hasProfileImage || showStats;
  const plotTitleVisible = false;
  const htmlSize = formatEstimatedHtmlSize((nTraces * nPoints + nSnapshots * snapshotHeight * snapshotWidth + profileImageHeight * profileImageWidth) * 4);
  const thumbnailSize = clampThumbnailSize(snapshotThumbnailSize);
  const plotHeight = Math.round(clampValue(Number.isFinite(plotHeightPx) ? plotHeightPx : 390, 220, 720));
  const sidePanelWidth = Math.round(clampValue(Number.isFinite(sidePanelWidthPx) ? sidePanelWidthPx : 360, 300, 640));
  const snapshotColumnCount = Math.round(clampValue(Number.isFinite(snapshotColumns) ? snapshotColumns : 2, 1, 4));
  const normalisedSnapshotContrastPreset = normaliseSnapshotContrastPreset(snapshotContrastPreset);
  const imageLut = React.useMemo(() => COLORMAPS[imageCmap] || COLORMAPS.cividis || COLORMAPS.gray, [imageCmap]);
  const snapshotFftLut = React.useMemo(
    () => COLORMAPS[snapshotFftCmap] || COLORMAPS.magma || imageLut,
    [imageLut, snapshotFftCmap],
  );
  const themedSelect = {
    ...controlPanel.select,
    bgcolor: themeColors.controlBg,
    color: themeColors.text,
    "& .MuiSelect-select": { py: 0.5 },
    "& .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.border },
    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.accent },
  };
  const themedMenuProps = {
    ...upwardMenuProps,
    PaperProps: { sx: { bgcolor: themeColors.controlBg, color: themeColors.text, border: `1px solid ${themeColors.border}` } },
  };

  const selectSnapshotGroup = React.useCallback((value: number) => {
    if (groupCount <= 0) return;
    const groupIdx = clampValue(Math.round(value), 0, groupCount - 1);
    setSelectedSnapshotGroupIdx(groupIdx);
    const imageIdx = snapshotGroups[groupIdx]?.find((idx) => !isSnapshotImageHidden(idx)) ?? -1;
    setSelectedSnapshotIdx(imageIdx);
  }, [groupCount, isSnapshotImageHidden, setSelectedSnapshotGroupIdx, setSelectedSnapshotIdx, snapshotGroups]);

  const selectSnapshotImage = React.useCallback((imageIdx: number) => {
    if (imageIdx < 0 || imageIdx >= nSnapshots) return;
    setSelectedSnapshotIdx(imageIdx);
    const groupIdx = snapshotImageGroups[imageIdx] ?? selectedGroup;
    if (groupIdx >= 0 && groupIdx < groupCount) setSelectedSnapshotGroupIdx(groupIdx);
  }, [groupCount, nSnapshots, selectedGroup, setSelectedSnapshotGroupIdx, setSelectedSnapshotIdx, snapshotImageGroups]);

  const toggleStarredTrial = React.useCallback((label: string) => {
    const clean = String(label || "").trim();
    if (!clean) return;
    const key = trialKey(clean);
    const current = uniqueStrings(starredSnapshotImageLabels ?? []);
    if (current.some((value) => trialKey(value) === key)) {
      setStarredSnapshotImageLabels(current.filter((value) => trialKey(value) !== key));
      return;
    }
    setStarredSnapshotImageLabels([...current, clean]);
  }, [setStarredSnapshotImageLabels, starredSnapshotImageLabels]);

  const hideTrial = React.useCallback((label: string) => {
    const clean = String(label || "").trim();
    if (!clean) return;
    const key = trialKey(clean);
    const hidden = uniqueStrings(hiddenSnapshotImageLabels ?? []);
    const starred = uniqueStrings(starredSnapshotImageLabels ?? []);
    if (!hidden.some((value) => trialKey(value) === key)) {
      setHiddenSnapshotImageLabels([...hidden, clean]);
    }
    if (starred.some((value) => trialKey(value) === key)) {
      setStarredSnapshotImageLabels(starred.filter((value) => trialKey(value) !== key));
    }
  }, [hiddenSnapshotImageLabels, setHiddenSnapshotImageLabels, setStarredSnapshotImageLabels, starredSnapshotImageLabels]);

  const showAllTrials = React.useCallback(() => {
    setHiddenSnapshotImageLabels([]);
  }, [setHiddenSnapshotImageLabels]);

  const setTrialNoteForLabel = React.useCallback((label: string, note: string) => {
    const clean = String(label || "").trim();
    if (!clean) return;
    const key = trialKey(clean);
    const next: Record<string, string> = {};
    for (const [rawLabel, value] of Object.entries(trialNotes ?? {})) {
      if (trialKey(rawLabel) !== key && String(value || "").trim()) next[rawLabel] = String(value);
    }
    if (note.trim()) next[clean] = note;
    setTrialNotes(next);
  }, [setTrialNotes, trialNotes]);

  const toggleTrialTag = React.useCallback((label: string, tag: string) => {
    const clean = String(label || "").trim();
    const cleanTag = String(tag || "").trim();
    if (!clean || !cleanTag) return;
    const key = trialKey(clean);
    const next: Record<string, string[]> = {};
    let current: string[] = [];
    for (const [rawLabel, values] of Object.entries(trialTags ?? {})) {
      const list = Array.isArray(values) ? values.map(String).filter(Boolean) : [];
      if (trialKey(rawLabel) === key) current = list;
      else if (list.length) next[rawLabel] = uniqueStrings(list);
    }
    next[clean] = current.some((value) => value === cleanTag)
      ? current.filter((value) => value !== cleanTag)
      : uniqueStrings([...current, cleanTag]);
    if (!next[clean].length) delete next[clean];
    setTrialTags(next);
  }, [setTrialTags, trialTags]);

  const starBestTrial = React.useCallback(() => {
    const best = sortedTrialRows.find((row) => row.label && !hiddenTrialKeys.has(trialKey(row.label)));
    if (!best?.label) return;
    const current = uniqueStrings(starredSnapshotImageLabels ?? []);
    if (!current.some((value) => trialKey(value) === trialKey(best.label || ""))) {
      setStarredSnapshotImageLabels([...current, best.label]);
    }
  }, [hiddenTrialKeys, setStarredSnapshotImageLabels, sortedTrialRows, starredSnapshotImageLabels]);

  const hideWorstTrial = React.useCallback(() => {
    const candidates = sortedTrialRows.filter((row) => row.label && !hiddenTrialKeys.has(trialKey(row.label)) && !starredTrialKeys.has(trialKey(row.label)));
    const worst = candidates[candidates.length - 1];
    if (worst?.label) hideTrial(worst.label);
  }, [hiddenTrialKeys, hideTrial, sortedTrialRows, starredTrialKeys]);

  const scheduleHover = React.useCallback((value: HoverPoint | null) => {
    pendingHoverRef.current = value;
    if (hoverRafRef.current !== null) return;
    hoverRafRef.current = window.requestAnimationFrame(() => {
      hoverRafRef.current = null;
      setHover(pendingHoverRef.current);
    });
  }, []);

  React.useEffect(() => () => {
    if (hoverRafRef.current !== null) window.cancelAnimationFrame(hoverRafRef.current);
  }, []);

  React.useEffect(() => {
    if (snapshotThumbnailSize !== thumbnailSize) setSnapshotThumbnailSize(thumbnailSize);
  }, [setSnapshotThumbnailSize, snapshotThumbnailSize, thumbnailSize]);

  React.useEffect(() => {
    if (snapshotColumns !== snapshotColumnCount) setSnapshotColumns(snapshotColumnCount);
  }, [setSnapshotColumns, snapshotColumnCount, snapshotColumns]);

  React.useEffect(() => {
    if (focusedTrace >= 0 && hiddenTraceSet.has(focusedTrace)) setFocusedTrace(-1);
  }, [focusedTrace, hiddenTraceSet, setFocusedTrace]);

  React.useEffect(() => {
    if (plotHeightPx !== plotHeight) setPlotHeightPx(plotHeight);
  }, [plotHeight, plotHeightPx, setPlotHeightPx]);

  React.useEffect(() => {
    if (sidePanelWidthPx !== sidePanelWidth) setSidePanelWidthPx(sidePanelWidth);
  }, [sidePanelWidth, sidePanelWidthPx, setSidePanelWidthPx]);

  React.useEffect(() => {
    if (!COLORMAPS[imageCmap]) setImageCmap("cividis");
  }, [imageCmap, setImageCmap]);

  React.useEffect(() => {
    if (!COLORMAPS[snapshotFftCmap]) setSnapshotFftCmap("magma");
  }, [setSnapshotFftCmap, snapshotFftCmap]);

  React.useEffect(() => {
    if (snapshotContrastPreset !== normalisedSnapshotContrastPreset) {
      setSnapshotContrastPreset(normalisedSnapshotContrastPreset);
    }
  }, [normalisedSnapshotContrastPreset, setSnapshotContrastPreset, snapshotContrastPreset]);

  React.useEffect(() => {
    snapshotFftCacheRef.current.clear();
  }, [snapshotData, snapshotHeight, snapshotHeights, snapshotWidth, snapshotWidths]);

  React.useEffect(() => {
    if (!snapshotPlaying) return;
    if (!hasSnapshots || groupCount <= 1) {
      setSnapshotPlaying(false);
    }
  }, [groupCount, hasSnapshots, setSnapshotPlaying, snapshotPlaying]);

  React.useEffect(() => {
    if (!hasSnapshots || selectedSnapshot < 0) {
      setSnapshotHistogramBins(new Array(256).fill(0));
      setSnapshotHistogramRange([0, 1]);
      setSnapshotHistogramClipRange([0, 1]);
      setSnapshotHistogramBackend("off");
      return;
    }
    const imageHeight = snapshotHeights?.[selectedSnapshot] || snapshotHeight;
    const imageWidth = snapshotWidths?.[selectedSnapshot] || snapshotWidth;
    const image = extractPackedImage(snapshotData, selectedSnapshot, snapshotHeight, snapshotWidth, imageHeight, imageWidth);
    const dataRange = findDataRange(image);
    const dataMin = dataRange.min;
    const dataMax = dataRange.max > dataRange.min ? dataRange.max : dataRange.min + 1;
    const clipRange = resolveSnapshotDisplayRange(image, normalisedSnapshotContrastPreset);
    setSnapshotHistogramRange([dataMin, dataMax]);
    setSnapshotHistogramClipRange(clipRange);
    let canceled = false;
    const useCpu = () => {
      if (canceled) return;
      setSnapshotHistogramBins(computeHistogramFromBytes(image, 256, dataMin, dataMax));
      setSnapshotHistogramBackend("cpu");
    };
    if (!preferWebgpu) {
      useCpu();
      return () => { canceled = true; };
    }
    void getGPUColormapEngine().then(async (engine) => {
      if (!engine || canceled) {
        useCpu();
        return;
      }
      try {
        engine.uploadData(histogramSlotRef.current, image, imageWidth, imageHeight);
        const bins = await engine.computeHistogramWithRange(histogramSlotRef.current, dataMin, dataMax, false);
        if (canceled) return;
        setSnapshotHistogramBins(bins);
        setSnapshotHistogramBackend("webgpu");
      } catch {
        useCpu();
      }
    });
    return () => { canceled = true; };
  }, [
    hasSnapshots,
    preferWebgpu,
    normalisedSnapshotContrastPreset,
    selectedSnapshot,
    snapshotData,
    snapshotHeight,
    snapshotHeights,
    snapshotWidth,
    snapshotWidths,
  ]);

  React.useEffect(() => {
    if (!snapshotPlaying || !hasSnapshots || groupCount <= 1) return;
    let raf = 0;
    let previous = 0;
    const intervalMs = 1000 / clampSnapshotFps(snapshotFps);
    const tick = (now: number) => {
      if (!previous) previous = now;
      if (now - previous >= intervalMs) {
        selectSnapshotGroup((selectedGroup + 1) % groupCount);
        previous = now;
      }
      raf = window.requestAnimationFrame(tick);
    };
    raf = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(raf);
  }, [groupCount, hasSnapshots, selectSnapshotGroup, selectedGroup, snapshotFps, snapshotPlaying]);

  React.useEffect(() => {
    if (!exportStatus) return;
    const preparing = exportStatus.startsWith("Preparing ") || exportStatus.startsWith("Exporting ");
    if (preparing) setExportBusy(true);
    else if (!pendingExportRef.current) setExportBusy(false);
  }, [exportStatus]);

  React.useEffect(() => {
    if (!localExportStatus || exportBusy) return;
    if (localExportStatus.startsWith("Preparing ") || localExportStatus.startsWith("Saving ")) return;
    const id = window.setTimeout(() => {
      setLocalExportStatus((current) => current === localExportStatus ? "" : current);
    }, 5000);
    return () => window.clearTimeout(id);
  }, [localExportStatus, exportBusy]);

  React.useEffect(() => {
    const pending = pendingExportRef.current;
    if (!pending || exportPayloadId !== pending.id) return;
    const bytes = extractBytes(exportPayload ?? EMPTY_BYTES);
    if (bytes.length === 0) return;
    let canceled = false;
    const save = async () => {
      const payload = bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength ? bytes : bytes.slice();
      const filename = exportPayloadFilename || pending.filename;
      const blob = new Blob([payload as BlobPart], { type: "text/html;charset=utf-8" });
      try {
        if (pending.handle) {
          setLocalExportStatus(`Saving ${filename}...`);
          const writable = await pending.handle.createWritable();
          await writable.write(blob);
          await writable.close();
        } else {
          downloadBlob(blob, filename);
        }
        if (canceled) return;
        pendingExportRef.current = null;
        setExportBusy(false);
        setLocalExportStatus(`Saved ${filename} (${formatSavedBytes(bytes.byteLength)})`);
        setExportRequest(JSON.stringify({ mode: "clear", id: `${pending.id}-clear` }));
      } catch (err) {
        if (canceled) return;
        pendingExportRef.current = null;
        setExportBusy(false);
        setLocalExportStatus(`Export failed: ${err instanceof Error ? err.message : String(err)}`);
        setExportRequest(JSON.stringify({ mode: "clear", id: `${pending.id}-clear` }));
      }
    };
    void save();
    return () => { canceled = true; };
  }, [exportPayload, exportPayloadId, exportPayloadFilename, setExportRequest]);

  React.useEffect(() => {
    const canvas = profileCanvasRef.current;
    if (!canvas || !hasProfileImage) return;
    drawFloatImage(canvas, profileImageData, profileImageHeight, profileImageWidth, profileLine ?? [], imageLut, {
      bg: themeColors.bgAlt,
      border: themeColors.border,
      accent: themeColors.accent,
      text: themeColors.text,
    });
  }, [hasProfileImage, imageLut, profileImageData, profileImageHeight, profileImageWidth, profileLine, themeColors]);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(plotSize.width * dpr);
    canvas.height = Math.round(plotSize.height * dpr);
    canvas.style.width = `${plotSize.width}px`;
    canvas.style.height = `${plotSize.height}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = themeColors.bg;
    ctx.fillRect(0, 0, plotSize.width, plotSize.height);

    const geom: PlotGeometry = {
      left: 64,
      right: 18,
      top: plotTitleVisible ? 28 : 16,
      bottom: 46,
      width: plotSize.width,
      height: plotSize.height,
      plotW: Math.max(1, plotSize.width - 82),
      plotH: Math.max(1, plotSize.height - (plotTitleVisible ? 74 : 62)),
      xMin: effectiveXRange[0],
      xMax: effectiveXRange[1],
      yMin: effectiveYRange[0],
      yMax: effectiveYRange[1],
      logScale,
    };
    geomRef.current = geom;

    ctx.strokeStyle = themeColors.border;
    ctx.lineWidth = 1;
    ctx.strokeRect(geom.left, geom.top, geom.plotW, geom.plotH);
    ctx.save();
    ctx.rect(geom.left, geom.top, geom.plotW, geom.plotH);
    ctx.clip();

    const xTicks = niceTicks(geom.xMin, geom.xMax, methodLabels?.length ? Math.min(methodLabels.length, 6) : 6);
    const yTickValues = logScale
      ? niceTicks(log10(geom.yMin), log10(geom.yMax), 5).map((value) => Math.pow(10, value))
      : niceTicks(geom.yMin, geom.yMax, 5);

    if (showGrid) {
      ctx.strokeStyle = themeColors.border;
      ctx.globalAlpha = 0.45;
      ctx.setLineDash([3, 4]);
      for (const tick of xTicks) {
        const x = dataToX(tick, geom);
        ctx.beginPath();
        ctx.moveTo(x, geom.top);
        ctx.lineTo(x, geom.top + geom.plotH);
        ctx.stroke();
      }
      for (const tick of yTickValues) {
        if (logScale && tick <= 0) continue;
        const y = dataToY(tick, geom);
        ctx.beginPath();
        ctx.moveTo(geom.left, y);
        ctx.lineTo(geom.left + geom.plotW, y);
        ctx.stroke();
      }
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }

    for (const marker of markers ?? []) {
      if (!Number.isFinite(marker.x)) continue;
      const x = dataToX(Number(marker.x), geom);
      if (x < geom.left || x > geom.left + geom.plotW) continue;
      ctx.strokeStyle = marker.kind === "checkpoint" ? "#f59e0b" : themeColors.textMuted;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(x, geom.top);
      ctx.lineTo(x, geom.top + geom.plotH);
      ctx.stroke();
      ctx.setLineDash([]);
      if (marker.label) {
        ctx.fillStyle = themeColors.textMuted;
        ctx.font = "10px system-ui, sans-serif";
        ctx.fillText(marker.label, x + 4, geom.top + 12);
      }
    }

    if (selectedSnapshotIteration !== null) {
      const x = dataToX(selectedSnapshotIteration, geom);
      if (x >= geom.left && x <= geom.left + geom.plotW) {
        ctx.strokeStyle = themeColors.accent;
        ctx.globalAlpha = 0.75;
        ctx.beginPath();
        ctx.moveTo(x, geom.top);
        ctx.lineTo(x, geom.top + geom.plotH);
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }

    for (let trace = 0; trace < nTraces; trace += 1) {
      if (hiddenTraceSet.has(trace)) continue;
      const isFocused = focusedTrace < 0 || focusedTrace === trace;
      ctx.strokeStyle = cssColor(colors ?? [], trace);
      ctx.lineWidth = Math.max(1, lineWidth) * (focusedTrace === trace ? 1.8 : 1);
      ctx.globalAlpha = isFocused ? 1 : 0.22;
      ctx.beginPath();
      let active = false;
      const offset = trace * nPoints;
      for (let point = 0; point < nPoints; point += 1) {
        const xValue = xData.length > point ? xData[point] : point;
        const yValue = yData[offset + point];
        if (!Number.isFinite(xValue) || !Number.isFinite(yValue) || (logScale && yValue <= 0)) {
          active = false;
          continue;
        }
        if (xValue < geom.xMin || xValue > geom.xMax) continue;
        const px = dataToX(xValue, geom);
        const py = dataToY(yValue, geom);
        if (!active) {
          ctx.moveTo(px, py);
          active = true;
        } else {
          ctx.lineTo(px, py);
        }
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    if (showSnapshotThumbnails && hasSnapshots && groupCount > 0) {
      const thumbW = thumbnailSize;
      const lanes = Math.max(1, Math.min(3, Math.floor((geom.plotH - 12) / (thumbnailSize + 6)) || 1));
      for (let groupIdx = 0; groupIdx < groupCount; groupIdx += 1) {
        const imageIndices = (snapshotGroups[groupIdx] ?? []).filter((idx) => !isSnapshotImageHidden(idx)).sort((a, b) => {
          const aLabel = imageLabelForIndex(a);
          const bLabel = imageLabelForIndex(b);
          if (isReferenceLabel(aLabel) !== isReferenceLabel(bLabel)) return isReferenceLabel(aLabel) ? -1 : 1;
          const ar = Number(trialRowByKey.get(trialKey(aLabel))?.rank ?? Number.MAX_SAFE_INTEGER);
          const br = Number(trialRowByKey.get(trialKey(bLabel))?.rank ?? Number.MAX_SAFE_INTEGER);
          return ar - br || aLabel.localeCompare(bLabel);
        });
        const imageIdx = imageIndices[0];
        if (imageIdx === undefined) continue;
        const thumbMetrics = snapshotGroupThumbnailMetrics(imageIndices.length, thumbW);
        if (thumbMetrics.width <= 0 || thumbMetrics.height <= 0) continue;
        const iteration = Number.isFinite(snapshotGroupIterations?.[groupIdx])
          ? Number(snapshotGroupIterations[groupIdx])
          : snapshotIterations?.[imageIdx];
        if (!Number.isFinite(iteration)) continue;
        const xCenter = dataToX(Number(iteration), geom);
        if (xCenter < geom.left - thumbW || xCenter > geom.left + geom.plotW + thumbW) continue;
        const images = imageIndices.map((idx) => {
          const imageHeight = snapshotHeights?.[idx] || snapshotHeight;
          const imageWidth = snapshotWidths?.[idx] || snapshotWidth;
          return {
            data: extractPackedImage(snapshotData, idx, snapshotHeight, snapshotWidth, imageHeight, imageWidth),
            height: imageHeight,
            width: imageWidth,
          };
        });
        const lane = lanes > 1 ? groupIdx % lanes : 0;
        const x0 = clampValue(xCenter - thumbMetrics.width / 2, geom.left + 2, geom.left + geom.plotW - thumbMetrics.width - 2);
        const y0 = Math.min(geom.top + geom.plotH - thumbMetrics.height - 4, geom.top + 8 + lane * (thumbnailSize + 4));
        ctx.strokeStyle = groupIdx === selectedGroup ? themeColors.accent : themeColors.border;
        ctx.globalAlpha = groupIdx === selectedGroup ? 0.55 : 0.28;
        ctx.beginPath();
        ctx.moveTo(xCenter, geom.top);
        ctx.lineTo(xCenter, y0 + thumbMetrics.height);
        ctx.stroke();
        ctx.globalAlpha = 1;
        drawSnapshotGroupThumbnail(
          ctx,
          images,
          x0,
          y0,
          thumbW,
          groupIdx === selectedGroup,
          imageLut,
          themeColors,
          normalisedSnapshotContrastPreset,
        );
      }
    }

    if (hover) {
      ctx.strokeStyle = themeColors.textMuted;
      ctx.globalAlpha = 0.45;
      ctx.setLineDash([2, 4]);
      ctx.beginPath();
      ctx.moveTo(hover.px, geom.top);
      ctx.lineTo(hover.px, geom.top + geom.plotH);
      ctx.moveTo(geom.left, hover.py);
      ctx.lineTo(geom.left + geom.plotW, hover.py);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
      ctx.fillStyle = cssColor(colors ?? [], hover.trace);
      ctx.beginPath();
      ctx.arc(hover.px, hover.py, 3.5, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.restore();
    ctx.fillStyle = themeColors.text;
    ctx.font = "12px system-ui, sans-serif";
    ctx.textAlign = "center";
    for (const tick of xTicks) {
      const x = dataToX(tick, geom);
      let label = formatAxisValue(tick);
      if (methodLabels?.length) {
        const idx = Math.round(tick);
        if (Math.abs(idx - tick) < 0.05 && idx >= 0 && idx < methodLabels.length) label = shortMethodLabel(methodLabels[idx]);
      }
      ctx.fillText(label, x, geom.top + geom.plotH + 18);
    }
    ctx.textAlign = "right";
    for (const tick of yTickValues) {
      if (logScale && tick <= 0) continue;
      const y = dataToY(tick, geom);
      ctx.fillText(formatRangeValue(tick), geom.left - 8, y + 4);
    }
    ctx.textAlign = "center";
    const xlabel = xLabel ? `${xLabel}${xUnit ? ` (${xUnit})` : ""}` : xUnit;
    const ylabel = yLabel ? `${yLabel}${yUnit ? ` (${yUnit})` : ""}` : yUnit;
    if (xlabel) ctx.fillText(xlabel, geom.left + geom.plotW / 2, plotSize.height - 8);
    if (ylabel) {
      ctx.save();
      ctx.translate(14, geom.top + geom.plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(ylabel, 0, 0);
      ctx.restore();
    }
    if (plotTitleVisible) {
      ctx.textAlign = "left";
      ctx.font = "600 13px system-ui, sans-serif";
      ctx.fillText(title, geom.left, 18);
    }
    if (nPoints === 0 || nTraces === 0) {
      ctx.fillStyle = themeColors.textMuted;
      ctx.textAlign = "center";
      ctx.font = "13px system-ui, sans-serif";
      ctx.fillText("No data", geom.left + geom.plotW / 2, geom.top + geom.plotH / 2);
    }
  }, [
    plotSize,
    themeColors,
    title,
    plotTitleVisible,
    yData,
    xData,
    nTraces,
    nPoints,
    labels,
    colors,
    methodLabels,
    xLabel,
    yLabel,
    xUnit,
    yUnit,
    logScale,
    showGrid,
    lineWidth,
    focusedTrace,
    hiddenTraceSet,
    effectiveXRange,
    effectiveYRange,
    markers,
    hover,
    selectedSnapshotIteration,
    showSnapshotThumbnails,
    hasSnapshots,
    groupCount,
    imageLabelForIndex,
    isSnapshotImageHidden,
    snapshotGroups,
    snapshotGroupIterations,
    snapshotIterations,
    snapshotHeights,
    snapshotWidths,
    snapshotData,
    snapshotHeight,
    snapshotWidth,
    selectedGroup,
    trialRowByKey,
    thumbnailSize,
    imageLut,
    normalisedSnapshotContrastPreset,
  ]);

  const nearestPoint = React.useCallback((clientX: number, clientY: number): HoverPoint | null => {
    const canvas = canvasRef.current;
    const geom = geomRef.current;
    if (!canvas || !geom || nPoints <= 0 || nTraces <= 0) return null;
    const rect = canvas.getBoundingClientRect();
    const px = clientX - rect.left;
    const py = clientY - rect.top;
    if (px < geom.left || px > geom.left + geom.plotW || py < geom.top || py > geom.top + geom.plotH) return null;
    let best: HoverPoint | null = null;
    let bestDist = 24 * 24;
    for (let trace = 0; trace < nTraces; trace += 1) {
      if (hiddenTraceSet.has(trace)) continue;
      const offset = trace * nPoints;
      for (let point = 0; point < nPoints; point += 1) {
        const x = xData.length > point ? xData[point] : point;
        const y = yData[offset + point];
        if (!Number.isFinite(x) || !Number.isFinite(y) || (logScale && y <= 0)) continue;
        if (x < geom.xMin || x > geom.xMax || y < geom.yMin || y > geom.yMax) continue;
        const dx = dataToX(x, geom) - px;
        const dy = dataToY(y, geom) - py;
        const dist = dx * dx + dy * dy;
        if (dist < bestDist) {
          bestDist = dist;
          best = { trace, point, x, y, px: dataToX(x, geom), py: dataToY(y, geom) };
        }
      }
    }
    return best;
  }, [hiddenTraceSet, nPoints, nTraces, xData, yData, logScale]);

  const handlePointerMove = React.useCallback((event: React.PointerEvent<HTMLCanvasElement>) => {
    scheduleHover(nearestPoint(event.clientX, event.clientY));
  }, [nearestPoint, scheduleHover]);

  const handlePointerLeave = React.useCallback(() => scheduleHover(null), [scheduleHover]);

  const handleClick = React.useCallback(() => {
    if (hover) setFocusedTrace(hover.trace);
  }, [hover, setFocusedTrace]);

  const resetRanges = React.useCallback(() => {
    setXRange([]);
    setYRange([]);
    setFocusedTrace(-1);
    setSnapshotImageView({ zoom: 1, panX: 0, panY: 0 });
    setSnapshotFftView({ zoom: 1, panX: 0, panY: 0 });
  }, [setFocusedTrace, setXRange, setYRange]);

  const handleWheel = React.useCallback((event: React.WheelEvent<HTMLCanvasElement>) => {
    const geom = geomRef.current;
    if (!geom) return;
    event.preventDefault();
    const rect = event.currentTarget.getBoundingClientRect();
    const px = event.clientX - rect.left;
    const py = event.clientY - rect.top;
    const factor = Math.exp(event.deltaY * 0.001);
    if (event.shiftKey) {
      const center = pixelToY(py, geom);
      const lo = center + (geom.yMin - center) * factor;
      const hi = center + (geom.yMax - center) * factor;
      if (!geom.logScale || lo > 0) setYRange([lo, hi]);
    } else {
      const center = pixelToX(px, geom);
      const lo = center + (geom.xMin - center) * factor;
      const hi = center + (geom.xMax - center) * factor;
      setXRange(clampRange([lo, hi], fullXRange));
    }
  }, [fullXRange, setXRange, setYRange]);

  const handleExportSelect = React.useCallback(async (kind: "html" | "csv" | "png") => {
    setExportAnchor(null);
    if (kind === "csv") {
      const filename = makeExportFilename(title, nTraces, nPoints, "csv");
      const blob = new Blob([csvForTraces(xData, yData, labels ?? [], nTraces, nPoints)], { type: "text/csv;charset=utf-8" });
      downloadBlob(blob, filename);
      setLocalExportStatus(`Saved ${filename} (${formatSavedBytes(blob.size)})`);
      return;
    }
    if (kind === "png") {
      const filename = makeExportFilename(title, nTraces, nPoints, "png");
      canvasRef.current?.toBlob((blob) => {
        if (!blob) {
          setLocalExportStatus("Export failed: canvas unavailable");
          return;
        }
        downloadBlob(blob, filename);
        setLocalExportStatus(`Saved ${filename} (${formatSavedBytes(blob.size)})`);
      }, "image/png");
      return;
    }
    if (!exportEnabled) {
      setLocalExportStatus("HTML export requires a live Python kernel");
      return;
    }
    const filename = makeExportFilename(title, nTraces, nPoints, "html");
    const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    setExportBusy(true);
    setLocalExportStatus("Choose export location...");
    const picker = (window as Show1DWindow).showSaveFilePicker;
    let handle: Show1DFileHandle | null = null;
    if (picker) {
      try {
        handle = await picker({
          suggestedName: filename,
          types: [{ description: "Standalone HTML", accept: { "text/html": [".html"] } }],
        });
      } catch (err) {
        if (isAbortLikeError(err)) {
          setExportBusy(false);
          setLocalExportStatus("Export canceled");
          return;
        }
        setExportBusy(false);
        setLocalExportStatus(`Export failed: ${err instanceof Error ? err.message : String(err)}`);
        return;
      }
    }
    pendingExportRef.current = { id, filename, handle };
    setLocalExportStatus(`Preparing ${filename}...`);
    setExportRequest(JSON.stringify({ mode: "single", encoding: "full", id, filename, download: true }));
  }, [exportEnabled, labels, nPoints, nTraces, setExportRequest, title, xData, yData]);

  const handleHandoffToShow2D = React.useCallback(() => {
    const images = selectedGroupImageIndices.map((idx) => imageLabelForIndex(idx));
    setViewMenuAnchor(null);
    setHandoffRequest(JSON.stringify({
      mode: "show2d",
      id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
      group: selectedGroup,
      images,
      respect_review_filters: false,
    }));
  }, [imageLabelForIndex, selectedGroup, selectedGroupImageIndices, setHandoffRequest]);

  const handleClosePreparedView = React.useCallback(() => {
    setHandoffRequest(JSON.stringify({
      mode: "clear",
      id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
    }));
  }, [setHandoffRequest]);

  const selectedSnapshotLabel = selectedGroup >= 0
    ? snapshotGroupLabels?.[selectedGroup] || snapshotLabels?.[selectedSnapshot] || `snapshot ${selectedGroup + 1}`
    : "";
  const selectedSnapshotImageLabel = selectedSnapshot >= 0 ? imageLabelForIndex(selectedSnapshot) : "";
  const selectedTrialNote = selectedSnapshotImageLabel ? lookupByTrialKey(trialNotes, selectedSnapshotImageLabel) ?? "" : "";
  const selectedTrialTags = selectedSnapshotImageLabel ? lookupByTrialKey(trialTags, selectedSnapshotImageLabel) ?? [] : [];
  const selectedSnapshotPosition = selectedSnapshotIteration !== null && !labelAlreadyContainsValue(selectedSnapshotLabel, selectedSnapshotIteration)
    ? axisPositionText(selectedSnapshotIteration, xLabel, xUnit)
    : "";
  const snapshotSliderMarks = React.useMemo(
    () => groupCount > 1 && groupCount <= 24 ? Array.from({ length: groupCount }, (_, value) => ({ value })) : [],
    [groupCount],
  );
  const selectedImageColumns = Math.max(1, Math.min(snapshotColumnCount, Math.max(1, selectedGroupImageIndices.length)));

  return (
    <Box
      ref={rootRef}
      sx={{
        width: "100%",
        maxWidth: 1180,
        bgcolor: themeColors.bg,
        color: themeColors.text,
        border: `1px solid ${themeColors.border}`,
        borderRadius: 1,
        overflow: "hidden",
        fontFamily: "system-ui, sans-serif",
      }}
    >
      {controlsVisible && (
        <Box sx={{ px: 1.25, py: 0.75, borderBottom: `1px solid ${themeColors.border}`, bgcolor: themeColors.bg }}>
          <Stack direction="row" alignItems="center" spacing={0.75} sx={{ minWidth: 0, mb: 0.75 }}>
            <ShowChartIcon sx={{ fontSize: 18, color: themeColors.accent, flexShrink: 0 }} />
            {showTitle && (
              <Typography sx={{ fontWeight: 600, fontSize: 13, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {title || "Show1D"}
              </Typography>
            )}
          </Stack>
          <Stack direction="row" alignItems="center" spacing={1} sx={{ flexWrap: "wrap", rowGap: 0.75 }}>
            <Button
              size="small"
              variant="outlined"
              startIcon={<RestartAltIcon fontSize="small" />}
              onClick={resetRanges}
              sx={{ color: themeColors.text, borderColor: themeColors.border, textTransform: "none", height: 30, flexShrink: 0 }}
            >
              Reset
            </Button>
            <Tooltip title={showGrid ? "Hide Grid" : "Show Grid"}>
              <IconButton size="small" onClick={() => setShowGrid(!showGrid)} sx={{ color: themeColors.text }}>
                {showGrid ? <GridOnIcon fontSize="small" /> : <GridOffIcon fontSize="small" />}
              </IconButton>
            </Tooltip>
            <Stack direction="row" alignItems="center" spacing={0.5}>
              <Typography sx={{ fontSize: 12, color: themeColors.textMuted }}>Log</Typography>
              <Switch size="small" checked={Boolean(logScale)} onChange={(_, checked) => setLogScale(checked)} sx={switchStyles.small} />
            </Stack>
            <Stack direction="row" alignItems="center" spacing={0.5}>
              <Typography sx={{ fontSize: 12, color: themeColors.textMuted }}>Stats</Typography>
              <Switch size="small" checked={Boolean(showStats)} onChange={(_, checked) => setShowStats(checked)} sx={switchStyles.small} />
            </Stack>
            <Stack direction="row" alignItems="center" spacing={0.5}>
              <Typography sx={{ fontSize: 12, color: themeColors.textMuted }}>Legend</Typography>
              <Switch size="small" checked={Boolean(showLegend)} onChange={(_, checked) => setShowLegend(checked)} sx={switchStyles.small} />
            </Stack>
            {hasSnapshots && (
            <Stack direction="row" alignItems="center" spacing={0.5}>
              <Typography sx={{ fontSize: 12, color: themeColors.textMuted }}>Snapshots</Typography>
              <Switch size="small" checked={Boolean(showSnapshots)} onChange={(_, checked) => setShowSnapshots(checked)} sx={switchStyles.small} />
            </Stack>
          )}
            {hasSnapshots && (
            <Stack direction="row" alignItems="center" spacing={0.5}>
              <Typography sx={{ fontSize: 12, color: themeColors.textMuted }}>Thumbs</Typography>
              <Switch size="small" checked={Boolean(showSnapshotThumbnails)} onChange={(_, checked) => setShowSnapshotThumbnails(checked)} sx={switchStyles.small} />
            </Stack>
            )}
            {hasSnapshots && (
              <Stack direction="row" alignItems="center" spacing={0.5} sx={{ flexShrink: 0 }}>
                <Typography sx={{ fontSize: 12, color: themeColors.textMuted }}>cols</Typography>
                <Select
                  size="small"
                  value={snapshotColumnCount}
                  onChange={(event) => setSnapshotColumns(Number(event.target.value))}
                  sx={{ ...themedSelect, minWidth: 50, height: 30, fontSize: 11 }}
                  MenuProps={themedMenuProps}
                  inputProps={{ "aria-label": "Snapshot image columns" }}
                >
                  {[1, 2, 3, 4].map((value) => (
                    <MenuItem key={value} value={value}>{value}</MenuItem>
                  ))}
                </Select>
              </Stack>
            )}
            {nTraces > 1 && (
              <Stack direction="row" alignItems="center" spacing={0.5} sx={{ flexShrink: 0 }}>
                <Typography sx={{ fontSize: 12, color: themeColors.textMuted }}>Starred</Typography>
                <Switch
                  size="small"
                  checked={Boolean(showStarredOnly)}
                  onChange={(_, checked) => setShowStarredOnly(checked)}
                  sx={switchStyles.small}
                  slotProps={{ input: { "aria-label": "Show starred trials only" } }}
                />
              </Stack>
            )}
            {nTraces > 1 && (
              <Stack direction="row" alignItems="center" spacing={0.5} sx={{ flexShrink: 0 }}>
                <Typography sx={{ fontSize: 12, color: themeColors.textMuted }}>rank</Typography>
                <Select
                  size="small"
                  value={normalisedSortKey}
                  onChange={(event) => setTrialSortKey(String(event.target.value))}
                  sx={{ ...themedSelect, minWidth: 92, height: 30, fontSize: 11 }}
                  MenuProps={themedMenuProps}
                  inputProps={{ "aria-label": "Trial ranking objective" }}
                >
                  <MenuItem value="final_loss">final loss</MenuItem>
                  <MenuItem value="min_loss">min loss</MenuItem>
                  <MenuItem value="rmse">RMSE</MenuItem>
                  <MenuItem value="flicker">flicker</MenuItem>
                  <MenuItem value="lambda">lambda</MenuItem>
                  <MenuItem value="object_quality">object</MenuItem>
                  <MenuItem value="probe_quality">probe</MenuItem>
                  <MenuItem value="alert_count">alerts</MenuItem>
                  <MenuItem value="label">label</MenuItem>
                </Select>
                <Tooltip title={trialSortDescending ? "Descending" : "Ascending"}>
                  <Switch
                    size="small"
                    checked={Boolean(trialSortDescending)}
                    onChange={(_, checked) => setTrialSortDescending(checked)}
                    sx={switchStyles.small}
                    slotProps={{ input: { "aria-label": "Reverse trial ranking order" } }}
                  />
                </Tooltip>
              </Stack>
            )}
            {nTraces > 1 && (
              <Stack direction="row" alignItems="center" spacing={0.5} sx={{ flexShrink: 0 }}>
                <Typography sx={{ fontSize: 12, color: themeColors.textMuted }}>top</Typography>
                <Select
                  size="small"
                  value={topTrialLimit}
                  onChange={(event) => setTopTrialCount(Number(event.target.value))}
                  sx={{ ...themedSelect, minWidth: 54, height: 30, fontSize: 11 }}
                  MenuProps={themedMenuProps}
                  inputProps={{ "aria-label": "Top trial count" }}
                >
                  <MenuItem value={0}>all</MenuItem>
                  {[1, 2, 3, 5, 10].map((value) => (
                    <MenuItem key={value} value={value}>{value}</MenuItem>
                  ))}
                </Select>
              </Stack>
            )}
            {nTraces > 1 && (
              <TextField
                size="small"
                value={trialFilterText || ""}
                onChange={(event) => setTrialFilterText(event.target.value)}
                placeholder="filter"
                inputProps={{ "aria-label": "Filter trials" }}
                InputProps={{
                  endAdornment: trialFilterText ? (
                    <InputAdornment position="end">
                      <IconButton
                        size="small"
                        onClick={() => setTrialFilterText("")}
                        aria-label="Clear trial filter"
                        sx={{ color: themeColors.textMuted, p: 0.25 }}
                      >
                        <ClearIcon fontSize="inherit" />
                      </IconButton>
                    </InputAdornment>
                  ) : null,
                }}
                sx={{
                  width: 104,
                  flexShrink: 0,
                  "& .MuiInputBase-root": { height: 30, fontSize: 11, bgcolor: themeColors.controlBg, color: themeColors.text },
                  "& .MuiInputAdornment-root": { ml: 0.25 },
                  "& .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.border },
                  "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.accent },
                }}
              />
            )}
            {nTraces > 1 && (
              <Button
                size="small"
                variant="outlined"
                onClick={starBestTrial}
                sx={{ color: themeColors.text, borderColor: themeColors.border, textTransform: "none", height: 30, px: 1, flexShrink: 0 }}
                aria-label="Star best ranked trial"
              >
                Star best
              </Button>
            )}
            {nTraces > 1 && (
              <Button
                size="small"
                variant="outlined"
                onClick={hideWorstTrial}
                sx={{ color: themeColors.text, borderColor: themeColors.border, textTransform: "none", height: 30, px: 1, flexShrink: 0 }}
                aria-label="Hide worst ranked trial"
              >
                Hide worst
              </Button>
            )}
            {hiddenTrialCount > 0 && (
              <Button
                size="small"
                variant="outlined"
                onClick={showAllTrials}
                sx={{ color: themeColors.text, borderColor: themeColors.border, textTransform: "none", height: 30, px: 1, flexShrink: 0 }}
                aria-label={`Show all hidden trials (${hiddenTrialCount})`}
              >
                Show all ({hiddenTrialCount})
              </Button>
            )}
            {nTraces > 1 && (
            <Select
              size="small"
              value={String(focusedTrace)}
              onChange={(event) => setFocusedTrace(Number(event.target.value))}
              sx={{ ...themedSelect, minWidth: 120, height: 30, fontSize: 12 }}
              MenuProps={themedMenuProps}
            >
              <MenuItem value="-1">{hiddenTraceIndices.length ? "All Visible" : "All Traces"}</MenuItem>
              {visibleTraceIndices.map((idx) => (
                <MenuItem key={idx} value={String(idx)}>{labels?.[idx] || `Trace ${idx + 1}`}</MenuItem>
              ))}
            </Select>
            )}
            <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "nowrap", flexShrink: 0, minWidth: sidePanelVisible ? 284 : 136 }}>
            <Stack direction="row" alignItems="center" spacing={0.75} sx={{ flexShrink: 0 }}>
              <Typography sx={{ fontSize: 12, color: themeColors.textMuted, whiteSpace: "nowrap" }}>plot height</Typography>
              <Slider
                size="small"
                value={plotHeight}
                min={220}
                max={720}
                step={10}
                onChange={(_, value) => setPlotHeightPx(Array.isArray(value) ? value[0] : value)}
                sx={{ ...sliderStyles.small, width: 64, color: themeColors.accent }}
                aria-label="Plot height"
                valueLabelDisplay="auto"
              />
            </Stack>
            {sidePanelVisible && (
              <Stack direction="row" alignItems="center" spacing={0.75} sx={{ flexShrink: 0 }}>
                <Typography sx={{ fontSize: 12, color: themeColors.textMuted, whiteSpace: "nowrap" }}>panel width</Typography>
                <Slider
                  size="small"
                  value={sidePanelWidth}
                  min={300}
                  max={640}
                  step={10}
                  onChange={(_, value) => setSidePanelWidthPx(Array.isArray(value) ? value[0] : value)}
                  sx={{ ...sliderStyles.small, width: 64, color: themeColors.accent }}
                  aria-label="Side panel width"
                  valueLabelDisplay="auto"
                />
              </Stack>
            )}
            </Box>
            <Box sx={{ flex: 1 }} />
            {handoffEnabled && hasSnapshots && (
              <>
                <Button
                  size="small"
                  variant="outlined"
                  onClick={(event) => setViewMenuAnchor(event.currentTarget)}
                  sx={{ color: themeColors.text, borderColor: themeColors.border, textTransform: "none", height: 30 }}
                  aria-label="Open view options"
                  aria-controls={viewMenuAnchor ? "show1d-view-menu" : undefined}
                  aria-expanded={viewMenuAnchor ? "true" : undefined}
                  aria-haspopup="menu"
                  title={handoffStatus || "View options"}
                >
                  View
                </Button>
                <Menu
                  id="show1d-view-menu"
                  anchorEl={viewMenuAnchor}
                  open={Boolean(viewMenuAnchor)}
                  onClose={() => setViewMenuAnchor(null)}
                  MenuListProps={{ "aria-label": "View options" }}
                  {...themedMenuProps}
                >
                  <MenuItem
                    onClick={handleHandoffToShow2D}
                    disabled={selectedGroupImageIndices.length === 0}
                    sx={{ fontSize: 12 }}
                  >
                    View selected as 2D
                  </MenuItem>
                </Menu>
                {handoffStatus && (
                  <Typography
                    sx={{
                      fontSize: 10.5,
                      color: handoffStatus.startsWith("View failed") ? "#b91c1c" : themeColors.textMuted,
                      maxWidth: 130,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                    title={handoffStatus}
                  >
                    {handoffStatus}
                  </Typography>
                )}
              </>
            )}
            <Button
            size="small"
            variant="outlined"
            startIcon={<DownloadIcon />}
            onClick={(event) => setExportAnchor(event.currentTarget)}
            disabled={exportBusy}
            sx={{ color: themeColors.text, borderColor: themeColors.border, textTransform: "none", height: 30 }}
          >
            Export
            </Button>
            <Menu anchorEl={exportAnchor} open={Boolean(exportAnchor)} onClose={() => setExportAnchor(null)}>
            <MenuItem onClick={() => void handleExportSelect("html")} disabled={!exportEnabled}>
              <DownloadIcon fontSize="small" sx={{ mr: 1 }} /> HTML full float32 ({htmlSize})
            </MenuItem>
            <MenuItem onClick={() => void handleExportSelect("csv")}>
              <TableChartIcon fontSize="small" sx={{ mr: 1 }} /> CSV traces
            </MenuItem>
            <MenuItem onClick={() => void handleExportSelect("png")}>
              <ImageIcon fontSize="small" sx={{ mr: 1 }} /> PNG view
            </MenuItem>
            </Menu>
          </Stack>
        </Box>
      )}
      <Box sx={{ display: "grid", gridTemplateColumns: sidePanelVisible ? { xs: "1fr", md: `minmax(0, 1fr) ${sidePanelWidth}px` } : "1fr", alignItems: "start", minHeight: 360 }}>
        <Box sx={{ minWidth: 0, p: 1 }}>
          <Box ref={plotHostRef} sx={{ position: "relative", height: { xs: Math.max(280, Math.min(plotHeight, 520)), md: plotHeight }, minWidth: 0 }}>
            <canvas
              ref={canvasRef}
              onPointerMove={handlePointerMove}
              onPointerLeave={handlePointerLeave}
              onClick={handleClick}
              onDoubleClick={resetRanges}
              onWheel={handleWheel}
              style={{ display: "block", width: "100%", height: "100%", cursor: "crosshair", touchAction: "none" }}
            />
            {hover && (
              <Box
                sx={{
                  position: "absolute",
                  left: Math.min(Math.max(hover.px + 12, 8), Math.max(8, plotSize.width - 190)),
                  top: Math.min(Math.max(hover.py - 12, 8), Math.max(8, plotSize.height - 70)),
                  bgcolor: themeColors.bgAlt,
                  color: themeColors.text,
                  border: `1px solid ${themeColors.border}`,
                  borderRadius: 1,
                  px: 0.75,
                  py: 0.5,
                  fontSize: 11,
                  pointerEvents: "none",
                  boxShadow: "0 4px 14px rgba(0,0,0,0.18)",
                  maxWidth: 180,
                }}
              >
                <Typography sx={{ fontSize: 11, fontWeight: 600, color: cssColor(colors ?? [], hover.trace), lineHeight: 1.25 }}>
                  {labels?.[hover.trace] || `Trace ${hover.trace + 1}`}
                </Typography>
                <Typography sx={{ fontSize: 11, lineHeight: 1.25 }}>
                  {methodLabels?.[hover.point] ? shortMethodLabel(methodLabels[hover.point]) : axisPositionText(hover.x, xLabel, xUnit)}
                </Typography>
                <Typography sx={{ fontSize: 11, lineHeight: 1.25 }}>y {formatNumber(hover.y, 4)}</Typography>
              </Box>
            )}
          </Box>
          {showLegend && visibleTraceIndices.length > 0 && (
            <Stack direction="row" spacing={1} sx={{ px: 1, py: 0.5, flexWrap: "wrap", rowGap: 0.5 }}>
              {visibleTraceIndices.map((idx) => (
                <Stack key={idx} direction="row" alignItems="center" spacing={0.5} sx={{ opacity: focusedTrace < 0 || focusedTrace === idx ? 1 : 0.45 }}>
                  <Box sx={{ width: 16, height: 3, bgcolor: cssColor(colors ?? [], idx), borderRadius: 1 }} />
                  <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>{labels?.[idx] || `Trace ${idx + 1}`}</Typography>
                </Stack>
              ))}
            </Stack>
          )}
          {(localExportStatus || exportStatus) && (
            <Typography sx={{ px: 1, pb: 0.75, fontSize: 11, color: themeColors.textMuted }}>
              {localExportStatus || exportStatus}
            </Typography>
          )}
        </Box>
        {sidePanelVisible && (
          <Stack spacing={1} sx={{ borderLeft: { xs: "none", md: `1px solid ${themeColors.border}` }, borderTop: { xs: `1px solid ${themeColors.border}`, md: "none" }, p: 1, minWidth: 0 }}>
            {hasProfileImage && (
              <Box>
                <Typography sx={{ fontSize: 12, fontWeight: 600, mb: 0.5 }}>Profile Image</Typography>
                <Box sx={{ height: 170, bgcolor: themeColors.bgAlt, border: `1px solid ${themeColors.border}`, borderRadius: 1, overflow: "hidden" }}>
                  <canvas ref={profileCanvasRef} style={{ width: "100%", height: "100%", display: "block" }} />
                </Box>
              </Box>
            )}
            {showSnapshots && hasSnapshots && (
              <Box>
                <Box
                  sx={{
                    display: "grid",
                    gridTemplateColumns: `repeat(${selectedImageColumns}, minmax(0, 1fr))`,
                    gap: 0,
                    maxHeight: 430,
                    overflowY: "auto",
                    pr: 0,
                    border: `1px solid ${themeColors.border}`,
                    bgcolor: themeColors.border,
                  }}
                >
                  {selectedGroupImageIndices.map((imageIdx) => {
                    const imageHeight = snapshotHeights?.[imageIdx] || snapshotHeight;
                    const imageWidth = snapshotWidths?.[imageIdx] || snapshotWidth;
                    const imageLabel = imageLabelForIndex(imageIdx);
                    const imageKey = trialKey(imageLabel);
                    const starred = starredTrialKeys.has(imageKey);
                    const selected = imageIdx === selectedSnapshot;
                    return (
                      <Box
                        key={imageIdx}
                        sx={{
                          minWidth: 0,
                          position: "relative",
                          bgcolor: themeColors.bgAlt,
                          overflow: "hidden",
                          outline: selected ? `2px solid ${themeColors.accent}` : `1px solid ${themeColors.border}`,
                          outlineOffset: -1,
                        }}
                      >
                        <SnapshotImageCanvas
                          data={snapshotData}
                          imageIndex={imageIdx}
                          packedHeight={snapshotHeight}
                          packedWidth={snapshotWidth}
                          imageHeight={imageHeight}
                          imageWidth={imageWidth}
                          lut={imageLut}
                          fftLut={snapshotFftLut}
                          colors={themeColors}
                          showFft={Boolean(showSnapshotFft)}
                          fftWindow={Boolean(snapshotFftWindow)}
                          preferWebgpu={Boolean(preferWebgpu)}
                          contrastPreset={normalisedSnapshotContrastPreset}
                          selected={selected}
                          label={imageLabel}
                          scaleBarVisible={Boolean(scaleBarVisible)}
                          pixelSize={Number.isFinite(pixelSize) && pixelSize > 0 ? pixelSize : 1}
                          pixelUnit={pixelSize > 0 ? pixelUnit : "px"}
                          imageView={snapshotImageView}
                          fftView={snapshotFftView}
                          fftCacheRef={snapshotFftCacheRef}
                          fftGpuRef={snapshotFftGpuRef}
                          onImageViewChange={setSnapshotImageView}
                          onFftViewChange={setSnapshotFftView}
                          onSelect={() => selectSnapshotImage(imageIdx)}
                        />
                        <Tooltip title={starred ? `Unstar ${imageLabel}` : `Star ${imageLabel}`}>
                          <IconButton
                            size="small"
                            onClick={(event) => {
                              event.stopPropagation();
                              toggleStarredTrial(imageLabel);
                            }}
                            aria-label={starred ? `Unstar ${imageLabel} candidate` : `Star ${imageLabel} as candidate`}
                            sx={{
                              position: "absolute",
                              top: 4,
                              right: 4,
                              zIndex: 4,
                              width: 24,
                              height: 24,
                              p: 0,
                              color: starred ? "#facc15" : "rgba(255,255,255,0.92)",
                              bgcolor: "rgba(0,0,0,0.28)",
                              "&:hover": { bgcolor: "rgba(0,0,0,0.5)", color: starred ? "#fde047" : "#fff" },
                            }}
                          >
                            {starred ? <StarIcon sx={{ fontSize: 17 }} /> : <StarBorderIcon sx={{ fontSize: 17 }} />}
                          </IconButton>
                        </Tooltip>
                        <Tooltip title={`Hide ${imageLabel}`}>
                          <IconButton
                            size="small"
                            onClick={(event) => {
                              event.stopPropagation();
                              hideTrial(imageLabel);
                            }}
                            aria-label={`Hide ${imageLabel} trial`}
                            sx={{
                              position: "absolute",
                              top: 4,
                              left: 4,
                              zIndex: 4,
                              width: 24,
                              height: 24,
                              p: 0,
                              color: "rgba(255,255,255,0.9)",
                              bgcolor: "rgba(0,0,0,0.28)",
                              "&:hover": { bgcolor: "rgba(0,0,0,0.5)", color: "#fff" },
                            }}
                          >
                            <VisibilityOffIcon sx={{ fontSize: 16 }} />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    );
                  })}
                  {selectedGroupAllImageIndices.length > 0 && selectedGroupImageIndices.length === 0 && (
                    <Box
                      sx={{
                        minHeight: 170,
                        gridColumn: `span ${selectedImageColumns}`,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        flexDirection: "column",
                        gap: 1,
                        bgcolor: themeColors.bgAlt,
                        color: themeColors.textMuted,
                        border: `1px solid ${themeColors.border}`,
                      }}
                    >
                      <Typography sx={{ fontSize: 12 }}>All trials hidden</Typography>
                      <Button
                        size="small"
                        variant="outlined"
                        onClick={showAllTrials}
                        sx={{ color: themeColors.text, borderColor: themeColors.border, textTransform: "none", height: 28 }}
                      >
                        Show all
                      </Button>
                    </Box>
                  )}
                </Box>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", mt: 0.75 }}>
                  {selectedSnapshotLabel}{selectedSnapshotPosition ? ` · ${selectedSnapshotPosition}` : ""}
                </Typography>
                {starredTrialLabels.length > 0 && (
                  <Typography sx={{ fontSize: 11, color: themeColors.textMuted, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", mt: 0.25 }}>
                    starred: {starredTrialLabels.join(", ")}
                  </Typography>
                )}
                {selectedSnapshotImageLabel && !isReferenceLabel(selectedSnapshotImageLabel) && (
                  <Box sx={{ ...controlRow, width: "100%", border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, mt: 0.5, mb: 0.5 }}>
                    <Typography sx={{ ...typography.label, color: themeColors.textMuted, flexShrink: 0 }}>note</Typography>
                    <TextField
                      size="small"
                      value={selectedTrialNote}
                      onChange={(event) => setTrialNoteForLabel(selectedSnapshotImageLabel, event.target.value)}
                      placeholder="add note"
                      inputProps={{ "aria-label": `Note for ${selectedSnapshotImageLabel}` }}
                      sx={{
                        flex: "1 1 130px",
                        minWidth: 120,
                        "& .MuiInputBase-root": { height: 26, fontSize: 11, bgcolor: themeColors.bg, color: themeColors.text },
                        "& .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.border },
                      }}
                    />
                    {["best", "bad start", "probe drift", "object issue"].map((tag) => {
                      const active = selectedTrialTags.includes(tag);
                      return (
                        <Button
                          key={tag}
                          size="small"
                          variant={active ? "contained" : "outlined"}
                          onClick={() => toggleTrialTag(selectedSnapshotImageLabel, tag)}
                          sx={{
                            minWidth: 0,
                            height: 24,
                            px: 0.75,
                            py: 0,
                            fontSize: 10,
                            textTransform: "none",
                            color: active ? "#fff" : themeColors.text,
                            bgcolor: active ? themeColors.accent : "transparent",
                            borderColor: active ? themeColors.accent : themeColors.border,
                          }}
                          aria-label={`${active ? "Remove" : "Add"} ${tag} tag for ${selectedSnapshotImageLabel}`}
                        >
                          {tag}
                        </Button>
                      );
                    })}
                  </Box>
                )}
                <Box sx={{ ...controlRow, width: "100%", flexWrap: "nowrap", border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, mb: 0.5 }}>
                  <Typography sx={{ ...typography.label, color: themeColors.textMuted, flexShrink: 0 }}>play</Typography>
                  <Stack direction="row" spacing={0} sx={{ flexShrink: 0 }}>
                    <Tooltip title="Previous Snapshot">
                      <IconButton size="small" onClick={() => selectSnapshotGroup(Math.max(0, selectedGroup - 1))} sx={{ color: themeColors.textMuted, p: 0.25 }} aria-label="Previous snapshot">
                        <FastRewindIcon sx={{ fontSize: 18 }} />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title={snapshotPlaying ? "Pause" : "Play"}>
                      <IconButton size="small" onClick={() => setSnapshotPlaying(!snapshotPlaying)} sx={{ color: themeColors.accent, p: 0.25 }} aria-label={snapshotPlaying ? "Pause playback" : "Play"}>
                        {snapshotPlaying ? <PauseIcon sx={{ fontSize: 18 }} /> : <PlayArrowIcon sx={{ fontSize: 18 }} />}
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Next Snapshot">
                      <IconButton size="small" onClick={() => selectSnapshotGroup(Math.min(groupCount - 1, selectedGroup + 1))} sx={{ color: themeColors.textMuted, p: 0.25 }} aria-label="Next snapshot">
                        <FastForwardIcon sx={{ fontSize: 18 }} />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Stop">
                      <IconButton
                        size="small"
                        onClick={() => {
                          setSnapshotPlaying(false);
                          selectSnapshotGroup(0);
                        }}
                        sx={{ color: themeColors.textMuted, p: 0.25 }}
                        aria-label="Stop and rewind snapshots"
                      >
                        <StopIcon sx={{ fontSize: 16 }} />
                      </IconButton>
                    </Tooltip>
                  </Stack>
                  <Slider
                    size="small"
                    value={Math.max(0, selectedGroup)}
                    min={0}
                    max={Math.max(0, groupCount - 1)}
                    step={1}
                    marks={snapshotSliderMarks}
                    valueLabelDisplay="auto"
                    valueLabelFormat={(value) => `${Number(value) + 1}/${Math.max(1, groupCount)}`}
                    onChange={(_, value) => selectSnapshotGroup(Array.isArray(value) ? value[0] : value)}
                    sx={{ ...sliderStyles.small, flex: "1 1 130px", minWidth: 86, color: themeColors.accent, "& .MuiSlider-mark": { bgcolor: themeColors.accent, width: 4, height: 4, borderRadius: "50%", top: "50%", transform: "translate(-50%, -50%)" } }}
                    aria-label={`Current snapshot group (${Math.max(0, selectedGroup) + 1} of ${Math.max(1, groupCount)})`}
                  />
                </Box>
                <Box sx={{ ...controlRow, width: "100%", border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, mb: 0.5 }}>
                  <Typography sx={{ ...typography.label, color: themeColors.textMuted, flexShrink: 0 }}>fps</Typography>
                  <Slider
                    size="small"
                    value={clampSnapshotFps(snapshotFps)}
                    min={0.25}
                    max={12}
                    step={0.25}
                    onChange={(_, value) => setSnapshotFps(Array.isArray(value) ? value[0] : value)}
                    sx={{ ...sliderStyles.small, width: 56, flexShrink: 0, color: themeColors.accent }}
                    aria-label="Snapshot playback frames per second"
                    valueLabelDisplay="auto"
                  />
                  <Typography sx={{ ...typography.value, color: themeColors.textMuted, minWidth: 28, textAlign: "right", flexShrink: 0 }}>
                    {formatCompactValue(clampSnapshotFps(snapshotFps), 2)}
                  </Typography>
                  <Typography sx={{ ...typography.label, color: themeColors.textMuted, flexShrink: 0 }}>thumb size</Typography>
                  <Slider
                    size="small"
                    value={thumbnailSize}
                    min={24}
                    max={112}
                    step={4}
                    onChange={(_, value) => setSnapshotThumbnailSize(Array.isArray(value) ? value[0] : value)}
                    sx={{ ...sliderStyles.small, width: 64, flexShrink: 0, color: themeColors.accent }}
                    aria-label="Snapshot thumbnail size"
                    valueLabelDisplay="auto"
                  />
                  <Typography sx={{ ...typography.value, color: themeColors.textMuted, minWidth: 28, textAlign: "right", flexShrink: 0 }}>
                    {thumbnailSize}
                  </Typography>
                  <Typography sx={{ ...typography.label, color: themeColors.textMuted, flexShrink: 0 }}>plot thumbs</Typography>
                  <Switch
                    size="small"
                    checked={Boolean(showSnapshotThumbnails)}
                    onChange={(_, checked) => setShowSnapshotThumbnails(checked)}
                    sx={{ ...switchStyles.small, flexShrink: 0 }}
                    slotProps={{ input: { "aria-label": "Show snapshot thumbnails on plot" } }}
                  />
                </Box>
                <Box sx={{ ...controlRow, width: "100%", border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, mb: 0.5 }}>
                  <Typography sx={{ ...typography.label, color: themeColors.textMuted, flexShrink: 0 }}>FFT</Typography>
                  <Switch
                    size="small"
                    checked={Boolean(showSnapshotFft)}
                    onChange={(_, checked) => setShowSnapshotFft(checked)}
                    sx={{ ...switchStyles.small, flexShrink: 0 }}
                    slotProps={{ input: { "aria-label": "Show snapshot FFT panels" } }}
                  />
                </Box>
                {showSnapshotFft && (
                  <Box sx={{ ...controlRow, width: "100%", border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, mb: 0.75 }}>
                    <Typography sx={{ ...typography.label, color: themeColors.textMuted, flexShrink: 0 }}>FFT</Typography>
                    <Select
                      size="small"
                      value={COLORMAPS[snapshotFftCmap] ? snapshotFftCmap : "magma"}
                      onChange={(event) => setSnapshotFftCmap(event.target.value)}
                      sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }}
                      MenuProps={themedMenuProps}
                      inputProps={{ "aria-label": "Snapshot FFT colormap" }}
                    >
                      {COLORMAP_NAMES.map((name) => (
                        <MenuItem key={name} value={name}>{name}</MenuItem>
                      ))}
                    </Select>
                    <Typography sx={{ ...typography.label, color: themeColors.textMuted, flexShrink: 0 }}>win</Typography>
                    <Switch
                      size="small"
                      checked={Boolean(snapshotFftWindow)}
                      onChange={(_, checked) => setSnapshotFftWindow(checked)}
                      sx={{ ...switchStyles.small, flexShrink: 0 }}
                      slotProps={{ input: { "aria-label": "Apply Hann window before snapshot FFT" } }}
                    />
                  </Box>
                )}
                {selectedSnapshot >= 0 && (
                  <Box sx={{ mb: 0.75 }}>
                    <MiniHistogram
                      bins={snapshotHistogramBins}
                      dataMin={snapshotHistogramRange[0]}
                      dataMax={snapshotHistogramRange[1]}
                      clipMin={snapshotHistogramClipRange[0]}
                      clipMax={snapshotHistogramClipRange[1]}
                      backend={snapshotHistogramBackend}
                      colors={themeColors}
                    />
                    <Box sx={{ ...controlRow, width: "100%", border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg, mt: 0.5 }}>
                      <Typography sx={{ ...typography.label, color: themeColors.textMuted, flexShrink: 0 }}>cmap</Typography>
                      <Select
                        size="small"
                        value={COLORMAPS[imageCmap] ? imageCmap : "cividis"}
                        onChange={(event) => setImageCmap(event.target.value)}
                        sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }}
                        MenuProps={themedMenuProps}
                        inputProps={{ "aria-label": "Snapshot image colormap" }}
                      >
                        {COLORMAP_NAMES.map((name) => (
                          <MenuItem key={name} value={name}>{name}</MenuItem>
                        ))}
                      </Select>
                      <Typography sx={{ ...typography.label, color: themeColors.textMuted, flexShrink: 0 }}>clip</Typography>
                      {snapshotContrastPresets.map((preset) => {
                        const active = preset.value === normalisedSnapshotContrastPreset;
                        return (
                          <Button
                            key={preset.value}
                            size="small"
                            variant={active ? "contained" : "outlined"}
                            onClick={() => setSnapshotContrastPreset(preset.value)}
                            sx={{
                              minWidth: preset.value === "full" ? 38 : 48,
                              height: 24,
                              px: 0.75,
                              py: 0,
                              fontSize: 10,
                              lineHeight: 1,
                              textTransform: "none",
                              color: active ? "#fff" : themeColors.text,
                              bgcolor: active ? themeColors.accent : "transparent",
                              borderColor: active ? themeColors.accent : themeColors.border,
                              "&:hover": {
                                bgcolor: active ? themeColors.accent : themeColors.bgAlt,
                                borderColor: themeColors.accent,
                              },
                            }}
                            aria-label={`Set snapshot contrast clip ${preset.label}`}
                          >
                            {preset.label}
                          </Button>
                        );
                      })}
                    </Box>
                  </Box>
                )}
              </Box>
            )}
            {nTraces > 1 && (
              <Box>
                <Stack direction="row" alignItems="center" spacing={0.75} sx={{ mb: 0.5 }}>
                  <Typography sx={{ fontSize: 12, fontWeight: 600 }}>Review</Typography>
                  <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>
                    best {bestTrialLabel || String(runSummary?.best_trial || sortedTrialRows[0]?.label || "")}
                  </Typography>
                  <Box sx={{ flex: 1 }} />
                  <Typography sx={{ fontSize: 10.5, color: themeColors.textMuted }}>
                    {activeReviewCount}/{nTraces} visible · {trialAlerts?.length ?? 0} alerts
                  </Typography>
                </Stack>
                <Divider sx={{ borderColor: themeColors.border, mb: 0.5 }} />
                {trialAlerts?.length > 0 && (
                  <Box sx={{ mb: 0.5 }}>
                    {trialAlerts.slice(0, 3).map((alert, idx) => (
                      <Typography key={`${alert.label || "run"}-${alert.kind || idx}`} sx={{ fontSize: 10.5, color: alert.severity === "error" ? "#b91c1c" : themeColors.textMuted, lineHeight: 1.25 }}>
                        {alert.label ? `${alert.label}: ` : ""}{alert.message || alert.kind}
                      </Typography>
                    ))}
                  </Box>
                )}
                <Box component="table" sx={{ width: "100%", borderCollapse: "collapse", tableLayout: "fixed", "& th, & td": { fontSize: 10.5, py: 0.28, textAlign: "right", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }, "& th:first-of-type, & td:first-of-type": { textAlign: "left" } }}>
                  <thead>
                    <tr>
                      <th>Trial</th><th>{normalisedSortKey.replace("_", " ")}</th><th>Alerts</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sortedTrialRows.slice(0, 6).map((row) => {
                      const key = trialKey(row.label || "");
                      const hidden = hiddenTrialKeys.has(key) || !passesReviewFilter(row.label || "");
                      return (
                        <tr key={key} style={{ opacity: hidden ? 0.45 : 1 }}>
                          <td title={row.label || ""}>
                            <Box component="span" sx={{ display: "inline-block", width: 7, height: 7, bgcolor: cssColor(colors ?? [], Number(row.trace_index) || 0), mr: 0.5, borderRadius: "50%" }} />
                            {starredTrialKeys.has(key) ? "* " : ""}{row.label}
                          </td>
                          <td>{rankingDisplayValue(row, normalisedSortKey)}</td>
                          <td>{Number(row.alert_count) || 0}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </Box>
              </Box>
            )}
            {showStats && visibleTraceIndices.length > 0 && (
              <Box>
                <Stack direction="row" alignItems="center" spacing={0.75} sx={{ mb: 0.5 }}>
                  <Typography sx={{ fontSize: 12, fontWeight: 600 }}>Stats</Typography>
                  <Box sx={{ flex: 1 }} />
                  <Tooltip title={showStats ? "Hide Stats" : "Show Stats"}>
                    <IconButton size="small" onClick={() => setShowStats(!showStats)} sx={{ color: themeColors.textMuted }}>
                      {showStats ? <VisibilityIcon fontSize="inherit" /> : <VisibilityOffIcon fontSize="inherit" />}
                    </IconButton>
                  </Tooltip>
                </Stack>
                <Divider sx={{ borderColor: themeColors.border, mb: 0.5 }} />
                <Box component="table" sx={{ width: "100%", borderCollapse: "collapse", tableLayout: "fixed", "& th, & td": { fontSize: 10.5, py: 0.35, textAlign: "right", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }, "& th:first-of-type, & td:first-of-type": { textAlign: "left" } }}>
                  <thead>
                    <tr>
                      <th>Trace</th><th>Mean</th><th>Min</th><th>Max</th><th>Std</th>
                    </tr>
                  </thead>
                  <tbody>
                    {visibleTraceIndices.map((idx) => (
                      <tr key={idx}>
                        <td title={labels?.[idx] || `Trace ${idx + 1}`}>
                          <Box component="span" sx={{ display: "inline-block", width: 7, height: 7, bgcolor: cssColor(colors ?? [], idx), mr: 0.5, borderRadius: "50%" }} />
                          {labels?.[idx] || `Trace ${idx + 1}`}
                        </td>
                        <td>{formatNumber(statsMean?.[idx] ?? NaN, 2)}</td>
                        <td>{formatNumber(statsMin?.[idx] ?? NaN, 2)}</td>
                        <td>{formatNumber(statsMax?.[idx] ?? NaN, 2)}</td>
                        <td>{formatNumber(statsStd?.[idx] ?? NaN, 2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </Box>
              </Box>
            )}
          </Stack>
        )}
      </Box>
      {handoffEnabled && preparedViewWidget != null && (
        <EmbeddedWidgetView
          hostModel={model}
          widgetModel={preparedViewWidget}
          title="2D view"
          onClose={handleClosePreparedView}
          themeColors={themeColors}
          linkedTraits={SHOW1D_TO_SHOW2D_LINKED_TRAITS}
        />
      )}
    </Box>
  );
}

export default { render: createRender(Show1DWidget) };
