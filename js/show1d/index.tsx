import * as React from "react";
import { createRender, useModel, useModelState } from "@anywidget/react";
import { useHideStaticFallback } from "../staticFallback";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import IconButton from "@mui/material/IconButton";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import Select from "@mui/material/Select";
import Slider from "@mui/material/Slider";
import Stack from "@mui/material/Stack";
import Switch from "@mui/material/Switch";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";
import DownloadIcon from "@mui/icons-material/Download";
import FastForwardIcon from "@mui/icons-material/FastForward";
import FastRewindIcon from "@mui/icons-material/FastRewind";
import ImageIcon from "@mui/icons-material/Image";
import PauseIcon from "@mui/icons-material/Pause";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import ShowChartIcon from "@mui/icons-material/ShowChart";
import StopIcon from "@mui/icons-material/Stop";
import TableChartIcon from "@mui/icons-material/TableChart";
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
  mobile_label?: string;
  kind?: string;
};

function markerColor(marker: Marker, colors: ReturnType<typeof useTheme>["colors"]): string {
  if (marker.kind === "increase") return "#00897b";
  if (marker.kind === "decrease") return "#d1495b";
  if (marker.kind === "checkpoint") return "#f59e0b";
  return colors.textMuted;
}

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

type ImageViewApiState = {
  zoom: number;
  center: number[];
};

type SnapshotOverlayPosition = "top-left" | "top-right" | "bottom-left" | "bottom-right";

type SnapshotFftCacheEntry = {
  data: Float32Array;
  width: number;
  height: number;
  backend: string;
};

type SnapshotFftCacheRef = React.MutableRefObject<Map<string, SnapshotFftCacheEntry>>;
type SnapshotFftPendingRef = React.MutableRefObject<Map<string, Promise<SnapshotFftCacheEntry>>>;

type PlotThumbnailCacheEntry = {
  canvas: HTMLCanvasElement;
  width: number;
  height: number;
  iteration: number;
};

type PlotThumbnailHitArea = {
  groupIdx: number;
  x0: number;
  y0: number;
  width: number;
  height: number;
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

type Show1DInitialInteractiveState = {
  logScale: boolean;
  showStats: boolean;
  showLegend: boolean;
  plotHeightPx: number;
  sidePanelWidthPx: number;
  snapshotPanelWidthPx: number;
  focusedTrace: number;
  xRange: number[];
  yRange: number[];
  selectedSnapshotIdx: number;
  selectedSnapshotGroupIdx: number;
  bookmarkedSnapshotGroups: number[];
  hiddenSnapshotImageLabels: string[];
  showSnapshotFft: boolean;
  snapshotFftLayout: string;
  snapshotFftWindow: boolean;
  snapshotFftCmap: string;
  snapshotContrastPreset: string;
  snapshotContrastRange: number[];
  snapshotThumbnailSize: number;
  snapshotOverlayPosition: string;
  imageCmap: string;
  snapshotRealSpaceZoom: number;
  snapshotRealSpaceCenter: number[];
  snapshotFftZoom: number;
  snapshotFftCenter: number[];
  showSnapshotProfile: boolean;
  snapshotProfileLine: ProfilePoint[];
  snapshotPlaying: boolean;
  snapshotFps: number;
  snapshotLoop: boolean;
  snapshotBounce: boolean;
};

const EMPTY_BYTES = new Uint8Array(0);
const DEFAULT_SIZE = { width: 820, height: 380 };
const DEFAULT_PLOT_HEIGHT = 390;
const SNAPSHOT_PLAYBACK_CONTROL_WIDTH = 620;
const SNAPSHOT_OVERLAY_POSITIONS: SnapshotOverlayPosition[] = ["top-left", "top-right", "bottom-left", "bottom-right"];
const PROFILE_COLORS = ["#4fc3f7", "#81c784", "#ffb74d", "#ce93d8", "#ef5350", "#ffd54f", "#90a4ae", "#a1887f"];
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
const compactButton = {
  fontSize: 10,
  py: 0.25,
  px: 1,
  minWidth: 0,
  textTransform: "none" as const,
  "&.Mui-disabled": {
    color: "#666",
    borderColor: "#444",
  },
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
const MIN_SIDE_PANEL_WIDTH = 300;
const MAX_SIDE_PANEL_WIDTH = 4096;
const MIN_SNAPSHOT_VIEWPORT_WIDTH = 220;
const MIN_SNAPSHOT_TILE_WIDTH = 48;
const MAX_SNAPSHOT_VIEWPORT_WIDTH = 4096;
const MIN_PLOT_WIDTH = 220;
const MIN_PLOT_HEIGHT = 220;
const MAX_PLOT_HEIGHT = 960;
const FFT_DISPLAY_RANGE: [number, number] = [0, 1];
let snapshotFftWebGpuUnavailable = false;
let snapshotFftWebGpuInitPromise: Promise<WebGPUFFT | null> | null = null;
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

function useViewportSize() {
  const [size, setSize] = React.useState(() => ({
    width: typeof window === "undefined" ? 1280 : window.innerWidth,
    height: typeof window === "undefined" ? 800 : window.innerHeight,
  }));
  React.useEffect(() => {
    const update = () => setSize({ width: window.innerWidth, height: window.innerHeight });
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);
  return size;
}

function copyNumberArray(value: number[] | null | undefined): number[] {
  return Array.isArray(value) ? value.map(Number).filter((item) => Number.isFinite(item)) : [];
}

function copyStringArray(value: string[] | null | undefined): string[] {
  return Array.isArray(value) ? value.map(String) : [];
}

function copyProfileLine(value: ProfilePoint[] | null | undefined): ProfilePoint[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((point) => ({ row: Number(point?.row), col: Number(point?.col) }))
    .filter((point) => Number.isFinite(point.row) && Number.isFinite(point.col));
}

function numberArraysEqual(a: number[] | null | undefined, b: number[] | null | undefined): boolean {
  const left = Array.isArray(a) ? a : [];
  const right = Array.isArray(b) ? b : [];
  if (left.length !== right.length) return false;
  for (let idx = 0; idx < left.length; idx += 1) {
    if (left[idx] !== right[idx]) return false;
  }
  return true;
}

function stringArraysEqual(a: string[] | null | undefined, b: string[] | null | undefined): boolean {
  const left = Array.isArray(a) ? a : [];
  const right = Array.isArray(b) ? b : [];
  if (left.length !== right.length) return false;
  for (let idx = 0; idx < left.length; idx += 1) {
    if (left[idx] !== right[idx]) return false;
  }
  return true;
}

function profileLinesEqual(a: ProfilePoint[] | null | undefined, b: ProfilePoint[] | null | undefined): boolean {
  const left = copyProfileLine(a);
  const right = copyProfileLine(b);
  if (left.length !== right.length) return false;
  for (let idx = 0; idx < left.length; idx += 1) {
    if (left[idx].row !== right[idx].row || left[idx].col !== right[idx].col) return false;
  }
  return true;
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

function compactScienceLabel(label: string): string {
  let text = String(label || "").trim();
  if (!text) return text;
  text = text.replace(/_/g, " ");
  text = text.replace(/\bframe\s*[- ]\s*by\s*[- ]\s*frame\b/gi, "frame");
  text = text.replace(/\bjoint\s+lambda\s*/gi, "λ");
  text = text.replace(/\blambda\s*/gi, "λ");
  text = text.replace(/\balpha\s*/gi, "α");
  text = text.replace(/\bbeta\s*/gi, "β");
  text = text.replace(/\bgamma\s*/gi, "γ");
  text = text.replace(/\bsigma\s*/gi, "σ");
  text = text.replace(/\s*\/\s*/g, "/");
  return text.replace(/\s+/g, " ").trim();
}

function normaliseSnapshotOverlayPosition(value: string): SnapshotOverlayPosition {
  const clean = String(value || "").toLowerCase().replace(/_/g, "-");
  return SNAPSHOT_OVERLAY_POSITIONS.includes(clean as SnapshotOverlayPosition)
    ? clean as SnapshotOverlayPosition
    : "top-right";
}

function normaliseSnapshotFftLayout(value: string): "overlay" | "below" {
  return String(value || "").toLowerCase() === "below" ? "below" : "overlay";
}

function labelPositionAwayFromInset(insetPosition: string): "top-left" | "top-right" {
  return normaliseSnapshotOverlayPosition(insetPosition).endsWith("right") ? "top-left" : "top-right";
}

function shortMethodLabel(label: string): string {
  return compactScienceLabel(label);
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

function axisPositionText(value: number, label: string, unit: string): string {
  const formatted = formatAxisValue(value);
  if (!formatted) return "";
  const axis = label.trim() || "x";
  return `${axis} ${formatted}${unit.trim() ? ` ${unit.trim()}` : ""}`;
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
  return Math.round(clampValue(Number.isFinite(value) ? value : 2, 1, 24));
}

function clampThumbnailSize(value: number): number {
  return Math.round(clampValue(Number.isFinite(value) ? value : 48, 24, 112));
}

function autoSnapshotColumnsForCount(count: number): number {
  if (count >= 12) return 6;
  if (count >= 9) return 5;
  if (count >= 6) return 4;
  if (count >= 3) return 3;
  return Math.max(1, count);
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

function normaliseImageViewCenter(center: number[] | null | undefined, imageH: number, imageW: number): [number, number] {
  if (center && center.length === 2 && Number.isFinite(center[0]) && Number.isFinite(center[1])) {
    return [
      clampValue(Number(center[0]), 0, Math.max(1, imageH)),
      clampValue(Number(center[1]), 0, Math.max(1, imageW)),
    ];
  }
  return [Math.max(1, imageH) / 2, Math.max(1, imageW) / 2];
}

function imageApiToView(
  zoom: number,
  center: number[] | null | undefined,
  canvasW: number,
  canvasH: number,
  imageW: number,
  imageH: number,
): ImageViewState {
  const cleanZoom = clampImageZoom(zoom);
  const [centerRow, centerCol] = normaliseImageViewCenter(center, imageH, imageW);
  const fit = Math.min(canvasW / Math.max(1, imageW), canvasH / Math.max(1, imageH));
  return clampImagePan(
    (Math.max(1, imageW) / 2 - centerCol) * fit * cleanZoom,
    (Math.max(1, imageH) / 2 - centerRow) * fit * cleanZoom,
    cleanZoom,
    canvasW,
    canvasH,
    imageW,
    imageH,
  );
}

function imageViewToApi(
  view: ImageViewState,
  canvasW: number,
  canvasH: number,
  imageW: number,
  imageH: number,
): ImageViewApiState {
  const cleanZoom = clampImageZoom(view.zoom);
  if (cleanZoom <= 1.0001) return { zoom: 1, center: [] };
  const fit = Math.min(canvasW / Math.max(1, imageW), canvasH / Math.max(1, imageH));
  const scale = Math.max(fit * cleanZoom, 1e-12);
  const centerRow = Math.max(1, imageH) / 2 - view.panY / scale;
  const centerCol = Math.max(1, imageW) / 2 - view.panX / scale;
  return {
    zoom: Number(cleanZoom.toFixed(4)),
    center: [Number(centerRow.toFixed(3)), Number(centerCol.toFixed(3))],
  };
}

function sampleLineProfile(data: Float32Array, width: number, height: number, row0: number, col0: number, row1: number, col1: number): Float32Array {
  const dc = col1 - col0;
  const dr = row1 - row0;
  const length = Math.sqrt(dc * dc + dr * dr);
  const n = Math.max(2, Math.ceil(length) + 1);
  const out = new Float32Array(n);
  for (let idx = 0; idx < n; idx += 1) {
    const t = idx / Math.max(1, n - 1);
    const col = col0 + t * dc;
    const row = row0 + t * dr;
    const ci = Math.floor(col);
    const ri = Math.floor(row);
    const cf = col - ci;
    const rf = row - ri;
    const c0 = clampValue(ci, 0, width - 1);
    const c1 = clampValue(ci + 1, 0, width - 1);
    const r0 = clampValue(ri, 0, height - 1);
    const r1 = clampValue(ri + 1, 0, height - 1);
    out[idx] = data[r0 * width + c0] * (1 - cf) * (1 - rf)
      + data[r0 * width + c1] * cf * (1 - rf)
      + data[r1 * width + c0] * (1 - cf) * rf
      + data[r1 * width + c1] * cf * rf;
  }
  return out;
}

function pointToSegmentDistance(col: number, row: number, col0: number, row0: number, col1: number, row1: number): number {
  const dc = col1 - col0;
  const dr = row1 - row0;
  const lenSq = dc * dc + dr * dr;
  if (lenSq <= 1e-12) return Math.sqrt((col - col0) ** 2 + (row - row0) ** 2);
  const t = clampValue(((col - col0) * dc + (row - row0) * dr) / lenSq, 0, 1);
  const projCol = col0 + t * dc;
  const projRow = row0 + t * dr;
  return Math.sqrt((col - projCol) ** 2 + (row - projRow) ** 2);
}

function isFiniteProfilePoint(point: ProfilePoint | undefined): point is Required<ProfilePoint> {
  return Boolean(point && Number.isFinite(point.row) && Number.isFinite(point.col));
}

function clampProfilePoint(point: ProfilePoint, height: number, width: number): Required<ProfilePoint> {
  return {
    row: clampValue(Number(point.row ?? 0), 0, Math.max(0, height - 1)),
    col: clampValue(Number(point.col ?? 0), 0, Math.max(0, width - 1)),
  };
}

function normaliseSnapshotContrastPreset(value: string): string {
  return snapshotContrastPresets.some((preset) => preset.value === value) ? value : "full";
}

function normaliseSnapshotContrastRange(value: number[] | undefined | null): [number, number] | null {
  if (!value || value.length !== 2) return null;
  const lo = Number(value[0]);
  const hi = Number(value[1]);
  return Number.isFinite(lo) && Number.isFinite(hi) && lo < hi ? [lo, hi] : null;
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

function resolveSnapshotDisplayRange(data: Float32Array, presetValue: string, customRange?: [number, number] | null): [number, number] {
  if (customRange) return customRange;
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

async function getSnapshotFftWebGpu(
  gpuFftRef: React.MutableRefObject<WebGPUFFT | null>,
): Promise<WebGPUFFT | null> {
  if (gpuFftRef.current) return gpuFftRef.current;
  if (snapshotFftWebGpuUnavailable) return null;
  if (!snapshotFftWebGpuInitPromise) {
    snapshotFftWebGpuInitPromise = getWebGPUFFT()
      .then((fft) => {
        if (!fft) snapshotFftWebGpuUnavailable = true;
        return fft;
      })
      .catch(() => {
        snapshotFftWebGpuUnavailable = true;
        return null;
      })
      .finally(() => {
        snapshotFftWebGpuInitPromise = null;
      });
  }
  const fft = await snapshotFftWebGpuInitPromise;
  gpuFftRef.current = fft;
  return fft;
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
  if (preferWebgpu && !snapshotFftWebGpuUnavailable) {
    try {
      const fft = await getSnapshotFftWebGpu(gpuFftRef);
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
      snapshotFftWebGpuUnavailable = true;
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

function snapshotFftCacheKey(
  imageIndex: number,
  width: number,
  height: number,
  useWindow: boolean,
  preferWebgpu: boolean,
): string {
  return `${imageIndex}:${width}x${height}:${useWindow ? "hann" : "raw"}:${preferWebgpu ? "gpu" : "cpu"}`;
}

function computeSnapshotFftCached(
  cacheKey: string,
  image: Float32Array,
  width: number,
  height: number,
  useWindow: boolean,
  preferWebgpu: boolean,
  cacheRef: SnapshotFftCacheRef,
  pendingRef: SnapshotFftPendingRef,
  gpuFftRef: React.MutableRefObject<WebGPUFFT | null>,
): Promise<SnapshotFftCacheEntry> {
  const cached = cacheRef.current.get(cacheKey);
  if (cached) return Promise.resolve(cached);
  const pending = pendingRef.current.get(cacheKey);
  if (pending) return pending;
  const promise = computeSnapshotFft(image, width, height, useWindow, preferWebgpu, gpuFftRef)
    .then((entry) => {
      cacheRef.current.set(cacheKey, entry);
      return entry;
    })
    .finally(() => {
      pendingRef.current.delete(cacheKey);
    });
  pendingRef.current.set(cacheKey, promise);
  return promise;
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
  drawImageBorder = true,
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
  if (drawImageBorder) {
    ctx.strokeStyle = colors.border;
    ctx.lineWidth = 1;
    ctx.strokeRect(x0, y0, drawW, drawH);
  }
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

function createSnapshotGroupThumbnailCanvas(
  images: { data: Float32Array; height: number; width: number }[],
  size: number,
  lut: Uint8Array,
  colors: { bgAlt: string; border: string; accent: string; text: string },
  contrastPreset: string,
): { canvas: HTMLCanvasElement; width: number; height: number } | null {
  if (typeof document === "undefined") return null;
  const valid = images.filter((image) => image.data.length && image.height > 0 && image.width > 0).slice(0, 9);
  if (!valid.length || size <= 0) return null;
  const metrics = snapshotGroupThumbnailMetrics(valid.length, size);
  if (metrics.width <= 0 || metrics.height <= 0) return null;
  const canvas = document.createElement("canvas");
  canvas.width = metrics.width;
  canvas.height = metrics.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  drawSnapshotGroupThumbnail(ctx, valid, 0, 0, size, false, lut, colors, contrastPreset);
  return { canvas, width: metrics.width, height: metrics.height };
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
  overlayPosition,
  pixelSize,
  pixelUnit,
  ariaLabel,
  viewZoom,
  viewCenter,
  overlayTextVisible = true,
  imageBorderVisible = true,
  profileActive = false,
  profileLine = [],
  onViewChange,
  onProfileLineChange,
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
  overlayPosition: string;
  pixelSize: number;
  pixelUnit: string;
  ariaLabel: string;
  viewZoom: number;
  viewCenter: number[];
  overlayTextVisible?: boolean;
  imageBorderVisible?: boolean;
  profileActive?: boolean;
  profileLine?: ProfilePoint[];
  onViewChange: (view: ImageViewApiState) => void;
  onProfileLineChange?: (line: ProfilePoint[]) => void;
  onSelect?: () => void;
}) {
  const hostRef = React.useRef<HTMLDivElement>(null);
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const overlayRef = React.useRef<HTMLCanvasElement>(null);
  const viewRef = React.useRef<ImageViewState>({ zoom: 1, panX: 0, panY: 0 });
  const pendingViewRef = React.useRef<ImageViewState | null>(null);
  const viewRafRef = React.useRef<number | null>(null);
  const dragRef = React.useRef<{ pointerId: number; x: number; y: number; panX: number; panY: number; zoom: number } | null>(null);
  const profileDragRef = React.useRef<{
    pointerId: number;
    mode: "start" | "end" | "line" | "new";
    x: number;
    y: number;
    row: number;
    col: number;
    p0?: Required<ProfilePoint>;
    p1?: Required<ProfilePoint>;
  } | null>(null);
  const [profileHoverMode, setProfileHoverMode] = React.useState<"endpoint" | "line" | null>(null);
  const [drawTick, setDrawTick] = React.useState(0);
  const resolvedOverlayPosition = normaliseSnapshotOverlayPosition(overlayPosition);
  const overlayOnRight = resolvedOverlayPosition.endsWith("right");
  const overlayOnBottom = resolvedOverlayPosition.startsWith("bottom");

  const currentApiView = React.useCallback(() => {
    const rect = canvasRef.current?.getBoundingClientRect();
    return imageApiToView(
      viewZoom,
      viewCenter,
      rect?.width ?? Math.max(1, width),
      rect?.height ?? Math.max(1, height),
      width,
      height,
    );
  }, [height, viewCenter, viewZoom, width]);

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

  const imagePointFromClient = React.useCallback((clientX: number, clientY: number) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect || width <= 0 || height <= 0) return null;
    const view = viewRef.current;
    const fit = Math.min(rect.width / Math.max(1, width), rect.height / Math.max(1, height));
    const zoom = clampImageZoom(view.zoom);
    const scale = Math.max(1e-12, fit * zoom);
    const drawW = width * scale;
    const drawH = height * scale;
    const x0 = rect.width / 2 - drawW / 2 + view.panX;
    const y0 = rect.height / 2 - drawH / 2 + view.panY;
    const col = (clientX - rect.left - x0) / scale;
    const row = (clientY - rect.top - y0) / scale;
    if (col < 0 || col > width - 1 || row < 0 || row > height - 1) return null;
    return {
      row,
      col,
      hitRadius: 10 / scale,
    };
  }, [height, width]);

  const profileHitTest = React.useCallback((row: number, col: number, hitRadius: number) => {
    if (!profileActive || !isFiniteProfilePoint(profileLine[0]) || !isFiniteProfilePoint(profileLine[1])) return null;
    const p0 = clampProfilePoint(profileLine[0], height, width);
    const p1 = clampProfilePoint(profileLine[1], height, width);
    const d0 = Math.sqrt((col - p0.col) ** 2 + (row - p0.row) ** 2);
    const d1 = Math.sqrt((col - p1.col) ** 2 + (row - p1.row) ** 2);
    if (d0 <= hitRadius || d1 <= hitRadius) return { mode: d0 <= d1 ? "start" as const : "end" as const, p0, p1 };
    if (pointToSegmentDistance(col, row, p0.col, p0.row, p1.col, p1.row) <= hitRadius) return { mode: "line" as const, p0, p1 };
    return null;
  }, [height, profileActive, profileLine, width]);

  const scheduleView = React.useCallback((next: ImageViewState) => {
    const clean = cleanView(next);
    pendingViewRef.current = clean;
    viewRef.current = clean;
    if (viewRafRef.current !== null) return;
    viewRafRef.current = window.requestAnimationFrame(() => {
      viewRafRef.current = null;
      if (pendingViewRef.current) {
        const rect = canvasRef.current?.getBoundingClientRect();
        onViewChange(imageViewToApi(
          pendingViewRef.current,
          rect?.width ?? Math.max(1, width),
          rect?.height ?? Math.max(1, height),
          width,
          height,
        ));
      }
    });
  }, [cleanView, height, onViewChange, width]);

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
    const localView = cleanView(currentApiView());
    viewRef.current = localView;
    drawFloatImage(canvas, data, height, width, [], lut, {
      bg: colors.bgAlt,
      border: selected ? colors.accent : colors.border,
      accent: colors.accent,
      text: colors.text,
    }, localView, false, displayRange, imageBorderVisible);
  }, [cleanView, colors, currentApiView, data, displayRange, drawTick, height, imageBorderVisible, lut, selected, width]);

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
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    const localView = cleanView(currentApiView());
    if (profileActive && isFiniteProfilePoint(profileLine[0])) {
      const fit = Math.min(cssW / Math.max(1, width), cssH / Math.max(1, height));
      const zoom = clampImageZoom(localView.zoom);
      const scale = fit * zoom;
      const drawW = width * scale;
      const drawH = height * scale;
      const x0 = cssW / 2 - drawW / 2 + localView.panX;
      const y0 = cssH / 2 - drawH / 2 + localView.panY;
      const toScreen = (point: ProfilePoint) => {
        const clean = clampProfilePoint(point, height, width);
        return { x: x0 + clean.col * scale, y: y0 + clean.row * scale };
      };
      const p0 = toScreen(profileLine[0]);
      ctx.save();
      ctx.lineCap = "round";
      ctx.shadowColor = "rgba(0,0,0,0.75)";
      ctx.shadowBlur = 2;
      if (isFiniteProfilePoint(profileLine[1])) {
        const p1 = toScreen(profileLine[1]);
        ctx.strokeStyle = colors.accent;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(p0.x, p0.y);
        ctx.lineTo(p1.x, p1.y);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = colors.accent;
        ctx.beginPath();
        ctx.arc(p1.x, p1.y, 4, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.fillStyle = colors.accent;
      ctx.beginPath();
      ctx.arc(p0.x, p0.y, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
    if (!scaleBarVisible) return;
    const pxSize = pixelSize > 0 ? pixelSize : 1;
    const unit = pixelSize > 0 ? pixelUnit : "px";
    drawPanelScaleBarHiDPI(overlay, dpr, localView.zoom, pxSize, unit, width);
  }, [cleanView, colors.accent, currentApiView, drawTick, height, pixelSize, pixelUnit, profileActive, profileLine, scaleBarVisible, width]);

  const handleWheel = React.useCallback((event: WheelEvent) => {
    const target = canvasRef.current ?? hostRef.current;
    if (!target) return;
    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation();
    onSelect?.();
    const rect = target.getBoundingClientRect();
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

  React.useEffect(() => {
    const node = hostRef.current;
    if (!node) return;
    node.addEventListener("wheel", handleWheel, { capture: true, passive: false });
    return () => node.removeEventListener("wheel", handleWheel, { capture: true });
  }, [handleWheel]);

  const handlePointerDown = React.useCallback((event: React.PointerEvent<HTMLCanvasElement>) => {
    onSelect?.();
    if (profileActive && onProfileLineChange) {
      const point = imagePointFromClient(event.clientX, event.clientY);
      if (!point) return;
      const hit = profileHitTest(point.row, point.col, point.hitRadius);
      profileDragRef.current = {
        pointerId: event.pointerId,
        mode: hit?.mode ?? "new",
        x: event.clientX,
        y: event.clientY,
        row: point.row,
        col: point.col,
        p0: hit?.p0,
        p1: hit?.p1,
      };
      event.currentTarget.setPointerCapture(event.pointerId);
      event.preventDefault();
      return;
    }
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
  }, [imagePointFromClient, onProfileLineChange, onSelect, profileActive, profileHitTest]);

  const handlePointerMove = React.useCallback((event: React.PointerEvent<HTMLCanvasElement>) => {
    if (profileActive && onProfileLineChange) {
      const drag = profileDragRef.current;
      const point = imagePointFromClient(event.clientX, event.clientY);
      if (drag && drag.pointerId === event.pointerId && point) {
        if (drag.mode === "start" && drag.p1) {
          onProfileLineChange([clampProfilePoint(point, height, width), drag.p1]);
        } else if (drag.mode === "end" && drag.p0) {
          onProfileLineChange([drag.p0, clampProfilePoint(point, height, width)]);
        } else if (drag.mode === "line" && drag.p0 && drag.p1) {
          let dRow = point.row - drag.row;
          let dCol = point.col - drag.col;
          const minRow = Math.min(drag.p0.row, drag.p1.row);
          const maxRow = Math.max(drag.p0.row, drag.p1.row);
          const minCol = Math.min(drag.p0.col, drag.p1.col);
          const maxCol = Math.max(drag.p0.col, drag.p1.col);
          dRow = clampValue(dRow, -minRow, (height - 1) - maxRow);
          dCol = clampValue(dCol, -minCol, (width - 1) - maxCol);
          onProfileLineChange([
            { row: drag.p0.row + dRow, col: drag.p0.col + dCol },
            { row: drag.p1.row + dRow, col: drag.p1.col + dCol },
          ]);
        } else if (drag.mode === "new") {
          onProfileLineChange([
            clampProfilePoint({ row: drag.row, col: drag.col }, height, width),
            clampProfilePoint(point, height, width),
          ]);
        }
        event.preventDefault();
        return;
      }
      if (point) {
        const hit = profileHitTest(point.row, point.col, point.hitRadius);
        setProfileHoverMode(hit?.mode === "line" ? "line" : hit ? "endpoint" : null);
      } else if (profileHoverMode !== null) {
        setProfileHoverMode(null);
      }
      return;
    }
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    event.preventDefault();
    scheduleView({
      zoom: drag.zoom,
      panX: drag.panX + event.clientX - drag.x,
      panY: drag.panY + event.clientY - drag.y,
    });
  }, [height, imagePointFromClient, onProfileLineChange, profileActive, profileHitTest, profileHoverMode, scheduleView, width]);

  const stopDrag = React.useCallback((event: React.PointerEvent<HTMLCanvasElement>) => {
    const profileDrag = profileDragRef.current;
    if (profileDrag?.pointerId === event.pointerId) {
      const point = imagePointFromClient(event.clientX, event.clientY);
      const moved = Math.sqrt((event.clientX - profileDrag.x) ** 2 + (event.clientY - profileDrag.y) ** 2);
      if (profileActive && onProfileLineChange && profileDrag.mode === "new" && point && moved < 3) {
        const cleanPoint = clampProfilePoint(point, height, width);
        if (!isFiniteProfilePoint(profileLine[0]) || isFiniteProfilePoint(profileLine[1])) {
          onProfileLineChange([cleanPoint]);
        } else {
          onProfileLineChange([clampProfilePoint(profileLine[0], height, width), cleanPoint]);
        }
      }
      profileDragRef.current = null;
      return;
    }
    if (dragRef.current?.pointerId === event.pointerId) dragRef.current = null;
  }, [height, imagePointFromClient, onProfileLineChange, profileActive, profileLine, width]);

  const resetView = React.useCallback((event: React.MouseEvent<HTMLCanvasElement>) => {
    event.stopPropagation();
    onSelect?.();
    scheduleView({ zoom: 1, panX: 0, panY: 0 });
  }, [onSelect, scheduleView]);

  return (
    <Box ref={hostRef} sx={{ position: "relative", width: "100%", height: "100%", overflow: "hidden", overscrollBehavior: "contain", bgcolor: colors.bgAlt }}>
      <canvas
        ref={canvasRef}
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
          cursor: profileDragRef.current
            ? "grabbing"
            : profileActive
              ? profileHoverMode ? "grab" : "crosshair"
              : dragRef.current ? "grabbing" : clampImageZoom(viewZoom) > 1.01 ? "grab" : "zoom-in",
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
      {overlayTextVisible && (
        <Box
          sx={{
            position: "absolute",
            zIndex: 2,
            top: overlayOnBottom ? "auto" : 6,
            bottom: overlayOnBottom ? (scaleBarVisible && overlayOnRight ? 34 : 6) : "auto",
            left: overlayOnRight ? "auto" : 6,
            right: overlayOnRight ? 6 : "auto",
            maxWidth: "calc(100% - 12px)",
            px: 0,
            boxSizing: "border-box",
            color: "rgba(255,255,255,0.95)",
            pointerEvents: "none",
            textAlign: overlayOnRight ? "right" : "left",
            textShadow: "1px 1px 0 rgba(0,0,0,0.85), 0 0 3px rgba(0,0,0,0.75)",
            userSelect: "none",
          }}
        >
          <Typography sx={{ fontSize: 11, fontWeight: 700, lineHeight: 1.15, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {label}
          </Typography>
          <Typography sx={{ fontSize: 11, fontWeight: 600, lineHeight: 1.15, fontVariantNumeric: "tabular-nums" }}>
            {clampImageZoom(viewZoom).toFixed(1)}x
          </Typography>
        </Box>
      )}
      {loading && overlayTextVisible && (
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
  deferFft,
  fftWindow,
  preferWebgpu,
  fftLayout,
  contrastPreset,
  contrastRange,
  selected,
  label,
  scaleBarVisible,
  overlayPosition,
  pixelSize,
  pixelUnit,
  imageViewZoom,
  imageViewCenter,
  fftViewZoom,
  fftViewCenter,
  profileActive,
  profileLine,
  fftCacheRef,
  fftPendingRef,
  fftGpuRef,
  fftCacheVersion,
  onImageViewChange,
  onFftViewChange,
  onProfileLineChange,
  onFftOverlayPositionChange,
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
  deferFft: boolean;
  fftWindow: boolean;
  preferWebgpu: boolean;
  fftLayout: "overlay" | "below";
  contrastPreset: string;
  contrastRange: [number, number] | null;
  selected: boolean;
  label: string;
  scaleBarVisible: boolean;
  overlayPosition: string;
  pixelSize: number;
  pixelUnit: string;
  imageViewZoom: number;
  imageViewCenter: number[];
  fftViewZoom: number;
  fftViewCenter: number[];
  profileActive: boolean;
  profileLine: ProfilePoint[];
  fftCacheRef: SnapshotFftCacheRef;
  fftPendingRef: SnapshotFftPendingRef;
  fftGpuRef: React.MutableRefObject<WebGPUFFT | null>;
  fftCacheVersion: number;
  onImageViewChange: (view: ImageViewApiState) => void;
  onFftViewChange: (view: ImageViewApiState) => void;
  onProfileLineChange: (line: ProfilePoint[]) => void;
  onFftOverlayPositionChange: (position: SnapshotOverlayPosition) => void;
  onSelect: () => void;
}) {
  const imageContainerRef = React.useRef<HTMLDivElement | null>(null);
  const fftOverlayDragRef = React.useRef<{
    pointerId: number;
    startClientX: number;
    startClientY: number;
    startLeft: number;
    startTop: number;
    overlayW: number;
    overlayH: number;
    containerW: number;
    containerH: number;
    moved: boolean;
  } | null>(null);
  const fftOverlayDragFrameRef = React.useRef<number | null>(null);
  const fftOverlayDragPreviewRef = React.useRef<{ left: number; top: number } | null>(null);
  const [fftOverlayDragPreview, setFftOverlayDragPreview] = React.useState<{ left: number; top: number } | null>(null);
  const image = React.useMemo(
    () => extractPackedImage(data, imageIndex, packedHeight, packedWidth, imageHeight, imageWidth),
    [data, imageHeight, imageIndex, imageWidth, packedHeight, packedWidth],
  );
  const imageDisplayRange = React.useMemo(
    () => resolveSnapshotDisplayRange(image, contrastPreset, contrastRange),
    [contrastPreset, contrastRange, image],
  );
  const fftCacheKey = React.useMemo(
    () => snapshotFftCacheKey(imageIndex, imageWidth, imageHeight, fftWindow, preferWebgpu),
    [fftWindow, imageHeight, imageIndex, imageWidth, preferWebgpu],
  );
  const cachedFftEntry = React.useMemo(
    () => fftCacheRef.current.get(fftCacheKey) ?? null,
    [fftCacheKey, fftCacheRef, fftCacheVersion],
  );
  const [fftEntryState, setFftEntryState] = React.useState<{ key: string; entry: SnapshotFftCacheEntry } | null>(() => {
    const entry = fftCacheRef.current.get(fftCacheKey);
    return entry ? { key: fftCacheKey, entry } : null;
  });
  const [fftLoading, setFftLoading] = React.useState(false);
  const fftEntry = fftEntryState?.key === fftCacheKey ? fftEntryState.entry : cachedFftEntry;

  React.useEffect(() => {
    if (!showFft) {
      setFftLoading(false);
      return;
    }
    if (!image.length || imageWidth <= 0 || imageHeight <= 0) {
      setFftLoading(false);
      return;
    }
    const cached = fftCacheRef.current.get(fftCacheKey);
    if (cached) {
      setFftEntryState({ key: fftCacheKey, entry: cached });
      setFftLoading(false);
      return;
    }
    if (deferFft) {
      setFftLoading(false);
      return;
    }
    let canceled = false;
    setFftLoading(true);
    void computeSnapshotFftCached(
      fftCacheKey,
      image,
      imageWidth,
      imageHeight,
      fftWindow,
      preferWebgpu,
      fftCacheRef,
      fftPendingRef,
      fftGpuRef,
    )
      .then((entry) => {
        if (canceled) return;
        setFftEntryState({ key: fftCacheKey, entry });
      })
      .catch(() => {
        if (!canceled) {
          setFftEntryState({
            key: fftCacheKey,
            entry: { data: new Float32Array(0), width: imageWidth, height: imageHeight, backend: "error" },
          });
        }
      })
      .finally(() => {
        if (!canceled) setFftLoading(false);
      });
    return () => { canceled = true; };
  }, [
    deferFft,
    fftCacheKey,
    fftCacheRef,
    fftGpuRef,
    fftPendingRef,
    fftWindow,
    image,
    imageHeight,
    imageWidth,
    preferWebgpu,
    showFft,
  ]);

  const displayFft = fftEntry ?? {
    data: new Float32Array(0),
    width: nextPow2(Math.max(1, imageWidth)),
    height: nextPow2(Math.max(1, imageHeight)),
    backend: fftLoading ? "..." : "pending",
  };
  const fftReady = Boolean(fftEntry && fftEntry.data.length > 0 && fftEntry.width > 0 && fftEntry.height > 0);
  const fftLabel = "FFT";
  const resolvedFftOverlayPosition = normaliseSnapshotOverlayPosition(overlayPosition);
  const fftOverlayRequested = showFft && fftLayout === "overlay";
  const fftOverlayEnabled = fftOverlayRequested && fftReady;
  const fftBelowEnabled = showFft && fftLayout === "below" && fftReady;
  const fftInsetPlacement = {
    top: resolvedFftOverlayPosition.startsWith("top") ? 7 : "auto",
    bottom: resolvedFftOverlayPosition.startsWith("bottom") ? 7 : "auto",
    left: resolvedFftOverlayPosition.endsWith("left") ? 7 : "auto",
    right: resolvedFftOverlayPosition.endsWith("right") ? 7 : "auto",
  };
  const mainLabelPosition = fftOverlayRequested
    ? labelPositionAwayFromInset(resolvedFftOverlayPosition)
    : "top-right";
  const scheduleFftOverlayDragPreview = React.useCallback((preview: { left: number; top: number }) => {
    fftOverlayDragPreviewRef.current = preview;
    if (typeof window === "undefined" || typeof window.requestAnimationFrame !== "function") {
      setFftOverlayDragPreview(preview);
      return;
    }
    if (fftOverlayDragFrameRef.current !== null) return;
    fftOverlayDragFrameRef.current = window.requestAnimationFrame(() => {
      fftOverlayDragFrameRef.current = null;
      setFftOverlayDragPreview(fftOverlayDragPreviewRef.current);
    });
  }, []);
  React.useEffect(() => () => {
    if (fftOverlayDragFrameRef.current !== null && typeof window !== "undefined") {
      window.cancelAnimationFrame(fftOverlayDragFrameRef.current);
    }
  }, []);
  const handleFftOverlayPointerDown = React.useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    const containerRect = imageContainerRef.current?.getBoundingClientRect();
    const overlayRect = event.currentTarget.getBoundingClientRect();
    if (!containerRect || containerRect.width <= 0 || containerRect.height <= 0 || overlayRect.width <= 0 || overlayRect.height <= 0) return;
    event.preventDefault();
    event.stopPropagation();
    const startLeft = clampValue(overlayRect.left - containerRect.left, 0, Math.max(0, containerRect.width - overlayRect.width));
    const startTop = clampValue(overlayRect.top - containerRect.top, 0, Math.max(0, containerRect.height - overlayRect.height));
    fftOverlayDragPreviewRef.current = { left: startLeft, top: startTop };
    setFftOverlayDragPreview({ left: startLeft, top: startTop });
    fftOverlayDragRef.current = {
      pointerId: event.pointerId,
      startClientX: event.clientX,
      startClientY: event.clientY,
      startLeft,
      startTop,
      overlayW: overlayRect.width,
      overlayH: overlayRect.height,
      containerW: containerRect.width,
      containerH: containerRect.height,
      moved: false,
    };
    try {
      event.currentTarget.setPointerCapture(event.pointerId);
    } catch {
      // Pointer capture is best-effort; global state still keeps the drag bounded.
    }
  }, []);
  const handleFftOverlayPointerMove = React.useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    const drag = fftOverlayDragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    event.preventDefault();
    event.stopPropagation();
    if (Math.hypot(event.clientX - drag.startClientX, event.clientY - drag.startClientY) > 4) {
      drag.moved = true;
    }
    if (!drag.moved) return;
    scheduleFftOverlayDragPreview({
      left: clampValue(drag.startLeft + event.clientX - drag.startClientX, 0, Math.max(0, drag.containerW - drag.overlayW)),
      top: clampValue(drag.startTop + event.clientY - drag.startClientY, 0, Math.max(0, drag.containerH - drag.overlayH)),
    });
  }, [scheduleFftOverlayDragPreview]);
  const finishFftOverlayDrag = React.useCallback((event: React.PointerEvent<HTMLDivElement>, commit: boolean) => {
    const drag = fftOverlayDragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    event.preventDefault();
    event.stopPropagation();
    fftOverlayDragRef.current = null;
    if (fftOverlayDragFrameRef.current !== null && typeof window !== "undefined") {
      window.cancelAnimationFrame(fftOverlayDragFrameRef.current);
      fftOverlayDragFrameRef.current = null;
    }
    const preview = fftOverlayDragPreviewRef.current;
    fftOverlayDragPreviewRef.current = null;
    setFftOverlayDragPreview(null);
    try {
      event.currentTarget.releasePointerCapture(event.pointerId);
    } catch {
      // Ignore stale pointer captures from browser-level pointer cancellation.
    }
    if (!commit || !drag.moved || !preview) return;
    const centerX = preview.left + drag.overlayW / 2;
    const centerY = preview.top + drag.overlayH / 2;
    const vertical = centerY < drag.containerH / 2 ? "top" : "bottom";
    const horizontal = centerX < drag.containerW / 2 ? "left" : "right";
    onFftOverlayPositionChange(`${vertical}-${horizontal}` as SnapshotOverlayPosition);
  }, [onFftOverlayPositionChange]);
  const fftInsetDragPlacement = fftOverlayDragPreview
    ? { top: fftOverlayDragPreview.top, left: fftOverlayDragPreview.left, right: "auto", bottom: "auto" }
    : fftInsetPlacement;

  return (
    <Box sx={{ display: "flex", flexDirection: "column", width: "100%", minWidth: 0, bgcolor: colors.bgAlt }}>
      <Box ref={imageContainerRef} sx={{ position: "relative", aspectRatio: `${Math.max(1, imageWidth)} / ${Math.max(1, imageHeight)}`, minHeight: 0 }}>
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
          overlayPosition={mainLabelPosition}
          pixelSize={pixelSize}
          pixelUnit={pixelUnit}
          ariaLabel={`${label} snapshot image`}
          viewZoom={imageViewZoom}
          viewCenter={imageViewCenter}
          profileActive={profileActive}
          profileLine={profileLine}
          imageBorderVisible={false}
          onViewChange={onImageViewChange}
          onProfileLineChange={onProfileLineChange}
          onSelect={onSelect}
        />
        {fftOverlayEnabled && (
          <Box
            data-testid={`show1d-snapshot-fft-overlay-${imageIndex}`}
            sx={{
              position: "absolute",
              ...fftInsetDragPlacement,
              width: "38%",
              aspectRatio: `${Math.max(1, displayFft.width)} / ${Math.max(1, displayFft.height)}`,
              minWidth: 42,
              maxWidth: "58%",
              zIndex: 6,
              overflow: "hidden",
              bgcolor: "transparent",
              border: "none",
              outline: "none",
              boxShadow: "none",
              WebkitMaskImage: "radial-gradient(ellipse at center, #000 0%, #000 64%, rgba(0,0,0,0.72) 78%, transparent 100%)",
              maskImage: "radial-gradient(ellipse at center, #000 0%, #000 64%, rgba(0,0,0,0.72) 78%, transparent 100%)",
              pointerEvents: "auto",
              touchAction: "none",
              cursor: fftOverlayDragPreview ? "grabbing" : "grab",
              willChange: fftOverlayDragPreview ? "top, left" : "auto",
            }}
            onPointerDownCapture={handleFftOverlayPointerDown}
            onPointerMoveCapture={handleFftOverlayPointerMove}
            onPointerUpCapture={(event) => finishFftOverlayDrag(event, true)}
            onPointerCancelCapture={(event) => finishFftOverlayDrag(event, false)}
            aria-label={`${label} FFT inset drag handle`}
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
              scaleBarVisible={false}
              overlayPosition="top-right"
              pixelSize={1}
              pixelUnit="px"
              loading={fftLoading}
              overlayTextVisible={false}
              imageBorderVisible={false}
              ariaLabel={`${label} FFT overlay`}
              viewZoom={fftViewZoom}
              viewCenter={fftViewCenter}
              onViewChange={onFftViewChange}
              onSelect={onSelect}
            />
          </Box>
        )}
      </Box>
      {fftBelowEnabled && (
        <Box
          sx={{
            position: "relative",
            aspectRatio: `${Math.max(1, displayFft.width)} / ${Math.max(1, displayFft.height)}`,
            minHeight: 0,
            borderTop: "none",
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
            scaleBarVisible={false}
            overlayPosition={overlayPosition}
            pixelSize={1}
            pixelUnit="px"
            loading={fftLoading}
            overlayTextVisible={false}
            imageBorderVisible={false}
            ariaLabel={`${label} FFT`}
            viewZoom={fftViewZoom}
            viewCenter={fftViewCenter}
            onViewChange={onFftViewChange}
            onSelect={onSelect}
          />
        </Box>
      )}
    </Box>
  );
}

function SnapshotProfilePlot({
  data,
  imageIndices,
  packedHeight,
  packedWidth,
  imageHeights,
  imageWidths,
  imageLabels,
  selectedImageIndex,
  profileLine,
  height,
  pixelSize,
  pixelUnit,
  colors,
}: {
  data: Float32Array;
  imageIndices: number[];
  packedHeight: number;
  packedWidth: number;
  imageHeights: number[];
  imageWidths: number[];
  imageLabels: string[];
  selectedImageIndex: number;
  profileLine: ProfilePoint[];
  height: number;
  pixelSize: number;
  pixelUnit: string;
  colors: { bgAlt: string; border: string; text: string; textMuted: string; accent: string };
}) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const profiles = React.useMemo(() => {
    if (!isFiniteProfilePoint(profileLine[0]) || !isFiniteProfilePoint(profileLine[1])) return [];
    return imageIndices.map((imageIdx) => {
      const imageHeight = imageHeights?.[imageIdx] || packedHeight;
      const imageWidth = imageWidths?.[imageIdx] || packedWidth;
      const image = extractPackedImage(data, imageIdx, packedHeight, packedWidth, imageHeight, imageWidth);
      return {
        imageIdx,
        label: imageLabels?.[imageIdx] || `image ${imageIdx + 1}`,
        values: image.length
          ? sampleLineProfile(image, imageWidth, imageHeight, Number(profileLine[0].row), Number(profileLine[0].col), Number(profileLine[1].row), Number(profileLine[1].col))
          : new Float32Array(0),
      };
    });
  }, [data, imageHeights, imageIndices, imageLabels, imageWidths, packedHeight, packedWidth, profileLine]);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const cssW = Math.max(1, Math.round(rect.width));
    const cssH = Math.max(1, Math.round(height));
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    canvas.style.height = `${cssH}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = colors.bgAlt;
    ctx.fillRect(0, 0, cssW, cssH);

    const validProfiles = profiles.filter((profile) => profile.values.length >= 2);
    if (!validProfiles.length) return;
    const padLeft = 38;
    const padRight = 8;
    const padTop = 6;
    const padBottom = 17;
    const plotW = Math.max(1, cssW - padLeft - padRight);
    const plotH = Math.max(1, cssH - padTop - padBottom);
    let gMin = Infinity;
    let gMax = -Infinity;
    for (const profile of validProfiles) {
      for (let idx = 0; idx < profile.values.length; idx += 1) {
        const value = profile.values[idx];
        if (!Number.isFinite(value)) continue;
        gMin = Math.min(gMin, value);
        gMax = Math.max(gMax, value);
      }
    }
    if (!Number.isFinite(gMin) || !Number.isFinite(gMax)) return;
    const range = Math.max(gMax - gMin, 1e-12);
    ctx.strokeStyle = colors.border;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padLeft, padTop);
    ctx.lineTo(padLeft, padTop + plotH);
    ctx.lineTo(padLeft + plotW, padTop + plotH);
    ctx.stroke();

    for (let pIdx = 0; pIdx < validProfiles.length; pIdx += 1) {
      const profile = validProfiles[pIdx];
      const values = profile.values;
      const active = profile.imageIdx === selectedImageIndex || validProfiles.length === 1;
      ctx.strokeStyle = validProfiles.length === 1 ? colors.accent : PROFILE_COLORS[pIdx % PROFILE_COLORS.length];
      ctx.lineWidth = active ? 1.6 : 1;
      ctx.globalAlpha = active ? 1 : 0.48;
      ctx.beginPath();
      for (let idx = 0; idx < values.length; idx += 1) {
        const x = padLeft + (idx / Math.max(1, values.length - 1)) * plotW;
        const y = padTop + plotH - ((values[idx] - gMin) / range) * plotH;
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    const p0 = profileLine[0];
    const p1 = profileLine[1];
    const distPx = Math.sqrt((Number(p1.col) - Number(p0.col)) ** 2 + (Number(p1.row) - Number(p0.row)) ** 2);
    const totalDist = pixelSize > 0 ? distPx * pixelSize : distPx;
    const unit = pixelSize > 0 ? pixelUnit : "px";
    const tickY = padTop + plotH;
    const tickStep = roundToNiceValue(totalDist / Math.max(2, Math.floor(plotW / 78)));
    ctx.font = "9px system-ui, sans-serif";
    ctx.fillStyle = colors.textMuted;
    ctx.textBaseline = "top";
    ctx.strokeStyle = colors.border;
    for (let value = 0; value <= totalDist + tickStep * 0.01; value += tickStep) {
      if (value > totalDist * 1.001) break;
      const frac = totalDist > 0 ? value / totalDist : 0;
      const x = padLeft + frac * plotW;
      ctx.beginPath();
      ctx.moveTo(x, tickY);
      ctx.lineTo(x, tickY + 3);
      ctx.stroke();
      ctx.textAlign = frac < 0.05 ? "left" : frac > 0.95 ? "right" : "center";
      const label = value % 1 === 0 ? value.toFixed(0) : value.toFixed(1);
      ctx.fillText(value + tickStep > totalDist ? `${label} ${unit}` : label, x, tickY + 4);
    }
    ctx.textAlign = "right";
    ctx.textBaseline = "top";
    ctx.fillText(formatCompactValue(gMax, 3), padLeft - 4, padTop);
    ctx.textBaseline = "bottom";
    ctx.fillText(formatCompactValue(gMin, 3), padLeft - 4, padTop + plotH);

    ctx.textBaseline = "top";
    ctx.font = "9px system-ui, sans-serif";
    let legendX = cssW - 4;
    for (let pIdx = validProfiles.length - 1; pIdx >= 0; pIdx -= 1) {
      const profile = validProfiles[pIdx];
      const label = compactScienceLabel(profile.label);
      const color = validProfiles.length === 1 ? colors.accent : PROFILE_COLORS[pIdx % PROFILE_COLORS.length];
      const textW = ctx.measureText(label).width;
      if (legendX - textW < padLeft + 20) break;
      ctx.globalAlpha = profile.imageIdx === selectedImageIndex ? 1 : 0.55;
      ctx.fillStyle = color;
      ctx.fillRect(legendX - textW - 10, 3, 6, 6);
      ctx.fillStyle = colors.textMuted;
      ctx.fillText(label, legendX, 1);
      legendX -= textW + 16;
    }
    ctx.globalAlpha = 1;
  }, [colors, height, pixelSize, pixelUnit, profileLine, profiles, selectedImageIndex]);

  return (
    <canvas
      ref={canvasRef}
      data-testid="show1d-snapshot-profile-plot"
      style={{
        width: "100%",
        height,
        display: "block",
        border: `1px solid ${colors.border}`,
        borderTop: "none",
        boxSizing: "border-box",
      }}
    />
  );
}

function MiniHistogram({
  bins,
  dataMin,
  dataMax,
  clipMin,
  clipMax,
  colors,
  onClipRangeChange,
  width = 360,
  height = 52,
}: {
  bins: number[];
  dataMin: number;
  dataMax: number;
  clipMin: number;
  clipMax: number;
  colors: { bgAlt: string; border: string; textMuted: string; accent: string };
  onClipRangeChange?: (range: [number, number]) => void;
  width?: number;
  height?: number;
}) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const dragRef = React.useRef<{
    mode: "left" | "right" | "middle";
    pointerId: number;
    startClientX: number;
    startMin: number;
    startMax: number;
  } | null>(null);
  const pendingClipRef = React.useRef<[number, number] | null>(null);
  const clipRafRef = React.useRef<number | null>(null);
  const dataSpan = Math.max(dataMax - dataMin, 1e-12);
  const minClipSpan = Math.max(dataSpan * 0.002, 1e-12);

  const queueClipRange = React.useCallback((range: [number, number]) => {
    if (!onClipRangeChange) return;
    pendingClipRef.current = range;
    if (clipRafRef.current !== null) return;
    clipRafRef.current = window.requestAnimationFrame(() => {
      clipRafRef.current = null;
      const next = pendingClipRef.current;
      pendingClipRef.current = null;
      if (next) onClipRangeChange(next);
    });
  }, [onClipRangeChange]);

  React.useEffect(() => () => {
    if (clipRafRef.current !== null) window.cancelAnimationFrame(clipRafRef.current);
  }, []);

  const xToValue = React.useCallback((x: number, width: number) => (
    dataMin + clampValue(x / Math.max(width, 1), 0, 1) * dataSpan
  ), [dataMin, dataSpan]);

  const valueToX = React.useCallback((value: number, width: number) => (
    clampValue(((value - dataMin) / dataSpan) * width, 0, width)
  ), [dataMin, dataSpan]);

  const clampClipRange = React.useCallback((lo: number, hi: number): [number, number] => {
    let nextMin = clampValue(lo, dataMin, dataMax - minClipSpan);
    let nextMax = clampValue(hi, nextMin + minClipSpan, dataMax);
    if (nextMax - nextMin < minClipSpan) {
      nextMax = clampValue(nextMin + minClipSpan, dataMin + minClipSpan, dataMax);
      nextMin = clampValue(nextMin, dataMin, nextMax - minClipSpan);
    }
    return [nextMin, nextMax];
  }, [dataMax, dataMin, minClipSpan]);

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
    const x0 = valueToX(clipMin, width);
    const x1 = valueToX(clipMax, width);
    ctx.fillStyle = colors.accent;
    ctx.globalAlpha = 0.08;
    ctx.fillRect(Math.min(x0, x1), 0, Math.max(1, Math.abs(x1 - x0)), height);
    ctx.globalAlpha = 1;
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
      ctx.strokeStyle = colors.accent;
      ctx.globalAlpha = 0.9;
      ctx.beginPath();
      ctx.moveTo(x0, 0);
      ctx.lineTo(x0, height);
      ctx.moveTo(x1, 0);
      ctx.lineTo(x1, height);
      ctx.stroke();
      ctx.globalAlpha = 1;
      const knobY = Math.round(height * 0.5);
      ctx.fillStyle = colors.accent;
      ctx.strokeStyle = colors.bgAlt;
      for (const handleX of [x0, x1]) {
        ctx.beginPath();
        ctx.arc(handleX, knobY, 5.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }
    ctx.strokeStyle = colors.border;
    ctx.strokeRect(0.5, 0.5, width - 1, height - 1);
  }, [bins, clipMax, clipMin, colors, dataMax, dataMin, valueToX]);

  const handlePointerDown = React.useCallback((event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!onClipRangeChange || dataMax <= dataMin) return;
    event.preventDefault();
    event.stopPropagation();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const x0 = valueToX(clipMin, rect.width);
    const x1 = valueToX(clipMax, rect.width);
    const handleHit = 11;
    let mode: "left" | "right" | "middle";
    const leftDistance = Math.abs(x - x0);
    const rightDistance = Math.abs(x - x1);
    if (leftDistance <= handleHit || rightDistance <= handleHit) {
      mode = leftDistance <= rightDistance ? "left" : "right";
    } else if (x > Math.min(x0, x1) && x < Math.max(x0, x1)) {
      mode = "middle";
    } else {
      mode = leftDistance <= rightDistance ? "left" : "right";
    }
    dragRef.current = {
      mode,
      pointerId: event.pointerId,
      startClientX: event.clientX,
      startMin: clipMin,
      startMax: clipMax,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
    if (mode !== "middle") {
      const value = xToValue(x, rect.width);
      queueClipRange(mode === "left" ? clampClipRange(value, clipMax) : clampClipRange(clipMin, value));
    }
  }, [clampClipRange, clipMax, clipMin, dataMax, dataMin, onClipRangeChange, queueClipRange, valueToX, xToValue]);

  const handlePointerMove = React.useCallback((event: React.PointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    event.preventDefault();
    event.stopPropagation();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dxValue = ((event.clientX - drag.startClientX) / Math.max(rect.width, 1)) * dataSpan;
    if (drag.mode === "middle") {
      const width = Math.min(drag.startMax - drag.startMin, dataSpan);
      const nextMin = clampValue(drag.startMin + dxValue, dataMin, dataMax - width);
      queueClipRange([nextMin, nextMin + width]);
    } else if (drag.mode === "left") {
      queueClipRange(clampClipRange(drag.startMin + dxValue, drag.startMax));
    } else {
      queueClipRange(clampClipRange(drag.startMin, drag.startMax + dxValue));
    }
  }, [clampClipRange, dataMax, dataMin, dataSpan, queueClipRange]);

  const handlePointerUp = React.useCallback((event: React.PointerEvent<HTMLCanvasElement>) => {
    if (dragRef.current?.pointerId !== event.pointerId) return;
    dragRef.current = null;
    event.currentTarget.releasePointerCapture(event.pointerId);
  }, []);

  return (
    <Box sx={{ width, maxWidth: "100%" }}>
      <Box sx={{ height, bgcolor: colors.bgAlt, border: `1px solid ${colors.border}`, borderRadius: 1, overflow: "hidden", position: "relative" }}>
        <Typography
          data-testid="show1d-histogram-range-label"
          sx={{
            position: "absolute",
            top: 1,
            right: 4,
            zIndex: 1,
            fontSize: 10,
            lineHeight: 1,
            color: colors.textMuted,
            fontVariantNumeric: "tabular-nums",
            bgcolor: "rgba(255,255,255,0.72)",
            px: 0.25,
            pointerEvents: "none",
            userSelect: "none",
          }}
        >
          {formatRangeValue(clipMin)} - {formatRangeValue(clipMax)}
        </Typography>
        <canvas
          ref={canvasRef}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
          aria-label="Snapshot contrast range histogram"
          title="Drag endpoints to change contrast; drag between them to move the range"
          style={{ width: "100%", height: "100%", display: "block", cursor: onClipRangeChange ? "ew-resize" : "default", touchAction: "none" }}
        />
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
  const [showGrid] = useModelState<boolean>("show_grid");
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
  const [hiddenSnapshotImageLabels, setHiddenSnapshotImageLabels] = useModelState<string[]>("hidden_snapshot_image_labels");
  const [trialSortKey] = useModelState<string>("trial_sort_key");
  const [trialSortDescending] = useModelState<boolean>("trial_sort_descending");
  const [trialRankings] = useModelState<TrialRanking[]>("trial_rankings");
  const [snapshotGroupIndices] = useModelState<number[]>("snapshot_group_indices");
  const [snapshotGroupIterations] = useModelState<number[]>("snapshot_group_iterations");
  const [snapshotGroupLabels] = useModelState<string[]>("snapshot_group_labels");
  const [nSnapshotGroups] = useModelState<number>("n_snapshot_groups");
  const [selectedSnapshotIdx, setSelectedSnapshotIdx] = useModelState<number>("selected_snapshot_idx");
  const [selectedSnapshotGroupIdx, setSelectedSnapshotGroupIdx] = useModelState<number>("selected_snapshot_group_idx");
  const [bookmarkedSnapshotGroups, setBookmarkedSnapshotGroups] = useModelState<number[]>("bookmarked_snapshot_groups");
  const [showSnapshots] = useModelState<boolean>("show_snapshots");
  const [showSnapshotThumbnails] = useModelState<boolean>("show_snapshot_thumbnails");
  const [showSnapshotFft, setShowSnapshotFft] = useModelState<boolean>("show_snapshot_fft");
  const [snapshotFftLayout, setSnapshotFftLayout] = useModelState<string>("snapshot_fft_layout");
  const [snapshotFftWindow, setSnapshotFftWindow] = useModelState<boolean>("snapshot_fft_window");
  const [snapshotFftCmap, setSnapshotFftCmap] = useModelState<string>("snapshot_fft_cmap");
  const [snapshotContrastPreset, setSnapshotContrastPreset] = useModelState<string>("snapshot_contrast_preset");
  const [snapshotContrastRange, setSnapshotContrastRange] = useModelState<number[]>("snapshot_contrast_range");
  const [snapshotHistogramWidth] = useModelState<number>("snapshot_histogram_width");
  const [snapshotHistogramHeight] = useModelState<number>("snapshot_histogram_height");
  const [snapshotThumbnailSize, setSnapshotThumbnailSize] = useModelState<number>("snapshot_thumbnail_size");
  const [snapshotPanelWidthPx, setSnapshotPanelWidthPx] = useModelState<number>("snapshot_panel_width_px");
  const [snapshotColumns, setSnapshotColumns] = useModelState<number>("snapshot_columns");
  const [snapshotOverlayPosition, setSnapshotOverlayPosition] = useModelState<string>("snapshot_overlay_position");
  const [imageCmap, setImageCmap] = useModelState<string>("image_cmap");
  const [snapshotRealSpaceZoom, setSnapshotRealSpaceZoom] = useModelState<number>("snapshot_real_space_zoom");
  const [snapshotRealSpaceCenter, setSnapshotRealSpaceCenter] = useModelState<number[]>("snapshot_real_space_center");
  const [snapshotFftZoom, setSnapshotFftZoom] = useModelState<number>("snapshot_fft_zoom");
  const [snapshotFftCenter, setSnapshotFftCenter] = useModelState<number[]>("snapshot_fft_center");
  const [showSnapshotProfile, setShowSnapshotProfile] = useModelState<boolean>("show_snapshot_profile");
  const [snapshotProfileLine, setSnapshotProfileLine] = useModelState<ProfilePoint[]>("snapshot_profile_line");
  const [snapshotProfileHeight] = useModelState<number>("snapshot_profile_height");
  const [scaleBarVisible] = useModelState<boolean>("scale_bar_visible");
  const [pixelSize] = useModelState<number>("pixel_size");
  const [pixelUnit] = useModelState<string>("pixel_unit");
  const [preferWebgpu] = useModelState<boolean>("prefer_webgpu");
  const [snapshotPlaying, setSnapshotPlaying] = useModelState<boolean>("snapshot_playing");
  const [snapshotFps, setSnapshotFps] = useModelState<number>("snapshot_fps");
  const [snapshotLoop, setSnapshotLoop] = useModelState<boolean>("snapshot_loop");
  const [snapshotBounce, setSnapshotBounce] = useModelState<boolean>("snapshot_bounce");
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
  const [transientSnapshotGroupIdx, setTransientSnapshotGroupIdx] = React.useState<number | null>(null);
  const initialInteractiveStateRef = React.useRef<Show1DInitialInteractiveState | null>(null);
  const [resetBaselineReady, setResetBaselineReady] = React.useState(false);

  const { colors: themeColors } = useTheme(Boolean(offlineForTheme));
  const snapshotPanelResizeGripSx = React.useMemo(() => ({
    position: "absolute",
    right: 0,
    bottom: 0,
    width: 18,
    height: 18,
    cursor: "nwse-resize",
    opacity: 0.8,
    pointerEvents: "auto",
    touchAction: "none",
    background: "transparent",
    clipPath: "polygon(100% 0, 100% 100%, 0 100%)",
    zIndex: 6,
    "&::before": {
      content: '""',
      position: "absolute",
      right: 2,
      bottom: 3,
      width: 16,
      height: 2,
      borderRadius: 999,
      backgroundColor: themeColors.accent,
      transform: "rotate(-45deg)",
      transformOrigin: "right bottom",
      boxShadow: `0 -5px 0 ${themeColors.accent}, 0 -10px 0 ${themeColors.accent}`,
      filter: "drop-shadow(0 0 1px rgba(0,0,0,0.85))",
      pointerEvents: "none",
    },
    "&:hover, &:focus-visible": { opacity: 1, outline: "none" },
  }), [themeColors.accent]);
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
  const hiddenTrialKeys = React.useMemo(() => new Set(hiddenTrialLabels.map(trialKey)), [hiddenTrialLabels]);
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
        hidden: hiddenTrialKeys.has(key),
      });
    }
    return Array.from(rowsByKey.values()).filter((row) => Number.isFinite(Number(row.trace_index)));
  }, [hiddenTrialKeys, labels, nPoints, nTraces, trialRankings, yData]);
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
  const imageLabelForIndex = React.useCallback(
    (imageIdx: number) => snapshotImageLabels?.[imageIdx] || snapshotLabels?.[imageIdx] || `image ${imageIdx + 1}`,
    [snapshotImageLabels, snapshotLabels],
  );
  const isSnapshotImageHidden = React.useCallback(
    (imageIdx: number) => {
      const label = imageLabelForIndex(imageIdx);
      return hiddenTrialKeys.has(trialKey(label));
    },
    [hiddenTrialKeys, imageLabelForIndex],
  );
  const hiddenTraceIndices = React.useMemo(
    () => Array.from({ length: nTraces }, (_, idx) => idx).filter((idx) => {
      const label = labels?.[idx] || `Trace ${idx + 1}`;
      return hiddenTrialKeys.has(trialKey(label));
    }),
    [hiddenTrialKeys, labels, nTraces],
  );
  const hiddenTraceSet = React.useMemo(() => new Set(hiddenTraceIndices), [hiddenTraceIndices]);
  const visibleTraceIndices = React.useMemo(
    () => sortedTrialRows
      .map((row) => Number(row.trace_index))
      .filter((idx) => Number.isInteger(idx) && idx >= 0 && idx < nTraces && !hiddenTraceSet.has(idx)),
    [hiddenTraceSet, nTraces, sortedTrialRows],
  );
  const rootRef = React.useRef<HTMLDivElement>(null);
  // Hide the saved-notebook static-image sibling while the live view is mounted.
  useHideStaticFallback(model, rootRef);
  const mainGridRef = React.useRef<HTMLDivElement>(null);
  const plotPanelRef = React.useRef<HTMLDivElement>(null);
  const plotHostRef = React.useRef<HTMLDivElement>(null);
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const profileCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const snapshotGridRef = React.useRef<HTMLDivElement>(null);
  const viewportSize = useViewportSize();
  const mainGridSize = useElementSize(mainGridRef, {
    width: Math.max(600, viewportSize.width),
    height: Math.max(360, viewportSize.height),
  });
  const plotSize = useElementSize(plotHostRef, DEFAULT_SIZE);
  const geomRef = React.useRef<PlotGeometry | null>(null);
  const plotThumbnailHitAreasRef = React.useRef<PlotThumbnailHitArea[]>([]);
  const [hover, setHover] = React.useState<HoverPoint | null>(null);
  const [hoverSnapshotGroupIdx, setHoverSnapshotGroupIdx] = React.useState<number | null>(null);
  const [hoverSnapshotImageIdx, setHoverSnapshotImageIdx] = React.useState<number | null>(null);
  const hoverSnapshotGroupRef = React.useRef<number | null>(null);
  const [sidePanelWidthUserAdjusted, setSidePanelWidthUserAdjusted] = React.useState(false);
  const plotResizePointerIdRef = React.useRef<number | null>(null);
  const plotResizeCleanupRef = React.useRef<(() => void) | null>(null);
  const plotResizeStartRef = React.useRef<{
    x: number;
    y: number;
    plotWidth: number;
    plotHeight: number;
    gridWidth: number;
    sidePanelWidth: number;
    nonPlotWidth: number;
  } | null>(null);
  const snapshotResizePointerIdRef = React.useRef<number | null>(null);
  const snapshotResizeCleanupRef = React.useRef<(() => void) | null>(null);
  const snapshotResizeStartRef = React.useRef<{
    x: number;
    y: number;
    width: number;
    columns: number;
    tileAspect: number;
  } | null>(null);
  const snapshotViewportWidthRef = React.useRef(0);
  const histogramSlotRef = React.useRef(9000 + Math.floor(Math.random() * 100000));
  const [snapshotHistogramBins, setSnapshotHistogramBins] = React.useState<number[]>(new Array(256).fill(0));
  const [snapshotHistogramRange, setSnapshotHistogramRange] = React.useState<[number, number]>([0, 1]);
  const [snapshotHistogramClipRange, setSnapshotHistogramClipRange] = React.useState<[number, number]>([0, 1]);
  const [, setSnapshotHistogramBackend] = React.useState("cpu");
  const snapshotFftCacheRef = React.useRef<Map<string, SnapshotFftCacheEntry>>(new Map());
  const snapshotFftPendingRef = React.useRef<Map<string, Promise<SnapshotFftCacheEntry>>>(new Map());
  const snapshotFftGpuRef = React.useRef<WebGPUFFT | null>(null);
  const [snapshotFftCacheVersion, setSnapshotFftCacheVersion] = React.useState(0);
  const snapshotBounceDirectionRef = React.useRef<1 | -1>(1);
  const hoverRafRef = React.useRef<number | null>(null);
  const pendingHoverRef = React.useRef<HoverPoint | null>(null);
  const pendingHoverSnapshotGroupRef = React.useRef<number | null | undefined>(undefined);
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
  const modelSelectedSnapshotGroup = selectedSnapshotGroupIdx >= 0
    ? selectedSnapshotGroupIdx
    : (snapshotImageGroups[legacySelectedSnapshot] ?? legacySelectedSnapshot);
  const activeSelectedSnapshotGroup = hoverSnapshotGroupIdx !== null
    ? hoverSnapshotGroupIdx
    : transientSnapshotGroupIdx !== null
      ? transientSnapshotGroupIdx
      : modelSelectedSnapshotGroup;
  const selectedGroup = groupCount > 0
    ? clampValue(
      Math.round(activeSelectedSnapshotGroup),
      0,
      groupCount - 1,
    )
    : -1;
  const selectedGroupLabel = selectedGroup >= 0
    ? snapshotGroupLabels?.[selectedGroup] || `Snapshot ${selectedGroup + 1}`
    : "";
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
  const normalisedBookmarkedSnapshotGroups = React.useMemo(() => {
    const groups = new Set<number>();
    for (const raw of bookmarkedSnapshotGroups ?? []) {
      const value = Math.round(Number(raw));
      if (Number.isFinite(value) && value >= 0 && value < groupCount) groups.add(value);
    }
    return Array.from(groups).sort((a, b) => a - b);
  }, [bookmarkedSnapshotGroups, groupCount]);
  const currentSnapshotGroupBookmarked = selectedGroup >= 0
    && normalisedBookmarkedSnapshotGroups.includes(selectedGroup);
  const sidePanelVisible = (showSnapshots && hasSnapshots) || hasProfileImage || showStats;
  const plotTitleVisible = false;
  const htmlSize = formatEstimatedHtmlSize((nTraces * nPoints + nSnapshots * snapshotHeight * snapshotWidth + profileImageHeight * profileImageWidth) * 4);
  const thumbnailSize = clampThumbnailSize(snapshotThumbnailSize);
  const plotHeight = Math.round(clampValue(Number.isFinite(plotHeightPx) ? plotHeightPx : DEFAULT_PLOT_HEIGHT, MIN_PLOT_HEIGHT, MAX_PLOT_HEIGHT));
  const plotHeightExplicit = Math.abs(plotHeight - DEFAULT_PLOT_HEIGHT) > 0.5;
  const snapshotOverview = hasSnapshots && selectedGroupImageIndices.length >= 6;
  const rawSidePanelWidth = Number.isFinite(sidePanelWidthPx) ? Number(sidePanelWidthPx) : 360;
  const autoSidePanelWidth = snapshotOverview && !sidePanelWidthUserAdjusted && Math.round(rawSidePanelWidth) <= 360
    ? 620
    : rawSidePanelWidth;
  const rawSnapshotPanelWidth = Number.isFinite(snapshotPanelWidthPx) ? Number(snapshotPanelWidthPx) : 0;
  const requestedSidePanelWidth = rawSnapshotPanelWidth > 0
    ? rawSnapshotPanelWidth
    : autoSidePanelWidth;
  const availableSidePanelWidth = Math.round(clampValue(
    Math.min(MAX_SIDE_PANEL_WIDTH, mainGridSize.width - MIN_PLOT_WIDTH),
    MIN_SIDE_PANEL_WIDTH,
    MAX_SIDE_PANEL_WIDTH,
  ));
  const sidePanelWidth = Math.round(clampValue(requestedSidePanelWidth, MIN_SIDE_PANEL_WIDTH, availableSidePanelWidth));
  const availableSnapshotViewportWidth = Math.round(clampValue(
    Math.min(MAX_SNAPSHOT_VIEWPORT_WIDTH, sidePanelWidth),
    MIN_SNAPSHOT_VIEWPORT_WIDTH,
    MAX_SNAPSHOT_VIEWPORT_WIDTH,
  ));
  const rawSnapshotColumnCount = Math.round(clampValue(Number.isFinite(snapshotColumns) ? snapshotColumns : 0, 0, 8));
  const snapshotColumnCount = rawSnapshotColumnCount > 0
    ? rawSnapshotColumnCount
    : autoSnapshotColumnsForCount(selectedGroupImageIndices.length);
  const selectedImageColumns = Math.max(
    1,
    Math.min(snapshotColumnCount, Math.max(1, selectedGroupImageIndices.length)),
  );
  const minimumSnapshotViewportWidth = Math.min(
    availableSnapshotViewportWidth,
    Math.max(MIN_SIDE_PANEL_WIDTH, selectedImageColumns * MIN_SNAPSHOT_TILE_WIDTH),
  );
  const resolvedSnapshotOverlayPosition = normaliseSnapshotOverlayPosition(snapshotOverlayPosition);
  const resolvedSnapshotFftLayout = normaliseSnapshotFftLayout(snapshotFftLayout);
  const viewportShellHeight = { xs: "none", md: "calc(100vh - 8px)" };
  const mainGridViewportHeight = { xs: "auto", md: controlsVisible ? "calc(100vh - 82px)" : "calc(100vh - 8px)" };
  const mainGridTemplateColumns = sidePanelVisible
    ? {
      xs: "1fr",
      md: `minmax(${MIN_PLOT_WIDTH}px, 1fr) minmax(${MIN_SIDE_PANEL_WIDTH}px, ${sidePanelWidth}px)`,
    }
    : "1fr";
  const normalisedSnapshotContrastPreset = normaliseSnapshotContrastPreset(snapshotContrastPreset);
  const customSnapshotContrastRange = React.useMemo(
    () => normaliseSnapshotContrastRange(snapshotContrastRange),
    [snapshotContrastRange],
  );
  const imageLut = React.useMemo(() => COLORMAPS[imageCmap] || COLORMAPS.cividis || COLORMAPS.gray, [imageCmap]);
  const snapshotFftLut = React.useMemo(
    () => COLORMAPS[snapshotFftCmap] || COLORMAPS.magma || imageLut,
    [imageLut, snapshotFftCmap],
  );
  const plotThumbnailColors = React.useMemo(
    () => ({
      bgAlt: themeColors.bgAlt,
      border: themeColors.border,
      accent: themeColors.accent,
      text: themeColors.text,
    }),
    [themeColors.accent, themeColors.bgAlt, themeColors.border, themeColors.text],
  );
  const plotThumbnailCache = React.useMemo(() => {
    const cache = new Map<number, PlotThumbnailCacheEntry>();
    if (!showSnapshotThumbnails || !hasSnapshots || groupCount <= 0) return cache;
    for (let groupIdx = 0; groupIdx < groupCount; groupIdx += 1) {
      const imageIndices = (snapshotGroups[groupIdx] ?? []).filter((idx) => !isSnapshotImageHidden(idx)).sort((a, b) => {
        const aLabel = imageLabelForIndex(a);
        const bLabel = imageLabelForIndex(b);
        if (isReferenceLabel(aLabel) !== isReferenceLabel(bLabel)) return isReferenceLabel(aLabel) ? -1 : 1;
        const ar = Number(trialRowByKey.get(trialKey(aLabel))?.rank ?? Number.MAX_SAFE_INTEGER);
        const br = Number(trialRowByKey.get(trialKey(bLabel))?.rank ?? Number.MAX_SAFE_INTEGER);
        return ar - br || aLabel.localeCompare(bLabel);
      });
      const firstImageIdx = imageIndices[0];
      if (firstImageIdx === undefined) continue;
      const iteration = Number.isFinite(snapshotGroupIterations?.[groupIdx])
        ? Number(snapshotGroupIterations[groupIdx])
        : snapshotIterations?.[firstImageIdx];
      if (!Number.isFinite(iteration)) continue;
      const images = imageIndices.map((idx) => {
        const imageHeight = snapshotHeights?.[idx] || snapshotHeight;
        const imageWidth = snapshotWidths?.[idx] || snapshotWidth;
        return {
          data: extractPackedImage(snapshotData, idx, snapshotHeight, snapshotWidth, imageHeight, imageWidth),
          height: imageHeight,
          width: imageWidth,
        };
      });
      const rendered = createSnapshotGroupThumbnailCanvas(
        images,
        thumbnailSize,
        imageLut,
        plotThumbnailColors,
        normalisedSnapshotContrastPreset,
      );
      if (rendered) {
        cache.set(groupIdx, {
          canvas: rendered.canvas,
          width: rendered.width,
          height: rendered.height,
          iteration: Number(iteration),
        });
      }
    }
    return cache;
  }, [
    groupCount,
    hasSnapshots,
    imageLabelForIndex,
    imageLut,
    isSnapshotImageHidden,
    normalisedSnapshotContrastPreset,
    plotThumbnailColors,
    showSnapshotThumbnails,
    snapshotData,
    snapshotGroupIterations,
    snapshotGroups,
    snapshotHeight,
    snapshotHeights,
    snapshotIterations,
    snapshotWidth,
    snapshotWidths,
    thumbnailSize,
    trialRowByKey,
  ]);
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
  const themedTopMenuProps = {
    PaperProps: themedMenuProps.PaperProps,
    sx: { zIndex: 9999 },
  };
  const snapshotControlMenuProps = {
    PaperProps: {
      sx: {
        bgcolor: themeColors.controlBg,
        color: themeColors.text,
        border: "none",
        boxShadow: "0 8px 24px rgba(15, 23, 42, 0.14)",
      },
    },
    sx: { zIndex: 9999 },
  };
  const toolbarLabelSx = { ...typography.label, fontSize: 10, color: themeColors.textMuted };
  const selectedGroupRef = React.useRef(selectedGroup);

  React.useEffect(() => {
    selectedGroupRef.current = selectedGroup;
  }, [selectedGroup]);

  const selectSnapshotGroup = React.useCallback((value: number, syncModel = true) => {
    if (groupCount <= 0) return;
    const groupIdx = clampValue(Math.round(value), 0, groupCount - 1);
    selectedGroupRef.current = groupIdx;
    if (!syncModel) {
      setTransientSnapshotGroupIdx(groupIdx);
      return;
    }
    setTransientSnapshotGroupIdx(null);
    setSelectedSnapshotGroupIdx(groupIdx);
    const imageIdx = snapshotGroups[groupIdx]?.find((idx) => !isSnapshotImageHidden(idx)) ?? -1;
    setSelectedSnapshotIdx(imageIdx);
  }, [groupCount, isSnapshotImageHidden, setSelectedSnapshotGroupIdx, setSelectedSnapshotIdx, snapshotGroups]);

  const selectSnapshotImage = React.useCallback((imageIdx: number) => {
    if (imageIdx < 0 || imageIdx >= nSnapshots) return;
    setTransientSnapshotGroupIdx(null);
    setSelectedSnapshotIdx(imageIdx);
    const groupIdx = snapshotImageGroups[imageIdx] ?? selectedGroup;
    if (groupIdx >= 0 && groupIdx < groupCount) setSelectedSnapshotGroupIdx(groupIdx);
  }, [groupCount, nSnapshots, selectedGroup, setSelectedSnapshotGroupIdx, setSelectedSnapshotIdx, snapshotImageGroups]);

  const clearPlotThumbnailPreview = React.useCallback(() => {
    hoverSnapshotGroupRef.current = null;
    setHoverSnapshotGroupIdx(null);
    if (canvasRef.current) canvasRef.current.style.cursor = "crosshair";
  }, []);

  const findPlotThumbnailHit = React.useCallback((clientX: number, clientY: number): PlotThumbnailHitArea | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const px = clientX - rect.left;
    const py = clientY - rect.top;
    const hitAreas = plotThumbnailHitAreasRef.current;
    for (let idx = hitAreas.length - 1; idx >= 0; idx -= 1) {
      const area = hitAreas[idx];
      if (px >= area.x0 && px <= area.x0 + area.width && py >= area.y0 && py <= area.y0 + area.height) {
        return area;
      }
    }
    return null;
  }, []);

  const previewPlotThumbnailAtPointer = React.useCallback((clientX: number, clientY: number) => {
    if (snapshotPlaying) return false;
    const hit = findPlotThumbnailHit(clientX, clientY);
    if (!hit) {
      if (hoverSnapshotGroupRef.current !== null) clearPlotThumbnailPreview();
      return false;
    }
    if (hoverSnapshotGroupRef.current !== hit.groupIdx) {
      hoverSnapshotGroupRef.current = hit.groupIdx;
      setHoverSnapshotGroupIdx(hit.groupIdx);
    }
    if (canvasRef.current) canvasRef.current.style.cursor = "pointer";
    return true;
  }, [clearPlotThumbnailPreview, findPlotThumbnailHit, snapshotPlaying]);

  const commitPlotThumbnailAtPointer = React.useCallback((clientX: number, clientY: number) => {
    const hit = findPlotThumbnailHit(clientX, clientY);
    if (!hit) return false;
    clearPlotThumbnailPreview();
    selectSnapshotGroup(hit.groupIdx);
    return true;
  }, [clearPlotThumbnailPreview, findPlotThumbnailHit, selectSnapshotGroup]);

  const hideTrial = React.useCallback((label: string) => {
    const clean = String(label || "").trim();
    if (!clean) return;
    const key = trialKey(clean);
    const hidden = uniqueStrings(hiddenSnapshotImageLabels ?? []);
    if (!hidden.some((value) => trialKey(value) === key)) {
      setHiddenSnapshotImageLabels([...hidden, clean]);
    }
  }, [hiddenSnapshotImageLabels, setHiddenSnapshotImageLabels]);

  const showAllTrials = React.useCallback(() => {
    setHiddenSnapshotImageLabels([]);
  }, [setHiddenSnapshotImageLabels]);

  const toggleCurrentSnapshotGroupBookmark = React.useCallback(() => {
    if (selectedGroup < 0 || selectedGroup >= groupCount) return;
    const next = new Set(normalisedBookmarkedSnapshotGroups);
    if (next.has(selectedGroup)) next.delete(selectedGroup);
    else next.add(selectedGroup);
    setBookmarkedSnapshotGroups(Array.from(next).sort((a, b) => a - b));
  }, [groupCount, normalisedBookmarkedSnapshotGroups, selectedGroup, setBookmarkedSnapshotGroups]);

  const scheduleHover = React.useCallback((value: HoverPoint | null, snapshotGroupIdx?: number | null) => {
    pendingHoverRef.current = value;
    pendingHoverSnapshotGroupRef.current = snapshotGroupIdx;
    if (hoverRafRef.current !== null) return;
    hoverRafRef.current = window.requestAnimationFrame(() => {
      hoverRafRef.current = null;
      setHover(pendingHoverRef.current);
      const nextGroup = pendingHoverSnapshotGroupRef.current;
      if (nextGroup !== undefined && hoverSnapshotGroupRef.current !== nextGroup) {
        hoverSnapshotGroupRef.current = nextGroup;
        setHoverSnapshotGroupIdx(nextGroup);
      }
    });
  }, []);

  React.useEffect(() => () => {
    if (hoverRafRef.current !== null) window.cancelAnimationFrame(hoverRafRef.current);
  }, []);

  React.useEffect(() => () => {
    plotResizeCleanupRef.current?.();
    snapshotResizeCleanupRef.current?.();
  }, []);

  const handlePlotResizePointerDown = React.useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    plotResizeCleanupRef.current?.();
    const plotRect = plotPanelRef.current?.getBoundingClientRect() ?? plotHostRef.current?.getBoundingClientRect();
    const plotHostRect = plotHostRef.current?.getBoundingClientRect();
    if (!plotRect) return;
    const pointerId = event.pointerId;
    const gridRect = mainGridRef.current?.getBoundingClientRect();
    const gridWidth = Math.max(
      plotRect.width,
      gridRect?.width ?? plotRect.width + (sidePanelVisible ? sidePanelWidth : 0),
    );
    plotResizePointerIdRef.current = pointerId;
    plotResizeStartRef.current = {
      x: event.clientX,
      y: event.clientY,
      plotWidth: plotRect.width,
      plotHeight: plotHostRect?.height ?? plotRect.height,
      gridWidth,
      sidePanelWidth,
      nonPlotWidth: Math.max(0, gridWidth - plotRect.width - (sidePanelVisible ? sidePanelWidth : 0)),
    };
    event.currentTarget.setPointerCapture(pointerId);
    const handleWindowPointerMove = (moveEvent: PointerEvent) => {
      const start = plotResizeStartRef.current;
      if (plotResizePointerIdRef.current !== pointerId || !start) return;
      moveEvent.preventDefault();
      const dx = moveEvent.clientX - start.x;
      const dy = moveEvent.clientY - start.y;
      setPlotHeightPx(Math.round(clampValue(start.plotHeight + dy, MIN_PLOT_HEIGHT, MAX_PLOT_HEIGHT)));
      if (!sidePanelVisible) return;
      const minPlotWidth = MIN_PLOT_WIDTH;
      const minSidePanelWidth = MIN_SIDE_PANEL_WIDTH;
      const maxSidePanelWidth = Math.min(MAX_SIDE_PANEL_WIDTH, start.gridWidth - start.nonPlotWidth - minPlotWidth);
      const maxPlotWidth = start.gridWidth - start.nonPlotWidth - minSidePanelWidth;
      if (maxSidePanelWidth < minSidePanelWidth || maxPlotWidth < minPlotWidth) return;
      const nextPlotWidth = clampValue(start.plotWidth + dx, minPlotWidth, maxPlotWidth);
      const nextSidePanelWidth = clampValue(
        start.gridWidth - start.nonPlotWidth - nextPlotWidth,
        minSidePanelWidth,
        maxSidePanelWidth,
      );
      setSidePanelWidthUserAdjusted(true);
      setSidePanelWidthPx(Math.round(nextSidePanelWidth));
      if (showSnapshots && hasSnapshots) {
        setSnapshotPanelWidthPx(Math.round(nextSidePanelWidth));
      }
    };
    const handleWindowPointerUp = (upEvent: PointerEvent) => {
      if (plotResizePointerIdRef.current !== pointerId) return;
      upEvent.preventDefault();
      plotResizeCleanupRef.current?.();
      plotResizeCleanupRef.current = null;
    };
    window.addEventListener("pointermove", handleWindowPointerMove, { capture: true });
    window.addEventListener("pointerup", handleWindowPointerUp, { capture: true });
    window.addEventListener("pointercancel", handleWindowPointerUp, { capture: true });
    plotResizeCleanupRef.current = () => {
      window.removeEventListener("pointermove", handleWindowPointerMove, { capture: true });
      window.removeEventListener("pointerup", handleWindowPointerUp, { capture: true });
      window.removeEventListener("pointercancel", handleWindowPointerUp, { capture: true });
      plotResizePointerIdRef.current = null;
      plotResizeStartRef.current = null;
    };
  }, [hasSnapshots, setPlotHeightPx, setSidePanelWidthPx, setSnapshotPanelWidthPx, showSnapshots, sidePanelVisible, sidePanelWidth]);

  const setSnapshotViewportWidth = React.useCallback((width: number) => {
    const maximumResizableWidth = Math.min(
      MAX_SNAPSHOT_VIEWPORT_WIDTH,
      availableSidePanelWidth,
    );
    const nextWidth = Math.round(clampValue(
      width,
      minimumSnapshotViewportWidth,
      maximumResizableWidth,
    ));
    setSnapshotPanelWidthPx(nextWidth);
    if (sidePanelVisible) {
      setSidePanelWidthUserAdjusted(true);
      setSidePanelWidthPx(Math.round(clampValue(
        nextWidth,
        MIN_SIDE_PANEL_WIDTH,
        availableSidePanelWidth,
      )));
    }
  }, [
    availableSidePanelWidth,
    minimumSnapshotViewportWidth,
    setSidePanelWidthPx,
    setSnapshotPanelWidthPx,
    sidePanelVisible,
  ]);

  const handleSnapshotGridResizePointerDown = React.useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    snapshotResizeCleanupRef.current?.();
    const pointerId = event.pointerId;
    const gridRect = snapshotGridRef.current?.getBoundingClientRect();
    const fallbackGridWidth = snapshotViewportWidthRef.current || sidePanelWidth;
    const gridWidth = Math.max(1, gridRect?.width ?? fallbackGridWidth);
    const gridHeight = Math.max(1, gridRect?.height ?? gridWidth);
    const columns = Math.max(1, selectedImageColumns);
    const rows = Math.max(1, Math.ceil(Math.max(1, selectedGroupImageIndices.length) / columns));
    const tileWidth = gridWidth / columns;
    const tileHeight = gridHeight / rows;
    snapshotResizePointerIdRef.current = pointerId;
    snapshotResizeStartRef.current = {
      x: event.clientX,
      y: event.clientY,
      width: snapshotViewportWidthRef.current || gridWidth,
      columns,
      tileAspect: tileHeight / Math.max(1, tileWidth),
    };
    event.currentTarget.setPointerCapture(pointerId);
    let resizeFrame = 0;
    let latestWidth = snapshotResizeStartRef.current.width;
    const handleWindowPointerMove = (moveEvent: PointerEvent) => {
      const start = snapshotResizeStartRef.current;
      if (snapshotResizePointerIdRef.current !== pointerId || !start) return;
      moveEvent.preventDefault();
      const deltaTileX = moveEvent.clientX - start.x;
      const deltaTileY = (moveEvent.clientY - start.y) / Math.max(1e-6, start.tileAspect);
      const deltaTile = Math.abs(deltaTileX) >= Math.abs(deltaTileY) ? deltaTileX : deltaTileY;
      latestWidth = start.width + deltaTile * start.columns;
      if (resizeFrame) return;
      resizeFrame = window.requestAnimationFrame(() => {
        resizeFrame = 0;
        setSnapshotViewportWidth(latestWidth);
      });
    };
    const handleWindowPointerUp = (upEvent: PointerEvent) => {
      if (snapshotResizePointerIdRef.current !== pointerId) return;
      upEvent.preventDefault();
      if (resizeFrame) window.cancelAnimationFrame(resizeFrame);
      resizeFrame = 0;
      setSnapshotViewportWidth(latestWidth);
      snapshotResizeCleanupRef.current?.();
      snapshotResizeCleanupRef.current = null;
    };
    window.addEventListener("pointermove", handleWindowPointerMove, { capture: true });
    window.addEventListener("pointerup", handleWindowPointerUp, { capture: true });
    window.addEventListener("pointercancel", handleWindowPointerUp, { capture: true });
    snapshotResizeCleanupRef.current = () => {
      if (resizeFrame) window.cancelAnimationFrame(resizeFrame);
      resizeFrame = 0;
      window.removeEventListener("pointermove", handleWindowPointerMove, { capture: true });
      window.removeEventListener("pointerup", handleWindowPointerUp, { capture: true });
      window.removeEventListener("pointercancel", handleWindowPointerUp, { capture: true });
      snapshotResizePointerIdRef.current = null;
      snapshotResizeStartRef.current = null;
    };
  }, [selectedGroupImageIndices.length, selectedImageColumns, setSnapshotViewportWidth, sidePanelWidth]);

  React.useEffect(() => {
    if (snapshotThumbnailSize !== thumbnailSize) setSnapshotThumbnailSize(thumbnailSize);
  }, [setSnapshotThumbnailSize, snapshotThumbnailSize, thumbnailSize]);

  React.useEffect(() => {
    const fps = clampSnapshotFps(snapshotFps);
    if (snapshotFps !== fps) setSnapshotFps(fps);
  }, [setSnapshotFps, snapshotFps]);

  React.useEffect(() => {
    if (!snapshotBounce) snapshotBounceDirectionRef.current = 1;
  }, [snapshotBounce]);

  React.useEffect(() => {
    if (snapshotColumns !== rawSnapshotColumnCount) setSnapshotColumns(rawSnapshotColumnCount);
  }, [rawSnapshotColumnCount, setSnapshotColumns, snapshotColumns]);

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
    if (initialInteractiveStateRef.current !== null) return;
    const timeout = window.setTimeout(() => {
      if (initialInteractiveStateRef.current !== null) return;
      initialInteractiveStateRef.current = {
        logScale: Boolean(logScale),
        showStats: Boolean(showStats),
        showLegend: Boolean(showLegend),
        plotHeightPx: Number.isFinite(plotHeightPx) ? plotHeightPx : DEFAULT_PLOT_HEIGHT,
        sidePanelWidthPx: Number.isFinite(sidePanelWidthPx) ? sidePanelWidthPx : 360,
        snapshotPanelWidthPx: Number.isFinite(snapshotPanelWidthPx) ? snapshotPanelWidthPx : 0,
        focusedTrace: Number.isFinite(focusedTrace) ? focusedTrace : -1,
        xRange: copyNumberArray(xRange),
        yRange: copyNumberArray(yRange),
        selectedSnapshotIdx: Number.isFinite(selectedSnapshotIdx) ? selectedSnapshotIdx : -1,
        selectedSnapshotGroupIdx: Number.isFinite(selectedSnapshotGroupIdx) ? selectedSnapshotGroupIdx : -1,
        bookmarkedSnapshotGroups: copyNumberArray(bookmarkedSnapshotGroups),
        hiddenSnapshotImageLabels: copyStringArray(hiddenSnapshotImageLabels),
        showSnapshotFft: Boolean(showSnapshotFft),
        snapshotFftLayout: String(snapshotFftLayout || "overlay"),
        snapshotFftWindow: Boolean(snapshotFftWindow),
        snapshotFftCmap: String(snapshotFftCmap || "magma"),
        snapshotContrastPreset: String(snapshotContrastPreset || "full"),
        snapshotContrastRange: copyNumberArray(snapshotContrastRange),
        snapshotThumbnailSize: Number.isFinite(snapshotThumbnailSize) ? snapshotThumbnailSize : 48,
        snapshotOverlayPosition: String(snapshotOverlayPosition || "top-right"),
        imageCmap: String(imageCmap || "cividis"),
        snapshotRealSpaceZoom: Number.isFinite(snapshotRealSpaceZoom) ? snapshotRealSpaceZoom : 1,
        snapshotRealSpaceCenter: copyNumberArray(snapshotRealSpaceCenter),
        snapshotFftZoom: Number.isFinite(snapshotFftZoom) ? snapshotFftZoom : 1,
        snapshotFftCenter: copyNumberArray(snapshotFftCenter),
        showSnapshotProfile: Boolean(showSnapshotProfile),
        snapshotProfileLine: copyProfileLine(snapshotProfileLine),
        snapshotPlaying: Boolean(snapshotPlaying),
        snapshotFps: Number.isFinite(snapshotFps) ? snapshotFps : 2,
        snapshotLoop: Boolean(snapshotLoop),
        snapshotBounce: Boolean(snapshotBounce),
      };
      setResetBaselineReady(true);
    }, 0);
    return () => window.clearTimeout(timeout);
  }, [
    focusedTrace,
    bookmarkedSnapshotGroups,
    hiddenSnapshotImageLabels,
    imageCmap,
    logScale,
    plotHeightPx,
    selectedSnapshotGroupIdx,
    selectedSnapshotIdx,
    showLegend,
    showSnapshotFft,
    showSnapshotProfile,
    showStats,
    sidePanelWidthPx,
    snapshotBounce,
    snapshotContrastPreset,
    snapshotContrastRange,
    snapshotFftCenter,
    snapshotFftCmap,
    snapshotFftLayout,
    snapshotFftWindow,
    snapshotFftZoom,
    snapshotFps,
    snapshotLoop,
    snapshotOverlayPosition,
    snapshotPanelWidthPx,
    snapshotProfileLine,
    snapshotRealSpaceCenter,
    snapshotRealSpaceZoom,
    snapshotThumbnailSize,
    snapshotPlaying,
    xRange,
    yRange,
  ]);

  React.useEffect(() => {
    hoverSnapshotGroupRef.current = hoverSnapshotGroupIdx;
  }, [hoverSnapshotGroupIdx]);

  React.useEffect(() => {
    if (
      hoverSnapshotGroupIdx === null
      || (
        hasSnapshots
        && groupCount > 0
        && hoverSnapshotGroupIdx >= 0
        && hoverSnapshotGroupIdx < groupCount
        && !snapshotPlaying
      )
    ) {
      return;
    }
    clearPlotThumbnailPreview();
  }, [
    clearPlotThumbnailPreview,
    groupCount,
    hasSnapshots,
    hoverSnapshotGroupIdx,
    snapshotPlaying,
  ]);

  React.useEffect(() => {
    if (hoverSnapshotGroupIdx === null) return;
    const handleWindowPointerMove = (event: PointerEvent) => {
      const canvas = canvasRef.current;
      if (!canvas) {
        clearPlotThumbnailPreview();
        return;
      }
      const rect = canvas.getBoundingClientRect();
      if (
        event.clientX < rect.left
        || event.clientX > rect.right
        || event.clientY < rect.top
        || event.clientY > rect.bottom
      ) {
        clearPlotThumbnailPreview();
      }
    };
    window.addEventListener("pointermove", handleWindowPointerMove, true);
    window.addEventListener("blur", clearPlotThumbnailPreview);
    return () => {
      window.removeEventListener("pointermove", handleWindowPointerMove, true);
      window.removeEventListener("blur", clearPlotThumbnailPreview);
    };
  }, [clearPlotThumbnailPreview, hoverSnapshotGroupIdx]);

  React.useEffect(() => {
    snapshotFftCacheRef.current.clear();
    snapshotFftPendingRef.current.clear();
    setSnapshotFftCacheVersion((value) => value + 1);
  }, [snapshotData, snapshotHeight, snapshotHeights, snapshotWidth, snapshotWidths]);

  React.useEffect(() => {
    if (!showSnapshotFft || !hasSnapshots || groupCount <= 0 || nSnapshots <= 0 || snapshotData.length === 0) return;
    let canceled = false;
    const anchorGroup = selectedGroup >= 0 ? selectedGroup : 0;
    const orderedGroups = Array.from({ length: groupCount }, (_, groupIdx) => groupIdx)
      .sort((a, b) => Math.abs(a - anchorGroup) - Math.abs(b - anchorGroup) || a - b);

    const warmSnapshotFftCache = async () => {
      for (const groupIdx of orderedGroups) {
        const imageIndices = snapshotGroups[groupIdx] ?? [];
        for (const imageIdx of imageIndices) {
          if (canceled) return;
          const imageHeight = snapshotHeights?.[imageIdx] || snapshotHeight;
          const imageWidth = snapshotWidths?.[imageIdx] || snapshotWidth;
          if (imageIdx < 0 || imageHeight <= 0 || imageWidth <= 0) continue;
          const cacheKey = snapshotFftCacheKey(
            imageIdx,
            imageWidth,
            imageHeight,
            Boolean(snapshotFftWindow),
            Boolean(preferWebgpu),
          );
          if (snapshotFftCacheRef.current.has(cacheKey)) continue;
          const image = extractPackedImage(snapshotData, imageIdx, snapshotHeight, snapshotWidth, imageHeight, imageWidth);
          if (!image.length) continue;
          try {
            const entry = await computeSnapshotFftCached(
              cacheKey,
              image,
              imageWidth,
              imageHeight,
              Boolean(snapshotFftWindow),
              Boolean(preferWebgpu),
              snapshotFftCacheRef,
              snapshotFftPendingRef,
              snapshotFftGpuRef,
            );
            if (!canceled && groupIdx === anchorGroup && entry.data.length > 0) {
              setSnapshotFftCacheVersion((value) => value + 1);
            }
          } catch {
            // Keep the visible image layer stable; a failed FFT can be retried on the next state change.
          }
          await new Promise<void>((resolve) => { window.setTimeout(resolve, 0); });
        }
      }
    };

    void warmSnapshotFftCache();
    return () => { canceled = true; };
  }, [
    groupCount,
    hasSnapshots,
    nSnapshots,
    preferWebgpu,
    selectedGroup,
    showSnapshotFft,
    snapshotData,
    snapshotFftWindow,
    snapshotGroups,
    snapshotHeight,
    snapshotHeights,
    snapshotWidth,
    snapshotWidths,
  ]);

  React.useEffect(() => {
    if (!snapshotPlaying) return;
    if (!hasSnapshots || groupCount <= 1) {
      setSnapshotPlaying(false);
    }
  }, [groupCount, hasSnapshots, setSnapshotPlaying, snapshotPlaying]);

  React.useEffect(() => {
    if (snapshotPlaying || transientSnapshotGroupIdx === null) return;
    if (groupCount <= 0) {
      setTransientSnapshotGroupIdx(null);
      return;
    }
    const groupIdx = clampValue(Math.round(transientSnapshotGroupIdx), 0, groupCount - 1);
    setTransientSnapshotGroupIdx(null);
    setSelectedSnapshotGroupIdx(groupIdx);
    const imageIdx = snapshotGroups[groupIdx]?.find((idx) => !isSnapshotImageHidden(idx)) ?? -1;
    setSelectedSnapshotIdx(imageIdx);
  }, [
    groupCount,
    isSnapshotImageHidden,
    setSelectedSnapshotGroupIdx,
    setSelectedSnapshotIdx,
    snapshotGroups,
    snapshotPlaying,
    transientSnapshotGroupIdx,
  ]);

  React.useEffect(() => {
    if (snapshotPlaying) return;
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
    setSnapshotHistogramRange([dataMin, dataMax]);
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
    selectedSnapshot,
    snapshotPlaying,
    snapshotData,
    snapshotHeight,
    snapshotHeights,
    snapshotWidth,
    snapshotWidths,
  ]);

  React.useEffect(() => {
    if (snapshotPlaying) return;
    if (!hasSnapshots || selectedSnapshot < 0) {
      setSnapshotHistogramClipRange([0, 1]);
      return;
    }
    const imageHeight = snapshotHeights?.[selectedSnapshot] || snapshotHeight;
    const imageWidth = snapshotWidths?.[selectedSnapshot] || snapshotWidth;
    const image = extractPackedImage(snapshotData, selectedSnapshot, snapshotHeight, snapshotWidth, imageHeight, imageWidth);
    setSnapshotHistogramClipRange(resolveSnapshotDisplayRange(image, normalisedSnapshotContrastPreset, customSnapshotContrastRange));
  }, [
    customSnapshotContrastRange,
    hasSnapshots,
    normalisedSnapshotContrastPreset,
    selectedSnapshot,
    snapshotPlaying,
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
        const currentGroup = selectedGroupRef.current >= 0 ? selectedGroupRef.current : 0;
        if (snapshotBounce) {
          let direction = snapshotBounceDirectionRef.current;
          let nextGroup = currentGroup + direction;
          if (nextGroup >= groupCount) {
            direction = -1;
            nextGroup = Math.max(0, groupCount - 2);
          } else if (nextGroup < 0) {
            if (!snapshotLoop) {
              selectSnapshotGroup(0, false);
              setSnapshotPlaying(false);
              previous = now;
              return;
            }
            direction = 1;
            nextGroup = Math.min(groupCount - 1, 1);
          }
          snapshotBounceDirectionRef.current = direction;
          selectSnapshotGroup(nextGroup, false);
        } else {
          const nextGroup = currentGroup + 1;
          if (nextGroup >= groupCount) {
            if (!snapshotLoop) {
              selectSnapshotGroup(groupCount - 1, false);
              setSnapshotPlaying(false);
              previous = now;
              return;
            }
            selectSnapshotGroup(0, false);
          } else {
            selectSnapshotGroup(nextGroup, false);
          }
        }
        previous = now;
      }
      raf = window.requestAnimationFrame(tick);
    };
    raf = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(raf);
  }, [groupCount, hasSnapshots, selectSnapshotGroup, setSnapshotPlaying, snapshotBounce, snapshotFps, snapshotLoop, snapshotPlaying]);

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
    if (!canvas) {
      plotThumbnailHitAreasRef.current = [];
      return;
    }
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(plotSize.width * dpr);
    canvas.height = Math.round(plotSize.height * dpr);
    canvas.style.width = `${plotSize.width}px`;
    canvas.style.height = `${plotSize.height}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      plotThumbnailHitAreasRef.current = [];
      return;
    }
    const nextPlotThumbnailHitAreas: PlotThumbnailHitArea[] = [];
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = themeColors.bg;
    ctx.fillRect(0, 0, plotSize.width, plotSize.height);

    const geom: PlotGeometry = {
      left: 64,
      right: 8,
      top: plotTitleVisible ? 28 : 16,
      bottom: 46,
      width: plotSize.width,
      height: plotSize.height,
      plotW: Math.max(1, plotSize.width - 72),
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

    const plotMarkers = markers ?? [];
    for (let markerIndex = 0; markerIndex < plotMarkers.length; markerIndex += 1) {
      const marker = plotMarkers[markerIndex];
      if (!Number.isFinite(marker.x)) continue;
      const x = dataToX(Number(marker.x), geom);
      if (x < geom.left || x > geom.left + geom.plotW) continue;
      const color = markerColor(marker, themeColors);
      ctx.strokeStyle = color;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(x, geom.top);
      ctx.lineTo(x, geom.top + geom.plotH);
      ctx.stroke();
      ctx.setLineDash([]);
      const markerLabel = geom.plotW < 520 && marker.mobile_label !== undefined
        ? marker.mobile_label
        : marker.label;
      if (markerLabel) {
        ctx.fillStyle = color;
        ctx.font = "10px system-ui, sans-serif";
        ctx.fillText(markerLabel, x + 4, geom.top + 12 + (markerIndex % 3) * 12);
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
        const thumb = plotThumbnailCache.get(groupIdx);
        if (!thumb) continue;
        const xCenter = dataToX(thumb.iteration, geom);
        if (xCenter < geom.left - thumbW || xCenter > geom.left + geom.plotW + thumbW) continue;
        const lane = lanes > 1 ? groupIdx % lanes : 0;
        const x0 = clampValue(xCenter - thumb.width / 2, geom.left + 2, geom.left + geom.plotW - thumb.width - 2);
        const y0 = Math.min(geom.top + geom.plotH - thumb.height - 4, geom.top + 8 + lane * (thumbnailSize + 4));
        nextPlotThumbnailHitAreas.push({ groupIdx, x0, y0, width: thumb.width, height: thumb.height });
        ctx.strokeStyle = groupIdx === selectedGroup ? themeColors.accent : themeColors.border;
        ctx.globalAlpha = groupIdx === selectedGroup ? 0.55 : 0.28;
        ctx.beginPath();
        ctx.moveTo(xCenter, geom.top);
        ctx.lineTo(xCenter, y0 + thumb.height);
        ctx.stroke();
        ctx.globalAlpha = 1;
        ctx.drawImage(thumb.canvas, x0, y0, thumb.width, thumb.height);
        ctx.strokeStyle = groupIdx === selectedGroup ? themeColors.accent : themeColors.border;
        ctx.lineWidth = groupIdx === selectedGroup ? 2 : 1;
        ctx.strokeRect(x0 + 0.5, y0 + 0.5, thumb.width - 1, thumb.height - 1);
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
    plotThumbnailHitAreasRef.current = nextPlotThumbnailHitAreas;
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
    selectedGroup,
    plotThumbnailCache,
    thumbnailSize,
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

  const snapshotGroupForPoint = React.useCallback((point: HoverPoint | null): number | null => {
    if (!point || !hasSnapshots || groupCount <= 0) return null;
    const tolerance = Math.max(1e-5, Math.abs(point.x) * 1e-6);
    let bestGroup: number | null = null;
    let bestDistance = Number.POSITIVE_INFINITY;
    for (let groupIdx = 0; groupIdx < groupCount; groupIdx += 1) {
      const iteration = Number(snapshotGroupIterations?.[groupIdx]);
      if (!Number.isFinite(iteration)) continue;
      const distance = Math.abs(iteration - point.x);
      if (distance <= tolerance && distance < bestDistance) {
        bestDistance = distance;
        bestGroup = groupIdx;
      }
    }
    return bestGroup;
  }, [groupCount, hasSnapshots, snapshotGroupIterations]);

  const handlePointerMove = React.useCallback((event: React.PointerEvent<HTMLCanvasElement>) => {
    if (previewPlotThumbnailAtPointer(event.clientX, event.clientY)) {
      scheduleHover(null);
      return;
    }
    const point = nearestPoint(event.clientX, event.clientY);
    scheduleHover(point, snapshotGroupForPoint(point));
  }, [nearestPoint, previewPlotThumbnailAtPointer, scheduleHover, snapshotGroupForPoint]);

  const handlePointerLeave = React.useCallback(() => {
    clearPlotThumbnailPreview();
    scheduleHover(null, null);
  }, [clearPlotThumbnailPreview, scheduleHover]);

  const handleClick = React.useCallback((event: React.MouseEvent<HTMLCanvasElement>) => {
    if (commitPlotThumbnailAtPointer(event.clientX, event.clientY)) {
      scheduleHover(null);
      return;
    }
    const point = nearestPoint(event.clientX, event.clientY) ?? hover;
    const snapshotGroup = snapshotGroupForPoint(point);
    if (snapshotGroup !== null) selectSnapshotGroup(snapshotGroup);
    if (point) setFocusedTrace(point.trace);
  }, [commitPlotThumbnailAtPointer, hover, nearestPoint, scheduleHover, selectSnapshotGroup, setFocusedTrace, snapshotGroupForPoint]);

  const setSnapshotRealSpaceView = React.useCallback((view: ImageViewApiState) => {
    setSnapshotRealSpaceZoom(view.zoom);
    setSnapshotRealSpaceCenter(view.center);
  }, [setSnapshotRealSpaceCenter, setSnapshotRealSpaceZoom]);

  const setSnapshotFftView = React.useCallback((view: ImageViewApiState) => {
    setSnapshotFftZoom(view.zoom);
    setSnapshotFftCenter(view.center);
  }, [setSnapshotFftCenter, setSnapshotFftZoom]);

  const resetPlotAndSnapshotViews = React.useCallback(() => {
    setXRange([]);
    setYRange([]);
    setFocusedTrace(-1);
    setSnapshotRealSpaceZoom(1);
    setSnapshotRealSpaceCenter([]);
    setSnapshotFftZoom(1);
    setSnapshotFftCenter([]);
  }, [setFocusedTrace, setSnapshotFftCenter, setSnapshotFftZoom, setSnapshotRealSpaceCenter, setSnapshotRealSpaceZoom, setXRange, setYRange]);

  const resetToInitialState = React.useCallback(() => {
    const initial = initialInteractiveStateRef.current;
    if (!initial) return;
    setLogScale(initial.logScale);
    setShowStats(initial.showStats);
    setShowLegend(initial.showLegend);
    setPlotHeightPx(initial.plotHeightPx);
    setSidePanelWidthPx(initial.sidePanelWidthPx);
    setSnapshotPanelWidthPx(initial.snapshotPanelWidthPx);
    setSidePanelWidthUserAdjusted(false);
    setXRange([...initial.xRange]);
    setYRange([...initial.yRange]);
    setFocusedTrace(initial.focusedTrace);
    setSelectedSnapshotIdx(initial.selectedSnapshotIdx);
    setSelectedSnapshotGroupIdx(initial.selectedSnapshotGroupIdx);
    setBookmarkedSnapshotGroups([...initial.bookmarkedSnapshotGroups]);
    setTransientSnapshotGroupIdx(null);
    setHoverSnapshotGroupIdx(null);
    setHoverSnapshotImageIdx(null);
    hoverSnapshotGroupRef.current = null;
    if (canvasRef.current) canvasRef.current.style.cursor = "crosshair";
    setHiddenSnapshotImageLabels([...initial.hiddenSnapshotImageLabels]);
    setShowSnapshotFft(initial.showSnapshotFft);
    setSnapshotFftLayout(initial.snapshotFftLayout);
    setSnapshotFftWindow(initial.snapshotFftWindow);
    setSnapshotFftCmap(initial.snapshotFftCmap);
    snapshotFftCacheRef.current.clear();
    snapshotFftPendingRef.current.clear();
    setSnapshotFftCacheVersion((value) => value + 1);
    setSnapshotContrastPreset(initial.snapshotContrastPreset);
    setSnapshotContrastRange([...initial.snapshotContrastRange]);
    setSnapshotThumbnailSize(initial.snapshotThumbnailSize);
    setSnapshotOverlayPosition(initial.snapshotOverlayPosition);
    setImageCmap(initial.imageCmap);
    setSnapshotRealSpaceZoom(initial.snapshotRealSpaceZoom);
    setSnapshotRealSpaceCenter([...initial.snapshotRealSpaceCenter]);
    setSnapshotFftZoom(initial.snapshotFftZoom);
    setSnapshotFftCenter([...initial.snapshotFftCenter]);
    setShowSnapshotProfile(initial.showSnapshotProfile);
    setSnapshotProfileLine(initial.snapshotProfileLine.map((point) => ({ ...point })));
    setSnapshotPlaying(initial.snapshotPlaying);
    setSnapshotFps(initial.snapshotFps);
    setSnapshotLoop(initial.snapshotLoop);
    setSnapshotBounce(initial.snapshotBounce);
    snapshotBounceDirectionRef.current = 1;
    setLocalExportStatus("");
    setViewMenuAnchor(null);
    setExportAnchor(null);
  }, [
    setFocusedTrace,
    setBookmarkedSnapshotGroups,
    setHiddenSnapshotImageLabels,
    setImageCmap,
    setLogScale,
    setPlotHeightPx,
    setSelectedSnapshotGroupIdx,
    setSelectedSnapshotIdx,
    setShowLegend,
    setShowSnapshotFft,
    setShowSnapshotProfile,
    setShowStats,
    setSidePanelWidthPx,
    setSnapshotBounce,
    setSnapshotContrastPreset,
    setSnapshotContrastRange,
    setSnapshotFftCenter,
    setSnapshotFftCmap,
    setSnapshotFftLayout,
    setSnapshotFftWindow,
    setSnapshotFftZoom,
    setSnapshotFps,
    setSnapshotLoop,
    setSnapshotOverlayPosition,
    setSnapshotPanelWidthPx,
    setSnapshotProfileLine,
    setSnapshotRealSpaceCenter,
    setSnapshotRealSpaceZoom,
    setSnapshotThumbnailSize,
    setSnapshotPlaying,
    setXRange,
    setYRange,
  ]);

  const needsReset = React.useMemo(() => {
    if (!resetBaselineReady) return false;
    const initial = initialInteractiveStateRef.current;
    if (!initial) return false;
    return (
      logScale !== initial.logScale
      || showStats !== initial.showStats
      || showLegend !== initial.showLegend
      || plotHeightPx !== initial.plotHeightPx
      || sidePanelWidthPx !== initial.sidePanelWidthPx
      || snapshotPanelWidthPx !== initial.snapshotPanelWidthPx
      || sidePanelWidthUserAdjusted
      || focusedTrace !== initial.focusedTrace
      || !numberArraysEqual(xRange, initial.xRange)
      || !numberArraysEqual(yRange, initial.yRange)
      || selectedSnapshotIdx !== initial.selectedSnapshotIdx
      || selectedSnapshotGroupIdx !== initial.selectedSnapshotGroupIdx
      || !numberArraysEqual(bookmarkedSnapshotGroups, initial.bookmarkedSnapshotGroups)
      || !stringArraysEqual(hiddenSnapshotImageLabels, initial.hiddenSnapshotImageLabels)
      || showSnapshotFft !== initial.showSnapshotFft
      || snapshotFftLayout !== initial.snapshotFftLayout
      || snapshotFftWindow !== initial.snapshotFftWindow
      || snapshotFftCmap !== initial.snapshotFftCmap
      || snapshotContrastPreset !== initial.snapshotContrastPreset
      || !numberArraysEqual(snapshotContrastRange, initial.snapshotContrastRange)
      || snapshotThumbnailSize !== initial.snapshotThumbnailSize
      || snapshotOverlayPosition !== initial.snapshotOverlayPosition
      || imageCmap !== initial.imageCmap
      || snapshotRealSpaceZoom !== initial.snapshotRealSpaceZoom
      || !numberArraysEqual(snapshotRealSpaceCenter, initial.snapshotRealSpaceCenter)
      || snapshotFftZoom !== initial.snapshotFftZoom
      || !numberArraysEqual(snapshotFftCenter, initial.snapshotFftCenter)
      || showSnapshotProfile !== initial.showSnapshotProfile
      || !profileLinesEqual(snapshotProfileLine, initial.snapshotProfileLine)
      || snapshotPlaying !== initial.snapshotPlaying
      || snapshotFps !== initial.snapshotFps
      || snapshotLoop !== initial.snapshotLoop
      || snapshotBounce !== initial.snapshotBounce
    );
  }, [
    focusedTrace,
    bookmarkedSnapshotGroups,
    hiddenSnapshotImageLabels,
    imageCmap,
    logScale,
    plotHeightPx,
    resetBaselineReady,
    selectedSnapshotGroupIdx,
    selectedSnapshotIdx,
    showLegend,
    showSnapshotFft,
    showSnapshotProfile,
    showStats,
    sidePanelWidthPx,
    sidePanelWidthUserAdjusted,
    snapshotBounce,
    snapshotContrastPreset,
    snapshotContrastRange,
    snapshotFftCenter,
    snapshotFftCmap,
    snapshotFftLayout,
    snapshotFftWindow,
    snapshotFftZoom,
    snapshotFps,
    snapshotLoop,
    snapshotOverlayPosition,
    snapshotPanelWidthPx,
    snapshotProfileLine,
    snapshotRealSpaceCenter,
    snapshotRealSpaceZoom,
    snapshotThumbnailSize,
    snapshotPlaying,
    xRange,
    yRange,
  ]);

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

  const snapshotSliderMarks = React.useMemo(() => {
    const values = groupCount > 1 && groupCount <= 24
      ? Array.from({ length: groupCount }, (_, value) => value)
      : normalisedBookmarkedSnapshotGroups;
    return values.map((value) => ({ value }));
  }, [groupCount, normalisedBookmarkedSnapshotGroups]);
  const snapshotBookmarkMarkStyles = React.useMemo(() => {
    const styles: Record<string, unknown> = {};
    for (const value of normalisedBookmarkedSnapshotGroups) {
      const markIndex = snapshotSliderMarks.findIndex((mark) => mark.value === value);
      if (markIndex < 0) continue;
      styles[`& .MuiSlider-mark[data-index="${markIndex}"]`] = {
        bgcolor: "#ffc107",
        width: 7,
        height: 7,
        zIndex: 1,
      };
    }
    return styles;
  }, [normalisedBookmarkedSnapshotGroups, snapshotSliderMarks]);
  const snapshotTimelineWidth = Math.round(clampValue(140 + Math.min(Math.max(groupCount, 1), 18) * 6, 180, 240));
  const snapshotTileWidth = Math.max(
    1,
    ...selectedGroupImageIndices.map((imageIdx) => snapshotWidths?.[imageIdx] || snapshotWidth || 1),
  );
  const selectedImageRows = Math.max(1, Math.ceil(Math.max(1, selectedGroupImageIndices.length) / selectedImageColumns));
  const snapshotTallestAspect = Math.max(
    1,
    ...selectedGroupImageIndices.map((imageIdx) => {
      const imageHeight = snapshotHeights?.[imageIdx] || snapshotHeight || 1;
      const imageWidth = snapshotWidths?.[imageIdx] || snapshotWidth || 1;
      const baseAspect = imageHeight / Math.max(1, imageWidth);
      return showSnapshotFft && resolvedSnapshotFftLayout === "below" ? baseAspect * 2 : baseAspect;
    }),
  );
  const snapshotGridHeightCap = Math.round(clampValue(viewportSize.height - (controlsVisible ? 330 : 260), 420, 920));
  const snapshotNaturalViewportWidth = Math.round(clampValue(selectedImageColumns * snapshotTileWidth, 240, availableSnapshotViewportWidth));
  const snapshotFitAllViewportWidth = Math.floor(snapshotGridHeightCap * selectedImageColumns / Math.max(1, selectedImageRows * snapshotTallestAspect));
  const snapshotFullViewMaxWidth = Math.round(clampValue(
    Math.min(availableSnapshotViewportWidth, Math.max(1, snapshotFitAllViewportWidth)),
    Math.min(120, availableSnapshotViewportWidth),
    availableSnapshotViewportWidth,
  ));
  const snapshotManualViewportWidth = rawSnapshotPanelWidth > 0
    ? Math.round(rawSnapshotPanelWidth)
    : 0;
  const snapshotViewportWidth = snapshotManualViewportWidth > 0
    ? Math.round(clampValue(
      snapshotManualViewportWidth,
      minimumSnapshotViewportWidth,
      availableSnapshotViewportWidth,
    ))
    : Math.round(clampValue(
      Math.min(snapshotNaturalViewportWidth, snapshotFitAllViewportWidth),
      Math.min(minimumSnapshotViewportWidth, snapshotFullViewMaxWidth),
      snapshotFullViewMaxWidth,
    ));
  snapshotViewportWidthRef.current = snapshotViewportWidth;
  const snapshotViewportSx = {
    width: "100%",
    maxWidth: `${snapshotViewportWidth}px`,
    minWidth: 0,
    alignSelf: "flex-start",
  };
  const snapshotTileDisplayWidth = snapshotViewportWidth / Math.max(1, selectedImageColumns);
  const snapshotTileDisplayHeight = Math.ceil(snapshotTileDisplayWidth * snapshotTallestAspect);
  const snapshotFullGridHeight = snapshotTileDisplayHeight * selectedImageRows;
  const snapshotGridHeight = Math.ceil(snapshotFullGridHeight);
  const snapshotProfilePlotHeight = Math.round(clampValue(Number.isFinite(snapshotProfileHeight) ? snapshotProfileHeight : 76, 44, 220));
  const snapshotHistogramDisplayWidth = Math.round(clampValue(Number.isFinite(snapshotHistogramWidth) ? snapshotHistogramWidth : 360, 110, Math.min(640, sidePanelWidth)));
  const snapshotHistogramDisplayHeight = Math.round(clampValue(Number.isFinite(snapshotHistogramHeight) ? snapshotHistogramHeight : 52, 36, 110));
  const snapshotPanelContentHeight = showSnapshots && hasSnapshots
    ? snapshotGridHeight
      + 58
      + (showSnapshotProfile ? snapshotProfilePlotHeight : 0)
      + (selectedSnapshot >= 0 ? snapshotHistogramDisplayHeight + 42 : 0)
      + (showStats ? 130 : 0)
    : 0;
  const plotNonCanvasHeightEstimate = showLegend && visibleTraceIndices.length > 0 ? 54 : 12;
  const effectivePlotHeight = snapshotOverview && !plotHeightExplicit
    ? Math.round(clampValue(
      Math.max(plotHeight, snapshotPanelContentHeight - plotNonCanvasHeightEstimate),
      260,
      MAX_PLOT_HEIGHT,
    ))
    : plotHeight;
  const statsPanel = showStats && visibleTraceIndices.length > 0 ? (
    <Box data-testid="show1d-stats-table" sx={{ width: "100%", mb: 0.5 }}>
      <Box
        component="table"
        sx={{
          width: "100%",
          borderCollapse: "collapse",
          tableLayout: "fixed",
          "& th, & td": {
            fontSize: 10.5,
            py: 0.35,
            textAlign: "right",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          },
          "& th:first-of-type, & td:first-of-type": { textAlign: "left" },
        }}
      >
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
                {compactScienceLabel(labels?.[idx] || `Trace ${idx + 1}`)}
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
  ) : null;

  const snapshotPanelControls = (
    <Box
      data-testid="show1d-panel-controls"
      sx={{
        ...controlRow,
        width: "fit-content",
        maxWidth: "100%",
        flex: "0 1 auto",
        flexWrap: "wrap",
        rowGap: 0.25,
        border: "none",
        bgcolor: "transparent",
        px: 0,
        py: 0,
        mb: 0,
        minHeight: 28,
      }}
    >
      <Typography sx={{ ...toolbarLabelSx, flexShrink: 0 }}>cols</Typography>
      <Select
        size="small"
        value={rawSnapshotColumnCount}
        onChange={(event) => setSnapshotColumns(Number(event.target.value))}
        sx={{ ...themedSelect, minWidth: 58, fontSize: 10 }}
        MenuProps={snapshotControlMenuProps}
        inputProps={{ "aria-label": "Snapshot image columns" }}
      >
        <MenuItem value={0}>auto</MenuItem>
        {[1, 2, 3, 4, 5, 6, 7, 8].map((value) => (
          <MenuItem key={value} value={value}>{value}</MenuItem>
        ))}
      </Select>
      <Typography sx={{ ...toolbarLabelSx, flexShrink: 0 }}>Profile</Typography>
      <Switch
        size="small"
        checked={Boolean(showSnapshotProfile)}
        onChange={(_, checked) => {
          setShowSnapshotProfile(checked);
          if (!checked) setSnapshotProfileLine([]);
        }}
        sx={{ ...switchStyles.small, flexShrink: 0 }}
        slotProps={{ input: { "aria-label": "Show snapshot line profile" } }}
      />
      <Typography sx={{ ...toolbarLabelSx, flexShrink: 0 }}>FFT:</Typography>
      <Switch
        size="small"
        checked={Boolean(showSnapshotFft)}
        onChange={(_, checked) => setShowSnapshotFft(checked)}
        sx={{ ...switchStyles.small, flexShrink: 0 }}
        slotProps={{ input: { "aria-label": "Show snapshot FFT panels" } }}
      />
      {showSnapshotFft && (
        <>
          <Select
            size="small"
            value={resolvedSnapshotOverlayPosition}
            onChange={(event) => setSnapshotOverlayPosition(String(event.target.value))}
            sx={{ ...themedSelect, minWidth: 74, fontSize: 10 }}
            MenuProps={snapshotControlMenuProps}
            inputProps={{ "aria-label": "Snapshot FFT overlay position" }}
          >
            <MenuItem value="top-left">top left</MenuItem>
            <MenuItem value="top-right">top right</MenuItem>
            <MenuItem value="bottom-left">bottom left</MenuItem>
            <MenuItem value="bottom-right">bottom right</MenuItem>
          </Select>
          <Select
            size="small"
            value={COLORMAPS[snapshotFftCmap] ? snapshotFftCmap : "magma"}
            onChange={(event) => setSnapshotFftCmap(event.target.value)}
            sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }}
            MenuProps={snapshotControlMenuProps}
            inputProps={{ "aria-label": "Snapshot FFT colormap" }}
          >
            {COLORMAP_NAMES.map((name) => (
              <MenuItem key={name} value={name}>{name}</MenuItem>
            ))}
          </Select>
          <Typography sx={{ ...toolbarLabelSx, flexShrink: 0 }}>win</Typography>
          <Switch
            size="small"
            checked={Boolean(snapshotFftWindow)}
            onChange={(_, checked) => setSnapshotFftWindow(checked)}
            sx={{ ...switchStyles.small, flexShrink: 0 }}
            slotProps={{ input: { "aria-label": "Apply Hann window before snapshot FFT" } }}
          />
        </>
      )}
    </Box>
  );
  const panelToolbarWidth = sidePanelVisible
    ? Math.round(showSnapshots && hasSnapshots ? snapshotViewportWidth : sidePanelWidth)
    : 0;
  const plotToolbarControls = (
    <Stack
      direction="row"
      alignItems="center"
      spacing={1}
      useFlexGap
      sx={{ flexWrap: "wrap", rowGap: 0.5, minHeight: 28, minWidth: 0 }}
    >
      <Stack direction="row" alignItems="center" spacing={0.5}>
        <Typography sx={toolbarLabelSx}>Log</Typography>
        <Switch
          size="small"
          checked={Boolean(logScale)}
          onChange={(_, checked) => setLogScale(checked)}
          sx={switchStyles.small}
          slotProps={{ input: { "aria-label": "Use log scale" } }}
        />
      </Stack>
      <Stack direction="row" alignItems="center" spacing={0.5}>
        <Typography sx={toolbarLabelSx}>Stats</Typography>
        <Switch
          size="small"
          checked={Boolean(showStats)}
          onChange={(_, checked) => setShowStats(checked)}
          sx={switchStyles.small}
          slotProps={{ input: { "aria-label": "Show stats panel" } }}
        />
      </Stack>
      <Stack direction="row" alignItems="center" spacing={0.5}>
        <Typography sx={toolbarLabelSx}>Legend</Typography>
        <Switch
          size="small"
          checked={Boolean(showLegend)}
          onChange={(_, checked) => setShowLegend(checked)}
          sx={switchStyles.small}
          slotProps={{ input: { "aria-label": "Show legend" } }}
        />
      </Stack>
    </Stack>
  );
  const topToolbarActions = (
    <Box sx={{ display: "flex", alignItems: "center", justifyContent: "flex-end", gap: 1, flexWrap: "wrap", flex: "0 0 auto" }}>
      {handoffEnabled && hasSnapshots && (
        <>
          <Button
            size="small"
            sx={compactButton}
            onClick={(event) => setViewMenuAnchor(event.currentTarget)}
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
            {...themedTopMenuProps}
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
                ...toolbarLabelSx,
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
        sx={{ ...compactButton, color: themeColors.accent }}
        onClick={(event) => setExportAnchor(event.currentTarget)}
        disabled={exportBusy}
        title={localExportStatus || exportStatus || "Export traces, view, or standalone HTML"}
      >
        {exportBusy ? "Exporting" : "Export"}
      </Button>
      <Menu
        anchorEl={exportAnchor}
        open={Boolean(exportAnchor)}
        onClose={() => setExportAnchor(null)}
        anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
        transformOrigin={{ vertical: "top", horizontal: "right" }}
        {...themedTopMenuProps}
      >
        <MenuItem onClick={() => void handleExportSelect("html")} disabled={!exportEnabled} sx={{ fontSize: 12 }}>
          <DownloadIcon fontSize="small" sx={{ mr: 1 }} /> HTML full float32 ({htmlSize})
        </MenuItem>
        <MenuItem onClick={() => void handleExportSelect("csv")} sx={{ fontSize: 12 }}>
          <TableChartIcon fontSize="small" sx={{ mr: 1 }} /> CSV traces
        </MenuItem>
        <MenuItem onClick={() => void handleExportSelect("png")} sx={{ fontSize: 12 }}>
          <ImageIcon fontSize="small" sx={{ mr: 1 }} /> PNG view
        </MenuItem>
      </Menu>
      {(localExportStatus || exportStatus) && (
        <Typography
          sx={{
            ...toolbarLabelSx,
            maxWidth: 120,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            color: (localExportStatus || exportStatus).startsWith("Export failed") ? "#b91c1c" : themeColors.textMuted,
          }}
          title={localExportStatus || exportStatus}
        >
          {localExportStatus || exportStatus}
        </Typography>
      )}
      <Button size="small" sx={compactButton} disabled={!needsReset} onClick={resetToInitialState}>
        Reset
      </Button>
    </Box>
  );

  return (
    <Box
      ref={rootRef}
      data-testid="show1d-root"
      sx={{
        width: "100%",
        maxWidth: "100%",
        bgcolor: themeColors.bg,
        color: themeColors.text,
        border: "none",
        borderRadius: 0,
        overflow: "hidden",
        maxHeight: viewportShellHeight,
        overscrollBehavior: "contain",
        fontFamily: "system-ui, sans-serif",
      }}
    >
      {controlsVisible && (
        <Box sx={{ px: 0, py: 0.75, bgcolor: themeColors.bg }}>
          <Stack direction="row" alignItems="center" spacing={0.75} sx={{ minWidth: 0, mb: 0.75, px: 1.25 }}>
            <ShowChartIcon sx={{ fontSize: 18, color: themeColors.accent, flexShrink: 0 }} />
            {showTitle && (
              <Typography sx={{ fontWeight: 600, fontSize: 13, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {title || "Show1D"}
              </Typography>
            )}
          </Stack>
          <Box
            data-testid="show1d-toolbar-grid"
            sx={{
              display: "grid",
              gridTemplateColumns: mainGridTemplateColumns,
              alignItems: "center",
              columnGap: 0,
              width: "100%",
              minHeight: 28,
              boxSizing: "border-box",
            }}
          >
            <Box
              data-testid="show1d-plot-toolbar"
              sx={{
                minWidth: 0,
                pl: 0.75,
                pr: sidePanelVisible ? 0.25 : 1.25,
                display: "flex",
                alignItems: "center",
                gap: 1,
              }}
            >
              {plotToolbarControls}
              {!sidePanelVisible && (
                <>
                  <Box sx={{ flex: 1 }} />
                  {topToolbarActions}
                </>
              )}
            </Box>
            {sidePanelVisible && (
              <Box
                data-testid="show1d-panel-toolbar"
                sx={{
                  width: `${panelToolbarWidth}px`,
                  maxWidth: "100%",
                  minWidth: 0,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  gap: 1,
                  flexWrap: "wrap",
                  rowGap: 0.5,
                }}
              >
                <Box sx={{ minWidth: 0, flex: "1 1 auto", display: "flex", alignItems: "center" }}>
                  {showSnapshots && hasSnapshots && snapshotPanelControls}
                </Box>
                {topToolbarActions}
              </Box>
            )}
          </Box>
        </Box>
      )}
      <Box
        ref={mainGridRef}
        data-testid="show1d-main-grid"
        sx={{
          display: "grid",
          gridTemplateColumns: mainGridTemplateColumns,
          alignItems: "start",
          minHeight: { xs: 360, md: 0 },
          height: "auto",
          maxHeight: mainGridViewportHeight,
          columnGap: 0,
          overflow: "hidden",
        }}
      >
        <Box
          ref={plotPanelRef}
          data-testid="show1d-plot-panel"
          sx={{ alignSelf: "start", minWidth: 0, pl: 0.75, pr: 0.25, py: 0.5, position: "relative", overflow: "hidden" }}
        >
          <Box ref={plotHostRef} sx={{ position: "relative", height: { xs: Math.max(260, Math.min(effectivePlotHeight, 520)), md: effectivePlotHeight }, minWidth: 0 }}>
            <canvas
              ref={canvasRef}
              onPointerMove={handlePointerMove}
              onPointerLeave={handlePointerLeave}
              onClick={handleClick}
              onDoubleClick={resetPlotAndSnapshotViews}
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
                  {compactScienceLabel(labels?.[hover.trace] || `Trace ${hover.trace + 1}`)}
                </Typography>
                <Typography sx={{ fontSize: 11, lineHeight: 1.25 }}>
                  {methodLabels?.[hover.point] ? shortMethodLabel(methodLabels[hover.point]) : axisPositionText(hover.x, xLabel, xUnit)}
                </Typography>
                <Typography sx={{ fontSize: 11, lineHeight: 1.25 }}>y {formatNumber(hover.y, 4)}</Typography>
              </Box>
            )}
          </Box>
          {showLegend && visibleTraceIndices.length > 0 && (
            <Stack
              direction="row"
              spacing={1}
              sx={{
                px: 1,
                py: 0.5,
                flexWrap: "wrap",
                justifyContent: "center",
                alignItems: "center",
                rowGap: 0.5,
                width: "100%",
                boxSizing: "border-box",
              }}
            >
              {visibleTraceIndices.map((idx) => (
                <Stack
                  key={idx}
                  direction="row"
                  alignItems="center"
                  spacing={0.5}
                  role="button"
                  tabIndex={0}
                  aria-pressed={focusedTrace === idx}
                  onClick={() => setFocusedTrace(focusedTrace === idx ? -1 : idx)}
                  onKeyDown={(event) => {
                    if (event.key !== "Enter" && event.key !== " ") return;
                    event.preventDefault();
                    setFocusedTrace(focusedTrace === idx ? -1 : idx);
                  }}
                  sx={{
                    opacity: focusedTrace < 0 || focusedTrace === idx ? 1 : 0.45,
                    cursor: "pointer",
                    outline: "none",
                    borderRadius: 0.5,
                    "&:focus-visible": { boxShadow: `0 0 0 2px ${themeColors.accent}` },
                  }}
                >
                  <Box sx={{ width: 16, height: 3, bgcolor: cssColor(colors ?? [], idx), borderRadius: 1 }} />
                  <Typography data-testid={`show1d-legend-label-${idx}`} sx={{ fontSize: 11, color: themeColors.textMuted }} title={labels?.[idx] || `Trace ${idx + 1}`}>
                    {compactScienceLabel(labels?.[idx] || `Trace ${idx + 1}`)}
                  </Typography>
                </Stack>
              ))}
            </Stack>
          )}
          <Box
            className="show1d-plot-resize-handle"
            data-testid="show1d-plot-resize"
            aria-hidden="true"
            onPointerDown={handlePlotResizePointerDown}
            sx={{
              position: "absolute",
              right: 0,
              bottom: 0,
              width: 28,
              height: 28,
              zIndex: 7,
              cursor: "nwse-resize",
              opacity: 0.68,
              pointerEvents: "auto",
              touchAction: "none",
              "&::before": {
                content: '""',
                position: "absolute",
                right: 5,
                bottom: 5,
                width: 12,
                height: 12,
                borderRight: `2px solid ${themeColors.accent}`,
                borderBottom: `2px solid ${themeColors.accent}`,
                opacity: 0.7,
              },
              "&:hover::before": { opacity: 1 },
            }}
          />
        </Box>
        {sidePanelVisible && (
          <Stack
            data-testid="show1d-side-panel"
            spacing={0.5}
            sx={{
              alignSelf: "start",
              p: 0,
              minWidth: 0,
              maxHeight: mainGridViewportHeight,
              overflowY: "auto",
              overflowX: "hidden",
              overscrollBehavior: "contain",
            }}
          >
            {hasProfileImage && (
              <Box>
                <Typography sx={{ fontSize: 12, fontWeight: 600, mb: 0.5 }}>Profile Image</Typography>
                <Box sx={{ height: 170, bgcolor: themeColors.bgAlt, border: `1px solid ${themeColors.border}`, borderRadius: 1, overflow: "hidden" }}>
                  <canvas ref={profileCanvasRef} style={{ width: "100%", height: "100%", display: "block" }} />
                </Box>
              </Box>
            )}
            {showSnapshots && hasSnapshots && (
              <Box sx={snapshotViewportSx}>
                <Typography
                  data-testid="show1d-snapshot-group-label"
                  title={selectedGroupLabel}
                  sx={{
                    px: 0.5,
                    py: 0.25,
                    minHeight: 22,
                    color: themeColors.text,
                    fontSize: 11,
                    fontWeight: 600,
                    lineHeight: 1.25,
                    overflowWrap: "anywhere",
                  }}
                >
                  {selectedGroupLabel}
                </Typography>
                <Box
                  data-testid="show1d-snapshot-grid"
                  ref={snapshotGridRef}
                  sx={{
                    display: "grid",
                    position: "relative",
                    gridTemplateColumns: `repeat(${selectedImageColumns}, minmax(0, 1fr))`,
                    gap: 0,
                    height: `${snapshotGridHeight}px`,
                    overflow: "hidden",
                    overscrollBehavior: "contain",
                    pr: 0,
                    border: "none",
                    bgcolor: themeColors.bg,
                  }}
                >
                  {selectedGroupImageIndices.map((imageIdx, gridSlot) => {
                    const imageHeight = snapshotHeights?.[imageIdx] || snapshotHeight;
                    const imageWidth = snapshotWidths?.[imageIdx] || snapshotWidth;
                    const imageLabel = imageLabelForIndex(imageIdx);
                    const selected = imageIdx === selectedSnapshot;
                    const hideDisabled = selectedGroupImageIndices.length <= 1;
                    const hideLabel = hideDisabled ? "Cannot hide the last visible panel" : `Hide ${imageLabel}`;
                    const showHideButton = hoverSnapshotImageIdx === imageIdx;
                    const isFirstGridColumn = gridSlot % Math.max(1, selectedImageColumns) === 0;
                    const isFirstGridRow = gridSlot < Math.max(1, selectedImageColumns);
                    return (
                      <Box
                        key={imageIdx}
                        data-testid={`show1d-snapshot-panel-${imageIdx}`}
                        title={imageLabel}
                        onMouseEnter={() => setHoverSnapshotImageIdx(imageIdx)}
                        onMouseOver={() => setHoverSnapshotImageIdx(imageIdx)}
                        onMouseLeave={() => setHoverSnapshotImageIdx((current) => (current === imageIdx ? null : current))}
                        onPointerEnter={() => setHoverSnapshotImageIdx(imageIdx)}
                        onPointerOver={() => setHoverSnapshotImageIdx(imageIdx)}
                        onPointerLeave={() => setHoverSnapshotImageIdx((current) => (current === imageIdx ? null : current))}
                        onFocus={() => setHoverSnapshotImageIdx(imageIdx)}
                        onBlur={(event) => {
                          if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
                            setHoverSnapshotImageIdx((current) => (current === imageIdx ? null : current));
                          }
                        }}
                        sx={{
                          minWidth: 0,
                          position: "relative",
                          bgcolor: themeColors.bg,
                          overflow: "hidden",
                          ml: isFirstGridColumn ? 0 : "-1px",
                          mt: isFirstGridRow ? 0 : "-1px",
                          width: isFirstGridColumn ? "100%" : "calc(100% + 1px)",
                          zIndex: selected ? 2 : 1,
                          outline: "none",
                          "&:hover .show1d-panel-hide-button, &:focus-within .show1d-panel-hide-button": {
                            opacity: hideDisabled ? 0.28 : 1,
                            transform: "translateY(0)",
                            pointerEvents: "auto",
                          },
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
                          deferFft={Boolean(snapshotPlaying)}
                          fftWindow={Boolean(snapshotFftWindow)}
                          preferWebgpu={Boolean(preferWebgpu)}
                          fftLayout={resolvedSnapshotFftLayout}
                          contrastPreset={normalisedSnapshotContrastPreset}
                          contrastRange={customSnapshotContrastRange}
                          selected={selected}
                          label={compactScienceLabel(imageLabel)}
                          scaleBarVisible={Boolean(scaleBarVisible)}
                          overlayPosition={resolvedSnapshotOverlayPosition}
                          pixelSize={Number.isFinite(pixelSize) && pixelSize > 0 ? pixelSize : 1}
                          pixelUnit={pixelSize > 0 ? pixelUnit : "px"}
                          imageViewZoom={snapshotRealSpaceZoom}
                          imageViewCenter={snapshotRealSpaceCenter}
                          fftViewZoom={snapshotFftZoom}
                          fftViewCenter={snapshotFftCenter}
                          profileActive={Boolean(showSnapshotProfile)}
                          profileLine={snapshotProfileLine ?? []}
                          fftCacheRef={snapshotFftCacheRef}
                          fftPendingRef={snapshotFftPendingRef}
                          fftGpuRef={snapshotFftGpuRef}
                          fftCacheVersion={snapshotFftCacheVersion}
                          onImageViewChange={setSnapshotRealSpaceView}
                          onFftViewChange={setSnapshotFftView}
                          onProfileLineChange={setSnapshotProfileLine}
                          onFftOverlayPositionChange={setSnapshotOverlayPosition}
                          onSelect={() => selectSnapshotImage(imageIdx)}
                        />
                        <IconButton
                          className="show1d-panel-hide-button"
                          data-testid={`show1d-panel-hide-${imageIdx}`}
                          size="small"
                          disabled={hideDisabled}
                          onMouseDown={(event) => event.stopPropagation()}
                          onClick={(event) => {
                            event.stopPropagation();
                            if (!hideDisabled) hideTrial(imageLabel);
                          }}
                          aria-label={hideLabel}
                          title={hideLabel}
                          sx={{
                            position: "absolute",
                            top: 5,
                            left: 5,
                            zIndex: 4,
                            width: 22,
                            height: 22,
                            p: 0,
                            opacity: showHideButton ? (hideDisabled ? 0.28 : 1) : 0,
                            transform: showHideButton ? "translateY(0)" : "translateY(-2px)",
                            transition: "opacity 120ms ease, transform 120ms ease, background-color 120ms ease, color 120ms ease",
                            color: hideDisabled ? "rgba(255,255,255,0.25)" : "rgba(255,255,255,0.75)",
                            bgcolor: "rgba(0,0,0,0.22)",
                            pointerEvents: showHideButton ? "auto" : "none",
                            "&:hover, &:focus-visible": {
                              bgcolor: "rgba(0,0,0,0.42)",
                              color: "rgba(255,255,255,0.95)",
                            },
                          }}
                        >
                          <VisibilityOffIcon sx={{ fontSize: 15 }} />
                        </IconButton>
                        {controlsVisible && (
                          <Box
                            className="show1d-snapshot-panel-resize-handle"
                            data-testid={`show1d-snapshot-panel-resize-${imageIdx}`}
                            role="separator"
                            aria-label={`Resize all snapshot panels from ${imageLabel}`}
                            aria-orientation="horizontal"
                            title="Resize all snapshot panels"
                            onPointerDown={handleSnapshotGridResizePointerDown}
                            sx={snapshotPanelResizeGripSx}
                          />
                        )}
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
                        border: "none",
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
                {showSnapshotProfile && isFiniteProfilePoint(snapshotProfileLine?.[0]) && isFiniteProfilePoint(snapshotProfileLine?.[1]) && selectedGroupImageIndices.length > 0 && (
                  <Box sx={{ width: "100%" }}>
                    <SnapshotProfilePlot
                      data={snapshotData}
                      imageIndices={selectedGroupImageIndices}
                      packedHeight={snapshotHeight}
                      packedWidth={snapshotWidth}
                      imageHeights={snapshotHeights ?? []}
                      imageWidths={snapshotWidths ?? []}
                      imageLabels={snapshotImageLabels ?? []}
                      selectedImageIndex={selectedSnapshot}
                      profileLine={snapshotProfileLine ?? []}
                      height={snapshotProfilePlotHeight}
                      pixelSize={Number.isFinite(pixelSize) && pixelSize > 0 ? pixelSize : 0}
                      pixelUnit={pixelSize > 0 ? pixelUnit : "px"}
                      colors={themeColors}
                    />
                  </Box>
                )}
                {statsPanel}
                <Box
                  data-testid="show1d-snapshot-playback-controls"
                  sx={{
                    ...controlRow,
                    width: `min(100%, ${SNAPSHOT_PLAYBACK_CONTROL_WIDTH}px)`,
                    flexWrap: { xs: "wrap", md: "nowrap" },
                    rowGap: 0.25,
                    border: `1px solid ${themeColors.border}`,
                    bgcolor: themeColors.controlBg,
                    mb: 0.5,
                  }}
                >
                  <Stack direction="row" spacing={0} sx={{ flexShrink: 0 }}>
                    <Tooltip title="Previous Snapshot">
                      <IconButton
                        size="small"
                        onClick={() => {
                          snapshotBounceDirectionRef.current = -1;
                          selectSnapshotGroup(Math.max(0, selectedGroup - 1));
                        }}
                        sx={{ color: themeColors.textMuted, p: 0.25 }}
                        aria-label="Previous snapshot"
                      >
                        <FastRewindIcon sx={{ fontSize: 18 }} />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title={snapshotPlaying ? "Pause" : "Play"}>
                      <IconButton size="small" onClick={() => setSnapshotPlaying(!snapshotPlaying)} sx={{ color: themeColors.accent, p: 0.25 }} aria-label={snapshotPlaying ? "Pause playback" : "Play"}>
                        {snapshotPlaying ? <PauseIcon sx={{ fontSize: 18 }} /> : <PlayArrowIcon sx={{ fontSize: 18 }} />}
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Next Snapshot">
                      <IconButton
                        size="small"
                        onClick={() => {
                          snapshotBounceDirectionRef.current = 1;
                          selectSnapshotGroup(Math.min(groupCount - 1, selectedGroup + 1));
                        }}
                        sx={{ color: themeColors.textMuted, p: 0.25 }}
                        aria-label="Next snapshot"
                      >
                        <FastForwardIcon sx={{ fontSize: 18 }} />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Stop">
                      <IconButton
                        size="small"
                        onClick={() => {
                          setSnapshotPlaying(false);
                          snapshotBounceDirectionRef.current = 1;
                          selectSnapshotGroup(0);
                        }}
                        sx={{ color: themeColors.textMuted, p: 0.25 }}
                        aria-label="Stop and rewind snapshots"
                      >
                        <StopIcon sx={{ fontSize: 16 }} />
                      </IconButton>
                    </Tooltip>
                  </Stack>
                  <Typography sx={{ ...typography.label, color: themeColors.textMuted, flexShrink: 0 }}>fps</Typography>
                  <Slider
                    size="small"
                    value={clampSnapshotFps(snapshotFps)}
                    min={1}
                    max={24}
                    step={1}
                    onChange={(_, value) => setSnapshotFps(clampSnapshotFps(Array.isArray(value) ? value[0] : value))}
                    sx={{ ...sliderStyles.small, width: 54, flexShrink: 0, color: themeColors.accent }}
                    aria-label="Snapshot playback frames per second"
                    valueLabelDisplay="auto"
                    valueLabelFormat={(value) => String(clampSnapshotFps(Number(value)))}
                  />
                  <Typography sx={{ ...typography.value, color: themeColors.textMuted, minWidth: 18, textAlign: "right", flexShrink: 0 }}>
                    {clampSnapshotFps(snapshotFps)}
                  </Typography>
                  <Typography sx={{ ...typography.label, color: themeColors.textMuted, flexShrink: 0 }}>loop</Typography>
                  <Switch
                    size="small"
                    checked={Boolean(snapshotLoop)}
                    onChange={() => setSnapshotLoop(!snapshotLoop)}
                    sx={{ ...switchStyles.small, flexShrink: 0 }}
                    slotProps={{ input: { "aria-label": "Toggle loop playback" } }}
                  />
                  <Typography sx={{ ...typography.label, color: themeColors.textMuted, flexShrink: 0 }}>bounce</Typography>
                  <Switch
                    size="small"
                    checked={Boolean(snapshotBounce)}
                    onChange={() => setSnapshotBounce(!snapshotBounce)}
                    sx={{ ...switchStyles.small, flexShrink: 0 }}
                    slotProps={{ input: { "aria-label": "Toggle bounce playback" } }}
                  />
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
                    sx={{
                      ...sliderStyles.small,
                      flex: { xs: "1 1 100%", md: "0 1 auto" },
                      width: { xs: "100%", md: `${snapshotTimelineWidth}px` },
                      minWidth: { xs: 0, md: 140 },
                      mx: { xs: 0.75, md: 0 },
                      color: themeColors.accent,
                      "& .MuiSlider-mark": { bgcolor: themeColors.accent, width: 4, height: 4, borderRadius: "50%", top: "50%", transform: "translate(-50%, -50%)" },
                      ...snapshotBookmarkMarkStyles,
                    }}
                    aria-label={`Current snapshot group (${Math.max(0, selectedGroup) + 1} of ${Math.max(1, groupCount)})`}
                  />
                  <IconButton
                    size="small"
                    onClick={toggleCurrentSnapshotGroupBookmark}
                    disabled={selectedGroup < 0}
                    aria-pressed={currentSnapshotGroupBookmarked}
                    aria-label={`${currentSnapshotGroupBookmarked ? "Unstar" : "Star"} snapshot group ${Math.max(0, selectedGroup) + 1}`}
                    title={`${currentSnapshotGroupBookmarked ? "Unstar" : "Star"} snapshot group ${Math.max(0, selectedGroup) + 1}`}
                    sx={{
                      color: currentSnapshotGroupBookmarked ? "#ffc107" : themeColors.textMuted,
                      p: 0.25,
                      width: 22,
                      height: 22,
                      flexShrink: 0,
                      "&:hover": { color: currentSnapshotGroupBookmarked ? "#ffc107" : themeColors.text },
                    }}
                  >
                    <Box component="span" sx={{ fontSize: 18, lineHeight: "18px" }}>
                      {currentSnapshotGroupBookmarked ? "★" : "☆"}
                    </Box>
                  </IconButton>
                </Box>
                {selectedSnapshot >= 0 && (
                  <Box sx={{ mb: 0.75, width: "fit-content", maxWidth: "100%", alignSelf: "flex-start" }}>
                    <MiniHistogram
                      bins={snapshotHistogramBins}
                      dataMin={snapshotHistogramRange[0]}
                      dataMax={snapshotHistogramRange[1]}
                      clipMin={snapshotHistogramClipRange[0]}
                      clipMax={snapshotHistogramClipRange[1]}
                      colors={themeColors}
                      onClipRangeChange={(range) => setSnapshotContrastRange([range[0], range[1]])}
                      width={snapshotHistogramDisplayWidth}
                      height={snapshotHistogramDisplayHeight}
                    />
                    <Box sx={{ ...controlRow, width: "fit-content", maxWidth: "100%", border: "none", bgcolor: "transparent", px: 0, py: 0.25, mt: 0.5 }}>
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
                            variant={active && !customSnapshotContrastRange ? "contained" : "outlined"}
                            onClick={() => {
                              setSnapshotContrastRange([]);
                              setSnapshotContrastPreset(preset.value);
                            }}
                            sx={{
                              minWidth: preset.value === "full" ? 38 : 48,
                              height: 24,
                              px: 0.75,
                              py: 0,
                              fontSize: 10,
                              lineHeight: 1,
                              textTransform: "none",
                              color: active && !customSnapshotContrastRange ? "#fff" : themeColors.text,
                              bgcolor: active && !customSnapshotContrastRange ? themeColors.accent : "transparent",
                              borderColor: active && !customSnapshotContrastRange ? themeColors.accent : themeColors.border,
                              "&:hover": {
                                bgcolor: active && !customSnapshotContrastRange ? themeColors.accent : themeColors.bgAlt,
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
            {!(showSnapshots && hasSnapshots) && statsPanel}
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
